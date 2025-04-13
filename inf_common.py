#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

import torch
from torch import Tensor

import torch_geometric

# print(torch.__config__.parallel_info())

from typing import Dict, List, Tuple, Set, Optional

from multiprocessing import Pool
import subprocess

import numpy as np

import sys, random, math

import hyperparams as HP

from collections import defaultdict
from itertools import chain

from dataclasses import dataclass

def default_defaultdict_of_list():
  return defaultdict(list)

EVENT_ADD = 0
EVENT_REM = 1
EVENT_SEL = 2

@dataclass
class VampResult:
  status: str
  instructions: int
  activations: int
  nn_warmup: int
  nn_gnn: int
  nn_bulks: int

def vampire_perfrom(prob,opts):
  to_run = " ".join(["./run_lawa_vampire.sh",HP.VAMPIRE_EXECUTABLE,opts,prob])
  # print(to_run)
  output = subprocess.getoutput(to_run)

  status = None
  instructions = 0
  activations = 0
  nn_warmup = 0
  nn_gnn = 0
  nn_bulks = 0

  for line in output.split("\n"):
    # print("  ",line)
    if line.startswith("%"):
      if line.startswith("% Activations started:"):
        activations = int(line.split()[-1])
      if line.startswith("% Instructions burned:"):
        instructions = int(line.split()[-2])
      if line.startswith("% Neural model warmup:"):
        nn_warmup = int(line.split()[-1])
      if line.startswith("% Gnn eval:"):
        nn_gnn = int(line.split()[-1])
      if line.startswith("% Bulk evals:"):
        nn_bulks = int(line.split()[-1])

      if line.startswith("% SZS status"):
        if "Satisfiable" in line or "CounterSatisfiable" in line:
          status = "sat"
        elif "Theorem" in line or "Unsatisfiable" in line or "ContradictoryAxioms" in line:
          status = "uns"

  # print(status,instructions,activations)
  return VampResult(status,instructions,activations,nn_warmup,nn_gnn,nn_bulks)

def get_clause_valuator():
  layer_list = [torch.nn.ReLU()]
  for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
    layer_list.append(torch.nn.Linear(HP.INTERAL_SIZE,HP.INTERAL_SIZE))
    layer_list.append(torch.nn.ReLU())
  layer_list.append(torch.nn.Linear(HP.INTERAL_SIZE,1,bias=False))

  return torch.nn.Sequential(*layer_list)

def get_conv():
  return torch_geometric.nn.SAGEConv(
      (HP.GNN_INTERNAL_SIZE,HP.GNN_INTERNAL_SIZE),
      HP.GNN_INTERNAL_SIZE,
      aggr=HP.GNN_SAGE_AGGREG,
      normalize=False,  # a bit like a layernorm on the output?
      root_weight=True, # like a self-loop; i.e. allow then target node to talk as well
      project=HP.GNN_SAGE_PROJECT,    # extra non-lineary before aggregating
      bias=True)        # and why not add a bias before the relu that's about to come?

class SingleEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(embedding_dim))  # Learnable vector

    def forward(self,input):
        return self.embedding  # Return the stored embedding directly

class MonsterModules(torch.nn.Module):
  # this class only stores all the necessary modules, but does no actual work

  def __init__(self):
    super().__init__()

    adding_problem_features = HP.NUM_PROBLEM_FEATURES if HP.FEED_PROBLEM_FEAUTURES_TO_GNN else 0

    self.gnn_node_init = [("sort",  torch.nn.Linear(3+adding_problem_features,HP.GNN_INTERNAL_SIZE)),
                ("symbol",  torch.nn.Linear(11+adding_problem_features,HP.GNN_INTERNAL_SIZE)),
                ("clause", torch.nn.Linear(10+adding_problem_features,HP.GNN_INTERNAL_SIZE)),
                ("term", torch.nn.Linear(10+adding_problem_features,HP.GNN_INTERNAL_SIZE)),
                ("var", torch.nn.Linear(1+adding_problem_features,HP.GNN_INTERNAL_SIZE)),] # TODO: discretize to have only a few embeddings? but non-linearly spread?

    self.gnn_clause_final = torch.nn.Sequential(
        torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
        torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE))
    self.gnn_symbol_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)
    self.gnn_sort_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)

    nested_modules = { "gnn_node_init:"+kind : embed for kind,embed in self.gnn_node_init}

    self.gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]] = []
    for lidx in range(HP.GNN_NUM_LAYERS*HP.GNN_MULTIPLIER):
      if lidx % HP.GNN_MULTIPLIER == 0:
        last_fresh_layer = lidx
      layer = []
      for i,(src,tgt) in enumerate([('symbol', 'sort'), ('sort', 'symbol'), ('symbol', 'symbol'), ('symbol', 'symbol'),
                                    ('clause', 'term'), ('term', 'clause'), ('term', 'term'), ('term', 'term'),
                                    ('clause', 'var'), ('var', 'clause'), ('var', 'sort'), ('sort', 'var'),
                                    ('term', 'var'), ('var', 'term'), ('term', 'symbol'), ('symbol', 'term')]):
        # in the last layer, no need for any other output than ["symbol","clause","sort"]
        # and vars don't need to talk to terms in the second to last layer (as vars never link to literals and only literal-terms talk to clauses)
        if (lidx != HP.GNN_NUM_LAYERS-1 or tgt in ["symbol","clause","sort"]) and (lidx != HP.GNN_NUM_LAYERS-2 or (src,tgt) != ('var', 'term')):
          # effectively copies the same convolution for HP.GNN_MULTIPLIER many times
          if lidx == last_fresh_layer:
            conv = get_conv()
          else:
            conv = nested_modules[f"gnn_layer[{last_fresh_layer}]:{src}->{tgt}:{i}"]
          nested_modules[f"gnn_layer[{lidx}]:{src}->{tgt}:{i}"] = conv
          layer.append((src,tgt,i,conv))
      self.gnn_layers.append(layer)

    self.gnn_nested_modules = torch.nn.ModuleDict(nested_modules)

    self.gage_rule_embed = torch.nn.Embedding(num_embeddings=HP.NUM_INFERENCE_RULES, embedding_dim=HP.GAGE_EMBEDDING_SIZE)
    self.gage_combine = torch.nn.Sequential(
      torch.nn.Linear(3*HP.GAGE_EMBEDDING_SIZE,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Dropout(HP.TREE_DROPOUT) if HP.TREE_DROPOUT > 0.0 else torch.nn.Identity(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
      torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE)
    )

    # TODO: the var embed is LayerNormalized, so that it "lives in the same space as the other term embeddings"
    # self.gweight_var_embed = torch.nn.Sequential(
    #    SingleEmbedding(embedding_dim=HP.GWEIGHT_EMBEDDING_SIZE),
    #    torch.nn.LayerNorm(HP.GWEIGHT_EMBEDDING_SIZE))
    # so far was unstable?
    self.gweight_var_embed = SingleEmbedding(embedding_dim=HP.GWEIGHT_EMBEDDING_SIZE)

    self.gweight_term_combine = torch.nn.Sequential(
      torch.nn.Linear(3*HP.GWEIGHT_EMBEDDING_SIZE+1,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Dropout(HP.TREE_DROPOUT) if HP.TREE_DROPOUT > 0.0 else torch.nn.Identity(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE),
      torch.nn.LayerNorm(HP.GWEIGHT_EMBEDDING_SIZE)
    )

    self.problem_embedder = torch.nn.Linear(HP.NUM_PROBLEM_FEATURES,HP.INTERAL_SIZE,bias=False)

    clause_embbeder_input_size = ((HP.NUM_CLAUSE_FEATURES if HP.USE_SIMPLE_FEATURES else 0)
                                + (HP.GAGE_EMBEDDING_SIZE if HP.USE_GAGE else 0)
                                + (HP.GWEIGHT_EMBEDDING_SIZE if HP.USE_GWEIGHT else 0))

    self.clause_embedder = torch.nn.Linear(clause_embbeder_input_size,HP.INTERAL_SIZE)
    self.clause_valuator = get_clause_valuator()

def get_initial_model():
  return MonsterModules()

class MonsterNN(torch.nn.Module):
  # good old simple feature stuff - modules
  # problem_embedder: torch.nn.Module
  # clause_embedder: torch.nn.Module
  # clause_valuator: torch.nn.Module

  # good old simple feature stuff - records
  problem_features: Tensor
  clause_simple_features: Dict[int,Tensor]
  journal: List[Tuple[int,int]]
  # proof_units: List[int] # no need to store here, just dump them to file

  # gnn "modules"
  gnn_node_init: List[Tuple[str,torch.nn.modules.linear.Linear]]
  gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]]
  # gnn_clause_final: torch.nn.Module
  # gnn_symbol_final: torch.nn.Module

  # gnn records
  init_gnn_nodes: Dict[str,Tensor]
  gnn_edges: List[Tuple[str,str,Tensor]]
  gnn_init_clause_nums: list[int]

  # gnn helper data
  gnn_nodes: Dict[str,Tensor]

  # gage modules
  # gage_rule_embed: torch.nn.Module
  # gage_combine: torch.nn.Module

  # gage records
  gage_infers: list[Tuple[int,int,list[int]]]

  # gage helper data
  gage_embed_store: Dict[int,Tensor]
  gage_cl_layers: Dict[int,int]
  gage_cur_base_layer: int
  gage_todo_layers: list[list[Tuple[int,int,list[int]]]]

  # gweight modules
  # gweight_var_embed:
  # gweight_term_combine:

  # gweight records
  gweight_terms: List[Tuple[int,int,float,list[int]]]
  gweight_clauses: List[Tuple[int,list[int]]]

  # gweight helper data
  gweight_symbol_embeds: Tensor
  gweight_term_embed_store: Dict[int,Tensor]
  gweight_term_layers: Dict[int,int]
  gweight_cur_base_layer: int
  gweight_todo_layers: list[list[Tuple[int,int,float,list[int]]]]

  gweight_clause_todo: List[Tuple[int,list[int]]]
  gweight_clause_embeds: Dict[int,Tensor]

  def __init__(self,
              problem_embedder, clause_embedder, clause_valuator,
              gnn_node_init,gnn_layers,gnn_clause_final,gnn_symbol_final,gnn_sort_final,
              gage_rule_embed, gage_combine,
              gweight_var_embed, gweight_term_combine
              ):
    super().__init__()

    self.recording = False
    self.computing = False
    self.old_computing = False

    # This is crazy, but while we don't need this for any computation, things don't jit.script witout it!
    # (maybe its necessary so that the annotations above can be digested?
    # I think it's the gnn_node_init and gnn_layers, who's types include modules)
    self.dummy = torch_geometric.nn.SAGEConv((1,1),1)
    self.dummy2 = torch.nn.Linear(1,1)

    # modules
    self.problem_embedder = problem_embedder
    self.clause_embedder = clause_embedder
    self.clause_valuator = clause_valuator

    # records
    self.problem_features = torch.zeros(0) # dummy, overwritten by set_problem_features
    self.clause_simple_features = {} # filled up gradually, only when recording
    self.journal = [] # filled up gradually, only when recording
    self.proof_units = []

    # modules-like
    self.gnn_node_init = gnn_node_init
    self.gnn_layers = gnn_layers
    self.gnn_clause_final = gnn_clause_final
    self.gnn_symbol_final = gnn_symbol_final
    self.gnn_sort_final = gnn_sort_final

    # records
    self.init_gnn_nodes = {}
    self.gnn_nodes = {}
    self.gnn_edges = []
    self.gnn_init_clause_nums = [] # will be set from Vampire later anyway

    # modules
    self.gage_rule_embed = gage_rule_embed
    self.gage_combine = gage_combine

    # records
    self.gage_infers = []

    # helpers
    self.gage_embed_store = {}
    self.gage_cl_layers = {}
    self.gage_cur_base_layer = 1
    self.gage_todo_layers = []

    # modules
    self.gweight_var_embed = gweight_var_embed
    self.gweight_term_combine = gweight_term_combine

    # records
    self.gweight_terms = []
    self.gweight_clauses = []

    # helpers
    self.gweight_symbol_embeds = torch.zeros(0) # dummy, overwritten by gnn_perform
    self.gweight_term_embed_store = {}
    self.gweight_term_layers = {}
    self.gweight_cur_base_layer = 1
    self.gweight_todo_layers = []

    self.gweight_clause_todo = []
    self.gweight_clause_embeds = {}

  @torch.jit.export
  def use_problem_features(self) -> bool:
    return HP.USE_PROBLEM_FEATURES

  @torch.jit.export
  def use_simple_features(self) -> bool:
    return HP.USE_SIMPLE_FEATURES

  @torch.jit.export
  def use_gage(self) -> bool:
    return HP.USE_GAGE

  @torch.jit.export
  def use_gweight(self) -> bool:
    return HP.USE_GWEIGHT

  @torch.jit.export
  def gage_embedding_size(self) -> int:
    return HP.GAGE_EMBEDDING_SIZE

  @torch.jit.export
  def gweight_embedding_size(self) -> int:
    return HP.GWEIGHT_EMBEDDING_SIZE

  @torch.jit.export
  def gage_stat(self) -> int:
    return self.gage_cur_base_layer

  @torch.jit.export
  def gweight_stat(self) -> int:
    return self.gweight_cur_base_layer

  @torch.jit.export
  def set_recording(self):
    self.recording = True

  @torch.jit.export
  def set_computing(self):
    self.computing = True

  @torch.jit.export
  def set_problem_features(self, features: Tensor):
    self.problem_features = features.clone()

    if self.computing:
      with torch.no_grad():
        self.clause_embedder.bias.add_(self.problem_embedder(features))

  @torch.jit.export
  def gnn_node_kind(self,what: str,features: Tensor):
    if self.recording:
      self.init_gnn_nodes[what] = features.clone()

    # the caller guarantees the tensor will still be alive when gnn_perform is called
    self.gnn_nodes[what] = features

  @torch.jit.export
  def gnn_edge_kind(self,src: str, tgt: str, src_idxs: list[int], tgt_idxs: list[int]):
    src_idxs_t = torch.tensor(src_idxs)
    tgt_idxs_t = torch.tensor(tgt_idxs)

    self.gnn_edges.append((src,tgt,torch.stack([src_idxs_t,tgt_idxs_t])))
    # also record the opposite edge
    self.gnn_edges.append((tgt,src,torch.stack([tgt_idxs_t,src_idxs_t])))

  @torch.jit.export
  def gnn_perform(self, clause_nums: list[int]) -> Tuple[Tensor,Tensor]:
    # the clause numbers in clause_nums are promised to go in the same order as the clauses in previously added via gnnNodeKind("clause",...)
    if self.recording:
      self.gnn_init_clause_nums = clause_nums

    if self.computing:
      for key,embedder in self.gnn_node_init:
        node_features = self.gnn_nodes[key]

        if HP.FEED_PROBLEM_FEAUTURES_TO_GNN:
          problem_features_expanded = self.problem_features.unsqueeze(0).expand(node_features.size(0), -1)
          node_features = torch.cat((node_features,problem_features_expanded),dim=1)

        self.gnn_nodes[key] = embedder.forward(node_features).relu()
        if HP.GNN_DROPOUT > 0.0:
          self.gnn_nodes[key] = torch.nn.functional.dropout(self.gnn_nodes[key],HP.GNN_DROPOUT,self.training)

      for layer in self.gnn_layers:
        out_dict: Dict[str, Tensor] = {}
        for src,tgt,i,conv in layer:
          out = conv.forward((self.gnn_nodes[src],self.gnn_nodes[tgt]),self.gnn_edges[i][2])
          # print(src,tgt,i)
          # print(out)
          if tgt in out_dict:
            out_dict[tgt] = out_dict[tgt] + out
          else:
            out_dict[tgt] = out

        for key, out in out_dict.items():
          self.gnn_nodes[key] = out.relu()
          if HP.GNN_DROPOUT > 0.0:
            self.gnn_nodes[key] = torch.nn.functional.dropout(self.gnn_nodes[key],HP.GNN_DROPOUT,self.training)
          out_dict = {}

      # TODO: in the future could also pool things and extract a (more refined) problem embedding to use

      initial_clause_gage = self.gnn_clause_final.forward(self.gnn_nodes["clause"])
      self.gweight_symbol_embeds = torch.cat(
        (self.gnn_symbol_final.forward(self.gnn_nodes["symbol"]),self.gnn_sort_final.forward(self.gnn_nodes["sort"])),dim=0)

      if not self.old_computing:
        return initial_clause_gage,self.gweight_symbol_embeds

      # pass on the gage-style clause embeddings to the gage part (using clause_nums)
      for i,cl_num in enumerate(clause_nums):
        self.gage_embed_store[cl_num] = initial_clause_gage[i]
        self.gage_cl_layers[cl_num] = 0

      # also initialized the variable embedding for terms
      self.gweight_term_embed_store[0] = self.gweight_var_embed.forward(torch.tensor(0.0)) # the input will be ignored

      # TODO: could drop all the gnn stuff not needed anymore (hard to do in script?)
      '''
      empty_gnn_node_init: List[Tuple[str, torch.nn.modules.linear.Linear]] = []
      self.gnn_node_init = empty_gnn_node_init
      empty_gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]] = []
      self.gnn_layers = empty_gnn_layers
      self.gnn_clause_final = self.dummy2
      self.gnn_symbol_final = None
      if not self.recording:
        self.gnn_nodes = None
        self.gnn_edges = None
      '''
    return torch.zeros(0),torch.zeros(0)

  @torch.jit.export
  def journal_record(self, tag: int, cl_num: int):
    self.journal.append((tag, cl_num))

  @torch.jit.export
  def set_proof_units_and_save_recorded(self, proof_units: list[int], filename: str):
    # does not really matter, just save them below
    # self.proof_units = proof_units

    torch.save((self.problem_features,self.clause_simple_features,self.journal,proof_units,
                self.init_gnn_nodes,self.gnn_edges,self.gnn_init_clause_nums,
                self.gage_infers,self.gweight_terms,self.gweight_clauses),filename)

  def gage_enqueue_one(self,cl_num: int, inf_rule: int, parents: list[int]):
    if parents:
      layer_idx = max(1+max(self.gage_cl_layers[p] for p in parents),self.gage_cur_base_layer)
    else:
      # for inference rules without parents (such as "function definition" introduction)
      layer_idx = self.gage_cur_base_layer
    # index (counting from 0 with the initials) where cl_num could (and will) be derived
    self.gage_cl_layers[cl_num] = layer_idx

    eff_layer_idx = layer_idx-self.gage_cur_base_layer
    if len(self.gage_todo_layers) == eff_layer_idx:
      empty_todo_layer: list[Tuple[int,int,list[int]]] = []
      self.gage_todo_layers.append(empty_todo_layer)
    self.gage_todo_layers[eff_layer_idx].append((cl_num,inf_rule,parents))

  @torch.jit.export
  def gage_enqueue(self,cl_num: int, inf_rule: int, parents: list[int]):
    if self.recording:
      self.gage_infers.append((cl_num,inf_rule,parents))

    '''
    if self.computing:
      self.gage_enqueue_one(cl_num,inf_rule,parents)
    '''

  def gage_embed_pending(self):
    for todos in self.gage_todo_layers:
      # print("gage layers:",len(todos))
      # creating an input to the bulk
      ruleIdxs: list[int] = [] # into gage_rule_embed
      mainPrems = []
      otherPrems = []
      for clNum,infRule,parents in todos:
        ruleIdxs.append(infRule)
        if len(parents) == 0:
          mainPrems.append(torch.zeros(HP.GAGE_EMBEDDING_SIZE))
          otherPrems.append(torch.zeros(HP.GAGE_EMBEDDING_SIZE))
        else:
          mainPrems.append(self.gage_embed_store[parents[0]])
          if len(parents) == 1:
            otherPrems.append(torch.zeros(HP.GAGE_EMBEDDING_SIZE))
          elif len(parents) == 2:
            otherPrems.append(self.gage_embed_store[parents[1]])
          else:
            # this would work even in the binary case, but let's not invoke the monster if we don't need to
            otherPrem = torch.sum(torch.stack([self.gage_embed_store[p] for p in parents[1:]]),dim=0)/(len(parents)-1)
            otherPrems.append(otherPrem)
      ruleEbeds = self.gage_rule_embed(torch.tensor(ruleIdxs))
      mainPremEbeds = torch.stack(mainPrems)
      otherPremEbeds = torch.stack(otherPrems)
      res = self.gage_combine(torch.cat((ruleEbeds, mainPremEbeds, otherPremEbeds), dim=1))
      for j,(clNum,_,_) in enumerate(todos):
        self.gage_embed_store[clNum] = res[j]

    self.gage_cur_base_layer += len(self.gage_todo_layers)
    empty_todo_layers: list[list[Tuple[int,int,list[int]]]] = []
    self.gage_todo_layers = empty_todo_layers

  def gweight_enqueue_one_term(self,id: int, functor: int, sign: float, args: list[int]):
    if args:
      # layer_idx = 1+max(self.gweight_term_layers[a] for a in args if a >= 0)
      layer_idx = 0
      for a in args:
        if a > 0:
          v = self.gweight_term_layers[a]
          if v > layer_idx:
            layer_idx = v
      layer_idx += 1
    else:
      layer_idx = 0
    layer_idx = max(layer_idx,self.gweight_cur_base_layer)

    self.gweight_term_layers[id] = layer_idx

    eff_layer_idx = layer_idx-self.gweight_cur_base_layer
    if len(self.gweight_todo_layers) == eff_layer_idx:
      empty_todo_layer: list[Tuple[int,int,float,list[int]]] = []
      self.gweight_todo_layers.append(empty_todo_layer)
    self.gweight_todo_layers[eff_layer_idx].append((id,functor,sign,args))

  @torch.jit.export
  def gweight_enqueue_term(self,id: int, functor: int, sign: float, args: list[int]):
    if self.recording:
      self.gweight_terms.append((id,functor,sign,args))

    '''
    if self.computing:
      self.gweight_enqueue_one_term(id,functor,sign,args)
    '''

  @torch.jit.export
  def gweight_enqueue_clause(self,cl_num: int, lits: list[int]):
    if self.recording:
      self.gweight_clauses.append((cl_num,lits))

    '''
    if self.computing:
      self.gweight_clause_todo.append((cl_num,lits))
    '''

  def gweight_embed_pending(self):
    # first like what gage does with clause, but here with terms
    for todos in self.gweight_todo_layers:
      # print("gweight layers:",len(todos))

      # TODO: could maybe directly write to a giant tensor via slicing!
      functors = []
      signs = []
      first_args = []
      other_args = []
      for id,functor,sign,args in todos:
        functors.append(self.gweight_symbol_embeds[functor])
        signs.append(torch.tensor([sign]))
        if len(args) == 0:
          first_args.append(torch.zeros(HP.GWEIGHT_EMBEDDING_SIZE))
          other_args.append(torch.zeros(HP.GWEIGHT_EMBEDDING_SIZE))
        else:
          first_args.append(self.gweight_term_embed_store[args[0]])
          if len(args) == 1:
            other_args.append(torch.zeros(HP.GWEIGHT_EMBEDDING_SIZE))
          else:
            other_arg = torch.sum(torch.stack([self.gweight_term_embed_store[a] for a in args[1:]]),dim=0)/(len(args)-1)
            other_args.append(other_arg)

      res = self.gweight_term_combine(torch.cat((torch.stack(functors), torch.stack(signs), torch.stack(first_args), torch.stack(other_args)), dim=1))
      for j,(id,_,_,_) in enumerate(todos):
        self.gweight_term_embed_store[id] = res[j]

    self.gweight_cur_base_layer += len(self.gweight_todo_layers)
    empty_todo_layers: list[list[Tuple[int,int,float,list[int]]]] = []
    self.gweight_todo_layers = empty_todo_layers

    # second, do the clauses part
    for j,(cl_num,lits) in enumerate(self.gweight_clause_todo):
      lit_embeds = torch.stack([self.gweight_term_embed_store[lit] for lit in lits])
      # TODO: try: avg over lits, max over lits, attention over lits, extra non-linearity level, ...
      self.gweight_clause_embeds[cl_num] = torch.sum(lit_embeds,dim=0)
    empty_clause_todo: List[Tuple[int, List[int]]] = []
    self.gweight_clause_todo = empty_clause_todo

  @torch.jit.export
  def embed_pending(self):
    if HP.USE_GAGE:
      self.gage_embed_pending()
    if HP.USE_GWEIGHT:
      self.gweight_embed_pending()

  @torch.jit.export
  def eval_clauses(self, clause_nums: list[int], clause_features: Tensor, gage_embeds: Tensor, gweight_embeds: Tensor) -> Tensor:
    if self.recording:
      for i,cl_num in enumerate(clause_nums):
        self.clause_simple_features[cl_num] = clause_features[i].clone()

    if self.computing:
      feature_parts = []
      if HP.USE_SIMPLE_FEATURES:
        feature_parts.append(clause_features)
      if HP.USE_GAGE:
        feature_parts.append(gage_embeds)
      if HP.USE_GWEIGHT:
        feature_parts.append(gweight_embeds)

      # assumes problems features are already hardwired into clause_embedder's bias
      return self.clause_valuator(self.clause_embedder(torch.cat(feature_parts, dim=1)))

    return torch.zeros(0)


# see: https://discuss.pytorch.org/t/using-torschscript-to-save-a-model-with-multiple-heads/158709
"""
@torch.jit.interface
class LinearInterface(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
      pass
"""

def export_model(model_state_dict,name):
  # we start from a fresh model and just load its state from a saved dict
  m = get_initial_model()
  m.load_state_dict(model_state_dict)

  # eval mode and no gradient
  m.eval()
  for param in m.parameters():
    param.requires_grad = False

  module = MonsterNN(m.problem_embedder,m.clause_embedder,m.clause_valuator,
                    m.gnn_node_init,m.gnn_layers,m.gnn_clause_final,m.gnn_symbol_final,m.gnn_sort_final,
                    m.gage_rule_embed,m.gage_combine,
                    m.gweight_var_embed,m.gweight_term_combine)
  script = torch.jit.script(module)
  script.save(name)

def gage_stats(init_clases,infers):
  """
    Following the logic of gage_enqueue_one, computes the height and width of the gage tree.
  """
  widths = defaultdict(int)
  cl_layers = { cl_num:0 for cl_num in init_clases }
  widths[0] = len(init_clases)
  for (cl_num,inf_rule,parents) in infers:
    if parents:
      layer_idx = 1+max(cl_layers[p] for p in parents)
    else:
      # print(inf_rule)
      layer_idx = 0
    cl_layers[cl_num] = layer_idx
    widths[layer_idx] += 1
  # print("gage_stats",widths)
  return len(widths),max(widths.values())

def gweight_stats(terms):
  """
    Following the logic of gweight_enqueue_one_term, computes the height and width of the gweight tree.
  """
  widths = defaultdict(int)
  term_layers = {}
  for (id,_functor,_sign,args) in terms:
    if args:
      layer_idx = 0
      for a in args:
        if a > 0:
          v = term_layers[a]
          if v > layer_idx:
            layer_idx = v
      layer_idx += 1
    else:
      layer_idx = 0
    term_layers[id] = layer_idx
    widths[layer_idx] += 1
  # print("gweight_stats",widths)
  return len(widths),max(widths.values(),default=0)


def trace_good_for_learning(trace_file_path,logfile=None):
  # open what's been saved and check it
  # if good, save with additional info as needed by learning

  (problem_features,clause_simple_features,journal,proof_units,
   init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
   gage_infers,gweight_terms,gweight_clauses) = torch.load(trace_file_path)

  # scan the journal an check if there are any selections with a good clause in passive at that time
  # also, bake proof_units into the journal, so that we don't need the lookups anymore
  newjournal = []
  passive = set() # just for consistency checking in the loop below
  good_in_passive = 0
  num_good_selections = 0
  for tag,cl_num in journal:
    if tag == EVENT_ADD:
      assert cl_num not in passive
      passive.add(cl_num)
      if cl_num in proof_units:
        good_in_passive += 1
    else:
      assert(tag == EVENT_REM or tag == EVENT_SEL)
      assert cl_num in passive
      passive.remove(cl_num)
      if cl_num in proof_units:
        good_in_passive -= 1
    if tag == EVENT_SEL:
      if good_in_passive > 0: # there is a COM021+4 which gets solved using 30+ selection none of which happens while there is a single proof clause in passive (it's a lrs thing)
        num_good_selections += 1
    newjournal.append((tag,cl_num,cl_num in proof_units))

  gage_h,gage_w = gage_stats(gnn_init_clause_nums,gage_infers)
  gweight_h,gweight_w = gweight_stats(gweight_terms)

  kbSize = os.path.getsize(trace_file_path)//1024

  if logfile is not None:
    # if not num_good_selections:
    #  logfile.write(f"Dropping {trace_file_path} - no steps to learn from\n")
    if HP.USE_GAGE and gage_h > HP.MAX_GAGE_HEIGHT:
      logfile.write(f"Dropping {trace_file_path} - exceeded MAX_GAGE_HEIGHT with its {gage_h}\n")
    if HP.USE_GWEIGHT and gweight_h > HP.MAX_GWEIGHT_HEIGHT:
      logfile.write(f"Dropping {trace_file_path} - exceeded MAX_GWEIGHT_HEIGHT with its {gweight_h}\n")
    if len(clause_simple_features) > HP.MAX_BOX_SIZE:
      logfile.write(f"Dropping {trace_file_path} - exceeded MAX_BOX_SIZE with its {len(clause_simple_features)}\n")
    if kbSize > HP.MAX_KBSIZE:
      logfile.write(f"Dropping {trace_file_path} - exceeded MAX_KBSIZE with its {kbSize}\n")

  if (num_good_selections
      and (not HP.USE_GAGE or gage_h <= HP.MAX_GAGE_HEIGHT)
      and (not HP.USE_GWEIGHT or gweight_h <= HP.MAX_GWEIGHT_HEIGHT)
      and (len(clause_simple_features) <= HP.MAX_BOX_SIZE)
      and kbSize <= HP.MAX_KBSIZE):
    torch.save((problem_features,clause_simple_features,newjournal,num_good_selections,
                init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
                gage_infers,gweight_terms,gweight_clauses),trace_file_path)
    return True, (gage_h,gage_w), (gweight_h,gweight_w)
  else:
    return False, (gage_h,gage_w), (gweight_h,gweight_w)


class LearningModel(torch.nn.Module):
  def __init__(self,
      verbose,
      m: MonsterModules,
      trace_tuple):
    super().__init__()

    self.trace_tuple = trace_tuple
    self.nn = MonsterNN(m.problem_embedder,m.clause_embedder,m.clause_valuator,
                    m.gnn_node_init,m.gnn_layers,m.gnn_clause_final,m.gnn_symbol_final,m.gnn_sort_final,
                    m.gage_rule_embed,m.gage_combine,
                    m.gweight_var_embed,m.gweight_term_combine)
    self.verbose = verbose
    if verbose:
      print("Got verbose")

  def forward(self):
    (problem_features,clause_simple_features,journal,num_good_selections,
                init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
                gage_infers,gweight_terms,gweight_clauses) = self.trace_tuple

    self.nn.computing = True
    self.nn.old_computing = True # some parts are otherwise skipped (as outsourced to cpp)

    self.nn.problem_features = problem_features # not calling set_problem_features (that's only from Vampire)

    if HP.USE_GAGE or HP.USE_GWEIGHT:
      self.nn.gnn_nodes = init_gnn_nodes
      self.nn.gnn_edges = gnn_edges
      self.nn.gnn_perform(gnn_init_clause_nums)

    if HP.USE_GAGE:
      for (cl_num,inf_rule,parents) in gage_infers:
        self.nn.gage_enqueue_one(cl_num,inf_rule,parents)

    if HP.USE_GWEIGHT:
      for (id,functor,sign,args) in gweight_terms:
        self.nn.gweight_enqueue_one_term(id,functor,sign,args)
      self.nn.gweight_clause_todo = gweight_clauses

    if HP.USE_GAGE or HP.USE_GWEIGHT:
      self.nn.embed_pending()

    num2idx = {}

    # emulating MonsterNN.eval_clauses (but note the slight difference with problem_features / problem_embedder)
    simple_feature_vecs = []
    gage_feature_vecs = []
    gweight_feature_vecs = []
    for idx,(cl_num,features) in enumerate(clause_simple_features.items()):
      num2idx[cl_num] = idx
      if HP.USE_SIMPLE_FEATURES:
        simple_feature_vecs.append(features)
      if HP.USE_GAGE:
        gage_feature_vecs.append(self.nn.gage_embed_store[cl_num])
      if HP.USE_GWEIGHT:
        gweight_feature_vecs.append(self.nn.gweight_clause_embeds[cl_num])

    feature_parts = []
    if HP.USE_SIMPLE_FEATURES:
      feature_parts.append(torch.stack(simple_feature_vecs))
    if HP.USE_GAGE:
      feature_parts.append(torch.stack(gage_feature_vecs))
    if HP.USE_GWEIGHT:
      feature_parts.append(torch.stack(gweight_feature_vecs))

    all_features = torch.cat(feature_parts, dim=1)
    embedded_features = self.nn.clause_embedder(all_features)
    if HP.USE_PROBLEM_FEATURES:
      # broadcasting the problem embedding over the matrix clause embeddings?
      embedded_features += self.nn.problem_embedder(problem_features)
    logits = self.nn.clause_valuator(embedded_features)
    logits = logits.squeeze(1) # squeeze-away the second dimension, where the feartures were

    # print("logits",logits.shape)

    good_action_reward_loss = torch.tensor(0.0)
    num_good_steps = 0

    # TODO: couldn't this be one-off compiled to get much more efficient?

    passive = [0]*len(all_features)
    passive_good = [0]*len(all_features)

    learn_for_every = num_good_selections / HP.MAX_TRAINS_PER_TRACE
    learn_for_every_sum = 0.0
    learn_ord = 0
    for tag,cl_num,isGood in journal:
      if cl_num not in num2idx:
        # ignoring clauses we never even had to evaluate in the run
        continue

      idx = num2idx[cl_num]
      if tag == EVENT_ADD:
        passive[idx] += 1
        if isGood:
          passive_good[idx] += 1
        continue
      if tag == EVENT_REM:
        passive[idx] -= 1
        if isGood:
          passive_good[idx] -= 1
        continue

      assert tag == EVENT_SEL

      # newly, we don't merge clauses based on common abstractions, so these arrays are just 0-1
      assert max(passive) <= 1
      assert max(passive_good) <= 1

      # don't learn from every selection for traces with many-many of them (but go and learn at least once)
      if sum(passive_good): # can learn
        if (num_good_steps==0 or random.uniform(0.0, 1.0) < HP.MAX_TRAINS_PER_TRACE / num_good_selections): # is randomized
          # if learn_for_every_sum <= learn_ord: # a deterministic version of the randomized above
          learn_for_every_sum += learn_for_every

          passive_good_t = torch.tensor(passive_good,dtype=logits.dtype)
          passive_t = torch.tensor(passive,dtype=logits.dtype)

          masked_logits = logits[passive_t > 0.0]           # exactly the logis of passive
          passive_good_t = passive_good_t[passive_t > 0.0]  # same lenght as masked_logits, but only contains 1s if it's a good clause

          # manually computing log_softmax with multiplicities
          c = torch.max(masked_logits,dim=-1)[0] # the second part, which we ignore, is the argmax' idx
          exp_logits = torch.exp(masked_logits - c)
          # print("exp_logits.shape",exp_logits.shape)
          logsumexp = torch.log(torch.sum(exp_logits))

          if HP.GOOD_LOGIT_MAX:
            good_logit_max = torch.max(masked_logits[passive_good_t > 0.0],dim=-1)[0]
            good_lsm = good_logit_max-c-logsumexp
          else:
            good_logit_avg = torch.sum(masked_logits[passive_good_t > 0.0])/sum(passive_good)
            good_lsm = good_logit_avg-c-logsumexp

          good_action_reward_loss += -good_lsm

          num_good_steps += 1

        learn_ord += 1

      passive[idx] -= 1
      if isGood:
        passive_good[idx] -= 1

    assert num_good_steps, "The training example was still degenerate!"
    return good_action_reward_loss/num_good_steps
