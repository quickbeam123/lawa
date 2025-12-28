#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

import torch
from torch import Tensor

from typing import List, Final

import torch_geometric

# print(torch.__config__.parallel_info())

from typing import Dict, List, Tuple, Set, Optional

from multiprocessing import Pool

import numpy as np

import sys, random, math

import hyperparams as HP

from collections import defaultdict
from itertools import chain
from sortedcontainers import SortedList


def default_defaultdict_of_list():
  return defaultdict(list)

EVENT_ADD = 0
EVENT_REM = 1
EVENT_SEL = 2

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

STATIC_FEATURES_SIZE : Final[int] = (
                        (HP.NUM_STRATEGY_FEATURES if HP.USE_STRATEGY_FEATURES else 0)
                      + (HP.NUM_PROBLEM_FEATURES if HP.USE_PROBLEM_FEATURES else 0))

CLAUSE_EMBEDDER_INPUT_SIZE : Final[int] = ((HP.NUM_CLAUSE_FEATURES if HP.USE_SIMPLE_FEATURES else 0)
                            + (HP.GAGE_EMBEDDING_SIZE if HP.USE_GAGE else 0)
                            + (HP.GWEIGHT_EMBEDDING_SIZE if HP.USE_GWEIGHT else 0))

class MyFinal(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(size))
        torch.nn.init.kaiming_uniform_(self.weight.unsqueeze(0),a=math.sqrt(5))
        self.maybe_dropout = torch.nn.Dropout(HP.FINAL_LAYER_DROPOUT) if HP.FINAL_LAYER_DROPOUT > 0.0 else torch.nn.Identity()

    def forward(self, x : Tensor):
      x = torch.nn.functional.relu(x)
      x = self.maybe_dropout(x)
      if self.training:
        w_hat = self.weight / (self.weight.norm() + 1e-8)
      else: # saves some cycles for Vampire, but eval loss will be a bit weird?
        w_hat = self.weight

      return torch.matmul(x, w_hat)

    def forward_with_tweaks(self, x : Tensor, tweaks : Tensor):
      if HP.TWEAKS_AS_BIAS:
        x = x + tweaks.unsqueeze(1)

      x = torch.nn.functional.relu(x)
      x = self.maybe_dropout(x)

      if HP.TWEAKS_AS_BIAS:
        if self.training:
          w_hat = self.weight / (torch.linalg.vector_norm(self.weight) + 1e-8)
        else: # saves some cycles for Vampire, but eval loss will be a bit weird?
          w_hat = self.weight

        return torch.matmul(x, w_hat)
      else:
        # we completely ignore self.weight here; assuming it comes as one of the tweaks

        if self.training:
          ws_hat = tweaks / (torch.linalg.vector_norm(tweaks,dim=1,keepdim=True) + 1e-8)
        else:
          ws_hat = tweaks

        return torch.matmul(ws_hat, x.T)

def get_clause_valuator_pair():
  layer_list = [torch.nn.Linear(CLAUSE_EMBEDDER_INPUT_SIZE,HP.INTERAL_SIZE)]

  # so far, we never used multiple layers here
  for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
    layer_list.append(torch.nn.ReLU())
    layer_list.append(torch.nn.Linear(HP.INTERAL_SIZE,HP.INTERAL_SIZE))

  return torch.nn.Sequential(*layer_list),MyFinal(HP.INTERAL_SIZE)

def get_new_tweak():
  return torch.nn.Parameter(torch.zeros(HP.INTERAL_SIZE))

def get_neutral_tweak(myFinalToStealFrom: MyFinal, detached):
  if HP.TWEAKS_AS_BIAS:
    return torch.nn.Parameter(torch.zeros(HP.INTERAL_SIZE))
  else: # tweaks as the final dotter (init from "model")
    if detached:
      return torch.nn.Parameter(myFinalToStealFrom.weight.detach().clone())
    else:
      return myFinalToStealFrom.weight

class MonsterModules(torch.nn.Module):
  # this class only stores all the necessary modules, but does no actual work

  def __init__(self):
    super().__init__()

    self.gnn_node_init = [("sort",  torch.nn.Linear(3,HP.GNN_INTERNAL_SIZE)),
                ("symbol",  torch.nn.Linear(11,HP.GNN_INTERNAL_SIZE)),
                ("clause", torch.nn.Linear(10,HP.GNN_INTERNAL_SIZE)),
                ("term", torch.nn.Linear(10,HP.GNN_INTERNAL_SIZE)),
                ("var", torch.nn.Linear(1,HP.GNN_INTERNAL_SIZE)),] # TODO: discretize to have only a few embeddings? but non-linearly spread?

    self.gnn_clause_final = torch.nn.Sequential(
        torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
        torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE))
    self.gnn_symbol_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)
    self.gnn_sort_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)

    self.gnn_static_embedder = torch.nn.Sequential(
      torch.nn.Linear(STATIC_FEATURES_SIZE,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GNN_INTERNAL_SIZE),
    )

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
    self.gage_static_embedder = torch.nn.Sequential(
      torch.nn.Linear(STATIC_FEATURES_SIZE,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GAGE_EMBEDDING_SIZE)
    )

    # TODO: should the var embed be LayerNormalized, so that it "lives in the same space as the other term embeddings"?
    # first attempt to do this was unstable in training
    # - now we don't do it, but also don't try to add static features to this var embedding (just let it stay weird)

    # TODO: there is no reason for this to be a module; when refectoring, turn this into a parameter (should get fixed on Vampire side too)
    self.gweight_var_embed = SingleEmbedding(embedding_dim=HP.GWEIGHT_EMBEDDING_SIZE)

    self.gweight_term_combine = torch.nn.Sequential(
      torch.nn.Linear(3*HP.GWEIGHT_EMBEDDING_SIZE+1,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Dropout(HP.TREE_DROPOUT) if HP.TREE_DROPOUT > 0.0 else torch.nn.Identity(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE),
      torch.nn.LayerNorm(HP.GWEIGHT_EMBEDDING_SIZE)
    )
    self.gweight_static_embedder = torch.nn.Sequential(
      torch.nn.Linear(STATIC_FEATURES_SIZE,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Linear(HP.INTERAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)
    )

    self.final_static_embedder = torch.nn.Sequential(
      torch.nn.Linear(STATIC_FEATURES_SIZE,HP.INTERAL_SIZE),
      torch.nn.ReLU(),
      torch.nn.Linear(HP.INTERAL_SIZE,CLAUSE_EMBEDDER_INPUT_SIZE)
    )
    self.clause_valuator_fst, self.clause_valuator_snd = get_clause_valuator_pair()

    # by default our MonsterModules carry just one tweak
    self.tweaky = get_new_tweak()

    # while tweaky was for training individual tweaks per problem to go for a nice spread,
    # tweaks are a small set that should stay in and define the generalize search directions to use in strategies by vampire
    self.tweaks = torch.nn.ParameterList([get_new_tweak() for _ in range(HP.TWEAKS_TO_PICK)])


def get_initial_model():
  return MonsterModules()

class MonsterNN(torch.nn.Module):
  # good old simple feature stuff - modules
  # final_static_embedder: torch.nn.Module
  # clause_embedder: torch.nn.Module
  # clause_valuator: torch.nn.Module

  # good old simple feature stuff - records
  static_features: Tensor
  clause_simple_features: Dict[int,Tensor]
  journal: List[Tuple[int,int]]
  # proof_units: List[int] # no need to store here, just dump them to file

  final_static_tweak: Tensor

  # gnn "modules"
  gnn_node_init: List[Tuple[str,torch.nn.modules.linear.Linear]]
  gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]]
  # gnn_clause_final: torch.nn.Module
  # gnn_symbol_final: torch.nn.Module
  # gnn_static_embedder: torch.nn.Module

  # gnn records
  init_gnn_nodes: Dict[str,Tensor]
  gnn_edges: List[Tuple[str,str,Tensor]]
  gnn_init_clause_nums: List[int]

  # gnn helper data
  gnn_nodes: Dict[str,Tensor]

  gnn_static_tweak: Tensor

  # gage modules
  # gage_rule_embed: torch.nn.Module
  # gage_combine: torch.nn.Module
  # gage_static_embedder: torch.nn.Module

  # gage records
  gage_infers: List[Tuple[int,int,List[int]]]

  # gage helper data
  gage_embed_store: Dict[int,Tensor]
  gage_cl_layers: Dict[int,int]
  gage_cur_base_layer: int
  gage_todo_layers: List[List[Tuple[int,int,List[int]]]]

  gage_static_tweak: Tensor

  # gweight modules
  # gweight_var_embed: SingleEmbedding
  # gweight_term_combine: torch.nn.Module
  # gweight_static_embedder: torch.nn.Module

  # gweight records
  gweight_terms: List[Tuple[int,int,float,List[int]]]
  gweight_clauses: List[Tuple[int,List[int]]]

  # gweight helper data
  gweight_symbol_embeds: Tensor
  gweight_term_embed_store: Dict[int,Tensor]
  gweight_term_layers: Dict[int,int]
  gweight_cur_base_layer: int
  gweight_todo_layers: List[List[Tuple[int,int,float,List[int]]]]

  gweight_clause_todo: List[Tuple[int,List[int]]]
  gweight_clause_embeds: Dict[int,Tensor]

  gweight_static_tweak: Tensor

  def __init__(self,
              gnn_node_init,gnn_layers,gnn_clause_final,gnn_symbol_final,gnn_sort_final,gnn_static_embedder,
              gage_rule_embed, gage_combine, gage_static_embedder,
              gweight_var_embed, gweight_term_combine, gweight_static_embedder,
              final_static_embedder, clause_valuator_fst, clause_valuator_snd, tweaky, tweaks):
    super().__init__()

    self.recording = False
    self.computing = False
    self.old_computing = False

    # This is crazy, but while we don't need this for any computation, things don't jit.script witout it!
    # (maybe its necessary so that the annotations above can be digested?
    # I think it's the gnn_node_init and gnn_layers, who's types include modules)
    self.dummy = torch_geometric.nn.SAGEConv((1,1),1)
    self.dummy2 = torch.nn.Linear(1,1)

    # modules-like
    self.gnn_node_init = gnn_node_init
    self.gnn_layers = gnn_layers
    self.gnn_clause_final = gnn_clause_final
    self.gnn_symbol_final = gnn_symbol_final
    self.gnn_sort_final = gnn_sort_final
    self.gnn_static_embedder = gnn_static_embedder

    # records
    self.init_gnn_nodes = {}
    self.gnn_nodes = {}
    self.gnn_edges = []
    self.gnn_init_clause_nums = [] # will be set from Vampire later anyway
    self.gnn_static_tweak = torch.zeros(HP.GNN_INTERNAL_SIZE) # dummy, overwritten by set_static_features

    # modules
    self.gage_rule_embed = gage_rule_embed
    self.gage_combine = gage_combine
    self.gage_static_embedder = gage_static_embedder

    # records
    self.gage_infers = []
    self.gage_static_tweak = torch.zeros(HP.GAGE_EMBEDDING_SIZE) # dummy, overwritten by set_static_features

    # helpers
    self.gage_embed_store = {}
    self.gage_cl_layers = {}
    self.gage_cur_base_layer = 1
    self.gage_todo_layers = []

    # modules
    self.gweight_var_embed = gweight_var_embed
    self.gweight_term_combine = gweight_term_combine
    self.gweight_static_embedder = gweight_static_embedder

    # records
    self.gweight_terms = []
    self.gweight_clauses = []
    self.gweight_static_tweak = torch.zeros(HP.GWEIGHT_EMBEDDING_SIZE) # dummy, overwritten by set_static_features

    # helpers
    self.gweight_symbol_embeds = torch.zeros(0) # dummy, overwritten by gnn_perform
    self.gweight_term_embed_store = {}
    self.gweight_term_layers = {}
    self.gweight_cur_base_layer = 1
    self.gweight_todo_layers = []

    self.gweight_clause_todo = []
    self.gweight_clause_embeds = {}

    # modules
    self.final_static_embedder = final_static_embedder
    self.clause_valuator_fst = clause_valuator_fst
    self.clause_valuator_snd = clause_valuator_snd

    # records
    self.static_features = torch.zeros(0) # dummy, overwritten by set_static_features
    self.clause_simple_features = {} # filled up gradually, only when recording
    self.journal = [] # filled up gradually, only when recording
    self.proof_units = []
    self.final_static_tweak = torch.zeros(CLAUSE_EMBEDDER_INPUT_SIZE) # dummy, overwritten by set_static_features

    # modules/parameters:
    self.tweaky = tweaky
    self.tweaks = tweaks

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
  def bake_tweak(self, tweak): # because it actually adds, it only makes sense to call this once!
    with torch.no_grad():
      if HP.TWEAKS_AS_BIAS:
        self.clause_valuator_fst[-1].bias.add_(tweak)
      else:
        self.clause_valuator_snd.weight.copy_(tweak)

  @torch.jit.export
  def set_static_features(self, features: Tensor):
    # print("set_static_features",features)
    if self.recording:
      self.static_features = features.clone()

    if self.computing:
      total_len = HP.NUM_STRATEGY_FEATURES + HP.NUM_PROBLEM_FEATURES
      # we assume vampire is giving us everything, but if we don't want something, we need to crop it out!
      idx_from = 0 if HP.USE_PROBLEM_FEATURES else HP.NUM_PROBLEM_FEATURES
      idx_to = total_len if HP.USE_STRATEGY_FEATURES else total_len-HP.NUM_STRATEGY_FEATURES

      features = features[idx_from:idx_to]

      if (HP.USE_GAGE or HP.USE_GWEIGHT) and HP.FEED_STATIC_FEAUTURES_TO_GNN:
        self.gnn_static_tweak = self.gnn_static_embedder(features)

      if HP.USE_GAGE and HP.FEED_STATIC_FEAUTURES_TO_THE_TREES:
        self.gage_static_tweak = self.gage_static_embedder.forward(features)

      if HP.USE_GWEIGHT and HP.FEED_STATIC_FEAUTURES_TO_THE_TREES:
        self.gweight_static_tweak = self.gweight_static_embedder.forward(features)

      if HP.FEED_STATIC_FEATURES_FINAL_MLP:
        self.final_static_tweak = self.final_static_embedder.forward(features)

    """
    print("set_static_features - gnn_static_tweak",self.gnn_static_tweak)
    print("set_static_features - gage_static_twea",self.gage_static_tweak)
    print("set_static_features - gweight_static_tweak",self.gweight_static_tweak)
    print("set_static_features - final_static_tweak",self.final_static_tweak)
    """

  @torch.jit.export
  def gnn_node_kind(self,what: str,features: Tensor):
    if self.recording:
      self.init_gnn_nodes[what] = features.clone()

    # the caller guarantees the tensor will still be alive when gnn_perform is called
    self.gnn_nodes[what] = features

  @torch.jit.export
  def gnn_edge_kind(self,src: str, tgt: str, src_idxs: List[int], tgt_idxs: List[int]):
    src_idxs_t = torch.tensor(src_idxs)
    tgt_idxs_t = torch.tensor(tgt_idxs)

    self.gnn_edges.append((src,tgt,torch.stack([src_idxs_t,tgt_idxs_t])))
    # also record the opposite edge
    self.gnn_edges.append((tgt,src,torch.stack([tgt_idxs_t,src_idxs_t])))

  @torch.jit.export
  def gnn_perform(self, clause_nums: List[int]) -> Tuple[Tensor,Tensor]:
    # the clause numbers in clause_nums are promised to go in the same order as the clauses in previously added via gnnNodeKind("clause",...)
    if self.recording:
      self.gnn_init_clause_nums = clause_nums

    if self.computing:
      for key,embedder in self.gnn_node_init:
        self.gnn_nodes[key] = embedder.forward(self.gnn_nodes[key]).relu() + self.gnn_static_tweak # broadcasting to every embedded node
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
      initial_clause_gage += self.gage_static_tweak # broadcasting for every inital clause

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
  def set_proof_units_and_save_recorded(self, proof_units: List[int], filename: str):
    # does not really matter, just save them below
    # self.proof_units = proof_units

    torch.save((self.static_features,self.clause_simple_features,self.journal,proof_units,
                self.init_gnn_nodes,self.gnn_edges,self.gnn_init_clause_nums,
                self.gage_infers,self.gweight_terms,self.gweight_clauses),filename)

  def gage_enqueue_one(self,cl_num: int, inf_rule: int, parents: List[int]):
    layer_idx = 0
    for p in parents:
      layer_idx = max(layer_idx,self.gage_cl_layers[p])
    layer_idx += 1
    layer_idx = max(layer_idx,self.gage_cur_base_layer)

    # index (counting from 0 with the initials) where cl_num could (and will) be derived
    self.gage_cl_layers[cl_num] = layer_idx

    eff_layer_idx = layer_idx-self.gage_cur_base_layer
    if len(self.gage_todo_layers) == eff_layer_idx:
      empty_todo_layer: List[Tuple[int,int,List[int]]] = []
      self.gage_todo_layers.append(empty_todo_layer)
    self.gage_todo_layers[eff_layer_idx].append((cl_num,inf_rule,parents))

  @torch.jit.export
  def gage_enqueue(self,cl_num: int, inf_rule: int, parents: List[int]):
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
      ruleIdxs: List[int] = [] # into gage_rule_embed
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
      res += self.gage_static_tweak # broadcasting for every line in res
      for j,(clNum,_,_) in enumerate(todos):
        self.gage_embed_store[clNum] = res[j]

    self.gage_cur_base_layer += len(self.gage_todo_layers)
    empty_todo_layers: List[List[Tuple[int,int,List[int]]]] = []
    self.gage_todo_layers = empty_todo_layers

  def gweight_enqueue_one_term(self,id: int, functor: int, sign: float, args: List[int]):
    layer_idx = 0
    for a in args:
      if a > 0:
        layer_idx = max(layer_idx,self.gweight_term_layers[a])
    layer_idx += 1
    layer_idx = max(layer_idx,self.gweight_cur_base_layer)

    self.gweight_term_layers[id] = layer_idx

    eff_layer_idx = layer_idx-self.gweight_cur_base_layer
    if len(self.gweight_todo_layers) == eff_layer_idx:
      empty_todo_layer: List[Tuple[int,int,float,List[int]]] = []
      self.gweight_todo_layers.append(empty_todo_layer)
    self.gweight_todo_layers[eff_layer_idx].append((id,functor,sign,args))

  @torch.jit.export
  def gweight_enqueue_term(self,id: int, functor: int, sign: float, args: List[int]):
    if self.recording:
      self.gweight_terms.append((id,functor,sign,args))

    '''
    if self.computing:
      self.gweight_enqueue_one_term(id,functor,sign,args)
    '''

  @torch.jit.export
  def gweight_enqueue_clause(self,cl_num: int, lits: List[int]):
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
      res += self.gweight_static_tweak # broadcasting for every line in res
      for j,(id,_,_,_) in enumerate(todos):
        self.gweight_term_embed_store[id] = res[j]

    self.gweight_cur_base_layer += len(self.gweight_todo_layers)
    empty_todo_layers: List[List[Tuple[int,int,float,List[int]]]] = []
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

  def pre_eval_clauses(self, clause_features: Tensor, gage_embeds: Tensor, gweight_embeds: Tensor) -> Tensor:
    feature_parts = []
    if HP.USE_SIMPLE_FEATURES:
      feature_parts.append(clause_features)
    if HP.USE_GAGE:
      feature_parts.append(gage_embeds)
    if HP.USE_GWEIGHT:
      feature_parts.append(gweight_embeds)

    all_features = torch.cat(feature_parts, dim=1) + self.final_static_tweak # broadcasting for every clause
    return self.clause_valuator_fst(all_features)

  @torch.jit.export
  def eval_clauses(self, clause_nums: List[int], clause_features: Tensor, gage_embeds: Tensor, gweight_embeds: Tensor) -> Tensor:
    if self.recording:
      for i,cl_num in enumerate(clause_nums):
        self.clause_simple_features[cl_num] = clause_features[i].clone()

    if self.computing:
      just_before_final = self.pre_eval_clauses(clause_features,gage_embeds,gweight_embeds)
      return self.clause_valuator_snd(just_before_final)

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

  module = MonsterNN(m.gnn_node_init,m.gnn_layers,m.gnn_clause_final,m.gnn_symbol_final,m.gnn_sort_final,m.gnn_static_embedder,
                     m.gage_rule_embed,m.gage_combine,m.gage_static_embedder,
                     m.gweight_var_embed,m.gweight_term_combine,m.gweight_static_embedder,
                     m.final_static_embedder,m.clause_valuator_fst,m.clause_valuator_snd,m.tweaky,m.tweaks)
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
    layer_idx = 0
    for p in parents:
      layer_idx = max(layer_idx,cl_layers[p])
    layer_idx += 1

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
    layer_idx = 0
    for a in args:
      if a > 0:
        layer_idx = max(layer_idx,term_layers[a])
    layer_idx += 1
    term_layers[id] = layer_idx
    widths[layer_idx] += 1
  # print("gweight_stats",widths)
  return len(widths),max(widths.values(),default=0)


def trace_good_for_learning(trace_file_path,logfile=None):
  # open what's been saved and check it
  # if good, save with additional info as needed by learning

  (static_features,clause_simple_features,journal,proof_units,
   init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
   gage_infers,gweight_terms,gweight_clauses) = torch.load(trace_file_path,weights_only=False)

  # scan the journal and check if there are any selections with a good clause in passive at that time
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

      if tag == EVENT_SEL:
        if good_in_passive > 0:
          num_good_selections += 1

      # it's only getting removed as part of this selection step!
      if cl_num in proof_units:
        good_in_passive -= 1

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

  non_trivial = num_good_selections
  passes_limits = ((not HP.USE_GAGE or gage_h <= HP.MAX_GAGE_HEIGHT)
                    and (not HP.USE_GWEIGHT or gweight_h <= HP.MAX_GWEIGHT_HEIGHT)
                    and (len(clause_simple_features) <= HP.MAX_BOX_SIZE)
                    and kbSize <= HP.MAX_KBSIZE)

  if (non_trivial and passes_limits):
    torch.save((static_features,clause_simple_features,newjournal,num_good_selections,
                init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
                gage_infers,gweight_terms,gweight_clauses),trace_file_path)

  return non_trivial, passes_limits, (gage_h,gage_w), (gweight_h,gweight_w)


class LearningModel(torch.nn.Module):
  def __init__(self,
      verbose,
      m: MonsterModules,
      trace_tuple):
    super().__init__()

    self.trace_tuple = trace_tuple

    self.nn = MonsterNN(
                    m.gnn_node_init,m.gnn_layers,m.gnn_clause_final,m.gnn_symbol_final,m.gnn_sort_final,m.gnn_static_embedder,
                    m.gage_rule_embed,m.gage_combine,m.gage_static_embedder,
                    m.gweight_var_embed,m.gweight_term_combine,m.gweight_static_embedder,
                    m.final_static_embedder,m.clause_valuator_fst,m.clause_valuator_snd,m.tweaky,m.tweaks)
    self.verbose = verbose
    if verbose:
      print("Got verbose")

  def pre_forward(self):
    (static_features,clause_simple_features,_journal,_num_good_selections,
                init_gnn_nodes,gnn_edges,gnn_init_clause_nums,
                gage_infers,gweight_terms,gweight_clauses) = self.trace_tuple

    self.nn.computing = True
    self.nn.old_computing = True # some parts are otherwise skipped (as outsourced to cpp)

    self.nn.set_static_features(static_features)

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

    return self.nn.pre_eval_clauses(torch.stack(simple_feature_vecs),torch.stack(gage_feature_vecs),torch.stack(gweight_feature_vecs)),num2idx

  def forward(self,just_before_final,num2idx,tweaks=None):
    (_static_features,_clause_simple_features,journal,num_good_selections,
                _init_gnn_nodes,_gnn_edges,_gnn_init_clause_nums,
                _gage_infers,_gweight_terms,_gweight_clauses) = self.trace_tuple

    # print("just_before_final",just_before_final.shape)
    if tweaks is not None:
      logits = self.nn.clause_valuator_snd.forward_with_tweaks(just_before_final,tweaks)
      squeeze_back = False
    else:
      logits = self.nn.clause_valuator_snd.forward(just_before_final).unsqueeze(0)
      squeeze_back = True

    num_loss_channels = logits.shape[0]
    # num_clauses = logits.shape[1]

    good_action_reward_loss = torch.zeros(num_loss_channels)
    num_good_steps = 0

    passive = set()
    passive_good = set()

    # to compute our metric (as opposed the loss)
    sorted_passive = [SortedList(key=lambda cl_idx, channel=chan : -logits[channel,cl_idx].item()) for chan in range(num_loss_channels)]

    num_sels = 0
    selection_hits = [0]*num_loss_channels
    dists_to_good = [0.0]*num_loss_channels

    for tag,cl_num,isGood in journal:
      if cl_num not in num2idx:
        # ignoring clauses we never even had to evaluate in the run
        continue

      idx = num2idx[cl_num]
      if tag == EVENT_ADD:
        passive.add(idx)
        for chan in range(num_loss_channels):
          sorted_passive[chan].add(idx)
        if isGood:
          passive_good.add(idx)
        continue

      if tag == EVENT_SEL and len(passive_good): # can learn

        # don't learn from every selection for traces with many-many of them (but go and learn at least once)
        if (num_good_steps==0 or random.uniform(0.0, 1.0) < HP.MAX_TRAINS_PER_TRACE / num_good_selections): # is randomized

          # computing the "ML statistics" about how close the good clauses would be to the beginning of our queue here
          num_sels += 1
          for chan in range(num_loss_channels):
            first_good = 0
            for first_good,ith_idx in enumerate(sorted_passive[chan]):
              if ith_idx in passive_good:
                break
            if first_good>0:
              dists_to_good[chan] += first_good / (len(sorted_passive[chan])-1) # make it span <0,1>
            else:
              selection_hits[chan] += 1

          # computing the loss
          passive_l = sorted(passive)
          passive_t = torch.tensor(passive_l, dtype=torch.long)
          c = 1/len(passive_good)
          passive_good_l = [c if idx in passive_good else 0.0 for idx in passive_l]
          passive_good_t = torch.tensor(passive_good_l, dtype=logits.dtype).unsqueeze(0).expand(num_loss_channels,-1)

          gathered_logits = logits[:,passive_t]

          good_action_reward_loss += torch.nn.functional.cross_entropy(
            gathered_logits,
            passive_good_t,
            reduction="none",
            label_smoothing=HP.LABEL_SMOOTHING)

          num_good_steps += 1

      passive.remove(idx)
      for chan in range(num_loss_channels):
        sorted_passive[chan].remove(idx)

      if isGood:
        passive_good.remove(idx)

    assert num_good_steps, "The training example was still degenerate!"
    for chan in range(num_loss_channels):
      selection_hits[chan] /= num_sels
      dists_to_good[chan] /= num_sels

    if squeeze_back:
      return good_action_reward_loss.squeeze(0)/num_good_steps, selection_hits[0], dists_to_good[0]
    return good_action_reward_loss/num_good_steps, selection_hits, dists_to_good
