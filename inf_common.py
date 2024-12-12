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

def default_defaultdict_of_list():
  return defaultdict(list)

EVENT_ADD = 0
EVENT_REM = 1
EVENT_SEL = 2

def vampire_perfrom(prob,opts):
  to_run = " ".join(["./run_lawa_vampire.sh",HP.VAMPIRE_EXECUTABLE,opts,prob])
  # print(to_run)
  output = subprocess.getoutput(to_run)

  status = None
  instructions = 0
  activations = 0

  for line in output.split("\n"):
    # print("  ",line)
    if line.startswith("%"):
      if line.startswith("% Activations started:"):
        activations = int(line.split()[-1])
      if line.startswith("% Instructions burned:"):
        instructions = int(line.split()[-2])
      if line.startswith("% SZS status"):
        if "Satisfiable" in line or "CounterSatisfiable" in line:
          status = "sat"
        elif "Theorem" in line or "Unsatisfiable" in line or "ContradictoryAxioms" in line:
          status = "uns"

  # print(status,instructions,activations)
  return (status,instructions,activations)

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
              gnn_node_init,gnn_layers,gnn_clause_final,gnn_symbol_final,
              gage_rule_embed, gage_combine,
              gweight_var_embed, gweight_term_combine
              ):
    super().__init__()

    self.recording = False
    self.computing = False

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
  def set_recording(self):
    self.recording = True

  @torch.jit.export
  def set_computing(self):
    self.computing = True

  @torch.jit.export
  def set_problem_features(self, features: Tensor):
    if self.recording:
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
  def gnn_perform(self, clause_nums: list[int]):
    if self.recording:
      self.gnn_init_clause_nums = clause_nums

    if self.computing:
      # the clause numbers in clause_nums are promised to go in the same order as the clauses in previously added via gnnNodeKind("clause",...)
      for key,embedder in self.gnn_node_init:
        self.gnn_nodes[key] = embedder.forward(self.gnn_nodes[key]).relu()

      for layer in self.gnn_layers:
        out_dict: Dict[str, Tensor] = {}
        for src,tgt,i,conv in layer:
          out = conv.forward((self.gnn_nodes[src],self.gnn_nodes[tgt]),self.gnn_edges[i][2])
          if tgt in out_dict:
            out_dict[tgt] = out_dict[tgt] + out
          else:
            out_dict[tgt] = out

        for key, out in out_dict.items():
          self.gnn_nodes[key] = out.relu()
          out_dict = {}

      # TODO: in the future could also pool things and extract a (more refined) problem embedding to use

      initial_clause_gage = self.gnn_clause_final.forward(self.gnn_nodes["clause"])
      self.gweight_symbol_embeds = self.gnn_symbol_final.forward(self.gnn_nodes["symbol"])

      # pass on the gage-style clause embeddings to the gage part (using clause_nums)
      for i,cl_num in enumerate(clause_nums):
        self.gage_embed_store[cl_num] = initial_clause_gage[i]
        self.gage_cl_layers[cl_num] = 0

      # can drop all the gnn stuff not needed anymore
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
    layer_idx = max(1+max(self.gage_cl_layers[p] for p in parents),self.gage_cur_base_layer)
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

    if self.computing:
      self.gage_enqueue_one(cl_num,inf_rule,parents)

  def gage_embed_pending(self):
    for todos in self.gage_todo_layers:
      # print("gage layers:",len(todos))
      # creating an input to the bulk
      ruleIdxs: list[int] = [] # into gage_rule_embed
      mainPrems = []
      otherPrems = []
      for clNum,infRule,parents in todos:
        ruleIdxs.append(infRule)
        mainPrems.append(self.gage_embed_store[parents[0]])
        if len(parents) == 1:
          otherPrems.append(torch.zeros(HP.GAGE_EMBEDDING_SIZE))
        elif len(parents) == 2:
          otherPrems.append(self.gage_embed_store[parents[1]])
        else:
          # this would work even in the binary case, but let's not invoke the monster if we don't need to
          otherPrem = torch.sum(torch.stack(self.gage_embed_store[parents[p]] for p in parents[1:]),dim=0)/(len(parents)-1)
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
        if a >= 0:
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

    if self.computing:
      self.gweight_enqueue_one_term(id,functor,sign,args)

  @torch.jit.export
  def gweight_enqueue_clause(self,cl_num: int, lits: list[int]):
    if self.recording:
      self.gweight_clauses.append((cl_num,lits))

    if self.computing:
      self.gweight_clause_todo.append((cl_num,lits))

  def get_subterm_embed(self,id: int) -> Tensor:
    if id < 0:
      return self.gweight_var_embed(torch.tensor([id % HP.GWEIGHT_NUM_VAR_EMBEDS]))[0]
    else:
      return self.gweight_term_embed_store[id]

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
          first_args.append(self.get_subterm_embed(args[0]))
          if len(args) == 1:
            other_args.append(torch.zeros(HP.GWEIGHT_EMBEDDING_SIZE))
          else:
            other_arg = torch.sum(torch.stack(self.get_subterm_embed(a) for a in args[1:]),dim=0)/(len(args)-1)
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
    self.gage_embed_pending()
    self.gweight_embed_pending()

  @torch.jit.export
  def eval_clauses(self, clause_nums: list[int], clause_features: Tensor):
    if self.recording:
      for i,cl_num in enumerate(clause_nums):
        self.clause_simple_features[cl_num] = clause_features[i].clone()

    # if self.computing:
    gage_features = torch.stack([self.gage_embed_store[cl_num] for cl_num in clause_nums])
    gweight_features = torch.stack([self.gweight_clause_embeds[cl_num] for cl_num in clause_nums])
    all_features = torch.cat((clause_features, gage_features, gweight_features), dim=1)

    # assumes problems features are already hardwired into clause_embedder's bias
    return self.clause_valuator(self.clause_embedder(all_features))


# see: https://discuss.pytorch.org/t/using-torschscript-to-save-a-model-with-multiple-heads/158709
"""
@torch.jit.interface
class LinearInterface(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
      pass
"""

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
      aggr="mean",      # TODO: consider trying out SUM/MAX
      normalize=False,  # a bit like a layernorm on the output?
      root_weight=True, # like a self-loop; i.e. allow then target node to talk as well
      project=False,    # extra non-lineary before aggregating
      bias=True)        # and why not add a bias before the relu that's about to come?


def get_monster():
  problem_embedder = torch.nn.Linear(HP.NUM_PROBLEM_FEATURES,HP.INTERAL_SIZE,bias=False)
  clause_embedder = torch.nn.Linear(HP.NUM_CLAUSE_FEATURES+HP.GAGE_EMBEDDING_SIZE+HP.GWEIGHT_EMBEDDING_SIZE,HP.INTERAL_SIZE)
  clause_valuator = get_clause_valuator()

  gnn_node_init = [("sort",  torch.nn.Linear(3,HP.GNN_INTERNAL_SIZE)),
               ("symbol",  torch.nn.Linear(10,HP.GNN_INTERNAL_SIZE)),
               ("clause", torch.nn.Linear(10,HP.GNN_INTERNAL_SIZE)),
               ("term", torch.nn.Linear(10,HP.GNN_INTERNAL_SIZE)),
               ("var", torch.nn.Linear(1,HP.GNN_INTERNAL_SIZE)),] # TODO: discretize to have only a few embeddings? but non-linearly spread?

  gnn_clause_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GAGE_EMBEDDING_SIZE)
  gnn_symbol_final = torch.nn.Linear(HP.GNN_INTERNAL_SIZE,HP.GWEIGHT_EMBEDDING_SIZE)

  modules = [embed for node_str,embed in gnn_node_init] + [gnn_clause_final,gnn_symbol_final]

  gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]] = []
  for lidx in range(HP.GNN_NUM_LAYERS):
    layer = []
    for i,(src,tgt) in enumerate([('symbol', 'sort'), ('sort', 'symbol'), ('symbol', 'symbol'), ('symbol', 'symbol'),
                                  ('clause', 'term'), ('term', 'clause'), ('term', 'term'), ('term', 'term'),
                                  ('clause', 'var'), ('var', 'clause'), ('var', 'sort'), ('sort', 'var'),
                                  ('term', 'var'), ('var', 'term'), ('term', 'symbol'), ('symbol', 'term')]):
      # in the last layer, no need for any other output than ["symbol","clause"]
      if lidx != HP.GNN_NUM_LAYERS-1 or tgt in ["symbol","clause"]:
        conv = get_conv()
        modules.append(conv)
        layer.append((src,tgt,i,conv))
    gnn_layers.append(layer)

  gage_rule_embed = torch.nn.Embedding(num_embeddings=HP.NUM_INFERENCE_RULES, embedding_dim=HP.GAGE_EMBEDDING_SIZE)
  gage_combine = torch.nn.Sequential(
     torch.nn.Linear(3*HP.GAGE_EMBEDDING_SIZE,HP.INTERAL_SIZE),
     torch.nn.ReLU(),
     # TODO: experiment with dropout?
     torch.nn.Linear(HP.INTERAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
     torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE)
  )

  gweight_var_embed = torch.nn.Embedding(num_embeddings=HP.GWEIGHT_NUM_VAR_EMBEDS, embedding_dim=HP.GWEIGHT_EMBEDDING_SIZE)
  gweight_term_combine = torch.nn.Sequential(
     torch.nn.Linear(3*HP.GAGE_EMBEDDING_SIZE+1,HP.INTERAL_SIZE),
     torch.nn.ReLU(),
     # TODO: experiment with dropout?
     torch.nn.Linear(HP.INTERAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
     torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE)
  )

  model = MonsterNN(problem_embedder,clause_embedder,clause_valuator,
                    gnn_node_init,gnn_layers,gnn_clause_final,gnn_symbol_final,
                    gage_rule_embed,gage_combine,
                    gweight_var_embed,gweight_term_combine)

  return model







def vampire_gather(prob,opts):
  to_run = " ".join(["./run_lawa_vampire.sh",HP.VAMPIRE_EXECUTABLE,opts,prob])
  # print(to_run)
  for _ in range(5): # sometimes, there are weird failures, but a restart could help!
    output = subprocess.getoutput(to_run)

    clauses = {}         # id -> (feature_vec)
    journal = []         # [event,id], where event is one of EVENT_ADD EVENT_SEL EVENT_REM,
    proof_flas = set()

    # just temporaries, here during the parsing
    num_sels = 0

    for line in output.split("\n"):
      # print(line)
      if "% Instruction limit reached!" in line:
        assert not proof_flas
        break # better than "id appeared again for:" failing just below
      if line.startswith("P: "):
        spl = line.split()
        problem_features = list(map(float,spl[1:]))
      if line.startswith("i: "):
        spl = line.split()
        id = int(spl[1])
        features = list(map(float,spl[2:]))
        assert len(features) == HP.NUM_CLAUSE_FEATURES
        # print(id,features)
        assert id not in clauses, "id appeared again for: "+to_run
        clauses[id] = features
      elif line.startswith("a: "):
        spl = line.split()
        id = int(spl[1])
        journal.append([EVENT_ADD,id])
      elif line.startswith("s: "):
        spl = line.split()
        id = int(spl[1])
        journal.append([EVENT_SEL,id])
        num_sels += 1
      elif line.startswith("r: "):
        spl = line.split()
        id = int(spl[1])
        journal.append([EVENT_REM,id])

      elif "Aborted by signal" in line:
        # print("Error line:",line)
        proof_flas = set() # so that we continue the outer loop below
        break
      elif line and line[0] in "123456789":
        spl = line.split(".")
        id = int(spl[0])
        proof_flas.add(id)

    if len(proof_flas) == 0:
      # print("Proof not found for",to_run)
      # print("Will retry!")
      continue

    # a success
    # in the first coordinate, however, we still say whether there is non-trivial stuff to learn from
    return (len(clauses) != 0 and num_sels != 0,(problem_features,clauses,journal,proof_flas))

  print("Repeatedly failing:",to_run)
  print(output)
  return None # meaning: "A major failure, consider keeping the used model for later debugging"

# taking into account that some clauses may have the same feature vector,
# let's not waste space and time on those and create a compressed trace where
# each unique feature vector appears only once
def compress_trace(problem_features,clauses,journal,proof_flas):
  id2idx = {}
  features2idx = {}
  clause_features = []
  for cl,features in clauses.items():
    tfeatures = tuple(features)
    if tfeatures in features2idx:
      id2idx[cl] = features2idx[tfeatures]
    else:
      newIdx = len(features2idx)
      features2idx[tfeatures] = newIdx
      id2idx[cl] = newIdx
      clause_features.append(features)
  newjournal = []
  passive = set() # just for consistency checking in the loop below
  good_in_passive = 0
  num_good_selections = 0
  for tag,id in journal:
    if tag == EVENT_ADD:
      assert id not in passive
      passive.add(id)
      if id in proof_flas:
        good_in_passive += 1
    else:
      assert(tag == EVENT_REM or tag == EVENT_SEL)
      assert id in passive
      passive.remove(id)
      if id in proof_flas:
        good_in_passive -= 1
    if tag == EVENT_SEL:
      if good_in_passive > 0: # there is a COM021+4 which gets solved using 30+ selection none of which happens while there is a single proof clause in passive (it's a lrs thing)
        num_good_selections += 1
    newjournal.append((tag,id2idx[id],id in proof_flas))
  return problem_features,clause_features,newjournal,num_good_selections

class SimpleClauseEvaluator(torch.nn.Module):
  def __init__(self):
    super().__init__()

    assert HP.CLAUSE_EMBEDDER_LAYERS > 0

    self.problem_embedder = torch.nn.Linear(HP.NUM_PROBLEM_FEATURES,HP.INTERAL_SIZE,bias=False)
    self.clause_embedder = torch.nn.Linear(HP.NUM_CLAUSE_FEATURES,HP.INTERAL_SIZE)

    layer_list = [torch.nn.ReLU()]
    for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
      layer_list.append(torch.nn.Linear(HP.INTERAL_SIZE,HP.INTERAL_SIZE))
      layer_list.append(torch.nn.ReLU())
    layer_list.append(torch.nn.Linear(HP.INTERAL_SIZE,1,bias=False))

    self.valuator = torch.nn.Sequential(*layer_list)

  def forward(self,problem_features,clause_feature_vecs) -> Tensor:
    return self.valuator(self.clause_embedder(clause_feature_vecs)+self.problem_embedder(problem_features))  # will broadcast work here?

def get_initial_model():
  return SimpleClauseEvaluator()

def export_model(model_state_dict,name):
  # we start from a fresh model and just load its state from a saved dict
  model = get_initial_model()
  model.load_state_dict(model_state_dict)

  # eval mode and no gradient
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    def __init__(self,problem_embedder : torch.nn.Module, clause_embedder : torch.nn.Module, valuator : torch.nn.Module):
      super().__init__()

      self.problem_embedder = problem_embedder
      self.clause_embedder = clause_embedder
      self.valuator = valuator

    @torch.jit.export
    def setProblemFeatures(self,features : Tensor):
      with torch.no_grad():
        self.clause_embedder.bias.add_(self.problem_embedder(features))

    @torch.jit.export
    def forward(self,features : Tensor):
      # assert len(features) == HP.NUM_CLAUSE_FEATURES

      return self.valuator(self.clause_embedder(features))

  module = NeuralPassiveClauseContainer(model.problem_embedder,model.clause_embedder,model.valuator)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      verbose,
      evaluator : torch.nn.Module,
      problem_features,clause_features,journal,num_good_selections):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(f"clause {len(clauses)} journal {len(journal)} proof_flas {len(proof_flas)}")

    self.verbose = verbose

    self.evaluator = evaluator                     # the SimpleClauseEvaluator for clause evaluation
    self.problem_features = problem_features
    self.clause_features = clause_features         # list of clause features (in a list); idx in the journal is into this list
    self.journal = journal                         # (event,idx,is_proof_cl), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.num_good_selections = num_good_selections # how many useful EVENT_SEL are there in journal?

    if verbose:
      print("Got verbose")
      print(len(clause_features))
      print(len(journal))
      print(num_good_selections)

  def forward(self):
    # let's a get a big matrix of feature_vec's, one for each clause idx
    clause_feature_vecs = torch.stack([torch.tensor(features) for features in self.clause_features])
    # print("feature_vecs.shape",feature_vecs.shape)

    logits = self.evaluator.forward(torch.tensor(self.problem_features),clause_feature_vecs)
    logits = logits.squeeze(1) # squeeze-away second dimension, where the feartures were

    # print("logits",logits.shape)

    good_action_reward_loss = torch.tensor(0.0)
    num_good_steps = 0

    # TODO: couldn't this be one-off compiled to get much more efficient?

    passive = [0]*len(self.clause_features)
    passive_good = [0]*len(self.clause_features)

    learn_for_every = self.num_good_selections / HP.MAX_TRAINS_PER_TRACE
    learn_for_every_sum = 0.0
    learn_ord = 0
    for tag,idx,isGood in self.journal:
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

      # don't learn from every selection for traces with many-many of them (but go and learn at least once)
      if sum(passive_good): # can learn
        if (num_good_steps==0 or random.uniform(0.0, 1.0) < HP.MAX_TRAINS_PER_TRACE / self.num_good_selections): # is randomized
          # if learn_for_every_sum <= learn_ord: # a deterministic version of the randomized above
          learn_for_every_sum += learn_for_every

          passive_good_t = torch.tensor(passive_good,dtype=logits.dtype)
          passive_t = torch.tensor(passive,dtype=logits.dtype)

          masked_logits = logits[passive_t > 0.0]
          passive_good_t = passive_good_t[passive_t > 0.0]
          passive_t = passive_t[passive_t > 0.0]

          # print("masked_logits",masked_logits.shape)
          # print("passive_good_t",passive_good_t.shape)
          # print("passive_t",passive_t.shape)

          # manually computing log_softmax with multiplicities
          c = torch.max(masked_logits,dim=-1)[0] # the second part, which we ignore, is the argmax' idx
          exp_logits = torch.exp(masked_logits - c)
          # print("exp_logits.shape",exp_logits.shape)
          logsumexp = torch.log(torch.matmul(exp_logits,passive_t))
          lsm = masked_logits-c-logsumexp
          # print("lsm.shape",lsm.shape)
          # print("lsm",lsm)
          good_lsm = torch.matmul(lsm,passive_good_t)
          # print("good_lsm",good_lsm.shape)

          assert not torch.isnan(good_lsm).any(), "Got nan in good_lsm " + str(good_lsm)

          good_action_reward_loss += -good_lsm/sum(passive_good)
          num_good_steps += 1

        learn_ord += 1

      passive[idx] -= 1
      if isGood:
        passive_good[idx] -= 1

    assert num_good_steps, "The training example was still degenerate!"
    return good_action_reward_loss/num_good_steps
