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
  proof_units: List[int,int]

  # gnn "modules"
  gnn_node_init: List[Tuple[str,torch.nn.modules.linear.Linear]]
  gnn_layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]]
  # gnn_clause_final: torch.nn.Module
  # gnn_symbol_final: torch.nn.Module

  # gnn records
  gnn_nodes: Dict[str,Tensor]
  gnn_edges: List[Tuple[str,str,Tensor]]
  gnn_init_clause_nums: list[int]

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

    # modules
    self.problem_embedder = problem_embedder
    self.clause_embedder = clause_embedder
    self.clause_valuator = clause_valuator

    # records
    # self.problem_features = None
    self.clause_simple_features = {} # filled up gradually, only when recording
    self.journal = [] # filled up gradually, only when recording
    # self.proof_units = None

    # modules-like
    self.gnn_node_init = gnn_node_init
    self.gnn_layers = gnn_layers
    self.gnn_clause_final = gnn_clause_final
    self.gnn_symbol_final = gnn_symbol_final

    # records
    self.gnn_nodes = {}
    self.gnn_edges = []

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
    # self.gweight_symbol_embeds
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
      self.problem_features = features

    if self.computing:
      with torch.no_grad():
        self.clause_embedder.bias.add_(self.problem_embedder(features))

  @torch.jit.export
  def gnn_node_kind(self,what: str,features: Tensor):
    self.gnn_nodes[what] = features

  @torch.jit.export
  def gnn_edge_kind(self,src: str, tgt: str, edge_index_there: Tensor, edge_index_back: Tensor):
    self.gnn_edges.append((src,tgt,edge_index_there))
    self.gnn_edges.append((tgt,src,edge_index_back))

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
      self.gnn_node_init = None
      self.gnn_layers = None
      self.gnn_clause_final = None
      self.gnn_symbol_final = None
      if not self.recording:
        self.gnn_nodes = None
        self.gnn_edges = None

  @torch.jit.export
  def journal(self, tag, cl_num):
    self.journal.append((tag, cl_num))

  @torch.jit.export
  def set_proof_units_and_clean_modules(self, proof_units: list[int]):
    self.proof_units = proof_units

    # clean all modules before saving
    self.problem_embedder = None
    self.clause_embedder = None
    self.clause_valuator = None

    self.gnn_node_init = None
    self.gnn_layers = None
    self.gnn_clause_final = None
    self.gnn_symbol_final = None

    self.gage_rule_embed = None
    self.gage_combine = None

    self.gage_embed_store = None
    self.gage_cl_layers = None
    self.gage_cur_base_layer = None
    self.gage_todo_layers = None

    # TODO: clean modules more!


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
      print("gage layers:",len(todos))
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
      layer_idx = 1+max(self.gweight_term_layers[a] for a in args)
    else:
      layer_idx = 0
    layer_idx = max(layer_idx,self.gweight_cur_base_layer)

    self.gweight_term_layers[id] = layer_idx

    eff_layer_idx = layer_idx-self.gweight_cur_base_layer
    if len(self.gweight_todo_layers) == eff_layer_idx:
      empty_todo_layer: list[Tuple[int,int,float,list[int]]] = []
      self.gweight_todo_layers.append(empty_todo_layer)
    self.gweight_todo_layers[eff_layer_idx].append((id,functor,sign,args))

  def gweight_enqueue_term(self,id: int, functor: int, sign: float, args: list[int]):
    if self.recording:
      self.gweight_terms.append((id,functor,sign,args))

    if self.computing:
      self.gweight_enqueue_one_term(id,functor,sign,args)

  def gweight_enqueue_clause(self,cl_num: int, lits: list[int]):
    if self.recording:
      self.gweight_clauses.append((cl_num,lits))

    if self.computing:
      self.gweight_clause_todo.append((cl_num,lits))

  def get_subterm_embed(self,id) -> Tensor:
    if id < 0:
      return self.gweight_var_embed[id % HP.GWEIGHT_NUM_VAR_EMBEDS]
    else:
      return self.gweight_term_embed_store[id]

  def gweight_embed_pending(self):
    # first like with gage does with clause, but here with terms
    for todos in self.gweight_todo_layers:
      print("gweight layers:",len(todos))

      # TODO: could maybe directly write to a giant tensor via slicing!
      functors = []
      signs = []
      first_args = []
      other_args = []
      for id,functor,sign,args in todos:
        functors.append(self.gweight_symbol_embeds[functor])
        signs.append(torch.tensor(sign))
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
    empty_todo_layers: list[list[Tuple[int,int,list[int]]]] = []
    self.gweight_todo_layers = empty_todo_layers

    # second, do the clauses part
    for j,(cl_num,lits) in enumerate(self.gweight_clause_todo):
      lit_embeds = torch.stack([self.gweight_term_embed_store[lit] for lit in lits])
      # TODO: try: avg over lits, max over lits, attention over lits, extra non-linearity level, ...
      self.gweight_clause_embeds[cl_num] = torch.sum(lit_embeds,dim=0)

    self.gweight_clause_todo = []

  @torch.jit.export
  def embed_pending(self):
    self.gage_embed_pending()
    self.gweight_embed_pending()

  @torch.jit.export
  def eval_clauses(self, clause_nums: list[int], clause_features: Tensor):
    if self.recording:
      for i,cl_num in enumerate(clause_nums):
        self.clause_simple_features[cl_num] = clause_features[i]

    # if self.computing:
    gage_features = torch.stack([self.gage_embed_store[cl_num] for cl_num in clause_nums])
    gweight_features = torch.stack([self.gweight_clause_embeds[cl_num] for cl_num in clause_nums])
    all_features = torch.cat((clause_features, gage_features, gweight_features), dim=1)

    # assumes problems features are already hardwired into clause_embedder's bias
    return self.clause_valuator(self.clause_embedder(all_features))


# An "empty version" of the GNN interface, which can be used to store the graph shape and pass it for training!"
class GnnStore(torch.nn.Module):
  nodes: Dict[str,Tensor]
  edges: List[Tuple[str,str,Tensor]]

  def __init__(self):
    super().__init__()

    self.nodes = {}
    self.edges = []

  @torch.jit.export
  def node_kind(self,what: str,features: Tensor):
    self.nodes[what] = features

  @torch.jit.export
  def edge_kind(self,src: str, tgt: str, edge_index_there: Tensor, edge_index_back: Tensor):
    self.edges.append((src,tgt,edge_index_there))
    self.edges.append((tgt,src,edge_index_back))

# see: https://discuss.pytorch.org/t/using-torschscript-to-save-a-model-with-multiple-heads/158709
"""
@torch.jit.interface
class LinearInterface(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
      pass
"""

# A version of Gnn which can compute forward pass
class GnnCompute(torch.nn.Module):
  nodes: Dict[str,Tensor]
  edges: List[Tuple[str,str,Tensor]]

  node_init: List[Tuple[str,torch.nn.modules.linear.Linear]]
  layers: List[List[Tuple[str,str,int,torch_geometric.nn.SAGEConv]]]

  # clause_final: torch.nn.Module
  # symbol_final: torch.nn.Module

  def __init__(self, node_init, layers, clause_final, symbol_final, modules = None):
    super().__init__()

    self.nodes = {}
    self.edges = []

    # This is crazy, but while we don't need this for any computation, things don't jit.script witout it!
    # (maybe its necessary so that the annotations above can be digested?)
    self.dummy = torch_geometric.nn.SAGEConv((1,1),1)

    self.node_init = node_init
    self.layers = layers
    self.clause_final = clause_final
    self.symbol_final = symbol_final

    # pass in modules for training, so that they become official Params
    if modules:
      self.modules = torch.nn.ModuleList(modules)

  @torch.jit.export
  def node_kind(self,what: str,features: Tensor):
    self.nodes[what] = features

  @torch.jit.export
  def edge_kind(self,src: str, tgt: str, edge_index_there: Tensor, edge_index_back: Tensor):
    self.edges.append((src,tgt,edge_index_there))
    self.edges.append((tgt,src,edge_index_back))

  @torch.jit.export
  def gnn_perform(self, clause_nums: list[int]):
    if self.recording:
      # just save clause_nums
      pass

    if self.computing:
      # the clause numbers in clause_nums are promised to go in the same order as the clauses in previously added via gnnNodeKind("clause",...)
      for key,embedder in self.node_init:
        self.nodes[key] = embedder.forward(self.nodes[key]).relu()

      for layer in self.layers:
        out_dict: Dict[str, Tensor] = {}
        for src,tgt,i,conv in layer:
          out = conv.forward((self.nodes[src],self.nodes[tgt]),self.edges[i][2])
          if tgt in out_dict:
            out_dict[tgt] = out_dict[tgt] + out
          else:
            out_dict[tgt] = out

        for key, out in out_dict.items():
          self.nodes[key] = out.relu()
          out_dict = {}

      # TODO: in the future could also pool things and extract a problem embedding to use

      # TODO: can drop all the stuff not needed anymore

      # TODO: pass on the age-style clause embeddings to gen-age part (using clause_nums)

      store_me1 = self.clause_final.forward(self.nodes["clause"])
      store_me2 = self.symbol_final.forward(self.nodes["symbol"])


GNN_NUM_LAYERS = 10
GNN_INTERNAL_SIZE = 16

RvNN_AGE_INTERNAL_SIZE = 16
RvNN_WEIGHT_INTERNAL_SIZE = 16

def get_conv():
  return torch_geometric.nn.SAGEConv(
      (GNN_INTERNAL_SIZE,GNN_INTERNAL_SIZE),
      GNN_INTERNAL_SIZE,
      aggr="mean",      # TODO: consider trying out SUM/MAX
      normalize=False,  # a bit like a layernorm on the output?
      root_weight=True, # like a self-loop; i.e. allow then target node to talk as well
      project=False,    # extra non-lineary before aggregating
      bias=True)        # and why not add a bias before the relu that's about to come?

# this fuction is bogus, we will need a real model, where all the submodules are parameters
# and from which we will create the inference module by "stealing the weights"
def get_gnn_compute(model: GnnStore, out_file_name):
  # we only use the store to read off the "signature"

  node_init = [("sort",  torch.nn.Linear(3,GNN_INTERNAL_SIZE)),
               ("symbol",  torch.nn.Linear(10,GNN_INTERNAL_SIZE)),
               ("clause", torch.nn.Linear(10,GNN_INTERNAL_SIZE)),
               ("term", torch.nn.Linear(10,GNN_INTERNAL_SIZE)),
               ("var", torch.nn.Linear(1,GNN_INTERNAL_SIZE)),] # TODO: discretize to have only a few embeddings? but non-linearly spread?

  clause_final = torch.nn.Linear(GNN_INTERNAL_SIZE,RvNN_AGE_INTERNAL_SIZE)
  symbol_final = torch.nn.Linear(GNN_INTERNAL_SIZE,RvNN_WEIGHT_INTERNAL_SIZE)

  modules = [embed for node_str,embed in node_init] + [clause_final,symbol_final]

  layers = []
  for i in range(GNN_NUM_LAYERS):
    layer = []
    for i,(src,tgt,_) in enumerate(model.edges):
      # in the last layer, no need for any other output than ["symbol","clause"]
      if i != GNN_NUM_LAYERS-1 or tgt in ["symbol","clause"]:
        conv = get_conv()
        modules.append(conv)
        layer.append((src,tgt,i,conv))
    layers.append(layer)

  # pass in modules as the last argument if you want to also train with GnnCompute
  model_to_script = GnnCompute(node_init,layers,clause_final,symbol_final)

  # script_model.nodes = model.nodes
  # script_model.edges = model.edges
  # res =  script_model.forward()
  # print(res[0].shape)
  # print(res[1].shape)

  print(model_to_script.state_dict())

  return

  model_to_script.eval()
  script = torch.jit.script(model_to_script)
  print(script)
  print(script.code)
  script.save(out_file_name)


class GenAgeNN(torch.nn.Module):
  inits: list[int]
  infers: list[Tuple[int,int,list[int]]]

  embed_store: Dict[int,Tensor]
  cl_layers: Dict[int,int]
  todo_layers: list[list[Tuple[int,int,list[int]]]]

  def __init__(self, rule_embed, combine):
    super().__init__()

    self.recording = False
    # needed for recording:
    self.inits = []
    self.infers = []

    self.computing = False
    # needed for computing:
    self.embed_store = {}
    self.cl_layers = {}
    self.cur_base_layer = 1
    self.todo_layers = []

    self.rule_embed = rule_embed
    self.combine = combine

  # seed the storage with initial embeddings (for the input clauses)
  @torch.jit.export
  def init_embed(self,clNum: int, embedding: Tensor):
    if self.recording:
      self.inits.append(clNum)

    if self.computing:
      self.embed_store[clNum] = embedding
      self.cl_layers[clNum] = 0

  def enqueue_one(self,clNum: int, infRule: int, parents: list[int]):
    layer_idx = max(1+max(self.cl_layers[p] for p in parents),self.cur_base_layer)
    # index (counting from 0 with the initials) where clNum could (and will) be derived
    self.cl_layers[clNum] = layer_idx

    eff_layer_idx = layer_idx-self.cur_base_layer
    if len(self.todo_layers) == eff_layer_idx:
      empty_todo_layer: list[Tuple[int,int,list[int]]] = []
      self.todo_layers.append(empty_todo_layer)
    self.todo_layers[eff_layer_idx].append((clNum,infRule,parents))

  def embed_enqueued(self):
    for todos in self.todo_layers:
      print(len(todos))
      # creating an input to the bulk
      ruleIdxs: list[int] = [] # into rule_embed
      mainPrems = []
      otherPrems = []
      for clNum,infRule,parents in todos:
        ruleIdxs.append(infRule)
        mainPrems.append(self.embed_store[parents[0]])
        if len(parents) == 1:
          otherPrems.append(torch.zeros(HP.GAGE_EMBEDDING_SIZE))
        elif len(parents) == 2:
          otherPrems.append(self.embed_store[parents[1]])
        else:
          # this would work even in the binary case, but let's not invoke the moster if we don't need to
          otherPrem = torch.sum(torch.stack(self.embed_store[parents[p]] for p in parents[1:]),dim=0)/(len(parents)-1)
          otherPrems.append(otherPrem)
      ruleEbeds = self.rule_embed(torch.tensor(ruleIdxs))
      mainPremEbeds = torch.stack(mainPrems)
      otherPremEbeds = torch.stack(otherPrems)
      res = self.combine(torch.cat((ruleEbeds, mainPremEbeds, otherPremEbeds), dim=1))
      for j,(clNum,_,_) in enumerate(todos):
        self.embed_store[clNum] = res[j]

    self.cur_base_layer += len(self.todo_layers)
    empty_todo_layers: list[list[Tuple[int,int,list[int]]]] = []
    self.todo_layers = empty_todo_layers


  # a request to compute an embedding for clNum, derived by infRule from parents
  # all parents have either been mentioned via init_embed or previes calls to this method
  @torch.jit.export
  def enqueue(self, clNum: int, infRule: int, parents: list[int]):
    if self.recording:
      self.infers.append((clNum,infRule,parents))

    if self.computing:
      self.enqueue_one(clNum,infRule,parents)

  @torch.jit.export
  def embed_pending(self):
    if self.computing:
      self.embed_enqueued()

  @torch.jit.export
  def lookup(self,clNum: int) -> Tensor:
    return self.embed_store[clNum]

  # if we just save all this: what is it, that should be remembered for learning?
  # - not the init_embeds themselves (they need to be chained explicitly, so that gradients can propagate!), but their clNums, yes
  # - all enqueue requests, but again, not necessarily the work they represent

def get_hollow_genAgeNN():
  model = GenAgeNN(torch.nn.Embedding(1,1),torch.nn.Identity())
  model.recording = True
  script = torch.jit.script(model)
  return script

def get_full_genAgeNN():
  rule_embed = torch.nn.Embedding(num_embeddings=HP.NUM_INFERENCE_RULES, embedding_dim=HP.GAGE_EMBEDDING_SIZE)
  combine = torch.nn.Sequential(
     # TODO: experiment with dropout?
     torch.nn.Linear(3*HP.GAGE_EMBEDDING_SIZE,HP.INTERAL_SIZE),
     torch.nn.ReLU(),
     torch.nn.Linear(HP.INTERAL_SIZE,HP.GAGE_EMBEDDING_SIZE),
     torch.nn.LayerNorm(HP.GAGE_EMBEDDING_SIZE)
  )
  model = GenAgeNN(rule_embed,combine)
  model.computing = True
  script = torch.jit.script(model)
  return script


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
