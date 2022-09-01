#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from torch import Tensor

torch.set_num_threads(1)

from typing import Dict, List, Tuple, Set, Optional

from multiprocessing import Pool
import subprocess

import numpy as np

import sys, random

import hyperparams as HP

def process_one(task):
  (prob,opts,ltd,res_idx) = task

  to_run = " ".join(["./run_lawa_vampire.sh",prob,opts])

  # print(to_run)

  output = subprocess.getoutput(to_run)

  if ltd:
    clauses = {} # id -> (feature_vec)
    journal = [] # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    proof_flas = set()

    num_sels = 0

    for line in output.split("\n"):
      if "% Instruction limit reached!" in line:
        assert not proof_flas      
        break # better than "id appeared again for:" failing just below
      # print(line)
      if line.startswith("i: "):
        spl = line.split()
        id = int(spl[1])
        features = process_features(list(map(float,spl[2:])))
        assert len(features) == num_features()
        # print(id,features)
        assert id not in clauses, "id appeared again for: "+to_run
        clauses[id] = features
      elif line.startswith("a: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_ADD))
      elif line.startswith("s: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_SEL))
        num_sels += 1
      elif line.startswith("r: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_REM))
      elif line and line[0] in "123456789":
        spl = line.split(".")
        id = int(spl[0])
        proof_flas.add(id)

    if len(proof_flas) == 0:
      print("Proof not found for",to_run)

    if len(clauses) == 0 or num_sels == 0 or len(proof_flas) == 0:
      return (res_idx,prob,None)
    else:
      return (res_idx,prob,(clauses,journal,proof_flas)) 
  else:
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

    return (res_idx,prob,(status,instructions,activations))

class Evaluator:
  def __init__(self, numcpu : int):
    self.pool = Pool(processes = numcpu)

  def close(self):
    self.pool.close()

  # gets a list of "jobs", each being a triple of 
  # (list_of_problems, vampire_options, load_traing_data : Bool)
  # returns a list of the same lenght of result dicts
  # either 1) as done by scan_and_store scripts - for load_traing_data False
  # or     2) like load_one - for load_traing_data True
  def perform(self,jobs,parallelly=True):
    tasks = []    
    for i,(probs,opts,ltd) in enumerate(jobs):
      for prob in probs:
        tasks.append((prob,opts,ltd,i))

    if parallelly:
      computed = self.pool.map(process_one, tasks, chunksize = 1)
    else:
      computed = []
      for task in tasks:
        computed.append(process_one(task))

    results = [{} for _ in jobs]
    for (res_idx,prob,data) in computed:
      results[res_idx][prob] = data

    return results

EVENT_ADD = 0
EVENT_SEL = 1
EVENT_REM = 2

def num_features():
  if HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return 2
  elif HP.FEATURE_SUBSET == HP.FEATURES_BAW:
    return 3
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return 4
  elif HP.FEATURE_SUBSET == HP.FEATURES_BPLAIN:
    return 5
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return 6
  elif HP.FEATURE_SUBSET == HP.FEATURES_BRICH:
    return 7
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return 10
  elif HP.FEATURE_SUBSET == HP.FEATURES_BALL:
    return 11

def process_features(full_features : List[float]) -> List[float]:
  f = full_features
  if HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return [f[0],f[2]]
  elif HP.FEATURE_SUBSET == HP.FEATURES_BAW:
    return [1.0,f[0],f[2]]
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return f[0:4]
  elif HP.FEATURE_SUBSET == HP.FEATURES_BPLAIN:
    return [1.0]+f[0:4]
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return f[0:6]
  elif HP.FEATURE_SUBSET == HP.FEATURES_BRICH:
    return [1.0]+f[0:6]
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return f
  elif HP.FEATURE_SUBSET == HP.FEATURES_BALL:
    return [1.0]+f
  else:
    assert False
    return []

class Embed(torch.nn.Module):
  weight: Tensor
  
  def __init__(self, dim : int):
    super().__init__()
    
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(dim))
    self.reset_parameters()
  
  def __repr__(self):
    return "Embed: "+str(self.weight.data)

  def reset_parameters(self):
    torch.nn.init.normal_(self.weight)

  def forward(self) -> Tensor:
    return self.weight

def get_initial_model():
  if HP.NUM_LAYERS == 0:
    clause_embedder = torch.nn.Identity()
  else:
    layer_list = [torch.nn.Linear(num_features(),HP.CLAUSE_INTERAL_SIZE),torch.nn.ReLU()]
    for _ in range(HP.NUM_LAYERS-1):
      layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,HP.CLAUSE_INTERAL_SIZE))
      layer_list.append(torch.nn.ReLU())
    clause_embedder = torch.nn.Sequential(*layer_list)
  
  clause_key = Embed(HP.CLAUSE_INTERAL_SIZE if HP.NUM_LAYERS > 0 else num_features() )

  return torch.nn.ModuleList([clause_embedder,clause_key])

# this is destructive on the model modules (best load from checkpoint file first)
def export_model(model,name):

  # eval mode and no gradient
  for part in model:    
    part.eval()
    for param in part.parameters():
      param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    clause_vals : Dict[int, float]
    clauses : Dict[int, int] # using it as a set

    def __init__(self,clause_embedder : torch.nn.Module,clause_key : torch.nn.Module):
      super().__init__()

      self.clause_embedder = clause_embedder
      self.clause_key = clause_key

      self.clause_vals = {}
      self.clauses = {}
    
    @torch.jit.export
    def regClause(self,id: int,features : Tuple[float, float, float, float, float, float, float, float, float, float]):
      tFeatures : Tensor = torch.tensor(process_features(features))
      tInternal : Tensor = self.clause_embedder(tFeatures)
      val = torch.dot(tInternal,self.clause_key.weight).item()
      self.clause_vals[id] = val

      # print("Got",id,"of val",val)

    @torch.jit.export
    def add(self,id: int):
      self.clauses[id] = 0 # whatever value

      # print("Adding",id)
    
    @torch.jit.export
    def remove(self,id: int):
      del self.clauses[id]

      # print("Removing",id)
    
    @torch.jit.export
    def popSelected(self, temp : float) -> int:
      if temp == 0.0: # the greedy selection (argmax)
        min : Optional[float] = None
        candidates : List[int] = []
        for id in sorted(self.clauses.keys()):
          val = self.clause_vals[id]
          if min is None or val < min:
            candidates = [id]
            min = val
          elif val == min:
            candidates.append(id)
        id = candidates[torch.randint(0, len(candidates), (1,)).item()]
      else: # softmax selection (taking temp into account)
        ids : List[int] = []
        vals : List[float] = []
        for id in sorted(self.clauses.keys()):
          val = self.clause_vals[id]
          ids.append(id)
          vals.append(val)

        distrib = torch.nn.functional.softmax(torch.tensor(vals)/temp,dim=0)
        idx = torch.multinomial(distrib,1).item()
        id = ids[idx]
      
      del self.clauses[id]
      return id

  module = NeuralPassiveClauseContainer(*model)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      clause_key: torch.nn.Module,      
      clauses,journal,proof_flas):
    super(LearningModel,self).__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_key = clause_key           # the key for multiplying an embedding to get a clause logits    
    self.clauses = clauses                 # id -> (feature_vec)
    self.journal = journal                 # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas           # set of the good ids

  def forward(self):
    # let's a get a big matrix of feature_vec's, one for each clause (id)
    clause_list = []
    id2idx = {}
    for i,(id,features) in enumerate(sorted(self.clauses.items())):
      id2idx[id] = i
      clause_list.append(torch.tensor(features))
    feature_vecs = torch.stack(clause_list)

    # in bulk for all the clauses
    embeddings = self.clause_embedder(feature_vecs)
    # since we are not doing LSTM (yet), the multiplication by key can happen emmediately as well
    # (and is probably kind of redundant in the architecture)
    logits = torch.matmul(embeddings,self.clause_key.weight)
    
    loss = torch.zeros(1)
    criterion = torch.nn.CrossEntropyLoss()

    steps = 0
    passive = set()
    for (id, event) in self.journal:
      if event == EVENT_ADD:
        passive.add(id)
      elif event == EVENT_REM:
        passive.remove(id)
      else:
        assert event == EVENT_SEL
        # we ignore what the selection was by the agent 
        # and instead assert the (multi-choice) ground truth

        passive_list = sorted(passive)

        # print(passive_list)

        indices = torch.tensor([id2idx[id] for id in passive_list])

        # print(indices)

        sub_logits = torch.index_select(logits,0,indices)

        # print(sub_logits)

        pst = 1.0/len(passive & self.proof_flas)
        targets = torch.tensor([pst if id in self.proof_flas else 0.0 for id in passive_list])

        # print(targets)

        loss += criterion(sub_logits,targets)
        steps += 1

    return loss/steps if steps > 0 else loss # normalized per problem (the else branch just returns the constant zero)
    

















# OLD STUFF BELOW HERE

# basically, a functionality similar to "scan_results_..." except it's home-grown here
def scan_result_folder(foldername):
  results = {} # probname -> (result, time, mega_instr, activations)

  root, dirs, files = next(os.walk(foldername)) # just take the log files immediately under solver_folder
  for filename in files:    
    longname = os.path.join(root, filename)

    if filename == "meta.info":
      continue

    assert filename.endswith(".p.log")
    assert filename.startswith("Problems_")

    probname = filename[13:-6]

    with open(longname, "r") as f:
      status = None
      time = None
      instructions = 0
      activations = 0

      for line in f:
        # vampiric part:
        if (line.startswith("% SZS status Timeout for") or line.startswith("% SZS status GaveUp") or
            line.startswith("% Time limit reached!") or line.startswith("% Refutation not found, incomplete strategy") or
            "% Instruction limit reached!" in line or 
            line.startswith("% Refutation not found, non-redundant clauses discarded") or
            line.startswith("Unsupported amount of allocated memory:") or 
            line.startswith("Memory limit exceeded!")):          
          status = "---"
    
        if line.startswith("% SZS status Unsatisfiable") or line.startswith("% SZS status Theorem") or line.startswith("% SZS status ContradictoryAxioms"):
          status = "uns"

        if line.startswith("% SZS status Satisfiable") or line.startswith("% SZS status CounterSatisfiable"):
          status = "sat"
    
        if line.startswith("Parsing Error on line"):
          status = "err"

        if line.startswith("% Time elapsed:"):
          time = float(line.split()[-2])
                              
        if line.startswith("% Instructions burned:"):
          # "% Instructions burned: 361 (million)"
          instructions = int(line.split()[-2])
    
        if line.startswith("% Activations started:"):
          activations = int(line.split()[-1])
  
      assert probname not in results
      results[probname] = (status,time,instructions,activations)

      # print(probname,status,time,instructions,activations)

      if status is None:
        print("Weird",longname)

  return results

def load_one(filename):
  clauses = {} # id -> (feature_vec)
  journal = [] # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
  proof_flas = set()

  num_sels = 0

  with open(filename,"r") as f:
    for line in f:      
      if line.startswith("i: "):
        spl = line.split()
        id = int(spl[1])
        features = process_features(list(map(float,spl[2:])))
        assert len(features) == num_features()
        # print(id,features)
        assert id not in clauses
        clauses[id] = features
      elif line.startswith("a: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_ADD))
      elif line.startswith("s: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_SEL))
        num_sels += 1
      elif line.startswith("r: "):
        spl = line.split()
        id = int(spl[1])
        journal.append((id,EVENT_REM))
      elif line[0] in "123456789":
        spl = line.split(".")
        id = int(spl[0])
        proof_flas.add(id)
    
  # print(clauses,journal,proof_flas)
  if len(clauses) == 0 or num_sels == 0:
    return None
  else:
    return (filename,clauses,journal,proof_flas)

