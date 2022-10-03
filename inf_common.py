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
torch.set_num_interop_threads(1)

# print(torch.__config__.parallel_info())

from typing import Dict, List, Tuple, Set, Optional

from multiprocessing import Pool
import subprocess

import numpy as np

import sys, random, math

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
    
    act_cost_sum = 0
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
      elif line.startswith("t: "):
        spl = line.split()
        prev_act_cost = int(spl[1])
        journal.append((prev_act_cost,EVENT_TIM))
        act_cost_sum += prev_act_cost
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

    # print(prob,"had total act cost",act_cost_sum)

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
EVENT_TIM = 1
EVENT_SEL = 2
EVENT_REM = 3

def num_features():
  if HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return 2
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return 4
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return 6
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return 10
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW_PLUS:
    return 8
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW_PLUS_TIMES:
    return 17

def process_features(full_features : List[float]) -> List[float]:
  f = full_features
  if HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return [f[0],f[2]]
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return f[0:4]
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return f[0:6]
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return f
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW_PLUS:
    return [f[0],f[0]*f[0],math.sqrt(f[0]),math.log(1.0+f[0]),f[2],f[2]*f[2],math.sqrt(f[2]),math.log(1.0+f[2])]
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW_PLUS_TIMES:
    a = f[0]
    w = f[2]
    a2 = a*a
    w2 = w*w
    a_sqrt = math.sqrt(a)
    w_sqrt = math.sqrt(w)
    a_log = math.log(1.0+a)
    w_log = math.log(1.0+w)

    return [a,a2,a_sqrt,a_log,w,w2,w_sqrt,w_log,a*w,a*w_sqrt,a*w_log,a_sqrt*w,a_sqrt*w_sqrt,a_sqrt*w_log,a_log*w,a_log*w_sqrt,a_log*w_log]
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
  if HP.CLAUSE_EMBEDDER_LAYERS == 0:
    clause_embedder = torch.nn.Identity()
  else:
    layer_list = [torch.nn.Linear(num_features(),HP.CLAUSE_INTERAL_SIZE),torch.nn.ReLU()]
    for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
      layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,HP.CLAUSE_INTERAL_SIZE))
      layer_list.append(torch.nn.ReLU())
    clause_embedder = torch.nn.Sequential(*layer_list)

  parts = [clause_embedder]

  clause_keys = torch.nn.Embedding(HP.NUM_EFFECTIVE_QUEUES,HP.CLAUSE_INTERAL_SIZE if HP.CLAUSE_EMBEDDER_LAYERS > 0 else num_features())
  parts.append(clause_keys)

  return torch.nn.ModuleList(parts)

# this is destructive on the model modules (best load from checkpoint file first)
def export_model(model,name):

  # eval mode and no gradient
  for part in model:    
    part.eval()
    for param in part.parameters():
      param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    clause_vals : Dict[int, Tensor]
    clauses : Dict[int, int] # using it as a set

    def __init__(self,clause_embedder : torch.nn.Module,clause_keys : torch.nn.Module):
      super().__init__()

      self.clause_embedder = clause_embedder
      self.clause_keys = clause_keys

      self.clause_vals = {}
      self.clauses = {}

      self.time = 0
    
    @torch.jit.export
    def regClause(self,id: int,features : Tuple[float, float, float, float, float, float, float, float, float, float]):
      # print("NN: Got",id,"with features",features)

      tFeatures : Tensor = torch.tensor(process_features(features))

      # print("NN: tFeatures",tFeatures)

      tInternal : Tensor = self.clause_embedder(tFeatures)
      vals = torch.matmul(self.clause_keys.weight,tInternal)
      self.clause_vals[id] = vals
      
      #print("NN: Val:",val)

    @torch.jit.export
    def add(self,id: int):
      self.clauses[id] = 0 # whatever value

      # print("NN: Adding",id)
    
    @torch.jit.export
    def remove(self,id: int):
      del self.clauses[id]

      # print("NN: Removing",id)
    
    @torch.jit.export
    def popSelected(self, temp : float) -> int:
      queue_idx = self.time % HP.NUM_EFFECTIVE_QUEUES
      self.time += 1

      if temp == 0.0: # the greedy selection (argmax)
        max : Optional[float] = None
        candidates : List[int] = []
        for id in sorted(self.clauses.keys()):
          val = self.clause_vals[id][queue_idx].item()
          if max is None or val > max:
            candidates = [id]
            max = val
          elif val == max:
            candidates.append(id)

        # print("NN: Cadidates",candidates)
        id = candidates[torch.randint(0, len(candidates), (1,)).item()]
        
        # print("NN: picked",id)

      else: # softmax selection (taking temp into account)
        ids : List[int] = []
        vals : List[float] = []
        for id in sorted(self.clauses.keys()):
          val = self.clause_vals[id][queue_idx].item()
          ids.append(id)
          vals.append(val)

        # print("NN: Ids",ids)

        distrib = torch.nn.functional.softmax(torch.tensor(vals)/temp,dim=0)

        # print("NN: Distrib",distrib)

        idx = torch.multinomial(distrib,1).item()
        id = ids[idx]

        # print("NN: picked",id)

      del self.clauses[id]
      return id
  
  module = NeuralPassiveClauseContainer(*model)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      clause_keys: torch.nn.Module,
      clauses,journal,proof_flas):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_keys = clause_keys         # the keys (one for each "queue") for multiplying an embedding to get a clause logits 
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
    logits = torch.matmul(self.clause_keys.weight,torch.t(embeddings))
    
    # print(logits)

    loss = torch.zeros(1)
    factor = 1.0

    # print("proof_flas",self.proof_flas)

    time = 0
    steps = 0
    passive = set()
    for (recorded_id, event) in self.journal:
      if event == EVENT_TIM:
        pass
      elif event == EVENT_ADD:
        passive.add(recorded_id)
      elif event == EVENT_REM:
        passive.remove(recorded_id)
      else:
        assert event == EVENT_SEL

        queue_idx = time % HP.NUM_EFFECTIVE_QUEUES
        time += 1

        passive_list = sorted(passive)

        # print(passive_list)

        indices = torch.tensor([id2idx[id] for id in passive_list])

        # print(indices)

        sub_logits = logits[queue_idx][indices]

        # print(sub_logits)

        num_good = len(passive & self.proof_flas)
        num_bad =  len(passive) - num_good

        # it's a bit weird, but num_good can be zero (avatar closed proof with a different unsat core?)
        # in any case, while num_good == 0 means division by zero below,
        # it does not even seem to make much sense to learn wiht num_bad == 0 either (both are expected to be rare anyways)
        if num_good and num_bad:
          good_idxs = []
          for i,id in enumerate(passive_list):
            if id in self.proof_flas:
              good_idxs.append(i)

          # print(good_idxs)

          lsm = torch.nn.functional.log_softmax(sub_logits,dim=0)
          if HP.USE_MIN_FOR_LOSS_REDUCE:
            cross_entropy = -min(lsm[good_idxs])
          else:
            cross_entropy = -sum(lsm[good_idxs])/num_good

          loss += factor*(num_bad/(num_good+num_bad))*cross_entropy

          if HP.ENTROPY_COEF > 0.0:
            minus_entropy = torch.dot(torch.exp(lsm),lsm)
            if HP.ENTROPY_NORMALIZED:
              minus_entropy /= torch.log(len(lsm))
            loss += HP.ENTROPY_COEF*minus_entropy

          steps += 1
          factor *= HP.DISCOUNT_FACTOR
        else:
          # print("num_good and num_bad skip")
          pass
        
        passive.remove(recorded_id)

    return loss/steps if steps > 0 else None # normalized per problem (the else branch just returns the constant zero)
    
class PrincipledLearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      clause_keys: torch.nn.Module,
      clauses,journal,proof_flas):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_keys = clause_keys         # the keys (one for each "queue") for multiplying an embedding to get a clause logits 
    self.clauses = clauses                 # id -> (feature_vec)
    self.proof_flas = proof_flas           # set of the good ids

    # let's preprocess the journal, so that it's easier to assign the time penalties
    last_tim = 0 # by our convention (and it kind of makes sense)
    tim_sum = 0
    good_steps = 0
    new_journal = []
    for (value, event) in reversed(journal):
      if event == EVENT_TIM:
        last_tim = value
        tim_sum += value
        # dropping the event for new_journal
      elif event == EVENT_SEL:
        recorded_id = value
        new_journal.append(((recorded_id,last_tim),event))
        if recorded_id in proof_flas:
          good_steps += 1
        # sel takes the last_tim (when read backwards), so, actually, how much time was spent on immediate processing of this clause
      else:
        new_journal.append((value,event))

    new_journal.reverse()
    self.journal = new_journal  # (value,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM and value is recorded_id for EVENT_ADD and EVENT_REM and it's (recorded_id,last_tim) for EVENT_SEL
    self.tim_sum = tim_sum # for penalty normalization
    self.good_steps = good_steps

  def forward(self):
    tim_sum = self.tim_sum
    if tim_sum == 0:
      # to avoid division by zero (if sum is zero, the penalties are all zero too anyway)
      tim_sum = 1
    good_steps = self.good_steps

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
    logits = torch.matmul(self.clause_keys.weight,torch.t(embeddings))

    # print(logits)

    loss = torch.zeros(1)
    factor = 1.0

    # print("proof_flas",self.proof_flas)

    time = 0
    steps = 0
    passive = set()
    for (value, event) in self.journal:
      if event == EVENT_ADD:
        # print("EVENT_ADD",recorded_id)
        passive.add(value)
      elif event == EVENT_REM:
        # print("EVENT_REM",recorded_id)
        passive.remove(value)
      else:
        # print("EVENT_SEL",recorded_id)
        assert event == EVENT_SEL
        (recorded_id,last_tim) = value

        if len(passive) < 2: # there was no chosing, can't correct the action
          continue

        queue_idx = time % HP.NUM_EFFECTIVE_QUEUES
        time += 1

        passive_list = sorted(passive)

        # print(passive_list)

        indices = torch.tensor([id2idx[id] for id in passive_list])

        # print(indices)

        sub_logits = logits[queue_idx][indices]

        # print(sub_logits)

        cur_idx = passive_list.index(recorded_id)
        lsm = torch.nn.functional.log_softmax(sub_logits,dim=0)
        cross_entropy = -lsm[cur_idx]

        reward = factor*((1/good_steps if recorded_id in self.proof_flas else 0.0) - HP.TIME_PENALTY_MIXING*last_tim/tim_sum)
        factor *= HP.DISCOUNT_FACTOR # TODO: this is not actually what a DISCOUNT_FACTOR should be doing; consider fixing (and keep 1.0 until then)

        loss += reward*cross_entropy

        steps += 1 # just to know about pathological derivations where nothing got selected and we still proved it (can these actually happen?)

        # TODO: we are now entropy penelizing at every state - that's probably going to dominate everything - rething this!
        if HP.ENTROPY_COEF > 0.0:
          minus_entropy = torch.dot(torch.exp(lsm),lsm)
          if HP.ENTROPY_NORMALIZED:
            minus_entropy /= torch.log(len(lsm))
          loss += HP.ENTROPY_COEF*minus_entropy

        passive.remove(recorded_id)

    return loss if steps > 0 else None
