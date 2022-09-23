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

  if HP.LEARNER == HP.LEARNER_RECURRENT:
    assert HP.CLAUSE_EMBEDDER_LAYERS > 0

    initial_key = Embed(HP.CLAUSE_INTERAL_SIZE)
    initial_state = Embed(HP.CLAUSE_INTERAL_SIZE)
    rnn = torch.nn.LSTM(HP.CLAUSE_INTERAL_SIZE, HP.CLAUSE_INTERAL_SIZE, HP.LSMT_LAYERS)

    parts.append(initial_key)
    parts.append(initial_state)
    parts.append(rnn)
  else:
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

  class NeuralRecurrentPassiveClauseContainer(torch.nn.Module):
    clause_embeddings : Dict[int, Tensor]
    clauses : Dict[int, int] # using it as a set

    def __init__(self,clause_embedder : torch.nn.Module,
                      initial_key : torch.nn.Module,
                      initial_state : torch.nn.Module,
                      rnn : torch.nn.Module):
      super().__init__()

      self.clause_embedder = clause_embedder
      # transpose these two vectors (for better processing by rnn)
      self.current_key = torch.unsqueeze(initial_key.weight,dim=0)
      self.current_state = torch.unsqueeze(initial_state.weight,dim=0)
      self.rnn = rnn

      self.clause_embeddings = {}
      self.clauses = {}
    
    @torch.jit.export
    def regClause(self,id: int,features : Tuple[float, float, float, float, float, float, float, float, float, float]):
      # print("NN: Got",id,"with features",features)

      tFeatures : Tensor = torch.tensor(process_features(features))
      tEmbed = self.clause_embedder(tFeatures)
      self.clause_embeddings[id] = tEmbed
      
      #print("NN: Embedded:",tEmbed)

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
      key = self.current_key

      if temp == 0.0: # the greedy selection (argmax)
        max : Optional[float] = None
        candidates : List[int] = []
        for id in sorted(self.clauses.keys()):
          val = torch.matmul(key,self.clause_embeddings[id]).item()
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
          val = torch.matmul(key,self.clause_embeddings[id]).item()
          ids.append(id)
          vals.append(val)

        # print("NN: Ids",ids)

        distrib = torch.nn.functional.softmax(torch.tensor(vals)/temp,dim=0)

        # print("NN: Distrib",distrib)

        idx = torch.multinomial(distrib,1).item()
        id = ids[idx]

        # print("NN: picked",id)

      # update the rnn
      input = torch.unsqueeze(self.clause_embeddings[id],dim=0)

      # output can be ignored, it's going to be a singleton with the new key
      _,(self.current_key,self.current_state) = self.rnn(input,(self.current_key,self.current_state))

      del self.clauses[id]
      return id

  if HP.LEARNER == HP.LEARNER_RECURRENT:
    module = NeuralRecurrentPassiveClauseContainer(*model)
  else:
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
    
class RecurrentLearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      initial_key : torch.nn.Module,
      initial_state : torch.nn.Module,
      rnn : torch.nn.Module,
      clauses,journal,proof_flas):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.initial_key = initial_key         # the key for multiplying the clauses at first selection call (and feeding the rnn as the initial hidden value h0)
    self.initial_state = initial_state     # the initial content value for the lsmt (c0)
    self.rnn = rnn                         # our lstm
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

    current_key = torch.unsqueeze(self.initial_key.weight,dim=0)
    current_state = torch.unsqueeze(self.initial_state.weight,dim=0)

    loss = torch.zeros(1)
    factor = 1.0

    # print("proof_flas",self.proof_flas)

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
        # we ignore what the selection was by the agent
        # and instead assert the (multi-choice) ground truth

        passive_list = sorted(passive)

        # print(passive_list)

        indices = torch.tensor([id2idx[id] for id in passive_list])
        sub_embeddings = torch.index_select(embeddings,0,indices)
        sub_logits = torch.matmul(sub_embeddings,torch.squeeze((current_key),dim=0))

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

        # update the rnn
        input = torch.unsqueeze(embeddings[id2idx[recorded_id]],dim=0)
        # output can be ignored, it's going to be a singleton with the new key
        _,(current_key,current_state) = self.rnn(input,(current_key,current_state))

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

        if recorded_id in self.proof_flas: # this was a good move, let's reward it
          good_idx = passive_list.index(recorded_id)

          lsm = torch.nn.functional.log_softmax(sub_logits,dim=0)
          cross_entropy = -lsm[good_idx]

          loss += factor*cross_entropy

          # TODO: we could entropy-penalize at every step, but then step+=1 should go out of the if too
          if HP.ENTROPY_COEF > 0.0:
            minus_entropy = torch.dot(torch.exp(lsm),lsm)
            if HP.ENTROPY_NORMALIZED:
              minus_entropy /= torch.log(len(lsm))
            loss += HP.ENTROPY_COEF*minus_entropy

          steps += 1
          factor *= HP.DISCOUNT_FACTOR
        else:
          # TODO: penalty for wrong selections
          pass

        passive.remove(recorded_id)

    return loss/steps if steps > 0 else None # normalized per problem (the else branch just returns the constant zero)

class EnigmaLearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      clause_keys: torch.nn.Module,
      clauses,journal,proof_flas):
    super().__init__()

    # does not make sense otherwise, since we don't track time
    assert HP.NUM_EFFECTIVE_QUEUES == 1

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_keys = clause_keys         # the keys (one for each "queue") for multiplying an embedding to get a clause logits 
    self.clauses = clauses                 # id -> (feature_vec)
    self.journal = journal                 # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas           # set of the good ids

  def forward(self):
    if HP.ENIGMA_JUST_SELECTED:
      selected = {id for (id, event) in self.journal if event == EVENT_SEL}
      filter = lambda x : x in selected
    else:
      filter = lambda x : True

    # let's a get a big matrix of feature_vec's, one for each clause (id)
    pos = 0
    neg = 0
    clause_list = []
    label_list = []
    good_idxs = []
    
    i = 0
    for (id,features) in sorted(self.clauses.items()):
      if filter(id):
        clause_list.append(torch.tensor(features))
        if id in self.proof_flas:
          pos += 1
          label_list.append(1.0)
          good_idxs.append(i)
        else:
          neg += 1
          label_list.append(0.0)
        i += 1

    feature_vecs = torch.stack(clause_list)
    targets = torch.tensor(label_list)

    # in bulk for all the clauses
    embeddings = self.clause_embedder(feature_vecs)
    # get the final logits
    logits = torch.matmul(self.clause_keys.weight,torch.t(embeddings))[0] # the [0] stands for the selection of the effective queue, of which there is always only one, the 0-th

    if HP.ENIGMA_BINARY_CLASSIF:
      if pos*neg > 0:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos]),reduction='sum')
        loss = criterion(logits,targets)
        loss /= 2*neg # each pos (there is pos many of those) counts as neg/pos and each neg (there is neg many) counts as one, that's 2*neg total
      else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(logits,targets)
    else:
      lsm = torch.nn.functional.log_softmax(logits,dim=0)
      loss = -sum(lsm[good_idxs])/pos

    return loss
