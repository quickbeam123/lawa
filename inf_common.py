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

EVENT_ADD = 0
EVENT_REM = 1
EVENT_SEL = 2

def process_one(task):
  (prob,opts,ltd,res_idx) = task

  to_run = " ".join(["./run_lawa_vampire.sh",opts,prob])

  # print(to_run)

  output = subprocess.getoutput(to_run)

  if ltd:
    clauses = {}         # id -> (feature_vec)
    journal = []         # [event,id,time], where event is one of EVENT_ADD EVENT_SEL EVENT_REM,
                         # time only for EVENT_SEL
    proof_flas = set()

    warmup_time = None  # how long it took till first selection
    select_time = 0     # how long all the followup selections took together (strictly speaking, this is redundant)

    # just temporaries, here during the parsing
    num_sels = 0
    last_sel_idx = None

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
        journal.append([EVENT_ADD,id])
      elif line.startswith("t: "):
        spl = line.split()
        prev_act_cost = int(spl[1])
        if last_sel_idx is None:
          warmup_time = prev_act_cost
        else:
          assert len(journal[last_sel_idx]) == 2
          journal[last_sel_idx].append(prev_act_cost)
          select_time += prev_act_cost
      elif line.startswith("s: "):
        spl = line.split()
        id = int(spl[1])
        last_sel_idx = len(journal)
        journal.append([EVENT_SEL,id])
        num_sels += 1
      elif line.startswith("r: "):
        spl = line.split()
        id = int(spl[1])
        journal.append([EVENT_REM,id])
      elif line and line[0] in "123456789":
        spl = line.split(".")
        id = int(spl[0])
        proof_flas.add(id)

    if len(proof_flas) == 0:
      print("Proof not found for",to_run)

    # print(prob,"had total act cost",act_cost_sum)
    if last_sel_idx is not None:
      assert len(journal[last_sel_idx]) == 2 # still waiting for its time
      # so, let's formally satisfy it (pretending the last selection finished things off in 0 time):
      journal[last_sel_idx].append(0)

    # degenerate or broken
    if len(clauses) == 0 or num_sels == 0 or len(proof_flas) == 0:
      return (res_idx,prob,None)
    else:
      return (res_idx,prob,(clauses,journal,proof_flas,warmup_time,select_time))
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

def num_features():
  if HP.FEATURE_SUBSET == HP.FEATURES_LEN:
    return 6
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return 2
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return 8
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return 9
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return 12
  elif HP.FEATURE_SUBSET == HP.FEATURES_ORIGRICH:
    return 7
  else:
    return 0

def process_features(full_features : List[float]) -> List[float]:
  f = full_features
  if HP.FEATURE_SUBSET == HP.FEATURES_LEN:
    return f[0:6]
  elif HP.FEATURE_SUBSET == HP.FEATURES_AW:
    return [f[6],f[7]]
  elif HP.FEATURE_SUBSET == HP.FEATURES_PLAIN:
    return f[0:8]
  elif HP.FEATURE_SUBSET == HP.FEATURES_RICH:
    return f[0:9]
  elif HP.FEATURE_SUBSET == HP.FEATURES_ALL:
    return f
  elif HP.FEATURE_SUBSET == HP.FEATURES_ORIGRICH:
    return [f[0]+f[1]+f[2]+f[3]]+f[6:]
    '''
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
    '''
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

  clause_key = Embed(HP.CLAUSE_INTERAL_SIZE if HP.CLAUSE_EMBEDDER_LAYERS > 0 else num_features())
  parts.append(clause_key)

  return torch.nn.ModuleList(parts)

def export_model(parts_model_state_file_path,name):
  # we start from a fresh model and just load its state from a saved dict
  model = get_initial_model()
  model.load_state_dict(torch.load(parts_model_state_file_path))

  # eval mode and no gradient
  for part in model:
    part.eval()
    for param in part.parameters():
      param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    clause_vals : Dict[int, Tensor]
    clauses : Dict[int, int] # using it as a set

    def __init__(self,clause_embedder : torch.nn.Module,clause_key : torch.nn.Module):
      super().__init__()

      self.clause_embedder = clause_embedder
      self.clause_key = clause_key

      self.clause_vals = {}
      self.clauses = {}

    @torch.jit.export
    def regClause(self,id: int,features : Tuple[float, float, float, float, float, float, float, float, float, float, float, float]):
      # print("NN: Got",id,"with features",features)

      tFeatures : Tensor = torch.tensor(process_features(features))

      # print("NN: tFeatures",tFeatures)

      tInternal : Tensor = self.clause_embedder(tFeatures)
      val = torch.dot(self.clause_key.weight,tInternal)
      self.clause_vals[id] = val

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
      if temp == 0.0: # the greedy selection (argmax)
        max : Optional[float] = None
        candidates : List[int] = []
        for id in sorted(self.clauses.keys()):
          val = self.clause_vals[id].item()
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
          val = self.clause_vals[id].item()
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
      clause_key: torch.nn.Module,
      clauses,journal,proof_flas,warmup_time,select_time):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_key = clause_key           # the key for multiplying an embedding to get a clause logits
    self.clauses = clauses                 # id -> (feature_vec)
    self.journal = journal                 # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas           # set of the good ids
    self.warmup_time = warmup_time
    self.select_time = select_time

  def forward(self):
    # let's a get a big matrix of feature_vec's, one for each clause (id)
    clause_list = []
    id2idx = {}
    for i,(id,features) in enumerate(sorted(self.clauses.items())):
      id2idx[id] = i
      clause_list.append(torch.tensor(features))
    feature_vecs = torch.stack(clause_list)

    # print("forward-feature_vecs",feature_vecs)

    # in bulk for all the clauses
    embeddings = self.clause_embedder(feature_vecs)
    logits = torch.matmul(self.clause_key.weight,torch.t(embeddings))
    # print("forward-logits",logits)

    good_action_reward_loss = torch.zeros(1)
    num_good_steps = 0

    time_penalty_loss = torch.zeros(1)
    time_penalty_volume = 0

    entropy_loss = torch.zeros(1)
    num_steps = 0

    passive = set()
    for event in self.journal:
      event_tag = event[0]
      recorded_id = event[1]
      if event_tag == EVENT_ADD:
        passive.add(recorded_id)
      elif event_tag == EVENT_REM:
        passive.remove(recorded_id)
      else:
        assert event_tag == EVENT_SEL
        event_time = event[2]

        if len(passive) < 2: # there was no chosing, can't correct the action
          continue

        passive_list = sorted(passive)
        # print("forward-passive_list",passive_list)
        indices = torch.tensor([id2idx[id] for id in passive_list])
        # print("forward-indices",indices)
        sub_logits = logits[indices]
        # print("forward-sub_logits",sub_logits)

        lsm = torch.nn.functional.log_softmax(sub_logits,dim=0)

        if HP.LEARN_FROM_ALL_GOOD:
          good_idxs = []
          for i,id in enumerate(passive_list):
            if id in self.proof_flas:
              good_idxs.append(i)
          # print(good_idxs)
          if len(good_idxs):
            good_action_reward_loss += -sum(lsm[good_idxs])/len(good_idxs)
            num_good_steps += 1
        else:
          if recorded_id in self.proof_flas:
            cur_idx = passive_list.index(recorded_id)
            good_action_reward_loss += -lsm[cur_idx]
            num_good_steps += 1

        if HP.TIME_PENALTY_MIXING > 0.0:
          cur_idx = passive_list.index(recorded_id)
          time_penalty_loss += HP.TIME_PENALTY_MIXING*event_time*lsm[cur_idx]
          time_penalty_volume += event_time

        if HP.ENTROPY_COEF > 0.0:
          minus_entropy = torch.dot(torch.exp(lsm),lsm)
          if HP.ENTROPY_NORMALIZED:
            minus_entropy /= torch.log(len(lsm))
          entropy_loss += HP.ENTROPY_COEF*minus_entropy
          num_steps += 1

        passive.remove(recorded_id)

    return ((good_action_reward_loss,num_good_steps),(time_penalty_loss,time_penalty_volume),(entropy_loss,num_steps))
