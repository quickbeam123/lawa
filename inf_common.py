#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

import torch
from torch import Tensor

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

def vampire_eval(prob,opts):
  to_run = " ".join(["./run_lawa_vampire.sh",opts,prob])
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

  return (status,instructions,activations)

def vampire_gather(prob,opts):
  to_run = " ".join(["./run_lawa_vampire.sh",opts,prob])
  # print(to_run)
  for _ in range(5): # sometimes, there are weird failures, but a restart could help!
    output = subprocess.getoutput(to_run)

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

    # print(prob,"had total act cost",act_cost_sum)
    if last_sel_idx is not None:
      assert len(journal[last_sel_idx]) == 2 # still waiting for its time
      # so, let's formally satisfy it (pretending the last selection finished things off in 0 time):
      journal[last_sel_idx].append(0)

    # a success
    # in the first coordinate, however, we still say whether there is non-trivial stuff to learn from
    return (len(clauses) != 0 and num_sels != 0,(clauses,journal,proof_flas,warmup_time,select_time))

  print("Repeatedly failing:",to_run)
  print(output)
  return None # meaning: "A major failure, consider keeping the used model for later debugging"

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
  elif HP.FEATURE_SUBSET == HP.FEATURES_ORIGPLAIN:
    return 4
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
  elif HP.FEATURE_SUBSET == HP.FEATURES_ORIGPLAIN:
    return [f[0]+f[1]+f[2]+f[3]]+f[6:9]
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

class Tweak(torch.nn.Module):
  weight: Tensor

  def __init__(self, dim1 : int, dim2 : int, dim3: int):
    super().__init__()

    self.dim1 = dim1
    self.dim2 = dim2
    self.dim3 = dim3

    self.weight = torch.nn.parameter.Parameter(torch.Tensor(dim1+dim2+dim3))
    self.reset_parameters()

  def __repr__(self):
    return "Tweak: "+str(self.weight.data)

  def reset_parameters(self):
    # the GSD_INPUT_MUL and GSD_INPUT_ADD should be zeros
    torch.nn.init.constant_(self.weight, 0.0)
    if self.dim3 > 0:
      # the GSD_FINAL_BLENDERS ones should be ones
      with torch.no_grad():
        offset = self.dim1+self.dim2
        for i in range(self.dim3):
          self.weight[offset+i] = 1.0

    # print(self.weight)

  def forward(self) -> Tensor:
    return self.weight

def get_default_tweak():
  return Tweak(HP.GSD_INPUT_MUL,HP.GSD_INPUT_ADD,HP.GSD_FINAL_BLENDERS)

def nice_module_name(str):
  # for some reasons, "." are not allowed as keys in module dicts
  return str.replace(".","_")

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

def get_initial_model(prob_list):
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

  tw_mul = torch.nn.Linear(HP.GSD_INPUT_MUL,num_features())
  tw_add = torch.nn.Linear(HP.GSD_INPUT_ADD,num_features())
  tweaks = torch.nn.ModuleDict({nice_module_name(prob) : get_default_tweak() for prob in prob_list})

  parts += [tw_mul,tw_add,tweaks]

  return torch.nn.ModuleList(parts)

def export_model(model_state_dict,name,prob_list):
  # we start from a fresh model and just load its state from a saved dict
  model = get_initial_model(prob_list)
  model.load_state_dict(model_state_dict)

  # we drop the learned tweaks
  model = model[:-1]

  # eval mode and no gradient
  for part in model:
    part.eval()
    for param in part.parameters():
      param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    def __init__(self,clause_embedder : torch.nn.Module,clause_key : torch.nn.Module, tw_mul: torch.nn.Module, tw_add: torch.nn.Module):
      super().__init__()

      self.clause_embedder = clause_embedder
      self.clause_key = clause_key
      self.tw_mul = tw_mul
      self.tw_add = tw_add

      # a bit of an overhead, as we normally expect to get tweaks supplied (via eatMyTweaks just below)
      # (but taking care of the possibility of the user not saying anything,
      # and so we have a reasonable default for that occasion)
      self.feature_mul = tw_mul(torch.zeros(HP.GSD_INPUT_MUL))
      self.feature_add = tw_add(torch.zeros(HP.GSD_INPUT_ADD))
      self.final_blend = torch.ones(HP.GSD_FINAL_BLENDERS)

    @torch.jit.export
    def eatMyTweaks(self,tweaks : List[float]):
      assert len(tweaks) == HP.GSD_INPUT_MUL+HP.GSD_INPUT_ADD+HP.GSD_FINAL_BLENDERS
      self.feature_mul = self.tw_mul(torch.tensor(tweaks[:HP.GSD_INPUT_MUL]))
      self.feature_add = self.tw_add(torch.tensor(tweaks[HP.GSD_INPUT_MUL:HP.GSD_INPUT_MUL+HP.GSD_INPUT_ADD]))
      self.final_blend = torch.tensor(tweaks[-HP.GSD_FINAL_BLENDERS:])

    @torch.jit.export
    def forward(self,id: int,features : Tuple[float, float, float, float, float, float, float, float, float, float, float, float]):
      # print("NN: Got",id,"with features",features)

      tFeatures : Tensor = torch.mul(self.feature_mul,torch.tensor(process_features(features))) + self.feature_add
      tInternal : Tensor = self.clause_embedder(tFeatures)

      if HP.GSD_FINAL_BLENDERS:
        assert len(tInternal) % HP.GSD_FINAL_BLENDERS == 0

        pre_val = torch.mul(self.clause_key.weight,tInternal)
        blocks_summed = torch.sum(pre_val.reshape(-1,HP.GSD_FINAL_BLENDERS),0)
        val = torch.dot(blocks_summed,self.final_blend)
      else:
        val = torch.dot(self.clause_key.weight,tInternal)

      return val.item()

  module = NeuralPassiveClauseContainer(*model)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_embedder : torch.nn.Module,
      clause_key: torch.nn.Module,
      tw_mul: torch.nn.Module,
      tw_add: torch.nn.Module,
      tweaks: torch.nn.Module,
      clauses,journal,proof_flas,warmup_time,select_time):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_embedder = clause_embedder # the MLP for clause feature vects to their embeddings
    self.clause_key = clause_key           # the key for multiplying an embedding to get a clause logits
    self.tw_mul = tw_mul
    self.tw_add = tw_add
    self.tweaks = tweaks

    self.clauses = clauses                 # id -> (feature_vec)
    self.journal = journal                 # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas           # set of the good ids
    self.warmup_time = warmup_time
    self.select_time = select_time

  def forward(self,prob):
    my_tweak = self.tweaks[nice_module_name(prob)].weight

    feature_mul = self.tw_mul(my_tweak[:HP.GSD_INPUT_MUL])
    feature_add = self.tw_mul(my_tweak[HP.GSD_INPUT_MUL:HP.GSD_INPUT_MUL+HP.GSD_INPUT_ADD])
    final_blend = my_tweak[-HP.GSD_FINAL_BLENDERS:]

    # let's a get a big matrix of feature_vec's, one for each clause (id)
    clause_list = []
    id2idx = {}
    for i,(id,features) in enumerate(sorted(self.clauses.items())):
      id2idx[id] = i
      clause_list.append(torch.tensor(features))
    feature_vecs = torch.stack(clause_list)
    feature_vecs = feature_mul*feature_vecs+feature_add

    # print("forward-feature_vecs",feature_vecs)

    # in bulk for all the clauses
    embeddings = self.clause_embedder(feature_vecs)
    if HP.GSD_FINAL_BLENDERS > 0:
      pre_logits = torch.mul(embeddings,self.clause_key.weight)
      # print(pre_logits.shape)
      blocks_summed = torch.sum(pre_logits.reshape(pre_logits.shape[0],-1,HP.GSD_FINAL_BLENDERS),1)
      # print(blocks_summed.shape)
      logits = torch.matmul(final_blend,torch.t(blocks_summed))
    else:
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
