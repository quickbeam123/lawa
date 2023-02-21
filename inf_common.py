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

from itertools import chain

import hyperparams as HP

EVENT_ADD = 0
EVENT_REM = 1
EVENT_SEL = 2

def vampire_perfrom(prob,opts):
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

  # print(status,instructions,activations)
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
        features = list(map(float,spl[2:]))
        assert len(features) == HP.NUM_FEATURES
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

def get_base_network():
  assert HP.CLAUSE_EMBEDDER_LAYERS > 0
  layer_list = [torch.nn.Linear(HP.NUM_FEATURES,HP.CLAUSE_INTERAL_SIZE),torch.nn.ReLU()]
  for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
    layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,HP.CLAUSE_INTERAL_SIZE))
    layer_list.append(torch.nn.ReLU())
  layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,1,bias=False))

  return torch.nn.Sequential(*layer_list)

class TweakedClauseEvaluator(torch.nn.Module):
  # if this got born with fewer networks/tweaks than the current value of HP.NUM_TWEAKS, let's add some (default initialized) to make the numbers work
  def fillup_networks_and_tweaks(self):
    assert len(self.networks) == len(self.tweaks)+1
    while len(self.tweaks) < HP.NUM_TWEAKS:
      self.networks.append(get_base_network())
      self.tweaks.append(torch.nn.Parameter(torch.tensor(0.0)))

  def __init__(self):
    super().__init__()

    self.networks = torch.nn.ModuleList([get_base_network()])
    self.tweaks = torch.nn.ParameterList()
    self.fillup_networks_and_tweaks()

  def getKey(self,idx=0):
    # by default, we look at the the last layer of the base network
    return self.networks[idx][-1]

  def setTweakVals(self,new_tweaks : List[float]):
    with torch.no_grad():
      for i,tweak in enumerate(new_tweaks):
        self.tweaks[i].fill_(tweak)

  def getTweakVals(self):
    return list(tweak.item() for tweak in self.tweaks)

  # return the active parameters (except the tweaks)
  def getActiveParams(self):
    for network in self.networks[HP.ACTIVE_FROM]:
      for param in network.parameters():
        yield param

  def getTweaksAsParams(self):
    return self.tweaks.parameters()

  def myforward(self,input) -> Tensor:
    if HP.NUM_TWEAKS == 0:
      # print("forward-in",input.shape)
      return self.networks[0].forward(input)

    for i,module in enumerate(self.networks[0]):
      # print("forward",i,"shape",input.shape)

      if isinstance(module,torch.nn.Linear):
        my_weight = torch.clone(module.weight)
        if module.bias is not None:
          my_bias = torch.clone(module.bias)
        else:
          my_bias = None

        for j,next_network in enumerate(self.networks[1:]):
          next_module = next_network[i]
          next_tweak = self.tweaks[j]
          my_weight += next_tweak*next_module.weight
          if my_bias is not None:
            my_bias += next_tweak*next_module.bias

        input = torch.nn.functional.linear(input, my_weight, my_bias)

      elif False:
        # TODO: when needed, should implement other network building blocks
        pass
      else:
        # this is for our non-linearities
        input = module(input)

    # print("forward-done",input.shape)
    return input

def get_initial_model():
  return TweakedClauseEvaluator()

def export_model(model_state_dict,name):
  # we start from a fresh model and just load its state from a saved dict
  model = get_initial_model()
  model.load_state_dict(model_state_dict)

  # eval mode and no gradient
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    def __init__(self,weights0 : List[Tensor],biases0: List[Tensor], weights2 : List[Tensor]):
      super().__init__()

      self.weights0 = weights0
      self.biases0 = biases0
      self.weights2 = weights2

    @torch.jit.export
    def eatMyTweaks(self,tweaks : List[float]):
      for i,tweak in enumerate(tweaks):
        self.weights0[0].add_(self.weights0[i+1],alpha=tweak)
        self.biases0[0].add_(self.biases0[i+1],alpha=tweak)
        self.weights2[0].add_(self.weights2[i+1],alpha=tweak)

    @torch.jit.export
    def forward(self,id: int,features : Tensor):
      # print("NN: Got",id,"with features",features)
      assert len(features) == HP.NUM_FEATURES

      # TODO: this will not be needed if A) vampire gives us 32bit floats or B) we move to 64 in torch (see torch.set_default_dtype(torch.float64) in dlooper)
      tFeatures : Tensor = features.float()
      val = torch.nn.functional.linear(tFeatures, self.weights0[0], self.biases0[0])
      val = torch.nn.functional.relu(val,inplace=True)
      val = torch.nn.functional.linear(val, self.weights2[0])
      return val.item()

  # TODO: ugly -- hardwiring the current architecture since jit scripting is not flexible enough atm
  assert HP.CLAUSE_EMBEDDER_LAYERS == 1
  weights0 = [network[0].weight for network in model.networks]
  biases0 = [network[0].bias for network in model.networks]
  weights2 = [network[2].weight for network in model.networks]

  module = NeuralPassiveClauseContainer(weights0,biases0,weights2)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_evaluator : torch.nn.Module,
      clauses,journal,proof_flas,warmup_time,select_time):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(clauses,journal,proof_flas)

    self.clause_evaluator = clause_evaluator # the TweakedClauseEvaluator for clause evaluation
    self.clauses = clauses                   # id -> (feature_vec)
    self.journal = journal                   # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas             # set of the good ids
    self.warmup_time = warmup_time
    self.select_time = select_time

  def forward(self,tweaks_to_try):
    # let's a get a big matrix of feature_vec's, one for each clause (id)
    clause_list = []
    id2idx = {}
    for i,(id,features) in enumerate(sorted(self.clauses.items())):
      # we could also do some cropping, if vampire gave us more and we want thed fewer
      assert len(features) == HP.NUM_FEATURES
      id2idx[id] = i
      clause_list.append(torch.tensor(features))
    feature_vecs = torch.stack(clause_list)
    # print("feature_vecs.shape",feature_vecs.shape)

    # print("forward-feature_vecs",feature_vecs)

    outer_dim = len(tweaks_to_try)
    assert not self.training or outer_dim == 1

    # in bulk for all the clauses
    logits_list = []
    for tweak in tweaks_to_try:
      self.clause_evaluator.setTweakVals(tweak)
      logits_for_this_tweak = self.clause_evaluator.myforward(feature_vecs)
      logits_for_this_tweak = torch.squeeze(logits_for_this_tweak,dim=-1)
      # print("logits_for_this_tweak",logits_for_this_tweak.shape)
      logits_list.append(logits_for_this_tweak)
    logits = torch.stack(logits_list)
    # print("logits.shape",logits.shape)

    good_action_reward_loss = torch.zeros(outer_dim)
    num_good_steps = 0

    time_penalty_loss = torch.zeros(outer_dim)
    time_penalty_volume = 0

    entropy_loss = torch.zeros(outer_dim)
    num_steps = 0

    # TODO: couldn't this be one-off compiled to get much more efficient?

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
        sub_logits = logits[:,indices]
        # print("forward-sub_logits",sub_logits)

        # print("sub_logits.shape",sub_logits.shape)
        lsm = torch.nn.functional.log_softmax(sub_logits,dim=-1)
        # print("lsm.shape",lsm.shape)

        if HP.LEARN_FROM_ALL_GOOD:
          good_idxs = []
          for i,id in enumerate(passive_list):
            if id in self.proof_flas:
              good_idxs.append(i)
          # print(good_idxs)
          if len(good_idxs):
            good_action_reward_loss += -torch.sum(lsm[:,good_idxs],dim=-1)/len(good_idxs)
            num_good_steps += 1
        else:
          if recorded_id in self.proof_flas:
            cur_idx = passive_list.index(recorded_id)
            good_action_reward_loss += -lsm[:,cur_idx]
            num_good_steps += 1

        if HP.TIME_PENALTY_MIXING > 0.0:
          cur_idx = passive_list.index(recorded_id)
          time_penalty_loss += HP.TIME_PENALTY_MIXING*event_time*lsm[:,cur_idx]
          time_penalty_volume += event_time

        if HP.ENTROPY_COEF > 0.0:
          # TODO: this needs debugging under tweaks
          minus_entropy = torch.dot(torch.exp(lsm,dim=-1),lsm)
          if HP.ENTROPY_NORMALIZED:
            minus_entropy /= torch.log(len(lsm))
          entropy_loss += HP.ENTROPY_COEF*minus_entropy
          num_steps += 1

        passive.remove(recorded_id)

    something = False
    loss = torch.zeros(outer_dim)
    for (l,n) in [(good_action_reward_loss,num_good_steps),(time_penalty_loss,time_penalty_volume),(entropy_loss,num_steps)]:
      if n > 0:
        something = True
        loss += l/n
    assert something, "The training example was still be degenerate!"
    return loss