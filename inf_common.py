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

from collections import defaultdict
from itertools import chain

def default_defaultdict_of_list():
  return defaultdict(list)

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
    journal = []         # [event,id], where event is one of EVENT_ADD EVENT_SEL EVENT_REM,
    proof_flas = set()

    # just temporaries, here during the parsing
    num_sels = 0

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
    return (len(clauses) != 0 and num_sels != 0,(clauses,journal,proof_flas))

  print("Repeatedly failing:",to_run)
  print(output)
  return None # meaning: "A major failure, consider keeping the used model for later debugging"

class SimpleClauseEvaluator(torch.nn.Module):
  def __init__(self):
    super().__init__()

    assert HP.CLAUSE_EMBEDDER_LAYERS > 0
    layer_list = [torch.nn.Linear(HP.NUM_FEATURES,HP.CLAUSE_INTERAL_SIZE),torch.nn.ReLU()]
    for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
      layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,HP.CLAUSE_INTERAL_SIZE))
      layer_list.append(torch.nn.ReLU())

    self.feature_processor = torch.nn.Sequential(*layer_list)
    self.default_key = torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,1,bias=False)

  def getKey(self):
    return self.default_key.weight

  def forward(self,input) -> Tensor:
    return self.default_key(self.feature_processor(input))

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
    def __init__(self,feature_processor : torch.nn.Module,
                      default_key : torch.nn.Module):
      super().__init__()

      self.feature_processor = feature_processor
      self.default_key = default_key

    @torch.jit.export
    def forward(self,id: int,features : Tensor):
      # print("NN: Got",id,"with features",features)

      assert len(features) == HP.NUM_FEATURES

      # TODO: this will not be needed if A) vampire gives us 32bit floats or B) we move to 64 in torch (see torch.set_default_dtype(torch.float64) in dlooper)
      # tFeatures : Tensor = features.float()
      processed = self.feature_processor(features)
      val = self.default_key(processed)
      return val.item()

  module = NeuralPassiveClauseContainer(model.feature_processor,model.default_key)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_evaluator : torch.nn.Module,
      clauses,journal,proof_flas):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(f"clause {len(clauses)} journal {len(journal)} proof_flas {len(proof_flas)}")

    self.clause_evaluator = clause_evaluator # the SimpleClauseEvaluator for clause evaluation
    self.clauses = clauses                   # id -> (feature_vec)
    self.journal = journal                   # (id,event), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.proof_flas = proof_flas             # set of the good ids

  def forward(self,dummy):
    # let's a get a big matrix of feature_vec's, one for each clause (id)
    clause_list = []
    id2idx = {}
    for i,(id,features) in enumerate(sorted(self.clauses.items())):
      # we could also do some cropping, if vampire gave us more and we wanted fewer
      assert len(features) == HP.NUM_FEATURES
      id2idx[id] = i
      clause_list.append(torch.tensor(features))
    feature_vecs = torch.stack(clause_list)
    # print("feature_vecs.shape",feature_vecs.shape)

    # print("forward-feature_vecs",feature_vecs)

    outer_dim = 1
    assert not self.training or outer_dim == 1

    # in bulk for all the clauses
    logits_list = []
    logits_for_this_tweak = self.clause_evaluator.forward(feature_vecs)
    logits_for_this_tweak = torch.squeeze(logits_for_this_tweak,dim=-1)
    # print("logits_for_this_tweak",logits_for_this_tweak.shape)
    logits_list.append(logits_for_this_tweak)
    logits = torch.stack(logits_list)
    # print("logits.shape",logits.shape)

    good_action_reward_loss = torch.zeros(outer_dim)
    num_good_steps = 0

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
    for (l,n) in [(good_action_reward_loss,num_good_steps),(entropy_loss,num_steps)]:
      if n > 0:
        something = True
        loss += l/n
    assert something, "The training example was still be degenerate!"
    return loss