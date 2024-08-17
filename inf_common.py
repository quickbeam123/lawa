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
      # print(line)
      if "% Instruction limit reached!" in line:
        assert not proof_flas
        break # better than "id appeared again for:" failing just below
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

# taking into account that some clauses may have the same feature vector,
# let's not waste space and time on those and create a compressed trace where
# each unique feature vector appears only once
def compress_trace(clauses,journal,proof_flas):
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
  num_selections = 0
  for tag,id in journal:
    if tag == EVENT_ADD:
      assert id not in passive
      passive.add(id)
    else:
      assert(tag == EVENT_REM or tag == EVENT_SEL)
      assert id in passive
      passive.remove(id)
    if tag == EVENT_SEL:
      num_selections += 1
    newjournal.append((tag,id2idx[id],id in proof_flas))
  return clause_features,newjournal,num_selections

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
    def forward(self,features : Tensor):
      # print("NN: Got",id,"with features",features)

      # assert len(features) == HP.NUM_FEATURES

      # TODO: this will not be needed if A) vampire gives us 32bit floats or B) we move to 64 in torch (see torch.set_default_dtype(torch.float64) in dlooper)
      # tFeatures : Tensor = features.float()
      processed = self.feature_processor(features)
      val = self.default_key(processed)
      return val

  module = NeuralPassiveClauseContainer(model.feature_processor,model.default_key)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      clause_evaluator : torch.nn.Module,
      clause_features,journal,num_selections):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(f"clause {len(clauses)} journal {len(journal)} proof_flas {len(proof_flas)}")

    self.verbose = verbose

    self.clause_evaluator = clause_evaluator # the SimpleClauseEvaluator for clause evaluation
    self.clause_features = clause_features   # list of clause features (in a list); idx in the journal is into this list
    self.journal = journal                   # (event,idx,is_proof_cl), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.num_selections = num_selections     # how many EVENT_SEL are there in journal?

    if verbose:
      print("Got verbose")
      print(len(clause_features))
      print(len(journal))
      print(num_selections)

  def forward(self):
    # let's a get a big matrix of feature_vec's, one for each clause idx
    feature_vecs = torch.stack([torch.tensor(features) for features in self.clause_features])
    # print("feature_vecs.shape",feature_vecs.shape)

    logits = self.clause_evaluator.forward(feature_vecs)
    logits = logits.squeeze(1) # squeeze-away second dimension, where the feartures were

    # print("logits",logits.shape)

    good_action_reward_loss = torch.tensor(0.0)
    num_good_steps = 0

    # TODO: couldn't this be one-off compiled to get much more efficient?

    passive = [0]*len(self.clause_features)
    passive_good = [0]*len(self.clause_features)

    learn_for_every = self.num_selections / HP.MAX_TRAINS_PER_TRACE
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
      # if sum(passive_good) and (num_good_steps==0 or random.uniform(0.0, 1.0) < HP.MAX_TRAINS_PER_TRACE / self.num_selections): -- didn't like a non-deterministic solution
      if sum(passive_good): # can learn
        if learn_for_every_sum <= learn_ord:
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