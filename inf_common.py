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
    predecessors = {}    # id -> (rule,parents,age)
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
      elif line.startswith("p: "):
        spl = line.split()
        id = int(spl[1])
        age = float(spl[-1])
        rule = int(spl[2])
        parents = [int(p) for p in spl[3:-1]]
        predecessors[id] = (rule,parents,age)
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
    return (len(clauses) != 0 and num_sels != 0,(clauses,journal,predecessors,proof_flas))

  print("Repeatedly failing:",to_run)
  print(output)
  return None # meaning: "A major failure, consider keeping the used model for later debugging"

DEFAULT_AGE_SHIFTS = [0.0]*HP.NUM_INFERENCES
for i in range(HP.GENERIC_GENERATING_INFERENCE,HP.INTERNAL_GENERATING_INFERNCE_LAST):
    DEFAULT_AGE_SHIFTS[i] = 1.0

AGE_LEAF = 0
AGE_MAX = 1
AGE_SUM = 2

# taking into account that some clauses may have the same feature vector,
# let's not waste space and time on those and create a compressed trace where
# each unique feature vector appears only once
def compress_trace(clauses,journal,predecessors,proof_flas,curAgeCorrections):
  # compute the age abstraction as a perfectly shared expression DAG using
  # leaf-const node (could be something else than 0.0 for s2a)
  # max over nodes (could be more than binary max for, e.g., urr)
  # sum of a node and a sparse sorted list of inference corrections

  # since "id" is reserved for clause, we will use the term "node" for an id of an expression

  id2node = {} # orig clause id to its node

  expr2node = {} # expressions to nodes
  node2expr = {} # the reverse of the above

  # implements the perfect sharing idea, introducing new nodes if not-previously-seen experssion arrives
  def insert(expr):
    if expr in expr2node:
      node = expr2node[expr]
    else:
      node = len(expr2node)
      expr2node[expr] = node
      node2expr[node] = expr
    return node

  # a convenience function to extend a sum if that's the kind of node we want to add something to
  def addTo(premise_node,rule):
    premise_expr = node2expr[premise_node]
    if premise_expr[0] == AGE_SUM:
      predecessor_node = premise_expr[1]
      summands = list(premise_expr[2])
      summands.append(rule)
      summands.sort()
      return insert((AGE_SUM,predecessor_node,tuple(summands)))
    else:
      return insert((AGE_SUM,premise_node,(rule,)))

  def almostEqual(a,b):
    EPSI = 0.0001
    if a == 0.0 or b == 0.0:
      return abs(a-b) < EPSI
    return abs(a-b) < EPSI*max(abs(a),abs(b))

  for id,(rule,parents,age) in predecessors.items():
    if len(parents) == 0: # we ignore the rule in this case (it's probably 15 (cnf transformation), or 0 (input) for cnf problems)
      id2node[id] = insert((AGE_LEAF,age))
      computed_age = age

    elif rule > HP.GENERIC_GENERATING_INFERENCE and rule < HP.INTERNAL_GENERATING_INFERNCE_LAST: # a generating inference
      parent_nodes = [id2node[p] for p in parents]
      if len(set(parent_nodes))==1:
        # special case for max over a singleton - don't do it!
        id2node[id] = addTo(parent_nodes[0],rule)
      else:
        # first create the max expressions (we assume parents have already been hashed)
        max_expr = (AGE_MAX,tuple(sorted(parent_nodes)))
        max_node = insert(max_expr)

        sum_expr = (AGE_SUM,max_node,(rule,)) # the max_node + singleton tuple to hash the rule
        id2node[id] = insert(sum_expr)

      computed_age = max(predecessors[p][-1] for p in parents)+curAgeCorrections[rule]
      assert almostEqual(computed_age,age), f"Age computation mismatch for a generating inference ({computed_age} vs {age})"

    else: # reductions or anything that builds age using the main (= first) premise
      id2node[id] = addTo(id2node[parents[0]],rule)

      computed_age = predecessors[parents[0]][-1]+curAgeCorrections[rule]
      assert almostEqual(computed_age,age), f"Age computation mismatch for a non-generating inference ({computed_age} vs {age})"

    # print(id,"reported",age,"now is",computed_age)

  id2idx = {}
  features2idx = {}
  clause_features = []
  for cl,features in clauses.items():
    # here we replace age from vampire with a node describing how it is computed from leaf values
    features[0] = id2node[cl] # AGE must be the 0-th feature

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
  return clause_features,node2expr,newjournal,num_good_selections

class SimpleClauseEvaluatorWithAgeCorrections(torch.nn.Module):
  def __init__(self):
    super().__init__()

    assert HP.CLAUSE_EMBEDDER_LAYERS > 0
    layer_list = [torch.nn.Linear(HP.NUM_FEATURES,HP.CLAUSE_INTERAL_SIZE),torch.nn.ReLU()]
    for _ in range(HP.CLAUSE_EMBEDDER_LAYERS-1):
      layer_list.append(torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,HP.CLAUSE_INTERAL_SIZE))
      layer_list.append(torch.nn.ReLU())

    self.feature_processor = torch.nn.Sequential(*layer_list)
    self.default_key = torch.nn.Linear(HP.CLAUSE_INTERAL_SIZE,1,bias=False)

    self.age_corrections = torch.nn.Parameter(torch.tensor(DEFAULT_AGE_SHIFTS))

  def getKey(self):
    return self.default_key.weight

  def forward(self,input) -> Tensor:
    return self.default_key(self.feature_processor(input))

def get_initial_model():
  return SimpleClauseEvaluatorWithAgeCorrections()

def export_model(model_state_dict,name):
  # we start from a fresh model and just load its state from a saved dict
  model = get_initial_model()
  model.load_state_dict(model_state_dict)

  # eval mode and no gradient
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  class NeuralPassiveClauseContainer(torch.nn.Module):
    # knowns : Dict[Tensor,Tensor]

    def __init__(self,feature_processor : torch.nn.Module,
                      default_key : torch.nn.Module,
                      age_corrections : torch.nn.Module):
      super().__init__()

      self.feature_processor = feature_processor
      self.default_key = default_key
      self.age_corrections = age_corrections
      # self.knowns = {}

    @torch.jit.export
    def getAgeCorrections(self) -> Tensor:
      return self.age_corrections

    @torch.jit.export
    def forward(self,features : Tensor):
      # print("NN: Got",id,"with features",features)

      # if features in self.knowns:
      #   return self.knowns[features]

      # assert len(features) == HP.NUM_FEATURES

      # TODO: this will not be needed if A) vampire gives us 32bit floats or B) we move to 64 in torch (see torch.set_default_dtype(torch.float64) in dlooper)
      # tFeatures : Tensor = features.float()
      processed = self.feature_processor(features)
      val = self.default_key(processed)
      # self.knowns[features] = val
      return val

  module = NeuralPassiveClauseContainer(model.feature_processor,model.default_key,model.age_corrections)
  script = torch.jit.script(module)
  script.save(name)

class LearningModel(torch.nn.Module):
  def __init__(self,
      verbose,
      clause_evaluator : torch.nn.Module,
      clause_features,node2expr,journal,num_good_selections):
    super().__init__()

    # print(clause_embedder,clause_key)
    # print(f"clause {len(clauses)} journal {len(journal)} proof_flas {len(proof_flas)}")

    self.verbose = verbose

    self.clause_evaluator = clause_evaluator       # the SimpleClauseEvaluator for clause evaluation
    self.clause_features = clause_features         # list of clause features (in a list); idx in the journal is into this list
    self.node2expr = node2expr                     # when computing age, this is how we get back from abstract age-expression nodes to floats
    self.journal = journal                         # (event,idx,is_proof_cl), where event is one of EVENT_ADD EVENT_SEL EVENT_REM
    self.num_good_selections = num_good_selections # how many useful EVENT_SEL are there in journal?

    if verbose:
      print("Got verbose")
      print(len(clause_features))
      print(len(journal))
      print(num_good_selections)

  def forward(self):
    # compute age for every node in node2expr first:
    age_values = {} # node -> scalar tensor computing that age
    for node,expr in self.node2expr.items(): # we rely on these being processed in insertion order
      if expr[0] == AGE_LEAF:
        age_values[node] = torch.tensor(expr[1])
      elif expr[0] == AGE_MAX:
        args = expr[1]
        age_values[node] = torch.max(torch.stack([age_values[a] for a in args]))
      else:
        assert expr[0] == AGE_SUM
        total = age_values[expr[1]]
        for rule in expr[2]:
          # can't use += below, which is in-place!
          total = total + self.clause_evaluator.age_corrections[rule]
        age_values[node] = total

    # let's a get a big matrix of feature_vec's, one for each clause idx
    feature_vecs = torch.stack([
        torch.stack([age_values[features[0]]]+[torch.tensor(f) for f in features[1:]]) for features in self.clause_features])
    # print("feature_vecs.shape",feature_vecs.shape)

    logits = self.clause_evaluator.forward(feature_vecs)
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
