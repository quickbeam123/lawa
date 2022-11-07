#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time

import subprocess
from collections import defaultdict

if __name__ == "__main__":
  # Run the part of looper which trains a model
  #
  # ./train_debugger.py expers/exper17/loop0/parts-model.pt expers/exper17/loop0/train_data.pt

  training_data = torch.load(sys.argv[2])

  # let's have a quick look at the training data first:
  buf = []
  for prob,its_proofs in training_data.items():
    for (clauses,journal,proof_flas,warmup_time,select_time) in its_proofs:
      buf.append((len(journal),len(clauses),len(proof_flas),warmup_time,select_time,prob))

  buf.sort()
  for vec in buf:
    print(vec)

  exit(0)

  model = torch.load(sys.argv[1])

  if HP.OPTIMIZER ==  HP.OPTIMIZER_SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
  elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  # the actual training
  start_time = time.time()
  # SGD style (one step per problem)
  factor = 1.0/len(training_data)
  print("  training",end="")
  training_data_in_order = list(training_data.items())
  random.shuffle(training_data_in_order)
  for i,(prob,its_data) in enumerate(training_data_in_order):
    # print(prob)
    print(">",end="")

    f = factor / len(its_data)
    for (clauses,journal,proof_flas) in its_data:
      print(".",end="")

      optimizer.zero_grad()
      if HP.INCLUDE_LSMT:
        lm = IC.RecurrentLearningModel(*model,clauses,journal,proof_flas)
      else:
        lm = IC.LearningModel(*model,clauses,journal,proof_flas)
      lm.train()
      loss = lm.forward()
      if loss is None:
        continue
      loss *= f # scale down by the number of problems we trained on
      loss.backward()
      optimizer.step()
  print()
  print("Training took",time.time()-start_time)
  print()
  sys.stdout.flush()
