#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Load vampire-lawa log (to be run with "-spt on -p on"), or logs
  # and turn them into a pickle for learning (updating) a model
  #
  # To be called as in: ./log_loader.py testing/PUZ001+1.discount.npcc-random.spt-on.log
  
  log_name = sys.argv[1]
  print("Loading",filename)
  (clauses,journal,proof_flas) = IC.load_one(log_name)
  out_name = log_name+".pt"
  torch.save((clauses,journal,proof_flas), out_name)
  print("Savedto",out_name)