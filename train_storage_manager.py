#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Load one or more train_storage.pt files
  # List the problems / proofs stored there.
  # Possibly merge those multiple files into one (to have an as large as possible collection for the next wave of trianing)
  #
  # To be called as in: ./train_storage_manager.py 

  master_storage = dict()
  master_prob_facts = dict()

  for storage_file_name in sys.argv[1:]:
    print("Opening",storage_file_name)
    train_storage,prob_facts = torch.load(storage_file_name)

    for prob,temp in train_storage:
      if (prob,temp) not in master_storage:
        print("Adding",(prob,temp))
        master_storage[(prob,temp)] = train_storage[(prob,temp)]
        master_prob_facts[prob] = 1

    print("So far tracking solutions for",len(master_prob_facts),"problems.")
    print()

  print("Created master_storage with",len(master_storage),"records. Tracking solutions of",len(master_prob_facts),"problems.")

  torch.save((master_storage,master_prob_facts), "master_train_storage.pt")