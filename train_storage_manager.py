#!/usr/bin/env python3

from collections import defaultdict

import torch

import sys,os

from multiprocessing import Pool

if __name__ == "__main__":
  # Load one or more train_storage.pt files
  # List the problems / proofs stored there.
  # Possibly merge those multiple files into one (to have an as large as possible collection for the next wave of trianing)
  #
  # To be called as in: ./train_storage_manager.py master_train_storage.pt

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

  if False:
    torch.save((master_storage,master_prob_facts), "master_train_storage.pt")

  if True:
    MISSIONS = ["train","test"]

    def get_empty_trace_index():
      return { m : defaultdict(dict) for m in MISSIONS} # mission -> problem -> temp -> (loop when obtained,trace_file_name)

    def get_trace_file_path(traces_dir,mission,prob,temp):
      return os.path.join(traces_dir,"{}_{}_{}.pt".format(mission,prob.replace("/","_"),temp))

    TRACE_INDEX = "trace-index.pt"

    def save_trace_index(cur_dir,trace_index):
      trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
      torch.save(trace_index, trace_index_file_path)

    MASTER_TRACES = "master_traces"

    trace_index = get_empty_trace_index()
    for (prob,temp),(loop,trace) in master_storage.items():
      trace_file_path = get_trace_file_path(MASTER_TRACES,"train",prob,temp)
      torch.save(trace,trace_file_path)
      trace_index["train"][prob][temp] = (loop,trace_file_path)
      print("train",prob,temp,loop,trace_file_path)
    save_trace_index(".",trace_index)
