#!/usr/bin/env python3

from collections import defaultdict

import torch

import inf_common as IC

import sys,os

from multiprocessing import Pool

def trace_index_content_summary(index):
  num_probs = len(index)
  num_prob_temps = sum(len(temp_dict) for _prob,temp_dict in index.items())
  num_traces = sum(len(trace_list) for prob,temp_dict in index.items() for temp, trace_list in temp_dict.items())
  return "{} probs, {} prob_temps, and {} traces in total.".format(num_probs,num_prob_temps,num_traces)

if __name__ == "__main__":
  # Load one or more train_storage.pt files
  # List the problems / proofs stored there.
  # Possibly merge those multiple files into one (to have an as large as possible collection for the next wave of trianing)
  #
  # To be called as in: ./train_index_manager.py trace-index.pt ...

  master_index = None

  for trace_index_file_name in sys.argv[1:]:
    print("Opening",trace_index_file_name)
    new_index = torch.load(trace_index_file_name)
    print("   Loaded a trace_index with",trace_index_content_summary(new_index))

    if master_index is None:
      master_index = new_index
    else:
      # merging
      for prob,temp_dict in new_index.items():
        if prob not in master_index:
          master_index[prob] = temp_dict
          # print("Added",prob)
        else:
          master_temp_dict = master_index[prob]
          for temp, trace_list in temp_dict.items():
            if temp not in master_temp_dict:
              master_temp_dict[temp] = trace_list
              # print("Added",prob,"/",temp)

    print("Master index grown to",trace_index_content_summary(master_index))

  if True:
    print("Saving to master-trace-index.pt")
    torch.save(master_index, "master-trace-index.pt")

