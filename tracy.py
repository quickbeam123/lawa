#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain

import multiprocessing
import numpy

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

from elooper import TraceIndex

if __name__ == "__main__":
  # see mini.py for info

  trace_index = torch.load(sys.argv[1],weights_only=False)

  if False:
    print(trace_index.traces["Problems/SYN/SYN759-1.p"])
    print(trace_index.prob_scores["Problems/SYN/SYN759-1.p"])

    del trace_index.traces["Problems/SYN/SYN759-1.p"]

    torch.save(trace_index, sys.argv[1])

    exit(0)

  if False:
    tasks = []
    for prob,traces in trace_index.traces.items():
      for idx,trace in enumerate(traces):
        tasks.append((trace,prob,idx))

    def process_trace(task):
      trace_path,prob,idx = task

      ttuple = torch.load(trace_path)
      # check if it's an int
      if isinstance(ttuple[3], int):
        pass
        # print("Already processed",trace_path, prob, idx)
      else:
        trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_path, sys.stdout)
        if not trace_kept:
          print("Dropping trace", trace_path, prob, idx)
          # delete the trace file:
          os.remove(trace_path)

    with multiprocessing.Pool(120) as pool:
      pool.map(process_trace, tasks)

    exit(0)

  max_size = 0
  size_sum = 0
  num_traces = 0

  for prob,traces in trace_index.traces.items():
    print(prob,traces)
    assert len(traces) == 1
    size = os.path.getsize(traces[0])
    max_size = max(max_size,size)
    size_sum += size
    num_traces += 1
    print("Ofsize:",size//1024,"KB")
    """
    if trace_index.prob_scores[prob] <= 0:
      print(trace_index.prob_traces(prob)[0])
    """
  print("max_size",max_size//1024//1024,"MB")
  print("avg_size",size_sum/num_traces//1024//1024,"MB")

