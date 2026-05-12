#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain

import resource

import multiprocessing
import numpy

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

import warnings
warnings.filterwarnings("ignore", message=r"Initializing zero-element tensors is a no-op", category=UserWarning)

if __name__ == "__main__":
  # A fake run of either training or eval in a single process, to report time taken and peak memory usage
  # Used to establish cutoff values for various trace and model sizes (and put them into HP)
  # When the data is collected using this script, e.g.:
  #   find ~/jar2026/loop1_unconstrained/traces -type f | xargs -I {} -P 32 ./mini.py ~/jar2026/loop1_unconstrained/loop0/loop-model.tar {} > trainjobs_01.txt 2>&1 &
  #
  # the results can be plotted and otherwise analyzed using check_mini.py
  #
  # the traces to run on can either be obtained by find (scanning the traces folder) or by running ./tracy.py
  #
  # Note that not all the raw traces file from an interruped ./elooper.py are in the right form
  # - that's why we have the if False code just below to process them one more time

  model_et_at_path = sys.argv[1]
  trace_path = sys.argv[2]

  if False:
    with open("traces100k.txt","r") as f:
      traces = f.readlines()

      def process_trace(trace_path):
        trace_path = trace_path.strip()
        ttuple = torch.load(trace_path)
        # check if it's an int
        if isinstance(ttuple[3], int):
          print("Already processed")
        else:
          trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_path, sys.stdout)
          if not trace_kept:
            print("Dropping trace", trace_path)
            # delete the trace file:
            os.remove(trace_path)

      with multiprocessing.Pool(120) as pool:
        pool.map(process_trace, traces)

    exit(0)

  print("Input:",trace_path.split("/")[-1])
  print("Ofsize:",os.path.getsize(trace_path)//1024,"KB")

  if True:
    btime = time.time()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    local_model = IC.get_initial_model()
    local_model.load_state_dict(torch.load(model_et_at_path)[1])
    ttuple = torch.load(trace_path)
    learn_model = IC.LearningModel(True,local_model,ttuple)
    # learn_model.eval()
    learn_model.train()

    fwd_start = time.time()
    just_before_final,num2idx = learn_model.pre_forward()
    notweaks = IC.get_neutral_tweak(local_model.clause_valuator_snd, detached = False).unsqueeze(0)
    losses,selection_hit_rates,dist_to_goods = learn_model.forward(just_before_final, num2idx, notweaks)
    loss = losses[0]
    bwd_start = time.time()
    loss.backward()
    print("FwdTook",bwd_start-fwd_start)
    print("BwdTook",time.time()-bwd_start)
    print("Loss:",loss.item())
    print("Took:",time.time()-btime)

  usage = resource.getrusage(resource.RUSAGE_SELF)
  max_memory_kb = usage.ru_maxrss
  print(f"Peak memory usage: {max_memory_kb//1024} MB")
