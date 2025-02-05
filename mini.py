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
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,gatherwish,mission,prob,opts1,opts2)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,counter,opts)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL = 2    # construct our network to get the loss of this trace (no training to do)
  # input:     (prob,fact,trace_file_paths,model_file_path)
  # output:    the computed loss
JK_TRAIN = 3   # construct our network to get the loss of this trace and do one training step
  # input:     (prob,fact,trace_file_paths,train_model_file_path)
  # output:    the computed loss

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()

    if job_kind == JK_PERFORM:
      (res_filename,gatherwish,mission,prob,i,lrs_trace_file,opts1,opts2) = input
      result = IC.vampire_perfrom(prob,opts1+opts2)
      q_out.put((job_kind,input,result))
    elif job_kind == JK_GATHER:
      (mission,prob,lrs_trace_file,trace_file_path,opts) = input
      (status,instructions,activations) = IC.vampire_perfrom(prob,opts)

      assert status == "uns", f"Ran {(prob,opts)} got {(status,instructions,activations)}"
      assert os.path.isfile(trace_file_path)
      trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_file_path)

      q_out.put((job_kind,input,(trace_kept,gage_stats, gweight_stats)))

    elif job_kind == JK_EVAL:
      (prob,fact,trace_file_paths,model_file_path) = input

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      local_fact = 1/len(trace_file_paths)
      trace_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      # print("EVAL on",prob,fact,trace_file_paths)

      try:
        loss = torch.zeros(1)
        for trace_tuple in trace_tuples:
          learn_model = IC.LearningModel(False,local_model,trace_tuple)
          learn_model.eval()
          # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
          # print(tweaks_to_try)
          loss += local_fact*learn_model.forward()
      except Exception as e:
        print(f"Exception {e} occurred in EVAL")
        sys.stdout.flush()

      # print("EVAL on",prob,fact,trace_file_paths,loss.item())

      q_out.put((job_kind,input,fact*loss.item()))

    elif job_kind == JK_TRAIN:
      (prob,fact,trace_file_paths,train_model_file_path) = input

      local_fact = 1/len(trace_file_paths)
      trace_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(train_model_file_path))

      verbose = False # (prob in {'Problems/COM/COM021+4.p'})

      # print("TRAIN on",prob,fact,trace_file_paths)

      try:
        loss = torch.zeros(1)
        for trace_tuple in trace_tuples:
          learn_model = IC.LearningModel(verbose,local_model,trace_tuple)
          learn_model.train()

          loss += local_fact*learn_model.forward()

        loss.backward()
      except Exception as e:
        print(f"Exception {e} occurred in TRAIN")
        sys.stdout.flush()

      # print("TRAIN on",prob,fact,trace_file_paths,loss.item())

      for param in local_model.parameters():
        grad = param.grad
        param.requires_grad = False # to allow the in-place operation just below
        if grad is not None:
          param.copy_(grad)
        else:
          param.zero_()

      # use the same file also for the journey back (which brings the gradients inside the actual params)
      torch.save(local_model.state_dict(), train_model_file_path)

      q_out.put((job_kind,input,fact*loss.item()))


if __name__ == "__main__":
  # A fake run of either training or eval in a single process, to report time taken and peak memory usage
  # Used to establish cutoff values for various trace and model sizes (and put them into HP)
  # When the data is collected using this script, e.g.:
  #   cat traces100k_01 | xargs -I {} ./mini.py ~/mtpa-gnn/tptpOverfit100k/loop0/loop-model-and-optimizer.tar {} > trainjobs_01.txt 2>&1 &
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
    ttuple = torch.load(sys.argv[2])
    learn_model = IC.LearningModel(True,local_model,ttuple)
    # learn_model.eval()
    learn_model.train()
    fwd_start = time.time()
    loss = learn_model.forward()
    bwd_start = time.time()
    loss.backward()
    print("FwdTook",bwd_start-fwd_start)
    print("BwdTook",time.time()-bwd_start)
    print("Loss:",loss.item())
    print("Took:",time.time()-btime)

  usage = resource.getrusage(resource.RUSAGE_SELF)
  max_memory_kb = usage.ru_maxrss
  print(f"Peak memory usage: {max_memory_kb//1024} MB")
