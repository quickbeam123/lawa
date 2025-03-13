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

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

class TraceIndex:
  def __init__(self):
    self.traces = {} # problem -> [trace_file_name]  # a list to support more than one sample per problem per loop (c.f. HP.NUM_PERFORMS)
    self.last_solved = {}
    self.prob_scores = {}

    if HP.CUMULATIVE:
      self.BASE = math.pow(HP.CUM_MAX_STRENGTH,1/(2*HP.CUM_STALE_AFTER))
      print("trace index establish a factor base of",self.BASE)

  def loop_finished(self):
    if not HP.CUMULATIVE:
      self.traces = {}

  def cur_problems(self):
    return self.traces.keys()

  def prob_traces(self,prob):
    return self.traces[prob]

  def prob_factor(self,prob):
    if not HP.CUMULATIVE:
      return 1.0
    return math.pow(self.BASE,self.prob_scores[prob])

  def add_prob_trace(self,loop,prob,trace_file_path):
    if (prob not in self.traces or
        # idea: if we solve the problem now, we throw its old traces away
        (HP.CUMULATIVE and prob in self.last_solved and self.last_solved[prob] < loop)):
      self.traces[prob] = [trace_file_path]
    else:
      self.traces[prob].append(trace_file_path)

    self.last_solved[prob] = loop

  def report_trivial_trace(self,loop,prob):
    if HP.CUMULATIVE and prob in self.traces:
      del self.traces[prob]

  def update_scores(self,loop):
    if not HP.CUMULATIVE:
      return
    num_stale = 0
    num_new = 0
    num_routine = 0
    num_losing = 0
    for prob,last in self.last_solved.items():
      if loop-last > HP.CUM_STALE_AFTER:
        if prob in self.traces:
          del self.traces[prob]
        if prob in self.prob_scores:
          del self.prob_scores[prob]
        num_stale += 1
      elif last == loop:
        if prob not in self.prob_scores:
          # newly solved, or solved after being long forgotten
          self.prob_scores[prob] = 0
          num_new += 1
        else:
          # we already know we are solving it, so let's experiment with a lower score
          self.prob_scores[prob] -= 1
          num_routine += 1
      else:
        # we seem to be losing this problem, let's try to catch up
        self.prob_scores[prob] += 2
        num_losing += 1
    print("trace index score update:")
    print("  new   :",num_new)
    print("  easing:",num_routine)
    print("  losing:",num_losing)
    print("  staled:",num_stale)
    print("  score hist")
    hist = defaultdict(int)
    for _,score in self.prob_scores.items():
      hist[score] += 1
    for score,val in sorted(hist.items()):
      print("    {:>6} {:>6}".format(score, val))
    print()

  def report(self):
    trace_cnt = 0
    for prob,trace_list in self.traces.items():
      trace_cnt += len(trace_list)
    print("trace_index has\n  ",len(self.traces),"probs with a total of",trace_cnt,"traces")

if __name__ == "__main__":
  # see mini.py for info

  trace_index = torch.load(sys.argv[1])

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

