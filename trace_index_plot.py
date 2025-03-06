#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time, math, random

from collections import defaultdict

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

TRACE_INDEX = "trace-index.pt"


if __name__ == "__main__":
  # Show the history of problems through training and their contibution factors
  #
  # To be called as in: ./trace_index_plot.py exper_folder

  exper_dir = sys.argv[1]

  trajectories = defaultdict(list) # prob -> [(time,factor)]

  root, dirs, files = next(os.walk(exper_dir))
  loop = 2^31
  for dir in dirs:
    if dir.startswith("loop"):
      dirs_loop = int(dir[4:])
      if dirs_loop < loop:
        loop = dirs_loop
  if loop == 0:
    loop = 1
  while True:
    loop_str = "loop{}".format(loop)
    cur_dir = os.path.join(exper_dir,loop_str)
    if not os.path.isdir(cur_dir):
      break

    trace_index = torch.load(os.path.join(cur_dir,TRACE_INDEX))

    for prob in trace_index.cur_problems():
      trajectories[prob].append((loop,trace_index.prob_factor(prob)))

    loop += 1

  import matplotlib.pyplot as plt
  import matplotlib.lines as mlines

  fig, ax1 = plt.subplots(figsize=(8,6))
  color_cycle = ax1._get_lines.prop_cycler
  handles = []

  total_plotted = 0
  traj_items = list(trajectories.items())
  random.shuffle(traj_items)
  for prob,trajectory in traj_items:
    # skip the boring ones!
    if prob in trace_index.prob_scores and (trace_index.prob_scores[prob] == -loop+2):
      continue

    col = next(color_cycle)['color']

    Xs = []
    Ys = []
    def commit(Xs,Ys):
      if len(Xs) == 1:
        ax1.plot(Xs, Ys, marker='o', markersize=2, color=col)
      elif len(Xs) > 1:
        ax1.plot(Xs, Ys, linewidth = 1, color=col)
      Xs.clear()
      Ys.clear()

    for l,factor in trajectory:
      if len(Xs) > 0 and Xs[-1] != l-1:
        commit(Xs,Ys)
      Xs.append(l)
      Ys.append(factor)
    commit(Xs,Ys)

    h = mlines.Line2D([], [], color=col, label=prob)
    handles.append(h)

    total_plotted += 1
    if total_plotted > 9:
      break

  plt.legend(handles = handles, loc='upper left')
  plt.savefig("trace_index_plot.pdf",format="pdf", bbox_inches="tight")
  plt.close(fig)





