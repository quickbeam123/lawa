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

  def process_trace(trace_path):
    size = os.path.getsize(trace_path)
    trace_tuple = torch.load(trace_path)
    clnum = len(trace_tuple[4])
    return size, clnum

  tasks = []
  for prob,traces in trace_index.traces.items():
    assert len(traces) == 1
    tasks.append(traces[0])

  with multiprocessing.Pool(120) as pool:
    results = pool.map(process_trace, tasks)

  all_sizes = [r[0]//1024 for r in results]
  all_clnums = [r[1] for r in results]

  num_traces = len(results)
  print("max_size",max(all_sizes),"KB")
  print("avg_size",int(sum(all_sizes)/num_traces),"KB")
  print("median_size",int(numpy.median(all_sizes)),"KB")

  print("max_clnum",max(all_clnums))
  print("avg_clnum",sum(all_clnums)/num_traces)
  print("median_clnum",numpy.median(all_clnums))

  import matplotlib.pyplot as plt
  import numpy as np

  if True:
    TITLE = "neurally-guided (iter. 2)"
    SUFFIX = "neural"
    COLORS = ["red", "red"]
  else:
    TITLE = "default strategy (iter. 1)"
    SUFFIX = "default"
    COLORS = ["blue", "blue"]

  COLOR_THEMES = {
    "blue": {"edge": "#1f77b4", "face": "#aec7e8", "median": "#0b3d91"},
    "red":  {"edge": "#d62728", "face": "#f4a582", "median": "#8b0000"},
  }

  fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5))

  for ax, data, title, ylabel, color in zip(axes.flat,
      [all_sizes, all_clnums],
      ["trace file size (KB)", "clause count"],
      ["bytes", "clauses"],
      COLORS):
    theme = COLOR_THEMES[color]
    log_data = [np.log10(max(v, 1)) for v in data]
    vp = ax.violinplot(log_data, showmedians=True)
    for body in vp['bodies']:
      body.set_facecolor(theme["face"])
      body.set_edgecolor(theme["edge"])
    for part in ['cmins', 'cmaxes', 'cbars']:
      vp[part].set_edgecolor(theme["edge"])
    vp['cmedians'].set_edgecolor(theme["median"])
    max_exp = int(np.ceil(max(log_data)))
    ticks = list(range(0, max_exp + 1))
    ax.set_yticks(ticks)
    ax.set_ylim(0, max_exp)
    ax.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_title(title)

  fig.suptitle(TITLE)
  plt.tight_layout()
  plt.savefig(f"tracy_violin_{SUFFIX}.pdf", format="pdf", bbox_inches="tight")
  plt.close(fig)

