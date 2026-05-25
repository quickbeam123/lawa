#!/usr/bin/env python3

import workers as W
import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time, copy

from collections import defaultdict

# only the first exper counts (by default);
# elooper may run more, but usually not for "test" and
# even with "train" the subsequent onces are not "full"
# (so computing fractions would be tricky!)
EXPERS_TO_CONSIDER = lambda i : i == 0

def rowwise_mean(lol):
  """Mean of each row in a (possibly ragged) list of lists."""
  return [sum(row) / len(row) for row in lol]

def rowwise_std(lol):
  """Population std-dev of each row in a (possibly ragged) list of lists."""
  means = rowwise_mean(lol)
  return [(sum((x - m) ** 2 for x in row) / len(row)) ** 0.5
          for row, m in zip(lol, means)]

STYLES = { "train" : "-", "test" : "--"}

UPDATE_LOOP_IDX_BY = -1 # so that the plots start at 0 (instead of 1)

if __name__ == "__main__":
  # Plotting some training curves, automatically getting the data from the exper directories left behind by looper
  #
  # To be called as in: ./plotterNew.py exper_folder1 exper_folder2 ...

  expers = defaultdict(lambda : defaultdict(dict)) # "train" -> {}, (optionally "test" -> {}); inside each entry is an loop_idx (int) -> percentage_solved (float)

  for exper_dir in sys.argv[1:]:
    print(exper_dir)
    root, dirs, files = next(os.walk(exper_dir))
    for dir in dirs:
      if dir.startswith("loop"):
        loop_idx = int(dir[4:])
      else:
        continue

      cur_dir = os.path.join(exper_dir,dir)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file == "train_res.pt":
          sub_exper = expers[exper_dir]["train"]
        elif file == "test_res.pt":
          sub_exper = expers[exper_dir]["test"]
        else:
          continue

        (_meta,results) = torch.load(os.path.join(cur_dir,file),weights_only=False)
        # results is like {"ProblemName" -> [(0, 16000, VampResult(status='uns', instructions=1200, activations=30, nn_warmup=495, nn_gnn=356, nn_bulks=137, strategy=None))]}
        solveds = {prob for prob,runs in results.items() for (i,ilim,info) in runs if (EXPERS_TO_CONSIDER(i) and info.status == "uns") }

        sub_exper[loop_idx+UPDATE_LOOP_IDX_BY] = [len(solveds)/len(results)] # starting a singleton list, to be compatible with the optional group-by phase
    print("  read",len(expers[exper_dir]["train"]),"train and",len(expers[exper_dir]["test"]),"result entries")

  if True: # group by the first segment ("_"-delimited) of the exper_dir name
    orig_expers = copy.deepcopy(expers)
    expers = defaultdict(lambda : defaultdict(dict))

    for exper_dir, data in orig_expers.items():
      red_exper_dir = "_".join(exper_dir.split("_")[1:])
      for mission, mission_results in data.items():
        red_mission_results = expers[red_exper_dir][mission]
        for loop_idx, values in mission_results.items():
          if not loop_idx in red_mission_results:
            red_mission_results[loop_idx] = values
          else:
            red_mission_results[loop_idx] += values

  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator

  # fig, ax1 = plt.subplots(figsize=(3.2,3))
  fig, ax1 = plt.subplots(figsize=(6,5))
  color_cycle = ax1._get_lines.prop_cycler
  handles = []

  for exper_dir,data in expers.items():
    col = next(color_cycle)['color']

    for mission, mission_results in data.items():
      # if mission == "test":
      #  continue

      Xs,Ys = zip(*sorted(mission_results.items()))
      mean_Ys = rowwise_mean(Ys)
      std_Ys  = rowwise_std(Ys)
      counts  = [len(row) for row in Ys]
      stderr_Ys = [s / c**0.5 for s, c in zip(std_Ys, counts)]
      ci95 = [1.96 * se for se in stderr_Ys]

      h, = ax1.plot(Xs, mean_Ys, STYLES[mission], linewidth = 1, label = exper_dir, color=col)
      if mission == "train":
        ax1.fill_between(Xs,[m - c for m, c in zip(mean_Ys, ci95)],[m + c for m, c in zip(mean_Ys, ci95)],color=col,alpha=0.25,linewidth=0)

      if mission == "train":
        handles.append(h)

  # Apply integer-only ticks to X-axis
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

  # ax1.set_xlim(xmin=0,xmax=24)
  ax1.set_ylim(ymin=0.5)
  # ax1.axhline(y=0.5386, color='gray', linestyle='--', linewidth=0.5) # for freshQuarter
  # ax1.axhline(y=0.5493, color='gray', linestyle='--', linewidth=0.5) # for rms_baseTraceSet_i16K

  plt.xlabel("improvement loop iteration")
  plt.ylabel(f"percentage problems proven")

  plt.legend(handles = handles, loc='lower right') # loc = 'best' is rumored to be unpredictable
  plt.savefig("current_plot.pdf".format("+".join(os.path.basename(dir) for dir in sys.argv[1:])),format="pdf", bbox_inches="tight")
  plt.close(fig)





