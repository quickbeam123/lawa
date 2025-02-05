#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]
STYLES = { MISSIONS[0] : "-", MISSIONS[1] : "--"}

MAXINT = 2**32

def get_status(info):
  if isinstance(info,IC.VampResult):
    return info.status
  else:
    return info[0]

if __name__ == "__main__":
  # Plotting some training curves, automatically getting the data from the exper directories left behind by looper
  #
  # To be called as in: ./plotter.py exper_folder1 exper_folder2 ...

  # for each evaluation mode found (e.g. "train_t0.25.pt"),
  # keep storing pairs (solutions,time)
  expers = {}

  for exper_dir in sys.argv[1:]:
    # print(exper_dir)

    plottables = {m : ([],[]) for m in MISSIONS}

    root, dirs, files = next(os.walk(exper_dir))
    loop = MAXINT
    for dir in dirs:
      if dir.startswith("loop"):
        dirs_loop = int(dir[4:])
        if dirs_loop < loop:
          loop = dirs_loop
    while True:
      loop_str = "loop{}".format(loop)
      cur_dir = os.path.join(exper_dir,loop_str)
      if not os.path.isdir(cur_dir):
        break

      # print("  ",cur_dir)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file in ["stats.pt","tweak_map.pt","train_data.pt","train_storage.pt","parts-model.pt","after-train-tweak_map.pt",
                    "script-model.pt","script-model-after.pt","optimizer.pt","parts-model-state.tar","optimizer-state.tar","loop-model-and-optimizer.tar","trace-index.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file))
        # print("      ",meta)

        if len(next(iter(results.values()))[0]) == 3:
          fractional = sum(1/len(runs) for prob,runs in results.items() for (status,instructions,activations) in runs if status == "uns")
        else:
          # started using NUM_PERFORMS with different params; only the 0-labeled run, however, counts
          fractional = sum(1.0 for prob,runs in results.items() for (i,info) in runs if (i == 0 and get_status(info) == "uns"))

        fractional /= len(results.items())

        # print("     -> ",successes)
        for m in MISSIONS:
          if file.startswith(m):
            plottables[m][0].append(loop)
            plottables[m][1].append(fractional)

      loop += 1

    expers[exper_dir] = plottables

  import matplotlib.pyplot as plt

  fig, ax1 = plt.subplots(figsize=(8,6))
  color_cycle = ax1._get_lines.prop_cycler
  handles = []

  common_prefix = os.path.commonprefix(list(expers.keys()))

  for exper_dir,plottables in expers.items():
    col = next(color_cycle)['color']

    for m,(Xs,Ys) in plottables.items():
      if Xs:
        h, = ax1.plot(Xs, Ys, STYLES[m], linewidth = 1, label = exper_dir[len(common_prefix)-1:]+"_"+m, color=col)
        handles.append(h)

        max_val,max_idx = (0.0,0)
        imax_val,imax_idx = (0.0,0)
        for idx,val in zip(Xs,Ys):
          if val > max_val:
            max_val = val
            max_idx = idx
          if idx > 1 and val > imax_val:
            imax_val = val
            imax_idx = idx

        print(exper_dir,m,"max with",max_val,"at",max_idx)
        print("Also imax with",imax_val,"at",imax_idx)

  plt.legend(handles = handles, loc='lower right') # loc = 'best' is rumored to be unpredictable
  plt.savefig("{}_plot.pdf".format("+".join(os.path.basename(dir) for dir in sys.argv[1:])),format="pdf", bbox_inches="tight")
  plt.close(fig)





