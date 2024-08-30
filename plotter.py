#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]

MAXINT = 2^32

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
        if file in ["tweak_map.pt","train_data.pt","train_storage.pt","parts-model.pt","after-train-tweak_map.pt",
                    "script-model.pt","script-model-after.pt","optimizer.pt","parts-model-state.tar","optimizer-state.tar","loop-model-and-optimizer.tar","trace-index.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file))
        # print("      ",meta)

        fractional = sum(1/len(runs) for prob,runs in results.items() for (status,instructions,activations) in runs if status == "uns")

        # print("     -> ",successes)
        for m in MISSIONS:
          if file.startswith(m):
            plottables[m][0].append(loop)
            plottables[m][1].append(fractional)

      loop += 1

    expers[exper_dir] = plottables

  import matplotlib.pyplot as plt

  for m in MISSIONS:
    fig, ax1 = plt.subplots()

    plotted = False
    handles = []

    for exper_dir,plottables in expers.items():
      Xs,Ys = plottables[m]
      if Xs:
        h, = ax1.plot(Xs, Ys, "--", linewidth = 1, label = exper_dir)
        handles.append(h)
        plotted = True

        max_val = 0.0
        max_idx = 0
        for idx,val in zip(Xs,Ys):
          if val > max_val:
            max_val = val
            max_idx = idx

        print(exper_dir,m,"max with",max_val,"at",max_idx)

    # ax1.set_ylim(ymin=0)

    if plotted:
      plt.legend(handles = handles, loc='lower right') # loc = 'best' is rumored to be unpredictable
      plt.ylim(1200, None)
      plt.savefig("{}_{}_plot.png".format("+".join(os.path.basename(dir) for dir in sys.argv[1:]),m),dpi=250)
    plt.close(fig)





