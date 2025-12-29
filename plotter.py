#!/usr/bin/env python3

import workers as W
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
        if file not in ["train_res.pt","test_res.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file))
        # print("      ",meta)

        sample_record = next(iter(results.values()))[0]
        if len(sample_record) == 3:
          if isinstance(sample_record[2],W.VampResult):
            fractional = sum(1.0 for prob,runs in results.items() for (i,ilim,info) in runs if (i == 0 and info.status == "uns"))
          else:
            fractional = sum(1/len(runs) for prob,runs in results.items() for (status,instructions,activations) in runs if status == "uns")
        else:
          # started using NUM_PERFORMS with different params; only the 0-labeled run, however, counts
          fractional = sum(1.0 for prob,runs in results.items() for (i,info) in runs if (i == 0 and get_status(info) == "uns"))

        fractional /= len(results.items())

        print(fractional)

        """
        for prob,runs in results.items():
          for (i,info) in runs:
            if info.status == None and info.instructions >= 10000:
            print(prob,info)
        """

        # print("     -> ",successes)
        for m in MISSIONS:
          if file.startswith(m):
            plottables[m][0].append(loop) # NOTE: add -1 here to get the plots starting at 0, like for CADE
            plottables[m][1].append(fractional)

      loop += 1

    expers[exper_dir] = plottables

  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator

  # fig, ax1 = plt.subplots(figsize=(3.2,3))
  fig, ax1 = plt.subplots(figsize=(6,5))
  color_cycle = ax1._get_lines.prop_cycler
  handles = []

  common_prefix = os.path.commonprefix(list(expers.keys()))

  REPLACE = {"/home/sudamar2/mtpa-gnn/newSplit30k":"base",
             "/home/sudamar2/mtpa-gnn/newSplit30k-noImit": "noImit",
             "/home/sudamar2/mtpa-gnn/newSplit30k-cumul-np5": "boost",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul": "base10k",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noGage": "noGage",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noGweight": "noGweight",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noSF": "noSF",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justGage": "justGage",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justGweight": "justGweight",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justSF": "justSF",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-is128": "m=128",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-es48": "n=48",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-es16": "m=16",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl4": "k=4",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl8": "k=8",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl16": "k=16",
             }

  for exper_dir,plottables in expers.items():
    col = next(color_cycle)['color']

    for m,(Xs,Ys) in plottables.items():
      if Xs:
        print(exper_dir)
        if exper_dir in REPLACE:
          lab = REPLACE[exper_dir]
        else:
          lab = exper_dir[len(common_prefix)-2:]+"_"+m

        h, = ax1.plot(Xs, Ys, STYLES[m], linewidth = 1, label = lab, color=col)
        if m == "train":
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

  # Apply integer-only ticks to X-axis
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

  # ax1.set_xlim(xmin=0,xmax=24)
  # ax1.set_ylim(ymin=0.42,ymax=0.54)
  # ax1.axhline(y=0.5386, color='gray', linestyle='--', linewidth=0.5) # for freshQuarter
  ax1.axhline(y=0.5294, color='gray', linestyle='--', linewidth=0.5) # for freshFull

  plt.xlabel("improvement loop iteration")
  plt.ylabel(f"percentage problems proven")

  plt.legend(handles = handles, loc='lower right') # loc = 'best' is rumored to be unpredictable
  plt.savefig("current_plot.pdf".format("+".join(os.path.basename(dir) for dir in sys.argv[1:])),format="pdf", bbox_inches="tight")
  plt.close(fig)





