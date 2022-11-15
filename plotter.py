#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]

if __name__ == "__main__":
  # Plotting some training curves, automatically getting the data from the exper directories left behind by looper
  #
  # To be called as in: ./plotter.py exper_folder1 exper_folder2 ...
  #
  # with multiple exper folders, these are "connected" sequentially (time-wise)

  # for each evaluation mode found (e.g. "train_t0.25.pt"),
  # keep storing pairs (solutions,time)
  expers = defaultdict(list)
  time = 0

  for exper_dir in sys.argv[1:]:
    print(exper_dir)
    loop = 0
    while True:
      loop_str = "loop{}".format(loop)
      cur_dir = os.path.join(exper_dir,loop_str)
      if not os.path.isdir(cur_dir):
        if loop == 0:
          loop += 1
          continue
        else:
          break

      print("  ",cur_dir)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file in ["train_data.pt","parts-model.pt","script-model.pt","optimizer.pt","parts-model-state.tar","optimizer-state.tar"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file))
        # print("      ",meta)

        successes = sum(1 for prob,(status,instructions,activations) in results.items() if status == "uns")

        # print("     -> ",successes)

        expers[file].append((successes,time))

      loop += 1
      time += 1
    # go one step back to seamlessly connect with the followup exper, if present
    time -= 1

  import matplotlib.pyplot as plt

  for mission in MISSIONS:
    fig, ax1 = plt.subplots()

    handles = []

    mission_max = 0
    mission_max_at = None

    for filename,run in expers.items():
      if not filename.startswith(mission):
        continue

      successes = []
      times = []
      for (s,t) in run:
        successes.append(s)
        times.append(t)
        if s > mission_max:
          mission_max = s
          mission_max_at = "iter{}_{}".format(t,filename)

      h, = ax1.plot(times, successes, "--", linewidth = 1, label = filename[:-3])

      handles.append(h)

    print(mission,"maxed with",mission_max,"at",mission_max_at)

    # ax1.set_ylim(ymin=0)

    plt.legend(handles = handles, loc='upper left') # loc = 'best' is rumored to be unpredictable

    plt.savefig("{}_{}_plot.png".format(os.path.basename(sys.argv[1]),mission),dpi=250)
    plt.close(fig)





