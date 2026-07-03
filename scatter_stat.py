#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

def get_info(probinfo,prob):
  if prob not in probinfo:
    return 0.0
  return probinfo[prob.split("/")[-1]]

if __name__ == "__main__":
  # Scatter plot of gage/gweight stats as collected during looping.
  #
  # call as in, e.g.: ./scatter_stat.py deleteme/loop1/stats.pt

  with open("/nfs/sudamar2/TPTP-v9.0.0/probinfo9.0.0.pkl",'rb') as f:
    probinfo = pickle.load(f)

  PLOT_WHATS = [(0,"height"),(1,"width")]
  PLOT_WHAT = 1

  stats = torch.load(sys.argv[1],weights_only=False)

  total = 0
  trivial = 0
  full = 0
  degenerate = 0
  partial = 0

  total_num_selections = 0
  total_num_good_selections = 0

  for prob, records in stats.items():
    assert len(records) == 1
    num_selections, num_good_selections, gage_stats, gweight_stats = records[0]

    total += 1
    if num_selections == 0:
      trivial += 1
    elif num_selections == num_good_selections:
      full += 1
    elif num_good_selections == 0:
      degenerate += 1
    else:
      partial += 1

    total_num_selections += num_selections
    total_num_good_selections += num_good_selections

  print("total",total)
  print("trivial",trivial)
  print("full",full)
  print("degenerate",degenerate)
  print("partial",partial)
  print()

  print("total_num_good_selections",total_num_good_selections)
  print("total_num_selections",total_num_selections)
  print("ratio",total_num_good_selections/total_num_selections)

  exit(0)

  Xs = []
  Ys = []

  gage_height_sum = 0
  gage_heights = []
  gage_width_sum = 0
  gage_height_max = 0
  gweight_height_sum = 0
  gweight_heights = []
  gweight_width_sum = 0
  gweight_height_max = 0
  count = 0

  for prob, records in stats.items():
    assert len(records) == 1
    num_selections, num_good_selections, gage_stats, gweight_stats = records[0]

    gage_height = gage_stats[0]
    gage_width = gage_stats[1]
    gweight_height = gweight_stats[0]
    gweight_width = gweight_stats[1]

    gage_heights.append(gage_height)
    gweight_heights.append(gweight_height)

    gage_height_sum += gage_height
    gage_width_sum += gage_width
    gage_height_max = max(gage_height_max, gage_height)
    gweight_height_sum += gweight_height
    gweight_width_sum += gweight_width
    gweight_height_max = max(gweight_height_max, gweight_height)
    count += 1

    if gage_stats[PLOT_WHATS[PLOT_WHAT][0]] > 500 or gweight_stats[PLOT_WHATS[PLOT_WHAT][0]] > 500:
      print(f"Skipping extreme {prob} with {(gage_stats, gweight_stats)}")
    else:
      Xs.append(gage_stats[PLOT_WHATS[PLOT_WHAT][0]])
      Ys.append(gweight_stats[PLOT_WHATS[PLOT_WHAT][0]])

  print(f"gage_height_avg: {gage_height_sum/count}")
  print(f"gage_height_median: {sorted(gage_heights)[len(gage_heights)//2]}")
  print(f"gage_width_avg: {gage_width_sum/count}")
  print(f"gage_height_max: {gage_height_max}")
  print(f"gweight_height_avg: {gweight_height_sum/count}")
  print(f"gweight_height_median: {sorted(gweight_heights)[len(gweight_heights)//2]}")
  print(f"gweight_width_avg: {gweight_width_sum/count}")
  print(f"gweight_height_max: {gweight_height_max}")

  exit(0)

  import matplotlib.pyplot as plt

  fig, ax1 = plt.subplots(figsize=(6,6))

  ax1.scatter(Xs,Ys,s=1)

  plt.xlabel(f"gage {PLOT_WHATS[PLOT_WHAT][1]}")
  plt.ylabel(f"gweight {PLOT_WHATS[PLOT_WHAT][1]}")

  # both axis in log scale
  # ax1.set_xscale('log')
  # ax1.set_yscale('log')

  # same maximal value for both axes
  # max_val = max(max(Xs), max(Ys))
  # ax1.set_xlim([1, 10000])
  # ax1.set_ylim([1, 10000])

  plt.savefig("stat_scatter.pdf",format="pdf", bbox_inches="tight")
  plt.close(fig)

