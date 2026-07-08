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

  PLOT_WHATS = [(0,"depth"),(1,"width")]
  PLOT_WHAT = 0

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

    if not isinstance(gage_stats, tuple):
      print("Failed to reproduce success for",prob,"Skipping!")
      continue

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

  # exit(0)

  gage_depth_sum = 0
  gage_depths = []
  gage_width_sum = 0
  gage_widths = []
  gage_depth_max = 0
  gage_width_max = 0
  gweight_depth_sum = 0
  gweight_depths = []
  gweight_width_sum = 0
  gweight_widths = []
  gweight_depth_max = 0
  gweight_width_max = 0
  count = 0

  for prob, records in stats.items():
    assert len(records) == 1
    num_selections, num_good_selections, gage_stats, gweight_stats = records[0]

    if not isinstance(gage_stats, tuple):
      print("Failed to reproduce success for",prob,"Skipping!")
      continue

    gage_depth = gage_stats[0]
    gage_width = gage_stats[1]
    gweight_depth = gweight_stats[0]
    gweight_width = gweight_stats[1]

    """
    if gage_depth > 1000 or gweight_depth > 1000:
      print(f"Skipping extreme {prob} with {(gage_stats, gweight_stats)}")
      continue
    """

    gage_depths.append(gage_depth)
    gweight_depths.append(gweight_depth)

    gage_depth_sum += gage_depth
    gage_width_sum += gage_width
    gage_widths.append(gage_width)
    gage_depth_max = max(gage_depth_max, gage_depth)
    gage_width_max = max(gage_width_max, gage_width)
    gweight_depth_sum += gweight_depth
    gweight_width_sum += gweight_width
    gweight_widths.append(gweight_width)
    gweight_depth_max = max(gweight_depth_max, gweight_depth)
    gweight_width_max = max(gweight_width_max, gweight_width)
    count += 1

  print(f"gage_depth_avg: {gage_depth_sum/count}")
  print(f"gage_depth_median: {sorted(gage_depths)[len(gage_depths)//2]}")
  print(f"gage_depth_max: {gage_depth_max}")

  print(f"gage_width_avg: {gage_width_sum/count}")
  print(f"gage_width_median: {sorted(gage_widths)[len(gage_widths)//2]}")
  print(f"gage_width_max: {gage_width_max}")

  print(f"gweight_depth_avg: {gweight_depth_sum/count}")
  print(f"gweight_depth_median: {sorted(gweight_depths)[len(gweight_depths)//2]}")
  print(f"gweight_depth_max: {gweight_depth_max}")

  print(f"gweight_width_avg: {gweight_width_sum/count}")
  print(f"gweight_width_median: {sorted(gweight_widths)[len(gweight_widths)//2]}")
  print(f"gweight_width_max: {gweight_width_max}")

  # exit(0)

  import matplotlib.pyplot as plt
  import numpy as np

  if True:
    TITLE = "neurally-guided (iter. 2)"
    COLORS = ["red", "red", "red", "red"]
  else:
    TITLE = "default strategy (iter. 1)"
    COLORS = ["blue", "blue", "blue", "blue"]

  COLOR_THEMES = {
    "blue": {"edge": "#1f77b4", "face": "#aec7e8", "median": "#0b3d91"},
    "red":  {"edge": "#d62728", "face": "#f4a582", "median": "#8b0000"},
  }

  fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5))

  for ax, data, title, max_exp, color in zip(axes.flat,
      [gage_depths, gage_widths, gweight_depths, gweight_widths],
      ["gage depth", "gage width", "gweight depth", "gweight width"],
      [5, 6, 5, 6],
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
    ticks = list(range(0, max_exp + 1))
    ax.set_yticks(ticks)
    ax.set_ylim(0, max_exp)
    ax.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_title(title)

  fig.suptitle(TITLE)
  plt.tight_layout()
  plt.savefig("stat_violin.pdf", format="pdf", bbox_inches="tight")
  plt.close(fig)

