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
  PLOT_WHAT = 0

  stats = torch.load(sys.argv[1])

  Xs = []
  Ys = []

  for prob, records in stats.items():
    assert len(records) == 1
    gage_stats, gweight_stats = records[0]

    if gage_stats[PLOT_WHATS[PLOT_WHAT][0]] > 500 or gweight_stats[PLOT_WHATS[PLOT_WHAT][0]] > 500:
      print(f"Skipping extreme {prob} with {(gage_stats, gweight_stats)}")
    else:
      Xs.append(gage_stats[PLOT_WHATS[PLOT_WHAT][0]])
      Ys.append(gweight_stats[PLOT_WHATS[PLOT_WHAT][0]])

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

