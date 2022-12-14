#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

import torch
import numpy as np

if __name__ == "__main__":
  # Load a tweak map and do some plotting
  #
  # To be called as in: ./tweak_analyzer.py tweak_map.pt

  x_idx = 0
  y_idx = 1

  with open("probinfo7.5.0.pkl",'rb') as f:
    probinfo = pickle.load(f,encoding='utf-8')

  tweak_map = torch.load(sys.argv[1])
  Xs = []
  Ys = []
  Ls = []
  Zs = [] # problem TPTP ratings

  for prob,tweak in tweak_map.items():
    if np.linalg.norm(tweak) > 0.000001:
      Xs.append(tweak[x_idx])
      Ys.append(tweak[y_idx])
      probname = prob.split("/")[-1]
      Ls.append(probname[:-2])

      info = probinfo[probname]
      Zs.append(info[0])
      if info[0] > 0.9:
        print(probname,info[0])

  import matplotlib.pyplot as plt
  cm = plt.cm.get_cmap('RdYlBu_r')
  fig, ax = plt.subplots()
  sc = ax.scatter(Xs, Ys, c= Zs, vmin=0, vmax=1, s=35, cmap=cm)
  plt.colorbar(sc)

  for i, l in enumerate(Ls):
    ax.annotate(l, (Xs[i], Ys[i]))

  plt.show()