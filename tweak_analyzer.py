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
  z_idx = 2

  is3d = False

  with open("probinfo8.2.0.pkl",'rb') as f:
    probinfo = pickle.load(f,encoding='utf-8')

  tweak_map = torch.load(sys.argv[1])

  print(np.std(list(tweak_map.values()),axis=0))

  Xs = []
  Ys = []
  Zs = []
  Ls = []
  Rs = [] # problem TPTP ratings

  for prob,tweak in tweak_map.items():
    probname = prob.split("/")[-1]
    info = probinfo[probname]
    #if info[0] < 0.8:
    #  continue

    if np.linalg.norm(tweak) > 0.000001:
      Xs.append(tweak[x_idx])
      if y_idx < len(tweak):
        Ys.append(tweak[y_idx])
      else:
        Ys.append(tweak[x_idx])
      if is3d:
        if z_idx < len(tweak):
          Zs.append(tweak[z_idx])
        else:
          Zs.append(0.0)
      Ls.append(probname[:-2])

      Rs.append(info[0])
      if info[0] > 0.9:
        print(probname,info[0])

  import matplotlib.pyplot as plt
  cm = plt.colormaps['RdYlBu_r']
  fig = plt.figure()
  if is3d:
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(Xs, Ys, Zs, c= Rs, vmin=0, vmax=1, s=5, cmap=cm)
  else:
    ax = fig.add_subplot()
    sc = ax.scatter(Xs, Ys, c= Rs, vmin=0, vmax=1, s=5, cmap=cm)

  plt.colorbar(sc)

  # show TPTP names
  if True:
    for i, l in enumerate(Ls):
      ax.annotate(l, (Xs[i], Ys[i]))

  plt.show()