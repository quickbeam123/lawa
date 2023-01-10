#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

import torch

if __name__ == "__main__":
  # Load a tweak map and do some plotting
  #
  # To be called as in: ./tweak_analyzer.py tweak_map.pt

  tweak_map = torch.load(sys.argv[1])
  Xs = []
  Ys = []
  Ls = []
  for prob,(x,y) in tweak_map.items():
    if (x,y) != (0.0,0.0):
      Xs.append(x)
      Ys.append(y)
      Ls.append(prob.split("/")[-1][:-2])

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.scatter(Xs, Ys)

  for i, l in enumerate(Ls):
    ax.annotate(l, (Xs[i], Ys[i]))

  plt.show()