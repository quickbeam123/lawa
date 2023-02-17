#!/usr/bin/env python3

from re import L
import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]

if __name__ == "__main__":
  # Plot the age-weight -> logit graph for a trained model with NUM_FEATURES == 2
  #
  # To be called as in: ./2dplotter.py loop-model-and-optimizer.tar

  model = IC.get_initial_model()
  (loop,model_state_dict,optimizer_state_dict) = torch.load(sys.argv[1])
  model.load_state_dict(model_state_dict)

  import numpy as np
  dx, dy = 0.05, 0.05

  x = np.arange(-0.1, 20.05, dx)
  y = np.arange(-0.1, 20.05, dy)
  X, Y = np.meshgrid(x, y)

  def get_logit(age, weight):
    with torch.no_grad():
      features = np.column_stack((age.ravel(),weight.ravel()))
      features = np.pad(features,((0,0),(0,HP.NUM_FEATURES-2)),'constant')

      logits = model(torch.tensor(features).float())

      return logits.reshape(age.shape)

  extent = np.min(x), np.max(x), np.min(y), np.max(y)

  Z = get_logit(X, Y)

  from matplotlib import pyplot as plt
  # img = plt.imshow(Z, cmap=plt.cm.hsv, interpolation='nearest',extent=extent, origin='lower')
  img = plt.contourf(Z,extent=extent, origin='lower')

  plt.colorbar()
  plt.savefig(f"aw_map_loop{loop}.png",dpi=250)

