#!/usr/bin/env python3

from re import L
import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]

if __name__ == "__main__":
  # Plot the age-weight -> logit graph for a trained model with FEATURE_SUBSET == FEATURES_AW
  #
  # To be called as in: ./2dplotter.py parts-model.pt (index of the effective queue if > 0 is wished)

  model = IC.get_initial_model()
  model.load_state_dict(torch.load(sys.argv[1]))
  if len(sys.argv) > 2:
    idx = int(sys.argv[2])
  else:
    idx = 0

  # TODO: index does not work yet

  clause_embedder,clause_keys = model

  import numpy as np
  dx, dy = 0.05, 0.05

  x = np.arange(0.0, 100.05, dx)
  y = np.arange(0.0, 120.05, dy)
  X, Y = np.meshgrid(x, y)

  def get_logit(age, weight):
    with torch.no_grad():
      features = torch.tensor(np.column_stack((age.ravel(),weight.ravel()))).float()
      embeddings = clause_embedder(features)
      logits = torch.matmul(clause_keys.weight,torch.t(embeddings))

      return logits.reshape(age.shape)

  extent = np.min(x), np.max(x), np.min(y), np.max(y)

  Z = get_logit(X, Y)

  from matplotlib import pyplot as plt
  img = plt.imshow(Z, cmap=plt.cm.hsv, interpolation='nearest',extent=extent, origin='lower')

  plt.colorbar()
  plt.savefig("aw_map.png",dpi=250)

