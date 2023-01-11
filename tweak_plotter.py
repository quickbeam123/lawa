#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

import inf_common as IC
import hyperparams as HP

import torch

if __name__ == "__main__":
  # Load a model (loop model optimizer) and a trace and 
  #
  # To be called as in: ./tweak_plotter.py tmp/loop-model-and-optimizer.tar tmp/test_Problems_GRP_GRP073-1.p_0.0.pt

  (loop,model_state_dict,optimizer_state_dict) = torch.load(sys.argv[1])

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  print("Loaded some model of loop",loop)

  proof_tuple = torch.load(sys.argv[2])
  print("Loaded proof_tuple with",len(proof_tuple[0]),"clauses and",len(proof_tuple[1]),"journal steps")

  prob_tweak = None

  if len(sys.argv) > 3:
    tweak_map = torch.load(sys.argv[3])

    # e.g. train_Problems_COL_COL042-6.p_1.0.pt
    trace_name = sys.argv[2]
    prob_name = "/".join(trace_name.split("_")[1:-1])

    prob_tweak = tweak_map[prob_name]

    print(f"Loaded prob's ({prob_name}) tweak: {prob_tweak}")

  learn_model = IC.LearningModel(model,*proof_tuple)
  learn_model.eval()

  import numpy as np

  AXLEN = 0.01
  STEPS = 25

  dx, dy = AXLEN/STEPS, AXLEN/STEPS

  x = np.arange(-AXLEN, AXLEN+dx, dx)
  y = np.arange(-AXLEN, AXLEN+dy, dy)
  X, Y = np.meshgrid(x, y)

  def get_loss(Xs, Ys):
    with torch.no_grad():
      stack_of_tweaks = np.column_stack((Xs.ravel(),Ys.ravel()))
      losses = learn_model.forward(stack_of_tweaks)
      return losses.reshape(Xs.shape)

  extent = np.min(x), np.max(x), np.min(y), np.max(y)

  Z = get_loss(X, Y)

  from matplotlib import pyplot as plt
  plt.imshow(Z, cmap=plt.cm.hsv, interpolation='nearest',extent=extent, origin='lower')

  if prob_tweak is not None:
    plt.plot([prob_tweak[0]],[prob_tweak[1]],marker='o')

  plt.colorbar()
  plt.savefig("tweak_map_{}.png".format(os.path.basename(sys.argv[2])),dpi=250)

