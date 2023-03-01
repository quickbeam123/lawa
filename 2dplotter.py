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
  dx, dy = 1.0, 1.0

  x = np.arange(0.0, 31.00, dx)
  y = np.arange(0.0, 31.00, dy)
  X, Y = np.meshgrid(x, y)

  def get_logit(age, weight):
    with torch.no_grad():
      features = np.column_stack((age.ravel(),weight.ravel()))
      # features = np.pad(features,((0,0),(0,HP.NUM_FEATURES-2)),'constant')

      logits = model(torch.tensor(features).float())

      return logits.reshape(age.shape)

  extent = np.min(x)-0.5, np.max(x)+0.5, np.min(y)-0.5, np.max(y)+0.5

  Z = get_logit(X, Y)

  from matplotlib import pyplot as plt
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  fig = plt.figure(figsize=(3.5, 3.5))
  ax = plt.gca()
  # fig2=plt.figure(figsize=(2.5, 2.0))
  ax.set_xlabel('age')
  ax.set_ylabel('weigth')
  img = ax.imshow(Z, cmap=plt.cm.hsv, interpolation='none',extent=extent, origin='lower')
  # img = plt.contourf(Z,levels = [-20.0,-15.0,-10.0,-5.0,0.0],extent=extent, origin='lower')

  # plt.locator_params(axis='x', nbins=3)

  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = plt.colorbar(img, cax=cax)
  # cbar.set_label('computed logit', rotation=270)

  fig.tight_layout()

  plt.savefig(f"aw_map_loop{loop}.png",dpi=300)
  plt.close()

  fig = plt.figure(figsize=(2.9, 2.9))
  ax = plt.gca()
  ax.set_xlabel('age')
  ax.set_ylabel('logit')
  for i in [1,3,6,9]:
    plt.plot(X[i],Z[i],label=f"w={int(Y[i][0])}")
  plt.legend(loc="upper right",ncol=2,columnspacing=0.8)
  fig.tight_layout()
  plt.savefig(f"horiz_loop{loop}.png",dpi=300)
  plt.close()

  fig = plt.figure(figsize=(3.5, 3.5))
  ax = plt.gca()
  ax.set_xlabel('weight')
  ax.set_ylabel('logit')
  for i in [0,1,2,4,8]:
    plt.plot(Y[:,i],Z[:,i],label=f"a={int(X[:,i][0])}")
  plt.legend(loc="lower left")
  ax.axvline(3,ymin=0.5, ymax=0.98,ls='--', lw=1,color="gray")
  fig.tight_layout()
  plt.savefig(f"verti_loop{loop}.png",dpi=300)
  plt.close()