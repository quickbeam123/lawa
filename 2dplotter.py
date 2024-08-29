#!/usr/bin/env python3

from re import L
import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]

# Age,Weight                     1,2
# pLen,nLen                      3,4
# justEq, justNeq                5,6
# numVarOcc,VarOcc/W             7,8
# Sine0,SineMax,SineLevel,   9,10,11
# numSplits                       12

DIMS = ["age","weight","pLen","nLen","justEq","justNeq","numVarOcc","VarOcc/W","Sine0","SineMax","SineLeve","numSplits"]

if __name__ == "__main__":
  # Plot the age-weight -> logit graph for a trained model with NUM_FEATURES == 2
  #
  # To be called as in: ./2dplotter.py loop_folder_of_an_exper_where_it_all_can_be_found

  clause_hist = defaultdict(int)
  ftr_hist = defaultdict(lambda : defaultdict(int))

  index = torch.load(os.path.join(sys.argv[1],"trace-index.pt"))
  for prob,trace_file_paths in index.items():
    proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]
    for proof_tuple in proof_tuples:
      clause_features,newjournal,num_good_selections = proof_tuple
      for fv in clause_features:
        clause_hist[tuple(fv)] += 1
        for f,dim in zip(fv,DIMS):
          ftr_hist[dim][f] += 1

  if True:
    for dim,dim_hist in ftr_hist.items():
      print(dim,len(dim_hist),min(dim_hist),max(dim_hist))
      i = 0
      for val,cnt in sorted(dim_hist.items(),key=lambda x : -x[1]):
        print("  ",val,cnt)
        i += 1
        if i > 25:
          break

  common_fv = None
  for fv,cnt in sorted(clause_hist.items(),key=lambda x : -x[1]):
    if common_fv is None:
      common_fv = fv
    if False:
      print(fv,cnt)
    else:
      break

  assert common_fv
  print()
  print("common_fv",common_fv)

  # load a model
  model = torch.jit.load(os.path.join(sys.argv[1],"script-model.pt"))

  from matplotlib import pyplot as plt

  def addToPlot(what,idx,low,high,div=1.0):
    Xs = []
    samples = []
    for fea in range(low,high):
      x = fea/div
      Xs.append(x)
      a_clause = list(common_fv)
      a_clause[idx] = x
      samples.append(torch.tensor(a_clause))
    samples = torch.stack(samples)
    results = torch.squeeze(model.forward(samples),-1)

    plt.plot(Xs,results,label=what)

  def saveIt(what):
    plt.legend()
    spl = sys.argv[1].split("/")
    plt.savefig(f"{what}_{spl[-2]}_{spl[-1]}.png",dpi=300)
    plt.close()

  def plotIt(what,idx,low,high,div=1.0):
    addToPlot(what,idx,low,high,div)
    saveIt(what)

  # plotIt("age",0,0,30)
  # plotIt("weight",1,1,100)
  # plotIt("pLen",2,0,15)
  # plotIt("nLen",3,0,15)
  # plotIt("justEq",4,0,2)
  # plotIt("justNeq",5,0,2)
  # plotIt("numVarOcc",6,0,100)
  # plotIt("VarOccW",7,0,25,35)

  # plotIt("SineLevel",10,0,101,100)
  # plotIt("splits",11,0,25)

  save_common_fv = common_fv

  for eq in [0,1]:
    for neq in [0,1]:
      temp = list(save_common_fv)
      temp[0] = 5.0 # hardcoding age (rather than 0) here
      temp[4] = float(eq)
      temp[5] = float(neq)
      common_fv = tuple(temp)
      addToPlot(f"eq-{eq},neq-{neq}",1,0,30)
  saveIt("weight")

  exit(0)

  import numpy as np
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  def plotIt2D(whatX,idxX,lowX,highX,dX,whatY,idxY,lowY,highY,dY):
    x = np.arange(lowX, highX, dX)
    y = np.arange(lowY, highY, dY)
    X, Y = np.meshgrid(x, y)

    def get_logit(X, Y):
      with torch.no_grad():
        noodleX = X.ravel()
        noodleY = Y.ravel()
        copies = np.tile(np.array(common_fv), (len(noodleX), 1))

        copies[:, idxX] = noodleX
        copies[:, idxY] = noodleY

        logits = model(torch.tensor(copies).float())

        return logits.reshape(X.shape)

    extent = np.min(x)-0.5, np.max(x)+0.5, np.min(y)-0.5, np.max(y)+0.5

    Z = get_logit(X, Y)

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()
    ax.set_xlabel(whatX)
    ax.set_ylabel(whatY)
    img = ax.imshow(Z, cmap=plt.cm.hsv, interpolation='none',extent=extent,aspect='auto', origin='lower')
    # img = plt.contourf(Z,levels = [-20.0,-15.0,-10.0,-5.0,0.0],extent=extent, origin='lower')

    # plt.locator_params(axis='x', nbins=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    # cbar.set_label('computed logit', rotation=270)

    fig.tight_layout()

    spl = sys.argv[1].split("/")
    plt.savefig(f"{whatX}-{whatY}_map_{spl[-2]}_{spl[-1]}.png",dpi=300)
    plt.close()

  # plotIt2D("age",0,0.0,30.0,0.05,"weight",1,1.0,30.0,0.05)
  # plotIt2D("pLen",2,0.0,15.0,0.05,"nLen",3,1.0,15.0,0.05)
  # plotIt2D("numVarOcc",6,0.0,10.0,0.1,"splits",11,0.0,10.0,0.1)

  exit(0)

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