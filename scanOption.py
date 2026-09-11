#!/usr/bin/env python3

import math, random
from collections import defaultdict
import subprocess

import numpy as np

import matplotlib.pyplot as plt

import workers as W

from multiprocessing import Pool

class ProbSolve:
  def __init__(self):
    self.succ = 0
    self.cnt = 0

  def add(self,val):
    self.cnt += 1
    if (val is not None):
      self.succ += 1

  def prob(self):
    return self.succ/self.cnt

def prepare_one(task):
  val,seed = task

  base_with_val = base_strat.format(val)

  res = W.vampire_perfrom(prob_name,f"-t 60 {base_with_val} {shuffle_opts} --random_seed {seed} -i {ilim}",None)

  return val,seed,res.status,res.instructions

if __name__ == "__main__":
  # to be run as in: ./scanFDI.py

  option = "npcct"

  vampire = "./vampire_rel_mtpa-gnn-2026_10736"
  ilim = 16000
  syntax = "tptp"
  base_strat = f"-npcc on -ncem /home/sudamar2/jar2026/seed42_nd_noSplitC/loop28/script-model.pt -{option} {{}}"
  shuffle_opts = "-si on -rtra on"
  prob_name = "Problems/SCT/SCT114+1.p"
  prob_name_short = prob_name.split("/")[-1]
  width = 200

  tasks = []
  for val_iter in range(0,width):
    val = val_iter / 100
    for _ in range(40): # for now just one sample per iter
      tasks.append((val,random.randint(1,0x7ffffffff)))

  pool = Pool(processes=128)
  results = pool.map(prepare_one, tasks, chunksize = 1)

  buckets = defaultdict(list)
  probsolve = defaultdict(ProbSolve)

  Xs = []
  Ys = []

  for val,seed,result,instructions in results:
    assert result != "sat", f"Found saturation for {val} {seed}"
    print(val,seed,result,instructions)

    # instructions = math.log(instructions*1000000,10)

    if result is not None:
      Xs.append(val)
      Ys.append(instructions)

    buckets[val].append(instructions)
    probsolve[val].add(result)

  fig, ax1 = plt.subplots(figsize=(12, 6))
  hb = plt.hexbin(Xs,Ys,reduce_C_function=np.sum, bins='log',linewidths=0.1,gridsize=(width,50))
  cb = fig.colorbar(hb, ax=ax1)
  cb.set_label('numhits')

  XYZPs = []

  for val,iter_list in buckets.items():
    x = val
    y = np.mean(iter_list)
    z = np.std(iter_list)
    p = probsolve[val].prob()

    XYZPs.append((x,y,z,p))

  XYZPs.sort()

  Xs = np.array([x for (x,y,z,p) in XYZPs])
  Ys = np.array([y for (x,y,z,p) in XYZPs])
  Zs = np.array([z for (x,y,z,p) in XYZPs])
  Ps = np.array([p for (x,y,z,p) in XYZPs])

  ax1.plot(Xs, Ys, "-", label = "average", zorder=3, color = "deeppink", linewidth = 1.0)
  # ax1.fill_between(Xs, Ys-Zs, Ys+Zs, zorder=2, facecolor="blue", alpha=0.5)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('prob of solving', color=color)
  ax2.plot(Xs, Ps, color=color, linewidth = 1.0)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  ax1.set_xlabel(f'{option}')
  ax1.set_ylabel(r'$\log_{10}(\mathrm{instructions})$')

  plt.subplots_adjust(bottom=0.2,left=0.2)

  plt.savefig(f"{prob_name_short}.{option}.pdf",format="pdf", bbox_inches="tight")
