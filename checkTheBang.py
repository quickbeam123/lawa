#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]
STYLES = { MISSIONS[0] : "-", MISSIONS[1] : "--"}

MAXINT = 2**32

def get_status(info):
  if isinstance(info,IC.VampResult):
    return info.status
  else:
    return info[0]

if __name__ == "__main__":
  # Report newly (with regards to loop iterations) solving hard (TPTP rating 1.0) problems
  #
  # To be called as in: ./checkTheBang.py exper_folder1

  with open("/nfs/sudamar2/TPTP-v9.0.0/probinfo9.0.0.pkl",'rb') as f:
    probinfo = pickle.load(f)

  total = 0
  covered = set()

  for exper_dir in sys.argv[1:]:
    print(exper_dir)
    root, dirs, files = next(os.walk(exper_dir))
    min_loop = MAXINT
    max_loop = 0
    for dir in dirs:
      if dir.startswith("loop"):
        dirs_loop = int(dir[4:])
        if dirs_loop < min_loop:
          min_loop = dirs_loop
        if dirs_loop > max_loop:
          max_loop = dirs_loop
    # loop,update = max_loop,-1
    loop,update = min_loop,1
    while True:
      loop_str = "loop{}".format(loop)
      cur_dir = os.path.join(exper_dir,loop_str)
      if not os.path.isdir(cur_dir):
        break

      print("  loop",loop)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file in ["stats.pt","tweak_map.pt","train_data.pt","train_storage.pt","parts-model.pt","after-train-tweak_map.pt",
                    "script-model.pt","script-model-after.pt","optimizer.pt","parts-model-state.tar","optimizer-state.tar","loop-model-and-optimizer.tar","trace-index.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file))
        # print("      ",meta)

        for prob,runs in results.items():
          for (i,vr) in runs:
            if get_status(vr) == "uns":
              if prob not in covered:
                covered.add(prob)
                info = probinfo[prob.split("/")[-1]]
                rate = float(info[0]) if info[0] is not None else 0.5 # whatever
                if rate > 0.99:
                  total += 1
                  print("      ",i,prob,total,rate,vr)

      loop += update


