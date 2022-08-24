#!/usr/bin/env python3

import inf_common as IC

import hyperparams as HP

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

from multiprocessing import Pool

def load_one(task):
  i,logname = task
  
  print(i)
  start_time = time.time()
  result = IC.load_one(logname) # ,max_size=15000)
  print("Took",time.time()-start_time)
  if result:
    probdata,time_elapsed = result
    return (logname,time_elapsed),probdata
  else:
    None

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a set of logs, passed in a file as the final argument, each will become a "training example tree"
  # Scan the set and create a histogram (to learn which initial and derivational networks will be needed; including the "dropout" defaults)
  # - save it to "data_hist.pt"
  # normalize the training data and save that to "training_data.pt" / "validation_data.pt" using a 80:20 split
  #
  # To be called as in: ./log_loader.py <folder> *.log-files-listed-line-by-line-in-a-file (an "-s4k on" run of vampire)
  #
  # optionally, also a file with problem easyness line-by-line records, e.g. mizar_common/prob_easiness_strats1234.txt
  #
  # data_sign.pt and raw_log_data_*.pt are created in <folder>

  prob_easiness = {}
  if len(sys.argv) > 3:
    with open(sys.argv[3],"r") as f:
      for line in f:
        spl = line.split()
        prob_easiness[spl[0]] = int(spl[1])

  prob_data_list = [] # [(logname,(init,deriv,pars,selec,good)]

  if True: # parallel
    tasks = []
    with open(sys.argv[2],"r") as f:
      for i,line in enumerate(f):
        logname = line[:-1]
        tasks.append((i,logname))
    pool = Pool(processes=30) # number of cores to use
    results = pool.map(load_one, tasks, chunksize = 100)
    pool.close()
    pool.join()
    del pool
    prob_data_list = list(filter(None, results))
  else: # sequential
    prob_data_list = []
    with open(sys.argv[2],"r") as f:
      for i,line in enumerate(f):
        logname = line[:-1]
        result = load_one((i,logname))
        if result is not None:
          prob_data_list.append(result)

  print(len(prob_data_list),"problems loaded!")

  # assign weights to problems, especially if prob_easiness file has been provided
  times = []
  sizes = []
  easies = []
  for i,((logname,time_elapsed),probdata) in enumerate(prob_data_list):
    probname = IC.logname_to_probname(logname)
    easy = prob_easiness[probname] if probname in prob_easiness else 1
    
    probweight = 1.0/easy
    
    prob_data_list[i] = (probname,probweight),probdata
    
    # uncomment for plotting below:
    times.append(time_elapsed)
    sizes.append(len(probdata[0])+len(probdata[1])) # len(init)+len(deriv)
    easies.append(easy)

  # plot the time_elapsed vs size distribution
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(20,10))
  ax.set_yscale('log')
  sc = plt.scatter(times,sizes,c=easies,marker="+")
  plt.colorbar(sc)
  plt.savefig("{}/times_sizes{}.png".format(sys.argv[1],sys.argv[2].split("/")[0]),dpi=250)

  thax_sign,sine_sign,deriv_arits,axiom_hist = IC.prepare_signature(prob_data_list)

  if HP.THAX_SOURCE == HP.ThaxSource_AXIOM_NAMES: # We want to use axiom names rather than theory_axiom ids:
    thax_sign,prob_data_list,thax_to_str = IC.axiom_names_instead_of_thax(thax_sign,axiom_hist,prob_data_list)
  else:
    thax_to_str = {}

  print("thax_sign",thax_sign)
  print("sine_sign",sine_sign)
  print("deriv_arits",deriv_arits)
  print("thax_to_str",thax_to_str)
  '''
  actual_hist = defaultdict(int)
  for name,occ in axiom_hist.items():
    print(name,occ)
    actual_hist[int(occ)] += 1
  for occs, cnt in sorted(actual_hist.items()):
    print(occs,cnt)
  '''

  filename = "{}/data_sign.pt".format(sys.argv[1])
  print("Saving singature to",filename)
  torch.save((thax_sign,sine_sign,deriv_arits,thax_to_str), filename)
  print()

  filename = "{}/raw_log_data{}".format(sys.argv[1],IC.name_raw_data_suffix())
  print("Saving raw data to",filename)
  torch.save(prob_data_list, filename)
  print()

  # print(prob_data_list[0])

  print("Done")

