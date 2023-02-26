#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

from multiprocessing import Pool

import inf_common as IC
import hyperparams as HP

import torch

def eval_one(tweak):
  num_tries = 10
  num_succ = 0
  for i in range(num_tries):
    seed = random.randint(1,0x7fffff)
    twstr = ",".join(str(t) for t in tweak)
    (status,instructions,activations) = IC.vampire_perfrom(prob_name,"-t 10 -i 5000 -p off --random_seed {} -npcc {} -npcct {} -npccw {}".format(seed,os.path.join(folder,"script-model.pt"),temp,twstr))
    if status == "uns":
      num_succ += 1
  # just for once, also go for a trace and save it
  '''
  if num_succ > num_tries//2:
    result = IC.vampire_gather(prob_name,"-t 100 -i 50000 -spt on --random_seed {} -npcc {} -npcct {} -npccw {}".format(seed,os.path.join(folder,"script-model.pt"),temp,twstr))
    if result is not None and result[0]: # non-degenerate
      trace_file_path = "newtrace_{}_{}_{}_{}.pt".format(twstr.replace(",","-"),seed,prob_name.replace("/","_"),temp)
      torch.save(result[1],trace_file_path)
  '''
  return num_succ/num_tries

if __name__ == "__main__":
  # Load a model (loop model optimizer) and a trace and
  #
  # To be called as in: ./tweak_plotter.py folder_with_everything tmp/test_Problems_GRP_GRP073-1.p_0.0.pt (i.e, a trace)

  folder = sys.argv[1]

  (loop,num_tweaks,active_from,model_state_dict,optimizer_state_dict) = torch.load(os.path.join(folder,"info-model-and-optimizer.tar"))

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  print("Loaded some model of loop",loop,"num_tweaks",num_tweaks,"active_from",active_from)

  proof_tuple = torch.load(sys.argv[2])
  print("Loaded proof_tuple with",len(proof_tuple[0]),"clauses and",len(proof_tuple[1]),"journal steps")

  prob_tweak = None
  tweak_map_file_path = os.path.join(folder,"tweak_map.pt")
  if os.path.exists(tweak_map_file_path):
    tweak_map = torch.load(tweak_map_file_path)

    # e.g. train_Problems_COL_COL042-6.p_1.0.pt
    trace_name = sys.argv[2]
    trace_name_spl = trace_name.split("_")
    temp = trace_name_spl[-1][:-3]
    prob_name = "/".join(trace_name_spl[1:-1])

    prob_tweak = tweak_map[prob_name][temp]

    print(f"Loaded prob's ({prob_name}) of temp {temp} tweak: {prob_tweak}")

  trace_file_path = os.path.join(folder,"trace-index.pt")
  if os.path.exists(trace_file_path):
    # print("Found trace-index.pt")
    trace_index = torch.load(trace_file_path)
    for m,mdict in trace_index.items():
      if prob_name in mdict:
        print("Found the problem (",prob_name,") in trace-index.pt under",m)
        for t,trace in mdict[prob_name].items():
          print("  has trace for t=",t,trace)

  learn_model = IC.LearningModel(model,*proof_tuple)
  learn_model.eval()

  import numpy as np

  AXLEN = 0.01
  STEPS = 25
  BIGSTEPS = 5

  dx, dy = AXLEN/STEPS, AXLEN/STEPS

  x = np.arange(-AXLEN, AXLEN+dx, dx)
  y = np.arange(-AXLEN, AXLEN+dy, dy)
  X, Y = np.meshgrid(x, y)

  subx = x[BIGSTEPS//2::BIGSTEPS]
  suby = y[BIGSTEPS//2::BIGSTEPS]
  sX, sY = np.meshgrid(subx, suby)

  def get_loss(Xs, Ys):
    # return Xs
    with torch.no_grad():
      stack_of_tweaks = np.column_stack((Xs.ravel(),Ys.ravel()))
      losses = learn_model.forward(stack_of_tweaks)
      return losses.reshape(Xs.shape)

  extent = np.min(x), np.max(x), np.min(y), np.max(y)

  Z = get_loss(X, Y)

  from matplotlib import pyplot as plt
  plt.imshow(Z, cmap=plt.cm.hsv, interpolation='nearest',extent=extent, origin='lower')

  def flatten(l):
    return [i for sl in l for i in sl]

  fsX = flatten(sX)
  fsY = flatten(sY)
  if prob_tweak is not None:
    fsX.append(prob_tweak[0])
    fsY.append(prob_tweak[1])

  def eval_with_vamp(fsX,fsY):
    # return [random.uniform(0.0,1.0) for _ in fsX]
    pool = Pool(processes=10)
    results = pool.map(eval_one, list(zip(fsX,fsY)))
    pool.close()
    pool.join()
    del pool
    return results

  plt.colorbar()
  plt.scatter(fsX,fsY,c=eval_with_vamp(fsX,fsY),marker='o',cmap="gray")
  plt.savefig("tweak_map_{}.png".format(os.path.basename(sys.argv[2])),dpi=250)

