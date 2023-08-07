#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

from multiprocessing import Pool

import inf_common as IC
import hyperparams as HP

import torch

TEMP_SCRIPT_MODEL = "temp-script-model.pt"

def eval_one(tweak):
  num_tries = 10
  num_succ = 0
  for i in range(num_tries):
    seed = random.randint(1,0x7fffff)
    twstr = ",".join(str(t) for t in tweak)
    (status,instructions,activations) = IC.vampire_perfrom(prob,"-t 10 -i 5000 -p off --random_seed {} -npcc {} -nnf {} -npccw {}".format(seed,TEMP_SCRIPT_MODEL,HP.NUM_FEATURES,twstr))
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
  # print(num_succ/num_tries)
  return num_succ/num_tries

if __name__ == "__main__":
  # Load a model (loop model optimizer) and a trace and
  #
  # To be called as in: ./tweak_plotter.py deleteme/loop13 Problems/GRP/GRP703-10.p

  folder = sys.argv[1]
  (loop,model_state_dict,optimizer_state_dict) = torch.load(os.path.join(folder,"loop-model-and-optimizer.tar"))

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  print("Loaded some model of loop",loop)

  trace_index_file_path = os.path.join(folder,"trace-index.pt")
  trace_index = torch.load(trace_index_file_path)
  print("Loaded a trace index with",len(trace_index["train"]),"train and",len(trace_index["valid"]),"validation problems registered.")

  prob = sys.argv[2]
  print("Plotting for",prob)
  trace_list = None
  for mission in ["train","valid"]:
    if prob in trace_index[mission]:
      trace_list = trace_index[mission][prob]
      break
  assert trace_list is not None
  print("  from mission",mission,"with",len(trace_list),"traces")

  # tweak_map_file_path = os.path.join(folder,"tweak_map.pt")
  tweak_map_file_path = os.path.join(folder,"after-train-tweak_map.pt")
  tweak_map = torch.load(tweak_map_file_path)
  print("  with tweak",tweak_map[prob])
  prob_tweak = tweak_map[prob]

  import numpy as np

  AXLEN = float(1+int(max(abs(t) for t in prob_tweak)))
  STEPS = 25
  BIGSTEPS = 5

  dx, dy = AXLEN/STEPS, AXLEN/STEPS

  x = np.arange(-AXLEN, AXLEN+dx, dx)
  y = np.arange(-AXLEN, AXLEN+dy, dy)
  X, Y = np.meshgrid(x, y)

  subx = x[BIGSTEPS//2::BIGSTEPS]
  suby = y[BIGSTEPS//2::BIGSTEPS]
  sX, sY = np.meshgrid(subx, suby)

  local_fact = 1/len(trace_list)
  proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_list]

  def get_loss(Xs, Ys):
    # return Xs
    with torch.no_grad():
      stack_of_tweaks = np.column_stack((Xs.ravel(),Ys.ravel()))
      losses = torch.zeros(stack_of_tweaks.shape[0])
      for proof_tuple in proof_tuples:
        clauses,journal,proof_flas,warmup_time,select_time = proof_tuple
        print(f"clause {len(clauses)} journal {len(journal)} proof_flas {len(proof_flas)}")
        learn_model = IC.LearningModel(model,*proof_tuple)
        learn_model.eval()
        losses += local_fact*learn_model.forward(stack_of_tweaks)

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

  IC.export_model(model.state_dict(),TEMP_SCRIPT_MODEL)

  def eval_with_vamp(fsX,fsY):
    # return [random.uniform(0.0,1.0) for _ in fsX]
    pool = Pool(processes=120)
    results = pool.map(eval_one, list(zip(fsX,fsY)))
    # print(fsX,fsY,results)
    pool.close()
    pool.join()
    del pool
    return results

  plt.colorbar()
  plt.scatter(fsX,fsY,c=eval_with_vamp(fsX,fsY),marker='o',cmap="gray")
  plt.savefig("tweak_map_{}.png".format(os.path.basename(sys.argv[2])),dpi=250)

