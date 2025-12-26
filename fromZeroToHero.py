#!/usr/bin/env python3

import math
import inf_common as IC
import hyperparams as HP
import workers as W
from collections import defaultdict
from collections import Counter
import torch,sys
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os, time, random

from multiprocessing import Pool

from elooper import TraceIndex, no_dots, ilim2tlim

def eval_tweaks(trace_file_path,tweaks):
  eval_begin = time.time()

  trace_tuple = torch.load(trace_file_path)
  learn_model = IC.LearningModel(False,model,trace_tuple)
  learn_model.eval()

  with torch.no_grad():
    just_before_final,num2idx = learn_model.pre_forward()
    losses,selection_hit_rates,dists_to_good = learn_model.forward(just_before_final,num2idx,tweaks)

  return losses.detach()

N = 21

def run_one(task):
  i,prob,opts = task
  res = W.vampire_perfrom(prob,opts,None)
  return i,0 if res.status != "uns" else res.activations


NUM_SHUFFLED_RERUNS = 3

SCRIPT_MODEL_FILE_PATH = "script-model.pt"

def eval_tweaks_vamp(prob,tweaks_to_try):
  for i in range(N):
    model.tweaks.append(torch.nn.Parameter((tweaks_to_try[i])))

  saved = HP.TWEAKS_TO_PICK
  HP.TWEAKS_TO_PICK = N

  IC.export_model(model.state_dict(),SCRIPT_MODEL_FILE_PATH)

  ilim = HP.INSTRUCTION_LIMIT

  tasks = []
  for i in range(N):
    for _ in range(NUM_SHUFFLED_RERUNS):
      seed = random.randint(1,0x7fffff)
      # -npcct 2.0
      tasks.append((i,prob,f"-t {ilim2tlim(ilim)} -i {ilim} -p off --random_seed {seed} -si on -rtra on -npcc on -ncem {SCRIPT_MODEL_FILE_PATH} -ncem_gsd {i+1}"))

  pool = Pool(processes=3*N)
  results = pool.map(run_one, tasks)
  pool.close()
  pool.join()
  del pool

  HP.TWEAKS_TO_PICK = saved

  stacked_results = [[] for I in range(NUM_SHUFFLED_RERUNS)]
  idx = 0
  for i,val in sorted(results,key=lambda i_val : i_val[0] ):
    stacked_results[idx % NUM_SHUFFLED_RERUNS].append(val)
    idx += 1

  # print(stacked_results)

  return stacked_results


if __name__ == "__main__":
  # on dai-07
  model_file_path = "/home/sudamar2/mtpa-gnn-gsd/freshQuarter_mtpa-gnn-gsd-lin_tweaky2/loop9/loop-model.tar"
  model = IC.get_initial_model()
  aloop,amodel_state_dict = torch.load(model_file_path)
  model.load_state_dict(amodel_state_dict)
  print("Loaded model. It has",len(model.tweaks),"tweaks")

  trace_index_file_path = "/home/sudamar2/mtpa-gnn-gsd/freshQuarter_mtpa-gnn-gsd-lin_tweaky2/loop10/trace-index.pt"
  trace_index = torch.load(trace_index_file_path)
  print("Loaded trace_index with",len(trace_index.cur_problems()),"problems")

  tweak_map_file_path = "/nfs/sudamar2/lawa-temp/freshQuarter_mtpa-gnn-gsd-lin_tweaky2/justOneTweakingPhase/loop10/tweak-map-after-tweaking.tar"
  tweak_map = torch.load(tweak_map_file_path)
  print("Loaded tweak_map of len",len(tweak_map))

  cur_problems = list(trace_index.cur_problems())

  a_idx = random.randrange(len(cur_problems))
  b_idx = random.randrange(len(cur_problems))

  a_problem = cur_problems[a_idx]
  a_problem_short = a_problem.split("/")[-1][:-2]

  b_problem = cur_problems[a_idx]
  b_problem_short = b_problem.split("/")[-1][:-2]

  a_trace_file_path = trace_index.prob_traces(a_problem)[0] # these days, there is always one here anyway
  b_trace_file_path = trace_index.prob_traces(b_problem)[0] # these days, there is always one here anyway

  print("Picked a problem",a_problem,"with a trace of size",os.path.getsize(a_trace_file_path))
  print("Picked b problem",b_problem,"with a trace of size",os.path.getsize(b_trace_file_path))

  a_tweak = tweak_map[no_dots(a_problem)]
  b_tweak = tweak_map[no_dots(b_problem)]

  Xs = torch.linspace(-0.5, 1.5, N)

  # print("len(Xs)",len(Xs))

  tasks = [# (a_trace_file_path,a_tweak,f"{a_problem_short}-trace, {a_problem_short}-tweak"),
           (a_trace_file_path,b_tweak,f"{a_problem_short}-tr, {b_problem_short}-tw"),
           # (b_trace_file_path,a_tweak,f"{b_problem_short}-trace, {a_problem_short}-tweak"),
           # (b_trace_file_path,b_tweak,f"{b_problem_short}-trace, {b_problem_short}-tweak"),
           ]

  import matplotlib.pyplot as plt
  fig, ax_y = plt.subplots(figsize=(6,6))
  ax_y.set_ylabel("loss", color="C0")
  ax_y.tick_params(axis="y", labelcolor="C0")
  ax_z = ax_y.twinx()
  ax_z.set_ylabel("activations", color="C1")
  ax_z.tick_params(axis="y", labelcolor="C1")

  for trace_file_path, tweak, task_name in tasks:
    tweaks_to_try = Xs.unsqueeze(1) * tweak
    Ys = eval_tweaks(trace_file_path,tweaks_to_try)

    Zs = eval_tweaks_vamp(a_problem,tweaks_to_try)

    x_np = Xs.detach().cpu().numpy()
    y_np = Ys.detach().cpu().numpy()

    ax_y.plot(x_np, y_np,label=task_name)
    for i in range(NUM_SHUFFLED_RERUNS):
      ax_z.plot(x_np, Zs[i],label=task_name + " - acts",linestyle="None", marker="o",markersize = 2, color="C1")

  ax_y.set_ylim(bottom=0)
  ax_z.set_ylim(bottom=0)

  lines_y, labels_y = ax_y.get_legend_handles_labels()
  lines_z, labels_z = ax_z.get_legend_handles_labels()
  ax_y.legend(lines_y + lines_z, labels_y + labels_z, loc="best")

  plt.xlabel("from gen to tweak")
  plt.axvline(0, color="black", linestyle="--", linewidth=1)
  plt.savefig(f"fromZeroToHero_for_{a_problem_short}_cross_{b_problem_short}.pdf", format="pdf", bbox_inches="tight")
  plt.close()