#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time, subprocess

from collections import defaultdict

MISSIONS = ["train","test"]
STYLES = { MISSIONS[0] : "-", MISSIONS[1] : "--"}

MAXINT = 2**32

def get_status(info):
  if isinstance(info,IC.VampResult):
    return info.status
  else:
    return info[0]

from multiprocessing import Pool

def retry_in_vamp(task):
  (prob,opts) = task
  return (IC.vampire_perfrom(prob,opts),opts)

def retry_in_POOL(tasks):
  if True:
    pool = Pool(processes=60) # number of cores to use
    results = pool.map(retry_in_vamp, tasks, chunksize = 1)
    pool.close()
    pool.join()
    del pool
  else:
    results = []
    for task in tasks:
      results.append(retry_in_vamp(task))
  return results

if __name__ == "__main__":
  # Report newly (with regards to loop iterations) solving hard (TPTP rating 1.0) problems
  #
  # To be called as in: ./checkTheBang.py exper_folder1

  POIS = ["Problems/SEU/SEU389+2.p"]

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

                  if False: # for the proof reconstruction, desperation mode
                    if prob in POIS:
                      print("      ",i,prob,total,rate,vr)

                      temp_extra = HP.PERFORMS_SPECIAL[i]

                      print("RERUN AND REPRODUCE")
                      sys.stdout.flush()
                      tasks = []
                      for _ in range(128):
                        seed = random.randint(1,0x7fffffff)
                        opts = f"-t 300 -i 100000 --input_syntax tptp -sa otter -ncem /home/sudamar2/mtpa-gnn/tptpOverfit100k3/loop{loop}/script-model.pt -npcc on {temp_extra} -si on -rtra on --random_seed {seed}"
                        tasks.append((prob,opts))

                      best_instrs = 200000
                      best_opts = None
                      best_has_npcct = True

                      # CAREFUL; we need HP.VAMPIRE_EXECUTABLE to be "vampire_rel_mtpa-gnn_8867" for the following! 

                      for res,opts in retry_in_POOL(tasks):
                        if res.status == "uns":
                          print(res,"for",opts)
                          has_npcct = "npcct" in opts
                          if best_has_npcct and not has_npcct:
                            best_has_npcct = False
                            best_opts = opts
                            best_instrs = res.instructions
                          elif res.instructions < best_instrs:
                            # print("Improving best_instrs from",best_instrs,"to",res.instructions,"for",opts)
                            best_instrs = res.instructions
                            best_opts = opts

                      if best_opts:
                        print("Choosing",best_opts)
                        result_file = "hards/"+prob.split("/")[-1]+"rf"
                        running = f"./vampire_rel_mtpa-gnn_8867 {best_opts} {prob} -p on"
                        subprocess.run(f"echo {running} > {result_file}", shell=True)
                        subprocess.run(f"./run_lawa_vampire.sh {running} >> {result_file}", shell=True)

                        print("Written proof to",result_file)
                      else:
                        print("Failed to reproduce for",prob)

                      sys.stdout.flush()

                      # TODO: fix the HP's vampire version back
                      # rerun with with "-p on"

                      # print(f"./vampire_rel_mtpa-gnn_8867  -p on")


                    continue


                  print("      ",i,prob,total,rate,vr)
                  res = subprocess.run(["grep", "Rating", prob], stdout=subprocess.PIPE, text=True)
                  output = res.stdout
                  spl = output.split()
                  assert spl[0] == "%"
                  assert spl[1] == "Rating"
                  assert spl[2] == ":"
                  short = len(spl) == 5
                  if short:
                      print("SHORT!")
                  print(output)
                  print(info[1])
                  print(info[2])

                  if short and False: # for the proof reconstruction
                    print("RERUN AND REPRODUCE")
                    sys.stdout.flush()
                    tasks = []
                    for temp_extra in [" -npcct 1.0", " -npcct 0.333", " -npcct 0.111", " -npcct 0.037", ""]:
                      for seed in range(1,13):
                        opts = f"-t 300 -i 100000 --input_syntax tptp -sa otter -ncem /home/sudamar2/mtpa-gnn/tptpOverfit100k3/loop{loop}/script-model.pt -npcc on {temp_extra} -si on -rtra on --random_seed {seed}"
                        tasks.append((prob,opts))

                    best_instrs = 200000
                    best_opts = None
                    best_has_npcct = True

                    # CAREFUL; we need HP.VAMPIRE_EXECUTABLE to be "vampire_rel_mtpa-gnn_8867" for the following! 

                    for res,opts in retry_in_POOL(tasks):
                      if res.status == "uns":
                        print(res,"for",opts)
                        has_npcct = "npcct" in opts
                        if best_has_npcct and not has_npcct:
                          best_has_npcct = False
                          best_opts = opts
                          best_instrs = res.instructions
                        elif res.instructions < best_instrs:
                          # print("Improving best_instrs from",best_instrs,"to",res.instructions,"for",opts)
                          best_instrs = res.instructions
                          best_opts = opts

                    if best_opts:
                      print("Choosing",best_opts)
                      result_file = "hards/"+prob.split("/")[-1]+"rf"
                      running = f"./vampire_rel_mtpa-gnn_8867 {best_opts} {prob} -p on"
                      subprocess.run(f"echo {running} > {result_file}", shell=True)
                      subprocess.run(f"./run_lawa_vampire.sh {running} >> {result_file}", shell=True)

                      print("Written proof to",result_file)
                    else:
                      print("Failed to reproduce for",prob)

                    sys.stdout.flush()

                    # TODO: fix the HP's vampire version back
                    # rerun with with "-p on"

                    # print(f"./vampire_rel_mtpa-gnn_8867  -p on")

      loop += update


