#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random

import subprocess
from multiprocessing import Pool
from collections import defaultdict

MISSIONS = ["train","test"]

def look_for_baselines(some_dir):
  baselines = defaultdict(dict) # "train"/"test" -> { reference_run_file_name -> results }

  root, dirs, files = next(os.walk(some_dir)) # just take the files immediately under some_dir
  for filename in files:
    if not filename.endswith(".pkl"):
      continue

    for mission in MISSIONS:
      if filename.startswith(mission):
        break
    else:
      continue

    with open(os.path.join(some_dir,filename),'rb') as f:
      baselines[mission][filename] = pickle.load(f)

  return baselines

def compare_to_baselines(results,baseline):
  total = len(results)
  for refname, (refmeta,refres) in sorted(baseline.items()):
    print("    Compared to",refname)
    neur_only = 0.0
    neur_better = 0.0
    neur_dunno = 0.0
    same = 0.0
    sat_and_shit = 0.0
    base_dunno = 0.0
    base_better = 0.0
    base_only = 0.0
    for prob,(status,time,instructions,activations) in results.items():
      fact = 1.0/len(refres[prob])
      for (sta,tim,ins,act) in refres[prob]:
        if status == "uns":
          if sta == "uns":
            if activations < act:
              neur_better += fact
            elif activations == act:
              same += fact
            else:
              base_better += fact
          else:
            if activations < act:
              neur_only += fact
            else:
              neur_dunno += fact
        elif sta == "uns":
          if activations <= act:
            base_dunno += fact
          else:
            base_only += fact
        else:
          sat_and_shit += fact

    print("    {:10.4f}".format(neur_only/total),"neur_only   ",neur_only)
    print("    {:10.4f}".format(neur_better/total),"neur_better ",neur_better)
    print("    {:10.4f}".format(neur_dunno/total),"neur_dunno  ",neur_dunno)
    print("    {:10.4f}".format(same/total),"same        ",same)
    print("    {:10.4f}".format(base_dunno/total),"base_dunno  ",base_dunno)
    print("    {:10.4f}".format(base_better/total),"base_better ",base_better)
    print("    {:10.4f}".format(base_only/total),"base_only   ",base_only)
    print("    {:10.4f}".format(sat_and_shit/total),"sat_and_shit",sat_and_shit)
    print()


if __name__ == "__main__":
  # Automating the vamp_eval - model_train - model_export loop.
  #
  # We need a folder naming scheme, checkpointing, result scanning, ...
  #
  # To be called as in: ./looper.py loop_count parallelism vampire_executable campaign_folder (must exist) exper_folder (will be created - ideally have this on local/scratch) [optionally starting parts-model file]
  #
  # campaign_folder contains "train.txt", "test.txt" - the files listing the problems, and a bunch of train/test_something.pkl's with results of baseline (non-neural) runs for reporting
  # 
  # ./looper.py 10 70 ./vampire_rel_lawa_6438 campaigns/small/ /local/sudamar2/lawa/exper1 [optional - starting parts-model file]

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])

  vampire = sys.argv[3]
  campaign_dir = sys.argv[4]
  exper_dir =  sys.argv[5]

  if len(sys.argv) > 6:
    model = torch.load(sys.argv[6])
  else:
    model = IC.get_initial_model()

  optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)        
  
  '''
  # start a new expreriment folder
  os.mkdir(exper_dir)

  # CAREFUL: this way can only call looper from lawa folder ...
  shutil.copy("hyperparams.py",exper_dir)
  with open(os.path.join(exper_dir,"campaign.txt"),"w") as f:
    f.write(campaign_dir)
  '''

  # load the reference runs from campaign
  baselines = look_for_baselines(campaign_dir)

  loop = 0
  assert loop_count > 0
  while True:
    loop_str = "loop{}".format(loop)
    print(loop_str)
    cur_dir = os.path.join(exper_dir,loop_str)

    # os.mkdir(cur_dir)

    # let's save the model we got from last time (either initial or the last training)
    parts_model_file_path = os.path.join(cur_dir,"parts-model.pt")
    # torch.save(model, parts_model_file_path)

    # let's also export it for scripting (note we load a fresh copy of model -- export_model is possibly destructive)
    script_model_file_path = os.path.join(cur_dir,"script-model.pt")    
    # IC.export_model(torch.load(parts_model_file_path),script_model_file_path)
    
    successes = {}
    for mission in MISSIONS:
      prob_list_file = os.path.join(campaign_dir,mission+".txt")
      sub_dir = os.path.join(cur_dir,mission)
      what_to_run = './run_in_parallel_plus_local.sh {} {} ./run_lawa_vampire.sh {} "-spt on -npcc {}" {}'.format(parallelism,prob_list_file,vampire,script_model_file_path,sub_dir)
      # os.system(what_to_run)

      # get the successes
      output = subprocess.getoutput(["grep -rl Tanya {}".format(sub_dir)])
      successes[mission] = output.split("\n")
      solved = len(successes[mission])
      outof = int(subprocess.getoutput(["wc -l {}".format(prob_list_file)]).split()[0])
      print("  ",solved/outof,"% =",solved,"/",outof,"succusses on",mission)

      # get the fine-grained results to compare against baseline
      results = IC.scan_result_folder(sub_dir)
      compare_to_baselines(results,baselines[mission])
      
    loop += 1
    # traning the model here (in the final loop, there is just eval)
    if loop == loop_count:
      break
    
    continue

    # load successful runs for training
    if True: # parallel
      pool = Pool(processes=parallelism) # number of cores to use
      results = pool.map(IC.load_one, successes["train"], chunksize = 1)
      training_data = list(filter(None, results))
      pool.close()
      pool.join()
    else: # sequential
      training_data = []
      for log_name in successes["train"]:
        print("Loading",log_name)
        result = IC.load_one(log_name)
        if result is not None:
          training_data.append(result)

    # save the bulk
    torch.save(results, os.path.join(cur_dir,"train_data.pt"))

    # the actual training
        
    random.shuffle(results)
    # SGD style (one step per problem)
    factor = 1.0/len(training_data)
    print("  training",end="")
    for i,(filename,clauses,journal,proof_flas) in enumerate(training_data):
      print(".",end="")
      
      optimizer.zero_grad()
      lm = IC.LearningModel(*model,clauses,journal,proof_flas)
      lm.train()
      loss = lm.forward()
      loss *= factor # scale down by the number of problems we trained on
      loss.backward()
      optimizer.step()
    print()
      