#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys

import subprocess
from multiprocessing import Pool

if __name__ == "__main__":
  # Automating the vamp_eval - model_train - model_export loop.
  #
  # We need a folder naming scheme, checkpointing, result scanning, ...
  #
  # To be called as in: ./looper.py loop_count parallelism vampire_executable train_problems.txt test_problems.txt dumpfolder(must exists; ideally empty)_deally_on_local(or scratch) [optionally start from a checkpoint number i]
  #
  # ./looper.py 10 50 ./vampire_rel_lawa_6437 train1000.txt test500.txt /local/sudamar2/lawa/exper1

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])

  vampire = sys.argv[3]
  train_probs = sys.argv[4]
  test_probs = sys.argv[5]
  exper_dir =  sys.argv[6]

  model = IC.get_initial_model()

  loop = 0
  assert loop_count > 0
  while True:
    loop_str = "loop{}".format(loop)
    print(loop_str)
    cur_dir = os.path.join(exper_dir,loop_str)

    # os.mkdir(cur_dir)

    # let's save the model we got from last time (either initial or the last training)
    # parts_model_file_path = os.path.join(cur_dir,"parts-model.pt")
    # torch.save(model, parts_model_file_path)

    # let's also export it for scripting (note we load a fresh copy of model -- export_model is possibly destructive)
    script_model_file_path = os.path.join(cur_dir,"script-model.pt")    
    # IC.export_model(torch.load(parts_model_file_path),script_model_file_path)

    successes = {}
    for (mission,prob_list_file) in [("train",train_probs),("test",test_probs)]:
      sub_dir = os.path.join(cur_dir,mission)
      what_to_run = './run_in_parallel_plus_local.sh {} {} ./run_lawa_vampire.sh {} "-npcc {}" {}'.format(parallelism,prob_list_file,vampire,script_model_file_path,sub_dir)
      # os.system(what_to_run)

      # get successes
      output = subprocess.getoutput(["grep -rl Tanya {}".format(sub_dir)])
      successes[mission] = output.split("\n")
      print("  ",len(successes[mission]),"/",subprocess.getoutput(["wc -l {}".format(prob_list_file)]).split()[0],"succusses on",mission)
        
    loop += 1
    # traning the model here (in the final loop, there is just eval)
    if loop == loop_count:
      break
    
    # load successful runs for training
    if True: # parallel
      pool = Pool(processes=parallelism) # number of cores to use
      results = pool.map(IC.load_one, successes["train"], chunksize = 1)
      pool.close()
      pool.join()
    else: # sequential
      results = []
      for log_name in successes["train"]:
        print("Loading",log_name)
        results.append(IC.load_one(log_name))

    # save the bulk
    torch.save(results, os.path.join(cur_dir,"train_data.pt"))

    # sequential (for now) training
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)        
    # SGD style (one step per problem)
    for (clauses,journal,proof_flas) in results:
      optimizer.zero_grad()
      lm = IC.LearningModel(*model,clauses,journal,proof_flas)
      lm.train()
      loss = lm.forward()
      loss.backward()
      optimizer.step()
    

  