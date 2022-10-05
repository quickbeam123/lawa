#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time, math

from operator import add

import subprocess
from collections import defaultdict

MISSIONS = ["train","test"]

def look_for_baselines(some_dir):
  baselines = defaultdict(dict) # "train"/"test" -> { reference_run_file_name -> results }

  _root, _dirs, files = next(os.walk(some_dir)) # just take the files immediately under some_dir
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
  for refname, (_refmeta,refres) in sorted(baseline.items()):
    print("    Compared to",refname)
    neur_only = 0.0
    neur_better = 0.0
    neur_dunno = 0.0
    same = 0.0
    sat_and_shit = 0.0
    base_dunno = 0.0
    base_better = 0.0
    base_only = 0.0
    for prob,(status,_instrs,activations) in results.items():
      # Problems/SWB/SWB093+1.p
      short_prob_name = prob[13:-2]
      fact = 1.0/len(refres[short_prob_name])
      for (sta,tim,ins,act) in refres[short_prob_name]:
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
    print("    {:10.4f}".format(base_better/total),"base_better ",base_better)
    print("    {:10.4f}".format(base_only/total),"base_only   ",base_only)
    print("    ---------------------------")
    print("    {:10.4f}".format(neur_dunno/total),"neur_dunno  ",neur_dunno)
    print("    {:10.4f}".format(same/total),"same        ",same)
    print("    {:10.4f}".format(base_dunno/total),"base_dunno  ",base_dunno)
    print("    ---------------------------")
    print("    {:10.4f}".format(sat_and_shit/total),"sat_and_shit",sat_and_shit)
    print()

def formatI2T(str2format,instrs):
  secs = max(1,instrs // 200) # it's 10 times more than the instrlimit on a 2GH machine
  return str2format.format(secs,instrs)

if __name__ == "__main__":
  # Automating the vamp_eval - model_train - model_export loop.
  #
  # We need a folder naming scheme, checkpointing, result scanning, ...
  #
  # To be called as in: ./looper.py loop_count parallelism campaign_folder (must exist) exper_folder (will be created - ideally have this on local/scratch) [optionally, last completed loop dir]
  #
  # campaign_folder contains "train.txt", "test.txt" - the files listing the problems, and a bunch of train/test_something.pkl's with results of baseline (non-neural) runs for reporting
  #
  # ./looper.py 10 70 campaigns/small/ /local/sudamar2/lawa/exper1 [optionally, last completed loop dir]

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])

  campaign_dir = sys.argv[3]
  exper_dir =  sys.argv[4]

  if len(sys.argv) > 5:
    load_dir = sys.argv[5]
    model = torch.load(os.path.join(load_dir,"parts-model.pt"))
    training_data = torch.load(os.path.join(load_dir,"train_data.pt"))
    # TODO: this is broken anyway: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # optimizer = torch.load(os.path.join(load_dir,"optimizer.pt"))
    if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
      optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
    elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
      optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

    # TODO: would we want to set a different momentum too?

    need_next_eval = False
  else:
    model = IC.get_initial_model()
    training_data = defaultdict(list) # group the results by problem
    if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
      optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
    elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
      optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

    need_next_eval = True

  evaluator = IC.Evaluator(parallelism)
  def cleanup():
    evaluator.close()
  atexit.register(cleanup)

  # start a new expreriment folder
  os.mkdir(exper_dir)

  # CAREFUL: this way can only call looper from lawa folder ...
  shutil.copy("hyperparams.py",exper_dir)
  with open(os.path.join(exper_dir,"campaign.txt"),"w") as f:
    f.write(campaign_dir)

  # get training and test files' content:
  prob_lists = {}
  for mission in MISSIONS:
    prob_list_file = os.path.join(campaign_dir,mission+".txt")
    with open(prob_list_file,"r") as f:
      lines = f.readlines()
      prob_lists[mission] = [line.rstrip() for line in lines]

  # load the reference runs from campaign
  baselines = look_for_baselines(campaign_dir)

  loop = 0
  assert loop_count > 0
  while True:
    loop_str = "loop{}".format(loop)
    print(loop_str)
    cur_dir = os.path.join(exper_dir,loop_str)

    os.mkdir(cur_dir)

    if need_next_eval:
      # let's save the model we got from last time (either initial or the last training)
      parts_model_file_path = os.path.join(cur_dir,"parts-model.pt")
      torch.save(model, parts_model_file_path)
      optimizer_file_path = os.path.join(cur_dir,"optimizer.pt")
      torch.save(optimizer, optimizer_file_path)

      print("Key {}".format(repr(model[1])))

      # let's also export it for scripting (note we load a fresh copy of model -- export_model is possibly destructive)
      script_model_file_path = os.path.join(cur_dir,"script-model.pt")
      IC.export_model(torch.load(parts_model_file_path),script_model_file_path)

      start_time = time.time()

      seed = random.randint(1,0x7fffff)
      jobs_for_eval = []
      metas = []
      for mission in MISSIONS:
        for temperature in HP.TEMPERATURES:
          result_file_name = "{}_t{}.pt".format(mission,temperature)
          opts1 = formatI2T("-t {} -i {} -p off",HP.INSTRUCTION_LIMIT if mission == "train" else HP.INSTRUCTION_LIMIT_TEST)
          opts2 = " --random_seed {} -npcc {} -npcct {}".format(seed,script_model_file_path,temperature)
          metas.append((result_file_name,mission,temperature,opts1,opts2))
          jobs_for_eval.append((prob_lists[mission],opts1+opts2,False))

      jobs_for_training = []
      last_mission = None

      best_solveds = defaultdict(int)
      best_results = {}

      num_fails = 0
      sum_failss_activations = 0

      for (result_file_name,mission,temperature,opts1,opts2),results in zip(metas,evaluator.perform(jobs_for_eval)):
        torch.save((opts1+opts2,results), os.path.join(cur_dir,result_file_name))

        if mission != last_mission:
          last_mission = mission
          print(" ",mission)
          solved_accum = set()

        successes = []
        succ_delta = []
        for prob,(status,instructions,activations) in results.items():
          if status == "uns":
            if prob not in solved_accum:
              succ_delta.append(prob)
            solved_accum.add(prob)
            successes.append(prob)
          elif status is None:
            num_fails += 1
            sum_failss_activations += activations

        print("    t={}  {:10.4f}% = {} / {}   +{} (accum {})".format(temperature,len(successes)/len(results),len(successes),len(results),len(succ_delta),len(solved_accum)))

        # get the fine-grained results to compare against baseline

        if mission == "train":
          if HP.FIRST_PROOF_ONLY:
            jobs_for_training.append((succ_delta,formatI2T("-t {} -i {} -spt on",10*HP.INSTRUCTION_LIMIT)+opts2,True))
          else:
            jobs_for_training.append((successes,formatI2T("-t {} -i {} -spt on",10*HP.INSTRUCTION_LIMIT)+opts2,True))

        if len(successes) >= best_solveds[mission]:
          best_solveds[mission] = len(successes)
          best_results[mission] = results

      print()
      print("Seen the avarege of",sum_failss_activations/num_fails,"activations over failed runs.")
      print()
      for mission in MISSIONS:
        print("  Comparing best",mission,"to baseline")
        compare_to_baselines(best_results[mission],baselines[mission])

      print()
      print("Eval took",time.time()-start_time)
      print()
      sys.stdout.flush()

      start_time = time.time()
      if not HP.CUMMULATIVE:
        # cleanup training_data since the last iteration
        training_data = defaultdict(list) # group the results by problem
      for results in evaluator.perform(jobs_for_training):
        for prob,data in results.items():
          if data is not None:
            training_data[prob].append(data)

      if HP.CUMMULATIVE:
        # by a convention, we keep the last "len(HP.TEMPERATURES)" solutions
        for prob in training_data:
          training_data[prob] = training_data[prob][-len(HP.TEMPERATURES):]

      print("Collected proofs from",len(training_data),"problems to learn from")
      sys.stdout.flush()

      # save the bulk
      torch.save(training_data, os.path.join(cur_dir,"train_data.pt"))

      print("Data gathering took",time.time()-start_time)
      print()
      sys.stdout.flush()

    need_next_eval = True

    loop += 1
    # traning the model here (in the final loop, there is just eval)
    if loop == loop_count:
      break

    # the actual training
    start_time = time.time()
    print("  training",end="")
    # one bulk training step (mega-batch)
    optimizer.zero_grad()

    loss = torch.zeros(1)
    if not HP.PER_PROBLEM_NORMALIZED:
      loss_norm_tots = ((torch.zeros(1),0),(torch.zeros(1),0),(torch.zeros(1),0))

    factor = 1.0/len(training_data)
    for (prob,its_proofs) in training_data.items():
      # print(prob)
      print(">",end="")

      inner_factor = factor/len(its_proofs)
      for proof_tuple in its_proofs:
        print(".",end="")

        lm = IC.LearningModel(*model,*proof_tuple)
        lm.train()
        loss_norms = lm.forward()

        if HP.PER_PROBLEM_NORMALIZED:
          for (l,n) in loss_norms:
            if n > 0:
              loss += inner_factor*l/n
        else:
          # loss_norm_tots += loss_norms (i.e., keep accumulating, vectore-wise)
          loss_norm_tots = tuple(map(add,loss_norms,loss_norm_tots))

    if not HP.PER_PROBLEM_NORMALIZED:
      for (l,n) in loss_norm_tots:
        if n:
          loss += l/n

    loss.backward()
    optimizer.step()
    print()
    print("Training took",time.time()-start_time)
    print()
    sys.stdout.flush()
