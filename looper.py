#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time, math

import subprocess
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
    for prob,(status,instructions,activations) in results.items():
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


if __name__ == "__main__":
  # Automating the vamp_eval - model_train - model_export loop.
  #
  # We need a folder naming scheme, checkpointing, result scanning, ...
  #
  # To be called as in: ./looper.py loop_count parallelism campaign_folder (must exist) exper_folder (will be created - ideally have this on local/scratch) [optionally starting parts-model file]
  #
  # campaign_folder contains "train.txt", "test.txt" - the files listing the problems, and a bunch of train/test_something.pkl's with results of baseline (non-neural) runs for reporting
  # 
  # ./looper.py 10 70 campaigns/small/ /local/sudamar2/lawa/exper1 [optional - starting parts-model file]

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])

  campaign_dir = sys.argv[3]
  exper_dir =  sys.argv[4]

  if len(sys.argv) > 5:
    model = torch.load(sys.argv[5])
  else:
    model = IC.get_initial_model()

  # TODO: for restarting from the middle, the optimizer state should be saved too
  # and we should be loading "training_data.pt", as they are the list thing getting saved

  if HP.OPTIMIZER ==  HP.OPTIMIZER_SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
  elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

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

  training_data = defaultdict(list) # group the results by problem

  # TODO: this also starts from scratch when training gets interrupted
  num_evals = 0
  num_successes = defaultdict(int) # for each problem, how many times we saw it solved

  loop = 0
  assert loop_count > 0
  while True:
    loop_str = "loop{}".format(loop)
    print(loop_str)
    cur_dir = os.path.join(exper_dir,loop_str)

    os.mkdir(cur_dir)

    # let's save the model we got from last time (either initial or the last training)
    parts_model_file_path = os.path.join(cur_dir,"parts-model.pt")
    torch.save(model, parts_model_file_path)

    keys = model[1]
    if HP.INCLUDE_LSMT:
      print("Initial key:",repr(keys))
    else:
      for i in range(HP.NUM_EFFECTIVE_QUEUES):
        print("Key {} {}".format(i,repr(keys(torch.tensor([i]))[0])))

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
        opts1 = "-i {} -p off".format(HP.INSTRUCTION_LIMIT)
        opts2 = " --random_seed {} -npcc {} -npcct {}".format(seed,script_model_file_path,temperature)
        metas.append((result_file_name,mission,temperature,opts1,opts2))
        jobs_for_eval.append((prob_lists[mission],opts1+opts2,False))

    jobs_for_training = []
    last_mission = None

    best_solveds = defaultdict(int)
    best_results = {}

    for (result_file_name,mission,temperature,opts1,opts2),results in zip(metas,evaluator.perform(jobs_for_eval)):
      torch.save((opts1+opts2,results), os.path.join(cur_dir,result_file_name))

      if mission != last_mission:
        last_mission = mission
        print(" ",mission)

      num_evals += 1 

      successes = []
      for prob,(status,instructions,activations) in results.items():
        if status == "uns":
          successes.append(prob)
          num_successes[prob] += 1

      print("    t={}  {:10.4f}% = {} / {}".format(temperature,len(successes)/len(results),len(successes),len(results)))

      # get the fine-grained results to compare against baseline

      if mission == "train":
        jobs_for_training.append((successes,"-i {} -spt on".format(10*HP.INSTRUCTION_LIMIT)+opts2,True))

      if len(successes) > best_solveds[mission]:
        best_solveds[mission] = len(successes)
        best_results[mission] = results
    
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

    print("Collected proofs from",len(training_data),"solved problems")
    sys.stdout.flush()

    # save the bulk
    torch.save(training_data, os.path.join(cur_dir,"train_data.pt"))

    print("Data gathering took",time.time()-start_time)
    print()
    sys.stdout.flush()

    loop += 1
    # traning the model here (in the final loop, there is just eval)
    if loop == loop_count:
      break

    # the actual training
    start_time = time.time()        
    # SGD style (one step per problem)
    factor = 1.0/len(training_data) # to scale down by the number of problems we trained on
    print("  training",end="")
    training_data_in_order = list(training_data.items())
    random.shuffle(training_data_in_order)
    for i,(prob,its_data) in enumerate(training_data_in_order):
      # print(prob)
      print(">",end="")

      inner_factor = factor

      # possibly scale up for problems that are not getting solved always
      difficulty = num_evals/num_successes[prob]
      inner_factor *= min(HP.DIFFICULTY_BOOST_CAP,math.pow(difficulty,HP.DIFFICULTY_BOOST_COEF))
      
      # and further scale down by the number of iterations on this problem
      inner_factor *= 1.0 / len(its_data) 

      for (clauses,journal,proof_flas) in its_data:        
        print(".",end="")
        optimizer.zero_grad()
        if HP.INCLUDE_LSMT:
          lm = IC.RecurrentLearningModel(*model,clauses,journal,proof_flas)
        else:
          lm = IC.LearningModel(*model,clauses,journal,proof_flas)
        lm.train()
        loss = lm.forward()
        if loss is None:
          continue
        loss *= inner_factor
        loss.backward()
        optimizer.step()
    print()
    print("Training took",time.time()-start_time)
    print()
    sys.stdout.flush()
      