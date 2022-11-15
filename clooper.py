#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque

import multiprocessing

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

MISSIONS = ["train","test"]

def print_model_part():
  print("Key {}".format(repr(model[1])))
  pass

def load_train_tests_problem_lists(campaign_dir):
  prob_lists = {}
  for mission in MISSIONS:
    prob_list_file = os.path.join(campaign_dir,mission+".txt")
    with open(prob_list_file,"r") as f:
      lines = f.readlines()
      prob_lists[mission] = [line.rstrip() for line in lines]
  return prob_lists

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

  # non-compulsory; reporting on what the baselines actually are
  '''
  for mission in MISSIONS:
    for refname, (_refmeta,refres) in sorted(baselines[mission].items()):
      print(refname,_refmeta)
      for instr_cap in [5000,10000,50000]:
        print("  instr_cap",instr_cap)
        uns_uni = 0
        sat_uni = 0
        uns_cnts = defaultdict(int)
        sat_cnts = defaultdict(int)
        for prob,res_list in refres.items():
          had_uns = False
          had_sat = False
          for i,(sta,tim,ins,act) in enumerate(res_list):
            if ins > instr_cap:
              continue
            if sta == "uns":
              uns_cnts[i] += 1
              had_uns = True
            elif sta == "sat":
              sat_cnts[i] += 1
              had_sat = True
          assert not had_uns or not had_sat
          if had_uns:
            uns_uni += 1
          elif had_sat:
            sat_uni += 1
        print("    UNS:",end=" ")
        print(sum(uns_cnts.values())/len(uns_cnts),"union",uns_uni,"of",len(uns_cnts))
        print("    SAT:",end=" ")
        print(sum(sat_cnts.values())/len(sat_cnts),"union",sat_uni,"of",len(sat_cnts))
  '''

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
    sys.stdout.flush()

def claim_loop_dir(loop):
  loop_str = "loop{}".format(loop)
  print(loop_str)
  sys.stdout.flush()
  cur_dir = os.path.join(exper_dir,loop_str)
  os.mkdir(cur_dir)
  return cur_dir

def save_model_and_optimizer(cur_dir,model,optimizer):
  parts_model_state_file_path = os.path.join(cur_dir,"parts-model-state.tar")
  torch.save(model.state_dict(), parts_model_state_file_path)
  optimizer_file_path = os.path.join(cur_dir,"optimizer-state.tar")
  torch.save(optimizer.state_dict(), optimizer_file_path)

def ilim2tlim(ilim):
  secs = max(5,ilim // 200) # it's 10 times more than the instrlimit on a 2GHz machine
  return secs

# Kinds of jobs a worker can be asked to do
JK_EVAL = 0
JK_GATHER = 1
JK_LEARN = 2

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()

    # print("worker",job_kind)

    if job_kind == JK_EVAL:
      (mission,res_filename,script_model_file_path,prob,opts1,opts2) = input
      result = IC.vampire_eval(prob,opts1+opts2)
      q_out.put((job_kind,input,result))
    elif job_kind == JK_GATHER:
      (script_model_file_path,prob,opts) = input
      result = IC.vampire_gather(prob,opts)
      # TODO: possibly, the training data (if large) could go through a file
      q_out.put((job_kind,input,result))
    elif job_kind == JK_LEARN:
      (prob,proof_tuple,parts_model_file_path) = input

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(parts_model_file_path))

      # not sure if there is any after load -- TODO: check if necessary
      for param in local_model.parameters():
        # taken from Optmizier zero_grad, roughly
        if param.grad is not None:
          print("Loaded param with with a grad")
          param.grad.detach_()
          param.grad.zero_()

      learn_model = IC.LearningModel(*local_model,*proof_tuple)
      learn_model.train()
      loss_norms = learn_model.forward()

      something = False
      loss = torch.zeros(1)
      for (l,n) in loss_norms:
        if n > 0:
          something = True
          loss += l/n

      input = (prob,parts_model_file_path) # let's not send back the large proof_tuple
      if not something:
        # can the training example can still be degenerate?
        q_out.put((job_kind,input,None))
      else:
        loss.backward()

        for param in local_model.parameters():
          grad = param.grad
          param.requires_grad = False # to allow the in-place operation just below
          if grad is not None:
            param.copy_(grad)
          else:
            param.zero_()

        # use the same file also for the journey back (which brings the gradients inside the actual params)
        torch.save(local_model.state_dict(), parts_model_file_path)

        q_out.put((job_kind,input,parts_model_file_path))

if __name__ == "__main__":
  # Automating the vamp_eval - model_train - model_export loop.
  # Clooper does it in "continuous" fashion, like it was done in deepier
  #
  # To be called as in: ./clooper.py loop_count parallelism campaign_folder (must exist) exper_folder (will be created - ideally have this on local/scratch) [optionally, last completed loop dir]
  #
  # campaign_folder contains "train.txt", "test.txt" - the files listing the problems, and a bunch of train/test_something.pkl's with results of baseline (non-neural) runs for reporting
  #
  # ./clooper.py 15 70 campaigns/small/ /local/sudamar2/lawa/exper1 [optionally, last completed loop dir]

  # loop_count should be the number of epochs, i.e.
  # - the number of times the master model gets trained (for roughtly as many rounds as there are training examples)
  # - evaluted and
  # - a checkpoint saved

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])
  assert parallelism > 0
  campaign_dir = sys.argv[3]
  exper_dir =  sys.argv[4]

  # start a new expreriment folder
  os.mkdir(exper_dir)

  # Documentation: save hyperparams and campaign
  # (CAREFUL: this way can only call looper from lawa folder)
  shutil.copy("hyperparams.py",exper_dir)
  with open(os.path.join(exper_dir,"campaign.txt"),"w") as f:
    f.write(campaign_dir)

  # Load training and testing problem lists
  prob_lists = load_train_tests_problem_lists(campaign_dir)

  # load the reference runs from campaign
  baselines = look_for_baselines(campaign_dir)

  model = IC.get_initial_model()
  if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
  elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  loop = 0

  if len(sys.argv) > 5:
    # start from this checkpoint
    load_dir = sys.argv[5]
    model.load_state_dict(torch.load(os.path.join(load_dir,"parts-model-state.tar")))
    optimizer.load_state_dict(torch.load(os.path.join(load_dir,"optimizer-state.tar")))

    # allow for adapting the params from current HP
    # (TODO: in principle a larger class of things can be adaptively changed after resumig the training process)
    if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
      for param_group in optimizer.param_groups:
        param_group['lr'] = HP.LEARNING_RATE
        param_group['momentum'] = HP.MOMENTUM
    elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
      for param_group in optimizer.param_groups:
        param_group['lr'] = HP.LEARNING_RATE
        param_group['weight_decay'] = HP.WEIGHT_DECAY
  else:
    # since we were not loading, let's save the initial model instead
    assert loop == 0
    cur_dir = claim_loop_dir(loop)
    save_model_and_optimizer(cur_dir,model,optimizer)

  print_model_part()

  assert loop_count > 0

  # create our worker processes and register a cleanup
  q_in = multiprocessing.Queue()
  q_out = multiprocessing.Queue()
  my_processes = []
  for i in range(parallelism):
    p = multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  def cleanup():
    for p in my_processes:
      p.kill()
  atexit.register(cleanup)

  # only after the forks, otherwise weird trouble
  '''
  terminate called after throwing an instance of 'c10::Error'
  what():  pool INTERNAL ASSERT FAILED at "../aten/src/ATen/ParallelOpenMP.cpp":65, please report a bug to PyTorch. Invalid thread pool!
  Exception raised from set_num_threads at ../aten/src/ATen/ParallelOpenMP.cpp:65 (most recent call first):
  ...
  and this would happen only in conjunction with calling print_model_part, i.e., super-weird!
  see also
  https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
  or maybe (pytorch specific help on this seemed scarce)
  https://github.com/pytorch/pytorch/issues/75147
  '''
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  num_active_tasks = 0

  script_model_version = 0
  script_model_ref_counts = defaultdict(int)
  parts_model_version = 0
  grad_loader_temp = IC.get_initial_model()

  while True:
    loop += 1
    if loop > loop_count:
      break

    # for this iteration, we write stuff here:
    cur_dir = claim_loop_dir(loop)

    eval_jobs = []

    # reset every loop:
    result_metas = [] # res_filename(s) with additional info, in order
    result_dicts = defaultdict(dict) # for saving results to file
    solved_accum = set()

    # stages can be thought of as indices into MISSIONS
    stage = -1
    mission = None
    # the overall flow kind of continues after "stage += 1" below by creating the training-eval jobs (and later test-eval josbs)

    loop_start_time = time.time()

    while True:
      if num_active_tasks < parallelism:
        if eval_jobs:
          (mission,res_filename,prob,seed,temp,ilim) = eval_jobs.pop()

          # snapshot model for eval
          script_model_file_path = os.path.join(HP.SCRATCH,"script-model_{}_{}.pt".format(os.getpid(),script_model_version))
          if script_model_ref_counts[script_model_file_path] > 0:
            # just mark as used one more time
            script_model_ref_counts[script_model_file_path] += 1
          else:
            # export current script_model_version for eval
            IC.export_model(model.state_dict(),script_model_file_path)
            script_model_ref_counts[script_model_file_path] = 1

          # will change for the data gathering job
          opts1 = "-t {} -i {} -p off".format(ilim2tlim(ilim),ilim)
          # will stay the same
          opts2 = " --random_seed {} -npcc {} -npcct {}".format(seed,script_model_file_path,temp)

          q_in.put((JK_EVAL,(mission,res_filename,script_model_file_path,prob,opts1,opts2)))
          num_active_tasks += 1

          continue # could we immediately distribute more?

      # either no jobs left in queues or all workers busy

      if num_active_tasks == 0:
        # no worker busy -> no jobs left
        assert not eval_jobs

        if stage >= 0:
          print(" ",mission)
          seen_successes = set()
          best_solveds = -1
          best_results = None
          # all results arrived now, let's save and clear
          for (res_filename,temp,seed,ilim) in result_metas:
            results = result_dicts[res_filename]
            torch.save(("temp: {} seed: {} ilim:".format(temp,seed,ilim),results), os.path.join(cur_dir,res_filename))

            successes = {prob for prob,(status,instructions,activations) in results.items() if status == "uns"}
            num_new = len(successes-seen_successes)
            seen_successes = seen_successes | successes
            print("    t={}  {:10.4f}% = {} / {}   +{} (accum {})".format(temp,len(successes)/len(results),len(successes),len(results),num_new,len(seen_successes)))

            if len(successes) > best_solveds:
              best_solveds = len(successes)
              best_results = results

          print()
          print("  Comparing best",mission,"to baseline")
          compare_to_baselines(best_results,baselines[mission])

          result_dicts = defaultdict(dict)
          result_metas = []
          solved_accum = set()
          sys.stdout.flush()

          print("  Mission {} took".format(mission),time.time()-stage_start_time)
          print()
          sys.stdout.flush()

          if mission == "train":
            save_model_and_optimizer(cur_dir,model,optimizer)
            print_model_part()

        stage += 1
        if stage == len(MISSIONS):
          break
        mission = MISSIONS[stage]

        stage_start_time = time.time()

        # create a batch of eval jobs (for training the pipeline will continue, for testing it ends)

        # we store abstract jobs in eval_jobs (because "model" is drifing and will be only fixed just before dispatch to the worker)
        # an abstract eval job is (mission,result_file_name,problem,seed,temperature,instr_limit)
        ilim = HP.INSTRUCTION_LIMIT if mission == "train" else HP.INSTRUCTION_LIMIT_TEST
        for ti,temp in enumerate(HP.TEMPERATURES):
          res_filename = "{}_t{}_ti{}.pt".format(mission,temp,ti)
          seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp
          eval_jobs += [(mission,res_filename,prob,seed,temp,ilim) for prob in prob_lists[mission]]
          result_metas.append((res_filename,temp,seed,ilim))
        # nice to take these jobs in random order every time around
        if mission == "train":
          # no need to shuffle for eval (although it's nice to have, of course)
          random.shuffle(eval_jobs)

        continue

      # result collecting (workers get a new job immediately, or get freed up)

      (job_kind,input,result) = q_out.get() # this may block

      # process result, depending on what the job was
      if job_kind == JK_EVAL:
        (mission,res_filename,script_model_file_path,prob,opts1,opts2) = input
        result_dicts[res_filename][prob] = result

        (status,instructions,activations) = result

        if mission == "train" and status == "uns" and not (HP.FIRST_PROOF_ONLY and prob in solved_accum):
          solved_accum.add(prob)
          # turn this into gather job
          ilim = 10*HP.INSTRUCTION_LIMIT
          q_in.put((JK_GATHER,(script_model_file_path,prob,"-t {} -i {} -spt on".format(ilim2tlim(ilim),ilim)+opts2)))
        else:
          # don't need the script model file anymore ...
          script_model_ref_counts[script_model_file_path] -= 1
          if script_model_ref_counts[script_model_file_path] == 0:
            os.remove(script_model_file_path)
          num_active_tasks -= 1

      elif job_kind == JK_GATHER:
        (script_model_file_path,prob,opts) = input

        if result is not None:
          # don't need the script model file anymore ...
          script_model_ref_counts[script_model_file_path] -= 1
          if script_model_ref_counts[script_model_file_path] == 0:
            os.remove(script_model_file_path)
        else:
          pass
          # here we are leaking the script model file on purpose,
          # (there should be an error message in the logs, showing how this gather run failed (5 times))

        if result is not None and result[0]: # result[0] = "non-degenerate"
          parts_model_version += 1
          parts_model_file_path = os.path.join(HP.SCRATCH,"parts-model-state_{}_{}.tar".format(os.getpid(),parts_model_version))
          torch.save(model.state_dict(), parts_model_file_path)

          q_in.put((JK_LEARN,(prob,result[1],parts_model_file_path)))
        else:
          # trivial proof, let's keep going
          num_active_tasks -= 1

      elif job_kind == JK_LEARN:
        (prob,parts_model_file_path) = input

        if result is not None:
          # copy from result parameters to our model's gradients
          grad_loader_temp.load_state_dict(torch.load(parts_model_file_path))
          # copy_grads_back_from_param
          for param, param_copy in zip(model.parameters(),grad_loader_temp.parameters()):
            param.grad = param_copy

          optimizer.step()
          script_model_version +=1
        else:
          print("Training for",prob,"was trivial.")

        os.remove(parts_model_file_path)
        num_active_tasks -= 1

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()
