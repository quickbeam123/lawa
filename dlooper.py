#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque

import multiprocessing

import numpy

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = str(HP.INTRAOP_PARALLELISM)
os.environ["OPENBLAS_NUM_THREADS"] = str(HP.INTRAOP_PARALLELISM)
os.environ["MKL_NUM_THREADS"] = str(HP.INTRAOP_PARALLELISM)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(HP.INTRAOP_PARALLELISM)
os.environ["NUMEXPR_NUM_THREADS"] = str(HP.INTRAOP_PARALLELISM)
import torch

# TODO: is float64 wasteful? (note we do everything in doubles on the vampire side!)
# torch.set_default_dtype(torch.float64)

MISSIONS = ["train","test"]

def print_model_part():
  for i in range(HP.NUM_TWEAKS+1):
    print("Key {}".format(repr(model.getKey(idx=i).weight.data)))
  pass

def load_train_tests_problem_lists(campaign_dir):
  prob_lists = {}
  for mission in MISSIONS:
    prob_list_file = os.path.join(campaign_dir,mission+".txt")
    with open(prob_list_file,"r") as f:
      lines = f.readlines()
      prob_lists[mission] = [line.rstrip() for line in lines]
  return prob_lists

def claim_loop_dir(loop):
  loop_str = "loop{}".format(loop)
  print(loop_str)
  sys.stdout.flush()
  cur_dir = os.path.join(exper_dir,loop_str)
  os.mkdir(cur_dir)
  return cur_dir

INFO_MODEL_AND_OPTIMIZER = "info-model-and-optimizer.tar"

def save_info_model_and_optimizer(cur_dir,loop,model,optimizer):
  loop_model_and_optimizer_state_file_path = os.path.join(cur_dir,INFO_MODEL_AND_OPTIMIZER)
  torch.save((loop,HP.NUM_TWEAKS,HP.ACTIVE_FROM,model.state_dict(),optimizer.state_dict()), loop_model_and_optimizer_state_file_path)

def possibly_load_info_model_and_optimizer_state(load_dir):
  loop_model_and_optimizer_state_file_path = os.path.join(load_dir,INFO_MODEL_AND_OPTIMIZER)
  if os.path.exists(loop_model_and_optimizer_state_file_path):
    (loop,num_tweaks,active_from,model_state_dict,optimizer_state_dict) = torch.load(loop_model_and_optimizer_state_file_path)
    print("Loaded model and optimizer from",load_dir,"with loop/num_tweaks/active_from:",loop,num_tweaks,active_from)
    model.load_state_dict(model_state_dict,strict=False)
    try:
      optimizer.load_state_dict(optimizer_state_dict)
    except ValueError:
      # adding more tweaks, probably.
      # It's not so bad to keep the optimizer in it's default state...
      pass
    return True,(loop,num_tweaks,active_from,model_state_dict,optimizer_state_dict)
  return False,()

def get_empty_trace_index():
  return defaultdict(IC.default_defaultdict_of_list) # problem -> temp -> [(loop when obtained,trace_file_name)]

def trace_index_content_summary(index):
  num_probs = len(index)
  num_prob_temps = sum(len(temp_dict) for _prob,temp_dict in index.items())
  num_traces = sum(len(trace_list) for prob,temp_dict in index.items() for temp, trace_list in temp_dict.items())
  return "{} probs, {} prob_temps, and {} traces in total.".format(num_probs,num_prob_temps,num_traces)

TRACE_INDEX = "trace-index.pt"

def save_trace_index(cur_dir,trace_index):
  trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
  torch.save(trace_index, trace_index_file_path)

def possibly_load_trace_index(load_dir,old_index):
  trace_index_file_path = os.path.join(load_dir,TRACE_INDEX)
  if os.path.exists(trace_index_file_path):
    new_index = torch.load(trace_index_file_path)
    print("Loaded a trace_index with",trace_index_content_summary(new_index))
    return new_index
  return old_index

TWEAK_MAP = "tweak_map.pt"

def get_empty_tweak_map():
  # each problem (both train and test) and a temp are associated with at most one tweak, a list of floats of len = HP.NUM_TWEAKS
  return defaultdict(dict) # problem -> temp -> tweak (which is a list)

def enum_all_tweaks(tweak_map):
  for _prob,temp_dict in tweak_map.items():
    for _temp, tweak in temp_dict.items():
      yield tweak

def save_tweak_map(cur_dir,tweak_map):
  tweak_map_file_path = os.path.join(cur_dir,TWEAK_MAP)
  torch.save(tweak_map, tweak_map_file_path)

def possibly_load_tweak_map(load_dir,old_map):
  tweak_map_file_path = os.path.join(load_dir,TWEAK_MAP)
  if os.path.exists(tweak_map_file_path):
    new_map = torch.load(tweak_map_file_path)
    num_probs = len(new_map)
    num_prob_temps = sum(len(temp_dict) for _prob,temp_dict in new_map.items())
    print("Loaded tweak map from",load_dir,"for",num_probs,"probs and",num_prob_temps,"prob_temps")
    for _prob,temp_dict in new_map.items():
      for _temp, tweak in temp_dict.items():
        while len(tweak) < HP.NUM_TWEAKS:
          tweak.append(0.0)
    return new_map
  return old_map

def ilim2tlim(ilim):
  secs = max(5,ilim // 200) # it's 10 times more than the instrlimit on a 2GHz machine
  return secs

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,mission,prob,temp,used_tweak,opts1,opts2)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,temp,opts)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL = 2    # construct our network to get the loss of this trace (no training to do)
  # input:     (prob,temp,tweak_start,fact,trace_file_path,model_file_path)
  # output:    (last_loss,last_tweak,always_improved)
JK_TRAIN = 3   # with train also do loss.backward and send the gradients back
  # input:     (prob,temp,used_tweak,fact,trace_file_path,model_file_path)
  # output:    the computed loss # the out tweak will be recovered from the returned model_file_path (after optimizer step)

def get_trace_file_path(prob,loop,temp):
  return os.path.join(traces_dir,"loop{}_{}_{}.pt".format(loop,prob.replace("/","_"),temp))

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(HP.INTRAOP_PARALLELISM)

  while True:
    (job_kind,input) = q_in.get()

    if job_kind == JK_PERFORM:
      (res_filename,mission,prob,temp,used_tweak,opts1,opts2) = input
      result = IC.vampire_perfrom(prob,opts1+opts2)
      q_out.put((job_kind,input,result))
    elif job_kind == JK_GATHER:
      (prob,temp,trace_file_path,opts) = input
      result = IC.vampire_gather(prob,opts)
      non_degenerate = False
      if result is not None and result[0]: # non-degenerate
        non_degenerate = True
        torch.save(result[1],trace_file_path)
      q_out.put((job_kind,input,non_degenerate))
    elif job_kind == JK_EVAL:
      (prob,temp,tweak_start,fact,trace_file_path,model_file_path) = input

      proof_tuple = torch.load(trace_file_path)

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      always_improved = True

      if HP.NUM_TWEAKS == 0:
        learn_model = IC.LearningModel(local_model,*proof_tuple)
        learn_model.eval()
        # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
        # print(tweaks_to_try)
        loss = learn_model.forward([tweak_start])
        out_tweak = tweak_start
        # print("min-ning at",min_idx)
      else:
        local_optimizer = torch.optim.Adam([{'params': local_model.getTweaksAsParams(), 'lr': HP.TWEAKS_LEARNING_RATE}])
        local_model.train()

        out_tweak = tweak_start
        # print("Starting with",out_tweak)
        last_loss = float('inf')

        for i in range(HP.VALIDATION_TWEAK_DESCENT_STEPS):
          local_optimizer.zero_grad()
          learn_model = IC.LearningModel(local_model,*proof_tuple)
          loss = learn_model.forward([out_tweak])
          loss.backward()
          local_optimizer.step()
          out_tweak = local_model.getTweakVals()

          now_loss = loss.item()
          if now_loss >= last_loss:
            always_improved = False
          last_loss = now_loss

          # print("Step",i,"loss",now_loss,out_tweak)

      q_out.put((job_kind,input,(fact*loss.item(),out_tweak,always_improved)))
    elif job_kind == JK_TRAIN:
      (prob,temp,used_tweak,fact,trace_file_path,model_file_path) = input

      proof_tuple = torch.load(trace_file_path)

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      learn_model = IC.LearningModel(local_model,*proof_tuple)
      learn_model.train()
      loss = learn_model.forward([used_tweak])

      loss.backward()

      for param in local_model.parameters():
        grad = param.grad
        param.requires_grad = False # to allow the in-place operation just below
        if grad is not None:
          param.copy_(grad)
        else:
          param.zero_()

      # use the same file also for the journey back (which brings the gradients inside the actual params)
      torch.save(local_model.state_dict(), model_file_path)

      q_out.put((job_kind,input,fact*loss.item()))

if __name__ == "__main__":
  # Automating the vamp_perform - model_train - model_export loop.
  # Dlooper uses the general flow of clooper but tries sever new ideas:
  # - gather traces even on test problems, so that we can "train until test-loss does not improve" (careful, this is no longer unbiased wrt the test set)
  # - as we do this, we can start reporting the loss observed (both on train and test) and create stats on "what value of loss is enough for this problem to get solved (and under how many iters)"
  # - all these gathered pythonized vampire runs (traces) will go into files and will get communicated with the workers that way (too much RAM used started hurting clooper under CUMULATIVE)
  # - Maybe: subsample long derivation to (HYPERPARAM proper training steps) for some speed (although still need to replay the whole journal and evaluted all mentioned clauses)
  # (to get non-drifting values, we give up on the ``continuous'' aspect idea - TODO: do we have to, really?)
  # - also store the loop index (let it keep growing across interrupted runs)
  #
  # To be called as in: ./dlooper.py loop_count parallelism campaign_folder (must exist) exper_folder (will be created - ideally have this on local/scratch) [optionally, last completed loop dir]
  #
  # campaign_folder contains "train.txt", "test.txt" - the files listing the problems, and a bunch of train/test_something.pkl's with results of baseline (non-neural) runs for reporting
  #
  # ./dlooper.py 15 70 campaigns/small/ /local/sudamar2/lawa/exper1 [optionally, last completed loop dir]

  # loop_count should be the number of epochs, i.e.
  # - the number of times the master model gets trained (for roughtly as many rounds as there are training examples)
  # - evaluted and
  # - a checkpoint saved

  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])
  assert parallelism > 0
  campaign_dir = sys.argv[3]
  exper_dir =  sys.argv[4]

  # Start a new expreriment folder to contain
  # 1) hyperparms and campaing we ran on
  # 2) loop folders each with train / test results and the model/optimizer from that loop
  # 3) a traces folder with eval / train trace files (which are persistent across the loops but get overwritten to save space in each successive gather stage)
  os.mkdir(exper_dir)
  traces_dir = os.path.join(exper_dir,"traces")
  # NOTE: if we want to share traces with a previous run we use CUMULATIVE and load a trace index which will point to old exper's traces folder (until overwritten)
  os.mkdir(traces_dir)

  # Documentation: save hyperparams and campaign
  # (CAREFUL: this way can only call looper from lawa folder)
  shutil.copy("hyperparams.py",exper_dir)
  with open(os.path.join(exper_dir,"campaign.txt"),"w") as f:
    f.write(campaign_dir)

  # Load training and testing problem lists
  prob_lists = load_train_tests_problem_lists(campaign_dir)

  # Initializing a model and an optimizer (might still get a better one below from load_dir if given)
  model = IC.get_initial_model()
  parameter_list = [{'params':model.getActiveParams()},{'params': model.getTweaksAsParams(), 'lr': HP.TWEAKS_LEARNING_RATE}]
  if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
    optimizer = torch.optim.SGD(parameter_list, lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
  elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
    optimizer = torch.optim.Adam(parameter_list, lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  trace_index = get_empty_trace_index()

  tweak_map = get_empty_tweak_map()
  # note that when loaded, these tweaks might be too short and the default semantics should be to pad each by 0s to the desired length
  assert HP.NUM_TWEAKS == 0 or HP.CUMULATIVE > 0, "let's not think about what it should mean to work with proper tweaks and not be CUMULATIVE"

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

  loop = 0
  loaded_model = False
  if len(sys.argv) > 5:
    # start from this checkpoint
    load_dir = sys.argv[5]

    res,data = possibly_load_info_model_and_optimizer_state(load_dir)
    if res:
      loaded_model = True
      (loop,num_tweaks,active_from,model_state_dict,optimizer_state_dict) = data
      # TODO: readujst the model if new networks and tweaks are needed
      # TODO: also possibly rewire the optimizer to account for what's newly learnable!

    trace_index = possibly_load_trace_index(load_dir,trace_index)
    tweak_map = possibly_load_tweak_map(load_dir,tweak_map)
    # TODO: also here, we might need new tweaks! (or we pad them on demand)

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

  if not loaded_model:
    # since we were not loading, let's save the initial model instead
    assert loop == 0
    cur_dir = claim_loop_dir(loop)
    save_info_model_and_optimizer(cur_dir,loop,model,optimizer)

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

  def do_in_parallel(tasks,max_parallelism,process_results_callback):
    num_active_tasks = 0

    # we assume there is at least one task
    have_tasks = True
    while have_tasks or num_active_tasks:
      # first of all: make all the workers busy, if possible
      if have_tasks and num_active_tasks < max_parallelism:
        # we assume tasks are not None
        task = next(tasks,None)
        if task is None:
          have_tasks = False
        else:
          # print("PUT:",task)
          q_in.put(task)
          num_active_tasks += 1
        continue

      # result collecting (workers get a new job immediately, or get freed up)
      (job_kind,input,result) = q_out.get() # this may block
      # print("GOT:",(job_kind,input,result))

      num_active_tasks -= process_results_callback(job_kind,input,result)

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
  torch.set_num_interop_threads(HP.INTRAOP_PARALLELISM)

  while True:
    if loop_count == 0:
      break
    loop += 1
    loop_count -= 1

    loop_start_time = time.time()

    have_tweak_std = False
    if len(tweak_map) > 0:
      have_tweak_std = True
      tweak_std = numpy.std(list(enum_all_tweaks(tweak_map)),axis=0)
      print("  tweak_std now",tweak_std)

    # for this iteration, we write stuff here:
    cur_dir = claim_loop_dir(loop)

    if HP.CUMULATIVE == 0: # forget what we solved until now
      trace_index = get_empty_trace_index()

    # STAGE 1: PERFORM and GATHER (TODO: could/should be done in one call to workers?)
    stage_start_time = time.time()

    result_metas = [] # res_filename(s) with additional info, in order
    result_dicts = defaultdict(dict) # for saving results to a file

    script_model_file_path = os.path.join(cur_dir,"script-model.pt")
    IC.export_model(model.state_dict(),script_model_file_path)

    def get_perform_tasks():
      # for both missions and all temperatures
      for mission in MISSIONS:
        # don't test with bling tweaks (for now)
        if HP.NUM_TWEAKS > 0 and mission == "test":
          continue

        ilim = HP.INSTRUCTION_LIMIT if mission == "train" else HP.INSTRUCTION_LIMIT_TEST
        for ti,temp in enumerate(HP.TEMPERATURES):  # we us ti in the filename too, for the case temps are repeated
          res_filename = "{}_t{}_ti{}.pt".format(mission,temp,ti)
          result_metas.append((res_filename,mission,temp,ilim))

          # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
          opts1 = "-t {} -i {} -p off".format(ilim2tlim(ilim),ilim)
          # will stay the same
          opts2_base = " -npcc {} -nnf {} -npcct {}".format(script_model_file_path,HP.NUM_FEATURES,temp)

          for prob in prob_lists[mission]:
            if HP.NUM_TWEAKS == 0:
              yield (JK_PERFORM,(res_filename,mission,prob,temp,[],opts1,opts2_base + " --random_seed {}".format(random.randint(1,0x7fffff))))
            else:
              for _ in range(HP.NUM_PERFORMS_ON_HARD_PROBLEMS if prob not in trace_index else 1):
                if prob in tweak_map and temp in tweak_map[prob]:
                  used_tweak = tweak_map[prob][temp]
                  # print("Reusing tweak",used_tweak,"from last time for",prob)
                elif have_tweak_std:
                  used_tweak = list(numpy.random.normal(HP.NUM_TWEAKS*[0.0],tweak_std))
                  # print("Will try tweak",used_tweak,"for",prob)
                  # TODO: there also was the option to draw a tweak that already exists somewhere in the map
                  # (but with a different problem)
                  # in any case, this is only interesting for solving more problems during training,
                  # what we don't have a cheap solution for how to reasoably search for tweaks for the test problems
                else:
                  used_tweak = HP.NUM_TWEAKS*[0.0,]
                  # print("Defaulting tweak",used_tweak,"for",prob)

                tweak_str = ",".join([str(t) for t in used_tweak])
                # print(tweak_str)
                opts2 = opts2_base + " -npccw {} --random_seed {}".format(tweak_str,random.randint(1,0x7fffff))

                yield (JK_PERFORM,(res_filename,mission,prob,temp,used_tweak,opts1,opts2))

    def process_results_from_perform_and_gather(job_kind,input,result):
      workers_freed = 1
      if job_kind == JK_PERFORM:
        (res_filename,mission,prob,temp,used_tweak,opts1,opts2) = input
        (status,instructions,activations) = result

        had_unsat_already = (prob in result_dicts[res_filename] and result_dicts[res_filename][prob][0] == "uns")

        # we add things conditionally, because under NUM_PERFORMS_ON_HARD_PROBLEMS > 1, we want to keep the good results
        if not had_unsat_already:
          result_dicts[res_filename][prob] = result

        if status == "uns":
          tweak_map[prob][temp] = used_tweak

          if mission == "train" and not had_unsat_already:
            ilim = 10*HP.INSTRUCTION_LIMIT
            task = (JK_GATHER,(prob,temp,get_trace_file_path(prob,loop,temp),"-t {} -i {} -spt on".format(ilim2tlim(ilim),ilim)+opts2))
            # print("PUT:",task)
            q_in.put(task)
            workers_freed = 0 # as we keep computing the JK_GATHER job

      elif job_kind == JK_GATHER:
        (prob,temp,trace_file_path,opts) = input
        non_degenerate = result
        if non_degenerate:
          # the trace pkl has been saved to a file, let's store it in trace_index (and potentially evict some older ones)
          target_list = trace_index[prob][temp]
          target_list.append((loop,trace_file_path))
          while len(target_list) > max(1,HP.CUMULATIVE):
            old_loop,old_trace_file_path = target_list[0]
            # print("Evicting trace",old_trace_file_path,"of loop",old_loop)
            if old_trace_file_path.startswith(traces_dir):
              # don't delete files from older experiments!
              os.remove(old_trace_file_path)
            else:
              pass
              # print("Didn't delete an oldie!")
            del target_list[0]
      else:
        assert False, f"Surprised by job_kind {job_kind}"

      return workers_freed

    do_in_parallel(get_perform_tasks(),parallelism,process_results_from_perform_and_gather)

    # let's report what happened so far:
    prev_mission = None
    for (res_filename,mission,temp,ilim) in result_metas:
      if mission != prev_mission:
        print(" ",mission)
        seen_successes = set()
        prev_mission = mission
      results = result_dicts[res_filename]
      torch.save(("temp: {} ilim: {}".format(temp,ilim),results), os.path.join(cur_dir,res_filename))

      successes = {prob for prob,(status,instructions,activations) in results.items() if status == "uns"}
      num_new = len(successes-seen_successes)
      seen_successes = seen_successes | successes
      if len(results):
        print("    t={}  {:10.4f}% = {} / {}   +{} (accum {})".format(temp,100.0*len(successes)/len(results),len(successes),len(results),num_new,len(seen_successes)))
    print()
    print("  Stage 1 took",time.time()-stage_start_time)
    print()
    sys.stdout.flush()

    if HP.CUMULATIVE > 0:
      save_trace_index(cur_dir,trace_index)
      print("Saving trace_index with",trace_index_content_summary(trace_index))
      print()
      sys.stdout.flush()

    # STAGE 2: alternate EVAL, TRAIN, EVAL until no longer improving
    stage_start_time = time.time()

    VIW = HP.VALID_IMPROVE_WINDOW
    assert VIW > 0
    eval_models = [None]*VIW
    eval_losses = [None]*VIW
    stage2iter = 0

    if VIW > 1:
      traced_problems = list(trace_index.keys())
      random.shuffle(traced_problems)
      for_validation = set(traced_problems[:int(HP.VALIDATION_SET_FRAC*len(traced_problems))])
    else:
      for_validation = set()
    print("Will train on",len(trace_index)-len(for_validation),"problems,",len(for_validation),"set aside for validation.")

    while True:
      if VIW > 1:
        # EVAL on validation problems
        eval_model_file_path = os.path.join(HP.SCRATCH,"eval-model-state_{}_{}.tar".format(os.getpid(),stage2iter))
        torch.save(model.state_dict(), eval_model_file_path)
        if eval_models[stage2iter % VIW] is not None:
          os.remove(eval_models[stage2iter % VIW])
        eval_models[stage2iter % VIW] = eval_model_file_path

        fact = 1/len(for_validation)
        tasks = ((JK_EVAL,(prob,temp,tweak_map[prob][temp] if prob in tweak_map and temp in tweak_map[prob] else HP.NUM_TWEAKS*[0.0,],
          fact/len(temp_dict),loop_trace_file_path_list[0][1],eval_model_file_path))
          for prob, temp_dict in trace_index.items() for temp,loop_trace_file_path_list in temp_dict.items() if prob in for_validation)

        weighted_validation_loss = 0.0
        num_valids = 0
        num_improves = 0

        def process_results_from_eval(job_kind,input,result):
          global weighted_validation_loss
          global num_valids
          global num_improves

          assert job_kind == JK_EVAL
          (prob,temp,tweak_start,fact,trace_file_path,model_file_path) = input
          (loss,out_tweak,always_improved) = result

          weighted_validation_loss += loss # multiplied by fact already in the child

          # print("Validating on",prob,"updated tweak from",tweak_start,"to",out_tweak)
          tweak_map[prob][temp] = out_tweak

          num_valids += 1
          if always_improved:
            num_improves += 1

          return 1

        do_in_parallel(tasks,parallelism,process_results_from_eval)

        print("Weighted valid-loss",weighted_validation_loss,"validated on",num_valids,"always improved on",num_improves)
        sys.stdout.flush()
        eval_losses[stage2iter % VIW] = weighted_validation_loss
        stage2iter += 1
        if stage2iter >= VIW: # we have written everywhere (no None there anymore)
          oldest_idx = stage2iter % VIW
          oldest_val = eval_losses[oldest_idx]
          if all((el >= oldest_val for el in eval_losses)):
            print("Validation loss didn't improve for",VIW-1,"iterations now")
            if stage2iter == VIW:
              # TODO: halve the LR when this happens?
              print("Actually, it never improved! Will apply one training step anyway!")
              model.load_state_dict(torch.load(eval_models[1]))
            else:
              model.load_state_dict(torch.load(eval_models[oldest_idx]))
            for eval_model_file_path in eval_models:
              os.remove(eval_model_file_path)
            break
          if stage2iter > HP.MAX_TEST_IMPROVE_ITER:
            print("Taking too long to converge (stage2iter > HP.MAX_TEST_IMPROVE_ITER), will take the best from the last HP.TEST_IMPROVE_WINDOW observed.")
            best_idx = 0
            best_idx_val = eval_losses[0]
            for i,eloss in enumerate(eval_losses):
              if eloss < best_idx_val:
                best_idx_val = eloss
                best_idx = i
            model.load_state_dict(torch.load(eval_models[best_idx]))
            for eval_model_file_path in eval_models:
              os.remove(eval_model_file_path)
            break

      # TRAIN on train problems
      for _ in range(HP.NUM_TRAIN_CYCLES_BETWEEN_EVALS):
        train_model_version = 0
        def get_train_tasks():
          fact = 1/(len(trace_index)-len(for_validation))
          proto_tasks = [[prob,temp,tweak_map[prob][temp] if prob in tweak_map and temp in tweak_map[prob] else HP.NUM_TWEAKS*[0.0,],fact/len(temp_dict),loop_trace_file_path_list[0][1]]
            for prob, temp_dict in trace_index.items() for temp,loop_trace_file_path_list in temp_dict.items() if prob not in for_validation]
          random.shuffle(proto_tasks)

          global train_model_version
          for arg_list in proto_tasks:
            train_model_version += 1
            train_model_file_path = os.path.join(HP.SCRATCH,"train-model-state_{}_{}.tar".format(os.getpid(),train_model_version))
            torch.save(model.state_dict(), train_model_file_path)
            arg_list.append(train_model_file_path)
            yield (JK_TRAIN,tuple(arg_list))

        weighted_train_loss = 0.0

        def process_results_from_train(job_kind,input,result):
          global weighted_train_loss

          assert job_kind == JK_TRAIN
          (prob,temp,used_tweak,fact,trace_file_path,train_model_file_path) = input
          loss = result

          weighted_train_loss += loss # multiplied by fact already in the child
          # print(input,result)

          model.setTweakVals(used_tweak)

          # copy from result parameters to our model's gradients
          grad_loader_temp.load_state_dict(torch.load(train_model_file_path))
          # copy_grads_back_from_param
          for param, param_copy in zip(model.parameters(),grad_loader_temp.parameters()):
            param.grad = param_copy

          optimizer.step()

          # here we have a new updated tweak inside model
          new_tweak = model.getTweakVals()
          # print("Training on",prob,"under temp",temp,"Updated tweak from",used_tweak,"to",new_tweak)
          tweak_map[prob][temp] = new_tweak

          os.remove(train_model_file_path)
          return 1

        do_in_parallel(get_train_tasks(),min(parallelism,HP.TRAINING_PARALLELISM),process_results_from_train)

        print("Weighted train-loss",weighted_train_loss)
        sys.stdout.flush()

      if VIW == 1:
        break

    # stage 2
    print("  Stage 2 took",time.time()-stage_start_time,"seconds and",stage2iter-0.5,"eval/train iterations")
    print()
    sys.stdout.flush()

    print_model_part()
    save_info_model_and_optimizer(cur_dir,loop,model,optimizer)
    save_tweak_map(cur_dir,tweak_map)

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()
