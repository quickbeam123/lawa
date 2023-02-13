#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque

import multiprocessing

import numpy

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

# TODO: is float64 wasteful? (note we do everything in doubles on the vampire side!)
# torch.set_default_dtype(torch.float64)

MISSIONS = ["train","test"]

def print_model_part():
  print("Key {}".format(repr(model.getKey().weight.data)))
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

LOOP_MODEL_AND_OPTIMIZER = "loop-model-and-optimizer.tar"

def save_loop_model_and_optimizer(cur_dir,loop,model,optimizer):
  loop_model_and_optimizer_state_file_path = os.path.join(cur_dir,LOOP_MODEL_AND_OPTIMIZER)
  torch.save((loop,model.state_dict(),optimizer.state_dict()), loop_model_and_optimizer_state_file_path)

def possibly_load_loop_model_and_optimizer_state(load_dir,loop,model,optimizer):
  loop_model_and_optimizer_state_file_path = os.path.join(load_dir,LOOP_MODEL_AND_OPTIMIZER)
  if os.path.exists(loop_model_and_optimizer_state_file_path):
    (loop,model_state_dict,optimizer_state_dict) = torch.load(loop_model_and_optimizer_state_file_path)
    print("Loaded model and optimizer from",load_dir,"with loop",loop)
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    return loop,True
  return loop,False

def get_empty_trace_index():
  return { m : defaultdict(dict) for m in MISSIONS} # mission -> problem -> temp -> (loop when obtained,trace_file_name)

TRACE_INDEX = "trace-index.pt"

def save_trace_index(cur_dir,trace_index):
  trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
  torch.save(trace_index, trace_index_file_path)

def possibly_load_trace_index(load_dir,old_index):
  trace_index_file_path = os.path.join(load_dir,TRACE_INDEX)
  if os.path.exists(trace_index_file_path):
    new_index = torch.load(trace_index_file_path)
    print("Loaded trace index from",load_dir,"with",len(new_index["train"]),"train and",len(new_index["test"]),"test problems registered.")
    # trace index culling (for the case the loaded problems are a superset of current campaign's)
    deleted = False
    for m in MISSIONS:
      relevant = set(prob_lists[m])
      for prob in list(new_index[m]):
        if prob not in relevant:
          del new_index[m][prob]
          deleted = True
    if deleted:
      print("Reduced to",len(new_index["train"]),"train and",len(new_index["test"]),"test problems during culling.")
    return new_index
  return old_index

TWEAK_MAP = "tweak_map.pt"

def save_tweak_map(cur_dir,tweak_map):
  tweak_map_file_path = os.path.join(cur_dir,TWEAK_MAP)
  torch.save(tweak_map, tweak_map_file_path)

def possibly_load_tweak_map(load_dir,old_map):
  tweak_map_file_path = os.path.join(load_dir,TWEAK_MAP)
  if os.path.exists(tweak_map_file_path):
    new_map = torch.load(tweak_map_file_path)
    print("Loaded tweak map from",load_dir,"with",len(new_map),"entries")
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
  # input:     (prob,temp,tweak_start,tweak_std,fact,trace_file_path,model_file_path)
  # output:    (best_loss,best_tweak) out of the HP.NUM_HILL_TRIES_ON_EVAL tries with different tweaks from around tweak_start
JK_TRAIN = 3   # with train also do loss.backward and send the gradients back
  # input:     (prob,temp,used_tweak,fact,trace_file_path,model_file_path)
  # output:    the computed loss # the out tweak will be recovered from the returned model_file_path (after optimizer step)

def get_trace_file_path(mission,prob,temp):
  return os.path.join(traces_dir,"{}_{}_{}.pt".format(mission,prob.replace("/","_"),temp))

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()

    if job_kind == JK_PERFORM:
      (res_filename,mission,prob,temp,used_tweak,opts1,opts2) = input
      result = IC.vampire_perfrom(prob,opts1+opts2)
      q_out.put((job_kind,input,result))
    elif job_kind == JK_GATHER:
      (mission,prob,temp,opts) = input
      result = IC.vampire_gather(prob,opts)
      trace_file_path = None
      if result is not None and result[0]: # non-degenerate
        trace_file_path = get_trace_file_path(mission,prob,temp)
        torch.save(result[1],trace_file_path)
      q_out.put((job_kind,input,trace_file_path))
    elif job_kind == JK_EVAL:
      (prob,temp,tweak_start,tweak_std,fact,trace_file_path,model_file_path) = input

      proof_tuple = torch.load(trace_file_path)

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      learn_model = IC.LearningModel(local_model,*proof_tuple)
      learn_model.eval()

      # TODO: don't do this with ||tweak_std|| = 0, e.g., when we don't do proper tweaking!
      tweaks_to_try = numpy.random.normal(tweak_start,HP.TWEAK_SEARCH_SPREAD_FACTOR*tweak_std,size=(HP.NUM_HILL_TRIES_ON_EVAL,len(tweak_start)))

      # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
      # print(tweaks_to_try)

      losses = learn_model.forward(tweaks_to_try)

      # print("Got",losses)

      min_idx = torch.argmin(losses)

      # print("min-ning at",min_idx)

      q_out.put((job_kind,input,(fact*losses[min_idx].item(),tweaks_to_try[min_idx])))
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
  # - all these gathered pythonized vampire runs (traces) will go into files and will get communicated with the workers that way (too much RAM used started hurting clooper under CUMMULATIVE)
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
  # NOTE: if we want to share traces with a previous run we use CUMMULATIVE and load a trace index which will point to old exper's traces folder (until overwritten)
  os.mkdir(traces_dir)

  # Documentation: save hyperparams and campaign
  # (CAREFUL: this way can only call looper from lawa folder)
  shutil.copy("hyperparams.py",exper_dir)
  with open(os.path.join(exper_dir,"campaign.txt"),"w") as f:
    f.write(campaign_dir)

  # Load training and testing problem lists
  prob_lists = load_train_tests_problem_lists(campaign_dir)

  # Initializing a model and an optimizer (might still get better one below from load_dir if given)
  model = IC.get_initial_model()
  if HP.OPTIMIZER == HP.OPTIMIZER_SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=HP.LEARNING_RATE, momentum=HP.MOMENTUM)
  elif HP.OPTIMIZER == HP.OPTIMIZER_ADAM:
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  trace_index = get_empty_trace_index()

  tweak_map = {} # each problem (both train and test) is associated with at most one tweak, a list of floats of len = HP.NUM_TWEAKS
  assert HP.NUM_TWEAKS == 0 or HP.CUMMULATIVE > 0, "let's not think about what it should mean to work with proper tweaks and not be CUMMULATIVE"

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

  loop = 0
  loaded_model = False
  if len(sys.argv) > 5:
    # start from this checkpoint
    load_dir = sys.argv[5]

    loop,loaded_model = possibly_load_loop_model_and_optimizer_state(load_dir,loop,model,optimizer)
    trace_index = possibly_load_trace_index(load_dir,trace_index)
    tweak_map = possibly_load_tweak_map(load_dir,tweak_map)

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
    save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)

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
  torch.set_num_interop_threads(1)

  while True:
    if loop_count == 0:
      break
    loop += 1
    loop_count -= 1

    loop_start_time = time.time()

    # TODO: at the very fresh beginning, this will not get initialized!
    have_tweak_std = False
    if len(tweak_map) > 0:
      have_tweak_std = True
      tweak_std = numpy.std(list(tweak_map.values()),axis=0)
      print("  tweak_std now",tweak_std)

    # for this iteration, we write stuff here:
    cur_dir = claim_loop_dir(loop)

    if HP.CUMMULATIVE == 0: # forget what we solved until now
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
        ilim = HP.INSTRUCTION_LIMIT if mission == "train" else HP.INSTRUCTION_LIMIT_TEST
        for ti,temp in enumerate(HP.TEMPERATURES):
          seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp

          res_filename = "{}_t{}_ti{}.pt".format(mission,temp,ti)
          result_metas.append((res_filename,mission,temp,seed,ilim))

          # will change for the gathering job
          opts1 = "-t {} -i {} -p off".format(ilim2tlim(ilim),ilim)

          for prob in prob_lists[mission]:
            if prob in tweak_map:
              used_tweak = tweak_map[prob]
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

            # will stay the same
            opts2 = " --random_seed {} -npcc {} -nnf {} -npcct {} -npccw {}".format(seed,script_model_file_path,HP.NUM_FEATURES,temp,tweak_str)

            yield (JK_PERFORM,(res_filename,mission,prob,temp,used_tweak,opts1,opts2))

    def process_results_from_perform_and_gather(job_kind,input,result):
      workers_freed = 0
      if job_kind == JK_PERFORM:
        (res_filename,mission,prob,temp,used_tweak,opts1,opts2) = input
        result_dicts[res_filename][prob] = result

        (status,instructions,activations) = result
        if status == "uns":
          tweak_map[prob] = used_tweak

          ilim = 10*HP.INSTRUCTION_LIMIT
          task = (JK_GATHER,(mission,prob,temp,"-t {} -i {} -spt on".format(ilim2tlim(ilim),ilim)+opts2))
          # print("PUT:",task)
          q_in.put(task)
        else:
          workers_freed = 1
      elif job_kind == JK_GATHER:
        (mission,prob,temp,opts) = input
        trace_file_path = result
        # the trace pkl has been saved to a file, let's just remember that we have it:
        if trace_file_path is not None:
          trace_index[mission][prob][temp] = (loop,trace_file_path)
        workers_freed = 1
      else:
        assert False, f"Surprised by job_kind {job_kind}"

      return workers_freed

    do_in_parallel(get_perform_tasks(),parallelism,process_results_from_perform_and_gather)

    # let's report what happened so far:
    prev_mission = None
    for (res_filename,mission,temp,seed,ilim) in result_metas:
      if mission != prev_mission:
        print(" ",mission)
        seen_successes = set()
        prev_mission = mission
      results = result_dicts[res_filename]
      torch.save(("temp: {} seed: {} ilim:".format(temp,seed,ilim),results), os.path.join(cur_dir,res_filename))

      successes = {prob for prob,(status,instructions,activations) in results.items() if status == "uns"}
      num_new = len(successes-seen_successes)
      seen_successes = seen_successes | successes
      if len(results):
        print("    t={}  {:10.4f}% = {} / {}   +{} (accum {})".format(temp,len(successes)/len(results),len(successes),len(results),num_new,len(seen_successes)))
    print()
    print("  Stage 1 took",time.time()-stage_start_time)
    print()
    sys.stdout.flush()

    if HP.CUMMULATIVE > 0:
      save_trace_index(cur_dir,trace_index)
      print("Registering",len(trace_index["train"]),"train and",len(trace_index["test"]),"test problems in trace_index.")
      print()
      sys.stdout.flush()

    # STAGE 2: alternate EVAL, TRAIN, EVAL until no longer improving
    stage_start_time = time.time()

    TIW = HP.TEST_IMPROVE_WINDOW
    assert TIW > 0
    eval_models = [None]*TIW
    eval_losses = [None]*TIW
    stage2iter = 0

    # given prob's temp_dict in trace_index, tell me how many times we want to think it's present in the our set (when computing overall loss)
    def prob_importance_coef(prob,temp_dict):
      max_loop = 0
      for _temp,(l,_trace_file_path) in temp_dict.items():
        if l > max_loop:
          max_loop = l
      # print("prob_importance_coef for",prob,"is",min(loop-max_loop+1,HP.CUMMULATIVE))
      # the more long time ago the most recent solution was, the more we care about bringing the problem back (i.e., easy problems don't pull on loss too hard)
      return min(loop-max_loop+1,HP.CUMMULATIVE)

    while True:
      if TIW > 1:
        # EVAL on test problems
        eval_model_file_path = os.path.join(HP.SCRATCH,"eval-model-state_{}_{}.tar".format(os.getpid(),stage2iter))
        torch.save(model.state_dict(), eval_model_file_path)
        if eval_models[stage2iter % TIW] is not None:
          os.remove(eval_models[stage2iter % TIW])
        eval_models[stage2iter % TIW] = eval_model_file_path

        # it's not clear whether test loss should reflect the magic-accum formula of clooper (that tries to pull more strongly on not-so-recently solved problems)
        fact = 1/sum(prob_importance_coef(prob,temp_dict) for prob, temp_dict in trace_index["test"].items())
        tasks = ((JK_EVAL,(prob,temp,tweak_map[prob] if prob in tweak_map else HP.NUM_TWEAKS*[0.0,],tweak_std,
          fact*prob_importance_coef(prob,temp_dict)/len(temp_dict),trace_file_path,eval_model_file_path))
          for prob, temp_dict in trace_index["test"].items() for temp,(_,trace_file_path) in temp_dict.items())

        weighted_test_loss = 0.0

        def process_results_from_eval(job_kind,input,result):
          global weighted_test_loss

          assert job_kind == JK_EVAL
          (prob,temp,tweak_start,tweak_std,fact,trace_file_path,model_file_path) = input
          (loss,out_tweak) = result

          weighted_test_loss += loss # multiplied by fact already in the child

          # print("Test eval on",prob,"updated tweak from",tweak_start,"+/-",tweak_std,"to",out_tweak)

          tweak_map[prob] = out_tweak

          return 1

        do_in_parallel(tasks,parallelism,process_results_from_eval)

        print("Weighted test loss",weighted_test_loss)
        sys.stdout.flush()
        eval_losses[stage2iter % TIW] = weighted_test_loss
        stage2iter += 1
        if stage2iter >= TIW: # we have written everywhere (no None there anymore)
          oldest_idx = stage2iter % TIW
          oldest_val = eval_losses[oldest_idx]
          if all((el >= oldest_val for el in eval_losses)):
            print("Eval loss didn't improve for",TIW-1,"iterations now")
            if stage2iter == TIW:
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
          fact = 1/sum(prob_importance_coef(prob,temp_dict) for prob, temp_dict in trace_index["train"].items())
          # TODO: also here we could consider exerting extra force on harder problems (according to how recently they got solved) under CUMMULATIVE
          proto_tasks = [[prob,temp,tweak_map[prob] if prob in tweak_map else HP.NUM_TWEAKS*[0.0,],fact*prob_importance_coef(prob,temp_dict)/len(temp_dict),trace_file_path]
            for prob, temp_dict in trace_index["train"].items() for temp,(_,trace_file_path) in temp_dict.items()]
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

          model.setTweak(used_tweak)

          # copy from result parameters to our model's gradients
          grad_loader_temp.load_state_dict(torch.load(train_model_file_path))
          # copy_grads_back_from_param
          for param, param_copy in zip(model.parameters(),grad_loader_temp.parameters()):
            param.grad = param_copy

          optimizer.step()

          # here we have a new updated tweak inside model
          new_tweak = model.getTweaks()
          # print("Training on",prob,"updated tweak from",used_tweak,"to",new_tweak)
          tweak_map[prob] = new_tweak

          os.remove(train_model_file_path)
          return 1

        do_in_parallel(get_train_tasks(),min(parallelism,HP.TRAINING_PARALLELISM),process_results_from_train)

        print("Weighted train loss",weighted_train_loss)
        sys.stdout.flush()

      if TIW == 1:
        break

    # stage 2
    print("  Stage 2 took",time.time()-stage_start_time,"seconds and",stage2iter-0.5,"eval/train iterations")
    print()
    sys.stdout.flush()

    print_model_part()
    save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)
    save_tweak_map(cur_dir,tweak_map)

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()
