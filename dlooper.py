#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain

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

MISSIONS = ["train","valid"]

def print_model_part():
  print("Key {}".format(repr(model.default_key.weight.data)))
  print("Tweaker {}".format(repr(model.key_tweaker.weight.data)))
  pass

def load_train_valid_problem_lists(campaign_dir):
  prob_lists = {}
  for mission in MISSIONS:
    prob_list_file = os.path.join(campaign_dir,mission+".txt")
    with open(prob_list_file,"r") as f:
      lines = f.readlines()
      cur_list = [line.rstrip() for line in lines]
      prob_lists[mission] = cur_list
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
  return { m : defaultdict(list) for m in MISSIONS} # mission -> problem -> [trace_file_name]

TRACE_INDEX = "trace-index.pt"

def save_trace_index(cur_dir,trace_index):
  trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
  torch.save(trace_index, trace_index_file_path)

def possibly_load_trace_index(load_dir,old_index):
  trace_index_file_path = os.path.join(load_dir,TRACE_INDEX)
  if os.path.exists(trace_index_file_path):
    new_index = torch.load(trace_index_file_path)
    print("Loaded a trace index from",load_dir,"with",len(new_index["train"]),"train and",len(new_index["valid"]),"validation problems registered.")
    return new_index
  return old_index

TWEAK_MAP = "tweak_map.pt"

def get_empty_tweak_map():
  # each problem (both train and valid) are associated with at most one tweak, a list of floats of len = HP.NUM_TWEAKS
  return dict() # problem -> [tweaks]

def save_tweak_map(cur_dir,tweak_map):
  tweak_map_file_path = os.path.join(cur_dir,TWEAK_MAP)
  torch.save(tweak_map, tweak_map_file_path)

def possibly_load_tweak_map(load_dir,old_map):
  tweak_map_file_path = os.path.join(load_dir,TWEAK_MAP)
  if os.path.exists(tweak_map_file_path):
    new_map = torch.load(tweak_map_file_path)
    num_probs = len(new_map)
    print("Loaded tweak map from",load_dir,"for",num_probs,"probs")
    return new_map
  return old_map

def ilim2tlim(ilim):
  secs = max(5,ilim // 1000) # it's 2 times more than the instrlimit on a 2GHz machine
  return secs

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,gatherwish,mission,prob,used_tweak,opts1,opts2)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,counter,opts)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL = 2    # construct our network to get the loss of this trace (no training to do)
  # input:     (prob,fact,descend_tw,tweak_start,trace_file_paths,model_file_path)
  # output:    the computed loss
JK_TRAIN = 3   #
  # input:     (prob,fact,used_tweak,trace_file_paths,train_model_file_path)
  # output:    the computed loss

def get_trace_file_path(mission,prob,counter):
  return os.path.join(traces_dir,"{}_{}_{}.pt".format(mission,prob.replace("/","_"),counter))

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()

    if job_kind == JK_PERFORM:
      (res_filename,gatherwish,mission,prob,used_tweak,opts1,opts2) = input
      result = IC.vampire_perfrom(prob,opts1+opts2)
      q_out.put((job_kind,input,result))

    elif job_kind == JK_GATHER:
      (mission,prob,counter,opts) = input
      result = IC.vampire_gather(prob,opts)
      trace_file_path = None
      if result is not None and result[0]: # non-degenerate
        trace_file_path = get_trace_file_path(mission,prob,counter)
        torch.save(result[1],trace_file_path)
      q_out.put((job_kind,input,trace_file_path))

    elif job_kind == JK_EVAL:
      (prob,fact,descend_tw,tweak_start,trace_file_paths,model_file_path) = input

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      finished = True
      numiter = 0
      start_time = time.time()

      local_fact = 1/len(trace_file_paths)
      proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      if not descend_tw:
        loss = torch.zeros(1)
        for proof_tuple in proof_tuples:
          learn_model = IC.LearningModel(local_model,*proof_tuple)
          learn_model.eval()
          # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
          # print(tweaks_to_try)
          loss += local_fact*learn_model.forward([tweak_start]).item()
        out_tweak = tweak_start
        telapsed = time.time() - start_time
      else:
        local_optimizer = torch.optim.Adam([{'params': local_model.getTweaksAsParams(), 'lr': HP.TWEAKS_LEARNING_RATE}])
        local_model.train()

        out_tweak = tweak_start
        # print("Starting with",out_tweak)
        last_loss = float('inf')

        # TODO: here we could try compiling (with Torch 2.0)!
        while True:
          numiter += 1
          local_optimizer.zero_grad()
          loss = torch.zeros(1)
          for proof_tuple in proof_tuples:
            learn_model = IC.LearningModel(local_model,*proof_tuple)
            loss += local_fact*learn_model.forward([out_tweak])
          loss.backward()
          local_optimizer.step()
          out_tweak = local_model.getTweakVals()

          telapsed = time.time() - start_time
          now_loss = loss.item()

          if now_loss > last_loss:
            break
          last_loss = now_loss

          if telapsed > HP.TWEAK_DESCENT_MAX_SECOND:
            finished = False
            break

        # print("EVALed:",prob,"for",numiter,"iter in time",telapsed,"finished:",finished,"last_loss",last_loss)

      # print(prob,tweak_start,loss.item())

      q_out.put((job_kind,input,(fact*loss.item(),out_tweak,numiter,telapsed,finished)))

    elif job_kind == JK_TRAIN:
      (prob,fact,used_tweak,trace_file_paths,train_model_file_path) = input

      local_fact = 1/len(trace_file_paths)
      proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(train_model_file_path))

      loss = torch.zeros(1)
      for proof_tuple in proof_tuples:
        learn_model = IC.LearningModel(local_model,*proof_tuple)
        learn_model.train()
        loss += local_fact*learn_model.forward([used_tweak])

      loss.backward()

      for param in local_model.parameters():
        grad = param.grad
        param.requires_grad = False # to allow the in-place operation just below
        if grad is not None:
          param.copy_(grad)
        else:
          param.zero_()

      # use the same file also for the journey back (which brings the gradients inside the actual params)
      torch.save(local_model.state_dict(), train_model_file_path)

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

  # Load training and validation problem lists
  prob_lists = load_train_valid_problem_lists(campaign_dir)

  # Initializing a model and an optimizer (might still get better one below from load_dir if given)
  model = IC.get_initial_model()

  optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)
  # optimizer = torch.optim.Adam(model.getTheFullTweakyPartAsParams(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  trace_index = get_empty_trace_index()

  tweak_map = get_empty_tweak_map()

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

    # HACKING THE OPTIMIZER TO ONLY WORK WITH THE TWEAKY PART
    # optimizer = torch.optim.Adam(model.getTheFullTweakyPartAsParams(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

    # allow for adapting the params from current HP
    # (TODO: in principle a larger class of things can be adaptively changed after resumig the training process)
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

  # an experimantal stage which will
  # - go through all the problems for which we have traces and tweaks in tweak_map (should be the same set)
  #  by the way, it does not matter which mission the problem belongs to
  # - we show the learned tweak
  # - we measure the generalist and tweaked loss
  # - we eval the generalist and tweaked real performance

  '''
  script_model_file_path = "temp-script-model.pt"
  IC.export_model(model.state_dict(),script_model_file_path)
  eval_model_file_path = "temp-eval-model.tar"
  torch.save(model.state_dict(), eval_model_file_path)

  TOTAL_EVALS = 10

  def get_tasks():
    ilim = HP.INSTRUCTION_LIMIT
    opts1 = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"
    opts2_base = f" -npcc {script_model_file_path} -nnf {HP.NUM_FEATURES}"

    for mission in MISSIONS:
      for prob,trace_list in trace_index[mission].items():
        used_tweak = tweak_map[prob]
        tweak_str = ",".join([str(t) for t in used_tweak])
        for i in range(TOTAL_EVALS):
          seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp
          yield (JK_PERFORM,("",False,mission,prob,False,opts1,opts2_base + f" --random_seed {seed}"))
          yield (JK_PERFORM,("",False,mission,prob,True,opts1,opts2_base + f" -npccw {tweak_str} --random_seed {seed}"))

        yield (JK_EVAL,(prob,1.0,False,None,trace_list,eval_model_file_path))
        yield (JK_EVAL,(prob,1.0,False,tweak_map[prob],trace_list,eval_model_file_path))

  prob_successes = { tweaked: defaultdict(int) for tweaked in [False,True]}
  prob_activations = { tweaked : defaultdict(int) for tweaked in [False,True]}
  prob_losses = { tweaked : dict() for tweaked in [False,True]}

  def process_results(job_kind,input,result):
    if job_kind == JK_PERFORM:
      (res_filename,gatherwish,mission,prob,used_tweak,opts1,opts2) = input
      (status,instructions,activations) = result
      if status == "uns":
        prob_successes[used_tweak][prob] += 1
        prob_activations[used_tweak][prob] += activations

    if job_kind == JK_EVAL:
      (prob,fact,descend_tw,tweak_start,trace_list,eval_model_file_path) = input
      (loss,out_tweak,numiter,telapsed,finished) = result
      prob_losses[tweak_start is not None][prob] = loss

    return 1

  do_in_parallel(get_tasks(),parallelism,process_results)

  for mission in MISSIONS:
    for prob in trace_index[mission]:
      print(prob,"with tweak",tweak_map[prob])
      print(f"  loss     {prob_losses[False][prob]} -> {prob_losses[True][prob]}")
      print(f"  succrate {prob_successes[False][prob]/TOTAL_EVALS} -> {prob_successes[True][prob]/TOTAL_EVALS}")
      if prob_successes[False][prob] > 0:
        activ_av_gen = prob_activations[False][prob]/prob_successes[False][prob]
      else:
        activ_av_gen = None
      if prob_successes[True][prob] > 0:
        activ_av_tw = prob_activations[True][prob]/prob_successes[True][prob]
      else:
        activ_av_tw = None
      print(f"  activ_av {activ_av_gen} -> {activ_av_tw}")
  '''

  while True:
    if loop_count == 0:
      break
    loop += 1
    loop_count -= 1

    loop_start_time = time.time()

    # for this iteration, we write stuff here:
    cur_dir = claim_loop_dir(loop)

    if HP.CUMULATIVE == 0: # forget what we solved until now
      trace_index = get_empty_trace_index()
    else:
      assert False, "Didn't think of CUMULATIVE yet" # see esp. process_results_from_perform_and_gather/GATHER

    # STAGE 1: PERFORM and GATHER
    stage_start_time = time.time()

    # There is going to be files to store the results, ...
    result_metas = [] # ... will store the file names and some additional info (in order of generation)
    result_dicts = defaultdict(lambda : IC.default_defaultdict_of_list()) # ... will collect the dicts to go into the respective files

    script_model_file_path = os.path.join(cur_dir,"script-model.pt")
    IC.export_model(model.state_dict(),script_model_file_path)

    def get_perform_tasks():
      for generalist in [True, False]:
        # the generalist' round is only for PERF, no GATHERing, (so that we know how the generalists fares after accomodating from the tweaking)
        # but if we are fully into GENERALIST_TRAINING_WEIGHT, we still gather (essentially for the generalist) via the False branch
        if generalist and HP.GENERALIST_TRAINING_WEIGHT == 1.0:
          continue

        for mission in MISSIONS:
          ilim = HP.INSTRUCTION_LIMIT

          if generalist:
            res_filename = f"{mission}_generalist.pt"
          else:
            res_filename = f"{mission}_tweaked.pt"
          result_metas.append((res_filename,mission,ilim))

          for i in range(HP.NUM_PERFORMS[generalist]):
            seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp

            # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
            opts1 = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"
            # will stay the same
            opts2_base = f" -npcc {script_model_file_path} -nnf {HP.NUM_FEATURES}"

            for prob in prob_lists[mission]:
              if generalist or HP.NUM_TWEAKS == 0:
                yield (JK_PERFORM,(res_filename,not generalist,mission,prob,[],opts1,opts2_base + f" --random_seed {seed}"))
              else:
                if prob in tweak_map:
                  used_tweak = tweak_map[prob]
                  # print("Reusing tweak",used_tweak,"from last time for",prob)
                elif len(tweak_map) > 0: # steal somebody else's
                  steal_from = random.choice(list(tweak_map.keys()))
                  used_tweak = tweak_map[steal_from]
                  # print(f"Stealing tweak {used_tweak} from {steal_from} for {prob}")
                  '''
                  elif have_tweak_std:
                    used_tweak = list(numpy.random.normal(HP.NUM_TWEAKS*[0.0],tweak_std))
                    # print("Will try tweak",used_tweak,"for",prob)
                  '''
                else:
                  used_tweak = HP.NUM_TWEAKS*[0.0,]
                  # print("Defaulting tweak",used_tweak,"for",prob)

                tweak_str = ",".join([str(t) for t in used_tweak])
                # print(tweak_str)
                opts2 = opts2_base + f" -npccw {tweak_str} --random_seed {seed}"

                yield (JK_PERFORM,(res_filename,not generalist,mission,prob,used_tweak,opts1,opts2))

    per_prob_trace_cnt = defaultdict(int)
    currently_solving = set()

    def process_results_from_perform_and_gather(job_kind,input,result):
      global per_prob_trace_cnt
      workers_freed = 0
      if job_kind == JK_PERFORM:
        (res_filename,gatherwish,mission,prob,used_tweak,opts1,opts2) = input
        result_dicts[res_filename][prob].append(result)

        (status,instructions,activations) = result
        if status == "uns" and gatherwish and (prob not in tweak_map or tweak_map[prob] == used_tweak):
          # Either this prob got solved for the first time, or
          # it has a well-establised tweak (which got used for all tries)
          # or it got by chance solved for second/third time (in this iter) with a tweak we know of already
          # (the main goal is to have one tweak for all traces we will be using)
          tweak_map[prob] = used_tweak

          counter = per_prob_trace_cnt[prob]
          per_prob_trace_cnt[prob] += 1

          ilim = 10*HP.INSTRUCTION_LIMIT
          task = (JK_GATHER,(mission,prob,counter,f"-t {ilim2tlim(ilim)} -i {ilim} -spt on"+opts2))
          # print("PUT:",task)
          q_in.put(task)
        else:
          workers_freed = 1
      elif job_kind == JK_GATHER:
        (mission,prob,counter,opts) = input
        currently_solving.add(prob)
        trace_file_path = result
        # the trace pkl has been saved to a file, let's just remember that we have it:
        if trace_file_path is not None:
          trace_index[mission][prob].append(trace_file_path) # TODO: this is perhaps not very wise with cumulative, as it would keep growing (while many of the traces would be getting overwritten)
        workers_freed = 1
      else:
        assert False, f"Surprised by job_kind {job_kind}"

      return workers_freed

    do_in_parallel(get_perform_tasks(),parallelism,process_results_from_perform_and_gather)

    # let's report what happened so far (and save the results into files, for later analysis):
    for (res_filename,mission,ilim) in result_metas:
      results = result_dicts[res_filename]
      torch.save((f"ilim: {ilim}",results), os.path.join(cur_dir,res_filename))

      prob_solved = 0
      prob_fractional = 0.0
      attempts = None
      for prob,runs in results.items():
        succs = sum(1 for (status,instructions,activations) in runs if status == "uns")
        if attempts is None:
          attempts = len(runs)
        else:
          assert attempts == len(runs)

        if succs > 0:
          prob_solved += 1
        prob_fractional += succs/attempts

      print(res_filename)
      print("    {:10.4f}% = {:10.1f} / {} ({} attempts) {} total".format(prob_fractional/len(results),prob_fractional,len(results),attempts,prob_solved))

    print()
    print("  Stage 1 took",time.time()-stage_start_time)
    print()
    sys.stdout.flush()

    for m in MISSIONS:
      trace_cnt = 0
      for prob,trace_list in trace_index[m].items():
        trace_cnt += len(trace_list)

      print("trace_index for",m,"has\n  ",len(trace_index[m]),"probs with a total of",trace_cnt,"traces")

    if HP.CUMULATIVE == 0:
      has_trace_for = set().union(*[set(trace_index[m]) for m in MISSIONS])
      # forget about tweaks for problems we did not solve this time!
      lost = 0
      discarded = 0
      for prob in list(tweak_map.keys()):
        if not prob in currently_solving:
          del tweak_map[prob]
          lost += 1
        elif not prob in has_trace_for:
          del tweak_map[prob]
          discarded += 1

      print(f"Lost {lost} problems since last iter (and now forgotten their tweaks, to start sampling from scratch next iter)")
      print(f"  also discarded {discarded} problems, because they are solved trivially and didn't generate traces.")

    save_trace_index(cur_dir,trace_index)

    print()
    sys.stdout.flush()

    # STAGE 2: alternate EVAL, TRAIN, EVAL until no longer improving
    for trainOnlyTweaks in [True,False]:
      internal_optimizer = optimizer
      effective_generalist_training_weight = HP.GENERALIST_TRAINING_WEIGHT
      if trainOnlyTweaks:
        effective_generalist_training_weight = 0.0
        print("First learn only the tweaky part!")
        internal_optimizer = torch.optim.Adam(model.getTheFullTweakyPartAsParams(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)
      else:
        print("The second round is a changing also the generalist.")

      print()
      sys.stdout.flush()
      stage_start_time = time.time()

      TIW = HP.TEST_IMPROVE_WINDOW
      assert TIW > 0
      eval_models = [None]*TIW
      eval_losses = [None]*TIW
      tweak_maps =  [None]*TIW
      stage2iter = 0

      while True:
        if TIW > 1:
          # EVAL on test problems
          eval_model_file_path = os.path.join(HP.SCRATCH,"eval-model-state_{}_{}.tar".format(os.getpid(),stage2iter))
          torch.save(model.state_dict(), eval_model_file_path)
          if eval_models[stage2iter % TIW] is not None:
            os.remove(eval_models[stage2iter % TIW])
          eval_models[stage2iter % TIW] = eval_model_file_path

          def get_eval_tasks(mission,generalists):
            fact = 1/len(trace_index[mission])
            if generalists and effective_generalist_training_weight > 0.0:
              fact *= effective_generalist_training_weight
              return ((JK_EVAL,(prob,fact,False,None,trace_list,eval_model_file_path)) for prob, trace_list in trace_index[mission].items())
            if not generalists and effective_generalist_training_weight < 1.0:
              fact *= (1.0-effective_generalist_training_weight)
              return ((JK_EVAL,(prob,fact,True,tweak_map[prob],trace_list,eval_model_file_path)) for prob, trace_list in trace_index[mission].items())
            return []

          def process_results_from_eval(job_kind,input,result):
            global weighted_eval_loss
            global num_eval_tasks
            global num_finisheds
            global total_time
            global total_iter

            assert job_kind == JK_EVAL
            (prob,fact,descend_tw,tweak_start,trace_list,model_file_path) = input
            (loss,out_tweak,numiter,telapsed,finished) = result

            weighted_eval_loss += loss # multiplied by fact already in the child

            if tweak_start is not None:
              assert descend_tw
              # print("Validating on",prob,"updated tweak from",tweak_start,"to",out_tweak)
              tweak_map[prob] = out_tweak
              num_eval_tasks += 1
              total_iter += numiter
              total_time += telapsed
              if finished:
                num_finisheds += 1

            return 1

          weighted_eval_loss = 0.0
          num_eval_tasks = 0
          num_finisheds = 0
          total_time = 0.0
          total_iter = 0
          do_in_parallel(chain(get_eval_tasks("valid",True),get_eval_tasks("valid",False)),parallelism,process_results_from_eval)
          print("Eval loss on valid",weighted_eval_loss,"num tasks",num_eval_tasks,"finished on",num_finisheds)
          if num_eval_tasks > 0:
            print("   average iter",total_iter/num_eval_tasks,"average time",total_time/num_eval_tasks)
          sys.stdout.flush()

          eval_losses[stage2iter % TIW] = weighted_eval_loss

          if effective_generalist_training_weight < 1.0:
            # we do eval also on train to have the tweaks descend analogously
            weighted_eval_loss = 0.0
            num_eval_tasks = 0
            num_finisheds = 0
            total_time = 0.0
            total_iter = 0
            do_in_parallel(get_eval_tasks("train",False),parallelism,process_results_from_eval)
            weighted_eval_loss /= (1-effective_generalist_training_weight) # because we only did the tweaky (non-generalist) part
            print("  Repositioned tweaks on train; achieved (tweaky) loss:",weighted_eval_loss,"num tasks",num_eval_tasks,"finished on",num_finisheds)
            print("   average iter",total_iter/num_eval_tasks,"average time",total_time/num_eval_tasks)
            sys.stdout.flush()

          tweak_maps[stage2iter % TIW] = tweak_map.copy()

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
                tweak_map = tweak_maps[1]
              else:
                model.load_state_dict(torch.load(eval_models[oldest_idx]))
                tweak_map = tweak_maps[oldest_idx]
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
              tweak_map = tweak_maps[best_idx]
              for eval_model_file_path in eval_models:
                os.remove(eval_model_file_path)
              break

        # TRAIN on train problems
        train_model_version = 0
        def get_train_tasks():
          fact = 1/len(trace_index["train"])

          # TODO: also here we could consider exerting extra force on harder problems (according to how recently they got solved) under CUMMULATIVE
          proto_tasks = []

          coef = effective_generalist_training_weight
          if coef > 0.0:
            proto_tasks += [[prob,fact*coef,None,trace_list] for prob, trace_list in trace_index["train"].items()]
          coef = 1.0-coef
          if coef > 0.0:
            proto_tasks += [[prob,fact*coef,tweak_map[prob],trace_list] for prob, trace_list in trace_index["train"].items()]

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
          (prob,fact,used_tweak,trace_file_paths,train_model_file_path) = input
          loss = result

          weighted_train_loss += loss # multiplied by fact already in the child
          # print(input,result)

          if used_tweak is not None:
            model.setTweakVals(used_tweak)

          # copy from result parameters to our model's gradients
          grad_loader_temp.load_state_dict(torch.load(train_model_file_path))
          # copy_grads_back_from_param
          for param, param_copy in zip(model.parameters(),grad_loader_temp.parameters()):
            param.grad = param_copy

          internal_optimizer.step()

          # here we have a new updated tweak inside model
          if used_tweak is not None:
            new_tweak = model.getTweakVals()
            # print("Training on",prob,"Updated tweak from",used_tweak,"to",new_tweak)
            tweak_map[prob] = new_tweak

          os.remove(train_model_file_path)
          return 1

        do_in_parallel(get_train_tasks(),min(parallelism,HP.TRAINING_PARALLELISM),process_results_from_train)

        print("Weighted train loss",weighted_train_loss)
        print()
        sys.stdout.flush()

        if TIW == 1:
          os.remove(eval_model_file_path)
          break

      # stage 2
      print()
      print(f"  Stage 2 - trainOnlyTweaks {trainOnlyTweaks} took {time.time()-stage_start_time} seconds and {stage2iter-0.5} eval/train iterations")
      print()
      sys.stdout.flush()

    tweak_map_file_path = os.path.join(cur_dir,"after-train-"+TWEAK_MAP)
    torch.save(tweak_map, tweak_map_file_path)

    if len(tweak_map) > 0:
      tweak_std = numpy.std(list(tweak_map.values()),axis=0)
      print("  tweak_std now",tweak_std)

    print_model_part()
    save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)
    print()
    sys.stdout.flush()

    # STAGE 3 -- try solving new problems by using other problem's tweaks
    if HP.GENERALIST_TRAINING_WEIGHT < 1.0:
      stage_start_time = time.time()

      script_model_file_path = os.path.join(cur_dir,"script-model-after.pt")
      IC.export_model(model.state_dict(),script_model_file_path)

      TOTAL_EVALS = 3

      def get_eval_with_tweak_tasks(tweak):
        ilim = HP.INSTRUCTION_LIMIT
        tweak_str = ",".join([str(t) for t in tweak])
        opts1 = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"
        for i in range(TOTAL_EVALS):
          seed = random.randint(1,0x7fffff)
          opts2 = f" -npcc {script_model_file_path} -nnf {HP.NUM_FEATURES} -npccw {tweak_str} --random_seed {seed}"
          for mission in MISSIONS:
            for prob in prob_lists[mission]:
              yield (JK_PERFORM,("",False,mission,prob,True,opts1,opts2))

      def process_results(job_kind,input,result):
        global prob_successes
        if job_kind == JK_PERFORM:
          (res_filename,gatherwish,mission,prob,used_tweak,opts1,opts2) = input
          (status,instructions,activations) = result
          if status == "uns":
            prob_successes[prob] += 1
        return 1

      new_tweak_map = get_empty_tweak_map()
      best_prob_success = defaultdict(int)
      print("Old tweak_map",len(tweak_map))

      num_prob_tweaks_tested = 0

      prob_tweaks_to_try = list(tweak_map.items())
      random.shuffle(prob_tweaks_to_try)
      prob_tweaks_to_try = [(None,HP.NUM_TWEAKS*[0.0])]+prob_tweaks_to_try

      for the_prob,tweak in prob_tweaks_to_try:
        prob_successes = defaultdict(int)
        do_in_parallel(get_eval_with_tweak_tasks(tweak),parallelism,process_results)

        print(f"Tested tweak {tweak} from prob {the_prob}")
        success = False
        if the_prob not in prob_successes:
          print("Not solved with its own tweak at all:",the_prob)
        for prob,cnt in prob_successes.items():
          if prob not in new_tweak_map:
            print("  newly solved",prob,"with",cnt,"hits")
            best_prob_success[prob] = cnt
            new_tweak_map[prob] = tweak
            success = True
          elif cnt > best_prob_success[prob]:
            print("  improved on",prob,"with",cnt,"currently",best_prob_success[prob])
            best_prob_success[prob] = cnt
            new_tweak_map[prob] = tweak
            success = True
          elif prob == the_prob:
            print("  failed to improve with own tweak for",prob)

        print("Know how to solve",len(new_tweak_map),"at the moment")
        sys.stdout.flush()
        print()

        num_prob_tweaks_tested += 1

        if success:
          since_last_success = 0
        else:
          since_last_success += 1
        if since_last_success > 10:
          print("Taking too long to solve new problems -- abort trying problem's native tweaks")
          break

      # do it one more time, but now with random tweaks
      since_last_success = 0
      num_rand_tweaks_tested = 0
      while True:
        tweak = list(numpy.random.normal(HP.NUM_TWEAKS*[0.0],tweak_std))
        prob_successes = defaultdict(int)
        do_in_parallel(get_eval_with_tweak_tasks(tweak),parallelism,process_results)

        print(f"Tested tweak {tweak}")
        success = False
        for prob,cnt in prob_successes.items():
          if prob not in new_tweak_map:
            print("  newly solved",prob,"with",cnt,"hits")
            best_prob_success[prob] = cnt
            new_tweak_map[prob] = tweak
            success = True
          elif cnt > best_prob_success[prob]:
            print("  improved on",prob,"with",cnt,"currently",best_prob_success[prob])
            best_prob_success[prob] = cnt
            new_tweak_map[prob] = tweak
            success = True

        print("Know how to solve",len(new_tweak_map),"at the moment")
        sys.stdout.flush()
        print()

        num_rand_tweaks_tested += 1

        if success:
          since_last_success = 0
        else:
          since_last_success += 1
        if since_last_success > 10:
          print("Taking too long to solve new problems -- abort trying random tweaks")
          break

      tweak_map = new_tweak_map

      print("  Stage 3 took",time.time()-stage_start_time,"seconds and tested",num_prob_tweaks_tested,"problem's tweaks and",num_rand_tweaks_tested,"random ones")
      print()
      sys.stdout.flush()

    save_tweak_map(cur_dir,tweak_map)

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()
