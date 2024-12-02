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

import torch_geometric.nn

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

def print_model_part():
  print("Key {}".format(repr(model.valuator[-1].weight.data)))

TRAIN_PROBLEMS_FILE = "train.txt"
TEST_PROBLEMS_FILE = "test.txt"

def steal_problems_from(folder_with_train_test_problems):
  with open(os.path.join(folder_with_train_test_problems,TRAIN_PROBLEMS_FILE),"r") as f:
    train_problems = [line.rstrip() for line in f.readlines()]
  with open(os.path.join(folder_with_train_test_problems,TEST_PROBLEMS_FILE),"r") as f:
    test_problems = [line.rstrip() for line in f.readlines()]
  return train_problems,test_problems

def train_test_problem_split():
  with open(HP.PROBLEM_LIST,"r") as f:
    full_list = [line.rstrip() for line in f.readlines()]
  assert len(full_list) >= HP.NUM_TRAIN_PROBLEMS + HP.NUM_TEST_PROBLEMS
  our_problems = random.sample(full_list,HP.NUM_TRAIN_PROBLEMS + HP.NUM_TEST_PROBLEMS)
  random.shuffle(our_problems)
  return our_problems[:HP.NUM_TRAIN_PROBLEMS],our_problems[HP.NUM_TRAIN_PROBLEMS:]

def save_train_test_problems(exper_dir,train_problems,test_problems):
  for filename,problems in [(TRAIN_PROBLEMS_FILE,train_problems),(TEST_PROBLEMS_FILE,test_problems)]:
    with open(os.path.join(exper_dir,filename),"w") as f:
      for p in problems:
        f.write(p)
        f.write("\n")

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

def load_loop_model_and_optimizer(adir):
  loop_model_and_optimizer_state_file_path = os.path.join(adir,LOOP_MODEL_AND_OPTIMIZER)
  return torch.load(loop_model_and_optimizer_state_file_path)

def get_empty_trace_index():
  return defaultdict(list) # problem -> [trace_file_name]  # used to more than one with different temperatures, can also have more than one with different seeds

TRACE_INDEX = "trace-index.pt"

def save_trace_index(cur_dir,trace_index):
  trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
  torch.save(trace_index, trace_index_file_path)

def load_trace_index(adir):
  trace_index_file_path = os.path.join(adir,TRACE_INDEX)
  return torch.load(trace_index_file_path)

def report_on_trace_index(trace_index):
  trace_cnt = 0
  for prob,trace_list in trace_index.items():
    trace_cnt += len(trace_list)
  print("trace_index has\n  ",len(trace_index),"probs with a total of",trace_cnt,"traces")

def ilim2tlim(ilim):
  secs = max(5,ilim // 1000) # it's 2 times more than the instrlimit on a 2GHz machine
  return secs

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,gatherwish,mission,prob,opts1,opts2)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,counter,opts)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL = 2    # construct our network to get the loss of this trace (no training to do)
  # input:     (prob,fact,trace_file_paths,model_file_path)
  # output:    the computed loss
JK_TRAIN = 3   # construct our network to get the loss of this trace and do one training step
  # input:     (prob,fact,trace_file_paths,train_model_file_path)
  # output:    the computed loss

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()

    if job_kind == JK_PERFORM:
      (res_filename,gatherwish,mission,prob,i,lrs_trace_file,opts1,opts2) = input
      result = IC.vampire_perfrom(prob,opts1+opts2)
      q_out.put((job_kind,input,result))
    elif job_kind == JK_GATHER:
      (mission,prob,lrs_trace_file,counter,opts) = input
      result = IC.vampire_gather(prob,opts)
      trace_file_path = None
      if result is not None and result[0]: # non-degenerate
        compressed_trace = IC.compress_trace(*result[1])
        if compressed_trace[-1] > 0: # num_good_selections can be zero in degenerate cases
          trace_file_path = os.path.join(traces_dir,"{}_{}.pt".format(prob.replace("/","_"),counter))
          torch.save(compressed_trace,trace_file_path)
      q_out.put((job_kind,input,trace_file_path))

    elif job_kind == JK_EVAL:
      (prob,fact,trace_file_paths,model_file_path) = input

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      local_fact = 1/len(trace_file_paths)
      proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      # print("EVAL on",prob,fact,trace_file_paths)

      loss = torch.zeros(1)
      for proof_tuple in proof_tuples:
        learn_model = IC.LearningModel(False,local_model,*proof_tuple)
        learn_model.eval()
        # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
        # print(tweaks_to_try)
        loss += local_fact*learn_model.forward()

      # print("EVAL on",prob,fact,trace_file_paths,loss.item())

      q_out.put((job_kind,input,fact*loss.item()))

    elif job_kind == JK_TRAIN:
      (prob,fact,trace_file_paths,train_model_file_path) = input

      local_fact = 1/len(trace_file_paths)
      proof_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(train_model_file_path))

      verbose = False # (prob in {'Problems/COM/COM021+4.p'})

      # print("TRAIN on",prob,fact,trace_file_paths)

      loss = torch.zeros(1)
      for proof_tuple in proof_tuples:
        learn_model = IC.LearningModel(verbose,local_model,*proof_tuple)
        learn_model.train()

        loss += local_fact*learn_model.forward()

      # print("TRAIN on",prob,fact,trace_file_paths,loss.item())

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
  # Elooper continues the tradition, builds on the experience from dlooper and attempts to simplify and streamline things
  # (first version will ditch the whole gsd aspects as they complicate things quite a lot;
  # the hope is that there will be more to gain from gsd later, with richer features - so that overfitting to a single problem will be easier)
  #
  # One thing that's now going to be different is that each exper (if starting from scratch) will commit to a fresh train/test split.
  # The split will be carved out from a shuffled problem totality specified in HP.PROBLEM_LIST and recorded in the exper.
  # Test problems (holdout) will be only used for performance checking of the final product (no early stopping based on this; in fact, maybe should not even eval on this in each loop)
  # For early stopping tricks there will be a validation (devel) subset of the train set traces. Changing this subset every loop is fine.
  # Carving it out of train set suggest this is an "inner split", for which the algorithm is responsible, whereas the train/test on is somehow external (used for hygienic reasons)
  #
  # The other new thing is going to be use of torch tracing to create static graphs for training on prover experiences, these will be saved into files and communicated with workers via filenames.
  # (actually, this ended not actually functioning as anticiapted and never got implemented)
  #
  # To be called as in:
  #
  # ./elooper.py 15 120 /home/sudamar2/lawa/exper1
  #
  # A previous experiment can be continued in the following ways:
  # 1) 4th argument specifies the previous experiment folder
  #  if this is specified, the currently starting experiment will steal the train test problem selection from previous experiment
  # 2) 5th argument specifies a loop <L> in the previous experiment
  # 3a) if the 6th argument contains a "m", a model/optimizer will be stolen from loop<L> there
  # 3b) if the 6th argument contains a "t", a trace index will be stolen from loop<L+1> there
  # if the 6th argument is missing, it defaults to "m" - so we load the model, but evaluate freshly on our own
  #
  # example: ./elooper.py 15 120 deleteme /home/sudamar2/mtpa/exper131 6 t
  # in this example exper131 produced its best model at the end of loop 6
  # this model got then evalauted and loop 7 has the largest set of traces stored under loop 7
  # with the example call, we'll use those traces, but learn a new model from scratch (this is a "fight the loss of plasticity" experiment)

  if False:
    m = torch.jit.load("genAgeOut.pt")
    print(m.inits)
    print(m.infers)

    exit(0)

  if True:
    script = IC.get_full_genAgeNN()
    script.save("genAgeNN.pt")
    exit(0)

  if False:
    script = torch.jit.script(IC.GnnStore())
    script.save("forVampire.pt")
    exit(0)

  m = torch.jit.load("fromVampireBig.pt")

  t = time.time()
  IC.get_gnn_compute(m,"gnnCompute.pt")
  print("Took",time.time()-t)

  exit(0)

  script = torch.jit.script(IC.GnnStore())
  script.save("forVampire.pt")
  exit(0)

  gnn = IC.vampire_gather("Problems/PUZ/PUZ001+1.p","-t 10 -spt on -sig on")

  print(gnn)

  t = time.time()

  out = gnn.forward()

  print(out)
  print("Took",time.time()-t)



  loop_count = int(sys.argv[1])
  parallelism = int(sys.argv[2])
  assert parallelism > 0
  exper_dir =  sys.argv[3]

  folder_with_prev_exper = sys.argv[4] if len(sys.argv) > 4 else None

  # loop_count should be the number of epochs, i.e.
  # - the number of times the master model gets trained (for roughtly as many rounds as there are training examples)
  # - evaluted and a checkpoint saved

  # Start a new experiment folder to contain
  # 1) hyperparms and train/test problems we ran on
  # 2) loop folders each with train / test results and the model/optimizer from that loop
  # 3) a traces folder with eval / train trace files (which are persistent across the loops but get overwritten to save space in each successive gather stage)
  os.mkdir(exper_dir)
  traces_dir = os.path.join(exper_dir,"traces")
  # NOTE: if we want to share traces with a previous run we use CUMMULATIVE and load a trace index which will point to old exper's traces folder (until overwritten)
  os.mkdir(traces_dir)

  # Documentation: save hyperparams and campaign
  # (CAREFUL: this way can only call looper from lawa folder)
  shutil.copy("hyperparams.py",exper_dir)

  if folder_with_prev_exper:
    train_problems,test_problems = steal_problems_from(folder_with_prev_exper)
  else:
    train_problems,test_problems = train_test_problem_split()
  save_train_test_problems(exper_dir,train_problems,test_problems)

  # Initializing a model and an optimizer (might still get better one below from load_dir if given)
  model = IC.get_initial_model()
  optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

  trace_index = get_empty_trace_index()

  loop = 0
  if len(sys.argv) > 5: # we already know the folder, but which loop to copy from there?
    loop = int(sys.argv[5])
    load_dir = os.path.join(folder_with_prev_exper,f"loop{loop}")

    load_model = True
    load_traces = False

    if len(sys.argv) > 6:
      load_model = "m" in sys.argv[6]
      load_traces = "t" in sys.argv[6]

    if load_model:
      aloop,amodel_state_dict,anoptimizer_state_dict = load_loop_model_and_optimizer(load_dir)
      assert aloop == loop
      model.load_state_dict(amodel_state_dict)
      optimizer.load_state_dict(anoptimizer_state_dict)

    if load_traces:
      trace_index = load_trace_index(os.path.join(folder_with_prev_exper,f"loop{loop+1}"))
      print("Starting from loop",loop,"and a half")
      report_on_trace_index(trace_index)

  else:
    cur_dir = claim_loop_dir(loop)
    save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)

  print_model_part()

  assert loop_count > 0

  # ===========================================================================
  # ===========================================================================
  # the parallel business set up here:

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

  # ===========================================================================
  # ===========================================================================

  iter = 0
  while True:
    if loop_count == 0:
      break
    loop += 1
    loop_count -= 1
    iter += 1

    loop_start_time = time.time()

    # for this iteration, we write stuff here:
    cur_dir = claim_loop_dir(loop)

    # ===========================================================================
    if len(trace_index) == 0:
      # STAGE 1: PERFORM and GATHER
      stage_start_time = time.time()

      # There is going to be files to store the results, ...
      result_metas = [] # ... will store the file names and some additional info (in order of generation)
      result_dicts = defaultdict(lambda : IC.default_defaultdict_of_list()) # ... will collect the dicts to go into the respective files

      script_model_file_path = os.path.join(cur_dir,"script-model.pt")
      IC.export_model(model.state_dict(),script_model_file_path)

      def get_perform_tasks():
        ilim = HP.INSTRUCTION_LIMIT
        for mission,gatherwish,prob_lists in [("train",True,train_problems),("test",False,test_problems)]:
          if not HP.EVAL_ON_TEST and mission == "test":
            continue
          res_filename = f"{mission}_res.pt"

          result_metas.append((res_filename,mission,ilim))

          for i in range(HP.NUM_PERFORMS):
            seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp

            # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
            opts1_base = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"
            # will stay the same
            opts2_base = f" -sa {HP.SATURATION_ALGORITHM} -ncf {HP.NUM_CLAUSE_FEATURES} -npf {HP.NUM_PROBLEM_FEATURES}"
            if not HP.IMITATE or loop > 1:
              opts2_base += f" -npcc on -ncem {script_model_file_path}"

            opts2_base += HP.PERFORMS_SPECIAL[i]

            for prob in prob_lists:
              opts1 = opts1_base
              if HP.SATURATION_ALGORITHM.startswith("lrs"):
                lrs_trace_file = os.path.join(HP.SCRATCH,"{}_{}_{}_{}.lrs".format(prob.replace("/","_"),i,seed,os.getpid()))
                opts1 += f" -lstf {lrs_trace_file}"
              else:
                lrs_trace_file = ""

              yield (JK_PERFORM,(res_filename,gatherwish,mission,prob,i,lrs_trace_file,opts1,opts2_base + f" --random_seed {seed}"))

      per_prob_trace_cnt = defaultdict(int)
      currently_solving = set()

      def process_results_from_perform_and_gather(job_kind,input,result):
        global per_prob_trace_cnt
        workers_freed = 0
        if job_kind == JK_PERFORM:
          (res_filename,gatherwish,mission,prob,i,lrs_trace_file,opts1,opts2) = input
          result_dicts[res_filename][prob].append((i,result))

          (status,instructions,activations) = result
          if status == "uns" and gatherwish:
            counter = per_prob_trace_cnt[prob]
            per_prob_trace_cnt[prob] += 1

            ilim = 10*HP.INSTRUCTION_LIMIT
            lrs_trace_str = f" -lltf {lrs_trace_file}" if lrs_trace_file else ""
            task = (JK_GATHER,(mission,prob,lrs_trace_file,counter,f"-t {ilim2tlim(ilim)} -i {ilim} -spt on -sig on {lrs_trace_str}"+opts2))
            # print("PUT:",task)
            q_in.put(task)
          else:
            workers_freed = 1
            if lrs_trace_file and os.path.isfile(lrs_trace_file):
              os.remove(lrs_trace_file)
        elif job_kind == JK_GATHER:
          (mission,prob,lrs_trace_file,counter,opts) = input
          currently_solving.add(prob)
          trace_file_path = result
          # the trace pkl has been saved to a file, let's just remember that we have it:
          if trace_file_path is not None:
            trace_index[prob].append(trace_file_path) # TODO: this is perhaps not very wise with CUMULATIVE, as it would keep growing (while many of the traces would be getting overwritten)
          workers_freed = 1
          if lrs_trace_file and os.path.isfile(lrs_trace_file):
            os.remove(lrs_trace_file)
        else:
          assert False, f"Surprised by job_kind {job_kind}"

        return workers_freed

      do_in_parallel(get_perform_tasks(),parallelism,process_results_from_perform_and_gather)

      # let's report what happened so far (and save the results into files, for later analysis):
      for (res_filename,mission,ilim) in result_metas:
        results = result_dicts[res_filename]
        torch.save((f"ilim: {ilim}",results), os.path.join(cur_dir,res_filename))

        by_performs = defaultdict(int)
        by_performs_set = defaultdict(set)

        prob_solved = 0
        prob_fractional = 0.0
        attempts = None
        for prob,runs in results.items():
          succs = 0
          for (i,(status,instructions,activations)) in runs:
            if status == "uns":
              succs += 1
              by_performs[i] += 1
              by_performs_set[i].add(prob)
          if attempts is None:
            attempts = len(runs)
          else:
            assert attempts == len(runs)

          if succs > 0:
            prob_solved += 1
          prob_fractional += succs/attempts

        print(res_filename)
        print("    {:10.4f}% = {:10.1f} / {} ({} attempts) {} total".format(prob_fractional/len(results),prob_fractional,len(results),attempts,prob_solved))

        covered = set()
        adds = []

        best_i = -1
        best_p = 0
        for i in range(HP.NUM_PERFORMS):
          adds.append(len(by_performs_set[i]-covered))
          covered = covered | by_performs_set[i]

          if by_performs[i] > best_p:
            best_p = by_performs[i]
            best_i = i

        for i in range(HP.NUM_PERFORMS):
          print("   {}  {} {:6.4f} {:>5}:{}".format(i,"*" if i == best_i else " ",by_performs[i]/len(results),adds[i],HP.PERFORMS_SPECIAL[i]))

      print()
      print("  Stage 1 took",time.time()-stage_start_time)
      print()
      sys.stdout.flush()

      report_on_trace_index(trace_index)
      save_trace_index(cur_dir,trace_index)

      print()
      sys.stdout.flush()

    # ===========================================================================

    # STAGE 2: alternate EVAL, TRAIN, EVAL until no longer improving
    print()
    sys.stdout.flush()
    stage_start_time = time.time()

    TIW = HP.TEST_IMPROVE_WINDOW
    assert TIW > 0
    eval_models = [None]*TIW
    eval_losses = [None]*TIW
    stage2iter = 0

    trace_problems = list(trace_index)
    if TIW > 1: # we will need to single out the validation traces!
      random.shuffle(trace_problems) # Note: this is a source of non-determinism!
      # an 80:20 split
      cut_idx = int(0.8*len(trace_problems))
      train_trace_problems = trace_problems[:cut_idx]
      valid_trace_problems = trace_problems[cut_idx:]
    else:
      train_trace_problems = trace_problems
      valid_trace_problems = None

    while True:
      if TIW > 1:
        # EVAL on validation problems
        eval_model_file_path = os.path.join(HP.SCRATCH,"eval-model-state_{}_{}.tar".format(os.getpid(),stage2iter))
        torch.save(model.state_dict(), eval_model_file_path)
        if eval_models[stage2iter % TIW] is not None:
          os.remove(eval_models[stage2iter % TIW])
        eval_models[stage2iter % TIW] = eval_model_file_path

        def get_eval_tasks():
          fact = 1/len(valid_trace_problems)
          for prob in valid_trace_problems:
              # print((JK_EVAL,(prob,fact,trace_list,eval_model_file_path)))
              yield (JK_EVAL,(prob,fact,trace_index[prob],eval_model_file_path))

        def process_results_from_eval(job_kind,input,result):
          global weighted_eval_loss
          assert job_kind == JK_EVAL
          weighted_eval_loss += result # (= the loss) multiplied by fact already in the child
          return 1

        weighted_eval_loss = 0.0
        do_in_parallel(get_eval_tasks(),parallelism,process_results_from_eval)
        print("Eval loss on valid",weighted_eval_loss)
        sys.stdout.flush()

        eval_losses[stage2iter % TIW] = weighted_eval_loss

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
          ITER_LIMIT = HP.MAX_TEST_IMPROVE_FIRST_ITER if (iter == 1) else HP.MAX_TEST_IMPROVE_ITER
          if stage2iter > ITER_LIMIT:
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
      train_model_version = 0
      def get_train_tasks():
        fact = 1/len(train_trace_problems)

        # TODO: also here we could consider exerting extra force on harder problems (according to how recently they got solved) under CUMMULATIVE
        proto_tasks = [[prob,fact,trace_index[prob]] for prob in train_trace_problems]

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
        (prob,fact,trace_file_paths,train_model_file_path) = input
        loss = result

        weighted_train_loss += result # (= the loss) multiplied by fact already in the child
        # print(input,result)

        # copy from result parameters to our model's gradients
        grad_loader_temp.load_state_dict(torch.load(train_model_file_path))
        # copy_grads_back_from_param
        for param, param_copy in zip(model.parameters(),grad_loader_temp.parameters()):
          param.grad = param_copy

        optimizer.step()

        os.remove(train_model_file_path)
        return 1

      do_in_parallel(get_train_tasks(),min(parallelism,HP.TRAINING_PARALLELISM),process_results_from_train)

      print("Weighted train loss",weighted_train_loss)
      print()
      sys.stdout.flush()

      if TIW == 1:
        os.remove(eval_model_file_path) # ???
        break

    # stage 2
    print()
    print(f"  Stage 2 - took {time.time()-stage_start_time} seconds and {stage2iter-0.5} eval/train iterations")
    print()
    sys.stdout.flush()

    if HP.LEARNING_RATE_DECAY < 1.0:
      print("Learning rate decrease")
      for g in optimizer.param_groups:
        print(" from",g['lr'],end=" ")
        g['lr'] *= HP.LEARNING_RATE_DECAY
        print("to",g['lr'])
      print()

    print_model_part()
    save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)
    print()
    sys.stdout.flush()

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()

    # so that we can distinguish standard run from a "loop and a half" start (which reads a ready trace index and then skips Stage 1)
    trace_index = get_empty_trace_index()
