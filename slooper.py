#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain

import multiprocessing
import numpy

import gc

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

# taken from snake's minilib
def str2dict(str):
  spl = str.split("_")

  assert len(spl) >= 3

  res = {}

  # hack in long versions of saturation algorithm names (convert from the 3letter ones used in the decode string)
  sa_longs = {"lrs" : "lrs", "dis" : "discount", "ott" : "otter", "ins" : "inst_gen", "fmb" : "fmb"}
  res["sa"] = sa_longs[spl[0][:3]]

  res["s"] = int(spl[0][3:])
  res["awr"] = spl[1]
  res["t"] = spl[-1]

  rest = "_".join(spl[2:-1])
  if rest:
    for optpair in rest.split(":"):
      opt,val = optpair.split("=")
      res[opt] = val

  return res

def dict2str(dict):
  middle = ":".join("{}={}".format(opt,val) for opt,val in dict.items() if opt not in ["sa","s","awr","t"])

  return "{}{}{}_{}_{}_{}".format(dict["sa"][:3],"+" if int(dict["s"])>=0 else "", dict["s"],dict["awr"],middle,dict["t"])

def sorted_dict2str(dict):
  middle = ":".join("{}={}".format(opt,val) for opt,val in sorted(dict.items()) if opt not in ["sa","s","awr","t"])
  return "{}{}{}_{}_{}_{}".format(dict["sa"][:3],"+" if int(dict["s"])>=0 else "",dict["s"],dict["awr"],middle,dict["t"])



def print_model_part():
  pass
  """
  t = model.clause_valuator[-1].weight
  print("Key {}".format(repr(t.data)))
  print(f"  of shape {t.shape}")
  """

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

class TraceIndex:
  def __init__(self):
    self.traces = {} # problem -> [trace_file_name]  # a list to support more than one sample per problem per loop (c.f. HP.NUM_PERFORMS)
    self.last_solved = {}
    self.prob_scores = {}

    if HP.CUMULATIVE:
      self.BASE = math.pow(HP.CUM_MAX_STRENGTH,1/(2*HP.CUM_STALE_AFTER))
      print("trace index establish a factor base of",self.BASE)

  def loop_finished(self):
    if not HP.CUMULATIVE:
      self.traces = {}

  def cur_problems(self):
    return self.traces.keys()

  def prob_traces(self,prob):
    return self.traces[prob]

  def prob_factor(self,prob):
    if not HP.CUMULATIVE:
      return 1.0
    return math.pow(self.BASE,self.prob_scores[prob])

  def add_prob_trace(self,loop,prob,trace_file_path):
    if (prob not in self.traces or
        # idea: if we solve the problem now, we throw its old traces away
        (HP.CUMULATIVE and prob in self.last_solved and self.last_solved[prob] < loop)):
      self.traces[prob] = [trace_file_path]
    else:
      self.traces[prob].append(trace_file_path)

    self.last_solved[prob] = loop

  def report_trivial_trace(self,loop,prob):
    if HP.CUMULATIVE and prob in self.traces:
      del self.traces[prob]

  def update_scores(self,loop):
    if not HP.CUMULATIVE:
      return
    num_stale = 0
    num_new = 0
    num_routine = 0
    num_losing = 0
    for prob,last in self.last_solved.items():
      if loop-last > HP.CUM_STALE_AFTER:
        if prob in self.traces:
          del self.traces[prob]
        if prob in self.prob_scores:
          del self.prob_scores[prob]
        num_stale += 1
      elif last == loop:
        if prob not in self.prob_scores:
          # newly solved, or solved after being long forgotten
          self.prob_scores[prob] = 0
          num_new += 1
        else:
          # we already know we are solving it, so let's experiment with a lower score
          self.prob_scores[prob] -= 1
          num_routine += 1
      else:
        # we seem to be losing this problem, let's try to catch up
        self.prob_scores[prob] += 2
        num_losing += 1
    print("trace index score update:")
    print("  new   :",num_new)
    print("  easing:",num_routine)
    print("  losing:",num_losing)
    print("  staled:",num_stale)
    print("  score hist")
    hist = defaultdict(int)
    for _,score in self.prob_scores.items():
      hist[score] += 1
    for score,val in sorted(hist.items()):
      print("    {:>6} {:>6}".format(score, val))
    print()

  def report(self):
    trace_cnt = 0
    for prob,trace_list in self.traces.items():
      trace_cnt += len(trace_list)
    print("trace_index has\n  ",len(self.traces),"probs with a total of",trace_cnt,"traces")

TRACE_INDEX = "trace-index.pt"

def save_trace_index(cur_dir,trace_index):
  trace_index_file_path = os.path.join(cur_dir,TRACE_INDEX)
  torch.save(trace_index, trace_index_file_path)

def load_trace_index(adir):
  trace_index_file_path = os.path.join(adir,TRACE_INDEX)
  return torch.load(trace_index_file_path)

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
      (mission,prob,lrs_trace_file,trace_file_path,opts) = input
      vamp_res = IC.vampire_perfrom(prob,opts)

      assert vamp_res.status == "uns", f"Ran {(prob,opts)} got {vamp_res}"
      assert os.path.isfile(trace_file_path)
      trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_file_path,train_log)

      q_out.put((job_kind,input,(trace_kept, gage_stats, gweight_stats)))

    elif job_kind == JK_EVAL:
      (prob,fact,trace_file_paths,model_file_path) = input

      # print("EVAL",prob,fact,trace_file_paths,model_file_path)
      # sys.stdout.flush()

      eval_begin = time.time()

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(model_file_path))

      local_fact = 1/len(trace_file_paths)
      trace_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      # print("EVAL on",prob,fact,trace_file_paths)

      try:
        loss = torch.zeros(1)
        for trace_tuple in trace_tuples:
          learn_model = IC.LearningModel(False,local_model,trace_tuple)
          learn_model.eval()
          # print("For",prob,temp,"with",tweak_start,tweak_std,"will try")
          # print(tweaks_to_try)
          loss += local_fact*learn_model.forward()
      except Exception as e:
        with open(f"exception{os.getpid()}.log", "w") as f:
          f.write(f"{e} occurred in EVAL\n")
          f.write(f"(prob: {prob},fact {fact},trace_file_paths {trace_file_paths},model_file_path {model_file_path})")
        raise

      took = time.time()-eval_begin
      if took > HP.WORTH_REPORTING:
        train_log.write(f"EVAL of {prob} took {took}\n")

      # print("EVAL on",prob,fact,trace_file_paths,loss.item())

      q_out.put((job_kind,input,fact*loss.item()))

      del loss
      del learn_model
      del local_model
      gc.collect()  # Force garbage collection

    elif job_kind == JK_TRAIN:
      (prob,fact,trace_file_paths,train_model_file_path) = input

      # print("TRAIN",prob,fact,trace_file_paths,train_model_file_path)
      # sys.stdout.flush()

      train_begin = time.time()

      local_fact = 1/len(trace_file_paths)
      trace_tuples = [torch.load(trace_file_path) for trace_file_path in trace_file_paths]

      local_model = IC.get_initial_model()
      local_model.load_state_dict(torch.load(train_model_file_path))

      verbose = False # (prob in {'Problems/COM/COM021+4.p'})

      # print("TRAIN on",prob,fact,trace_file_paths)

      try:
        loss = torch.zeros(1)
        for trace_tuple in trace_tuples:
          learn_model = IC.LearningModel(verbose,local_model,trace_tuple)
          learn_model.train()

          loss += local_fact*learn_model.forward()

        loss.backward()
      except Exception as e:
        with open(f"exception{os.getpid()}.log", "w") as f:
          f.write(f"{e} occurred in TRAIN\n")
          f.write(f"(prob: {prob},fact {fact},trace_file_paths {trace_file_paths},model_file_path {model_file_path})")
        raise

      # print("TRAIN on",prob,fact,trace_file_paths,loss.item())

      for param in local_model.parameters():
        grad = param.grad
        param.requires_grad = False # to allow the in-place operation just below
        if grad is not None:
          param.copy_(grad)
        else:
          param.zero_()

      # use the same file also for the journey back (which brings the gradients inside the actual params)
      torch.save(local_model.state_dict(), train_model_file_path)

      took = time.time()-train_begin
      if took > HP.WORTH_REPORTING:
        train_log.write(f"TRAIN of {prob} took {took}\n")

      q_out.put((job_kind,input,fact*loss.item()))

      del loss
      del learn_model
      del local_model
      gc.collect()  # Force garbage collection




















from multiprocessing import Pool

def collect_traces(task):
  prob,solns = task

  print("collect_traces for",prob,"with",len(solns),"avaliable solns")

  if HP.SNAKE_SORT_BY_INSTR:
    solns.sort(reverse = True) # we pop the small solutions from the end
  else:
    random.shuffle(solns)

  fauls = 0

  traces_collected = []
  while len(traces_collected) < HP.SNAKE_MAX_TRACES_PER_PROBLEM and fauls < HP.SNAKE_MAX_FAULS:
    (instr,stratstr) = solns.pop()

    # if expected to take too long, don't even bother reproving
    if instr > HP.SNAKE_MAX_INSTRUCTIONS:
      continue

    print("    popped soln of instr",instr,"and stratstr",stratstr)

    # add 10% extra (and don't be a Scrooge)
    ilim = max(int(1.1*instr),125)

    # try reprove under shuffling
    lrs_trace_file = None
    for i in range(HP.SNAKE_MAX_TRIES):
      if i == HP.SNAKE_MAX_TRIES-1:
        print("      will try unshuffled one")
        opt_random = ""
      else:
        seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp
        opt_random = f"{HP.SHUFFLING_OPTIONS} --random_seed {seed}"

      # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
      opts1 = f"-t {ilim2tlim(ilim)} -i {ilim}"
      opts2 = f" -p off --parsing_does_not_count on {opt_random} --decode {stratstr}"
      if stratstr.startswith("lrs"):
        lrs_trace_file = os.path.join(HP.SCRATCH,"{}_{}.lrs".format(prob.replace("/","_"),os.getpid()))
        opts1 += f" -lstf {lrs_trace_file}"

      result = IC.vampire_perfrom(prob,opts1+opts2)
      if result.status == "uns":
        trace_file_path = os.path.join(traces_dir,"{}_{}.pt".format(prob.replace("/","_"),len(traces_collected)))
        ilim = max(10*ilim,5000) # to have enough instructions/time to load a model
        lrs_trace_str = f" -lltf {lrs_trace_file}" if lrs_trace_file else ""
        opts1 = f"-t {ilim2tlim(ilim)} -i {ilim} {lrs_trace_str} -ncem {random_script_model_file_path} -nar {trace_file_path} -ncf {HP.NUM_CLAUSE_FEATURES} -npf {HP.NUM_PROBLEM_FEATURES}"

        vamp_res = IC.vampire_perfrom(prob,opts1+opts2)
        print("      gather",opts1+opts2)

        try:
          if vamp_res.status != "uns":
            raise AssertionError(f"Gather failed to reproduce for {prob} {opts1+opts2}")

          trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_file_path,train_log)
          if trace_kept:
            traces_collected.append(trace_file_path)
          else:
            raise ValueError(f"Trace was either trivial or too big for {prob} {opts1+opts2}")
        except Exception as e:
          print(e)
          os.remove(trace_file_path)
          fauls += 1

        # if the trace was too ugly, don't even try again with this strategy
        break
      else:
        print("      Iter",i,"failed to reprove",prob,opts1+opts2)

    if lrs_trace_file and os.path.isfile(lrs_trace_file):
              os.remove(lrs_trace_file)

  print("  Collected",len(traces_collected),"traces for",prob)
  sys.stdout.flush()
  return (prob,traces_collected)

def collect_traces_POOL(solutions_list):
  if True:
    pool = Pool(processes=parallelism) # number of cores to use
    results = pool.map(collect_traces, solutions_list, chunksize = 1)
    pool.close()
    pool.join()
    del pool
  else:
    results = []
    for task in solutions_list:
      results.append(collect_traces(task))
  return results


if __name__ == "__main__":
  # Slooper is elooper updated for creating master brain models from strategy runs (delivered by snake)

  parallelism = int(sys.argv[1])

  RECOVERING = True

  if RECOVERING:
    traces_from =  sys.argv[2]

    partial_trace_index = defaultdict(list)
    all_trace_paths = []

    traces_dir = os.path.join(traces_from,"traces")
    # scan traces_dir for files; they will be of the shape "Problems_KRS_KRS101+1.p_0.pt"
    root, dirs, files = next(os.walk(traces_dir))
    for trace_file in files:
      trace_file_path = os.path.join(traces_dir,trace_file)

      all_trace_paths.append(trace_file_path)

      spl = trace_file.split("_")
      prob = "/".join(spl[:-1])
      partial_trace_index[prob].append(trace_file_path)

    #for prob,traces_collected in partial_trace_index.items():
    #  print(prob,len(traces_collected))
    print("Collected traces for",len(partial_trace_index),"problems")

  if False:
    def process_trace(trace_path):
      trace_path = trace_path.strip()
      try:
        ttuple = torch.load(trace_path)
        # check if it's an int
        if isinstance(ttuple[3], int):
          pass
        else:
          trace_kept, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_path, sys.stdout)
          if not trace_kept:
            print("Not good for learning", trace_path)
            raise AssertionError()
      except:
        print("Will delete",trace_path)
        os.remove(trace_path)

    with multiprocessing.Pool(120) as pool:
      pool.map(process_trace, all_trace_paths)

  if RECOVERING:
    exper_dir = sys.argv[3]
  else:
    exper_dir = sys.argv[2]

  # get the exper dir ready fast
  os.mkdir(exper_dir)
  # Documentation: save hyperparams and campaign
  # (CAREFUL: this way can only call looper from lawa folder)
  shutil.copy("hyperparams.py",exper_dir)

  if not RECOVERING:
    traces_dir = os.path.join(exper_dir,"traces")
    os.mkdir(traces_dir)

  train_log = open(os.path.join(exper_dir,'detailed.log'), 'w', buffering=1)

  # Initializing a model and an optimizer (might still get better one below from load_dir if given)
  model = IC.get_initial_model()
  optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  random_script_model_file_path = os.path.join(exper_dir,"random-script-model.pt")
  IC.export_model(model.state_dict(),random_script_model_file_path)

  if not RECOVERING:
    # Load the strategy data:
    solutions = defaultdict(list) # from problems to list[(instr,strat)]
    for folder in HP.SNAKE_INPUT_DIRS:
      root, dirs, files = next(os.walk(folder))
      for pklfile in files:
        if not pklfile.endswith(".pkl"):
          continue
        pklpath = os.path.join(folder, pklfile)

        with open(pklpath,"rb") as f:
          meta,results = pickle.load(f)

        stratstr = meta[2]
        evallimit = int(meta[-1])
        # print(pklfile,stratstr,evallimit)

        if stratstr == "lrs+10_1:1__0": # the default strat
          # print("Default was with",evallimit)
          stratstr = f"lrs+10_1:1_sil={evallimit}:i={evallimit}_0"

        # make the strategy limitless!
        strdict = str2dict(stratstr)
        # this is needed at least because of the order in which we present our explicit "-i <ilim>" and the "--decode strat"
        # (normally the -i as a documentation, showig us the value with which minimizer finished on the witness problem)
        if "i" in strdict:
          del strdict["i"]
        # acc=model is evil, as there is no causal parent for (some) avatar splits
        if "acc" in strdict and strdict["acc"] == "model":
          strdict["acc"] = "on"
        stratstr = sorted_dict2str(strdict)

        for longname,instr_parse,instr,res in results:
          if res == "uns": # or res == "sat"
            solutions[longname].append((instr,stratstr))

    tasks = []
    for prob,solns in solutions.items():
      #if prob in partial_trace_index and len(partial_trace_index[prob]) >= HP.SNAKE_MAX_TRACES_PER_PROBLEM:
      #  print("Already happy for",prob)
      #else:
      tasks.append((prob,solns))

    primitive_trace_index = collect_traces_POOL(tasks)

  # for now, we just scan the dir
  # save_trace_index(exper_dir,primitive_trace_index)


  # ===========================================================================
  # ===========================================================================
  # the parallel business set up here:

  # create our worker processes and register a cleanup
  eval_and_train_in = multiprocessing.Queue()
  eval_and_train_out = multiprocessing.Queue()
  my_processes = []

  eval_and_train_parallelism = min(parallelism,HP.TRAINING_PARALLELISM)

  for i in range(eval_and_train_parallelism):
    p = multiprocessing.Process(target=worker, args=(eval_and_train_in,eval_and_train_out))
    p.start()
    my_processes.append(p)

  def cleanup():
    for p in my_processes:
      p.kill()
    train_log.close()
  atexit.register(cleanup)

  def do_in_parallel(q_in,q_out,tasks,max_parallelism,process_results_callback):
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

  def eval_and_train_in_parallel(tasks,process_results_callback):
    do_in_parallel(eval_and_train_in,eval_and_train_out,tasks,eval_and_train_parallelism,process_results_callback)


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

  trace_index = {}
  if RECOVERING:
    if False: # just for quick debugging
      for prob,traces in list(partial_trace_index.items())[:10]:
        trace_index[prob] = traces[:1]
    else:
      for prob,traces in list(partial_trace_index.items()):
        trace_index[prob] = traces
  else:
    for prob,traces in primitive_trace_index:
      if traces:
        trace_index[prob] = traces

  print("Starting with trace_index of",len(trace_index))
  sys.stdout.flush()

  # ===========================================================================

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

  iter = 1
  loop = 1

  # for this iteration, we write stuff here:
  cur_dir = claim_loop_dir(loop)

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

  trace_problems = list(trace_index.keys())
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
            yield (JK_EVAL,(prob,fact*1.0,trace_index[prob],eval_model_file_path))

      def process_results_from_eval(job_kind,input,result):
        global weighted_eval_loss
        assert job_kind == JK_EVAL
        weighted_eval_loss += result # (= the loss) multiplied by fact already in the child
        return 1

      pre_eval = time.time()
      weighted_eval_loss = 0.0
      eval_and_train_in_parallel(get_eval_tasks(),process_results_from_eval)
      print("Eval loss on valid",weighted_eval_loss,"in",int(time.time()-pre_eval),"s")
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
      proto_tasks = [[prob,fact*1.0,trace_index[prob]] for prob in train_trace_problems]

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

    pre_train = time.time()
    eval_and_train_in_parallel(get_train_tasks(),process_results_from_train)

    print("Weighted train loss",weighted_train_loss,"in",int(time.time()-pre_train),"s")
    print()
    sys.stdout.flush()

    torch.save(model.state_dict(), os.path.join(cur_dir,f"model_snapshot{stage2iter}.tar"))

    if TIW == 1:
      os.remove(eval_model_file_path) # ???
      break

  # stage 2
  print()
  print(f"  Stage 2 - took {time.time()-stage_start_time} seconds and {stage2iter-0.5} eval/train iterations")
  print()
  sys.stdout.flush()
  """
  if HP.LEARNING_RATE_DECAY < 1.0:
    print("Learning rate decrease")
    for g in optimizer.param_groups:
      print(" from",g['lr'],end=" ")
      g['lr'] *= HP.LEARNING_RATE_DECAY
      print("to",g['lr'])
    print()
  """

  save_loop_model_and_optimizer(cur_dir,loop,model,optimizer)
  print()
  sys.stdout.flush()
