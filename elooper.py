#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP
import workers as W

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain
from dataclasses import dataclass

import multiprocessing
import numpy

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

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

LOOP_MODEL = "loop-model.tar"

def save_loop_model(cur_dir,loop,model):
  loop_model_state_file_path = os.path.join(cur_dir,LOOP_MODEL)
  torch.save((loop,model.state_dict()), loop_model_state_file_path)

def load_loop_model(adir):
  loop_model_state_file_path = os.path.join(adir,LOOP_MODEL)
  return torch.load(loop_model_state_file_path)

def is_sound(trace_file_name):
  if not os.path.isfile(trace_file_name):
    print("Trace file",trace_file_name,"no longer exists")
    return False
  tt = torch.load(trace_file_name)
  if isinstance(tt[3], int):
    return True
  print("Unconverted trace file",trace_file_name)
  return False

def filter_trace_file_names(task):
  prob,trace_file_names = task
  return prob,[tfn for tfn in trace_file_names if is_sound(tfn)]

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

  def report_bad_trace(self,loop,prob):
    if (HP.CUMULATIVE and prob in self.last_solved and self.last_solved[prob] < loop):
      # we didn't solve it this loop yet
      if prob in self.traces:
        del self.traces[prob]

    self.last_solved[prob] = loop

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

  def consolidate(self):
    with multiprocessing.Pool(120) as pool:
      for prob,trace_file_names_updated in pool.map(filter_trace_file_names, list(self.traces.items())):
        self.traces[prob] = trace_file_names_updated

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

def no_dots(name):
  return name.replace(".","_")

def ilim2tlim(ilim):
  secs = max(5,ilim // 1000) # it's 2 times more than the instrlimit on a 2GHz machine
  return secs

def luby(min,max):
  next = min
  seq = [min]
  i = 0
  while True:
    while i < len(seq):
      yield seq[i]
      i += 1
    next *= 2
    if next > max:
      break
    seq = seq + seq
    seq.append(next)
  # do "repeat forever", for now
  while True:
    for i in seq:
      yield i


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

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

  trace_index = TraceIndex()

  tweak_map = torch.nn.ParameterDict()

  script_model_to_steal = None
  skip_first_stage = False

  loop = 0
  if len(sys.argv) > 5: # we already know the folder, but which loop to copy from there?
    loop = int(sys.argv[5])
    load_dir = os.path.join(folder_with_prev_exper,f"loop{loop}")

    load_model = True
    load_traces_new = False
    load_traces_old = False
    steal_script_model = False

    if len(sys.argv) > 6:
      load_model = "m" in sys.argv[6]
      load_traces_new = "t" in sys.argv[6]
      load_traces_old = "T" in sys.argv[6]
      steal_script_model = "s" in sys.argv[6]

    if load_model:
      aloop,amodel_state_dict = load_loop_model(load_dir)
      assert aloop == loop
      model.load_state_dict(amodel_state_dict)

    if load_traces_new:
      trace_index = load_trace_index(os.path.join(folder_with_prev_exper,f"loop{loop+1}"))
      skip_first_stage = True
      print("Starting from loop",loop,"and a half")
      trace_index.consolidate()
      trace_index.report()

    if load_traces_old:
      trace_index = load_trace_index(os.path.join(folder_with_prev_exper,f"loop{loop}"))
      print("Starting from loop",loop)
      trace_index.consolidate()
      trace_index.report()

    if steal_script_model:
      script_model_to_steal = os.path.join(load_dir,"script-model.pt")

  else:
    cur_dir = claim_loop_dir(loop)
    save_loop_model(cur_dir,loop,model)

  print_model_part()

  assert loop_count > 0

  # ===========================================================================
  # ===========================================================================
  # the parallel business set up here:

  W.train_log = open(os.path.join(exper_dir,'detailed.log'), 'w', buffering=1)

  # create our worker processes and register a cleanup
  perf_and_gather = W.create_workforce(parallelism)
  eval_and_train_parallelism = min(parallelism,HP.TRAINING_PARALLELISM)
  eval_and_train = W.create_workforce(eval_and_train_parallelism)
  W.workforces_finished()

  def perform_and_gather_in_parallel(tasks,process_results_callback):
    W.do_in_parallel(perf_and_gather,tasks,parallelism,process_results_callback)

  def eval_and_train_in_parallel(tasks,process_results_callback):
    W.do_in_parallel(eval_and_train,tasks,eval_and_train_parallelism,process_results_callback)

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

    if not skip_first_stage:
      # ===========================================================================
      # STAGE 1: PERFORM and GATHER
      stage_start_time = time.time()

      # There is going to be files to store the results, ...
      result_metas = [] # ... will store the file names and some additional info (in order of generation)
      result_dicts = defaultdict(lambda : IC.default_defaultdict_of_list()) # ... will collect the dicts to go into the respective files
      stats = IC.default_defaultdict_of_list()

      script_model_file_path = os.path.join(cur_dir,"script-model.pt")
      if script_model_to_steal:
        shutil.copy(script_model_to_steal,script_model_file_path)
        print("Stolen script model from",script_model_to_steal)
        script_model_to_steal = None
      else:
        IC.export_model(model.state_dict(),script_model_file_path)

      # we need this, so that we don't gather to the same index if two perf jobs come too close one after another
      per_prob_trace_cnt = defaultdict(int)

      def get_perform_tasks():
        global per_prob_trace_cnt
        for mission,gatherwish,prob_lists in [("train",True,train_problems),("test",False,test_problems)]:
          if not HP.EVAL_ON_TEST and mission == "test":
            continue
          res_filename = f"{mission}_res.pt"

          result_metas.append((res_filename,mission))

          for i,ilim in enumerate(luby(HP.INSTRUCTION_LIMIT_MIN,HP.INSTRUCTION_LIMIT_MAX)):
            if i >= HP.NUM_PERFORMS and loop > 1 or i >= HP.INITIAL_NUM_PERFORMS:
              break
            seed = random.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp

            # print(i,"for",ilim)

            # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
            opts1_base = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"

            if HP.RANDOMIZED_STRATEGIES:
              saturation_algorithm = ""
              opts1_base += f" --sample_strategy {HP.RANDOMIZED_STRATEGIES}"
            else:
              saturation_algorithm = f"-sa {HP.SATURATION_ALGORITHM}"

            # will stay the same
            opts2_base = f" {HP.SHUFFLING_OPTIONS} {saturation_algorithm} -ncf {HP.NUM_CLAUSE_FEATURES} -npf {HP.NUM_PROBLEM_FEATURES}"

            if not HP.IMITATE or loop > 1:
              opts2_base += f" -npcc on -ncem {script_model_file_path}"
            if HP.IMITATE and loop > 1:
              opts2_base = HP.NON_IMIT_EXTRA + opts2_base

            if HP.USE_SPECIAL:
              opts2_base += HP.PERFORMS_SPECIAL[i]

            for prob in prob_lists:
              if per_prob_trace_cnt[prob] >= HP.MAX_TRACES_TO_KEEP:
                # print("Skipping for",prob,"who already has enough")
                # we are starting to skip problems that already have enough traces
                continue

              opts1 = opts1_base
              if HP.SATURATION_ALGORITHM.startswith("lrs") or HP.RANDOMIZED_STRATEGIES:
                lrs_trace_file = os.path.join(HP.SCRATCH,"{}_{}_{}_{}.lrs".format(prob.replace("/","_"),i,seed,os.getpid()))
                opts1 += f" -lstf {lrs_trace_file}"
              else:
                lrs_trace_file = ""

              # perf_log = os.path.join(cur_dir,"{}_{}_{}_{}.perf".format(prob.replace("/","_"),i,seed,os.getpid()))
              perf_log = None

              yield (W.JK_PERFORM,(res_filename,gatherwish,mission,prob,i,ilim,lrs_trace_file,opts1,opts2_base + f" --random_seed {seed}",perf_log))

      def process_results_from_perform_and_gather(job_kind,input,result):
        global per_prob_trace_cnt
        workers_freed = 0
        if job_kind == W.JK_PERFORM:
          (res_filename,gatherwish,mission,prob,i,ilim,lrs_trace_file,opts1,opts2,perf_log) = input
          result_dicts[res_filename][prob].append((i,ilim,result))

          # the discrepancy between num_contemporary_traces and counter might lead to us in the end having a bit more traces than NUM_TRACES_TO_KEEP, but that's better than fewerwho already has enough
          if result.status == "uns" and gatherwish and (per_prob_trace_cnt[prob] < HP.MAX_TRACES_TO_KEEP):
            counter = per_prob_trace_cnt[prob]
            per_prob_trace_cnt[prob] += 1

            # print("Gather for",prob,"counter:",counter,"ncp:",trace_index.num_contemporary_traces(loop,prob))

            trace_file_path = os.path.join(traces_dir,"{}_{}.pt".format(prob.replace("/","_"),counter))

            ilim *= 10
            lrs_trace_str = f" -lltf {lrs_trace_file}" if lrs_trace_file else ""
            # -nar needs a model, and with imitation it's not added to the JK_PERFORM options
            model_for_imitation = f"-ncem {script_model_file_path}" if HP.IMITATE and loop == 1 else ""

            if HP.RANDOMIZED_STRATEGIES:
              strat_str = result.strategy
              # kick out the implicit time limit at the end after the last _
              strat_str = "_".join(strat_str.split("_")[:-1] + ["0"])
              insert_decode = f"--decode {strat_str}"
            else:
              insert_decode = ""

            gather_log = perf_log+".gather" if perf_log else None

            task = (W.JK_GATHER,(mission,prob,lrs_trace_file,trace_file_path,
                                f"-t {ilim2tlim(ilim)} {insert_decode} -i {ilim} {model_for_imitation} -nar {trace_file_path} {lrs_trace_str}"+opts2,opts1+opts2,gather_log))
            # print("PUT:",task)
            perf_and_gather[0].put(task)
          else:
            workers_freed = 1
            if lrs_trace_file and os.path.isfile(lrs_trace_file):
              os.remove(lrs_trace_file)
        elif job_kind == W.JK_GATHER:
          (mission,prob,lrs_trace_file,trace_file_path,opts,eval_opts,gather_log) = input
          non_trivial, passes_limits, gage_stats, gweight_stats = result
          stats[prob].append((gage_stats, gweight_stats))
          if non_trivial and passes_limits:
            trace_index.add_prob_trace(loop,prob,trace_file_path)
          else:
            # TODO: in a future version, we shouldn't first overwrite the old trace file with the new raw one
            # not until trace_good_for_learning decides what the new one looks like
            #
            # but now that the old is overwritten and the new one cannot be learned from for one of the reasons, let's just delete the file
            try:
              os.remove(trace_file_path)
            except FileNotFoundError:
              # TODO: think; how could it be that the file does not exists? (It happened though)
              pass
            trace_index.report_bad_trace(loop,prob)

          workers_freed = 1
          if lrs_trace_file and os.path.isfile(lrs_trace_file):
            os.remove(lrs_trace_file)
        else:
          assert False, f"Surprised by job_kind {job_kind}"

        return workers_freed

      perform_and_gather_in_parallel(get_perform_tasks(),process_results_from_perform_and_gather)

      torch.save(stats,os.path.join(cur_dir,"stats.pt"))

      # let's report what happened so far (and save the results into files, for later analysis):
      for (res_filename,mission) in result_metas:
        results = result_dicts[res_filename]
        torch.save(("Unused",results), os.path.join(cur_dir,res_filename))

        by_performs_attempted = defaultdict(set)
        by_performs_solved = defaultdict(set)
        ilims = defaultdict(int)

        max_i = 0

        prob_solved = 0
        prob_fractional = 0.0
        for prob,runs in results.items():
          succs = 0
          for (i,ilim,vamp_res) in runs:
            max_i = max(i,max_i)
            ilims[i] = ilim
            by_performs_attempted[i].add(prob)
            if vamp_res.status == "uns":
              succs += 1
              by_performs_solved[i].add(prob)

          if succs > 0:
            prob_solved += 1

        print(res_filename)
        print("          {:10.4f} = {:>5} / {:>5} ADDING".format(prob_solved/len(results),prob_solved,len(results)))

        covered = set()
        adds = []

        for i in range(max_i+1):
          adds.append(len(by_performs_solved[i]-covered))
          covered = covered | by_performs_solved[i]

        for i in range(max_i+1):
          print("   {:>3} {:>6} {:6.4f} = {:>5} / {:>5} {:>5} {}".format(
                  i,ilims[i],len(by_performs_solved[i])/len(by_performs_attempted[i]),len(by_performs_solved[i]),len(by_performs_attempted[i]),adds[i],HP.PERFORMS_SPECIAL[i] if i < len(HP.PERFORMS_SPECIAL) else None))

      print()
      print("  Stage 1 took",time.time()-stage_start_time)
      print()
      sys.stdout.flush()

      trace_index.update_scores(loop)
      trace_index.report()
      save_trace_index(cur_dir,trace_index)

      print()
      sys.stdout.flush()

    skip_first_stage = False # only possibly skipped for the first loop

    # ===========================================================================

    # don't run the full last loop - otherwise, we are training a model nobody will see evaluated
    if loop_count == 0:
      break

    # STAGE 2a: TWEAKIT - i.e., look for favorable tweaks to all gathered traces

    pre_tweaking = time.time()
    tweaking_model_file_path = os.path.join(HP.SCRATCH,"tweaking-model_{}.tar".format(os.getpid()))
    torch.save(model.state_dict(), tweaking_model_file_path)

    tweak_file_version = 0
    def get_tweaking_tasks():
      global tweak_file_version
      prob_records = W.create_prob_records(list(trace_index.cur_problems()),trace_index)
      prob_records.sort(key = lambda rec : -rec.szs) # descending, first by the filesizes (i.e., the big ones first)
      for record in prob_records:
        tweak_file_version += 1
        tweak_file_path = os.path.join(HP.SCRATCH,"tweak_{}_{}.tar".format(os.getpid(),tweak_file_version))

        prob_no_dots = no_dots(record.prob)
        if record.prob not in tweak_map:
          tweak_map[prob_no_dots] = IC.get_fresh_tweak()

        torch.save(tweak_map[prob_no_dots], tweak_file_path)
        yield (W.JK_EVAL_TWEAK,(record,tweaking_model_file_path,tweak_file_path))

    weighted_tweaking_stats = defaultdict(float)
    def process_results_from_tweaking(job_kind,input,result):
      global weighted_tweaking_stats

      record,_tweaking_model_file_path,tweak_file_path = input

      assert job_kind == W.JK_EVAL_TWEAK
      stat_dict = result
      for k,v in stat_dict.items(): # includes the loss; all multiplied by fact already in the child
        weighted_tweaking_stats[k] += v

      prob_no_dots = no_dots(record.prob)
      tweak_map[prob_no_dots] = torch.load(tweak_file_path)
      os.remove(tweak_file_path)
      return 1

    eval_and_train_in_parallel(get_tweaking_tasks(),process_results_from_tweaking)
    os.remove(tweaking_model_file_path)

    print("Tweaking on all",weighted_tweaking_stats,"in",int(time.time()-pre_tweaking),"s")
    sys.stdout.flush()

    # STAGE 2b: alternate EVAL, TRAIN, EVAL until no longer improving
    print()
    sys.stdout.flush()
    stage_start_time = time.time()

    # compute LR for our loop, taking into account our decay
    lr_wish = HP.LEARNING_RATE * (HP.LEARNING_RATE_DECAY ** (loop-1))
    print("Learning rate now at",lr_wish)

    # newly only lives one iter, so no need to save it
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},      #, "lr": lr_wish},
        {"params": tweak_map.parameters()}], # TODO: tweaks could have a different learning rate!
        lr=lr_wish, weight_decay=HP.WEIGHT_DECAY)

    TIW = HP.TEST_IMPROVE_WINDOW
    assert TIW > 0
    eval_models = [None]*TIW
    eval_losses = [None]*TIW
    stage2iter = 0

    trace_problems = list(trace_index.cur_problems())
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
          prob_records = W.create_prob_records(valid_trace_problems,trace_index)
          prob_records.sort(key = lambda rec : -rec.szs) # descending by the filesizes (i.e., the big ones first)
          for record in prob_records:
            yield (W.JK_EVAL_TWEAK,(record,eval_model_file_path,None))

        weighted_eval_stats = defaultdict(float)
        def process_results_from_eval(job_kind,input,result):
          global weighted_eval_stats

          assert job_kind == W.JK_EVAL_TWEAK
          stat_dict = result
          for k,v in stat_dict.items(): # includes the loss; all multiplied by fact already in the child
            weighted_eval_stats[k] += v
          return 1

        pre_eval = time.time()
        eval_and_train_in_parallel(get_eval_tasks(),process_results_from_eval)
        print("Eval on valid",weighted_eval_stats,"in",int(time.time()-pre_eval),"s")
        sys.stdout.flush()

        eval_losses[stage2iter % TIW] = weighted_eval_stats["loss"]

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
        prob_records = W.create_prob_records(train_trace_problems,trace_index)
        random.shuffle(prob_records)
        global train_model_version
        for record in prob_records:
          train_model_version += 1
          train_model_file_path = os.path.join(HP.SCRATCH,"train-model-state_{}_{}.tar".format(os.getpid(),train_model_version))
          prob_no_dots = no_dots(record.prob)
          model.tweaks[IC.MAIN_TWEAK_NAME] = tweak_map[prob_no_dots]
          torch.save(model.state_dict(), train_model_file_path)
          yield (W.JK_TRAIN,(record,train_model_file_path))

      weighted_train_loss = 0.0
      weigthed_train_selection_hit_rate = 0.0
      weighted_train_dist_to_good = 0.0

      def process_results_from_train(job_kind,input,result):
        global weighted_train_loss
        global weigthed_train_selection_hit_rate
        global weighted_train_dist_to_good

        assert job_kind == W.JK_TRAIN
        (record,train_model_file_path) = input
        loss, selection_hit_rate, dist_to_good, took = result

        weighted_train_loss += loss # (= the loss) multiplied by fact already in the child
        weigthed_train_selection_hit_rate += selection_hit_rate
        weighted_train_dist_to_good += dist_to_good

        # train_model_version = int(train_model_file_path.split("_")[-1][:-4])
        # print(f"    BACK: {train_model_version:4d} after {took}s")

        # gnn_norm, gage_norm, gweight_norm, cleval_norm = 0.0, 0.0, 0.0, 0.0

        prob_no_dots = no_dots(record.prob)
        print(f"   Train tweak for {record.prob} before: {tweak_map[prob_no_dots].norm()}")

        # copy from result parameters to our model's gradients
        grad_loader_temp.load_state_dict(torch.load(train_model_file_path))

        # copy_grads_back_from_param
        for (name, param), param_copy in zip(model.named_parameters(),grad_loader_temp.parameters()):
          '''
          if name.startswith("gnn"):
            gnn_norm += param_copy.data.norm(2) ** 2
          elif name.startswith("gage"):
            gage_norm += param_copy.data.norm(2) ** 2
          elif name.startswith("gweight"):
            gweight_norm += param_copy.data.norm(2) ** 2
          else:
            cleval_norm  += param_copy.data.norm(2) ** 2
          '''
          param.grad = param_copy

        # what the worker thought was happening for model.tweaks[IC.MAIN_TWEAK_NAME], we now stage (for the optimizer) in tweak_map[prob_no_dots]:
        tweak_map[prob_no_dots].grad = model.tweaks[IC.MAIN_TWEAK_NAME].grad
        model.tweaks[IC.MAIN_TWEAK_NAME].grad = None

        optimizer.step()
        print(f"   Train tweak for {record.prob} after: {tweak_map[prob_no_dots].norm()}")

        # print("       with norms gnn/gage/gweight/cleval:",gnn_norm ** 0.5, gage_norm ** 0.5, gweight_norm ** 0.5, cleval_norm ** 0.5)
        os.remove(train_model_file_path)
        return 1

      pre_train = time.time()
      eval_and_train_in_parallel(get_train_tasks(),process_results_from_train)

      print("Weighted train loss",weighted_train_loss,"selection_hit_rate",weigthed_train_selection_hit_rate,"dist_to_good",weighted_train_dist_to_good,"in",int(time.time()-pre_train),"s")
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

    print_model_part()
    save_loop_model(cur_dir,loop,model)
    print()
    sys.stdout.flush()

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()

    # so that we can distinguish standard run from a "loop and a half" start (which reads a ready trace index and then skips Stage 1)
    trace_index.loop_finished()
