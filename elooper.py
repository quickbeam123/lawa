#!/usr/bin/env python3

# PRETEND THERE IS NO CUDA HERE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import inf_common as IC
import hyperparams as HP
import workers as W

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain
from dataclasses import dataclass

import multiprocessing
import numpy as np

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

rng = random.Random(HP.RANDOM_SEED)

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
  our_problems = rng.sample(full_list,HP.NUM_TRAIN_PROBLEMS + HP.NUM_TEST_PROBLEMS)
  rng.shuffle(our_problems)
  return our_problems[:HP.NUM_TRAIN_PROBLEMS],our_problems[HP.NUM_TRAIN_PROBLEMS:]

def save_train_test_problems(exper_dir,train_problems,test_problems):
  for filename,problems in [(TRAIN_PROBLEMS_FILE,train_problems),(TEST_PROBLEMS_FILE,test_problems)]:
    with open(os.path.join(exper_dir,filename),"w") as f:
      for p in problems:
        f.write(p)
        f.write("\n")

def is_sound(trace_file_name):
  if not os.path.isfile(trace_file_name):
    print("Trace file",trace_file_name,"no longer exists")
    return False
  try:
    tt = torch.load(trace_file_name)
  except EOFError:
    print("Corrupted (unfinished) trace file",trace_file_name)
    return False
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
  return torch.load(trace_index_file_path,weights_only=False)

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

# ============================================================================================

LOOP_MODEL = "loop-model.tar"

class Context:
  def __init__(self):
    self.loop = 0
    self.model = IC.get_initial_model()
    self.trace_index = TraceIndex()

  def init_train_test(self,folder_with_prev_exper):
    if folder_with_prev_exper:
      self.train_problems,self.test_problems = steal_problems_from(folder_with_prev_exper)
    else:
      self.train_problems,self.test_problems = train_test_problem_split()
    save_train_test_problems(exper_dir,self.train_problems,self.test_problems)

  def claim_loop_dir(self):
    loop_str = "loop{}".format(self.loop)
    self.cur_dir = os.path.join(exper_dir,loop_str)
    os.mkdir(self.cur_dir)

    return loop_str

  def save_loop_model(self):
    loop_model_state_file_path = os.path.join(self.cur_dir,LOOP_MODEL)
    torch.save((self.loop,self.model.state_dict()), loop_model_state_file_path)

  def load_loop_model(self,adir):
    loop_model_state_file_path = os.path.join(adir,LOOP_MODEL)
    aloop,amodel_state_dict = torch.load(loop_model_state_file_path)
    assert aloop == self.loop
    self.model.load_state_dict(amodel_state_dict)

  def load_loop_model_old(self,adir):
    loop_model_state_file_path = os.path.join(adir,"loop-model-and-optimizer.tar")
    aloop,amodel_state_dict,_an_optim_dict = torch.load(loop_model_state_file_path)
    assert aloop == self.loop
    self.model.load_state_dict(amodel_state_dict,strict=False)


# ============================================================================================

# taken from snake's minilib
def str2dict(str):
  spl = str.split("_")
  assert len(spl) >= 3
  res = {}

  # hack in long versions of saturation algorithm names (convert from the 3letter ones used in the decode string)
  sa_longs = {"lrs" : "lrs", "dis" : "discount", "ott" : "otter", "fmb" : "fmb"}
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

def load_starts_from_folders(folders):
  strats = {}
  for folder in folders:
    root, dirs, files = next(os.walk(folder))
    for pklfile in files:
      if not pklfile.endswith(".pkl"):
        continue

      pklpath = os.path.join(folder, pklfile)
      spl = pklfile.split("_")
      stamp = int(spl[-2])

      with open(pklpath,"rb") as f:
        meta,results = pickle.load(f)
        eval_instr = int(meta[-1])
        strat = meta[2]
        perf_vec = []

        for longname,instr_parse,instr,res in results:
          if res == "uns":
            perf_vec.append((1+instr,longname))

        strats[(pklpath,strat)] = sorted(perf_vec)

  print("Loaded",len(strats),"strats")
  return strats

GREED_FACTOR = 1.3

def contrib_fla(len_covered_prob):
  return 0.5**len_covered_prob

def accum_contrib_fla(len_covered_prob):
  accum = 0.0
  for i in range(len_covered_prob):
    accum += contrib_fla(i)
  return accum

def get_best_contrib(perf_vec,base,covered):
  new_perf_vec = []
  best_weight = 0.0
  best_instr = 0

  contributing = 0
  for instr,prob in perf_vec:
    len_covered_prob = len(covered[prob])
    if len_covered_prob >= HP.SNAKE_MAX_TRACES_PER_PROBLEM:
      continue

    new_perf_vec.append((instr,prob))
    contributing += contrib_fla(len_covered_prob)
    weight = math.pow(contributing,GREED_FACTOR)/(instr-base)

    if weight > best_weight:
      best_weight = weight
      best_instr = instr

  return new_perf_vec,best_weight,best_instr

def prune_contributing_strat(perf_vec,strat,newbase,covered):
  new_perf_vec = []
  added = 0.0
  for instr,prob in perf_vec:
    if instr <= newbase:
      added += contrib_fla(len(covered[prob]))
      covered[prob].append((strat,instr))
    else:
      new_perf_vec.append((instr,prob))
  return new_perf_vec,added

def construct_schedule(strats,covered):
  total_budget = 0

  cur_sched = { strat : 0 for strat in strats }

  while True:
    best_weight = 0.0
    best_strat = None
    best_instr = None
    for strat in strats:
      strats[strat],weight,instr = get_best_contrib(strats[strat],cur_sched[strat],covered)
      if (weight > best_weight):
        best_weight = weight
        best_strat = strat
        best_instr = instr

    if best_strat == None:
      break

    print("Ext from",cur_sched[best_strat],"to",best_instr,"weigth",best_weight,"with",best_strat)

    cost = best_instr - cur_sched[best_strat]
    total_budget += cost
    cur_sched[best_strat] = best_instr

    strats[best_strat],added = prune_contributing_strat(strats[best_strat],best_strat,best_instr,covered)
    print("Added",added,"problems for a cost of",cost,"instructions")

    print("Now covering",sum(accum_contrib_fla(len(cover)) for prob,cover in covered.items()),"for a budget of",total_budget)

  return covered

def collect_trace(task):
  prob,trace_id,strat,instr,script_model_file_path = task
  stratstr = strat[1]
  print("collect_trace for",prob,instr,stratstr)

  # make the strategy limitless!
  strdict = str2dict(stratstr)
  # this is needed at least because of the order in which we present our explicit "-i <ilim>" and the "--decode strat"
  # (normally the -i as a documentation, showing us the value with which minimizer finished on the witness problem)
  if "i" in strdict:
    del strdict["i"]
  stratstr = sorted_dict2str(strdict)

  # if expected to take too long, don't even bother reproving
  if instr > HP.SNAKE_MAX_INSTRUCTIONS:
    return None

  result = None

  # add 10% extra (and don't be a Scrooge)
  ilim = max(int(1.1*instr),125)

  # try reprove under shuffling
  lrs_trace_file = None
  for i in range(HP.SNAKE_MAX_TRIES):
    if i == HP.SNAKE_MAX_TRIES-1:
      # print("      will try unshuffled one")
      opt_random = ""
    else:
      seed = random.randint(1,0x7fffff)
      opt_random = f"{HP.SHUFFLING_OPTIONS} --random_seed {seed}"

    # will change for the gathering job (but note that "-t something" is always the first option pair via a convention in run_lawa_vampire)
    opts1 = f"-t {HP.SNAKE_RECONSTRUCT_TIMEOUT_SECOND} -i {ilim}"
    opts2 = f" -p off --parsing_does_not_count on {opt_random} --decode {stratstr}"
    if stratstr.startswith("lrs"):
      lrs_trace_file = os.path.join(HP.SCRATCH,"{}_{}.lrs".format(prob.replace("/","_"),os.getpid()))
      opts1 += f" -lstf {lrs_trace_file}"

    if W.vampire_perfrom(prob,opts1+opts2,None).status == "uns":
      trace_file_path = os.path.join(traces_dir,"{}_{}.pt".format(prob.replace("/","_"),trace_id))
      ilim = max(10*ilim,5000) # to have enough instructions/time to load a model
      lrs_trace_str = f" -lltf {lrs_trace_file}" if lrs_trace_file else ""
      opts1 = f"-t {HP.SNAKE_RECONSTRUCT_TIMEOUT_SECOND} -i {ilim} {lrs_trace_str} -ncem {script_model_file_path} -nar {trace_file_path} -ncf {HP.NUM_CLAUSE_FEATURES} -npf {HP.NUM_PROBLEM_FEATURES}"

      vamp_res = W.vampire_perfrom(prob,opts1+opts2,None) # since opts2 comes second, the ncem from a neural strategy will overrule our ncem=random_script_model_file_path
      # print("      gather",opts1+opts2)

      try:
        if vamp_res.status != "uns":
          raise AssertionError(f"Gather failed to reproduce for {prob} {opts1+opts2}")

        non_trivial, passes_limits, gage_stats, gweight_stats = IC.trace_good_for_learning(trace_file_path,W.train_log)
        if non_trivial and passes_limits:
          result = (prob,trace_id,trace_file_path)
          break
        else:
          raise ValueError(f"Trace was either trivial or too big for {prob} {opts1+opts2}")
      except Exception as e:
        print(e)
        if os.path.isfile(trace_file_path):
          os.remove(trace_file_path)

      # if the trace was too ugly, don't even try again with this strategy
      break
    else:
      print("      Iter",i,"failed to reprove",prob,opts1+opts2)
      pass

  if lrs_trace_file and os.path.isfile(lrs_trace_file):
    os.remove(lrs_trace_file)

  sys.stdout.flush()
  return result

def collect_traces_POOL(tasks):
  if True:
    pool = multiprocessing.Pool(processes=parallelism) # number of cores to use
    results = pool.map(collect_trace, tasks, chunksize = 1)
    pool.close()
    pool.join()
    del pool
  else:
    results = []
    for task in tasks:
      results.append(collect_trace(task))
  return results

def stage_snake_gather(ctx):
  covered = defaultdict(list) # allow problems to be covered up to HP.SNAKE_MAX_TRACES_PER_PROBLEM many times (with exponentially diminishing rewards)
  for dirs in HP.SNAKE_INPUT_DIRS:
    strats = load_starts_from_folders(dirs)
    covered = construct_schedule(strats,covered)
    print("Num problems covered",len(covered))

  script_model_file_path = os.path.join(ctx.cur_dir,"script-model.pt")
  IC.export_model(ctx.model.state_dict(),script_model_file_path)

  results = collect_traces_POOL([(prob,i,strat,instr,script_model_file_path) for prob,cover in covered.items() for i,(strat,instr) in enumerate(cover) ])

  for result in results:
    if result is not None:
      prob,trace_id,trace_file_path = result
      ctx.trace_index.add_prob_trace(ctx.loop,prob,trace_file_path)

  ctx.trace_index.update_scores(ctx.loop)
  ctx.trace_index.report()
  save_trace_index(ctx.cur_dir,ctx.trace_index)

  print()
  sys.stdout.flush()

# ============================================================================================

def stage_perf_gather(ctx):
  stage_start_time = time.time()

  # There is going to be files to store the results, ...
  result_metas = [] # ... will store the file names and some additional info (in order of generation)
  result_dicts = defaultdict(lambda : IC.default_defaultdict_of_list()) # ... will collect the dicts to go into the respective files
  stats = IC.default_defaultdict_of_list()

  script_model_file_path = os.path.join(ctx.cur_dir,"script-model.pt")
  IC.export_model(ctx.model.state_dict(),script_model_file_path)

  # we need this, so that we don't gather to the same index if two perf jobs come too close one after another
  per_prob_trace_cnt = defaultdict(int)

  def get_perform_tasks():
    nonlocal per_prob_trace_cnt
    for mission,gatherwish,prob_lists in [("train",True,ctx.train_problems),("test",False,ctx.test_problems)]:
      if not HP.EVAL_ON_TEST and mission == "test":
        continue
      res_filename = f"{mission}_res.pt"

      result_metas.append((res_filename,mission))

      for i,ilim in enumerate(luby(HP.INSTRUCTION_LIMIT_MIN,HP.INSTRUCTION_LIMIT_MAX)):
        if mission == "test" and i >= 1:
          break
        if i >= HP.NUM_PERFORMS and ctx.loop > 1 or ctx.loop == 1 and i >= HP.INITIAL_NUM_PERFORMS:
          break
        seed = rng.randint(1,0x7fffff) # temperatures can be same (repeated), so let's have a new seed per temp

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

        if not HP.IMITATE or ctx.loop > 1:
          opts2_base += f" -npcc on -ncem {script_model_file_path}"
        if HP.IMITATE and ctx.loop > 1:
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
    nonlocal per_prob_trace_cnt
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
        model_for_imitation = f"-ncem {script_model_file_path}" if HP.IMITATE and ctx.loop == 1 else ""

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
        ctx.trace_index.add_prob_trace(ctx.loop,prob,trace_file_path)
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
        ctx.trace_index.report_bad_trace(ctx.loop,prob)

      workers_freed = 1
      if lrs_trace_file and os.path.isfile(lrs_trace_file):
        os.remove(lrs_trace_file)
    else:
      assert False, f"Surprised by job_kind {job_kind}"

    return workers_freed

  perform_and_gather_in_parallel(get_perform_tasks(),process_results_from_perform_and_gather)

  torch.save(stats,os.path.join(ctx.cur_dir,"stats.pt"))

  # let's report what happened so far (and save the results into files, for later analysis):
  for (res_filename,mission) in result_metas:
    results = result_dicts[res_filename]
    torch.save(("Unused",results), os.path.join(ctx.cur_dir,res_filename))

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
      what_ran = ""
      if HP.USE_SPECIAL:
        what_ran = HP.PERFORMS_SPECIAL[i] if i < len(HP.PERFORMS_SPECIAL) else None

      print("   {:>3} {:>6} {:6.4f} = {:>5} / {:>5} {:>5}{}".format(
              i,ilims[i],len(by_performs_solved[i])/len(by_performs_attempted[i]),len(by_performs_solved[i]),len(by_performs_attempted[i]),adds[i],what_ran))

  print()
  print("  Stage 1 took",time.time()-stage_start_time)
  print()
  sys.stdout.flush()

  ctx.trace_index.update_scores(ctx.loop)
  ctx.trace_index.report()
  save_trace_index(ctx.cur_dir,ctx.trace_index)

  print()
  sys.stdout.flush()

# ============================================================================================

# ============================================================================================

def stage_eval_train_eval(ctx,trace_problems,with_early_stopping):
  # newly only lives one iter, so no need to save it
  optimizer = torch.optim.Adam(ctx.model.parameters(),
      lr=lr_wish, weight_decay=HP.WEIGHT_DECAY)

  TIW = HP.TEST_IMPROVE_WINDOW if with_early_stopping else HP.NUM_ALL_STEPS
  assert TIW > 0
  eval_models = [None]*TIW
  eval_criters = [None]*TIW
  stage2iter = 0

  if with_early_stopping: # we will need to single out the validation traces!
    rng.shuffle(trace_problems)
    # an 80:20 split
    cut_idx = int(0.8*len(trace_problems))
    train_trace_problems = trace_problems[:cut_idx]
    valid_trace_problems = trace_problems[cut_idx:]
  else:
    train_trace_problems = trace_problems
    valid_trace_problems = None

  while True:
    if with_early_stopping:
      # EVAL on validation problems
      eval_model_file_path = os.path.join(HP.SCRATCH,"eval-model-state_{}_{}.tar".format(os.getpid(),stage2iter))
      torch.save(ctx.model.state_dict(), eval_model_file_path)
      if eval_models[stage2iter % TIW] is not None:
        os.remove(eval_models[stage2iter % TIW])
      eval_models[stage2iter % TIW] = eval_model_file_path

      def get_eval_tasks():
        prob_records = W.create_prob_records(valid_trace_problems,ctx.trace_index)
        prob_records.sort(key = lambda rec : -rec.szs) # descending by the filesizes (i.e., the big ones first)
        for record in prob_records:
          yield (W.JK_EVAL,(record,eval_model_file_path))

      weighted_eval_stats = defaultdict(float)
      def process_results_from_eval(job_kind,input,result):
        nonlocal weighted_eval_stats

        assert job_kind == W.JK_EVAL
        stat_dict = result
        for k,v in stat_dict.items(): # includes the loss; all multiplied by fact already in the child
          weighted_eval_stats[k] += v
        return 1

      pre_eval = time.time()
      eval_and_train_in_parallel(get_eval_tasks(),process_results_from_eval)
      weighted_eval_criter = weighted_eval_stats[HP.EARLY_STOP_ON]
      print(f"Eval {HP.EARLY_STOP_ON} {weighted_eval_criter} in",int(time.time()-pre_eval),"s")
      for k,v in weighted_eval_stats.items():
        print(f"    e_{k}",v)
      sys.stdout.flush()

      eval_criters[stage2iter % TIW] = weighted_eval_criter

    stage2iter += 1

    if with_early_stopping:
      if stage2iter >= TIW: # we have written everywhere (no None there anymore)
        oldest_idx = stage2iter % TIW
        oldest_val = eval_criters[oldest_idx]
        if all((el >= oldest_val for el in eval_criters)):
          print(f"Eval {HP.EARLY_STOP_ON} didn't improve for",TIW-1,"iterations now")
          if stage2iter == TIW:
            if HP.ANYWAY_STEP_ALL:
              actual_idx = TIW-1
              print("Actually, it never improved! Will apply ALL training steps anyway!")
            else:
              actual_idx = 1
              print("Actually, it never improved! Will apply one training step anyway!")
            ctx.model.load_state_dict(torch.load(eval_models[actual_idx]))
          else:
            actual_idx = oldest_idx
            ctx.model.load_state_dict(torch.load(eval_models[oldest_idx]))
          print(f"  took model with eval {HP.EARLY_STOP_ON}",eval_criters[actual_idx])

          for eval_model_file_path in eval_models:
            os.remove(eval_model_file_path)
          break
        ITER_LIMIT = HP.MAX_TEST_IMPROVE_FIRST_ITER if (iter == 1) else HP.MAX_TEST_IMPROVE_ITER
        if stage2iter > ITER_LIMIT:
          print("Taking too long to converge (stage2iter > HP.MAX_TEST_IMPROVE_ITER), will take the best from the last HP.TEST_IMPROVE_WINDOW observed.")
          best_idx = 0
          best_idx_val = eval_criters[0]
          for i,ecrit in enumerate(eval_criters):
            if ecrit < best_idx_val:
              best_idx_val = ecrit
              best_idx = i
          ctx.model.load_state_dict(torch.load(eval_models[best_idx]))
          for eval_model_file_path in eval_models:
            os.remove(eval_model_file_path)
          break

    # TRAIN on train problems
    train_model_version = 0
    def get_train_tasks():
      prob_records = W.create_prob_records(train_trace_problems,ctx.trace_index)
      rng.shuffle(prob_records)
      nonlocal train_model_version
      for record in prob_records:
        train_model_version += 1
        train_model_file_path = os.path.join(HP.SCRATCH,"train-model-state_{}_{}.tar".format(os.getpid(),train_model_version))
        torch.save(ctx.model.state_dict(), train_model_file_path)
        yield (W.JK_TRAIN,(record,train_model_file_path))

    weighted_train_loss = 0.0
    weighted_train_stats = defaultdict(float)
    minibatch_count = 0
    def process_results_from_train(job_kind,input,result):
      nonlocal weighted_train_loss
      nonlocal weighted_train_stats
      nonlocal minibatch_count

      assert job_kind == W.JK_TRAIN
      (record,train_model_file_path) = input
      loss, stat_dict = result

      weighted_train_loss += loss # (= the loss) multiplied by fact already in the child
      for k,v in stat_dict.items(): # includes the loss; all multiplied by fact already in the child
          weighted_train_stats[k] += v

      # train_model_version = int(train_model_file_path.split("_")[-1][:-4])
      # print(f"    BACK: {train_model_version:4d} after {took}s")

      # gnn_norm, gage_norm, gweight_norm, cleval_norm = 0.0, 0.0, 0.0, 0.0

      # copy from result parameters to our model's gradients
      grad_loader_temp.load_state_dict(torch.load(train_model_file_path))

      # copy_grads_back_from_param
      for (name, param), param_copy in zip(ctx.model.named_parameters(),grad_loader_temp.parameters()):
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
        if param.grad is None:
          param.grad = param_copy.clone()
        else:
          param.grad.add_(param_copy)

      minibatch_count += 1
      if minibatch_count >= HP.TRAIN_MINIBATCH_SIZE:
        optimizer.step()
        optimizer.zero_grad()
        minibatch_count = 0
      os.remove(train_model_file_path)
      return 1

    pre_train = time.time()
    eval_and_train_in_parallel(get_train_tasks(),process_results_from_train)

    if minibatch_count > 0:
      optimizer.step()
      optimizer.zero_grad()

    print("Train loss",weighted_train_loss,"in",int(time.time()-pre_train),"s")
    for i,(k,v) in enumerate(weighted_train_stats.items()):
      print(f"    t_{k}",v)
      if i == 2: # just some hardcoded pretty-printing
        print()
    print()
    sys.stdout.flush()

    if not with_early_stopping and stage2iter >= TIW:
      break

  ctx.save_loop_model()

  # stage 2
  print()
  print(f"  Stage 2a (with_early_stopping {with_early_stopping}) - took {time.time()-stage_start_time} seconds and {stage2iter-0.5} eval/train iterations")
  print()
  sys.stdout.flush()



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

  torch.manual_seed(HP.RANDOM_SEED)

  ctx = Context()
  ctx.init_train_test(folder_with_prev_exper)

  skip_first_stage = False

  if len(sys.argv) > 5: # we already know the folder, but which loop to copy from there?
    assert folder_with_prev_exper is not None

    ctx.loop = int(sys.argv[5])
    load_dir = os.path.join(folder_with_prev_exper,f"loop{ctx.loop}")

    load_model = True
    load_model_old = False
    load_traces_new = False
    load_traces_old = False
    steal_script_model = False

    if len(sys.argv) > 6:
      load_model = "m" in sys.argv[6]
      load_model_old = "M" in sys.argv[6]
      load_traces_new = "t" in sys.argv[6]
      load_traces_old = "T" in sys.argv[6]

    if load_model:
      ctx.load_loop_model(load_dir)

    if load_model_old:
      ctx.load_loop_model_old(load_dir)

    if load_traces_new:
      ctx.trace_index = load_trace_index(os.path.join(folder_with_prev_exper,f"loop{ctx.loop+1}"))
      skip_first_stage = True
      print("Starting from loop",ctx.loop,"and a half")
      ctx.trace_index.consolidate()
      ctx.trace_index.report()

    if load_traces_old:
      ctx.trace_index = load_trace_index(os.path.join(folder_with_prev_exper,f"loop{ctx.loop}"))
      print("Starting from loop",ctx.loop)
      ctx.trace_index.consolidate()
      ctx.trace_index.report()


  else:
    loop_str = ctx.claim_loop_dir()
    print(loop_str)
    sys.stdout.flush()

    ctx.save_loop_model()

  assert loop_count > 0

  # temporary model used for the gradient trick
  grad_loader_temp = IC.get_initial_model()

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
    loop_count -= 1
    iter += 1

    loop_start_time = time.time()

    # for this iteration, we write stuff here:
    ctx.loop += 1
    loop_str = ctx.claim_loop_dir()
    print(loop_str)
    sys.stdout.flush()

    if not skip_first_stage:
      if HP.SNAKE_STYLE_GATHER:
        stage_snake_gather(ctx)
      else:
        stage_perf_gather(ctx)

    skip_first_stage = False # only possibly skipped for the first loop

    # ===========================================================================

    # don't run the full last loop - otherwise, we are training a model nobody will see evaluated
    if loop_count == 0 and not HP.SNAKE_STYLE_GATHER:
      break

    print()
    sys.stdout.flush()
    stage_start_time = time.time()

    # compute LR for our loop, taking into account our decay
    lr_wish = HP.LEARNING_RATE * (HP.LEARNING_RATE_DECAY ** (ctx.loop-1))
    print("Learning rate now at",lr_wish)
    with_early_stopping = ctx.loop <= HP.NUM_EARLY_STOP_ROUNDS
    print("with_early_stopping",with_early_stopping)
    print()
    sys.stdout.flush()

    trace_problems = list(ctx.trace_index.cur_problems())

    stage_eval_train_eval(ctx,trace_problems,with_early_stopping=with_early_stopping)

    print()
    sys.stdout.flush()

    print("Loop took",time.time()-loop_start_time)
    print()
    sys.stdout.flush()

    # so that we can distinguish standard run from a "loop and a half" start (which reads a ready trace index and then skips Stage 1)
    ctx.trace_index.loop_finished()
