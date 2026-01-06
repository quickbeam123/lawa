#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, random, atexit, time, pickle, math
from collections import defaultdict
from collections import deque
from itertools import chain

import multiprocessing
import subprocess
import numpy as np
from dataclasses import dataclass

import random

# first environ, then load torch, also later we set_num_treads (in "main")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)


@dataclass
class VampResult:
  status: str
  instructions: int
  activations: int
  nn_warmup: int
  nn_gnn: int
  nn_bulks: int
  strategy: str

def vampire_perfrom(prob,opts,log):
  if True:
    f = open(log,"w") if log else None
  else:
    f = sys.stdout

  to_run = " ".join(["./run_lawa_vampire.sh",HP.VAMPIRE_EXECUTABLE,opts,prob])
  if f:
    print(to_run,file=f)

  try:
    last_line = None

    status = None
    instructions = 0
    activations = 0
    nn_warmup = 0
    nn_gnn = 0
    nn_bulks = 0
    strategy = None

    # sometimes, we get a mangled output that messes up with a decoder inside getoutput
    # "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0"
    output = subprocess.getoutput(to_run)

    for line in output.split("\n"):
      last_line = line
      if f:
        print(line,file=f)
      if line.startswith("%"):
        if line.startswith("% Random strategy:"):
          strategy = line.split()[-1]
        if line.startswith("% Activations started"):
          activations = int(line.split()[-1])
        if line.startswith("% Instructions burned:"):
          instructions = int(line.split()[-2])
        if line.startswith("% Neural model warmup"):
          nn_warmup = int(line.split()[-1])
        if line.startswith("% Gnn eval"):
          nn_gnn = int(line.split()[-1])
        if line.startswith("% Bulk evals"):
          nn_bulks = int(line.split()[-1])

        if line.startswith("% SZS status"):
          if "Satisfiable" in line or "CounterSatisfiable" in line:
            status = "sat"
          elif "Theorem" in line or "Unsatisfiable" in line or "ContradictoryAxioms" in line:
            status = "uns"
  except Exception as e:
    print("Error reading vampire output on line",last_line)
    print("Exception type:", type(e).__name__)
    print("Exception message:", str(e))

  if f and f != sys.stdout:
    f.close()
  # print(status,instructions,activations)
  return VampResult(status,instructions,activations,nn_warmup,nn_gnn,nn_bulks,strategy)


@dataclass
class ProbRecord:
  szs: int
  prob: str
  prob_fact: float
  prob_traces: list[str]

def create_prob_records(trace_problems,trace_index):
  fact = 1/len(trace_problems)
  prob_records = []
  for prob in trace_problems:
    prob_traces = trace_index.prob_traces(prob)
    # prob_fact = trace_index.prob_factor(prob)*fact # TODO: careful, if even if we do CUMUL, this will not affect the loss!
    prob_fact = fact
    prob_trace_size_sum = sum(os.path.getsize(trace_file) for trace_file in prob_traces)
    prob_records.append(ProbRecord(prob_trace_size_sum,prob,prob_fact,prob_traces))
  return prob_records

train_log = None

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,gatherwish,mission,prob,opts1,opts2,perf_log)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,counter,opts,eval_opts - just for reporting purposes,gather_log)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL_TWEAK_MATRIX = 2  # construct our network to get the loss of this trace (no training to do);
  # if tweak_file_path specified, go load the tweak from there an also find a good tweak for the given record by local gradient descent; extend the stat_dict further and save the resulting tweak back to the file
  # input:     (record: ProbRecord,model_file_path,tweak_file_path or None,compute_matrix: bool)
  # output:    stat_dict
  #   contains the computed loss - for the
  #   further ML statistics - for each trace separately, a) for the initial tweak, b) for the final tweak
  #        - selection_hit_rate: percentage of time moments, in which a proof clause has minimial logit
  #        - dist_to_good: average (over time moments) distance to a future proof clauase (in the logit-sorted order)
  #        (selection_hit_rate == 1.0 => dist_to_good == 0.0)
JK_TRAIN = 3   # construct our network to get the loss of this trace and do one training step
  # input:     (record: ProbRecord,train_model_file_path,epsilon)
  # output:    stat_dict

def job_perform(input):
  (res_filename,gatherwish,mission,prob,i,ilim,lrs_trace_file,opts1,opts2,perf_log) = input
  return vampire_perfrom(prob,opts1+opts2,perf_log)

def job_gather(input):
  (mission,prob,lrs_trace_file,trace_file_path,opts,eval_opts,gather_log) = input
  vamp_res = vampire_perfrom(prob,opts,gather_log)

  if vamp_res.status == "uns":
    #assert vamp_res.status == "uns", f"Ran {(prob,opts)} got {vamp_res} eval_opts were {eval_opts}"
    assert os.path.isfile(trace_file_path)

    return IC.trace_good_for_learning(trace_file_path,train_log)
  else:
    print("Failed to reproduce success for",prob,opts)
    return False, False, 0, 0

def look_for_a_tweak(learn_model,just_before_final,num2idx,tweak_in):
  local_optimizer = torch.optim.Adam([tweak_in], lr=HP.TWEAKS_LEARNING_RATE)

  last_loss = float('inf')

  timed_out = 0.0
  to_perfection = 0.0

  numiter = 0
  start_time = time.time()
  while True:
    numiter += 1

    local_optimizer.zero_grad()
    losses,selection_hit_rates,dists_to_good = learn_model.forward(just_before_final,num2idx,
                                                                   tweak_in.unsqueeze(0))

    loss = losses[0]
    loss.backward()
    local_optimizer.step()

    now_loss = loss.item()
    if now_loss > last_loss:
      break

    last_loss = now_loss
    last_shr = selection_hit_rates[0]
    last_dtg = dists_to_good[0]
    last_norm = torch.norm(tweak_in).item()
    tweak_out = tweak_in.clone()

    if dists_to_good[0] < 0.000001:
      to_perfection = 1.0
      break

    telapsed = time.time() - start_time
    if telapsed > HP.TWEAK_SEARCH_MAX_TIME: # TODO: make a HP
      timed_out = 1.0
      break

  return tweak_out, {"tweaked_loss": last_loss,
          "tweaked_selection_hit_rate": last_shr,
          "tweaked_dist_to_good": last_dtg,
          "tweaks_norm": last_norm,
          "tweakings_timed_out": timed_out,
          "tweakings_to_perfection": to_perfection,
          "tweakings_numiter": numiter}


def job_eval_tweak_matrix(input):
  (record,model_file_path,tweak_file_path,compute_matrix) = input

  eval_begin = time.time()

  local_model = IC.get_initial_model()
  local_model.load_state_dict(torch.load(model_file_path))

  if tweak_file_path is not None and compute_matrix:
    active_tweak_selection = torch.load(tweak_file_path)
    active_tweak_selection = active_tweak_selection

  # print("EVAL on",prob,fact,trace_file_paths)

  local_fact = 1/len(record.prob_traces)

  stat_dict = defaultdict(float)

  for trace_file_path in record.prob_traces:
    try:
      trace_tuple = torch.load(trace_file_path)
      learn_model = IC.LearningModel(False,local_model,trace_tuple)
      learn_model.eval()

      with torch.no_grad():
        just_before_final,num2idx = learn_model.pre_forward()

      if compute_matrix:
        with torch.no_grad():
          assert len(record.prob_traces) == 1
          # TODO: should I tell learn_model I don't need the metrics?
          losses,selection_hit_rates,dist_to_goods = learn_model.forward(just_before_final,num2idx,torch.stack(active_tweak_selection))
      else:
        with torch.no_grad():
          notweaks = IC.get_neutral_tweak(local_model.clause_valuator_snd, detached = False).unsqueeze(0)
          losses,selection_hit_rates,dist_to_goods = learn_model.forward(just_before_final,num2idx,notweaks)

        stat_dict["loss"] += local_fact*losses[0].item()
        stat_dict["selection_hit_rate"] += local_fact*selection_hit_rates[0]
        stat_dict["dist_to_good"] += local_fact*dist_to_goods[0]

        # CAREFUL: this gets a bit weird if there is more than one trace for a problem
        if tweak_file_path is not None and not compute_matrix:
          tweak_in = torch.load(tweak_file_path)
          tweak_out, tweaked_stats = look_for_a_tweak(learn_model,just_before_final,num2idx,tweak_in)
          for k,v in tweaked_stats.items():
            stat_dict[k] += local_fact*v
          torch.save(tweak_out, tweak_file_path)

    except Exception as e:
      with open(f"exception{os.getpid()}.log", "w") as f:
        f.write(f"{e} occurred in EVAL/TWEAK\n")
        f.write(f"(prob {record.prob}, fact {record.prob_fact}*{local_fact}, trace_file_path {trace_file_path}, model_file_path {model_file_path}, tweak_file_path {tweak_file_path})")
      raise

  took = time.time()-eval_begin
  if took > HP.WORTH_REPORTING:
    train_log.write(f"EVAL/TWEAK of {record} took {took}\n")

  if compute_matrix:
    return [loss.item() for loss in losses], selection_hit_rates, dist_to_goods, took
  else:
    # print("EVAL on",prob,fact,trace_file_paths,loss.item())
    for k in stat_dict.keys():
      stat_dict[k] *= record.prob_fact

    return stat_dict

def random_argmin(a):
    a = np.asarray(a)
    mins = np.flatnonzero(a == a.min())
    return random.choice(mins),len(mins)

def job_train(input):
  (record,train_model_file_path,epsilon) = input

  train_begin = time.time()

  local_model = IC.get_initial_model()
  local_model.load_state_dict(torch.load(train_model_file_path))

  verbose = False # (prob in {'Problems/COM/COM021+4.p'})

  local_fact = 1/len(record.prob_traces)

  # print("TRAIN on",prob,fact,trace_file_paths)

  stat_dict = defaultdict(float)

  loss = torch.tensor(0.0, requires_grad=True)

  for trace_file_path in record.prob_traces:
    try:
      trace_tuple = torch.load(trace_file_path)
      learn_model = IC.LearningModel(verbose,local_model,trace_tuple)
      learn_model.train()

      just_before_final,num2idx = learn_model.pre_forward()

      if epsilon is not None:
        """
        mytweak = local_model.tweaky
        notweak = IC.get_neutral_tweak(local_model.clause_valuator_snd, detached = False)
        both_tweaks = torch.stack([notweak,mytweak])
        """

        tweaks = [IC.get_neutral_tweak(local_model.clause_valuator_snd, detached = False)] + [tw for tw in local_model.tweaks]

        losses,selection_hit_rates,dists_to_good = learn_model.forward(just_before_final, num2idx, torch.stack(tweaks))

        winner,freedom = random_argmin(dists_to_good)
        stat_dict["b_loss"] += local_fact*losses[winner].item()
        stat_dict["b_selection_hit_rate"] += local_fact*selection_hit_rates[winner]
        stat_dict["b_dist_to_good"] += local_fact*dists_to_good[winner]

        loser = np.argmax(dists_to_good)
        stat_dict["c_loss"] += local_fact*losses[loser].item()
        stat_dict["c_selection_hit_rate"] += local_fact*selection_hit_rates[loser]
        stat_dict["c_dist_to_good"] += local_fact*dists_to_good[loser]

        stat_dict[f"z_freedom[{freedom}]"] += local_fact

        if random.uniform(0.0, 1.0) < epsilon:
          winner = random.randint(0,len(tweaks)-1)
          stat_dict["tweaked randomly"] += local_fact
        else:
          stat_dict[f"tweaked winner {winner}"] += local_fact

        cosines = torch.tensor(0.0, requires_grad=True)
        for j in range(len(tweaks)):
          if j != winner:
            cosines = cosines + torch.nn.functional.cosine_similarity(tweaks[winner], tweaks[j].detach(), dim=0)
        cosines = cosines / (len(tweaks) - 1)
        stat_dict["cosines"] += local_fact*cosines.item()

        loss = loss + local_fact*losses[winner] + HP.TWEAKS_COSINE_LOSS_FACTOR*local_fact*cosines

        # the generalist's stats
        stat_dict["loss"] += local_fact*losses[winner].item()
        stat_dict["selection_hit_rate"] += local_fact*selection_hit_rates[winner]
        stat_dict["dist_to_good"] += local_fact*dists_to_good[winner]
        # the tweaked stats
        """
        stat_dict["tweaked_loss"] += local_fact*losses[1].item()
        stat_dict["tweaked_selection_hit_rate"] += local_fact*selection_hit_rates[1]
        stat_dict["tweaked_dist_to_good"] += local_fact*dists_to_good[1]
        """

      else:
        notweaks = IC.get_neutral_tweak(local_model.clause_valuator_snd, detached = False).unsqueeze(0)
        losses,selection_hit_rates,dist_to_goods = learn_model.forward(just_before_final, num2idx, notweaks)

        loss = loss + local_fact*losses[0]

        # the generalist's stats
        stat_dict["loss"] += local_fact*losses[0].item()
        stat_dict["selection_hit_rate"] += local_fact*selection_hit_rates[0]
        stat_dict["dist_to_good"] += local_fact*dist_to_goods[0]

    except Exception as e:
      with open(f"exception{os.getpid()}.log", "w") as f:
        f.write(f"{e} occurred in TRAIN\n")
        f.write(f"(prob {record.prob}, fact {record.prob_fact}*{local_fact}, trace_file_path {trace_file_path}, model_file_path {train_model_file_path})")
      raise

  loss.backward()

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
    train_log.write(f"TRAIN of {record} took {took}\n")

  for k in stat_dict.keys():
    stat_dict[k] *= record.prob_fact

  return record.prob_fact*loss.item(), stat_dict



JOB_DISPATCH = {JK_PERFORM : job_perform,
                JK_GATHER: job_gather,
                JK_EVAL_TWEAK_MATRIX: job_eval_tweak_matrix,
                JK_TRAIN: job_train}

def worker(q_in, q_out):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  while True:
    (job_kind,input) = q_in.get()
    q_out.put((job_kind,input,JOB_DISPATCH[job_kind](input)))
    # gc.collect()  # Force garbage collection

my_processes = []

def create_workforce(numworkers):
  queue_in = multiprocessing.Queue()
  queue_out = multiprocessing.Queue()
  for i in range(numworkers):
    p = multiprocessing.Process(target=worker, args=(queue_in,queue_out))
    p.start()
    my_processes.append(p)
  return (queue_in,queue_out)

def cleanup():
  for p in my_processes:
    p.kill()
  train_log.close()

def workforces_finished():
  atexit.register(cleanup)

def do_in_parallel(queues,tasks,max_parallelism,process_results_callback):
  q_in,q_out = queues
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
