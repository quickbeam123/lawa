#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import os, sys, shutil, atexit, time, pickle, math, random
from collections import defaultdict
from collections import deque
from itertools import chain

import multiprocessing
import subprocess
import numpy
from dataclasses import dataclass

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


@dataclass
class VampResult:
  status: str
  instructions: int
  activations: int
  nn_warmup_start: int
  nn_warmup: int
  nn_gnn_start: int
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
    nn_warmup_start = 0
    nn_warmup = 0
    nn_gnn_start = 0
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

        if line.startswith("% Neural warmup startat"):
          nn_warmup_start = int(line.split()[-1])
        if line.startswith("% Neural model warmup"):
          nn_warmup = int(line.split()[-1])

        if line.startswith("% Gnn startat"):
          nn_gnn_start = int(line.split()[-1])
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
  return VampResult(status,instructions,activations,
                    nn_warmup_start,nn_warmup,
                    nn_gnn_start,nn_gnn,nn_bulks,strategy)


@dataclass
class ProbRecord:
  szs: int
  prob: str
  norm_factor: float   # used to normalize statistics (basically, compute the average on the fly)
  scale_factor: float  # used to act on the loss (e.g. easy problems pull less)
  prob_traces: list[str]

def create_prob_records(trace_problems,trace_index):
  norm_fact = 1/len(trace_problems)
  prob_records = []
  for prob in trace_problems:
    prob_traces = trace_index.prob_traces(prob)
    scale_fact = trace_index.prob_factor(prob) if HP.CUMULATIVE_LOSS_SCALING else 1.0
    prob_trace_size_sum = sum(os.path.getsize(trace_file) for trace_file in prob_traces)
    prob_records.append(ProbRecord(prob_trace_size_sum,prob,norm_fact,scale_fact,prob_traces))
  return prob_records

train_log = None

# Kinds of jobs a worker can be asked to do
JK_PERFORM = 0 # runs vampire in "real-time" mode to assess its performance
  # input:     (res_filename,gatherwish,mission,prob,opts1,opts2,perf_log)
  # output:    result as coming from IC.vampire_eval
JK_GATHER = 1  # runs vampire in "show passive traffic" to gather a training trace
  # input:     (mission,prob,counter,opts,eval_opts - just for reporting purposes,gather_log)
  # output:    filename where got saved if got a non-degenerate trace; or None
JK_EVAL = 2  # construct our network to get the loss of this trace (no training to do)
  # input:     (record: ProbRecord,model_file_path)
  # output:    stat_dict
JK_TRAIN = 3   # construct our network to get the loss of this trace and do one training step
  # input:     (record: ProbRecord,train_model_file_path)
  # output:    stat_dict

def job_perform(input):
  (res_filename,gatherwish,mission,prob,i,ilim,lrs_trace_file,opts1,opts2,perf_log) = input
  return vampire_perfrom(prob,opts1+opts2,perf_log)

def job_gather(input):
  (mission,prob,lrs_trace_file,trace_file_path,opts,eval_opts,gather_log) = input
  vamp_res = vampire_perfrom(prob,opts,gather_log)

  if vamp_res.status == "uns":
    #assert vamp_res.status == "uns", f"Ran {(prob,opts)} got {vamp_res} eval_opts were {eval_opts}"
    assert os.path.isfile(trace_file_path), f"Ran {(prob,opts)} got {vamp_res} eval_opts were {eval_opts}"

    return IC.trace_good_for_learning(trace_file_path,train_log)
  else:
    print("Failed to reproduce success for",prob,opts)
    train_log.write(f"Failed to reproduce success for {prob} {opts}\n")
    return 0, False, 0, 0, 0

def eval_one_trace(trace_file_path,local_model,local_fact,stat_dict,prob):
  trace_tuple = torch.load(trace_file_path)
  learn_model = IC.LearningModel(False,local_model,trace_tuple)
  learn_model.eval()

  with torch.no_grad():
    loss,selection_hit_rate,dist_to_good = learn_model.forward()

  if loss.item() > HP.WORTH_REPORTING_LOSS:
    train_log.write(f"EVAL of {prob} found loss {loss.item()}; trace_file_path {trace_file_path}, local_fact {local_fact}\n")

  stat_dict["loss"] += local_fact*loss.item()
  stat_dict["selection_hit_rate"] += local_fact*selection_hit_rate
  stat_dict["dist_to_good"] += local_fact*dist_to_good

def job_eval(input):
  (record,model_file_path) = input

  eval_begin = time.time()

  local_model = IC.get_initial_model()
  local_model.load_state_dict(torch.load(model_file_path))

  local_fact = 1.0/len(record.prob_traces)

  stat_dict = defaultdict(float)

  for trace_file_path in record.prob_traces:
    try:
      eval_one_trace(trace_file_path,local_model,local_fact,stat_dict,record.prob)
    except Exception as e:
      with open(f"exception{os.getpid()}.log", "w") as f:
        f.write(f"{type(e).__name__}: {e} occurred in EVAL\n")
        f.write(f"(prob {record.prob}, fact {record.norm_factor}*{local_fact}, trace_file_path {trace_file_path}, model_file_path {model_file_path})")
      raise

  took = time.time()-eval_begin
  if took > HP.WORTH_REPORTING:
    train_log.write(f"EVAL of {record} took {took}\n")

  for k in stat_dict.keys():
    stat_dict[k] *= record.norm_factor

  return stat_dict

def train_one_trace(trace_file_path,local_model,local_fact,scale_fact,stat_dict,prob):
  trace_tuple = torch.load(trace_file_path)
  learn_model = IC.LearningModel(False,local_model,trace_tuple)
  learn_model.train()

  loss,selection_hit_rate,dist_to_good = learn_model.forward()

  if loss.item() > HP.WORTH_REPORTING_LOSS:
    train_log.write(f"TRAIN of {prob} found loss {loss.item()}; trace_file_path {trace_file_path}, local_fact {local_fact}, scale_fact {scale_fact}\n")

  stat_dict["loss"] += local_fact*loss.item()
  stat_dict["selection_hit_rate"] += local_fact*selection_hit_rate
  stat_dict["dist_to_good"] += local_fact*dist_to_good

  scaled_loss = scale_fact*local_fact*loss
  scaled_loss.backward() # is OK to call once per trace, it anyway accummulates
  return scaled_loss.item() # just the float goes back for statistics

def job_train(input):
  (record,train_model_file_path) = input

  train_begin = time.time()

  local_model = IC.get_initial_model()
  local_model.load_state_dict(torch.load(train_model_file_path))

  local_fact = 1.0/len(record.prob_traces)

  # print("TRAIN on",prob,fact,trace_file_paths)

  stat_dict = defaultdict(float)

  loss = 0.0

  for trace_file_path in record.prob_traces:
    try:
      loss += train_one_trace(trace_file_path,local_model,local_fact,record.scale_factor,stat_dict,record.prob)
    except Exception as e:
      with open(f"exception{os.getpid()}.log", "w") as f:
        f.write(f"{type(e).__name__}: {e} occurred in TRAIN\n")
        f.write(f"(prob {record.prob}, fact {record.norm_factor}*{local_fact}*{record.scale_factor}, trace_file_path {trace_file_path}, model_file_path {train_model_file_path})")
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
    train_log.write(f"TRAIN of {record} took {took}\n")

  for k in stat_dict.keys():
    stat_dict[k] *= record.norm_factor

  return record.norm_factor*loss, stat_dict

JOB_DISPATCH = {JK_PERFORM : job_perform,
                JK_GATHER: job_gather,
                JK_EVAL: job_eval,
                JK_TRAIN: job_train}

def worker(q_in, q_out, worker_id):
  # tell each worker we don't want any extra threads
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)

  random.seed(HP.RANDOM_SEED + worker_id)
  torch.manual_seed(HP.RANDOM_SEED + worker_id)

  while True:
    (job_kind,input) = q_in.get()
    q_out.put((job_kind,input,JOB_DISPATCH[job_kind](input)))
    # gc.collect()  # Force garbage collection

my_processes = []

def create_workforce(numworkers):
  queue_in = multiprocessing.Queue()
  queue_out = multiprocessing.Queue()
  for i in range(numworkers):
    p = multiprocessing.Process(target=worker, args=(queue_in,queue_out,i))
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
