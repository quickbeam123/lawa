#!/usr/bin/env python3

import sys, os, torch, random

import hyperparams as HP
import inf_common as IC
import workers as W

def ilim2tlim(ilim):
  secs = max(5,ilim // 1000) # it's 2 times more than the instrlimit on a 2GHz machine
  return secs

if __name__ == "__main__":
  # Given a problem file and a script model path, this runs vampire on the given problem
  # (under the model's neural guidance) as elooper would, trying different random seeds,
  # until the run succeeds and then it gathers a traces for the run, and saves it to a file.
  #
  # ./trace_finder.py Problems/SEU/SEU076+1.p ~/jar2026/seed42_nd/loop2/script-model.pt traces/Problems_SEU_SEU076+1.L2.pt

  prob = sys.argv[1]
  script_model_file_path = sys.argv[2]
  output_trace_path = sys.argv[3]

  ilim = HP.INSTRUCTION_LIMIT

  # Perform phase: try random seeds until Vampire solves the problem
  i = 0
  while True:
    i += 1
    seed = random.randint(1,0x7fffff)
    print(f"Attempt {i}, seed {seed} ...",end=" ",flush=True)

    opts1 = f"-t {ilim2tlim(ilim)} -i {ilim} -p off"

    if HP.SATURATION_ALGORITHM.startswith("lrs"):
      lrs_trace_file = os.path.join(HP.SCRATCH,"{}_{}_{}_{}.lrs".format(prob.replace("/","_"),i,seed,os.getpid()))
      opts1 += f" -lstf {lrs_trace_file}"
    else:
      lrs_trace_file = ""

    opts2 = f" {HP.SHUFFLING_OPTIONS} -sa {HP.SATURATION_ALGORITHM} -ncf {HP.NUM_CLAUSE_FEATURES} -npf {HP.NUM_PROBLEM_FEATURES} -npcc on -ncem {script_model_file_path} --random_seed {seed}"

    print(prob,opts1+opts2)

    result = W.vampire_perfrom(prob,opts1+opts2,None)
    print(f"status={result.status}, instructions={result.instructions}, activations={result.activations}")

    if result.status == "uns":
      print(f"Solved on attempt {i}!")
      break
    else:
      if lrs_trace_file and os.path.isfile(lrs_trace_file):
        os.remove(lrs_trace_file)

  # Gather phase: rerun with trace recording
  gather_ilim = ilim * 10
  lrs_trace_str = f" -lltf {lrs_trace_file}" if lrs_trace_file else ""

  gather_opts = f"-t {ilim2tlim(gather_ilim)} -i {gather_ilim} -nar {output_trace_path} {lrs_trace_str}" + opts2

  print(f"Gathering trace ...",flush=True)
  gather_result = W.vampire_perfrom(prob,gather_opts,None)

  if lrs_trace_file and os.path.isfile(lrs_trace_file):
    os.remove(lrs_trace_file)

  if gather_result.status != "uns":
    print(f"ERROR: Gather failed to reproduce proof! status={gather_result.status}")
    sys.exit(1)

  # Process trace and print statistics
  num_good_selections, passes_limits, (gage_h,gage_w), (gweight_h,gweight_w), num_selections = IC.trace_good_for_learning(output_trace_path,sys.stdout)

  kbSize = os.path.getsize(output_trace_path)//1024

  print(f"Trace statistics:")
  print(f"  num_selections:      {num_selections}")
  print(f"  num_good_selections: {num_good_selections}")
  print(f"  gage height/width:   {gage_h}/{gage_w}")
  print(f"  gweight height/width:{gweight_h}/{gweight_w}")
  print(f"  passes_limits:       {passes_limits}")
  print(f"  trace file size:     {kbSize} KB")

  if num_good_selections > 0 and passes_limits:
    print(f"Trace saved to {output_trace_path}")
  else:
    print(f"WARNING: Trace is not good for learning (trivial or exceeds limits)")
    os.remove(output_trace_path)
    sys.exit(1)
