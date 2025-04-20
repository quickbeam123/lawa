#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time, math

from collections import defaultdict

def get_info(probinfo,prob):
  prob0 = prob.split("/")[-1]
  if prob0 not in probinfo:
    return None
  return probinfo[prob0]

if __name__ == "__main__":
  # Scatter plot of two result files (can be the same exper or two different ones)
  # - each (successfully solved) problem's coordinates are based on the number of activations before contradition found
  # - also we try to check whether any problems of good TPTP rating were solved
  #
  # call as in, e.g.: ./scatter.py ~/mtpa-gnn/exper100imit/loop1/train_res.pt ~/mtpa-gnn/exper100imit/loop34/train_res.pt

  with open("/nfs/sudamar2/TPTP-v9.0.0/probinfo9.0.0.pkl",'rb') as f:
    probinfo = pickle.load(f)

  (_meta1,results1) = torch.load(sys.argv[1])
  if False:
    for prob, res_list in results1.items():
      for res in res_list:
        vr = res[1]
        if vr.status == "uns":
          rate = get_info(probinfo,prob)[0]
          if rate > 0.97:
            print(prob,rate,vr)

    exit(0)
  (_meta2,results2) = torch.load(sys.argv[2])

  Xs = []
  Ys = []

  Vs = []
  Ws = []

  only_first = []
  only_second = []
  all_problems = set(results1.keys()) | set(results2.keys())

  total_common = 0
  vr1_act_log_sum = 0.0
  vr2_act_log_sum = 0.0
  second_more = 0

  for prob in all_problems:
    assert prob in results1.keys() and prob in results2.keys()
    assert len(results1[prob]) == len(results2[prob]) == 1
    vr1 = results1[prob][0][1]
    vr2 = results2[prob][0][1]
    # if status1 == None:
    #   print(prob,instructions1,activations1)

    if vr1.status == "uns":
      if vr2.status == "uns":
        if vr1.activations != 0 and vr2.activations != 0:
          Xs.append(vr1.activations)
          Ys.append(vr2.activations)

          total_common += 1
          vr1_act_log_sum += math.log(vr1.activations)
          vr2_act_log_sum += math.log(vr2.activations)

          if vr2.activations > vr1.activations:
            second_more += 1

      else:
        only_first.append((get_info(probinfo,prob),prob,vr1.instructions,vr1.activations,vr2))
    elif vr2.status == "uns":
      only_second.append((get_info(probinfo,prob),prob,vr2.instructions,vr2.activations,vr1))

    if vr2.status == "uns":
      continue
    else:
      Vs.append(vr1.instructions)
      Ws.append(vr2.nn_bulks)

  print("Only in first",len(only_first),"problems, such as:")
  for rating,prob,instructions,activations,rest in sorted(only_first)[:5]:
    print(" ",rating,prob,instructions,activations,rest)
  print("  ...")
  for rating,prob,instructions,activations,rest in sorted(only_first)[-15:]:
    print(" ",rating,prob,instructions,activations,rest)

  print()
  print("Only in second",len(only_second),"problems, such as:")
  for rating,prob,instructions,activations,rest in sorted(only_second)[:5]:
    print(" ",rating,prob,instructions,activations,rest)
  print("  ...")
  for rating,prob,instructions,activations,rest in sorted(only_second)[-15:]:
    print(" ",rating,prob,instructions,activations,rest)

  print("total commonly solved",total_common)
  val = (vr1_act_log_sum - vr2_act_log_sum)/total_common
  print("geomeand vr1/vr2 act",math.exp(val))
  print("second more rate",second_more/total_common)

  if False: # figure out how much the newtwork is taking up, looking at problems not solved by it (so that it ran for the whole time slot)
    print("Starting the network computation overhead analysis")
    num_warmups = 0
    sum_warmups = 0

    num_remains = 0
    sum_bulks = 0
    for rating,prob,instructions,activations,vr2 in only_first:
      if vr2.nn_warmup == 0:
        print(prob,vr2)
      else:
        num_warmups += 1
        sum_warmups += vr2.nn_warmup

        if vr2.nn_gnn == 0:
          print(prob,vr2)
        else:
          num_remains += 1
          sum_bulks += vr2.nn_bulks

    print("Avg nn_warmup",sum_warmups/num_warmups)
    print("Avg nn_bulks",sum_bulks/num_remains)

    exit(0)

  if True:
    print("Network computation overhead analysis 2 - all failed neural runs (not just the wins of the default strategy)")
    not_parsed = 0
    parsed_but_no_gnn = 0

    num_warmups = 0
    sum_warmups = 0

    num_remains = 0
    sum_gnn = 0
    sum_bulks = 0
    for prob,vr2s in results2.items():
      # print(prob,vr2s)
      vr2 = vr2s[0][1]
      if vr2.status != None or vr2.instructions < 30000:
        continue

      if vr2.nn_warmup == 0:
        # print(prob,vr2)
        not_parsed += 1
      else:
        num_warmups += 1
        sum_warmups += vr2.nn_warmup

        if vr2.nn_gnn == 0:
          parsed_but_no_gnn += 1
        else:
          num_remains += 1
          sum_bulks += vr2.nn_bulks
          sum_gnn += vr2.nn_gnn

    print("Not parsed",not_parsed)
    print("Parsed but no gnn",parsed_but_no_gnn)

    print("Avg nn_warmup",sum_warmups/num_warmups)
    print("Avg nn_bulks",sum_bulks/num_remains)
    print("Avg nn_gnn",sum_gnn/num_remains)

    exit(0)

  import matplotlib.pyplot as plt

  fig, ax1 = plt.subplots(figsize=(3,3))

  if True:
    ax1.scatter(Xs,Ys,s=1)

    # both axis in log scale
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # same maximal value for both axes
    # max_val = max(max(Xs), max(Ys))
    ax1.set_xlim([1, 200000])
    ax1.set_ylim([1, 200000])

    # make sure the same ticks appear on both axes:
    ax1.set_xticks([1,10,100,1000,10000,100000])
    ax1.set_yticks([1,10,100,1000,10000,100000])

    plt.xlabel("# actvations default strategy")
    plt.ylabel("# actvations neural guidance")

    plt.savefig("activations_scatter.pdf",format="pdf", bbox_inches="tight")
    plt.close(fig)
  else:
    ax1.scatter(Vs,Ws,s=1)

    # both axis in log scale
    # ax1.set_xscale('log')
    #ax1.set_yscale('log')

    # same maximal value for both axes
    # max_val = max(max(Xs), max(Ys))
    # ax1.set_xlim([1, 200000])
    # ax1.set_ylim([1, 200000])

    # make sure the same ticks appear on both axes:
    # ax1.set_xticks([1,10,100,1000,10000,100000])
    # ax1.set_yticks([1,10,100,1000,10000,100000])

    plt.xlabel("default strategys instructions")
    plt.ylabel("neural strats spent in nn")

    plt.savefig("instrucitons_vs_bulks.pdf",format="pdf", bbox_inches="tight")
    plt.close(fig)

