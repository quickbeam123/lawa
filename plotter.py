#!/usr/bin/env python3

import workers as W
import inf_common as IC
import hyperparams as HP

import torch

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

MISSIONS = ["train","test"]
STYLES = { MISSIONS[0] : "-", MISSIONS[1] : "--"}

MAXINT = 2**32

def get_status(info):
  if isinstance(info,IC.VampResult):
    return info.status
  else:
    return info[0]


DEEPER = True

COLLECT_DEEPER_TEST_PERF = False
AT_INDEX = 11 # translates to 10 in the rendered plot
NUM_CHARS_TO_CUT = len("../deeper/B-mesh-tf0-out/")
def normalize_prob_name(str):
  return str[NUM_CHARS_TO_CUT:]

SPLIT_MULTI = False

GREEDY_START_GREEDY_END = True # to compare a greedy sequence of champs before and after they get boosted. Only don't on the 0-th slice; i.e. as if temp==0.0
GREEDY_FROM_TEST = True
COMMONNIFY_NAMES = True
SAVE_TO_FILES = False

def common_name(name):
  if not COMMONNIFY_NAMES:
    return name
  # e.g. "../deeper/B-mesh-tf0-out/HOL-MicroJava/0050_BVSpecTypeSafe/prob_01238_041871.p"
  new_name = "/".join(name.split("/")[3:])
  # print(new_name)
  return new_name

if __name__ == "__main__":
  # Plotting some training curves, automatically getting the data from the exper directories left behind by looper
  #
  # To be called as in: ./plotter.py exper_folder1 exper_folder2 ...

  # for each evaluation mode found (e.g. "train_t0.25.pt"),
  # keep storing pairs (solutions,time)
  expers = {}

  ever_seen = set()
  covereds = {} # experdir -> last loop's covered problem set

  if COLLECT_DEEPER_TEST_PERF:
    test_perf = defaultdict(set) # experdir -> largest set of test problems solved

  if GREEDY_START_GREEDY_END:
    start_sets = {} # exper -> set of problems solved by slice 0 at iter 0
    best_sets = {} # exper -> set of problems solved by slice 0 at iter argmax

  for exper_dir in sys.argv[1:]:
    print(exper_dir)
    best_covered = set()
    plottables = {m : ([],[]) for m in MISSIONS}

    if SPLIT_MULTI:
      extra_plottables = {idx : {m : ([],[]) for m in MISSIONS} for idx in range(5)}

    root, dirs, files = next(os.walk(exper_dir))
    loop = MAXINT
    for dir in dirs:
      if dir.startswith("loop"):
        dirs_loop = int(dir[4:])
        if dirs_loop < loop:
          loop = dirs_loop
    while True:
      loop_str = "loop{}".format(loop)
      cur_dir = os.path.join(exper_dir,loop_str)
      if not os.path.isdir(cur_dir):
        break

      # print("  ",cur_dir)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file not in ["train_res.pt","test_res.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file),weights_only=False)
        # print("      ",meta)

        sample_record = next(iter(results.values()))[0]
        if len(sample_record) == 3:
          if isinstance(sample_record[2],W.VampResult):
            covered = {prob for prob,runs in results.items() for (i,ilim,info) in runs if (info.status == "uns" and i == 0) }
            ever_seen |= covered
            fractional = len(covered)
            if len(covered) > len(best_covered):
              best_covered = covered

            if SPLIT_MULTI:
              # special hack to decompose "all5champs" into separate lines
              ran_on_howmanies = [len({prob for prob,runs in results.items() for (i,ilim,info) in runs if(i == idx)}) for idx in range(5)]
              if all(howmany== len(results) for howmany in ran_on_howmanies): # all 5 attempts ran on all problems
                for idx in range(5):
                  local_plottables = extra_plottables[idx]
                  local_factional = len({prob for prob,runs in results.items() for (i,ilim,info) in runs if (info.status == "uns") and (i == idx)}) / len(results)
                  for m in MISSIONS:
                    if not file.startswith(m): # on purpose (ab)use "the other mission"
                      local_plottables[m][0].append(loop) # NOTE: add -1 here to get the plots starting at 0, like for CADE
                      local_plottables[m][1].append(local_factional)
            if GREEDY_START_GREEDY_END and (not GREEDY_FROM_TEST or file == "test_res.pt"):
              covered0 = {common_name(prob) for prob,runs in results.items() for (i,ilim,info) in runs if (info.status == "uns" and i == 0) }
              if loop == 1:
                start_sets[exper_dir] = covered0
                print(f"start_sets[{exper_dir}] = {len(covered0)}")
              if exper_dir not in best_sets or len(best_sets[exper_dir]) < len(covered0):
                best_sets[exper_dir] = covered0
                print(f"best_sets[{exper_dir}] = {len(covered0)}")
                # print(f"best_sets[{exper_dir}] = {len(covered0)}")

            if COLLECT_DEEPER_TEST_PERF and file == "test_res.pt":
              cur_norm_set = {normalize_prob_name(prob) for prob,runs in results.items() for (i,ilim,info) in runs if (info.status == "uns" and i == 0) }
              if loop == AT_INDEX: # len(cur_norm_set) > len(test_perf[exper_dir]):
                print(exper_dir,"test-perf snapshot at",loop,"scoring",len(cur_norm_set))
                test_perf[exper_dir] = cur_norm_set

          else:
            fractional = sum(1/len(runs) for prob,runs in results.items() for (status,instructions,activations) in runs if status == "uns")
        else:
          # started using NUM_PERFORMS with different params; only the 0-labeled run, however, counts
          fractional = sum(1.0 for prob,runs in results.items() for (i,info) in runs if (i == 0 and get_status(info) == "uns"))

        fractional /= len(results) # divide by number of problems in results (includes the ones on whic we failed)

        if DEEPER:
          fractional *= 100 # to go "percent" as a unit

        print(fractional)

        """
        for prob,runs in results.items():
          for (i,info) in runs:
            if info.status == None and info.instructions >= 10000:
            print(prob,info)
        """

        # print("     -> ",successes)
        for m in MISSIONS:
          if file.startswith(m):
            plottables[m][0].append(loop -1 ) # NOTE: add -1 here to get the plots starting at 0, like for CADE
            plottables[m][1].append(fractional)

      loop += 1

    expers[exper_dir] = plottables
    if SPLIT_MULTI:
      for idx in range(5):
        local_plottables = extra_plottables[idx]
        if any(len(local_plottables[m][0]) for m in MISSIONS):
          expers[f"{exper_dir}_{idx+1}"] = local_plottables

    covereds[exper_dir] = best_covered

  if GREEDY_START_GREEDY_END:
    print("start strat boosting table")

    iter = 0
    greed0 = set()
    greed_boost = set()
    covered_1_plain = None
    covered_1_boosted = None
    for exper_dir, covered0 in start_sets.items():
      covered_boost = best_sets[exper_dir]
      if SAVE_TO_FILES:
        new_name = exper_dir.split("/")[-1]+".best.txt"
        with open(new_name,"w") as f:
          for prob in covered_boost:
            f.write(prob+"\n")
      if iter == 0:
        covered_1_plain = covered0
        covered_1_boosted = covered_boost
      iter += 1
      percent_boost = 100*(len(covered_boost) / len(covered0) - 1.0)

      print(f"{iter} & \\num{{{len(covered0 - greed0)}}} & \\num{{{len(covered0)}}} & +\\SI{{{percent_boost:.1f}}}{{\\percent}} & \\num{{{len(covered_boost)}}} & \\num{{{len(covered_boost - greed_boost)}}} & ${{{iter}}}'$ \\\\")
      greed0 |= covered0
      greed_boost |= covered_boost

    percent_boost = 100*(len(greed_boost) / len(greed0) - 1.0)
    print("\hline")
    print(f"union:        & \\num{{{len(greed0)}}} & $\\longrightarrow$ & +\\SI{{{percent_boost:.1f}}}{{\\percent}}  & $\\longrightarrow$ & \\num{{{len(greed_boost)}}} \\\\")
    percent_1_plain = 100*(len(greed0) / len(covered_1_plain) - 1.0)
    percent_1_boosted = 100*(len(greed_boost) / len(covered_1_boosted) - 1.0)
    print(f"  & (\\SI{{{percent_1_plain:.1f}}}{{\\percent}} of 1) &&&& (\\SI{{{percent_1_boosted:.1f}}}{{\\percent}} of $1'$)")

    print()

  if COLLECT_DEEPER_TEST_PERF:
    print("Collected Deeper test perf")
    total = set()
    for exper_dir, covered0 in test_perf.items():
      with open(exper_dir.split("/")[-1]+"atLoop10.txt","w") as f:
        for prob in covered0:
          print(prob,file=f)
      print(exper_dir,len(covered0))
      total |= covered0
    print("Total",len(total))


  if True:
    print("Greedy cover of best sets from each exper:")
    total = set()
    while True:
      best_dir = None
      best_dir_adds = 0
      for exper_dir,covers in covereds.items():
        adds = len(covers - total)
        if adds > best_dir_adds:
          best_dir = exper_dir
          best_dir_adds = adds
      if best_dir is not None:
        print(best_dir,"adds",best_dir_adds)
        # TODO: print only the high-rating ones here:
        # for prob in covereds[best_dir] - total:
        #     print("  ",prob)
        total |= covereds[best_dir]
      else:
        print("  in total",len(total))
        break
    print("And ever_seen",len(ever_seen))

  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator

  # fig, ax1 = plt.subplots(figsize=(3.2,3))
  if DEEPER:
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
  else:
    fig, ax1 = plt.subplots(figsize=(6,5))
  color_cycle = ax1._get_lines.prop_cycler
  handles = []

  common_prefix = os.path.commonprefix(list(expers.keys()))

  REPLACE = {"/home/sudamar2/mtpa-gnn/newSplit30k":"base",
             "/home/sudamar2/mtpa-gnn/newSplit30k-noImit": "noImit",
             "/home/sudamar2/mtpa-gnn/newSplit30k-cumul-np5": "boost",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul": "base10k",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noGage": "noGage",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noGweight": "noGweight",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-noSF": "noSF",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justGage": "justGage",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justGweight": "justGweight",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-justSF": "justSF",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-is128": "m=128",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-es48": "n=48",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-es16": "n=16",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl4": "k=4",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl8": "k=8",
             "/home/sudamar2/mtpa-gnn/newbase10k-cumul-gl16": "k=16",
             "/home/sudamar2/ijcar2026/newDefaults_i32K": "default",
             "/home/sudamar2/ijcar2026/newDefaults_smartAgain_i32K": "smartAgain",
             "/home/sudamar2/holawa/all192516shuffled_10730_defaults_theTF0CompletishChamp_i32K-B": "TF0",
             "/home/sudamar2/holawa/all192516shuffled_6946_defaults_theTH0CompletishChamp_i32K": "TH0",
             "/home/sudamar2/holawa/all192516shuffled_6946_defaults_theTH1CompletishChamp_i32K": "TH1",
             "/home/sudamar2/holawa/all246277shuffled120K-tf0_6946_defaults_tf0Champ1_i16K": "TF0",
             "/home/sudamar2/holawa/all246277shuffled120K-th0_6946_defaults_th0Champ1_i16K": "TH0",
             "/home/sudamar2/holawa/all246277shuffled120K-th1_6946_defaults_th1Champ1_i16K": "TH1",
             }

  for exper_dir,plottables in expers.items():
    col = next(color_cycle)['color']

    print(col)

    for m,(Xs,Ys) in plottables.items():
      if Xs:
        if DEEPER:
          Xs = Xs[:12]
          Ys = Ys[:12]

        print(exper_dir)
        if exper_dir in REPLACE:
          lab = REPLACE[exper_dir]
        else:
          lab = exper_dir[len(common_prefix)-2:]+"_"+m

        h, = ax1.plot(Xs, Ys, STYLES[m], linewidth = 1.5, label = lab, color=col)
        if m == "train":
          handles.append(h)

        max_val,max_idx = (0.0,0)
        imax_val,imax_idx = (0.0,0)
        for idx,val in zip(Xs,Ys):
          if val > max_val:
            max_val = val
            max_idx = idx
          if idx > 1 and val > imax_val:
            imax_val = val
            imax_idx = idx

        print(exper_dir,m,"max with",max_val,"at",max_idx)
        print("Also imax with",imax_val,"at",imax_idx)

  # Apply integer-only ticks to X-axis
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

  # ax1.set_xlim(xmin=0,xmax=24)
  # ax1.set_ylim(ymin=20 ,ymax=70)
  if DEEPER:
    # ax1.set_xlim(xmin=0,xmax=11)
    ax1.set_ylim(ymin=20,ymax= 70)
  # ax1.axhline(y=0.5386, color='gray', linestyle='--', linewidth=0.5) # for freshQuarter
  # ax1.axhline(y=0.5294, color='gray', linestyle='--', linewidth=0.5) # for freshFull

  plt.xlabel("improvement loop iteration")
  plt.ylabel(f"percentage problems solved")

  if DEEPER:
    plt.legend(handles = handles,loc="lower center",ncols=3)
  else:
    plt.legend(handles = handles, loc='lower right') # loc = 'best' is rumored to be unpredictable
  plt.savefig("current_plot.pdf".format("+".join(os.path.basename(dir) for dir in sys.argv[1:])),format="pdf", bbox_inches="tight")
  plt.close(fig)





