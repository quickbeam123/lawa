#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time, subprocess

from collections import defaultdict

MISSIONS = ["train","test"]
STYLES = { MISSIONS[0] : "-", MISSIONS[1] : "--"}

MAXINT = 2**32

from multiprocessing import Pool

def retry_in_vamp(task):
  (prob,opts) = task
  return (IC.vampire_perfrom(prob,opts),opts)

def retry_in_POOL(tasks):
  if True:
    pool = Pool(processes=60) # number of cores to use
    results = pool.map(retry_in_vamp, tasks, chunksize = 1)
    pool.close()
    pool.join()
    del pool
  else:
    results = []
    for task in tasks:
      results.append(retry_in_vamp(task))
  return results

if __name__ == "__main__":
  # Report newly (with regards to loop iterations) solving hard (TPTP rating 1.0) problems
  #
  # To be called as in: ./checkTheBang.py exper_folder1

  POIS = []

  with open("/nfs/sudamar2/TPTP-v9.1.0/probinfo9.1.0.pkl",'rb') as f:
    probinfo = pickle.load(f)

  total = 0
  shorts = 0
  unknowns = 0
  cnfs = 0
  fofs = 0
  tf0s = 0
  chainies = 0
  covered = set()
  all_loops = set()

  additions = defaultdict(int)
  problem_solutions = defaultdict(dict) # prob -> {loop: instructions}

  for exper_dir in sys.argv[1:]:
    print(exper_dir)
    root, dirs, files = next(os.walk(exper_dir))
    min_loop = MAXINT
    max_loop = 0
    for dir in dirs:
      if dir.startswith("loop"):
        dirs_loop = int(dir[4:])
        if dirs_loop < min_loop:
          min_loop = dirs_loop
        if dirs_loop > max_loop:
          max_loop = dirs_loop
    # loop,update = max_loop,-1
    loop,update = min_loop,1
    while True:
      loop_str = "loop{}".format(loop)
      cur_dir = os.path.join(exper_dir,loop_str)
      if not os.path.isdir(cur_dir):
        break

      print("  loop",loop)
      all_loops.add(loop)
      root, dirs, files = next(os.walk(cur_dir))
      for file in files:
        if file not in ["train_res.pt","test_res.pt"]:
          continue

        # print("    ",file)
        (meta,results) = torch.load(os.path.join(cur_dir,file),weights_only=False)
        # print("      ",meta)

        for prob,runs in results.items():
          for (i,ilim,vr) in runs:
            if vr.status == "uns":
              info = probinfo[prob.split("/")[-1]]
              rate = float(info[0]) if info[0] is not None else 0.5 # whatever

              if rate > 0.99 and loop not in problem_solutions[prob]:
                problem_solutions[prob][loop] = vr.instructions

              if prob not in covered:
                covered.add(prob)
                if rate > 0.99:
                  total += 1
                  additions[loop] += 1

                  if False: # for the proof reconstruction, desperation mode
                    if prob in POIS:
                      print("      ",i,prob,total,rate,vr)

                      temp_extra = HP.PERFORMS_SPECIAL[i]

                      print("RERUN AND REPRODUCE")
                      sys.stdout.flush()
                      tasks = []
                      for _ in range(128):
                        seed = random.randint(1,0x7fffffff)
                        opts = f"-t 300 -i 100000 --input_syntax tptp -sa otter -ncem /home/sudamar2/mtpa-gnn/tptpOverfit100k3/loop{loop}/script-model.pt -npcc on {temp_extra} -si on -rtra on --random_seed {seed}"
                        tasks.append((prob,opts))

                      best_instrs = 200000
                      best_opts = None
                      best_has_npcct = True

                      # CAREFUL; we need HP.VAMPIRE_EXECUTABLE to be "vampire_rel_mtpa-gnn_8867" for the following! 

                      for res,opts in retry_in_POOL(tasks):
                        if res.status == "uns":
                          print(res,"for",opts)
                          has_npcct = "npcct" in opts
                          if best_has_npcct and not has_npcct:
                            best_has_npcct = False
                            best_opts = opts
                            best_instrs = res.instructions
                          elif res.instructions < best_instrs:
                            # print("Improving best_instrs from",best_instrs,"to",res.instructions,"for",opts)
                            best_instrs = res.instructions
                            best_opts = opts

                      if best_opts:
                        print("Choosing",best_opts)
                        result_file = "hards/"+prob.split("/")[-1]+"rf"
                        running = f"./vampire_rel_mtpa-gnn_8867 {best_opts} {prob} -p on"
                        subprocess.run(f"echo {running} > {result_file}", shell=True)
                        subprocess.run(f"./run_lawa_vampire.sh {running} >> {result_file}", shell=True)

                        print("Written proof to",result_file)
                      else:
                        print("Failed to reproduce for",prob)

                      sys.stdout.flush()

                      # TODO: fix the HP's vampire version back
                      # rerun with with "-p on"

                      # print(f"./vampire_rel_mtpa-gnn_8867  -p on")


                    continue

                  if True:
                    res = subprocess.run(["grep", "Rating", prob], stdout=subprocess.PIPE, text=True)
                    output = res.stdout
                    spl = output.split()
                    assert spl[0] == "%"
                    assert spl[1] == "Rating"
                    assert spl[2] == ":"
                    short = len(spl) == 5
                    # if short:
                    #     print("SHORT!")
                    # print(output)
                    # print(info[1])
                    # print(info[2])
                  if True:
                    res = subprocess.run(["grep", "Comments", prob], stdout=subprocess.PIPE, text=True)
                    output = res.stdout
                    chainy = "Chainy" in output
                  if chainy:
                    chainies += 1

                  unknown = "_UNK_" in info[1]
                  if unknown:
                    unknowns += 1
                  if short:
                    shorts += 1
                  assert unknown <= short, (prob,info)

                  logic = info[1][:3]
                  if logic == "FOF":
                    fofs += 1
                  elif logic == "CNF":
                    cnfs += 1
                  else:
                    assert logic == "TF0"
                    tf0s += 1

                  print("      ",i,prob,total,rate,info,vr.__dict__,"SHORT" if short else "")

                  if False and short: # for the proof reconstruction
                    print("RERUN AND REPRODUCE")
                    sys.stdout.flush()
                    tasks = []
                    for temp_extra in [" -npcct 1.0", " -npcct 0.333", " -npcct 0.111", " -npcct 0.037", ""]:
                      for seed in range(1,13):
                        opts = f"-t 300 -i 100000 --input_syntax tptp -sa otter -ncem /home/sudamar2/mtpa-gnn/tptpOverfit100k3/loop{loop}/script-model.pt -npcc on {temp_extra} -si on -rtra on --random_seed {seed}"
                        tasks.append((prob,opts))

                    best_instrs = 200000
                    best_opts = None
                    best_has_npcct = True

                    # CAREFUL; we need HP.VAMPIRE_EXECUTABLE to be "vampire_rel_mtpa-gnn_8867" for the following! 

                    for res,opts in retry_in_POOL(tasks):
                      if res.status == "uns":
                        print(res,"for",opts)
                        has_npcct = "npcct" in opts
                        if best_has_npcct and not has_npcct:
                          best_has_npcct = False
                          best_opts = opts
                          best_instrs = res.instructions
                        elif res.instructions < best_instrs:
                          # print("Improving best_instrs from",best_instrs,"to",res.instructions,"for",opts)
                          best_instrs = res.instructions
                          best_opts = opts

                    if best_opts:
                      print("Choosing",best_opts)
                      result_file = "hards/"+prob.split("/")[-1]+"rf"
                      running = f"./vampire_rel_mtpa-gnn_8867 {best_opts} {prob} -p on"
                      subprocess.run(f"echo {running} > {result_file}", shell=True)
                      subprocess.run(f"./run_lawa_vampire.sh {running} >> {result_file}", shell=True)

                      print("Written proof to",result_file)
                    else:
                      print("Failed to reproduce for",prob)

                    sys.stdout.flush()

                    # TODO: fix the HP's vampire version back
                    # rerun with with "-p on"

                    # print(f"./vampire_rel_mtpa-gnn_8867  -p on")

              break

      loop += update

  print("In total found,",total,"problems of rating > 0.99")
  print("of them",shorts,"never rerated and",unknowns,"of status UNK")
  print("Cnfs:",cnfs,"Fofs:",fofs,"Tf0s:",tf0s)
  print("Chainies:",chainies)

  print(additions)

  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_pdf import PdfPages
  from matplotlib.ticker import MaxNLocator

  with PdfPages("checkTheBang.pdf") as pdf:
    X_TICKS = list(range(0, 51, 5))

    def unify_x(ax):
      ax.set_xticks(X_TICKS)
      ax.set_xlim(-0.5, 50.5)

    xs = sorted(additions)
    ys = [additions[x] for x in xs]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(xs, ys, color="#4269a3", width=0.7)
    ax.set_xlabel("loop")
    ax.set_ylabel("newly solved")
    ax.set_title("Newly solved problems of rating > 0.99 per loop")
    unify_x(ax)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
      ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    solved_per_loop = defaultdict(int)
    for prob, loop_instrs in problem_solutions.items():
      for loop in loop_instrs:
        solved_per_loop[loop] += 1

    xs2 = sorted(solved_per_loop)
    ys2 = [solved_per_loop[x] for x in xs2]
    print("solved_per_loop",ys2)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(xs2, ys2, color="#4269a3", width=0.7)
    ax.set_xlabel("loop")
    ax.set_ylabel("solved")
    ax.set_title("Problems of rating > 0.99 solved per loop")
    unify_x(ax)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
      ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    solved_sets = defaultdict(set)
    for prob, loop_instrs in problem_solutions.items():
      for loop in loop_instrs:
        solved_sets[loop].add(prob)

    xs3 = sorted(all_loops)
    added3 = []
    lost3 = []
    prev_solved = set()
    for x in xs3:
      cur_solved = solved_sets[x]
      added3.append(len(cur_solved - prev_solved))
      lost3.append(-len(prev_solved - cur_solved))
      prev_solved = cur_solved

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(xs3, added3, color="#4269a3", width=0.7, label="added")
    ax.bar(xs3, lost3, color="#e34948", width=0.7, label="lost")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel("loop")
    ax.set_ylabel("problems (vs. previous loop)")
    ax.set_title("Problems of rating > 0.99 added/lost vs. previous loop")
    unify_x(ax)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for spine in ["top","right"]:
      ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    solve_counts = defaultdict(int)
    for prob, loop_instrs in problem_solutions.items():
      solve_counts[len(loop_instrs)] += 1

    xs4 = list(range(1, max(solve_counts) + 1))
    ys4 = [solve_counts[x] for x in xs4]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(xs4, ys4, color="#4269a3", width=0.7)
    ax.set_xlabel("number of loops solved in")
    ax.set_ylabel("number of problems")
    ax.set_title("Problems of rating > 0.99 by number of solving loops")
    unify_x(ax)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
      ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    '''
    fig, ax = plt.subplots(figsize=(10,6))
    problem_solutions_list = list(problem_solutions.items())
    # random.shuffle(problem_solutions_list)
    for prob, loop_instrs in problem_solutions_list:
      loops = sorted(loop_instrs)
      instrs = [loop_instrs[l] for l in loops]
      ax.plot(loops, instrs, "-o", color="#4269a3", alpha=0.25,
              linewidth=0.6, markersize=3)
    ax.set_xlabel("loop")
    ax.set_ylabel("instructions at solve")
    ax.set_title("Per-problem solve instructions across loops (rating > 0.99)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
      ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    '''
  