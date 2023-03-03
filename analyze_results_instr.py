#!/usr/bin/env python3

import sys, os

import pickle
from collections import defaultdict

def get_info(probinfo,prob):
  long_name = "{}.p".format(prob)
  return probinfo[long_name]

def recognize_success(res_and_rest,success_kinds):
  if isinstance(res_and_rest,list):
    # return res_and_rest[0][0] in success_kinds
    for res_item in res_and_rest:
      if res_item[0] in success_kinds:
        return True
    return False
  else:
    return res_and_rest[0] in success_kinds

if __name__ == "__main__":
  confs = [] # to have solvers in order

  strats = {} # conf -> its strat string
  results = {} # conf -> (bechmark -> time)

  solveds_sat = {} # conf -> set_of_solved
  solveds_uns = {} # conf -> set_of_solved

  bigUnion_sat = set()
  bigUnion_uns = set()

  with open("/nfs/sudamar2/TPTP-v7.5.0/probinfo7.5.0.pkl",'rb') as f:
    probinfo = pickle.load(f)

  # print(probinfo)

  # tmp = []

  for filename in sys.argv[1:]:
    with open(filename,'rb') as f:
      confs.append(filename)
      (meta,result) = pickle.load(f)
      results[filename] = result

      print(meta)
      # meta_spl = meta.split()
      # print(meta_spl)
      ''' # this used to work with the minimizer runs
      for i,bit in enumerate(meta_spl):
        if bit == "--decode":
          break
      strats[filename] = meta_spl[i+1]
      '''
      # strats[filename] = "["+" ".join(meta_spl[4:8])+"]"
      # print(strats[filename])

      # print(filename)
      # print(meta_spl)
      # print(strats[filename])
      # print()

      '''
      for prob,(res,time,inst,acts) in result.items():
        tmp.append((inst,time,prob))
      '''
      # print(result.items())

      solved_sat = {prob for prob,res_and_rest in result.items() if recognize_success(res_and_rest,["sat"])}
      solveds_sat[filename] = solved_sat

      solved_uns = {prob for prob,res_and_rest in result.items() if recognize_success(res_and_rest,["uns"])}
      solveds_uns[filename] = solved_uns

      # print(len(solved_sat),len(solved_uns),meta_spl[i+1])

      bigUnion_sat |= solved_sat
      bigUnion_uns |= solved_uns

  '''
  tmp.sort(key=lambda x : x[0])
  for tmprec in tmp:
    print(tmprec)
  '''

  # TODO: these are a bit meaningless without "ran of what"
  print("Sort by SAT")
  for conf, set_of_solved in sorted(solveds_sat.items(),key = lambda x : len(x[1])):
    print(len(set_of_solved), [conf])
    for prob in set_of_solved:
      info = get_info(probinfo,prob)
      spc = info[1]
      stat = spc.split("_")[1]
      if stat not in ["SAT","CSA"]:
        print(prob,stat,results[conf][prob])
  print()

  print("Sort by UNS")
  for conf, set_of_solved in sorted(solveds_uns.items(),key = lambda x : len(x[1])):
    print(len(set_of_solved), [conf])
    for prob in set_of_solved:
      info = get_info(probinfo,prob)
      rate = float(info[0])
      if rate >= 0.95:
        print(prob,rate,results[conf][prob])

  print()

  print("Total", len(bigUnion_sat),"/",len(bigUnion_uns))

  # exit(0)

  # TODO: needs to be updated to use solveds_sat/solveds_uns/strats

  print()
  print("Greedy cover (UNS):")
  covered = set()
  while True:
    best_val = 0
    for strat, solved in solveds_uns.items():
      val = len(solved - covered)
      if val > best_val:
        best_val = val
        best_strat = strat
    if best_val > 0:
      best_solved = solveds_uns[best_strat].copy()
      print("#",best_strat, "contributes", best_val, "total", len(best_solved))
      # print('./run_in_parallel_plus_local.sh 36 problemsSTD.txt ./run_vampire_free.sh ./vampire_z3_rel_shuffling_6350 "--decode {} -i 250000" /local/sudamar2/strats/{}_250000'.format(metas[best_strat][-2],best_strat.split("/")[-1][:-4]))

      '''
      print(best_strat, "contributes", best_val, "total", len(best_solved),end=" ") 
      for strat, solved in solveds.items():
        if strat != best_strat:
          best_solved = best_solved - solved
      print("uniques", len(best_solved)) #, sorted([(probinfo[prob][0],prob) for prob in best_solved],reverse=True)[:10]
      '''

      '''
      for prob in best_solved:
        info = probinfo[prob+".p"]
        print(info[1].split("_")[1],info[0],prob,results[best_strat][prob],metas[best_strat].split()[-2])
      '''
      covered = covered | solveds_uns[best_strat]
    else:
      break
  print("Total", len(covered))

  # Cactus plot (on unsat) - special for the lawa paper

  # for zoom plot
  if True:
    label_seen = set()
    print()
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(2.7, 2.5))
    limit = 100000
    total = None
    for filename,result in results.items():
      print(filename,len(result))
      if total is None:
        total = len(result)
      else:
        assert len(result) == total

      successInstrs = []
      for prob,(stat,time,instrs,acts) in result.items():
        if stat == "uns" and instrs <= limit:
          successInstrs.append(instrs)

      successInstrs.sort()
      Xs = [0] + successInstrs
      Ys = list(range(len(Xs)))
      Xs.append(limit)
      Ys.append(Ys[-1])
      Ys = list(map(lambda val : 100.0*val / total,Ys))
      # have the line reach the end

      myalpha = 1.0
      if "awr" in filename:
        if "nwc5" in filename:
          mylabel = "awr 1:5+"
          mycolor = "tab:blue"
        elif "av-off" in filename:
          mylabel = "awr 1:5$-$"
          mycolor = "tab:grey"
        else:
          mylabel = "awr 1:5"
          mycolor = "tab:green"
      else:
        myalpha = 0.6
        if "NF2" in filename:
          mylabel = "NN(2)"
          mycolor = "tab:orange"
        else:
          mylabel = "NN(12)"
          mycolor = "tab:red"

      if mylabel in label_seen:
        mylabel = ""
      else:
        label_seen.add(mylabel)

      ax1.plot(Xs,Ys,label=mylabel,linewidth=1,color=mycolor,alpha=myalpha)

    # ax1.axvline(5000,ymin=0.05, ymax=0.95,ls='--', lw=1,color="gray")
    # ax1.axvline(50000,ymin=0.05, ymax=0.95,ls='--', lw=1,color="gray")
    plt.xlim([5000, limit])
    plt.ylim([35.0, 50.0])
    plt.xscale("log")
    ax1.set_xlabel('mega instructions')
    ax1.set_ylabel('percent solved')
    # box = ax1.get_position()
    # plt.legend(ncol=1,columnspacing=0.8,handlelength=1,bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig("/nfs/sudamar2/lawa/zoom_plot.png",format='png', dpi=300)
    plt.close(fig)

  # for master plot
  if False:
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    limit = 100000
    total = None
    for filename,result in results.items():
      print(filename,len(result))
      if total is None:
        total = len(result)
      else:
        assert len(result) == total

      successInstrs = []
      for prob,(stat,time,instrs,acts) in result.items():
        if stat == "uns" and instrs <= limit:
          successInstrs.append(instrs)

      successInstrs.sort()
      Xs = [0] + successInstrs
      Ys = list(range(len(Xs)))
      Xs.append(limit)
      Ys.append(Ys[-1])
      Ys = list(map(lambda val : 100.0*val / total,Ys))
      # have the line reach the end

      if "awr" in filename:
        if "nwc5" in filename:
          mylabel = "awr 1:5+"
          mycolor = "tab:blue"
        elif "av-off" in filename:
          mylabel = "awr 1:5$-$"
          mycolor = "tab:grey"
        else:
          mylabel = "awr 1:5"
          mycolor = "tab:green"
      else:
        if "NF2" in filename:
          mylabel = "NN(2)"
          mycolor = "tab:orange"
        else:
          mylabel = "NN(12)"
          mycolor = "tab:red"

      ax1.plot(Xs,Ys,label=mylabel,linewidth=1,color=mycolor)

    # ax1.axvline(5000,ymin=0.05, ymax=0.95,ls='--', lw=1,color="gray")
    # ax1.axvline(50000,ymin=0.05, ymax=0.95,ls='--', lw=1,color="gray")
    plt.xlim([60, 5000])
    plt.ylim([0.0, 50.0])
    plt.xscale("log")
    ax1.set_xlabel('mega instructions')
    ax1.set_ylabel('percent solved')
    # fig.tight_layout()
    # plt.legend(loc='lower right',ncol=1,columnspacing=0.8,handlelength=1)
    plt.legend(ncol=1,columnspacing=0.8,handlelength=1,bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 1.0, 1.0])
    plt.savefig("/nfs/sudamar2/lawa/master_plot.png",format='png', dpi=300)
    plt.close(fig)


