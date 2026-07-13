#!/usr/bin/env python3

import sys, os, torch
import matplotlib.pyplot as plt
import statistics

from collections import defaultdict

if __name__ == "__main__":
  # ./result_reader.py ~/jar2026/split42_boostScale/loop25/train_res.pt

  res_file_path = sys.argv[1]

  (_meta,results) = torch.load(res_file_path,weights_only=False)

  # Problems/LAT/LAT344+3.p VampResult(status=None, instructions=16001, activations=0, nn_warmup_start=7003, nn_warmup=485, nn_gnn_start=7654, nn_gnn=7824, nn_bulks=0, strategy=None)

  # let's go from unsolved to solved
  # first group does not even get to nn_warmup_start

  tot = 0
  failed_to_preprocess = 0
  killed_during_warmup = 0
  warmuppers = 0
  warmup_sum = 0

  warm_but_never_gnn = 0
  killed_during_gnn = 0
  gnn_values = []

  gnn_but_never_act = 0
  gnn_but_never_act_failing = 0
  proper_saturators = 0
  proper_saturators_failed = 0
  proper_saturators_success = 0
  total_satur_instrs = 0
  total_bulk_instrs = 0
  spent_in_nn_values = []
  spent_in_nn_rate_values = []

  for prob,runs in results.items():
    for (i,ilim,info) in runs:
      if i == 0: # let's ignore the reruns, we want to faithfully represent the split between solved and unsolved
        # print(prob,info)
        tot += 1

        if info.nn_warmup_start == 0:
          failed_to_preprocess += 1
        elif info.nn_warmup == 0:
          killed_during_warmup += 1
        else:
          warmuppers += 1
          warmup_sum += info.nn_warmup

          if info.nn_gnn_start == 0:
            warm_but_never_gnn += 1
          elif info.nn_gnn == 0:
            killed_during_gnn += 1
          else:
            gnn_values.append(info.nn_gnn)

            if info.activations == 0:
              gnn_but_never_act += 1
              if info.status == None:
                gnn_but_never_act_failing += 1

            else:
              proper_saturators += 1
              if info.status == None:
                proper_saturators_failed += 1
              else:
                proper_saturators_success += 1

              staturation_starts_at = info.nn_gnn_start + info.nn_gnn
              satuation_ends_at = info.instructions
              satur_instrs = satuation_ends_at - staturation_starts_at

              total_satur_instrs += satur_instrs
              total_bulk_instrs += info.nn_bulks

              # if satur_instrs > 500:
              spent_in_nn_values.append(info.nn_bulks)
              spent_in_nn_rate_values.append(info.nn_bulks/satur_instrs)

  print("tot",tot)
  print("failed_to_preprocess",failed_to_preprocess,"which is",failed_to_preprocess/tot)
  print("killed_during_warmup",killed_during_warmup,"which is",killed_during_warmup/tot)
  print("warmup takes",warmup_sum/warmuppers)
  print("warm_but_never_gnn",warm_but_never_gnn,"which is",warm_but_never_gnn/tot)
  print("killed_during_gnn",killed_during_gnn,"which is",killed_during_gnn/tot)
  print("nn_gnn min",min(gnn_values),"max",max(gnn_values),"median",statistics.median(gnn_values),"mean",statistics.mean(gnn_values))
  print("gnn_but_never_act",gnn_but_never_act,"which is",gnn_but_never_act/tot)
  print("gnn_but_never_act_failing",gnn_but_never_act_failing)
  print("proper_saturators",proper_saturators,"which is",proper_saturators/tot)
  print("proper_saturators_failed",proper_saturators_failed)
  print("proper_saturators_success",proper_saturators_success)
  print("on average saturating",total_satur_instrs/proper_saturators,"instrs")
  print("on average NN evaling",total_bulk_instrs/proper_saturators,"instrs")
  print("nn_bulk min",min(spent_in_nn_values),"max",max(spent_in_nn_values),"median",statistics.median(spent_in_nn_values),"mean",statistics.mean(spent_in_nn_values))
  print("nn_bulk_rate_to_satur min",min(spent_in_nn_rate_values),"max",max(spent_in_nn_rate_values),"median",statistics.median(spent_in_nn_rate_values),"mean",statistics.mean(spent_in_nn_rate_values))

  plt.figure()
  plt.hist(spent_in_nn_rate_values, bins=50, range=(0, 1), edgecolor='black')
  plt.xlabel('NN bulk rate (nn_bulks / saturation instrs)')
  plt.ylabel('Count')
  plt.title('Distribution of time spent in NN evaluation')
  plt.tight_layout()
  plt.savefig(os.path.splitext(res_file_path)[0] + "_nn_rate_hist.pdf")
  # plt.show()

  plt.figure()
  plt.hist(spent_in_nn_values, bins=50, edgecolor='black')
  plt.xlabel('NN bulk instructions')
  plt.ylabel('Count')
  plt.title('Distribution of instructions spent in NN evaluation')
  plt.tight_layout()
  plt.savefig(os.path.splitext(res_file_path)[0] + "_nn_bulk_hist.pdf")
  # plt.show()

  # Stacked bar: problem set breakdown by pipeline stage
  labels = ['Did not finish CNF', 'Killed during NN warmup', 'Killed during GNN',
            'GNN finished but no activation', 'Saturation ran (solved)', 'Saturation ran (failed)']
  counts = [failed_to_preprocess, killed_during_warmup + warm_but_never_gnn, killed_during_gnn,
            gnn_but_never_act, proper_saturators_success, proper_saturators_failed]
  colors = ['#d62728', '#ff7f0e', '#bcbd22', '#17becf', '#2ca02c', '#aec7e8']

  fig, ax = plt.subplots(figsize=(10, 1))
  left = 0
  for label, count, color in zip(labels, counts, colors):
    bar = ax.barh(0, count, height=0.35, left=left, color=color, edgecolor='white', label=f'{label} ({count})')
    left += count
  ax.set_xlim(0, tot)
  # ax.set_ylim(-1.0, 1.0)
  ax.set_yticks([])
  # ax.spines['top'].set_visible(False)
  # ax.spines['right'].set_visible(False)
  # ax.spines['left'].set_visible(False)
  ax.set_xlabel('Number of problems')
  ax.set_title('Problem set breakdown by pipeline stage')
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.7), ncol=3, fontsize='small')
  # fig.subplots_adjust(bottom=0.55)
  plt.savefig(os.path.splitext(res_file_path)[0] + "_breakdown.pdf", bbox_inches='tight')
  # plt.show()
