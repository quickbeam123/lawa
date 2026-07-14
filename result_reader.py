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

  filling_up_passive = 0
  filling_up_passive_failed = 0
  filling_up_passive_success = 0

  proper_saturators = 0
  proper_saturators_failed = 0
  proper_saturators_success = 0
  total_satur_instrs = 0
  total_bulk_instrs = 0
  spent_in_nn_values_fail = []
  spent_in_nn_rate_values_fail = []
  spent_in_nn_values_succ = []
  spent_in_nn_rate_values_succ = []

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

            if info.nn_bulks == 0:
              filling_up_passive += 1
              if info.status == None:
                filling_up_passive_failed += 1
              else:
                filling_up_passive_success += 1
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

              if info.status == None:
                spent_in_nn_values_fail.append(info.nn_bulks)
                spent_in_nn_rate_values_fail.append(info.nn_bulks/satur_instrs)
              else:
                spent_in_nn_values_succ.append(info.nn_bulks)
                spent_in_nn_rate_values_succ.append(info.nn_bulks/satur_instrs)

  print("tot",tot)
  print("failed_to_preprocess",failed_to_preprocess,"which is",failed_to_preprocess/tot)
  print("killed_during_warmup",killed_during_warmup,"which is",killed_during_warmup/tot)
  print("warmup takes",warmup_sum/warmuppers)
  print("warm_but_never_gnn",warm_but_never_gnn,"which is",warm_but_never_gnn/tot)
  print("killed_during_gnn",killed_during_gnn,"which is",killed_during_gnn/tot)
  print("nn_gnn min",min(gnn_values),"max",max(gnn_values),"median",statistics.median(gnn_values),"mean",statistics.mean(gnn_values))
  print("filling_up_passive",filling_up_passive,"which is",filling_up_passive/tot)
  print("filling_up_passive_failed",filling_up_passive_failed)
  print("filling_up_passive_success",filling_up_passive_success)
  print("proper_saturators",proper_saturators,"which is",proper_saturators/tot)
  print("proper_saturators_failed",proper_saturators_failed)
  print("proper_saturators_success",proper_saturators_success)
  print("on average saturating",total_satur_instrs/proper_saturators,"instrs")
  print("on average NN evaling",total_bulk_instrs/proper_saturators,"instrs")
  print("nn_bulk (fail) min",min(spent_in_nn_values_fail),"max",max(spent_in_nn_values_fail),"median",statistics.median(spent_in_nn_values_fail),"mean",statistics.mean(spent_in_nn_values_fail))
  print("nn_bulk_rate_to_satur (fail) min",min(spent_in_nn_rate_values_fail),"max",max(spent_in_nn_rate_values_fail),"median",statistics.median(spent_in_nn_rate_values_fail),"mean",statistics.mean(spent_in_nn_rate_values_fail))
  print("nn_bulk (succ) min",min(spent_in_nn_values_succ),"max",max(spent_in_nn_values_succ),"median",statistics.median(spent_in_nn_values_succ),"mean",statistics.mean(spent_in_nn_values_succ))
  print("nn_bulk_rate_to_satur (succ) min",min(spent_in_nn_rate_values_succ),"max",max(spent_in_nn_rate_values_succ),"median",statistics.median(spent_in_nn_rate_values_succ),"mean",statistics.mean(spent_in_nn_rate_values_succ))

  for suffix, title, ymax, nn_rate_vals, nn_bulk_vals in [
    ("fail", "From failed runs", 420, spent_in_nn_rate_values_fail, spent_in_nn_values_fail),
    ("succ", "From successful runs", 5600, spent_in_nn_rate_values_succ, spent_in_nn_values_succ),
  ]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    fig.suptitle(title)

    ax1.hist(nn_bulk_vals, bins=30, edgecolor='black')
    ax1.set_xlabel('NN eval instructions')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, ymax)
    ax1.set_xlim(-500, 15000)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax2.hist(nn_rate_vals, bins=30, range=(0, 1), edgecolor='black')
    ax2.set_xlabel('NN eval rate (NN / saturation instrs)')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, ymax)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.splitext(res_file_path)[0] + f"_nn_hist_{suffix}.pdf")

  # Stacked bar: problem set breakdown by pipeline stage
  labels = ['Killed in preprocessing', 'Killed in GNN computation',
            'Success filling up passive', 'Killed filling up passive',
            'Saturation ran (failed)', 'Saturation ran (solved)']
  counts = [failed_to_preprocess + killed_during_warmup + warm_but_never_gnn, killed_during_gnn,
            filling_up_passive_success, filling_up_passive_failed,
            proper_saturators_failed, proper_saturators_success]
  colors = ['#d62728', '#ff7f0e', '#bcbd22', '#17becf', '#aec7e8','#2ca02c']

  fig, ax = plt.subplots(figsize=(9, 0.6))
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
  # ax.set_title('Problem set breakdown by pipeline stage')
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -1.0), ncol=3, fontsize='small')
  # fig.subplots_adjust(bottom=0.55)
  plt.savefig(os.path.splitext(res_file_path)[0] + "_breakdown.pdf", bbox_inches='tight')
  # plt.show()
