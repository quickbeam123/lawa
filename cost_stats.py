#!/usr/bin/env python3

# Statistics for calibrating HP.COST_ALPHA, on traces processed by trace_good_for_learning (with log_costs as element 8).
#
# Model-free: log_cost over the bad clauses that made it to passive, split into ever selected / never selected.
# Model-dependent, at the learning snapshots (selections with a good clause in passive), as in LearningModel.forward:
#   std   - spread of the logits over the passive set (the scale the cost shift competes with)
#   gap   - top good logit minus top bad logit
#   topc  - log_cost of the top-ranked bad clause (the competitor that decides greedy selection)
#   shift - log(pi~(B)/pi(B)) for COST_ALPHA=1, i.e., the probability-weighted log_cost on the bad side as seen by the loss
# For COST_ALPHA=a, the loss moves the bad side by roughly a*shift (compare with std and gap).
#
# Usage: ./cost_stats.py <loop-model-and-optimizer.tar> <trace_file> [<trace_file> ...]

import sys, torch
import numpy as np

from collections import defaultdict

import inf_common as IC
import hyperparams as HP

from passive_plotter import compute_logits

def summary(vals):
  if len(vals) == 0:
    return "n=0"
  a = np.asarray(vals)
  p50, p90, p99 = np.quantile(a, [0.5, 0.9, 0.99])
  return f"n={len(a)} mean={a.mean():.2f} p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} max={a.max():.2f} frac>0={(a > 0).mean():.2f}"

def clause_costs(journal, num2idx, log_costs):
  added = set()
  selected = set()
  good = set()
  for tag, cl_num, isGood in journal:
    if tag not in [IC.EVENT_ADD, IC.EVENT_SEL] or cl_num not in num2idx:
      continue
    if tag == IC.EVENT_ADD:
      added.add(cl_num)
    else:
      selected.add(cl_num)
    if isGood:
      good.add(cl_num)

  bad = added - good
  def costs_of(cl_nums):
    return log_costs[torch.tensor([num2idx[cn] for cn in cl_nums], dtype=torch.long)].tolist()
  return costs_of(bad & selected), costs_of(bad - selected)

def snapshot_stats(journal, num2idx, logits, log_costs, stride):
  stats = defaultdict(list)
  passive = set()
  passive_good = set()
  num_learn = 0
  for tag, cl_num, isGood in journal:
    if tag in [IC.EVENT_AVATAR_BRANCH, IC.EVENT_AVATAR_REFUTED] or cl_num not in num2idx:
      continue

    idx = num2idx[cl_num]
    if tag == IC.EVENT_ADD:
      passive.add(idx)
      if isGood:
        passive_good.add(idx)
      continue

    if tag == IC.EVENT_SEL and passive_good:
      num_learn += 1
      passive_bad = passive - passive_good
      if passive_bad and num_learn % stride == 0:
        passive_t = torch.tensor(sorted(passive), dtype=torch.long)
        good_t = torch.tensor(sorted(passive_good), dtype=torch.long)
        bad_t = torch.tensor(sorted(passive_bad), dtype=torch.long)

        bad_logits = logits[bad_t]
        bad_costs = log_costs[bad_t]
        if len(passive_t) > 1:
          stats["std"].append(logits[passive_t].std().item())
        stats["gap"].append((logits[good_t].max() - bad_logits.max()).item())
        stats["topc"].append(bad_costs[bad_logits.argmax()].item())
        stats["shift"].append((torch.logsumexp(bad_logits + bad_costs, 0) - torch.logsumexp(bad_logits, 0)).item())

    passive.remove(idx)
    if isGood:
      passive_good.remove(idx)

  return stats

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <loop-model-and-optimizer.tar> <trace_file> [<trace_file> ...]")
    sys.exit(1)

  model_path = sys.argv[1]
  _loop, model_state_dict = torch.load(model_path, weights_only=False)
  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  model.eval()
  print(f"Loaded model (loop {_loop}) from {model_path}")

  totals = defaultdict(list)
  for trace_path in sys.argv[2:]:
    trace_tuple = torch.load(trace_path, weights_only=False)
    if len(trace_tuple) < 9:
      print(f"{trace_path}: no log_costs (processed before COST_ALPHA was introduced), skipping")
      continue
    journal, num_good_selections, num2idx, log_costs = trace_tuple[2], trace_tuple[3], trace_tuple[4], trace_tuple[8]

    with torch.no_grad():
      logits = compute_logits(model, trace_tuple)

    sel, unsel = clause_costs(journal, num2idx, log_costs)
    # like forward, look at no more than about MAX_TRAINS_PER_TRACE snapshots per trace
    stride = max(1, -(-num_good_selections // HP.MAX_TRAINS_PER_TRACE))
    stats = snapshot_stats(journal, num2idx, logits, log_costs, stride)

    print(trace_path)
    print(f"  log_cost bad selected:   {summary(sel)}")
    print(f"  log_cost bad unselected: {summary(unsel)}")
    for key in ["std", "gap", "topc", "shift"]:
      print(f"  {key:5s} (every {stride}. snapshot): {summary(stats[key])}")

    totals["sel"] += sel
    totals["unsel"] += unsel
    for key, vals in stats.items():
      totals[key] += vals

  print("ALL TRACES")
  print(f"  log_cost bad selected:   {summary(totals['sel'])}")
  print(f"  log_cost bad unselected: {summary(totals['unsel'])}")
  for key in ["std", "gap", "topc", "shift"]:
    print(f"  {key:5s}: {summary(totals[key])}")
