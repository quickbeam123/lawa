#!/usr/bin/env python3

# Plot passive set evolution for a given trace and model.
#
# X-axis: selection step (every EVENT_SEL, faithful to abstract time)
# Y-axis-1 (left): position of each good clause in sorted_passive (0 = top);
#   shaded area shows total passive set size
# Y-axis-2 (right): logit of the clause being selected at each step
#
# Usage: ./passive_plotter.py <loop-model-and-optimizer.tar> <trace_file> <output.pdf>

import sys, os, torch

from collections import defaultdict
from sortedcontainers import SortedList

import inf_common as IC
import hyperparams as HP

EVENT_ADD = IC.EVENT_ADD
EVENT_REM = IC.EVENT_REM
EVENT_SEL = IC.EVENT_SEL

def compute_logits(model, trace_tuple):
  """Run model forward just enough to get logits for all clauses."""
  (static_features, simple_features_stacked, journal, num_good_selections,
   num2idx, gnn_data, gage_data, gweight_data) = trace_tuple[:8]

  init_gnn_nodes, gnn_edges, gnn_init_clause_nums = gnn_data

  learn_model = IC.LearningModel(False, model, trace_tuple)
  learn_model.eval()
  nn = learn_model.nn

  nn.computing = True
  nn.set_static_features(static_features)

  initial_clause_gage = None
  gweight_symbol_embeds = None
  if HP.USE_GAGE or HP.USE_GWEIGHT:
    nn.gnn_nodes = init_gnn_nodes
    nn.gnn_edges = gnn_edges
    initial_clause_gage, gweight_symbol_embeds = nn.gnn_perform(gnn_init_clause_nums)

  gage_features = IC.run_vectorized_gage(
    initial_clause_gage, gage_data, nn.gage_rule_embed, nn.gage_combine, False
  ) if HP.USE_GAGE else None

  gweight_features = IC.run_vectorized_gweight(
    gweight_symbol_embeds, nn.gweight_var_embed, gweight_data, nn.gweight_term_combine, False
  ) if HP.USE_GWEIGHT else None

  logits = nn.eval_clauses_logits(simple_features_stacked, gage_features, gweight_features)
  return logits

def replay_journal(journal, num2idx, logits):
  """Replay journal events, collecting plot data at every EVENT_SEL."""
  passive = set()
  passive_good = set()
  sorted_passive = SortedList(key=lambda cl_idx: -logits[cl_idx].item())

  sel_step = 0
  passive_sizes = []
  selected_logits = []
  good_traces = defaultdict(list)  # idx -> list of (sel_step, position)

  for tag, cl_num, isGood in journal:
    if cl_num not in num2idx:
      # print(f"sel_step = {sel_step}: skipping {cl_num} with no index")
      continue

    idx = num2idx[cl_num]
    if tag == EVENT_ADD:
      # print(f"sel_step = {sel_step}: adding clause of idx {idx} and logit {logits[idx].item()}")
      passive.add(idx)
      sorted_passive.add(idx)
      if isGood:
        passive_good.add(cl_num)
        # print("  which is a good clause")
      continue

    if tag == EVENT_SEL:
      # print(f"sel_step = {sel_step}: about to select a clause {idx}")
      passive_sizes.append(len(sorted_passive)+0.5) # +0.5 to better "cover" all the clauses inside visually
      selected_logits.append(logits[idx].item())
      if sorted_passive[0] != idx:
        print("Warning: EVENT_SEL for a clause not considered the best - perhaps running with a wrong model?")
        print(f"  logit selected {logits[idx].item()}, logit best {logits[sorted_passive[0]].item()}")
      for good_num in passive_good:
        pos = sorted_passive.index(num2idx[good_num])
        good_traces[good_num].append((sel_step, pos+1)) # starting from 1, for a better visual rendering
        # print(f"    good_idx = {good_idx} is at {pos}")

      # print(f"  passive_sizes = {len(sorted_passive)}, selected_logit = {logits[idx].item()}")
      sel_step += 1
    else:
      pass
      # print(f"sel_step = {sel_step}: removing a clause of idx {idx}")

    passive.remove(idx)
    sorted_passive.remove(idx)
    if isGood:
      passive_good.remove(cl_num)

  return good_traces, passive_sizes, selected_logits

def plot_passive_evolution(good_traces, passive_sizes, selected_logits, plot_path):
  """Create and save the passive set evolution plot."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  steps = list(range(len(passive_sizes)))

  fig, ax1 = plt.subplots(figsize=(14, 7))
  ax2 = ax1.twinx()

  # shaded area for total passive set size
  if False:
    ax1.fill_between(steps, 0.5, passive_sizes, alpha=0.15, color='gray', label='passive size')

  # one color per good clause, plot position segments
  all_clauses = sorted(good_traces.keys())
  cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(all_clauses), 1))
  for ci, cl_idx in enumerate(all_clauses):
    color = cmap(ci % 20)
    trace = good_traces[cl_idx]
    # split into contiguous segments (consecutive sel_step values)
    segments = []
    seg_steps = [trace[0][0]]
    seg_pos = [trace[0][1]]
    for i in range(1, len(trace)):
      if trace[i][0] == trace[i-1][0] + 1:
        seg_steps.append(trace[i][0])
        seg_pos.append(trace[i][1])
      else:
        segments.append((seg_steps, seg_pos))
        seg_steps = [trace[i][0]]
        seg_pos = [trace[i][1]]
    segments.append((seg_steps, seg_pos))
    for si, (ss, sp) in enumerate(segments):
      if len(ss) == 1:
        ax1.plot(ss, sp, marker='o', markersize=4, color=color, linestyle='None', alpha=0.7,
                 label=f'clause {cl_idx}' if si == 0 else None)
      else:
        ax1.plot(ss, sp, color=color, linewidth=3.0, alpha=0.7,
                 label=f'clause {cl_idx}' if si == 0 else None)

  ax1.set_xlabel('selection step')
  ax1.set_ylabel('position in passive (1 = top)')
  ax1.set_ylim(bottom=0)

  # selected logit on second y-axis
  if True:
    ax2.plot(steps, selected_logits, color='green', linewidth=0.5, linestyle='--', alpha=0.5, label='selected logit')
    ax2.set_ylabel('logit of selected clause', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

  # combined legend
  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  max_legend = 25
  if len(h1) > max_legend:
    h1, l1 = h1[:max_legend], l1[:max_legend]
  ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize='small', ncol=2)
  ax1.legend(h1, l1, loc='upper right', fontsize='small', ncol=2)

  fig.tight_layout()
  fig.savefig(plot_path, dpi=150)
  plt.close(fig)

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <loop-model-and-optimizer.tar> <trace_file> <output.pdf>")
    sys.exit(1)

  model_path = sys.argv[1]
  trace_path = sys.argv[2]
  output_path = sys.argv[3]

  # load model
  _loop,model_state_dict = torch.load(model_path, weights_only=False)
  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  print(f"Loaded model (loop {_loop}) from {model_path}")

  # load trace (already processed by trace_good_for_learning)
  trace_tuple = torch.load(trace_path, weights_only=False)
  journal = trace_tuple[2]
  num2idx = trace_tuple[4]
  print(f"Loaded trace from {trace_path}: {len(num2idx)} clauses, {len(journal)} journal events")

  with torch.no_grad():
    model.eval()
    logits = compute_logits(model, trace_tuple)

  good_traces, passive_sizes, selected_logits = replay_journal(journal, num2idx, logits)
  print(f"Replayed {len(passive_sizes)} selection steps, tracking {len(good_traces)} good clauses")

  plot_passive_evolution(good_traces, passive_sizes, selected_logits, output_path)
  print(f"Saved plot to {output_path}")
