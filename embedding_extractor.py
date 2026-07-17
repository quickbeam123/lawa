#!/usr/bin/env python3

# Extract final clause embeddings (before the last linear layer) from traces.
#
# For each trace, runs the model forward (GNN -> GAGE -> GWEIGHT -> clause_valuator)
# but intercepts the clause_valuator to capture the hidden embedding before the
# final Linear(INTERAL_SIZE, 1) scoring layer.
#
# Usage: ./embedding_extractor.py <loop-model-and-optimizer.tar> <traces_folder> <output.pt>

import sys, os, glob, torch

import inf_common as IC
import hyperparams as HP

def eval_clauses_embeddings(nn, clause_features, gage_embeds, gweight_embeds):
  """Like nn.eval_clauses_logits, but returns the pre-logit embeddings only."""
  feature_parts = []
  if HP.USE_SIMPLE_FEATURES:
    feature_parts.append(clause_features)
  if HP.USE_GAGE:
    feature_parts.append(gage_embeds)
  if HP.USE_GWEIGHT:
    feature_parts.append(gweight_embeds)

  features = torch.cat(feature_parts, dim=1)
  if HP.FEED_STATIC_FEATURES_FINAL_MLP:
    features = features + nn.final_static_tweak

  # Run through all layers except the last one (the Linear -> scalar layer)
  layers = list(nn.clause_valuator.children())
  for layer in layers[:-1]:
    features = layer(features)
  return features  # shape [N, INTERAL_SIZE]

def compute_embeddings(model, trace_tuple):
  """Run model forward on a trace and return embeddings."""
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

  embeddings = eval_clauses_embeddings(
    nn, simple_features_stacked, gage_features, gweight_features
  )
  return embeddings

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <loop-model-and-optimizer.tar> <traces_folder> <output.pt>")
    sys.exit(1)

  model_path = sys.argv[1]
  traces_folder = sys.argv[2]
  output_path = sys.argv[3]

  # Load model
  _loop, model_state_dict = torch.load(model_path, weights_only=False)
  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)
  model.eval()
  print(f"Loaded model (loop {_loop}) from {model_path}")

  # Scan for trace files
  trace_files = sorted(glob.glob(os.path.join(traces_folder, "*.pt")))
  if not trace_files:
    print(f"No .pt files found in {traces_folder}")
    sys.exit(1)
  print(f"Found {len(trace_files)} trace files in {traces_folder}")

  # Extract the final layer weight once from the model (Linear(INTERAL_SIZE, 1, bias=False))
  final_layer = list(model.clause_valuator.children())[-1]
  final_weight = final_layer.weight.detach().clone()  # shape [1, INTERAL_SIZE]
  print(f"Final layer weight shape: {final_weight.shape}")

  results = []
  with torch.no_grad():
    for i, trace_path in enumerate(trace_files):
      trace_tuple = torch.load(trace_path, weights_only=False)
      num2idx = trace_tuple[4]

      embeddings = compute_embeddings(model, trace_tuple)

      results.append({
        "embeddings": embeddings,  # [N, INTERAL_SIZE]
        "trace_file": os.path.basename(trace_path),
        "num_clauses": len(num2idx),
      })

      print(f"  [{i+1}/{len(trace_files)}] {os.path.basename(trace_path)}: "
            f"{len(num2idx)} clauses, embedding shape {embeddings.shape}")

  output = {
    "final_weight": final_weight,  # [1, INTERAL_SIZE] — logits = embeddings @ final_weight.T
    "traces": results,
  }
  torch.save(output, output_path)
  print(f"Saved {len(results)} trace embeddings to {output_path}")
