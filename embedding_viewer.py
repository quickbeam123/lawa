#!/usr/bin/env python3

# Visualize clause embeddings from embedding_extractor.py output via 3D PCA.
#
# Picks a random subset of traces, projects all their embeddings into 3D via PCA,
# and produces an interactive HTML scatter plot (one color per trace).
#
# Usage: ./embedding_viewer.py <embeddings.pt> <output.html> [num_traces]
#
# num_traces: how many traces to sample (default 10)

import sys, random
import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Visualization mode:
#   "pca"        — plain 3D PCA on raw embeddings
#   "logit_null" — X = logit direction, Y/Z = 2D PCA of null space
#   "null"       — 3D PCA on null space only (logit direction removed)
MODE = "logit_null"

if __name__ == "__main__":
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(f"Usage: {sys.argv[0]} <embeddings.pt> <output.html> [num_traces=10]")
    sys.exit(1)

  input_path = sys.argv[1]
  output_path = sys.argv[2]
  num_traces = int(sys.argv[3]) if len(sys.argv) == 4 else 10

  # Load embeddings produced by embedding_extractor.py
  data = torch.load(input_path, weights_only=False)
  final_weight = data["final_weight"].numpy()  # [1, D]
  all_traces = data["traces"]
  print(f"Loaded {len(all_traces)} traces, final_weight shape {final_weight.shape}")

  # Collect all embeddings for fitting PCA (on the full dataset)
  all_arrays = [t["embeddings"].numpy() for t in all_traces]
  all_embeddings = np.concatenate(all_arrays, axis=0)
  print(f"Total clauses: {all_embeddings.shape[0]} across {len(all_traces)} traces, dim {all_embeddings.shape[1]}")

  # Shared null-space projection for "logit_null" and "null" modes
  if MODE in ("logit_null", "null"):
    w = final_weight.squeeze(0)            # [D]
    w_unit = w / np.linalg.norm(w)         # [D]
    logit_proj = all_embeddings @ w_unit   # [N]
    nullspace = all_embeddings - np.outer(logit_proj, w_unit)  # [N, D]

  if MODE == "pca":
    pca = PCA(n_components=3)
    pca.fit(all_embeddings)
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

    def project(emb):
      xyz = pca.transform(emb)
      return xyz[:, 0], xyz[:, 1:]

    axis_labels = dict(
      xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
      yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
      zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
    )
    mode_label = "3D PCA"

  elif MODE == "logit_null":
    pca_null = PCA(n_components=2)
    pca_null.fit(nullspace)
    print(f"Null-space PCA explained variance ratios: {pca_null.explained_variance_ratio_}")

    def project(emb):
      lp = emb @ w_unit
      ns = emb - np.outer(lp, w_unit)
      ns_2d = pca_null.transform(ns)
      return lp, ns_2d

    axis_labels = dict(
      xaxis_title="logit direction",
      yaxis_title=f"null PC1 ({pca_null.explained_variance_ratio_[0]:.1%})",
      zaxis_title=f"null PC2 ({pca_null.explained_variance_ratio_[1]:.1%})",
    )
    mode_label = "logit + null-space PCA"

  elif MODE == "null":
    pca_null = PCA(n_components=3)
    pca_null.fit(nullspace)
    print(f"Null-space PCA explained variance ratios: {pca_null.explained_variance_ratio_}")

    def project(emb):
      ns = emb - np.outer(emb @ w_unit, w_unit)
      xyz = pca_null.transform(ns)
      return xyz[:, 0], xyz[:, 1:]

    axis_labels = dict(
      xaxis_title=f"null PC1 ({pca_null.explained_variance_ratio_[0]:.1%})",
      yaxis_title=f"null PC2 ({pca_null.explained_variance_ratio_[1]:.1%})",
      zaxis_title=f"null PC3 ({pca_null.explained_variance_ratio_[2]:.1%})",
    )
    mode_label = "null-space 3D PCA"

  else:
    print(f"Unknown MODE: {MODE}")
    sys.exit(1)

  # Sample a random subset of traces for display
  if num_traces >= len(all_traces):
    selected = all_traces
  else:
    selected = random.sample(all_traces, num_traces)
  print(f"Selected {len(selected)} traces for visualization")

  # Build plotly figure, one scatter trace per problem trace
  fig = go.Figure()
  total_points = 0
  for t in selected:
    emb = t["embeddings"].numpy()
    x_vals, yz_vals = project(emb)
    lg = (emb @ final_weight.T).squeeze(-1)
    ages = t["ages"].numpy()
    weights = t["weights"].numpy()
    hover_data = np.column_stack([lg, ages, weights])  # [N, 3]
    fig.add_trace(go.Scatter3d(
      x=x_vals, y=yz_vals[:, 0], z=yz_vals[:, 1],
      mode='markers',
      marker=dict(size=2, opacity=0.6),
      name=t["trace_file"],
      hovertemplate="logit: %{customdata[0]:.3f}, age: %{customdata[1]:.1f}, weight: %{customdata[2]:.1f}<extra>%{fullData.name}</extra>",
      customdata=hover_data,
    ))
    total_points += emb.shape[0]

  fig.update_layout(
    title=f"Clause embeddings — {mode_label} ({len(selected)} traces, {total_points} points)",
    scene=axis_labels,
    legend=dict(itemsizing='constant'),
    margin=dict(l=0, r=0, t=40, b=0),
  )

  fig.write_html(output_path)
  print(f"Saved interactive plot to {output_path}")
