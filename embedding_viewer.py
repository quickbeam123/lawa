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

  # Fit PCA on ALL traces so the coordinate system is stable
  all_arrays = [t["embeddings"].numpy() for t in all_traces]
  all_embeddings = np.concatenate(all_arrays, axis=0)
  print(f"Fitting PCA on {all_embeddings.shape[0]} clauses across all {len(all_traces)} traces, dim {all_embeddings.shape[1]}")

  pca = PCA(n_components=3)
  pca.fit(all_embeddings)
  print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

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
    xyz = pca.transform(emb)
    lg = (emb @ final_weight.T).squeeze(-1)
    fig.add_trace(go.Scatter3d(
      x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
      mode='markers',
      marker=dict(size=2, opacity=0.6),
      name=t["trace_file"],
      hovertemplate="logit: %{customdata:.3f}<extra>%{fullData.name}</extra>",
      customdata=lg,
    ))
    total_points += emb.shape[0]

  fig.update_layout(
    title=f"Clause embeddings — 3D PCA ({len(selected)} traces, {total_points} points)",
    scene=dict(
      xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
      yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
      zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
    ),
    legend=dict(itemsizing='constant'),
    margin=dict(l=0, r=0, t=40, b=0),
  )

  fig.write_html(output_path)
  print(f"Saved interactive plot to {output_path}")
