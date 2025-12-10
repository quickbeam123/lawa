#!/usr/bin/env python3

from collections import Counter, defaultdict
import torch,sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  tweak_map_file_name = sys.argv[1]
  mini_tweak_dict_file = sys.argv[2]
  corresponding_winners_file = sys.argv[3]

  tweak_map = torch.load(tweak_map_file_name,weights_only=False)
  mini_tweak_dict = torch.load(mini_tweak_dict_file,weights_only=False)
  winners = torch.load(corresponding_winners_file,weights_only=False)

  hist = defaultdict(int)
  for prob_no_dot, winner in winners.items():
    hist[winner] += 1
  for winner,cnt in sorted(hist.items(),key= lambda x : -x[1]):
    print(winner,cnt)

  names = []
  tweaks = []
  name2idx = {}
  for idx,(name,tweak) in enumerate(tweak_map.items()):
    names.append(name)
    tweaks.append(tweak)
    name2idx[name] = idx

  names = np.array(names)
  X = torch.stack(tweaks).detach().numpy()
  N = X.shape[0]

  if True:
    import plotly.express as px
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    labels = np.zeros(N, dtype=int)
    for prob_no_dot, winner in winners.items():
      labels[name2idx[prob_no_dot]] = winner

    fig = px.scatter_3d(
        x=X3[:,0],
        y=X3[:,1],
        z=X3[:,2],
        color=labels.astype(str),  # cast to string so categorical colors are used
        opacity=0.8,
        size=[4]*len(labels),      # constant marker size
        hover_name=names,          # optional: show names on hover
    )

    fig.update_traces(marker=dict(size=4))  # small dots, like your matplotlib s=2

    fig.update_layout(
        title="3D PCA projection — colored by cluster",
        legend_title="Cluster ID",
    )

    fig.write_html(f"pca3d_cloud_with_winners.html")

    exit(0)