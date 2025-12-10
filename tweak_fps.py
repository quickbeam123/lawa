#!/usr/bin/env python3

from collections import Counter
import torch,sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  tweak_map_file_name = sys.argv[1]
  tweak_map = torch.load(tweak_map_file_name,weights_only=False)

  names = []
  tweaks = []
  for name,tweak in tweak_map.items():
    names.append(name)
    tweaks.append(tweak)

  names = np.array(names)
  X = torch.stack(tweaks).detach().numpy()

  N = X.shape[0]
  origin = np.zeros(X.shape[1])

  dist_rep = pairwise_distances(X, origin.reshape(1, -1)).flatten()

  k = 8
  reps = []

  perc_from = 80
  perc_to = 90

  for i in range(0, k):
    # Compute percentiles
    perc_from_val = np.percentile(dist_rep, perc_from)
    perc_to_val = np.percentile(dist_rep, perc_to)
    # Get candidate indices where dist_rep is between the 90th and 95th percentile
    candidates = np.where((dist_rep >= perc_from_val) & (dist_rep <= perc_to_val))[0]

    # Debug: print number of candidates
    print(f"p{perc_from}:", perc_from_val, f"p{perc_to}:", perc_to_val,"Num candidates:", len(candidates))

    # Uniform random selection among candidates
    new = np.random.choice(candidates)
    reps.append(new)

    print(f"{i}-th new got dist {dist_rep[new]}")

    plt.figure()
    plt.hist(dist_rep, bins=50)
    plt.xlabel("L2 distance to origin")
    plt.ylabel("Count")
    plt.title("Distance Distribution")
    plt.savefig(f"distance_distrib_{i}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # update distances
    dist_rep = np.minimum(
        dist_rep,
        pairwise_distances(X, X[new:new+1]).flatten()
    )

  mini_tweak_dict = {(his_name := names[new]):tweak_map[his_name] for new in reps}
  # print("mini_tweak_dict",mini_tweak_dict)
  mini_tweak_dict_file = f"selected_{k}_tweaks.tar"
  torch.save(mini_tweak_dict,mini_tweak_dict_file)
  print("Saved to",mini_tweak_dict_file)

  if False:
    import plotly.express as px
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    labels = np.zeros(N, dtype=int)
    for i,rep in enumerate(reps):
      labels[rep] = i+1

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

    fig.write_html(f"pca3d_cloud_with_{k}_dots.html")

    exit(0)