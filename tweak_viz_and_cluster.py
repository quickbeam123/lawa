#!/usr/bin/env python3

from collections import Counter
import torch,sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  tweak_map_file_name = sys.argv[1]

  tweak_map = torch.load(tweak_map_file_name,weights_only=False)

  names = []
  tweaks = []

  for name,tweak in tweak_map.items():
    names.append(name[13:-2])
    tweaks.append(tweak)

  names = np.array(names)
  data_np = torch.stack(tweaks).detach().numpy()

  # Identify outlier points
  threshold = 30  # can adjust based on your data
  is_outlier = np.any(np.abs(data_np) > threshold, axis=1)
  print(f"Number of outliers detected: {np.sum(is_outlier)}")
  # Remove them
  # TODO: consider saving the old guy
  data_np = data_np[~is_outlier]
  names = names[~is_outlier]

  if True:
    # Run K-Means in the original 256-D space
    k = 8
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(data_np)  # shape (N,)

    print(cluster_labels)
  else:
    import hdbscan
    # --- HDBSCAN clustering ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,   # tune: number of points to form a cluster
        min_samples=None,      # None means equal to min_cluster_size
        metric='euclidean'
    )

    cluster_labels = clusterer.fit_predict(data_np)

    cluster_labels += 1
    print(cluster_labels)

  # Print cluster sizes
  cluster_sizes = np.bincount(cluster_labels)
  k = len(cluster_sizes)
  for i, size in enumerate(cluster_sizes):
    print(f"Cluster {i}: {size} points")

  if True:
    # Step 2: 2D PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca_embedding = pca.fit_transform(data_np)

    # Step 3: Scatter plot of PCA projection
    plt.figure(figsize=(6,6))
    for cluster in range(k):
        mask = cluster_labels == cluster
        plt.scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            s=5,
            label=f'Cluster {cluster}'
        )
    plt.title("2D PCA Projection of 256-D Tensors")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(markerscale=2)
    plt.savefig("pca_projection_clusters.pdf", format="pdf", bbox_inches="tight")
    plt.close()

  if True:
    import plotly.express as px
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X3 = pca.fit_transform(data_np)

    labels = cluster_labels

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

    fig.write_html("pca3d_clusters.html")

    exit(0)

  # Step 3: Loop over UMAP parameters and plot with colors
  for nghbs in [5, 20, 50]: # [5, 10, 15, 20, 30, 50]:
    for md in [0.1, 0.5, 0.8]: # [0.1, 0.2, 0.4, 0.8]:
      reducer = umap.UMAP(
          n_neighbors=nghbs,
          min_dist=md,
          metric='euclidean',
          # random_state=42
      )
      embedding = reducer.fit_transform(data_np)

      # Plot with cluster colors
      plt.figure(figsize=(6,6))
      for cluster in range(k):
          mask = cluster_labels == cluster
          plt.scatter(
              embedding[mask, 0],
              embedding[mask, 1],
              s=2,
              label=f'C{cluster}'
          )
      plt.title(f"UMAP Projection (n_neighbors={nghbs}, min_dist={md})")
      plt.legend(markerscale=2)
      plt.savefig(f"umapping_tweaks_n{nghbs}_mindist{md}.pdf",
                  format="pdf", bbox_inches="tight")
      plt.close()
