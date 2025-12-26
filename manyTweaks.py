#!/usr/bin/env python3

from collections import Counter
import torch,sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  raw_many_tweaks = torch.load(sys.argv[1],weights_only=False)

  all_tweaks = []
  for prob,datum in list(raw_many_tweaks.items()):
    print(prob)
    gen_loss,tweak_pairs = datum
    print("  ",gen_loss,sum([tw_loss for tw_loss,tweak in tweak_pairs])/100)

    for tw_loss,tweak in tweak_pairs:
      all_tweaks.append(tweak.numpy())

  all_tweaks = np.array(all_tweaks)

  pca = PCA(n_components=2, random_state=41)
  pca_embedding = pca.fit_transform(all_tweaks)

  plt.figure(figsize=(6,6))
  for prob,datum in list(raw_many_tweaks.items())[15:20]:
    gen_loss,tweak_pairs = datum
    his_tweaks = []
    for tw_loss,tweak in tweak_pairs:
      his_tweaks.append(tweak.numpy())
    his_tweaks = np.array(his_tweaks)

    pca_embeddings_for_prob = pca.transform(his_tweaks)

    plt.scatter(
        pca_embeddings_for_prob[:,0],
        pca_embeddings_for_prob[:,1],
        s=5,
        label=prob
    )
  # show us hero-the-zero
  zero = np.zeros((1,256))
  pca_zero = pca.transform(zero)
  plt.scatter(
        pca_zero[:,0],
        pca_zero[:,1],
        s=10,
        label="zero",
        color="black"
    )
  # plt.legend(markerscale=2)
  plt.savefig("manytweaks_clusters.pdf", format="pdf", bbox_inches="tight")
  plt.close()

