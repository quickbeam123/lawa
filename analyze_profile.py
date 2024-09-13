#!/usr/bin/env python3

import sys, os, math

import pickle
from collections import defaultdict

if __name__ == "__main__":
  with open("profiledProbsTPTP-v9.0.0.pkl",'rb') as f:
    profile = pickle.load(f)

  hist = defaultdict(int)

  for prob,features in profile.items():
    if features[0] == "User":
      continue

    hist[int(math.log10(1+int(features[2])))] += 1

    # if int(features[1])&(2**1):
    #   print(prob)

  print(hist)

  exit(0)

