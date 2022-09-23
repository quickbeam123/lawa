#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Print the module paramaters in cpp-ish format
  #
  # To be called as in: ./model_printer.py expers/exper50/loop13/parts-model.pt 
  
  in_name = sys.argv[1]
  model = torch.load(in_name)
  
  torch.set_printoptions(precision=10)

  embedder = model[0]
  key = model[1]

  print("Embedder")
  t = embedder[0].weight
  ym,xm = t.shape
  print("  weight = {")
  for i in range(ym):
    print("   ",end="")
    for j in range(xm):
      print(t[i,j].item(),end=", " if j != xm-1 else ",\n")
  print("   }")

  print("  bias = {",end="")
  t = embedder[0].bias
  last = len(t)-1
  for i in range(last+1):
    print(t[i].item(),end=", " if i != last else "}\n")

  print("  followed by",embedder[1])

  print("Key")
  # t = key.weight
  t = key.weight[0]
  print("  kweight = {",end="")
  last = len(t)-1
  for i in range(last+1):
    print(t[i].item(),end=", " if i != last else "}\n")
