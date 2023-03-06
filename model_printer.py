#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Print the module paramaters in cpp-ish format
  #
  # To be called as in: ./model_printer.py ../lawa77/expers6466/experNF12c/loop4/
  
  in_name = sys.argv[1]
  (loop,model_state_dict,optimizer_state_dict) = torch.load(in_name)
  
  torch.set_printoptions(precision=10)

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict,strict=False)

  assert HP.CLAUSE_EMBEDDER_LAYERS == 1

  embedder = model[0]
  nonlinear = model[1]
  key = model[2]

  print("Embedder")
  t = embedder.weight
  ym,xm = t.shape
  print("  weight = {")
  for i in range(ym):
    print("   ",end="")
    for j in range(xm):
      print(t[i,j].item(),end=", " if j != xm-1 else ",\n")
  print("   }")

  print("  bias = {",end="")
  t = embedder.bias
  last = len(t)-1
  for i in range(last+1):
    print(t[i].item(),end=", " if i != last else "}\n")

  print("  followed by",nonlinear)

  print("Key")
  # t = key.weight
  t = key.weight
  ym,xm = t.shape
  print("  kweight = {",)
  for i in range(ym):
    print("   ",end="")
    for j in range(xm):
      print(t[i,j].item(),end=", " if j != xm-1 else ",\n")
  print("   }")
