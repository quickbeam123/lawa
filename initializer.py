#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Create a (randomly intialized) parts-model ready for learning (and running in vampire - after a jit.script export)
  #
  # To be called as in: ./initializer.py model.pt
  
  model = IC.get_initial_model()
  out_name = sys.argv[1]
  torch.save(model, out_name)
  print("Fresh parts-model written to",out_name)