#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Take our parts-model and turn it into a torchscript beast for vampire to interact with
  #
  # To be called as in: ./exporter.py testing/test.pt models/test.pt
  
  in_name = sys.argv[1]
  model = torch.load(in_name)
  print("Loaded parts-model from",in_name)

  out_name = sys.argv[2]  
  IC.export_model(model,out_name)  
  print("Script-model written to",out_name)
  