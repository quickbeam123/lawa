#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

from multiprocessing import Pool

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)


if __name__ == "__main__":
  # Open a trace a show what's inside
  #
  # To be called as in: ./trace_viewer.py trace_file_name.pt

  trace_file = sys.argv[1]

  (clause_features,journal,num_selections) = torch.load(trace_file)

  print(len(clause_features))
  print(len(journal))
  print(num_selections)
