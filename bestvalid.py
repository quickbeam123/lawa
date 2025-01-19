#!/usr/bin/env python3

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

if __name__ == "__main__":
  # Open a bunch of log files and look for the best validation loss (in the first loop)
  # Print each log files name and the best loss recorded, possibly sorrted by the best loss
  #
  # call as in, e.g.: ./bestvalid.py gnnexperlogs/exper100imit*.log

  for logname in sys.argv[1:]:
    with open(logname, "r") as f:
      bestloss = 1e9
      for line in f:
        if False:
          if line.startswith("loop2"):
            break

        if line.startswith("Eval loss on valid"):
          loss = float(line.split()[-1])
          bestloss = min(bestloss, loss)

      print(logname, bestloss)
