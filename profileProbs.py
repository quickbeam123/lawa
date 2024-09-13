#!/usr/bin/env python3

import sys, os

import pickle
from collections import defaultdict

from multiprocessing import Pool
import subprocess
import atexit

def process_one(task):
  to_run = f"./vampire_rel_ourProfile_8480 -m 8192 -t 120 --input_syntax tptp {task}"

  output = subprocess.getoutput(to_run)

  for line in output.split("\n"):
    features = line.split()
    break

  print(task,features)
  return task,features

if __name__ == "__main__":
  # for every relevant problem, run vampire ourProfile and store the results in a pickle
  # call as in: ./profileProbs.py fofPlain4/problemsFOFbyParseDescJustUNS.txt

  run_on_what = []
  with open(sys.argv[1]) as f:
    for line in f:
      run_on_what.append(line.strip())

  pool = Pool(processes = 100)
  atexit.register(lambda : pool.close())

  results = pool.map(process_one, run_on_what, chunksize = 1)

  as_map = {prob : features for prob,features in results}

  # overwrite probinfo, should be the same anyway
  with open("profiledProbsTPTP-v9.0.0.pkl",'wb') as f:
    pickle.dump(as_map,f)

