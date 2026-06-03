#!/usr/bin/env python3

import sys, os, torch

if __name__ == "__main__":
  # ./result_reader.py ~/jar2026/split42_boostScale/loop25/train_res.pt

  res_file_path = sys.argv[1]

  (_meta,results) = torch.load(res_file_path,weights_only=False)

  for prob,runs in results.items():
    for (i,ilim,info) in runs:
      if info.status == "uns":
        print(prob,f"iter={i}",info)
