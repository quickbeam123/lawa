#!/usr/bin/env python3

import torch,sys

if __name__ == "__main__":
  new_map = torch.load(sys.argv[1])
  for prob,temp_dict in new_map.items():
    print(prob,temp_dict)