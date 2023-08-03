#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # Load (loop,model_state_dict,optimizer_state_dict) as saved by dlooper these days,
  # make sure the tweaky part is adjusted to HP.NUM_TWEAKS
  #
  # To be called as in: ./model_surgeon.py expers6807/pretweaking/loop10/loop-model-and-optimizer.tar 3 expers6807/pretweaking/loop10tw3/loop-model-and-optimizer.tar

  (loop,model_state_dict,optimizer_state_dict) = torch.load(sys.argv[1])

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict)

  new_num_tweaks = int(sys.argv[2])

  model.forceNewNumTweaks(new_num_tweaks)

  optimizer = torch.optim.Adam(model.parameters(), lr=HP.LEARNING_RATE, weight_decay=HP.WEIGHT_DECAY)

  torch.save((loop,model.state_dict(),optimizer.state_dict()), sys.argv[3])