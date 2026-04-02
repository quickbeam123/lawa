#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import sys

from multiprocessing import Pool

if __name__ == "__main__":
  # To be called as in: ./exporter.py ~/casc2026/loop1_withStratFeaturesEvenToGnn/loop1/loop-model.tar /nfs/sudamar2/snake/casc2026/models/loop1.pt

  in_name = sys.argv[1]
  out_name = sys.argv[2]
  model_input = torch.load(in_name,weights_only=False)
  if len(model_input) == 2:
    (loop,model_state_dict) = model_input
  else:
    model_state_dict = model_input

  model = IC.get_initial_model()
  model.load_state_dict(model_state_dict,strict=False)

  '''
  with torch.no_grad():
    model.clause_valuator_snd[-1].weight.mul_(0.01)
    print("Norm reduced to ",torch.norm(model.clause_valuator_snd[-1].weight).item())
  '''

  IC.export_model(model.state_dict(),out_name)
