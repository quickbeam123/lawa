#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch

import warnings
warnings.filterwarnings("ignore", message=r"You are using `torch.load`", category=FutureWarning)

import os, sys, shutil, pickle, random, atexit, time

from collections import defaultdict

def compare_snapshots(snapshot1_path, snapshot2_path):
    # Load state_dicts
    (loop1,state_dict1,optimizer_state_dict1) = torch.load(snapshot1_path)
    (loop2,state_dict2,optimizer_state_dict2) = torch.load(snapshot2_path)

    print("Comparing loops",loop1,"and",loop2)

    # Ensure both snapshots have the same keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    if keys1 != keys2:
        raise ValueError(f"Snapshots have different parameters. Keys in snapshot1 but not snapshot2: {keys1 - keys2}, and vice versa: {keys2 - keys1}")

    # Compare parameters
    differences = {}
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # Compute the difference and its norm
        diff = param1 - param2
        norm_diff = torch.norm(diff).item()
        differences[key] = norm_diff

    # Sort parameters by the norm of their differences
    sorted_differences = sorted(differences.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print("Parameter differences (sorted by norm):")
    for key, norm in sorted_differences:
        print(f"{key}: Norm of difference = {norm}")

    return sorted_differences

if __name__ == "__main__":
  # Run the comparison
  sorted_differences = compare_snapshots(sys.argv[1], sys.argv[2])
