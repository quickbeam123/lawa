#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import torch

# Architecture
CLAUSE_NUM_FEATURES = 10
CLAUSE_INTERAL_SIZE = 16

# Optimizer
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0 # Corresponds to L2 regularization