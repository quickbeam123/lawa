#!/usr/bin/env python3

from typing import Final, List

# Data gathering
INSTRUCTION_LIMIT = 500
INSTRUCTION_LIMIT_TEST = 500

# TEMPERATURES = ["1.0"]
# TEMPERATURES = ["0.00","0.25","0.50","0.75","1.00"]
# TEMPERATURES = ["0.00","0.125","0.25","0.375","0.50"]
TEMPERATURES = ["0.000","0.500","1.000"]

CUMMULATIVE = False

# only learn from the first proof found for each problem (when traversing the training results in the TEMPERATURES lists)
FIRST_PROOF_ONLY = False

# Features

# the features come (unnormalized) as:
# [0 - age (nat), 1 - length (nat), 2 - weight (nat), 3 - splitWeight (nat), 4 - derivedFromGoal (bool), 5 - sineLevel (char),
# isPureTheoryDescendant (bool), th_ancestors (float+), all_ancestors (float+), th_ancestors/all_ancestors (float-pst)]

# the B-versions below only make sense for NUM_LAYERS == 0 where we only do the dot product with key
# there a bias channel adds one more constant-one feature to every clause

# just age and weight
FEATURES_AW : Final[int] = 0
# features 0 - 3
FEATURES_PLAIN : Final[int] = 1
# features 0 - 5
FEATURES_RICH : Final[int] = 2
# just the whole thing that comes from vampire
FEATURES_ALL : Final[int] = 3
# back to just age and weight - but add square, sqrt and log of each
FEATURES_AW_PLUS : Final[int] = 4
# back to just age and weight - but add square, sqrt and log of each (and their cross multiples)
FEATURES_AW_PLUS_TIMES : Final[int] = 5

FEATURE_SUBSET : Final[int] = FEATURES_RICH

# todo: think of normalization / regularization ...

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # 0 means - just multiply by key at the end (which needs to be of size num_features())
# for NUM_LAYERS > 0 the following internal size is used:
CLAUSE_INTERAL_SIZE : Final[int] = 8

# True means the "original" learning setup in which all good clause seletions are rewarded at each step
# False was called "principled" and is more RL-like (whereas the above looks a bit more like training a classfier)
LEARN_FROM_ALL_GOOD = True
# Time penalty mixing makes more conceptual sense only with "principled" (false)

# a coeff of how much of a penalty we want to distribute (per successfule run we learn from) in comparison to the reward 1.0 (split across the good selections) we give per problem
TIME_PENALTY_MIXING = 0.0

# a coeff of how much the entropy legurazation term should influence the overall loss
ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
ENTROPY_NORMALIZED = True

# whether we normalize (True) loss such that each problem contributes the same (and each proof for a problem)
# or (with False) each experienced time moment contributes the same
PER_PROBLEM_NORMALIZED = False

# Optimizer - before, there used to be ADAM only
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1

OPTIMIZER = OPTIMIZER_ADAM

LEARNING_RATE : Final[float] = 0.0001
MOMENTUM = 0.9 # only for SGD
WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
