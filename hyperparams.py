#!/usr/bin/env python3

from typing import Final, List

# Data gathering
INSTRUCTION_LIMIT = 5000

# TEMPERATURES = ["0.00","0.25","0.50","0.75","1.00"]
TEMPERATURES = ["0.00","0.125","0.25","0.375","0.50"]

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

FEATURE_SUBSET : Final[int] = FEATURES_RICH

# todo: think of normalization / regularization ...

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # 0 means - just multiply by key at the end (which needs to be of size num_features())
# for NUM_LAYERS > 0 the following internal size is used:
CLAUSE_INTERAL_SIZE : Final[int] = 8

INCLUDE_LSMT : Final[bool] = False
LSMT_LAYERS : Final[int] = 1

# Optimizer - before, there used to be ADAM only
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1

OPTIMIZER = OPTIMIZER_ADAM

LEARNING_RATE : Final[float] = 0.0001
MOMENTUM = 0.9 # only for SGD
WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

DISCOUNT_FACTOR = 0.999

USE_MIN_FOR_LOSS_REDUCE = False

# 16 added discount factor
#
# 17 adding the fix of the bug in learning model which never deleted the selected passive clause
# 18 - the same for rnn

# 19 & 20 - like 17 and 18, but with the new (more reasonable and efficient way of doing loss with classes)
# (typo in TEMPERATURES made it evaluate 0.5 twice - however both runs would have been used for training)

