#!/usr/bin/env python3

from typing import Final, List

# Data gathering
INSTRUCTION_LIMIT = 10000
INSTRUCTION_LIMIT_TEST = 5000

# TEMPERATURES = ["1.0"]
# TEMPERATURES = ["0.00","0.25","0.50","0.75","1.00"]
# TEMPERATURES = ["0.00","0.125","0.25","0.375","0.50"]
TEMPERATURES = ["0.000","0.125","0.250","0.500","1.000"]

CUMMULATIVE = False

# only learn from the first proof found for each problem (when traversing the training results in the TEMPERATURES lists)
FIRST_PROOF_ONLY = False

DIFFICULTY_BOOST_COEF = 0.0
DIFFICULTY_BOOST_CAP = 3.0

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

# The first version used at AITP: faithul passive at each step, learns from all positive actions 
LEARNER_ORIGINAL : Final[int] = 0
# faithul passive at each step, but only learns from the action taken (may include penalty for expensive steps)
LEARNER_PRINCIPLED : Final[int] = 2

LEARNER = LEARNER_ORIGINAL

# allows for learning more than one clause embedding (works with LEARNER_ORIGINAL and LEARNER_PRINCIPLED)
NUM_EFFECTIVE_QUEUES : Final[int] = 1

# with more than one positive action, how is the NLL loss distributed between them? False means average, True means minimum (only makes sense with LEARNER_ORIGINAL and LEARNER_RECURRENT)
USE_MIN_FOR_LOSS_REDUCE = False

ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
ENTROPY_NORMALIZED = True

# a ceoff of how much of a penalty we want to distribute (per successfule run we learn from) in comparison to the reward 1.0 (split across the good selections) we give per problem
TIME_PENALTY_MIXING = 0.0

# Optimizer - before, there used to be ADAM only
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1

OPTIMIZER = OPTIMIZER_ADAM

LEARNING_RATE : Final[float] = 0.0001
MOMENTUM = 0.9 # only for SGD
WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# To be experimented with later (does not make much sense with LEARNER_ORIGINAL/LEARNER_RECURRENT; no sense at all for LEARNER_ENIGMA)
DISCOUNT_FACTOR = 1.0
# Around 100 activations done in 5000Mi and 0.99^100 = 0.36
# Around 135 activations done in 10000Mi and 0.995^100 = 0.5
