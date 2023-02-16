#!/usr/bin/env python3

from typing import Final, List

SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

INTRAOP_PARALLELISM = 10

# Data gathering
INSTRUCTION_LIMIT = 5000
INSTRUCTION_LIMIT_TEST = 5000

# TEMPERATURES = ["1.0"]
# TEMPERATURES = ["0.00","0.25","0.50","0.75","1.00"]
# TEMPERATURES = ["0.00","0.125","0.25","0.375","0.50"]
TEMPERATURES = ["0.0","0.5","1.0"]

# in clooper:
# learn from the last proof you found for this setting
# 0 - don't do it (i.e., only learn from the proofs discovered during this eval)
# >0 - do do it
# >1 - each problem has a score starting at CUMMULATIVE and dropping by 1 until 1 every iteration where the problem gets solved...
# ... and increasing by 1 (until the starting max value) if it does not get solved
# Idea: super easy problems will get to 1 (ten times less then max) and stay there
# on the other hand, hard problems will be pulling harder (as long as they stay unsolved)
# in dlooper, for now, just boolean like functionality (no extra multiplier)
CUMMULATIVE : Final[int] = 1

# only learn from the first proof found for each problem (when traversing the training results in the TEMPERATURES lists)
# in clooper, this might be especially important as otherwise easy problems will train |TEMPERATURES|-times more than
# those solved only by some temps
# (no use in dlooper; yet)
FIRST_PROOF_ONLY = False

# in dlooper, maybe we don't want to parallelize too much
# (after all, all the workers are modifying the same model
# so maybe, let's not be too "hogwild"?)
TRAINING_PARALLELISM = 10



NUM_TRAIN_CYCLES_BETWEEN_EVALS = 10

# also in dlooper:
# for value of 1, we don't repeat eval after first train (that's the old way of doing things, very reinforced)
# for higher values, we wait until the oldest test-eval loss value out of TEST_IMPROVE_WINDOW many
# is the best, retrive that model (unless it's the first and we would not progress), and finish the loop there
TEST_IMPROVE_WINDOW = 1

# if it seems to be taking forever to converge, let's just rerun the perform/gather part
MAX_TEST_IMPROVE_ITER = 30

# there is always going to be (1+NUM_TWEAKS) many copies of the main network in the trained model
# also, each problem will maintain a list of NUM_TWEAKS many tweaks which best describe it
# by convention, we train those tweaks which correspond to active_networks (e.g. ACTIVE_FROM==2 means we train all except the first tweak)
NUM_TWEAKS = 2

# with NUM_TWEAKS > 0, it makes sense to fix some tweaks (as well as the main, default, network) and only train (some of the) tweaky parts
# note the indixing issue: ACTIVE_FROM == 0 means we are training the main newtwork (at index 0), whose formal tweak is the constant 1.0
ACTIVE_FROM = 0

# when evaluating on test problems, where do we get our tweak?
# we do a blind hill climb around what we used last time (with some spread determined below)
# and keep the best scoring one!
NUM_HILL_TRIES_ON_EVAL = 5

# we take the std of the tweaks seen so far (both train and test)
# sample from gauss ball with mean = the last tweak, and std = the above std times the blow factor:
TWEAK_SEARCH_SPREAD_FACTOR = 0.1

# Features
# in the latest lawa vampire, features go in the following order (let's for the time being not experiment with subsets)
# Age,Weight                  2
# Splits,                     3
# posEq,negEq,posNeq,negNeq,  7
# VarOcc,VarOcc/W             8,9
# Sine0,SineMax,SineLevel,    12
# DistVars,DistVars/W         13,14
NUM_FEATURES : Final[int] = 12

# todo: think of normalization / regularization ...

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # must be at least 1, to simplify things
# the following internal size is used:
CLAUSE_INTERAL_SIZE : Final[int] = 8

# True means the "original" learning setup in which all good clause seletions are rewarded at each step
# False was called "principled" and is more RL-like (whereas the above looks a bit more like training a classfier)
LEARN_FROM_ALL_GOOD = True
# Time penalty mixing makes more conceptual sense only with "principled" (false)

# a coeff of how much of a penalty we want to distribute (per successfule run we learn from) in comparison to the reward 1.0 (split across the good selections) we give per problem
TIME_PENALTY_MIXING = 0.0

# a coeff of how much the entropy regularization term should influence the overall loss
ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
ENTROPY_NORMALIZED = True

# Optimizer - before, there used to be ADAM only
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1

OPTIMIZER = OPTIMIZER_ADAM

LEARNING_RATE : Final[float] = 0.001
MOMENTUM = 0.9 # only for SGD
WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
