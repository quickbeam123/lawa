#!/usr/bin/env python3

from typing import Final, List

SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

# Data gathering
INSTRUCTION_LIMIT = 5000

# How many times do we try to solve the same problem (and thus to collect a trace for training problems)?
NUM_PERFORMS = {False : 10, True: 3} # how many to look for tweaks and how many to just show the performance of the generalist

# in clooper:
# learn from the last proof you found for this setting
# 0 - don't do it (i.e., only learn from the proofs discovered during this eval)
# >0 - do do it
# >1 - each problem has a score starting at CUMULATIVE and dropping by 1 until 1 every iteration where the problem gets solved...
# ... and increasing by 1 (until the starting max value) if it does not get solved
# Idea: super easy problems will get to 1 (ten times less then max) and stay there
# on the other hand, hard problems will be pulling harder (as long as they stay unsolved)
# in dlooper, for now, just boolean like functionality (no extra multiplier)
CUMULATIVE : Final[int] = 0

# in dlooper, maybe we don't want to parallelize too much
# (after all, all the workers are modifying the same model
# so maybe, let's not be too "hogwild"?)
# specifies the number of cores
TRAINING_PARALLELISM = 20

# also in dlooper:
# for value of 1, we don't repeat eval after first train (that's the old way of doing things, very reinforced)
# for higher values, we wait until the oldest test-eval loss value out of TEST_IMPROVE_WINDOW many
# is the best, retrieve that model (unless it's the first and we would not progress), and finish the loop there
TEST_IMPROVE_WINDOW = 5

# when computing the loss (during validation) or before we actually train (in training), we make a few descent steps just with the tweak part
TWEAK_DESCENT_MAX_SECOND = 10

# if the seems to be taking forever to converge, let's just rerun the perform/gather part
MAX_TEST_IMPROVE_ITER = 30

# the GSD trick - specifieas the dimension of the tweak vector
NUM_TWEAKS = 2

# how much do we value training the generalist and how much the tweaked versions (as a ratio between 0.0 and 1.0)
GENERALIST_TRAINING_WEIGHT = 0.9

# Features
# in the latest lawa vampire, features go in the following order (let's for the time being not experiment with subsets)
# Age,Weight                     1,2
# pLen,nLen                      3,4
# justEq, justNeq                5,6
# numVarOcc,VarOcc/W             7,8
# Sine0,SineMax,SineLevel,   9,10,11
# numSplits                       12
NUM_FEATURES : Final[int] = 12
# todo: think of normalization / regularization ...

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # must be at least 1, to simplify things
# the following internal size is used:
CLAUSE_INTERAL_SIZE : Final[int] = 16

# PROBABLY DON'T WANT TO CHANGE ANYTHING BELOW BESIDES, PERHAPS, THE LEARNING_RATE, FOR NOW

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

LEARNING_RATE : Final[float] = 0.001
TWEAKS_LEARNING_RATE : Final[float] = 0.05

WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
