#!/usr/bin/env python3

from typing import Final, List

# TODO: clean this folder when not running an experiment from time to time
SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

PROBLEM_LIST = "problemsSTD.txt"
NUM_TRAIN_PROBLEMS = 300
NUM_TEST_PROBLEMS = 300

# Data gathering
INSTRUCTION_LIMIT = 5000

# in elooper:
# keep this false (no support yet)
# This is just a reminder it migh make sense to learn from traces we currently (in this loop, with this model) cannot solve
# - such traces, however, are weirdly out of sync with the current model, so some off-policy theory might/should be applied here
# - definitely an interesting direction for a future research
CUMULATIVE : Final[int] = 0

# How many times do we try to solve the same problem (and thus to collect a trace for training problems)?
# - this makes a difference, because we use different seeds (so might get lucky with some and unlucky with others)
# - along similar lines we also used to play with different temperatures (but temp 0.0 on Vampire side, is simply the best)
NUM_PERFORMS = 1


# in elooper, maybe we don't want to parallelize too much
# (after all, all the workers are modifying the same model so maybe, let's not be too "hogwild"?)
# specifies the number of cores used while training a model
TRAINING_PARALLELISM = 20

# also in elooper:
# for value of 1, we don't repeat eval after first train (that's the old way of doing things, very reinforced)
# for higher values, we wait until the oldest valid-eval loss value out of TEST_IMPROVE_WINDOW many
# is the best, retrieve that model (unless it's the first and we would not progress), and finish the loop there
TEST_IMPROVE_WINDOW = 5

# if that seems to be taking forever to converge, let's just rerun the perform/gather part
MAX_TEST_IMPROVE_ITER = 30




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

# a coeff of how much the entropy regularization term should influence the overall loss
ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
ENTROPY_NORMALIZED = True

LEARNING_RATE : Final[float] = 0.001
TWEAKS_LEARNING_RATE : Final[float] = 0.1

WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
