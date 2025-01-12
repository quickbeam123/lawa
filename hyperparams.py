#!/usr/bin/env python3

from typing import Final, List

# TODO: clean this folder when not running an experiment from time to time
SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

VAMPIRE_EXECUTABLE = "./vampire_rel_mtpa-gnn_8627"

SATURATION_ALGORITHM = "lrs" # can also be "discount" or "otter" (lrs needs special treatment, to save traces for reproducibility)

PROBLEM_LIST = "problemsSTD.txt"
NUM_TRAIN_PROBLEMS = 15000
EVAL_ON_TEST = False
NUM_TEST_PROBLEMS = 4486 # the rest of current TPTP

# currently not supported with mtpa-gnn!
IMITATE = True # should the first loop use the given clause selection heuristic? (if False, use the usual "-npcc on -ncem ..." with the randomly initialized model)

# Data gathering
INSTRUCTION_LIMIT = 5000
# in elooper:
# This is a reminder that it might make sense to learn from traces we currently (in this loop, with this model) cannot solve
# - such traces, however, are weirdly out of sync with the current model, so some off-policy theory might/should be applied here
# - when set to True, elooper will keep traces of problems not solved in the last loop and still try to learn from them (sometimes)
CUMULATIVE : Final[bool] = True
CUM_STALE_AFTER = 5 # if we can't solve a problem for this many loops, let's give up on it
CUM_MAX_STRENGTH = 2.0
# - a problem is born (when first solved) with a score=0 and natural strength 1.0 = BASE^(score=0)
# - it should be able to reach max strength if not solved from then on in CUM_STALE_AFTER loops,
#   when each time it is not solved, we give him 2 more strength points
# so, roughly BASE^(2*CUM_STALE_AFTER) = CUM_MAX_STRENGTH
# if, on the other hand, a problem gets solve repetitively, its strengh score drops by one, each time this happens

# How many times do we try to solve the same problem (and thus to collect a trace for training problems)?
# - this makes a difference, because we use different seeds (so might get lucky with some and unlucky with others)
# - along similar lines we also used to play with different temperatures (but temp 0.0 on Vampire side, is simply the best)
NUM_PERFORMS = 4

# each subsequent "PERFORM" shall be fed with these given extra options
PERFORMS_SPECIAL = ["", " -npcct 0.01", " -npcct 0.1", " -npcct 1.0"]


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
MAX_TEST_IMPROVE_FIRST_ITER = 100 # this is for the first loop (if you don't like it, set it to the same thing as MAX_TEST_IMPROVE_ITER below)
MAX_TEST_IMPROVE_ITER = 30


# Features
# in the latest lawa vampire, features go in the following order (let's for the time being not experiment with subsets)
# Age,Weight                     1,2
# pLen,nLen                      3,4
# justEq, justNeq                5,6
# numVarOcc,VarOcc/W             7,8
# Sine0,SineMax,SineLevel,   9,10,11
# numSplits                       12
NUM_CLAUSE_FEATURES : Final[int] = 12
# todo: think of normalization / regularization ...

NUM_PROBLEM_FEATURES : Final[int] = 15

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # must be at least 1, to simplify things
# the following internal size is used:
INTERAL_SIZE : Final[int] = 256

GNN_SAGE_PROJECT = False # rather experiment with different Convs
GNN_SAGE_AGGREG = "mean"

GNN_NUM_LAYERS : Final[int] = 10
GNN_INTERNAL_SIZE : Final[int] = 32

NUM_INFERENCE_RULES : Final[int] = 205
GAGE_EMBEDDING_SIZE : Final[int] = 32

GWEIGHT_EMBEDDING_SIZE : Final[int] = 32
GWEIGHT_NUM_VAR_EMBEDS : Final[int] = 1  # THIS is now actually hard-coded on the cpp side!

USE_PROBLEM_FEATURES : Final[bool] = False # True seemed slighly worse on TPTP (let;s not consider this part of the official architecture for now)
USE_SIMPLE_FEATURES : Final[bool] = True
USE_GAGE : Final[bool] = True
USE_GWEIGHT : Final[bool] = True


# PROBABLY DON'T WANT TO CHANGE ANYTHING BELOW BESIDES, PERHAPS, THE LEARNING_RATE, FOR NOW

# only learn from maximum this many clause selection moments along a single trace
MAX_TRAINS_PER_TRACE = 1000

# traces bigger than these will be considered "failed" (and not learned from)
MAX_GAGE_HEIGHT = 500
MAX_GWEIGHT_HEIGHT = 500

# True means the "original" learning setup in which all good clause seletions are rewarded at each step
# False was called "principled" and is more RL-like (whereas the above looks a bit more like training a classfier)
# LEARN_FROM_ALL_GOOD = True
# Time penalty mixing makes more conceptual sense only with "principled" (false)

# a coeff of how much the entropy regularization term should influence the overall loss
# ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
# ENTROPY_NORMALIZED = True

LEARNING_RATE : Final[float] = 0.0001 # 0.0002 seemed a tad better and could become the default for the official experiments
TWEAKS_LEARNING_RATE : Final[float] = 0.1

LEARNING_RATE_DECAY = 0.87055 # (0.5)^(1/5) = halving every five epochs

WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
