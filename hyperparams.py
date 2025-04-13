#!/usr/bin/env python3

from typing import Final, List

SNAKE_INPUT_DIRS = ["../snake/mtpa2025/evals",
                    "../snake/mtpa2025/evals74",
                    "../snake/mtpa2025/evals79",
                    "../snake/mtpa2025/evals_2",
                    "../snake/mtpa2025/evals74_2",
                    "../snake/mtpa2025/evals79_2",
                    "../snake/mtpa2025/evals74_3",
                    "../snake/mtpa2025/evals79_3",
                    "../snake/mtpa2025/evals74_4",
                    "../snake/mtpa2025/evals79_4",
                    "../snake/mtpa2025/evals79_5"]

SNAKE_SORT_BY_INSTR: Final[bool] = True # instead of random strat, let's prefer strats that solve the problem fastests
SNAKE_MAX_TRACES_PER_PROBLEM = 1
# don't even try to look for a solution that originally took longer than this
SNAKE_MAX_INSTRUCTIONS = 100000

SNAKE_MAX_FAULS = 5 # how many strategies to try per problem, if they seem to be failing for "Too big or trivial" reason
SNAKE_MAX_TRIES = 6 # for particular strategy, try this many times shuffled (and the last one will be unshuffled)




# TODO: clean this folder when not running an experiment from time to time
SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

VAMPIRE_EXECUTABLE = "./vampire_rel_mtpa-gnn_8902"
SHUFFLING_OPTIONS = "-si on -rtra on" # set to empty for no shuffling

SATURATION_ALGORITHM = "lrs" # can also be "discount" or "otter" (lrs needs special treatment, to save traces for reproducibility)

PROBLEM_LIST = "problemsSTDclean.txt"
NUM_TRAIN_PROBLEMS = 15000
EVAL_ON_TEST = True
NUM_TEST_PROBLEMS = 4477 # the rest of current TPTP

# currently not supported with mtpa-gnn!
IMITATE = True # should the first loop use the given clause selection heuristic? (if False, use the usual "-npcc on -ncem ..." with the randomly initialized model)
NON_IMIT_EXTRA = " -lpd off"

# Data gathering
INSTRUCTION_LIMIT = 10000
# in elooper:
# This is a reminder that it might make sense to learn from traces we currently (in this loop, with this model) cannot solve
# - such traces, however, are weirdly out of sync with the current model, so some off-policy theory might/should be applied here
# - when set to True, elooper will keep traces of problems not solved in the last loop and still try to learn from them (sometimes)
CUMULATIVE : Final[bool] = False
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
NUM_PERFORMS = 1
KEEP_ALL_TRACES = False

# each subsequent "PERFORM" shall be fed with these given extra options
PERFORMS_SPECIAL = ["", " -npcct 0.037", " -npcct 0.111", " -npcct 0.333", " -npcct 1.0"]


# in elooper, maybe we don't want to parallelize too much
# (after all, all the workers are modifying the same model so maybe, let's not be too "hogwild"?)
# specifies the number of cores used while training a model
TRAINING_PARALLELISM = 60
WORTH_REPORTING = 60 # more than this many seconds and a new line goes into detailed.log file in exper_dir

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

GNN_NUM_LAYERS : Final[int] = 5
GNN_MULTIPLIER : Final[int] = 1
GNN_INTERNAL_SIZE : Final[int] = 32

GNN_DROPOUT : Final[float] = 0.0

NUM_INFERENCE_RULES : Final[int] = 205
GAGE_EMBEDDING_SIZE : Final[int] = 32

GWEIGHT_EMBEDDING_SIZE : Final[int] = 32
# GWEIGHT_NUM_VAR_EMBEDS : Final[int] = 1  # THIS is now actually hard-coded on the cpp side!

TREE_DROPOUT : Final[float] = 0.0 # maybe is good, but also contributes to higher variance (ingore by default)

USE_PROBLEM_FEATURES : Final[bool] = True # means they are fed to the final MLP; actually helps a bit
USE_SIMPLE_FEATURES : Final[bool] = True
USE_GAGE : Final[bool] = True
USE_GWEIGHT : Final[bool] = True

FEED_PROBLEM_FEAUTURES_TO_GNN: Final[bool] = True # additionally feed these features already to the GNN
FEED_PROBLEM_FEAUTURES_TO_THE_TREES: Final[bool] = True

# PROBABLY DON'T WANT TO CHANGE ANYTHING BELOW BESIDES, PERHAPS, THE LEARNING_RATE, FOR NOW

# only learn from maximum this many clause selection moments along a single trace
MAX_TRAINS_PER_TRACE = 1000

# traces bigger than these will be considered "failed" (and not learned from)
MAX_GAGE_HEIGHT = 500
MAX_GWEIGHT_HEIGHT = 500
MAX_BOX_SIZE = 95000
MAX_KBSIZE = 150000

# True means the "original" learning setup in which all good clause seletions are rewarded at each step
# False was called "principled" and is more RL-like (whereas the above looks a bit more like training a classfier)
# LEARN_FROM_ALL_GOOD = True
# Time penalty mixing makes more conceptual sense only with "principled" (false)

# a coeff of how much the entropy regularization term should influence the overall loss
# ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
# ENTROPY_NORMALIZED = True

LEARNING_RATE : Final[float] = 0.0002 # 0.0002 seemed a tad better and could become the default for the official experiments
TWEAKS_LEARNING_RATE : Final[float] = 0.1

LEARNING_RATE_DECAY = 0.87055 # (0.5)^(1/5) = halving every five epochs

WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# should be use AVG (False) or the MAX (True) for picking, at each time moment, which good action to reinforce?
GOOD_LOGIT_MAX : Final[bool] = False # GOOD_LOGIT_MAX (nor MIN for that matter) seemed better than the default AVG (so we preach complete democracy in this regard in the paper)
# also sticking with democracy in what regards the time stamp of the experience along the trace. But early and late decisions matter (similarly); at least so it seemed in one abandoned experiment

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
