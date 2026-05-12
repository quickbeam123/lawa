#!/usr/bin/env python3

from typing import Final, List

# this is a non-default path in which we create/updat our trace index by looking into SNAKE_INPUT_DIRS and extracting runs from there
SNAKE_STYLE_GATHER = True

SNAKE_INPUT_DIRS = [
    ["../snake/casc2026/evalsN_14","../snake/casc2026/evalsN_15"],
    ["../snake/casc2026/evalsN_11","../snake/casc2026/evalsN_12","../snake/casc2026/evalsN_13"], # loop 6
    ["../snake/casc2026/evalsN_10"],                                                             # loop 5
    ["../snake/casc2026/evalsN_9"],                                                              # loop 4
    ["../snake/casc2026/evalsN_8"],                                                              # loop 3
    ["../snake/casc2026/evalsN_7"],                                                              # all5champsBiggishL14
    ["../snake/casc2026/evalsN_3","../snake/casc2026/evalsN_4","../snake/casc2026/evalsN_5"],    # loop 2
    ["../snake/casc2026/evalsN_1","../snake/casc2026/evalsN_2"],                                 # loop 1
    ["../snake/casc2026/evals_1"],                                                               # plain
                   ]

SNAKE_MAX_TRACES_PER_PROBLEM = 12

# don't even try to look for a solution that originally took longer than this
SNAKE_MAX_INSTRUCTIONS = 150000
SNAKE_RECONSTRUCT_TIMEOUT_SECOND = 120

SNAKE_MAX_TRIES = 6 # for particular strategy, try this many times shuffled (and the last one will be unshuffled)


# TODO: clean this folder when not running an experiment from time to time
SCRATCH = "/home/sudamar2/scratch" # used to be: "/scratch/sudamar2/" # add /raid/. for dgx

RANDOM_SEED: Final[int] = 42

VAMPIRE_EXECUTABLE = "./vampire_rel_mtpa-gnn-2026_10730"

SHUFFLING_OPTIONS = "-si on -rtra on" # set to empty for no shuffling

RANDOMIZED_STRATEGIES = None # set to a sampler file like ""samplerFOL.txt"" if you want random strategies
# only if RANDOMIZED_STRATEGIES is None does the SATURATION_ALGORITHM below kick in!
SATURATION_ALGORITHM = "lrs" # can also be "discount" or "otter" (lrs needs special treatment, to save traces for reproducibility)

PROBLEM_LIST = "problemsCNFFOFTF0noSATnoARI.txt" # newly, we don't want to run on SAT
NUM_TRAIN_PROBLEMS = 12400 # 15500 is the full set
EVAL_ON_TEST = True
NUM_TEST_PROBLEMS = 3100 # the rest of current TPTP

# currently not supported with mtpa-gnn!
IMITATE = True # should the first loop use the given clause selection heuristic? (if False, use the usual "-npcc on -ncem ..." with the randomly initialized model)
NON_IMIT_EXTRA = " -lpd off"

# Data gathering - this luby-iterates between MIN and MAX and then repeats, if (INITIAL_)NUM_PERFORMS needs more
# typically, one does the luby thing only under RANDOMIZED_STRATEGIES != None
INSTRUCTION_LIMIT = 16000
INSTRUCTION_LIMIT_MIN = INSTRUCTION_LIMIT
INSTRUCTION_LIMIT_MAX = INSTRUCTION_LIMIT
# use the same value, to just have one value

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
INITIAL_NUM_PERFORMS = 5
NUM_PERFORMS = 5
MAX_TRACES_TO_KEEP = 1 # should be at least 1!
# setting the above to different values makes sense when running in "snake"-mode (then, e.g., INITIAL_NUM_PERFORMS = 135 , NUM_PERFORMS = 45, MAX_TRACES_TO_KEEP = 3)

SKEW_LOCAL_FACT = False

USE_SPECIAL = True
# each subsequent "PERFORM" shall be fed with these given extra options
PERFORMS_SPECIAL = ["", " -npcct 0.037", " -npcct 0.111", " -npcct 0.333", " -npcct 1.0"]

TWEAKS_AS_BIAS = False

USE_TWEAKING = False # instead of looking things up in PERFORMS_SPECIAL, we just keep increasing an argument to ncem_gsd

TWEAK_MATRIX_SIZE = 256
TWEAKS_TO_PICK = 0

TRAIN_MINIBATCH_SIZE : Final[int] = 1 # accumulate gradients from this many workers before one optimizer step

# in elooper, maybe we don't want to parallelize too much
# (after all, all the workers are modifying the same model so maybe, let's not be too "hogwild"?)
# specifies the number of cores used while training a model
# EVAL_PARALLELISM = 16 # should be TRAINING_PARALLELISM / NUM_GSD_FEATURES, but I think I can afford a bit leeway
TRAINING_PARALLELISM = 16
WORTH_REPORTING = 120 # more than this many seconds and a new line goes into detailed.log file in exper_dir

# also in elooper:
# for value of 1, we don't repeat eval after first train (that's the old way of doing things, very reinforced)
# for higher values, we wait until the oldest valid-eval loss value out of TEST_IMPROVE_WINDOW many
# is the best, retrieve that model (unless it's the first and we would not progress), and finish the loop there
TEST_IMPROVE_WINDOW = 8

NUM_EARLY_STOP_ROUNDS = 2

EARLY_STOP_ON = "dist_to_good"

ANYWAY_STEP_ALL = True # True seems to be better, False ( = "take just one") was the long standing default

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

# these two together are the STATIC features for a particular vampire run
NUM_PROBLEM_FEATURES : Final[int] = 15
NUM_STRATEGY_FEATURES : Final[int] = 30
NUM_GSD_FEATURES : Final[int] = 8

# non-linearity
USE_SILU : Final[bool] = True

# normalizations
USE_RMS : Final[bool] = True # False means LayerNorm

# Architecture
CLAUSE_EMBEDDER_LAYERS : Final[int] = 1  # must be at least 1, to simplify things
# the following internal size is used:
INTERAL_SIZE : Final[int] = 256 # "big" is 384

GNN_SAGE_PROJECT = False # rather experiment with different Convs
GNN_SAGE_AGGREG = "mean"

GNN_NUM_LAYERS : Final[int] = 5 # "big" is 8
GNN_INTERNAL_SIZE : Final[int] = 32 # "big" is 48

GNN_DROPOUT : Final[float] = 0.1

NUM_INFERENCE_RULES : Final[int] = 202
GAGE_EMBEDDING_SIZE : Final[int] = 32 # "big" is 48

GWEIGHT_EMBEDDING_SIZE : Final[int] = 32 # "big" is 48
# GWEIGHT_NUM_VAR_EMBEDS : Final[int] = 1  # THIS is now actually hard-coded on the cpp side!

LIT_TO_CLAUSE_AGGREG = "sum"

TREE_DROPOUT : Final[float] = 0.0 # maybe is good, but also contributes to higher variance (ingore by default)

USE_SIMPLE_FEATURES : Final[bool] = True
USE_GAGE : Final[bool] = True
USE_GWEIGHT : Final[bool] = True

FINAL_LAYER_DROPOUT : Final[float] = 0.0

# these are kind of more or less ignored (vampire will always tell the model everything), but the model may decide to ignore (see below)
USE_STRATEGY_FEATURES : Final[bool] = True
USE_PROBLEM_FEATURES : Final[bool] = False
USE_GSD : Final[bool] = False

# this is the main flag for STRATEGY and PROBLEM usage, if set to true, all the three below will trigger and start producing tweeks in the respective part of the network
USE_STATIC_FEATURES : Final[bool] = True
FEED_STATIC_FEAUTURES_TO_GNN: Final[bool] = True # additionally feed these features already to the GNN
FEED_STATIC_FEAUTURES_TO_THE_TREES: Final[bool] = False # currently does nothing (consider FiLM-style conditioning in the future)
FEED_STATIC_FEATURES_FINAL_MLP: Final[bool] = USE_STATIC_FEATURES

# PROBABLY DON'T WANT TO CHANGE ANYTHING BELOW BESIDES, PERHAPS, THE LEARNING_RATE, FOR NOW

# only learn from maximum this many clause selection moments along a single trace
MAX_TRAINS_PER_TRACE = 1000

LABEL_SMOOTHING = 0.0

ONLY_LEARN_FROM_EVER_SELECTED = True

# traces bigger than these will be considered "failed" (and not learned from)
MAX_GAGE_HEIGHT = 1000
MAX_GWEIGHT_HEIGHT = 1000
MAX_BOX_SIZE = 200000
MAX_KBSIZE = 150000 # "big" is 100000

# True means the "original" learning setup in which all good clause seletions are rewarded at each step
# False was called "principled" and is more RL-like (whereas the above looks a bit more like training a classfier)
# LEARN_FROM_ALL_GOOD = True
# Time penalty mixing makes more conceptual sense only with "principled" (false)

# a coeff of how much the entropy regularization term should influence the overall loss
# ENTROPY_COEF = 0.0
# next time I play with the entropy regularization, let me try the normalized one
# ENTROPY_NORMALIZED = True

# all the GSD hyperparams only make sense with inf_common_GSD.py (parked in lawa-devel for now)

GSD_NEGENTROPY_COEF = 0.0

GSD_TEMP_INIT = 0.1   # maybe the thing to tune could be THIS (so that we lower it and bring the commitment phase closer to the beginning)
GSD_TEMP_FACT = 0.9   # 0.87055 = (0.5)^(1/5) = halving every five epochs
GSD_TEMP_MIN = 0.001  # 0.005 reached this temp in ~ 50 iters when starting from 2.0

# relative streght of the gumbel noise towards the logits
# (this is quite low, because the logits start all zero and never get too far with our default LR)
GUMBEL_STRENGTH : Final[float] = 0.001 # divided by 2.5 further; divided by two since last time -> earlier commitement

GSD_TWEAK_LEARNING_SPEEDUP = 100

LEARNING_RATE : Final[float] = 0.0002 # 0.0002 seemed a tad better and could become the default for the official experiments
LEARNING_RATE_DECAY = 0.933 # 0.87055 = (0.5)^(1/5) = halving every five epochs

TWEAKS_LEARNING_RATE = 0.05
TWEAK_SEARCH_MAX_TIME = 30.0 # in seconds

WEIGHT_DECAY : Final[float] = 0.0 # Corresponds to L2 regularization

# TODO: To be experimented with later
DISCOUNT_FACTOR = 1.0
