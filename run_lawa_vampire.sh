#!/bin/bash

cd ../projects/vampire/z3/build > /dev/null
export LD_LIBRARY_PATH=`pwd`
cd - > /dev/null

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# by convention, the arguments to this script should start "-t number" (which will work both with timelimit and vampire)
timelimit $1 $2 -T 1 ./vampire_rel_lawa_6459 -t 0 -i 0 -m 8192 --input_syntax tptp -stat full -sa discount -si on -rawr on -rtra on "$@"
exit 0
