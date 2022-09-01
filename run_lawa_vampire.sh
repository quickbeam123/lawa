#!/bin/bash

cd ../projects/vampire/z3/build > /dev/null
export LD_LIBRARY_PATH=`pwd`
cd - > /dev/null

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

timelimit -t 60 -T 1 ./vampire_rel_lawa_6441 -t 0 -i 0 -m 8192 --input_syntax tptp -stat full -sa discount -si on -rawr on -rtra on "$@"
exit 0
