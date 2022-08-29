#!/bin/bash

cd ../projects/vampire/z3/build > /dev/null
export LD_LIBRARY_PATH=`pwd`
cd - > /dev/null

# echo $1 -t 0 -i 50000 -m 8192 --input_syntax tptp -stat full -sa discount -spt on -si on -rawr on -rtra on --random_seed $$ $3 $2 ">" "$4"/`(./filenamify_path.sh "$2")`.log
timelimit -t 60 -T 1 $1 -t 0 -i 50000 -m 8192 --input_syntax tptp -stat full -sa discount -si on -rawr on -rtra on --random_seed $$ $2 $3 > "$4"/`(./filenamify_path.sh "$2")`.log 2>&1
exit 0
