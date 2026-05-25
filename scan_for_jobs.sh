#!/bin/bash

set -u
shopt -s nullglob

WATCH_LOG_DIR="$1"
OUTPUT_BASE="$2"

while true; do
    files=("$WATCH_LOG_DIR"/*.py)

    if [ ${#files[@]} -eq 0 ]; then
        echo "No more .py files found. Exiting."
        break
    fi

    src="${files[0]}"
    filename=$(basename "$src" .py)
    # strip a two digits and underscore prefix, e.g. 01_, used only for tasks sorting purposes
    base="${filename#[0-9][0-9]_}"

    echo "Processing: $base"

    mv "$src" ./hyperparams.py

    ./elooper.py 25 120 "$OUTPUT_BASE/$base" > "$WATCH_LOG_DIR/$base.log" 2>&1

    echo "Finished: $base"
done