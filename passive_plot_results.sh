#!/bin/bash

#!/bin/bash

MODEL=~/jar2026/seed42_nd_noSplitB/loop1/loop-model.tar
TRACE_DIR=~/jar2026/seed42_nd_noSplitBB/traces
OUT_DIR=trace_plots

mkdir -p "$OUT_DIR"

while read -r line; do
    problem=$(echo "$line" | awk '{print $1}')

    # Problems/CSR/CSR113+1.p -> Problems_CSR_CSR113+1.p
    trace_base=$(echo "$problem" | tr '/' '_')

    # Problems/CSR/CSR113+1.p -> Problems_CSR113+1
    pdf_base=$(echo "$problem" | sed 's#Problems/\([^/]*\)/#Problems_#' | sed 's/\.p$//')

    ./passive_plotter.py \
        "$MODEL" \
        "$TRACE_DIR/${trace_base}_0.pt" \
        "$OUT_DIR/${pdf_base}.pdf"

done

# where the input should come from calling, e.g., ./result_reader.py ~/jar2026/split42_boostScale/loop25/train_res.pt | grep iter=0 | ./passive_plot_results.sh