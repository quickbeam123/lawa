# Deepire 2.0

Training neural clause selection models for the [Vampire](https://vprover.github.io/) automated theorem prover.
(Vampire with clause selection support currently lives in a [branch](https://github.com/vprover/vampire/tree/mtpa-gnn-2026).)

The system runs in a loop: evaluate Vampire with the current model on [TPTP](https://www.tptp.org/) problems, gather proof traces from successful runs, train the model on those traces, export an updated model, and repeat.

## Key Files

- `elooper.py` — Main training loop. Orchestrates stage 1 (Vampire performance gathering + trace collection) and stage 2 (eval/train with optional early stopping).
- `hyperparams.py` — All hyperparameters as module-level constants with `Final` type annotations. Imported as `HP` everywhere. Edit this to configure experiments.
- `inf_common.py` — Model architecture, trace processing, and model export.
- `workers.py` — Worker process implementations for running Vampire, collecting traces, evaluating the model, and computing gradients.
- `run_lawa_vampire.sh` — Shell wrapper that sets up `LD_LIBRARY_PATH` for z3 and invokes Vampire via `timelimit`.

## Model Architecture

The neural model (`MonsterModules` in `inf_common.py`) combines three embedding approaches for clauses:

- **GNN (Graph Neural Network):** SAGEConv message-passing over the problem's symbol/term/clause/sort/var graph, producing initial clause and symbol embeddings.
- **GAGE:** Recursive tree-structured embeddings following the proof derivation (inference rule + parent clause embeddings).
- **GWEIGHT:** Recursive tree-structured embeddings following term structure (functor + argument embeddings).

These are concatenated with simple clause features and passed through an MLP to produce a scalar clause selection score.

## Running Experiments

```bash
# Start a new experiment with 15 loops using 120 parallel workers
./elooper.py 15 120 /path/to/exper_dir

# Continue from a previous experiment (steal train/test split + model from loop 6)
./elooper.py 15 120 /path/to/new_exper /path/to/prev_exper 6

# Continue with model (m) and/or traces (t/T)
./elooper.py 15 120 /path/to/new_exper /path/to/prev_exper 6 mt
```

The 6th argument flags: `m` = load model, `t` = load traces from loop+1, `T` = load traces from same loop.

## Requirements

- PyTorch, torch_geometric, sortedcontainers, numpy
- Custom Vampire build with neural clause selection support (`HP.VAMPIRE_EXECUTABLE`)
- Scratch directory (`HP.SCRATCH`) for temporary model/gradient files between workers
