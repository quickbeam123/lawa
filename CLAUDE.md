# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LAWA (Learning-Assisted Weighted Amalgamation) trains neural clause selection models for the Vampire automated theorem prover. The system runs in a loop: evaluate Vampire with the current model on TPTP problems, gather proof traces from successful runs, train the model on those traces, export an updated model, repeat.

The neural model (`MonsterModules` in `inf_common.py`) combines three embedding approaches for clauses:
- **GNN (Graph Neural Network):** SAGEConv message-passing over the problem's symbol/term/clause/sort/var graph, producing initial clause and symbol embeddings
- **GAGE:** Recursive tree-structured embeddings following the proof derivation (inference rule + parent clause embeddings)
- **GWEIGHT:** Recursive tree-structured embeddings following term structure (functor + argument embeddings)

These are concatenated with simple clause features and passed through an MLP to produce a scalar clause selection score.

## Key Files

- `hyperparams.py` - All hyperparameters as module-level constants with `Final` type annotations. Imported as `HP` everywhere. Edit this to configure experiments.
- `inf_common.py` - Model architecture (`MonsterModules` for storage, `MonsterNN` for TorchScript-compatible inference, `LearningModel` for training), trace processing, and model export
- `elooper.py` - Main training loop. Orchestrates stage 1 (Vampire performance gathering + trace collection) and stage 2 (eval/train with optional early stopping). Entry point: `./elooper.py <loop_count> <parallelism> <exper_dir> [prev_exper] [prev_loop] [flags]`
- `workers.py` - Worker process implementations for 4 job kinds: `JK_PERFORM` (run Vampire), `JK_GATHER` (collect traces), `JK_EVAL_TWEAK_MATRIX` (evaluate model on traces), `JK_TRAIN` (compute gradients on one problem's traces)
- `slooper.py` - "Snake" mode looper, gathers traces from pre-existing strategy evaluations rather than running Vampire live
- `run_lawa_vampire.sh` - Shell wrapper that sets up LD_LIBRARY_PATH for z3 and invokes Vampire via `timelimit`

## Running Experiments

```bash
# Start a new experiment with 15 loops using 120 parallel workers
./elooper.py 15 120 /path/to/exper_dir

# Continue from a previous experiment (steal train/test split + model from loop 6)
./elooper.py 15 120 /path/to/new_exper /path/to/prev_exper 6

# Continue with model (m) and/or traces (t/T) and/or tweak map (w)
./elooper.py 15 120 /path/to/new_exper /path/to/prev_exper 6 mt
```

The 6th argument flags: `m` = load model, `M` = load old-format model, `t` = load traces from loop+1, `T` = load traces from same loop, `w` = load tweak map.

## Architecture Details

### Parallelism Model
Two separate worker pools created via `multiprocessing`:
- **perf_and_gather**: Full parallelism (e.g., 120 workers) for running Vampire and collecting traces
- **eval_and_train**: Limited parallelism (`TRAINING_PARALLELISM`, default 64) for model evaluation and gradient computation

Training is "mildly Hogwild": each worker computes gradients independently, but `optimizer.step()` is called serially in the main process after each worker returns results. Gradients are communicated via the filesystem (model state dicts saved/loaded as `.tar` files through `HP.SCRATCH`).

### Trace Format
Traces are 10-tuples saved with `torch.save`:
```
(static_features, clause_simple_features, journal, num_good_selections,
 init_gnn_nodes, gnn_edges, gnn_init_clause_nums,
 gage_infers, gweight_terms, gweight_clauses)
```
The `journal` is a list of `(event_tag, clause_num, is_good)` triples tracking clause additions, removals, and selections during a Vampire run.

### Model Export for Vampire
`IC.export_model()` creates a TorchScript module (`MonsterNN`) that Vampire loads at runtime. The `MonsterNN` class has `@torch.jit.export` decorated methods that Vampire calls: `set_static_features`, `gnn_node_kind`, `gnn_edge_kind`, `gnn_perform`, `gage_enqueue`, `gweight_enqueue_term`, `gweight_enqueue_clause`, `embed_pending`, `eval_clauses`.

### Training Loss
Cross-entropy over clause selections: at each "good selection" moment in the journal, the model's logits over passive clauses are compared against the distribution of proof-relevant clauses. Controlled by `MAX_TRAINS_PER_TRACE` (subsampling for long traces) and `LABEL_SMOOTHING`.

## Environment Requirements
- PyTorch, torch_geometric, sortedcontainers, numpy
- Custom Vampire build (`HP.VAMPIRE_EXECUTABLE`) with neural clause selection support
- Scratch directory (`HP.SCRATCH`) for temporary model/gradient files between workers
- All threading env vars are set to "1" to avoid interference with multiprocessing
- CUDA is explicitly disabled in elooper.py (`CUDA_VISIBLE_DEVICES=""`)

## Conventions
- `HP.` prefix for all hyperparameters (imported from `hyperparams.py`)
- `IC.` prefix for model/infrastructure functions (imported from `inf_common.py`)
- `W.` prefix for worker functions (imported from `workers.py`)
- Problem names use `/` separators internally but `_` when used in filenames (`prob.replace("/","_")`)
- Tweak map keys use `_` instead of `.` in problem names (`no_dots()` function)
