# Chess Legality Probe

## Overview

This project studies whether Chess-GPT internally represents move legality
before it emits an illegal move.

The workflow is intentionally split into two stages:

1. Generate a reusable dataset of Chess-GPT activations and legality labels.
2. Train linear probes on that fixed dataset, often many times, with different
   probe settings.

The repository keeps those stages separate so you can regenerate data only when
needed and iterate on probe experiments quickly.

## Project Layout

- `generate_games.py`
  Generates datasets by running Chess-GPT self-play and caching activations.

- `chess_gpt_probe.py`
  Loads a saved dataset and trains one probe per layer.

- `chess_probe_common.py`
  Shared dataset schema and save/load helpers.

- `configs/generation.yaml`
  Editable defaults for dataset generation.

- `configs/probe.yaml`
  Editable defaults for probe experiments.

- `data/`
  Recommended location for generated datasets.

- `outputs/`
  Recommended location for experiment outputs or saved summaries.

## Setup

Clone the companion Chess-GPT repository next to this project and install the
Python dependencies:

```bash
git clone https://github.com/adamkarvonen/chess_gpt_eval.git ../chess_gpt_eval
uv sync
```

Download a checkpoint from `https://huggingface.co/adamkarvonen/chess_llms` and
place it in:

```text
../chess_gpt_eval/nanogpt/out/
```

The commonly used checkpoint is:

```text
stockfish_16layers_ckpt_no_optimizer.pt
```

## Config System

This repo uses YAML config files for organization:

- `configs/generation.yaml`
  Stores settings related to dataset generation such as checkpoint, sampling
  settings, and output dataset path.

- `configs/probe.yaml`
  Stores settings related to probe experiments such as dataset path, optimizer
  settings, fold count, and experiment naming.

The current Python entry points still accept command-line arguments directly.
The YAML files are meant to be the editable source of truth you update between
runs, then translate into the corresponding CLI arguments when launching an
experiment.

This keeps generation settings stable while making probe settings easy to copy
and modify repeatedly.

## Generate Data

Start by editing `configs/generation.yaml` with the dataset settings you want.

The default structure includes:

- `paths.chess_gpt_eval_repo`
- `paths.checkpoint`
- `paths.output_dataset`
- `generation.device`
- `generation.positions`
- `generation.max_plies`
- `generation.temperature`
- `generation.top_k`
- `generation.max_new_tokens`
- `generation.random_opening_plies`
- `generation.stop_on_illegal`
- `generation.seed`

Then run dataset generation using the values from that config:

```bash
uv run python generate_games.py \
  --repo ../chess_gpt_eval \
  --checkpoint stockfish_16layers_ckpt_no_optimizer.pt \
  --output data/stockfish16_t1p3_n4000.pt \
  --device auto \
  --positions 4000 \
  --max-plies 120 \
  --temperature 1.3 \
  --top-k 200 \
  --max-new-tokens 10 \
  --random-opening-plies 2 \
  --seed 7
```

If you want a different dataset, edit `configs/generation.yaml` and rerun the
command with the new values.

## Run Probe Experiments

Once you have a dataset, edit `configs/probe.yaml` for the probe settings you
want to test.

The default structure includes:

- `paths.dataset`
- `paths.results_dir`
- `probe.epochs`
- `probe.lr`
- `probe.weight_decay`
- `probe.folds`
- `probe.seed`
- `probe.no_pos_weight`
- `experiment.name`

Run the probe script using the values from that config:

```bash
uv run python chess_gpt_probe.py \
  --dataset data/stockfish16_t1p3_n4000.pt \
  --epochs 200 \
  --lr 0.01 \
  --weight-decay 0.01 \
  --folds 5 \
  --seed 7
```

Because dataset generation is usually the expensive step, the intended pattern
is to keep `configs/generation.yaml` fixed for a while and make many edits or
copies of `configs/probe.yaml`.

## Creating New Experiment Configs

To create a new probe experiment on the same dataset, copy `configs/probe.yaml`
to a new file and change only the settings you care about.

Example:

```text
configs/probe_layer8.yaml
```

You might change:

- `experiment.name`
- `probe.epochs`
- `probe.lr`
- `probe.weight_decay`
- `probe.folds`
- `probe.seed`
- `paths.dataset`

This makes it easy to keep a record of multiple probe runs without touching the
dataset-generation config.
