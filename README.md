# Chess Legality Probe

## Overview

We study whether Chess-GPT's internal activations contain
information about whether its next generated chess move will be legal.

The workflow is split into reusable stages:

1. Generate Chess-GPT self-play positions, candidate moves, legality labels,
   and cached activations.
2. Train probes on the fixed activation dataset.
3. Compare where legality information is easiest to decode: residual stream
   activations, nonlinear probes over the same activations, and MLP activation
   dimensions.
4. Run activation-clamping interventions on selected MLP activation dimensions
   to test whether perturbing those dimensions changes generated-move behavior.

The probes are diagnostic tools. Strong probe performance is evidence that
legality-related information is available in an activation space; it is not, by
itself, proof that a specific activation dimension definitively encodes legality
or that the model uses that dimension causally.

## What The Model Sees

![Move history to legality probe](docs/figures/move-history-to-legality-probe.svg)

Chess-GPT is prompted with PGN-style move history, and the
generator converts the current game into text such as:

```text
;1.e4 e5 2.Nf3 Nc6
```

The model receives this move-history prompt and generates the next move as
text. A chess rules engine reconstructs the board from the move history for
labeling, but the model input remains the PGN-style transcript.

## How Moves Are Generated And Labeled

![Candidate move legality](docs/figures/candidate-move-legality.svg)

For each self-play position, Chess-GPT samples a candidate next move from the
PGN-style prompt. The code extracts the first generated move token and
asks `python-chess` whether that move is legal in the reconstructed position.

![Generated move labeling pipeline](docs/figures/labeling-pipeline.svg)

Each saved example contains the prompt-derived position, the attempted move,
the binary legality label, and cached activations at the final prompt token.
The main target used by the probes is `is_illegal = 1 - is_legal`, so AUROC is
more informative than raw accuracy when illegal moves are rare.

## Probe experiments

![Probe locations](docs/figures/probe-locations.svg)

## Residual-Stream Linear Probe

`chess_gpt_probe.py` trains a linear classifier on residual stream activations.
The dataset stores the residual stream after embedding and after each
transformer block, always at the final prompt token before the sampled move.

This answers a narrow question: at which layers is future move legality
linearly decodable from the residual stream?

## MLP Nonlinear Probe

`chess_gpt_mlp_probe.py` trains a small MLP probe on the same cached residual
stream activations. It uses the same labels and fold structure as the linear
probe, but allows a nonlinear decision boundary.

The comparison is intentionally conservative:

- linear probe: asks whether legality is linearly decodable from the residual
  stream
- MLP probe: asks whether the same residual-stream information becomes more
  accessible with a small nonlinear probe

Higher MLP AUROC should be described as a nonlinear decoding result, not as
proof that the model's own downstream computation uses the same classifier.

## Neuron-Level Probe On MLP Activations

The neuron extension captures post-GELU MLP activations for every transformer
block. `neurons-extension/chess_gpt_neuron_probe.py` trains linear probes on
those MLP activations block by block.

The script also ranks activation dimensions by the mean absolute magnitude of
their learned probe weights across folds. These ranked dimensions are useful
candidate features for inspection and intervention, but the ranking should not
be read as definitive evidence that individual neurons encode legality.

## Neuron/Activation Clamping Intervention

![Neuron clamping intervention](docs/figures/neuron-clamping-intervention.svg)

`neurons-extension/clamp_neurons_experiment.py` uses the ranked MLP activation
dimensions from `top_neurons.csv` and clamps selected dimensions during
generation. The hook modifies only the final-token MLP activations, matching
the activation position used by the probes.

The intervention is a causal test of whether perturbing selected activation
dimensions changes generated-move legality rates. Results should be interpreted
with care: a change under clamping is evidence that the perturbed activations
matter for this generation setup, but it does not by itself identify a clean,
human-interpretable legality neuron.

## Updates and Writeups

Progress writeups are in [`docs/updates/`](docs/updates/):

- [`docs/updates/4-22-Update.md`](docs/updates/4-22-Update.md) — Initial
  pipeline setup, probe design, and what to look for in a full run.
- [`docs/updates/4-26-Update.md`](docs/updates/4-26-Update.md) — Extended
  results: MLP probe, neuron-level probe, activation clamping, and inline plots.

SVG diagrams referenced by those writeups are in [`docs/figures/`](docs/figures/).

## Project Layout

- `download_model.py`
  Downloads the `stockfish_16layers_ckpt_no_optimizer.pt` checkpoint from
  HuggingFace and places it in the expected location. Run this once after
  cloning:
  ```bash
  uv run python download_model.py
  ```

- `generate_games.py`
  Generates PGN-prompted self-play examples, legality labels, and residual
  stream activations.

- `chess_gpt_probe.py`
  Trains residual-stream linear probes.

- `chess_gpt_mlp_probe.py`
  Trains nonlinear MLP probes on residual-stream activations.

- `chess_probe_common.py`
  Shared dataset schema and save/load helpers for residual-stream datasets.

- `plot_comparison.py`
  Generates headline comparison plots (AUROC by layer across probe types,
  clamping sweep results) from the per-fold CSV files. Does not require the
  full activation dataset.

- `plot_ply_analysis.py`
  Generates a plot of probe AUROC as a function of game depth (ply). Requires
  the full activation `.pt` dataset.

- `plot_probe_distribution.py`
  Per-layer AUROC and accuracy strip/sina plots for a single probe CSV.

- `neurons-extension/generate_games_with_neurons.py`
  Generates datasets with both residual stream activations and post-GELU MLP
  activations.

- `neurons-extension/chess_gpt_neuron_probe.py`
  Trains block-level probes on MLP activations and writes ranked activation
  dimensions by probe-weight magnitude.

- `neurons-extension/analyze_legality_directions.py`
  Compares residual-stream legality directions with neuron-level directions.

- `neurons-extension/clamp_neurons_experiment.py`
  Runs activation-clamping sweeps during generation.

- `neurons-extension/plot_neuron_results.py`
  Produces summary plots for residual, neuron-level, and direction-analysis
  outputs.

- `configs/generation.yaml`
  Dataset-generation defaults for the residual-stream pipeline.

- `configs/probe.yaml`
  Probe-training defaults for the residual-stream linear probe.

- `neurons-extension/configs/generation_neurons.yaml`
  Dataset-generation defaults for the neuron/MLP activation pipeline.

## Configuration Reference

### `configs/generation.yaml` — dataset generation

| Key | Default | Meaning |
|-----|---------|---------|
| `paths.chess_gpt_eval_repo` | `../chess_gpt_eval` | Path to the companion chess_gpt_eval repository (contains the model code and checkpoint directory). |
| `paths.checkpoint` | `stockfish_16layers_ckpt_no_optimizer.pt` | Filename of the model checkpoint inside `chess_gpt_eval/nanogpt/out/`. |
| `paths.output_dataset` | `data/stockfish16_t1p3_n4000.pt` | Where the generated `.pt` dataset is written. |
| `generation.device` | `auto` | Compute device. `auto` picks CUDA → MPS → CPU. |
| `generation.positions` | `30000` | Number of (position, candidate move, legality label) examples to collect. |
| `generation.max_plies` | `160` | **Maximum game length in plies** (a *ply* is one player's move, so 160 plies = 80 full moves). Games that haven't ended naturally are truncated here. |
| `generation.temperature` | `1.3` | Sampling temperature. Higher values make the model more random and produce more illegal moves. |
| `generation.top_k` | `200` | Top-k sampling cutoff for move generation. |
| `generation.max_new_tokens` | `10` | Maximum token budget when sampling one move. |
| `generation.random_opening_plies` | `2` | Number of initial plies played with uniform-random moves to diversify starting positions. |
| `generation.stop_on_illegal` | `true` | **What happens when the model generates an illegal move.** When `true` (default): the illegal example is recorded and the game ends immediately. When `false`: the illegal example is still recorded, but the game continues — an actual legal move is pushed to the board so subsequent positions are reachable. Setting this to `false` lets you collect more illegal examples, but the board states that follow an illegal move are not positions the model navigated to on its own. |
| `generation.seed` | `1` | Random seed for reproducibility. |

### `configs/probe.yaml` — probe training

| Key | Default | Meaning |
|-----|---------|---------|
| `paths.dataset` | `data/stockfish16_t1p3_n4000.pt` | Dataset to train probes on. |
| `probe.epochs` | `200` | Training epochs per fold. |
| `probe.lr` | `0.01` | Learning rate. |
| `probe.weight_decay` | `0.01` | L2 regularization. |
| `probe.folds` | `5` | Number of cross-validation folds. |
| `probe.seed` | `7` | Fold-split seed. |
| `probe.no_pos_weight` | `false` | When `false` (default), class imbalance is corrected by upweighting the rare illegal-move class in the loss. Set to `true` to train without this correction. |

## Setup

Clone the companion Chess-GPT repository next to this project and install the
Python dependencies:

```bash
git clone https://github.com/adamkarvonen/chess_gpt_eval.git ../chess_gpt_eval
uv sync --python 3.12
```

Download the model checkpoint automatically:

```bash
uv run python download_model.py
```

Or download manually from <https://huggingface.co/adamkarvonen/chess_llms>
and place the file at:

```text
../chess_gpt_eval/nanogpt/out/stockfish_16layers_ckpt_no_optimizer.pt
```

This repository depends on plain `torch` rather than a CUDA-specific wheel.
CPU and macOS users can use the default `uv sync`. CUDA users should install
the CUDA-enabled PyTorch build matching their system.

Verify the installed PyTorch build and available backends:

```bash
uv run --python 3.12 python -c "import torch; print(torch.__version__); print('mps:', torch.backends.mps.is_available()); print('cuda:', torch.cuda.is_available())"
```

The experiment scripts accept `--device auto`, `--device mps`,
`--device cpu`, and `--device cuda`. With `--device auto`, the code chooses:

```text
cuda -> mps -> cpu
```

## How To Run The Main Scripts

Generate a residual-stream dataset from `configs/generation.yaml`:

```bash
uv run python generate_games.py --config configs/generation.yaml
```

Override selected generation settings from the command line:

```bash
uv run python generate_games.py \
  --config configs/generation.yaml \
  --positions 80 \
  --output data/quicktest.pt
```

Run the residual-stream linear probe:

```bash
uv run python chess_gpt_probe.py \
  --config configs/probe.yaml \
  --per-fold-csv data/per_fold_t1p3_n30000.csv
```

Run the MLP nonlinear probe:

```bash
uv run python chess_gpt_mlp_probe.py \
  --dataset data/stockfish16_t1p3_n30000.pt \
  --device auto \
  --hidden 64 \
  --num-hidden-layers 1 \
  --dropout 0.1 \
  --epochs 50 \
  --batch-size 256 \
  --per-fold-csv data/per_fold_mlp_t1p3_n30000.csv
```

Generate a dataset that also includes MLP activations:

```bash
uv run python neurons-extension/generate_games_with_neurons.py \
  --config neurons-extension/configs/generation_neurons.yaml \
  --repo ../chess_gpt_eval
```

Run the neuron-level probe on MLP activations:

```bash
uv run python neurons-extension/chess_gpt_neuron_probe.py \
  --dataset data/stockfish16_t1p3_n30000_neurons.pt \
  --device auto \
  --per-fold-csv data/per_fold_neurons.csv \
  --top-neurons-csv data/top_neurons.csv \
  --top-k 20
```

Run direction analysis:

```bash
uv run python neurons-extension/analyze_legality_directions.py \
  --dataset data/stockfish16_t1p3_n30000_neurons.pt \
  --repo ../chess_gpt_eval \
  --residual-probe-layer 12 \
  --top-k 20 \
  --output data/direction_analysis.csv
```

Run the activation-clamping intervention:

```bash
uv run python neurons-extension/clamp_neurons_experiment.py \
  --dataset data/stockfish16_t1p3_n30000_neurons.pt \
  --repo ../chess_gpt_eval \
  --top-neurons-csv data/top_neurons.csv \
  --output data/clamp_sweep_results.csv \
  --eval-positions 2000 \
  --device auto
```

Generate neuron-extension plots:

```bash
uv run python neurons-extension/plot_neuron_results.py \
  --neuron-csv data/per_fold_neurons.csv \
  --residual-csv data/per_fold_t1p3_n30000.csv \
  --top-neurons-csv data/top_neurons.csv \
  --direction-csv data/direction_analysis.csv \
  --out plots/neurons
```

Generate headline comparison plots from the per-fold CSVs (no dataset needed):

```bash
uv run python plot_comparison.py \
  --linear-csv  data/per_fold_t1p3_n30000.csv \
  --mlp-csv     data/per_fold_mlp_t1p3_n30000.csv \
  --neuron-csv  data/per_fold_neurons.csv \
  --clamp-csv   data/clamp_sweep_results.csv \
  --out         plots/
```

Generate the per-ply AUROC plot (requires the full `.pt` dataset):

```bash
uv run python plot_ply_analysis.py \
  --dataset data/stockfish16_t1p3_n30000.pt \
  --layer 12 \
  --out plots/auroc_by_ply.png
```

Because dataset generation is the expensive step, the intended pattern is to
keep generation configs fixed for a run and iterate on probe settings and
analysis scripts afterward.
