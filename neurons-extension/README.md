# Neurons extension

This package lives in its own subfolder of the project root and shares the
parent's `data/` directory and `chess_probe_common.py`.

## Layout

```
chess-legality-probe/
├── chess_probe_common.py        (shared dataset utilities)
├── chess_gpt_probe.py           (residual-stream linear probe)
├── generate_games.py            (game generation)
├── configs/
│   └── generation.yaml
├── data/                        (shared dataset directory)
├── plots/                       (shared plot output directory)
└── neurons-extension/           (this package)
    ├── _paths.py
    ├── chess_probe_common_neurons.py
    ├── generate_games_with_neurons.py
    ├── chess_gpt_neuron_probe.py
    ├── analyze_legality_directions.py
    ├── clamp_neurons_experiment.py
    ├── plot_neuron_results.py
    ├── configs/
    │   └── generation_neurons.yaml
    └── README.md (this file)
```

## How paths work

All relative paths (e.g. `data/foo.pt`) are resolved against the **project
root**, not this subfolder. So `--dataset data/foo.pt` always points to
`chess-legality-probe/data/foo.pt` regardless of where you run the script from.

The `_paths.py` helper inserts the project root on `sys.path` so
`import chess_probe_common` works.

## Workflow

All commands use `uv run python` and are run from the **project root**
(`chess-legality-probe/`):

```bash
# 1) Generate dataset with both residual stream and MLP activations
#    (~30 min, ~1–2 GB written to data/)
uv run python neurons-extension/generate_games_with_neurons.py \
    --config neurons-extension/configs/generation_neurons.yaml \
    --repo ../chess_gpt_eval

# 2) Train per-block neuron probes (~2 min on GPU)
uv run python neurons-extension/chess_gpt_neuron_probe.py \
    --dataset data/stockfish16_t1p3_n30000_neurons.pt \
    --per-fold-csv data/per_fold_neurons.csv \
    --top-neurons-csv data/top_neurons.csv \
    --device auto

# 3) Direction analysis (~30 sec, no training)
uv run python neurons-extension/analyze_legality_directions.py \
    --dataset data/stockfish16_t1p3_n30000_neurons.pt \
    --repo ../chess_gpt_eval \
    --residual-probe-layer 12 \
    --top-k 20 \
    --output data/direction_analysis.csv

# 4) Neuron-extension plots
uv run python neurons-extension/plot_neuron_results.py \
    --neuron-csv data/per_fold_neurons.csv \
    --residual-csv data/per_fold_t1p3_n30000.csv \
    --top-neurons-csv data/top_neurons.csv \
    --direction-csv data/direction_analysis.csv \
    --out plots/neurons

# 5) Activation-clamping intervention sweep (~20 min on GPU)
uv run python neurons-extension/clamp_neurons_experiment.py \
    --dataset data/stockfish16_t1p3_n30000_neurons.pt \
    --repo ../chess_gpt_eval \
    --top-neurons-csv data/top_neurons.csv \
    --output data/clamp_sweep_results.csv \
    --eval-positions 2000 \
    --device auto
```
