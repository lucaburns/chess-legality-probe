# Neurons extension (subfolder layout)

This package lives in its own subfolder of your project root and
shares the parent's `data/` directory and `chess_probe_common.py`.

## Layout (after extracting)

```
chess-legality-probe/
├── chess_probe_common.py        (your existing file)
├── chess_gpt_probe.py           (your existing file)
├── generate_games.py            (your existing file)
├── configs/
│   └── generation.yaml          (your existing file)
├── data/                        (your existing dataset folder)
├── plots/                       (will be created if missing)
└── neurons-extension/           (this package)
    ├── _paths.py
    ├── chess_probe_common_neurons.py
    ├── generate_games_with_neurons.py
    ├── chess_gpt_neuron_probe.py
    ├── analyze_legality_directions.py
    ├── plot_neuron_results.py
    ├── configs/
    │   └── generation_neurons.yaml
    └── README.md (this file)
```

## How paths work

All relative paths (e.g., `data/foo.pt`) are resolved against the
**project root**, not the subfolder. So `--dataset data/foo.pt`
points to `chess-legality-probe/data/foo.pt` regardless of where
you run the script from.

The `_paths.py` helper inserts the project root on `sys.path` so
`import chess_probe_common` works.

## Workflow

Run from the `neurons-extension/` subfolder:

```bash
cd neurons-extension

# 1) Generate dataset (~30 min, ~5 GB written to ../data/)
..\.venv\Scripts\python.exe generate_games_with_neurons.py ^
    --config configs\generation_neurons.yaml ^
    --repo C:\Users\monke\chess_gpt_eval

# 2) Train per-block neuron probes (~2 min on GPU)
..\.venv\Scripts\python.exe chess_gpt_neuron_probe.py ^
    --dataset data\stockfish16_t1p3_n30000_neurons.pt ^
    --per-fold-csv data\per_fold_neurons.csv ^
    --top-neurons-csv data\top_neurons.csv ^
    --device auto

# 3) Directional analysis (~30 sec, no training)
..\.venv\Scripts\python.exe analyze_legality_directions.py ^
    --dataset data\stockfish16_t1p3_n30000_neurons.pt ^
    --repo C:\Users\monke\chess_gpt_eval ^
    --residual-probe-layer 12 --top-k 20 ^
    --output data\direction_analysis.csv

# 4) Plots
..\.venv\Scripts\python.exe plot_neuron_results.py ^
    --neuron-csv data\per_fold_neurons.csv ^
    --residual-csv data\per_fold_t1p3_n30000.csv ^
    --top-neurons-csv data\top_neurons.csv ^
    --direction-csv data\direction_analysis.csv ^
    --out plots\neurons
```

All `--dataset`, `--per-fold-csv`, etc. paths are relative to the
project root, so they're identical to the flat-layout version.
