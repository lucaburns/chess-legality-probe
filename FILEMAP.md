# File Map

## Top-Level Docs

- `README.md`
  Main project narrative, PGN/move-history framing, setup, and commands for the
  residual-stream, MLP, neuron-level, and clamping workflows.

- `4-22-Update.md`
  Earlier project update focused on the initial residual-stream linear probe
  pipeline and evaluation plan.

- `4-26-Update.md`
  Current project update covering finalized PGN-based prompting, updated SVG
  diagrams, the MLP probe, neuron-level MLP activation probes, activation
  clamping, interpretation, and next steps.

- `Initial_Project_Proposal.pdf`
  Original proposal and project background.

- `FILEMAP.md`
  This file.

## Core Pipeline

- `generate_games.py`
  Runs Chess-GPT self-play from PGN-style move-history prompts, samples
  candidate moves, labels legality with `python-chess`, captures residual
  stream activations, and writes reusable `.pt` datasets.

- `chess_gpt_probe.py`
  Trains one residual-stream linear legality probe per cached layer. Supports
  direct CLI arguments and `--config configs/probe.yaml`.

- `chess_gpt_mlp_probe.py`
  Trains small nonlinear MLP probes on the cached residual-stream activations
  for comparison against the linear probe.

- `chess_probe_common.py`
  Shared dataclass/schema and save/load helpers for residual-stream activation
  datasets.

- `config_utils.py`
  Small YAML loading and config-section flattening helpers.

- `plot_probe_distribution.py`
  Plotting helper for residual-stream probe result distributions.

## Neuron Extension

- `neurons-extension/README.md`
  Subfolder notes for the neuron-level workflow.

- `neurons-extension/_paths.py`
  Makes project-root imports and project-root-relative paths work from the
  neuron-extension scripts.

- `neurons-extension/generate_games_with_neurons.py`
  Generates datasets that include both residual stream activations and
  post-GELU MLP activations.

- `neurons-extension/chess_probe_common_neurons.py`
  Save/load helpers and schema support for datasets with MLP activations.

- `neurons-extension/chess_gpt_neuron_probe.py`
  Trains block-level probes on MLP activations and ranks activation dimensions
  by mean absolute probe-weight magnitude.

- `neurons-extension/analyze_legality_directions.py`
  Compares residual-stream legality directions with directions implied by
  ranked MLP activation dimensions.

- `neurons-extension/clamp_neurons_experiment.py`
  Runs activation-clamping sweeps during generation using ranked MLP activation
  dimensions from `top_neurons.csv`.

- `neurons-extension/plot_neuron_results.py`
  Creates neuron-extension summary plots from CSV outputs.

- `neurons-extension/configs/generation_neurons.yaml`
  Generation config for datasets that capture MLP activations.

## Configs

- `configs/generation.yaml`
  Residual-stream dataset generation settings: companion repo path,
  checkpoint, output path, device, sampling parameters, stopping behavior, and
  seed.

- `configs/probe.yaml`
  Residual-stream linear probe settings: dataset path, output directory,
  optimizer parameters, folds, seed, and experiment metadata.

## Docs And Figures

- `docs/figures/`
  Finalized SVGs linked from `README.md`:
  `candidate-move-legality.svg`, `move-history-to-legality-probe.svg`,
  `labeling-pipeline.svg`, `probe-locations.svg`, and
  `neuron-clamping-intervention.svg`.

- `docs/chess_legality_svgs/`
  Original working location for the chess legality SVGs before copying the
  finalized versions into `docs/figures/`.

## Data And Outputs

- `data/`
  Local generated datasets and CSV outputs. Current examples include residual
  activation datasets, neuron-activation sidecars, per-fold probe metrics,
  top activation-dimension rankings, direction analysis, and clamping sweep
  results.

- `plots/`
  Generated plots for residual-stream probe distributions.

- `plots/neurons/`
  Generated plots for neuron-level probes, direction analysis, and activation
  dimension rankings.

- `outputs/`
  Intended location for additional experiment outputs, logs, tables, or saved
  summaries.

## Environment And Metadata

- `pyproject.toml`
  Python project metadata and dependency list used by `uv`.

- `uv.lock`
  Locked dependency resolution for the local `uv` environment.

- `.venv/`
  Local virtual environment managed on this machine.

- `__pycache__/`
  Python bytecode cache generated automatically.

- `.gitignore`
  Git ignore rules.
