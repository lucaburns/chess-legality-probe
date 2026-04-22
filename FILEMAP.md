# File Map

## Core Files

- `README.md`
  Project overview, setup steps, and the recommended workflow.

- `pyproject.toml`
  Python project metadata and dependency list used by `uv`.

- `chess_gpt_probe.py`
  Trains one linear legality probe per layer from a saved activation dataset.

- `chess_probe_common.py`
  Shared dataset schema plus save/load helpers for `.pt` payloads.

- `generate_games.py`
  Runs Chess-GPT self-play, captures activations, labels move legality, and
  writes datasets for later probe experiments.

- `Initial_Project_Proposal.pdf`
  Proposal and project background.

- `4-22-Update.md`
  Project description and update for 4-22 submission

## Directories

- `configs/`
  Human-editable YAML configs.
  `generation.yaml` stores dataset-generation settings.
  `probe.yaml` stores probe-experiment settings.

- `data/`
  Intended location for generated datasets such as cached activation payloads.

- `outputs/`
  Intended location for experiment results, logs, tables, or saved summaries.

- `.venv/`
  Local virtual environment managed on this machine.

- `__pycache__/`
  Python bytecode cache generated automatically.

## Notes

- `test.py`
  Left on disk but removed from git tracking because it is stale relative to the
  current project structure.
