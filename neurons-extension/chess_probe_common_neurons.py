"""Save/load helpers for neuron-level datasets.

Extends chess_probe_common with an additional `mlp_activations` field.
The existing `chess_probe_common.load_examples` will still work for reading
the residual-stream subset of these datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class NeuronExample:
    """One labeled example with both residual-stream and MLP-hidden activations.

    Fields mirror the original `Example` dataclass plus `mlp_hidden`.
    """
    game_id: int
    ply_index: int
    is_legal: int                  # 1 = legal, 0 = illegal
    residual: torch.Tensor         # shape (n_layers + 1, d_model)
    mlp_hidden: torch.Tensor       # shape (n_layers, d_mlp)


def save_neuron_examples(
    path: Path,
    examples: list[NeuronExample],
    config: dict[str, Any],
) -> None:
    """Save examples in the same on-disk shape conventions as the original
    pipeline, plus an `mlp_activations` tensor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not examples:
        raise ValueError("Cannot save empty examples list")

    n = len(examples)
    n_layers_plus_one, d_model = examples[0].residual.shape
    n_layers, d_mlp = examples[0].mlp_hidden.shape

    residual = torch.empty((n, n_layers_plus_one, d_model), dtype=torch.float32)
    mlp_hidden = torch.empty((n, n_layers, d_mlp), dtype=torch.float32)
    is_legal = torch.empty((n,), dtype=torch.int8)
    game_id = torch.empty((n,), dtype=torch.int32)
    ply_index = torch.empty((n,), dtype=torch.int32)

    for i, ex in enumerate(examples):
        residual[i] = ex.residual
        mlp_hidden[i] = ex.mlp_hidden
        is_legal[i] = ex.is_legal
        game_id[i] = ex.game_id
        ply_index[i] = ex.ply_index

    payload = {
        "activations": residual,           # name kept for backwards compat
        "mlp_activations": mlp_hidden,
        "is_legal": is_legal,
        "game_id": game_id,
        "ply_index": ply_index,
        "config": config,
        "version": "neurons-1",
    }
    torch.save(payload, path)


def load_neuron_examples(path: Path) -> dict[str, Any]:
    """Load a neuron-level dataset. Raises if the file lacks mlp activations."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "mlp_activations" not in payload:
        raise KeyError(
            f"{path} does not contain 'mlp_activations'. Was it generated "
            f"with generate_games_with_neurons.py?"
        )
    return payload
