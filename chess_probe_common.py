"""Shared utilities for the Chess-GPT legality probe pipeline.

This module defines the Example dataclass and functions to save/load a
dataset of examples to disk. Both the game-generation script and the
probe-training script import from here so the on-disk format has a
single source of truth.

On-disk format (single .pt file):
    {
        "activations": Tensor[n_examples, n_layers + 1, d_model],
        "is_legal":    Tensor[n_examples] (int8, 0 or 1),
        "ply":         Tensor[n_examples] (int32),
        "game_id":     Tensor[n_examples] (int32),
        "move_text":   list[str] of length n_examples,
        "prompt":      list[str] of length n_examples,
        "fen":         list[str] of length n_examples,
        "config":      dict of generation config for provenance,
        "n_layers":    int,
        "d_model":     int,
    }

String fields are stored as Python lists (torch doesn't natively handle
them). Numeric fields are stored as tensors so loading is cheap and
slicing is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


@dataclass
class Example:
    layer_activations: list[Any]  # list of torch.Tensor, one per layer boundary
    is_legal: int
    move_text: str
    prompt: str
    ply: int          # ply index within the game (0 = before any move)
    game_id: int      # which self-play game this example came from
    fen: str          # FEN of the position before the attempted move


def save_examples(
    examples: list[Example],
    path: Path,
    config: dict,
) -> None:
    """Serialize a list of Examples plus generation config to a .pt file.

    config should contain the settings used to generate the data
    (checkpoint, temperature, seed, max_plies, etc.) for provenance.
    """
    import torch  # imported lazily so common module doesn't force torch at import time

    if not examples:
        raise ValueError("Cannot save empty example list.")

    n = len(examples)
    n_layers_plus_embed = len(examples[0].layer_activations)
    d_model = examples[0].layer_activations[0].shape[-1]

    # Stack into a single (n, n_layers+1, d_model) tensor.
    activations = torch.empty((n, n_layers_plus_embed, d_model), dtype=torch.float32)
    for i, ex in enumerate(examples):
        for layer_idx, act in enumerate(ex.layer_activations):
            activations[i, layer_idx] = act.float()

    payload = {
        "activations": activations,
        "is_legal": torch.tensor([e.is_legal for e in examples], dtype=torch.int8),
        "ply": torch.tensor([e.ply for e in examples], dtype=torch.int32),
        "game_id": torch.tensor([e.game_id for e in examples], dtype=torch.int32),
        "move_text": [e.move_text for e in examples],
        "prompt": [e.prompt for e in examples],
        "fen": [e.fen for e in examples],
        "config": config,
        "n_layers": n_layers_plus_embed - 1,  # blocks only; embedding is separate
        "d_model": d_model,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

    # Also drop a small JSON sidecar for human inspection.
    sidecar = path.with_suffix(path.suffix + ".json")
    summary = {
        "n_examples": n,
        "n_legal": int(sum(e.is_legal for e in examples)),
        "n_illegal": int(n - sum(e.is_legal for e in examples)),
        "n_games": int(max(e.game_id for e in examples) + 1) if examples else 0,
        "n_layers": n_layers_plus_embed - 1,
        "d_model": d_model,
        "config": config,
    }
    sidecar.write_text(json.dumps(summary, indent=2, default=str))


def load_examples(path: Path) -> dict:
    """Load a saved dataset. Returns the raw payload dict; callers can
    either rebuild Example objects or work directly with the tensor
    fields (faster)."""
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # weights_only=False because we store Python lists and a config dict.
    return torch.load(path, map_location="cpu", weights_only=False)


def payload_to_examples(payload: dict) -> list[Example]:
    """Reconstruct a list of Example objects from a loaded payload.

    Most downstream code doesn't need this — working directly with the
    stacked tensor is faster. Provided for convenience / debugging.
    """
    n = payload["activations"].shape[0]
    examples = []
    for i in range(n):
        acts = [payload["activations"][i, j] for j in range(payload["activations"].shape[1])]
        examples.append(
            Example(
                layer_activations=acts,
                is_legal=int(payload["is_legal"][i]),
                move_text=payload["move_text"][i],
                prompt=payload["prompt"][i],
                ply=int(payload["ply"][i]),
                game_id=int(payload["game_id"][i]),
                fen=payload["fen"][i],
            )
        )
    return examples
