"""Decompose a residual-stream legality direction into MLP-neuron contributions.

Given a trained linear probe at residual-stream layer L (the "legality
direction"), we ask: which neurons across all layers <= L write into that
direction most strongly?

For an MLP with output projection W_out (d_mlp -> d_model), neuron i in
that layer writes its scalar activation along the column W_out[:, i].
Projection score onto the legality direction d:

    score_i = d @ W_out[:, i]

This script trains a probe at the chosen residual layer using the saved
dataset, then computes scores for every neuron in every block <= L.

Usage:
    python analyze_legality_directions.py \\
        --dataset data/...neurons.pt \\
        --residual-probe-layer 12 --top-k 20 \\
        --repo C:\\path\\to\\chess_gpt_eval \\
        --output data/direction_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

from _paths import setup_paths, resolve_path
setup_paths()

from chess_probe_common_neurons import load_neuron_examples


def load_chess_gpt_for_weights(repo_path: Path, checkpoint_name: str):
    """Load model just to access W_out matrices. CPU is fine."""
    nanogpt_dir = repo_path / "nanogpt"
    sys.path.insert(0, str(nanogpt_dir))
    from model import GPT, GPTConfig  # type: ignore

    ckpt_path = nanogpt_dir / "out" / checkpoint_name
    if not ckpt_path.exists():
        for candidate in [nanogpt_dir / checkpoint_name,
                          repo_path / checkpoint_name]:
            if candidate.exists():
                ckpt_path = candidate
                break
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = ckpt["model"]
    for k in list(state.keys()):
        if k.startswith("_orig_mod."):
            state[k[len("_orig_mod."):]] = state.pop(k)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def train_residual_probe(activations, is_legal, layer, *,
                          epochs=200, lr=1e-2, weight_decay=1e-2,
                          use_pos_weight=True):
    """Train a single linear probe (no CV) on the chosen residual layer
    and return its weight vector — used as the legality direction."""
    x = activations[:, layer, :].float()
    y = (1 - is_legal.float())
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_n = (x - mean) / std

    pos_weight = None
    if use_pos_weight:
        n_pos = y.sum().item()
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    probe = nn.Linear(x_n.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        logits = probe(x_n).squeeze(-1)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

    # Return the probe direction in the *original* (unstandardized) basis
    # by undoing the standardization: w_orig = w_std / std.
    w = probe.weight.detach().squeeze(0)
    w_orig = w / std.squeeze(0)
    return w_orig.numpy()


def get_mlp_out_projections(model):
    """Return list of W_out matrices (one per block), each shape (d_model, d_mlp)."""
    projections = []
    for block in model.transformer.h:
        # nanoGPT MLP has c_fc (d_model -> d_mlp) and c_proj (d_mlp -> d_model)
        c_proj = block.mlp.c_proj
        # Linear stores weight as (out, in), so c_proj.weight shape is (d_model, d_mlp)
        projections.append(c_proj.weight.detach().cpu().numpy())
    return projections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--residual-probe-layer", type=int, default=12,
                   help="Residual stream layer index (1-indexed: 0=embed, "
                        "1=after blk0, ..., 16=after blk15). The probe is "
                        "trained on this layer; neurons in earlier blocks "
                        "are scored.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    payload = load_neuron_examples(resolve_path(args.dataset))
    activations = payload["activations"]
    is_legal = payload["is_legal"].int()
    config = payload.get("config", {})
    checkpoint_name = config.get("checkpoint", "stockfish_16layers_ckpt_no_optimizer.pt")

    layer = args.residual_probe_layer
    n_slots = activations.shape[1]
    if not (0 <= layer < n_slots):
        raise SystemExit(f"--residual-probe-layer {layer} out of range [0,{n_slots-1}]")

    print(f"Training residual-stream probe at layer slot {layer} "
          f"({'embed' if layer == 0 else f'after blk{layer-1}'})...")
    direction = train_residual_probe(activations, is_legal, layer)

    print("Loading model to access W_out projections...")
    model, cfg = load_chess_gpt_for_weights(Path(args.repo), checkpoint_name)
    projections = get_mlp_out_projections(model)
    n_blocks = len(projections)
    d_mlp = projections[0].shape[1]
    print(f"Model: {n_blocks} blocks, d_mlp={d_mlp}")

    # Score neurons in blocks whose output is incorporated into the
    # residual stream at the chosen layer. layer=0 -> none. layer=k>0 ->
    # blocks 0..k-1 contributed.
    contributing_blocks = list(range(layer))
    print(f"Scoring neurons in blocks {contributing_blocks} (their output "
          f"feeds the residual stream at layer slot {layer}).")

    import numpy as np
    rows = []
    for block_idx in contributing_blocks:
        W_out = projections[block_idx]  # (d_model, d_mlp)
        # Project each neuron's write-direction onto the legality direction.
        # direction: (d_model,), W_out: (d_model, d_mlp)
        scores = direction @ W_out  # (d_mlp,)
        # Rank by absolute contribution.
        order = np.argsort(-np.abs(scores))[:args.top_k]
        for rank, n_idx in enumerate(order):
            rows.append({
                "block": block_idx,
                "neuron_idx": int(n_idx),
                "rank": rank,
                "score": float(scores[n_idx]),
                "abs_score": float(abs(scores[n_idx])),
            })

    out = resolve_path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "block", "neuron_idx", "rank", "score", "abs_score"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} neuron scores to {out}")
    print(f"(top {args.top_k} per block × {len(contributing_blocks)} blocks)")


if __name__ == "__main__":
    main()
