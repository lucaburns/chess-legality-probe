"""Visualize neuron-level analysis results.

Generates four plots:
  1. AUROC comparison: residual-stream probe vs. neuron-basis probe per block.
  2. Top-neuron weight magnitudes per block.
  3. Direction analysis heatmap: top neurons per block by |projection score|.
  4. Concentration: cumulative weight share of top-k neurons per block.

Usage:
    python plot_neuron_results.py \\
        --neuron-csv data/per_fold_neurons.csv \\
        --residual-csv data/per_fold_t1p3_n30000.csv \\
        --top-neurons-csv data/top_neurons.csv \\
        --direction-csv data/direction_analysis.csv \\
        --out plots/neurons
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from _paths import setup_paths, resolve_path
setup_paths()


def load_per_fold(path: Path):
    by_layer = defaultdict(list)
    layer_to_idx = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            label = row["layer"]
            by_layer[label].append(float(row["test_auc"]))
            layer_to_idx[label] = int(row["layer_idx"])
    return by_layer, layer_to_idx


def mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def plot_auroc_comparison(neuron_csv, residual_csv, out_path, title=""):
    n_by, n_idx = load_per_fold(neuron_csv)
    r_by, r_idx = load_per_fold(residual_csv)

    # Neuron CSV blocks are blk0..blk{N-1}. Residual CSV has embed + blk0..blk{N-1}.
    # We align by block name.
    blocks = sorted(n_by.keys(), key=lambda k: n_idx[k])

    n_means, n_stds, r_means, r_stds = [], [], [], []
    for b in blocks:
        nm, ns = mean_std(n_by[b])
        n_means.append(nm); n_stds.append(ns)
        if b in r_by:
            rm, rs = mean_std(r_by[b])
        else:
            rm, rs = float("nan"), 0.0
        r_means.append(rm); r_stds.append(rs)

    x = list(range(len(blocks)))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(x, [m - s for m, s in zip(r_means, r_stds)],
                       [m + s for m, s in zip(r_means, r_stds)],
                    alpha=0.2, color="tab:blue")
    ax.plot(x, r_means, marker="o", color="tab:blue",
            label="Residual-stream probe (d=512)")
    ax.fill_between(x, [m - s for m, s in zip(n_means, n_stds)],
                       [m + s for m, s in zip(n_means, n_stds)],
                    alpha=0.2, color="tab:green")
    ax.plot(x, n_means, marker="s", color="tab:green",
            label="MLP-neuron probe (d=2048)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(blocks, rotation=45, ha="right")
    ax.set_ylabel("AUROC"); ax.set_xlabel("block")
    ax.set_title(title or "Probe AUROC: residual stream vs. MLP neurons")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_top_neuron_weights(top_csv, out_path, title=""):
    """Bar chart of top-k neuron mean-abs weights per block."""
    by_block = defaultdict(list)
    with Path(top_csv).open() as f:
        for row in csv.DictReader(f):
            by_block[int(row["block"])].append(float(row["mean_abs_weight"]))

    blocks = sorted(by_block.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    for b in blocks:
        vals = sorted(by_block[b], reverse=True)
        ax.plot(range(len(vals)), vals, marker=".", alpha=0.7,
                label=f"blk{b}" if b in (0, len(blocks) // 2, len(blocks) - 1) else None)
    ax.set_xlabel("rank within block (0 = highest |weight|)")
    ax.set_ylabel("mean |probe weight| across folds")
    ax.set_title(title or "Top-neuron probe weights per block")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_direction_heatmap(direction_csv, out_path, title=""):
    """Heatmap: rows = blocks, columns = top-k ranks, color = |projection score|."""
    import numpy as np
    rows = []
    with Path(direction_csv).open() as f:
        for row in csv.DictReader(f):
            rows.append((int(row["block"]), int(row["rank"]), float(row["abs_score"])))
    if not rows:
        print(f"No rows in {direction_csv}; skipping heatmap.")
        return
    blocks = sorted({r[0] for r in rows})
    max_rank = max(r[1] for r in rows) + 1
    grid = np.full((len(blocks), max_rank), float("nan"))
    block_to_row = {b: i for i, b in enumerate(blocks)}
    for b, rank, abs_score in rows:
        grid[block_to_row[b], rank] = abs_score

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xlabel("rank (top-k neurons by |projection on legality direction|)")
    ax.set_ylabel("block")
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"blk{b}" for b in blocks])
    ax.set_title(title or "MLP-neuron contribution to legality direction")
    fig.colorbar(im, ax=ax, label="|projection score|")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_direction_concentration(direction_csv, out_path, title=""):
    """Cumulative |score| share among top-k neurons per block."""
    import numpy as np
    by_block = defaultdict(list)
    with Path(direction_csv).open() as f:
        for row in csv.DictReader(f):
            by_block[int(row["block"])].append(float(row["abs_score"]))

    blocks = sorted(by_block.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    for b in blocks:
        vals = np.array(sorted(by_block[b], reverse=True))
        cum = np.cumsum(vals) / vals.sum()
        ax.plot(range(1, len(cum) + 1), cum, marker=".", alpha=0.7,
                label=f"blk{b}" if b in (0, len(blocks) // 2, len(blocks) - 1) else None)
    ax.set_xlabel("top-k rank")
    ax.set_ylabel("cumulative share of |projection score|")
    ax.set_title(title or "Concentration of legality contribution across top neurons")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron-csv", required=True,
                   help="per_fold CSV from chess_gpt_neuron_probe.py")
    p.add_argument("--residual-csv", required=True,
                   help="per_fold CSV from chess_gpt_probe.py")
    p.add_argument("--top-neurons-csv", required=True,
                   help="top_neurons CSV from chess_gpt_neuron_probe.py")
    p.add_argument("--direction-csv", required=True,
                   help="output of analyze_legality_directions.py")
    p.add_argument("--out", default="plots/neurons")
    p.add_argument("--title-suffix", default="")
    args = p.parse_args()

    out = resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)

    plot_auroc_comparison(
        resolve_path(args.neuron_csv), resolve_path(args.residual_csv),
        out / "auroc_comparison.png",
        title=f"AUROC: residual vs. neuron probe{args.title_suffix}",
    )
    plot_top_neuron_weights(
        resolve_path(args.top_neurons_csv),
        out / "top_neuron_weights.png",
        title=f"Top neuron weights per block{args.title_suffix}",
    )
    plot_direction_heatmap(
        resolve_path(args.direction_csv),
        out / "direction_heatmap.png",
        title=f"Neuron contributions to legality direction{args.title_suffix}",
    )
    plot_direction_concentration(
        resolve_path(args.direction_csv),
        out / "direction_concentration.png",
        title=f"Top-neuron concentration{args.title_suffix}",
    )
    print(f"Saved plots to {out.resolve()}")


if __name__ == "__main__":
    main()
