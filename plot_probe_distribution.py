"""Plot per-fold probe metric distributions across layers.

Consumes the per-fold CSV written by chess_gpt_probe.py (--per-fold-csv).

Usage:
    python plot_probe_distribution.py --csv per_fold.csv --out plots/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(csv_path: Path):
    """Return {metric: {layer_label: [per-fold values]}} keyed by insertion order."""
    per_layer = defaultdict(lambda: defaultdict(list))
    layer_order: list[str] = []
    seen = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            label = row["layer"]
            if label not in seen:
                layer_order.append(label)
                seen.add(label)
            for metric in ("test_auc", "test_acc", "test_loss"):
                per_layer[metric][label].append(float(row[metric]))
    return per_layer, layer_order


def plot_strip(per_layer, layer_order, metric, ylabel, out_path, hline=None,
               title=""):
    """Layer index on x, individual fold values as dots, mean overlaid."""
    data_by_layer = per_layer[metric]
    fig, ax = plt.subplots(figsize=(11, 5))
    means = []
    for i, label in enumerate(layer_order):
        vals = data_by_layer[label]
        xs = [i] * len(vals)
        ax.scatter(xs, vals, alpha=0.55, s=35, color="tab:blue",
                   edgecolor="white", linewidth=0.5, zorder=3)
        means.append(sum(vals) / len(vals))

    ax.plot(range(len(layer_order)), means, color="tab:red",
            linewidth=2, marker="o", markersize=5, label="mean", zorder=4)

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1,
                   label=f"chance ({hline})")

    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("layer")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sina(per_layer, layer_order, metric, ylabel, out_path, hline=None,
              title="", jitter_width=0.15, seed=0):
    """Sina plot: points jittered horizontally, with jitter width proportional
    to the local density at that y-value. Gives a sense of distribution shape
    while showing every individual fold value."""
    import random as _random
    import numpy as np

    rng = _random.Random(seed)
    data_by_layer = per_layer[metric]

    fig, ax = plt.subplots(figsize=(11, 5))
    means = []
    for i, label in enumerate(layer_order):
        vals = data_by_layer[label]
        if not vals:
            means.append(float("nan"))
            continue

        # Local-density-scaled jitter. With only 5 points per layer, a full
        # KDE is over-kill; use a simple spread based on proximity rank.
        if len(vals) == 1:
            jitters = [0.0]
        else:
            vals_sorted_idx = sorted(range(len(vals)), key=lambda k: vals[k])
            # Assign each point a deterministic-ish jitter based on rank,
            # then add a small random wiggle so identical values don't stack.
            jitters = [0.0] * len(vals)
            n = len(vals)
            for rank, original_idx in enumerate(vals_sorted_idx):
                # Spread evenly from -1 to 1 then scale
                pos = (rank - (n - 1) / 2) / max((n - 1) / 2, 1)
                wiggle = (rng.random() - 0.5) * 0.3
                jitters[original_idx] = (pos + wiggle) * jitter_width

        xs = [i + j for j in jitters]
        ax.scatter(xs, vals, alpha=0.7, s=40, color="tab:blue",
                   edgecolor="white", linewidth=0.5, zorder=3)
        means.append(sum(vals) / len(vals))

    ax.plot(range(len(layer_order)), means, color="tab:red",
            linewidth=2, marker="o", markersize=5, label="mean", zorder=4)

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1,
                   label=f"chance ({hline})")

    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("layer")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="plots")
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_layer, layer_order = load(Path(args.csv))

    plot_strip(per_layer, layer_order, "test_auc", "AUROC",
               out_dir / "auroc_distribution.png", hline=0.5,
               title=f"Per-layer AUROC — per-fold values{args.title_suffix}")
    plot_sina(per_layer, layer_order, "test_auc", "AUROC",
              out_dir / "auroc_sina.png", hline=0.5,
              title=f"Per-layer AUROC — sina plot{args.title_suffix}")
    plot_strip(per_layer, layer_order, "test_acc", "test accuracy",
               out_dir / "accuracy_distribution.png",
               title=f"Per-layer test accuracy — per-fold values{args.title_suffix}")

    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
