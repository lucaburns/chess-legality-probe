"""Generate headline comparison plots for the legality probe report.

Produces two figures:
  1. auroc_by_layer.png  — AUROC vs transformer layer depth for the linear
     probe, MLP probe, and neuron-level probe, with per-fold error bars.
  2. clamp_sweep.png     — illegal-rate change vs clamp coefficient, grouped
     by number of neurons clamped per block.

Usage:
    uv run python plot_comparison.py \\
        --linear-csv  data/per_fold_t1p3_n30000.csv \\
        --mlp-csv     data/per_fold_mlp_t1p3_n30000.csv \\
        --neuron-csv  data/per_fold_neurons.csv \\
        --clamp-csv   data/clamp_sweep_results.csv \\
        --out         plots/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_per_fold(csv_path: Path) -> tuple[dict, list[str]]:
    """Return ({layer_label: [per-fold AUROC]}, layer_order)."""
    per_layer: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    seen: set[str] = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            label = row["layer"]
            if label not in seen:
                order.append(label)
                seen.add(label)
            per_layer[label].append(float(row["test_auc"]))
    return dict(per_layer), order


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else (float(arr.mean()), 0.0)


# ---------------------------------------------------------------------------
# Plot 1: AUROC by layer — linear vs MLP vs neuron
# ---------------------------------------------------------------------------

def plot_auroc_comparison(
    linear_csv: Path,
    mlp_csv: Path,
    neuron_csv: Path,
    out_path: Path,
) -> None:
    linear_data, linear_order = load_per_fold(linear_csv)
    mlp_data, mlp_order = load_per_fold(mlp_csv)
    neuron_data, neuron_order = load_per_fold(neuron_csv)

    # Use the linear probe's layer ordering as the x-axis spine.
    # Neuron probe is indexed by block (blk0…blkN) same naming.
    # Align neuron probe to same x positions as the corresponding block.
    # Linear order: embed, blk0, blk1, …, blk15  (17 points, indices 0..16)
    # Neuron order: blk0, blk1, …, blk15           (16 points, same block names)

    x_labels = linear_order  # embed + 16 blocks
    xs_linear = list(range(len(x_labels)))

    # For MLP probe: same labels
    xs_mlp = list(range(len(mlp_order)))
    assert mlp_order == linear_order, "MLP and linear probe layers should match"

    # For neuron probe: map blkN -> index in linear_order
    label_to_idx = {lbl: i for i, lbl in enumerate(x_labels)}
    xs_neuron = [label_to_idx[lbl] for lbl in neuron_order if lbl in label_to_idx]
    neuron_order_filtered = [lbl for lbl in neuron_order if lbl in label_to_idx]

    def extract_series(data, order):
        means, stds = [], []
        for lbl in order:
            m, s = mean_std(data[lbl])
            means.append(m)
            stds.append(s)
        return np.array(means), np.array(stds)

    lin_m, lin_s = extract_series(linear_data, linear_order)
    mlp_m, mlp_s = extract_series(mlp_data, mlp_order)
    neu_m, neu_s = extract_series(neuron_data, neuron_order_filtered)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.errorbar(xs_linear, lin_m, yerr=lin_s, fmt="o-", color="tab:blue",
                capsize=3, linewidth=1.8, markersize=5,
                label="Residual-stream linear probe")
    ax.errorbar(xs_mlp, mlp_m, yerr=mlp_s, fmt="s--", color="tab:orange",
                capsize=3, linewidth=1.8, markersize=5,
                label="Residual-stream MLP probe")
    ax.errorbar(xs_neuron, neu_m, yerr=neu_s, fmt="^:", color="tab:green",
                capsize=3, linewidth=1.8, markersize=5,
                label="Post-GELU neuron-level probe")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.5)")
    ax.set_xticks(xs_linear)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUROC (5-fold CV, mean ± 1 SD)")
    ax.set_xlabel("Layer / block")
    ax.set_title("Probe AUROC by layer depth — linear vs MLP vs neuron-level probe")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0.45, 0.80)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Clamp sweep — illegal-rate change vs coefficient
# ---------------------------------------------------------------------------

def plot_clamp_sweep(clamp_csv: Path, out_path: Path) -> None:
    rows: list[dict] = []
    with clamp_csv.open() as f:
        for row in csv.DictReader(f):
            if row["top_k"] == "0":
                continue  # baseline row
            rows.append({
                "top_k": int(row["top_k"]),
                "coeff": float(row["coeff"]),
                "pct_change": float(row["pct_change_vs_baseline"]),
                "illegal_rate": float(row["illegal_rate"]),
            })

    # Group by top_k
    by_k: dict[int, list] = defaultdict(list)
    for r in rows:
        by_k[r["top_k"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (k, group), color in zip(sorted(by_k.items()), colors):
        group_sorted = sorted(group, key=lambda r: r["coeff"])
        coeffs = [r["coeff"] for r in group_sorted]
        pcts = [r["pct_change"] for r in group_sorted]
        n_neurons = group_sorted[0].get("n_neurons_clamped", "?") if group_sorted else "?"
        ax.plot(coeffs, pcts, "o-", color=color, linewidth=1.8, markersize=6,
                label=f"top-{k} neurons/block ({n_neurons} total)")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, label="No change")
    ax.set_xlabel("Clamp coefficient  (0 = no change, 1 = zero-out, 2 = flip+double)")
    ax.set_ylabel("% change in illegal-move rate vs baseline")
    ax.set_title("Effect of neuron clamping on illegal-move rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--linear-csv", type=Path,
                   default=Path("data/per_fold_t1p3_n30000.csv"))
    p.add_argument("--mlp-csv", type=Path,
                   default=Path("data/per_fold_mlp_t1p3_n30000.csv"))
    p.add_argument("--neuron-csv", type=Path,
                   default=Path("data/per_fold_neurons.csv"))
    p.add_argument("--clamp-csv", type=Path,
                   default=Path("data/clamp_sweep_results.csv"))
    p.add_argument("--out", type=Path, default=Path("plots"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    plot_auroc_comparison(
        args.linear_csv, args.mlp_csv, args.neuron_csv,
        args.out / "auroc_by_layer.png",
    )
    plot_clamp_sweep(args.clamp_csv, args.out / "clamp_sweep.png")


if __name__ == "__main__":
    main()
