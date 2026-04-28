"""Compute and plot probe AUROC as a function of ply (moves into the game).

Loads the saved activation dataset, trains a logistic regression probe at a
chosen layer, runs it on all examples, then buckets the predictions by ply
and reports AUROC per bucket.

This answers: "Does the model become better (or worse) at encoding move
legality as the game progresses?"

Usage:
    uv run python plot_ply_analysis.py \\
        --dataset data/stockfish16_t1p3_n30000.pt \\
        --layer 12 \\
        --bucket-size 10 \\
        --out plots/auroc_by_ply.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path: Path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # activations: (n, n_layers+1, d_model)
    acts = payload["activations"].numpy().astype(np.float32)
    labels = payload["is_legal"].numpy().astype(np.int32)
    plies = payload["ply"].numpy().astype(np.int32)
    return acts, labels, plies


def train_probe(X: np.ndarray, y: np.ndarray):
    """Fit a logistic regression probe and return the model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                              solver="lbfgs", n_jobs=-1)
    clf.fit(X_scaled, y)
    return clf, scaler


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def plot_ply_auroc(
    dataset_path: Path,
    layer_idx: int,
    bucket_size: int,
    out_path: Path,
) -> None:
    print(f"Loading dataset from {dataset_path} …")
    acts, labels, plies = load_dataset(dataset_path)

    print(f"Dataset: {len(labels)} examples, {acts.shape[1]-1} transformer blocks")
    print(f"Illegal rate: {(1-labels).mean():.3%}")
    print(f"Ply range: {plies.min()} – {plies.max()}")

    X = acts[:, layer_idx, :]
    y = 1 - labels  # is_illegal (1 = illegal)

    print(f"Training logistic regression on layer {layer_idx} (all examples) …")
    clf, scaler = train_probe(X, y)
    scores = clf.predict_proba(scaler.transform(X))[:, 1]  # P(illegal)

    # Bucket by ply
    max_ply = int(plies.max())
    bucket_edges = list(range(0, max_ply + bucket_size, bucket_size))
    bucket_labels = []
    bucket_aurocs = []
    bucket_counts = []

    for lo in bucket_edges[:-1]:
        hi = lo + bucket_size
        mask = (plies >= lo) & (plies < hi)
        n = mask.sum()
        if n < 10:
            continue
        a = auroc(y[mask], scores[mask])
        bucket_labels.append(f"{lo}–{hi-1}")
        bucket_aurocs.append(a)
        bucket_counts.append(int(n))

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = range(len(bucket_labels))
    bars = ax.bar(xs, bucket_aurocs, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.5)")

    # Annotate with example counts
    for bar, n in zip(bars, bucket_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                str(n), ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"AUROC (layer {layer_idx})")
    ax.set_xlabel(f"Ply bucket (bucket size = {bucket_size})")
    ax.set_title(
        f"Probe AUROC by move depth — layer {layer_idx}\n"
        f"(numbers above bars = example count in bucket)"
    )
    ax.legend()
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path,
                   default=Path("data/stockfish16_t1p3_n30000.pt"))
    p.add_argument("--layer", type=int, default=12,
                   help="Layer index to evaluate (0=embed, 1=after blk0, …).")
    p.add_argument("--bucket-size", type=int, default=10,
                   help="Width of each ply bucket.")
    p.add_argument("--out", type=Path, default=Path("plots/auroc_by_ply.png"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_ply_auroc(args.dataset, args.layer, args.bucket_size, args.out)


if __name__ == "__main__":
    main()
