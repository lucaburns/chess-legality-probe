"""Train linear legality probes on cached Chess-GPT activations.

This script consumes a dataset produced by generate_games.py. It does not
load the Chess-GPT model itself, which makes iteration on probe
hyperparameters very fast: generation is O(minutes to hours) on GPU,
probe training is O(seconds).

Usage:
    python chess_gpt_probe.py --dataset datasets/stockfish16_t1.3_n4000.pt
    python chess_gpt_probe.py --dataset ... --per-fold-csv per_fold.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch

from chess_probe_common import load_examples
from config_utils import flatten_sections, load_yaml_config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via Mann-Whitney U, robust to class imbalance and ties."""
    scores = scores.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1).long()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    all_scores = torch.cat([pos, neg])
    order = all_scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    sorted_scores = all_scores[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i: j + 1]]).mean()
            ranks[order[i: j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[: pos.numel()].sum().item()
    n_pos = pos.numel()
    n_neg = neg.numel()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def majority_baseline(labels: list[int] | torch.Tensor) -> float:
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if not labels:
        return 0.0
    ones = sum(int(x) for x in labels)
    return max(ones, len(labels) - ones) / len(labels)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """Full-batch AdamW on a linear probe. Returns
    (train_acc, test_acc, test_loss, test_auroc)."""
    dim = train_x.shape[1]
    probe = torch.nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        logits = probe(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        train_logits = probe(train_x).squeeze(-1)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        test_logits = probe(test_x).squeeze(-1)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
        test_loss = eval_loss_fn(test_logits, test_y).item()
        try:
            test_auc = auroc(torch.sigmoid(test_logits), test_y)
        except Exception:
            test_auc = float("nan")
    train_acc = (train_pred == train_y).float().mean().item()
    test_acc = (test_pred == test_y).float().mean().item()
    return train_acc, test_acc, test_loss, test_auc


def kfold_indices(n: int, k: int, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(order):
        folds[i % k].append(idx)
    return folds


def probe_layer(
    activations: torch.Tensor,
    is_legal: torch.Tensor,
    layer: int,
    folds: list[list[int]],
    epochs: int,
    lr: float,
    weight_decay: float,
    use_pos_weight: bool = True,
) -> dict:
    x_all = activations[:, layer, :].float()
    y_all = (1 - is_legal.float())

    accs: list[float] = []
    aucs: list[float] = []
    losses: list[float] = []
    train_accs: list[float] = []
    for i, test_idx in enumerate(folds):
        train_idx = [j for f, fold in enumerate(folds) if f != i for j in fold]
        train_x, train_y = x_all[train_idx], y_all[train_idx]
        test_x, test_y = x_all[test_idx], y_all[test_idx]

        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        pos_weight = None
        if use_pos_weight:
            n_pos = train_y.sum().item()
            n_neg = len(train_y) - n_pos
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        train_acc, test_acc, test_loss, test_auc = train_probe(
            train_x, train_y, test_x, test_y,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            pos_weight=pos_weight,
        )
        train_accs.append(train_acc)
        accs.append(test_acc)
        aucs.append(test_auc)
        losses.append(test_loss)

    def stats(xs: list[float]) -> tuple[float, float]:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return mean, math.sqrt(var)

    acc_mean, acc_std = stats(accs)
    auc_mean, auc_std = stats([a for a in aucs if not math.isnan(a)] or [float("nan")])
    train_mean, _ = stats(train_accs)
    loss_mean, _ = stats(losses)
    return {
        "train_acc": train_mean,
        "test_acc": acc_mean,
        "test_acc_std": acc_std,
        "test_auc": auc_mean,
        "test_auc_std": auc_std,
        "test_loss": loss_mean,
        "dim": x_all.shape[1],
        # Per-fold raw values for distribution plotting
        "fold_train_accs": train_accs,
        "fold_test_accs": accs,
        "fold_test_aucs": aucs,
        "fold_test_losses": losses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(args) -> None:
    dataset_path = Path(args.dataset).expanduser().resolve()
    payload = load_examples(dataset_path)
    activations: torch.Tensor = payload["activations"]
    is_legal: torch.Tensor = payload["is_legal"].int()
    config = payload.get("config", {})

    n_examples, n_slots, d_model = activations.shape
    n_illegal = int((is_legal == 0).sum())
    n_legal = int((is_legal == 1).sum())
    print(f"Loaded dataset: {dataset_path}")
    if config:
        print(
            f"  checkpoint={config.get('checkpoint')} "
            f"temperature={config.get('temperature')} "
            f"seed={config.get('seed')} "
            f"stop_on_illegal={config.get('stop_on_illegal')}"
        )
    print(
        f"  {n_examples} examples, {n_slots - 1} transformer blocks, "
        f"d_model={d_model}"
    )
    print(
        f"  legal: {n_legal}, illegal: {n_illegal} "
        f"({100.0 * n_illegal / max(n_examples, 1):.1f}% illegal)"
    )

    if n_illegal == 0 or n_legal == 0:
        raise SystemExit("Dataset has only one class; cannot train a probe.")

    baseline = majority_baseline(is_legal)
    print(f"Majority-class baseline: {baseline:.3f} (probe target is is_illegal=1)")

    rng = random.Random(args.seed)
    labels_list = is_legal.tolist()
    folds = kfold_indices(n_examples, args.folds, rng)
    for _ in range(10):
        if all(0 < sum(labels_list[j] for j in fold) < len(fold) for fold in folds):
            break
        folds = kfold_indices(n_examples, args.folds, rng)

    print()
    print("layer  dim    train_acc  test_acc ± std  test_auc ± std  test_loss")
    print("-" * 68)
    per_fold_rows = []
    for slot in range(n_slots):
        stats = probe_layer(
            activations, is_legal, slot, folds,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            use_pos_weight=not args.no_pos_weight,
        )
        label = "embed" if slot == 0 else f"blk{slot - 1}"
        print(
            f"{label:>5}  {stats['dim']:<5}  "
            f"{stats['train_acc']:.3f}      "
            f"{stats['test_acc']:.3f} ±{stats['test_acc_std']:.3f}  "
            f"{stats['test_auc']:.3f} ±{stats['test_auc_std']:.3f}  "
            f"{stats['test_loss']:.3f}"
        )

        for fold_idx in range(len(folds)):
            per_fold_rows.append({
                "layer": label,
                "layer_idx": slot,
                "fold": fold_idx,
                "train_acc": stats["fold_train_accs"][fold_idx],
                "test_acc": stats["fold_test_accs"][fold_idx],
                "test_auc": stats["fold_test_aucs"][fold_idx],
                "test_loss": stats["fold_test_losses"][fold_idx],
            })

    if args.per_fold_csv:
        out_path = Path(args.per_fold_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer", "layer_idx", "fold",
                "train_acc", "test_acc", "test_auc", "test_loss",
            ])
            writer.writeheader()
            writer.writerows(per_fold_rows)
        print(f"\nSaved per-fold metrics to {out_path}")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to a YAML config file.")
    config_args, remaining = config_parser.parse_known_args()

    defaults = {
        "dataset": None,
        "epochs": 200,
        "lr": 1e-2,
        "weight_decay": 1e-2,
        "no_pos_weight": False,
        "folds": 5,
        "seed": 7,
        "per_fold_csv": None,
    }

    if config_args.config:
        config = load_yaml_config(config_args.config)
        yaml_values = flatten_sections(config, "paths", "probe")
        defaults.update(
            {
                "dataset": yaml_values.get("dataset", defaults["dataset"]),
                "epochs": yaml_values.get("epochs", defaults["epochs"]),
                "lr": yaml_values.get("lr", defaults["lr"]),
                "weight_decay": yaml_values.get("weight_decay", defaults["weight_decay"]),
                "no_pos_weight": yaml_values.get(
                    "no_pos_weight", defaults["no_pos_weight"]
                ),
                "folds": yaml_values.get("folds", defaults["folds"]),
                "seed": yaml_values.get("seed", defaults["seed"]),
            }
        )

    parser = argparse.ArgumentParser(
        description="Train legality probes on a saved dataset.",
        parents=[config_parser],
    )
    parser.add_argument("--dataset",
                        help="Path to a dataset saved by generate_games.py.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="Disable class-imbalance reweighting in the probe loss.")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--per-fold-csv", default=None,
                        help="Optional path to write one CSV row per (layer, fold).")
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    if args.dataset is None:
        parser.error("--dataset is required unless provided via --config.")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
