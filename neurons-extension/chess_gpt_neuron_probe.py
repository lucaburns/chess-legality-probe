"""Train per-block linear probes on MLP hidden activations (post-GELU) and
identify top-contributing neurons per block.

Operates on datasets produced by generate_games_with_neurons.py. The probe
input dimension is d_mlp (typically 2048), much larger than the residual
stream's d_model (512). Otherwise the training procedure mirrors
chess_gpt_probe.py.

Usage:
    python chess_gpt_neuron_probe.py --dataset data/.._neurons.pt
    python chess_gpt_neuron_probe.py --dataset ... --per-fold-csv ... --top-neurons-csv ... --device auto
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from _paths import setup_paths, resolve_path
setup_paths()

torch = None
load_neuron_examples = None


def load_runtime_imports() -> None:
    global torch, load_neuron_examples
    if torch is None:
        import torch as _torch
        from chess_probe_common_neurons import load_neuron_examples as _load_neuron_examples

        torch = _torch
        load_neuron_examples = _load_neuron_examples


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Reuse the AUROC implementation from the linear probe by copy.
def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
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
    i, n = 0, len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i: j + 1]]).mean()
            ranks[order[i: j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[: pos.numel()].sum().item()
    n_pos, n_neg = pos.numel(), neg.numel()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def kfold_indices(n: int, k: int, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(order):
        folds[i % k].append(idx)
    return folds


def train_one_fold(train_x, train_y, test_x, test_y, *,
                   epochs, lr, weight_decay, pos_weight, device):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    dim = train_x.shape[1]
    probe = torch.nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        logits = probe(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()

    eval_loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        train_logits = probe(train_x).squeeze(-1)
        test_logits = probe(test_x).squeeze(-1)
        train_acc = ((torch.sigmoid(train_logits) >= 0.5).float() == train_y).float().mean().item()
        test_acc = ((torch.sigmoid(test_logits) >= 0.5).float() == test_y).float().mean().item()
        test_loss = eval_loss_fn(test_logits, test_y).item()
        try:
            test_auc = auroc(torch.sigmoid(test_logits), test_y)
        except Exception:
            test_auc = float("nan")
    # Return weights too — needed for top-neuron extraction.
    weight = probe.weight.detach().cpu().squeeze(0).numpy()  # shape (d_mlp,)
    return train_acc, test_acc, test_loss, test_auc, weight


def probe_block(activations, is_legal, block_idx, folds, *,
                epochs, lr, weight_decay, use_pos_weight, device):
    x_all = activations[:, block_idx, :].float()
    y_all = (1 - is_legal.float())  # illegal = positive class

    accs, aucs, losses, train_accs, weights = [], [], [], [], []
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

        tr, te, lo, au, w = train_one_fold(
            train_x, train_y, test_x, test_y,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            pos_weight=pos_weight, device=device,
        )
        train_accs.append(tr); accs.append(te); losses.append(lo); aucs.append(au)
        weights.append(w)

    def stats(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return m, math.sqrt(v)

    auc_clean = [a for a in aucs if not math.isnan(a)] or [float("nan")]
    return {
        "train_acc": stats(train_accs)[0],
        "test_acc": stats(accs)[0], "test_acc_std": stats(accs)[1],
        "test_auc": stats(auc_clean)[0], "test_auc_std": stats(auc_clean)[1],
        "test_loss": stats(losses)[0],
        "fold_train_accs": train_accs, "fold_test_accs": accs,
        "fold_test_aucs": aucs, "fold_test_losses": losses,
        "fold_weights": weights,  # list of (d_mlp,) numpy arrays
    }


def top_neurons_from_weights(weights_per_fold, top_k=20):
    """Aggregate per-fold weights into a single ranking.

    For each neuron, take the mean absolute weight across folds (a neuron is
    'important' if it has consistently large weight regardless of sign).
    Returns list of (neuron_idx, mean_abs_weight, mean_signed_weight) sorted
    by mean_abs_weight descending.
    """
    import numpy as np
    W = np.stack(weights_per_fold, axis=0)  # (n_folds, d_mlp)
    mean_abs = np.mean(np.abs(W), axis=0)
    mean_signed = np.mean(W, axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return [(int(i), float(mean_abs[i]), float(mean_signed[i])) for i in order]


def run(args):
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    payload = load_neuron_examples(resolve_path(args.dataset))
    mlp_acts = payload["mlp_activations"]   # (N, n_blocks, d_mlp)
    is_legal = payload["is_legal"].int()
    config = payload.get("config", {})

    n_examples, n_blocks, d_mlp = mlp_acts.shape
    n_illegal = int((is_legal == 0).sum())
    n_legal = int((is_legal == 1).sum())
    print(f"Loaded dataset: {resolve_path(args.dataset)}")
    if config:
        print(f"  n_blocks={n_blocks}, d_mlp={d_mlp}")
        print(f"  checkpoint={config.get('checkpoint')} "
              f"temperature={config.get('temperature')} seed={config.get('seed')}")
    print(f"  {n_examples} examples; legal={n_legal}, illegal={n_illegal} "
          f"({100.0 * n_illegal / max(n_examples, 1):.1f}% illegal)")
    print(f"Device: {device}")

    if n_illegal == 0 or n_legal == 0:
        raise SystemExit("Single-class dataset; cannot train probe.")

    rng = random.Random(args.seed)
    folds = kfold_indices(n_examples, args.folds, rng)

    print()
    print("block  d_mlp  train_acc  test_acc ± std  test_auc ± std  test_loss")
    print("-" * 70)
    per_fold_rows = []
    top_neuron_rows = []

    for block_idx in range(n_blocks):
        s = probe_block(
            mlp_acts, is_legal, block_idx, folds,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            use_pos_weight=not args.no_pos_weight, device=device,
        )
        label = f"blk{block_idx}"
        print(
            f"{label:>5}  {d_mlp:<5}  {s['train_acc']:.3f}      "
            f"{s['test_acc']:.3f} ±{s['test_acc_std']:.3f}  "
            f"{s['test_auc']:.3f} ±{s['test_auc_std']:.3f}  "
            f"{s['test_loss']:.3f}"
        )
        for fold_idx in range(len(folds)):
            per_fold_rows.append({
                "layer": label, "layer_idx": block_idx, "fold": fold_idx,
                "train_acc": s["fold_train_accs"][fold_idx],
                "test_acc": s["fold_test_accs"][fold_idx],
                "test_auc": s["fold_test_aucs"][fold_idx],
                "test_loss": s["fold_test_losses"][fold_idx],
            })
        if args.top_neurons_csv:
            top = top_neurons_from_weights(s["fold_weights"], top_k=args.top_k)
            for rank, (n_idx, mabs, msgn) in enumerate(top):
                top_neuron_rows.append({
                    "block": block_idx, "rank": rank,
                    "neuron_idx": n_idx,
                    "mean_abs_weight": mabs, "mean_signed_weight": msgn,
                })

    if args.per_fold_csv:
        path = resolve_path(args.per_fold_csv); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "layer", "layer_idx", "fold",
                "train_acc", "test_acc", "test_auc", "test_loss"])
            w.writeheader(); w.writerows(per_fold_rows)
        print(f"\nSaved per-fold metrics to {path}")

    if args.top_neurons_csv and top_neuron_rows:
        path = resolve_path(args.top_neurons_csv); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "block", "rank", "neuron_idx",
                "mean_abs_weight", "mean_signed_weight"])
            w.writeheader(); w.writerows(top_neuron_rows)
        print(f"Saved top-neuron rankings to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', 'mps', or a torch device string.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--no-pos-weight", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per-fold-csv", default=None)
    p.add_argument("--top-neurons-csv", default=None)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    load_runtime_imports()
    run(args)


if __name__ == "__main__":
    main()
