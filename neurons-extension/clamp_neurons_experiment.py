"""Neuron-clamping intervention experiment.

Loads neurons identified by chess_gpt_neuron_probe.py (top_neurons.csv),
registers forward hooks on the model's MLP GELUs that zero or scale those
neurons' activations during inference, then plays new self-play games
under each intervention and measures illegal-move rate vs. baseline.

Sweeps:
    - neurons per block: {5, 20, 50, 100}
    - steering coefficient: {0, 0.5, 1.0, 2.0, 4.0}
        (0 = baseline/no intervention, 1.0 = full zero-out,
         >1 = push past zero, <1 = partial)

The intervention formula for a neuron n with pre-hook activation a:
    a' = a - coeff * a   (i.e., a' = (1 - coeff) * a)
So coeff=0 leaves a unchanged; coeff=1 zeroes it; coeff=2 flips and doubles.
This is equivalent to projecting out `coeff * a` from the neuron.

Usage:
    python clamp_neurons_experiment.py \\
        --dataset data/stockfish16_t1p3_n30000_neurons.pt \\
        --repo C:\\Users\\monke\\chess_gpt_eval \\
        --top-neurons-csv data/top_neurons.csv \\
        --output data/clamp_sweep_results.csv \\
        --eval-positions 2000
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

from _paths import setup_paths, resolve_path
setup_paths()

torch = None
load_neuron_examples = None
ChessGPT = None
load_runtime_dependencies = None
random_opening = None
parse_san_or_none = None
NeuronCaptureChessGPT = None
find_gelu = None


def load_runtime_imports() -> None:
    global torch, load_neuron_examples, ChessGPT, load_runtime_dependencies
    global random_opening, parse_san_or_none, NeuronCaptureChessGPT, find_gelu
    if torch is None:
        import torch as _torch
        from chess_probe_common_neurons import load_neuron_examples as _load_neuron_examples
        from generate_games import (
            ChessGPT as _ChessGPT,
            load_runtime_dependencies as _load_runtime_dependencies,
            random_opening as _random_opening,
            parse_san_or_none as _parse_san_or_none,
        )
        from generate_games_with_neurons import (
            NeuronCaptureChessGPT as _NeuronCaptureChessGPT,
            find_gelu as _find_gelu,
        )

        torch = _torch
        load_neuron_examples = _load_neuron_examples
        ChessGPT = _ChessGPT
        load_runtime_dependencies = _load_runtime_dependencies
        random_opening = _random_opening
        parse_san_or_none = _parse_san_or_none
        NeuronCaptureChessGPT = _NeuronCaptureChessGPT
        find_gelu = _find_gelu


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Neuron selection from top_neurons.csv
# ---------------------------------------------------------------------------


def load_top_neurons(csv_path: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {block: [(neuron_idx, mean_abs_weight), ...]} sorted by weight desc.

    CSV columns: block, rank, neuron_idx, mean_abs_weight, mean_signed_weight
    """
    by_block: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            by_block[int(row["block"])].append((
                int(row["neuron_idx"]),
                float(row["mean_abs_weight"]),
            ))
    # Already ranked in CSV, but re-sort defensively.
    for b in by_block:
        by_block[b].sort(key=lambda t: -t[1])
    return dict(by_block)


def pick_top_k_per_block(
    by_block: dict[int, list[tuple[int, float]]], k: int
) -> dict[int, list[int]]:
    """Return {block: [neuron_idx,...]} taking the top-k neurons per block."""
    return {b: [n for n, _ in pairs[:k]] for b, pairs in by_block.items()}


# ---------------------------------------------------------------------------
# Intervention hooks
# ---------------------------------------------------------------------------


class NeuronClampHooks:
    """Install forward hooks on each block's MLP GELU that modify selected
    neurons' activations by: a' = a - coeff * a.

    Applied only at the last token position — the position about to be
    sampled. This matches the semantics of the probe (legality of *next
    move*), rather than retroactively perturbing earlier positions.
    """

    def __init__(self, gelus: list, clamp_spec: dict[int, list[int]],
                 coeff: float, device):
        self.gelus = gelus
        self.clamp_spec = clamp_spec  # {block_idx: [neuron_indices]}
        self.coeff = coeff
        self.device = device
        self.handles = []

    def _make_hook(self, block_idx: int):
        neurons = self.clamp_spec.get(block_idx, [])
        if not neurons or self.coeff == 0.0:
            return None
        idx_tensor = torch.tensor(neurons, dtype=torch.long, device=self.device)
        coeff = self.coeff

        def hook(_module, _inputs, out):
            # `out` shape: (B, T, d_mlp). Only touch the last token position.
            out[:, -1, idx_tensor] = out[:, -1, idx_tensor] * (1.0 - coeff)
            return out

        return hook

    def __enter__(self):
        for i, gelu in enumerate(self.gelus):
            h = self._make_hook(i)
            if h is not None:
                handle = gelu.register_forward_hook(h)
                self.handles.append(handle)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


# ---------------------------------------------------------------------------
# Self-play eval under intervention
# ---------------------------------------------------------------------------


def play_game_simple(chess_gpt, rng, *, max_plies, temperature, top_k,
                      max_new_tokens, random_opening_plies, stop_on_illegal):
    """Play one self-play game and return list of (is_legal, ply_index)."""
    board = random_opening(rng, random_opening_plies)
    results = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        prompt = chess_gpt.compact_prompt(board)
        move_text = chess_gpt.generate_move_text(
            prompt, temperature, top_k, max_new_tokens
        )
        legal_move = parse_san_or_none(board, move_text)
        is_legal = 1 if legal_move is not None else 0
        results.append((is_legal, board.ply()))

        if legal_move is not None:
            board.push(legal_move)
        else:
            if stop_on_illegal:
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))

    return results


def eval_config(chess_gpt, gelus, clamp_spec, coeff, gen_cfg,
                rng, target_positions, label):
    """Run games under the given clamp config until target_positions are collected."""
    t0 = time.time()
    examples = []
    game_id = 0
    last_print = time.time()

    with NeuronClampHooks(gelus, clamp_spec, coeff, chess_gpt.device):
        while len(examples) < target_positions:
            game_results = play_game_simple(
                chess_gpt, rng,
                max_plies=gen_cfg["max_plies"],
                temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                random_opening_plies=gen_cfg["random_opening_plies"],
                stop_on_illegal=gen_cfg["stop_on_illegal"],
            )
            examples.extend(game_results)
            game_id += 1

            now = time.time()
            if now - last_print > 10.0:
                n_illegal = sum(1 for r in examples if r[0] == 0)
                pct = 100.0 * n_illegal / max(len(examples), 1)
                print(f"    [{label}] {len(examples)}/{target_positions}, "
                      f"{game_id} games, {n_illegal} illegal ({pct:.2f}%)")
                last_print = now

    examples = examples[:target_positions]
    n_illegal = sum(1 for r in examples if r[0] == 0)
    n_legal = len(examples) - n_illegal
    mean_ply = sum(r[1] for r in examples) / max(len(examples), 1)
    elapsed = time.time() - t0
    return {
        "n_positions": len(examples),
        "n_games": game_id,
        "n_legal": n_legal,
        "n_illegal": n_illegal,
        "illegal_rate": n_illegal / max(len(examples), 1),
        "mean_ply": mean_ply,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Existing neurons dataset (used to read config only).")
    p.add_argument("--repo", required=True)
    p.add_argument("--top-neurons-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-positions", type=int, default=2000,
                   help="Positions to collect per (top_k, coeff) config. "
                        "More = tighter illegal-rate estimates, slower sweep.")
    p.add_argument("--top-k-sweep", nargs="+", type=int,
                   default=[5, 20, 50, 100])
    p.add_argument("--coeff-sweep", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', 'mps', or a torch device string.")
    p.add_argument("--seed", type=int, default=1234,
                   help="Different from training seed to avoid repeated games.")
    return p.parse_args()


def main():
    args = parse_args()
    load_runtime_imports()

    load_runtime_dependencies()

    # Pull config from the dataset sidecar so we use the same gen params.
    payload = load_neuron_examples(resolve_path(args.dataset))
    ds_config = payload.get("config", {})
    gen_cfg = {
        "max_plies": ds_config.get("max_plies", 160),
        "temperature": ds_config.get("temperature", 1.3),
        "top_k": ds_config.get("top_k", 200),
        "max_new_tokens": ds_config.get("max_new_tokens", 10),
        "random_opening_plies": ds_config.get("random_opening_plies", 2),
        "stop_on_illegal": ds_config.get("stop_on_illegal", True),
    }
    checkpoint_name = ds_config.get(
        "checkpoint", "stockfish_16layers_ckpt_no_optimizer.pt"
    )

    device = resolve_device(args.device)
    base = ChessGPT(Path(args.repo).expanduser().resolve(),
                    checkpoint_name, device)
    chess_gpt = NeuronCaptureChessGPT(base)
    print(f"Loaded model on {device}. {base.n_layers} blocks.")

    # Free up the dataset from memory — we don't need it for eval.
    del payload

    top_neurons_by_block = load_top_neurons(resolve_path(args.top_neurons_csv))
    available_blocks = sorted(top_neurons_by_block.keys())
    print(f"Top-neuron data for blocks: {available_blocks}")
    max_k_available = min(len(v) for v in top_neurons_by_block.values())
    print(f"Max neurons per block in CSV: {max_k_available}")

    for k in args.top_k_sweep:
        if k > max_k_available:
            print(f"Warning: top-{k} exceeds CSV availability ({max_k_available}). "
                  f"Will clip to {max_k_available}.")

    # Establish baseline first (coeff=0 is mathematically identical regardless
    # of top_k; we run it once). Then sweep.
    sweep_rows = []

    print(f"\n=== Baseline (no intervention) ===")
    rng = random.Random(args.seed)
    baseline = eval_config(
        chess_gpt, chess_gpt._gelus, {}, coeff=0.0,
        gen_cfg=gen_cfg, rng=rng, target_positions=args.eval_positions,
        label="baseline",
    )
    print(f"  illegal_rate={baseline['illegal_rate']:.4f} "
          f"({baseline['n_illegal']}/{baseline['n_positions']}) "
          f"in {baseline['elapsed_sec']:.0f}s")
    sweep_rows.append({
        "top_k": 0, "coeff": 0.0, **baseline,
    })

    for k in args.top_k_sweep:
        k_eff = min(k, max_k_available)
        clamp_spec = pick_top_k_per_block(top_neurons_by_block, k_eff)
        total_neurons = sum(len(v) for v in clamp_spec.values())

        for coeff in args.coeff_sweep:
            if coeff == 0.0:
                # Skip — identical to baseline, we already ran it.
                continue
            label = f"k={k_eff},c={coeff}"
            print(f"\n=== {label} ({total_neurons} neurons clamped) ===")
            rng = random.Random(args.seed)  # reset so RNG differences don't
                                             # confound intervention effect
            result = eval_config(
                chess_gpt, chess_gpt._gelus, clamp_spec, coeff=coeff,
                gen_cfg=gen_cfg, rng=rng, target_positions=args.eval_positions,
                label=label,
            )
            delta = result["illegal_rate"] - baseline["illegal_rate"]
            pct_change = (100.0 * delta / max(baseline["illegal_rate"], 1e-9))
            print(f"  illegal_rate={result['illegal_rate']:.4f} "
                  f"(baseline {baseline['illegal_rate']:.4f}, "
                  f"delta {delta:+.4f} = {pct_change:+.1f}%) "
                  f"in {result['elapsed_sec']:.0f}s")
            sweep_rows.append({
                "top_k": k_eff, "coeff": coeff,
                "n_neurons_clamped": total_neurons,
                **result,
                "delta_vs_baseline": delta,
                "pct_change_vs_baseline": pct_change,
            })

    # Save results
    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "top_k", "coeff", "n_neurons_clamped",
        "n_positions", "n_games", "n_legal", "n_illegal",
        "illegal_rate", "mean_ply", "elapsed_sec",
        "delta_vs_baseline", "pct_change_vs_baseline",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sweep_rows:
            # Fill missing fields (baseline row) with empty strings
            for k in fieldnames:
                row.setdefault(k, "")
            w.writerow(row)

    print(f"\n\n=== Summary (saved to {out_path}) ===")
    print(f"{'top_k':>6} {'coeff':>6} {'illegal_rate':>12} {'delta':>10} {'%change':>10}")
    print("-" * 52)
    for row in sweep_rows:
        print(f"{str(row['top_k']):>6} {str(row['coeff']):>6} "
              f"{row['illegal_rate']:>12.4f} "
              f"{row.get('delta_vs_baseline', 0.0):>+10.4f} "
              f"{row.get('pct_change_vs_baseline', 0.0):>+9.1f}%")


if __name__ == "__main__":
    main()
