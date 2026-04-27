"""Generate Chess-GPT self-play games and capture both residual stream and
post-GELU MLP hidden activations.

This wraps the working ChessGPT class from the project's existing
generate_games.py to guarantee identical game-play and prompt construction
behavior. The only addition is post-GELU MLP activation capture during the
forward pass.

Usage:
    python generate_games_with_neurons.py \\
        --config configs/generation_neurons.yaml \\
        --repo C:\\Users\\monke\\chess_gpt_eval
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _paths import setup_paths, resolve_path
setup_paths()

torch = None
yaml = None
NeuronExample = None
save_neuron_examples = None
ChessGPT = None
load_runtime_dependencies = None
random_opening = None
parse_san_or_none = None


def load_runtime_imports() -> None:
    global torch, yaml, NeuronExample, save_neuron_examples
    global ChessGPT, load_runtime_dependencies, random_opening, parse_san_or_none
    if torch is None:
        import torch as _torch
        import yaml as _yaml
        from chess_probe_common_neurons import (
            NeuronExample as _NeuronExample,
            save_neuron_examples as _save_neuron_examples,
        )
        from generate_games import (
            ChessGPT as _ChessGPT,
            load_runtime_dependencies as _load_runtime_dependencies,
            random_opening as _random_opening,
            parse_san_or_none as _parse_san_or_none,
        )

        torch = _torch
        yaml = _yaml
        NeuronExample = _NeuronExample
        save_neuron_examples = _save_neuron_examples
        ChessGPT = _ChessGPT
        load_runtime_dependencies = _load_runtime_dependencies
        random_opening = _random_opening
        parse_san_or_none = _parse_san_or_none


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# MLP hook installation on top of the existing ChessGPT class
# ---------------------------------------------------------------------------


def find_gelu(mlp_module):
    """Find the GELU submodule inside a nanoGPT MLP block."""
    import torch.nn as nn
    for child in mlp_module.modules():
        if isinstance(child, nn.GELU):
            return child
        cls_name = type(child).__name__.lower()
        if "gelu" in cls_name:
            return child
    return None


class NeuronCaptureChessGPT:
    """Wraps a ChessGPT instance and captures post-GELU MLP activations
    alongside the residual-stream activations."""

    def __init__(self, base: ChessGPT):
        self.base = base
        self.model = base.model
        self.encode = base.encode
        self.decode = base.decode
        self.device = base.device
        self.n_layers = base.n_layers
        self.tokenizer_kind = base.tokenizer_kind

        # Locate GELU modules so we can hook them.
        self._gelus = []
        for i, block in enumerate(self.model.transformer.h):
            gelu = find_gelu(block.mlp)
            if gelu is None:
                raise RuntimeError(
                    f"Could not find GELU in block {i}'s MLP. "
                    f"MLP structure: {block.mlp}"
                )
            self._gelus.append(gelu)

    def compact_prompt(self, board):
        return self.base.compact_prompt(board)

    def generate_move_text(self, prompt, temperature, top_k, max_new_tokens):
        return self.base.generate_move_text(prompt, temperature, top_k, max_new_tokens)

    def activations_by_layer_with_mlp(self, prompt):
        """Return (residual_list, mlp_list).

        residual_list: same shape and ordering as ChessGPT.activations_by_layer
                       — n_layers + 1 tensors of shape (d_model,).
        mlp_list:      n_layers tensors of shape (d_mlp,) with the post-GELU
                       activation at the last token of each block's MLP.
        """
        x = self.base._prepare_input(prompt)
        residual_captured: list[torch.Tensor] = []
        mlp_captured: list[torch.Tensor] = []
        handles = []

        transformer = self.model.transformer
        first_block = transformer.h[0]

        # Same residual hooks as ChessGPT.activations_by_layer.
        def embed_pre_hook(_module, inputs):
            tensor = inputs[0]
            residual_captured.append(
                tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0)
            )
        handles.append(first_block.register_forward_pre_hook(embed_pre_hook))

        def make_block_hook():
            def hook(_module, _inputs, out):
                tensor = out[0] if isinstance(out, tuple) else out
                residual_captured.append(
                    tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0)
                )
            return hook

        for block in transformer.h:
            handles.append(block.register_forward_hook(make_block_hook()))

        # MLP GELU hooks for neuron capture.
        def make_gelu_hook():
            def hook(_module, _inputs, out):
                tensor = out[0] if isinstance(out, tuple) else out
                mlp_captured.append(
                    tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0)
                )
            return hook

        for gelu in self._gelus:
            handles.append(gelu.register_forward_hook(make_gelu_hook()))

        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for handle in handles:
                handle.remove()

        if len(residual_captured) != self.n_layers + 1:
            raise RuntimeError(
                f"Residual hook count mismatch: expected {self.n_layers + 1}, "
                f"got {len(residual_captured)}."
            )
        if len(mlp_captured) != self.n_layers:
            raise RuntimeError(
                f"MLP hook count mismatch: expected {self.n_layers}, "
                f"got {len(mlp_captured)}."
            )
        return residual_captured, mlp_captured


# ---------------------------------------------------------------------------
# Self-play loop (mirrors generate_games.play_self_game)
# ---------------------------------------------------------------------------


def play_self_game_with_neurons(
    chess_gpt: NeuronCaptureChessGPT,
    rng: random.Random,
    *,
    game_id: int,
    max_plies: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    random_opening_plies: int,
    stop_on_illegal: bool,
) -> list[NeuronExample]:
    board = random_opening(rng, random_opening_plies)
    examples: list[NeuronExample] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        prompt = chess_gpt.compact_prompt(board)
        residual, mlp_hidden = chess_gpt.activations_by_layer_with_mlp(prompt)
        move_text = chess_gpt.generate_move_text(
            prompt, temperature, top_k, max_new_tokens
        )
        legal_move = parse_san_or_none(board, move_text)
        is_legal = 1 if legal_move is not None else 0

        residual_tensor = torch.stack(residual, dim=0)   # (n_layers+1, d_model)
        mlp_tensor = torch.stack(mlp_hidden, dim=0)      # (n_layers, d_mlp)

        examples.append(NeuronExample(
            game_id=game_id,
            ply_index=board.ply(),
            is_legal=is_legal,
            residual=residual_tensor,
            mlp_hidden=mlp_tensor,
        ))

        if legal_move is not None:
            board.push(legal_move)
        else:
            if stop_on_illegal:
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))

    return examples


def collect_examples(
    chess_gpt: NeuronCaptureChessGPT,
    cfg: dict,
    rng: random.Random,
    progress_every: int = 100,
) -> list[NeuronExample]:
    examples: list[NeuronExample] = []
    target = cfg["positions"]
    game_id = 0
    last_reported = 0
    last_print = time.time()

    while len(examples) < target:
        new_examples = play_self_game_with_neurons(
            chess_gpt, rng,
            game_id=game_id,
            max_plies=cfg["max_plies"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            max_new_tokens=cfg["max_new_tokens"],
            random_opening_plies=cfg["random_opening_plies"],
            stop_on_illegal=cfg["stop_on_illegal"],
        )
        examples.extend(new_examples)
        game_id += 1

        now = time.time()
        if (len(examples) - last_reported >= progress_every
                or now - last_print > 5.0
                or len(examples) >= target):
            n_illegal = sum(1 for e in examples if e.is_legal == 0)
            pct = 100.0 * n_illegal / max(len(examples), 1)
            print(
                f"  progress: {len(examples)}/{target} examples, "
                f"{game_id} games, {n_illegal} illegal ({pct:.1f}%)"
            )
            last_reported = len(examples)
            last_print = now

    return examples[:target]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--repo", required=True,
                        help="Path to chess_gpt_eval repo (containing nanogpt/)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime_imports()

    with open(args.config) as f:
        cfg_full = yaml.safe_load(f)

    model_cfg = cfg_full["model"]
    gen_cfg = cfg_full["generation"]
    out_path = resolve_path(cfg_full["output"]["path"])

    rng = random.Random(gen_cfg["seed"])
    torch.manual_seed(gen_cfg["seed"])

    load_runtime_dependencies()

    device = resolve_device(gen_cfg["device"])

    base = ChessGPT(
        Path(args.repo).expanduser().resolve(),
        model_cfg["checkpoint"],
        device,
    )
    print(f"Loaded model: {base.n_layers} layers, tokenizer={base.tokenizer_kind}")
    print(f"Device: {device}")

    chess_gpt = NeuronCaptureChessGPT(base)

    # Sniff d_mlp by running one forward pass on the initial position.
    import chess as _chess_mod
    test_board = _chess_mod.Board()
    test_prompt = chess_gpt.compact_prompt(test_board)
    _, mlp_test = chess_gpt.activations_by_layer_with_mlp(test_prompt)
    d_mlp = mlp_test[0].shape[0]
    d_model = base.model.config.n_embd
    print(f"d_model={d_model}, d_mlp={d_mlp}")

    examples = collect_examples(chess_gpt, gen_cfg, rng)

    n_legal = sum(1 for e in examples if e.is_legal == 1)
    n_illegal = len(examples) - n_legal
    mean_ply = sum(e.ply_index for e in examples) / max(len(examples), 1)
    n_games = max(e.game_id for e in examples) + 1 if examples else 0

    print(f"\nCollected {len(examples)} examples from {n_games} games.")
    print(f"  legal: {n_legal}, illegal: {n_illegal} "
          f"({100.0 * n_illegal / max(len(examples), 1):.1f}%)")
    print(f"  mean ply index: {mean_ply:.1f}")

    config_record = {
        "checkpoint": model_cfg["checkpoint"],
        **gen_cfg,
        "device": device,
        "n_layers": base.n_layers,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "tokenizer_kind": base.tokenizer_kind,
    }
    save_neuron_examples(out_path, examples, config_record)
    print(f"Saved dataset to {out_path}")

    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    with sidecar.open("w") as f:
        json.dump({
            "config": config_record,
            "n_examples": len(examples),
            "n_legal": n_legal,
            "n_illegal": n_illegal,
            "mean_ply_index": mean_ply,
            "n_games": n_games,
        }, f, indent=2)
    print(f"Summary sidecar at {sidecar}")


if __name__ == "__main__":
    main()
