"""Layer-wise legality probe for Adam Karvonen's Chess-GPT checkpoints.

This script expects a local checkout of:
https://github.com/adamkarvonen/chess_gpt_eval

It loads a NanoGPT chess checkpoint from that repo, generates moves from random
reachable positions, captures residual-stream activations immediately before
move generation, and trains one linear probe per transformer layer to predict
whether the generated move is legal.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib.util
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Callable

torch = None
chess = None


def require(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {module_name}\n"
            "Install the Chess-GPT dependencies first:\n"
            "  python3 -m pip install torch python-chess tiktoken numpy\n"
            "The original repo also recommends transformers, datasets, wandb, and tqdm."
        ) from exc


def load_runtime_dependencies() -> None:
    global torch, chess
    if torch is None:
        torch = require("torch")
    if chess is None:
        chess = require("chess")


@dataclass
class Example:
    layer_activations: list["torch.Tensor"]
    is_legal: int
    move_text: str
    prompt: str


def import_nanogpt_model(repo: Path):
    model_path = repo / "nanogpt" / "model.py"
    if not model_path.exists():
        raise SystemExit(f"Could not find {model_path}. Pass --repo /path/to/chess_gpt_eval.")
    spec = importlib.util.spec_from_file_location("chess_gpt_eval_nanogpt_model", model_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import NanoGPT model from {model_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GPT, module.GPTConfig


def load_tokenizer(repo: Path, checkpoint: dict) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    meta_path = repo / "nanogpt" / "out" / "meta.pkl"
    if "config" in checkpoint and "dataset" in checkpoint["config"] and meta_path.exists():
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        stoi, itos = meta["stoi"], meta["itos"]
        return lambda text: [stoi[c] for c in text], lambda ids: "".join(itos[i] for i in ids)

    tiktoken = require("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    return (
        lambda text: enc.encode(text, allowed_special={"<|endoftext|>"}),
        lambda ids: enc.decode(ids),
    )


class ChessGPT:
    def __init__(self, repo: Path, checkpoint_name: str, device: str) -> None:
        GPT, GPTConfig = import_nanogpt_model(repo)
        ckpt_path = repo / "nanogpt" / "out" / checkpoint_name
        if not ckpt_path.exists():
            raise SystemExit(
                f"Checkpoint not found: {ckpt_path}\n"
                "Download one from https://huggingface.co/adamkarvonen/chess_llms "
                "and place it in chess_gpt_eval/nanogpt/out/."
            )

        self.repo = repo
        self.device = torch.device(device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = GPTConfig(**checkpoint["model_args"])
        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            if key.startswith("_orig_mod."):
                state_dict[key[len("_orig_mod.") :]] = state_dict.pop(key)

        self.model = GPT(config)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self.encode, self.decode = load_tokenizer(repo, checkpoint)

    def compact_prompt(self, board: "chess.Board") -> str:
        transcript = board_to_compact_pgn(board)
        return ";" + transcript

    def activation_before_move(self, prompt: str) -> list["torch.Tensor"]:
        ids = self.encode(prompt)
        if len(ids) > self.model.config.block_size:
            ids = ids[-self.model.config.block_size :]
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None, :]

        captured: list[torch.Tensor] = []
        handles = []
        for block in self.model.transformer.h:
            handles.append(block.register_forward_hook(lambda _m, _inp, out: captured.append(out[:, -1, :].detach().cpu().squeeze(0))))
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def generate_move_text(self, prompt: str, temperature: float, top_k: int, max_new_tokens: int) -> str:
        ids = self.encode(prompt)
        if len(ids) > self.model.config.block_size:
            ids = ids[-self.model.config.block_size :]
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None, :]
        with torch.no_grad():
            y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        generated = self.decode(y[0].tolist()[len(ids) :])
        generated = generated.split(";")[0]
        return first_move(generated)


def board_to_compact_pgn(board: "chess.Board") -> str:
    replay = chess.Board()
    parts: list[str] = []
    for move in board.move_stack:
        if replay.turn == chess.WHITE:
            parts.append(f"{replay.fullmove_number}.{replay.san(move)}")
        else:
            parts.append(replay.san(move))
        replay.push(move)
    return " ".join(parts)


def first_move(text: str) -> str:
    text = re.sub(r"^\s*\d+\.\s*", "", text.strip())
    return text.split()[0] if text.split() else ""


def random_board(rng: random.Random, plies: int) -> "chess.Board":
    board = chess.Board()
    for _ in range(rng.randint(0, plies)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
    return board


def is_legal_san(board: "chess.Board", move_text: str) -> int:
    if move_text in {"", "1-0", "0-1", "1/2-1/2"}:
        return 0
    try:
        move = board.parse_san(move_text)
    except ValueError:
        return 0
    return int(move in board.legal_moves)


def collect_examples(
    chess_gpt: ChessGPT,
    n_positions: int,
    seed: int,
    max_plies: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n_positions):
        board = random_board(rng, max_plies)
        prompt = chess_gpt.compact_prompt(board)
        activations = chess_gpt.activation_before_move(prompt)
        move_text = chess_gpt.generate_move_text(prompt, temperature, top_k, max_new_tokens)
        examples.append(Example(activations, is_legal_san(board, move_text), move_text, prompt))
    return examples


def train_probe(train_x, train_y, test_x, test_y, epochs: int, lr: float):
    dim = train_x.shape[1]
    probe = torch.nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        logits = probe(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = (torch.sigmoid(probe(train_x).squeeze(-1)) >= 0.5).float()
        test_logits = probe(test_x).squeeze(-1)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
        test_loss = loss_fn(test_logits, test_y).item()
    return (train_pred == train_y).float().mean().item(), (test_pred == test_y).float().mean().item(), test_loss


def run(args) -> None:
    load_runtime_dependencies()
    if args.positions < 1:
        raise SystemExit("--positions must be at least 1.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check_setup:
        check_setup(Path(args.repo).expanduser().resolve(), args.checkpoint, args.device)
        return
    chess_gpt = ChessGPT(Path(args.repo).expanduser().resolve(), args.checkpoint, args.device)
    examples = collect_examples(
        chess_gpt,
        args.positions,
        args.seed,
        args.max_plies,
        args.temperature,
        args.top_k,
        args.max_new_tokens,
    )
    positives = sum(example.is_legal for example in examples)
    print(f"Collected {len(examples)} examples: {positives} legal, {len(examples) - positives} illegal.")
    if positives == 0 or positives == len(examples):
        print("Warning: labels have one class only. Increase --temperature or --positions for a useful probe.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = max(1, int(0.8 * len(examples)))
    train, test = examples[:split], examples[split:] or examples[:1]

    print("layer  dim   train_acc  test_acc  test_loss")
    print("-" * 47)
    n_layers = len(examples[0].layer_activations)
    for layer in range(n_layers):
        train_x = torch.stack([example.layer_activations[layer] for example in train]).float()
        test_x = torch.stack([example.layer_activations[layer] for example in test]).float()
        train_y = torch.tensor([example.is_legal for example in train], dtype=torch.float32)
        test_y = torch.tensor([example.is_legal for example in test], dtype=torch.float32)
        train_acc, test_acc, test_loss = train_probe(train_x, train_y, test_x, test_y, args.epochs, args.lr)
        print(f"{layer:>5}  {train_x.shape[1]:>4}  {train_acc:.3f}      {test_acc:.3f}     {test_loss:.3f}")


def check_setup(repo: Path, checkpoint: str, device: str) -> None:
    model_path = repo / "nanogpt" / "model.py"
    ckpt_path = repo / "nanogpt" / "out" / checkpoint
    print("Chess-GPT setup check")
    print(f"torch: {torch.__version__}")
    print(f"chess: {chess.__version__}")
    print(f"device: {device}")
    print(f"repo: {repo}")
    print(f"model.py: {'ok' if model_path.exists() else 'missing'}")
    print(f"checkpoint: {'ok' if ckpt_path.exists() else 'missing'} ({ckpt_path})")
    if not model_path.exists() or not ckpt_path.exists():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Chess-GPT residual stream for move legality.")
    parser.add_argument("--repo", default=os.environ.get("CHESS_GPT_EVAL_REPO", "../chess_gpt_eval"))
    parser.add_argument("--checkpoint", default="stockfish_16layers_ckpt_no_optimizer.pt")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-setup", action="store_true", help="Verify dependencies, repo path, and checkpoint path without running probes.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
