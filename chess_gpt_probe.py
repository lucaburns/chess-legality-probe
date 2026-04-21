"""Layer-wise legality probe for Adam Karvonen's Chess-GPT checkpoints.

This script expects a local checkout of:
https://github.com/adamkarvonen/chess_gpt_eval

It loads a NanoGPT chess checkpoint from that repo, generates moves from random
reachable positions, captures residual-stream activations immediately before
move generation, and trains one linear probe per transformer layer to predict
whether the generated move is legal.

Changes from the first draft:
- Hook now captures the residual stream at each layer boundary, including the
  post-embedding pre-block-0 state, so "layer 0" really means embeddings.
- Robust to NanoGPT forks where Block.forward returns a tuple.
- Tokenizer fallback is explicit: if meta.pkl is missing for a char-level
  checkpoint, we warn and refuse to silently use tiktoken.
- Probes are trained with k-fold CV; we report mean ± std test accuracy and
  AUROC alongside a majority-class baseline.
- Move parsing handles "1..." / "1...e4" / leading-space variants.
- Prompt format is validated against a sample before running the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib.util
import math
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
    layer_activations: list["torch.Tensor"]  # one per layer boundary (including embeddings)
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


def load_tokenizer(
    repo: Path,
    checkpoint: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str], str]:
    """Return (encode, decode, kind) where kind is 'char' or 'tiktoken'.

    Chess-GPT's Stockfish/lichess checkpoints are character-level and need
    meta.pkl's stoi/itos tables. We only fall back to tiktoken when the
    checkpoint clearly wasn't trained with a custom dataset.
    """
    meta_path = repo / "nanogpt" / "out" / "meta.pkl"
    has_custom_dataset = (
        isinstance(checkpoint.get("config"), dict)
        and "dataset" in checkpoint["config"]
    )

    if has_custom_dataset:
        if not meta_path.exists():
            raise SystemExit(
                f"Checkpoint was trained on dataset '{checkpoint['config'].get('dataset')}' "
                f"but {meta_path} is missing. Copy the meta.pkl that ships with the checkpoint "
                "into nanogpt/out/ so the character-level tokenizer can be restored."
            )
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        stoi, itos = meta["stoi"], meta["itos"]

        def encode(text: str) -> list[int]:
            missing = {c for c in text if c not in stoi}
            if missing:
                raise SystemExit(
                    f"Prompt contains characters absent from the char-level vocab: {sorted(missing)!r}. "
                    "This usually means the prompt format doesn't match the training format."
                )
            return [stoi[c] for c in text]

        def decode(ids: list[int]) -> str:
            return "".join(itos[i] for i in ids)

        return encode, decode, "char"

    tiktoken = require("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    return (
        lambda text: enc.encode(text, allowed_special={"<|endoftext|>"}),
        lambda ids: enc.decode(ids),
        "tiktoken",
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
        self.encode, self.decode, self.tokenizer_kind = load_tokenizer(repo, checkpoint)
        self.n_layers = len(self.model.transformer.h)

    def compact_prompt(self, board: "chess.Board") -> str:
        """Produce the ';1.e4 e5 2.Nf3 Nc6' style prompt used by Karvonen's
        Stockfish-trained checkpoints."""
        transcript = board_to_compact_pgn(board)
        return ";" + transcript + (" " if transcript else "")

    def _prepare_input(self, prompt: str) -> "torch.Tensor":
        ids = self.encode(prompt)
        if len(ids) > self.model.config.block_size:
            ids = ids[-self.model.config.block_size :]
        return torch.tensor(ids, dtype=torch.long, device=self.device)[None, :]

    @staticmethod
    def _last_token_residual(out) -> "torch.Tensor":
        """NanoGPT's Block.forward typically returns a tensor (x,) but some
        forks return (x, aux). Accept both."""
        tensor = out[0] if isinstance(out, tuple) else out
        return tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0)

    def activations_by_layer(self, prompt: str) -> list["torch.Tensor"]:
        """Return one residual-stream snapshot per layer boundary, starting
        with post-embedding (index 0) and followed by the output of each
        transformer block.

        For a 16-layer model this returns 17 tensors: [embed, block_0_out,
        block_1_out, ..., block_15_out].
        """
        x = self._prepare_input(prompt)
        captured: list[torch.Tensor] = []
        handles = []

        # Embedding layer: we hook whichever submodule sums token + position
        # embeddings. NanoGPT stores this as transformer.wte / wpe; the
        # embedded sequence is the input to transformer.drop (or to block 0
        # if no dropout module exists). We hook the first block's forward
        # *pre-hook* to grab its input, which is the post-embedding state.
        transformer = self.model.transformer
        first_block = transformer.h[0]

        def embed_pre_hook(_module, inputs):
            tensor = inputs[0]
            captured.append(tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0))

        handles.append(first_block.register_forward_pre_hook(embed_pre_hook))

        def make_block_hook():
            def hook(_module, _inputs, out):
                captured.append(self._last_token_residual(out))
            return hook

        for block in transformer.h:
            handles.append(block.register_forward_hook(make_block_hook()))

        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for handle in handles:
                handle.remove()

        expected = self.n_layers + 1  # embedding + one per block
        if len(captured) != expected:
            raise RuntimeError(
                f"Hook count mismatch: expected {expected}, got {len(captured)}. "
                "Check whether the NanoGPT fork's Block.forward signature matches."
            )
        return captured

    def generate_move_text(
        self, prompt: str, temperature: float, top_k: int, max_new_tokens: int
    ) -> str:
        x = self._prepare_input(prompt)
        with torch.no_grad():
            y = self.model.generate(
                x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
            )
        generated = self.decode(y[0].tolist()[x.shape[1] :])
        # The model should emit something like "e4 e5 2.Nf3"; stop at the first
        # token that looks like the start of the next move number or a game
        # terminator.
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


_MOVE_NUMBER_PREFIX = re.compile(r"^\s*\d+\.+\s*")


def first_move(text: str) -> str:
    """Extract the first SAN token from the model's output.

    Handles '1.e4', '1... e4', ' e4', 'e4 e5 2.Nf3', etc.
    """
    text = _MOVE_NUMBER_PREFIX.sub("", text.strip())
    tokens = text.split()
    return tokens[0] if tokens else ""


def random_board(rng: random.Random, plies: int) -> "chess.Board":
    board = chess.Board()
    for _ in range(rng.randint(0, plies)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
    return board


def is_legal_san(board: "chess.Board", move_text: str) -> int:
    if move_text in {"", "1-0", "0-1", "1/2-1/2", "*"}:
        return 0
    try:
        move = board.parse_san(move_text)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
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
    verbose_first_prompts: int = 3,
) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(n_positions):
        board = random_board(rng, max_plies)
        prompt = chess_gpt.compact_prompt(board)
        if i < verbose_first_prompts:
            print(f"[prompt sample] {prompt!r}")
        activations = chess_gpt.activations_by_layer(prompt)
        move_text = chess_gpt.generate_move_text(prompt, temperature, top_k, max_new_tokens)
        examples.append(
            Example(activations, is_legal_san(board, move_text), move_text, prompt)
        )
    return examples


def auroc(scores: "torch.Tensor", labels: "torch.Tensor") -> float:
    """Standard AUROC via Mann-Whitney U, robust to class imbalance."""
    scores = scores.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1).long()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # For each positive, count negatives with strictly lower score + 0.5 * ties.
    # O(n log n) via sorting.
    all_scores = torch.cat([pos, neg])
    order = all_scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    # Handle ties: average rank within equal groups.
    sorted_scores = all_scores[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i : j + 1]]).mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[: pos.numel()].sum().item()
    n_pos = pos.numel()
    n_neg = neg.numel()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def train_probe(
    train_x: "torch.Tensor",
    train_y: "torch.Tensor",
    test_x: "torch.Tensor",
    test_y: "torch.Tensor",
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[float, float, float, float]:
    """Full-batch AdamW on a linear probe. Returns
    (train_acc, test_acc, test_loss, test_auroc)."""
    dim = train_x.shape[1]
    probe = torch.nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        logits = probe(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_x).squeeze(-1)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        test_logits = probe(test_x).squeeze(-1)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
        test_loss = loss_fn(test_logits, test_y).item()
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
    examples: list[Example],
    layer: int,
    folds: list[list[int]],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    n = len(examples)
    x_all = torch.stack([examples[i].layer_activations[layer] for i in range(n)]).float()
    y_all = torch.tensor([examples[i].is_legal for i in range(n)], dtype=torch.float32)

    accs: list[float] = []
    aucs: list[float] = []
    losses: list[float] = []
    train_accs: list[float] = []
    for i, test_idx in enumerate(folds):
        train_idx = [j for f, fold in enumerate(folds) if f != i for j in fold]
        train_x, train_y = x_all[train_idx], y_all[train_idx]
        test_x, test_y = x_all[test_idx], y_all[test_idx]

        # Standardise using train-fold stats only.
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        train_acc, test_acc, test_loss, test_auc = train_probe(
            train_x, train_y, test_x, test_y, epochs=epochs, lr=lr, weight_decay=weight_decay
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
    }


def majority_baseline(labels: list[int]) -> float:
    if not labels:
        return 0.0
    ones = sum(labels)
    return max(ones, len(labels) - ones) / len(labels)


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
    print(f"Loaded model: {chess_gpt.n_layers} layers, tokenizer={chess_gpt.tokenizer_kind}")

    examples = collect_examples(
        chess_gpt,
        args.positions,
        args.seed,
        args.max_plies,
        args.temperature,
        args.top_k,
        args.max_new_tokens,
    )
    labels = [e.is_legal for e in examples]
    positives = sum(labels)
    print(
        f"Collected {len(examples)} examples: {positives} legal, "
        f"{len(examples) - positives} illegal."
    )
    if positives == 0 or positives == len(examples):
        raise SystemExit(
            "Labels have a single class only. Increase --temperature or --positions, "
            "or verify that the prompt format matches the checkpoint's training data."
        )
    baseline = majority_baseline(labels)
    print(f"Majority-class baseline: {baseline:.3f}")

    rng = random.Random(args.seed + 1)
    folds = kfold_indices(len(examples), args.folds, rng)
    # Make sure every fold has both classes -- otherwise AUROC is undefined
    # and accuracy becomes a lottery. If any fold is single-class, re-shuffle.
    for _ in range(10):
        if all(
            0 < sum(labels[j] for j in fold) < len(fold)
            for fold in folds
        ):
            break
        folds = kfold_indices(len(examples), args.folds, rng)

    n_slots = chess_gpt.n_layers + 1  # embedding + blocks
    print()
    print("layer  dim    train_acc  test_acc ± std  test_auc ± std  test_loss")
    print("-" * 68)
    for slot in range(n_slots):
        stats = probe_layer(
            examples, slot, folds, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
        )
        label = "embed" if slot == 0 else f"blk{slot - 1}"
        print(
            f"{label:>5}  {stats['dim']:<5}  "
            f"{stats['train_acc']:.3f}      "
            f"{stats['test_acc']:.3f} ±{stats['test_acc_std']:.3f}  "
            f"{stats['test_auc']:.3f} ±{stats['test_auc_std']:.3f}  "
            f"{stats['test_loss']:.3f}"
        )


def check_setup(repo: Path, checkpoint: str, device: str) -> None:
    model_path = repo / "nanogpt" / "model.py"
    ckpt_path = repo / "nanogpt" / "out" / checkpoint
    meta_path = repo / "nanogpt" / "out" / "meta.pkl"
    print("Chess-GPT setup check")
    print(f"torch: {torch.__version__}")
    print(f"chess: {chess.__version__}")
    print(f"device: {device}")
    print(f"repo: {repo}")
    print(f"model.py: {'ok' if model_path.exists() else 'missing'}")
    print(f"checkpoint: {'ok' if ckpt_path.exists() else 'missing'} ({ckpt_path})")
    print(f"meta.pkl (char-level vocab): {'ok' if meta_path.exists() else 'missing'}")
    if not model_path.exists() or not ckpt_path.exists():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Chess-GPT residual stream for move legality.")
    parser.add_argument("--repo", default=os.environ.get("CHESS_GPT_EVAL_REPO", "../chess_gpt_eval"))
    parser.add_argument("--checkpoint", default="stockfish_16layers_ckpt_no_optimizer.pt")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--positions", type=int, default=512,
                        help="Number of positions to sample. Probe stability needs several hundred.")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-setup", action="store_true",
                        help="Verify dependencies, repo path, and checkpoint path without running probes.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
