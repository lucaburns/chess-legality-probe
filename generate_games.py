"""Generate self-play games with a Chess-GPT checkpoint and save per-position
activations to disk for later probing.

Usage:
    python generate_games.py \\
        --repo ../chess_gpt_eval \\
        --checkpoint stockfish_16layers_ckpt_no_optimizer.pt \\
        --positions 4000 \\
        --temperature 1.3 \\
        --output datasets/stockfish16_t1.3_n4000.pt

The output .pt file can be consumed by chess_gpt_probe.py, which trains
per-layer probes on the cached activations without re-running the model.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Callable

from chess_probe_common import Example, save_examples
from config_utils import flatten_sections, load_yaml_config

torch = None
chess = None


def require(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {module_name}\n"
            "Install the Chess-GPT dependencies first:\n"
            "  python3 -m pip install torch python-chess tiktoken numpy"
        ) from exc


def load_runtime_dependencies() -> None:
    global torch, chess
    if torch is None:
        torch = require("torch")
    if chess is None:
        chess = require("chess")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


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
                state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)

        self.model = GPT(config)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self.encode, self.decode, self.tokenizer_kind = load_tokenizer(repo, checkpoint)
        self.n_layers = len(self.model.transformer.h)

    def compact_prompt(self, board: "chess.Board") -> str:
        transcript = board_to_compact_pgn(board)
        return ";" + transcript + (" " if transcript else "")

    def _prepare_input(self, prompt: str) -> "torch.Tensor":
        ids = self.encode(prompt)
        if len(ids) > self.model.config.block_size:
            ids = ids[-self.model.config.block_size:]
        return torch.tensor(ids, dtype=torch.long, device=self.device)[None, :]

    @staticmethod
    def _last_token_residual(out) -> "torch.Tensor":
        tensor = out[0] if isinstance(out, tuple) else out
        return tensor[:, -1, :].detach().to("cpu", copy=True).squeeze(0)

    def activations_by_layer(self, prompt: str) -> list["torch.Tensor"]:
        """Residual stream at post-embedding + after each block. Returns
        n_layers + 1 tensors for a model with n_layers transformer blocks."""
        x = self._prepare_input(prompt)
        captured: list[torch.Tensor] = []
        handles = []

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

        expected = self.n_layers + 1
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
        generated = self.decode(y[0].tolist()[x.shape[1]:])
        generated = generated.split(";")[0]
        return first_move(generated)


# ---------------------------------------------------------------------------
# Chess helpers and self-play
# ---------------------------------------------------------------------------


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


def random_opening(rng: random.Random, plies: int) -> "chess.Board":
    """Seed the opening with a handful of random legal plies for diversity."""
    board = chess.Board()
    for _ in range(rng.randint(0, plies)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
    return board


def parse_san_or_none(board: "chess.Board", move_text: str):
    """Return a chess.Move if move_text is a legal SAN for board, else None."""
    if move_text in {"", "1-0", "0-1", "1/2-1/2", "*"}:
        return None
    try:
        move = board.parse_san(move_text)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return None
    return move if move in board.legal_moves else None


def play_self_game(
    chess_gpt: "ChessGPT",
    rng: random.Random,
    game_id: int,
    max_plies: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    random_opening_plies: int,
    stop_on_illegal: bool,
) -> list[Example]:
    """Play one self-play game, recording an Example at every ply.

    On legal moves, push the model's move and continue.
    On illegal moves, record the example then either:
      - stop_on_illegal=True: end the game.
      - stop_on_illegal=False: push a random legal move and continue.
    """
    board = random_opening(rng, random_opening_plies)
    examples: list[Example] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        prompt = chess_gpt.compact_prompt(board)
        activations = chess_gpt.activations_by_layer(prompt)
        move_text = chess_gpt.generate_move_text(prompt, temperature, top_k, max_new_tokens)
        legal_move = parse_san_or_none(board, move_text)
        is_legal = 1 if legal_move is not None else 0

        examples.append(
            Example(
                layer_activations=activations,
                is_legal=is_legal,
                move_text=move_text,
                prompt=prompt,
                ply=board.ply(),
                game_id=game_id,
                fen=board.fen(),
            )
        )

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
    chess_gpt: ChessGPT,
    n_positions: int,
    seed: int,
    max_plies: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    random_opening_plies: int,
    stop_on_illegal: bool,
    verbose_first_prompts: int = 3,
    progress_every: int = 100,
) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    game_id = 0
    last_reported = 0
    while len(examples) < n_positions:
        game_examples = play_self_game(
            chess_gpt,
            rng,
            game_id=game_id,
            max_plies=max_plies,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            random_opening_plies=random_opening_plies,
            stop_on_illegal=stop_on_illegal,
        )
        if game_examples and game_id < verbose_first_prompts:
            print(f"[prompt sample game {game_id}] {game_examples[0].prompt!r}")
            print(
                f"  game {game_id}: {len(game_examples)} plies, "
                f"{sum(1 for e in game_examples if e.is_legal == 0)} illegal"
            )
        examples.extend(game_examples)
        game_id += 1
        if len(examples) - last_reported >= progress_every:
            n_illegal = sum(1 for e in examples if e.is_legal == 0)
            print(
                f"  progress: {len(examples)}/{n_positions} examples, "
                f"{game_id} games, {n_illegal} illegal "
                f"({100.0 * n_illegal / max(len(examples), 1):.1f}%)"
            )
            last_reported = len(examples)
    return examples[:n_positions]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(args) -> None:
    load_runtime_dependencies()
    if args.positions < 1:
        raise SystemExit("--positions must be at least 1.")
    args.device = resolve_device(args.device)

    chess_gpt = ChessGPT(Path(args.repo).expanduser().resolve(), args.checkpoint, args.device)
    print(f"Loaded model: {chess_gpt.n_layers} layers, tokenizer={chess_gpt.tokenizer_kind}")
    print(f"Device: {args.device}")

    examples = collect_examples(
        chess_gpt,
        args.positions,
        args.seed,
        args.max_plies,
        args.temperature,
        args.top_k,
        args.max_new_tokens,
        random_opening_plies=args.random_opening_plies,
        stop_on_illegal=args.stop_on_illegal,
    )
    n = len(examples)
    n_legal = sum(e.is_legal for e in examples)
    n_illegal = n - n_legal
    n_games = max(e.game_id for e in examples) + 1 if examples else 0
    avg_ply = sum(e.ply for e in examples) / n if n else 0.0
    print()
    print(f"Collected {n} examples from {n_games} games.")
    print(f"  legal: {n_legal}, illegal: {n_illegal} ({100.0 * n_illegal / max(n, 1):.1f}%)")
    print(f"  mean ply index: {avg_ply:.1f}")

    config = {
        "checkpoint": args.checkpoint,
        "repo": str(Path(args.repo).expanduser().resolve()),
        "device": args.device,
        "positions": args.positions,
        "max_plies": args.max_plies,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "random_opening_plies": args.random_opening_plies,
        "stop_on_illegal": args.stop_on_illegal,
        "seed": args.seed,
        "tokenizer_kind": chess_gpt.tokenizer_kind,
        "n_layers": chess_gpt.n_layers,
    }
    output_path = Path(args.output).expanduser().resolve()
    save_examples(examples, output_path, config)
    print(f"Saved dataset to {output_path}")
    print(f"Summary sidecar at {output_path}.json")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to a YAML config file.")
    config_args, remaining = config_parser.parse_known_args()

    defaults = {
        "repo": os.environ.get("CHESS_GPT_EVAL_REPO", "../chess_gpt_eval"),
        "checkpoint": "stockfish_16layers_ckpt_no_optimizer.pt",
        "device": "auto",
        "output": None,
        "positions": 4000,
        "max_plies": 120,
        "temperature": 1.3,
        "top_k": 200,
        "max_new_tokens": 10,
        "random_opening_plies": 2,
        "stop_on_illegal": False,
        "seed": 7,
    }

    if config_args.config:
        config = load_yaml_config(config_args.config)
        yaml_values = flatten_sections(config, "paths", "generation")
        defaults.update(
            {
                "repo": yaml_values.get("chess_gpt_eval_repo", defaults["repo"]),
                "checkpoint": yaml_values.get("checkpoint", defaults["checkpoint"]),
                "device": yaml_values.get("device", defaults["device"]),
                "output": yaml_values.get("output_dataset", defaults["output"]),
                "positions": yaml_values.get("positions", defaults["positions"]),
                "max_plies": yaml_values.get("max_plies", defaults["max_plies"]),
                "temperature": yaml_values.get("temperature", defaults["temperature"]),
                "top_k": yaml_values.get("top_k", defaults["top_k"]),
                "max_new_tokens": yaml_values.get("max_new_tokens", defaults["max_new_tokens"]),
                "random_opening_plies": yaml_values.get(
                    "random_opening_plies", defaults["random_opening_plies"]
                ),
                "stop_on_illegal": yaml_values.get(
                    "stop_on_illegal", defaults["stop_on_illegal"]
                ),
                "seed": yaml_values.get("seed", defaults["seed"]),
            }
        )

    parser = argparse.ArgumentParser(
        description="Generate self-play games and save activations.",
        parents=[config_parser],
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--output",
                        help="Path to save the dataset (.pt). A .pt.json summary is also written.")
    parser.add_argument("--positions", type=int,
                        help="Target number of examples to collect.")
    parser.add_argument("--max-plies", type=int,
                        help="Maximum ply count per self-play game.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--random-opening-plies", type=int,
                        help="Max random legal plies to play before the model takes over.")
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-illegal", dest="stop_on_illegal", action="store_true",
                            help="End games on illegal move instead of patching with a random legal move.")
    stop_group.add_argument("--no-stop-on-illegal", dest="stop_on_illegal", action="store_false",
                            help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(remaining)
    if args.output is None:
        parser.error("--output is required unless provided via --config.")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
