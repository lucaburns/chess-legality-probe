# Chess Legality Probe

We ask whether a chess-playing language model internally represents a
legality signal before it emits an illegal move. This version keeps the experimental
shape in two stages:

1. `chess_probe.py`: dependency-free scaffold with proxy chess features.
2. `chess_gpt_probe.py`: Chess-GPT version for Adam Karvonen's `chess_gpt_eval`
   NanoGPT checkpoints.

## Chess-GPT Version

We do the following:

1. Load a local `chess_gpt_eval` checkout.
2. Load a NanoGPT chess checkpoint from `chess_gpt_eval/nanogpt/out/`.
3. Generate random reachable chess positions.
4. Format prompts the same compact PGN way as `nanogpt/nanogpt_module.py`.
5. Capture each transformer block's residual stream at the final prompt token,
   immediately before move generation.
6. Generate Chess-GPT's next move and label it legal or illegal with
   `python-chess`.
7. Train one linear probe per layer and report layer-wise accuracy.

Setup:

```bash
git clone https://github.com/adamkarvonen/chess_gpt_eval.git ../chess_gpt_eval
uv sync
```

Download a checkpoint from `https://huggingface.co/adamkarvonen/chess_llms`
into the location expected by `chess_gpt_eval`:

```bash
mkdir -p ../chess_gpt_eval/nanogpt/out
curl -L \
  -o ../chess_gpt_eval/nanogpt/out/stockfish_16layers_ckpt_no_optimizer.pt \
  https://huggingface.co/adamkarvonen/chess_llms/resolve/main/stockfish_16layers_ckpt_no_optimizer.pt
```

The original repo README recommends `stockfish_16layers_ckpt_no_optimizer.pt` as
the strongest model. To use a different checkpoint from that page, replace the
filename in both the `-o` path and the Hugging Face URL.

Run:

```bash
uv run python chess_gpt_probe.py \
  --repo ../chess_gpt_eval \
  --checkpoint stockfish_16layers_ckpt_no_optimizer.pt \
  --positions 64 \
  --device cpu
```

Check the local setup without running the probe:

```bash
uv run python chess_gpt_probe.py \
  --check-setup \
  --repo ../chess_gpt_eval \
  --checkpoint stockfish_16layers_ckpt_no_optimizer.pt
```

Use a higher temperature or more positions if the generated labels are too
imbalanced:

```bash
uv run python chess_gpt_probe.py --temperature 1.2 --positions 256
```

## Dependency-Free Scaffold

`chess_probe.py` remains useful when you do not have Torch or the checkpoint
installed. It uses handcrafted board features as layer-like proxies:

The feature blocks are:

- `material`: only a crude material summary.
- `board_state`: one-hot board representation.
- `candidate_move`: board state plus the proposed move.
- `tactical_legality`: pseudo-legal and king-safety signals.

This is not yet mechanistic interpretability on Chess-GPT activations. It is the
minimum runnable scaffold for the project: data generation, legal/illegal labels,
linear probe training, and layer-wise reporting.

Run the scaffold:

```bash
uv run python chess_probe.py
```

For a faster run:

```bash
uv run python chess_probe.py --positions 120 --epochs 4
```

Run the smoke tests:

```bash
uv run python test.py
```

## Next Steps

- Stratify illegal examples by failure type: empty source square, own-piece
  capture, piece movement violation, and moving into check.
- Compare probe generalization across positions, pieces, and illegal-move types.
- Add intervention tests by shifting activations along the learned legality
  direction before decoding.
