"""Download the Chess-GPT model checkpoint from HuggingFace.

The checkpoint lives at:
    https://huggingface.co/adamkarvonen/chess_llms

and should be placed at:
    <chess_gpt_eval_repo>/nanogpt/out/stockfish_16layers_ckpt_no_optimizer.pt

Usage:
    # Default: place next to this repo (../chess_gpt_eval)
    uv run python download_model.py

    # Explicit destination directory
    uv run python download_model.py --out-dir ../chess_gpt_eval/nanogpt/out

    # If you have huggingface_hub installed, this script will use it
    # for resumable downloads. Otherwise it falls back to urllib.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ID = "adamkarvonen/chess_llms"
FILENAME = "stockfish_16layers_ckpt_no_optimizer.pt"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "chess_gpt_eval" / "nanogpt" / "out"
HF_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"


def download_with_hf_hub(out_path: Path) -> None:
    from huggingface_hub import hf_hub_download
    import shutil

    print(f"Downloading via huggingface_hub …")
    tmp = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp, out_path)


def download_with_urllib(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {HF_URL}")
    print(f"Destination: {out_path}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            print(f"\r  {pct:5.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(HF_URL, out_path, reporthook=report)
    print()  # newline after progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to place the checkpoint in. "
             f"Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--filename",
        default=FILENAME,
        help=f"Checkpoint filename. Default: {FILENAME}",
    )
    args = parser.parse_args()

    out_path = args.out_dir / args.filename

    if out_path.exists():
        print(f"Checkpoint already exists at {out_path}")
        print("Delete it first if you want to re-download.")
        sys.exit(0)

    try:
        import huggingface_hub  # noqa: F401
        use_hf = True
    except ImportError:
        use_hf = False

    if use_hf:
        download_with_hf_hub(out_path)
    else:
        download_with_urllib(out_path)

    print(f"\nCheckpoint saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
