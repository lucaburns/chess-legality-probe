"""Path helpers for running this extension from a subfolder.

All scripts in this directory call `setup_paths()` at import time, which:
  1. Adds the parent directory (project root) to sys.path so `chess_probe_common`
     can be imported from there.
  2. Resolves relative data/plots paths against the parent directory by default.

This lets you keep the extension neatly in its own folder while still
sharing your project's `data/` directory, `chess_probe_common.py`, and
your existing per-fold CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def setup_paths() -> None:
    """Insert the project root on sys.path (idempotent)."""
    project_str = str(PROJECT_ROOT)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)


def resolve_path(p: str | Path) -> Path:
    """Resolve a path: if absolute, return as-is; if relative, resolve
    against the *project root*, not the current working directory.

    This way `--dataset data/foo.pt` works the same way whether you run
    the script from the project root or from the subfolder.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()
