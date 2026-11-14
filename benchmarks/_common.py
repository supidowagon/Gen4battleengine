"""Helpers the benchmark scripts share: plain-ASCII output and the git state a result
record names."""

from __future__ import annotations

import subprocess
from pathlib import Path

from nuzline.text import ascii_text

ROOT = Path(__file__).resolve().parent.parent


def say(text: str) -> None:
    """``text`` printed as plain ASCII, flushed."""
    print(ascii_text(text), flush=True)


def commit() -> str:
    """``HEAD``'s hash; ``"unknown"`` when git fails."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def dirty() -> bool | None:
    """Whether tracked files differ from the commit (``None``: git failed)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())
