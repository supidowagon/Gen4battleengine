"""Plain-ASCII text for the command line (controller ruling: a default Windows console is
cp1252, and Kaizo names carry characters it cannot print)."""

from __future__ import annotations

__all__ = ["ascii_text"]


def ascii_text(text: str) -> str:
    """``text`` with every non-ASCII character escaped (``backslashreplace``)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")
