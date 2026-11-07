"""Plain-ASCII text for the command line (controller ruling: a default Windows console is
cp1252, and Kaizo names carry characters it cannot print)."""

from __future__ import annotations

__all__ = ["ascii_text", "class_text"]

# The font's two-cell PKMN glyph, which the charmap decodes as U+20A7 U+20A6 (the ROM's
# "PKMN Trainer" class name, Lucas and Dawn in vanilla, the rivals in Kaizo).
PKMN_GLYPHS = "\u20a7\u20a6"


def ascii_text(text: str) -> str:
    """``text`` with every non-ASCII character escaped (``backslashreplace``)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def class_text(trainer_class: str) -> str:
    """A trainer class for display: the PKMN glyph pair as ``PKMN``, then ``ascii_text``."""
    return ascii_text(trainer_class.replace(PKMN_GLYPHS, "PKMN"))
