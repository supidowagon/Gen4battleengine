"""Plain-ASCII text for the command line (controller ruling: a default Windows console is
cp1252, and Kaizo names carry characters it cannot print)."""

from __future__ import annotations

import math

__all__ = ["ascii_text", "class_text", "decimals", "tenths"]

# The font's two-cell PKMN glyph, which the charmap decodes as U+20A7 U+20A6 (the ROM's
# "PKMN Trainer" class name, Lucas and Dawn in vanilla, the rivals in Kaizo).
PKMN_GLYPHS = "\u20a7\u20a6"


def ascii_text(text: str) -> str:
    """``text`` with every non-ASCII character escaped (``backslashreplace``)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def class_text(trainer_class: str) -> str:
    """A trainer class for display: the PKMN glyph pair as ``PKMN``, then ``ascii_text``."""
    return ascii_text(trainer_class.replace(PKMN_GLYPHS, "PKMN"))


def decimals(x: float, places: int, *, up: bool = False) -> str:
    """``x`` to ``places`` decimals, rounded down or, with ``up``, rounded up: a chance or count
    the player must not underrate (a loss chance, expected deaths) never shows lower than it
    is. The scaled value is rounded to 9 places first, so a float such as 0.999 * 1000 =
    998.9999999999999 is taken as 999."""
    scaled = round(x * 10**places, 9)
    return f"{(math.ceil(scaled) if up else math.floor(scaled)) / 10**places:.{places}f}"


def tenths(p: float, *, up: bool = False) -> str:
    """``p`` as a percentage to one decimal, rounded down (a win chance never shows higher than
    it is: 0.9996 is "99.9", not "100.0") or, with ``up``, rounded up (a loss chance never shows
    lower)."""
    return decimals(100 * p, 1, up=up)
