"""E1 Ruling R6: the per-move read bounds (``nuzline.ai.read_bounds``) are the decomp's.

``tools/ai_read_bounds.py`` walks script.s from each transcribed flag's entry; the committed
table must be what it generates now, for every transcribed effect.
"""

from pathlib import Path

import numpy as np
import pytest

from nuzline.ai.layout import KAIZO, fight_slot_reads, move_read_rows
from nuzline.ai.outcome import KAIZO_TRANSCRIBED
from nuzline.ai.read_bounds import READ_BOUND_BITS, READ_BOUNDS

DECOMP = Path(__file__).resolve().parents[2] / "external/pokeplatinum"


def test_read_bounds_are_the_decomps(kaizo_tables):
    """The committed table is the generator's output, row for row, and covers every
    transcribed effect in all four kinds (damaging or not, quadruple or not)."""
    import sys

    from tools.ai_read_bounds import Script, read_bounds

    if not (DECOMP / "src/battle/trainer_ai/script.s").exists():
        pytest.skip("decomp not present")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    script = Script(DECOMP, np.asarray(kaizo_tables.move_ai_effect))
    assert read_bounds(script, KAIZO_TRANSCRIBED) == READ_BOUNDS
    assert {e for e, _, _ in READ_BOUNDS} == set(KAIZO_TRANSCRIBED)


def test_read_bounds_decide_by_the_move(kaizo_tables):
    """Hand-checked rows (script.s): Selfdestruct (HALVE_DEFENSE, Normal, never quadruple)
    makes :117/:304 guesses and :1859, :1870, :1875 (BASIC 2, EXPERT 3) and EvalAttack's
    :6374 (1), not the quad roll :6383 a quadruple-capable type adds; Risky :6530 (1);
    SETUP_FIRST_TURN 0 (not in its table, :6435-6497); a damaging plain hit has no Expert
    entry (0)."""
    bits = READ_BOUND_BITS
    halve = dict(zip(bits, READ_BOUNDS[(7, False, False)], strict=True))
    assert (halve[0], halve[1], halve[2], halve[4], halve[3]) == (2, 1, 3, 1, 0)
    assert READ_BOUNDS[(7, False, True)][1] == 2  # the quad roll :6383 on top
    hit = dict(zip(bits, READ_BOUNDS[(0, True, False)], strict=True))
    assert hit[2] == 0 and hit[1] == 0


def test_fight_slot_reads(kaizo_tables, kaizo_roark_party):
    """A fight's bound is the most, over its opponents' moves, of the reads its flag set's
    scripts make (E1 Ruling R6): Roark's party under 0x17 is KAIZO's 7; a party without
    transcribed moves still gets 1; the bound follows the flags."""
    t = kaizo_tables
    moves = [m for spec in kaizo_roark_party for m in spec.moves]
    rows = move_read_rows(t, moves)
    assert fight_slot_reads(rows, KAIZO.flags) == KAIZO.slot_reads == 7
    assert fight_slot_reads(rows, 0x7) == 6  # without RISKY's :6530
    assert fight_slot_reads(move_read_rows(t, []), 0x17) == 1
