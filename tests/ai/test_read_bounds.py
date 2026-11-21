"""E1 Ruling R6: the per-move read bounds (``nuzline.ai.read_bounds``) are the decomp's.

``tools/ai_read_bounds.py`` walks script.s from each transcribed flag's entry; the committed
table must be what it generates now, for every transcribed effect.
"""

from pathlib import Path

import numpy as np
import pytest

from nuzline.ai.layout import KAIZO, fight_slot_reads, move_read_rows
from nuzline.ai.outcome import KAIZO_TRANSCRIBED
from nuzline.ai.read_bounds import DRAWN_READ_BOUNDS, READ_BOUND_BITS, READ_BOUNDS

DECOMP = Path(__file__).resolve().parents[2] / "external/pokeplatinum"


@pytest.mark.fullset
def test_read_bounds_are_the_decomps(kaizo_tables):
    """The committed table is the generator's output, row for row, and covers every
    transcribed effect in all four kinds (damaging or not, quadruple or not)."""
    import sys

    from tools.ai_read_bounds import Script, drawn_read_bounds, read_bounds

    if not (DECOMP / "src/battle/trainer_ai/script.s").exists():
        pytest.skip("decomp not present")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    script = Script(DECOMP, np.asarray(kaizo_tables.move_ai_effect))
    assert read_bounds(script, KAIZO_TRANSCRIBED) == READ_BOUNDS
    assert drawn_read_bounds(script, KAIZO_TRANSCRIBED, READ_BOUNDS) == DRAWN_READ_BOUNDS
    assert {e for e, _, _ in READ_BOUNDS} == set(KAIZO_TRANSCRIBED)


def test_damage_commands_read_by_the_moveset():
    """E1 ruling G-39, hand-checked rows. A damaging hit (effect 0) in a fight with a drawing
    move reads EvalAttack's FlagMoveDamageScore (:6361-6362, compared with NOT_HIGHEST) and,
    when it draws itself, IfCurrentMoveKills (:6358): EVAL_ATTACK 0 -> 1 -> 2; Basic_Main's
    two compares with AI_NO_COMPARISON_MADE (:63, :113) read nothing (BASIC stays 2). U-turn
    (SWITCH_HIT, 228) reads IfPartyMemberDealsMoreDamage (:5118) once more in EXPERT. Copycat
    (242) reads IfBattlerDealsMoreDamage at :5560 and :5573 on one path even in a fight with no
    drawing move (the player's last move may draw): EXPERT 3."""
    bits = READ_BOUND_BITS
    base = dict(zip(bits, READ_BOUNDS[(0, True, False)], strict=True))
    fight = dict(zip(bits, DRAWN_READ_BOUNDS[(0, True, False, False)], strict=True))
    own = dict(zip(bits, DRAWN_READ_BOUNDS[(0, True, False, True)], strict=True))
    assert (base[1], fight[1], own[1]) == (0, 1, 2)
    assert base[0] == fight[0] == own[0] == 2
    u_turn = dict(zip(bits, READ_BOUNDS[(228, True, False)], strict=True))
    u_turn_drawn = dict(zip(bits, DRAWN_READ_BOUNDS[(228, True, False, False)], strict=True))
    assert u_turn_drawn[2] == u_turn[2] + 1
    assert dict(zip(bits, READ_BOUNDS[(242, False, False)], strict=True))[2] == 3


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
    # COMPARE_SPEED_FASTER's tie is a read (E1 Ruling R8): Expert_Tailwind (:4982-5006) makes
    # :4992, the tie at :4993 and :4996 (3); Expert_Payback (:5169-5187) the tie at :5177 and
    # :5179 (2).
    assert dict(zip(bits, READ_BOUNDS[(225, False, False)], strict=True))[2] == 3
    assert dict(zip(bits, READ_BOUNDS[(230, True, False)], strict=True))[2] == 2


def test_tag_strategy_column_is_walked_at_b2():
    """E4-A0: bit 7 (TagStrategy_Main, script.s:6637-7338) joins ``READ_BOUND_BITS``. Hand-checked
    rows: a damaging plain hit makes the R@6694 and the effectiveness roll (2), a Rain Dance none
    (0), Follow Me one (R@7053 or its band's roll), Skill Swap one (the defender's ability coin),
    Gravity one (R@6959; the cleared defender partner's Levitate / Flying / Magnet Rise tests,
    :6964-6968, never jump, so R@6972 never counts), a damaging hit that draws itself one more
    (3: IfCurrentMoveKills :6660 or the CheckIfHighestDamageWithPartner compare :6685, with
    R@6694; a HALF path never takes DOUBLE's R@6722, trainer_ai.c:1311-1346). The partner
    arguments' HP percent is 0 at B = 2, so the TryScoreMinus rolls (:6666, :6678), Trick
    Room's and Future Sight's speed ranks never count."""
    assert 7 in READ_BOUND_BITS
    col = READ_BOUND_BITS.index(7)
    assert READ_BOUNDS[(0, True, False)][col] == 2
    assert DRAWN_READ_BOUNDS[(0, True, False, True)][col] == 3
    assert READ_BOUNDS[(136, False, False)][col] == 0
    assert READ_BOUNDS[(172, False, False)][col] == 1
    assert READ_BOUNDS[(191, False, False)][col] == 1
    assert READ_BOUNDS[(215, False, False)][col] == 1
    assert READ_BOUNDS[(259, False, False)][col] == 0  # Trick Room: -30 at once
    assert READ_BOUNDS[(148, False, False)][col] == 0  # Future Sight: End


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
