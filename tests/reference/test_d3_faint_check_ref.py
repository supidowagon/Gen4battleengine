"""D3-7 on the reference: the format side-out cases through ``ref_step`` (the engine side is
asserted equal inside ``play``), ``_blacked_out`` per format, and the ``(battler, slot)`` payout
(``_grant_exp`` reads the fainted battler's own party)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS
from nuzline.engine.step import WINNER_OPPONENT, WINNER_PLAYER
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _blacked_out, _sides_out, ref_grant_exp
from tests.engine.d2_states import four
from tests.engine.test_d3_faint_check import BLACKOUT, empty, only_alive, result, tackle


@pytest.mark.parametrize("fmt", [DOUBLES, TWO_TRAINERS, MULTI])
def test_ref_blacked_out_per_format(tables, names, fmt):
    s = four(tables, names, fmt)
    for b, parties in BLACKOUT[fmt].items():
        for p in range(fmt.P):
            alive = s._replace(hp=jnp.zeros_like(s.hp).at[p, 0].set(1))
            assert _blacked_out(to_ref(alive), b) == (p not in parties), (b, p)


def test_ref_sides_out_per_format(tables, names):
    two = to_ref(only_alive(empty(four(tables, names, TWO_TRAINERS), 1), 2, 0))
    assert _sides_out(two) == (False, False)  # party 2 still has a Pokemon
    gone = to_ref(empty(empty(four(tables, names, TWO_TRAINERS), 1), 2))
    assert _sides_out(gone) == (False, True)
    multi = to_ref(empty(only_alive(four(tables, names, MULTI), 0, 0), 1))
    assert _sides_out(multi) == (False, False)  # party 3 whole


def test_ref_step_format_cases(tables, names):
    s = only_alive(four(tables, names, DOUBLES), 1, 0)
    assert result(tackle(tables, s, 0, 1)) == (1, WINNER_PLAYER)
    s = only_alive(empty(four(tables, names, TWO_TRAINERS), 2), 1, 0)
    assert result(tackle(tables, s, 0, 1)) == (1, WINNER_PLAYER)
    s = only_alive(four(tables, names, MULTI), 0, 0)
    assert result(tackle(tables, s, 1, 0)) == (1, WINNER_OPPONENT)
    s = only_alive(four(tables, names, MULTI), 1, 0)  # party 3 whole: the battle goes on
    assert result(tackle(tables, s, 0, 1)) == (0, 0)


def test_grant_exp_reads_the_fainted_battlers_party(tables, names):
    """TWO_TRAINERS: battler 3's Pokemon is party 2's; its level, not party 1's, sets the exp."""
    s = four(tables, names, TWO_TRAINERS)
    s = s._replace(level=s.level.at[2, 0].set(80).at[1, 0].set(10))
    s = s._replace(exp_mask=jnp.full_like(s.exp_mask, 1))
    np_tables = to_numpy(tables)
    third = from_ref(ref_grant_exp(np_tables, to_ref(s), 0, battler=3))
    first = from_ref(ref_grant_exp(np_tables, to_ref(s), 0, battler=1))
    assert int(third.exp[0]) > int(first.exp[0])
