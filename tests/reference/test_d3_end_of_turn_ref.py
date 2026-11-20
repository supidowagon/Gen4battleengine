"""D3-11: the reference's end of the turn over four battlers (the first cases of
tests/engine/test_d3_end_of_turn.py, with expected values asserted on the reference itself)."""

from __future__ import annotations

import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _speed_order
from tests.engine.test_d3_action_loop import hp
from tests.engine.test_d3_end_of_turn import (
    PAIR_2_3,
    _fmt,
    base,
    drain_and_poison,
    ref_end_turn,
    seeded_pair,
    tie_bits,
    with_sort_tie,
)
from tests.engine.test_d3_turn_order import speeds


def _order(tables, state, outcome):
    return _speed_order(to_numpy(tables), to_ref(state), outcome_to_py(outcome))


def test_order_reads_row_b_and_the_post_action_speed(tables, names):
    s = speeds(base(tables, names), (300, 250, 100, 100))
    o = empty_outcome(_fmt(s))
    assert _order(tables, s, o) == [0, 1, 2, 3]
    assert _order(tables, s, with_sort_tie(o, tie_bits(PAIR_2_3))) == [0, 1, 3, 2]


def test_poison_before_the_drain_faints_the_seeder(tables, names):
    s = seeded_pair(tables, names, (100, 200, 300, 150))
    out = ref_end_turn(tables, s, empty_outcome(_fmt(s)))
    assert hp(out, 2) == 0 and hp(out, 3) == hp(s, 3)


def test_the_drain_before_the_poison_keeps_the_seeder(tables, names):
    s = seeded_pair(tables, names, (100, 300, 200, 250))
    drain, poison = drain_and_poison(s)
    out = ref_end_turn(tables, s, empty_outcome(_fmt(s)))
    assert hp(out, 2) == 5 + drain - poison and hp(out, 3) == hp(s, 3) - drain


def test_absent_battler_is_skipped(tables, names):
    from tests.engine.test_d3_end_of_turn import make_absent

    s = make_absent(base(tables, names), 2)
    s = s._replace(future_timer=s.future_timer.at[2].set(2))
    out = ref_end_turn(tables, s, empty_outcome(_fmt(s)))
    assert int(np.asarray(out.future_timer)[2]) == 2
