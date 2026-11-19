"""D2-2: the legal targets per range (``GetTargetSelectLayout``, battle_subscreen.c:1336-1364,
``sMoveTargetSlotFlags`` :797-810), JAX and reference."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import chosen_targets, legal_targets
from nuzline.engine.ids import MoveTarget
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _legal_targets
from tests.engine.d2_states import as_ghost, four, mon, with_hp, with_moves
from tests.engine.test_d2_action_encoding import B2_CASES

ALL = {0, 1, 2, 3}
# move, range, battler 0's set, battler 2's set (the table of the sheet, D2-2 rule 5)
ROWS = (
    ("Tackle", MoveTarget.RANGE_SINGLE_TARGET, {1, 2, 3}, {0, 1, 3}),
    ("Counter", MoveTarget.RANGE_SINGLE_TARGET_SPECIAL, {0}, {2}),
    ("Outrage", MoveTarget.RANGE_RANDOM_OPPONENT, {0}, {2}),
    ("Growl", MoveTarget.RANGE_ADJACENT_OPPONENTS, {1, 3}, {1, 3}),
    ("Earthquake", MoveTarget.RANGE_ALL_ADJACENT, {1, 2, 3}, {0, 1, 3}),
    ("Swords Dance", MoveTarget.RANGE_USER, {0}, {2}),
    ("Reflect", MoveTarget.RANGE_USER_SIDE, {0, 2}, {0, 2}),
    ("Sunny Day", MoveTarget.RANGE_FIELD, ALL, ALL),
    ("Spikes", MoveTarget.RANGE_OPPONENT_SIDE, {1, 3}, {1, 3}),
    ("Helping Hand", MoveTarget.RANGE_ALLY, {2}, {0}),
    ("Acupressure", MoveTarget.RANGE_USER_OR_ALLY, {0, 2}, {0, 2}),
    ("Me First", MoveTarget.RANGE_SINGLE_TARGET_ME_FIRST, {1, 3}, {1, 3}),
)


def _set(mask) -> set[int]:
    return {i for i, v in enumerate(mask.tolist()) if v}


def _both(tables, s, b, slot=0) -> set[int]:
    got = legal_targets(tables, s, b, slot)
    ref = _legal_targets(to_numpy(tables), to_ref(s), b, slot)
    assert ref == got.tolist()
    return _set(got)


def _with(tables, names, move, b):
    return with_moves(four(tables, names), names, b, (move,))


def test_every_range_row_for_battlers_0_and_2(tables, names):
    for move, rng, want0, want2 in ROWS:
        assert int(tables.move_target[names.move(move)]) == rng, move
        assert _both(tables, _with(tables, names, move, 0), 0) == want0, move
        assert _both(tables, _with(tables, names, move, 2), 2) == want2, move


def test_curse_range_follows_the_user_type(tables, names):
    s = with_moves(four(tables, names), names, 0, ("Curse",))
    assert _both(tables, s, 0) == {0}  # non-Ghost: RANGE_USER (bcp.c:4344-4349)
    assert _both(tables, as_ghost(s, 0), 0) == {1, 2, 3}  # the data range, SINGLE_TARGET


def test_absent_battlers_are_off(tables, names):
    for move, _rng, want0, _ in ROWS:
        s = with_hp(_with(tables, names, move, 0), 3, 0)
        assert _both(tables, s, 0) == want0 - {3}, move
    s = with_hp(_with(tables, names, "Helping Hand", 0), 2, 0)
    assert _both(tables, s, 0) == set()  # RANGE_ALLY, partner absent (bcp.c:4352-4357)
    s = with_hp(_with(tables, names, "Acupressure", 0), 2, 0)
    assert _both(tables, s, 0) == {0}


def test_battlers_1_and_3_get_the_same_relations(tables, names):
    for move, _rng, _, _ in ROWS:
        for b in (1, 3):
            s = _with(tables, names, move, b)
            got = _both(tables, s, b)
            if move == "Tackle":
                assert got == ALL - {b}
            elif move == "Helping Hand":
                assert got == {b ^ 2}
            elif move in ("Counter", "Swords Dance", "Outrage"):
                assert got == {b}
            elif move in ("Growl", "Spikes", "Me First"):
                assert got == {b ^ 1, b ^ 3}
            elif move == "Earthquake":
                assert got == ALL - {b}
            elif move in ("Reflect", "Acupressure"):
                assert got == {b, b ^ 2}
            else:
                assert got == ALL


@pytest.mark.parametrize("case", B2_CASES, ids=[c[0] for c in B2_CASES])
def test_b2_one_hot_equals_the_written_value(tables, names, case):
    move, _rng, want = case
    s = make_state(tables, [mon(names, (move,))], [mon(names, ("Tackle",))])
    got = legal_targets(tables, s, 0, 0)
    assert _set(got) == {want}
    assert _both(tables, s, 0) == {want}
    written = chosen_targets(tables, s, jnp.asarray([0, 0], jnp.int32), jnp.zeros(2))
    assert int(written[0]) == want
