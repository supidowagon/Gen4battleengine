"""D2-3: the player's joint decision at B = 4 (``legal_choices``, ``decision_pairs``; party.c:
1618-1674, bcp.c:570-597), JAX and reference."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import (
    NUM_ACTIONS,
    decision_pairs,
    legal_actions,
    legal_choices,
    switch_targets,
)
from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _decision_pairs, _legal_choices, _switch_targets
from tests.engine.d2_states import four, mon, with_moves


def _pairs_match_reference(tables, s):
    got = np.asarray(decision_pairs(tables, s))
    ref = np.asarray(_decision_pairs(to_numpy(tables), to_ref(s)))
    np.testing.assert_array_equal(got, ref)
    return got


def _choices_match_reference(tables, s, b):
    got = np.asarray(legal_choices(tables, s, b))
    ref = np.asarray(_legal_choices(to_numpy(tables), to_ref(s), b))
    np.testing.assert_array_equal(got, ref)
    ref_sw = _switch_targets(to_numpy(tables), to_ref(s), b)
    assert ref_sw == np.asarray(switch_targets(tables, s, b)).tolist()
    return got


def _targets(choices, a) -> set[int]:
    return {t for t in range(choices.shape[1]) if choices[a, t]}


@pytest.mark.parametrize("fmt", (DOUBLES, TWO_TRAINERS), ids=["doubles", "two_trainers"])
def test_two_switches_must_name_different_slots(tables, names, fmt):
    s = four(tables, names, fmt)
    assert s.active.tolist()[0] == 0 and s.active.tolist()[2] == 1
    pairs = _pairs_match_reference(tables, s)
    for k in range(2, 6):
        assert not pairs[4 + k, :, 4 + k, :].any()
    assert pairs[6, 0, 7, 2] and pairs[7, 0, 6, 2]
    # battler 0 may not take battler 2's active slot 1, battler 2 not battler 0's slot 0
    c0 = _choices_match_reference(tables, s, 0)
    c2 = _choices_match_reference(tables, s, 2)
    assert not c0[4 + 0].any() and not c0[4 + 1].any()
    assert not c2[4 + 0].any() and not c2[4 + 1].any()
    assert not pairs[4, :, :, :].any() and not pairs[5, :, :, :].any()
    assert not pairs[:, :, 4, :].any() and not pairs[:, :, 5, :].any()


def test_trapped_battler_cannot_switch(tables, names):
    s = four(tables, names)
    s = s._replace(mean_look=s.mean_look.at[0].set(2))
    c0 = _choices_match_reference(tables, s, 0)
    c2 = _choices_match_reference(tables, s, 2)
    assert not c0[4:].any()
    assert c2[4 + 2].any()
    pairs = _pairs_match_reference(tables, s)
    assert not pairs[4:, :, :, :].any() and pairs[0, 1, 6, 2]


def test_locked_battler_has_only_its_locked_move(tables, names):
    s = with_moves(four(tables, names), names, 0, ("Tackle", "Rollout"))
    s = s._replace(locked_move=s.locked_move.at[0].set(names.move("Rollout")))
    c0 = _choices_match_reference(tables, s, 0)
    assert c0.sum() == 1 and c0[1, 0]
    c2 = _choices_match_reference(tables, s, 2)
    assert c2[0].any() and c2[4 + 2].any()
    _pairs_match_reference(tables, s)


def test_absent_partner(tables, names):
    s = four(tables, names)
    # battler 2's Pokemon (slot 1) has fainted and the bench is gone
    hp = s.hp.at[0, 1:].set(0)
    s = s._replace(hp=hp)
    c2 = _choices_match_reference(tables, s, 2)
    assert c2.sum() == 1 and c2[0, 2]
    s = with_moves(s, names, 0, ("Helping Hand",))
    c0 = _choices_match_reference(tables, s, 0)
    assert _targets(c0, 0) == {0}  # no target is written
    _pairs_match_reference(tables, s)


def test_move_targets(tables, names):
    base = four(tables, names)
    for move, want in (
        ("Tackle", {1, 2, 3}),
        ("Sunny Day", {0, 1, 2, 3}),
        ("Earthquake", {1, 2, 3}),
    ):
        s = with_moves(base, names, 0, (move,))
        c0 = _choices_match_reference(tables, s, 0)
        assert _targets(c0, 0) == want, move
    s = with_moves(base, names, 0, ("Tackle",), pp=0)  # Struggling: action 0 only
    c0 = _choices_match_reference(tables, s, 0)
    assert _targets(c0, 0) == {0}
    assert not c0[1:4].any()


def test_multi_player_commands_battler_0_alone(tables, names):
    s = four(tables, names, MULTI)
    c0 = _choices_match_reference(tables, s, 0)
    assert all(c0[4 + k, 0] for k in range(1, 6)) and not c0[4, 0]  # battler 2's party is apart
    c2 = _choices_match_reference(tables, s, 2)
    assert c2[4 + 1, 2]
    with pytest.raises(ValueError):
        decision_pairs(tables, s)


def test_b2_switch_targets_are_todays_switch_mask(tables, names):
    members = [mon(names) for _ in range(3)]
    s = make_state(tables, members, members)
    for case in (
        s,
        s._replace(trapped=s.trapped.at[0].set(2)),
        s._replace(locked_move=s.locked_move.at[0].set(s.moves[0, 0, 0])),
        s._replace(hp=s.hp.at[0, 2].set(0)),
    ):
        for b in (0, 1):
            assert (
                switch_targets(tables, case, b).tolist()
                == legal_actions(tables, case, b)[4:].tolist()
            )
            assert len(legal_actions(tables, case, b)) == NUM_ACTIONS
    assert jnp.asarray(switch_targets(tables, s, 0)).tolist() == [False, True, True] + [False] * 3
