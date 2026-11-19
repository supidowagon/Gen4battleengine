"""D2-11: Follow Me's per-side turn scratch ``ts_follow_me`` (1 + followMeUser; script.c:6261-6270,
cleared by SetupNextTurn, lib.c:2251-2252) and its reader ``targets.follow_me_target``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine import targets
from nuzline.engine.ids import Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import TURN_SCRATCH, make_state
from tests.engine.d2_states import four, with_hp
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon, _resume

O = empty_outcome()


def _pause_after_follow_me(tables, names, user_moves, asleep=False):
    """The player's Togepi uses a Follow Me (priority +3, or called by Sleep Talk when asleep and
    faster); the foe's slower U-turn pauses at its party list later in the same turn."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Togepi", user_moves, 50)],
        [_mon(n, "Geodude", ["U-turn"], 5), _mon(n, "Geodude", ["Tackle"], 5)],
    )
    if asleep:
        s = s.with_status(0, 0, Status.SLEEP, 5)
    return s, assert_matches_reference(
        tables, s, (user_moves.index("Follow Me") if not asleep else 0, 0), O
    )


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_follow_me_sets_the_side_field_and_the_turn_end_clears_it(tables, names):
    _, paused = _pause_after_follow_me(tables, names, ["Follow Me"])
    assert int(paused.pivot_battler) == 2
    assert np.asarray(paused.ts_follow_me).tolist() == [1, 0]  # 1 + battler 0
    after = _resume(tables, paused, 1)  # the resumed half reads it back; the turn ends
    assert int(after.pivot_battler) == 0
    assert np.asarray(after.ts_follow_me).tolist() == [0, 0]
    for name in TURN_SCRATCH:
        assert not np.any(np.asarray(getattr(after, name))), name


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_a_called_follow_me_counts(tables, names):
    # Sleep Talk's only candidate is Follow Me (effect_used is the called move's)
    _, paused = _pause_after_follow_me(tables, names, ["Sleep Talk", "Follow Me"], asleep=True)
    assert np.asarray(paused.ts_follow_me).tolist() == [1, 0]
    after = _resume(tables, paused, 1)
    assert np.asarray(after.ts_follow_me).tolist() == [0, 0]


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_a_turn_without_follow_me_leaves_it_clear(tables, names):
    s = make_state(
        tables, [_mon(names, "Togepi", ["Tackle"], 50)], [_mon(names, "Geodude", ["Tackle"], 5)]
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert np.asarray(out.ts_follow_me).tolist() == [0, 0]


def test_follow_me_target_reader(tables, names):
    s = four(tables, names)
    fm = s._replace(ts_follow_me=jnp.asarray([0, 4], jnp.int8))  # side 1's user is battler 3
    assert int(targets.follow_me_target(fm, 0)) == 3  # the user alive
    assert int(targets.follow_me_target(fm, 2)) == 3
    assert int(targets.follow_me_target(with_hp(fm, 3, 0), 0)) == targets.NO_DEFENDER  # fainted
    assert int(targets.follow_me_target(s, 0)) == targets.NO_DEFENDER  # flag clear
    assert int(targets.follow_me_target(fm, 1)) == targets.NO_DEFENDER  # the user's own side
    assert int(targets.follow_me_target(fm, 3)) == targets.NO_DEFENDER
