"""D45-6: ``legal_opponent_actions(tables, state, b)`` per AI battler (``format.ai_battlers``) and
no AI bags in MULTI (``BattleControllerPlayer_InitAI``, battle_controller_player.c:4796-4813;
``BATTLE_TYPE_NO_AI_ITEMS``, constants/battle.h:46-51; encounter.c:730-735). Fast tier except the
item-action step test (a B = 4 ``step`` compile)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    NUM_ACTIONS,
    legal_actions,
    legal_opponent_actions,
    usable_trainer_items,
)
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS, ai_battlers
from nuzline.engine.state import BattleState, initial_arrays
from tests.engine.d2_states import mon, with_hp
from tests.strategies import four_state, supported_state

FOURS = (DOUBLES, TWO_TRAINERS, MULTI)


def _state(tables, names, fmt, opp_items=(), opp2_items=None):
    """A turn-0 state of ``fmt`` with the trainers' own bags passed whatever the format."""
    members = [mon(names) for _ in range(3)]
    parties = [list(members) for _ in range(fmt.P)]
    if opp2_items is None and fmt.B == 4:
        opp2_items = ()
    a = initial_arrays(
        fmt,
        tables,
        parties,
        opp_items=[names.item(i) for i in opp_items],
        opp2_items=None if opp2_items is None else [names.item(i) for i in opp2_items],
    )
    return BattleState(**{k: jnp.asarray(v) for k, v in a.items()})


def _items(tables, s, b):
    return np.asarray(legal_opponent_actions(tables, s, b))[NUM_ACTIONS:].tolist()


@settings(
    max_examples=30,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@given(data=st.data())
def test_singles_equals_the_old_body(tables, names, data):
    """Test 1: B = 2, default b and b = 1 give the old body."""
    s = data.draw(supported_state(tables, names))
    old = np.concatenate(
        [np.asarray(legal_actions(tables, s, 1)), np.asarray(usable_trainer_items(tables, s, 1))]
    )
    assert np.asarray(legal_opponent_actions(tables, s)).tolist() == old.tolist()
    assert np.asarray(legal_opponent_actions(tables, s, 1)).tolist() == old.tolist()


def test_two_trainers_each_battler_its_own_bag(tables, names):
    """Test 2."""
    s = _state(tables, names, TWO_TRAINERS, ["Potion"], ["Full Restore", "Hyper Potion"])
    assert _items(tables, s, 1) == [True, False, False, False]
    assert _items(tables, s, 3) == [True, True, False, False]
    assert np.asarray(s.trainer_item_count).tolist() == [0, 1, 0, 2]


def test_doubles_copy_and_talked_to(tables, names):
    """Test 3: the spotted pair holds a copy of the bag; talked to, battler 3 has none."""
    s = _state(tables, names, DOUBLES, ["Potion"], ["Potion"])
    assert _items(tables, s, 1) == [True, False, False, False]
    assert _items(tables, s, 3) == [True, False, False, False]
    s = _state(tables, names, DOUBLES, ["Potion"], [])
    assert _items(tables, s, 3) == [False, False, False, False]


def test_multi_loads_no_bag(tables, names):
    """Test 4: MULTI drops both passed bags; every AI battler's item part is empty."""
    s = _state(tables, names, MULTI, ["Potion"], ["Potion"])
    assert not np.asarray(s.trainer_items).any()
    assert not np.asarray(s.trainer_item_count).any()
    for b in (1, 2, 3):
        assert _items(tables, s, b) == [False] * 4
    got = np.asarray(legal_opponent_actions(tables, s, 2))[:NUM_ACTIONS]
    assert got.tolist() == np.asarray(legal_actions(tables, s, 2)).tolist()


@pytest.mark.parametrize("fmt", [SINGLES, *FOURS])
def test_non_ai_battler_raises(tables, names, fmt):
    """Test 5: a player-commanded battler has no opponent action set."""
    s = _state(tables, names, fmt) if fmt.B == 4 else _singles(tables, names)
    with pytest.raises(ValueError):
        legal_opponent_actions(tables, s, 0)
    for b in range(1, 4):
        if b not in ai_battlers(fmt):
            with pytest.raises(ValueError):
                legal_opponent_actions(tables, s, b)


def _singles(tables, names):
    a = initial_arrays(SINGLES, tables, [[mon(names)], [mon(names)]])
    return BattleState(**{k: jnp.asarray(v) for k, v in a.items()})


def test_fainted_or_locked_battler_has_no_item(tables, names):
    """Test 6: no item at 0 HP, none while locked into a move."""
    s = _state(tables, names, TWO_TRAINERS, ["Potion"], ["Full Restore", "Hyper Potion"])
    assert _items(tables, with_hp(s, 3, 0), 3) == [False] * 4
    locked = s._replace(locked_move=s.locked_move.at[3].set(names.move("Tackle")))
    assert _items(tables, locked, 3) == [False] * 4
    assert _items(tables, locked, 1) == [True, False, False, False]


@pytest.mark.slow
def test_item_action_spends_its_own_battlers_bag(tables, names):
    """Test 7: battler 3 uses slot 0; battler 1's bag is untouched (JAX against the reference)."""
    from tests.engine.test_d3_action_loop import play

    s = _state(tables, names, TWO_TRAINERS, ["Potion"], ["Full Restore", "Hyper Potion"])
    s = with_hp(s, 3, 1)
    sw = 1  # Swords Dance
    out, _ = play(tables, s, [sw, sw, sw, FIRST_ITEM_ACTION + 0], None)
    assert int(out.trainer_items[3, 0]) == 0
    assert out.trainer_items[1].tolist() == s.trainer_items[1].tolist()


@settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_four_state_bags(tables, names, data):
    """Test 8: MULTI draws no bags."""
    s = data.draw(four_state(tables, names, MULTI))
    assert not np.asarray(s.trainer_items).any()
    assert not np.asarray(s.trainer_item_count).any()


def test_four_state_other_formats_draw_bags(tables, names):
    """Test 8: DOUBLES and TWO_TRAINERS draws still give non-empty bags in some example."""
    seen = {DOUBLES: False, TWO_TRAINERS: False}

    for fmt in seen:

        @settings(
            max_examples=20,
            deadline=None,
            derandomize=True,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        )
        @given(data=st.data())
        def run(data, fmt=fmt):
            s = data.draw(four_state(tables, names, fmt))
            if np.asarray(s.trainer_item_count).any():
                seen[fmt] = True

        run()
    assert all(seen.values())
