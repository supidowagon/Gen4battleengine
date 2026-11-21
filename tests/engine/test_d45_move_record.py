"""D45-4: the move record per deciding AI battler (``format.ai_battlers``,
``step.record_last_moves``; trainer_ai.c:2705-2717, :265, :295, :362-375) and D45-4a: an Encored
or recharging AI battler's FIGHT command never opens the move menu (battle_controller_player.c:
442-454; ``Battler_CanPickCommand``, battle_lib.c:6224-6236), so it records nothing. JAX
``_record_last_move`` against the reference ``_record_last_moves`` on the same input (direct
calls: no ``step`` compile, fast tier), plus one slow B = 4 ``step`` test."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS, locked_slot
from nuzline.engine.format import (
    DOUBLES,
    MULTI,
    SINGLES,
    TWO_TRAINERS,
    ai_battlers,
    format_of,
)
from nuzline.engine.order import selected_move
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import ai_record_moves, ai_record_write, record_seen_move
from nuzline.engine.step import _record_last_move, record_last_moves, step
from nuzline.reference import engine as ref
from nuzline.reference.convert import from_ref, to_ref
from tests.engine.d2_states import four, with_hp, with_moves
from tests.engine.helpers import roark_state
from tests.strategies import legal_action_pair, supported_state

SW = MOVE_ACTIONS + 2  # a switch action


def _record(tables, s, acts):
    """Both engines' record on ``s`` with ``acts``; asserted equal; returns the JAX rows."""
    out = _record_last_move(tables, s, jnp.asarray(acts, jnp.int32))
    r = to_ref(s)
    t = to_numpy(tables)
    choices = [ref._choose(t, r, b, acts[b]) for b in range(format_of(s).B)]
    ref._record_last_moves(t, r, tuple(acts), choices)
    assert np.asarray(out.ai_seen_moves).tolist() == r.ai_seen_moves
    return np.asarray(out.ai_seen_moves)


def _singles(tables, names):
    """Roark's fixture with the player's last move Tackle and nothing recorded yet."""
    s = roark_state(tables, names)
    return s._replace(last_move=s.last_move.at[0].set(names.move("Tackle")))


def test_encored_enemy_records_nothing(tables, names):
    """D45-4a, bcp.c:442-454: battler 1 is Encored into a move it knows; the same state without
    the Encore records Tackle in row 0."""
    s = _singles(tables, names)
    tackle = names.move("Tackle")
    assert _record(tables, s, [0, 0])[0].tolist() == [tackle, 0, 0, 0]
    encored = s._replace(
        encore_move=s.encore_move.at[1].set(tackle),
        encore_slot=s.encore_slot.at[1].set(0),
        encore_turns=s.encore_turns.at[1].set(3),
    )
    assert _record(tables, encored, [0, 0])[0].tolist() == [0, 0, 0, 0]


def test_recharging_enemy_records_nothing(tables, names):
    """D45-4a, battle_lib.c:6228: a recharging enemy picks no command."""
    s = _singles(tables, names)
    s = s._replace(recharge=s.recharge.at[1].set(1))
    assert _record(tables, s, [0, 0])[0].tolist() == [0, 0, 0, 0]


def _rows(seen) -> dict:
    """``{battler: [moves]}`` for the non-empty rows."""
    return {t: [int(m) for m in row if m] for t, row in enumerate(seen) if row.any()}


def _fmt4(tables, names, fmt, last=("Tackle", "Ember", "Curse", "Growl")):
    s = four(tables, names, fmt)
    ids = [names.move(m) if m else 0 for m in last]
    return s._replace(last_move=jnp.asarray(ids, s.last_move.dtype))


def _expect(names, *rows) -> dict:
    """The record of battlers ``rows`` of the default last moves (Tackle, Ember, Curse, Growl)."""
    own = ("Tackle", "Ember", "Curse", "Growl")
    return {t: [names.move(own[t])] for t in rows}


def test_ai_battlers():
    assert ai_battlers(SINGLES) == (1,)
    assert ai_battlers(DOUBLES) == ai_battlers(TWO_TRAINERS) == (1, 3)
    assert ai_battlers(MULTI) == (1, 2, 3)


def test_doubles_enemies_record_player_rows(tables, names):
    s = _fmt4(tables, names, DOUBLES)
    assert _rows(_record(tables, s, [0, 0, 0, 0])) == _expect(names, 0, 2)  # test 1


def test_a_battler_without_hp_is_absent(tables, names):
    s = _fmt4(tables, names, DOUBLES)
    # battler 1 at 0 HP decides nothing; battler 3 records both rows (R5)
    out = _record(tables, with_hp(s, 1, 0), [0, 0, 0, 0])
    assert _rows(out) == _expect(names, 0, 2)  # test 2
    # a target at 0 HP is not recorded (trainer_ai.c:363)
    out = _record(tables, with_hp(s, 2, 0), [0, 0, 0, 0])
    assert _rows(out) == _expect(names, 0)  # test 3


def test_switching_encored_locked_and_struggling_enemies_record_nothing(tables, names):
    s = _fmt4(tables, names, DOUBLES)
    assert _rows(_record(tables, s, [0, SW, 0, SW])) == {}  # test 4
    tackle = names.move("Tackle")
    encored = s._replace(
        encore_move=s.encore_move.at[1].set(tackle),
        encore_slot=s.encore_slot.at[1].set(0),
        encore_turns=s.encore_turns.at[1].set(3),
    )
    assert _rows(_record(tables, encored, [0, 0, 0, SW])) == {}  # test 5, D45-4a
    locked = s._replace(locked_move=s.locked_move.at[1].set(tackle))
    locked = with_moves(locked, names, 3, ["Tackle"], pp=0)  # Struggling (bcp.c:433)
    assert _rows(_record(tables, locked, [0, 0, 0, 0])) == {}  # test 6


def test_two_trainers_records_rows_0_and_2(tables, names):
    s = _fmt4(tables, names, TWO_TRAINERS)
    assert _rows(_record(tables, s, [0, 0, 0, 0])) == _expect(names, 0, 2)  # test 7


def test_multi_partner_records_the_enemy_rows(tables, names):
    s = _fmt4(tables, names, MULTI)
    assert _rows(_record(tables, s, [0, SW, 0, SW])) == _expect(names, 1, 3)  # test 8
    assert _rows(_record(tables, s, [0, 0, SW, 0])) == _expect(names, 0, 2)  # test 9
    assert _rows(_record(tables, s, [0, 0, 0, 0])) == _expect(names, 0, 1, 2, 3)  # test 10


def test_the_record_is_not_duplicated_and_none_writes_nothing(tables, names):
    s = _fmt4(tables, names, DOUBLES)
    tackle = names.move("Tackle")
    listed = s._replace(ai_seen_moves=s.ai_seen_moves.at[0, 0].set(tackle))
    assert _rows(_record(tables, listed, [0, 0, 0, 0])) == {
        0: [tackle],
        2: [names.move("Curse")],
    }  # test 11
    none = s._replace(last_move=s.last_move.at[0].set(0))
    assert _rows(_record(tables, none, [0, 0, 0, 0])) == _expect(names, 2)  # test 12


def test_record_last_moves_takes_a_prefix_of_deciders(tables, names):
    s = _fmt4(tables, names, MULTI)
    acts = jnp.asarray([0, 0, 0, 0], jnp.int32)
    out = record_last_moves(tables, s, acts, (1,))
    assert _rows(np.asarray(out.ai_seen_moves)) == _expect(names, 0, 2)  # test 13


def _old_record(tables, s, acts):
    """The B = 2 rule before D45-4: battler 1 fights (a move action, not Struggling, not
    locked) -> row 0 takes ``last_move[0]``."""
    _, struggling = selected_move(tables, s, 1, acts[1])
    fight = (acts[1] < MOVE_ACTIONS) & ~struggling & (locked_slot(s, 1) < 0)
    row = record_seen_move(ai_record_moves(s, 0), s.last_move[0])
    return ai_record_write(s, "ai_seen_moves", 0, s.active[0], row, fight)


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@given(data=st.data())
def test_singles_record_equals_the_old_rule_unless_encored(tables, names, data):
    """Test 14: at B = 2 the new record is the old row-0 rule while battler 1 is not encored
    and not recharging, in both engines."""
    s = data.draw(supported_state(tables, names))
    acts = [int(a) for a in np.asarray(data.draw(legal_action_pair(tables, s)))]
    if int(s.encore_move[1]) != 0 or int(s.recharge[1]) != 0:
        return
    new = _record(tables, s, acts)
    old = _old_record(tables, s, jnp.asarray(acts, jnp.int32))
    assert new.tolist() == np.asarray(old.ai_seen_moves).tolist()


@pytest.mark.slow
def test_step_records_before_the_actions(tables, names):
    """Test 16: B = 4 ``step`` against ``ref_step``: rows 0 and 2 hold the turn-start
    ``last_move`` values (the record runs before the actions overwrite ``last_move``)."""
    s = _fmt4(tables, names, DOUBLES)
    acts = jnp.asarray([0, 0, 0, 0], jnp.int32)
    targets = [1, 0, 1, 0]
    o = empty_outcome(DOUBLES)
    out = step(tables, s, acts, o, jnp.asarray(targets, jnp.int8))
    ref_out = from_ref(
        ref.ref_step(
            to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o), targets
        )
    )
    assert _rows(np.asarray(out.ai_seen_moves)) == _expect(names, 0, 2)
    for name, a, b in zip(out._fields, out, ref_out, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
