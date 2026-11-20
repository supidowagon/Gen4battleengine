"""Turn order (``BattleControllerPlayer_CalcTurnOrder``, battle_controller_player.c:707, and
``BattleSystem_CompareBattlerSpeed``, battle_lib.c:1188)."""

import jax.numpy as jnp
import pytest

from nuzline.engine.field import with_trick_room
from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import roark_state

SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe
MACH_PUNCH, SCRATCH = 3, 0  # Monferno's slots
TACKLE = 0  # every fixture opponent's slot 0


def with_speed(s, side, speed):
    return s._replace(stats=s.stats.at[side, 0, SPE].set(speed))


def order(tables, s, actions, **outcome):
    o = empty_outcome()._replace(**outcome)
    return action_order(tables, s, jnp.array(actions, jnp.int32), o).tolist()


def test_order_dtype_and_speed(tables, names):
    s = roark_state(tables, names)  # Monferno Spe 30 vs Geodude Spe 10
    assert action_order(tables, s, jnp.array([0, 0]), empty_outcome()).dtype == jnp.int8
    assert order(tables, s, [SCRATCH, TACKLE]) == [0, 1]
    assert order(tables, with_speed(s, 1, 31), [SCRATCH, TACKLE]) == [1, 0]


def test_priority_beats_speed(tables, names):
    s = with_speed(roark_state(tables, names), 1, 200)
    assert order(tables, s, [MACH_PUNCH, TACKLE]) == [0, 1]
    assert order(tables, s, [SCRATCH, TACKLE]) == [1, 0]


def test_switch_goes_before_any_move(tables, names):
    s = roark_state(tables, names)
    assert order(tables, s, [MACH_PUNCH, 4 + 1]) == [1, 0]


def test_two_switches_go_by_speed_ignoring_priority(tables, names):
    s = roark_state(tables, names)
    assert order(tables, s, [4 + 1, 4 + 1]) == [0, 1]
    assert order(tables, with_speed(s, 1, 31), [4 + 1, 4 + 1]) == [1, 0]


def test_paralysis_quarters_speed(tables, names):
    s = with_speed(roark_state(tables, names), 1, 8)
    assert order(tables, s, [SCRATCH, TACKLE]) == [0, 1]
    para = s.with_status(0, 0, Status.PARALYSIS)  # 30 // 4 = 7 < 8
    assert order(tables, para, [SCRATCH, TACKLE]) == [1, 0]


def test_speed_stages_and_simple(tables, names):
    s = with_speed(roark_state(tables, names), 1, 20)
    slowed = s.with_stages(0, spe=-1)  # 30 * 10 // 15 = 20: a speed tie
    assert order(tables, slowed, [SCRATCH, TACKLE], speed_tie=jnp.int8(0)) == [0, 1]
    assert order(tables, slowed, [SCRATCH, TACKLE], speed_tie=jnp.int8(1)) == [1, 0]
    assert order(tables, s.with_stages(0, spe=-2), [SCRATCH, TACKLE]) == [1, 0]  # 15 < 20
    simple = s._replace(ability=s.ability.at[0, 0].set(AbilityId.SIMPLE)).with_stages(0, spe=-1)
    assert order(tables, simple, [SCRATCH, TACKLE]) == [1, 0]  # Simple: -1 counts as -2


def test_trick_room_reverses_speed(tables, names):
    s = with_trick_room(roark_state(tables, names), 5)
    assert order(tables, s, [SCRATCH, TACKLE]) == [1, 0]
    assert order(tables, s, [MACH_PUNCH, TACKLE]) == [0, 1]  # priority still first


def test_tailwind_doubles_speed(tables, names):
    s = with_speed(roark_state(tables, names), 1, 20)
    tail = s._replace(tailwind=s.tailwind.at[1].set(3))
    assert order(tables, tail, [SCRATCH, TACKLE]) == [1, 0]


def with_priorities(tables, priorities):
    """``tables`` with ``move_priority`` replaced for the given move ids."""
    prio = tables.move_priority
    for move, value in priorities.items():
        prio = prio.at[move].set(value)
    return tables._replace(move_priority=prio)


@pytest.mark.slow
def test_priority_plus_seven_beats_plus_five(tables, names):
    # K2b Task 15 regression guard (passes before K2b): Kaizo's Trick Room is +7 (spec 5.2);
    # order.action_order and enumerate.outcome_laws cast move_priority (int8) to int32.
    from nuzline.engine.enumerate import outcome_laws

    s = roark_state(tables, names)  # Monferno (Scratch, slot 0) is faster than Geodude
    t = with_priorities(tables, {names.move("Tackle"): 7, names.move("Scratch"): 5})
    assert order(t, s, [SCRATCH, TACKLE]) == [1, 0]  # the slower +7 user moves first
    tied = roark_state(tables, names, equal_speed=True)
    actions = jnp.array([SCRATCH, TACKLE], jnp.int32)
    assert not bool(outcome_laws(t, tied, actions).relevant.speed_tie)  # +5 vs +7: no tie to draw
    same = with_priorities(tables, {names.move("Tackle"): 7, names.move("Scratch"): 7})
    assert bool(outcome_laws(same, tied, actions).relevant.speed_tie)
