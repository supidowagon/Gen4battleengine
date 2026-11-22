"""An armed Trace mid-turn (``outcome.trace_copies``): the laws read the abilities it can copy.

A Trace holder whose ``traceAnnounced`` is clear fires at the next
``BattleSystem_TriggerEffectOnSwitch`` that finds a target: ``BattleControllerPlayer_MoveEnd``
runs it after every action (battle_controller_player.c:3973; a bag item reaches MoveEnd through
:1971-1973, a switch through CHECK_ANY_FAINTED, :1984, :4008). ``SWITCH_IN_CHECK_STATE_TRACE``
(battle_lib.c:3711-3737) takes ``ChooseTraceTarget``'s pick (:7750-7781, the raw
``BattleMon.ability``, so a Gastro Acid suppressed Static is copied; subscript_trace.s:6-7). The
copied Static then rolls on the next contact hit on its holder (battle_lib.c:4213-4222)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.damage import STATUS
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId
from nuzline.engine.outcome import CRIT_WRITE_VARIANTS, Outcome, empty_outcome, trace_sources
from tests.engine.d2_states import four, party_of, with_moves
from tests.engine.helpers import A, roark_state
from tests.engine.test_enumerate import _assert_mask_sound

_laws = jax.jit(outcome_laws)


def _laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, jnp.asarray(actions, jnp.int32)))


def _bag_trace(tables, names, *, trace=AbilityId.TRACE, announced=0):
    """Player Pikachu (Static, suppressed by Gastro Acid) with Spark against an opposing Pikachu
    holding ``trace`` with ``trace_announced`` ``announced``, at half HP, whose action is a bag
    Potion (``FIRST_ITEM_ACTION``): the item goes first and ends at MoveEnd."""
    s = roark_state(
        tables,
        names,
        player_species="Pikachu",
        player_moves=("Spark",),
        player_ability="Static",
        opp_species="Pikachu",
    )
    return s._replace(
        ability=s.ability.at[1, 0].set(trace),
        trace_announced=s.trace_announced.at[1].set(announced),
        ability_suppressed=s.ability_suppressed.at[0].set(1),
        hp=s.hp.at[1, 0].set(s.max_hp[1, 0] // 2),
    )


_ACTS = np.array([0, FIRST_ITEM_ACTION], np.int32)


def test_an_armed_trace_reads_the_contact_roll_of_the_static_it_copies(tables, names):
    # The foe's Potion's MoveEnd fires the armed Trace, which copies the raw Static; the player's
    # Spark then reads Static's roll (contact_proc[defending side, hit]).
    laws = _laws_of(tables, _bag_trace(tables, names), _ACTS)
    assert bool(laws.relevant.contact_proc[1, 0])
    # The copy can change the hit's accuracy threshold and crit odds: both read whole.
    assert laws.hit_whole.tolist() == [True, True]


def test_an_announced_trace_reads_no_contact_roll(tables, names):
    # Control: a Trace that has announced never fires again (battle_lib.c:3726), and no other
    # Static is on the field.
    laws = _laws_of(tables, _bag_trace(tables, names, announced=1), _ACTS)
    assert not bool(laws.relevant.contact_proc[1, 0])
    assert laws.hit_whole.tolist() == [False, False]
    assert not laws.crit_possible[:, CRIT_WRITE_VARIANTS:].any()


@pytest.mark.slow  # about 70 s, 8.6 GB peak locally
def test_an_armed_trace_board_steps_alike_whatever_it_leaves_undrawn(tables, names):
    # The mask check of the failing board: every element the laws leave undrawn takes another
    # value (the contact roll proc'ing, and every other field off its empty value).
    s = _bag_trace(tables, names)
    empty = empty_outcome()
    ones = Outcome(*(np.ones_like(np.asarray(f)) for f in empty))
    procs = empty._replace(contact_proc=jnp.ones_like(empty.contact_proc))
    for arbitrary in (procs, ones):
        _assert_mask_sound(tables, s, _ACTS, arbitrary, 13498)


def test_an_armed_trace_that_can_copy_super_luck_classes_the_crit_over_every_stage(tables, names):
    # Super Luck raises the copier's crit stage (battle_lib.c:7126): its odds at the hit can be
    # any stage's, so the trace columns of crit_variants are possible for it, and only for it.
    s = _bag_trace(tables, names)
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.SUPER_LUCK))
    laws = _laws_of(tables, s, _ACTS)
    trace = laws.crit_possible[:, CRIT_WRITE_VARIANTS:]
    assert trace[1].all() and not trace[0].any()


@pytest.mark.slow  # about 96 s locally (the B = 4 laws' compile)
def test_b4_an_armed_trace_reads_the_contact_roll_of_a_static_partner(tables, names):
    # DOUBLES: battler 1's armed Trace can pick battler 0 or battler 2 (ChooseTraceTarget's coin,
    # battle_lib.c:7754-7767); battler 2 holds Static, so battler 0's Tackle on battler 1 reads
    # the contact roll. Every battler uses Tackle.
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ("Tackle",))
    p1, p2 = party_of(s, 1), party_of(s, 2)
    s = s._replace(
        ability=s.ability.at[p1, int(s.active[1])]
        .set(AbilityId.TRACE)
        .at[p2, int(s.active[2])]
        .set(AbilityId.STATIC),
        trace_announced=s.trace_announced.at[1].set(0),
    )
    laws = _laws_of(tables, s, np.zeros(4, np.int32))
    assert bool(laws.relevant.contact_proc[1, 0])
    announced = s._replace(trace_announced=s.trace_announced.at[1].set(1))
    laws = _laws_of(tables, announced, np.zeros(4, np.int32))
    assert not bool(laws.relevant.contact_proc[1, 0])


def test_a_role_play_of_a_trace_holder_arms_its_user(tables, names):
    # Role Play copies the raw ability with no Trace test and leaves traceAnnounced
    # (subscript_copy_ability.s:5-15); Transform clears it (battle_script.c:5352). The user's
    # Trace fires at its move's end (bcp.c:3672-3673, :3973): it is armed for the turn.
    s = roark_state(tables, names, player_moves=("Role Play",), opp_moves=("Leer",))
    s = s._replace(
        ability=s.ability.at[1, 0].set(AbilityId.TRACE),
        trace_announced=s.trace_announced.at[1].set(1),
    )
    armed = trace_sources(tables, s, A("Role Play", "Leer", s, names)).armed
    assert np.asarray(armed).tolist() == [True, False]
    calm = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Leer",))
    calm = calm._replace(ability=s.ability, trace_announced=s.trace_announced)
    armed = trace_sources(tables, calm, A("Tackle", "Leer", calm, names)).armed
    assert np.asarray(armed).tolist() == [False, False]


def test_no_damaging_move_acts_after_a_roar_drag(tables, kaizo_tables):
    # A Roar newcomer enters in the Roar script (subscript_force_target_to_switch_or_flee.s:
    # 20-31), priority -6: no damaging move of either game is at -6 or below, so no hit of the turn
    # meets what a Trace copies from it (outcome.trace_sources).
    for t in (tables, kaizo_tables):
        priority = np.asarray(t.move_priority).astype(int)
        damaging = (np.asarray(t.move_category) != STATUS) & (np.asarray(t.move_power) != 0)
        assert not np.any(damaging & (priority <= -6))
