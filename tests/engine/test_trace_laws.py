"""An armed Trace mid-turn (``outcome.trace_copies``): the laws read the abilities it can copy.

A Trace holder whose ``traceAnnounced`` is clear fires at the next
``BattleSystem_TriggerEffectOnSwitch`` that finds a target: ``BattleControllerPlayer_MoveEnd``
runs it after every action (battle_controller_player.c:3973; a bag item reaches MoveEnd through
:1971-1973, a switch through CHECK_ANY_FAINTED, :1984, :4008). ``SWITCH_IN_CHECK_STATE_TRACE``
(battle_lib.c:3711-3737) takes ``ChooseTraceTarget``'s pick (:7750-7781, the raw
``BattleMon.ability``; subscript_trace.s:6-7). ShowBattleMon runs the same pass before every
turn's commands (bcp.c:256), so at a turn start an unannounced holder faces nothing traceable
(the Trace invariant, src/nuzline/engine/STATE.md): it copies only what enters or changes."""

from __future__ import annotations

import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine import outcome as outcome_mod
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.damage import STATUS
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import (
    ACCURACY_ABILITIES,
    CRIT_WRITE_VARIANTS,
    Outcome,
    ability_whole,
    doubles_ability_change,
    empty_outcome,
    trace_copies,
    trace_sources,
)
from tests.engine.d2_states import four, party_of, with_moves
from tests.engine.helpers import A, roark_state
from tests.engine.test_enumerate import _assert_mask_sound
from tests.strategies import settle_traces

_laws = jax.jit(outcome_laws)


def _laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, jnp.asarray(actions, jnp.int32)))


_ITEM = np.array([0, FIRST_ITEM_ACTION], np.int32)  # the foe's bag Potion goes first


def test_an_unannounced_trace_facing_a_target_is_no_turn_start_state(tables, names):
    # Two generator falsifiers of the first fix (test_enumerate.py::test_mask_sound): an
    # unannounced Trace facing a traceable foe whose bag item reached MoveEnd, the copy then
    # changing a later read (a suppressed Static's contact roll; a Compound Eyes accuracy).
    # ShowBattleMon had fired that Trace before the turn's commands (bcp.c:256; battle_lib.c:
    # 3711-3737), so no turn start holds it: the generator settles it (strategies.settle_traces)
    # and the laws read no copy.
    static = roark_state(
        tables,
        names,
        player_species="Pikachu",
        player_moves=("Spark",),
        player_ability="Static",
        opp_species="Pikachu",
    )
    static = static._replace(
        ability=static.ability.at[1, 0].set(AbilityId.TRACE),
        ability_suppressed=static.ability_suppressed.at[0].set(1),
        hp=static.hp.at[1, 0].set(static.max_hp[1, 0] // 2),
    )
    eyes = roark_state(tables, names, player_species="Sandshrew")
    eyes = eyes._replace(
        ability=eyes.ability.at[0, 0].set(AbilityId.TRACE).at[1, 0].set(AbilityId.COMPOUNDEYES),
        hp=eyes.hp.at[1, 0].set(eyes.max_hp[1, 0] // 2),
    )
    for board, holder in ((static, 1), (eyes, 0)):
        settled = settle_traces(tables, board)
        flags = np.asarray(settled.trace_announced).tolist()
        assert flags == [int(b == holder) for b in range(2)]
        assert not np.asarray(trace_copies(tables, settled, _ITEM)).any()
        laws = _laws_of(tables, settled, _ITEM)
        assert laws.hit_whole.tolist() == [False, False]
        assert not laws.crit_possible[:, CRIT_WRITE_VARIANTS:].any()
        if holder == 1:
            assert not bool(laws.relevant.contact_proc[1, 0])


def _standoff(tables, names, bench_ability=AbilityId.SUPER_LUCK):
    """Both leads hold Trace, which neither can copy (ChooseTraceTarget skips Trace,
    battle_lib.c:7755-7757), so both stay armed at the turn start (G-25); the player's
    slot 1 holds ``bench_ability``."""
    s = roark_state(tables, names, player_moves=("Tackle",))
    return s._replace(
        ability=s.ability.at[0, 0]
        .set(AbilityId.TRACE)
        .at[1, 0]
        .set(AbilityId.TRACE)
        .at[0, 1]
        .set(bench_ability)
    )


_SWITCH = np.array([MOVE_ACTIONS + 1, 0], np.int32)  # the player switches to slot 1


def test_a_standoff_trace_copies_the_newcomer_and_worry_seeds_insomnia(tables, names):
    # The foe's armed Trace takes the player's newcomer (ChooseTraceTarget at its switch-in or
    # a MoveEnd), and a Worry Seed's Insomnia (subscript_give_target_insomnia.s:11) where one is
    # used; nothing else enters or changes, so nothing else is copied.
    s = _standoff(tables, names)
    assert np.asarray(settle_traces(tables, s).trace_announced).tolist() == [0, 0]
    copies = trace_copies(tables, s, _SWITCH, (AbilityId.SUPER_LUCK,))
    assert np.asarray(copies).tolist() == [False, True]
    stay = A("Tackle", "Tackle", s, names)
    assert not np.asarray(trace_copies(tables, s, stay)).any()
    seed = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Worry Seed",))
    seed = seed._replace(ability=s.ability)
    acts = A("Tackle", "Worry Seed", seed, names)
    # The foe's Worry Seed writes the player's Insomnia: the foe's Trace can copy it at that
    # MoveEnd; the player's Trace is gone (it is the target), so only the foe copies.
    insomnia = trace_copies(tables, seed, acts, (AbilityId.INSOMNIA,))
    assert np.asarray(insomnia).tolist() == [False, True]
    assert not np.asarray(trace_copies(tables, seed, acts, (AbilityId.SUPER_LUCK,))).any()
    # B = 2: the foe's Role Play takes the player's Trace (subscript_copy_ability.s:14-15), which
    # ChooseTraceTarget skips: a changed foe gives the player's armed Trace nothing to copy.
    role = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Role Play",))
    role = role._replace(ability=s.ability)
    assert not np.asarray(trace_copies(tables, role, A("Tackle", "Role Play", role, names))).any()


def test_a_trace_newcomer_can_copy_a_foe_that_switches_out_after_it(tables, names):
    # The player switches in a Trace Pokemon while the foe's Static lead switches out: if the
    # player's switch goes first, the newcomer's entry trace meets the Static (ChooseTraceTarget,
    # battle_lib.c:7750-7781); ``faced`` counts only staying foes, so the newcomer is armed and
    # the foe's active as it stands is a source (``TraceSources.start``).
    s = roark_state(tables, names, player_moves=("Tackle",))
    s = s._replace(ability=s.ability.at[0, 1].set(AbilityId.TRACE).at[1, 0].set(AbilityId.STATIC))
    both = np.array([MOVE_ACTIONS + 1, MOVE_ACTIONS + 1], np.int32)
    assert bool(trace_copies(tables, s, both, (AbilityId.STATIC,))[0])
    # The foe stays: the newcomer traces the Static on entry (``_entry_abilities``), no armed one.
    stay = np.array([MOVE_ACTIONS + 1, 0], np.int32)
    assert not bool(trace_sources(tables, s, stay).armed[0])


def test_a_paused_state_is_no_turn_start_for_the_invariant(tables, names):
    # A state paused mid-turn (a learn prompt, a U-turn's party list) follows no ShowBattleMon:
    # an armed Trace there may face a traceable foe, which it copies at the next MoveEnd, and a
    # copied accuracy ability keeps the hit whole (outcome.ability_whole). At a turn start the
    # invariant says no such holder exists: no copy, the roll keeps its classes.
    s = roark_state(tables, names, player_species="Sandshrew")
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.TRACE).at[1, 0].set(AbilityId.COMPOUNDEYES)
    )
    acts = A("Scratch", "Tackle", s, names)
    eyes = (AbilityId.COMPOUNDEYES,)
    for field in ("learn_site", "pivot_battler"):
        paused = s._replace(**{field: jnp.asarray(1, getattr(s, field).dtype)})
        assert np.asarray(trace_copies(tables, paused, acts, eyes)).tolist() == [True, False]
        assert np.asarray(ability_whole(tables, paused, acts, None)).all()
    assert not np.asarray(trace_copies(tables, s, acts)).any()
    assert not np.asarray(ability_whole(tables, s, acts, None)).any()


def test_only_an_accuracy_copy_keeps_the_hit_whole(tables, names):
    # A state paused at a learn prompt (``learn_site``) follows no ShowBattleMon: the foe's armed
    # Trace can copy the player's active as it stands. Super Luck changes no accuracy check
    # (bcp.c:2876-2947), Compoundeyes does.
    for ability, whole in ((AbilityId.SUPER_LUCK, False), (AbilityId.COMPOUNDEYES, True)):
        s = _standoff(tables, names, ability)
        s = s._replace(learn_site=jnp.asarray(1, s.learn_site.dtype))
        s = s._replace(ability=s.ability.at[0, 0].set(ability))
        acts = A("Tackle", "Tackle", s, names)
        assert bool(trace_copies(tables, s, acts)[1])
        assert np.asarray(ability_whole(tables, s, acts, None)).tolist() == [whole, whole]


def test_accuracy_abilities_hold_every_ability_the_threshold_code_reads():
    # ability_whole keeps a hit whole only for a copy in ACCURACY_ABILITIES: every ability the
    # threshold code tests (outcome._rate, the turn-start and live thresholds; stage_total, the
    # stage sum) must be there. The weather setters and Cloud Nine / Air Lock reach the check
    # through the weather, Klutz through state.held_item; both are in the tuple too.
    read = set()
    for fn in (outcome_mod._rate, outcome_mod.stage_total):
        read |= set(re.findall(r"AbilityId\.([A-Z_0-9]+)", inspect.getsource(fn)))
    read.discard("NONE")
    assert read, "the threshold code reads no ability: the test lost its sources"
    assert read <= {a.name for a in ACCURACY_ABILITIES}, read - {a.name for a in ACCURACY_ABILITIES}


def test_a_super_luck_newcomer_is_copied_in_the_view_not_by_the_trace_columns(tables, names):
    # At B = 2 the one opposing newcomer is copied at its own switch-in, before any move: the
    # laws' view (outcome.pivot_view, G-25) holds the copy, so the foe's crit odds are Super
    # Luck's (stage 1: 1 in 8, battle_lib.c:7097, :7126) and no trace column is needed.
    laws = _laws_of(tables, _standoff(tables, names), _SWITCH)
    assert int(laws.crit_variants[1, 0]) == 8
    assert not laws.crit_possible[:, CRIT_WRITE_VARIANTS:].any()


@pytest.mark.slow  # about 70 s, 8.6 GB peak locally
def test_a_standoff_newcomer_board_steps_alike_whatever_it_leaves_undrawn(tables, names):
    # The mask check of the newcomer board: every element the laws leave undrawn takes another
    # value (every contact roll firing, and every other field off its empty value).
    s = _standoff(tables, names)
    empty = empty_outcome()
    ones = Outcome(*(np.ones_like(np.asarray(f)) for f in empty))
    procs = empty._replace(contact_proc=jnp.ones_like(empty.contact_proc))
    for arbitrary in (procs, ones):
        _assert_mask_sound(tables, s, _SWITCH, arbitrary, 13498)


@pytest.mark.slow  # about 96 s locally (the B = 4 laws' compile)
def test_b4_an_armed_trace_reads_the_contact_roll_of_a_skill_swapped_static(tables, names):
    # DOUBLES: battler 1's Trace is armed (battlers 0 and 2 hold Trace, untraceable, announced).
    # Battler 2's Skill Swap takes battler 3's Static (subscript_exchange_abilities.s:21-24);
    # at its MoveEnd battler 1 can copy it (bcp.c:3973; ChooseTraceTarget's coin between 0 and
    # 2, battle_lib.c:7754-7767), so battler 0's later Tackle on battler 1 reads the contact roll.
    s = four(tables, names, DOUBLES)
    for b, move in enumerate(("Tackle", "Tackle", "Skill Swap", "Tackle")):
        s = with_moves(s, names, b, (move,))
    ability = s.ability
    for b in (0, 1, 2):
        ability = ability.at[party_of(s, b), int(s.active[b])].set(AbilityId.TRACE)
    ability = ability.at[party_of(s, 3), int(s.active[3])].set(AbilityId.STATIC)
    announced = s.trace_announced.at[0].set(1).at[1].set(0).at[2].set(1)
    s = s._replace(ability=ability, trace_announced=announced)
    assert np.asarray(settle_traces(tables, s).trace_announced).tolist() == [1, 0, 1, 0]
    acts = np.zeros(4, np.int32)
    assert bool(trace_copies(tables, s, acts, (AbilityId.STATIC,))[1])
    assert bool(_laws_of(tables, s, acts).relevant.contact_proc[1, 0])
    # Control: battler 2 Tackles: nothing changes, nothing is copied or read.
    calm = with_moves(s, names, 2, ("Tackle",))
    assert not np.asarray(trace_copies(tables, calm, acts)).any()
    assert not bool(_laws_of(tables, calm, acts).relevant.contact_proc[1, 0])


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


def _plain_four(tables, names, moves):
    """DOUBLES, every party member with Overgrow, battler ``b`` knowing ``moves[b]``."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, (moves[b],))
    return s._replace(ability=jnp.full_like(s.ability, AbilityId.OVERGROW))


def _set_active(s, b, ability):
    return s._replace(ability=s.ability.at[party_of(s, b), int(s.active[b])].set(ability))


def test_b4_a_battler_waiting_for_show_battle_mon_traces_the_actives_as_they_stand(tables, names):
    # A four-battler start (state.py initial_arrays) or a replacement round (step.replace_round)
    # sets entry_pending until ShowBattleMon: the switch-in states have not run, so battler 1's
    # unannounced Trace fires at the first MoveEnd (hooks.move_end_switch_in; bcp.c:3973,
    # battle_lib.c:3718-3727) on battler 0's Static as it stands. With no battler waiting the
    # state is a turn start, where the invariant says no such holder exists: no copy counted.
    s = _plain_four(tables, names, ("Tackle",) * 4)
    s = _set_active(_set_active(s, 1, AbilityId.TRACE), 0, AbilityId.STATIC)
    s = s._replace(trace_announced=jnp.zeros_like(s.trace_announced))
    acts = np.zeros(4, np.int32)
    static = (AbilityId.STATIC,)
    assert np.asarray(s.entry_pending).all()  # four(): the start state, before ShowBattleMon
    assert np.asarray(trace_copies(tables, s, acts, static)).tolist() == [False, True, False, False]
    # Only battler 0 waits (a replacement round's newcomer): battler 1 meets it as a new target.
    one = s._replace(entry_pending=jnp.zeros_like(s.entry_pending).at[0].set(1))
    assert bool(trace_copies(tables, one, acts, static)[1])
    shown = s._replace(entry_pending=jnp.zeros_like(s.entry_pending))
    assert not np.asarray(trace_copies(tables, shown, acts, static)).any()


@pytest.mark.slow  # the B = 4 laws' compile, about 96 s locally
def test_b4_a_skill_swap_of_a_trace_foe_copies_a_third_battlers_static(tables, names):
    # Battler 0 Skill Swaps with battler 1, whose Trace has announced: the swap tests no Trace and
    # leaves traceAnnounced (subscript_exchange_abilities.s:4-24), so battler 0 holds Trace with
    # its flag clear and traces at the move's end (bcp.c:3672-3673, :3973); battler 3's Static
    # is a target (ChooseTraceTarget, battle_lib.c:7750-7781). Battler 1's later Tackle on
    # battler 0 then reads the contact roll (battle_lib.c:4213-4222).
    s = _plain_four(tables, names, ("Skill Swap", "Tackle", "Tackle", "Tackle"))
    s = _set_active(_set_active(s, 1, AbilityId.TRACE), 3, AbilityId.STATIC)
    s = s._replace(trace_announced=s.trace_announced.at[1].set(1))
    acts = np.zeros(4, np.int32)
    src = trace_sources(tables, s, acts)
    assert np.asarray(src.armed).tolist() == [True, False, False, False]
    assert bool(trace_copies(tables, s, acts, (AbilityId.STATIC,), src=src)[0])
    # The laws read it (``gains``: the gainer's copy, and at B = 4 the Skill Swap's ``handed``).
    laws = _laws_of(tables, s, acts)
    assert bool(laws.relevant.contact_proc[0, 0])
    # Control: no Trace to take, and nobody else holds Static: nothing to read.
    calm = _plain_four(tables, names, ("Tackle",) * 4)
    calm = _set_active(calm, 1, AbilityId.TRACE)
    calm = calm._replace(trace_announced=calm.trace_announced.at[1].set(1))
    assert not np.asarray(trace_sources(tables, calm, acts).armed).any()
    assert not bool(_laws_of(tables, calm, acts).relevant.contact_proc[0, 0])


def test_b4_a_called_roar_drags_in_a_trace_and_a_static_source(tables, names):
    # At B = 4 an invoker's Roar (Metronome: sCannotMetronomeMoves, battle_lib.c:7152-7179, lacks
    # Roar and Whirlwind) runs at priority 0, and in a plain trainer double the drag goes through
    # (subscript_force_target_to_switch_or_flee.s:8 fails only on DOUBLES|2vs2|AI; the newcomer
    # enters at :20-31). A bench Trace Pokemon can come in and trace at that MoveEnd (bcp.c:3973),
    # and a bench Static Pokemon can come in as a target. ``moves`` holds the called move.
    s = _plain_four(tables, names, ("Metronome", "Tackle", "Tackle", "Tackle"))
    bench_trace, bench_static = 3, 4  # DOUBLES: slots 0 and 1 of each party lead
    s = s._replace(
        ability=s.ability.at[party_of(s, 1), bench_trace]
        .set(AbilityId.TRACE)
        .at[party_of(s, 1), bench_static]
        .set(AbilityId.STATIC)
    )
    acts = np.zeros(4, np.int32)
    tackle = names.move("Tackle")
    called = np.array([names.move("Roar"), tackle, tackle, tackle], np.int32)
    src = trace_sources(tables, s, acts, called)
    assert np.asarray(src.armed).all()
    assert np.asarray(trace_copies(tables, s, acts, (AbilityId.STATIC,), src=src)).all()
    assert bool(doubles_ability_change(tables, s, acts, called, src))
    # Control: Metronome calls Tackle: no drag, nothing armed.
    plain = np.full(4, tackle, np.int32)
    src = trace_sources(tables, s, acts, plain)
    assert not np.asarray(src.armed).any()
    assert not bool(doubles_ability_change(tables, s, acts, plain, src))
