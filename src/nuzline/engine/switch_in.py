"""``BattleSystem_TriggerEffectOnSwitch`` over four battlers (D3-13; battle_lib.c:3649-4174).

The decomp runs the switch-in checks as a state machine (``switchInCheckState``) over one
``monSpeedOrder``: each state loops ``i = 0 .. maxBattlers - 1`` over ``battler = monSpeedOrder[i]``,
fires for the first battler that qualifies, and is entered again from ``i = 0`` until none does;
the per-battler "announced" flags stop a repeat (:3720, :3748-3786, :3799-3803, :3822-3838).
Nothing inside it sorts: one order serves the whole run. ``hooks.py`` has the per-battler
functions (each takes the battler as an argument); this module composes them over the order and
copies none of their logic. At B = 2 ``hooks.switch_in_effects`` / ``hooks.show_battle_mon`` stay
the entry points (the ``war_swaps`` deferral, ruling G-26).

* ``run_states``: the states over a given order, for the battlers whose flags are clear (``fresh``:
  newcomers; a battler that traced in the Trace state joins them: it acts at once). Called by
  ``step`` for a mid-action newcomer (a drag, a switch action, a resumed U-turn / Baton Pass,
  Transform) with the running action's order, and by ``show_battle_mon4``.
* ``show_battle_mon4``: ``ShowBattleMon``'s TrainerMessage sort (bcp.c:251; ``order.sort_mon_speed``
  on ``qc_roll`` and the ``ties`` it reads), then ``run_states`` for the battlers whose
  ``entry_pending`` is set (a replacement's newcomer, or a lead of a four-battler start state).
  The turn-start sort's ``ties`` are an input, not an ``Outcome`` field: the callers draw them
  (``step.sample_root`` at the battle start; D6-D7 / D11 later).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine import field as fcm
from nuzline.engine import targets
from nuzline.engine.format import format_of, party
from nuzline.engine.hooks import (
    SWITCH_IN,
    after_damage_items,
    download,
    forecast,
    intimidate,
    recover_status_by_ability,
    reveal_ability,
    trace,
    trace_coin,
    weather_ability,
)
from nuzline.engine.order import record_speeds, sort_mon_speed
from nuzline.engine.state import STARF_UNDRAWN, BattleState

_I32 = jnp.int32

# lib.c:3715-3737: the Trace loop restarts at i = 0 after each fire, so a holder that found no
# target at its turn (every foe held Trace) tries again once another holder's Trace gave a foe an
# ability to copy. A fire needs a target without Trace, so of four battlers at most three fire.
TRACE_ROUNDS = 3


def _trace_state(tables: Tables, state: BattleState, order, trace_pick):
    """``(state, traced bool[B])``: ``SWITCH_IN_CHECK_STATE_TRACE`` (lib.c:3712-3737). Each round
    finds the first position of ``order`` whose battler has not announced and has a target
    (``targets.trace_target``: HP, Trace, no Griseous Orb; ``ChooseTraceTarget``, :7750-7781, with
    the battler's coin ``hooks.trace_coin``) and fires it, as the decomp's restart does; a later
    holder can copy an ability an earlier one just traced."""
    fmt = format_of(state)
    battlers = jnp.arange(fmt.B)
    traced = jnp.zeros(fmt.B, bool)
    for _ in range(TRACE_ROUNDS):
        ready = []
        for k in range(fmt.B):
            b = order[k]
            coin, _ = trace_coin(state, trace_pick, b)
            _, found = targets.trace_target(tables, state, b, coin)
            ready.append((state.trace_announced[b] == 0) & found)
        ready = jnp.stack(ready)
        first = order[jnp.argmax(ready)]
        coin, counted = trace_coin(state, trace_pick, first)
        state, did = trace(tables, state, first, coin, jnp.any(ready), count=counted)
        traced = traced | ((battlers == first) & did)
    return state, traced


def run_states(
    tables: Tables, state: BattleState, order, fresh, sub_hit=None, trace_pick=None
) -> BattleState:
    """``BattleSystem_TriggerEffectOnSwitch``'s states (battle_lib.c:3712-4165) at B = 4 over one
    ``monSpeedOrder``: Trace, the weather abilities, Intimidate, Download, the switch-in
    announcements (Mold Breaker, Pressure), the Castform form change, the forbidden-status check
    and the held-item check, each a static loop over the positions ``k`` of ``order``, battler
    ``order[k]``. ``order``: ``int32[4]``. ``fresh``: ``bool[4]``, the battlers whose announced flags
    are clear (a newcomer since its entry); a battler that traced acts in the later states too
    (its flags were clear). ``sub_hit``: ``bool[4]`` or None, each battler's
    ``SELF_TURN_FLAG_SUBSTITUTE_HIT`` of the running action (``hooks.intimidate``: the resumed
    U-turn's, ``ts_sub_hit``; a hit of the running action). ``trace_pick``: ``hooks.trace_coin``'s.

    Weather (rule 4): the weather left is the last holder's in ``order``; Trace (rule 5): a later
    holder copies what an earlier one traced. Intimidate, Download, the announcements, the status
    check and the item check read and write only per-battler things (stat drops on targets
    commute), so their order is state-free. ``entry_pending`` is cleared for ``fresh``."""
    fmt = format_of(state)
    if fmt.B != 4:
        raise ValueError("run_states is the B = 4 entry point; B = 2 keeps hooks.switch_in_effects")
    order = jnp.asarray(order, _I32)
    fresh = jnp.asarray(fresh, bool)
    state, traced = _trace_state(tables, state, order, trace_pick)
    act = fresh | traced
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES (:3740-3793)
        state = weather_ability(state, order[k], act[order[k]])
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_INTIMIDATE (:3795-3813)
        state = intimidate(state, order[k], act[order[k]], sub_hit=sub_hit)
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_DOWNLOAD (:3815-3859), after every Intimidate
        b = order[k]
        state = download(state, b, targets.opposing(fmt, b), act[b])
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_MOLD_BREAKER / _PRESSURE (:4075-4110)
        b = order[k]
        mon = state.active[b]
        alive = state.hp[party(fmt, b), mon] > 0
        state = reveal_ability(state, b, mon, SWITCH_IN, act[b] & alive)
    state = forecast(tables, state)  # SWITCH_IN_CHECK_STATE_FORM_CHANGE (:4115-4121), F-6
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS (:4135-4149); arms Unburden
        state = recover_status_by_ability(state, order[k])
    for k in range(fmt.B):  # SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS (:4151-4165)
        state = after_damage_items(tables, state, order[k])
    pending = jnp.where(fresh, 0, state.entry_pending).astype(state.entry_pending.dtype)
    return state._replace(entry_pending=pending)


def show_battle_mon4(
    tables: Tables, state: BattleState, ties, trace_pick=None, battle_start=False
) -> BattleState:
    """``ShowBattleMon``'s ``BattleSystem_TriggerEffectOnSwitch`` (bcp.c:254-267) at B = 4: the
    TrainerMessage sort before it (bcp.c:251; ``sort_mon_speed`` on the state, its ``qc_roll`` and
    ``ties``, ``int[6]``, the bits of the position pairs it reads: ``Outcome.sort_tie`` has no row
    for it), then ``run_states`` with ``fresh = entry_pending != 0`` and no ``sub_hit``.

    ``battle_start`` (a bool, traceable): the battle's first ShowBattleMon. The field weather is not
    set yet for the sort (``SWITCH_IN_CHECK_STATE_FIELD_WEATHER`` sets the start condition inside
    ``TriggerEffectOnSwitch``, lib.c:3662-3710; ``hooks.battle_start``), ``BattleSystem_InitBattleMon``
    has armed ``canUnburden`` for every lead holding an item (bcp.c:227, lib.c:161-166), and a Starf
    Berry eaten by a lead keeps its stat open for the root's draw (``starf_open``, F-10)."""
    fmt = format_of(state)
    if fmt.B != 4:
        raise ValueError(
            "show_battle_mon4 is the B = 4 entry point; B = 2 keeps hooks.show_battle_mon"
        )
    start = jnp.asarray(battle_start, bool)
    held = state.item[party(fmt, jnp.arange(fmt.B)), state.active] != 0
    state = state._replace(
        can_unburden=jnp.where(start & held, 1, state.can_unburden).astype(
            state.can_unburden.dtype
        ),
        starf_roll=jnp.where(start, STARF_UNDRAWN, state.starf_roll).astype(state.starf_roll.dtype),
    )
    # The start condition is a field-mask weather bit (``field``); the start's sort sees none.
    bare = fcm.without_weather(state).field_mask
    sorted_on = state._replace(field_mask=jnp.where(start, bare, state.field_mask))
    order, _ = sort_mon_speed(tables, sorted_on, state.qc_roll, jnp.asarray(ties))
    state = run_states(tables, state, order, state.entry_pending != 0, None, trace_pick)
    state = state._replace(
        starf_roll=jnp.where(start, 0, state.starf_roll).astype(state.starf_roll.dtype)
    )
    # ShowBattleMon's SortMonSpeedOrder after the switch-in states (bcp.c:263) writes
    # monSpeedValues (``order.record_speeds``, E3-B2-15; ``hooks.battle_start`` at B = 2).
    return record_speeds(tables, state)
