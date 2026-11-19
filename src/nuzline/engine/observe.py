"""What the player sees of one turn: ``observe(tables, before, actions, outcome, after)``
(Stage B spec §2; plan ruling 1). Pure ``jnp``, fixed shapes, jit/vmap-able.

Two children of a belief node with equal observations are one information set: the
player's next decision is one action for all of them. The observation is an
``int32[OBS_SIZE]`` vector, laid out by ``OBS_FIELDS``; ``after`` is the state after the
opponent's forced replacement (``search.playout.send_replacements``), since the
replacement is seen. A player replacement runs no action: ``replacement_observation``
is the same vector with every event at its no-action value (faint-budget spec §4).

**The turn's events** are recorded by ``step`` itself: ``step.turn_events`` replays the
action phase (``_use_move``'s status checks and hits, ``execute_actions``' order, items,
Pursuit and faint checks) through the same code ``step`` runs, so the two cannot drift;
``step.step_events`` returns them with the stepped state, and ``observation`` projects
them without playing the actions again. ``after`` alone loses the second actor's state
at its turn (a status inflicted earlier in the turn, a flinch, whether it fainted
first), so the events cannot be read off it (notes/stage-b/observe.md §2). Every
outcome-derived flag is **effect-gated**: it is
set only when the draw had a visible effect, judged on the battler's state when it acts
(full paralysis only for a battler paralyzed at its turn, a crit only on a hit that
landed and dealt damage). A draw with no effect never changes the observation.

Events, per side (``[side]``; 0 = player):

* ``act_kind``/``act_id``: the action that visibly ran: a move used (``ACT_MOVE``, its
  id, Struggle's when struggling; the PP delta of critic.md B.2 names the same slot), a
  switch (``ACT_SWITCH``, the party slot), a bag item (``ACT_ITEM``, its item id), a
  move refused by Taunt (``ACT_TAUNTED``, the move: "can't use MOVE after the taunt",
  step.py:352-353), by Disable (``ACT_DISABLED``, the move: "MOVE is disabled!",
  ``subscript_move_is_disabled.s``; no PP spent), or by a foe's Imprison (``ACT_IMPRISONED``, the
  move: "can't use the sealed MOVE!", ``subscript_move_is_imprisoned.s``; no PP spent), or by
  Gravity (``ACT_GRAVITY``, the move: "can't use MOVE because of gravity!",
  ``subscript_move_fail_gravity.s``). A move that never ran (the battler fainted first, slept, froze,
  flinched, was fully paralyzed or hurt itself) is not shown (``ACT_NONE``, id 0).
* ``order``: the side that acted first when both showed a message at their turn
  (``action_order``, order.py:67-85), else -1; ``pursuit``: Pursuit hit the switching
  Pokemon first (step.py:545-590).
* ``missed`` (accuracy check failed, step.py:398; not for a Fake Out that failed, whose
  "But it failed!" subscript_missed.s prints instead), ``no_target`` (the foe fainted
  earlier this turn, step.py:390), ``hits`` (hits that landed: "Hit N times",
  battle_controller_player.c:3777-3812), ``crit[side, i]`` (hit ``i`` landed, dealt
  damage and was a crit: ``subscript_critical_hit.s``, bcp.c:4547-4554).
* ``magnitude``: the level "Magnitude N!" showed (4-10) when the side's Magnitude hit
  and dealt damage, else 0 (``subscript_magnitude.s``: the message is a primary effect
  that an ineffective hit does not run).
* ``protected``: the side's move was stopped by the foe's Protect / Detect ("{foe}
  protected itself!", subscript_missed.s:63; also on an accuracy miss, which it replaces;
  not for a failed Fake Out, which prints "But it failed!"). ``protect_ok`` /
  ``protect_failed``: the side's own Protect / Detect succeeded ("{user} protected
  itself!", subscript_protect.s) or failed ("But it failed!" after "X used Protect!").
* ``present_heal``: Present's heal path ran on a hit: 1 "regained health", 2 "HP is full!"
  (``subscript_recover_hp.s``), else 0. Its damage path's power is never printed (only the
  HP change shows it).
* ``called``: the move Nature Power turned into ("Nature Power turned into MOVE!",
  effect_script_0173.s), once its script ran, else 0; likewise the move Sleep Talk,
  Metronome or Copycat called (its own attack message after the invoker's, GoToMoveScript),
  0 when the call failed. ``act_id`` stays the chosen move; the called move's hits, misses and
  crits are the side's.
* ``dragged``: the party slot + 1 the foe's Roar / Whirlwind dragged in on this side ("{0} was
  dragged out!", subscript_force_target_to_switch_or_flee.s), else 0: a drag told apart from
  a switch. The newcomer runs no action this turn. A Roar that fails shows only its
  ``act_id`` ("But it failed!"; a miss or a Protect shows ``missed`` / ``protected``).
* The status checks before a move (step.py:327-377, bcp.c:2417-2728): ``asleep``
  (stays asleep), ``woke``, ``frozen`` (stays frozen), ``thawed`` (its own thaw roll or a
  thawing move), ``flinched``, ``confused`` ("is confused"), ``snapped_out``,
  ``self_hit`` (hurt itself in confusion), ``full_para`` (fully paralyzed), ``in_love``
  ("X is in love with Y!", subscript_infatuated.s: the attract check ran) and ``immobilized``
  ("X is immobilized by love!", subscript_immobilized_by_love.s).
* ``quick_claw``: its Quick Claw's held-item animation played before its move (no text;
  ``subscript_check_quick_claw.s``): both chose a move, its draw of the turn before
  activated the claw (``state.qc_roll``) and it acts first (``step._quick_claw_shown``). The
  turn's own draw (``Outcome.quick_claw``) shows only in the end-of-turn order, which is
  not in the vector (below).
* ``thawed_by_hit``: a Fire hit thawed it (effects.py:368-375); ``contact_ability``:
  its Static / Poison Point / Flame Body gave its attacker a status, or its Cute Charm
  infatuated it (effects._contact_ability), the message that names the ability.

The end-of-turn order (``end_speed_tie``) is not in the vector: it shows only in the
order of residual messages, so states it separates stay in one set (coarser than the
game, never finer). Likewise a crit into a substitute: the game prints "A critical hit!"
whenever a landed hit's ``criticalMul != 1`` (bcp.c:4547-4554, whatever took the
damage), but ``crit`` is gated on the target's HP dropping (``step._hit_events``), so
such a crit is not recorded: states that differ only there stay in one set (coarser, never
finer). Substitute (E2 batch 10, B10-4) shows only whether one is up ("{0} made a
substitute!", "The substitute took damage for {0}!", "{0}'s substitute faded!",
subscript_try_substitute.s, subscript_hit_substitute.s:12, :23); its HP is hidden.

The visible projection of ``after``: both actives and their current types (Camouflage's
"X transformed into the TYPE type!", effect_script_0213.s), every slot's species, status and
held/consumed item (a Rindo / Chilan Berry's "weakened" message,
``subscript_type_resist_berry.s:_245``, is always followed by its ``RemoveItem``, so
``item_consumed`` shows it); the non-zero-ness of ``confusion``, ``taunt``, ``trapped``,
``leech_seed`` and ``substitute_hp`` (their counters are hidden), of ``infatuated`` ("fell in
love", subscript_infatuate.s) and of ``uproar`` ("caused an uproar!", subscript_uproar.s); stat
stages, Water
Sport, hazards (Stealth Rock, Spikes layers), screens and the other side conditions
(non-zero), weather and the field conditions; the player's exact HP and PP per party
slot (the player's healthbox and move menu); the opponent's HP as HP-bar pixels per
slot (``hp_pixels``: the enemy healthbox never shows numbers, ``HEALTHBOX_INFO_NOT_ON_ENEMY``,
include/battle/healthbox.h:33, healthbox.c:633-640); ``over`` and ``winner``.

The player's own party as its summary and the exp messages show it (K3d-exp design §3.5): per
slot its ``level``, ``stats``, ``max_hp`` (it depends on the EVs a payout adds, so two
histories can differ in it alone; the healthbox shows it), ``moves``, ``exp`` ("gained N Exp.
Points!", battle_script.c:9738) and the learn prompt it is shown (``learn_level`` and
``learn_cursor``: the level-up move the game asks about, ``engine.levelup``). On a state
``levelup.advance_learns`` has resolved, learn work in a battle that goes on is always a
prompt (the scan clears a level or stops at a full move list). At the battle's end it may be
unscanned work instead (ruling D8 keeps the work of a battle-ending KO, which no scan reaches),
so the two learn fields read 0 once ``over`` is set: no prompt is shown, and the player has no
decision left. The caller observes a resolved state (Stage B's learn stop, T8).

Never in the vector: the opponent's exact HP, ``status_counter``, any counter value,
the ``ai_*`` fields, an outcome element without a visible effect. The opponent's levels,
stats and moves are constant over a battle and left out. Intermediate HP-bar positions
during the turn are seen in game but left out (notes/stage-b/observe.md §3, coarser only).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.format import format_of
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import MOVES, SIDES, SLOTS, BattleState
from nuzline.engine.step import (
    ACT_DISABLED,
    ACT_GRAVITY,
    ACT_IMPRISONED,
    ACT_ITEM,
    ACT_MOVE,
    ACT_NONE,
    ACT_SWITCH,
    ACT_TAUNTED,
    EVENT_FLAGS,
    _empty_events,
    turn_events,
)

__all__ = [
    "ACT_DISABLED",
    "ACT_GRAVITY",
    "ACT_IMPRISONED",
    "ACT_ITEM",
    "ACT_MOVE",
    "ACT_NONE",
    "ACT_SWITCH",
    "ACT_TAUNTED",
    "HP_BAR_PIXELS",
    "OBS_FIELDS",
    "OBS_SIZE",
    "hp_pixels",
    "observation",
    "observe",
    "replacement_observation",
]

_I32 = jnp.int32
HP_BAR_PIXELS = 48  # 8 * HEALTHBOX_HP_CELL_COUNT (healthbox.c:61, FillCells :1571)

_SIDE, _SLOT = (SIDES,), (SIDES, SLOTS)
# Non-zero-ness of these per-side fields of ``after``.
_NONZERO_SIDE = (
    "confusion",
    "taunt",
    "trapped",
    "leech_seed",
    "substitute_hp",
    "water_sport",
    "stealth_rock",
    "reflect",
    "light_screen",
    "safeguard",
    "mist",
    "lucky_chant",
    "tailwind",
    "foresight",
    "curled",
    "bide_turns",
    "mud_sport",
    "aqua_ring",
    "infatuated",
    # "{0} caused an uproar!" / "{0} calmed down." (subscript_uproar.s, subscript_uproar_end.s).
    "uproar",
    # "{0} foresaw an attack!" (subscript_future_sight_start.s) until the hit's message.
    "future_timer",
    # E2 batch 7: "{0} planted its roots!" (effect_script_0181.s:8), "{0} can no longer escape!"
    # (subscript_mean_look.s:12), "{0} sealed the opponent's move(s)!" (effect_script_0192.s:7),
    # and a Roost that healed this turn ("regained health", subscript_recover_hp.s; the flag is
    # kept only at a pause or a pending replacement).
    "ingrain",
    "mean_look",
    "imprison",
    "roosting",
    # E2 batch 8: "{0} took aim at {1}!" (subscript_lock_on.s:11) until its count runs out.
    "lock_on",
    # "{0} made {1} drowsy!" (subscript_yawn.s:24) until it falls asleep.
    "yawn",
    # "{0} cut its own HP and laid a curse on {1}!" (subscript_curse_ghost.s:17).
    "curse",
    # E2 batch 9: Minimize's own message (it used the move).
    "minimized",
    # E2 batch 10: "{0} got an encore!" (subscript_encore_start.s) until "{0}'s encore ended!";
    # the count is hidden (B10-9).
    "encore_move",
    # E2 batch 12: "{0} is getting pumped!" (subscript_focus_energy.s:7) and "{0} switched its
    # Attack and Defense!" (subscript_user_swap_atk_and_def.s:10).
    "focus_energy",
    "power_trick",
    # E2 batch 13: "{0} can't use items anymore!" (subscript_embargo_start.s:16) until "{0} can
    # use items again!" (subscript_embargo_end.s:6); the count is hidden (B13-31).
    "embargo",
    # E3 batch 1: "{0} was subjected to torment!" (subscript_torment_start.s) until it leaves.
    "torment",
    # E3 batch 1: "{0}'s ability was suppressed!" (subscript_suppress_target_ability.s:12) until it
    # leaves (Baton Pass keeps it, so the message is not re-shown: the bit stays set).
    "ability_suppressed",
)
_NONZERO_FIELD = ("trick_room", "gravity")

_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("order", ()),
    ("pursuit", ()),
    ("act_kind", _SIDE),
    ("act_id", _SIDE),
    ("crit", (SIDES, MAX_HITS)),
    ("magnitude", _SIDE),
    ("present_heal", _SIDE),
    ("called", _SIDE),
    ("dragged", _SIDE),
    *((name, _SIDE) for name in EVENT_FLAGS),
    ("active", _SIDE),
    ("active_types", (SIDES, 2)),
    ("species", _SLOT),
    ("status", _SLOT),
    ("item", _SLOT),
    ("item_consumed", _SLOT),
    ("player_hp", (SLOTS,)),
    ("opp_pixels", (SLOTS,)),
    ("player_pp", (SLOTS, MOVES)),
    ("player_level", (SLOTS,)),
    ("player_stats", (SLOTS, 5)),
    ("player_max_hp", (SLOTS,)),
    ("player_moves", (SLOTS, MOVES)),
    ("player_exp", (SLOTS,)),
    ("learn_level", (SLOTS,)),
    ("learn_cursor", (SLOTS,)),
    ("stages", (SIDES, 8)),
    ("disabled_move", _SIDE),
    ("locked_move", _SIDE),
    # B10-12: the vanished battler ("flew up high!", subscript_vanish_on_charge_turn.s), its kind.
    ("semi_invulnerable", _SIDE),
    *((name, _SIDE) for name in _NONZERO_SIDE),
    ("spikes", _SIDE),
    ("toxic_spikes", _SIDE),
    ("perish_count", _SIDE),
    ("weather", ()),
    *((name, ()) for name in _NONZERO_FIELD),
    ("over", ()),
    ("winner", ()),
)


def _slices() -> tuple[tuple[str, slice], ...]:
    out, start = [], 0
    for name, shape in _LAYOUT:
        size = 1
        for n in shape:
            size *= n
        out.append((name, slice(start, start + size)))
        start += size
    return tuple(out)


# (name, slice of the observation vector) of every field, in order.
OBS_FIELDS: tuple[tuple[str, slice], ...] = _slices()
OBS_SIZE: int = OBS_FIELDS[-1][1].stop


def hp_pixels(hp, max_hp) -> jnp.ndarray:
    """``int32``: the HP bar's filled pixels at rest, ``0`` for no HP else
    ``max(1, hp * 48 // max_hp)``. The decomp's ``FillCells`` (healthbox.c:1559-1590)
    computes ``*temp * 48 / max`` with ``*temp`` in 1/256 HP when ``max < 48`` (then
    ``>> 8``, the same integer) and draws one pixel for a living Pokemon that rounds to
    0 (:1585-1587); ``App_PixelCount`` (src/unk_0208C098.c:41-49) is the same formula.
    Vectorized; an empty slot (``max_hp`` 0) shows 0."""
    hp = jnp.asarray(hp, _I32)
    max_hp = jnp.asarray(max_hp, _I32)
    pixels = hp * HP_BAR_PIXELS // jnp.maximum(max_hp, 1)
    return jnp.where(hp > 0, jnp.maximum(pixels, 1), 0)


def _nonzero(x) -> jnp.ndarray:
    return (jnp.asarray(x) != 0).astype(_I32)


def observe(
    tables: Tables, before: BattleState, actions, outcome: Outcome, after: BattleState
) -> jnp.ndarray:
    """``int32[OBS_SIZE]``: the player's observation of the turn ``before`` ->
    ``after`` under ``actions`` and ``outcome`` (module docstring; ``OBS_FIELDS`` names
    the slices). ``after`` is ``step``'s result after ``send_replacements``."""
    return observation(turn_events(tables, before, actions, outcome), after)


def observation(events: dict, after: BattleState) -> jnp.ndarray:
    """``observe`` from the turn's events (``step.step_events``, ``step.turn_events``):
    a program that already ran ``step_events`` projects without playing the actions
    again."""
    events = dict(events)
    shown, order = events.pop("shown"), events.pop("order")
    both = (shown[0] != 0) & (shown[1] != 0)
    values = {
        **events,
        "order": jnp.where(both, order[0], -1),
        "active": after.active,
        "active_types": after.types[jnp.arange(SIDES), after.active],
        "species": after.species,
        "status": after.status,
        "item": after.item,
        "item_consumed": after.item_consumed,
        "player_hp": after.hp[0],
        "opp_pixels": hp_pixels(after.hp[1], after.max_hp[1]),
        "player_pp": after.pp[0],
        "player_level": after.level[0],
        "player_stats": after.stats[0],
        "player_max_hp": after.max_hp[0],
        "player_moves": after.moves[0],
        "player_exp": after.exp,
        "learn_level": jnp.where(after.over == 0, after.learn_level, 0),
        "learn_cursor": jnp.where(after.over == 0, after.learn_cursor, 0),
        "stages": after.stages,
        "disabled_move": after.disabled_move,
        "locked_move": after.locked_move,
        "semi_invulnerable": after.semi_invulnerable,
        **{name: _nonzero(getattr(after, name)) for name in _NONZERO_SIDE},
        "spikes": after.spikes,
        "toxic_spikes": after.toxic_spikes,
        # "{0}'s perish count fell to {1}!" (subscript_continue_perish_song.s:3): 1 + the count
        # while the effect is up, else 0.
        "perish_count": jnp.where(after.perish_song != 0, 1 + after.perish_song_turns, 0),
        "weather": after.weather,
        **{name: _nonzero(getattr(after, name)) for name in _NONZERO_FIELD},
        "over": after.over,
        "winner": after.winner,
    }
    parts = []
    for name, shape in _LAYOUT:
        value = jnp.asarray(values[name]).astype(_I32)
        if value.shape != shape:
            raise ValueError(f"observation field {name}: shape {value.shape}, want {shape}")
        parts.append(value.reshape(-1))
    return jnp.concatenate(parts)


def replacement_observation(after: BattleState) -> jnp.ndarray:
    """The observation of a replacement: no action ran, so every event is at its
    no-action value (``ACT_NONE``, flags 0, ``order`` -1, no Pursuit), and the visible
    projection is ``after``'s (both newcomers on screen). Built from the full event set
    ``step_events`` returns, so a new event field cannot be missed."""
    events = {
        **_empty_events(format_of(after)),
        "order": jnp.full((2,), -1, _I32),
        "pursuit": jnp.int32(0),
    }
    return observation(events, after)
