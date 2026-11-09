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
  step.py:352-353), or by Disable (``ACT_DISABLED``, the move: "MOVE is disabled!",
  ``subscript_move_is_disabled.s``; no PP spent). A move that never ran (the battler fainted first, slept, froze,
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
* The status checks before a move (step.py:327-377, bcp.c:2417-2728): ``asleep``
  (stays asleep), ``woke``, ``frozen`` (stays frozen), ``thawed`` (its own thaw roll or a
  thawing move), ``flinched``, ``confused`` ("is confused"), ``snapped_out``,
  ``self_hit`` (hurt itself in confusion), ``full_para`` (fully paralyzed).
* ``thawed_by_hit``: a Fire hit thawed it (effects.py:368-375); ``contact_ability``:
  its Static / Poison Point / Flame Body gave its attacker a status (effects.py:343-365), the
  message that names the ability.

The end-of-turn order (``end_speed_tie``) is not in the vector: it shows only in the
order of residual messages, so states it separates stay in one set (coarser than the
game, never finer). Likewise a crit into a substitute: the game prints "A critical hit!"
whenever a landed hit's ``criticalMul != 1`` (bcp.c:4547-4554, whatever took the
damage), but ``crit`` is gated on the target's HP dropping (``step._hit_events``), so
such a crit is not recorded. No implemented move sets ``substitute_hp`` (it is only
reset, ``_VOLATILE`` step.py:94; notes/stage-b/observe.md section 1), so the case cannot
arise yet; were it to, its states would stay in one set (coarser, never finer).

The visible projection of ``after``: both actives, every slot's species, status and
held/consumed item (a Rindo / Chilan Berry's "weakened" message,
``subscript_type_resist_berry.s:_245``, is always followed by its ``RemoveItem``, so
``item_consumed`` shows it); the non-zero-ness of ``confusion``, ``taunt``, ``trapped``,
``leech_seed`` and ``substitute_hp`` (their counters are hidden); stat stages, Water
Sport, hazards (Stealth Rock, Spikes layers), screens and the other side conditions
(non-zero), weather and the field conditions; the player's exact HP and PP per party
slot (the player's healthbox and move menu); the opponent's HP as HP-bar pixels per
slot (``hp_pixels``: the enemy healthbox never shows numbers, ``HEALTHBOX_INFO_NOT_ON_ENEMY``,
include/battle/healthbox.h:33, healthbox.c:633-640); ``over`` and ``winner``.

Never in the vector: the opponent's exact HP, ``status_counter``, any counter value,
the ``ai_*`` fields, an outcome element without a visible effect. Fields constant over
a battle (levels, stats, moves) are left out. Intermediate HP-bar positions during
the turn are seen in game but left out (notes/stage-b/observe.md §3, coarser only).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import MOVES, SIDES, SLOTS, BattleState
from nuzline.engine.step import (
    ACT_DISABLED,
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
)
_NONZERO_FIELD = ("trick_room", "gravity")

_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("order", ()),
    ("pursuit", ()),
    ("act_kind", _SIDE),
    ("act_id", _SIDE),
    ("crit", (SIDES, MAX_HITS)),
    ("magnitude", _SIDE),
    *((name, _SIDE) for name in EVENT_FLAGS),
    ("active", _SIDE),
    ("species", _SLOT),
    ("status", _SLOT),
    ("item", _SLOT),
    ("item_consumed", _SLOT),
    ("player_hp", (SLOTS,)),
    ("opp_pixels", (SLOTS,)),
    ("player_pp", (SLOTS, MOVES)),
    ("stages", (SIDES, 8)),
    ("disabled_move", _SIDE),
    ("locked_move", _SIDE),
    *((name, _SIDE) for name in _NONZERO_SIDE),
    ("spikes", _SIDE),
    ("toxic_spikes", _SIDE),
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
        "species": after.species,
        "status": after.status,
        "item": after.item,
        "item_consumed": after.item_consumed,
        "player_hp": after.hp[0],
        "opp_pixels": hp_pixels(after.hp[1], after.max_hp[1]),
        "player_pp": after.pp[0],
        "stages": after.stages,
        "disabled_move": after.disabled_move,
        "locked_move": after.locked_move,
        **{name: _nonzero(getattr(after, name)) for name in _NONZERO_SIDE},
        "spikes": after.spikes,
        "toxic_spikes": after.toxic_spikes,
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
    events = {**_empty_events(), "order": jnp.full((2,), -1, _I32), "pursuit": jnp.int32(0)}
    return observation(events, after)
