"""What the player sees of one turn: ``observe(tables, before, actions, outcome, after)``
(Stage B spec §2; plan ruling 1). Pure ``jnp``, fixed shapes, jit/vmap-able.

Two children of a belief node with equal observations are one information set: the
player's next decision is one action for all of them. The observation is an
``int32[OBS_SIZE]`` vector, laid out by ``OBS_FIELDS``; ``after`` is the state after the
opponent's forced replacement (``search.playout.send_replacements``), since the
replacement is seen.

**The turn's events** come from a replay of ``step``'s action phase (``_replay``):
``execute_actions`` (step.py:529-562) with the same order, item spending, Natural Cure,
Pursuit hit and faint checks, and ``_use_move``'s status checks and hits
(step.py:238-368) mirrored by ``_move`` so that each check's outcome is known. The
replay is checked against ``execute_actions`` field by field
(tests/engine/test_observe.py). ``step`` keeps no event record and ``after`` loses the
second actor's state at its turn (a status inflicted earlier in the turn, a flinch,
whether it fainted first), so the events cannot be read off ``after`` alone
(notes/stage-b/observe.md §2). Every outcome-derived flag is **effect-gated**: it is
set only when the draw had a visible effect, judged on the battler's state when it acts
(full paralysis only for a battler paralyzed at its turn, a crit only on a hit that
landed and dealt damage). A draw with no effect never changes the observation.

Events, per side (``[side]``; 0 = player):

* ``act_kind``/``act_id``: the action that visibly ran: a move used (``ACT_MOVE``, its
  id, Struggle's when struggling; the PP delta of critic.md B.2 names the same slot), a
  switch (``ACT_SWITCH``, the party slot), a bag item (``ACT_ITEM``, its item id), a
  move refused by Taunt (``ACT_TAUNTED``, the move: "can't use MOVE after the taunt",
  step.py:288). A move that never ran (the battler fainted first, slept, froze,
  flinched, was fully paralyzed or hurt itself) is not shown (``ACT_NONE``, id 0).
* ``order``: the side that acted first when both showed a message at their turn
  (``action_order``, order.py:67-85), else -1; ``pursuit``: Pursuit hit the switching
  Pokemon first (step.py:429-474).
* ``missed`` (accuracy check failed, step.py:329), ``no_target`` (the foe fainted
  earlier this turn, step.py:321), ``hits`` (hits that landed: "Hit N times",
  battle_controller_player.c:3777-3812), ``crit[side, i]`` (hit ``i`` landed, dealt
  damage and was a crit: ``subscript_critical_hit.s``, bcp.c:4547-4554).
* The status checks before a move (step.py:260-308, bcp.c:2417-2728): ``asleep``
  (stays asleep), ``woke``, ``frozen`` (stays frozen), ``thawed`` (its own thaw roll or a
  thawing move), ``flinched``, ``confused`` ("is confused"), ``snapped_out``,
  ``self_hit`` (hurt itself in confusion), ``full_para`` (fully paralyzed).
* ``thawed_by_hit``: a Fire hit thawed it (effects.py:368-375); ``contact_ability``:
  its Static / Poison Point gave its attacker a status (effects.py:343-365), the
  message that names the ability.

The end-of-turn order (``end_speed_tie``) is not in the vector: it shows only in the
order of residual messages, so states it separates stay in one set (coarser than the
game, never finer).

The visible projection of ``after``: both actives, every slot's species, status and
held/consumed item; the non-zero-ness of ``confusion``, ``taunt``, ``trapped``,
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

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import confusion_damage
from nuzline.engine.effects import apply_move_effect, change_stage, defender_always, hit_count
from nuzline.engine.hooks import after_damage_items
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status
from nuzline.engine.order import action_order, is_item, is_switch, selected_move
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import MOVES, SIDES, SLOTS, BattleState
from nuzline.engine.step import (
    _bag_slot,
    _consume_trainer_items,
    _faint_check,
    _natural_cure,
    _pursuit_hit,
    _record_last_move,
    _send_out,
    _set_mon,
    _use_trainer_item,
    _where,
    pursuit_fires,
)

_I32 = jnp.int32
HP_BAR_PIXELS = 48  # 8 * HEALTHBOX_HP_CELL_COUNT (healthbox.c:61, FillCells :1571)

# ``act_kind`` values.
ACT_NONE, ACT_MOVE, ACT_SWITCH, ACT_ITEM, ACT_TAUNTED = 0, 1, 2, 3, 4

_SIDE, _SLOT = (SIDES,), (SIDES, SLOTS)
# Per-side event flags, in vector order.
_FLAGS = (
    "missed",
    "no_target",
    "hits",
    "full_para",
    "flinched",
    "asleep",
    "woke",
    "frozen",
    "thawed",
    "thawed_by_hit",
    "confused",
    "snapped_out",
    "self_hit",
    "contact_ability",
)
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
)
_NONZERO_FIELD = ("trick_room", "gravity", "mud_sport")

_LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("order", ()),
    ("pursuit", ()),
    ("act_kind", _SIDE),
    ("act_id", _SIDE),
    ("crit", (SIDES, MAX_HITS)),
    *((name, _SIDE) for name in _FLAGS),
    ("active", _SIDE),
    ("species", _SLOT),
    ("status", _SLOT),
    ("item", _SLOT),
    ("item_consumed", _SLOT),
    ("player_hp", (SLOTS,)),
    ("opp_pixels", (SLOTS,)),
    ("player_pp", (SLOTS, MOVES)),
    ("stages", (SIDES, 8)),
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


def _put(events: dict, side, values: dict, cond) -> dict:
    """``events[name][side] = values[name]`` where ``cond``."""
    out = dict(events)
    for name, value in values.items():
        arr = events[name]
        new = jnp.where(cond, jnp.asarray(value).astype(arr.dtype), arr[side])
        out[name] = arr.at[side].set(new)
    return out


def _hit_events(before: BattleState, after: BattleState, side, landed, crit) -> dict:
    """What one hit of ``side``'s move showed, from the states around it: damage dealt
    (the target's HP drop: only the hit itself lowers it, effects.py:441-444), a crit
    on a landed hit that dealt damage, the defender's contact ability (the only change
    of the attacker's own status inside a hit, effects.py:343-365) and a Fire hit's
    thaw (the only FREEZE -> NONE change of the target, effects.py:368-375)."""
    foe = 1 - side
    user, target = before.active[side], before.active[foe]
    dealt = before.hp[foe, target] > after.hp[foe, target]
    contact = (before.status[side, user] == Status.NONE) & (after.status[side, user] != Status.NONE)
    thaw = (before.status[foe, target] == Status.FREEZE) & (
        after.status[foe, target] == Status.NONE
    )
    return {"crit": landed & dealt & crit, "contact": contact, "thaw": thaw}


def _move(
    tables: Tables, state: BattleState, side, action, choice, outcome: Outcome, target_moved
) -> tuple[BattleState, dict]:
    """``step._use_move`` (step.py:238-368) with each check's result: the state after
    the move (``last_move`` and ``move_hit`` are not written: nothing later in the
    action phase reads them) and the move's events (module docstring). The comments
    name the decomp checks; step.py cites their lines."""
    move, struggling = choice
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    foe = 1 - side
    target = state.active[foe]
    ready = (state.over == 0) & (state.hp[side, user] > 0)

    # CHECK_STATUS_STATE_SLEEP: still asleep disrupts, waking up goes on.
    status = state.status[side, user]
    asleep = ready & (status == Status.SLEEP)
    sleep_left = jnp.maximum(state.status_counter[side, user].astype(_I32) - 1, 0)
    sleeping = asleep & (sleep_left > 0)
    state = _set_mon(state, "status_counter", side, user, sleep_left, asleep)
    state = _set_mon(state, "status", side, user, Status.NONE, asleep & ~sleeping)
    # CHECK_STATUS_STATE_FREEZE: the thaw roll, or a thawing move goes on frozen.
    thaw_move = (tables.move_effect[move] == EffectId.SECONDARY_STATUS) & (
        tables.move_param[move, 2] == 1
    )
    frozen = ready & (status == Status.FREEZE)
    thawed = frozen & (outcome.thaw[side] != 0)
    still_frozen = frozen & ~thawed
    state = _set_mon(state, "status", side, user, Status.NONE, thawed)
    go = ready & ~sleeping & ~(still_frozen & ~thaw_move)
    # CHECK_STATUS_STATE_FLINCH, with Steadfast's +1 Speed.
    flinched = go & (state.flinched[side] != 0)
    steadfast = flinched & (state.ability[side, user] == AbilityId.STEADFAST)
    state = change_stage(state, side, Stat.SPE, 1, steadfast)
    go = go & ~flinched
    # CHECK_STATUS_STATE_TAUNT.
    taunted = go & (state.taunt[side] > 0) & (tables.move_power[move] == 0)
    go = go & ~taunted
    # CHECK_STATUS_STATE_CONFUSION: the counter drops; above 0 the self-hit roll.
    confused = go & (state.confusion[side] > 0)
    confusion = state.confusion[side] - confused.astype(state.confusion.dtype)
    still_confused = confused & (confusion > 0)
    self_hit = still_confused & (outcome.confusion_hit[side] != 0)
    user_hp = state.hp[side, user].astype(_I32)
    hurt = jnp.minimum(confusion_damage(tables, state, side, outcome.roll[side, 0]), user_hp)
    user_hp = user_hp - jnp.where(self_hit, hurt, 0)
    state = state._replace(
        confusion=state.confusion.at[side].set(confusion),
        hp=state.hp.at[side, user].set(user_hp.astype(state.hp.dtype)),
    )
    go = go & ~self_hit
    # CHECK_STATUS_STATE_PARALYSIS, gated on reaching it (the effect gate).
    full_para = go & (state.status[side, user] == Status.PARALYSIS) & (outcome.full_para[side] != 0)
    can_act = go & ~full_para
    # CHECK_STATUS_STATE_SELF_THAW.
    self_thaw = can_act & still_frozen
    state = _set_mon(state, "status", side, user, Status.NONE, self_thaw)

    spend = can_act & ~struggling
    pp = state.pp.at[side, user, slot].add(jnp.where(spend, -1, 0).astype(state.pp.dtype))
    state = state._replace(pp=pp)
    never_misses = tables.move_accuracy[move] == 0
    has_target = (state.hp[foe, target] > 0) | defender_always(tables, move)
    hit = (outcome.hit[side, 0] != 0) | never_misses
    connected = can_act & has_target & hit
    n_hits = hit_count(tables, move, outcome, side)

    def one_hit(i, carry):
        state, hits, crits, contact, thaw = carry
        landed = (
            connected & (i < n_hits) & ((state.hp[foe, target] > 0) | defender_always(tables, move))
        )
        hit_state = apply_move_effect(
            tables,
            state,
            side,
            move,
            outcome,
            i,
            connected=connected,
            target_moved=target_moved,
            struggling=struggling,
        )
        seen = _hit_events(state, hit_state, side, landed, outcome.crit[side, i] != 0)
        state = after_damage_items(tables, hit_state, side, connected)
        state = after_damage_items(tables, state, foe, connected)
        return (
            state,
            hits + landed.astype(_I32),
            crits.at[i].set(seen["crit"]),
            contact | seen["contact"],
            thaw | seen["thaw"],
        )

    init = (state, _I32(0), jnp.zeros((MAX_HITS,), bool), jnp.bool_(False), jnp.bool_(False))
    state, hits, crits, contact, thaw = jax.lax.fori_loop(0, MAX_HITS, one_hit, init)
    kind = jnp.where(can_act, ACT_MOVE, jnp.where(taunted, ACT_TAUNTED, ACT_NONE))
    events = {
        "act_kind": kind,
        "act_id": jnp.where(can_act | taunted, move, 0),
        "missed": can_act & has_target & ~hit,
        "no_target": can_act & ~has_target,
        "hits": hits,
        "crit": crits,
        "full_para": full_para,
        "flinched": flinched,
        "asleep": sleeping,
        "woke": asleep & ~sleeping,
        "frozen": still_frozen & ~thaw_move,
        "thawed": thawed | self_thaw,
        "confused": still_confused,
        "snapped_out": confused & ~still_confused,
        "self_hit": self_hit,
        "shown": ready,
    }
    return state, {"events": events, "contact": contact, "thaw": thaw}


def _empty_events() -> dict:
    side = jnp.zeros(_SIDE, _I32)
    events = {name: side for name in ("act_kind", "act_id", "shown", *_FLAGS)}
    events["crit"] = jnp.zeros((SIDES, MAX_HITS), _I32)
    return events


def _replay(
    tables: Tables, before: BattleState, actions, outcome: Outcome
) -> tuple[BattleState, dict, jnp.ndarray, jnp.ndarray]:
    """``step``'s action phase (``_record_last_move``, ``action_order``,
    ``execute_actions``: step.py:529-562, 692-697) with the events of each action:
    ``(state after both actions, events, order, pursued)``. The state matches
    ``execute_actions`` in every field but ``last_move`` and ``move_hit``."""
    actions = jnp.asarray(actions, _I32)
    state = _record_last_move(tables, before, actions)
    order = action_order(tables, state, actions, outcome)
    first = order[0].astype(_I32)
    picks = [selected_move(tables, state, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    items = jnp.stack(
        [state.trainer_items[side, _bag_slot(actions[side])] for side in (0, 1)]
    ).astype(_I32)
    state = _consume_trainer_items(state, actions)
    pursued = pursuit_fires(tables, state, first, actions, choices)
    state = _where(
        is_switch(actions[first]) & (state.over == 0), _natural_cure(state, first), state
    )
    events = _empty_events()

    # The Pursuit hit on the switching Pokemon (step.py:451-474): the pursuer's move ran.
    pursuer = 1 - first
    hit_state = _pursuit_hit(tables, state, pursuer, actions, choices, outcome)
    seen = _hit_events(state, hit_state, pursuer, True, outcome.crit[pursuer, 0] != 0)
    events = _put(
        events,
        pursuer,
        {
            "act_kind": ACT_MOVE,
            "act_id": choices[0][pursuer],
            "hits": 1,
            "crit": jnp.zeros((MAX_HITS,), _I32).at[0].set(seen["crit"]),
        },
        pursued,
    )
    events = _put(events, first, {"contact_ability": seen["contact"]}, pursued)
    events = _put(events, first, {"thawed_by_hit": seen["thaw"]}, pursued)
    state = _faint_check(state, _where(pursued, hit_state, state))

    def act(position, carry):
        state, events = carry
        side = order[position].astype(_I32)
        second = position == 1
        action = actions[side]
        live = state.over == 0
        moved, found = _move(
            tables,
            state,
            side,
            action,
            (choices[0][side], choices[1][side]),
            outcome,
            second,
        )
        switch = is_switch(action) & live
        item = is_item(action) & live
        switched = _send_out(tables, state, side, jnp.clip(action - MOVE_ACTIONS, 0, SLOTS - 1))
        used = _use_trainer_item(tables, state, side, items[side])
        acted = _where(switch, switched, _where(item, used, moved))
        skip = second & pursued  # the pursuer does not act again
        acted = _where(skip, state, acted)
        own = found["events"]
        kind = jnp.where(switch, ACT_SWITCH, jnp.where(item, ACT_ITEM, own["act_kind"]))
        act_id = jnp.where(
            switch, action - MOVE_ACTIONS, jnp.where(item, items[side], own["act_id"])
        )
        is_move = ~is_switch(action) & ~is_item(action)
        values = {name: jnp.where(is_move, value, 0) for name, value in own.items()}
        values.update(act_kind=kind, act_id=act_id, shown=switch | item | own["shown"])
        events = _put(events, side, values, ~skip)
        foe = 1 - side
        foe_values = {"contact_ability": found["contact"], "thawed_by_hit": found["thaw"]}
        foe_values = {name: v | (events[name][foe] != 0) for name, v in foe_values.items()}
        events = _put(events, foe, foe_values, ~skip & is_move)
        return _faint_check(state, acted), events

    state, events = jax.lax.fori_loop(0, 2, act, (state, events))
    return state, events, order, pursued


def _nonzero(x) -> jnp.ndarray:
    return (jnp.asarray(x) != 0).astype(_I32)


def observe(
    tables: Tables, before: BattleState, actions, outcome: Outcome, after: BattleState
) -> jnp.ndarray:
    """``int32[OBS_SIZE]``: the player's observation of the turn ``before`` ->
    ``after`` under ``actions`` and ``outcome`` (module docstring; ``OBS_FIELDS`` names
    the slices). ``after`` is ``step``'s result after ``send_replacements``."""
    _, events, order, pursued = _replay(tables, before, actions, outcome)
    shown = events.pop("shown")
    both = (shown[0] != 0) & (shown[1] != 0)
    values = {
        "order": jnp.where(both, order[0], -1),
        "pursuit": pursued,
        **events,
        "active": after.active,
        "species": after.species,
        "status": after.status,
        "item": after.item,
        "item_consumed": after.item_consumed,
        "player_hp": after.hp[0],
        "opp_pixels": hp_pixels(after.hp[1], after.max_hp[1]),
        "player_pp": after.pp[0],
        "stages": after.stages,
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
