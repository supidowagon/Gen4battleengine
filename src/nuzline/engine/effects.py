"""Move effects, one hit at a time (pure ``jnp``, jit/vmap-able, every update masked).

``apply_move_effect`` plays hit ``hit_index`` of a move: its damage, then the
move's effect, dispatched on ``tables.move_effect[move]`` with one function per
``EffectId`` family. ``step`` calls it for every hit index (``hit_count`` says how
many are real). Decomp sources:

* Which side effects run: ``BattleSystem_TriggerPrimaryEffect`` (status and stat
  moves, "direct") and ``BattleSystem_TriggerSecondaryEffect`` (effects of a
  damaging move, "indirect"), ``src/battle/battle_lib.c:1481`` and ``:1519``. A
  direct effect needs a living target and a move that connected (a ``RANGE_USER``
  move's target is its user, so a fainted foe does not stop it); a probabilistic
  indirect one needs its roll, a living recipient and a move that hit and was not
  ineffective; an ``ON_HIT`` one (chance 0 in the tables, e.g. Hammer Arm, drain,
  recoil) only the latter. ``step`` therefore applies chance-0 secondaries without
  reading ``outcome.secondary``.
* Stat stages: ``BtlCmd_ChangeStatStage`` (``src/battle/battle_script.c:2804``):
  clamped to +-6; a drop on the foe is blocked by Keen Eye for accuracy (an
  ignorable ability, so Mold Breaker pierces it). Simple is not applied here: it
  doubles the stage where the stat is used (``damage.simple_stage``).
* Status: ``res/battle/scripts/subscripts/subscript_{fall_asleep,poison,burn,
  freeze,paralyze}.s``: an existing status is never overwritten; Fire types
  cannot be burned, Ice types cannot be frozen, Poison and Steel types cannot be
  poisoned, and nothing is frozen in sun. Sleep lasts ``outcome.duration`` turns
  (``Random 3, 2``). Status moves skip the type chart except those marked in
  ``move_param[m, 1]`` (Thunder Wave, ``BattleControllerPlayer_CheckTypeChart``).
* Flinch: ``subscript_flinch_mon.s``: only a target that has not moved yet this
  turn, not Inner Focus (ignorable). Steadfast's Speed boost is in ``step``.
* Drain / recoil: ``subscript_drain_half_damage_dealt.s``,
  ``subscript_recoil_1_{2,4}.s``: a fraction of ``hitDamage``, the damage dealt
  capped at the target's HP (``BtlCmd_UpdateHealthBarValue``, battle_script.c:1701),
  through ``BattleSystem_Divide`` (at least 1). Rock Head prevents recoil.
* Multi-hit: ``BtlCmd_SetMultiHit`` (battle_script.c:2634) and
  ``BattleControllerPlayer_LoopMultiHit`` (battle_controller_player.c:3777): the
  move script reruns per hit with ``SYSCTL_MULTI_HIT_MOVE``, which skips the
  accuracy check, so only ``outcome.hit[side, 0]`` matters while crit and roll
  are per hit; the loop stops once the target faints.
* ``HIGH_CRIT`` only raises the crit stage ``sample_outcome`` uses; here the
  engine reads ``outcome.crit``.
* Stealth Rock: ``effect_script_0266.s`` sets the flag on the defender's side
  (fails if already set); the switch-in damage is ``stealth_rock_damage``.
* Bind (``TRAP``): ``effect_script_0042.s`` -> ``subscript_bind_start.s``, an
  on-hit indirect effect needing a living target that is not bound yet: the
  counter is ``Random 3, 3`` (``outcome.duration``, 3-6) and ``trap_move`` the
  move. The end-of-turn tick is in ``step``.
* Taunt: ``subscript_taunt_start.s``: fails on a taunted target, else the counter
  is ``Random 2, 3`` (``outcome.duration``, 3-5). The move failure and the
  countdown are in ``step``.
* Confusion (``CONFUSE_FOE``): ``subscript_confuse.s``: fails on a confused
  target, else ``Random 3, 2`` turns (``outcome.duration``, 2-5). The move-time
  check and self-hit are in ``step``.
* Confusion, Psybeam, ... (``SECONDARY_CONFUSE``): ``effect_script_0076.s`` is a
  probabilistic indirect effect: the roll (``outcome.secondary``) is cleared when the
  target fainted (battle_lib.c:1556-1560), then ``subscript_confuse.s`` fails on a
  confused target, else the counter is ``Random 3, 2`` (``outcome.duration``, 2-5).
* Mud Sport: ``effect_script_0201.s`` sets the user's flag (fails if set);
  ``damage.py`` halves Electric power while any active battler has it.
* Bug Bite, Pluck (``EAT_BERRY``): ``effect_script_0224.s``, an ``ON_HIT`` indirect effect
  (no HP check, battle_lib.c:1519-1524) running ``subscript_pluck.s``: ``hooks.pluck_berry``.
* Leech Seed: ``subscript_leech_seed_start.s``: Grass types and already seeded
  targets are immune. The end-of-turn drain is in ``step``.
* Water Sport: ``effect_script_0210.s`` sets the user's flag (fails if set);
  ``damage.py`` halves Fire power while any active battler has it.
* Night Shade, Seismic Toss (``LEVEL_DAMAGE``), Hidden Power, Magnitude: damage and type
  come from ``damage.py`` (``fixed_damage``, ``hidden_power``, ``MAGNITUDE_POWER``); no
  effect function runs for them.
* Foresight, Odor Sleuth: ``effect_script_0113.s`` / ``subscript_foresight.s`` set
  ``foresight`` on the defender whenever the move lands on a living target (no
  failure check). ``damage.py`` skips the Ghost immunities for it and ``outcome.py``
  ignores the defender's positive evasion.
* Pursuit is a plain hit here; its pre-switch hit at double power is in ``step``.
* Static / Poison Point: ``BattleSystem_TriggerAbilityOnHit`` (battle_lib.c:4200),
  run per hit after the secondary effect (``ONE_HIT_TRIGGER_ABILITY`` /
  ``MULTI_HIT_TRIGGER_ABILITY``, battle_controller_player.c:3513, :3573): the
  defender's own ability (Mold Breaker does not matter), a contact move that hit,
  was not ineffective and dealt damage, a living attacker with no status, and the
  roll ``RandNext() % 10 < 3`` (``outcome.contact_proc[defending side, hit]``). The
  defender need not be alive. The status goes through ``subscript_paralyze.s`` /
  ``subscript_poison.s`` as an ability side effect: Safeguard and Substitute do not
  apply, the type immunities do (``can_have_status``). Its message reveals the
  holder's ability to the AI (``hooks.reveal_ability``).
* Thaw on a Fire hit: ``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (battle_controller_player.c:
  3710), after the contact ability: a frozen defender that is alive and took damage
  from a Fire-type move thaws. The user's own thaw (Flame Wheel) is in ``step``.

Unreachable branches are not modelled: Substitute, Mist, Safeguard and Uproar
come from unsupported moves; Shield Dust, Clear Body, Hyper Cutter, Immunity,
Limber, Insomnia, Leaf Guard, Liquid Ooze, Magic Guard, Reckless, Serene Grace,
Skill Link, Own Tempo, Levitate, Normalize and Color Change are unsupported
abilities; Big Root and Grip Claw are unsupported items.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import STATUS, damage_for_roll, effectiveness, move_type_of
from nuzline.engine.ids import (
    AbilityId,
    EffectId,
    MoveFlag,
    MoveTarget,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import STAGE_MAX, STAGE_MIN, BattleState

_I32 = jnp.int32
_E = EffectId


class _Hit(NamedTuple):
    """What one hit's effect functions need to know."""

    side: jnp.ndarray  # attacking side
    target_side: jnp.ndarray
    user: jnp.ndarray  # party slots
    target: jnp.ndarray
    effect: jnp.ndarray  # EffectId
    param: jnp.ndarray  # int32[3], tables.move_param[move]
    certain: jnp.ndarray  # chance 0: an ON_HIT secondary
    landed: jnp.ndarray  # the move connected on this hit
    effective: jnp.ndarray  # ... and was not ineffective (damage > 0, or a passed type gate)
    proc: jnp.ndarray  # the secondary applies on this hit (its roll, or certain)
    dealt: jnp.ndarray  # HP the target lost on this hit
    target_moved: jnp.ndarray  # the target has already acted this turn
    move: jnp.ndarray  # the move's id
    target_ignorable: jnp.ndarray  # target's ability as the attacker sees it (Mold Breaker)
    contact_proc: jnp.ndarray  # the target's contact-ability roll for this hit


def hit_count(tables: Tables, move, outcome: Outcome, side) -> jnp.ndarray:
    """``int32``: hits the move makes if nothing faints: ``outcome.n_hits[side]`` for
    ``MULTI_HIT``, else 1."""
    multi = tables.move_effect[move] == _E.MULTI_HIT
    return jnp.where(multi, outcome.n_hits[side].astype(_I32), 1)


def _alive(state: BattleState, side, slot) -> jnp.ndarray:
    return state.hp[side, slot] > 0


def change_stage(state: BattleState, side, stat, delta, cond) -> BattleState:
    """Add ``delta`` to ``side``'s stage ``stat`` where ``cond``, clamped to +-6."""
    current = state.stages[side, stat].astype(_I32)
    new = jnp.clip(current + delta, STAGE_MIN, STAGE_MAX)
    new = jnp.where(cond, new, current).astype(state.stages.dtype)
    return state._replace(stages=state.stages.at[side, stat].set(new))


def _foe_stage(state: BattleState, h: _Hit, cond) -> BattleState:
    stat, delta = h.param[0], h.param[1]
    keen_eye = (h.target_ignorable == AbilityId.KEEN_EYE) & (stat == Stat.ACC) & (delta < 0)
    return change_stage(state, h.target_side, stat, delta, cond & ~keen_eye)


def _stat_self(tables, state, h: _Hit) -> BattleState:
    return change_stage(
        state, h.side, h.param[0], h.param[1], (h.effect == _E.STAT_SELF) & h.landed
    )


def _stat_foe(tables, state, h: _Hit) -> BattleState:
    cond = (h.effect == _E.STAT_FOE) & h.landed & _alive(state, h.target_side, h.target)
    return _foe_stage(state, h, cond)


def _secondary_stat_self(tables, state, h: _Hit) -> BattleState:
    recipient_ok = h.certain | _alive(state, h.side, h.user)
    cond = (h.effect == _E.SECONDARY_STAT_SELF) & h.effective & h.proc & recipient_ok
    return change_stage(state, h.side, h.param[0], h.param[1], cond)


def _secondary_stat_foe(tables, state, h: _Hit) -> BattleState:
    recipient_ok = h.certain | _alive(state, h.target_side, h.target)
    cond = (h.effect == _E.SECONDARY_STAT_FOE) & h.effective & h.proc & recipient_ok
    return _foe_stage(state, h, cond)


def can_have_status(state: BattleState, side, slot, status) -> jnp.ndarray:
    """No status yet and no type (or weather) immunity to ``status``."""
    types = state.types[side, slot]
    has = lambda t: jnp.any(types == t)
    immune = (
        ((status == Status.BURN) & has(Type.FIRE))
        | (
            ((status == Status.POISON) | (status == Status.TOXIC))
            & (has(Type.POISON) | has(Type.STEEL))
        )
        | ((status == Status.FREEZE) & (has(Type.ICE) | (state.weather == Weather.SUN)))
    )
    return (state.status[side, slot] == Status.NONE) & ~immune


def _inflict(state: BattleState, h: _Hit, outcome: Outcome, cond) -> BattleState:
    status = h.param[0]
    cond = cond & _alive(state, h.target_side, h.target)
    cond = cond & can_have_status(state, h.target_side, h.target, status)
    counter = jnp.where(status == Status.SLEEP, outcome.duration[h.side], 0)
    old_status = state.status[h.target_side, h.target]
    old_counter = state.status_counter[h.target_side, h.target]
    return state._replace(
        status=state.status.at[h.target_side, h.target].set(
            jnp.where(cond, status, old_status).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[h.target_side, h.target].set(
            jnp.where(cond, counter, old_counter).astype(state.status_counter.dtype)
        ),
    )


def _status_foe(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    return _inflict(state, h, outcome, (h.effect == _E.STATUS_FOE) & h.effective)


def _secondary_status(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    return _inflict(state, h, outcome, (h.effect == _E.SECONDARY_STATUS) & h.effective & h.proc)


def _flinch(tables, state, h: _Hit) -> BattleState:
    cond = (
        (h.effect == _E.FLINCH)
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & ~h.target_moved
        & (h.target_ignorable != AbilityId.INNER_FOCUS)
    )
    old = state.flinched[h.target_side]
    return state._replace(
        flinched=state.flinched.at[h.target_side].set(jnp.where(cond, 1, old).astype(old.dtype))
    )


def _percent_of_dealt(h: _Hit) -> jnp.ndarray:
    """``BattleSystem_Divide(hitDamage, 100 / percent)``: at least 1 of a non-zero hit."""
    return jnp.where(h.dealt > 0, jnp.maximum(h.dealt * h.param[0] // 100, 1), 0)


def _drain(tables, state, h: _Hit) -> BattleState:
    heal = jnp.where((h.effect == _E.DRAIN) & h.effective, _percent_of_dealt(h), 0)
    hp = state.hp[h.side, h.user].astype(_I32)
    new = jnp.minimum(hp + heal, state.max_hp[h.side, h.user].astype(_I32))
    return state._replace(hp=state.hp.at[h.side, h.user].set(new.astype(state.hp.dtype)))


def _recoil(tables, state, h: _Hit) -> BattleState:
    rock_head = state.ability[h.side, h.user] == AbilityId.ROCK_HEAD
    cost = jnp.where((h.effect == _E.RECOIL) & h.effective & ~rock_head, _percent_of_dealt(h), 0)
    hp = jnp.maximum(state.hp[h.side, h.user].astype(_I32) - cost, 0)
    return state._replace(hp=state.hp.at[h.side, h.user].set(hp.astype(state.hp.dtype)))


def targets_user(tables: Tables, move) -> jnp.ndarray:
    """The move's decomp defender is the user itself: ``RANGE_USER``,
    ``RANGE_USER_SIDE`` and ``RANGE_FIELD`` (battle_lib.c:1665-1669), e.g. Growth,
    Reflect, Water Sport."""
    target = tables.move_target[move]
    return (
        (target == MoveTarget.RANGE_USER)
        | (target == MoveTarget.RANGE_USER_SIDE)
        | (target == MoveTarget.RANGE_FIELD)
    )


def defender_always(tables: Tables, move) -> jnp.ndarray:
    """``BattleSystem_Defender`` (battle_lib.c:1599) finds a defender for the move
    whatever the foe's HP: the user (``targets_user``) or, for ``RANGE_OPPONENT_SIDE``
    (Stealth Rock), the foe's battler slot (``BattleSystem_RandomOpponent``, :4194). The
    no-target check (battle_controller_player.c:2306) never fails such a move."""
    return targets_user(tables, move) | (tables.move_target[move] == MoveTarget.RANGE_OPPONENT_SIDE)


def _set(state: BattleState, field: str, index, value, cond) -> BattleState:
    """``state.<field>[index] = value`` where ``cond``."""
    arr = getattr(state, field)
    new = jnp.where(cond, value, arr[index]).astype(arr.dtype)
    return state._replace(**{field: arr.at[index].set(new)})


def _stealth_rock(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.STEALTH_ROCK) & h.landed & (state.stealth_rock[h.target_side] == 0)
    return _set(state, "stealth_rock", h.target_side, 1, cond)


def _trap(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    # MOVE_SIDE_EFFECT_CHECK_HP_AND_SUBSTITUTE: the target survived the hit.
    cond = (
        (h.effect == _E.TRAP)
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & (state.trapped[h.target_side] == 0)
    )
    state = _set(state, "trapped", h.target_side, outcome.duration[h.side], cond)
    return _set(state, "trap_move", h.target_side, h.move, cond)


def _taunt(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.TAUNT)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.taunt[h.target_side] == 0)
    )
    return _set(state, "taunt", h.target_side, outcome.duration[h.side], cond)


def _confuse_foe(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.CONFUSE_FOE)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.confusion[h.target_side] == 0)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


def _secondary_confuse(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.SECONDARY_CONFUSE)
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & (state.confusion[h.target_side] == 0)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


def _mud_sport(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.MUD_SPORT) & h.landed & (state.mud_sport[h.side] == 0)
    return _set(state, "mud_sport", h.side, 1, cond)


def _eat_berry(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    # Local import: hooks imports this module.
    from nuzline.engine.hooks import pluck_berry

    return pluck_berry(tables, state, h.side, (h.effect == _E.EAT_BERRY) & h.effective)


def _leech_seed(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    grass = jnp.any(state.types[h.target_side, h.target] == Type.GRASS)
    cond = (
        (h.effect == _E.LEECH_SEED)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & ~grass
        & (state.leech_seed[h.target_side] == 0)
    )
    return _set(state, "leech_seed", h.target_side, 1, cond)


def _water_sport(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.WATER_SPORT) & h.landed & (state.water_sport[h.side] == 0)
    return _set(state, "water_sport", h.side, 1, cond)


def _foresight(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Foresight, Odor Sleuth (``effect_script_0113.s``: an ``ON_HIT`` indirect effect,
    ``subscript_foresight.s``): the defender is identified whenever the move lands on a
    living target; there is no failure check."""
    cond = (h.effect == _E.FORESIGHT) & h.landed
    return _set(state, "foresight", h.target_side, 1, cond)


def _set_status(state: BattleState, side, slot, status, cond) -> BattleState:
    """``side``'s party ``slot`` gets ``status`` (counter 0) where ``cond``."""
    state = _set(state, "status", (side, slot), status, cond)
    return _set(state, "status_counter", (side, slot), 0, cond)


def _struggle_recoil(state: BattleState, h: _Hit, struggling) -> BattleState:
    """Struggle (``effect_script_0254.s``): an ``ON_HIT`` secondary effect,
    ``subscript_struggle.s``: ``BattleSystem_Divide(-maxHP, 4)``, at least 1, on a
    hit."""
    max_hp = state.max_hp[h.side, h.user].astype(_I32)
    cost = jnp.where(struggling & h.landed, jnp.maximum(max_hp // 4, 1), 0)
    hp = jnp.maximum(state.hp[h.side, h.user].astype(_I32) - cost, 0)
    return state._replace(hp=state.hp.at[h.side, h.user].set(hp.astype(state.hp.dtype)))


def _contact_ability(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    ability = state.ability[h.target_side, h.target]
    status = jnp.where(
        ability == AbilityId.STATIC,
        Status.PARALYSIS,
        jnp.where(ability == AbilityId.POISON_POINT, Status.POISON, Status.NONE),
    )
    contact = (tables.move_flags[h.move] & MoveFlag.CONTACT) != 0
    cond = (
        (status != Status.NONE)
        & contact
        & h.effective
        & (h.dealt > 0)
        & h.contact_proc
        & _alive(state, h.side, h.user)
        & can_have_status(state, h.side, h.user, status)
    )
    # The status message tags the holder's ability (``hooks.REVEALS_ON``). hooks imports
    # this module, so the import is local.
    from nuzline.engine.hooks import CONTACT_STATUS, reveal_ability

    state = reveal_ability(state, h.target_side, h.target, CONTACT_STATUS, cond)
    return _set_status(state, h.side, h.user, status, cond)


def _thaw_target(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (state.status[h.target_side, h.target] == Status.FREEZE)
        & (move_type_of(tables, state, h.side, h.move) == Type.FIRE)
        & (h.dealt > 0)
        & _alive(state, h.target_side, h.target)
    )
    return _set_status(state, h.target_side, h.target, Status.NONE, cond)


def stealth_rock_damage(tables: Tables, state: BattleState, side) -> BattleState:
    """Stealth Rock on ``side``'s active Pokemon as it switches in
    (``subscript_hazards_check.s`` -> ``BtlCmd_CheckStealthRock``, battle_script.c:8302):
    with the flag set and HP left, ``BattleSystem_Divide(maxHP, 32/16/8/4/2)`` by the
    Rock matchup (``BattleSystem_TypeMatchupMultiplier``), i.e. ``maxHP * eff // 800``
    in ``effectiveness``'s x100 encoding, at least 1; nothing when immune."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    eff = effectiveness(tables, Type.ROCK, state.types[side, mon])
    max_hp = state.max_hp[side, mon].astype(_I32)
    hp = state.hp[side, mon].astype(_I32)
    damage = jnp.where(eff > 0, jnp.maximum(max_hp * eff // 800, 1), 0)
    damage = jnp.where((state.stealth_rock[side] != 0) & (hp > 0), jnp.minimum(damage, hp), 0)
    return state._replace(hp=state.hp.at[side, mon].set((hp - damage).astype(state.hp.dtype)))


def apply_move_effect(
    tables: Tables,
    state: BattleState,
    side,
    move,
    outcome: Outcome,
    hit_index,
    *,
    connected,
    target_moved,
    power_mul=10,
    struggling=False,
) -> BattleState:
    """Hit ``hit_index`` (0-4, an int or a traced int32) of ``move`` by ``side``'s active Pokemon: damage with
    this hit's crit and roll, then the move's effect.

    ``connected``: the move is being used and passed its one accuracy check
    against a living target (any target for a ``RANGE_USER`` move). ``target_moved``:
    the target acted earlier this turn (no flinch). ``power_mul``: power scale in
    tenths (Pursuit on a switch: 20). ``struggling``: the move is Struggle, whose
    recoil is its secondary effect. A hit beyond ``hit_count`` or after the target
    fainted does nothing. The order is ``BattleControllerPlayer_AfterMoveMessage``'s
    (battle_controller_player.c:3467): the move's own effect (``*_TRIGGER_SECONDARY``)
    before the contact ability (``*_TRIGGER_ABILITY``), then the defender's thaw.
    """
    side = jnp.asarray(side, _I32)
    target_side = 1 - side
    user, target = state.active[side], state.active[target_side]
    move = jnp.asarray(move, _I32)
    # A RANGE_USER move (Growth, Withdraw, Rock Polish) lands on its user, so the
    # foe's active having fainted earlier this turn does not stop it.
    landed = (
        connected
        & (hit_index < hit_count(tables, move, outcome, side))
        & (_alive(state, target_side, target) | defender_always(tables, move))
    )

    damage = damage_for_roll(
        tables,
        state,
        side,
        move,
        outcome.crit[side, hit_index] != 0,
        outcome.roll[side, hit_index],
        power_mul,
        outcome.magnitude[side].astype(_I32),
    )
    target_hp = state.hp[target_side, target].astype(_I32)
    dealt = jnp.where(landed & ~targets_user(tables, move), jnp.minimum(damage, target_hp), 0)
    state = state._replace(
        hp=state.hp.at[target_side, target].set((target_hp - dealt).astype(state.hp.dtype))
    )

    effect = tables.move_effect[move].astype(_I32)
    param = tables.move_param[move].astype(_I32)
    is_status = tables.move_category[move] == STATUS
    # A type-checked status move (Thunder Wave) fails on a 0x matchup.
    type_ok = (param[1] == 0) | (
        effectiveness(
            tables,
            move_type_of(tables, state, side, move),
            state.types[target_side, target],
            state.foresight[target_side] != 0,
        )
        > 0
    )
    passes = jnp.where(is_status, (effect != _E.STATUS_FOE) | type_ok, damage > 0)
    a_ability = state.ability[side, user]
    d_ability = state.ability[target_side, target]
    chance = tables.move_effect_chance[move]
    h = _Hit(
        side=side,
        target_side=target_side,
        user=user,
        target=target,
        effect=effect,
        param=param,
        certain=chance == 0,
        landed=landed,
        effective=landed & passes,
        proc=(chance == 0) | (outcome.secondary[side, hit_index] != 0),
        dealt=dealt,
        target_moved=jnp.asarray(target_moved, jnp.bool_),
        move=move,
        target_ignorable=jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, d_ability),
        contact_proc=outcome.contact_proc[target_side, hit_index] != 0,
    )
    for fn in (
        _stat_self,
        _stat_foe,
        _secondary_stat_self,
        _secondary_stat_foe,
        _flinch,
        _drain,
        _recoil,
    ):
        state = fn(tables, state, h)
    for fn in (
        _status_foe,
        _secondary_status,
        _stealth_rock,
        _trap,
        _taunt,
        _confuse_foe,
        _secondary_confuse,
        _leech_seed,
        _water_sport,
        _mud_sport,
        _eat_berry,
        _foresight,
    ):
        state = fn(tables, state, h, outcome)
    state = _struggle_recoil(state, h, jnp.asarray(struggling, jnp.bool_))
    for fn in (_contact_ability, _thaw_target):
        state = fn(tables, state, h, outcome)
    return state
