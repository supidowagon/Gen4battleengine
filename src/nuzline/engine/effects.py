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
* Fake Out (``FAKE_OUT``): ``effect_script_0158.s`` fails unless ``fakeOutTurnNumber ==
  totalTurns`` (``step._use_move``: the move then does not connect), else it is an indirect
  TO_DEFENDER flinch at the move's chance through the same ``subscript_flinch_mon.s``; there
  Inner Focus's "won't flinch" message (:18-23), shown for ``MOVE_FAKE_OUT`` only, names the
  ability (``hooks.FLINCH_BLOCKED``).
* Thunder Fang (``FANG_PARALYZE_FLINCH``): ``effect_script_0275.s`` is an indirect ``ON_HIT``
  TO_DEFENDER effect (no outer roll, battle_lib.c:1524-1529) running
  ``subscript_paralyze_or_flinch.s``: ``CheckEffectActivation`` (``RandNext() % 100 <
  chance`` and the target has HP, battle_script.c:8377-8378; ``outcome.secondary``) then
  ``subscript_paralyze.s`` as a secondary paralysis (``_inflict``, non-direct: Limber is
  silent, no type immunity), then a second ``CheckEffectActivation``
  (``outcome.fang_flinch``) then ``subscript_flinch_mon.s`` as for ``FLINCH`` (no Inner Focus
  message: not Fake Out, not direct). Both run before the contact ability (``_fang``).
* Recover (``RECOVER``): ``effect_script_0032.s`` -> ``subscript_recover_half_max_hp.s``:
  ``DivideVarByValue(maxHP, 2)`` (``BattleSystem_Divide``, at least 1), then
  ``subscript_recover_hp.s``: "HP is full!" at full HP (:6), else the HP update, capped at
  max HP. Heal Block (which would stop it in ``CheckStatusDisruption``) comes from an
  unsupported move.
* AncientPower (``RAISE_ALL_STATS_HIT``): ``effect_script_0140.s``, an indirect TO_ATTACKER
  effect at the move's chance (battle_lib.c:1570-1586: the roll, then the recipient, the
  user, must have HP; the target's HP does not matter) running
  ``subscript_boost_all_stats.s``: Atk, Def, Spe, SpA, SpD +1, each its own
  ``BtlCmd_ChangeStatStage`` (a capped one is skipped).
* Curse, non-Ghost user (``CURSE``): ``effect_script_0109.s`` -> ``subscript_curse_normal.s``,
  a direct TO_ATTACKER effect: Spe -1, then Atk +1, then Def +1, each its own
  ``BtlCmd_ChangeStatStage`` (a capped stage prints "won't go higher/lower" and the next one
  still runs; none sets ``MOVE_STATUS_FAILED``). Its defender is the user
  (``targets_user``). A Ghost-type user takes ``subscript_curse_ghost.s``: refused by
  coverage.
* Selfdestruct, Explosion (``SELF_DESTRUCT``): ``effect_script_0007.s`` sets the user's HP
  to 0 before ``TryMove`` (``step._use_move``: before the accuracy check and the type chart;
  not when the move has no target), then a plain hit on the halved Defense (``damage.py``).
  No effect function runs for it. Its Damp branch is unreachable (Damp is unsupported).
* Drain / recoil:``subscript_drain_half_damage_dealt.s``,
  ``subscript_recoil_1_{2,4}.s``: a fraction of ``hitDamage``, the damage dealt
  capped at the target's HP (``BtlCmd_UpdateHealthBarValue``, battle_script.c:1701),
  through ``BattleSystem_Divide`` (at least 1). Rock Head prevents recoil. A Big Root
  holder's drain is then ``* (100 + 30) / 100`` (``big_root``).
* Multi-hit: ``BtlCmd_SetMultiHit`` (battle_script.c:2634) and
  ``BattleControllerPlayer_LoopMultiHit`` (battle_controller_player.c:3777): the
  move script reruns per hit with ``SYSCTL_MULTI_HIT_MOVE``, which skips the
  accuracy check, so only ``outcome.hit[side, 0]`` matters while crit and roll
  are per hit; the loop stops once the target faints. ``SetMultiHit 0`` draws 2-5
  (``outcome.n_hits``); ``SetMultiHit 2`` (``effect_script_0044.s``) is two hits
  (``move_param[m, 0]``). Focus Sash activates only on a hit taken at full HP
  (bcp.c:3386-3404) and then stays active for the rest of the move (``play_hit``).
* Rindo / Chilan Berry (``RESIST_BERRY``): ``subscript_update_hp.s:9`` calls
  ``subscript_type_resist_berry.s`` on every move hit taken during an action (``play_hit``,
  Pursuit's hit through it, and the confusion self-hit in ``step``): ``resist_berry``,
  ``resist_berry_damage``.
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
  (no HP check, battle_lib.c:1519-1524) running ``subscript_pluck.s``: ``hooks.pluck_berry``
  (after every other effect function: each move has one effect).
* Leech Seed: ``subscript_leech_seed_start.s``: Grass types and already seeded
  targets are immune. The end-of-turn drain is in ``step``.
* Water Sport: ``effect_script_0210.s`` sets the user's flag (fails if set);
  ``damage.py`` halves Fire power while any active battler has it.
* Light Screen, Reflect: ``effect_script_0035.s`` / ``_0065.s`` (``BtlCmd_TryLightScreen``,
  script.c:4214): fail when the user's side has it, else 5 turns (``_screen``); the damage
  halving is in ``damage.py``, the countdown in ``step``.
* Brick Break: ``effect_script_0186.s``, a direct ``BREAK_SCREENS`` effect after the hit,
  even on an immune target (``_brick_break``); its own damage ignores the screens.
* Night Shade, Seismic Toss (``LEVEL_DAMAGE``), Hidden Power, Magnitude: damage and type
  come from ``damage.py`` (``fixed_damage``, ``hidden_power``, ``MAGNITUDE_POWER``); no
  effect function runs for them.
* Foresight, Odor Sleuth: ``effect_script_0113.s`` / ``subscript_foresight.s`` set
  ``foresight`` on the defender whenever the move lands on a living target (no
  failure check). ``damage.py`` skips the Ghost immunities for it and ``outcome.py``
  ignores the defender's positive evasion.
* Disable: ``effect_script_0086.s`` / ``BtlCmd_TryDisable`` (battle_script.c:4581): on a
  hit, fails unless the target has no disabled move and its ``movePrevByBattler`` is a
  move it knows with PP left; else that move is disabled for ``outcome.duration`` (3-6,
  ``RandNext() % 4 + 3``). ``actions.usable_moves`` and ``step`` honour it; the
  end-of-turn countdown is in ``step``.
* Bide: ``effect_script_0026.s`` / ``subscript_bide_start.s``: a first use (not locked) locks
  the user into the move with two turns to go and ``storedDamage = 0``; ``apply_move_effect``
  adds every hit the Biding battler takes (``BattleControllerPlayer_UpdateHP``,
  battle_controller_player.c:3407). The turn count, the unleash (``2 * storedDamage`` at the
  foe, ``damage.fixed_damage``) and the no-target case are in ``step`` (``_use_move``).
* Rollout, Ice Ball: ``effect_script_0117.s`` / ``BtlCmd_CalcRolloutPower``
  (battle_script.c:5661): a first use locks the user with ``rolloutCount = 5``; each use
  decrements it and the lock ends at 0 (the fifth use) or when the move does not connect
  (a miss runs ``subscript_missed.s``, which unlocks). Power is ``damage.rollout_multiplier``.
* Defense Curl: ``effect_script_0156.s`` sets ``curled`` before the Defense stage.
* Pursuit is a plain hit here; its pre-switch hit at double power is in ``step``.
* Static / Poison Point / Flame Body: ``BattleSystem_TriggerAbilityOnHit``
  (battle_lib.c:4200; Static :4214, Poison Point :4303, Flame Body :4321), run per hit
  after the secondary effect (``ONE_HIT_TRIGGER_ABILITY`` /
  ``MULTI_HIT_TRIGGER_ABILITY``, battle_controller_player.c:3513, :3573): the
  defender's own ability (``Battler_Ability``: Mold Breaker does not matter), a contact
  move that hit, was not ineffective and dealt damage, a living attacker with no status,
  and the roll ``RandNext() % 10 < 3`` (``outcome.contact_proc[defending side, hit]``).
  The defender need not be alive. The status goes through ``subscript_paralyze.s`` /
  ``subscript_poison.s`` / ``subscript_burn.s`` as an ability side effect: Safeguard and
  Substitute do not apply, the type immunities do (``can_have_status``: a Fire type is
  never burned), and an attacker's own Limber stops the paralysis silently
  (subscript_paralyze.s:5, :_170). Its message reveals the holder's ability to the AI
  (``hooks.reveal_ability``).
* Rough Skin (battle_lib.c:4257-4271, same function, so never with the above): a living
  attacker (Magic Guard is unsupported), a contact move that was not ineffective and
  dealt damage, no roll: ``subscript_rough_skin.s`` takes ``BattleSystem_Divide(maxHP,
  8)`` from the attacker through ``BATTLE_SUBSCRIPT_UPDATE_HP`` (it may faint; not
  Bide's stored damage, which only the controller's ``UpdateHP`` adds) and its message
  tags the holder's ability. The faint stops a multi-hit move (``step``).
* Limber (``_inflict``): a paralysis (Thunder Wave, a paralysis secondary effect) on a
  living target with Limber as the attacker sees it (``target_ignorable``) inflicts
  nothing (subscript_paralyze.s:5); the "prevents paralysis" message (:_170) is printed,
  and reveals the ability, only for a direct effect (a status move). A pierced Limber
  (Mold Breaker) cures at the move's end (``hooks.recover_status_by_ability``).
* Insomnia (``_inflict``): a sleep move (or sleep secondary effect) on a living target
  with Insomnia as the attacker sees it (``target_ignorable``: Mold Breaker pierces it)
  inflicts nothing (subscript_fall_asleep.s:_055, :_202), whatever the target's other
  status; the "stayed awake" message (:_221) reveals the ability to the AI.
* Thaw on a Fire hit: ``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (battle_controller_player.c:
  3710), after the contact ability: a frozen defender that is alive and took damage
  from a Fire-type move thaws. The user's own thaw (Flame Wheel) is in ``step``.

Unreachable branches are not modelled: Substitute, Mist, Safeguard and Uproar
come from unsupported moves; Shield Dust, Clear Body, Hyper Cutter, Immunity,
Leaf Guard, Liquid Ooze, Magic Guard, Reckless, Serene Grace,
Skill Link, Own Tempo, Levitate, Normalize and Color Change are unsupported
abilities; Grip Claw is an unsupported item.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import STATUS, damage_for_roll, divide, effectiveness, move_type_of
from nuzline.engine.ids import (
    MOVE_FAKE_OUT,
    MOVE_STRUGGLE,
    RESIST_ANY_NORMAL,
    AbilityId,
    EffectId,
    ItemId,
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
BIDE_TURNS = 2  # VOLATILE_CONDITION_BIDE_1 (condition.h:41): the 2-bit counter's value 2
ROLLOUT_USES = 5  # BtlCmd_CalcRolloutPower starts rolloutCount at 5


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
    """``int32``: hits the move makes if nothing faints: for ``MULTI_HIT`` its fixed count
    ``move_param[m, 0]`` (``SetMultiHit 2``), or ``outcome.n_hits[side]`` when that is 0
    (``SetMultiHit 0``, the 2-5 law); else 1."""
    multi = tables.move_effect[move] == _E.MULTI_HIT
    fixed = tables.move_param[move, 0].astype(_I32)
    drawn = outcome.n_hits[side].astype(_I32)
    return jnp.where(multi, jnp.where(fixed > 0, fixed, drawn), 1)


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
    cond = (h.effect == _E.STAT_SELF) & h.landed
    state = change_stage(state, h.side, h.param[0], h.param[1], cond)
    # Defense Curl (move_param[m, 2] = 1) sets the flag whatever the stage change does.
    return _set(state, "curled", h.side, 1, cond & (h.param[2] == 1))


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


def _inflict(state: BattleState, h: _Hit, outcome: Outcome, cond, direct) -> BattleState:
    """``direct``: a status move's own effect (``SIDE_EFFECT_TYPE_DIRECT``), not a damaging
    move's secondary effect (``_INDIRECT``)."""
    status = h.param[0]
    cond = cond & _alive(state, h.target_side, h.target)
    # subscript_fall_asleep.s:_055/_202/_221: an ignorable Insomnia stops the sleep before
    # any other check, and its "stayed awake" message tags the ability.
    blocked = cond & (status == Status.SLEEP) & (h.target_ignorable == AbilityId.INSOMNIA)
    # subscript_paralyze.s:5/_170: an ignorable Limber stops a paralysis before any other
    # check; "prevents paralysis" (TAG_NICKNAME_ABILITY) is printed for a direct effect only.
    limber = cond & (status == Status.PARALYSIS) & (h.target_ignorable == AbilityId.LIMBER)
    from nuzline.engine.hooks import (  # hooks imports this module
        PARALYSIS_BLOCKED,
        SLEEP_BLOCKED,
        reveal_ability,
    )

    state = reveal_ability(state, h.target_side, h.target, SLEEP_BLOCKED, blocked)
    state = reveal_ability(state, h.target_side, h.target, PARALYSIS_BLOCKED, limber & direct)
    blocked = blocked | limber
    cond = cond & ~blocked & can_have_status(state, h.target_side, h.target, status)
    counter = jnp.where(status == Status.SLEEP, outcome.duration[h.side], 0)
    old_status = state.status[h.target_side, h.target]
    old_counter = state.status_counter[h.target_side, h.target]
    state = state._replace(
        status=state.status.at[h.target_side, h.target].set(
            jnp.where(cond, status, old_status).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[h.target_side, h.target].set(
            jnp.where(cond, counter, old_counter).astype(state.status_counter.dtype)
        ),
    )
    # subscript_fall_asleep.s:_176 and subscript_freeze.s:_073 unlock the new sleeper / ice cube.
    sleeps = (status == Status.SLEEP) | (status == Status.FREEZE)
    return unlock_move(state, h.target_side, cond & sleeps)


def _status_foe(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    return _inflict(state, h, outcome, (h.effect == _E.STATUS_FOE) & h.effective, True)


def _secondary_status(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.SECONDARY_STATUS) & h.effective & h.proc
    return _inflict(state, h, outcome, cond, False)


def _flinch(tables, state, h: _Hit) -> BattleState:
    """``FLINCH`` and ``FAKE_OUT`` (a Fake Out that failed did not connect):
    ``subscript_flinch_mon.s``."""
    reached = (
        ((h.effect == _E.FLINCH) | (h.effect == _E.FAKE_OUT))
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & ~h.target_moved
    )
    inner_focus = h.target_ignorable == AbilityId.INNER_FOCUS
    # :18-23: Inner Focus's message (TAG_NICKNAME_ABILITY) is shown for Fake Out only.
    from nuzline.engine.hooks import FLINCH_BLOCKED, reveal_ability  # hooks imports this module

    shown = reached & inner_focus & (h.move == MOVE_FAKE_OUT)
    state = reveal_ability(state, h.target_side, h.target, FLINCH_BLOCKED, shown)
    cond = reached & ~inner_focus
    old = state.flinched[h.target_side]
    return state._replace(
        flinched=state.flinched.at[h.target_side].set(jnp.where(cond, 1, old).astype(old.dtype))
    )


def _fang(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``FANG_PARALYZE_FLINCH``: ``subscript_paralyze_or_flinch.s:5-10``, the paralysis roll
    (``h.proc``) then the flinch roll (``outcome.fang_flinch``), each needing a living
    target (``BtlCmd_CheckEffectActivation``); the flinch also needs a target that has not
    moved (``subscript_flinch_mon.s:5``) and no Inner Focus as the attacker sees it (:7, its
    message is for Fake Out or a direct flinch only, :17-27)."""
    fang = (h.effect == _E.FANG_PARALYZE_FLINCH) & h.effective
    state = _inflict(state, h, outcome, fang & h.proc, False)
    cond = (
        fang
        & (outcome.fang_flinch[h.side] != 0)
        & _alive(state, h.target_side, h.target)
        & ~h.target_moved
        & (h.target_ignorable != AbilityId.INNER_FOCUS)
    )
    old = state.flinched[h.target_side]
    return state._replace(
        flinched=state.flinched.at[h.target_side].set(jnp.where(cond, 1, old).astype(old.dtype))
    )


def _recover(tables, state, h: _Hit) -> BattleState:
    """Recover: ``subscript_recover_half_max_hp.s`` + ``subscript_recover_hp.s`` (an ON_HIT
    direct effect on the user; module docstring)."""
    hp = state.hp[h.side, h.user].astype(_I32)
    max_hp = state.max_hp[h.side, h.user].astype(_I32)
    cond = (h.effect == _E.RECOVER) & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + jnp.maximum(max_hp // 2, 1), max_hp)
    return _set(state, "hp", (h.side, h.user), healed, cond)


# subscript_boost_all_stats.s and subscript_curse_normal.s, in script order.
ALL_STATS_RAISED = (Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD)
CURSE_STAGES = ((Stat.SPE, -1), (Stat.ATK, 1), (Stat.DEF, 1))


def _raise_all_stats(tables, state, h: _Hit) -> BattleState:
    user_alive = _alive(state, h.side, h.user)
    cond = (h.effect == _E.RAISE_ALL_STATS_HIT) & h.effective & h.proc & user_alive
    for stat in ALL_STATS_RAISED:
        state = change_stage(state, h.side, stat, 1, cond)
    return state


def _curse(tables, state, h: _Hit) -> BattleState:
    # BattleSystem_TriggerPrimaryEffect (battle_lib.c:1491): the recipient (the user) has HP.
    cond = (h.effect == _E.CURSE) & h.landed & _alive(state, h.side, h.user)
    for stat, delta in CURSE_STAGES:
        state = change_stage(state, h.side, stat, delta, cond)
    return state


def _percent_of_dealt(h: _Hit) -> jnp.ndarray:
    """``BattleSystem_Divide(hitDamage, 100 / percent)``: at least 1 of a non-zero hit."""
    return jnp.where(h.dealt > 0, jnp.maximum(h.dealt * h.param[0] // 100, 1), 0)


def big_root(tables: Tables, state: BattleState, side, heal) -> jnp.ndarray:
    """``heal`` for ``side``'s active Pokemon after Big Root (``HOLD_EFFECT_LEECH_BOOST``):
    ``heal * (100 + param) / 100`` (param 30), plain integer division after the base
    amount, no minimum (``subscript_drain_half_damage_dealt.s:_011-_031``,
    ``subscript_leech_seed_effect.s``)."""
    item = state.item[side, state.active[side]]
    boost = tables.item_param[item, 0].astype(_I32) + 100
    root = tables.item_effect[item] == ItemId.BIG_ROOT
    return jnp.where(root, heal * boost // 100, heal)


def _drain(tables, state, h: _Hit) -> BattleState:
    heal = jnp.where((h.effect == _E.DRAIN) & h.effective, _percent_of_dealt(h), 0)
    heal = big_root(tables, state, h.side, heal)
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
    Reflect, Water Sport; and a non-Ghost user's Curse (``RANGE_SINGLE_TARGET`` in the
    data): ``BattleControllerPlayer_MustSelectTarget`` (bcp.c:4344-4346) makes its range
    ``RANGE_USER``, so the chosen target is the user (:4363-4364) and
    ``BattleSystem_Defender``'s single-target branch returns it (battle_lib.c:1709-1718)."""
    target = tables.move_target[move]
    return (
        (target == MoveTarget.RANGE_USER)
        | (target == MoveTarget.RANGE_USER_SIDE)
        | (target == MoveTarget.RANGE_FIELD)
        | (tables.move_effect[move] == _E.CURSE)
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


def unlock_move(state: BattleState, side, cond) -> BattleState:
    """``Battler_UnlockMoveChoice`` (battle_lib.c:1834) on ``side`` where ``cond``: the move
    lock, Bide's turns and stored damage (hidden state, 0 outside a Bide) and the Rollout
    count go to 0."""
    for field in ("locked_move", "bide_turns", "bide_damage", "rollout_count"):
        state = _set(state, field, side, 0, cond)
    return state


def _bide(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``subscript_bide_start.s``: only a first use (``effect_script_0026.s`` skips the
    start while ``VOLATILE_CONDITION_MOVE_LOCKED`` is set or on the unleash turn,
    ``SYSCTL_LAST_OF_MULTI_TURN``) locks the user in with ``BIDE_1`` (two turns) and
    ``storedDamage = 0``."""
    cond = (h.effect == _E.BIDE) & h.landed & (state.locked_move[h.side] == 0)
    state = _set(state, "locked_move", h.side, h.move, cond)
    state = _set(state, "bide_turns", h.side, BIDE_TURNS, cond)
    return _set(state, "bide_damage", h.side, 0, cond)


def _rollout(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``BtlCmd_CalcRolloutPower`` (battle_script.c:5661): a first use sets
    ``rolloutCount = 5``, every use decrements it, and the lock is released at 0. A hit
    that dealt no damage (an immunity) runs ``subscript_missed.s``, which unlocks."""
    first = state.locked_move[h.side] == 0
    left = jnp.where(first, ROLLOUT_USES - 1, jnp.maximum(state.rollout_count[h.side] - 1, 0))
    keep = h.effective & (left > 0)
    cond = (h.effect == _E.ROLLOUT) & h.landed
    state = _set(state, "locked_move", h.side, jnp.where(keep, h.move, 0), cond)
    return _set(state, "rollout_count", h.side, jnp.where(keep, left, 0), cond)


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


def _disable(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Disable (``BtlCmd_TryDisable``, battle_script.c:4581; the script skips it when the
    move missed): the target's last move must sit in one of its slots with PP left, and
    nothing may be disabled already. The target's ``last_move`` is ``movePrevByBattler``:
    0 after a move that failed, Struggle's id for Struggle (in no slot)."""
    target_moves = state.moves[h.target_side, h.target]
    last = state.last_move[h.target_side]
    in_slot = (target_moves == last) & (last != 0)
    has_pp = jnp.any(in_slot & (state.pp[h.target_side, h.target] > 0))
    cond = (
        (h.effect == _E.DISABLE) & h.effective & (state.disabled_move[h.target_side] == 0) & has_pp
    )
    state = _set(state, "disabled_move", h.target_side, last, cond)
    return _set(state, "disable_turns", h.target_side, outcome.duration[h.side], cond)


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


def _aqua_ring(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.AQUA_RING) & h.landed & (state.aqua_ring[h.side] == 0)
    return _set(state, "aqua_ring", h.side, 1, cond)


def _mud_sport(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.MUD_SPORT) & h.landed & (state.mud_sport[h.side] == 0)
    return _set(state, "mud_sport", h.side, 1, cond)


SCREEN_TURNS = 5  # NUM_SCREEN_TURNS (include/constants/battle.h:142)


def _screen(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Light Screen, Reflect: ``BtlCmd_TryLightScreen`` / ``TryReflect`` (script.c:4214,
    :4257) fail when the user's side has it, else set it for ``NUM_SCREEN_TURNS``. Light
    Clay (``HOLD_EFFECT_EXTEND_SCREENS``) is an unmapped item, so coverage refuses it."""
    for effect, field in ((_E.LIGHT_SCREEN, "light_screen"), (_E.REFLECT, "reflect")):
        cond = (h.effect == effect) & h.landed & (getattr(state, field)[h.side] == 0)
        state = _set(state, field, h.side, SCREEN_TURNS, cond)
    return state


def _brick_break(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Brick Break: a direct ``BREAK_SCREENS`` effect (lib.c:1481-1510) that still applies
    on an immune target (``MOVE_STATUS_DID_NOT_AFFECT``) but not on a miss;
    ``BtlCmd_TryBreakScreens`` (script.c:6550) clears both screens of the defender's side."""
    cond = (h.effect == _E.BRICK_BREAK) & h.landed
    state = _set(state, "reflect", h.target_side, 0, cond)
    return _set(state, "light_screen", h.target_side, 0, cond)


def _eat_berry(tables, state, h: _Hit, outcome: Outcome) -> tuple[BattleState, jnp.ndarray]:
    """``(state, pluck_flag)`` (``hooks.pluck_berry``)."""
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
        jnp.where(
            ability == AbilityId.POISON_POINT,
            Status.POISON,
            jnp.where(ability == AbilityId.FLAME_BODY, Status.BURN, Status.NONE),
        ),
    )
    contact = (tables.move_flags[h.move] & MoveFlag.CONTACT) != 0
    hit = contact & h.effective & (h.dealt > 0) & _alive(state, h.side, h.user)
    # subscript_paralyze.s:5: the attacker's own Limber (the side-effect battler is the
    # attacker itself, so no Mold Breaker can pierce it) ends Static's paralysis silently
    # (:_170 ends an ABILITY side effect with no message).
    limber = (status == Status.PARALYSIS) & (state.ability[h.side, h.user] == AbilityId.LIMBER)
    cond = (
        (status != Status.NONE)
        & hit
        & h.contact_proc
        & ~limber
        & can_have_status(state, h.side, h.user, status)
    )
    # Rough Skin (battle_lib.c:4257-4271): no roll; subscript_rough_skin.s takes
    # BattleSystem_Divide(maxHP, 8) from the attacker (it may faint).
    rough = (ability == AbilityId.ROUGH_SKIN) & hit
    max_hp = state.max_hp[h.side, h.user].astype(_I32)
    hp = state.hp[h.side, h.user].astype(_I32)
    hurt = jnp.where(rough, jnp.maximum(hp - jnp.maximum(max_hp // 8, 1), 0), hp)
    state = state._replace(hp=state.hp.at[h.side, h.user].set(hurt.astype(state.hp.dtype)))
    # The status message and Rough Skin's message tag the holder's ability
    # (``hooks.REVEALS_ON``). hooks imports this module, so the import is local.
    from nuzline.engine.hooks import CONTACT_DAMAGE, CONTACT_STATUS, reveal_ability

    state = reveal_ability(state, h.target_side, h.target, CONTACT_STATUS, cond)
    state = reveal_ability(state, h.target_side, h.target, CONTACT_DAMAGE, rough)
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


class MoveHit(NamedTuple):
    """What one hit leaves for the rest of the action (``play_hit``)."""

    # DEFENDER_SELF_TURN_FLAGS.focusItemActivated (bcp.c:3386-3393): set on a hit taken at
    # full HP holding a Focus Sash; it lasts until BattleContext_Init at the action's end
    # (bcp.c:3998, battle_lib.c:2014), so every later hit of the move is capped too.
    focus_armed: jnp.ndarray
    # ATTACKER_SELF_TURN_FLAGS SELF_TURN_FLAG_PLUCK_BERRY left set by a Pluck / Bug Bite
    # whose berry did nothing for the eater (hooks.pluck_berry).
    pluck_flag: jnp.ndarray


def focus_sash_damage(state: BattleState, side, damage, armed) -> tuple:
    """``(dealt, endured)`` for ``damage`` (> 0) taken by ``side``'s active Pokemon:
    ``BattleControllerPlayer_UpdateHP`` (bcp.c:3386-3404) / ``BtlCmd_CheckHoldOnWith1HP``
    (script.c:9098-9124). ``armed`` (the hit's focus flag, or a full-HP Focus Sash
    holder) and ``curHP + damage <= 0`` leave it at 1 HP (``MOVE_STATUS_ENDURED_ITEM``);
    else the damage is capped at its HP. ``subscript_move_followup_message.s:12-19``
    then removes the item only while it still is a Focus Sash
    (``CheckItemHoldEffect ... HOLD_EFFECT_ENDURE``)."""
    mon = state.active[side]
    hp = state.hp[side, mon].astype(_I32)
    endured = armed & (damage > 0) & (damage >= hp)
    return jnp.where(endured, hp - 1, jnp.minimum(damage, hp)), endured


# Effects whose script sets SYSCTL_IGNORE_TYPE_CHECKS or SYSCTL_IGNORE_IMMUNITIES: Night Shade
# and Seismic Toss (effect_script_0087.s:5), Foresight (effect_script_0113.s:6, no damage).
# subscript_type_resist_berry.s:5 skips the berry for such a hit. tests/tools/
# test_effect_map.py checks the list against every mapped effect script.
IGNORES_TYPE_CHECKS = (_E.LEVEL_DAMAGE, _E.FORESIGHT)


def resist_berry(
    tables: Tables, state: BattleState, side, move_type, super_effective
) -> jnp.ndarray:
    """``side``'s active Pokemon holds a Rindo or Chilan Berry that halves the hit it takes
    (``subscript_type_resist_berry.s``, from ``subscript_update_hp.s:9``, which runs it for
    every damaging hit while a battler still has an action, ``waitingBattlers != 0``, and
    no ``SYSCTL_SKIP_SPRITE_BLINK``: never for recoil, Rough Skin or the end of turn).
    ``move_type`` is the script's ``calcTemp``: ``battleCtx->moveType`` (Hidden Power's
    type) when it is not Normal, else ``CURRENT_MOVE_DATA.type`` (:_020-_031; Normalize,
    which would make it Normal, is unsupported). ``super_effective`` is
    ``MOVE_STATUS_SUPER_EFFECTIVE``. Chilan (``RESIST_ANY_NORMAL``) needs a Normal type
    only (:_128); Rindo a super-effective hit of its type (:_156). The caller skips
    ``IGNORES_TYPE_CHECKS`` effects (:5; OHKO moves, :6, are unsupported)."""
    item = state.item[side, state.active[side]]
    resisted = tables.item_param[item, 1].astype(_I32)
    move_type = jnp.asarray(move_type, _I32)
    halves = jnp.where(
        resisted == RESIST_ANY_NORMAL,
        move_type == Type.NORMAL,
        jnp.asarray(super_effective) & (move_type == resisted),
    )
    return (tables.item_effect[item] == ItemId.RESIST_BERRY) & halves


def resist_berry_damage(state: BattleState, side, damage, resisted, dealt) -> jnp.ndarray:
    """The HP ``side``'s active Pokemon loses: where ``resisted``, the berry's
    ``DivideVarByValue BTLVAR_HP_CALC_TEMP, 2`` (``BattleSystem_Divide``: at least 1) of
    ``damage``, ``battleCtx->damage`` after ``UpdateHP``'s endure cap (bcp.c:3396-3404) but
    before ``BtlCmd_UpdateHealthBarValue`` caps it at the HP (battle_script.c:1699-1704);
    else ``dealt``. The berry is then removed (``RemoveItem BTLSCR_MSG_TEMP``)."""
    hp = state.hp[side, state.active[side]].astype(_I32)
    return jnp.where(resisted, jnp.minimum(divide(damage, 2), hp), dealt)


def holds(tables: Tables, state: BattleState, side, item: ItemId) -> jnp.ndarray:
    """``side``'s active Pokemon holds an item with effect ``item``."""
    return tables.item_effect[state.item[side, state.active[side]]] == item


def consume_item(state: BattleState, side, cond) -> BattleState:
    """``BtlCmd_RemoveItem`` (battle_script.c:8972) on ``side``'s active Pokemon where
    ``cond``: the item goes to ``item_consumed``."""
    mon = state.active[side]
    item = state.item[side, mon]
    state = _set(state, "item_consumed", (side, mon), item, cond)
    return _set(state, "item", (side, mon), 0, cond)


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
    """``play_hit``'s state, for a hit that is the first (or only) of its action."""
    return play_hit(
        tables,
        state,
        side,
        move,
        outcome,
        hit_index,
        connected=connected,
        target_moved=target_moved,
        power_mul=power_mul,
        struggling=struggling,
    )[0]


def play_hit(
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
    focus_armed=False,
) -> tuple[BattleState, MoveHit]:
    """Hit ``hit_index`` (0-4, an int or a traced int32) of ``move`` by ``side``'s active Pokemon: damage with
    this hit's crit and roll, then the move's effect.

    ``connected``: the move is being used and passed its one accuracy check
    against a living target (any target for a ``RANGE_USER`` move). ``target_moved``:
    the target acted earlier this turn (no flinch). ``power_mul``: power scale in
    tenths (Pursuit on a switch: 20). ``struggling``: the move is Struggle, whose
    recoil is its secondary effect. ``focus_armed``: an earlier hit of this action
    activated the defender's Focus Sash (``MoveHit``). A hit beyond ``hit_count`` or after
    the target fainted does nothing. The order is ``BattleControllerPlayer_AfterMoveMessage``'s
    (battle_controller_player.c:3467): the move's own effect (``*_TRIGGER_SECONDARY``)
    before the contact ability (``*_TRIGGER_ABILITY``), then the defender's thaw.

    Focus Sash (``HOLD_EFFECT_ENDURE``): ``BattleControllerPlayer_UpdateHP`` runs for every
    damaging hit (``battleCtx->damage != 0``) and sets ``focusItemActivated`` when the
    defender holds a Focus Sash at full HP (bcp.c:3391-3393; Endure, the move, is
    unsupported); while it is set, a hit that would KO leaves 1 HP (``focus_sash_damage``)
    and the followup message removes the sash. The flag is never cleared between hits, so
    after a full-HP hit every later hit of the same move is capped as well (a holder at 1 HP
    takes 0); a holder below full HP at the first hit is never protected. Bide stores the
    capped damage (bcp.c:3407), drain and the contact ability see it. Sturdy only blocks
    OHKO moves (``MOVE_STATUS_ONE_HIT_KO``, bcp.c:3342), none of which is supported.
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
    # Bide is a RANGE_USER move whose unleash hits the foe (bcp.c:2702-2713).
    hits_foe = ~targets_user(tables, move) | (tables.move_effect[move] == _E.BIDE)
    update_hp = landed & hits_foe & (damage > 0)  # BattleControllerPlayer_UpdateHP runs
    sash = holds(tables, state, target_side, ItemId.FOCUS_SASH)
    full = target_hp == state.max_hp[target_side, target].astype(_I32)
    armed = jnp.asarray(focus_armed) | (update_hp & sash & full)
    capped, endured = focus_sash_damage(state, target_side, damage, armed & update_hp)
    # battleCtx->damage after the endure cap: what subscript_update_hp.s receives.
    taken = jnp.where(endured, target_hp - 1, damage)
    effect = tables.move_effect[move].astype(_I32)
    move_type = move_type_of(tables, state, side, move)
    # MOVE_STATUS_SUPER_EFFECTIVE: BattleControllerPlayer_CheckTypeChart (bcp.c:2350-2358)
    # applies the chart to a move with power that is not RANGE_USER (Bide); its
    # ApplyTypeMultiplier (battle_lib.c:7549-7573) sets the flag when the x2 entries outnumber
    # the x0.5 ones, only for table power != 0 (`update`); Struggle returns before the
    # chart (battle_lib.c:2572). Only a hit with damage reaches the berry, so no immunity.
    super_effective = (
        (effect != _E.BIDE)
        & (tables.move_power[move] != 0)
        & (move != MOVE_STRUGGLE)
        & (
            effectiveness(
                tables,
                move_type,
                state.types[target_side, target],
                state.foresight[target_side] != 0,
            )
            > 100
        )
    )
    resisted = (
        update_hp
        & ~jnp.isin(effect, jnp.asarray(IGNORES_TYPE_CHECKS, effect.dtype))
        & resist_berry(tables, state, target_side, move_type, super_effective)
    )
    dealt = jnp.where(landed & hits_foe, capped, 0)
    dealt = resist_berry_damage(state, target_side, taken, resisted, dealt)
    state = state._replace(
        hp=state.hp.at[target_side, target].set((target_hp - dealt).astype(state.hp.dtype))
    )
    state = consume_item(state, target_side, (endured & sash) | resisted)
    # BattleControllerPlayer_UpdateHP (bcp.c:3407): a Biding target stores battleCtx->damage,
    # before the berry halves the HP change: what it lost, or the unhalved amount after a
    # resist berry (a target that faints is cleared at the faint check, so the HP cap of
    # `dealt` changes nothing there).
    biding = state.bide_turns[target_side] > 0
    stored = jnp.where(resisted, taken, dealt)
    state = _set(state, "bide_damage", target_side, state.bide_damage[target_side] + stored, biding)

    param = tables.move_param[move].astype(_I32)
    is_status = tables.move_category[move] == STATUS
    # A type-checked status move (Thunder Wave) fails on a 0x matchup.
    type_ok = (param[1] == 0) | (
        effectiveness(
            tables,
            move_type,
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
        _raise_all_stats,
        _curse,
        _recover,
        _flinch,
        _drain,
        _recoil,
    ):
        state = fn(tables, state, h)
    for fn in (
        _status_foe,
        _secondary_status,
        _fang,
        _stealth_rock,
        _trap,
        _taunt,
        _disable,
        _confuse_foe,
        _secondary_confuse,
        _leech_seed,
        _water_sport,
        _mud_sport,
        _aqua_ring,
        _screen,
        _brick_break,
        _foresight,
        _bide,
        _rollout,
    ):
        state = fn(tables, state, h, outcome)
    state, pluck_flag = _eat_berry(tables, state, h, outcome)
    state = _struggle_recoil(state, h, jnp.asarray(struggling, jnp.bool_))
    for fn in (_contact_ability, _thaw_target):
        state = fn(tables, state, h, outcome)
    return state, MoveHit(focus_armed=armed, pluck_flag=pluck_flag)
