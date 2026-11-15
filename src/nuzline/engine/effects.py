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
* Present (``PRESENT``): ``effect_script_0122.s`` runs ``BtlCmd_Present`` (battle_script.c:
  5789-5806, ``outcome.present``) before ``TryMove``. Power 40/80/120: ``CalcCrit``,
  ``CalcDamage`` and a plain Normal hit at that power (``damage.PRESENT_POWER``; the type
  chart runs, a Ghost is immune). The heal (``PRESENT_HEAL``) jumps past both (no crit, no
  roll, damage 0) and sets the direct ``ON_HIT`` effect ``PRESENT_HEAL`` with
  ``SYSCTL_IGNORE_IMMUNITIES`` (``CheckTypeChart`` skips the chart, bcp.c:2355: a Ghost is
  healed): on a hit (battle_lib.c:1485-1490) ``subscript_present_heal.s`` (Heal Block is
  unsupported) runs ``subscript_recover_hp.s`` on the target with
  ``BattleSystem_Divide(maxHP, 4)``: "HP is full!" at full HP, else the capped HP update
  (``_present``).
* Protect, Detect (``PROTECT``): ``effect_script_0111.s`` -> ``subscript_protect.s`` ->
  ``BtlCmd_TryProtection`` (battle_script.c:5149-5190) in ``step._use_move``: the user's
  roll (``outcome.protect``, 2^-n by ``protect_count``), the last-actor failure and the
  chain; the block of the foe's move is ``protect_blocks`` (``CheckMoveHitOverrides``). No
  effect function runs for it.
* Tri Attack (``TRI_ATTACK``; Nature Power's call at BUILDING and SPECIAL terrain):
  ``effect_script_0036.s`` runs ``Random 2, 3`` (``outcome.tri_attack``: burn, freeze or
  paralysis, ``TRI_ATTACK_STATUSES``) before ``CalcCrit`` and sets it as an indirect
  probabilistic ``TO_DEFENDER`` effect: ``BattleSystem_TriggerSecondaryEffect``
  (battle_lib.c:1570-1586) rolls ``RandNext() % 100 < chance`` (``outcome.secondary``), then
  needs a living defender, no substitute hit and no ``MOVE_STATUS_NO_EFFECTS`` (a miss, an
  immune target), and runs ``subscript_burn.s`` / ``subscript_freeze.s`` /
  ``subscript_paralyze.s`` as for a ``SECONDARY_STATUS`` move (``_inflict``, non-direct: Fire
  is never burned, Ice never frozen and nothing frozen in sun, an existing status blocks,
  Limber blocks a paralysis silently; paralysis has no type immunity). ``_tri_attack``.
* Camouflage (``CAMOUFLAGE``): ``effect_script_0213.s`` -> ``BtlCmd_TryCamouflage``
  (battle_script.c:7997-8022), in the move script before ``TryMove`` (which cannot stop this
  ``RANGE_USER``, accuracy-0 move): it fails for Multitype (``ABILITY_MULTITYPE``) or when
  either of the user's types already is ``TERRAIN_TYPES[min(terrain, TERRAIN_SPECIAL)]``,
  else writes that type to both (``_camouflage``). ``step`` restores the party types when
  the battler leaves (``BattleSystem_InitBattleMon`` re-reads them, battle_lib.c:114-115).
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
* Life Orb (``LIFE_ORB``): ``life_orb_recoil``, run by ``step`` once per move after its
  hits (``AFTER_MOVE_HIT_STATE_LIFE_ORB``) and after Pursuit's pre-switch hit; its damage
  multiplier is in ``damage.life_orb``.
* Resist berries (``RESIST_BERRY``): ``subscript_update_hp.s:9`` calls
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
  script.c:4214): fail when the user's side has it, else 5 turns, more with Light Clay
  (``_screen``); the damage halving is in ``damage.py``, the countdown in ``step``.
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
* Cute Charm (battle_lib.c:4339-4355, same function): a living attacker that is not in love,
  a contact move that was not ineffective and dealt damage, a defender still alive after the
  hit, then the same roll (``outcome.contact_proc``, the last ``&&`` operand). It runs
  ``subscript_infatuate.s`` as an ability side effect on the attacker: ``BtlCmd_TryAttract``
  (battle_script.c:5727-5742) fails silently (:60-65) on equal genders or a genderless
  battler, else the attacker falls in love with the holder (``infatuated``); its message
  (:17, ``TAG_NICKNAME_ABILITY_NICKNAME`` with ``BTLSCR_MSG_BATTLER_TEMP``, the holder)
  reveals the holder's ability. Oblivious (:5) is an unsupported ability.
* Attract (``ATTRACT``): ``effect_script_0120.s``, a direct TO_DEFENDER effect (no ON_HIT:
  ``BattleSystem_TriggerPrimaryEffect`` needs a living defender, battle_lib.c:1492-1505)
  running ``subscript_infatuate.s``: ``BtlCmd_TryAttract`` as above, failing on a target
  already in love too ("But it failed!", :60-62). No type chart (a status move), and Protect
  blocks it (``MOVE_FLAG_CAN_PROTECT``). ``_attract``; the status check is in ``step``.
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
  3710), after the held-item checks and the Sticky Barb transfer of ``AfterMoveEffects``
  (``thaw_target``, run by ``step``): a frozen defender that is alive and took damage from a
  Fire-type move thaws (an Aspear or Lum Berry it holds cures the freeze first). The user's
  own thaw (Flame Wheel) is in ``step``.
* King's Rock, Razor Fang (``KINGS_ROCK``): ``_kings_rock``, per hit after the contact
  ability (``CheckExtraFlinch``).
* Light Clay (``LIGHT_CLAY``): ``_screen``. Shell Bell (``SHELL_BELL``): ``shell_bell_heal``,
  run by ``step`` after the move's hits (and after Pursuit's hit). Sticky Barb
  (``STICKY_BARB``): ``sticky_barb_transfer``, run by ``step`` after every hit's item checks
  (and after Pursuit's hit); its end-of-turn damage is in ``step``.

Unreachable branches are not modelled: Substitute, Mist, Safeguard and Uproar
come from unsupported moves; Clear Body, Hyper Cutter, Immunity, Leaf Guard, Liquid Ooze,
Magic Guard, Skill Link, Own Tempo, Normalize and Color Change are unsupported abilities;
Grip Claw is an unsupported item. Shield Dust (``_shield_dust``) and the weather moves
(``_weather``) are E2 batch 3's; Serene Grace acts on the secondary chance
(``outcome.secondary_chance``), Reckless on the damage (``damage.script_power_mul``),
Levitate and Scrappy on the type chart (``damage.levitated``, ``damage.identified``) and
Dry Skin's Water absorb in ``step._use_move``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import (
    PRESENT_HEAL,
    STATUS,
    damage_for_roll,
    decomp_abilities,
    divide,
    effectiveness,
    grounded,
    identified,
    ignorable,
    levitated,
    move_type_of,
    spikes_loss,
    stealth_rock_loss,
    weather_suppressed,
)
from nuzline.engine.ids import (
    ABILITY_MULTITYPE,
    MOVE_FAKE_OUT,
    MOVE_STRUGGLE,
    RESIST_ANY_NORMAL,
    SPIKES_MAX,
    TAILWIND_TURNS,
    TERRAIN_TYPES,
    TOXIC_SPIKES_MAX,
    TRI_ATTACK_STATUSES,
    WEATHER_ABILITY_IDS,
    WEATHER_TURNS,
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
from nuzline.engine.order import move_effects, terrain_index, terrain_known
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
    # BattleControllerPlayer_UpdateHP ran for this hit with a nonzero damage: the defender's
    # physical / specialDamageTaken (bcp.c:3411-3424), which King's Rock reads (MoveHit.damaged).
    update_hp: jnp.ndarray
    hit_index: jnp.ndarray  # the hit's index (per-hit outcome fields)


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


def _shield_dust(h: _Hit) -> jnp.ndarray:
    """The target's Shield Dust as the attacker sees it (``Battler_IgnorableAbility``) stops
    every ``SIDE_EFFECT_TYPE_INDIRECT`` effect on it, a damaging move's secondary effect
    (``ON_HIT`` or rolled), silently: the stat drop of ``BtlCmd_ChangeStatStage``
    (battle_script.c:2918-2921, ``jumpBlocked``), and the checks of subscript_paralyze.s:13,
    subscript_burn.s:27, subscript_freeze.s:11, subscript_poison.s:20,
    subscript_badly_poison.s:38, subscript_fall_asleep.s:31, subscript_confuse.s:15 and
    subscript_flinch_mon.s:9 (King's Rock's flinch included, bcp.c:4646)."""
    return h.target_ignorable == AbilityId.SHIELD_DUST


def _secondary_stat_foe(tables, state, h: _Hit) -> BattleState:
    recipient_ok = h.certain | _alive(state, h.target_side, h.target)
    cond = (h.effect == _E.SECONDARY_STAT_FOE) & h.effective & h.proc & recipient_ok
    cond = cond & ~_shield_dust(h)
    return _foe_stage(state, h, cond)


def can_have_status(tables: Tables, state: BattleState, side, slot, status) -> jnp.ndarray:
    """No status yet and no type (or weather) immunity to ``status``. Sun stops a freeze
    only without a weather lock: ``subscript_freeze.s:6`` runs ``CheckIgnoreWeather`` before
    the sun check at :7, and ``BtlCmd_CheckIgnoreWeather`` (battle_script.c:9195-9206) jumps
    past it while a battler on the field with HP left has Cloud Nine or Air Lock
    (``damage.weather_suppressed``, read on ``state``: the field when the effect runs)."""
    types = state.types[side, slot]
    sun = (state.weather == Weather.SUN) & ~weather_suppressed(tables, state)
    has = lambda t: jnp.any(types == t)
    immune = (
        ((status == Status.BURN) & has(Type.FIRE))
        | (
            ((status == Status.POISON) | (status == Status.TOXIC))
            & (has(Type.POISON) | has(Type.STEEL))
        )
        | ((status == Status.FREEZE) & (has(Type.ICE) | sun))
    )
    return (state.status[side, slot] == Status.NONE) & ~immune


def _inflict(
    tables: Tables, state: BattleState, h: _Hit, outcome: Outcome, cond, direct
) -> BattleState:
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
    # Shield Dust, after Insomnia / Limber (Leaf Guard, Water Veil and the like in between are
    # unsupported), for an indirect effect only.
    blocked = blocked | limber | (cond & ~jnp.asarray(direct, bool) & _shield_dust(h))
    # subscript_fall_asleep.s:43-45 (:14-16 for a move effect): a target not asleep yet stays
    # awake while FIELD_CONDITION_UPROAR is set (Soundproof, which skips it, is unsupported);
    # a direct effect then ends in MOVE_STATUS_NO_MORE_WORK (``step``: ``kept_awake``), an
    # indirect one silently (:_292).
    blocked = blocked | (cond & (status == Status.SLEEP) & uproar_on(state))
    cond = cond & ~blocked & can_have_status(tables, state, h.target_side, h.target, status)
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
    return _inflict(tables, state, h, outcome, (h.effect == _E.STATUS_FOE) & h.effective, True)


def _secondary_status(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.SECONDARY_STATUS) & h.effective & h.proc
    return _inflict(tables, state, h, outcome, cond, False)


def _tri_attack(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``TRI_ATTACK``: the drawn status (``outcome.tri_attack``) as a secondary status at the
    move's chance (module docstring)."""
    statuses = jnp.asarray(TRI_ATTACK_STATUSES, _I32)
    status = statuses[jnp.clip(outcome.tri_attack[h.side].astype(_I32), 0, len(statuses) - 1)]
    cond = (h.effect == _E.TRI_ATTACK) & h.effective & h.proc
    return _inflict(
        tables, state, h._replace(param=h.param.at[0].set(status)), outcome, cond, False
    )


def _camouflage(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``CAMOUFLAGE``: ``BtlCmd_TryCamouflage`` (module docstring). An unknown terrain
    (``order.terrain_known``) writes nothing (coverage refuses such a fight)."""
    new = jnp.asarray(TERRAIN_TYPES, _I32)[terrain_index(state)]
    types = state.types[h.side, h.user]
    multitype = decomp_abilities(tables, state)[h.side, h.user] == ABILITY_MULTITYPE
    cond = (
        (h.effect == _E.CAMOUFLAGE)
        & h.landed
        & terrain_known(state)
        & ~multitype
        & ~jnp.any(types.astype(_I32) == new)
    )
    value = jnp.where(cond, new, types.astype(_I32)).astype(types.dtype)
    return state._replace(types=state.types.at[h.side, h.user].set(value))


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
    # :8-9: both flinch effects are indirect, so Shield Dust stops them after Inner Focus.
    cond = reached & ~inner_focus & ~_shield_dust(h)
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
    state = _inflict(tables, state, h, outcome, fang & h.proc, False)
    cond = (
        fang
        & (outcome.fang_flinch[h.side] != 0)
        & _alive(state, h.target_side, h.target)
        & ~h.target_moved
        & (h.target_ignorable != AbilityId.INNER_FOCUS)
        & ~_shield_dust(h)
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


def _present(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Present's heal path: the target regains ``Divide(maxHP, 4)``, capped, unless it is at
    full HP (module docstring). ``landed``, not ``effective``: immunities are ignored."""
    hp = state.hp[h.target_side, h.target].astype(_I32)
    max_hp = state.max_hp[h.target_side, h.target].astype(_I32)
    heal = (h.effect == _E.PRESENT) & (outcome.present[h.side] == PRESENT_HEAL)
    cond = heal & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + jnp.maximum(max_hp // 4, 1), max_hp)
    return _set(state, "hp", (h.target_side, h.target), healed, cond)


def recovers_at_full_hp(tables, state: BattleState, side, move, outcome: Outcome) -> jnp.ndarray:
    """``bool``: ``side``'s ``move``, if it lands, runs ``subscript_recover_hp.s`` on a
    recipient at full HP: Recover on its user (``subscript_recover_half_max_hp.s:16``),
    Present's heal on its target (``subscript_present_heal.s:9``). That subscript's ``_036``
    ("HP is full!") sets ``MOVE_STATUS_NO_MORE_WORK``, so
    ``BattleControllerPlayer_CheckMoveFailure`` (bcp.c:3313-3315) jumps to
    ``LOOP_SPREAD_MOVES`` and the move's ``AfterMoveEffects`` (bcp.c:3638-3769: the attacker's
    and defender's held items, ``HELD_ITEM_STATUS``) never run (``step._use_move``)."""
    side = jnp.asarray(side, _I32)
    effect = tables.move_effect[jnp.asarray(move, _I32)]
    user, target = state.active[side], state.active[1 - side]
    user_full = state.hp[side, user] == state.max_hp[side, user]
    target_full = state.hp[1 - side, target] == state.max_hp[1 - side, target]
    heal = (effect == _E.PRESENT) & (outcome.present[side] == PRESENT_HEAL)
    # Roost runs the same subscript (subscript_roost.s:10).
    recovers = (effect == _E.RECOVER) | (effect == _E.ROOST)
    return (recovers & user_full) | (heal & target_full)


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
    # Rock Head; Magic Guard skips the subscript (subscript_recoil_1_2.s, _1_4.s:6, CheckAbility).
    rock_head = state.ability[h.side, h.user] == AbilityId.ROCK_HEAD
    magic_guard = state.ability[h.side, h.user] == AbilityId.MAGIC_GUARD
    cost = jnp.where(
        (h.effect == _E.RECOIL) & h.effective & ~rock_head & ~magic_guard, _percent_of_dealt(h), 0
    )
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


# Ranges whose decomp defender is the user itself in singles (``BattleSystem_Defender``,
# battle_lib.c:1665-1683): such a move records nothing on the foe's ``move_hit``.
SELF_DEFENDER_RANGES = (
    MoveTarget.RANGE_SINGLE_TARGET_SPECIAL,
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
)


def self_defender(tables: Tables, move) -> jnp.ndarray:
    """The move's decomp defender is its user: a ``SELF_DEFENDER_RANGES`` range or a
    non-Ghost Curse (``targets_user``)."""
    move_range = tables.move_target[move]
    in_range = jnp.isin(move_range, jnp.asarray(SELF_DEFENDER_RANGES, move_range.dtype))
    return in_range | targets_user(tables, move)


def protect_blocks(tables: Tables, move, unleash) -> jnp.ndarray:
    """``BattleControllerPlayer_CheckMoveHitOverrides`` (battle_controller_player.c:
    3001-3014): a protecting defender stops ``move`` when it has ``MOVE_FLAG_CAN_PROTECT``
    (``MoveFlag.PROTECT``), the foe is its defender (not a ``self_defender`` move, which a
    non-Ghost Curse is) and it is not the non-final turn of a multi-turn move
    (``Move_IsMultiTurn``, battle_lib.c:2975-2992: of the supported moves only Bide, whose
    final turn is the unleash, ``unleash``, when ``subscript_bide_end.s`` sets
    ``SYSCTL_LAST_OF_MULTI_TURN`` and the defender is the last hitter, bcp.c:2696-2722)."""
    flagged = (tables.move_flags[move] & MoveFlag.PROTECT) != 0
    bide = tables.move_effect[move] == _E.BIDE
    return flagged & jnp.where(bide, unleash, ~self_defender(tables, move))


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
    """``Battler_UnlockMoveChoice`` (battle_lib.c:1834-1841) on ``side`` where ``cond``: the move
    lock, Bide's turns and stored damage (hidden state, 0 outside a Bide) and the Rollout
    count go to 0. It clears ``VOLATILE_CONDITION_MOVE_LOCKED``, not ``moveLockedInto`` nor
    ``VOLATILE_CONDITION_UPROAR``: an uproaring battler stays locked into its move
    (``Battler_CanPickCommand``, battle_lib.c:6224-6235), so ``locked_move`` stays while
    ``uproar`` is up (ruling B5-5)."""
    for field in ("bide_turns", "bide_damage", "rollout_count"):
        state = _set(state, field, side, 0, cond)
    return _set(state, "locked_move", side, 0, cond & (state.uproar[side] == 0))


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


def _cure_paralysis(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """SmellingSalt (effect_script_0171.s:12): against a defender paralyzed when the script ran,
    an ON_HIT indirect effect (``BattleSystem_TriggerSecondaryEffect``, battle_lib.c:1524-1529:
    not ``MOVE_STATUS_NO_EFFECTS``) runs subscript_heal_target_paralysis.s, which cures a
    defender with HP left (:5-6). The hit leaves the status alone, so it is still paralysis."""
    cond = (
        (h.effect == _E.DOUBLE_POWER)
        & (h.param[1] == 1)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.status[h.target_side, h.target] == Status.PARALYSIS)
    )
    return _set_status(state, h.target_side, h.target, Status.NONE, cond)


# subscript_swagger.s:6: Attack +2 below +6 (the stage 12 test).
SWAGGER_STAGES = 2


def swagger_fails(state: BattleState, target_side) -> jnp.ndarray:
    """``bool``: subscript_swagger.s:7-8 fails Swagger (``MOVE_STATUS_FAILED``) on a defender
    at +6 Attack that is already confused."""
    return (state.stages[target_side, Stat.ATK] >= STAGE_MAX) & (state.confusion[target_side] > 0)


def _swagger(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Swagger (effect_script_0118.s, a direct ON_HIT effect; subscript_swagger.s): Attack +2
    below +6 (``UPDATE_STAT_STAGE``), then subscript_confuse.s unless the defender is confused
    (:13, no message): ``Random 3, 2`` turns (``outcome.duration``). Substitute (:5), Own Tempo
    and Safeguard (subscript_confuse.s:12, :26) come from unsupported moves and abilities; the
    failure (``swagger_fails``) skips AfterMoveEffects (``step``)."""
    cond = (h.effect == _E.SWAGGER) & h.landed & _alive(state, h.target_side, h.target)
    below = state.stages[h.target_side, Stat.ATK] < STAGE_MAX
    state = change_stage(state, h.target_side, Stat.ATK, SWAGGER_STAGES, cond & below)
    confuse = cond & (state.confusion[h.target_side] == 0)
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], confuse)


DRAGON_DANCE_STAGES = ((Stat.ATK, 1), (Stat.SPE, 1))  # subscript_user_atk_and_speed_up_1_stage.s


def dragon_dance_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: both of the user's stages are at +6: the subscript prints "won't go higher"
    and sets ``MOVE_STATUS_NO_MORE_WORK`` (:5-6, :21-29), which skips AfterMoveEffects."""
    return (state.stages[side, Stat.ATK] >= STAGE_MAX) & (state.stages[side, Stat.SPE] >= STAGE_MAX)


def _dragon_dance(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Dragon Dance (effect_script_0212.s: direct TO_ATTACKER, the user has HP,
    battle_lib.c:1492-1504): Attack +1 and Speed +1, each skipped on its own at +6 (:13-16,
    ``ChangeStatStage``'s ``jumpNoChange``)."""
    cond = (h.effect == _E.DRAGON_DANCE) & h.landed & _alive(state, h.side, h.user)
    for stat, delta in DRAGON_DANCE_STAGES:
        state = change_stage(state, h.side, stat, delta, cond)
    return state


def _roost(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Roost (subscript_roost.s, a direct ON_HIT TO_ATTACKER effect): ``Divide(maxHP, 2)``
    through subscript_recover_hp.s (``recovers_at_full_hp``: "HP is full!" at full HP); the
    roosting flag is ``step``'s (``roosted``)."""
    hp = state.hp[h.side, h.user].astype(_I32)
    max_hp = state.max_hp[h.side, h.user].astype(_I32)
    cond = (h.effect == _E.ROOST) & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + divide(max_hp, 2), max_hp)
    return _set(state, "hp", (h.side, h.user), healed, cond)


def _side_layers(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Tailwind (subscript_tailwind_start.s:11) on the user's side; Spikes / Toxic Spikes
    (``BtlCmd_TrySpikes`` / ``TryToxicSpikes``) one layer on the defender's side; each fails when
    full (``primary_effect_fails``)."""
    state = _set(
        state,
        "tailwind",
        h.side,
        TAILWIND_TURNS,
        (h.effect == _E.TAILWIND) & h.landed & (state.tailwind[h.side] == 0),
    )
    for effect, field, cap in (
        (_E.SPIKES, "spikes", SPIKES_MAX),
        (_E.TOXIC_SPIKES, "toxic_spikes", TOXIC_SPIKES_MAX),
    ):
        layers = getattr(state, field)[h.target_side].astype(_I32)
        state = _set(
            state,
            field,
            h.target_side,
            layers + 1,
            (h.effect == effect) & h.landed & (layers < cap),
        )
    return state


def _destiny_bond(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Destiny Bond (effect_script_0098.s: an ON_HIT indirect effect): subscript_destiny_bond.s:9
    sets the user's flag; there is no failure on a repeated use."""
    return _set(state, "destiny_bond", h.side, 1, (h.effect == _E.DESTINY_BOND) & h.landed)


def _uproar(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Uproar (effect_script_0159.s:5-6): for a user not in an uproar yet, an ON_HIT indirect
    effect (``BattleSystem_TriggerSecondaryEffect``, battle_lib.c:1524-1529: not
    ``MOVE_STATUS_NO_EFFECTS``, so a hit that was not ineffective) runs subscript_uproar.s:
    ``moveLockedInto`` = moveCur (:5), ``Random 3, 3`` turns (:6-8, ``outcome.duration``, 3-6) and
    the battler's ``FIELD_CONDITION_UPROAR`` bit (:9-11, ``uproar != 0``)."""
    cond = (h.effect == _E.UPROAR) & h.effective & (state.uproar[h.side] == 0)
    state = _set(state, "locked_move", h.side, h.move, cond)
    return _set(state, "uproar", h.side, outcome.duration[h.side], cond)


def uproar_on(state: BattleState) -> jnp.ndarray:
    """``bool``: ``FIELD_CONDITION_UPROAR`` is set: some battler is in an uproar
    (``BattleState.uproar``)."""
    return jnp.any(state.uproar != 0)


def primary_effect_fails(tables, state: BattleState, side, target_side, move) -> jnp.ndarray:
    """``bool``: ``side``'s ``move``, if it lands, has a direct effect that ends it with
    ``MOVE_STATUS_FAILED`` or ``_NO_MORE_WORK``, which ``BattleControllerPlayer_CheckMoveFailure``
    (bcp.c:3312-3331) sends past AfterMoveEffects: Swagger (``swagger_fails``), Dragon Dance
    (``dragon_dance_fails``). ``target_side``: the defender's side."""
    effect = tables.move_effect[jnp.asarray(move, _I32)]
    return (
        ((effect == _E.SWAGGER) & swagger_fails(state, target_side))
        | ((effect == _E.DRAGON_DANCE) & dragon_dance_fails(state, side))
        # subscript_tailwind_start.s:5, :18 and the effect scripts' TrySpikes / TryToxicSpikes
        # jumps (effect_script_0112.s, _0249.s): MOVE_STATUS_FAILED.
        | ((effect == _E.TAILWIND) & (state.tailwind[side] != 0))
        | ((effect == _E.SPIKES) & (state.spikes[target_side] >= SPIKES_MAX))
        | ((effect == _E.TOXIC_SPIKES) & (state.toxic_spikes[target_side] >= TOXIC_SPIKES_MAX))
        # subscript_mean_look.s:7-8, :48: MOVE_STATUS_FAILED.
        | ((effect == _E.MEAN_LOOK) & mean_look_fails(state, target_side))
    )


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
        & ~_shield_dust(h)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


def _aqua_ring(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.AQUA_RING) & h.landed & (state.aqua_ring[h.side] == 0)
    return _set(state, "aqua_ring", h.side, 1, cond)


def _ingrain(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Ingrain (effect_script_0181.s:5-6): the user's ``MOVE_EFFECT_INGRAIN``. Its script fails a
    user that has it already (:5, :19-20: MOVE_STATUS_FAILED, ``step._use_move``'s ``failed``)."""
    cond = (h.effect == _E.INGRAIN) & h.landed & (state.ingrain[h.side] == 0)
    return _set(state, "ingrain", h.side, 1, cond)


def mean_look_fails(state: BattleState, target_side) -> jnp.ndarray:
    """``bool``: subscript_mean_look.s:7-8 fails (MOVE_STATUS_FAILED, :48) on a defender already
    trapped by Mean Look or behind a substitute (``CheckSubstitute``)."""
    return (state.mean_look[target_side] != 0) | (state.substitute_hp[target_side] != 0)


def _mean_look(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Mean Look, Block, Spider Web (effect_script_0106.s: a direct ``CHECK_HP | TO_DEFENDER``
    subscript_mean_look.s, run for a defender with HP, battle_lib.c:1491-1504, on a move that
    connected: accuracy 0, so only Protect stops it): the defender's
    ``VOLATILE_CONDITION_MEAN_LOOK`` and its ``meanLookTarget`` (:9-10): the script ORs the
    attacker's battler into the 2-bit field (``OPCODE_FLAG_ON``), whose old value nothing clears
    but a switch-in or a faint of the trapped battler (ruling B7-15); kept as ``1 + raw``, 0 for a
    zeroed field. A failure (``mean_look_fails``) skips AfterMoveEffects
    (``primary_effect_fails``)."""
    cond = (
        (h.effect == _E.MEAN_LOOK)
        & h.landed
        & _alive(state, h.target_side, h.target)
        & ~mean_look_fails(state, h.target_side)
    )
    state = _set(state, "mean_look", h.target_side, 1, cond)
    raw = jnp.maximum(state.mean_look_target[h.target_side].astype(_I32) - 1, 0)
    value = 1 + (raw | jnp.asarray(h.side, _I32))  # battler b is on side b in singles
    return _set(state, "mean_look_target", h.target_side, value, cond)


def _imprison(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Imprison (effect_script_0192.s, ``BtlCmd_TryImprison`` battle_script.c:6674-6727): the
    user's ``MOVE_EFFECT_IMPRISON``. The script fails (MOVE_STATUS_FAILED, :10) a user that has it
    or that shares no move with a foe (``imprison_fails``, ``step._use_move``'s ``failed``); the
    foes' ``MOVE_EFFECT_IMPRISONED`` is set before either test, whenever the script runs
    (``step._use_move``)."""
    cond = (h.effect == _E.IMPRISON) & h.landed & (state.imprison[h.side] == 0)
    return _set(state, "imprison", h.side, 1, cond)


def imprison_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``BtlCmd_TryImprison`` jumps to failure (battle_script.c:6692-6693, :6721-6722):
    ``side``'s battler already has ``MOVE_EFFECT_IMPRISON``, or none of its moves is a move of a
    battler of the other side (the loop over battlers, :6697-6716: their ``battleMons`` moves,
    whatever their HP; doubles convention 4, battler b is on side b in singles)."""
    own = state.moves[side, state.active[side]].astype(_I32)
    shared = jnp.bool_(False)
    for battler in range(state.active.shape[0]):
        theirs = state.moves[battler, state.active[battler]].astype(_I32)
        match = (own[:, None] == theirs[None, :]) & (own[:, None] != 0) & (theirs[None, :] != 0)
        shared = shared | ((jnp.asarray(side) != battler) & jnp.any(match))
    return (state.imprison[side] != 0) | ~shared


def camouflaged(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``side``'s active battler has ``MOVE_EFFECT_CAMOUFLAGE`` (effect_script_0213.s:6).
    ``BtlCmd_TryCamouflage`` fails a battler that has the terrain's type already, so a Camouflage
    that ran left both types on a type the personal row lacks, and nothing else writes ``types``:
    the bit is set exactly when the types differ from the personal row's."""
    mon = state.active[side]
    row = tables.species_form_row[state.species[side, mon], state.form[side, mon]]
    return jnp.any(state.types[side, mon] != tables.species_types[row].astype(state.types.dtype))


# The moveEffectsMask bits (moves.h:7-36) the engine keeps, by field, besides Ingrain.
MOVE_EFFECT_MASK_FIELDS = (
    "leech_seed",
    "lock_on",
    "perish_song",
    "yawn",
    "imprison",
    "imprisoned",
    "mud_sport",
    "water_sport",
    "aqua_ring",
)


def ingrain_anchors(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: subscript_force_target_to_switch_or_flee.s:7 finds the defender's
    ``moveEffectsMask`` equal (``OPCODE_EQU``) to ``MOVE_EFFECT_INGRAIN``: ``side``'s battler is
    ingrained and has no other bit of the mask: none of ``MOVE_EFFECT_MASK_FIELDS`` (the Leech
    Seed bits sit on the seeded battler, subscript_leech_seed_start.s:13-14) and no Camouflage
    (``camouflaged``). The other bits come from unsupported moves (Fly, Dig, Dive, Shadow Force,
    Minimize, Charge, Grudge, Gastro Acid, Miracle Eye, Power Trick, Heal Block, Embargo, Magnet
    Rise); each joins the tuple with its move."""
    other = camouflaged(tables, state, side)
    for name in MOVE_EFFECT_MASK_FIELDS:
        other = other | (getattr(state, name)[side] != 0)
    return (state.ingrain[side] != 0) & ~other


def _mud_sport(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.MUD_SPORT) & h.landed & (state.mud_sport[h.side] == 0)
    return _set(state, "mud_sport", h.side, 1, cond)


SCREEN_TURNS = 5  # NUM_SCREEN_TURNS (include/constants/battle.h:142)


def _screen(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Light Screen, Reflect: ``BtlCmd_TryLightScreen`` / ``TryReflect`` (script.c:4214,
    :4257) fail when the user's side has it, else set it for ``NUM_SCREEN_TURNS``, plus the
    hold param of the user's Light Clay (``HOLD_EFFECT_EXTEND_SCREENS``, script.c:4229, :4272:
    ``Battler_HeldItemEffect`` / ``Battler_HeldItemPower`` of the attacker)."""
    item = state.item[h.side, h.user]
    clay = tables.item_effect[item] == ItemId.LIGHT_CLAY
    turns = SCREEN_TURNS + jnp.where(clay, tables.item_param[item, 0].astype(_I32), 0)
    for effect, field in ((_E.LIGHT_SCREEN, "light_screen"), (_E.REFLECT, "reflect")):
        cond = (h.effect == effect) & h.landed & (getattr(state, field)[h.side] == 0)
        state = _set(state, field, h.side, turns, cond)
    return state


def _brick_break(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Brick Break: a direct ``BREAK_SCREENS`` effect (lib.c:1481-1510) that still applies
    on an immune target (``MOVE_STATUS_DID_NOT_AFFECT``) but not on a miss;
    ``BtlCmd_TryBreakScreens`` (script.c:6550) clears both screens of the defender's side."""
    cond = (h.effect == _E.BRICK_BREAK) & h.landed
    state = _set(state, "reflect", h.target_side, 0, cond)
    return _set(state, "light_screen", h.target_side, 0, cond)


def _weather(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Sandstorm, Rain Dance, Sunny Day, Hail (``WEATHER``; effect_script_0115.s, _0136.s,
    _0137.s, _0164.s, run before ``TryMove``, which cannot stop this accuracy-0
    ``RANGE_FIELD`` move): ``MOVE_STATUS_FAILED`` when its weather is up already, either kind
    (``move_param[m, 1] == 1``) or, for Hail, only a temporary one (``weather_timer != 0``);
    else ``FIELD_CONDITION_WEATHER`` is cleared and the temporary weather set with
    ``WEATHER_TURNS`` turns plus the param of the user's matching weather rock
    (``CheckItemHoldEffect`` / ``GetItemEffectParam`` on the attacker). The failure skips
    ``AfterMoveEffects`` (``step._use_move``)."""
    weather = h.param[0]
    item = state.item[h.side, h.user]
    rock = (tables.item_effect[item] == ItemId.WEATHER_ROCK) & (
        tables.item_param[item, 1].astype(_I32) == weather
    )
    turns = WEATHER_TURNS + jnp.where(rock, tables.item_param[item, 0].astype(_I32), 0)
    cond = (h.effect == _E.WEATHER) & h.landed & ~weather_fails(state, h.param)
    return state._replace(
        weather=jnp.where(cond, weather, state.weather).astype(state.weather.dtype),
        weather_timer=jnp.where(cond, turns, state.weather_timer).astype(state.weather_timer.dtype),
    )


def weather_fails(state: BattleState, param) -> jnp.ndarray:
    """``bool``: a ``WEATHER`` move with ``param`` (``move_param[m]``) fails on this field
    (``_weather``)."""
    up = state.weather == param[0]
    temporary = state.weather_timer != 0
    return up & ((param[1] == 1) | temporary)


def armored(state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: ``target_side``'s active battler has Battle Armor or Shell Armor as
    ``side``'s sees it (``Battler_IgnorableAbility``): ``BattleSystem_CalcCriticalMulti``
    (battle_lib.c:7135-7136) never crits it, whatever the roll (``Outcome.crit``)."""
    seen = ignorable(state, side, target_side)
    return (seen == AbilityId.SHELL_ARMOR) | (seen == AbilityId.BATTLE_ARMOR)


def _eat_berry(
    tables, state, h: _Hit, outcome: Outcome, foe_custap
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, pluck_flag)`` (``hooks.pluck_berry``)."""
    # Local import: hooks imports this module.
    from nuzline.engine.hooks import pluck_berry

    cond = (h.effect == _E.EAT_BERRY) & h.effective
    return pluck_berry(tables, state, h.side, h.target_side, cond, foe_custap)


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


def _attractable(state: BattleState, side, slot, foe_side, foe_slot) -> jnp.ndarray:
    """``BtlCmd_TryAttract`` (battle_script.c:5727-5742) succeeds for target ``side`` /
    ``slot`` and source ``foe_side`` / ``foe_slot``: different genders, neither genderless
    (``GENDER_NONE`` is 2, ``BattleState.gender``), and the target not in love yet."""
    mine, theirs = state.gender[side, slot], state.gender[foe_side, foe_slot]
    return (mine != theirs) & (mine != 2) & (theirs != 2) & (state.infatuated[side] == 0)


def _attract(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Attract: the living target of the opposite gender falls in love with the user."""
    cond = (
        (h.effect == _E.ATTRACT)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & _attractable(state, h.target_side, h.target, h.side, h.user)
    )
    return _set(state, "infatuated", h.target_side, 1, cond)


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
        & can_have_status(tables, state, h.side, h.user, status)
    )
    # Rough Skin (battle_lib.c:4257-4271): no roll; subscript_rough_skin.s takes
    # BattleSystem_Divide(maxHP, 8) from the attacker (it may faint).
    # battle_lib.c:4259: not on a Magic Guard attacker (Battler_Ability).
    rough = (
        (ability == AbilityId.ROUGH_SKIN)
        & hit
        & (state.ability[h.side, h.user] != AbilityId.MAGIC_GUARD)
    )
    max_hp = state.max_hp[h.side, h.user].astype(_I32)
    hp = state.hp[h.side, h.user].astype(_I32)
    hurt = jnp.where(rough, jnp.maximum(hp - jnp.maximum(max_hp // 8, 1), 0), hp)
    state = state._replace(hp=state.hp.at[h.side, h.user].set(hurt.astype(state.hp.dtype)))
    # The status message and Rough Skin's message tag the holder's ability
    # (``hooks.REVEALS_ON``). hooks imports this module, so the import is local.
    from nuzline.engine.hooks import CONTACT_DAMAGE, CONTACT_STATUS, reveal_ability

    # Cute Charm (battle_lib.c:4339-4355): the attacker not in love yet and the defender
    # alive, then the roll and subscript_infatuate.s (BtlCmd_TryAttract fails silently).
    charm = (
        (ability == AbilityId.CUTE_CHARM)
        & hit
        & _alive(state, h.target_side, h.target)
        & h.contact_proc
        & _attractable(state, h.side, h.user, h.target_side, h.target)
    )
    state = _set(state, "infatuated", h.side, 1, charm)
    state = reveal_ability(state, h.target_side, h.target, CONTACT_STATUS, cond | charm)
    state = reveal_ability(state, h.target_side, h.target, CONTACT_DAMAGE, rough)
    return _set_status(state, h.side, h.user, status, cond)


def _kings_rock(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``BattleControllerPlayer_CheckExtraFlinch`` (battle_controller_player.c:4630-4653), the
    last step of ``AfterMoveMessage`` (``ONE_HIT_EXTRA_FLINCH`` / ``MULTI_HIT_EXTRA_FLINCH``,
    :3524, :3590), after the contact ability, for every hit: the attacker holds a
    ``HOLD_EFFECT_SOMETIMES_FLINCH`` item (King's Rock, Razor Fang), the hit was not
    ``MOVE_STATUS_NO_EFFECTS`` and the defender took damage this action (``update_hp``: its
    ``physicalDamageTaken`` / ``specialDamageTaken``, set by ``UpdateHP``, bcp.c:3411-3424),
    ``RandNext() % 100 < param`` (``outcome.flinch_item``), the move has
    ``MOVE_FLAG_TRIGGERS_KINGS_ROCK`` and the defender has HP left. ``subscript_flinch_mon.s``
    then flinches a target that has not moved (:5), unless its Inner Focus (as the attacker
    sees it, :7; a silent block: not Fake Out, not a direct effect) or its Shield Dust (:9,
    ``_shield_dust``). Substitute (:6) is an unsupported move. A U-turn that
    switches makes its check with the newcomer's item (``kings_rock_flinch``, from ``step._use_move``)."""
    cond = h.landed & h.update_hp & (outcome.flinch_item[h.side, h.hit_index] != 0)
    return kings_rock_flinch(tables, state, h.side, h.target_side, h.move, cond & ~h.target_moved)


def kings_rock_flinch(tables, state, side, target_side, move, cond) -> BattleState:
    """``CheckExtraFlinch``'s flinch (``_kings_rock``) by ``side``'s active battler on
    ``target_side``'s where ``cond`` (the hit landed with damage, its ``flinch_item`` roll passed
    and the target has not moved): the attacker's item now (``Battler_HeldItemEffect(attacker)``,
    bcp.c:4633; after a U-turn switch the newcomer's, with no U-turn guard, :4636-4642), the move's
    ``MOVE_FLAG_TRIGGERS_KINGS_ROCK``, a living target, and subscript_flinch_mon.s's Inner Focus
    (:7) and Shield Dust (:9) as this attacker sees them (``Battler_IgnorableAbility``)."""
    user, target = state.active[side], state.active[target_side]
    item = state.item[side, user]
    flag = (tables.move_flags[move] & MoveFlag.KINGS_ROCK) != 0
    seen = ignorable(state, side, target_side)
    on = (
        jnp.asarray(cond)
        & (tables.item_effect[item] == ItemId.KINGS_ROCK)
        & flag
        & _alive(state, target_side, target)
        & (seen != AbilityId.INNER_FOCUS)
        & (seen != AbilityId.SHIELD_DUST)  # CheckExtraFlinch's side effect is INDIRECT (:4646)
    )
    old = state.flinched[target_side]
    return state._replace(
        flinched=state.flinched.at[target_side].set(jnp.where(on, 1, old).astype(old.dtype))
    )


def thaw_target(tables: Tables, state: BattleState, side, target_side, move, dealt, cond) -> tuple:
    """``(state, thawed)``: ``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (battle_controller_player.c:
    3708-3727), after the held-item checks and the Sticky Barb transfer of the hit
    (``step._use_move``): a frozen defender with HP left that took damage from a Fire-type
    ``move`` of ``side`` (``dealt``: its HP drop on this hit) thaws, where ``cond`` (the
    move's ``AfterMoveEffects`` run). ``target_side``: the defender's side."""
    target = state.active[target_side]
    thawed = (
        jnp.asarray(cond)
        & (state.status[target_side, target] == Status.FREEZE)
        & (move_type_of(tables, state, side, move) == Type.FIRE)
        & (jnp.asarray(dealt) > 0)
        & _alive(state, target_side, target)
    )
    return _set_status(state, target_side, target, Status.NONE, thawed), thawed


def sticky_barb_transfer(tables: Tables, state: BattleState, side, foe, move, cond) -> BattleState:
    """``side``'s attacker takes the defender's Sticky Barb where ``cond`` (the defender took
    damage this action: ``physicalDamageTaken`` / ``specialDamageTaken``):
    ``BattleSystem_TriggerHeldItemOnHit`` (battle_lib.c:5386-5398;
    ``AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT``, battle_controller_player.c:3698-3706, and
    ``subscript_pursuit.s:44``, :90) needs a defender holding a
    ``HOLD_EFFECT_DMG_USER_CONTACT_XFR`` item (alive or not), a living attacker whose
    ``heldItem`` is ``ITEM_NONE`` and a move with ``MOVE_FLAG_MAKES_CONTACT``; Knock Off,
    U-turn and the knocked-off mask come from unsupported moves.
    ``subscript_transfer_sticky_barb.s`` moves the item: the attacker holds it, the defender
    nothing (no ``RemoveItem``: ``item_consumed`` is untouched). ``foe``: the defender's side."""
    user, target = state.active[side], state.active[foe]
    barb = state.item[foe, target]
    contact = (tables.move_flags[move] & MoveFlag.CONTACT) != 0
    on = (
        jnp.asarray(cond)
        & (tables.item_effect[barb] == ItemId.STICKY_BARB)
        & _alive(state, side, user)
        & (state.item[side, user] == 0)
        & contact
    )
    state = _set(state, "item", (side, user), barb, on)
    state = forget_item(state, foe, target, on)
    return _set(state, "item", (foe, target), 0, on)


def sticky_barb_ready(tables: Tables, state: BattleState, side, foe, move) -> jnp.ndarray:
    """``bool``: ``side``'s attacker would take ``foe``'s Sticky Barb (``sticky_barb_transfer``'s
    tests but the damage flag)."""
    user, target = state.active[side], state.active[foe]
    contact = (tables.move_flags[move] & MoveFlag.CONTACT) != 0
    return (
        (tables.item_effect[state.item[foe, target]] == ItemId.STICKY_BARB)
        & _alive(state, side, user)
        & (state.item[side, user] == 0)
        & contact
    )


def pivot_item(
    tables: Tables, state: BattleState, side, foe, move, shell_dealt, move_hit, damaged
) -> BattleState:
    """``BattleSystem_TriggerHeldItemOnPivotMove`` (battle_lib.c:7336-7386) once, at U-turn's
    subscript_attack_then_switch_out.s:11: four independent ``if``s each overwrite the subscript,
    so only the last match runs: the defender's Sticky Barb (:7378-7385: a living attacker holding
    nothing, the defender's damage taken, a contact move; no U-turn guard, the knocked-off mask comes
    with Knock Off), else the defender's Jaboca Berry (:7368-7375, ``HOLD_EFFECT_RECOIL_PHYSICAL``,
    an unsupported item), else the attacker's Life Orb (:7357-7366: ``SYSCTL_MOVE_HIT``, not a
    status move, a living holder without Magic Guard) or Shell Bell (:7345-7355; one holder holds
    one of them). ``move_hit``: ``SYSCTL_MOVE_HIT``; ``damaged``: the defender's
    physical / specialDamageTaken; ``shell_dealt``: ``shellBellDamageDealt``."""
    barb = jnp.asarray(damaged) & sticky_barb_ready(tables, state, side, foe, move)
    state = sticky_barb_transfer(tables, state, side, foe, move, barb)
    hit = jnp.asarray(move_hit) & ~barb
    state = shell_bell_heal(tables, state, side, shell_dealt, hit)
    return life_orb_recoil(tables, state, side, hit & (tables.move_category[move] != STATUS))


def shell_bell_heal(tables: Tables, state: BattleState, side, dealt, cond) -> BattleState:
    """Shell Bell (``HOLD_EFFECT_HP_RESTORE_ON_DMG``) where ``cond`` (``SYSCTL_MOVE_HIT``):
    ``AFTER_MOVE_HIT_STATE_SHELL_BELL`` (battle_controller_player.c:4736-4752), before Life
    Orb's state, and ``TriggerHeldItemOnPivotMove`` after Pursuit's hit (battle_lib.c:
    7345-7355): ``dealt`` (``shellBellDamageDealt``, the move's damage to the foe, each hit
    capped at its HP) is nonzero and the holder is alive below full HP; it regains
    ``BattleSystem_Divide(dealt, param)`` through ``subscript_restore_a_little_hp.s``
    (``BATTLE_SUBSCRIPT_UPDATE_HP``, capped at max HP). The defender is the foe whenever
    ``dealt`` is nonzero (attacker != defender); U-turn is unsupported."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    item = state.item[side, mon]
    hp = state.hp[side, mon].astype(_I32)
    max_hp = state.max_hp[side, mon].astype(_I32)
    dealt = jnp.asarray(dealt, _I32)
    on = (
        jnp.asarray(cond)
        & (tables.item_effect[item] == ItemId.SHELL_BELL)
        & (dealt != 0)
        & (hp > 0)
        & (hp < max_hp)
    )
    heal = divide(dealt, tables.item_param[item, 0].astype(_I32))
    # "{0} restored a little HP using its {1}!" names the item (BTLSCR_MSG_BATTLER_TEMP, the
    # attacker: subscript_restore_a_little_hp.s:10, bcp.c:4745, battle_lib.c:7352).
    state = reveal_item(state, side, mon, on)
    return _set(state, "hp", (side, mon), jnp.minimum(hp + heal, max_hp), on)


def entry_hazards(tables: Tables, state: BattleState, side) -> BattleState:
    """subscript_hazards_check.s on ``side``'s active Pokemon as it switches in, skipped whole
    for a Magic Guard holder (:5, CheckAbility):

    * Toxic Spikes (:14-28, ``BtlCmd_CheckToxicSpikes``, battle_script.c:7629-7651) on a
      ``damage.grounded`` newcomer: a Poison type absorbs them (the side's layers go to 0);
      else one layer poisons, two badly poison (counter 0) through subscript_poison.s /
      subscript_badly_poison.s's Toxic Spikes branch (:5-10): not for its own Immunity, Leaf Guard
      in sun without a weather lock, a status or a Poison / Steel type (``can_have_status``;
      Safeguard is an unsupported move).
    * Spikes (:31, ``damage.spikes_loss``) on a grounded newcomer with HP, then Stealth Rock
      (:41, ``damage.stealth_rock_loss``) on one with HP, whatever its grounding; each capped at
      its HP."""
    from nuzline.engine.ids import ITEM_ABILITY_IDS

    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    guarded = state.ability[side, mon] == AbilityId.MAGIC_GUARD
    down = grounded(state, side, mon) & ~guarded
    layers = state.toxic_spikes[side].astype(_I32)
    poison_type = jnp.any(state.types[side, mon] == Type.POISON)
    absorbed = down & (layers > 0) & poison_type
    state = _set(state, "toxic_spikes", side, 0, absorbed)
    ability = decomp_abilities(tables, state)[side, mon]
    sun = (state.weather == Weather.SUN) & ~weather_suppressed(tables, state)
    status = jnp.where(layers >= 2, Status.TOXIC, Status.POISON)
    # subscript_poison.s:5-10 / subscript_badly_poison.s:5-10 (CheckAbility, before the status
    # and type checks): Immunity, or Leaf Guard in sun, jumps to "{0}'s {1} prevents
    # poisoning!" (subscript_poison.s:80-82, subscript_badly_poison.s:93-103), whose
    # TAG_NICKNAME_ABILITY runs BattleAI_SetAbility (battle_script.c:11798-11806).
    blocks = (ability == ITEM_ABILITY_IDS["Immunity"]) | (
        sun & (ability == ITEM_ABILITY_IDS["Leaf Guard"])
    )
    stopped = down & (layers > 0) & ~poison_type & blocks
    known = jnp.where(stopped & (side == 0), 1, state.ai_ability_known[mon])
    state = state._replace(
        ai_ability_known=state.ai_ability_known.at[mon].set(
            known.astype(state.ai_ability_known.dtype)
        )
    )
    poisoned = (
        down
        & (layers > 0)
        & ~poison_type
        & ~blocks
        & can_have_status(tables, state, side, mon, status)
    )
    state = _set_status(state, side, mon, status, poisoned)
    hp = state.hp[side, mon].astype(_I32)
    spikes = jnp.where((hp > 0) & ~guarded, jnp.minimum(spikes_loss(state, side, mon), hp), 0)
    hp = hp - spikes
    rock = jnp.where(
        (hp > 0) & ~guarded, jnp.minimum(stealth_rock_loss(tables, state, side, mon), hp), 0
    )
    return state._replace(hp=state.hp.at[side, mon].set((hp - rock).astype(state.hp.dtype)))


class MoveHit(NamedTuple):
    """What one hit leaves for the rest of the action (``play_hit``)."""

    # DEFENDER_SELF_TURN_FLAGS.focusItemActivated (bcp.c:3386-3393): set on a hit taken at
    # full HP holding a Focus Sash; it lasts until BattleContext_Init at the action's end
    # (bcp.c:3998, battle_lib.c:2014), so every later hit of the move is capped too.
    focus_armed: jnp.ndarray
    # ATTACKER_SELF_TURN_FLAGS SELF_TURN_FLAG_PLUCK_BERRY left set by a Pluck / Bug Bite
    # whose berry did nothing for the eater (hooks.pluck_berry).
    pluck_flag: jnp.ndarray
    # BattleControllerPlayer_UpdateHP ran for this hit (battleCtx->damage != 0, bcp.c:3340):
    # it sets SYSCTL_MOVE_HIT (:3441), which Life Orb's recoil and Shell Bell read, and the
    # defender's physical / specialDamageTaken (:3411-3424), which the Sticky Barb transfer
    # and the thaw read.
    update_hp: jnp.ndarray
    # The HP the target lost on this hit (thaw_target).
    dealt: jnp.ndarray
    # UpdateHP left the defender's physical / specialDamageTaken nonzero (bcp.c:3411-3424): the
    # damage after False Swipe's cap is not 0 (the Sticky Barb transfer reads it).
    damaged: jnp.ndarray
    # UpdateHP's ATTACKER_SELF_TURN_FLAGS.shellBellDamageDealt of this hit (bcp.c:3427-3431):
    # battleCtx->damage after the endure cap, capped at the target's HP (shell_bell_heal).
    shell_bell: jnp.ndarray
    # subscript_pursuit.s:21-29's shellBellDamageDealt for Pursuit's pre-switch hit: the
    # damage before CheckHoldOnWith1HP, capped at the target's HP.
    pursuit_shell_bell: jnp.ndarray
    # battleCtx->damage after the endure caps where UpdateHP ran (the defender's turnFlags
    # damage taken, bcp.c:3413-3434: Counter, Mirror Coat, Metal Burst), else 0.
    taken: jnp.ndarray
    # The hit reached its effects: landed and not MOVE_STATUS_DID_NOT_HIT (the move passed the type
    # chart, ``_Hit.effective``): a U-turn's ON_HIT switch needs it (bcp.c:3486-3487).
    effective: jnp.ndarray


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
# and Seismic Toss (effect_script_0087.s:5), Foresight (effect_script_0113.s:6, no damage), Future
# Sight (effect_script_0148.s:7, no damage at its use).
# subscript_type_resist_berry.s:5 skips the berry for such a hit. tests/tools/
# test_effect_map.py checks the list against every mapped effect script.
IGNORES_TYPE_CHECKS = (_E.LEVEL_DAMAGE, _E.FORESIGHT, _E.HALVE_HP, _E.COUNTER, _E.FUTURE_SIGHT)
# subscript_type_resist_berry.s:6 also skips a MOVE_STATUS_ONE_HIT_KO hit.
RESIST_BERRY_SKIPS = (*IGNORES_TYPE_CHECKS, _E.OHKO)


def resist_berry(
    tables: Tables, state: BattleState, side, move_type, super_effective
) -> jnp.ndarray:
    """``side``'s active Pokemon holds a resist berry that halves the hit it takes
    (``subscript_type_resist_berry.s``, from ``subscript_update_hp.s:9``, which runs it for
    every damaging hit while a battler still has an action, ``waitingBattlers != 0``, and
    no ``SYSCTL_SKIP_SPRITE_BLINK``: never for recoil, Rough Skin or the end of turn).
    ``move_type`` is the script's ``calcTemp``: ``battleCtx->moveType`` (Hidden Power's
    type) when it is not Normal, else ``CURRENT_MOVE_DATA.type`` (:_020-_031; Normalize,
    which would make it Normal, is unsupported). ``super_effective`` is
    ``MOVE_STATUS_SUPER_EFFECTIVE``. Chilan (``RESIST_ANY_NORMAL``) needs a Normal type
    only (:_128); every other one a super-effective hit of its type (:_135-_240). The caller skips
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


def life_orb_recoil(tables: Tables, state: BattleState, side, cond) -> BattleState:
    """Life Orb's recoil on ``side``'s active Pokemon where ``cond`` (``SYSCTL_MOVE_HIT``: a
    hit of the move dealt damage, and the move, ``CURRENT_MOVE_DATA``, is not a status move):
    ``AFTER_MOVE_HIT_STATE_LIFE_ORB`` (battle_controller_player.c:4758-4772) after a move,
    ``BattleSystem_TriggerHeldItemOnPivotMove`` (battle_lib.c:7357-7366) after Pursuit's
    pre-switch hit. A living holder (``curHP``) without Magic Guard (read by decomp ability
    id; Magic Guard itself is unsupported) and with no U-turn switch pending (U-turn is
    unsupported) loses ``BattleSystem_Divide(maxHP, 10)`` through
    ``subscript_lose_hp_from_item.s``: ``BATTLE_SUBSCRIPT_UPDATE_HP`` with
    ``SYSCTL_SKIP_SPRITE_BLINK`` (no resist berry), capped at its HP; no Bide memory (only the
    controller's ``UpdateHP`` stores damage). It may faint."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    hp = state.hp[side, mon].astype(_I32)
    magic_guard = decomp_abilities(tables, state)[side, mon] == WEATHER_ABILITY_IDS["Magic Guard"]
    on = jnp.asarray(cond) & holds(tables, state, side, ItemId.LIFE_ORB) & (hp > 0) & ~magic_guard
    loss = divide(state.max_hp[side, mon].astype(_I32), 10)
    return _set(state, "hp", (side, mon), jnp.maximum(hp - loss, 0), on)


def holds(tables: Tables, state: BattleState, side, item: ItemId) -> jnp.ndarray:
    """``side``'s active Pokemon holds an item with effect ``item``."""
    return tables.item_effect[state.item[side, state.active[side]]] == item


def consume_item(state: BattleState, side, cond) -> BattleState:
    """``BtlCmd_RemoveItem`` (battle_script.c:8972) on ``side``'s active Pokemon where
    ``cond``: the item goes to ``item_consumed``; its ``BattleMon_CopyToParty`` (:8981) then
    clears the AI's record of the item (``forget_item``)."""
    mon = state.active[side]
    item = state.item[side, mon]
    state = _set(state, "item_consumed", (side, mon), item, cond)
    state = forget_item(state, side, mon, cond)
    return _set(state, "item", (side, mon), 0, cond)


def reveal_item(state: BattleState, side, mon, cond) -> BattleState:
    """A message names the held item of ``side``'s Pokemon in party slot ``mon`` where
    ``cond``: ``BattleMessage_ItemTag`` (battle_script.c:11726-11745) runs
    ``BattleAI_SetHeldItem`` (:12119-12122) for an item tag of ``BTLSCR_ATTACKER``,
    ``BTLSCR_DEFENDER`` or ``BTLSCR_MSG_BATTLER_TEMP`` (not ``BTLSCR_MSG_TEMP``, which names
    ``msgItemTemp``), so ``ai_item_known[mon]`` becomes the item. Only the player's
    (``side`` 0) is kept: ``AICmd_IfHeldItemEqualTo`` and ``AICmd_LoadHeldItemEffect``
    (trainer_ai.c:1901-1934) read ``battlerHeldItems`` for the other side's battlers only."""
    on = jnp.asarray(cond) & (jnp.asarray(side) == 0)
    known = state.ai_item_known
    value = jnp.where(on, state.item[side, mon], known[mon]).astype(known.dtype)
    return state._replace(ai_item_known=known.at[mon].set(value))


def forget_item(state: BattleState, side, mon, cond) -> BattleState:
    """``side``'s Pokemon in party slot ``mon`` loses its held item where ``cond``: every such
    path (``BtlCmd_RemoveItem``, battle_script.c:8979-8981; ``UpdateMonData`` setting
    ``BATTLEMON_HELD_ITEM``, :3077-3078, as subscript_transfer_sticky_barb.s:7 does) runs
    ``BattleMon_CopyToParty``, which clears the AI's record of a battler holding nothing
    (``BattleAI_ClearKnownItem``, battle_lib.c:1819-1823, :7735-7738). Player only
    (``reveal_item``)."""
    on = jnp.asarray(cond) & (jnp.asarray(side) == 0)
    known = state.ai_item_known
    return state._replace(
        ai_item_known=known.at[mon].set(jnp.where(on, 0, known[mon]).astype(known.dtype))
    )


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
    foe_custap=False,
    extra_flinch=True,
    counter_damage=0,
) -> tuple[BattleState, MoveHit]:
    """Hit ``hit_index`` (0-4, an int or a traced int32) of ``move`` by ``side``'s active Pokemon: damage with
    this hit's crit and roll, then the move's effect.

    ``connected``: the move is being used and passed its one accuracy check
    against a living target (any target for a ``RANGE_USER`` move). ``target_moved``:
    the target acted earlier this turn (no flinch). ``power_mul``: power scale in
    tenths (Pursuit on a switch: 20). ``struggling``: the move is Struggle, whose
    recoil is its secondary effect. ``focus_armed``: an earlier hit of this action
    activated the defender's Focus Sash (``MoveHit``). ``foe_custap``: the target's turn order
    activated its Custap Berry (``hooks.pluck_berry``). ``extra_flinch`` (static): the hit
    reaches ``AfterMoveMessage``'s King's Rock check; Pursuit's pre-switch hit does not
    (``subscript_pursuit.s`` never runs ``CheckExtraFlinch``, bcp.c:3527, :3593). A hit beyond ``hit_count`` or after
    the target fainted does nothing. The order is ``BattleControllerPlayer_AfterMoveMessage``'s
    (battle_controller_player.c:3467): the move's own effect (``*_TRIGGER_SECONDARY``)
    before the contact ability (``*_TRIGGER_ABILITY``), then King's Rock (``*_EXTRA_FLINCH``).
    ``AfterMoveEffects`` (the held items, the Sticky Barb transfer, the defender's thaw) is
    the caller's (``step._use_move``).

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
        (outcome.crit[side, hit_index] != 0) & ~armored(state, side, target_side),
        outcome.roll[side, hit_index],
        power_mul,
        outcome.magnitude[side].astype(_I32),
        outcome.present[side].astype(_I32),
    )
    # Counter / Mirror Coat / Metal Burst (``counter_damage``: the amount step._use_move read
    # off the turn): battleCtx->damage, under SYSCTL_IGNORE_TYPE_CHECKS; the chart still flags an
    # immunity (battle_lib.c:7550-7551). No Counter-family move is RANGE_USER in either game
    # (Counter, Mirror Coat, Metal Burst: RANGE_SINGLE_TARGET_SPECIAL; Kaizo's Bide: single
    # target), so CheckTypeChart's RANGE_USER skip (bcp.c:2352) never applies.
    is_counter = tables.move_effect[move] == _E.COUNTER
    counter_type = move_type_of(tables, state, side, move)
    immune = (
        effectiveness(
            tables,
            counter_type,
            state.types[target_side, target],
            identified(state, side, target_side),
        )
        == 0
    ) | levitated(state, side, target_side, counter_type)
    damage = jnp.where(is_counter, jnp.where(immune, 0, jnp.asarray(counter_damage, _I32)), damage)
    target_hp = state.hp[target_side, target].astype(_I32)
    # False Swipe (bcp.c:3381-3384, inside UpdateHP before the endure checks): a hit that would
    # KO leaves 1 HP; against 1 HP it deals 0, but UpdateHP has run (SYSCTL_MOVE_HIT).
    swipe = (tables.move_effect[move] == _E.FALSE_SWIPE) & (damage > 0) & (damage >= target_hp)
    raw = damage
    damage = jnp.where(swipe, target_hp - 1, damage)
    # Bide is a RANGE_USER move whose unleash hits the foe (bcp.c:2702-2713).
    hits_foe = ~targets_user(tables, move) | (tables.move_effect[move] == _E.BIDE) | is_counter
    update_hp = landed & hits_foe & (raw > 0)  # BattleControllerPlayer_UpdateHP runs
    sash = holds(tables, state, target_side, ItemId.FOCUS_SASH)
    full = target_hp == state.max_hp[target_side, target].astype(_I32)
    armed = jnp.asarray(focus_armed) | (update_hp & sash & full)
    capped, endured = focus_sash_damage(state, target_side, damage, armed & update_hp)
    # battleCtx->damage after the endure cap: what subscript_update_hp.s receives.
    taken = jnp.where(endured, target_hp - 1, damage)
    # Secret Power's secondary effect is the terrain's (order.move_effects).
    effect, param = move_effects(tables, state, move)
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
                identified(state, side, target_side),
            )
            > 100
        )
    )
    resisted = (
        update_hp
        & ~jnp.isin(effect, jnp.asarray(RESIST_BERRY_SKIPS, effect.dtype))
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

    is_status = tables.move_category[move] == STATUS
    # A type-checked status move (Thunder Wave) fails on a 0x matchup.
    type_ok = (param[1] == 0) | (
        effectiveness(
            tables,
            move_type,
            state.types[target_side, target],
            identified(state, side, target_side),
        )
        > 0
    ) & ~levitated(state, side, target_side, move_type)
    passes = jnp.where(is_status, (effect != _E.STATUS_FOE) | type_ok, damage > 0)
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
        target_ignorable=ignorable(state, side, target_side),
        contact_proc=outcome.contact_proc[target_side, hit_index] != 0,
        update_hp=update_hp & (taken != 0),
        hit_index=jnp.asarray(hit_index, _I32),
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
        _tri_attack,
        _fang,
        _present,
        _camouflage,
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
        _ingrain,
        _mean_look,
        _imprison,
        _screen,
        _brick_break,
        _foresight,
        _attract,
        _bide,
        _rollout,
        _weather,
        _cure_paralysis,
        _swagger,
        _dragon_dance,
        _roost,
        _side_layers,
        _destiny_bond,
        _uproar,
    ):
        state = fn(tables, state, h, outcome)
    state, pluck_flag = _eat_berry(tables, state, h, outcome, foe_custap)
    state = _struggle_recoil(state, h, jnp.asarray(struggling, jnp.bool_))
    state = _contact_ability(tables, state, h, outcome)
    if extra_flinch is not False:  # a traced bool: U-turn defers its check (step._use_move)
        state = _kings_rock(tables, state, h._replace(landed=h.landed & extra_flinch), outcome)
    # UpdateHP's shellBellDamageDealt (bcp.c:3427-3431) and subscript_pursuit.s's (:21-29).
    shell = jnp.where(update_hp, jnp.minimum(taken, target_hp), 0)
    pursuit_shell = jnp.where(update_hp, jnp.minimum(damage, target_hp), 0)
    return state, MoveHit(
        focus_armed=armed,
        pluck_flag=pluck_flag,
        update_hp=update_hp,
        dealt=dealt,
        damaged=update_hp & (taken != 0),
        shell_bell=shell,
        pursuit_shell_bell=pursuit_shell,
        taken=jnp.where(update_hp, taken, 0),
        effective=h.effective,
    )
