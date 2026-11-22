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
  else writes that type to both and sets the user's ``MOVE_EFFECT_CAMOUFLAGE``
  (``camouflage``, effect_script_0213.s:6; ``_camouflage``). ``step`` restores the party types
  when the battler leaves (``BattleSystem_InitBattleMon`` re-reads them, battle_lib.c:114-115)
  and clears the bit with the volatiles (battle_lib.c:2071, :2166; Baton Pass drops it, :2076).
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
  (``targets_user``). A Ghost-type user takes ``subscript_curse_ghost.s`` (``_curse``, ``curse_fails``;
  E2 batch 8, B8-5).
* Selfdestruct, Explosion (``SELF_DESTRUCT``): ``effect_script_0007.s`` sets the user's HP
  to 0 before ``TryMove`` (``step._use_move``: before the accuracy check and the type chart;
  not when the move has no target), then a plain hit on the halved Defense (``damage.py``).
  No effect function runs for it. Its Damp branch is unreachable (Damp is unsupported).
* Drain / recoil:``subscript_drain_half_damage_dealt.s``,
  ``subscript_recoil_1_{2,3,4}.s``: a fraction of ``hitDamage``, the damage dealt
  capped at the target's HP (``BtlCmd_UpdateHealthBarValue``, battle_script.c:1701),
  through ``BattleSystem_Divide`` (at least 1). Rock Head prevents recoil. A Big Root
  holder's drain is then ``* (100 + 30) / 100`` (``big_root``).
* Multi-hit: ``BtlCmd_SetMultiHit`` (battle_script.c:2634) and
  ``BattleControllerPlayer_LoopMultiHit`` (battle_controller_player.c:3777): the
  move script reruns per hit with ``SYSCTL_MULTI_HIT_MOVE``, which skips the
  accuracy check, so only ``outcome.hit[side, 0]`` matters while crit and roll
  are per hit; the loop stops once the target faints. ``SetMultiHit 0`` draws 2-5
  (``outcome.n_hits``); ``SetMultiHit 2`` (``effect_script_0044.s``) is two hits
  (``move_param[m, 0]``). Twineedle (``effect_script_0077.s``, ``MoveFlag.HIT_TWICE``) is a
  ``SECONDARY_STATUS`` poison hit twice: the reloaded script sets its indirect effect again
  and ``MULTI_HIT_TRIGGER_SECONDARY`` (bcp.c:3544-3554) rolls it per hit
  (``outcome.secondary[side, hit]``), B13-1. Triple Kick (``TRIPLE_KICK``, ``SetMultiHit 3,
  SYSCTL_TRIPLE_KICK``): three hits at power k x step (``damage.triple_kick_power``), each later
  one with its own accuracy roll without the overrides, the first miss ending the move
  (``step._use_move``, B13-5, B13-6). Beat Up (``BEAT_UP``): a hit per eligible party member
  (``damage.beat_up_slots``), its damage the member's (``damage.beat_up_damage``), B13-9,
  B13-10. Focus Sash activates only on a hit taken at full HP
  (bcp.c:3386-3404) and then stays active for the rest of the move (``play_hit``).
* Life Orb (``LIFE_ORB``): ``life_orb_recoil``, run by ``step`` once per move after its
  hits (``AFTER_MOVE_HIT_STATE_LIFE_ORB``) and after Pursuit's pre-switch hit; its damage
  multiplier is in ``damage.life_orb``.
* Resist berries (``RESIST_BERRY``): ``subscript_update_hp.s:9`` calls
  ``subscript_type_resist_berry.s`` on every move hit taken during an action (``play_hit``,
  Pursuit's hit through it, and the confusion self-hit in ``step``): ``resist_berry``,
  ``resist_berry_damage``.
* ``HIGH_CRIT`` only raises the crit stage; the engine reads ``outcome.crit`` with the odds
  at the hit (``outcome.crit_lands``).
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
  targets are immune; the recipient is the user's battler (``leech_seed_recipient``, :14-15).
  The end-of-turn drain is in ``step``.
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
* Miracle Eye: ``effect_script_0216.s`` / ``subscript_miracle_eye.s`` set ``miracle_eye`` on the
  defender like Foresight's flag (``_miracle_eye``); ``outcome.py`` ignores its positive evasion and
  ``damage.chart_entry`` skips its Dark immunity rows (E3-B2-10).
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
  reveals the holder's ability. The attacker's own Oblivious (:5, ``CheckIgnorableAbility``; the
  attacker is the side-effect battler, so no Mold Breaker pierces it) ends it silently
  (:_055 -> :_082 -> :_117), H-2.
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

Substitute (E2 batch 10, B10-4): ``play_hit``'s hit into it and the gates of every effect
(``_Hit.sub_hit``, ``_Hit.shielded``), ``_substitute`` the move.
Safeguard (E2 batch 10, B10-7): ``_safeguard`` and its gates on status and confusion.
Mist (E2 batch 12, B12-23): ``_mist`` the move, ``mist_blocks_drop`` its block in every drop by
another battler (``_foe_stage``, ``hooks.intimidate``).
Leaf Guard and Synchronize (E1 Gardenia batch, rulings G-12..G-17): Leaf Guard in sun in
``_inflict``, ``_contact_ability`` and ``synchronize`` (Toxic Spikes, the status orbs and Yawn
read it by decomp id); ``synchronize`` is AfterMoveEffects' step, run by ``step._use_move``.
Immunity, Water Veil, Magma Armor (``_inflict``, ``_contact_ability``), Own Tempo (the confusion
effects) and Oblivious (``_attract``, ``_contact_ability``): E1 batch H, H-2; their cures are
``hooks.recover_status_by_ability`` and ``step._restore_status_on_switch``. Color Change
(``_contact_ability``) and Soundproof's uproar sleep test (``_inflict``) and Perish Song test
(``perish_song_eligible``): H-3.
Unreachable branches are not modelled: Clear Body, Hyper Cutter, Liquid Ooze,
Magic Guard, Skill Link and Normalize are unsupported abilities
(Grip Claw is E1 Batch D's, ``_trap``). Shield Dust (``_shield_dust``) and the weather moves
(``_weather``) are E2 batch 3's; Serene Grace acts on the secondary chance
(``outcome.secondary_chance``), Reckless on the damage (``damage.script_power_mul``),
Levitate and Scrappy on the type chart (``damage.levitated``, ``damage.identified``) and
Dry Skin's Water absorb in ``step._use_move``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine import field as fcm
from nuzline.engine.damage import (
    NUM_TYPES,
    PHYSICAL,
    PRESENT_HEAL,
    STATUS,
    battle_stats,
    battler_decomp_ability,
    beat_up_slots,
    conv2_choice,
    conv2_eligible,
    damage_for_roll,
    decomp_abilities,
    decomp_ability_id,
    divide,
    effectiveness,
    flung_view,
    ground_immune,
    grounded,
    identified,
    ignorable,
    me_first_after_calc,
    move_type_of,
    multitype_plate,
    spikes_loss,
    stealth_rock_loss,
    weather_suppressed,
    wonder_guarded,
)
from nuzline.engine.format import foe as _foe
from nuzline.engine.format import format_of, party, party_sides, side_of
from nuzline.engine.ids import (
    ABILITY_DECOMP_IDS,
    ABILITY_MULTITYPE,
    ACUPRESSURE_STATS,
    CHARGE_PARAM,
    CHARGE_TURNS,
    GENDER_NONE,
    GRIP_CLAW_TURNS,
    ITEM_ABILITY_IDS,
    ITEM_GRISEOUS_ORB,
    LUCKY_CHANT_TURNS,
    MAGNET_RISE_TURNS,
    MIMIC_FORBIDDEN,
    MIMIC_PP_CAP,
    MINIMIZE_PARAM,
    MOVE_FAKE_OUT,
    MOVE_HEAL_BELL,
    MOVE_IMPRISON,
    MOVE_KNOCK_OFF,
    MOVE_LAST_RESORT,
    MOVE_MIMIC,
    MOVE_STRUGGLE,
    POWER_IF_TARGET_ASLEEP,
    POWER_IF_TARGET_MOVED,
    REFRESH_STATUSES,
    RESIST_ANY_NORMAL,
    REST_TURNS,
    SPECIES_CHATOT,
    SPIKES_MAX,
    STAT_MOVE_ABILITY_IDS,
    TAILWIND_TURNS,
    TERRAIN_TYPES,
    TOXIC_SPIKES_MAX,
    TRI_ATTACK_STATUSES,
    UNROLLED_EFFECTS,
    WEATHER_ABILITY_IDS,
    WEATHER_TURNS,
    WISH_TURNS,
    YAWN_ABILITY_IDS,
    AbilityId,
    EffectId,
    FlingEffect,
    ItemId,
    MoveFlag,
    MoveTarget,
    SemiKind,
    Stat,
    Status,
    Type,
    sleepless,
)
from nuzline.engine.order import encore_slot_of, move_effects, sketch_copy, terrain_index
from nuzline.engine.outcome import Outcome, crit_lands, read_chance, secondary_procs
from nuzline.engine.state import (
    MOVES,
    STAGE_MAX,
    STAGE_MIN,
    BattleState,
    battler_ability,
    field_index,
    held_item,
    knocked_off_at,
)
from nuzline.engine.targets import NO_DEFENDER

_I32 = jnp.int32
_E = EffectId
BIDE_TURNS = 2  # VOLATILE_CONDITION_BIDE_1 (condition.h:41): the 2-bit counter's value 2
SAFEGUARD_TURNS = 5  # BtlCmd_TrySafeguard (battle_script.c:5765)
MIST_TURNS = 5  # BtlCmd_TryMist: NUM_SCREEN_TURNS (battle_script.c:4312; battle.h:142)
ROLLOUT_USES = 5  # BtlCmd_CalcRolloutPower starts rolloutCount at 5
SPITE_PP = 4  # BtlCmd_TrySpite's ``dec`` (battle_script.c:5000)


class _Hit(NamedTuple):
    """What one hit's effect functions need to know."""

    side: jnp.ndarray  # attacking side
    target_side: jnp.ndarray
    defender: jnp.ndarray  # BattleSystem_Defender's battler (targets.resolve), NO_DEFENDER for none
    user: jnp.ndarray  # party slots
    target: jnp.ndarray
    effect: jnp.ndarray  # EffectId
    param: jnp.ndarray  # int32[3], tables.move_param[move]
    certain: jnp.ndarray  # chance 0: an ON_HIT secondary
    landed: jnp.ndarray  # the move connected on this hit
    effective: jnp.ndarray  # ... and was not ineffective (damage > 0, or a passed type gate)
    proc: jnp.ndarray  # the secondary applies on this hit (its roll, or certain)
    dealt: jnp.ndarray  # HP the target lost on this hit
    target_moved: jnp.ndarray  # D3-6 Fact A: the defender's PICK_COMMAND is MOVE_END (lib.c:5365)
    move: jnp.ndarray  # the move's id
    target_ignorable: jnp.ndarray  # target's ability as the attacker sees it (Mold Breaker)
    contact_proc: jnp.ndarray  # the target's contact-ability roll for this hit
    # BattleControllerPlayer_UpdateHP ran for this hit with a nonzero damage: the defender's
    # physical / specialDamageTaken (bcp.c:3411-3424), which King's Rock reads (MoveHit.damaged).
    update_hp: jnp.ndarray
    hit_index: jnp.ndarray  # the hit's index (per-hit outcome fields)
    # The defender's SELF_TURN_FLAG_SUBSTITUTE_HIT after this hit (bcp.c:3371): a hit of this
    # action landed on its substitute. ``Battler_SubstituteWasHit`` (battle_lib.c:6259-6268).
    sub_hit: jnp.ndarray
    # ``BtlCmd_CheckSubstitute`` on the defender after this hit (battle_script.c:9173-9186):
    # its substitute is up or ``sub_hit``.
    shielded: jnp.ndarray
    # The item Fling's script removed this action (``step._use_move``), else 0: the item whose
    # fling effect ``_fling`` applies to the defender (E3-B2-1).
    fling_item: jnp.ndarray
    # The stat (``Stat`` + 1, 0 for none) a flung Starf Berry raises, picked at TryFling
    # (``hooks.fling_try``, battle_lib.c:6110-6127), E3-B2-1.
    fling_pick: jnp.ndarray
    # The user's chosen move slot (``moveSlot``): Magic Coat's flag keeps it for the reflection's
    # Pressure drain (``_magic_coat``, E3-B2-6).
    coat_slot: jnp.ndarray


# Triple Kick's ``SetMultiHit 3`` (effect_script_0104.s:5; Kaizo's be_seq 104 and 88), B13-5.
TRIPLE_KICK_HITS = 3


def hit_count(tables: Tables, move, outcome: Outcome, side, state=None) -> jnp.ndarray:
    """``int32``: hits the move makes if nothing faints: for ``MULTI_HIT`` its fixed count
    ``move_param[m, 0]`` (``SetMultiHit 2``), or ``outcome.n_hits[side]`` when that is 0
    (``SetMultiHit 0``, the 2-5 law); 2 for a ``MoveFlag.HIT_TWICE`` move of another family
    (Twineedle's ``SetMultiHit 2``, effect_script_0077.s:6, B13-1); 3 for ``TRIPLE_KICK``
    (B13-5; a later miss ends it sooner, ``step._use_move``); for ``BEAT_UP`` the eligible party
    members of ``state`` (``damage.beat_up_slots``, B13-9); else 1."""
    effect = tables.move_effect[move]
    multi = effect == _E.MULTI_HIT
    twice = (tables.move_flags[move].astype(_I32) & MoveFlag.HIT_TWICE) != 0
    fixed = tables.move_param[move, 0].astype(_I32)
    drawn = outcome.n_hits[side].astype(_I32)
    other = jnp.where(twice, 2, jnp.where(effect == _E.TRIPLE_KICK, TRIPLE_KICK_HITS, 1))
    if state is not None:
        other = jnp.where(effect == _E.BEAT_UP, beat_up_slots(state, side)[1], other)
    return jnp.where(multi, jnp.where(fixed > 0, fixed, drawn), other)


def _alive(state: BattleState, side, slot) -> jnp.ndarray:
    fmt = format_of(state)
    return state.hp[party(fmt, side), slot] > 0


def change_stage(state: BattleState, side, stat, delta, cond) -> BattleState:
    """Add ``delta`` to ``side``'s stage ``stat`` where ``cond``, clamped to +-6."""
    current = state.stages[side, stat].astype(_I32)
    new = jnp.clip(current + delta, STAGE_MIN, STAGE_MAX)
    new = jnp.where(cond, new, current).astype(state.stages.dtype)
    return state._replace(stages=state.stages.at[side, stat].set(new))


def mist_blocks_drop(state: BattleState, target_side, delta) -> jnp.ndarray:
    """``bool``: ``BtlCmd_ChangeStatStage``'s first test of a drop on another battler
    (battle_script.c:2874-2882): the target's side has ``mistTurns`` set ("is protected by
    Mist!"), B12-23. It comes before Clear Body, the -6 test, Shield Dust and the substitute,
    and reads no ability, so the attacker's Mold Breaker does not pierce it. Exempt: a drop on
    the attacker itself (:2876, ``attacker != sideEffectMon``: a user's own drop such as Hammer
    Arm's, ``_secondary_stat_self``, never calls this) and one with
    ``MOVE_SIDE_EFFECT_CANNOT_PREVENT`` (:2875), which no script sets. In doubles an ally's drop
    is blocked too (the test is on the battler, not the side)."""
    return (jnp.asarray(delta) < 0) & (state.mist[side_of(target_side)] != 0)


def _foe_stage(state: BattleState, h: _Hit, cond) -> BattleState:
    stat, delta = h.param[0], h.param[1]
    # BtlCmd_ChangeStatStage (battle_script.c:2874-2902), a drop by another battler: the target
    # side's Mist (B12-23), then Clear Body, then Keen Eye on accuracy and Hyper Cutter on
    # Attack (Battler_IgnorableAbility), B9-6.
    seen = h.target_ignorable
    keen_eye = (seen == AbilityId.KEEN_EYE) & (stat == Stat.ACC)
    hyper_cutter = (seen == AbilityId.HYPER_CUTTER) & (stat == Stat.ATK)
    # White Smoke is Clear Body's twin (battle_script.c:2883-2884, ruling G-46).
    clear_body = (seen == AbilityId.CLEAR_BODY) | (seen == AbilityId.WHITE_SMOKE)
    blocked = (delta < 0) & (clear_body | keen_eye | hyper_cutter)
    blocked = blocked | mist_blocks_drop(state, h.target_side, delta)
    # Then the stage already at -6, then (indirect) Shield Dust, then the substitute's bit
    # (VOLATILE_CONDITION_SUBSTITUTE only, :2921-2923): a direct drop fails, an indirect one is
    # skipped silently (:2937-2945). Each blocks the drop; raises are never tested.
    blocked = blocked | ((delta < 0) & (state.substitute_hp[h.target_side] != 0))
    return change_stage(state, h.target_side, stat, delta, cond & ~blocked)


def _stat_self(tables, state, h: _Hit) -> BattleState:
    cond = (h.effect == _E.STAT_SELF) & h.landed
    # Minimize (subscript_minimize.s:5) sets MOVE_EFFECT_MINIMIZE before its Evasion +1 (B9-10).
    state = _set(state, "minimized", h.side, 1, cond & (h.param[2] == MINIMIZE_PARAM))
    # Charge (subscript_charge.s:5-6): MOVE_EFFECT_CHARGE and chargedTurns = 2 before the Sp. Def
    # stage (:11-13), whatever the stage does (E3-B2-8).
    state = _set(state, "charged_turns", h.side, CHARGE_TURNS, cond & (h.param[2] == CHARGE_PARAM))
    state = change_stage(state, h.side, h.param[0], h.param[1], cond)
    # Defense Curl (move_param[m, 2] = 1) sets the flag whatever the stage change does.
    return _set(state, "curled", h.side, 1, cond & (h.param[2] == 1) & (h.param[0] == Stat.DEF))


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
    # A rolled secondary on the defender needs ~SubstituteWasHit (battle_lib.c:1580-1582).
    cond = cond & ~_shield_dust(h) & ~(h.sub_hit & ~h.certain)
    return _foe_stage(state, h, cond)


def can_have_status(
    tables: Tables, state: BattleState, side, slot, status, current=None
) -> jnp.ndarray:
    """No status yet (``current``, default ``state.status[side, slot]``: a fainted battler's
    ``BattleMon.status``, which its faint keeps, ``effects.synchronize``) and no type (or
    weather) immunity to ``status``. Sun stops a freeze
    only without a weather lock: ``subscript_freeze.s:6`` runs ``CheckIgnoreWeather`` before
    the sun check at :7, and ``BtlCmd_CheckIgnoreWeather`` (battle_script.c:9195-9206) jumps
    past it while a battler on the field with HP left has Cloud Nine or Air Lock
    (``damage.weather_suppressed``, read on ``state``: the field when the effect runs)."""
    fmt = format_of(state)
    types = state.types[party(fmt, side), slot]
    sun = _sunny(tables, state)
    has = lambda t: jnp.any(types == t)
    immune = (
        ((status == Status.BURN) & has(Type.FIRE))
        | (
            ((status == Status.POISON) | (status == Status.TOXIC))
            & (has(Type.POISON) | has(Type.STEEL))
        )
        | ((status == Status.FREEZE) & (has(Type.ICE) | sun))
    )
    current = state.status[party(fmt, side), slot] if current is None else current
    return (current == Status.NONE) & ~immune


def _inflict(
    tables: Tables, state: BattleState, h: _Hit, outcome: Outcome, cond, direct
) -> BattleState:
    """``direct``: a status move's own effect (``SIDE_EFFECT_TYPE_DIRECT``), not a damaging
    move's secondary effect (``_INDIRECT``)."""
    fmt = format_of(state)
    status = h.param[0]
    cond = cond & _alive(state, h.target_side, h.target)
    # subscript_fall_asleep.s:_055/_202/_221: an ignorable Insomnia or Vital Spirit (:23-24,
    # G-47) stops the sleep before any other check, and its "stayed awake" message tags the
    # ability.
    blocked = cond & (status == Status.SLEEP) & sleepless(h.target_ignorable)
    # subscript_paralyze.s:5/_170: an ignorable Limber stops a paralysis before any other
    # check; "prevents paralysis" (TAG_NICKNAME_ABILITY) is printed for a direct effect only.
    limber = cond & (status == Status.PARALYSIS) & (h.target_ignorable == AbilityId.LIMBER)
    # H-2: likewise an ignorable Immunity stops any poison (subscript_poison.s:13,
    # subscript_badly_poison.s:31), Water Veil a burn (subscript_burn.s:20) and Magma Armor a
    # freeze (subscript_freeze.s:5), each before any other check. A direct effect prints
    # "prevents poisoning" (poison :_177 -> :_201, badly_poison :_249) or "prevents burns"
    # (burn :_264), TAG_NICKNAME_ABILITY on BTLSCR_SIDE_EFFECT_MON (battle_script.c:11798-11810);
    # an indirect one ends silently. Magma Armor's block prints the attack message only
    # (subscript_freeze.s:_128): it never reveals.
    poisons = (status == Status.POISON) | (status == Status.TOXIC)
    immune = cond & poisons & (h.target_ignorable == AbilityId.IMMUNITY)
    veil = cond & (status == Status.BURN) & (h.target_ignorable == AbilityId.WATER_VEIL)
    magma = cond & (status == Status.FREEZE) & (h.target_ignorable == AbilityId.MAGMA_ARMOR)
    from nuzline.engine.hooks import (  # hooks imports this module
        BURN_BLOCKED,
        PARALYSIS_BLOCKED,
        POISON_BLOCKED,
        SLEEP_BLOCKED,
        STATUS_BLOCKED,
        reveal_ability,
    )

    state = reveal_ability(state, h.target_side, h.target, SLEEP_BLOCKED, blocked)
    state = reveal_ability(state, h.target_side, h.target, PARALYSIS_BLOCKED, limber & direct)
    state = reveal_ability(state, h.target_side, h.target, POISON_BLOCKED, immune & direct)
    state = reveal_ability(state, h.target_side, h.target, BURN_BLOCKED, veil & direct)
    blocked = blocked | limber | immune | veil | magma
    # Leaf Guard in sun (ruling G-14), after the status's own ability test (Insomnia, Vital
    # Spirit, Limber, Immunity, Water Veil): ``CheckIgnoreWeather`` then
    # FIELD_CONDITION_SUNNY then an ignorable ABILITY_LEAF_GUARD (subscript_fall_asleep.s:23-25,
    # subscript_poison.s:14-16, subscript_badly_poison.s:32-34, subscript_burn.s:21-23,
    # subscript_paralyze.s:6-8; subscript_freeze.s has no such test). A direct effect prints the
    # ability's message (TAG_NICKNAME_ABILITY: poison :_177/_201, badly_poison :_249/_266, burn
    # :_264, paralyze :_170, fall_asleep :_202/_221), which the AI learns; an indirect one ends
    # silently.
    guarded = (
        (status == Status.SLEEP)
        | (status == Status.POISON)
        | (status == Status.TOXIC)
        | (status == Status.BURN)
        | (status == Status.PARALYSIS)
    )
    leaf = (
        cond
        & ~blocked
        & guarded
        & _sunny(tables, state)
        & (h.target_ignorable == AbilityId.LEAF_GUARD)
    )
    state = reveal_ability(state, h.target_side, h.target, STATUS_BLOCKED, leaf & direct)
    # Shield Dust, after the abilities above and Leaf Guard, for an indirect effect only.
    blocked = blocked | leaf | (cond & ~jnp.asarray(direct, bool) & _shield_dust(h))
    # Then CheckSubstitute on the side-effect battler (subscript_paralyze.s:21, burn :36,
    # freeze :19, poison :29, badly_poison :46, fall_asleep :40, before its uproar test at
    # :43): a direct effect prints "But it failed!", an indirect one stops silently. An
    # ABILITY side effect skips the test (paralyze :20 and the like; not through here).
    blocked = blocked | (cond & h.shielded)
    # Then the target's side's Safeguard (paralyze :28, burn :45, freeze :25, poison :41,
    # badly_poison :55, fall_asleep :51, after its uproar test at :45), B10-7: silent for an
    # indirect effect.
    blocked = blocked | (cond & (state.safeguard[side_of(h.target_side)] != 0))
    # subscript_fall_asleep.s:43-45 (:14-16 for a move effect): a target not asleep yet stays
    # awake while FIELD_CONDITION_UPROAR is set unless its own Soundproof skips the test
    # (CheckAbility, :15, :44: Battler_Ability, Mold Breaker moot; H-3); a direct effect then
    # ends in MOVE_STATUS_NO_MORE_WORK (``step``: ``kept_awake``), an indirect one silently
    # (:_292).
    soundproof = battler_ability(state, h.target_side, h.target) == AbilityId.SOUNDPROOF
    blocked = blocked | (cond & (status == Status.SLEEP) & fcm.uproar_any(state) & ~soundproof)
    cond = cond & ~blocked & can_have_status(tables, state, h.target_side, h.target, status)
    counter = jnp.where(status == Status.SLEEP, outcome.duration[h.side], 0)
    old_status = state.status[party(fmt, h.target_side), h.target]
    old_counter = state.status_counter[party(fmt, h.target_side), h.target]
    state = state._replace(
        status=state.status.at[party(fmt, h.target_side), h.target].set(
            jnp.where(cond, status, old_status).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[party(fmt, h.target_side), h.target].set(
            jnp.where(cond, counter, old_counter).astype(state.status_counter.dtype)
        ),
    )
    # subscript_fall_asleep.s:_176 and subscript_freeze.s:_073 unlock the new sleeper / ice cube.
    sleeps = (status == Status.SLEEP) | (status == Status.FREEZE)
    return unlock_move(state, h.target_side, cond & sleeps)


def _status_foe(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    return _inflict(tables, state, h, outcome, (h.effect == _E.STATUS_FOE) & h.effective, True)


def _secondary_status(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    # RECOIL_STATUS: after its recoil (``_recoil``), the subscript's BATTLE_SUBSCRIPT_PARALYZE /
    # _BURN under SYSCTL_APPLY_SECONDARY_EFFECT: the roll passed and the defender had HP
    # (battle_lib.c:1560-1565), an indirect effect (B10-1).
    status_effect = (h.effect == _E.SECONDARY_STATUS) | (h.effect == _E.RECOIL_STATUS)
    cond = status_effect & h.effective & h.proc
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
    """``CAMOUFLAGE``: ``BtlCmd_TryCamouflage`` (module docstring): on success both types and
    the user's ``camouflage`` bit (effect_script_0213.s:5-6; a failure jumps to :16 past it)."""
    fmt = format_of(state)
    new = jnp.asarray(TERRAIN_TYPES, _I32)[terrain_index(state)]
    types = state.types[party(fmt, h.side), h.user]
    # BtlCmd_TryCamouflage (battle_script.c:8001): Battler_Ability.
    multitype = battler_decomp_ability(tables, state, h.side, h.user) == ABILITY_MULTITYPE
    cond = (h.effect == _E.CAMOUFLAGE) & h.landed & ~multitype & ~jnp.any(types.astype(_I32) == new)
    value = jnp.where(cond, new, types.astype(_I32)).astype(types.dtype)
    state = state._replace(types=state.types.at[party(fmt, h.side), h.user].set(value))
    return _set(state, "camouflage", h.side, 1, cond)


# The ranges for which UpdateFlagsWhenHit clears the defender's conversion2 fields instead of
# recording the move (battle_controller_player.c:4511-4521).
CONV2_CLEARING_RANGES = (
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_OPPONENT_SIDE,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
)


def conv2_fails(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``BtlCmd_TryConversion2`` (battle_script.c:4765-4820) takes its failure jump for
    ``side``'s battler (``subscript_conversion_2.s:5, :12-14``: ``MOVE_STATUS_FAILED``, and the
    attack message after it is skipped): a raw Multitype ability (:4770, ``Battler_Ability``), no
    recorded move (``conv2_move`` 0, ``conv2_battler`` none: :4775-4776), a recorded multi-turn
    move (``Move_IsMultiTurn``, battle_lib.c:2975-2992: Bide and the charge moves,
    ``MoveFlag.CHARGE``) whose attacker is ``VOLATILE_CONDITION_MOVE_LOCKED`` (``locked_move``,
    :4778-4782), or no eligible row for the recorded type (:4783-4817; ``damage.conv2_eligible``).
    E3-B2-3."""
    fmt = format_of(state)
    user = state.active[side]
    multitype = battler_decomp_ability(tables, state, side, user) == ABILITY_MULTITYPE
    move = state.conv2_move[side].astype(_I32)
    source = state.conv2_battler[side].astype(_I32)
    multi_turn = ((tables.move_flags[move] & MoveFlag.CHARGE) != 0) | (
        tables.move_effect[move] == _E.BIDE
    )
    locked = state.locked_move[jnp.clip(source - 1, 0, fmt.B - 1)] != 0
    kind = jnp.clip(state.conv2_type[side].astype(_I32), 0, NUM_TYPES - 1)
    eligible = jnp.any(conv2_eligible(tables, state, side, kind))
    recorded = (move != 0) & (source != 0)
    return multitype | ~recorded | (multi_turn & locked) | ~eligible


def _conversion_2(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``CONVERSION_2`` (effect_script_0093.s:5, a direct ON_HIT ``subscript_conversion_2.s``):
    unless ``conv2_fails``, the user's two types both become the defending type ``BtlCmd_
    TryConversion2`` picked (battle_script.c:4795-4797, ``Outcome.conv2[user, recorded type]``,
    ``damage.conv2_choice``). The types stay until the battler leaves the field (its party row is
    restored at the switch-in, like Camouflage's). E3-B2-3."""
    fmt = format_of(state)
    kind = jnp.clip(state.conv2_type[h.side].astype(_I32), 0, NUM_TYPES - 1)
    _, picked = conv2_choice(tables, state, h.side, kind, outcome.conv2[h.side, kind])
    cond = (h.effect == _E.CONVERSION_2) & h.landed & ~conv2_fails(tables, state, h.side)
    types = state.types[party(fmt, h.side), h.user]
    value = jnp.where(cond, picked, types.astype(_I32)).astype(types.dtype)
    return state._replace(types=state.types.at[party(fmt, h.side), h.user].set(value))


def spite_slot(tables: Tables, state: BattleState, target_side) -> tuple:
    """``(fails, slot, taken)``: ``BtlCmd_TrySpite`` (battle_script.c:4990-5016) on ``target_side``'s
    battler. ``slot`` is the first slot holding its ``movePrevByBattler`` (``last_move``,
    ``Battler_SlotForMove``, battle_lib.c:2376-2387); the move fails (jump to subscript_reduce_
    target_pp.s:12-14, ``MOVE_STATUS_FAILED``, the attack message skipped) when that is none
    (:4995, :5011-5013), is in no slot (Struggle, a move the battler no longer holds) or its slot
    has 0 PP (:4996-4998). ``taken`` = ``min(4, ppCur)`` (:5000-5003). E3-B2-4."""
    fmt = format_of(state)
    mon = state.active[target_side]
    last = state.last_move[target_side].astype(_I32)
    moves = state.moves[party(fmt, target_side), mon].astype(_I32)
    held = (moves == last) & (last != 0)
    slot = jnp.argmax(held).astype(_I32)
    left = state.pp[party(fmt, target_side), mon, slot].astype(_I32)
    fails = (last == 0) | ~jnp.any(held) | (left == 0)
    return fails, slot, jnp.minimum(left, SPITE_PP)


def _spite(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """``SPITE`` (effect_script_0100.s:5, a direct ON_HIT ``subscript_reduce_target_pp.s`` that
    skips a miss: ``MOVE_STATUS_MISSED`` / ``_SEMI_INVULNERABLE`` :5): unless ``spite_slot`` fails,
    the target's slot of its last move loses ``min(4, ppCur)`` (:4999-5005; the BattleMon's PP, then
    ``BattleMon_CopyToParty``: the party row it is). Through a Substitute, no ability test, no draw.
    E3-B2-4."""
    fmt = format_of(state)
    fails, slot, taken = spite_slot(tables, state, h.target_side)
    cond = (h.effect == _E.SPITE) & h.landed & ~fails
    row, mon = party(fmt, h.target_side), state.active[h.target_side]
    left = state.pp[row, mon, slot].astype(_I32) - jnp.where(cond, taken, 0)
    return state._replace(pp=state.pp.at[row, mon, slot].set(left.astype(state.pp.dtype)))


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

    # :6: CheckSubstitute comes first (a substitute ends it silently), before Inner Focus (:7).
    reached = reached & ~h.shielded
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
        # The second CheckEffectActivation reads the chance as the first (battle_script.c:
        # 8369-8378): Serene Grace as the attacker's ability is now.
        & secondary_procs(tables, state, h.side, h.move, outcome.fang_flinch[h.side])
        & _alive(state, h.target_side, h.target)
        & ~h.target_moved
        & (h.target_ignorable != AbilityId.INNER_FOCUS)
        & ~_shield_dust(h)
        & ~h.shielded  # subscript_flinch_mon.s:6
    )
    old = state.flinched[h.target_side]
    return state._replace(
        flinched=state.flinched.at[h.target_side].set(jnp.where(cond, 1, old).astype(old.dtype))
    )


def _recover(tables, state, h: _Hit) -> BattleState:
    """Recover: ``subscript_recover_half_max_hp.s`` + ``subscript_recover_hp.s`` (an ON_HIT
    direct effect on the user; module docstring)."""
    fmt = format_of(state)
    hp = state.hp[party(fmt, h.side), h.user].astype(_I32)
    max_hp = state.max_hp[party(fmt, h.side), h.user].astype(_I32)
    cond = (h.effect == _E.RECOVER) & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + jnp.maximum(max_hp // 2, 1), max_hp)
    return _set(state, "hp", (h.side, h.user), healed, cond)


def _present(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Present's heal path: the target regains ``Divide(maxHP, 4)``, capped, unless it is at
    full HP (module docstring). ``landed``, not ``effective``: immunities are ignored."""
    fmt = format_of(state)
    hp = state.hp[party(fmt, h.target_side), h.target].astype(_I32)
    max_hp = state.max_hp[party(fmt, h.target_side), h.target].astype(_I32)
    heal = (h.effect == _E.PRESENT) & (outcome.present[h.side] == PRESENT_HEAL)
    cond = heal & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + jnp.maximum(max_hp // 4, 1), max_hp)
    return _set(state, "hp", (h.target_side, h.target), healed, cond)


def recovers_at_full_hp(
    tables, state: BattleState, side, move, outcome: Outcome, target_side=None
) -> jnp.ndarray:
    """``bool``: ``side``'s ``move``, if it lands, runs ``subscript_recover_hp.s`` on a
    recipient at full HP: Recover on its user (``subscript_recover_half_max_hp.s:16``),
    Present's heal on its target (``subscript_present_heal.s:9``). That subscript's ``_036``
    ("HP is full!") sets ``MOVE_STATUS_NO_MORE_WORK``, so
    ``BattleControllerPlayer_CheckMoveFailure`` (bcp.c:3313-3315) jumps to
    ``LOOP_SPREAD_MOVES`` and the move's ``AfterMoveEffects`` (bcp.c:3638-3769: the attacker's
    and defender's held items, ``HELD_ITEM_STATUS``) never run (``step._use_move``).
    ``target_side``: the action's defender (Present's BTLSCR_DEFENDER, effect_script_0122.s);
    default the battler across."""
    fmt = format_of(state)
    side = jnp.asarray(side, _I32)
    foe = _foe(side) if target_side is None else jnp.asarray(target_side, _I32)
    effect = tables.move_effect[jnp.asarray(move, _I32)]
    user, target = state.active[side], state.active[foe]
    user_full = state.hp[party(fmt, side), user] == state.max_hp[party(fmt, side), user]
    target_full = state.hp[party(fmt, foe), target] == state.max_hp[party(fmt, foe), target]
    heal = (effect == _E.PRESENT) & (outcome.present[side] == PRESENT_HEAL)
    # Roost runs the same subscript (subscript_roost.s:10), and so does the weather heal
    # (subscript_present_heal.s:9, B12-13).
    recovers = (effect == _E.RECOVER) | (effect == _E.ROOST) | (effect == _E.WEATHER_HEAL)
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
    # effect_script_0109.s:5-7: the non-Ghost branch (subscript_curse_normal.s).
    fmt = format_of(state)
    ghost = ghost_user(state, h.side)
    cond = (h.effect == _E.CURSE) & h.landed & _alive(state, h.side, h.user) & ~ghost
    for stat, delta in CURSE_STAGES:
        state = change_stage(state, h.side, stat, delta, cond)
    # The Ghost branch (:9-17): a direct ON_HIT subscript_curse_ghost.s on the foe, unless it
    # fails (``curse_fails``): the foe's VOLATILE_CONDITION_CURSE, then the user loses
    # Divide(maxHP, 2) through UPDATE_HP (:9-15; it can faint).
    hexed = (h.effect == _E.CURSE) & h.landed & ghost & ~curse_fails(state, h.target_side)
    state = _set(state, "curse", h.target_side, 1, hexed)
    cost = divide(state.max_hp[party(fmt, h.side), h.user].astype(_I32), 2)
    return hp_update(state, h.side, h.user, -cost, hexed)


def nightmare_fails(state: BattleState, target_side) -> jnp.ndarray:
    """``bool``: effect_script_0107.s fails (MOVE_STATUS_FAILED, :11-12) on a defender behind a
    Substitute (:5, ``CheckSubstitute``: no hit of the action yet), with
    VOLATILE_CONDITION_NIGHTMARE (:6) or not asleep (:7); subscript_nightmare_start.s re-tests the
    same three (:5-7, :29-30) on the hit (G-31). ``target_side``: the defender's battler."""
    fmt = format_of(state)
    status = state.status[party(fmt, target_side), state.active[target_side]]
    return (
        (state.substitute_hp[target_side] != 0)
        | (state.nightmare[target_side] != 0)
        | (status != Status.SLEEP)
    )


def _nightmare(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Nightmare (effect_script_0107.s:8, a direct ON_HIT subscript_nightmare_start.s): unless
    ``nightmare_fails`` (the script's failure, ``step._use_move``, already stopped the move; the
    subscript's own :5-7 find the same), the defender's VOLATILE_CONDITION_NIGHTMARE (:9). No
    ability test: a Magic Guard holder takes the bit and loses nothing at the turn end (G-31)."""
    cond = (h.effect == _E.NIGHTMARE) & h.landed & ~nightmare_fails(state, h.target_side)
    return _set(state, "nightmare", h.target_side, 1, cond)


def clear_nightmare(state: BattleState, battler, cond) -> BattleState:
    """VOLATILE_CONDITION_NIGHTMARE off on ``battler`` where ``cond``: subscript_wake_up.s:20,
    subscript_held_item_slp_restore.s:13 / _multi_restore.s:13, subscript_ability_restore_status.s:6,
    subscript_ability_forbids_status.s:18 (the sleep branch), BtlCmd_TryPartyStatusRefresh
    (battle_script.c:5041, :5052, :5063, :5070) and a bag sleep cure (battle_system.c:469-472), G-32."""
    return _set(state, "nightmare", battler, 0, cond)


def curse_fails(state: BattleState, target_side) -> jnp.ndarray:
    """``bool``: subscript_curse_ghost.s fails (MOVE_STATUS_FAILED, :22-24) on a defender behind
    a substitute (:6) or already cursed (:7); its semi-invulnerable test (:5) cannot hold."""
    return (state.substitute_hp[target_side] != 0) | (state.curse[target_side] != 0)


def _percent_of_dealt(h: _Hit) -> jnp.ndarray:
    """``BattleSystem_Divide(hitDamage, 100 / percent)``: at least 1 of a non-zero hit."""
    return jnp.where(h.dealt > 0, jnp.maximum(h.dealt * h.param[0] // 100, 1), 0)


def big_root(tables: Tables, state: BattleState, side, heal) -> jnp.ndarray:
    """``heal`` for ``side``'s active Pokemon after Big Root (``HOLD_EFFECT_LEECH_BOOST``):
    ``heal * (100 + param) / 100`` (param 30), plain integer division after the base
    amount, no minimum (``subscript_drain_half_damage_dealt.s:_011-_031``,
    ``subscript_leech_seed_effect.s``)."""
    item = held_item(state, side)  # CheckItemHoldEffect / GetItemEffectParam (B13-29)
    boost = tables.item_param[item, 0].astype(_I32) + 100
    root = tables.item_effect[item] == ItemId.BIG_ROOT
    return jnp.where(root, heal * boost // 100, heal)


def _drain(tables, state, h: _Hit) -> BattleState:
    fmt = format_of(state)
    heal = jnp.where((h.effect == _E.DRAIN) & h.effective, _percent_of_dealt(h), 0)
    heal = big_root(tables, state, h.side, heal)
    hp = state.hp[party(fmt, h.side), h.user].astype(_I32)
    new = jnp.minimum(hp + heal, state.max_hp[party(fmt, h.side), h.user].astype(_I32))
    return state._replace(
        hp=state.hp.at[party(fmt, h.side), h.user].set(new.astype(state.hp.dtype))
    )


def _recoil(tables, state, h: _Hit) -> BattleState:
    # Rock Head; Magic Guard skips the subscript (subscript_recoil_1_2.s, _1_4.s:6, CheckAbility).
    rock_head = battler_ability(state, h.side, h.user) == AbilityId.ROCK_HEAD
    magic_guard = battler_ability(state, h.side, h.user) == AbilityId.MAGIC_GUARD
    # subscript_recoil_1_{2,3,4}.s: DivideVarByValue(hitDamage, 2 / 3 / 4), at least 1 (B9-4).
    # RECOIL_STATUS keeps its divisor in param 1 (subscript_recoil_1_3_chance_to_*.s:5 calls
    # BATTLE_SUBSCRIPT_RECOIL_1_3 first, on every hit that hit, B10-1).
    status_recoil = h.effect == _E.RECOIL_STATUS
    divisor = jnp.where(status_recoil, h.param[1], h.param[0])
    third = jnp.maximum(h.dealt // jnp.maximum(divisor, 1), 1)
    recoils = (h.effect == _E.RECOIL) | status_recoil
    cost = jnp.where(
        recoils & h.effective & ~rock_head & ~magic_guard & (h.dealt > 0),
        third,
        0,
    )
    return hp_update(state, h.side, h.user, -cost, cost > 0)


def ghost_user(state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``side``'s active battler has the Ghost type (``Move_IsGhostCurse``'s and
    effect_script_0109.s:5-6's test, its ``BattleMon`` types; Camouflage can take it away, no
    implemented move gives it, so selection and execution agree)."""
    fmt = format_of(state)
    return jnp.any(state.types[party(fmt, side), state.active[side]] == Type.GHOST)


def targets_user(tables: Tables, move, ghost=False) -> jnp.ndarray:
    """The move's decomp defender is the user itself: ``RANGE_USER``,
    ``RANGE_USER_SIDE`` and ``RANGE_FIELD`` (battle_lib.c:1665-1669), e.g. Growth,
    Reflect, Water Sport; and a non-Ghost user's Curse (``RANGE_SINGLE_TARGET`` in the
    data): ``BattleControllerPlayer_MustSelectTarget`` (bcp.c:4344-4346) makes its range
    ``RANGE_USER``, so the chosen target is the user (:4363-4364) and
    ``BattleSystem_Defender``'s single-target branch returns it (battle_lib.c:1709-1718).
    ``ghost``: the user has the Ghost type (``ghost_user``), whose Curse targets the foe (B8-5)."""
    target = tables.move_target[move]
    return (
        (target == MoveTarget.RANGE_USER)
        | (target == MoveTarget.RANGE_USER_SIDE)
        | (target == MoveTarget.RANGE_FIELD)
        # In singles BattleSystem_Defender gives RANGE_ALLY (Helping Hand) and RANGE_USER_OR_ALLY
        # (Acupressure) the attacker itself (battle_lib.c:1670-1684), B9-5.
        | (target == MoveTarget.RANGE_ALLY)
        | (target == MoveTarget.RANGE_USER_OR_ALLY)
        | ((tables.move_effect[move] == _E.CURSE) & ~jnp.asarray(ghost, bool))
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


def self_defender(tables: Tables, move, ghost=False) -> jnp.ndarray:
    """The move's decomp defender is its user: a ``SELF_DEFENDER_RANGES`` range or a
    non-Ghost Curse (``targets_user``; ``ghost``: the user is Ghost-type)."""
    move_range = tables.move_target[move]
    in_range = jnp.isin(move_range, jnp.asarray(SELF_DEFENDER_RANGES, move_range.dtype))
    return in_range | targets_user(tables, move, ghost)


def defender_side(target_side, defender) -> jnp.ndarray:
    """``int32``: the battler ``BattleSystem_Defender`` picked (``targets.resolve``,
    battle_lib.c:1600-1730; ``defender``), or ``target_side`` when it is BATTLER_NONE: the user
    itself for a ``self_defender`` move at B = 2, the partner or the chosen battler for
    RANGE_ALLY / RANGE_USER_OR_ALLY at B = 4. Counter's later defender (the last hitter) is the
    caller's."""
    defender = jnp.asarray(defender, _I32)
    return jnp.where(defender == NO_DEFENDER, jnp.asarray(target_side, _I32), defender)


# BattleControllerPlayer_DecrementPP's range switch (battle_controller_player.c:2241-2267):
# these ranges count the living Pressure holders other than the user (RANGE_ALL_ADJACENT,
# RANGE_FIELD: COUNT_ALIVE_BATTLERS_EXCEPT_ME) or of the other side (RANGE_ADJACENT_OPPONENTS,
# RANGE_OPPONENT_SIDE: COUNT_ALIVE_BATTLERS_THEIR_SIDE, battle_lib.c:2905-2967); these add
# nothing; every other range (the default case) reads the defender's ability alone.
PRESSURE_COUNT_EXCEPT_ME = (MoveTarget.RANGE_ALL_ADJACENT, MoveTarget.RANGE_FIELD)
PRESSURE_COUNT_THEIR_SIDE = (MoveTarget.RANGE_ADJACENT_OPPONENTS, MoveTarget.RANGE_OPPONENT_SIDE)
PRESSURE_NONE = (
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_OR_ALLY,
    MoveTarget.RANGE_ALLY,
)
# ``defender`` of ``pressure_pp_cost`` when the decomp's is BATTLER_NONE (``targets.NO_DEFENDER``).


def pressure_holder(state: BattleState, battler) -> jnp.ndarray:
    """``Battler_Ability(battler) == ABILITY_PRESSURE`` (battle_lib.c:3088-3106,
    ``state.battler_ability``; ``battler`` is a battler, its active Pokemon)."""
    return battler_ability(state, battler) == AbilityId.PRESSURE


def pressure_pp_cost(tables: Tables, state: BattleState, attacker, defender, move) -> jnp.ndarray:
    """``int32`` PP ``BattleControllerPlayer_DecrementPP`` (battle_controller_player.c:2235-2270)
    takes for ``moveTemp`` ``move`` of ``attacker`` with ``defender`` (the battler
    ``BattleSystem_Defender`` picked at the fight command, ``NO_DEFENDER`` for BATTLER_NONE):
    1, plus, when there is a defender (:2238; ``skipPressureCheck`` is set only by script
    commands that run after this, battle_script.c:3521, :4648, :4701, :5395, :7599, and
    BattleContext_Init clears it for the next action, battle_lib.c:2014), for Imprison (by move
    id, :2239-2240) or a counting range (``PRESSURE_COUNT_*``) one per living Pressure holder of
    the other side / other than the user (a fainted holder and the user's own Pressure never
    count), for a ``PRESSURE_NONE`` range nothing, and otherwise (the default case, :2261-2266)
    one when the defender is not the user and its ability is Pressure, whatever its HP. The
    caller spends ``min(cost, ppCur)`` (:2276-2280). Ruling G-8."""
    fmt = format_of(state)
    move_range = tables.move_target[move]
    # D2-15: COUNT_ALIVE_BATTLERS_EXCEPT_ME (every living holder but the user, its partner
    # included) and COUNT_ALIVE_BATTLERS_THEIR_SIDE (living holders of the other side), battle_lib.c:
    # 2905-2967. At B = 2 the only other battler is the foe, so the two sums agree.
    except_me = jnp.zeros((), _I32)
    their_side = jnp.zeros((), _I32)
    for battler in range(state.active.shape[0]):
        mon = state.active[battler]
        alive_holder = pressure_holder(state, battler) & (state.hp[party(fmt, battler), mon] > 0)
        except_me = except_me + (alive_holder & (attacker != battler)).astype(_I32)
        their_side = their_side + (alive_holder & (side_of(battler) != side_of(attacker))).astype(
            _I32
        )
    by_side = (move == MOVE_IMPRISON) | jnp.isin(
        move_range, jnp.asarray(PRESSURE_COUNT_THEIR_SIDE, move_range.dtype)
    )
    by_all = jnp.isin(move_range, jnp.asarray(PRESSURE_COUNT_EXCEPT_ME, move_range.dtype))
    counted = by_side | by_all
    holders = jnp.where(by_side, their_side, except_me)
    silent = jnp.isin(move_range, jnp.asarray(PRESSURE_NONE, move_range.dtype))
    defender = jnp.asarray(defender, _I32)
    has_defender = defender != NO_DEFENDER
    named = jnp.clip(defender, 0, state.active.shape[0] - 1)
    single = (defender != attacker) & pressure_holder(state, named)
    extra = jnp.where(counted, holders, jnp.where(silent, 0, single.astype(_I32)))
    return 1 + jnp.where(has_defender, extra, 0)


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


def targets_foe_side(tables: Tables, move) -> jnp.ndarray:
    """``bool``: ``move``'s range is ``RANGE_OPPONENT_SIDE`` (Spikes, Toxic Spikes, Stealth Rock):
    it aims at the foe's side, not a battler. The one test of that range (doubles design section 5
    rule 6): ``defender_always``, the semi-invulnerable test of CheckMoveHitOverrides (bcp.c:3035,
    ``step``) and the Future Sight hit's (B10-14)."""
    return tables.move_target[jnp.asarray(move)] == MoveTarget.RANGE_OPPONENT_SIDE


def defender_always(tables: Tables, move, ghost=False) -> jnp.ndarray:
    """``BattleSystem_Defender`` (battle_lib.c:1599) finds a defender for the move
    whatever the foe's HP: the user (``targets_user``) or, for ``RANGE_OPPONENT_SIDE``
    (Stealth Rock, ``targets_foe_side``), the foe's battler slot
    (``BattleSystem_RandomOpponent``, :4194). The no-target check (battle_controller_player.c:2306)
    never fails such a move."""
    return targets_user(tables, move, ghost) | targets_foe_side(tables, move)


def _set(state: BattleState, field: str, index, value, cond) -> BattleState:
    """``state.<field>[index] = value`` where ``cond``."""
    arr = getattr(state, field)
    index = field_index(state, field, index)
    new = jnp.where(cond, value, arr[index]).astype(arr.dtype)
    return state._replace(**{field: arr.at[index].set(new)})


def hp_update(state: BattleState, side, slot, delta, cond) -> BattleState:
    """``BATTLE_SUBSCRIPT_UPDATE_HP`` with hpCalcTemp = ``delta`` (negative: a loss) on
    ``side``'s party ``slot`` where ``cond``: its HP becomes ``curHP + delta`` within [0, maxHP],
    then ``turnFlags[battler].assuranceDamageMask``: subscript_update_hp.s:18-19 stores
    hpCalcTemp unless it is above 0, so a loss sets the flag (``ts_assurance``), a zero update
    clears it and a heal leaves it. The flag is per battler: a switch keeps it (only
    BattleSystem_CleanupFaintedMon, battle_lib.c:2190-2193, and SetupNextTurn, :2231, clear it),
    and only the actions read it (Assurance). Every action-phase HP loss goes through here
    (B13-17)."""
    fmt = format_of(state)
    delta = jnp.asarray(delta).astype(_I32)
    cond = jnp.asarray(cond, jnp.bool_)
    hp = state.hp[party(fmt, side), slot].astype(_I32)
    new_hp = jnp.clip(hp + delta, 0, state.max_hp[party(fmt, side), slot].astype(_I32))
    state = _set(state, "hp", (side, slot), new_hp, cond)
    old = state.ts_assurance[side]
    mark = jnp.where(cond & (delta < 0), 1, jnp.where(cond & (delta == 0), 0, old))
    return state._replace(ts_assurance=state.ts_assurance.at[side].set(mark.astype(jnp.int8)))


def unlock_move(state: BattleState, side, cond) -> BattleState:
    """``Battler_UnlockMoveChoice`` (battle_lib.c:1834-1841) on ``side`` where ``cond``: the move
    lock, Bide's turns and stored damage (hidden state, 0 outside a Bide) and the Rollout
    count go to 0. It clears ``VOLATILE_CONDITION_MOVE_LOCKED``, not ``moveLockedInto`` nor
    ``VOLATILE_CONDITION_UPROAR``: an uproaring battler stays locked into its move
    (``Battler_CanPickCommand``, battle_lib.c:6224-6235), so ``locked_move`` stays while
    ``uproar`` is up (ruling B5-5)."""
    # :1838 also clears MOVE_EFFECT_SEMI_INVULNERABLE (B10-12).
    for field in ("bide_turns", "bide_damage", "rollout_count", "semi_invulnerable"):
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
    defender with HP left (:5-6). The hit leaves the status alone, so it is still paralysis.
    Wake-Up Slap (effect_script_0217.s:9) is the same with MON_CONDITION_SLEEP and
    subscript_heal_target_sleep.s (:5-6: the sleep bits off, any count), B13-16."""
    fmt = format_of(state)
    cured = jnp.where(h.param[0] == POWER_IF_TARGET_ASLEEP, Status.SLEEP, Status.PARALYSIS).astype(
        state.status.dtype
    )
    cond = (
        (h.effect == _E.DOUBLE_POWER)
        & (h.param[1] == 1)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.status[party(fmt, h.target_side), h.target] == cured)
        # effect_script_0171.s:5: CheckSubstitute before the damage jumps to a plain hit (no x2,
        # no cure). The flag cannot be set before the move's own (only) hit, so this is the bit
        # at the script's start: up then exactly when this hit took the substitute.
        & ~h.sub_hit
    )
    return _set_status(state, h.target_side, h.target, Status.NONE, cond)


def swagger_fails(state: BattleState, target_side, stat=Stat.ATK) -> jnp.ndarray:
    """``bool``: subscript_swagger.s / subscript_flatter.s fail the move (``MOVE_STATUS_FAILED``)
    on a defender behind a substitute (:5) or with ``stat`` (Attack / Sp. Atk, ``move_param[m,
    0]``) at +6 and already confused (:6-8)."""
    maxed = (state.stages[target_side, stat] >= STAGE_MAX) & (state.confusion[target_side] > 0)
    return (state.substitute_hp[target_side] != 0) | maxed


def _swagger(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Swagger (effect_script_0118.s, a direct ON_HIT effect; subscript_swagger.s) and Flatter
    (effect_script_0166.s; subscript_flatter.s, the same script on Sp. Atk +1, B12-2): the stat
    (``move_param[m, 0]``) raised by ``move_param[m, 1]`` below +6 (``UPDATE_STAT_STAGE``), then
    subscript_confuse.s unless the defender is confused (:13, no message): ``Random 3, 2`` turns
    (``outcome.duration``). In subscript_confuse.s (a direct side effect) the defender's
    ignorable Own Tempo comes first (:12 -> :_141): no confusion, "prevents confusion" (:73,
    TAG_NICKNAME_ABILITY: the AI learns it), the raise kept (H-2); then Safeguard (:26). The
    failure (``swagger_fails``, a substitute included) skips AfterMoveEffects (``step``)."""
    stat, stages = h.param[0], h.param[1]
    cond = (h.effect == _E.SWAGGER) & h.landed & _alive(state, h.target_side, h.target)
    cond = cond & ~swagger_fails(state, h.target_side, stat)
    below = state.stages[h.target_side, stat] < STAGE_MAX
    state = change_stage(state, h.target_side, stat, stages, cond & below)
    # subscript_confuse.s (a direct side effect): Own Tempo (:12, H-2), the substitute is
    # swagger_fails'; then Safeguard (:26, B10-7), after the raise.
    calls = cond & (state.confusion[h.target_side] == 0)
    tempo = calls & (h.target_ignorable == AbilityId.OWN_TEMPO)
    state = _reveal_confusion_block(state, h, tempo)
    confuse = calls & ~tempo & (state.safeguard[side_of(h.target_side)] == 0)
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], confuse)


def _reveal_confusion_block(state: BattleState, h: _Hit, cond) -> BattleState:
    """subscript_confuse.s:12 -> :_141, :73: a direct confusion that an ignorable Own Tempo stops
    prints "prevents confusion" (TAG_NICKNAME_ABILITY on BTLSCR_SIDE_EFFECT_MON,
    battle_script.c:11798-11810), where ``cond`` (H-2)."""
    from nuzline.engine.hooks import CONFUSION_BLOCKED, reveal_ability  # hooks imports this module

    return reveal_ability(state, h.target_side, h.target, CONFUSION_BLOCKED, cond)


# The two stats in move_param[m, 0..1], in order (Attack and Speed: Dragon Dance; Attack and
# Defense: Bulk Up; Sp. Atk and Sp. Def: Calm Mind; Defense and Sp. Def: Cosmic Power),
# subscript_user_*_up_1_stage.s:13-16 (B9-10, B12-1).


def dragon_dance_fails(state: BattleState, side, first, second) -> jnp.ndarray:
    """``bool``: both of the user's stages are at +6 (``first`` and ``second``): the subscript
    prints "won't go higher" and sets ``MOVE_STATUS_NO_MORE_WORK`` (:5-6, :21-29), which skips
    AfterMoveEffects."""
    return (state.stages[side, first] >= STAGE_MAX) & (state.stages[side, second] >= STAGE_MAX)


def belly_drum_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: subscript_belly_drum.s:3-6 fails (MOVE_STATUS_FAILED) at Attack +6 or with
    ``curHP <= Divide(maxHP, 2)``."""
    fmt = format_of(state)
    mon = state.active[side]
    half = jnp.maximum(state.max_hp[party(fmt, side), mon].astype(_I32) // 2, 1)
    return (state.stages[side, Stat.ATK] >= STAGE_MAX) | (
        state.hp[party(fmt, side), mon].astype(_I32) <= half
    )


def substitute_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: Substitute ends in ``MOVE_STATUS_NO_MORE_WORK``: subscript_try_substitute.s:7,
    the user is behind one already (CheckSubstitute: its bit; its flag is clear in its own
    action); :8, ``BtlCmd_TrySubstitute`` (battle_script.c:5215-5218) at ``curHP <=
    Divide(maxHP, 4)``."""
    fmt = format_of(state)
    mon = state.active[side]
    quarter = divide(state.max_hp[party(fmt, side), mon].astype(_I32), 4)
    return (state.substitute_hp[side] != 0) | (
        state.hp[party(fmt, side), mon].astype(_I32) <= quarter
    )


def _substitute(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Substitute (effect_script_0079.s, a direct ON_HIT subscript_try_substitute.s on the user, a
    RANGE_USER defender): ``BtlCmd_TrySubstitute`` (battle_script.c:5211-5226) sets
    ``substituteHP`` to Divide(maxHP, 4) and clears the user's ``VOLATILE_CONDITION_BIND``
    (:5222), then :12-13 the user loses it through UPDATE_HP (B10-4). A failing one never gets
    here (``primary_effect_fails``)."""
    fmt = format_of(state)
    cond = (h.effect == _E.SUBSTITUTE) & h.landed & ~substitute_fails(state, h.side)
    quarter = divide(state.max_hp[party(fmt, h.side), h.user].astype(_I32), 4)
    state = _set(state, "substitute_hp", h.side, quarter, cond)
    state = _set(state, "trapped", h.side, 0, cond)
    state = _set(state, "trap_move", h.side, 0, cond)
    return hp_update(state, h.side, h.user, -quarter, cond)


def _pain_split(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Pain Split (effect_script_0091.s, a direct ON_HIT subscript_pain_split.s): avg = (the
    attacker's curHP + the defender's) / 2 (:7-10, OPCODE_DIV), then each battler's HP changes
    by avg - its curHP through UPDATE_HP (:12-22, capped at its max HP; SKIP_SPRITE_BLINK, a
    status move: no resist berry), B10-5. A substitute fails it first (:5, ``h.shielded``;
    ``primary_effect_fails``)."""
    fmt = format_of(state)
    cond = (h.effect == _E.PAIN_SPLIT) & h.landed & ~h.shielded
    mine = state.hp[party(fmt, h.side), h.user].astype(_I32)
    theirs = state.hp[party(fmt, h.target_side), h.target].astype(_I32)
    avg = (mine + theirs) // 2
    # hpCalcTemp = avg - curHP for each (:11-13, :17-19).
    state = hp_update(state, h.side, h.user, avg - mine, cond)
    return hp_update(state, h.target_side, h.target, avg - theirs, cond)


def _encore(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Encore (effect_script_0090.s, a direct ON_HIT subscript_encore_start.s): unless
    ``encore_slot_of`` fails it, encoredMove = the target's last move, encoredMoveSlot its slot,
    encoredTurns = RandNext() % 5 + 3 (battle_script.c:4740-4742; ``outcome.duration`` of the
    user's side, ``DURATION_ENCORE``), B10-9."""
    fails, slot = encore_slot_of(tables, state, h.target_side)
    cond = (h.effect == _E.ENCORE) & h.landed & _alive(state, h.target_side, h.target) & ~fails
    state = _set(state, "encore_move", h.target_side, state.last_move[h.target_side], cond)
    state = _set(state, "encore_slot", h.target_side, slot, cond)
    return _set(state, "encore_turns", h.target_side, outcome.duration[h.side], cond)


def _safeguard(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Safeguard (BtlCmd_TrySafeguard, battle_script.c:5754-5772): safeguardTurns = 5 on the
    user's side (the field-condition pass counts it down, ``step.FIELD_SIDE_TIMERS``); with it up
    already the move fails (``primary_effect_fails``), B10-7."""
    cond = (h.effect == _E.SAFEGUARD) & h.landed & (state.safeguard[side_of(h.side)] == 0)
    return _set(state, "safeguard", h.side, SAFEGUARD_TURNS, cond)


def _mist(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Mist (effect_script_0046.s, a direct ON_HIT subscript_mist.s; BtlCmd_TryMist,
    battle_script.c:4300-4317): mistTurns = 5 on the user's side (the field-condition pass counts
    it down, ``step.FIELD_SIDE_TIMERS``, bcp.c:993-1004); with SIDE_CONDITION_MIST up already the
    move fails (``primary_effect_fails``), B12-23. mistUser (:4313) has no reader."""
    cond = (h.effect == _E.MIST) & h.landed & (state.mist[side_of(h.side)] == 0)
    return _set(state, "mist", h.side, MIST_TURNS, cond)


def _lucky_chant(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Lucky Chant (effect_script_0240.s, an indirect ON_HIT subscript_lucky_chant_start.s):
    the 3-bit counter of SIDE_CONDITION_LUCKY_CHANT (bits 12-14) set to 5 on the user's side
    (:5-6; the message is cosmetic). The field-condition pass counts it down
    (``step.FIELD_SIDE_TIMERS``, bcp.c:1062-1083); with the condition up already the move fails
    (:5, ``primary_effect_fails``). The crit it blocks is ``crit_blocked``'s. E3-B2-7."""
    cond = (h.effect == _E.LUCKY_CHANT) & h.landed & (state.lucky_chant[side_of(h.side)] == 0)
    return _set(state, "lucky_chant", h.side, LUCKY_CHANT_TURNS, cond)


def _belly_drum(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Belly Drum (effect_script_0142.s, a direct ON_HIT subscript_belly_drum.s; the user, a
    RANGE_USER defender, has HP): Attack set to +6, then Divide(maxHP, 2) through UPDATE_HP (:8-12).
    A failing one never gets here (``primary_effect_fails``)."""
    fmt = format_of(state)
    cond = (h.effect == _E.BELLY_DRUM) & h.landed & ~belly_drum_fails(state, h.side)
    state = change_stage(state, h.side, Stat.ATK, STAGE_MAX * 2, cond)
    half = jnp.maximum(state.max_hp[party(fmt, h.side), h.user].astype(_I32) // 2, 1)
    return hp_update(state, h.side, h.user, -half, cond)


def _dragon_dance(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Dragon Dance (effect_script_0212.s: direct TO_ATTACKER, the user has HP,
    battle_lib.c:1492-1504) and its kin (``dragon_dance_fails``): the two stats +1, each skipped
    on its own at +6 (:13-16, ``ChangeStatStage``'s ``jumpNoChange``)."""
    cond = (h.effect == _E.DRAGON_DANCE) & h.landed & _alive(state, h.side, h.user)
    state = change_stage(state, h.side, h.param[0], 1, cond)
    return change_stage(state, h.side, h.param[1], 1, cond)


def tickle_fails(state: BattleState, target_side, first, second) -> jnp.ndarray:
    """``bool``: subscript_target_atk_and_def_down_1_stage.s:5-6 finds both of the defender's
    stages at -6: "won't go lower" and ``MOVE_STATUS_NO_MORE_WORK`` (:24-33), B12-3."""
    low_first = state.stages[target_side, first] <= STAGE_MIN
    return low_first & (state.stages[target_side, second] <= STAGE_MIN)


def _tickle(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Tickle (effect_script_0205.s: a direct ``CHECK_HP | TO_DEFENDER`` subscript, the defender
    has HP, battle_lib.c:1492-1504): unless ``tickle_fails``, ``move_param[m, 0]`` then ``[1]``
    -1 through UPDATE_STAT_STAGE (:11-15), each its own ``BtlCmd_ChangeStatStage``
    (``_foe_stage``: Clear Body, Hyper Cutter on Attack, -6, the substitute), B12-3. A general miss
    still runs the subscript (``MOVE_SIDE_EFFECT_CHECK_HP``) but :9 fails it (no change)."""
    cond = (h.effect == _E.TICKLE) & h.landed & _alive(state, h.target_side, h.target)
    cond = cond & ~tickle_fails(state, h.target_side, h.param[0], h.param[1])
    for stat in (h.param[0], h.param[1]):
        one = h._replace(param=jnp.stack([stat, -jnp.ones_like(stat), jnp.zeros_like(stat)]))
        state = _foe_stage(state, one, cond)
    return state


def _focus_energy(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Focus Energy (effect_script_0047.s: a direct TO_ATTACKER subscript_focus_energy.s:6 unless
    the flag is set, :5): the user's VOLATILE_CONDITION_FOCUS_ENERGY, B12-4."""
    cond = (h.effect == _E.FOCUS_ENERGY) & h.landed & _alive(state, h.side, h.user)
    return _set(state, "focus_energy", h.side, 1, cond & (state.focus_energy[h.side] == 0))


def _haze(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Haze (effect_script_0025.s: an indirect ON_HIT subscript_reset_all_stat_stages.s):
    BtlCmd_ResetAllStatChanges (battle_script.c:3534-3549) loops every battler: its stages to
    neutral and its Focus Energy cleared, B12-5."""
    cond = (h.effect == _E.HAZE) & h.landed
    stages = jnp.where(cond, jnp.zeros_like(state.stages), state.stages)
    focus = jnp.where(cond, jnp.zeros_like(state.focus_energy), state.focus_energy)
    return state._replace(stages=stages, focus_energy=focus)


def copy_stat_stages(state: BattleState, side, target_side, cond) -> BattleState:
    """BtlCmd_CopyStatStages (battle_script.c:6042-6053) where ``cond``: every stage of
    ``target_side``'s battler onto ``side``'s, and its Focus Energy ORed into ``side``'s (:6050);
    no failure check, B12-6."""
    copied = jnp.where(cond, state.stages[target_side], state.stages[side])
    state = state._replace(stages=state.stages.at[side].set(copied))
    focus = state.focus_energy[side] | state.focus_energy[target_side]
    return _set(state, "focus_energy", side, focus, cond)


def _psych_up(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Psych Up (effect_script_0143.s:5): ``copy_stat_stages``. The command runs in the effect
    script, before TryMove (bcp.c:3211-3213), so a semi-invulnerable miss copies too
    (``step._use_move``); this is the path of a move that connected, B12-6."""
    return copy_stat_stages(state, h.side, h.target_side, (h.effect == _E.PSYCH_UP) & h.landed)


def _stage_swap(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Power Swap / Guard Swap (effect_script_0243.s / _0244.s: an indirect ON_HIT
    subscript_exchange_{atk_and_spatk,def_and_spdef}_stages.s:5-12): the attacker's and the
    defender's stages of ``move_param[m, 0]`` and ``[1]`` swapped; no failure check, B12-7."""
    cond = (h.effect == _E.STAGE_SWAP) & h.landed
    for stat in (h.param[0], h.param[1]):
        mine, theirs = state.stages[h.side, stat], state.stages[h.target_side, stat]
        stages = state.stages.at[h.side, stat].set(jnp.where(cond, theirs, mine))
        state = state._replace(
            stages=stages.at[h.target_side, stat].set(jnp.where(cond, mine, theirs))
        )
    return state


def captivate_fails(tables, state: BattleState, side, target_side) -> tuple:
    """``(fails, by_ability)`` of effect_script_0265.s on ``target_side``'s battler: its Oblivious
    as ``side``'s battler sees it (``CheckIgnorableAbility``, :5, "made ... ineffective", tagging
    the ability), a genderless attacker or defender (:6-7), or the same gender (:8-9); each ends
    the move with ``MOVE_STATUS_NO_MORE_WORK`` (:47, :56), B12-8."""
    fmt = format_of(state)
    mon = state.active[target_side]
    own = battler_decomp_ability(tables, state, target_side, mon)
    mold = battler_ability(state, side) == AbilityId.MOLD_BREAKER
    by_ability = ~mold & (own == STAT_MOVE_ABILITY_IDS["Oblivious"])
    mine = state.gender[party(fmt, side), state.active[side]]
    theirs = state.gender[party(fmt, target_side), mon]
    genderless = (mine == GENDER_NONE) | (theirs == GENDER_NONE)
    return by_ability | genderless | (mine == theirs), by_ability


def _captivate(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Captivate (effect_script_0265.s): unless ``captivate_fails``, a direct TO_DEFENDER
    ``move_param[m, 0]`` change of ``move_param[m, 1]`` (Sp. Atk -2, :10) through
    ``_foe_stage``; a blocking Oblivious is tagged (on a move that did not connect too: the
    effect script runs before TryMove, bcp.c:3211-3213, ``step._use_move``), B12-8."""
    run = (h.effect == _E.CAPTIVATE) & h.landed & _alive(state, h.target_side, h.target)
    fails, by_ability = captivate_fails(tables, state, h.side, h.target_side)
    state = know_ability(state, h.target_side, h.target, run & by_ability)
    return _foe_stage(state, h, run & ~fails)


def acupressure_valid(state: BattleState, side) -> jnp.ndarray:
    """``bool[7]``: which of ``ACUPRESSURE_STATS`` (BATTLE_STAT_ATTACK .. EVASION, the decomp's
    order: Speed before the specials) ``side``'s battler has below +6 (battle_script.c:
    8943-8947)."""
    idx = jnp.asarray([int(x) for x in ACUPRESSURE_STATS], _I32)
    return state.stages[side][idx] < STAGE_MAX


def acupressure_fails(state: BattleState, defender) -> jnp.ndarray:
    """``bool``: effect_script_0226.s fails (MOVE_STATUS_FAILED, :8-9): its ``defender``'s side
    (``defender_side``: the user in singles) is behind a substitute (:5, CheckSubstitute on
    BTLSCR_DEFENDER) or has no stage below +6 (:6, BoostRandomStatBy2 on the defender)."""
    return (state.substitute_hp[defender] != 0) | ~jnp.any(acupressure_valid(state, defender))


def acupressure_stat(state: BattleState, side, draw) -> jnp.ndarray:
    """``int32``: the ``ids.Stat`` Acupressure raises: ``stats[RandNext() % validStats]``
    (:8950), ``draw`` being the RandNext() value (``Outcome.acupressure``, in [0, 420)) and
    ``validStats`` the count of ``acupressure_valid``'s stats now, ranked in decomp order."""
    valid = acupressure_valid(state, side)
    rank = jnp.cumsum(valid.astype(_I32)) - 1
    count = jnp.maximum(jnp.sum(valid.astype(_I32)), 1)
    hit = valid & (rank == jnp.asarray(draw, _I32) % count)
    order = jnp.asarray([int(x) for x in ACUPRESSURE_STATS], _I32)
    return jnp.sum(jnp.where(hit, order, 0))


def _acupressure(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Acupressure (effect_script_0226.s): on its defender (RANGE_USER_OR_ALLY, ``defender_side``:
    the user in singles), unless ``acupressure_fails``, a direct TO_DEFENDER +2
    (MOVE_SUBSCRIPT_PTR_ATTACK_UP_2_STAGES + the stat) on ``acupressure_stat``; the stat is below
    +6, so it rises (B12-9). The draw is the acting battler's (``Outcome.acupressure[h.side]``)."""
    defender = defender_side(h.target_side, h.defender)
    cond = (h.effect == _E.ACUPRESSURE) & h.landed
    cond = cond & _alive(state, defender, state.active[defender])
    cond = cond & ~acupressure_fails(state, defender)
    stat = acupressure_stat(state, defender, outcome.acupressure[h.side])
    return change_stage(state, defender, stat, 2, cond)


def _power_trick(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Power Trick (effect_script_0238.s: an indirect ON_HIT subscript_user_swap_atk_and_def.s):
    the user's MOVE_EFFECT_POWER_TRICK toggled (:5, XOR); its BattleMon Attack and Defense
    swapped (:6-9) is the bit itself (``damage.battle_stats``), B12-10."""
    cond = (h.effect == _E.POWER_TRICK) & h.landed
    return _set(state, "power_trick", h.side, 1 - state.power_trick[h.side], cond)


# --- E2 batch 12, the healing family (B12-12..) ----------------------------------------------


def rest_fails(tables, state: BattleState, side) -> tuple:
    """``(fails, by_ability)`` of subscript_rest.s for ``side``'s battler: its own Insomnia or
    Vital Spirit (``CheckAbility``, Battler_Ability: no Mold Breaker, :7-8, "stayed awake",
    tagging the ability), its sleep (:9), an uproar on the field unless it has Soundproof
    (:10-11), its HP full (:14-15); each ends the move with ``MOVE_STATUS_NO_MORE_WORK`` (:73),
    B12-12."""
    fmt = format_of(state)
    mon = state.active[side]
    own = battler_decomp_ability(tables, state, side, mon)
    ids = STAT_MOVE_ABILITY_IDS
    by_ability = (own == ids["Insomnia"]) | (own == ids["Vital Spirit"])
    asleep = state.status[party(fmt, side), mon] == Status.SLEEP
    uproar = (own != ids["Soundproof"]) & fcm.uproar_any(state)
    full = state.hp[party(fmt, side), mon] == state.max_hp[party(fmt, side), mon]
    return by_ability | asleep | uproar | full, by_ability


def _rest(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Rest (effect_script_0037.s: a direct ON_HIT subscript_rest.s): unless ``rest_fails``, the
    user's status becomes sleep for ``REST_TURNS`` (any other status cleared first, :24-25,
    :32) and it regains ``maxHP`` through UPDATE_HP, capped (:14, :36), B12-12."""
    fmt = format_of(state)
    cond = (h.effect == _E.REST) & h.landed & _alive(state, h.side, h.user)
    fails, by_ability = rest_fails(tables, state, h.side)
    state = know_ability(state, h.side, h.user, cond & by_ability)
    ok = cond & ~fails
    state = _set_status(state, h.side, h.user, Status.SLEEP, ok)
    state = _set(state, "status_counter", (h.side, h.user), REST_TURNS, ok)
    return _set(state, "hp", (h.side, h.user), state.max_hp[party(fmt, h.side), h.user], ok)


def weather_heal_amount(tables, state: BattleState, side) -> jnp.ndarray:
    """``int32``: BtlCmd_WeatherHPRecovery (battle_script.c:5966-5979) for ``side``'s battler:
    ``maxHP / 2`` with no weather or a weather lock (NO_WEATHER, include/battle/common.h:65-67;
    deep fog is a weather), ``Divide(maxHP * 20, 30)`` with any SUNNY bit, else
    ``Divide(maxHP, 4)`` (script.c:5970-5976)."""
    fmt = format_of(state)
    max_hp = state.max_hp[party(fmt, side), state.active[side]].astype(_I32)
    w = fcm.weather_bits(state)
    no_weather = ~fcm.any_weather(w) | weather_suppressed(tables, state)
    sun = fcm.sun(w)
    return jnp.where(
        no_weather, max_hp // 2, jnp.where(sun, divide(max_hp * 20, 30), divide(max_hp, 4))
    )


def _weather_heal(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Morning Sun, Synthesis, Moonlight (effect_script_0132.s): ``weather_heal_amount`` through
    subscript_present_heal.s -> subscript_recover_hp.s on the user, its defender (RANGE_USER):
    "HP is full!" at full HP (``recovers_at_full_hp``), else the heal, capped, B12-13."""
    fmt = format_of(state)
    hp = state.hp[party(fmt, h.side), h.user].astype(_I32)
    max_hp = state.max_hp[party(fmt, h.side), h.user].astype(_I32)
    cond = (h.effect == _E.WEATHER_HEAL) & h.landed & (hp != max_hp)
    healed = jnp.minimum(hp + weather_heal_amount(tables, state, h.side), max_hp)
    return _set(state, "hp", (h.side, h.user), healed, cond)


def _wish(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Wish (effect_script_0179.s): BtlCmd_TryWish (battle_script.c:6373-6385) sets the user's
    battler's wishTurns to ``WISH_TURNS`` unless set (MOVE_STATUS_FAILED, ``primary_effect_fails``);
    wishTarget only names the Pokemon in the message (bcp.c:1096), B12-14."""
    cond = (h.effect == _E.WISH) & h.landed & (state.wish_timer[h.side] == 0)
    return _set(state, "wish_timer", h.side, WISH_TURNS, cond)


def heal_bell_spared(tables, state: BattleState, side, move) -> jnp.ndarray:
    """``bool[SLOTS]``: the party slots of ``side`` Heal Bell leaves statused: the user (its
    active battler) with Soundproof as Battler_Ability (BtlCmd_TryPartyStatusRefresh, battle_script.c:
    5038-5043), any other party Pokemon with Soundproof unless the user has Mold Breaker
    (BtlIOCmd_RefreshPartyStatus, battle_io_command.c:645-653); Aromatherapy spares none."""
    fmt = format_of(state)
    # The party Pokemon's own MON_DATA_ABILITY (battle_io_command.c:648): raw.
    abilities = decomp_abilities(tables, state)[party(fmt, side)]
    sound = abilities == STAT_MOVE_ABILITY_IDS["Soundproof"]
    user = state.active[side]
    # Raw BattleMon.ability in the message (battle_controller.c:1539, battle_io_command.c:645).
    mold = state.ability[party(fmt, side), user] == AbilityId.MOLD_BREAKER
    slots = jnp.arange(abilities.shape[0])
    # The user's own: Battler_Ability (battle_script.c:5038).
    user_sound = battler_decomp_ability(tables, state, side) == STAT_MOVE_ABILITY_IDS["Soundproof"]
    spared = jnp.where(slots == user, user_sound, sound & ~mold)
    return spared & (jnp.asarray(move, _I32) == MOVE_HEAL_BELL)


def _heal_bell(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Heal Bell, Aromatherapy (effect_script_0102.s: an indirect ON_HIT subscript_heal_bell.s):
    every Pokemon of the user's party loses its status but ``heal_bell_spared`` (the user's own
    Soundproof is tagged by the "blocks" message, subscript_heal_bell.s:18-21), B12-15."""
    fmt = format_of(state)
    cond = (h.effect == _E.HEAL_BELL) & h.landed
    spared = heal_bell_spared(tables, state, h.side, h.move)
    state = know_ability(state, h.side, h.user, cond & spared[h.user])
    cure = cond & ~spared & (state.species[party(fmt, h.side)] != 0)
    status = jnp.where(cure, Status.NONE, state.status[party(fmt, h.side)]).astype(
        state.status.dtype
    )
    counter = jnp.where(cure, 0, state.status_counter[party(fmt, h.side)]).astype(
        state.status_counter.dtype
    )
    state = state._replace(
        status=state.status.at[party(fmt, h.side)].set(status),
        status_counter=state.status_counter.at[party(fmt, h.side)].set(counter),
    )
    # BtlCmd_TryPartyStatusRefresh (battle_script.c:5039-5070) clears VOLATILE_CONDITION_NIGHTMARE
    # with the status of the user (unless its Soundproof spares it, :5038-5043) and of its doubles
    # partner (unless spared, :5049-5053): each battler of the user's party whose active slot the
    # refresh cures (G-32).
    for battler in range(fmt.B):
        mine = jnp.asarray(party(fmt, battler)) == jnp.asarray(party(fmt, h.side))
        state = clear_nightmare(state, battler, mine & cure[state.active[battler]])
    return state


def refresh_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: subscript_recover_psn_prz_brn.s:5, :15-16: the user has none of
    ``REFRESH_STATUSES`` (MOVE_STATUS_FAILED)."""
    fmt = format_of(state)
    status = state.status[party(fmt, side), state.active[side]]
    return ~jnp.isin(status, jnp.asarray(REFRESH_STATUSES, status.dtype))


def _refresh(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Refresh (effect_script_0193.s: a direct ON_HIT subscript_recover_psn_prz_brn.s): the
    user's status set to MON_CONDITION_NONE (:7) unless ``refresh_fails``, B12-16."""
    cond = (h.effect == _E.REFRESH) & h.landed & ~refresh_fails(state, h.side)
    state = _set_status(state, h.side, h.user, Status.NONE, cond)
    return _set(state, "status_counter", (h.side, h.user), 0, cond)


def psycho_shift_status(status) -> jnp.ndarray:
    """``int32``: the status subscript_give_target_own_status.s gives for the user's ``status``:
    sleep, poison, burn, paralysis in that test order (:5-22), else badly poisoned (:25)."""
    status = jnp.asarray(status, _I32)
    kept = (
        (status == Status.SLEEP)
        | (status == Status.POISON)
        | (status == Status.BURN)
        | (status == Status.PARALYSIS)
    )
    return jnp.where(kept, status, Status.TOXIC)


def psycho_shift_fails(state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: BtlCmd_CheckCanShareStatus (battle_script.c:7542-7553): the defender has a status
    or its substitute bit, or the attacker has none (MOVE_STATUS_FAILED)."""
    fmt = format_of(state)
    theirs = state.status[party(fmt, target_side), state.active[target_side]]
    mine = state.status[party(fmt, side), state.active[side]]
    return (theirs != Status.NONE) | (state.substitute_hp[target_side] != 0) | (mine == Status.NONE)


def _psycho_shift(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Psycho Shift (effect_script_0234.s): unless ``psycho_shift_fails``, a direct TO_DEFENDER
    subscript_give_target_own_status.s: the user's status (``psycho_shift_status``) through its
    status subscript as a direct effect (``_inflict``: the defender's immunities, Safeguard, the
    uproar, the sleep's ``Random 3, 2``), then the user's status cleared when the defender has
    one (:28-30), B12-17."""
    fmt = format_of(state)
    cond = (h.effect == _E.PSYCHO_SHIFT) & h.landed
    cond = cond & ~psycho_shift_fails(state, h.side, h.target_side)
    give = psycho_shift_status(state.status[party(fmt, h.side), h.user])
    shifted = h._replace(param=h.param.at[0].set(give.astype(h.param.dtype)))
    state = _inflict(tables, state, shifted, outcome, cond, True)
    took = cond & (state.status[party(fmt, h.target_side), h.target] != Status.NONE)
    state = _set_status(state, h.side, h.user, Status.NONE, took)
    return _set(state, "status_counter", (h.side, h.user), 0, took)


def _roost(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Roost (subscript_roost.s, a direct ON_HIT TO_ATTACKER effect): ``Divide(maxHP, 2)``
    through subscript_recover_hp.s (``recovers_at_full_hp``: "HP is full!" at full HP); the
    roosting flag is ``step``'s (``roosted``)."""
    fmt = format_of(state)
    hp = state.hp[party(fmt, h.side), h.user].astype(_I32)
    max_hp = state.max_hp[party(fmt, h.side), h.user].astype(_I32)
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
        (h.effect == _E.TAILWIND) & h.landed & (state.tailwind[side_of(h.side)] == 0),
    )
    for effect, field, cap in (
        (_E.SPIKES, "spikes", SPIKES_MAX),
        (_E.TOXIC_SPIKES, "toxic_spikes", TOXIC_SPIKES_MAX),
    ):
        layers = getattr(state, field)[side_of(h.target_side)].astype(_I32)
        state = _set(
            state,
            field,
            h.target_side,
            layers + 1,
            (h.effect == effect) & h.landed & (layers < cap),
        )
    return state


def _grudge(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Grudge (effect_script_0194.s, an indirect ON_HIT message): :6 sets the user's
    MOVE_EFFECT_GRUDGE. Its :5 fail branch (the bit already set) never runs: the user's move
    attempt cleared the bit first (``CHECK_STATUS_START``, bcp.c:2427). E3-B2-9."""
    return _set(state, "grudge", h.side, 1, (h.effect == _E.GRUDGE) & h.landed)


def try_grudge(
    state: BattleState, side, mon, slot, fainted_side, fainted_has_grudge, struggling, cond
) -> BattleState:
    """``BtlCmd_TryGrudge`` (battle_script.c:6743-6763), run by subscript_faint_check_destiny_bond.s:28
    and subscript_attack_then_switch_out.s:40: where ``cond`` (the defender fainted in this action)
    the fainted battler held MOVE_EFFECT_GRUDGE (``fainted_has_grudge``), the attacker ``side``'s
    battler (party slot ``mon``) is on the other side (``fainted_side``) with HP left and the move
    is not Struggle: its PP of the selected move slot ``slot`` (``ATTACKER_MOVE_SLOT``) goes to 0.
    The write is the BattleMon's: ``BattleMon_CopyToParty`` skips a Mimic-flagged slot, so the
    party's copy (``shadow_pp``) is untouched. E3-B2-9."""
    fmt = format_of(state)
    hit = cond & fainted_has_grudge & (side_of(side) != side_of(fainted_side))
    hit = hit & (state.hp[party(fmt, side), mon] > 0) & ~jnp.asarray(struggling, bool)
    pp = state.pp.at[party(fmt, side), mon, slot].set(
        jnp.where(hit, 0, state.pp[party(fmt, side), mon, slot]).astype(state.pp.dtype)
    )
    return state._replace(pp=pp)


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
    state = fcm.uproar_start(state, h.side, cond)
    state = _set(state, "locked_move", h.side, h.move, cond)
    return _set(state, "uproar", h.side, outcome.duration[h.side], cond)


def primary_effect_fails(
    tables, state: BattleState, side, target_side, defender, move, foe_claw=False, foe_custap=False
) -> jnp.ndarray:
    """``bool``: ``side``'s ``move``, if it lands, has a direct effect that ends it with
    ``MOVE_STATUS_FAILED`` or ``_NO_MORE_WORK``, which ``BattleControllerPlayer_CheckMoveFailure``
    (bcp.c:3312-3331) sends past AfterMoveEffects: Swagger (``swagger_fails``), Dragon Dance
    (``dragon_dance_fails``). ``target_side``: the battler the hit reads as its target;
    ``defender``: ``targets.resolve``'s (Acupressure's); ``foe_claw`` /
    ``foe_custap``: its quickClaw / custapBerry flags (``play_hit``'s)."""
    effect = tables.move_effect[jnp.asarray(move, _I32)]
    param = tables.move_param[jnp.asarray(move, _I32)].astype(_I32)
    return (
        ((effect == _E.SWAGGER) & swagger_fails(state, target_side, param[0]))
        | ((effect == _E.DRAGON_DANCE) & dragon_dance_fails(state, side, param[0], param[1]))
        | ((effect == _E.BELLY_DRUM) & belly_drum_fails(state, side))
        # subscript_try_substitute.s:7-8 (MOVE_STATUS_NO_MORE_WORK), B10-4.
        | ((effect == _E.SUBSTITUTE) & substitute_fails(state, side))
        # subscript_pain_split.s:5, :50-52 (MOVE_STATUS_FAILED), B10-5.
        | ((effect == _E.PAIN_SPLIT) & (state.substitute_hp[target_side] != 0))
        # BtlCmd_TrySafeguard (battle_script.c:5760-5762, MOVE_STATUS_FAILED), B10-7.
        | ((effect == _E.SAFEGUARD) & (state.safeguard[side_of(side)] != 0))
        # BtlCmd_TryMist (battle_script.c:4307-4309, MOVE_STATUS_FAILED), B12-23.
        | ((effect == _E.MIST) & (state.mist[side_of(side)] != 0))
        # effect_script_0240.s:5, :9-11 (MOVE_STATUS_FAILED), E3-B2-7.
        | ((effect == _E.LUCKY_CHANT) & (state.lucky_chant[side_of(side)] != 0))
        # Defog's evasion drop behind a Substitute (subscript_update_stat_stage.s:60), E3-B2-14.
        | (
            (effect == _E.DEFOG)
            & defog_stage_fails(state, target_side, ignorable(state, side, target_side), param[0])
        )
        # BtlCmd_TryEncore's failure (MOVE_STATUS_FAILED), B10-9.
        | ((effect == _E.ENCORE) & encore_slot_of(tables, state, target_side)[0])
        # BtlCmd_TrySpite's failure (subscript_reduce_target_pp.s:12-14, MOVE_STATUS_FAILED), E3-B2-4.
        | ((effect == _E.SPITE) & spite_slot(tables, state, target_side)[0])
        # effect_script_0178.s:5-6, subscript_copy_ability.s:6-9 (MOVE_STATUS_FAILED), E3-B2-5.
        | ((effect == _E.ROLE_PLAY) & role_play_fails(tables, state, side, target_side))
        # subscript_exchange_abilities.s:6-13 -> :156 (MOVE_STATUS_FAILED), review b2 #2.
        | ((effect == _E.SKILL_SWAP) & skill_swap_fails(tables, state, side, target_side))
        # subscript_tailwind_start.s:5, :18 and the effect scripts' TrySpikes / TryToxicSpikes
        # jumps (effect_script_0112.s, _0249.s): MOVE_STATUS_FAILED.
        | ((effect == _E.TAILWIND) & (state.tailwind[side_of(side)] != 0))
        | ((effect == _E.SPIKES) & (state.spikes[side_of(target_side)] >= SPIKES_MAX))
        | (
            (effect == _E.TOXIC_SPIKES)
            & (state.toxic_spikes[side_of(target_side)] >= TOXIC_SPIKES_MAX)
        )
        # subscript_mean_look.s:7-8, :48: MOVE_STATUS_FAILED.
        | ((effect == _E.MEAN_LOOK) & mean_look_fails(state, target_side))
        # subscript_lock_on.s:6, :33.
        | ((effect == _E.LOCK_ON) & lock_on_fails(state, target_side))
        # BtlCmd_TryPerishSong (battle_script.c:5482-5484): no battler eligible.
        | ((effect == _E.PERISH_SONG) & ~jnp.any(perish_song_eligible(state, side)))
        # subscript_curse_ghost.s:22-24 (a Ghost user's Curse).
        | ((effect == _E.CURSE) & ghost_user(state, side) & curse_fails(state, target_side))
        # subscript_yawn.s's failures (``yawn_blocks``).
        | ((effect == _E.YAWN) & yawn_blocks(tables, state, side, target_side)[0])
        # E2 batch 12. subscript_target_atk_and_def_down_1_stage.s:5-6, :24-33 (NO_MORE_WORK).
        | ((effect == _E.TICKLE) & tickle_fails(state, target_side, param[0], param[1]))
        # effect_script_0047.s:5, :10-11 (MOVE_STATUS_FAILED).
        | ((effect == _E.FOCUS_ENERGY) & (state.focus_energy[side] != 0))
        # effect_script_0265.s:5-9 (MOVE_STATUS_NO_MORE_WORK).
        | ((effect == _E.CAPTIVATE) & captivate_fails(tables, state, side, target_side)[0])
        # subscript_embargo_start.s:5-11, :57 (MOVE_STATUS_FAILED), B13-31.
        | ((effect == _E.EMBARGO) & embargo_fails(tables, state, target_side, foe_claw, foe_custap))
        # effect_script_0226.s:5, BtlCmd_BoostRandomStatBy2's jump (MOVE_STATUS_FAILED), on its
        # defender (RANGE_USER_OR_ALLY: defender_side, the user in singles).
        | (
            (effect == _E.ACUPRESSURE)
            & acupressure_fails(state, defender_side(target_side, defender))
        )
        # subscript_rest.s:73 (NO_MORE_WORK), BtlCmd_TryWish (FAILED), subscript_recover_psn_prz_
        # brn.s:15-16 and BtlCmd_CheckCanShareStatus (FAILED).
        | ((effect == _E.REST) & rest_fails(tables, state, side)[0])
        | ((effect == _E.WISH) & (state.wish_timer[side] != 0))
        | ((effect == _E.REFRESH) & refresh_fails(state, side))
        | ((effect == _E.PSYCHO_SHIFT) & psycho_shift_fails(state, side, target_side))
    )


def _stealth_rock(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.STEALTH_ROCK) & h.landed & (state.stealth_rock[side_of(h.target_side)] == 0)
    )
    return _set(state, "stealth_rock", h.target_side, 1, cond)


def _trap(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    # MOVE_SIDE_EFFECT_CHECK_HP_AND_SUBSTITUTE (battle_lib.c:1537-1544): the target survived the
    # hit and its substitute was not hit; subscript_bind_start.s:5 then CheckSubstitute.
    cond = (
        (h.effect == _E.TRAP)
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & (state.trapped[h.target_side] == 0)
        & ~h.shielded
    )
    # subscript_bind_start.s:10-12: the attacker's Grip Claw (HOLD_EFFECT_EXTEND_TRAPPING,
    # CheckItemHoldEffect: Battler_HeldItemEffect) sets the count to 6 after the draw (G-57).
    grip = holds(tables, state, h.side, ItemId.GRIP_CLAW)
    turns = jnp.where(grip, GRIP_CLAW_TURNS, outcome.duration[h.side])
    state = _set(state, "trapped", h.target_side, turns, cond)
    return _set(state, "trap_move", h.target_side, h.move, cond)


def _taunt(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.TAUNT)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.taunt[h.target_side] == 0)
    )
    return _set(state, "taunt", h.target_side, outcome.duration[h.side], cond)


def _torment(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Torment (effect_script_0165.s:5 ON_HIT; ``subscript_torment_start.s``): fails on a miss or a
    semi-invulnerable target (:6, ``h.effective``) and on a target that has the bit (:5); no
    Substitute test and no ability test, so it goes through a Substitute. No count: the bit stays
    until the battler leaves (E3-2-1)."""
    cond = (
        (h.effect == _E.TORMENT)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & (state.torment[h.target_side] == 0)
    )
    return _set(state, "torment", h.target_side, 1, cond)


def trick_room_runs(h: _Hit) -> jnp.ndarray:
    """``bool``: Trick Room's script runs on this hit (``_trick_room``; ``MoveHit.trick_room``)."""
    return (h.effect == _E.TRICK_ROOM) & h.landed


def _trick_room(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Trick Room (effect_script_0259.s:5-12): with any FIELD_CONDITION_TRICK_ROOM bit up the move
    clears them (:12, "restored the twisted dimensions"), else it sets FIELD_CONDITION_TRICK_ROOM_INIT
    (:6, the counter 5, ``ids.TRICK_ROOM_TURNS``). It never fails (:20-21 are unreachable) and has no
    accuracy roll (RANGE_FIELD, accuracy 0). ``step._trick_room_check`` counts it down
    (battle_controller_player.c:1822-1830); the order it reverses is ``order.action_order``'s
    (battle_lib.c:1432-1439). E3-2-4."""
    return fcm.trick_room_toggle(state, trick_room_runs(h))


def _worry_seed(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Worry Seed (effect_script_0247.s:5 ON_HIT | TO_DEFENDER; ``subscript_give_target_insomnia.s``):
    fails (:5-9) on a miss or a semi-invulnerable target (``h.effective``), behind a Substitute
    (:6 CheckSubstitute, ``h.shielded``), and when the defender's ``BattleMon.ability`` is Truant
    (:7) or Multitype (:8) or it holds a Griseous Orb (:9, the raw ``heldItem``): the raw reads
    are ``decomp_abilities`` and ``state.item``, which a Gastro Acid suppression does not hide.
    Otherwise (:11) ``BattleMon.ability`` becomes Insomnia, whatever it was (an Insomnia holder
    succeeds again, a suppressed one stays suppressed), and ``UpdateMonData`` tags it for the AI
    (``BattleAI_SetAbility``, battle_script.c:3073-3075: ``know_ability``). The write is the
    battler's: ``step._restore_ability`` puts its party Pokemon's back when it leaves
    (battle_lib.c:124); ``damage.decomp_abilities`` reads the written ability. The sleeper wakes
    at the move's end (``step._move_end``: ``hooks.recover_status_by_ability``,
    battle_controller_player.c:3965-3967; battle_lib.c:4404-4410)."""
    fmt = format_of(state)
    pf, mon = party(fmt, h.target_side), h.target
    raw = decomp_abilities(tables, state)[pf, mon]  # raw: subscript_give_target_insomnia.s:7-8
    cond = (
        (h.effect == _E.WORRY_SEED)
        & h.effective
        & _alive(state, h.target_side, mon)
        & ~h.shielded
        & (raw != ABILITY_DECOMP_IDS[AbilityId.TRUANT])
        & (raw != ABILITY_MULTITYPE)
        & (state.item[pf, mon] != ITEM_GRISEOUS_ORB)
    )
    # The write itself (:11): BattleMon.ability, raw.
    value = jnp.where(cond, AbilityId.INSOMNIA, state.ability[pf, mon]).astype(state.ability.dtype)
    state = state._replace(ability=state.ability.at[pf, mon].set(value))
    return know_ability(state, h.target_side, mon, cond)


def _gastro_acid(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Gastro Acid (effect_script_0239.s:5 ON_HIT | TO_DEFENDER;
    ``subscript_suppress_target_ability.s``): fails (:5-8, MOVE_STATUS_FAILED) behind a
    Substitute (:5 CheckSubstitute, ``h.shielded``), on a defender that has
    MOVE_EFFECT_ABILITY_SUPPRESSED already (:6), on a miss or a semi-invulnerable defender (:7,
    ``h.effective``), and on a raw Multitype ability (:8: ``decomp_abilities``, not the
    ``Battler_Ability`` the bit changes). Otherwise (:10) the defender's
    ``ability_suppressed`` is set: until it leaves (``VOLATILE_FIELDS``), kept by Baton Pass
    (``BATON_PASSED_FIELDS``). ``BattleMon.ability`` itself is not written, so no AI tag
    (the AI reads the bit, trainer_ai.c:1177-1178). E3-2-6."""
    fmt = format_of(state)
    raw = decomp_abilities(tables, state)[party(fmt, h.target_side), h.target]
    cond = (
        (h.effect == _E.GASTRO_ACID)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & ~h.shielded
        & (state.ability_suppressed[h.target_side] == 0)
        & (raw != ABILITY_MULTITYPE)
    )
    return _set(state, "ability_suppressed", h.target_side, 1, cond)


FLING_STATUSES = (
    (FlingEffect.PARALYZE, Status.PARALYSIS),
    (FlingEffect.POISON, Status.POISON),
    (FlingEffect.BADLY_POISON, Status.TOXIC),
    (FlingEffect.BURN, Status.BURN),
)


def _fling(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Fling's hit-time half (effect_script_0233.s:8 ``MOVE_SIDE_EFFECT_ON_HIT | TO_DEFENDER``
    ``MOVE_SUBSCRIPT_PTR_FLING``; the TryFling-time half and the removal are ``step._use_move``'s,
    ``hooks.fling_try``): ``BattleSystem_TriggerSecondaryEffect`` runs it after a hit that is not
    MOVE_STATUS_NO_EFFECTS / DID_NOT_HIT (battle_lib.c:1524-1529, ``h.effective``);
    ``subscript_fling.s:5-7`` ends it for a defender at 0 HP or behind a Substitute
    (``h.shielded``: CheckSubstitute), and ``BattleSystem_FlingItem`` zeroes ``flingScript`` for a
    defender with MOVE_EFFECT_EMBARGO (battle_lib.c:6145-6146). The script is the one
    ``flingScript`` named at TryFling. Starf's stat is the one TryFling picked from the stages
    before the hit (``h.fling_pick``, ``hooks.fling_try``: the defender's ``Outcome.starf`` draw,
    F-10, review b2 #4); for every other item the defender's state at the hit still selects the
    same script: between TryFling and here only a critical hit's Anger Point
    (subscript_critical_hit.s:10-15, ONE_HIT_CRITICAL bcp.c:3472 before ONE_HIT_TRIGGER_SECONDARY
    :3484) writes the defender (its Attack stage; Anger Point is not modelled: whoever adds it
    must also model that subscript's :16 ``msgTemp = 1``, which turns any flung stat raise into an
    Attack raise). The flung item's effect then acts on the defender: ``hooks.fling_effect`` (the
    heals, cures, stat raises; Lansat, Micle), and the
    INDIRECT flinch and status subscripts here, with the secondary-effect checks they share
    (Shield Dust, Inner Focus, Substitute, Safeguard, immunities, ``_inflict``; the flinch needs
    a target that has not moved, subscript_flinch_mon.s:5). The attacker's item is gone: the
    restore subscripts' ``PLUCK_CHECK`` keeps the defender's own item (the flag set,
    battle_lib.c:6150-6152), which is why nothing here removes one. E3-B2-1."""
    from nuzline.engine.hooks import fling_effect  # hooks imports this module

    item = h.fling_item
    kind = tables.item_fling_effect[item].astype(_I32)
    reached = (
        (h.effect == _E.FLING)
        & h.effective
        & ~h.shielded
        & _alive(state, h.target_side, h.target)
        & (state.embargo[h.target_side] == 0)
        & (item != 0)
        & (tables.item_fling_power[item] != 0)
    )
    state = fling_effect(tables, state, h.target_side, h.target, item, reached, h.fling_pick)
    flinch = (
        reached
        & (kind == FlingEffect.FLINCH)
        & ~h.target_moved
        & (h.target_ignorable != AbilityId.INNER_FOCUS)
        & ~_shield_dust(h)
    )
    old = state.flinched[h.target_side]
    state = state._replace(
        flinched=state.flinched.at[h.target_side].set(jnp.where(flinch, 1, old).astype(old.dtype))
    )
    for fling_kind, status in FLING_STATUSES:
        inflict = h._replace(param=h.param.at[0].set(int(status)))
        state = _inflict(tables, state, inflict, outcome, reached & (kind == fling_kind), False)
    return state


def _skill_swap(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Skill Swap (effect_script_0191.s:5, a direct ``CHECK_HP | TO_DEFENDER``
    ``subscript_exchange_abilities.s``; ``MOVE_FLAG_CAN_PROTECT`` is set
    (res/moves/skill_swap/data.json), so a Protect stops it; the subscript has no Substitute test).
    The script runs for a living defender (battle_lib.c:1492-1507); it fails (:5-13,
    MOVE_STATUS_FAILED) on ``MOVE_STATUS_MISSED | MOVE_STATUS_SEMI_INVULNERABLE`` (:5: the move
    status, not the defender's bit, so Lock-On / Mind Reader or either side's No Guard, which
    return before the semi-invulnerable test in CheckMoveHitOverrides, bcp.c:3016-3021, still
    swap: ``h.landed``), on a raw Wonder Guard (:6-7)
    or Multitype (:8-9) ability of either battler, on a held Griseous Orb of either (:10-11), and
    when both abilities are ABILITY_NONE (:12-13; the user's alone is fine). Every test reads the
    raw ``BattleMon.ability`` / ``heldItem`` (``decomp_abilities``, ``state.item``), so a Gastro
    Acid suppression hides nothing. Otherwise (:17-24) ``truant`` of both becomes ``totalTurns &
    1`` and the two battlers' ``BattleMon.ability`` swap; the suppression bit stays with its
    battler. The user's or the defender's new Slow Start (:25-38) cannot arise: Slow Start is an
    unsupported ability, which coverage refuses on any Pokemon, so no battler holds it to give.
    ``UpdateMonDataFromVar`` tags both abilities for the AI (``BattleAI_SetAbility``,
    battle_script.c:3402-3404, :12106-12109). The writes are the battlers': ``step._restore_ability``
    puts each party Pokemon's own back when it leaves (battle_lib.c:124). An entry ability one
    of them now holds acts at the move's end (``hooks.move_end_switch_in``: its announced flag is
    still clear, battle_lib.c:3720-3851)."""
    fmt = format_of(state)
    side, user, foe, target = h.side, h.user, h.target_side, h.target
    pu, pt = party(fmt, side), party(fmt, foe)
    cond = (
        (h.effect == _E.SKILL_SWAP)
        & h.landed
        & _alive(state, foe, target)
        & ~skill_swap_fails(tables, state, side, foe)
    )
    # Raw: subscript_exchange_abilities.s:21-24 swaps BATTLEMON_ABILITY.
    a_user, a_target = state.ability[pu, user], state.ability[pt, target]
    parity = (state.turn.astype(_I32) & 1).astype(state.truant.dtype)

    def put(arr, index, value):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    state = state._replace(
        ability=put(put(state.ability, (pu, user), a_target), (pt, target), a_user),
        truant=put(put(state.truant, side, parity), foe, parity),
    )
    state = know_ability(state, side, user, cond)
    return know_ability(state, foe, target, cond)


def skill_swap_fails(tables: Tables, state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: ``side``'s Skill Swap on ``target_side``'s battler, if it lands, ends with
    ``MOVE_STATUS_FAILED`` (subscript_exchange_abilities.s:6-13 -> :156): a raw Wonder Guard (:6-7)
    or Multitype (:8-9) ``BattleMon.ability`` of either battler, a held Griseous Orb of either
    (:10-11), or both abilities ABILITY_NONE (:12-13; the user's alone is fine). The miss and
    semi-invulnerable test (:5) is ``landed``'s. The failure goes to CheckMoveFailure's
    subscript_missed.s (bcp.c:3311-3327): the attack message shows (:5) and LOOP_FAINTED skips
    AfterMoveEffects (``primary_effect_fails``), review b2 #2."""
    fmt = format_of(state)
    user, target = state.active[side], state.active[target_side]
    pu, pt = party(fmt, side), party(fmt, target_side)
    raw = decomp_abilities(tables, state)  # raw: subscript_exchange_abilities.s:6-13
    mine, theirs = raw[pu, user], raw[pt, target]
    wonder = ABILITY_DECOMP_IDS[AbilityId.WONDER_GUARD]
    banned = jnp.asarray([wonder, ABILITY_MULTITYPE], raw.dtype)
    return (
        jnp.isin(mine, banned)
        | jnp.isin(theirs, banned)
        | (state.item[pu, user] == ITEM_GRISEOUS_ORB)
        | (state.item[pt, target] == ITEM_GRISEOUS_ORB)
        | ((mine == 0) & (theirs == 0))
    )


def role_play_fails(tables: Tables, state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: ``side``'s Role Play, if it lands, ends with ``MOVE_STATUS_FAILED``
    (effect_script_0178.s:5-6, :10-12 and subscript_copy_ability.s:6-9, :31-32; the attack message
    after them is skipped, so ``step`` clears the move's record): the user's ``Battler_Ability``
    (``CheckAbility``, battle_script.c:3145-3183; Multitype survives Gastro Acid, battle_lib.c:
    3090-3091) or raw ability is Multitype (effect script :5, subscript :8), its raw held item is a
    Griseous Orb (:6), or the defender's raw ``BattleMon.ability`` is Wonder Guard (subscript :6),
    Multitype (:7) or none (:9). The miss and semi-invulnerable test (:5) is ``landed``'s. Every
    read is raw, so a Gastro Acid suppression hides nothing. E3-B2-5."""
    fmt = format_of(state)
    user, target = state.active[side], state.active[target_side]
    raw = decomp_abilities(tables, state)
    mine, theirs = raw[party(fmt, side), user], raw[party(fmt, target_side), target]
    wonder = ABILITY_DECOMP_IDS[AbilityId.WONDER_GUARD]
    return (
        (battler_decomp_ability(tables, state, side, user) == ABILITY_MULTITYPE)
        | (mine == ABILITY_MULTITYPE)
        | (state.item[party(fmt, side), user] == ITEM_GRISEOUS_ORB)
        | (theirs == wonder)
        | (theirs == ABILITY_MULTITYPE)
        | (theirs == 0)
    )


def _role_play(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Role Play (effect_script_0178.s:5-7, a direct ``CHECK_HP | TO_DEFENDER``
    ``subscript_copy_ability.s``; no ``MOVE_FLAG_CAN_PROTECT`` (moves.json), so a Protect does not
    stop it, and the subscript has no Substitute test). The script runs for a living defender
    (battle_lib.c:1492-1507); unless ``role_play_fails`` (MOVE_STATUS_MISSED / _SEMI_INVULNERABLE is
    ``landed``'s, as Skill Swap's), the user's ``truant`` becomes ``totalTurns & 1`` (:11-13) and its
    ``BattleMon.ability`` the defender's raw one (:14-15: a Gastro Acid suppressed defender still
    gives it, the user's own suppression bit stays). Only the user changes. The AI learns the user's
    new ability (``UpdateMonDataFromVar`` SET, ``BattleAI_SetAbility``, battle_script.c:3401-3404;
    the GET at :14 reveals nothing). A copied Slow Start (:16-21) cannot arise: Slow Start is an
    unsupported ability, which coverage refuses on any Pokemon. As Skill Swap's, the write is the
    battler's (``step._restore_ability``, battle_lib.c:124) and an entry ability it now holds acts at
    the move's end (``hooks.move_end_switch_in``: no announced flag is touched). E3-B2-5."""
    fmt = format_of(state)
    side, user, foe, target = h.side, h.user, h.target_side, h.target
    cond = (
        (h.effect == _E.ROLE_PLAY)
        & h.landed
        & _alive(state, foe, target)
        & ~role_play_fails(tables, state, side, foe)
    )
    pu, pt = party(fmt, side), party(fmt, foe)
    parity = (state.turn.astype(_I32) & 1).astype(state.truant.dtype)
    new = jnp.where(cond, state.ability[pt, target], state.ability[pu, user])
    state = state._replace(
        ability=state.ability.at[pu, user].set(new.astype(state.ability.dtype)),
        truant=state.truant.at[side].set(jnp.where(cond, parity, state.truant[side])),
    )
    return know_ability(state, side, user, cond)


def _magic_coat(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Magic Coat (effect_script_0183.s:5, ``BtlCmd_TrySetMagicCoat``, battle_script.c:6459-6478):
    unless its user acts last (``waitingBattlers == 1``: no other battler's action is pending, a
    count over every battler, which ``step._use_move`` fails as MOVE_STATUS_FAILED, so such a
    Magic Coat never lands; review b2 #9), ``turnFlags[user].magicCoat`` is set at script time,
    before TryMove, so a Magic Coat that then went wrong still stands. The flag keeps ``1 + the
    user's moveSlot`` (the Pressure drain of its reflection, bcp.c:3083). The :8 message is
    cosmetic. E3-B2-6."""
    cond = (h.effect == _E.MAGIC_COAT) & h.landed
    return _set(state, "magic_coat", h.side, 1 + h.coat_slot, cond)


def _disable(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Disable (``BtlCmd_TryDisable``, battle_script.c:4581; the script skips it when the
    move missed): the target's last move must sit in one of its slots with PP left, and
    nothing may be disabled already. The target's ``last_move`` is ``movePrevByBattler``:
    0 after a move that failed, Struggle's id for Struggle (in no slot)."""
    fmt = format_of(state)
    target_moves = state.moves[party(fmt, h.target_side), h.target]
    last = state.last_move[h.target_side]
    in_slot = (target_moves == last) & (last != 0)
    has_pp = jnp.any(in_slot & (state.pp[party(fmt, h.target_side), h.target] > 0))
    cond = (
        (h.effect == _E.DISABLE) & h.effective & (state.disabled_move[h.target_side] == 0) & has_pp
    )
    state = _set(state, "disabled_move", h.target_side, last, cond)
    return _set(state, "disable_turns", h.target_side, outcome.duration[h.side], cond)


def _confuse_foe(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    # subscript_confuse.s:12 (H-2): an ignorable Own Tempo stops it before every other check and,
    # a direct effect, reveals itself (:_141).
    runs = (h.effect == _E.CONFUSE_FOE) & h.effective & _alive(state, h.target_side, h.target)
    tempo = runs & (h.target_ignorable == AbilityId.OWN_TEMPO)
    state = _reveal_confusion_block(state, h, tempo)
    cond = (
        runs
        & ~tempo
        & (state.confusion[h.target_side] == 0)
        & ~h.shielded  # subscript_confuse.s:23
        & (state.safeguard[side_of(h.target_side)] == 0)  # :26 (B10-7)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


def _secondary_confuse(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (
        (h.effect == _E.SECONDARY_CONFUSE)
        & h.effective
        & h.proc
        & _alive(state, h.target_side, h.target)
        & (h.target_ignorable != AbilityId.OWN_TEMPO)  # subscript_confuse.s:12, silent (H-2)
        & (state.confusion[h.target_side] == 0)
        & ~_shield_dust(h)
        & ~h.shielded  # the rolled-secondary gate (battle_lib.c:1580-1582); confuse :23
        & (state.safeguard[side_of(h.target_side)] == 0)  # :26 (B10-7)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


def _aqua_ring(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.AQUA_RING) & h.landed & (state.aqua_ring[h.side] == 0)
    return _set(state, "aqua_ring", h.side, 1, cond)


def magnet_rise_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: effect_script_0252.s:5-7 fails (:16, MOVE_STATUS_FAILED) a user with
    ``magnetRiseTurns`` (the turns, not the mask bit), with Levitate (``CheckAbility
    CHECK_HAVE``, through ``Battler_Ability``: none for an ingrained holder and under Gravity,
    battle_lib.c:3095-3103) or with ``MOVE_EFFECT_INGRAIN`` (G-66)."""
    levitate = (
        (battler_ability(state, side) == AbilityId.LEVITATE)
        & (state.ingrain[side] == 0)
        & ~fcm.gravity_on(state)
    )
    return (state.magnet_rise_turns[side] != 0) | levitate | (state.ingrain[side] != 0)


def _magnet_rise(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Magnet Rise (effect_script_0252.s:8-9): ``MOVE_EFFECT_MAGNET_RISE`` on the user's
    ``moveEffectsMask`` and ``magnetRiseTurns = MAGNET_RISE_TURNS``, set by the script itself
    (its ON_HIT subscript is a message). A failing user is ``step._use_move``'s ``failed``."""
    cond = (h.effect == _E.MAGNET_RISE) & h.landed & ~magnet_rise_fails(state, h.side)
    state = _set(state, "magnet_rise", h.side, 1, cond)
    return _set(state, "magnet_rise_turns", h.side, MAGNET_RISE_TURNS, cond)


def gravity_ground(state: BattleState, cond=True) -> BattleState:
    """The grounding loop of ``subscript_gravity_start.s`` (:13-85, per battler in speed order; the
    order changes nothing here), for every battler with HP: ``magnetRiseTurns`` goes to 0 (the
    ``MOVE_EFFECT_MAGNET_RISE`` bit stays), else a Fly / Bounce user (``MOVE_EFFECT_AIRBORNE``) is
    unlocked (``Battler_UnlockMoveChoice`` and ToggleVanish); the other branches (Gastro Acid,
    Ingrain, Levitate, a Flying type) only print. ``cond``: where the loop runs. G-67."""
    fmt = format_of(state)
    for side in range(fmt.B):
        alive = (state.hp[party(fmt, side), state.active[side]] > 0) & jnp.asarray(cond)
        floating = alive & (state.magnet_rise_turns[side] != 0)
        state = _set(state, "magnet_rise_turns", side, 0, floating)
        airborne = alive & ~floating & (state.semi_invulnerable[side] == SemiKind.AIRBORNE)
        state = unlock_move(state, side, airborne)
    return state


def _gravity(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Gravity (effect_script_0215.s, the indirect subscript_gravity_start.s): the field
    condition for ``GRAVITY_TURNS`` and the grounding loop (``gravity_ground``). With Gravity up
    the script fails (:3, :9-10: ``step._use_move``'s ``failed``). G-67."""
    cond = (h.effect == _E.GRAVITY) & h.landed & ~fcm.gravity_on(state)
    return gravity_ground(fcm.gravity_start(state, cond), cond)


def _mimic(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Mimic (effect_script_0082.s:3, a direct ON_HIT ``subscript_mimic.s``): ``BtlCmd_TryMimic``
    (battle_script.c:4476-4524) fails (subscript_mimic.s:16, MOVE_STATUS_FAILED) when the
    defender's ``movePrevByBattler`` cannot be Mimicked (``MIMIC_FORBIDDEN``,
    battle_lib.c:7152-7158, :7182-7192) or is MOVE_NONE, when the defender has a Substitute
    (:4483), or when the user already knows the move (:4490-4492). Otherwise slot ``j``, the
    user's first Mimic slot (:4494-4499; one exists: Metronome and Assist, :6430, never call
    Mimic, sCannotMetronomeMoves, battle_lib.c:7152-7158, and Sleep Talk calls a known move),
    takes the move with ``min(5, its table PP)`` (:4502-4513) and its bit joins
    ``mimickedMoveSlot`` (:4515): the party Pokemon never sees the copy, so the slot's move and
    its PP after Mimic's own use go to ``shadow_moves`` / ``shadow_pp`` (``step._restore_moves``
    gives them back). A Transformed user fails (:4482, G-69)."""
    fmt = format_of(state)
    side, user = h.side, h.user
    own = party(fmt, side)
    last = state.last_move[h.target_side].astype(_I32)
    moves = state.moves[own, user].astype(_I32)
    mimics = moves == MOVE_MIMIC
    j = jnp.argmax(mimics)
    cond = (
        (h.effect == _E.MIMIC)
        & h.landed
        & ~jnp.any(last == jnp.asarray(MIMIC_FORBIDDEN, _I32))
        & (state.transformed[side] == 0)  # :4482, VOLATILE_CONDITION_TRANSFORM
        & (last != 0)
        # BtlCmd_TryMimic's own test (:4483): the defender's VOLATILE_CONDITION_SUBSTITUTE only.
        # subscript_mimic.s has no CheckSubstitute, so not ``h.shielded``'s SUBSTITUTE_HIT.
        & (state.substitute_hp[h.target_side] == 0)
        & ~jnp.any(moves == last)
    )
    copy_pp = jnp.minimum(tables.move_pp[last].astype(_I32), MIMIC_PP_CAP)
    live_pp = state.pp[own, user, j]

    def put(arr, index, value):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    return state._replace(
        moves=put(state.moves, (own, user, j), last),
        pp=put(state.pp, (own, user, j), copy_pp),
        # :4514-4516 a copied Last Resort zeroes lastResortCount (E3-B2-11).
        last_resort_count=put(
            state.last_resort_count,
            side,
            jnp.where(last == MOVE_LAST_RESORT, 0, state.last_resort_count[side]),
        ),
        shadow_moves=put(state.shadow_moves, (side, j), moves[j]),
        shadow_pp=put(state.shadow_pp, (side, j), live_pp),
        mimic_slots=put(state.mimic_slots, side, state.mimic_slots[side].astype(_I32) | (1 << j)),
    )


def _sketch(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Sketch (effect_script_0095.s:5, a direct ON_HIT ``subscript_sketch.s``):
    ``CheckSubstitute BTLSCR_DEFENDER`` (:5) and ``BtlCmd_TrySketch`` (battle_script.c:4839-4884)
    fail the move (subscript_sketch.s:15, MOVE_STATUS_FAILED) on a defender with a Substitute, a
    Transformed user (:4847), a defender's ``moveSketched`` (``state.sketch_move``) that is
    MOVE_NONE, Struggle, Sketch or Chatter (``SKETCH_FORBIDDEN``, :4848-4851) or that the user
    already knows in a slot that is not Sketch (:4855-4860; the forbidden Sketch itself makes
    that test the same as ``moves == last``). Otherwise the first Sketch slot (:4862-4865,
    :4868) takes the move with its table PP (:4869-4870, ``MOVE_DATA().pp``: no cap) and a copied
    Last Resort zeroes ``lastResortCount`` (:4876-4878). The slot's ``ppUps`` is not written
    (TrySketch sets ``moves`` and ``ppCur`` only; ``BtlIOCmd_UpdatePartyMon`` copies move and PP,
    battle_io_command.c:581-582), so its max PP (``hooks.max_pp``, MoveTable_CalcMaxPP,
    move_table.c:25-34) is the copy's table PP with Sketch's slot's PP Ups: none in play. PP Up
    needs a max PP of 5 or more (item_use_pokemon.c:20, :187) and Sketch has 1, and every way
    Sketch enters a slot zeroes its ``ppUps`` first (an in-battle level-up, battle_script.c:10170;
    evolution.c:632; move_reminder.c:934; party_menu/callbacks.c:1111): ``Pokemon_SetMoveSlot``
    alone keeps them (pokemon.c:3920-3933). The engine's
    ``moves`` / ``pp`` are the party Pokemon's, and ``BtlIOCmd_UpdatePartyMon`` writes every
    unflagged slot (battle_io_command.c:575-587), so the write is permanent: no shadow and no
    ``mimic_slots`` bit (Mimic cannot copy Sketch, ``Move_CanBeMimicked``). Only Sleep Talk
    reaches Sketch through another move (a known move, battle_script.c:4916-4925), so a Sketch
    slot exists wherever this runs."""
    fmt = format_of(state)
    side, user = h.side, h.user
    own = party(fmt, side)
    # BtlCmd_TrySketch's own tests (``order.sketch_copy``, shared with ``order.sketched_state``).
    copies, j, last = sketch_copy(state, side, h.target_side)
    cond = (
        (h.effect == _E.SKETCH)
        & h.landed
        & ~h.shielded  # subscript_sketch.s:5, BtlCmd_CheckSubstitute (battle_script.c:9180-9181)
        & copies
    )

    def put(arr, index, value):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    return state._replace(
        moves=put(state.moves, (own, user, j), last),
        pp=put(state.pp, (own, user, j), tables.move_pp[last].astype(_I32)),
        last_resort_count=put(
            state.last_resort_count,
            side,
            jnp.where(last == MOVE_LAST_RESORT, 0, state.last_resort_count[side]),
        ),
    )


def _transform(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Transform (effect_script_0057.s:3, a direct ``CHECK_HP | TO_DEFENDER``
    ``subscript_transform_into_target.s``, no CAN_PROTECT): it fails (:23-24, MOVE_STATUS_FAILED)
    on a semi-invulnerable or already Transformed defender (:5-6; a Substitute is not tested).
    ``BtlCmd_Transform`` (battle_script.c:5330-5373) then copies the defender's BattleMon up to and
    including ``ability`` (battle_mon.h:57-84): species, Attack, Defense, Speed, Sp. Atk, Sp. Def
    (the BattleMon values: ``damage.battle_stats``, a Power-Tricked defender's swapped ones; the
    user's own swap is then off, ``power_trick_desync``), the four moves, the IVs, all eight stat
    stages, both types, the form and the ability; each move's PP is ``min(5, table PP)``
    (:5364-5370). Not copied: HP, max HP, level, status, the volatile state, held item, gender
    (:5337-5338 write ``transformedPID`` / ``transformedGender``, cosmetic). Disable ends
    (:5335-5336), ``mimickedMoveSlot`` clears (:5339: a mimicked slot is not restored by Transform,
    its party value stays in the shadow), ``truant = totalTurns & 1`` (:5359). The party Pokemon's
    own values go to the ``shadow_*`` (once: a Transformed user transforming again keeps them).
    The switch-in checks run again after the move (``step._use_move``, :5350-5362)."""
    fmt = format_of(state)
    side, user, foe, target = h.side, h.user, h.target_side, h.target
    pf = party(fmt, foe)
    cond = (
        (h.effect == _E.TRANSFORM)
        & h.landed
        & (state.hp[pf, target] > 0)
        & (state.semi_invulnerable[foe] == 0)
        & (state.transformed[foe] == 0)
    )
    fresh = state.transformed[side] == 0
    masked = ((state.mimic_slots[side].astype(_I32) >> jnp.arange(MOVES)) & 1) == 1
    keep = masked | ~fresh  # party values saved already: Mimic's, or a first Transform's
    # The AI's record of the user's own ability settles to its id now: the copy changes what a
    # "known, unnamed" record (-1) would read (``decomp_ability_id``), G-69.
    record = state.ai_ability_known[user].astype(_I32)
    # Raw: the record is BattleAI_SetAbility's BattleMon.ability (battle_script.c:11805-11806).
    own_id = decomp_abilities(tables, state)[party(fmt, side), user]
    settled = jnp.where(cond & (party(fmt, side) == 0) & (record < 0), own_id, record)

    def put(arr, index, value):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    foe_moves = state.moves[pf, target]
    copy_pp = jnp.minimum(tables.move_pp[foe_moves].astype(_I32), MIMIC_PP_CAP)
    own = (party(fmt, side), user)
    return state._replace(
        shadow_moves=put(
            state.shadow_moves, side, jnp.where(keep, state.shadow_moves[side], state.moves[own])
        ),
        shadow_pp=put(state.shadow_pp, side, jnp.where(keep, state.shadow_pp[side], state.pp[own])),
        shadow_species=put(
            state.shadow_species,
            side,
            jnp.where(fresh, state.species[own], state.shadow_species[side]),
        ),
        shadow_form=put(
            state.shadow_form, side, jnp.where(fresh, state.form[own], state.shadow_form[side])
        ),
        # formNum is in the copied bytes (battle_mon.h:79, :5346-5348), H-5.
        form_num=put(state.form_num, side, state.form_num[foe]),
        shadow_stats=put(
            state.shadow_stats, side, jnp.where(fresh, state.stats[own], state.shadow_stats[side])
        ),
        shadow_ivs=put(
            state.shadow_ivs, side, jnp.where(fresh, state.ivs[own], state.shadow_ivs[side])
        ),
        shadow_ability_slot=put(
            state.shadow_ability_slot,
            side,
            jnp.where(fresh, state.ability_slot[own], state.shadow_ability_slot[side]),
        ),
        shadow_ability_form=put(
            state.shadow_ability_form,
            side,
            jnp.where(fresh, state.ability_form[own], state.shadow_ability_form[side]),
        ),
        species=put(state.species, own, state.species[pf, target]),
        form=put(state.form, own, state.form[pf, target]),
        stats=put(state.stats, own, battle_stats(state, foe, target)),
        ivs=put(state.ivs, own, state.ivs[pf, target]),
        # A copied Multitype Arceus gives the user its own plate's types, not the target's
        # (Battler_MonType reads the user's raw heldItem, battle_lib.c:7629-7699), F-3.
        types=put(
            state.types,
            own,
            multitype_plate(
                tables,
                state.species[pf, target],
                state.ability[pf, target],
                state.item[own],
                state.types[pf, target],
            ),
        ),
        # Raw: BtlCmd_Transform copies BattleMon.ability (battle_script.c:5342-5348).
        ability=put(state.ability, own, state.ability[pf, target]),
        ability_slot=put(state.ability_slot, own, state.ability_slot[pf, target]),
        ability_form=put(state.ability_form, own, state.ability_form[pf, target]),
        moves=put(state.moves, own, foe_moves),
        pp=put(state.pp, own, copy_pp),
        stages=put(state.stages, side, state.stages[foe]),
        disabled_move=put(state.disabled_move, side, 0),
        disable_turns=put(state.disable_turns, side, 0),
        last_resort_count=put(state.last_resort_count, side, 0),  # :5340 (E3-B2-11)
        mimic_slots=put(state.mimic_slots, side, (1 << MOVES) - 1),
        transformed=put(state.transformed, side, 1),
        power_trick_desync=put(state.power_trick_desync, side, state.power_trick[side]),
        truant=put(state.truant, side, state.turn.astype(_I32) & 1),
        # :5350-5362 clear the announce flags: the switch-in checks run again for the copy.
        trace_announced=put(state.trace_announced, side, 0),
        weather_announced=put(state.weather_announced, side, 0),
        intimidate_announced=put(state.intimidate_announced, side, 0),
        download_announced=put(state.download_announced, side, 0),
        ai_ability_known=state.ai_ability_known.at[user].set(
            settled.astype(state.ai_ability_known.dtype)
        ),
    )


def _chatter(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Chatter (effect_script_0267.s:3-5): a plain hit whose ON_HIT indirect ``subscript_chatter.s``
    runs on a hit that did not fail (MOVE_STATUS_NO_EFFECTS clear, battle_lib.c:1524-1529):
    ``CheckChatterActivation`` (battle_script.c:8397-8436) skips unless the attacker's species is
    Chatot and it is not Transformed, and the defender has HP (:8403-8405); then ``RandNext() %
    100 <= effectChance`` (:8428-8430, ``Outcome.chatter``: 0 / 10 / 30 of the Chatot cry, 0 for
    every enemy) confuses by ``subscript_confuse.s``: the effect is ``SIDE_EFFECT_TYPE_INDIRECT``
    (effect_script_0267.s:5, battle_lib.c:1520-1522), so the defender's Shield Dust as the
    attacker sees it stops it (subscript_confuse.s:14-15, ``CheckIgnorableAbility``: a Mold
    Breaker attacker ignores it), after the roll is drawn; then not confused :24, no Substitute
    :23, no Safeguard :26 (G-70). Before them the defender's ignorable Own Tempo (:12 -> :_141)
    ends it silently (H-2)."""
    fmt = format_of(state)
    chatot = (state.species[party(fmt, h.side), h.user] == SPECIES_CHATOT) & (
        state.transformed[h.side] == 0
    )
    cond = (
        (h.effect == _E.CHATTER)
        & h.effective
        & chatot
        & _alive(state, h.target_side, h.target)
        & (outcome.chatter[h.side] != 0)
        & (h.target_ignorable != AbilityId.OWN_TEMPO)
        & (state.confusion[h.target_side] == 0)
        & ~_shield_dust(h)
        & ~h.shielded
        & (state.safeguard[side_of(h.target_side)] == 0)
    )
    return _set(state, "confusion", h.target_side, outcome.duration[h.side], cond)


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
    whatever their HP; the user's partner is on its own side and does not count)."""
    fmt = format_of(state)
    own = state.moves[party(fmt, side), state.active[side]].astype(_I32)
    shared = jnp.bool_(False)
    for battler in range(state.active.shape[0]):
        theirs = state.moves[party(fmt, battler), state.active[battler]].astype(_I32)
        match = (own[:, None] == theirs[None, :]) & (own[:, None] != 0) & (theirs[None, :] != 0)
        other_side = side_of(jnp.asarray(side)) != side_of(battler)
        shared = shared | (other_side & jnp.any(match))
    return (state.imprison[side] != 0) | ~shared


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
    "magnet_rise",  # G-66
    "minimized",
    "semi_invulnerable",  # B10-12
    "power_trick",  # B12-10
    "ability_suppressed",  # MOVE_EFFECT_ABILITY_SUPPRESSED, E3-2-6
    "charged_turns",  # MOVE_EFFECT_CHARGE, E3-B2-8
    "grudge",  # MOVE_EFFECT_GRUDGE, E3-B2-9
    "miracle_eye",  # MOVE_EFFECT_MIRACLE_EYE, E3-B2-10
    "camouflage",  # MOVE_EFFECT_CAMOUFLAGE (G-20)
)


def ingrain_anchors(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: subscript_force_target_to_switch_or_flee.s:7 finds the defender's
    ``moveEffectsMask`` equal (``OPCODE_EQU``) to ``MOVE_EFFECT_INGRAIN``: ``side``'s battler is
    ingrained and has no other bit of the mask: none of ``MOVE_EFFECT_MASK_FIELDS`` (the Leech
    Seed bits sit on the seeded battler, subscript_leech_seed_start.s:13-14). The other bits come
    from unsupported moves (Heal Block); each joins the tuple with its move."""
    other = jnp.bool_(False)
    for name in MOVE_EFFECT_MASK_FIELDS:
        other = other | (getattr(state, name)[side] != 0)
    return (state.ingrain[side] != 0) & ~other


LOCK_ON_TURNS = 2  # MOVE_EFFECT_LOCK_ON_1 in the 2-bit counter (moves.h:12, subscript_lock_on.s:5)


def lock_on_fails(state: BattleState, target_side) -> jnp.ndarray:
    """``bool``: subscript_lock_on.s:5-6 fails (MOVE_STATUS_FAILED, :33) a defender behind a
    substitute (``CheckSubstitute``); its miss test (:5) never holds for accuracy 0."""
    return state.substitute_hp[target_side] != 0


def _lock_on(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Lock-On, Mind Reader (effect_script_0094.s: a direct ``CHECK_HP | TO_DEFENDER``
    subscript_lock_on.s for a living defender on a move that connected, battle_lib.c:1491-1504):
    the defender's MOVE_EFFECT_LOCK_ON counter is set to ``LOCK_ON_TURNS`` (:7-8, whatever it was)
    and its lockOnTarget to the attacker (:9, ``OPCODE_SET``), as ``1 + battler``."""
    cond = (
        (h.effect == _E.LOCK_ON)
        & h.landed
        & _alive(state, h.target_side, h.target)
        & ~lock_on_fails(state, h.target_side)
    )
    state = _set(state, "lock_on", h.target_side, LOCK_ON_TURNS, cond)
    return _set(state, "lock_on_target", h.target_side, 1 + jnp.asarray(h.side, _I32), cond)


PERISH_SONG_TURNS = 3  # BtlCmd_TryPerishSong (battle_script.c:5478)


def perish_song_eligible(state: BattleState, user_side) -> jnp.ndarray:
    """``bool[B]``: the battlers ``BtlCmd_TryPerishSong`` marks (battle_script.c:5471-5479) for
    the Perish Song of battler ``user_side``: no MOVE_EFFECT_PERISH_SONG yet, HP left and no
    ``Battler_IgnorableAbility(attacker, i, ABILITY_SOUNDPROOF)`` (:5474, H-3): a Soundproof user
    skips itself, the user's Mold Breaker pierces another's. Every battler, whatever its side
    (doubles convention 4). The "blocks" message tags BTLSCR_MSG_TEMP
    (subscript_perish_song_start.s:12-19): no reveal (battle_script.c:11801-11802)."""
    fmt = format_of(state)
    battlers = range(state.active.shape[0])
    sides = jnp.arange(state.active.shape[0])
    deaf = jnp.stack([ignorable(state, user_side, b) == AbilityId.SOUNDPROOF for b in battlers])
    return (state.perish_song == 0) & (state.hp[party(fmt, sides), state.active] > 0) & ~deaf


def _perish_song(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Perish Song (effect_script_0114.s: a direct ON_HIT subscript_perish_song_start.s for a
    move that connected; RANGE_FIELD, its own defender): ``BtlCmd_TryPerishSong`` gives every
    eligible battler (``perish_song_eligible``) the effect and a count of ``PERISH_SONG_TURNS``;
    with none eligible it fails (:5482-5484, MOVE_STATUS_FAILED: ``primary_effect_fails``)."""
    eligible = perish_song_eligible(state, h.side)
    for battler in range(state.active.shape[0]):
        cond = (h.effect == _E.PERISH_SONG) & h.landed & eligible[battler]
        state = _set(state, "perish_song", battler, 1, cond)
        state = _set(state, "perish_song_turns", battler, PERISH_SONG_TURNS, cond)
    return state


YAWN_TURNS = 2  # BtlCmd_TryYawn: 1 << (MOVE_EFFECT_YAWN_SHIFT + 1) (battle_script.c:6587)


def _sunny(tables, state: BattleState) -> jnp.ndarray:
    """``CheckIgnoreWeather`` then ``FIELD_CONDITION_SUNNY`` (battle_script.c:9195-9206): any
    SUNNY bit of the mask (``field.sun``)."""
    return fcm.sun(fcm.weather_bits(state)) & ~weather_suppressed(tables, state)


def yawn_blocks(tables, state: BattleState, side, target_side) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(fails, by_ability)`` of subscript_yawn.s on ``target_side``'s battler: an ignorable
    Insomnia or Vital Spirit, or Leaf Guard in sun (:5-9, "made it ineffective", tagging the
    ability), a substitute (:14), an uproar unless its own Soundproof (:15-16), any status (:18),
    Safeguard on its side (:19), a Yawn already counting (``TryYawn``, :21; battle_script.c:6584).
    Every one ends the move without AfterMoveEffects (MOVE_STATUS_FAILED or _NO_MORE_WORK)."""
    fmt = format_of(state)
    mon = state.active[target_side]
    own = battler_decomp_ability(tables, state, target_side, mon)
    mold = battler_ability(state, side) == AbilityId.MOLD_BREAKER
    seen = jnp.where(mold, 0, own)  # Battler_IgnorableAbility
    ids = YAWN_ABILITY_IDS
    by_ability = (seen == ids["Insomnia"]) | (seen == ids["Vital Spirit"])
    by_ability = by_ability | (_sunny(tables, state) & (seen == ids["Leaf Guard"]))
    fails = (
        by_ability
        | (state.substitute_hp[target_side] != 0)
        | ((own != ids["Soundproof"]) & fcm.uproar_any(state))
        | (state.status[party(fmt, target_side), mon] != Status.NONE)
        | (state.safeguard[side_of(target_side)] != 0)
        | (state.yawn[target_side] != 0)
    )
    return fails, by_ability


def know_ability(state: BattleState, side, slot, cond) -> BattleState:
    """A message tags the ability of ``side``'s party ``slot`` (``BattleAI_SetAbility``,
    battle_script.c:11806): the AI knows it, for a party-0 battler (ruling D3-1a)."""
    on = jnp.asarray(cond) & (party(format_of(state), jnp.asarray(side, _I32)) == 0)
    # Raw: BattleMessage_AbilityTag reads BattleMon.ability (battle_script.c:11805).
    record = decomp_ability_id(
        state.ability[party(format_of(state), jnp.asarray(side, _I32)), slot]
    )
    known = jnp.where(on, record, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
    return state._replace(ai_ability_known=state.ai_ability_known.at[slot].set(known))


def _yawn(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Yawn (effect_script_0187.s: a direct ``CHECK_HP | TO_DEFENDER`` subscript_yawn.s for a
    living defender on a move that connected): unless ``yawn_blocks``, the defender's
    MOVE_EFFECT_YAWN counter becomes ``YAWN_TURNS``; an ability that blocks it is tagged. On a
    semi-invulnerable miss the subscript still runs and tags it (``step._use_move``, B10-13)."""
    run = (h.effect == _E.YAWN) & h.landed & _alive(state, h.target_side, h.target)
    fails, by_ability = yawn_blocks(tables, state, h.side, h.target_side)
    state = know_ability(state, h.target_side, h.target, run & by_ability)
    return _set(state, "yawn", h.target_side, YAWN_TURNS, run & ~fails)


def yawn_sleep(tables, state: BattleState, side, outcome: Outcome) -> BattleState:
    """``MON_COND_CHECK_STATE_YAWN`` (bcp.c:1684-1697) on ``side``'s battler: the counter drops,
    and at 0 subscript_fall_asleep.s runs as SIDE_EFFECT_TYPE_MOVE_EFFECT (:4-14): its own
    (``CheckAbility``, not ignorable) Insomnia or Vital Spirit, or Leaf Guard in sun, keeps it
    awake ("stayed awake because of its ...", tagging the ability, :_202/_221); a sleeper, an
    uproar (unless its Soundproof) or any status stops it silently; else it sleeps
    ``Outcome.yawn_sleep[side]`` turns (``Random 3, 2``, :_147) and is unlocked (:_176). No
    Safeguard test on this path."""
    fmt = format_of(state)
    mon = state.active[side]
    on = state.yawn[side] != 0
    left = state.yawn[side].astype(_I32) - 1
    state = _set(state, "yawn", side, left, on)
    falls = on & (left == 0)
    own = battler_decomp_ability(tables, state, side, mon)
    ids = YAWN_ABILITY_IDS
    awake = (own == ids["Insomnia"]) | (own == ids["Vital Spirit"])
    awake = awake | (_sunny(tables, state) & (own == ids["Leaf Guard"]))
    state = know_ability(state, side, mon, falls & awake)
    sleeps = (
        falls
        & ~awake
        & ~((own != ids["Soundproof"]) & fcm.uproar_any(state))
        & (state.status[party(fmt, side), mon] == Status.NONE)
    )
    state = _set(state, "status", (side, mon), Status.SLEEP, sleeps)
    state = _set(state, "status_counter", (side, mon), outcome.yawn_sleep[side], sleeps)
    return unlock_move(state, side, sleeps)


def locked_onto(state: BattleState, attacker, defender) -> jnp.ndarray:
    """``bool``: ``MON_IS_LOCKED_ONTO(attacker, defender)`` (include/battle/common.h:51): the
    defender's MOVE_EFFECT_LOCK_ON counter is up and its lockOnTarget is the attacker."""
    return (state.lock_on[defender] != 0) & (state.lock_on_target[defender] == 1 + attacker)


def _mud_sport(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    cond = (h.effect == _E.MUD_SPORT) & h.landed & (state.mud_sport[h.side] == 0)
    return _set(state, "mud_sport", h.side, 1, cond)


SCREEN_TURNS = 5  # NUM_SCREEN_TURNS (include/constants/battle.h:142)


def _screen(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Light Screen, Reflect: ``BtlCmd_TryLightScreen`` / ``TryReflect`` (script.c:4214,
    :4257) fail when the user's side has it, else set it for ``NUM_SCREEN_TURNS``, plus the
    hold param of the user's Light Clay (``HOLD_EFFECT_EXTEND_SCREENS``, script.c:4229, :4272:
    ``Battler_HeldItemEffect`` / ``Battler_HeldItemPower`` of the attacker)."""
    item = held_item(state, h.side, h.user)
    clay = tables.item_effect[item] == ItemId.LIGHT_CLAY
    turns = SCREEN_TURNS + jnp.where(clay, tables.item_param[item, 0].astype(_I32), 0)
    for effect, field in ((_E.LIGHT_SCREEN, "light_screen"), (_E.REFLECT, "reflect")):
        cond = (h.effect == effect) & h.landed & (getattr(state, field)[side_of(h.side)] == 0)
        state = _set(state, field, h.side, turns, cond)
    return state


def _brick_break(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Brick Break: a direct ``BREAK_SCREENS`` effect (lib.c:1481-1510) that still applies
    on an immune target (``MOVE_STATUS_DID_NOT_AFFECT``) but not on a miss;
    ``BtlCmd_TryBreakScreens`` (script.c:6550) clears both screens of the defender's side."""
    cond = (h.effect == _E.BRICK_BREAK) & h.landed
    state = _set(state, "reflect", h.target_side, 0, cond)
    return _set(state, "light_screen", h.target_side, 0, cond)


def defog_stage_fails(state: BattleState, target_side, seen, stat) -> jnp.ndarray:
    """``bool``: Defog's evasion drop ends in ``result == 2`` (``MOVE_STATUS_FAILED``,
    subscript_update_stat_stage.s:60) at ``BtlCmd_ChangeStatStage`` (battle_script.c:2874-2923):
    the target's side has no Mist (:2877), the target's ability as the attacker sees it
    (``seen``) is not Clear Body or White Smoke (:2883-2884), its ``stat`` stage is above
    -6 (:2904) and its Substitute is up (:2921-2923); a drop that meets one of the earlier
    tests is ``result == 1`` (no failure; the else-if chain never reaches the substitute)."""
    blocked = mist_blocks_drop(state, target_side, -1)
    blocked = blocked | (seen == AbilityId.CLEAR_BODY)
    blocked = blocked | (seen == AbilityId.WHITE_SMOKE)
    blocked = blocked | (state.stages[target_side, stat] <= STAGE_MIN)
    return ~blocked & (state.substitute_hp[target_side] != 0)


def _defog(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Defog (effect_script_0258.s: a direct ON_HIT effect, ``BattleSystem_TriggerPrimaryEffect``
    battle_lib.c:1481-1491, no HP test; subscript_defog.s): the defender's evasion -1 through
    ``_foe_stage`` (:19-20: Mist, Clear Body / White Smoke, -6, Substitute; the subscript goes on
    whatever it does), then on the defender's side Reflect, Light Screen, Mist (after the drop met
    it), Safeguard, Spikes, Toxic Spikes and Stealth Rock cleared (:21-61) and the field's deep fog
    (:64-65: ``FIELD_CONDITION_DEEP_FOG`` off, ``field.defog_clear``; the other mask bits and
    ``weather_turns`` stay). The user's own side is never touched. E3-B2-14."""
    cond = (h.effect == _E.DEFOG) & h.landed
    state = _foe_stage(state, h, cond)
    for field in ("reflect", "light_screen", "mist", "safeguard", "spikes", "toxic_spikes"):
        state = _set(state, field, h.target_side, 0, cond)
    state = _set(state, "stealth_rock", h.target_side, 0, cond)
    return fcm.defog_clear(state, cond)


def _weather(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Sandstorm, Rain Dance, Sunny Day, Hail (``WEATHER``; effect_script_0115.s, _0136.s,
    _0137.s, _0164.s, run before ``TryMove``, which cannot stop this accuracy-0
    ``RANGE_FIELD`` move): ``MOVE_STATUS_FAILED`` while a bit of its weather's group is set
    (``move_param[m, 1] == 1``) or, for Hail, its TEMP bit (``weather_fails``); else
    ``FIELD_CONDITION_WEATHER`` is cleared and the temporary weather set with
    ``WEATHER_TURNS`` turns plus the param of the user's matching weather rock
    (``CheckItemHoldEffect`` / ``GetItemEffectParam`` on the attacker). The failure skips
    ``AfterMoveEffects`` (``step._use_move``)."""
    weather = h.param[0]
    item = held_item(state, h.side, h.user)  # Battler_HeldItemEffect (B13-29)
    rock = (tables.item_effect[item] == ItemId.WEATHER_ROCK) & (
        tables.item_param[item, 1].astype(_I32) == weather
    )
    turns = WEATHER_TURNS + jnp.where(rock, tables.item_param[item, 0].astype(_I32), 0)
    cond = (h.effect == _E.WEATHER) & h.landed & ~weather_fails(state, h.param)
    return fcm.weather_move(state, weather, turns, cond)


def weather_fails(state: BattleState, param) -> jnp.ndarray:
    """``bool``: a ``WEATHER`` move with ``param`` (``move_param[m]``) fails on this field
    (``_weather``): Sandstorm, Rain Dance, Sunny Day test their whole group (``param[1] == 1``:
    effect_script_0115.s, _0136.s, _0137.s :5), Hail only HAILING_TEMP (effect_script_0164.s:5),
    so Hail succeeds under a HAILING_PERM-only mask."""
    return fcm.weather_move_fails(fcm.weather_bits(state), param[0], param[1] == 1)


def armored(state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: ``target_side``'s active battler has Battle Armor or Shell Armor as
    ``side``'s sees it (``Battler_IgnorableAbility``): ``BattleSystem_CalcCriticalMulti``
    (battle_lib.c:7135-7136) never crits it, whatever the roll (``Outcome.crit``)."""
    seen = ignorable(state, side, target_side)
    return (seen == AbilityId.SHELL_ARMOR) | (seen == AbilityId.BATTLE_ARMOR)


def crit_blocked(state: BattleState, side, target_side) -> jnp.ndarray:
    """``bool``: ``BattleSystem_CalcCriticalMulti`` (battle_lib.c:7134-7137) gives no crit to
    ``side``'s hit on ``target_side``, whatever the roll: the defender's Battle Armor / Shell
    Armor (``armored``) or Lucky Chant on the defender's side (``sideConditions`` is
    ``BATTLECTX_SIDE_CONDITIONS_MASK`` of ``battleCtx->defender``, battle_script.c:2158-2163).
    The defender is the hit's own: the user for a lock reading its stored target (singles
    follow-up (ii)), so the turn-start cut (``Outcome.crit``, the foe's side) cannot stand in.
    The draw is spent either way (:7132 runs first). A Lucky Chant used earlier this turn
    counts, as the conditions read now (E3-B2-7)."""
    return armored(state, side, target_side) | (state.lucky_chant[side_of(target_side)] != 0)


def _eat_berry(
    tables, state, h: _Hit, outcome: Outcome, foe_custap
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, pluck_flag)`` (``hooks.pluck_berry``)."""
    # Local import: hooks imports this module.
    from nuzline.engine.hooks import pluck_berry

    # BattleSystem_PluckBerry returns early when the defender's substitute was hit
    # (battle_lib.c:5546).
    cond = (h.effect == _E.EAT_BERRY) & h.effective & ~h.sub_hit
    return pluck_berry(tables, state, h.side, h.target_side, cond, foe_custap)


def _knock_off(tables, state, h: _Hit, foe_custap, foe_claw) -> BattleState:
    """Knock Off (effect_script_0188.s: an indirect ON_HIT subscript_knock_off.s on a hit that
    was not MOVE_STATUS_NO_EFFECTS, battle_lib.c:1524-1529), B10-6. The subscript ends with
    nothing done for a defender behind a substitute (:5, ``h.shielded``), with Multitype
    (:6, its BATTLEMON_ABILITY), holding the Griseous Orb (:7) or with its quickClaw (:8,
    ``foe_claw``) or custapBerry (:9, ``foe_custap``) flag set (D3-6 Fact C: the flags stand until
    the holder's own move starts or a Pokemon enters its slot; the caller passes them as carried,
    with no ``moved`` term); then BtlCmd_TryKnockOff
    (battle_script.c:6608-6632): no item does nothing; an ignorable Sticky Hold keeps it (its
    C-built "made ineffective" message tags nothing for the AI); else heldItem = ITEM_NONE and
    the slot's knockedOffItemsMask bit (:6626-6627). Neither RemoveItem nor CopyToParty runs: the
    AI's record of the item (``ai_item_known``) stays until the battler's next CopyToParty
    (ruling B10-6), and canUnburden stays as it was (an armed Unburden holder speeds up)."""
    fmt = format_of(state)
    item = state.item[party(fmt, h.target_side), h.target]
    # Raw: subscript_knock_off.s:6 (BATTLEMON_ABILITY Multitype).
    raw = decomp_abilities(tables, state)[party(fmt, h.target_side), h.target]
    # BtlCmd_TryKnockOff's Sticky Hold: Battler_IgnorableAbility (battle_script.c:6608-6632).
    mold_breaker = battler_ability(state, h.side, h.user) == AbilityId.MOLD_BREAKER
    sticky = (
        battler_decomp_ability(tables, state, h.target_side, h.target)
        == ITEM_ABILITY_IDS["Sticky Hold"]
    ) & ~mold_breaker
    cond = (
        (h.effect == _E.KNOCK_OFF)
        & h.effective
        & ~h.shielded
        & (raw != ABILITY_MULTITYPE)
        & (item != ITEM_GRISEOUS_ORB)
        & ~jnp.asarray(foe_claw, bool)
        & ~jnp.asarray(foe_custap, bool)
        & (item != 0)
        & ~sticky
    )
    # The bit is the defender SIDE's (``sideConditions[defending]``, :6627): every party of the
    # side sends slot ``h.target`` under it, so each one's party item freezes where the bit is new
    # (UpdatePartyMon skips the item from now on, battle_io_command.c:590-592; D3 fix B): its item
    # now, which for the defender is the item Knock Off takes. A party whose slot ``h.target`` is
    # active elsewhere keeps its battle item until it is loaded again (InitBattleMon,
    # battle_lib.c:161, ``step._switch_in``). B = 2: one party per side, the defender's.
    fmt_sides = party_sides(fmt)
    fresh = cond & ~knocked_off_at(state, h.target_side, h.target)
    defending = side_of(jnp.asarray(h.target_side, _I32))
    for p in range(fmt.P):
        mine = fresh & (defending == fmt_sides[p])
        kept = jnp.where(mine, state.item[p, h.target], state.knocked_item[p, h.target])
        state = state._replace(
            knocked_item=state.knocked_item.at[p, h.target].set(
                kept.astype(state.knocked_item.dtype)
            )
        )
    state = _set(state, "item", (h.target_side, h.target), 0, cond)
    return _set(state, "knocked_off", (h.target_side, h.target), 1, cond)


EMBARGO_TURNS = 5  # subscript_embargo_start.s:14


def embargo_fails(tables, state: BattleState, target_side, foe_claw=False, foe_custap=False):
    """``bool``: subscript_embargo_start.s fails (MOVE_STATUS_FAILED, :57-58) on a defender that
    landed (its miss / semi-invulnerable test, :5, is ``h.effective``'s): behind a substitute
    (:6, CheckSubstitute; the script's own test, effect_script_0232.s:5, is the same), with
    Multitype (:7, its BATTLEMON_ABILITY), already under Embargo (:8, embargoTurns), holding the
    Griseous Orb (:9, its BATTLEMON_HELD_ITEM: the raw item) or with its quickClaw (:10,
    ``foe_claw``) or custapBerry (:11, ``foe_custap``) flag set (B13-31; D3-6 Fact C: the carried
    flags, no ``moved`` term)."""
    fmt = format_of(state)
    target_side = jnp.asarray(target_side, _I32)
    target = state.active[target_side]
    raw = decomp_abilities(tables, state)[party(fmt, target_side), target]  # :7, raw
    return (
        (state.substitute_hp[target_side] != 0)
        | (raw == ABILITY_MULTITYPE)
        | (state.embargo[target_side] != 0)
        | (state.item[party(fmt, target_side), target] == ITEM_GRISEOUS_ORB)
        | jnp.asarray(foe_claw, bool)
        | jnp.asarray(foe_custap, bool)
    )


def _embargo(tables, state, h: _Hit, foe_custap, foe_claw) -> BattleState:
    """Embargo (effect_script_0232.s: a direct ON_HIT TO_DEFENDER subscript_embargo_start.s, run
    for a move without MOVE_STATUS_NO_EFFECTS, battle_lib.c:1484-1489): unless it fails
    (``embargo_fails``), the defender's embargoTurns = 5 with its MOVE_EFFECT_EMBARGO bit
    (:13-14; ``BattleState.embargo``). The message tags the defender's nickname only. A failure
    skips AfterMoveEffects (``primary_effect_fails``), B13-31."""
    cond = (
        (h.effect == _E.EMBARGO)
        & h.effective
        & ~h.shielded
        & ~embargo_fails(tables, state, h.target_side, foe_claw, foe_custap)
    )
    return _set(state, "embargo", h.target_side, EMBARGO_TURNS, cond)


def _leech_seed(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    fmt = format_of(state)
    grass = jnp.any(state.types[party(fmt, h.target_side), h.target] == Type.GRASS)
    cond = (
        (h.effect == _E.LEECH_SEED)
        & h.effective
        & _alive(state, h.target_side, h.target)
        & ~h.shielded  # subscript_leech_seed_start.s:5, before the Grass test
        & ~grass
        & (state.leech_seed[h.target_side] == 0)
    )
    # subscript_leech_seed_start.s:14-15: the recipient is the battler that used the move.
    state = _set(state, "leech_seed_recipient", h.target_side, 1 + h.side, cond)
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


def _miracle_eye(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Miracle Eye (``effect_script_0216.s``: an ``ON_HIT`` indirect effect,
    ``subscript_miracle_eye.s:5``, run unless ``MOVE_STATUS_NO_EFFECTS``, battle_lib.c:1524-1529):
    the defender holds ``MOVE_EFFECT_MIRACLE_EYE`` whenever the move lands on a living target; there
    is no failure check (a set bit stays set) and no Substitute test. The script's :6
    ``SYSCTL_IGNORE_IMMUNITIES`` is inert (``IGNORES_TYPE_CHECKS``; power 0 skips ``CheckTypeChart``,
    bcp.c:2352-2357). The target is ``h.target_side``, so a doubles partner's hit reads it too
    (E3-B2-10)."""
    cond = (h.effect == _E.MIRACLE_EYE) & h.landed
    return _set(state, "miracle_eye", h.target_side, 1, cond)


def _attractable(state: BattleState, side, slot, foe_side, foe_slot) -> jnp.ndarray:
    """``BtlCmd_TryAttract`` (battle_script.c:5727-5742) succeeds for target ``side`` /
    ``slot`` and source ``foe_side`` / ``foe_slot``: different genders, neither genderless
    (``GENDER_NONE`` is 2, ``BattleState.gender``), and the target not in love yet."""
    fmt = format_of(state)
    mine, theirs = (
        state.gender[party(fmt, side), slot],
        state.gender[party(fmt, foe_side), foe_slot],
    )
    return (mine != theirs) & (mine != 2) & (theirs != 2) & (state.infatuated[side] == 0)


def _attract(tables, state, h: _Hit, outcome: Outcome) -> BattleState:
    """Attract: the living target of the opposite gender falls in love with the user. The
    target's ignorable Oblivious stops it first (subscript_infatuate.s:5 -> :_055): "prevents
    romance" (:39, TAG_NICKNAME_ABILITY on BTLSCR_SIDE_EFFECT_MON: the AI learns it), H-2."""
    runs = (h.effect == _E.ATTRACT) & h.effective & _alive(state, h.target_side, h.target)
    oblivious = runs & (h.target_ignorable == AbilityId.OBLIVIOUS)
    from nuzline.engine.hooks import ROMANCE_BLOCKED, reveal_ability  # hooks imports this module

    state = reveal_ability(state, h.target_side, h.target, ROMANCE_BLOCKED, oblivious)
    cond = runs & ~oblivious & _attractable(state, h.target_side, h.target, h.side, h.user)
    return _set(state, "infatuated", h.target_side, 1, cond)


def _set_status(state: BattleState, side, slot, status, cond) -> BattleState:
    """``side``'s party ``slot`` gets ``status`` (counter 0) where ``cond``."""
    state = _set(state, "status", (side, slot), status, cond)
    return _set(state, "status_counter", (side, slot), 0, cond)


def _struggle_recoil(state: BattleState, h: _Hit, struggling) -> BattleState:
    """Struggle (``effect_script_0254.s``): an ``ON_HIT`` secondary effect,
    ``subscript_struggle.s``: ``BattleSystem_Divide(-maxHP, 4)``, at least 1, on a
    hit."""
    fmt = format_of(state)
    max_hp = state.max_hp[party(fmt, h.side), h.user].astype(_I32)
    cost = jnp.where(struggling & h.landed, jnp.maximum(max_hp // 4, 1), 0)
    return hp_update(state, h.side, h.user, -cost, cost > 0)


def _contact_ability(
    tables, state, h: _Hit, outcome: Outcome, move_type
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, sem_proc)``: ``BattleSystem_TriggerAbilityOnHit`` (battle_lib.c:4200-4371)
    after hit ``h``; ``move_type`` is the action's ``battleCtx->moveType`` or the table type
    (``move_type_of``), which Color Change reads. ``sem_proc``: a trigger that moved
    ``sideEffectMon`` to the attacker before its subscript ran, whether or not that lands: Cute
    Charm's (the attacker not in love, the roll; :4339-4350, H-4) and Static's, Poison Point's and
    Flame Body's (an attacker with no status, the roll; :4214-4224, :4303-4313, :4321-4331, H-M2),
    whose subscript may then stop on the attacker's Limber, Immunity, Water Veil, Leaf Guard in sun
    or type (subscript_paralyze.s:5-8, subscript_poison.s:13-16, :34-37, subscript_burn.s:20-23,
    :40-41); an ABILITY side effect skips the substitute (subscript_paralyze.s:20, poison :28, burn
    :35) and Safeguard (paralyze :26, poison :39, burn :43)."""
    fmt = format_of(state)
    # BattleSystem_TriggerAbilityOnHit (battle_lib.c:4200-4371): Battler_Ability.
    ability = battler_ability(state, h.target_side, h.target)
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
    # BattleSystem_TriggerAbilityOnHit returns early when the defender's substitute was hit
    # (battle_lib.c:4209).
    hit = hit & ~h.sub_hit
    # subscript_paralyze.s:5: the attacker's own Limber (the side-effect battler is the
    # attacker itself, so no Mold Breaker can pierce it) ends Static's paralysis silently
    # (:_170 ends an ABILITY side effect with no message).
    own = battler_ability(state, h.side, h.user)
    # Likewise its own Immunity stops Poison Point's poison (subscript_poison.s:13 -> :_177) and
    # its own Water Veil Flame Body's burn (subscript_burn.s:20 -> :_264), silently (H-2).
    own_block = (
        ((status == Status.PARALYSIS) & (own == AbilityId.LIMBER))
        | ((status == Status.POISON) & (own == AbilityId.IMMUNITY))
        | ((status == Status.BURN) & (own == AbilityId.WATER_VEIL))
    )
    # Then the attacker's own Leaf Guard in sun (paralyze :6-8, poison :14-16, burn :21-23; ruling
    # G-14): an ABILITY side effect ends silently (:_170, :_177, :_264).
    leaf = _sunny(tables, state) & (battler_ability(state, h.side, h.user) == AbilityId.LEAF_GUARD)
    # The trigger itself (ATTACKING_MON.status == MON_CONDITION_NONE, the roll): sideEffectMon =
    # the attacker (battle_lib.c:4224, :4313, :4331) before the subscript's blocks (H-M2).
    status_proc = (
        (status != Status.NONE)
        & hit
        & h.contact_proc
        & (state.status[party(fmt, h.side), h.user] == Status.NONE)
    )
    cond = status_proc & ~own_block & ~leaf & can_have_status(tables, state, h.side, h.user, status)
    # Rough Skin (battle_lib.c:4257-4271): no roll; subscript_rough_skin.s takes
    # BattleSystem_Divide(maxHP, 8) from the attacker (it may faint).
    # battle_lib.c:4259: not on a Magic Guard attacker (Battler_Ability).
    rough = (
        (ability == AbilityId.ROUGH_SKIN)
        & hit
        & (battler_ability(state, h.side, h.user) != AbilityId.MAGIC_GUARD)
    )
    # Aftermath (battle_lib.c:4358-4371): the defender is faintedMon (this hit's UPDATE_HP fainted
    # it: TryFaintMon, battle_script.c:1768-1770), a living attacker without Magic Guard, no Damp
    # on the field (an unsupported ability), no MOVE_STATUS_NO_EFFECTS, a contact move:
    # subscript_aftermath.s takes Divide(maxHP, 4) (B9-7).
    aftermath = (
        (ability == AbilityId.AFTERMATH)
        & hit
        & ~_alive(state, h.target_side, h.target)
        & (battler_ability(state, h.side, h.user) != AbilityId.MAGIC_GUARD)
    )
    max_hp = state.max_hp[party(fmt, h.side), h.user].astype(_I32)
    cut = jnp.where(rough, jnp.maximum(max_hp // 8, 1), 0)
    cut = jnp.where(aftermath, jnp.maximum(max_hp // 4, 1), cut)
    state = hp_update(state, h.side, h.user, -cut, cut > 0)
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
        # subscript_infatuate.s:5 -> :_082 -> :_117: the attacker's own Oblivious, silent (H-2).
        & (own != AbilityId.OBLIVIOUS)
        & _attractable(state, h.side, h.user, h.target_side, h.target)
    )
    state = _set(state, "infatuated", h.side, 1, charm)
    # The trigger itself (battle_lib.c:4339-4350): the same tests but the gender / Oblivious ones,
    # which are subscript_infatuate.s's TryAttract and CheckIgnorableAbility, after sideEffectMon
    # moved (H-4). SYSCTL_FIRST_OF_MULTI_TURN / _UTURN_ACTIVE hold no value in a hit that got here.
    charm_proc = (
        (ability == AbilityId.CUTE_CHARM)
        & hit
        & _alive(state, h.target_side, h.target)
        & h.contact_proc
        & (state.infatuated[h.side] == 0)
    ) | charm
    # Color Change (battle_lib.c:4232-4253, H-3): no contact or attacker test. The type is
    # battleCtx->moveType when set, else the table's (``move_type``; the attacker's Normalize,
    # :4235-4236, is an unsupported ability). The defender has HP, no MOVE_STATUS_NO_EFFECTS
    # (``h.effective``), not Struggle, physical / specialDamageTaken from this hit's UpdateHP
    # (bcp.c:3413-3424: ``h.dealt`` on the HP path; a substitute hit returned at :4209), no
    # SYSCTL_UTURN_ACTIVE (the resumed U-turn tail never reaches here), table power, and neither
    # BATTLEMON_TYPE is the type. Those are the raw types: a roosting defender's Flying reads as
    # Mystery in ``state.types`` only for the chart's sake (``damage.chart_types``; no battler has
    # Mystery otherwise). subscript_color_change.s:9-10 sets both; :6 tags the defender's ability.
    types = state.types[party(fmt, h.target_side), h.target].astype(_I32)
    roosting = state.roosting[h.target_side] != 0
    raw = jnp.where(roosting & (types == Type.MYSTERY), Type.FLYING, types)
    color = (
        (ability == AbilityId.COLOR_CHANGE)
        & _alive(state, h.target_side, h.target)
        & h.effective
        & (h.move != MOVE_STRUGGLE)
        & (h.dealt > 0)
        & ~h.sub_hit
        & (tables.move_power[h.move] != 0)
        & ~jnp.any(raw == move_type)
    )
    new_types = jnp.where(color, jnp.stack([move_type, move_type]), types)
    state = state._replace(
        types=state.types.at[party(fmt, h.target_side), h.target].set(
            new_types.astype(state.types.dtype)
        )
    )
    from nuzline.engine.hooks import TYPE_CHANGE

    state = reveal_ability(state, h.target_side, h.target, TYPE_CHANGE, color)
    state = reveal_ability(state, h.target_side, h.target, CONTACT_STATUS, cond | charm)
    # subscript_aftermath.s:6 tags BTLSCR_DEFENDER's ability (BattleMessage_AbilityTag).
    state = reveal_ability(state, h.target_side, h.target, CONTACT_DAMAGE, rough | aftermath)
    return _set_status(state, h.side, h.user, status, cond), charm_proc | status_proc


SYNC_STATUSES = (Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)


def sync_landed(before: BattleState, after: BattleState, side) -> jnp.ndarray:
    """``bool``: ``side``'s active battler (``before``'s) went from no status to poison, bad
    poison, burn or paralysis between the two states: a status subscript reached its end
    (subscript_poison.s:63-69, subscript_badly_poison.s:83-88, subscript_burn.s:79-84,
    subscript_paralyze.s:50-56), which toggles ``SYSCTL_TRY_SYNCHRONIZE_STATUS`` (set when
    clear, cleared when set). Sleep and freeze never touch it (subscript_fall_asleep.s,
    subscript_freeze.s)."""
    fmt = format_of(before)
    mon = before.active[side]
    new = after.status[party(fmt, side), mon]
    return (before.status[party(fmt, side), mon] == Status.NONE) & jnp.isin(
        new, jnp.asarray(SYNC_STATUSES, new.dtype)
    )


def synchronize(
    tables: Tables, state: BattleState, attacker, defender, flag, sem, cond, held=None
) -> tuple[BattleState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``BattleSystem_SynchronizeStatus`` (battle_lib.c:4493-4534), the
    ``AFTER_MOVE_EFFECT_SYNCHRONIZE_STATUS`` step of ``AfterMoveEffects`` (bcp.c:3665-3670), where
    ``cond`` (the step runs: after every hit of a move whose AfterMoveEffects run). Returns
    ``(state, flag, sem, ran)``; ``ran``: a status script ran, which returns before the form change
    and the knot (:4515-4532; ``sync_step``). ``flag``: ``SYSCTL_TRY_SYNCHRONIZE_STATUS``; ``sem``: the
    ``sideEffectMon`` (a battler; the side in singles). ``attacker`` / ``defender``: the action's
    battlers (the defender is the attacker itself for a self-defender move: Baton Pass).

    With the flag set: a defender whose ``Battler_Ability`` is Synchronize and which is the
    side-effect battler passes to the attacker (:4499-4504); else a Synchronize attacker that
    is the side-effect battler (a Static-like ability of the defender statused it) passes to
    the defender (:4506-4511). Either way ``sideEffectMon`` becomes the receiver. The passer's
    status picks the script (:4514-4520): any poison runs subscript_poison.s (a bad poison
    passes as a plain poison), a burn subscript_burn.s, a paralysis subscript_paralyze.s; a
    sleep or a freeze passes nothing. As ``SIDE_EFFECT_TYPE_ABILITY`` (:4524) the receiver's
    own Immunity / Water Veil / Limber (the ignorable checks read the receiver's own ability:
    neither battler of a pass has Mold Breaker in place of the ability tested), Leaf Guard in
    sun (poison :13-16, burn :20-23, paralyze :5-8), any status or a Poison / Steel (poison) or
    Fire (burn) type ends it silently; it skips CheckSubstitute (poison :28, burn :35, paralyze
    :20) and Safeguard (:39, :43, :26). Nothing tests HP: the status lands on a receiver of any
    HP, but a fainted one's is cleared at its faint (``step._faint_check``,
    battle_controller.c:1482-1486), so it is set only on a living one here. The passer's status
    is its ``BattleMon.status``, which a faint does not clear (only the party copy is zeroed,
    battle_controller.c:1481-1486): ``held`` (``int[2]``, per battler, default each side's active
    Pokemon's ``state.status``) gives it for a passer that fainted since (ruling G-24). The message (poison
    :57, burn :68, paralyze :44: ``BTLSCR_MSG_BATTLER_TEMP``'s ability, the passer) tags
    Synchronize: the AI learns a player's (``hooks.REVEALS_ON``). A status that lands toggles
    the flag off again (rulings G-12, G-13). ``held`` gives the receiver's status test its
    ``BattleMon.status`` likewise: a receiver that fainted with a status takes nothing (G-31)."""
    fmt = format_of(state)
    from nuzline.engine.hooks import SYNCHRONIZED, reveal_ability  # hooks imports this module

    attacker = jnp.asarray(attacker, _I32)
    defender = jnp.asarray(defender, _I32)
    flag = jnp.asarray(flag, bool)
    sem = jnp.asarray(sem, _I32)
    a_mon, d_mon = state.active[attacker], state.active[defender]
    on = jnp.asarray(cond, bool) & flag
    first = (
        on & (battler_ability(state, defender, d_mon) == AbilityId.SYNCHRONIZE) & (sem == defender)
    )
    second = (
        on
        & ~first
        & (battler_ability(state, attacker, a_mon) == AbilityId.SYNCHRONIZE)
        & (sem == attacker)
    )
    run = first | second
    passer = jnp.where(first, defender, attacker)
    receiver = jnp.where(first, attacker, defender)
    p_mon, r_mon = state.active[passer], state.active[receiver]
    # The receiver's status test reads its BattleMon.status too (subscript_paralyze.s:41-42,
    # subscript_poison.s, subscript_burn.s), kept by a faint: a fainted receiver that was
    # statused takes nothing (ruling G-31).
    r_held = (
        state.status[party(fmt, receiver), r_mon] if held is None else jnp.asarray(held)[receiver]
    )
    held = state.status[party(fmt, passer), p_mon] if held is None else jnp.asarray(held)[passer]
    kind = jnp.where(
        (held == Status.POISON) | (held == Status.TOXIC),
        Status.POISON,
        jnp.where(
            held == Status.BURN,
            Status.BURN,
            jnp.where(held == Status.PARALYSIS, Status.PARALYSIS, Status.NONE),
        ),
    )
    own = battler_decomp_ability(tables, state, receiver, r_mon)
    ability = battler_ability(state, receiver, r_mon)
    leaf = _sunny(tables, state) & (ability == AbilityId.LEAF_GUARD)
    shield = jnp.where(
        kind == Status.POISON,
        own == ITEM_ABILITY_IDS["Immunity"],
        jnp.where(
            kind == Status.BURN,
            own == ITEM_ABILITY_IDS["Water Veil"],
            ability == AbilityId.LIMBER,
        ),
    )
    lands = (
        run
        & (kind != Status.NONE)
        & ~shield
        & ~leaf
        & can_have_status(tables, state, receiver, r_mon, kind, r_held)
    )
    state = _set_status(state, receiver, r_mon, kind, lands & _alive(state, receiver, r_mon))
    state = reveal_ability(state, passer, p_mon, SYNCHRONIZED, lands)
    return state, flag & ~lands, jnp.where(run, receiver, sem), run & (kind != Status.NONE)


def sync_step(
    tables: Tables, state: BattleState, attacker, defender, flag, sem, loved, cond, held=None
) -> tuple[BattleState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(state, flag, sem, loved, failed)``: the whole of ``BattleSystem_SynchronizeStatus``
    (battle_lib.c:4493-4572) where ``cond``: ``synchronize``, which returns when a status script
    ran; then ``BattleSystem_TriggerFormChange`` (:4536-4544), which returns when a form changes
    (``form_change_pending``; the change itself runs at the later form-change checks, so the knot
    of this hit is lost, H-4, H-5); then ``knot``. ``failed``: the knot's ``MOVE_STATUS_FAILED``."""
    state, flag, sem, ran = synchronize(tables, state, attacker, defender, flag, sem, cond, held)
    knots = jnp.asarray(cond, bool) & ~ran & ~form_change_pending(tables, state)
    state, sem, loved, failed = knot(tables, state, attacker, defender, loved, sem, knots)
    return state, flag, sem, loved, failed


def form_change_pending(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool``: ``BattleSystem_TriggerFormChange`` (battle_lib.c:6305-6451) would change a form
    now: any of its four blocks (``hooks.form_change``: Castform's types, Cherrim's and Arceus's
    ``form_num``, a Transformed Giratina-Origin's rebuild, which shows in ``form_num``), H-5. A
    Giratina Origin that is not Transformed and lacks the Orb is refused by coverage: it can lack
    it only from the battle's start, since no move takes the Orb (``hooks.form_change``)."""
    from nuzline.engine.hooks import form_change  # hooks imports this module

    new = form_change(tables, state)
    return jnp.any(new.types != state.types) | jnp.any(new.form_num != state.form_num)


def knot(
    tables: Tables, state: BattleState, attacker, defender, loved, sem, cond
) -> tuple[BattleState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(state, sem, loved, failed)``: the Destiny Knot half of ``BattleSystem_SynchronizeStatus``
    (battle_lib.c:4546-4566) where ``cond``, the step reached it (``synchronize`` did not run a
    status script and ``form_change_pending`` is false: both return first, :4515-4532, :4536-4544).
    ``loved`` (``bool[B]``) is each battler's ``SELF_TURN_FLAG_INFATUATED``, set by
    ``subscript_infatuate.s:29`` on a landed Attract, Cute Charm or knot, cleared at the action's
    end (battle_lib.c:2013-2015); ``sem`` is ``sideEffectMon`` (a battler).

    A defender (``Battler_HeldItemEffect``: Klutz / Embargo hide it) holding the knot that is
    ``sem`` and flagged reflects to the attacker; else an attacker holding it that is ``sem`` and
    flagged reflects to the defender (:4546-4559). ``sem`` becomes the receiver before the
    script. ``subscript_infatuate.s`` (type ``HELD_ITEM``): the receiver's ignorable Oblivious
    (:5, :98: no effect, the message names both, the AI learns the ability and the holder's item;
    the receiver must have HP, ``BtlCmd_CheckIgnorableAbility`` :7716-7720); else
    ``BtlCmd_TryAttract`` (bs:5727-5742: genders differ, neither genderless, the receiver not in
    love), which on failure sets ``MOVE_STATUS_FAILED`` (:108-110: ``failed``, which the action's
    ``NO_EFFECTS`` tests read: Conversion 2's record, the Metronome count); on success the
    receiver's attract bit, the message (the AI learns the holder's item) and its flag (:29).
    The missed branch (:6) is never met: the step runs after a hit that connected."""
    from nuzline.engine.hooks import ROMANCE_BLOCKED, reveal_ability  # hooks imports this module

    attacker = jnp.asarray(attacker, _I32)
    defender = jnp.asarray(defender, _I32)
    sem = jnp.asarray(sem, _I32)
    loved = jnp.asarray(loved, bool)
    a_mon, d_mon = state.active[attacker], state.active[defender]
    on = jnp.asarray(cond, bool)
    first = (
        on
        & (tables.item_effect[held_item(state, defender, d_mon)] == ItemId.DESTINY_KNOT)
        & (sem == defender)
        & loved[defender]
    )
    second = (
        on
        & ~first
        & (tables.item_effect[held_item(state, attacker, a_mon)] == ItemId.DESTINY_KNOT)
        & (sem == attacker)
        & loved[attacker]
    )
    run = first | second
    holder = jnp.where(first, defender, attacker)
    receiver = jnp.where(first, attacker, defender)
    h_mon, r_mon = state.active[holder], state.active[receiver]
    oblivious = (
        run
        & (ignorable(state, attacker, receiver) == AbilityId.OBLIVIOUS)
        & _alive(state, receiver, r_mon)
    )
    landed = run & ~oblivious & _attractable(state, receiver, r_mon, holder, h_mon)
    failed = run & ~oblivious & ~landed
    state = reveal_ability(state, receiver, r_mon, ROMANCE_BLOCKED, oblivious)
    state = reveal_item(state, holder, h_mon, oblivious | landed)
    state = _set(state, "infatuated", receiver, 1, landed)
    loved = loved.at[receiver].set(loved[receiver] | landed)
    return state, jnp.where(run, receiver, sem), loved, failed


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
    ``_shield_dust``), after CheckSubstitute (:6: ``h.shielded``). A U-turn that
    switches makes its check with the newcomer's item (``kings_rock_flinch``, from ``step._use_move``)."""
    cond = h.landed & h.update_hp & (outcome.flinch_item[h.side, h.hit_index] != 0)
    cond = cond & ~h.shielded
    return kings_rock_flinch(tables, state, h.side, h.target_side, h.move, cond & ~h.target_moved)


def kings_rock_flinch(tables, state, side, target_side, move, cond) -> BattleState:
    """``CheckExtraFlinch``'s flinch (``_kings_rock``) by ``side``'s active battler on
    ``target_side``'s where ``cond`` (the hit landed with damage, its ``flinch_item`` roll passed
    and the target has not moved): the attacker's item now (``Battler_HeldItemEffect(attacker)``,
    bcp.c:4633; after a U-turn switch the newcomer's, with no U-turn guard, :4636-4642), the move's
    ``MOVE_FLAG_TRIGGERS_KINGS_ROCK``, a living target, and subscript_flinch_mon.s's Inner Focus
    (:7) and Shield Dust (:9) as this attacker sees them (``Battler_IgnorableAbility``)."""
    user, target = state.active[side], state.active[target_side]
    item = held_item(state, side, user)
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


def thaw_target(
    tables: Tables, state: BattleState, side, target_side, move, dealt, cond, gift=0
) -> tuple:
    """``(state, thawed)``: ``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (battle_controller_player.c:
    3708-3727), after the held-item checks and the Sticky Barb transfer of the hit
    (``step._use_move``): a frozen defender with HP left that took damage from a Fire-type
    ``move`` of ``side`` (``dealt``: its HP drop on this hit) thaws, where ``cond`` (the
    move's ``AfterMoveEffects`` run). ``target_side``: the defender's side."""
    fmt = format_of(state)
    target = state.active[target_side]
    thawed = (
        jnp.asarray(cond)
        & (state.status[party(fmt, target_side), target] == Status.FREEZE)
        & (move_type_of(tables, state, side, move, gift) == Type.FIRE)
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
    ``heldItem`` is ``ITEM_NONE``, not knocked off, a move with ``MOVE_FLAG_MAKES_CONTACT`` that
    is not Knock Off (:5390-5391, B10-6); U-turn's guard is ``step``'s.
    ``subscript_transfer_sticky_barb.s`` moves the item: the attacker holds it, the defender
    nothing (no ``RemoveItem``: ``item_consumed`` is untouched). ``foe``: the defender's side."""
    fmt = format_of(state)
    user, target = state.active[side], state.active[foe]
    # The defender's Battler_HeldItemEffect (:5383, B13-29); the attacker's raw heldItem (:5389).
    barb = held_item(state, foe, target)
    contact = (tables.move_flags[move] & MoveFlag.CONTACT) != 0
    on = (
        jnp.asarray(cond)
        & (tables.item_effect[barb] == ItemId.STICKY_BARB)
        & _alive(state, side, user)
        & (state.item[party(fmt, side), user] == 0)
        & contact
        # :5390-5391: not an attacker whose side has its slot's knockedOffItemsMask bit, not
        # Knock Off itself (B10-6; the side's bit, D3 fix B).
        & ~knocked_off_at(state, side, user)
        & (move != MOVE_KNOCK_OFF)
    )
    state = _set(state, "item", (side, user), barb, on)
    state = forget_item(state, foe, target, on)
    return _set(state, "item", (foe, target), 0, on)


def on_hit_berry(
    tables: Tables, state: BattleState, side, foe, physical, special, super_effective, cond
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, recoiled)``: the berry cases of ``BattleSystem_TriggerHeldItemOnHit``
    (battle_lib.c:5370-5435) on ``foe``'s (the defender's side) active Pokemon after a hit of
    ``side``'s move, where ``cond`` (the caller's gate: ``AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT``,
    bcp.c:3698-3706, or subscript_pursuit.s:44, :90; the substitute was not hit this action,
    :5378). One switch on the defender's hold effect (``Battler_HeldItemEffect``; Klutz and
    Embargo are unsupported), so the Sticky Barb case (``sticky_barb_transfer``) never meets these:

    * Jaboca Berry (``HOLD_EFFECT_RECOIL_PHYSICAL``, :5400-5409): the defender's
      ``physicalDamageTaken`` (``physical``), a living attacker without Magic Guard; Rowap Berry
      (``_RECOIL_SPECIAL``, :5411-5419) the same with ``specialDamageTaken`` (``special``). The
      holder may have fainted (no HP test on it). subscript_held_item_recoil_when_hit.s: the
      attacker loses ``BattleSystem_Divide(maxHP * -1, itemPower)`` through
      ``BATTLE_SUBSCRIPT_UPDATE_HP`` with ``SYSCTL_SKIP_SPRITE_BLINK`` (no resist berry; it may
      faint), then ``RemoveItem BTLSCR_DEFENDER``. ``recoiled``: this ran (the caller then
      re-checks the attacker's held item, ``AFTER_MOVE_EFFECT_HELD_ITEM_STATUS``, bcp.c:3730).
    * Enigma Berry (``HOLD_EFFECT_HP_RESTORE_SE``, :5421-5428): a living defender and
      ``MOVE_STATUS_SUPER_EFFECTIVE`` (``super_effective``): subscript_held_item_hp_restore.s
      heals ``BattleSystem_Divide(maxHP, itemPower)``, capped, and its ``PLUCK_CHECK`` removes
      the berry (the attacker's pluck flag cannot be set: a Pluck of this hit ate the defender's
      berry first).

    ``physical`` / ``special``: the damage-taken flags the move's class set this action
    (bcp.c:3411-3424). Ruling G-54, G-55."""
    fmt = format_of(state)
    side = jnp.asarray(side, _I32)
    foe = jnp.asarray(foe, _I32)
    user, target = state.active[side], state.active[foe]
    item = held_item(state, foe, target)  # Battler_HeldItemEffect / _HeldItemPower (B13-29)
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    cond = jnp.asarray(cond)
    a_hp = state.hp[party(fmt, side), user].astype(_I32)
    magic_guard = (
        battler_decomp_ability(tables, state, side, user) == WEATHER_ABILITY_IDS["Magic Guard"]
    )
    recoil = (
        cond
        & (
            ((effect == ItemId.JABOCA_BERRY) & jnp.asarray(physical))
            | ((effect == ItemId.ROWAP_BERRY) & jnp.asarray(special))
        )
        & (a_hp > 0)
        & ~magic_guard
    )
    loss = divide(state.max_hp[party(fmt, side), user].astype(_I32), jnp.maximum(power, 1))
    state = hp_update(state, side, user, -loss, recoil)
    d_hp = state.hp[party(fmt, foe), target].astype(_I32)
    d_max = state.max_hp[party(fmt, foe), target].astype(_I32)
    heal = cond & (effect == ItemId.ENIGMA_BERRY) & (d_hp > 0) & jnp.asarray(super_effective)
    gain = divide(d_max, jnp.maximum(power, 1))
    state = hp_update(state, foe, target, gain, heal)
    return consume_item(state, foe, recoil | heal), recoil


def sticky_barb_ready(tables: Tables, state: BattleState, side, foe, move) -> jnp.ndarray:
    """``bool``: ``side``'s attacker would take ``foe``'s Sticky Barb (``sticky_barb_transfer``'s
    tests but the damage flag)."""
    fmt = format_of(state)
    user, target = state.active[side], state.active[foe]
    contact = (tables.move_flags[move] & MoveFlag.CONTACT) != 0
    return (
        (tables.item_effect[held_item(state, foe, target)] == ItemId.STICKY_BARB)
        & _alive(state, side, user)
        & (state.item[party(fmt, side), user] == 0)
        & contact
        # TriggerHeldItemOnPivotMove (battle_lib.c:7380): not an attacker whose side has its
        # slot's knockedOffItemsMask bit (B10-6; the side's bit, D3 fix B).
        & ~knocked_off_at(state, side, user)
    )


def pivot_item(
    tables: Tables, state: BattleState, side, foe, move, shell_dealt, move_hit, damaged
) -> BattleState:
    """``BattleSystem_TriggerHeldItemOnPivotMove`` (battle_lib.c:7336-7386) once, at U-turn's
    subscript_attack_then_switch_out.s:11: four independent ``if``s each overwrite the subscript,
    so only the last match runs: the defender's Sticky Barb (:7378-7385: a living attacker holding
    nothing, the defender's damage taken, a contact move; no U-turn guard, the knocked-off mask comes
    with Knock Off), else the defender's Jaboca Berry (:7368-7375, ``HOLD_EFFECT_RECOIL_PHYSICAL``:
    the defender's physicalDamageTaken, a living attacker without Magic Guard;
    ``on_hit_berry``'s recoil, G-54), else the attacker's Life Orb (:7357-7366:
    ``SYSCTL_MOVE_HIT``, not a status move, a living holder without Magic Guard) or Shell Bell
    (:7345-7355; one holder holds one of them). ``move_hit``: ``SYSCTL_MOVE_HIT``; ``damaged``:
    the defender's physical / specialDamageTaken; ``shell_dealt``: ``shellBellDamageDealt``."""
    barb = jnp.asarray(damaged) & sticky_barb_ready(tables, state, side, foe, move)
    state = sticky_barb_transfer(tables, state, side, foe, move, barb)
    physical = jnp.asarray(damaged) & (tables.move_category[move] == PHYSICAL)
    holder = held_item(state, foe)  # defenderItemEffect (battle_lib.c:7342, B13-29)
    jaboca = ~barb & physical & (tables.item_effect[holder] == ItemId.JABOCA_BERRY)
    state, jabbed = on_hit_berry(tables, state, side, foe, physical, False, False, jaboca)
    hit = jnp.asarray(move_hit) & ~barb & ~jabbed
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
    fmt = format_of(state)
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    item = held_item(state, side, mon)  # bcp.c:4719-4720 (B13-29)
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    max_hp = state.max_hp[party(fmt, side), mon].astype(_I32)
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


def toxic_spikes_reached(tables: Tables, state: BattleState, side, layers) -> jnp.ndarray:
    """``bool``: subscript_hazards_check.s reaches ``CheckToxicSpikes`` (:14) for ``side``'s active
    Pokemon with ``layers`` (the side's count before an absorb) above 0: no Magic Guard (:5), and
    ``damage.grounded`` (:6-11). ``BtlCmd_CheckToxicSpikes`` then makes it the sideEffectMon
    (battle_script.c:7638-7641), poisoned or not (H-4)."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD
    return grounded(tables, state, side, mon) & ~guarded & (jnp.asarray(layers, _I32) > 0)


def entry_hazards(tables: Tables, state: BattleState, side) -> tuple[BattleState, jnp.ndarray]:
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
      its HP.

    Returns ``(state, stopped)``: ``stopped`` is MOVE_STATUS_NO_MORE_WORK of the Toxic Spikes
    block (Immunity, or Leaf Guard in sun without a weather lock, plain ability:
    subscript_poison.s:5-10, :127; subscript_badly_poison.s:5-10, :140). The caller's move action
    owns the skip (E3-B2-32)."""
    fmt = format_of(state)
    from nuzline.engine.ids import ITEM_ABILITY_IDS

    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD
    layers = state.toxic_spikes[side_of(side)].astype(_I32)
    reached = toxic_spikes_reached(tables, state, side, layers)
    poison_type = jnp.any(state.types[party(fmt, side), mon] == Type.POISON)
    absorbed = reached & poison_type
    state = _set(state, "toxic_spikes", side, 0, absorbed)
    ability = battler_decomp_ability(tables, state, side, mon)
    sun = _sunny(tables, state)
    status = jnp.where(layers >= 2, Status.TOXIC, Status.POISON)
    # subscript_poison.s:5-10 / subscript_badly_poison.s:5-10 (CheckAbility, before the status
    # and type checks): Immunity, or Leaf Guard in sun, jumps to "{0}'s {1} prevents
    # poisoning!" (subscript_poison.s:80-82, subscript_badly_poison.s:93-103), whose
    # TAG_NICKNAME_ABILITY runs BattleAI_SetAbility (battle_script.c:11798-11806).
    blocks = (ability == ITEM_ABILITY_IDS["Immunity"]) | (
        sun & (ability == ITEM_ABILITY_IDS["Leaf Guard"])
    )
    stopped = reached & ~poison_type & blocks
    known = jnp.where(stopped & (party(fmt, side) == 0), ability, state.ai_ability_known[mon])
    state = state._replace(
        ai_ability_known=state.ai_ability_known.at[mon].set(
            known.astype(state.ai_ability_known.dtype)
        )
    )
    poisoned = (
        reached
        & ~poison_type
        & ~blocks
        # subscript_poison.s:29 / subscript_badly_poison.s:46: CheckSubstitute after the ability
        # tests, silent: a Baton Pass newcomer keeps the passer's substitute (B10-4).
        & (state.substitute_hp[side] == 0)
        # subscript_poison.s:41 / _badly_poison.s:55: the side's Safeguard (B10-7).
        & (state.safeguard[side_of(side)] == 0)
        & can_have_status(tables, state, side, mon, status)
    )
    state = _set_status(state, side, mon, status, poisoned)
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    spikes = jnp.where(
        (hp > 0) & ~guarded, jnp.minimum(spikes_loss(tables, state, side, mon), hp), 0
    )
    hp = hp - spikes
    rock = jnp.where(
        (hp > 0) & ~guarded, jnp.minimum(stealth_rock_loss(tables, state, side, mon), hp), 0
    )
    # subscript_hazards_check.s: each through UPDATE_HP (B13-17; replace() clears the flag of a
    # turn-end replacement, SetupNextTurn).
    state = hp_update(state, side, mon, -spikes, spikes > 0)
    return hp_update(state, side, mon, -rock, rock > 0), stopped


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
    # The defender's SELF_TURN_FLAG_SUBSTITUTE_HIT (bcp.c:3371), set by this hit or an earlier one
    # of the action; it lasts until BattleContext_Init at the action's end (battle_lib.c:2014).
    sub_hit: jnp.ndarray
    # MOVE_STATUS_SUPER_EFFECTIVE after the hit's type chart (the Enigma Berry, on_hit_berry).
    super_effective: jnp.ndarray
    # Trick Room's script ran on this hit (``_trick_room``'s condition, set or restore): it sets
    # trickRoomActivated (effect_script_0259.s:18; battle_script.c:7859-7866), so the action's
    # MoveEnd re-sorts the order (bcp.c:3989-3994; battle_lib.c:7276-7300), SF2 (k).
    trick_room: jnp.ndarray
    # A contact-ability trigger passed on this hit (Cute Charm, H-4; Static, Poison Point, Flame
    # Body, H-M2): sideEffectMon is the attacker, landed or not (``_contact_ability``).
    sem_proc: jnp.ndarray


def focus_sash_damage(state: BattleState, side, damage, armed) -> tuple:
    """``(dealt, endured)`` for ``damage`` (> 0) taken by ``side``'s active Pokemon:
    ``BattleControllerPlayer_UpdateHP`` (bcp.c:3386-3404) / ``BtlCmd_CheckHoldOnWith1HP``
    (script.c:9098-9124). ``armed`` (the hit's focus flag, or a full-HP Focus Sash
    holder) and ``curHP + damage <= 0`` leave it at 1 HP (``MOVE_STATUS_ENDURED_ITEM``);
    else the damage is capped at its HP. ``subscript_move_followup_message.s:12-19``
    then removes the item only while it still is a Focus Sash
    (``CheckItemHoldEffect ... HOLD_EFFECT_ENDURE``)."""
    fmt = format_of(state)
    mon = state.active[side]
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    endured = armed & (damage > 0) & (damage >= hp)
    return jnp.where(endured, hp - 1, jnp.minimum(damage, hp)), endured


# Effects whose script sets SYSCTL_IGNORE_TYPE_CHECKS or SYSCTL_IGNORE_IMMUNITIES: Night Shade
# and Seismic Toss (effect_script_0087.s:5), Foresight (effect_script_0113.s:6, no damage), Future
# Sight (effect_script_0148.s:7, no damage at its use).
# subscript_type_resist_berry.s:5 skips the berry for such a hit. tests/tools/
# test_effect_map.py checks the list against every mapped effect script.
IGNORES_TYPE_CHECKS = (
    _E.LEVEL_DAMAGE,
    _E.FORESIGHT,
    _E.MIRACLE_EYE,  # effect_script_0216.s:6 (E3-B2-10)
    _E.HALVE_HP,
    _E.COUNTER,
    _E.FUTURE_SIGHT,
    _E.ENDEAVOR,  # effect_script_0189.s:10 (B10-5)
    _E.BEAT_UP,  # effect_script_0154.s:5 (B13-10)
)
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
    item = held_item(state, side)  # CheckItemHoldEffect / GetItemEffectParam (B13-29)
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
    fmt = format_of(state)
    hp = state.hp[party(fmt, side), state.active[side]].astype(_I32)
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
    fmt = format_of(state)
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    magic_guard = (
        battler_decomp_ability(tables, state, side, mon) == WEATHER_ABILITY_IDS["Magic Guard"]
    )
    on = jnp.asarray(cond) & holds(tables, state, side, ItemId.LIFE_ORB) & (hp > 0) & ~magic_guard
    loss = divide(state.max_hp[party(fmt, side), mon].astype(_I32), 10)
    return hp_update(state, side, mon, -loss, on)


METRONOME_MAX_TURNS = 10  # BattleSystem_UpdateMetronomeCount's cap (battle_lib.c:6168)


LAST_RESORT_MAX = 4  # lastResortMoves[4] (battle_mon.h:50): BattleSystem_UpdateLastResort's cap


def last_resort_update(state: BattleState, side, move, cond) -> BattleState:
    """``BattleSystem_UpdateLastResort`` (battle_lib.c:3443-3459) for ``side``'s battler where
    ``cond`` (the end of ``BattleControllerPlayer_BeforeMove`` without ``MOVE_STATUS_NO_EFFECTS``,
    bcp.c:3209-3218, the point ``metronome_update`` reads, but past DecrementPP's no-PP flag).
    ``move`` is ``moveTemp``: the chosen or encored move, Struggle, an invoker itself (never the
    move it calls). Last Resort itself and a full list (4) record nothing (:3445-3448), nor does a
    move already among the first ``lastResortCount`` entries (:3451-3455); else it takes the next
    entry and the count rises (:3457-3458). The ids are kept, not slots: a Mimic-ed slot keeps
    Mimic's id counted and Struggle's counts too. E3-B2-11."""
    side = jnp.asarray(side, _I32)
    move = jnp.asarray(move, _I32)
    count = state.last_resort_count[side].astype(_I32)
    listed = state.last_resort_moves[side].astype(_I32)
    place = jnp.arange(LAST_RESORT_MAX)
    seen = jnp.any((listed == move) & (place < count))
    record = jnp.asarray(cond) & (move != MOVE_LAST_RESORT) & (count < LAST_RESORT_MAX) & ~seen
    moves = jnp.where(record & (place == count), move, listed)
    state = state._replace(
        last_resort_moves=state.last_resort_moves.at[side].set(
            moves.astype(state.last_resort_moves.dtype)
        )
    )
    return _set(state, "last_resort_count", side, count + 1, record)


def last_resort_fails(state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``BtlCmd_TryLastResort`` (battle_script.c:7567-7579) jumps to
    ``MOVE_STATUS_FAILED`` (effect_script_0246.s:5, :10-11) for ``side``'s battler: its
    ``lastResortCount`` is under ``numMoves - 1`` or it knows fewer than 2 moves, ``numMoves``
    being ``Battler_CountMoves`` (battle_lib.c:3461-3471): the index of the first empty slot of
    the battler's own moves. E3-B2-11."""
    fmt = format_of(state)
    moves = state.moves[party(fmt, side), state.active[side]]
    known = jnp.sum(jnp.cumprod((moves != 0).astype(_I32)))
    return (state.last_resort_count[side].astype(_I32) < known - 1) | (known < 2)


def metronome_update(tables: Tables, state: BattleState, side, move, cond) -> BattleState:
    """``BattleSystem_UpdateMetronomeCount`` (battle_lib.c:6160-6178) for ``side``'s battler
    where ``cond`` (the end of ``BattleControllerPlayer_BeforeMove``, bcp.c:3221: the move got
    past the status checks and has a target). A holder of ``HOLD_EFFECT_BOOST_REPEATED``
    (``Battler_HeldItemEffect``) that is not uproaring nor ``VOLATILE_CONDITION_MOVE_LOCKED``
    (``locked_move``; Thrash is unsupported, and ``SYSCTL_LAST_OF_MULTI_TURN`` is clear at
    BeforeMove, BattleContext_Init cleared it) counts ``moveTemp`` (``move``: the chosen or
    encored move, Struggle, an invoker itself): the same move as ``metronomeMove`` adds 1 up to
    10, another one sets the count to 0 and becomes ``metronomeMove``. A battler that holds no
    Metronome has its own count reset to 0 (:6176-6177). As the update follows the
    ``MOVE_STATUS_NO_EFFECTS`` branch (bcp.c:3209-3221), a move DecrementPP flagged
    ``MOVE_STATUS_NO_PP`` (bcp.c:2285) counts too; only Spite (``_spite``, E3-B2-4) empties a chosen move's PP
    before its use, as does the encored move with no PP (B10-9). G-60."""
    side = jnp.asarray(side, _I32)
    move = jnp.asarray(move, _I32)
    cond = jnp.asarray(cond)
    holder = holds(tables, state, side, ItemId.METRONOME_ITEM)
    exempt = (state.uproar[side] != 0) | (state.locked_move[side] != 0)
    turns = state.metronome_turns[side].astype(_I32)
    same = state.metronome_move[side].astype(_I32) == move
    counts = cond & holder & ~exempt
    new = jnp.where(same, jnp.minimum(turns + 1, METRONOME_MAX_TURNS), 0)
    turns = jnp.where(counts, new, jnp.where(cond & ~holder, 0, turns))
    state = _set(state, "metronome_turns", side, turns, True)
    return _set(state, "metronome_move", side, move, counts & ~same)


def metronome_verify(
    tables: Tables, state: BattleState, side, move, no_effects, repeated, last_of_multi, cond
) -> BattleState:
    """``BattleSystem_VerifyMetronomeCount`` (battle_lib.c:6181-6198) at
    ``UpdateMoveBuffers`` (bcp.c:3957) where ``cond``: a holder whose move had
    ``MOVE_STATUS_NO_EFFECTS`` (``no_effects``) takes back the count it added: ``moveTemp`` is
    ``metronomeMove``, the count is above 0, ``repeatedMoveCount`` is 0 (``repeated``: Rollout's
    count before its script, battle_script.c:5665), not uproaring, not
    ``SYSCTL_LAST_OF_MULTI_TURN`` (``last_of_multi``: a charge move's strike, Bide's end), not
    move-locked (``locked_move`` now). A battler that holds no Metronome has its own count reset
    to 0 (:6195-6196). G-60."""
    side = jnp.asarray(side, _I32)
    cond = jnp.asarray(cond)
    holder = holds(tables, state, side, ItemId.METRONOME_ITEM)
    turns = state.metronome_turns[side].astype(_I32)
    back = (
        cond
        & holder
        & jnp.asarray(no_effects)
        & (state.metronome_move[side].astype(_I32) == jnp.asarray(move, _I32))
        & (turns > 0)
        & ~jnp.asarray(repeated)
        & (state.uproar[side] == 0)
        & ~jnp.asarray(last_of_multi)
        & (state.locked_move[side] == 0)
    )
    turns = jnp.where(back, turns - 1, jnp.where(cond & ~holder, 0, turns))
    return _set(state, "metronome_turns", side, turns, True)


def holds(tables: Tables, state: BattleState, side, item: ItemId) -> jnp.ndarray:
    """``side``'s active Pokemon holds an item with effect ``item`` as ``Battler_HeldItemEffect``
    reads it (``state.held_item``: none under Klutz or Embargo, B13-29)."""
    return tables.item_effect[held_item(state, side)] == item


def consume_item(state: BattleState, side, cond) -> BattleState:
    """``BtlCmd_RemoveItem`` (battle_script.c:8972) on ``side``'s active Pokemon where
    ``cond``: the item goes to ``recycleItem[battler]`` (``recycle_item``, :8977, B13-19) and
    ``item_consumed``; its ``BattleMon_CopyToParty`` (:8981) then clears the AI's record of the
    item (``forget_item``)."""
    fmt = format_of(state)
    mon = state.active[side]
    item = state.item[party(fmt, side), mon]
    state = _set(state, "recycle_item", side, item, cond)
    state = _set(state, "item_consumed", (side, mon), item, cond)
    state = forget_item(state, side, mon, cond)
    return _set(state, "item", (side, mon), 0, cond)


def reveal_item(state: BattleState, side, mon, cond) -> BattleState:
    """A message names the held item of ``side``'s Pokemon in party slot ``mon`` where
    ``cond``: ``BattleMessage_ItemTag`` (battle_script.c:11726-11745) runs
    ``BattleAI_SetHeldItem`` (:12119-12122) for an item tag of ``BTLSCR_ATTACKER``,
    ``BTLSCR_DEFENDER`` or ``BTLSCR_MSG_BATTLER_TEMP`` (not ``BTLSCR_MSG_TEMP``, which names
    ``msgItemTemp``), so ``ai_item_known[mon]`` becomes the item. Only a party-0 battler's
    (ruling D3-1a) is kept: ``AICmd_IfHeldItemEqualTo`` and ``AICmd_LoadHeldItemEffect``
    (trainer_ai.c:1901-1934) read ``battlerHeldItems`` for the other side's battlers only."""
    fmt = format_of(state)
    on = jnp.asarray(cond) & (party(fmt, jnp.asarray(side, _I32)) == 0)
    known = state.ai_item_known
    value = jnp.where(on, state.item[party(fmt, side), mon], known[mon]).astype(known.dtype)
    return state._replace(ai_item_known=known.at[mon].set(value))


def forget_item(state: BattleState, side, mon, cond) -> BattleState:
    """``side``'s Pokemon in party slot ``mon`` loses its held item where ``cond``: every such
    path (``BtlCmd_RemoveItem``, battle_script.c:8979-8981; ``UpdateMonData`` setting
    ``BATTLEMON_HELD_ITEM``, :3077-3078, as subscript_transfer_sticky_barb.s:7 does) runs
    ``BattleMon_CopyToParty``, which clears the AI's record of a battler holding nothing
    (``BattleAI_ClearKnownItem``, battle_lib.c:1819-1823, :7735-7738). A party-0 battler
    only (ruling D3-1a, as ``reveal_item``)."""
    on = jnp.asarray(cond) & (party(format_of(state), jnp.asarray(side, _I32)) == 0)
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
    target_side=None,
    defender=None,
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
        target_side=target_side,
        defender=defender,
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
    sub_hit=False,
    pursuit_switch=False,
    grudge_slot=None,
    foe_claw=False,
    foe_enduring=False,
    in_power=0,
    gift=0,
    target_side=None,
    defender=None,
    script_calc=False,
    fling=0,
    fling_pick=0,
    coat_slot=0,
) -> tuple[BattleState, MoveHit]:
    """Hit ``hit_index`` (0-5, an int or a traced int32) of ``move`` by ``side``'s active Pokemon: damage with
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
    defender holds a Focus Sash at full HP (bcp.c:3391-3393), or a Focus Band
    (``HOLD_EFFECT_MAYBE_ENDURE``) whose roll passes
    (bcp.c:3387-3389: drawn on every such hit, lethal or not, ``Outcome.focus_band[side,
    hit_index]``, ruling G-1); while it is set, a hit that would KO leaves 1 HP
    (``focus_sash_damage``) and the followup message names the item (``reveal_item``) and
    removes it only while it is a Focus Sash. The flag is never cleared between hits, so
    after a full-HP hit every later hit of the same move is capped as well (a holder at 1 HP
    takes 0); a holder below full HP at the first hit is never protected. Bide stores the
    capped damage (bcp.c:3407), drain and the contact ability see it. Sturdy only blocks
    OHKO moves (``MOVE_STATUS_ONE_HIT_KO``, bcp.c:3342), none of which is supported.

    Endure (G-65, ``foe_enduring``: the defender's ``turnFlags.enduring``, set by an earlier
    Endure this turn, battle_script.c:5174-5177): inside ``if (enduring == 0)`` (bcp.c:3386-3394)
    the Focus Band roll is not drawn and the Focus Sash is not armed; then ``enduring ||
    focusItemActivated`` with ``curHP + damage <= 0`` leaves 1 HP (:3396-3404) with
    ``MOVE_STATUS_ENDURED`` (a message only) when ``enduring``, else ``MOVE_STATUS_ENDURED_ITEM``
    (the item is named and a Sash removed). An earlier hit's ``focus_armed`` still applies (an OR).

    Substitute (bcp.c:3359-3378, B10-4): a defender behind one takes the damage on it instead:
    min(damage, its HP) (``hitDamage``: drain and recoil read it, and shellBellDamageDealt), the
    substitute breaking at 0, and ``SELF_TURN_FLAG_SUBSTITUTE_HIT``. UpdateHP then returns
    before False Swipe's cap, the Focus Sash, Bide's store, the damage-taken fields, the resist
    berry and ``SYSCTL_MOVE_HIT``. ``sub_hit``: an earlier hit of this action set the flag.
    ``pursuit_switch`` (static): Pursuit's pre-switch hit, whose subscript_pursuit.s:20, :56-73
    takes the substitute down only when it breaks (``substituteHP <= damage``) and otherwise
    leaves its HP as it was (no ``UpdateMonData`` on that path), sets no flag and skips the
    on-hit ability and items and ``SYSCTL_MOVE_HIT`` (it goes to :158). ``grudge_slot``: with
    ``pursuit_switch``, the pursuer's selected slot (moveSlot, not the PP slot) for the TryGrudge of
    subscript_pursuit.s:85 (subscript_faint_check_destiny_bond.s:28), which runs after the Destiny
    Bond take-down and before the contact ability (:86-87).

    ``script_calc``: the move's script ran for hit 0 (``step._use_move``: the user acted, had a
    target, its script did not fail and it is not charging), so its CalcDamage ran before TryMove's
    accuracy roll and CheckMoveHitOverrides (bcp.c:3214-3216, :3254-3270): the Me First counter
    update (battle_script.c:1356-1363) happens on a miss, a Protect or a semi-invulnerable foe too.
    """
    fmt = format_of(state)
    side = jnp.asarray(side, _I32)
    # ``target_side`` / ``defender``: the fight command's ``targets.resolve`` (D2-4); omitted
    # (a direct call of one hit), today's singles pair: the foe, and the user for a
    # ``self_defender`` move, else the foe.
    if target_side is None:
        target_side = _foe(side)
    target_side = jnp.asarray(target_side, _I32)
    if defender is None:
        defender = jnp.where(
            self_defender(tables, move, ghost_user(state, side)), side, target_side
        )
    defender = jnp.asarray(defender, _I32)
    user, target = state.active[side], state.active[target_side]
    move = jnp.asarray(move, _I32)
    # A RANGE_USER move (Growth, Withdraw, Rock Polish) lands on its user, so the
    # foe's active having fainted earlier this turn does not stop it.
    landed = (
        connected
        & (hit_index < hit_count(tables, move, outcome, side, state))
        & (
            _alive(state, target_side, target)
            | defender_always(tables, move, ghost_user(state, side))
        )
    )

    # Payback (BtlCmd_CalcPaybackPower, battle_script.c:7198-7209): movePower x2 when the target
    # has acted this turn, as a powerMul of 20 (movePower * powerMul / 10, battle_lib.c:6688).
    payback = (tables.move_effect[move] == _E.DOUBLE_POWER) & (
        tables.move_param[move, 0] == POWER_IF_TARGET_MOVED
    )
    power_mul = jnp.where(payback & jnp.asarray(target_moved, bool), 20, power_mul)
    damage = damage_for_roll(
        tables,
        state,
        side,
        move,
        # The crit draw read with the odds at this hit (battle_lib.c:7105-7132); Fling's CalcCrit
        # ran before its RemoveItem (``flung_view``).
        crit_lands(
            tables, flung_view(state, side, fling), side, move, outcome.crit[side, hit_index]
        )
        & ~crit_blocked(state, side, target_side),
        outcome.roll[side, hit_index],
        power_mul,
        outcome.magnitude[side].astype(_I32),
        outcome.present[side].astype(_I32),
        # Triple Kick's per-hit power, Beat Up's party member (B13-5, B13-10).
        hit=hit_index,
        # Gyro Ball's / Trump Card's power from the action (step._use_move, B13-14, B13-15).
        in_power=in_power,
        # Natural Gift's berry, removed by its script this action (damage.gift_item, B13-37).
        gift=gift,
        # Fling's item, removed by its script this action after CalcDamage (damage.flung_view).
        fling=fling,
        # The defender (targets.resolve): at B = 2 the foe, or the user for a lock reading its
        # stored target (follow-up (ii)).
        def_side=target_side,
    )
    # CalcDamage ran for this hit (a move with a rolled table damage whose script ran: hit 0 hit or
    # miss, ``script_calc``; a later hit when it lands): the Me First counter update of the same
    # script command (battle_script.c:1356-1363, E3-2-3).
    calc = (
        (landed | ((jnp.asarray(hit_index) == 0) & jnp.asarray(script_calc, bool)))
        & (tables.move_category[move] != STATUS)
        & (tables.move_power[move] != 0)
        & ~jnp.isin(tables.move_effect[move], jnp.asarray(UNROLLED_EFFECTS, _I32))
    )
    state = me_first_after_calc(state, side, calc)
    # Counter / Mirror Coat / Metal Burst (``counter_damage``: the amount step._use_move read
    # off the turn): battleCtx->damage, under SYSCTL_IGNORE_TYPE_CHECKS; the chart still flags an
    # immunity (battle_lib.c:7550-7551). No Counter-family move is RANGE_USER in either game
    # (Counter, Mirror Coat, Metal Burst: RANGE_SINGLE_TARGET_SPECIAL; Kaizo's Bide: single
    # target), so CheckTypeChart's RANGE_USER skip (bcp.c:2352) never applies.
    is_counter = tables.move_effect[move] == _E.COUNTER
    counter_type = move_type_of(tables, state, side, move, gift)
    immune = (
        effectiveness(
            tables,
            counter_type,
            state.types[party(fmt, target_side), target],
            identified(tables, state, side, target_side),
        )
        == 0
    ) | ground_immune(tables, state, side, target_side, counter_type)
    # Wonder Guard flags them too (power 1; MOVE_STATUS_WONDER_GUARD, ruling G-18).
    immune = immune | wonder_guarded(tables, state, side, move, counter_type, target_side)
    damage = jnp.where(is_counter, jnp.where(immune, 0, jnp.asarray(counter_damage, _I32)), damage)
    target_hp = state.hp[party(fmt, target_side), target].astype(_I32)
    # False Swipe (bcp.c:3381-3384, inside UpdateHP before the endure checks): a hit that would
    # KO leaves 1 HP; against 1 HP it deals 0, but UpdateHP has run (SYSCTL_MOVE_HIT).
    swipe = (tables.move_effect[move] == _E.FALSE_SWIPE) & (damage > 0) & (damage >= target_hp)
    raw = damage
    damage = jnp.where(swipe, target_hp - 1, damage)
    # Bide is a RANGE_USER move whose unleash hits the foe (bcp.c:2702-2713).
    hits_foe = (
        ~targets_user(tables, move, ghost_user(state, side))
        | (tables.move_effect[move] == _E.BIDE)
        | is_counter
    )
    update_hp = landed & hits_foe & (raw > 0)  # BattleControllerPlayer_UpdateHP runs
    # lastHitByBattler[defender] = the attacker on every UpdateHP with damage, before the
    # Substitute branch (bcp.c:3347, :3357; D2-14): a partner's hit and a Substitute's count, and
    # nothing clears it. Written at B = 4 only. At B = 2 it is the foe, or the user after a lock
    # hit itself (singles follow-up (ii)); its one reader, Bide's release (bcp.c:2700-2705), needs
    # storedDamage, which subscript_bide_start.s zeroes and only the foe's hits during Bide add to
    # (bcp.c:3357, :3407, each also setting lastHitByBattler), so it reads the foe either way.
    # Pursuit's hit on a switcher is not the controller's UpdateHP: subscript_pursuit.s:36 calls
    # BATTLE_SUBSCRIPT_UPDATE_HP, and bcp.c:3357 is the only lastHitByBattler write, so it
    # writes nothing (D3-1 note).
    if fmt.B == 4 and not pursuit_switch:
        state = _set(state, "last_hit_by", target_side, side + 1, update_hp)
    # bcp.c:3359-3378: a substitute takes battleCtx->damage (before False Swipe's cap) and
    # UpdateHP returns early; `real`: the rest of UpdateHP runs on the defender's HP.
    sub_hp = state.substitute_hp[target_side].astype(_I32)
    sub_take = update_hp & (sub_hp != 0)
    to_sub = jnp.where(sub_take, jnp.minimum(raw, sub_hp), 0)
    real = update_hp & ~sub_take
    sash = holds(tables, state, target_side, ItemId.FOCUS_SASH)
    full = target_hp == state.max_hp[party(fmt, target_side), target].astype(_I32)
    # Focus Band (bcp.c:3386-3389, ruling G-1): RandNext() % 100 < param on every UpdateHP that
    # reaches the endure checks, lethal or not (Outcome.focus_band).
    # Endure (G-65): ``if (enduring == 0)`` wraps both the Band roll and the Sash test.
    enduring = jnp.asarray(foe_enduring, bool)
    band = holds(tables, state, target_side, ItemId.FOCUS_BAND)
    rolled = band & (outcome.focus_band[side, hit_index] != 0) & ~enduring
    armed = jnp.asarray(focus_armed) | (real & sash & full & ~enduring) | (real & rolled)
    # :3396-3404: ``enduring || focusItemActivated``; only the item path is named and consumed.
    capped, endured_any = focus_sash_damage(state, target_side, damage, (armed | enduring) & real)
    endured = endured_any & ~enduring
    # battleCtx->damage after the endure cap: what subscript_update_hp.s receives.
    taken = jnp.where(endured_any, target_hp - 1, damage)
    # Secret Power's secondary effect is the terrain's (order.move_effects).
    effect, param = move_effects(tables, state, move)
    move_type = move_type_of(tables, state, side, move, gift)
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
                state.types[party(fmt, target_side), target],
                identified(tables, state, side, target_side),
            )
            > 100
        )
    )
    resisted = (
        real
        & ~jnp.isin(effect, jnp.asarray(RESIST_BERRY_SKIPS, effect.dtype))
        & resist_berry(tables, state, target_side, move_type, super_effective)
    )
    dealt = jnp.where(landed & hits_foe & ~sub_take, capped, 0)
    dealt = resist_berry_damage(state, target_side, taken, resisted, dealt)
    # UpdateHP's subscript_update_hp.s with hpCalcTemp = -battleCtx->damage after the endure
    # cap (bcp.c:3437-3438) and a resist berry's halving: 0 for a 1-HP False Swipe or endure
    # target (B13-17); ``dealt`` has its sign.
    state = hp_update(state, target_side, target, -dealt, real)
    if pursuit_switch:
        state = _set(state, "substitute_hp", target_side, 0, sub_take & (to_sub == sub_hp))
    else:
        state = _set(state, "substitute_hp", target_side, sub_hp - to_sub, sub_take)
        sub_hit = jnp.asarray(sub_hit) | sub_take
    if pursuit_switch:
        # subscript_pursuit.s:85 FAINT_CHECK_DESTINY_BOND runs before TriggerAbilityOnHit (:86-87):
        # a KO'd switcher holding Destiny Bond takes the pursuer down while it has HP
        # (subscript_faint_check_destiny_bond.s:5-7, :11-16), so every contact-ability case
        # then fails on the attacker's HP (battle_lib.c:4215-4363). No reveal, no draw. The
        # engine has no Destiny Bond event (step.EVENT_FLAGS), here or for a FIGHT's take-down.
        ko = (target_hp > 0) & (state.hp[party(fmt, target_side), target] == 0)
        bonded = (
            ko & (state.destiny_bond[target_side] != 0) & (state.hp[party(fmt, side), user] > 0)
        )
        state = _set(state, "hp", (side, user), 0, bonded)
        # The same subscript's :28 TryGrudge (battle_script.c:6743-6763), still before
        # TriggerAbilityOnHit (subscript_pursuit.s:86-87): a KO'd switcher holding Grudge zeroes
        # the pursuer's selected slot while the pursuer has HP (:6751), so a Rough Skin or
        # Aftermath that then faints the pursuer finds the slot zeroed already (E3-B2-9).
        if grudge_slot is not None:
            state = try_grudge(
                state,
                side,
                user,
                grudge_slot,
                target_side,
                state.grudge[target_side] != 0,
                struggling,
                ko,
            )
    # subscript_move_followup_message.s:12-16: "hung on using its {1}" tags the defender's item
    # (BattleMessage_ItemTag, battle_script.c:11736-11739): the AI learns a Focus Band that stays.
    state = reveal_item(state, target_side, target, endured)
    state = consume_item(state, target_side, (endured & sash) | resisted)
    # BattleControllerPlayer_UpdateHP (bcp.c:3407): a Biding target stores battleCtx->damage,
    # before the berry halves the HP change: what it lost, or the unhalved amount after a
    # resist berry (a target that faints is cleared at the faint check, so the HP cap of
    # `dealt` changes nothing there).
    biding = (state.bide_turns[target_side] > 0) & ~sub_take
    stored = jnp.where(resisted, taken, dealt)
    state = _set(state, "bide_damage", target_side, state.bide_damage[target_side] + stored, biding)

    is_status = tables.move_category[move] == STATUS
    # A type-checked status move (Thunder Wave) fails on a 0x matchup.
    type_ok = (param[1] == 0) | (
        effectiveness(
            tables,
            move_type,
            state.types[party(fmt, target_side), target],
            identified(tables, state, side, target_side),
        )
        > 0
    ) & ~ground_immune(tables, state, side, target_side, move_type)
    passes = jnp.where(is_status, (effect != _E.STATUS_FOE) | type_ok, damage > 0)
    chance = read_chance(tables, move)
    h = _Hit(
        side=side,
        target_side=target_side,
        defender=defender,
        user=user,
        target=target,
        effect=effect,
        param=param,
        certain=chance == 0,
        landed=landed,
        effective=landed & passes,
        # The chance is read at the hit, Serene Grace as the attacker's ability is now
        # (battle_lib.c:1552-1556, :1571-1575; outcome.secondary_procs).
        proc=(chance == 0)
        | secondary_procs(tables, state, side, move, outcome.secondary[side, hit_index]),
        # hitDamage: what the substitute took on its path (bcp.c:3363, :3367).
        dealt=jnp.where(sub_take, to_sub, dealt),
        target_moved=jnp.asarray(target_moved, jnp.bool_),
        move=move,
        target_ignorable=ignorable(state, side, target_side),
        contact_proc=outcome.contact_proc[target_side, hit_index] != 0,
        update_hp=real & (taken != 0),
        hit_index=jnp.asarray(hit_index, _I32),
        # Pursuit's switch path sets no flag but skips the on-hit ability all the same (:108).
        sub_hit=sub_hit | (sub_take & pursuit_switch),
        shielded=sub_hit | sub_take | (state.substitute_hp[target_side] != 0),
        fling_item=jnp.asarray(fling, _I32),
        fling_pick=jnp.asarray(fling_pick, _I32),
        coat_slot=jnp.asarray(coat_slot, _I32),
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
        _conversion_2,
        _spite,
        _stealth_rock,
        _trap,
        _taunt,
        _torment,
        _nightmare,
        _worry_seed,
        _gastro_acid,
        _fling,
        _trick_room,
        _skill_swap,
        _role_play,
        _magic_coat,
        _disable,
        _confuse_foe,
        _secondary_confuse,
        _leech_seed,
        _water_sport,
        _mud_sport,
        _aqua_ring,
        _magnet_rise,
        _gravity,
        _mimic,
        _sketch,
        _transform,
        _chatter,
        _ingrain,
        _mean_look,
        _imprison,
        _lock_on,
        _perish_song,
        _yawn,
        _screen,
        _brick_break,
        _defog,
        _foresight,
        _attract,
        _bide,
        _rollout,
        _weather,
        _cure_paralysis,
        _swagger,
        _dragon_dance,
        _tickle,
        _focus_energy,
        _haze,
        _psych_up,
        _stage_swap,
        _captivate,
        _acupressure,
        _power_trick,
        _rest,
        _weather_heal,
        _wish,
        _heal_bell,
        _refresh,
        _psycho_shift,
        _belly_drum,
        _substitute,
        _pain_split,
        _safeguard,
        _mist,
        _lucky_chant,
        _encore,
        _roost,
        _side_layers,
        _destiny_bond,
        _grudge,
        _miracle_eye,
        _uproar,
    ):
        state = fn(tables, state, h, outcome)
    state, pluck_flag = _eat_berry(tables, state, h, outcome, foe_custap)
    state = _knock_off(tables, state, h, foe_custap, foe_claw)
    state = _embargo(tables, state, h, foe_custap, foe_claw)
    state = _struggle_recoil(state, h, jnp.asarray(struggling, jnp.bool_))
    if not pursuit_switch:  # subscript_pursuit.s runs no AfterMoveMessage (F-16)
        from nuzline.engine.hooks import shaymin_land  # hooks imports this module

        # The check reads BTLVAR_DEFENDER, the user itself for a self-targeted move
        # (BattleSystem_Defender, battle_lib.c:1665-1669): the foe's Shaymin is left as it is.
        on_foe = h.landed & ~targets_user(tables, move, ghost_user(state, side))
        state = shaymin_land(tables, state, target_side, on_foe)
    state, sem_proc = _contact_ability(tables, state, h, outcome, move_type)
    if extra_flinch is not False:  # a traced bool: U-turn defers its check (step._use_move)
        state = _kings_rock(tables, state, h._replace(landed=h.landed & extra_flinch), outcome)
    # UpdateHP's shellBellDamageDealt (bcp.c:3427-3431; the substitute's share, :3361, :3366)
    # and subscript_pursuit.s's (:21-29).
    shell = jnp.where(real, jnp.minimum(taken, target_hp), to_sub)
    pursuit_shell = jnp.where(real, jnp.minimum(damage, target_hp), 0)
    return state, MoveHit(
        focus_armed=armed,
        pluck_flag=pluck_flag,
        update_hp=real,
        dealt=dealt,
        damaged=real & (taken != 0),
        shell_bell=shell,
        pursuit_shell_bell=pursuit_shell,
        taken=jnp.where(real, taken, 0),
        effective=h.effective,
        sub_hit=sub_hit,
        # ApplyTypeChart's end (battle_lib.c:2649-2678) clears the bit under
        # SYSCTL_IGNORE_TYPE_CHECKS / _IGNORE_IMMUNITIES (IGNORES_TYPE_CHECKS); Present's heal
        # skips CheckTypeChart (effect_script_0122.s:13, bcp.c:2350-2356).
        super_effective=super_effective
        & ~jnp.isin(effect, jnp.asarray(IGNORES_TYPE_CHECKS, effect.dtype))
        & ~((effect == _E.PRESENT) & (outcome.present[side] == PRESENT_HEAL)),
        trick_room=trick_room_runs(h),
        sem_proc=sem_proc,
    )
