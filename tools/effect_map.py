"""Hand-maintained map from decomp move effects, abilities and items to engine ids.

Anything absent from these maps is ``UNSUPPORTED`` (spec §3 Unsupported
mechanics): ``tools/build_tables.py`` looks every decomp id up here and falls
back to the ``UNSUPPORTED`` member, never to an approximation.

Sources (decomp at the pin in ``tools/clone_decomp.sh``):

* Effect ids are line numbers of ``generated/move_battle_effects.txt``; each key's
  constant name is in ``DECOMP_EFFECT_NAMES`` and every stat entry was read off
  ``res/battle/scripts/effects/effect_script_NNNN.s``.
  ``tests/tools/test_effect_map.py`` re-checks both against the decomp.
* Ability, item and trainer-item keys are the decomp's English names as they appear in
  ``data/intermediate/{abilities,items}.json``.
* The Soundproof and Iron Fist move lists (``sSoundMoves`` and ``sPunchingMoves`` in
  ``src/battle/battle_lib.c``) are not here: they are per-game data in ``move_lists.json``,
  by move id, and ``tools/build_tables.py`` sets ``MoveFlag.SOUND`` / ``PUNCH`` from it.

Per game (spec §7.2, classified in ``nuzline.data.game_constants``):
``TYPE_CHART_STATUS_MOVES`` (Thunder Wave) is checked by name against every game's data at
load. Every other map here is keyed by name or decomp effect id, so it is looked up in each
game's own data: it describes the engine semantics of the vanilla scripts, and a game that
changes those scripts declares it in ``rom_diff.toml`` (Ruling 13).

Deliberately unsupported although close to a supported family:

* ``SPEED_UP``, ``ACC_UP``, ``SP_DEF_UP``, ``SP_ATK_DOWN``, ``SP_DEF_DOWN``,
  ``ACC_UP_2``, ``EVA_UP_2``, ``SP_ATK_DOWN_2``, ``EVA_DOWN_2``, ``ACC_DOWN_2``,
  ``LOWER_EVASION_HIT`` (12, 14, 15, 21, 22, 55, 56, 61, 63, 64, 74): no move uses
  them and their scripts are a bare ``CalcCrit``/``CalcDamage`` hit, not the stat
  change their names suggest.
* ``UNSUPPORTED_MOVES``: single moves of a supported family whose decomp
  behaviour the family's engine code does not model; see the comment there.

Status families reviewed against their decomp subscripts
(``res/battle/scripts/subscripts/subscript_{fall_asleep,poison,burn,paralyze,freeze}.s``):
the only type checks there are the target-type immunities poison vs
Poison/Steel, burn vs Fire and freeze vs Ice (and no freezing in sun), which
``engine/effects.py`` applies. Paralysis and sleep have none in Gen 4. Their ability immunities
are Insomnia and Vital Spirit (alike at every site; Vital Spirit since E1 Batch C, G-47),
Limber and Leaf Guard in sun (implemented in ``engine/effects.py``; Leaf Guard since the E1
Gardenia batch, G-14), Shield Dust against a damaging move's secondary status
(``effects._inflict``), and Immunity, Water Veil and Magma Armor (``AbilityId.UNSUPPORTED``), and
Safeguard and Substitute are unsupported moves, so none of those branches can be
reached silently. Uproar's sleep block (subscript_fall_asleep.s:14-16, :43-45) is modelled
(E2 batch 5, ``effects._inflict``).
"""

from __future__ import annotations

from nuzline.engine.ids import (
    COUNTER_ANY,
    COUNTER_PHYSICAL,
    COUNTER_SPECIAL,
    FLAIL_FALLOFF,
    MINIMIZE_PARAM,
    POWER_IF_TARGET_ASLEEP,
    POWER_IF_TARGET_HALF_HP,
    POWER_IF_TARGET_MOVED,
    POWER_IF_TARGET_PARALYZED,
    POWER_IF_USER_STATUSED,
    RESIST_ANY_NORMAL,
    SNORE_PARAM,
    SPECIES_CHANSEY,
    SPECIES_FARFETCHD,
    STOMP_PARAM,
    AbilityId,
    EffectId,
    Flavor,
    ItemId,
    MoveFlag,
    Stat,
    Status,
    Type,
    Weather,
)

_E = EffectId
_ATK, _DEF, _SPA, _SPD, _SPE, _ACC, _EVA = (
    Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE, Stat.ACC, Stat.EVA
)  # fmt: skip

# decomp effect id -> (constant name, EffectId, params)
_EFFECTS: dict[int, tuple[str, EffectId, tuple[int, int, int]]] = {
    0: ("BATTLE_EFFECT_HIT", _E.PLAIN, (0, 0, 0)),
    1: ("BATTLE_EFFECT_STATUS_SLEEP", _E.STATUS_FOE, (Status.SLEEP, 0, 0)),
    2: ("BATTLE_EFFECT_POISON_HIT", _E.SECONDARY_STATUS, (Status.POISON, 0, 0)),
    # subscript_drain_half_damage_dealt: damage / 2 == damage * 50 // 100
    3: ("BATTLE_EFFECT_RECOVER_HALF_DAMAGE_DEALT", _E.DRAIN, (50, 0, 0)),
    # Dream Eater (E2 batch 9, B9-12): effect_script_0008.s ends with MOVE_STATUS_NO_MORE_WORK
    # ("wasn't affected", :9-16) unless the defender is asleep (:5; its substitute test, :4, is an
    # unsupported move's), else an ON_HIT subscript_dream_eater.s: Divide(hitDamage, 2), then Big
    # Root (Heal Block, :3, is unsupported): DRAIN 50 with move_param[m, 1] = 1.
    8: ("BATTLE_EFFECT_RECOVER_DAMAGE_SLEEP", _E.DRAIN, (50, 1, 0)),
    # Payback, Kaizo's Revenge (B9-12): effect_script_0230.s, BtlCmd_CalcPaybackPower
    # (battle_script.c:7198-7209): movePower x2 once the defender's action is over
    # (Battler_MovedThisTurn's test, battle_lib.c:5365-5368).
    230: (
        "BATTLE_EFFECT_DOUBLE_POWER_IF_MOVING_SECOND",
        _E.DOUBLE_POWER,
        (POWER_IF_TARGET_MOVED, 0, 0),
    ),
    4: ("BATTLE_EFFECT_BURN_HIT", _E.SECONDARY_STATUS, (Status.BURN, 0, 0)),
    5: ("BATTLE_EFFECT_FREEZE_HIT", _E.SECONDARY_STATUS, (Status.FREEZE, 0, 0)),
    6: ("BATTLE_EFFECT_PARALYZE_HIT", _E.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),
    10: ("BATTLE_EFFECT_ATK_UP", _E.STAT_SELF, (_ATK, 1, 0)),
    11: ("BATTLE_EFFECT_DEF_UP", _E.STAT_SELF, (_DEF, 1, 0)),
    13: ("BATTLE_EFFECT_SP_ATK_UP", _E.STAT_SELF, (_SPA, 1, 0)),
    16: ("BATTLE_EFFECT_EVA_UP", _E.STAT_SELF, (_EVA, 1, 0)),
    18: ("BATTLE_EFFECT_ATK_DOWN", _E.STAT_FOE, (_ATK, -1, 0)),
    19: ("BATTLE_EFFECT_DEF_DOWN", _E.STAT_FOE, (_DEF, -1, 0)),
    20: ("BATTLE_EFFECT_SPEED_DOWN", _E.STAT_FOE, (_SPE, -1, 0)),
    23: ("BATTLE_EFFECT_ACC_DOWN", _E.STAT_FOE, (_ACC, -1, 0)),
    24: ("BATTLE_EFFECT_EVA_DOWN", _E.STAT_FOE, (_EVA, -1, 0)),
    29: ("BATTLE_EFFECT_MULTI_HIT", _E.MULTI_HIT, (0, 0, 0)),
    # Double Kick, Bonemerang, Double Hit, Kaizo's DoubleSlap: effect_script_0044.s is
    # SetMultiHit 2, SYSCTL_MULTI_HIT_MOVE (accuracy once), CalcCrit, CalcDamage: two hits.
    44: ("BATTLE_EFFECT_HIT_TWICE", _E.MULTI_HIT, (2, 0, 0)),
    # Recover, Softboiled, Milk Drink, Slack Off, Heal Order (Kaizo: Lunar Dance too):
    # effect_script_0032.s, an ON_HIT direct effect running subscript_recover_half_max_hp.s
    # (maxHP / 2, fails at full HP).
    32: ("BATTLE_EFFECT_RESTORE_HALF_HP", _E.RECOVER, (0, 0, 0)),
    # Fake Out: effect_script_0158.s fails unless fakeOutTurnNumber == totalTurns, else an
    # indirect TO_DEFENDER flinch at the move's chance (subscript_flinch_mon.s).
    158: ("BATTLE_EFFECT_ALWAYS_FLINCH_FIRST_TURN_ONLY", _E.FAKE_OUT, (0, 0, 0)),
    # AncientPower, Silver Wind, Ominous Wind: effect_script_0140.s, an indirect TO_ATTACKER
    # effect at the move's chance running subscript_boost_all_stats.s.
    140: ("BATTLE_EFFECT_RAISE_ALL_STATS_HIT", _E.RAISE_ALL_STATS_HIT, (0, 0, 0)),
    # Curse: effect_script_0109.s; a non-Ghost user runs subscript_curse_normal.s (direct,
    # TO_ATTACKER). The Ghost branch (subscript_curse_ghost.s) is refused by coverage.
    109: ("BATTLE_EFFECT_CURSE", _E.CURSE, (0, 0, 0)),
    # Selfdestruct, Explosion (Kaizo: Memento too): effect_script_0007.s checks Damp (an
    # unsupported ability, so never met), sets the attacker's HP to 0, then CalcCrit,
    # CalcDamage; BattleSystem_CalcMoveDamage halves defenseStat for this effect
    # (battle_lib.c:6943-6945).
    7: ("BATTLE_EFFECT_HALVE_DEFENSE", _E.SELF_DESTRUCT, (0, 0, 0)),
    # Flail, Reversal: effect_script_0099.s runs CalcFlailPower (battle_script.c:4958) before
    # CalcCrit and CalcDamage; the power comes from the user's HP pixels.
    99: ("BATTLE_EFFECT_INCREASE_POWER_WITH_LESS_HP", _E.FLAIL, (0, 0, 0)),
    31: ("BATTLE_EFFECT_FLINCH_HIT", _E.FLINCH, (0, 0, 0)),
    # Swift, Shock Wave, Kaizo's Aurora Beam: effect_script_0017.s is CalcCrit, CalcDamage;
    # the never-miss comes from accuracy 0.
    17: ("BATTLE_EFFECT_BYPASS_ACCURACY", _E.PLAIN, (0, 0, 0)),
    # Toxic: effect_script_0033.s sets a direct TO_DEFENDER badly-poison subscript; its
    # immunities (Poison, Steel, Immunity, Leaf Guard in sun, status, Safeguard, substitute)
    # are can_have_status's, and it skips the type chart like Poison Powder (effect 66).
    33: ("BATTLE_EFFECT_STATUS_BADLY_POISON", _E.STATUS_FOE, (Status.TOXIC, 0, 0)),
    # Pay Day: effect_script_0034.s, a hit plus subscript_pay_day.s (prize money only).
    34: ("BATTLE_EFFECT_INCREASE_PRIZE_MONEY", _E.PLAIN, (0, 0, 0)),
    42: ("BATTLE_EFFECT_BIND_HIT", _E.TRAP, (0, 0, 0)),
    # Gust: effect_script_0149.s hits a Fly / Bounce user (SYSCTL_HIT_DURING_FLY) and doubles the
    # power against MOVE_EFFECT_AIRBORNE (MoveFlag.HIT_AIR, DOUBLE_SEMI: SEMI_HIT_EFFECTS, B10-12).
    149: ("BATTLE_EFFECT_DOUBLE_DAMAGE_FLY_OR_BOUNCE", _E.PLAIN, (0, 0, 0)),
    # Sand Tomb (Kaizo), Whirlpool: effect_script_0261.s is Bind's script (the same
    # BIND_TARGET indirect effect) plus SYSCTL_HIT_DURING_DIVE and a x2 against
    # MOVE_EFFECT_UNDERWATER (SEMI_HIT_EFFECTS, B10-12).
    261: ("BATTLE_EFFECT_WHIRLPOOL", _E.TRAP, (0, 0, 0)),
    43: ("BATTLE_EFFECT_HIGH_CRITICAL", _E.HIGH_CRIT, (0, 0, 0)),
    # subscript_recoil_1_4.s: DivideVarByValue(hitDamage, 4) (E2 batch 9: RECOIL's param 0 is
    # the divisor, B9-4).
    48: ("BATTLE_EFFECT_RECOIL_QUARTER", _E.RECOIL, (4, 0, 0)),
    49: ("BATTLE_EFFECT_STATUS_CONFUSE", _E.CONFUSE_FOE, (0, 0, 0)),
    50: ("BATTLE_EFFECT_ATK_UP_2", _E.STAT_SELF, (_ATK, 2, 0)),
    51: ("BATTLE_EFFECT_DEF_UP_2", _E.STAT_SELF, (_DEF, 2, 0)),
    52: ("BATTLE_EFFECT_SPEED_UP_2", _E.STAT_SELF, (_SPE, 2, 0)),
    53: ("BATTLE_EFFECT_SP_ATK_UP_2", _E.STAT_SELF, (_SPA, 2, 0)),
    54: ("BATTLE_EFFECT_SP_DEF_UP_2", _E.STAT_SELF, (_SPD, 2, 0)),
    58: ("BATTLE_EFFECT_ATK_DOWN_2", _E.STAT_FOE, (_ATK, -2, 0)),
    59: ("BATTLE_EFFECT_DEF_DOWN_2", _E.STAT_FOE, (_DEF, -2, 0)),
    60: ("BATTLE_EFFECT_SPEED_DOWN_2", _E.STAT_FOE, (_SPE, -2, 0)),
    62: ("BATTLE_EFFECT_SP_DEF_DOWN_2", _E.STAT_FOE, (_SPD, -2, 0)),
    66: ("BATTLE_EFFECT_STATUS_POISON", _E.STATUS_FOE, (Status.POISON, 0, 0)),
    67: ("BATTLE_EFFECT_STATUS_PARALYZE", _E.STATUS_FOE, (Status.PARALYSIS, 0, 0)),
    68: ("BATTLE_EFFECT_LOWER_ATTACK_HIT", _E.SECONDARY_STAT_FOE, (_ATK, -1, 0)),
    69: ("BATTLE_EFFECT_LOWER_DEFENSE_HIT", _E.SECONDARY_STAT_FOE, (_DEF, -1, 0)),
    70: ("BATTLE_EFFECT_LOWER_SPEED_HIT", _E.SECONDARY_STAT_FOE, (_SPE, -1, 0)),
    71: ("BATTLE_EFFECT_LOWER_SP_ATK_HIT", _E.SECONDARY_STAT_FOE, (_SPA, -1, 0)),
    72: ("BATTLE_EFFECT_LOWER_SP_DEF_HIT", _E.SECONDARY_STAT_FOE, (_SPD, -1, 0)),
    73: ("BATTLE_EFFECT_LOWER_ACCURACY_HIT", _E.SECONDARY_STAT_FOE, (_ACC, -1, 0)),
    84: ("BATTLE_EFFECT_STATUS_LEECH_SEED", _E.LEECH_SEED, (0, 0, 0)),
    # Quick Attack, Mach Punch: the priority lives in the move's priority column.
    103: ("BATTLE_EFFECT_PRIORITY_1", _E.PLAIN, (0, 0, 0)),
    # Flame Wheel, Sacred Fire: effect_script_0125.s is BURN_HIT's script; the effect id
    # itself lets a frozen user move and thaw (battle_controller_player.c:2481, :2728),
    # flagged by move_param[m, 2] = 1. RECOIL_BURN_HIT (Flare Blitz, 253) adds a third
    # recoil: RECOIL_STATUS (E2 batch 10, below).
    # Foresight, Odor Sleuth: effect_script_0113.s, an on-hit indirect effect
    # (subscript_foresight.s sets VOLATILE_CONDITION_FORESIGHT on the defender).
    # Confusion, Psybeam, Dizzy Punch, ...: effect_script_0076.s, a probabilistic indirect
    # effect (no ON_HIT flag) running subscript_confuse.s.
    76: ("BATTLE_EFFECT_CONFUSE_HIT", _E.SECONDARY_CONFUSE, (0, 0, 0)),
    # Light Screen, Reflect: effect_script_0035.s / _0065.s, an ON_HIT direct effect running
    # subscript_light_screen.s / subscript_reflect.s (BtlCmd_TryLightScreen, script.c:4214).
    35: ("BATTLE_EFFECT_SET_LIGHT_SCREEN", _E.LIGHT_SCREEN, (0, 0, 0)),
    65: ("BATTLE_EFFECT_SET_REFLECT", _E.REFLECT, (0, 0, 0)),
    # Brick Break: effect_script_0186.s, damage plus a direct BREAK_SCREENS effect
    # (BtlCmd_TryBreakScreens, script.c:6550).
    186: ("BATTLE_EFFECT_REMOVE_SCREENS", _E.BRICK_BREAK, (0, 0, 0)),
    # Aqua Ring: effect_script_0251.s sets MOVE_EFFECT_AQUA_RING on the user (fails if set).
    251: ("BATTLE_EFFECT_RESTORE_HP_EVERY_TURN", _E.AQUA_RING, (0, 0, 0)),
    # Mud Sport: effect_script_0201.s sets MOVE_EFFECT_MUD_SPORT on the user.
    201: ("BATTLE_EFFECT_HALVE_ELECTRIC_DAMAGE", _E.MUD_SPORT, (0, 0, 0)),
    # Bug Bite, Pluck: effect_script_0224.s, an ON_HIT indirect effect (subscript_pluck.s).
    224: ("BATTLE_EFFECT_EAT_BERRY", _E.EAT_BERRY, (0, 0, 0)),
    # Disable: effect_script_0086.s, an on-hit indirect effect (subscript_disable_start.s).
    86: ("BATTLE_EFFECT_DISABLE", _E.DISABLE, (0, 0, 0)),
    113: ("BATTLE_EFFECT_FORESIGHT", _E.FORESIGHT, (0, 0, 0)),
    # Night Shade, Seismic Toss: effect_script_0087.s, damage = the attacker's level.
    87: ("BATTLE_EFFECT_LEVEL_DAMAGE_FLAT", _E.LEVEL_DAMAGE, (0, 0, 0)),
    # Dragon Rage, SonicBoom (E2 batch 9, B9-9): effect_script_0041.s / _0130.s set
    # SYSCTL_IGNORE_TYPE_CHECKS and BTLVAR_DAMAGE -40 / -20: Night Shade's family with the
    # amount in param 0 (their BTLVAR_REGULATION_FLAG failure reads battleSys->rulesetMask,
    # battle_script.c:9647-9648, which only a regulated battle sets, field_battle_data_transfer.c:
    # 110, :376-379).
    41: ("BATTLE_EFFECT_40_DAMAGE_FLAT", _E.LEVEL_DAMAGE, (40, 0, 0)),
    130: ("BATTLE_EFFECT_20_DAMAGE_FLAT", _E.LEVEL_DAMAGE, (20, 0, 0)),
    # Hidden Power: effect_script_0135.s CalcHiddenPowerParams (type and power from IVs).
    135: ("BATTLE_EFFECT_RANDOM_POWER_BASED_ON_IVS", _E.HIDDEN_POWER, (0, 0, 0)),
    # Magnitude: the decomp's name is PSYWAVE but effect_script_0126.s is Magnitude's.
    126: ("BATTLE_EFFECT_PSYWAVE", _E.MAGNITUDE, (0, 0, 0)),
    125: ("BATTLE_EFFECT_THAW_AND_BURN_HIT", _E.SECONDARY_STATUS, (Status.BURN, 0, 1)),
    128: ("BATTLE_EFFECT_HIT_BEFORE_SWITCH", _E.PURSUIT, (0, 0, 0)),
    138: ("BATTLE_EFFECT_RAISE_DEF_HIT", _E.SECONDARY_STAT_SELF, (_DEF, 1, 0)),
    139: ("BATTLE_EFFECT_RAISE_ATTACK_HIT", _E.SECONDARY_STAT_SELF, (_ATK, 1, 0)),
    # Defense Curl: effect_script_0156.s sets VOLATILE_CONDITION_DEFENSE_CURL, then Def +1;
    # move_param[m, 2] = 1 sets `curled` (Rollout and Ice Ball double their power).
    156: ("BATTLE_EFFECT_DEF_UP_DOUBLE_ROLLOUT_POWER", _E.STAT_SELF, (_DEF, 1, 1)),
    # Bide: effect_script_0026.s -> subscript_bide_start.s (lock, BIDE_1, storedDamage = 0).
    26: ("BATTLE_EFFECT_BIDE", _E.BIDE, (0, 0, 0)),
    # Rollout, Ice Ball: effect_script_0117.s, CalcRolloutPower (battle_script.c:5661).
    117: ("BATTLE_EFFECT_DOUBLE_POWER_EACH_TURN_LOCK_INTO", _E.ROLLOUT, (0, 0, 0)),
    167: ("BATTLE_EFFECT_STATUS_BURN", _E.STATUS_FOE, (Status.BURN, 0, 0)),
    175: ("BATTLE_EFFECT_TAUNT", _E.TAUNT, (0, 0, 0)),
    210: ("BATTLE_EFFECT_HALVE_FIRE_DAMAGE", _E.WATER_SPORT, (0, 0, 0)),
    # Hammer Arm: certain self Speed drop (effect_script_0218.s, ON_HIT|TO_ATTACKER).
    218: ("BATTLE_EFFECT_SPEED_DOWN_HIT", _E.SECONDARY_STAT_SELF, (_SPE, -1, 0)),
    266: ("BATTLE_EFFECT_STEALTH_ROCK", _E.STEALTH_ROCK, (0, 0, 0)),
    # subscript_recoil_1_2.s: DivideVarByValue(hitDamage, 2).
    269: ("BATTLE_EFFECT_RECOIL_HALF", _E.RECOIL, (2, 0, 0)),
    # Double-Edge, Wood Hammer: effect_script_0198.s (Reckless x1.2, ON_HIT
    # subscript_recoil_1_3.s: DivideVarByValue(hitDamage, 3)) (B9-4).
    198: ("BATTLE_EFFECT_RECOIL_THIRD", _E.RECOIL, (3, 0, 0)),
    # E2 batch 10, B10-1: effect_script_0262.s / _0253.s (Reckless x1.2, then an indirect
    # PROBABILISTIC TO_DEFENDER subscript_recoil_1_3_chance_to_{paralyze,burn}.s): recoil 1/3,
    # then the status at the move's chance; 253 lets a frozen user move and thaws it
    # (battle_controller_player.c:2481, :2730).
    262: ("BATTLE_EFFECT_RECOIL_PARALYZE_HIT", _E.RECOIL_STATUS, (Status.PARALYSIS, 3, 0)),
    253: ("BATTLE_EFFECT_RECOIL_BURN_HIT", _E.RECOIL_STATUS, (Status.BURN, 3, 1)),
    # Jump Kick, Hi Jump Kick (B10-2): effect_script_0045.s (Reckless x1.2, SYSCTL_CRASH_DAMAGE,
    # CalcCrit, CalcDamage); subscript_crash_on_miss.s from subscript_missed.s:98.
    45: ("BATTLE_EFFECT_CRASH_ON_MISS", _E.CRASH, (0, 0, 0)),
    271: ("BATTLE_EFFECT_LOWER_SP_DEF_2_HIT", _E.SECONDARY_STAT_FOE, (_SPD, -2, 0)),
    # effect_script_0275.s -> subscript_paralyze_or_flinch.s: a paralysis roll, then a flinch
    # roll (engine/effects.py ``_fang``).
    275: ("BATTLE_EFFECT_FLINCH_PARALYZE_HIT", _E.FANG_PARALYZE_FLINCH, (Status.PARALYSIS, 0, 0)),
    # E2 batch 9 (B9-2): effect_script_0273.s / _0274.s are _0275.s with
    # subscript_burn_or_flinch.s / subscript_freeze_or_flinch.s, which are
    # subscript_paralyze_or_flinch.s with BATTLE_SUBSCRIPT_BURN / _FREEZE in place of _PARALYZE:
    # the same family with its status in param 0 (Fire Fang; Ice Fang, Kaizo's Avalanche).
    273: ("BATTLE_EFFECT_FLINCH_BURN_HIT", _E.FANG_PARALYZE_FLINCH, (Status.BURN, 0, 0)),
    274: ("BATTLE_EFFECT_FLINCH_FREEZE_HIT", _E.FANG_PARALYZE_FLINCH, (Status.FREEZE, 0, 0)),
    # Thunder (effect_script_0152.s) and Blizzard (_0260.s): a probabilistic TO_DEFENDER
    # paralysis / freeze, as effects 6 / 5; their weather accuracy is by effect id in
    # CheckMoveHitAccuracy / CheckMoveHitOverrides (bcp.c:2915-2917, :3024-3031), carried as
    # MoveFlag.THUNDER / BLIZZARD (tools/build_tables.py); Thunder's SYSCTL_HIT_DURING_FLY is
    # MoveFlag.HIT_AIR (SEMI_HIT_EFFECTS, B10-12).
    152: ("BATTLE_EFFECT_THUNDER", _E.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),
    260: ("BATTLE_EFFECT_BLIZZARD", _E.SECONDARY_STATUS, (Status.FREEZE, 0, 0)),
    # effect_script_0122.s: BtlCmd_Present (battle_script.c:5789-5806) draws the power or the
    # heal path (engine/effects.py ``_present``, damage.PRESENT_POWER).
    122: ("BATTLE_EFFECT_RANDOM_POWER_MAYBE_HEAL", _E.PRESENT, (0, 0, 0)),
    # effect_script_0111.s (Protect, Detect): a direct TO_ATTACKER effect running
    # subscript_protect.s -> BtlCmd_TryProtection (battle_script.c:5149-5190); the block is
    # CheckMoveHitOverrides (battle_controller_player.c:3001-3014; engine/step.py).
    111: ("BATTLE_EFFECT_PROTECT", _E.PROTECT, (0, 0, 0)),
    # effect_script_0116.s (Endure): the same direct TO_ATTACKER subscript_protect.s as Protect
    # (effect_script_0111.s:3); BtlCmd_TryProtection sets turnFlags.enduring (battle_script.c:5174-5177).
    116: ("BATTLE_EFFECT_SURVIVE_WITH_1_HP", _E.ENDURE, (0, 0, 0)),
    # effect_script_0252.s (Magnet Rise): fails on magnetRiseTurns, Levitate or Ingrain, else sets
    # MOVE_EFFECT_MAGNET_RISE and magnetRiseTurns = 5 on the user.
    252: ("BATTLE_EFFECT_GIVE_GROUND_IMMUNITY", _E.MAGNET_RISE, (0, 0, 0)),
    # effect_script_0215.s (Gravity): fails with FIELD_CONDITION_GRAVITY up, else the indirect
    # subscript_gravity_start.s (FIELD_CONDITION_GRAVITY_INIT and the grounding loop).
    215: ("BATTLE_EFFECT_GRAVITY", _E.GRAVITY, (0, 0, 0)),
    # effect_script_0082.s (Mimic): a direct ON_HIT subscript_mimic.s -> BtlCmd_TryMimic
    # (battle_script.c:4476-4523), engine/effects.py ``_mimic`` (G-68).
    82: ("BATTLE_EFFECT_COPY_MOVE_FOR_BATTLE", _E.MIMIC, (0, 0, 0)),
    # effect_script_0057.s (Transform): CHECK_HP | TO_DEFENDER subscript_transform_into_target.s
    # -> BtlCmd_Transform (battle_script.c:5330-5373), engine/effects.py ``_transform`` (G-69).
    57: ("BATTLE_EFFECT_TRANSFORM", _E.TRANSFORM, (0, 0, 0)),
    # effect_script_0267.s (Chatter): CalcCrit / CalcDamage with the indirect ON_HIT
    # subscript_chatter.s (engine/effects.py ``_chatter``, G-70); the move's effect_chance is never
    # read (the ON_HIT branch of BattleSystem_TriggerSecondaryEffect, battle_lib.c:1524-1529).
    267: ("BATTLE_EFFECT_CHATTER", _E.CHATTER, (0, 0, 0)),
    # effect_script_0120.s (Attract): a direct TO_DEFENDER effect running
    # subscript_infatuate.s -> BtlCmd_TryAttract (battle_script.c:5727-5742); the status
    # check is CHECK_STATUS_STATE_ATTRACT (battle_controller_player.c:2669-2692;
    # engine/effects.py ``_attract``, engine/step.py).
    120: ("BATTLE_EFFECT_INFATUATE", _E.ATTRACT, (0, 0, 0)),
    276: ("BATTLE_EFFECT_RAISE_SP_ATK_HIT", _E.SECONDARY_STAT_SELF, (_SPA, 1, 0)),
    # effect_script_0036.s (Tri Attack): Random 2, 3 picks burn, freeze or paralysis as the
    # indirect TO_DEFENDER effect at the move's chance (engine/effects.py ``_tri_attack``,
    # Outcome.tri_attack).
    36: ("BATTLE_EFFECT_TRI_ATTACK", _E.TRI_ATTACK, (0, 0, 0)),
    # effect_script_0173.s (Nature Power): BtlCmd_GetTerrainMove (battle_script.c:8036-8048)
    # then GoToMoveScript into the terrain's move (engine/order.py ``called_move``).
    173: ("BATTLE_EFFECT_NATURE_POWER", _E.NATURE_POWER, (0, 0, 0)),
    # E2 batch 5: the invokers (engine/order.py ``invoked_move``). effect_script_0097.s (Sleep
    # Talk), _0083.s (Metronome), _0242.s (Copycat).
    97: ("BATTLE_EFFECT_USE_RANDOM_LEARNED_MOVE_SLEEP", _E.SLEEP_TALK, (0, 0, 0)),
    83: ("BATTLE_EFFECT_CALL_RANDOM_MOVE", _E.METRONOME, (0, 0, 0)),
    242: ("BATTLE_EFFECT_USE_LAST_USED_MOVE", _E.COPYCAT, (0, 0, 0)),
    # effect_script_0213.s (Camouflage): BtlCmd_TryCamouflage (battle_script.c:7997-8022;
    # engine/effects.py ``_camouflage``).
    213: ("BATTLE_EFFECT_CAMOUFLAGE", _E.CAMOUFLAGE, (0, 0, 0)),
    # effect_script_0028.s (Roar, Whirlwind): ON_HIT | subscript_force_target_to_switch_or_flee
    # -> BtlCmd_TryWhirlwind (battle_script.c:5241-5320), BattleSystem_CanWhirlwind
    # (battle_lib.c:3072-3086) (engine/step.py ``_use_move``; Outcome.roar_level, roar_target).
    28: ("BATTLE_EFFECT_FORCE_SWITCH", _E.ROAR, (0, 0, 0)),
    # Sandstorm, Rain Dance, Sunny Day, Hail: (Weather, blocked by the permanent weather too,
    # 0). effect_script_0115.s, _0136.s, _0137.s fail on FIELD_CONDITION_SANDSTORM / _RAINING /
    # _SUNNY (temporary or permanent, include/constants/battle/condition.h:127-130);
    # effect_script_0164.s fails only on FIELD_CONDITION_HAILING_TEMP, so Hail replaces a
    # permanent hail with a 5-turn one (engine/effects.py ``_weather``).
    115: ("BATTLE_EFFECT_WEATHER_SANDSTORM", _E.WEATHER, (Weather.SANDSTORM, 1, 0)),
    136: ("BATTLE_EFFECT_WEATHER_RAIN", _E.WEATHER, (Weather.RAIN, 1, 0)),
    137: ("BATTLE_EFFECT_WEATHER_SUN", _E.WEATHER, (Weather.SUN, 1, 0)),
    164: ("BATTLE_EFFECT_WEATHER_HAIL", _E.WEATHER, (Weather.HAIL, 0, 0)),
    # E2 batch 4 (engine/ids.py EffectId, rulings B4-n).
    # SmellingSalt: effect_script_0171.s, x2 and an ON_HIT cure against a paralyzed defender
    # (CheckSubstitute first: Substitute is an unsupported move).
    171: (
        "BATTLE_EFFECT_DOUBLE_POWER_AND_CURE_PARALYSIS",
        _E.DOUBLE_POWER,
        (POWER_IF_TARGET_PARALYZED, 1, 0),
    ),
    169: (
        "BATTLE_EFFECT_DOUBLE_POWER_WHEN_STATUSED",
        _E.DOUBLE_POWER,
        (POWER_IF_USER_STATUSED, 0, 0),
    ),
    221: (
        "BATTLE_EFFECT_DOUBLE_POWER_WHEN_BELOW_HALF",
        _E.DOUBLE_POWER,
        (POWER_IF_TARGET_HALF_HP, 0, 0),
    ),
    237: ("BATTLE_EFFECT_INCREASE_POWER_WITH_MORE_HP", _E.HP_POWER, (0, 0, 0)),
    196: ("BATTLE_EFFECT_INCREASE_POWER_WITH_WEIGHT", _E.WEIGHT_POWER, (0, 0, 0)),
    40: ("BATTLE_EFFECT_HALVE_HP", _E.HALVE_HP, (0, 0, 0)),
    38: ("BATTLE_EFFECT_ONE_HIT_KO", _E.OHKO, (0, 0, 0)),
    101: ("BATTLE_EFFECT_LEAVE_WITH_1_HP", _E.FALSE_SWIPE, (0, 0, 0)),
    # Earthquake: effect_script_0147.s hits a Dig user (SYSCTL_HIT_DURING_DIG) and doubles the
    # power against MOVE_EFFECT_UNDERGROUND (SEMI_HIT_EFFECTS, B10-12).
    147: ("BATTLE_EFFECT_DOUBLE_DAMAGE_DIG", _E.PLAIN, (0, 0, 0)),
    # Poison Fang: effect_script_0202.s, an indirect TO_DEFENDER badly-poison at the move's
    # chance (subscript_badly_poison.s).
    202: ("BATTLE_EFFECT_BADLY_POISON_HIT", _E.SECONDARY_STATUS, (Status.TOXIC, 0, 0)),
    # Swagger: subscript_swagger.s:6, :11 (Attack +2 below +6); Flatter (E2 batch 12, B12-2):
    # subscript_flatter.s:6, :11, the same script with Sp. Atk +1.
    118: ("BATTLE_EFFECT_ATK_UP_2_STATUS_CONFUSION", _E.SWAGGER, (_ATK, 2, 0)),
    166: ("BATTLE_EFFECT_SP_ATK_UP_CAUSE_CONFUSION", _E.SWAGGER, (_SPA, 1, 0)),
    # Snore: effect_script_0092.s, an indirect TO_DEFENDER flinch at the move's chance, failing
    # unless the user is asleep (move_param[m, 2] = SNORE_PARAM).
    92: ("BATTLE_EFFECT_DAMAGE_WHILE_ASLEEP", _E.FLINCH, (0, 0, SNORE_PARAM)),
    214: ("BATTLE_EFFECT_HEAL_HALF_REMOVE_FLYING_TYPE", _E.ROOST, (0, 0, 0)),
    225: ("BATTLE_EFFECT_DOUBLE_SPEED_3_TURNS", _E.TAILWIND, (0, 0, 0)),
    112: ("BATTLE_EFFECT_SET_SPIKES", _E.SPIKES, (0, 0, 0)),
    249: ("BATTLE_EFFECT_TOXIC_SPIKES", _E.TOXIC_SPIKES, (0, 0, 0)),
    98: ("BATTLE_EFFECT_KO_MON_THAT_DEFEATED_USER", _E.DESTINY_BOND, (0, 0, 0)),
    197: ("BATTLE_EFFECT_SECRET_POWER", _E.SECRET_POWER, (0, 0, 0)),
    # Counter, Mirror Coat: 2x the physical / special damage taken this turn
    # (battle_script.c:4627, :4680); Metal Burst (Kaizo's Bide too): 15 / 10 of the last
    # damage taken (:7166).
    89: ("BATTLE_EFFECT_COUNTER", _E.COUNTER, (COUNTER_PHYSICAL, 20, 0)),
    144: ("BATTLE_EFFECT_MIRROR_COAT", _E.COUNTER, (COUNTER_SPECIAL, 20, 0)),
    227: ("BATTLE_EFFECT_METAL_BURST", _E.COUNTER, (COUNTER_ANY, 15, 0)),
    # Dragon Dance / Bulk Up (E2 batch 9, B9-10), Calm Mind / Cosmic Power and Defend Order
    # (batch 12, B12-1): subscript_user_{atk_and_speed,atk_and_def,spatk_and_spdef,def_and_spdef}_
    # up_1_stage.s differ only in their two stats (:5-6, :13-16), the first then the second.
    212: ("BATTLE_EFFECT_ATK_SPD_UP", _E.DRAGON_DANCE, (_ATK, _SPE, 0)),
    208: ("BATTLE_EFFECT_ATK_DEF_UP", _E.DRAGON_DANCE, (_ATK, _DEF, 0)),
    211: ("BATTLE_EFFECT_SP_ATK_SP_DEF_UP", _E.DRAGON_DANCE, (_SPA, _SPD, 0)),
    206: ("BATTLE_EFFECT_DEF_SPD_UP", _E.DRAGON_DANCE, (_DEF, _SPD, 0)),
    # E2 batch 12, the stat family (B12-n). Tickle: effect_script_0205.s, a direct CHECK_HP |
    # TO_DEFENDER subscript_target_atk_and_def_down_1_stage.s (Attack, then Defense, -1).
    205: ("BATTLE_EFFECT_ATK_DEF_DOWN", _E.TICKLE, (_ATK, _DEF, 0)),
    # Focus Energy: effect_script_0047.s (subscript_focus_energy.s).
    47: ("BATTLE_EFFECT_CRIT_UP_2", _E.FOCUS_ENERGY, (0, 0, 0)),
    # Haze: effect_script_0025.s (subscript_reset_all_stat_stages.s).
    25: ("BATTLE_EFFECT_RESET_STAT_CHANGES", _E.HAZE, (0, 0, 0)),
    # Psych Up: effect_script_0143.s (BtlCmd_CopyStatStages).
    143: ("BATTLE_EFFECT_COPY_STAT_CHANGES", _E.PSYCH_UP, (0, 0, 0)),
    # Power Swap / Guard Swap: effect_script_0243.s / _0244.s (subscript_exchange_atk_and_spatk_
    # stages.s / _def_and_spdef_stages.s).
    243: ("BATTLE_EFFECT_SWAP_ATK_SP_ATK_STAT_CHANGES", _E.STAGE_SWAP, (_ATK, _SPA, 0)),
    244: ("BATTLE_EFFECT_SWAP_DEF_SP_DEF_STAT_CHANGES", _E.STAGE_SWAP, (_DEF, _SPD, 0)),
    # Captivate: effect_script_0265.s, then a direct TO_DEFENDER Sp. Atk -2 (:10).
    265: ("BATTLE_EFFECT_SP_ATK_DOWN_2_OPPOSITE_GENDER", _E.CAPTIVATE, (_SPA, -2, 0)),
    # Acupressure: effect_script_0226.s (BtlCmd_BoostRandomStatBy2).
    226: ("BATTLE_EFFECT_RANDOM_STAT_UP_2", _E.ACUPRESSURE, (0, 0, 0)),
    # Power Trick: effect_script_0238.s (subscript_user_swap_atk_and_def.s).
    238: ("BATTLE_EFFECT_SWAP_ATK_DEF", _E.POWER_TRICK, (0, 0, 0)),
    # Mist (B12-23): effect_script_0046.s, a direct ON_HIT subscript_mist.s (BtlCmd_TryMist).
    46: ("BATTLE_EFFECT_PREVENT_STAT_REDUCTION", _E.MIST, (0, 0, 0)),
    # E2 batch 12, the healing family (B12-12..). Rest: effect_script_0037.s (subscript_rest.s).
    37: ("BATTLE_EFFECT_REST", _E.REST, (0, 0, 0)),
    # Morning Sun, Synthesis, Moonlight: effect_script_0132.s (WeatherHPRecovery,
    # subscript_present_heal.s).
    132: ("BATTLE_EFFECT_HEAL_HALF_MORE_IN_SUN", _E.WEATHER_HEAL, (0, 0, 0)),
    # Wish: effect_script_0179.s (TryWish).
    179: ("BATTLE_EFFECT_HEAL_IN_3_TURNS", _E.WISH, (0, 0, 0)),
    # Heal Bell, Aromatherapy: effect_script_0102.s (subscript_heal_bell.s).
    102: ("BATTLE_EFFECT_CURE_PARTY_STATUS", _E.HEAL_BELL, (0, 0, 0)),
    # Refresh: effect_script_0193.s (subscript_recover_psn_prz_brn.s).
    193: ("BATTLE_EFFECT_HEAL_STATUS", _E.REFRESH, (0, 0, 0)),
    # Psycho Shift: effect_script_0234.s (CheckCanShareStatus, subscript_give_target_own_status.s).
    234: ("BATTLE_EFFECT_TRANSFER_STATUS", _E.PSYCHO_SHIFT, (0, 0, 0)),
    # Belly Drum: effect_script_0142.s, a direct ON_HIT subscript_belly_drum.s (B9-10).
    142: ("BATTLE_EFFECT_MAX_ATK_LOSE_HALF_MAX_HP", _E.BELLY_DRUM, (0, 0, 0)),
    # Substitute (B10-4): effect_script_0079.s, a direct ON_HIT subscript_try_substitute.s.
    79: ("BATTLE_EFFECT_SET_SUBSTITUTE", _E.SUBSTITUTE, (0, 0, 0)),
    # Endeavor, Pain Split (B10-5): effect_script_0189.s, effect_script_0091.s.
    189: ("BATTLE_EFFECT_SET_HP_EQUAL_TO_USER", _E.ENDEAVOR, (0, 0, 0)),
    91: ("BATTLE_EFFECT_AVERAGE_HP", _E.PAIN_SPLIT, (0, 0, 0)),
    # Knock Off (B10-6): effect_script_0188.s, an indirect ON_HIT subscript_knock_off.s.
    188: ("BATTLE_EFFECT_REMOVE_HELD_ITEM", _E.KNOCK_OFF, (0, 0, 0)),
    # Safeguard (B10-7): effect_script_0124.s, a direct ON_HIT subscript_safeguard_start.s.
    124: ("BATTLE_EFFECT_PREVENT_STATUS", _E.SAFEGUARD, (0, 0, 0)),
    # Focus Punch (B10-8): effect_script_0170.s, CalcCrit / CalcDamage unless it lost its focus.
    170: ("BATTLE_EFFECT_HIT_LAST_WHIFF_IF_HIT", _E.FOCUS_PUNCH, (0, 0, 0)),
    # Encore (B10-9): effect_script_0090.s, a direct ON_HIT subscript_encore_start.s.
    90: ("BATTLE_EFFECT_ENCORE", _E.ENCORE, (0, 0, 0)),
    # Assist (B10-11): effect_script_0180.s, TryAssist then GoToMoveScript (an invoker).
    180: ("BATTLE_EFFECT_USE_RANDOM_ALLY_MOVE", _E.ASSIST, (0, 0, 0)),
    # Minimize: effect_script_0108.s, a direct TO_ATTACKER subscript_minimize.s: the user's
    # MOVE_EFFECT_MINIMIZE, then Evasion +1 (STAT_SELF with move_param[m, 2] = MINIMIZE_PARAM).
    108: ("BATTLE_EFFECT_EVA_UP_2_MINIMIZE", _E.STAT_SELF, (_EVA, 1, MINIMIZE_PARAM)),
    # Stomp: effect_script_0150.s: a flinch at the move's chance, powerMul 20 against a minimized
    # defender (FLINCH with move_param[m, 2] = STOMP_PARAM).
    150: ("BATTLE_EFFECT_FLINCH_MINIMIZE_DOUBLE_HIT", _E.FLINCH, (0, 0, STOMP_PARAM)),
    # E2 batch 5. Uproar: effect_script_0159.s, a hit plus, for a user not in an uproar, an ON_HIT
    # indirect subscript_uproar.s (engine/effects.py ``_uproar``, engine/step.py).
    159: ("BATTLE_EFFECT_UPROAR", _E.UPROAR, (0, 0, 0)),
    # Future Sight, Doom Desire: effect_script_0148.s, TryFutureSight then an ON_HIT message
    # (engine/step.py ``_use_move``, ``_future_sight``).
    148: ("BATTLE_EFFECT_HIT_IN_3_TURNS", _E.FUTURE_SIGHT, (0, 0, 0)),
    # E2 batch 6. U-turn: effect_script_0228.s (ON_HIT | ATTACK_THEN_SWITCH_OUT); Baton Pass:
    # effect_script_0127.s (a direct ON_HIT subscript_baton_pass.s) (engine/step.py ``_use_move``).
    228: ("BATTLE_EFFECT_SWITCH_HIT", _E.U_TURN, (0, 0, 0)),
    127: ("BATTLE_EFFECT_PASS_STATS_AND_STATUS", _E.BATON_PASS, (0, 0, 0)),
    # E2 batch 7. Ingrain: effect_script_0181.s (MOVE_EFFECT_INGRAIN on the user, failing if set);
    # Mean Look / Block / Spider Web: effect_script_0106.s (a direct subscript_mean_look.s);
    # Imprison: effect_script_0192.s (BtlCmd_TryImprison) (engine/effects.py, engine/step.py).
    181: ("BATTLE_EFFECT_GROUND_TRAP_USER_CONTINUOUS_HEAL", _E.INGRAIN, (0, 0, 0)),
    106: ("BATTLE_EFFECT_PREVENT_ESCAPE", _E.MEAN_LOOK, (0, 0, 0)),
    192: ("BATTLE_EFFECT_MAKE_SHARED_MOVES_UNUSEABLE", _E.IMPRISON, (0, 0, 0)),
    # E2 batch 8. Lock-On / Mind Reader: effect_script_0094.s (a direct subscript_lock_on.s).
    94: ("BATTLE_EFFECT_NEXT_ATTACK_ALWAYS_HITS", _E.LOCK_ON, (0, 0, 0)),
    # Perish Song: effect_script_0114.s (a direct ON_HIT subscript_perish_song_start.s).
    114: ("BATTLE_EFFECT_ALL_FAINT_3_TURNS", _E.PERISH_SONG, (0, 0, 0)),
    # Yawn: effect_script_0187.s (a direct subscript_yawn.s).
    187: ("BATTLE_EFFECT_STATUS_SLEEP_NEXT_TURN", _E.YAWN, (0, 0, 0)),
    # E2 batch 9 (B9-5). Surf: effect_script_0257.s hits a Dive user (SYSCTL_HIT_DURING_DIVE) and
    # doubles the power against MOVE_EFFECT_UNDERWATER (SEMI_HIT_EFFECTS, B10-12).
    257: ("BATTLE_EFFECT_DOUBLE_DAMAGE_DIVE", _E.PLAIN, (0, 0, 0)),
    # Follow Me: effect_script_0172.s (BtlCmd_FollowMe, battle_script.c:6261-6270, then an ON_HIT
    # message). Helping Hand: effect_script_0176.s (BtlCmd_TryHelpingHand, :6291-6314).
    172: ("BATTLE_EFFECT_MAKE_GLOBAL_TARGET", _E.FOLLOW_ME, (0, 0, 0)),
    176: ("BATTLE_EFFECT_BOOST_ALLY_POWER_BY_50_PERCENT", _E.HELPING_HAND, (0, 0, 0)),
    # E2 batch 10 family 7, the charge turn (B10-12..): each charge move is its attack turn's
    # effect; MoveFlag.CHARGE and its kin (CHARGE_FLAG_EFFECTS) carry the charge (step._use_move).
    # Razor Wind: effect_script_0039.s, CRITICAL_BOOSTS + 1, CalcCrit, CalcDamage (:17-19).
    39: ("BATTLE_EFFECT_CHARGE_TURN_HIGH_CRIT", _E.PLAIN, (0, 0, 0)),
    # Sky Attack: effect_script_0075.s, CRITICAL_BOOSTS + 1, CalcCrit, CalcDamage, then the
    # indirect TO_DEFENDER flinch at the move's chance (:16-19), as Headbutt's (31).
    75: ("BATTLE_EFFECT_CHARGE_TURN_HIGH_CRIT_FLINCH", _E.FLINCH, (0, 0, 0)),
    # Skull Bash: effect_script_0145.s, CalcCrit, CalcDamage (:17-18); its Defense +1 is the
    # charge's (MoveFlag.CHARGE_DEF_UP).
    145: ("BATTLE_EFFECT_CHARGE_TURN_DEF_UP", _E.PLAIN, (0, 0, 0)),
    # SolarBeam: effect_script_0151.s, CalcCrit, CalcDamage (:22-23); no charge in sun.
    151: ("BATTLE_EFFECT_SKIP_CHARGE_TURN_IN_SUN", _E.PLAIN, (0, 0, 0)),
    # Fly, Dive, Dig: effect_script_0155.s / _0255.s / _0256.s, CalcCrit, CalcDamage (:20-21).
    155: ("BATTLE_EFFECT_FLY", _E.PLAIN, (0, 0, 0)),
    255: ("BATTLE_EFFECT_DIVE", _E.PLAIN, (0, 0, 0)),
    256: ("BATTLE_EFFECT_DIG", _E.PLAIN, (0, 0, 0)),
    # Bounce: effect_script_0263.s, the indirect TO_DEFENDER paralysis at the move's chance, then
    # CalcCrit, CalcDamage (:20-22), as Thunderbolt's (6).
    263: ("BATTLE_EFFECT_BOUNCE", _E.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),
    # Shadow Force: effect_script_0272.s, an ON_HIT subscript_feint.s (:21) and CalcCrit,
    # CalcDamage: the feint only clears the defender's turnFlags.protecting (subscript_feint.s:20),
    # which in singles nothing reads after the defender's own action (it acted, as it protected).
    272: ("BATTLE_EFFECT_SHADOW_FORCE", _E.PLAIN, (0, 0, 0)),
    # Twister: effect_script_0146.s, Headbutt's flinch (:5) with SYSCTL_HIT_DURING_FLY and a x2
    # against MOVE_EFFECT_AIRBORNE (SEMI_HIT_EFFECTS).
    146: ("BATTLE_EFFECT_FLINCH_DOUBLE_DAMAGE_FLY_OR_BOUNCE", _E.FLINCH, (0, 0, 0)),
    # Sky Uppercut: effect_script_0207.s, SYSCTL_HIT_DURING_FLY then CalcCrit, CalcDamage.
    207: ("BATTLE_EFFECT_HIT_FLY", _E.PLAIN, (0, 0, 0)),
    # E2 batch 13. Twineedle (effect_script_0077.s): effect 2's indirect TO_DEFENDER poison (:5)
    # plus SetMultiHit 2, SYSCTL_MULTI_HIT_MOVE (:6): two hits, each its own effect-chance roll
    # (the script reloads per hit, bcp.c:3789; MULTI_HIT_TRIGGER_SECONDARY :3544-3554);
    # MoveFlag.HIT_TWICE (SCRIPT_FLAG_EFFECTS), B13-1.
    77: ("BATTLE_EFFECT_POISON_MULTI_HIT", _E.SECONDARY_STATUS, (Status.POISON, 0, 0)),
    # Blaze Kick (_0200.s), Poison Tail / Cross Poison (_0209.s): BTLVAR_CRITICAL_BOOSTS + 1
    # (:5, MoveFlag.CRIT_UP) then effect 4's burn / effect 2's poison (:6), B13-2.
    200: ("BATTLE_EFFECT_HIGH_CRITICAL_BURN_HIT", _E.SECONDARY_STATUS, (Status.BURN, 0, 0)),
    209: ("BATTLE_EFFECT_HIGH_CRITICAL_POISON_HIT", _E.SECONDARY_STATUS, (Status.POISON, 0, 0)),
    # E2 batch 13, the multi-hit power family. Triple Kick (effect_script_0104.s): SetMultiHit 3,
    # SYSCTL_TRIPLE_KICK (:5), then per hit BTLVAR_MOVE_POWER += 10 (:7), CalcCrit, CalcDamage:
    # hit k at power 10k, each later hit with its own accuracy roll (B13-5..B13-8).
    104: ("BATTLE_EFFECT_HIT_THREE_TIMES", _E.TRIPLE_KICK, (10, 0, 0)),
    # Beat Up (effect_script_0154.s): SYSCTL_IGNORE_IMMUNITIES (:5), CalcCrit, BtlCmd_BeatUp
    # (battle_script.c:6173-6247): a hit per eligible party member (B13-9..B13-11).
    154: ("BATTLE_EFFECT_BEAT_UP", _E.BEAT_UP, (0, 0, 0)),
    # E2 batch 13, single-hit variable power and type (B13-12..). Water Spout / Eruption:
    # effect_script_0190.s, CalcHPFalloffPower (FLAIL kind FLAIL_FALLOFF), B13-13.
    190: ("BATTLE_EFFECT_DECREASE_POWER_WITH_LESS_USER_HP", _E.FLAIL, (FLAIL_FALLOFF, 0, 0)),
    # Weather Ball: effect_script_0203.s, CalcWeatherBallParams (B13-12).
    203: ("BATTLE_EFFECT_CHANGE_TYPE_WITH_WEATHER", _E.WEATHER_BALL, (0, 0, 0)),
    # Wake-Up Slap: effect_script_0217.s, SmellingSalt's script on MON_CONDITION_SLEEP with
    # subscript_heal_target_sleep.s (B13-16).
    217: (
        "BATTLE_EFFECT_DOUBLE_POWER_HEAL_SLEEP",
        _E.DOUBLE_POWER,
        (POWER_IF_TARGET_ASLEEP, 1, 0),
    ),
    # Gyro Ball: effect_script_0219.s, CalcGyroBallPower (B13-14).
    219: ("BATTLE_EFFECT_POWER_BASED_ON_LOW_SPEED", _E.GYRO_BALL, (0, 0, 0)),
    # Assurance: effect_script_0231.s, x2 on BTLVAR_DEFENDER_ASSURANCE_DAMAGE_MASK (B13-17).
    231: ("BATTLE_EFFECT_DOUBLE_POWER_IF_TARGET_HIT", _E.ASSURANCE, (0, 0, 0)),
    # Trump Card: effect_script_0235.s, CalcTrumpCardPower (B13-15).
    235: ("BATTLE_EFFECT_HIGHER_POWER_WHEN_LOW_PP", _E.TRUMP_CARD, (0, 0, 0)),
    # Punishment: effect_script_0245.s, CalcPunishmentPower (B13-15).
    245: ("BATTLE_EFFECT_INCREASE_POWER_WITH_MORE_STAT_UP", _E.PUNISHMENT, (0, 0, 0)),
    # Recycle: effect_script_0184.s, the battler's recycleItem back (B13-19).
    184: ("BATTLE_EFFECT_RECYCLE", _E.RECYCLE, (0, 0, 0)),
    # Natural Gift: effect_script_0222.s, the held berry's power and type, eaten (B13-20).
    222: ("BATTLE_EFFECT_NATURAL_GIFT", _E.NATURAL_GIFT, (0, 0, 0)),
    # Embargo: effect_script_0232.s, subscript_embargo_start.s: embargoTurns 5 (B13-31).
    232: ("BATTLE_EFFECT_PREVENT_ITEM_USE", _E.EMBARGO, (0, 0, 0)),
    # Judgment: effect_script_0268.s, the held plate's type (B13-12).
    268: ("BATTLE_EFFECT_JUDGEMENT", _E.JUDGMENT, (0, 0, 0)),
    # Splash (Kaizo: Thief, power 40): effect_script_0085.s:5 sets MOVE_STATUS_SPLASH only: no
    # CalcCrit, no CalcDamage (E3-1-1).
    85: ("BATTLE_EFFECT_DO_NOTHING", _E.DO_NOTHING, (0, 0, 0)),
    # Teleport: effect_script_0153.s:5, :9-10, MOVE_STATUS_FAILED in a trainer battle (E3-1-4).
    153: ("BATTLE_EFFECT_FLEE_FROM_WILD_BATTLE", _E.FLEE, (0, 0, 0)),
    # Torment: effect_script_0165.s:5 -> subscript_torment_start.s (E3-2-1).
    165: ("BATTLE_EFFECT_TORMENT", _E.TORMENT, (0, 0, 0)),
    # Worry Seed: effect_script_0247.s:5 -> subscript_give_target_insomnia.s (E3-2-2).
    247: ("BATTLE_EFFECT_SET_ABILITY_TO_INSOMNIA", _E.WORRY_SEED, (0, 0, 0)),
    # Me First: effect_script_0241.s -> BtlCmd_TryMeFirst, then GoToMoveScript (E3-2-3).
    241: ("BATTLE_EFFECT_USE_MOVE_FIRST", _E.ME_FIRST, (0, 0, 0)),
    # Trick Room: effect_script_0259.s:5-12 toggles FIELD_CONDITION_TRICK_ROOM (E3-2-4).
    259: ("BATTLE_EFFECT_TRICK_ROOM", _E.TRICK_ROOM, (0, 0, 0)),
    # Skill Swap: effect_script_0191.s:5 -> subscript_exchange_abilities.s (E3-2-5).
    191: ("BATTLE_EFFECT_SWITCH_ABILITIES", _E.SKILL_SWAP, (0, 0, 0)),
    # Gastro Acid: effect_script_0239.s:5 -> subscript_suppress_target_ability.s (E3-2-6).
    239: ("BATTLE_EFFECT_SUPRESS_ABILITY", _E.GASTRO_ACID, (0, 0, 0)),
    # Fling: effect_script_0233.s, the held item's fling power and effect (E3-B2-1).
    233: ("BATTLE_EFFECT_FLING", _E.FLING, (0, 0, 0)),
    # Mirror Move: effect_script_0009.s -> BtlCmd_SetMirrorMove, then GoToMoveScript (E3-B2-2).
    9: ("BATTLE_EFFECT_COPY_MOVE", _E.MIRROR_MOVE, (0, 0, 0)),
    # Conversion 2: effect_script_0093.s -> subscript_conversion_2.s, BtlCmd_TryConversion2 (E3-B2-3).
    93: ("BATTLE_EFFECT_CONVERSION2", _E.CONVERSION_2, (0, 0, 0)),
    # Spite: effect_script_0100.s -> subscript_reduce_target_pp.s, BtlCmd_TrySpite (E3-B2-4).
    100: ("BATTLE_EFFECT_DECREASE_LAST_MOVE_PP", _E.SPITE, (0, 0, 0)),
    # Role Play: effect_script_0178.s -> subscript_copy_ability.s (E3-B2-5).
    178: ("BATTLE_EFFECT_COPY_ABILITY", _E.ROLE_PLAY, (0, 0, 0)),
    # Magic Coat: effect_script_0183.s -> BtlCmd_TrySetMagicCoat (E3-B2-6).
    183: ("BATTLE_EFFECT_APPLY_MAGIC_COAT", _E.MAGIC_COAT, (0, 0, 0)),
    # Lucky Chant: effect_script_0240.s -> subscript_lucky_chant_start.s (E3-B2-7).
    240: ("BATTLE_EFFECT_PREVENT_CRITS", _E.LUCKY_CHANT, (0, 0, 0)),
}

# The Kaizo ROM's changed battle effect scripts that data/kaizo/rom_diff.toml marks ``modelled``
# (K2b Ruling 13): decomp effect id -> (what the Kaizo script is, EffectId, params), used for that
# game in place of ``_EFFECTS`` (tools/build_tables.py; every modelled be_seq member must be
# here and nothing else). B9-4:
# * 48: effect_script_0048.s with MOVE_SUBSCRIPT_PTR_RECOIL_1_2 for _QUARTER_RECOIL, the Reckless
#   power check unchanged: 1/2 recoil (Hyper Beam, Outrage, Superpower, Hydro Cannon, Giga
#   Impact, Draco Meteor).
# * 182: byte-identical to the vanilla effect 48 script: Reckless and 1/4 recoil (Take Down, Close
#   Combat, Brave Bird).
# * 141 (E2 batch 13, B13-3): the unused plain script becomes effect 76's indirect TO_DEFENDER
#   confusion (effect_script_0076.s:5) plus SYSCTL_HIT_DURING_FLY: Hurricane. Its flags are
#   KAIZO_FLAG_EFFECTS'.
# * 104, 88 (E2 batch 13, B13-5): Triple Kick's three hits at k x the move's power.
KAIZO_EFFECTS: dict[int, tuple[str, EffectId, tuple[int, int, int]]] = {
    48: ("Kaizo effect 48: Reckless, 1/2 recoil", _E.RECOIL, (2, 0, 0)),
    182: ("Kaizo effect 182: Reckless, 1/4 recoil", _E.RECOIL, (4, 0, 0)),
    141: ("Kaizo effect 141: confusion hit that hits a Fly user", _E.SECONDARY_CONFUSE, (0, 0, 0)),
    # E2 batch 13 (B13-5): 104 is Triple Kick with GetCurrentMoveData MOVEATTRIBUTE_POWER added
    # per hit in place of 10 (hit k at k x the move's power: Triple Kick, Rock Wrecker); 88, the
    # Psywave script in vanilla, is byte-identical to Kaizo's 104 (Triple Axel, Fury Cutter).
    # Step 0 = the move's own power (ids.EffectId.TRIPLE_KICK). Neither sets a MoveFlag bit.
    104: ("Kaizo effect 104: three hits at 1x, 2x, 3x power", _E.TRIPLE_KICK, (0, 0, 0)),
    88: ("Kaizo effect 88: three hits at 1x, 2x, 3x power", _E.TRIPLE_KICK, (0, 0, 0)),
}
# The MoveFlag bits of a modelled Kaizo script (tools/build_tables.py ``_move_flags``), used for
# that game in place of the vanilla effect-keyed bits of the same id. 141 (B13-3):
# SYSCTL_HIT_DURING_FLY (MoveFlag.HIT_AIR, as Thunder's effect_script_0152.s:6). Its
# MoveFlag.THUNDER (B13-4) comes from the overlay16 compares, ``KAIZO_THUNDER_EFFECTS`` (E1 G-50).
KAIZO_FLAG_EFFECTS: dict[int, int] = {141: MoveFlag.HIT_AIR}

DECOMP_EFFECT_NAMES: dict[int, str] = {k: v[0] for k, v in _EFFECTS.items()}
DECOMP_EFFECT_TO_ID: dict[int, tuple[EffectId, tuple[int, int, int]]] = {
    k: (eid, tuple(int(p) for p in params)) for k, (_, eid, params) in _EFFECTS.items()
}

# SECONDARY_* effects the decomp applies on every hit (MOVE_SIDE_EFFECT_ON_HIT)
# rather than with the move's effect chance; such moves carry chance 0. Any other
# SECONDARY_* move with chance 0 never fires its proc (K2b Ruling 21: Kaizo's Ice Ball
# family): build_tables packs it PLAIN when its params carry nothing beyond the secondary
# (params[2] == 0), else UNSUPPORTED (effect 125's thaws_user), and lists it in the Kaizo
# manifest's no_secondary_moves.
CERTAIN_SECONDARY_EFFECTS: frozenset[int] = frozenset({218})

# Moves forced to UNSUPPORTED although their effect family is supported (none now).
UNSUPPORTED_MOVES: frozenset[str] = frozenset()

# Status moves the decomp runs through the type chart: status moves skip it (power
# 0) except Thunder Wave, which src/battle/battle_controller_player.c
# BattleControllerPlayer_CheckTypeChart (`|| battleCtx->moveCur == MOVE_THUNDER_WAVE`)
# sends through it, so Ground types are immune. build_tables sets move_param[m, 1] = 1
# for them and the engine fails the move on a 0x matchup (engine/effects.py). Glare
# (same effect 67) is not special-cased there, so it hits Ghost types in Platinum.
TYPE_CHART_STATUS_MOVES: frozenset[str] = frozenset({"Thunder Wave"})

# Move ids the battle code compares against (code constants, not move data: the Kaizo ROM
# leaves BtlCmd_TrySleepTalk, Move_IsInvoker and Move_CanBeMetronomed unchanged, rom_diff.md,
# so these ids hold in both games whatever a slot is named; Kaizo renamed Snatch, Covet, Trick
# and Switcheroo's slots, which stay excluded). tools/build_tables.py turns them into
# ``MoveFlag.NO_SLEEP_TALK`` / ``NO_METRONOME``.
# Move_IsInvoker (battle_lib.c:3032-3045): MOVE_NONE, Sleep Talk, Copycat, Assist, Me First,
# Mirror Move, Metronome.
INVOKER_MOVES: tuple[int, ...] = (0, 214, 383, 274, 382, 119, 118)
# BtlCmd_TrySleepTalk's own list (battle_script.c:4918-4922): Focus Punch, Uproar, Chatter.
SLEEP_TALK_EXCLUDED_MOVES: tuple[int, ...] = (264, 253, 448)
# The weather-accuracy effects of CheckMoveHitAccuracy / CheckMoveHitOverrides (bcp.c:2915-2917,
# :3024-3031): decomp effect id -> its MoveFlag (tools/build_tables.py ``_move_flags``).
THUNDER_EFFECT = 152  # BATTLE_EFFECT_THUNDER
BLIZZARD_EFFECT = 260  # BATTLE_EFFECT_BLIZZARD
# The Kaizo ROM's overlay16 change (E1 Batch C, ruling G-50; docs/kaizo/rom_diff.md,
# data/kaizo/rom_diff.toml): CheckMoveHitAccuracy's sun branch (bcp.c:2915) and
# CheckMoveHitOverrides' rain branch (:3025) compare the move's effect with 152 or 141 (each
# Thunder compare gains a `beq` past a `cmp r0, #0x8d`, rom:overlay16@0x0224F068 and
# @0x0224F384; the Blizzard compare, 260, is unchanged): a Kaizo effect-141 move (Hurricane)
# carries MoveFlag.THUNDER.
KAIZO_THUNDER_EFFECTS: frozenset[int] = frozenset({141})
# Move_IsMultiTurn (battle_lib.c:2975-2992), by decomp effect id: BIDE, CHARGE_TURN_HIGH_CRIT,
# CHARGE_TURN_HIGH_CRIT_FLINCH, CHARGE_TURN_DEF_UP, SKIP_CHARGE_TURN_IN_SUN, FLY, DIVE, DIG,
# BOUNCE, SHADOW_FORCE (generated/move_battle_effects.txt).
MULTI_TURN_EFFECTS: tuple[int, ...] = (26, 39, 75, 145, 151, 155, 255, 256, 263, 272)
# E2 batch 10 family 7 (B10-12..): decomp effect id -> the MoveFlag bits of its charge
# (tools/build_tables.py ``_move_flags``; ids.MoveFlag cites each script).
CHARGE_FLAG_EFFECTS: dict[int, int] = {
    39: MoveFlag.CHARGE | MoveFlag.CRIT_UP,
    75: MoveFlag.CHARGE | MoveFlag.CRIT_UP,
    145: MoveFlag.CHARGE | MoveFlag.CHARGE_DEF_UP,
    151: MoveFlag.CHARGE | MoveFlag.SUN_SKIP,
    155: MoveFlag.CHARGE | MoveFlag.SEMI_AIR,
    263: MoveFlag.CHARGE | MoveFlag.SEMI_AIR,
    256: MoveFlag.CHARGE | MoveFlag.SEMI_DIG,
    255: MoveFlag.CHARGE | MoveFlag.SEMI_DIVE,
    272: MoveFlag.CHARGE | MoveFlag.SEMI_SHADOW,
}
# The scripts that set SYSCTL_HIT_DURING_* (and BTLVAR_POWER_MULTI 20 against that bit): decomp
# effect id -> MoveFlag bits. effect_script_0146.s:6-9, _0149.s:5-8 (Fly); _0152.s:6, _0207.s:5
# (Fly, no x2); _0126.s:5-8, _0147.s:5-8 (Dig); _0257.s:5-8, _0261.s:5-8 (Dive); _0148.s:7,
# _0168.s:11, _0176.s:9 (all four).
_ALL_HIT = MoveFlag.HIT_AIR | MoveFlag.HIT_DIG | MoveFlag.HIT_DIVE | MoveFlag.HIT_SHADOW
SEMI_HIT_EFFECTS: dict[int, int] = {
    146: MoveFlag.HIT_AIR | MoveFlag.DOUBLE_SEMI,
    149: MoveFlag.HIT_AIR | MoveFlag.DOUBLE_SEMI,
    152: MoveFlag.HIT_AIR,
    207: MoveFlag.HIT_AIR,
    126: MoveFlag.HIT_DIG | MoveFlag.DOUBLE_SEMI,
    147: MoveFlag.HIT_DIG | MoveFlag.DOUBLE_SEMI,
    257: MoveFlag.HIT_DIVE | MoveFlag.DOUBLE_SEMI,
    261: MoveFlag.HIT_DIVE | MoveFlag.DOUBLE_SEMI,
    148: _ALL_HIT,
    168: _ALL_HIT,
    176: _ALL_HIT,
}
# E2 batch 13: bits of a script that adds a command to another family's effect, by decomp effect
# id. Twineedle's SetMultiHit 2 (effect_script_0077.s:6): HIT_TWICE (B13-1); Blaze Kick's and
# Poison Tail's BTLVAR_CRITICAL_BOOSTS + 1 (_0200.s:5, _0209.s:5): CRIT_UP (B13-2).
SCRIPT_FLAG_EFFECTS: dict[int, int] = {
    77: MoveFlag.HIT_TWICE,
    200: MoveFlag.CRIT_UP,
    209: MoveFlag.CRIT_UP,
}
# sCannotMetronomeMoves up to FORBIDDEN_BY_METRONOME_DELIM (battle_lib.c:7152-7178):
# Metronome, Struggle, Sketch, Mimic, Chatter, Sleep Talk, Assist, Mirror Move, Counter, Mirror
# Coat, Protect, Detect, Endure, Destiny Bond, Thief, Follow Me, Snatch, Helping Hand, Covet,
# Trick, Focus Punch, Feint, Copycat, Me First, Switcheroo.
METRONOME_EXCLUDED_MOVES: tuple[int, ...] = (
    118, 165, 166, 102, 448, 214, 274, 119, 68, 243, 182, 197, 203, 194, 168, 266, 289, 270,
    343, 271, 264, 364, 383, 382, 415,
)  # fmt: skip

DECOMP_ABILITY_TO_ID: dict[str, AbilityId] = {
    "-": AbilityId.NONE,  # ABILITY_NONE (id 0): a species' empty second ability slot
    "Sturdy": AbilityId.STURDY,
    "Rock Head": AbilityId.ROCK_HEAD,
    "Mold Breaker": AbilityId.MOLD_BREAKER,
    "Blaze": AbilityId.BLAZE,
    "Torrent": AbilityId.TORRENT,
    "Overgrow": AbilityId.OVERGROW,
    "Intimidate": AbilityId.INTIMIDATE,
    "Rivalry": AbilityId.RIVALRY,
    "Keen Eye": AbilityId.KEEN_EYE,
    "Simple": AbilityId.SIMPLE,
    "Unaware": AbilityId.UNAWARE,
    "Natural Cure": AbilityId.NATURAL_CURE,
    "Poison Point": AbilityId.POISON_POINT,
    "Inner Focus": AbilityId.INNER_FOCUS,
    "Static": AbilityId.STATIC,
    "Guts": AbilityId.GUTS,
    "Steadfast": AbilityId.STEADFAST,
    "Insomnia": AbilityId.INSOMNIA,
    "Cloud Nine": AbilityId.CLOUD_NINE,
    "Compoundeyes": AbilityId.COMPOUNDEYES,
    "Shed Skin": AbilityId.SHED_SKIN,
    "Sand Veil": AbilityId.SAND_VEIL,
    "Snow Cloak": AbilityId.SNOW_CLOAK,
    "Swift Swim": AbilityId.SWIFT_SWIM,
    "Rough Skin": AbilityId.ROUGH_SKIN,
    "Flame Body": AbilityId.FLAME_BODY,
    "Limber": AbilityId.LIMBER,
    "Adaptability": AbilityId.ADAPTABILITY,
    "Solid Rock": AbilityId.SOLID_ROCK,
    # BattleSystem_TriggerAbilityOnHit (battle_lib.c:4339-4355); Oblivious, which would block
    # it (subscript_infatuate.s:5), stays unsupported (coverage refuses it).
    "Cute Charm": AbilityId.CUTE_CHARM,
    # Halves the pinch berries' and Custap Berry's param (battle_lib.c:4774-4776, :1289-1291):
    # the only decomp reads of ABILITY_GLUTTONY.
    "Gluttony": AbilityId.GLUTTONY,
    # E2 batch 3 (engine/ids.py AbilityId, rulings B3-n).
    "Reckless": AbilityId.RECKLESS,
    "Technician": AbilityId.TECHNICIAN,
    "Truant": AbilityId.TRUANT,
    "Dry Skin": AbilityId.DRY_SKIN,
    "Lightningrod": AbilityId.LIGHTNINGROD,
    "No Guard": AbilityId.NO_GUARD,
    "Serene Grace": AbilityId.SERENE_GRACE,
    "Drought": AbilityId.DROUGHT,
    "Honey Gather": AbilityId.HONEY_GATHER,
    "Levitate": AbilityId.LEVITATE,
    "Quick Feet": AbilityId.QUICK_FEET,
    "Scrappy": AbilityId.SCRAPPY,
    "Shell Armor": AbilityId.SHELL_ARMOR,
    "Battle Armor": AbilityId.BATTLE_ARMOR,
    "Shield Dust": AbilityId.SHIELD_DUST,
    "Solar Power": AbilityId.SOLAR_POWER,
    "Super Luck": AbilityId.SUPER_LUCK,
    "Tangled Feet": AbilityId.TANGLED_FEET,
    "Thick Fat": AbilityId.THICK_FAT,
    "Tinted Lens": AbilityId.TINTED_LENS,
    "Trace": AbilityId.TRACE,
    "Drizzle": AbilityId.DRIZZLE,
    "Sand Stream": AbilityId.SAND_STREAM,
    "Snow Warning": AbilityId.SNOW_WARNING,
    # E2 batch 4.
    "Filter": AbilityId.FILTER,
    "Magic Guard": AbilityId.MAGIC_GUARD,
    "Sniper": AbilityId.SNIPER,
    # E2 batch 9.
    "Chlorophyll": AbilityId.CHLOROPHYLL,
    "Huge Power": AbilityId.HUGE_POWER,
    "Pure Power": AbilityId.PURE_POWER,
    "Swarm": AbilityId.SWARM,
    "Unburden": AbilityId.UNBURDEN,
    "Flower Gift": AbilityId.FLOWER_GIFT,
    "Volt Absorb": AbilityId.VOLT_ABSORB,
    "Water Absorb": AbilityId.WATER_ABSORB,
    "Rain Dish": AbilityId.RAIN_DISH,
    "Clear Body": AbilityId.CLEAR_BODY,
    "Hyper Cutter": AbilityId.HYPER_CUTTER,
    "Aftermath": AbilityId.AFTERMATH,
    "Poison Heal": AbilityId.POISON_HEAL,
    "Bad Dreams": AbilityId.BAD_DREAMS,
    "Arena Trap": AbilityId.ARENA_TRAP,
    "Shadow Tag": AbilityId.SHADOW_TAG,
    "Magnet Pull": AbilityId.MAGNET_PULL,
    "Pressure": AbilityId.PRESSURE,
    "Leaf Guard": AbilityId.LEAF_GUARD,
    "Synchronize": AbilityId.SYNCHRONIZE,
    "Wonder Guard": AbilityId.WONDER_GUARD,
    "Download": AbilityId.DOWNLOAD,
    "Ice Body": AbilityId.ICE_BODY,
    "White Smoke": AbilityId.WHITE_SMOKE,
    "Vital Spirit": AbilityId.VITAL_SPIRIT,
    "Speed Boost": AbilityId.SPEED_BOOST,
    "Iron Fist": AbilityId.IRON_FIST,
    "Plus": AbilityId.PLUS,
    "Minus": AbilityId.MINUS,
    "Klutz": AbilityId.KLUTZ,
}

DECOMP_ITEM_TO_ID: dict[str, ItemId] = {
    "None": ItemId.NONE,  # ITEM_NONE (id 0)
    "Oran Berry": ItemId.ORAN_BERRY,  # HOLD_EFFECT_HP_RESTORE, param 10 HP
    "Sitrus Berry": ItemId.SITRUS_BERRY,  # HOLD_EFFECT_HP_PCT_RESTORE, param 25 %
    "Focus Sash": ItemId.FOCUS_SASH,  # HOLD_EFFECT_ENDURE
    "Big Root": ItemId.BIG_ROOT,  # HOLD_EFFECT_LEECH_BOOST, param 30 %
    "Berry Juice": ItemId.BERRY_JUICE,  # HOLD_EFFECT_HP_RESTORE, param 20 HP; PLUCK_EFFECT_NONE
    "Chesto Berry": ItemId.CHESTO_BERRY,  # HOLD_EFFECT_SLP_RESTORE
    "Pecha Berry": ItemId.PECHA_BERRY,  # HOLD_EFFECT_PSN_RESTORE
    "Lum Berry": ItemId.LUM_BERRY,  # HOLD_EFFECT_STATUS_RESTORE
    "White Herb": ItemId.WHITE_HERB,  # HOLD_EFFECT_STATDOWN_RESTORE
    "Lagging Tail": ItemId.LAGGING_TAIL,  # HOLD_EFFECT_PRIORITY_DOWN
    "Quick Claw": ItemId.QUICK_CLAW,  # HOLD_EFFECT_SOMETIMES_PRIORITY, param 20
    # The resist berries (subscript_type_resist_berry.s), each name's hold effect read off
    # data/intermediate/items.json (tests/tools/test_effect_map.py re-checks them).
    "Occa Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_FIRE (19)
    "Passho Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_WATER (20)
    "Wacan Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_ELECTRIC (21)
    "Rindo Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_GRASS (22)
    "Yache Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_ICE (23)
    "Chople Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_FIGHT (24)
    "Kebia Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_POISON (25)
    "Shuca Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_GROUND (26)
    "Coba Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_FLYING (27)
    "Payapa Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_PSYCHIC (28)
    "Tanga Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_BUG (29)
    "Charti Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_ROCK (30)
    "Kasib Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_GHOST (31)
    "Haban Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_DRAGON (32)
    "Colbur Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_DARK (33)
    "Babiri Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_STEEL (34)
    "Chilan Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_NORMAL (35)
    # Berries are mapped by name: Pluck / Bug Bite read an item's pluck effect
    # (res/items/data/*.json ``pluckEffect``) and Item_IsBerry, not its hold effect.
    "Cheri Berry": ItemId.CHERI_BERRY,  # HOLD_EFFECT_PRZ_RESTORE (5), PLUCK_EFFECT_PRZ_RESTORE
    "Rawst Berry": ItemId.RAWST_BERRY,  # HOLD_EFFECT_BRN_RESTORE (8), PLUCK_EFFECT_BRN_RESTORE
    "Aspear Berry": ItemId.ASPEAR_BERRY,  # HOLD_EFFECT_FRZ_RESTORE (9), PLUCK_EFFECT_FRZ_RESTORE
    "Persim Berry": ItemId.PERSIM_BERRY,  # HOLD_EFFECT_CONFUSE_RESTORE (11), _CNF_RESTORE
    # The stat pinch berries (PINCH_STAT_HOLD_EFFECTS), PLUCK_EFFECT_ATK_UP .. _SPDEF_UP.
    "Liechi Berry": ItemId.PINCH_BERRY,  # HOLD_EFFECT_PINCH_ATK_UP (36), param 4
    "Ganlon Berry": ItemId.PINCH_BERRY,  # HOLD_EFFECT_PINCH_DEF_UP (37), param 4
    "Salac Berry": ItemId.PINCH_BERRY,  # HOLD_EFFECT_PINCH_SPEED_UP (38), param 4
    "Petaya Berry": ItemId.PINCH_BERRY,  # HOLD_EFFECT_PINCH_SPATK_UP (39), param 4
    "Apicot Berry": ItemId.PINCH_BERRY,  # HOLD_EFFECT_PINCH_SPDEF_UP (40), param 4
    "Custap Berry": ItemId.CUSTAP_BERRY,  # HOLD_EFFECT_PINCH_PRIORITY (45), PLUCK_EFFECT_NONE
    # E1 Batch D (G-54 .. G-59).
    "Leppa Berry": ItemId.LEPPA_BERRY,  # HOLD_EFFECT_PP_RESTORE (10), PLUCK_EFFECT_PP_RESTORE
    "Enigma Berry": ItemId.ENIGMA_BERRY,  # HOLD_EFFECT_HP_RESTORE_SE (43), PLUCK_EFFECT_NONE
    "Micle Berry": ItemId.MICLE_BERRY,  # HOLD_EFFECT_PINCH_ACC_UP (44), PLUCK_EFFECT_TEMP_ACC_UP
    "Jaboca Berry": ItemId.JABOCA_BERRY,  # HOLD_EFFECT_RECOIL_PHYSICAL (46), PLUCK_EFFECT_NONE
    "Rowap Berry": ItemId.ROWAP_BERRY,  # HOLD_EFFECT_RECOIL_SPECIAL (47), PLUCK_EFFECT_NONE
}

# The stat pinch berries' hold effects (BattleSystem_TriggerHeldItem, battle_lib.c:4773-4836,
# and ..._OnStatus, :5196-5259): decomp hold effect -> (name, the Stat it raises by 1).
# build_tables fills a PINCH_BERRY item's item_param[i, 1] from here; the same stats are
# BtlCmd_TryPluck's PLUCK_EFFECT_ATK_UP .. _SPDEF_UP (:5753-5796).
PINCH_STAT_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    36: ("HOLD_EFFECT_PINCH_ATK_UP", Stat.ATK),
    37: ("HOLD_EFFECT_PINCH_DEF_UP", Stat.DEF),
    38: ("HOLD_EFFECT_PINCH_SPEED_UP", Stat.SPE),
    39: ("HOLD_EFFECT_PINCH_SPATK_UP", Stat.SPA),
    40: ("HOLD_EFFECT_PINCH_SPDEF_UP", Stat.SPD),
}
# The berries whose trigger divides maxHP by their param (the stat pinch berries, Custap, Micle:
# battle_lib.c:4778, :1294, :4902): a param 0 (Kaizo's Ganlon and Apicot Berries), or 1 halved by
# Gluttony, divides by zero, and the ROM's ``_u32_div_f`` returns the dividend then
# (``order.pinch_limit``, ruling G-53), so every param is supported.
PINCH_DIVISOR_ITEMS = (ItemId.PINCH_BERRY, ItemId.CUSTAP_BERRY)
# Every other item that divides by its hold param, and the smallest param that never divides by
# zero: Sticky Barb's end-of-turn ``Divide(maxHP, param)`` (battle_lib.c:5330-5336) and Shell
# Bell's ``Divide(dealt, param)`` (bcp.c:4736-4752, battle_lib.c:7345-7355) (1). build_tables
# maps an item below its minimum to UNSUPPORTED.
DIVISOR_PARAM_MIN: dict[ItemId, int] = {
    ItemId.STICKY_BARB: 1,
    ItemId.SHELL_BELL: 1,
    # BattleSystem_Divide(maxHP, itemPower) (battle_lib.c:4700 etc., :5680 etc.), ruling G-4.
    ItemId.FLAVOR_BERRY: 1,
    # BattleSystem_Divide(maxHP, itemPower) (battle_lib.c:5407, :5416, :5423), Batch D.
    ItemId.JABOCA_BERRY: 1,
    ItemId.ROWAP_BERRY: 1,
    ItemId.ENIGMA_BERRY: 1,
}

# sTypeBoostingItems (src/battle/battle_lib.c:6514-6549), applied at :6716-6722: decomp hold
# effect (generated/item_hold_effects.txt line N+1 is id N) -> (name, the type it boosts).
# build_tables fills a TYPE_BOOST item's item_param[i, 1] from here.
# tests/tools/test_effect_map.py reads the table back from the decomp.
TYPE_BOOST_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    57: ("HOLD_EFFECT_STRENGTHEN_BUG", Type.BUG),
    68: ("HOLD_EFFECT_STRENGTHEN_STEEL", Type.STEEL),
    72: ("HOLD_EFFECT_STRENGTHEN_GROUND", Type.GROUND),
    73: ("HOLD_EFFECT_STRENGTHEN_ROCK", Type.ROCK),
    74: ("HOLD_EFFECT_STRENGTHEN_GRASS", Type.GRASS),
    75: ("HOLD_EFFECT_STRENGTHEN_DARK", Type.DARK),
    76: ("HOLD_EFFECT_STRENGTHEN_FIGHT", Type.FIGHTING),
    77: ("HOLD_EFFECT_STRENGTHEN_ELECTRIC", Type.ELECTRIC),
    78: ("HOLD_EFFECT_STRENGTHEN_WATER", Type.WATER),
    79: ("HOLD_EFFECT_STRENGTHEN_FLYING", Type.FLYING),
    80: ("HOLD_EFFECT_STRENGTHEN_POISON", Type.POISON),
    81: ("HOLD_EFFECT_STRENGTHEN_ICE", Type.ICE),
    82: ("HOLD_EFFECT_STRENGTHEN_GHOST", Type.GHOST),
    83: ("HOLD_EFFECT_STRENGTHEN_PSYCHIC", Type.PSYCHIC),
    84: ("HOLD_EFFECT_STRENGTHEN_FIRE", Type.FIRE),
    85: ("HOLD_EFFECT_STRENGTHEN_DRAGON", Type.DRAGON),
    86: ("HOLD_EFFECT_STRENGTHEN_NORMAL", Type.NORMAL),
    126: ("HOLD_EFFECT_ARCEUS_FIRE", Type.FIRE),
    127: ("HOLD_EFFECT_ARCEUS_WATER", Type.WATER),
    128: ("HOLD_EFFECT_ARCEUS_ELECTRIC", Type.ELECTRIC),
    129: ("HOLD_EFFECT_ARCEUS_GRASS", Type.GRASS),
    130: ("HOLD_EFFECT_ARCEUS_ICE", Type.ICE),
    131: ("HOLD_EFFECT_ARCEUS_FIGHTING", Type.FIGHTING),
    132: ("HOLD_EFFECT_ARCEUS_POISON", Type.POISON),
    133: ("HOLD_EFFECT_ARCEUS_GROUND", Type.GROUND),
    134: ("HOLD_EFFECT_ARCEUS_FLYING", Type.FLYING),
    135: ("HOLD_EFFECT_ARCEUS_PSYCHIC", Type.PSYCHIC),
    136: ("HOLD_EFFECT_ARCEUS_BUG", Type.BUG),
    137: ("HOLD_EFFECT_ARCEUS_ROCK", Type.ROCK),
    138: ("HOLD_EFFECT_ARCEUS_GHOST", Type.GHOST),
    139: ("HOLD_EFFECT_ARCEUS_DRAGON", Type.DRAGON),
    140: ("HOLD_EFFECT_ARCEUS_DARK", Type.DARK),
    141: ("HOLD_EFFECT_ARCEUS_STEEL", Type.STEEL),
}

# Held items mapped by their decomp hold effect, for every item DECOMP_ITEM_TO_ID does not
# name: the effect is the hold effect's, with each item's own hold param (item_param[i, 0]).
# The item block of BattleSystem_CalcMoveDamage (battle_lib.c:6716-6783), Choice Scarf's
# Speed (BattleSystem_CompareBattlerSpeed, :1248), Expert Belt (BattleSystem_ApplyTypeChart,
# :2663) and Life Orb (BattleScript_CalcMoveDamage, battle_script.c:1346; recoil
# battle_controller_player.c:4758). tests/tools/test_effect_map.py checks the names.
HOLD_EFFECT_TO_ID: dict[int, tuple[str, ItemId]] = {
    **{h: (name, ItemId.TYPE_BOOST) for h, (name, _) in TYPE_BOOST_HOLD_EFFECTS.items()},
    2: ("HOLD_EFFECT_GIRATINA_BOOST", ItemId.GRISEOUS_ORB),
    3: ("HOLD_EFFECT_DIALGA_BOOST", ItemId.ADAMANT_ORB),
    4: ("HOLD_EFFECT_PALKIA_BOOST", ItemId.LUSTROUS_ORB),
    55: ("HOLD_EFFECT_CHOICE_ATK", ItemId.CHOICE_BAND),
    60: ("HOLD_EFFECT_LATI_SPECIAL", ItemId.SOUL_DEW),
    61: ("HOLD_EFFECT_CLAMPERL_SPATK", ItemId.DEEP_SEA_TOOTH),
    62: ("HOLD_EFFECT_CLAMPERL_SPDEF", ItemId.DEEP_SEA_SCALE),
    71: ("HOLD_EFFECT_PIKA_SPATK_UP", ItemId.LIGHT_BALL),
    90: ("HOLD_EFFECT_DITTO_DEF_UP", ItemId.METAL_POWDER),
    91: ("HOLD_EFFECT_CUBONE_ATK_UP", ItemId.THICK_CLUB),
    94: ("HOLD_EFFECT_POWER_UP_PHYS", ItemId.MUSCLE_BAND),
    95: ("HOLD_EFFECT_POWER_UP_SPEC", ItemId.WISE_GLASSES),
    96: ("HOLD_EFFECT_POWER_UP_SE", ItemId.EXPERT_BELT),
    98: ("HOLD_EFFECT_HP_DRAIN_ON_ATK", ItemId.LIFE_ORB),
    115: ("HOLD_EFFECT_CHOICE_SPEED", ItemId.CHOICE_SCARF),
    125: ("HOLD_EFFECT_CHOICE_SPATK", ItemId.CHOICE_SPECS),
    # E2 batch 2 (no berries: those are named in DECOMP_ITEM_TO_ID). BattleSystem_TriggerLeftovers
    # (battle_lib.c:4922-4965), BattleSystem_TriggerDetrimentalHeldItem (:5307-5350) and
    # TriggerHeldItemOnHit's transfer (:5387-5398), the accuracy check
    # (battle_controller_player.c:2949-2965), CheckExtraFlinch (:4630-4653), Shell Bell
    # (:4736-4752), BtlCmd_TryLightScreen / TryReflect (battle_script.c:4229, :4272).
    48: ("HOLD_EFFECT_ACC_REDUCE", ItemId.BRIGHT_POWDER),
    56: ("HOLD_EFFECT_SOMETIMES_FLINCH", ItemId.KINGS_ROCK),
    69: ("HOLD_EFFECT_HP_RESTORE_GRADUAL", ItemId.LEFTOVERS),
    88: ("HOLD_EFFECT_HP_RESTORE_ON_DMG", ItemId.SHELL_BELL),
    93: ("HOLD_EFFECT_ACCURACY_UP", ItemId.WIDE_LENS),
    97: ("HOLD_EFFECT_EXTEND_SCREENS", ItemId.LIGHT_CLAY),
    100: ("HOLD_EFFECT_PSN_USER", ItemId.TOXIC_ORB),
    101: ("HOLD_EFFECT_BRN_USER", ItemId.FLAME_ORB),
    104: ("HOLD_EFFECT_ACCURACY_UP_SLOWER", ItemId.ZOOM_LENS),
    109: ("HOLD_EFFECT_HP_RESTORE_PSN_TYPE", ItemId.BLACK_SLUDGE),
    116: ("HOLD_EFFECT_DMG_USER_CONTACT_XFR", ItemId.STICKY_BARB),
    # Power Herb: a charge move hits on its first use (effect_script_0039.s:7, the other charge
    # scripts' CheckItemHoldEffect HOLD_EFFECT_CHARGE_SKIP), B10-12.
    99: ("HOLD_EFFECT_CHARGE_SKIP", ItemId.POWER_HERB),
}

# The weather rocks (effect_script_0115.s, _0136.s, _0137.s, _0164.s: the attacker's
# HOLD_EFFECT_EXTEND_* adds its param to BTLVAR_WEATHER_TURNS): decomp hold effect -> (name, the
# Weather its move must set). build_tables fills a WEATHER_ROCK item's item_param[i, 1] from here.
WEATHER_ROCK_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    110: ("HOLD_EFFECT_EXTEND_HAIL", Weather.HAIL),
    111: ("HOLD_EFFECT_EXTEND_SANDSTORM", Weather.SANDSTORM),
    112: ("HOLD_EFFECT_EXTEND_SUN", Weather.SUN),
    113: ("HOLD_EFFECT_EXTEND_RAIN", Weather.RAIN),
}
HOLD_EFFECT_TO_ID.update(
    {h: (name, ItemId.WEATHER_ROCK) for h, (name, _) in WEATHER_ROCK_HOLD_EFFECTS.items()}
)

# The crit-stage items of BattleSystem_CalcCriticalMulti (battle_lib.c:7123-7128): decomp hold
# effect -> (name, the species the item needs, 0 for any holder). build_tables fills a CRIT_ITEM
# item's item_param[i, 1] from here (E2 batch 4).
CRIT_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    67: ("HOLD_EFFECT_CRITRATE_UP", 0),
    89: ("HOLD_EFFECT_CHANSEY_CRITRATE_UP", SPECIES_CHANSEY),
    92: ("HOLD_EFFECT_FARFETCHD_CRITRATE_UP", SPECIES_FARFETCHD),
}
HOLD_EFFECT_TO_ID.update(
    {h: (name, ItemId.CRIT_ITEM) for h, (name, _) in CRIT_HOLD_EFFECTS.items()}
)

# E1 Gardenia batch. Focus Band (BattleControllerPlayer_UpdateHP, bcp.c:3386-3389;
# BtlCmd_CheckHoldOnWith1HP, battle_script.c:9109-9112), ruling G-1.
HOLD_EFFECT_TO_ID[65] = ("HOLD_EFFECT_MAYBE_ENDURE", ItemId.FOCUS_BAND)

# E1 Batch D. The Metronome item (BtlCmd_CalcDamage, battle_script.c:1350-1352;
# BattleSystem_UpdateMetronomeCount / _VerifyMetronomeCount, battle_lib.c:6160-6198), G-60; Iron
# Ball (sSpeedHalvingItemEffects, battle_lib.c:1160-1169, and the grounding sites), G-56; Grip
# Claw (subscript_bind_start.s:10-12), G-57.
HOLD_EFFECT_TO_ID.update(
    {
        105: ("HOLD_EFFECT_BOOST_REPEATED", ItemId.METRONOME_ITEM),
        106: ("HOLD_EFFECT_SPEED_DOWN_GROUNDED", ItemId.IRON_BALL),
        114: ("HOLD_EFFECT_EXTEND_TRAPPING", ItemId.GRIP_CLAW),
    }
)

# The flavor berries (BattleSystem_TriggerHeldItem, battle_lib.c:4698-4771, and ..._OnStatus,
# :5121-5194; BattleSystem_PluckBerry's PLUCK_EFFECT_HP_RESTORE_*, :5678-5751): decomp hold
# effect -> (name, the Flavor its msgTemp names). build_tables fills a FLAVOR_BERRY item's
# item_param[i, 1] from here (ruling G-4).
FLAVOR_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    14: ("HOLD_EFFECT_HP_RESTORE_SPICY", Flavor.SPICY),
    15: ("HOLD_EFFECT_HP_RESTORE_DRY", Flavor.DRY),
    16: ("HOLD_EFFECT_HP_RESTORE_SWEET", Flavor.SWEET),
    17: ("HOLD_EFFECT_HP_RESTORE_BITTER", Flavor.BITTER),
    18: ("HOLD_EFFECT_HP_RESTORE_SOUR", Flavor.SOUR),
}
HOLD_EFFECT_TO_ID.update(
    {h: (name, ItemId.FLAVOR_BERRY) for h, (name, _) in FLAVOR_HOLD_EFFECTS.items()}
)

# HOLD_EFFECT_NONE. An item without a hold effect is NONE (no implemented path reads it by
# id), except a berry: Item_IsBerry (src/item.c:357-365) reads sBerryItemIDs, the items
# FIRST_BERRY_IDX..LAST_BERRY_IDX (include/constants/items.h:56-58: Cheri Berry to Rowap
# Berry, every item with berryData, tools/dataproc/src/itemproc.c:272), and Pluck / Bug Bite
# eat such a berry with no effect (BattleSystem_PluckBerry, battle_lib.c:5832-5835):
# NO_EFFECT_BERRY.
HOLD_EFFECT_NONE = 0
FIRST_BERRY, LAST_BERRY = "Cheri Berry", "Rowap Berry"


def held_item_ids(items: list[dict]) -> list[ItemId]:
    """The ``ItemId`` of every item (``data/intermediate/items.json`` rows, in id order): by
    name (``DECOMP_ITEM_TO_ID``), else by hold effect (``HOLD_EFFECT_TO_ID``), else
    ``NONE`` / ``NO_EFFECT_BERRY`` for no hold effect, else ``UNSUPPORTED``."""
    index = {item["name"]: n for n, item in enumerate(items)}
    first, last = index[FIRST_BERRY], index[LAST_BERRY]
    out = []
    for n, item in enumerate(items):
        hold = item["hold_effect"]
        if item["name"] in DECOMP_ITEM_TO_ID:
            out.append(DECOMP_ITEM_TO_ID[item["name"]])
        elif hold in HOLD_EFFECT_TO_ID:
            out.append(HOLD_EFFECT_TO_ID[hold][1])
        elif hold == HOLD_EFFECT_NONE:
            out.append(ItemId.NO_EFFECT_BERRY if first <= n <= last else ItemId.NONE)
        else:
            out.append(ItemId.UNSUPPORTED)
    return out


# The hold effects subscript_type_resist_berry.s reads (decomp hold effect id, i.e. line
# id + 1 of generated/item_hold_effects.txt -> (name, the type it resists)): each of 19-34
# halves a super-effective hit of its type (_135-_240); 35 (Chilan) halves any Normal-type
# hit (_128, before the super-effective test). build_tables fills a RESIST_BERRY item's
# item_param[i, 1] from here; DECOMP_ITEM_TO_ID names every berry with one of these hold
# effects. tests/tools/test_effect_map.py reads both against the decomp.
RESIST_BERRY_HOLD_EFFECTS: dict[int, tuple[str, int]] = {
    19: ("HOLD_EFFECT_WEAKEN_SE_FIRE", Type.FIRE),
    20: ("HOLD_EFFECT_WEAKEN_SE_WATER", Type.WATER),
    21: ("HOLD_EFFECT_WEAKEN_SE_ELECTRIC", Type.ELECTRIC),
    22: ("HOLD_EFFECT_WEAKEN_SE_GRASS", Type.GRASS),
    23: ("HOLD_EFFECT_WEAKEN_SE_ICE", Type.ICE),
    24: ("HOLD_EFFECT_WEAKEN_SE_FIGHT", Type.FIGHTING),
    25: ("HOLD_EFFECT_WEAKEN_SE_POISON", Type.POISON),
    26: ("HOLD_EFFECT_WEAKEN_SE_GROUND", Type.GROUND),
    27: ("HOLD_EFFECT_WEAKEN_SE_FLYING", Type.FLYING),
    28: ("HOLD_EFFECT_WEAKEN_SE_PSYCHIC", Type.PSYCHIC),
    29: ("HOLD_EFFECT_WEAKEN_SE_BUG", Type.BUG),
    30: ("HOLD_EFFECT_WEAKEN_SE_ROCK", Type.ROCK),
    31: ("HOLD_EFFECT_WEAKEN_SE_GHOST", Type.GHOST),
    32: ("HOLD_EFFECT_WEAKEN_SE_DRAGON", Type.DRAGON),
    33: ("HOLD_EFFECT_WEAKEN_SE_DARK", Type.DARK),
    34: ("HOLD_EFFECT_WEAKEN_SE_STEEL", Type.STEEL),
    35: ("HOLD_EFFECT_WEAKEN_NORMAL", RESIST_ANY_NORMAL),
}

# HOLD_EFFECT_FRIENDSHIP_UP (generated/item_hold_effects.txt:54, id 53): a bag item with it
# raises BattleSystem_UseBagItem's friendship gain by half (battle_system.c:746-748);
# tables.item_party_param's ItemParam.HOLD_FRIENDSHIP_UP column.
HOLD_EFFECT_FRIENDSHIP_UP = 53
