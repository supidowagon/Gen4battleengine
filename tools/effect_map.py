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
are Insomnia and Limber (implemented in ``engine/effects.py``), Shield Dust against a
damaging move's secondary status (``effects._inflict``), and Vital Spirit, Immunity, Water
Veil, Magma Armor and Leaf Guard (``AbilityId.UNSUPPORTED``), and
Safeguard and Substitute are unsupported moves, so none of those branches can be
reached silently. Uproar's sleep block (subscript_fall_asleep.s:14-16, :43-45) is modelled
(E2 batch 5, ``effects._inflict``).
"""

from __future__ import annotations

from nuzline.engine.ids import (
    COUNTER_ANY,
    COUNTER_PHYSICAL,
    COUNTER_SPECIAL,
    MINIMIZE_PARAM,
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
    ItemId,
    Stat,
    Status,
    TrainerItemId,
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
    # Gust: effect_script_0149.s doubles the power against MOVE_EFFECT_AIRBORNE (Fly,
    # Bounce), which no supported move sets (test_gust_note_holds).
    149: ("BATTLE_EFFECT_DOUBLE_DAMAGE_FLY_OR_BOUNCE", _E.PLAIN, (0, 0, 0)),
    # Sand Tomb (Kaizo), Whirlpool: effect_script_0261.s is Bind's script (the same
    # BIND_TARGET indirect effect) plus a x2 against MOVE_EFFECT_UNDERWATER (Dive, unsupported).
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
    # MoveFlag.THUNDER / BLIZZARD (tools/build_tables.py); Thunder's SYSCTL_HIT_DURING_FLY only
    # matters against Fly / Bounce, unsupported moves (B9-1).
    152: ("BATTLE_EFFECT_THUNDER", _E.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),
    260: ("BATTLE_EFFECT_BLIZZARD", _E.SECONDARY_STATUS, (Status.FREEZE, 0, 0)),
    # effect_script_0122.s: BtlCmd_Present (battle_script.c:5789-5806) draws the power or the
    # heal path (engine/effects.py ``_present``, damage.PRESENT_POWER).
    122: ("BATTLE_EFFECT_RANDOM_POWER_MAYBE_HEAL", _E.PRESENT, (0, 0, 0)),
    # effect_script_0111.s (Protect, Detect): a direct TO_ATTACKER effect running
    # subscript_protect.s -> BtlCmd_TryProtection (battle_script.c:5149-5190); the block is
    # CheckMoveHitOverrides (battle_controller_player.c:3001-3014; engine/step.py).
    111: ("BATTLE_EFFECT_PROTECT", _E.PROTECT, (0, 0, 0)),
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
    # Earthquake: effect_script_0147.s doubles the power against MOVE_EFFECT_UNDERGROUND (Dig),
    # which no supported move sets, like Gust (149); SYSCTL_HIT_DURING_DIG only matters against
    # a digging target.
    147: ("BATTLE_EFFECT_DOUBLE_DAMAGE_DIG", _E.PLAIN, (0, 0, 0)),
    # Poison Fang: effect_script_0202.s, an indirect TO_DEFENDER badly-poison at the move's
    # chance (subscript_badly_poison.s).
    202: ("BATTLE_EFFECT_BADLY_POISON_HIT", _E.SECONDARY_STATUS, (Status.TOXIC, 0, 0)),
    118: ("BATTLE_EFFECT_ATK_UP_2_STATUS_CONFUSION", _E.SWAGGER, (0, 0, 0)),
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
    # Dragon Dance / Bulk Up (E2 batch 9, B9-10): subscript_user_atk_and_speed_up_1_stage.s and
    # subscript_user_atk_and_def_up_1_stage.s differ only in the second stat (:4, :12).
    212: ("BATTLE_EFFECT_ATK_SPD_UP", _E.DRAGON_DANCE, (_SPE, 0, 0)),
    208: ("BATTLE_EFFECT_ATK_DEF_UP", _E.DRAGON_DANCE, (_DEF, 0, 0)),
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
    # E2 batch 9 (B9-5). Surf: effect_script_0257.s doubles the power against
    # MOVE_EFFECT_UNDERWATER (Dive), which no supported move sets, like Gust (149) and Earthquake
    # (147); SYSCTL_HIT_DURING_DIVE only matters against a diving target.
    257: ("BATTLE_EFFECT_DOUBLE_DAMAGE_DIVE", _E.PLAIN, (0, 0, 0)),
    # Follow Me: effect_script_0172.s (BtlCmd_FollowMe, battle_script.c:6261-6270, then an ON_HIT
    # message). Helping Hand: effect_script_0176.s (BtlCmd_TryHelpingHand, :6291-6314).
    172: ("BATTLE_EFFECT_MAKE_GLOBAL_TARGET", _E.FOLLOW_ME, (0, 0, 0)),
    176: ("BATTLE_EFFECT_BOOST_ALLY_POWER_BY_50_PERCENT", _E.HELPING_HAND, (0, 0, 0)),
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
KAIZO_EFFECTS: dict[int, tuple[str, EffectId, tuple[int, int, int]]] = {
    48: ("Kaizo effect 48: Reckless, 1/2 recoil", _E.RECOIL, (2, 0, 0)),
    182: ("Kaizo effect 182: Reckless, 1/4 recoil", _E.RECOIL, (4, 0, 0)),
}

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
# Move_IsMultiTurn (battle_lib.c:2975-2992), by decomp effect id: BIDE, CHARGE_TURN_HIGH_CRIT,
# CHARGE_TURN_HIGH_CRIT_FLINCH, CHARGE_TURN_DEF_UP, SKIP_CHARGE_TURN_IN_SUN, FLY, DIVE, DIG,
# BOUNCE, SHADOW_FORCE (generated/move_battle_effects.txt).
MULTI_TURN_EFFECTS: tuple[int, ...] = (26, 39, 75, 145, 151, 155, 255, 256, 263, 272)
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
# The berries whose trigger divides maxHP by their param (the stat pinch berries, Custap): a
# param below 2 divides by zero (param 0, or 1 halved by Gluttony; battle_lib.c:4778, :1294),
# whose result is the ROM's division routine's, not the decomp's: build_tables maps such an
# item to UNSUPPORTED (Kaizo's Ganlon and Apicot Berries have param 0).
PINCH_DIVISOR_ITEMS = (ItemId.PINCH_BERRY, ItemId.CUSTAP_BERRY)
# Every item that divides by its hold param, and the smallest param that never divides by
# zero: the pinch-divisor berries above (2: Gluttony halves the param), Sticky Barb's
# end-of-turn ``Divide(maxHP, param)`` (battle_lib.c:5330-5336) and Shell Bell's
# ``Divide(dealt, param)`` (bcp.c:4736-4752, battle_lib.c:7345-7355) (1). build_tables maps an
# item below its minimum to UNSUPPORTED.
DIVISOR_PARAM_MIN: dict[ItemId, int] = {
    **{item: 2 for item in PINCH_DIVISOR_ITEMS},
    ItemId.STICKY_BARB: 1,
    ItemId.SHELL_BELL: 1,
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

# Bag items a trainer's AI may use (TrainerAI_ShouldUseItem, trainer_ai.c:4056).
DECOMP_TRAINER_ITEM_TO_ID: dict[str, TrainerItemId] = {
    "None": TrainerItemId.NONE,  # ITEM_NONE (id 0): an empty bag slot
    "Potion": TrainerItemId.POTION,  # itemUseParams.hpRestored 20
}
