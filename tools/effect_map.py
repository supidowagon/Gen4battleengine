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

* ``BATTLE_EFFECT_RECOIL_THIRD`` (198): the decomp divides damage by 3
  (``subscript_recoil_1_3.s``), which a ``RECOIL`` percent cannot express exactly.
* ``BATTLE_EFFECT_STATUS_BADLY_POISON`` / ``BADLY_POISON_HIT`` (33, 202): Toxic
  infliction is not in M1.
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
are Insomnia and Limber (implemented in ``engine/effects.py``) and Vital Spirit, Immunity,
Water Veil, Magma Armor, Leaf Guard and Shield Dust (``AbilityId.UNSUPPORTED``), and
Safeguard, Substitute and Uproar are unsupported moves, so none of those branches can be
reached silently.
"""

from __future__ import annotations

from nuzline.engine.ids import (
    RESIST_ANY_NORMAL,
    AbilityId,
    EffectId,
    ItemId,
    Stat,
    Status,
    TrainerItemId,
    Type,
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
    # subscript_recoil_1_4: damage / 4 == damage * 25 // 100
    48: ("BATTLE_EFFECT_RECOIL_QUARTER", _E.RECOIL, (25, 0, 0)),
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
    # flagged by move_param[m, 2] = 1. RECOIL_BURN_HIT (Flare Blitz, 254) also has a
    # third recoil and stays unsupported.
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
    # subscript_recoil_1_2: damage / 2 == damage * 50 // 100
    269: ("BATTLE_EFFECT_RECOIL_HALF", _E.RECOIL, (50, 0, 0)),
    271: ("BATTLE_EFFECT_LOWER_SP_DEF_2_HIT", _E.SECONDARY_STAT_FOE, (_SPD, -2, 0)),
    # effect_script_0275.s -> subscript_paralyze_or_flinch.s: a paralysis roll, then a flinch
    # roll (engine/effects.py ``_fang``). Fire Fang (273) and Ice Fang (274) call other
    # subscripts and stay unmapped until those are read (docs/kaizo/k3b2-rules.md §11 item 7).
    275: ("BATTLE_EFFECT_FLINCH_PARALYZE_HIT", _E.FANG_PARALYZE_FLINCH, (Status.PARALYSIS, 0, 0)),
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
    "Rindo Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_SE_GRASS (22)
    "Chilan Berry": ItemId.RESIST_BERRY,  # HOLD_EFFECT_WEAKEN_NORMAL (35)
}

# The hold effects subscript_type_resist_berry.s reads (decomp hold effect id, i.e. line
# id + 1 of generated/item_hold_effects.txt -> (name, the type it resists)): each of 19-34
# halves a super-effective hit of its type (_135-_240); 35 (Chilan) halves any Normal-type
# hit (_128, before the super-effective test). build_tables fills a RESIST_BERRY item's
# item_param[i, 1] from here; only the items DECOMP_ITEM_TO_ID names are supported, the
# others stay UNSUPPORTED. tests/tools/test_effect_map.py reads both against the decomp.
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
