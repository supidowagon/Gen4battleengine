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
* ``SOUND_MOVES`` / ``PUNCH_MOVES`` mirror ``sSoundMoves`` and ``sPunchingMoves``
  in ``src/battle/battle_lib.c``; the decomp has no move-data flag for either.

Deliberately unsupported although close to a supported family:

* ``BATTLE_EFFECT_RECOIL_THIRD`` (198): the decomp divides damage by 3
  (``subscript_recoil_1_3.s``), which a ``RECOIL`` percent cannot express exactly.
* ``BATTLE_EFFECT_STATUS_BADLY_POISON`` / ``BADLY_POISON_HIT`` (33, 202): Toxic
  infliction is not in M1.
* ``BATTLE_EFFECT_HIT_TWICE`` (44): a fixed hit count, unlike ``MULTI_HIT``'s 2-5.
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
(Insomnia, Vital Spirit, Limber, Immunity, Water Veil, Magma Armor, Leaf Guard,
Shield Dust) map to ``AbilityId.UNSUPPORTED``, and Safeguard, Substitute and
Uproar are unsupported moves, so none of those branches can be reached silently.
"""

from __future__ import annotations

from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat, Status, TrainerItemId

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
    31: ("BATTLE_EFFECT_FLINCH_HIT", _E.FLINCH, (0, 0, 0)),
    42: ("BATTLE_EFFECT_BIND_HIT", _E.TRAP, (0, 0, 0)),
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
    # Mud Sport: effect_script_0201.s sets MOVE_EFFECT_MUD_SPORT on the user.
    201: ("BATTLE_EFFECT_HALVE_ELECTRIC_DAMAGE", _E.MUD_SPORT, (0, 0, 0)),
    # Bug Bite, Pluck: effect_script_0224.s, an ON_HIT indirect effect (subscript_pluck.s).
    224: ("BATTLE_EFFECT_EAT_BERRY", _E.EAT_BERRY, (0, 0, 0)),
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
    # Defense Curl: Rollout doubling is a later milestone (plan ruling, Task 8b).
    156: ("BATTLE_EFFECT_DEF_UP_DOUBLE_ROLLOUT_POWER", _E.STAT_SELF, (_DEF, 1, 0)),
    167: ("BATTLE_EFFECT_STATUS_BURN", _E.STATUS_FOE, (Status.BURN, 0, 0)),
    175: ("BATTLE_EFFECT_TAUNT", _E.TAUNT, (0, 0, 0)),
    210: ("BATTLE_EFFECT_HALVE_FIRE_DAMAGE", _E.WATER_SPORT, (0, 0, 0)),
    # Hammer Arm: certain self Speed drop (effect_script_0218.s, ON_HIT|TO_ATTACKER).
    218: ("BATTLE_EFFECT_SPEED_DOWN_HIT", _E.SECONDARY_STAT_SELF, (_SPE, -1, 0)),
    266: ("BATTLE_EFFECT_STEALTH_ROCK", _E.STEALTH_ROCK, (0, 0, 0)),
    # subscript_recoil_1_2: damage / 2 == damage * 50 // 100
    269: ("BATTLE_EFFECT_RECOIL_HALF", _E.RECOIL, (50, 0, 0)),
    271: ("BATTLE_EFFECT_LOWER_SP_DEF_2_HIT", _E.SECONDARY_STAT_FOE, (_SPD, -2, 0)),
    276: ("BATTLE_EFFECT_RAISE_SP_ATK_HIT", _E.SECONDARY_STAT_SELF, (_SPA, 1, 0)),
}

DECOMP_EFFECT_NAMES: dict[int, str] = {k: v[0] for k, v in _EFFECTS.items()}
DECOMP_EFFECT_TO_ID: dict[int, tuple[EffectId, tuple[int, int, int]]] = {
    k: (eid, tuple(int(p) for p in params)) for k, (_, eid, params) in _EFFECTS.items()
}

# SECONDARY_* effects the decomp applies on every hit (MOVE_SIDE_EFFECT_ON_HIT)
# rather than with the move's effect chance; such moves carry chance 0.
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
}

DECOMP_ITEM_TO_ID: dict[str, ItemId] = {
    "None": ItemId.NONE,  # ITEM_NONE (id 0)
    "Oran Berry": ItemId.ORAN_BERRY,  # HOLD_EFFECT_HP_RESTORE, param 10 HP
    "Sitrus Berry": ItemId.SITRUS_BERRY,  # HOLD_EFFECT_HP_PCT_RESTORE, param 25 %
}

# Bag items a trainer's AI may use (TrainerAI_ShouldUseItem, trainer_ai.c:4056).
DECOMP_TRAINER_ITEM_TO_ID: dict[str, TrainerItemId] = {
    "None": TrainerItemId.NONE,  # ITEM_NONE (id 0): an empty bag slot
    "Potion": TrainerItemId.POTION,  # itemUseParams.hpRestored 20
}

# src/battle/battle_lib.c sSoundMoves (Soundproof).
SOUND_MOVES: frozenset[str] = frozenset({
    "Growl", "Roar", "Sing", "Supersonic", "Screech", "Snore", "Uproar", "Metal Sound",
    "GrassWhistle", "Hyper Voice", "Bug Buzz", "Chatter",
})  # fmt: skip

# src/battle/battle_lib.c sPunchingMoves (Iron Fist).
PUNCH_MOVES: frozenset[str] = frozenset({
    "Ice Punch", "Fire Punch", "ThunderPunch", "Mach Punch", "Focus Punch", "Dizzy Punch",
    "DynamicPunch", "Hammer Arm", "Mega Punch", "Comet Punch", "Meteor Mash",
    "Shadow Punch", "Drain Punch", "Bullet Punch", "Sky Uppercut",
})  # fmt: skip
