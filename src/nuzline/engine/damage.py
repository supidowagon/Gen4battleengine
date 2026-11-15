"""Gen 4 damage calculation (pure ``jnp``, ``int32`` throughout, jit/vmap-able).

Mirrors the decomp's pipeline for one hit of a damaging move:

1. ``BattleSystem_CalcMoveDamage`` (``src/battle/battle_lib.c:6601``): power and
   stat modifiers, stages, ``A * power * level_factor / D / 50``, burn, screens,
   weather, ``+ 2``.
2. ``BattleScript_CalcMoveDamage`` (``src/battle/battle_script.c:1344``): ``* criticalMul``, then
   Life Orb (``:1346``, ``life_orb``).
3. ``BattleSystem_CalcDamageVariance`` (``battle_lib.c:7081``): ``* (100 - rand % 16) / 100``,
   at least 1. Our ``roll`` is ``15 - rand % 16``, so roll ``r`` multiplies by
   ``85 + r`` and ``damage_table`` is ascending.
4. ``BattleSystem_ApplyTypeChart`` (``battle_lib.c:2560``, run later by
   ``BattleControllerPlayer_CheckTypeChart``): STAB ``* 15 / 10`` (``* 2`` with
   Adaptability), then each defender type's multiplier through ``BattleSystem_Divide``
   (``:3599``, a non-zero result never truncates to 0), in the order their rows appear
   in ``sTypeMatchupMultipliers`` (``:2399``, walked at ``:2611-2646``), which is not
   always type 1 then type 2 (``tables.type_chart_order``), then Solid Rock on a
   super-effective hit (``Divide(damage * 3, 4)``, ``:2657-2660``) and the attacker's
   Expert Belt (``:2663-2665``).

Held items: the item block of ``BattleSystem_CalcMoveDamage`` (``:6716-6783``: the
type-boosting items, Choice Band / Specs, the species items, Muscle Band, Wise Glasses;
``_item_modifiers``) reads the attacker's and the defender's items, so it is in the trainer
AI's estimate and the confusion self-hit too. Life Orb's ``* (100 + param) / 100`` is the move
script's (``BattleScript_CalcMoveDamage``, battle_script.c:1346-1348, after the crit): only
``damage_table`` and ``damage_for_roll`` apply it (``life_orb``), never the AI's estimate, the
self-hit (``CALC_SELF_HIT`` calls ``BattleSystem_CalcMoveDamage`` directly) or a fixed-damage
move (no ``CalcDamage`` in its script).

The trainer AI's estimate (``TrainerAI_CalcDamage``, ``trainer_ai.c:2868``) runs
steps 1, 4 and then 3: ``ai_damage_for_roll`` (crit multiplier 1, variance last).

Only mechanics of the implemented set are modelled. Abilities, items and moves
outside it are ``UNSUPPORTED`` ids and never reach the branches here; the decomp
modifiers they would trigger (Hustle, Flash Fire, Heatproof, ...) are therefore omitted, not
approximated. E2 batch 3 adds Technician, Thick Fat, Dry Skin's Fire boost, Solar Power and
Reckless's powerMul (``script_power_mul``) to the formula and Scrappy, Tinted Lens and
Levitate to the type chart (``_finish``). Cloud Nine and Air Lock gate the
weather modifiers (``weather_suppressed``). Doubles-only
spread and screen rules do not apply (singles).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import (
    ABILITY_DECOMP_IDS,
    ITEM_GRISEOUS_ORB,
    MOVE_STRUGGLE,
    POWER_IF_TARGET_HALF_HP,
    POWER_IF_TARGET_PARALYZED,
    POWER_IF_USER_STATUSED,
    SPECIES_CLAMPERL,
    SPECIES_CUBONE,
    SPECIES_DIALGA,
    SPECIES_DITTO,
    SPECIES_GIRATINA,
    SPECIES_LATIAS,
    SPECIES_LATIOS,
    SPECIES_MAROWAK,
    SPECIES_PALKIA,
    SPECIES_PIKACHU,
    UNTRACEABLE_ABILITIES,
    WEATHER_ABILITIES,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.state import BattleState

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category
CRIT_MUL = 2  # BattleSystem_CalcCriticalMulti (battle_lib.c:7105-7144)
SNIPER_CRIT_MUL = 3  # :7142-7144: x3 for the attacker's Sniper (Battler_Ability)
# sWeightToPower (include/data/battle/weight_to_power.h): the first bound >= the defender's
# weight (hectograms) gives the power, else 120 (BtlCmd_CalcWeightBasedPower,
# battle_script.c:6808-6826).
WEIGHT_BOUNDS = (100, 250, 500, 1000, 2000)
WEIGHT_POWER = (20, 40, 60, 80, 100, 120)
# MON_CONDITION_FACADE_BOOST (include/constants/battle/condition.h:29).
FACADE_STATUSES = (Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)
ROLLS = 16
CONFUSION_POWER = 40  # CALC_SELF_HIT(MOVE_STRUGGLE, 40), battle_controller_player.c:2633

# sStatStageBoosts (battle_lib.c:6550), indexed by stage + 6.
_STAGE_NUM = jnp.array([10, 10, 10, 10, 10, 10, 10, 15, 20, 25, 30, 35, 40], jnp.int32)
_STAGE_DEN = jnp.array([40, 35, 30, 25, 20, 15, 10, 10, 10, 10, 10, 10, 10], jnp.int32)

_I32 = jnp.int32

# BtlCmd_CalcMagnitudePower (battle_script.c:5820-5852): the power of Magnitude levels 4-10.
MAGNITUDE_POWER = jnp.array([10, 30, 50, 70, 90, 110, 150], jnp.int32)
MAGNITUDE_MIN = 4
# BtlCmd_Present (battle_script.c:5795-5800): movePower 40, 80 or 120 for Outcome.present 0,
# 1, 2; PRESENT_HEAL (3) is the heal path, which runs no CalcDamage (effect_script_0122.s).
# PRESENT_TABLE (no drawn value) keeps the table power, as BattleSystem_CalcMoveDamage does
# with no input power (battle_lib.c:6674): the turn-start tables and the AI's own estimates.
PRESENT_POWER = jnp.array([40, 80, 120], jnp.int32)
PRESENT_HEAL = 3
PRESENT_TABLE = -1
# BtlCmd_CalcFlailPower (battle_script.c:4958-4970): sHPPixelsToFlailPower (:4941-4948) as
# (largest pixel count of the band, power); the first row with pixels <= its bound wins.
# The last row's bound is 64, the most App_PixelCount(cur, max, 64) can return.
FLAIL_BOUNDS = (1, 5, 12, 21, 42, 64)
FLAIL_POWER = jnp.array([200, 150, 100, 80, 40, 20], jnp.int32)
FLAIL_PIXELS = 64


def hp_pixels(cur, max_hp):
    """``App_PixelCount(cur, max, 64)`` (src/unk_0208C098.c:41-49): ``cur * 64 / max`` in
    integer arithmetic, but at least 1 while ``cur > 0``."""
    cur, max_hp = _i32(cur), _i32(max_hp)
    pixels = cur * FLAIL_PIXELS // jnp.maximum(max_hp, 1)
    return jnp.where((pixels == 0) & (cur > 0), 1, pixels)


def flail_power(cur, max_hp):
    """``int32``: Flail / Reversal's power at ``cur`` of ``max_hp`` HP (battle_script.c:4958)."""
    pixels = hp_pixels(cur, max_hp)
    row = jnp.sum(pixels > jnp.asarray(FLAIL_BOUNDS, jnp.int32))
    return FLAIL_POWER[jnp.minimum(row, len(FLAIL_BOUNDS) - 1)]


def _i32(x) -> jnp.ndarray:
    return jnp.asarray(x).astype(_I32)


# ABILITY_DECOMP_IDS as an array indexed by AbilityId (-1 for UNSUPPORTED, never traced).
_ABILITY_DECOMP = jnp.array([ABILITY_DECOMP_IDS.get(a, -1) for a in AbilityId], jnp.int32)


def decomp_abilities(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[2, SLOTS]``: every slot's decomp ability id (``Battler_Ability``), from
    ``tables.species_abilities_decomp`` at the Pokemon's personal row and ability slot; for
    a Trace holder whose ``ability`` is no longer Trace (it traced, ``hooks.trace``), the
    copied ability's (``ids.ABILITY_DECOMP_IDS``)."""
    row = _i32(tables.species_form_row[_i32(state.species), _i32(state.form)])
    slot = _i32(state.ability_slot)
    own = _i32(tables.species_abilities_decomp[row, slot])
    trace = ABILITY_DECOMP_IDS[AbilityId.TRACE]
    traced = (own == trace) & (_i32(state.ability) != AbilityId.TRACE)
    return jnp.where(traced, _ABILITY_DECOMP[_i32(state.ability)], own)


def ignorable(state: BattleState, atk_side, def_side, slot=None) -> jnp.ndarray:
    """``int32``: ``def_side``'s ability (its active battler's, or party ``slot``'s) as
    ``atk_side``'s active battler sees it: ``Battler_IgnorableAbility`` (battle_lib.c:
    3108-3121) hides it behind the attacker's Mold Breaker."""
    slot = state.active[def_side] if slot is None else slot
    attacker = _i32(state.ability[atk_side, state.active[atk_side]])
    return jnp.where(
        attacker == AbilityId.MOLD_BREAKER, AbilityId.NONE, _i32(state.ability[def_side, slot])
    )


def levitated(state: BattleState, atk_side, def_side, move_type) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` sets ``MOVE_STATUS_LEVITATED`` (battle_lib.c:
    2600-2603) for a Ground-type move against a Levitate holder as the attacker sees it
    (``Battler_IgnorableAbility``; the Iron Ball, Gravity and Ingrain exceptions are
    unsupported). The flag is part of ``MOVE_STATUS_IMMUNE``: no damage (``_finish``)."""
    seen = ignorable(state, atk_side, def_side)
    return (seen == AbilityId.LEVITATE) & (_i32(move_type) == Type.GROUND)


def identified(state: BattleState, atk_side, def_side) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` stops at the 0xFE separator, skipping the
    Ghost immunities (battle_lib.c:2611-2615), for a Foresight-identified defender or an
    attacker with Scrappy (``Battler_Ability``)."""
    scrappy = state.ability[atk_side, state.active[atk_side]] == AbilityId.SCRAPPY
    return (state.foresight[def_side] != 0) | scrappy


def trace_copy(tables: Tables, state: BattleState, side, foe, slot=None) -> jnp.ndarray:
    """``bool``: ``SWITCH_IN_CHECK_STATE_TRACE`` (battle_lib.c:3720-3735) copies ``foe``'s
    active battler's ability to ``side``'s battler (its active one, or party ``slot`` as it
    enters): that battler has HP, holds no Griseous Orb and has Trace (``Battler_Ability``;
    one that traced has another ability now, so ``traceAnnounced`` is never read apart from
    that), and ``ChooseTraceTarget`` (:7750-7775) finds the foe: alive, its
    ``BattleMon.ability`` not Forecast, Trace or Multitype (decomp ids). In singles both of
    the target's slots name the one foe. ``subscript_trace.s`` then writes the foe's ability
    (``hooks.trace``)."""
    slot = state.active[side] if slot is None else slot
    target = state.active[foe]
    foe_decomp = decomp_abilities(tables, state)[foe, target]
    traceable = ~jnp.isin(foe_decomp, jnp.asarray(UNTRACEABLE_ABILITIES, foe_decomp.dtype))
    return (
        (state.ability[side, slot] == AbilityId.TRACE)
        & (state.hp[side, slot] > 0)
        & (state.item[side, slot] != ITEM_GRISEOUS_ORB)
        & (state.hp[foe, target] > 0)
        & traceable
    )


def ability_weather(ability) -> jnp.ndarray:
    """``int32``: the ``Weather`` a switch-in weather ability sets (``ids.WEATHER_ABILITIES``),
    else ``Weather.NONE``."""
    out = jnp.asarray(Weather.NONE, _I32)
    for setter, weather in WEATHER_ABILITIES.items():
        out = jnp.where(_i32(ability) == setter, weather, out)
    return out


def stealth_rock_loss(tables: Tables, state: BattleState, side, slot) -> jnp.ndarray:
    """``int32``: what Stealth Rock takes from ``side``'s party ``slot`` as it switches in
    (``subscript_hazards_check.s`` -> ``BtlCmd_CheckStealthRock``, battle_script.c:8302): with
    the side's flag set, ``BattleSystem_Divide(maxHP, 32/16/8/4/2)`` by the Rock matchup
    (``BattleSystem_TypeMatchupMultiplier``), i.e. ``maxHP * eff // 800`` in ``effectiveness``'s
    x100 encoding, at least 1; 0 when immune or without rocks. Not capped at the HP
    (``effects.stealth_rock_damage`` caps it)."""
    eff = effectiveness(tables, Type.ROCK, state.types[side, slot])
    rock = jnp.where(eff > 0, jnp.maximum(_i32(state.max_hp[side, slot]) * eff // 800, 1), 0)
    return jnp.where(state.stealth_rock[side] != 0, rock, 0)


def grounded(state: BattleState, side, slot) -> jnp.ndarray:
    """``bool``: subscript_hazards_check.s:6-11 lets the entry spikes reach ``side``'s party
    ``slot``: Gravity grounds it (:6; Iron Ball, :7, is an unsupported item), else its own
    Levitate (``CheckAbility``) or a Flying type (``BATTLEMON_TYPE_1`` / ``_2``) keeps it off (Magnet
    Rise, :11, is an unsupported move)."""
    airborne = (state.ability[side, slot] == AbilityId.LEVITATE) | jnp.any(
        state.types[side, slot] == Type.FLYING
    )
    return (state.gravity != 0) | ~airborne


def spikes_loss(state: BattleState, side, slot) -> jnp.ndarray:
    """``int32``: what Spikes takes from ``side``'s party ``slot`` as it switches in
    (``BtlCmd_CheckSpikes``, battle_script.c:5434-5436): ``Divide(maxHP, (5 - layers) * 2)``, 1/8,
    1/6, 1/4, for a grounded newcomer (``grounded``); 0 without layers. Not capped."""
    layers = _i32(state.spikes[side])
    divisor = jnp.maximum((5 - layers) * 2, 1)
    loss = jnp.maximum(_i32(state.max_hp[side, slot]) // divisor, 1)
    return jnp.where((layers > 0) & grounded(state, side, slot), loss, 0)


def hazards_loss(tables: Tables, state: BattleState, side, slot) -> jnp.ndarray:
    """``int32``: Spikes and Stealth Rock together on ``side``'s newcomer ``slot``
    (subscript_hazards_check.s:31, :41), 0 for a Magic Guard holder (:5); not capped (the
    newcomer survives when this is below its HP: Stealth Rock needs HP left, :41)."""
    loss = spikes_loss(state, side, slot) + stealth_rock_loss(tables, state, side, slot)
    return jnp.where(state.ability[side, slot] == AbilityId.MAGIC_GUARD, 0, loss)


def chart_types(state: BattleState, side, slot, roosting) -> jnp.ndarray:
    """``int32[2]``: ``side``'s party ``slot``'s types as ``BattleSystem_ApplyTypeChart`` walks them:
    while the battler is ``roosting`` (``turnFlags.roosting``, Roost), ``BasicTypeMulApplies``
    skips every chart row whose defending type is Flying (battle_lib.c:2540-2543), which is the
    chart of ``Type.MYSTERY`` (no row names it): Flying reads as Mystery. Only the chart reads
    this; STAB, grounding and the hazards see Flying (``Battler_MonType``, :367-369, :7618)."""
    types = _i32(state.types[side, slot])
    return jnp.where(jnp.asarray(roosting, bool) & (types == Type.FLYING), Type.MYSTERY, types)


def entry_ability(tables: Tables, state: BattleState, side, slot, foe) -> jnp.ndarray:
    """The ``AbilityId`` ``side``'s party ``slot`` holds once it has switched in: the foe's
    active battler's when it traces it (``trace_copy``, ``hooks.trace``; ``foe``: the other
    side), else its own. For the turn-start laws of the foe's move (``outcome``,
    ``enumerate``): a switch goes first, and ``SWITCH_IN_CHECK_STATE_TRACE`` runs before the
    foe moves (ruling B3-19)."""
    traced = trace_copy(tables, state, side, foe, slot)
    return jnp.where(traced, state.ability[foe, state.active[foe]], state.ability[side, slot])


def entry_weather(tables: Tables, state: BattleState, side, slot, foe) -> jnp.ndarray:
    """``int32``: the weather once ``side``'s party ``slot`` has switched in
    (``hooks.on_switch_in``): its entry hazards (``hazards_loss``) leave
    it HP, and its weather ability, its own or the one it traces from the foe's active
    battler (``entry_ability``; ``foe``: the other side), sets that weather for good; else
    ``state.weather``. For the turn-start laws of the foe's move (``outcome``): a switch goes
    first."""
    alive = _i32(state.hp[side, slot]) - hazards_loss(tables, state, side, slot) > 0
    weather = ability_weather(entry_ability(tables, state, side, slot, foe))
    return jnp.where(alive & (weather != Weather.NONE), weather, _i32(state.weather))


# effect_script_0048.s, _0269.s (RECOIL: Take Down, Double-Edge, ...): a Reckless attacker
# (CheckAbility, Battler_Ability) sets BTLVAR_POWER_MULTI to 12 before CalcCrit / CalcDamage;
# the other Reckless scripts (0045, 0198, 0253, 0262) are unsupported effects.
RECKLESS_POWER_MUL = 12


# BTLVAR_POWER_MULTI 20 (effect_script_0169.s, _0171.s, _0221.s).
DOUBLE_POWER_MUL = 20


def double_power(tables: Tables, state: BattleState, atk_side, def_side, move) -> jnp.ndarray:
    """``bool``: ``atk_side``'s ``DOUBLE_POWER`` move doubles against ``def_side``'s active
    battler: SmellingSalt against a paralyzed defender (effect_script_0171.s:6), Facade for a
    user with ``MON_CONDITION_FACADE_BOOST`` (effect_script_0169.s:5), Brine against
    ``curHP <= maxHP / 2`` (effect_script_0221.s:5-8; ``OPCODE_DIV`` is a plain ``/``,
    battle_script.c:2711-2712). SmellingSalt's ``CheckSubstitute`` meets no substitute:
    Substitute is an unsupported move."""
    move = _i32(move)
    cond = _i32(tables.move_param[move, 0])
    user, target = state.active[atk_side], state.active[def_side]
    paralyzed = state.status[def_side, target] == Status.PARALYSIS
    statused = jnp.isin(state.status[atk_side, user], jnp.asarray(FACADE_STATUSES, jnp.int8))
    half = _i32(state.hp[def_side, target]) <= _i32(state.max_hp[def_side, target]) // 2
    holds = jnp.where(
        cond == POWER_IF_TARGET_PARALYZED,
        paralyzed,
        jnp.where(
            cond == POWER_IF_USER_STATUSED, statused, half & (cond == POWER_IF_TARGET_HALF_HP)
        ),
    )
    return (tables.move_effect[move] == EffectId.DOUBLE_POWER) & holds


def script_power_mul(
    tables: Tables, state: BattleState, atk_side, def_side, move, power_mul
) -> jnp.ndarray:
    """``int32``: ``battleCtx->powerMul`` when ``atk_side``'s move script runs ``CalcDamage``
    against ``def_side``'s active battler:
    ``RECKLESS_POWER_MUL`` for a Reckless user's ``RECOIL`` move, ``DOUBLE_POWER_MUL`` for a
    ``DOUBLE_POWER`` move whose condition holds (``double_power``; those scripts set 10
    otherwise), else ``power_mul``. The trainer AI's estimate and the confusion self-hit run no
    move script."""
    atk_side = _i32(atk_side)
    reckless = state.ability[atk_side, state.active[atk_side]] == AbilityId.RECKLESS
    recoil = tables.move_effect[_i32(move)] == EffectId.RECOIL
    doubled = double_power(tables, state, atk_side, def_side, move)
    out = jnp.where(reckless & recoil, RECKLESS_POWER_MUL, _i32(power_mul))
    return jnp.where(doubled, DOUBLE_POWER_MUL, out)


def weight_power(tables: Tables, state: BattleState, def_side) -> jnp.ndarray:
    """``int32``: Grass Knot's / Low Kick's power against ``def_side``'s active battler
    (``WEIGHT_BOUNDS``): ``BattleMon.weight`` is the species' Pokedex weight
    (``Pokedex_HeightWeightData_Weight``, battle_lib.c:150), ``tables.species_weight``."""
    weight = _i32(tables.species_weight[state.species[def_side, state.active[def_side]]])
    index = jnp.sum(weight > jnp.asarray(WEIGHT_BOUNDS, _I32))
    return jnp.asarray(WEIGHT_POWER, _I32)[index]


def hp_power(state: BattleState, def_side) -> jnp.ndarray:
    """``int32``: Wring Out's / Crush Grip's power, ``1 + 120 * curHP / maxHP`` of
    ``def_side``'s active battler (``BtlCmd_CalcWringOutPower``, battle_script.c:7265)."""
    mon = state.active[def_side]
    max_hp = jnp.maximum(_i32(state.max_hp[def_side, mon]), 1)
    return 1 + 120 * _i32(state.hp[def_side, mon]) // max_hp


def weather_suppressed(tables: Tables, state: BattleState, active=None) -> jnp.ndarray:
    """``bool[]``: not ``NO_CLOUD_NINE`` (include/battle/common.h:63-64): an active
    battler with HP left has Cloud Nine or Air Lock (``BattleSystem_CountAbility``,
    ``COUNT_ALIVE_BATTLERS``), which turns off every weather effect. ``active``
    (``int[2]`` party slots, default ``state.active``) names the battlers on the field
    when that differs from the state's, as for a move that meets a Pokemon switched in
    this turn (``outcome.hit_threshold``)."""
    sides = jnp.arange(2)
    active = state.active if active is None else active
    ability = decomp_abilities(tables, state)[sides, active]
    lock = (ability == WEATHER_ABILITY_IDS["Cloud Nine"]) | (
        ability == WEATHER_ABILITY_IDS["Air Lock"]
    )
    return jnp.any(lock & (state.hp[sides, active] > 0))


def hidden_power(ivs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(type, power)`` of Hidden Power for a battler with ``ivs`` (``int[6]`` in
    ``ids.Stat`` order): ``BtlCmd_CalcHiddenPowerParams`` (battle_script.c:6007-6033).
    The decomp's bit order is HP, Atk, Def, Spe, SpA, SpD: power from bit 1 of each IV,
    ``* 40 / 63 + 30``; type from bit 0, ``* 15 / 63 + 1``, one more from ``TYPE_MYSTERY``
    up. ``TrainerAI_MoveType`` (trainer_ai.c:3208-3220) computes the same type."""
    iv = _i32(ivs)
    hp, atk, dfn = iv[Stat.HP], iv[Stat.ATK], iv[Stat.DEF]
    spe, spa, spd = iv[Stat.SPE], iv[Stat.SPA], iv[Stat.SPD]
    power = ((hp & 2) >> 1) | (atk & 2) | ((dfn & 2) << 1) | ((spe & 2) << 2)
    power = power | ((spa & 2) << 3) | ((spd & 2) << 4)
    kind = (hp & 1) | ((atk & 1) << 1) | ((dfn & 1) << 2) | ((spe & 1) << 3)
    kind = kind | ((spa & 1) << 4) | ((spd & 1) << 5)
    kind = kind * 15 // 63 + 1
    kind = jnp.where(kind >= Type.MYSTERY, kind + 1, kind)
    return kind, power * 40 // 63 + 30


def move_type_of(tables: Tables, state: BattleState, side, move) -> jnp.ndarray:
    """``int32``: the type of ``move`` used by ``side``'s active Pokemon: Hidden Power's
    IV type, else ``tables.move_type`` (battle_lib.c:6682 ``moveType``)."""
    move = _i32(move)
    side = _i32(side)
    base = _i32(tables.move_type[move])
    iv_type = hidden_power(state.ivs[side, state.active[side]])[0]
    return jnp.where(tables.move_effect[move] == EffectId.HIDDEN_POWER, iv_type, base)


def fixed_damage(
    tables: Tables, state: BattleState, atk_side, move
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(is_fixed, damage)``: a ``LEVEL_DAMAGE`` move (Night Shade, Seismic Toss;
    ``effect_script_0087.s``) deals the attacker's level with ``SYSCTL_IGNORE_TYPE_CHECKS``:
    no crit, roll, STAB or multiplier (``ApplyTypeMultiplier``, battle_lib.c:7544), but the
    chart's immunity bit is still set (:7551), so a type immunity (Foresight-aware) deals
    0.

    Super Fang deals ``Divide(curHP, 2)`` of the defender and an OHKO move its max HP (0
    against an ignorable Sturdy), both through the immunity like Night Shade.

    Bide is fixed too: on its unleash turn (locked with no turns left) it deals
    ``2 * storedDamage`` (``CHECK_STATUS_STATE_BIDE``, battle_controller_player.c:2702); a
    ``RANGE_USER`` move skips the type chart (:2350-2358), so no immunity applies. Any
    other Bide turn deals 0."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    effect = tables.move_effect[_i32(move)]
    is_level = effect == EffectId.LEVEL_DAMAGE
    is_bide = effect == EffectId.BIDE
    level = _i32(state.level[atk_side, state.active[atk_side]])
    kind = move_type_of(tables, state, atk_side, move)
    d_types = state.types[def_side, state.active[def_side]]
    seen = identified(state, atk_side, def_side)
    immune = effectiveness(tables, kind, d_types, seen) == 0
    immune = immune | levitated(state, atk_side, def_side, kind)
    unleash = (state.bide_turns[atk_side] == 0) & (state.locked_move[atk_side] != 0)
    stored = jnp.where(unleash, 2 * _i32(state.bide_damage[atk_side]), 0)
    # Super Fang (effect_script_0040.s): Divide(curHP, 2) of the defender under
    # SYSCTL_IGNORE_TYPE_CHECKS; ApplyTypeMultiplier still flags an immunity (battle_lib.c:
    # 7544-7552), so a Ghost takes 0.
    target = state.active[def_side]
    is_halve = effect == EffectId.HALVE_HP
    halved = divide(_i32(state.hp[def_side, target]), 2)
    # Fissure / Guillotine / Horn Drill: BtlCmd_TryOHKOMove (battle_script.c:4334-4386) sets
    # MOVE_STATUS_ONE_HIT_KO and UpdateHP deals the defender's maxHP (bcp.c:3343-3345); the
    # chart (power 1, bcp.c:2352-2367: Ghost, Flying, Levitate) and an ignorable Sturdy
    # (MOVE_STATUS_STURDY, :4340-4341) stop it. Its hit test is outcome._rate's.
    is_ohko = effect == EffectId.OHKO
    sturdy = ignorable(state, atk_side, def_side) == AbilityId.STURDY
    ohko = jnp.where(sturdy, 0, _i32(state.max_hp[def_side, target]))
    amount = jnp.where(is_ohko, ohko, jnp.where(is_halve, halved, level))
    # Future Sight / Doom Desire (effect_script_0148.s): no CalcDamage at the use, so no damage;
    # the stored hit is step._future_sight's.
    is_future = effect == EffectId.FUTURE_SIGHT
    fixed = is_level | is_bide | is_halve | is_ohko | is_future
    amount = jnp.where(is_future, 0, amount)
    return fixed, jnp.where(is_bide, stored, jnp.where(immune, 0, amount))


def rollout_multiplier(state: BattleState, atk_side) -> jnp.ndarray:
    """``int32``: the power multiplier of this use of Rollout / Ice Ball
    (``BtlCmd_CalcRolloutPower``, battle_script.c:5661). A first use starts the lock at
    ``rolloutCount = 5``, and each use decrements it before the power is read:
    ``2 ** (4 - rolloutCount)``, so 1, 2, 4, 8, 16 on the five uses; a curled user
    (``VOLATILE_CONDITION_DEFENSE_CURL``) doubles it again."""
    atk_side = _i32(atk_side)
    locked = state.locked_move[atk_side] != 0
    left = jnp.where(locked, jnp.maximum(_i32(state.rollout_count[atk_side]) - 1, 0), 4)
    curled = jnp.where(state.curled[atk_side] != 0, 2, 1)
    return jnp.left_shift(1, jnp.clip(4 - left, 0, 4)) * curled


def divide(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """``BattleSystem_Divide`` (battle_lib.c:3599) for non-negative ``x``."""
    return jnp.where(x == 0, 0, jnp.maximum(x // d, 1))


def apply_stage(stat: jnp.ndarray, stage: jnp.ndarray) -> jnp.ndarray:
    """``stat * sStatStageBoosts[stage].numerator / .denominator`` (battle_lib.c:6956-6957;
    the same table scales Speed in ``BattleSystem_CompareBattlerSpeed``, :1225)."""
    return stat * _STAGE_NUM[stage + 6] // _STAGE_DEN[stage + 6]


def simple_stage(stage: jnp.ndarray) -> jnp.ndarray:
    """Simple: stage doubled, clamped to [-6, 6] (battle_lib.c:6850-6884, :1171)."""
    return jnp.clip(stage * 2, -6, 6)


def chart_entry(tables: Tables, move_type, def_type, identified=False) -> jnp.ndarray:
    """``int32`` chart entry in tenths. The rows after the 0xFE separator (the Ghost
    immunities) are skipped for an ``identified`` defender (Foresight, battle_lib.c:2611),
    i.e. count as neutral."""
    mul = _i32(tables.type_chart[move_type, def_type])
    skipped = jnp.asarray(identified, jnp.bool_) & (
        tables.type_chart_foresight[move_type, def_type] != 0
    )
    return jnp.where(skipped, 10, mul)


def effectiveness(tables: Tables, move_type, def_types, identified=False) -> jnp.ndarray:
    """Product of the two chart entries in tenths (neutral ``100``, immune ``0``).
    Type 2 counts only when it differs from type 1 (battle_lib.c:2635). ``identified``
    is the defender's Foresight flag."""
    t1, t2 = _i32(def_types[0]), _i32(def_types[1])
    e1 = chart_entry(tables, move_type, t1, identified)
    e2 = jnp.where(t2 != t1, chart_entry(tables, move_type, t2, identified), 10)
    return e1 * e2


def _item_modifiers(
    tables: Tables, s: BattleState, atk_side, def_side, move_type, category, power, stats
):
    """``(power, (attack, sp_attack, defense, sp_defense))`` after the held-item block of
    ``BattleSystem_CalcMoveDamage`` (battle_lib.c:6716-6783), in its order. The items are
    ``BattleSystem_GetItemData`` of ``Battler_HeldItem`` (:6662-6668; Klutz and Embargo,
    which hide it, are unsupported), the attacker's and the defender's (the attacker's own
    for the confusion self-hit). ``movePower`` and the stats are ``u16`` there; no value the
    engine reaches comes near 65536, so the stores never truncate. Soul Dew's Battle Frontier
    test (:6731, :6736) never holds for a trainer battle; the Griseous Orb's Transform test
    (:6770) never holds while Transform is an unsupported move."""
    attack, sp_attack, defense, sp_defense = stats
    a_slot, d_slot = s.active[atk_side], s.active[def_side]
    a_item, d_item = _i32(s.item[atk_side, a_slot]), _i32(s.item[def_side, d_slot])
    a_effect, d_effect = _i32(tables.item_effect[a_item]), _i32(tables.item_effect[d_item])
    a_param = _i32(tables.item_param[a_item, 0])  # attackerParams.heldItemPower (:6664)
    a_species, d_species = _i32(s.species[atk_side, a_slot]), _i32(s.species[def_side, d_slot])

    def boost(x):  # movePower * (100 + heldItemPower) / 100
        return x * (100 + a_param) // 100

    # :6716-6722 sTypeBoostingItems: the item's type (item_param[i, 1]) is the move's.
    boosted = (a_effect == ItemId.TYPE_BOOST) & (move_type == _i32(tables.item_param[a_item, 1]))
    power = jnp.where(boosted, boost(power), power)
    # :6724-6729 Choice Band / Choice Specs.
    attack = jnp.where(a_effect == ItemId.CHOICE_BAND, attack * 150 // 100, attack)
    sp_attack = jnp.where(a_effect == ItemId.CHOICE_SPECS, sp_attack * 150 // 100, sp_attack)
    # :6730-6739 Soul Dew: a Latias / Latios attacker's Sp. Atk, defender's Sp. Def.
    a_lati = (a_species == SPECIES_LATIOS) | (a_species == SPECIES_LATIAS)
    d_lati = (d_species == SPECIES_LATIOS) | (d_species == SPECIES_LATIAS)
    sp_attack = jnp.where((a_effect == ItemId.SOUL_DEW) & a_lati, sp_attack * 150 // 100, sp_attack)
    sp_defense = jnp.where(
        (d_effect == ItemId.SOUL_DEW) & d_lati, sp_defense * 150 // 100, sp_defense
    )
    # :6740-6747 DeepSeaTooth / DeepSeaScale (Clamperl).
    a_clamperl, d_clamperl = a_species == SPECIES_CLAMPERL, d_species == SPECIES_CLAMPERL
    sp_attack = jnp.where(
        (a_effect == ItemId.DEEP_SEA_TOOTH) & a_clamperl, sp_attack * 2, sp_attack
    )
    sp_defense = jnp.where(
        (d_effect == ItemId.DEEP_SEA_SCALE) & d_clamperl, sp_defense * 2, sp_defense
    )
    # :6748-6751 Light Ball (Pikachu): movePower * 2, whatever the move's class.
    power = jnp.where(
        (a_effect == ItemId.LIGHT_BALL) & (a_species == SPECIES_PIKACHU), power * 2, power
    )
    # :6752-6755 Metal Powder (a defending Ditto).
    defense = jnp.where(
        (d_effect == ItemId.METAL_POWDER) & (d_species == SPECIES_DITTO), defense * 2, defense
    )
    # :6756-6759 Thick Club (Cubone, Marowak).
    cubone = (a_species == SPECIES_CUBONE) | (a_species == SPECIES_MAROWAK)
    attack = jnp.where((a_effect == ItemId.THICK_CLUB) & cubone, attack * 2, attack)
    # :6760-6775 the orbs: Dragon or the species' own second type.
    dragon = move_type == Type.DRAGON
    orb = (
        (
            (a_effect == ItemId.ADAMANT_ORB)
            & (dragon | (move_type == Type.STEEL))
            & (a_species == SPECIES_DIALGA)
        )
        | (
            (a_effect == ItemId.LUSTROUS_ORB)
            & (dragon | (move_type == Type.WATER))
            & (a_species == SPECIES_PALKIA)
        )
        | (
            (a_effect == ItemId.GRISEOUS_ORB)
            & (dragon | (move_type == Type.GHOST))
            & (a_species == SPECIES_GIRATINA)
        )
    )
    power = jnp.where(orb, boost(power), power)
    # :6776-6783 Muscle Band (physical) / Wise Glasses (special): MOVE_DATA(move).class.
    band = ((a_effect == ItemId.MUSCLE_BAND) & (category == PHYSICAL)) | (
        (a_effect == ItemId.WISE_GLASSES) & (category == SPECIAL)
    )
    power = jnp.where(band, boost(power), power)
    return power, (attack, sp_attack, defense, sp_defense)


def life_orb(tables: Tables, s: BattleState, atk_side, damage) -> jnp.ndarray:
    """``int32``: ``damage`` (after the crit multiplier) through the attacker's Life Orb,
    ``BattleScript_CalcMoveDamage`` (battle_script.c:1346-1348): ``damage * (100 + param) /
    100`` for ``HOLD_EFFECT_HP_DRAIN_ON_ATK`` (``Battler_HeldItemPower``, Life Orb's 30)."""
    atk_side = _i32(atk_side)
    item = _i32(s.item[atk_side, s.active[atk_side]])
    orb = tables.item_effect[item] == ItemId.LIFE_ORB
    return jnp.where(orb, damage * (100 + _i32(tables.item_param[item, 0])) // 100, damage)


def _pre_random(
    tables: Tables,
    s: BattleState,
    atk_side,
    move,
    crit,
    *,
    power_mul=10,
    self_hit: bool = False,
    magnitude=MAGNITUDE_MIN,
    present=PRESENT_TABLE,
    ai: bool = False,
):
    """Damage after ``+ 2`` and the crit multiplier, before the random roll.

    ``power_mul`` is ``battleCtx->powerMul`` in tenths (Pursuit on a switch: 20).
    ``self_hit`` is the confusion self-hit, ``CALC_SELF_HIT(MOVE_STRUGGLE, 40)``
    (``include/battle/common.h:35``): the attacker is its own defender, power 40, and
    no side or field conditions are passed (no screens, no weather). ``magnitude`` is
    Magnitude's level (``Outcome.magnitude``); ``present`` is Present's draw
    (``Outcome.present``: 0-2 is the power 40/80/120, any other value keeps the table
    power). ``ai`` is ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109), which has no Flail,
    Rollout or Present case: they keep the move's table power."""
    atk_side = _i32(atk_side)
    def_side = atk_side if self_hit else 1 - atk_side
    move = _i32(move)
    crit = jnp.asarray(crit, jnp.bool_)
    a_slot, d_slot = _i32(s.active[atk_side]), _i32(s.active[def_side])

    base_power = _i32(tables.move_power[move])
    # :6674 movePower = inPower when given (the self-hit's 40), else MOVE_DATA(move).power
    effect = _i32(tables.move_effect[move])
    # CalcHiddenPowerParams (battle_script.c:6007) / CalcMagnitudePower (:5820) set movePower.
    # CalcRolloutPower (:5661) doubles movePower per use, and again when curled.
    # CalcFlailPower (:4958) reads the attacker's current and max HP.
    rollout_mul = 1 if ai else rollout_multiplier(s, atk_side)
    # CalcWringOutPower (battle_script.c:7265) and CalcWeightBasedPower (:6808-6826) set
    # movePower from the defender; the AI's estimate keeps the table power (no case here).
    defender_power = jnp.where(
        effect == EffectId.HP_POWER, hp_power(s, def_side), weight_power(tables, s, def_side)
    )
    variable = jnp.where(
        effect == EffectId.HIDDEN_POWER,
        hidden_power(s.ivs[atk_side, a_slot])[1],
        jnp.where(
            effect == EffectId.ROLLOUT,
            base_power * rollout_mul,
            jnp.where(
                effect == EffectId.FLAIL,
                flail_power(s.hp[atk_side, a_slot], s.max_hp[atk_side, a_slot]),
                MAGNITUDE_POWER[jnp.clip(_i32(magnitude) - MAGNITUDE_MIN, 0, 6)],
            ),
        ),
    )
    has_variable = (
        (effect == EffectId.HIDDEN_POWER)
        | (effect == EffectId.MAGNITUDE)
        | (effect == EffectId.ROLLOUT)
        | ((effect == EffectId.FLAIL) & (not ai))
    )
    power = jnp.where(has_variable, variable, base_power)
    by_defender = ((effect == EffectId.HP_POWER) | (effect == EffectId.WEIGHT_POWER)) & (not ai)
    power = jnp.where(by_defender, defender_power, power)
    # BtlCmd_Present (battle_script.c:5795-5800) sets movePower before CalcDamage.
    present = _i32(present)
    drawn = (effect == EffectId.PRESENT) & (present >= 0) & (present < PRESENT_HEAL) & (not ai)
    power = jnp.where(drawn, PRESENT_POWER[jnp.clip(present, 0, PRESENT_HEAL - 1)], power)
    power = jnp.asarray(CONFUSION_POWER, _I32) if self_hit else power
    power = power * _i32(power_mul) // 10  # :6688 movePower * powerMul / 10
    # :6698-6702 Technician (attackerParams.ability): movePower <= 60 after powerMul (Charge
    # and Helping Hand, between them, are unsupported), never Struggle (the self-hit's move).
    technician = (_i32(s.ability[atk_side, a_slot]) == AbilityId.TECHNICIAN) & (power <= 60)
    technician = technician & (move != MOVE_STRUGGLE)
    power = jnp.where(technician, power * 15 // 10, power)
    # :6682 moveType = MOVE_DATA(move).type, or Hidden Power's IV type (battle_script.c:6028)
    move_type = move_type_of(tables, s, atk_side, move)
    category = _i32(tables.move_category[move])  # :6704 moveClass
    level = _i32(s.level[atk_side, a_slot])
    a_ability = _i32(s.ability[atk_side, a_slot])
    d_ability = _i32(s.ability[def_side, d_slot])
    # Battler_IgnorableAbility (:3108): Mold Breaker ignores the defender's ability.
    d_ability_eff = jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, d_ability)
    a_status = _i32(s.status[atk_side, a_slot])
    a_hp, a_max_hp = _i32(s.hp[atk_side, a_slot]), _i32(s.max_hp[atk_side, a_slot])
    a_gender, d_gender = _i32(s.gender[atk_side, a_slot]), _i32(s.gender[def_side, d_slot])
    d_types = _i32(s.types[def_side, d_slot])

    a_stats, d_stats = _i32(s.stats[atk_side, a_slot]), _i32(s.stats[def_side, d_slot])
    attack, sp_attack = a_stats[Stat.ATK - 1], a_stats[Stat.SPA - 1]  # :6635, :6637
    defense, sp_defense = d_stats[Stat.DEF - 1], d_stats[Stat.SPD - 1]  # :6636, :6638
    a_stages, d_stages = _i32(s.stages[atk_side]), _i32(s.stages[def_side])
    attack_stage, sp_attack_stage = a_stages[Stat.ATK], a_stages[Stat.SPA]  # :6639, :6641
    defense_stage, sp_defense_stage = d_stages[Stat.DEF], d_stages[Stat.SPD]  # :6640, :6642

    # :6716-6783 the held items, before every ability modifier below.
    power, (attack, sp_attack, defense, sp_defense) = _item_modifiers(
        tables, s, atk_side, def_side, move_type, category, power,
        (attack, sp_attack, defense, sp_defense),
    )  # fmt: skip

    # :6785-6788 Thick Fat (Battler_IgnorableAbility): a Fire or Ice move's power halved.
    thick_fat = (d_ability_eff == AbilityId.THICK_FAT) & (
        (move_type == Type.FIRE) | (move_type == Type.ICE)
    )
    power = jnp.where(thick_fat, power // 2, power)

    # :6793 Guts: Atk x1.5 when statused.
    attack = jnp.where(
        (a_ability == AbilityId.GUTS) & (a_status != Status.NONE), attack * 150 // 100, attack
    )

    # :6811 Mud Sport / :6815 Water Sport: any active battler's sport halves the power.
    power = jnp.where((move_type == Type.ELECTRIC) & jnp.any(s.mud_sport != 0), power // 2, power)
    power = jnp.where((move_type == Type.FIRE) & jnp.any(s.water_sport != 0), power // 2, power)

    # :6819-6837 Overgrow / Blaze / Torrent: power x1.5 at curHP <= maxHP / 3.
    pinch = (
        ((move_type == Type.GRASS) & (a_ability == AbilityId.OVERGROW))
        | ((move_type == Type.FIRE) & (a_ability == AbilityId.BLAZE))
        | ((move_type == Type.WATER) & (a_ability == AbilityId.TORRENT))
    ) & (a_hp <= a_max_hp // 3)
    power = jnp.where(pinch, power * 150 // 100, power)
    # :6845-6848 Dry Skin (Battler_IgnorableAbility): a Fire move's power x125/100 (Heatproof,
    # just before it, is unsupported).
    dry_skin = (d_ability_eff == AbilityId.DRY_SKIN) & (move_type == Type.FIRE)
    power = jnp.where(dry_skin, power * 125 // 100, power)

    # :6850 attacker Simple doubles its offensive stages.
    a_simple = a_ability == AbilityId.SIMPLE
    attack_stage = jnp.where(a_simple, simple_stage(attack_stage), attack_stage)
    sp_attack_stage = jnp.where(a_simple, simple_stage(sp_attack_stage), sp_attack_stage)
    # :6868 defender Simple (ignorable) doubles its defensive stages.
    d_simple = d_ability_eff == AbilityId.SIMPLE
    defense_stage = jnp.where(d_simple, simple_stage(defense_stage), defense_stage)
    sp_defense_stage = jnp.where(d_simple, simple_stage(sp_defense_stage), sp_defense_stage)
    # :6886 defender Unaware (ignorable) ignores the attacker's stages.
    d_unaware = d_ability_eff == AbilityId.UNAWARE
    attack_stage = jnp.where(d_unaware, 0, attack_stage)
    sp_attack_stage = jnp.where(d_unaware, 0, sp_attack_stage)
    # :6891 attacker Unaware ignores the defender's stages.
    a_unaware = a_ability == AbilityId.UNAWARE
    defense_stage = jnp.where(a_unaware, 0, defense_stage)
    sp_defense_stage = jnp.where(a_unaware, 0, sp_defense_stage)

    # :6901 / :6907 Rivalry: x1.25 same gender, x0.75 opposite, nothing if either is genderless.
    gendered = (a_ability == AbilityId.RIVALRY) & (a_gender != 2) & (d_gender != 2)
    power = jnp.where(gendered & (a_gender == d_gender), power * 125 // 100, power)
    power = jnp.where(gendered & (a_gender != d_gender), power * 75 // 100, power)

    # :6921 NO_CLOUD_NINE gates the weather modifiers here and at :7046 (the self-hit
    # passes no field conditions).
    weather = jnp.where(weather_suppressed(tables, s), Weather.NONE, s.weather)
    # :6922-6924 Solar Power (attackerParams.ability) in sun: Sp. Atk x15/10 (the self-hit
    # passes no field conditions).
    solar = (weather == Weather.SUN) & (a_ability == AbilityId.SOLAR_POWER) & (not self_hit)
    sp_attack = jnp.where(solar, sp_attack * 15 // 10, sp_attack)
    # :6926 sandstorm: a Rock-type defender's SpD x1.5.
    rock = jnp.any(d_types == Type.ROCK)
    sand = (weather == Weather.SANDSTORM) & rock & (not self_hit)
    sp_defense = jnp.where(sand, sp_defense * 15 // 10, sp_defense)
    # :6943-6945 Selfdestruct / Explosion (BATTLE_EFFECT_HALVE_DEFENSE) halve defenseStat,
    # before the stage scales it (coverage refuses a Defense the -6 stage would bring to 0).
    defense = jnp.where(effect == EffectId.SELF_DESTRUCT, defense // 2, defense)

    # :6947 / :6992 physical uses Atk/Def, special SpA/SpD.
    is_physical = category == PHYSICAL
    a_stat = jnp.where(is_physical, attack, sp_attack)
    a_stage = jnp.where(is_physical, attack_stage, sp_attack_stage)
    d_stat = jnp.where(is_physical, defense, sp_defense)
    d_stage = jnp.where(is_physical, defense_stage, sp_defense_stage)
    # :6948-6958 a crit ignores the attacker's non-positive stages;
    # :6963-6973 and the defender's non-negative ones.
    a_stage = jnp.where(crit & (a_stage <= 0), 0, a_stage)
    d_stage = jnp.where(crit & (d_stage >= 0), 0, d_stage)

    damage = apply_stage(a_stat, a_stage)  # :6956-6957
    damage = damage * power  # :6960
    damage = damage * (level * 2 // 5 + 2)  # :6961
    damage = damage // apply_stage(d_stat, d_stage)  # :6975
    damage = damage // 50  # :6976

    # :6978 burn halves physical damage unless Guts.
    burned = is_physical & (a_status == Status.BURN) & (a_ability != AbilityId.GUTS)
    damage = jnp.where(burned, damage // 2, damage)
    # :6982 Reflect / :7023 Light Screen on the defender's side halve it, not on a crit.
    screen = jnp.where(is_physical, s.reflect[def_side], s.light_screen[def_side]) != 0
    screen = screen & (not self_hit) & (effect != EffectId.BRICK_BREAK)  # :6984, :7025
    damage = jnp.where(screen & ~crit, damage // 2, damage)

    # :7047 rain: Fire x0.5, Water x1.5; :7062 sun: Fire x1.5, Water x0.5. Fog: nothing.
    rain = (weather == Weather.RAIN) & (not self_hit)
    sun = (weather == Weather.SUN) & (not self_hit)
    fire, water = move_type == Type.FIRE, move_type == Type.WATER
    damage = jnp.where(rain & fire, damage // 2, damage)
    damage = jnp.where(rain & water, damage * 15 // 10, damage)
    damage = jnp.where(sun & fire, damage * 15 // 10, damage)
    damage = jnp.where(sun & water, damage // 2, damage)

    damage = damage + 2  # :7078
    # battle_script.c:1344, x3 for the attacker's Sniper (battle_lib.c:7142-7144).
    crit_mul = jnp.where(a_ability == AbilityId.SNIPER, SNIPER_CRIT_MUL, CRIT_MUL)
    damage = jnp.where(crit, damage * crit_mul, damage)
    return damage, move_type, category, base_power


def _finish(
    tables: Tables,
    s: BattleState,
    atk_side,
    move,
    damage,
    move_type,
    category,
    power,
    roll,
    *,
    variance_last: bool = False,
):
    """Random roll, STAB and type chart; 0 for status moves and immunities. Struggle
    skips STAB and the chart (``ApplyTypeChart`` returns early for it, :2573).

    The battle applies the variance first (``BattleSystem_CalcDamageVariance``, :7081,
    at least 1), then the chart. ``variance_last`` is the trainer AI's order instead
    (``TrainerAI_CalcDamage``, trainer_ai.c:3096-3109): STAB and the chart first, then
    ``BattleSystem_Divide(damage * variance, 100)``."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    a_types = _i32(s.types[atk_side, s.active[atk_side]])
    d_types = _i32(s.types[def_side, s.active[def_side]])
    a_ability = _i32(s.ability[atk_side, s.active[atk_side]])
    # Battler_IgnorableAbility (:3108): Mold Breaker ignores the defender's ability.
    d_ability = ignorable(s, atk_side, def_side)
    variance = 85 + _i32(roll)

    if not variance_last:
        # battle_lib.c:7084-7089 variance, at least 1.
        damage = jnp.maximum(damage * variance // 100, 1)
    typeless = _i32(move) == MOVE_STRUGGLE
    unchanged = damage
    # :2592-2597 STAB: x2 with Adaptability (Battler_Ability), else x1.5.
    stab = jnp.where(a_ability == AbilityId.ADAPTABILITY, damage * 2, damage * 15 // 10)
    damage = jnp.where(jnp.any(a_types == move_type), stab, damage)
    # :2611-2646 each defender type through ApplyTypeMultiplier (:7547), in chart
    # row order; type 2 is skipped when it equals type 1 (:2635). Unlisted pairs
    # are neutral (x10/10 leaves damage unchanged), so their order is irrelevant.
    t1, t2 = d_types[0], d_types[1]
    # :2611-2615 the 0xFE batch break: Foresight, or the attacker's Scrappy.
    seen = identified(s, atk_side, def_side)
    e1 = chart_entry(tables, move_type, t1, seen)
    e2 = jnp.where(t2 != t1, chart_entry(tables, move_type, t2, seen), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    first, second = jnp.where(t2_first, e2, e1), jnp.where(t2_first, e1, e2)
    damage = divide(divide(damage * first, 10) * second, 10)
    # :2657-2660 Solid Rock (Battler_IgnorableAbility): BattleSystem_Divide(damage * 3, 4)
    # when MOVE_STATUS_SUPER_EFFECTIVE is set. ApplyTypeMultiplier (:7557-7574) sets it for a
    # x2 entry unless NOT_VERY_EFFECTIVE was set, which it clears instead (and the other way
    # round), so it stands exactly when the product is above neutral; an immunity leaves 0.
    super_effective = first * second > 100
    # Filter shares Solid Rock's line (:2657-2661, Battler_IgnorableAbility).
    solid_rock = super_effective & (
        (d_ability == AbilityId.SOLID_ROCK) | (d_ability == AbilityId.FILTER)
    )
    damage = jnp.where(solid_rock, divide(damage * 3, 4), damage)
    # :2663-2665 then the attacker's Expert Belt (Battler_HeldItemEffect): damage *
    # (100 + param) / 100 on the same super-effective hit (attackerItemPower, :2578).
    a_item = _i32(s.item[atk_side, s.active[atk_side]])
    belt = super_effective & (tables.item_effect[a_item] == ItemId.EXPERT_BELT)
    damage = jnp.where(belt, damage * (100 + _i32(tables.item_param[a_item, 0])) // 100, damage)
    # :2668-2672 Tinted Lens (Battler_Ability) doubles a MOVE_STATUS_NOT_VERY_EFFECTIVE hit,
    # which ApplyTypeMultiplier sets, like the super-effective flag above, exactly when the
    # product is below neutral (an immunity leaves 0).
    not_very = first * second < 100
    damage = jnp.where(not_very & (a_ability == AbilityId.TINTED_LENS), damage * 2, damage)
    # :2600-2603 Levitate: MOVE_STATUS_LEVITATED instead of the chart walk, part of
    # MOVE_STATUS_IMMUNE: no damage (its DID_NOT_HIT ends the move, bcp.c:3325; the AI's
    # estimate zeroes an immune hit, trainer_ai.c:3106).
    damage = jnp.where(levitated(s, atk_side, def_side, move_type), 0, damage)
    damage = jnp.where(typeless, unchanged, damage)
    if variance_last:
        # trainer_ai.c:3109; an immunity is already 0 (:3106).
        damage = divide(damage * variance, 100)
    # Status moves never calculate damage; CheckTypeChart needs base power
    # (battle_controller_player.c:2354).
    return jnp.where((category == STATUS) | (power == 0), 0, damage)


def damage_table(tables: Tables, state: BattleState, atk_side, move, crit) -> jnp.ndarray:
    """``int32[16]``: damage of ``move`` by ``atk_side``'s active Pokemon against the
    other side's active Pokemon for rolls 0..15 (``85 + roll`` percent). A ``LEVEL_DAMAGE``
    move has no roll: all 16 entries are ``fixed_damage``. Magnitude is taken at level 4
    (its weakest hit; the playout heuristic's documented under-estimate); Bide's unleash
    stays 0 (it is a forced locked action, never a chosen attack). Present keeps its table
    power (no draw: ``PRESENT_TABLE``)."""
    # The defender is the other side's active Pokemon (this table's contract).
    power_mul = script_power_mul(tables, state, atk_side, 1 - _i32(atk_side), move, 10)
    damage, move_type, category, power = _pre_random(
        tables, state, atk_side, move, crit, power_mul=power_mul
    )
    damage = life_orb(tables, state, atk_side, damage)  # battle_script.c:1346
    rolls = jnp.arange(ROLLS, dtype=_I32)
    table = _finish(tables, state, atk_side, move, damage, move_type, category, power, rolls)
    fixed, amount = fixed_damage(tables, state, atk_side, move)
    amount = jnp.where(tables.move_effect[_i32(move)] == EffectId.BIDE, 0, amount)
    return jnp.where(fixed, amount, table)


def hit_damage_table(tables: Tables, state: BattleState, atk_side, move) -> jnp.ndarray:
    """``int32[2, ROLLS]``: one hit's damage for every ``(crit, roll)`` pair (``crit`` 0
    or 1, then ``damage_table``'s rolls), at ``state``. It equals ``damage_for_roll`` at
    the default ``power_mul`` for every move but Magnitude (``damage_table`` takes level 4)
    and Bide (0 here); ``enumerate``'s damage classes use neither."""
    return jnp.stack([damage_table(tables, state, atk_side, move, crit) for crit in (False, True)])


def damage_for_roll(
    tables: Tables,
    state: BattleState,
    atk_side,
    move,
    crit,
    roll,
    power_mul=10,
    magnitude=MAGNITUDE_MIN,
    present=PRESENT_TABLE,
) -> jnp.ndarray:
    """``int32``: ``damage_table(...)[roll]``, the single value the engine uses.
    ``power_mul`` (tenths) scales the move's power first (Pursuit on a switch: 20);
    ``magnitude`` is Magnitude's level; ``present`` is Present's draw, and its heal path
    (``PRESENT_HEAL``) deals 0: effect_script_0122.s jumps past ``CalcCrit`` and
    ``CalcDamage``."""
    damage, move_type, category, power = _pre_random(
        tables,
        state,
        atk_side,
        move,
        crit,
        # Reckless, the DOUBLE_POWER scripts; the defender is the other side's active Pokemon.
        power_mul=script_power_mul(tables, state, atk_side, 1 - _i32(atk_side), move, power_mul),
        magnitude=magnitude,
        present=present,
    )
    damage = life_orb(tables, state, atk_side, damage)  # battle_script.c:1346
    rolled = _finish(tables, state, atk_side, move, damage, move_type, category, power, _i32(roll))
    fixed, amount = fixed_damage(tables, state, atk_side, move)
    # The only place Present's heal path deals 0: ``_pre_random`` above still ran at the
    # table power (it ignores a ``present`` outside 0-2), so a caller that needs the heal's
    # damage must come through here, not ``_pre_random`` / ``calc_move_damage``.
    heal = (_i32(tables.move_effect[_i32(move)]) == EffectId.PRESENT) & (
        _i32(present) == PRESENT_HEAL
    )
    return jnp.where(heal, 0, jnp.where(fixed, amount, rolled))


def calc_move_damage(
    tables: Tables, state: BattleState, atk_side, move, power_mul=10
) -> jnp.ndarray:
    """``int32``: ``BattleSystem_CalcMoveDamage`` (battle_lib.c:6601) alone, with
    ``criticalMul`` 1: step 1 above, ending with the ``+ 2`` (so a power-0 move gives 2).
    No variance, STAB, type chart or Life Orb. ``power_mul`` is ``battleCtx->powerMul`` in
    tenths."""
    return _pre_random(tables, state, atk_side, move, False, power_mul=power_mul)[0]


def ai_damage_for_roll(tables: Tables, state: BattleState, atk_side, move, roll) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109) for a move with no
    special calculation: ``BattleSystem_CalcMoveDamage`` with ``criticalMul`` 1 (screens,
    weather, abilities, items and stages as in battle), then STAB and the type chart,
    then the variance ``85 + roll`` percent last, at least 1 when non-zero."""
    damage, move_type, category, power = _pre_random(tables, state, atk_side, move, False, ai=True)
    return _finish(
        tables,
        state,
        atk_side,
        move,
        damage,
        move_type,
        category,
        power,
        _i32(roll),
        variance_last=True,
    )


def confusion_damage(tables: Tables, state: BattleState, side, roll) -> jnp.ndarray:
    """``int32``: the confusion self-hit of ``side``'s active Pokemon
    (battle_controller_player.c:2633-2634): ``CALC_SELF_HIT(MOVE_STRUGGLE, 40)``, a
    Normal physical hit on itself with its own stats and stages and no crit, then
    ``BattleSystem_CalcDamageVariance`` with ``roll``. No STAB or type chart
    (``ApplyTypeChart`` is not called)."""
    damage, _, _, _ = _pre_random(tables, state, side, MOVE_STRUGGLE, False, self_hit=True)
    return jnp.maximum(damage * (85 + _i32(roll)) // 100, 1)
