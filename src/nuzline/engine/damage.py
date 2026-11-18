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
    MOVE_SOLAR_BEAM,
    MOVE_STRUGGLE,
    POWER_IF_TARGET_HALF_HP,
    POWER_IF_TARGET_PARALYZED,
    POWER_IF_USER_STATUSED,
    SEMI_HIT_FLAGS,
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
    STOMP_PARAM,
    UNTRACEABLE_ABILITIES,
    WEATHER_ABILITIES,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
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
    ``tables.species_abilities_decomp`` at the Pokemon's ability row (``ability_form``) and
    ability slot; for
    a Trace holder whose ``ability`` is no longer Trace (it traced, ``hooks.trace``), the
    copied ability's (``ids.ABILITY_DECOMP_IDS``)."""
    row = _i32(tables.species_form_row[_i32(state.species), _i32(state.ability_form)])
    slot = _i32(state.ability_slot)
    own = _i32(tables.species_abilities_decomp[row, slot])
    trace = ABILITY_DECOMP_IDS[AbilityId.TRACE]
    traced = (own == trace) & (_i32(state.ability) != AbilityId.TRACE)
    return jnp.where(traced, _ABILITY_DECOMP[_i32(state.ability)], own)


def battle_stats(state: BattleState, side, slot) -> jnp.ndarray:
    """``int32[5]``: the BattleMon Attack, Defense, Sp. Atk, Sp. Def and Speed of ``side``'s party
    ``slot``: its stats, Attack and Defense swapped while it is the active battler and its swap
    is on: MOVE_EFFECT_POWER_TRICK (subscript_user_swap_atk_and_def.s:6-9; battle_lib.c:2137-2141
    for a Baton Pass newcomer) unless a level-up's reload undid it (``power_trick_desync``,
    battle_lib.c:169-174), B12-10."""
    stats = _i32(state.stats[side, slot])
    on = (state.power_trick[side] != 0) != (state.power_trick_desync[side] != 0)
    swapped = on & (state.active[side] == slot)
    trick = (
        stats.at[Stat.ATK - 1].set(stats[Stat.DEF - 1]).at[Stat.DEF - 1].set(stats[Stat.ATK - 1])
    )
    return jnp.where(swapped, trick, stats)


def ignorable(state: BattleState, atk_side, def_side, slot=None) -> jnp.ndarray:
    """``int32``: ``def_side``'s ability (its active battler's, or party ``slot``'s) as
    ``atk_side``'s active battler sees it: ``Battler_IgnorableAbility`` (battle_lib.c:
    3108-3121) hides it behind the attacker's Mold Breaker."""
    active = slot is None
    slot = state.active[def_side] if slot is None else slot
    attacker = _i32(state.ability[atk_side, state.active[atk_side]])
    ability = _i32(state.ability[def_side, slot])
    if active:
        # Battler_Ability (battle_lib.c:3100-3103): an ingrained battler's Levitate is
        # ABILITY_NONE. Only the active battler's: a party slot being read is a switch action's
        # newcomer (the laws), which carries no Ingrain (only Baton Pass keeps it).
        rooted = (state.ingrain[def_side] != 0) & (ability == AbilityId.LEVITATE)
        ability = jnp.where(rooted, AbilityId.NONE, ability)
    return jnp.where(attacker == AbilityId.MOLD_BREAKER, AbilityId.NONE, ability)


def iron_ball(tables: Tables, state: BattleState, side, slot=None) -> jnp.ndarray:
    """``bool``: ``side``'s active battler (or party ``slot``) holds an Iron Ball
    (``HOLD_EFFECT_SPEED_DOWN_GROUNDED`` through ``Battler_HeldItemEffect``; Klutz and Embargo
    are unsupported), ruling G-56."""
    slot = state.active[side] if slot is None else slot
    return tables.item_effect[state.item[side, slot]] == ItemId.IRON_BALL


def levitated(tables: Tables, state: BattleState, atk_side, def_side, move_type) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` sets ``MOVE_STATUS_LEVITATED`` (battle_lib.c:
    2600-2603) for a Ground-type move against a Levitate holder as the attacker sees it
    (``Battler_IgnorableAbility``, whose ``Battler_Ability`` reads an ingrained holder's
    Levitate as none: ``ignorable``) that holds no Iron Ball (:2602, ``iron_ball``, G-56). The
    Gravity exception of ``CalcEffectiveness`` (:2701) is not ``ApplyTypeChart``'s. The flag is
    part of ``MOVE_STATUS_IMMUNE``: no damage (``_finish``)."""
    seen = ignorable(state, atk_side, def_side)
    return (
        (seen == AbilityId.LEVITATE)
        & (_i32(move_type) == Type.GROUND)
        & ~iron_ball(tables, state, def_side)
    )


def magnet_risen(tables: Tables, state: BattleState, def_side, move_type) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` sets ``MOVE_STATUS_MAGNET_RISE`` (battle_lib.c:
    2605-2609, moves.h:77) for a Ground-type move against a defender with ``magnetRiseTurns``,
    no ``MOVE_EFFECT_INGRAIN`` and no Iron Ball (``HOLD_EFFECT_SPEED_DOWN_GROUNDED``). Mold Breaker
    does not lift it; ``CalcEffectiveness`` has no such test. Part of ``MOVE_STATUS_IMMUNE``
    (moves.h:87), so the chart walk is skipped and the hit deals nothing (G-66)."""
    return (
        (state.magnet_rise_turns[def_side] != 0)
        & (state.ingrain[def_side] == 0)
        & (_i32(move_type) == Type.GROUND)
        & ~iron_ball(tables, state, def_side)
    )


def ground_immune(tables: Tables, state: BattleState, atk_side, def_side, move_type) -> jnp.ndarray:
    """``bool``: ``ApplyTypeChart``'s ground immunity of ``def_side``'s battler to ``move_type``,
    the two bits of ``MOVE_STATUS_IMMUNE`` it adds before the walk: ``MOVE_STATUS_LEVITATED``
    (``levitated``) or ``MOVE_STATUS_MAGNET_RISE`` (``magnet_risen``)."""
    return levitated(tables, state, atk_side, def_side, move_type) | magnet_risen(
        tables, state, def_side, move_type
    )


# MoveIsOnDamagingTurn (battle_lib.c:7588-7604): these decomp battle effects
# (generated/move_battle_effects.txt, line - 1) are on their damaging turn only under
# SYSCTL_LAST_OF_MULTI_TURN: BIDE, CHARGE_TURN_HIGH_CRIT, CHARGE_TURN_HIGH_CRIT_FLINCH,
# CHARGE_TURN_DEF_UP, SKIP_CHARGE_TURN_IN_SUN, FLY, DIVE, DIG, BOUNCE and FLINCH_BURN_HIT
# (Fire Fang, listed by the decomp's bug: :7600, docs/bugs_and_glitches.md).
MULTI_TURN_BATTLE_EFFECTS = (26, 39, 75, 145, 151, 155, 255, 256, 263, 273)


def on_damaging_turn(tables: Tables, move, last_of_multi_turn=False) -> jnp.ndarray:
    """``bool``: ``MoveIsOnDamagingTurn`` (battle_lib.c:7588-7604) by the move's decomp battle
    effect (``tables.move_ai_effect``, ``MOVE_DATA(move).effect``): any effect but
    ``MULTI_TURN_BATTLE_EFFECTS``, which pass only under ``SYSCTL_LAST_OF_MULTI_TURN``
    (``last_of_multi_turn``). Only subscript_charge_move_cleanup.s:7 and subscript_bide_end.s:12
    set that flag, and ``BattleContext_Init`` clears it (``SYSCTL_INIT_P1``, battle_lib.c:2011,
    after every action and at the turn's end), so Fire Fang (``BATTLE_EFFECT_FLINCH_BURN_HIT``)
    is never on its damaging turn: in battle and in every trainer-AI read it bypasses Wonder
    Guard (ruling G-18). Bide never reaches the chart in battle (``RANGE_USER``, bcp.c:2352). A
    charge move's strike runs subscript_charge_move_cleanup.s before TryMove's chart, so in
    battle it is on its damaging turn (``charge_strike``); its charge turn skips the chart
    (SYSCTL_FIRST_OF_MULTI_TURN, bcp.c:2356), and every AI read comes after BattleContext_Init,
    where it is not (ruling G-27)."""
    effect = _i32(tables.move_ai_effect[_i32(move)])
    listed = jnp.isin(effect, jnp.asarray(MULTI_TURN_BATTLE_EFFECTS, _I32))
    return ~listed | jnp.asarray(last_of_multi_turn, jnp.bool_)


def charge_strike(tables: Tables, move) -> jnp.ndarray:
    """``bool``: in battle, ``SYSCTL_LAST_OF_MULTI_TURN`` holds when ``move``'s type chart runs.
    Every charge move's script (``MoveFlag.CHARGE``: effect_script_0039.s, _0075.s, _0145.s,
    _0151.s, _0155.s, _0255.s, _0256.s, _0263.s, _0272.s) ends each path that does not charge in
    ``Call BATTLE_SUBSCRIPT_CHARGE_MOVE_CLEANUP`` (the locked second turn, the Power Herb's
    skip, SolarBeam in sun), which sets the flag (subscript_charge_move_cleanup.s:7), and the
    script runs before TryMove (bcp.c:3214-3216); the charging path sets
    SYSCTL_FIRST_OF_MULTI_TURN instead, which skips the chart (bcp.c:2356). No other battle path
    sets it with a chart to run (Bide's subscript_bide_end.s:12: RANGE_USER), ruling G-27."""
    return (_i32(tables.move_flags[_i32(move)]) & MoveFlag.CHARGE) != 0


def super_effective_flag(first, second, update=True) -> jnp.ndarray:
    """``bool``: ``MOVE_STATUS_SUPER_EFFECTIVE`` after ``ApplyTypeMultiplier`` (battle_lib.c:
    7542-7577) meets the chart entries ``first`` then ``second`` (tenths, in chart-row order;
    10 for no entry) from a clear mask: an immune entry clears both effectiveness bits, a x2
    entry clears NOT_VERY_EFFECTIVE when set and else sets SUPER_EFFECTIVE, a x0.5 entry the
    other way round; ``update`` (the move's table power, :2627) gates the x2 / x0.5 updates.
    The two bits are never set together (each set clears the other), so
    ``MOVE_STATUS_BASIC_EFFECTIVENESS`` (both) never holds."""
    se, nve = jnp.bool_(False), jnp.bool_(False)
    update = jnp.asarray(update, jnp.bool_)
    for mul in (_i32(first), _i32(second)):
        halve, double = update & (mul == 5), update & (mul == 20)
        se_next = jnp.where(halve, False, jnp.where(double, ~nve, se))
        nve_next = jnp.where(halve, ~se, jnp.where(double, False, nve))
        se = jnp.where(mul == 0, False, se_next)
        nve = jnp.where(mul == 0, False, nve_next)
    return se


def chart_walk(tables: Tables, state: BattleState, atk_side, def_side, move_type):
    """``(first, second)``: the chart entries (tenths, 10 for none) ``ApplyTypeChart``'s walk
    (battle_lib.c:2611-2646) meets for ``move_type`` against ``def_side``'s active battler, in
    chart-row order (``tables.type_chart_order``); type 2 is skipped when it equals type 1
    (:2635). The walk's flags are ``identified``'s (Foresight, the attacker's Scrappy,
    Ingrain)."""
    d_types = _i32(state.types[def_side, state.active[def_side]])
    t1, t2 = d_types[0], d_types[1]
    seen = identified(tables, state, atk_side, def_side)
    move_type = _i32(move_type)
    e1 = chart_entry(tables, move_type, t1, seen)
    e2 = jnp.where(t2 != t1, chart_entry(tables, move_type, t2, seen), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    return jnp.where(t2_first, e2, e1), jnp.where(t2_first, e1, e2)


def wonder_guarded(
    tables: Tables,
    state: BattleState,
    atk_side,
    move,
    move_type,
    def_side,
    last_of_multi_turn=False,
) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` sets ``MOVE_STATUS_WONDER_GUARD`` (battle_lib.c:
    2649-2654, ruling G-18) for ``move`` (of type ``move_type``) by ``atk_side``'s active battler
    against ``def_side``'s (the defender battler, passed in by the caller): the defender has Wonder Guard as the
    attacker sees it (``Battler_IgnorableAbility``: an attacker's Mold Breaker hides it,
    ``ignorable``), the move is on its damaging turn (``on_damaging_turn`` under
    ``last_of_multi_turn``: Fire Fang never is, a battle caller passes ``charge_strike``), the walk left ``MOVE_STATUS_SUPER_EFFECTIVE`` clear (``super_effective_flag``; an
    immune or neutral hit, a resisted one) and the move has table power (a status move never
    is guarded; the fixed-damage moves have power 1 and are). Struggle returns before
    (:2573). The flag is part of ``MOVE_STATUS_DID_NOT_HIT`` / ``_IMMUNE``: the move does no
    damage and runs subscript_missed.s (bcp.c:3324-3327), but ``ApplyTypeChart`` does not zero
    ``battleCtx->damage`` (the crash of Jump Kick reads it, ``step._use_move``)."""
    atk_side = _i32(atk_side)
    def_side = _i32(def_side)
    move = _i32(move)
    first, second = chart_walk(tables, state, atk_side, def_side, move_type)
    power = tables.move_power[move] != 0
    return (
        (ignorable(state, atk_side, def_side) == AbilityId.WONDER_GUARD)
        & on_damaging_turn(tables, move, last_of_multi_turn)
        & ~super_effective_flag(first, second, power)
        & power
        & (move != MOVE_STRUGGLE)
    )


CHART_IDENTIFIED = 1  # ``chart_entry`` flag: skip the Ghost immunities (Foresight, Scrappy)
CHART_GROUNDED = 2  # ``chart_entry`` flag: skip the Flying immunity (Ingrain)


def identified(tables: Tables, state: BattleState, atk_side, def_side) -> jnp.ndarray:
    """``int32`` flags of ``BattleSystem_ApplyTypeChart``'s walk for ``def_side``'s battler
    (``chart_entry``): ``CHART_IDENTIFIED`` when it stops at the 0xFE separator, skipping the
    Ghost immunities (battle_lib.c:2611-2615), for a Foresight-identified defender or an
    attacker with Scrappy (``Battler_Ability``); ``CHART_GROUNDED`` when ``BasicTypeMulApplies``
    (:2534-2537) skips the rows of an immune Flying defender for an ingrained one or one holding
    an Iron Ball (``iron_ball``, G-56; Gravity's clause, :2545-2549, is unsupported). Only
    ``ApplyTypeChart`` walks the chart this way (:2626, :2636)."""
    scrappy = state.ability[atk_side, state.active[atk_side]] == AbilityId.SCRAPPY
    seen = ((state.foresight[def_side] != 0) | scrappy).astype(_I32) * CHART_IDENTIFIED
    rooted = (state.ingrain[def_side] != 0) | iron_ball(tables, state, def_side)
    return seen | rooted.astype(_I32) * CHART_GROUNDED


def trace_copy(
    tables: Tables, state: BattleState, side, foe, slot=None, target=None
) -> jnp.ndarray:
    """``bool``: ``SWITCH_IN_CHECK_STATE_TRACE`` (battle_lib.c:3720-3735) copies ``foe``'s
    active battler's ability to ``side``'s battler (its active one, or party ``slot`` as it
    enters): that battler has HP, holds no Griseous Orb and has Trace (``Battler_Ability``;
    one that traced has another ability now, so ``traceAnnounced`` is never read apart from
    that), and ``ChooseTraceTarget`` (:7750-7775) finds the foe: alive, its
    ``BattleMon.ability`` not Forecast, Trace or Multitype (decomp ids). In singles both of
    the target's slots name the one foe. ``subscript_trace.s`` then writes the foe's ability
    (``hooks.trace``). ``target``: the foe's party slot to read (default its active one; the
    one it switches in this turn, ``outcome.pivot_view``)."""
    slot = state.active[side] if slot is None else slot
    target = state.active[foe] if target is None else target
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


def grounded(
    tables: Tables, state: BattleState, side, slot, ingrained=None, magnet=None
) -> jnp.ndarray:
    """``bool``: subscript_hazards_check.s:6-11 lets the entry spikes reach ``side``'s party
    ``slot``: Gravity (:6) or its Iron Ball (:7, ``CheckItemHoldEffect``, G-56) grounds it, else its own
    Levitate (``CheckAbility``, battle_script.c:3159-3172, through ``Battler_Ability``, which
    reads an ingrained holder's as none, battle_lib.c:3100-3103), a Flying type
    (``BATTLEMON_TYPE_1`` / ``_2``) or ``MOVE_EFFECT_MAGNET_RISE`` (the mask bit, :11; G-66) keeps
    it off. ``magnet``: the newcomer's Magnet Rise bit at the check, like ``ingrained`` (a Baton
    Pass keeps it).
    ``ingrained``: the newcomer's ``MOVE_EFFECT_INGRAIN`` at the check, by default the side's
    battler's (``ingrain``: after the switch, where only a Baton Pass keeps it, battle_lib.c:2079;
    ruling B7-14); a switch action's newcomer, read before it switches, has none."""
    ingrained = state.ingrain[side] != 0 if ingrained is None else jnp.asarray(ingrained)
    magnet = state.magnet_rise[side] != 0 if magnet is None else jnp.asarray(magnet)
    levitate = (state.ability[side, slot] == AbilityId.LEVITATE) & ~ingrained
    airborne = levitate | jnp.any(state.types[side, slot] == Type.FLYING) | magnet
    return (state.gravity != 0) | iron_ball(tables, state, side, slot) | ~airborne


def spikes_loss(
    tables: Tables, state: BattleState, side, slot, ingrained=None, magnet=None
) -> jnp.ndarray:
    """``int32``: what Spikes takes from ``side``'s party ``slot`` as it switches in
    (``BtlCmd_CheckSpikes``, battle_script.c:5434-5436): ``Divide(maxHP, (5 - layers) * 2)``, 1/8,
    1/6, 1/4, for a grounded newcomer (``grounded``); 0 without layers. Not capped."""
    layers = _i32(state.spikes[side])
    divisor = jnp.maximum((5 - layers) * 2, 1)
    loss = jnp.maximum(_i32(state.max_hp[side, slot]) // divisor, 1)
    return jnp.where((layers > 0) & grounded(tables, state, side, slot, ingrained, magnet), loss, 0)


def hazards_loss(
    tables: Tables, state: BattleState, side, slot, ingrained=None, magnet=None
) -> jnp.ndarray:
    """``int32``: Spikes and Stealth Rock together on ``side``'s newcomer ``slot``
    (subscript_hazards_check.s:31, :41), 0 for a Magic Guard holder (:5); not capped (the
    newcomer survives when this is below its HP: Stealth Rock needs HP left, :41)."""
    loss = spikes_loss(tables, state, side, slot, ingrained, magnet)
    loss = loss + stealth_rock_loss(tables, state, side, slot)
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
    # A switch action's newcomer: no Ingrain, no Magnet Rise (only Baton Pass keeps them).
    alive = _i32(state.hp[side, slot]) - hazards_loss(tables, state, side, slot, False, False) > 0
    weather = ability_weather(entry_ability(tables, state, side, slot, foe))
    return jnp.where(alive & (weather != Weather.NONE), weather, _i32(state.weather))


# effect_script_0048.s, _0269.s (RECOIL: Take Down, Double-Edge, ...): a Reckless attacker
# (CheckAbility, Battler_Ability) sets BTLVAR_POWER_MULTI to 12 before CalcCrit / CalcDamage;
# effect_script_0045.s (CRASH), _0253.s and _0262.s (RECOIL_STATUS) check it the same way.
RECKLESS_POWER_MUL = 12
RECKLESS_EFFECTS = (EffectId.RECOIL, EffectId.RECOIL_STATUS, EffectId.CRASH)


# BTLVAR_POWER_MULTI 20 (effect_script_0169.s, _0171.s, _0221.s).
DOUBLE_POWER_MUL = 20


def double_power(tables: Tables, state: BattleState, atk_side, def_side, move) -> jnp.ndarray:
    """``bool``: ``atk_side``'s ``DOUBLE_POWER`` move doubles against ``def_side``'s active
    battler: SmellingSalt against a paralyzed defender (effect_script_0171.s:6), Facade for a
    user with ``MON_CONDITION_FACADE_BOOST`` (effect_script_0169.s:5), Brine against
    ``curHP <= maxHP / 2`` (effect_script_0221.s:5-8; ``OPCODE_DIV`` is a plain ``/``,
    battle_script.c:2711-2712). SmellingSalt's ``CheckSubstitute`` (effect_script_0171.s:5) jumps
    past the x2 for a defender behind a substitute (its bit: no hit of the action came first)."""
    move = _i32(move)
    cond = _i32(tables.move_param[move, 0])
    user, target = state.active[atk_side], state.active[def_side]
    paralyzed = (state.status[def_side, target] == Status.PARALYSIS) & (
        state.substitute_hp[def_side] == 0
    )
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


# MoveFlag.HIT_* by ids.SemiKind (index 0, no bit, hits nothing).
_SEMI_HIT_BITS = (0, *(int(SEMI_HIT_FLAGS[k]) for k in sorted(SEMI_HIT_FLAGS)))


def hits_semi(tables: Tables, move, kind) -> jnp.ndarray:
    """``bool``: ``move``'s script sets the SYSCTL_HIT_DURING_* of the semi-invulnerable
    ``kind`` (an ``ids.SemiKind``; CheckMoveHitOverrides, bcp.c:3034-3040), B10-12."""
    bits = jnp.asarray(_SEMI_HIT_BITS, _I32)[_i32(kind)]
    return (_i32(tables.move_flags[_i32(move)]) & bits) != 0


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
    # effect_script_0045.s, _0253.s, _0262.s set it too (E2 batch 10, B10-1, B10-2).
    recoil = jnp.isin(tables.move_effect[_i32(move)], jnp.asarray(RECKLESS_EFFECTS, jnp.int32))
    doubled = double_power(tables, state, atk_side, def_side, move)
    # Stomp (effect_script_0150.s:6-8): powerMul 10, or 20 against MOVE_EFFECT_MINIMIZE (B9-10).
    stomp = (tables.move_effect[_i32(move)] == EffectId.FLINCH) & (
        _i32(tables.move_param[_i32(move), 2]) == STOMP_PARAM
    )
    doubled = doubled | (stomp & (state.minimized[def_side] != 0))
    # Gust / Twister, Earthquake / Magnitude, Surf / Whirlpool (effect_script_0146.s, _0149.s,
    # _0126.s, _0147.s, _0257.s, _0261.s: line 7-8): powerMul 20 against a defender with the
    # MOVE_EFFECT_* bit the move hits (MoveFlag.DOUBLE_SEMI), B10-12.
    semi_double = (_i32(tables.move_flags[_i32(move)]) & MoveFlag.DOUBLE_SEMI) != 0
    doubled = doubled | (semi_double & hits_semi(tables, move, state.semi_invulnerable[def_side]))
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
    tables: Tables, state: BattleState, atk_side, move, wonder_guard=True
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
    other Bide turn deals 0.

    Wonder Guard (``wonder_guarded``, where ``wonder_guard``) stops them like an immunity: they
    have table power 1, so ``ApplyTypeChart`` flags any that is not super effective (ruling
    G-18)."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    effect = tables.move_effect[_i32(move)]
    is_level = effect == EffectId.LEVEL_DAMAGE
    is_bide = effect == EffectId.BIDE
    level = _i32(state.level[atk_side, state.active[atk_side]])
    kind = move_type_of(tables, state, atk_side, move)
    d_types = state.types[def_side, state.active[def_side]]
    seen = identified(tables, state, atk_side, def_side)
    immune = effectiveness(tables, kind, d_types, seen) == 0
    immune = immune | ground_immune(tables, state, atk_side, def_side, kind)
    immune = immune | (
        jnp.asarray(wonder_guard) & wonder_guarded(tables, state, atk_side, move, kind, def_side)
    )
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
    # Dragon Rage / SonicBoom (effect_script_0041.s, _0130.s): the amount in param 0 (B9-9).
    flat = _i32(tables.move_param[_i32(move), 0])
    level = jnp.where(is_level & (flat > 0), flat, level)
    amount = jnp.where(is_ohko, ohko, jnp.where(is_halve, halved, level))
    # Endeavor (effect_script_0189.s:5-9): the defender's curHP minus the attacker's under
    # SYSCTL_IGNORE_TYPE_CHECKS; a difference <= 0 fails the move (step._use_move), B10-5.
    is_endeavor = effect == EffectId.ENDEAVOR
    gap = _i32(state.hp[def_side, target]) - _i32(state.hp[atk_side, state.active[atk_side]])
    amount = jnp.where(is_endeavor, jnp.maximum(gap, 0), amount)
    # Future Sight / Doom Desire (effect_script_0148.s): no CalcDamage at the use, so no damage;
    # the stored hit is step._future_sight's.
    is_future = effect == EffectId.FUTURE_SIGHT
    fixed = is_level | is_bide | is_halve | is_ohko | is_future | is_endeavor
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


# TrainerAI_CalcDamage's move-id cases (trainer_ai.c:2885-3071) that the AI damage models;
# decomp move ids (line - 1 of generated/moves.txt). The others (Natural Gift, Judgment, Gyro
# Ball) are refused (ai.commands.CALC_DAMAGE_MOVES).
AI_RETURN, AI_FRUSTRATION, AI_LOW_KICK, AI_GRASS_KNOT = 216, 218, 67, 447
AI_HIDDEN_POWER = 237
AI_DRAGON_RAGE, AI_SONIC_BOOM, AI_SEISMIC_TOSS, AI_NIGHT_SHADE = 82, 49, 69, 101
# The two cases that draw: MOVE_PSYWAVE's damage ``level * (RandNext() % 11 + 5) / 10``
# (trainer_ai.c:3014-3016) and MOVE_MAGNITUDE's power, ``RandNext() % 100`` bucketed
# (:3028-3048). Kaizo keeps the ids (Triple Axel on 149, Bulldoze on 222), so its moves there
# take these cases. ``draw`` below is the draw's class: Psywave's ``RandNext() % 11`` (0-10),
# Magnitude's bucket (0-6, ``AI_MAGNITUDE_POWER``); ai.outcome.CALC_DRAW_P holds their laws.
AI_PSYWAVE, AI_MAGNITUDE = 149, 222
AI_DRAWN_MOVES = (AI_PSYWAVE, AI_MAGNITUDE)
AI_DRAW_CLASSES = 11  # the most classes one draw has (Psywave's % 11)
AI_MAGNITUDE_POWER = jnp.array([10, 30, 50, 70, 90, 110, 150], jnp.int32)
# sWeightToPower (include/data/battle/weight_to_power.h): the first row whose weight
# (hectograms) is at least the defender's gives the power, else 120 (trainer_ai.c:3055-3071).
_WEIGHT_LIMITS = jnp.array([100, 250, 500, 1000, 2000], jnp.int32)
_WEIGHT_POWERS = jnp.array([20, 40, 60, 80, 100, 120], jnp.int32)


def ai_case_power(
    tables: Tables, s: BattleState, atk_side, move, *, def_side=None, ivs=None, draw=0
) -> jnp.ndarray:
    """``int32``: the power ``TrainerAI_CalcDamage`` passes for ``move`` (0 = the table's,
    ``BattleSystem_CalcMoveDamage``, battle_lib.c:6673-6677): Return
    ``friendship * 10 / 25`` and Frustration ``(255 - friendship) * 10 / 25``
    (trainer_ai.c:3018-3026; ``battleMons[attacker].friendship``), Low Kick and Grass Knot by
    the defender's weight (:3055-3071; ``battleMons[defender].weight``, its species' Pokedex
    weight, battle_lib.c:150), Hidden Power its IV power (:2973-2993, ``hidden_power``),
    Magnitude the power of its draw's bucket (:3028-3048; ``draw`` 0-6, ``AI_MAGNITUDE_POWER``:
    ``RandNext() % 100`` below 5, 15, 35, 65, 85, 95 gives 10, 30, 50, 70, 90, 110, else 150).
    A computed 0 is the table's power too. ``def_side`` is ``AI_CONTEXT.defender``'s side
    (default the other side) and ``ivs`` the ``ivs`` argument (default the attacker's own):
    ``IfBattlerDealsMoreDamage`` (trainer_ai.c:2187-2240) passes the player as attacker,
    the player again as defender and the AI attacker's IVs."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side if def_side is None else _i32(def_side)
    move = _i32(move)
    friendship = _i32(s.friendship[atk_side, s.active[atk_side]])
    weight = _i32(tables.species_weight[s.species[def_side, s.active[def_side]]])
    by_weight = _WEIGHT_POWERS[jnp.sum(_WEIGHT_LIMITS < weight)]
    ivs = s.ivs[atk_side, s.active[atk_side]] if ivs is None else ivs
    hidden = hidden_power(ivs)[1]
    magnitude = AI_MAGNITUDE_POWER[jnp.clip(_i32(draw), 0, 6)]
    return jnp.select(
        [
            move == AI_RETURN,
            move == AI_FRUSTRATION,
            (move == AI_LOW_KICK) | (move == AI_GRASS_KNOT),
            move == AI_HIDDEN_POWER,
            move == AI_MAGNITUDE,
        ],
        [friendship * 10 // 25, (255 - friendship) * 10 // 25, by_weight, hidden, magnitude],
        0,
    )


def ai_move_type(tables: Tables, s: BattleState, side, move, *, ivs=None) -> jnp.ndarray:
    """``int32``: ``TrainerAI_MoveType`` (trainer_ai.c:3127-3222) of ``move`` used by
    ``side``'s active Pokemon: Hidden Power (by id, :3208-3220) its IV type
    (``hidden_power``), any other move its table type (the function's TYPE_NORMAL, which
    ``ApplyTypeChart`` and ``CalcMoveDamage`` read as the table's, battle_lib.c:2588-2590,
    :6681-6684). Natural Gift, Judgment and Weather Ball are refused
    (``ai.commands.MOVE_TYPE_MOVES``). It reads ``ivs``, by default ``side``'s own;
    ``IfBattlerDealsMoreDamage`` (trainer_ai.c:2206-2237) types the player's move with the AI
    attacker's IVs (``TrainerAI_CalcDamage``'s Hidden Power case, :2973-2993)."""
    side = _i32(side)
    move = _i32(move)
    ivs = s.ivs[side, s.active[side]] if ivs is None else ivs
    iv_type = hidden_power(ivs)[0]
    return jnp.where(move == AI_HIDDEN_POWER, iv_type, _i32(tables.move_type[move]))


def ai_case_fixed(s: BattleState, atk_side, move, draw=0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(is_fixed, damage)``: ``TrainerAI_CalcDamage``'s fixed damage (trainer_ai.c:
    3005-3016, :3051-3053): Dragon Rage 40, Sonic Boom 20, Seismic Toss and Night Shade the
    attacker's level, Psywave ``level * (draw + 5) / 10`` (``draw`` its ``RandNext() % 11``,
    0-10; ``battleMons[attacker].level``). A nonzero amount skips ``CalcMoveDamage`` and goes
    through the type chart with ``SYSCTL_IGNORE_TYPE_CHECKS`` (:3081-3104): no STAB or
    multiplier, an immunity still 0. A 0 (Psywave at level 1 with a draw below 5) is not
    fixed: ``damage == 0`` sends it to ``CalcMoveDamage`` with power 0, the table's (:3073)."""
    atk_side = _i32(atk_side)
    move = _i32(move)
    level = _i32(s.level[atk_side, s.active[atk_side]])
    psywave = level * (jnp.clip(_i32(draw), 0, AI_DRAW_CLASSES - 1) + 5) // 10
    damage = jnp.select(
        [
            move == AI_DRAGON_RAGE,
            move == AI_SONIC_BOOM,
            (move == AI_SEISMIC_TOSS) | (move == AI_NIGHT_SHADE),
            move == AI_PSYWAVE,
        ],
        [40, 20, level, psywave],
        0,
    )
    return damage > 0, damage


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
    """``int32`` chart entry in tenths. ``identified``: a bool (Foresight) or ``identified``'s
    flags. The rows after the 0xFE separator (the Ghost immunities) are skipped for an
    identified defender (Foresight, battle_lib.c:2611), and an immune row of a Flying defender
    for a grounded one (Ingrain, :2534-2537), i.e. count as neutral."""
    flags = _i32(identified)
    mul = _i32(tables.type_chart[move_type, def_type])
    ghost = ((flags & CHART_IDENTIFIED) != 0) & (
        tables.type_chart_foresight[move_type, def_type] != 0
    )
    flying = ((flags & CHART_GROUNDED) != 0) & (_i32(def_type) == Type.FLYING) & (mul == 0)
    return jnp.where(ghost | flying, 10, mul)


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


def metronome_boost(tables: Tables, s: BattleState, atk_side, damage) -> jnp.ndarray:
    """``int32``: ``damage`` through the attacker's Metronome item, right after Life Orb in
    ``BattleScript_CalcMoveDamage`` (battle_script.c:1350-1352): ``damage * (10 +
    metronomeTurns) / 10`` for ``HOLD_EFFECT_BOOST_REPEATED`` (its param is not read), G-60."""
    atk_side = _i32(atk_side)
    item = _i32(s.item[atk_side, s.active[atk_side]])
    held = tables.item_effect[item] == ItemId.METRONOME_ITEM
    turns = _i32(s.metronome_turns[atk_side])
    return jnp.where(held, damage * (10 + turns) // 10, damage)


def pinch_ability(move_type, ability) -> jnp.ndarray:
    """``bool``: ``ability`` boosts ``move_type`` in a pinch (battle_lib.c:6819-6838): Overgrow
    Grass, Blaze Fire, Torrent Water, Swarm Bug."""
    return (
        ((move_type == Type.GRASS) & (ability == AbilityId.OVERGROW))
        | ((move_type == Type.FIRE) & (ability == AbilityId.BLAZE))
        | ((move_type == Type.WATER) & (ability == AbilityId.TORRENT))
        | ((move_type == Type.BUG) & (ability == AbilityId.SWARM))
    )


def side_active_slots(s: BattleState, side) -> tuple:
    """The party slots of ``side``'s battlers on the field (the decomp's battleMons of the
    side; the bench never counts). The state layout holds one active battler per side
    (``active[side]``), so a doubles layout extends only this function."""
    return (_i32(s.active[side]),)


def side_ability_count(s: BattleState, side, ability) -> jnp.ndarray:
    """``COUNT_ALIVE_BATTLERS_OUR_SIDE`` (battle_lib.c:2886-2893): the number of battlers of
    ``side`` on the field, the asking battler included, with ``curHP != 0`` and
    ``Battler_Ability == ability`` (``state.ability`` is the effective ability)."""
    count = jnp.int32(0)
    for slot in side_active_slots(s, side):
        count += ((_i32(s.ability[side, slot]) == ability) & (_i32(s.hp[side, slot]) > 0)).astype(
            _I32
        )
    return count


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
    def_side=None,
    ai_ivs=None,
    ai_draw=0,
):
    """Damage after ``+ 2`` and the crit multiplier, before the random roll.

    ``power_mul`` is ``battleCtx->powerMul`` in tenths (Pursuit on a switch: 20).
    ``self_hit`` is the confusion self-hit, ``CALC_SELF_HIT(MOVE_STRUGGLE, 40)``
    (``include/battle/common.h:35``): the attacker is its own defender, power 40, and
    no side or field conditions are passed (no screens, no weather). ``magnitude`` is
    Magnitude's level (``Outcome.magnitude``); ``present`` is Present's draw
    (``Outcome.present``: 0-2 is the power 40/80/120, any other value keeps the table
    power). ``ai`` is ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109), which has no Flail,
    Rollout or Present case: they keep the move's table power. ``def_side`` (``ai`` only)
    is ``AI_CONTEXT.defender``'s side, default the other side, and ``ai_ivs`` the IVs it
    passes, default the attacker's own (``ai_case_power``), and ``ai_draw`` the class of
    its ``RandNext`` draw (Magnitude's case)."""
    atk_side = _i32(atk_side)
    if self_hit:
        def_side = atk_side
    elif def_side is None:
        def_side = 1 - atk_side
    else:
        assert ai, "a defender side is the AI damage's (IfBattlerDealsMoreDamage)"
        def_side = _i32(def_side)
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
    # The AI has move-id cases instead of the battle's effect ones (ai_case_power, Hidden
    # Power's by id); it has none for Magnitude's effect, Rollout or Flail.
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
        | (effect == EffectId.FLAIL)
    )
    if ai:
        case = ai_case_power(tables, s, atk_side, move, def_side=def_side, ivs=ai_ivs, draw=ai_draw)
        power = jnp.where(case > 0, case, base_power)
    else:
        power = jnp.where(has_variable, variable, base_power)
        # CalcWringOutPower (battle_script.c:7265) and CalcWeightBasedPower (:6808-6826) set
        # movePower from the defender (the AI's Low Kick / Grass Knot are ai_case_power's).
        defender_power = jnp.where(
            effect == EffectId.HP_POWER, hp_power(s, def_side), weight_power(tables, s, def_side)
        )
        by_defender = (effect == EffectId.HP_POWER) | (effect == EffectId.WEIGHT_POWER)
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
    # :6682 moveType = MOVE_DATA(move).type, or Hidden Power's IV type (battle_script.c:6028);
    # the AI passes type 0 (the table's) for every case it models (trainer_ai.c:2881) but
    # Hidden Power's IV type (:2973-2993; ai_move_type).
    move_type = (
        ai_move_type(tables, s, atk_side, move, ivs=ai_ivs)
        if ai
        else move_type_of(tables, s, atk_side, move)
    )
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

    a_stats = battle_stats(s, atk_side, a_slot)
    d_stats = battle_stats(s, def_side, d_slot)
    attack, sp_attack = a_stats[Stat.ATK - 1], a_stats[Stat.SPA - 1]  # :6635, :6637
    defense, sp_defense = d_stats[Stat.DEF - 1], d_stats[Stat.SPD - 1]  # :6636, :6638
    a_stages, d_stages = _i32(s.stages[atk_side]), _i32(s.stages[def_side])
    attack_stage, sp_attack_stage = a_stages[Stat.ATK], a_stages[Stat.SPA]  # :6639, :6641
    defense_stage, sp_defense_stage = d_stages[Stat.DEF], d_stages[Stat.SPD]  # :6640, :6642

    # :6706-6708 Huge Power / Pure Power (attackerParams.ability): attackStat x2, before the
    # held items (Slow Start, just after, is unsupported).
    huge = (a_ability == AbilityId.HUGE_POWER) | (a_ability == AbilityId.PURE_POWER)
    attack = jnp.where(huge, attack * 2, attack)

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

    # :6802-6809 Plus / Minus (attackerParams.ability): spAttackStat x150/100 when a living
    # battler of the attacker's side (COUNT_ALIVE_BATTLERS_OUR_SIDE) has the other ability.
    plus = (a_ability == AbilityId.PLUS) & (side_ability_count(s, atk_side, AbilityId.MINUS) > 0)
    minus = (a_ability == AbilityId.MINUS) & (side_ability_count(s, atk_side, AbilityId.PLUS) > 0)
    sp_attack = jnp.where(plus | minus, sp_attack * 150 // 100, sp_attack)

    # :6811 Mud Sport / :6815 Water Sport: any active battler's sport halves the power.
    power = jnp.where((move_type == Type.ELECTRIC) & jnp.any(s.mud_sport != 0), power // 2, power)
    power = jnp.where((move_type == Type.FIRE) & jnp.any(s.water_sport != 0), power // 2, power)

    # :6819-6838 Overgrow / Blaze / Torrent / Swarm: power x1.5 at curHP <= maxHP / 3.
    pinch = pinch_ability(move_type, a_ability) & (a_hp <= a_max_hp // 3)
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
    # :6913-6918 Iron Fist (attackerParams.ability): a move of sPunchingMoves (MoveFlag.PUNCH,
    # the ROM's list) x12/10, after Rivalry (ruling G-49). The self-hit's move is Struggle.
    punch = (tables.move_flags[move] & MoveFlag.PUNCH) != 0
    power = jnp.where((a_ability == AbilityId.IRON_FIST) & punch, power * 12 // 10, power)

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
    # :6930-6941 Flower Gift in sun: COUNT_ALIVE_BATTLERS_OUR_SIDE (battle_lib.c:2886-2893, a
    # living battler of the side with the ability, Battler_Ability): the attacker's side's makes
    # attackStat x15/10, the defender's spDefenseStat x15/10 unless the attacker has Mold Breaker.
    # In singles each side's only battler is its active one (the self-hit passes no weather).
    sunny = (weather == Weather.SUN) & (not self_hit)
    a_gift = (a_ability == AbilityId.FLOWER_GIFT) & (a_hp > 0)
    d_gift = (
        (_i32(s.ability[def_side, d_slot]) == AbilityId.FLOWER_GIFT)
        & (_i32(s.hp[def_side, d_slot]) > 0)
        & (a_ability != AbilityId.MOLD_BREAKER)
    )
    attack = jnp.where(sunny & a_gift, attack * 15 // 10, attack)
    sp_defense = jnp.where(sunny & d_gift, sp_defense * 15 // 10, sp_defense)
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
    # :7058-7060 SolarBeam (by move id) under FIELD_CONDITION_SOLAR_DOWN (rain, sandstorm, hail,
    # fog: condition.h:136-139) is halved, between rain and sun (B10-13).
    solar_down = (
        (weather == Weather.RAIN)
        | (weather == Weather.SANDSTORM)
        | (weather == Weather.HAIL)
        | (weather == Weather.FOG)
    )
    damage = jnp.where(solar_down & (move == MOVE_SOLAR_BEAM) & (not self_hit), damage // 2, damage)
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
    def_side=None,
    wonder_guard=True,
):
    """Random roll, STAB and type chart; 0 for status moves and immunities. Struggle
    skips STAB and the chart (``ApplyTypeChart`` returns early for it, :2573).

    The battle applies the variance first (``BattleSystem_CalcDamageVariance``, :7081,
    at least 1), then the chart. ``variance_last`` is the trainer AI's order instead
    (``TrainerAI_CalcDamage``, trainer_ai.c:3096-3109): STAB and the chart first, then
    ``BattleSystem_Divide(damage * variance, 100)``, and it computes a status or power-0
    move too: ``CalcMoveDamage``'s ``+ 2`` (battle_lib.c:7078) goes through the chart, which
    ``IfBattlerDealsMoreDamage`` reaches with the player's last move (trainer_ai.c:2229-2237;
    the other AI callers pass only damaging moves, ``ai.commands.is_damaging``).
    ``def_side`` is ``AI_CONTEXT.defender``'s side there (default the other side).

    Wonder Guard (``wonder_guarded``, ruling G-18) gives 0 where ``wonder_guard``: the hit's
    ``MOVE_STATUS_WONDER_GUARD`` ends the move in subscript_missed.s (no damage), and the AI's
    estimate zeroes a ``MOVE_STATUS_IMMUNE`` result (trainer_ai.c:3106). ``wonder_guard=False``
    keeps ``battleCtx->damage`` as ``ApplyTypeChart`` leaves it (it does not zero it): Jump
    Kick's crash reads that (``step._use_move``)."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side if def_side is None else _i32(def_side)
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
    seen = identified(tables, s, atk_side, def_side)
    e1 = chart_entry(tables, move_type, t1, seen)
    e2 = jnp.where(t2 != t1, chart_entry(tables, move_type, t2, seen), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    first, second = jnp.where(t2_first, e2, e1), jnp.where(t2_first, e1, e2)
    damage = divide(divide(damage * first, 10) * second, 10)
    # :2657-2660 Solid Rock (Battler_IgnorableAbility): BattleSystem_Divide(damage * 3, 4)
    # when MOVE_STATUS_SUPER_EFFECTIVE is set. ApplyTypeMultiplier (:7557-7574) sets it for a
    # x2 entry unless NOT_VERY_EFFECTIVE was set, which it clears instead (and the other way
    # round), so it stands exactly when the product is above neutral; an immunity leaves 0.
    # ApplyTypeMultiplier updates the flags only for a move with table power (``update`` is
    # ``movePower``, :2627, :7558, :7568), and the bonuses need it too (:2656, :2667): a status
    # move or MOVE_NONE, which the AI path computes (IfBattlerDealsMoreDamage), takes none.
    has_power = _i32(power) > 0
    super_effective = (first * second > 100) & has_power
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
    not_very = (first * second < 100) & has_power
    damage = jnp.where(not_very & (a_ability == AbilityId.TINTED_LENS), damage * 2, damage)
    # :2600-2603 Levitate: MOVE_STATUS_LEVITATED instead of the chart walk, part of
    # MOVE_STATUS_IMMUNE: no damage (its DID_NOT_HIT ends the move, bcp.c:3325; the AI's
    # estimate zeroes an immune hit, trainer_ai.c:3106).
    damage = jnp.where(ground_immune(tables, s, atk_side, def_side, move_type), 0, damage)
    # :2649-2654 Wonder Guard: MOVE_STATUS_WONDER_GUARD, part of MOVE_STATUS_IMMUNE (ruling G-18).
    # A battle hit of a charge move is its strike (``charge_strike``); the AI's estimate
    # (``variance_last``) runs after BattleContext_Init, where it is not (G-27).
    strike = False if variance_last else charge_strike(tables, move)
    guarded = wonder_guarded(tables, s, atk_side, move, move_type, def_side, strike)
    damage = jnp.where(jnp.asarray(wonder_guard) & guarded, 0, damage)
    damage = jnp.where(typeless, unchanged, damage)
    if variance_last:
        # trainer_ai.c:3109; an immunity is already 0 (:3106). TrainerAI_CalcDamage has no
        # status-move test: it returns this.
        return divide(damage * variance, 100)
    # Status moves never calculate damage; CheckTypeChart needs base power
    # (battle_controller_player.c:2354).
    return jnp.where((category == STATUS) | (power == 0), 0, damage)


def damage_table(
    tables: Tables, state: BattleState, atk_side, move, crit, wonder_guard=True
) -> jnp.ndarray:
    """``int32[16]``: damage of ``move`` by ``atk_side``'s active Pokemon against the
    other side's active Pokemon for rolls 0..15 (``85 + roll`` percent). A ``LEVEL_DAMAGE``
    move has no roll: all 16 entries are ``fixed_damage``. Magnitude is taken at level 4
    (its weakest hit; the playout heuristic's documented under-estimate); Bide's unleash
    stays 0 (it is a forced locked action, never a chosen attack). Present keeps its table
    power (no draw: ``PRESENT_TABLE``). ``wonder_guard``: as ``_finish``'s."""
    # The defender is the other side's active Pokemon (this table's contract).
    power_mul = script_power_mul(tables, state, atk_side, 1 - _i32(atk_side), move, 10)
    damage, move_type, category, power = _pre_random(
        tables, state, atk_side, move, crit, power_mul=power_mul
    )
    damage = life_orb(tables, state, atk_side, damage)  # battle_script.c:1346
    damage = metronome_boost(tables, state, atk_side, damage)  # :1350-1352 (G-60)
    rolls = jnp.arange(ROLLS, dtype=_I32)
    table = _finish(
        tables, state, atk_side, move, damage, move_type, category, power, rolls,
        wonder_guard=wonder_guard,
    )  # fmt: skip
    fixed, amount = fixed_damage(tables, state, atk_side, move, wonder_guard)
    amount = jnp.where(tables.move_effect[_i32(move)] == EffectId.BIDE, 0, amount)
    return jnp.where(fixed, amount, table)


def hit_damage_table(
    tables: Tables, state: BattleState, atk_side, move, wonder_guard=True
) -> jnp.ndarray:
    """``int32[2, ROLLS]``: one hit's damage for every ``(crit, roll)`` pair (``crit`` 0
    or 1, then ``damage_table``'s rolls), at ``state``. It equals ``damage_for_roll`` at
    the default ``power_mul`` for every move but Magnitude (``damage_table`` takes level 4)
    and Bide (0 here); ``enumerate``'s damage classes use neither. ``wonder_guard``: as
    ``_finish``'s."""
    return jnp.stack(
        [damage_table(tables, state, atk_side, move, crit, wonder_guard) for crit in (False, True)]
    )


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
    wonder_guard=True,
) -> jnp.ndarray:
    """``int32``: ``damage_table(...)[roll]``, the single value the engine uses.
    ``power_mul`` (tenths) scales the move's power first (Pursuit on a switch: 20);
    ``magnitude`` is Magnitude's level; ``present`` is Present's draw, and its heal path
    (``PRESENT_HEAL``) deals 0: effect_script_0122.s jumps past ``CalcCrit`` and
    ``CalcDamage``. ``wonder_guard``: as ``_finish``'s."""
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
    damage = metronome_boost(tables, state, atk_side, damage)  # :1350-1352 (G-60)
    rolled = _finish(
        tables, state, atk_side, move, damage, move_type, category, power, _i32(roll),
        wonder_guard=wonder_guard,
    )  # fmt: skip
    fixed, amount = fixed_damage(tables, state, atk_side, move, wonder_guard)
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


def ai_damage_for_roll(
    tables: Tables, state: BattleState, atk_side, move, roll, *, def_side=None, ivs=None, draw=0
) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109) for a move with no
    special calculation: ``BattleSystem_CalcMoveDamage`` with ``criticalMul`` 1 (screens,
    weather, abilities, items and stages as in battle), then STAB and the type chart,
    then the variance ``85 + roll`` percent last, at least 1 when non-zero. A status or
    power-0 move gives the ``+ 2`` through the chart (``_finish``). ``def_side`` is
    ``AI_CONTEXT.defender``'s side (default the other side) and ``ivs`` the IVs passed
    (default the attacker's own): ``IfBattlerDealsMoreDamage`` (trainer_ai.c:2229-2237) has
    the player attack itself with the AI attacker's IVs. ``draw`` is the class of the
    ``RandNext`` draw of Psywave's or Magnitude's case (``ai_case_fixed``, ``ai_case_power``;
    ignored for any other move)."""
    damage, move_type, category, power = _pre_random(
        tables, state, atk_side, move, False, ai=True, def_side=def_side, ai_ivs=ivs, ai_draw=draw
    )
    computed = _finish(
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
        def_side=def_side,
    )
    # The fixed cases (ai_case_fixed): an immunity 0 (:3106), else the variance (:3109).
    fixed, amount = ai_case_fixed(state, atk_side, move, draw)
    def_side = 1 - _i32(atk_side) if def_side is None else _i32(def_side)
    d_types = state.types[def_side, state.active[def_side]]
    identified = state.foresight[def_side] != 0
    immune = effectiveness(tables, move_type, d_types, identified) == 0
    # :3106 MOVE_STATUS_IMMUNE includes MOVE_STATUS_WONDER_GUARD (ruling G-18).
    immune = immune | wonder_guarded(tables, state, atk_side, move, move_type, def_side)
    fixed_damage_ = jnp.where(immune, 0, divide(amount * (85 + _i32(roll)), 100))
    return jnp.where(fixed, fixed_damage_, computed)


def confusion_damage(tables: Tables, state: BattleState, side, roll) -> jnp.ndarray:
    """``int32``: the confusion self-hit of ``side``'s active Pokemon
    (battle_controller_player.c:2633-2634): ``CALC_SELF_HIT(MOVE_STRUGGLE, 40)``, a
    Normal physical hit on itself with its own stats and stages and no crit, then
    ``BattleSystem_CalcDamageVariance`` with ``roll``. No STAB or type chart
    (``ApplyTypeChart`` is not called)."""
    damage, _, _, _ = _pre_random(tables, state, side, MOVE_STRUGGLE, False, self_hit=True)
    return jnp.maximum(damage * (85 + _i32(roll)) // 100, 1)
