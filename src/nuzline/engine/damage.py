"""Gen 4 damage calculation (pure ``jnp``, ``int32`` throughout, jit/vmap-able).

Mirrors the decomp's pipeline for one hit of a damaging move:

1. ``BattleSystem_CalcMoveDamage`` (``src/battle/battle_lib.c:6601``): power and
   stat modifiers, stages, ``A * power * level_factor / D / 50``, burn, screens,
   weather, ``+ 2``.
2. ``BattleScript_CalcMoveDamage`` (``src/battle/battle_script.c:1344``): ``* criticalMul``.
3. ``BattleSystem_CalcDamageVariance`` (``battle_lib.c:7081``): ``* (100 - rand % 16) / 100``,
   at least 1. Our ``roll`` is ``15 - rand % 16``, so roll ``r`` multiplies by
   ``85 + r`` and ``damage_table`` is ascending.
4. ``BattleSystem_ApplyTypeChart`` (``battle_lib.c:2560``, run later by
   ``BattleControllerPlayer_CheckTypeChart``): STAB ``* 15 / 10``, then each
   defender type's multiplier through ``BattleSystem_Divide`` (``:3599``, a
   non-zero result never truncates to 0), in the order their rows appear in
   ``sTypeMatchupMultipliers`` (``:2399``, walked at ``:2611-2646``), which is
   not always type 1 then type 2 (``tables.type_chart_order``).

The trainer AI's estimate (``TrainerAI_CalcDamage``, ``trainer_ai.c:2868``) runs
steps 1, 4 and then 3: ``ai_damage_for_roll`` (crit multiplier 1, variance last).

Only mechanics of the implemented set are modelled. Abilities, items and moves
outside it are ``UNSUPPORTED`` ids and never reach the branches here; the decomp
modifiers they would trigger (Technician, type-boosting items, Thick Fat, Flash
Fire, Cloud Nine, ...) are therefore omitted, not approximated. Doubles-only
spread and screen rules do not apply (singles).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import MOVE_STRUGGLE, AbilityId, EffectId, Stat, Status, Type, Weather
from nuzline.engine.state import BattleState

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category
CRIT_MUL = 2  # BattleSystem_CalcCriticalMulti (battle_lib.c:7105); x3 Sniper is not implemented
ROLLS = 16
CONFUSION_POWER = 40  # CALC_SELF_HIT(MOVE_STRUGGLE, 40), battle_controller_player.c:2633

# sStatStageBoosts (battle_lib.c:6550), indexed by stage + 6.
_STAGE_NUM = jnp.array([10, 10, 10, 10, 10, 10, 10, 15, 20, 25, 30, 35, 40], jnp.int32)
_STAGE_DEN = jnp.array([40, 35, 30, 25, 20, 15, 10, 10, 10, 10, 10, 10, 10], jnp.int32)

_I32 = jnp.int32

# BtlCmd_CalcMagnitudePower (battle_script.c:5820-5852): the power of Magnitude levels 4-10.
MAGNITUDE_POWER = jnp.array([10, 30, 50, 70, 90, 110, 150], jnp.int32)
MAGNITUDE_MIN = 4


def _i32(x) -> jnp.ndarray:
    return jnp.asarray(x).astype(_I32)


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
    0."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    is_fixed = tables.move_effect[_i32(move)] == EffectId.LEVEL_DAMAGE
    level = _i32(state.level[atk_side, state.active[atk_side]])
    kind = move_type_of(tables, state, atk_side, move)
    d_types = state.types[def_side, state.active[def_side]]
    identified = state.foresight[def_side] != 0
    immune = effectiveness(tables, kind, d_types, identified) == 0
    return is_fixed, jnp.where(immune, 0, level)


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
):
    """Damage after ``+ 2`` and the crit multiplier, before the random roll.

    ``power_mul`` is ``battleCtx->powerMul`` in tenths (Pursuit on a switch: 20).
    ``self_hit`` is the confusion self-hit, ``CALC_SELF_HIT(MOVE_STRUGGLE, 40)``
    (``include/battle/common.h:35``): the attacker is its own defender, power 40, and
    no side or field conditions are passed (no screens, no weather). ``magnitude`` is
    Magnitude's level (``Outcome.magnitude``)."""
    atk_side = _i32(atk_side)
    def_side = atk_side if self_hit else 1 - atk_side
    move = _i32(move)
    crit = jnp.asarray(crit, jnp.bool_)
    a_slot, d_slot = _i32(s.active[atk_side]), _i32(s.active[def_side])

    base_power = _i32(tables.move_power[move])
    # :6674 movePower = inPower when given (the self-hit's 40), else MOVE_DATA(move).power
    effect = _i32(tables.move_effect[move])
    # CalcHiddenPowerParams (battle_script.c:6007) / CalcMagnitudePower (:5820) set movePower.
    variable = jnp.where(
        effect == EffectId.HIDDEN_POWER,
        hidden_power(s.ivs[atk_side, a_slot])[1],
        MAGNITUDE_POWER[jnp.clip(_i32(magnitude) - MAGNITUDE_MIN, 0, 6)],
    )
    has_variable = (effect == EffectId.HIDDEN_POWER) | (effect == EffectId.MAGNITUDE)
    power = jnp.where(has_variable, variable, base_power)
    power = jnp.asarray(CONFUSION_POWER, _I32) if self_hit else power
    power = power * _i32(power_mul) // 10  # :6688 movePower * powerMul / 10
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

    # :6926 sandstorm: a Rock-type defender's SpD x1.5.
    rock = jnp.any(d_types == Type.ROCK)
    sand = (s.weather == Weather.SANDSTORM) & rock & (not self_hit)
    sp_defense = jnp.where(sand, sp_defense * 15 // 10, sp_defense)

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
    screen = screen & (not self_hit)
    damage = jnp.where(screen & ~crit, damage // 2, damage)

    # :7047 rain: Fire x0.5, Water x1.5; :7062 sun: Fire x1.5, Water x0.5. Fog: nothing.
    rain = (s.weather == Weather.RAIN) & (not self_hit)
    sun = (s.weather == Weather.SUN) & (not self_hit)
    fire, water = move_type == Type.FIRE, move_type == Type.WATER
    damage = jnp.where(rain & fire, damage // 2, damage)
    damage = jnp.where(rain & water, damage * 15 // 10, damage)
    damage = jnp.where(sun & fire, damage * 15 // 10, damage)
    damage = jnp.where(sun & water, damage // 2, damage)

    damage = damage + 2  # :7078
    damage = jnp.where(crit, damage * CRIT_MUL, damage)  # battle_script.c:1344
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
    variance = 85 + _i32(roll)

    if not variance_last:
        # battle_lib.c:7084-7089 variance, at least 1.
        damage = jnp.maximum(damage * variance // 100, 1)
    typeless = _i32(move) == MOVE_STRUGGLE
    unchanged = damage
    # :2592-2596 STAB x1.5 (Adaptability is not implemented).
    damage = jnp.where(jnp.any(a_types == move_type), damage * 15 // 10, damage)
    # :2611-2646 each defender type through ApplyTypeMultiplier (:7547), in chart
    # row order; type 2 is skipped when it equals type 1 (:2635). Unlisted pairs
    # are neutral (x10/10 leaves damage unchanged), so their order is irrelevant.
    t1, t2 = d_types[0], d_types[1]
    identified = s.foresight[def_side] != 0  # :2615 the Foresight batch break
    e1 = chart_entry(tables, move_type, t1, identified)
    e2 = jnp.where(t2 != t1, chart_entry(tables, move_type, t2, identified), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    first, second = jnp.where(t2_first, e2, e1), jnp.where(t2_first, e1, e2)
    damage = divide(divide(damage * first, 10) * second, 10)
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
    (its weakest hit; the playout heuristic's documented under-estimate)."""
    damage, move_type, category, power = _pre_random(tables, state, atk_side, move, crit)
    rolls = jnp.arange(ROLLS, dtype=_I32)
    table = _finish(tables, state, atk_side, move, damage, move_type, category, power, rolls)
    fixed, amount = fixed_damage(tables, state, atk_side, move)
    return jnp.where(fixed, amount, table)


def damage_for_roll(
    tables: Tables,
    state: BattleState,
    atk_side,
    move,
    crit,
    roll,
    power_mul=10,
    magnitude=MAGNITUDE_MIN,
) -> jnp.ndarray:
    """``int32``: ``damage_table(...)[roll]``, the single value the engine uses.
    ``power_mul`` (tenths) scales the move's power first (Pursuit on a switch: 20);
    ``magnitude`` is Magnitude's level."""
    damage, move_type, category, power = _pre_random(
        tables, state, atk_side, move, crit, power_mul=power_mul, magnitude=magnitude
    )
    rolled = _finish(tables, state, atk_side, move, damage, move_type, category, power, _i32(roll))
    fixed, amount = fixed_damage(tables, state, atk_side, move)
    return jnp.where(fixed, amount, rolled)


def calc_move_damage(
    tables: Tables, state: BattleState, atk_side, move, power_mul=10
) -> jnp.ndarray:
    """``int32``: ``BattleSystem_CalcMoveDamage`` (battle_lib.c:6601) alone, with
    ``criticalMul`` 1: step 1 above, ending with the ``+ 2`` (so a power-0 move gives 2).
    No variance, STAB or type chart. ``power_mul`` is ``battleCtx->powerMul`` in tenths."""
    return _pre_random(tables, state, atk_side, move, False, power_mul=power_mul)[0]


def ai_damage_for_roll(tables: Tables, state: BattleState, atk_side, move, roll) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109) for a move with no
    special calculation: ``BattleSystem_CalcMoveDamage`` with ``criticalMul`` 1 (screens,
    weather, abilities, items and stages as in battle), then STAB and the type chart,
    then the variance ``85 + roll`` percent last, at least 1 when non-zero."""
    damage, move_type, category, power = _pre_random(tables, state, atk_side, move, False)
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
