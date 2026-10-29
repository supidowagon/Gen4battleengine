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

Only mechanics of the implemented set are modelled. Abilities, items and moves
outside it are ``UNSUPPORTED`` ids and never reach the branches here; the decomp
modifiers they would trigger (Technician, type-boosting items, Thick Fat, Flash
Fire, Cloud Nine, ...) are therefore omitted, not approximated. Doubles-only
spread and screen rules do not apply (singles).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import AbilityId, Stat, Status, Type, Weather
from nuzline.engine.state import BattleState

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category
CRIT_MUL = 2  # BattleSystem_CalcCriticalMulti (battle_lib.c:7105); x3 Sniper is not implemented
NEUTRAL_EFFECTIVENESS = 100  # effectiveness() of two neutral chart entries (10 * 10)
ROLLS = 16

# sStatStageBoosts (battle_lib.c:6550), indexed by stage + 6.
_STAGE_NUM = jnp.array([10, 10, 10, 10, 10, 10, 10, 15, 20, 25, 30, 35, 40], jnp.int32)
_STAGE_DEN = jnp.array([40, 35, 30, 25, 20, 15, 10, 10, 10, 10, 10, 10, 10], jnp.int32)

_I32 = jnp.int32


def _i32(x) -> jnp.ndarray:
    return jnp.asarray(x).astype(_I32)


def _divide(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """``BattleSystem_Divide`` (battle_lib.c:3599) for non-negative ``x``."""
    return jnp.where(x == 0, 0, jnp.maximum(x // d, 1))


def _staged(stat: jnp.ndarray, stage: jnp.ndarray) -> jnp.ndarray:
    """``stat * sStatStageBoosts[stage].numerator / .denominator`` (battle_lib.c:6956-6957)."""
    return stat * _STAGE_NUM[stage + 6] // _STAGE_DEN[stage + 6]


def _simple(stage: jnp.ndarray) -> jnp.ndarray:
    """Simple: stage doubled, clamped to [-6, 6] (battle_lib.c:6850-6884)."""
    return jnp.clip(stage * 2, -6, 6)


def effectiveness(tables: Tables, move_type, def_types) -> jnp.ndarray:
    """Product of the two chart entries in tenths (neutral ``100``, immune ``0``).
    Type 2 counts only when it differs from type 1 (battle_lib.c:2635)."""
    t1, t2 = _i32(def_types[0]), _i32(def_types[1])
    e1 = _i32(tables.type_chart[move_type, t1])
    e2 = jnp.where(t2 != t1, _i32(tables.type_chart[move_type, t2]), 10)
    return e1 * e2


def _pre_random(tables: Tables, s: BattleState, atk_side, move, crit):
    """Damage after ``+ 2`` and the crit multiplier, before the random roll."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    move = _i32(move)
    crit = jnp.asarray(crit, jnp.bool_)
    a_slot, d_slot = _i32(s.active[atk_side]), _i32(s.active[def_side])

    base_power = _i32(tables.move_power[move])
    power = base_power  # :6674 movePower = MOVE_DATA(move).power
    move_type = _i32(tables.move_type[move])  # :6682 moveType = MOVE_DATA(move).type
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
    power = jnp.where((move_type == Type.ELECTRIC) & (s.mud_sport != 0), power // 2, power)
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
    attack_stage = jnp.where(a_simple, _simple(attack_stage), attack_stage)
    sp_attack_stage = jnp.where(a_simple, _simple(sp_attack_stage), sp_attack_stage)
    # :6868 defender Simple (ignorable) doubles its defensive stages.
    d_simple = d_ability_eff == AbilityId.SIMPLE
    defense_stage = jnp.where(d_simple, _simple(defense_stage), defense_stage)
    sp_defense_stage = jnp.where(d_simple, _simple(sp_defense_stage), sp_defense_stage)
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
    sand = (s.weather == Weather.SANDSTORM) & rock
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

    damage = _staged(a_stat, a_stage)  # :6956-6957
    damage = damage * power  # :6960
    damage = damage * (level * 2 // 5 + 2)  # :6961
    damage = damage // _staged(d_stat, d_stage)  # :6975
    damage = damage // 50  # :6976

    # :6978 burn halves physical damage unless Guts.
    burned = is_physical & (a_status == Status.BURN) & (a_ability != AbilityId.GUTS)
    damage = jnp.where(burned, damage // 2, damage)
    # :6982 Reflect / :7023 Light Screen on the defender's side halve it, not on a crit.
    screen = jnp.where(is_physical, s.reflect[def_side], s.light_screen[def_side]) != 0
    damage = jnp.where(screen & ~crit, damage // 2, damage)

    # :7047 rain: Fire x0.5, Water x1.5; :7062 sun: Fire x1.5, Water x0.5. Fog: nothing.
    rain, sun = s.weather == Weather.RAIN, s.weather == Weather.SUN
    fire, water = move_type == Type.FIRE, move_type == Type.WATER
    damage = jnp.where(rain & fire, damage // 2, damage)
    damage = jnp.where(rain & water, damage * 15 // 10, damage)
    damage = jnp.where(sun & fire, damage * 15 // 10, damage)
    damage = jnp.where(sun & water, damage // 2, damage)

    damage = damage + 2  # :7078
    damage = jnp.where(crit, damage * CRIT_MUL, damage)  # battle_script.c:1344
    return damage, move_type, category, base_power


def _finish(tables: Tables, s: BattleState, atk_side, damage, move_type, category, power, roll):
    """Random roll, STAB and type chart; 0 for status moves and immunities."""
    atk_side = _i32(atk_side)
    def_side = 1 - atk_side
    a_types = _i32(s.types[atk_side, s.active[atk_side]])
    d_types = _i32(s.types[def_side, s.active[def_side]])

    # battle_lib.c:7084-7089 variance, at least 1.
    damage = jnp.maximum(damage * (85 + roll) // 100, 1)
    # :2592-2596 STAB x1.5 (Adaptability is not implemented).
    damage = jnp.where(jnp.any(a_types == move_type), damage * 15 // 10, damage)
    # :2611-2646 each defender type through ApplyTypeMultiplier (:7547), in chart
    # row order; type 2 is skipped when it equals type 1 (:2635). Unlisted pairs
    # are neutral (x10/10 leaves damage unchanged), so their order is irrelevant.
    t1, t2 = d_types[0], d_types[1]
    e1 = _i32(tables.type_chart[move_type, t1])
    e2 = jnp.where(t2 != t1, _i32(tables.type_chart[move_type, t2]), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    first, second = jnp.where(t2_first, e2, e1), jnp.where(t2_first, e1, e2)
    damage = _divide(_divide(damage * first, 10) * second, 10)
    # Status moves never calculate damage; CheckTypeChart needs base power
    # (battle_controller_player.c:2354).
    return jnp.where((category == STATUS) | (power == 0), 0, damage)


def damage_table(tables: Tables, state: BattleState, atk_side, move, crit) -> jnp.ndarray:
    """``int32[16]``: damage of ``move`` by ``atk_side``'s active Pokemon against the
    other side's active Pokemon for rolls 0..15 (``85 + roll`` percent)."""
    damage, move_type, category, power = _pre_random(tables, state, atk_side, move, crit)
    rolls = jnp.arange(ROLLS, dtype=_I32)
    return _finish(tables, state, atk_side, damage, move_type, category, power, rolls)


def damage_for_roll(tables: Tables, state: BattleState, atk_side, move, crit, roll) -> jnp.ndarray:
    """``int32``: ``damage_table(...)[roll]``, the single value the engine uses."""
    damage, move_type, category, power = _pre_random(tables, state, atk_side, move, crit)
    return _finish(tables, state, atk_side, damage, move_type, category, power, _i32(roll))
