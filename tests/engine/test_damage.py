"""Damage calculation, checked against hand computations from the decomp's formula
(``BattleSystem_CalcMoveDamage``, ``src/battle/battle_lib.c:6601``; see damage.py).

The base case throughout: L100 (level factor 42), stat 300 vs 200, power 100 gives
``300*100*42 // 200 // 50 = 126``, then ``+ 2 = 128``.
"""

import jax
import jax.numpy as jnp
import pytest

from nuzline.engine.damage import damage_for_roll, damage_table, effectiveness
from nuzline.engine.ids import MOVE_STRUGGLE, AbilityId, Status, Type, Weather
from tests.engine.helpers import synthetic_state

NEUTRAL_128 = [108, 110, 111, 112, 113, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 128]


def rolls(pre_random: int) -> list[int]:
    """The 16 rolls of a pre-random damage value: ``d * (85 + roll) // 100``."""
    return [pre_random * (85 + r) // 100 for r in range(16)]


def physical(tables, **kw):
    """Fighting 100 BP physical (Hammer Arm), Atk 300 vs Def 200 at L100, neutral."""
    args = {
        "atk_level": 100,
        "atk_stats": [300, 100, 100, 100, 100],
        "def_stats": [100, 200, 100, 100, 100],
        "move_power": 100,
        "move_physical": True,
        "atk_types": ("Normal", "Normal"),
        "def_types": ("Fire", "Fire"),
        "move_type": "Fighting",
    }
    return synthetic_state(tables, **(args | kw))


def special(tables, move_type="Fire", **kw):
    """95 BP special (Flamethrower / Thunderbolt), SpA 300 vs SpD 200 at L100:
    ``300*95*42 // 200 // 50 = 119``."""
    args = {
        "atk_level": 100,
        "atk_stats": [100, 100, 300, 100, 100],
        "def_stats": [100, 100, 100, 200, 100],
        "move_power": 95,
        "move_physical": False,
        "atk_types": ("Normal", "Normal"),
        "def_types": ("Normal", "Normal"),
        "move_type": move_type,
    }
    return synthetic_state(tables, **(args | kw))


def table(tables, s, move, crit=False):
    return damage_table(tables, s, 0, move, crit=crit).tolist()


# --- the brief's five tests (defender Fire/Fire so Fighting is neutral) ----------


def test_neutral_table_matches_hand_computation(tables):
    # L100 attacker, Atk 300 vs Def 200, 100 BP physical, no STAB, neutral type
    s, move = physical(tables)
    assert table(tables, s, move) == NEUTRAL_128


def test_stab_applies_after_random(tables):
    s, move = physical(tables, atk_types=("Fighting", "Normal"))
    assert table(tables, s, move) == [v * 3 // 2 for v in NEUTRAL_128]


def test_crit_doubles_before_random(tables):
    # crit x2 is applied to the value after +2 (battle_script.c:1344), so 128 * 2.
    s, move = physical(tables)
    assert table(tables, s, move, crit=True) == [256 * (85 + r) // 100 for r in range(16)]


def test_immunity_and_effectiveness(tables, names):
    assert (
        int(
            effectiveness(
                tables, names.type("Ghost"), jnp.array([names.type("Normal"), names.type("Normal")])
            )
        )
        == 0
    )
    assert (
        int(
            effectiveness(
                tables, names.type("Fire"), jnp.array([names.type("Grass"), names.type("Ice")])
            )
        )
        == 400
    )


def test_status_move_deals_zero(tables, names):
    s, _ = synthetic_state(
        tables,
        atk_level=14,
        atk_stats=[40] * 5,
        def_stats=[40] * 5,
        move_power=0,
        move_physical=True,
        atk_types=("Normal", "Normal"),
        def_types=("Normal", "Normal"),
        move_type="Normal",
    )
    assert damage_table(tables, s, 0, names.move("Leer"), crit=False).tolist() == [0] * 16


# --- decomp details ----------------------------------------------------------


def test_type_ids_match_manifest(names):
    assert [t.name for t in Type] == [n.replace("???", "MYSTERY") for n in names.type_names]


def test_same_type_twice_counts_once(tables, names):
    # ApplyTypeChart skips type 2 when it equals type 1 (battle_lib.c:2635).
    grass = names.type("Grass")
    assert int(effectiveness(tables, names.type("Fire"), jnp.array([grass, grass]))) == 200
    s, move = physical(tables, def_types=("Normal", "Normal"))
    assert table(tables, s, move) == [2 * v for v in NEUTRAL_128]


def test_type_multipliers_apply_in_chart_row_order(tables):
    # ApplyTypeChart walks sTypeMatchupMultipliers row by row (battle_lib.c:2611-2646)
    # and applies whichever defender type's row comes first. Fire's Bug row (x2,
    # battle_lib.c:2406) precedes its Rock row (x0.5, :2407), so against Rock/Bug the
    # roll is doubled, then halved: back to the neutral value exactly. Type 1 first
    # would halve first and lose 1 HP on every odd roll.
    neutral = rolls(121)  # special(): 119 + 2 = 121 before the roll
    assert any(v % 2 for v in neutral)
    for def_types in [("Rock", "Bug"), ("Bug", "Rock")]:
        s, move = special(tables, def_types=def_types)
        assert table(tables, s, move) == neutral, def_types


def test_immune_type_deals_zero(tables):
    s, move = physical(tables, def_types=("Ghost", "Ghost"))
    assert table(tables, s, move) == [0] * 16


def test_type_multiplier_never_rounds_to_zero(tables):
    # L1 (level factor 2), SpA 1 vs SpD 999, Ember: base 0, +2 = 2, every roll 1;
    # x0.5 twice through BattleSystem_Divide (battle_lib.c:3599) keeps 1.
    s, move = synthetic_state(
        tables,
        atk_level=1,
        atk_stats=[1, 1, 1, 1, 1],
        def_stats=[1, 1, 1, 999, 1],
        move_power=40,
        move_physical=False,
        atk_types=("Normal", "Normal"),
        def_types=("Water", "Rock"),
        move_type="Fire",
    )
    assert table(tables, s, move) == [1] * 16


def test_stat_stages(tables):
    s, move = physical(tables)
    assert table(tables, s.with_stages(0, atk=1), move) == rolls(191)  # Atk 450: 189 + 2
    assert table(tables, s.with_stages(1, def_=2), move) == rolls(65)  # Def 400: 63 + 2
    assert table(tables, s.with_stages(0, spa=6), move) == NEUTRAL_128  # wrong stat: no effect


def test_crit_ignores_bad_stages_only(tables):
    s, move = physical(tables)
    assert table(tables, s.with_stages(0, atk=-1).with_stages(1, def_=2), move, crit=True) == rolls(
        256
    )
    assert table(tables, s.with_stages(0, atk=1), move, crit=True) == rolls(382)  # 191 * 2


def test_simple_doubles_and_unaware_ignores_stages(tables):
    s, move = physical(tables, atk_ability=AbilityId.SIMPLE)
    assert table(tables, s.with_stages(0, atk=1), move) == rolls(254)  # +2: Atk 600, 252 + 2
    assert table(tables, s.with_stages(0, atk=4), move) == table(
        tables, s.with_stages(0, atk=6), move
    )
    s, move = physical(tables, def_ability=AbilityId.UNAWARE)
    assert table(tables, s.with_stages(0, atk=2), move) == NEUTRAL_128
    s, move = physical(tables, atk_ability=AbilityId.UNAWARE)
    assert table(tables, s.with_stages(1, def_=2), move) == NEUTRAL_128


def test_mold_breaker_ignores_defender_unaware_and_simple(tables):
    s, move = physical(tables, atk_ability=AbilityId.MOLD_BREAKER, def_ability=AbilityId.UNAWARE)
    assert table(tables, s.with_stages(0, atk=1), move) == rolls(191)
    s, move = physical(tables, atk_ability=AbilityId.MOLD_BREAKER, def_ability=AbilityId.SIMPLE)
    s = s.with_stages(1, def_=1)
    plain = s._replace(ability=s.ability.at[1, 0].set(AbilityId.NONE))
    assert table(tables, s, move) == table(tables, plain, move) == rolls(86)  # Def 300: 84 + 2


def test_burn_halves_physical_unless_guts(tables):
    s, move = physical(tables)
    assert table(tables, s.with_status(0, 0, Status.BURN), move) == rolls(65)  # 126 // 2 + 2
    s, move = physical(tables, atk_ability=AbilityId.GUTS)
    assert table(tables, s.with_status(0, 0, Status.BURN), move) == rolls(
        191
    )  # Atk x1.5, no halving
    s, move = special(tables)
    assert table(tables, s.with_status(0, 0, Status.BURN), move) == rolls(121)


def test_screens_halve_except_on_crit(tables):
    s, move = physical(tables)
    s = s._replace(reflect=s.reflect.at[1].set(5))
    assert table(tables, s, move) == rolls(65)
    assert table(tables, s, move, crit=True) == rolls(256)
    s, move = special(tables)
    assert table(tables, s._replace(reflect=s.reflect.at[1].set(5)), move) == rolls(121)
    assert table(tables, s._replace(light_screen=s.light_screen.at[1].set(5)), move) == rolls(
        61
    )  # 59 + 2


def test_weather(tables):
    s, move = special(tables)
    assert table(tables, s._replace(weather=jnp.int8(Weather.SUN)), move) == rolls(
        180
    )  # 119 * 1.5 + 2
    assert table(tables, s._replace(weather=jnp.int8(Weather.RAIN)), move) == rolls(61)
    assert table(tables, s._replace(weather=jnp.int8(Weather.FOG)), move) == rolls(121)
    # Sandstorm raises a Rock-type defender's SpD x1.5: 300*95*42 // 300 // 50 = 79.
    s, move = special(tables, move_type="Electric", def_types=("Rock", "Rock"))
    assert table(tables, s._replace(weather=jnp.int8(Weather.SANDSTORM)), move) == rolls(81)


def test_pinch_abilities(tables):
    s, move = special(tables, atk_ability=AbilityId.BLAZE)
    third = int(s.max_hp[0, 0]) // 3
    assert table(tables, s.replace_hp(0, 0, third), move) == rolls(180)  # power 142
    assert table(tables, s.replace_hp(0, 0, third + 1), move) == rolls(121)
    s, move = special(tables, atk_ability=AbilityId.TORRENT)
    assert table(tables, s.replace_hp(0, 0, third), move) == rolls(121)  # Torrent: Water only


def test_sports_halve_power(tables):
    s, move = special(tables)
    assert table(tables, s._replace(water_sport=s.water_sport.at[1].set(1)), move) == rolls(
        61
    )  # power 47
    s, move = special(tables, move_type="Electric")
    assert table(tables, s._replace(mud_sport=jnp.int8(1)), move) == rolls(61)
    assert table(tables, s._replace(water_sport=s.water_sport.at[0].set(1)), move) == rolls(121)


def test_rivalry(tables):
    s, move = physical(tables, atk_ability=AbilityId.RIVALRY, atk_gender=0, def_gender=0)
    assert table(tables, s, move) == rolls(159)  # power 125
    s, move = physical(tables, atk_ability=AbilityId.RIVALRY, atk_gender=0, def_gender=1)
    assert table(tables, s, move) == rolls(96)  # power 75
    s, move = physical(tables, atk_ability=AbilityId.RIVALRY, atk_gender=0, def_gender=2)
    assert table(tables, s, move) == NEUTRAL_128


def test_opponent_side_attacks_player(tables):
    s, move = physical(tables)
    swapped = jax.tree.map(lambda x: x[::-1] if x.ndim and x.shape[0] == 2 else x, s)
    assert damage_table(tables, swapped, 1, move, crit=False).tolist() == NEUTRAL_128


def test_damage_for_roll_and_jit(tables):
    s, move = physical(tables, atk_types=("Fighting", "Normal"))
    full = damage_table(tables, s, 0, move, crit=False)
    assert full.dtype == jnp.int32 and full.shape == (16,)
    f = jax.jit(damage_for_roll)
    for roll in (0, 7, 15):
        out = f(tables, s, jnp.int32(0), jnp.int32(move), jnp.bool_(False), jnp.int32(roll))
        assert out.dtype == jnp.int32 and int(out) == int(full[roll])


@pytest.mark.parametrize("crit", [False, True])
def test_vmap_over_states(tables, crit):
    s, move = physical(tables)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), s)
    out = jax.vmap(lambda st: damage_table(tables, st, 0, move, crit))(batch)
    assert out.tolist() == [table(tables, s, move, crit)] * 2


def test_struggle_is_typeless(tables, names):
    # BattleSystem_ApplyTypeChart (battle_lib.c:2573) and BattleSystem_CalcEffectiveness
    # (:2686) return early for MOVE_STRUGGLE: no STAB and no type chart, so a Normal-type
    # user Struggling against a Ghost still does the bare 50 BP physical damage.
    assert names.move("Struggle") == MOVE_STRUGGLE
    s, cut = physical(tables, move_power=50, move_type="Normal", def_types=("Ghost", "Ghost"))
    assert damage_table(tables, s, 0, cut, False).tolist() == [0] * 16  # a Normal move: immune
    # 300 * 50 * 42 // 200 // 50 + 2 = 65 before the roll.
    assert damage_table(tables, s, 0, MOVE_STRUGGLE, False).tolist() == rolls(65)


def test_crit_stage_rule_applies_after_simple_doubles(tables):
    # docs/discrepancies.md #6: Simple doubles first (battle_lib.c:6850, :6868), then a
    # crit ignores Atk <= 0 and Def >= 0 (:6947-6973), so both crits are neutral 256.
    s, move = physical(tables, atk_ability=AbilityId.SIMPLE)
    assert table(tables, s.with_stages(0, atk=-1), move, crit=True) == rolls(256)
    s, move = physical(tables, def_ability=AbilityId.SIMPLE)
    assert table(tables, s.with_stages(1, def_=1), move, crit=True) == rolls(256)


def test_sandstorm_spd_boost_comes_before_the_stage(tables):
    # docs/discrepancies.md #5: SpD 200 x1.5 = 300 (battle_lib.c:6925), then -1 stage
    # (:7008) gives 200 again; the other order (200 -> 133 -> 199) gives 120 + 2.
    s, move = special(tables, move_type="Electric", def_types=("Rock", "Rock"))
    sand = s._replace(weather=jnp.int8(Weather.SANDSTORM)).with_stages(1, spd=-1)
    assert table(tables, sand, move) == table(tables, s, move) == rolls(121)
