"""TrainerAI_CalcDamage's move-id cases the AI damage models (E1 batch 13;
docs/ai/kaizo_roark.md 2.4): Return and Frustration by friendship, Low Kick and Grass Knot by
the defender's weight, and the fixed Dragon Rage, Sonic Boom, Seismic Toss and Night Shade
(trainer_ai.c:3005-3071, :3081-3109)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.engine import damage
from nuzline.engine.field import with_weather
from nuzline.engine.ids import Weather
from tests.ai.test_kaizo_outcome import _with_player
from tests.ai.test_kaizo_policy import BONSLY, _ids, _state


def _with_friendship(s, friendship: int):
    return s._replace(friendship=s.friendship.at[1, 0].set(friendship))


@pytest.mark.parametrize(
    ("move", "friendship", "power"),
    [
        ("Return", 255, 102),  # 255 * 10 / 25 (trainer_ai.c:3018-3021)
        ("Return", 100, 40),
        ("Return", 2, 0),  # a computed 0: the table's power (battle_lib.c:6673-6677)
        ("Frustration", 0, 102),  # (255 - 0) * 10 / 25 (:3023-3026)
        ("Frustration", 255, 0),
    ],
)
def test_friendship_power(kaizo_tables, kaizo_names, kaizo_roark_party, move, friendship, power):
    t, n = kaizo_tables, kaizo_names
    s = _with_friendship(
        _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "", "", "")), friendship
    )
    assert int(damage.ai_case_power(t, s, 1, n.move_names.index(move))) == power


@pytest.mark.parametrize(
    ("player", "power"),
    [
        # sWeightToPower (weight_to_power.h): the first row at or above the weight
        ("Gible", None),
        ("Snorlax", 120),  # 460.0 kg, above every row
        ("Pichu", 20),  # 2.0 kg
    ],
)
def test_weight_power(kaizo_tables, kaizo_names, kaizo_roark_party, player, power):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Grass Knot", "", "", ""))
    species = n.species_names.index(player)
    s = _with_player(t, s, species, 0, known=True)
    weight = int(t.species_weight[species])
    if power is None:  # by hand from the table
        limits, powers = (100, 250, 500, 1000, 2000), (20, 40, 60, 80, 100)
        power = next((p for w, p in zip(limits, powers, strict=True) if w >= weight), 120)
    for move in ("Grass Knot", "Low Kick"):
        assert int(damage.ai_case_power(t, s, 1, n.move_names.index(move))) == power, (move, weight)


@pytest.mark.parametrize("move", ["Seismic Toss", "Night Shade", "Dragon Rage", "SonicBoom"])
def test_fixed_damage(kaizo_tables, kaizo_names, kaizo_roark_party, move):
    """The fixed cases skip CalcMoveDamage and take the variance after the chart
    (trainer_ai.c:3081-3109): BattleSystem_Divide(fixed * (85 + roll), 100); a chart immunity
    gives 0 (:3106)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "", "", ""))
    m = n.move_names.index(move)
    level = int(s.level[1, 0])
    fixed = {"Dragon Rage": 40, "SonicBoom": 20}.get(move, level)
    for roll in (0, 7, 15):
        expect = max(fixed * (85 + roll) // 100, 1)
        assert int(damage.ai_damage_for_roll(t, s, 1, m, roll)) == expect, roll
    # a player the chart makes immune to the move's type, where it has one (Ghost against
    # Seismic Toss, Normal against Night Shade): 0
    move_type = jnp.int32(t.move_type[m])
    immune = [
        sp
        for sp in range(1, 493)
        if int(damage.effectiveness(t, move_type, jnp.asarray(t.species_types[sp]), False)) == 0
    ]
    if immune:
        s2 = _with_player(t, s, immune[0], 0, known=True)
        assert int(damage.ai_damage_for_roll(t, s2, 1, m, 15)) == 0


def test_alt_power_moves_always_compare(kaizo_tables):
    """sAltPowerMoveEffects (trainer_ai.c:48-61): a move with one of these effects is
    damage-estimated whatever its power (:2823-2828)."""
    t = kaizo_tables
    effects = np.asarray(t.move_ai_effect)
    alt = [m for m in range(1, effects.shape[0]) if int(effects[m]) in c.ALT_POWER_EFFECTS]
    assert alt
    for m in alt:
        assert bool(c.is_damaging(t, jnp.int32(m))), m


@pytest.mark.parametrize(("weight", "power"), [(250, 40), (251, 60), (2000, 100), (2001, 120)])
def test_weight_power_on_a_row(kaizo_tables, kaizo_names, kaizo_roark_party, weight, power):
    """A weight exactly on a row takes that row (``sWeightToPower[i][0] >= weight``,
    trainer_ai.c:3059-3061); one hectogram more takes the next."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Grass Knot", "", "", ""))
    species = int(s.species[0, 0])
    t2 = t._replace(species_weight=jnp.asarray(t.species_weight).at[species].set(weight))
    assert int(damage.ai_case_power(t2, s, 1, n.move_names.index("Grass Knot"))) == power


def test_fixed_damage_takes_no_stab(kaizo_tables, kaizo_names, kaizo_roark_party):
    """A Ghost attacker's Night Shade is still its level (SYSCTL_IGNORE_TYPE_CHECKS skips
    STAB, battle_lib.c:2592; trainer_ai.c:3093)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Night Shade", "", "", ""))
    ghost = c.TYPE_IDS["Ghost"]
    s = s._replace(types=s.types.at[1, 0].set(jnp.asarray([ghost, ghost], s.types.dtype)))
    level = int(s.level[1, 0])
    assert int(damage.ai_damage_for_roll(t, s, 1, n.move_names.index("Night Shade"), 15)) == level


def test_return_power_zero_takes_the_table_power(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Return at friendship 2 computes power 0, so CalcMoveDamage takes the table's
    (battle_lib.c:6673-6677; Kaizo Return 121), so it hits harder than at friendship 100
    (power 40)."""
    t, n = kaizo_tables, kaizo_names
    ret = n.move_names.index("Return")
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Return", "", "", ""))
    low = int(damage.ai_damage_for_roll(t, _with_friendship(s, 2), 1, ret, 15))
    mid = int(damage.ai_damage_for_roll(t, _with_friendship(s, 100), 1, ret, 15))
    assert int(t.move_power[ret]) > 40 and low > mid > 0


# --- E2 batch 13 (B13-25..26): Natural Gift's and Judgment's cases, TrainerAI_MoveType ------


def _holding(t, n, kaizo_roark_party, move: str, item: str | None):
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "", "", ""))
    return s._replace(item=s.item.at[1, 0].set(n.item_names.index(item) if item else 0))


@pytest.mark.parametrize(
    ("item", "power", "type_name"),
    [
        ("Cheri Berry", 60, "Fire"),  # naturalGiftPower / _Type (trainer_ai.c:2885-2895)
        ("Chilan Berry", 60, None),  # Normal gift type: TYPE_NORMAL, the table's
        ("Leftovers", 0, None),  # power 0: the table's power, TYPE_NORMAL
        (None, 0, None),
    ],
)
def test_natural_gift_case(kaizo_tables, kaizo_names, kaizo_roark_party, item, power, type_name):
    t, n = kaizo_tables, kaizo_names
    s = _holding(t, n, kaizo_roark_party, "Natural Gift", item)
    move = n.move_names.index("Natural Gift")
    expected = c.TYPE_IDS[type_name] if type_name else int(t.move_type[move])
    assert int(damage.ai_case_power(t, s, 1, move)) == power
    assert int(damage.ai_case_type(t, s, 1, move)) == expected
    # The heldItem argument, not the attacker's own (IfPartyMemberDealsMoreDamage, :2156-2165).
    bare = s._replace(item=s.item.at[1, 0].set(0))
    held = s.item[1, 0]
    assert int(damage.ai_case_power(t, bare, 1, move, item=held)) == power
    assert int(damage.ai_damage_for_roll(t, bare, 1, move, 15, item=held)) == int(
        damage.ai_damage_for_roll(t, s, 1, move, 15)
    )


def test_natural_gift_damage_is_the_berry_hit(kaizo_tables, kaizo_names, kaizo_roark_party):
    """The Cheri Berry case is CalcMoveDamage at power 60 and type Fire (trainer_ai.c:
    3074-3097): the same as a Fire move of power 60 and Natural Gift's class."""
    t, n = kaizo_tables, kaizo_names
    s = _holding(t, n, kaizo_roark_party, "Natural Gift", "Cheri Berry")
    move = n.move_names.index("Natural Gift")
    fire = c.TYPE_IDS["Fire"]
    t2 = t._replace(
        move_power=t.move_power.at[move].set(60), move_type=t.move_type.at[move].set(fire)
    )
    plain = s._replace(item=s.item.at[1, 0].set(0))
    got = int(damage.ai_damage_for_roll(t, s, 1, move, 15))
    assert got == int(damage.ai_damage_for_roll(t2, plain, 1, move, 15)) and got > 0


@pytest.mark.parametrize(("item", "type_name"), [("Flame Plate", "Fire"), (None, None)])
def test_judgment_case(kaizo_tables, kaizo_names, kaizo_roark_party, item, type_name):
    """Judgment's case (trainer_ai.c:2897-2971): power 0 (the table's), the plate's type, else
    TYPE_NORMAL; TrainerAI_MoveType's (:3135-3206) the same from the battler's item."""
    t, n = kaizo_tables, kaizo_names
    s = _holding(t, n, kaizo_roark_party, "Judgment", item)
    move = n.move_names.index("Judgment")
    expected = c.TYPE_IDS[type_name] if type_name else int(t.move_type[move])
    assert int(damage.ai_case_power(t, s, 1, move)) == 0
    assert int(damage.ai_case_type(t, s, 1, move)) == expected
    assert int(damage.ai_move_type(t, s, 1, move)) == expected


@pytest.mark.parametrize(
    ("item", "type_name"),
    [("Cheri Berry", "Fire"), ("Leftovers", "???"), (None, None), ("Chilan Berry", None)],
)
def test_natural_gift_move_type(kaizo_tables, kaizo_names, kaizo_roark_party, item, type_name):
    """TrainerAI_MoveType's Natural Gift (trainer_ai.c:3132-3134): Battler_NaturalGiftType of
    the held item whatever its power; a non-berry's 0x1F matches no type (AI_TYPELESS)."""
    t, n = kaizo_tables, kaizo_names
    s = _holding(t, n, kaizo_roark_party, "Natural Gift", item)
    move = n.move_names.index("Natural Gift")
    if type_name == "???":
        expected = int(damage.AI_TYPELESS)
    else:
        expected = c.TYPE_IDS[type_name] if type_name else int(t.move_type[move])
    assert int(damage.ai_move_type(t, s, 1, move)) == expected


@pytest.mark.parametrize(
    ("weather", "type_name"),
    [
        (Weather.RAIN, "Water"),
        (Weather.SANDSTORM, "Rock"),
        (Weather.SUN, "Fire"),
        (Weather.HAIL, "Ice"),
    ],
)
def test_weather_ball_move_type(kaizo_tables, kaizo_names, kaizo_roark_party, weather, type_name):
    """TrainerAI_MoveType's Weather Ball (trainer_ai.c:3222-3240): the weather's type, assigned in
    rain, sand, sun and hail without Cloud Nine; its AI damage takes the default case
    (table power and type, :3074-3078) whatever the weather."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Weather Ball", "", "", ""))
    s = with_weather(s, weather)
    move = n.move_names.index("Weather Ball")
    assert bool(damage.ai_weather_ball_typed(t, s))
    assert int(damage.ai_move_type(t, s, 1, move)) == c.TYPE_IDS[type_name]
    assert int(damage.ai_case_power(t, s, 1, move)) == 0
    assert int(damage.ai_case_type(t, s, 1, move)) == int(t.move_type[move])


@pytest.mark.fullset
@pytest.mark.parametrize("weather", [Weather.FOG, Weather.NONE])
def test_weather_ball_move_type_is_untyped_outside_the_four_weathers(
    kaizo_tables, kaizo_names, kaizo_roark_party, weather
):
    """F-O1: with no weather or fog alone the case assigns nothing (trainer_ai.c:3222-3240) and the
    function returns an uninitialised local, which the engine does not model: the typed flag
    is false and the policy refuses the move (``ai_refusals``, test_kaizo_policy)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Weather Ball", "", "", ""))
    s = with_weather(s, weather)
    move = n.move_names.index("Weather Ball")
    assert not bool(damage.ai_weather_ball_typed(t, s))
    assert int(damage.ai_case_power(t, s, 1, move)) == 0
    assert int(damage.ai_case_type(t, s, 1, move)) == int(t.move_type[move])


def test_party_member_natural_gift_reads_its_item(kaizo_tables, kaizo_names, kaizo_roark_party):
    """IfPartyMemberDealsMoreDamage passes each member's held item (trainer_ai.c:2156-2165):
    a benched Natural Gift holding Cheri Berry is estimated with it, as the active."""
    t, n = kaizo_tables, kaizo_names
    s = _holding(t, n, kaizo_roark_party, "Natural Gift", "Cheri Berry")
    move = n.move_names.index("Natural Gift")
    est = c.estimate(t, s, move, 15)
    bare = s._replace(item=s.item.at[1, 0].set(0))
    assert int(c.estimate(t, bare, move, 15, item=s.item[1, 0]).value[0]) == int(est.value[0])
    assert int(c.estimate(t, bare, move, 15).value[0]) != int(est.value[0])
