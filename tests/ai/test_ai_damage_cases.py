"""TrainerAI_CalcDamage's move-id cases the AI damage models (E1 batch 13;
docs/ai/kaizo_roark.md 2.4): Return and Frustration by friendship, Low Kick and Grass Knot by
the defender's weight, and the fixed Dragon Rage, Sonic Boom, Seismic Toss and Night Shade
(trainer_ai.c:3005-3071, :3081-3109)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.engine import damage
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
