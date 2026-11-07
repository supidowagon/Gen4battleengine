"""``tools.build_tables._move_effect`` on synthetic move dicts (no tables needed)."""

import pytest

from nuzline.engine.ids import EffectId
from tools.build_tables import BuildError, _move_effect

# Exactly the keys _move_effect reads: name, effect (BATTLE_EFFECT_BURN_HIT = 4) and
# effect_chance.
EMBER = {"name": "Ember", "effect": 4, "effect_chance": 10}
HAMMER_ARM = {"name": "Hammer Arm", "effect": 218, "effect_chance": 0}


def test_secondary_with_chance_keeps_its_class():
    eid, params = _move_effect(EMBER)
    assert eid == int(EffectId.SECONDARY_STATUS)
    assert params != (0, 0, 0)


def test_chance_zero_secondary_is_plain():
    assert _move_effect({**EMBER, "effect_chance": 0}) == (int(EffectId.PLAIN), (0, 0, 0))


def test_certain_secondary_stays_certain():
    assert _move_effect(HAMMER_ARM)[0] == int(EffectId.SECONDARY_STAT_SELF)


def test_certain_secondary_with_chance_raises():
    with pytest.raises(BuildError, match="Hammer Arm"):
        _move_effect({**HAMMER_ARM, "effect_chance": 30})


def test_chance_zero_secondary_with_more_than_the_secondary_is_unsupported():
    # K2b Ruling 21: effect 125 (THAW_AND_BURN_HIT) also thaws the user (params[2] = 1),
    # which PLAIN cannot say.
    flame_wheel = {"name": "Flame Wheel", "effect": 125, "effect_chance": 0}
    assert _move_effect(flame_wheel)[0] == int(EffectId.UNSUPPORTED)
    assert _move_effect({**flame_wheel, "effect_chance": 10})[0] == int(EffectId.SECONDARY_STATUS)


def test_changed_effect_is_unsupported():  # K2b Ruling 13
    assert _move_effect(EMBER, frozenset({4}))[0] == int(EffectId.UNSUPPORTED)
    assert _move_effect(EMBER, frozenset({5}))[0] == int(EffectId.SECONDARY_STATUS)
