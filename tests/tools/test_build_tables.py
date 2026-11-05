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
