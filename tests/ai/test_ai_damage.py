"""The AI's damage estimate follows ``TrainerAI_CalcDamage`` (trainer_ai.c:2868-3109):
the variance is applied after STAB and the type chart (:3109), while the battle
applies it before them (``BattleSystem_CalcDamageVariance``, battle_lib.c:7081)."""

import jax
import jax.numpy as jnp
import pytest

from nuzline.ai.commands import ai_damage
from nuzline.engine.damage import ai_damage_for_roll, damage_for_roll
from tests.engine.helpers import synthetic_state


def _base_100(tables, **kw):
    """The AI's side (1) attacks with a 100 BP Fighting physical move: L100, Atk 350
    vs Def 300 gives ``350*100*42 // 300 // 50 + 2 = 100`` before the roll; STAB
    (Fighting user) and super effective (Normal defender)."""
    args = {
        "atk_level": 100,
        "atk_stats": [350, 100, 100, 100, 100],
        "def_stats": [100, 300, 100, 100, 100],
        "move_power": 100,
        "move_physical": True,
        "atk_types": ("Fighting", "Fighting"),
        "def_types": ("Normal", "Normal"),
        "move_type": "Fighting",
    }
    s, move = synthetic_state(tables, **(args | kw))
    # synthetic_state attacks from side 0; swap the sides so the AI (side 1) attacks.
    swapped = jax.tree.map(lambda x: x[::-1] if x.ndim and x.shape[0] == 2 else x, s)
    return swapped, move


def test_ai_estimate_applies_variance_last(tables):
    s, move = _base_100(tables)
    # Battle: 100 * 85 // 100 = 85, STAB 127, x2 = 254.
    assert int(damage_for_roll(tables, s, 1, move, False, 0)) == 254
    # AI: STAB 150, x2 = 300, then 300 * 85 // 100 = 255.
    assert int(ai_damage_for_roll(tables, s, 1, move, 0)) == 255
    assert int(ai_damage(tables, s, move, 0)) == 255


def test_ai_estimate_agrees_with_battle_at_variance_100(tables):
    s, move = _base_100(tables)
    battle = int(damage_for_roll(tables, s, 1, move, False, 15))
    assert battle == 300
    assert int(ai_damage(tables, s, move, 15)) == battle


def test_ai_estimate_is_zero_on_immunity(tables):
    s, move = _base_100(tables, def_types=("Ghost", "Ghost"))
    assert int(ai_damage(tables, s, move, 0)) == 0


@pytest.mark.parametrize("roll", [0, 7, 15])
def test_ai_estimate_jits(tables, roll):
    s, move = _base_100(tables)
    f = jax.jit(ai_damage_for_roll)
    out = f(tables, s, jnp.int32(1), jnp.int32(move), jnp.int32(roll))
    assert out.dtype == jnp.int32
    assert int(out) == int(ai_damage_for_roll(tables, s, 1, move, roll))
