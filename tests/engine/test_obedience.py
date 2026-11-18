"""Kaizo patches ``BattleControllerPlayer_CheckObedience`` so every Pokemon the player sends
obeys (data/kaizo/rom_diff.toml). The engine never had an obedience check, so this pins it."""

import jax.numpy as jnp
import pytest

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import step
from tests.engine.helpers import synthetic_state


@pytest.mark.slow
def test_player_mon_always_obeys(tables):
    # L16 attacker with no badge or trainer-ID state, using move slot 0.
    s, move = synthetic_state(
        tables,
        atk_level=16,
        atk_stats=[60, 40, 40, 40, 40],
        def_stats=[40, 40, 40, 40, 40],
        move_power=40,
        move_physical=True,
        atk_types=("Normal", "Normal"),
        def_types=("Fire", "Fire"),
        move_type="Fighting",
        def_level=50,
    )
    # Give the opponent the move too: with no PP it would use Struggle, whose recoil
    # would also lower its own HP.
    s = s._replace(moves=s.moves.at[1, 0, 0].set(move), pp=s.pp.at[1, 0, 0].set(30))
    o = empty_outcome()._replace(roll=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = step(tables, s, jnp.array([0, 0]), o)
    expected = int(damage_for_roll(tables, s, 0, move, crit=False, roll=0))
    assert expected > 0
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == expected


def test_state_has_no_obedience_fields():
    for field in BattleState._fields:
        assert not any(w in field for w in ("obey", "obedien", "badge", "trainer_id", "original"))
