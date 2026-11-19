"""SF2 (h): Destiny Bond before the contact ability in Pursuit's KO path (B = 2).

subscript_pursuit.s runs FAINT_CHECK_DESTINY_BOND (:85) before TriggerAbilityOnHit (:86-87), so a
KO'd switcher that holds Destiny Bond takes the pursuer down first and every contact ability then
fails on the attacker's HP (battle_lib.c:4215-4363): no message, no reveal, no draw. Each case runs
through ``assert_matches_reference``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import enumerate_turn
from nuzline.engine.ids import Status, Weather
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step_events
from tests.engine.helpers import assert_matches_reference, jit_step

pytestmark = pytest.mark.fullset

O = empty_outcome()
SWITCH = MOVE_ACTIONS + 1  # switch to party slot 1


def _mon(names, species, moves, level=30, item="", ability=None) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    slot = 0
    if ability is not None:
        slot = names.species_abilities(names.species(species)).index(names.ability(ability))
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=names.item(item) if item else 0,
        ability_slot=slot,
        gender=0,
        friendship=70,
    )


def _i8(x, v):
    return jnp.asarray(v, x.dtype)


def _chaser(n):
    return _mon(n, "Houndour", ["Pursuit"], 35)


def _weak(n, species, ability):
    return _mon(n, species, ["Tackle"], 5, ability=ability)


def _bench(n):
    return _mon(n, "Geodude", ["Tackle"], 30)


def _bond(s, side):
    return s._replace(destiny_bond=s.destiny_bond.at[side].set(_i8(s.destiny_bond, 1)))


def _proc(v):
    return O._replace(contact_proc=jnp.full((2, MAX_HITS), v, jnp.int8))


def _events(tables, s, acts, o):
    _, events = jax.jit(step_events)(tables, s, jnp.asarray(acts), o)
    return {k: np.asarray(v) for k, v in events.items()}


def _player_pursuit(tables, n, ability, bond, pursuer_hp=None, species="Carvanha"):
    s = make_state(
        tables, [_chaser(n), _bench(n)], [_weak(n, species, ability), _bench(n)], Weather.NONE
    )
    if pursuer_hp is not None:
        s = s.replace_hp(0, 0, pursuer_hp)
    return _bond(s, 1) if bond else s


@pytest.mark.parametrize("pursuer_hp", [None, 1])
def test_bond_takes_the_pursuer_before_rough_skin(tables, names, pursuer_hp):
    n = names
    s = _player_pursuit(tables, n, "Rough Skin", True, pursuer_hp)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[0, 0]) == 0
    assert int(out.hp[1, 0]) == 0
    ev = _events(tables, s, (0, SWITCH), _proc(1))
    assert int(ev["contact_ability"][1]) == 0


@pytest.mark.parametrize("pursuer_hp", [None, 1])
def test_bond_leaves_the_players_switcher_ability_unrevealed(tables, names, pursuer_hp):
    # The AI tracks only the player's Pokemon: mirror the case with the foe chasing.
    n = names
    s = make_state(
        tables, [_weak(n, "Carvanha", "Rough Skin"), _bench(n)], [_chaser(n), _bench(n)]
    )
    s = _bond(s, 0)
    if pursuer_hp is not None:
        s = s.replace_hp(1, 0, pursuer_hp)
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.hp[1, 0]) == 0
    assert int(out.ai_ability_known[0]) == 0
    # Without the bond Rough Skin reveals itself (test 4's contrast).
    s = make_state(
        tables, [_weak(n, "Carvanha", "Rough Skin"), _bench(n)], [_chaser(n), _bench(n)]
    )
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.ai_ability_known[0]) != 0


def test_static_switcher_with_bond_paralyzes_nobody_for_either_draw(tables, names):
    n = names
    s = _player_pursuit(tables, n, "Static", True, species="Pikachu")
    outs = [assert_matches_reference(tables, s, (0, SWITCH), _proc(v)) for v in (0, 1)]
    for out in outs:
        assert int(out.hp[0, 0]) == 0
        assert int(out.status[0, 0]) == Status.NONE
    for x, y in zip(outs[0], outs[1], strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # The enumerator's rows all play to one state (the draw is merged away).
    rows, _ = enumerate_turn(tables, s, (0, SWITCH))
    states = set()
    for i in range(len(rows.hit)):
        row = jax.tree.map(lambda f, i=i: f[i], rows)
        out = jit_step(tables, s, jnp.asarray((0, SWITCH)), row)
        states.add(tuple(np.asarray(f).tobytes() for f in out))
    assert len(states) == 1


def test_rough_skin_switcher_without_bond_acts_as_before(tables, names):
    n = names
    s = _player_pursuit(tables, n, "Rough Skin", False)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # Rough Skin's damage
    ev = _events(tables, s, (0, SWITCH), _proc(1))
    assert int(ev["contact_ability"][1]) != 0


def test_surviving_bond_switcher_gets_no_take_down(tables, names):
    n = names
    s = make_state(
        tables, [_mon(n, "Houndour", ["Pursuit"], 5), _bench(n)],
        [_mon(n, "Geodude", ["Tackle"], 60, ability="Rough Skin"), _bench(n)],
    )
    s = _bond(s, 1)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[1, 0]) > 0
    ev = _events(tables, s, (0, SWITCH), _proc(1))
    assert int(ev["contact_ability"][1]) != 0  # Rough Skin ran as today
