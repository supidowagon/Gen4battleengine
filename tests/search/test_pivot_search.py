"""E2 batch 6: the mid-turn switch decision in the search (rulings B6-6, B6-7): ``expand``
stops a turn at a party list and resumes it as a chance node of its own; the resumed node
against independent sampling; a Stage B solve through a ``pivot`` decision."""

from __future__ import annotations

import math
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS, decision_actions
from nuzline.engine.outcome import sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from nuzline.search.expand import expand
from nuzline.search.stage_b import SolveSettings, solve
from tests.search.test_stage_b import ZERO, batch


def _mon(names, species: str, moves, level=30) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


def _u_turn_state(tables, names):
    return make_state(
        tables,
        [_mon(names, "Scyther", ["U-turn"], 30), _mon(names, "Geodude", ["Tackle"], 30)],
        [_mon(names, "Bulbasaur", ["Tackle"], 35)],
    )


def _take(store, row):
    return jax.tree.map(lambda f: jnp.asarray(np.asarray(f)[row]), store)


def test_expand_pauses_then_resumes(tables, names):
    s = _u_turn_state(tables, names)
    store, out = expand(tables, batch(s), [(0, 0, 0)])
    kids = out[0]
    assert math.isclose(math.fsum(c.prob for c in kids), 1.0)
    paused = [c for c in kids if int(np.asarray(store.pivot_battler)[c.state_row]) == 1]
    assert paused, "a U-turn that hits pauses at the player's list"
    state = _take(store, paused[0].state_row)
    store2, out2 = expand(tables, batch(state), [(0, MOVE_ACTIONS + 1, 0)])
    resumed = out2[0]
    assert math.isclose(math.fsum(c.prob for c in resumed), 1.0)
    for c in resumed:
        assert int(np.asarray(store2.pivot_battler)[c.state_row]) == 0
        assert int(np.asarray(store2.active)[c.state_row, 0]) == 1
    # The resumed chance node against independent draws of the whole Outcome
    # (sample_outcome reads no relevance mask): the children's frequencies within 5 sigma.
    expected: dict[bytes, float] = {}
    for c in resumed:
        row = _take(store2, c.state_row)
        key = b"".join(np.asarray(f).tobytes() for f in row)
        expected[key] = expected.get(key, 0.0) + c.prob
    count = 2000
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    keys = jax.random.split(jax.random.key(3), count)
    outcomes = jax.vmap(sample_outcome, in_axes=(0, None, None, None))(keys, tables, state, acts)
    stepped = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))(tables, state, acts, outcomes)
    got = Counter(b"".join(np.asarray(f)[i].tobytes() for f in stepped) for i in range(count))
    assert set(got) <= set(expected)
    for key, p in expected.items():
        k = got.get(key, 0)
        assert abs(k / count - p) <= 5 * math.sqrt(p * (1 - p) / count) + 1e-9


def test_solve_goes_through_a_pivot_decision(tables, names):
    s = _u_turn_state(tables, names)
    settings = SolveSettings(max_depth=1, epsilon=0.0, n_depth0=8)
    result = solve(tables, names, s, jax.random.PRNGKey(0), settings, leaf_fn=ZERO)
    decisions = []

    def walk(nodes):
        for node in nodes:
            decisions.append(node.decision)
            for kids in node.children.values():
                walk(kids)

    walk([result.plan])
    assert "pivot" in decisions
    assert {r.action for r in result.ranking} == {0, MOVE_ACTIONS + 1}


def test_the_opponents_list_is_resolved_by_its_ai(tables, names):
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Tackle"], 35)],
        [_mon(names, "Scyther", ["U-turn"], 30), _mon(names, "Geodude", ["Tackle"], 30)],
    )
    store, out = expand(tables, batch(s), [(0, 0, 0)])
    paused = [c for c in out[0] if int(np.asarray(store.pivot_battler)[c.state_row]) == 2]
    assert paused
    state = _take(store, paused[0].state_row)
    assert np.flatnonzero(np.asarray(decision_actions(tables, state))).tolist() == [0]
    store2, out2 = expand(tables, batch(state), [(0, 0, 0)])
    assert math.isclose(math.fsum(c.prob for c in out2[0]), 1.0)
    for c in out2[0]:
        assert not c.refused
        assert int(np.asarray(store2.active)[c.state_row, 1]) == 1
