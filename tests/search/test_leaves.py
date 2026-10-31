"""Leaf and forced-turn evaluation (``search/leaves.py``; plan rulings 5 and 8): playouts
in chunks of one padded shape, two disjoint key halves per leaf."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.policy import ai_action_distribution
from nuzline.engine.actions import legal_actions
from nuzline.search.expand import expand
from nuzline.search.leaves import (
    LEAF_CHUNK,
    LeafValue,
    allocate,
    evaluate_forced,
    evaluate_leaves,
    leaf_keys,
)
from nuzline.search.playout import REFUSED_WON, batched_playout, playout_from_action
from tests.engine.helpers import roark_state
from tests.search.fixtures import shinx_vs_cranidos
from tests.search.test_expand import batch

KEY = jax.random.PRNGKey(11)


def test_halves_disjoint_and_reproducible(tables, names):
    state = roark_state(tables, names)
    states = batch(state, state.replace_hp(1, 0, 20))
    counts = np.array([8, 12])
    one = evaluate_leaves(tables, states, counts, KEY)
    assert one == evaluate_leaves(tables, states, counts, KEY)
    assert [(v.n_a, v.n_b) for v in one] == [(4, 4), (6, 6)]
    for i, n in enumerate(counts):
        a, b = (np.asarray(k) for k in leaf_keys(KEY, i, int(n)))
        assert len(a) == len(b) == n // 2
        assert not {r.tobytes() for r in a} & {r.tobytes() for r in b}


def test_matches_batched_playout(tables, names):
    state = roark_state(tables, names)
    counts = np.array([16])
    (value,) = evaluate_leaves(tables, batch(state), counts, KEY)
    a, b = leaf_keys(KEY, 0, 16)
    keys = jnp.concatenate([a, b])
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (16, *x.shape)), state)
    out = batched_playout(tables, states, keys)
    assert not np.any(np.asarray(out.refused))
    won = np.asarray(out.won) == 1
    assert (value.wins_a, value.wins_b) == (int(won[:8].sum()), int(won[8:].sum()))
    assert not value.refused


def test_forced_matches_playout_from_action(tables, names):
    # Only the opponent's first move has PP: its AI row is one-hot.
    state = roark_state(tables, names, player_count=1, opp_count=1)
    state = state._replace(pp=state.pp.at[1, 0, 1:].set(0))
    q = np.asarray(ai_action_distribution(tables, state))
    opp = int(np.argmax(q))
    assert q[opp] == 1.0
    action = int(np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))[0])
    (value,) = evaluate_forced(
        tables, batch(state), np.array([[action, opp]], np.int32), np.array([16]), KEY
    )
    a, b = leaf_keys(KEY, 0, 16)
    out = jax.vmap(lambda k: playout_from_action(tables, state, action, k))(jnp.concatenate([a, b]))
    assert not np.any(np.asarray(out.refused))
    won = np.asarray(out.won) == 1
    assert (value.wins_a, value.wins_b) == (int(won[:8].sum()), int(won[8:].sum()))
    assert not value.refused


def test_forced_first_turn_matches_expand(tables, names):
    """A one-turn cap makes the value the exact P(child over and won) of the expansion."""
    state = shinx_vs_cranidos(tables, names).replace_hp(1, 0, 1)
    spark, tackle = names.move_slot(state, 0, "Spark"), names.move_slot(state, 1, "Tackle")
    _, children = expand(tables, batch(state), [(0, spark, tackle)])
    assert sum(c.prob for c in children[0]) == pytest.approx(1.0, abs=1e-12)
    n = 2000
    (value,) = evaluate_forced(
        tables,
        batch(state),
        np.array([[spark, tackle]], np.int32),
        np.array([n]),
        KEY,
        max_turns=1,
    )
    # Cranidos at 1 HP faints to Spark when it connects; it may also miss or be
    # outsped, so the won fraction is the Spark-connects mass, not 1.
    store, kids = expand(tables, batch(state), [(0, spark, tackle)])
    won_mass = sum(c.prob for c in kids[0] if int(store.over[c.state_row]) != 0)
    sd = np.sqrt(won_mass * (1 - won_mass) / (n / 2))
    assert abs(value.p_a - won_mass) < 5 * sd + 1e-9
    assert abs(value.p_b - won_mass) < 5 * sd + 1e-9


def test_allocate():
    w = np.array([1.0, 3.0, 0.0, 6.0])
    n = allocate(w, 1000)
    assert np.all(n % 2 == 0) and np.all(n >= 4)
    assert n[2] == 4
    assert abs(int(n.sum()) - 1000) <= 4 + len(w)
    assert n[3] / n[1] == pytest.approx(2.0, rel=0.05)
    assert list(allocate(np.array([1.0, 1.0]), 10, floor=3)) == [6, 6]
    assert list(allocate(np.zeros(3), 100)) == [4, 4, 4]


def test_leaf_value_properties():
    v = LeafValue(wins_a=3, n_a=4, wins_b=1, n_b=2, refused=False)
    assert v.p_a == 0.75 and v.p_b == 0.5
    p = 2 / 4
    assert v.var_b == pytest.approx(p * (1 - p) / 2)


def test_terminal_state_rejected(tables, names):
    state = roark_state(tables, names)
    done = state._replace(over=jnp.int8(1))
    with pytest.raises(ValueError, match="terminal"):
        evaluate_leaves(tables, batch(done), np.array([4]), KEY)


def test_chunk_boundary_invariant(tables, names):
    """More rows than one chunk give the same totals as the per-leaf key rule says."""
    assert LEAF_CHUNK == 4096
    state = roark_state(tables, names)
    (value,) = evaluate_leaves(tables, batch(state), np.array([LEAF_CHUNK + 10]), KEY)
    assert value.n_a + value.n_b == LEAF_CHUNK + 10
    assert REFUSED_WON == -1


_SCRIPT = """
import json, jax, numpy as np
from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import cache_root, code_version, set_compile_cache_dir
set_compile_cache_dir(cache_root() / "jax-tests" / code_version())
from nuzline.search.leaves import evaluate_forced, evaluate_leaves
from tests.engine.helpers import roark_state
from tests.search.test_expand import batch

assert jax.device_count() == DEVICES
tables, names = load_tables(), load_names()
state = roark_state(tables, names)
states = batch(state, state.replace_hp(1, 0, 20))
key = jax.random.PRNGKey(11)
leaves = evaluate_leaves(tables, states, np.array([10, 14]), key)
forced = evaluate_forced(tables, states, np.array([[0, 0], [1, 0]], np.int32),
                         np.array([6, 10]), key)
print(json.dumps([[v.wins_a, v.n_a, v.wins_b, v.n_b, v.refused] for v in leaves + forced]))
"""


def _on_devices(devices: int):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
    }
    done = subprocess.run(
        [sys.executable, "-c", f"DEVICES = {devices}\n" + _SCRIPT],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_device_count_invariant(tables):
    one = _on_devices(1)
    assert len(one) == 4
    assert _on_devices(3) == one
