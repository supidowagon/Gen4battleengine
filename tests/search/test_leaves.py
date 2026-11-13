"""Leaf and forced-turn evaluation (``search/leaves.py``; plan rulings 5 and 8): playouts
in chunks of one padded shape, two disjoint key halves per leaf."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.policy import ai_action_distribution
from nuzline.engine.actions import legal_actions
from nuzline.engine.step import WINNER_PLAYER
from nuzline.search import deaths
from nuzline.search.expand import expand
from nuzline.search.leaves import (
    LEAF_CHUNK,
    LeafValue,
    _forced_turn,
    allocate,
    evaluate_forced,
    evaluate_leaves,
    leaf_keys,
)
from nuzline.search.playout import REFUSED_WON, batched_playout, playout_from_action
from tests.engine.helpers import roark_state
from tests.search.fixtures import at_replacement, dead_monferno
from tests.search.test_expand import batch, row_keys

pytestmark = pytest.mark.slow

KEY = jax.random.PRNGKey(11)
N_SPLIT = LEAF_CHUNK + 10  # playouts of one leaf that span two chunks

# The forced first turn over a batch of keys (one compile, shared by the tests below).
_turns = jax.jit(jax.vmap(_forced_turn, in_axes=(None, None, None, 0)))


def _spark_into_cranidos(tables, names, hp: int = 1):
    """Shinx's Spark against Cranidos at ``hp`` HP (one Pokemon a side), Tackle back. At
    26 HP the first turn wins about 41 % of the time (the damage roll and the crit
    decide), so one-turn playouts mix wins and non-wins."""
    state = roark_state(
        tables, names, player_count=1, opp_count=1, player_species="Shinx", opp_species="Cranidos"
    ).replace_hp(1, 0, hp)
    pair = (names.move_slot(state, 0, "Spark"), names.move_slot(state, 1, "Tackle"))
    return state, jnp.asarray(pair, jnp.int32)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_forced_first_turn_matches_expand(tables, names):
    """The forced first turn's child distribution (keys as evaluate_forced uses them)
    matches the exact children of ``expand``, bucketed by state. At 26 HP the turn
    mixes wins and non-wins (at 1 HP every child wins: the won mass is 1 up to rounding)."""
    state, pair = _spark_into_cranidos(tables, names, hp=26)
    spark, tackle = (int(a) for a in pair)
    store, children = expand(tables, batch(state), [(0, spark, tackle)])
    assert sum(c.prob for c in children[0]) == pytest.approx(1.0, abs=1e-12)
    keys_of = row_keys(store)
    want: dict[bytes, float] = {}
    for c in children[0]:
        want[keys_of[c.state_row]] = want.get(keys_of[c.state_row], 0.0) + c.prob
    won_mass = sum(
        c.prob for c in children[0] if int(store.over[c.state_row]) == 1 and not c.refused
    )
    assert 0.0 < won_mass < 1.0
    n = N_SPLIT  # the shape test_chunk_boundary_invariant compiles
    keys = jnp.concatenate(leaf_keys(KEY, 0, n))
    after, refused, _ = jax.device_get(_turns(tables, state, pair, keys))
    assert not np.any(refused)
    got = row_keys(after)
    assert set(got) <= set(want)
    for k, p in want.items():
        freq = got.count(k) / n
        assert abs(freq - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-9, (p, freq)
    # evaluate_forced's own win rate, with a one-turn cap, is the won mass.
    (value,) = evaluate_forced(
        tables, batch(state), np.array([pair]), np.array([n]), KEY, max_turns=1
    )
    sd = np.sqrt(won_mass * (1 - won_mass) / (n / 2))
    assert abs(value.p_a - won_mass) < 5 * sd
    assert abs(value.p_b - won_mass) < 5 * sd


def test_allocate():
    w = np.array([1.0, 3.0, 0.0, 6.0])
    n = allocate(w, 1000)
    # The default floor is 1 per half (2026-10-31 amendment): 2 playouts at least.
    assert np.all(n % 2 == 0) and np.all(n >= 2)
    assert n[2] == 2
    assert abs(int(n.sum()) - 1000) <= 2 + len(w)
    assert n[3] / n[1] == pytest.approx(2.0, rel=0.05)
    assert list(allocate(np.array([1.0, 1.0]), 10, floor=3)) == [6, 6]
    assert list(allocate(np.zeros(3), 100)) == [2, 2, 2]
    assert list(allocate(np.zeros(3), 100, floor=2)) == [4, 4, 4]


def test_leaf_value_properties():
    v = LeafValue(wins_a=3, n_a=4, wins_b=1, n_b=2, refused=False)
    assert v.p_a == 0.75 and v.p_b == 0.5
    p = 2 / 4
    assert v.var_b == pytest.approx(p * (1 - p) / 2)
    # var_a mirrors var_b on half A: p~ = (k_a + 1) / (n_a + 2), p~ (1 - p~) / n_a.
    q = 4 / 6
    assert v.var_a == pytest.approx(q * (1 - q) / 4)


def _playouts(tables, state, n=64):
    """``evaluate_leaves``'s value of ``state`` and the playouts behind it (same keys)."""
    (value,) = evaluate_leaves(tables, batch(state), np.array([n]), KEY)
    keys = jnp.concatenate(leaf_keys(KEY, 0, n))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), state)
    out = jax.device_get(batched_playout(tables, states, keys))
    assert not np.any(out.refused)
    return value, out


@pytest.mark.slow
def test_wins_count_successes(tables, names):
    state = dead_monferno(tables, names)
    value, out = _playouts(tables, state)
    won = np.asarray(out.won) == 1
    assert won.any()  # the playouts do win ...
    assert (value.wins_a, value.wins_b) == (0, 0)  # ... but not within k = 0
    value, out = _playouts(tables, state.with_budget(1))
    ok = (np.asarray(out.won) == 1) & (np.asarray(out.player_faints) <= 1)
    assert 0 < ok.sum()
    assert (value.wins_a, value.wins_b) == (int(ok[:32].sum()), int(ok[32:].sum()))


def test_protected_death_is_a_failure(tables, names):
    state = dead_monferno(tables, names).with_budget(3)
    monferno = int(np.flatnonzero(np.asarray(state.deaths))[0])
    state = state._replace(protected=state.protected.at[monferno].set(1))
    value, out = _playouts(tables, state)
    assert (np.asarray(out.won) == 1).any()
    assert (value.wins_a, value.wins_b) == (0, 0)


def test_deaths_and_buckets(tables, names):
    state = dead_monferno(tables, names).with_budget(2)
    value, out = _playouts(tables, state)
    faints = np.asarray(out.player_faints).astype(int)
    assert (value.deaths_a, value.deaths_b) == (int(faints[:32].sum()), int(faints[32:].sum()))
    assert value.e_deaths_b == value.deaths_b / 32
    bucket = deaths.bucket(
        np.asarray(out.won) == 1, out.ended, faints, out.deaths, np.asarray(state.protected)
    )
    assert value.buckets_b == tuple(np.bincount(bucket[32:], minlength=8).tolist())
    assert sum(value.buckets_b) == 32


def test_merge():
    a = LeafValue(1, 2, 3, 4, False, 5, 6, (1, 0, 0, 0, 0, 0, 3, 0))
    b = LeafValue(2, 4, 1, 2, True, 1, 2, (0, 2, 0, 0, 0, 0, 0, 0))
    assert a.merge(b) == LeafValue(3, 6, 4, 6, True, 6, 8, (1, 2, 0, 0, 0, 0, 3, 0))
    assert a.merge(a).refused is False


def test_pending_leaf(tables, names):
    state = at_replacement(roark_state(tables, names)).with_budget(2)
    value, out = _playouts(tables, state, 16)
    ok = (np.asarray(out.won) == 1) & (np.asarray(out.player_faints) <= 2)
    assert ok.any()
    assert (value.wins_a, value.wins_b) == (int(ok[:8].sum()), int(ok[8:].sum()))
    assert int(np.asarray(out.player_faints).min()) >= 1


def test_terminal_state_rejected(tables, names):
    state = roark_state(tables, names)
    done = state._replace(over=jnp.int8(1))
    with pytest.raises(ValueError, match="terminal"):
        evaluate_leaves(tables, batch(done), np.array([4]), KEY)
    with pytest.raises(ValueError, match="evaluate_forced: a terminal"):
        evaluate_forced(tables, batch(done), np.array([[0, 0]]), np.array([4]), KEY)


def test_shapes_checked(tables, names):
    """Counts and pairs must match the states one for one (checked before any program
    runs)."""
    state = roark_state(tables, names)
    two = batch(state, state)
    with pytest.raises(ValueError, match=r"evaluate_leaves: counts has shape \(1,\)"):
        evaluate_leaves(tables, two, np.array([4]), KEY)
    with pytest.raises(ValueError, match=r"evaluate_forced: counts has shape \(3,\)"):
        evaluate_forced(tables, two, np.zeros((2, 2)), np.array([4, 4, 4]), KEY)
    with pytest.raises(ValueError, match=r"evaluate_forced: pairs has shape \(1, 2\)"):
        evaluate_forced(tables, two, np.array([[0, 0]]), np.array([4, 4]), KEY)
    with pytest.raises(ValueError, match=r"evaluate_forced: pairs has shape \(2,\)"):
        evaluate_forced(tables, two, np.array([0, 0]), np.array([4, 4]), KEY)


def test_chunk_boundary_invariant(tables, names):
    """A leaf whose playouts span two chunks (half B crosses the boundary; the second
    chunk is padded) gets the totals its keys give played one by one: one-turn forced
    playouts, whose result is the first turn's (won iff that turn ends the battle in the
    player's favour)."""
    assert LEAF_CHUNK == 4096
    state, pair = _spark_into_cranidos(tables, names, hp=26)
    (value,) = evaluate_forced(tables, batch(state), pair[None], np.array([N_SPLIT]), KEY, 1)
    keys = jnp.concatenate(leaf_keys(KEY, 0, N_SPLIT))
    after, refused, _ = jax.device_get(_turns(tables, state, pair, keys))
    assert not np.any(refused)
    won = (np.asarray(after.over) != 0) & (np.asarray(after.winner) == WINNER_PLAYER)
    h = N_SPLIT // 2
    assert (value.n_a, value.n_b) == (h, h)
    assert (value.wins_a, value.wins_b) == (int(won[:h].sum()), int(won[h:].sum()))
    assert 0 < value.wins_a < h and 0 < value.wins_b < h  # a mix of wins and non-wins
    assert not value.refused
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


@pytest.mark.slow
def test_device_count_invariant(tables):
    one = _on_devices(1)
    assert len(one) == 4
    assert _on_devices(3) == one


def test_both_halves_have_buckets(tables, names):
    states = [dead_monferno(tables, names).with_budget(2), roark_state(tables, names)]
    values = evaluate_leaves(tables, jax.tree.map(lambda *x: np.stack(x), *states), [64, 64], KEY)
    for i, (state, value) in enumerate(zip(states, values, strict=True)):
        keys = jnp.concatenate(leaf_keys(KEY, i, 64))
        batch_ = jax.tree.map(lambda x: jnp.broadcast_to(x, (64, *x.shape)), state)
        out = jax.device_get(batched_playout(tables, batch_, keys))
        assert not np.any(out.refused)
        bucket = deaths.bucket(
            np.asarray(out.won) == 1,
            out.ended,
            np.asarray(out.player_faints).astype(int),
            out.deaths,
            np.asarray(state.protected),
        )
        assert sum(value.buckets_a) == value.n_a and sum(value.buckets_b) == value.n_b
        assert value.buckets_a == tuple(np.bincount(bucket[:32], minlength=8).tolist())
        assert value.buckets_b == tuple(np.bincount(bucket[32:], minlength=8).tolist())
        assert value.hp_left == pytest.approx(float(np.asarray(out.hp_left, np.float64).sum()))
        assert value.pp_left == pytest.approx(float(np.asarray(out.pp_left, np.float64).sum()))
        assert 0 < value.hp_left <= 64 and 0 < value.pp_left <= 64


def test_merge_sums_new_fields():
    a = LeafValue(1, 2, 3, 4, False, 5, 6, (1,) * 8, (0, 2, 0, 0, 0, 0, 0, 0), 1.5, 2.5)
    b = LeafValue(2, 4, 1, 2, True, 1, 2, (), (1, 0, 0, 0, 0, 0, 0, 0), 0.25, 0.5)
    m = a.merge(b)
    assert m.buckets_a == (1, 2, 0, 0, 0, 0, 0, 0) and m.buckets_b == (1,) * 8
    assert (m.hp_left, m.pp_left) == (1.75, 3.0)
    assert b.merge(a) == m.__class__(
        3, 6, 4, 6, True, 6, 8, (1,) * 8, (1, 2, 0, 0, 0, 0, 0, 0), 1.75, 3.0
    )
    assert LeafValue(1, 2, 1, 2, False).merge(LeafValue(1, 2, 1, 2, False)).buckets_a == ()
