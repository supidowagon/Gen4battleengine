"""Batched exact turn expansion (``search/expand.py``; Stage B spec §1-2, plan rulings
2-4): every (state, player action, opponent action) request's children, merged on
(state bytes, observation bytes), with float64 probabilities summing to 1."""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.outcome import sample_ai_outcome
from nuzline.ai.policy import ai_policy
from nuzline.engine.actions import legal_actions, legal_opponent_actions
from nuzline.engine.enumerate import SamplingOnly, enumerate_outcomes, outcome_laws
from nuzline.engine.observe import observe
from nuzline.engine.order import is_switch, selected_move
from nuzline.engine.outcome import sample_outcome
from nuzline.engine.step import PURSUIT_POWER_MUL, pursuit_fires, step
from nuzline.search import expand as expand_mod
from nuzline.search import playout
from nuzline.search.expand import CHUNK, Child, ai_rows, expand
from nuzline.search.playout import send_replacements
from tests.engine.helpers import roark_real_state
from tests.search.fixtures import shinx_vs_cranidos, spike_cannon_root, switch_refusal_root

SMALL = 256  # the independent re-step program's batch (test_merged_child_equals_each_member)


def batch(*states):
    """Numpy ``BattleState`` with one row per state."""
    return jax.tree.map(lambda *fs: np.stack([np.asarray(f) for f in fs]), *states)


def row_keys(tree) -> list[bytes]:
    """Per-row bytes of a stacked numpy pytree (a state batch)."""
    leaves = jax.tree.leaves(tree)
    n = len(leaves[0])
    flat = np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in leaves], axis=1
    )
    return [flat[i].tobytes() for i in range(n)]


def legal_pairs(tables, state, index: int = 0) -> list[tuple[int, int, int]]:
    mine = np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))
    theirs = np.flatnonzero(np.asarray(legal_opponent_actions(tables, state)))
    return [(index, int(a), int(b)) for a in mine for b in theirs]


@pytest.fixture(scope="module")
def roark_expansion(tables, names):
    """``expand`` of every legal pair at the real Roark start and two variants (the
    leads at 20 HP), with the row count of each ``step_rows`` call recorded
    (``test_expand_streams``): about 1.7 ``CHUNK``s of rows."""
    root = roark_real_state(tables, names)
    roots = [root, root.replace_hp(1, 0, 20), root.replace_hp(0, 0, 20)]
    pairs = [p for i, s in enumerate(roots) for p in legal_pairs(tables, s, i)]
    sizes = []
    real = expand_mod.step_rows

    def recording(tables, states, actions, outcomes):
        sizes.append(len(actions))
        return real(tables, states, actions, outcomes)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(expand_mod, "step_rows", recording)
        store, children = expand(tables, batch(*roots), pairs)
    return roots, pairs, store, children, sizes


def test_children_sum_to_one(roark_expansion):
    _, pairs, store, children, _ = roark_expansion
    assert sorted(children) == list(range(len(pairs)))
    rows = len(store.hp)
    for r, kids in children.items():
        assert isinstance(kids, list), (pairs[r], kids)
        assert kids
        assert all(isinstance(c, Child) and 0 <= c.state_row < rows for c in kids)
        assert all(c.prob > 0 for c in kids)
        assert sum(c.prob for c in kids) == pytest.approx(1.0, abs=1e-12)
        keys = [(c.state_row, c.obs) for c in kids]
        assert len(set(keys)) == len(keys), "children not merged"


def test_expand_streams(roark_expansion):
    *_, sizes = roark_expansion
    assert len(sizes) >= 2, sizes  # the request set spans several chunks
    assert max(sizes) <= CHUNK


def _turn_row(tables, before, actions, outcome):
    """One member outcome re-stepped on its own path: ``step``, the opponent's forced
    replacement (playout.py:177-194) and ``observe``."""
    picks = [selected_move(tables, before, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    pursued = pursuit_fires(tables, before, 0, actions, choices)
    power_mul = jnp.where(pursued, PURSUIT_POWER_MUL, 10)
    stepped = step(tables, before, actions, outcome)
    after, refused = send_replacements(tables, stepped, is_switch(actions[1]), power_mul)
    return after, refused, observe(tables, before, actions, outcome, after)


_restep = jax.jit(jax.vmap(_turn_row, in_axes=(None, None, None, 0)))


def test_merged_child_equals_each_member(tables, names):
    state = shinx_vs_cranidos(tables, names, slower_opp=True)
    # Cranidos at 1 HP: Spark KOs it with or without a crit, one state seen two ways
    # (ruling 2 keeps the two children apart).
    low = state.replace_hp(1, int(state.active[1]), 1)
    spark, tackle = names.move_slot(state, 0, "Spark"), names.move_slot(state, 1, "Tackle")
    pairs = [(0, spark, tackle), (0, names.move_slot(state, 0, "Tackle"), tackle)]
    pairs.append((1, spark, tackle))
    roots = [state, low]
    store, children = expand(tables, batch(*roots), pairs)
    store_keys = row_keys(store)
    rows = [c.state_row for c in children[2]]
    assert len(set(rows)) < len(rows), "no state reached with two observations"
    for r, (s, a, b) in enumerate(pairs):
        state = roots[s]
        actions = jnp.asarray([a, b], jnp.int32)
        outcomes, probs = enumerate_outcomes(jax.device_get(outcome_laws(tables, state, actions)))
        members: dict[tuple, float] = defaultdict(float)
        for start in range(0, len(probs), SMALL):
            idx = np.arange(start, min(start + SMALL, len(probs)))
            pad = np.concatenate([idx, np.full(SMALL - len(idx), idx[0])])
            chunk = jax.tree.map(lambda f, pad=pad: f[pad], outcomes)
            after, refused, obs = jax.device_get(_restep(tables, state, actions, chunk))
            keys = row_keys(after)
            for i in range(len(idx)):
                key = (keys[i], np.asarray(obs[i], np.int32).tobytes(), bool(refused[i]))
                members[key] += float(probs[idx[i]])
        got = {(store_keys[c.state_row], c.obs, c.refused): c.prob for c in children[r]}
        assert got.keys() == members.keys()
        for key, p in members.items():
            assert got[key] == pytest.approx(p, abs=1e-12)


def _playout_turn(tables, state, a, key):
    """The playout's own turn (playout.py:177-194) and its observation."""
    k_ai, k_turn = jax.random.split(key)
    opp = ai_policy(tables, state, sample_ai_outcome(k_ai, tables, state))
    played, _, ai_switched, power_mul = playout._play(tables, state, a, key)
    after, refused = send_replacements(tables, played, ai_switched, power_mul)
    actions = jnp.stack([a, jnp.maximum(opp, 0)]).astype(jnp.int32)
    obs = observe(tables, state, actions, sample_outcome(k_turn, tables, state, actions), after)
    return opp, after, refused, obs


def test_matches_playout_turn(tables, names):
    state = roark_real_state(tables, names)
    legal = np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))
    n = 200
    mine = jnp.asarray(legal[np.arange(n) % len(legal)], jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    turn = jax.jit(jax.vmap(_playout_turn, in_axes=(None, None, 0, 0)))
    opp, after, refused, obs = jax.device_get(turn(tables, state, mine, keys))
    after_keys = row_keys(after)
    kept = [i for i in range(n) if opp[i] >= 0]
    assert kept
    pairs = sorted({(0, int(mine[i]), int(opp[i])) for i in kept})
    store, children = expand(tables, batch(state), pairs)
    store_keys = row_keys(store)
    for i in kept:
        kids = children[pairs.index((0, int(mine[i]), int(opp[i])))]
        want = (after_keys[i], np.asarray(obs[i], np.int32).tobytes(), bool(refused[i]))
        hits = [c for c in kids if (store_keys[c.state_row], c.obs, c.refused) == want]
        assert len(hits) == 1, (i, pairs, len(hits))


def test_sampling_only_reported(tables, names):
    state = spike_cannon_root(tables, names)
    cannons = (names.move_slot(state, 0, "Spike Cannon"), names.move_slot(state, 1, "Spike Cannon"))
    scratch, tackle = names.move_slot(state, 0, "Scratch"), names.move_slot(state, 1, "Tackle")
    pairs = [(0, *cannons), (0, scratch, tackle)]
    _, children = expand(tables, batch(state), pairs)
    assert isinstance(children[0], SamplingOnly)
    assert "width cap" in children[0].reason
    assert isinstance(children[1], list)
    assert sum(c.prob for c in children[1]) == pytest.approx(1.0, abs=1e-12)


def test_replacement_refusal_flagged(tables, names):
    state = switch_refusal_root(tables, names)
    to_geodude = 4 + 0  # switch to party slot 0
    q = ai_rows(tables, batch(state))
    assert q.shape == (1, 14) and q.dtype == np.float32
    assert q[0, to_geodude] > 0
    tackle = names.move_slot(state, 0, "Tackle")
    _, children = expand(tables, batch(state), [(0, tackle, to_geodude)])
    kids = children[0]
    assert any(c.refused for c in kids)
    assert sum(c.prob for c in kids) == pytest.approx(1.0, abs=1e-12)


_MULTI_DEVICE_SCRIPT = """
import hashlib, json, jax
from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import cache_root, code_version, set_compile_cache_dir
set_compile_cache_dir(cache_root() / "jax-tests" / code_version())
from nuzline.search.expand import expand
from tests.search.fixtures import shinx_vs_cranidos
from tests.search.test_expand import batch, row_keys

assert jax.device_count() == DEVICES, jax.device_count()
tables, names = load_tables(), load_names()
state = shinx_vs_cranidos(tables, names, slower_opp=True)
spark, tackle = names.move_slot(state, 0, "Spark"), names.move_slot(state, 1, "Tackle")
pairs = [(0, spark, tackle), (0, names.move_slot(state, 0, "Leer"), tackle)]
store, children = expand(tables, batch(state), pairs)
out = {
    "store": [hashlib.sha256(k).hexdigest() for k in row_keys(store)],
    "children": {
        str(r): [(c.state_row, hashlib.sha256(c.obs).hexdigest(), c.prob.hex(), c.refused)
                 for c in kids]
        for r, kids in children.items()
    },
}
print(json.dumps(out))
"""


def _expand_on_devices(devices: int) -> dict:
    """``_MULTI_DEVICE_SCRIPT``'s output in a subprocess with ``devices`` host devices
    (pattern: tests/search/test_stage_a.py)."""
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
        [sys.executable, "-c", f"DEVICES = {devices}\n" + _MULTI_DEVICE_SCRIPT],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_device_count_invariant(tables):
    # 3 divides neither CHUNK nor the chunk sizes, so the padding differs.
    one = _expand_on_devices(1)
    assert one["store"] and one["children"]
    assert _expand_on_devices(3) == one
