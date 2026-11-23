"""``expand.laws_rows``' call size (E3-B2-35): calls of ``min(LAWS_CHUNK, bucket_rows(N))``
rows, the same laws at any call size, and the memory of one capped call."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine import state as st
from nuzline.engine.actions import legal_actions, legal_opponent_actions
from nuzline.engine.enumerate import call_variants
from nuzline.search import expand as expand_mod
from nuzline.search.expand import LAWS_CHUNK, calls_rows, laws_rows
from nuzline.search.rows import take
from tests.engine.helpers import roark_real_state
from tests.search.fixtures import shinx_vs_cranidos

# Bound by purpose: one capped ``_laws`` call (LAWS_CHUNK = 4096 rows, 1 device), temp + output,
# must fit the owner's 32 GB PC beside the CLI's other programs. Measured
# (``compile().memory_analysis()``): 395916d1 + the cap: temp 17,196,603,408 B + output
# 493,122,616 B = 17.69 GB; 153dc32d (after H-1 7074311e, Future Sight's hit law per
# post-Acid-Rain mask) and 01a94ae8 (train/wave3, + e2-lawsbag): temp 17,229,453,328 B + output
# 953,185,336 B = 18,182,638,664 B = 18.18 GB. The bound moves when state or outcome fields are
# added (the output is the OutcomeLaws fields per row). claude/e4-future-values b5f837eb (on main
# d2ec0fc1; OutcomeLaws.future_values bool[B, 101] in place of future_thresholds / future_possible):
# temp 17,229,453,328 B + output 34,190,408 B = 17,263,643,736 B (pod run q-e4-87b9e0a). Bound: that
# + 5%. E2's sorted speed-tie test (D6-22, ``pair_ties``) lowers the temp to ~2.17 MB per row, the
# total to ~9 GB: lower the bound then.
LAWS_CALL_BYTES = 18_126_826_000


def _call_size(monkeypatch, n: int) -> int:
    """The ``per_call`` ``laws_rows`` hands ``_run`` for ``n`` rows (no program runs)."""
    seen = []
    monkeypatch.setattr(expand_mod, "_run", lambda *a, **k: seen.append(a[6]))
    laws_rows(None, None, np.zeros((n, 2), np.int32), np.zeros((n, 2), np.int32))
    return seen[0]


def test_laws_call_size(monkeypatch):
    """Fails before the cap (``per_call`` was None, i.e. ``CHUNK`` = 16384 rows always)."""
    sizes = [_call_size(monkeypatch, n) for n in (1, 256, 257, 1000, 4096, 4097, 50000)]
    assert sizes == [256, 256, 512, 1024, 4096, LAWS_CHUNK, LAWS_CHUNK]
    assert LAWS_CHUNK == 4096


def _specs(rows: int):
    spec = st.BattleState(
        **{k: jax.ShapeDtypeStruct((rows, *s), d) for k, (s, d) in st.LAYOUT.items()}
    )
    pair = jax.ShapeDtypeStruct((rows, 2), jnp.int32)
    return spec, pair, pair, jax.ShapeDtypeStruct((rows, 2), jnp.bool_)


@pytest.mark.slow
def test_laws_shapes_at_the_cap(tables):
    """Invariant: every ``OutcomeLaws`` field at ``LAWS_CHUNK`` rows has the per-row shape and
    dtype it has at ``CHUNK`` rows, all int32 or bool (so no call-size rounding). Slow: 71 s on the pod
    (e3-lm-b694acd)."""
    capped = jax.eval_shape(expand_mod._laws, tables, *_specs(LAWS_CHUNK))
    full = jax.eval_shape(expand_mod._laws, tables, *_specs(expand_mod.CHUNK))
    assert capped._fields == full._fields
    for name, a, b in zip(capped._fields, capped, full, strict=True):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            assert x.shape == (LAWS_CHUNK, *y.shape[1:]), name
            assert x.dtype == y.dtype, name
            assert x.dtype in (jnp.int32, jnp.bool_), name


def _frontier(tables, names):
    """Laws rows as ``expand`` builds them (``calls_rows`` + ``call_variants``) for every
    legal pair of three roots: more rows than the small call size below."""
    roots = [
        shinx_vs_cranidos(tables, names),
        shinx_vs_cranidos(tables, names, slower_opp=True),
        roark_real_state(tables, names),
    ]
    states = jax.tree.map(lambda *fs: np.stack([np.asarray(f) for f in fs]), *roots)
    index, actions = [], []
    for i, s in enumerate(roots):
        for a in np.flatnonzero(np.asarray(legal_actions(tables, s, 0))):
            for b in np.flatnonzero(np.asarray(legal_opponent_actions(tables, s))):
                index.append(i)
                actions.append((a, b))
    index = np.asarray(index, np.int64)
    actions = np.asarray(actions, np.int32)
    cl = calls_rows(tables, take(states, index), actions)
    variants = [call_variants(take(cl, r)) for r in range(len(index))]
    req = np.asarray([r for r, vs in enumerate(variants) for _ in vs], np.int64)
    call = np.asarray([v.call for vs in variants for v in vs], np.int32).reshape(-1, 2)
    taken = np.asarray([v.taken for vs in variants for v in vs], bool).reshape(-1, 2)
    return take(states, index[req]), actions[req], call, taken


@pytest.mark.slow
def test_laws_rows_do_not_depend_on_the_call_size(tables, names, monkeypatch):
    """Invariant: the same frontier through the default call (one ``bucket_rows`` call) and
    through 32-row calls (the last one padded) gives identical laws, field by field. Slow:
    two ``_laws`` compiles of ~65 s each."""
    monkeypatch.setattr(expand_mod, "CHUNK", 1024)  # calls_rows' call; laws are bucketed
    rows = _frontier(tables, names)
    n = len(rows[1])
    # Not a multiple of 32: the last 32-row call is padded and trimmed.
    assert n > 32 and n % 32 != 0, n
    one = laws_rows(tables, *rows)
    small = laws_rows(tables, *rows, per_call=32)
    for name, a, b in zip(one._fields, one, small, strict=True):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            assert x.shape[0] == n, name
            assert x.dtype == y.dtype and x.shape == y.shape, name
            assert np.array_equal(x, y), name


@pytest.mark.slow
def test_laws_call_memory(tables, monkeypatch):
    """One ``_laws`` call at the size ``laws_rows`` picks for a large frontier, compiled on
    abstract inputs (nothing allocated): temp + output within ``LAWS_CALL_BYTES``. Fails
    before the cap (16384 rows: 68.8 GB temp). ~65 s compile, a few GB of compiler RAM."""
    rows = _call_size(monkeypatch, 50000)
    monkeypatch.undo()
    mem = jax.jit(expand_mod._laws).lower(tables, *_specs(rows)).compile().memory_analysis()
    total = mem.temp_size_in_bytes + mem.output_size_in_bytes
    assert total <= LAWS_CALL_BYTES, (rows, mem.temp_size_in_bytes, mem.output_size_in_bytes)
