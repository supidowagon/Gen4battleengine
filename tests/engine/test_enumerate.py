"""Exact outcome enumeration (``engine.enumerate``): the laws match the decomp's draws
as ``sample_outcome`` makes them, the relevance mask never drops a field ``step``
reads, and the enumerated children match sampled frequencies and the reference
engine."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.ai.policy import ai_action_distribution
from nuzline.data.tables import to_numpy
from nuzline.engine.actions import legal_actions, legal_opponent_actions
from nuzline.engine.enumerate import (
    WIDTH_CAP,
    SamplingOnly,
    enumerate_outcomes,
    outcome_laws,
)
from nuzline.engine.ids import EffectId, Status
from nuzline.engine.order import selected_move
from nuzline.engine.outcome import (
    Outcome,
    crit_odds,
    empty_outcome,
    hit_threshold,
    outcome_to_py,
    sample_outcome,
)
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.helpers import A, roark_real_state, roark_state
from tests.search.fixtures import shinx_vs_cranidos, spike_cannon_root, spore_1v1
from tests.strategies import any_outcome, legal_action_pair, outcome_law_state

CHUNK = 1024  # one compiled step program for every batch in this file
EMPTY = Outcome(*(np.asarray(f) for f in empty_outcome()))
HYPOTHESIS = {
    "deadline": None,
    "suppress_health_check": [
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
}

_laws = jax.jit(outcome_laws)
_step = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_sample = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))


def laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, actions))


def enumerate_pair(tables, state, actions):
    return enumerate_outcomes(laws_of(tables, state, actions))


def rows(outcomes: Outcome, idx) -> Outcome:
    return Outcome(*(np.asarray(f)[idx] for f in outcomes))


def step_bytes(tables, state, actions, outcomes: Outcome) -> list[bytes]:
    """The state bytes ``step`` gives for each outcome row, in fixed ``CHUNK`` batches
    (padding = copies of row 0)."""
    n = len(outcomes.hit)
    out = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        stepped = _step(tables, state, actions, rows(outcomes, pad))
        flat = np.concatenate(
            [
                np.ascontiguousarray(np.asarray(f)).reshape(CHUNK, -1).view(np.uint8)
                for f in stepped
            ],
            axis=1,
        )
        out += [flat[i].tobytes() for i in range(len(idx))]
    return out


def state_bytes(state, template) -> bytes:
    """``state`` serialized as ``step_bytes`` does, with ``template``'s dtypes."""
    return b"".join(
        np.ascontiguousarray(np.asarray(f, np.asarray(t).dtype)).tobytes()
        for f, t in zip(state, template)
    )


def children(tables, state, actions) -> dict[bytes, float]:
    outcomes, probs = enumerate_pair(tables, state, actions)
    merged: dict[bytes, float] = {}
    for key, p in zip(step_bytes(tables, state, actions, outcomes), probs):
        merged[key] = merged.get(key, 0.0) + float(p)
    return merged


def sampled(tables, state, actions, n) -> Counter:
    keys = jax.random.split(jax.random.key(0), n)
    outcomes = jax.tree.map(np.asarray, _sample(keys, tables, state, actions))
    return Counter(step_bytes(tables, state, actions, outcomes))


def check_frequencies(expected: dict[bytes, float], counts: Counter, n: int) -> None:
    """Every sampled child is enumerated; children with ``n*p >= 5`` match their
    frequency within 5 sigma, the rest are pooled and checked the same way."""
    assert set(counts) <= set(expected), "a sampled child is missing from the enumeration"

    def ok(p, k):
        return abs(k / n - p) <= 5 * math.sqrt(max(p * (1 - p), 0.0) / n) + 1e-9

    pooled_p, pooled_k = 0.0, 0
    for key, p in expected.items():
        if n * p >= 5:
            assert ok(p, counts[key]), (p, counts[key] / n)
        else:
            pooled_p, pooled_k = pooled_p + p, pooled_k + counts[key]
    assert ok(pooled_p, pooled_k), (pooled_p, pooled_k / n)


def legal_pairs(tables, state):
    player = np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))
    opp = np.flatnonzero(np.asarray(legal_opponent_actions(tables, state)))
    return [(int(a), int(b)) for a in player for b in opp]


# --- exact cases -------------------------------------------------------------------


def test_probs_sum_to_one(tables, names):
    s = roark_real_state(tables, names)
    ai = np.asarray(ai_action_distribution(tables, s))
    pairs = [(a, b) for a, b in legal_pairs(tables, s) if ai[b] > 0]
    assert pairs
    for pair in pairs:
        outcomes, probs = enumerate_pair(tables, s, np.asarray(pair, np.int32))
        assert abs(probs.sum() - 1) < 1e-12, pair
        assert (probs > 0).all() and len(probs) <= 16384, pair
        assert all(np.asarray(f).shape[0] == len(probs) for f in outcomes)


def test_switches_only_fixed(tables, names):
    s = roark_real_state(tables, names)
    pairs = [(a, b) for a, b in legal_pairs(tables, s) if 4 <= a < 10 and 4 <= b < 10]
    assert pairs
    for pair in pairs:
        acts = np.asarray(pair, np.int32)
        outcomes, probs = enumerate_pair(tables, s, acts)
        for name, field, empty in zip(Outcome._fields, outcomes, EMPTY):
            field = np.asarray(field)
            if name in ("speed_tie", "end_speed_tie"):
                values = set(field.tolist())
                assert values <= {0, 1}
                if len(values) == 2:
                    assert math.isclose(probs[field == 0].sum(), probs[field == 1].sum())
            else:
                assert (field == empty).all(), (pair, name)
        merged = children(tables, s, acts)
        assert len(merged) == 1 and abs(next(iter(merged.values())) - 1) < 1e-12


def test_speed_tie_enumerated(tables, names):
    s = roark_state(tables, names, equal_speed=True)
    outcomes, probs = enumerate_pair(tables, s, A("Scratch", "Tackle", s, names))
    tie = np.asarray(outcomes.speed_tie)
    assert set(tie.tolist()) == {0, 1}
    assert abs(math.fsum(probs[tie == 0]) - 0.5) < 1e-15
    assert abs(math.fsum(probs[tie == 1]) - 0.5) < 1e-15


def test_width_cap(tables, names):
    s = spike_cannon_root(tables, names)
    with pytest.raises(SamplingOnly, match=str(WIDTH_CAP)) as info:
        enumerate_pair(tables, s, A("Spike Cannon", "Spike Cannon", s, names))
    assert "width cap" in info.value.reason


def test_sleep_duration_enumerated(tables, names):
    s = spore_1v1(tables, names)
    acts = A("Spore", "Tackle", s, names)
    outcomes, probs = enumerate_pair(tables, s, acts)
    duration = np.asarray(outcomes.duration)[:, 0]
    for d in (2, 3, 4, 5):  # Random 3, 2 (subscript_fall_asleep.s)
        assert abs(math.fsum(probs[duration == d]) - 0.25) < 1e-15
    asleep: Counter = Counter()
    for key, p in children(tables, s, acts).items():
        state = from_bytes_like(s, key)
        assert int(state["status"][1, 0]) == Status.SLEEP
        asleep[int(state["status_counter"][1, 0])] += p
    assert len(asleep) == 4 and all(abs(p - 0.25) < 1e-12 for p in asleep.values())


def from_bytes_like(template, key: bytes) -> dict[str, np.ndarray]:
    """The fields of a state serialized by ``step_bytes``, shaped like ``template``."""
    out, offset = {}, 0
    for name, f in zip(template._fields, template):
        f = np.asarray(f)
        out[name] = np.frombuffer(key, f.dtype, f.size, offset).reshape(f.shape)
        offset += f.nbytes
    return out


@pytest.mark.parametrize(
    ("case", "player", "opp"),
    [
        ("roark", "Ember", "Rock Throw"),
        ("roark", 4 + 1, "Rock Throw"),  # switch to Prinplup
        ("shinx", "Spark", "Tackle"),
        ("shinx_slower", "Spark", "Tackle"),
    ],
)
def test_matches_sampled_frequencies(tables, names, case, player, opp):
    if case == "roark":
        s = roark_real_state(tables, names)
    else:
        s = shinx_vs_cranidos(tables, names, slower_opp=case == "shinx_slower")
    acts = A(player, opp, s, names)
    expected = children(tables, s, acts)
    assert abs(math.fsum(expected.values()) - 1) < 1e-12
    if case == "shinx_slower":
        paralyzed = [from_bytes_like(s, k)["status"][1, 0] == Status.PARALYSIS for k in expected]
        assert any(paralyzed)
    n = 20_000
    check_frequencies(expected, sampled(tables, s, acts, n), n)


# --- generated cases ---------------------------------------------------------------


def decomp_law(tables, state, actions, name: str, index: tuple) -> dict[int, Fraction]:
    """The law ``sample_outcome`` draws element ``name[index]`` from, transcribed from
    outcome.py:143-193 (not from ``engine.enumerate``)."""
    side = index[0] if index else 0
    move = int(selected_move(tables, state, side, actions[side])[0])

    def coin(p: Fraction) -> dict[int, Fraction]:
        return {1: p, 0: 1 - p}

    if name in ("speed_tie", "end_speed_tie", "confusion_hit"):
        return coin(Fraction(1, 2))
    if name == "hit":
        t = int(hit_threshold(tables, state, actions)[side])
        return coin(Fraction(min(max(t, 0), 100), 100))
    if name == "crit":
        lucky = int(state.lucky_chant[1 - side]) != 0
        return coin(
            Fraction(0) if lucky else Fraction(1, int(crit_odds(tables, state, actions)[side]))
        )
    if name == "roll":
        return {r: Fraction(1, 16) for r in range(16)}
    if name == "secondary":
        chance = int(tables.move_effect_chance[move])
        return coin(Fraction(1) if chance == 0 else Fraction(chance, 100))
    if name == "contact_proc":
        return coin(Fraction(3, 10))
    if name == "n_hits":
        return {2: Fraction(3, 8), 3: Fraction(3, 8), 4: Fraction(1, 8), 5: Fraction(1, 8)}
    if name == "duration":
        effect = int(tables.move_effect[move])
        if effect == EffectId.TAUNT:
            return {d: Fraction(1, 3) for d in (3, 4, 5)}
        low = 3 if effect == EffectId.TRAP else 2
        return {d: Fraction(1, 4) for d in range(low, low + 4)}
    if name == "full_para":
        return coin(Fraction(1, 4))
    if name == "thaw":
        return coin(Fraction(1, 5))
    return {0: Fraction(1)}  # flinch_item, quick_claw


@settings(max_examples=50, **HYPOTHESIS)
@given(data=st.data())
def test_field_marginals_match_decomp_laws(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    try:
        outcomes, probs = enumerate_outcomes(laws)
    except SamplingOnly:
        return  # above WIDTH_CAP
    assert abs(math.fsum(probs) - 1) < 1e-12
    for name, field, mask, empty in zip(Outcome._fields, outcomes, laws.relevant, EMPTY):
        field = np.asarray(field)
        for index in np.ndindex(empty.shape):
            column = field[(slice(None), *index)]
            if not mask[index]:
                assert (column == empty[index]).all(), (name, index)
                continue
            law = decomp_law(tables, s, acts, name, index)
            assert set(column.tolist()) <= set(law), (name, index)
            for value, p in law.items():
                assert abs(math.fsum(probs[column == value]) - float(p)) <= 1e-15, (name, index)


@settings(max_examples=200, **HYPOTHESIS)
@given(data=st.data())
def test_mask_sound(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    arbitrary = data.draw(any_outcome())
    seed = data.draw(st.integers(0, 2**31 - 1))
    laws = laws_of(tables, s, acts)
    try:
        outcomes, _ = enumerate_outcomes(laws)
        k = len(outcomes.hit)
        # Every row when the enumeration fits one padded chunk (no new compile); above
        # that, CHUNK // 2 distinct random rows, so a relevance hole confined to a few
        # rows of a small enumeration cannot be missed.
        idx = (
            np.arange(k)
            if k <= CHUNK
            else np.random.default_rng(seed).choice(k, CHUNK // 2, replace=False)
        )
        picked = rows(outcomes, idx)
    except SamplingOnly:
        # Above the cap: CHUNK // 2 law draws with the irrelevant elements at their empty
        # value, which is what an enumerated row holds. 5000 keys reuse the compiled
        # sampler shape of test_matches_sampled_frequencies_random.
        keys = jax.random.split(jax.random.key(seed), 5000)
        drawn = jax.tree.map(lambda f: np.asarray(f)[: CHUNK // 2], _sample(keys, tables, s, acts))
        picked = Outcome(*(np.where(m, f, e) for f, m, e in zip(drawn, laws.relevant, EMPTY)))
    changed = Outcome(
        *(np.where(m, f, np.asarray(a)) for f, m, a in zip(picked, laws.relevant, arbitrary))
    )
    n = len(picked.hit)
    both = Outcome(*(np.concatenate([a, b]) for a, b in zip(picked, changed)))
    stepped = step_bytes(tables, s, acts, both)
    assert stepped[:n] == stepped[n:]


@settings(max_examples=15, derandomize=True, **HYPOTHESIS)
@given(data=st.data())
def test_matches_sampled_frequencies_random(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    try:
        outcomes, _ = enumerate_pair(tables, s, acts)
    except SamplingOnly:
        assume(False)
    assume(len(outcomes.hit) <= 16384)  # stepping time
    n = 5000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


@settings(max_examples=10, **HYPOTHESIS)
@given(data=st.data())
def test_reference_child_set(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    try:
        outcomes, probs = enumerate_pair(tables, s, acts)
    except SamplingOnly:
        assume(False)
    assume(len(probs) <= 256)  # the reference engine takes ~1.5 ms a row
    ref_tables, ref_state = to_numpy(tables), to_ref(s)
    pair = tuple(int(x) for x in acts)
    ref: dict[bytes, float] = {}
    for k, p in enumerate(probs):
        out = ref_step(ref_tables, ref_state, pair, outcome_to_py(rows(outcomes, k)))
        key = state_bytes(from_ref(out), s)
        ref[key] = ref.get(key, 0.0) + float(p)
    jax_children = children(tables, s, acts)
    assert set(ref) == set(jax_children)
    for key, p in ref.items():
        assert abs(p - jax_children[key]) < 1e-15
