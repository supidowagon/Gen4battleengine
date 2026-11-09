"""Exact outcome enumeration (``engine.enumerate``): the laws match the decomp's draws
as ``sample_outcome`` makes them, the relevance mask never drops a field ``step``
reads, and the enumerated children match sampled frequencies and the reference
engine."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import partial

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from nuzline.ai.policy import ai_action_distribution
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.actions import legal_actions, legal_opponent_actions
from nuzline.engine.enumerate import (
    WIDTH_CAP,
    SamplingOnly,
    enumerate_drawn,
    enumerate_outcomes,
    outcome_laws,
)
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status
from nuzline.engine.observe import observation
from nuzline.engine.order import selected_move
from nuzline.engine.outcome import (
    Outcome,
    crit_odds,
    empty_outcome,
    hit_threshold,
    hit_thresholds,
    outcome_to_py,
    sample_outcome,
)
from nuzline.engine.state import BattleState, make_state
from nuzline.engine.step import step, step_events
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
_loose_laws = jax.jit(partial(outcome_laws, loose=True))
_step = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_sample = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))


def laws_of(tables, state, actions, *, loose=False):
    return jax.tree.map(np.asarray, (_loose_laws if loose else _laws)(tables, state, actions))


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


def test_magnitude_law_exact(tables, names):
    s = roark_state(
        tables, names, player_species="Geodude", player_moves=("Magnitude",), opp_moves=("Leer",)
    )
    acts = A("Magnitude", "Leer", s, names)
    laws = laws_of(tables, s, acts)
    assert laws.relevant.magnitude.tolist() == [True, False]
    outcomes, probs = enumerate_outcomes(laws)
    column = np.asarray(outcomes.magnitude)[:, 0]
    expected = dict(zip(range(4, 11), (Fraction(n, 100) for n in (5, 10, 20, 30, 20, 10, 5)),
                        strict=True))  # fmt: skip
    assert set(column.tolist()) == set(expected)
    for level, p in expected.items():
        assert math.fsum(probs[column == level]) == float(p), level
    # The foe's (irrelevant) element keeps its empty value.
    assert (np.asarray(outcomes.magnitude)[:, 1] == 4).all()
    # Magnitude is an ordinary damaging move: crit and roll are read.
    assert bool(laws.relevant.crit[0, 0]) and bool(laws.relevant.roll[0, 0])


def test_night_shade_has_no_crit_or_roll(tables, names):
    s = roark_state(
        tables, names, player_species="Spinarak", player_moves=("Night Shade",), opp_moves=("Leer",)
    )
    acts = A("Night Shade", "Leer", s, names)
    laws = laws_of(tables, s, acts)
    assert not laws.relevant.crit[0].any() and not laws.relevant.roll[0].any()
    assert not laws.relevant.magnitude.any()
    assert bool(laws.relevant.hit[0, 0])
    outcomes, probs = enumerate_outcomes(laws)
    assert (np.asarray(outcomes.crit)[:, 0] == 0).all()
    assert (np.asarray(outcomes.roll)[:, 0] == 15).all()
    assert abs(math.fsum(probs) - 1) < 1e-12


def test_confusion_secondary_and_duration_laws(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Venonat",
        player_moves=("Confusion",),
        opp_species="Geodude",
        opp_moves=("Leer",),
    )
    acts = A("Confusion", "Leer", s, names)
    laws = laws_of(tables, s, acts)
    assert int(laws.secondary_chance[0]) == 10
    assert bool(laws.relevant.secondary[0, 0]) and bool(laws.relevant.duration[0])
    # The foe (not the Confusion user) may hit itself the turn it is confused.
    assert laws.relevant.confusion_hit.tolist() == [False, True]
    outcomes, probs, drawn = enumerate_drawn(laws)
    secondary = np.asarray(outcomes.secondary)[:, 0, 0]
    assert math.fsum(probs[secondary == 1]) == pytest.approx(0.10)
    # (secondary, duration): the duration is read only when the confusion procs.
    read = np.asarray(drawn.duration)[:, 0]
    assert (read == (secondary == 1)).all()
    duration = np.asarray(outcomes.duration)[:, 0]
    assert set(duration[read].tolist()) == {2, 3, 4, 5}
    for d in (2, 3, 4, 5):
        assert math.fsum(probs[read & (duration == d)]) == pytest.approx(0.10 * 0.25)


def test_disable_duration_law(tables, names):
    # BtlCmd_TryDisable (battle_script.c:4590): RandNext() % 4 + 3, uniform on 3-6.
    s = roark_state(
        tables,
        names,
        player_species="Psyduck",
        player_moves=("Disable",),
        opp_species="Geodude",
        opp_moves=("Rock Throw",),
    )
    laws = laws_of(tables, s, A("Disable", "Rock Throw", s, names))
    assert bool(laws.relevant.duration[0]) and not bool(laws.relevant.duration[1])
    outcomes, probs, drawn = enumerate_drawn(laws)
    # Disable (80 %) reads its duration only when it hits.
    read = np.asarray(drawn.duration)[:, 0]
    assert (read == (np.asarray(outcomes.hit)[:, 0, 0] != 0)).all()
    duration = np.asarray(outcomes.duration)[:, 0]
    assert set(duration[read].tolist()) == {3, 4, 5, 6}
    hits = math.fsum(probs[read])
    for d in (3, 4, 5, 6):
        assert math.fsum(probs[read & (duration == d)]) == pytest.approx(0.25 * hits)


def test_shed_skin_marginal_is_three_tenths(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_ability="Shed Skin",
        player_moves=("Growl",),
    )
    laws = laws_of(tables, s, A("Growl", "Tackle", s, names))
    assert laws.relevant.shed_skin.tolist() == [True, False]
    outcomes, probs = enumerate_outcomes(laws)
    shed = np.asarray(outcomes.shed_skin)[:, 0]
    assert set(shed.tolist()) == {0, 1}
    assert math.fsum(probs[shed == 1]) == pytest.approx(0.3, abs=1e-15)
    assert math.fsum(probs[shed == 0]) == pytest.approx(0.7, abs=1e-15)
    # an incoming holder counts as well; a holder-free state never rolls it
    plain = roark_state(tables, names)
    assert not laws_of(tables, plain, A(0, "Tackle", plain, names)).relevant.shed_skin.any()
    bench = plain._replace(ability=plain.ability.at[0, 1].set(AbilityId.SHED_SKIN))
    incoming = laws_of(tables, bench, A(4 + 1, "Tackle", bench, names))
    assert incoming.relevant.shed_skin.tolist() == [True, False]


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
        # The roll u = RandNext() % 100 + 1 (bcp.c:2976), each of 1-100 at 1/100, grouped
        # by whether it hits at each threshold the move can have: a group is its smallest
        # roll, or 0 when it hits at none (outcome.hit_class).
        ts, ok = (np.asarray(x)[side] for x in hit_thresholds(tables, state, actions))
        assert int(ts[0]) == int(hit_threshold(tables, state, actions)[side])
        groups: dict[tuple, list[int]] = {}
        for u in range(1, 101):
            groups.setdefault(tuple(u <= t for t in ts[ok]), []).append(u)
        return {(min(us) if any(key) else 0): Fraction(len(us), 100) for key, us in groups.items()}
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
        low = 3 if effect in (EffectId.TRAP, EffectId.DISABLE) else 2
        return {d: Fraction(1, 4) for d in range(low, low + 4)}
    if name == "full_para":
        return coin(Fraction(1, 4))
    if name == "thaw":
        return coin(Fraction(1, 5))
    if name == "shed_skin":  # ShedSkin (battle_lib.c:3567): RandNext() % 10 < 3
        return coin(Fraction(3, 10))
    if name == "magnitude":  # BtlCmd_CalcMagnitude (battle_script.c:5820-5852)
        return dict(zip(range(4, 11), (Fraction(n, 100) for n in (5, 10, 20, 30, 20, 10, 5)),
                        strict=True))  # fmt: skip
    return {0: Fraction(1)}  # flinch_item, quick_claw


@settings(max_examples=50, **HYPOTHESIS)
@given(data=st.data())
def test_field_marginals_match_decomp_laws(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    try:
        outcomes, probs, drawn = enumerate_drawn(laws)
    except SamplingOnly:
        return  # above WIDTH_CAP
    assert abs(math.fsum(probs) - 1) < 1e-12
    for name, field, mask, empty, got in zip(
        Outcome._fields, outcomes, laws.relevant, EMPTY, drawn, strict=True
    ):
        field, got = np.asarray(field), np.asarray(got)
        for index in np.ndindex(empty.shape):
            column = field[(slice(None), *index)]
            read = got[(slice(None), *index)]
            # A row that does not draw the element holds its empty value.
            assert (column[~read] == empty[index]).all(), (name, index)
            if not mask[index]:
                assert not read.any(), (name, index)
                continue
            # The draw is independent of the condition that gates it (which reads other
            # elements only): within the rows that draw it, the element has its law.
            law = decomp_law(tables, s, acts, name, index)
            assert set(column[read].tolist()) <= set(law), (name, index)
            total = math.fsum(probs[read])
            for value, p in law.items():
                got_p = math.fsum(probs[read & (column == value)])
                assert abs(got_p - float(p) * total) <= 1e-14, (name, index)


@settings(max_examples=200, **HYPOTHESIS)
@given(data=st.data())
def test_mask_sound(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    arbitrary = data.draw(any_outcome())
    seed = data.draw(st.integers(0, 2**31 - 1))
    laws = laws_of(tables, s, acts)
    try:
        outcomes, _, drawn = enumerate_drawn(laws)
        k = len(outcomes.hit)
        # Every row when the enumeration fits one padded chunk (no new compile); above
        # that, CHUNK // 2 distinct random rows, so a relevance hole confined to a few
        # rows of a small enumeration cannot be missed.
        idx = (
            np.arange(k)
            if k <= CHUNK
            else np.random.default_rng(seed).choice(k, CHUNK // 2, replace=False)
        )
        picked, kept = rows(outcomes, idx), rows(drawn, idx)
    except SamplingOnly:
        # Above the cap: CHUNK // 2 law draws with the irrelevant elements at their empty
        # value, which is what an enumerated row holds. 5000 keys reuse the compiled
        # sampler shape of test_matches_sampled_frequencies_random.
        keys = jax.random.split(jax.random.key(seed), 5000)
        sampled_ = jax.tree.map(
            lambda f: np.asarray(f)[: CHUNK // 2], _sample(keys, tables, s, acts)
        )
        picked = Outcome(*(np.where(m, f, e) for f, m, e in zip(sampled_, laws.relevant, EMPTY)))
        kept = Outcome(*(np.broadcast_to(m, np.shape(f)) for f, m in zip(picked, laws.relevant)))
    # Every element a row does not draw (irrelevant, or gated off by the values the row
    # drew) takes an arbitrary value: the stepped state must not change.
    changed = Outcome(*(np.where(m, f, np.asarray(a)) for f, m, a in zip(picked, kept, arbitrary)))
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


# --- order branches and conditional reads (K3b2 Task 2, rules §10.3 items 1-4) ------

BRUTE_MAX = 16384  # rows of the brute-force enumeration a generated case may step


def _turn_out(tables, state, actions, outcome):
    after, events = step_events(tables, state, actions, outcome)
    return after, observation(events, after)


_turns = jax.jit(jax.vmap(_turn_out, in_axes=(None, None, None, 0)))


def turn_children(tables, state, actions, outcomes: Outcome, probs) -> dict[bytes, float]:
    """``(state bytes, observation bytes) -> probability``: the merged children as
    ``search.expand`` keys them (before the opponent's replacement), in ``CHUNK``
    batches."""
    n = len(probs)
    merged: dict[bytes, list[float]] = {}
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        after, obs = _turns(tables, state, actions, rows(outcomes, pad))
        flat = np.concatenate(
            [
                np.ascontiguousarray(np.asarray(f)).reshape(CHUNK, -1).view(np.uint8)
                for f in (*after, obs)
            ],
            axis=1,
        )
        for i, row in enumerate(idx):
            merged.setdefault(flat[i].tobytes(), []).append(float(probs[row]))
    return {key: math.fsum(ps) for key, ps in merged.items()}


def assert_equals_bruteforce(tables, state, actions) -> tuple[int, int]:
    """The exact enumeration (order branches, conditional reads) and the brute-force one
    (``loose``: every element any order or event can read, as an independent product)
    give the same merged children. Returns both widths."""
    loose = laws_of(tables, state, actions, loose=True)
    brute_outcomes, brute_probs, brute_drawn = enumerate_drawn(loose)
    # The brute force is a plain product: every relevant element drawn in every row.
    for got, mask in zip(brute_drawn, loose.relevant, strict=True):
        assert (np.asarray(got) == np.asarray(mask)[None]).all()
    outcomes, probs = enumerate_outcomes(laws_of(tables, state, actions))
    assert len(probs) <= len(brute_probs)
    assert abs(math.fsum(probs) - 1) < 1e-12
    exact = turn_children(tables, state, actions, outcomes, probs)
    brute = turn_children(tables, state, actions, brute_outcomes, brute_probs)
    assert set(exact) == set(brute)
    for key, p in brute.items():
        assert abs(exact[key] - p) < 1e-12
    return len(probs), len(brute_probs)


# Moves of the order-gated and conditional reads: flinch (Headbutt, Zen Headbutt at 90 %,
# Bite, Fake Out), status inflicted directly or as a secondary (Thunder Wave, Spore, Spark,
# Ice Beam, Ember, Flame Wheel), confusion (Confusion, Psybeam, Supersonic, Confuse Ray), a
# duration (Bind, Taunt), plain and multi-hit damage and an accuracy drop.
ORDER_MOVES = (
    "Headbutt",
    "Zen Headbutt",
    "Bite",
    "Fake Out",
    "Thunder Wave",
    "Spore",
    "Spark",
    "Ice Beam",
    "Ember",
    "Flame Wheel",
    "Confusion",
    "Psybeam",
    "Supersonic",
    "Confuse Ray",
    "Bind",
    "Taunt",
    "Tackle",
    "Fury Swipes",
    "Sand-Attack",
)
FLINCH_MOVES = ("Headbutt", "Zen Headbutt", "Bite", "Fake Out")  # drawn twice as often
ORDER_ABILITIES = (
    AbilityId.INNER_FOCUS,
    AbilityId.MOLD_BREAKER,
    AbilityId.LIMBER,
    AbilityId.STATIC,
    AbilityId.STEADFAST,
    AbilityId.INSOMNIA,
)


@st.composite
def order_state(draw, tables, names) -> BattleState:
    """``outcome_law_state`` with an ``ORDER_MOVES`` move in each active Pokemon's slot
    0 (full PP) most of the time, sometimes an ``ORDER_ABILITIES`` ability, Fake Out's
    first turn half the time, Lucky Chant on both sides half the time, and in three cases
    of four equal effective speeds (the Spe stat copied, no Speed stage, no Tailwind, the
    same paralysis state; items, abilities and weather can still break the tie)."""
    s = draw(outcome_law_state(tables, names))
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    move_pp = np.asarray(tables.move_pp)
    for side in (0, 1):
        mon = int(a["active"][side])
        if draw(st.integers(0, 3)) > 0:
            move = names.move(draw(st.sampled_from(ORDER_MOVES + FLINCH_MOVES)))
            a["moves"][side, mon, 0] = move
            a["pp"][side, mon, 0] = move_pp[move]
            if a["disabled_move"][side] == move:
                a["disabled_move"][side], a["disable_turns"][side] = 0, 0
        if draw(st.integers(0, 3)) == 0:
            a["ability"][side, mon] = draw(st.sampled_from(ORDER_ABILITIES))
        if draw(st.booleans()):
            a["fake_out_turn"][side] = a["turn"]
    if draw(st.booleans()):  # no crits on either side: smaller brute-force products
        a["lucky_chant"][:] = 5
    if draw(st.integers(0, 3)) > 0:
        m0, m1 = int(a["active"][0]), int(a["active"][1])
        a["stats"][1, m1, Stat.SPE - 1] = a["stats"][0, m0, Stat.SPE - 1]
        a["stages"][:, Stat.SPE] = 0
        a["tailwind"][:] = 0
        para0 = a["status"][0, m0] == Status.PARALYSIS
        if (a["status"][1, m1] == Status.PARALYSIS) != para0:
            a["status"][1, m1] = a["status"][0, m0] if para0 else Status.NONE
            a["status_counter"][1, m1] = a["status_counter"][0, m0] if para0 else 0
    return BattleState(**{name: np.asarray(a[name]) for name in BattleState._fields})


@st.composite
def order_actions(draw, tables, state) -> np.ndarray:
    """Mostly each side's slot 0 (the ``ORDER_MOVES`` move) when legal, else any legal
    pair."""
    pair = np.asarray(draw(legal_action_pair(tables, state))).copy()
    legal = (legal_actions(tables, state, 0), legal_opponent_actions(tables, state))
    for side in (0, 1):
        if bool(np.asarray(legal[side])[0]) and draw(st.integers(0, 3)) > 0:
            pair[side] = 0
    return pair.astype(np.int32)


@settings(max_examples=200, **HYPOTHESIS)
@given(data=st.data())
def test_order_branches_equal_bruteforce(tables, names, data):
    s = data.draw(order_state(tables, names))
    acts = data.draw(order_actions(tables, s))
    try:
        brute = enumerate_outcomes(laws_of(tables, s, acts, loose=True))[1]
    except SamplingOnly:
        assume(False)
    assume(len(brute) <= BRUTE_MAX)
    width, _ = assert_equals_bruteforce(tables, s, acts)
    laws = laws_of(tables, s, acts)
    event(f"speed tie: {bool(laws.relevant.speed_tie)}")
    event(f"sure flinch: {bool(laws.reads.flinch_sure.any())}")
    event(f"narrower than brute force: {width < len(brute)}")


def _tie(s):
    return s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(s.stats[0, 0, Stat.SPE - 1]))


@pytest.mark.parametrize(
    ("player", "opp", "tie", "setup"),
    [
        # A tie between two flinch moves: each is read only by the side going first.
        ("Headbutt", "Zen Headbutt", True, None),
        # Psybeam's confusion is read by the foe only when Psybeam goes first; its duration
        # only when it procs; Zen Headbutt's flinch stops Psybeam when it goes first (Lucky
        # Chant on both sides keeps the brute force under the cap).
        ("Psybeam", "Zen Headbutt", True, "lucky_chant"),
        # Thunder Wave first: the foe's full paralysis is read only if it was inflicted.
        ("Thunder Wave", "Tackle", True, None),
        # Ice Beam's freeze then the foe's thaw roll; Spark on an already paralyzed foe.
        ("Ice Beam", "Tackle", True, None),
        ("Spark", "Tackle", False, "paralyzed"),
        # Inner Focus blocks Headbutt's flinch: the foe's move is always read.
        ("Headbutt", "Tackle", True, "inner_focus"),
        # A frozen foe hit by Flame Wheel; Spore into Insomnia; Supersonic (55 %).
        ("Flame Wheel", "Tackle", True, "frozen"),
        ("Spore", "Tackle", True, "insomnia"),
        ("Supersonic", "Zen Headbutt", True, None),
        # Review probes. Mold Breaker pierces Inner Focus: the flinch is sure again.
        ("Headbutt", "Tackle", True, "mold_breaker_vs_inner_focus"),
        # Fake Out's first turn into a Steadfast, confused (3) and paralyzed foe: a sure
        # flinch skips its confusion and paralysis rolls, not the Speed boost.
        ("Fake Out", "Tackle", True, "steadfast_confused_paralyzed"),
        # Pursuit against a switch, from either side, into a Static holder: the pre-switch
        # hit reads crit, roll and the contact roll with no accuracy roll.
        (4 + 1, "Pursuit", False, "static_player"),
        ("Pursuit", 4 + 1, False, "static_opp"),
        # A flinch into a full-HP Focus Sash holder; a foe that wakes this turn (counter 1)
        # and can then be flinched; Ember on a frozen foe (blocked, then the thaw by hit).
        ("Headbutt", "Tackle", True, "focus_sash"),
        ("Headbutt", "Tackle", True, "waking"),
        ("Ember", "Tackle", True, "frozen"),
    ],
)
def test_order_branches_cases(tables, names, player, opp, tie, setup):
    s = roark_state(
        tables,
        names,
        player_species="Venonat",
        player_moves=(player,) if isinstance(player, str) else None,
        opp_species="Cranidos",
        opp_moves=(opp,) if isinstance(opp, str) else None,
    )
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.STURDY))  # no Mold Breaker
    if tie:
        s = _tie(s)
    if setup in ("paralyzed", "frozen"):
        status = Status.PARALYSIS if setup == "paralyzed" else Status.FREEZE
        s = s._replace(status=s.status.at[1, 0].set(status))
    elif setup == "inner_focus":
        s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.INNER_FOCUS))
    elif setup == "insomnia":
        s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.INSOMNIA))
    elif setup == "lucky_chant":
        s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    elif setup == "mold_breaker_vs_inner_focus":
        ability = s.ability.at[1, 0].set(AbilityId.INNER_FOCUS)
        s = s._replace(ability=ability.at[0, 0].set(AbilityId.MOLD_BREAKER))
    elif setup == "steadfast_confused_paralyzed":
        s = s._replace(
            fake_out_turn=s.fake_out_turn.at[0].set(s.turn),
            ability=s.ability.at[1, 0].set(AbilityId.STEADFAST),
            confusion=s.confusion.at[1].set(3),
            status=s.status.at[1, 0].set(Status.PARALYSIS),
        )
    elif setup in ("static_player", "static_opp"):
        side = 0 if setup == "static_player" else 1
        s = s._replace(ability=s.ability.at[side, 0].set(AbilityId.STATIC))
    elif setup == "focus_sash":
        s = s._replace(item=s.item.at[1, int(s.active[1])].set(names.item("Focus Sash")))
    elif setup == "waking":
        s = s._replace(
            status=s.status.at[1, 0].set(Status.SLEEP),
            status_counter=s.status_counter.at[1, 0].set(1),
        )
    width, brute = assert_equals_bruteforce(tables, s, A(player, opp, s, names))
    if tie:
        assert width < brute


def test_read_builders_registered():
    # Every conditional-read builder is registered once the engine entry points are
    # imported (a builder left out by import order would silently drop its gates).
    import nuzline.search.expand
    import nuzline.search.stage_b  # noqa: F401
    from nuzline.engine.enumerate import _READ_BUILDERS

    assert [b.__name__ for b in _READ_BUILDERS] == ["_move_reads"]


def _venonat_vs_cranidos(tables, names, **kw):
    return roark_state(
        tables,
        names,
        player_species="Venonat",
        player_moves=("Psybeam",),
        opp_species="Cranidos",
        opp_moves=("Zen Headbutt",),
        **kw,
    )


def test_flinch_read_only_by_the_first_mover(tables, names):
    # Zen Headbutt is slower: its flinch roll can never matter.
    s = _venonat_vs_cranidos(tables, names)
    s = s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(1))
    laws = laws_of(tables, s, A("Psybeam", "Zen Headbutt", s, names))
    assert not bool(laws.relevant.speed_tie)
    assert not laws.relevant.secondary[1].any()
    assert bool(laws.relevant.secondary[0, 0])


def test_flinch_and_miss_skip_reads(tables, names):
    # Cranidos first with Zen Headbutt (90 %): a miss reads no crit, roll or flinch roll;
    # a sure flinch reads nothing of Venonat's Psybeam.
    s = _venonat_vs_cranidos(tables, names, opp_faster=True)
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.STURDY))
    acts = A("Psybeam", "Zen Headbutt", s, names)
    outcomes, probs, drawn = enumerate_drawn(laws_of(tables, s, acts))
    hit = np.asarray(outcomes.hit)[:, 1, 0]
    flinch = np.asarray(outcomes.secondary)[:, 1, 0]
    missed = hit == 0
    assert missed.any()
    for name in ("crit", "roll", "secondary"):
        assert not np.asarray(getattr(drawn, name))[missed, 1].any(), name
    flinched = ~missed & (flinch == 1)
    assert flinched.any()
    for name in ("hit", "crit", "roll", "secondary"):
        assert not np.asarray(getattr(drawn, name))[flinched, 0].any(), name
    assert not np.asarray(drawn.duration)[flinched, 0].any()
    assert math.fsum(probs[flinched]) == pytest.approx(0.9 * 0.2)


def _kaizo_spec(names, species, move, level):
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=(names.move(move), 0, 0, 0),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


def test_width_drops_for_goldeen_psybeam_vs_cranidos(kaizo_tables, kaizo_names):
    # Rules §10.2: Goldeen's Psybeam against Cranidos's Zen Headbutt at a speed tie (28
    # in the saves; here both Spe stats are set equal). Kaizo's Zen Headbutt is 100 %
    # accurate, so the turn-start product is 131072 (the rules' column B).
    t, n = kaizo_tables, kaizo_names
    goldeen = _kaizo_spec(n, "Goldeen", "Psybeam", 16)
    cranidos = _kaizo_spec(n, "Cranidos", "Zen Headbutt", 14)
    s = _tie(make_state(t, [goldeen], [cranidos]))
    acts = np.zeros(2, np.int32)
    brute = enumerate_outcomes(laws_of(t, s, acts, loose=True))[1]
    assert len(brute) == 131072
    _, probs = enumerate_outcomes(laws_of(t, s, acts))
    assert len(probs) <= 30000
    assert abs(math.fsum(probs) - 1) < 1e-12
