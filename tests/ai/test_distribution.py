"""The factored ``ai_action_distribution`` against its oracle, the 3840-outcome
enumeration ``ai_action_distribution_enumerated``: equal to atol 1e-6 on random states
(Roark's real party with random leads, HP, bags, ``move_hit`` and knowledge, and the
random M1 states of ``tests.strategies``), NaN on exactly the same refused states."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.ai.outcome import TIEBREAK_MOD, TIEBREAK_P
from nuzline.ai.policy import (
    _refused,
    ai_action_distribution,
    ai_action_distribution_enumerated,
    ai_refusals,
    ai_scores,
)
from nuzline.engine.state import BAG_SLOTS, SLOTS, BattleState
from tests.ai.fixtures import CRANIDOS, outcome, trace_a, trace_e, trace_f, versus
from tests.strategies import draw_seen_moves, supported_state

# Traced (jit), so a refused state gives NaN instead of raising, as in Stage A.
factored = jax.jit(ai_action_distribution)

_LEADS = (None, "Starly", "Prinplup", "Grotle", "Shinx", "Budew")
_HIT_MOVES = ("Wing Attack", "Tackle", "Ember", "Bubble", "Razor Leaf", "Leer")
_SETTINGS = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _flags(draw, shape) -> np.ndarray:
    size = int(np.prod(shape))
    return np.reshape(draw(st.lists(st.integers(0, 1), min_size=size, max_size=size)), shape)


@st.composite
def roark_states(draw, tables, names) -> BattleState:
    """Roark's real party (any of the three as the active) against the sample party led
    by any of its members: random HP on both sides (benched Pokemon may have fainted),
    turn, ``move_hit`` (so the voluntary-switch party passes run), trapped turns and
    attack stages (the step 1 and 9 gates), the bag (Potions or empty slots, so item use
    is exercised) and the AI's knowledge of the player."""
    # Starly's Wing Attack against Roark's resisting bench is what makes the party
    # passes switch (roark.md 13 trace F), so it is drawn half the time.
    lead = draw(st.one_of(st.just("Starly"), st.sampled_from(_LEADS)))
    s = versus(tables, names, draw(st.integers(0, 2)), lead)
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    for side in (0, 1):
        active = int(a["active"][side])
        for slot in range(SLOTS):
            if a["species"][side, slot] > 0:
                low = 1 if slot == active else 0
                a["hp"][side, slot] = draw(st.integers(low, int(a["max_hp"][side, slot])))
    a["turn"][...] = draw(st.integers(0, 5))
    hit = [0, *(names.move(m) for m in _HIT_MOVES)]
    a["move_hit"][1] = draw(st.one_of(st.just(names.move("Wing Attack")), st.sampled_from(hit)))
    a["trapped"][1] = draw(st.sampled_from([0, 0, 0, 2]))
    a["stages"][1, 1:3] = draw(st.lists(st.integers(0, 2), min_size=2, max_size=2))
    count = draw(st.integers(0, BAG_SLOTS))
    bag = draw(st.lists(st.sampled_from([0, names.item("Potion")]), min_size=count, max_size=count))
    a["trainer_items"][1] = bag + [0] * (BAG_SLOTS - count)
    a["trainer_item_count"][1] = count
    draw_seen_moves(draw, a)
    a["ai_ability_known"][...] = _flags(draw, a["ai_ability_known"].shape)
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})


def _both(tables, s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(factored(tables, s)), np.asarray(ai_action_distribution_enumerated(tables, s))


def _assert_same(new: np.ndarray, old: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(new), np.isnan(old))
    np.testing.assert_allclose(new, old, atol=1e-6)


@pytest.mark.slow
@settings(_SETTINGS, max_examples=300)
@given(data=st.data())
def test_factored_distribution_matches_enumeration(tables, names, data):
    _assert_same(*_both(tables, data.draw(roark_states(tables, names))))


@pytest.mark.slow
@settings(_SETTINGS, max_examples=200)
@given(data=st.data())
def test_factored_distribution_matches_enumeration_on_random_m1_states(tables, names, data):
    """Random M1 parties: mostly refused (NaN in both), the rest any species' moves."""
    _assert_same(*_both(tables, data.draw(supported_state(tables, names))))


@pytest.mark.slow
@settings(_SETTINGS, max_examples=60)
@given(data=st.data())
def test_distribution_sums_to_one(tables, names, data):
    s = data.draw(roark_states(tables, names))
    assume(not ai_refusals(tables, s))
    d = np.asarray(ai_action_distribution(tables, s), np.float64)
    assert np.all(d >= 0)
    assert abs(d.sum() - 1.0) <= 1e-6


def test_matches_enumeration_on_switch_and_potion_traces(tables, names):
    potion, switch = trace_e(11)(tables, names), trace_f(tables, names)
    for s in (potion, switch):
        new, old = _both(tables, s)
        _assert_same(new, old)
        assert not np.isnan(new).any()
    assert abs(_both(tables, potion)[0][10] - 1) <= 1e-6  # the Potion
    assert 0 < _both(tables, switch)[0][4] < 1  # a voluntary switch


@pytest.mark.fullset
def test_factored_distribution_refused_states_nan(tables, names):
    scoring = trace_a(tables, names)
    scoring = scoring._replace(moves=scoring.moves.at[1, 0, 2].set(names.move("Ember")))
    assert bool(_refused(tables, scoring).scoring)
    new, old = _both(tables, scoring)
    assert np.isnan(new).all()
    assert np.isnan(old).all()


def test_tie_shares_use_tiebreak_probabilities(tables, names):
    """Three Tackles tie on every outcome, so slot k gets the tiebreak residues t with
    ``t % 3 == k`` (``RandNext % 12``, uneven over a u16), not exactly 1/3 each."""
    s = versus(tables, names, CRANIDOS)
    tackle = jnp.asarray([names.move("Tackle")] * 3 + [0], jnp.int16)
    s = s._replace(moves=s.moves.at[1, CRANIDOS].set(tackle))
    assert ai_refusals(tables, s) == []
    for guess in (0, 1):
        scores = np.asarray(ai_scores(tables, s, outcome(ability_guess=guess)))
        assert scores[0] == scores[1] == scores[2] > scores[3], scores
    t = np.arange(TIEBREAK_MOD)
    expected = [TIEBREAK_P[t % 3 == k].sum() for k in range(3)]
    assert expected[0] != expected[2]  # the uneven residues are what is checked
    d = np.asarray(ai_action_distribution(tables, s), np.float64)
    np.testing.assert_allclose(d[:3], expected, atol=1e-6, rtol=0)
    assert d[3:].sum() == 0
