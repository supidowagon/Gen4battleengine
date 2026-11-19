"""D3-1: the four-battler draws of ``tests/strategies.py`` (``four_state``, ``legal_actions_four``,
``any_outcome(fmt)``) hold the constraints their docstrings state. Fast tier: 25 examples each,
no jit."""

import jax
import numpy as np
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from nuzline.engine.actions import MOVE_ACTIONS, legal_choices
from nuzline.engine.format import DOUBLES, FORMATS, MULTI, SINGLES, TWO_TRAINERS, format_of, party
from nuzline.engine.outcome import empty_outcome
from tests.strategies import any_outcome, four_state, legal_actions_four

SMALL = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


def _owners(fmt, p):
    return [b for b in range(fmt.B) if fmt.party_of[b] == p]


@SMALL
@given(data=st.data())
def test_four_state_is_the_drawn_format(tables, names, data):
    fmt = data.draw(st.sampled_from((DOUBLES, TWO_TRAINERS, MULTI)))
    s = data.draw(four_state(tables, names, fmt))
    assert format_of(s) == fmt
    assert format_of(data.draw(four_state(tables, names))).B == 4


@SMALL
@given(data=st.data())
def test_no_two_battlers_of_a_party_share_a_slot(tables, names, data):
    s = data.draw(four_state(tables, names))
    fmt = format_of(s)
    for p in range(fmt.P):
        slots = [int(s.active[b]) for b in _owners(fmt, p)]
        assert len(set(slots)) == len(slots), (fmt, p, slots)
        for slot in slots:
            assert int(s.species[p, slot]) > 0


@SMALL
@given(data=st.data())
def test_a_zero_hp_active_has_no_replacement(tables, names, data):
    s = data.draw(four_state(tables, names))
    fmt = format_of(s)
    for b in range(fmt.B):
        p = party(fmt, b)
        if int(s.hp[p, int(s.active[b])]) > 0:
            continue
        # rule 3: the party's only other living Pokemon are the actives of the battlers
        # sending from it
        actives = {int(s.active[c]) for c in _owners(fmt, p)}
        living = {int(k) for k in np.flatnonzero((np.asarray(s.species[p]) > 0) & (np.asarray(s.hp[p]) > 0))}
        assert living <= actives, (fmt, b, living, actives)


@SMALL
@given(data=st.data())
def test_each_side_has_a_living_battler(tables, names, data):
    s = data.draw(four_state(tables, names))
    fmt = format_of(s)
    for side in (0, 1):
        alive = [
            int(s.hp[party(fmt, b), int(s.active[b])]) > 0 for b in range(fmt.B) if b & 1 == side
        ]
        assert any(alive), (fmt, side)


@SMALL
@given(data=st.data())
def test_every_battler_has_a_legal_pair(tables, names, data):
    s = data.draw(four_state(tables, names))
    for b in range(format_of(s).B):
        assert np.asarray(legal_choices(tables, s, b)).any(), b


@SMALL
@given(data=st.data())
def test_legal_actions_four_pairs_are_legal(tables, names, data):
    s = data.draw(four_state(tables, names))
    fmt = format_of(s)
    acts, targets = data.draw(legal_actions_four(tables, s))
    assert acts.shape == (fmt.B,) and targets.shape == (fmt.B,)
    for b in range(fmt.B):
        assert np.asarray(legal_choices(tables, s, b))[int(acts[b]), int(targets[b])], b
    # no two switches to one slot inside a party
    for p in range(fmt.P):
        own = _owners(fmt, p)
        picks = [int(acts[b]) for b in own if int(acts[b]) >= MOVE_ACTIONS]
        assert len(picks) == len(set(picks)), (fmt, p, picks)


def test_any_outcome_shapes_equal_empty_outcome_for_every_format():
    for fmt in FORMATS:
        @seed(0)
        @settings(max_examples=3, deadline=None, database=None, suppress_health_check=list(HealthCheck))
        @given(o=any_outcome(fmt))
        def check(o, fmt=fmt):
            ref = empty_outcome(fmt)
            for name, got, want in zip(o._fields, o, ref, strict=True):
                assert np.asarray(got).shape == np.asarray(want).shape, (fmt, name)

        check()


def _collect(strategy, value):
    drawn = []

    @seed(value)
    @settings(max_examples=25, deadline=None, database=None, suppress_health_check=list(HealthCheck))
    @given(o=strategy)
    def run(o):
        drawn.append(jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), o))

    run()
    return drawn


def test_singles_any_outcome_is_the_default_draw():
    assert _collect(any_outcome(), 7) == _collect(any_outcome(SINGLES), 7)
