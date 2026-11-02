"""Batched turn-0 states (plan 4c Task 2)."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.party import load_party
from nuzline.engine.ids import Status
from nuzline.engine.state import SLOTS, make_state
from nuzline.prep import states as states_module
from nuzline.prep.states import STATE_CHUNK, make_states
from nuzline.search.rows import row_bytes
from tests.engine.helpers import SAMPLE_PARTY, roark_party, roark_trainer

STATUSES = [Status.NONE, Status.POISON, Status.BURN, Status.PARALYSIS]


@st.composite
def team_and_budget(draw, tables, names):
    full = load_party(SAMPLE_PARTY, names)
    maxes = np.asarray(make_state(tables, full, roark_party(names)).max_hp)[0]
    order = draw(st.permutations(range(len(full))))
    n = draw(st.integers(1, len(full)))
    team = []
    for i in order[:n]:
        hp = draw(st.integers(1, int(maxes[i])))
        status = int(draw(st.sampled_from(STATUSES)))
        team.append(replace(full[i], current_hp=hp, status=status))
    budget = draw(st.integers(0, 3))
    protected = sorted(draw(st.sets(st.sampled_from(range(n)))))
    return team, budget, protected


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_make_states_matches_make_state(tables, names, data):
    opp, bag = roark_party(names), roark_trainer().items
    drawn = [data.draw(team_and_budget(tables, names)) for _ in range(3)]
    budget = drawn[0][1]
    teams = [team for team, _, _ in drawn]
    protected = [[s for s in prot if s < len(team)] for team, _, prot in drawn]
    batch = make_states(tables, teams, opp, opp_items=bag, budget=budget, protected=protected)
    assert len(batch.hp) == len(teams)
    rows = row_bytes(batch, len(teams))
    for i, team in enumerate(teams):
        ref = make_state(tables, list(team), opp, opp_items=bag).with_budget(budget, protected[i])
        ref_row = row_bytes(type(ref)(*(np.asarray(f)[None] for f in ref)), 1)[0]
        assert rows[i].tobytes() == ref_row.tobytes()


def test_make_states_chunks(tables, names, monkeypatch):
    team = load_party(SAMPLE_PARTY, names)[:2]
    calls = []
    real = states_module.cached_export

    def counting(fn, args, **kw):
        calls.append((kw["name"], tuple(a.shape for a in __import__("jax").tree.leaves(args))))
        return real(fn, args, **kw)

    monkeypatch.setattr(states_module, "cached_export", counting)
    n = STATE_CHUNK + 1
    batch = make_states(tables, [team] * n, roark_party(names), opp_items=roark_trainer().items)
    assert len(batch.hp) == n
    assert [c[0] for c in calls] == ["make_states", "make_states"]
    assert calls[0][1] == calls[1][1]


def test_make_states_rejects_bad_budget(tables, names):
    team = load_party(SAMPLE_PARTY, names)[:2]
    opp = roark_party(names)
    with pytest.raises(ValueError):
        make_states(tables, [team], opp, budget=4)
    fainted = [team[0], replace(team[1], current_hp=0)]
    with pytest.raises(ValueError):
        make_states(tables, [fainted], opp, protected=[[1]])
    with pytest.raises(ValueError):
        make_states(tables, [team], opp, protected=[[SLOTS]])
