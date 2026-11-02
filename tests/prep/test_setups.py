"""Team setups, neighbours and batched setup states (plan 4c Task 5)."""

from dataclasses import replace
from math import comb

import numpy as np

from nuzline.engine.state import make_state
from nuzline.prep.problem import Resources
from nuzline.prep.setups import (
    Setup,
    dedupe,
    neighbours,
    protected_slots,
    setup_cost,
    setup_specs,
    setup_states,
    team_setups,
)
from nuzline.prep.variants import (
    RELEARNER,
    Member,
    base_member,
    designated_change,
    max_hp_of,
    member_spec,
)
from nuzline.search.rows import row_bytes
from tests.engine.helpers import roark_party, roark_trainer, sample_party
from tests.prep.fixtures import candidate, problem


def _pool(names, n, **kw):
    party = sample_party(names)
    return [candidate(names, party[i % len(party)], i, **kw) for i in range(n)]


def _by_candidate(setup):
    return sorted(setup.members, key=lambda m: m.candidate)


def test_team_setups_count(tables, names):
    for n, cap, want in ((3, 2, 9), (4, 6, 32), (11, 6, 7018)):
        prob = problem(tables, names, _pool(names, n))
        setups = team_setups(prob, cap)
        assert len(setups) == want == sum(comb(n, k) * k for k in range(1, min(cap, n) + 1))
        assert len(set(setups)) == want
        teams = {}
        for s in setups:
            ids = [m.candidate for m in s.members]
            assert ids[1:] == sorted(ids[1:]) and len(set(ids)) == len(ids)
            assert all(m == base_member(prob, m.candidate) for m in s.members)
            teams.setdefault(frozenset(ids), []).append(ids[0])
        for team, leads in teams.items():
            assert sorted(leads) == sorted(team)


def test_neighbours_one_change(tables, names):
    party = sample_party(names)
    bite = names.move("Bite")
    tm10, hidden = names.item("TM10"), names.move("Hidden Power")
    tm11, sunny = names.item("TM11"), names.move("Sunny Day")
    cands = [
        candidate(names, party[i], i, relearner=(bite,), tms=((tm10, hidden), (tm11, sunny)))
        for i in range(3)
    ]
    res = Resources(1, ((tm10, 1),), (), ())
    prob = problem(tables, names, cands, resources=res, levelled=[()] * 3)
    start = next(s for s in team_setups(prob) if len(s.members) == 3)
    near = neighbours(tables, prob, start)
    assert near and len(set(near)) == len(near) and start not in near
    base = _by_candidate(start)
    for s in near:
        assert setup_cost(prob, s).fits(res)  # no TM11 (not in the bag), one scale at most
        if [m.candidate for m in s.members] != [m.candidate for m in start.members]:
            assert _by_candidate(s) == base  # a new lead only
            continue
        diff = [(a, b) for a, b in zip(start.members, s.members, strict=True) if a != b]
        assert len(diff) == 1
        a, b = diff[0]
        fields = ("moves", "item", "hp", "status")
        assert sum(getattr(a, f) != getattr(b, f) for f in fields) == 1
    assert {s.members[0].candidate for s in near} >= {1, 2}

    # after a relearner change no neighbour adds another (one Heart Scale), and a changed
    # slot is never changed again
    changed = next(
        s for s in near if any(c.source == RELEARNER for m in s.members for c in m.moves)
    )
    again = neighbours(tables, prob, changed)
    assert again
    for s in again:
        assert sum(c.source == RELEARNER for m in s.members for c in m.moves) == 1
        for old, new in zip(_by_candidate(changed), _by_candidate(s), strict=True):
            assert new.moves[: len(old.moves)] == old.moves
            added = new.moves[len(old.moves) :]
            assert not {c.slot for c in added} & {c.slot for c in old.moves}

    pruned = [m for s in near for m in s.members if m not in start.members]
    kept = neighbours(tables, prob, start, pruned=pruned)
    assert len(kept) < len(near)
    assert all(m not in pruned for s in kept for m in s.members)


def test_neighbours_designated_relearner(tables, names):
    party = sample_party(names)
    moves = tuple(names.move(m) for m in ("Bite", "Ember", "Leer", "Surf", "Thunderbolt"))
    cands = [candidate(names, party[i], i, relearner=moves) for i in range(2)]
    res = Resources(1, (), (), ())
    prob = problem(tables, names, cands, resources=res, levelled=[()] * 2)
    start = next(s for s in team_setups(prob) if len(s.members) == 2)
    kw = {"tms": False, "items": False, "entry_hp": False}

    def relearns(relearner):
        out = []
        for s in neighbours(tables, prob, start, relearner=relearner, **kw):
            for m in s.members:
                out += [(m.candidate, c) for c in m.moves if c.source == RELEARNER]
        return out

    designated = relearns("designated")
    for cand_i, change in designated:
        assert change == designated_change(tables, prob, cand_i)
    assert len(designated) <= len(start.members)
    assert len(relearns("all")) > len(designated)
    assert relearns("none") == []
    chained = next(
        s
        for s in neighbours(tables, prob, start, relearner="designated", **kw)
        if any(m.moves for m in s.members)
    )
    for s in neighbours(tables, prob, chained, relearner="designated", **kw):
        assert sum(c.source == RELEARNER for m in s.members for c in m.moves) <= 1


def test_setup_states_and_protected(tables, names):
    prob = problem(tables, names, _pool(names, 3), protected=[2], deaths=2)
    setup = Setup(tuple(base_member(prob, i) for i in (0, 1, 2)))
    other = Setup(tuple(base_member(prob, i) for i in (2, 0, 1)))
    assert protected_slots(prob, setup) == (2,)
    states = setup_states(tables, prob, [setup, other])
    assert np.asarray(states.protected)[0].tolist() == [0, 0, 1, 0, 0, 0]
    assert np.asarray(states.protected)[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert np.asarray(states.faint_budget).tolist() == [2, 2]
    rows = row_bytes(states, 2)
    for i, s in enumerate((setup, other)):
        ref = make_state(
            tables, setup_specs(prob, s), roark_party(names), 0, roark_trainer().items
        ).with_budget(2, protected_slots(prob, s))
        ref_row = row_bytes(type(ref)(*(np.asarray(f)[None] for f in ref)), 1)[0]
        assert rows[i].tobytes() == ref_row.tobytes()


def test_dedupe_keeps_the_cheapest(tables, names):
    prob = problem(tables, names, _pool(names, 2))
    plain = Setup(tuple(base_member(prob, i) for i in (0, 1)))
    top = max_hp_of(tables, member_spec(prob, plain.members[0], tables))
    full = Setup((replace(plain.members[0], hp=top), plain.members[1]))
    low = Setup((replace(plain.members[0], hp=top - 1), plain.members[1]))
    setups = [plain, full, low]
    states = setup_states(tables, prob, setups)
    assert dedupe(states, setups, [2.0, 1.0, 5.0]) == [1, 2]
    assert dedupe(states, setups, [1.0, 1.0, 5.0]) == [0, 2]
    assert isinstance(plain.members[0], Member)
