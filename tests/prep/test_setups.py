"""Team setups, neighbours and batched setup states (plan 4c Task 5)."""

from dataclasses import replace
from math import comb

import numpy as np
import pytest

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
    LEVEL_UP,
    RELEARNER,
    Member,
    base_member,
    designated_change,
    max_hp_of,
    member_spec,
    single_changes,
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


@pytest.mark.slow
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
            tables,
            setup_specs(prob, s),
            roark_party(names),
            0,
            roark_trainer().items,
            terrain=roark_trainer().terrain,
        ).with_budget(2, protected_slots(prob, s))
        ref_row = row_bytes(type(ref)(*(np.asarray(f)[None] for f in ref)), 1)[0]
        assert rows[i].tobytes() == ref_row.tobytes()


def test_field_defog_keeps_a_defog_holder_in_the_team(tables, names):
    """With ``field_move`` (G-15) every team setup holds a member that knows Defog after its
    changes: the party that clears the fog with field Defog fights (field_map_change_flags.c:
    80-85: a warp onto the map clears the flag, so no PC between). A neighbour that overwrites
    the move is dropped; the start mask follows the start weather (bit 15, FCM.md 4.3)."""
    from nuzline.engine.ids import Weather
    from nuzline.prep.setups import knows_field_move
    from nuzline.prep.variants import MoveChange

    defog = names.move("Defog")
    party = sample_party(names, override_moves={1: ["Defog", "Tackle"]})
    pool = [candidate(names, party[i], i) for i in range(3)]
    prob = replace(problem(tables, names, pool), field_move=defog)
    setups = team_setups(prob, 6)
    assert len(setups) == 1 + 2 * 2 + 3  # teams holding candidate 1, each lead
    assert all(any(m.candidate == 1 for m in s.members) for s in setups)
    lone = Setup((base_member(prob, 1),))
    overwritten = Setup((replace(base_member(prob, 1), moves=(MoveChange(0, 33, LEVEL_UP),)),))
    assert knows_field_move(prob, lone) and not knows_field_move(prob, overwritten)
    masks = [
        int(
            np.asarray(setup_states(tables, replace(prob, start_condition=w), [lone]).field_mask)[0]
        )
        for w in (Weather.FOG, Weather.NONE)
    ]
    assert masks == [0x8000, 0]


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


def test_levelled_knob_drops_free_replacements(tables, names):
    """A levelled move that meets a full set is a free LEVEL_UP replacement only with
    ``levelled=True``; ``base_member`` is the same either way."""
    bite = names.move("Bug Bite")
    spec = replace(sample_party(names)[0], level=16)
    cand = candidate(names, spec, 0)
    prob = problem(tables, names, [cand], levelled=[((16, bite),)])
    assert base_member(prob, 0) == Member(0)
    setup = Setup((base_member(prob, 0),))

    def level_ups(members):
        return [c for m in members for c in m.moves if c.source == LEVEL_UP]

    kw = {"relearner": "none", "tms": False, "items": False, "entry_hp": False}
    assert level_ups(single_changes(tables, prob, 0, **kw)) != []
    assert level_ups(single_changes(tables, prob, 0, levelled=False, **kw)) == []
    assert level_ups(m for s in neighbours(tables, prob, setup, **kw) for m in s.members) != []
    off = neighbours(tables, prob, setup, levelled=False, **kw)
    assert level_ups(m for s in off for m in s.members) == []


def test_team_count_and_the_team_space_limit():
    """The team phase registers every (team, lead) at once: its size is counted before
    any work and refused above the limit, naming --exclude and --max-team."""
    import pytest

    from nuzline.prep.setups import MAX_TEAM_SETUPS, check_team_space, team_count

    assert team_count(4) == 32
    assert team_count(11) == 7018
    assert team_count(20) == 333_280
    assert team_count(20, 2) == 20 + 2 * 190
    check_team_space(11, 6)
    check_team_space(13, 6)  # 20,618
    with pytest.raises(ValueError, match="--exclude.*--max-team") as e:
        check_team_space(20, 6)
    assert "333,280" in str(e.value) and f"{MAX_TEAM_SETUPS:,}" in str(e.value)
    check_team_space(20, 6, limit=400_000)


def test_run_terrain_is_a_prep_step_and_in_the_checkpoint(tables, names):
    """A fight whose start tiles give several battle terrains runs on the one the run names
    (G-18): the plan says to start on such a tile (CalcTerrain reads the tile under the player,
    field_battle_data_transfer.c:467-494), and a checkpoint of another terrain does not resume
    it. A fight with no named terrain keeps its digest."""
    from nuzline.data.trainer import TERRAIN_UNKNOWN
    from nuzline.prep.checkpoint import fingerprint
    from nuzline.prep.report import prep_steps
    from nuzline.prep.search import PrepSettings

    pool = [candidate(names, sample_party(names)[0], 0)]
    base = problem(tables, names, pool)
    mixed = replace(base.trainer, terrain=TERRAIN_UNKNOWN, terrains=(2, 6))
    unnamed = replace(base, trainer=mixed)
    on = {t: replace(base, trainer=replace(mixed, terrain=t)) for t in (2, 6)}
    setup = Setup((base_member(base, 0),))
    assert prep_steps(on[6], names, setup, tables)[-1] == (
        "Start the fight standing on a tile whose battle terrain is TERRAIN_SNOW (the tile "
        "under the player decides it)."
    )
    assert not any("battle terrain" in s for s in prep_steps(base, names, setup, tables))
    # G-24: a save in the Route 213 script rain plans clear behind a warp onto the map.
    warp = prep_steps(replace(base, script_weather_warp=True), names, setup, tables)
    assert warp[-1].startswith("Warp onto the fight's map before the fight")
    assert not any(s.startswith("Warp onto") for s in prep_steps(base, names, setup, tables))
    digest = {
        k: fingerprint(p, PrepSettings(), 0, "t", "c")["problem"]
        for k, p in (("base", base), ("unnamed", unnamed), *on.items())
    }
    assert digest["unnamed"] == digest["base"]
    assert len({digest["base"], digest[2], digest[6]}) == 3
