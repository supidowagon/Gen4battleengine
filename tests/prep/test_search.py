"""The prep funnel: halving, expansions, finalists and the anytime budget (plan 4c Task 7).

Most tests patch ``screen`` (and ``solve``) with fakes that read a setup's identity back from
its turn-0 state bytes, so the funnel's bookkeeping is tested without playout noise; the
reproducibility test runs the real engine.
"""

import itertools

import jax
import pytest

from nuzline.prep import search
from nuzline.prep.problem import Resources
from nuzline.prep.screen import ScreenStats, prep_order, scored
from nuzline.prep.search import PrepSettings, phase_id, run_prep
from nuzline.prep.setups import neighbours, protected_slots, team_setups
from nuzline.search import deaths
from nuzline.search.rows import row_bytes
from nuzline.search.stage_b import ActionValue, PlanNode, SolveResult
from tests.engine.helpers import sample_party
from tests.prep.fixtures import candidate, problem

KEY = jax.random.PRNGKey(7)

SETTINGS = PrepSettings(
    screen_n=4,
    team_keep=2,
    variant_rounds=1,
    variant_keep=2,
    finalists=2,
    max_depth=1,
    finalist_playouts=200,
    max_team=2,
)


def make_problem(tables, names, **kw):
    party = sample_party(names)
    bite = names.move("Bite")
    cands = [candidate(names, party[i], i, relearner=(bite,)) for i in range(3)]
    return problem(
        tables, names, cands, resources=Resources(1, (), (), ()), levelled=[()] * 3, **kw
    )


class Fakes:
    """Patches ``setup_states`` (to learn which state row is which setup) and ``screen``.
    ``cum(setup, index)`` gives the fake cumulative P(win, deaths <= j) for j = 0..3 of the
    setup registered ``index``-th; ``refuse(setup)`` marks a setup's playouts refused."""

    def __init__(self, monkeypatch, cum, refuse=lambda setup: False):
        self.cum, self.refuse = cum, refuse
        self.of_bytes: dict[bytes, tuple] = {}
        self.calls: list[tuple[list, int, object]] = []  # (setups, n, states)
        self.total: dict = {}
        self.order: list = []
        real = search.setup_states

        def states_of(tables, prob, setups):
            states = real(tables, prob, setups)
            for i, blob in enumerate(row_bytes(states, len(setups))):
                key = blob.tobytes()
                if key not in self.of_bytes:
                    self.of_bytes[key] = setups[i]
                    self.order.append(setups[i])
            return states

        monkeypatch.setattr(search, "setup_states", states_of)
        monkeypatch.setattr(search, "screen", self.screen)

    def setup_of(self, states):
        return [self.of_bytes[r.tobytes()] for r in row_bytes(states, len(states.hp))]

    def screen(self, tables, states, n, key, max_turns=100):
        setups = self.setup_of(states)
        self.calls.append((setups, n, states))
        out = []
        for s in setups:
            self.total[s] = self.total.get(s, 0) + n
            cum = self.cum(s, self.order.index(s))
            counts, prev = [], 0.0
            for c in cum:
                counts.append(round(n * (c - prev)))
                prev = c
            counts += [0, 0, n - sum(counts), 0]
            out.append(ScreenStats(tuple(counts), 0, 0.0, 0.0, self.refuse(s)))
        return out


def falling(setup, index):
    return (1 - 0.02 * index,) * 4


def test_team_phase_halving(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    settings = PrepSettings(
        screen_n=4, team_keep=2, ladder_keep=1, variant_rounds=0, max_depth=0, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    assert [(len(s), n) for s, n, _ in fakes.calls] == [(9, 4), (5, 8), (3, 16)]
    assert result.stats.team_rounds == 3
    assert result.stats.phase == "teams"
    assert result.plan is None and result.stats.stopped is None
    survivors = fakes.calls[-1][0]
    assert all(fakes.total[s] == 28 for s in survivors)  # T_2 = 4 * (2**3 - 1)
    assert result.screen.n == 28
    assert [r.k for r in result.ladder] == [0, 1, 2, 3]


def test_variant_round_screens_every_neighbour(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    settings = PrepSettings(
        screen_n=4, team_keep=2, variant_keep=2, variant_rounds=1, max_depth=0, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    teams = team_setups(prob, 2)
    parents = teams[:2]  # the fake ranks by registration order
    team_calls = 3
    known = set(teams)
    dropped = {d for d, _ in result.pruned}
    expected = []
    for p in parents:
        for s in neighbours(tables, prob, p, pruned=dropped):
            if s not in known and s not in expected:
                expected.append(s)
    assert expected
    first_setups, first_n, _ = fakes.calls[team_calls]
    assert first_n == 4  # T_0
    assert len(first_setups) == len(set(first_setups)) == len(expected)
    assert set(first_setups) == set(expected)  # no parent: each already has n >= T_0
    assert not set(first_setups) & set(parents)
    # round 1 tops everything up to T_1 = 12 and a parent (n = 28) gets nothing
    second_setups, second_n, _ = fakes.calls[team_calls + 1]
    assert second_n == 8
    assert not set(second_setups) & set(parents)
    assert all(fakes.total[s] >= 12 for s in second_setups)
    assert all(fakes.total[p] >= 28 for p in parents)
    assert result.stats.variant_rounds == 1
    assert result.stats.variant_halvings == len(fakes.calls) - team_calls >= 2
    assert result.stats.phase == "variants"


def test_parent_can_win_a_variant_round(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    teams = team_setups(prob, 2)
    best = teams[0]
    Fakes(monkeypatch, lambda s, i: ((0.9,) if s == best else (0.3,)) * 4)
    settings = PrepSettings(
        screen_n=4, team_keep=2, variant_keep=1, variant_rounds=1, max_depth=0, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    assert result.stats.variant_rounds == 1
    assert result.chosen == best
    assert result.ladder[0].setup == best


def test_ladder_keeps_other_rungs_best(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    x = team_setups(prob, 2)[8]

    def cum(setup, index):
        if setup == x:
            return (0.1, 1.0, 1.0, 1.0)
        return (0.9 - 0.02 * index,) * 4

    fakes = Fakes(monkeypatch, cum)
    settings = PrepSettings(
        screen_n=4, team_keep=2, ladder_keep=1, variant_rounds=0, max_depth=0, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    rounds = [s for s, _, _ in fakes.calls]
    assert len(rounds) == 3 and all(x in s for s in rounds)  # screened every team round
    assert result.chosen != x
    assert result.ladder[0].setup != x
    assert result.ladder[1].setup == x
    assert result.ladder[1].source == "screen"
    assert result.ladder[0].source == "screen"
    assert len(rounds[1]) < len(rounds[0])


def test_phase_keys_are_disjoint():
    ids = {
        phase_id("teams"),
        phase_id("deep"),
        phase_id("finalists"),
        *(phase_id("variants", v) for v in range(64)),
    }
    assert len(ids) == 67
    with pytest.raises(ValueError):
        PrepSettings(variant_rounds=-1)
    with pytest.raises(ValueError):
        PrepSettings(ladder_keep=0)
    for bad in ({"screen_n": 7}, {"screen_n": 0}, {"team_keep": 0}, {"finalists": 0},
                {"max_depth": -1}, {"variant_keep": 0}):  # fmt: skip
        with pytest.raises(ValueError):
            PrepSettings(**bad)


def fake_solve(fakes, record, values):
    """A ``solve`` returning ``values(setup, call_index)`` as the leading action's estimate."""

    def solve(tables, names, state, key, settings):
        setup = fakes.of_bytes[row_bytes(jax.tree.map(lambda f: f[None], state), 1)[0].tobytes()]
        record.append((setup, settings.max_depth))
        e = values(setup, len(record) - 1)
        action = ActionValue(
            0, "a", e, e - 0.01, e + 0.01, e, 0.0, 0.0, 0.0, False, 0, 0.0, 0.0,
            deaths.DeathDistribution.from_counts([e, 1 - e, 0, 0, 0, 0, 0, 0]),
        )  # fmt: skip
        plan = PlanNode(None, 1.0, 0, {}, {}, "decision")
        return SolveResult([action], {}, settings.max_depth, plan, 10, 0.0, settings)

    return solve


@pytest.mark.parametrize("max_depth", [1, 2])
def test_finalists_then_deepening(tables, names, monkeypatch, max_depth):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    record: list = []
    # the second finalist solved gets the better value: the leader is not the screen's top
    monkeypatch.setattr(search, "solve", fake_solve(fakes, record, lambda s, i: 0.4 + 0.2 * i))
    settings = PrepSettings(
        screen_n=4, team_keep=2, variant_rounds=0, finalists=2, max_depth=max_depth, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    assert [d for _, d in record] == ([1, 1] if max_depth == 1 else [1, 1, 2])
    assert len(result.finalists) == 2
    leader = record[1][0]
    assert result.chosen == leader == result.finalists[0].setup
    if max_depth == 2:
        assert record[2][0] == leader
        assert result.plan.depth == 2
    else:
        assert result.plan.depth == 1
    assert result.plan is result.finalists[0].result
    assert result.ladder[0].source == "plan"
    assert [r.source for r in result.ladder[1:]] == ["screen"] * 3
    assert result.stats.phase == "finalists"


def raising_solve(fakes, raised, bad, inner):
    """``solve`` raising HiddenLegality for the setups ``bad(setup, depth)`` selects (they are
    appended to ``raised``)."""
    ok = fake_solve(fakes, [], inner)

    def solve(tables, names, state, key, settings):
        setup = fakes.of_bytes[row_bytes(jax.tree.map(lambda f: f[None], state), 1)[0].tobytes()]
        if bad(setup, settings.max_depth):
            raised.append(setup)
            raise search.HiddenLegality("hidden")
        return ok(tables, names, state, key, settings)

    return solve


def _assert_excluded_everywhere(result, gone):
    assert len(result.excluded) == len(gone)
    assert result.chosen not in gone
    assert all(r.setup not in gone for r in result.ladder)
    assert all(f.setup not in gone for f in result.finalists)


def _exclusion_settings(max_depth=1):
    return PrepSettings(
        screen_n=4, team_keep=2, variant_rounds=0, finalists=2, max_depth=max_depth, max_team=2
    )


def test_stage_b_exclusion_of_one_finalist(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    monkeypatch.setattr(search, "solve", fake_solve(fakes, [], lambda s, i: 0.5))
    probe = run_prep(tables, names, prob, _exclusion_settings(), KEY)
    first = probe.chosen
    raised: list = []
    monkeypatch.setattr(
        search, "solve", raising_solve(fakes, raised, lambda s, d: s == first, lambda s, i: 0.5)
    )
    result = run_prep(tables, names, prob, _exclusion_settings(), KEY)
    assert raised == [first]
    _assert_excluded_everywhere(result, raised)
    assert len(result.finalists) == 1


def test_stage_b_exclusion_of_every_finalist(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    monkeypatch.setattr(search, "solve", fake_solve(fakes, [], lambda s, i: 0.5))
    probe = run_prep(tables, names, prob, _exclusion_settings(), KEY)
    gone = [f.setup for f in probe.finalists]
    raised: list = []
    monkeypatch.setattr(
        search, "solve", raising_solve(fakes, raised, lambda s, d: s in gone, lambda s, i: 0.5)
    )
    result = run_prep(tables, names, prob, _exclusion_settings(), KEY)
    assert sorted(map(str, raised)) == sorted(map(str, gone))
    assert result.finalists == ()
    assert result.plan is None
    _assert_excluded_everywhere(result, gone)


def test_deep_failure_drops_the_leader(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    raised: list = []
    monkeypatch.setattr(
        search, "solve",
        raising_solve(fakes, raised, lambda s, d: d == 2, lambda s, i: 0.4 + 0.2 * i),
    )  # fmt: skip
    result = run_prep(tables, names, prob, _exclusion_settings(2), KEY)
    assert len(raised) == 1
    _assert_excluded_everywhere(result, raised)
    assert len(result.finalists) == 1
    assert result.chosen == result.finalists[0].setup
    assert result.plan.depth == 1


def test_budget_stops_after_first_round(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    fakes = Fakes(monkeypatch, falling)
    tick = itertools.count(0, 10)
    settings = PrepSettings(screen_n=4, team_keep=2, budget_seconds=1, max_team=2)
    result = run_prep(tables, names, prob, settings, KEY, clock=lambda: float(next(tick)))
    assert len(fakes.calls) == 1
    assert result.stats.stopped == "budget"
    assert result.plan is None and not result.finalists
    setups = fakes.calls[0][0]
    rows = [scored(_stats(fakes, s), 0, 0.0) for s in setups]
    assert result.chosen == setups[prep_order(rows)[0]]
    assert all(r.source == "screen" for r in result.ladder)


def _stats(fakes, setup):
    n = fakes.total[setup]
    cum = fakes.cum(setup, fakes.order.index(setup))
    counts, prev = [], 0.0
    for c in cum:
        counts.append(round(n * (c - prev)))
        prev = c
    return ScreenStats((*counts, 0, 0, n - sum(counts), 0), 0, 0.0, 0.0, False)


def test_interrupt_returns_best_so_far(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    Fakes(monkeypatch, falling)
    seen = []

    def progress(p):
        seen.append(p)
        if len(seen) == 2:
            raise KeyboardInterrupt

    settings = PrepSettings(screen_n=4, team_keep=2, variant_rounds=1, max_team=2)
    result = run_prep(tables, names, prob, settings, KEY, progress=progress)
    assert result.stats.stopped == "interrupted"
    assert result.stats.team_rounds == 2
    assert result.stats.variant_rounds == 0
    assert result.plan is None
    assert [p.phase for p in seen] == ["teams", "teams"]
    assert [p.round for p in seen] == [0, 1]
    assert [p.alive for p in seen] == [5, 3]
    assert seen[-1].leader == result.chosen


def test_refused_setups_are_left_out(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    teams = team_setups(prob, 2)
    bad = teams[0]  # the fake's best
    Fakes(monkeypatch, falling, refuse=lambda s: s == bad)
    settings = PrepSettings(
        screen_n=4, team_keep=2, variant_rounds=0, max_depth=0, ladder_keep=1, max_team=2
    )
    result = run_prep(tables, names, prob, settings, KEY)
    assert result.chosen != bad
    assert all(r.setup != bad for r in result.ladder)
    assert len(result.excluded) == 1
    assert prob.candidates[bad.members[0].candidate].label in result.excluded[0]


def test_all_refused_raises(tables, names, monkeypatch):
    prob = make_problem(tables, names)
    Fakes(monkeypatch, falling, refuse=lambda s: True)
    with pytest.raises(ValueError, match="every setup refused"):
        run_prep(tables, names, prob, PrepSettings(screen_n=4, max_team=2), KEY)


def test_states_carry_budget_and_protection(tables, names, monkeypatch):
    prob = make_problem(tables, names, deaths=1, protected=(0,))
    fakes = Fakes(monkeypatch, falling)
    settings = PrepSettings(screen_n=4, team_keep=2, variant_rounds=1, max_depth=0, max_team=2)
    run_prep(tables, names, prob, settings, KEY)
    assert len(fakes.calls) > 3  # the expansion's states too
    for setups, _, states in fakes.calls:
        assert (states.faint_budget == 1).all()
        for i, setup in enumerate(setups):
            slots = protected_slots(prob, setup)
            assert [s for s in range(6) if states.protected[i, s]] == list(slots)
            assert len(slots) == sum(m.candidate == 0 for m in setup.members)
    assert any(protected_slots(prob, s) for ss, _, _ in fakes.calls for s in ss)


def test_reproducible(tables, names):
    prob = make_problem(tables, names)
    a = run_prep(tables, names, prob, SETTINGS, KEY)
    b = run_prep(tables, names, prob, SETTINGS, KEY)
    assert a == b
    assert a.plan is not None and a.finalists
    assert a.stats.phase == "finalists" and a.stats.stopped is None
    assert abs(sum(a.distribution.probs) - 1) < 1e-9
