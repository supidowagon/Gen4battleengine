"""The prep funnel: halving, expansions, finalists and the anytime budget (plan 4c Task 7).

Most tests patch ``screen`` (and ``solve``) with fakes that read a setup's identity back from
its turn-0 state bytes, so the funnel's bookkeeping is tested without playout noise; the
reproducibility test runs the real engine.
"""

import itertools
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from nuzline.prep import search
from nuzline.prep.problem import Resources
from nuzline.prep.screen import ScreenStats, prep_order, scored
from nuzline.prep.search import PrepSettings, phase_id, run_prep
from nuzline.prep.setups import neighbours, protected_slots, setup_cost, team_setups
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
                {"max_depth": -1}, {"variant_keep": 0}, {"tie_playouts": -1}):  # fmt: skip
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
    reg = search._Funnel(tables, names, prob, settings, KEY, None, float)  # same cost, tie rank
    reg.register(setups)
    rows = [
        scored(_stats(fakes, s), 0, reg.cost[reg.index[s]], reg.tie[reg.index[s]]) for s in setups
    ]
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


def _saturated(wins0: int, n: int, hp: float) -> ScreenStats:
    """``wins0`` deathless wins of ``n``, the rest one-death wins; mean HP left ``hp``."""
    return ScreenStats((wins0, n - wins0, 0, 0, 0, 0, 0, 0), n - wins0, hp * n, 0.5 * n, False)


def test_halving_key_ignores_hp_pp_and_index():
    """Whole-party HP left favours big teams (bench members never lose HP); in the rung-0 gate
    it dropped the two best setups (#933, #92 at 248/248). Only score, deaths, cost and the
    seeded tie rank order a cut."""
    hi, lo = _saturated(6, 6, 0.9), _saturated(6, 6, 0.3)
    assert search.halving_key(hi, 0, 0.0, 5) == search.halving_key(lo, 0, 0.0, 5)
    assert search.halving_key(_saturated(761, 762, 0.9), 0, 0.0, 0) > search.halving_key(
        lo, 0, 0.0, 1
    )  # the point estimate first, whatever the counts
    assert search.halving_key(lo, 0, 0.0, 9) < search.halving_key(lo, 0, 3.0, 0)  # cost, then tie
    assert search.halving_key(lo, 0, 0.0, 1) < search.halving_key(lo, 0, 0.0, 2)


def test_tie_rank_is_seeded_and_order_free():
    assert search.tie_rank(b"seed", b"row") == search.tie_rank(b"seed", b"row")
    assert search.tie_rank(b"seed", b"row") != search.tie_rank(b"seed2", b"row")
    assert 0 <= search.tie_rank(b"", b"") < 2**64


class Expansion:
    """A bare ``_Funnel`` holding ``parents`` (index ``0..``, with the given stats) and one
    variant expansion's new neighbours (the following indices, 0 playouts). The fake
    ``screen`` plays a neighbour's next playouts from ``losses(i, m)``, its cumulative loss
    count after ``m`` playouts; ``hp[i]`` is its mean HP left. A turn-0 state row is the setup
    index itself."""

    def __init__(
        self, monkeypatch, parents, n_new, losses, hp, keep=8, cost=None, tie=None,
        tie_playouts=1016,
    ):  # fmt: skip
        settings = PrepSettings(
            screen_n=8, variant_keep=keep, ladder_keep=4, tie_playouts=tie_playouts
        )  # fmt: skip
        funnel = search._Funnel(None, None, SimpleNamespace(deaths=0), settings, KEY, None, float)
        total = len(parents) + n_new
        funnel.setups = [f"setup {i}" for i in range(total)]
        funnel.rows = [np.asarray(i) for i in range(total)]
        funnel.cost = list(cost) if cost is not None else [0.0] * total
        funnel.tie = list(tie) if tie is not None else list(range(total))
        empty = ScreenStats((0,) * len(deaths.BUCKETS), 0, 0.0, 0.0, False)
        funnel.stats = {i: (parents[i] if i < len(parents) else empty) for i in range(total)}
        self.parents = list(range(len(parents)))
        self.new = list(range(len(parents), total))
        funnel.register = lambda setups: list(self.new)
        self.funnel = funnel

        def screen(tables, states, n, key, max_turns=100):
            out = []
            for i in np.asarray(states).tolist():
                m = funnel.stats[i].n
                lost = losses(i, m + n) - losses(i, m)
                counts = (n - lost, 0, 0, 0, 0, 0, lost, 0)
                out.append(ScreenStats(counts, 0, hp[i] * n, 0.5 * n, False))
            return out

        monkeypatch.setattr(search, "screen", screen)
        monkeypatch.setattr(search, "neighbours", lambda *args, **kwargs: [])

    def run(self) -> list[int]:
        """The expansion's survivors on rung 0."""
        return self.funnel.expansion(0, [list(self.parents)] * search.RUNGS)[0]


@pytest.mark.parametrize("where", ["first", "last"])
def test_expansion_keeps_a_better_neighbour_against_tight_parents(monkeypatch, where):
    """Finding N1: 8 parents at 0.997 over 4088 playouts (a second expansion), 60 neighbours
    truly 0.99 and one truly 1.000. The parents' tight lower bound beat the 1.000 neighbour's
    ``n/n`` at every cut, wherever it was registered; it must survive. Every other neighbour
    loses one of its first 8, so the 1.000 neighbour leads from round 0 and is compared with the
    parents at their 4088 playouts (Task 5)."""
    parents = [_saturated(4076, 4088, 0.6)] * 8
    n_new = 61
    best = 8 if where == "first" else 8 + n_new - 1
    hp = {i: (0.95 if i == best else 0.3 + 0.005 * (i % 60)) for i in range(8 + n_new)}

    def losses(i, m):
        return 0 if i == best else (0 if m == 0 else 1 + m // 100)  # about 0.99

    out = Expansion(monkeypatch, parents, n_new, losses, hp).run()
    assert best in out


def test_expansion_keeps_a_parent_against_lucky_neighbours(monkeypatch):
    """The opposite bias: 8 parents at 0.997 over 762 playouts, 1000 neighbours truly 0.99
    that win their first 504 playouts (``n/n`` through round 5) and lose 1 in 50 after. The
    parents must survive: the neighbours' final estimate, over 1016, is below theirs."""
    parents = [_saturated(760, 762, 0.6)] * 8
    n_new = 1000
    hp = {i: 0.3 + 0.0005 * (i % 1000) for i in range(8 + n_new)}

    def losses(i, m):
        return 0 if m <= 504 else -(-(m - 504) // 50)

    out = Expansion(monkeypatch, parents, n_new, losses, hp).run()
    assert set(out) == set(range(8))


def _bare(stats, tie=None):
    funnel = search._Funnel(None, None, SimpleNamespace(deaths=0), PrepSettings(), KEY, None, float)
    funnel.stats = dict(enumerate(stats))
    funnel.cost = [0.0] * len(stats)
    funnel.tie = list(tie) if tie is not None else list(range(len(stats)))
    return funnel


def test_top_keeps_the_tied_block():
    stats = [_saturated(10, 10, 0.5)] * 6 + [_saturated(9, 10, 0.5)] * 4
    stats += [_saturated(8, 10, 0.5)] * 2
    f = _bare(stats)
    assert f.top(range(12), 0, 8, ties=True) == list(range(10))  # ranks 7-10 tie at the cut
    assert f.top(range(12), 0, 8) == list(range(8))
    assert f.top(range(12), 0, 6, ties=True) == list(range(6))  # no tie across the boundary
    assert f.top(range(12), 0, 12, ties=True) == list(range(12))


def test_floor_cut_screens_an_exact_tie_further(monkeypatch):
    """12 setups, keep 8: the best (last by tie rank) never loses; the other 11 lose exactly one
    playout after 300. A cut at n/n used to go by HP, then index, and dropped it."""
    best = 11

    def losses(i, m):
        return 0 if i == best else int(m > 300)

    hp = {i: (0.1 if i == best else 0.9) for i in range(12)}
    pid = phase_id("teams")
    e = Expansion(monkeypatch, [], 12, losses, hp, keep=8)
    out = e.funnel.halving("teams", "teams", pid, [list(e.new)] * search.RUNGS, 8)[0]
    assert best in out and len(out) == 8
    assert max(e.funnel.stats[i].n for i in out) == 1016  # bounded by tie_playouts
    assert e.funnel.playouts == 12 * 1016
    off = Expansion(monkeypatch, [], 12, losses, hp, keep=8, tie_playouts=0)
    out = off.funnel.halving("teams", "teams", pid, [list(off.new)] * search.RUNGS, 8)[0]
    assert best not in out and len(out) == 8  # retention off: the old saturated cut


def test_expansion_compares_parents_and_neighbours_at_equal_n(monkeypatch):
    """The rung-0 gate miss, expansion 1's final cut (trace cut 9): 32 parents at 248, two of
    them truly ~0.999 with little HP left (#933, #92), one lucky (#3357, truly ~0.989, the most
    HP); 105 cost-3 neighbours stop halving at 120/120 with more HP. The one-step cut at
    unequal n kept the neighbours and the lucky parent and dropped both safe parents."""
    safe, lucky = (0, 1), 2
    parents = [
        _saturated(248, 248, 0.944),
        _saturated(248, 248, 0.886),
        _saturated(248, 248, 0.9665),
    ] + [_saturated(246, 248, 0.5)] * 29
    n_new = 105
    total = 32 + n_new
    hp = {i: (0.963 + 0.007 * (i % 10) / 10 if i >= 32 else 0.5) for i in range(total)}
    cost = [0.0] * 32 + [3.0] * n_new

    def losses(i, m):
        if i in safe:
            return 0
        if i == lucky:
            return max(0, m - 248) // 90
        if i < 32:
            return max(0, m - 248) // 120
        return max(0, m - 120) // 100

    e = Expansion(monkeypatch, parents, n_new, losses, hp, keep=8, cost=cost)
    out = e.run()
    assert set(safe) <= set(out) and len(out) == 8
    assert {e.funnel.stats[i].n for i in out} == {1016}  # one count for the whole cut

    def key(i):
        return search.halving_key(e.funnel.stats[i], 0, cost[i], e.funnel.tie[i])

    assert min(out, key=key) in safe


def test_ladder_round():
    got = [search.ladder_round(8, n) for n in (1, 24, 25, 248, 762, 1016, 4088)]
    assert got == [1, 1, 2, 4, 6, 6, 8]


@pytest.mark.parametrize("scales", [0, 1])
def test_a_reserved_heart_scale_is_never_spent(tables, names, monkeypatch, scales):
    """Reserving the Heart Scale leaves 0 in the problem's bag (problem.resources_without);
    no setup the funnel registers may then use the Move Relearner."""
    party = sample_party(names)
    bite = names.move("Bite")
    cands = [candidate(names, party[i], i, relearner=(bite,)) for i in range(3)]
    prob = problem(tables, names, cands, resources=Resources(scales, (), (), ()), levelled=[()] * 3)
    fakes = Fakes(monkeypatch, falling)
    settings = PrepSettings(
        screen_n=4, team_keep=2, variant_keep=2, variant_rounds=1, max_depth=0, max_team=2
    )
    run_prep(tables, names, prob, settings, KEY)
    spent = [setup_cost(prob, s).heart_scales for s in fakes.order]
    assert fakes.order
    if scales == 0:
        assert not any(spent)
    else:
        assert any(spent)  # the neighbour space does offer the relearner when allowed
