"""The line search (spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md §5):
the candidate rules per foe, the race and its acceptance, the sweeps to a fixed point, the
best line seen and the kicks, the key discipline (the hold-out never shares keys with the
search), the time budget and the start line. Most tests replace ``evaluate_lines`` with a
recorder or a table of win rates, so only the two marked slow compile a playout program."""

import zlib

import jax
import numpy as np
import pytest

from nuzline.engine.ids import EffectId, Status
from nuzline.search import deaths, line_search
from nuzline.search.line import Line, empty_line
from nuzline.search.line_search import (
    HOLD_OUT_SALT,
    MAJOR_STATUSES,
    RESTART_SALT,
    LineSearchSettings,
    entry_candidates,
    find_line,
    joint_candidates,
    who_what_candidates,
)
from nuzline.search.player_eval import PlayerOdds, evaluate_lines
from tests.engine.helpers import roark_state

KEY = jax.random.PRNGKey(11)
SMALL = LineSearchSettings(
    race=((16, None), (32, 4), (64, 2)),
    accept_n=64,
    max_sweeps=2,
    restarts=0,
    final_n=128,
    max_turns=40,
)
FAST = LineSearchSettings(
    race=((16, None), (32, 4), (64, 2)), accept_n=96, max_sweeps=4, restarts=0, final_n=64
)


def _with_moves(state, slot: int, moves: list[int]):
    """``state`` with the player's party ``slot`` knowing exactly ``moves``."""
    m = np.asarray(state.moves).copy()
    m[0, slot] = moves + [0] * (4 - len(moves))
    return state._replace(moves=m)


def _living(state) -> set[int]:
    return {int(i) for i in np.flatnonzero((state.species[0] > 0) & (state.hp[0] > 0))}


def test_who_what_candidates(tables, names):
    s = roark_state(tables, names)
    power = np.asarray(tables.move_power)
    effect = np.asarray(tables.move_effect)
    param = np.asarray(tables.move_param)
    moves = np.asarray(s.moves)
    cands = who_what_candidates(tables, s, 0)
    assert cands
    firsts = set()
    for use, move, first in cands:
        assert use in _living(s)
        assert power[moves[0, use, move]] > 0
        if first != -1:
            m = moves[0, use, first]
            assert power[m] == 0
            assert effect[m] == EffectId.STATUS_FOE and param[m, 0] in MAJOR_STATUSES
            firsts.add((use, first))
    # Budew (slot 5) knows Stun Spore in move slot 3: a paralysis opener.
    assert (5, 3) in firsts

    # Openers the foe cannot receive are left out: Thunder Wave (type-checked) against
    # Geodude (Ground), and any status against a foe that already has one.
    tw = _with_moves(s, 0, [names.move("Scratch"), names.move("Thunder Wave")])
    tw_firsts = {(u, f) for u, _, f in who_what_candidates(tables, tw, 0) if f != -1}
    assert (0, 1) not in tw_firsts and (5, 3) in tw_firsts
    status = np.asarray(s.status).copy()
    status[1, 0] = Status.BURN
    burned = s._replace(status=status)
    assert all(f == -1 for _, _, f in who_what_candidates(tables, burned, 0))

    # A Ground move is never offered against a Flying-type foe.
    eq = names.move("Earthquake")
    flying = _with_moves(
        roark_state(tables, names, opp_species="Starly"), 0, [eq, names.move("Ember")]
    )
    offered = {(u, m) for u, m, _ in who_what_candidates(tables, flying, 0)}
    assert (0, 0) not in offered
    assert (0, 1) in offered

    # A Pokemon whose only damaging move cannot touch the foe yields no candidate.
    immune = _with_moves(flying, 0, [eq, names.move("Leer")])
    assert all(u != 0 for u, _, _ in who_what_candidates(tables, immune, 0))


def test_entry_candidates(tables, names):
    s = roark_state(tables, names)
    hp = np.asarray(s.hp).copy()
    hp[0, 2] = 0
    s = s._replace(hp=hp)
    assert entry_candidates(s, 1) == [(-1, 0), (-1, 1), (0, 0), (3, 0), (4, 0), (5, 0)]


def test_joint_candidates(tables, names):
    s = roark_state(tables, names, player_count=4)
    joint = joint_candidates(tables, s, 0)
    expected = [
        (use, move, first, via, wait)
        for use, move, first in who_what_candidates(tables, s, 0)
        for via, wait in entry_candidates(s, use)
    ]
    assert joint == expected
    assert len(set(joint)) == len(joint)


def _odds(n: int, tag: float = 0.0, wins: int = 0, refused: int = 0) -> PlayerOdds:
    return PlayerOdds(
        n=n,
        wins=(wins,) * 4,
        wins_any=wins,
        deaths_total=0,
        distribution=deaths.DeathDistribution.from_counts([1] + [0] * (len(deaths.BUCKETS) - 1)),
        mean_turns=tag,
        fallback_share=0.0,
        refused=refused,
    )


def test_merged_adds_counts():
    a = PlayerOdds(
        n=3,
        wins=(1, 2, 2, 2),
        wins_any=2,
        deaths_total=4,
        distribution=deaths.DeathDistribution.from_counts([1, 1, 0] + [0] * 6 + [1]),
        mean_turns=10.0,
        fallback_share=0.5,
        refused=1,
        refused_key=np.array([1, 2], np.uint32),
    )
    b = PlayerOdds(
        n=1,
        wins=(1, 1, 1, 1),
        wins_any=1,
        deaths_total=0,
        distribution=deaths.DeathDistribution.from_counts([1] + [0] * 9),
        mean_turns=20.0,
        fallback_share=0.0,
    )
    m = line_search.merged(a, b)
    assert (m.n, m.wins, m.wins_any, m.deaths_total, m.refused) == (4, (2, 3, 3, 3), 3, 4, 1)
    assert m.distribution.n == 4
    assert np.allclose(m.distribution.probs, [0.5, 0.25, 0] + [0] * 6 + [0.25])
    assert m.mean_turns == pytest.approx(12.5)
    assert m.fallback_share == pytest.approx(15 / 50)  # fallback turns / turns
    assert np.array_equal(m.refused_key, a.refused_key)
    assert line_search.merged(None, b) is b


class _Recorder:
    """Stands in for ``evaluate_lines``: records every call, returns equal odds tagged
    with the call number (so nothing ever beats the incumbent)."""

    def __init__(self):
        self.calls = []
        self.refused = []

    def __call__(self, tables, state, lines, n, key, max_turns=100, refused="raise"):
        out = [_odds(n, float(len(self.calls))) for _ in lines]
        self.calls.append((list(lines), n, np.asarray(jax.random.key_data(key)), out))
        self.refused.append(refused)
        return out


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(line_search, "evaluate_lines", rec)
    return rec


class _Rates:
    """Stands in for ``evaluate_lines`` from a table of win rates: line ``i``'s playout
    ``j`` wins when a uniform draw is below ``rate(line)``. The draw is key ``j``'s shared
    one with probability ``rho`` (common random numbers) and the (line, key)'s own
    otherwise, all seeded by the batch key and the line, so a run is deterministic.
    ``rho=None``: no noise, exactly ``round(rate * n)`` wins. Records every call."""

    def __init__(self, rate, rho: float | None = 0.9):
        self.rate, self.rho = rate, rho
        self.calls = []

    def __call__(self, tables, state, lines, n, key, max_turns=100, refused="raise"):
        seed = [int(x) for x in np.asarray(jax.random.key_data(key))]
        shared = np.random.default_rng(seed).random(n)
        out = []
        for line in lines:
            p = self.rate(line)
            if self.rho is None:
                out.append(_odds(n, wins=round(p * n)))
                continue
            tag = zlib.crc32(b"".join(np.asarray(f).tobytes() for f in line))
            rng = np.random.default_rng([*seed, tag])
            own, pick = rng.random(n), rng.random(n) < self.rho
            out.append(_odds(n, wins=int(np.sum(np.where(pick, shared, own) < p))))
        self.calls.append((list(lines), n, np.asarray(seed), out))
        return out


def _same(a: Line, b: Line) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _rule(line, foe: int = 0) -> tuple[int, ...]:
    """``(use, move, first, via, via_from, wait)`` of ``foe``."""
    return tuple(int(f[foe]) for f in line)


def _key(k) -> np.ndarray:
    return np.asarray(jax.random.key_data(k))


def test_hold_out_keys_are_fresh(tables, names, recorder):
    s = roark_state(tables, names, player_count=3, opp_count=2)
    settings = LineSearchSettings(
        race=((8, None), (8, 3), (8, 2)), accept_n=16, restarts=1, final_n=32
    )
    result = find_line(tables, names, s, KEY, settings)
    hold = _key(jax.random.fold_in(KEY, HOLD_OUT_SALT))
    restart = _key(jax.random.fold_in(KEY, RESTART_SALT))
    *search, last = recorder.calls
    assert search
    assert all(not np.array_equal(key, hold) for _, _, key, _ in search)
    assert all(not np.array_equal(key, restart) for _, _, key, _ in search)
    assert len({key.tobytes() for _, _, key, _ in search}) == len(search)  # one key per batch
    assert np.array_equal(last[2], hold) and last[1] == 32
    assert _same(last[0][0], result.line)
    assert result.odds is last[3][0]
    assert result.evaluations == sum(len(lines) * n for lines, n, _, _ in recorder.calls)
    # Every search batch skips all-refused lines; the hold-out leaves refused playouts
    # out (an all-refused hold-out raises).
    assert recorder.refused == ["skip"] * len(search) + ["exclude"]


def test_race_keys_and_counts(tables, names, recorder):
    # One foe, every line scores the same: one visit races, nothing is accepted, the
    # sweep converges and the hold-out follows (no end pick: one line was seen).
    s = roark_state(tables, names, player_count=3, opp_count=1)
    settings = LineSearchSettings(
        race=((8, None), (16, 3), (32, 2)), accept_n=24, max_sweeps=3, restarts=0, final_n=40
    )
    result = find_line(tables, names, s, KEY, settings)
    pool = len(joint_candidates(tables, s, 0))  # "no rule" is the incumbent: left out
    plan = [(1 + pool, 8), (1 + 3, 16), (1 + 2, 32), (2, 24), (1, 40)]
    assert [(len(lines), n) for lines, n, _, _ in recorder.calls] == plan
    for lines, _, _, _ in recorder.calls:
        assert _same(lines[0], empty_line())  # the incumbent is in every batch
    # Survivors are the best by the ranking key; ties keep the earlier candidates.
    first = recorder.calls[0][0]
    for (lines, *_), keep in zip(recorder.calls[1:3], (3, 2), strict=True):
        assert all(_same(a, b) for a, b in zip(lines[1:], first[1 : 1 + keep], strict=True))
    assert _same(recorder.calls[3][0][1], first[1])  # the challenger
    hold = _key(jax.random.fold_in(KEY, HOLD_OUT_SALT))
    restart = _key(jax.random.fold_in(KEY, RESTART_SALT))
    keys = [key for _, _, key, _ in recorder.calls[:-1]]
    assert len({k.tobytes() for k in keys}) == len(keys)
    assert not any(np.array_equal(k, hold) or np.array_equal(k, restart) for k in keys)
    assert result.evaluations == sum(a * b for a, b in plan)
    assert (result.stopped, result.sweeps) == ("converged", 1)


def test_pivot_rule_is_found(tables, names, monkeypatch):
    # The best rule for foe 0 is Starly (3) via Grotle (2); Starly with a plain switch-in
    # scores below Monferno (0) plain, so a who/what stage with plain entry never keeps it.
    s = roark_state(tables, names, player_count=4, opp_count=1)

    def rate(line):
        use, _, _, via, _, wait = _rule(line)
        if wait:
            return 0.2
        return {(3, 2): 0.6, (0, -1): 0.4, (3, -1): 0.3}.get((use, via), 0.2)

    monkeypatch.setattr(line_search, "evaluate_lines", _Rates(rate))
    result = find_line(tables, names, s, KEY, FAST)
    use, _, _, via, _, wait = _rule(result.line)
    assert (use, via, wait) == (3, 2, 0)


def test_from_polish(tables, names, monkeypatch):
    # Starly (3) via Grotle (2) pays most with ``via_from`` Monferno (0, the lead); the
    # polish finds it. Foe 1's rule (Prinplup, 1) adds 1 to the ``via_from`` candidates.
    s = roark_state(tables, names, player_count=4, opp_count=2)

    def rate(line):
        use, _, _, via, via_from, wait = _rule(line)
        if (use, via, wait) != (3, 2, 0):
            return 0.3
        return 0.7 if via_from == 0 else 0.45

    fake = _Rates(rate, rho=None)
    monkeypatch.setattr(line_search, "evaluate_lines", fake)
    start = line_search._with_rule(empty_line(), 1, 1, 0)
    result = find_line(tables, names, s, KEY, FAST, start=start)
    assert _rule(result.line)[3:5] == (2, 0) and _rule(result.line)[0] == 3
    froms = set()
    for lines, *_ in fake.calls:
        for line in lines:
            use, _, _, via, via_from, _ = _rule(line)
            if via_from >= 0:
                assert via_from not in (use, via)
                froms.add(via_from)
    assert froms == {0, 1}
    # The candidates: unset, the lead and the other rules' ``use``, minus ``use`` and ``via``.
    line = line_search._with_rule(start, 0, 0, 0, -1, 2)
    assert line_search._from_candidates(s, line, 0) == [-1, 1]
    line = line_search._with_rule(start, 0, 3, 0, -1, 2)
    assert line_search._from_candidates(s, line, 0) == [-1, 0, 1]


def _one_visit(tables, names, monkeypatch, fake, seed: int):
    """One visit to the lead foe of a 6 v 1 fight (99 candidates with "no rule", about
    Kaizo Roark's 97) with the default race, seed ``seed``: the rule in place after it."""
    s = roark_state(tables, names, player_count=6, opp_count=1)
    monkeypatch.setattr(line_search, "evaluate_lines", fake)
    settings = LineSearchSettings(max_sweeps=1, restarts=0, final_n=8)
    return _rule(find_line(tables, names, s, jax.random.PRNGKey(seed), settings).line)


def test_planted_best_is_found(tables, names, monkeypatch):
    # Common random numbers only (``rho=1``): every line's playout ``j`` reads the same
    # draw, so the planted rule wins every key the others win. With weaker coupling the
    # 64-key first round often drops it (measured in the Task 4c report).
    s = roark_state(tables, names, player_count=6, opp_count=1)
    pool = joint_candidates(tables, s, 0)
    assert 90 <= len(pool) + 1 <= 105
    planted = next(c for c in pool[len(pool) // 2 :] if c[3] == -1 and c[4] == 0)
    want = (*planted[:4], -1, planted[4])

    def rate(line):
        return 0.40 if _rule(line) == want else 0.36

    fake = _Rates(rate, rho=1.0)
    found = [_one_visit(tables, names, monkeypatch, fake, seed) == want for seed in range(10)]
    assert sum(found) >= 9


def test_neutral_challenger_rarely_accepted(tables, names, monkeypatch):
    # Every rule scores the same, the draws are 90 % shared: the challenger beats the
    # incumbent on its fresh keys only by chance, a tie keeps the incumbent, so a little
    # under half the visits accept (each of these seeds' runs is deterministic).
    fake = _Rates(lambda line: 0.36, rho=0.9)
    rules = [_one_visit(tables, names, monkeypatch, fake, seed) for seed in range(20)]
    accepted = sum(r != _rule(empty_line()) for r in rules)
    assert accepted <= 12


def test_converges_and_alternates(tables, names, monkeypatch):
    # Prinplup (1) as the answer adds 0.2 per foe; nothing else matters. Sweep 1 sets it
    # for every foe, sweep 2 (in reverse) changes nothing.
    s = roark_state(tables, names, player_count=3, opp_count=3)

    def rate(line):
        return 0.1 + 0.2 * sum(int(line.use[f]) == 1 for f in range(3))

    monkeypatch.setattr(line_search, "evaluate_lines", _Rates(rate, rho=None))
    order = []
    visit = line_search._visit

    def recording(run, line, foe, *a):
        order.append(foe)
        return visit(run, line, foe, *a)

    monkeypatch.setattr(line_search, "_visit", recording)
    result = find_line(tables, names, s, KEY, FAST)
    assert (result.stopped, result.sweeps) == ("converged", 2)
    assert FAST.max_sweeps > 2
    assert order == [0, 1, 2, 2, 1, 0]
    assert all(int(result.line.use[f]) == 1 for f in range(3))


def test_best_seen_is_kept(tables, names, monkeypatch):
    # Scripted: the first ascent's sweep ends are A then C1, the kick's C2. A is best on
    # the shared batch: the kick starts from it, and it is returned.
    s = roark_state(tables, names, player_count=3, opp_count=2)
    lines = {
        name: line_search._with_rule(empty_line(), 0, use, 0)
        for name, use in (("A", 0), ("C1", 1), ("C2", 2))
    }
    rates = {"A": 0.6, "C1": 0.4, "C2": 0.5}

    def rate(line):
        return next((rates[k] for k, v in lines.items() if _same(v, line)), 0.1)

    fake = _Rates(rate, rho=None)
    monkeypatch.setattr(line_search, "evaluate_lines", fake)
    script = iter([("C1", ["A", "C1"]), ("C2", ["C2"])])
    kicked_from = []

    def ascend(run, line, *a):
        end, ends = next(script)
        return lines[end], len(ends), "converged", [lines[e] for e in ends]

    def kick(rng, line, *a):
        kicked_from.append(line)
        return lines["C2"]

    monkeypatch.setattr(line_search, "_ascend", ascend)
    monkeypatch.setattr(line_search, "_kick", kick)
    settings = LineSearchSettings(accept_n=100, restarts=1, final_n=50)
    result = find_line(tables, names, s, KEY, settings)
    assert len(kicked_from) == 1 and _same(kicked_from[0], lines["A"])
    assert _same(result.line, lines["A"])
    end_pick = fake.calls[-2]
    assert end_pick[1] == 100 and len(end_pick[0]) == 3
    assert result.sweeps == 3


def test_best_seen_on_a_noisy_search(tables, names, monkeypatch):
    # A real (noisy) search with a kick: the returned line scores at least as well as
    # every sweep-end line on the shared end batch.
    s = roark_state(tables, names, player_count=3, opp_count=2)

    def rate(line):
        return 0.2 + 0.1 * (int(line.use[0]) == 2) + 0.1 * (int(line.use[1]) == 1)

    fake = _Rates(rate, rho=0.5)
    monkeypatch.setattr(line_search, "evaluate_lines", fake)
    ends = []
    ascend = line_search._ascend

    def recording(*a):
        out = ascend(*a)
        ends.extend([*out[3], out[0]])
        return out

    monkeypatch.setattr(line_search, "_ascend", recording)
    settings = LineSearchSettings(
        race=((16, None), (32, 4), (64, 2)), accept_n=96, restarts=2, final_n=64
    )
    result = find_line(tables, names, s, KEY, settings)
    lines, n, _, odds = fake.calls[-2]
    assert n == settings.accept_n
    distinct = []
    for e in ends:
        if not any(_same(e, d) for d in distinct):
            distinct.append(e)
    assert len(distinct) > 1
    assert len(lines) == len(distinct)
    assert all(any(_same(e, x) for x in lines) for e in distinct)
    chosen = next(i for i, x in enumerate(lines) if _same(x, result.line))
    k = int(s.faint_budget)
    assert all(odds[chosen].score(k) >= o.score(k) for o in odds)


def _rule_of_foe_0(line) -> tuple[int, int, int]:
    return int(line.use[0]), int(line.move[0]), int(line.first[0])


REFUSED_SETTINGS = LineSearchSettings(race=((100, None),), accept_n=200, restarts=0, final_n=100)


def _often_refused_search(
    tables, names, monkeypatch, refused_share: float, at=None, start_on_rule=False
):
    """``find_line`` on a 3 v 2 fight where every line with foe 0's first who/what
    candidate wins every counted playout, with ``refused_share`` of its playouts refused
    (only in batches of ``at`` keys when given), and every other line wins none; with
    ``start_on_rule`` the search starts on that rule. ``(result, that rule, modes)``."""
    s = roark_state(tables, names, player_count=3, opp_count=2)
    rule = who_what_candidates(tables, s, 0)[0]
    modes = []

    def fake(tables_, state, lines, n, key, max_turns=100, refused="raise"):
        modes.append(refused)
        out = []
        for line in lines:
            if _rule_of_foe_0(line) == rule:
                bad = round(refused_share * n) if at in (None, n) else 0
                out.append(_odds(n - bad, wins=n - bad, refused=bad))
            else:
                out.append(_odds(n))
        return out

    monkeypatch.setattr(line_search, "evaluate_lines", fake)
    start = line_search._with_rule(empty_line(), 0, *rule) if start_on_rule else None
    # Batches of 100 and 200 keys: a share of 0.01 is exactly 1 % of each, and of both.
    return find_line(tables, names, s, KEY, REFUSED_SETTINGS, start=start), rule, modes


def test_an_often_refused_candidate_is_never_chosen(tables, names, monkeypatch):
    # Refused in 1 % of its playouts (at the limit) the best-scoring rule wins ...
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.01)
    assert line_search.MAX_REFUSED_SHARE == 0.01
    assert _rule_of_foe_0(result.line) == rule
    assert result.refused_candidates == 0
    # ... above 1 % it is skipped (never the winner) and counted ...
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.1)
    assert _rule_of_foe_0(result.line) != rule
    assert result.refused_candidates >= 1
    # ... and refused in every playout (n = 0 odds under "skip") it does not stop the search.
    result, rule, modes = _often_refused_search(tables, names, monkeypatch, 1.0)
    assert _rule_of_foe_0(result.line) != rule and result.refused_candidates >= 1
    assert set(modes[:-1]) == {"skip"} and modes[-1] == "exclude"


def test_the_accept_filter_alone_skips_a_refused_challenger(tables, names, monkeypatch):
    # Refused only in the accept batches (200 keys): the race keeps it as the best, the
    # cumulative counts after the accept batch (20 of 300) drop it.
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.1, at=200)
    assert _rule_of_foe_0(result.line) != rule
    assert result.refused_candidates >= 1


def test_a_too_refused_incumbent_is_replaced(tables, names, monkeypatch):
    # The start's rule scores best but is 10 % refused: the challenger replaces it
    # although it scores less.
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.1, start_on_rule=True)
    assert _rule_of_foe_0(result.line) != rule


def test_kicks_meet_among_kept_lines(tables, names, monkeypatch):
    # Two starts (the empty line and one kicked line R) finish unchanged; R scores higher
    # at the end pick. R is chosen unless it is refused in more than 1 % of its playouts.
    s = roark_state(tables, names, player_count=3, opp_count=2)
    kicked = line_search._with_rule(empty_line(), 0, *who_what_candidates(tables, s, 0)[0])
    monkeypatch.setattr(line_search, "_kick", lambda rng, line, *a: kicked)
    monkeypatch.setattr(
        line_search, "_ascend", lambda run, line, *a: (line, 1, "converged", [line])
    )
    settings = LineSearchSettings(accept_n=200, restarts=1, final_n=100)

    def search(share: float):
        def fake(tables_, state, lines, n, key, max_turns=100, refused="raise"):
            out = []
            for line in lines:
                bad = round(share * n) if _same(line, kicked) else 0
                out.append(_odds(n - bad, wins=n - bad if _same(line, kicked) else 0, refused=bad))
            return out

        monkeypatch.setattr(line_search, "evaluate_lines", fake)
        return find_line(tables, names, s, KEY, settings)

    assert _same(search(0.0).line, kicked)
    refused = search(0.1)
    assert _same(refused.line, empty_line()) and refused.refused_candidates == 1


def test_kick_resets_foes_to_no_rule_or_a_top_rule():
    rng = np.random.default_rng(0)
    base = empty_line()
    for f in range(3):
        base = line_search._with_rule(base, f, 0, 0)
    tops = {0: [(1, 1, -1, -1, -1, 0), (2, 0, -1, 1, -1, 0)], 1: [], 2: [(2, 1, -1, -1, -1, 1)]}
    for _ in range(20):
        kicked = line_search._kick(rng, base, [0, 1, 2], tops, 2)
        changed = [f for f in range(3) if _rule(kicked, f) != _rule(base, f)]
        assert len(changed) <= 2
        for f in changed:
            assert _rule(kicked, f) in [_rule(empty_line(), f), *tops[f]]


def test_budget_stops_the_search(tables, names, recorder):
    s = roark_state(tables, names, player_count=3, opp_count=2)
    start = empty_line()
    start.use[1], start.move[1] = 0, 0
    settings = LineSearchSettings(budget_seconds=0, final_n=32)
    result = find_line(tables, names, s, KEY, settings, start=start, clock=lambda: 100.0)
    assert result.stopped == "budget"
    assert _same(result.line, start)
    assert len(recorder.calls) == 1  # the hold-out only
    assert recorder.calls[0][1] == 32
    assert result.odds is recorder.calls[0][3][0]


def test_start_line_is_used(tables, names, recorder):
    s = roark_state(tables, names, player_count=3, opp_count=2)
    # The start's rule is on foe 1 with ``wait``: no foe-0 candidate generates it, so
    # every line of the first batch (foe 0's race) carries it only if ``start`` is used.
    start = empty_line()
    start.use[1], start.move[1], start.wait[1] = 1, 0, 1
    find_line(tables, names, s, KEY, SMALL, start=start)
    first_batch = recorder.calls[0][0]
    assert first_batch
    for line in first_batch:
        assert all(int(f[1]) == int(g[1]) for f, g in zip(line, start, strict=True))


@pytest.fixture(scope="module")
def small_fight(tables, names):
    s = roark_state(tables, names, player_count=3, opp_count=2)
    return s, find_line(tables, names, s, KEY, SMALL)


@pytest.mark.slow
def test_find_line_is_no_worse_than_fallback(tables, small_fight):
    s, result = small_fight
    k = int(s.faint_budget)
    fallback, found = evaluate_lines(
        tables, s, [empty_line(), result.line], 256, jax.random.PRNGKey(99), SMALL.max_turns
    )
    assert found.score(k)[0] >= fallback.score(k)[0] - 5
    assert result.stopped in ("converged", "max_sweeps")
    assert result.odds.n == SMALL.final_n


@pytest.mark.slow
def test_find_line_is_deterministic(tables, names, small_fight):
    s, first = small_fight
    again = find_line(tables, names, s, KEY, SMALL)
    assert _same(first.line, again.line)
    assert first.odds == again.odds
