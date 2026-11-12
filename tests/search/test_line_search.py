"""The line search (spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md §5):
the candidate rules per foe, the key discipline (the hold-out never shares keys with the
search), the time budget and the start line. Most tests replace ``evaluate_lines`` with a
recorder, so only the two marked slow compile a playout program."""

import jax
import numpy as np
import pytest

from nuzline.engine.ids import EffectId, Status
from nuzline.search import deaths, line_search
from nuzline.search.line import Line, empty_line
from nuzline.search.line_search import (
    HOLD_OUT_SALT,
    MAJOR_STATUSES,
    LineSearchSettings,
    entry_candidates,
    find_line,
    who_what_candidates,
)
from nuzline.search.player_eval import PlayerOdds, evaluate_lines
from tests.engine.helpers import roark_state

KEY = jax.random.PRNGKey(11)
SMALL = LineSearchSettings(
    screen_n=32, keep=2, refine_n=64, max_sweeps=2, restarts=0, final_n=128, max_turns=40
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


def _same(a: Line, b: Line) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_hold_out_keys_are_fresh(tables, names, recorder):
    s = roark_state(tables, names, player_count=3, opp_count=2)
    settings = LineSearchSettings(screen_n=8, keep=2, refine_n=16, restarts=1, final_n=32)
    result = find_line(tables, names, s, KEY, settings)
    hold = np.asarray(jax.random.key_data(jax.random.fold_in(KEY, HOLD_OUT_SALT)))
    *search, last = recorder.calls
    assert search
    assert all(not np.array_equal(key, hold) for _, _, key, _ in search)
    assert len({key.tobytes() for _, _, key, _ in search}) == len(search)  # one key per batch
    assert np.array_equal(last[2], hold) and last[1] == 32
    assert _same(last[0][0], result.line)
    assert result.odds is last[3][0]
    assert result.evaluations == sum(len(lines) * n for lines, n, _, _ in recorder.calls)
    # Every search batch skips all-refused lines; the hold-out leaves refused playouts
    # out (an all-refused hold-out raises).
    assert recorder.refused == ["skip"] * len(search) + ["exclude"]


def _rule_of_foe_0(line) -> tuple[int, int, int]:
    return int(line.use[0]), int(line.move[0]), int(line.first[0])


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
    # Batches of 100 and 200 keys: a share of 0.01 is exactly 1 % of each.
    settings = LineSearchSettings(screen_n=100, keep=2, refine_n=200, restarts=0, final_n=100)
    return find_line(tables, names, s, KEY, settings, start=start), rule, modes


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


def test_the_refine_filter_alone_skips_a_refused_finalist(tables, names, monkeypatch):
    # Refused only in the refine batches (200 keys): the screen keeps it as the best,
    # the refine skips it.
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.1, at=200)
    assert _rule_of_foe_0(result.line) != rule
    assert result.refused_candidates >= 1


def test_a_too_refused_incumbent_is_replaced(tables, names, monkeypatch):
    # The start's rule scores best but is 10 % refused: the best kept finalist replaces
    # it although it scores less.
    result, rule, _ = _often_refused_search(tables, names, monkeypatch, 0.1, start_on_rule=True)
    assert _rule_of_foe_0(result.line) != rule


def test_restarts_meet_among_kept_lines(tables, names, monkeypatch):
    # Two starts (the empty line and one restart line R) finish unchanged; R scores higher
    # at the meeting. R is chosen unless it is refused in more than 1 % of its playouts.
    s = roark_state(tables, names, player_count=3, opp_count=2)
    restart = line_search._with_rule(empty_line(), 0, *who_what_candidates(tables, s, 0)[0])
    monkeypatch.setattr(line_search, "_random_line", lambda rng, foes, options: restart)
    monkeypatch.setattr(line_search, "_ascend", lambda run, line, *a: (line, 1, "converged"))
    settings = LineSearchSettings(refine_n=200, restarts=1, final_n=100)

    def search(share: float):
        def fake(tables_, state, lines, n, key, max_turns=100, refused="raise"):
            out = []
            for line in lines:
                bad = round(share * n) if _same(line, restart) else 0
                out.append(_odds(n - bad, wins=n - bad if _same(line, restart) else 0, refused=bad))
            return out

        monkeypatch.setattr(line_search, "evaluate_lines", fake)
        return find_line(tables, names, s, KEY, settings)

    assert _same(search(0.0).line, restart)
    refused = search(0.1)
    assert _same(refused.line, empty_line()) and refused.refused_candidates == 1


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
    # every line of the first batch (foe 0's screen) carries it only if ``start`` is used.
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
