"""Searching for a line (docs/superpowers/specs/2026-11-12-l1-line-player-design.md §5):
host-side coordinate ascent on common random numbers around ``evaluate_lines``. A
heuristic: it finds a good line, not the best one.

Candidates for the foe in opposing slot ``f`` (one with ``state.species[1, f] > 0``):

* who and what (``who_what_candidates``): ``use`` a living party slot, ``move`` one of its
  move slots with power > 0 whose ``damage_table`` against ``f`` (``use`` and ``f`` both
  active, the given state otherwise) has a positive roll, ``first`` -1 or one of its
  power-0 move slots that inflicts a major status (``MAJOR_STATUSES``) the foe can receive
  at the given state (``_receives``).
* the entry (``entry_candidates``): plain, ``wait``, or ``via`` each other living slot.
* the joint pool (``joint_candidates``): every who/what with every entry for its ``use``,
  ``via_from`` -1; plus "no rule" (every field unset).

A race (``_Run.race``) scores candidates that differ from the incumbent, the rest of the
line fixed. Round ``r`` of ``settings.race`` (``(new keys, survivors)``) first keeps the
best ``survivors`` (``None``: all; never before round 0) by the ranking key
``wins[k] + wins_any`` on the counts summed over the earlier rounds
(``k = int(state.faint_budget)``; ties keep the earlier candidate), then plays them and
the incumbent on ``new keys`` fresh keys and adds the counts (``merged``). After the last
round the best survivor by ``score(k)`` on its summed counts (the challenger) and the
incumbent play ``accept_n`` fresh keys; the challenger replaces the incumbent only when
its ``score(k)`` there is strictly greater. A race's top ``KICK_TOP`` survivors by
``score(k)`` are kept for the kicks.

A visit to foe ``f`` races the joint pool; then, when the rule in place has a ``via``, a
small race over its ``via_from`` (``_from_candidates``: unset, the player's lead
``state.active[0]`` and the ``use`` of every other rule, minus the rule's ``use`` and
``via``). Sweep ``s`` (from 1) visits the foes in party order when ``s`` is odd, in
reverse when even; sweeps repeat until one changes nothing (``"converged"``),
``max_sweeps`` or the time budget (``"budget"``, checked before each race and each kick:
a started race finishes).

The first start is ``start`` or the empty line. The line at the end of every sweep and
of every start is kept. ``restarts`` kicks follow, each from the best kept line so far
(``_pick``): ``kick_foes`` distinct foes (one ``numpy.random.default_rng`` seeded by
``key_data(fold_in(key, RESTART_SALT))``) each get "no rule" or one of their last joint
race's top rules, uniformly, and the ascent starts again. At the end ``_pick`` scores the
distinct kept lines on one batch of ``accept_n`` fresh keys (none when only one was kept)
and takes the best by ``score(k)``. The chosen line is scored once more on ``final_n``
hold-out keys (``fold_in(key, HOLD_OUT_SALT)``); only these odds are reported.

Every search batch has its own key ``fold_in(key, c)``, ``c`` counting batches from 0
and skipping the two salts, so no two batches, and never the hold-out, share keys.

Refused playouts (spec §4) are left out of every batch: the search batches use
``evaluate_lines(..., refused="skip")`` (a line refused in every playout gets ``n = 0``
odds rather than aborting the search), the hold-out ``"exclude"`` (its ``refused`` is in
``odds``; an all-refused hold-out raises ``UnsupportedAi``). A line whose summed counts
have ``refused / (n + refused) > MAX_REFUSED_SHARE`` is too refused. A too-refused
candidate drops out of the race (after any round, or after the accept batch for the
challenger): it is never the challenger and never replaces the incumbent. A too-refused
incumbent (on its counts summed over the race and the accept batch) is replaced by a
challenger that is not, even without a greater score. ``_pick`` chooses among the lines
that are not too refused on its batch (all of them when none is).
``LineSearchResult.refused_candidates`` counts the distinct lines dropped.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import numpy as np

from nuzline.data.tables import Names, Tables
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import damage_table, effectiveness, move_type_of
from nuzline.engine.effects import can_have_status
from nuzline.engine.ids import EffectId, Status
from nuzline.engine.state import BattleState
from nuzline.search import deaths
from nuzline.search.line import Line, empty_line
from nuzline.search.player_eval import PlayerOdds, evaluate_lines
from nuzline.search.playout import _facing

__all__ = [
    "HOLD_OUT_SALT",
    "KICK_TOP",
    "MAJOR_STATUSES",
    "MAX_REFUSED_SHARE",
    "RESTART_SALT",
    "LineSearchResult",
    "LineSearchSettings",
    "entry_candidates",
    "find_line",
    "joint_candidates",
    "merged",
    "who_what_candidates",
]

HOLD_OUT_SALT = 0x1E1D  # fold_in(key, HOLD_OUT_SALT): the hold-out keys
RESTART_SALT = 0xBEEF  # fold_in(key, RESTART_SALT): the kicks' seed
MAX_REFUSED_SHARE = 0.01  # a candidate refused in more of its playouts is dropped
KICK_TOP = 4  # a kick picks among this many of a foe's last race's best

# The statuses a ``first`` opener may inflict. The engine has one status-move effect,
# ``EffectId.STATUS_FOE``, whose ``tables.move_param[m, 0]`` is the ``Status`` it applies
# (ids.py ``EffectId`` docstring; ``effects._status_foe`` -> ``effects._inflict`` writes
# ``h.param[0]`` into ``state.status``, effects.py:353-393). An opener is a
# power-0 ``STATUS_FOE`` move whose param is one of these ``ids.Status`` values: sleep,
# poison, toxic, burn, paralysis (``Status.FREEZE``, the sixth, is left out by the spec).
MAJOR_STATUSES = (Status.SLEEP, Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)


@dataclass(frozen=True)
class LineSearchSettings:
    # (new keys this round, survivors kept before the round; None = all):
    # cumulative keys 64, 256, 512
    race: tuple[tuple[int, int | None], ...] = ((64, None), (192, 16), (256, 4))
    accept_n: int = 768  # fresh keys for the one challenger vs the incumbent
    max_sweeps: int = 12
    restarts: int = 1  # kicks from the best line
    kick_foes: int = 2
    final_n: int = 2000
    budget_seconds: float | None = None
    max_turns: int = 100


DEFAULT_SETTINGS = LineSearchSettings()  # frozen: safe as ``find_line``'s default


@dataclass(frozen=True)
class LineSearchResult:
    line: Line
    odds: PlayerOdds  # hold-out: keys the search never used
    sweeps: int  # sweeps finished, summed over the starts
    evaluations: int  # line x key playouts run, hold-out included
    stopped: str  # "converged", "max_sweeps" or "budget"
    refused_candidates: int = 0  # distinct lines dropped for refused playouts


@jax.jit
def _rolls(tables: Tables, state: BattleState, use, foe, move):
    """``damage_table`` of the player's ``use`` with ``move`` against opposing ``foe``,
    both sent in."""
    s = _facing(_facing(state, 0, MOVE_ACTIONS + use), 1, MOVE_ACTIONS + foe)
    return damage_table(tables, s, 0, move, False)


@jax.jit
def _receives(tables: Tables, state: BattleState, use, foe, move):
    """Opposing ``foe`` can receive ``move``'s ``STATUS_FOE`` status from the player's
    ``use``, both sent in: the two checks the engine applies to it. A type-checked move
    (``move_param[m, 1] == 1``, Thunder Wave) needs a non-zero matchup (effects.py:1120-1131,
    ``type_ok``), and ``can_have_status`` must hold (``_inflict``, effects.py:375). The
    Insomnia / Limber blocks there (effects.py:362-365) are not applied."""
    s = _facing(_facing(state, 0, MOVE_ACTIONS + use), 1, MOVE_ACTIONS + foe)
    param = tables.move_param[move]
    hits = effectiveness(
        tables, move_type_of(tables, s, 0, move), s.types[1, foe], s.foresight[1] != 0
    )
    type_ok = (param[1] == 0) | (hits > 0)
    return type_ok & can_have_status(tables, s, 1, foe, param[0])


def _living(state: BattleState) -> list[int]:
    alive = (np.asarray(state.species[0]) > 0) & (np.asarray(state.hp[0]) > 0)
    return [int(i) for i in np.flatnonzero(alive)]


def who_what_candidates(tables: Tables, state: BattleState, foe: int) -> list[tuple[int, int, int]]:
    """Every ``(use, move, first)`` for opposing slot ``foe`` (module docstring), in
    ``use``, ``move``, ``first`` order with ``first == -1`` first."""
    power = np.asarray(tables.move_power)
    effect = np.asarray(tables.move_effect)
    param = np.asarray(tables.move_param)
    moves = np.asarray(state.moves)
    out = []
    for use in _living(state):
        ids = [int(m) for m in moves[0, use]]
        damaging = [
            slot
            for slot, m in enumerate(ids)
            if m > 0 and power[m] > 0 and np.any(np.asarray(_rolls(tables, state, use, foe, m)) > 0)
        ]
        openers = [
            slot
            for slot, m in enumerate(ids)
            if m > 0
            and power[m] == 0
            and effect[m] == EffectId.STATUS_FOE
            and param[m, 0] in MAJOR_STATUSES
            and bool(_receives(tables, state, use, foe, m))
        ]
        out += [(use, move, first) for move in damaging for first in (-1, *openers)]
    return out


def entry_candidates(state: BattleState, use: int) -> list[tuple[int, int]]:
    """Every ``(via, wait)`` for an answer ``use``: plain, ``wait``, then ``via`` each
    other living slot."""
    return [(-1, 0), (-1, 1), *((v, 0) for v in _living(state) if v != use)]


def joint_candidates(
    tables: Tables, state: BattleState, foe: int
) -> list[tuple[int, int, int, int, int]]:
    """Every ``(use, move, first, via, wait)`` for opposing slot ``foe``: each
    ``who_what_candidates`` entry with each ``entry_candidates`` entry for its ``use``,
    in that order (``via_from`` -1)."""
    return [
        (use, move, first, via, wait)
        for use, move, first in who_what_candidates(tables, state, foe)
        for via, wait in entry_candidates(state, use)
    ]


def _with_rule(line: Line, foe: int, use=-1, move=-1, first=-1, via=-1, via_from=-1, wait=0):
    """A copy of ``line`` with ``foe``'s rule set to these fields."""
    fields = [np.array(f, np.int32, copy=True) for f in line]
    for f, v in zip(fields, (use, move, first, via, via_from, wait), strict=True):
        f[foe] = v
    return Line(*fields)


def _same(a: Line, b: Line) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _bytes(line: Line) -> bytes:
    return b"".join(np.asarray(f, np.int32).tobytes() for f in line)


def merged(a: PlayerOdds | None, b: PlayerOdds) -> PlayerOdds:
    """One line's odds over both batches (``a`` ``None``: ``b``): ``n``, ``wins``,
    ``wins_any``, ``deaths_total`` and ``refused`` summed; the distribution from the
    summed bucket counts; ``mean_turns`` weighted by playouts and ``fallback_share`` by
    turns (both exact); ``refused_key`` the first batch's that has one."""
    if a is None:
        return b
    n = a.n + b.n
    counts = [pa * a.n + pb * b.n for pa, pb in zip(a.distribution.probs, b.distribution.probs)]
    turns_a, turns_b = a.mean_turns * a.n, b.mean_turns * b.n
    turns = turns_a + turns_b
    return PlayerOdds(
        n=n,
        wins=tuple(x + y for x, y in zip(a.wins, b.wins, strict=True)),
        wins_any=a.wins_any + b.wins_any,
        deaths_total=a.deaths_total + b.deaths_total,
        distribution=deaths.DeathDistribution(tuple(c / n if n else 0.0 for c in counts), n),
        mean_turns=turns / n if n else 0.0,
        fallback_share=(
            (a.fallback_share * turns_a + b.fallback_share * turns_b) / turns if turns else 0.0
        ),
        refused=a.refused + b.refused,
        refused_key=a.refused_key if a.refused_key is not None else b.refused_key,
    )


def _too_refused(odds: PlayerOdds) -> bool:
    """``refused / (n + refused) > MAX_REFUSED_SHARE`` (module docstring)."""
    return odds.refused > MAX_REFUSED_SHARE * (odds.n + odds.refused)


class _Run:
    """One search's key counter, playout count and clock."""

    def __init__(self, tables, state, key, settings: LineSearchSettings, clock):
        self.tables, self.state, self.key, self.settings = tables, state, key, settings
        self.k = int(state.faint_budget)
        self.counter = 0
        self.evaluations = 0
        self.clock = clock
        self.started = clock()
        self.expired = False
        self.skipped: set[bytes] = set()  # the lines dropped for refused playouts

    def out_of_time(self) -> bool:
        budget = self.settings.budget_seconds
        if budget is not None and self.clock() - self.started >= budget:
            self.expired = True
        return self.expired

    def evaluate(
        self, lines: Sequence[Line], n: int, key=None, refused: str = "skip"
    ) -> list[PlayerOdds]:
        """``evaluate_lines`` on the next batch key (or ``key``), with ``refused``."""
        if key is None:
            while self.counter in (HOLD_OUT_SALT, RESTART_SALT):
                self.counter += 1
            key = jax.random.fold_in(self.key, self.counter)
            self.counter += 1
        self.evaluations += len(lines) * n
        return evaluate_lines(
            self.tables, self.state, lines, n, key, self.settings.max_turns, refused=refused
        )

    def kept(self, lines: Sequence[Line], odds: Sequence[PlayerOdds]) -> list[int]:
        """The indices of ``lines`` not refused too often (module docstring); the others
        are recorded as skipped."""
        out = []
        for i, (line, o) in enumerate(zip(lines, odds, strict=True)):
            if _too_refused(o):
                self.skipped.add(_bytes(line))
            else:
                out.append(i)
        return out

    def _rank(self, odds: PlayerOdds) -> int:
        return odds.wins[self.k] + odds.wins_any

    def race(self, incumbent: Line, candidates: list[Line]):
        """``(line, odds, top)``: the line in place after the race (module docstring),
        its accept-batch odds (``None``: nothing was scored) and the race's best
        ``KICK_TOP`` survivors."""
        others = [c for c in candidates if not _same(c, incumbent)]
        if self.out_of_time() or not others:
            return incumbent, None, []
        totals: list[PlayerOdds | None] = [None] * len(others)
        base = None  # the incumbent's summed counts
        alive = list(range(len(others)))
        for r, (new, survivors) in enumerate(self.settings.race):
            if r and survivors is not None:
                alive = sorted(alive, key=lambda i: self._rank(totals[i]), reverse=True)
                alive = alive[:survivors]
            odds = self.evaluate([incumbent, *(others[i] for i in alive)], new)
            base = merged(base, odds[0])
            for i, o in zip(alive, odds[1:], strict=True):
                totals[i] = merged(totals[i], o)
            alive = [alive[j] for j in self.kept([others[i] for i in alive], [totals[i] for i in alive])]
            if not alive:
                return incumbent, None, []
        order = sorted(alive, key=lambda i: totals[i].score(self.k), reverse=True)
        top = [others[i] for i in order[:KICK_TOP]]
        best = order[0]
        held, tried = self.evaluate([incumbent, others[best]], self.settings.accept_n)
        base = merged(base, held)
        if self.kept([others[best]], [merged(totals[best], tried)]):
            if tried.score(self.k) > held.score(self.k) or _too_refused(base):
                return others[best], tried, top
        return incumbent, held, top


def _rule_of(line: Line, foe: int) -> tuple[int, ...]:
    return tuple(int(f[foe]) for f in line)


def _from_candidates(state: BattleState, line: Line, foe: int) -> list[int]:
    """The ``via_from`` values a polish tries for ``foe``'s rule (module docstring),
    ascending."""
    use, via = int(line.use[foe]), int(line.via[foe])
    others = {int(u) for g, u in enumerate(np.asarray(line.use)) if g != foe and u >= 0}
    return sorted({-1, int(np.asarray(state.active[0])), *others} - {use, via})


def _visit(run: _Run, line: Line, foe: int, options, tops) -> tuple[Line, PlayerOdds | None]:
    """The joint race for ``foe``, then the ``via_from`` polish when its rule has a
    ``via``; ``tops[foe]`` gets the joint race's top rules."""
    pool = [_with_rule(line, foe)]
    pool += [_with_rule(line, foe, u, m, f, v, -1, w) for u, m, f, v, w in options]
    line, odds, top = run.race(line, pool)
    if top:
        tops[foe] = [_rule_of(t, foe) for t in top]
    use, move, first, via, _, wait = _rule_of(line, foe)
    if via < 0:
        return line, odds
    polish = [
        _with_rule(line, foe, use, move, first, via, f, wait)
        for f in _from_candidates(run.state, line, foe)
    ]
    line, polished, _ = run.race(line, polish)
    return line, polished or odds


def _say_sweep(progress, label: str, sweep: int, odds: PlayerOdds | None, k: int) -> None:
    if progress is None:
        return
    if odds is None or odds.n == 0:
        progress(f"{label}sweep {sweep}: nothing to score")
        return
    rung, win = 100 * odds.rung(k), 100 * odds.wins_any / odds.n
    progress(f"{label}sweep {sweep}: {rung:.1f} % at <={k}, {win:.1f} % win ({odds.n} playouts)")


def _ascend(run: _Run, line: Line, foes, options, tops, progress, label: str):
    """Coordinate ascent from ``line``: ``(line, sweeps finished, stopped, the line at
    the end of each finished sweep)``."""
    odds, ends = None, []
    for sweep in range(1, run.settings.max_sweeps + 1):
        before = line
        for foe in foes if sweep % 2 else foes[::-1]:
            line, scored = _visit(run, line, foe, options[foe], tops)
            odds = scored or odds
            if run.expired:
                return line, sweep - 1, "budget", ends
        ends.append(line)
        _say_sweep(progress, label, sweep, odds, run.k)
        if _same(line, before):
            return line, sweep, "converged", ends
    return line, run.settings.max_sweeps, "max_sweeps", ends


def _kick(rng: np.random.Generator, line: Line, foes, tops, kick_foes: int) -> Line:
    """``line`` with ``kick_foes`` random foes set to "no rule" or one of their ``tops``
    rules (module docstring)."""
    for foe in rng.choice(foes, size=min(kick_foes, len(foes)), replace=False):
        rules = tops.get(int(foe), [])
        pick = int(rng.integers(len(rules) + 1))
        line = _with_rule(line, int(foe), *rules[pick - 1]) if pick else _with_rule(line, int(foe))
    return line


def _pick(run: _Run, seen: list[tuple[Line, str]]) -> tuple[Line, str]:
    """The best distinct kept line and its ascent's stop (module docstring)."""
    distinct: dict[bytes, tuple[Line, str]] = {}
    for line, why in seen:
        distinct.setdefault(_bytes(line), (line, why))
    pool = list(distinct.values())
    if len(pool) == 1:
        return pool[0]
    odds = run.evaluate([line for line, _ in pool], run.settings.accept_n)
    kept = run.kept([line for line, _ in pool], odds) or range(len(pool))
    return pool[max(kept, key=lambda i: odds[i].score(run.k))]


def find_line(
    tables: Tables,
    names: Names,
    state: BattleState,
    key,
    settings: LineSearchSettings = DEFAULT_SETTINGS,
    start: Line | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> LineSearchResult:
    """The line the search settles on and its hold-out odds (module docstring). ``key``
    is a legacy ``PRNGKey``; ``clock`` measures the budget (injectable for tests)."""
    del names  # the search works on ids; kept for the commands' uniform signature
    run = _Run(tables, state, key, settings, clock)
    foes = [int(f) for f in np.flatnonzero(np.asarray(state.species[1]) > 0)]
    options = {f: joint_candidates(tables, state, f) for f in foes}
    seed = np.asarray(jax.random.key_data(jax.random.fold_in(key, RESTART_SALT)))
    rng = np.random.default_rng(seed)
    tops: dict[int, list[tuple[int, ...]]] = {}
    seen: list[tuple[Line, str]] = []
    line, sweeps = (empty_line() if start is None else start), 0
    for i in range(settings.restarts + 1):
        if i:
            if run.out_of_time():
                break
            line = _kick(rng, _pick(run, seen)[0], foes, tops, settings.kick_foes)
        label = f"kick {i} " if i else ""
        end, done, why, ends = _ascend(run, line, foes, options, tops, progress, label)
        sweeps += done
        seen += [(e, why) for e in (*ends, end)]
    chosen, why = _pick(run, seen)
    hold_out = jax.random.fold_in(key, HOLD_OUT_SALT)
    final = run.evaluate([chosen], settings.final_n, hold_out, refused="exclude")[0]
    return LineSearchResult(
        line=chosen,
        odds=final,
        sweeps=sweeps,
        evaluations=run.evaluations,
        stopped="budget" if run.expired else why,
        refused_candidates=len(run.skipped),
    )
