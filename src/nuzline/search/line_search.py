"""Searching for a line (docs/superpowers/specs/2026-11-12-l1-line-player-design.md §5):
host-side coordinate ascent on common random numbers around ``evaluate_lines``. A
heuristic: it finds a good line, not the best one.

Candidates for the foe in opposing slot ``f`` (one with ``state.species[1, f] > 0``):

* who and what (``who_what_candidates``): ``use`` a living party slot, ``move`` one of its
  move slots with power > 0 whose ``damage_table`` against ``f`` (``use`` and ``f`` both
  active, the given state otherwise) has a positive roll, ``first`` -1 or one of its
  power-0 move slots that inflicts a major status (``MAJOR_STATUSES``) the foe can receive
  at the given state (``_receives``); plain entry. Plus "no rule" (every field unset).
* the entry for the rule then in place (``entry_candidates``): plain, ``wait``, or
  ``via`` each other living slot; ``via_from`` is never searched (it stays -1).

A stage scores its candidates that differ from the incumbent, the rest of the line fixed:
one batch of ``screen_n`` keys over all of them (skipped when there are at most ``keep``:
it could not cut any), then one batch of ``refine_n`` fresh keys over the incumbent and
the best ``keep``. The best of those replaces the incumbent only when its ``score(k)``
(``k = int(state.faint_budget)``) is strictly greater than the incumbent's on the same
keys; ties keep the earlier candidate. A sweep runs both stages for each foe
in party order; sweeps repeat until one changes nothing (``"converged"``),
``max_sweeps`` or the time budget (``"budget"``, checked before each stage: a started
stage finishes).

The first start is ``start`` or the empty line; ``restarts`` more start from random lines
(each foe: "no rule" or a who/what candidate, uniformly, plain entry; one
``numpy.random.default_rng`` seeded by ``key_data(fold_in(key, RESTART_SALT))``). With
more than one start the finished lines meet on one shared batch of ``refine_n`` keys.
The chosen line is scored once more on ``final_n`` hold-out keys
(``fold_in(key, HOLD_OUT_SALT)``); only these odds are reported.

Every search batch has its own key ``fold_in(key, c)``, ``c`` counting batches from 0
and skipping the two salts, so no two batches, and never the hold-out, share keys.

Refused playouts (spec §4) are left out of every batch: the search batches use
``evaluate_lines(..., refused="skip")`` (a line refused in every playout gets ``n = 0``
odds rather than aborting the search), the hold-out ``"exclude"`` (its ``refused`` is in
``odds``; an all-refused hold-out raises ``UnsupportedAi``). A line whose batch has
``refused / (n + refused) > MAX_REFUSED_SHARE`` is too refused. A too-refused candidate
is skipped in that stage, at the screen or the refine: it is never the winner and never
replaces the incumbent. A too-refused incumbent is replaced by the best kept finalist
even without a greater score. The restarts' meeting chooses among the finished lines
that are not too refused (all of them when none is). ``LineSearchResult.refused_candidates``
counts the distinct lines skipped.
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
from nuzline.search.line import Line, empty_line
from nuzline.search.player_eval import PlayerOdds, evaluate_lines
from nuzline.search.playout import _facing

__all__ = [
    "HOLD_OUT_SALT",
    "MAJOR_STATUSES",
    "MAX_REFUSED_SHARE",
    "RESTART_SALT",
    "LineSearchResult",
    "LineSearchSettings",
    "entry_candidates",
    "find_line",
    "who_what_candidates",
]

HOLD_OUT_SALT = 0x1E1D  # fold_in(key, HOLD_OUT_SALT): the hold-out keys
RESTART_SALT = 0xBEEF  # fold_in(key, RESTART_SALT): the random restart lines' seed
MAX_REFUSED_SHARE = 0.01  # a candidate refused in more of its playouts is skipped

# The statuses a ``first`` opener may inflict. The engine has one status-move effect,
# ``EffectId.STATUS_FOE``, whose ``tables.move_param[m, 0]`` is the ``Status`` it applies
# (ids.py ``EffectId`` docstring; ``effects._status_foe`` -> ``effects._inflict`` writes
# ``h.param[0]`` into ``state.status``, effects.py:353-393). An opener is a
# power-0 ``STATUS_FOE`` move whose param is one of these ``ids.Status`` values: sleep,
# poison, toxic, burn, paralysis (``Status.FREEZE``, the sixth, is left out by the spec).
MAJOR_STATUSES = (Status.SLEEP, Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)


@dataclass(frozen=True)
class LineSearchSettings:
    screen_n: int = 128
    keep: int = 4
    refine_n: int = 512
    max_sweeps: int = 4
    restarts: int = 1
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
    refused_candidates: int = 0  # distinct lines skipped for refused playouts


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


def _with_rule(line: Line, foe: int, use=-1, move=-1, first=-1, via=-1, via_from=-1, wait=0):
    """A copy of ``line`` with ``foe``'s rule set to these fields."""
    fields = [np.array(f, np.int32, copy=True) for f in line]
    for f, v in zip(fields, (use, move, first, via, via_from, wait), strict=True):
        f[foe] = v
    return Line(*fields)


def _same(a: Line, b: Line) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


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
        self.skipped: set[bytes] = set()  # the lines skipped for refused playouts

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
                self.skipped.add(b"".join(np.asarray(f).tobytes() for f in line))
            else:
                out.append(i)
        return out

    def stage(self, incumbent: Line, candidates: list[Line]) -> tuple[Line, PlayerOdds | None]:
        """The stage's winner (module docstring) and its refine odds (``None``: nothing
        was scored)."""
        others = [c for c in candidates if not _same(c, incumbent)]
        if self.out_of_time() or not others:
            return incumbent, None
        finalists = others
        if len(others) > self.settings.keep:
            screened = self.evaluate(others, self.settings.screen_n)
            order = sorted(
                self.kept(others, screened), key=lambda i: screened[i].score(self.k), reverse=True
            )
            finalists = [others[i] for i in order[: self.settings.keep]]
        if not finalists:
            return incumbent, None
        refined = self.evaluate([incumbent, *finalists], self.settings.refine_n)
        fine = [i + 1 for i in self.kept(finalists, refined[1:])]
        if fine:
            best = max(fine, key=lambda i: refined[i].score(self.k))
            beats = refined[best].score(self.k) > refined[0].score(self.k)
            if beats or _too_refused(refined[0]):
                return finalists[best - 1], refined[best]
        return incumbent, refined[0]


def _too_refused(odds: PlayerOdds) -> bool:
    """``refused / (n + refused) > MAX_REFUSED_SHARE`` (module docstring)."""
    return odds.refused > MAX_REFUSED_SHARE * (odds.n + odds.refused)


def _rule_of(line: Line, foe: int) -> tuple[int, ...]:
    return tuple(int(f[foe]) for f in line)


def _visit(run: _Run, line: Line, foe: int, options) -> tuple[Line, PlayerOdds | None]:
    """Both stages for ``foe``: who and what, then the entry for the winner."""
    who = [_with_rule(line, foe)] + [_with_rule(line, foe, *o) for o in options]
    line, odds = run.stage(line, who)
    use, move, first = _rule_of(line, foe)[:3]
    if use < 0 or use not in _living(run.state):
        return line, odds
    entries = [
        _with_rule(line, foe, use, move, first, via, -1, wait)
        for via, wait in entry_candidates(run.state, use)
    ]
    line, entry_odds = run.stage(line, entries)
    return line, entry_odds or odds


def _say_sweep(progress, label: str, sweep: int, odds: PlayerOdds | None, k: int) -> None:
    if progress is None:
        return
    if odds is None:
        progress(f"{label}sweep {sweep}: nothing to score")
        return
    rung, win = 100 * odds.rung(k), 100 * odds.wins_any / odds.n
    progress(f"{label}sweep {sweep}: {rung:.1f} % at <={k}, {win:.1f} % win ({odds.n} playouts)")


def _ascend(run: _Run, line: Line, foes, options, progress, label: str):
    """Coordinate ascent from ``line``: ``(line, sweeps finished, stopped)``."""
    odds = None
    for sweep in range(1, run.settings.max_sweeps + 1):
        before = line
        for foe in foes:
            line, scored = _visit(run, line, foe, options[foe])
            odds = scored or odds
            if run.expired:
                return line, sweep - 1, "budget"
        _say_sweep(progress, label, sweep, odds, run.k)
        if _same(line, before):
            return line, sweep, "converged"
    return line, run.settings.max_sweeps, "max_sweeps"


def _random_line(rng: np.random.Generator, foes, options) -> Line:
    line = empty_line()
    for foe in foes:
        pick = int(rng.integers(len(options[foe]) + 1))
        if pick:
            line = _with_rule(line, foe, *options[foe][pick - 1])
    return line


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
    options = {f: who_what_candidates(tables, state, f) for f in foes}
    seed = np.asarray(jax.random.key_data(jax.random.fold_in(key, RESTART_SALT)))
    rng = np.random.default_rng(seed)
    starts = [empty_line() if start is None else start]
    starts += [_random_line(rng, foes, options) for _ in range(settings.restarts)]
    finished, sweeps, reasons = [], 0, []
    for i, first in enumerate(starts):
        if i and run.out_of_time():
            break
        label = f"restart {i} " if i else ""
        line, done, why = _ascend(run, first, foes, options, progress, label)
        finished.append(line)
        sweeps += done
        reasons.append(why)
    chosen = 0
    if len(finished) > 1:
        odds = run.evaluate(finished, settings.refine_n)
        pool = run.kept(finished, odds) or range(len(odds))
        chosen = max(pool, key=lambda i: odds[i].score(run.k))
    hold_out = jax.random.fold_in(key, HOLD_OUT_SALT)
    final = run.evaluate([finished[chosen]], settings.final_n, hold_out, refused="exclude")[0]
    return LineSearchResult(
        line=finished[chosen],
        odds=final,
        sweeps=sweeps,
        evaluations=run.evaluations,
        stopped="budget" if run.expired else reasons[chosen],
        refused_candidates=len(run.skipped),
    )
