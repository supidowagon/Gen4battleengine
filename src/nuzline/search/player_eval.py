"""Scoring simulated players on shared playout keys
(docs/superpowers/specs/2026-11-12-l1-line-player-design.md): every ``params`` entry of
one ``Player`` (a "line") plays ``player_playout`` from the same state on the same ``n``
keys (``line_keys``; common random numbers), so lines differ only by their choices.

The program runs over flat rows, one per (line, key) pair: row ``i * n + j`` is line
``i`` on key ``j``. Rows go in chunks of ``ROW_CHUNK`` rounded up to a multiple of the
device count, padded with copies of the chunk's row 0 (dropped after the call), sharded
over the host devices and exported through ``cached_export`` as ``leaves._run`` does, so
one program compiles per player (its ``name`` is part of the export's name and static
key) and the odds depend neither on the device count nor on how many lines are passed.
``Player.name`` identifies the cached program, so every player needs a unique name (the
export's static key also holds ``act``'s and ``replace``'s ``__qualname__``).
Small chunks run faster than large ones (measured on Kaizo Roark, 20 devices: 200-800
rows about 250-280 playouts/s, 8000 rows 63/s).

``PlayerOdds.wins[k]`` counts ``deaths.success`` with ``k`` deaths allowed and the
state's ``protected`` set; ``distribution`` buckets every counted playout
(``deaths.bucket``).

A refused playout (``PlayerResult.refused``: it met a mechanic the engine deliberately
does not model) has no result and is never counted as a win or a loss (spec §4). With
``refused="raise"`` (the default) any refused row raises ``UnsupportedAi``; with
``"exclude"`` each line's refused rows are left out: ``PlayerOdds.n`` counts the rest,
every rate and interval is over it, ``refused`` counts those left out and ``refused_key``
is the first one's key (for ``player.refusal_reason``). A line whose playouts are all
refused raises ``UnsupportedAi`` either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from nuzline.ai.layout import AiLayout, layout_of
from nuzline.ai.policy import UnsupportedAi
from nuzline.data.tables import Tables
from nuzline.engine.state import MAX_DEATH_BUDGET, BattleState
from nuzline.jaxcache import cached_export, code_version
from nuzline.search import deaths
from nuzline.search.line import LINE, Line
from nuzline.search.player import Player, PlayerResult, player_playout
from nuzline.search.stage_a import wilson  # also registers Tables/BattleState

__all__ = [
    "ROW_CHUNK",
    "PlayerOdds",
    "evaluate_lines",
    "evaluate_players",
    "line_keys",
    "stack_lines",
]

ROW_CHUNK = 512  # rows per program call, before rounding up to the device count

for _cls in (Line, PlayerResult):
    jax.export.register_namedtuple_serialization(_cls, serialized_name=f"nuzline.{_cls.__name__}")


@dataclass(frozen=True)
class PlayerOdds:
    """One line's results over its ``n`` counted playouts (module docstring)."""

    n: int  # playouts counted (refused ones left out)
    # playouts won with at most k deaths, k = 0..3 (no protected loss)
    wins: tuple[int, int, int, int]
    wins_any: int  # playouts won
    deaths_total: int  # player faints summed over the n playouts
    distribution: deaths.DeathDistribution
    mean_turns: float
    fallback_share: float  # fallback turns / turns
    refused: int = 0  # refused playouts left out (``refused="exclude"``)
    # the first refused playout's key (``None``: none refused), for ``player.refusal_reason``
    refused_key: np.ndarray | None = field(default=None, compare=False)

    def rung(self, k: int) -> float:
        """P(win with at most ``k`` deaths, no protected loss): ``wins[k] / n``."""
        return self.wins[k] / self.n

    def interval(self, k: int | None) -> tuple[float, float]:
        """The 95 % Wilson interval of ``rung(k)``; ``None``: of any win."""
        return wilson(self.wins_any if k is None else self.wins[k], self.n)

    def score(self, k: int) -> tuple[int, int, int]:
        """The sort key at budget ``k`` (higher is better): rung ``k``'s wins, then any
        win, then fewer deaths."""
        return (self.wins[k], self.wins_any, -self.deaths_total)


def line_keys(key, n: int):
    """The ``n`` playout keys every line shares: ``jax.random.split(key, n)``."""
    return jax.random.split(key, n)


def _stack(params: Sequence):
    """``params`` (one pytree shape) stacked on a leading axis, numpy; a leafless pytree
    stays leafless."""
    if not params:
        raise ValueError("no params to evaluate")
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *params)


def stack_lines(lines: Sequence[Line]) -> Line:
    """``lines`` as one ``Line`` with a leading axis ``L`` (numpy)."""
    return _stack(lines)


def _rows_program(
    tables: Tables,
    state: BattleState,
    params,
    keys,
    player: Player,
    max_turns: int,
    layout: AiLayout,
) -> PlayerResult:
    """``player_playout`` from the one ``state`` for every row's ``params`` and key."""
    return jax.vmap(lambda p, k: player_playout(tables, state, player, p, k, max_turns, layout))(
        params, keys
    )


def _play_rows(tables, state, player: Player, params, keys, line_of, max_turns: int):
    """The ``PlayerResult`` (numpy) of every row: row ``r`` plays ``params[line_of[r]]``
    with ``keys[r]``, chunked, padded and sharded as the module docstring says."""
    rows = len(line_of)
    devices = jax.device_count()
    size = -(-ROW_CHUNK // devices) * devices
    sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
    layout = layout_of(tables)  # concrete here; the program takes it as static
    program = partial(_rows_program, player=player, max_turns=max_turns, layout=layout)
    parts = []
    for start in range(0, rows, size):
        idx = np.arange(start, min(start + size, rows))
        pad = np.concatenate([idx, np.full(size - len(idx), idx[0])])
        chunk = (jax.tree.map(lambda f, p=pad: f[line_of[p]], params), keys[pad])
        chunk = jax.device_put(chunk, sharding) if devices > 1 else jax.device_put(chunk)
        args = (tables, state, *chunk)
        exported = cached_export(
            program,
            args,
            name=f"player_playouts_{player.name}",
            static={
                "player": player.name,
                "act": player.act.__qualname__,
                "replace": player.replace.__qualname__,
                "max_turns": max_turns,
                "layout": layout.name,
            },
            version=code_version(),
        )
        out = jax.device_get(exported(*args))
        kept = len(idx)
        parts.append(jax.tree.map(lambda x, m=kept: np.asarray(x)[:m], out))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *parts)


def _odds(out: PlayerResult, protected: np.ndarray, refused: int, refused_key) -> PlayerOdds:
    won = out.won == 1
    faints = out.player_faints.astype(np.int64)
    wins = tuple(
        int(np.sum(deaths.success(won, faints, out.deaths, k, protected)))
        for k in range(MAX_DEATH_BUDGET + 1)
    )
    buckets = deaths.bucket(won, out.ended, faints, out.deaths, protected)
    turns = out.turns.astype(np.int64)
    return PlayerOdds(
        n=len(won),
        wins=wins,
        wins_any=int(won.sum()),
        deaths_total=int(faints.sum()),
        distribution=deaths.DeathDistribution.from_counts(
            np.bincount(buckets, minlength=len(deaths.BUCKETS))
        ),
        mean_turns=float(turns.mean()),
        fallback_share=float(out.fallback_turns.astype(np.int64).sum() / max(turns.sum(), 1)),
        refused=refused,
        refused_key=refused_key,
    )


def evaluate_players(
    tables: Tables,
    state: BattleState,
    player: Player,
    params: Sequence,
    n: int,
    key,
    max_turns: int = 100,
    refused: str = "raise",
) -> list[PlayerOdds]:
    """One ``PlayerOdds`` per ``params`` entry of ``player``, each from ``n`` playouts
    from ``state`` on ``line_keys(key, n)`` (module docstring). For ``GREEDY`` pass
    ``params=[()]``. ``refused``: ``"raise"`` raises ``UnsupportedAi`` when any playout
    is refused, ``"exclude"`` leaves refused playouts out of each entry's odds (an entry
    whose playouts are all refused still raises)."""
    if n <= 0:
        raise ValueError("evaluate_players needs n > 0")
    if refused not in ("raise", "exclude"):
        raise ValueError(f"refused must be 'raise' or 'exclude', not {refused!r}")
    stacked = _stack(params)
    lines = len(params)
    state = jax.tree.map(np.asarray, state)
    keys = np.asarray(line_keys(key, n))
    line_of = np.repeat(np.arange(lines), n)
    out = _play_rows(tables, state, player, stacked, np.tile(keys, (lines, 1)), line_of, max_turns)
    bad = out.refused != 0
    if refused == "raise" and bad.any():
        raise UnsupportedAi(
            f"{int(bad.sum())} of {lines * n} line playouts refused; "
            "run `nuzline coverage` for the fight"
        )
    protected = np.asarray(state.protected)
    odds = []
    for i in range(lines):
        left_out = bad[i * n : (i + 1) * n]
        if left_out.all():
            raise UnsupportedAi(
                f"every playout of entry {i} ({n}) refused; run `nuzline coverage` for the fight"
            )
        kept = jax.tree.map(lambda x, i=i, m=~left_out: x[i * n : (i + 1) * n][m], out)
        first = keys[int(np.argmax(left_out))].copy() if left_out.any() else None
        odds.append(_odds(kept, protected, int(left_out.sum()), first))
    return odds


def evaluate_lines(
    tables: Tables,
    state: BattleState,
    lines: Sequence[Line],
    n: int,
    key,
    max_turns: int = 100,
    refused: str = "raise",
) -> list[PlayerOdds]:
    """``evaluate_players`` with ``LINE``: one ``PlayerOdds`` per line."""
    return evaluate_players(tables, state, LINE, lines, n, key, max_turns, refused)
