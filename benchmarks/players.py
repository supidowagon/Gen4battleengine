"""The player scoreboard (L1 spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md
section 7a): every registered simulated player on the fixed fights of
``benchmarks/fights.toml``, all on the same keys, so later players compare by number.

    uv run python -m benchmarks.players [--n 2000] [--seed 0] [--devices N] [--fight NAME]

Rules:

- A fight (``[[fight]]``) names its save by a key of ``data/local.toml`` (``tools.rom.inputs``),
  never by a path; the save is never copied. A fight whose save is not configured (or missing)
  is skipped with one line saying so. The fight is ``nuzline line``'s (``cli.line_setup``): the
  ``team`` file's ``[[team]]`` from the save, the trainer, ``deaths`` as the death budget.
- An ``Entry`` is one row: a ``Player`` and ``params(ctx)``, its params pytree for the fight
  (``FightContext``), or ``None`` when it does not apply there (the row is left out). Adding a
  later player is one ``Entry`` in ``entries()``. Registered: ``greedy`` (``GREEDY``, ``()``),
  ``user line`` (``LINE``, the team file's rules) and ``found line`` (``LINE``, the rules of the
  fight's ``found`` file, which must hold the same ``[[team]]``; ``None`` while the file does
  not exist). A found file whose ``#`` header names a search ``deaths N`` other than the
  fight's ``deaths`` is scored with a warning line (it was searched for another budget).
- Every entry of a fight plays ``evaluate_players(..., refused="exclude")`` on the same ``n``
  keys, ``line_keys(jax.random.PRNGKey(seed), n)`` (common random numbers), at most
  ``cli.LINE_ODDS_MAX_TURNS`` turns. Refused playouts (a mechanic the engine does not model)
  are left out and counted; a row prints that count when it is above 0. An entry whose
  playouts are all refused gets no row, one line naming the mechanic, and the run goes on.
- Printed per fight, one row per entry: win % and win with at most k deaths for k = 0..3, each
  with its 95 % Wilson interval, the default player's share of turns. A bad ``fights.toml`` or
  line file, or a found file whose ``[[team]]`` differs, is one ``error:`` line and exit 1.
  Written:
  ``benchmarks/results/players_<time>.json`` (git-ignored).

``nuzline.cli`` and this module import no JAX at import time (``--devices`` must be set
first), so ``ENTRIES`` (``entries()``) is built on first access.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks._common import commit, dirty, say
from nuzline import cli

if TYPE_CHECKING:
    from nuzline.search.player import Player

BENCH = Path(__file__).resolve().parent
FIGHTS = BENCH / "fights.toml"
RESULTS = BENCH / "results"
_FIGHT_KEYS = ("name", "game", "save", "trainer_id", "team", "deaths", "found")


@dataclass(frozen=True)
class Fight:
    """One ``[[fight]]`` of ``fights.toml`` (paths relative to ``benchmarks/``)."""

    name: str
    game: str
    save: str  # a key of data/local.toml
    trainer_id: int
    team: str
    deaths: int
    found: str | None = None


@dataclass(frozen=True)
class FightContext:
    """A fight set up to play: its ``Fight`` and ``cli.LineSetup``."""

    fight: Fight
    setup: cli.LineSetup


@dataclass(frozen=True)
class Entry:
    """One scoreboard row: ``player`` with ``params(ctx)`` (``None``: not applicable)."""

    label: str
    player: Player
    params: Callable


def read_fights(path: Path = FIGHTS) -> list[Fight]:
    """The ``[[fight]]`` tables of ``path``; ``ValueError`` for an unknown or missing key."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    fights = []
    for i, table in enumerate(data.get("fight", [])):
        extra = sorted(set(table) - set(_FIGHT_KEYS))
        missing = sorted(set(_FIGHT_KEYS[:-1]) - set(table))
        if extra or missing:
            raise ValueError(f"{path}: fight {i}: unknown {extra}, missing {missing}")
        fights.append(Fight(**table))
    return fights


def fight_context(fight: Fight, save: Path) -> FightContext:
    """``fight`` set up from ``save`` (``cli.line_setup``)."""
    setup = cli.line_setup(
        fight.game, fight.trainer_id, BENCH / fight.team, save_path=save, deaths=fight.deaths
    )
    return FightContext(fight, setup)


def _found_line(ctx: FightContext):
    """The ``Line`` of the fight's ``found`` file; ``None`` without one."""
    from nuzline.search.line_text import read_line_file, resolve_rules, rules_to_line

    if ctx.fight.found is None or not (BENCH / ctx.fight.found).exists():
        return None
    file = read_line_file(BENCH / ctx.fight.found)
    if file.team != ctx.setup.file.team:
        raise ValueError(f"{ctx.fight.found}: [[team]] differs from {ctx.fight.team}'s")
    s = ctx.setup
    return rules_to_line(resolve_rules(file, s.names, s.player, s.opp))


def found_deaths(path: Path) -> int | None:
    """The ``deaths N`` its first ``#`` line names (``line_roark``'s header); ``None``
    without one."""
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        if text.startswith("#"):
            match = re.search(r"\bdeaths (\d+)\b", text)
            return int(match.group(1)) if match else None
    return None


def entries() -> list[Entry]:
    """The registered players, one ``Entry`` each (module docstring)."""
    from nuzline.search.line import LINE
    from nuzline.search.player import GREEDY

    return [
        Entry("greedy", GREEDY, lambda ctx: ()),
        Entry("user line", LINE, lambda ctx: ctx.setup.line),
        Entry("found line", LINE, _found_line),
    ]


def __getattr__(name: str):
    if name == "ENTRIES":  # built on first access: importing the players imports JAX
        globals()["ENTRIES"] = registered = entries()
        return registered
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def scoreboard(
    fights: Sequence[FightContext],
    entries: Sequence[Entry],
    n: int,
    seed: int,
    report: Callable[[str], None] = say,
) -> list[dict]:
    """One row per fight and applicable entry: ``fight``, ``label`` and the ``PlayerOdds``
    fields but ``refused_key``; every entry on ``line_keys(PRNGKey(seed), n)``. An entry
    whose playouts are all refused gets no row and one ``report`` line instead."""
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.search import player, player_eval

    key = jax.random.PRNGKey(seed)
    rows = []
    for ctx in fights:
        s = ctx.setup
        for entry in entries:
            params = entry.params(ctx)
            if params is None:
                continue
            try:
                (odds,) = player_eval.evaluate_players(
                    s.tables,
                    s.state,
                    entry.player,
                    [params],
                    n,
                    key,
                    cli.LINE_ODDS_MAX_TURNS,
                    refused="exclude",
                )
            except UnsupportedAi as e:
                first = player_eval.line_keys(key, n)[0]
                reason = player.refusal_reason(
                    s.tables, s.state, entry.player, params, first, cli.LINE_ODDS_MAX_TURNS
                )
                report(f"{entry.label}: all playouts refused: {reason or e}")
                continue
            fields = {
                f.name: getattr(odds, f.name)
                for f in dataclasses.fields(odds)
                if f.name != "refused_key"
            }
            rows.append({"fight": ctx.fight.name, "label": entry.label, **fields})
    return rows


CELL = 19  # "100.0 [100.0,100.0]"


def _cell(wins: int, n: int) -> str:
    from nuzline.search.stage_a import wilson

    low, high = wilson(wins, n)
    return f"{100 * wins / n:5.1f} [{100 * low:4.1f},{100 * high:4.1f}]"


def _table(rows: Sequence[dict]) -> list[str]:
    """The rows of one fight: label, win any, win with <=k deaths (k = 0..3), each with its
    95 % Wilson interval, the default player's share of turns, refused playouts if any."""
    titles = ["win, any"] + [f"<={k} deaths" for k in range(cli.MAX_DEATHS + 1)]
    out = [f"{'player':<10} " + " ".join(f"{t:<{CELL}}" for t in titles) + " default"]
    for r in rows:
        cells = [_cell(r["wins_any"], r["n"])] + [_cell(w, r["n"]) for w in r["wins"]]
        text = f"{r['label']:<10} " + " ".join(f"{c:<{CELL}}" for c in cells)
        text += f" {100 * r['fallback_share']:5.1f} %"
        if r["refused"]:
            text += f"  ({r['refused']} refused playouts left out of {r['n'] + r['refused']})"
        out.append(text)
    return out


def _row_json(row: dict) -> dict:
    out = dict(row)
    out["distribution"] = list(row["distribution"].probs)
    out["wins"] = list(row["wins"])
    return out


def run(args) -> int:
    import jax

    from benchmarks.rank_timing import machine
    from nuzline.data.games import get_game
    from nuzline.engine.coverage import UnsupportedMechanics
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from tools.rom.inputs import RomInputError, load_inputs

    fights = read_fights()
    if args.fight is not None:
        fights = [f for f in fights if f.name == args.fight]
        if not fights:
            say(f"no fight named {args.fight!r} in {FIGHTS}")
            return 1
    try:
        inputs = load_inputs()
    except RomInputError:
        inputs = None
    registered = entries()
    started = time.perf_counter()
    rows, ran = [], []
    for fight in fights:
        save = getattr(inputs, fight.save, None)
        if save is None or not save.exists():
            say(f"{fight.name}: skipped, save {fight.save!r} is not configured (data/local.toml)")
            continue
        enable_compile_cache(
            table_version(get_game(fight.game).tables_dir) + code_version(), fight.game
        )
        try:
            ctx = fight_context(fight, save)
        except (cli.FightRefused, UnsupportedMechanics) as e:
            say(f"{fight.name}: refused: {e}")
            continue
        say("")
        say(f"== {fight.name}: {cli.trainer_line(ctx.setup.trainer)}, <={fight.deaths} deaths")
        for line in cli.team_lines(ctx.setup):
            say(line)
        found = None if fight.found is None else BENCH / fight.found
        if found is not None and not found.exists():
            say(f"found line: {fight.found} does not exist yet; left out")
        elif found is not None and found_deaths(found) not in (None, fight.deaths):
            say(
                f"warning: {fight.found} was searched with deaths {found_deaths(found)}, "
                f"this fight allows {fight.deaths}"
            )
        fight_start = time.perf_counter()
        fight_rows = scoreboard([ctx], registered, args.n, args.seed)
        say(
            f"{args.n} playouts per player, seed {args.seed} (the same keys for every player); "
            f"{time.perf_counter() - fight_start:.1f} s"
        )
        for line in _table(fight_rows):
            say(line)
        rows += fight_rows
        ran.append(dataclasses.asdict(fight))
    total_s = time.perf_counter() - started
    say("")
    say(f"total: {total_s:.1f} s, {jax.device_count()} devices")
    if not rows:
        return 1
    RESULTS.mkdir(exist_ok=True)
    now = datetime.now()
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "commit": commit(),
        "dirty": dirty(),
        "command": "python -m benchmarks.players " + " ".join(sys.argv[1:]),
        "n": args.n,
        "seed": args.seed,
        "devices": jax.device_count(),
        "max_turns": cli.LINE_ODDS_MAX_TURNS,
        "fights": ran,
        "rows": [_row_json(r) for r in rows],
        "total_s": round(total_s, 2),
        "machine": machine(),
    }
    out = RESULTS / f"players_{now:%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    say(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score every registered player on the fights of benchmarks/fights.toml."
    )
    parser.add_argument("--n", type=int, default=2000, help="playouts per player and fight")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--devices", type=int, default=os.cpu_count(), help="host CPU devices")
    parser.add_argument("--fight", help="only the fight with this name")
    args = parser.parse_args(argv)
    if args.n < 1 or args.devices < 1:
        parser.error("--n and --devices must be at least 1")
    cli.configure_cpu_devices(args.devices)  # before jax is imported (run imports it)
    try:
        return run(args)
    except ValueError as e:  # a bad fights.toml or line file, a [[team]] mismatch
        say(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
