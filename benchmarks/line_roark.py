"""Kaizo Roark line benchmark (L1 spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md
section 7): the user's line, the line the search finds and the empty line, scored together.

    uv run python -m benchmarks.line_roark [--n 4000] [--deaths 1] [--seed 0] [--devices N]
        [--budget 1800]

The fight is ``nuzline line``'s (``cli.line_setup``): Kaizo Leader Roark (trainer id 246),
the team of ``benchmarks/lines/kaizo_roark_user.toml`` from the pre-Roark save (its path from
``tools.rom.inputs``, ``kaizo_save_pre_roark``; the save is never copied; exit 1 without it),
``--deaths K``. Then:

1. ``find_line`` from the EMPTY line (the user's rules are not its start) with the default
   ``LineSearchSettings`` but ``budget_seconds=--budget`` and ``final_n=--n``.
2. One ``evaluate_lines(..., refused="exclude")`` call scores the user's line, the found line
   and the empty line on the same ``--n`` hold-out keys, ``fold_in(PRNGKey(seed),
   line_search.HOLD_OUT_SALT)``: keys no search batch used. With ``final_n = --n`` these are
   the keys of ``find_line``'s own hold-out, so the found line's odds must equal its
   ``result.odds`` (printed as a check).
3. Printed: each line with its ladder, 95 % Wilson intervals and refused playouts; found minus
   user at each rung and any win (``evaluate_lines`` returns no per-key results, so the two
   intervals are shown and no paired interval); the found rules as TOML; wall times and
   playouts. Written: ``benchmarks/results/line_roark_<time>.json`` (git-ignored; with the
   commit and whether tracked files were dirty) and
   ``benchmarks/lines/kaizo_roark_found_k<K>_s<seed>.toml`` (a ``#`` comment with the command,
   seed, deaths and commit, then the user file's team and the found rules).

When the search or the shared scoring raises ``UnsupportedAi`` (a line refused in every
playout), the benchmark prints ``refused:``,
writes the JSON with what is known so far (the search's result if it finished) and exits 2.
The defaults (``--n 4000``, ``--deaths 1``, ``--budget 1800``) are the controller's choice.

A heuristic search: the found line is a good line, not a proven best.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from nuzline import cli
from nuzline.text import ascii_text

ROOT = Path(__file__).resolve().parent.parent
LINES = Path(__file__).resolve().parent / "lines"
USER_LINE = LINES / "kaizo_roark_user.toml"
RESULTS = Path(__file__).resolve().parent / "results"
GAME, ROARK_ID = "kaizo", 246  # Kaizo Leader Roark


def _say(text: str) -> None:
    print(ascii_text(text), flush=True)


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _dirty() -> bool | None:
    """Whether tracked files differ from the commit (``None``: git failed)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def _write_record(record: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"line_roark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _say(f"wrote {out}")
    return out


def _odds_json(odds) -> dict:
    rungs = {
        f"<={k}": {"p": odds.rung(k), "ci95": list(odds.interval(k))}
        for k in range(cli.MAX_DEATHS + 1)
    }
    return {
        "n": odds.n,
        "refused": odds.refused,
        "wins": list(odds.wins),
        "wins_any": odds.wins_any,
        "rungs": rungs,
        "any_win": {"p": odds.wins_any / odds.n, "ci95": list(odds.interval(None))},
        "deaths_total": odds.deaths_total,
        "mean_turns": odds.mean_turns,
        "fallback_share": odds.fallback_share,
    }


def _difference_lines(user, found) -> list[str]:
    """Found minus user at each rung and any win, with both Wilson intervals."""
    out = ["found - user on the shared keys (paired interval not available: evaluate_lines"]
    out.append("returns no per-key results; the two 95 % Wilson intervals are shown):")
    for k in (*range(cli.MAX_DEATHS + 1), None):
        label = "win, any" if k is None else "win, 0 deaths" if k == 0 else f"win, <={k} deaths"
        pu = user.wins_any / user.n if k is None else user.rung(k)
        pf = found.wins_any / found.n if k is None else found.rung(k)
        (ul, uh), (fl, fh) = user.interval(k), found.interval(k)
        out.append(
            f"  {label:<16} {100 * (pf - pu):+6.1f} pp  (found [{100 * fl:4.1f}, "
            f"{100 * fh:4.1f}], user [{100 * ul:4.1f}, {100 * uh:4.1f}])"
        )
    return out


def run(args) -> int:
    import jax

    from benchmarks.rank_timing import machine
    from nuzline.ai.policy import UnsupportedAi
    from nuzline.data.games import get_game
    from nuzline.engine.coverage import UnsupportedMechanics
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from nuzline.search import line_search
    from nuzline.search.line import empty_line
    from nuzline.search.line_text import line_to_rules, render_rules, write_line_file
    from nuzline.search.player_eval import evaluate_lines
    from tools.rom.inputs import RomInputError, load_inputs

    try:
        save = load_inputs().kaizo_save_pre_roark
    except RomInputError as e:
        save, why = None, str(e)
    else:
        why = "kaizo_save_pre_roark is not configured (data/local.toml)"
    if save is None or not save.exists():
        print(f"line_roark: no pre-Roark save: {why if save is None else save}", file=sys.stderr)
        return 1
    tables_version = table_version(get_game(GAME).tables_dir)
    enable_compile_cache(tables_version + code_version(), GAME)
    started = time.perf_counter()
    try:
        setup = cli.line_setup(GAME, ROARK_ID, USER_LINE, save_path=save, deaths=args.deaths)
    except (cli.FightRefused, UnsupportedMechanics) as e:
        _say(f"refused: {e}")
        cli.print_report(e.report)
        return 2
    _say(cli.trainer_line(setup.trainer))
    for line in cli.team_lines(setup):
        _say(line)

    settings = line_search.LineSearchSettings(budget_seconds=args.budget, final_n=args.n)
    key = jax.random.PRNGKey(args.seed)
    commit = _commit()
    command = "python -m benchmarks.line_roark " + " ".join(sys.argv[1:])
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": _dirty(),
        "command": command,
        "game": GAME,
        "trainer_id": ROARK_ID,
        "table_version": tables_version,
        "deaths": args.deaths,
        "seed": args.seed,
        "n": args.n,
        "devices": jax.device_count(),
        "settings": dataclasses.asdict(settings),
        "machine": machine(),
    }

    def refused(e: UnsupportedAi) -> int:
        _say(f"refused: {e}")
        record["refused"] = str(e)
        record["total_s"] = round(time.perf_counter() - started, 2)
        _write_record(record)
        return 2

    def progress(text: str) -> None:
        print(ascii_text(f"[line] {text}"), file=sys.stderr, flush=True)

    search_start = time.perf_counter()
    try:
        result = line_search.find_line(
            setup.tables, setup.names, setup.state, key, settings, start=None, progress=progress
        )
    except UnsupportedAi as e:
        return refused(e)
    search_s = time.perf_counter() - search_start
    progress(
        f"stopped: {result.stopped} after {result.sweeps} sweep(s), {result.evaluations} playouts"
    )
    found_rules = line_to_rules(result.line)
    toml = render_rules(found_rules, setup.names, setup.player, setup.opp)
    record["search"] = {
        "wall_s": round(search_s, 2),
        "playouts": result.evaluations,
        "sweeps": result.sweeps,
        "stopped": result.stopped,
        "refused_candidates": result.refused_candidates,
        "hold_out": _odds_json(result.odds),
    }
    record["found_toml"] = toml

    lines = [setup.line, result.line, empty_line()]
    hold_out = jax.random.fold_in(key, line_search.HOLD_OUT_SALT)
    eval_start = time.perf_counter()
    try:
        user, found, empty = evaluate_lines(
            setup.tables,
            setup.state,
            lines,
            args.n,
            hold_out,
            settings.max_turns,
            refused="exclude",
        )
    except UnsupportedAi as e:
        return refused(e)
    eval_s = time.perf_counter() - eval_start
    total_s = time.perf_counter() - started

    shared = f"{args.n} hold-out keys of seed {args.seed} (fold_in HOLD_OUT_SALT)"
    blocks = (
        ("the user's line", setup.rules, user, setup.line),
        (
            "found by search (a heuristic: a good line, not a proven best)",
            found_rules,
            found,
            result.line,
        ),
        ("the empty line (the default player decides every turn)", [], empty, lines[2]),
    )
    for title, rules, odds, line in blocks:
        _say("")
        _say(f"== {title}; {shared}")
        for text in cli.line_report_lines(setup, rules, odds, args.seed, args.deaths):
            _say(text)
        left_out = cli.left_out_lines(setup, line, odds, settings.max_turns)
        for text in left_out or [f"refused playouts left out: {odds.refused}"]:
            _say(text)
    _say("")
    for text in _difference_lines(user, found):
        _say(text)
    same = (result.odds.wins, result.odds.wins_any, result.odds.n) == (
        found.wins,
        found.wins_any,
        found.n,
    )
    _say(f"check: find_line's own hold-out odds {'match' if same else 'DIFFER FROM'} these")
    if result.refused_candidates:
        _say(
            f"search skipped {result.refused_candidates} candidate lines refused in more "
            "than 1 % of their playouts"
        )

    _say("")
    _say("found line (TOML):")
    _say("\n\n".join(toml) if toml else "# no rules: the default player decides every turn")
    playouts = result.evaluations + len(lines) * args.n
    _say("")
    _say(
        f"search: {search_s:.1f} s, {result.evaluations} playouts, {result.sweeps} sweep(s), "
        f"stopped: {result.stopped}"
    )
    _say(f"shared scoring: {eval_s:.1f} s, {len(lines) * args.n} playouts")
    _say(f"total: {total_s:.1f} s, {playouts} playouts, {jax.device_count()} devices")

    found_path = LINES / f"kaizo_roark_found_k{args.deaths}_s{args.seed}.toml"
    header = f"{command}; seed {args.seed}, deaths {args.deaths}, commit {commit}"
    write_line_file(found_path, setup.file, toml, header=header)
    _say(f"wrote {found_path}")
    record["search"]["hold_out_matches_shared"] = same
    record.update(
        {
            "shared_scoring_s": round(eval_s, 2),
            "total_s": round(total_s, 2),
            "playouts": playouts,
            "found_file": found_path.relative_to(ROOT).as_posix(),
            "lines": {
                "user": {
                    "rules": [dataclasses.asdict(r) for r in setup.rules],
                    **_odds_json(user),
                },
                "found": {
                    "rules": [dataclasses.asdict(r) for r in found_rules],
                    **_odds_json(found),
                },
                "empty": {"rules": [], **_odds_json(empty)},
            },
        }
    )
    _write_record(record)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=4000, help="shared hold-out playouts")
    parser.add_argument("--deaths", type=int, default=1, help="player faints allowed, 0..3")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--devices", type=int, default=os.cpu_count(), help="host CPU devices")
    parser.add_argument(
        "--budget", type=float, default=1800, help="search seconds after which no race starts"
    )
    args = parser.parse_args(argv)
    if args.n < 1 or args.devices < 1 or args.budget < 0:
        parser.error("--n and --devices must be at least 1, --budget not negative")
    if not 0 <= args.deaths <= cli.MAX_DEATHS:
        parser.error(f"--deaths must be between 0 and {cli.MAX_DEATHS}")
    cli.configure_cpu_devices(args.devices)  # before jax is imported (run imports it)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
