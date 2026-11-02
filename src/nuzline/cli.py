"""The ``nuzline`` command line (docs/design.md §4.3, plan M1b Task 8).

* ``nuzline coverage --party P (--trainer NAME | --trainer-id N) [--lead 0]
  [--opp-active 0]``: the fight's ``CoverageReport`` for the same turn-0 state ``rank``
  would use; exit code 2 when anything is missing. With ``--save`` it adds ``[--cap N]`` and
  a second report for the candidate pool (``prep.pool``: every party and box Pokemon at
  level ``N``, default the highest party level, with its current, Move Relearner and TM
  moves); the exit code follows the party fight alone (plan 4a ruling 7), and ``--cap``
  without ``--save`` is an error.
* ``nuzline rank --party P (--trainer NAME | --trainer-id N) [--n 2000] [--seed 0]
  [--lead 0] [--opp-active 0] [--devices N] [--deaths K] [--protect NAME ...]``:
  ``require_coverage``, then the Stage A ranking of the player's actions at the fight's
  turn-0 state (the trainer's real built party, bag and start condition; party slots
  ``--lead`` and ``--opp-active`` active). ``--deaths K`` (0..3, default 0) ranks by
  P(win with at most K player faints) and lists every rung of the ladder; ``--protect``
  (repeatable) names a Pokemon by species or 0-based party index (like ``--lead``): a line
  where it faints fails on every rung. Exit code 2 when the fight is refused, by the
  coverage check or by a refused playout.
* ``nuzline solve --party P (--trainer NAME | --trainer-id N) [--lead 0] [--opp-active 0]
  [--max-depth 2] [--budget SECONDS] [--precision PP | --playouts N] [--epsilon 1e-4]
  [--seed 0] [--devices N] [--tree FILE.json] [--show-depth 1] [--deaths K]
  [--protect NAME ...]``: the Stage B plan
  (``search.stage_b``): the ranking with intervals and bounds, the plan tree under the best
  action, whether the precision target was met, optionally the whole tree as JSON, and a run
  record under ``runs/`` (party hash, trainer, settings, seed, result). Each pass tops up its
  leaf playouts until the leader's half-width is at most ``--precision`` percentage points
  (default 1.5); ``--playouts N`` instead fixes every pass's leaf budget (no top-ups). Exit
  code 2 when the fight or the opponent AI is refused, or a belief node's states differ in
  the player's legal actions (``stage_b.HiddenLegality``).
* ``nuzline prep --save SAVE (--trainer NAME | --trainer-id N) [--cap N] [--deaths K]
  [--protect NAME ...] [--exclude NAME ...] [--reserve ITEM ...] [--prestatus STATUS ...]
  [--no-entry-hp] [--weight KIND=W ...] [--budget SECONDS] [--checkpoint FILE | --resume FILE]
  [--max-depth 2] [--finalists 3] [--screen-n 8] [--seed 0] [--devices N] [--tree FILE.json]
  [--show-depth 1]``: the prep search (``prep.search``): which Pokemon, moves, items and entry
  HP to bring from the save's party and boxes, the costed prep steps, the odds ladder for
  0-3 deaths and the Stage B plan, a run record ``runs/prep_<time>.json`` and a checkpoint
  (default ``runs/prep_<time>.ckpt.json``) that ``--resume`` continues. ``--cap`` defaults to
  ``config/level_caps.toml``. ``--protect`` and ``--exclude`` name a Pokemon by species,
  nickname, ``party:N`` or ``box:B:S`` (the labels of ``save show``). Exit code 2 when the
  opponent is unsupported or every setup is refused; an interrupted or budget-stopped run
  exits 0 with its best answer.
* ``nuzline trainers``: every trainer's id, name and class.
* ``nuzline save show SAVE [--json] [--allow-corrupt]``: the save's trainer, loaded copy,
  party and boxes (one line per Pokemon), or the whole save as JSON.
* ``nuzline save export SAVE -o PARTY.toml [--allow-corrupt]``: the save's party (eggs
  dropped) as a ``--party`` TOML file.

``coverage``, ``rank`` and ``solve`` take ``--save SAVE [--allow-corrupt]`` instead of
``--party``: the party is the save's, eggs left out (each reported on stderr; ``--lead``
indexes the egg-free party), and the solve run record holds ``save_sha256``. A bad save
or record is one line on stderr and exit code 1.

``--trainer NAME`` is the first trainer whose name matches case-insensitively and whose
class starts with "leader", else the first match; ``--trainer-id`` selects exactly
(plan ruling). All output is plain ASCII (controller ruling: a default Windows console
is cp1252).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from nuzline.data.trainer import Trainer

INTERMEDIATE = Path(__file__).resolve().parents[2] / "data" / "intermediate"
RUNS = Path(__file__).resolve().parents[2] / "runs"

DEFAULT_PRECISION_PP = 1.5  # nuzline solve's precision target, percentage points

ALL_ZERO_FOOTER = "all lines 0 %: no zero-faint line found at this depth"
MAX_DEATHS = 3  # nuzline.engine.state.MAX_DEATH_BUDGET; this module imports no jax


def all_zero_footer(k: int) -> str:
    """The footer when every line scores 0 at budget ``k``."""
    if k == 0:
        return ALL_ZERO_FOOTER
    return f"all lines 0 %: no line within {k} deaths found at this depth"


def protected_slots(player: Sequence, names, values: Sequence[str]) -> tuple[int, ...]:
    """The 0-based party indices ``--protect`` names (sorted, deduplicated): a species name
    (case-insensitive; two party members of it are ambiguous) or digits, a 0-based index
    into ``player`` like ``--lead``. Raises ``ValueError`` naming the flag and the value
    for an unknown or ambiguous name, an index outside the party or a fainted Pokemon."""
    out: set[int] = set()
    for value in values:
        text = value.strip()
        if text.isdigit():
            slot = int(text)
            if slot >= len(player):
                raise ValueError(
                    f"--protect {value}: no party index {slot} (the party has {len(player)})"
                )
        else:
            matches = [
                i
                for i, spec in enumerate(player)
                if names.species_names[spec.species].lower() == text.lower()
            ]
            if not matches:
                raise ValueError(f"--protect {value}: no such Pokemon in the party")
            if len(matches) > 1:
                raise ValueError(
                    f"--protect {value}: ambiguous, party indices {matches} are all {value}"
                )
            slot = matches[0]
        if player[slot].current_hp == 0:
            raise ValueError(f"--protect {value}: that Pokemon has fainted")
        out.add(slot)
    return tuple(sorted(out))


def _check_deaths(args) -> None:
    if not 0 <= args.deaths <= MAX_DEATHS:
        raise ValueError(f"--deaths {args.deaths}: must be between 0 and {MAX_DEATHS}")


EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def resolve_trainer(
    trainers: Sequence[Trainer], *, name: str | None = None, trainer_id: int | None = None
) -> Trainer:
    """The trainer ``--trainer``/``--trainer-id`` names (module docstring); raises
    ``ValueError`` when there is none."""
    if trainer_id is not None:
        for t in trainers:
            if t.id == trainer_id:
                return t
        raise ValueError(f"no trainer with id {trainer_id}")
    matches = [t for t in trainers if t.name.lower() == (name or "").lower()]
    if not matches:
        raise ValueError(f"no trainer named {name!r} (see `nuzline trainers`)")
    leaders = [t for t in matches if t.trainer_class.lower().startswith("leader")]
    return (leaders or matches)[0]


_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def configure_cpu_devices(count: int | None, environ=os.environ) -> None:
    """Make XLA expose ``count`` (default ``os.cpu_count()``) host CPU devices by
    appending the device-count flag to ``environ["XLA_FLAGS"]``, keeping any existing
    flags. A user-set device-count flag wins. Must run before ``jax`` is imported."""
    flags = environ.get("XLA_FLAGS", "")
    if _DEVICE_FLAG in flags:
        return
    flag = f"{_DEVICE_FLAG}={count or os.cpu_count() or 1}"
    environ["XLA_FLAGS"] = f"{flags} {flag}".strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nuzline", description="Nuzlocke line finder for Pokemon Platinum battles."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def fight(cmd: argparse.ArgumentParser) -> None:
        source = cmd.add_mutually_exclusive_group(required=True)
        source.add_argument("--party", type=Path, help="the player's party TOML")
        source.add_argument("--save", type=Path, help="a Platinum .sav/.dsv: its party is used")
        cmd.add_argument(
            "--allow-corrupt", action="store_true", help="with --save: load a corrupt save anyway"
        )
        who = cmd.add_mutually_exclusive_group(required=True)
        who.add_argument("--trainer", help="trainer name (a Leader is preferred)")
        who.add_argument("--trainer-id", type=int, help="exact trainer id")
        cmd.add_argument("--lead", type=int, default=0, help="the player's active party slot")
        cmd.add_argument("--opp-active", type=int, default=0, help="the opponent's active slot")

    coverage = sub.add_parser("coverage", help="check that the engine supports a fight")
    fight(coverage)
    coverage.add_argument("--cap", type=int, default=None, help="with --save: the pool's level cap")

    def budget(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--deaths",
            type=int,
            default=0,
            help="player faints allowed, 0..3 (default 0): maximise P(win with at most K deaths)",
        )
        cmd.add_argument(
            "--protect",
            action="append",
            default=[],
            metavar="NAME",
            help="a Pokemon that must not faint: species name or 0-based party index "
            "(like --lead); repeatable",
        )

    rank = sub.add_parser("rank", help="rank the player's actions at the fight's start")
    fight(rank)
    budget(rank)
    rank.add_argument("--n", type=int, default=2000, help="playouts per action")
    rank.add_argument("--seed", type=int, default=0, help="PRNG seed of the playouts")
    rank.add_argument(
        "--devices",
        type=int,
        default=os.cpu_count(),
        help="host CPU devices the playouts are sharded over (default: all cores; "
        "1 keeps the single-device path)",
    )
    solve = sub.add_parser("solve", help="solve the fight's start with the Stage B plan search")
    fight(solve)
    budget(solve)
    solve.add_argument("--max-depth", type=int, default=2, help="deepest pass (0 = Stage A)")
    solve.add_argument(
        "--budget", type=float, default=None, help="seconds after which no deeper pass starts"
    )
    solve.add_argument(
        "--playouts",
        type=int,
        default=None,
        help="fixed leaf playouts per pass, no precision target (default: 2000 per action "
        "in round 0, then top-ups to --precision)",
    )
    solve.add_argument(
        "--precision",
        type=float,
        default=None,
        help=f"target half-width of the leader's interval, percentage points (default "
        f"{DEFAULT_PRECISION_PP}; not with --playouts)",
    )
    solve.add_argument("--epsilon", type=float, default=1e-4, help="prune reach below this")
    solve.add_argument("--seed", type=int, default=0, help="PRNG seed of the playouts")
    solve.add_argument("--devices", type=int, default=os.cpu_count(), help="host CPU devices")
    solve.add_argument("--tree", type=Path, default=None, help="write the plan tree as JSON")
    solve.add_argument("--show-depth", type=int, default=1, help="plan levels printed")
    prep = sub.add_parser("prep", help="find the setup to bring from a save, with its odds")
    prep.add_argument("--save", type=Path, required=True, help="a Platinum .sav/.dsv")
    prep.add_argument("--allow-corrupt", action="store_true", help="load a corrupt save anyway")
    who = prep.add_mutually_exclusive_group(required=True)
    who.add_argument("--trainer", help="trainer name (a Leader is preferred)")
    who.add_argument("--trainer-id", type=int, help="exact trainer id")
    prep.add_argument(
        "--cap", type=int, default=None, help="level cap (default: config/level_caps.toml)"
    )
    prep.add_argument("--deaths", type=int, default=0, help="player faints allowed, 0..3")
    for flag, what in (("--protect", "must not faint"), ("--exclude", "is not brought")):
        prep.add_argument(
            flag,
            action="append",
            default=[],
            metavar="NAME",
            help=f"a Pokemon that {what}: species, nickname, party:N or box:B:S; repeatable",
        )
    prep.add_argument(
        "--reserve",
        action="extend",
        nargs="+",
        default=[],
        metavar="ITEM",
        help="bag items the search must not use",
    )
    prep.add_argument(
        "--prestatus",
        action="append",
        default=[],
        metavar="STATUS",
        help="a status you may inflict before the fight: poison, burn or paralysis; repeatable",
    )
    prep.add_argument(
        "--no-entry-hp", action="store_true", help="always enter at full HP (no HP variants)"
    )
    prep.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="KIND=W",
        help="cost weight of candy, heart_scale, tm, hm or held (tie-break); repeatable",
    )
    prep.add_argument(
        "--budget", type=float, default=None, help="seconds after which no new work starts"
    )
    prep.add_argument("--checkpoint", type=Path, default=None, help="checkpoint file to write")
    prep.add_argument("--resume", type=Path, default=None, help="continue from this checkpoint")
    prep.add_argument("--max-depth", type=int, default=2, help="deepest Stage B pass (0 = screen)")
    prep.add_argument("--finalists", type=int, default=3, help="setups solved by Stage B")
    prep.add_argument("--screen-n", type=int, default=8, help="playouts per setup in round 0")
    prep.add_argument("--seed", type=int, default=0, help="PRNG seed")
    prep.add_argument("--devices", type=int, default=os.cpu_count(), help="host CPU devices")
    prep.add_argument("--tree", type=Path, default=None, help="write the plan tree as JSON")
    prep.add_argument("--show-depth", type=int, default=1, help="plan levels printed")
    sub.add_parser("trainers", help="list trainer ids, names and classes")
    save = sub.add_parser("save", help="inspect or export a Platinum save")
    save_sub = save.add_subparsers(dest="save_command", required=True)
    show = save_sub.add_parser("show", help="print the save's trainer, party and boxes")
    export = save_sub.add_parser("export", help="write the save's party as a party TOML")
    for cmd in (show, export):
        cmd.add_argument("save_path", type=Path, metavar="SAVE", help="a .sav or .dsv file")
        cmd.add_argument("--allow-corrupt", action="store_true", help="load a corrupt save anyway")
    show.add_argument("--json", action="store_true", help="print the whole save as JSON")
    export.add_argument("-o", "--output", required=True, type=Path, help="party TOML to write")
    return parser


def _trainer_line(t: Trainer) -> str:
    return f"{t.name} ({t.trainer_class}, id {t.id})"


def _print_report(report) -> None:
    print(f"Support: {report.level.name}")
    for title, lines in (("Missing", report.missing), ("Sampling only", report.sampling_only)):
        if not lines:
            print(f"{title}: none")
            continue
        print(f"{title} ({len(lines)}):")
        for line in lines:
            print(f"  {line}")


def _fight(args):
    """(tables, names, trainer, player party, opponent party, start weather); raises
    ``ValueError`` when ``--lead`` names a fainted Pokemon."""
    from nuzline.data.party import load_party
    from nuzline.data.tables import load_names, load_tables
    from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
    from nuzline.engine.coverage import start_weather

    trainer = resolve_trainer(
        load_trainers(INTERMEDIATE), name=args.trainer, trainer_id=args.trainer_id
    )
    tables, names = load_tables(), load_names()
    if args.save is not None:
        from nuzline.saves import load_save

        save = load_save(args.save, allow_corrupt=args.allow_corrupt, tables=tables, names=names)
        player = []
        for slot, mon in enumerate(save.party):
            if mon.is_egg:
                print(f"party slot {slot}: egg left out", file=sys.stderr)
            else:
                player.append(mon.to_spec(tables))
        if not player:
            raise ValueError("the save's party has no Pokemon besides eggs")
    else:
        player = load_party(args.party, names)
    opp = build_trainer_party(trainer, IntermediateData(INTERMEDIATE))
    if 0 <= args.lead < len(player) and player[args.lead].current_hp == 0:
        raise ValueError(f"--lead {args.lead}: that Pokemon has fainted")
    return tables, names, trainer, player, opp, start_weather(trainer)


def _coverage(args) -> int:
    from nuzline.engine.coverage import check_coverage

    if args.cap is not None and args.save is None:
        raise ValueError("--cap needs --save")
    tables, names, trainer, player, opp, weather = _fight(args)
    print(_trainer_line(trainer))
    leads = (args.lead, args.opp_active)
    report = check_coverage(tables, names, player, trainer, opp, weather, leads)
    _print_report(report)
    if args.save is not None:
        _print_pool(args, tables, names, player, trainer)
    return EXIT_REFUSED if report.missing else EXIT_OK


def _print_pool(args, tables, names, player, trainer) -> None:
    """The pool section of ``coverage --save``; it never changes the exit code. The cap is
    ``--cap``, else the trainer's in ``config/level_caps.toml``, else the highest party level."""
    from nuzline.data.learnsets import load_learnsets
    from nuzline.prep.caps import default_cap
    from nuzline.prep.pool import check_pool, pool_candidates
    from nuzline.saves import load_save

    save = load_save(args.save, allow_corrupt=args.allow_corrupt, tables=tables, names=names)
    cap = args.cap
    if cap is None:
        try:
            cap = default_cap(trainer)
        except ValueError:
            cap = max(spec.level for spec in player)
    candidates, left_out = pool_candidates(save, tables, names, load_learnsets(), cap)
    print(f"Pool at level cap {cap} ({len(candidates)} Pokemon):")
    _print_report(check_pool(tables, names, candidates))
    for line in left_out:
        print(line)


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _rank(args) -> int:
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.data import tables as tables_module
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from nuzline.search.plan_tree import distribution_line
    from nuzline.search.stage_a import rank_actions

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    _check_deaths(args)
    # Before anything compiles: JAX opens the persistent cache at the first compile.
    enable_compile_cache(table_version(tables_module.DEFAULT) + code_version())
    tables, names, trainer, player, opp, weather = _fight(args)
    slots = protected_slots(player, names, args.protect)
    leads = (args.lead, args.opp_active)
    print(_trainer_line(trainer))
    try:
        require_coverage(tables, names, player, trainer, opp, weather, leads)
    except UnsupportedMechanics as e:
        _print_report(e.report)
        return EXIT_REFUSED
    state = make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)
    state = state.with_budget(args.deaths, slots)
    try:
        ranking = rank_actions(tables, names, state, args.n, jax.random.PRNGKey(args.seed))
    except UnsupportedAi as e:
        print(f"refused: {e}")
        return EXIT_REFUSED
    mine = names.species_names[player[args.lead].species]
    theirs = names.species_names[opp[args.opp_active].species]
    print(f"{mine} vs {theirs}: {args.n} playouts per action, seed {args.seed}")
    width = max(len("action"), *(len(r.label) for r in ranking))
    win = "P(win, 0 faints)" if args.deaths == 0 else f"P(win, <={args.deaths} deaths)"
    rungs = " ".join(f"{f'<={k}':>7}" for k in range(MAX_DEATHS + 1))
    print(
        f"     #  {'action':<{width}}  {win + ' [95% CI]':<27}  {'P(win)':>7}  E[deaths]  "
        f"{rungs} (screen)"
    )
    for i, r in enumerate(ranking, 1):
        mark = "*" if r.tied_with_best else " "
        odds = f"{_pct(r.score)} [{100 * r.ci_low:4.1f}, {100 * r.ci_high:5.1f}]"
        ladder = [_pct(v) for v in r.ladder] or ["-"] * (MAX_DEATHS + 1)
        cells = " ".join(f"{c:>7}" for c in ladder)
        print(
            f"  {mark} {i:2d}  {r.label:<{width}}  {odds:<27}  {_pct(r.p_win):>7}  "
            f"{r.e_faints:9.2f}  {cells}"
        )
    if any(r.tied_with_best for r in ranking):
        print("* the interval overlaps the leader's: a statistical tie, not reordered")
    if ranking and ranking[0].distribution is not None:
        print(distribution_line(ranking[0].distribution))
    if slots:
        who = ", ".join(names.species_names[player[i].species] for i in slots)
        print(f"protected: {who} (a line where one faints fails)")
    if ranking.all_zero:
        print(all_zero_footer(args.deaths))
    return EXIT_OK


def _solve(args) -> int:
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.data import tables as tables_module
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from nuzline.search.plan_tree import render, to_json
    from nuzline.search.stage_b import HiddenLegality, SolveSettings, solve

    if args.max_depth < 0 or args.show_depth < 0:
        raise ValueError("--max-depth and --show-depth must not be negative")
    if args.playouts is not None and args.playouts < 1:
        raise ValueError("--playouts must be at least 1")
    if args.precision is not None and args.playouts is not None:
        raise ValueError("--precision and --playouts are alternatives: give one")
    if args.precision is not None and not args.precision > 0:
        raise ValueError("--precision must be positive")
    if args.epsilon < 0:
        raise ValueError("--epsilon must not be negative")
    if args.budget is not None and args.budget < 0:
        raise ValueError("--budget must not be negative")
    _check_deaths(args)
    settings = SolveSettings(
        max_depth=args.max_depth,
        budget_playouts=args.playouts,
        budget_seconds=args.budget,
        epsilon=args.epsilon,
        precision=None
        if args.playouts is not None
        else (args.precision or DEFAULT_PRECISION_PP) / 100,
    )
    # _fight loads the tables from tables_module.DEFAULT: the version the run records.
    tables_version = table_version(tables_module.DEFAULT)
    enable_compile_cache(tables_version + code_version())
    tables, names, trainer, player, opp, weather = _fight(args)
    slots = protected_slots(player, names, args.protect)
    leads = (args.lead, args.opp_active)
    print(_trainer_line(trainer))
    try:
        require_coverage(tables, names, player, trainer, opp, weather, leads)
    except UnsupportedMechanics as e:
        print(f"refused: {e}")
        _print_report(e.report)
        return EXIT_REFUSED
    state = make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)
    state = state.with_budget(args.deaths, slots)
    try:
        result = solve(tables, names, state, jax.random.PRNGKey(args.seed), settings)
    except (UnsupportedAi, HiddenLegality) as e:
        print(f"refused: {e}")
        return EXIT_REFUSED
    if not result.ranking:
        # Every root action's own turn is sampling-only: nothing to recommend (spec section 5).
        print("refused: " + "; ".join(result.dropped.values()))
        return EXIT_REFUSED
    mine = names.species_names[player[args.lead].species]
    theirs = names.species_names[opp[args.opp_active].species]
    print(f"{mine} vs {theirs}: seed {args.seed}")
    for line in render(result, names, args.show_depth, root=state, trainer=trainer.name):
        print(line)
    tree = to_json(result, names, root=state, trainer=trainer.name, table_version=tables_version)
    if args.tree is not None:
        args.tree.write_text(json.dumps(tree, indent=1))
        print(f"plan tree: {args.tree}")
    record = {
        **_source_hash(args),
        "trainer_id": trainer.id,
        "trainer": trainer.name,
        "lead": args.lead,
        "opp_active": args.opp_active,
        "deaths": args.deaths,
        "protected": list(slots),
        "settings": tree["settings"],
        "seed": args.seed,
        "devices": jax.device_count(),
        "result": tree,
    }
    print(f"run record: {_write_run_record(record)}")
    return EXIT_OK


def _parse_weights(values: Sequence[str]) -> tuple[tuple[str, float], ...]:
    """The cost weights: the defaults overridden by ``--weight KIND=W`` values."""
    from nuzline.prep.variants import DEFAULT_WEIGHTS

    weights = dict(DEFAULT_WEIGHTS)
    for value in values:
        kind, sep, number = value.partition("=")
        try:
            weight = float(number)
        except ValueError:
            weight = -1.0
        if kind not in weights or not sep or not weight >= 0:
            raise ValueError(
                f"--weight {value}: expected KIND=W with KIND one of {', '.join(weights)} "
                "and W a number of at least 0"
            )
        weights[kind] = weight
    return tuple(weights.items())


def _prep(args) -> int:
    import jax

    from nuzline.data import tables as tables_module
    from nuzline.data.learnsets import load_learnsets
    from nuzline.data.tables import load_names, load_tables
    from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
    from nuzline.engine.coverage import check_opponent, start_weather
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from nuzline.prep import report
    from nuzline.prep.caps import default_cap
    from nuzline.prep.problem import build_problem
    from nuzline.prep.search import PrepSettings, run_prep
    from nuzline.saves import load_save

    _check_deaths(args)
    if args.resume is not None and args.checkpoint is not None:
        raise ValueError("--resume continues and rewrites its own file: not with --checkpoint")
    if args.budget is not None and args.budget < 0:
        raise ValueError("--budget must not be negative")
    if args.show_depth < 0:
        raise ValueError("--show-depth must not be negative")
    settings = PrepSettings(
        screen_n=args.screen_n,
        finalists=args.finalists,
        max_depth=args.max_depth,
        entry_hp=not args.no_entry_hp,
        budget_seconds=args.budget,
        weights=_parse_weights(args.weight),
    )
    tables_version = table_version(tables_module.DEFAULT)
    # Before anything compiles: JAX opens the persistent cache at the first compile.
    enable_compile_cache(tables_version + code_version())
    trainer = resolve_trainer(
        load_trainers(INTERMEDIATE), name=args.trainer, trainer_id=args.trainer_id
    )
    cap = args.cap if args.cap is not None else default_cap(trainer)
    tables, names = load_tables(), load_names()
    save = load_save(args.save, allow_corrupt=args.allow_corrupt, tables=tables, names=names)
    opp = build_trainer_party(trainer, IntermediateData(INTERMEDIATE))
    weather = start_weather(trainer)
    problem, warnings = build_problem(
        tables,
        names,
        save,
        trainer,
        opp,
        weather,
        load_learnsets(),
        cap=cap,
        deaths=args.deaths,
        protect=args.protect,
        exclude=args.exclude,
        reserve=args.reserve,
        prestatus=args.prestatus,
    )
    for line in warnings:
        print(line, file=sys.stderr)
    print(_trainer_line(trainer))
    theirs = check_opponent(tables, names, trainer, opp, weather)
    if theirs.missing:
        _print_report(theirs)
        return EXIT_REFUSED

    def progress(p) -> None:
        print(
            f"[prep] {p.phase} round {p.round}: {p.alive} setups, {p.elapsed:.1f} s",
            file=sys.stderr,
            flush=True,
        )

    checkpoint = args.checkpoint
    if checkpoint is None and args.resume is None:
        checkpoint = RUNS / f"prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt.json"
    try:
        result = run_prep(
            tables,
            names,
            problem,
            settings,
            jax.random.PRNGKey(args.seed),
            checkpoint=checkpoint,
            resume=args.resume,
            progress=progress,
            table_version=tables_version,
        )
    except ValueError as e:
        if not str(e).startswith("every setup refused"):
            raise
        print(f"refused: {e}")
        return EXIT_REFUSED
    except (KeyError, TypeError, IndexError) as e:
        if args.resume is None:
            raise
        raise ValueError(f"checkpoint {args.resume} is damaged: {e!r}") from e
    for line in report.render(result, problem, names, args.show_depth):
        print(line)
    rec = report.record(result, problem, names, settings, args.seed, tables_version, args.reserve)
    if args.tree is not None:
        if rec["plan"] is None:
            print("no plan tree: screen only")
        else:
            args.tree.write_text(json.dumps(rec["plan"], indent=1))
            print(f"plan tree: {args.tree}")
    rec = {**_source_hash(args), "devices": jax.device_count(), **rec}
    print(f"run record: {_write_run_record(rec, 'prep')}")
    print(f"checkpoint: {checkpoint or args.resume}")
    return EXIT_OK


def _source_hash(args) -> dict:
    """The run record's hash of the player's party source: ``party_sha256`` of the TOML,
    or ``save_sha256`` of the save file."""
    if args.save is not None:
        return {"save_sha256": hashlib.sha256(args.save.read_bytes()).hexdigest()}
    return {"party_sha256": hashlib.sha256(args.party.read_bytes()).hexdigest()}


def _write_run_record(record: dict, prefix: str = "solve") -> Path:
    """Write ``record`` to a new ``runs/<prefix>_<time to the microsecond>[_n].json`` and
    return its path; an existing record is never overwritten (open mode ``x``)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    text = json.dumps(record, indent=1)
    for n in itertools.count():
        path = RUNS / f"{prefix}_{stamp}{f'_{n}' if n else ''}.json"
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(text)
        except FileExistsError:
            continue
        return path


def _trainers(args) -> int:
    from nuzline.data.trainer import load_trainers

    for t in load_trainers(INTERMEDIATE):
        print(f"{t.id:4d}  {t.name}  ({t.trainer_class})")
    return EXIT_OK


def _save_show(args) -> int:
    from nuzline.data.tables import load_names
    from nuzline.saves import load_save
    from nuzline.saves.export import save_json, show_lines

    names = load_names()
    save = load_save(args.save_path, allow_corrupt=args.allow_corrupt, names=names)
    if args.json:
        print(json.dumps(save_json(save, names), indent=1))  # ensure_ascii: ASCII only
    else:
        for line in show_lines(save, names):
            print(line)
    return EXIT_OK


def _save_export(args) -> int:
    from nuzline.data.tables import load_names
    from nuzline.saves import load_save
    from nuzline.saves.export import party_toml

    names = load_names()
    save = load_save(args.save_path, allow_corrupt=args.allow_corrupt, names=names)
    text = party_toml(save, names)  # may raise RecordError before anything is written
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("rank", "solve", "prep"):
        if args.devices is not None and args.devices < 1:
            print("nuzline: error: --devices must be at least 1", file=sys.stderr)
            return EXIT_ERROR
        configure_cpu_devices(args.devices)
    command = {
        "coverage": _coverage,
        "rank": _rank,
        "solve": _solve,
        "prep": _prep,
        "trainers": _trainers,
        "save": lambda a: {"show": _save_show, "export": _save_export}[a.save_command](a),
    }[args.command]
    try:
        return command(args)
    except (ValueError, OSError) as e:
        print(f"nuzline: error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
