"""The ``nuzline`` command line (docs/design.md §4.3, plan M1b Task 8).

* ``nuzline coverage --party P (--trainer NAME | --trainer-id N) [--lead 0]
  [--opp-active 0]``: the fight's ``CoverageReport`` for the same turn-0 state ``rank``
  would use; exit code 2 when anything is missing.
* ``nuzline rank --party P (--trainer NAME | --trainer-id N) [--n 2000] [--seed 0]
  [--lead 0] [--opp-active 0] [--devices N]``: ``require_coverage``, then the Stage A ranking of the
  player's actions at the fight's turn-0 state (the trainer's real built party, bag and
  start condition; party slots ``--lead`` and ``--opp-active`` active). Exit code 2 when
  the fight is refused, by the coverage check or by a refused playout.
* ``nuzline solve --party P (--trainer NAME | --trainer-id N) [--lead 0] [--opp-active 0]
  [--max-depth 2] [--budget SECONDS] [--precision PP | --playouts N] [--epsilon 1e-4]
  [--seed 0] [--devices N] [--tree FILE.json] [--show-depth 1]``: the Stage B plan
  (``search.stage_b``): the ranking with intervals and bounds, the plan tree under the best
  action, whether the precision target was met, optionally the whole tree as JSON, and a run
  record under ``runs/`` (party hash, trainer, settings, seed, result). Each pass tops up its
  leaf playouts until the leader's half-width is at most ``--precision`` percentage points
  (default 1.5); ``--playouts N`` instead fixes every pass's leaf budget (no top-ups). Exit
  code 2 when the fight or the opponent AI is refused, or a belief node's states differ in
  the player's legal actions (``stage_b.HiddenLegality``).
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

    fight(sub.add_parser("coverage", help="check that the engine supports a fight"))
    rank = sub.add_parser("rank", help="rank the player's actions at the fight's start")
    fight(rank)
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

    tables, names, trainer, player, opp, weather = _fight(args)
    print(_trainer_line(trainer))
    leads = (args.lead, args.opp_active)
    report = check_coverage(tables, names, player, trainer, opp, weather, leads)
    _print_report(report)
    return EXIT_REFUSED if report.missing else EXIT_OK


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _rank(args) -> int:
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.data import tables as tables_module
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.jaxcache import code_version, enable_compile_cache, table_version
    from nuzline.search.stage_a import rank_actions

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    # Before anything compiles: JAX opens the persistent cache at the first compile.
    enable_compile_cache(table_version(tables_module.DEFAULT) + code_version())
    tables, names, trainer, player, opp, weather = _fight(args)
    leads = (args.lead, args.opp_active)
    print(_trainer_line(trainer))
    try:
        require_coverage(tables, names, player, trainer, opp, weather, leads)
    except UnsupportedMechanics as e:
        _print_report(e.report)
        return EXIT_REFUSED
    state = make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)
    try:
        ranking = rank_actions(tables, names, state, args.n, jax.random.PRNGKey(args.seed))
    except UnsupportedAi as e:
        print(f"refused: {e}")
        return EXIT_REFUSED
    mine = names.species_names[player[args.lead].species]
    theirs = names.species_names[opp[args.opp_active].species]
    print(f"{mine} vs {theirs}: {args.n} playouts per action, seed {args.seed}")
    width = max(len("action"), *(len(r.label) for r in ranking))
    print(
        f"     #  {'action':<{width}}  {'P(win, 0 faints) [95% CI]':<27}  {'P(win)':>7}  E[faints]"
    )
    for i, r in enumerate(ranking, 1):
        mark = "*" if r.tied_with_best else " "
        odds = f"{_pct(r.p_win0)} [{100 * r.ci_low:4.1f}, {100 * r.ci_high:5.1f}]"
        print(
            f"  {mark} {i:2d}  {r.label:<{width}}  {odds:<27}  {_pct(r.p_win):>7}  {r.e_faints:9.2f}"
        )
    if any(r.tied_with_best for r in ranking):
        print("* the interval overlaps the leader's: a statistical tie, not reordered")
    if ranking.all_zero:
        print(ALL_ZERO_FOOTER)
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
    leads = (args.lead, args.opp_active)
    print(_trainer_line(trainer))
    try:
        require_coverage(tables, names, player, trainer, opp, weather, leads)
    except UnsupportedMechanics as e:
        print(f"refused: {e}")
        _print_report(e.report)
        return EXIT_REFUSED
    state = make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)
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
        "settings": tree["settings"],
        "seed": args.seed,
        "devices": jax.device_count(),
        "result": tree,
    }
    print(f"run record: {_write_run_record(record)}")
    return EXIT_OK


def _source_hash(args) -> dict:
    """The run record's hash of the player's party source: ``party_sha256`` of the TOML,
    or ``save_sha256`` of the save file."""
    if args.save is not None:
        return {"save_sha256": hashlib.sha256(args.save.read_bytes()).hexdigest()}
    return {"party_sha256": hashlib.sha256(args.party.read_bytes()).hexdigest()}


def _write_run_record(record: dict) -> Path:
    """Write ``record`` to a new ``runs/solve_<time to the microsecond>[_n].json`` and
    return its path; an existing record is never overwritten (open mode ``x``)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    text = json.dumps(record, indent=1)
    for n in itertools.count():
        path = RUNS / f"solve_{stamp}{f'_{n}' if n else ''}.json"
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
    if args.command in ("rank", "solve"):
        if args.devices is not None and args.devices < 1:
            print("nuzline: error: --devices must be at least 1", file=sys.stderr)
            return EXIT_ERROR
        configure_cpu_devices(args.devices)
    command = {
        "coverage": _coverage,
        "rank": _rank,
        "solve": _solve,
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
