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
  [--max-depth 2] [--finalists 3] [--screen-n 8] [--max-team 6] [--max-setups 25000]
  [--seed 0] [--devices N] [--tree FILE.json] [--show-depth 1]``: the prep search
  (``prep.search``): which Pokemon, moves, items and entry
  HP to bring from the save's party and boxes, the costed prep steps, the odds ladder for
  0-3 deaths and the Stage B plan, a run record ``runs/prep_<time>.json`` and a checkpoint
  (default ``runs/prep_<time>.ckpt.json``) that ``--resume`` continues. ``--cap`` defaults to
  ``config/level_caps.toml``. ``--protect`` and ``--exclude`` name a Pokemon by species,
  nickname, ``party:N`` or ``box:B:S`` (the labels of ``save show``). Exit code 2 when the
  opponent is unsupported or every setup is refused; an interrupted or budget-stopped run
  exits 0 with its best answer.
* ``nuzline trainers [--save SAVE [--allow-corrupt]]``: every trainer's id, label (else
  name) and class, marked ``[optional]``, ``[gauntlet GROUP #P]`` and with its format when
  it is not a single battle; with ``--save`` only the rival variants of the save's starter.
* ``nuzline save show SAVE [--json] [--allow-corrupt]``: the save's trainer, loaded copy,
  party and boxes (one line per Pokemon), or the whole save as JSON.
* ``nuzline save export SAVE -o PARTY.toml [--allow-corrupt]``: the save's party (eggs
  dropped) as a ``--party`` TOML file.

Every command takes ``--game vanilla|kaizo`` (default vanilla, plan K2b Task 18): the data
set its tables, names and trainers come from (``nuzline.data.games``). Missing or
mismatched data is one line naming the build command and exit code 1. The game is never
inferred from a save (spec section 8): every loaded save's contents that are impossible in
the chosen game are printed to stderr as ``warning:`` lines, and the command goes on. Run
records (solve, prep) and the plan tree JSON carry ``"game"`` next to ``table_version``.

``coverage``, ``rank`` and ``solve`` take ``--save SAVE [--allow-corrupt]`` instead of
``--party``: the party is the save's, eggs left out (each reported on stderr; ``--lead``
indexes the egg-free party), and the solve run record holds ``save_sha256``. A bad save
or record is one line on stderr and exit code 1.

``--trainer NAME`` (``resolve_trainer``): the trainers whose name or label equals NAME
case-insensitively, else those whose label contains it; with a save, the rival variants of
another starter are dropped. Several starter variants of one fight left is an error (pass
``--save`` or ``--trainer-id``); otherwise the first whose class starts with "leader",
else the first match. ``--trainer-id`` selects exactly (plan ruling). A fight whose start
condition is a pending user call ("unknown") or whose party lacks set data the engine
cannot compute (``missing_set_data``) is refused (exit 2) before its party is built.
All output is plain ASCII (controller ruling: a default Windows console is cp1252).
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

from nuzline.data.games import ALIASES
from nuzline.data.trainer import Trainer
from nuzline.text import ascii_text, class_text

# The vanilla trainers directory (benchmarks/ and tests read it; commands use the game's).
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


class _Refused(Exception):
    """A fight refused before its party is built (``_check_buildable``): ``main`` prints
    the trainer line and ``report``, exit code 2."""

    def __init__(self, trainer: Trainer, report) -> None:
        super().__init__(f"{trainer.id}: refused")
        self.trainer, self.report = trainer, report


def _fight_name(t: Trainer) -> str:
    """The fight a starter variant belongs to: its label without the trailing
    ``[Starter]`` (labels like "Pokemon Trainer Barry #3 [Empoleon]")."""
    label = (t.label or t.name).strip()
    if label.endswith("]") and "[" in label:
        label = label[: label.rindex("[")]
    return label.strip()


def _variant_key(t: Trainer) -> str:
    return _fight_name(t).lower()


def resolve_trainer(
    trainers: Sequence[Trainer],
    *,
    name: str | None = None,
    trainer_id: int | None = None,
    player_starter: int | None = None,
) -> Trainer:
    """The trainer ``--trainer``/``--trainer-id`` names (module docstring; plan K2b
    preflight #39): exact case-insensitive ``name`` or ``label`` matches, else a
    case-insensitive substring of ``label``; then the variants whose ``player_starter``
    differs from a given ``player_starter`` are dropped. Several starter variants of one
    fight left is a ``ValueError`` listing their ids; otherwise a Leader is preferred, then
    the first match. Raises ``ValueError`` when there is none."""
    if trainer_id is not None:
        for t in trainers:
            if t.id == trainer_id:
                return t
        raise ValueError(f"no trainer with id {trainer_id}")
    key = (name or "").lower()
    matches = [t for t in trainers if key in (t.name.lower(), (t.label or "").lower())]
    if not matches and key:
        matches = [t for t in trainers if t.label is not None and key in t.label.lower()]
    if not matches:
        raise ValueError(f"no trainer named {name!r} (see `nuzline trainers`)")
    # Starter-variant trainers of several fights (Barry #2 and #3): never guess the fight,
    # with or without a save (fix round 2). Other names keep the first match (ruling #39).
    fights = list(dict.fromkeys(_fight_name(t) for t in matches if t.player_starter is not None))
    if len(fights) > 1:
        raise ValueError(
            f"{name!r} matches {len(fights)} rival fights: {'; '.join(fights)}; pass the "
            f"fight's name (e.g. --trainer {fights[-1]!r}) or --trainer-id"
        )
    if player_starter is not None:
        kept = [t for t in matches if t.player_starter in (None, player_starter)]
        if not kept:
            ids = ", ".join(str(t.id) for t in matches)
            raise ValueError(
                f"no trainer named {name!r} for the save's starter (species {player_starter}): "
                f"trainer ids {ids} are other starters' variants; pass --trainer-id"
            )
        matches = kept
    if (
        len(matches) > 1
        and all(t.player_starter is not None for t in matches)
        and len({_variant_key(t) for t in matches}) == 1
    ):
        _starter_variants(name, matches)
    leaders = [t for t in matches if t.trainer_class.lower().startswith("leader")]
    chosen = (leaders or matches)[0]
    if player_starter is None and chosen.player_starter is not None:
        # A rival or partner team depends on the starter: never guess it (fix round 1, I2).
        _starter_variants(name, [t for t in matches if t.player_starter is not None])
    return chosen


def _starter_variants(name: str | None, variants: Sequence[Trainer]) -> None:
    """Raise the error for a name whose fights are starter variants."""
    ids = ", ".join(str(t.id) for t in variants)
    raise ValueError(
        f"{name!r} is fought with one team per player starter (trainer ids {ids}): "
        "pass --save or --trainer-id"
    )


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

    def game(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--game",
            choices=tuple(ALIASES),
            default="vanilla",
            help="the game's data set (default vanilla)",
        )

    def fight(cmd: argparse.ArgumentParser) -> None:
        game(cmd)
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
    game(prep)
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
        help=(
            "cost weight of candy, heart_scale, tm, hm or held "
            "(last tie-break, after win chance and deaths); repeatable"
        ),
    )
    prep.add_argument(
        "--budget", type=float, default=None, help="seconds after which no new work starts"
    )
    prep.add_argument("--checkpoint", type=Path, default=None, help="checkpoint file to write")
    prep.add_argument("--resume", type=Path, default=None, help="continue from this checkpoint")
    prep.add_argument("--max-depth", type=int, default=2, help="deepest Stage B pass (0 = screen)")
    prep.add_argument("--finalists", type=int, default=3, help="setups solved by Stage B")
    prep.add_argument("--screen-n", type=int, default=8, help="playouts per setup in round 0")
    prep.add_argument("--max-team", type=int, default=6, help="largest team brought (1-6)")
    prep.add_argument(
        "--max-setups",
        type=int,
        default=None,
        help="refuse a team space larger than this (default 25000; see --exclude)",
    )
    prep.add_argument("--seed", type=int, default=0, help="PRNG seed")
    prep.add_argument("--devices", type=int, default=os.cpu_count(), help="host CPU devices")
    prep.add_argument("--tree", type=Path, default=None, help="write the plan tree as JSON")
    prep.add_argument("--show-depth", type=int, default=1, help="plan levels printed")
    trainers = sub.add_parser("trainers", help="list trainer ids, names and classes")
    game(trainers)
    trainers.add_argument(
        "--save", type=Path, default=None, help="list only the rival variants of its starter"
    )
    trainers.add_argument(
        "--allow-corrupt", action="store_true", help="with --save: load a corrupt save anyway"
    )
    save = sub.add_parser("save", help="inspect or export a Platinum save")
    save_sub = save.add_subparsers(dest="save_command", required=True)
    show = save_sub.add_parser("show", help="print the save's trainer, party and boxes")
    export = save_sub.add_parser("export", help="write the save's party as a party TOML")
    for cmd in (show, export):
        cmd.add_argument("save_path", type=Path, metavar="SAVE", help="a .sav or .dsv file")
        game(cmd)
        cmd.add_argument("--allow-corrupt", action="store_true", help="load a corrupt save anyway")
    show.add_argument("--json", action="store_true", help="print the whole save as JSON")
    export.add_argument("-o", "--output", required=True, type=Path, help="party TOML to write")
    return parser


def _say(text: str) -> None:
    """Print ``text`` as plain ASCII (``nuzline.text.ascii_text``): trainer, species and
    move names can carry characters a cp1252 console cannot print."""
    print(ascii_text(text))


def _trainer_line(t: Trainer) -> str:
    return ascii_text(f"{t.label or t.name} ({class_text(t.trainer_class)}, id {t.id})")


def _print_report(report) -> None:
    print(f"Support: {report.level.name}")
    for title, lines in (("Missing", report.missing), ("Sampling only", report.sampling_only)):
        if not lines:
            print(f"{title}: none")
            continue
        print(f"{title} ({len(lines)}):")
        for line in lines:
            _say(f"  {line}")


def _game(args):
    from nuzline.data.games import get_game

    return get_game(args.game)


def _table_version(game) -> str:
    """``jaxcache.table_version`` of ``game``'s tables; missing tables are ``load_game``'s
    ``GameDataError`` naming the build command, not a bare ``FileNotFoundError``."""
    from nuzline.jaxcache import table_version

    try:
        return table_version(game.tables_dir)
    except FileNotFoundError:
        from nuzline.data.games import load_game

        load_game(game)  # raises GameDataError when a file is missing
        raise


def _load_trainers(game) -> list[Trainer]:
    """``game``'s trainers; a missing ``trainers.json`` names the command that writes it."""
    from nuzline.data.games import EXTRACT, GameDataError
    from nuzline.data.trainer import load_trainers

    path = game.intermediate_dir / "trainers.json"
    if not path.exists():
        extract = EXTRACT.get(game.id, "the game's extractor")
        raise GameDataError(f"{game.id}: no {path}; extract it with: {extract}")
    return load_trainers(game.intermediate_dir)


def _load_save(path: Path, allow_corrupt: bool, game, tables, names):
    """The save at ``path`` read with ``game``'s data. Spec section 8: the game is never
    inferred from a save, so every content impossible in ``game`` is a ``warning:`` line on
    stderr (``saves.check.impossible_contents``) and the caller goes on."""
    from nuzline.data.learnsets import load_learnsets
    from nuzline.saves import load_save
    from nuzline.saves.check import impossible_contents

    save = load_save(path, allow_corrupt=allow_corrupt, tables=tables, names=names, game=game.id)
    lines = impossible_contents(save, tables, names, load_learnsets(game.tables_dir))
    for line in lines:
        print(ascii_text(f"warning: {line} (impossible in {game.id})"), file=sys.stderr)
    if lines:
        print(
            ascii_text(
                f"warning: {len(lines)} impossible content(s): is {path} a save of another "
                "game? (--game)"
            ),
            file=sys.stderr,
        )
    return save


def _starter(save) -> int | None:
    """The save's starter species for ``resolve_trainer``; None before the choice (0)."""
    return None if save is None else save.player_starter or None


def _check_buildable(tables, names, trainer: Trainer, inter) -> None:
    """Raise ``_Refused`` with the trainer-only coverage report when the fight cannot be set
    up: its start condition is a pending user call ("unknown", never read as a weather) or a
    party member lacks set data (``missing_set_data``; the party is then not built, as
    ``build_trainer_party`` would raise). Plan K2b preflight #34."""
    from nuzline.data import trainer as trainer_module
    from nuzline.engine.coverage import check_opponent, start_weather, unknown_start
    from nuzline.engine.ids import Weather

    missing = trainer_module.missing_set_data(trainer)
    unknown = unknown_start(trainer)
    if not missing and not unknown:
        return
    opp = [] if missing else trainer_module.build_trainer_party(trainer, inter)
    weather = Weather.NONE if unknown else start_weather(trainer)
    raise _Refused(trainer, check_opponent(tables, names, trainer, opp, weather))


def _fight(args):
    """(game, tables, names, trainer, player party, opponent party, start weather, save or
    None) for ``args.game``. Raises
    ``ValueError`` when ``--lead`` names a fainted Pokemon and ``_Refused`` when the fight
    cannot be set up (``_check_buildable``)."""
    from nuzline.data.games import load_game

    game = _game(args)
    # load_game first: missing Kaizo data is its one-line GameDataError, before any jax.
    tables, names = load_game(game)
    trainers = _load_trainers(game)
    from nuzline.data.party import load_party
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import start_weather

    save = None
    if args.save is not None:
        save = _load_save(args.save, args.allow_corrupt, game, tables, names)
        player = []
        for slot, mon in enumerate(save.party):
            if mon.is_egg:
                print(f"party slot {slot}: egg left out", file=sys.stderr)
            else:
                player.append(mon.to_spec(tables))
        if not player:
            raise ValueError("the save's party has no Pokemon besides eggs")
    trainer = resolve_trainer(
        trainers, name=args.trainer, trainer_id=args.trainer_id, player_starter=_starter(save)
    )
    if save is None:
        player = load_party(args.party, names)
    inter = IntermediateData(game.intermediate_dir)
    _check_buildable(tables, names, trainer, inter)
    opp = build_trainer_party(trainer, inter)
    if 0 <= args.lead < len(player) and player[args.lead].current_hp == 0:
        raise ValueError(f"--lead {args.lead}: that Pokemon has fainted")
    return game, tables, names, trainer, player, opp, start_weather(trainer), save


def _coverage(args) -> int:
    if args.cap is not None and args.save is None:
        raise ValueError("--cap needs --save")
    game, tables, names, trainer, player, opp, weather, save = _fight(args)
    from nuzline.engine.coverage import check_coverage

    print(_trainer_line(trainer))
    leads = (args.lead, args.opp_active)
    report = check_coverage(tables, names, player, trainer, opp, weather, leads)
    _print_report(report)
    if save is not None:
        _print_pool(args, game, tables, names, player, trainer, opp, save)
    return EXIT_REFUSED if report.missing else EXIT_OK


def _print_pool(args, game, tables, names, player, trainer, opp, save) -> None:
    """The pool section of ``coverage --save``; it never changes the exit code. The cap is
    ``--cap``, else the trainer's in the game's caps file (``prep.caps.default_cap``), else,
    for vanilla only, the highest party level. A Kaizo cap is never defaulted (its split
    gaps are user calls): with none, the pool is left out and ``default_cap``'s message,
    which asks for ``--cap``, is the error line."""
    from nuzline.data.games import VANILLA
    from nuzline.data.learnsets import load_learnsets
    from nuzline.prep.caps import default_cap
    from nuzline.prep.pool import check_pool, pool_candidates
    from nuzline.prep.problem import HELD_POCKETS

    cap = args.cap
    if cap is None:
        try:
            cap = default_cap(trainer, game=game.id)
        except ValueError as e:
            if game.id != VANILLA:
                print(ascii_text(f"nuzline: error: pool left out: {e}"), file=sys.stderr)
                return
            cap = max(spec.level for spec in player)
    learnsets = load_learnsets(game.tables_dir)
    candidates, left_out = pool_candidates(save, tables, names, learnsets, cap)
    print(f"Pool at level cap {cap} ({len(candidates)} Pokemon):")
    bag = [entry for pocket in HELD_POCKETS for entry in save.bag.get(pocket, ())]
    _print_report(check_pool(tables, names, candidates, opp, trainer, bag))
    for line in left_out:
        _say(line)


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _rank(args) -> int:
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.jaxcache import code_version, enable_compile_cache
    from nuzline.search.plan_tree import distribution_line
    from nuzline.search.stage_a import rank_actions

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    _check_deaths(args)
    game = _game(args)
    # Before anything compiles: JAX opens the persistent cache at the first compile.
    enable_compile_cache(_table_version(game) + code_version(), game.id)
    _, tables, names, trainer, player, opp, weather, _ = _fight(args)
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
        _say(f"refused: {e}")
        return EXIT_REFUSED
    mine = names.species_names[player[args.lead].species]
    theirs = names.species_names[opp[args.opp_active].species]
    _say(f"{mine} vs {theirs}: {args.n} playouts per action, seed {args.seed}")
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
        _say(
            f"  {mark} {i:2d}  {r.label:<{width}}  {odds:<27}  {_pct(r.p_win):>7}  "
            f"{r.e_faints:9.2f}  {cells}"
        )
    if any(r.tied_with_best for r in ranking):
        print("* the interval overlaps the leader's: a statistical tie, not reordered")
    if ranking and ranking[0].distribution is not None:
        print(distribution_line(ranking[0].distribution))
    if slots:
        who = ", ".join(names.species_names[player[i].species] for i in slots)
        _say(f"protected: {who} (a line where one faints fails)")
    if ranking.all_zero:
        print(all_zero_footer(args.deaths))
    return EXIT_OK


def _solve(args) -> int:
    import jax

    from nuzline.ai.policy import UnsupportedAi
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.jaxcache import code_version, enable_compile_cache
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
    game = _game(args)
    # _fight loads the tables from game.tables_dir: the version the run records.
    tables_version = _table_version(game)
    enable_compile_cache(tables_version + code_version(), game.id)
    _, tables, names, trainer, player, opp, weather, _ = _fight(args)
    slots = protected_slots(player, names, args.protect)
    leads = (args.lead, args.opp_active)
    print(_trainer_line(trainer))
    try:
        require_coverage(tables, names, player, trainer, opp, weather, leads)
    except UnsupportedMechanics as e:
        _say(f"refused: {e}")
        _print_report(e.report)
        return EXIT_REFUSED
    state = make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)
    state = state.with_budget(args.deaths, slots)
    try:
        result = solve(tables, names, state, jax.random.PRNGKey(args.seed), settings)
    except (UnsupportedAi, HiddenLegality) as e:
        _say(f"refused: {e}")
        return EXIT_REFUSED
    if not result.ranking:
        # Every root action's own turn is sampling-only: nothing to recommend (spec section 5).
        _say("refused: " + "; ".join(result.dropped.values()))
        return EXIT_REFUSED
    mine = names.species_names[player[args.lead].species]
    theirs = names.species_names[opp[args.opp_active].species]
    _say(f"{mine} vs {theirs}: seed {args.seed}")
    shown = trainer.label or trainer.name
    for line in render(result, names, args.show_depth, root=state, trainer=shown):
        _say(line)
    tree = to_json(result, names, root=state, trainer=trainer.name, table_version=tables_version)
    tree["game"] = game.id
    if args.tree is not None:
        args.tree.write_text(json.dumps(tree, indent=1))
        print(f"plan tree: {args.tree}")
    record = {
        **_source_hash(args),
        "game": game.id,
        "table_version": tables_version,
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

    from nuzline.data.games import load_game
    from nuzline.data.learnsets import load_learnsets
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import check_opponent, start_weather
    from nuzline.jaxcache import code_version, enable_compile_cache
    from nuzline.prep import report
    from nuzline.prep.caps import default_cap
    from nuzline.prep.problem import build_problem
    from nuzline.prep.search import PrepSettings, run_prep
    from nuzline.prep.setups import MAX_TEAM_SETUPS, check_team_space

    _check_deaths(args)
    if args.resume is not None and args.checkpoint is not None:
        raise ValueError("--resume continues and rewrites its own file: not with --checkpoint")
    if args.budget is not None and args.budget < 0:
        raise ValueError("--budget must not be negative")
    if args.show_depth < 0:
        raise ValueError("--show-depth must not be negative")
    if not 1 <= args.max_team <= 6:
        raise ValueError(f"--max-team {args.max_team}: must be between 1 and 6")
    settings = PrepSettings(
        screen_n=args.screen_n,
        max_team=args.max_team,
        finalists=args.finalists,
        max_depth=args.max_depth,
        entry_hp=not args.no_entry_hp,
        budget_seconds=args.budget,
        weights=_parse_weights(args.weight),
    )
    game = _game(args)
    tables_version = _table_version(game)
    # Before anything compiles: JAX opens the persistent cache at the first compile.
    enable_compile_cache(tables_version + code_version(), game.id)
    # load_game before the trainers: missing Kaizo data is its GameDataError line.
    tables, names = load_game(game)
    trainers = _load_trainers(game)
    save = _load_save(args.save, args.allow_corrupt, game, tables, names)
    trainer = resolve_trainer(
        trainers, name=args.trainer, trainer_id=args.trainer_id, player_starter=_starter(save)
    )
    inter = IntermediateData(game.intermediate_dir)
    _check_buildable(tables, names, trainer, inter)
    opp = build_trainer_party(trainer, inter)
    weather = start_weather(trainer)
    # A refused fight exits 2 before the cap, the pool or the team space can fail it. No
    # player party here: the team is not chosen yet, so build_problem's usable_pool checks
    # each candidate's side of the Kaizo ROM-change conditions instead.
    theirs = check_opponent(tables, names, trainer, opp, weather)
    if theirs.missing:
        print(_trainer_line(trainer))
        _print_report(theirs)
        return EXIT_REFUSED
    cap = args.cap if args.cap is not None else default_cap(trainer, game=game.id)
    problem, warnings = build_problem(
        tables,
        names,
        save,
        trainer,
        opp,
        weather,
        load_learnsets(game.tables_dir),
        cap=cap,
        deaths=args.deaths,
        protect=args.protect,
        exclude=args.exclude,
        reserve=args.reserve,
        prestatus=args.prestatus,
    )
    for line in warnings:
        print(ascii_text(line), file=sys.stderr)
    limit = MAX_TEAM_SETUPS if args.max_setups is None else args.max_setups
    check_team_space(len(problem.candidates), settings.max_team, limit)
    print(_trainer_line(trainer))

    def progress(p) -> None:
        print(
            f"[prep] {p.phase} round {p.round}: {p.alive} setups, {p.elapsed:.1f} s",
            file=sys.stderr,
            flush=True,
        )

    checkpoint = args.checkpoint
    if checkpoint is None and args.resume is None:
        checkpoint = RUNS / f"prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt.json"
    # Up front, so a crashed run can be continued with --resume.
    print(f"[prep] checkpoint: {checkpoint or args.resume}", file=sys.stderr, flush=True)
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
            game=game.id,
        )
    except ValueError as e:
        if not str(e).startswith("every setup refused"):
            raise
        _say(f"refused: {e}")
        return EXIT_REFUSED
    except (KeyError, TypeError, IndexError) as e:
        if args.resume is None:
            raise
        raise ValueError(f"checkpoint {args.resume} is damaged: {e!r}") from e
    for line in report.render(result, problem, names, args.show_depth, tables=tables):
        _say(line)
    rec = report.record(
        result,
        problem,
        names,
        settings,
        args.seed,
        tables_version,
        args.reserve,
        game=game.id,
        tables=tables,
    )
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


def _listing_line(t: Trainer) -> str:
    """One ``nuzline trainers`` line: id, label (else name), the optional, gauntlet and
    non-single format marks, and the class."""
    marks = ""
    if t.optional:
        marks += " [optional]"
    if t.gauntlet is not None:
        group, position = t.gauntlet
        marks += f" [gauntlet {group} #{position}]"
    if t.format != "single":
        marks += f" [{t.format}]"
    return ascii_text(f"{t.id:4d}  {t.label or t.name}{marks}  ({class_text(t.trainer_class)})")


def _trainers(args) -> int:
    game = _game(args)
    trainers = _load_trainers(game)
    starter = None
    if args.save is not None:
        from nuzline.data.games import load_game

        tables, names = load_game(game)
        starter = _starter(_load_save(args.save, args.allow_corrupt, game, tables, names))
    for t in trainers:
        if starter is not None and t.player_starter not in (None, starter):
            continue  # another starter's rival variant
        print(_listing_line(t))
    return EXIT_OK


def _load_save_command(args):
    """(game, tables, names, save) of ``nuzline save show|export``, read with ``--game``'s
    data."""
    from nuzline.data.games import load_game

    game = _game(args)
    tables, names = load_game(game)
    return game, tables, names, _load_save(args.save_path, args.allow_corrupt, game, tables, names)


def _save_show(args) -> int:
    from nuzline.saves.export import save_json, show_lines

    _, _, names, save = _load_save_command(args)
    if args.json:
        print(json.dumps(save_json(save, names), indent=1))  # ensure_ascii: ASCII only
    else:
        for line in show_lines(save, names):
            print(line)
    return EXIT_OK


def _save_export(args) -> int:
    from nuzline.saves.export import party_toml

    game, tables, names, save = _load_save_command(args)
    # The game's own tables: SaveMon.to_spec checks the party against them.
    text = party_toml(save, names, tables, game=game.id)  # may raise RecordError first
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
    except _Refused as r:
        print(_trainer_line(r.trainer))
        _print_report(r.report)
        return EXIT_REFUSED
    except (ValueError, OSError) as e:
        print(ascii_text(f"nuzline: error: {e}"), file=sys.stderr)
        return EXIT_ERROR
    except KeyError as e:  # a Names lookup ("unknown move 'Water Sport': ... is 'Scald'")
        print(ascii_text(f"nuzline: error: {e.args[0] if e.args else e}"), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
