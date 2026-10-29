"""The ``nuzline`` command line (docs/design.md §4.3, plan M1b Task 8).

* ``nuzline coverage --party P (--trainer NAME | --trainer-id N) [--lead 0]
  [--opp-active 0]``: the fight's ``CoverageReport`` for the same turn-0 state ``rank``
  would use; exit code 2 when anything is missing.
* ``nuzline rank --party P (--trainer NAME | --trainer-id N) [--n 2000] [--seed 0]
  [--lead 0] [--opp-active 0]``: ``require_coverage``, then the Stage A ranking of the
  player's actions at the fight's turn-0 state (the trainer's real built party, bag and
  start condition; party slots ``--lead`` and ``--opp-active`` active). Exit code 2 when
  the fight is refused, by the coverage check or by a refused playout.
* ``nuzline trainers``: every trainer's id, name and class.

``--trainer NAME`` is the first trainer whose name matches case-insensitively and whose
class starts with "leader", else the first match; ``--trainer-id`` selects exactly
(plan ruling). All output is plain ASCII (controller ruling: a default Windows console
is cp1252).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nuzline.data.trainer import Trainer

INTERMEDIATE = Path(__file__).resolve().parents[2] / "data" / "intermediate"

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nuzline", description="Nuzlocke line finder for Pokemon Platinum battles."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def fight(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--party", required=True, type=Path, help="the player's party TOML")
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
    sub.add_parser("trainers", help="list trainer ids, names and classes")
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
    from nuzline.engine.coverage import UnsupportedMechanics, require_coverage
    from nuzline.engine.state import make_state
    from nuzline.search.stage_a import rank_actions

    if args.n < 1:
        raise ValueError("--n must be at least 1")
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


def _trainers(args) -> int:
    from nuzline.data.trainer import load_trainers

    for t in load_trainers(INTERMEDIATE):
        print(f"{t.id:4d}  {t.name}  ({t.trainer_class})")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = {"coverage": _coverage, "rank": _rank, "trainers": _trainers}[args.command]
    try:
        return command(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"nuzline: error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
