"""PP-stall fights through the whole prep funnel (plan 4c Task 11, spec section 10.2).

    uv run python -m benchmarks.pp_stall [--stall 5 20 30] [--devices N]

For each stall length (Rock Throw's PP) runs ``run_prep`` with default settings (and the fight's
own horizon, Cranidos's total PP) on ``tests.prep.fixtures.pp_stall_problem``, prints one line
per length and writes ``benchmarks/results/pp_stall_<time>.json`` (gitignored): the chosen team
and lead, the rung-0 plan estimate and interval, whether the chosen team holds the tank and an
attacker, the best screen value of a setup without the tank, and the seconds. The short stall
is required of the search (``tests/prep/test_pp_stall.py``); the long ones are reported only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TANK = 0  # candidate index of the tank in ``pp_stall_problem``; 1 and 2 are attackers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stall", type=int, nargs="+", default=[5, 20, 30])
    parser.add_argument("--devices", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)

    from nuzline.cli import configure_cpu_devices

    configure_cpu_devices(args.devices)  # before jax is imported

    import jax

    from nuzline.data.tables import load_names, load_tables
    from nuzline.prep.search import PrepSettings, run_prep, setup_label
    from tests.prep.fixtures import pp_stall_max_turns, pp_stall_problem, recorded_screens

    tables, names = load_tables(), load_names()
    rows = []
    for stall in args.stall:
        prob = pp_stall_problem(tables, names, stall)
        settings = PrepSettings(max_turns=pp_stall_max_turns(stall))
        t0 = time.perf_counter()
        with recorded_screens() as screens:
            result = run_prep(tables, names, prob, settings, jax.random.PRNGKey(0))
        seconds = time.perf_counter() - t0
        chosen = result.chosen
        team = [m.candidate for m in chosen.members]
        rung0 = result.ladder[0]
        tankless = [
            s.rung(0)
            for setup, s in screens.items()
            if TANK not in [m.candidate for m in setup.members] and s.n
        ]
        row = {
            "stall_pp": stall,
            "team": setup_label(names, prob, chosen),
            "lead": prob.candidates[team[0]].label,
            "estimate": rung0.p,
            "ci_low": rung0.ci_low,
            "ci_high": rung0.ci_high,
            "source": rung0.source,
            "holds_tank": TANK in team,
            "holds_attacker": bool({1, 2} & set(team)),
            "best_tankless_screen": max(tankless, default=None),
            "seconds": seconds,
        }
        rows.append(row)
        found = row["holds_tank"] and row["holds_attacker"]
        print(
            f"stall {stall:>2}: {'found' if found else 'not found'}  rung 0 "
            f"{rung0.p:.3f} [{rung0.ci_low:.3f}, {rung0.ci_high:.3f}] ({rung0.source})  "
            f"best tank-less screen {row['best_tankless_screen']}  {seconds:.0f} s  "
            f"{row['team']}"
        )
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"pp_stall_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
