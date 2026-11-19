"""Seeded differential runs: the JAX ``step`` against the reference ``ref_step`` on N random
turns per seed, from the generator of ``tests/test_differential.py``.

    uv run python -m tools.differential N SEED [SEED ...]
    uv run python -m tools.differential N A-B          # seeds A..B inclusive

A seed always draws the same N turns, so rerunning a seed replays its failure. A failure is
shrunk and printed (the drawn state, actions and outcome, and the first field that differs);
``--no-shrink`` skips the shrink. The exit code is the number of failed seeds.
"""

import argparse
import sys
import time
import traceback
import warnings

from hypothesis import HealthCheck, Phase, given, seed, settings
from hypothesis import strategies as st

from nuzline.data.games import GAMES, VANILLA
from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import cache_root, jax_version, set_compile_cache_dir
from tests.strategies import any_outcome, legal_action_pair, supported_state
from tests.test_differential import _PAUSES, _step_both

TABLES_DIR = GAMES[VANILLA].tables_dir

# The lazy strategy's repr carries the tables; Hypothesis warns when it builds it.
warnings.filterwarnings("ignore", "Generating overly large repr")


def _seeds(specs: list[str]) -> list[int]:
    out: list[int] = []
    for spec in specs:
        lo, _, hi = spec.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def run_seed(tables, names, n: int, value: int, *, shrink: bool) -> bool:
    """N turns drawn from ``value``; True when every turn matches field for field."""
    phases = [Phase.explicit, Phase.generate] + ([Phase.shrink] if shrink else [])

    @seed(value)
    @settings(
        max_examples=n,
        deadline=None,
        database=None,
        phases=phases,
        print_blob=True,
        suppress_health_check=list(HealthCheck),
    )
    @given(data=st.data())
    def check(data):
        s = data.draw(supported_state(tables, names))
        acts = data.draw(legal_action_pair(tables, s))
        o = data.draw(any_outcome())
        _step_both(tables, s, acts, o, data)

    try:
        check()
    except Exception:  # noqa: BLE001 (any engine error is a failed seed; printed)
        traceback.print_exc()
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("n", type=int, help="turns per seed")
    p.add_argument("seeds", nargs="+", help="seeds: integers or ranges A-B")
    p.add_argument("--no-shrink", action="store_true", help="report a failure unshrunk")
    args = p.parse_args(argv)
    if not (TABLES_DIR / "tables.npz").exists():
        sys.exit("tables not built (tools/ops/build_tables.sh)")
    set_compile_cache_dir(cache_root() / "jax-tests" / jax_version())
    tables, names = load_tables(TABLES_DIR), load_names(TABLES_DIR)
    failed = []
    for value in _seeds(args.seeds):
        start, pauses = time.monotonic(), len(_PAUSES)
        ok = run_seed(tables, names, args.n, value, shrink=not args.no_shrink)
        took = time.monotonic() - start
        print(
            f"seed {value}: {'ok' if ok else 'FAIL'} ({args.n} turns, "
            f"{len(_PAUSES) - pauses} pauses, {took:.0f} s)",
            flush=True,
        )
        if not ok:
            failed.append(value)
    print(f"failed seeds: {failed or 'none'}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
