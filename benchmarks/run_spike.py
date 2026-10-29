"""Benchmark harness for the M0 spike: compile time and steady-state state-steps/s."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.spike_step import batched_step, make_state

RESULTS_DIR = Path(__file__).parent / "results"


def measure(batch: int, n_steps: int, seed: int = 0) -> dict:
    """Time the first (compiling) call, then n_steps steady-state calls."""
    state = make_state(seed, batch)
    effect_id = jnp.zeros((batch, 2), jnp.int32)

    t0 = time.perf_counter()
    state = jax.block_until_ready(batched_step(state, effect_id))
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = batched_step(state, effect_id)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    return {
        "batch": batch,
        "compile_s": compile_s,
        "steps_per_s": batch * n_steps / elapsed,
        "n_steps": n_steps,
    }


def _unique_path(out_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"spike_{stamp}.json"
    counter = 0
    while path.exists():
        counter += 1
        path = out_dir / f"spike_{stamp}_{counter}.json"
    return path


def run(batches: list[int], n_steps: int, out_dir: Path) -> Path:
    results = [measure(b, n_steps) for b in batches]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(out_dir)
    payload = {
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"{'batch':>10} {'compile_s':>10} {'steps/s':>14}")
    for r in results:
        print(f"{r['batch']:>10} {r['compile_s']:>10.2f} {r['steps_per_s']:>14,.0f}")
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run(args.batches, args.steps, args.out)


if __name__ == "__main__":
    main()
