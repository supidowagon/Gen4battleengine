# Milestone 0 performance spike

Throwaway benchmark: one masked JAX `step` with 100 data-driven effects
(`N_EFFECTS` in `spike_step.py`), jitted and vmapped, on CPU.

## How to run

    uv run python -m benchmarks.run_spike --batches 1000 10000 100000 --steps 30

Results are written to `benchmarks/results/spike_<YYYYMMDD_HHMMSS>.json` (gitignored);
the numbers that matter are recorded below.

## Go/no-go thresholds

At batch 10 000 on CPU: compile time <= 120 s and >= 50 000 state-steps/s.
(10 000 playouts x 30 turns = 300 000 steps, so 50 000/s is ~6 s per action evaluation.)

## Measured (CPU, 30 steps, effect ids all 0)

Environment: jax 0.12.0.1, cpu backend, 13th Gen Intel(R) Core(TM) i5-13600K, Windows 11.

| batch   | compile_s | state-steps/s |
|---------|-----------|---------------|
| 1 000   | 1.32      | 1 381 425     |
| 10 000  | 1.34      | 2 024 074     |
| 100 000 | 1.40      | 1 952 616     |

Batch 10 000: compile 1.34 s (limit 120 s), 2.02 M state-steps/s (minimum 0.05 M).
A controller re-measure with random effect ids in [0, 100) gave compile 1.39 s and 2.02 M state-steps/s, so the all-zero ids do not flatter the result.

## Decision

**GO** - the single masked step with 100 effects compiles in about 1 s and runs roughly
40x above the throughput threshold. Caveat: the synthetic state is much smaller than the
real engine's, so real numbers will be lower; re-measure in milestone 1.
(Had it been NO-GO, the first mitigation would be spec 4.2 "Per-fight specialization":
reduce `N_EFFECTS` to 30 and re-measure.)
