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

# Milestone 1b: `nuzline rank` wall time

    uv run nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --n 2000

Sample party (Monferno lead) against Roark (trainer 246, real party and bag), nine legal
actions x 2000 playouts, 100-turn cap. Environment: jax 0.12.0.1, cpu backend,
13th Gen Intel(R) Core(TM) i5-13600K (20 logical CPUs), Windows 11, 2026-10-29.

| n    | wall time (whole command, incl. compile) |
|------|------------------------------------------|
| 20   | 1 min 6 s (`--lead 2 --opp-active 1`)    |
| 2000 | 11 min 37 s                              |

**Over the 60 s target.** The `n = 20` run is almost all trace and compile of the
batched playout (`stage_a._playouts`: `step` plus `ai_policy` inside a 100-turn loop),
so both compile time and per-playout throughput need work. This is the first target
for the per-fight specialization of spec §4.2 (compile only the move effects, abilities
and items present in the fight, and Roark's AI script paths).

## After the ranking speed work (branch perf-ranking)

    uv run python -m benchmarks.rank_timing --n 2000

Runs the command above twice in subprocesses sharing a fresh temporary `NUZLINE_CACHE_DIR`
(cold: empty compile and trace caches; warm: the second run reuses them) and writes
`benchmarks/results/rank_<timestamp>.json` (gitignored). Same fight and `n` as the table
above, measured 2026-10-29 on a 13th Gen Intel(R) Core(TM) i5-13600K (20 logical CPUs,
all used as host devices), jax 0.12.0.1, Windows 11.

| run                                  | wall time | vs 60 s target |
|--------------------------------------|-----------|----------------|
| before (M1b Task 8)                  | 11 min 37 s (697 s) | over |
| after, cold (empty caches)           | 69.9 s    | over by 9.9 s  |
| after, warm (caches populated)       | 21.5 s    | met (warm only) |
| target                               | 60 s      |                |

**Target met warm, missed cold by 10 s.** The warm run (21.5 s) is under the 60 s
target; the cold run (first run on a machine, or after a code or table change) is
69.9 s, about 10 s over. The milestone's headline goal is therefore met only once the
compile and trace caches are populated, not on a fresh machine.

## After the cold-start work (branch perf-ranking, 2026-10-30)

Same command, fight, `n` and machine. A cold-run profile showed the playouts themselves
take only 6.5 s; tracing (27.5 s) and XLA compilation (26.6 s) of the batched playout
were the rest, both proportional to program size (12.7 MB of StableHLO). Python-unrolled
loops put many copies of the same code in the program: the 5 hits of a move (11 copies
of the move-effect code per turn), the two actors of a turn, the AI's 4 move slots per
flag script, and the 5 replacement rounds. Each is now a `jax.lax.fori_loop` (one copy;
results bit-identical, 5.4 MB of StableHLO). The AI's refusal flags (`policy._refused`,
`switching._layout`) are jitted, so the coverage check no longer runs hundreds of eager
ops (4.5 s to 1.7 s).

| run                                  | wall time | vs 60 s target |
|--------------------------------------|-----------|----------------|
| cold (empty caches)                  | 36.8 s    | met            |
| warm (caches populated)              | 14.8 s    | met            |

Cold-run phases at n = 2000 before / after: trace 27.5 / 12.7 s, compile 26.6 / 11.5 s,
run 6.5 / 6.5 s. XLA compiler options were tried and rejected: optimisation level 0
saves 5 s of compile but doubles the run; level 1 saves 3.5 s; neither beats shrinking
the program.

Remaining levers, if ever needed: chunked playouts with compaction and racing (spec 4.3)
now only touch the 6.5 s run phase (playouts: mean 8.6 turns, max 21, 100-turn cap);
`make_state`'s eager switch-in hooks cost about 1 s.

# Milestone 2: `nuzline solve` wall time (Stage B)

    uv run python -m benchmarks.solve_timing [--precision 1.5] [--playouts D=B ...]

Runs `nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth D
--precision 1.5` for D = 1 and 2 in subprocesses sharing a fresh temporary
`NUZLINE_CACHE_DIR` (cold, then warm) and writes `benchmarks/results/solve_<timestamp>.json`
(gitignored): wall time, the leaf-playout phase (`evaluate_leaves` and `evaluate_forced`,
every round) and the rest (expansion: exact chance and AI enumeration, merging, backup),
peak resident set, completed depth, the playouts run, the top-up rounds of the final pass,
the leader's half-A half-width and whether the target was met, and the leader's printed
interval. It exits non-zero if the output has no ranking header. `--playouts D=B` runs depth
`D` with the fixed leaf budget `B` instead (no top-ups).

Budgets come from the solver's own precision target (spec section 3, "Amendment
2026-10-31: precision target"): each pass runs round 0 at `B_d = n_depth0 *` legal root
actions (18 000 here) with a floor of 1 playout per half, then doubles its playouts in
top-up rounds, shared by reach, until the half-A leader's half-A half-width is at most 1.5
percentage points (at most 8 top-ups). This replaces the earlier calibration, which doubled
a fixed `B_d` from `2 * K_d * 4300` and overshot the needed precision about 7 times at
depth 2.

Targets (spec section 7, cold, at a leading-action half-width of at most 1.5 percentage
points): depth 1 within 2 minutes, depth 2 within 10 minutes. Environment as above (13th Gen
Intel(R) Core(TM) i5-13600K, 20 host devices, Windows 11). Sample party (Monferno lead)
against Roark, leader = the top-ranked action.

| depth | run  | wall | leaf playouts | expansion | peak RSS | playouts run | top-up rounds | leader half-width (half A) | target met | time target |
|-------|------|------|---------------|-----------|----------|--------------|---------------|----------------------------|------------|-------------|
| 1     | cold | 66 s | 46 s          | 16 s      | 2128 MB  | 36 488       | 1             | 1.06 pp (printed: 1.1 pp)  | yes        | 120 s: met  |
| 1     | warm | 39 s | 30 s          | 5 s       | 1934 MB  | 36 488       | 1             | 1.06 pp (printed: 1.1 pp)  | yes        |             |
| 2     | cold | 207 s | 178 s        | 24 s      | 3019 MB  | 179 910      | 2             | 1.44 pp (printed: 1.6 pp)  | yes        | 600 s: met  |
| 2     | warm | 161 s | 143 s        | 13 s      | 3016 MB  | 179 910      | 2             | 1.44 pp (printed: 1.6 pp)  | yes        |             |

Measured 2026-10-31. The stopping rule reads half A only (ruling 5), so the printed half-B
interval can come out slightly wider than the target, as at depth 2 (1.6 pp printed for
1.44 pp on half A); the two halves are independent samples of the same size.

Earlier fixed-budget measurements (2026-10-31, before the precision target): depth 1 at
`B_1` = 77 400 took 85 s cold (0.75 pp); depth 2 at `B_2` = 696 600 took 909 s cold (0.60
pp, the 600 s target missed), and a quarter of that budget reached 1.15 pp in 314 s cold.
Leaf playouts were 97 percent of the depth-2 time.
