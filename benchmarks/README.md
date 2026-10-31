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

    uv run python -m benchmarks.solve_timing --calibrate

Runs `nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth D
--playouts B` for D = 1 and 2 in subprocesses sharing a fresh temporary `NUZLINE_CACHE_DIR`
(cold, then warm) and writes `benchmarks/results/solve_<timestamp>.json` (gitignored): wall
time, the leaf-playout phase (`evaluate_leaves` and `evaluate_forced`) and the rest
(expansion: exact chance and AI enumeration, merging, backup), peak resident set, completed
depth, the leader's interval half-width and the playouts run. It exits non-zero if the output
has no ranking header. `--playouts D=B` fixes a budget instead of calibrating.

Targets (spec section 7, cold, at a leading-action half-width of at most 1.5 percentage
points): depth 1 within 2 minutes, depth 2 within 10 minutes. Budgets: `B_d` starts at
`2 * K_d * 4300` (K_1 = 9, K_2 = 81 player-action combinations; 4300 effective playouts per
1.5 pp half-width, notes/critic.md C7; the factor 2 is the ruling-5 half split) and doubles
until the leader's half-width is at most 1.5 pp. Environment as above (13th Gen Intel(R)
Core(TM) i5-13600K, 20 host devices, Windows 11), 2026-10-31. Sample party (Monferno lead)
against Roark, leader = the top-ranked action.

| depth | B_d     | run  | wall   | leaf playouts | expansion | peak RSS | playouts run | leader half-width | target |
|-------|---------|------|--------|---------------|-----------|----------|--------------|-------------------|--------|
| 1     | 77 400  | cold | 85.2 s | 65.4 s        | 15.4 s    | 2.1 GB   | 77 794       | 0.75 pp           | 120 s: met |
| 1     | 77 400  | warm | 57.8 s | 49.2 s        | 4.4 s     | 1.9 GB   | 77 794       | 0.75 pp           | |
| 2     | 696 600 | cold | 909 s  | 880 s         | 23.8 s    | 3.2 GB   | 1 420 086    | 0.60 pp           | 600 s: **missed by 309 s** |
| 2     | 696 600 | warm | 859 s  | 842 s         | 12.5 s    | 3.2 GB   | 1 420 086    | 0.60 pp           | |
| 2     | 174 150 | cold | 314 s  | 286 s         | 23.6 s    | 3.1 GB   | 380 514      | 1.15 pp           | 600 s: met |
| 2     | 174 150 | warm | 268 s  | 250 s         | 12.8 s    | 3.1 GB   | 380 514      | 1.15 pp           | |

The calibration rule's starting budgets already give the leader a half-width under 1.5 pp
(the leader sits near 93 to 96 percent, where a binomial interval is narrow), so no doubling
happened and the rows for `B_d = 2 * K_d * 4300` are the rule's answer. At depth 1 the target
is met cold (85 s of 120 s). At depth 2 the target is **missed** at that budget: 909 s cold
against 600 s. Nearly all of it (880 s, 97 percent) is leaf playouts; expansion is 24 s
cold. The fixed leaf floor (2 per half per evaluated state, ruling 8) pushes the playouts
actually run to 1.42 M, twice `B_2`, and they run at about 1 600 per second, below the
2 800 per second of the Stage A spike, because forced-pair leaves and deeper positions are
costlier per playout. The extra row, a quarter of that budget, is not a calibration result:
it shows that the same fight reaches a 1.15 pp half-width in 314 s cold, so the miss comes
from the starting budget rule overshooting the precision needed, not from a failure to reach
the precision in time. The targets are unchanged.

Levers for the depth-2 miss, in the order of spec section 7 (none applied here):
1. a faster playout policy per turn (it dominates turn cost);
2. chunked playouts with compaction (spec section 6);
3. smaller leaf floors (`leaf_floor`, which doubled the playouts run at depth 2).
