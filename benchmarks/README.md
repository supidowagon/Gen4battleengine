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

# Milestone 4: PP stall

    uv run python -m benchmarks.pp_stall --stall 5 20 30

Roark's Cranidos alone (Rock Throw with the given PP, Leer with 30) against a tank (Geodude,
Defense Curl only, stats overridden so it survives every Rock Throw), an attacker (Starly) and
a decoy (Shinx); no bag, no resources (`tests/prep/fixtures.py::pp_stall_problem`). The only
deathless lines stall Rock Throw's PP with the tank, then bring in an attacker. `run_prep` with
default settings and the fight's own horizon (Cranidos's total PP, 35 / 50 / 60 turns), seed 0.
Measured 2026-11-02, 13th Gen Intel(R) Core(TM) i5-13600K (20 host devices), Windows 11, with
warm compile caches.

| stall (Rock Throw PP) | found | rung-0 estimate (95 % interval) | best tank-less screen | seconds |
|-----------------------|-------|---------------------------------|-----------------------|---------|
| 5                     | yes   | 1.000 (0.996 - 1.000)           | 0.125                 | 83      |
| 20                    | yes   | 1.000 (0.998 - 1.000)           | 0.125                 | 74      |
| 30                    | yes   | 1.000 (0.998 - 1.000)           | 0.054                 | 78      |

"Found" means the chosen team holds the tank and an attacker. The long stalls (spec section
10.2: reported, not blocking) fall short nowhere on this fight: the default funnel finds the
stall line at 20 and 30 PP as well (Starly leads and switches to the tank), the tank-less
screen values staying far below the pick. The short stall is required by
`tests/prep/test_pp_stall.py`.

# Milestone 4: exhaustive check

    uv run python -m benchmarks.exhaustive_roark --save "$NUZLINE_USER_SAVE" [--resume FILE]

The funnel's pick on Roark from the user's save against the value of every setup of the
restricted space (spec section 8, plan 4c ruling 14): the six party Pokemon plus the first
Venonat (box 1 slot 1) and Geodude (box 1 slot 4), the configured cap, no items, full HP, every
team of 1-6 with every lead, each member its base moves plus at most one member on its
designated Move Relearner change (`prep/exhaustive.py::restricted_setups`), and the *pruned*
setups (`pruned_setups`: the restricted shape with one member on a variant the conservative prune
dropped from the `relearner="all"` variants, built with no free level-up replacements; none, and then the result says so, when the prune
drops nothing on the save). Per rung 0 and 1: phase 1 values every setup by Stage B depth 1 at
2000 playouts, phase 2 re-solves at the 1.5 pp precision target the setups overlapping the best,
the pick and every pruned setup whose phase-1 interval reaches above the pick's. The run
checkpoints after every setup (`--resume FILE` continues across nights; it prints an ETA after 20
setups) and writes `benchmarks/exhaustive_roark.json`. The benchmark's funnel runs with
`levelled=False` (no free level-up slot replacements, which the valued space does not contain)
and the problem's Heart Scales are pinned to at most one. A failed verdict is a finding for the
controller, not an error.

Test sizes (`tests/prep/test_exhaustive.py`, one device, 13th Gen Intel(R) Core(TM) i5-13600K,
measured 2026-11-02): default suite, whole file about 300 s (`test_funnel_matches_exhaustive_ci`
160 s: 44 depth-1 solves plus a funnel run, per-solve overhead dominates, so lowering
`budget_playouts` from 100 to 50 saved only 5 s; `test_pruned_setups_are_valued` 85 s;
`test_funnel_matches_exhaustive_small` 25 s each), over the ~60 s default-file limit.
`NUZLINE_SLOW_TESTS=1`: `test_funnel_matches_exhaustive_ci_full` (every team of the 4-Pokemon pool,
rungs 0 and 1, 500 playouts) 484 s and 394 s, 880 s in all; both rungs pass. The default-suite
overrun is reported to the controller rather than gated (plan Step 3).

Result (`benchmarks/exhaustive_roark.json`, user's save, 20 host devices, i5-13600K, run
2026-11-02 to 2026-11-04, about 37 h of compute over several resumes): 3555 restricted setups, 0
pruned (the prune drops nothing on this save). **Rung 0 FAILED**: pick #3357 (PSYDUCK / FURRET /
ZUBAT / SPINARAK / VENONAT / GEODUDE with Rock Polish) 0.989 [0.982, 0.996], best #891 (PSYDUCK /
FURRET with Fury Swipes / MONFERNO / ZUBAT) 0.999 [0.996, 1.0]; phase 2 covered 2335 setups. **Rung
1 PASSED**: pick #2961 1.0 [0.998, 1.0], e_deaths 0.005, ties best #23 1.0 [0.993, 1.0]; phase 2
covered 2930 setups. Phase seconds: rung 0 funnel 165, phase 1 10903, phase 2 39998; rung 1 funnel
105, phase 1 16675, phase 2 54313. The rung-0 miss was traced to the funnel's saturated tie-breaks
(unequal playout counts at the variant cut, whole-party HP left, and the cost-first leader rule);
the fixes are checked against these stored values rather than by a new run.

# Milestone 4: `nuzline prep` wall time

    uv run python -m benchmarks.prep_timing --save "$NUZLINE_USER_SAVE" [--devices N]

`nuzline prep --save S --trainer roark` (defaults otherwise, 20 host devices) as one cold
subprocess (fresh `NUZLINE_CACHE_DIR`) and then one warm one, against the 900 s target
(`TARGET_SECONDS`, spec section 8). User's save, 13th Gen Intel(R) Core(TM) i5-13600K,
Windows 11, measured 2026-11-02. Wall seconds; phase seconds are the cold run's.

| settings (cumulative)               | cold  | warm  | teams | variants | finalists | screened | playouts  | rung 0 |
|-------------------------------------|-------|-------|-------|----------|-----------|----------|-----------|--------|
| **defaults** (`screen_n` 8, 32 / 3 / 2) | 964.2 | 878.2 | 737.7 | 127.8    | 92.9      | 9770     | 1,144,478 | 1.000  |
| `team_keep` 16                      | 925.5 | 842.8 | 779.7 | 60.2     | 80.5      | 8377     | 1,100,648 | 1.000  |
| + `finalists` 2                     | 923.6 | 840.1 | 781.7 | 60.2     | 76.5      | 8377     | 1,090,648 | 1.000  |
| + `variant_rounds` 1                | 929.6 | 845.8 | 781.1 | 48.5     | 94.8      | 8102     | 1,093,818 | 0.997  |
| `screen_n` 4, `finalists` 2         | 636.8 | crash | 437.8 | 84.8     | 108.6     | 9621     | 608,244   | 0.992  |
| `screen_n` 6, `finalists` 2         | 817.1 | 730.4 | 599.6 | 111.5    | 100.5     | 9727     | 841,772   | 0.993  |
| `screen_n` 6 (32 / 3 / 2)           | 822.1 | 734.6 | 600.1 | 111.8    | 104.8     | 9727     | 853,772   | 0.993  |

The first version of this table passed no `--finalists` (the CLI's own default of 3 overrides
`PrepSettings.finalists`), so its `finalists` 2 row changed nothing; rows 3 and 4 were re-measured
with `python -m benchmarks.prep_timing --finalists 2` (`team_keep` and `variant_rounds` are not CLI
options, so they were set by temporarily editing the `PrepSettings` defaults).

With the defaults (`screen_n` 8, first row) the cold run misses the 900 s target by about 64 s
(964.2 s), and none of the three reductions in the plan's order (`team_keep`, `finalists`,
`variant_rounds`) met it (best 923.6 s). The teams phase is 740-780 s of those runs: round 0 screens
the whole team space (3509 setups) and every halving round costs about as much again, all
proportional to `screen_n`. Lowering `screen_n` to 6 brings the cold run to 822.1 s, under the
target, but the funnel then lost the best deathless setup: rung 0 dropped from 1.000 to 0.993 (the
chosen setup had Furret with Fury Swipes by Move Relearner and 10 HP instead of Monferno with two
relearned moves); `screen_n` 4 gives 636.8 s cold and 0.992. By controller ruling (Task 13) the
default stays at `screen_n` 8: the user ranks a safer answer over about a minute of run time, and the
spec's anti-goal is never to return a worse answer silently. **The Roark cold time therefore misses
the 900 s target by about 64 s (~964 s).** Faster round-0 screening (the teams phase) is follow-up
work. The row `screen_n` 4 has no warm time: that run died with a Windows access
violation (exit 3221226505) after the cold run; the intermittent crash is the one in the Task 13 ruling.
Cold minus warm is the 85-90 s compile cost. Two earlier attempts of the first measurements died with
a Windows access violation (exit 3221225477) in the teams phase and were repeated; the table holds the
completed runs.

Chosen setup on the defaults run: PSYDUCK (party slot 5, unchanged), MONFERNO (party slot 2: Mach
Punch, Leer, Fury Swipes by Move Relearner for one Heart Scale, Rock Smash by HM06), SPINARAK
(party slot 4, unchanged); rung 0 value 1.000 (interval 0.995 - 1.000, from the plan); cost 1
Heart Scale and HM06.
