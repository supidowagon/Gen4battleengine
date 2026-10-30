# Ranking Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** bring `nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --n 2000` from 11.6 min to under the 60 s CPU target recorded in plan M1b Task 8, without changing any result the engine or the trainer AI computes.

**Architecture:** three independent levers measured on 2026-10-29 (scratch profile, i5-13600K, 20 logical CPUs, n = 200 playouts per action, 9 legal actions):

| measurement | value |
|---|---|
| `_playouts` trace + lower | 23-27 s |
| `_playouts` XLA compile | 25-26 s |
| run, exact 3840-outcome opponent distribution per turn, 1 device | 51-103 s |
| run, opponent distribution from 32 sampled outcomes, 1 device | 3.9 s |
| same, keys sharded over 6 / 12 / 20 CPU devices | 1.5 / 1.3 / 1.3 s |
| playout turns: mean / max | 8.6 / 18 |

(1) The player's playout policy calls `ai_action_distribution`, which runs the AI policy on all 3840 `AiOutcome` combinations every turn of every playout: over 95 % of run time. The distribution factorizes exactly (Task 1), so it needs 32 scorings. (2) XLA on CPU runs the whole batch on one device; sharding the playout keys over host CPU devices gives about 3x (Task 2). (3) Every run re-traces and re-compiles for about 50 s; a persistent compilation cache plus a serialized-trace cache removes both on repeat runs (Task 3). Task 4 measures and records.

**Tech Stack:** Python 3.12 (uv), JAX CPU (`jax[cpu]`), pytest, Hypothesis.

**Spec:** `docs/design.md` (§4.2 Trainer AI, §4.3 Stage A, §6 Performance: "Persistent compilation cache. JAX's persistent cache is enabled so per-fight specializations survive restarts; cache keys include the table version.").

## Global Constraints

- No result changes: the trainer AI (`ai_policy`, `enumerate_ai_outcomes`, `sample_ai_outcome`), the engine `step`, and the playout policy's decision rule stay as they are. The opponent distribution stays exact.
- `ai_action_distribution` after Task 1 equals the 3840-outcome enumeration on every state the tests generate, to `atol=1e-6`, including NaN on exactly the same (refused) states.
- Common random numbers are preserved: `rank_actions` uses `jax.random.split(key, n)` as today, and a given `(state, n, key, max_turns)` gives the same ranking whatever the device count.
- No new runtime dependencies.
- Decomp fidelity rules of M1 still bind: the decomp (`external/pokeplatinum`, commit 579d2d5c6) is authoritative for AI behavior; cite `trainer_ai.c` lines for any AI claim.
- Test output must be pristine; the full suite and `uv run ruff check .` pass before every commit.
- All CLI output stays plain ASCII.

## Review Focus

1. A state where the AI's scoring is refused (`_refused(...).scoring`) or its switch layout is refused: the factored distribution must be NaN exactly when the enumeration is (Task 1 tests).
2. Tie-break probabilities: `RandNext % 12` is not uniform (65536 = 12 * 5461 + 4), so tie shares must come from `_TIEBREAK_P`, not 1/n (Task 1 test with 3 tied slots).
3. `n` not divisible by the device count (e.g. `--n 2001` on 20 devices): padding must be dropped and results must equal the single-device run (Task 2 test).
4. A stale trace cache after a source edit or a table rebuild: the key must change, never load an old program (Task 3 test).
5. `XLA_FLAGS` already set by the user: the CLI must not clobber it (Task 2 test).

## Out of scope

Chunked playouts with compaction and racing (spec §4.3, §6) are not in this plan. They are the next levers if Task 4's measurement misses the target; Task 4 records that.

---

### Task 1: Exact factored opponent distribution

**Files:**
- Modify: `src/nuzline/ai/policy.py` (`_distribution`, `ai_action_distribution`), `src/nuzline/ai/outcome.py` (export what Task 1 needs: the tie-break and switch-stop probability helpers)
- Test: `tests/ai/test_policy.py` (or a new `tests/ai/test_distribution.py`)

**Interfaces:**
- Consumes: `_scores(tables, state, ai_outcome)`, `switch_action(tables, state, stop)`, `switch_stop_probs(switch_sites(tables, state))`, `script_roll_thresholds(state)`, `_use_item`, `_refused`, `_pick`'s tie rule, `_TIEBREAK_P`.
- Produces: `ai_action_distribution(tables, state) -> float32[NUM_OPP_ACTIONS]` (same signature and contract, now factored); `ai_action_distribution_enumerated(tables, state)`, the old 3840-outcome implementation kept as the documented test oracle (traced-safe, `jax.jit`).

Why it is exact: `_policy` is `switch if switch != NO_SWITCH else item if item != NO_ITEM else move`. `maybe_switch` reads only `switch_stop`; `_use_item` reads no draw; `_scores` reads only `ability_guess`, `script_rolls` and `sim_roll` (fixed at 15) (`ai/commands.py:308,349,456,471`); `_pick` reads `tiebreak` only through `tiebreak % n` over the `n` tied slots. So:

```
dist = sum_s p_switch[s] * onehot(switch_action(s))          over stops s with switch_action(s) != NO_SWITCH
     + P_noswitch * ( onehot(item)                                         if item != NO_ITEM
                      sum_{g, r} p_g * p_r * tieshare(scores(g, r))        otherwise )
P_noswitch = sum of p_switch[s] over stops with switch_action(s) == NO_SWITCH
tieshare(scores)[slot] = sum over t in 0..11 with t % n == rank(slot among tied) of _TIEBREAK_P[t], 0 for untied slots
```

`g` ranges over the 2 ability guesses (1/2 each), `r` over the 16 script-roll bit patterns with `prod(where(bit, p_jump, 1 - p_jump))`, exactly as `enumerate_ai_outcomes` weights them; the move slot when no move is usable is 0 (Struggle), as in `_pick`. NaN rule: NaN exactly when the enumeration's `any(actions < 0)` would be true, which includes probability-0 combinations: `_refused(tables, state).any`, or `switch_action` negative for any of the `SWITCH_SITES + 1` stops.

- [ ] **Step 1: Write the failing tests.** `test_factored_distribution_matches_enumeration` (Hypothesis, `max_examples` >= 200, the AI test strategies already used in `tests/ai/`; states covering Potion use, voluntary-switch layouts, all of Roark's leads): `np.testing.assert_allclose(ai_action_distribution(T, s), ai_action_distribution_enumerated(T, s), atol=1e-6)` and NaN positions equal. `test_factored_distribution_refused_states_nan`: a scoring-refused state and a switch-refused state give all-NaN in both. `test_tie_shares_use_tiebreak_probabilities`: a state where 3 slots tie, shares equal `_TIEBREAK_P` summed over residues (computed in the test from `_TIEBREAK_P`, not 1/3). `test_distribution_sums_to_one` on unrefused states (atol 1e-6).
- [ ] **Step 2: Run them; the first fails** because `ai_action_distribution_enumerated` does not exist.
- [ ] **Step 3: Implement** the factored `_distribution` (vmap over the 32 `(g, r)` combinations and the `SWITCH_SITES + 1` stops only) and rename the old body to `ai_action_distribution_enumerated` (keep `_check` refusal raising on concrete states in the public function as today).
- [ ] **Step 4: Run** `uv run pytest tests/ai tests/search -q`: all pass; then the full suite.
- [ ] **Step 5: Measure** with a throwaway script (not committed): `jax.jit(jax.vmap(...))` of old vs new over 500 copies of Roark's turn-0 state; put both run times in the report.
- [ ] **Step 6: Commit** `perf(ai): exact factored opponent action distribution (32 scorings instead of 3840)`.

### Task 2: Shard Stage A playouts over CPU devices

**Files:**
- Modify: `src/nuzline/search/stage_a.py`, `src/nuzline/cli.py`
- Test: `tests/search/test_stage_a.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `_playouts(tables, state, actions, keys, max_turns)`.
- Produces: `rank_actions(...)` unchanged signature; when `jax.device_count() > 1` the `[n]` key axis is padded to a multiple of the device count with copies of `keys[0]`, placed with `NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))`, and the padded columns are dropped before any statistic. `nuzline.cli.configure_cpu_devices(count: int | None, environ=os.environ) -> None`: unless `environ["XLA_FLAGS"]` already contains `--xla_force_host_platform_device_count`, appends `--xla_force_host_platform_device_count=<count or os.cpu_count()>` to `XLA_FLAGS` (keeping any existing flags). `main()` calls it before anything imports `jax` (assert this in a test with a subprocess: `python -c "import nuzline.cli, sys; assert 'jax' not in sys.modules"`). New `rank` option `--devices N` (default: `os.cpu_count()`); `--devices 1` keeps today's single-device path.

- [ ] **Step 1: Failing tests.** `test_configure_cpu_devices_appends_flag`, `test_configure_cpu_devices_keeps_user_flag` (existing `--xla_force_host_platform_device_count=3` untouched), `test_cli_module_does_not_import_jax`. `test_rank_actions_same_on_multiple_devices`: a subprocess with `XLA_FLAGS=--xla_force_host_platform_device_count=4` runs `rank_actions` on Roark's turn-0 state with `n=6` (not a multiple of 4), `max_turns=3`, key 0, prints the `ActionResult` tuples as JSON; the in-process single-device result with the same arguments must be identical.
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite and ruff pass.**
- [ ] **Step 5: Commit** `perf(search): shard Stage A playout keys over host CPU devices`.

### Task 3: Persistent compilation cache and trace cache

**Files:**
- Create: `src/nuzline/jaxcache.py`
- Modify: `src/nuzline/search/stage_a.py`, `src/nuzline/cli.py`, `tests/conftest.py`, `.gitignore`
- Test: `tests/test_jaxcache.py`

**Interfaces:**
- Produces:
  - `cache_root(environ=os.environ) -> Path`: `$NUZLINE_CACHE_DIR` if set, else `<repo root>/.cache` when running from a source checkout (the directory containing `pyproject.toml` above the package), else `~/.cache/nuzline`.
  - `table_version(tables_dir) -> str`: first 16 hex chars of sha256 of `manifest.json` bytes plus `tables.npz` bytes in that directory.
  - `code_version() -> str`: first 16 hex chars of sha256 over every `*.py` under the `nuzline` package (sorted relative paths and bytes), plus `jax.__version__` and `jaxlib.__version__`.
  - `enable_compile_cache(version: str) -> Path`: sets `jax_compilation_cache_dir` to `cache_root()/"jax"/version`, `jax_persistent_cache_min_compile_time_secs` to 1.0, and returns the directory. Spec §6: the key includes the table version.
  - `cached_export(fn, args, *, name: str, static: dict, version: str)`: returns a callable equal to `jax.jit(fn)` on `args`; the `jax.export.Exported` of `fn` at `args`' shapes, dtypes and shardings is serialized to `cache_root()/"export"/version/<name>-<sha16 of (static, avals, sharding, device count, code_version())>.bin` on first use and deserialized on later runs, so tracing is skipped. A deserialization failure deletes the file and falls back to exporting again (never a crash, never a stale program).
- `rank_actions` gets `_playouts` through `cached_export`; the CLI calls `enable_compile_cache(table_version + code_version)` before ranking. `tests/conftest.py` enables the compile cache (only the XLA compilation cache, not the export cache) under `cache_root()/"jax-tests"/...` so reruns of the suite skip XLA compilation. `.gitignore` gains `.cache/`.

- [ ] **Step 1: Failing tests.** `test_cache_root_env_override`, `test_table_version_changes_with_tables` (tmp copies, one byte changed), `test_code_version_changes_with_source` (monkeypatched package dir in tmp), `test_cached_export_roundtrip` (a small jitted function: first call writes one `.bin`, a fresh `cached_export` call loads it without calling `fn`'s Python body, asserted with a call counter, same outputs), `test_cached_export_key_changes_with_static_and_shape`, `test_cached_export_corrupt_file_recovers`.
- [ ] **Step 2: Run; they fail.**
- [ ] **Step 3: Implement.** If `jax.export` cannot serialize the sharded `_playouts` in this JAX version, export the unsharded function and apply sharding at call time; report what you found.
- [ ] **Step 4: Measure** in the report: two consecutive runs of `uv run nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --n 200` with an empty `NUZLINE_CACHE_DIR`: wall time of each. The second run's time spent before the first playout result must be under 10 s.
- [ ] **Step 5: Full suite and ruff pass. Commit** `perf: persistent compilation cache and serialized-trace cache for Stage A`.

### Task 4: Measure and record

**Files:**
- Create: `benchmarks/rank_timing.py`
- Modify: `benchmarks/README.md`

- [ ] **Step 1:** `benchmarks/rank_timing.py [--n 2000] [--devices N]` runs `nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --n N` twice in subprocesses with a fresh temporary `NUZLINE_CACHE_DIR`, prints cold and warm wall times, and writes them with the machine (CPU model, logical CPUs, jax version) to `benchmarks/results/rank_<timestamp>.json`.
- [ ] **Step 2:** Run it at `--n 2000`. Record in `benchmarks/README.md` a table: before (11 min 37 s, M1b Task 8), cold after, warm after, and the 60 s target, with the date and machine. If either number misses 60 s, add a line naming the next levers (chunked playouts with compaction, racing; spec §4.3, §6) with the measured mean / max playout turns.
- [ ] **Step 3: Commit** `bench: record nuzline rank timing after the speed work`.
