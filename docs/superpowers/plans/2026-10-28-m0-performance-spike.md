# Milestone 0: Performance Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a single masked JAX `step` with ~100 data-driven effects compiles and runs fast enough on CPU to justify the engine design in the spec, before any real engine code exists.

**Architecture:** A throwaway synthetic battle state (same shapes and dtypes the real engine will use) and a `step` that applies 100 fake effects as masked selects, exactly as the real engine will. A benchmark harness jits and vmaps it, measures compile time and steps per second at several batch sizes, and writes a JSON result. The project scaffold (uv, pytest, ruff) is created here because every later plan needs it.

**Tech Stack:** Python 3.12, uv, `jax[cpu]`, numpy, pytest, ruff.

**Spec:** `docs/design.md` §6 (Performance, "Milestone 0 spike"), §8 milestone 0, §4.2 (state dtypes, masked phases).

## Global Constraints

- Python 3.12, managed by `uv` (spec §7).
- `jax[cpu]` only on native Windows; no CUDA (spec §3 Hardware).
- State arrays are `int8`/`int16` for storage, `int32` only where arithmetic needs it; **no 64-bit promotion** anywhere (spec §4.2, §5 JAX hygiene).
- No Python control flow inside `jit`; fixed loop bounds with masking (spec §6).
- Package name `nuzline`, source under `src/nuzline/` (spec §7).
- Go/no-go thresholds (this plan fixes them, spec leaves them open): at batch 10 000 on CPU, **compile time ≤ 120 s** and **≥ 50 000 state-steps/s**. Rationale: 10 000 playouts × 30 turns = 300 000 steps, so 50 000/s means ~6 s per action evaluation, which fits "seconds per turn".

## Review Focus

1. A batch size of 1 must still run and produce the same per-element result as inside a large batch (users will run single-state evaluation in the UI). Test added in Task 2.
2. An effect id outside the implemented range must leave the state unchanged, not index out of bounds (JAX clamps indices silently). Test added in Task 2.
3. Re-running the benchmark must not report the first run's compile as steady-state throughput (the harness must warm up). Test added in Task 3.
4. Dtype promotion to int64 under `jax_enable_x64` left at its default must not occur; a single `int64` literal in an effect poisons the whole graph. Test added in Task 2.
5. Results written with the same batch sizes twice must not overwrite the previous run (the file name carries a timestamp). Test added in Task 3.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/nuzline/__init__.py`
- Create: `tests/test_scaffold.py`
- Create: `README.md`

**Interfaces:**
- Produces: an installable `nuzline` package; `uv run pytest` and `uv run ruff check .` work from the repo root.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
def test_package_imports():
    import nuzline
    assert nuzline.__version__ == "0.0.1"

def test_jax_is_cpu_and_32bit():
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    assert jnp.arange(3).dtype == jnp.int32
```

- [ ] **Step 2: Create `pyproject.toml`**

`[project]` name `nuzline`, version `0.0.1`, `requires-python = ">=3.12,<3.13"`, dependencies `jax[cpu]>=0.4.35`, `numpy>=2.0`. `[dependency-groups] dev = ["pytest>=8", "hypothesis>=6", "ruff>=0.6"]`. `[tool.pytest.ini_options] testpaths = ["tests"]`. `[tool.ruff] line-length = 100, target-version = "py312"`. Build backend `hatchling` with `packages = ["src/nuzline"]`.

- [ ] **Step 3: Create `src/nuzline/__init__.py` with `__version__ = "0.0.1"` and a `README.md` with the one-paragraph purpose from spec §1 and the commands `uv sync`, `uv run pytest`.**

- [ ] **Step 4: Run `uv sync` then `uv run pytest -v`**

Expected: both tests PASS.

- [ ] **Step 5: Run `uv run ruff check .`**

Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/nuzline/__init__.py tests/test_scaffold.py README.md
git commit -m "chore: scaffold nuzline package with uv, pytest, ruff, jax[cpu]"
```

---

### Task 2: Synthetic state and masked step

**Files:**
- Create: `benchmarks/__init__.py` (empty)
- Create: `benchmarks/spike_step.py`
- Test: `tests/test_spike_step.py`

**Interfaces:**
- Produces:
  - `class SpikeState(NamedTuple)` with fields `hp: int16[2,6]`, `stats: int16[2,6,6]`, `stages: int8[2,8]`, `status: int8[2,6]`, `active: int8[2]`, `turn: int16[]`, `field: int8[8]`. Unbatched shapes; batching is added by `vmap`.
  - `make_state(seed: int, batch: int) -> SpikeState` (batched leading axis `batch`).
  - `N_EFFECTS: int = 100`.
  - `step(state: SpikeState, effect_id: int8[2]) -> SpikeState` — unbatched; applies, for each side, the effect `effect_id[side]` via masked selects over all `N_EFFECTS` effects, then an end-of-turn phase (hp -= 1 where status != 0), then `turn + 1`.
  - `batched_step = jax.jit(jax.vmap(step))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spike_step.py
import jax, jax.numpy as jnp, numpy as np
from benchmarks.spike_step import SpikeState, make_state, step, batched_step, N_EFFECTS

def test_step_keeps_shapes_and_dtypes():
    s = make_state(0, 4)
    out = batched_step(s, jnp.zeros((4, 2), jnp.int8))
    for a, b in zip(s, out):
        assert a.shape == b.shape and a.dtype == b.dtype

def test_no_64bit_anywhere():
    s = make_state(0, 4)
    out = batched_step(s, jnp.zeros((4, 2), jnp.int8))
    assert all(x.dtype.itemsize <= 4 for x in out)

def test_batch_of_one_matches_element_of_batch():
    big = make_state(1, 8)
    ids = jnp.full((8, 2), 7, jnp.int8)
    out_big = batched_step(big, ids)
    one = jax.tree.map(lambda x: x[3:4], big)
    out_one = batched_step(one, ids[3:4])
    for a, b in zip(out_big, out_one):
        np.testing.assert_array_equal(np.asarray(a[3]), np.asarray(b[0]))

def test_unknown_effect_id_is_noop_except_end_of_turn():
    s = make_state(2, 1)
    s = s._replace(status=jnp.zeros_like(s.status))
    out = batched_step(s, jnp.full((1, 2), N_EFFECTS + 5, jnp.int8))
    np.testing.assert_array_equal(np.asarray(out.hp), np.asarray(s.hp))
    assert int(out.turn[0]) == int(s.turn[0]) + 1

def test_each_effect_changes_something():
    s = make_state(3, 1)
    for k in range(N_EFFECTS):
        out = batched_step(s, jnp.array([[k, k]], jnp.int8))
        assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(s, out))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_spike_step.py -v`
Expected: FAIL with `ModuleNotFoundError: benchmarks.spike_step`.

- [ ] **Step 3: Implement `benchmarks/spike_step.py`**

Effects are generated, not hand-written: effect `k` is `(field, delta)` chosen deterministically from `k` (e.g. `k % 4` picks hp/stats/stages/status, `(k // 4) % 7 - 3` is the delta, applied to the acting side's active slot). Apply with `jnp.where(effect_id[side] == k, modified, state_field)` inside a Python `for k in range(N_EFFECTS)` loop (unrolled at trace time; that is the point of the spike). Clamp hp at 0 with `int16`. Use `jnp.int8`/`jnp.int16` literals via `.astype` so no int32 creeps into storage arrays.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_spike_step.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/__init__.py benchmarks/spike_step.py tests/test_spike_step.py
git commit -m "feat(spike): synthetic masked step with 100 generated effects"
```

---

### Task 3: Benchmark harness and go/no-go record

**Files:**
- Create: `benchmarks/run_spike.py`
- Create: `benchmarks/README.md`
- Test: `tests/test_run_spike.py`

**Interfaces:**
- Produces:
  - `measure(batch: int, n_steps: int, seed: int = 0) -> dict` with keys `batch`, `compile_s` (float), `steps_per_s` (float, state-steps = batch × n_steps / wall time after warm-up), `n_steps`.
  - `run(batches: list[int], n_steps: int, out_dir: Path) -> Path` — runs `measure` per batch, writes `out_dir / f"spike_{YYYYMMDD_HHMMSS}.json"` with `{"jax_version", "platform", "results": [...]}`, prints a table, returns the path.
  - `main()` CLI: `uv run python -m benchmarks.run_spike --batches 1000 10000 100000 --steps 30`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_spike.py
import json
from benchmarks.run_spike import measure, run

def test_measure_separates_compile_from_steady_state():
    r = measure(batch=64, n_steps=5)
    assert r["compile_s"] > 0 and r["steps_per_s"] > 0
    assert r["batch"] == 64 and r["n_steps"] == 5

def test_run_writes_unique_file_each_call(tmp_path):
    p1 = run([16], 2, tmp_path)
    p2 = run([16], 2, tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()
    data = json.loads(p1.read_text())
    assert data["results"][0]["batch"] == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_spike.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `benchmarks/run_spike.py`**

`measure`: build state, call `batched_step` once with `.block_until_ready()` and time it as `compile_s`; then time `n_steps` further calls (feeding the output back in, `block_until_ready` on the last) for `steps_per_s`. Timestamp in the file name uses seconds so two calls in the same second must append a counter suffix (`_1`, `_2`) rather than overwrite.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_spike.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run the real spike**

Run: `uv run python -m benchmarks.run_spike --batches 1000 10000 100000 --steps 30`
Expected: a table and a file under `benchmarks/results/`. Record the batch-10 000 row.

- [ ] **Step 6: Write `benchmarks/README.md`**

Contents: how to run, the thresholds from Global Constraints, the measured batch-10 000 numbers, and one line: **GO** or **NO-GO**. If NO-GO, the README names the first mitigation to try from spec §4.2 "Per-fight specialization" (reduce `N_EFFECTS` to 30 and re-measure to confirm specialization would rescue it).

- [ ] **Step 7: Commit**

```bash
git add benchmarks/run_spike.py benchmarks/README.md tests/test_run_spike.py
git commit -m "feat(spike): benchmark harness and milestone 0 go/no-go record"
```
