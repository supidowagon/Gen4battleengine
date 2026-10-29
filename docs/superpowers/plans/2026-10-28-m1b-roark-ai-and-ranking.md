# Milestone 1b: Roark AI, Coverage Registry, Stage A Ranking and Golden Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Roark fight playable end to end from the command line: a decomp-accurate Roark AI with explicit random draws and knowledge state, a coverage registry that refuses unsupported mechanics, Stage A Monte Carlo ranking of the player's actions, and a first emulator-generated golden trace.

**Architecture:** The AI is a pure JAX function fed an `AiOutcome` of its own random draws, mirroring the decomp's script interpreter command by command for the flags Roark uses. The coverage registry is a static list of implemented ids checked before any search. Stage A batches one-ply-policy playouts under `vmap` with common random numbers and reports P(win with zero faints) with Wilson intervals. A Lua script dumps battle RAM from DeSmuME to produce replayable traces.

**Tech Stack:** Python 3.12, uv, `jax[cpu]`, numpy, pytest, hypothesis; DeSmuME with Lua scripting for traces; a Platinum (US) ROM and save the user supplies (never committed).

**Spec:** `docs/design.md` §4.2 (Trainer AI, Coverage registry), §4.3 (Stage A), §5 (AI tests, golden traces, chance-enumeration tests), §8 milestone 1. Requires plan `2026-10-28-m1a-data-and-engine.md` complete.

## Global Constraints

- The AI mirrors the decomp's script interpreter for the trainer's AI flags; it never uses heuristics of its own (spec §4.2 Trainer AI).
- The AI's random draws are inputs (`AiOutcome`), never drawn inside `ai_policy` (spec §4.2).
- The AI scores only the player moves it has seen and guesses an unrevealed ability (spec §4.2); knowledge lives in `BattleState.ai_seen_moves` / `ai_ability_known` from plan 1a Task 6.
- Coverage: refuse loudly and name what is missing, with a per-candidate check in prep search (spec §3). In this plan the check is per fight.
- Stage A: playouts end at the first player faint or battle end or a hard cap of 100 turns; opponent follows `ai_policy`; actions compared at a node share outcome seeds (spec §4.3, §6).
- Reported odds carry a 95 % Wilson interval (spec §4.3).
- The ROM, saves and traces containing ROM data are never committed; `tests/golden/*.json` holds only extracted battle state, not ROM bytes.

## Review Focus

1. When every player action has P(win, 0 faints) = 0, the ranking must still be deterministic and the output must say so rather than present noise as a preference. Test in Task 7.
2. An AI "seen moves" update must happen only for moves actually executed (not for a flinched or fully paralysed attempt that never revealed the move). Test in Task 4.
3. The AI's ability guess for a two-ability player species must be a chance node (50/50) and must collapse once the ability reveals itself (Intimidate on switch-in reveals it). Test in Task 4.
4. `check_coverage` must report every missing mechanic in one error, not stop at the first. Test in Task 5.
5. A player team with fewer than six members and an opponent that switches must never index an empty slot in the playout. Test in Task 6.

---

### Task 1: Transcribe Roark's AI from the decomp

**Files:**
- Create: `docs/ai/roark.md`
- Create: `docs/ai/script_commands.md`

**Interfaces:**
- `docs/ai/roark.md` produces: Roark's `ai_flags` value and the named flags it decodes to; for each flag, the script it runs; the ordered list of script commands those scripts use; every point at which a script reads the RNG, with its range; what the AI reads about the player (seen moves, ability resolution rule, item visibility); the switching rule and the item-use rule as they apply to Roark (he holds no items in the gym fight unless the decomp says otherwise); the tie-break rule among top-scored moves.
- `docs/ai/script_commands.md` produces one row per script command used by Roark's flags: `command | arguments | semantics | RNG? | JAX function name`. The JAX function names are decided here and used in Task 3.

- [ ] **Step 1: Read the AI interpreter and scripts** at the paths in `docs/decomp_paths.md` and write both documents.

- [ ] **Step 2: Verify completeness**

Run: `grep -c '^|' docs/ai/script_commands.md`
Expected: one row per command used by Roark's scripts; cross-check by grepping each command name in the decomp scripts Roark's flags select.

- [ ] **Step 3: Commit**

```bash
git add docs/ai
git commit -m "docs(ai): transcribe Roark's AI flags, scripts, RNG draws and knowledge rules"
```

---

### Task 2: AI outcome struct and sampler

**Files:**
- Create: `src/nuzline/ai/__init__.py` (empty)
- Create: `src/nuzline/ai/outcome.py`
- Test: `tests/ai/test_ai_outcome.py`

**Interfaces:**
- `class AiOutcome(NamedTuple)`: `sim_roll int8[4]` (the AI's simulated damage roll per opponent move, range from `docs/ai/roark.md`), `ability_guess int8[]` (0/1 slot when the player's ability is unknown), `script_rolls int8[MAX_SCRIPT_ROLLS]` (one per RNG read point, in script order; `MAX_SCRIPT_ROLLS` is fixed from `docs/ai/roark.md`), `tiebreak int8[]` (index among tied top moves).
- `sample_ai_outcome(key, state) -> AiOutcome` with the ranges from the document.
- `enumerate_ai_outcomes(state) -> tuple[AiOutcome (batched), float32[K]]` over all combinations, merged later in Task 3 by resulting action (spec §10 "AI outcome width"). If the product of ranges exceeds 4096, the function raises `AiOutcomeTooWide` and the registry marks the fight sampling-only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ai/test_ai_outcome.py
import jax, jax.numpy as jnp
from nuzline.ai.outcome import AiOutcome, sample_ai_outcome, enumerate_ai_outcomes, MAX_SCRIPT_ROLLS
from tests.engine.helpers import roark_state

def test_sample_within_ranges(tables, names):
    s = roark_state(tables, names)
    o = sample_ai_outcome(jax.random.PRNGKey(0), s)
    assert o.sim_roll.shape == (4,) and o.script_rolls.shape == (MAX_SCRIPT_ROLLS,)
    assert int(o.ability_guess) in (0, 1)

def test_enumeration_probabilities_sum_to_one(tables, names):
    s = roark_state(tables, names)
    outs, probs = enumerate_ai_outcomes(s)
    assert abs(float(probs.sum()) - 1.0) < 1e-5 and outs.sim_roll.shape[0] == probs.shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/test_ai_outcome.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `outcome.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ai/test_ai_outcome.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/ai/__init__.py src/nuzline/ai/outcome.py tests/ai/test_ai_outcome.py
git commit -m "feat(ai): AiOutcome struct with sampler and enumerator"
```

---

### Task 3: AI script commands and policy

**Files:**
- Create: `src/nuzline/ai/commands.py`
- Create: `src/nuzline/ai/policy.py`
- Create: `tests/ai/fixtures.py` (the hand-traced cases `TRACED`)
- Test: `tests/ai/test_policy.py`

**Interfaces:**
- `commands.py`: one JAX function per row of `docs/ai/script_commands.md`, each `(ctx: AiContext, ...) -> AiContext`, where `class AiContext(NamedTuple)` holds `scores int32[4]`, the move under evaluation, the AI's view of the player (seen moves, resolved ability), and the `AiOutcome`. The AI's own damage estimate uses `damage_for_roll` from plan 1a with `sim_roll` per move.
- `policy.py`:
  - `ai_scores(tables, state, ai_outcome) -> int32[4]` running Roark's flag scripts in the decomp's order from a base score (the decomp's starting value, documented in `docs/ai/roark.md`).
  - `ai_policy(tables, state, ai_outcome) -> int32` returning the chosen action in plan 1a's encoding: highest score, ties by `ai_outcome.tiebreak`; switching and item use per the rules in `docs/ai/roark.md` (if Roark's flags never switch, the function still routes through a `maybe_switch` that returns "no switch" so later trainers only extend it).
  - `ai_action_distribution(tables, state) -> float32[10]` = enumerate `AiOutcome`s, run `ai_policy` under `vmap`, and sum probabilities by action.

- [ ] **Step 1: Write the failing tests**

Hand-trace at least three states through the decomp scripts while writing `docs/ai/roark.md`, record the expected score vectors there, and pin them:

```python
# tests/ai/test_policy.py
import jax.numpy as jnp
from nuzline.ai.policy import ai_scores, ai_policy, ai_action_distribution
from nuzline.ai.outcome import AiOutcome
from tests.ai.fixtures import TRACED   # list of (description, state_builder, AiOutcome, expected_scores:int32[4], expected_action)

def test_hand_traced_scores(tables, names):
    for desc, build, ai_o, exp_scores, exp_action in TRACED:
        s = build(tables, names)
        assert ai_scores(tables, s, ai_o).tolist() == exp_scores, desc
        assert int(ai_policy(tables, s, ai_o)) == exp_action, desc

def test_distribution_sums_to_one_and_is_legal(tables, names):
    s = TRACED[0][1](tables, names)
    d = ai_action_distribution(tables, s)
    assert abs(float(d.sum()) - 1.0) < 1e-5
    assert float(d[4]) == 0.0            # cannot switch to its own slot

def test_ko_move_preferred_when_available(tables, names):
    s = TRACED[0][1](tables, names).replace_hp(0, 0, 1)
    d = ai_action_distribution(tables, s)
    assert float(d[:4].max()) > 0.9      # a damaging move that KOs dominates under the "try to faint" flag
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/test_policy.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `commands.py` and `policy.py`**

Each command function carries a comment with the decomp command name and file line. Scripts are straight-line in JAX: every command runs, masked by the script's branch conditions, so no Python control flow depends on traced values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ai -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/ai tests/ai
git commit -m "feat(ai): Roark AI script commands, scoring and action distribution"
```

---

### Task 4: AI knowledge updates in the engine

**Files:**
- Modify: `src/nuzline/engine/step.py`
- Modify: `src/nuzline/ai/policy.py`
- Modify: `src/nuzline/reference/engine.py`
- Test: `tests/ai/test_knowledge.py`

**Interfaces:**
- `step` sets `ai_seen_moves[player_slot, move_slot] = 1` when the player's move actually executes (after sleep/paralysis/flinch/confusion checks pass), and `ai_ability_known[player_slot] = 1` when the ability visibly activates (Intimidate on switch-in, Blaze boost does not reveal; the rule per ability is a table `REVEALS_ON` in `hooks.py`).
- `policy.py` gains `resolve_player_ability(tables, state, ai_outcome) -> int32` (the real ability when `ai_ability_known[active]`, else the species' ability in slot `ability_guess`, which is the single ability when the species has one). `ai_scores` ignores player moves with `ai_seen_moves == 0` wherever the script reads the player's moves and uses `resolve_player_ability` wherever it reads the ability.
- The reference engine mirrors both updates.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ai/test_knowledge.py
import jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome
from nuzline.ai.policy import ai_scores
from nuzline.ai.outcome import sample_ai_outcome
import jax
from tests.engine.helpers import roark_state, A

def test_seen_move_recorded_only_when_executed(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_faster=True)
    flinched = step(tables, s, A("Scratch", "Headbutt", s, names), empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8)))
    assert int(flinched.ai_seen_moves[0, 0]) == 0
    executed = step(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert int(executed.ai_seen_moves[0, 0]) == 1

def test_ability_guess_is_chance_until_revealed(tables, names):
    from nuzline.ai.policy import resolve_player_ability
    s = roark_state(tables, names, player_species="Shinx", player_ability="Rivalry")   # Shinx: Rivalry / Intimidate
    o0 = sample_ai_outcome(jax.random.PRNGKey(0), s)
    assert int(s.ai_ability_known[0]) == 0
    guessed = {int(resolve_player_ability(tables, s, o0._replace(ability_guess=jnp.int8(g)))) for g in (0, 1)}
    assert guessed == {names.ability("Rivalry"), names.ability("Intimidate")}
    revealed = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))
    for g in (0, 1):
        assert int(resolve_player_ability(tables, revealed, o0._replace(ability_guess=jnp.int8(g)))) == names.ability("Rivalry")

def test_intimidate_reveals_ability(tables, names):
    s = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert int(s.ai_ability_known[0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/test_knowledge.py -v`
Expected: assertion failures.

- [ ] **Step 3: Implement; re-run the differential suite**

Run: `uv run pytest tests/ai/test_knowledge.py tests/test_differential.py -v`
Expected: all PASS (the reference engine now mirrors the knowledge updates).

- [ ] **Step 4: Commit**

```bash
git add src/nuzline/engine/step.py src/nuzline/engine/hooks.py src/nuzline/ai/policy.py src/nuzline/reference/engine.py tests/ai/test_knowledge.py
git commit -m "feat(ai): seen-move and ability-reveal knowledge tracked in state and used by the AI"
```

---

### Task 5: Coverage registry

**Files:**
- Create: `src/nuzline/engine/coverage.py`
- Test: `tests/engine/test_coverage.py`

**Interfaces:**
- `class Support(IntEnum)`: `NONE=0, SAMPLING_ONLY=1, EXACT=2`.
- `IMPLEMENTED_EFFECTS: dict[EffectId, Support]`, `IMPLEMENTED_ABILITIES: dict[AbilityId, Support]`, `IMPLEMENTED_ITEMS: dict[ItemId, Support]`, `IMPLEMENTED_AI_FLAGS: dict[int, Support]` (bit → support), `IMPLEMENTED_WEATHER: dict[Weather, Support]`.
- `@dataclass class CoverageReport`: `missing: list[str]` (human strings like `"Roark: Metronome (move effect UNSUPPORTED)"`), `sampling_only: list[str]`, `level: Support` (min over everything present).
- `check_coverage(tables, names, player: list[PokemonSpec], trainer: Trainer, opp: list[PokemonSpec], start_condition: int) -> CoverageReport`; `require_coverage(...)` raises `UnsupportedMechanics(report)` whose message lists every missing item.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_coverage.py
import pytest
from nuzline.engine.coverage import check_coverage, require_coverage, UnsupportedMechanics, Support
from tests.engine.helpers import sample_party, roark_trainer, roark_party

def test_roark_vs_sample_party_is_fully_covered(tables, names):
    r = check_coverage(tables, names, sample_party(names), roark_trainer(), roark_party(names), 0)
    assert r.missing == [] and r.level == Support.EXACT

def test_all_missing_reported_at_once(tables, names):
    party = sample_party(names, override_moves={0: ["Metronome", "Scratch", "Leer", "Ember"]}, override_item={1: "Leftovers"})
    with pytest.raises(UnsupportedMechanics) as e:
        require_coverage(tables, names, party, roark_trainer(), roark_party(names), 0)
    msg = str(e.value)
    assert "Metronome" in msg and "Leftovers" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_coverage.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `coverage.py`**

The dicts list exactly the ids implemented in plan 1a Tasks 8a–8e and this plan's Task 3; every entry `EXACT` for now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_coverage.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/coverage.py tests/engine/test_coverage.py tests/engine/helpers.py
git commit -m "feat(engine): coverage registry with exact/sampling-only support levels"
```

---

### Task 6: Playout policy and batched playouts

**Files:**
- Create: `src/nuzline/search/__init__.py` (empty)
- Create: `src/nuzline/search/playout.py`
- Test: `tests/search/test_playout.py`

**Interfaces:**
- `playout_policy(tables, state, key) -> int32`: one-ply expectimax over legal player actions using exact damage tables: score each action by expected P(no player faint this turn) × (1 + fraction of opponent HP removed), using `ai_action_distribution` for the opponent and the 16 rolls × hit probability for both sides; `key` breaks exact ties only. Illegal actions score −∞.
- `class PlayoutResult(NamedTuple)`: `won int8[]`, `player_faints int8[]`, `turns int16[]`.
- `playout(tables, state, key, max_turns: int = 100) -> PlayoutResult`: `lax.while_loop` until `over` or `max_turns`; each iteration: player action from `playout_policy`, opponent action from `ai_policy(sample_ai_outcome)`, outcome from `sample_outcome`, `step`, then `replace` for the opponent (first legal replacement slot) when needed.
- `batched_playout = jax.jit(jax.vmap(playout, in_axes=(None, 0, 0)))` with `max_turns` static.
- `playout_from_action(tables, state, action: int32, key, max_turns) -> PlayoutResult`: applies the given first player action, then `playout`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_playout.py
import jax, jax.numpy as jnp
from nuzline.search.playout import playout, playout_from_action, playout_policy, batched_playout
from nuzline.engine.actions import legal_actions
from tests.engine.helpers import roark_state

def test_policy_returns_legal_action(tables, names):
    s = roark_state(tables, names)
    a = int(playout_policy(tables, s, jax.random.PRNGKey(0)))
    assert bool(legal_actions(tables, s, 0)[a])

def test_playout_terminates_with_result(tables, names):
    s = roark_state(tables, names)
    r = playout(tables, s, jax.random.PRNGKey(1))
    assert int(r.won) in (0, 1) and 0 < int(r.turns) <= 100

def test_batched_playout_shapes(tables, names):
    s = roark_state(tables, names)
    B = 64
    bs = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s)
    r = batched_playout(tables, bs, jax.random.split(jax.random.PRNGKey(2), B))
    assert r.won.shape == (B,) and r.turns.dtype == jnp.int16

def test_short_party_and_opponent_switch_do_not_index_empty_slot(tables, names):
    s = roark_state(tables, names, player_count=2, opp_count=3)
    r = playout(tables, s, jax.random.PRNGKey(3))
    assert int(r.turns) > 0

def test_player_faint_ends_playout_as_loss(tables, names):
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])   # no switch; must Struggle at 1 HP: recoil faints
    r = playout(tables, s, jax.random.PRNGKey(4))
    assert int(r.won) == 0 and int(r.player_faints) == 1 and int(r.turns) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/search/test_playout.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `playout.py`**

`roark_state` gains `player_count`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/search/test_playout.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/search tests/search tests/engine/helpers.py
git commit -m "feat(search): one-ply playout policy and batched playouts"
```

---

### Task 7: Stage A ranking with common random numbers

**Files:**
- Create: `src/nuzline/search/stage_a.py`
- Test: `tests/search/test_stage_a.py`

**Interfaces:**
- `@dataclass class ActionResult`: `action: int, label: str` ("Ember" / "Switch → Starly"), `p_win0: float, ci_low: float, ci_high: float, p_win: float, e_faints: float, n: int`.
- `wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]`.
- `rank_actions(tables, names, state, n: int, key, max_turns: int = 100) -> list[ActionResult]`: for every legal action, `n` playouts via `playout_from_action` with the **same** `n` keys for every action (common random numbers), sorted by `p_win0` desc then `p_win` desc then `action`; ties within 0.005 keep action order. A final field `all_zero: bool` on the returned list wrapper `class Ranking(list)` is `True` when every `p_win0 == 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_stage_a.py
import jax
from nuzline.search.stage_a import rank_actions, wilson
from tests.engine.helpers import roark_state

def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0

def test_ranking_covers_legal_actions_and_is_sorted(tables, names):
    s = roark_state(tables, names)
    r = rank_actions(tables, names, s, n=200, key=jax.random.PRNGKey(0))
    assert len(r) == 9 and all(a.ci_low <= a.p_win0 <= a.ci_high for a in r)
    assert [a.p_win0 for a in r] == sorted([a.p_win0 for a in r], reverse=True)

def test_common_random_numbers_are_deterministic(tables, names):
    s = roark_state(tables, names)
    a = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    b = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    assert [(x.action, x.p_win0) for x in a] == [(x.action, x.p_win0) for x in b]

def test_all_zero_flag(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    s = s._replace(trapped=s.trapped.at[0].set(3))
    r = rank_actions(tables, names, s, n=50, key=jax.random.PRNGKey(1))
    assert r.all_zero is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/search/test_stage_a.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `stage_a.py`**

Batch all actions × `n` keys in one `vmap` call (shape `[A, n]`) so compile happens once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/search/test_stage_a.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/search/stage_a.py tests/search/test_stage_a.py
git commit -m "feat(search): Stage A Monte Carlo ranking with Wilson intervals and common random numbers"
```

---

### Task 8: Command-line interface

**Files:**
- Create: `src/nuzline/cli.py`
- Modify: `pyproject.toml` (`[project.scripts] nuzline = "nuzline.cli:main"`)
- Test: `tests/test_cli.py`

**Interfaces:**
- `nuzline coverage --party <toml> --trainer <name>`: prints the `CoverageReport`; exit code 2 if anything is missing.
- `nuzline rank --party <toml> --trainer <name> [--n 2000] [--seed 0] [--lead 0] [--opp-active 0]`: runs `require_coverage`, builds the state (player lead index `--lead`, opponent active `--opp-active`, start condition from the trainer), prints a table: action label, P(win 0 faints) with CI, P(win), E[faints], and a footer line "all lines 0 %: no zero-faint line found at this depth" when `all_zero`.
- `nuzline trainers`: lists trainer names and ids.
- `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from nuzline.cli import main

def test_coverage_ok(capsys):
    assert main(["coverage", "--party", "tests/fixtures/sample_party.toml", "--trainer", "roark"]) == 0
    assert "missing: none" in capsys.readouterr().out.lower()

def test_rank_prints_nine_rows(capsys):
    assert main(["rank", "--party", "tests/fixtures/sample_party.toml", "--trainer", "roark", "--n", "50"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 9 and "Switch" in out

def test_coverage_failure_exit_code():
    assert main(["coverage", "--party", "tests/fixtures/sample_party_unsupported.toml", "--trainer", "roark"]) == 2
```

`tests/fixtures/sample_party_unsupported.toml` gives Monferno Metronome.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cli.py` with `argparse`**

- [ ] **Step 4: Run tests, then the real command**

```bash
uv run pytest tests/test_cli.py -v
uv run nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --n 2000
```
Expected: tests PASS; the command prints nine ranked rows. Record the wall time in `benchmarks/README.md`; if it exceeds 60 s on CPU, note it there as the first target for the per-fight specialization in spec §4.2.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/cli.py pyproject.toml tests/test_cli.py tests/fixtures/sample_party_unsupported.toml benchmarks/README.md
git commit -m "feat(cli): coverage, rank and trainers commands"
```

---

### Task 9: Emulator trace script and first Roark golden trace

**Files:**
- Create: `tools/trace_battle.lua`
- Create: `tools/trace_to_state.py`
- Create: `docs/tracing.md`
- Create: `tests/golden/roark_01.json` (extracted state only)
- Test: `tests/test_golden.py`

**Interfaces:**
- `tools/trace_battle.lua` (DeSmuME): on each frame, detects turn boundaries via the battle-state RAM addresses listed in `docs/tracing.md` (derived from the decomp's symbol map / `.sym` file for the US ROM), and on each boundary appends a JSON line with: turn number, both sides' per-slot `species, level, hp, max_hp, status, moves, pp, item`, active indices, stat stages, field weather, and the 32-bit RNG state. Output path from a Lua global `TRACE_PATH`.
- `tools/trace_to_state.py`: `trace_to_states(path: Path, tables, names) -> list[tuple[BattleState, tuple[int,int], Outcome]]`: converts each turn's dump to a `BattleState` and infers the realized `Outcome` and both actions from the before/after dumps (hit/miss from HP change and messages, crit/roll by matching against `damage_table`, raising `AmbiguousTrace` when more than one roll fits and recording all candidates).
- `tools/trace_to_state.py` also exports `COMPARED_FIELDS: tuple[str, ...]`, the `BattleState` field names the dump fills; fields it lacks are listed in `docs/tracing.md`.
- `tests/test_golden.py`: for each consecutive pair in the trace, `step(before, actions, outcome)` must equal `after` on every field in `COMPARED_FIELDS`.

- [ ] **Step 1: Write `docs/tracing.md`** with the RAM addresses, how to run DeSmuME with the script, and the exact Roark battle recorded (sample party at the fixture's levels, which the user must prepare in the save; the fixture may be adjusted to whatever the user's save holds, and the adjusted party saved as `tests/fixtures/golden_party_roark_01.toml`).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_golden.py
import pytest
from pathlib import Path
from nuzline.engine.step import step
from tools.trace_to_state import trace_to_states, COMPARED_FIELDS

TRACE = Path("tests/golden/roark_01.json")
pytestmark = pytest.mark.skipif(not TRACE.exists(), reason="no golden trace")

def test_roark_trace_replays(tables, names):
    turns = trace_to_states(TRACE, tables, names)
    assert len(turns) >= 3
    for (before, actions, outcome), (after, _, _) in zip(turns, turns[1:]):
        out = step(tables, before, actions, outcome)
        for name in COMPARED_FIELDS:
            assert getattr(out, name).tolist() == getattr(after, name).tolist(), f"turn {int(before.turn)} field {name}"
```

- [ ] **Step 3: Write the Lua script and `trace_to_state.py`; record one Roark battle; convert it**

Run DeSmuME with the ROM, load the prepared save, run `tools/trace_battle.lua`, fight Roark to the end, then:
`uv run python -m tools.trace_to_state --trace <raw jsonl> --out tests/golden/roark_01.json --party tests/fixtures/golden_party_roark_01.toml`

- [ ] **Step 4: Run the golden test**

Run: `uv run pytest tests/test_golden.py -v`
Expected: PASS. A failure is a real engine discrepancy: fix per the decomp and log in `docs/discrepancies.md`.

- [ ] **Step 5: Run the whole suite and commit**

```bash
uv run pytest -q
uv run ruff check .
git add tools/trace_battle.lua tools/trace_to_state.py docs/tracing.md tests/golden/roark_01.json tests/test_golden.py tests/fixtures/golden_party_roark_01.toml
git commit -m "test: emulator trace script and first Roark golden trace"
```
