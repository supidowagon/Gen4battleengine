# L1 Line Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A simulated player that follows a per-opposing-Pokemon line, a search that finds a good line for a team, and `nuzline line odds|find`.

**Architecture:** A line is six `int32[6]` arrays indexed by opposing party slot, so lines are data: one compiled JAX program plays any line and `vmap` runs many lines on the same keys. Host-side Python does the search (coordinate ascent on common random numbers), the TOML line file and the CLI. The trainer AI, `step` and the chance model are not touched.

**Tech Stack:** Python 3.12, JAX (CPU, host devices), pytest, uv. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-12-l1-line-player-design.md` (read it first; this plan does not repeat its §3 turn rule).

## Global Constraints

- Battle rules come from the decomp (`external/pokeplatinum`), cited by file and line, never from memory. L1 adds player choices only; if a task finds it needs a battle rule, stop and report.
- Do not change `step`, the AI (`nuzline.ai`), `sample_outcome`, or what `playout`, `playout_policy`, `replacement_policy` return. `tests/search/test_k0_regression.py` and `tests/search/test_playout.py` must stay green untouched.
- Mid-fight level-ups are a known gap owned by K3d: do not model them.
- The user's saves are never copied or committed. Tests and benchmarks get the path from `tests/conftest.py`'s `kaizo_save_pre_roark_path` fixture or `tools.rom.inputs.load_inputs().kaizo_save_pre_roark`, and skip without it.
- Never create, change or delete anything in a folder named `backup`. Never push. Do not merge.
- All CLI output is plain ASCII (`cli._say`). `nuzline.cli` must not import jax at module import (`tests/test_cli.py::test_cli_module_does_not_import_jax`).
- Test cadence (user ruling 2026-11-10): an implementer runs only the test files its task touches or names. The full suite runs once per task alongside the review, and once at the milestone close. Mark as `@pytest.mark.slow` any test that compiles a full playout program or needs a save.
- `ruff check` and `ruff format --check` clean on touched files (line length 100).
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Never call a fight "solved" in output or docs: `find` returns "a good line found by search".

## Review Focus

1. A rule whose action is illegal this turn (answer trapped, locked into a move, move out of PP, Taunt on `first`): the playout must fall back, never play an illegal action. Task 1.
2. Both actives down at once: the replacement must answer the Pokemon the AI is about to send, not the fainted one. Task 1.
3. A line file naming a species the opponent has twice, a team member twice, a move the Pokemon does not know, or a foe not in the fight: one-line error and exit 1, no traceback. Task 3.
4. `first` against a foe that can never be statused: the playout still ends (PP runs out, then `move`); the turn cap is the backstop. Task 1.
5. The search's reported odds must come from keys the search never used, and the same seed must give the same line. Task 4.

---

## File structure

| file | responsibility |
|------|----------------|
| `src/nuzline/search/line.py` (new) | `Line`, `line_action`, `line_replacement`, `line_playout`: the traced player |
| `src/nuzline/search/playout.py` (modify `_choose`, `_policy`) | optional mask of allowed actions |
| `src/nuzline/search/player_eval.py` (new) | `evaluate_lines`, `PlayerOdds`: batched, sharded scoring on shared keys |
| `src/nuzline/search/line_search.py` (new) | candidates, coordinate ascent, hold-out |
| `src/nuzline/search/line_text.py` (new) | the TOML line file: team and rules in, rules out. No jax import |
| `src/nuzline/cli.py` (modify) | `nuzline line odds|find` |
| `benchmarks/line_roark.py`, `benchmarks/lines/kaizo_roark_user.toml` (new) | the Kaizo Roark benchmark and the user's line |
| `tests/search/test_line.py`, `test_player_eval.py`, `test_line_search.py`, `test_line_text.py`, `tests/test_cli_line.py`, `tests/kaizo/test_line_roark.py` (new) | tests |

---

### Task 1: The traced line player

**Files:**
- Create: `src/nuzline/search/line.py`, `tests/search/test_line.py`
- Modify: `src/nuzline/search/playout.py` (`_choose`, `_policy`)

**Interfaces:**
- Consumes: `playout._policy`, `_play`, `send_replacements`, `replacement_pass`, `replacement_policy`, `_carried_mul`, `_select`, `root_state`, `_resources_left`; `actions.legal_actions`, `legal_replacements`, `MOVE_ACTIONS`; `ai.replacement.post_ko_switch_in`; `step.needs_replacement`.
- Produces:

```python
class Line(NamedTuple):          # each int32[6], indexed by opposing party slot; -1 = unset
    use: jnp.ndarray             # player's party slot that answers the foe
    move: jnp.ndarray            # its move slot 0-3
    first: jnp.ndarray           # move slot played while the foe's status == 0
    via: jnp.ndarray             # party slot to switch through
    via_from: jnp.ndarray        # via applies only when this slot is active
    wait: jnp.ndarray            # 1 = never hard-switch the answer in

def empty_line() -> Line                                   # numpy arrays, all -1, wait 0
MODE_FALLBACK, MODE_RULE, MODE_MOVES_ONLY = 0, 1, 2
def line_action(tables, state, line) -> tuple[jnp.ndarray, jnp.ndarray]   # (action int32, mode int32)
def line_replacement(tables, state, line) -> jnp.ndarray                  # party slot int32
class LineResult(NamedTuple):    # PlayoutResult's eight fields, then:
    fallback_turns: jnp.ndarray  # int16[], turns decided by the fallback (modes 0 and 2)
def line_playout(tables, state, line, key, max_turns: int = 100, layout: AiLayout = VANILLA) -> LineResult
```

- `playout._choose(tables, state, q, key, allowed=None)` and `playout._policy(tables, state, key, layout=VANILLA, allowed=None)`: `allowed` is a `bool[NUM_ACTIONS]` mask ANDed with `legal_actions`; `None` means all. If the mask leaves no legal action, ignore the mask.

- [ ] **Step 1: Write the failing tests** in `tests/search/test_line.py` (fixtures `tables`, `names`; states from `tests.engine.helpers.roark_state`, which is the sample party against Geodude, Onix, Cranidos). One test per line of spec §3's turn rule and replacement rule, on concrete states, calling `jax.jit`-free `line_action` / `line_replacement`:

  - `test_no_rule_is_fallback`: `empty_line()` gives mode `MODE_FALLBACK`.
  - `test_answer_active_plays_move`: rule `use=0, move=2` against foe slot 0 with slot 0 active gives `(2, MODE_RULE)`.
  - `test_first_until_status`: with `first=1` the action is 1 while `state.status[1, 0] == 0` and 2 after `state.with_status(1, 0, Status.POISON)`.
  - `test_first_without_pp_plays_move`: `first`'s PP set to 0 (`with_pp`) gives `move`.
  - `test_answer_on_bench_switches_in`: `use=1` with slot 0 active gives `(MOVE_ACTIONS + 1, MODE_RULE)`.
  - `test_via_then_answer`: `use=1, via=2`: slot 0 active gives `MOVE_ACTIONS + 2`; with slot 2 active (`state._replace(active=...)`) gives `MOVE_ACTIONS + 1`.
  - `test_via_from_narrows`: `via_from=3` with slot 0 active goes straight to `MOVE_ACTIONS + 1`; with slot 3 active goes to `MOVE_ACTIONS + 2`.
  - `test_fainted_via_is_skipped`, `test_fainted_answer_is_fallback` (`replace_hp(0, slot, 0)`).
  - `test_wait_keeps_the_active_in`: `use=1, wait=1` with slot 0 active gives mode `MODE_MOVES_ONLY`.
  - `test_illegal_rule_action_is_fallback` (Review Focus 1): the active trapped (`state._replace(trapped=state.trapped.at[0].set(2))`) with the answer on the bench gives `MODE_FALLBACK`; the answer's `move` at 0 PP gives `MODE_FALLBACK`.
  - `test_replacement_sends_the_answer`: the player's active at 0 HP, rule for the foe's active with a living bench `use` returns that slot; with `use` fainted returns `replacement_policy(tables, state)`.
  - `test_replacement_answers_the_incoming_foe` (Review Focus 2): both actives at 0 HP; the returned slot is the `use` of the rule for `post_ko_switch_in(tables, state, _carried_mul(state))`, and differs from the rule for the fainted foe (give the two rules different answers).
  - `test_choose_mask_default_unchanged`: `_policy(tables, s, key)` equals `_policy(tables, s, key, allowed=jnp.ones(NUM_ACTIONS, bool))` on three states; with `allowed` = moves only the action is `< MOVE_ACTIONS`.
  - `test_empty_line_equals_playout` (slow): for 8 keys, `line_playout(tables, s, empty_line(), k, 60)` has the same `won`, `player_faints`, `turns`, `deaths` as `playout(tables, root_state(tables, s, k), k, 60)`, and `fallback_turns == turns`.
  - `test_line_changes_the_first_turn` (slow): a 1v1 state and a line `use=0, move=3` for foe 0: `line_playout(..., max_turns=1)` leaves the PP of move 3 one lower than full and `fallback_turns == 0`. (Return the final state from an inner `_line_run` so the test can read PP; `line_playout` wraps it.)
  - `test_unstatusable_foe_still_ends` (slow, Review Focus 4): a line whose `first` is a status move the foe is immune to (pick from the sample party; if none exists, give the lead Thunder Wave by `roark_state(player_moves=[...])` against Geodude) ends with `turns <= max_turns` and `ended == 1` or `turns == max_turns`.

- [ ] **Step 2: Run them** — `uv run pytest tests/search/test_line.py -q`. Expected: import error for `nuzline.search.line`.

- [ ] **Step 3: Add the mask to `_choose` and `_policy`** as in Interfaces. `_run` and every existing caller keep calling without it.

- [ ] **Step 4: Implement `line.py`.** `line_action` follows spec §3 steps 1-5 with `jnp.where` (no Python branching on traced values); index with `jnp.maximum(x, 0)` wherever a field may be -1. `line_playout` is `playout._run`'s loop with: `root_state` first; a pending player replacement resolved by `line_replacement`; per turn the same `jax.random.split(key, 3)` as `_run`; the fallback computed once per turn as `_policy(..., allowed=moves_only if mode == MODE_MOVES_ONLY else all)`; a refused AI distribution refuses the turn only when the fallback's action is used; after `_play`, `send_replacements`, then a `while_loop` of `replacement_pass` with `line_replacement` (at most `SLOTS - 1` passes). Module docstring states that these are search heuristics, not battle rules, and names the spec.

- [ ] **Step 5: Run** `uv run pytest tests/search/test_line.py tests/search/test_playout.py tests/search/test_k0_regression.py -q -m "not slow"`, then the slow tests of `test_line.py` only: `uv run pytest tests/search/test_line.py -q`. Expected: all pass.

- [ ] **Step 6: Commit** — `feat(search): line-shaped playout player`.

---

### Task 1b: A pluggable player (scope addition, 2026-11-12)

The managing session's addition: later players (exact solve, look-ahead search, a learned network) must drop in without touching the playout loop and be comparable by number. So the loop Task 1 wrote moves behind one small interface.

**Files:**
- Create: `src/nuzline/search/player.py`, `tests/search/test_player.py`
- Modify: `src/nuzline/search/line.py` (the loop moves out; `line_playout` becomes a thin wrapper), `tests/search/test_line.py` (only names that moved)

**Interfaces:**
- Consumes: Task 1's `line_action`, `line_replacement`, the loop of `line_playout`.
- Produces, in `player.py`:

```python
class Player(NamedTuple):        # static and hashable: passed to jit as a static argument
    name: str
    act: Callable                # (tables, state, params, key, layout) -> (action int32, refused bool, fallback bool)
    replace: Callable            # (tables, state, params) -> party slot int32

class PlayerResult(NamedTuple):  # PlayoutResult's eight fields, then fallback_turns (int16)
    ...

GREEDY: Player                   # playout_policy and replacement_policy; params ignored (pass ()); fallback always True
def player_playout(tables, state, player: Player, params, key, max_turns: int = 100,
                   layout: AiLayout = VANILLA) -> PlayerResult
```

  `params` is any pytree of arrays (traced; a batch of them can be `vmap`ped). `refused` from `act` refuses the turn (the playout stops with `refused = 1`, as today). `fallback` marks a turn the player handed to the default policy; `fallback_turns` counts them. `player_playout` draws the root's hidden values first and uses the per-turn key splits of `playout._run`.
- In `line.py`: `LINE: Player` (`params` is a `Line`), `LineResult = PlayerResult` is dropped in favour of `PlayerResult`, and `line_playout(tables, state, line, key, max_turns, layout)` is `player_playout(tables, state, LINE, line, key, max_turns, layout)`.

- [ ] **Step 1: Write the failing tests** in `tests/search/test_player.py`:
  - `test_greedy_equals_playout` (slow): for 8 keys `player_playout(tables, s, GREEDY, (), k, 60)` equals `playout(tables, root_state(tables, s, k), k, 60)` on `won`, `player_faints`, `turns`, `deaths`.
  - `test_a_new_player_plugs_in` (slow): a test-local `Player` whose `act` always returns action 0 (not refused, not fallback) and whose `replace` is `replacement_policy`; a 1v1 playout of `max_turns=3` ends with `fallback_turns == 0` and only move slot 0's PP lowered. This is the proof that a new player needs no change to the loop.
  - `test_players_are_hashable` (not slow): `hash(GREEDY)`, `hash(LINE)` work and `GREEDY != LINE`.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_player.py -q`. Expected: import error.
- [ ] **Step 3: Implement** by moving the loop; no behaviour change.
- [ ] **Step 4: Run** `uv run pytest tests/search/test_player.py tests/search/test_line.py -q`. Expected: all pass, Task 1's tests included.
- [ ] **Step 5: Commit** — `refactor(search): playout loop behind a Player interface`.

---

### Task 2: Scoring players on shared keys

**Files:**
- Create: `src/nuzline/search/player_eval.py`, `tests/search/test_player_eval.py`

**Interfaces:**
- Consumes: Task 1's `Line`, `line_playout`, `LineResult`; `stage_a.wilson`; `deaths.success`, `deaths.bucket`, `deaths.DeathDistribution`; `jaxcache.cached_export`, `code_version`; `ai.layout.layout_of`; `ai.policy.UnsupportedAi`.
- Produces:

```python
@dataclass(frozen=True)
class PlayerOdds:
    n: int
    wins: tuple[int, int, int, int]      # playouts won with at most k deaths, k = 0..3 (no protected loss)
    wins_any: int                        # playouts won
    deaths_total: int                    # player faints summed over the n playouts
    distribution: DeathDistribution
    mean_turns: float
    fallback_share: float                # fallback turns / turns
    def rung(self, k: int) -> float              # wins[k] / n
    def interval(self, k: int | None) -> tuple[float, float]   # Wilson; None = any win
    def score(self, k: int) -> tuple[int, int, int]            # (wins[k], wins_any, -deaths_total)

def stack_lines(lines: Sequence[Line]) -> Line            # leading axis L, numpy
def evaluate_players(tables, state, player: Player, params: Sequence, n: int, key,
                     max_turns: int = 100) -> list[PlayerOdds]   # one PlayerOdds per params entry
def evaluate_lines(tables, state, lines: Sequence[Line], n: int, key, max_turns: int = 100) -> list[PlayerOdds]
    # = evaluate_players(tables, state, LINE, lines, n, key, max_turns)
def line_keys(key, n: int)                                # jax.random.split(key, n)
```

  `evaluate_players` plays every `params` entry of one `player` (Task 1b's `Player`; `params` entries share one pytree shape and are stacked) on the same `n` keys; "line" below means one `params` entry. The export's `name` and `static` include `player.name`, so each player compiles its own program. For `GREEDY` pass `params=[()]`. The program runs over flat rows, one row per (line, key) pair (`L * n` rows, row `i * n + j` = line `i` on key `j`), chunked, padded with copies of the chunk's row 0 and sharded over the host devices exactly as `leaves._run` does, through `cached_export`. Two row-chunk sizes only, so at most two programs compile: `SMALL_CHUNK = 256` when `L * n <= 1024`, else `ROW_CHUNK = 2048` (each rounded up to a multiple of the device count). The padding is dropped, so the result does not depend on the device count or on how many lines are passed. Register `Line` and `LineResult` with `jax.export.register_namedtuple_serialization`. Any refused playout raises `UnsupportedAi(f"{count} of {total} line playouts refused; run `nuzline coverage` for the fight")`.

- [ ] **Step 1: Write the failing tests** (`pytestmark = pytest.mark.slow`; a small 2v2 `roark_state(player_count=2, opp_count=2)`, `max_turns=40`, `n=64`):
  - `test_matches_line_playout`: `evaluate_lines(tables, s, [empty_line(), rule_line], 64, key)` gives, per line, `wins_any` equal to the count from `jax.vmap(line_playout)` over `line_keys(key, 64)`.
  - `test_greedy_matches_the_empty_line`: `evaluate_players(tables, s, GREEDY, [()], 64, key)[0]` has the same `wins`, `wins_any` and `deaths_total` as `evaluate_lines(tables, s, [empty_line()], 64, key)[0]` (same seeds for every player).
  - `test_same_line_same_odds`: the same line passed twice gives equal `PlayerOdds`; passing 1 line or 3 lines gives the same odds for the shared line (padding does not leak).
  - `test_ladder_is_monotone`: `wins[0] <= wins[1] <= wins[2] <= wins[3] <= wins_any <= n`; `interval(k)` contains `rung(k)`.
  - `test_refused_playout_raises`: a state whose AI is refused (reuse the state `tests/search/test_playout.py::test_refused_playout_cannot_be_counted_as_a_loss` builds) raises `UnsupportedAi`.
  - `test_score_orders_by_rung_then_wins` (not slow, pure dataclass): hand-built `PlayerOdds`.

- [ ] **Step 2: Run** `uv run pytest tests/search/test_player_eval.py -q`. Expected: import error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: pass.
- [ ] **Step 5: Commit** — `feat(search): score lines on shared playout keys`.

---

### Task 3: The line file

**Files:**
- Create: `src/nuzline/search/line_text.py`, `tests/search/test_line_text.py`

**Interfaces:**
- Consumes: `data.tables.Names` (`species_names`, `move_names`, `item_names`), `data.spec.PokemonSpec`, `saves` (`SaveMon.location`, `.nickname`, `.is_egg`, `.to_spec(tables)`), Task 1's `Line`/`empty_line` shapes (build numpy arrays; do not import jax here: import `Line` lazily or define the arrays through a small helper in `line.py` that is only called, not imported at module top).
- Produces:

```python
@dataclass(frozen=True)
class Rule:                      # names resolved to indices
    against: int                 # opposing party slot
    use: int                     # player's party slot
    move: int                    # move slot 0-3
    first: int = -1
    via: int = -1
    via_from: int = -1
    wait: bool = False

@dataclass(frozen=True)
class LineFile:
    team: tuple[dict, ...]       # the [[team]] tables as read (pokemon, moves?, item?), () when absent
    rules: tuple[dict, ...]      # the [[rule]] tables as read

def read_line_file(path: Path) -> LineFile
def team_from_save(file: LineFile, save, tables, names) -> list[PokemonSpec]
def resolve_rules(file: LineFile, names, player: Sequence[PokemonSpec], opp: Sequence[PokemonSpec]) -> list[Rule]
def rules_to_line(rules: Sequence[Rule]) -> Line
def line_to_rules(line: Line) -> list[Rule]
def render_rules(rules: Sequence[Rule], names, player, opp) -> list[str]   # TOML [[rule]] tables
def write_line_file(path: Path, file: LineFile, rules_toml: Sequence[str]) -> None
```

  Format and checks: spec §6. `team_from_save`: `pokemon` is a species name, nickname, `party:N` or `box:B:S` (the labels of `save show`: box 1-based; reuse the parsing in `prep.problem.pick` if it can be called without building a prep problem, else mirror its syntax and messages); eggs cannot be named; `moves` replaces all four slots with full PP (`pp=None`), `item` replaces the held item (`""` = none); 1-6 entries, no Pokemon twice. `resolve_rules`: `against` is an opposing species name (case-insensitive) or `slot:N` (0-based); a species the opponent has twice needs `slot:N`; `use`, `via`, `from` name a team member by species or 0-based `slot:N`. Every failure is a `ValueError` whose message names the file's table (`rule 3 (against Gible): ...`).

- [ ] **Step 1: Write the failing tests** (not slow; `tables`, `names` fixtures; player from `load_party(SAMPLE_PARTY, names)`, opponent from `tests.engine.helpers` Roark party; write TOML to `tmp_path`):
  - `test_round_trip`: rules -> `rules_to_line` -> `line_to_rules` -> `render_rules` -> re-read -> `resolve_rules` gives the same rules.
  - `test_user_shape_parses`: a file with `first`, `via`, `from` resolves to the expected indices.
  - Review Focus 3, one test each, asserting `pytest.raises(ValueError, match=...)`: unknown foe species; foe species held twice without `slot:N` (build an opponent list with a duplicate); unknown team member; a move the Pokemon does not know; `from` without `via`; `wait = true` with `via`; two rules for one foe; `via` equal to `use`.
  - `test_team_from_save` (skips without the vanilla sample save fixture `tests/test_cli.py` uses, see its `save_path` fixture): a `[[team]]` naming `party:0` with an `item` override returns a spec with that item; a `moves` list with an unknown move raises.
  - `test_module_does_not_import_jax`: `subprocess` `python -c "import nuzline.search.line_text, sys; assert 'jax' not in sys.modules"`.

- [ ] **Step 2: Run** `uv run pytest tests/search/test_line_text.py -q`. Expected: import error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same. Expected: pass.
- [ ] **Step 5: Commit** — `feat(search): line file (team and rules as TOML)`.

---

### Task 4: The search

**Files:**
- Create: `src/nuzline/search/line_search.py`, `tests/search/test_line_search.py`

**Interfaces:**
- Consumes: Task 1 `Line`, `empty_line`; Task 2 `evaluate_lines`, `PlayerOdds`; `engine.damage.damage_table`; the tables' move effect ids (`engine.ids`).
- Produces:

```python
@dataclass(frozen=True)
class LineSearchSettings:
    screen_n: int = 128
    keep: int = 4
    refine_n: int = 512
    max_sweeps: int = 4
    restarts: int = 1
    final_n: int = 2000
    budget_seconds: float | None = None
    max_turns: int = 100

@dataclass(frozen=True)
class LineSearchResult:
    line: Line
    odds: PlayerOdds               # hold-out: keys the search never used
    sweeps: int
    evaluations: int             # line x key playouts run, hold-out included
    stopped: str                 # "converged", "max_sweeps" or "budget"

def who_what_candidates(tables, state, foe: int) -> list[tuple[int, int, int]]   # (use, move, first)
def entry_candidates(state, use: int) -> list[tuple[int, int]]                  # (via, wait)
def find_line(tables, names, state, key, settings=LineSearchSettings(), start: Line | None = None,
              progress: Callable[[str], None] | None = None) -> LineSearchResult
```

  Procedure: spec §5, with these pins. Keys: `fold_in(key, c)` with a counter `c` that increases by one per evaluation batch, so screening, refining, the restart comparison and the hold-out never share keys; the hold-out uses `fold_in(key, 0x1E1D)`. A candidate replaces the incumbent only if its `score(k)` (k = `int(state.faint_budget)`) on the refine keys is strictly greater than the incumbent's on the same keys. `who_what_candidates`: `use` over living player slots; `move` over that slot's move slots with `tables.move_power > 0` and a positive roll in `damage_table` against the foe with `use` active (build the facing state with `playout._facing`); `first` is -1 or a move slot of power 0 whose effect id is one of the engine's major-status effects. Find those ids in `nuzline.engine.ids` / the tables (the same ids `step` reads to apply sleep, poison, toxic, burn, paralysis); name them in one module constant with a comment saying where each comes from; do not key on move names. `entry_candidates`: `(-1, 0)`, `(-1, 1)`, and `(v, 0)` for every other living slot `v`. Random restart lines draw each foe's rule uniformly from "no rule" plus its who/what candidates with plain entry, from `numpy.random.default_rng` seeded by `jax.random.key_data(fold_in(key, 0xBEEF))`. `start`, when given, is the first start instead of the empty line. `progress` gets one line per finished sweep (`sweep 2: 61.3 % at <=1, 78.9 % win (512 playouts)`).

- [ ] **Step 1: Write the failing tests:**
  - `test_who_what_candidates` (not slow): on `roark_state` every `(use, move, first)` has a living `use`, a damaging `move`, and `first` either -1 or a power-0 slot; a slot with a Ground move never offers it against a Flying-type foe (construct with `roark_state(opp_species=...)`), and an immune-only Pokemon yields no candidate.
  - `test_entry_candidates` (not slow): excludes `use` itself and fainted slots.
  - `test_find_line_is_no_worse_than_fallback` (slow): a 3v2 state, `LineSearchSettings(screen_n=32, keep=2, refine_n=64, max_sweeps=2, restarts=0, final_n=128, max_turns=40)`; then `evaluate_lines(tables, s, [empty_line(), result.line], 256, fresh_key)`: the found line's `score(k)` is at least the empty line's minus a 5-playout slack.
  - `test_find_line_is_deterministic` (slow, Review Focus 5): two runs with the same key give equal `line` arrays and equal `odds`.
  - `test_hold_out_keys_are_fresh` (not slow, Review Focus 5): monkeypatch `line_search.evaluate_lines` with a recorder returning fixed odds; no key passed during the search equals the hold-out key, and `result.odds` is the recorder's return for the hold-out call.
  - `test_budget_stops_the_search` (not slow, same monkeypatch plus a fake clock): `budget_seconds=0` returns `stopped == "budget"` with the start line and still a hold-out evaluation.
  - `test_start_line_is_used` (not slow, monkeypatch): the first evaluation batch contains the `start` line.

- [ ] **Step 2: Run** `uv run pytest tests/search/test_line_search.py -q`. Expected: import error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same. Expected: pass.
- [ ] **Step 5: Commit** — `feat(search): coordinate-ascent search over lines`.

---

### Task 5: `nuzline line odds` and `nuzline line find`

**Files:**
- Modify: `src/nuzline/cli.py` (parser, two command functions, module docstring, `main`'s device set-up tuple and command table)
- Create: `tests/test_cli_line.py`

**Interfaces:**
- Consumes: Tasks 2-4; `cli._fight`'s pieces (`_game`, `load_game`, `_load_trainers`, `_load_save`, `resolve_trainer`, `_check_buildable`, `build_trainer_party`, `start_weather`), `protected_slots`, `require_coverage`, `make_state`, `enable_compile_cache`.
- Produces: `nuzline line odds` and `nuzline line find` exactly as spec §6. Flags: common `--game`, `--party P | --save S [--allow-corrupt]`, `--trainer | --trainer-id`, `--line FILE` (required), `--deaths K` (default 0), `--protect`, `--seed 0`, `--devices`; `odds` adds `--n 2000`; `find` adds `--out FILE`, `--budget SECONDS`, `--restarts N`, `--final-n N`, `--sweeps N`. With `--save`, the file must have `[[team]]`; with `--party`, a `[[team]]` in the file is an error. The lead is team slot 0, the opponent's lead slot 0.

  Output of both (after the trainer line and the team, one Pokemon per line with moves and item):

  ```
  line:
    vs Gible: Delibird, Powder Snow, via Rhyhorn from Wormadam
    vs Corsola: Wormadam, Toxic then Leech Life
    vs Onix: (no rule: default player)
  2000 playouts, seed 0, budget <=3 deaths
    win, 0 deaths     1.5 % [ 1.0,  2.1]
    win, <=1 deaths  41.7 % [39.6, 43.9]
    win, <=2 deaths  ...
    win, <=3 deaths  ...
    win, any         68.2 % [66.1, 70.2]
  <plan_tree.distribution_line(odds.distribution)>
  mean turns 23.0; default player decided 4 % of turns
  ```

  `find` prints `[line] ...` progress to stderr, then `found by search (a heuristic: a good line, not a proven best)`, the same block from the hold-out sample, and the `[[rule]]` TOML; `--out` gets the input's `[[team]]` plus the found rules. Exit codes: 0; 1 for a bad file, name or flag (`ValueError`); 2 for a refused fight (`require_coverage`) or refused playouts (`UnsupportedAi`).

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_line.py` (copy `tests/test_cli.py`'s autouse fixtures for tables and XLA flags):
  - `test_line_odds_prints_the_ladder` (slow): `--party tests/fixtures/sample_party.toml --trainer roark --line <tmp rule file> --n 32 --devices 1` exits 0; stdout has the five ladder rows and is ASCII.
  - `test_line_odds_bad_rule_exit_1`: unknown `against` exits 1 with one `nuzline: error:` line on stderr, before anything compiles (monkeypatch `player_eval.evaluate_lines` to fail if called).
  - `test_line_odds_team_table_needs_save`: `--party` with a `[[team]]` file exits 1.
  - `test_line_find_writes_out` (monkeypatch `line_search.find_line` to return a fixed result): `--out` file exists, re-reads with `read_line_file`, and holds the found rules.
  - `test_line_find_refused_exit_2` (monkeypatch `find_line` to raise `UnsupportedAi`).
  - `test_line_commands_set_devices`: `main(["line", "odds", ..., "--devices", "3"])` sets the XLA flag (as `test_rank_devices_option_sets_the_flag`).
  - `tests/test_cli.py::test_cli_module_does_not_import_jax` still passes.

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_line.py -q`. Expected: argparse error exit for the unknown command.
- [ ] **Step 3: Implement**, including the two entries in the module docstring.
- [ ] **Step 4: Run** `uv run pytest tests/test_cli_line.py tests/test_cli.py -q -m "not slow"` then `uv run pytest tests/test_cli_line.py -q`. Expected: pass.
- [ ] **Step 5: Commit** — `feat(cli): nuzline line odds and line find`.

---

### Task 6: Kaizo Roark benchmark, numbers and docs

**Files:**
- Create: `benchmarks/line_roark.py`, `benchmarks/lines/kaizo_roark_user.toml`, `tests/kaizo/test_line_roark.py`
- Modify: `benchmarks/README.md` (new section at the end), `docs/design.md` (§8 item 4a: one sentence under sub-item 2 pointing at the spec and the benchmark), `README.md` (usage of the two commands, if it lists commands)

**Interfaces:**
- Consumes: the CLI internals of Task 5 (call the Python functions, not a subprocess), `tools.rom.inputs.load_inputs().kaizo_save_pre_roark`.
- Produces: `uv run python -m benchmarks.line_roark [--n 2000] [--deaths 3] [--seed 0] [--devices N] [--budget SECONDS]`, writing `benchmarks/results/line_roark_<time>.json` (git-ignored) and printing: the user's line's ladder, the found line (TOML) and its hold-out ladder, both lines on one shared set of `--n` keys, the search's wall time and playout count, and the fallback (`empty_line`) ladder on the same keys.

  `benchmarks/lines/kaizo_roark_user.toml` is the user's line: team in order Prinplup (item Quick Claw), Wormadam (moves: its save moves with Toxic in the empty slot: `["Leech Life", "Nature Power", "Camouflage", "Toxic"]`), Rhyhorn, Delibird, Sandshrew (item Oran Berry); rules: Bonsly -> Prinplup Aqua Cutter; Lileep -> Wormadam Leech Life; Gible -> Delibird Powder Snow via Rhyhorn from Wormadam; Cranidos -> Sandshrew Dig; Corsola and Shuckle -> Wormadam Leech Life, first Toxic. It holds species, move and item names only, nothing from the save.

- [ ] **Step 1: Write `tests/kaizo/test_line_roark.py`** (slow; skips without `kaizo_save_pre_roark_path` or the Kaizo tables): `line odds` internals on the user's file with `--deaths 3`, 400 playouts, seed 0: any-win in [0.60, 0.80], at-most-one-death in [0.32, 0.52]; the empty line on the same keys wins less than 0.25. (The managing session measured 72 % / 42.5 % at 200 playouts and a prototype measured 68.2 % / 41.7 % at 400; the bands are those figures with sampling error.)
- [ ] **Step 2: Run it.** Expected: fails only if Tasks 1-5 mis-play the line; if it fails, stop and report the numbers.
- [ ] **Step 3: Write `benchmarks/line_roark.py`** and run it: `uv run python -m benchmarks.line_roark --deaths 3` and again with `--deaths 1` (background; each may take tens of minutes). 
- [ ] **Step 4: Judge the result against spec §7.** The found line must be at least the user's line on both any-win and at-most-one-death, on the shared keys. If it is not: report the numbers and what the search chose; do not retune silently. The controller decides the search change.
- [ ] **Step 5: Record** in `benchmarks/README.md`: machine, commit, command, the three ladders with intervals, the found line, wall time, playouts, settings used. Update `docs/design.md` and `README.md`.
- [ ] **Step 6: Commit** — `bench(line): Kaizo Roark line odds and search`.
- [ ] **Step 7: Commit the found line** as `benchmarks/lines/kaizo_roark_found.toml` (same team table, the found rules, a comment with the command and seed that found it) in the same commit.

---

### Task 7: The player scoreboard (scope addition, 2026-11-12)

Later players must be comparable to today's by number: one command scores every registered player on a fixed list of fights, same seeds for all.

**Files:**
- Create: `benchmarks/players.py`, `benchmarks/fights.toml`, `tests/test_players_benchmark.py`
- Modify: `benchmarks/README.md` (section "Player scoreboard"), `docs/design.md` (§8 4a: one sentence naming the `Player` interface and the scoreboard as the yardstick for later players)

**Interfaces:**
- Consumes: Task 1b `Player`, `GREEDY`, `LINE`; Task 2 `evaluate_players`, `PlayerOdds`; Task 3 line files; Task 5's fight set-up callable.
- Produces:

  `benchmarks/fights.toml`, one table per fight:

      [[fight]]
      name = "kaizo-roark-user-five"
      game = "kaizo"
      save = "kaizo_save_pre_roark"      # a key of data/local.toml (tools.rom.inputs), never a path
      trainer_id = 246
      team = "lines/kaizo_roark_user.toml"   # its [[team]] table is the setup
      deaths = 3

  `benchmarks/players.py`:

  ```python
  @dataclass(frozen=True)
  class Entry:                   # one scoreboard row
      label: str                 # "greedy", "user line", "found line"
      player: Player
      params: Callable           # (fight context) -> one params pytree for this fight, or None = not applicable

  ENTRIES: list[Entry]           # greedy (GREEDY, ()), user line (LINE, rules of the fight's team file),
                                 # found line (LINE, rules of lines/<fight's found file>, None when absent)
  def scoreboard(fights, entries, n: int, seed: int) -> list[dict]   # fight, label, n, PlayerOdds fields
  ```

  A fight names its found-line file with an optional `found = "lines/kaizo_roark_found.toml"`. `uv run python -m benchmarks.players [--n 2000] [--seed 0] [--devices N] [--fight NAME]` prints, per fight, one row per entry: label, win % and win-with-at-most-k % for k = 0..3, each with its 95 % Wilson interval, fallback share; every entry on the same keys (`jax.random.PRNGKey(seed)`). A fight whose save is not configured is skipped with one line saying so. Results also go to `benchmarks/results/players_<time>.json` (git-ignored). Adding a later player is one `Entry`.

- [ ] **Step 1: Write the failing tests** in `tests/test_players_benchmark.py`:
  - `test_fights_file_parses` (not slow): `benchmarks/fights.toml` loads, every `team`/`found` path exists, every `save` is a key `tools.rom.inputs` knows, none looks like a path.
  - `test_scoreboard_rows` (not slow): monkeypatch `evaluate_players` with a recorder returning fixed `PlayerOdds`; a fake fight context; rows come back one per applicable entry, every call got the same key and `n`, and an entry whose `params` returns `None` is left out.
  - `test_scoreboard_on_roark` (slow, skips without the save): `--n 200` on `kaizo-roark-user-five`: greedy wins less than the user line; the user line's any-win is in [0.58, 0.82].
- [ ] **Step 2: Run** `uv run pytest tests/test_players_benchmark.py -q`. Expected: import error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests, then the scoreboard** `uv run python -m benchmarks.players --n 2000` (background).
- [ ] **Step 5: Record** the table for the three players (greedy, the user's scripted line, the found line) in `benchmarks/README.md` with machine, commit, command and seed.
- [ ] **Step 6: Commit** — `bench(players): scoreboard of players on fixed fights`.
