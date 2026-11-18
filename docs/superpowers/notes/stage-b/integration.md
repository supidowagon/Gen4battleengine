# Stage B integration map (facts from the code, 2026-10-30, HEAD 6e817e1)

All paths are relative to the repo root `<repo>`.
`src/nuzline/...` is abbreviated as `nuzline/...` in places. Line numbers are from HEAD.
No repo file was edited. Benchmark scripts used for the measurements are in the scratchpad
(`.../scratchpad/stageb/bench_step.py`, `bench_host.py`, `probe.py`).

---

## 0. Headline facts the plan must respect

1. **The battle ends on the first player faint.** `_faint_check` (`src/nuzline/engine/step.py:477-510`)
   sets `over = 1, winner = WINNER_OPPONENT (1)` whenever any player Pokemon fainted in that check
   (`player_fainted > 0`, step.py:496-500), and this wins over a simultaneous last-opponent faint
   (winner chosen by `jnp.where(player_fainted > 0, WINNER_OPPONENT, WINNER_PLAYER)`, step.py:499).
   `step` returns an over state unchanged (step.py:700: `_where(state.over != 0, state, s)`).
   `needs_replacement` therefore only ever returns 1 for the opponent (docstring step.py:703-706).
   Constants: `WINNER_OPPONENT, WINNER_PLAYER = 1, 2` (step.py:73). `BattleState.winner`: 0 none,
   1 player lost, 2 player won (state.py docstring, line ~19).
   => Terminal test for Stage B: `over != 0`; win iff `winner == 2` (playout uses exactly
   `(s.over != 0) & (s.winner == WINNER_PLAYER)`, playout.py:229). With this rule P(win) ==
   P(win, 0 faints) for every state with `player_faints == 0` at the root.
2. **Opponent replacement is not part of `step`.** After `step`, the opponent's fainted active is
   replaced by `send_replacements(tables, state, ai_switched, power_mul)` (playout.py:157-176),
   which needs two per-turn inputs: whether the opponent's action this turn was a switch, and
   `power_mul` (20 if Pursuit fired on the player's switch, else 10). Stage B's enumeration must
   carry these per (state, action pair) and apply `send_replacements` after `step`
   (see section 2.4). Refused replacement => that child is refused.
3. **No `SAMPLING_ONLY` grade exists anywhere** (section 6). A test guards that no registry holds it
   (`tests/engine/test_coverage.py:92-104`); the plan must change that test when it adds the path.
4. **`nuzline.cli` must not import jax at module import** (`tests/test_cli.py:201-211`); all heavy
   imports go inside the command function, as `_rank` does (cli.py:147-154).
5. **Measured**: one `jit(vmap(step))` over 50,000 Roark-start states takes ~3.5 s to compile cold
   and **~72-78 ms per call warm on 1 device (1.44 us/state)**, ~33-37 ms on 20 host devices
   (0.65 us/state). Details in section 13.
6. **`code_version()` hashes every `*.py` under `src/nuzline`** (jaxcache.py:74-84 of the file;
   see section 4), so adding `search/stage_b.py` / `engine/enumerate.py` / `engine/observe.py`
   changes the version and invalidates (and prunes) the compile and export caches, including the
   tests' compile cache (`tests/conftest.py:16-18`). Expect a cold first test run after every
   source edit.

---

## 1. `src/nuzline/search/playout.py` (311 lines)

### Public names
`__all__` (playout.py:63-73): `REFUSED_WON, PlayoutResult, batched_playout, playout,
playout_from_action, playout_policy, refusal_reason, replacement_refusal, send_replacements`.

Constants: `_I32, _F32 = jnp.int32, jnp.float32` (:75); `_NORMAL_POWER_MUL = 10` (:76);
`REFUSED_WON = -1` (:77). `PURSUIT_POWER_MUL = 20` is imported from step.py (step.py:74).

### `PlayoutResult` (playout.py:80-84), a `NamedTuple`
| field | dtype/shape (unbatched) | meaning |
|---|---|---|
| `won` | `int8[]` | 1 = player won, 0 = not, `REFUSED_WON` (-1) = refused |
| `player_faints` | `int8[]` | `s.player_faints` at the end |
| `turns` | `int16[]` | turns played (refused turn not counted) |
| `refused` | `int8[]` | 1 = stopped at a refused mechanic |
Built in `_run` (playout.py:229-235). `jax.export` serialization of it is registered in
stage_a.py:87-88 (`serialized_name="nuzline.PlayoutResult"`).

### Signatures
- `playout_policy(tables: Tables, state: BattleState, key) -> jnp.ndarray` (:149-154): int32 action;
  raises `UnsupportedAi` on a concrete refused state (via `ai_action_distribution`), meaningless
  when traced on a refused state. Internally `_policy` (:140-146) returns `(action, refused)`;
  `_choose` is `@jax.jit` (:129-137), a one-ply expectimax: `values[NUM_ACTIONS, NUM_OPP_ACTIONS] @ q`,
  illegal -> `-inf`, exact ties broken uniformly with `key`.
  **It reads the full concrete state** (opponent exact HP via `damage_table` vs `target.hp`,
  :108-113) — this is the "leaves peek" the spec accepts.
- `send_replacements(tables, state, ai_switched, power_mul) -> tuple[BattleState, jnp.ndarray]`
  (:157-176), `@jax.jit`. A `fori_loop` of `SLOTS - 1` (=5) iterations; each iteration: if
  `needs_replacement(state)[1] == 1`, pick `slot = post_ko_switch_in(tables, state, power_mul)`;
  `bad = need & (ai_switched | slot < 0)`; else `replace(tables, state, 1, slot)`.
  Returns `(state, refused: bool[])`; on refusal the state is left as at that point.
- `_play(tables, state, player_action, key)` (:179-191): samples AI outcome -> `ai_policy`,
  `ai_refused = opp_action < 0`; `actions = stack([player_action, max(opp_action, 0)])` int32[2];
  `pursued = pursuit_fires(tables, state, 0, actions, choices)` where
  `choices = (moves int32[2], struggling bool[2])` from `selected_move` per side;
  `played = step(tables, state, actions, sample_outcome(k_turn, tables, state, actions))`;
  `power_mul = where(pursued, 20, 10)`; returns `(played, ai_refused, is_switch(actions[1]), power_mul)`.
  **This is the exact recipe Stage B must reproduce per (state, player action, opponent action,
  outcome)**: `ai_switched = is_switch(opp_action)`, `power_mul` from `pursuit_fires(..., side=0, ...)`
  evaluated on the *pre-step* state.
- `_turn(tables, state, player_action, key)` (:194-199): `_play` then `send_replacements`;
  returns `(_select(refused, state, played), refused)`.
- `_run(tables, state, key, max_turns: int, first_action)` (:210-235): `lax.while_loop` while
  `s.over == 0 & turns < max_turns & ~refused`. Per turn: `key, k_policy, k_turn = split(key, 3)`;
  policy action unless turn 0 and `first_action >= 0`; `q_refused` (NaN distribution) refuses the
  turn only when the policy's choice is used (:220). Result `won = where(refused, -1, won)`.
- `playout(tables, state, key, max_turns: int = 100) -> PlayoutResult` (:238-241),
  `@partial(jax.jit, static_argnames=("max_turns",))`, `first_action = -1`.
- `playout_from_action(tables, state, action, key, max_turns: int = 100) -> PlayoutResult`
  (:244-250), same jit; the same key gives every first action the same draws (CRN).
- `batched_playout(tables, states, keys, max_turns: int = 100)` (:253-257): `jax.vmap` of
  `playout` over leading axis of `states` and `keys`; `tables` shared; static `max_turns`.
  Test: `tests/search/test_playout.py:40-45` (B=64 broadcast states, `won.shape == (B,)`,
  `turns.dtype == int16`).
- `replacement_refusal(tables, state, ai_switched, power_mul) -> str | None` (:260-277):
  non-jit mirror of `send_replacements` for messages.
- `refusal_reason(tables, state, action, key, max_turns=100) -> str | None` (:285-311): replays
  one playout turn by turn with `_policy_jit`, `_turn_jit`, `_play_jit` (:280-282) and returns
  `"; ".join(ai_refusals(...))`, `"the opponent's action distribution is refused"`, or the
  replacement reason.

### Refusal semantics (module docstring :25-29, code above)
Refusals never raise under jit. A refused playout stops with `refused = 1`, `won = -1`, the
refused turn not played. Sources: (a) `ai_policy` returns `REFUSED_ACTION` (-2) (`opp_action < 0`);
(b) `ai_action_distribution` NaN when the player's policy would use it; (c) `send_replacements`
refuses (replacement in a turn the AI switched — roark.md 14 item 3 — or `post_ko_switch_in`
returns `REFUSED_SLOT` -1 for the stale stage-2 case, ai/replacement.py:115-130).
Callers must exclude refused playouts or refuse loudly (stage_a does the latter).

### Stage B leaf use
Stage B leaves can call `batched_playout(tables, states[B], keys[B], max_turns)` directly
(policy from turn 0 at the leaf), or `playout_from_action` vmapped (stage_a's `_playouts`
pattern). A refused leaf playout must be handled (refuse loudly, or count as a sampled/refused
mass); `won == -1` must never be summed as a loss.

---

## 2. Engine public API for enumeration

### 2.1 `src/nuzline/engine/step.py` (732 lines)
- `step(tables: Tables, state: BattleState, actions, outcome: Outcome) -> BattleState` (:692-700).
  Not decorated with jit (callers jit/vmap it; `tests/test_differential.py:26`:
  `jit_step = jax.jit(step)`; `tests/test_jax_hygiene.py:16`:
  `jax.jit(jax.vmap(step, in_axes=(None, 0, 0, 0)))`). `actions` is `int32[2]` (player, opponent).
  Order: `_record_last_move` -> `action_order` -> `execute_actions` -> `end_of_turn` -> `turn += 1`;
  over state returned unchanged (:700). Pure, draws nothing, fixed shapes, no 64-bit types
  (`tests/test_jax_hygiene.py:24-27`).
- `needs_replacement(state) -> int8[2]` (:703-710): 1 where active fainted, a healthy bench mon
  exists and `over == 0`.
- `replace(tables, state, side, slot) -> BattleState` (:713-732): `_switch_in` + `on_switch_in` +
  `_faint_check(entry_turn=state.turn)`.
- `pursuit_fires(tables, state, side, actions, choices) -> bool` (:429-448): `side` switches, the
  other side chose Pursuit and can act.
- `execute_actions(tables, state, actions, outcome, order)` (:529), `end_of_turn(...)` (:676-689),
  `end_of_turn_order(state, outcome) -> int32[2]` (:663-673).
- Constants: `WINNER_OPPONENT, WINNER_PLAYER = 1, 2` (:73), `PURSUIT_POWER_MUL = 20` (:74),
  `TOXIC_COUNTER_MAX = 15` (:77), `END_OF_TURN_ORDER` (:642-660).
- Faint/terminal: `_faint_check(before, after, entry_turn=None)` (:477-510), see section 0.1.
  Also clears status/counter of every newly fainted mon (:489-495), increments
  `player_faints` (:501).
- Not implemented (docstring :38-44): hail/sandstorm chip, screen/Tailwind/Trick Room countdowns.

### 2.2 `src/nuzline/engine/outcome.py` (193 lines) — the chance layer to enumerate
`Outcome(NamedTuple)` (:44-58), all `int8` after `sample_outcome` (:193):
| field | shape | range / meaning |
|---|---|---|
| `speed_tie` | `[]` | 0 = player first on equal speed |
| `hit` | `[2,5]` | per side, per hit index |
| `crit` | `[2,5]` | |
| `roll` | `[2,5]` | 0-15 (85-100 %) |
| `secondary` | `[2,5]` | attacker's own secondary effect |
| `contact_proc` | `[2,5]` | indexed by the **defending** side |
| `n_hits` | `[2]` | 2-5 |
| `full_para` | `[2]` | |
| `confusion_hit` | `[2]` | |
| `duration` | `[2]` | by the inflicting side; 2-5 sleep/confusion, 3-6 Bind, 3-5 Taunt |
| `thaw` | `[2]` | |
| `flinch_item` | `[2]` | reserved, always 0 (unsupported items) |
| `quick_claw` | `[2]` | reserved, always 0 |
| `end_speed_tie` | `[]` | 0 = player first in end-of-turn order |
`MAX_HITS = 5` (:34).

Exact probabilities used by `sample_outcome(key, tables, state, actions) -> Outcome` (:143-193):
- hit: `randint(0,100)+1 <= hit_threshold` (:155); `hit_threshold(tables, state, actions) -> int32[2]`
  (:125-129), 100 or more never misses; switch/item/accuracy 0 -> 100 (:121-122).
- crit: `randint(0, crit_odds) == 0 & ~lucky_chant(defender)` (:157-158);
  `crit_odds(tables, state, actions) -> int32[2]` (:132-140): 1 in `[16,8,4,3,2][stage]`, only
  `HIGH_CRIT` raises the stage.
- secondary: `randint(0,100) < chance | chance == 0` (:161-162), chance = `tables.move_effect_chance[move]`.
- n_hits: `first = randint(0,4)`, `second = randint(0,4)`, `n = first+2 if first < 2 else second+2`
  (:164-165), so exactly P(2) = P(3) = 3/8, P(4) = P(5) = 1/8.
- duration: Taunt `randint(0,3)+3`; else `randint(0,4) + (3 if TRAP else 2)` (:170-174).
- speed_tie, end_speed_tie: `randint(0,2)` (:176, :191).
- roll: `randint(0,16)` (:179); contact_proc `randint(0,10) < 3` (:181); full_para `randint(0,4)==0`
  (:183); confusion_hit `randint(0,2)==1` (:184); thaw `randint(0,5)==0` (:186).
- Hit and crit odds are read from the **start-of-turn state** (with the switching target swapped
  in, `_target_mon` :88-97); mid-turn-dependent rolls are drawn unconditionally and gated by `step`
  (docstring :11-18). So enumeration can compute all probabilities from the pre-step state.
- `empty_outcome() -> Outcome` (:61-79): all hit, no crit, roll 15, no procs, n_hits 2, duration 2.
  Its fields are **not** cast to int8 via `sample_outcome`'s path — `hits`/`sides` are int8
  (`_I8`) so it is int8 already, `speed_tie` int8.
- `outcome_to_py(outcome) -> Outcome` (:82-85) for the reference engine.

### 2.3 `src/nuzline/engine/actions.py` (81 lines) — legality
- `MOVE_ACTIONS = 4` (:36), `NUM_ACTIONS = MOVE_ACTIONS + SLOTS = 10` (:37),
  `FIRST_ITEM_ACTION = NUM_ACTIONS = 10` (:38), `NUM_OPP_ACTIONS = FIRST_ITEM_ACTION + BAG_SLOTS = 14` (:39).
  Actions 0-3 move slot, 4-9 switch to slot `a-4`, opponent 10-13 bag slot `a-10`.
- `usable_moves(tables, state, side) -> bool[4]` (:42-49).
- `legal_replacements(state, side) -> bool[6]` (:52-56).
- `legal_actions(tables, state, side) -> bool[NUM_ACTIONS]` (:59-65): action 0 legal when
  struggling; switches illegal while `trapped > 0`.
- `usable_trainer_items(tables, state, side) -> bool[BAG_SLOTS]` (:68-73).
- `legal_opponent_actions(tables, state) -> bool[NUM_OPP_ACTIONS]` (:76-81).
- Player legality depends only on observable things (own moves/PP/taunt/trapped/party HP), so it
  is identical across an information set, as the spec assumes.

### 2.4 Other engine helpers used by the turn recipe
- `order.is_switch(action)` (order.py:32-34: `4 <= a < 10`), `is_item(action)` (:37-39: `a >= 10`),
  `selected_move(tables, state, side, action) -> (move int32, struggling bool)` (:42-54),
  `effective_speed(state, side) -> int32` (:57-64),
  `action_order(tables, state, actions, outcome) -> int8[2]` (:67-85).
- `damage.damage_table(tables, state, atk_side, move, crit) -> int32[16]` (damage.py:257-262),
  against the other side's *active* Pokemon (so for a target switching in, build the facing state
  as playout `_facing` does, playout.py:91-100). `damage_for_roll(..., power_mul=10)` (:265-273),
  `ROLLS = 16` (:39), `CRIT_MUL = 2` (:38).
- `state.make_state(tables, player, opp, start_condition=Weather.NONE, opp_items=(), leads=(0,0))
  -> BattleState` (state.py:258-301).

### 2.5 `BattleState` (state.py:60-147), 63 fields, `LAYOUT` (state.py:150-216)
Storage dtypes int8/int16 only. Shapes: per side x slot `[2,6]` (species i16, form i8, level i8,
hp i16, max_hp i16, stats i16[2,6,5], types i8[2,6,2], ability i16, ability_slot i8, item i16,
item_consumed i16, moves i16[2,6,4], pp i8[2,6,4], status i8, status_counter i8, gender i8,
friendship i16, ivs i8[2,6,6]); per side `[2]` (active i8, stages i8[2,8], confusion, leech_seed,
substitute_hp i16, taunt, trapped, trap_move i16, flinched, choice_lock i16, last_move i16,
protect_count, rollout_count, charge_turn, recharge, pursuit_target, water_sport, fake_out_turn
i16, move_hit i16, spikes, toxic_spikes, stealth_rock, reflect, light_screen, safeguard, mist,
lucky_chant, tailwind, wish_timer, wish_hp i16, future_timer, future_damage i16,
trainer_items i16[2,4], trainer_item_count); scalars (weather, weather_timer, trick_room,
gravity, mud_sport, turn i16, over, winner, player_faints); `ai_seen_moves i8[6,4]`,
`ai_ability_known i8[6]`.
**One unbatched state = 752 bytes** (measured, sum of field nbytes). Helpers: `replace_hp`,
`with_stages`, `with_pp`, `with_status` (:149-170 region). Constants `SIDES, SLOTS, MOVES = 2, 6, 4`,
`BAG_SLOTS = 4`, `STAGE_MIN, STAGE_MAX = -6, 6` (:46-48).
Merging note: `turn` increments every step and `fake_out_turn`/`last_move`/`move_hit`/
`ai_seen_moves`/`ai_ability_known` are written by step, so two outcomes merge only when all these
agree too (they normally do within one action pair). The hidden fields for observation:
`status_counter` (sleep), `confusion`, `trapped`, `taunt` counters, opponent exact `hp`.

---

## 3. AI API (opponent chance layer)
`src/nuzline/ai/policy.py`, `__all__` (:71-81): `NO_ITEM, UnsupportedAi, ai_action_distribution,
ai_action_distribution_enumerated, ai_policy, ai_refusals, ai_scores, maybe_switch, maybe_use_item`.
- `class UnsupportedAi(ValueError)` (:91-92).
- `ai_action_distribution(tables, state) -> float32[NUM_OPP_ACTIONS]` (:654-660): exact, factored;
  runs `_check(tables, state, ai_refusals)` first (raises `UnsupportedAi` on a concrete refused
  state; skipped when traced, :457-467), then `_distribution` (`@jax.jit`, :627-651) which returns
  NaN for a refused state. **For batched use, vmap `_distribution` (private) or vmap
  `ai_action_distribution` (traced, so no raise)** and treat NaN rows as refused.
- `ai_action_distribution_enumerated(tables, state)` (:577-590): brute-force oracle over 3840
  outcomes (test oracle).
- `ai_policy(tables, state, ai_outcome) -> int32` (:568-574): raises on concrete refused,
  `REFUSED_ACTION` (-2) traced.
- `ai_refusals(tables, state) -> list[str]` (:447-454).
- `ai/outcome.py`: `AiOutcome` (:135-140: sim_roll i8[4], ability_guess i8[], script_rolls
  i8[MAX_SCRIPT_ROLLS=4], tiebreak i8[], switch_stop i8[]); `ENUMERATION_WIDTH` (:124, = 3840);
  `MAX_ENUMERATED = 4096` (:88); `sample_ai_outcome(key, tables, state)` (:223-236);
  `enumerate_ai_outcomes(tables, state) -> (AiOutcome[K], float32[K])` (:253-270).
- `ai/replacement.py`: `post_ko_switch_in(tables, state, power_mul=10) -> int32` (:115-130),
  `REFUSED_SLOT = -1` (:49); raises `UnsupportedAi` concrete, -1 traced.
- The AI's decision reads hidden-to-player state only through its own inputs; Stage B's
  opponent layer per belief state s is `q = ai_action_distribution(s)`, nonzero entries only.

**Roark start (sample party, real Roark trainer 246)** measured: player legal actions
`[0,1,2,3,5,6,7,8,9]` (9; moves Scratch, Leer, Ember, Mach Punch; Monferno lead);
opponent legal `[0,1,5,6,10,11]`; `ai_action_distribution` = `{0: 0.75 (Stealth Rock), 1: 0.25
(Rock Throw)}` (float32, shape (14,)). Opponent party Geodude, Onix, Cranidos (HP 32/31/43),
bag `(17, 17)` (two Potions). Player HP `[44,44,47,30,37,30]`.

---

## 4. `src/nuzline/jaxcache.py` (207 lines)
`__all__` (:30-38): `cache_root, cached_export, code_version, enable_compile_cache,
prune_other_versions, set_compile_cache_dir, table_version`.
- `cache_root(environ=os.environ) -> Path` (:56-64): `$NUZLINE_CACHE_DIR`, else `<repo>/.cache`
  in a source checkout, else `~/.cache/nuzline`.
- `table_version(tables_dir: Path) -> str` (:67-71): 16 hex of sha256(manifest.json, tables.npz).
- `code_version() -> str` (:74-84), `@functools.cache`: 16 hex over jax/jaxlib versions and every
  `*.py` under the package. New Stage B modules change it.
- `prune_other_versions(directory, root=None)` (:87-108): deletes sibling 16/32-hex dirs.
- `set_compile_cache_dir(directory)` (:111-116): `jax_compilation_cache_dir`,
  `jax_persistent_cache_min_compile_time_secs = 1.0`; only effective before the first compile.
- `enable_compile_cache(version: str) -> Path` (:119-125): `cache_root()/"jax"/version`.
  CLI calls `enable_compile_cache(table_version(tables_module.DEFAULT) + code_version())`
  **before anything compiles** (cli.py:158-159).
- `cached_export(fn, args: tuple, *, name: str, static: dict, version: str) -> Callable`
  (:168-207): exports `jax.jit(fn)` at the avals/shardings of `args` to
  `cache_root()/"export"/version/<name>-<key>.bin`, reloads later; key `_export_key` (:138-148)
  covers `static`, treedef, specs (shape, dtype, weak_type, sharding), `jax.device_count()`,
  backend, `code_version()`. Bad file or failing first call -> re-export. Process-level memo
  `_LOADED` (:42). Every pytree in args/result needs
  `jax.export.register_namedtuple_serialization` (stage_a.py:87-88 registers `Tables`,
  `BattleState`, `PlayoutResult`; **Stage B must register `Outcome` and any new NamedTuple**,
  and registering the same class twice raises in jax — import stage_a's registration rather
  than repeating it, or guard it).
- Each distinct argument shape => a new trace/export/compile. Stage B batches should use a few
  fixed padded sizes (as stage_a pads to `NUM_ACTIONS` rows and to a device-count multiple).

---

## 5. `src/nuzline/search/stage_a.py` (170 lines)
`__all__` (:37): `ActionResult, Ranking, rank_actions, wilson`.
- `@dataclass(frozen=True) class ActionResult` (:40-50): `action: int, label: str, p_win0: float,
  ci_low: float, ci_high: float, p_win: float, e_faints: float, n: int,
  tied_with_best: bool = False`.
- `class Ranking(list)` (:53-70): `all_zero: bool`; `Ranking.of(results)` sorts by
  `(-p_win0, -p_win, action)`, sets `tied_with_best` = `r.ci_high >= best.ci_low and r.ci_low <=
  best.ci_high` for non-leaders; `all_zero = all(p_win0 == 0)`.
- `wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]` (:73-83): raises `ValueError`
  for `n <= 0`; exact 0 / 1 at k=0 / k=n.
- `_playouts(tables, state, actions, keys, max_turns) -> PlayoutResult[A, n]` (:91-98):
  `vmap(per_action)(actions)` with inner `vmap` over keys of `playout_from_action`.
- `_label(names, state, action: int, struggle: bool) -> str` (:101-107):
  `"Switch -> {Species}"`, `"Struggle"`, or the move name. Species names come out of the manifest
  in upper case (e.g. `MONFERNO`; the CLI test lowercases output).
- `rank_actions(tables, names, state, n: int, key, max_turns: int = 100) -> Ranking` (:110-170):
  - `legal = flatnonzero(legal_actions(tables, state, 0))`; `keys = jax.random.split(key, n)`.
  - **Sharding** (:118-128): `devices = jax.device_count()`; `padded_n = ceil(n/devices)*devices`;
    if `devices > 1`: pad keys with copies of `keys[0]`, `NamedSharding(Mesh(jax.devices(),
    ("d",)), PartitionSpec("d"))`, `jax.device_put`. Padded columns dropped (:142) so results do
    not depend on device count (test :138-141 compares 4 vs 1 devices in subprocesses).
  - Actions padded to `NUM_ACTIONS` rows repeating `legal[0]` (:131-132).
  - `cached_export(partial(_playouts, max_turns=max_turns), args, name="playouts",
    static={"max_turns": max_turns}, version=code_version())` (:134-140), `args = (tables, state,
    padded_actions, sharded_keys)`.
  - Refusal: any refused playout of an action -> `refusal_reason(...)` on the first refused key and
    `raise UnsupportedAi(f"{k} of {n} playouts from {label!r} refused; playout {i}: {reason}")`
    (:147-153).
  - `wins0 = sum(won & faints == 0)`, `p_win0 = wins0/n`, Wilson CI, `p_win = mean(won)`,
    `e_faints = mean(faints)` (:154-168).
- "Depth 0 = Stage A on the same keys": to match exactly, Stage B depth 0 must call
  `rank_actions` (or `_playouts` with the same `jax.random.split(key, n)` keys and same
  `max_turns`) — the leaf keys per action are `keys[j]`, shared across actions (CRN).

---

## 6. `src/nuzline/engine/coverage.py` (253 lines)
- `class Support(IntEnum)`: `NONE = 0, SAMPLING_ONLY = 1, EXACT = 2` (:60-63).
- Registries (all `dict[..., Support]`, only EXACT or NONE values):
  `IMPLEMENTED_EFFECTS` (:66-68), `IMPLEMENTED_ABILITIES` (:69-71), `IMPLEMENTED_ITEMS` (:72-74),
  `IMPLEMENTED_TRAINER_ITEMS` (:75-77), `IMPLEMENTED_AI_FLAGS = {0,1,2: EXACT}` (:92),
  `POLICY_AI_FLAGS = 0b111` (:94), `IMPLEMENTED_WEATHER` (only NONE exact, :95-102),
  `START_CONDITIONS` (:104-111), `AI_FLAG_NAMES` (:79-91).
- `@dataclass class CoverageReport` (:114-122): `missing: list[str]`, `sampling_only: list[str]`,
  `level: Support = Support.EXACT`.
- `class UnsupportedMechanics(ValueError)` (:125-132), `.report`.
- `start_weather(trainer) -> Weather` (:135-140).
- `_Collector.add(support, what, kind)` (:143-153): `level = min(level, support)`; NONE ->
  `missing.append(f"{what} ({kind} UNSUPPORTED)")`; SAMPLING_ONLY ->
  `sampling_only.append(f"{what} ({kind} SAMPLING_ONLY)")`. **The plumbing for a
  SAMPLING_ONLY grade already exists here**; nothing assigns it.
- `check_coverage(tables, names, player, trainer, opp, start_condition: int, leads=(0,0))
  -> CoverageReport` (:213-236): party checks, trainer bag/AI flags, weather, then
  `ai_refusals` on the turn-0 `make_state` (as NONE "trainer AI").
- `require_coverage(...)` same args (:239-253): raises `UnsupportedMechanics` only when
  `report.missing` is non-empty (sampling_only does not raise).
- Docstring :9-21 states the sampling-only path is unimplemented and what adding it needs
  (per-trainer AI width etc.). The CLI already prints a "Sampling only" section
  (cli.py:101-109, `_print_report`).
- Guard test `tests/engine/test_coverage.py:92-104` (`test_sampling_only_is_unimplemented`)
  asserts every registry value is in `{EXACT, NONE}`.
- **How to add a sampling-only grade for Stage B**: the spec's sampling-only is a property of a
  (state, action pair) at search time (width cap exceeded), not of a fight, so it does not need a
  registry entry: Stage B can count such nodes itself and report them. If a mechanic is graded
  `SAMPLING_ONLY` in a registry, `_Collector.add` will list it in `report.sampling_only` and
  `level` becomes `SAMPLING_ONLY`; the guard test must then be updated, and the docstring
  :9-21 rewritten. `_check_party` (:156-175) is where a move effect would be graded.

---

## 7. `src/nuzline/cli.py` (218 lines)
- Docstring (:1-17): commands coverage/rank/trainers; ASCII-only output (cp1252 console).
- `INTERMEDIATE = Path(__file__).resolve().parents[2] / "data" / "intermediate"` (:29).
- `ALL_ZERO_FOOTER = "all lines 0 %: no zero-faint line found at this depth"` (:31).
- **Exit codes** `EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2` (:33).
- `resolve_trainer(trainers, *, name=None, trainer_id=None) -> Trainer` (:36-50).
- `configure_cpu_devices(count: int | None, environ=os.environ) -> None` (:56-64): appends
  `--xla_force_host_platform_device_count=N` to `XLA_FLAGS` unless already present; must run
  before jax import.
- `_parser()` (:67-94): subparsers `dest="command", required=True`; helper `fight(cmd)` (:73-79)
  adds `--party` (Path, required), mutually exclusive required `--trainer` / `--trainer-id` (int),
  `--lead` (int, 0), `--opp-active` (int, 0). `rank` adds `--n` (2000), `--seed` (0),
  `--devices` (default `os.cpu_count()`).
- `_trainer_line(t) -> "Name (class, id N)"` (:97-98); `_print_report(report)` (:101-109).
- `_fight(args)` (:112-128): returns `(tables, names, trainer, player, opp, start_weather)`;
  raises `ValueError` for a fainted `--lead`. Loads `load_trainers(INTERMEDIATE)`,
  `load_tables()`, `load_names()`, `load_party(args.party, names)`,
  `build_trainer_party(trainer, IntermediateData(INTERMEDIATE))`.
- `_coverage(args) -> int` (:131-139).
- `_pct(p) -> f"{100*p:5.1f} %"` (:142-143).
- `_rank(args) -> int` (:146-191), the template for `_solve`:
  1. lazy imports incl. jax (:147-154);
  2. validate (`--n < 1` -> `ValueError`, :156-157);
  3. `enable_compile_cache(table_version(tables_module.DEFAULT) + code_version())` (:159);
  4. `_fight(args)`; print trainer line;
  5. `require_coverage(...)`; on `UnsupportedMechanics` print report, `return EXIT_REFUSED` (:163-167);
  6. `make_state(tables, player, opp, weather, opp_items=trainer.items, leads=leads)` (:168);
  7. `rank_actions(...)`; on `UnsupportedAi` print `f"refused: {e}"`, `return EXIT_REFUSED` (:169-173);
  8. header `f"{mine} vs {theirs}: {n} playouts per action, seed {seed}"` (:176), table rows
     `f"  {mark} {i:2d}  {label:<{width}}  {odds:<27}  {_pct(p_win):>7}  {e_faints:9.2f}"`
     with `odds = f"{_pct(p)} [{100*lo:4.1f}, {100*hi:5.1f}]"` (:177-186); tie legend (:187-188);
     `ALL_ZERO_FOOTER` (:189-190).
- `main(argv=None) -> int` (:202-214): for `rank`, validates `--devices >= 1` (stderr
  `"nuzline: error: --devices must be at least 1"`, `EXIT_ERROR`) then `configure_cpu_devices`
  (:204-208); dispatch dict `{"coverage": _coverage, "rank": _rank, "trainers": _trainers}` (:209);
  `ValueError`/`FileNotFoundError` -> stderr `f"nuzline: error: {e}"`, `EXIT_ERROR` (:210-214).
  A `solve` command needs: a parser entry, the `--devices` branch extended to `solve`, and a
  dispatch entry. Console script: `nuzline = "nuzline.cli:main"` (pyproject.toml:13-14).
- Note `UnsupportedMechanics` is a `ValueError` subclass; `_rank` catches it before `main` would.

### `tests/test_cli.py` (211 lines)
- Fixtures: autouse `_data(tables)` skips without tables or `data/intermediate/trainers.json`
  (:15-20); autouse `_restore_xla_flags(monkeypatch)` (:23-30) because `main(["rank", ...])`
  appends to `XLA_FLAGS`.
- Paths: `PARTY = tests/fixtures/sample_party.toml`, `UNSUPPORTED =
  tests/fixtures/sample_party_unsupported.toml` (Metronome) (:10-12); fainted fixture
  `sample_party_fainted.toml` (lead 5 fainted).
- **How rank is run cheaply**: (a) real runs with small `--n` (`"50"` :76, `"20"` :83, :92, :133);
  `main()` inside pytest appends the device flag but jax is already imported, so these run on
  1 device; (b) `monkeypatch.setattr("nuzline.cli._rank", lambda args: 0)` to test flag
  handling without running (:36); (c) `monkeypatch.setattr(stage_a, "rank_actions", lambda *a,
  **k: stage_a.Ranking.of(rows))` to test formatting (:98-103) — this works because `_rank`
  imports `rank_actions` from the module at call time (`from nuzline.search.stage_a import
  rank_actions`, cli.py:154). A `_solve` should import the same way so tests can patch
  `stage_b.<solve fn>`.
- Assertions style: exit code, substring checks on `capsys.readouterr().out/.err`,
  `out.encode("cp1252"); assert out.isascii()` (:81-86).
- `test_cli_module_does_not_import_jax` (:201-211): subprocess `import nuzline.cli` must not load jax.
- Measured durations (this machine, warm-ish caches): `test_rank_prints_nine_rows` 28.2 s,
  `test_rank_output_is_plain_ascii` 22.5 s (section 12).

---

## 8. Reference engine (`src/nuzline/reference/`)
- `reference/convert.py`: `to_ref(state: BattleState) -> RefState` (:22-34);
  `from_ref(ref: RefState) -> BattleState` (:37-54), storage dtypes, `OverflowError` on overflow,
  `ValueError` on shape mismatch.
- `reference/engine.py` docstring entry points (:12-26):
  `ref_step(tables, state: RefState, actions: tuple[int, int], outcome) -> RefState` (:1009);
  input not modified (`copy.deepcopy`, :1011); over state returned as-is (:1012-1013);
  `ref_replace(tables, state: RefState, side: int, slot: int) -> RefState` (:1092-1105);
  `ref_damage_table(tables, state, atk_side, move, crit) -> list[int]` (:370).
  `tables` must be `data.tables.to_numpy(tables)` (tables.py:51-54), `outcome` must be
  `outcome_to_py(outcome)` (outcome.py:82-85). Imports nothing from `nuzline.engine` but `ids`.
  Actions are assumed legal.
- `reference/state.py`: `@dataclass RefPokemon` (:23-43), `RefSide` (:45-82), `RefState` (:84-97:
  `sides, weather, weather_timer, trick_room, gravity, mud_sport, turn, over, winner,
  player_faints, ai_seen_moves, ai_ability_known`); `MON_FIELDS, SIDE_FIELDS, FIELD_FIELDS`
  (:100-102).
- No reference `send_replacements`/AI: a reference "child set" check for `enumerate_outcomes`
  would apply `ref_step` per enumerated outcome and compare to the JAX children
  (`from_ref(...)` then field-equal, as `tests/test_differential.py:35-44` does).

---

## 9. Test helpers and fixtures
### `tests/conftest.py` (35 lines)
- `TABLES_DIR = <repo>/data/gen4_platinum` (:11).
- At import: `_COMPILE_CACHE = cache_root() / "jax-tests" / code_version()`,
  `prune_other_versions`, `set_compile_cache_dir` (:16-18). Only the compile cache; the export
  cache used by `rank_actions` goes to the normal `cache_root()/export` (repo `.cache/export`).
- Session fixtures `tables` (:26-29) -> `load_tables(TABLES_DIR)`, `names` (:32-35) ->
  `load_names(TABLES_DIR)`; both skip when `tables.npz` is absent.

### `tests/engine/helpers.py` (280 lines)
- Constants: `ROOT`, `INTERMEDIATE`, `SAMPLE_PARTY` (:29-31); `OPP_FIXTURE_MOVES = ("Tackle",
  "Rock Throw", "Stealth Rock", "Leer")` (:34); `PHYSICAL, SPECIAL, STATUS = 0, 1, 2` (:27).
- `find_move(tables, power, type_id, category) -> int` (:42).
- `synthetic_state(tables, *, atk_level, atk_stats, def_stats, move_power, move_physical,
  atk_types, def_types, move_type, def_level=50, atk_ability=..., def_ability=..., atk_gender=2,
  def_gender=2) -> (BattleState, move id)` (:77-121): one Pokemon per side — a natural base for
  tiny constructed endgames.
- Private cached: `_roark()` (:123-128), `_roark_party()` (:130-133), `_inter()` (:135-138),
  `_with_moves` (:140), `_lead(party, species, inter)` (:151).
- `roark_trainer() -> Trainer` (:170-172); `roark_party(names) -> list[PokemonSpec]` (:175-178);
  `sample_party(names, override_moves=None, override_item=None)` (:181-194).
- `roark_state(tables, names, *, equal_speed=False, player_count=6, opp_count=3,
  player_species=None, player_moves=None, player_ability=None, player_gender=None,
  opp_species=None, opp_moves=None, opp_gender=None, opp_faster=False) -> BattleState`
  (:197-257): **a test fixture, not Roark's real fight**: every opponent knows Tackle, Rock Throw,
  Stealth Rock, Leer; Roark's bag (two Potions). `player_count=1, opp_count=1` gives a 1v1.
- `roark_real_state(tables, names, *, player_species=None)` (:260-270): sample party vs Roark's
  real party and bag (what the CLI builds, leads 0/0).
- `A(player, opp, state, names) -> int32[2]` (:273-280): move names -> slots.
- Small-endgame pattern used in tests: `roark_state(..., player_count=1).replace_hp(0, 0, 1)
  .with_pp(0, 0, [0,0,0,0])` (test_stage_a.py:82; test_playout.py:122-127).

### `tests/strategies.py` (291 lines, hypothesis)
- `M1_SPECIES` (:44-56), `M1_ITEMS` (:57), `M1_BAG_ITEMS` (:58).
- `supported_pokemon(tables, names)` (:133), `_party` (:176-189),
  `supported_state(tables, names) -> BattleState` (:191-246): random mid-battle state, not over;
  **draws any `Weather`** (:236) and screens/Tailwind/Trick Room — states the coverage registry
  would refuse; fine for step differential tests but AI calls on them may be refused (NaN).
- `legal_action_pair(tables, state) -> int32[2]` (:249-258) (player legal + opponent legal incl. items).
- `any_outcome() -> Outcome` (:272-291), every field uniform over its range.
- `_DURATION = (2, 6)` (:65), `_SLEEP_TURNS = (1, 5)` (:62).

### `tests/ai/fixtures.py`
`outcome(rolls=(0,0,0,0), tiebreak=0, ability_guess=0, switch_stop=SWITCH_NONE)` (:26),
`versus(...)` (:38), traced setups `trace_a..trace_g` (:58-115), `TRACED` (:117).

---

## 10. Test runtime conventions
- No slow markers, no custom pytest marks, no hypothesis profiles: `grep` finds only
  `pytest.mark.parametrize`/`skipif` (e.g. `tests/test_golden.py:20`,
  `tests/data/test_trainer.py:17`). pytest config is just `testpaths = ["tests"]`,
  `pythonpath = ["."]` (pyproject.toml:30-32). Run with `uv run pytest` (README.md:9).
- Hypothesis settings are per test: `tests/test_differential.py:29-33`
  (`max_examples=500, deadline=None, suppress_health_check=[too_slow, function_scoped_fixture]`);
  `tests/ai/test_distribution.py:30-33` (`_SETTINGS`, then `max_examples=300/200/60`).
- Statistical tests use fixed `jax.random.PRNGKey(n)` seeds.
- Multi-device tests run in subprocesses with `XLA_FLAGS` set (`tests/search/test_stage_a.py:96-141`),
  `PYTHONPATH = src + root`.
- Measured: `uv run pytest tests/test_cli.py tests/search --durations=12` -> **48 passed in 279.65 s**.
  Slowest: `test_rank_actions_same_on_multiple_devices` 57.1 s; `test_rank_prints_nine_rows` 28.2 s;
  `test_common_random_numbers_are_deterministic` 23.8 s; `test_ranking_covers_legal_actions_and_is_sorted`
  22.9 s; `test_rank_output_is_plain_ascii` 22.5 s; `test_labels_name_the_move_or_the_switch_target`
  21.3 s; `test_batched_playout_shapes` 16.4 s; playout tests 11-16 s each. Every distinct playout
  batch shape (each `n`) re-traces/compiles the playout program (~10-20 s). Stage B tests should
  reuse one or two leaf batch shapes across tests to keep the suite tolerable.

---

## 11. `runs/` and run records
- **No run-record code exists.** `runs/` does not exist in the working tree and is gitignored
  (`.gitignore`: `runs/` under "Generated data and runs"). No Python references `runs/`.
- Only references: `docs/design.md:59` (Reproducibility: save hash, trainer, settings, seeds,
  table versions, result), `docs/design.md:226` (`runs/  # run records: ...`), and the Stage B spec.
- Available ingredients: `table_version(tables_module.DEFAULT)` and `code_version()`
  (jaxcache.py), `trainer.id/name`, party file (hash its bytes), seed, settings, jax version.
  `benchmarks/rank_timing.py` writes JSON results to `benchmarks/results/rank_<timestamp>.json`
  (gitignored) with a `machine()` dict (`cpu`, `logical_cpus`, `jax`, `os`, `python`) — a pattern
  to copy. Root for runs/: `Path(__file__).resolve().parents[2] / "runs"` mirrors `INTERMEDIATE`
  (cli.py:29); tests should redirect it (tmp_path / monkeypatch) so the suite writes nothing.

---

## 12. Docs and spec cross-references
- `docs/design.md` §4.2/§4.3 (parent spec), §6 (perf: chunked playouts with compaction, persistent
  cache, lines ~215-217), §7 layout (`engine/... observe`, `search/  # battle search (MC,
  expectimax, plan tree)`, line ~232).
- `benchmarks/README.md:41-75`: rank timings (cold 69.9 s, warm 21.5 s, n=2000, 20 devices).

---

## 13. Measurements (this machine: i5-13600K, 20 logical CPUs, Windows 11, jax per uv.lock)

Setup: the CLI's Roark start (`make_state(sample party, Roark trainer 246 real party, weather 0,
opp_items=(17,17), leads=(0,0))`), broadcast to B states; random action pairs (both sides move
slots 0-3); outcomes from `vmap(sample_outcome)`; `jax.jit(jax.vmap(step, in_axes=(None,0,0,0)))`.
No persistent compile cache enabled for these runs (cold compiles).

| measurement | 1 device | 20 host devices (sharded on batch axis) |
|---|---|---|
| B = 50,000: step trace + lower | 1.72 s | 1.80 s |
| B = 50,000: step XLA compile | 3.53 s | 4.40 s |
| B = 50,000: step first run | 0.078 s | 0.037 s |
| B = 50,000: step warm run (5 runs) | 0.072-0.078 s (**1.44 us/state**) | 0.033-0.037 s (**0.65 us/state**) |
| B = 50,000: `vmap(sample_outcome)` compile+run / warm | 1.07 s / 0.113 s | 1.24 s / 0.019 s |
| B = 200,000: step compile / warm run | 3.56 s / 0.235-0.264 s (1.17 us/state) | - |
| B = 200,000: `sample_outcome` warm | 0.489 s | - |
| `device_get` of 50k stepped states | ~0 s (1 dev), 0.007 s (20 dev) | |

Other per-50k costs on 1 device (bench_host.py, states after one step):
- `jit(vmap(ai.policy._distribution))` (opponent action distribution): compile+run 3.6 s,
  **warm 0.519 s** (~10 us/state) — 7x the cost of `step`; per belief node it is computed once per
  state, not per outcome, so it matters less, but it is the most expensive per-state kernel.
- `jit(vmap(send_replacements(tables, s, False, 10)))`: compile+run 1.1 s, **warm 0.061 s**.
- Host merging: packing the 63 fields into a `uint8[50000, 752]` row matrix 0.021 s;
  **`np.unique(rows, axis=0)` 3.42 s (too slow)**; a Python `dict` keyed by `row.tobytes()` over
  50k rows **0.022 s** (210 distinct states in that random batch). Use bytes-keyed dicts (or a
  hashed `void` view) for merging, not `np.unique(axis=0)`.

Sizing implication: stepping is ~1.5 us/state on one device, so even 10^5-10^6 enumerated
combinations per level cost 0.15-1.5 s of step time; leaf playouts (~2,800/s full playouts per
the spec) dominate by orders of magnitude. A fixed padded batch size (e.g. 2^14 or 2^16) avoids
recompiles; each new shape costs ~3.5-5 s cold compile for step alone.
