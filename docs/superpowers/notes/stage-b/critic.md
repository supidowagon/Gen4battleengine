# Stage B critic: verification of the three mapping reports, corrections, and gaps

Repo `G:/Farry/Claude Workspace/Pokemon Platinum Game Engine`, HEAD 6e817e1, no repo file edited.
Inputs reviewed: the design doc `docs/superpowers/specs/2026-10-30-stage-b-solver-design.md` and the
reports `outcomes.md`, `observe.md`, `integration.md` (same scratchpad folder).
Probe scripts I ran (scratchpad `stageb/`): `critic_probe.py`, `critic_ref.py`, `critic_moves.py`.
Paths below are relative to the repo root; `src/nuzline/` is written in full.

---

## A. The 10 most plan-critical claims, spot-checked against the code

| # | Claim (report) | Verdict | Evidence |
|---|---|---|---|
| 1 | Any player faint ends the battle as a loss, even with a simultaneous last-opponent faint; `step` returns an `over` state unchanged (integration §0.1) | **Confirmed** | `src/nuzline/engine/step.py:477-510` (`player_fainted`, `ended`, `winner = where(player_fainted > 0, WINNER_OPPONENT, WINNER_PLAYER)` at :499); `step` :692-700, last line `return _where(state.over != 0, state, s)` (:700) |
| 2 | Turn recipe Stage B must copy: `ai_switched = is_switch(actions[1])`, `power_mul = 20 if pursuit_fires(tables, pre_state, 0, actions, choices) else 10`, then `send_replacements` (integration §1, outcomes §5) | **Confirmed** | `src/nuzline/search/playout.py:179-191` (`_play`), :194-199 (`_turn`), `send_replacements` decorated `@jax.jit` at :157, def :158-176; refused when `need & (ai_switched \| slot < 0)` (:172) |
| 3 | `Outcome` fields/shapes, every field int8, `empty_outcome` values (hit 1, crit 0, roll 15, secondary 0, contact 0, n_hits 2, duration 2, ties 0) | **Confirmed** | `src/nuzline/engine/outcome.py:44-58`, :61-79; cast at :193 |
| 4 | Only `hit[side,0]` is read (`step.py:329`); `roll[side,0]` also feeds the confusion self-hit (`step.py:296`) | **Confirmed** | `step.py:329` `connected = can_act & has_target & ((outcome.hit[side, 0] != 0) \| never_misses)`; `step.py:296` `confusion_damage(..., outcome.roll[side, 0])` |
| 5 | Laws: P(hit) = clip(t,0,100)/100 from turn-start state; crit 1/odds unless Lucky Chant on the defender; secondary = 1 when chance 0; n_hits 3/8,3/8,1/8,1/8; duration Taunt {3,4,5}, Trap {3..6}, else {2..5}; para 1/4, conf 1/2, thaw 1/5, contact 3/10, ties 1/2 | **Confirmed** | `outcome.py:100-129` (threshold, 100 for switch/item/acc 0 at :121-122), :132-140, :143-193 |
| 6 | `secondary` read via `proc = (chance == 0) \| outcome.secondary[side, hit_index]` | **Confirmed** | `src/nuzline/engine/effects.py:468`; `duration` only for SLEEP in `_inflict` (`effects.py:189`) |
| 7 | `ai_action_distribution(tables, state) -> float32[14]`, raises `UnsupportedAi` on a concrete refused state, NaN when traced; `_distribution` is `@jax.jit` | **Confirmed** (line nit: `_distribution` def is :628, decorator :627) | `src/nuzline/ai/policy.py:627-651`, :654-660, `_check` :457-464 |
| 8 | `rank_actions` keys = `jax.random.split(key, n)`, shared across actions (CRN); CLI key `jax.random.PRNGKey(args.seed)`; Wilson CI; `tied_with_best` overlap rule | **Confirmed** | `src/nuzline/search/stage_a.py:110-170` (split :115), `src/nuzline/cli.py:170`; `Ranking` :53-70; `wilson` :73-83 |
| 9 | No registry assigns `SAMPLING_ONLY`; guard test `tests/engine/test_coverage.py:92-104`; `Support` enum NONE/SAMPLING_ONLY/EXACT = 0/1/2 | **Confirmed** | `src/nuzline/engine/coverage.py:60-63` (enum), docstring :9-21; `tests/engine/test_coverage.py:92-104` |
| 10 | `nuzline.cli` must not import jax at module import; exit codes 0/1/2; `_rank` imports `rank_actions` at call time | **Confirmed** | `tests/test_cli.py:201-211`; `cli.py:33`, :146-191 (import at :154) |
| 11 (bonus) | HP-bar pixels `= 0 if hp == 0 else max(1, hp*48 // max_hp)` | **Confirmed** | `external/pokeplatinum/src/battle/healthbox.c:1559-1590` (`offset = 8 * cellNum`, both branches, `pixels == 0 && updated > 0 -> 1`); `src/unk_0208C098.c:41-49` |
| 12 (bonus) | `code_version()` hashes every `*.py` under the package | **Confirmed** | `src/nuzline/jaxcache.py:74-84` |

No claim checked was wrong in substance. Corrections are in section B.

---

## B. Corrections and refinements to the reports

1. **observe.md §4, "unrun opponent choice can leave `move_hit[0]` different": resolved by the code,
   it cannot survive the turn.** In `execute_actions` the per-actor `ran` is
   `(state.over == 0) & (state.hp[side, active] > 0)` evaluated *before* the actor's turn
   (`step.py:557`), so a player who is flinched or fully paralyzed still has `ran = True` and
   `_move_end` clears `move_hit[0]` (`step.py:513-526`, clear at :524-525). Cases:
   - opponent first and disrupted: it writes `move_hit[0] = 0` or leaves it (range-dependent,
     `step.py:364-366`), then the player acts (`ran`) and `move_hit[0]` is cleared to 0;
   - opponent second and disrupted: player already cleared it to 0, opponent writes 0 or keeps 0;
   - player fainted: terminal (value 0 either way).
   So unrun/disrupted opponent choices never leave distinct live states. A unit test is still
   cheap, but the plan need not design around it.
2. **observe.md §2 "detecting whether the second actor ran needs the intermediate state" is
   partly avoidable: PP deltas reveal it.** `pp` is per party slot (`int8[2,6,4]`), decremented
   only when `can_act & ~struggling` (`step.py:312-314`) or by Pursuit (`step.py:461`), and it is
   **not** in `_VOLATILE` (`step.py:84-101`), so a faint does not wipe it (unlike `last_move`,
   which `_reset_battler` clears, `step.py:159-166`). Hence `before.pp[s] - after.pp[s]` names the
   move slot that actually ran for each side (except Struggle). What PP cannot tell: a disruption
   (full paralysis/flinch/sleep message) of a battler that then fainted later in the same turn.
   **For Roark + sample party that case cannot occur**: the opponent can only be disrupted by full
   paralysis (no player move flinches, sleeps, confuses or taunts; outcomes.md §6), and a paralyzed
   opponent cannot also be burned/poisoned, has no recoil (Rock Head; no recoil moves), so it cannot
   faint after a full-paralysis turn. The player can be flinched (Cranidos Headbutt) but a player
   faint is terminal. So for Roark "which actions ran" = PP deltas + switch/item actions (always run
   while live, `step.py:419-426`).
3. **integration §0.5 / spec §7 throughput**: the "2,800 playouts/s" is with **20 host devices**.
   Source: `benchmarks/README.md:80-101` (18,000 playouts = 9 actions x 2000 in a 6.5 s run phase,
   "mean 8.6 turns, max 21"). Cold trace 12.7 s + compile 11.5 s for the `[10, 2000]` playout
   program; **each new leaf batch shape costs another ~20-25 s cold** (trace and compile are both
   proportional to program size, README :83-88). Stage B leaf playouts must use one fixed padded
   shape (chunks), or the 2-minute depth-1 target is lost to recompiles.
4. **integration §4 "registering the same class twice raises"**: confirmed at runtime:
   `ValueError: Duplicate serialization registration for type <class 'nuzline.engine.state.BattleState'>`
   (`critic_probe.py`). `Outcome` is **not** registered yet and registering it works. Registration
   lives at module import of `stage_a.py:87-88`; Stage B must import `stage_a` (or guard) rather than
   re-register `Tables`/`BattleState`/`PlayoutResult`.
5. Line nits: `jax.jit(jax.vmap(step, in_axes=(None, 0, 0, 0)))` is `tests/test_jax_hygiene.py:17`
   (not :16); other jitted `step` users: `tests/test_golden.py:22`, `tests/test_trace_to_state.py:67`.
6. **observe.md signature note**: the spec header (`observe(state_before, actions, state_after)`,
   spec :93) and its practical form (uses the outcome, spec :109-113) disagree; the plan should fix
   one signature. Facts that drive it: crit/miss need the `Outcome`; the replacement (and anything
   it triggers) is seen, so `state_after` should be the state **after** `send_replacements`; faints
   are still detectable there because a fainted slot keeps `hp == 0` and `before.hp > 0`
   (`step.py:488`). Suggested shape: `observe(tables, before, actions, outcome, after_replaced)`.

---

## C. Gaps none of the reports covers, answered from the code

### C1. For Roark + the sample party, every information set is a single state (plan-critical)

- Opponent max HP is `[32, 31, 43]`, player `[44, 44, 47, 30, 37, 30]` (runtime, `critic_probe.py`
  on `roark_real_state`, `tests/engine/helpers.py:260-270`). With `max_hp <= 48`,
  `hp*48 // max_hp` is injective on `0..max_hp` (step 48/max_hp >= 1), checked numerically: every
  Roark mon is **injective**. So the opponent's exact HP is fully observable in this fight.
- No hidden counter can arise: no sleep, confusion, Taunt or Bind source on either side
  (outcomes.md §6 move table; confirmed against `EffectId` values in `src/nuzline/engine/ids.py:12-48`).
  Toxic not present. Remaining H fields from observe.md §1 summary are therefore all constant.
- Consequence: for Roark, grouping by observation never merges two distinct live states; belief
  nodes are singletons and the anti-peeking machinery is **not exercised by the Roark fight**. The
  spec's info-set tests (spec §6: "HP-within-a-pixel pair", "hidden counters", "peeking would
  change the decision") must use constructed states:
  - HP within a pixel: an opponent with `max_hp > 48` (e.g. `synthetic_state(..., def_level=50)`,
    `tests/engine/helpers.py:77-121`, or `roark_state(...).replace_hp(...)` with a high-level spec).
  - Hidden counters: implemented, accuracy-100 moves (runtime `critic_moves.py`): **Spore** (id 147,
    STATUS_FOE sleep, acc 100), **Confuse Ray** (id 109, CONFUSE_FOE, acc 100). Others: Sing 47 (55),
    Sleep Powder 79 (75), Hypnosis 95 (60), Supersonic 48 (55), Sweet Kiss 186 (75). Multi-hit (for
    width/`n_hits` tests): Spike Cannon 131, Arm Thrust 292, Bullet Seed 331, Icicle Spear 333 (acc 100).
    `roark_state(..., player_moves=..., opp_moves=...)` takes move names (`helpers.py:197-257`).
- The same state can still be reached through different observations (crit KO vs non-crit KO,
  outcomes.md §4 "merge key caution"): merge on (observation, state), not state alone, if labels
  must be kept.

### C2. Host float precision and the AI distribution's dtype

- `jax_enable_x64` is **False** (runtime). Every device-side probability is float32.
- `ai_action_distribution` is float32 (`policy.py:654-660`); at the Roark root it is exactly
  `[0.75, 0.25, 0, ...]`, sum 1.0 (runtime). In general its entries are products of k/65536
  (`TIEBREAK_P`), N/256 script rolls and switch-stop probabilities, summed in float32, so a row
  can miss 1 by ~1e-7 (`ai_action_distribution_enumerated` docstring notes ~2e-6 drift for a
  naive sum, `policy.py:580-587`).
- Battle-outcome laws are exact rationals with denominators from {100, 16, 8, 4, 3, 2, 5}
  (`outcome.py:143-193`); compute them on the host in float64 (numpy) or as fractions, not in
  float32 on device. For the "sums to 1" test use a tolerance of ~1e-12 for battle outcomes and
  ~1e-6 once AI weights are multiplied in. The exhaustive-solver equality test (spec §6) is exact
  only if both solvers use the **same** float32 AI row; state that tolerance in the plan.
- AI probabilities can be small: a single script roll is N/256 and a tiebreak 5461/65536 ~ 0.083,
  so opponent actions with mass near or below the suggested epsilon 1e-4 can exist in other states
  (not at the Roark root).

### C3. PRNG keys: prefix-stable splits

`jax.random.split(k, 10)[:5] == jax.random.split(k, 5)` (runtime, jax 0.12.0.1,
`jax_threefry_partitionable = True`, keys `uint32[...,2]` from `PRNGKey`). So a leaf can grow its
budget (anytime / reach-based allocation) with `split(leaf_key, n_new)[:n_old]` reproducing the
earlier draws, and "depth 0 = Stage A on the same keys" holds by calling `rank_actions` with
`jax.random.PRNGKey(seed)` and the same `n` (CI there is Wilson, `stage_a.py:73-83`, not a
propagated normal interval, so depth 0 should delegate to `rank_actions` outright).

### C4. Reference engine speed (relevant to an observation route through `reference/`)

`critic_ref.py`, Roark root, Ember vs Rock Throw, `empty_outcome`: `ref_step` **0.29 ms/call**
(includes its `copy.deepcopy`, `src/nuzline/reference/engine.py:1009-1013`), `to_ref` 0.09 ms,
`from_ref` 1.13 ms. Pre-merge combinations at depth 2 are of order 10^6 (estimate in C7), so
observations computed per pre-merge combination in Python would cost ~5 min; per merged child
(~10^5) about 30-40 s. Either compute observations on the JAX side from (before, outcome, after)
(PP deltas, B.2) or only per merged (state, outcome-class) child.

### C5. Player legality is a function of observable fields (spec's assumption holds)

`legal_actions` reads `moves`, `pp`, `taunt[side] > 0` and `move_power` (via `usable_moves`,
`src/nuzline/engine/actions.py:42-49`), party `species`/`hp` and `trapped[side] == 0`
(`actions.py:52-65`). Only the non-zero-ness of `taunt`/`trapped` is used, and both start/end
with messages (observe.md §2.4, §2.6). So the observation must include `taunt[0] > 0` and
`trapped[0] > 0` (or the start/end events) for legality to be constant over a set. Action
encoding: 0-3 move slot (0 is Struggle when no slot usable, `actions.py:62`), 4-9 switch to slot
`a-4`; opponent 10-13 bag slot (`actions.py:36-39`).

### C6. Mid-tree refusals: the spec is silent, the code offers these sources

Spec §5 covers refusals only at the CLI level. Below the root, three things can refuse a child:
(a) `ai_action_distribution` NaN / `UnsupportedAi` for a child state (`policy.py:654-660`;
reasons from `ai_refusals`, `policy.py:447-454`: scoring, item, switching layout);
(b) `send_replacements` refusal (replacement needed in a turn the AI switched, or
`post_ko_switch_in` returns -1; `playout.py:170-176`, `src/nuzline/ai/replacement.py:115-130`);
(c) a refused leaf playout (`won == REFUSED_WON == -1`, `playout.py:77`, :229-235). Stage A raises
`UnsupportedAi` on any refused playout of an action (`stage_a.py:147-153`). The plan must decide
(refuse the root action, or count refused reach mass like sampled mass and report it). Never sum
`won == -1` as a loss. For Roark today, `nuzline rank --n 2000` completes, so (c) is at least rare
from the root; (b) needs an AI voluntary switch, which is 0 at the root distribution.

### C7. Size of the depth-2 tree and the leaf-floor cost (estimates, to size the budget formula)

From spec §1 (1 to ~111, mean ~14 merged children per action pair) and the root facts (9 player
actions, 2 opponent actions with mass; integration §3):
- depth-1 belief nodes ~ 9 x 2 x 14 ~ 250;
- depth-2 leaves ~ 250 x (9 player actions x ~2-3 opponent actions x ~14) ~ 6-9 x 10^4 (minus
  terminals);
- pre-merge combinations stepped at depth 2: ~10^6 at ~0.65 us/state on 20 devices (integration
  §13) ~ 1 s: negligible.
At ~2,800 playouts/s (20 devices), a leaf floor of f playouts costs ~25-30 f seconds for depth 2.
The precision target (leader's 95 % CI <= +-1.5 pp) needs total variance <= (0.015/1.96)^2 =
5.9e-5; with reach-proportional allocation `Var ~ sum_i w_i p_i(1-p_i)/N <= 0.25/N`, i.e. about
**3,300 (p ~ 0.74) to 4,300 (p = 0.5) effective playouts under the leading root action**. So
depth 1 within 2 minutes cold is easy if leaf batches reuse one compiled shape (B.3); depth 2 is
dominated by the floor: keep it small (f <= ~4) to stay well inside 10 minutes.

### C8. Max over noisy children below the root

The spec mitigates max-bias only at the root (`tied_with_best`). At every depth-1 player node the
value is a max over ~9 actions each valued by few leaf playouts, which biases values upward,
worst where floors are small (exactly where C7 pushes). This is not covered by the reports or the
spec's "Confidence" paragraph; the plan should either (a) pick the action per belief node on one
batch of playouts and value it on an independent batch (two disjoint key ranges, C3 makes that
cheap), or (b) state the bias in the report. Also decide whether the CI treats the argmax at inner
nodes as fixed (variance of the chosen child only).

### C9. Leaf details the plan needs

- Leaves sit at a player decision point after replacement, so use `batched_playout(tables,
  states[B], keys[B], max_turns)` (`playout.py:253-257`; policy from the first turn), not
  `playout_from_action`. `max_turns` counts from the leaf (default 100, `playout.py:238-241`).
- Value terminal children exactly (`over != 0`: 1 iff `winner == 2`, `playout.py:229`; state.py
  docstring) and do not spend playouts on them.
- `turn` increments every `step` (`step.py:699`), so states at different depths never coincide;
  transposition dedupe only works within one depth.
- The playout policy peeks (`playout._move_odds` reads `target.hp`, `playout.py:103-113`); per C1
  this is not even peeking in the Roark fight, since all its state is observable.

### C10. Labels and output ingredients

- `Names` (`src/nuzline/data/tables.py:67-79`): `species_names`, `move_names`, `item_names`,
  `ability_names`, `type_names`, `nature_names` (tuples, index = decomp id). Species names are upper
  case (e.g. `MONFERNO`). Roark's bag item 17 is `"Potion"` (runtime). Action labels:
  `stage_a._label(names, state, action, struggle)` (`stage_a.py:101-107`), private but reusable.
- `ActionResult` (`stage_a.py:40-50`) has `p_win`, `e_faints`, `n` fields that Stage B does not
  optimize (spec §4: P(win) == primary while the battle ends on the first player faint); a Stage B
  result type, or `p_win = p_win0`, `e_faints = 0.0`, needs a decision. `Ranking.of` sorts by
  `(-p_win0, -p_win, action)` (`stage_a.py:53-70`).
- `Tables` fields a relevance mask needs (`src/nuzline/data/tables.py:19-43`): `move_power` int16,
  `move_type` int8, `move_category` int8 (0/1/2), `move_accuracy` int8 (0 = never misses),
  `move_priority` int8, `move_target` int8, `move_effect` int16 (`EffectId`), `move_effect_chance`
  int8, `move_param` int16[M,3], `move_flags` int16 (`MoveFlag.CONTACT = 1`, `ids.py:124`).
  `EffectId` values (`ids.py:29-48`): UNSUPPORTED 0, PLAIN 1, STAT_SELF 2, STAT_FOE 3,
  SECONDARY_STAT_SELF 4, SECONDARY_STAT_FOE 5, STATUS_FOE 6, SECONDARY_STATUS 7, FLINCH 8,
  HIGH_CRIT 9, DRAIN 10, STEALTH_ROCK 11, TRAP 12, PURSUIT 13, TAUNT 14, CONFUSE_FOE 15,
  LEECH_SEED 16, MULTI_HIT 17, WATER_SPORT 18, RECOIL 19. `Status`: NONE 0, SLEEP 1, POISON 2,
  BURN 3, FREEZE 4, PARALYSIS 5, TOXIC 6 (`ids.py:98-105`).
- Width cap and multi-hit: with per-hit independent crit x 16 rolls, a 5-hit move gives up to
  (2 x 16)^5 ~ 3.4e7 raw combinations before roll collapsing; with the spec's ~1e5 cap, multi-hit
  moves will be sampling-only unless rolls are collapsed per hit index first (not Roark).
- Lint/style: ruff, `line-length = 100`, `target-version = "py312"` (`pyproject.toml:34-36`).

### C11. Statistical-test pattern to copy for `enumerate_outcomes`

`tests/engine/test_outcome.py:64-112`: module fixture vmaps `sample_outcome` over
`jax.random.split(jax.random.key(0), N)` on `roark_state`, and asserts frequencies within
`4 * sqrt(p(1-p)/N)`. The "enumerated distribution matches `sample_outcome` + `step`" test can
reuse it: sample N outcomes, step, bucket the states by bytes, compare with the enumerated child
probabilities at the same 4-sigma bound.

---

## D. Short list of decisions the plan must make that the facts force

1. Observation route (JAX-side from before/outcome/after with PP deltas, or reference engine per
   merged child); cost figures in C4.
2. Merge key: (observation, state) vs state only (C1 last bullet; outcomes.md §4).
3. Mid-tree refusal policy (C6).
4. A fixed leaf playout batch shape and chunking (B.3), with one `cached_export` program.
5. Inner-node max-bias handling and how the CI treats inner argmaxes (C8).
6. Constructed test fights for info sets, since Roark never forms a multi-state set (C1).
7. Float64 host arithmetic and stated tolerances (C2).
