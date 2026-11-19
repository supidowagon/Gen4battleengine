# E2 mechanics ledger

Three parts: the current state, the orders, the rulings. Designs kept for reference:
e2-mechanics-ledger-archive.md. Process: docs/process.md.

## 1. Current state

Orders for a restart after a clear, in this order:

1. D1 is on main (a838c20). D2 (16 groups) is complete on e2-d2 at f5f1508 (main a838c20
   merged, review FIX done, rulings D2-1..D2-31), worktree _scratch/e2/wt_d2. The manager runs
   its pod run (d2d + fullset) and the merge-gate review and sends each result. On a failure: fix
   on e2-d2, then send the tip. Add no other commits to e2-d2.
2. Remove the merged worktrees _scratch/e2/wt_b (e2-d1b) and _scratch/e2/wt_pay
   (e2-payexp-pin) with `git worktree remove`.
3. Queue, each on its own branch cut from f5f1508 in its own worktree under _scratch/e2/:
   a. Done: the snap_compare follow-up (Owed 14; closed in the rulings, "Owed 14 closed").
   b. Done (ruling D2-33, branch e2-singles-fix): the singles follow-up (i) Follow Me / non-Ghost
      Curse (lib.c:1714-1717), (ii) Encore ending under a Rollout lock (bcp.c:1861-1903; the B = 2
      chosen_target write on), (iii) randomized non-Ghost Curse. Its snap_compare diff is
      explained in SF2-snap; queue 3b is closed.
4. Next milestone after D2: D3 (four-actor turn order, action loop, faints and replacement). It
   owes: the speed order before each action (bcp.c:888), resuming an action after U-turn or
   Pursuit with the action's defender, the format draw in the differential, and rulings D2-17,
   D2-31, D2-32. Send the D3 family list to the manager before starting.
5. Pod tips: three lines (branch and sha; table inputs changed yes/no; the pod list path
   _scratch/e2/podlist.txt in the main checkout). One pod run at a time per stream. Prose only
   for a ruling or a finding the manager must act on.
6. Tests run on the test pod. Locally: one heavy process (pytest or script) at a time, one file,
   only with more than 6 GB RAM free. test_invokers.py never runs locally. A subagent never
   kills a process it did not start. Use PYTHONPATH="src;." (semicolon) from the worktree. Run
   tools.build_tables only that way, or it writes into the main checkout.
7. Ownership: E2 owns doubles D2-D12. From 19 Nov E3 owns singles move effects
   (_scratch/e2/handover_e3.md): Owed 1-5 and 12 are E3's; delete them here once E3's ledger
   carries them. E1 owns abilities, held items, battle conditions and AI flags.
8. Parked: batch 14 (the other Metronome moves). Moved to E1: AI blocks #16, #57, #114, #173,
   #215, #267, #109, #116, #252, #82, plus Plus and Minus; mechanics #57, #215, #267, #116,
   #252, #82 and sub_seq #238.

### Doubles: the manager's decisions (settled; do not re-open)

- First target: the multi format, Gardenia's 13 TAG_STRATEGY fights.
- Monte Carlo comes first as build order only. Enumeration at B=4 (D6-D7) is required: a
  doubles fight counts as playable only when both paths exist.
- Plain doubles fights are reported as playable when they get there.
- Before each doubles batch, send the manager E1's file list for that batch.

### D1: what to build (docs/doubles-state-design.md sections 4 and 5)

1. Re-index the state and the reference engine to battlers 0-3: side = b & 1, partner = b ^ 2.
2. A static Format giving B (2 or 4), P and party_of.
3. Arrays: [B] per battler, [2] per side, [P, 6] per party.
4. Pointer fields stored as 1 + battler.
5. Singles stays byte-identical. The net: tests/test_differential.py and the singles tests.
6. E1-owned files D1 re-indexes without changing their logic: the volatile, item and ability
   state fields, and the side conditions (screens, Mist, Safeguard, Tailwind, Lucky Chant).

### D1: change plan

Rules for every file (D = doubles-state-design.md):

- R1 Format: `engine/format.py` defines frozen `Format(B, P, party_of)`: SINGLES (2, 2, (0, 1)),
  DOUBLES (4, 2, (0, 1, 0, 1)), TWO_TRAINERS (4, 3, (0, 1, 0, 2)), MULTI (4, 4, (0, 1, 2, 3)).
  `format_of(state)` reads B = `state.active.shape[0]`, P = `state.species.shape[0]`; shapes are
  static under jit, so no jit signature gains an argument. Helpers: `side_of(b) = b & 1`,
  `partner(b) = b ^ 2`, `party(fmt, b)` (b when party_of is the identity, `b & 1` when P = 2,
  else a gather of party_of), `foe(b) = b ^ 1` (the default defender until D2's resolver).
- R2 `1 - x` with x a battler or side -> `foe(x)` (B = 2: equal to `1 - x`). `1 - x` with x a
  flag or probability stays (effects.py:1276 power_trick toggle; `1 - p` in ai/, search/).
- R3 `for side in (0, 1)` / `range(2)` / `jnp.arange(2)` over battlers -> `range(fmt.B)` /
  `jnp.arange(fmt.B)`; over sides stays `(0, 1)`.
- R4 Per-side field indexed by a battler variable -> `X[side_of(b)]` (D 5.2). Per-party field
  `X[b, slot]` -> `X[party(fmt, b), slot]`.
- R5 Parameters named `side` keep their names and hold battlers. Literal 0 / 1 battler indices
  in B = 2 entry points stay. No BattleState field is renamed or changes dtype or encoding.
- R6 Pointer clips `jnp.clip(p - 1, 0, 1)` -> `jnp.clip(p - 1, 0, fmt.B - 1)`.
- R7 Two-actor turn logic (`first`/`second`, `position == 1`, `speed_tie`, `end_speed_tie`,
  `war_swaps` bits, enumerate's two-order `Branch`) keeps its structure; only R2-R4 apply
  (D3, D6-D7).

Field decisions (engine/state.py and reference/state.py):

- [P, 6] party: species..ivs, knocked_off, evs, nature (all `_SLOT` fields).
- [B] battler: active, stages, every volatile, qc_roll, recycle_item, wish_timer, wish_hp (Wish is
  per battler, D 4), future_timer/damage/move/attacker, entry_pending, every ts_* `[2]` field,
  ts_taken [B, 6].
- [2] side: spikes, toxic_spikes, stealth_rock, reflect, light_screen, safeguard, mist,
  lucky_chant, tailwind, future_sight.
- [P]: ts_alive (a party's HP bitmask, D 4).
- [B]: trainer_items [B, 4], trainer_item_count: the bag of enemy battler b is
  `trainerItems[b >> 1]` (ruling D1-5).
- Party 0 only, unchanged: ai_seen_moves, ai_item_known, ai_ability_known, the exp block, deaths,
  protected (D 4). Scalars unchanged: weather..winner, speed_tie, war_swaps, ts_cursor, ts_used.
- 1 + battler pointers: lock_on_target, mean_look_target, future_attacker, pivot_battler,
  ts_pursued, ts_taken's attacker columns (already so); pursuit_target becomes 1 + battler, 0 =
  none (only ever cleared today, so values stay 0). `infatuated` stays 0/1 in D1: it is an E1
  volatile and the pointer is a value change; it goes to D5 on E1's list (D 5.3; ruling D1-1).
- ts_order [B] holds battler indices (no none value, not a pointer).
- E1 trial fields (merge of main 235f2f8): [B] magnet_rise, magnet_rise_turns, mimic_slots,
  shadow_moves [B, 4], shadow_pp [B, 4], transformed, shadow_species, shadow_form, shadow_stats
  [B, 5], shadow_ivs [B, 6], shadow_ability_slot, shadow_ability_form, truant, ts_endure_ok (on
  RefBattler); [P] chatot_cry (on RefParty, ruling D1-2). Taken as the trial made them:
  ai_seen_moves and ai_ability_known int16 (party 0 only), `Tables.gravity_rearm`.

Ownership (confirmed by the manager): E1 owns src/nuzline/ai/outcome.py (group B, item 5);
src/nuzline/engine/outcome.py is E2's (group A, item 9).

Group A, E2, after main is merged into e2-d1, one commit per line, in this order:

1. src/nuzline/engine/format.py (new): R1.
2. src/nuzline/engine/state.py + src/nuzline/engine/STATE.md: shape keys `_MON (P, 6)`,
   `_BATTLER (B,)`, `_SIDE (2,)`, `_PARTY (P,)`; the field decisions; `initial_arrays(fmt, parties,
   leads)`; `make_state(tables, player, opp, leads, ...)` keeps its signature and calls it with
   SINGLES; `held_items` over `range(fmt.B)`; `SIDES = 2` stays (the side count).
3. src/nuzline/reference/state.py: split `RefSide` into `RefParty` (mons, trainer_items,
   trainer_item_count), `RefBattler` (active, volatiles, qc_roll, wish, future_*, recycle_item)
   and `RefSideConds` (the [2] fields); `RefState` holds `parties[P]`, `battlers[B]`, `sides[2]`;
   field tuples MON_, PARTY_, BATTLER_, SIDE_, FIELD_FIELDS.
4. src/nuzline/reference/convert.py: to_ref / from_ref over the three lists (R3).
5. src/nuzline/reference/engine.py: `st.sides[x].f` -> `st.battlers[b].f`, `st.sides[side_of(b)].f`
   or `st.parties[party(fmt, b)]` by the field decisions; R2-R7.
6. src/nuzline/engine/order.py: R2-R4 (`effective_speed`: tailwind via side_of, D 5.2).
7. src/nuzline/engine/damage.py: `def_side` becomes a required argument (D 5.1); callers pass
   `foe(atk_side)`; reflect / light_screen via side_of.
8. src/nuzline/engine/effects.py, src/nuzline/engine/actions.py: R2-R4.
9. src/nuzline/engine/outcome.py: Outcome fields `[2, ...]` -> `[B, ...]`; `empty_outcome(fmt=
   SINGLES)`; R2-R4, R7.
10. src/nuzline/engine/step.py: R2-R7.
11. src/nuzline/engine/enumerate.py: Reads `[2]` / `[2, 2]` -> `[B]` / `[2, B]`; R2-R4, R7.
12. src/nuzline/engine/coverage.py: R2 (:656).
13. tools/trace_to_state.py: R2, R3.
14. tests/reference/test_reference_smoke.py: `out.sides[1].party[0]` -> `out.parties[1].mons[0]`.
    tests/strategies.py: no draw-order change; only constructor calls follow item 2.

No D1 edit: engine/observe.py (D6-D7), engine/levelup.py (party 1 = the foe's, D10), search/,
prep/, cli.py, saves/, tools/differential.py (D11), every other test (B = 2 literals are valid).
Done when every `1 - ` left in engine/ and reference/ has a non-battler operand.

Group B, E1, re-index only (R2-R6, no logic change), after group A and only after the manager
reports E1's trial batch merged; commit subjects carry `[e1-model]`; each is listed in the batch
report:

1. src/nuzline/engine/hooks.py: R2 (:766 default `foe(side)`, :819), R3 (:322, :337-338), R4.
2. src/nuzline/engine/bag.py: R4 (trainer_items by party; `mist[1]` is a side read, stays).
3. src/nuzline/ai/commands.py: R4; `Battler.DEFENDER/ATTACKER` and `_side` stay (D8 changes
   `_side`).
4. src/nuzline/ai/switching.py: R2 (:348, :353), R4.
5. src/nuzline/ai/policy.py, src/nuzline/ai/replacement.py, src/nuzline/ai/outcome.py: R4, R3
   where an axis is per battler.
6. The trainer bag per opponent, with bag.py (merge check, post-trial items 5b, 5c):
   `trainer_items` / `trainer_item_count` per party [P] is wrong for plain DOUBLES. The decomp
   keeps `aiContext.trainerItems[MAX_OPPONENTS][4]` (include/battle/ai_context.h:31), indexed by
   `battler >> 1` (battle_controller_player.c:4808, trainer_ai.c:4092-4093), filled from
   `trainers[battler]`; a plain double starts with enemyTrainer1ID == enemyTrainer2ID
   (encounter.c:736, :751-752). Confirm from the decomp (Opus). ~~Make both per opponent [2]
   indexed by `battler >> 1`.~~ (superseded by ruling D1-5: [B]). And state.py `initial_arrays` fills party 1's bag only (`opp_items`
   -> `trainer_items[1]`): the second opponent in TWO_TRAINERS / MULTI starts without items.
   Done in group B as ruling D1-5: per battler [B] (the opponent's `battler >> 1` bag), battler
   3 filled from `initial_arrays`' `opp2_items`, required for every four-battler fight.

Test net (singles byte-identical):

- Before any edit, on the merged pre-D1 tip, after `bash tools/ops/build_tables.sh` (vanilla and
  Kaizo): write `_scratch/e2/d1_snap.py` with `capture DIR A-B` / `compare DIR A-B`. Capture, per
  seed of tools/differential.py's generator, 5 turns: the drawn input state, actions and outcome;
  `step_events` and `jit(step)` outputs; `ref_step` via from_ref; `sample_outcome(PRNGKey(seed))`;
  `enumerate_turn` (or its SamplingOnly refusal); `ai_action_distribution`; `decision_actions`;
  `root_belief`. Also roark_01's replay states and `rank_actions` at `roark_real_state`, N = 256,
  PRNGKey(0). Compare reruns from the saved inputs and checks dtype, shape and bytes per field.
- `uv run python _scratch/e2/d1_snap.py capture _scratch/e2/d1_pre 0-199` in chunks of 25 seeds.
- After each commit of group A and group B:
  `uv run python _scratch/e2/d1_snap.py compare _scratch/e2/d1_pre 0-199` (same chunks);
  `uv run python -m tools.differential 50 0-39`;
  `uv run pytest tests/prep/test_variants.py tests/data/test_game_constants.py
  tests/tools/test_backlog.py tests/test_cli.py tests/test_golden.py tests/test_trace_to_state.py
  tests/reference -m "not slow"`.
- At the end of each group: `uv run pytest tests/engine tests/ai tests/search -m "not slow"` and,
  one at a time, `uv run pytest -m slow tests/test_differential.py tests/test_golden.py
  tests/search/test_k0_regression.py`.
- Limits: every test and every snap chunk under 60 s and 4 GB; a chunk over either is halved; a
  test over either is marked slow. Any byte difference stops the group until it is fixed.

Who: a Sonnet subagent per group-A item (and per group-B item) makes the edit and runs the compare,
the differential and the guards on the spot. Opus writes format.py, the reference split (items
3-5) and the snap script, decides any field not listed above from the decomp, and fixes any
compare failure.

Group A status and decisions (commits 1-5):

- Done: 1 format.py 1129e71; 2 state.py 7cb3ad2; 3 reference/state.py f466510; 4
  reference/convert.py 1397931; 5 reference/engine.py 84da8d1.
- `ts_alive` lives on RefParty ([P]).
- `entry_pending` and the `ts_*` [2] per-battler turn fields live on RefBattler.
- `fake_out_turn` is per battler.
- `LAYOUT` stays `layout(SINGLES)` for outside callers.
- `initial_arrays` takes `(fmt, tables, parties, ...)`; prep/states.py and
  tests/engine/test_e1_batch_a.py got call-site edits.
- Done: 6 order.py 92c1c0b; 7 damage.py 87f6da4; 8 effects.py + actions.py 4db7d93; 9 outcome.py
  446e6a9; 10 step.py fb7adc2; 11 enumerate.py 31be426; 12 coverage.py 19290b3; 13
  tools/trace_to_state.py 01494eb; 14 no edit left (the smoke test's `out.parties[1].mons[0]`
  came with 84da8d1; strategies.py has no constructor call to change).
- Snap compares run: 0-19 identical after commits 1, 2, 5, 6-11 and 13; none after 3-4 (not
  runnable alone) or 14 (no edit); 0-199 identical at 84da8d1, 19290b3 and 5c32327. Merged tip
  (main 235f2f8 into e2-d1, plus the merge-check notes and tools/snap_compare.py) against a
  baseline captured at 235f2f8, 0-199: pending pod run.
- `def_side` is required only on the internal `_pre_random` / `_finish`; the public damage entry
  points pass `foe(atk_side)`. `ai_case_power` and `ai_damage_for_roll` keep an optional `def_side`
  defaulting to `foe(atk_side)`, so ai/commands.py (E1) is unchanged.
- Helpers: `state.field_index(state, field, index)`; `format.foes(x)` replaces `x[::-1]` on a
  battler axis.
- The `RefSide` aliases are removed. tests/engine/test_screens.py and
  tests/test_damage_oracle.py pass `def_side` to `_pre_random`.
- Open: party-to-battler reductions rewritten by reading are not exercised under singles; the D3
  and D6 doubles tests cover them.
- Review fixes (group A): R6 clips to `fmt.B - 1` in outcome.py (`future_threshold`,
  `future_thresholds` x2, `pivot_view`) and enumerate.py (`_paused_rock`); search/playout.py:484
  `pivot_pick` keeps its clip (search/ is not a D1 file).
- step.py `_empty_events(fmt)` sizes per-battler events [B] / [B, MAX_HITS]; observe.py's
  `replacement_observation` passes `format_of(after)` (its `order` stays [2]: D6-D7).
- step.py `sample_root` draws `qc_roll` with shape (B,): the same key and shape, so B = 2 is
  bit-identical.
- enumerate.py `_empty(b)` is cached per battler count (an Outcome depends on B only), B read
  from `laws.relevant`; `enumerate_turn`'s `taken` tags broadcast to (rows, B).
- outcome.py `_can_go_first`, `_threshold_grid`, `sample_outcome` pass `fmt` to `empty_outcome`.
- tools/trace_to_state.py reads per-party state fields through `_party_of(state, b)` (R4):
  `_hit_state`, `_holds_quick_claw`, `_used_slots`, `_roared`, `action_choices`, the contact
  ability check, `_roar_options`, `_statuses`. The trace's own per-side records stay (B = 2 input).
- Docstrings: `damage._pre_random`'s required `def_side`; `[2]` -> `[B]` (per battler), `[P]` (per
  party: `_alive_mask`, `decomp_abilities`) in the touched files; speed_tie-branch `[2]` axes stay.
- Rule: tests run on the pod (send the manager the tip and the test-file list during a batch and
  for the final tip); locally one process, one file, more than 6 GB free.
- Group A review 1 = FIX (fixed in 5c32327). Review 2 of 5c32327 = MERGE.
- Merge check of commits 1-5 (post-trial, fixed; singles unchanged, each reduces at B = 2):
  - reference/engine.py: `turn.fainted` holds (battler, slot) throughout; `ts_alive` is built
    from it through the battler's party, and `_resume` maps a paused faint back to the party's
    battler whose active it is (`_fainted_battler`; singles: battler k).
  - reference/engine.py: Mud Sport / Water Sport read every battler
    (`BattleSystem_AnyBattlersWithMoveEffect`, battle_lib.c:6812); `_faint_check`'s payout branch
    tests `side_of(b)`; its death and `player_faints` branch tests the battler's party
    (`_party(fmt, b) == 0`, as step.py counts `fainted[0]`); `_Turn` takes `battler_count` and sizes its per-battler
    lists; the turn's items and the roosting clears are B long.
  - state.py: the field comments give [P] / [B] / [S] shapes; `initial_arrays` rejects two
    battlers of one party leading from one slot.
  - order.py `compare_speed` builds every battler's speed over `range(fmt.B)` and takes an
    optional `battler2` (default `foe(battler1)`).
  - state.py: the side shape key is "S" (not the literal 2); `field_index` documents that it
    maps a battler once (not idempotent for TWO_TRAINERS); `format.party` returns a Python int
    for a Python int battler (the reference indexes lists with it).

Group B status (E1 files, re-index only):

- Done: 1 hooks.py 7fba32b; 2 bag.py 1f7dc8c; 3 ai/commands.py e86ad54; 4
  ai/switching.py 133dbac; 5 ai/policy.py, ai/replacement.py, ai/outcome.py: no edit (every
  state read there is a literal battler 0 / 1, valid in every format: R5); trainer bag per
  opposing battler 60f6f37 (ruling D1-5).
- hooks.py: `reveal_ability` / `_know` test `party(fmt, side) == 0` (the AI tracks party 0);
  `berry_confuse` reads Safeguard through side_of; `show_battle_mon` / `battle_start` loop
  `range(fmt.B)` with `foe(side)`; `weather_war` / `order_swaps` keep their two-actor reads (R7).
- bag.py `_set` writes through `state.field_index`.
- ai/commands.py reads parties through `_row(state, b)` = `party(format_of(state), b)`; the
  side-condition and Spikes reads through side_of.
- Not re-indexed (party axis, not battler): policy.py `_refused` and switching.py `_layout`
  build `[2, SLOTS]` refusals over parties 0 / 1 (`jnp.arange(2)`, `enumerate` of two ability
  lists): at P > 2 the shapes do not broadcast. Owed to D8 (which parties an AI scores).
- Group B review = MERGE.

### D2: plan

Family: targets, the action space and redirection. 16 groups, one batch. Decomp paths below are
under external/pokeplatinum (at 579d2d5); lib.c = src/battle/battle_lib.c, bcp.c =
src/battle/battle_controller_player.c, script.c = src/battle/battle_script.c, sys.c =
src/battle/battle_system.c, subs/ = res/battle/scripts/subscripts/. Engine paths under
src/nuzline/.

Orders:

1. Send the manager the family list and the E1/E3 file list below; start on its answer.
2. An Opus rule writer writes one sheet per group to `_scratch/e2/rules/d2/D2-<n>.md`: every
   rule cited by file:line from both the C and the scripts; per process.md section 2 step 2 each
   sheet states the group's effect on hit chance, damage inputs, crit inputs, held items read
   later in the turn and the trainer AI's view, or "none" with the cite. The writer re-reads
   every line cited here; a cite that does not hold is fixed in the sheet and in this plan.
3. Sonnet builders implement each sheet in the JAX engine and reference/engine.py, with the
   coverage check and tests; rulings go in section 3 as D2-<n>.
4. Fixed for every group: the resolver lives in a new `engine/targets.py` (E2), mirrored in
   reference/engine.py; singles stays byte-identical (at B = 2 the target follows from the
   range, bcp.c:4363-4367, and `BattleSystem_RandomOpponent` draws nothing, lib.c:4177-4198);
   every new random draw is an Outcome field indexed by the acting battler ([B]); AI target
   choice and scoring stay D8's; search/prep/CLI with joint actions stay D11's; the four-battler
   `monSpeedOrder` recomputed before each action (bcp.c:888, lib.c:3364-3383), its tie draws and
   the resumption of an action after a U-turn / Pursuit pause at B = 4 stay D3's (ruling D2-2).
   New Outcome fields and their sample keys: D2-4 sheet, section R8.

Groups (rule source; files; owner flags):

- D2-1 Action encoding. Each battler's action is today's scalar plus a target battler: `step`
  takes `actions` int32[B] and `targets` int8[B]. Source: the target stored per battler at
  `battlerActions[b][1]` = menu input - 1 (bcp.c:534-551); the AI's pick enters the same slot
  (`BATTLECTX_AICTX_SELECTED_TARGET` + 1, src/battle/battle_display.c:3748-3752); at B = 2 the
  target is `battler ^ 1`, or the user for a RANGE_MY_SIDE range or a non-Ghost Curse
  (bcp.c:4342-4370, Curse :4344-4349; include/constants/battle/moves.h:91). D2-1 owns the stored
  target `chosen_target` int8[B] (initial `b ^ 1`), kept between turns (ruling D2-1). Files:
  engine/state.py, STATE.md, reference/state.py, engine/actions.py, step.py (`step`,
  `_action_phase`), reference/engine.py, tools/differential.py and tests/test_differential.py
  (targets drawn at B = 2 only; the format draw is D3 / D6's, ruling D2-3).
- D2-2 Legal targets per range. Doubles always asks for a target except a RANGE_ALLY move whose
  partner is switching (bcp.c:4342-4370); a non-Ghost Curse is RANGE_USER (:4344-4349). The
  menu's selectable battlers: `GetTargetSelectLayout` (src/battle/battle_subscreen.c:1336-1364),
  `sMoveTargetSlotFlags` (:797-810), slot i = battler type 2 + i, absent battlers off
  (:2567-2582; `BattleSystem_SetBattlersByType`, sys.c:1477-1484; types
  include/constants/battle.h:101-107). Range list: generated/move_ranges.txt:1-17 (a mask,
  generated/meson.build:72). Files: engine/actions.py (`legal_targets`), reference/engine.py.
  The effects.py helpers (`targets_user`, `self_defender`, `defender_side`) are the resolver's:
  D2-4 / D2-5.
- D2-3 Joint action legality at B = 4: the player's pair of actions; the second battler's party
  menu receives the partner's pending switch slot (bcp.c:570-587); the item command draws a
  random opponent as defender (bcp.c:1905-1908, lib.c:4177-4198), which no state reads: no
  Outcome field, no engine change (ruling D2-8). Files: engine/actions.py
  (`legal_actions`, `decision_actions` per battler plus the pair mask).
- D2-4 Single-target resolution and retargeting: Follow Me user if alive, else the chosen target
  if alive, else `BattleSystem_RandomOpponent` (lib.c:1709-1727; one `RandNext() & 1` in
  doubles, the other opponent if that one fainted, lib.c:4177-4198); opponent slots
  `BattleSystem_GetEnemyInSlot` (sys.c:413-433; slots include/battle/battle_system.h:39-40).
  RANGE_SINGLE_TARGET_ME_FIRST without randomize takes this branch. Files: engine/targets.py,
  outcome.py (`target_draw` [B]; `targets.resolve`, `targets.opponents`), step.py (`_use_move` `target_side`, ~862),
  effects.py (`play_hit` `target_side`, ~3198), reference/engine.py.
- D2-5 Own-side ranges: RANGE_USER / USER_SIDE / SINGLE_TARGET_SPECIAL / FIELD give the user
  (lib.c:1665-1669); RANGE_ALLY the partner in doubles (:1670-1675; `BattleSystem_GetPartner`,
  sys.c:394-411); RANGE_USER_OR_ALLY the chosen battler, the user if it fainted (:1676-1684);
  randomized RANGE_USER_OR_ALLY a coin flip for the partner in doubles (:1651-1660). Files:
  engine/targets.py, effects.py (`defender_side`, Acupressure's defender ~1604), outcome.py.
- D2-6 RANGE_OPPONENT_SIDE and randomized Me First: a random opponent (lib.c:1661-1664).
  Files: engine/targets.py, effects.py (hazard setters' defender).
- D2-7 RANGE_RANDOM_OPPONENT and every randomized move: Follow Me user if alive, else a coin
  flip between two living opponents, else the living one (lib.c:1685-1708). Files:
  engine/targets.py, outcome.py.
- D2-8 Spread ranges' first defender: RANGE_ADJACENT_OPPONENTS the first living enemy in
  `monSpeedOrder`, RANGE_ALL_ADJACENT the first living battler other than the user; both leave
  `battlerCounter` one past it (lib.c:1611-1650). The per-target loop is D4's. The speed order is
  an input (D3's, ruling D2-2); the B = 4 tests pass an explicit order. Files: engine/targets.py.
- D2-9 When the fight command randomizes: Struggle, Encore, an Encore slot that no longer holds
  the move, a selected move not in its slot; a move lock does not and reads D2-1's `chosen_target`
  (bcp.c:1861-1903).
  Disobedience's other move: randomized, and the chosen target overwritten (bcp.c:2186-2192).
  Files: step.py, order.py (Encore / Struggle sites), reference/engine.py.
- D2-10 No target: `BattleControllerPlayer_HasNoTarget` (bcp.c:2306-2338; `NO_TARGET`,
  include/battle/common.h:56-61), checked after PP is spent (bcp.c:3181-3187, cost 1 with no
  defender, :2238; then :3189-3194) and again at TryMove (:3239-3246); the "no target" script unlocks the move choice and clears Rage
  (subs/subscript_no_target.s:4-18); a charge turn with no defender targets the user
  (bcp.c:2326-2334). Files: step.py, effects.py, reference/engine.py.
- D2-11 Follow Me: set per side with its user (script.c:6261-6270), cleared each turn
  (lib.c:2251-2252, in `BattleSystem_SetupNextTurn` from :2228), read at lib.c:1694-1697,
  :1714-1717, :1743-1745. One new per-side turn-scratch field `ts_follow_me` (1 + user battler,
  0 = off; ruling D2-9) in state.py, STATE.md (rows :530-531), reference/state.py. Files: those, effects.py (EffectId
  FOLLOW_ME, ids.py:270-277), engine/targets.py.
- D2-12 Lightning Rod and Storm Drain: `BattleSystem_CheckRedirectionAbilities` (lib.c:1732-1790:
  none with no defender, a Normalize or Mold Breaker attacker, or a living Follow Me user;
  Electric / Water by `CalcMoveType`, Normal falls back to the data type (:1748-1750; Weather
  Ball with no weather reads an uninitialized type, ruling D2-6); single-target or
  random-opponent range; the multi-turn-first test (:1755, :1773) is never true at either call
  site (SYSCTL_FIRST_OF_MULTI_TURN cleared at :2009); first living holder other than the
  attacker in the speed order, an input, ruling D2-2), run at BEFORE_MOVE_STATE_REDIRECT_TARGET (bcp.c:3204-3206);
  `BattleSystem_TriggerRedirectionAbilities` (lib.c:1792-1817) at TryMove (bcp.c:3247-3252),
  message subs/subscript_lightning_rod_redirected.s:4-12. Files: engine/targets.py, hooks.py,
  ids.py (the LIGHTNINGROD comment :610-612 names
  the wrong function: the redirection is `BattleSystem_CheckRedirectionAbilities`), reference/engine.py.
- D2-13 Invoked and reflected moves: `BtlCmd_GoToMoveScript` resolves a randomized defender,
  runs the redirection check and overwrites `chosen_target` (script.c:2111-2134); D2 builds this
  part only (ruling D2-4). Reference rules for the effects' builders: Mirror Move takes
  `moveCopied[attacker]`, in doubles when that is 0 draws among `moveCopiedHit`, and resolves the
  copied move by its own range with randomize (:3483-3525; `inRange` is ignored for a nonzero
  move, lib.c:1604-1608), writing `chosen_target` (:3517); Magic Coat (:6487-6513) sends the
  reflected move to the living Follow Me user of the original attacker's side, else the original
  attacker; its computed target (:6502) is unused. Files: step.py, engine/targets.py, outcome.py
  (`call_target_draw` [B]), reference/engine.py.
- D2-14 Counter, Mirror Coat, Metal Burst in doubles: the last attacker, or the Follow Me user
  on its side (script.c:4615-4653, :4668-4706, :7153-7186). Bide's release joins it (ruling
  D2-7): the last battler that hit the user, else a random opponent, a new coin (bcp.c:2702-2714).
  Files: step.py, effects.py, engine/targets.py, state.py (`last_hit_by`), outcome.py
  (`bide_target_draw` [B]), reference/engine.py.
- D2-15 Pressure's PP cost reads the resolved defender (bcp.c:2235-2270) at B = 4. Files:
  effects.py (`pressure_pp_cost`, ruling G-8), step.py.
- D2-16 Switch-in abilities over both foes (Owed from D1): Intimidate lowers every living
  battler of the other side without a Substitute, in speed order (lib.c:3795-3813;
  subs/subscript_intimidate.s:10-23); Download sums Defense and Sp. Def over every living foe
  without a Substitute (lib.c:3815-3859); Trace draws between two eligible foes
  (lib.c:3712-3737; `ChooseTraceTarget` lib.c:7750-7781; the coin is rolled on every pass
  for every battler, at B = 2 too, where it cannot change the pick); Intimidate also skips a foe
  whose Substitute was hit during the running action (script.c:9180-9181). Model them as
  multi-target, not a defender choice. Files: hooks.py (`show_battle_mon` ~907-926, `switch_in_effects` ~840,
  `intimidate` ~786, `download` ~219, `trace` ~279), outcome.py (`trace_pick` int8[B, 5] with the turn-scratch
  `ts_trace_fires`, ruling D2-5),
  reference/engine.py.

Singles changes, ruled by the managing session 18 Nov: decomp-correct, approved. D2 builds all three
byte-identical to today's singles. One follow-up commit right after the D2 merge makes the three
changes, with its own tests and a snap_compare diff limited to those states, each diff explained,
and deletes the batch 9 note "changes nothing in singles" at enumerate.py:816:

- (i) Follow Me sends a non-Ghost Curse to a living Follow Me user at B = 2 (lib.c:1714-1717),
  against batch 9's "Follow Me changes nothing in singles" (enumerate.py:816). Sheets D2-4, D2-11.
  It needs the B = 2 `ts_follow_me` write, off since ruling D2-32: this follow-up turns it on.
- (ii) A battler whose stored target is itself (a RANGE_MY_SIDE selection) and that a faster Encore
  forces into Rollout reads that target once the Encore ends inside the lock (bcp.c:1885-1888,
  lib.c:1709-1719); at B = 2 that makes the user Rollout's defender. Sheets D2-1, D2-3, D2-4, D2-9.
  It needs the B = 2 `chosen_target` write, off since ruling D2-28: this follow-up turns it on.
- (iii) A randomized non-Ghost Curse at B = 2 takes the foe as its defender (ruling D2-12); today it
  keeps the user. Added by the managing session 18 Nov.

Files E2 touches that another stream owns (send with the family list):

- E1: engine/hooks.py (D2-12, D2-16); effects.py `pressure_pp_cost` (D2-15).
- E3: engine/actions.py, step.py, effects.py, outcome.py, order.py, ids.py (comments only),
  coverage.py (refusals lifted per group), reference/engine.py.

### D2-D12

- D2: targets, the action space and redirection (plan above).
- D3: the four-actor turn.
  - Owed (D3/D6): step.py `_faint_check` (~2733) reads species[0]/species[1] as the two sides; for
    TWO_TRAINERS/MULTI a side is out only when all its parties are empty. The same in
    reference/engine.py `_faint_check` (`s.parties[0]` / `s.parties[1]`), and its payout of a side-1
    faint indexes `s.parties[1]` (`_grant_exp`): party 2's slot in TWO_TRAINERS (D10 with exp).
  - Owed (D3): the switch-in abilities' order over four battlers. `BattleSystem_TriggerEffectOnSwitch`
    runs each switch-in state over `monSpeedOrder` (battle_lib.c:3740-3790); hooks.py `weather_war`
    and `order_swaps` (~336-362) read only battlers 0 and 1. Model the four-battler
    `monSpeedOrder` and run the weather and other switch-in abilities over it.
- D4: spread moves and the doubles modifiers.
- D5: ally mechanics.
- D6-D7: outcomes and enumeration at B=4.
  - Owed: `action_order` is int8[2] (order.py ~669) while `ts_order` is declared [B] (state.py ~689;
    comment at ~390 still says int8[2]); step.py ~2762/~2768 use jnp.arange(2) against order: resize
    together at B=4.
- D8-D9: AI per-target scoring, the TAG_STRATEGY script and partner arguments.
  - Owed (D8): per-battler AI knowledge. The decomp keeps the AI's record per battler
    (`aiContext.battlerAbilities[MAX_BATTLERS]`, `battlerHeldItems[MAX_BATTLERS]`,
    include/battle/ai_context.h:28-29); the engine keeps it on party 0 by slot, and three tests of
    "the player" disagree at B=4: JAX hooks.py `_know` / `reveal_ability` (~258-268) test
    `party(fmt, side) == 0` (battlers 0 and 2 in DOUBLES; 0 only in MULTI); the reference's checks
    before `_know_ability` test battler 0 and record `s.battlers[0].active` (reference/engine.py
    ~2562-2563, ~2882-2883, ~3141, ~3469; 28 calls of `_know_ability(s, s.battlers[0].active)`),
    so battler 2 is never recorded; effects.py `forget_item` (~3090) tests battler `side == 0`.
    Equal at B=2. Rework all three on a per-battler record.
  - Owed (D8): the AI attacker as data. ai/policy.py, ai/replacement.py and ai/outcome.py read
    literal battlers 0 / 1 (R5), and the bag refusals read only `trainer_items[1]`
    (policy.py ~3950, ~4138, ~4359-4361; search/stage_b.py ~447; engine/actions.py
    `legal_opponent_actions` ~218-222, `usable_trainer_items(..., 1)`): battler 3's bag
    (`trainerItems[1]`) is never offered. Make the attacker (and its `battler >> 1` bag) an input.
- D10: the multi format, with partner teams from the calc repo.
- D11: search, prep, the line player and the CLI with two actives.
  - Owed (D11 or tool note): tools/trace_to_state.py calls empty_outcome() without a format (B=2
    traces only); search/playout.py ~484 battler clip to 1 is outside D1 scope (search/ is D11).
- D12: a golden trace of a recorded Gardenia multi fight.
- Later: two-trainer battles.

## 2. Orders (Owed)

Each is open; close it with its ruling.

1. Fling #233 (B13-22, B13-33): model BattleSystem_FlingItem (battle_lib.c:5854-6156): the 30
   fling effects on the target through the INDIRECT ON_HIT flingScript, each skipped under
   Embargo (:6145), with the pluck-flag semantics; White Herb's reset at TryFling time, before
   the accuracy roll and CalcDamage (:6029-6034). Power Battler_ItemFlingPower (Embargo only, no
   Klutz, :5491-5498), effect Battler_ItemFlingEffect. CalcDamage / CalcCrit read the item still
   held; ApplyTypeChart (Expert Belt) and the accuracy (Wide / Zoom Lens) see it gone. Each
   effect needs its law relevance (status, stage, confusion, flinch, crit stage, Micle, order
   speeds) and a per-time item view in damage. Recycle of a flung item makes
   `outcome.kings_rock_chance` (turn-start item) and the Focus Band / Bright Powder / Quick Claw
   reads time-dependent. Then transcribe Basic_CheckFling / Expert_Fling.
2. Gyro Ball's AI damage estimate (B13-36): model monSpeedValues as written there (a
   per-battler state field in both engines and the differential; an AI context bit "a compare
   ran", flag-major, slot-minor; refuse the correlated per-slot state when the two powers
   differ), then lift the refusal of TrainerAI_CalcDamage's case (trainer_ai.c:2995-3003).
3. Vanilla Fury Cutter #119 (CalcFuryCutterPower, effect_script_0119.s:5: a per-battler
   furyCutterCount that doubles the power per use and resets on a miss or another move) and the
   recharge #80 (effect_script_0080.s:5, VOLATILE_CONDITION_RECHARGING: the forced recharge turn
   in CheckStatusDisruption and the action choice). Vanilla only (Kaizo's Fury Cutter is be_seq
   88). Both stay UNSUPPORTED until modelled.
4. Water Spout / Eruption prep cost (B13-40): `prep.variants.falloff_cuts` multiplies the
   variant product (x4.36 for one HP-44 holder). Add a prep-side reduction (cuts only where a
   power band changes a damage or KO class against this opponent), proven to keep the same lines.
5. Embargo (B13-31): add the two tests its laws rely on: Future Sight's roll kept whole while an
   Embargo is used or up (its turn-end count can end before the hit), and the turn end's speed tie
   enumerated whenever a foe's Embargo can hide a Choice Scarf.
6. Future Sight's accuracy law versus Knock Off: the delayed hit's law
   (`outcome.future_threshold`, B5-14) against a Knock Off earlier in the same turn that removes
   an accuracy item. Settle it from the decomp, then fix or rule, with a test.
7. tests/test_cli.py `test_cli_rank_default_budget_keeps_the_header`: bring it under 8 GB (last
   killed at 8.23 GB).
8. tests/test_cli.py `test_cli_rank_deaths_and_protect`: 8.12 GB since MAX_HITS 6 and the
   per-hit tables; bring it under 8 GB.
9. Add a test for `leaves._run` with `replacement=True` (none since the leaves test was replaced
   by an engine test of the replacement flag).
10. The sampler's cold-compile cost: leaves and Stage A pay about 9.1 GB on a cold compile for a
    party that has or can learn Encore or an invoker (`batched_playout`, 16 playouts:
    replacement=True 9.08 GB, False 7.42 GB). Find a cheaper form.
11. raw_rolls seed 5 (`test_accuracy_classes_equal_raw_rolls`, slow): run
    `drive_rawrolls.ps1 5` under the side lock (`<project>/_scratch/e2/`, phases of 6 examples
    via rawrolls_phase.py; at least 12 GB free at start). Seeds 1-4 are done.
12. The AI move-effect blocks #71, #85, #86, #153, #165, #191, #218, #239, #241, #247, #259 are
    E2's (manager, 18 Nov; #233 goes with Fling). They go in the next singles batch, after D1.
13. tests/engine/test_accuracy_classes.py (45.8 GB in one process): moved to E1's speed pass by
    the managing session, 18 Nov. Cause: the unjitted `sample_outcome` in
    `test_accuracy_classes_equal_raw_rolls` (slow-marked).
16. Owed to D11 (or whoever owns tests/strategies.py), found by the D2-33 review: the strategies
    draw neither `chosen_target == b` (a stored user) nor a set `ts_follow_me`, so
    tests/test_differential.py does not cover D2-33 (i) or (ii); the stream lead decided
    strategies stay untouched in D2-33 (Q3). Add both draws.
17. Cost of D2-33, unmeasured, after ruling D2-35: `chosen_target` is canonical at a turn's end,
    so it splits rows only for a locked or encored battler (where it is read) and in a state
    paused at a U-turn / Baton Pass party list; `ts_follow_me` is cleared at the turn's end and
    splits only such paused states (as D2-32 found). Measure the row-merge counts and the prep
    time on a fixed split before and after.
18. Found by the D2-33 gate review: only the accuracy law of a self lock has an
    enumeration-vs-step mass check (`test_a_self_lock_keeps_its_accuracy_roll_whole_in_the_laws`).
    Add the same check for its crit, damage classes, flinch and status cuts on a self-lock
    state (each foe-keyed cut in `enumerate.outcome_laws` lifted for `self_lock`).

19. Acid Rain: fieldConditionsMask -= faintedMon at a Pursuit KO (subscript_pursuit.s:100, :115;
    script.c:1769-1770, :3282-3287; recoil operand lib.c:4257-4269, :4358-4371, :5400-5409,
    :7357-7374); refused at run time (acid_rain) in every format; owed: FCM.md (E1). Field
    conditions as the decomp mask (FCM.md section 4); then rule 8 is a subtraction and the
    refusal goes.

### Closes with its mechanic

Rules left out because another mechanic is missing; each closes when that mechanic lands.

- Trick, Thief, Skill Swap, Worry Seed join `outcome.CRIT_ODDS_EFFECTS` when they land.
- `outcome.kings_rock_chance` reads the turn-start item: Trick, Thief, Fling, Recycle of a King's
  Rock break it.
- Batch 9 leftovers not yet modelled: Volt Tackle's family #262 (recoil plus a PROBABILISTIC
  paralysis on every hit, battle_lib.c:1551-1567), Pressure (PP cost by range,
  skipPressureCheck, its switch-in reveal), Download (downloadAnnounced), the flavor berries'
  dislike confusion, the terrain entries (data). Check the backlog before starting one.
- The weather lock read by Thunder / Blizzard, Sand Veil, Snow Cloak is the turn start's
  battlers (`outcome._rate`'s `on_field`); a Cloud Nine / Air Lock holder fainting before the
  second mover's check is not followed.
- Sticky Barb transfer vs Knock Off's mask (`effects.sticky_barb_transfer`;
  battle_lib.c:5390-5393).
- Fire Fang always bypasses Wonder Guard (battle_lib.c:7600).
- Gravity: `Move_FailsInHighGravity` under `CHECK_INVALID_GRAVITY` (battle_lib.c:2292, :2348;
  trainer_ai.c:234) is ignored by `actions.usable_moves`, Sleep Talk's slots
  (`order.sleep_talk_slots`, battle_script.c:4927) and `Move_CanBeMetronomed`
  (battle_lib.c:7204-7207: `order.metronome_candidates`, `order.copycat_call`,
  `coverage.metronome_calls`); grounding a Fly user (B10).
- The Iron Ball and Gravity clauses of `BasicTypeMulApplies` / `Battler_Ability`
  (battle_lib.c:2534, :2545-2549, :3095-3098).
- Transform vs a vanished target (B10).
- Around the pivots (B6-2..B6-4): Grudge (subscript_attack_then_switch_out.s:40), Rage and
  Shaymin's form check after a switch (bcp.c:3497-3530); what Baton Pass keeps of Heal Block,
  Magnet Rise, Gastro Acid (battle_lib.c:2056-2086, :2105-2115, :2136-2140); Pressure on
  U-turn's Pursuit (battle_script.c:6919-6922).
- Rest and Yawn read `FIELD_CONDITION_UPROAR` (subscript_rest.s, subscript_yawn.s): test
  `effects.uproar_on` there (B5-6).
- The status orbs vs Safeguard (subscript_badly_poison.s, subscript_burn.s).
- Color Change (unsupported, `effects.py:238`): with it a non-Ghost Curse selector can
  turn Ghost before acting. Its stored target is then the user, so `targets.resolve` gives the
  user and the Ghost path of effect_script_0109.s runs `SetRandomTarget BTLSCR_ATTACKER` when
  attacker == defender (:11-12; battle_script.c:9218-9224, `BattleSystem_RandomOpponent`, lib.c:4177-4198: the
  foe at B = 2, a RandNext draw at B = 4). Model the re-pick (and its draw and law at B = 4) with
  Color Change; today the engine would curse the user, unreachable.
- Kaizo's Ganlon and Apicot Berries (param 0) stay refused by `effect_map.DIVISOR_PARAM_MIN`
  (B2-3): `curHP <= maxHP / itemPower` (battle_lib.c:4778, :1294) calls the SDK's signed-divide
  helper, not in the decomp; read its divide-by-zero branch from the ROM's arm9, then model.
- `tests/search/test_leaves.py` goes past the 8 GB per-file cap: restructure.
- `tests/search/exhaustive.py` plays no pause: its endgames have no U-turn or Baton Pass.
- A Trump Card called by an invoker (moveSlot = the invoker's) has no hand test.

### Doubles notes (for D1-D12)

- Flower Gift counts a living holder over the whole side (COUNT_ALIVE_BATTLERS_OUR_SIDE,
  battle_lib.c:2886-2893, read at :6930-6941); `damage._pre_random` reads only the attacker's /
  defender's own ability.
- RANGE_ALLY and RANGE_USER_OR_ALLY name the partner in doubles (`effects.targets_user`,
  battle_lib.c:1670-1684); Helping Hand (fails in singles, B9-5) boosts the ally's move.
- `ai_item_known` is per player party slot; `AICmd_LoadHeldItemEffect` (trainer_ai.c:1907-1910)
  and `IfHeldItemEqualTo` (:1924-1928) read `battlerHeldItems` for any battler: keep it per
  battler on both sides.
- `commands.if_party_member_status` (`AICmd_IfPartyMemberStatus`, trainer_ai.c:1348-1379) leaves
  out the doubles branch (:1359-1364, the partner's selected slot skipped at :1361) and the egg
  check (:1373).
- `BattleAI_PostKOSwitchIn`'s `BattleSystem_RandomOpponent` draw (battle_lib.c:4183-4187, used
  at :7949) is a chance node in doubles: a new Outcome element per battler.
- Mid-turn takeover: a short list of alternatives per battler, each keyed by its event (two foes
  can Disable and Encore one battler in a turn).

### Notes for E1 (AI owner)

- Untranscribed AI scripts whose engine side is E2's: Copycat #242 (B5-1), Baton Pass #127,
  Future Sight / Doom Desire #148 (read `BattleState.future_sight` per side, not
  `future_timer`, B5-10), Mean Look / binding #106, Imprison #192, Ingrain #181.
- Commit 6212174 changed E1 item paths without the "e1-model:" title (B13-39): E1 reviews
  `hooks.pluck_berry` and `play_hit`'s `foe_custap`.

## 3. Rulings

One bullet per ruling, with the decomp citation. Conventions for new code: section 5 of
docs/doubles-state-design.md.

- B1-1 Item order: the held-item block runs after powerMul, Charge, Helping Hand, Technician,
  Huge Power / Pure Power and Slow Start (battle_lib.c:6688-6714), before every ability modifier
  from :6785 on (Thick Fat, Hustle, Guts, ...), in decomp order (type item, Choice Band/Specs,
  Soul Dew, DeepSeaTooth/Scale, Light Ball, Metal Powder, Thick Club, orbs, Muscle Band/Wise
  Glasses): Thick Club x2 truncates before Guts x1.5; the u16 stores never truncate in reach
  (battle_lib.c:6716-6783 before :6793).
- B1-2 Light Ball doubles `movePower` for any move class (battle_lib.c:6748-6751).
- B1-3 Species items read `BATTLEMON_SPECIES`; Soul Dew's Battle Frontier test never holds in a
  trainer battle, the Griseous Orb's Transform test never holds while Transform is unsupported
  (battle_lib.c:6730-6739, :6768-6772).
- B1-4 The confusion self-hit gets the holder's own `BattleSystem_CalcMoveDamage` items (Choice
  Band, a Normal-type item for Struggle's type, Muscle Band, Light Ball, its own Metal Powder /
  DeepSeaScale / Soul Dew as defender), never Life Orb or Expert Belt (`CALC_SELF_HIT`,
  include/battle/common.h:35; battle_controller_player.c:2633).
- B1-5 Life Orb's x1.3 is the move script's (`BattleScript_CalcMoveDamage`, battle_script.c:1346,
  after `criticalMul`): not in the AI's `TrainerAI_CalcDamage` (trainer_ai.c:3081), not on a
  fixed-damage move (Night Shade / Seismic Toss: effect_script_0087.s has no CalcDamage; Bide);
  Pursuit's switch hit gets it (subscript_pursuit.s CalcDamage).
- B1-6 Life Orb recoil is `Divide(maxHP, 10)` (not the param), once per move after all hits, when
  a hit's `UpdateHP` ran with damage != 0 (`SYSCTL_MOVE_HIT`, battle_controller_player.c:3340,
  :3441), `moveCur` is not a status move and the holder lives: fixed-damage hits recoil, a hit
  capped to 0 by a 1-HP Focus Sash counts, the confusion self-hit never does
  (battle_controller_player.c:4758-4772).
- B1-7 Life Orb recoil also follows Pursuit's pre-switch hit unless it missed
  (subscript_pursuit.s:17, :48-49, :94-95 -> `BattleSystem_TriggerHeldItemOnPivotMove`,
  battle_lib.c:7357-7366).
- B1-8 Life Orb recoil runs inside the action, before its faint check: a holder that KOs the last
  foe and faints to recoil ends the battle with both sides out, a loss (ruling 2;
  `TRIGGER_AFTER_HIT_EFFECTS` before `MOVE_END`, battle_controller_player.c:3897-3915).
- B1-9 Choice lock: set at `UpdateMoveBuffers` only when the attack message showed, to `moveTemp`
  (Nature Power itself, not its call), never to Struggle; a non-holder's lock is cleared there; a
  disrupted move (sleep, freeze, flinch, Disable, Taunt, confusion self-hit, paralysis, love)
  changes nothing (battle_controller_player.c:3921-3934).
- B1-10 The defender's lock is cleared in `UpdateFlagsWhenHit` unless it holds a Choice item, when
  the attack message showed (battle_controller_player.c:4489-4496); `BtlCmd_TryPursuit` locks a
  Choice holder into Pursuit (battle_script.c:6943-6950).
- B1-11 A Choice-locked battler whose locked move has no PP (or is disabled) Struggles; a lock the
  battler does not know is cleared, read as no lock (battle_lib.c:2307-2315; `STRUGGLING_ALL`,
  battle_controller_player.c:433).
- B1-12 A lock set on a battler that fainted during its move is never read and is cleared at the
  next switch-in (battle_lib.c:2100-2103); the engine clears it at the faint check.
- B1-13 Choice Scarf's x1.5 Speed: after Swift Swim, before paralysis (battle_lib.c:1228, :1248,
  :1260).
- B1-14 Expert Belt multiplies a `MOVE_STATUS_SUPER_EFFECTIVE` hit after Solid Rock / Filter,
  inside `BattleSystem_ApplyTypeChart`, so the AI's estimate has it (battle_lib.c:2657-2665;
  trainer_ai.c:3096).
- B1-15 Items with no hold effect do nothing, except a berry (`Item_IsBerry`, src/item.c:357-365)
  is eaten by Pluck / Bug Bite with no effect, leaving the pluck flag set (battle_lib.c:5832-5835,
  :5848); prep never offers such a berry (it can only help the foe).
- B2-1 Cheri, Rawst, Aspear, Persim cure paralysis, burn, freeze, confusion at every held-item site
  (after each hit, switch-in, `USE_ITEM` and `ITEM_CONDITION` at turn end); one eaten by Pluck /
  Bug Bite cures the eater the same way (battle_lib.c:4601-4660, :4991-5050, :5569-5641).
- B2-2 A stat pinch berry (Liechi, Ganlon, Salac, Petaya, Apicot) raises its stat one stage at
  `curHP <= maxHP / param` (C division; the holder's Gluttony halves the param), only below +6, at
  the same sites; one eaten by Pluck / Bug Bite raises the eater's stat with no HP test
  (battle_lib.c:4773-4836, :5196-5259, :5753-5796).
- B2-3 A pinch berry or Custap Berry with param below 2 is unsupported: its trigger divides by zero
  (with Gluttony's halving), left by the decomp to the ROM's division routine (battle_lib.c:4778,
  :1294); Kaizo's Ganlon and Apicot have param 0.
- B2-4 Custap Berry in its pinch sets the Quick Claw flag in every `CompareBattlerSpeed`: turn
  order, end-of-turn order, the AI's speed compares (battle_lib.c:1288-1300); its `custapBerry`
  flag is written by `CalcTurnOrder`'s FIGHT-vs-FIGHT compare (bcp.c:772-786) and by
  `BattleSystem_SortMonActionOrder` (battle_lib.c:7276-7300, same compare with
  `ignoreQuickClaw = FALSE`, :1296, :1362) after a Trick Room activation (bcp.c:3990-3993; Trick
  Room is unsupported: adding it must re-set the flag there); the holder eats the berry when its
  move starts, before the status checks, first or last (subscript_check_quick_claw.s:26-28,
  bcp.c:3140-3143).
- B2-5 While that flag is up, a foe's Pluck / Bug Bite does nothing to the holder: no berry eaten,
  no pluck flag (battle_script.c:8125-8127).
- B2-6 Leftovers and Black Sludge act at `MON_COND_CHECK_STATE_LEFTOVERS`, after `USE_ITEM`, before
  Leech Seed (bcp.c:1369): `Divide(maxHP, 16)` below full HP, Black Sludge only on a current
  Poison type; any other Black Sludge holder loses `Divide(maxHP, 8)` unless Magic Guard; neither
  is used up (battle_lib.c:4922-4965).
- B2-7 Toxic Orb, Flame Orb, Sticky Barb act at `MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT`, the
  last check (bcp.c:1710): an orb skips a holder with Immunity / Water Veil, Leaf Guard in sun
  without a weather lock, any status, or a Poison or Steel / Fire type, else badly poisons
  (counter 0) or burns (subscript_badly_poison.s:13-28, subscript_burn.s:5-17); Safeguard
  (unsupported move) is not read.
- B2-8 Sticky Barb's holder loses `Divide(maxHP, param)` (param 8) at that check unless Magic Guard;
  it may faint (battle_lib.c:5330-5336).
- B2-9 Sticky Barb moves to the attacker after a contact hit once the defender took damage this
  action, when the attacker lives and holds nothing (an eaten berry leaves nothing); the defender
  may have fainted; not a `RemoveItem` (no recycle item) (`AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT`,
  battle_lib.c:5386-5398, subscript_transfer_sticky_barb.s); Pursuit's pre-switch hit moves it too
  (subscript_pursuit.s:44, :90).
- B2-10 After every hit: attacker's item, defender's item, Sticky Barb, defender's thaw, then the
  held-item-status pass (bcp.c:3684-3730): a frozen defender's Aspear or Lum Berry cures the
  freeze before a Fire hit thaws it, and is used up.
- B2-11 Accuracy items come after abilities and deep fog, before Gravity: the defender's
  BrightPowder / Lax Incense `* (100 - param) / 100`, then the attacker's Wide Lens and, against a
  target that has moved, Zoom Lens, each `* (100 + param) / 100` (bcp.c:2949-2965); "moved" is
  `Battler_MovedThisTurn` (battle_lib.c:5365-5368): in singles, the attacker acts second (a foe
  that switched in this turn has moved).
- B2-12 King's Rock / Razor Fang roll per hit after the contact ability (`*_EXTRA_FLINCH`) when the
  hit's `UpdateHP` ran, `RandNext() % 100 < param`, the move has `MOVE_FLAG_TRIGGERS_KINGS_ROCK`
  and the defender lives; the flinch needs a target that has not moved and no Inner Focus as the
  attacker sees it (bcp.c:4630-4653, subscript_flinch_mon.s:5-9): `Outcome.flinch_item[2, 5]`.
- B2-13 Shell Bell heals `Divide(dealt, param)` after the move, before Life Orb; `dealt` sums each
  hit's `battleCtx->damage` after the endure cap, capped at the target's HP, not halved by a
  resist berry (bcp.c:3427-3431, :4736-4752); after Pursuit's hit `dealt` is the script's damage
  before the Focus Sash cap, capped at the HP (subscript_pursuit.s:21-29, battle_lib.c:7345-7355).
- B2-14 Light Clay: holder's Reflect / Light Screen last 5 + param (3) turns (battle_script.c:4229,
  :4272).
- B2-15 Gluttony is supported: the pinch berries and Custap are its only decomp readers
  (battle_lib.c:1289, :1355, :4774-4851, :5197-5275).
- B2-16 Kaizo changes item params only, read from each game's table: BrightPowder, Lax Incense,
  King's Rock param 11 (vanilla 10); its Berserk Gene is a renamed Shoal Salt slot, hold effect 36,
  param 0, so unsupported (renamed slot, and B2-3).
- B3-1 Technician (`attackerParams.ability`) raises `movePower <= 60` by x15/10 after powerMul
  (Charge and Helping Hand, between, are unsupported), before held items, never for Struggle (so
  not the confusion self-hit); it reads the input power (Hidden Power's, Magnitude's, Flail's,
  Rollout's) and is in the AI's estimate (battle_lib.c:6688-6702; trainer_ai.c:3078).
- B3-2 Reckless is only the recoil scripts' `BTLVAR_POWER_MULTI = 12` (effect_script_0048.s,
  _0269.s: the supported RECOIL moves; 0045, 0198, 0253, 0262 are unsupported effects): no AI
  estimate, no self-hit; Pursuit is no recoil move (`damage.script_power_mul`).
- B3-3 Thick Fat (ignorable) halves Fire / Ice power right after the items (battle_lib.c:6785-6788);
  Dry Skin (ignorable) raises Fire power x125/100 after Overgrow / Blaze / Torrent (:6845-6848);
  Solar Power raises Sp. Atk x15/10 in sun inside `NO_CLOUD_NINE` (:6922-6924; the self-hit passes
  no field conditions).
- B3-4 In `BattleSystem_ApplyTypeChart`: Levitate (ignorable) sets `MOVE_STATUS_LEVITATED` for a
  Ground move (:2600-2603), part of `MOVE_STATUS_IMMUNE` and `DID_NOT_HIT`: no damage, no
  AfterMoveEffects, zero in the AI estimate (trainer_ai.c:3106); Scrappy breaks the walk at 0xFE
  like Foresight (:2611-2615) for every chart reader (damage, Thunder Wave, resist-berry flag, Night
  Shade's immunity); Tinted Lens doubles a not-very-effective hit after Expert Belt (:2668-2672).
  Status moves other than Thunder Wave skip the chart (bcp.c:2350-2358).
- B3-5 Battle Armor and Shell Armor (same ignorable test, battle_lib.c:7135-7136) stop every crit;
  Super Luck adds 1 crit stage from the battler's own ability (:7126). Both engines gate the drawn
  crit with the armor at the hit; the laws draw none (`outcome.armored`).
- B3-6 No Guard of either battler (`Battler_Ability`, not ignorable) clears `MOVE_STATUS_MISSED`
  after the roll (bcp.c:3016-3021): threshold 100. Tangled Feet (ignorable) halves the rate against
  a confused target after fog, before items (bcp.c:2944-2947).
- B3-7 Quick Feet with any status (`MON_CONDITION_ANY`) raises Speed x15/10 instead of the paralysis
  quartering (battle_lib.c:1257-1261, :1323-1327).
- B3-8 Serene Grace doubles `effectChance` (a u16, then `% 100 <`) in
  `BattleSystem_TriggerSecondaryEffect` and `BtlCmd_CheckEffectActivation` (battle_lib.c:1552-1556,
  :1571-1575; battle_script.c:8369-8373): every probabilistic secondary, Fake Out's flinch, both
  Thunder Fang rolls; not the contact abilities' or King's Rock's rolls.
- B3-9 Shield Dust (ignorable) silently stops every `SIDE_EFFECT_TYPE_INDIRECT` effect on its
  holder: secondary status, confusion, flinch (King's Rock's too), stat drop
  (battle_script.c:2918-2921; subscript_paralyze.s:13, _burn.s:27, _freeze.s:11, _poison.s:20,
  _badly_poison.s:38, _fall_asleep.s:31, _confuse.s:15, _flinch_mon.s:9), after Insomnia / Limber
  / Inner Focus; not Bind's start (no check) or the contact abilities.
- B3-10 Truant: `Battler_CheckTruant` loafs when `truant != (totalTurns & 1)`, `truant` being
  `(totalTurns + 1) & 1` from entry (battle_lib.c:2119, :2197, :3304-3313): `fake_out_turn`'s
  parity, 0 for a lead, whatever the battler did. Checked after sleep and freeze, before flinch
  (bcp.c:2502-2510); subscript_loafing_around.s unlocks and tags no ability; a loafing pursuer
  does not hit a switcher (battle_script.c:6903).
- B3-11 Dry Skin takes a damaging Water move that hit (Hidden Power's type too;
  `TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES`, bcp.c:3281-3286, :2782-2810,
  battle_lib.c:3538-3544): `Divide(maxHP, 4)` below full HP, ability tagged either way, then
  `MOVE_STATUS_NO_MORE_WORK` (no hit, secondary or AfterMoveEffects; attack message showed). At
  turn end sun takes `Divide(maxHP, 8)` from Dry Skin and Solar Power (not Magic Guard), rain gives
  it to Dry Skin below full HP (battle_script.c:5585-5625), each tagging the ability.
- B3-12 Weather moves fail on their own weather, temporary or permanent, except Hail, which fails
  only on temporary hail (effect_script_0164.s checks `FIELD_CONDITION_HAILING_TEMP`): it turns a
  permanent hail into a 5-turn one. Success clears every weather bit and sets 5 turns plus the
  user's matching rock param (Damp / Heat / Smooth / Icy Rock); failure is `MOVE_STATUS_FAILED`
  (no AfterMoveEffects). They are `RANGE_FIELD`, accuracy 0.
- B3-13 Drizzle, Sand Stream, Drought, Snow Warning set their weather permanently at the holder's
  switch-in check when it has HP (after Stealth Rock), whatever weather was up; the script (and
  its message) is skipped only under that same permanent weather (battle_lib.c:3740-3790); Cloud
  Nine does not stop it.
- B3-14 Trace (battle_lib.c:3718-3737, `ChooseTraceTarget` :7750-7775): a holder with HP and no
  Griseous Orb copies a living foe's `BattleMon.ability` unless Forecast, Trace or Multitype; the
  battler's only (no party write), so it ends when it leaves; the AI learns the tracer's and
  target's abilities (battle_script.c:3403, :11806). A Trace facing Trace stays armed and copies
  the next newcomer mid-turn, which the turn-start chance laws do not follow: Trace in both parties
  was refused (`coverage.trace_standoff_line`; modelled since E1 G-25). A player Trace newcomer
  that entered against a fainted foe traces when the foe's replacement enters (`hooks.on_switch_in`
  runs Trace for the other battler too; `playout.replacement_pass` defers only the newcomer's own
  Intimidate).
- B3-15 Two weather abilities setting different weathers, one per party, were refused
  (`coverage.weather_war_line`; modelled since E1 G-26): entering together (battle start, a double
  KO's replacements) the `monSpeedOrder` loop (battle_lib.c:3740-3790, sorted by
  `CompareBattlerSpeed` at bcp.c:251, :263) leaves the slower one's weather; an exact tie is a
  `RandNext() & 1` the engine does not draw there.
- B3-16 Lightningrod redirects only to a battler other than attacker and defender
  (battle_lib.c:1756-1772), never in singles; Honey Gather acts only after the battle
  (battle_script.c:7825): both supported, no in-battle effect.
- B3-17 Kaizo changes none of these: its accuracy changes add effect 141 (Hurricane, unsupported)
  to Thunder's sun and rain branches (docs/kaizo/rom_diff.md, CheckMoveHitAccuracy /
  CheckMoveHitOverrides), and TriggerEffectOnSwitch only its field-weather switch.
- B3-18 The leads' switch-in checks run as one `BattleSystem_TriggerEffectOnSwitch` pass over both
  battlers, state by state (battle_lib.c:3649-4168; Trace :3718-3737 for every lead before
  Intimidate :3795-3808, each announced once): a lead that traced Intimidate lowers the foe once
  (`hooks.battle_start`, from `make_state` and prep's `make_states`).
- B3-19 The turn-start laws read a switching-in target's entry ability, the copy when it traces
  (`damage.entry_ability`): its switch-in checks run before the foe moves, so a Trace newcomer
  facing a Sand Veil attacker in sand is hit at x80/100 (bcp.c:2927-2928), one facing a Static
  contact attacker rolls Static, one copying Battle Armor / Shell Armor is never crit
  (`outcome._rate`, `outcome.armored`, `enumerate._holds`, Cute Charm's `charm_later`).
- B3-20 Prep applies the party-pair refusals per candidate against the trainer's party
  (`weather_war_line`, `trace_standoff_line`, both deleted by E1 G-25 / G-26; `trace_copy_line`
  remains): `prep.problem.usable_pool` leaves the candidate out, `prep.pool.check_pool` refuses it.
- B3-21 The AI reads a battler's current ability (`damage.decomp_abilities`; ai/commands.py
  `attacker_ability` / `_defender_abilities`, ai/policy.py and ai/switching.py refusals), so it
  sees the copy `subscript_trace.s` writes with `BattleAI_SetAbility` (battle_script.c:3401-3403,
  :12106-12109) as `LoadBattlerAbility` / `CheckBattlerAbility` do (trainer_ai.c:1170-1209); the
  engine writes the copy and marks it known for the player's side (`hooks.trace`, `_know`) and
  clears that on a switch (battle_lib.c:7726, `step`). The blanket refusal (`trace_ai_line`) is
  lifted. Remaining: a copy the AI refuses on the holder's side (e.g. Filter copied by the player,
  Tinted Lens by the trainer: `ai_refusals` is side-indexed, turn-0 state only):
  `coverage.trace_copy_refusals` runs `ai_refusals` with each copy the holder can make
  (`ChooseTraceTarget`'s Forecast / Trace / Multitype skipped, battle_lib.c:7750-7775) in place
  and lists lines the armed state lacks; `check_coverage` adds them, prep applies them per
  candidate (`trace_copy_line`, B3-20). `check_opponent` has no Trace line (it depends on the
  player's party).
- B3-22 Stealth Rock's damage has one formula (`damage.stealth_rock_loss`), read by the switch-in
  (`effects.stealth_rock_damage`) and a newcomer's turn-start weather (`damage.entry_weather`).
- B4-1 A trainer battle's terrain is `CalcTerrain` of the player's tile
  (field_battle_data_transfer.c:467-505): `tools/terrain.py` narrows each trainer to its
  sight-line and talk tiles (trainer_encounter.c:165-343, unk_0203C954.c:62-97) or its coord-event
  rectangle (:208-225) when nothing before the battle can move the player, else the whole header.
  Roark split: Barry #2 (247-249) is TERRAIN_PLAIN; Lass Kaitlin (323) is GRASS (B4-19).
- B4-2 Metronome stays unsupported: `BtlCmd_Metronome` (battle_script.c:4535-4562) draws from move
  ids 1-467 minus the user's moves and `sCannotMetronomeMoves` (battle_lib.c:7152-7178): 442
  candidates in both games, 145 (Kaizo) / 174 (vanilla) with unsupported effects; no partial set
  is sampled.
- B4-3 SmellingSalt, Facade, Brine (and Kaizo's Wring Out, effect 221) set `powerMul` 20 before
  CalcDamage (effect_script_0171.s:6, _0169.s:5, _0221.s:5-8: a paralyzed defender, a user with
  `MON_CONDITION_FACADE_BOOST`, `curHP <= maxHP / 2` with a plain `/`), else 10
  (`damage.script_power_mul`); a burned Facade user is still halved (battle_lib.c:6978).
  SmellingSalt's cure is an ON_HIT indirect effect on a defender with HP left
  (subscript_heal_target_paralysis.s:5-6).
- B4-4 Vanilla Wring Out / Crush Grip (effect 237): `1 + 120 * curHP / maxHP` of the defender
  (battle_script.c:7265); Grass Knot / Low Kick: `sWeightToPower` on the defender's species weight
  (battle_script.c:6808-6826, battle_lib.c:150). The AI's estimate keeps the table power for both
  (no case in the engine's `TrainerAI_CalcDamage` path).
- B4-5 Super Fang deals `Divide(curHP, 2)` under `SYSCTL_IGNORE_TYPE_CHECKS` (effect_script_0040.s):
  no crit, roll, STAB or Life Orb; immunity still applies (battle_lib.c:7544-7552); resist berry
  skipped (subscript_type_resist_berry.s:5).
- B4-6 OHKO moves (`BtlCmd_TryOHKOMove`, battle_script.c:4334-4386): no stage, ability, item or
  weather accuracy factor; hit when `RandNext() % 100 < accuracy + (Lu - Ld)` at `Lu >= Ld`,
  always at `Lu >= Ld` with No Guard on either side; ignorable Sturdy blocks them; damage is the
  defender's max HP (bcp.c:3343-3345), a full-HP Focus Sash endures it; no crit; resist berry
  skipped (subscript_type_resist_berry.s:6); type-chart immunities apply.
- B4-7 False Swipe caps a KO at 1 HP inside UpdateHP before the endure checks (bcp.c:3381-3384); at
  1 HP it deals 0 with UpdateHP run (SYSCTL_MOVE_HIT: Life Orb, Shell Bell), but records 0 damage
  taken, so no Sticky Barb transfer, no King's Rock flinch.
- B4-8 Earthquake's x2 against a digging target (effect_script_0147.s) is never met (Dig
  unsupported): a plain hit. Poison Fang: badly-poison secondary at the move's chance
  (effect_script_0202.s, subscript_badly_poison.s).
- B4-9 Swagger (subscript_swagger.s): Attack +2 below +6, then confusion (Random 3, 2) unless
  confused; at +6 and confused it fails (MOVE_STATUS_FAILED, no AfterMoveEffects). Dragon Dance:
  Attack +1, Speed +1, each skipped at +6; both at +6 is MOVE_STATUS_NO_MORE_WORK.
- B4-10 Filter shares Solid Rock's `Divide(damage * 3, 4)` (battle_lib.c:2657-2661, ignorable);
  Sniper makes the crit x3 (:7142-7144, Battler_Ability); crit items add 1 stage (Scope Lens, Razor
  Claw) or 2 on their species (Lucky Punch on Chansey, Stick on Farfetch'd) (:7123-7128).
- B4-11 Magic Guard (Battler_Ability / CheckAbility, never ignorable) skips: Leech Seed's drain
  (bcp.c:1381), full paralysis before its draw (:2652-2654), poison, toxic (counter still rises,
  :1405-1415) and burn damage (subscript_poison_damage.s:5, _burn_damage.s:5), Bind
  (subscript_bind_effect.s:5), recoil (subscript_recoil_1_2.s, _1_4.s:6; not Struggle's), Rough
  Skin on its holder (battle_lib.c:4259), Life Orb, Black Sludge, Sticky Barb, weather chip, and
  every entry hazard (subscript_hazards_check.s:5).
- B4-12 Roost (subscript_roost.s): `Divide(maxHP, 2)` through subscript_recover_hp.s ("HP is full!"
  is NO_MORE_WORK: no flag, no AfterMoveEffects), then `turnFlags.roosting`, read only by
  `BattleSystem_ApplyTypeChart` (`BasicTypeMulApplies`, battle_lib.c:2540-2543: every row whose
  defending type is Flying is skipped, read as the never-listed Mystery type, `damage.chart_types`)
  for the rest of the turn; STAB, grounding and hazards still see Flying. turnFlags live within
  `step`'s turn (the first actor's `roosted` event), as `protecting` does.
- B4-13 Tailwind (subscript_tailwind_start.s): both TAILWIND bits, counter 3 counted down by the
  field pass (bcp.c:1039-1058), x2 Speed (battle_lib.c:1274-1276); already up is MOVE_STATUS_FAILED
  (no AfterMoveEffects). The end-of-turn order reads it (the turn's order was fixed before).
- B4-14 Spikes (max 3) and Toxic Spikes (max 2) add a layer to the defender's side, else fail
  (battle_script.c:5394, :7598). subscript_hazards_check.s on every switch-in (switch, drag,
  replacement): nothing for Magic Guard (:5); a grounded newcomer (Gravity, or no Levitate
  (CheckAbility) and no Flying type, :6-11) meets Toxic Spikes (a Poison type absorbs them; one
  layer poisons, two badly poison, stopped by its own Immunity, Leaf Guard in sun, a status, Poison
  / Steel) then Spikes (`Divide(maxHP, (5 - layers) * 2)` with HP left); then Stealth Rock on any
  newcomer with HP (`effects.entry_hazards`).
- B4-15 Destiny Bond (`destiny_bond`, volatile): set by subscript_destiny_bond.s:9 (repeats don't
  fail), cleared at the holder's next `CHECK_STATUS_START` (bcp.c:2425) whatever the checks find; a
  defender that fainted in the attacker's action with the flag takes a living attacker of the other
  side down after the hits (LOOP_FAINTED, bcp.c:3820-3823;
  subscript_faint_check_destiny_bond.s:5-7), before Shell Bell / Life Orb; Pursuit's hit on a
  switcher too (subscript_pursuit.s:85). Residual faints never trigger it.
- B4-16 Snore: a sleeping user goes ahead (bcp.c:2455, `moveCur == MOVE_SNORE` or `moveTemp ==
  MOVE_SLEEP_TALK`); the script fails (MOVE_STATUS_FAILED) unless the user is still asleep
  (effect_script_0092.s:5), so a user that woke this turn fails; else a 30 % indirect flinch.
- B4-17 Secret Power: a plain hit whose indirect TO_DEFENDER effect at the move's chance is
  `sTerrainSideEffect[min(terrain, TERRAIN_SPECIAL)]` (battle_script.c:8063-8075,
  to_secondary_effect.h): accuracy -1 (plain, sand), sleep (grass, puddle), flinch (mountain,
  cave), freeze (snow, ice), Attack -1 (water), paralysis (building, special), Speed -1 (great
  marsh), evasion -1 (bridge) (`ids.SECRET_POWER_EFFECTS`, `order.move_effects`).
- B4-18 Counter, Mirror Coat, Metal Burst (Kaizo's Bide is effect 227: Metal Burst, Dark, single
  target) read the user's `turnFlags` damage taken this turn (bcp.c:3413-3434: each UpdateHP
  assigns battleCtx->damage after the endure caps, by moveCur's class and to lastDamageTaken; a
  multi-hit leaves its last hit; Pursuit's subscript records none) as the first actor's events:
  2x physical / 2x special / `last * 15 / 10` (battle_script.c:4627, :4680, :7166); 0 taken or a
  fainted attacker is MOVE_STATUS_FAILED; SYSCTL_IGNORE_TYPE_CHECKS (no STAB or multiplier,
  immunity stays, none for a RANGE_USER move), no crit or roll, no resist berry. A
  RANGE_SINGLE_TARGET_SPECIAL one is its own defender until its command, so it never meets the
  no-target check and writes the foe's moveHit only on success.
- B4-19 A trainer's battle terrain is its first battle's: a split's fight is the first encounter; a
  Vs. Seeker rematch is not a split fight (manager's decision). A rematch row's base keeps its own
  facings; only a rematch-only id is narrowed with the spun facings (`tools/terrain.py`
  `narrowed_terrains`). Lass Kaitlin (323): LOOK_SOUTH, sight 6, sight and talk tiles GRASS. Kept
  fact: `NoUniqueRematches(TRAINER_LASS_KAITLIN)` (vs_seeker.c:252) rematches her as 323 once
  beaten, spun (:659-773), and that rematch can start on PLAIN tiles. Mixed trainers 217 -> 212
  (vanilla), 242 -> 235 (Kaizo). CalcTerrain with its caller is field_battle_data_transfer.c:467-505
  everywhere. Each `SAFE_COMMANDS` handler is cited (`tools/terrain.py` `_SAFE_HANDLERS`); none
  moves the player (LockAll / LockLastTalked only let a step under way finish;
  `MapObject_StepDir` moves the tile at the step's start, map_object_move.c:849-858).
- B4-20 The foe's SmellingSalt cure (effect_script_0171.s:12, subscript_heal_target_paralysis.s:5-6)
  frees the side's end speed from paralysis before the end-of-turn sort
  (battle_controller_player.c:888): `enumerate` `para_free`.
- B4-21 Sturdy against an OHKO move (BtlCmd_TryOHKOMove, battle_script.c:4340-4341,
  MOVE_STATUS_STURDY, no roll) is revealed to the AI: subscript_missed.s:67-69 prints "{0} was
  protected by {1}!" (TAG_NICKNAME_ABILITY, BattleAI_SetAbility, battle_script.c:11798-11806)
  unless protected (:18), Levitate (:34) or ineffective (:59) prints first (`step._use_move`,
  `hooks.OHKO_BLOCKED`).
- B4-22 Immunity, or Leaf Guard in sun without a weather lock (CheckAbility, not ignorable), stops
  Toxic Spikes' poison before the status and type checks and prints "{0}'s {1} prevents
  poisoning!" (subscript_poison.s:5-10, :80-82; subscript_badly_poison.s:5-10, :93-103): the AI
  learns the ability (`effects.entry_hazards`).
- B4-23 Counter, Mirror Coat, Metal Burst read the turnFlags last attacker, stored as `1 + battler`
  with the amount (bcp.c:3413-3434), and fail on 0 taken, an attacker of the user's side or one
  with no HP (battle_script.c:4619-4625, :4672-4678, :7157-7162): `phys_from` / `spec_from` /
  `last_from` events (convention 3).
- B4-24 No Counter-family move is RANGE_USER in either game (Counter, Mirror Coat, Metal Burst:
  RANGE_SINGLE_TARGET_SPECIAL; Kaizo's Bide: RANGE_SINGLE_TARGET), so the chart's RANGE_USER skip
  (bcp.c:2352) is never met: removed from `effects.play_hit` and the reference.
  `damage.script_power_mul` takes the defender (convention 1).
- B5-1 Sleep Talk, Metronome, Copycat (`EffectId.SLEEP_TALK` / `METRONOME` / `COPYCAT`) run the
  move their script picks through GoToMoveScript (battle_script.c:2111-2134) as Nature Power does:
  from the accuracy check on, everything reads the called move (moveCur: Protect, chart, damage,
  effects, moveHit, the Protect chain); status checks, PP, priority, movePrevByBattler, the
  battle's movePrev and the Choice lock stay the invoker's (moveTemp); a called move spends no PP.
  Their own defender is the user (RANGE_USER), so the no-target check passes and the foe's moveHit
  is written only when the called move's re-picked defender is the foe. Sleep Talk
  (effect_script_0097.s:5) fails unless the user is asleep now (one that woke this turn fails);
  BtlCmd_TrySleepTalk (:4907-4937) draws uniformly (65536 % 4 == 0) over slots not
  `MoveFlag.NO_SLEEP_TALK` (Move_IsInvoker, Focus Punch, Uproar, Chatter, and Move_IsMultiTurn by
  the move's decomp effect, :4918-4925; battle_lib.c:2975-2992, :3032-3045) and not flagged by
  CheckInvalidMoves without NO_PP (disabled, power 0 while taunted, a Choice holder's other moves;
  :4927); none fails the move. `Outcome.call` holds u in 0..11; the call is allowed slot
  `u % count` of the set the move sees when it runs (12 is a multiple of every count 1-4). Copycat
  (BtlCmd_TryCopycat :7328-7341) reads `BattleState.move_prev` (battle-wide `movePrev`) at its
  turn and fails on MOVE_NONE or a move of sCannotMetronomeMoves (every invoker is one). movePrev
  is set on every acting path by UpdateMoveBuffers (bcp.c:3936-3942) and UpdateFlagsWhenHit
  (:4499-4508): moveTemp when the attack message showed, else MOVE_NONE; Pursuit's pre-switch hit
  (battle_script.c:6931-6935), switches and items leave it. A failed call is MOVE_STATUS_FAILED
  ("But it failed!", subscript_missed.s). The id lists are code constants in both games
  (rom_diff.md: the functions are unchanged; Kaizo's renamed Snatch / Covet / Trick / Switcheroo
  slots stay excluded), `MoveFlag` bits at table build, checked against the decomp
  (tests/tools/test_effect_map.py).
- B5-2 Metronome (BtlCmd_Metronome :4535-4562) draws `RandNext() % 467 + 1` until the move is not
  the user's and `Move_CanBeMetronomed` allows it: candidate m weighs 141 (m <= 65536 % 467 = 156)
  or 140 of 65536 (`order.metronome_weights`); `Outcome.call` holds its rank. Modelled in both
  engines, sampler and enumerator (one call variant per candidate; the set cannot change within a
  turn). Coverage refuses a holder while any candidate would be refused as its move
  (`coverage.metronome_calls`); the terrain, Kaizo ROM-change and halved-Defense checks run over
  the candidates too (`fight_calls`). 150 (vanilla) / 128 (Kaizo) of the 442 candidates have
  unsupported effects, so every Metronome stays refused (supersedes B4-2's counts); no partial set
  is sampled.
- B5-3 The call laws are read at turn start (`enumerate.call_laws`, `call_variants`,
  `enumerate_turn`; `outcome.sample_outcome` draws the same): each variant is one
  `outcome_laws(..., call)` enumerated and weighted, rows carrying their `Outcome.call`; the
  variants together are held to `WIDTH_CAP`. Copycat draws nothing: its two orders (its foe moved
  before it, or not) are variants keeping only their consistent `speed_tie` branches. The only
  implemented change of Sleep Talk's set within a turn is a faster foe's Disable
  (CheckInvalidMoves, battle_lib.c:2266-2269), which the turn-start laws cannot follow (the called
  move's draws are laid out at the start): a Sleep Talk user facing a reachable Disable (a Disable
  move, a Copycat with Disable in the fight, a Metronome that can call it) is refused
  (`coverage.call_disable_line`; prep's pool and problem per candidate), as Trace in both parties
  is (B3-14). **Refusal superseded by B5-13** (the Disable is modelled). The engines play the live
  set exactly. The sampler and the laws read Copycat's call by the order (`order.used_moves`).
- B5-4 Uproar (`EffectId.UPROAR`, effect 159 in both games, the move Uproar only): a plain hit
  (CalcCrit, CalcDamage; effect_script_0159.s:9-10) whose ON_HIT indirect effect, for a user not in
  an uproar yet (:5), runs when the hit was not `MOVE_STATUS_NO_EFFECTS` (battle_lib.c:1524-1529:
  not missed, protected or ineffective): subscript_uproar.s sets `moveLockedInto` = moveCur (:5),
  `Random 3, 3` (3-6, `Outcome.duration`, the Bind law `DURATION_TRAP`, :6-8) into
  `VOLATILE_CONDITION_UPROAR` (`BattleState.uproar`, per battler) and the battler's
  `FIELD_CONDITION_UPROAR` bit (:9-11, condition.h:118, :147), set and cleared with the count, so
  it is `uproar != 0`. The first use spends PP; later turns spend none (`Battler_CanPickCommand`
  FALSE, battle_lib.c:6224-6235, sets `ppDecremented` at selection, bcp.c:335).
- B5-5 The lock: `Battler_UnlockMoveChoice` (battle_lib.c:1834-1841) clears only
  `VOLATILE_CONDITION_MOVE_LOCKED` (and Bide, Rollout), never the Uproar count or
  `moveLockedInto`, so every disruption (flinch, Disable, confusion, paralysis, love, Truant,
  freeze, miss, Protect, no target) leaves an uproarer locked: `effects.unlock_move` keeps
  `locked_move` while `uproar` is up. `FightCommand` runs `moveLockedInto` whether or not the
  battler knows it (bcp.c:1885-1888; `Battler_SlotForMove` finds no slot, nothing is spent,
  :2272-2292), so a lock a call made (Copycat calling Uproar, Bide or Rollout: moveLockedInto is
  moveCur, battle_lib.c:1828-1832) is still a lock: `actions.locked_slot` gives action 0 then (the
  reference's `_locked_slot` too).
- B5-6 No sleep in an uproar: a sleeper's status check wakes it at once while
  `FIELD_CONDITION_UPROAR` is set (a faster uproar this turn included), without the counter drop,
  and the checks go on (bcp.c:2432-2439, subscript_wake_up.s:19 clears every sleep bit;
  Soundproof, the exception, is unsupported). subscript_fall_asleep.s blocks a sleep on a target
  not yet asleep (:43-45 after Insomnia, Shield Dust and the asleep test; :14-16 for a move
  effect): a direct one prints "kept awake" and sets `MOVE_STATUS_NO_MORE_WORK` (:_292, :_330: no
  AfterMoveEffects, `step`'s `kept_awake`), an indirect one (Secret Power's) ends silently
  (`effects._inflict`). Rest and Yawn (unsupported) read the bit too (Owed).
- B5-7 Turn end: `MON_COND_CHECK_STATE_UPROAR` (bcp.c:1505-1541) runs after Bind (Bad Dreams,
  between, unsupported) and before Disable (Thrash, between, unsupported): for a battler in an
  uproar, every battler asleep with HP wakes, in battler order, repeating until none sleeps
  (:1508-1521); then the count drops by 1 and the uproar ends (subscript_uproar_end.s, volatile
  and field bits cleared, :1525-1535) when the count reached 0 or the battler's `moveFailFlags`
  this turn are set (`BattleContext_MoveFailed`, battle_lib.c:2808-2820: flinch, Disable, Taunt,
  confusion self-hit, full paralysis, love, bcp.c:2531-2681, and `noEffect`: CheckTypeChart found
  the move ineffective, bcp.c:2350-2370, run once the move passed the no-target check, hit or
  miss, :3271-3277); sleep, freeze, Truant, a miss and Protect set none. Ending frees the command
  (`locked_move` 0: no MOVE_LOCKED was set). The flags are the turn's `move_failed` event
  (`step._use_move`), cleared every turn (`SetupNextTurn`, battle_lib.c:2232). Neither check
  changes HP, so Uproar and Disable share one faint check (`step._uproar_then_disable`). The
  player sees the uproar ("caused an uproar!", "calmed down."): `uproar`'s non-zero-ness is in
  the observation.
- B5-8 Future Sight / Doom Desire (`EffectId.FUTURE_SIGHT`, effect 148: vanilla 80 / 90 % and
  120 / 85 %, Kaizo 120 / 100 % and 250 / 100 %, no `MOVE_FLAG_CAN_PROTECT` in either): once past
  the no-target check, `BtlCmd_TryFutureSight` (battle_script.c:6065-6100) fails ("But it
  failed!", subscript_but_it_failed.s: `MOVE_STATUS_NO_MORE_WORK`, no AfterMoveEffects, no miss
  shown) while the defender's battler has a count; else it sets the defender side's
  `SIDE_CONDITION_FUTURE_SIGHT` (`future_sight`) and, on the defender's battler, 3 turns,
  moveCur, the attacker (`1 + battler`) and the damage of that moment:
  `BattleSystem_CalcMoveDamage` with criticalMul 1 and the defender side's and field's conditions
  (:6079-6089: no crit, type chart, STAB or Life Orb), then `BattleSystem_CalcDamageVariance`
  (:6090, battle_lib.c:7081-7094: `Outcome.roll[side, 0]`; the + 2 keeps it nonzero, so its sign
  never flips); Helping Hand is unsupported. The script then sets `SYSCTL_NONSTANDARD_ACC_CHECK`
  and `SYSCTL_IGNORE_IMMUNITIES` (effect_script_0148.s:7): no accuracy roll (bcp.c:2911-2913), no
  type chart (:2354, so no `noEffect`), and the use deals nothing (no CalcDamage:
  `ids.UNROLLED_EFFECTS`, `damage.fixed_damage`); its ON_HIT effect only prints. The fields are
  field state: they stay when either battler leaves.
- B5-9 The hit: `SIDE_COND_CHECK_STATE_FUTURE_SIGHT` (bcp.c:1758-1783), the first side-condition
  check, after `CheckMonConditions` (:1733), before `TurnEnd`: per battler in monSpeedOrder
  (end-of-turn order) with a count, the count drops; at 0 a battler with HP clears its side's bit
  and runs subscript_future_sight_damage.s: `CheckMoveHit` (:8; battle_script.c:6115-6135,
  bcp.c:217-221) of whatever battler is now in the stored attacker's slot (its stages, abilities,
  items; a fainted one included) against the defender with the stored move, an ordinary check (no
  SYSCTL flag up at turn end; Zoom Lens applies, every battler has moved,
  battle_lib.c:5365-5368); a new roll, `Outcome.future_hit[battler]`
  (`outcome.future_threshold`). Protect on that defender this turn stops a CAN_PROTECT move (none)
  and unlocks the attacker's battler (bcp.c:3005-3011); else either battler's No Guard clears the
  miss (:3016-3021). A miss is "But it failed!" (:53-59). Else Substitute and Rage (unsupported),
  `CheckHoldOnWith1HP` (:28: a full-HP Focus Sash leaves 1 HP, battle_script.c:9098-9124; Focus
  Band unsupported), UPDATE_HP with no resist berry (`waitingBattlers` is 0 after the actions,
  bcp.c:881-886; subscript_update_hp.s:9-10), then under `MOVE_STATUS_ENDURED_ITEM` the Focus
  Sash's removal (:40-48). No held-item check follows.
- B5-10 One pass, one set of flags: `moveStatusFlags` is cleared only by `BattleContext_Init`
  (battle_lib.c:1983; the last MoveEnd, bcp.c:3998), so when both battlers' hits fall due, the
  second reads the first's: a first miss or Protect fails the second (No Guard still clears a
  carried miss), a first endured hit removes the second defender's Focus Sash. Between them,
  `AnyFainted` (:1749, `onlyFaint`) faints the first defender without `CheckBattleOver`, so the
  second hit still lands: two last Pokemon down is a draw, a loss (ruling 2); `TurnEnd` pays the
  exp and checks the end (:1844-1847). A battler with no HP when its count runs out takes nothing
  and keeps its side's bit, which only a later hit clears (`future_sight`).
- B5-11 Next turn's `ShowBattleMon` (bcp.c:256-268) runs `BattleSystem_TriggerEffectOnSwitch`, whose
  `SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS` (battle_lib.c:4151-4165) checks every battler's item
  before command selection; a berry the Future Sight hit brought into range (it comes after
  `MON_COND_CHECK_STATE_ITEM_CONDITION`) is eaten there (`step._turn_start_items`, end of `step`;
  with a replacement pending, `replace`'s switch-in check runs it after the newcomer's hazards, as
  `ReplaceFainted` precedes `ShowBattleMon`). Without such a hit it finds nothing new: every earlier
  turn-end change was followed by an item check.
- B5-12 Laws: the use draws `roll[s, 0]` (16-value law; its "move" read needs only that the move
  runs) when the target's battler has no count at turn start (only the user's own Future Sight can
  set it before its move); no `hit` or `crit`. `future_hit[d]` is read when `future_timer[d]` is 1
  at turn start; its law is the 1-100 roll as classes over `outcome.future_thresholds`: every
  attacker / defender battler the slots can hold at turn end (the active, its switch target, any
  living one a foe's Roar can drag), every stage sum when an accuracy / evasion stat move, Foresight
  or a White Herb is in play (else the start stages or 0), the weather of the start, none, sandstorm
  or hail, the target confused or not; a Trace holder among them keeps every roll (`future_loose`).
  `sample_outcome` draws the same.
- B5-13 (supersedes B5-3's refusal) Sleep Talk facing a faster Disable is modelled.
  `BtlCmd_TryDisable` (battle_script.c:4581-4598) disables only the defender's movePrevByBattler
  (:4586), only while that move has PP and nothing is disabled (:4588-4591). A Sleep Talk user
  moving second has not moved, so the move is its turn-start `last_move`; its set at its move is the
  turn-start set or that set minus the last move's slot (`enumerate._disabled_state`;
  CheckInvalidMoves, battle_lib.c:2266-2269, via `order.sleep_talk_slots`). `call_laws` gives `alt`,
  each value's call from the second set, where it can arise (nothing disabled, a last move, the foe
  uses a move: any move can be an invoker calling Disable). `call_variants` splits a value group
  whose two calls differ into two variants of the group's weight, `CALL_KEEP` (value + 12) and
  `CALL_DISABLED` (value + 24); `outcome_laws` lays out the second one's reads from the shrunk set.
  `order.invoked_move` takes `call % count` (count 1-4 divides 12 and 24), so `step` plays every
  encoding of a value alike, on the live set; the sampler still draws 0-11. Each row is kept only in
  the variant whose assumption its turn met: `step`'s `disabled_at_move` event (a move of a battler
  disabled when its action starts) against the variant (`enumerate.call_consistent`; `expand`'s step
  program returns it, the merger drops the rest, `enumerate_turn` filters). Removed:
  `coverage.call_disable_line` and its calls in `check_coverage`, `prep/pool.py` and
  `prep/problem.py` (the review's missing `player_extra` there is moot). The brute force
  (`tests/engine/test_invokers.py::_raw_brute`) does not share `branch_ok` / `after_foe`: it loops
  over the raw `speed_tie`, takes the order from `order.action_order` as `step` does, derives
  Copycat's movePrev from it (its foe used a move before it), forces every branch open and cuts the
  rows to that `speed_tie`; hand cases add a faster Disable and a foe switching before Copycat.
- B5-14 Zoom Lens on the Future Sight hit (`outcome.future_threshold`, reference `_hit_rate(...,
  True)`): no change. `Battler_MovedThisTurn` (battle_lib.c:5365-5368) reads
  `battlerActions[d][PICK_COMMAND] == BATTLE_CONTROL_MOVE_END`; every action's MoveEnd sets it for
  the actor's slot (bcp.c:3989), a chosen switch included; `UpdateAfterSwitch` sets it for a battler
  dragged in by Roar / Whirlwind (battle_lib.c:2053), `PlayFaintAnimation` for a fainted one
  (battle_script.c:1801); only the next command selection rewrites it (bcp.c:337-512). So at the
  turn end's check (bcp.c:2963-2965) every battler has moved, a newcomer included
  (`tests/engine/test_batch5_review.py`).
- B5-15 Generator (`tests/strategies.py`): with an uproar under way no active Pokemon sleeps at turn
  start (B5-7's turn-end wake-up), also after the other side's draws.
- B6-1 (departure from the design) The turn scratch is flat `BattleState` fields `ts_*`
  (`state.TURN_SCRATCH`), not a nested `TurnScratch` field: `LAYOUT`, the reference state and the
  converters stay flat. One field beyond the design: `ts_move_failed` (moveFailFlags,
  BattleContext_MoveFailed battle_lib.c:2808-2820), read by the turn end's Uproar check (bcp.c:1525)
  after the resume. `ts_custap` keeps only a later actor's flag (an earlier one ate its berry,
  subscript_check_quick_claw.s:26-28); `ts_item` is 0 for a non-item action.
- B6-2 U-turn (`EffectId.U_TURN`, effect 228 in both games): its ON_HIT secondary effect runs at
  ONE_HIT_TRIGGER_SECONDARY (bcp.c:3484-3494) for a hit without MOVE_STATUS_NO_EFFECTS
  (battle_lib.c:1524-1529; `MoveHit.effective`). subscript_attack_then_switch_out.s: the defender
  side not blacked out (:5, BtlCmd_CheckBlackOut battle_script.c:8846) and a replacement for the
  user (:6, battle_lib.c:3125); the contact ability (:7, the hit's own: every TriggerAbilityOnHit
  case is guarded by SYSCTL_UTURN_ACTIVE, battle_lib.c:4219-4362, and the later call needs a living
  attacker when no switch happened, so it runs once); the pivot item (:11, `effects.pivot_item`);
  Destiny Bond (:15-34); a fainted user switches nothing (:48) and the action goes on, King's Rock
  included (CheckExtraFlinch has no attacker HP test, bcp.c:4630-4650); the foe's Pursuit (:54,
  BtlCmd_TryPursuit battle_script.c:6884-6955, the pursuer not yet acted: `_pursuer_ready`) with its
  faint check and payout (subscript_pursuit.s:106-108); a user alive after it (:55) gets its Natural
  Cure (:56-57) and the list (:71). A battle the Pursuit's faint check ends opens no list (the
  engine ends a battle at the faint check that finds a side out, `step._faint_check`).
- B6-3 Baton Pass (`EffectId.BATON_PASS`, effect 127): a direct ON_HIT effect at
  BATTLE_CONTROL_PRIMARY_EFFECT (bcp.c:3298-3308); subscript_baton_pass.s: no replacement fails (:5,
  :19-20: MOVE_STATUS_FAILED, subscript_missed.s, no AfterMoveEffects), else Natural Cure (:7-8) and
  the list (:17); no Pursuit. The switch (UpdateAfterSwitch under SYSCTL_BATON_PASS,
  battle_lib.c:2056-2086, :2105-2115; InitBattleMon :97-101) keeps, of the engine's fields, the stat
  stages, `confusion` (condition.h:72), `leech_seed`, `mud_sport`, `water_sport`, `aqua_ring`
  (moves.h:46-48) and `substitute_hp` (:2107) (`step._BATON_PASSED`); Focus Energy, Mean Look,
  Curse, Lock-On, Perish Song, Ingrain, Power Trick, Embargo, Heal Block, Magnet Rise and Gastro
  Acid come from unsupported moves (owed with them).
- B6-4 The resume (`step(paused, actions)`, `actions[pivot]` the switch to the pick): the list's
  SwitchAndUpdateMon, PokemonSendOut, the hazards and the faint check
  (subscript_show_party_list.s:9-24); that check plays the faint of a defender the U-turn KO'd (its
  SYSCTL_MON_FAINTED bit is set at UpdateHP, BtlCmd_TryFaintMon battle_script.c:1763-1777) before
  the tail; its exp is paid at MoveEnd (AnyExpPayout, bcp.c:3968). `ts_alive` is the party bitmask
  of the faints handled so far (the action's start, or after U-turn's Pursuit), the `before` of the
  list's faint check and of the MoveEnd payout. Then the tail on the newcomer: King's Rock with its
  item, the switch-in effects at TRIGGER_SWITCH_IN_EFFECTS (bcp.c:3672-3682), the item checks and
  the thaw (the guarded Sticky Barb, battle_lib.c:5393, Shell Bell bcp.c:4739 and Life Orb :4761
  skipped), UpdateMoveBuffers on the newcomer (bcp.c:3917-3944: movePrevByBattler, movePrev and
  moveProtect written after UpdateAfterSwitch cleared them, battle_lib.c:2119-2123), MoveEnd.
- B6-5 The Choice lock (bcp.c:3921-3934) tests `moveTemp`: U-turn under SYSCTL_UTURN_ACTIVE (set at
  :68, so only a resumed U-turn) and Baton Pass under SYSCTL_MOVE_SUCCEEDED (one whose attack
  message showed, a failed one included) never lock; a Sleep Talk calling U-turn locks the newcomer
  into Sleep Talk when it holds a Choice item.
- B6-6 (departure) The resumed half reads its own draws in the sampler too: `sample_outcome` and the
  laws read a paused state through `outcome.pivot_view` (the pivoting side switches to its pick
  first, the newcomer on the field, movePrev the paused move's moveTemp, the other side's action as
  chosen while it is still to act), and `outcome_laws` adds the paused action's King's Rock
  (`flinch_item[pivot, 0]`, with the newcomer's item param). The design's "playout loops `step` with
  the same Outcome" would draw the newcomer's King's Rock with the outgoing battler's param
  (`kings_rock_chance` reads the active's item): `playout.step_through` draws a fresh outcome of the
  paused state for each resume.
- B6-7 The search: `stage_b`'s pending kind `"pivot"` (before learn and replacement: a pause is
  mid-turn); `decision_actions` offers the player's `legal_replacements` at its own list and the
  placeholder action 0 at the opponent's; an answer is the turn request `(state, answer, 0)` through
  `expand` (a chance node), the opponent's pick made in the row program (`playout.turn_inputs`:
  `post_ko_switch_in` with `ts_pick_mul`, refused as a replacement's; `aiSwitchedPartySlot` is 6
  there, `Task_TrainerShowPartyMenu` battle_display.c:4543-4556), the turn's powerMul and AI switch
  read from `ts_power_mul` / `ts_switched`. A paused child runs no learnset scan and no replacement.
  A playout from a paused root resumes it first (`playout._play`'s `pick`), not counted as a turn;
  `leaves`' forced turns go through `turn_inputs` and `step_through` too.
- B6-8 A locked battler never Struggles (it picks no command, bcp.c:335, so the STRUGGLING_ALL test
  of :433 is not reached): `step` masks `selected_move`'s struggling with the lock, as the reference
  engine's `_choose` did (a value the scratch keeps).
- B6-9 `step._execute_actions` is a `while_loop` over (position, phase): a U-turn that reaches its
  foe's Pursuit stops after its head; the next pass plays the Pursuit at the one Pursuit site (a
  switch's pre-switch hit shares it), then the action's tail (`_use_move`'s `MODE_PURSUIT`); a list
  ends the loop with the scratch written. `_use_move` and `_act` stay one trace each: `MODE_PURSUIT`
  / `MODE_RESUME` drop the head's state at the boundary before the hit loop and read the paused
  inputs (`ts_used`, `ts_damaged`, `ts_dealt`) from the scratch; Roar's drag and the resume share
  one `hooks.switch_in_effects` site.
- B6-10 (B6 review) The resumed half's laws read the newcomer as the switch leaves it:
  `outcome.pivot_view` applies the pivot's switch to the viewed state (its stat stages and
  `state.VOLATILE_FIELDS` reset, a Baton Pass keeping the stages and `BATON_PASSED_FIELDS`:
  InitBattleMon battle_lib.c:97-101, UpdateAfterSwitch :2078-2079, :2107), and
  `outcome.fresh_switch` tells the thresholds (`_hit_threshold`, `hit_thresholds`, `_rate`'s
  confusion for Tangled Feet and Foresight) that this switch's battler is not fresh: its evasion is
  the view's, as `step`'s live check reads it (bcp.c:2873-2874). `future_thresholds` reads the
  view's stages too. Test: tests/engine/test_pivot_review.py (every enumerated row stepped against
  `live_hit_threshold`).
- B6-11 (B6 review) Stage B groups children by `(observation, refused, paused)`: the observation
  does not carry `pivot_battler`, so a group must not mix a paused state with a finished one
  (`_paused`).
- B6-12 (B6 review, doubles conventions) `ts_acted` (per battler, Battler_MovedThisTurn
  battle_lib.c:5365-5368: the actions before the paused one, MoveEnd bcp.c:3989) replaces
  "`ts_cursor` is 0" as "the target has not moved" in `pivot_view` and `enumerate._paused_rock`;
  `ts_switched` is per battler (was `ts_ai_switched`, side 1 only); `decision_actions` reads the
  pivot's side (1 + battler, battler b on side b in singles), not "2 means the opponent"; `_act`
  takes the battler U-turn's Pursuit reads (`foe`) from its caller.
- B6-13 (B6 review) Hand tests of both engines (tests/engine/test_pivot_review.py): Destiny Bond on
  U-turn (subscript :15-34), Rough Skin KO-ing the user (:48, `rock_now`), a U-turn KO with the
  list's faint check and the MoveEnd payout, hazards KO-ing the newcomer, two pauses in one turn,
  Pursuit KO-ing the user, Sleep Talk calling U-turn with a Choice newcomer (B6-5), Baton Pass
  without a bench, Natural Cure at the list, Shell Bell and Sticky Barb at the pivot item
  (battle_lib.c:7336-7386), Roar's drag: its hazards
  (subscript_force_target_to_switch_or_flee.s:31), then its switch-in effects at AfterMoveEffects
  (bcp.c:3672-3682), the site the resume shares. Generator: a drawn pivot half the time gets a
  living bench and a user free of sleep, freeze, confusion and love.
- B7-1 `ai_item_known` (aiContext.battlerHeldItems, per player party slot, the item id):
  `BattleMessage_ItemTag` (battle_script.c:11726-11745) runs `BattleAI_SetHeldItem` (:12119-12122)
  for an item tag of `BTLSCR_ATTACKER`, `_DEFENDER` or `_MSG_BATTLER_TEMP`, never for `_MSG_TEMP`
  (msgItemTemp), nor for a message built in C (Knock Off :6622, the pinch berries' stat message
  :2857, the Choice refusal battle_lib.c:2362). Of the modelled items: Leftovers and Black Sludge at
  the turn end (subscript_restore_a_little_hp.s:10, subscript_lose_hp_from_item_with_message.s:10;
  battle_lib.c:4955), Sticky Barb's damage (:5340) and Shell Bell's heal (bcp.c:4745,
  battle_lib.c:7352). Not revealing: Quick Claw (no message, subscript_check_quick_claw.s:6-10),
  Life Orb (subscript_lose_hp_from_item.s), the orbs (subscript_badly_poison.s:77,
  subscript_burn.s:73), every berry's cure or heal, Frisk (subscript_frisk.s:7). Custap (:17), Focus
  Sash (subscript_move_followup_message.s:16, subscript_future_sight_damage.s:44), the resist
  berries (subscript_type_resist_berry.s:113) and Pluck's victim (subscript_pluck.s:7) are named,
  then removed (B7-2).
- B7-2 The record is cleared by `BattleMon_CopyToParty` of a battler holding nothing
  (battle_lib.c:1819-1823), which every removal runs: `RemoveItem` (battle_script.c:8979-8981; the
  berries through subscript_pluck_check.s) and the Sticky Barb transfer's `UpdateMonData`
  (:3077-3078, subscript_transfer_sticky_barb.s:7); and on entry (`UpdateAfterSwitch`,
  battle_lib.c:2153). `CleanupFaintedMon`'s clear (:2225) is played at the replacement's entry, as
  for the known moves and ability: nothing reads a fainted player battler's record before. Only the
  player's is kept: the AI reads `battlerHeldItems` for the other side only (trainer_ai.c:1907-1913,
  :1924-1928).
- B7-3 Ingrain (effect 181, effect_script_0181.s): sets `MOVE_EFFECT_INGRAIN` (:5-6); heals
  `Divide(maxHP, 16)` (Big Root applied) at `MON_COND_CHECK_STATE_INGRAIN`, before Aqua Ring
  (bcp.c:1311-1326; one faint check serves both heals); traps (`Battler_IsTrapped`,
  battle_lib.c:5523, the party menu bcp.c:572; `actions.trapped`); grounds: `BasicTypeMulApplies`
  skips an immune Flying row (:2534-2537, `damage.CHART_GROUNDED`) and `Battler_Ability` reads its
  Levitate as none (:3100-3103, `damage.ignorable`). Baton Pass keeps it (moves.h:46).
- B7-4 Ingrain, Aqua Ring, Mud Sport and Water Sport fail a user that already has the effect
  (effect_script_0181.s:5, _0251.s:5, _0201.s:5, _0210.s:5: MOVE_STATUS_FAILED): the move does not
  connect, CheckMoveFailure prints "But it failed!" and skips AfterMoveEffects (bcp.c:3312-3331), as
  Fake Out's; both engines take Fake Out's path.
- B7-5 Roar's Ingrain test (subscript_force_target_to_switch_or_flee.s:7) is `OPCODE_EQU`: only a
  `moveEffectsMask` equal to `MOVE_EFFECT_INGRAIN` anchors the defender (:86-90, NO_MORE_WORK);
  beside any other kept bit (Leech Seed, Lock-On, Perish Song, Yawn, Imprison(ed), the sports, Aqua
  Ring, Camouflage) Roar drags it out (`effects.ingrain_anchors`).
- B7-6 Mean Look, Block, Spider Web (effect 106): subscript_mean_look.s runs as a direct `CHECK_HP |
  TO_DEFENDER` effect for a living defender on a move that connected (battle_lib.c:1491-1504;
  accuracy 0, so only Protect stops it); it fails (MOVE_STATUS_FAILED, no AfterMoveEffects) on a
  defender already trapped or behind a substitute (:7-8), else sets `VOLATILE_CONDITION_MEAN_LOOK`
  and ORs the attacker into `meanLookTarget` (:9-10), stored as `1 + battler` (in singles the OR
  always holds the one foe). It traps (`VOLATILE_CONDITION_TRAPPED`, condition.h:77-78). It ends
  when the trapper leaves by a switch that is not a Baton Pass (battle_lib.c:2055-2061) or faints
  (:2170-2172); Baton Pass keeps it on both sides (:2078, :2111).
- B7-7 Imprison (effect 192, `BtlCmd_TryImprison` battle_script.c:6674-6727): the foes'
  `MOVE_EFFECT_IMPRISONED` is set first, whenever the script runs (:6686-6690), and cleared only
  with that battler's volatiles; the script fails a user already imprisoning or sharing no move with
  a battler of the other side, whatever its HP (:6692-6722). A sealed move (`Move_Imprisoned`,
  battle_lib.c:3316-3338) is invalid at selection (CHECK_INVALID_IMPRISONED, :2287-2290; Sleep
  Talk's slots too, battle_script.c:4927) and, chosen before a faster foe's Imprison, fails at
  `CHECK_STATUS_STATE_IMPRISON` after Taunt (bcp.c:2572-2583): unlock, no PP,
  `moveFailFlags.imprisoned` (the Uproar check, battle_lib.c:2812), event `ACT_IMPRISONED`.
- B7-8 `roosting` (turnFlags.roosting, battle_context.h:64) replaces `ts_roosted`: a Roost that
  healed sets it (subscript_roost.s:11-12); a faint clears it (battle_lib.c:2189-2192) and
  `SetupNextTurn` clears it (:2229-2231) after the faint replacements (bcp.c:1843-1856), so it is
  kept at a pause and while a replacement is pending (`step.roosting_now`; `replace` clears it with
  the last one), else 0. Per battler slot: a Pokemon dragged in after its side's Roost keeps it, as
  the decomp does.
- B7-9 Observation: the non-zero-ness of `ingrain` ("planted its roots!"), `mean_look` ("can no
  longer escape!"), `imprison` ("sealed the opponent's move(s)!") and `roosting`, and the
  `ACT_IMPRISONED` action kind ("can't use the sealed MOVE!"). `imprisoned` and `ai_item_known` are
  not shown.
- B7-10 Enumerator: Ingrain is an input clause (the foe's chart and Levitate change, like Roost);
  Mean Look and Imprison are input-neutral (no damage input; a sealed move stops whole, like Taunt);
  all three are status moves outside the table damage.
- B7-11 Tests: the differential test and its generator (`strategies._draw_batch7`: the moves, a
  shared move for Imprison, the five fields drawn; Roar against an ingrained foe with and without
  other mask bits; `ai_item_known` none or the held item) reach every branch; no hand test beyond
  the updated order and observation tests.
- B7-12 (fix) `_act` evaluated the move branch for every action, and a switch action (4 + slot)
  clipped to move slot 3: a Baton Pass or U-turn there set `ctl["pause"]` / `["pursue"]`, so
  `_execute_actions` replayed the switch from the newcomer (its traced or other ability restored to
  the personal one, the hazards a second time). The turn branches on the picked command
  (`battlerActions[...][BATTLE_ACTION_PICK_COMMAND]`, bcp.c:893): a switch or a bag item runs no
  move script, so only a move or a resumed action can stop. The reference was right. Test:
  tests/engine/test_batch7_review.py.
- B7-13 (B7 review) Sleep Talk's slots drop a move a foe's Imprison seals: `BtlCmd_TrySleepTalk`
  runs CheckInvalidMoves with `~CHECK_INVALID_NO_PP` (battle_script.c:4927), which includes
  CHECK_INVALID_IMPRISONED (battle_lib.c:2287-2290); JAX's `order.sleep_talk_slots` missed it (the
  reference had it). A faster foe's Imprison this turn is the set's second change within a turn: a
  third call variant `CALL_SEALED` beside B5-13's KEEP / DISABLED (`enumerate._sealed_state`,
  `CallLaws.sealed`, `can_disable` / `can_seal`), each row kept by the turn's own `disabled_at_move`
  / `sealed_at_move` events relative to the turn start (`_split_events`; one foe move a turn, so
  never both). Tests: test_batch7_review.py, the `sleep_talk_imprison` brute-force case
  (test_invokers.py); the generator gives an Imprison user a move a Sleep Talk user can call
  (`strategies._share_with_sleep_talk`).
- B7-14 (review) A Baton Pass newcomer keeps Ingrain before its hazards (battle_lib.c:2079); the
  hazards' `CheckAbility` (subscript_hazards_check.s:8; battle_script.c:3159-3172) reads
  `Battler_Ability`, which hides an ingrained Levitate (battle_lib.c:3100-3103): Spikes and Toxic
  Spikes reach it (`damage.grounded(..., ingrained)`, both engines). A switch action's newcomer,
  which the laws read before it switches, has none (`entry_weather`). The Owed line is closed.
- B7-15 (review) `meanLookTarget` is ORed (`OPCODE_FLAG_ON`, subscript_mean_look.s:10) and nothing
  clears it but the trapped battler's own switch-in or faint (battle_lib.c:2056-2061, :2170-2172
  clear only the volatile bit): `mean_look_target` keeps `1 + raw` with its stale bits (replaces
  B7-6's "1 + attacker, 0 without the bit").
- B7-16 (review) A Roar whose target is anchored at the turn start and stays so (no switch, no move
  of the target side that adds a mask bit: Aqua Ring, the sports, Imprison, Camouflage, an invoker)
  stops before TryWhirlwind (subscript_force_target_to_switch_or_flee.s:7, :86): its `roar_level` /
  `roar_target` draws are not relevant (enumerate.py).
- B7-17 (review, settled) Imprison is `RANGE_USER` (res/moves/imprison/data.json), its own defender
  (`targets_user`), so its script runs with the foe fainted mid-turn and marks it imprisoned, as
  `TryImprison` loops the battlers whatever their HP (battle_script.c:6697-6716).
  `TrainerAI_CalcDamage` calls `BattleSystem_CalcMoveDamage` (trainer_ai.c:3082), whose Technician
  (battle_lib.c:6698) the AI estimate therefore applies: the merged `damage.py` is exact there.
- B7-18 (AI-side fix) `TrainerAI_ShouldSwitch`'s gate (trainer_ai.c:3906-3907) refuses a switch for
  `VOLATILE_CONDITION_TRAPPED` (Bind and Mean Look, condition.h:77-78) and `MOVE_EFFECT_INGRAIN`;
  `ai/switching._layout` reads `mean_look` and `ingrain` beside `trapped`. The rest of the gate
  (:3908-3911: Shadow Tag and Arena Trap on the player's side, Magnet Pull on a Steel AI battler)
  was already there; the comment at :3902-3904 (the AI ignores its own Shadow Tag, a Flying type vs
  Arena Trap, Shed Shell) is the decomp's own naivety, mirrored. Tests: tests/ai/test_switching.py
  `test_gates`, the brute force of tests/ai/test_kaizo_switching.py.
- Opponent-action legality (by design): neither `step` nor `legal_actions` checks the opponent's
  action; the search feeds `step` what the AI picks, trusted to be legal because the AI mirrors the
  decomp's own gates: the switch gate (`ai/switching._layout`, trainer_ai.c:3906-3913) and its
  living-bench test (:3924-3938); the move scores' usable mask (`TrainerAI_Init`'s
  CHECK_INVALID_ALL, trainer_ai.c:234: `actions.usable_moves` in `ai.commands` / `ai.policy`: no
  move, no PP, Disable, Taunt, the Choice lock, Imprison) with Struggle when none
  (ai/policy.py:2850, :3057); the locked battler's forced move (`actions.locked_slot`); the bag pick
  (`TrainerAI_ShouldUseItem`, `actions.usable_trainer_items`); the replacement pick
  (`ai.replacement`, a `legal_replacements` slot) and the pivot pick at its own list
  (`post_ko_switch_in`). `step` would silently accept an illegal opponent action: a switch to an
  empty, fainted or active slot (`_send_out` sends it; no check), a switch while trapped or locked
  (played as a switch), and a move slot the mask forbids: an empty slot runs move 0, a slot with no
  PP spends to -1 (`spend`), a Choice-locked holder's other move runs; Disable, Taunt and Imprison
  are still caught at their status checks (bcp.c:2543-2583), as for a faster foe's this turn. The
  player's actions are checked by the search (`legal_actions`, `decision_actions`).
- B8-1 Lock-On / Mind Reader (effect 94, effect_script_0094.s: a direct `CHECK_HP | TO_DEFENDER`
  subscript_lock_on.s): a living defender without a substitute gets its MOVE_EFFECT_LOCK_ON count
  set to 2 and lockOnTarget = the attacker (:5-9, `OPCODE_SET`, whatever it had); a substitute fails
  it (:6, :33). `MON_IS_LOCKED_ONTO(attacker, defender)` (include/battle/common.h:51) clears a miss
  beside No Guard (bcp.c:3016-3018), in TryOHKOMove's bypass (battle_script.c:4343-4361) and at the
  Future Sight hit (`outcome._rate`, both engines); `MON_COND_CHECK_STATE_LOCK_ON` drops the count
  at each turn end (bcp.c:1618-1624), so a lock set this turn serves the locker's move of the next.
- B8-2 Lock-On's leave rules: the locked battler's own count and target are volatile, Baton Pass
  keeping both (moves.h:46, battle_lib.c:2109); when the locker leaves by a switch the lock on it
  ends (battle_lib.c:2063-2067), by Baton Pass it restarts at the initial count (:2077-2085,
  moves.h:55); its faint leaves it (CleanupFaintedMon, :2156-2205, has no such loop) until the
  replacement comes in (corrected by B8-7).
- B8-3 Perish Song (effect 114, a direct ON_HIT subscript_perish_song_start.s):
  `BtlCmd_TryPerishSong` (battle_script.c:5465-5487) marks every living battler without the effect
  (count 3), failing with none (Soundproof is an unsupported ability).
  `SIDE_COND_CHECK_STATE_PERISH_SONG` (bcp.c:1791-1820), after Future Sight's: a battler found at 0
  loses the effect and all its HP (subscript_continue_perish_song.s:6-7), else the count drops; the
  pass's re-entries run only `AnyFainted` (:1749), the battle-over check is TurnEnd's (:1845), so
  `_future_sight` does not decide the battle mid-pass (two last Pokemon that faint in the pass: a
  draw, played as a loss). Volatile, kept by Baton Pass (battle_lib.c:2110). Observation:
  `perish_count` (1 + the count while up).
- B8-4 Yawn (effect 187, a direct `CHECK_HP | TO_DEFENDER` subscript_yawn.s): an ignorable Insomnia
  or Vital Spirit, or Leaf Guard in sun, makes it ineffective and is tagged (:5-9, :25-30); a
  substitute, an uproar unless Soundproof, any status, Safeguard or a Yawn counting fail it
  (:14-21); else the count is 2 (`TryYawn`, battle_script.c:6580-6590). `MON_COND_CHECK_STATE_YAWN`
  (bcp.c:1684-1697) drops it and at 0 runs subscript_fall_asleep.s as a move effect (:4-14): the
  battler's own (not ignorable) Insomnia / Vital Spirit / sunny Leaf Guard keep it awake (tagged,
  :_202/_221), a sleeper, an uproar or a status stop it silently, no Safeguard test; else `Random 3,
  2` turns (`Outcome.yawn_sleep`, its own draw and law) and the move lock released. Volatile, not
  kept by Baton Pass.
- B8-5 Curse by a Ghost user (effect 109's Ghost branch, effect_script_0109.s:9-17): its defender is
  the foe (`effects.ghost_user` through `targets_user` / `self_defender` / `defender_always`, both
  engines); subscript_curse_ghost.s fails behind a substitute or on a cursed foe (:6-7, :22-24),
  else curses the foe and costs the user Divide(maxHP, 2) through UPDATE_HP (:9-15; it can faint).
  Curse has no MOVE_FLAG_CAN_PROTECT (res/moves/curse/data.json): Protect never stops it.
  `MON_COND_CHECK_STATE_CURSE` (bcp.c:1451-1459, after Burn): Divide(maxHP, 4) unless Magic Guard
  (subscript_curse_damage.s). Volatile, kept by Baton Pass (condition.h:72-76). The coverage / prep
  refusal (`coverage.ghost_curse`) is removed.
- B8-6 (B8 review) The Future Sight hit's law (`outcome.future_thresholds`) reads Lock-On after
  MON_COND_CHECK_STATE_LOCK_ON drops the count in the per-battler turn-end checks (bcp.c:1618-1624),
  which come before the side-condition pass hits (:1758-1783); the hit reads MON_IS_LOCKED_ONTO
  (:3016-3018): a lock at 1 is gone at the hit. The grid has a lock axis: unlocked always possible
  (also covering a locker's switch, battle_lib.c:2063-2067, and a pivot view that kept a lock the
  switch clears), locked by the stored attacker's slot possible when the count is 2 or more at the
  start or a Lock-On is used this turn. Test: test_batch8.py
  `test_future_sight_law_reads_the_lock_after_the_countdown`, its expectation from step's live
  `future_threshold` on the state at the hit.
- B8-7 (B8 review) Corrects B8-2's text, the code was right: after the locker's faint the lock stays
  until the replacement comes in; its BtlCmd_SwitchAndUpdateMon runs UpdateAfterSwitch
  (battle_script.c:2522), which without Baton Pass clears every lock whose lockOnTarget is that
  battler (battle_lib.c:2063-2067). Test extended through `replace` (both engines).
- B8-8 (B8 review) The opponent-legality invariant checks concrete host-side picks only; the AI
  picks inside jitted playouts (most of them) are traced and unchecked (tests/ai_legality.py says
  so). `test_the_property_reaches_unrefused_states` checks one found state (Ingrain and Mean Look on
  the AI's side); its docstring says so.
- B9-1 Thunder (effect 152) and Blizzard (260) are a probabilistic TO_DEFENDER paralysis / freeze
  (effect_script_0152.s, _0260.s: `SECONDARY_STATUS`); their weather accuracy is by effect id:
  inside NO_CLOUD_NINE, Thunder's hitRate is 50 in sun before the stage scaling (bcp.c:2915-2917),
  and CheckMoveHitOverrides clears the miss of Thunder in rain and Blizzard in hail after No Guard /
  Lock-On (:3023-3031). Carried as `MoveFlag.THUNDER` / `BLIZZARD`, set by build_tables from the
  move's decomp effect (Kaizo's Ice Ball and Sheer Cold have effect 260). Thunder's
  SYSCTL_HIT_DURING_FLY matters only against Fly / Bounce (unsupported). The hit laws take the
  weather variants they already had (the foe's weather move, a switch-in ability).
- B9-2 Fire Fang (273) and Ice Fang (274; Kaizo's Avalanche) are Thunder Fang's family:
  subscript_burn_or_flinch.s / subscript_freeze_or_flinch.s are subscript_paralyze_or_flinch.s with
  BATTLE_SUBSCRIPT_BURN / _FREEZE (`FANG_PARALYZE_FLINCH` with the status in param 0). Fire Fang's
  Wonder Guard bypass (battle_lib.c:7600) waits for Wonder Guard (Owed).
- B9-3 Speed and damage abilities. Chlorophyll doubles speed in sun beside Swift Swim
  (battle_lib.c:1228-1238). Unburden doubles it with canUnburden and no item (:1268-1272);
  `BattleState.can_unburden` is set by RecoverStatusByAbility's Unburden case for a holder with an
  item (:4433-4437: every move end for attacker and defender, every switch-in check for every
  battler, :4135-4145) and by InitBattleMon at the battle start (:160-166), zeroed by
  UpdateAfterSwitch (:2097-2101, Baton Pass too). Huge Power / Pure Power double attackStat before
  the held items (:6706-6708; so x2 then Choice Band's x1.5). Swarm is Blaze's line for Bug
  (:6834-6838). Flower Gift in sun (inside NO_CLOUD_NINE, the self-hit passes no weather): a living
  holder on the attacker's side gives attackStat x15/10, on the defender's side spDefenseStat x15/10
  unless the attacker has Mold Breaker (:6930-6941, COUNT_ALIVE_BATTLERS_OUR_SIDE :2886-2893);
  Cherrim's form change (:6366-6392) writes only formNum, which only the display reads
  (battle_controller.c sprite messages). The end-of-turn speed grid (`enumerate._end_speeds`) takes
  Chlorophyll's sun beside Swift Swim's rain and an Unburden axis.
- B9-4 Recoil: `RECOIL`'s param 0 is the divisor of subscript_recoil_1_{2,3,4}.s (DivideVarByValue =
  BattleSystem_Divide, battle_script.c:4413: at least 1), which a percent could not give for 1/3:
  Double-Edge / Wood Hammer (198, effect_script_0198.s with Reckless) join. Kaizo's changed be_seq
  48 (1/2 recoil, Reckless) and 182 (vanilla 48's script: 1/4 recoil, Reckless) are `modelled` in
  rom_diff.toml through `effect_map.KAIZO_EFFECTS`, which build_tables uses for Kaizo in place of
  the vanilla entry (and checks against the modelled members). Vanilla 182 (Superpower) stays
  unsupported. Left: Volt Tackle's family (262: recoil 1/3 and a paralysis that
  BattleSystem_TriggerSecondaryEffect's PROBABILISTIC branch runs on every hit,
  battle_lib.c:1551-1567) and Jump Kick's crash (45, subscript_crash_on_miss.s: half the damage the
  miss would have dealt, capped at half the defender's max HP, so the miss reads the crit and roll).
- B9-5 Singles-only moves: in singles BattleSystem_Defender gives RANGE_ALLY and RANGE_USER_OR_ALLY
  the attacker (battle_lib.c:1670-1684; `effects.targets_user`, both engines). Follow Me (172) sets
  the user side's followMe / followMeUser (battle_script.c:6261-6270), read only by target selection
  (battle_lib.c:1714-1717), the redirection abilities (:1743) and Counter (battle_script.c:4628),
  which in singles name the target anyway: a message only (`EffectId.FOLLOW_ME`). Helping Hand (176)
  fails outside a double battle (BtlCmd_TryHelpingHand :6309-6311, MOVE_STATUS_FAILED,
  effect_script_0176.s:15-17; `EffectId.HELPING_HAND`). Surf (257) is Gust's case: its x2 needs a
  diving target (Dive is unsupported).
- B9-6 Volt Absorb / Water Absorb join Dry Skin's absorb (BattleSystem_TriggerImmunityAbility,
  battle_lib.c:3500-3512, run at TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES on a move that hit):
  Water Absorb has Dry Skin's test (a Water move with power); Volt Absorb takes any Electric move
  aimed at another battler, Thunder Wave included (no power test); both heal Divide(maxHP, 4) below
  full HP and tag the holder (subscript_ability_restores_hp.s). Rain Dish heals Divide(maxHP, 16)
  below full HP in rain (BtlCmd_EndOfTurnWeatherEffect, battle_script.c:5614-5618) and tags the
  holder (subscript_weather_continues.s:_076). Clear Body blocks any drop by another battler and
  Hyper Cutter an Attack drop, before Keen Eye's accuracy test (BtlCmd_ChangeStatStage, :2884-2902,
  Battler_IgnorableAbility), Intimidate's included (its attacker is the Intimidate holder,
  subscript_intimidate.s:6); those messages are built in C, so the AI learns nothing.
- B9-7 Aftermath (battle_lib.c:4358-4371): the defender is faintedMon (this hit's UPDATE_HP fainted
  it, TryFaintMon battle_script.c:1768-1770), a living attacker without Magic Guard, no Damp on the
  field (unsupported), no MOVE_STATUS_NO_EFFECTS, a contact move, not under SYSCTL_UTURN_ACTIVE
  (U-turn's contact ability runs before that flag, B6): Divide(maxHP, 4) from the attacker; the
  message tags the holder.
- B9-8 Poison Heal (subscript_poison_damage.s:4-12, poison and toxic alike): below full HP it heals
  Divide(maxHP, 8) instead of the damage and tags itself; at full HP nothing (the toxic counter
  still rises, bcp.c:1405-1415). Bad Dreams (MON_COND_CHECK_STATE_BAD_DREAMS, bcp.c:1484-1500, after
  Bind): an asleep battler with HP and no Magic Guard facing a living holder
  (COUNT_ALIVE_BATTLERS_THEIR_SIDE_FLAG, battle_lib.c:2915-2922) loses Divide(maxHP, 8) per holder,
  the message tagging the holder (subscript_bad_dreams.s).
- B9-9 Dragon Rage (41) and SonicBoom (130) are Night Shade's family with a fixed amount (40, 20) in
  param 0 (effect_script_0041.s, _0130.s: SYSCTL_IGNORE_TYPE_CHECKS, BTLVAR_DAMAGE). Their
  BTLVAR_REGULATION_FLAG failure reads battleSys->rulesetMask (battle_script.c:9647-9648), set only
  for a regulated battle (field_battle_data_transfer.c:110, :376-379).
- B9-10 Stat moves. Bulk Up (208) is Dragon Dance's family with Defense for Speed
  (subscript_user_atk_and_def_up_1_stage.s differs at :4, :12 only); since it raises Defense it is a
  clause effect for the foe's later hit. Belly Drum (142, subscript_belly_drum.s): fails at Attack
  +6 or curHP <= Divide(maxHP, 2) (MOVE_STATUS_FAILED, :3-6, :52), else Attack +6 and Divide(maxHP,
  2) through UPDATE_HP. Minimize (108, subscript_minimize.s): sets the user's MOVE_EFFECT_MINIMIZE
  (`BattleState.minimized`, volatile, not Baton Passed, a mask bit Roar's Ingrain test sees) then
  Evasion +1. Stomp (150, effect_script_0150.s): a flinch at its chance with powerMul 20 against a
  minimized defender (`damage.script_power_mul`).
- B9-11 Trapping abilities in `actions.trapped` (Battler_IsTrapped, battle_lib.c:5514-5536, read by
  the party menu, bcp.c:572): a living foe's Shadow Tag unless the battler has it too, a living
  foe's Magnet Pull on a Steel battler, a living foe's Arena Trap on a grounded battler
  (BattlerIsGrounded, :5500-5507: no Levitate, no Magnet Rise, not Flying, or Iron Ball, or Gravity;
  Ingrain does not ground here), by Battler_Ability over the other side's living battlers
  (:5509-5512). Shed Shell (:5518) is unsupported. The AI's own gate already had them (B7-18).
- B9-12 Dream Eater (8) is DRAIN 50 with param 1 = 1: effect_script_0008.s ends with
  MOVE_STATUS_NO_MORE_WORK ("wasn't affected") on an awake defender (:5, :9-16), else
  subscript_dream_eater.s: Divide(hitDamage, 2) then Big Root (as drain). Payback (230, Kaizo's
  Revenge too) doubles movePower once the defender's action is over (BtlCmd_CalcPaybackPower,
  battle_script.c:7198-7209; Battler_MovedThisTurn, battle_lib.c:5365-5368: a switched-in newcomer's
  action is MOVE_END too, :2053), as a powerMul of 20 in `effects.play_hit` (the same product,
  battle_lib.c:6688).
- B9-13 (B9 review) The end-of-turn speed grid (`enumerate._end_speeds`) took the outgoing battler's
  `can_unburden` for an incoming Pokemon. UpdateAfterSwitch zeroes MoveEffectsData
  (battle_lib.c:2101-2104) and the switch-in check arms canUnburden only for an Unburden holder that
  still holds an item (:4433-4437): a Pokemon other than the active one carries `(ability ==
  UNBURDEN) & (item != 0)` (law test `test_unburden_speed_axis_of_a_newcomer`). The generator does
  not give an item holder `can_unburden = 0` (its switch-in check set it).
- B9-14 (B9 review) The reference engine applied Belly Drum on a failing use (Attack +6, or curHP <=
  Divide(maxHP, 2)): subscript_belly_drum.s:3-6 fails before the stage and the HP, as the JAX engine
  had it (`effects.belly_drum_fails`). Reference fixed; found by the new hand test.
- B9-15 (B9 review) `build_tables._kaizo_rom_changes` checks the modelled be_seq members equal
  `effect_map.KAIZO_EFFECTS` exactly against the real rom_diff.toml (an entry with no modelled
  member would override the vanilla mapping of an unchanged script); a test's own small diff keeps
  the subset check. The exhaustive solver (`tests/search/exhaustive.py`) unpacked a fourth value its
  own `_turn` never returns: `test_stage_b.py` failed there. Hand tests: Payback's x2, Belly Drum's
  failures, battle cases for Chlorophyll, Swarm, Flower Gift, Water Absorb, Rain Dish, Aftermath,
  Hyper Cutter vs Intimidate, SonicBoom, the Kaizo recoil divisors, and the trapping abilities
  (`actions.trapped`: Shadow Tag vs Shadow Tag, Arena Trap vs Flying / Levitate / Gravity, Magnet
  Pull on Steel only, a fainted trapper).
- B10-1 (b10 f1) Volt Tackle #262 and Flare Blitz #253 are one effect, `EffectId.RECOIL_STATUS`
  (80): params (status, recoil divisor 3, thaws_user). Scripts (effect_script_0262.s, _0253.s) set
  the status under SYSCTL_APPLY_SECONDARY_EFFECT via the PROBABILISTIC branch
  (battle_lib.c:1551-1567), then subscript_recoil_1_3.s takes Divide(hitDamage, 3) from the user
  unless Rock Head or Magic Guard. Flare Blitz thaws its frozen user (bcp.c:2481 lets it act, :2730
  thaws), as THAW_AND_BURN_HIT. Chance 0 refused by build_tables. Reckless powers both (powerMul
  12, as RECOIL). AI: both -> Expert_RecoilMove (script.s:1796, :1802 -> :6289-6305); #253 added
  to the Kaizo transcribed sets and the dispatch (#262 was E1's batch 7).
- B10-2 (b10 f1) Jump Kick / Hi Jump Kick #45 are `EffectId.CRASH` (81): a plain hit;
  effect_script_0045.s sets SYSCTL_CRASH_DAMAGE before CalcCrit / CalcDamage, so a use that ran its
  script and did not connect (missed or protected) ends in subscript_missed.s, whose
  subscript_crash_on_miss.s (:11-19) takes min(Divide(damage, 2), Divide(defender maxHP, 2)) from
  the user, Magic Guard excepted. Damage: hit 0's crit and roll through the type chart, hit or miss
  (bcp.c:3271-3277); an immune target gives 0. Laws read crit and roll[s, 0] on a miss
  (`Reads.crash`). Reckless powers it. AI: no Expert_Main entry (E1 batch 3 transcribed it
  generically).
- B10-3 (b3-9 merge check) `BattleSystem_InitBattleMon` (bcp.c:227, battle start) arms
  canUnburden for every lead holding an item, any ability (battle_lib.c:161-166):
  `hooks.battle_start` does so before the switch-in states, so a lead that gains Unburden later
  (Trace) keeps it (test_any_item_holder_lead_arms_unburden). Baton Pass keeps Mean Look, Lock-On,
  Perish Song, Curse (battle_lib.c:2073-2115):
  test_baton_pass_keeps_mean_look_lock_on_perish_song_and_curse fails under a JAX-only mutation
  dropping the four from `_BATON_PASSED`. Citations fixed: Sleep Talk's CheckInvalidMoves is
  battle_script.c:4927; subscript_curse_ghost.s is :5-7 (tests), :9-15 (curse and HP), :17
  (message), :22-24 (failure); InitBattleMon's arm is :161-166 and UpdateAfterSwitch's clear
  :2100-2104. tests/ai_legality.py's first line: it checks host-side picks only.
- B10-4 (b10 f3) Substitute (`EffectId.SUBSTITUTE`, 82, decomp 79), both engines. Move: `effects._substitute`,
  `substitute_fails` (subscript_try_substitute.s:7-8, battle_script.c:5211-5226,
  MOVE_STATUS_NO_MORE_WORK). Hit (`play_hit`; bcp.c:3359-3378): the substitute takes min(damage,
  its HP), hitDamage for drain and recoil, shellBellDamageDealt, the flag (`MoveHit.sub_hit`, a
  hit-loop carry; reference `_Turn.sub_hit`); none of False Swipe's cap, Focus Sash, Bide's store,
  damage-taken records, resist berry, SYSCTL_MOVE_HIT. Gates: `_Hit.shielded` (CheckSubstitute:
  bit or flag) for the status subscripts, flinch (King's Rock too), confuse, Bind, Leech Seed,
  Swagger (whole move fails); `_Hit.sub_hit` (SubstituteWasHit) for rolled foe secondaries, Pluck
  (lib.c:5546), the contact ability (:4209), the Sticky Barb transfer (:5378); the bit only for a
  foe stat drop (script.c:2921-2923), Dream Eater (effect_script_0008.s:5), SmellingSalt's x2 and
  cure (effect_script_0171.s:5), Intimidate (subscript_intimidate.s:13), Toxic Spikes on a Baton
  Pass newcomer (subscript_poison.s:29, _badly_poison.s:46). Special paths: Future Sight's hit
  (subscript_future_sight_damage.s:13-24), Pursuit's pre-switch hit (subscript_pursuit.s:20,
  :56-73: only a break changes the substitute; `play_hit(pursuit_switch=True)`). Laws: SUBSTITUTE
  input-neutral, non-table (SmellingSalt is DOUBLE_POWER, never compressed). Observation: presence
  already public, HP hidden. AI (Basic :633-636, Expert_Substitute :2976-3042): E1's. Hand tests
  in tests/engine/test_batch10.py (move, hit and break, status / drop / drain, multi-hit with
  Static, King's Rock, Pursuit, Ghost Curse); generator `strategies._draw_substitute`. Also fixed:
  RECOIL_STATUS / CRASH classified in test_accuracy_classes, Flare Blitz's thaw flagged in
  test_effect_map's thaw test, batch 7's four observation cases added to
  test_observe.VISIBLE_CLASSES (a slow test).
- B10-5 (b10 f5) Endeavor (`EffectId.ENDEAVOR`, 83, decomp 189) and Pain Split
  (`EffectId.PAIN_SPLIT`, 84, decomp 91), both engines. Endeavor (effect_script_0189.s): fails
  (MOVE_STATUS_FAILED, :7, :13-14; `step._use_move`'s `failed`, reference `_script_fails`) unless
  the defender's curHP exceeds the attacker's; else battleCtx->damage is the difference under
  SYSCTL_IGNORE_TYPE_CHECKS (:10: no multiplier, an immunity still gives 0; no CalcCrit /
  CalcDamage: `UNROLLED_EFFECTS`, `damage.fixed_damage`; the resist berry skips it,
  `IGNORES_TYPE_CHECKS`); a substitute takes it like any hit (B10-4). Pain Split
  (subscript_pain_split.s): fails behind a substitute (:5, :50-52: `primary_effect_fails`); else
  avg = (a + d) / 2 (OPCODE_DIV), each battler's HP set to it through UPDATE_HP, capped at max HP
  (`effects._pain_split`). Both input-neutral, non-table (HP only). AI Expert_Endeavor /
  Expert_PainSplit: E1's. Hand tests in test_batch10.py; generator draws both in `_draw_batch10`
  with a low user.
- B10-6 (b10 f6) Knock Off (`EffectId.KNOCK_OFF`, 85, decomp 188), both engines, per "Family 6
  design". `effects._knock_off` (reference's KNOCK_OFF branch): a table hit, then
  subscript_knock_off.s's tests (substitute, Multitype, Griseous Orb, quickClaw, custapBerry) and
  BtlCmd_TryKnockOff (Sticky Hold, decomp ability 60, ignorable; else the item goes and the slot's
  bit is set). New state: `knocked_off` (per party slot) and turn scratch `ts_claw`
  (`order.claw_flags`: quickClaw flag from the FIGHT-vs-FIGHT order, cleared at its holder's move
  start; no persistence needed: only two FIGHT commands set it and the holder then reaches its move
  start or faints). The Sticky Barb transfer skips a knocked-off attacker and Knock Off itself
  (battle_lib.c:5390-5391; pivot :7380). Laws: a table-damage effect plus an input clause (the
  foe's later hit when it holds an item). AI Knock Off: E1's (branch claude/e1-prep-on-e2). AI
  record (E2's, end-to-end rule): TryKnockOff runs no CopyToParty, so `ai_item_known` keeps the
  knocked item until the battler's next CopyToParty (battle_lib.c:1819-1823): its next HP-bar
  update (battle_script.c:1720) or move attempt (bcp.c:2748). Modelled per segment (an action, the
  turn end) in both engines (`step._stale_known_item`, reference twin): exact, as a knocked-off
  slot never holds an item again and the AI reads the record only at its next choice;
  test_knock_off_leaves_the_ai_record_until_the_next_copy_to_party. Thief, Covet, Trick and Fling,
  which also read the mask, are unsupported. STATE.md maps quickClaw (`ts_claw`) and
  knockedOffItemsMask (`knocked_off`); test_state's audit list no longer pins quickClaw as later.
- B10-7 (b10 f8) Safeguard (`EffectId.SAFEGUARD`, 86, decomp 124), both engines.
  BtlCmd_TrySafeguard (battle_script.c:5754-5772): fails with the user's side's Safeguard up
  (MOVE_STATUS_FAILED, `primary_effect_fails`), else safeguardTurns = 5 (existing field-condition
  count-down, bcp.c:1016-1030). Its tests: every status subscript after the substitute and status
  tests (paralyze :28, burn :45, freeze :25, poison :41, badly_poison :55, fall_asleep :51, after
  its uproar test), subscript_confuse.s:26 (Swagger's confusion too, after its Attack raise), the
  held-item path (Flame Orb subscript_burn.s:16, Toxic Orb subscript_badly_poison.s:27) and Toxic
  Spikes on a newcomer (poison :41, badly_poison :55); Yawn's test already modelled (B8-3); the
  drowsy sleep (MOVE_EFFECT type) has none. AI Basic_CheckAlreadyUnderSafeguard: E1's. B10-4
  follow-up: Uproar's "kept awake" NO_MORE_WORK for a direct sleep move excludes a substituted
  target, whose CheckSubstitute (subscript_fall_asleep.s:40) fails it first (both engines). Law
  fix (B10-4): `Reads.flinch_sure` let the first mover's flinch roll drop the foe's hit reads even
  with the foe behind a substitute, where the flinch never applies (subscript_flinch_mon.s:6;
  found by test_damage_classes: 1-HP substitute, Trick Room, uproar); it now needs the foe's
  `substitute_hp == 0`; test_laws_no_sure_flinch_through_a_substitute fails without that clause.
- B10-8 (b10 f10) Focus Punch (`EffectId.FOCUS_PUNCH`, 87, decomp 170), both engines.
  effect_script_0170.s:5-6: a user whose turnFlags physical / specialDamageAttackerMask is set (a
  hit on its HP this turn, bcp.c:3416-3424; engine `phys_from` / `spec_from`, `foe_taken[3:5]`)
  gets MOVE_STATUS_LOST_FOCUS, a DID_NOT_HIT status: subscript_missed.s:8-10 prints "lost its
  focus" and the move ends like a failed script (`step`'s `failed`, reference lost_focus beside
  `_script_fails`); else a plain table hit (priority -3 is move data). A hit on its substitute sets
  no mask (B10-4). The turn-start "tightening its focus" message (bcp.c:811-834,
  PRE_MOVE_ACTION_STATE_TIGHTEN_FOCUS) changes no state; observation shows the move and the miss
  (coarser: "lost its focus" and a miss share a set). AI Expert_FocusPunch: E1's. Encore and Assist
  (rest of f10) not done.
- B10-9 (b10 f10) Encore (`EffectId.ENCORE`, 88, decomp 90), both engines; its AI
  (Basic_CheckCannotEncore script.s:655-657, Expert_Encore :3177-3240) still to transcribe (AI
  refuses the move until then). New volatile state `encore_move`, `encore_slot`, `encore_turns`
  (MoveEffectsData, cleared on a switch, not Baton-passed). BtlCmd_TryEncore
  (battle_script.c:4725-4748; `effects.encore_slot_of`, `_encore`): fails (MOVE_STATUS_FAILED) for
  an encored target, no last move, a last move not in its slots, one of sCannotEncoreMoves' battle
  effects (battle_lib.c:7212-7231, `ids.CANNOT_ENCORE_MOVES` via `move_ai_effect`) or no PP; else
  the move, its slot and RandNext() % 5 + 3 turns (`DURATION_ENCORE`, 3-7). Selection
  (`actions.usable_moves`): CheckInvalidMoves leaves only the encored move
  (battle_lib.c:2302-2304); with it valid the encored slot is the one FIGHT action (bcp.c:443-446),
  else Struggle. Action (FightCommand, bcp.c:1866-1883): after Struggle, the encored move runs
  whatever was chosen (a faster Encore this turn takes over), from its slot's PP (no PP:
  MOVE_STATUS_NO_PP, now in the JAX engine too); a slot no longer holding it ends the encore. Turn
  end (`step._encore_check`, bcp.c:1591-1612, between Disable and Lock-On): unknown or 0-PP move
  zeroes the count; the count drops, at 0 the encore ends. Observation shows an encore is on
  (count hidden). Owed (after E1's ALLIN merge lands in main): Encore's AI; Basic_CheckCannotEncore
  and Expert_Encore need AICmd_IfBattlerUnderEffect (CHECK_ENCORE, CHECK_DISABLE), not on this
  base, and Expert_Encore's IfRandomLessThan 30 (script.s:3192) goes through ALLIN's
  commands._slot_read with a read_bounds.READ_BOUNDS entry; until then coverage refuses Encore for
  the trainer AI. Differential fixes: the reference's volatile reset list lacked the three encore
  fields; its `_usable` ignored Encore (Struggle and the forced slot). Sleep Talk's callable slots
  drop every move but the encored one (CheckInvalidMoves at battle_script.c:4927;
  `order.sleep_talk_slots`, both engines). Manager ruling: batch 10 ships Encore refused by
  coverage for both sides (`coverage.IMPLEMENTED_EFFECTS[ENCORE] = NONE`, reason `ENCORE_REFUSAL`
  naming the takeover task; Metronome refused with it, as it can call Encore), engine code kept;
  batch 11 builds the takeover, then lifts the refusal and adds Encore's AI. Owed: the same-turn
  takeover (FightCommand, bcp.c:1869-1873) is exact in both engines but not in `outcome_laws`
  (reads planned from the chosen move: a hand case gives 1 child against 33 sampled outcomes) nor
  in `sample_outcome` (crit and secondary drawn with the chosen move's rates). Proposed: the
  encored move as a second candidate move for that side, reads unioned, crit / secondary drawn per
  candidate.
- B10-10 (law) `Reads.flinch_sure` counted Snore (FLINCH with SNORE_PARAM), whose flinch fails with
  its move unless the user is asleep when it acts (effect_script_0092.s:5): an awake Snore user
  moving first let the laws drop the foe's reads (found by
  test_matches_sampled_frequencies_random, Snore vs Future Sight). Snore is left out of
  `flinch_sure` (its foe's reads stay planned).
- B10-11 (b10 f10) Assist (`EffectId.ASSIST`, 89, decomp 180), an invoker (`order.CALL_ASSIST`),
  both engines. BtlCmd_TryAssist (battle_script.c:6405-6444; `order.assist_candidates`): in party
  order, every move of every other party slot holding a Pokemon (a fainted mate's too), not
  Move_IsInvoker (`ids.INVOKER_MOVES`, battle_lib.c:3032-3045) and Metronome-able (NO_METRONOME;
  Gravity and Heal Block owed with those moves); duplicates count; the call is RandNext() %
  numMoves (`sample_outcome`: uniform draw over the list, fold_in 34 + side; `call_laws`: each
  position weighs 1, grouped by called move in `call_variants`); an empty list fails it
  (MOVE_STATUS_FAILED). Then GoToMoveScript like the other invokers (step's generic invoker path;
  RANGE_SINGLE_TARGET_SPECIAL: its own defender is the user). AI entry transcribed generically by
  E1 (batch 3).
- B10-12 (b10 f7) The charge turn. Each charge move keeps its attack turn's EffectId (Razor Wind /
  Skull Bash / SolarBeam / Fly / Dig / Dive / Shadow Force PLAIN, Sky Attack FLINCH, Bounce
  SECONDARY_STATUS paralysis); `MoveFlag.CHARGE` and kin (from the decomp effect:
  `effect_map.CHARGE_FLAG_EFFECTS`, `move_flags` now int32) carry the charge. The scripts
  (effect_script_0039/0075/0145/0151/0155/0255/0256/0263/0272.s) attack when the user is
  VOLATILE_CONDITION_MOVE_LOCKED (line 5: second turn, `locked_move`; PP spent once, on the first:
  bcp.c:335), SolarBeam in sun without Cloud Nine / Air Lock (_0151.s:5-6, CheckIgnoreWeather
  battle_script.c:9195-9206) or with a Power Herb (HOLD_EFFECT_CHARGE_SKIP, `ItemId.POWER_HERB`,
  removed by subscript_item_skip_charge_turn.s:20; Skull Bash's Def +1 first,
  subscript_power_herb_skull_bash.s:14-17). Else they charge: SYSCTL_FIRST_OF_MULTI_TURN (no
  accuracy roll bcp.c:2905-2908, no Protect / Lock-On / semi test :3003-3005, no type chart :2354,
  no immunity abilities battle_lib.c:3509-3540), the direct subscript_vanish_on_charge_turn.s
  locks the user into moveCur (:10, Battler_LockMoveChoice lib.c:1828-1832; a called move's lock
  is the called move), the semi-invulnerable bit (new state `semi_invulnerable`, `ids.SemiKind`:
  AIRBORNE Fly/Bounce, UNDERGROUND Dig, UNDERWATER Dive, SHADOW_FORCE), Skull Bash's Def +1
  (_0145.s:8, chance 100). A charge with no living foe keeps the user as defender and charges
  (HasNoTarget bcp.c:2327-2334); a called charge move with none goes to subscript_no_target from
  GoToMoveScript (battle_script.c:2124-2126); a herb / sun attack with none ends in TryMove's
  HasNoTarget (LAST_OF_MULTI_TURN, common.h:57-60) after the herb went. The second turn's script
  clears the bit and the lock before TryMove (line 24 / cleanup :6). Interruptions: every
  disruption, miss, Protect and the sleeper unlock at turn setup run Battler_UnlockMoveChoice,
  which also clears the bit (lib.c:1838): move lost, next use charges again (flinch, full
  paralysis, confusion self-hit, Truant, sleep); a switch or faint resets it (volatile). Choice
  lock: FIRST_OF_MULTI_TURN updates the locks as the attack message does (bcp.c:3921-3922,
  :4490-4491). The charge turn shows no attack message (moveHit, movePrev, moveProtect MOVE_NONE)
  except Skull Bash, whose indirect Def +1 runs subscript_update_stat_stage.s's PrintAttackMessage
  (:6-7; battle_script.c:1431-1432). Razor Wind / Sky Attack crit +1 (CRITICAL_BOOSTS,
  `MoveFlag.CRIT_UP`, `outcome.crit_odds`). Shadow Force's ON_HIT subscript_feint.s only clears
  the defender's turnFlags.protecting (:20), unread afterwards in singles. Kaizo: be_seq 39-272
  unchanged (rom_diff.toml); Kaizo gives the charge effects to Razor Wind, SolarBeam, Fly and
  Shadow Force only (its Dive and Bounce are plain / paralysis hits); Kaizo Sky Uppercut is effect
  149, Twister 261.
- B10-13 The semi-invulnerable target (CheckMoveHitOverrides, bcp.c:3016-3040): after Protect and
  the Lock-On / No Guard return (either battler's No Guard; OHKO's BYPASSED_ACCURACY,
  battle_script.c:4343-4374, the same battlers), a move whose defender is the foe (a special
  Counter's and Bide's unleash included), not RANGE_OPPONENT_SIDE, misses
  (MOVE_STATUS_SEMI_INVULNERABLE, accuracy 0 too) unless its script sets the kind's
  SYSCTL_HIT_DURING_* (`MoveFlag.HIT_*`, `effect_map.SEMI_HIT_EFFECTS`: Fly 146/149/152/207, Dig
  126/147, Dive 257/261, all 148/168/176). A miss crashes (Jump Kick) and shows before Sturdy
  (subscript_missed.s:13). Exception, Yawn: a direct MOVE_SIDE_EFFECT_CHECK_HP effect
  (effect_script_0187.s:5), whose subscript BattleSystem_TriggerPrimaryEffect still runs on
  MOVE_STATUS_GENERAL_MISS for a living defender (battle_lib.c:1500-1503; moves.h:88);
  subscript_yawn.s:5-9 tests Insomnia / Vital Spirit / Leaf Guard in sun (ignorable) before its own
  miss test (:21), so against a semi-invulnerable holder the ability is printed with
  TAG_NICKNAME_ABILITY (:31-37 -> BattleAI_SetAbility, battle_script.c:11798-11806) and
  `ai_ability_known` set; no drowsiness either way. A protected target is not semi-missed
  (Protect returns first, bcp.c:3007-3013), so no reveal. x2 (powerMul 20, `MoveFlag.DOUBLE_SEMI`,
  `damage.script_power_mul`) for Gust / Twister vs AIRBORNE, Earthquake / Magnitude vs UNDERGROUND,
  Surf / Whirlpool vs UNDERWATER (line 7-8 of each). MOVE_EFFECT_NO_WEATHER_DAMAGE (UNDERGROUND,
  UNDERWATER: moves.h:50) skips the sandstorm, sun and hail turn-end branches (battle_script.c:5580,
  :5587, :5600), not rain. SolarBeam (by move id) is halved under FIELD_CONDITION_SOLAR_DOWN (rain,
  sand, hail, fog; lib.c:7058-7060). Twister (146) and Sky Uppercut (207) now supported. Damage
  classes: a foe's charge move changes a DOUBLE_SEMI hit's power, Skull Bash's charge the Defense.
  AI: Basic :158/:219, Expert_ChargeTurnNoInvuln (:3933-3973), _WithInvuln (:3992-4012),
  Expert_ShadowForce (:4014-4066, 80 roll at :4058); SolarBeam's Expert_Thunder entry (:1719) is
  dead (first jump wins). Not modelled (no supported mover): Gravity's grounding
  (subscript_gravity_start.s:16; Gravity refused by coverage), Transform's semi test, Hurricane
  (Kaizo 141, not reached).
- B10-14 (b10 review) Future Sight / Doom Desire's delayed hit vs a semi-invulnerable defender.
  subscript_future_sight_damage.s:8 CheckMoveHit (battle_script.c:6115-6131) runs
  CheckMoveHitOverrides (bcp.c:216-220): after Protect (:3005-3013) and the Lock-On / No Guard
  return (:3016-3021), :3034-3040 sets MOVE_STATUS_SEMI_INVULNERABLE on a defender with a semi bit.
  Every SYSCTL_HIT_DURING_* bit is cleared by BattleContext_Init at the last MoveEnd
  (battle_lib.c:2009, system_control.h:31-40; bcp.c:3998) and BYPASSED_ACCURACY with
  moveStatusFlags (:1983), so any semi kind is missed ("But it failed!"); the bit lasts to the
  holder's next action. The range test reads CURRENT_MOVE_DATA (common.h:24), i.e. moveCur, never
  reset: FightCommand's moveTemp (bcp.c:1898) or GoToMoveScript's called move
  (battle_script.c:2118). After a turn whose last such move is Spikes, Toxic Spikes or Stealth Rock
  (RANGE_OPPONENT_SIDE) the test is skipped and the hit lands. A turn with no FIGHT keeps an older
  moveCur, but then no battler is semi-invulnerable at turn end (its holder is locked into a
  FIGHT), so engines read only this turn's (`step._turn_move_cur`, event `move_cur`; reference
  `_Turn.move_cur`). The flag lasts the pass like the others: a second hit after a semi miss fails
  even under No Guard (which clears only MISSED).
- B10-15 (b10 review; corrects B10-13's OHKO clause) OHKO move vs a semi-invulnerable target.
  BtlCmd_TryOHKOMove sets SYSCTL_NONSTANDARD_ACC_CHECK first (battle_script.c:4338), so the
  Lock-On / No Guard return of CheckMoveHitOverrides is skipped (bcp.c:3016); the semi test is
  skipped only by MOVE_STATUS_BYPASSED_ACCURACY (:3035), which TryOHKOMove sets on the non-Sturdy
  branch when the defender has MOVE_EFFECT_LOCK_ON (any target) or either battler has No Guard
  (:4342-4372). Against an ignorable Sturdy (:4340-4341) No Guard / Lock-On do not help: the semi
  target is missed, and subscript_missed.s:11-12 shows that miss before the Sturdy message
  (:67-69), so the AI does not learn Sturdy (ai_ability_known).
- B10-16 (b10 review) effects.py's module note: Safeguard supported (B10-7); only Mist's branches
  stay unmodelled (no supported mover).
- B10-17 (b10 review) The pause scratch's `ts_claw` / `ts_custap` keep the flag of each battler
  whose action is still after the paused one in the order (`to_come`), a fact about the named
  battler (doubles design section 5 rule 5), not "the second actor at position 0"; same values in
  singles.
- B11-1 (b11) A chosen move replaced mid-turn: Encore's takeover (ledger "Design: a chosen move
  replaced mid-turn", as designed). BattleControllerPlayer_FightCommand (bcp.c:1866-1873) runs an
  encored battler's encored move, after Struggle, whatever it chose, so a faster foe's Encore this
  turn (BtlCmd_TryEncore, battle_script.c:4725-4748) replaces the chosen move. One event per
  replaced move: `step`'s `encored_at_move` (the encored move replaced the chosen one when the
  action started) with the turn-start test in `enumerate.taken_events`, beside `disabled_at_move` /
  `sealed_at_move` (`_split_events`). Laws: `order.takeover_possible` (the side uses a move without
  Struggling, is not encored at the start, `encore_slot_of` would not fail it, its last move
  differs from the chosen one, and its foe's move can run Encore: `order.can_call_encore`, Encore
  itself or Sleep Talk / Metronome / Assist / Copycat whose call can be Encore) gives the side a
  takeover variant: `call_variants` returns `Variant(call, taken, num, den)`, tag `taken` a field
  beside `call`; `outcome_laws(..., taken)` reads every move-dependent law of that side (move run,
  effect, crit odds, damage classes, thresholds, call law) from the encored move with
  `order.encored_state`, the order staying the chosen move's (CalcTurnOrder fixes it at turn
  start). `call_consistent(..., taken)` keeps a row only where the event matches the tag (for a
  side that can be taken), so each side's variants sum to 1 over the kept rows; `enumerate_turn`
  and `search.expand` carry the tag per row. Outcome shape unchanged. Sampler: outcome A as before;
  B redraws each side's move-dependent fields (`outcome.MOVE_FIELDS`: hit, crit, secondary,
  fang_flinch, duration, future_hit, call) for each alternative (the Disable-shrunk and
  Imprison-sealed Sleep Talk sets, `order.disabled_state` / `sealed_state`, and the encored move,
  `order.encored_state`); the turn's events run on A (`step.turn_events`, only where
  `outcome.replacement_possible`) and a side whose event fired takes B's fields, drawn once for the
  alternative that fired (the invokers' draws read none of the set changes). Cost: under `vmap` the
  `lax.cond` becomes a select, so the event pass runs on every row; playouts pay a second action
  phase per sampled turn, owed to the parked performance work. Exact: the event is fixed before that
  side's action starts, and the redrawn fields are read only after it. Singles: at most one
  replacement per side per turn (one foe move); doubles keeps a short list per battler keyed by its
  event (doubles design section 5).
- B11-2 (b11) The Sleep Talk + Disable / Imprison sampler gap (B5-13, B7-13): the sampler drew a
  split Sleep Talk's fields from the turn-start call; B11-1's pass now draws them for the move
  called from the set the turn leaves. Frequency tests (Ember 100 / Thunder 70, a faster Disable 80
  or Imprison) failed before and pass after.
- B11-3 (b11) Encore into the invokers. Move_CanBeEncored (battle_lib.c:7221-7231) refuses only
  Transform, Mimic, Sketch, Mirror Move, Encore and Struggle (ids.CANNOT_ENCORE_MOVES), so
  Metronome, Assist, Sleep Talk and Copycat can be encored; the takeover variant's call law is the
  encored invoker's, read with the encored state. An encored Sleep Talk always fails:
  BattleSystem_CheckInvalidMoves (battle_lib.c:2302-2304) flags every move but the encored one and
  TrySleepTalk (battle_script.c:4916-4930) flags the invoker itself, so no slot is left
  (STRUGGLING_ALL, :4928). An encored Copycat taking over after a faster Encore copies that Encore
  (movePrev, the foe's moveTemp), which then fails on the Encore user (its last move is Encore,
  refused). Both engines already ran these; the laws and the sampler now follow.
- B11-4 (b11) Encore's AI (ALLIN convention). Basic_CheckCannotEncore (script.s:655-658, entry
  :188): player already encored, -8; Expert_Encore (script.s:3177-3200, entry :1689): player
  disabled -> the +3 roll; else slower -2; else the player's last move's effect not in
  Expert_Encore_EncouragedMoveEffects (:3202-3285, `commands.ENCORE_ENCOURAGED_EFFECTS`, 82
  entries) -2; else +3 unless IfRandomLessThan 30 (:3192, `commands._slot_read`, bound in
  `read_bounds.READ_BOUNDS` row 90) jumps. IfBattlerUnderEffect (trainer_ai.c:1823-1848) reads
  `disabledTurns` / `encoredTurns`, not the move (`commands.if_battler_under_effect`). Effect 90
  joins `outcome.KAIZO_TRANSCRIBED` / `KAIZO_EXPERT_LABELS` / `KAIZO_EXPERT_TRANSCRIBED`; the
  coverage refusals of Encore (both sides) and of Metronome (which can call it) are lifted.
- B11-5 (b11, manager review) The generator (tests/strategies.py) met the batch 10 moves too
  rarely: `_draw_batch10` now runs one draw in two, a charge user awake or thawed 7 in 10, a
  status-move user untaunted 7 in 10 (invokers too, `_draw_batch5`), the side's own encore on
  another move cleared 7 in 10; `legal_action_pair` uses a known `RARE_EFFECTS` move or charge
  move one draw in two.
- B11-6 (b11 review) tests/engine/test_coverage_exp.py: the `endeavor_unsupported` fixture (a
  faked unsupported effect) is gone; the level-up refusal tests stand on a real unsupported
  learnset move, Budew's Worry Seed at L16 (vanilla), and the narrowed
  `test_the_exp_sensitive_moves_are_unsupported` is replaced by
  `test_every_exp_sensitive_move_meets_its_exp_line`: every Â§4.2 / Â§4.4 / Â§4.5 move, in both games
  and whatever its support, meets its family's `exp_refusals` line.
- B11-7 (b11 review; doubles design section 5) `step._stale_known_item` loops over the battlers
  (`forget_item` keeps only the player's record); `order.claw_flags` sets the flag of a FIGHT
  battler compared with any other FIGHT battler (CalcTurnOrder's pairwise sort, bcp.c:766-778);
  the RANGE_OPPONENT_SIDE tests of the semi-invulnerable miss, the Future Sight hit and the
  defender choice go through one helper, `effects.targets_foe_side` (rule 6).
- B11-8 (b11 review) `search.expand`: the opponent's party-list pick is made once, host-side (the
  laws read it); the row program steps those actions and reads only the pick's refusal
  (`playout.turn_inputs`); dropping the pick fails
  tests/search/test_pivot_search.py::test_the_opponents_list_is_resolved_by_its_ai.
- B11-9 (b11, found by the takeover law test) The laws never drew Encore's own duration
  (`OutcomeLaws.relevant.duration` lacked `EffectId.ENCORE`, though `duration_kind` had
  DURATION_ENCORE since B10-9): every enumerated Encore set the empty outcome's count. Fixed;
  unseen before because coverage refused Encore.
- B11-10 (b11, second review) No generated Pokemon knows a move twice, at the root: every move the
  generator writes goes through `strategies._teach` (the given slot, or the slot already holding
  the move), used by every give-style helper and the other slot writes (species item, Pursuit,
  pivot, invoker pair, Uproar, Future Sight, batch 7 and Imprison's shared move, the law moves);
  Uproar's Copycat swap is skipped when Copycat is known, and the outcome-laws strategy's
  invoker-to-Tackle swap leaves an empty slot when Tackle is known. A double `give("Yawn", ...)` is
  one call. The new property test asserts no duplicates on every generated state.
  tests/engine/test_batch11.py::test_the_replacement_predicates_cover_every_event (300 states, any
  outcome): a side's replacement event (`taken_events`, or a Sleep Talk side's `_split_events`)
  never fires where `order.takeover_possible` (taken) or `outcome.replacement_possible` (any) is
  false, read on the `pivot_view` as the laws and the sampler read them; the test steers three
  draws in four (`_steer`: a Speed 999 foe using Encore, Disable or Imprison, no Trick Room). It
  fails with `takeover_possible` forced false or `replacement_possible` reduced to the takeover.
  Note: `_split_events` also fires for a side not using Sleep Talk (a faster Disable on any move);
  the sampler then redraws that side's fields for the same move (`disabled_state` does not change a
  plain move), same law, so harmless.
- B11-11 (b11, second review; doubles rule 1) `order.takeover_possible` and
  `outcome.replacement_possible` take `side` and `foe` (the battler whose Encore, Disable or
  Imprison can land first) and return one `bool`, as `can_call_encore` takes `side`; neither
  computes `1 - side` or `uses_move[::-1]` any more. Callers (`enumerate.call_consistent`,
  `enumerate.call_laws`, `outcome.sample_outcome`) name the foe, stacking over the two sides.
- B11-12 (b11, second review; Kaizo check) The Kaizo ROM's Expert_Encore and its encourage table
  are vanilla. Read with tools/rom (`RomFiles.overlay(14)` of both ROMs, `SymbolMap` of the decomp
  map): gTrainerAITable starts at overlay14 0x022248A4 (sized 0 in the map; compared up to the
  next sized symbol, 0x0222EE60, 10607 words). One word differs, 0x022249CC (0x1A -> 0x57:
  rom_diff.md's Levitate -> Dry Skin compare, script.s:78); Expert_Encore (script.s:3177-3200,
  incl. IfRandomLessThan 30 at :3192) is byte-equal. Expert_Encore_EncouragedMoveEffects
  (script.s:3202-3285) is found once in each ROM, at 0x022292D8, as
  `commands.ENCORE_ENCOURAGED_EFFECTS` (82 entries) then TABLE_END.
- B12-1 Calm Mind (211) and Cosmic Power / Defend Order (206) run DRAGON_DANCE-style two-stat
  raises: the first +1, then the second (subscript_user_*_up_1_stage.s:5-6, :13-16).
- B12-2 Flatter is subscript_flatter.s: Swagger's script with Sp. Atk +1 (:5-16).
- B12-3 Tickle (subscript_target_atk_and_def_down_1_stage.s): NO_MORE_WORK only with both stats
  at -6 (:5-6, :24-33); FAILED on a miss (:9).
- B12-4 Focus Energy: +2 crit stages (battle_lib.c:7123), Baton Pass keeps it (condition.h:72),
  fails when already set (effect_script_0047.s:5).
- B12-5 Haze resets every battler's stages and Focus Energy (battle_script.c:3534-3549).
- B12-6 Psych Up copies the stages and ORs Focus Energy (battle_script.c:6042-6053); the copy runs
  before TryMove, so a semi-invulnerable miss still copies (bcp.c:3211-3213, :3034-3040).
- B12-7 Power Swap / Guard Swap need a hit (indirect ON_HIT); no failure check.
- B12-8 Captivate (effect_script_0265.s:5-9): Oblivious, then genderless, then same gender, each
  NO_MORE_WORK; Oblivious is tagged even on a miss or Protect (:48-53, before TryMove).
- B12-9 Acupressure draws RandNext() % validStats over the stats below +6 in BATTLE_STAT order
  (Atk, Def, Spe, SpA, SpD, Acc, Eva) (battle_script.c:8943-8950); fails behind the user's own
  substitute (effect_script_0226.s:5).
- B12-10 Power Trick XORs the bit and swaps (subscript_user_swap_atk_and_def.s:5-9); Baton Pass
  keeps the bit (moves.h:47); a level-up reload undoes the swap but keeps the bit
  (battle_lib.c:169-174; battle_script.c:9860-9865, :10022-10027, :10173-10177): field
  `power_trick_desync`, swap = bit XOR desync, cleared on any switch (battle_lib.c:2137-2141).
  Shaymin / Giratina form reloads (battle_lib.c:6439) would undo it too; no in-battle form change
  modelled yet.
- B12-11 Stat-family AI: Basic labels script.s:151/162/217/230/246/247/260/268/272/273/283;
  Expert :1648/1713/1727/1751/1752/1769/1781/1786/1787/1804.
- B12-12 Rest (subscript_rest.s): fails on its own Insomnia / Vital Spirit (tagged, :7-8), asleep
  (:9), an uproar unless Soundproof (:10-11), full HP (:14-15); else sleep count 3 (:32), full heal.
- B12-13 Morning Sun / Synthesis / Moonlight: maxHP/2 with no weather or Cloud Nine / Air Lock,
  Divide(maxHP*20, 30) in sun, else Divide(maxHP, 4) (battle_script.c:5966-5979; common.h:65-70).
- B12-14 Wish fails if the user's battler already has wishTurns, else sets 2
  (battle_script.c:6373-6385); heals Divide(maxHP of whoever is then on the field, 2), nothing at
  full HP (bcp.c:1085-1111; subscript_wish_heal.s).
- B12-15 Heal Bell spares a Soundproof user (battle_script.c:5038-5043) and a Soundproof party
  member unless the user has Mold Breaker (battle_io_command.c:645-653); Aromatherapy spares none.
- B12-16 Refresh cures poison / toxic / burn / paralysis (FACADE_BOOST), else FAILED
  (subscript_recover_psn_prz_brn.s).
- B12-17 Psycho Shift (battle_script.c:7542-7553; subscript_give_target_own_status.s:5-30): fails
  on a statused or substituted target or a status-free user; passes the status as a direct effect,
  cures the user only if the target ends up statused.
- B12-18 Healing AI: Expert_Rest / HealBell / Synthesis / Refresh / PsychoShift at script.s
  :1655/:1696/:1707/:1745/:1777; Wish (179) has no Basic or Expert entry. New commands:
  IfLoadedLessThan (trainer_ai.c:762), DiffStatStages (:2259), IfPartyMemberStatus (:1348).
- B12-19 Crit odds are read at the hit: BattleSystem_CalcCriticalMulti reads item, species,
  statusVolatile and ability then (battle_lib.c:7105-7123) and rolls RandNext() %
  sCriticalStageRates[stage] (:7132); Haze clears Focus Energy mid-turn (battle_script.c:3544).
  Outcome.crit holds v, crit iff (v-1) % odds_at_hit == 0; one draw in [0, 48) when the odds can
  move in the turn (outcome.CRIT_ODDS_EFFECTS), else the old draw.
- B12-20 Acupressure (RANGE_USER_OR_ALLY) acts on the resolved defender: the chosen target in
  doubles, the attacker in singles (battle_lib.c:1676-1684); effects.defender_side.
- B12-21 A single-target move whose target fainted earlier in the turn never loads its effect
  script: the defender resolves to BATTLER_NONE (battle_lib.c:1612-1631, :1713-1727, :4194) and
  BEFORE_MOVE_STATE_CHECK_TARGET_EXISTS (bcp.c:3189-3191) returns before the load at :3211.
- B12-22 Psycho Shift onto a Synchronize target: the user is cured at
  subscript_give_target_own_status.s:28-30, then Synchronize (AfterMoveEffects, bcp.c:3665-3669;
  battle_lib.c:4498-4522) passes the poison / burn / paralysis back. Owed with Synchronize (E1).
- B12-23 Mist: TryMist fails with SIDE_CONDITION_MIST up, else 5 turns on the user's side
  (battle_script.c:4300-4317; battle.h:142); counted down in the field pass (bcp.c:993-1004).
- B12-24 Mist blocks any drop on the battler by another battler while mistTurns is set, first,
  before ability / -6 / Shield Dust / substitute tests; Mold Breaker does not pierce it; direct
  drops message, indirect skip silently (battle_script.c:2874-2882, :2937-2945).
- B12-25 Exempt: the user's own drop (battle_script.c:2876); CANNOT_PREVENT is never set (only
  :2875 reads it). Intimidate is blocked (subscript_intimidate.s:8).
- B12-26 Haze, Brick Break, Baton Pass leave Mist (battle_script.c:3534, :6556-6561); Defog's
  evasion drop meets Mist before Defog clears it (subscript_defog.s:19-36).
- B12-27 AI: Basic_CheckAlreadyUnderMist -8 (script.s:161, :587-590), no Expert entry; effect 46
  is in the four CheckHP tables (:7782, :7830, :7884, :7938).
- B12-28 CRIT_ODDS_EFFECTS (outcome.py) maps each effect that can change another battler's
  CalcCrit inputs (battle_lib.c:7105-7132: Focus Energy, held item, species, ability) to its
  write: Haze (battle_script.c:3534-3549), Knock Off (:6608-6632). crit_variants applies every
  subset, gated per battler by before[b, o] (doubles rule 1) and the write changing the odds.
  Guard: test_batch12 plays every table effect on a foe with all inputs live.
- B12-29 sample_outcome's static replacement flag: outcome.replacement_known, host side, true
  when a party Pokemon knows or can still learn Encore or an order.call_kind invoker; False
  leaves turn_events and outcome B out of the program (exact: replacement_possible is then false
  at every reachable state). Outcome B runs inside the lax.cond. Threaded through step_through,
  _play, _turn, playout*, leaves, stage_a, tests.engine.helpers.sample_rows. Mimic, Sketch and
  Transform must join the flag when supported.
- B12-30 A RANGE_SINGLE_TARGET_SPECIAL move with no living foe still succeeds: FightCommand's
  defender (bcp.c:1900) is the user (BattleSystem_Defender battle_lib.c:1665-1669, before the
  randomize branch), so BeforeMove's HasNoTarget (bcp.c:3190, NO_TARGET include/battle/common.h:
  56-61) passes after the PP decrement (:3181-3185) and SYSCTL_MOVE_SUCCEEDED is set (:3212):
  movePrevByBattler = moveTemp, the invoker (UpdateMoveBuffers :3945-3946); movePrev and
  moveProtect follow the attack message as with a living foe (:3937-3939). This range: Sleep
  Talk, Metronome, Assist, Copycat (Mirror Move, not modelled). Awake Sleep Talk fails in its
  script (effect_script_0097.s:5, :11-12); asleep, GoToMoveScript re-picks the called move's
  defender (battle_script.c:2120-2126), a foe-aimed one goes to subscript_no_target, last_move
  still Sleep Talk. Me First (RANGE_SINGLE_TARGET_ME_FIRST, not modelled) differs: randomize
  FALSE takes the single-target branch (lib.c:1709-1726), gets BATTLER_NONE, fails at :3190 with
  movePrevByBattler MOVE_NONE. The reference exempts the whole range from its no-target check;
  JAX agrees. Guard: test_invokers test_sleep_talk_with_no_living_foe_still_succeeds (awake and
  asleep).
- B12-31 A faster foe's Haze, Psych Up or Acupressure changes the stages a slower move's check
  reads (battle_controller_player.c:2873-2874), so outcome.hit_thresholds follows them: Haze
  zeroes the side's accuracy and the foe's evasion (BtlCmd_ResetAllStatChanges,
  battle_script.c:3534-3549), Psych Up sets the foe's evasion to the side's own turn-start stage
  (BtlCmd_CopyStatStages :6042-6053), Acupressure can draw evasion (+2 on its user in singles;
  BtlCmd_BoostRandomStatBy2 :8932-8956 loops ATTACK..MAX, accuracy and evasion included). Each
  is the stage variant's one "changed" value (a move has one effect); the White Herb variants
  stay. HIT_VARIANTS stays 128. The other batch 12 effects write no accuracy or evasion stage
  (Tickle, Power / Guard Swap, Captivate name other stats; Focus Energy, Power Trick, the healing
  family none; Mist only blocks later drops, B12-23). Guards: test_accuracy_classes
  test_every_accuracy_or_evasion_writer_is_seen_by_hit_thresholds (_FOLLOWED),
  test_haze_psych_up_acupressure_are_seen_by_hit_thresholds and (slow)
  test_haze_psych_up_acupressure_classes_match_the_live_check.
- B13-1 Twineedle (effect 77, BATTLE_EFFECT_POISON_MULTI_HIT) is effect 2's indirect TO_DEFENDER
  poison plus SetMultiHit 2, SYSCTL_MULTI_HIT_MOVE (effect_script_0077.s:5-6): SECONDARY_STATUS
  poison with MoveFlag.HIT_TWICE (tools/effect_map.py SCRIPT_FLAG_EFFECTS; effects.hit_count,
  enumerate's per-hit masks, the reference's hit loop). LoopMultiHit reloads the script per hit
  (bcp.c:3789), so MULTI_HIT_TRIGGER_SECONDARY rolls it per hit (bcp.c:3544-3554,
  battle_lib.c:1570-1586): outcome.secondary[side, hit], each roll Serene Grace-doubled
  (:1571-1575). Accuracy once (SYSCTL_MULTI_HIT_MOVE), crit and roll per hit. A defender fainted
  by hit 1 ends the loop (faintedMon, bcp.c:3780-3782); its roll gives nothing (curHP,
  battle_lib.c:1580). A substitute broken by hit 1 leaves SELF_TURN_FLAG_SUBSTITUTE_HIT set
  (bcp.c:3371; BattleSystem_SetupLoop clears no turn flags, battle_lib.c:3357-3362): hit 2 hurts
  the Pokemon but its poison is stopped (Battler_SubstituteWasHit, battle_lib.c:1582, :6259-6268;
  CheckSubstitute, battle_script.c:9181). Shield Dust and Poison / Steel immunity act per hit
  (subscript_poison.s). Vanilla 25 power 20 %, Kaizo 50 / 30 % (ROM; be_seq 77 unchanged). A
  chance-0 Twineedle would pack PLAIN with HIT_TWICE: two plain hits (Ruling 21).
- B13-2 Blaze Kick (200, HIGH_CRITICAL_BURN_HIT), Poison Tail and Cross Poison (209,
  HIGH_CRITICAL_POISON_HIT): BTLVAR_CRITICAL_BOOSTS + 1 (effect_script_0200.s:5, _0209.s:5) then
  effect 4's burn / effect 2's poison (:6): SECONDARY_STATUS with MoveFlag.CRIT_UP (as Razor
  Wind's, B10-12), read by outcome.side_crit_odds (criticalBoosts into
  BattleSystem_CalcCriticalMulti, battle_script.c:2158-2163; sCriticalStageRates stage 1 = 1 in 8,
  battle_lib.c:7097-7130). Crit and secondary are independent draws (CalcCrit's RandNext, then
  MULTI/ONE_HIT_TRIGGER_SECONDARY's); the stage is static, so no CRIT_ODDS_EFFECTS entry. Guard:
  tests/tools/test_effect_map.py test_script_flag_effects_match_every_script (every script with
  CRITICAL_BOOSTS or SetMultiHit 2 is classified).
- B13-3 Kaizo be_seq 141 (Hurricane, move 140: 120 power, 70 accuracy, 30 %) is effect 76's
  indirect TO_DEFENDER confusion plus SYSCTL_HIT_DURING_FLY (rom_diff.toml be_seq 141;
  effect_script_0076.s:5, as Thunder's _0152.s:6 for the Fly bit): KAIZO_EFFECTS 141 =
  SECONDARY_CONFUSE, KAIZO_FLAG_EFFECTS 141 = MoveFlag.HIT_AIR (no DOUBLE_SEMI: no
  BTLVAR_POWER_MULTI). build_tables' _move_flags gives a modelled Kaizo script its
  KAIZO_FLAG_EFFECTS bits in place of the vanilla effect-keyed ones. Member marked modelled.
- B13-4 Kaizo's overlay16 adds effect 141 beside BATTLE_EFFECT_THUNDER in CheckMoveHitAccuracy's
  sun branch (hitRate = 50, bcp.c:2915-2917; rom 0x0224F068) and CheckMoveHitOverrides' rain
  branch (MOVE_STATUS_MISSED cleared, :3024-3027; rom 0x0224F386), both inside NO_CLOUD_NINE:
  MoveFlag.THUNDER, read by outcome._rate and the reference (B9-1); the bit comes from E1's
  KAIZO_THUNDER_EFFECTS (G-50, same rule), not KAIZO_FLAG_EFFECTS. Both code entries marked
  modelled. Test: an Air Lock foe keeps 70 and the miss (Kaizo has no Cloud Nine species).
- B13-5 Triple Kick (vanilla 104, BATTLE_EFFECT_HIT_THREE_TIMES) is EffectId.TRIPLE_KICK (step,
  0, 0): ``SetMultiHit 3, SYSCTL_TRIPLE_KICK`` (effect_script_0104.s:5), three hits; the script
  reloads per hit (bcp.c:3789) and adds its step to BTLVAR_MOVE_POWER (:7), which
  BattleContext_Init zeroes once per action (battle_lib.c:1979) and BattleSystem_SetupLoop keeps
  (:3357-3362): hit k has power k x step, CalcMoveDamage's inPower (:6673-6677) before powerMul
  and Technician's <= 60 (:6688-6702). Vanilla step 10 (the script's constant); Kaizo's be_seq
  104 adds GetCurrentMoveData MOVEATTRIBUTE_POWER (step = the move's own power) and be_seq 88 is
  byte-identical (rom_diff.toml): Triple Kick 30 and Rock Wrecker 30 (hits 30/60/90), Triple
  Axel 20 and Fury Cutter 20 (20/40/60). damage.triple_kick_power, effects.hit_count (3), the
  reference's _calc_move_damage. Members 88 and 104 marked modelled (KAIZO_EFFECTS; no
  KAIZO_FLAG_EFFECTS bits).
- B13-6 SYSCTL_TRIPLE_KICK lacks SYSCTL_SKIP_ACCURACY_CHECK but has _SKIP_ACCURACY_OVERRIDES
  (system_control.h:81-86): every later hit's TryMove rolls CheckMoveHitAccuracy again with that
  moment's state (bcp.c:3255-3259) and skips CheckMoveHitOverrides (:3264-3268: no No Guard,
  Lock-On or Thunder / Blizzard weather sure hit; Protect and the semi-invulnerable test cannot
  change mid-move). A later miss is MOVE_STATUS_MULTI_HIT_DISRUPTED (CheckMoveFailure,
  :3320-3323): no damage, no miss message, AfterMoveMessage and AfterMoveEffects run as after a
  hit (the engine's item pass repeats, finding what the last one found), then LoopMultiHit ends
  (:3782). Outcome.hit[s, k > 0] is hit k's roll as its class over the same threshold grid
  without the overrides (outcome.hit_threshold_pair, one pass; _rate's ``overrides``); the grid
  covers what a hit can change before the next check (a disliked flavor berry's confusion for
  Tangled Feet, G-5; a White Herb). step's loop carries ``disrupted``; enumerate reads hit[s, k]
  once hit[s, k - 1] can have landed and gates hit k's own reads on it
  (OutcomeLaws.hit_later_thresholds / _possible; Protect restricts them too). The disrupted hit's
  own CalcCrit (before TryMove) can still print "A critical hit!" (MULTI_HIT_CRITICAL,
  bcp.c:3538-3541; subscript_critical_hit.s:5): not observed (step._hit_events' coarsening; Anger
  Point, which it would trigger, is unsupported). The Micle Berry flag (cleared by the first
  check, bcp.c:2967-2970) is an unsupported item.
- B13-7 Damage classes per hit: OutcomeLaws.damage is [side, hit, crit, roll]; hit k of a
  TRIPLE_KICK move has its k x step table, a BEAT_UP hit its member's (damage.beat_up_damage),
  every other hit hit 0's. Both are TABLE_DAMAGE_EFFECTS and INPUT_NEUTRAL_EFFECTS (they write the
  foe's HP only); damage_inputs_can_change's own-move clause treats their later hits as
  MULTI_HIT's (inputs change only through the HP, contact and item clauses).
- B13-8 Guards: test_triple_kick_later_classes_equal_raw_rolls (each later class steps like the
  raw rolls it stands for, after a faster Sand-Attack), test_triple_kick_law_exact_and_sampled
  (slow: a No Guard user against brute force and the sampler; -1 accuracy against the sampler),
  test_effect_map test_triple_kick_and_beat_up_scripts (every SetMultiHit 3 script).
- B13-9 Beat Up (154, both games) is EffectId.BEAT_UP: BtlCmd_BeatUp (battle_script.c:6173-6247)
  walks the party in slot order (BattleSystem_GetPartyPokemon: party array, not display order)
  and hits once per slot BeatUpEligibleMon passes (:6157-6164): the user's own slot whatever its
  HP or status, or one with HP, a species, no egg and MON_CONDITION_NONE
  (damage.beat_up_slots). Nothing changes a benched Pokemon's HP or status mid-move, so the
  start's list holds (effects.hit_count). Accuracy once (multiHitAccuracyCheck =
  SYSCTL_MULTI_HIT_MOVE, :6187); the loop ends with the list or on a faint (bcp.c:3780-3782).
  A full party gives six hits: outcome.MAX_HITS 6. Every per-hit draw's sixth column comes from
  fold_in(key, 42) (outcome.SIXTH_HIT_FOLD, split per field), so hits 0-4 keep their samples
  (test_outcome's recorded draws); the 2-5 law's reads stay at 5.
- B13-10 Beat Up's damage (:6206-6218): the member's species base Attack x the move's power x
  (its level x 2 / 5 + 2) / the defender's species base Defense / 50 + 2 (both by personal row:
  SpeciesData_GetFormValue), x criticalMul (CalcCrit per hit with the user's crit stage; Sniper
  x3, battle_lib.c:7139-7144), Helping Hand (doubles), then the variance (:7081-7092). No
  CalcDamage: no stage, item (no Life Orb boost, battle_script.c:1346; its recoil still follows
  SYSCTL_MOVE_HIT, bcp.c:4758-4765), ability, status, screen or weather; and
  SYSCTL_IGNORE_IMMUNITIES (effect_script_0154.s:5) skips CheckTypeChart (bcp.c:2355): no STAB,
  effectiveness, Wonder Guard or noEffect, and subscript_type_resist_berry.s:5 skips the resist
  berry (effects.IGNORES_TYPE_CHECKS). The per-hit "{0}'s attack!" message and
  SYSCTL_MULTI_HIT_SKIP_MESSAGE (no "Hit N times!") are text only.
- B13-11 Vanilla Fury Cutter (#119) and the recharge (#80, Hyper Beam's family and vanilla Rock
  Wrecker) are vanilla only (no Kaizo move uses either: Kaizo moves.json) and each needs new
  battler state: UNSUPPORTED, Owed.
- B13-12 Weather Ball (203, EffectId.WEATHER_BALL): CalcWeatherBallParams (battle_script.c:
  6843-6872) under NO_CLOUD_NINE and any FIELD_CONDITION_WEATHER bit (fog included, condition.h:
  131-135) sets movePower = power x 2 and moveType Water / Rock / Fire / Ice (rain / sand / sun /
  hail; fog sets none, so Normal); with Cloud Nine / Air Lock alive it sets nothing. Judgment
  (268, EffectId.JUDGMENT): effect_script_0268.s:5-95 sets moveType from the attacker's
  Battler_HeldItemEffect HOLD_EFFECT_ARCEUS_* (any holder; Klutz and Embargo hide the item,
  battle_lib.c:5352-5363, both supported, B13-29..31), read from each game's
  Tables.item_hold_effect (damage.PLATE_TYPES; no item-table change, so no e1-model commit). The
  script runs before TryMove's accuracy, chart and absorbing-ability states (bcp.c:3212-3216),
  which read battleCtx->moveType (bcp.c:2358-2361, battle_lib.c:3490-3497, 2582-2597 STAB,
  6680-6684 and 6716-6722 the type items), so damage.move_type_of returns it, as for Hidden
  Power. Normalize (lib.c:6679) is an unsupported ability. Pure AI helpers:
  damage.weather_ball_params, damage.judgment_type.
- B13-13 Water Spout / Eruption (vanilla 190; Kaizo Eruption is effect 253): FLAIL with kind
  FLAIL_FALLOFF: BtlCmd_CalcHPFalloffPower (battle_script.c:6645-6657) power x curHP / maxHP, 1
  when 0, read at the action's only hit (damage.falloff_power). prep.variants adds every power
  band top (falloff_cuts) to the entry-HP regions, as Flail's bands (up to power - 1 cuts).
- B13-14 Gyro Ball (219, EffectId.GYRO_BALL): 1 + 25 x monSpeedValues[defender] /
  monSpeedValues[attacker], at most 150 (battle_script.c:7123-7131). monSpeedValues is written by
  every CompareBattlerSpeed (battle_lib.c:1372-1373); BranchActions runs SortMonSpeedOrder before
  each action (bcp.c:888, back after MoveEnd :3999), so the speeds are those at the user's action
  start (before its status checks and Custap Berry: a waking Quick Feet user keeps its x1.5, an
  Unburden user its held-berry Speed): step._use_move passes the power from its state_in as
  CalcMoveDamage's inPower (damage._pre_random in_power; the turn-start table reads the state). A
  user Speed that can reach 0 (below 16: -6 is 10/40, paralysis / 4, lib.c:1225, :1259-1261)
  divides by zero, left to the ROM's division routine: coverage.gyro_speed_line refuses it
  (fight, pool and prep, GYRO_SPEED_MIN; twice it for a sSpeedHalvingItemEffects holder, Iron
  Ball since G-56: / 2 between the stage and paralysis, battle_lib.c:1240-1246). Pure helper
  damage.gyro_ball_power.
- B13-15 Trump Card (235): sCurrentPPScaledPower[min(ppCur, 4)] (200, 80, 60, 50, 40) of
  moveSlot after BattleControllerPlayer_DecrementPP (bcp.c:3182; Pressure's cost included),
  moveSlot being the invoker's for a called move (common.h:8), passed as inPower from _use_move.
  Punishment (245): 60 + 20 x the defender's stages above neutral over all 8, at most 200
  (battle_script.c:7354-7371), read at the hit.
- B13-16 Wake-Up Slap (217): DOUBLE_POWER (POWER_IF_TARGET_ASLEEP, cures 1): SmellingSalt's
  script on MON_CONDITION_SLEEP (any count): CheckSubstitute first (:5), powerMul 20 and an ON_HIT
  indirect subscript_heal_target_sleep.s (sleep bits off on a target with HP left, :5-6;
  effects._cure_paralysis). ON_HIT rolls no chance, so Kaizo's chance 100 is inert.
- B13-17 Assurance (231, EffectId.ASSURANCE): movePower = the move's power, x 2 when the
  defender's turnFlags.assuranceDamageMask is nonzero (effect_script_0231.s:5-9).
  subscript_update_hp.s:18-19 stores every update's hpCalcTemp unless above 0: a loss sets it, a
  zero update (a 1-HP False Swipe or endure target, Pain Split at the average) clears it, a heal
  leaves it; the substitute path (bcp.c:3359-3378) runs no UPDATE_HP. turnFlags is per battler:
  SetupNextTurn (lib.c:2231, after the turn-end replacements) and CleanupFaintedMon (:2190-2193)
  clear it, a switch does not. Per-battler BattleState.ts_assurance (turn scratch: kept at a
  pause, 0 between turns; replace() clears a replacement's hazard mark), marked by
  effects.assurance_mark at the action-phase updates (hits, recoil, Struggle, contact abilities,
  the self-hit, the crash, Life Orb, the hazards, Curse, Substitute, Belly Drum, Pain Split); the
  residual updates come after every reader. Reference: _Turn.assurance, _lose_hp.
- B13-18 Damage classes: WEATHER_BALL and JUDGMENT are TABLE_DAMAGE_EFFECTS (the weather, switch
  and Knock Off clauses cover their inputs), Water Spout is FLAIL's (hp_sensitive); GYRO_BALL,
  TRUMP_CARD, PUNISHMENT and ASSURANCE are NON_TABLE (action-start speeds, the PP after the
  decrement, the defender's stages, the turn's flag); all six are INPUT_NEUTRAL (they write the
  foe's HP only). No new draw, no new Outcome field.
- B13-19 recycleItem is per battler (battle_context.h:307): BtlCmd_RemoveItem writes it
  (battle_script.c:8977), BtlCmd_TryRecycle reads and clears it (:9005-9007), nothing else
  writes it, so a switch keeps it and a newcomer recycles its predecessor's item.
  BattleState.recycle_item is written at every RemoveItem (effects.consume_item,
  hooks.held_item_check, hooks.pluck_berry, the reference's _remove_item); item_consumed stays the
  per-slot observation record. Recycle (184, EffectId.RECYCLE): effect_script_0184.s:5 fails a
  user holding an item, TryRecycle an empty record (MOVE_STATUS_FAILED); else the item is held
  again at the script (:7, UpdateMonDataFromVar: CopyToParty of a held item, no AI change; "found
  one {1}" tags BTLSCR_MSG_TEMP: no reveal), and AfterMoveEffects' held-item check sees it (a
  restored berry in a pinch is eaten at once). The script runs only past the PP and target checks
  (MOVE_STATUS_NO_PP is a NO_EFFECTS status, bcp.c:3205-3216).
- B13-20 Natural Gift (222, EffectId.NATURAL_GIFT): BtlCmd_CalcNaturalGiftParams
  (battle_script.c:8088-8102) reads Battler_NaturalGiftPower / _Type (battle_lib.c:5465-5480:
  Battler_HeldItem, which Klutz and Embargo empty, B13-29..31) from ItemData.naturalGiftPower /
  _Type (include/item.h:63-64; Tables.item_gift_power / item_gift_type, 0x1F = none,
  itemproc.c:242); power 0 (no item, a non-berry) fails it (effect_script_0222.s:11-13), moveType
  then kept 0. Otherwise CalcCrit, CalcDamage, then RemoveItem BTLSCR_ATTACKER (:8) before
  TryMove: a miss, Protect or an immunity still eats the berry (into recycleItem and
  item_consumed; Unburden then doubles). The hit's power and type come through damage.gift_item:
  the held item, or once removed the berry step._use_move passes in (never recycle_item, which
  outlives the action: B13-37); a berry is no damage, crit or accuracy input of its attacker, so
  the hit at its own time equals CalcDamage at the script's.
- B13-21 Laws (neither move draws anything new): NATURAL_GIFT is INPUT_NEUTRAL (it writes the
  foe's HP and removes its own berry, which the item clauses over-approximate) and NON_TABLE;
  RECYCLE is NON_TABLE and INPUT_CLAUSE with a recycle clause (a faster foe's Recycle: every hit
  of the side may change, conservative). Recycle's restored item may meet its user's move-end
  item check: stage_free and para_free treat a Recycle like a Pluck (Salac, White Herb, Lum,
  Cheri). _end_speeds' Unburden bit toggles: a holder of an item unburdened, a holder of none
  holding one again (stand-in: its battler's record, else item 1; no recyclable item acts on
  speed but through Unburden's heldItem test). Accuracy: both write no accuracy input (no
  accuracy item can be consumed, so none comes back). No new fold_in index.
- B13-22 UNSUPPORTED (Owed): Fling (233) and Embargo (232). Fling's flung item acts on its target
  through BattleSystem_FlingItem (battle_lib.c:5854-6156) and the 30 fling subscripts: heals,
  cures, White Herb's reset at TryFling time (before the accuracy roll and CalcDamage,
  :6029-6034), Light Ball / Poison Barb / orbs / King's Rock statuses and flinch on the defender
  as INDIRECT effects, pinch-berry raises, the flavor confusion, SELF_TURN_FLAG_PLUCK_BERRY left
  set when the subscript does not run; its CalcDamage / CalcCrit read the item still held but
  ApplyTypeChart (Expert Belt) and the accuracy (Wide / Zoom Lens) no longer do. Each needs its
  own law relevance (para / stage / confusion / flinch reads, the order speeds) and a per-time
  item view in damage; once a flung item is in recycleItem, Recycle can give back Bright Powder,
  Focus Band, Quick Claw, King's Rock (the hit thresholds, focus_band and flinch_item laws read
  the turn-start item). Embargo needs the Battler_HeldItem suppression (battle_lib.c:5352-5363,
  Klutz's too) at every held-item read of both engines and the laws (no single accessor: about 50
  direct reads), the per-battler embargoTurns, its turn-end count and subscript_embargo_end.s,
  Baton Pass keeping it, the bag block, and Fling's and Natural Gift's Embargo branches.
- B13-23 Trainer AI of batch 13 (outcome.KAIZO_TRANSCRIBED, by the Kaizo battle effect the
  scripts read): Basic_CheckCanRecycle (script.s:238 -> :911-915, LoadRecycleItem
  trainer_ai.c:2510-2518 = BattleState.recycle_item), Basic_CheckNaturalGift (:258 ->
  :1053-1059; Basic_NaturalGiftBerries :1061-1126 = item ids 149-212), NonStandard for Gyro
  Ball, Trump Card, Punishment (:256, :265, :274 -> :572-585); Expert_Recycle (:1737 ->
  :4517-4534, roll :4526), Expert_WaterSpout (:1742 -> :4610-4632), Expert_HighCritical for
  Blaze Kick and Poison Tail / Cross Poison (:1748, :1754 -> :2827-2843), Expert_WakeUpSlap
  (:1761 -> :4829-4847), Expert_GyroBall (:1763 -> :4869-4871, PopOrEnd only),
  Expert_Assurance (:1774 -> :5189-5216, rolls :5204, :5212; its RecoilBerries table holds item
  ids 211-212 compared with LoadHeldItemEffect's hold effect, trainer_ai.c:1901-1913, ids at
  most 146: never true, transcribed as written), Expert_TrumpCard (:1778 -> :5308-5359,
  LoadCurrentMovePP :2024-2028; Kaizo's Trump Card has 1 PP), Expert_Punishment (:1788 ->
  :5900-5951, SumPositiveStatStages :2243-2257; each TryScorePlusN adds and falls into the
  next, unlike the decomp's comment). New ai.commands: load_recycle_item, load_current_move_pp,
  sum_positive_stat_stages, load_held_item_effect. read_bounds.py regenerated
  (tools/ai_read_bounds.py); ALLIN_READS stays 18.
- B13-24 No flag script names Twineedle (77), Triple Kick / Rock Wrecker (104), Kaizo Hurricane
  (141, BATTLE_EFFECT_UNUSED_141), Beat Up (154), Weather Ball (203) or Judgment (268): no
  BATTLE_EFFECT_* name of theirs in script.s (Basic_ScoreMoveEffect :133-286, Expert_Main
  :1623-1811, the IfMoveEffectKnown / table entries); they take the generic paths. Other mentions
  are tables already read whole: Expert_Encore's list (:3254 Recycle, :3268 Natural Gift),
  Risky_RiskyEffects (:6556 Gyro Ball), CheckHP's attacker table (:7853 Water Spout), and
  Expert_LuckyChant's IfMoveEffectKnown of 200 / 209 (:5504-5505, the player's moves; Lucky
  Chant's label is not transcribed). Kaizo's 88 (Triple Axel, Fury Cutter) keeps vanilla's
  RANDOM_DAMAGE_1_TO_150_LEVEL paths (E1 batch 14) and 253 (Eruption) Expert_RecoilMove (E2
  batch 10): the AI reads the effect id, not the patched be_seq.
- B13-25 TrainerAI_CalcDamage's Natural Gift and Judgment cases (trainer_ai.c:2885-2971) are
  modelled (damage.ai_case_power / ai_case_type): Natural Gift's power is the heldItem argument's
  naturalGiftPower, its type the item's naturalGiftType when that power is not 0, else
  TYPE_NORMAL; Judgment's power 0 and its plate type, else TYPE_NORMAL; CalcMoveDamage reads 0 as
  the table power / type (battle_lib.c:6673-6684), ApplyTypeChart likewise (:2588-2590).
  heldItem is the attacker's battleMons item (:1041-1050, :1560-1569, :2130-2139, :2210-2219), a
  benched member's own for IfPartyMemberDealsMoreDamage (:2156-2165, commands.estimate's item),
  the player's for IfBattlerDealsMoreDamage's last move (:2228-2237). Klutz and embargoTurns gate
  both cases (:2886, :2898; B13-22), both supported (B13-29..31). Weather Ball takes the default
  case (table power and type, :3074-3078). The other batch-13 moves need no case:
  TrainerAI_CalcAllDamage calls TrainerAI_CalcDamage once per move (:2799-2838) and its default
  (:3074-3078) passes power 0 and TYPE_NORMAL, so Twineedle and Triple Kick are one hit at the
  table power, Beat Up, Trump Card, Punishment, Assurance and Wake-Up Slap their table power, and
  Water Spout is in sNoDamageCalcMoveEffects (:31-45: no estimate); the existing AI path does
  exactly that.
- B13-26 TrainerAI_MoveType (trainer_ai.c:3127-3245): Natural Gift is Battler_NaturalGiftType
  (battle_lib.c:5477-5481) of the battler's item, whatever its power (no item: 0, the table
  type; a non-berry: 0x1F, which matches no chart row or attacker type in ApplyTypeChart's u8
  moveType, :2564-2640: damage.AI_TYPELESS = TYPE_MYSTERY, whose chart row is neutral);
  Judgment the plate type of Battler_HeldItemEffect, else TYPE_NORMAL (the table type); Weather
  Ball Water / Rock / Fire / Ice under NO_CLOUD_NINE in rain / sand / sun / hail (:3209-3227).
  Elsewhere (no weather, fog alone, Cloud Nine / Air Lock) the case assigns nothing: the ROM's
  function (overlay14 0x0222327C) returns r5 unwritten, the caller's r5 (the BattleContext
  pointer in AICmd_CalcMaxEffectiveness, 0x02221064; the BattleSystem pointer in
  AICmd_IfMoveEffectivenessEquals, 0x0222111A), a heap address whose low byte becomes the type:
  not knowable, so policy._refused refuses an AI Weather Ball in those states only
  (commands.MOVE_TYPE_MOVES = Weather Ball; damage.ai_weather_ball_typed). Backlog name: "move-id
  case of TrainerAI_MoveType where it assigns no type".
- B13-27 Gyro Ball's TrainerAI_CalcDamage case (trainer_ai.c:2995-3003) stays refused
  (commands.CALC_DAMAGE_MOVES): it reads monSpeedValues, written by every CompareBattlerSpeed
  (battle_lib.c:1372-1373): the action-order sorts of the last turn
  (battle_controller_player.c:780, :888, :3992), and the AI's own IfSpeedCompareEqualTo
  (trainer_ai.c:1109, :1121) and :2081 compares earlier in the same decision, in flag-major,
  slot-minor order. Modelling it needs a per-battler speed record written at each sort (new
  state) and the compare order across the slots the per-slot scoring factors apart (Owed). Gyro
  Ball's scripts are transcribed (B13-23), so only this case refuses it.
- B13-28 Owed (AI): the voluntary-switch rule still refuses Natural Gift, Judgment and Weather
  Ball on both sides (switching._MOVES): AI_HasSuperEffectiveMove / AI_CannotDamageWonderGuard
  / AI_OnlyIneffectiveMoves type the AI's moves with TrainerAI_MoveType (modelled but Weather
  Ball's unassigned case, whose r5 differs per caller), the bench's with Move_CalcVariableType
  (battle_lib.c:8097, its Natural Gift, Judgment and Weather Ball cases), the player's moveHit
  with TrainerAI_MoveType (trainer_ai.c:3767). And Gyro Ball's AI damage (B13-27).
- B13-29 Battler_HeldItem's gate (battle_lib.c:5352-5363): ITEM_NONE under Klutz
  (Battler_Ability, :5354) or a nonzero embargoTurns (:5358), else heldItem: e1-model
  ``state.held_item`` (Embargo the active battler's; per-battler field ``embargo``, embargoTurns
  with its MOVE_EFFECT_EMBARGO bit, volatile, kept by Baton Pass, moves.h:46-48,
  battle_lib.c:2112). Gated (every Battler_HeldItemEffect / _HeldItemPower(ALL or 0) /
  _NaturalGift* read, :5438-5481): CalcMoveDamage's items (:6662-6668, the AI estimate too),
  ApplyTypeChart's Expert Belt (:2578), CompareBattlerSpeed's Choice Scarf, Quick Claw, Custap,
  Lagging Tail (:1215-1218), the crit items (:7117), the accuracy items (bcp.c:2949-2957), the
  item triggers (:4581, Leftovers :4927, detrimental :5310, on hit :5381, pivot :7340-7342),
  CheckItemHoldEffect / GetItemHoldEffect / GetItemEffectParam (battle_script.c:7912-7976: Big
  Root, resist berries, Grip Claw, Power Herb, Judgment), Light Clay (:4229), weather rocks,
  Life Orb / Metronome (:1346-1352), the Choice lock and its move ban (:6943, battle_lib.c:2257),
  Focus Band / Sash (:9106), King's Rock (bcp.c:4633), Shell Bell / Life Orb recoil
  (bcp.c:4719-4720), Iron Ball's grounding (:5505, the AI chart's trainer_ai.c:3333-3791),
  Natural Gift (:5465-5481), TrainerAI_MoveType (trainer_ai.c:3137). Raw heldItem: the
  speed-halving items (:1240-1246), Unburden (:1268-1272, :4434, :161-166), Knock Off
  (battle_script.c:6614-6627), Pluck (:8122-8127, battle_lib.c:5479, :5544, :5833), RemoveItem
  (:8978), Recycle (effect_script_0184.s:5), Trace's Griseous Orb (:3723), Sticky Barb's empty
  attacker (:5389, :7379), the exp payout's party items, the AI's LoadHeldItem /
  IfHeldItemEqualTo / LoadHeldItemEffect (trainer_ai.c:1891-1931); TrainerAI_CalcDamage's
  Natural Gift / Judgment cases test their ability / embargoTurns arguments (:2886, :2898): the
  active's gated item, a Klutz party member's none (its embargo FALSE, :2163). The laws read the
  gated items at the turn start (hit thresholds, crit odds, King's Rock chance); the possibility
  flags keep raw items (over-approximate). step.qc_kept keeps the raw item (over-keeping only
  splits states). Behaviour-neutral without Klutz or Embargo.
- B13-30 Klutz (AbilityId.KLUTZ, decomp 103) is supported: its battle reads are the gate (:5354)
  and BattleSystem_PluckBerry's eater test (:5839-5844, with MOVE_EFFECT_EMBARGO: the berry is
  eaten with no effect, scriptTemp 0, the pluck flag left set; hooks.pluck_berry, B13-31); its AI
  reads are in Fling's, Trick's and Skill Swap's scripts (trainer_ai/script.s:1245, :1273,
  :4270, :4293, :4322, :7204; unsupported moves). INPUT_NEUTRAL (it holds from the turn start;
  nothing changes it mid-turn).
- B13-31 Embargo (232, EffectId.EMBARGO): effect_script_0232.s:5 fails (MOVE_STATUS_FAILED,
  before TryMove) behind a substitute; else a direct ON_HIT TO_DEFENDER
  subscript_embargo_start.s (run without MOVE_STATUS_NO_EFFECTS, battle_lib.c:1484-1489) fails
  (:57) on a substitute, Multitype (raw ability), embargoTurns != 0, a raw Griseous Orb, the
  defender's quickClaw (until it moved) or custapBerry flag (:6-11; the custapBerry flag clears
  when its holder's move starts, subscript_check_quick_claw.s:26-27: play_hit's foe_custap reads
  it so, matching the reference's Pluck / Knock Off), skipping AfterMoveEffects
  (primary_effect_fails); else embargoTurns = 5 (:14). MON_COND_CHECK_STATE_EMBARGO
  (bcp.c:1671-1677) counts it down after Taunt and before Yawn and the item checks (a count
  reaching 0 lets ITEM_CONDITION / _DETRIMENTAL act that turn; Leftovers, :1369, came first). A
  switch clears it, Baton Pass keeps it (B13-29). Trainer bag items ignore it (the bag UI's
  embargoRemainingTurns, battle_display.c:3847, is the player's; TrainerAI_ShouldUseItem and
  BattleSystem_UseBagItem never read it). Laws: INPUT_CLAUSE (the embargo clause: a holder of a
  gated item), NON_TABLE; CRIT_ODDS_EFFECTS write (embargo on every battler); the hit grid's
  second half is a Wide Lens holder's threshold with the lens hidden (possible when the foe's
  Embargo can go first; a hidden Zoom Lens / Micle Berry is the first half); Future Sight's roll
  is kept whole while an Embargo is used or up (its turn-end count can end before the hit); the
  turn end's speed tie is enumerated whenever a foe's Embargo can hide a Choice Scarf. No new
  fold_in index, no Outcome field.
- B13-32 Trainer AI of Embargo: Basic_CheckEmbargo (script.s:262 -> :1184-1197: the defender's
  MOVE_EFFECT_EMBARGO -10; a defender recycle item in a Frontier battle -10, never in the story)
  and Expert_Embargo (:1775 -> :5223-5229, +1 unless the 128 roll at :5225 jumps);
  commands.MoveEffectMask.EMBARGO (moves.h:34), BATTLE_TYPE_FRONTIER (battle.h:33); read_bounds
  regenerated.
- B13-33 Fling (233) stays UNSUPPORTED (Owed): its 30 fling effects on the target
  (battle_lib.c:5854-6156) each need their own law relevance and a per-time item view; its AI
  entries (Basic_CheckFling, Expert_Fling) are not transcribed while the move is refused.
- B13-34 The Metronome item's count (BattleSystem_UpdateMetronomeCount, battle_lib.c:6160-6178)
  is updated at the end of BeforeMove (bcp.c:3221), before the move's script (:3214): a charge
  move's strike turn is still VOLATILE_CONDITION_MOVE_LOCKED there
  (subscript_charge_move_cleanup.s:6 clears it inside the script), so it does not count; a
  Recycle, Natural Gift or Power Herb changes the held item only after it. step.py runs the
  update right after _set_last_move, as the reference does. E1 owns the item (G-60).
- B13-35 The voluntary-switch rule and the post-KO replacement type a party member's move with
  Move_CalcVariableType (battle_lib.c:8097-8203; trainer_ai.c:3324, :3443, :3502, :3782;
  battle_lib.c:7998, :8051): Natural Gift the naturalGiftType of the member's party item
  (MON_DATA_HELD_ITEM, no Klutz / Embargo gate, whatever its power; 0 is the table type through
  CalcEffectiveness, :2692-2696; a non-berry's 0x1F no type), Judgment its HOLD_EFFECT_ARCEUS_*
  type (else TYPE_NORMAL), Weather Ball the weather's type under NO_CLOUD_NINE and rain, sand,
  sun or hail; elsewhere the case leaves ``type`` unassigned, an uninitialized register: refused
  there (switching.weather_ball_refused: an AI party Weather Ball or a Weather Ball moveHit,
  typed by TrainerAI_MoveType at :3767; replacement: a candidate's). The AI's active and the
  player's moveHit go through TrainerAI_MoveType (damage.ai_move_type, B13-26). The three moves
  leave switching._MOVES and PLAYER_MOVE_REFUSALS.
- B13-36 Gyro Ball's TrainerAI_CalcDamage case (trainer_ai.c:2995-3003) stays refused (Owed);
  model from the decomp: monSpeedValues is written only by CompareBattlerSpeed
  (battle_lib.c:1372-1373), after its both-alive short-circuit (:1204-1210 returns first), with
  values a function of the state at the call (no Quick Claw term). Writers: SortMonSpeedOrder
  (battle_lib.c:3364-3381) at TrainerMessage and ShowBattleMon (bcp.c:251, :263, battle start
  after the switch-in effects), BranchActions (bcp.c:888: before every action and once after the
  last, so an AI decision first reads the value of the previous turn's last BranchActions with
  both actives alive; replacements and the turn end write none), CalcTurnOrder (:780) and the
  Trick Room re-sort (:3991-3992), each followed by a BranchActions on the same state; and within
  the decision IfSpeedCompareEqualTo (trainer_ai.c:1109, :1121) and LoadBattlerSpeedRank
  (:2081), which write the current speeds (ShouldSwitch and ShouldUseItem make none). Needs a
  per-battler state field (both engines, the differential) and an AI context bit "a compare ran"
  carried flag-major, slot-minor (_run_flags), choosing the stale or current estimate at
  FlagMoveDamageScore, the kill checks and IfPartyMemberDealsMoreDamage
  (IfBattlerDealsMoreDamage's Copycat case divides the player's value by itself: 26). The
  per-slot exact distribution runs one read pattern for every slot (_slot_move_distribution), so
  a compare reached past a random read in an earlier slot correlates the slots: refuse that state
  when the two powers differ.
- B13-37 (review fix) Natural Gift reads only the held item: Battler_NaturalGiftPower / _Type
  take Battler_HeldItem (battle_lib.c:5465-5480), and power 0 fails the move
  (battle_script.c:8094-8099). recycleItem is per battler and outlives the action, the turn and
  switches (:8977, :9005-9007), so it is never an input: damage.gift_item(state, side, gift) and
  the reference's _gift_item take the berry this action's own RemoveItem (effect_script_0222.s:8)
  removed as an explicit argument (step._use_move's gift_removed, threaded through move_type_of,
  _pre_random, damage_for_roll, play_hit and thaw_target; the reference's _Turn.gift), else read
  the held item. A Natural Gift user holding nothing has the table type and power 0 in
  move_type_of, the turn-start tables and the enumeration (prep and playout included), whatever
  its record says.
- B13-38 (review fix, doubles convention 1) The defender is passed in: damage.beat_up_damage(...,
  def_side, ...) (DEFENDING_MON, battle_script.c:6209), and in the reference _beat_up_damage(...,
  defender, ...), _gyro_power(..., defender) (CalcGyroBallPower, :7123-7131) and
  _script_power(..., defender, ...) (BTLVAR_DEFENDER_ASSURANCE_DAMAGE_MASK,
  effect_script_0231.s:7). Singles callers name the other side once, at the action or table
  level (damage_table, damage_for_roll, enumerate, the reference's _use_move and _hit).
- B13-39 (process, for E1) The commit "feat(e2): batch 13 Embargo" changed E1's item paths
  without the "e1-model:" title: hooks.pluck_berry (a Klutz or embargoed eater eats the berry
  with no effect, battle_lib.c:5839-5844, and the eaten / effect-on split that moves the
  RemoveItem / recycle_item writes) and step's play_hit argument foe_custap, gated on
  ~target_moved (subscript_check_quick_claw.s:26-27). Already merged; E1 should review those
  paths.
- B13-40 (review measure) prep.variants.falloff_cuts gives a Water Spout / Eruption holder one
  entry-HP region per power band below the top (up to power - 1 = 149 cuts, capped by its max
  HP). Vanilla sample party vs Roark (member 0, max HP 44, Water Spout or Eruption in slot 4):
  that member's regions 7 -> 44, its one-change variants 10 -> 47, the party's one-change
  variants 47 -> 84 (x1.79), the product of (1 + variants) per member 310464 -> 1354752
  (x4.36). The cuts stay exact (Owed: the prep cost).
- B13-41 (merge note 1) Weather Ball's AI-type refusal (policy ~3811, switching.weather_ball_refused,
  replacement ~140) on the Kaizo splits' opponent teams: 4 battles carry Weather Ball. It fires in
  2: Byron split, Galactic Terra #3's Castform under fog (every state); Galactic split, Galactic
  Erripaus's Bronzong (clear start: until a weather is set). Byron split Worker Conrad's Blastoise
  and Volkner split Sailor Jack's Tropius start in permanent sun: typed, unless the player brings
  Cloud Nine / Air Lock. Player-side Weather Ball (moveHit) not counted.
- B13-42 (merge note 2) Every JAX action-phase HP update goes through effects.hp_update (HP within
  [0, maxHP] plus the assuranceDamageMask, subscript_update_hp.s:18-19), as the reference's
  _lose_hp: the hits, recoil, Struggle, the contact abilities, the self-hit, the crash, Life Orb,
  Jaboca / Rowap and the Enigma heal, the hazards (one update each), Curse, Substitute, Belly Drum,
  Pain Split. assurance_mark is gone.
- B13-43 (merge note 3) test_triple_kick_later_classes_equal_raw_rolls: 36 s and 50 s per case,
  peak 5.05 GB: marked slow.
- B13-44 (merge note 4, doubles convention 1) reference._fight takes the action's target battler;
  Gyro Ball's monSpeedValues read it (the turn skeleton passes 1 - side, as step.py's target_side).
- B13-45 (merge note 5) damage._pre_random's Gyro Ball / Trump Card fallback (in_power 0) feeds
  the heuristics' damage tables only: engine hits pass in_power, the trainer AI takes
  ai_case_power, the replacement AI skips both (table power 1, battle_lib.c:8053;
  res/moves/{gyro_ball,trump_card}/data.json).
- Owed 13 closed: tools/differential.py (`uv run python -m tools.differential N SEED|A-B ...`),
  the generator and field check of tests/test_differential.py; a seed replays its turns.
- D1-1 (manager, accepted) `infatuated` stays 0/1 in D1: singles stays byte-identical. Its
  1 + battler form (docs/doubles-state-design.md 5.3) is D5's, on E1's list.
- D1-2 `chatot_cry` is per party [P]: party 0 holds the save's cry, every other party 0.
  `BattleSystem_GetChatotCry` (battle_system.c:915-924) reads `chatotCries[battler]` in 2vs2 and
  for the enemy of a tag battle, `[battler & 1]` in plain doubles; only `BATTLER_PLAYER_1` is
  filled from the save (field_battle_data_transfer.c:260, :319, :385), every other entry is
  `ChatotCry_New` (:97), parameter 0. So in every format (encounter.c:730-740) the cry is the
  sending trainer's. Readers index `chatot_cry[party(fmt, b)]`; singles [P] = [2], byte-identical.
- D1-3 The battle-end restore (`step._faint_check`, `reference._faint_check`) runs
  `_restore_transform`, `_restore_types`, `_restore_ability`, `_restore_moves` for every battler
  whose `fmt.party_of` is 0 (the player's party is what the next battle carries); the
  opponent's battlers keep their copies for the payout. Singles: battler 0 only, as the trial.
- D1-4 Plus / Minus (`damage.side_ability_count`, `reference._side_active`) count every living
  battler on the field of the asker's side, the asker included: a loop over all B battlers with
  the other side masked out (`COUNT_ALIVE_BATTLERS_OUR_SIDE`, battle_lib.c:2886-2893).
  `damage.side_active_slots` is removed (its only caller was this count). Singles: the foe is
  masked out, the same count.
- D1-5 The trainer AI's bag is per opposing trainer slot: `aiContext.trainerItems[MAX_OPPONENTS]
  [MAX_TRAINER_ITEMS]` and `trainerItemCounts` (include/battle/ai_context.h:31, :36;
  `MAX_OPPONENTS` 2, include/constants/battle.h:68). `BattleControllerPlayer_InitAI` fills
  `trainerItems[battler >> 1]` for each `battler & BATTLER_THEM` from
  `BattleSystem_GetTrainerItem(battler)` = `trainers[battler].header.items`
  (battle_controller_player.c:4796-4811, :4803-4809; battle_system.c:1414-1417); `trainers[i]` is
  `dto->trainer[i]` (battle_main.c:1066), loaded per `dto->trainerIDs[i]`
  (trainer_data.c:31-37). The AI reads and spends `trainerItems[battler >> 1]`
  (trainer_ai.c:4092-4093, :4193); an AI partner never uses items (:4068-4072).
  `BattleSystem_GetTrainerItem` reads `trainers[battler]` directly (battle_system.c:1414-1416),
  not through `BattleSystem_GetTrainer`'s `battler & 1` map for a plain double (:301-309), so
  battler 3's bag is whatever `dto->trainer[3]` holds, and that depends on how the double starts:
  - Spotted (`Battles_ApproachingTrainerDoubles`, res/field/scripts/scripts_battles.s:156-173):
    the second approaching trainer is the twin map object `FindTrainerPartner` finds by the same
    trainer id (src/trainer_encounter.c:118-123, :356-371; both objects' scripts give one id
    through `Script_GetTrainerID`, script_manager.c:498-505), so `StartTrainerBattle VAR_0x8004,
    VAR_0x8005` passes the id twice; `enemyTrainer1ID == enemyTrainer2ID` gives
    `BATTLE_TYPE_TRAINER_DOUBLES` (encounter.c:736-737), both enemy ids are written (:751-752)
    and `Trainer_Encounter` loads the trainer into both (trainer_data.c:31-37): battler 3 holds a
    copy of the bag. A scripted `StartTrainerBattle t, t` is the same (tools/battle_scripts.py
    "self" claims; Kaizo uses it).
  - Talked to (`Battles_Trainer`, scripts_battles.s:8-37): `StartTrainerBattle VAR_0x8004` has
    one operand, `enemyTrainer2` defaults to 0 (asm/macros/scrcmd.inc:1427-1430;
    `ScrCmd_StartTrainerBattle`, src/scrcmd_trainer.c:124-142); 0 != id gives
    `BATTLE_TYPE_TRAINER` (encounter.c:738-739), made a double only by the trainer header's
    battle type (`dto->battleType |= trdata.header.battleType`, trainer_data.c:50);
    `trainerIDs[3]` is 0, so `dto->trainer[3]` stays cleared (field_battle_data_transfer.c:74-77,
    `Trainer_Encounter` skips id 0, trainer_data.c:31-33): battler 3's bag is EMPTY, and
    `trainerItems[1]` stays empty.
  - Kaizo: the common script file scripts_battles is scr_seq.narc member 1114
    (res/field/scripts/scripts.order:1115), which the Kaizo patch leaves unchanged
    (docs/kaizo/rom_diff.md:63 lists the changed members; 1114 is not one), so both paths hold
    in Kaizo; Kaizo's scripted `StartTrainerBattle t, t` doubles always copy.
  The start path of a map-object double is the player's choice at run time, not split data.
  Engine:
  `trainer_items` [B, 4] and `trainer_item_count` [B] on the enemy battler (1 -> bag 0,
  3 -> bag 1; `b >> 1` is a bijection on enemy battlers), the player's rows empty, so singles
  keeps [2, 4] with the bag on row 1, byte-identical (a [2] array indexed by `b >> 1` would move
  the singles bag to row 0). `initial_arrays` fills battler 3 from `opp2_items`, which every
  four-battler fight must pass (`ValueError` otherwise; no default): TWO_TRAINERS / MULTI the
  second trainer's bag, DOUBLES `opp_items` when spotted or scripted `t, t`, `()` when talked to.
  RefBattler holds the bag.

- D2-1 `chosen_target` int8[B] (`battlerActions[b][1]`, battler index, initial `b ^ 1`, kept between
  turns and switches) is D2-1's field. D2-9 reads it for a locked battler (bcp.c:1885-1888,
  lib.c:1711); D2-13 writes it for a called move (script.c:2123); Mirror Move's write (:3517) goes
  with that effect's builder.
- D2-2 The four-battler `monSpeedOrder` recomputed before each action (bcp.c:888, lib.c:3364-3383)
  and its tie draws (lib.c:1405-1437) are D3's. D2-8 and D2-12 take the order as an input (B = 2:
  `jnp.arange(2)`, order-free; no new draw); their B = 4 tests pass an explicit order. Resuming an
  action after a U-turn / Pursuit pause with its defender at B = 4 is D3's.
- D2-3 The differential draws `targets` at B = 2 only; drawing the format moves to D3 / D6.
- D2-4 D2-13 builds GoToMoveScript only (the existing invoked moves). Its Mirror Move and Magic
  Coat doubles-targeting rules (script.c:3483-3525, :6487-6513; bcp.c:3059-3112) stay in the sheet
  for whoever builds those effects; coverage keeps refusing both.
- D2-5 Trace's draw is `Outcome.trace_pick` int8[B, 5], indexed `[holder, ts_trace_fires[holder]]`
  with the turn-scratch `ts_trace_fires` int8[B]: a slot traces at most once for its turn-start
  occupant and once per entry (own action, three Roar / Whirlwind drags) (lib.c:3712-3737,
  :7750-7781).
- D2-6 Weather Ball with no weather, fog or Cloud Nine / Air Lock: `CalcMoveType` leaves `type`
  uninitialized (lib.c:7894-7913, no else); the redirection check treats it as Normal and falls back
  to the data type (lib.c:1748-1750).
- D2-7 Bide's release in doubles joins D2-14: the defender is `lastHitByBattler[attacker]` when it
  has HP, else `BattleSystem_RandomOpponent` (one new coin, `Outcome.bide_target_draw` [B]), no
  target when both opponents are down (bcp.c:2702-2716; lastHitByBattler written on every damaging
  UpdateHP, partner and Substitute hits included, never cleared, bcp.c:3357).
- D2-8 The item command's random defender (bcp.c:1905-1908) is read by no state: no Outcome field,
  no engine change.
- D2-9 Follow Me is one per-side turn-scratch field `ts_follow_me` (1 + user battler, 0 = off;
  `followMe` / `followMeUser`, script.c:6261-6270, cleared lib.c:2251-2252).
- D2-10 (build of D2-1) The differential's `targets` are drawn inside `tests/test_differential._step_both`
  (after the caller's outcome draw, once per half), so `test_jax_matches_reference` and
  `tools/differential.check` need no edit. `step`, `step_events`, `turn_events`, `_action_phase`
  and `ref_step` take `targets=None`; at B = 4 `None` raises ValueError, at B = 2 it is zeros.
- D2-11 (build of D2-2 / D2-3) `ghost_user` is imported inside `actions.selection_range`
  (effects imports actions). The reference had no `Battler_IsTrapped`: `reference/engine._trapped`
  mirrors `actions.trapped`. `decision_pairs` raises ValueError at B = 2 and for MULTI (the
  player commands battler 0 alone there: `legal_choices(tables, state, 0)`).
- D2-12 (build of D2-4) [Superseded by D2-33: a randomized non-Ghost Curse takes the foe.] At B = 2 a randomized call (an Encore takeover) of a non-Ghost Curse keeps
  today's self defender: `targets.resolve` applies the Curse rule in branch 9 as in branch 10,
  whereas lib.c:1685-1707 would give the foe (singles `attacker ^ 1`). Joins pending ruling (i)
  for the follow-up commit after the D2 merge.
- D2-13 (build of D2-4) D2-9 is not in this batch, so `targets.fight_randomizes(state, side,
  struggling)` holds its randomize condition: Struggle or any Encore on the attacker at the action's
  start (bcp.c:1866-1884); the selected-move-not-in-slot case (:1889-1893) cannot occur (the move
  is read from the slot).
- D2-14 (build of D2-4) `play_hit` / `apply_move_effect` take `target_side` and `defender` as optional
  keywords (None: today's pair, the foe, and the user for a `self_defender` move) because
  test_k3b_abilities and test_small_moves call `apply_move_effect` directly; `_use_move` and
  Pursuit's hit always pass them. `primary_effect_fails` takes `defender` after `target_side`.
- D2-15 (build of D2-4) The reference's `_fight` reads `chosen` from `s.battlers[side].chosen_target`
  and the speed order from `_Turn.speed_order` (None: the identity); it mirrors the resolver for the
  defender, the PP defender and `_hit`'s target only, the other `_foe(side)` reads of `_fight` stay
  B = 2 (D3's four-actor turn). `_use_move` takes `speed_order=None` (identity).
- D2-16 (build of D2-4) The `fullset` marker is registered in pyproject.toml (CLAUDE.md names it; it
  did not exist). `tests/reference/test_targets_ref.py` carries it.
- D2-17 (build of D2-6, D2-11) The B = 4 step tests (Stealth Rock landing on side 1, a foe's move
  redirected to the Follow Me user) need a four-actor `step`: owed to D3. Until then the resolver
  tests (`tests/engine/test_targets.py`, `tests/engine/test_d2_follow_me.py`) are the tests.
- D2-18 (build of D2-4..D2-8, D2-11) The `ts_follow_me` write sits behind step.py's `script_on`
  (script ran, not failed to find a target) as Imprison's does; D2-11 and the resolver share
  step.py / reference/engine.py edits, so the engine code is one commit.
- D2-19 (build of D2-9) `targets.fight_randomizes` already held the sheet's rule (`struggling |
  encore_move != 0`, ruling D2-13); D2-9 changes no code, adds its tests, and the docstring cites the
  six branches. Branch 1.5 needs no code. Owed to D10: `learn_pin` (state.py, one scalar) must pin
  per player battler at B = 4 (a partner's KO pays exp while two player battlers hold pending move
  actions). Disobedience (rule 4) stays unbuilt: the engine has no obedience check (Kaizo always obeys).
- D2-20 (build of D2-10) `nt_before` / `charge_keeps_user` / the TryMove check read the resolver's
  `defender` of `move` (moveTemp). A called move (invoker, Nature Power) keeps today's "foe has HP or
  `defender_always`" test as `called_target` (D2-13 re-picks its defender); every other move's
  `has_target` is `~nt_before & ~nt_try`, with `last_of_multi = can_act & ~nt_before & (second_turn |
  sun_now | herb)`. `succeeded`, `_unlock` and the `no_target` event are unchanged. `resolved.defender`
  stays NO_DEFENDER for a charge turn with no foe (the user is the defender for the checks only), so
  `play_hit`, Pressure and `record` read what they read today. The reference's `_fight` mirrors it
  (`no_target0`, the HasNoTarget block, the charge strike's TryMove check) and keeps its other
  `_foe(side)` reads. The four-battler step cases of the sheet (both foes down: PP, lock, `last_move`,
  Choice lock; Spikes; Helping Hand; Fly with and without a Power Herb) are owed to D3 with the
  four-actor `step` (as D2-17); `tests/engine/test_d2_no_target.py` composes the resolver and the
  three functions at B = 4 instead.
- D2-21 (build of D2-12) `targets.redirect` returns the holder in place of the defender; `_use_move`
  writes it into `resolved.defender` / `target_side` (and `target`) after the PP block, so the move
  script and `play_hit` read the holder, Pressure's cost the fight command's defender. The reveal
  sits before the accuracy roll with `redirected & can_act & has_target & ~failed`
  (`hooks.REDIRECTED`); the reference reveals at the same point and also on a charge turn (the
  engine's cond holds there). The step-level reveal-not-on-failure test needs a four-actor `step`
  (owed to D3, as D2-17); `test_d2_redirection.py` tests the hook and `redirect` directly. A
  Normalize attacker (not an `AbilityId`) is owed to E1: it adds `| (_ability(attacker) ==
  AbilityId.NORMALIZE)` to `redirect`'s exits. `targets._ability` is the single place E3's
  suppressed-ability function replaces. `ids.MOVE_*` for `CalcMoveType`'s four moves were added.
- D2-22 (build of D2-12) Storm Drain: `AbilityId.STORM_DRAIN = 90`, `ABILITY_DECOMP_IDS` 114,
  `effect_map`, `INPUT_NEUTRAL_ABILITIES`, `tests/strategies.py`, and the `game_constants.py`
  "ability" table row (the guard `tests/data/test_game_constants.py` compares it). The vanilla and
  Kaizo table rebuild (`tools.build_tables`) was not run by this builder (ordered not to): the
  tables keep refusing Storm Drain's holders until the managing session rebuilds; the
  differential's Storm Drain draw is gated by the supported set after the rebuild.
- D2-23 (build of D2-13) [The B = 2 stored-target write is turned on by D2-33.] `targets.called_defender` returns `(defender, redirected)`; `_use_move` rebinds `resolved` through `_redefend` for an invoker that found a move or Nature Power, and its `has_target` / `script_runs` read `called_def != NO_DEFENDER`. The stored-target write (script.c:2123) is made at B = 4 only and only for a defender (a BATTLER_NONE write is not stored: the C stores 255, which nothing may read); at B = 2 no state reads it, and skipping it keeps singles states unchanged. Parts B and C (Mirror Move, Magic Coat) are not built; coverage still refuses them.
- D2-24 (build of D2-14) `last_hit_by` is written at B = 4 only (`effects.play_hit`, on every damaging UpdateHP, Substitute hits included, before the Substitute branch). At B = 2 it stays 0 so singles states keep merging; `targets.bide_defender` then takes the random-opponent arm, which gives the foe while alive, the value the sheet's B = 2 argument reaches. Counter's `counter_ok` tests the side (a partner's hit fails it); its defender is `targets.counter_defender`, and Pressure reads it. The STATE.md rows for `ts_follow_me` and `last_hit_by` read plain "mapped" (`test_state` accepts no other status; the follow-me row was failing it).
- D2-25 (build of D2-15) The sheet's `pressure_pp_cost` edit and the reference `_pp_cost` are built as written; step.py already passed the fight command's pre-redirection defender, so it changes nothing. The redirected-Electric case is a step-level test owed to D3 (ledger D2-17).
- D2-26 (build of D2-16) `hooks.trace` takes `coin` and a `count` flag; `ts_trace_fires` is added to the fire only inside `step` (an `Outcome.trace_pick` of rank 2 passed down through `_send_out`, `_act`, `_use_move`, `replace`) and at B = 4 only. Outside a step (`battle_start`, `show_battle_mon`, `replace`) a `trace_pick` int8[B] (default zeros) gives one coin per slot. Intimidate's extra skip (SELF_TURN_FLAG_SUBSTITUTE_HIT, script.c:9180-9181) is built since E1 F-4 (amended SF2 (b)): the flag is `ts_sub_hit` int8[B], written at the paused or pre-Pursuit stop and read by the resumed U-turn's `switch_in_effects(..., sub_hit=)`, cleared on resume. `move_end_switch_in` takes no `sub_hit`, as no case reaches it in Gen 4: MoveEnd's pass (bcp.c:3973) does run before the clear (BattleContext_Init, bcp.c:3998; ClearFlags at :3986 leaves selfTurnFlags), but only a hit on a Substitute in UpdateHP sets the flag (bcp.c:3359-3371, its one setter; Pursuit on a switcher breaks the Substitute by script without it, subscript_pursuit.s:_108-_142), and in such an action no Intimidate is still unannounced at MoveEnd: intimidateAnnounced clears only at a switch-in (battle_lib.c:104) and in Transform (battle_script.c:5351); the ability writers are Trace, Role Play, Skill Swap, Worry Seed and the Shaymin / Giratina form changes (subscript_trace.s:7, subscript_copy_ability.s:15, subscript_exchange_abilities.s:23-24, subscript_give_target_insomnia.s:11, subscript_check_shaymin_form.s:23, subscript_giratina_form_change.s:16), and Transform, Role Play, Skill Swap and Worry Seed are CLASS_STATUS (res/moves/<move>/data.json); the one switch-in inside a damaging action, U-turn's newcomer, meets every battler's switch-in states (an armed Trace included) at AfterMoveEffects' TRIGGER_SWITCH_IN_EFFECTS (bcp.c:3672-3682) first, with the flag; Gen 4 has no damaging drag, Eject Button or Red Card. Owed to D3 (four-actor step): that AfterMoveEffects pass over all four battlers with the per-battler flag (`switch_in_effects` pairs the newcomer with one foe). Sheet build note 9 (`outcome._armed_trace` / `pivot_view` using `trace_target`) is not built: those pair one side with its one foe and need the four-battler entry view (D3). playout.py's calls keep today's B = 2 arguments (`download(..., (1,))` equals `opposing`). `trace_target` tests cannot give a battler an untraceable raw ability by writing `state.ability` (the check reads the species row, `damage.decomp_abilities`), so the Forecast / Trace / Multitype skip is covered by the existing singles tests; B = 4 tests cover HP and the coin.
- D2-27 (build of D2-13, D2-14, D2-16) Outcome gets `call_target_draw` (key 45), `bide_target_draw` (46), `trace_pick` int8[B, TRACE_FIRES = 5] (47); enumerate relevance is B = 4 only and over-approximate (any Trace in the battle reads all five fires), law `_coin`.
- D2-28 (D2 review fix, finding 1) [The B = 2 write is turned on by D2-33.] The selection's `chosen_target` write (D2-1, bcp.c:548, :4363-4367) is made at B = 4 only, in both engines (`step._action_phase`, reference `ref_step`), as D2-23 and D2-24 do: at B = 2 no state reads it (`targets.resolve` and `called_defender` ignore `chosen` there), and writing it split singles states that were equal (an asleep Tackle and Swords Dance gave 1 and 0), so `search/rows.row_bytes` stopped merging them. `actions.chosen_targets` / `_chosen_targets` keep the B = 2 rule; the D2-1 tests test the rule directly and assert the step writes nothing at B = 2. Singles follow-up (ii) (an Encore ending under a Rollout lock reads the stored target) must turn the B = 2 write on when it lands.
- D2-29 (D2 review fix, finding 2) `targets.redirect` runs at B = 4 only: in `_use_move` (the fight command's check, the merge of the called move's flag and the TryMove reveal, all under a static `fmt.B == 4`), in `targets.called_defender`, and in the reference `_fight` / `called_defender`. At B = 2 it can never fire (the holder must be a living Pokemon other than the attacker, the foe, which is then already the defender), so the singles step no longer traces `redirect_type` (Hidden Power, Weather Ball, Natural Gift, Judgment types) every action.
- D2-30 (D2 review, finding 3) Owed to D6 / D11: at B = 4 Trace outside a step, at battle start (`state.py` `make_state` -> `hooks.battle_start`) and on faint replacement (`playout` -> `replace` / `show_battle_mon`), never receives a `trace_pick`, so a Trace lead facing two eligible foes always copies the same one (ChooseTraceTarget, battle_lib.c:7750-7781, picks one by the random draw). The entry points that create those draws (D6, D11) pass it.
- D2-31 (D2 review, finding 5) Owed to D3 step tests: the redirect message's failure gate at B = 4 (`step._use_move` reveal `~failed` vs the reference reveal in `_fight`; the C's TryMove skips the Lightning Rod / Storm Drain message under MOVE_STATUS_NO_EFFECTS, battle_lib.c:1796, :1806) is unreachable until D3's four-actor step; D3 tests it there (with D2-17 and D2-21).
- D2-32 (D2 gate review fix, finding 2) [The B = 2 write is turned on by D2-33.] Follow Me's `ts_follow_me` write (D2-11, script.c:6261-6270) is made at B = 4 only, in both engines (`step._use_move`, reference `_fight`), as D2-28: at B = 2 nothing reads it (`targets.resolve` and redirection are B = 4 only; `counter_defender` gives the foe either way) and, kept over a U-turn / Baton Pass pause, it split equal singles states. The D2-11 singles step tests assert it stays clear; the B = 4 step write is tested with D3's four-actor step (with D2-31). Singles follow-up (i) turns the B = 2 write on. Finding 1: `test_download_sums_both_foes` pins the stat raised in two cases whose sum disagrees with one foe alone (battle_lib.c:3815-3851).
- Owed 14 closed (snap_compare follow-up; 9b3ec99, review fixes 6502a00, gate notes on
  e2-snap-notes): M1 capture.json records `content_hash` (`snap_compare.content_hash`: the
  running script, src/nuzline, tools, the test modules the child imports, the golden trace and
  every built file a part reads, each by content: a .npz by its arrays, a .json by its parsed
  value) and its `format`; compare fails on equal hashes unless --same-ok and prints both; no
  git: `dirty: null`, printed "not checkable"; a capture.json without `content_hash` / `dirty`
  or of another format is refused with "re-capture". N1 the final line names the groups (Roark
  parts in full) and the seed count, "seeds: none" without a seed part. N2/R5 a part with zero
  steps or (but "gen") zero calls fails. N3 --raise-warn 0.5 / --raise-fail 0.95, a usage error
  unless 0 <= warn <= fail < 1 (local runs peak at 0.33, 0-0.rest). N4 saved Outcome and
  action shape/dtype checks (`check_layout`, `check_actions`). N5 Python / np.float64 scalars
  compare by IEEE bytes, as arrays. N6 a raise against a value or structure is a divergence.
  N7 the dirty check covers the test modules the child imports. N8 tests in
  tests/tools/test_snap_compare.py. N9 ledger line numbers updated. R1/R2 docstring wording.
  R3 tests/test_pinned_cases.py (shares `_learns_match_reference` with the slow differential
  test). R4 `test_four_battler_fight_needs_battler_3s_bag`.
- D2-33 (singles follow-up: the three approved decomp fixes; own branch e2-singles-fix off aaa65a3; rule sheet _scratch/singles_fix_sheet.md) Both engines, one commit. The stream lead decided: no canonical `chosen_target` in `search/rows.row_bytes` (row-merge counts may change; the engines canonicalize it at the turn's end since D2-35), `ts_follow_me` written whenever the game writes it, `tests/strategies.py` untouched, the `_move_end` status-recovery difference left (Owed 15).
  - (i) Follow Me sends a non-Ghost Curse to its living user (lib.c:1714-1717, not gated on doubles): `targets.resolve` (reference `_resolve`) branch 10 at B = 2 is the living Follow Me user, else the stored target while it has HP, else the foe (:1718-1725). `ts_follow_me` is written at B = 2 (`step._use_move`, reference `_fight`; the reference also sets the state's side field at once, cleared at the turn's end like the engine's). The Curse then costs 2 PP against a Pressure Follow Me user (bcp.c:2261-2266), is tested like any move aimed at the foe (`aims_foe` gains `curse_at_foe`: a semi-invulnerable foe, bcp.c:3034-3041; Lock-On / No Guard, :3016-3022), and records itself on the foe (`move_hit`, `choice_lock` of a non-Choice holder, bcp.c:4488-4503); an invoker's `record` is now `called_def != side` (the called move's defender).
  - (iii) Branch 9 at B = 2 gives the living foe, else BATTLER_NONE (lib.c:1706-1707): a randomized non-Ghost Curse (Encore takeover, Metronome / Sleep Talk / Assist / Copycat landing on it) hits the foe like (i), and with no living foe runs subscript_no_target (PP 1, no stat change).
  - (ii) The selection's `chosen_target` write (bcp.c:534-542, :4363-4367) and GoToMoveScript's (script.c:2123) are on at B = 2 (`step._action_phase`, `_use_move`; reference `ref_step`, `_call_defender`): a RANGE_MY_SIDE or non-Ghost Curse selection stores the user, a called move stores its defender. A locked battler (Rollout, Ice Ball, a charge move's strike) not encored reads it (lib.c:1709-1719), so after an Encore that ended inside the lock the defender is the user. `resolve` returns `target_side` = the defender also when it is the user for a move that aims at it (a `self_defender` move keeps the foe). `step._use_move` then reads the user as the defender: its own evasion and accuracy stages, abilities and items (`live_hit_threshold(..., target=)`, `damage_for_roll(..., def_side=)`, `effects.play_hit`), no Protect / Endure / Roost of the defender, `Battler_MovedThisTurn(defender)` false (no Zoom Lens, flinch not blocked, no Quick Claw / Custap flag), Volt Absorb not (attacker != defender, lib.c:3500-3502) but Water Absorb / Dry Skin yes (:3507-3512, :3538-3544), the defender's events at the user (`ctl["target"]`, `_hit_events`); the charge strike clears the user's semi-invulnerable bit before the accuracy test (effect_script_0155.s:24, already so). The reference reads the target through `tgt` in `_fight` and `target` arguments of its helpers.
  - Enumeration (outcome.self_lock: locked, not encored, stored target the side itself, a RANGE_SINGLE_TARGET locked move): such a side may hit the user or, after a faster Encore or Follow Me, the foe, so its accuracy roll is kept whole (`OutcomeLaws.hit_whole`; the sampler draws the raw roll), its crit is not cut by the foe's armor, and the foe-keyed cuts are lifted for it (damage classes, sure flinch, Protect gate, status / confusion eligibility, order gate of a flinch, King's Rock, contact and Focus Band reads on its own side): over-approximate, sound.
  - Not modelled (no engine field): `conversion2*` and `moveCopied` writes on the defender (bcp.c:4488-4525); a Bide release's `lastHitByBattler` stays B = 4 only. The batch 9 note "Follow Me changes no defender" at `enumerate.py` is replaced.
- D2-34 (D2-33 review fixes) 1. Lucky Chant at the hit: CalcCrit passes `BATTLECTX_SIDE_CONDITIONS_MASK` of `battleCtx->defender` (battle_script.c:2158-2163) and `BattleSystem_CalcCriticalMulti` blocks under Lucky Chant (battle_lib.c:7137). The turn-start cut (`no_crit`, the foe's side) is lifted for a self lock, so step now reads it on the hit's own defender: `effects.crit_blocked` (armor or Lucky Chant on `side_of(target_side)`) at every crit site (`play_hit`, the hit loop's events, the crash, Pursuit), reference `_crit_blocked` likewise. For every other move the cut already zeroes the crit on the same side (Lucky Chant changes only at the turn's end; the move is unsupported), so the read is a no-op there. 2. The Ghost Curse re-pick (effect_script_0109.s:11-12) is logged under "Closes with its mechanic" with Color Change, its only source. 3. Queue 3b stays open for the snap_compare diff (pod, base aaa65a3); Owed 15 moved into the Owed list, Owed 16 (strategies), 17 (row-split cost) added. 4. Audit of the hit path for a defender that is the user: `play_hit` and its helpers read `target_side` (stat, status, flinch, Fang, King's Rock, Pluck, Knock Off, Embargo, contact, Focus Band / Sash, Substitute), step's hit loop reads `target_side` / `acc_target` (Synchronize, the defender's items, Sticky Barb, Jaboca / Rowap / Enigma, thaw, Destiny Bond, `move_hit`); `recovers_at_full_hp` reads the foe for Present only (not a lock move); the defender's Choice-lock clear on the user (bcp.c:4488-4496) is overwritten by UpdateMoveBuffers' own (bcp.c:3921-3934), and its `moveHit` by ClearFlags at its MoveEnd (:4389); `last_hit_by` stays B = 4 only, as its one reader (Bide's release, bcp.c:2700-2705) needs storedDamage, zeroed by subscript_bide_start.s and added only by the foe's hits during Bide (bcp.c:3357, :3407). No other foe-hardwired read found. Nits: the sampler's `lock` is gated by a move action as the law's; the crash damage passes its defender (`def_side=` / `defender=`); the batch 9 Curse case has distinct levels.
- D2-35 (D2-33 gate review fixes; the manager's ruling, no waiver) 1. The defender code of D2-33 / D2-34 passes the resolved defender (`targets.resolve`'s `target_side`; reference `turn.target_side`) instead of "the user or `_foe(side)`": `play_hit`'s `def_side`, `step._hit_events`, `acc_target`, the defender's events (`ctl["target"]`), reference `_hit`'s `tg` and `_fight`'s `tgt`. At B = 2 `target_side` is always the user or the foe, so singles do not change; at B = 4 one hit no longer mixes two defenders. 2. Canonical `chosen_target`: at the end of a turn at B = 2 (`step.canonical_target` in `step_events`, reference `_end_turn`) a battler that is neither locked (`locked_move`) nor encored (`encore_move`) holds `b ^ 1`. Its one B = 2 reader is `BattleSystem_Defender`'s branch 10 on an action that does not randomize (lib.c:1709-1719): a locked battler picks no command (bcp.c:334-338, :1885-1888), an encored one takes its slot without the menu (bcp.c:442-453) and can lock under the Encore and read the value once it ends; every other fight command overwrites it at selection (bcp.c:534-542, :4363-4367) first; lib.c:1678 and :6423 read it only in doubles (the decomp has no other read of BATTLE_ACTION_CHOOSE_TARGET; its other writes: the disobedience pick at bcp.c:2189-2191 and Mirror Move at battle_script.c:3517, unsupported; BtlCmd_GoToMoveScript's write of the called move's defender at battle_script.c:2123, made at B = 4 only by D2-23 and at B = 2 since D2-33, and canonicalized at the turn's end like the selection's). A further reader found: a turn paused at a U-turn / Baton Pass party list resumes with the actions still to run, which read this turn's selection (a pending non-Ghost Curse reads its stored user, branch 10), so a paused step does not canonicalize (the resumed step does, at its end). Tests: `test_singles_followup` (equal rows after a switch with stored user / foe; kept under a lock and under Encore); `test_d2_action_encoding` step tests assert the canonical value. Owed 17 narrowed to locked / encored battlers and paused states; Owed 18 (mass checks of the self-lock cuts) added. Notes: the Fly strike cite is effect_script_0155.s:24 (was 0156, Defense Curl); snap_compare's final line reads "seeds: 0-24 (25)" like "seeds: none".
- D2-36 (merge of main 7c635e2 and 38d22f4, E3 batch 1, into D2; numbered D2-33 on e2-d2, whose D2-33 to D2-35 are the singles follow-up's here) Resolutions. (1) `hooks.trace` keeps D2's `trace_target` pick (`coin`, `count`) and sets E3's `trace_announced` where it copied (battle_lib.c:3727); `trace_copy` (the eligibility both slots share) reads the holder through `battler_ability` (lib.c:3088-3093, Battler_Ability at :3720-3725) and the target's raw ability (:7754-7776). (2) `switch_in_effects`, `show_battle_mon` and E3's `move_end_switch_in` gate Trace by `trace_announced == 0` (:3726) and take its coin from `trace_coin`; `move_end_switch_in` (bcp.c:3973) gets `trace_pick` (`step._move_end` passes `Outcome.trace_pick`), Intimidate over every opposing battler (subscript_intimidate.s:10-23) and Download's sums over `targets.opposing` (lib.c:3814-3851). (3) `_intimidate_one` and the reference `_intimidate` set `intimidate_announced` for a living Intimidate holder whatever the target (:3799-3803); the reference `_download` sets `download_announced` before its sums over `opposing` (:3836). (4) Every D2 ability read goes through the shared reader: `targets._ability` is `state.battler_ability` (redirection, lib.c:1737-1738, :1756-1778); in the reference `_pp_cost` and Counter's Pressure (BattleSystem_CountAbility :2870-2967, DecPPForPressure :7390-7397), `trace_target`, `redirect`, `_trapped` (BattlerIsGrounded :5502, CountAbilityTheirSide :5509-5532) read `_ability`. The reveal of a redirect holder keeps the raw ability (BattleAI_SetAbility, script.c:11805). (5) `TRACE_FIRES` = 5 still bounds a slot's fires: the armed turn-start Trace plus one per action, a Skill Swap that hands Trace being one action. (6) Owed to D3 (E3 logic, not changed by the merge): Me First's copy reads `_foe(side)`'s chosen move (`step._use_move` through `pursuit`, `enumerate.used_moves` / `_side_call_law`, reference `_fight`); at B = 4 the decomp copies the defender's (BtlCmd_TryMeFirst, battle_script.c:7286-7312), which needs the four-actor step's per-battler actions.
- D2-37 (merge of e2-snap-notes, carrying main 38d22f4, into the singles follow-up) The follow-up's defender-based reads keep E3's shared reader (Battler_Ability, battle_lib.c:3088-3093): `step._use_move`'s No Guard on the user and the resolved defender (bcp.c:3016) and Crash's Magic Guard read `battler_ability`; the reference `_fight` No Guard and `_contact_ability` (its `target` argument) read `_ability`. E3's Me First rule (a failed Me First keeps a target, RANGE_SINGLE_TARGET_ME_FIRST, battle_lib.c:1599-1727: `self_invoker`) stands beside the follow-up's lock-target checks.
- D2-38 (numbered D2-34 on e2-d2; pod run on e2-d2 e688271: 2 of 3,570 failed; both tests were stale, the code is right) 1. `test_faints.py::test_new_fields_default_to_zero` (renamed `test_new_fields_start_values`) asserted every field after `replace_mul` is 0; D2-1's commit d061cd2 (not the merge) appended `chosen_target`, whose spec start is `b ^ 1` (D2-1, STATE.md; `make_state` state.py:1128): the game's 0 is never read, as every battler selects on turn 1 (bcp.c:507-531) and the selection writes the target (TARGET_SELECT_INIT passes `&battlerActions[i][1]` to BattleControllerPlayer_MustSelectTarget, bcp.c:534-542, which writes the user or `b ^ 1` at :4363-4367), and at B = 2 the step does not write it (D2-28), so a fresh singles state holds [1, 0]. The test now checks the turn scratch (`pivot_battler`, `ts_*`) and `last_hit_by` at 0 and `chosen_target` at `b ^ 1`; here D2-33 turns the B = 2 write on and D2-35 canonicalizes the value to the same `b ^ 1` at a turn's end, so the fresh state and the test are unchanged. 2. `test_field_pass.py::test_trick_room_is_not_counted_in_the_field_pass` assumed the countdown unbuilt. CheckFieldConditions (bcp.c:932-1254) has no Trick Room step; CheckSideConditions runs after it and its SIDE_COND_CHECK_STATE_TRICK_ROOM (bcp.c:1740-1742, :1822-1830) subtracts 1 << FIELD_CONDITION_TRICK_ROOM_SHIFT each turn end (condition.h:123-149), running subscript_trick_room_end at 0. E3's `step._trick_room_check` (after `_perish_song_check`) does that, so one turn takes 3 to 2. The test is now `test_trick_room_counts_down_once_per_turn` (3 to 2). No engine change.
- D2-39 (D2 snap_compare, BASE a838c20 vs NEW e07c0bd; log _scratch/rent/snapd2/compare.log in the main checkout) Accepted: the 20 "gen" parts DIVERGED at turn 1, all 102 engine parts (step, jitstep, sample, rest, ai, roark.trace, roark.rank) are identical. Cause, generator only: "gen" (tools/snap_compare.py:572-584, `draw_inputs` :507-565) redraws the Hypothesis inputs at NEW and diffs them against the captured ones (result check :1111); D2 changed `tests/strategies.py` (`any_outcome` draws four new fields, `target_draw`, `call_target_draw`, `bide_target_draw`, `trace_pick`; `BATCH3_ABILITIES` gained Storm Drain), so one `st.data()` stream is consumed differently and every later draw shifts. Turn 0 matches because Hypothesis's first example is the all-zero one. Every engine part reruns on the captured inputs, so an engine difference would show there; none does. Limit: engine parts see only captured inputs, so the new Outcome fields stay at `empty_outcome()`'s value there; D2's non-default draws are covered by its own tests (test_d2_*, test_outcome.py). Proposal, no tool change: "gen" reports "generator changed" (not a failure) when `tests/strategies.py` or the generator's imports differ between the two commits, and fails only on a divergence with an unchanged generator.
- SF2-f Acid Rain interim guard. A Pursuit KO of the switcher runs `fieldConditionsMask -= faintedMon`
  (subscript_pursuit.s:100, :115; OPCODE_SUB_TO_ZERO, script.c:3282-3287). Built: `BattleState.acid_rain`
  (int8, sticky, not a scratch field; reference `RefState.acid_rain`) written at the one Pursuit site of
  `step._execute_actions` and at the end of reference `_pursuit`: `ko` = the switcher alive before and at 0
  HP after the hit; `bonded` = `pre.destiny_bond[switcher]` (the game's order, Destiny Bond at :85 restores
  `faintedMon`, subscript_faint_check_destiny_bond.s:10, :25; not `_pursuit_hit`'s `bonded`); operand = the
  pursuer when it fell to Rough Skin, Aftermath, Jaboca or its Life Orb (lib.c:4257-4269, :4358-4371,
  :5400-5409, :7357-7374) and not `bonded`, else the switcher; written only when the operand is nonzero
  and a weather, Gravity, Trick Room or another battler's Uproar bit is up (the switcher's bit is gone at
  :115, lib.c:2213; a Destiny Bond faint clears the pursuer's too, a recoil faint keeps it). Refusal:
  `coverage.check_acid_rain` at `enumerate_turn` and `step.root_belief` (so stage A / B and prep through
  `solve`); `search.learn.learn_reason` carries the message for the playout / replay refusal reasons; the
  jitted readers of `exp_refused` also refuse `acid_rain != 0` (search/expand.py, learn.py, playout.py,
  player.py). Coverage: `CoverageReport.runtime` (non-refusing, `level` unchanged), the line "field
  condition: Acid Rain (runtime refusal)" when a trainer move is BATTLE_EFFECT_HIT_BEFORE_SWITCH
  (`check_opponent`); backlog kind "field condition", section "Playable, lines hitting the glitch refused".
  Tests: tests/engine/test_acid_rain_guard.py (`fullset`), tests/tools/test_backlog.py.
- SF2-h Destiny Bond before the contact ability in Pursuit's KO path. subscript_pursuit.s runs FAINT_CHECK_DESTINY_BOND (:85) before TriggerAbilityOnHit (:86-87), TriggerHeldItemOnHit (:90-91) and the pivot items (:94-96); it faints the pursuer when the KO'd switcher holds VOLATILE_CONDITION_DESTINY_BOND and the pursuer has HP (subscript_faint_check_destiny_bond.s:5-7, :11-16). Every contact ability then fails on the attacker's HP (Static lib.c:4215, Rough Skin :4258, Effect Spore :4274, Poison Point :4304, Flame Body :4322, Cute Charm :4340, Aftermath :4363; Color Change needs the defender's HP, :4243): no message, no reveal, no draw. A FIGHT's Destiny Bond runs after the contact ability (bcp.c:3517): unchanged. Built: `effects.play_hit` (`pursuit_switch` only) applies the take-down right after the HP write, before `_contact_ability`, and returns `MoveHit.bonded`; `step._pursuit_hit`'s own Destiny Bond step is gone; reference `_pursuit` runs it before `_contact_ability`. (f)'s Acid Rain operand keeps reading `pre.destiny_bond`. Engine and reference; laws unchanged (superset: the switcher's `contact_proc[d, 0]` stays a relevant draw). Tests: tests/engine/test_singles_followup2.py (`fullset`).
- SF2-c The paused view counts a pursuer as acted. TryPursuit sets the pursuer's PICK_COMMAND to MOVE_END (battle_script.c:6931), so it has moved (Battler_MovedThisTurn, battle_lib.c:5365-5368) and never acts again this turn; U-turn's Pursuit runs before the list (subscript_attack_then_switch_out.s:54, the list at :71). Built: `outcome.moved_mask` (`ts_acted` | `ts_pursued` bits; at B = 2 the stored `1 + pursuer` is the bitmask) read by `pivot_view` (the pursuer's view action is the move-less switch-to-self) and `enumerate._paused_rock` (target moved). Laws only; resumed outcome distribution unchanged.
- SF2-snap (queue 3b: the singles-fix snap_compare, BASE 0fc4dfa vs NEW 6c49088; logs in the main checkout: _scratch/rent/snapsf/compare.log, per segment _scratch/rent/segcheck/segcheck.log) Accepted. Exit 1: 108 parts identical, 14 DIVERGED, all `chosen_target` in "step" and "jitstep" (ref_step and jit_step agree), 7 segments, each at segment 0 of its turn. Cause: D2-33 (ii) turns the B = 2 selection write on, so a self-target pick (selection range 5, MY_SIDE / USER) stores the user (bcp.c:534-542, :4363-4367) where BASE kept `b ^ 1` (D2-28); D2-35's turn-end canonicalization (`b ^ 1`) skips a locked battler, an encored one and a paused step, so the user's index survives. segcheck (NEW ed20968; its src is 6c49088's) re-drew the BASE inputs of six chunks and printed EXPECTED for each: seed 23 turn 2, battler 0 Substitute, battler 1's U-turn paused the step (`pivot_battler` 2, the other battler), 1 -> 0; seed 33 turn 4 and seed 185 turn 4, battler 0 Bide (`locked_move` 117), 1 -> 0; seed 48 turn 1 and seed 124 turn 1, battler 1 Baton Pass (`pivot_battler` 2, its own pause), 0 -> 1; seed 154 turn 3, battler 0 Baton Pass (`pivot_battler` 1), 1 -> 0. Seed 103 turn 1 (chunk 100-109, battler 0, 1 -> 0) is not in segcheck.log (its run covered six chunks); the manager's reading is a Baton Pass pause, the same field and direction as seed 154. No engine change.
- SF2-roost Roost's chart reads the resolved defender. subscript_roost.s sets TURN_FLAG_ROOSTING on the attacker after a heal that worked (:10-12; at full HP, MOVE_STATUS_NO_MORE_WORK skips it, :11), and its one reader is BasicTypeMulApplies, which tests `turnFlags[defender].roosting` and drops the chart's Flying rows (battle_lib.c:2540-2543), called only by BattleSystem_ApplyTypeChart (:2626, :2636); the move's chart passes battleCtx->defender in CheckTypeChart (bcp.c:2350-2370, the call at :2358-2365) and in the damage scripts' BtlCmd_ApplyTypeEffectiveness (battle_script.c:6963-6977). No other code reads the flag (battle_script.c:7042 and :7102 are the script's compare and set). The type change is never written to the defender's types. The engine was right: `foe_roosting` (step.py, `read["roosted"][_foe(side)]`) is cleared for a self target (`& ~self_target`) and applied to the resolved `target_side` before the chart; at B = 2 the defender is the user or the foe, and the user never roosts before its own action's chart, so this is `turnFlags[defender].roosting`. The reference was wrong: it stripped Flying from `_active(s, _foe(side))` before `_fight` whatever the defender. Its damage chart already read the resolved `tg` (D2-35), so the strip showed only through the noEffect check, which also read the foe (`_matchup_x40` on `_foe(side)`); both feed only `move_failed` (the Uproar check) and `met_immune`, so no singles state changed. Fixed in the reference: `_strip_roost` strips the resolved defender (`turn.target_side`) after BattleSystem_Defender and after a redirection or Counter re-pick, and `_unstrip_roost` restores it at the action's end; the noEffect check reads `turn.target_side`. No engine change. D3-5's per-defender read at B = 4 replaces `_foe(side)` in step's `foe_roosting` with each defender's flag; the reference helper already takes the defender. Test: tests/engine/test_singles_followup2.py::test_dig_strike_at_its_user_after_a_foe_roost_reads_the_users_types (a Flying user stays immune to its own Dig, a Ground user takes it, after the foe's Roost).
- SF2-g Custap Berry's display. BEFORE_MOVE_STATE_QUICK_CLAW (bcp.c:3140-3143) loads subs/subscript_check_quick_claw.s for every battler that reaches its move start (LoadQuickClawCheck bcp.c:2823-2831). A Custap flag jumps :5 -> :12-13: the held-item animation always; the message "{0}'s {1} let it move first!" (:16-19) only when `waitingBattlers != 1` (:15; counted at BranchActions, bcp.c:881-886, the battler included). Quick Claw (:6-8): the animation only when `waitingBattlers != 1`, no message. Built: the `quick_claw` event is a bit field (bit 0 the animation, bit 1 the message; 0-3): `step._use_move`'s own `quick_claw` = `ready & custap[side]` (the removal's term) gives 1, plus 2 when the foe has not acted (`~target_moved`; the B = 2 "not last" fact, D3-3 item 7 replaces it by the count); `_quick_claw_shown` unchanged (Quick Claw bit 0); `_action_phase` ORs them. observe.py docstring: values 0-3. Event only; no state field changes; the reference has no events. Tests: tests/engine/test_singles_followup2.py (`fullset`).
- SF2-a MoveEnd cures the attacker and the defender before the battle-over check (FIGHT case). BattleControllerPlayer_MoveEnd runs RecoverStatusByAbility for the attacker, then the defender unless BATTLER_NONE, before AnyExpPayout and CheckBattleOver (bcp.c:3965-3969); the cure is UpdateMonData (subs/subscript_ability_forbids_status.s:13, :17, :22, :26), which ends in BattleMon_CopyToParty (battle_script.c:3077-3078), so it reaches the party. RecoverStatusByAbility reads only the ability and status (battle_lib.c:4378-4452). `step._move_end(..., defender, fought)` and reference `_finish_action`: the attacker's cure under `ran` / always; when the battle ends here the FIGHT's defender (`ctl["defender"]`, reference `turn.defender`: battleCtx->defender, the user for a self-range move, lib.c:1659-1664, or after a confusion self-hit, bcp.c:2629; none for BATTLER_NONE) is cured and no other battler (the FORBIDDEN_STATUS state, lib.c:4135-4148, is after the return at bcp.c:3969); while the battle goes on the foe call stands for TriggerEffectOnSwitch's pass over every battler (bcp.c:3973; Owed 15, closed by the manager's word in SF2 (a2)). `move_hit[side] = 0` stays under over == 0 (ClearFlags, bcp.c:3986). The switch-MoveEnd case (the newcomer cured before the battle-over check) is SF2-a2. The fix pass: `ctl["target"]` maps a self defender to the foe, so the engine now passes `ctl["defender"]` (`resolved.defender`, the user after a self-hit), and the reference sets `turn.defender = side` on a self-hit. Owed 20 closed: `_faint_check`'s battle-end `_restore_ability` (the party's ability) ran before the cures, which then read Limber back after Worry Seed / Skill Swap; both engines now run the cures on the battle abilities (`acted.ability`; reference: the abilities before `_faint_check`) and put the restored ones back after. Immunity's cure (battle_lib.c:4383-4388) is not built (ability: Immunity, mechanics_backlog.md), so the sheet's Immunity test case goes with that group. Tests: tests/engine/test_singles_followup2.py `test_battle_ending_*`, `test_move_end_cure_*`.
- SF2-a2 The switch / U-turn goes on after a Pursuit that empties a side (replaces SF2-a's switch-MoveEnd case; Owed 15 closes here: FIGHT case SF2-a, switch case this entry; the sheet _scratch/e2/d3_sheets/SF2-a2.md). R1 Nothing between the Pursuit and the action's MoveEnd checks for a battle end: a Pursuit-chain faint is only flagged (TryFaintMon, battle_script.c:1769-1772, from subs/subscript_update_hp.s:17) and played later (PlayFaintAnimation, :1794-1803) at subscript_switch_pokemon.s:44-48, subscript_show_party_list.s:23-24, LOOP_FAINTED (bcp.c:3820-3823) or CHECK_ANY_FAINTED (:4002-4008); CheckBattleOver runs at MoveEnd only (:3969), after RecoverStatusByAbility (:3965-3967) and AnyExpPayout (:3968). R2 Switch: SwitchCommand sets `attacker` to the newcomer (bcp.c:1981-1982, commandNext CHECK_ANY_FAINTED :1984 -> MOVE_END :4008); subscript_switch_pokemon.s: Natural Cure (:5-6), the Pursuit (:11), the recall only with HP (:12-20), SwitchAndUpdateMon either way (:22), PokemonSendOut (:39; exp participation only while the enemy battler has HP, battle_script.c:621-623, battle_lib.c:1461-1470), the hazards (:43), the faint (:44-48); MoveEnd cures the NEWCOMER (the attacker), pays exp, then CheckBattleOver returns at :3969, so TriggerEffectOnSwitch (:3973; lib.c:3622-3640) never runs for the newcomer of a decided battle: no switch-in ability, no status berry, no Intimidate. R3 Exp order: subscript_pursuit.s's KO path (FAINT_CHECK_DESTINY_BOND :85, contact ability and items :86-96, payout loop :100-113) pays the switcher's KO and a Destiny-Bond pursuer before the switch; a pursuer felled by a recoil (Rough Skin lib.c:4257-4269, Aftermath :4358-4371, Jaboca :5400-5409 / :7368-7374, Life Orb :7357-7365) is only flagged and AnyExpPayout pays it at the switch's MoveEnd (bcp.c:4449-4469), after the newcomer's hazards, whether or not the battle ends; CalcExpGain counts the mask slots with HP at that moment (battle_script.c:2196-2238, :2204-2206), so a hazard-KO'd newcomer gets nothing and is not in the divisor, and level-ups come after the hazards. R4 U-turn (subs/subscript_attack_then_switch_out.s): after the Pursuit (:54) only the user's HP is tested (:55); a survivor goes to the list (:71) with the pursuer's side out; the list runs SwitchAndUpdateMon (:9), PokemonSendOut (:18), the hazards (:22), FAINT_MON (:23-24); TRIGGER_SWITCH_IN_EFFECTS (bcp.c:3672-3682) has no battle-over test, so the U-turn's newcomer does run its switch-in effects, then the item checks (:3684-3708) and MoveEnd; a U-turn user cleared its Destiny Bond (bcp.c:2425), so its pursuer's faint is always paid at MoveEnd. R5 Both sides out (switcher KO'd by a Destiny-Bond Pursuit, newcomer killed by the hazards) = WIN | LOSE = a draw (bcp.c:4255-4260) = a loss (:4015-4016; ruling 2). Built: the Pursuit site's faint check has `decide=False`; the payout there skips a recoil-felled pursuer (`owed`: `fire & pursuer fell & ~bonded_ko`, the resumed U-turn's from `ts_pursued` and HP 0), which the action's MoveEnd pays (`base | owed`); `ran` reads the HP before the Pursuit hit; `_act`'s switch runs `_send_out(effects=False)` and `switch_in_effects` only while no side is out (`decided`); the list's faint check has `decide=False`; the U-turn's `pursuit_pause` now pauses at the end of a battle. Reference: `_pursuit` returns whether Destiny Bond fired, `_switch`, `_u_turn` and `_resume` split `owed` the same way and `_switch` skips its switch-in effects for a decided battle. No new state field. Tests: tests/engine/test_singles_followup2.py (the a2 block, nine cases). B = 4 (D3-9): decide=False at every Pursuit pass; paid = the switcher's KO and Destiny-Bond pursuers, owed = each recoil-felled pursuer of party 1 paid at MoveEnd lowest battler first (bcp.c:4452-4458); `ran` from the pre-Pursuit HP; the `decided` gate; the list's decide=False.
- SF2-l A Pursuit's crit armor is the outgoing switcher's. Pursuit's CalcCrit (subs/subscript_pursuit.s:10; battle_lib.c:7105-7140) is blocked by Battle Armor / Shell Armor of the defender through Battler_IgnorableAbility (:7135-7136; Mold Breaker ignores it) and by Lucky Chant on the defender's side (:7137); the defender is `switchedMon`, the OUTGOING Pokemon (battle_script.c:6927), as the hit comes before the switch (subs/subscript_switch_pokemon.s:11). The laws and the sampler judged the armor on the newcomer (`outcome.armored` via `_target_mon`); step and reference were right (`crit_blocked(pre, ...)`, `_crit_blocked`); split from SF2 (e) (SF2-e-laws.md L4). Built: `outcome.pursuit_chase` (bool[B] fires, the selected move; never paused) and `outcome.pursuit_armored`; `enumerate.outcome_laws` `no_crit` and `outcome._move_draws` (new `chase` keyword, passed by `sample_outcome` at both calls) use `pursuit_armored` where the Pursuit fires. No step or reference change. Tests: tests/engine/test_singles_followup2.py (`fullset`).
- SF2-b Intimidate skips a foe whose Substitute this action hit: built by E1 F-4 (batch F); SF2 adds the pause / Pursuit / next-turn tests. The flag is set broken or not (bcp.c:3359-3371), read by CheckSubstitute (battle_script.c:9179-9182; subscript_intimidate.s:13), kept through the party list and U-turn's Pursuit (subscript_attack_then_switch_out.s:54, :70 clears only the attacker's flags) and cleared at BattleContext_Init (battle_lib.c:2013-2016, bcp.c:3998). `move_end_switch_in` gets no `sub_hit` (unreachable, D2-26). No src change. Tests: tests/engine/test_singles_followup2.py (`fullset`): the flag at the pause ([0, 1]) and after the resume ([0, 0]), a Substitute that holds, the flag across the Pursuit chase, the next turn's switch-in (-1).
- SF2-i Enigma Berry in the resumed U-turn tail. U-turn's subscript is its ON_HIT secondary effect (effect_script_0228.s:5; lib.c:1524-1529; bcp.c:3484-3494); when it ends (the list's End, or :73 for a user at 0 HP) ExecScript returns to AFTER_MOVE_MESSAGE, then AfterMoveEffects (bcp.c:3115-3121, :3497-3530, :3604), whose TRIGGER_ITEMS_ON_HIT (:3698-3706) runs the Enigma case (HOLD_EFFECT_HP_RESTORE_SE, battle_lib.c:5421-5428): the defender (:5374), its SUBSTITUTE_HIT (:5378, lib.c:6259-6265), its curHP and `moveStatusFlags & MOVE_STATUS_SUPER_EFFECTIVE`; it heals Divide(maxHP, itemPower) (:5423) and PLUCK_CHECK removes the berry (subscript_held_item_hp_restore.s:5-13). The tail's defender is the U-turn's (subscript_pursuit.s:5, :80, :116 push and pop attacker and defender); its `moveStatusFlags` are the last Pursuit's when one ran (SetupLoop lib.c:3359 zeroes them, ApplyTypeEffectiveness subscript_pursuit.s:12 / script.c:6964-6973 sets them), else the U-turn's. Cases at B = 2: base (no Pursuit, listed): heals iff the defender is alive, not Substitute-hit, and the U-turn was super effective; (i-1) Pursuit ran, user listed: the pursuer heals iff alive after the chain, not Substitute-hit by the U-turn, and the PURSUIT was super effective on the user (the U-turn's own flag is lost); (i-2) user KO'd by Pursuit: the tail runs in the same action with the same test. Color Change never fires in the tail (nothing to change). Built: new scratch byte `ts_se` (int8, `_FIELD`; STATE.md row `BattleContext.moveStatusFlags`): `_use_move` carries `se0` (first hit's `super_effective & effective`) and writes it at `note`; `_pursuit_hit` returns `(state, super_effective)` and `_execute_actions` writes it at the U-turn chase (`ap`) only; `one_hit`'s Enigma part runs in a tail (`tail_enigma`: `~head_on & first & items & state_in.ts_se != 0 & state_in.ts_sub_hit[target_side] == 0`, read from the entry state, since the resumed switch-in call zeroes the live byte). E1 batch F's `ts_sub_hit` is a per-battler int8 (`_BATTLER`), not the sheet's bitmask: read as `ts_sub_hit[target_side] != 0`. Reference: `_Turn.move_se` (set at `_hit`'s chart and at `_pursuit`), `ts_se` in `_write_scratch` / `_SCRATCH_ZERO`, `_resume` runs `_on_hit_berry(..., True)` after the first `_held_item` loop when `ts_se` and the defender's flag is clear; (i-2) reads Pursuit's flag through `turn.move_se`. Today's reference heal at (i-2) on the U-turn's flag is gone. Note for the manager, not this item: the tail's King's Rock reads CURRENT_MOVE_DATA.flags = moveTemp's (bcp.c:4641, subscript_pursuit.s:81, :117) while the engine uses `ts_used` (the called move), step.py:1895: a separate check. Tests: tests/engine/test_singles_followup2.py (`fullset`); the Psychic Rough Skin case builds a Carvanha whose types are set to Psychic (no legal Psychic Rough Skin species).
- SF2-k The Quick Claw / Custap flags persist across the turn end. Rule 5: Trick Room's script sets trickRoomActivated (effect_script_0259.s:18; script.c:7859-7866); at the user's MoveEnd, unless CheckBattleOver ended the battle (bcp.c:3969), SortMonActionOrder (bcp.c:3991; lib.c:7276-7300) compares every pair with equal command, a FIGHT pair with both alive with ignoreQuickClaw = FALSE (lib.c:7289-7293, :1204-1211), which SETS quickClaw for an activated Quick Claw (:1283, :1349; this turn's speedRand, bcp.c:853-857 = `Outcome.quick_claw`) and custapBerry for a Custap holder in its pinch on the current HP (:1297, :1363); it never clears. Rule 6: only the holder's BeforeMove (subscript_check_quick_claw.s:22-28), UpdateAfterSwitch (lib.c:2100-2104; a Baton Pass keeps neither, :2107-2115) and CleanupFaintedMon (:2185-2188) clear them; SetupNextTurn (:2228-2253) and BattleContext_Init (:1974-2017) do not; a U-turn's pursuer becomes MOVE_END without BeforeMove (script.c:6931), so its CalcTurnOrder flags survive. Rule 10: Kaizo's turn-end `trick_room_rearm` runs no MoveEnd: no re-set. Built before D3 at every B (no `fmt.B` term; D3-18 adds the B = 4 order part): `claw_flag` / `custap_flag` int8[B] (`VOLATILE_FIELDS`, not Baton-passed; STATE.md rows `MoveEffectsData.quickClaw` / `.custapBerry`); `order.resort_claw_flags`; `_use_move` returns `ctl["started"]` (its own start pass) and `ctl["trick_room"]` (the field counter toggled inside the action); `_execute_actions` carries `claw` / `custap` through the loop (OR of the turn order and the standing flags at a fresh turn, cleared at the actor's start, an active change or 0 HP, set again at a Trick Room's MoveEnd; `ts_claw` / `ts_custap` = the carry at a pause; written `& alive` at the turn end); the defender readers lose `& ~tgt_moved` (Knock Off, Embargo); the Quick Claw bit of the `quick_claw` event is `_use_move`'s (`started & claw & ~target_moved`), `_quick_claw_shown` is gone. Reference: `turn.claw` / `turn.custap` OR the standing flags, cleared in `_enter` and `_cleanup_fainted`, re-set after `_finish_action` when `turn.trick_room_ran`, written first thing in `_end_turn`. Owed: none. Tests: tests/engine/test_singles_followup2.py (`fullset`).
