# E2 mechanics ledger (Kaizo Roark split)

Closed batches, designs and retired RESUME bullets: e2-mechanics-ledger-archive.md.

## RESUME (updated 2026-11-16 restart: batch 11 tip 49084d6 sent for merge check; batch 12 in progress)

- **Main 4676fd4 merged (2026-11-17, 08064fb; only CLAUDE.md came in, no conflicts).** Tables
  (vanilla + Kaizo) and backlog rebuilt (18 steps, 0 errors, no content change; the backlog's
  "Groups by kind" tie order flips between runs: Counter.most_common ties). Duplicate scan clean;
  fold_in indices 14-26, 28-31, 32/33, 34/35, 36, 40, 41 distinct. Targeted not-slow run, all
  green: guards 68 + test_cli 101+1, unsupported tests 204 + cli_line 7, batch12 35, batch11 10,
  invokers 16, enumerate 16, state 23, damage_classes 5, terrain 40, outcome 20, fangs 16, present
  25, batch9 20, quick_claw 27, gardenia 95, kaizo_policy batch12 77, read_bounds/commands/outcome
  96, stage_b_learn 5, differential 1 + diff.py seed 38 ok 300. Marked slow: kaizo_policy
  test_batch12_stat_family_exact_equals_loose (242 s, 5.7 GB) and _healing_ (146 s, 5.1 GB).
  OWED (killed by the free-RAM floor while the nightly held ~8 GB): test_accuracy_classes,
  test_protect, kaizo_policy `-k "not batch12"`, test_cli default_budget / deaths_and_protect.
- **Batch 11 tip merged + batch 12 second review + sampler cost (2026-11-16):** 2e7fdff merges
  d097e68 (Acupressure's draw moved to fold_in 36: 29 is Focus Band's; no other shared index),
  064636b the review follow-ups (B12-28), then the sampler flag (B12-29). OWED: B12-22's
  Psycho Shift onto Synchronize can now be checked against E1's synchronize; test_batch12 needs
  chunking under 4 GB after the merge (4.06 GB whole); the 8 GB items above; over 4 GB alone:
  test_batch11 encore_into_assist / _metronome / _copycat, an_encored_sleep_talk; test_invokers
  brute_force[sleep_talk, _disable, _imprison], sampled_calls_match, random_invoker_turns.
  DIFF seed 37 fixed (B12-30): the reference wrongly failed a RANGE_SINGLE_TARGET_SPECIAL
  invoker (awake Sleep Talk, the foe fainted to its confusion self-hit) at HasNoTarget; the
  decomp's defender is the user, so last_move is Sleep Talk as JAX had it.
- **Batch 11 merge (18:19):** main a5a60ad merged into 49084d6 on branch e2-b11-merge in
  _scratch/e2/testwt: 6777c8c merge, 14e5406 backlog, a1a6722 Metronome / Assist draw keys moved
  to 32+side / 34+side (they collided with Yawn 26, future_band 30, berry_confusion 31),
  29fa658 cherry-pick of e60f40e, 1bb69c3 Gardenia generators via _teach, d097e68
  test_stage_b_learn passes taken. Narrow checks green (diff seed 35 ok). FAST TIER running on
  d097e68 (log _scratch/e2/ft_d097e68.log). Then: merge d097e68 into this branch, send tip + F list.
  OWED over 8 GB: test_cli_rank_deaths_and_protect 8.47 GB alone, default_budget 7.82 alone
  (>8 together); test_stage_b_learn::test_a_learn_at_a_terminal_state_stays_terminal (slow) 12.49.
  Manager's non-blocking items: (b) sampler cost doubled (lax.cond -> select under vmap, so
  turn_events runs every turn): find a cheaper form; (c) Encore via an invoker has no property test.
- **Restart 2026-11-16 (manager brief).** Batch 11's merge-check tip = 49084d6 (sent). Batch 12
  WIP committed as 4eeba4d (untested), being reviewed and finished; families sent to the manager:
  stat stages (Calm Mind, Cosmic Power, Flatter, Tickle, Captivate, Acupressure, Haze, Psych Up,
  Power/Guard Swap, Focus Energy, Power Trick) and healing (Rest, Morning Sun/Synthesis/Moonlight,
  Wish, Heal Bell/Aromatherapy, Refresh, Psycho Shift). Batches 13 (power formulas, secondary
  hits, held-item moves) and 14 (field/side, move-restricting, ability-changing) wait: no batch 13
  until batch 11 is in main. Side lane: one pytest under 4 GB, lock dir _scratch/e2/side.lock,
  while E1's fast tier holds _scratch/fast_tier.lock. Test worktree _scratch/e2/testwt at 49084d6.
  When E1 merges into main: merge main, rebuild vanilla + Kaizo tables, scan for duplicate
  names/enum members/table rows, run the unsupported-asserting tests, then ONE fast tier
  (-m "not slow" --durations=40) under the lock, send tip + F list to the manager.
- **Stale unsupported tests: done** (e60f40e): each makes its own example unsupported
  (tests/engine/helpers.unmodelled, conftest metronome_refused). Ability / item / trainer-item
  examples (Leaf Guard, Levitate, Iron Ball, Full Restore, ...) are E1's area, left.
- **Batch 12 code done** (rulings B12-1..18; review WIP fixes: Power Trick desync on level-up,
  Psych Up / Captivate before TryMove). Checks under 4 GB passed (test_batch12 27, kaizo_policy
  batch12 46 + healing 26, exact==loose all but one, guards 69, diff seed 33 ok 300). OWED:
  kaizo_policy test_batch12_stat_family_exact_equals_loose[23] (>4 GB), test_cli, the review.
- **Batch 12 review (approve-with-changes) fixed** (1bf6c6c + next): crit decided at hit time
  (B12-19), Acupressure acts on the resolved defender (B12-20); Psych Up / Captivate into a
  fainted target needs no change (B12-21). Checks: test_batch12 28, outcome, damage_classes 46,
  damage 28, k3b_items 22, guards 69, decomp-law marginals, diff seed 34 ok. OWED (>4 GB on a cold
  compile cache, HEAD too): test_fangs, test_protect, test_present, test_self_destruct,
  test_observe, test_small_moves, test_power_moves, parts of test_enumerate (~70 tests).
  Owed code: Trick / Thief / Skill Swap / Worry Seed join outcome.CRIT_ODDS_EFFECTS when they
  land; the effect-based check ignores a Metronome-called Haze / Knock Off.
- **Mist done** (B12-23..27): move, 5-turn side timer, the drop block in ChangeStatStage incl.
  Intimidate, Basic AI. For E1: Focus Energy = BattleState.focus_energy[side] (Dire Hit sets the
  same bit, battle_system.c:617-622); Guard Spec writes BattleState.mist[side] = 5 when 0
  (battle_system.c:563-571). Checks: test_batch12 33, AI Mist 31 + exact==loose, Intimidate 4,
  guards 92, diff seed 36 ok. Batch 12 needs one more review (Mist, crit-at-hit) before merging.
- **Main = 2c2caea** (batch 10 merged as e022644: 042b1ab + main 8c22d37 + merge fixes c1728c9
  / 36bdc8f / c1f54cb / b5e4665 Yawn + the 6 stale tests its fast tier found 4cef8f2 / 4d165d8 /
  e022644). Main merged into this branch at 453e283. Never push.
- **Batch 11 (this branch, unmerged): code and review done** (e760d9e, 926874a, 71eccee,
  d78ba49, 8f1e469, 31b3a95 follow-ups, bb35cdc review follow-ups B11-10..12). OWED before
  sending the tip: test_invokers + test_batch11 incl. sampled tests (5-8 GB), tests/search
  test_expand / test_playout / test_pivot_search / test_stage_b, tests/test_cli.py, seed 32
  (log _scratch/e2/logs/b11_diff32.log). E1's merge fast tier holds the machine (~4-5 h from
  2026-11-16 morning): until it ends one side process under ~4 GB, heavy runs queued.
- **test_cli under 8 GB (mine):** 9.09 GB peak for test_cli_rank_default_budget_keeps_the_header
  in the fast tier; the earlier agent's work was stopped twice; restart after E1's tier.
- **Batch 12 plan (sent to the manager for a check against E1):** Metronome's 65 unmodelled
  Kaizo callable moves (list: _scratch/e2/metro_list.py), in families stat/HP, healing, power
  formulas, secondary hits, field/side, move-restricting; ability- and item-changing moves
  (Role Play, Skill Swap, Gastro Acid, Worry Seed, Fling, Natural Gift, Recycle) pending E1.
- **Found, not ours:** the new peak sampler shows tests/test_cli.py over 8 GB:
  test_cli_rank_deaths_and_protect 11.96 GB, test_cli_rank_default_budget_keeps_the_header 8.67,
  test_rank_lead_and_opponent_active 8.16 (likely the same expand CHUNK padding). Reported.
- **Owed from batch 10:** Gravity grounding a Fly user (Gravity refused by coverage), Transform
  vs a vanished target, Kaizo Hurricane (effect 141, no fight reaches it).
- **Batch 11 status (2026-11-16).** Committed: e760d9e the takeover mechanism, the Sleep Talk
  sampler fix, Encore into the invokers, B11-9, review items (c) (d) (B11-1..3, 7-9); 926874a
  Encore's AI, both Encore refusals lifted (backlog 230 -> 228), coverage tests on real data
  (B11-4, B11-6); 71eccee the generator (B11-5) and its duplicate-move fix (next commit);
  differential seeds 25, 26 clean (300 each). Frequency tests failed on 042b1ab (Sleep Talk +
  Disable 0.855 vs 0.73, + Imprison 0.855 vs 0.70, Encore into Hypnosis 1.0 vs 0.6), pass now.
  Checks: test_batch11 + test_invokers 24 passed; batch10 / accuracy / damage classes /
  enumerate / effect_map / coverage / coverage_exp / observe / pivot_search / read_bounds 222
  passed (before the sampler's single-alternative rewrite; its sampled tests rerun after: pass);
  AI + coverage_exp + game_constants 74 passed; test_variants + game_constants + test_backlog 69
  passed. Owed once the fast tier frees the machine (runs under 4 GB until then, no tests/search,
  no test_cli): test_invokers and test_batch11's hand tests after the sampler rewrite (peaks
  5-8 GB), tests/test_cli.py, the review. Sampler cost: see B11-1.
- **Batch 11 follow-ups after the merge (2026-11-16, on ef4fe39).** outcome.py's two duplicate
  `151` entries dropped; kaizo_roark.md 5.23 says CHARGE_TURN_DEF_UP is transcribed; the Yawn
  semi-miss test parametrized (Vital Spirit, Leaf Guard in and out of sun, Mold Breaker users;
  subscript_yawn.s:5-9, battle_lib.c:3108-3123); test_playout's duplicate `_replay` removed (the
  later body kept); the generator draws the foe's Yawn at a vanished battler (B10-13; 800 draws,
  seeds 1-2: the foe knows it 25, uses it 12). Light checks: test_batch10 41, test_batch11 hand
  tests 5 (`-k "not sampled"`), read_bounds 3, kaizo_policy `-k "encore or solar or charge or
  shadow"` 55, game_constants 16, tables 34, test_state 24 passed; differential seed 27 clean
  (300). **Owed heavy runs** (when the fast tier frees the machine):
  tests/engine/test_invokers.py, the sampled-frequency tests (test_batch11 `-k sampled`),
  tests/search incl. test_playout.py, tests/test_cli.py, differential seed 32 on the merge.
- **Batch 11 second review follow-ups (2026-11-16, on 31b3a95).** B11-10 (the generator never
  gives a move twice, `_teach`; the replacement-coverage property test), B11-11 (the takeover /
  replacement predicates take `side` and `foe`), B11-12 (Kaizo Expert_Encore and its table are
  vanilla, ROM-checked). Light checks: test_batch11 `-k "not sampled"` 6 passed (the new
  property test included; test_encore_into_assist reached 6.1 GB resident), test_batch10 41,
  kaizo_policy `-k encore` 12 passed; differential seeds 28, 29 clean (300 each). **Owed heavy runs** (unchanged, plus the
  new generator's draws): tests/engine/test_invokers.py, test_batch11 `-k sampled`, tests/search
  incl. test_playout.py, tests/test_cli.py, differential seed 32 on the merge.
- **Batch 11 (as briefed):** the takeover mechanism exactly
  as designed at 88fb6c5 (ledger "Design: a chosen move replaced mid-turn": one event for every
  replaced move, TAKEN law variant from a hypothetical encored state with its tag outside `call`,
  a second sampled outcome returned where the event fired, Outcome shape unchanged), including
  the Sleep Talk + Disable / Imprison sampler fix with a frequency test that fails today, Encore
  into Metronome / Assist / Sleep Talk / Copycat; then lift both Encore refusals and add Encore's
  AI (Basic_CheckCannotEncore, Expert_Encore) under the ALLIN convention.
- **ALLIN AI convention (from E1's merge):** gate a flag only through policy._flags /
  _run_flags; every random read through commands._slot_read with its bound in
  read_bounds.READ_BOUNDS; concrete code uses layout.fight_flags / fight_effects.
- **Process rules.** Per batch: focused tests + the four guards (tests/prep/test_variants.py,
  tests/data/test_game_constants.py, tests/tools/test_backlog.py, tests/test_cli.py) + one
  review; the fast tier once per merge, on the merge-check tip, under
  G:/Farry/Claude Workspace/_scratch/fast_tier.lock (launcher _scratch/e2/run_fast_tier.ps1;
  E1's merge takes the lock first). No third batch while two are unmerged. Report in a few lines
  to the manager (SendMessage local_a4231f03-556b-4b7f-bc82-38b27d410c76); a two-line note after
  90 minutes without a commit. E2 owns move-effect families end to end (mechanic, AI scoring,
  tests); E1 owns abilities, held items, battle conditions, AI flags. Keep commits small (E1
  builds on this branch: claude/e1-gardenia-on-e2). Context: do not read whole large files or
  full test logs (grep / tail), delegate file-heavy work to subagents. Runs narrow (one process,
  under 6 GB) while another fast tier holds the lock.
- **Scratch.** _scratch/e2/b10: diff.py N SEED (reduced differential, dumps fail_SEED.pkl),
  replay.py / replay3.py (replay a dump in both engines), edit.py (scripted edits), the decomp
  Substitute inventory. Test worktree _scratch/e2/testwt (now at d420167). Rebuild tables after
  an effect-map change (uv run python -m tools.build_tables; uv run --group rom python -m
  tools.build_tables --game kaizo); regenerate the backlog per family (uv run --group rom python
  -m tools.kaizo.backlog docs/kaizo/mechanics_backlog.md).

## Conventions for new code (user-approved 2026-11-13)

From section 5 of the doubles design (`docs/doubles-state-design.md` in the main checkout), so
singles mechanics need no rework when four-battler fights arrive. New code only; existing code,
the `step` turn skeleton and `enumerate.py`'s branch structure stay as they are.

1. Pass the other Pokemon in; never compute `1 - side` inside a mechanic (`hooks.pluck_berry`'s
   `foe = 1 - side` is the pattern not to copy; `effects._Hit` carries side and target).
2. Keep per-battler state (`BattleMon`, `turnFlags`, `[MAX_BATTLERS]` arrays) and per-side state
   (`SideConditions`) apart, as the decomp struct has it; say which in `STATE.md`; index side
   fields by the battler's side.
3. A field that names another Pokemon stores the battler (`1 + battler`), not a yes/no
   (Lock-On, Mean Look, Counter, Mirror Coat).
4. "Every Pokemon on the field" is a loop over battlers; "an ally has X" is a count over the
   battler's side.
5. State turn-order facts about a named battler ("the target has moved", "the target is
   protecting"), never "I am the second actor" (Payback, Avalanche, Sucker Punch).
6. Read a move's range only through `targets_user`, `self_defender`, `defender_always`; new
   chance fields are indexed by the acting battler, never a scalar meaning player vs opponent.

## Owed

Rules left out because another mechanic is missing; each closes with the mechanic named.

- **From E1's Fantina list (manager, 2026-11-16; not before batch 13):** 23 AI move-effect
  transcriptions (EVA_UP/Double Team, HIGH_CRITICAL_POISON_HIT, Transform, Perish Song, Gravity,
  Magnet Rise, Nature Power, Chatter, Gyro Ball and others; the backlog rows name them) and the
  Kaizo scripts be_seq#141 (Hurricane), be_seq#88 and sub_seq#238.

- Batch 9 left (Gardenia-split entries not done, by why): SolarBeam #151 (a charge turn: the
  two-turn lock and SYSCTL_FIRST_OF_MULTI_TURN's skips); Jump Kick / Hi Jump Kick #45
  (subscript_crash_on_miss.s reads the miss's damage: crit and roll on a miss, a law change);
  Volt Tackle's family #262 (recoil plus a PROBABILISTIC paralysis that runs on every hit,
  battle_lib.c:1551-1567: a new secondary shape); Endeavor #189, Pain Split #91 (Pain Split's
  UPDATE_HP meets the resist-berry subscript); Knock Off #188 (knockedOffItemsMask, Unburden,
  Sticky Barb's Owed test); Focus Punch #170, Safeguard #124, Substitute #79, Encore #90, Assist
  #180; abilities Leaf Guard (every status subscript's two checks), Synchronize, Wonder Guard
  (with Fire Fang's bypass), Pressure (PP cost by range, skipPressureCheck, its switch-in
  reveal), Download (downloadAnnounced, the search's replacement pass); held items Focus Band
  (a per-hit roll at three sites: a new Outcome field and law), Wiki Berry and the other flavor
  berries (the dislike confusion's duration draw at an item check); the terrain entries (a
  trainer whose maps give several terrains: data, not a mechanic). Metronome stays refused
  (B5-2).
- AI owner (E1), batch 9. Engine reads the AI meets: `order.effective_speed` (Chlorophyll,
  Unburden with the new `can_unburden`), `actions.trapped` (Shadow Tag, Arena Trap, Magnet Pull:
  the player's switch gate), `outcome._rate` (Thunder / Blizzard weather accuracy, by
  `MoveFlag.THUNDER` / `BLIZZARD`), `damage._pre_random` (Huge Power / Pure Power, Swarm, Flower
  Gift; `TrainerAI_CalcDamage` calls BattleSystem_CalcMoveDamage, trainer_ai.c:3082), the
  absorb abilities (Volt / Water Absorb), `minimized` (Stomp's x2 is the move script's, which the
  AI's estimate does not run). New EffectIds FOLLOW_ME, HELPING_HAND, BELLY_DRUM; Bulk Up is
  DRAGON_DANCE with Defense (move_param 0), Minimize STAT_SELF (Evasion, param 2), Stomp FLINCH
  (param 2), Payback DOUBLE_POWER (POWER_IF_TARGET_MOVED), Dream Eater DRAIN (param 1), Dragon
  Rage / SonicBoom LEVEL_DAMAGE (param 0 the amount), Double-Edge RECOIL (param 0 the divisor
  now, 2 / 3 / 4). The AI move-effect scripts of these (the backlog's AI entries) are E1's.
- The weather lock read by Thunder / Blizzard (and Sand Veil, Snow Cloak) in the hit laws is the
  turn start's battlers (`outcome._rate`'s `on_field`); a Cloud Nine / Air Lock holder that faints
  before the second mover's check (its own recoil, Life Orb) is not followed (pre-existing for
  Sand Veil; Thunder makes it a 70 vs 100 difference).

- The Sticky Barb transfer vs Knock Off and the knocked-off mask (effects.py
  `sticky_barb_transfer`; battle_lib.c:5390-5393): Knock Off (U-turn's guard is modelled since
  B6-4; the substitute test, :5378, since B10-4). Magic Guard's readers are reached since B4-11.
- Fire Fang (`BATTLE_EFFECT_FLINCH_BURN_HIT`) always bypasses Wonder Guard (the decomp's bug note,
  battle_lib.c:7600): Fire Fang / Wonder Guard.
- Gravity makes the moves that fail in high gravity (Jump Kick, Hi Jump Kick, ...) invalid:
  `Move_FailsInHighGravity` under `CHECK_INVALID_GRAVITY` in `BattleSystem_CheckInvalidMoves`
  (battle_lib.c:2292, :2348), which the AI also reads (trainer_ai.c:234); `actions.usable_moves`
  ignores Gravity, and so do Sleep Talk's slots (`order.sleep_talk_slots`, battle_script.c:4927)
  and `Move_CanBeMetronomed` (battle_lib.c:7204-7207: `order.metronome_candidates`,
  `order.copycat_call`, `coverage.metronome_calls`); its Heal Block test is an unsupported move's:
  Gravity.
- AI owner (E1), batch 7. New fields to read: `ai_item_known` (int16 per player party slot:
  `battlerHeldItems`, the item id, for `IfHeldItemEqualTo` / `LoadHeldItemEffect` on the player,
  trainer_ai.c:1901-1934), `ingrain`, `mean_look` / `mean_look_target` (1 + battler),
  `imprison`, `imprisoned`, `roosting` (per battler; nonzero only at a pause or while a faint
  replacement is pending, which is when `BattleAI_PostKOSwitchIn`'s chart reads it,
  battle_lib.c:2540). Behaviour E1 now meets: `actions.trapped` (Bind, Mean Look, Ingrain) gates
  the switch actions (switching.py reads only `trapped > 0`); `actions.move_imprisoned` removes
  sealed moves from `usable_moves` (TrainerAI_Init's CHECK_INVALID_ALL); `damage.identified` now
  returns flags (`CHART_IDENTIFIED | CHART_GROUNDED`), so every engine chart walk grounds an
  ingrained Flying defender and `damage.ignorable` hides an ingrained Levitate; E1's own
  `switching.chart_multipliers` does neither (the "Ingrain" and "Roost" refusals there can now be
  lifted on these fields). Scripts still untranscribed: #106 (Basic_CheckMeanLook /
  Expert_BindingMove), #192 (Basic_CheckCanImprison / Expert_Imprison), #181 Ingrain's.
- The Iron Ball and Gravity clauses of `BasicTypeMulApplies` / `Battler_Ability`
  (battle_lib.c:2534, :2545-2549, :3095-3098) beside Ingrain's (`damage.identified`): Gravity,
  Iron Ball.
- Doubles (batch 9 review): Flower Gift counts a living holder over the whole side
  (COUNT_ALIVE_BATTLERS_OUR_SIDE, battle_lib.c:2886-2893, read at :6930-6941); `damage._pre_random`
  reads the attacker's / defender's own ability, exact in singles only. RANGE_ALLY and
  RANGE_USER_OR_ALLY naming the user (`effects.targets_user`, battle_lib.c:1670-1684) is singles
  only: in doubles they name the partner, and Helping Hand (which fails in singles, B9-5) boosts
  the ally's move there.
- Doubles: `ai_item_known` is kept per player party slot, which is exact in singles only:
  `AICmd_LoadHeldItemEffect` (trainer_ai.c:1907-1910) reads `battlerHeldItems` for any battler
  but the AI's own, so in doubles it also reads the AI partner's record (and `IfHeldItemEqualTo`,
  :1924-1928, the other side's). The doubles milestone keeps it per battler on both sides.
- AI owner (E1), batch 8: `curse`, `perish_song` / `perish_song_turns`, `yawn`, `lock_on` /
  `lock_on_target` are now set (B8-1..B8-5), with the turn-end checks and the leave rules; the AI
  scripts of the moves (Lock-On, Perish Song, Yawn, Ghost Curse) are E1's. New Outcome field
  `yawn_sleep` (the turn-end sleep's Random 3, 2; `ai.outcome` is untouched). Lock-On makes the
  locker's next moves skip the miss in the engine's hit laws (`outcome._rate`): the AI's own
  accuracy reads are E1's. `coverage.ghost_curse` is gone (the AI's own Ghost Curse refusal in
  `ai.policy` stays E1's).
- Lock-On's lock survives its locker's faint (CleanupFaintedMon has no Lock-On loop) until the
  slot's replacement comes in, whose UpdateAfterSwitch clears it (B8-7).
- AI owner (E1): Copycat's AI move-effect script (#242: Basic_CheckCopycat / Expert_Copycat) is
  untranscribed (the backlog's AI entry); the engine side is B5-1. The battle's movePrev is
  `BattleState.move_prev`; trainer_ai.c reads only movePrevByBattler (`last_move`). (Sleep Talk
  #97 and Metronome #83 are transcribed on main, merged at 23e10f0.)
- Metronome (#83, 3 fights) stays refused by B5-2 until every move it can call is modelled
  (150 vanilla / 128 Kaizo of its 442 candidates have unsupported effects).
- Around the pivots (E2 batch 6, B6-2..B6-4), each with its mechanic: the defender's Jaboca Berry
  (HOLD_EFFECT_RECOIL_PHYSICAL) in `effects.pivot_item` (battle_lib.c:7368-7375; an unsupported
  item, refused); the knocked-off mask of its Sticky Barb test (:7381; Knock Off); Grudge
  (subscript_attack_then_switch_out.s:40), Rage and Shaymin's form check after a switch
  (bcp.c:3497-3530); what Baton Pass keeps of Focus Energy, Mean Look, Curse, Substitute,
  Lock-On, Perish Song, Ingrain, Power Trick, Embargo, Heal Block, Magnet Rise and Gastro Acid
  (battle_lib.c:2056-2086, :2105-2115, :2136-2140; E1's per-battler fields too); Pressure on U-turn's
  Pursuit (battle_script.c:6919-6922); doubles: `BattleAI_PostKOSwitchIn`'s
  `BattleSystem_RandomOpponent` draw (battle_lib.c:4183-4187) at the opponent's list.
- AI owner (E1): Baton Pass's AI side is untranscribed (the backlog's "AI move effect #127"
  entry, Basic_CheckBatonPass / Expert_BatonPass; `AI_FLAG_BATON_PASS`). U-turn's (#228,
  Expert_UTurn, script.s:5085-5145) is transcribed on main (`policy._expert_u_turn`), checked
  against the pivot at the merge (23e10f0): it scores at the turn's choice, never at a pause.
  The engine side is B6-1..B6-9; the AI is never asked at a paused state (the opponent's pick at its own list is `post_ko_switch_in` with
  `ts_pick_mul`, `playout.turn_inputs`). After a switch the newcomer's `last_move` is the paused
  move's moveTemp (UpdateMoveBuffers, bcp.c:3944), which the next turn's
  `TrainerAI_RecordLastMove` reads.
- Owed test restructuring: `tests/search/test_leaves.py`. One node, `test_wins_count_successes`,
  peaked at 7.4 GB at ffcdc54 (before batch 6); the file goes past 8 GB at the tip (the standing
  per-file cap is 8 GB).
- Tests: `tests/search/exhaustive.py` (the reference solver) plays no pause; its endgames have no
  U-turn or Baton Pass.
- Rest and Yawn (unsupported moves) read `FIELD_CONDITION_UPROAR` too (subscript_rest.s,
  subscript_yawn.s): whoever adds them must test `effects.uproar_on` there (B5-6).
- AI owner (E1): Future Sight / Doom Desire's AI side is untranscribed (the backlog's "AI move
  effect #148" entry: Basic_CheckFutureSight, trainer_ai/script.s:825-826, scores -12 when
  `SIDE_CONDITION_FUTURE_SIGHT` is up on the defender's or the attacker's side). That bit is
  `BattleState.future_sight` (per side), which can stay up after the count ran out on a fainted
  battler (B5-10): read it, not `future_timer`.
- The status orbs vs Safeguard (subscript_badly_poison.s, subscript_burn.s): Safeguard.
- `outcome.kings_rock_chance` reads the turn-start item (outcome.py), which breaks once a move
  can move items mid-turn: Trick, Thief, Fling.
- Kaizo's Ganlon and Apicot Berries (param 0) are refused by the divisor guard
  (`effect_map.DIVISOR_PARAM_MIN`, B2-3). Finding: `curHP <= maxHP / itemPower`
  (battle_lib.c:4778, :1294) is a C `/` on ints, which the compiler emits as a call to the
  Metrowerks / NitroSDK runtime's signed divide helper; that helper is linked from the SDK
  libraries (subprojects/NitroSDK, not in the decomp's own sources, and not fetched here), so
  the decomp does not say what a zero divisor returns. Closing it needs the helper's code read
  from the ROM's arm9 (its divide-by-zero branch), then the berries modelled with that result
  instead of refused.

## Rulings

(One line each, with the decomp citation.)

- B1-1 Item order: the held-item block runs after powerMul, Charge, Helping Hand, Technician,
  Huge Power / Pure Power and Slow Start (battle_lib.c:6688-6714), before every ability
  modifier from :6785 on (Thick Fat, Hustle, Guts, ...), and in the decomp's
  order (type item, Choice Band/Specs, Soul Dew, DeepSeaTooth/Scale, Light Ball, Metal Powder,
  Thick Club, orbs, Muscle Band/Wise Glasses), so e.g. Thick Club x2 truncates before Guts x1.5;
  the u16 stores never truncate in reach (battle_lib.c:6716-6783 before :6793).
- B1-2 Light Ball doubles `movePower` for any move class, physical and special alike
  (battle_lib.c:6748-6751).
- B1-3 The species items read `BATTLEMON_SPECIES`; Soul Dew's Battle Frontier test never holds in
  a trainer battle and the Griseous Orb's Transform test never holds while Transform is
  unsupported (battle_lib.c:6730-6739, :6768-6772).
- B1-4 The confusion self-hit gets the holder's own `BattleSystem_CalcMoveDamage` items (Choice
  Band, a Normal-type item for Struggle's type, Muscle Band, Light Ball, its own Metal Powder /
  DeepSeaScale / Soul Dew as defender), never Life Orb or Expert Belt (`CALC_SELF_HIT`,
  include/battle/common.h:35; battle_controller_player.c:2633).
- B1-5 Life Orb's x1.3 is the move script's (`BattleScript_CalcMoveDamage`, battle_script.c:1346,
  after `criticalMul`): not in the AI's `TrainerAI_CalcDamage` estimate (trainer_ai.c:3081) and not
  on a fixed-damage move (Night Shade / Seismic Toss: effect_script_0087.s has no CalcDamage; Bide);
  Pursuit's switch hit gets it (subscript_pursuit.s CalcDamage).
- B1-6 Life Orb recoil is `Divide(maxHP, 10)` (not the param), once per move after all hits,
  when a hit's `UpdateHP` ran with damage != 0 (`SYSCTL_MOVE_HIT`, battle_controller_player.c:3340,
  :3441), `moveCur` is not a status move and the holder lives: fixed-damage hits recoil too, a hit
  capped to 0 by a 1-HP Focus Sash still counts, the confusion self-hit never does
  (battle_controller_player.c:4758-4772).
- B1-7 Life Orb recoil also follows Pursuit's pre-switch hit unless it did not hit
  (subscript_pursuit.s:17, :48-49, :94-95 -> `BattleSystem_TriggerHeldItemOnPivotMove`,
  battle_lib.c:7357-7366).
- B1-8 Life Orb recoil runs inside the action, before its faint check: a holder that KOs the
  last foe and faints to its recoil ends the battle with both sides out, a loss (ruling 2;
  `TRIGGER_AFTER_HIT_EFFECTS` before `MOVE_END`, battle_controller_player.c:3897-3915).
- B1-9 Choice lock: set at `UpdateMoveBuffers` only when the attack message showed, to `moveTemp`
  (Nature Power itself, not its call), never to Struggle; a non-holder's lock is cleared there; a
  disrupted move (sleep, freeze, flinch, Disable, Taunt, confusion self-hit, paralysis, love)
  changes nothing (battle_controller_player.c:3921-3934).
- B1-10 The defender's lock is cleared in `UpdateFlagsWhenHit` unless it holds a Choice item, when
  the attack message showed (battle_controller_player.c:4489-4496); `BtlCmd_TryPursuit` locks a
  Choice holder into Pursuit (battle_script.c:6943-6950).
- B1-11 A Choice-locked battler whose locked move has no PP (or is disabled) has no usable move and
  Struggles; a lock the battler does not know is cleared, read as no lock
  (battle_lib.c:2307-2315; `STRUGGLING_ALL`, battle_controller_player.c:433).
- B1-12 A lock set on a battler that fainted during its move is never read and is cleared at the
  next switch-in (battle_lib.c:2100-2103); the engine clears it at the faint check.
- B1-13 Choice Scarf's x1.5 Speed comes after Swift Swim and before paralysis (battle_lib.c:1228,
  :1248, :1260).
- B1-14 Expert Belt multiplies a `MOVE_STATUS_SUPER_EFFECTIVE` hit after Solid Rock / Filter, inside
  `BattleSystem_ApplyTypeChart`, so the AI's estimate has it too (battle_lib.c:2657-2665;
  trainer_ai.c:3096).
- B1-15 Items with no hold effect do nothing, except that a berry among them (`Item_IsBerry`,
  src/item.c:357-365) is eaten by Pluck / Bug Bite with no effect, leaving the pluck flag set
  (battle_lib.c:5832-5835, :5848); the prep never offers such a berry (it can only help the foe).
- B2-1 Cheri, Rawst, Aspear and Persim Berries cure paralysis, burn, freeze and confusion at every
  held-item site (after each hit, on switch-in, `USE_ITEM` and `ITEM_CONDITION` at the turn's end),
  and a Pluck / Bug Bite that eats one cures the eater the same way (battle_lib.c:4601-4660,
  :4991-5050, :5569-5641).
- B2-2 A stat pinch berry (Liechi, Ganlon, Salac, Petaya, Apicot) raises its stat by one stage at
  `curHP <= maxHP / param` (C division; the holder's own Gluttony halves the param), only below
  +6, at the same sites; one eaten by Pluck / Bug Bite raises the eater's stat with no HP test
  (battle_lib.c:4773-4836, :5196-5259, :5753-5796).
- B2-3 A pinch berry or Custap Berry whose param is below 2 is unsupported: its trigger divides by
  zero (with Gluttony's halving), which the decomp leaves to the ROM's division routine
  (battle_lib.c:4778, :1294); Kaizo's Ganlon and Apicot Berries have param 0.
- B2-4 Custap Berry in its pinch sets the Quick Claw flag in every `CompareBattlerSpeed`: the turn
  order, the end-of-turn order and the AI's speed compares (battle_lib.c:1288-1300); its
  `custapBerry` flag is written by `CalcTurnOrder`'s FIGHT-against-FIGHT compare
  (bcp.c:772-786) and by `BattleSystem_SortMonActionOrder` (battle_lib.c:7276-7300, the same
  compare with `ignoreQuickClaw = FALSE`, :1296, :1362) after a Trick Room activation
  (bcp.c:3990-3993; Trick Room is unsupported: adding it must re-set the flag there), and the holder eats the berry when its move starts, before the status
  checks, first or last (subscript_check_quick_claw.s:26-28, bcp.c:3140-3143).
- B2-5 While that flag is up, a foe's Pluck / Bug Bite does nothing to the holder: no berry eaten,
  no pluck flag (battle_script.c:8125-8127).
- B2-6 Leftovers and Black Sludge act at `MON_COND_CHECK_STATE_LEFTOVERS`, right after `USE_ITEM`
  and before Leech Seed (bcp.c:1369): `Divide(maxHP, 16)` below full HP, for Black Sludge only on a
  Poison-type battler (its current types); any other Black Sludge holder loses `Divide(maxHP, 8)`
  unless Magic Guard; neither item is used up (battle_lib.c:4922-4965).
- B2-7 Toxic Orb, Flame Orb and Sticky Barb act at `MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT`, the
  last check (bcp.c:1710): an orb's held-item side effect skips its holder's own Immunity / Water
  Veil, Leaf Guard in sun without a weather lock, any status and a Poison or Steel / Fire type, else
  it badly poisons (counter 0) or burns (subscript_badly_poison.s:13-28, subscript_burn.s:5-17);
  Safeguard comes from an unsupported move and is not read.
- B2-8 Sticky Barb's holder loses `Divide(maxHP, param)` (param 8) at that check unless Magic Guard;
  it may faint (battle_lib.c:5330-5336).
- B2-9 Sticky Barb moves to the attacker after a hit of a contact move once the defender took damage
  this action, when the attacker lives and holds nothing (an eaten berry leaves nothing); the
  defender may have fainted, and the move is not a `RemoveItem` (no recycle item)
  (`AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT`, battle_lib.c:5386-5398,
  subscript_transfer_sticky_barb.s); Pursuit's pre-switch hit moves it too (subscript_pursuit.s:44,
  :90).
- B2-10 After every hit the order is the attacker's item, the defender's item, the Sticky Barb, the
  defender's thaw, then the held-item-status pass (bcp.c:3684-3730): a frozen defender's Aspear or
  Lum Berry cures the freeze before a Fire hit thaws it, and is used up (this also settles Lum,
  which the engines thawed first before batch 2).
- B2-11 Accuracy items come after the abilities and deep fog, before Gravity: the defender's
  BrightPowder / Lax Incense `* (100 - param) / 100`, then the attacker's Wide Lens and, against a
  target that has moved, Zoom Lens, each `* (100 + param) / 100` (bcp.c:2949-2965); "moved" is
  `Battler_MovedThisTurn` (battle_lib.c:5365-5368): in singles, the attacker acts second (a foe
  that switched in this turn has moved).
- B2-12 King's Rock / Razor Fang roll per hit after the contact ability (`*_EXTRA_FLINCH`) when the
  hit's `UpdateHP` ran, `RandNext() % 100 < param`, the move has `MOVE_FLAG_TRIGGERS_KINGS_ROCK` and
  the defender lives; the flinch needs a target that has not moved and no Inner Focus as the
  attacker sees it (bcp.c:4630-4653, subscript_flinch_mon.s:5-9): `Outcome.flinch_item[2, 5]`.
- B2-13 Shell Bell heals `Divide(dealt, param)` after the move, before Life Orb, where `dealt` sums each
  hit's `battleCtx->damage` after the endure cap, capped at the target's HP and not halved by a
  resist berry (bcp.c:3427-3431, :4736-4752); after Pursuit's hit `dealt` is the script's damage
  before the Focus Sash cap, capped at the HP (subscript_pursuit.s:21-29, battle_lib.c:7345-7355).
- B2-14 Light Clay makes its holder's Reflect / Light Screen last 5 + param (3) turns
  (battle_script.c:4229, :4272).
- B2-15 Gluttony is supported: the pinch berries and Custap are its only decomp readers
  (battle_lib.c:1289, :1355, :4774-4851, :5197-5275).
- B2-16 Kaizo changes item params only, read from each game's table: BrightPowder, Lax Incense and
  King's Rock have param 11 (vanilla 10); its Berserk Gene is a renamed Shoal Salt slot with
  hold effect 36 and param 0, so it stays unsupported (renamed slot, and B2-3).
- B3-1 Technician (`attackerParams.ability`) raises `movePower <= 60` by x15/10 after powerMul (Charge
  and Helping Hand, between them, are unsupported) and before the held items, never for Struggle
  (so not the confusion self-hit); it reads the input power (Hidden Power's, Magnitude's, Flail's,
  Rollout's) and is in the AI's estimate (battle_lib.c:6688-6702; trainer_ai.c:3078).
- B3-2 Reckless is only the recoil scripts' `BTLVAR_POWER_MULTI = 12` (effect_script_0048.s,
  _0269.s: the supported RECOIL moves; 0045, 0198, 0253, 0262 are unsupported effects): no AI
  estimate, no self-hit, and Pursuit is no recoil move (`damage.script_power_mul`).
- B3-3 Thick Fat (ignorable) halves Fire / Ice power right after the items (battle_lib.c:6785-6788);
  Dry Skin (ignorable) raises Fire power x125/100 after Overgrow / Blaze / Torrent (:6845-6848);
  Solar Power raises Sp. Atk x15/10 in sun inside `NO_CLOUD_NINE` (:6922-6924; the self-hit passes
  no field conditions).
- B3-4 In `BattleSystem_ApplyTypeChart`: Levitate (ignorable) sets `MOVE_STATUS_LEVITATED` for a
  Ground move (:2600-2603), part of `MOVE_STATUS_IMMUNE` and `DID_NOT_HIT`, so no damage, no
  AfterMoveEffects and a zero in the AI estimate (trainer_ai.c:3106); Scrappy breaks the walk at
  0xFE like Foresight (:2611-2615) for every chart reader (damage, Thunder Wave, the resist-berry
  flag, Night Shade's immunity); Tinted Lens doubles a not-very-effective hit after Expert Belt
  (:2668-2672). Status moves other than Thunder Wave skip the chart (bcp.c:2350-2358).
- B3-5 Battle Armor and Shell Armor (the same ignorable test, battle_lib.c:7135-7136) stop every
  crit; Super Luck adds 1 to the crit stage from the battler's own ability (:7126). Both engines
  gate the drawn crit with the armor at the hit; the laws draw none (`outcome.armored`).
- B3-6 No Guard of either battler (`Battler_Ability`, not ignorable) clears `MOVE_STATUS_MISSED` after
  the roll (bcp.c:3016-3021): threshold 100. Tangled Feet (ignorable) halves the rate against a
  confused target after fog and before the items (bcp.c:2944-2947).
- B3-7 Quick Feet with any status (`MON_CONDITION_ANY`) raises Speed x15/10 instead of the paralysis
  quartering (battle_lib.c:1257-1261, :1323-1327).
- B3-8 Serene Grace doubles `effectChance` (a u16, then `% 100 <`) in
  `BattleSystem_TriggerSecondaryEffect` and `BtlCmd_CheckEffectActivation` (battle_lib.c:1552-1556,
  :1571-1575; battle_script.c:8369-8373): every probabilistic secondary, Fake Out's flinch and both
  Thunder Fang rolls; not the contact abilities' or King's Rock's rolls.
- B3-9 Shield Dust (ignorable) silently stops every `SIDE_EFFECT_TYPE_INDIRECT` effect on its holder:
  a secondary status, confusion, flinch (King's Rock's too) or stat drop (battle_script.c:2918-2921;
  subscript_paralyze.s:13, _burn.s:27, _freeze.s:11, _poison.s:20, _badly_poison.s:38,
  _fall_asleep.s:31, _confuse.s:15, _flinch_mon.s:9), after Insomnia / Limber / Inner Focus; Bind's
  start (no check) and the contact abilities are not stopped.
- B3-10 Truant: `Battler_CheckTruant` loafs when `truant != (totalTurns & 1)`, `truant` being
  `(totalTurns + 1) & 1` from the entry (battle_lib.c:2119, :2197, :3304-3313): `fake_out_turn`'s
  parity, 0 for a lead, whatever the battler did. Checked after sleep and freeze, before the flinch
  (bcp.c:2502-2510); subscript_loafing_around.s unlocks and tags no ability; a loafing pursuer does
  not hit a switcher (battle_script.c:6903).
- B3-11 Dry Skin takes a damaging Water move that hit (Hidden Power's type too;
  `TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES`, bcp.c:3281-3286, :2782-2810, battle_lib.c:3538-3544):
  `Divide(maxHP, 4)` below full HP, its ability tagged either way, then `MOVE_STATUS_NO_MORE_WORK`
  (no hit, no secondary, no AfterMoveEffects; the attack message showed). At the turn's end sun
  takes `Divide(maxHP, 8)` from Dry Skin and Solar Power (Magic Guard skips it) and rain gives it to
  Dry Skin below full HP (battle_script.c:5585-5625), each tagging the ability.
- B3-12 The weather moves fail on their own weather, temporary or permanent, except Hail, which
  fails only on a temporary hail (effect_script_0164.s checks `FIELD_CONDITION_HAILING_TEMP`): it
  turns a permanent hail into a 5-turn one. Success clears every weather bit and sets 5 turns plus
  the user's matching rock param (Damp / Heat / Smooth / Icy Rock); a failure is
  `MOVE_STATUS_FAILED` (no AfterMoveEffects). They are `RANGE_FIELD`, accuracy 0.
- B3-13 Drizzle, Sand Stream, Drought, Snow Warning set their weather permanently at their holder's
  switch-in check when it has HP (after Stealth Rock), whatever weather was up; the script (and
  its ability message) is skipped only under that same permanent weather (battle_lib.c:3740-3790);
  Cloud Nine does not stop it.
- B3-14 Trace (battle_lib.c:3718-3737, `ChooseTraceTarget` :7750-7775): a holder with HP and no
  Griseous Orb copies a living foe's `BattleMon.ability` unless it is Forecast, Trace or Multitype;
  only the battler's (no form change: no party write), so it ends when it leaves; the AI learns the
  tracer's and the target's abilities (battle_script.c:3403, :11806). A Trace facing Trace stays
  armed and copies the next newcomer mid-turn, which the turn-start chance laws do not follow:
  Trace in both parties was refused (`coverage.trace_standoff_line`; modelled since E1 G-25). A player Trace newcomer that
  entered against a fainted foe traces when the foe's replacement enters (`hooks.on_switch_in`
  runs Trace for the other battler too; `playout.replacement_pass` defers only the newcomer's own
  Intimidate).
- B3-15 Two weather abilities that set different weathers, one in each party, were refused
  (`coverage.weather_war_line`; modelled since E1 G-26): entering together (battle start, a double KO's replacements) the
  `monSpeedOrder` loop (battle_lib.c:3740-3790, sorted by `CompareBattlerSpeed` at
  bcp.c:251, :263) leaves the slower one's weather, with an exact tie a `RandNext() & 1` the engine
  does not draw there.
- B3-16 Lightningrod redirects only to a battler other than the attacker and defender
  (battle_lib.c:1756-1772), which singles never has; Honey Gather acts only after the battle
  (battle_script.c:7825): both supported with no in-battle effect.
- B3-17 Kaizo changes none of these: its accuracy changes add effect 141 (Hurricane, unsupported) to
  Thunder's sun and rain branches (docs/kaizo/rom_diff.md, CheckMoveHitAccuracy /
  CheckMoveHitOverrides), and TriggerEffectOnSwitch only its field-weather switch.
- B3-18 The leads' switch-in checks run as one `BattleSystem_TriggerEffectOnSwitch` pass over both
  battlers, state by state (battle_lib.c:3649-4168; Trace :3718-3737 for every lead before
  Intimidate :3795-3808, each Intimidate announced once): a lead that traced Intimidate lowers the
  foe once, like any Intimidate lead (`hooks.battle_start`, from `make_state` and prep's
  `make_states`).
- B3-19 The turn-start laws read a switching-in target's entry ability, the copy when it traces
  (`damage.entry_ability`): its switch-in checks run before the foe moves, so a Trace newcomer
  facing a Sand Veil attacker in sand is hit at x80/100 (bcp.c:2927-2928), one facing a Static
  contact attacker rolls Static, and one that copies Battle Armor / Shell Armor is never crit
  (`outcome._rate`, `outcome.armored`, `enumerate._holds`, Cute Charm's `charm_later`).
- B3-20 Prep applies the party-pair refusals per candidate against the trainer's party
  (`weather_war_line`, `trace_standoff_line`, both deleted by E1 G-25 / G-26;
  `trace_copy_line` remains): `prep.problem.usable_pool` leaves the candidate out and
  `prep.pool.check_pool` refuses it.
- B3-21 (revised at the main merge, 2026-11-15) The AI reads a battler's current ability
  (`damage.decomp_abilities`, E1's fix: ai/commands.py `attacker_ability` / `_defender_abilities`,
  ai/policy.py and ai/switching.py refusals), so it sees the copy `subscript_trace.s` writes with
  `BattleAI_SetAbility` (battle_script.c:3401-3403, :12106-12109) as `LoadBattlerAbility` /
  `CheckBattlerAbility` do (trainer_ai.c:1170-1209); the engine already writes the copy and marks
  it known for the player's side (`hooks.trace`, `_know`) and clears that on a switch
  (battle_lib.c:7726, `step`). The blanket refusal (`trace_ai_line`) is lifted. What remains is
  a copy the AI refuses on the holder's side (e.g. Filter copied by the player, Tinted Lens by
  the trainer: `ai_refusals` is side-indexed and runs on the turn-0 state only):
  `coverage.trace_copy_refusals` runs `ai_refusals` with each copy the holder can make
  (`ChooseTraceTarget`'s Forecast / Trace / Multitype skipped, battle_lib.c:7750-7775) in place,
  and lists the lines the armed state does not have; `check_coverage` adds them, prep applies them
  per candidate (`trace_copy_line`, B3-20). `check_opponent` has no Trace line now (it depends on
  the player's party).
- B3-22 Stealth Rock's damage has one formula (`damage.stealth_rock_loss`), read by the switch-in
  (`effects.stealth_rock_damage`) and by the turn-start weather of a newcomer
  (`damage.entry_weather`).
- B4-1 A trainer battle's terrain is `CalcTerrain` of the tile the player stands on
  (field_battle_data_transfer.c:467-505): `tools/terrain.py` narrows each trainer to its
  sight-line and talk tiles (trainer_encounter.c:165-343, unk_0203C954.c:62-97) or its coord-event
  rectangle (:208-225) when nothing before the battle can move the player, else the whole header.
  Roark split: Barry #2 (247-249) is TERRAIN_PLAIN; Lass Kaitlin (323) is GRASS (B4-19).
- B4-2 Metronome stays unsupported: `BtlCmd_Metronome` (battle_script.c:4535-4562) draws from
  move ids 1-467 minus the user's moves and `sCannotMetronomeMoves` (battle_lib.c:7152-7178): 442
  candidates in both games, of which 145 (Kaizo) / 174 (vanilla) have unsupported effects; no
  partial set is sampled.
- B4-3 SmellingSalt, Facade, Brine (and Kaizo's Wring Out, effect 221) set `powerMul` 20 before
  CalcDamage (effect_script_0171.s:6, _0169.s:5, _0221.s:5-8: a paralyzed defender, a user with
  `MON_CONDITION_FACADE_BOOST`, `curHP <= maxHP / 2` with a plain `/`), else 10
  (`damage.script_power_mul`); a burned Facade user is still halved (battle_lib.c:6978).
  SmellingSalt's cure is an ON_HIT indirect effect on a defender with HP left
  (subscript_heal_target_paralysis.s:5-6).
- B4-4 Vanilla Wring Out / Crush Grip (effect 237): `1 + 120 * curHP / maxHP` of the defender
  (battle_script.c:7265); Grass Knot / Low Kick: `sWeightToPower` on the defender's species
  weight (battle_script.c:6808-6826, battle_lib.c:150). The trainer AI's estimate keeps the
  table power for both (no case in the engine's `TrainerAI_CalcDamage` path).
- B4-5 Super Fang deals `Divide(curHP, 2)` under `SYSCTL_IGNORE_TYPE_CHECKS` (effect_script_0040.s):
  no crit, roll, STAB or Life Orb boost, the immunity still applies (battle_lib.c:7544-7552), the
  resist berry is skipped (subscript_type_resist_berry.s:5).
- B4-6 The OHKO moves (`BtlCmd_TryOHKOMove`, battle_script.c:4334-4386): no stage, ability, item or
  weather accuracy factor; they hit when `RandNext() % 100 < accuracy + (Lu - Ld)` at `Lu >= Ld`,
  always at `Lu >= Ld` with No Guard on either side; an ignorable Sturdy blocks them; the damage is
  the defender's max HP (bcp.c:3343-3345), a full-HP Focus Sash endures it; no crit; the resist
  berry is skipped (subscript_type_resist_berry.s:6); the type chart's immunities apply.
- B4-7 False Swipe caps a KO at 1 HP inside UpdateHP before the endure checks (bcp.c:3381-3384);
  at 1 HP it deals 0 with UpdateHP run (SYSCTL_MOVE_HIT: Life Orb, Shell Bell), while the damage
  taken it records is 0, so no Sticky Barb transfer and no King's Rock flinch.
- B4-8 Earthquake's x2 against a digging target (effect_script_0147.s) is never met (Dig is
  unsupported): a plain hit. Poison Fang is a badly-poison secondary at the move's chance
  (effect_script_0202.s, subscript_badly_poison.s).
- B4-9 Swagger (subscript_swagger.s): Attack +2 below +6, then confusion (Random 3, 2) unless
  confused; at +6 and confused it fails (MOVE_STATUS_FAILED, no AfterMoveEffects). Dragon Dance:
  Attack +1 and Speed +1, each skipped at +6; both at +6 is MOVE_STATUS_NO_MORE_WORK.
- B4-10 Filter shares Solid Rock's `Divide(damage * 3, 4)` (battle_lib.c:2657-2661, ignorable);
  Sniper makes the crit x3 (:7142-7144, Battler_Ability); the crit items add 1 stage (Scope Lens,
  Razor Claw) or 2 on their species (Lucky Punch on Chansey, Stick on Farfetch'd) (:7123-7128).
- B4-11 Magic Guard (Battler_Ability / CheckAbility, never ignorable) skips: Leech Seed's drain
  (bcp.c:1381), full paralysis before its draw (:2652-2654), poison, toxic (the counter still
  rises, :1405-1415) and burn damage (subscript_poison_damage.s:5, _burn_damage.s:5), Bind
  (subscript_bind_effect.s:5), recoil (subscript_recoil_1_2.s, _1_4.s:6; Struggle's is not
  skipped), Rough Skin on its holder (battle_lib.c:4259), Life Orb, Black Sludge, Sticky Barb and
  the weather chip (already read), and every entry hazard (subscript_hazards_check.s:5).
- B4-12 Roost (subscript_roost.s): `Divide(maxHP, 2)` through subscript_recover_hp.s ("HP is
  full!" is NO_MORE_WORK: no flag, no AfterMoveEffects), then `turnFlags.roosting`, which only
  `BattleSystem_ApplyTypeChart` reads (`BasicTypeMulApplies`, battle_lib.c:2540-2543: every row
  whose defending type is Flying is skipped, read as the never-listed Mystery type,
  `damage.chart_types`) for the rest of the turn; STAB, grounding and hazards still see Flying.
  turnFlags live within `step`'s turn (the first actor's `roosted` event), as `protecting` does.
- B4-13 Tailwind (subscript_tailwind_start.s): both TAILWIND bits, a counter of 3 counted down by
  the field pass (bcp.c:1039-1058), x2 Speed (battle_lib.c:1274-1276); up already is
  MOVE_STATUS_FAILED (no AfterMoveEffects). The end-of-turn order reads it (the turn's own order
  was fixed before).
- B4-14 Spikes (max 3) and Toxic Spikes (max 2) add a layer to the defender's side, else fail
  (battle_script.c:5394, :7598). subscript_hazards_check.s on every switch-in (switch, drag,
  replacement): nothing for Magic Guard (:5); a grounded newcomer (Gravity, or no Levitate
  (CheckAbility) and no Flying type, :6-11) meets Toxic Spikes (a Poison type absorbs them;
  one layer poisons, two badly poison, stopped by its own Immunity, Leaf Guard in sun, a status,
  Poison / Steel) then Spikes (`Divide(maxHP, (5 - layers) * 2)` with HP left); then Stealth
  Rock on any newcomer with HP (`effects.entry_hazards`).
- B4-15 Destiny Bond (`destiny_bond`, a volatile): set by subscript_destiny_bond.s:9 (no failure
  on repeats), cleared at the holder's next `CHECK_STATUS_START` (bcp.c:2425) whatever the
  checks find; a defender that fainted in the attacker's action with the flag takes a living
  attacker of the other side down after the hits (LOOP_FAINTED, bcp.c:3820-3823;
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
- B4-18 Counter, Mirror Coat and Metal Burst (Kaizo's Bide is effect 227: Metal Burst, Dark,
  single target) read the user's `turnFlags` damage taken this turn (bcp.c:3413-3434: each UpdateHP
  assigns battleCtx->damage after the endure caps, by moveCur's class and to lastDamageTaken; a
  multi-hit leaves its last hit; Pursuit's subscript records none) as the first actor's events:
  2x physical / 2x special / `last * 15 / 10` (battle_script.c:4627, :4680, :7166); 0 taken or a
  fainted attacker is MOVE_STATUS_FAILED; SYSCTL_IGNORE_TYPE_CHECKS (no STAB or multiplier, the
  immunity stays, none for a RANGE_USER move), no crit or roll, no resist berry. A
  RANGE_SINGLE_TARGET_SPECIAL one is its own defender until its command, so it never meets the
  no-target check and writes the foe's moveHit only on success.
- B4-19 A trainer's battle terrain is its first battle's: a split's fight is the first encounter,
  and a Vs. Seeker rematch is not a split fight (manager's decision). So a rematch row's base keeps
  its own facings and only a rematch-only id is narrowed with the spun facings (`tools/terrain.py`
  `narrowed_terrains`). Lass Kaitlin (323): LOOK_SOUTH, sight 6, her sight and talk tiles are
  GRASS. Kept fact: `NoUniqueRematches(TRAINER_LASS_KAITLIN)` (vs_seeker.c:252) rematches her as
  323 once beaten, spun (:659-773), and that rematch can start on PLAIN tiles. Mixed trainers
  217 -> 212 (vanilla), 242 -> 235 (Kaizo). CalcTerrain with its caller is
  field_battle_data_transfer.c:467-505 everywhere. Each `SAFE_COMMANDS` handler is cited
  (`tools/terrain.py` `_SAFE_HANDLERS`); none moves the player (LockAll / LockLastTalked only let
  a step already under way finish, and `MapObject_StepDir` moves the tile at the step's start,
  map_object_move.c:849-858).
- B4-20 The foe's SmellingSalt cure (effect_script_0171.s:12, subscript_heal_target_paralysis.s:5-6)
  frees the side's end speed from paralysis before the end-of-turn sort
  (battle_controller_player.c:888): `enumerate` `para_free`.
- B4-21 Sturdy against an OHKO move (BtlCmd_TryOHKOMove, battle_script.c:4340-4341,
  MOVE_STATUS_STURDY, no roll) is revealed to the AI: subscript_missed.s:67-69 prints
  "{0} was protected by {1}!" (TAG_NICKNAME_ABILITY, BattleAI_SetAbility,
  battle_script.c:11798-11806) unless protected (:18), Levitate (:34) or ineffective (:59) prints
  first (`step._use_move`, `hooks.OHKO_BLOCKED`).
- B4-22 Immunity, or Leaf Guard in sun without a weather lock (CheckAbility, not ignorable), stops
  Toxic Spikes' poison before the status and type checks and prints "{0}'s {1} prevents
  poisoning!" (subscript_poison.s:5-10, :80-82; subscript_badly_poison.s:5-10, :93-103): the AI
  learns the ability (`effects.entry_hazards`).
- B4-23 Counter, Mirror Coat and Metal Burst read the turnFlags last attacker, stored as
  `1 + battler` with the amount (bcp.c:3413-3434), and fail on 0 taken, an attacker of the
  user's side or an attacker with no HP (battle_script.c:4619-4625, :4672-4678, :7157-7162):
  `phys_from` / `spec_from` / `last_from` events (convention 3).
- B4-24 No Counter-family move is RANGE_USER in either game (Counter, Mirror Coat, Metal Burst:
  RANGE_SINGLE_TARGET_SPECIAL; Kaizo's Bide: RANGE_SINGLE_TARGET), so the chart's RANGE_USER skip
  (bcp.c:2352) is never met: the branch is removed from `effects.play_hit` and the reference.
  `damage.script_power_mul` takes the defender (convention 1).
- B5-1 Sleep Talk, Metronome and Copycat (`EffectId.SLEEP_TALK` / `METRONOME` / `COPYCAT`) run
  the move their script picks through GoToMoveScript (battle_script.c:2111-2134) as Nature Power
  does: from the accuracy check on everything reads the called move (moveCur: Protect, the
  chart, damage, effects, moveHit, the Protect chain); the status checks, PP, priority,
  movePrevByBattler, the battle's movePrev and the Choice lock stay the invoker's (moveTemp); a
  called move spends no PP. Their own defender is the user (RANGE_USER), so the no-target check
  passes and the foe's moveHit is written only when the called move's re-picked defender is the
  foe. Sleep Talk (effect_script_0097.s:5) fails unless the user is asleep now (one that woke
  this turn fails); BtlCmd_TrySleepTalk (:4907-4937) draws uniformly (65536 % 4 == 0) over the
  slots not `MoveFlag.NO_SLEEP_TALK` (Move_IsInvoker, Focus Punch, Uproar, Chatter, and
  Move_IsMultiTurn by the move's decomp effect, :4918-4925; battle_lib.c:2975-2992, :3032-3045)
  and not flagged by CheckInvalidMoves without NO_PP (disabled, power 0 while taunted, a Choice
  holder's other moves; :4927); none fails the move. `Outcome.call` holds u in 0..11 and the
  call is allowed slot `u % count` of the set the move sees when it runs (12 is a multiple of
  every count 1-4). Copycat (BtlCmd_TryCopycat :7328-7341) reads `BattleState.move_prev`
  (battle-wide `movePrev`) at its turn and fails on MOVE_NONE or a move of sCannotMetronomeMoves
  (every invoker is one). movePrev is set on every acting path by UpdateMoveBuffers
  (bcp.c:3936-3942) and UpdateFlagsWhenHit (:4499-4508): moveTemp when the attack message
  showed, else MOVE_NONE; Pursuit's pre-switch hit (battle_script.c:6931-6935), switches and
  items leave it. A failed call is MOVE_STATUS_FAILED ("But it failed!", subscript_missed.s).
  The id lists are code constants in both games (rom_diff.md: the functions are unchanged; Kaizo's
  renamed Snatch / Covet / Trick / Switcheroo slots stay excluded), `MoveFlag` bits at table
  build, checked against the decomp (tests/tools/test_effect_map.py).
- B5-2 Metronome (BtlCmd_Metronome :4535-4562) draws `RandNext() % 467 + 1` until the move is
  not the user's and `Move_CanBeMetronomed` allows it: candidate m weighs 141 (m <= 65536 % 467 =
  156) or 140 of 65536 (`order.metronome_weights`); `Outcome.call` holds its rank. Modelled in both
  engines, the sampler and the enumerator (one call variant per candidate; the set cannot change
  within a turn). Coverage refuses a holder while any candidate would be refused as a move of it
  (`coverage.metronome_calls`); the terrain, Kaizo ROM-change and halved-Defense checks run over
  the candidates too (`fight_calls`). Today 150 (vanilla) / 128 (Kaizo) of the 442 candidates
  have unsupported effects, so every Metronome stays refused (supersedes B4-2's counts); no
  partial set is sampled.
- B5-3 The call laws are read at the turn start (`enumerate.call_laws`, `call_variants`,
  `enumerate_turn`; `outcome.sample_outcome` draws the same): each variant is one
  `outcome_laws(..., call)` enumerated and weighted, the rows carrying their `Outcome.call`; the
  variants together are held to `WIDTH_CAP`. Copycat draws nothing: its two orders (its foe used a
  move before it, or not) are variants that keep only their consistent `speed_tie` branches. The
  only implemented change of Sleep Talk's set within a turn is a faster foe's Disable
  (CheckInvalidMoves, battle_lib.c:2266-2269), which the turn-start laws cannot follow (the
  called move's own draws are laid out at the start): a Sleep Talk user facing a reachable Disable
  (a Disable move, a Copycat with Disable in the fight, a Metronome that can call it) is refused
  (`coverage.call_disable_line`; prep's pool and problem per candidate), as Trace in both
  parties is (B3-14). **Refusal superseded by B5-13** (the Disable is modelled). The engines play the live set exactly. The sampler and the laws read
  Copycat's call by the order (`order.used_moves`).
- B5-4 Uproar (`EffectId.UPROAR`, effect 159 in both games, the move Uproar only): a plain hit
  (CalcCrit, CalcDamage; effect_script_0159.s:9-10) whose ON_HIT indirect effect, for a user not
  in an uproar yet (:5), runs when the hit was not `MOVE_STATUS_NO_EFFECTS` (battle_lib.c:1524-1529:
  not missed, protected or ineffective): subscript_uproar.s sets `moveLockedInto` = moveCur (:5),
  `Random 3, 3` (3-6, `Outcome.duration`, the Bind law `DURATION_TRAP`, :6-8) into
  `VOLATILE_CONDITION_UPROAR` (`BattleState.uproar`, per battler) and the battler's
  `FIELD_CONDITION_UPROAR` bit (:9-11, condition.h:118, :147), which is set and cleared with the
  count and so is `uproar != 0`. The first use spends PP; later turns spend none
  (`Battler_CanPickCommand` FALSE, battle_lib.c:6224-6235, sets `ppDecremented` at selection,
  bcp.c:335).
- B5-5 The lock: `Battler_UnlockMoveChoice` (battle_lib.c:1834-1841) clears only
  `VOLATILE_CONDITION_MOVE_LOCKED` (and Bide, Rollout), never the Uproar count nor
  `moveLockedInto`, so every disruption (flinch, Disable, confusion, paralysis, love, Truant, a
  freeze, a miss, Protect, no target) leaves an uproarer locked: `effects.unlock_move` keeps
  `locked_move` while `uproar` is up. `FightCommand` runs `moveLockedInto` whether or not the
  battler knows it (bcp.c:1885-1888; `Battler_SlotForMove` finds no slot and nothing is spent,
  :2272-2292), so a lock a call made (Copycat calling Uproar, Bide or Rollout: moveLockedInto is
  moveCur, battle_lib.c:1828-1832) is still a lock: `actions.locked_slot` gives action 0 then
  (the reference's `_locked_slot` too). This also fixes Copycat's Bide / Rollout locks, which
  the engines read as no lock before.
- B5-6 No sleep in an uproar: a sleeper's status check wakes it at once while
  `FIELD_CONDITION_UPROAR` is set (a faster uproar of this turn included), without the counter
  drop, and the checks go on (bcp.c:2432-2439, subscript_wake_up.s:19 clears every sleep bit;
  Soundproof, the exception, is an unsupported ability). subscript_fall_asleep.s blocks a sleep
  on a target not asleep yet (:43-45 after Insomnia, Shield Dust and the asleep test; :14-16 for
  a move effect): a direct one prints "kept awake" and sets `MOVE_STATUS_NO_MORE_WORK` (:_292,
  :_330: no AfterMoveEffects, `step`'s `kept_awake`), an indirect one (Secret Power's) ends
  silently (`effects._inflict`). Rest and Yawn (unsupported) read the bit too (Owed).
- B5-7 The turn end: `MON_COND_CHECK_STATE_UPROAR` (bcp.c:1505-1541) runs after Bind (Bad
  Dreams, between, is unsupported) and before Disable (Thrash, between, is unsupported): for a
  battler in an uproar, every battler asleep with HP left wakes, in battler order, the check
  repeating until none sleeps (:1508-1521); then the count drops by 1 and the uproar ends
  (subscript_uproar_end.s, the volatile and field bits cleared, :1525-1535) when the count
  reached 0 or the battler's `moveFailFlags` of this turn are set (`BattleContext_MoveFailed`,
  battle_lib.c:2808-2820: flinch, Disable, Taunt, the confusion self-hit, full paralysis, love,
  bcp.c:2531-2681, and `noEffect`: CheckTypeChart found the move ineffective, bcp.c:2350-2370,
  run once the move passed the no-target check, hit or miss, :3271-3277); sleep, freeze, Truant,
  a miss and Protect set none. Ending frees the command (`locked_move` 0: no MOVE_LOCKED was
  set). The flags are the turn's `move_failed` event (`step._use_move`), cleared every turn
  (`SetupNextTurn`, battle_lib.c:2232). Neither check changes HP, so Uproar and Disable share
  one faint check (`step._uproar_then_disable`). The player sees the uproar ("caused an
  uproar!", "calmed down."): `uproar`'s non-zero-ness is in the observation.
- B5-8 Future Sight / Doom Desire (`EffectId.FUTURE_SIGHT`, effect 148: vanilla 80 / 90 % and
  120 / 85 %, Kaizo 120 / 100 % and 250 / 100 %, no `MOVE_FLAG_CAN_PROTECT` in either): once the
  move passed the no-target check, its script's `BtlCmd_TryFutureSight` (battle_script.c:
  6065-6100) fails ("But it failed!", subscript_but_it_failed.s: `MOVE_STATUS_NO_MORE_WORK`, no
  AfterMoveEffects, no miss shown) while the defender's battler has a count; else it sets the
  defender side's `SIDE_CONDITION_FUTURE_SIGHT` (`future_sight`) and, on the defender's battler,
  3 turns, moveCur, the attacker (`1 + battler`) and the damage of that moment:
  `BattleSystem_CalcMoveDamage` with criticalMul 1 and the defender side's and the field's
  conditions (:6079-6089: no crit, no type chart, no STAB, no Life Orb), then
  `BattleSystem_CalcDamageVariance` (:6090, battle_lib.c:7081-7094: `Outcome.roll[side, 0]`;
  the + 2 keeps it nonzero, so its sign never flips); Helping Hand is unsupported. The script then
  sets `SYSCTL_NONSTANDARD_ACC_CHECK` and `SYSCTL_IGNORE_IMMUNITIES` (effect_script_0148.s:7): no
  accuracy roll (bcp.c:2911-2913), no type chart (:2354, so no `noEffect`), and the use deals
  nothing (no CalcDamage: `ids.UNROLLED_EFFECTS`, `damage.fixed_damage`); its ON_HIT effect only
  prints. The fields are field state: they stay when either battler leaves.
- B5-9 The hit: `SIDE_COND_CHECK_STATE_FUTURE_SIGHT` (bcp.c:1758-1783), the first side-condition
  check, after `CheckMonConditions` (:1733) and before `TurnEnd`: per battler in monSpeedOrder
  (the end-of-turn order) with a count, the count drops; at 0 a battler with HP clears its
  side's bit and runs subscript_future_sight_damage.s: `CheckMoveHit` (:8; battle_script.c:
  6115-6135, bcp.c:217-221) of the battler in the stored attacker's slot now (stages, abilities,
  items: whoever is there, a fainted one included) against the defender with the stored move, an
  ordinary check (no SYSCTL flag is up at the turn's end; Zoom Lens applies, every battler has
  moved, battle_lib.c:5365-5368); a new roll, `Outcome.future_hit[battler]`
  (`outcome.future_threshold`). Protect on that defender this turn stops a CAN_PROTECT move
  (none) and unlocks the attacker's battler (bcp.c:3005-3011); else either battler's No Guard
  clears the miss (:3016-3021). A miss is "But it failed!" (:53-59). Else Substitute and Rage
  (unsupported), `CheckHoldOnWith1HP` (:28: a full-HP Focus Sash leaves 1 HP,
  battle_script.c:9098-9124; Focus Band is unsupported), UPDATE_HP with no resist berry
  (`waitingBattlers` is 0 after the actions, bcp.c:881-886; subscript_update_hp.s:9-10), then
  under `MOVE_STATUS_ENDURED_ITEM` the Focus Sash's removal (:40-48). No held-item check follows.
- B5-10 One pass, one set of flags: `moveStatusFlags` is cleared only by `BattleContext_Init`
  (battle_lib.c:1983; the last MoveEnd, bcp.c:3998), so when both battlers' hits fall due, the
  second reads the first's: a first miss or Protect fails the second (No Guard still clears a
  carried miss), and a first endured hit removes the second defender's Focus Sash. Between the
  two, `AnyFainted` (:1749, `onlyFaint`) faints the first defender without `CheckBattleOver`,
  so the second hit still lands: two last Pokemon down is a draw, a loss (ruling 2); `TurnEnd`
  pays the exp and checks the end (:1844-1847). A battler with no HP when its count runs out
  takes nothing and keeps its side's bit, which only a later hit clears (`future_sight`).
- B5-11 The next turn's `ShowBattleMon` (bcp.c:256-268) runs `BattleSystem_TriggerEffectOnSwitch`,
  whose `SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS` (battle_lib.c:4151-4165) checks every battler's
  item before the commands are chosen; a berry the Future Sight hit brought into range (it comes
  after `MON_COND_CHECK_STATE_ITEM_CONDITION`) is eaten there (`step._turn_start_items`, at the
  end of `step`; with a replacement pending, `replace`'s switch-in check runs it after the
  newcomer's hazards, as `ReplaceFainted` precedes `ShowBattleMon`). Without such a hit it finds
  nothing new: every earlier change of the turn's end was followed by an item check.
- B5-12 Laws: the use draws `roll[s, 0]` (the 16-value law; its "move" read needs only that
  the move runs) when the target's battler has no count at the turn start (only
  the user's own Future Sight can set it before its move); no `hit` or `crit`. `future_hit[d]` is
  read when `future_timer[d]` is 1 at the turn start; its law is the 1-100 roll as classes over
  `outcome.future_thresholds`: every attacker / defender battler the slots can hold at the turn's
  end (the active, its switch target, any living one a foe's Roar can drag), every stage sum when
  an accuracy / evasion stat move, Foresight or a White Herb is in play (else the start stages or
  0), the weather of the start, none, sandstorm or hail, the target confused or not; a Trace
  holder among them keeps every roll (`future_loose`). `sample_outcome` draws the same.
- B5-13 (supersedes B5-3's refusal) Sleep Talk facing a faster Disable is modelled, not refused.
  `BtlCmd_TryDisable` (battle_script.c:4581-4598) disables only the defender's movePrevByBattler
  (:4586), only while that move has PP and nothing is disabled yet (:4588-4591). A Sleep Talk
  user that moves second has not moved this turn, so the move is its turn-start `last_move`, and
  its set at its move is either the turn-start set or that set without the last move's slot
  (`enumerate._disabled_state`; CheckInvalidMoves, battle_lib.c:2266-2269, through
  `order.sleep_talk_slots`). `call_laws` gives `alt`, each value's call from the second set,
  where it can arise (nothing disabled, a last move, the foe uses a move: any move can be an
  invoker calling Disable). `call_variants` splits a value group whose two calls differ into two
  variants of the group's weight, `CALL_KEEP` (value + 12) and `CALL_DISABLED` (value + 24);
  `outcome_laws` lays out the second one's reads from the shrunk set. `order.invoked_move` takes
  `call % count` (count 1-4 divides 12 and 24), so `step` plays every encoding of a value alike, on
  the live set, and the sampler still draws 0-11. Each row is kept only in the variant whose
  assumption its turn met: `step`'s new `disabled_at_move` event (a move of the battler disabled
  when its action starts) against the variant (`enumerate.call_consistent`; `expand`'s step
  program returns it and the merger drops the rest, `enumerate_turn` filters). Removed:
  `coverage.call_disable_line` and its calls in `check_coverage`, `prep/pool.py` and
  `prep/problem.py` (so the review's missing `player_extra` there is moot). The brute force
  (`tests/engine/test_invokers.py::_raw_brute`) no longer shares `branch_ok` / `after_foe`: it
  loops over the raw `speed_tie`, takes the order from `order.action_order` as `step` does,
  derives Copycat's movePrev from it (its foe used a move before it), forces every branch open
  and cuts the rows to that `speed_tie`; the hand cases add a faster Disable and a foe that
  switches before Copycat.
- B5-14 Zoom Lens on the Future Sight hit (`outcome.future_threshold`, reference `_hit_rate(...,
  True)`): no change. `Battler_MovedThisTurn` (battle_lib.c:5365-5368) reads `battlerActions[d]
  [PICK_COMMAND] == BATTLE_CONTROL_MOVE_END`; every action's MoveEnd sets it for the actor's slot
  (bcp.c:3989), a switch by choice included, `UpdateAfterSwitch` sets it for a battler dragged in
  by Roar / Whirlwind (battle_lib.c:2053) and `PlayFaintAnimation` for a fainted one
  (battle_script.c:1801); only the next command selection writes it again (bcp.c:337-512). So at the turn end's
  check (bcp.c:2963-2965) every battler has moved, a newcomer included
  (`tests/engine/test_batch5_review.py`).
- B5-15 Generator (`tests/strategies.py`): with an uproar under way no active Pokemon sleeps at
  the turn start (B5-7's turn-end wake-up), also after the other side's draws.
- B6-1 (departure from the design's form) The turn scratch is flat `BattleState` fields
  `ts_*` (`state.TURN_SCRATCH`), not a nested `TurnScratch` field: `LAYOUT`, the reference state
  and the converters stay flat. One field more than the design lists: `ts_move_failed`
  (moveFailFlags, BattleContext_MoveFailed battle_lib.c:2808-2820), which the turn end's Uproar
  check reads (bcp.c:1525) after the resume. `ts_custap` keeps only a later actor's flag (an
  earlier one ate its berry, subscript_check_quick_claw.s:26-28); `ts_item` is 0 for a non-item
  action.
- B6-2 U-turn (`EffectId.U_TURN`, effect 228 in both games): its ON_HIT secondary effect runs at
  ONE_HIT_TRIGGER_SECONDARY (bcp.c:3484-3494) for a hit without MOVE_STATUS_NO_EFFECTS
  (battle_lib.c:1524-1529; the engine's `MoveHit.effective`). subscript_attack_then_switch_out.s:
  the defender side not blacked out (:5, BtlCmd_CheckBlackOut battle_script.c:8846) and a
  replacement for the user (:6, battle_lib.c:3125); the contact ability (:7, the hit's own:
  every TriggerAbilityOnHit case is guarded by SYSCTL_UTURN_ACTIVE, battle_lib.c:4219-4362, and
  the later call needs a living attacker when no switch happened, so it runs once); the pivot
  item (:11, `effects.pivot_item`); Destiny Bond (:15-34); a fainted user switches nothing
  (:48) and the action goes on, King's Rock included (CheckExtraFlinch has no attacker HP test,
  bcp.c:4630-4650); the foe's Pursuit (:54, BtlCmd_TryPursuit battle_script.c:6884-6955, the
  pursuer not yet acted: `_pursuer_ready`) with its faint check and payout
  (subscript_pursuit.s:106-108); a user alive after it (:55) gets its Natural Cure (:56-57) and
  the list (:71). A battle the Pursuit's faint check ends opens no list (the engine ends a
  battle at the faint check that finds a side out, `step._faint_check`).
- B6-3 Baton Pass (`EffectId.BATON_PASS`, effect 127): a direct ON_HIT effect at
  BATTLE_CONTROL_PRIMARY_EFFECT (bcp.c:3298-3308); subscript_baton_pass.s: no replacement fails
  (:5, :19-20: MOVE_STATUS_FAILED, subscript_missed.s, no AfterMoveEffects), else Natural Cure
  (:7-8) and the list (:17); no Pursuit. The switch (UpdateAfterSwitch under SYSCTL_BATON_PASS,
  battle_lib.c:2056-2086, :2105-2115; InitBattleMon :97-101) keeps of the engine's fields the stat
  stages, `confusion` (condition.h:72), `leech_seed`, `mud_sport`, `water_sport`, `aqua_ring`
  (moves.h:46-48) and `substitute_hp` (:2107) (`step._BATON_PASSED`); Focus Energy, Mean Look,
  Curse, Lock-On, Perish Song, Ingrain, Power Trick, Embargo, Heal Block, Magnet Rise and Gastro
  Acid come from unsupported moves (owed with them).
- B6-4 The resume (`step(paused, actions)`, `actions[pivot]` the switch to the pick): the list's
  SwitchAndUpdateMon, PokemonSendOut, the hazards and the faint check (subscript_show_party_list.s:
  9-24): that check plays the faint of a defender the U-turn KO'd (its SYSCTL_MON_FAINTED bit is
  set at UpdateHP, BtlCmd_TryFaintMon battle_script.c:1763-1777) before the tail; its exp is paid
  at MoveEnd (AnyExpPayout, bcp.c:3968). `ts_alive` is the party bitmask of the faints handled
  so far (the action's start, or after U-turn's Pursuit), the `before` of the list's faint check
  and of the MoveEnd payout. Then the tail on the newcomer: King's Rock with its item, the
  switch-in effects at TRIGGER_SWITCH_IN_EFFECTS (bcp.c:3672-3682), the item checks and the thaw
  (the guarded Sticky Barb, battle_lib.c:5393, Shell Bell bcp.c:4739 and Life Orb :4761
  skipped), UpdateMoveBuffers on the newcomer (bcp.c:3917-3944: movePrevByBattler, movePrev and
  moveProtect written after UpdateAfterSwitch cleared them, battle_lib.c:2119-2123), MoveEnd.
- B6-5 The Choice lock (bcp.c:3921-3934) tests `moveTemp`: U-turn under SYSCTL_UTURN_ACTIVE (set
  at :68, so only a resumed U-turn) and Baton Pass under SYSCTL_MOVE_SUCCEEDED (a Baton Pass whose
  attack message showed, a failed one included) never lock; a Sleep Talk that calls U-turn locks
  the newcomer into Sleep Talk when it holds a Choice item.
- B6-6 (departure) The resumed half reads its own draws in the sampler too: `sample_outcome` and
  the laws read a paused state through `outcome.pivot_view` (the pivoting side switches to its
  pick first, the newcomer on the field, movePrev the paused move's moveTemp, the other side's
  action as chosen while it is still to act), and the paused action's King's Rock
  (`flinch_item[pivot, 0]`, with the newcomer's item param) is added by `outcome_laws`. The
  design's "playout loops `step` with the same Outcome" would draw the newcomer's King's Rock
  with the outgoing battler's param (`kings_rock_chance` reads the active's item):
  `playout.step_through` draws a fresh outcome of the paused state for each resume.
- B6-7 The search: `stage_b`'s pending kind `"pivot"` (before learn and replacement: a pause is
  mid-turn); `decision_actions` offers the player's `legal_replacements` at its own list and the
  placeholder action 0 at the opponent's; an answer is the turn request `(state, answer, 0)`
  through `expand` (a chance node), the opponent's pick made in the row program
  (`playout.turn_inputs`: `post_ko_switch_in` with `ts_pick_mul`, refused as a replacement's;
  `aiSwitchedPartySlot` is 6 there, `Task_TrainerShowPartyMenu` battle_display.c:4543-4556) and
  the turn's powerMul and AI switch read from `ts_power_mul` / `ts_switched`. A paused child
  runs no learnset scan and no replacement. A playout from a paused root resumes it first
  (`playout._play`'s `pick`), not counted as a turn; `leaves`' forced turns go through
  `turn_inputs` and `step_through` too.
- B6-8 A locked battler never Struggles (it picks no command, bcp.c:335, so the STRUGGLING_ALL
  test of :433 is not reached): `step` now masks `selected_move`'s struggling with the lock, as the
  reference engine's `_choose` already did (a value the scratch keeps).
- B6-9 `step._execute_actions` is a `while_loop` over (position, phase): a U-turn that reaches its
  foe's Pursuit stops after its head, the next pass plays the Pursuit at the one Pursuit site (a
  switch's pre-switch hit shares it), then the action's tail (`_use_move`'s `MODE_PURSUIT`); a
  list ends the loop with the scratch written. `_use_move` and `_act` stay one trace each:
  `MODE_PURSUIT` / `MODE_RESUME` drop the head's state at the boundary before the hit loop and
  read the paused inputs (`ts_used`, `ts_damaged`, `ts_dealt`) from the scratch; Roar's drag and
  the resume share one `hooks.switch_in_effects` site.
- B6-10 (batch 6 review) The resumed half's laws read the newcomer as the switch leaves it:
  `outcome.pivot_view` applies the pivot's switch to the viewed state (its stat stages and
  `state.VOLATILE_FIELDS` reset, a Baton Pass keeping the stages and `BATON_PASSED_FIELDS`:
  InitBattleMon battle_lib.c:97-101, UpdateAfterSwitch :2078-2079, :2107), and
  `outcome.fresh_switch` tells the thresholds (`_hit_threshold`, `hit_thresholds`, `_rate`'s
  confusion for Tangled Feet and Foresight) that this switch's battler is not fresh: its evasion is
  the view's, as `step`'s live check reads it (bcp.c:2873-2874). `future_thresholds` reads the
  view's stages too. Before, a Baton Pass's +2 evasion was read as 0 (a foe Tackle's law offered
  95 where `step` reads 57). Test: tests/engine/test_pivot_review.py (every enumerated row stepped
  against `live_hit_threshold`).
- B6-11 (batch 6 review) Stage B groups children by `(observation, refused, paused)`: the
  observation does not carry `pivot_battler`, so a group must not mix a paused state with a
  finished one (`_paused`).
- B6-12 (batch 6 review, the doubles conventions) `ts_acted` (per battler, Battler_MovedThisTurn
  battle_lib.c:5365-5368: the actions before the paused one, MoveEnd bcp.c:3989) replaces
  "`ts_cursor` is 0" as "the target has not moved" in `pivot_view` and `enumerate._paused_rock`;
  `ts_switched` is per battler (was `ts_ai_switched`, side 1 only); `decision_actions` reads the
  pivot's side (1 + battler, battler b on side b in singles), not "2 means the opponent"; `_act`
  takes the battler U-turn's Pursuit reads (`foe`) from its caller.
- B6-13 (batch 6 review) Hand tests of both engines (tests/engine/test_pivot_review.py): Destiny
  Bond on U-turn (subscript :15-34), Rough Skin KO-ing the user (:48, `rock_now`), a U-turn KO with
  the list's faint check and the MoveEnd payout, hazards KO-ing the newcomer, two pauses in one
  turn, Pursuit KO-ing the user, Sleep Talk calling U-turn with a Choice newcomer (B6-5), Baton
  Pass without a bench, Natural Cure at the list, Shell Bell and Sticky Barb at the pivot item
  (battle_lib.c:7336-7386), Roar's drag: its hazards (subscript_force_target_to_switch_or_flee.s:31), then its
  switch-in effects at AfterMoveEffects (bcp.c:3672-3682), the site the resume shares. The generator: a drawn pivot half the time
  gets a living bench and a user free of sleep, freeze, confusion and love.
- B7-1 `ai_item_known` (aiContext.battlerHeldItems, per player party slot, the item id):
  `BattleMessage_ItemTag` (battle_script.c:11726-11745) runs `BattleAI_SetHeldItem`
  (:12119-12122) for an item tag of `BTLSCR_ATTACKER`, `_DEFENDER` or `_MSG_BATTLER_TEMP`, never
  for `_MSG_TEMP` (msgItemTemp), nor for a message built in C (Knock Off :6622, the pinch berries'
  stat message :2857, the Choice refusal battle_lib.c:2362). Of the modelled items that is
  Leftovers and Black Sludge at the turn end (subscript_restore_a_little_hp.s:10,
  subscript_lose_hp_from_item_with_message.s:10; battle_lib.c:4955), Sticky Barb's damage
  (:5340) and Shell Bell's heal (bcp.c:4745, battle_lib.c:7352). Not revealing: Quick Claw (no
  message, subscript_check_quick_claw.s:6-10), Life Orb (subscript_lose_hp_from_item.s), the
  orbs (subscript_badly_poison.s:77, subscript_burn.s:73), every berry's cure or heal, Frisk
  (subscript_frisk.s:7). Custap (:17), Focus Sash (subscript_move_followup_message.s:16,
  subscript_future_sight_damage.s:44), the resist berries (subscript_type_resist_berry.s:113) and
  Pluck's victim (subscript_pluck.s:7) are named, then removed (B7-2).
- B7-2 The record is cleared by `BattleMon_CopyToParty` of a battler holding nothing
  (battle_lib.c:1819-1823), which every removal runs: `RemoveItem` (battle_script.c:8979-8981;
  the berries through subscript_pluck_check.s) and the Sticky Barb transfer's `UpdateMonData`
  (:3077-3078, subscript_transfer_sticky_barb.s:7); and on entry (`UpdateAfterSwitch`,
  battle_lib.c:2153). `CleanupFaintedMon`'s clear (:2225) is played at the replacement's entry,
  as for the known moves and ability: nothing reads a fainted player battler's record before.
  Only the player's is kept: the AI reads `battlerHeldItems` for the other side only
  (trainer_ai.c:1907-1913, :1924-1928).
- B7-3 Ingrain (effect 181, effect_script_0181.s): sets `MOVE_EFFECT_INGRAIN` (:5-6); heals
  `Divide(maxHP, 16)` (Big Root applied) at `MON_COND_CHECK_STATE_INGRAIN`, before Aqua Ring
  (bcp.c:1311-1326; one faint check serves both heals); traps (`Battler_IsTrapped`,
  battle_lib.c:5523, the party menu bcp.c:572; `actions.trapped`); grounds: `BasicTypeMulApplies`
  skips an immune Flying row (:2534-2537, `damage.CHART_GROUNDED`) and `Battler_Ability` reads its
  Levitate as none (:3100-3103, `damage.ignorable`). Baton Pass keeps it (moves.h:46).
- B7-4 Ingrain, Aqua Ring, Mud Sport and Water Sport fail a user that has the effect already
  in their effect script (effect_script_0181.s:5, _0251.s:5, _0201.s:5, _0210.s:5:
  MOVE_STATUS_FAILED): the move does not connect, CheckMoveFailure prints "But it failed!" and
  skips AfterMoveEffects (bcp.c:3312-3331), as Fake Out's. Aqua Ring and the sports were played
  as a landed move with no effect; both engines now take Fake Out's path.
- B7-5 Roar's Ingrain test (subscript_force_target_to_switch_or_flee.s:7) is `OPCODE_EQU`: only a
  `moveEffectsMask` equal to `MOVE_EFFECT_INGRAIN` anchors the defender (:86-90,
  NO_MORE_WORK); beside any other kept bit (Leech Seed, Lock-On, Perish Song, Yawn, Imprison(ed),
  the sports, Aqua Ring, Camouflage) Roar drags it out (`effects.ingrain_anchors`).
- B7-6 Mean Look, Block, Spider Web (effect 106): subscript_mean_look.s runs as a direct
  `CHECK_HP | TO_DEFENDER` effect for a living defender on a move that connected (battle_lib.c:
  1491-1504; accuracy 0, so only Protect stops it); it fails (MOVE_STATUS_FAILED, no
  AfterMoveEffects) on a defender already trapped or behind a substitute (:7-8), else sets
  `VOLATILE_CONDITION_MEAN_LOOK` and ORs the attacker into `meanLookTarget` (:9-10), stored as
  `1 + battler` (in singles the OR always holds the one foe). It traps
  (`VOLATILE_CONDITION_TRAPPED`, condition.h:77-78). It ends when the trapper leaves by a switch
  that is not a Baton Pass (battle_lib.c:2055-2061) or faints (:2170-2172); Baton Pass keeps it
  on both sides (:2078, :2111).
- B7-7 Imprison (effect 192, `BtlCmd_TryImprison` battle_script.c:6674-6727): the foes'
  `MOVE_EFFECT_IMPRISONED` is set first, whenever the script runs (:6686-6690), and is cleared only
  with that battler's volatiles; the script fails a user already imprisoning or sharing no move
  with a battler of the other side, whatever its HP (:6692-6722). A sealed move
  (`Move_Imprisoned`, battle_lib.c:3316-3338) is invalid at selection (CHECK_INVALID_IMPRISONED,
  :2287-2290; Sleep Talk's slots too, battle_script.c:4927) and, chosen before a faster foe's
  Imprison, fails at `CHECK_STATUS_STATE_IMPRISON` after Taunt (bcp.c:2572-2583): unlock, no
  PP, `moveFailFlags.imprisoned` (the Uproar check, battle_lib.c:2812), event `ACT_IMPRISONED`.
- B7-8 `roosting` (turnFlags.roosting, battle_context.h:64) replaces `ts_roosted`: a Roost that
  healed sets it (subscript_roost.s:11-12); a faint clears it (battle_lib.c:2189-2192) and
  `SetupNextTurn` clears it (:2229-2231) after the faint replacements (bcp.c:1843-1856), so it is
  kept at a pause and while a replacement is pending (`step.roosting_now`; `replace` clears it
  with the last one), else 0. Per battler slot: a Pokemon dragged in after its side's Roost keeps
  it, as the decomp does.
- B7-9 Observation: the non-zero-ness of `ingrain` ("planted its roots!"), `mean_look` ("can no
  longer escape!"), `imprison` ("sealed the opponent's move(s)!") and `roosting`, and the
  `ACT_IMPRISONED` action kind ("can't use the sealed MOVE!"). `imprisoned` and `ai_item_known`
  are not shown.
- B7-10 Enumerator: Ingrain is an input clause (the foe's chart and Levitate change, like Roost);
  Mean Look and Imprison are input-neutral (no damage input; a sealed move stops whole, like
  Taunt); all three are status moves outside the table damage.
- B7-11 Tests: the differential test and its generator (`strategies._draw_batch7`: the moves, a
  shared move for Imprison, the five fields drawn; Roar against an ingrained foe with and
  without other mask bits; `ai_item_known` none or the held item) reach every branch; no hand
  test was needed beyond the updated order and observation tests.
- B7-12 (fix, from 470fbef) `_act` evaluates the move branch for every action, and a switch
  action (4 + slot) clips to move slot 3: a Baton Pass or U-turn there set `ctl["pause"]` /
  `["pursue"]`, so `_execute_actions` replayed the switch from the newcomer (its traced or other
  ability restored to the personal one, the hazards a second time). The turn branches on the
  picked command (`battlerActions[...][BATTLE_ACTION_PICK_COMMAND]`, bcp.c:893): a switch or a
  bag item runs no move script, so only a move or a resumed action can stop. The reference was
  right. Test: tests/engine/test_batch7_review.py (fails at f22b818).
- B7-13 (batch 7 review) Sleep Talk's slots drop a move a foe's Imprison seals:
  `BtlCmd_TrySleepTalk` runs CheckInvalidMoves with `~CHECK_INVALID_NO_PP` (battle_script.c:4927),
  which includes CHECK_INVALID_IMPRISONED (battle_lib.c:2287-2290); JAX's `order.sleep_talk_slots`
  missed it (the reference had it). A faster foe's Imprison this turn is the set's second change
  within a turn: a third call variant `CALL_SEALED` beside B5-13's KEEP / DISABLED
  (`enumerate._sealed_state`, `CallLaws.sealed`, `can_disable` / `can_seal`), each row kept by
  the turn's own `disabled_at_move` / `sealed_at_move` events relative to the turn start
  (`_split_events`; one foe move a turn, so never both). Tests: test_batch7_review.py, the
  `sleep_talk_imprison` brute-force case (test_invokers.py); the generator gives an Imprison user a
  move a Sleep Talk user can call (`strategies._share_with_sleep_talk`).
- B7-14 (review) A Baton Pass newcomer keeps Ingrain before its hazards (battle_lib.c:2079); the
  hazards' `CheckAbility` (subscript_hazards_check.s:8; battle_script.c:3159-3172) reads
  `Battler_Ability`, which hides an ingrained Levitate (battle_lib.c:3100-3103): the Spikes and
  Toxic Spikes reach it (`damage.grounded(..., ingrained)`, both engines). A switch action's
  newcomer, which the laws read before it switches, has none (`entry_weather`). The Owed line is
  closed.
- B7-15 (review) `meanLookTarget` is ORed (`OPCODE_FLAG_ON`, subscript_mean_look.s:10) and nothing
  clears it but the trapped battler's own switch-in or faint (battle_lib.c:2056-2061, :2170-2172
  clear only the volatile bit): `mean_look_target` keeps `1 + raw` with its stale bits (replaces
  B7-6's "1 + attacker, 0 without the bit").
- B7-16 (review) A Roar whose target is anchored at the turn start and stays so (no switch, no
  move of the target side that adds a mask bit: Aqua Ring, the sports, Imprison, Camouflage, an
  invoker) stops before TryWhirlwind (subscript_force_target_to_switch_or_flee.s:7, :86): its
  `roar_level` / `roar_target` draws are not relevant (enumerate.py).
- B7-17 (review, settled) Imprison is `RANGE_USER` (res/moves/imprison/data.json), its own
  defender (`targets_user`), so its script runs with the foe fainted mid-turn and marks it
  imprisoned, as `TryImprison` loops the battlers whatever their HP (battle_script.c:6697-6716).
  `TrainerAI_CalcDamage` calls `BattleSystem_CalcMoveDamage` (trainer_ai.c:3082), whose Technician
  (battle_lib.c:6698) the AI estimate therefore applies: the merged `damage.py` is exact there.
- B7-18 (AI-side fix by E2 with the manager's go) `TrainerAI_ShouldSwitch`'s gate
  (trainer_ai.c:3906-3907) refuses a switch for `VOLATILE_CONDITION_TRAPPED` (Bind and Mean Look,
  condition.h:77-78) and `MOVE_EFFECT_INGRAIN`; `ai/switching._layout` gated on `trapped` only and
  now reads `mean_look` and `ingrain` too. The rest of the gate (:3908-3911: Shadow Tag and Arena
  Trap on the player's side, Magnet Pull on a Steel AI battler) was already there; the comment at
  :3902-3904 (the AI ignores its own Shadow Tag, a Flying type vs Arena Trap, Shed Shell) is the
  decomp's own naivety, mirrored. Tests: tests/ai/test_switching.py `test_gates`, the brute force
  of tests/ai/test_kaizo_switching.py.
- Opponent-action legality (by design): neither `step` nor `legal_actions` checks the
  opponent's action; the search feeds `step` what the AI picks, and the AI is trusted to pick
  legally because it mirrors the decomp's own gates. The gates that trust rests on: the switch
  gate (`ai/switching._layout`, trainer_ai.c:3906-3913) and its living-bench test (:3924-3938);
  the move scores' usable mask (`TrainerAI_Init`'s CHECK_INVALID_ALL, trainer_ai.c:234:
  `actions.usable_moves` in `ai.commands` / `ai.policy`: no move, no PP, Disable, Taunt, the
  Choice lock, Imprison) with Struggle when none (ai/policy.py:2850, :3057); the locked battler's
  forced move (`actions.locked_slot`); the bag pick (`TrainerAI_ShouldUseItem`,
  `actions.usable_trainer_items`); the replacement pick (`ai.replacement`, a
  `legal_replacements` slot) and the pivot pick at its own list (`post_ko_switch_in`). Where
  `step` would accept an illegal opponent action silently: a switch to an empty, fainted or
  active slot (`_send_out` sends it; no check), a switch while trapped or locked (played as a
  switch), and a move slot the mask forbids: an empty slot runs move 0, a slot with no PP spends
  to -1 (`spend`), a Choice-locked holder's other move runs; Disable, Taunt and Imprison are
  still caught at their status checks (bcp.c:2543-2583), as for a faster foe's this turn. The
  player's actions are checked by the search (`legal_actions`, `decision_actions`).
- B8-1 Lock-On / Mind Reader (effect 94, effect_script_0094.s: a direct `CHECK_HP |
  TO_DEFENDER` subscript_lock_on.s): a living defender without a substitute gets its
  MOVE_EFFECT_LOCK_ON count set to 2 and lockOnTarget = the attacker (:5-9, `OPCODE_SET`, whatever
  it had); a substitute fails it (:6, :33). `MON_IS_LOCKED_ONTO(attacker, defender)`
  (include/battle/common.h:51) clears a miss beside No Guard (bcp.c:3016-3018), in TryOHKOMove's
  bypass (battle_script.c:4343-4361) and at the Future Sight hit (`outcome._rate`, both engines);
  `MON_COND_CHECK_STATE_LOCK_ON` drops the count at each turn end (bcp.c:1618-1624), so a lock
  set this turn serves the locker's move of the next.
- B8-2 Lock-On's leave rules: the locked battler's own count and target are volatile, Baton
  Pass keeping both (moves.h:46, battle_lib.c:2109); when the locker leaves by a switch the lock
  on it ends (battle_lib.c:2063-2067), by Baton Pass it restarts at the initial count
  (:2077-2085, moves.h:55); its faint leaves it (CleanupFaintedMon, :2156-2205, has no such loop)
  until the replacement comes in (corrected by B8-7).
- B8-3 Perish Song (effect 114, a direct ON_HIT subscript_perish_song_start.s):
  `BtlCmd_TryPerishSong` (battle_script.c:5465-5487) marks every living battler without the effect
  (count 3), failing with none (Soundproof is an unsupported ability).
  `SIDE_COND_CHECK_STATE_PERISH_SONG` (bcp.c:1791-1820), after Future Sight's: a battler found at
  0 loses the effect and all its HP (subscript_continue_perish_song.s:6-7), else the count
  drops; the pass's re-entries run only `AnyFainted` (:1749), the battle-over check is TurnEnd's
  (:1845), so `_future_sight` no longer decides the battle mid-pass (two last Pokemon that faint in
  the pass: a draw, played as a loss). Volatile, kept by Baton Pass (battle_lib.c:2110).
  Observation: `perish_count` (1 + the count while up).
- B8-4 Yawn (effect 187, a direct `CHECK_HP | TO_DEFENDER` subscript_yawn.s): an ignorable
  Insomnia or Vital Spirit, or Leaf Guard in sun, makes it ineffective and is tagged (:5-9,
  :25-30); a substitute, an uproar unless Soundproof, any status, Safeguard or a Yawn counting
  fail it (:14-21); else the count is 2 (`TryYawn`, battle_script.c:6580-6590).
  `MON_COND_CHECK_STATE_YAWN` (bcp.c:1684-1697) drops it and at 0 runs subscript_fall_asleep.s
  as a move effect (:4-14): the battler's own (not ignorable) Insomnia / Vital Spirit / sunny
  Leaf Guard keep it awake (tagged, :_202/_221), a sleeper, an uproar or a status stop it
  silently, no Safeguard test; else `Random 3, 2` turns (`Outcome.yawn_sleep`, its own draw and
  law) and the move lock released. Volatile, not kept by Baton Pass.
- B8-5 Curse by a Ghost user (effect 109's Ghost branch, effect_script_0109.s:9-17): its
  defender is the foe (`effects.ghost_user` through `targets_user` / `self_defender` /
  `defender_always`, both engines); subscript_curse_ghost.s fails behind a substitute or on a
  cursed foe (:6-7, :22-24), else curses the foe and costs the user Divide(maxHP, 2) through
  UPDATE_HP (:9-15; it can faint). Curse has no MOVE_FLAG_CAN_PROTECT (res/moves/curse/
  data.json): Protect never stops it. `MON_COND_CHECK_STATE_CURSE` (bcp.c:1451-1459, after Burn):
  Divide(maxHP, 4) unless Magic Guard (subscript_curse_damage.s). Volatile, kept by Baton Pass
  (condition.h:72-76). The coverage / prep refusal (`coverage.ghost_curse`) is removed.
- B8-6 (batch 8 review) The Future Sight hit's law (`outcome.future_thresholds`) read Lock-On as
  it stands at the turn start, but MON_COND_CHECK_STATE_LOCK_ON drops the count in the per-battler
  turn-end checks (bcp.c:1618-1624) before the side-condition pass hits (:1758-1783), and the hit
  reads MON_IS_LOCKED_ONTO (:3016-3018): a lock at 1 is gone at the hit. The grid now has a lock
  axis: unlocked always possible (also covering a locker's switch, battle_lib.c:2063-2067, and a
  pivot view that kept a lock the switch clears), locked by the stored attacker's slot possible
  when the count is 2 or more at the start or a Lock-On is used this turn. Test:
  test_batch8.py `test_future_sight_law_reads_the_lock_after_the_countdown`, its expectation from
  step's live `future_threshold` on the state at the hit.
- B8-7 (batch 8 review) Corrects B8-2's text, the code was right: after the locker's faint the
  lock stays until the replacement comes in; its BtlCmd_SwitchAndUpdateMon runs UpdateAfterSwitch
  (battle_script.c:2522), which without Baton Pass clears every lock whose lockOnTarget is that
  battler (battle_lib.c:2063-2067). Test extended through `replace` (both engines).
- B8-8 (batch 8 review) The opponent-legality invariant checks concrete host-side picks only; the
  AI picks inside jitted playouts (most of them) are traced and unchecked (tests/ai_legality.py
  says so). `test_the_property_reaches_unrefused_states` checks one found state (Ingrain and
  Mean Look on the AI's side); its docstring now says that.
- B9-1 Thunder (effect 152) and Blizzard (260) are a probabilistic TO_DEFENDER paralysis /
  freeze (effect_script_0152.s, _0260.s: `SECONDARY_STATUS`); their weather accuracy is by
  effect id: inside NO_CLOUD_NINE, Thunder's hitRate is 50 in sun before the stage scaling
  (bcp.c:2915-2917), and CheckMoveHitOverrides clears the miss of Thunder in rain and Blizzard in
  hail after No Guard / Lock-On (:3023-3031). Carried as `MoveFlag.THUNDER` / `BLIZZARD`, set by
  build_tables from the move's decomp effect (Kaizo's Ice Ball and Sheer Cold have effect 260).
  Thunder's SYSCTL_HIT_DURING_FLY matters only against Fly / Bounce (unsupported). The hit laws
  take the weather variants they already had (the foe's weather move, a switch-in ability).
- B9-2 Fire Fang (273) and Ice Fang (274; Kaizo's Avalanche) are Thunder Fang's family:
  subscript_burn_or_flinch.s / subscript_freeze_or_flinch.s are subscript_paralyze_or_flinch.s
  with BATTLE_SUBSCRIPT_BURN / _FREEZE (`FANG_PARALYZE_FLINCH` with the status in param 0).
  Fire Fang's Wonder Guard bypass (battle_lib.c:7600) waits for Wonder Guard (Owed).
- B9-3 Speed and damage abilities. Chlorophyll doubles speed in sun beside Swift Swim
  (battle_lib.c:1228-1238). Unburden doubles it with canUnburden and no item (:1268-1272);
  `BattleState.can_unburden` is set by RecoverStatusByAbility's Unburden case for a holder with
  an item (:4433-4437: every move end for attacker and defender, every switch-in check for every
  battler, :4135-4145) and by InitBattleMon at the battle start (:160-166), zeroed by
  UpdateAfterSwitch (:2097-2101, Baton Pass too). Huge Power / Pure Power double attackStat
  before the held items (:6706-6708; so x2 then Choice Band's x1.5). Swarm is Blaze's line for
  Bug (:6834-6838). Flower Gift in sun (inside NO_CLOUD_NINE, the self-hit passes no weather):
  a living holder on the attacker's side gives attackStat x15/10, on the defender's side
  spDefenseStat x15/10 unless the attacker has Mold Breaker (:6930-6941,
  COUNT_ALIVE_BATTLERS_OUR_SIDE :2886-2893); Cherrim's form change (:6366-6392) writes only
  formNum, which only the display reads (battle_controller.c sprite messages). The end-of-turn speed grid
  (`enumerate._end_speeds`) takes Chlorophyll's sun beside Swift Swim's rain and an Unburden
  axis.
- B9-4 Recoil: `RECOIL`'s param 0 is now the divisor of subscript_recoil_1_{2,3,4}.s
  (DivideVarByValue = BattleSystem_Divide, battle_script.c:4413: at least 1), which a percent could
  not give for 1/3: Double-Edge / Wood Hammer (198, effect_script_0198.s with Reckless) join.
  Kaizo's changed be_seq 48 (1/2 recoil, Reckless) and 182 (vanilla 48's script: 1/4 recoil,
  Reckless) are `modelled` in rom_diff.toml through `effect_map.KAIZO_EFFECTS`, which build_tables
  uses for Kaizo in place of the vanilla entry (and checks against the modelled members). Vanilla
  182 (Superpower) stays unsupported. Volt Tackle's family (262: recoil 1/3 and a paralysis that
  BattleSystem_TriggerSecondaryEffect's PROBABILISTIC branch runs on every hit, battle_lib.c:
  1551-1567) and Jump Kick's crash (45, subscript_crash_on_miss.s: half the damage the miss would
  have dealt, capped at half the defender's max HP, so the miss reads the crit and roll) are left.
- B9-5 Singles-only moves: in singles BattleSystem_Defender gives RANGE_ALLY and
  RANGE_USER_OR_ALLY the attacker (battle_lib.c:1670-1684; `effects.targets_user`, both engines).
  Follow Me (172) sets the user side's followMe / followMeUser (battle_script.c:6261-6270), read
  only by target selection (battle_lib.c:1714-1717), the redirection abilities (:1743) and
  Counter (battle_script.c:4628), which in singles name the target anyway: a message only
  (`EffectId.FOLLOW_ME`). Helping Hand (176) fails outside a double battle
  (BtlCmd_TryHelpingHand :6309-6311, MOVE_STATUS_FAILED, effect_script_0176.s:15-17;
  `EffectId.HELPING_HAND`). Surf (257) is Gust's case: its x2 needs a diving target (Dive is
  unsupported).
- B9-6 Volt Absorb / Water Absorb join Dry Skin's absorb (BattleSystem_TriggerImmunityAbility,
  battle_lib.c:3500-3512, run at TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES on a move that hit):
  Water Absorb has Dry Skin's test (a Water move with power); Volt Absorb takes any Electric move
  aimed at another battler, Thunder Wave included (no power test); both heal Divide(maxHP, 4)
  below full HP and tag the holder (subscript_ability_restores_hp.s). Rain Dish heals
  Divide(maxHP, 16) below full HP in rain (BtlCmd_EndOfTurnWeatherEffect, battle_script.c:
  5614-5618) and tags the holder (subscript_weather_continues.s:_076). Clear Body blocks any drop
  by another battler and Hyper Cutter an Attack drop, before Keen Eye's accuracy test
  (BtlCmd_ChangeStatStage, :2884-2902, Battler_IgnorableAbility), Intimidate's included (its
  attacker is the Intimidate holder, subscript_intimidate.s:6); those messages are built in C, so
  the AI learns nothing.
- B9-7 Aftermath (battle_lib.c:4358-4371): the defender is faintedMon (this hit's UPDATE_HP fainted
  it, TryFaintMon battle_script.c:1768-1770), a living attacker without Magic Guard, no Damp on the
  field (unsupported), no MOVE_STATUS_NO_EFFECTS, a contact move, not under SYSCTL_UTURN_ACTIVE
  (U-turn's contact ability runs before that flag, B6): Divide(maxHP, 4) from the attacker; the
  message tags the holder.
- B9-8 Poison Heal (subscript_poison_damage.s:4-12, poison and toxic alike): below full HP it heals
  Divide(maxHP, 8) instead of the damage and tags itself; at full HP nothing (the toxic counter
  still rises, bcp.c:1405-1415). Bad Dreams (MON_COND_CHECK_STATE_BAD_DREAMS, bcp.c:1484-1500,
  after Bind): an asleep battler with HP and no Magic Guard facing a living holder
  (COUNT_ALIVE_BATTLERS_THEIR_SIDE_FLAG, battle_lib.c:2915-2922) loses Divide(maxHP, 8) per
  holder, the message tagging the holder (subscript_bad_dreams.s).
- B9-9 Dragon Rage (41) and SonicBoom (130) are Night Shade's family with a fixed amount (40, 20) in
  param 0 (effect_script_0041.s, _0130.s: SYSCTL_IGNORE_TYPE_CHECKS, BTLVAR_DAMAGE). Their
  BTLVAR_REGULATION_FLAG failure reads battleSys->rulesetMask (battle_script.c:9647-9648), set only
  for a regulated battle (field_battle_data_transfer.c:110, :376-379).
- B9-10 Stat moves. Bulk Up (208) is Dragon Dance's family with Defense for Speed
  (subscript_user_atk_and_def_up_1_stage.s differs at :4, :12 only); since it raises Defense it is
  a clause effect for the foe's later hit. Belly Drum (142, subscript_belly_drum.s): fails at
  Attack +6 or curHP <= Divide(maxHP, 2) (MOVE_STATUS_FAILED, :3-6, :52), else Attack +6 and
  Divide(maxHP, 2) through UPDATE_HP. Minimize (108, subscript_minimize.s): sets the user's
  MOVE_EFFECT_MINIMIZE (`BattleState.minimized`, volatile, not Baton Passed, a mask bit Roar's
  Ingrain test sees) then Evasion +1. Stomp (150, effect_script_0150.s): a flinch at its chance
  with powerMul 20 against a minimized defender (`damage.script_power_mul`).
- B9-11 Trapping abilities in `actions.trapped` (Battler_IsTrapped, battle_lib.c:5514-5536, read by
  the party menu, bcp.c:572): a living foe's Shadow Tag unless the battler has it too, a living
  foe's Magnet Pull on a Steel battler, a living foe's Arena Trap on a grounded battler
  (BattlerIsGrounded, :5500-5507: no Levitate, no Magnet Rise, not Flying, or Iron Ball, or
  Gravity; Ingrain does not ground here), by Battler_Ability over the other side's living battlers
  (:5509-5512). Shed Shell (:5518) is unsupported. The AI's own gate already had them (B7-18).
- B9-12 Dream Eater (8) is DRAIN 50 with param 1 = 1: effect_script_0008.s ends with
  MOVE_STATUS_NO_MORE_WORK ("wasn't affected") on an awake defender (:5, :9-16), else
  subscript_dream_eater.s: Divide(hitDamage, 2) then Big Root (as drain). Payback (230, Kaizo's
  Revenge too) doubles movePower once the defender's action is over (BtlCmd_CalcPaybackPower,
  battle_script.c:7198-7209; Battler_MovedThisTurn, battle_lib.c:5365-5368: a switched-in
  newcomer's action is MOVE_END too, :2053), as a powerMul of 20 in `effects.play_hit` (the same
  product, battle_lib.c:6688).
- B9-13 (batch 9 review) The end-of-turn speed grid (`enumerate._end_speeds`) took the outgoing
  battler's `can_unburden` for an incoming Pokemon. UpdateAfterSwitch zeroes MoveEffectsData
  (battle_lib.c:2101-2104) and the switch-in check arms canUnburden only for an Unburden holder
  that still holds an item (:4433-4437): a Pokemon other than the active one now carries
  `(ability == UNBURDEN) & (item != 0)` (law test `test_unburden_speed_axis_of_a_newcomer`). The
  generator no longer gives an item holder `can_unburden = 0` (its switch-in check set it).
- B9-14 (batch 9 review) The reference engine applied Belly Drum on a failing use (Attack +6, or
  curHP <= Divide(maxHP, 2)): subscript_belly_drum.s:3-6 fails before the stage and the HP, as the
  JAX engine had it (`effects.belly_drum_fails`). Found by the new hand test; the reference fixed.
- B9-15 (batch 9 review) `build_tables._kaizo_rom_changes` checks the modelled be_seq members
  equal `effect_map.KAIZO_EFFECTS` exactly against the real rom_diff.toml (an entry with no
  modelled member would override the vanilla mapping of an unchanged script); a test's own small
  diff keeps the subset check. The exhaustive solver (`tests/search/exhaustive.py`) unpacked a
  fourth value its own `_turn` never returns (48e278c, batch 5): `test_stage_b.py` failed there. Batch 9's hand tests strengthened: Payback's x2,
  Belly Drum's failures, battle cases for Chlorophyll, Swarm, Flower Gift, Water Absorb, Rain
  Dish, Aftermath, Hyper Cutter vs Intimidate, SonicBoom, the Kaizo recoil divisors, and the
  trapping abilities (`actions.trapped`: Shadow Tag vs Shadow Tag, Arena Trap vs Flying /
  Levitate / Gravity, Magnet Pull on Steel only, a fainted trapper).
- B10-1 (batch 10, family 1) Volt Tackle's #262 and Flare Blitz's #253 are one effect,
  `EffectId.RECOIL_STATUS` (80): params (status, recoil divisor 3, thaws_user). The scripts
  (effect_script_0262.s, _0253.s) set the status under SYSCTL_APPLY_SECONDARY_EFFECT through the
  PROBABILISTIC branch (battle_lib.c:1551-1567), then subscript_recoil_1_3.s takes
  Divide(hitDamage, 3) from the user unless Rock Head or Magic Guard. Flare Blitz also thaws its
  frozen user (bcp.c:2481 lets it act, :2730 thaws), as THAW_AND_BURN_HIT does. Chance 0 is
  refused by build_tables. Reckless powers both (powerMul 12, as RECOIL). AI: both go to
  Expert_RecoilMove (script.s:1796, :1802 -> :6289-6305); #253 added to the Kaizo transcribed
  sets and the dispatch (#262 was E1's batch 7).
- B10-2 (batch 10, family 1) Jump Kick / Hi Jump Kick #45 are `EffectId.CRASH` (81): a plain hit;
  effect_script_0045.s sets SYSCTL_CRASH_DAMAGE before CalcCrit / CalcDamage, so a use that ran
  its script and did not connect (missed or protected) ends in subscript_missed.s, whose
  subscript_crash_on_miss.s (:11-19) takes min(Divide(damage, 2), Divide(defender maxHP, 2))
  from the user, Magic Guard excepted. The damage is hit 0's crit and roll through the type
  chart run hit or miss (bcp.c:3271-3277): an immune target gives 0. The laws read crit and
  roll[s, 0] on a miss (`Reads.crash`). Reckless powers it. AI: #45 has no Expert_Main entry
  (E1 batch 3 transcribed it generically).
- B10-3 (batches 3-9 merge check, follow-ups) `BattleSystem_InitBattleMon` (bcp.c:227 at the
  battle start) arms canUnburden for every lead holding an item, whatever its ability
  (battle_lib.c:161-166): `hooks.battle_start` now does so before the switch-in states, so a lead
  that gains Unburden later (Trace) keeps it (test_any_item_holder_lead_arms_unburden). What Baton
  Pass keeps (Mean Look, Lock-On, Perish Song, Curse; battle_lib.c:2073-2115) gets a fast hand test
  (test_baton_pass_keeps_mean_look_lock_on_perish_song_and_curse), which fails under a JAX-only
  mutation dropping the four from `_BATON_PASSED` (checked 2026-11-15) and passes without it.
  Citations fixed: Sleep Talk's CheckInvalidMoves is battle_script.c:4927; subscript_curse_ghost.s
  is :5-7 (tests), :9-15 (curse and HP), :17 (message), :22-24 (failure); InitBattleMon's arm is
  :161-166 and UpdateAfterSwitch's clear :2100-2104. tests/ai_legality.py says in its first line
  that it checks host-side picks only.
- B10-4 (batch 10, family 3) Substitute (`EffectId.SUBSTITUTE`, 82, decomp 79), both engines, as
  designed under "Family 3 design" (decomp inventory: `_scratch/e2/b10/substitute_inventory.md`).
  The move (`effects._substitute`, `substitute_fails`: subscript_try_substitute.s:7-8,
  battle_script.c:5211-5226, MOVE_STATUS_NO_MORE_WORK). The hit (`play_hit`; bcp.c:3359-3378):
  the substitute takes min(damage, its HP), hitDamage for drain and recoil, shellBellDamageDealt,
  the flag (`MoveHit.sub_hit`, a hit-loop carry; the reference's `_Turn.sub_hit`), and none of
  False Swipe's cap, the Focus Sash, Bide's store, the damage-taken records, the resist berry,
  SYSCTL_MOVE_HIT. Gates: `_Hit.shielded` (CheckSubstitute: the bit or the flag) for the status
  subscripts, flinch (King's Rock too), confuse, Bind, Leech Seed, Swagger (whole move fails);
  `_Hit.sub_hit` (SubstituteWasHit) for rolled foe secondaries, Pluck (lib.c:5546), the contact
  ability (:4209), the Sticky Barb transfer (:5378); the bit only for a foe stat drop
  (script.c:2921-2923), Dream Eater (effect_script_0008.s:5), SmellingSalt's x2 and cure
  (effect_script_0171.s:5), Intimidate (subscript_intimidate.s:13), Toxic Spikes on a Baton Pass
  newcomer (subscript_poison.s:29, _badly_poison.s:46). Special paths: Future Sight's hit
  (subscript_future_sight_damage.s:13-24), Pursuit's pre-switch hit (subscript_pursuit.s:20,
  :56-73: only a break changes the substitute; `play_hit(pursuit_switch=True)`). Laws: SUBSTITUTE
  is input-neutral and non-table (SmellingSalt is DOUBLE_POWER, never compressed). Observation:
  the substitute's presence was already public, its HP hidden. The AI's scripts (Basic :633-636,
  Expert_Substitute :2976-3042) were E1's. Hand tests in tests/engine/test_batch10.py (the move,
  the hit and the break, status / drop / drain, multi-hit with Static, King's Rock, Pursuit, Ghost
  Curse); the generator `strategies._draw_substitute`. Differential seed 3, 150: clean.
  Backlog regenerated: missing groups 241 -> 240, runnable battles 53 -> 57. Found on the way:
  family 1 left RECOIL_STATUS / CRASH unclassified in test_accuracy_classes and Flare Blitz's thaw
  unflagged in test_effect_map's thaw test; batch 7 (4a06792) left its four observation cases out
  of test_observe.VISIBLE_CLASSES (a slow test): all three fixed with this commit.
- B10-5 (batch 10, family 5) Endeavor (`EffectId.ENDEAVOR`, 83, decomp 189) and Pain Split
  (`EffectId.PAIN_SPLIT`, 84, decomp 91), both engines. Endeavor (effect_script_0189.s): fails
  (MOVE_STATUS_FAILED, :7, :13-14; `step._use_move`'s `failed`, the reference's `_script_fails`)
  unless the defender's curHP exceeds the attacker's; else battleCtx->damage is the difference
  under SYSCTL_IGNORE_TYPE_CHECKS (:10: no multiplier, an immunity still gives 0; no CalcCrit /
  CalcDamage: `UNROLLED_EFFECTS`, `damage.fixed_damage`; the resist berry skips it,
  `IGNORES_TYPE_CHECKS`); a substitute takes it like any hit (B10-4). Pain Split
  (subscript_pain_split.s): fails behind a substitute (:5, :50-52: `primary_effect_fails`); else
  avg = (a + d) / 2 (OPCODE_DIV) and each battler's HP goes to it through UPDATE_HP, capped at its
  max HP (`effects._pain_split`). Both input-neutral and non-table (HP only). The AI's
  Expert_Endeavor / Expert_PainSplit were E1's. Hand tests in test_batch10.py; the generator draws
  both in `_draw_batch10` with a low user.
  Differential seed 4, 150: clean. Backlog: missing groups 240 -> 238, runnable battles 57 -> 58.
- B10-6 (batch 10, family 6) Knock Off (`EffectId.KNOCK_OFF`, 85, decomp 188), both engines, as
  designed under "Family 6 design". `effects._knock_off` (the reference's KNOCK_OFF branch): a
  table hit, then subscript_knock_off.s's tests (substitute, Multitype, Griseous Orb, quickClaw,
  custapBerry) and BtlCmd_TryKnockOff (Sticky Hold, decomp ability 60, ignorable; else the item
  goes and the slot's bit). New state: `knocked_off` (per party slot) and the turn scratch
  `ts_claw` (`order.claw_flags`: the quickClaw flag from the FIGHT-vs-FIGHT order, cleared at its
  holder's move start; no persistence is needed, as only two FIGHT commands set it and the holder
  then reaches its move start or faints). The Sticky Barb transfer skips a knocked-off attacker
  and Knock Off itself (battle_lib.c:5390-5391; pivot :7380). Laws: a table-damage effect and an
  input clause (the foe's later hit when it holds an item). The AI's Knock Off is E1's (branch
  claude/e1-prep-on-e2). The AI's record (E2's under the end-to-end rule, manager 2026-11-15):
  TryKnockOff runs no CopyToParty, so `ai_item_known` keeps the knocked item until the
  battler's next CopyToParty (battle_lib.c:1819-1823): its next HP-bar update
  (battle_script.c:1720) or move attempt (bcp.c:2748). Modelled per segment (an action, the end
  of the turn) in both engines (`step._stale_known_item`, the reference's twin): exact, as a
  knocked-off slot never holds an item again and the AI reads the record only at its next
  choice; test_knock_off_leaves_the_ai_record_until_the_next_copy_to_party. Thief, Covet, Trick
  and Fling, which also
  read the mask, are unsupported.
  STATE.md maps quickClaw (`ts_claw`) and knockedOffItemsMask (`knocked_off`); test_state's
  audit list no longer pins quickClaw as later. Differential seed 5, 150: clean.
- B10-7 (batch 10, family 8) Safeguard (`EffectId.SAFEGUARD`, 86, decomp 124), both engines.
  BtlCmd_TrySafeguard (battle_script.c:5754-5772): fails with the user's side's Safeguard up
  (MOVE_STATUS_FAILED, `primary_effect_fails`), else safeguardTurns = 5 (the existing
  field-condition count-down, bcp.c:1016-1030). Its tests: every status subscript after the
  substitute and status tests (paralyze :28, burn :45, freeze :25, poison :41, badly_poison :55,
  fall_asleep :51, after its uproar test), subscript_confuse.s:26 (Swagger's confusion too, after
  its Attack raise), the held-item path (Flame Orb subscript_burn.s:16, Toxic Orb
  subscript_badly_poison.s:27) and Toxic Spikes on a newcomer (poison :41, badly_poison :55);
  Yawn's test was already modelled (B8-3); the drowsy sleep (MOVE_EFFECT type) has none. The AI's
  Basic_CheckAlreadyUnderSafeguard was E1's. Found on the way (B10-4 follow-up): Uproar's
  "kept awake" NO_MORE_WORK for a direct sleep move now excludes a substituted target, whose
  CheckSubstitute (subscript_fall_asleep.s:40) fails it first (both engines).
  And a law fix (B10-4): `Reads.flinch_sure` let the first mover's flinch roll drop the foe's
  hit reads even when the foe was behind a substitute, where the flinch never applies
  (subscript_flinch_mon.s:6): test_damage_classes' generated case (a 1-HP substitute, Trick
  Room, an uproar) found it; it now needs the foe's `substitute_hp == 0`, and
  test_laws_no_sure_flinch_through_a_substitute fails without that clause.
- B10-8 (batch 10, family 10) Focus Punch (`EffectId.FOCUS_PUNCH`, 87, decomp 170), both engines.
  effect_script_0170.s:5-6: a user whose turnFlags physical / specialDamageAttackerMask is set
  (a hit on its HP this turn, bcp.c:3416-3424; the engine's `phys_from` / `spec_from` records,
  `foe_taken[3:5]`) gets MOVE_STATUS_LOST_FOCUS, a DID_NOT_HIT status: subscript_missed.s:8-10
  prints "lost its focus" and the move ends like a failed script (`step`'s `failed`, the
  reference's lost_focus beside `_script_fails`); else a plain table hit (priority -3 is the
  move's data). A hit on its substitute sets no mask (B10-4). The turn-start "tightening its
  focus" message (bcp.c:811-834, PRE_MOVE_ACTION_STATE_TIGHTEN_FOCUS) changes no state; the
  observation shows the move and the miss (coarser: "lost its focus" and a miss share a set).
  The AI's Expert_FocusPunch was E1's. Encore and Assist (the rest of family 10) are not done.
- B10-9 (batch 10, family 10) Encore (`EffectId.ENCORE`, 88, decomp 90), both engines; its AI
  (Basic_CheckCannotEncore script.s:655-657, Expert_Encore :3177-3240) is still to transcribe
  (the AI refuses the move until then). New volatile state `encore_move`, `encore_slot`,
  `encore_turns` (MoveEffectsData, cleared on a switch, not Baton-passed). BtlCmd_TryEncore
  (battle_script.c:4725-4748; `effects.encore_slot_of`, `_encore`): fails (MOVE_STATUS_FAILED) for
  an encored target, no last move, a last move not in its slots, one of sCannotEncoreMoves'
  battle effects (battle_lib.c:7212-7231, `ids.CANNOT_ENCORE_MOVES` through `move_ai_effect`) or
  no PP; else the move, its slot and RandNext() % 5 + 3 turns (`DURATION_ENCORE`, 3-7). The
  selection (`actions.usable_moves`): CheckInvalidMoves leaves only the encored move
  (battle_lib.c:2302-2304), and with it valid the encored slot is the one FIGHT action
  (bcp.c:443-446), else Struggle. The action (FightCommand, bcp.c:1866-1883): after Struggle, the
  encored move runs whatever was chosen (a faster Encore this turn takes over), from its slot's PP
  (no PP: MOVE_STATUS_NO_PP, now modelled in the JAX engine too); a slot that no longer holds it
  ends the encore. The turn end (`step._encore_check`, bcp.c:1591-1612, between Disable and
  Lock-On): unknown or 0-PP move zeroes the count; the count drops, at 0 the encore ends. The
  observation shows that an encore is on (the count is hidden).
  Owed (this batch, after E1's ALLIN merge lands in main): Encore's AI. Basic_CheckCannotEncore
  and Expert_Encore need AICmd_IfBattlerUnderEffect (CHECK_ENCORE, CHECK_DISABLE), not on this
  base, and Expert_Encore's IfRandomLessThan 30 (script.s:3192) goes through ALLIN's
  commands._slot_read with a read_bounds.READ_BOUNDS entry. Until then coverage refuses Encore
  for the trainer AI. Found by the differential (seeds 9, 10): the reference's volatile reset list
  lacked the three encore fields; its usable-move check (`_usable`) ignored Encore (Struggle and
  the forced slot); both fixed. Sleep Talk's callable slots drop every move but the encored one
  (CheckInvalidMoves at battle_script.c:4927; `order.sleep_talk_slots`, both engines).
  Manager ruling 2026-11-15: batch 10 ships Encore refused by coverage for both sides
  (`coverage.IMPLEMENTED_EFFECTS[ENCORE] = NONE`, reason `ENCORE_REFUSAL` naming the takeover
  task; Metronome is refused with it, as it can call Encore), the engine code kept; batch 11 builds
  the takeover (design below) and then lifts the refusal and adds Encore's AI.
  Owed (decision asked of the manager 2026-11-15): the same-turn takeover (FightCommand,
  bcp.c:1869-1873) is exact in both engines but not in `outcome_laws` (reads are planned from the
  chosen move: a hand case gives 1 child against 33 sampled outcomes) nor in `sample_outcome`
  (crit and secondary drawn with the chosen move's rates). Proposed: the encored move as a second
  candidate move for that side, reads unioned and crit / secondary drawn per candidate.
- B10-10 (law) `Reads.flinch_sure` counted Snore (FLINCH with SNORE_PARAM), whose flinch fails
  with its move unless the user is asleep when it acts (effect_script_0092.s:5): an awake Snore
  user moving first let the laws drop the foe's reads (test_matches_sampled_frequencies_random's
  derandomized case, Snore vs Future Sight; on main d420167 too, reached now by the batch 10
  generator draws). Snore is left out of `flinch_sure` (its foe's reads stay planned).
- B10-11 (batch 10, family 10) Assist (`EffectId.ASSIST`, 89, decomp 180), an invoker
  (`order.CALL_ASSIST`), both engines. BtlCmd_TryAssist (battle_script.c:6405-6444;
  `order.assist_candidates`): in party order, every move of every other party slot holding a
  Pokemon (a fainted mate's too), not Move_IsInvoker (`ids.INVOKER_MOVES`, battle_lib.c:
  3032-3045) and Metronome-able (NO_METRONOME; Gravity and Heal Block owed with those moves);
  duplicates count; the call is RandNext() % numMoves (`sample_outcome`: a uniform draw over the
  list, fold_in 34 + side; `call_laws`: each position weighs 1, grouped by the move it calls in
  `call_variants`); none listed fails it (MOVE_STATUS_FAILED). Then GoToMoveScript like the other
  invokers (step's generic invoker path; RANGE_SINGLE_TARGET_SPECIAL: its own defender is the
  user). Its AI entry was transcribed generically by E1 (batch 3).
- B10-12 (batch 10, family 7) The charge turn. Each charge move keeps its attack turn's
  EffectId (Razor Wind / Skull Bash / SolarBeam / Fly / Dig / Dive / Shadow Force PLAIN, Sky
  Attack FLINCH, Bounce SECONDARY_STATUS paralysis); `MoveFlag.CHARGE` and kin (built from the
  decomp effect: `effect_map.CHARGE_FLAG_EFFECTS`, `move_flags` now int32) carry the charge.
  The scripts (effect_script_0039/0075/0145/0151/0155/0255/0256/0263/0272.s) attack when the
  user is VOLATILE_CONDITION_MOVE_LOCKED (line 5: the second turn, `locked_move`; PP spent once,
  on the first: bcp.c:335), SolarBeam in sun without Cloud Nine / Air Lock (_0151.s:5-6,
  CheckIgnoreWeather battle_script.c:9195-9206) or with a Power Herb (HOLD_EFFECT_CHARGE_SKIP,
  `ItemId.POWER_HERB`, removed by subscript_item_skip_charge_turn.s:20; Skull Bash's Def +1
  first, subscript_power_herb_skull_bash.s:14-17). Else they charge: SYSCTL_FIRST_OF_MULTI_TURN
  (no accuracy roll bcp.c:2905-2908, no Protect / Lock-On / semi test :3003-3005, no type chart
  :2354, no immunity abilities battle_lib.c:3509-3540), the direct subscript_vanish_on_charge_
  turn.s locks the user into moveCur (:10, Battler_LockMoveChoice lib.c:1828-1832; a called
  move's lock is the called move), the semi-invulnerable bit (new state `semi_invulnerable`,
  `ids.SemiKind`: AIRBORNE Fly/Bounce, UNDERGROUND Dig, UNDERWATER Dive, SHADOW_FORCE), Skull
  Bash's Def +1 (_0145.s:8, chance 100). A charge with no living foe keeps the user as defender
  and charges (HasNoTarget bcp.c:2327-2334); a called charge move with none goes to
  subscript_no_target from GoToMoveScript (battle_script.c:2124-2126); a herb / sun attack with
  none ends in TryMove's HasNoTarget (LAST_OF_MULTI_TURN, common.h:57-60) after the herb went.
  The second turn's script clears the bit and the lock before TryMove (line 24 / cleanup :6).
  Interruptions: every disruption, miss, Protect and the sleeper unlock at turn setup run
  Battler_UnlockMoveChoice, which also clears the bit (lib.c:1838): the move is lost and the
  next use charges again (flinch, full paralysis, confusion self-hit, Truant, sleep); a switch
  or faint resets it (volatile). Choice lock: FIRST_OF_MULTI_TURN updates the locks as the
  attack message does (bcp.c:3921-3922, :4490-4491). The charge turn shows no attack message
  (moveHit, movePrev, moveProtect MOVE_NONE) except Skull Bash, whose indirect Def +1 runs
  subscript_update_stat_stage.s's PrintAttackMessage (:6-7; battle_script.c:1431-1432).
  Razor Wind / Sky Attack crit +1 (CRITICAL_BOOSTS, `MoveFlag.CRIT_UP`, `outcome.crit_odds`).
  Shadow Force's ON_HIT subscript_feint.s only clears the defender's turnFlags.protecting
  (:20), unread afterwards in singles. Kaizo: be_seq 39-272 are unchanged (rom_diff.toml);
  Kaizo gives the charge effects to Razor Wind, SolarBeam, Fly and Shadow Force only (its Dive
  and Bounce are plain / paralysis hits); Kaizo Sky Uppercut is effect 149 and Twister 261.
- B10-13 The semi-invulnerable target (CheckMoveHitOverrides, bcp.c:3016-3040): after Protect
  and the Lock-On / No Guard return (either battler's No Guard; OHKO's BYPASSED_ACCURACY,
  battle_script.c:4343-4374, the same battlers), a move whose defender is the foe (a special
  Counter's and Bide's unleash included), not RANGE_OPPONENT_SIDE, misses
  (MOVE_STATUS_SEMI_INVULNERABLE, accuracy 0 too) unless its script sets the kind's
  SYSCTL_HIT_DURING_* (`MoveFlag.HIT_*`, `effect_map.SEMI_HIT_EFFECTS`: Fly 146/149/152/207,
  Dig 126/147, Dive 257/261, all 148/168/176). A miss crashes (Jump Kick) and shows before
  Sturdy (subscript_missed.s:13). Exception, Yawn: a direct MOVE_SIDE_EFFECT_CHECK_HP effect
  (effect_script_0187.s:5), whose subscript BattleSystem_TriggerPrimaryEffect still runs on
  MOVE_STATUS_GENERAL_MISS for a living defender (battle_lib.c:1500-1503; moves.h:88);
  subscript_yawn.s:5-9 test Insomnia / Vital Spirit / Leaf Guard in sun (ignorable) before its
  own miss test (:21), so against a semi-invulnerable holder the ability is printed with
  TAG_NICKNAME_ABILITY (:31-37 -> BattleAI_SetAbility, battle_script.c:11798-11806) and
  `ai_ability_known` set; no drowsiness either way. A protected target is not semi-missed
  (Protect returns first, bcp.c:3007-3013), so no reveal. x2 (powerMul 20, `MoveFlag.DOUBLE_SEMI`,
  `damage.script_power_mul`) for Gust / Twister vs AIRBORNE, Earthquake / Magnitude vs
  UNDERGROUND, Surf / Whirlpool vs UNDERWATER (line 7-8 of each). MOVE_EFFECT_NO_WEATHER_DAMAGE
  (UNDERGROUND, UNDERWATER: moves.h:50) skips the sandstorm, sun and hail branches of the turn
  end (battle_script.c:5580, :5587, :5600), not rain. SolarBeam (by move id) is halved under
  FIELD_CONDITION_SOLAR_DOWN (rain, sand, hail, fog; lib.c:7058-7060). Twister (146) and Sky
  Uppercut (207) are now supported. Damage classes: a foe's charge move changes a DOUBLE_SEMI
  hit's power, Skull Bash's charge the Defense. AI: Basic :158/:219, Expert_ChargeTurnNoInvuln
  (:3933-3973), _WithInvuln (:3992-4012), Expert_ShadowForce (:4014-4066, 80 roll at :4058);
  SolarBeam's Expert_Thunder entry (:1719) is dead (the first jump wins). Not modelled (no
  supported mover): Gravity's grounding (subscript_gravity_start.s:16; Gravity refused by
  coverage), Transform's semi test, Hurricane (Kaizo 141, not reached).
- B10-14 (batch 10 review) Future Sight / Doom Desire's delayed hit and a semi-invulnerable
  defender. subscript_future_sight_damage.s:8 CheckMoveHit (battle_script.c:6115-6131) runs
  CheckMoveHitOverrides (bcp.c:216-220): after Protect (:3005-3013) and the Lock-On / No Guard
  return (:3016-3021), :3034-3040 sets MOVE_STATUS_SEMI_INVULNERABLE on a defender with a semi
  bit. Every SYSCTL_HIT_DURING_* bit is cleared by BattleContext_Init at the last MoveEnd
  (battle_lib.c:2009, system_control.h:31-40; bcp.c:3998) and BYPASSED_ACCURACY with
  moveStatusFlags (:1983), so any semi kind is missed ("But it failed!"); the bit itself lasts
  to the holder's next action. The range test reads CURRENT_MOVE_DATA (common.h:24), i.e.
  moveCur, which nothing resets: FightCommand's moveTemp (bcp.c:1898) or GoToMoveScript's called
  move (battle_script.c:2118). After a turn whose last such move is Spikes, Toxic Spikes or Stealth
  Rock (the RANGE_OPPONENT_SIDE moves) the test is skipped and the hit lands. A turn with no FIGHT
  keeps an older moveCur, but then no battler is semi-invulnerable at the turn's end (its holder
  is locked into a FIGHT), so engines read only this turn's (`step._turn_move_cur`, event
  `move_cur`; reference `_Turn.move_cur`). The flag lasts the pass like the others: a second hit
  after a semi miss fails even under No Guard (which clears only MISSED).
- B10-15 (batch 10 review; corrects B10-13's OHKO clause) An OHKO move and a semi-invulnerable
  target. BtlCmd_TryOHKOMove sets SYSCTL_NONSTANDARD_ACC_CHECK first (battle_script.c:4338), so
  the Lock-On / No Guard return of CheckMoveHitOverrides is skipped (bcp.c:3016); the semi test is
  skipped only by MOVE_STATUS_BYPASSED_ACCURACY (:3035), which TryOHKOMove sets on the
  non-Sturdy branch when the defender has MOVE_EFFECT_LOCK_ON (any target) or either battler has
  No Guard (:4342-4372). Against an ignorable Sturdy (:4340-4341) No Guard / Lock-On do not help:
  the semi target is missed, and subscript_missed.s:11-12 shows that miss before the Sturdy
  message (:67-69), so the AI does not learn Sturdy (ai_ability_known).
- B10-16 (batch 10 review) effects.py's module note: Safeguard is supported (B10-7); only Mist's
  branches stay unmodelled (no supported mover).
- B10-17 (batch 10 review) The pause scratch's `ts_claw` / `ts_custap` keep the flag of each
  battler whose action is still after the paused one in the order (`to_come`), a fact about the
  named battler (doubles design section 5 rule 5) instead of "the second actor at position 0";
  the same values in singles.
- B11-1 (batch 11) A chosen move replaced mid-turn: Encore's takeover (ledger "Design: a chosen
  move replaced mid-turn", implemented as designed). BattleControllerPlayer_FightCommand
  (bcp.c:1866-1873) runs an encored battler's encored move, after Struggle, whatever it chose, so
  a faster foe's Encore this turn (BtlCmd_TryEncore, battle_script.c:4725-4748) replaces the
  chosen move. One event for every replaced move: `step`'s `encored_at_move` (the encored move
  replaced the chosen one when the action started) with the turn-start test in
  `enumerate.taken_events`, beside `disabled_at_move` / `sealed_at_move` (`_split_events`).
  Laws: `order.takeover_possible` (the side uses a move without Struggling, is not encored at the
  start, `encore_slot_of` would not fail it, its last move differs from the chosen one, and its
  foe's move can run Encore: `order.can_call_encore`, Encore itself or Sleep Talk / Metronome /
  Assist / Copycat whose call can be Encore) gives the side a takeover variant: `call_variants`
  returns `Variant(call, taken, num, den)`, the tag `taken` a field beside `call`; `outcome_laws(...,
  taken)` reads every move-dependent law of that side (the move run, effect, crit odds, damage
  classes, thresholds, the call law) from the encored move with `order.encored_state`, the order
  staying the chosen move's (CalcTurnOrder fixes it at the turn start). `call_consistent(...,
  taken)` keeps a row only where the event matches the tag (for a side that can be taken), so
  each side's variants sum to 1 over the kept rows; `enumerate_turn` and `search.expand` carry the
  tag per row. Outcome shape unchanged. Sampler: outcome A as before; B redraws each side's
  move-dependent fields (`outcome.MOVE_FIELDS`: hit, crit, secondary, fang_flinch, duration,
  future_hit, call) for each alternative (the Disable-shrunk and Imprison-sealed Sleep Talk sets,
  `order.disabled_state` / `sealed_state`, and the encored move, `order.encored_state`); the
  turn's events run on A (`step.turn_events`, only where `outcome.replacement_possible`) and a
  side whose event fired takes B's fields, drawn once for the alternative that fired (the
  invokers' draws read none of the set changes). Cost: under `vmap` the `lax.cond` becomes a
  select, so the event pass runs on every row: the sampled-frequency tests (test_enumerate,
  test_accuracy_classes) went from 49 s / 4.0 GB peak to 97 s / 5.2 GB; playouts pay the same
  (a second action phase per sampled turn), owed to the parked performance work. Exact: the event is fixed before
  that side's action starts, and the redrawn fields are read only after it. Singles: at most one
  replacement per side per turn (one foe move); doubles keeps a short list per battler keyed by
  its event (doubles design section 5).
- B11-2 (batch 11) The Sleep Talk + Disable / Imprison sampler gap (B5-13, B7-13): the sampler
  drew a split Sleep Talk's fields from the turn-start call; B11-1's pass now draws them for the
  move called from the set the turn leaves. Frequency tests (Ember 100 / Thunder 70, a faster
  Disable 80 or Imprison) failed before (0.85 against the exact 0.73 / 0.70) and pass after.
- B11-3 (batch 11) Encore into the invokers. Move_CanBeEncored (battle_lib.c:7221-7231) refuses
  only Transform, Mimic, Sketch, Mirror Move, Encore and Struggle (ids.CANNOT_ENCORE_MOVES), so
  Metronome, Assist, Sleep Talk and Copycat can be encored; the takeover variant's call law is
  the encored invoker's, read with the encored state. An encored Sleep Talk always fails:
  BattleSystem_CheckInvalidMoves (battle_lib.c:2302-2304) flags every move but the encored one
  and TrySleepTalk (battle_script.c:4916-4930) flags the invoker itself, so no slot is left
  (STRUGGLING_ALL, :4928). An encored Copycat taking over after a faster Encore copies that
  Encore (movePrev, the foe's moveTemp), which then fails on the Encore user (its last move is
  Encore, refused). Both engines already ran these; the laws and the sampler now follow.
- B11-4 (batch 11) Encore's AI (ALLIN convention). Basic_CheckCannotEncore (script.s:655-658,
  entry :188): the player already encored, -8; Expert_Encore (script.s:3177-3200, entry :1689):
  the player disabled -> the +3 roll; else slower -2; else the player's last move's effect not in
  Expert_Encore_EncouragedMoveEffects (:3202-3285, `commands.ENCORE_ENCOURAGED_EFFECTS`, 82
  entries) -2; else +3 unless IfRandomLessThan 30 (:3192, `commands._slot_read`, bound in
  `read_bounds.READ_BOUNDS` row 90) jumps. IfBattlerUnderEffect (trainer_ai.c:1823-1848) reads
  `disabledTurns` / `encoredTurns`, not the move (`commands.if_battler_under_effect`). Effect 90
  joins `outcome.KAIZO_TRANSCRIBED` / `KAIZO_EXPERT_LABELS` / `KAIZO_EXPERT_TRANSCRIBED`; the
  coverage refusals of Encore (both sides) and of Metronome (which can call it) are lifted.
- B11-5 (batch 11, manager's review items) The generator (tests/strategies.py) met the batch 10
  moves too rarely (900 examples: no vanish, Power Herb, Assist, Encore, Pain Split or Safeguard
  used): `_draw_batch10` now runs one draw in two, a charge user awake or thawed 7 in 10, a
  status-move user untaunted 7 in 10 (invokers too, `_draw_batch5`), the side's own encore on
  another move cleared 7 in 10; `legal_action_pair` uses a known `RARE_EFFECTS` move or charge
  move one draw in two. (A first try gave the move two slots: a duplicate move, which no Pokemon
  can know, and seed 25 then split the engines on `last_move`; dropped.) 900 draws (seed 1):
  uses Encore 17, Assist 22, Pain Split 34, Safeguard 28, Endeavor 10, Knock Off 6, Focus Punch
  6, Metronome 18, Copycat 34, Fly 19, Dig 30, SolarBeam 27; vanished 90, Power Herb 159 (counts
  by `_scratch/e2/b11/gen_counts.py`).
- B11-6 (batch 11, review) tests/engine/test_coverage_exp.py: the `endeavor_unsupported` fixture
  (a faked unsupported effect) is gone; the level-up refusal tests stand on a real unsupported
  learnset move, Budew's Worry Seed at L16 (vanilla), and the narrowed
  `test_the_exp_sensitive_moves_are_unsupported` is replaced by
  `test_every_exp_sensitive_move_meets_its_exp_line`: every §4.2 / §4.4 / §4.5 move, in both
  games and whatever its support, meets its family's `exp_refusals` line.
- B11-7 (batch 11, review; doubles design section 5) `step._stale_known_item` loops over the
  battlers (`forget_item` keeps only the player's record); `order.claw_flags` sets the flag of a
  FIGHT battler compared with any other FIGHT battler (CalcTurnOrder's pairwise sort,
  bcp.c:766-778); the RANGE_OPPONENT_SIDE tests of the semi-invulnerable miss, the Future Sight
  hit and the defender choice go through one helper, `effects.targets_foe_side` (rule 6).
- B11-8 (batch 11, review) `search.expand`: the opponent's party-list pick is made once,
  host-side (the laws read it); the row program steps those actions and reads only the pick's
  refusal (`playout.turn_inputs`), so dropping the pick fails
  tests/search/test_pivot_search.py::test_the_opponents_list_is_resolved_by_its_ai.
- B11-9 (batch 11, found by the takeover law test) The laws never drew Encore's own duration
  (`OutcomeLaws.relevant.duration` lacked `EffectId.ENCORE`, though `duration_kind` had
  DURATION_ENCORE since B10-9): every enumerated Encore set the empty outcome's count. Fixed;
  unseen before because coverage refused Encore.
- B11-10 (batch 11, second review) No generated Pokemon knows a move twice, at the root: every
  move the generator writes goes through `strategies._teach` (the given slot, or the slot that
  already holds the move), used by every give-style helper and the other slot writes (species
  item, Pursuit, pivot, invoker pair, Uproar, Future Sight, batch 7 and Imprison's shared move,
  the law moves); Uproar's Copycat swap is skipped when Copycat is known, and the outcome-laws
  strategy's invoker-to-Tackle swap leaves an empty slot when Tackle is known. 31b3a95's double
  `give("Yawn", ...)` is one call. The new property test asserts the no-duplicate rule on every
  generated state. tests/engine/test_batch11.py::test_the_replacement_predicates_cover_every_event
  (300 states, any outcome): a side's replacement event (`taken_events`, or a Sleep Talk side's
  `_split_events`) never fires where `order.takeover_possible` (taken) or
  `outcome.replacement_possible` (any) is false, read on the `pivot_view` as the laws and the
  sampler read them. The generator alone met no event in 300 draws, so the test steers three
  draws in four (`_steer`: a Speed 999 foe using Encore, Disable or Imprison, no Trick Room).
  300 draws (seed 0): 28 takeovers, 22 Disables, 8 Imprisons, 58 events, none uncovered; with
  `takeover_possible` forced false 28 uncovered, with `replacement_possible` reduced to the
  takeover 30 uncovered. Note: `_split_events` also fires for a side that is not using Sleep
  Talk (a faster Disable on any move); the sampler then redraws that side's fields for the same
  move (`disabled_state` does not change a plain move), the same law, so harmless.
- B11-11 (batch 11, second review; doubles rule 1) `order.takeover_possible` and
  `outcome.replacement_possible` take `side` and `foe` (the battler whose Encore, Disable or
  Imprison can land first) and return one `bool`, as `can_call_encore` takes `side`; neither
  computes `1 - side` or `uses_move[::-1]` any more. The callers (`enumerate.call_consistent`,
  `enumerate.call_laws`, `outcome.sample_outcome`) name the foe, stacking over the two sides.
- B11-12 (batch 11, second review; Kaizo check) The Kaizo ROM's Expert_Encore and its encourage
  table are vanilla. Read with tools/rom (`RomFiles.overlay(14)` of the vanilla ROM and the
  cached Kaizo ROM, `SymbolMap` of the decomp map): gTrainerAITable starts at overlay14
  0x022248A4 (sized 0 in the map; compared up to the next sized symbol, 0x0222EE60, 10607
  words). One word differs, 0x022249CC (0x1A -> 0x57: rom_diff.md's Levitate -> Dry Skin compare,
  script.s:78); Expert_Encore (script.s:3177-3200, incl. IfRandomLessThan 30 at :3192) is
  byte-equal. Expert_Encore_EncouragedMoveEffects (script.s:3202-3285) is found once in each ROM,
  at 0x022292D8, as `commands.ENCORE_ENCOURAGED_EFFECTS` (82 entries) then TABLE_END. Script
  (not kept): _scratch encore_rom.py, `uv run --group rom`.
- B12-1 Calm Mind (211) and Cosmic Power / Defend Order (206) run DRAGON_DANCE-style two-stat
  raises: the first +1, then the second (subscript_user_*_up_1_stage.s:5-6, :13-16).
- B12-2 Flatter is subscript_flatter.s: Swagger's script with Sp. Atk +1 (:5-16).
- B12-3 Tickle (subscript_target_atk_and_def_down_1_stage.s): NO_MORE_WORK only with both stats
  at -6 (:5-6, :24-33); FAILED on a miss (:9).
- B12-4 Focus Energy: +2 crit stages (battle_lib.c:7123), Baton Pass keeps it (condition.h:72),
  fails when already set (effect_script_0047.s:5).
- B12-5 Haze resets every battler's stages and Focus Energy (battle_script.c:3534-3549).
- B12-6 Psych Up copies the stages and ORs Focus Energy (battle_script.c:6042-6053); the copy
  runs before TryMove, so a semi-invulnerable miss still copies (bcp.c:3211-3213, :3034-3040).
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
  Shaymin / Giratina form reloads (battle_lib.c:6439) would undo it too: no in-battle form
  change is modelled yet.
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
  on a statused or substituted target or a status-free user; passes the status as a direct
  effect, cures the user only if the target ends up statused.
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
- B12-29 sample_outcome's replacement flag (static): outcome.replacement_known, host side, true
  when a party Pokemon knows or can still learn Encore or an order.call_kind invoker; False
  leaves turn_events and outcome B out of the program (exact: replacement_possible is then
  false at every reachable state). Outcome B now runs inside the lax.cond. Threaded through
  step_through, _play, _turn, playout*, leaves, stage_a and tests.engine.helpers.sample_rows.
  Mimic, Sketch and Transform must join the flag when supported.
- B12-30 A RANGE_SINGLE_TARGET_SPECIAL move with no living foe still succeeds: FightCommand's
  defender (bcp.c:1900) is the user (BattleSystem_Defender battle_lib.c:1665-1669, before the
  randomize branch), so BeforeMove's HasNoTarget (bcp.c:3190, NO_TARGET include/battle/common.h:
  56-61) passes after the PP decrement (:3181-3185) and SYSCTL_MOVE_SUCCEEDED is set (:3212):
  movePrevByBattler = moveTemp, the invoker (UpdateMoveBuffers :3945-3946); movePrev and
  moveProtect follow the attack message as with a living foe (:3937-3939). Sleep Talk, Metronome,
  Assist, Copycat (and Mirror Move, not modelled) all have this range. Awake Sleep Talk fails in
  its script (effect_script_0097.s:5, :11-12); asleep, the called move's defender is re-picked by
  GoToMoveScript (battle_script.c:2120-2126) and a foe-aimed one goes to subscript_no_target,
  last_move still Sleep Talk. Me First (RANGE_SINGLE_TARGET_ME_FIRST, not modelled) differs: with
  randomize FALSE it takes the single-target branch (lib.c:1709-1726), gets BATTLER_NONE and fails
  at :3190 with movePrevByBattler MOVE_NONE. The reference now exempts the whole range from its
  no-target check (it exempted only Nature Power and Counter's family); JAX already agreed. Guard:
  test_invokers test_sleep_talk_with_no_living_foe_still_succeeds (awake and asleep).
