# E2 mechanics ledger: archive (closed batches and designs)

## RESUME bullets retired 2026-11-16

- **History of batch 10's close:** below.
- **Batch 10 on this branch (unmerged), closed and sent to the manager for its merge check.**
  597ad95 recoil/crash (B10-1/2); ff4490b merge-check follow-ups (B10-3); 939bd45 Substitute
  (B10-4); a9806c2 Endeavor / Pain Split (B10-5); b051b45 Knock Off (B10-6); 683e5dd Safeguard
  (B10-7); 5efc1e9 Focus Punch (B10-8); 5e9b960 Encore engine side (B10-9, B10-10); 8df2d38
  Assist (B10-11), Encore refused by coverage for both sides; 6d37c38 the charge turn incl.
  semi-invulnerable and Power Herb (B10-12/13; backlog 234 -> 230); 4d174a8 test_pivot_search
  under 8 GB true peak (conftest prints each test's sampled 50 ms peak; peaks 15.5 -> 4.9 GB);
  ac351e1 merge of main; e3e0d0b review follow-ups (B10-14 Future Sight vs semi-invulnerable incl.
  the Spikes-last range quirk, B10-15 OHKO + Sturdy semi target, B10-16 comment, B10-17 ts_claw /
  ts_custap per battler).
- **Batch 10 checks.** test_damage_classes 46; test_expand + four guards passed
  (at ac351e1); after e3e0d0b: test_batch10 34, test_accuracy_classes 30, test_invokers 16,
  differential seeds 21-24 clean, the four guards 172 passed 1 skipped. Review: approve-with-changes, all fixed.

## History

- Batch 9, the Gardenia split (plan and rulings B9-1..B9-12 below; batch 8 review follow-ups
  B8-6..B8-8 at 0936d40). Commits: 2acd05f (plan), 9370721 (speed / damage abilities, Thunder /
  Blizzard, Fire / Ice Fang), 8705722 (recoil divisors, Kaizo be_seq 48 / 182 modelled, Follow
  Me, Helping Hand, Surf), then the absorb / Rain Dish / Clear Body / Hyper Cutter / Aftermath /
  Poison Heal / Bad Dreams / Dragon Rage / SonicBoom / stat moves / trapping abilities / Dream
  Eater / Payback commit (db0b6d7), and the ledger commit e57e767 (backlog regenerated: the Gardenia
  split's missing groups 78 -> 42, runnable battles 0 -> 9; Metronome's unmodelled callable moves
  115 -> 87). New state: `can_unburden`, `minimized`. Guard tests fixed at f7b2ae1 ("fix(e2):
  guard tests after batches 6-9"); the focused set (tests/data, tests/tools, tests/prep,
  test_accuracy_classes, test_damage_classes, test_faints, test_field_effects, test_learn,
  tests/search/test_stage_b_learn.py): 819 passed, 2 skipped.
- Batch 9 compile, the fast-tier lock file present (its pid 20932 dead; other sessions' pytest
  running beside): `jit(step)` 76.5 s (trace 4.8, compile 71.8), `jit(vmap(step))` 39.6 s; to
  re-time load-free (standing rule).
- Standing (coordinator 2026-11-15): every batch's focused set includes the files above.
- Left of batch 9's plan (Owed: "Batch 9 left"): SolarBeam, Jump Kick's crash, Volt Tackle's
  family (262), Endeavor, Pain Split, Knock Off, Focus Punch, Safeguard, Substitute, Encore,
  Assist, Leaf Guard, Synchronize, Wonder Guard, Pressure, Download, Focus Band, Wiki Berry, the
  terrain entries.

- Tip history: batch 6 and its review follow-ups (470fbef, af6b010, 639ee89); main with E1's
  batches 2c-17 merged in at 23e10f0 (clean: E1's six per-battler fields `curse`,
  `perish_song`, `perish_song_turns`, `yawn`, `lock_on`, `lock_on_target` arrived with no setter;
  the Trace AI refusal lifted at 6f9ad2b, B3-21; the merge's Owed lines closed at f22b818).
- Batch 7, the state the AI reads (B7-1..B7-11): `ai_item_known`, Ingrain, Mean Look / Block /
  Spider Web, Imprison and the public `roosting` turn flag, in both engines, STATE.md, the
  observation and the generator (`strategies._draw_batch7`). E1 reads these fields (Owed: "AI
  owner (E1), batch 7"). E1's six fields stay owed with their moves (Owed).
  Commits: 3c65508 (`ai_item_known`), 4a06792 (the moves and `roosting`), 0dfca0a (merged-tip
  test follow-ups: U-turn / Baton Pass damage classes, the turn-end order test's
  `uproar_disable`), and this ledger commit. The second family (E1's six fields with Curse
  (Ghost), Perish Song, Yawn, Lock-On, Mind Reader) was not started: Owed.
- After batch 7: ceb971c (B7-12, the divergence), 60aba75 (review follow-ups B7-13..B7-17),
  de44007 (the AI switch gate, B7-18, with the manager's go). Next: E1's six fields with Curse
  (Ghost), Perish Song, Yawn, Lock-On / Mind Reader: done in batch 8 (below).
- Batch 8 (B8-1..B8-5): 651bff4 (Lock-On / Mind Reader), 208e996 (Perish Song), ecd1a4f (Yawn),
  cbfcce1 (Curse by a Ghost user), ffa0766 (the opponent-legality test invariant:
  tests/ai_legality.py; first run over the playout, replacement and policy tests and the
  property: no violation; it checks only the concrete picks of host-side calls: the AI picks
  inside jitted playouts, most of them, are traced and go unchecked), and the ledger commit.
  Review follow-ups B8-6..B8-8 in "fix(e2): batch 8 review follow-ups". Backlog regenerated: Metronome's unmodelled
  callable moves 118 -> 115; no other entry (no Roark-split trainer uses these moves).
- Fixed after batch 7 (B7-12): the differential divergence found at f22b818 (a switch replaying
  on the newcomer), from 470fbef (batch 6's pause loop); the JAX engine was wrong.
  Also pre-existing at f22b818: tests/prep/test_variants.py `test_ai_hp_checks_pinned` /
  `_per_policy` expect no `if_party_member_deals_more_damage` among the AI's HP checks (E1's
  merged batch 15 added it): E1's or the prep owner's.
- Batch 7 compile, with the fast-tier lock held (pid 20932): `jit(step)` 59.0 s (trace 4.2,
  compile 54.8), `jit(vmap(step))` 27.6 s; to re-time with the lock free (standing rule).
  Differential: 380 examples at reduced counts (seeds 1-5), no batch-7 disagreement.
- Compile, both with the fast-tier lock held (E1's run, pid 14524): `jit(step)` 35.0 -> 52.5 s, `jit(vmap(step))` 24.2 -> 25.6 s (the end run also beside this batch's slow search tests, so the step figure is load-skewed: logs in `_scratch/e2/b6/` on G:). Load-free size: the step jaxpr 73,192 -> 80,110 equations (+9.5%, `_scratch/e2/b6/eqns.py`, a synthetic state): the 22 `ts_*` / `pivot_battler` leaves widen every whole-state `_where` and reset, plus the list's faint check in `_act`, the scratch writes and the pivot logic. To re-time with the lock free (standing rule).
- Left in the Roark split: Metronome (3 fights, B5-2: 123 of 442 callable moves unmodelled) and
  the terrain entry (2: Nature Power / Secret Power / Camouflage over several maps); the AI's
  U-turn and Baton Pass scripts are E1's (Owed).
- Standing: the doubles conventions below apply to all new code; one review per batch.
- Standing: record `jit(step)` and `vmap` compile time once per batch, measured with the
  fast-tier lock free.
- Standing: at the Roark-split close, also record run-time throughput (steps per second on the
  existing benchmark) against the e139cd1 baseline.
- Standing (process, coordinator 2026-11-14): (1) the fast-tier lock
  (`_scratch/fast_tier.lock`) names the running pytest's pid (`pid=<id>`); write it only after
  that process has started and is confirmed alive, never before launching and never from a
  wrapper that could outlive a failed launch; a lock whose pid is dead, or with no pid while no
  pytest runs, may be deleted by anyone. (2) Every background launch (test run, agent) gets a
  liveness check within two minutes: the process exists and its log grows; one not confirmed
  alive is not running. (3) A batch with no commit for 90 minutes tells the coordinator in two
  lines what it waits on; during a long run keep implementing, never wait silently.

Session E2 builds the battle mechanics the Roark split's trainers need (held items,
abilities, move effects, battle terrain), batch by family. The AI side belongs to E1.
Goal: no Roark-split fight blocked by a "move effect", "held item effect", "ability" or
"battle terrain" entry of `docs/kaizo/mechanics_backlog.md`.

Baseline (main e139cd1): 79 such entries (32 held item, 25 move effect, 21 ability,
1 terrain). Cold compile of `jit(step)` on vanilla Roark: 18.7 s (`jit(vmap(step))`, batch
64: 14.6 s), measured with `_scratch/e2/compile_time.py` on G:.

## Batches

1. Attacker damage items and resist berries: the type-boosting table, the species items
   of `BattleSystem_CalcMoveDamage`, Muscle Band / Wise Glasses, the Choice items, Expert
   Belt, Life Orb, every resist berry, the hold-effect-0 items.
2. Status orbs and the other berries and held items.
3. Abilities.
4. Move effects and Secret Power's terrain.
5. Invokers, Uproar, Future Sight (B5-n).
6. The mid-turn switch decision: U-turn, Baton Pass (B6-n).
7. The state the AI reads: `ai_item_known`, Ingrain, Mean Look / Block / Spider Web, Imprison,
   `roosting` (B7-n).
8. The moves behind the AI's six fields: Lock-On / Mind Reader, Perish Song, Yawn, Curse by a
   Ghost user (B8-n); the opponent-legality test invariant.
9. Gardenia split (B9-n): the plan below.
10. The rest of the Gardenia split (B10-n): the plan below.

## Batch 9: Gardenia split

Entries of the backlog at 697a626 ("Roark and Gardenia splits" table) that block a Gardenia-split
fight, of kind move effect, held item effect, ability or battle terrain; counts are fights
blocked. Out of scope: AI entries (E1), double / multi formats, gauntlet entries. Player-only
groups of the same families are added where they share the code. Order: largest family first.

1. Abilities, stat and speed modifiers (16): Chlorophyll 8, Pure Power 2, Huge Power 1, Swarm 2,
   Unburden 2, Flower Gift 1.
2. Weather accuracy and the fangs (15): Thunder #152 4, Blizzard #260 4, Fire Fang #273 4,
   Ice Fang / Avalanche #274 2, SolarBeam #151 1 (a charge turn).
3. Other moves (15): Follow Me #172 4, Surf #257 2, Knock Off #188 2, Payback / Revenge #230 1,
   Focus Punch #170 1, Safeguard #124 1, Substitute #79 1, Encore #90 1, Helping Hand #176 1,
   Assist #180 1.
4. Abilities on hit, stat drops and switch-in (14): Aftermath 5, Clear Body 3, Hyper Cutter 1,
   Leaf Guard 1, Synchronize 1, Wonder Guard 1, Pressure 1, Download 1.
5. Recoil and crash (12 + 3 ROM): Double-Edge / Wood Hammer #198 6, Volt Tackle family #262 1,
   Hi Jump Kick / Jump Kick #45 2, Kaizo #182 (1/4 recoil) 2 and its ROM entry 2, Kaizo #48
   (1/2 recoil) 1 and its ROM entry 1; vanilla #182 Superpower beside it.
6. Turn-end and absorb abilities (8): Rain Dish 2, Bad Dreams 2, Volt Absorb 2, Water Absorb 1,
   Poison Heal 1.
7. Fixed damage and HP moves (7): Dragon Rage #41 4, Endeavor #189 1, Pain Split #91 1,
   Dream Eater #8 1; player-only SonicBoom #130.
8. Stat moves (7): Stomp #150 3, Belly Drum #142 2, Bulk Up #208 1, Minimize #108 1.
9. Trapping abilities (4): Arena Trap 2, Shadow Tag 1, Magnet Pull 1.
10. Held items (4): Focus Band #65 3, Wiki Berry #15 1; player-only Figy #14 / Aguav #17.
11. Battle terrain (1 + Roark's 2): Secret Power / Nature Power / Camouflage on a trainer whose
    maps give several terrains.

## Batch 10: the rest of the Gardenia split

What batch 9 left (Owed: "Batch 9 left"), grouped by family, with the player-only groups of the
same families. Counts are fights blocked over the whole game (backlog at 3835f68, "All ten
splits"); the Gardenia split's own count in brackets. Order: the largest unlock first, each
family committed on its own, rulings B10-n.

1. Recoil and crash (58 + 2): Volt Tackle's family #262 (recoil 1/3 plus a PROBABILISTIC
   paralysis, battle_lib.c:1551-1567) 58 [1], with Flare Blitz's #253 (37, the same shape with a
   burn) where the decomp shows no other difference; Jump Kick / Hi Jump Kick #45 crash on a miss
   (the miss reads the crit and roll) 2 [2].
2. Held items that draw (53 + 1): Focus Band #65 (a per-hit roll at its three sites, a new Outcome
   field and law) 53 [3]; Wiki Berry #15 (the dislike confusion's duration draw) 1 [1] with the
   player-only Figy #14 / Aguav #17 and the rest of the flavor-berry family.
3. Substitute #79 (48 [1]): the substitute's HP, the hits it takes, and every effect's
   substitute test; with it the Owed King's Rock vs Substitute and Ghost Curse's test.
4. Pressure (33 [1]): PP cost by range, its switch-in message.
5. HP moves: Endeavor #189 (18 [1]), Pain Split #91 (14 [1]).
6. Knock Off #188 (18 [2]): knockedOffItemsMask, Unburden, the Owed Sticky Barb test.
7. The charge turn (16 [1] + player-only): SolarBeam #151, and the two-turn mechanism for every
   charge move the decomp has (Razor Wind #39, Fly #155 and the semi-invulnerable family, Skull
   Bash, Sky Attack where vanilla), with Power Herb #99.
8. Status guards: Synchronize (15 [1]), Safeguard #124 (12 [1]), Leaf Guard (1 [1]).
9. Wonder Guard (with Fire Fang's bypass) and Download (1 [1] each).
10. Encore #90, Focus Punch #170, Assist #180 (1 [1] each).

Out of scope: AI entries (E1), double / multi formats, gauntlets, the terrain entries (data).

### Family 3 design: Substitute (in progress; decomp inventory at
`G:/Farry/Claude Workspace/_scratch/e2/b10/substitute_inventory.md`)

- The move: `EffectId.SUBSTITUTE` (decomp 79). subscript_try_substitute.s:7 fails with a
  substitute up; BtlCmd_TrySubstitute (battle_script.c:5211-5226): q = Divide(maxHP, 4), fails
  when curHP <= q, else the user loses q through UPDATE_HP, `substitute_hp` = q, and its
  VOLATILE_CONDITION_BIND clears (:5222). The AI's scripts are E1's, already transcribed.
- State: `substitute_hp != 0` stands for VOLATILE_CONDITION_SUBSTITUTE (every break zeroes both
  but Pursuit's, whose stale HP nothing reads without the bit). SELF_TURN_FLAG_SUBSTITUTE_HIT
  lasts one action (BattleContext_Init, battle_lib.c:2014, after every action, bcp.c:3998): a
  carry of the hit loop (`MoveHit.sub_hit`, like `focus_armed`), not a state field.
- The hit (bcp.c:3359-3378): with a substitute and damage, the substitute takes min(damage, its
  HP) (hitDamage: drain and recoil read it; shellBellDamageDealt too), the flag sets, and
  UpdateHP returns early: no False Swipe cap, Focus Sash / Band, Bide store, damage-taken fields
  (Counter, Mirror Coat, Metal Burst), lastDamageTaken, SYSCTL_MOVE_HIT (Life Orb, Shell Bell),
  resist berry. Later hits of a multi-hit go on, on real HP once it broke, the flag still set.
- Gates: CheckSubstitute (battle_script.c:9173-9186) = the bit or the flag; SubstituteWasHit
  (battle_lib.c:6259) = the flag. Default-chance secondaries on the defender (lib.c:1570-1587),
  CHECK_HP_AND_SUBSTITUTE (Bind), the status / flinch / confuse / leech seed / lock-on / mean
  look / yawn / curse / swagger subscripts, ChangeStatStage's drop on a foe (bit only: direct
  fails, indirect silent), Dream Eater (no damage), SmellingSalt (no x2, no cure), the contact
  ability (lib.c:4209), the on-hit items (:5378), Pluck (:5546), Intimidate (subscript :13),
  Toxic Spikes on a Baton-Passed substitute. Not gated: self-recipient secondaries, ON_HIT,
  Taunt, Disable, Attract, Foresight, Roar, Perish Song, U-turn, sound moves.
- Special paths: Future Sight (subscript_future_sight_damage.s:13-24, no flag); Pursuit on a
  switcher (subscript_pursuit.s:20, :56-73: only a break changes it); confusion self-hit and Leech
  Seed's drain bypass it; Struggle's recoil is maxHP/4 anyway.

### Family 6 design: Knock Off (next; the AI is E1's, branch claude/e1-prep-on-e2)

- effect_script_0188.s: an indirect ON_HIT subscript_knock_off.s after CalcCrit / CalcDamage (a
  table hit). The subscript ends with nothing done when the defender is behind a substitute (:5,
  CheckSubstitute), has Multitype (:6, BATTLEMON_ABILITY), holds the Griseous Orb (:7), or has its
  moveEffectsData.quickClaw (:8) or custapBerry (:9) set; then BtlCmd_TryKnockOff
  (battle_script.c:6608-6632): no item, nothing; an ignorable Sticky Hold with an item, "{0}'s {1}
  made {2} ineffective!" (TAG_NICKNAME_ABILITY_MOVE); else heldItem = ITEM_NONE and the side's
  knockedOffItemsMask |= the slot's bit (:6626-6627).
- New state: `knocked_off` int8[SIDES, SLOTS] (the mask; it survives switches): InitBattleMon
  clears a knocked-off slot's item and canUnburden on entry (battle_lib.c:161-163), the Sticky Barb
  transfer skips a knocked-off attacker (:5390, :7380), Thief / Covet / Trick / Fling read it
  (script.c:5105, :6348). `quick_claw_flag` int8[SIDES] (moveEffectsData.quickClaw): set by
  CalcTurnOrder when the Quick Claw activates and the sort does not ignore it (battle_lib.c:1283,
  :1349), cleared by the holder's BEFORE_MOVE subscript_check_quick_claw.s:22-23 and with
  MoveEffectsData on a switch; it outlives a turn whose holder did not FIGHT. Check how the engine
  keeps the Custap flag (step `custap`, `ts_custap`) and whether it needs the same persistence.
- Knock Off leaves canUnburden as it was: an armed Unburden holder doubles its speed after it.

### Design: a chosen move replaced mid-turn (Encore's takeover; manager decision (a), 2026-11-15)

Problem. FightCommand (bcp.c:1866-1873) runs an encored battler's encored move whatever it chose, so
a faster Encore this turn replaces the move the turn-start laws and sampler assumed. The same
gap already exists, unflagged, for Sleep Talk's split variants (B5-13 Disable, B7-13 Imprison):
`call_laws` / `call_variants` / `call_consistent` enumerate the alternative called move exactly,
but `sample_outcome` draws every move-dependent field from the turn-start prediction
(`used_moves`), so a split variant whose called move differs in crit stage, effect chance or
duration law is sampled at the wrong rates. One mechanism covers both.

Revised after review (approve-with-changes, 2026-11-15): the first draft listed `hit` as
rate-free (it is the roll's class over the move's thresholds, outcome.py:967-968), missed `call`
(Encore can lock a battler into Metronome / Assist / Sleep Talk / Copycat: Encore is not
NO_METRONOME, and sCannotEncoreMoves leaves them), `fang_flinch` and `future_hit`, chose the copy by
a test that can disagree with the laws, and encoded the variant in `call` (which Metronome's and
Assist's `call % count` would corrupt). The plan now:

1. One event, shared by laws, engine and sampler: `encore_taken[battler]` = its encore_move was 0
   at the turn start and is set and held (moves[encore_slot] == encore_move) when its action
   starts, not struggling (bcp.c:1866-1873); a new `step_events` flag. With the existing
   `disabled_at_move` / `sealed_at_move` (`_split_events`) it decides every replaced-move variant.
2. Laws (no new Outcome fields): a TAKEN variant per battler whose foe can make it happen (a foe
   using Encore, or an invoker that can call it: over-approximate; `call_consistent` drops the
   rows that did not happen), computed wholly from a hypothetical state with the encore set (as
   `_disabled_state` does for Disable): moves, effects, crit odds, damage classes, thresholds and
   the battler's own call law when the encored move is an invoker. The variant tag is a field
   beside `call` (not an offset of it). Possible only when the target is not encored at the turn
   start, its last move is encorable, held, has PP and differs from the chosen one.
3. Sampler only: draw Outcome A as today; draw B = A with the move-dependent fields of each
   possibly-replaced battler (`hit`, `crit`, `secondary`, `fang_flinch`, `duration`, `call`,
   `future_hit`) redrawn for its alternative move; run the turn's events on A and return B for a
   battler whose event fired. Exact: the event is fixed before that battler's action starts, and
   the redrawn fields are read only after it. The same pass closes the Sleep Talk Disable /
   Imprison split's sampler gap (today drawn from the turn-start prediction).
4. Engine: unchanged (it already runs the replaced move; it only emits the event).
5. Doubles: a short list of alternatives per battler, each keyed by its event (two foes can Disable
   and Encore one battler in a turn); singles needs one per kind.
6. Tests: the hand case (1 child vs 33) as a law test and a frequency test; a Sleep Talk + faster
   Disable frequency test with calls of different accuracy (fails today); Encore into Metronome or
   Assist; each shown failing without its part.
7. Size: about two days (the reviewer's estimate for the first draft was 2-3; this route is
   smaller). Playout cost: a second event pass on turns where a replacement is possible.

### Family 10 rest: Encore and Assist (next; decomp sites, not yet modelled)

- Encore (effect 90, effect_script_0090.s: a direct ON_HIT subscript_encore_start.s ->
  BtlCmd_TryEncore, battle_script.c:4725-4748): fails (MOVE_STATUS_FAILED) when the defender is
  encored already, its last move (movePrevByBattler) is none, is not in its slots, cannot be
  encored (Move_CanBeEncored) or has no PP; else encoredMove / encoredMoveSlot and encoredTurns =
  RandNext() % 5 + 3 (3-7: a new duration law). Readers: command selection (bcp.c:435-450, the
  move forced), BattleSystem_CheckInvalidMoves (battle_lib.c:2302-2304: other moves invalid, so
  the legal actions and the AI's choices change), the action (bcp.c:1869-1883: the encored move
  used, or the encore dropped when the slot no longer holds it), the end-of-turn count
  (bcp.c:1592-1610: ends at 0 or when the move's PP is 0), Pursuit (script.c:6905), Sucker Punch
  (:7292), Me First (:7391). New state: the encored move, slot and turns per battler (volatile,
  Baton Pass?: check condition / moves.h); the AI's Encore scripts are E2's to transcribe.
- Assist (effect 180): the invoker family (Sleep Talk / Metronome / Copycat, B5-1): a random
  move of the user's party mates' known moves minus the exclusion list; find its command and the
  call law shape before starting.

## Batch 11: a chosen move replaced mid-turn, Encore's AI, review items

Implements the ledger's "Design: a chosen move replaced mid-turn" as approved (B11-1..B11-3),
Encore's AI with both coverage refusals lifted (B11-4), the manager's batch 10 review items
(B11-5..B11-8) and a law bug the takeover test found (B11-9). Rulings under "Rulings".

## Batch 6 design: mid-turn switch decision (U-turn, Baton Pass)

# Mid-turn switch decision (U-turn, Baton Pass): design, rev. 2

**Where the game decides.** U-turn's script only sets `MOVE_SIDE_EFFECT_ON_HIT | ..._ATTACK_THEN_SWITCH_OUT` (effect_script_0228.s:5); the switch runs as the hit's secondary effect, `ONE_HIT_TRIGGER_SECONDARY` (bcp.c:3484-3494). `subscript_attack_then_switch_out.s` runs, in order: the checks: the defender's side is not blacked out and the attacker has a replacement (:5-6, which also flags it `BATTLER_STATUS_SWITCHING`, battle_script.c:5876-5880); the hit's contact ability (:7) and one pivot item (:11, see the open points); Destiny Bond (:15-34), Grudge (:40); no switch if the attacker fainted (:48); the opponent's Pursuit (:54) and the user's status restore (:56-57); `SYSCTL_UTURN_ACTIVE` set (:68), then the party list (:71; `ShowParty` / `WaitMonSelection`, battle_script.c:2378-2470). After the pick: `SwitchAndUpdateMon`, the hazards and a faint check (subscript_show_party_list.s:9-24).

Baton Pass: subscript_baton_pass.s :5 (no replacement: the move fails, :19-20), status restore (:7-8), `SYSCTL_BATON_PASS` (:16), the list (:17); no Pursuit. The rest of the action then runs **on the attacker's battler slot, which now holds the newcomer**:
`AfterMoveMessage` continues with Shaymin's form check, Rage, `TriggerAbilityOnHit` (guarded) and `CheckExtraFlinch` (bcp.c:3497-3530). The extra flinch reads the newcomer's King's Rock, has no U-turn guard (:4630-4650), and U-turn carries `MOVE_FLAG_TRIGGERS_KINGS_ROCK` (res/moves/u_turn/data.json:25); `AfterMoveEffects` (:3638ff): its `TRIGGER_SWITCH_IN_EFFECTS` (:3672-3682, `BattleSystem_TriggerEffectOnSwitch`: the newcomer's Trace, weather, Intimidate, status cure, items) comes before the attacker / defender item checks (:3684ff); then `LOOP_FAINTED` (:3820). The party list itself runs only the hazards and a faint check (subscript_show_party_list.s:22-24).
`UpdateMoveBuffers` (:3917ff): no Choice lock for U-turn under the flag or a successful Baton Pass (:3927-3928). `moveProtect`, `movePrev` and `movePrevByBattler` are written onto the newcomer (:3934-3944), after `UpdateAfterSwitch` cleared them (battle_lib.c:2119, :2123); `MoveEnd` (:3962-3998): status recovery by ability, the exp payout, the battle-over check, `TriggerEffectOnSwitch` again (:3972; the announced flags stop repeats, battle_lib.c:3795), then `BattleContext_Init` (:3998), which clears both flags (system_control.h:33) and resets `aiSwitchedPartySlot` to 6 (battle_lib.c:2015; also at TurnEnd, bcp.c:1855).

Guarded by `SYSCTL_UTURN_ACTIVE`: the contact abilities in `TriggerAbilityOnHit` (battle_lib.c:4219-4362), Sticky Barb and Jaboca on hit (:5393, :5403), the Choice lock (bcp.c:3927), Shell Bell (:4739), Life Orb (:4761). Not guarded: King's Rock, the move buffers, the faint loop, `MoveEnd`.

**Opponent's pick:** `Task_TrainerShowPartyMenu` reads `aiSwitchedPartySlot` (6 here: `MoveEnd`'s init; it is set only by AI switch choices, e.g. trainer_ai.c:3340, :3460, :3478, :3519, :3537), then `BattleAI_PostKOSwitchIn`, then the first living slot (battle_display.c:4543-4556).

## 1. Pause and resume
- **State:** `pivot_battler: int8` (`1 + battler`, 0 = none; convention 3). Only one battler can owe a pivot at a time: each list is inside one action, and the flag clears at the switch (battle_script.c:2511) or the faint (battle_lib.c:2222).
- **`TurnScratch`** (a field of `BattleState`, all zero when not paused so states merge), with:
  - `cursor` (`turnOrderCounter`), `order[B]` (`battlerActionOrder`), the turn's real `actions[B]`;
  - the turn-start picks `move/slot/locked/item/custap[B]`, and `pursued` (`1 + battler`);
  - the events later actors read: `protect_ok`, `roosted`, `phys/spec/last_taken`, `dragged` `[B]`;
  - the paused action's tail inputs: the move, `landed`, the defender's damage-taken flags, `alive_at_act: uint8[2]` (per-party bitmask of HP > 0 at the action's start). Both `_faint_check` and `pay_exp_leveled` read only `before.hp > 0` from `before` (step.py `_faint_check`; levelup.py `pay_exp_leveled`, its `fainted` line); level, max HP and the learn work come from `after`;
  - the search's `power_mul` and `ai_switched` from the real actions, and `pick_mul` (`powerMul` at the list: 20 after the :54 Pursuit, battle_script.c:6928, else 10).
- **Pause:** inside `_use_move`, after the pre-list steps. `qc_roll` is zeroed at the pause; it was read when the turn order was fixed and is read again only at the next turn.
- **Resume:** `step(paused, actions, outcome)` with `actions[pivot] = MOVE_ACTIONS + slot`. It skips the turn prologue: the `learn_pin` reset, `_record_last_move`, `action_order`, `_quick_claw_shown` (step.py:1928-1931 in today's tree), bag items, the Pursuit pre-switch hit and natural cure. Then it re-enters the `fori_loop` at `cursor`:
  - **Switch-in:** `_act`'s existing switch branch feeds the move branch with `resuming = True`; in pivot mode `_send_out` runs only `entry_hazards` and the faint check, not the rest of `hooks.on_switch_in` (Trace, weather, Intimidate, status, items: hooks.py:483-496).
  - **Rest of the action:** `_use_move` runs only its tail. It rewrites `last_move` / `protect_count` onto the newcomer: `_VOLATILE` includes `last_move` (step.py:193, :202), so `_reset_battler` cleared it. Then King's Rock with the newcomer's item; the rest of `on_switch_in` once, at the `TRIGGER_SWITCH_IN_EFFECTS` slot, before the item checks (guarded ones skipped); `_faint_check`, `pay_exp` (its `pending` rule, step.py:1476, reads `TurnScratch.actions`, never the switch-shaped resume actions), `_move_end`.
  - **Then** the later positions and the end of the turn.
- **What `post_ko_switch_in` sees:** the paused state, before the faint check. A defender the U-turn KO'd still has HP 0, its Camouflage types and status, and no exp has been paid. That is the game's state at :71.

## 2. Enumerate, observe, search
- **Turn-start chance node:** each `Outcome` element is read at most once, so the turn splits at the pause. The turn-start node still enumerates the draws that only the second half reads (`end_speed_tie`, `quick_claw`, `shed_skin`, later actors' rolls, King's Rock), because whether the pause happens depends on first-half draws. Paused rows that differ only there step to the same bytes and merge. This costs rows, not tree width.
- **Resumed chance node:** `outcome_laws` on a paused state uses the one branch `TurnScratch.order` from `cursor` on, and reads King's Rock from the newcomer named by the switch action.
- **Observation:** `turn_events` on the first call gives the first half's events (later positions masked; `order` and `quick_claw` come from here). On the resume it gives only the resumed part, with `ACT_SWITCH` and the slot for the pivoting side and `order` read from `TurnScratch`. Each half's `observe` uses its own events, so Stage B groups each chance node by its own observation.
- **Search:**
  - `expand._turn` (expand.py:217-218) and playout's `_turn` read `power_mul` and `ai_switched` from `TurnScratch` when `before` is paused, never from the switch-shaped actions.
  - `replace_mul` keeps its end-of-turn role.
  - Stage B `_pending` gains a `"pivot"` kind, whose answers are turn requests through `expand`. `decision_actions` offers `legal_replacements(state, 0)`. The opponent's answer is `MOVE_ACTIONS + post_ko_switch_in(paused, pick_mul)`, refused as today.
  - Playout loops `step` while `pivot_battler != 0`, with the same `Outcome`.

## 3. The three cases
- **Opponent's U-turn:** a pause with one forced opponent action; no search decision.
- **Player's U-turn:** a `"pivot"` decision made after the first half's observation.
- **Baton Pass:** under `PIVOT_BATON_PASS`, `_reset_battler` keeps:
  - the stat stages (battle_lib.c:97-101);
  - `VOLATILE_CONDITION_BATON_PASSED` (condition.h:72);
  - `MOVE_EFFECT_BATON_PASSED` (moves.h:46-48), including `MOVE_EFFECT_NO_CRITICAL`, with Lock-On's timer refreshed (battle_lib.c:2073-2085);
  - the substitute HP, Lock-On / Mean Look targets and the Perish Song / Magnet Rise / Embargo / Heal Block counters (:2107-2115).

  A passed Power Trick swaps the newcomer's Attack and Defense again (:2136-2140). There is no Pursuit (no call in the subscript). Each item the engine models today is carried; the rest come with their mechanic.

## 4. Doubles (section 5)
- One `pivot_battler`, so there is no multi-battler resume and no joint mask.
- `cursor` is a position in the stored order, not "second" (convention 5).
- `_pivot` takes the attacker battler (convention 1).
- In doubles `PostKOSwitchIn` picks its defender with `BattleSystem_RandomOpponent`, which draws `RandNext() & 1` (battle_lib.c:4183-4187, used at :7949). So the opponent's pivot pick is a chance node there: a new `Outcome` element indexed by battler.

## 5. Reuse the faint-replacement decision?
Yes for the decision: the slot encoding, `legal_replacements`, `decision_actions`, `replacement_policy`, `post_ko_switch_in`, and Stage B's pending plumbing. No for the transition: `step.replace` / `replacement_pass` run after the turn (TurnEnd payout, no draws), and a pivot resumes inside an action. New pieces: `pivot_battler`, `TurnScratch`, the halt mask, the resume entry and `_use_move`'s tail entry.

## 6. Compile time
The resume is extra `fori_loop` iterations through the existing `_act` (dynamic start at `cursor`), with the switch branch's output feeding the move branch, plus a pre-list / tail split inside the one `_use_move`. It is not a second traced `_act` (that would cost roughly a second `_use_move`, tens of seconds). Added: about 30 small leaves on about 90 (every `_where` selects over them), the masks and the Baton Pass selects. Estimate +1 to +3 s on `jit(step)` (50.7 s), unmeasured. Size: about 350-450 lines engine, about 120 search, about 200 reference, plus tests.

**Open points, resolved:** `_VOLATILE` holds `last_move` (step.py:193, :202), so the switch-in clears it and the resume must rewrite it (bcp.c:3944). `TriggerHeldItemOnPivotMove` (battle_lib.c:7336-7386) tests four independent `if`s that each overwrite `*subscript`, and :11 runs it once. So at most one effect runs: the last match (defender's Sticky Barb > defender's Jaboca > attacker's Life Orb / Shell Bell).

**Answer:** pause inside `_use_move` before the list with `pivot_battler` and `TurnScratch`; resume re-enters the same `fori_loop` at `cursor`, switching in and then running the action's tail on the newcomer.
Reuse the faint-replacement decision but not its transition; the search reads `power_mul` / `ai_switched` from `TurnScratch`; expected compile cost +1 to +3 s.

## Performance, after the engine is complete (parked, manager decision 2026-11-14)

- **Measurement.** `jit(step)` was re-timed back to back with the fast-tier lock free. It grew
  gradually: 18.4-18.8 s at e139cd1, 19.7-20.3 at 5b65dcd, 23.9-30.8 at ac2b497, 28.9-36.0 at
  726e3f9. `vmap` grew from 14 to 19.5 s. Over the same span the jaxpr grew from 50.7k to 64.4k
  equations (+27%). The earlier 39-60 s readings were taken under machine load. Single-commit
  noise is up to 7 s.
- **Shape.** Every fight compiles and runs all 39 move-effect handlers, plus the ability and item
  logic, as `jnp.where` masks. Production compiles one program per game. Stubbing the handlers
  cuts 10% of the equations and 25% of compile time (25.4 -> 19.0 s). `_reset_battler`,
  `_restore_types` and the whole-state `_where` together are about 21% of the equations.
- **Options, in this order:**
  - (c) Narrow `_reset_battler`/`_where` to the fields that actually change. Cheap.
  - (a) A static per-fight feature set that skips absent handlers at trace time. Only valid with a
    provable upper bound on what a fight can reach (moves learned mid-search, items, Trace); means
    one program per fight.
  - (b) `lax.switch` on the effect id. Cuts compile, but under `vmap` every branch is still
    evaluated at run time.
- **The pivot paths (E2 batch 6), measured trace-only on a synthetic 2v2 state
  (`_scratch/e2/b6/prog_count.py`), ffcdc54 -> after the batch:** `step` 73,169 -> 80,110 jaxpr
  equations, one playout 236,143 -> 262,065 (+11%), `leaves._forced_turn` 127,788 -> 152,679
  (+19%), with only `playout.step_through`'s loop picking and drawing (one `step`, one
  `sample_outcome`, one `pivot_pick` per program; before, 310,620 / 183,980); the rest is `step`'s
  own growth and the loop's `pivot_pick` (`post_ko_switch_in`, `replacement_policy`, also traced
  by `finish_turn`'s replacements: one loop for both pick points would trace them once). Chance-node width unchanged (4,616 rows over the
  Roark start pairs at full and 10 HP, both trees). Option, parked (manager 2026-11-15): a static
  per-fight flag that traces the pause paths only when set. Its condition: the set that turns the
  paths on must be provable from the fight's setup: either party's moves and learnable moves, plus
  every move that can call or copy another (Metronome, Sleep Talk, Copycat, Mirror Move, Me First,
  Assist, Mimic, Sketch, Transform); it covers every party member on both sides plus the player's
  learnable moves, and keys on `move_effect in {U_TURN, BATON_PASS}`, not on move names (Kaizo
  remaps moves); a pivot move reached while the flag is off must fail loudly, never silently do
  nothing. The batch 6 review confirmed the bound sound given the current refusals.

