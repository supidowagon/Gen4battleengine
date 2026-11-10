# K3b2 rules from the decomp (read-only research, 2026-11-08)

Worktree `bold-mendeleev-477a50`, branch `claude/kaizo-k3b` (HEAD f5c27b9 plus uncommitted
coverage.py/state.py edits by another agent, not read for rules). No repository file was changed.

Paths: decomp files are under `external/pokeplatinum/`. bcp = `src/battle/battle_controller_player.c`,
lib = `src/battle/battle_lib.c`, script = `src/battle/battle_script.c`, ai = `src/battle/trainer_ai/trainer_ai.c`,
ais = `src/battle/trainer_ai/script.s`, eff = `res/battle/scripts/effects/`, sub = `res/battle/scripts/subscripts/`,
fbdt = `src/field_battle_data_transfer.c`. Engine paths are under `src/nuzline/`.

Every rule below was read in the decomp at the cited line. Where the code leaves a question open
it says **UNCERTAIN**, with what would settle it. Survey/audit statements were treated as
hypotheses. Three of them are corrected here (§0.2).

---

## 0. Overview

### 0.1 Summary table

| mechanic | size | new Outcome fields | new state | depends on |
|---|---|---|---|---|
| Thunder Fang (275) | S-M | `fang_flinch[2]` (the paralysis roll reuses `secondary`) | none | existing flinch/status code |
| Protect (111) | M | `protect[2]` | none (the `moveProtect` reset folds into the existing `protect_count`) | `last_move` semantics |
| Present (122) | M | `present[2]` | none | joint factor with crit/roll (§10) |
| Attract (120) + Cute Charm | M | `infatuation[2]` (Cute Charm reuses `contact_proc`) | `infatuated[2]` | switch/faint clearing; Roar |
| Quick Claw | M-L | `quick_claw[2]` (reserved today) | `qc_roll[2]` (hidden; carries `speedRand` into the next turn) | order.py seam, root chance node |
| Roar (28) | L | `roar_level[2]`, `roar_target[2]` | none | mid-move switch-in path, Attract/types/protect clearing |
| Camouflage (213) | S-M | none | `terrain` (field scalar, shared with NP) | terrain pipeline |
| Nature Power (173) → Tri Attack | M (+ pipeline) | `tri_attack[2]` | `terrain` | terrain pipeline, called-move plumbing |
| Terrain pipeline | M-L | none | `terrain` | ROM reader, coverage, rom_diff |
| Accuracy fix (finding §0.2c) | S-M | `hit` becomes an interval class | none | none |
| Enumeration restructure (§10) | M-L | none | none | every new field |

### 0.2 Corrections to the survey and audit

a. **Quick Claw is not a same-turn draw.** `speedRand[b]` is redrawn in
`PRE_MOVE_ACTION_STATE_SPEED_RNG` (bcp:853-857). That step runs **after**
`BattleControllerPlayer_CalcTurnOrder` (bcp:707-790), which sets `command = CHECK_PRE_MOVE_ACTIONS`
at bcp:790. So turn N's action order reads the value drawn during turn N-1 (turn 1 reads the
battle-start draw, `BattleContext_InitCounters` lib.c:2019-2025, called from bcp:185).
Turn N's own draw is read by:
- every `SortMonSpeedOrder` of turn N (bcp:888 before each action and after the last one, which
  gives the end-of-turn order);
- the turn-end re-sorts (bcp:251, :263, which order the replacement switch-ins);
- the AI's `IfSpeedCompare*` at the next command selection (ai:1109, :1121);
- turn N+1's `CalcTurnOrder`.

So survey §3.2's "reserved `quick_claw[2]`, law (4,1)/5" is right about the law but wrong
about timing. The value must be carried in state (§5).

b. **"Accuracy 100 still rolls" (audit W5) is true, but the threshold-100 roll can never miss.**
`(RandNext()%100)+1 > hitRate` (bcp:2976) with hitRate 100 is never true. So `hit` is a 1-valued
factor unless a modifier lowers the threshold. In this fight those modifiers are: Gible's Sand Veil
in the permanent sand (x80/100, bcp:2926-2928), accuracy below 100 (DoubleSlap 90, Sing 70), and
accuracy/evasion stages from Sand-Attack (see c).

c. **Existing exactness bug (pre-K3b2, reachable in this fight).** `outcome.hit` is a boolean drawn
against the turn-start threshold (outcome.py:108-152, :191; step.py:492). The docstring's premise
"no M1 move changes accuracy or evasion" (outcome.py:14-16) is false. Effects 23 `ACC_DOWN`
(Sand-Attack), 16 `EVA_UP` (Double Team), 24 and 73 are mapped (tools/effect_map.py:81, :85-86,
:149). The decomp computes the hit rate at the move's own `TryMove` (bcp:2865-2979), with the stages
of that moment. The probe-save Sandshrew knows Sand-Attack (move 28). When Sandshrew acts before a
Roark Pokemon, that Pokemon's move later in the same turn has rate 75 (stage -1, `_ACC_NUM[5]`).
The engine gives it 100.
- Fix (exact): store `hit` as an **interval class** of the roll `u = RandNext()%100+1`. Let
  t1 < t2 < … be the thresholds the move can have at its TryMove this turn (turn-start stages, plus
  every stage change an earlier action can make). The class of `u` is which interval it falls in.
  The law is the interval widths over 100. The comparison is done in `step` against the live
  threshold. Width: 2 classes per distinct reachable threshold, usually 2-3.

### 0.3 Draw-modelling conventions used below

- **RandNext is treated as i.i.d. per call** (the project's convention, enumerate.py:322-325). So
  the order of draws within a turn only matters for *whether* a draw happens. The values of
  different draws are independent. Every "when" below says which conditions gate a draw's existence.
- **`RandNext() % n` is not exactly uniform.** RandNext is a u16 (battle_system.c:1308-1312), and
  65536 is not a multiple of 3, 5, 6 or 100. The project treats `% n` as uniform (outcome.py `draw`).
  The exact weights are given where they differ (Quick Claw, Roar, Tri Attack). Decide once whether
  to keep the convention. Protect (masks against 0x7FFF etc.) and Present (`& 0xFF`) are exact powers
  of two.
- **Script before accuracy** (bcp:3209-3216 loads the effect script, :3237-3296 TryMove, :3298-3310
  primary effect). CalcCrit, CalcDamage, Present, `Random` (Tri Attack) and `GetTerrainMove` run
  before the accuracy roll. Secondary effects run after `UpdateHP`, in `AfterMoveMessage`
  (bcp:3486-3495), and are not run on `MOVE_STATUS_DID_NOT_HIT` (miss, protected, immune, failed;
  include/constants/battle/moves.h:81-85).

---

## 1. Thunder Fang (Kaizo effect 275 `FLINCH_PARALYZE_HIT`, 90 BP, acc 100, chance 10, contact, CAN_PROTECT)

### (a) Rules
- eff/effect_script_0275.s: indirect side effect `ON_HIT | TO_DEFENDER | PARALYZE_OR_FLINCH`, then
  `CalcCrit`, `CalcDamage`. This is a plain damaging hit plus one indirect subscript.
- `BattleSystem_TriggerSecondaryEffect` lib.c:1524-1529: the `ON_HIT` branch loads the subscript
  whenever `moveStatusFlags` has no `MOVE_STATUS_NO_EFFECTS`. There is **no outer chance roll** and
  no substitute check at this level. bcp:3486-3495 also requires that it did not miss (`DID_NOT_HIT`).
- sub/subscript_paralyze_or_flinch.s: `CheckEffectActivation` then `Call PARALYZE`, then
  `CheckEffectActivation` then `Call FLINCH_MON`. Both checks always run in this order.
- `BtlCmd_CheckEffectActivation` script.c:8363-8383: proceeds iff `RandNext()%100 < effectChance`
  (Serene Grace doubles it) **and** the side-effect Pokemon (the defender) has `curHP != 0`. The draw
  comes first (`&&` order), so both draws are made whenever the subscript runs.
- Paralysis (sub/subscript_paralyze.s, INDIRECT path, every failure silent):
  - Limber, ignorable, so Mold Breaker pierces it (:3);
  - Leaf Guard in sun (:4-6);
  - Shield Dust (:9);
  - substitute (:14);
  - already paralyzed or any other status (:16-17);
  - `MISSED|SEMI_INVULNERABLE` (:19);
  - Safeguard on the target's side (:20).
  - There is **no Electric-type immunity** in this script. Ground-type targets are immune to the
    hit itself through the type chart, so the subscript never runs on them.
  - Sets `MON_CONDITION_PARALYSIS`, message "X is paralyzed! It may be unable to move!"
- Flinch (sub/subscript_flinch_mon.s):
  - `IfMovedThisTurn` target → no flinch (:2). `Battler_MovedThisTurn` (lib.c:5365-5368) is
    `battlerActions[PICK_COMMAND] == MOVE_END`, so a target that already acted **or switched** this
    turn cannot flinch.
  - Substitute → silent (:3).
  - Inner Focus (ignorable) → silent for an indirect flinch (:4, :26-30: the message is only for
    Fake Out or DIRECT).
  - Shield Dust (:6).
  - Else sets `VOLATILE_CONDITION_FLINCH`.
- The contact-ability trigger (`TriggerAbilityOnHit`) runs after the secondary (bcp:3514-3524).
  Static / Flame Body / Cute Charm on the defender then roll (§4, lib.c:4214, :4321, :4339).
- Engine mapping: a new `EffectId.FANG` (params `(Status, 0, 0)`). It also covers vanilla 273
  Fire Fang (burn) and 274 Ice Fang (freeze), whose scripts are the same shape (check before relying
  on this). `effects.py` reuses `_inflict` (non-direct) for the first roll and `_flinch` for the
  second. `_flinch`'s `h.proc` would then read the new field.

### (b) Draws (per Thunder Fang action)
Order: crit (CalcCrit) → damage roll (CalcDamage) → accuracy (TryMove) → [UpdateHP] → paralysis
roll → flinch roll → contact-ability roll.

| draw | law | read when |
|---|---|---|
| `secondary[s,0]` (existing) | `RandNext()%100 < 10`: 10/100 | the hit landed, was not immune, and the target is alive after the hit, has no status at that moment, no Limber (or the attacker has Mold Breaker), no substitute, no Safeguard |
| `fang_flinch[s]` (new) | 10/100 | the hit landed, was not immune, the target is alive, the target has **not** acted/switched yet this turn, no Inner Focus (or Mold Breaker), no substitute |

Tight sound relevance (turn-start judgement):
- `secondary`: the user uses the move AND the target (active, or the incoming one if switching)
  - is not Ground (chart immunity; the type is read at hit time, so Camouflage matters, §7),
  - has status NONE at start or can be cured before the hit (holds Lum/Chesto/Pecha-type berry,
    switches in, Natural Cure switch),
  - has no Limber (unless Mold Breaker),
  - has no Safeguard on its side.
- `fang_flinch`: the same, plus "the user can act before the target". That is false if the target's
  action is a switch or item (it acts first, bcp:755-771), or the target's priority is strictly
  higher, or the order is fixed with the target first for every value of `speed_tie` and the stored
  `qc_roll`.

### (c) State
None new. `flinched` exists (`statusVolatile` FLINCH).

### (d) Observation
Paralysis shows in `after` (status). The flinch shows as the target's existing `flinched` event.
Both are already effect-gated. Nothing new.

### (e) AI (K3c)
Roark's Gible uses it, so the per-move script path for effect 275 is not transcribed (an existing
coverage line). The AI damage estimate uses power 90 (no special case).

### (f) Size and dependencies
S-M. Fang family EffectId, a new Outcome field (`fold_in(key, 16+)`), laws, relevance, reference
engine, strategies. Independent of the other mechanics.

---

## 2. Protect (effect 111; Kaizo Protect +3, acc 0, RANGE_USER; Detect 197 shares the effect)

### (a) Rules
- eff/effect_script_0111.s: a DIRECT side effect `CHECK_HP | TO_ATTACKER | PROTECT`, not ON_HIT.
  `TriggerPrimaryEffect` lib.c:1492-1505 runs it if the attacker has HP and there is no `NO_EFFECTS`.
  Protect has accuracy 0, so there is no accuracy roll (bcp:2900-2903). Its defender is the user.
- sub/subscript_protect.s: `TryProtection`. On success: attack message and animation, then the
  prepared message "X protected itself!". On failure: `MOVE_STATUS_FAILED`, which leads to
  subscript_missed.s with "But it failed!" (:97-100) after its `PrintAttackMessage`.
- `BtlCmd_TryProtection` script.c:5149-5190:
  1. If `moveProtect[attacker]` is not Protect, Detect or Endure (182/197/203), set
     `protectSuccessTurns = 0` (:5154-5158).
  2. `moreBattlersThisTurn = (waitingBattlers != 1)` (:5160-5165). `waitingBattlers` counts the
     battlers whose action is not yet `MOVE_END`, the current one included (bcp:881-886). So
     **the battler acting last this turn always fails.** In singles that means moving second.
  3. Success iff `sProtectSuccessRate[n] >= RandNext()` **and** `moreBattlersThisTurn` (:5167-5168).
     The table is {0xFFFF, 0x7FFF, 0x3FFF, 0x1FFF} (:5128-5133). Against a u16 this is exactly
     1, 1/2, 1/4, 1/8. The draw is made before the last-to-move test.
  4. On success: `turnFlags.protecting = TRUE` (Protect effect), and `protectSuccessTurns++` capped
     at index 3 (:5182-5184). On failure: `protectSuccessTurns = 0` (:5186).
- `moveProtect[attacker]` is written in `UpdateMoveBuffers`:
  - `moveCur` if `SYSCTL_ATTACK_MESSAGE_SHOWN`, else `MOVE_NONE` (bcp:3936-3942). It is not written
    on `SYSCTL_REUSE_LAST_MOVE` (Magic Coat/Snatch, not in the pool).
  - Reset to NONE by Magic Coat/Snatch reflection (bcp:3073, :3096), by switch-in
    (`UpdateAfterSwitch` lib.c:2120) and by faint (`CleanupFaintedMon` lib.c:2198).
  - `protectSuccessTurns` is in `moveEffectsData`, which is zeroed on switch-in (lib.c:2099-2102) and
    faint (lib.c:2179-2182).
- Blocking (`CheckMoveHitOverrides` bcp:3001-3014) is checked **after** the accuracy roll (TryMove
  order bcp:3253-3268). The move is blocked if the defender has `turnFlags.protecting`, and the move
  (`moveCur`) has `MOVE_FLAG_CAN_PROTECT`, and it is not a non-Ghost Curse, and it is not the charge
  turn of a multi-turn move. A blocked move: `Battler_UnlockMoveChoice(attacker)` and
  `MOVE_STATUS_PROTECTED`.
- `turnFlags` (`protecting`) is cleared for everyone at the turn end (`BattleSystem_SetupNextTurn`
  lib.c:2231) and for a fainted battler (lib.c:2190-2193).
- The message for a blocked move (sub/subscript_missed.s:18-63): "{defender} protected itself!".
  `PROTECTED` is tested before `MISSED` (:11-18 vs :157), so an accuracy miss into a Protect also
  prints "protected itself". The exception: for Fake Out, Snore, Nightmare, Fling, Last Resort,
  Psycho Shift and Natural Gift, if the move's own `FAILED` is set, "But it failed!" prints instead
  (:20-27, :58-59).
- Kaizo move flags: every damaging move of Roark and of the pool has CAN_PROTECT, and so do Attract,
  Roar, Toxic and Tri Attack. Nature Power itself has no CAN_PROTECT, but the check uses `moveCur` =
  Tri Attack, so Nature Power is blocked. Curse, Recover, Light Screen, Rock Polish, Stealth Rock,
  Aqua Ring and Camouflage are not blocked.
- Self-Destruct into Protect: the user's HP is set to 0 in the script before TryMove
  (eff/effect_script_0007.s), so the user faints and the target is untouched.
- **Fake Out vs Protect.** Kaizo gives both priority +3, so speed (with Quick Claw and Lagging Tail)
  decides.
  - Protect user first: Fake Out is blocked, and prints "But it failed!" if it was also not the
    user's first turn, else "protected itself!".
  - Fake Out first: the Protect user flinches. It never reaches UpdateMoveBuffers with an attack
    message, so `moveProtect = NONE` (the chain resets).
  - In this fight no cross-side pair exists: Fake Out (Skitty) and Protect (Wormadam) are both the
    player's.
- **Engine encoding (no new field).** Fold the `moveProtect` reset into `protect_count`. At the end
  of each move action by side s, set `protect_count[s] = 0` unless the move that showed its attack
  message is in {182, 197, 203}. Item actions and switches leave it untouched (a switch zeroes it
  through `_VOLATILE`). `TryProtection` then reads `protect_count` directly. The AI's
  `LoadProtectChain` (ai:2539-2552) returns `protectSuccessTurns` when `moveProtect` is in the family,
  else 0, which is the folded value.
  - **UNCERTAIN (minor):** whether `last_move` (= `movePrevByBattler`, `SYSCTL_MOVE_SUCCEEDED`) and
    "attack message shown" can differ for a Protect action. They agree on every path read here (a
    failed Protect still prints the attack message in subscript_missed). A test pinning a failed
    Protect followed by a Protect next turn (rate 1) settles it.

### (b) Draws
| draw | law | read when |
|---|---|---|
| `protect[s]` (new) | success with probability 2^-n, n = folded `protect_count[s]` (0..3) | s uses a Protect-effect move, gets past the status checks, **and is not the last to act**. n = 0 gives probability 1, a single value |

Relevance: `uses_move & effect == PROTECT & protect_count[s] > 0 & s can be first`. If s certainly
acts second, the move fails without the draw mattering.

Downstream: a successful Protect makes the foe's `hit`, crit, roll, secondary and contact rolls
irrelevant when the foe acts later. That is a conditional-enumeration saving (§10).

### (c) State
- `protect_count` (exists): `MoveEffectsData.protectSuccessTurns` with the `moveProtect` reset
  folded in.
- `turnFlags.protecting` is a per-turn local inside `step` (STATE.md row `BattleContext.turnFlags`
  stays "not needed").
- `BattleContext.moveProtect` (STATE.md:400 "later") becomes "folded into `protect_count`".

### (d) Observation
New per-side event flags:
- `protected`: this side's move was blocked ("X protected itself!", where X is the foe);
- `protect_ok` / `protect_failed`: the user's own outcome. Success prints "X protected itself!".
  Failure prints "But it failed!" after "X used Protect!".
The act is shown in both cases.

### (e) AI (K3c)
- ais:1701 `Expert_Protect` (the AI's own Protect, none on Roark).
- ais:3545/:3575 `LoadProtectChain AI_BATTLER_ATTACKER`.
- ais:3964/:4010 `IfMoveEffectKnown DEFENDER PROTECT`: the player's seen Protect changes charge-move
  scoring. Not reached by Roark's moves.
- ais:2700 is the attacker's own.

### (f) Size and dependencies
M. Fake Out interplay is order-only. It shares the "can act first" helper with Thunder Fang.

---

## 3. Present (effect 122; Kaizo 1 BP, acc 100, Normal, physical, CAN_PROTECT, no contact)

### (a) Rules
- eff/effect_script_0122.s: `Present _004`, then `CalcCrit`, `CalcDamage` (damage path).
  `_004` (heal path) sets the DIRECT effect `ON_HIT | PRESENT_HEAL` with `SYSCTL_IGNORE_IMMUNITIES`.
  **No CalcCrit or CalcDamage runs on the heal path.**
- `BtlCmd_Present` script.c:5789-5806: `rnd = RandNext() & 0xFF`.
  - rnd < 102 (`255*40/100`) → power 40;
  - rnd < 178 (`255*70/100`) → 80;
  - rnd < 204 (`255*80/100`) → 120;
  - else `hpCalcTemp = Divide(DEFENDING_MON.maxHP, 4)` and jump to the heal path.
  - The heal amount is the **target's** max HP / 4, computed in the script.
- Heal path:
  - The direct ON_HIT effect runs only without `NO_EFFECTS` (lib.c:1485-1490). A miss or a Protect
    gives the usual miss/protected message and no heal. A Ghost target is healed
    (IGNORE_IMMUNITIES).
  - sub/subscript_present_heal.s: attack message, then a Heal Block check (not in the pool), then
    `BATTLE_SUBSCRIPT_RECOVER_HP`.
  - sub/subscript_recover_hp.s: at full HP the message is "X's HP is full!" plus `NO_MORE_WORK`.
    Otherwise the heal message is "X regained health!". `SKIP_SPRITE_BLINK` is set, so
    subscript_update_hp's resist-berry check is skipped.
  - `damage = 0`, so `UpdateHP` does nothing (bcp:3347): no Focus Sash, no contact ability (Present
    is not contact anyway), and no `physicalDamageTaken`.
- Damage path: `movePower` 40/80/120 in the normal formula. It is a Normal-type physical hit (Rock
  resists; Delibird's Adaptability does nothing because Present is not STAB for it).
- AI: power 1 means the AI never computes Present damage (audit W2; ai:1035-1036 etc.).
  ais:204 routes the AI's own Present to `Basic_CheckNonStandardDamage`.

### (b) Draws
Order: Present roll → (damage path only: crit → damage roll) → accuracy.

| draw | law | read when |
|---|---|---|
| `present[s]` (new): 0/1/2 = 40/80/120, 3 = heal | 102/256, 76/256, 26/256, 52/256 (exact) | s uses Present and gets past the status checks |
| `crit[s,0]`, `roll[s,0]` | as today | **only when `present[s] != 3`** |

Enumerate `(present, crit, roll)` as one joint factor of 3*32 + 1 = 97 values instead of 128 (§10).
When `crit_odds` gives a single value (Lucky Chant), it is 3*16 + 1 = 49.

### (c) State
None.

### (d) Observation
- New event `present_heal`: 1 = healed ("regained health"), 2 = "HP is full".
- The damage path's power is never printed. It is seen only through the HP change, which is already
  in the observation.

### (e) AI (K3c)
- The AI scoring of the player's Present: it is a power-1 move, so no damage term.
- The engine must keep `ai_damage` gated by `is_damaging` (audit W2).

### (f) Size and dependencies
M. A new field, the heal branch, the joint factor, the reference engine and the AI gate. Present is
used by the player only (Delibird current, Skitty relearner).

---

## 4. Attract (effect 120) and Cute Charm (ability 56)

### (a) Rules
- eff/effect_script_0120.s: DIRECT `TO_DEFENDER | INFATUATE`, not ON_HIT. `msgBattlerTemp =
  attacker`. `TriggerPrimaryEffect` lib.c:1492-1505 requires the defender alive and no `NO_EFFECTS`.
  So a miss (acc 100; Gible's Sand Veil gives 80) or a Protect stops it.
- sub/subscript_infatuate.s:
  - Oblivious (ignorable) blocks it: "X's Oblivious prevents romance!" and `NO_MORE_WORK` (:2, :55-67).
  - `MISSED|SEMI_INVULNERABLE` → fail (:3).
  - `TryAttract` (:4) → fail (`_108`: `MOVE_STATUS_FAILED`, so "But it failed!") or "X fell in
    love!" (:9-10).
  - On success it sets `SELF_TURN_FLAG_INFATUATED` on the target (:49).
  - There is no substitute or type check in this script. **UNCERTAIN:** whether a substitute blocks
    Attract elsewhere. Not reachable: no substitute move in the pool or on Roark.
- `BtlCmd_TryAttract` script.c:5727-5742 fails if:
  - the genders are equal; or
  - the target already has any `VOLATILE_CONDITION_ATTRACT` bit; or
  - either Pokemon is genderless.
  Else it sets `FlagIndex(source) << ATTRACT_SHIFT` on the target, so the infatuation is **tied to
  the source battler**.
- Cute Charm (`TriggerAbilityOnHit` lib.c:4339-4355; the defender holds it). It procs if all of
  these hold, in this `&&` order, and **the `RandNext()%10 < 3` draw is the last operand**:
  - the attacker has HP;
  - the attacker has no ATTRACT bit;
  - no `NO_EFFECTS`;
  - not the first turn of a charge move;
  - not U-turn;
  - the defender took damage this action (`physical/specialDamageTaken`);
  - the move makes contact;
  - the defender still has HP.
  Then it runs subscript_infatuate as SIDE_EFFECT_TYPE_ABILITY with `sideEffectMon = attacker` and
  `msgBattlerTemp = defender`. Oblivious, gender or already-attracted failures are silent (`_082`,
  `_108` → `_117`). The message is "X's Cute Charm infatuated Y!" (:29-30).
  - The substitute check at lib.c:4208-4210 returns before any ability proc.
- Status check: `CHECK_STATUS_STATE_ATTRACT` (bcp:2669-2690) comes after CONFUSION and PARALYSIS and
  before BIDE (enum bcp:2378-2392).
  - `RandNext() & 1` nonzero → subscript_infatuated.s ("X is in love with Y!" plus animation) and
    the move goes on.
  - Zero → subscript_immobilized_by_love.s ("in love" message, then "X is immobilized by love!",
    `UnlockMoveChoice`), and `commandNext = UPDATE_MOVE_BUFFERS`. No attack message, so
    `last_move = 0` and the Protect chain resets.
- Clearing:
  - the source switches out or is dragged out (`UpdateAfterSwitch` lib.c:2087-2091) or faints
    (`CleanupFaintedMon` lib.c:2174-2176);
  - the holder itself switches (its `statusVolatile` reset, lib.c:2070) or faints (lib.c:2165);
  - Oblivious cures it at switch-in or move end (`RecoverStatusByAbility` lib.c:4426).
  - Mental Herb (a hold effect) also cures it. No Oblivious or Mental Herb holder exists here, so
    coverage refuses them.
- Kaizo data: Skitty is female (probe save box 1 slot 3, gender 1). Roark's mons are male except
  Corsola (F). Attract and Cute Charm work against Bonsly, Lileep, Gible, Shuckle and Cranidos, not
  against Corsola.

### (b) Draws
| draw | law | when / relevant iff |
|---|---|---|
| `infatuation[s]` (new): 1 = immobilized | 1/2 (`RandNext()&1 == 0`) | s uses a move, gets past sleep/freeze/flinch/Disable/Taunt/confusion/paralysis, and is infatuated at that moment. Relevance: `uses_move & (infatuated[s] at start, or the foe uses Attract and can act before s with compatible genders)` |
| `contact_proc[d,i]` (existing, extended) | 3/10 | as today for Static/Poison Point/Flame Body, plus a Cute Charm holder on side d when the attacker is not infatuated at start, both genders are known and different, and the attacker lacks Oblivious |

### (c) State
- `infatuated[2]` int8 = `statusVolatile` ATTRACT bits (singles: "infatuated with the foe's active
  battler").
- Add it to `_VOLATILE` (cleared on own switch-in).
- `_switch_in(side)` and the faint cleanup of side s must also clear `infatuated[1-s]`, like
  `_release_bind`.
- STATE.md: `statusVolatile` ATTRACT moves from "later" to `infatuated`.

### (d) Observation
- Add the non-zero-ness of `infatuated` to the visible projection ("fell in love" is printed).
- New events: `in_love` (the status check showed the message) and `immobilized` (lost the roll).
- Extend `contact_ability` to Cute Charm.

### (e) AI (K3c)
`IfVolatileStatus AI_BATTLER_DEFENDER, VOLATILE_CONDITION_ATTRACT` in ais:733, :2817 (Sand Tomb
binding move +1: **Roark's Shuckle reads the player's infatuation**), :3113, :3558, :3870, :4144,
:4547. ais:202 `Basic_CheckCannotAttract` is for the AI's own Attract (none). ais:4446 lists Cute
Charm in the desirable-abilities table (Skill Swap/Role Play only).

### (f) Size and dependencies
M. Roar and faint cleanup must clear the source-tied bit.

---

## 5. Quick Claw (item 217, `HOLD_EFFECT_SOMETIMES_PRIORITY`, param 20)

### (a) Rules
- `BattleSystem_CompareBattlerSpeed(b1, b2, ignoreQuickClaw)` lib.c:1188-1445:
  - If one battler has 0 HP, the other is first (lib.c:1204-1211), before anything below.
  - Quick Claw activates iff `speedRand[b] % (100/param) == 0` (lib.c:1278-1286, :1344-1352), i.e.
    `% 5 == 0`. **The flag is computed whatever `ignoreQuickClaw` is.** Only the write of
    `moveEffectsData.quickClaw` (read by Thief/Covet, script.c:5114, and by the before-move animation)
    is skipped when ignoring.
  - Priorities are compared only when `ignoreQuickClaw == FALSE` (lib.c:1375-1398). Otherwise both
    are 0.
  - Within equal priority (lib.c:1401-1440):
    - both Quick Claw active → speed (no Trick Room; tie → `RandNext()&1`);
    - one active → it goes first (even under Trick Room);
    - then Lagging Tail (both holders → reversed speed), then Stall, then Trick Room, then speed.
- Who calls it:
  - `CalcTurnOrder` (bcp:772-786): any two battlers with the same command.
    `ignoreQuickClaw = skipPriority = (command != FIGHT)`. Quick Claw therefore decides **switch vs
    switch** and **item vs item** too, since the flag is still evaluated. The holder is the battler
    active at ordering (the outgoing Pokemon for a switch).
  - `SortMonSpeedOrder` (lib.c:3364-3383, ignore TRUE): at bcp:888 before every action and after the
    last one (end-of-turn order), and at bcp:251/:263 (turn end, ordering replacement switch-in
    effects).
  - `SortMonActionOrder` (lib.c:7276-7300) after Trick Room activates (bcp:3990-3993; not in pool).
  - AI `IfSpeedCompareEqualTo/NotEqualTo` (ai:1102-1124, ignore TRUE).
  - AI `LoadBattlerSpeedRank` (ai:2081, a pairwise sort with ignore TRUE; added in K3b2
    Task 8's review round).
- **Timing** (§0.2a): `speedRand` is drawn at battle start (lib.c:2024) and at each turn's
  pre-move step (bcp:853-857), which is after `CalcTurnOrder`. The same value orders turn N's
  end-of-turn pass, the turn-end replacement switch-ins and the AI's speed checks at the next
  command selection, and turn N+1's action order.
- Visible effect (sub/subscript_check_quick_claw.s, first step of `BeforeMove`, bcp:3140-3143):
  - Only if the battler's `moveEffectsData.quickClaw` flag is set, which happens only in
    `CalcTurnOrder` between two FIGHT battlers.
  - If it is not the last to act (`WAITING_BATTLERS != 1`): **a held-item animation only, with no
    text**. The "let it move first!" text is Custap's path (:9-16).
  - Then the flag is cleared.
  - If Quick Claw activates but the holder moves second (lower priority), nothing shows.
- Quick Claw vs Lagging Tail: Quick Claw wins (lib.c:1402-1411 before :1412-1421). Already seamed in
  `order.player_second` (order.py:87-102).

### (b) Draws and the state carry
- `quick_claw[s]` (Outcome, reserved today): this turn's `speedRand[s] % 5 == 0`. Law 1/5 by
  convention. Exact: `RandNext()%5 == 0` is 13108/65536.
  - Read this turn by `SortMonSpeedOrder` (end-of-turn order, the `end_of_turn_order` speed tie
    logic).
  - Stored into the new state `qc_roll[s]` for the AI's next decision and next turn's
    `action_order`.
- `action_order` this turn reads `state.qc_roll`, never the Outcome.
- Relevance (sound, merging): `quick_claw[s]` is relevant iff side s's battler after the actions
  holds Quick Claw, or side s has a pending replacement and a living Quick Claw holder.
  - Otherwise store `qc_roll[s] = 0`. This is exact: nothing reads that `speedRand` before it is
    redrawn, because a switch-in's first read is the end-of-turn sort of the turn it enters.
  - With "always store whenever a holder is alive", every pair of the fight doubles while Prinplup
    lives, and several pairs overflow (§10). Use the tight rule.
- **Battle start:** the lead's Quick Claw is decided by the InitCounters draw before turn 1's AI
  decision and order. The root belief must hold both values (1/5, 4/5) with equal observation
  whenever a lead holds Quick Claw. The probe save leads with Prinplup.
  - **UNCERTAIN (design):** how Stage B builds a non-degenerate root belief. Settle in the K3b2 plan
    with the Stage B owner. The information-set rule (observe.py:4-5) already handles hidden state.

### (c) State
- `qc_roll[2]` int8 = `BattleContext.speedRand % (100/param)` == 0. Param 20 is the only
  SOMETIMES_PRIORITY item; register it in game_constants.
- STATE.md `speedRand` (:420) changes from "not needed" to `qc_roll`.
- `MoveEffectsData.quickClaw` stays "later" (only the animation and Thief/Covet read it).

### (d) Observation
New event `quick_claw` (the holder's animation) when both chose FIGHT, the holder's `qc_roll` is
1, and it was not last to act. The order of actions is already observed. The end-of-turn order is
not observed (observe.py:51-52), so coarser, never finer.

### (e) AI (K3c)
- `IfSpeedCompare*` (ai:1109, :1121) reads Quick Claw through `speedRand`, and draws `RandNext()&1`
  on a speed tie (the AI's comparison consumes the battle RNG).
- `LoadBattlerSpeedRank` (ai:2056-2093, the compare at :2081) sorts the battlers with
  `CompareBattlerSpeed(..., TRUE)`, so it reads Quick Claw too (and a tie's `RandNext()&1`).
- Roark's EXPERT paths that use it: Selfdestruct ais:1864, Recover ais:2650, Rock Polish ais:2070,
  and the speed-down/paralysis lines. The AI policy needs `qc_roll` of the player's active.

### (f) Size and dependencies
M-L. order.py and the end-of-turn order (Lagging Tail seam exists), a new state field, a root
chance node, the AI reader, relevance. Independent of the other mechanics.

---

## 6. Roar (effect 28 `FORCE_SWITCH`; Kaizo acc 100, priority -6, CAN_PROTECT, sound)

### (a) Rules
- eff/effect_script_0028.s: DIRECT `ON_HIT | FORCE_TARGET_TO_SWITCH_OR_FLEE`. The direct ON_HIT
  branch runs iff there is no `NO_EFFECTS` (lib.c:1485-1490). So any of these stops it with no
  message of its own:
  - an accuracy miss (acc 100, Sand Veil gives 80, stages);
  - Protect;
  - Soundproof, through `TriggerImmunityAbilities` (lib.c:3523-3530; Roar is in `sSoundMoves`
    lib.c:3473-3475; ignorable; "blocked by soundproof").
  - Status move, so no type-chart immunity (Ghosts are hit).
- sub/subscript_force_target_to_switch_or_flee.s:
  1. Semi-invulnerable → fail (:2). Suction Cups (ignorable) or Ingrain → anchor message and
     `NO_MORE_WORK` (:3-4, :79-90). Battle type exactly `DOUBLES|2vs2|AI` → fail (:5). None of these
     apply to Roark.
  2. `TryWhirlwind` (:6) → fail `_074` ("But it failed!").
  3. Attack message; `TryRestoreStatusOnSwitch BTLSCR_DEFENDER` (Natural Cure on the outgoing,
     script.c:9139) then status NONE.
  4. `DeletePokemon`. In a trainer battle: health box out, `SwitchAndUpdateMon BTLSCR_FORCED_OUT`,
     send-out, "X was dragged out!", `SWITCHED_MON = DEFENDER`, `BATTLE_SUBSCRIPT_HAZARDS_CHECK`.
     There is no Pursuit (unlike sub/subscript_switch_pokemon.s:11) and no faint call: a hazard
     faint is caught by the move's faint loop.
- `BtlCmd_TryWhirlwind` script.c:5241-5320 (trainer battle, singles branch):
  - Count the defender party's eligible slots: species set, not an egg, HP > 0. The active counts
    (:5287-5296). If eligible ≤ 1 (`maxActiveMons` = 1) → fail with **no draw** (:5299-5300).
  - Else `BattleSystem_CanWhirlwind` (lib.c:3072-3086):
    - if `attacker.level >= defender.level` → TRUE with no draw;
    - else TRUE iff `(((RandNext() & 0xFF) * (La + Ld)) >> 8) + 1 > Ld / 4`;
    - FALSE → fail.
  - Then repeat `i = RandNext() % partyCount` (`max = defenderPartyCount`, fainted slots included)
    until `i != selectedPartySlot[defender]` and slot i is eligible (:5303-5311).
    `switchedPartySlot[defender] = i`.
- `BtlCmd_SwitchAndUpdateMon` script.c:2488-2525: `selectedPartySlot = i`, `InitBattleMon` (types
  from the party data, lib.c:114-115; stages reset), `SwitchSlots`, then `UpdateAfterSwitch`
  (lib.c:2040-2155). `UpdateAfterSwitch` does all of the following:
  - clears the outgoing's volatiles and `moveEffectsMask` (lib.c:2070-2071);
  - clears Attract/Bind bits **sourced from** that battler (lib.c:2087-2096);
  - zeroes `moveEffectsData` (protect count, Quick Claw flag, disable, taunt…; lib.c:2099-2102);
  - sets `fakeOutTurnNumber = totalTurns+1` (lib.c:2116);
  - sets `moveProtect`, `moveHit`, `movePrevByBattler` = NONE (lib.c:2120-2124);
  - clears the AI's known moves, ability and item for that battler (lib.c:2152-2154).
  The outgoing's toxic counter is lost exactly as for a voluntary switch: the party write-back
  strips `MON_CONDITION_TOXIC_COUNTER` (src/battle/battle_controller.c:1481-1483).
  **UNCERTAIN (minor):** confirm the write-back message is emitted for a forced-out battler too.
- Hazards on the dragged-in Pokemon: sub/subscript_hazards_check.s (Toxic Spikes, Spikes, Stealth
  Rock; Magic Guard, Levitate, Flying, Gravity and Iron Ball gates). Switch-in abilities (Intimidate,
  Sand Stream…) run at `MoveEnd` through `BattleSystem_TriggerEffectOnSwitch` (bcp:3969-3976). The
  dragged-in Pokemon's held-item check follows as the defender in `AfterMoveEffects` (bcp:3691).
  - In this fight the player cannot set hazards on Roark's side and Roark's mons have no
    switch-in ability, so these are inert. Implement them generally by reusing
    `hooks.on_switch_in` with the forced-switch order.
  - **UNCERTAIN (minor):** whether the berry check after a hazard hit happens inside
    HAZARDS_CHECK's UPDATE_HP or in `AfterMoveEffects`. Read `subscript_update_hp.s` and bcp:3638-3760
    before reusing `on_switch_in` unchanged.
- Kaizo levels: Vulpix is L16, so `La >= Ld` against every Roark Pokemon (L13-16). **The level draw
  never happens in this fight.** It does happen when the user is lower-levelled.

### (b) Draws
Order: accuracy → [immunity/Protect] → level draw (only if La < Ld) → slot draws (rejection loop).

| draw | law | relevant iff |
|---|---|---|
| `roar_level[s]` (new) | success weight k/256, k = #{r in 0..255 : ((r*(La+Ld))>>8)+1 > Ld//4} (exact) | s uses Roar, La < Ld, the target side has ≥ 2 eligible |
| `roar_target[s]` (new): party slot | rejection-loop law. Exact: P(i) ∝ w_i = #{r in u16 : r % partyCount == i} over the eligible bench (Roark: partyCount 6, w = 10923 for slots 0-3, 10922 for 4-5). Convention: uniform | s uses Roar and the eligible bench has ≥ 2 Pokemon (with 1, the slot is forced) |

The ≤ 5 values are state-dependent support, like `hit`. Gate both on "the target may not be
protected/immune" for tightness.

### (c) State
None new. Uses `active`, `_switch_in`, `_forget` (side 0 only: Roark never Roars), `_entered`,
the Attract clearing (§4), the Camouflage type restore (§7), and the `protect_count` reset
(`_VOLATILE`). Transient decomp fields (`switchedPartySlot`) stay "not needed".

### (d) Observation
- The dragged-in species and HP bar are already visible through `active`. Add a `dragged` event
  (the slot) so it is told apart from a voluntary switch.
- A failure shows "But it failed!" (act shown). Soundproof, Suction Cups and Ingrain have their own
  messages (refused by coverage until a holder exists).

### (e) AI (K3c)
- The dragged-in Roark Pokemon gets `fake_out_turn = turn+1`, so `LoadIsFirstTurnInBattle` is true
  for its next decision (ai:2487 per state.py docs).
- The AI's replacement logic is not invoked: no faint.
- The player's Roar enters `ai_seen_moves`.
- ais:153/:1650 are for the AI's own Roar (none).

### (f) Size and dependencies
L. A mid-action switch-in path inside `_use_move` in both engines, the state-dependent slot law,
observation, and the reference engine. It depends on §4 and §7 clearing.

---

## 7. Camouflage (effect 213; acc 0, RANGE_USER, Snatch flag, no CAN_PROTECT)

### (a) Rules
- eff/effect_script_0213.s: `TryCamouflage _016`. On success: `moveEffectsMask |=
  MOVE_EFFECT_CAMOUFLAGE`, message "X transformed into the TYPE type!" (an ON_HIT indirect print).
  On failure: `MOVE_STATUS_FAILED` ("But it failed!").
- `BtlCmd_TryCamouflage` script.c:7997-8022:
  - Multitype → fail.
  - `terrain = BattleSystem_GetTerrain()`; if `terrain > TERRAIN_SPECIAL (= TERRAIN_AARON,
    include/constants/battle/terrain.h)`, it becomes TERRAIN_SPECIAL.
  - `type = sTerrainCamouflageType[terrain]` (include/data/terrain/to_type.h).
  - If `MON_IS_NOT_TYPE(attacker, type)` (neither type1 nor type2 equals it; include/battle/common.h:54)
    → `type1 = type2 = type`. Else fail.
- Building/special → Normal. Wormadam-Trash (Bug/Steel) → Normal/Normal.
- The types persist while the battler stays in. `InitBattleMon` re-reads the party types on the
  next switch-in (lib.c:114-115). `MOVE_EFFECT_CAMOUFLAGE` is cleared with `moveEffectsMask`
  (lib.c:2071).
- Taunt: power 0, so it fails under Taunt (bcp `CHECK_STATUS_STATE_TAUNT`).
- Every type reader reads `BattleMon.type1/type2`, through `BattleMon_Get` → `Battler_MonType`
  (lib.c:7618-7640; Arceus is the only override).
  - Battle sites:
    - STAB `MON_HAS_TYPE(attacker, moveType)` lib.c:2592;
    - the type chart lib.c:2625-2636 and the damage params lib.c:6657-6660;
    - Rock SpD in sand lib.c:6927;
    - Ghost Curse lib.c:3050;
    - sand/hail chip script.c:5570-5571;
    - Stealth Rock script.c:8310-8311;
    - Toxic Spikes script.c:7643;
    - status subscripts (burn/freeze/poison/badly_poison, Leech Seed Grass);
    - the Flying/Steel checks lib.c:3218, :3243, :5504, :5528.
  - AI sites: ai:946-958, :967-982, :999, :3334-3335, :3454-3532.
- **Engine encoding (write-through, few edits):** on success, write `types[side, active] = (t, t)`.
  Restore `types[side, slot]` from `tables.species_types[species_form_row]` (as state.py:319 does)
  when the battler leaves: switch-out, forced switch (§6) and faint cleanup.
  - Every existing reader already indexes `types[side, active[side]]`: damage.py:160, :285, :405-424;
    effects.py:270, :583, :681, :882, :914; step.py:1083; ai/commands.py, switching.py, replacement.py;
    reference engine.py:185-1508. So the readers need no edits.
  - The AI reads the player's **party** types for a non-active slot (ai/switching.py:280 reads the
    active one, replacement.py:84 reads the AI's own). It must never see a stale Camouflage type,
    which the restore-on-leave rule guarantees.
- Effects on Wormadam-Trash once it is Normal:
  - it loses its sandstorm immunity (Steel), so the sand chip applies (step.py:1083 reads the new
    types);
  - it loses its poison/Toxic immunity;
  - Fighting is super effective (Brick Break x2);
  - it gains Normal STAB for Nature Power→Tri Attack;
  - Leech Life loses Bug STAB.

### (b) Draws
None.

### (c) State
- `terrain` (field scalar, shared with §8).
- `BattleMon.type1/type2` → `types` (STATE.md:157-158): add "written for the active battler by
  Camouflage, restored from the party row when it leaves".
- `moveEffectsMask` CAMOUFLAGE stays "later": its only reader is the AI's own Camouflage scoring
  (ais:1023), and no Roark Pokemon has it.

### (d) Observation
Add the active battlers' current types to the visible projection, or an event `camouflage = type`.
The message names the type.

### (e) AI (K3c)
The AI reads the player's active types for chart and STAB decisions. Write-through gives it the
Normal type. ais:252 `Basic_CheckCamouflage` is for the AI's own.

### (f) Size and dependencies
S-M. Needs `terrain`. The restore hook is shared with Roar and the faint cleanup.

---

## 8. Nature Power (effect 173; acc 0, priority 0, RANGE_SINGLE_TARGET_SPECIAL, no CAN_PROTECT)

### (a) Rules
- eff/effect_script_0173.s:
  1. `PrintAttackMessage` ("X used Nature Power!") and animation.
  2. `GetTerrainMove`: script.c:8036-8048, `msgMoveTemp = sTerrainMove[clamp(terrain)]`
     (include/data/terrain/to_move.h; BUILDING and SPECIAL give Tri Attack).
  3. "Nature Power turned into Tri Attack!".
  4. `GoToMoveScript FALSE` (script.c:2111-2134):
     - clears `SKIP_ATTACK_MESSAGE`, so "X used Tri Attack!" prints too;
     - `moveCur = Tri Attack`;
     - `defender = BattleSystem_Defender(attacker, Tri Attack, TRUE, 0)`, redirection abilities,
       chosen target;
     - if the defender is NONE → `subscript_no_target`; else jump to Tri Attack's effect script.
- The rest is Tri Attack's (Kaizo 161: 95 BP, acc 100, chance 30, special, Normal, CAN_PROTECT),
  with `moveCur` = Tri Attack.
  - The accuracy check (bcp:3253-3257), Protect (bcp:3007-3009), the type chart, the damage and
    the secondary chance all read `moveCur`.
  - **Not re-run:** the status checks, Taunt (Nature Power has power 0, so a taunted user's
    Nature Power fails at the status check), Disable, the PP decrement (Nature Power's PP only;
    `BeforeMove` is skipped), and priority (Nature Power's 0).
- eff/effect_script_0036.s (Tri Attack):
  - `Random 2, 3` (`BtlCmd_Random` script.c:3200-3210: `RandNext() % 3 + 3`): 3/4/5 = burn,
    freeze, paralyze (include/data/move_side_effect_subscripts.h);
  - `+ 0x80000000` (= `MOVE_SIDE_EFFECT_TO_DEFENDER`, include/constants/battle/side_effects.h:12);
  - the result is the indirect flags; then `CalcCrit`, `CalcDamage`.
- The secondary goes to the generic branch of `TriggerSecondaryEffect` lib.c:1572-1585:
  `RandNext()%100 < chance`, then the defender is alive, no substitute was hit and there is no
  `NO_EFFECTS` → the status subscript. Each status has its own immunities:
  - burn: Fire type, Water Veil;
  - freeze: Ice type, sun, Magma Armor;
  - paralysis: Limber;
  - any of them: an existing status, Safeguard.
- Records:
  - `movePrevByBattler` = `moveTemp` = **Nature Power** (bcp:3943-3947; `moveTemp` is set in
    `FightCommand` bcp:1861-1898 and never changed by `GoToMoveScript`). So Disable disables Nature
    Power (`DEFENDER_LAST_MOVE`, include/battle/common.h:19; script.c:4586-4593), and the AI's
    seen-moves list gets Nature Power (`TrainerAI_RecordLastMove` ai:2705-2717).
  - `moveProtect` = `moveCur` = Tri Attack (bcp:3938). This only matters for the chain: Tri Attack
    is not Protect.
  - `moveHit[defender]` = `moveCur` = **Tri Attack** (bcp:4499-4500).
  - `moveCopied` uses `moveTemp` and only if it has the Mirror Move flag (bcp:4480-4485); Nature
    Power has none.
- **Engine plumbing:** add `called_move(tables, state, move) = where(effect == NATURE_POWER,
  terrain_move[state.terrain], move)`.
  - Use it for: `hit_threshold`, `crit_odds`, the damage, the effect dispatch, the secondary chance,
    the PROTECT/contact flags, `move_hit`, `has_target` (the re-picked defender) and the Present-style
    relevance.
  - Keep Nature Power for: `selected_move`, PP, Taunt/Disable, priority, `last_move` and
    `_record_last_move`.
  - Today `_SELF_DEFENDER` (step.py:153-160) treats RANGE_SINGLE_TARGET_SPECIAL as self. Nature
    Power must instead record Tri Attack on the foe's `move_hit`.
  - **UNCERTAIN (minor):** a Nature Power used when the foe already fainted this turn.
    `GoToMoveScript` picks the defender with `BattleSystem_Defender(..., randomize TRUE)`, then
    TryMove's `HasNoTarget` (bcp:2306-) should print "no target". Confirm `NO_TARGET`'s macro for
    this case, and whether `moveHit` is written then.

### (b) Draws (per Nature Power action, terrain BUILDING)
Order: [Nature Power: none] → Tri Attack `Random` → crit → damage roll → accuracy → (Protect) →
UpdateHP → secondary roll → (no contact: Tri Attack has no contact flag).

| draw | law | relevant iff |
|---|---|---|
| `tri_attack[s]` (new): 0/1/2 = burn/freeze/para | 1/3 each by convention. Exact `u16 % 3`: 21846/21845/21845 of 65536 | the secondary can apply (only joint with `secondary=1`: a joint factor of 1 + 3 = 4, not 6) |
| `secondary[s,0]` | 30/100 | the hit lands, the target is alive and can take at least one of the three statuses |
| `hit`, `crit`, `roll` | as Tri Attack's (Sand Veil 80 vs Gible) | as for any damaging move |

The target's `thaw` and `full_para` relevance gains "the foe's Nature Power→Tri Attack can inflict
it". Tighter: only through `tri_attack == freeze/para` and the foe acting later (§10 joint factor).

### (c) State
`terrain` (§9). No other state: `msgMoveTemp` and `moveCur` are per-action scratch (STATE.md:337,
:396-397 stay "not needed").

### (d) Observation
`act_id` stays Nature Power. Add a `called` event (161) and the Tri Attack hit events. The status
shows in `after`.

### (e) AI (K3c)
- Seen-moves has Nature Power (power 0, so no AI damage term).
- `moveHit` = Tri Attack: ai:3656-3661 reads `MOVE_DATA(moveHit).power`.
- The AI's own Nature Power: none.

### (f) Size and dependencies
M once `terrain` exists. It depends on §9 and on Camouflage for STAB interplay (tests).

---

## 9. Terrain pipeline

### 9.1 What the decomp does
- Every field battle DTO is filled through `FieldBattleDTO_Init` (fbdt:279-283) →
  `FieldBattleDTO_InitFromGameState` (fbdt:234-270). With a field system it calls
  `SetBackgroundAndTerrain` (fbdt:245). Without one it uses header BG and `TERRAIN_BUILDING`
  (fbdt:247-249). Trainer battles started from the overworld have a field system.
- `SetBackgroundAndTerrain` (fbdt:496-505):
  - `background = MapHeader_GetBattleBG(location->mapHeaderID)`, **the header of the matrix cell
    the player is standing in**;
  - `BACKGROUND_WATER` if surfing;
  - `terrain = CalcTerrain(fieldSystem, background)`.
- `CalcTerrain` (fbdt:467-494): the tile behaviour under the player
  (`TerrainCollisionManager_GetTileBehavior`, src/terrain_collision_manager.c:229-240: the land-data
  terrain attribute u16 of that tile masked by `TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK`; both
  collision managers read the land data, :160-199). First match wins:
  1. ice → ICE;
  2. tall or very tall grass → GRASS;
  3. sand → SAND;
  4. snow (4 kinds) → SNOW;
  5. mud or mud-with-grass → GREAT_MARSH;
  6. cave floor → CAVE;
  7. surfable (flag table) → WATER;
  8. else `sTerrainForBackground[background]` (fbdt:441-465): PLAIN, WATER, CITY→BUILDING,
     FOREST→GRASS, MOUNTAIN, SNOW, INDOORS 1-3→BUILDING, CAVE 1-3→CAVE, AARON…GIRATINA.
  The predicates are in src/map_tile_behavior.c:266-274, :326-334, :431-434, :486-512, :559-562.
- Wild-only override: `FieldBattleDTO_SetWaterTerrain` (fbdt:507-510, called from
  src/overlay006/wild_encounters.c:405) is fishing/surf wild encounters and never a trainer battle.
- In battle, terrain is read only by `GetTerrainMove`, `GetTerrainSecondaryEffect` (Secret Power,
  script.c:8050-) and `TryCamouflage` (rom_diff note), plus graphics and wild ball modifiers.

### 9.2 What data per trainer fight determines it
- **The cell's header** (`battleBG`: in Kaizo, bits 7-11 of the u16 at offset 22 of the d_test.narc
  member, per rom_diff.toml:115-120), the header's `mapMatrixID` (u16 at offset 2, rom_diff.toml:35-40),
  the matrix's per-cell `mapHeaderIDs` and `landDataIDs` (src/map_matrix.c:83-128), and the land
  data terrain attributes of the tile the player stands on, plus whether the player is surfing.
- **Terrain can vary within one map.** Any map containing tall grass, sand, cave-floor, snow, mud,
  ice or surfable tiles gives different terrains depending on where the battle starts. Outdoor
  matrices also mix headers per cell, so `battleBG` can differ cell to cell.
- Where the battle starts:
  - for a trainer who walks up after spotting the player, the player's tile anywhere in the
    trainer's sight line;
  - for a talk-to trainer (gym leader), a tile adjacent to the trainer.

### 9.3 Proposal: resolve terrain per trainer from the ROM
1. **ROM reader** (tools/rom). For each trainer, use the trainer → map join that already feeds
   `start_condition` (survey §4.3). That gives the header ids whose event scripts start the battle.
   For each header, walk the matrix cells whose `mapHeaderIDs` entry equals that header. For every
   tile of each cell's land data, compute `CalcTerrain(behaviour, battleBG(header))`, with the
   surfable rule. The result is the per-trainer set S.
2. If |S| = 1, that is the terrain. If |S| > 1, refine with the trainer's event object (position,
   facing, sight range from zone_event): candidate tiles are the sight-line tiles plus the four
   adjacent tiles. If still > 1, the terrain is **mixed**.
3. Store `Trainer.terrain` (decomp `TERRAIN_*` id, or −1 for mixed/unknown) in the K/V trainer
   tables, beside `start_condition`.
4. Pass it like `start_condition` (`make_state(..., terrain)`) into a constant field scalar
   `BattleState.terrain`. `step` reads `terrain_move[terrain]` and `terrain_type[terrain]`, two
   tables built from include/data/terrain/to_move.h and to_type.h and registered in game_constants
   (TERRAIN_SPECIAL clamp included).
5. Coverage: refuse effects 173/213 (and 197 Secret Power, still unsupported) only for a trainer
   whose terrain is mixed/unknown. The message names S (e.g. "terrain varies: GRASS, PLAIN").
   Optionally the user can give the battle tile.
6. Roark: header 47, matrix 113, land data 229, battleBG 6 (INDOORS_1). The 1024 tile behaviours
   are {0x00, 0x30, 0x31, 0x65} (survey §5.3, read from the patched ROM), and none matches a
   predicate. S = {BUILDING}, so Nature Power is Tri Attack and Camouflage gives Normal. A tested
   reader must reproduce this. Vanilla header 47 is the same, so vanilla Roark resolves the same way.

### 9.4 rom_diff.toml changes
- `MapHeader_GetBattleBG` (rom_diff.toml:115-120) and `MapHeader_GetMapMatrixID` (:35-40):
  `not_reached` → `modelled`, and **drop the `when = { move_effect = [173, 197, 213] }`**.
- The note should say: the terrain reader reads battleBG (bits 7-11 of u16@22) and mapMatrixID
  (u16@2) from the d_test.narc member exactly as the patched getters do. Extend
  `check_kaizo_map_headers` to those two fields, as K2b did for `MapHeader_GetWeatherType` and
  `MapHeader_GetScriptsArchiveID`. Cite the test.
- After that, the remaining refusal is the per-trainer "terrain mixed" coverage line, not a ROM line.
- **UNCERTAIN:** whether Kaizo changed any matrix or land_data members (survey: land data 229
  unchanged). The reader must read the patched ROM's NARCs regardless, so this only matters for
  rom_diff completeness.

---

## 10. Outcome width (WIDTH_CAP = 131072, enumerate.py:88; `k > cap` raises)

### 10.1 Inputs used (computed from K species.json/natures.json and the two saves)

Speeds:
- Roark: Bonsly 12, Lileep 15, Gible 20, Corsola 19, Shuckle 11, Cranidos 28.
- Pool: Prinplup 25 (Quick Claw), Wormadam 20, Rhyhorn 17, Delibird 32, Sandshrew 19, Luvdisc 38,
  Flaaffy 23 (Static), Shinx 13, Vulpix 29 (Flame Body), Goldeen 28, Meowth 35 (Limber), Skitty 21
  (Gentle, F, Cute Charm).
- Turn-start ties: Wormadam = Gible, Sandshrew = Corsola, Goldeen = Cranidos.

Roark has no contact-proc ability, so the player's attacks never draw `contact_proc`. Roark's
targets need no accuracy roll except Gible in sand (80) or after Sand-Attack.

Factor per action:
- damaging hit: crit 2 x roll 16 = 32;
- `secondary` 2;
- Thunder Fang flinch 2;
- contact into Static, Flame Body or Cute Charm: 2;
- duration (Water Pulse, Sand Tomb, Sing): 4;
- DoubleSlap (fixed 2 hits, acc 90): hit 2 x 32 x 32 = 2048;
- Present: 4 x 32 = 128 (joint: 97);
- Nature Power→Tri Attack: tri 3 x 32 x secondary 2 = 192 (joint: 128);
- Roar: target ≤ 5 (no level draw: Vulpix is L16).

Regimes:
- **A**: today's masks extended naively. Every new element is relevant when its move is used, and
  `full_para`/`confusion_hit`/`thaw` are relevant whenever the foe can inflict the status.
- **B**: tight relevance (order-gated flinch/status reads, gender gate, `qc_roll` tight rule).
- **C**: B plus joint/conditional factors: (present, crit, roll); (tri, secondary); (secondary,
  duration); enumerate `speed_tie` first and recompute the masks per order branch.

### 10.2 Worst pairs

| player action vs Roark action | A | B | C | note |
|---|---|---|---|---|
| Skitty DoubleSlap vs Corsola Water Pulse | 1,048,576 | **524,288** | **327,680** | overflows even before K3b2 (no new field involved) |
| Skitty DoubleSlap vs Gible Thunder Fang (into Cute Charm) | 1,048,576 | **262,144** | **262,144** | para 2 x CC 2; flinch off (Skitty 21 > Gible 20) |
| Skitty DoubleSlap vs Cranidos Zen Headbutt / Crunch (Cranidos first, CC) | 262,144 | **262,144** | **131,136** (ZH: conditional on flinch) / **262,144** (Crunch) | |
| Skitty DoubleSlap vs Shuckle Sand Tomb | 262,144 | **262,144** | **262,144** | Sand Tomb's effect is certain (chance 0), so duration 4 has no joint saving |
| Skitty DoubleSlap vs Lileep Sludge Bomb / AncientPower; vs Bonsly Brick Break (CC) | 131,072 | 131,072 | 131,072 | at the cap; any extra x2 overflows |
| Wormadam Nature Power vs Gible Thunder Fang (speed tie 20/20) | 6,291,456 | **1,572,864** | **131,072** | needs per-order masks and joint factors; exactly at the cap |
| Goldeen Psybeam vs Cranidos Zen Headbutt (tie 28) | 262,144 | 131,072 | about 29,000 | per-order branches, (secondary, duration, foe's confusion roll) joint, and no Goldeen move after a flinch |
| Luvdisc Water Pulse vs Corsola Water Pulse | 262,144 | 131,072 | 51,200 | |
| Delibird Present vs Gible Thunder Fang | 32,768 | 16,384 | 12,416 | |
| Prinplup (Quick Claw) BubbleBeam vs Gible Thunder Fang | 65,536 | 16,384 | 16,384 | x2 for `quick_claw` |
| Vulpix Roar vs Gible Thunder Fang (Flame Body) | 20,480 | 5,120 | 4,000 | |

Quick Claw storage matters. Under "store whenever a living holder exists", every pair above
doubles while Prinplup is alive. Then the at-cap pairs (DoubleSlap vs Lileep/Bonsly, Nature Power vs
Gible, Goldeen/Cranidos, Luvdisc/Corsola) all overflow. The tight rule (§5b) adds x2 only while
Prinplup is active or a replacement is pending.

### 10.3 Proposed gates (all exact)
1. **Order-gated reads.** `fang_flinch`, and `full_para`/`thaw`/`confusion_hit`/`infatuation` via
   foe infliction, are relevant only if the reading side can act after the inflicting side (and
   flinch: the target can act after the user).
2. **Status gating.** The paralysis/Tri Attack secondary needs a target that can take the status
   (§1b, §8b). The Cute Charm proc needs opposite known genders and a non-infatuated attacker.
3. **Joint factors** in `enumerate_outcomes`: (present, crit, roll); (tri, secondary); (secondary,
   duration); (secondary, foe's thaw/full_para). A joint factor lists only the combinations that
   change the state.
4. **Conditional enumeration on order:** enumerate `speed_tie` (and nothing else) first, compute
   masks per branch, and concatenate. This is needed for any turn-start speed tie with
   order-dependent reads.
5. **Damage-class compression** of `(crit, roll)` per hit: merge rolls that give the same final
   damage. Sound only when no earlier action this turn can change that hit's damage inputs
   (stages, screens, types via Camouflage, Flail HP, Present power, weather, abilities). Otherwise
   refine the classes over every reachable input variant.
   - This is the only gate that brings the DoubleSlap pairs under the cap. Skitty's DoubleSlap does
     about 5-10 HP to Gible and about 2-3 to Corsola: 2-5 distinct values per hit instead of 32.
   - Estimates: about 7,168 (vs Gible Thunder Fang) and about 2,250 (vs Corsola Water Pulse).
6. **`qc_roll` tight storage rule** (§5b).
7. **Accuracy interval classes** (§0.2c) keep `hit` at 2-3 values. Do not store the raw 1..100 roll.

### 10.4 Flagged overflows

Even with gates 1-4, these exceed the cap and need gate 5 or stay "sampled":
- Skitty DoubleSlap vs Corsola Water Pulse;
- Skitty DoubleSlap vs Gible Thunder Fang;
- Skitty DoubleSlap vs Cranidos Zen Headbutt / Crunch;
- Skitty DoubleSlap vs Shuckle Sand Tomb.

Nature Power vs Gible Thunder Fang at the speed tie is exactly at the cap only with gates 1-4.

---

## 11. UNCERTAIN list (what settles each)

1. Quick Claw root belief at battle start (§5b). Settle: the Stage B design for a non-degenerate
   root, with the information set by equal observation.
   **Settled (K3b2 Task 8):** the root of a solve is a chance node over the hidden `qc_roll`
   (`step.root_belief`: each side whose `qc_kept` holds splits 1/5 vs 4/5); the states form
   the root belief node (one information set). Stage A draws it per playout
   (`playout.root_state`). Stage B design doc, "Amendment 2026-11-09".
2. `last_move` vs attack-message-shown equivalence for the Protect chain fold (§2a). Settle: a
   failed-Protect-then-Protect test, and a read of every `SYSCTL_ATTACK_MESSAGE_SHOWN` writer for
   Protect's paths.
3. Attract through a substitute (§4a). Unreachable now. Settle: grep the TryMove immunity paths
   for substitute vs DIRECT status effects before Substitute enters any pool.
4. Roar: the toxic-counter write-back for a forced-out battler, and the berry timing after a hazard
   hit on the dragged-in Pokemon (§6a). Settle: battle_controller.c emit sites, and
   subscript_update_hp.s plus bcp:3638-3760.
5. Nature Power with an already-fainted foe: the `NO_TARGET` path and the `moveHit` write (§8a).
6. Terrain: whether Kaizo changed matrix/land_data members, and whether "mixed" trainers exist on
   the user's route (§9).
7. The Fang family (273 Fire Fang, 274 Ice Fang) having the same script shape as 275
   Thunder Fang (§1a). Settle: read effect_script_0273/0274.s before mapping them.
8. Convention: `% n` treated as uniform. The exact u16 weights are given for Quick Claw (13108/65536),
   Roar (10923/10922) and Tri Attack (21846/21845/21845). Decide once project-wide.

## 12. STATE.md row changes (for the K3b2 plan)

| decomp field | today | K3b2 |
|---|---|---|
| `BattleContext.moveProtect` | later (M6) | folded into `protect_count` (reset at the user's action end unless Protect/Detect/Endure showed its message) |
| `MoveEffectsData.protectSuccessTurns` | `protect_count` | unchanged (now read by Protect) |
| `BattleMon.statusVolatile` ATTRACT bits | later (M6) | `infatuated[2]` (source = the foe's active; cleared with the source) |
| `BattleContext.speedRand` | not needed | `qc_roll[2]` (`% (100/param) == 0`, hidden, carried into the next turn) |
| `MoveEffectsData.quickClaw` | later (M6) | stays: the animation flag only (Thief/Covet not in the pool) |
| `BattleMon.type1/type2` | `types` | `types`, written for the active battler by Camouflage and restored from the party row on leave |
| `moveEffectsMask` CAMOUFLAGE | later | stays: AI-own scoring only (ais:1023) |
| `BattleSystem` terrain | (absent) | `terrain` field scalar, constant per battle |
| `TurnFlags.protecting` | not needed (per turn) | unchanged: a local inside `step` |
| `BattleContext.switchedPartySlot`, `msgMoveTemp`, `moveCur` | not needed | unchanged (Roar target and Nature Power's called move are per-action scratch) |
