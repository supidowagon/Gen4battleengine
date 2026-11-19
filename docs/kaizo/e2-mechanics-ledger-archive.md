# E2 mechanics ledger: archive (designs)

Designs behind modelled mechanics (the rulings in e2-mechanics-ledger.md are the authority),
and one parked option. Finished-batch history is in git.

## Design: a chosen move replaced mid-turn (Encore's takeover; built in batch 11, B11-1..B11-3)

Problem. FightCommand (bcp.c:1866-1873) runs an encored battler's encored move whatever it chose, so
a faster Encore this turn replaces the move the turn-start laws and sampler assumed. The same
gap already exists, unflagged, for Sleep Talk's split variants (B5-13 Disable, B7-13 Imprison):
`call_laws` / `call_variants` / `call_consistent` enumerate the alternative called move exactly,
but `sample_outcome` draws every move-dependent field from the turn-start prediction
(`used_moves`), so a split variant whose called move differs in crit stage, effect chance or
duration law is sampled at the wrong rates. One mechanism covers both.

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

## Design: mid-turn switch decision (U-turn, Baton Pass; built in batch 6, B6-n)

**Where the game decides.** U-turn's script only sets `MOVE_SIDE_EFFECT_ON_HIT | ..._ATTACK_THEN_SWITCH_OUT` (effect_script_0228.s:5); the switch runs as the hit's secondary effect, `ONE_HIT_TRIGGER_SECONDARY` (bcp.c:3484-3494). `subscript_attack_then_switch_out.s` runs, in order: the checks: the defender's side is not blacked out and the attacker has a replacement (:5-6, which also flags it `BATTLER_STATUS_SWITCHING`, battle_script.c:5876-5880); the hit's contact ability (:7) and one pivot item (:11, see the open points); Destiny Bond (:15-34), Grudge (:40); no switch if the attacker fainted (:48); the opponent's Pursuit (:54) and the user's status restore (:56-57); `SYSCTL_UTURN_ACTIVE` set (:68), then the party list (:71; `ShowParty` / `WaitMonSelection`, battle_script.c:2378-2470). After the pick: `SwitchAndUpdateMon`, the hazards and a faint check (subscript_show_party_list.s:9-24).

Baton Pass: subscript_baton_pass.s :5 (no replacement: the move fails, :19-20), status restore (:7-8), `SYSCTL_BATON_PASS` (:16), the list (:17); no Pursuit. The rest of the action then runs **on the attacker's battler slot, which now holds the newcomer**:
`AfterMoveMessage` continues with Shaymin's form check, Rage, `TriggerAbilityOnHit` (guarded) and `CheckExtraFlinch` (bcp.c:3497-3530). The extra flinch reads the newcomer's King's Rock, has no U-turn guard (:4630-4650), and U-turn carries `MOVE_FLAG_TRIGGERS_KINGS_ROCK` (res/moves/u_turn/data.json:25); `AfterMoveEffects` (:3638ff): its `TRIGGER_SWITCH_IN_EFFECTS` (:3672-3682, `BattleSystem_TriggerEffectOnSwitch`: the newcomer's Trace, weather, Intimidate, status cure, items) comes before the attacker / defender item checks (:3684ff); then `LOOP_FAINTED` (:3820). The party list itself runs only the hazards and a faint check (subscript_show_party_list.s:22-24).
`UpdateMoveBuffers` (:3917ff): no Choice lock for U-turn under the flag or a successful Baton Pass (:3927-3928). `moveProtect`, `movePrev` and `movePrevByBattler` are written onto the newcomer (:3934-3944), after `UpdateAfterSwitch` cleared them (battle_lib.c:2119, :2123); `MoveEnd` (:3962-3998): status recovery by ability, the exp payout, the battle-over check, `TriggerEffectOnSwitch` again (:3972; the announced flags stop repeats, battle_lib.c:3795), then `BattleContext_Init` (:3998), which clears both flags (system_control.h:33) and resets `aiSwitchedPartySlot` to 6 (battle_lib.c:2015; also at TurnEnd, bcp.c:1855).

Guarded by `SYSCTL_UTURN_ACTIVE`: the contact abilities in `TriggerAbilityOnHit` (battle_lib.c:4219-4362), Sticky Barb and Jaboca on hit (:5393, :5403), the Choice lock (bcp.c:3927), Shell Bell (:4739), Life Orb (:4761). Not guarded: King's Rock, the move buffers, the faint loop, `MoveEnd`.

**Opponent's pick:** `Task_TrainerShowPartyMenu` reads `aiSwitchedPartySlot` (6 here: `MoveEnd`'s init; it is set only by AI switch choices, e.g. trainer_ai.c:3340, :3460, :3478, :3519, :3537), then `BattleAI_PostKOSwitchIn`, then the first living slot (battle_display.c:4543-4556).

### 1. Pause and resume
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

### 2. Enumerate, observe, search
- **Turn-start chance node:** each `Outcome` element is read at most once, so the turn splits at the pause. The turn-start node still enumerates the draws that only the second half reads (`end_speed_tie`, `quick_claw`, `shed_skin`, later actors' rolls, King's Rock), because whether the pause happens depends on first-half draws. Paused rows that differ only there step to the same bytes and merge. This costs rows, not tree width.
- **Resumed chance node:** `outcome_laws` on a paused state uses the one branch `TurnScratch.order` from `cursor` on, and reads King's Rock from the newcomer named by the switch action.
- **Observation:** `turn_events` on the first call gives the first half's events (later positions masked; `order` and `quick_claw` come from here). On the resume it gives only the resumed part, with `ACT_SWITCH` and the slot for the pivoting side and `order` read from `TurnScratch`. Each half's `observe` uses its own events, so Stage B groups each chance node by its own observation.
- **Search:**
  - `expand._turn` (expand.py:217-218) and playout's `_turn` read `power_mul` and `ai_switched` from `TurnScratch` when `before` is paused, never from the switch-shaped actions.
  - `replace_mul` keeps its end-of-turn role.
  - Stage B `_pending` gains a `"pivot"` kind, whose answers are turn requests through `expand`. `decision_actions` offers `legal_replacements(state, 0)`. The opponent's answer is `MOVE_ACTIONS + post_ko_switch_in(paused, pick_mul)`, refused as today.
  - Playout loops `step` while `pivot_battler != 0`, with the same `Outcome`.

### 3. The three cases
- **Opponent's U-turn:** a pause with one forced opponent action; no search decision.
- **Player's U-turn:** a `"pivot"` decision made after the first half's observation.
- **Baton Pass:** under `PIVOT_BATON_PASS`, `_reset_battler` keeps:
  - the stat stages (battle_lib.c:97-101);
  - `VOLATILE_CONDITION_BATON_PASSED` (condition.h:72);
  - `MOVE_EFFECT_BATON_PASSED` (moves.h:46-48), including `MOVE_EFFECT_NO_CRITICAL`, with Lock-On's timer refreshed (battle_lib.c:2073-2085);
  - the substitute HP, Lock-On / Mean Look targets and the Perish Song / Magnet Rise / Embargo / Heal Block counters (:2107-2115).

  A passed Power Trick swaps the newcomer's Attack and Defense again (:2136-2140). There is no Pursuit (no call in the subscript). Each item the engine models today is carried; the rest come with their mechanic.

### 4. Doubles (section 5)
- One `pivot_battler`, so there is no multi-battler resume and no joint mask.
- `cursor` is a position in the stored order, not "second" (convention 5).
- `_pivot` takes the attacker battler (convention 1).
- In doubles `PostKOSwitchIn` picks its defender with `BattleSystem_RandomOpponent`, which draws `RandNext() & 1` (battle_lib.c:4183-4187, used at :7949). So the opponent's pivot pick is a chance node there: a new `Outcome` element indexed by battler.

### 5. Reuse the faint-replacement decision?
Yes for the decision: the slot encoding, `legal_replacements`, `decision_actions`, `replacement_policy`, `post_ko_switch_in`, and Stage B's pending plumbing. No for the transition: `step.replace` / `replacement_pass` run after the turn (TurnEnd payout, no draws), and a pivot resumes inside an action. New pieces: `pivot_battler`, `TurnScratch`, the halt mask, the resume entry and `_use_move`'s tail entry.

### 6. Compile time
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
  , ffcdc54 -> after the batch:** `step` 73,169 -> 80,110 jaxpr
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
