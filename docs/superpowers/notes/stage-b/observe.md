# Facts for `engine/observe.py` (Stage B information sets)

Source spec: `docs/superpowers/specs/2026-10-30-stage-b-solver-design.md` §2 (lines 91-113).
All paths below are relative to the repo root `G:/Farry/Claude Workspace/Pokemon Platinum Game Engine`.
Decomp paths are under `external/pokeplatinum/` (present, pinned clone). `bcp.c` =
`external/pokeplatinum/src/battle/battle_controller_player.c`.

Two things the spec says that constrain everything below:

- Signature in the spec header is `observe(state_before, actions, state_after) -> Observation`
  (spec:93), but its "practical form" (spec:109-113) says the observation is computed from the
  two states **and the outcome** ("missed" and "crit" come from the outcome). The workflow
  task names `observe(state_before, actions, outcome, state_after)`.
- Two children with equal observations form one information set (spec:94-96); the
  observation must therefore contain only what the player can see, and anything not in it may
  legitimately differ between states of one belief node (spec:111-113).

---

## 0. Engine API the observation function sits on

| Item | Exact signature / type | Cite |
|---|---|---|
| `step` | `step(tables: Tables, state: BattleState, actions, outcome: Outcome) -> BattleState`; `actions` is `int32[2]` (player, opponent) | `src/nuzline/engine/step.py:692-700` |
| step phases | `_record_last_move` -> `action_order` -> `execute_actions` -> `end_of_turn` -> `turn += 1`; a state already `over` is returned unchanged | `step.py:694-700` |
| `step` emits **no event record**: it returns only the final `BattleState`. No message/event log exists anywhere in the JAX engine. | | `step.py:692-700` |
| `execute_actions` | `execute_actions(tables, state, actions, outcome: Outcome, order) -> BattleState` | `step.py:529-562` |
| `end_of_turn` | `end_of_turn(tables, state, actions, outcome, order) -> BattleState`; clears `flinched` at the end | `step.py:676-689` |
| `end_of_turn_order` | `end_of_turn_order(state, outcome) -> int32[2]` | `step.py:663-673` |
| `needs_replacement` | `needs_replacement(state) -> int8[2]` | `step.py:703-710` |
| `replace` | `replace(tables, state, side, slot) -> BattleState` | `step.py:713-732` |
| `pursuit_fires` | `pursuit_fires(tables, state, side, actions, choices) -> bool` | `step.py:429-448` |
| `send_replacements` (opponent forced replacement, deterministic) | `send_replacements(tables, state, ai_switched, power_mul) -> (BattleState, bool refused)` | `src/nuzline/search/playout.py:158-176` |
| Action encoding | `MOVE_ACTIONS = 4` (0-3 move slot), switches 4-9 (`action - 4` = party slot), `NUM_ACTIONS = 10`, `FIRST_ITEM_ACTION = 10`, opponent bag actions 10-13, `NUM_OPP_ACTIONS = 14` | `src/nuzline/engine/actions.py:36-39` |
| `is_switch(action)`, `is_item(action)` | `bool` | `src/nuzline/engine/order.py:32-39` |
| `selected_move(tables, state, side, action) -> (move id int32, struggling bool)` | Struggle (`MOVE_STRUGGLE = 165`, `ids.py:185`) when no slot usable | `order.py:42-54` |
| Reference engine (plain Python, branchy, one `if` per rule) | `ref_step(tables, state: RefState, actions, outcome) -> RefState` | `src/nuzline/reference/engine.py:1009-1089` |
| Converters | `to_ref(state: BattleState) -> RefState`, `from_ref(ref) -> BattleState` | `src/nuzline/reference/convert.py:22`, `:37` |

`BattleState` is unbatched, side axis 0 = player, 1 = opponent; slot axis = party order
(`src/nuzline/engine/state.py:3-5`). `SIDES, SLOTS, MOVES = 2, 6, 4`, `BAG_SLOTS = 4`
(`state.py:46-47`). Stage range `STAGE_MIN, STAGE_MAX = -6, 6` (`state.py:48`).

### Outcome (the per-turn chance record)

`class Outcome(NamedTuple)` (`src/nuzline/engine/outcome.py:44-58`), every field `int8`
after `sample_outcome` casts (`outcome.py:193`); `MAX_HITS = 5` (`outcome.py:34`).

| Field | Shape | Meaning | Decomp roll |
|---|---|---|---|
| `speed_tie` | `[]` | 0 = player first on equal speed | `outcome.py:176` (battle_lib.c:1445) |
| `hit` | `[2,5]` | per side, per hit index; only `hit[side,0]` is read (multi-hit checks accuracy once, `step.py:329`, `effects.py:33-37`) | `outcome.py:155` |
| `crit` | `[2,5]` | per hit; Lucky Chant on the defending side forces 0 | `outcome.py:157-158` |
| `roll` | `[2,5]` | 0-15 = 85-100 %; `roll[side,0]` is also the confusion self-hit roll (`step.py:296`) | `outcome.py:179` |
| `secondary` | `[2,5]` | attacker's own secondary effect proc (chance 0 => always 1) | `outcome.py:161-162` |
| `contact_proc` | `[2,5]` | indexed by the **defending** side (its Static/Poison Point) | `outcome.py:181` |
| `n_hits` | `[2]` | 2-5 | `outcome.py:164-165` |
| `full_para` | `[2]` | | `outcome.py:183` |
| `confusion_hit` | `[2]` | | `outcome.py:184` |
| `duration` | `[2]` | by the inflicting side: sleep/confusion 2-5, Bind 3-6, Taunt 3-5 | `outcome.py:170-174` |
| `thaw` | `[2]` | | `outcome.py:186` |
| `flinch_item`, `quick_claw` | `[2]` | reserved, always 0 | `outcome.py:188-189` |
| `end_speed_tie` | `[]` | end-of-turn order tie | `outcome.py:191` |

`empty_outcome()` (`outcome.py:61-79`): hit=1, crit=0, roll=15, secondary=0, contact_proc=0,
n_hits=2, full_para=0, confusion_hit=0, duration=2, thaw=0, ties 0.

---

## 1. Every `BattleState` field: dtype, shape, player visibility

Shapes/dtypes from `LAYOUT` (`state.py:157-221`, checked against the NamedTuple order at
`state.py:222-223`). Field comments at `state.py:60-128`.

Visibility classes used below:

- **V** visible directly on screen (healthbox, party screen, own move menu).
- **D** derivable: every change of it in the implemented engine is announced by a message or
  is a deterministic function of visible history, so two siblings with equal messages have
  equal values. Safe to put in the observation (or to omit: it cannot differ inside an
  information set that agrees on the messages).
- **C** constant during the battle (set by `make_state`, never written by `step`, or only
  ever reset to 0 and never set by an implemented mechanic). Identical across every state
  reachable from one root, so irrelevant to grouping; whether the player "knows" it in game
  does not matter for information sets (the solver knows the trainer data).
- **H** hidden and **can differ** between chance siblings: must NOT enter the observation.

Healthbox facts used: the enemy healthbox never shows current HP, max HP or EXP
(`HEALTHBOX_INFO_NOT_ON_ENEMY`, `external/pokeplatinum/include/battle/healthbox.h:33`, applied
at `src/battle/healthbox.c:633-640`); the player's solo healthbox keeps current/max HP
(`healthbox.c:654-657`); status is drawn on both (`HEALTHBOX_INFO_STATUS`,
`healthbox.h:22`, `healthbox.c:703`); level and gender text on both (`healthbox.c:682-688`).

### Per side x slot `[2,6]` (and wider)

| Field | dtype / shape | Player side (0) | Opponent side (1) | Written by `step` at |
|---|---|---|---|---|
| `species` | int16 `[2,6]` | V | V once sent out (C otherwise) | never |
| `form` | int8 `[2,6]` | C | C | never (no reader in engine) |
| `level` | int8 `[2,6]` | V | V (healthbox) | never |
| `hp` | int16 `[2,6]` | **V exact** (numbers) | **H**: only `hp_pixels(hp, max_hp)` (section 3) is visible, for the active battler; bench HP unchanged during battle in M1 | `step.py:300` (confusion), `effects.py:441-444` (damage), `:231-242` (drain/recoil), `:333-340` (Struggle), `step.py:406` (Potion), `step.py:565-631` (residuals), `hooks.py:116-118` (berries), `effects.py:378-391` (Stealth Rock) |
| `max_hp` | int16 `[2,6]` | V | C (not shown, `healthbox.h:33`) | never |
| `stats` | int16 `[2,6,5]` Atk,Def,SpA,SpD,Spe | V | C | never |
| `types` | int8 `[2,6,2]` | V (species) | V (species) | never |
| `ability` | int16 `[2,6]` `ids.AbilityId` | V | C (revealed by messages for Mold Breaker / Static / Poison Point / Intimidate) | never |
| `ability_slot` | int8 `[2,6]` | V | C | never |
| `item` | int16 `[2,6]` | V | D (only change: berry consumed, announced) | `hooks.py:119` |
| `item_consumed` | int16 `[2,6]` | D | D | `hooks.py:120-122` |
| `moves` | int16 `[2,6,4]` | V | C (opp moves revealed when used) | never |
| `pp` | int8 `[2,6,4]` | V | D (one PP per announced use, `step.py:312-314`; Pursuit `step.py:461`) | `step.py:313`, `:461` |
| `status` | int8 `[2,6]` `ids.Status` | V (icon) | V (icon) | `effects.py:186-200`, `:327-330`, `:368-375`; `step.py:142` (Natural Cure), `:270`, `:280`, `:308`; faint clears it `step.py:490-495` |
| `status_counter` | int8 `[2,6]` | **H for sleep** (both sides: the sleep turns are `outcome.duration`, `effects.py:190`); D for toxic (deterministic +1 per end of turn, `step.py:607`; reset on switch-out `step.py:153-155`) | same | `effects.py:197`, `step.py:269`, `:608`, `:155` |
| `gender` | int8 `[2,6]` | V | V | never |
| `friendship` | int16 `[2,6]` | C | C | never (no reader in engine) |
| `ivs` | int8 `[2,6,6]` | C | C | never |

### Per side, active battler `[2]`

| Field | dtype / shape | Visibility | Notes / writers |
|---|---|---|---|
| `active` | int8 `[2]` | V | `_switch_in` `step.py:227` |
| `stages` | int8 `[2,8]` (index `ids.Stat`, 0 neutral, index 0 unused) | D (every change prints "rose/fell" or "won't go higher/lower"; Intimidate prints) | `effects.py:134-139` (`change_stage`), Steadfast `step.py:285`, Intimidate `hooks.py:126-136`; reset on switch/faint `step.py:158-165` |
| `confusion` | int8 `[2]` turns left | **H counter**; `confusion > 0` is D ("became confused", "is confused", "snapped out") | set `effects.py:300-307` (`outcome.duration`, 2-5), decremented `step.py:292-301` |
| `leech_seed` | int8 `[2]` | D | `effects.py:310-319` |
| `substitute_hp` | int16 `[2]` | C (0; no implemented move sets it) | only reset `step.py:88` |
| `taunt` | int8 `[2]` turns left | **H counter**; `taunt > 0` D | set `effects.py:290-297` (3-5), tick `step.py:634-637` |
| `trapped` | int8 `[2]` turns left | **H counter**; `trapped > 0` D | set `effects.py:277-287` (3-6), tick `step.py:620-631`, released `step.py:120-125` |
| `trap_move` | int16 `[2]` | D | same writers |
| `flinched` | int8 `[2]` | always 0 after `step` (cleared `step.py:689`); within a turn D via "flinched" message | `effects.py:211-223` |
| `choice_lock` | int16 `[2]` | C (0) | only reset |
| `last_move` | int16 `[2]` (`movePrevByBattler`) | D ("X used Y" message or disruption message) | `step.py:179-182`, `:328`, `:463` |
| `protect_count`, `rollout_count`, `charge_turn`, `recharge`, `pursuit_target` | int8 `[2]` | C (0; only reset, `step.py:85-102`; `pursuit_target` "reserved", `state.py:96`) | |
| `water_sport` | int8 `[2]` | D | `effects.py:322-324` |
| `fake_out_turn` | int16 `[2]` | D (function of visible switch/faint turns) | `step.py:168-176` |
| `move_hit` | int16 `[2]` (`moveHit`) | D in practice; see the caveat in section 4 (an unrun self-targeting move) | `step.py:364-368`, `:513-526` |

### Per side `[2]` (side conditions, bag)

| Field | dtype / shape | Visibility | Notes |
|---|---|---|---|
| `spikes`, `toxic_spikes` | int8 `[2]` | C (no implemented setter) | |
| `stealth_rock` | int8 `[2]` | D ("Pointed stones float...") | `effects.py:272-274` |
| `reflect`, `light_screen`, `safeguard`, `mist`, `lucky_chant`, `tailwind` | int8 `[2]` | C in M1 (read only: `damage.py:188`, `outcome.py:157`, `order.py:64`; never set; countdowns not implemented, `step.py:35-40`) | |
| `wish_timer`, `future_timer` (int8), `wish_hp`, `future_damage` (int16) | `[2]` | C | no reader or writer |
| `trainer_items` | int16 `[2,4]` | D (bag slot zeroed at command selection, `step.py:375-394`; the item use is announced; the extra-slot zeroing depends on the visible alive count) | |
| `trainer_item_count` | int8 `[2]` | C | |

### Field scalars `[]`

| Field | dtype | Visibility |
|---|---|---|
| `weather` | int8 (`ids.Weather`) | V; C in M1 (start condition only, `state.py:297`) |
| `weather_timer` | int8 (0 = permanent) | C |
| `trick_room`, `gravity`, `mud_sport` | int8 | C (read only) |
| `turn` | int16 | D (deterministic) |
| `over` | int8 | V |
| `winner` | int8 (0 none, 1 player lost, 2 player won; `state.py:19`) | V |
| `player_faints` | int8 | V |

### AI knowledge of the player

| Field | dtype / shape | Visibility |
|---|---|---|
| `ai_seen_moves` | int8 `[6,4]` (player slot x move slot) | D (deterministic from the player's own last move and whether the opponent's action was FIGHT; see section 5) |
| `ai_ability_known` | int8 `[6]` | D (set only by announced ability messages, `hooks.py:87-95`) |

### Summary: the only fields that can differ between chance siblings with identical messages

1. `hp[1, *]` within one HP-bar pixel (exact opponent HP).
2. `status_counter[s, slot]` where `status == Status.SLEEP` (both sides).
3. `confusion[s]` value (not its non-zero-ness).
4. `taunt[s]` value.
5. `trapped[s]` value.
6. (edge) `move_hit[0]` / confusion counter differences from unrun opponent choices; section 4.

Everything else is V, D or C. A practical observation is therefore: the visible event
sequence of section 2, plus at turn end: `hp[0]` exact (all slots), opponent active
`hp_pixels`, `active`, `species` of revealed mons, `status`, `stages`, `confusion>0`,
`taunt>0`, `trapped>0`, `leech_seed`, `stealth_rock`, `item` (or consumed flags), `pp[0]`,
`over`, `winner`, and the opponent's replacement slot.

---

## 2. Battle messages and how to detect each from (before, outcome, after)

**Key structural fact:** `step` returns only the end-of-turn state (`step.py:692-700`), and a
fainted active battler's volatiles/stages are wiped by `_faint_check` (`step.py:477-510`), and
`flinched` is zeroed at turn end (`step.py:689`). Several events (flinch, multi-hit count,
whether the second actor ran, intermediate HP values, which hit a berry triggered after)
therefore **cannot be read off `before`/`after` alone**; they need the intermediate state
after the first actor. Options visible in the code base:

- Re-run the phases: `action_order` (`order.py:67`), then reproduce `_use_move`'s gating
  (`step.py:251-329`) on the state between actors. The JAX helpers are module-private
  (`_use_move`, `_act`, `_faint_check`) and `execute_actions` loops both actors inside one
  `jax.lax.fori_loop` (`step.py:552-562`), so there is no public "state after actor 1".
- Instrument the plain-Python reference engine (`reference/engine.py`), which already has
  every decision as an explicit branch: `_status_disrupted` returns `GOES_AHEAD /
  DISRUPTED / HURT_ITSELF` (`reference/engine.py:89`, `:711-775`), `_fight` (`:778-831`),
  `_hit` (`:672-708`), `_pursuit` (`:834-862`), `_switch` (`:865-883`), `_lose_hp`
  (`:120-128`, records faints in `turn.fainted`), the turn loop (`:1059-1081`), `_end_of_turn`
  (`:908`). It agrees with the JAX engine under the differential tests
  (`tests/test_differential.py`). Converters: `to_ref` / `from_ref`.
- Or add an events output to the JAX step (changes the compiled program).

(The choice is the builder's; the conditions below are written in the JAX engine's variables
so either route can implement them.)

Notation: `side` acts, `foe = 1 - side`, `user = active[side]`, `move, struggling =
selected_move(tables, before, side, actions[side])` (fixed at the turn's start,
`step.py:536-538`). `order = action_order(tables, before, actions, outcome)` (identical on
`before` and on `_record_last_move(before)`, which only touches `ai_seen_moves`).

### 2.1 Turn order and actions

| Message | Condition in the engine | Detectable from |
|---|---|---|
| Order of actions | `order` (`order.py:67-85`); items and switches before moves (`:72-84`), priority only between two moves (`:75`), speed with stages/Simple/paralysis/Tailwind (`:57-64`), Trick Room (`:78`), tie by `outcome.speed_tie` (`:77`) | `before`, `actions`, `outcome` (exact). See section 4 for when it is actually seen. |
| "Go! X" / opponent "sent out X" (switch action) | `is_switch(actions[s])` and `state.over == 0` when its turn comes (`step.py:424-426`) | `actions`, `after.active` |
| "Roark used a Potion!" | `is_item(actions[1])` and live (`step.py:425`); heals `min(hp+20, max)` if `hp > 0` (`step.py:397-406`) | `actions`; HP bar change |
| "X used MOVE!" (the attack message) | the move got past the status checks: `can_act` (`step.py:306`); printed even on a miss or no-target (`step.py:322-329`, `reference/engine.py:800-803`) | needs intermediate state (see 2.2) |
| Pursuit hits the switching Pokemon first | `pursuit_fires(tables, before', first, actions, choices)` (`step.py:429-448`, `:543`) where `first = order[0]` | `before`, `actions`, `outcome` |

### 2.2 Move-time disruption messages (`_use_move`, `step.py:238-368`)

Let `ready = (over == 0) & (hp[side, user] > 0)` evaluated **when this side's turn comes**
(`step.py:251`; after the first actor's action and its faint check for the second actor).

| Message | Engine condition | Notes |
|---|---|---|
| (nothing; action never runs) | `~ready`: side fainted (or battle ended) before its turn | `_act` still runs but every effect is gated on `ready` (`step.py:251`, `:557`) |
| "X is fast asleep." | `ready & status == SLEEP & max(counter-1,0) > 0` (`step.py:266-268`) | counter hidden |
| "X woke up!" | `ready & status == SLEEP & max(counter-1,0) == 0` (`step.py:270`) | then the checks go on |
| "X is frozen solid!" | `ready & status == FREEZE & outcome.thaw[side]==0 & ~thaw_move` (`step.py:274-281`) | `thaw_move` = SECONDARY_STATUS with `move_param[move,2]==1` (`:274-276`) |
| "X thawed out!" (own roll) | `ready & status == FREEZE & outcome.thaw[side] != 0` (`step.py:278-280`) | |
| "X thawed out!" (Flame Wheel self-thaw) | `can_act & still_frozen` (`step.py:308`) | |
| "X flinched!" | `go & flinched[side] != 0` (`step.py:283`), where `go = ready & ~sleeping & ~(still_frozen & ~thaw_move)` (`:281`). `flinched[side]` was set this turn by the first actor's FLINCH hit: `effect == FLINCH & effective & proc & target alive & ~target_moved & target ability != INNER_FOCUS` (`effects.py:211-223`) | `after.flinched` is always 0 (`step.py:689`): must be recomputed from the first actor's hit |
| Steadfast "+1 Speed" | `flinched & ability == STEADFAST` (`step.py:284-285`) | shows in `stages` |
| "X can't use MOVE after the taunt!" | `go & ~flinched & taunt[side] > 0 & move_power[move] == 0` (`step.py:288`) | **reveals the chosen move** even though it did not run |
| "X is confused!" | `go' & confusion[side] > 0` and the decremented counter `> 0` (`step.py:292-294`) | `go'` = after the taunt check |
| "X snapped out of its confusion!" | `go' & confusion[side] > 0` and the decremented counter `== 0` | |
| "It hurt itself in its confusion!" | `confused & counter_after > 0 & outcome.confusion_hit[side] != 0` (`step.py:294`); damage `confusion_damage(tables, state, side, outcome.roll[side,0])` (`step.py:296`, `damage.py:305-312`) | HP visible (player exact, opponent pixels) |
| "X is paralyzed! It can't move!" | `go & ~self_hit & status == PARALYSIS & outcome.full_para[side] != 0` (`step.py:305-306`) | |
| Move proceeds | `can_act = go & ~self_hit & ~full_para` (`step.py:306`) | PP spent if `~struggling` (`:312-314`) |
| "But there was no target..." | `can_act & ~has_target`, `has_target = hp[foe, target] > 0 | defender_always(move)` (`step.py:321`) | only after the foe fainted earlier this turn |
| "X's attack missed!" | `can_act & has_target & outcome.hit[side,0] == 0 & move_accuracy[move] != 0` (`step.py:320-329`) | outcome field only meaningful in this case; `hit_threshold >= 100` never misses (`outcome.py:125-129`) |

Decomp order of the checks: sleep, freeze, flinch, Taunt, confusion, paralysis, self-thaw
(`step.py:260-308`, citing bcp.c:2417-2728; reference `_status_disrupted`,
`reference/engine.py:711-775`).

### 2.3 Per-hit messages (`apply_move_effect`, `effects.py:394-499`, looped `step.py:333-355`)

Let `connected = can_act & has_target & (hit[side,0] | never_misses)` (`step.py:329`).
Per hit index `i`: `landed_i = connected & i < hit_count & (target alive | defender_always)`
(`effects.py:425-429`); `hit_count = n_hits[side]` for `MULTI_HIT`, else 1
(`effects.py:123-127`). Damage `damage_for_roll(..., crit[side,i], roll[side,i], power_mul)`
(`effects.py:431-439`); `dealt = min(damage, target_hp)` when landed and not a self-targeting
range (`effects.py:441`).

| Message | Condition | Detection |
|---|---|---|
| "A critical hit!" | hit landed and reached `AfterMoveMessage` with `criticalMul != 1`: decomp `subscript_critical_hit.s:5-7`, called first in `AFTER_MOVE_MESSAGE_ONE_HIT` (bcp.c:3470-3476, `BattleControllerPlayer_CriticalMessage` bcp.c:4547-4554). Engine: `landed_i & damage_i > 0 & outcome.crit[side,i] != 0` (status moves have 0 damage, `damage.py:254`; Lucky Chant zeroes crit at sampling, `outcome.py:157-158`) | `outcome.crit` + landed (needs per-hit replay for multi-hit, since later hits do not happen after a KO) |
| Pursuit crit | `outcome.crit[pursuer, 0]` on the pre-switch hit (`step.py:464-474`) | |
| "It's super effective!" / "not very effective..." | decomp `subscript_move_followup_message.s:76-86`, run by `BattleControllerPlayer_FollowupMessage` (bcp.c:4567-4588): once per move, for multi-hit only on the last hit / when the target fainted (bcp.c:4571-4576) | **Deterministic given move type and target types** (`damage.effectiveness`, `damage.py:69-75`): adds no information beyond "the move hit". Struggle skips the chart (`damage.py:229`, `:243`). |
| "It doesn't affect X..." | damaging move with `damage == 0` (immune: `passes = damage > 0`, `effects.py:454`), or Thunder Wave-style `type_ok` false (`effects.py:450-454`) | deterministic given types; landed but `~effective` |
| "Hit N time(s)!" | MULTI_HIT move that connected; N = hits actually made. Decomp `BattleControllerPlayer_LoopMultiHit` bcp.c:3777-3812: N = `multiHitNumHits` normally, else hits made until the target fainted (bcp.c:3800-3804); message `subscript_hit_x_times.s:5-7`. Engine: hits stop once the target faints (`effects.py:425-429`) | `N = min(outcome.n_hits[side], first i with target HP 0 after hit i, +1)`; needs per-hit HP (berries can heal between hits, `step.py:352-353`) |
| Damage on the player | exact HP drop | `after.hp[0]` (final), intermediate drops visible too |
| Damage on the opponent | HP bar animates to `hp_pixels` after **each** hit / heal / residual | final `hp_pixels(after.hp[1,a], max_hp)`; intermediate pixel values are also seen in game (see section 3 note) |

### 2.4 Effect messages (per hit, `effects.py`)

| Message | Engine condition (all `& landed/effective` as shown) | Detect via |
|---|---|---|
| Stat "rose/fell" (move) | `_stat_self` `effects.py:148-151`, `_stat_foe` `:154-156`, `_secondary_stat_self` `:159-162`, `_secondary_stat_foe` `:165-168`; Keen Eye blocks ACC drops (`:142-145`); clamp +-6 (`:134-139`) | `stages` before vs after (if the battler is still active; a faint resets them, `step.py:505-509`) |
| "won't go any higher/lower" | the same conditions with the stage already at +-6 (`change_stage` clamps silently, `effects.py:136-138`) | deterministic from `before.stages` + the effect firing; not stored |
| "X was poisoned/burned/paralyzed/fell asleep/was frozen" | `_status_foe` (`effects.py:203-204`), `_secondary_status` (`:207-208`) via `_inflict` (`:186-200`, `can_have_status` `:171-183`) | `status` 0 -> nonzero on a still-living Pokemon (a status inflicted and then fainted in the same turn is wiped, `step.py:490-495`) |
| Static / Poison Point "X's Static paralyzed Y!" | `_contact_ability` `effects.py:343-365`: contact move, effective, `dealt > 0`, `outcome.contact_proc[target_side,i]`, attacker alive and can take status; reveals ability to AI | attacker's `status` change; `ai_ability_known` change |
| "X flinched" (set) | `_flinch` `effects.py:211-223` | see 2.2; set silently, the message appears at the flinched side's move |
| Drain "had its energy drained" | `_drain` `effects.py:231-235` | HP |
| Recoil "is hit with recoil" | `_recoil` `effects.py:238-242` (Rock Head blocks) ; Struggle `_struggle_recoil` `:333-340` | HP |
| "Pointed stones float..." | `_stealth_rock` `effects.py:272-274` (fails if already set) | `stealth_rock[foe]` 0 -> 1 |
| "X was squeezed by Bind" | `_trap` `effects.py:277-287` | `trapped` 0 -> >0 (value H) |
| "X fell for the taunt!" | `_taunt` `effects.py:290-297` | `taunt` 0 -> >0 (value H) |
| "X became confused!" | `_confuse_foe` `effects.py:300-307` | `confusion` 0 -> >0 (value H) |
| "X was seeded!" / "evaded the attack" (Grass) | `_leech_seed` `effects.py:310-319` | `leech_seed` |
| Water Sport | `_water_sport` `effects.py:322-324` | `water_sport` |
| "X thawed out!" (Fire hit on frozen target) | `_thaw_target` `effects.py:368-375` | target `status` FREEZE -> NONE |
| Berry "X's Oran/Sitrus Berry restored health" | `after_damage_items` `hooks.py:98-123`: `hp>0 & hp*2 <= max_hp`, after every hit (both sides, `step.py:352-353`), on switch-in (`hooks.py:149-150`), at end of turn twice (`step.py:644`, `:659`) | `item[s,slot]` nonzero -> 0 with `item_consumed` set (`hooks.py:119-122`) |

### 2.5 Switch-in messages (`hooks.on_switch_in`, `hooks.py:139-150`; called by `_send_out` `step.py:231-235`, `replace` `step.py:731-732`, `make_state` `state.py:299-300`)

| Message | Condition | Detect |
|---|---|---|
| "Pointed stones dug into X!" | `stealth_rock[side] != 0 & hp > 0`, damage `max(maxHP*eff//800, 1)` (`effects.py:378-391`) | HP |
| "X's Intimidate cuts Y's Attack!" | `hooks.py:126-136` | foe `stages[ATK]` |
| "X breaks the mold!" | Mold Breaker newcomer alive (`hooks.py:80-84`, `:148`) | species/ability (C); AI `ai_ability_known` |
| Natural Cure (silent in Gen 4 switch-out) | `step.py:135-143` | status cleared |
| Berry on switch-in | `hooks.py:149-150` | `item` |

### 2.6 End-of-turn messages (`end_of_turn`, `step.py:676-689`; order `END_OF_TURN_ORDER` `step.py:642-660`, battler order `end_of_turn_order` `step.py:663-673`, tie by `outcome.end_speed_tie`)

| Message | Condition | Cite |
|---|---|---|
| Held-item check (berry) | as above | `step.py:644`, `:659` |
| "X's health is sapped by Leech Seed!" | seeded, alive, recipient alive; `max(maxHP//8,1)` capped | `step.py:570-580` |
| "X is hurt by poison!" | POISON, alive, `max(maxHP//8,1)` | `step.py:596-599` |
| Toxic | counter +1 (max 15), `max(maxHP//16,1)*counter` | `step.py:602-610` |
| "X is hurt by its burn!" | `step.py:613-617` | |
| "X is hurt by Bind!" / "X was freed from Bind!" | counter -1; damage while `> 0`, freed at 0 | `step.py:620-631` |
| "X's taunt wore off!" | counter -1 to 0 | `step.py:634-637` |
| (not implemented) weather chip, screen/Tailwind/Trick Room countdowns | | `step.py:35-40` |

Each check is followed by a faint check; nothing runs once `over` (`step.py:684-688`).
Detection: residual damage is per-battler, deterministic given the status flags at that
point; the visible effect is an HP change (player exact, opponent pixels) and the Bind/Taunt
end events (`trapped`/`taunt` going from >0 to 0 while the same battler stayed in).

### 2.7 Faints, end of battle, replacement

| Message | Condition | Detect |
|---|---|---|
| "X fainted!" | `before.hp > 0 & after.hp == 0` per `_faint_check` (`step.py:489`) | `before.hp`, `after.hp` (exact for faint detection on both sides: 0 HP <=> 0 pixels, section 3) |
| Battle won/lost | `over`, `winner` (`step.py:496-504`); any player faint is terminal (`:498-499`) | `after` |
| Opponent replacement | `needs_replacement(after)[1]` then `send_replacements` / `post_ko_switch_in` (`search/playout.py:158-176`) | the replacement species is seen; it is deterministic given the state |

---

## 3. HP-bar pixel formula (decomp; present at `external/pokeplatinum`)

**Battle healthbox gauge**: `HEALTHBOX_HP_CELL_COUNT 6` (`src/battle/healthbox.c:61`), 8
pixels per cell, so 48 pixels (`FillCells` `offset = 8 * cellNum`, `healthbox.c:1571`;
`App_BarColor(pixels, 8 * HEALTHBOX_HP_CELL_COUNT)`, `healthbox.c:1422`).

Drawing path: `HealthBox_DrawGauge` (`healthbox.c:1373-1404`) calls
`UpdateGauge(maxHP, curHP, damage, &hpTemp, 6, 1)` (`:1379`) then `DrawGauge` (`:1406`),
which calls `FillCells(maxHP, curHP, damage, &hpTemp, cells, 6)` (`:1420`).
`FillCells` (`healthbox.c:1559-1599`):

```c
offset = 8 * cellNum;                                  // 48   (:1571)
if (max < offset) pixels = (*temp * offset / max) >> 8; // temp in 1/256 HP (:1577-1578)
else              pixels =  *temp * offset / max;       // (:1579-1580)
if (pixels == 0 && updated > 0) { cells[0] = 1; final = 1; }  // (:1585-1587)
```

`UpdateGauge` (`healthbox.c:1482-1557`) animates `*temp` toward `updated = clamp(cur - diff,
0, max)` and ends with `*temp == updated` (or `updated << 8` when `max < 48`,
`:1488-1494`, `:1515-1535`). At rest (and at the battle-start draw, `HealthBox_DrawInfo`
-> `HealthBox_CalcHP(healthbox, 0)` -> `HealthBox_DrawGauge`, `healthbox.c:669-671`,
`:863-885`) the drawn value is therefore, for both branches:

```
hp_pixels(hp, max_hp) = 0                         if hp == 0
                      = max(1, (hp * 48) // max_hp) otherwise      (integer division)
```

This equals the shared helper `App_PixelCount(cur, max, 48)`
(`src/unk_0208C098.c:41-49`: `u8 pixels = cur * maxPixels / max; if (pixels == 0 && cur > 0)
pixels = 1;`, declared `include/unk_0208C098.h:37`), which the party screens use with 48
(`src/battle_sub_menus/battle_party_text.c:1238`, `HEALTH_BAR_MAX_PIXELS 48` at
`include/battle_sub_menus/battle_party.h:256`). The `max < 48` branch is the same integer
(nested floor: `((hp*256*48)//max)>>8 == (hp*48)//max`); checked numerically for every
`1 <= max_hp <= 999`, `0 <= hp <= max_hp`: 0 mismatches.

Consequences: `hp_pixels == 48` iff `hp == max_hp`; `hp_pixels == 0` iff `hp == 0`; any
`hp > 0` shows at least 1 pixel. Bar colour is a function of the pixel count only
(`App_BarColor`, `src/unk_0208C098.c:51-68`: green if `pixels > 24`, yellow if `> 9.6`
i.e. `pixels*256 > 48*256/5`, red if `> 0`), so it adds no information.

Opponent's HP numbers are never drawn (`healthbox.h:33`, `healthbox.c:636-637`).

Side channels the spec excludes (noted so the plan can state it): (a) the bar animates per
frame by 1 HP when `max_hp >= 48` (`UpdateGauge` `fillOffset = 1`, `healthbox.c:1379`,
`:1538-1550`), so animation length reveals exact damage to a frame-counting observer; (b)
intermediate pixel values after each hit/heal/residual are seen in game, while an
observation built only from the final state uses only the final pixel. Using less than the
player sees is conservative (no peeking), only coarser. (c) `HealthBar_Color` returns a
separate `BARCOLOR_MAX` for exactly full HP (`src/unk_0208C098.c:70-77`), used by the cry
modifier (`src/battle/battle_lib.c:6214-6219`: pinch cry when not MAX/GREEN or statused) and
the doubles target-select icons (`src/battle/battle_subscreen.c:2128`); in singles the cry
distinction reduces to `pixels > 24`, again a function of pixels.

---

## 4. Turn order exposure and unrun opponent actions

`action_order(tables, state, actions, outcome) -> int8[2]` (`order.py:67-85`): the two sides
in execution order. Rules: switch/item before move (`:72`, `:81-83`); two moves compare
priority (`:75`) then speed (`effective_speed`, `:57-64`; Trick Room `:78`); equal speed and
`outcome.speed_tie != 0` puts the player second (`:77`); an item and a switch go in battler
order (`:81-83`). `step` computes it once on `_record_last_move(state)` (`step.py:695-696`),
which equals `action_order(tables, state_before, ...)` (record only writes `ai_seen_moves`).
End-of-turn order is separate: `end_of_turn_order(state, outcome)` (`step.py:663-673`),
speed only, `outcome.end_speed_tie`; seen through the order of residual messages.

The order is **observable only through which side's message appears first**. Each actor
produces at least one message when its turn comes and it is `ready` (attack message, or a
disruption message from 2.2). Cases where the opponent's chosen action is not seen:

| Case | Engine condition | Is the choice seen? | Do siblings with different unrun choices end in identical states? |
|---|---|---|---|
| Opponent fainted before its turn | opponent second, `hp[1, active] == 0` at its turn: `ready` false (`step.py:251`); `ran` false (`step.py:557`) | No (order is moot) | **Yes**: `last_move` write is gated by `ready` (`step.py:328`), PP needs `can_act` (`:312`), `move_hit` record needs `ready` (`:364`). `_record_last_move` ran at turn start for **any** FIGHT choice (`step.py:185-201`), same for all move choices (switch/item choices always run first). |
| Battle ended before its turn | `over != 0` | No | terminal |
| Asleep / frozen / flinched / fully paralyzed | `ready` true, `can_act` false | No move shown, but the disruption message shows **when** (so order is seen) | `last_move := 0` regardless (`step.py:328`); PP not spent. `move_hit[foe]` is set to 0 only when `foe_defender(move)` (move range not self-targeting, `step.py:255-258`, `:364-365`), else left unchanged: **states can differ by the unrun move's range**. In practice the difference is erased when the player then acts (`_move_end` clears the actor's own `move_hit` when `ran`, `step.py:524-525`), and if the player does not act it fainted (terminal). Worth a unit test rather than an assumption. |
| Confusion self-hit | `self_hit` (`step.py:294`) | No (only "is confused / hurt itself") | identical (`record` needs `~self_hit`, `step.py:364`) |
| Taunt-blocked power-0 move | `step.py:288` | **Yes**: "can't use MOVE after the taunt" names the move | the confusion counter is not decremented on this path (`confused` needs `go`, `:292`), visible case anyway |
| Opponent's priority leaks via order | only when both sides produce messages | when the opponent did not act, observed sequence is just the player's action | n/a |
| Speed tie | `outcome.speed_tie` matters only when `effective_speed` equal (`order.py:77`) | the resulting order is seen whenever both act | |

Practical detection of "opponent ran" from (before, actions, outcome, after) alone is not
possible in general (needs the HP after the first actor); see section 2 intro.

Opponent switches and item uses always run when the battle is live: they go before every
move (`order.py:72`, `:81-83`), and `_act` only gates on `state.over == 0`
(`step.py:424-426`). Exception: the player's switch into Stealth Rock faints the player
first -> terminal. The bag slot is spent at command selection even if the action never runs
(`_consume_trainer_items`, `step.py:375-394`, applied first in `execute_actions`, `:542`).

The opponent's choice is itself a chance layer: `ai_action_distribution(tables, state) ->
float32[NUM_OPP_ACTIONS]` (`src/nuzline/ai/policy.py:654-660`), exact over every
`AiOutcome` (`src/nuzline/ai/outcome.py:135-140`; enumerated form
`enumerate_ai_outcomes(tables, state) -> (AiOutcome[K], float32[K])`, K = 3840,
`outcome.py:253-270`). The AI reads the full state (e.g. exact HP), so in a belief node the
distribution differs per member state.

---

## 5. Existing knowledge-update code (what the AI sees)

The engine already models the **AI's** knowledge of the player (not the player's knowledge
of the opponent). Fields: `ai_seen_moves int8[6,4]`, `ai_ability_known int8[6]`
(`state.py:126-128`); decomp `aiContext.battlerMoves` / `battlerAbilities`
(`src/nuzline/engine/STATE.md:84-88`).

| Code | What it does | Cite |
|---|---|---|
| `_record_last_move(tables, state, actions)` | At turn start, if the opponent's action is FIGHT (`actions[1] < 4`) and not Struggle, marks the player's active slot whose move equals `last_move[0]` as seen | `step.py:185-201` (trainer_ai.c:2705-2717) |
| `_forget(state, side, slot)` | On a player switch-in, clears that slot's seen moves and known ability | `step.py:204-216` (battle_lib.c:2151-2152); called from `_switch_in` `step.py:225` |
| `_set_last_move` / `last_move` | `movePrevByBattler`: move on success (miss counts), 0 when disrupted | `step.py:179-182`, `:322-328` |
| `REVEALS_ON`, `reveal_ability(state, side, slot, phase, cond)` | Abilities that print a tagged message: Mold Breaker on switch-in, Static / Poison Point on contact status; Intimidate, Keen Eye, Inner Focus etc. do not reveal | `hooks.py:63-95` |
| `commands.new_context` | AI's seen moves = stored `ai_seen_moves` OR the pending `last_move[0]` | `src/nuzline/ai/commands.py:281-293` |
| `commands.defender_ability` | AI's view of the player's ability: true when known (or trapping ability), else `AiOutcome.ability_guess` slot | `src/nuzline/ai/commands.py:264-279` |
| `policy.resolve_player_ability` | wrapper used by `ai_scores` | `src/nuzline/ai/policy.py:470-479` |
| Knowledge docstring | "the scripts read the player's moves only through the AI's seen list" | `src/nuzline/ai/policy.py:27-31` |
| `move_hit` (AI switching input) | read for the opponent's side only: `state.move_hit[1]` | `src/nuzline/ai/switching.py:289-291` |
| Tests | `tests/ai/test_knowledge.py:58-258` (record timing, flinch/para record nothing, misses seen, Struggle not recorded, Mold Breaker / Poison Point reveal, Intimidate does not); `tests/engine/test_ai_inputs.py:111-208` (`fake_out_turn`, `move_hit`) | |

Nothing models the **player's** knowledge: there is no `observe.py`, no HP-set tracker, no
pixel function anywhere under `src/` (grep for `observ|pixel|hp_set` finds only
`docs/design.md:27`, `:147`, `:157`, `:189`, which describe the planned HP tracker as
`engine/observe.py`, `docs/design.md:147`). The `ai_*` fields are all D for the player
(deterministic from the player's own actions and visible opponent action kinds), so they
never split or need to appear in an observation.

---

## 6. Points the plan should decide explicitly

1. Event extraction route (re-run JAX phases, instrument `reference/engine.py`, or add an
   events output to `step`), since `step` returns no events and `after` loses flinch,
   per-hit, intermediate-HP and "did the second actor run" information.
2. Whether the observation uses only final opponent pixels or also per-hit intermediate
   pixels (finer, still non-peeking; final-only is coarser and safe).
3. The observation should be taken on the state **after** the opponent's forced replacement
   (`send_replacements`), since the replacement and its Stealth Rock damage are seen.
4. Unit test for the unrun-self-targeting-move `move_hit[0]` difference (section 4) so
   bit-identical merging assumptions are checked, not assumed.
5. Hidden-counter list (section 1 summary) is exactly what may differ within one information
   set; `hp[1]` within a pixel is the HP-set tracking of `docs/design.md:147`.
