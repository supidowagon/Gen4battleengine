# Stage B facts: `Outcome`, `sample_outcome`, where `step` reads each field, damage tables, AI API, Roark widths

Repo: `<repo>` at commit 6e817e1 (clean main).
All paths are relative to the repo root. "outcome.py" = `src/nuzline/engine/outcome.py`,
"step.py" = `src/nuzline/engine/step.py`, "effects.py" = `src/nuzline/engine/effects.py`,
"damage.py" = `src/nuzline/engine/damage.py`, "order.py" = `src/nuzline/engine/order.py`.

Verified by running code (scratch scripts in
`<scratchpad>/`:
`dump_roark.py`, `check_rolls.py`, `widths.py`).

---

## 0. Module-level facts

- `MAX_HITS = 5` (outcome.py:34). `_I8, _I32 = jnp.int8, jnp.int32` (outcome.py:35).
- Accuracy table `_ACC_NUM`/`_ACC_DEN` (outcome.py:38-39), indexed by clamped stage sum 0..12:
  num `[33,36,43,50,60,75,1,133,166,2,233,133,3]`, den `[100,100,100,100,100,100,1,100,100,1,100,50,1]`.
- Crit odds `_CRIT_ODDS = [16, 8, 4, 3, 2]` (1 in N at crit stage 0..4) (outcome.py:41).
- Module docstring (outcome.py:11-18): hit and crit odds are read from the **turn-start state**,
  with one correction: a target that switches out this turn is replaced by the incoming Pokemon
  (neutral evasion stage, its own ability). Stage changes made earlier in the same turn are not
  seen (no M1 move changes accuracy/evasion). Rolls whose effect depends on mid-turn state are
  **drawn unconditionally and gated by `step`**.
- `hooks.py` reads no `Outcome` field (grep: no `outcome` reference in `src/nuzline/engine/hooks.py`).
  The only readers of `Outcome` fields in `src/` are order.py, step.py, effects.py (and the
  plain-Python `src/nuzline/reference/engine.py`).

## 1. The `Outcome` NamedTuple (outcome.py:44-58)

Every field is `int8` after `sample_outcome` (cast at outcome.py:193) and in `empty_outcome`
(outcome.py:63-79). Shapes verified at runtime (`check_rolls.py`):

| field | shape | dtype | indexed by | `empty_outcome` value (outcome.py:64-78) |
|---|---|---|---|---|
| `speed_tie` | `()` | int8 | - | 0 (player first on tie) |
| `hit` | `(2,5)` | int8 | attacking side, hit index | 1 (all hit) |
| `crit` | `(2,5)` | int8 | attacking side, hit index | 0 |
| `roll` | `(2,5)` | int8 | attacking side, hit index; 0..15 means 85..100 % | 15 |
| `secondary` | `(2,5)` | int8 | attacking side (its own move's effect), hit index | 0 |
| `contact_proc` | `(2,5)` | int8 | **defending** side (its contact ability), hit index | 0 |
| `n_hits` | `(2,)` | int8 | attacking side, 2..5 | 2 |
| `full_para` | `(2,)` | int8 | acting side | 0 |
| `confusion_hit` | `(2,)` | int8 | acting side | 0 |
| `duration` | `(2,)` | int8 | **acting (inflicting)** side | 2 |
| `thaw` | `(2,)` | int8 | acting (frozen) side | 0 |
| `flinch_item` | `(2,)` | int8 | reserved | 0 |
| `quick_claw` | `(2,)` | int8 | reserved | 0 |
| `end_speed_tie` | `()` | int8 | - | 0 (player first on tie) |

Caveats of the empty values: `duration=2` is outside the support for TAUNT (3-5) and TRAP (3-6);
`secondary=0` differs from what `sample_outcome` gives for chance-0 moves (always 1). Both are
harmless only because `step` does not read those fields in those cases (see §3).

Helpers:
- `empty_outcome() -> Outcome` (outcome.py:61).
- `outcome_to_py(outcome) -> Outcome` of Python lists (outcome.py:82-85), for `nuzline.reference`.
- `_target_mon(state, actions, side) -> (target_side, switching: bool, target_slot)` (outcome.py:88-97):
  the slot `side`'s move will hit: `actions[target]-MOVE_ACTIONS` if the target switches, else
  `state.active[target]`. Private but reusable for relevance checks.
- `hit_threshold(tables, state, actions) -> int32[2]` (outcome.py:125-129).
- `crit_odds(tables, state, actions) -> int32[2]` (outcome.py:132-140).
- `sample_outcome(key, tables, state, actions) -> Outcome` (outcome.py:143-193); `actions` cast to
  `int32[2]` (outcome.py:145); 14 keys `jax.random.split(key, 14)` (outcome.py:146); `draw(i, shape, n)`
  = `jax.random.randint(k[i], shape, 0, n, int32)` i.e. uniform `0..n-1` (outcome.py:149-151).
- `moves` used by sample_outcome: `selected_move(tables, state, side, actions[side])[0]` for both
  sides (outcome.py:153). `selected_move` (order.py:42-54) gives Struggle (`MOVE_STRUGGLE`) when no
  slot is usable; for switch/item actions the "move" is the slot-clipped move and is meaningless
  (callers mask it).

## 2. Probability law of every field (exactly as `sample_outcome` draws it)

All draws are independent across fields, sides and hit indices (separate keys / independent
elements). Probabilities below are exact rationals.

### speed_tie (outcome.py:176)
`draw(5, (), 2)`: uniform {0, 1}, P = 1/2 each. Unconditional.

### hit (outcome.py:154-155, threshold outcome.py:100-129)
`hit[s,i] = (U{0..99} + 1 <= threshold[s])`, same threshold for all 5 hit indices of a side.
P(hit=1) = `clip(threshold, 0, 100) / 100` (playout.py:120 uses exactly this formula).
Threshold for side `s` (`_hit_threshold`, outcome.py:100-122):
- `move` = selected move; `accuracy = tables.move_accuracy[move]` (int8 table, cast int32).
- target = `_target_mon` (switching target -> incoming slot).
- `a_ability = state.ability[s, active[s]]`; `d_ability = state.ability[target, target_mon]`, set to
  `NONE` if attacker has `MOLD_BREAKER` (outcome.py:104-106).
- `acc = stages[s, ACC]`; `eva = -stages[target, EVA]` (0 if the target switches) (outcome.py:109-110).
- Simple doubles: attacker Simple doubles `acc`, defender Simple doubles `eva` (outcome.py:111-112);
  defender Unaware zeroes `acc`, attacker Unaware zeroes `eva` (outcome.py:113-114).
- `total = clip(6 + eva + acc, 0, 12)`; `rate = accuracy * _ACC_NUM[total] // _ACC_DEN[total]` (outcome.py:115-116).
- Fog: `rate*6//10`; gravity: `rate*10//6` (outcome.py:117-118).
- Returns 100 when the side's action is a switch or item, or `accuracy == 0` (never-miss) (outcome.py:121-122).
- Read from the **turn-start** state (only correction: switching target).

### crit (outcome.py:156-158, odds outcome.py:132-140)
`crit[s,i] = (U{0..odds[s]-1} == 0) & ~no_crit[s]`; `no_crit[s] = state.lucky_chant[1-s] != 0`
(Lucky Chant on the defending side, `lucky_chant[::-1]`, outcome.py:157).
`odds[s] = _CRIT_ODDS[min(stage,4)]` with `stage = (move_effect[move] == HIGH_CRIT)` (0 or 1), so
P(crit) = 1/16, or 1/8 for `HIGH_CRIT` moves (Razor Leaf), or 0 under Lucky Chant. Focus Energy,
Super Luck, Scope Lens etc. are not implemented (docstring outcome.py:133-136). Same odds for
all 5 hit indices; each index independent.

### roll (outcome.py:179)
`draw(6, (2,5), 16)`: uniform {0..15}, P = 1/16 each, independent per (side, hit index).
Roll r multiplies damage by `85 + r` percent (damage.py:9-11, 230).

### secondary (outcome.py:159-162)
`chance = tables.move_effect_chance[move]` (int8). `secondary[s,i] = (U{0..99} < chance) | (chance == 0)`.
P(1) = chance/100 for chance > 0; **always 1 when chance == 0** (but step ignores the field then,
effects.py:468 `proc=(chance == 0) | outcome.secondary[...]`).

### contact_proc (outcome.py:181)
`draw(7, (2,5), 10) < 3`: P(1) = 3/10 per element, unconditional (defender's ability not consulted).

### n_hits (outcome.py:163-165)
`first, second = U{0..3}, U{0..3}` per side; `n_hits = first+2 if first < 2 else second+2`.
Law: P(2) = 3/8, P(3) = 3/8, P(4) = 1/8, P(5) = 1/8. Drawn for both sides whatever the move.

### duration (outcome.py:166-174)
By the acting side's selected move's `move_effect`:
- `TAUNT`: `U{0..2} + 3` -> {3,4,5}, 1/3 each (Random 2,3).
- `TRAP`: `U{0..3} + 3` -> {3,4,5,6}, 1/4 each (Random 3,3).
- anything else (sleep, `CONFUSE_FOE`): `U{0..3} + 2` -> {2,3,4,5}, 1/4 each (Random 3,2).
Note: law keyed on the move effect, not on what actually gets inflicted.

### full_para (outcome.py:183)
`draw(8, (2,), 4) == 0`: P(1) = 1/4. Unconditional (status not consulted).

### confusion_hit (outcome.py:184)
`draw(9, (2,), 2) == 1`: P(1) = 1/2. Unconditional.

### thaw (outcome.py:186)
`draw(11, (2,), 5) == 0`: P(1) = 1/5. Unconditional.

### flinch_item, quick_claw (outcome.py:187-189)
Always `zeros((2,), int32)` -> int8 0. Reserved; **no reader anywhere in step/effects/order**
(grep over `src/`). Always fix to 0; never enumerate.

### end_speed_tie (outcome.py:190-191)
`draw(12, (), 2)`: uniform {0,1}, 1/2 each. Unconditional.

### Draws made unconditionally and gated later (all of them)
Every field is drawn regardless of whether it can matter: hit/crit/roll/secondary/contact_proc for
all 5 indices and both sides (even switch/item actions), n_hits for non-multi-hit moves, full_para
without checking paralysis, confusion_hit without confusion, thaw without freeze, duration for any
move, both speed ties without checking speeds. Only `hit` (threshold 100 for switch/item/acc 0),
`crit` (Lucky Chant / HIGH_CRIT), `secondary` (chance) and `duration` (effect) have state- or
move-dependent laws; all others are fixed laws.

## 3. Where `step` reads each field (relevance conditions)

Call graph: `step` (step.py:692-700) -> `_record_last_move` (no outcome) -> `action_order`
(reads `speed_tie`) -> `execute_actions` (step.py:529-562; Pursuit pre-hit, then `_act` per actor ->
`_use_move` (step.py:238-368) -> `apply_move_effect` per hit index 0..4 in a `fori_loop`
(step.py:333-355; effects.py:394-499)) -> `end_of_turn` (step.py:676-689; reads `end_speed_tie`).
A state with `over != 0` is returned unchanged (step.py:700): then **no field is relevant**.

Important: `_act` computes the switch, item and move branches all and selects with `_where`
(step.py:419-426), so a field "read" inside an unselected branch has no effect.

### speed_tie — order.py:67-85
`tie = (speed0 == speed1) & (outcome.speed_tie != 0)` (order.py:77). Used only through
`player_second = (prio0 < prio1) | ((prio0 == prio1) & (slower | tie))` (order.py:79), and that
only when `same_kind` (both moves, or both switches; order.py:82-83). Priority counts only when both
choose moves (order.py:75). Speeds = `effective_speed` (order.py:57-64: stats[SPE-1] with stage
(Simple doubled), paralysis //4, Tailwind x2) on the turn-start state (`s` after
`_record_last_move`, which does not touch speed).
**Relevant iff** same kind of action (both moves with equal priority, or both switches) AND
`effective_speed(s,0) == effective_speed(s,1)`. Then it decides who goes first (player first on 0).

### hit — step.py:329 only (index `[side, 0]`)
`connected = can_act & has_target & ((outcome.hit[side, 0] != 0) | never_misses)` (step.py:329),
`never_misses = move_accuracy[move] == 0` (step.py:320). `hit[side, 1..4]` is **never read**
(multi-hit checks accuracy on the first hit only, effects.py:33-37). The Pursuit pre-switch hit
uses `connected=True` (step.py:471): no accuracy read.
**Relevant iff** side uses a move (not switch/item), the move has accuracy != 0, threshold < 100,
and the side can act (alive, not asleep/frozen/flinched/taunt-failed/confusion-self-hit/full-para)
and has a target (`has_target`, step.py:321: foe alive or `defender_always`). If it is not read,
fix to 1. Enumerate {1: t/100, 0: 1 - t/100}. A miss makes crit/roll/secondary/contact_proc of all
that side's hits irrelevant (`landed` false, effects.py:425-429).

### crit — effects.py:436 (via `damage_for_roll`), index `[side, hit_index]`
Read only as input to `damage_for_roll` (effects.py:431-439); `dealt` is masked by `landed`
(effects.py:441). Status moves / power 0 return damage 0 whatever the crit (damage.py:254), so crit
is irrelevant for them. Pursuit pre-hit reads `crit[pursuer, 0]` (step.py:464-474 with
`hit_index=0`). crit has no other effect (no crit-triggered ability like Anger Point is modelled).
**Relevant iff** the hit index lands (`hit_index < hit_count` and target alive at that hit,
effects.py:425-429) with a damaging move. Note: crit affects only the damage value, BUT a crit
is observable (crit message) so the observation function will want it even when the damage is
equal to a non-crit roll (see §5).

### roll — effects.py:437 and **step.py:296**
1. Move damage: `outcome.roll[side, hit_index]` into `damage_for_roll` (effects.py:431-439). Same
   relevance as crit.
2. **Confusion self-hit damage uses `outcome.roll[side, 0]`** (step.py:296:
   `hurt = min(confusion_damage(tables, state, side, outcome.roll[side, 0]), user_hp)`), applied
   only when `self_hit` (step.py:297). The self-hit and the move's hit 0 are mutually exclusive in
   one turn (self_hit -> `go` false -> `can_act` false -> `connected` false, step.py:302-306, 329),
   so for a given outcome roll[side,0] feeds exactly one of the two damage formulas.
3. Pursuit pre-hit: `roll[pursuer, 0]` with `power_mul=20` (step.py:464-474).
No other reader: the AI's simulated roll is a separate field `AiOutcome.sim_roll` (fixed at 15,
ai/outcome.py:231, 264; read in ai/commands.py:351, 458 through `ai_damage_for_roll`).
`playout.py:108` calls `damage_table` for the heuristic policy, not on an outcome.
**Conclusion: roll matters in step only through the damage it produces (move damage or confusion
self-hit damage).** Verified: Monferno Ember vs Geodude turn 0, rolls with equal damage
`{11:[0..3], 12:[4..14], 13:[15]}` give bit-identical `step` states (`check_rolls.py`).

### secondary — effects.py:468, index `[side, hit_index]`
`proc = (chance == 0) | (outcome.secondary[side, hit_index] != 0)`. `proc` is read by
`_secondary_stat_self` (:159-162), `_secondary_stat_foe` (:165-168), `_secondary_status`
(:207-208), `_flinch` (:211-223), `_trap` (:277-287). Not read by STAT_SELF/STAT_FOE/STATUS_FOE/
DRAIN/RECOIL/STEALTH_ROCK/TAUNT/CONFUSE_FOE/LEECH_SEED/WATER_SPORT/contact/thaw.
**Relevant iff** chance != 0 AND effect in {SECONDARY_STAT_SELF, SECONDARY_STAT_FOE,
SECONDARY_STATUS, FLINCH, TRAP} AND the hit is `effective` (landed & damage > 0, effects.py:454,
467) AND the effect-specific gate:
- SECONDARY_STAT_SELF: user alive (effects.py:160) — `certain` is false when chance != 0.
- SECONDARY_STAT_FOE: target alive after the hit (effects.py:166); Keen Eye blocks ACC drops (:144).
- SECONDARY_STATUS: target alive and `can_have_status` (no status yet; Fire immune to burn,
  Poison/Steel to poison/toxic, Ice or sun to freeze) (effects.py:186-189, 171-183). Use the actual
  target (incoming slot when switching).
- FLINCH: target alive, `~target_moved` (attacker is the **first** actor: `target_moved=second`,
  step.py:555-556), target ability (Mold Breaker-adjusted) != INNER_FOCUS (effects.py:211-219). Also
  only useful if the target then uses a move (the flinch check is in `_use_move`, step.py:283;
  `flinched` is cleared at end of turn, step.py:689). A switching/item-using target has already
  acted (switches/items go first), so `target_moved` is true.
- TRAP: target alive and not already trapped (effects.py:279-285).

### contact_proc — effects.py:473, index `[target_side, hit_index]` (defender's side)
Used in `_contact_ability` (effects.py:343-365). **Relevant iff** the defender's ability is
`STATIC` or `POISON_POINT` (`state.ability[target_side, target]`, no Mold Breaker adjustment),
the move has the CONTACT flag, the hit is effective with `dealt > 0`, the attacker is alive after
the hit and `can_have_status(attacker, status)` (effects.py:351-359). Also reveals the ability to
the AI (`reveal_ability`, effects.py:364). For Pursuit pre-hit: `contact_proc[switching side, 0]`.

### n_hits — effects.py:123-127
`hit_count = n_hits[side]` iff `move_effect == MULTI_HIT`, else 1. **Relevant iff** the selected
move is MULTI_HIT and it connects. Hits stop once the target faints (`landed` requires target alive,
effects.py:425-429), so n_hits values >= k are equivalent once the target faints at hit k (the
enumeration may collapse them; merging catches it anyway). For MULTI_HIT, crit/roll/secondary/
contact_proc of indices < n_hits are relevant.

### full_para — step.py:305
`full_para = (state.status[side, user] == PARALYSIS) & (outcome.full_para[side] != 0)`;
`can_act = go & ~full_para` (step.py:306). Status is read **at the moment the side acts** (after
the earlier actor's effects this turn). **Relevant iff** side uses a move (not switch/item), is
`ready` and `go` (alive, not still asleep, not still frozen unless thawing move, not flinched, not
Taunt-blocked, no confusion self-hit), and is paralyzed at that time — i.e. paralyzed at turn start,
OR could be paralyzed earlier in the turn (the other side acts first with a paralysis-inflicting
move — Stun Spore, Spark's secondary — or Static on a contact move; Natural Cure does not apply
mid-turn). Conservative mask: enumerate when the active is paralyzed at turn start or the other
side's action can paralyze it before it moves.

### confusion_hit — step.py:292-302
`confused = go & (confusion[side] > 0)`; counter drops first; `self_hit = confused & (confusion > 0)
& (outcome.confusion_hit[side] != 0)` (step.py:292-294). **Relevant iff** side uses a move, `go`,
and `confusion[side] >= 2` at that point (after the decrement it is still > 0). Confusion can be
inflicted earlier the same turn by a CONFUSE_FOE move of the first actor (counter 2..5, so >= 2 ->
relevant). When it is 1 the battler snaps out (no read).

### duration — effects.py:190, 286, 297, 307 (index `h.side` = acting side)
- `_inflict` (effects.py:186-200): counter = `duration[side]` **only when the inflicted status is
  SLEEP** (`status == Status.SLEEP`), via STATUS_FOE (:203-204) or SECONDARY_STATUS (:207-208).
- `_trap` (:277-287): `trapped = duration[side]` (TRAP).
- `_taunt` (:290-297): `taunt = duration[side]` (target not already taunted).
- `_confuse_foe` (:300-307): `confusion = duration[side]` (target not already confused).
**Relevant iff** the side's move is a sleep-inducing STATUS_FOE/SECONDARY_STATUS (param[0] ==
SLEEP), TRAP, TAUNT or CONFUSE_FOE AND the effect actually applies (same gates as the effect). One
value per side suffices (each side's move runs once per turn; multi-hit + these effects do not
co-occur).

### thaw — step.py:274-281, 308
`frozen = ready & (status == FREEZE)`; `thawed = frozen & (outcome.thaw[side] != 0)` (step.py:277-278).
**Relevant iff** side uses a move, is ready, and is frozen at that point (frozen at turn start or
frozen earlier in this turn by the other side's move). For a thawing move (SECONDARY_STATUS with
`move_param[m,2] == 1`, step.py:274-276) the user acts anyway and thaws at SELF_THAW (step.py:308),
so for a thawing move both `thaw` values give the same state (unfrozen, move goes on); `thaw` then
matters only for the message (thaw message timing), not for the state. The defender thaw on a Fire hit
(`_thaw_target`, effects.py:368-375) reads **no** outcome field.

### end_speed_tie — step.py:663-673
`tie = (speed0 == speed1) & (outcome.end_speed_tie != 0)` with `effective_speed` on the state
**after `execute_actions`** (step.py:683, `end_of_turn` receives the post-action state). Speeds can
change mid-turn (switch-ins, paralysis, Speed stage changes, Steadfast, faint cleanup resets stages).
**Relevant iff** the post-action effective speeds are equal (and practically only if the end-of-turn
order can change the result). Exact options: (a) run `_record_last_move` + `action_order` +
`execute_actions` (all importable from step.py; `_record_last_move` is private) and test speeds;
(b) enumerate {0,1} whenever a tie is possible and let state merging collapse it.

### flinch_item, quick_claw
Never read. Fix to 0.

### Pursuit (step.py:429-474, 543-549, 558-559)
`pursuit_fires(tables, state, side, actions, choices) -> bool` (step.py:429-448) is public. When it
fires, the pursuer hits with fields `[pursuer, 0]` (crit, roll; secondary is chance 0 for Pursuit;
`contact_proc[switcher, 0]`) and `power_mul=20`, no accuracy, and the pursuer does not act again
(step.py:558-559) — so its own `hit`, `full_para`, etc. are irrelevant. Note `pursuit_fires` does
not check paralysis/confusion/flinch (only alive, not asleep/frozen, PP, not struggling).

## 4. Damage functions (damage.py)

- `ROLLS = 16` (damage.py:39); `CRIT_MUL = 2` (:38); `PHYSICAL, SPECIAL, STATUS = 0, 1, 2` (:37);
  `CONFUSION_POWER = 40` (:40).
- `damage_table(tables: Tables, state: BattleState, atk_side, move, crit) -> jnp.ndarray`
  (damage.py:257-262): `int32[16]`, damage for rolls 0..15 against **the other side's current
  active** (`state.active[1 - atk_side]`), **power_mul fixed at 10** (no Pursuit doubling). Ascending
  in roll. Status/power-0 moves: all 0 (damage.py:254). NOT capped at the target's HP.
- `damage_for_roll(tables, state, atk_side, move, crit, roll, power_mul=10) -> jnp.ndarray`
  (damage.py:265-273): `int32` scalar, `damage_table(...)[roll]` generalised with `power_mul`. This is
  what `apply_move_effect` calls (effects.py:431-439). Then `dealt = min(damage, target_hp)` masked by
  `landed & ~targets_user` (effects.py:440-441).
- `calc_move_damage(tables, state, atk_side, move, power_mul=10) -> int32` (damage.py:276-282):
  pre-variance, crit 1.
- `ai_damage_for_roll(tables, state, atk_side, move, roll) -> int32` (damage.py:285-302): AI order
  (variance last); only used by the AI.
- `confusion_damage(tables, state, side, roll) -> int32` (damage.py:305-312): self-hit, power 40,
  typeless, no crit, `max(d*(85+roll)//100, 1)`.

### Are equal-damage rolls interchangeable in `step`?
Yes. Roll enters `step` only via (1) `damage_for_roll` -> `damage` -> `dealt` and `passes = damage > 0`
(effects.py:454; for a damaging move `damage > 0` does not depend on the roll: variance is floored
at 1 (damage.py:234) and `divide` never truncates a non-zero value to 0 (damage.py:53-55), so it is 0
only for an immunity), and (2) `confusion_damage` (step.py:296). Everything downstream (drain,
recoil, contact ability `dealt > 0`, berries, faint) depends on `dealt`/HP only. So rolls with equal
`min(damage, target_hp)` are interchangeable, and all KO rolls collapse (`dealt` = target HP).
Runtime check passed (§3 roll).

**Pitfalls for pre-merging rolls (exactness):**
1. The table must be computed on the state **at the moment of the hit**, not the turn-start state:
   the second actor's damage depends on what the first did (Leer/Growl/Screech stage drops,
   Intimidate on a switch-in, the attacker's HP for Blaze/Torrent/Overgrow pinch (damage.py:133-138),
   burn from Ember, a switched-in target with different stats and HP after Stealth Rock, Water Sport
   halving Fire). `damage_table` uses the passed state's actives, so a turn-start table is wrong for
   the second actor in those cases; a switching target needs the post-switch state (cf.
   `playout._facing`, playout.py:91-100, which only approximates: it ignores Stealth Rock/Intimidate).
2. `damage_table` has no `power_mul`: for Pursuit on a switch use `damage_for_roll(..., power_mul=20)`
   vmapped over rolls 0..15.
3. The KO cap uses the target's HP at hit time (may differ from turn start: berries, earlier hits,
   Stealth Rock on entry).
4. For multi-hit moves each hit index has its own table (target HP changes).
5. Confusion self-hit uses a different table (`confusion_damage`) on `roll[side,0]`; partition rolls
   conditional on `confusion_hit`.
Safe exact approach: treat pre-merge only as an optimisation whose partition is computed on the
hit-time state (or verify by stepping), and rely on the post-`step` bit-identical merge for
correctness (the spec's "merge only when step produced bit-identical states").

### Merge key caution (observation)
States can be bit-identical while observations differ: a crit vs a non-crit with equal capped
damage (KO either way), a miss vs a hit doing nothing new (e.g. Leer at -6, Stealth Rock already
up: both still write `last_move`/`move_hit` identically since those are set before the accuracy
check, step.py:327-328, 364-367), full paralysis vs a move that changes nothing, a secondary proc
that is blocked. If children are merged on state alone, the observation (spec §2 says "missed",
"crit" come from the outcome) is lost. Merge on (next_state, observation key) or compute the
observation before merging.

## 5. AI API (src/nuzline/ai/outcome.py, src/nuzline/ai/policy.py)

`AiOutcome(NamedTuple)` (ai/outcome.py:135-140):
- `sim_roll: int8[4]` — roll index per opponent move slot; always 15 for Roark (`SIM_ROLL_MAX`, :89, :231, :264).
- `ability_guess: int8[]` — player ability slot 0/1, 1/2 each.
- `script_rolls: int8[MAX_SCRIPT_ROLLS=4]` — 1 = `IfRandomLessThan N` jump, P = N/256 (:87, :233).
- `tiebreak: int8[]` — `RandNext % 12`, P = `TIEBREAK_P` = 5462/65536 for 0..3, 5461/65536 for 4..11 (:132).
- `switch_stop: int8[]` — first stopping switch draw 0..8, 9 = `SWITCH_NONE`, -1 refused.
Constants: `ENUMERATION_WIDTH = 3840` (:124-126), `MAX_ENUMERATED = 4096` (:88), `SWITCH_SITES = 9`
(ai/switching.py:62), `SWITCH_NONE = 9` (:63), `NO_SWITCH = -1` (:65), `REFUSED_ACTION = -2` (:66),
`NO_ITEM = -1` (policy.py:83).

Functions:
- `sample_ai_outcome(key, tables, state) -> AiOutcome` (ai/outcome.py:223-236).
- `enumerate_ai_outcomes(tables, state) -> tuple[AiOutcome (leading axis 3840), float32[3840]]` (:253-270).
- `script_roll_probs(state) -> (int8[16,4], float32[16])` (:193-199); `switch_stop_probs(switch_sites) -> float32[10]` (:202-210).
- `ai_policy(tables, state, ai_outcome) -> int32` (policy.py:568-574): switch `4+j`, else item `10+k`,
  else move slot 0..3; raises `UnsupportedAi` on a refused concrete state; `REFUSED_ACTION` traced.
- `ai_action_distribution(tables, state) -> float32[NUM_OPP_ACTIONS=14]` (policy.py:654-660): exact,
  factored (`_distribution`, policy.py:627-651, jitted): switch probabilities from `switch_stop`, then
  `p_stay` x (item one-hot if `_use_item` picks one, else the move distribution over 32 scorings x 12
  tiebreaks). Raises `UnsupportedAi` on a refused concrete state (`_check`), NaN when traced/refused.
  Probabilities are **float32**, not exact rationals (spec calls it exact; it is exact up to float32
  rounding: tiebreak/script/switch probabilities are k/65536 or N/256 products).
- `ai_action_distribution_enumerated(tables, state) -> float32[14]` (policy.py:577-590): brute-force
  oracle (3840 vmapped), NaN if any action refused.
- `ai_scores`, `maybe_switch`, `maybe_use_item` (policy.py:502-538).
- Action encoding (src/nuzline/engine/actions.py:36-39): `MOVE_ACTIONS=4`, `NUM_ACTIONS=10`,
  `FIRST_ITEM_ACTION=10`, `NUM_OPP_ACTIONS=14`. `legal_opponent_actions(tables, state) -> bool[14]` (:76-81).
- Roark turn 0 (Monferno vs Geodude): `ai_action_distribution` = `[0.75, 0.25, 0, ...]`
  (Stealth Rock 0.75, Rock Throw 0.25) (runtime).

**Does step need anything from AiOutcome?** No. `step(tables, state, actions, outcome)` takes only
`actions: int32[2]` and the battle `Outcome` (step.py:692-700); no AiOutcome field is read in the
engine. AI knowledge in the state (`ai_seen_moves`, `ai_ability_known`, `last_move`, `move_hit`) is
updated by step deterministically (step.py:185-201, 204-216; effects.py:364). After the turn, the
opponent replacement is `send_replacements(tables, state, ai_switched, power_mul) -> (state, refused)`
(playout.py:157-176), deterministic given state + `ai_switched = is_switch(actions[1])` + `power_mul =
20 if pursuit_fires(tables, pre_turn_state, 0, actions, choices) else 10` (playout.py:179-191,
`_NORMAL_POWER_MUL=10` :76). It uses `post_ko_switch_in(tables, state, power_mul) -> int32`
(ai/replacement.py:115-126; raises `UnsupportedAi` on a stale-score case). A replacement needed in a
turn the AI switched is **refused** (playout.py:170-171; roark.md 14 item 3) — Stage B must handle
that refusal. So the opponent action (plus the pre-turn state) is all the battle layer needs; the
AiOutcome is needed only to compute `ai_action_distribution`.
Observation note: the AI's internal draws are hidden by spec; only the action matters, and
distinct AiOutcomes leading to the same action are already summed by `ai_action_distribution`.

## 6. Roark fight + sample party: mechanics present and fields they touch

Runtime dump (`dump_roark.py`; `make_state` with `leads=(0,0)`, weather 0, opp bag `(17, 17)` =
2 Potions). Stats vector order `[ATK, DEF, SPA, SPD, SPE]` (`Stat.SPE-1` = index 4, order.py:62;
ids.py:108-118).

### Pokemon
| side/slot | species | lv | HP | ability | item | types | Spe | moves |
|---|---|---|---|---|---|---|---|---|
| P0 | Monferno | 14 | 44 | BLAZE | - | Fire/Fighting | 30 | Scratch, Leer, Ember, Mach Punch |
| P1 | Prinplup | 14 | 44 | TORRENT | - | Water | 21 | Pound, Growl, Bubble, Water Sport |
| P2 | Grotle | 14 | 47 | OVERGROW | Sitrus Berry | Grass | 17 | Tackle, Withdraw, Absorb, Razor Leaf |
| P3 | Starly | 10 | 30 | KEEN_EYE | Oran Berry | Normal/Flying | 19 | Tackle, Growl, Quick Attack, Wing Attack |
| P4 | Shinx | 13 | 37 | RIVALRY | - | Electric | 19 | Tackle, Leer, Spark |
| P5 | Budew | 10 | 30 | NATURAL_CURE | - | Grass/Poison | 18 | Absorb, Growth, Water Sport, Stun Spore |
| O0 | Geodude | 12 | 32 | ROCK_HEAD | - | Rock/Ground | 10 | Stealth Rock, Rock Throw |
| O1 | Onix | 12 | 31 | ROCK_HEAD | - | Rock/Ground | 22 | Stealth Rock, Rock Throw, Screech |
| O2 | Cranidos | 14 | 43 | MOLD_BREAKER | - | Rock | 24 | Headbutt, Pursuit, Leer |

### Moves (tables rows at runtime)
| move (id) | effect | chance | acc | pow | cat | prio | contact | Outcome fields it can touch |
|---|---|---|---|---|---|---|---|---|
| Pound (1), Scratch (10) | PLAIN | 0 | 100 | 40 | phys | 0 | yes | crit, roll |
| Wing Attack (17) | PLAIN | 0 | 100 | 60 | phys | 0 | yes | crit, roll |
| Quick Attack (98), Mach Punch (183) | PLAIN | 0 | 100 | 40 | phys | +1 | yes | crit, roll |
| Tackle (33) | PLAIN | 0 | 95 | 35 | phys | 0 | yes | hit (95%), crit, roll |
| Rock Throw (88) | PLAIN | 0 | 90 | 50 | phys | 0 | no | hit (90%), crit, roll |
| Headbutt (29) | FLINCH | 30 | 100 | 70 | phys | 0 | yes | crit, roll, secondary (flinch, only when Cranidos moves first and the target then moves) |
| Pursuit (228) | PURSUIT | 0 | 100 | 40 | phys (Dark) | 0 | yes | crit, roll (incl. pre-switch hit x2 power) |
| Ember (52) | SECONDARY_STATUS burn | 10 | 100 | 40 | spec | 0 | no | crit, roll, secondary (burn; targets are not Fire) |
| Bubble (145) | SECONDARY_STAT_FOE Spe -1 | 10 | 100 | 20 | spec | 0 | no | crit, roll, secondary |
| Spark (209) | SECONDARY_STATUS paralysis | 30 | 100 | 65 | phys | 0 | yes | crit, roll, secondary; immune (0 dmg) vs Rock/Ground Geodude/Onix -> nothing |
| Absorb (71) | DRAIN 50 | 0 | 100 | 20 | spec | 0 | no | crit, roll |
| Razor Leaf (75) | HIGH_CRIT | 0 | 95 | 55 | phys* | 0 | no | hit (95%), crit (1/8), roll |
| Leer (43) | STAT_FOE Def -1 | 0 | 100 | 0 | status | 0 | no | none |
| Growl (45) | STAT_FOE Atk -1 | 0 | 100 | 0 | status | 0 | no | none |
| Screech (103) | STAT_FOE Def -2 | 0 | 85 | 0 | status | 0 | no | hit (85%) |
| Stun Spore (78) | STATUS_FOE paralysis | 0 | 75 | 0 | status | 0 | no | hit (75%); enables full_para for the target later |
| Growth (74), Withdraw (110) | STAT_SELF | 0 | 0 | 0 | status | 0 | no | none (acc 0) |
| Water Sport (346) | WATER_SPORT | 0 | 0 | 0 | status | 0 | no | none |
| Stealth Rock (446) | STEALTH_ROCK | 0 | 0 | 0 | status | 0 | no | none |
| Struggle (165) | (table effect UNSUPPORTED; step uses `struggling`) | 0 | 0 | 50 | phys | 0 | yes | crit, roll |
(*Razor Leaf: `move_category = 0` (physical) in the tables, correct for Gen 4's per-move split.)

### Abilities / items and Outcome fields
- Blaze/Torrent/Overgrow, Rivalry, Rock Head, Mold Breaker, Natural Cure, Keen Eye: no Outcome
  field; they change damage values (pinch, Rivalry — all male, trainer mons gender 0 too) or block
  things deterministically. Keen Eye matters only vs accuracy drops (none in this fight).
- No STATIC / POISON_POINT anywhere -> **contact_proc never relevant**.
- No INNER_FOCUS, STEADFAST, SIMPLE, UNAWARE, INTIMIDATE, GUTS.
- Sitrus Berry, Oran Berry (hooks.after_damage_items), Potion x2 (opponent item action): no Outcome field.
- No Lucky Chant, fog, gravity, weather, screens.

### Fields that are **never relevant** in this fight
`contact_proc` (no Static/Poison Point), `n_hits` (no MULTI_HIT), `confusion_hit` (no confusion
source), `duration` (no sleep, TRAP, TAUNT, CONFUSE_FOE move), `thaw` (no freeze source),
`flinch_item`, `quick_claw`, all `[*, 1..4]` hit indices (no multi-hit), `hit[*,0]` for
accuracy-100 moves at neutral acc/eva stages (no acc/eva-changing move exists, so thresholds stay at
the move's accuracy; acc 100 -> P(hit) = 1).

### Speed ties
Player speeds {30, 21, 17, 19, 19, 18}; opponent {10, 22, 24}, paralyzed {2, 5, 6} (only the
opponent can be paralyzed: Stun Spore, Spark; the opponent has no status moves and no Static). No
Speed-stage moves exist except Bubble's Spe -1 on the opponent (stage -1: x10/15 -> Geodude 6,
Onix 14, Cranidos 16; -2: 5, 11, 12; -3: 4, 8, 9; -4: 3, 7, 8; -5: 2, 6, 6; -6: 2, 5, 6, via
`apply_stage` damage.py:58-61, `_STAGE_NUM/_DEN` :43-44, also combined with paralysis //4).
The player has no Speed-changing effect (Growth = SpA, Withdraw = Def; no paralysis source on the
opponent's side), so player speeds are always {30, 21, 17, 19, 19, 18}. No opponent value in any
stage/paralysis combination listed above (nor paralysis on top of a stage: all <= 5) equals 17, 18,
19, 21 or 30. **Hence speed_tie and end_speed_tie are never relevant for Roark + this party**;
the generic check should still be implemented (other parties/fights).

### Relevant fields, per action pair
- Player damaging move: `hit[0,0]` (only Tackle 95, Razor Leaf 95), `crit[0,0]`, `roll[0,0]`,
  `secondary[0,0]` (Ember 10, Bubble 10, Spark 30).
- Player Stun Spore: `hit[0,0]` (75). Screech is opponent-only.
- Opponent: `hit[1,0]` (Rock Throw 90, Screech 85), `crit[1,0]`, `roll[1,0]`, `secondary[1,0]`
  (Headbutt 30, gated by Cranidos moving first).
- `full_para[1]` when the opponent active is paralyzed at turn start, or the player acts first with
  Stun Spore / Spark this turn and the opponent then uses a move.
- Opponent switch/Potion: no opponent fields. Player switch: no player fields (Pursuit then uses
  `crit[1,0]`, `roll[1,0]`).

### Width estimate (pre-merge)
Per side, a damaging move gives at most `miss + 2 crit x 16 rolls x 2 secondary` = 65 combos (a
miss collapses everything else). After collapsing rolls to distinct capped damage values (runtime
`widths.py`, full-HP leads at turn 0: non-crit 1-6 distinct values, crit 1-7, union <= 11, e.g.
Cranidos Headbutt vs Monferno: nc 4, c 7), a typical side has ~ (1 miss) + (d_nc + d_c) x 2 <= ~25
combos. Joint worst case ~ 65 x 65 x 2 (full_para) = 8,450 without roll collapsing, a few hundred
with it — well under the spec's 10^5 cap. Spec's measured 1 to ~111 distinct next states (mean ~14)
after merging is consistent.

Distinct damage values at turn 0 (non-crit / crit, capped at target HP), from `widths.py`:
```
MONFERNO vs GEODUDE | Scratch 2/3; Ember 3/6; Mach Punch 3/1; O:Rock Throw 3/4
MONFERNO vs ONIX    | Scratch 1/2; Ember 3/4; Mach Punch 2/3; O:Rock Throw 2/3
MONFERNO vs CRANIDOS| Scratch 2/3; Ember 2/4; Mach Punch 3/1; O:Headbutt 4/7; O:Pursuit 2/3
PRINPLUP vs GEODUDE | Pound 2/2; Bubble 1/1; O:Rock Throw 3/4
PRINPLUP vs ONIX    | Pound 1/2; Bubble 2/1; O:Rock Throw 2/3
PRINPLUP vs CRANIDOS| Pound 2/3; Bubble 3/3; O:Headbutt 4/7; O:Pursuit 3/4
GROTLE vs GEODUDE   | Tackle 2/3; Absorb 1/1; Razor Leaf 1/1; O:Rock Throw 2/3
GROTLE vs ONIX      | Tackle 1/2; Absorb 2/1; Razor Leaf 2/1; O:Rock Throw 2/3
GROTLE vs CRANIDOS  | Tackle 2/3; Absorb 2/3; Razor Leaf 3/1; O:Headbutt 4/6; O:Pursuit 3/4
STARLY vs GEODUDE   | Tackle 2/3; Quick Attack 2/3; Wing Attack 2/3; O:Rock Throw 1/1
STARLY vs ONIX      | Tackle 2/2; Quick Attack 2/2; Wing Attack 2/3; O:Rock Throw 3/1
STARLY vs CRANIDOS  | Tackle 2/3; Quick Attack 3/3; Wing Attack 2/3; O:Headbutt 4/1; O:Pursuit 4/1
SHINX vs GEODUDE    | Tackle 2/2; Spark 1/1 (immune); O:Rock Throw 3/5
SHINX vs ONIX       | Tackle 1/2; Spark 1/1 (immune); O:Rock Throw 3/4
SHINX vs CRANIDOS   | Tackle 2/3; Spark 4/1; O:Headbutt 5/1; O:Pursuit 4/6
BUDEW vs GEODUDE    | Absorb 2/1; O:Rock Throw 3/1
BUDEW vs ONIX       | Absorb 2/1; O:Rock Throw 3/4
BUDEW vs CRANIDOS   | Absorb 2/3; O:Headbutt 6/1; O:Pursuit 4/1
```
(A "1" usually means every roll KOs or the target is immune.) Example turn-0 tables, Monferno
Ember vs Geodude: non-crit `[11,11,11,11,12×11,13]`, crit `[22,22,23,23,24×6,25,25,25,26,26,27]`;
Geodude Rock Throw vs Monferno: non-crit `[9,9,9,10×12,12]`, crit `[19,19,19,21×6,22×6,24]`;
`hit_threshold = [100, 90]`, `crit_odds = [16, 16]`.

## 7. Summary of the relevance mask (generic, for `enumerate_outcomes`)

For side s with selected move m (turn-start `selected_move`), target slot T (`_target_mon`):
- s switches or uses an item: no s-indexed field is relevant (except `contact_proc[s,*]` as a
  defender and Pursuit fields of the other side).
- `hit[s,0]`: s moves, `move_accuracy[m] != 0`, `hit_threshold[s] < 100`.
- `crit[s,i]`, `roll[s,i]`: i < hit_count, move damaging (category != STATUS and power != 0), hit lands.
- `secondary[s,i]`: as crit, plus chance != 0 and effect in {SECONDARY_STAT_SELF, SECONDARY_STAT_FOE,
  SECONDARY_STATUS, FLINCH, TRAP} and the effect's own gate (status possible on T; flinch only if s
  acts first and the target then moves).
- `contact_proc[1-s,i]`: T's ability STATIC/POISON_POINT, m CONTACT, damaging, s can take the status.
- `n_hits[s]`: m is MULTI_HIT.
- `full_para[s]`: s moves and is (or can become this turn before acting) paralyzed.
- `confusion_hit[s]`: s moves and confusion[s] >= 2 at action time (incl. confusion inflicted
  earlier this turn).
- `thaw[s]`: s moves and is (or can become this turn) frozen.
- `duration[s]`: m is sleep STATUS_FOE/SECONDARY_STATUS, TRAP, TAUNT or CONFUSE_FOE.
- `roll[s,0]` also when `confusion_hit[s]` relevant (self-hit damage).
- `speed_tie`: same-kind actions, equal priority, equal turn-start effective speeds.
- `end_speed_tie`: equal effective speeds after `execute_actions`.
- `flinch_item`, `quick_claw`: never.
- If `state.over != 0`: nothing relevant (step returns the state unchanged).
Mid-turn changes can only be handled conservatively by these masks (enumerate when possible);
exactness is preserved because the post-`step` merge collapses outcomes that turned out not to
matter.
