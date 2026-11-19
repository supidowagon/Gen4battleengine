# Doubles battle state: design on paper

Status: design only, nothing built. Date: 2026-11-13. Survey base: `main` at e139cd1.
Decomp: `external/pokeplatinum` at 579d2d5. Every battle rule below cites the decomp.

## Summary

- **Recommendation: index everything that belongs to a Pokemon on the field by *battler*, with the
  number of battlers fixed per fight (2 for singles, 4 for doubles, two-trainer and multi fights),
  using the game's own numbering: 0 player, 1 enemy, 2 player's second, 3 enemy's second.**
- With that numbering a singles fight has battlers 0 and 1, which are today's side 0 and side 1.
  Today's arrays keep their shapes and values, so singles fights compile to the same program and
  should run at the same speed (expected from the structure, not measured).
- A fixed four slots with a mask for every fight was considered and not recommended: 258 of the
  359 battles are singles and would pay for two masked battlers on every step.
- What changes for real is everything that says "the other side": a move's target becomes data,
  the turn runs up to four actors, and chance and AI outcomes get wider.
- The two singles streams can avoid most rework with seven conventions (section 5). The one file
  that will be largely rewritten regardless is `engine/enumerate.py`.
- Size: roughly 15 to 20 work batches (an estimate, section 6). The backlog's "about 40 % not
  singles" counts trainer rows; counted as battles it is 101 of 359, about 28 %.

Short file names used below: `lib.c` = `src/battle/battle_lib.c`, `bcp.c` =
`src/battle/battle_controller_player.c`, `sys.c` = `src/battle/battle_system.c`, `main.c` =
`src/battle/battle_main.c`, `script.c` = `src/battle/battle_script.c`, `ai.c` =
`src/battle/trainer_ai/trainer_ai.c`, `context.h` = `include/battle/battle_context.h`,
`battle.h` = `include/constants/battle.h`. Engine paths are under `src/nuzline/`.

## 1. Where singles is baked in today

The state has no second active slot. `docs/design.md:31` says the layout "keeps a slot dimension
for two active slots per side"; the code does not. `engine/state.py:85` sets `SIDES, SLOTS = 2, 6`,
party data is `[2, 6]`, and every volatile field is `[2]`, one active Pokemon per side
(`state.py:120-149`, `:269-297`). `engine/STATE.md:32-37` states this mapping openly.

**Cheap to change later (mechanical).**

- Functions that take `side` and read `state.active[side]`: most of `step.py`, `hooks.py`,
  `effects.py`, `order.py`, `actions.py`. With the recommended numbering `side` already is the
  battler in singles, so these change name and meaning, not code.
- Loops `for side in (0, 1)` (`state.py:492`, `step.py:1058-1059`, `order.py:157-159`).
- The `STATE.md` rows marked "not needed because in singles": `bindTarget` (:314), the four screen
  users (:346-352), Follow Me (:354-355), `wishTarget` (:372), `moveHitBattler` (:498). They are an
  honest list of what the doubles milestone must add.

**Medium (every site must be visited, each is small).**

- "The foe is `1 - side`": 26 sites in `step.py`, 15 in `outcome.py`, 35 in `reference/engine.py`,
  a few each in `hooks.py`, `effects.py`, `damage.py`. Examples: `effects.play_hit` sets
  `target_side = 1 - side` (`effects.py:1047`); `damage._pre_random` and `damage_table` take only
  `atk_side` and derive the defender (`damage.py:225`, `:460`); `hooks.intimidate` lowers "the
  foe" (`hooks.py:316-329`); `hooks.on_switch_in` runs the item check for `side`, then `1 - side`
  (`hooks.py:344-345`).
- Fields that hold "the other one" as a flag: `infatuated` is 0 or 1 (`state.py:39-42`).
- Player-only arrays keyed by party slot: `ai_seen_moves[6, 4]`, `ai_ability_known[6]`
  (`state.py:178-179`; `hooks.reveal_ability` tests `side == 0`, `hooks.py:126-134`), and the
  experience and death block (`state.py:181-196`).
- `make_state(player, opp, leads=(0, 0))` takes two parties and one lead each (`state.py:476-494`).
- The reference engine keeps a side's party, active slot, volatiles and side conditions in one
  class (`reference/state.py`, `RefSide`), so its battler and side state must be split.

**Costly (structure, not sites).**

- The turn skeleton. `step._execute_actions` (`step.py:1047-1117`) has exactly two actors: `first`,
  `pursuer = 1 - first` (:1066), `second = position == 1` (:1089), one `foe_protecting` flag
  (:1091), `fori_loop(0, 2, ...)` (:1116). `end_of_turn` loops two positions; `needs_replacement`
  and `replace` work per side (`step.py:1556-1566`).
- Turn order. `order.action_order` builds a two-entry order from one boolean, `player_second`
  (`order.py:208-234`); `compare_speed` takes one battler and uses `other = 1 - battler1` (:166-179).
- Actions. One scalar per side, `0-3` move, `4-9` switch, `10-13` item, and no target
  (`actions.py:42-45`). `NUM_ACTIONS` / `NUM_OPP_ACTIONS` are used in `ai/policy.py` (14 places),
  `search/playout.py`, `search/stage_a.py`, `search/line.py`, `search/expand.py`.
- Chance outcomes. `Outcome` has a scalar `speed_tie` "0 = player first" and `[2, 5]` / `[2]`
  fields (`outcome.py:62-86`). `enumerate.py` (1,946 lines) is built on two orders of two actors:
  `Reads` fields shaped `[speed_tie branch, side]`, such as `confused_by_foe` (`enumerate.py:249-270`).
- The AI. `Battler.DEFENDER = 0` and `ATTACKER = 1` are the side numbers themselves
  (`ai/commands.py:68-69`, `_side` :565); `if_target_is_partner` is a constant "no" (:596-598) and
  its jump is dropped (`ai/policy.py:146-148`); the policy scores four moves against one target.
- What the player sees. `observe.py` records events per side.
- Search and prep. `playout._turn` takes one player action (`search/playout.py:388`); Stage B,
  the plan tree, the line player and prep (one lead) all assume one active Pokemon. Coverage
  refuses the other formats outright (`engine/coverage.py:219-220`).

## 2. What the real game does (decomp only)

**Battlers, sides, partners.** The context holds four battlers: `battleMons[MAX_BATTLERS]`
(`context.h:265`), `MAX_BATTLERS 4`, two sides, two per side (`battle.h:59-61`). Ids are player 1 =
0, enemy 1 = 1, player 2 = 2, enemy 2 = 3 (`battle.h:95-98`; type table `main.c:965-968`). A
battler's side is its type's low bit (`sys.c:360-363`). The partner is the other battler on the
same side, or the battler itself outside doubles (`sys.c:394-411`); the AI uses `battler ^ 2`
(`ai.c:2776-2780`). Outside doubles the opponent is `attacker ^ 1` (`sys.c:419-421`). A singles
fight creates two battlers (`main.c:1325-1333`) and still flags battlers 2 and 3 as absent
(`lib.c:2031-2034`).

**Battle types and parties.** Flags: `DOUBLES`, `2vs2` ("two distinct battlers on each side"),
`TAG` ("a battle with a partner"), `AI` (`battle.h:27-32`); combinations at `battle.h:45-48`. All
three non-single kinds create four battlers (`main.c:1212-1324`). Which party a battler draws from
(`sys.c:124-133`): in `2vs2`, its own; in `TAG`, the enemies each have their own and the player's
two battlers share one; in plain doubles, `parties[battler & 1]`. Doubles is always Set mode
(`BATTLE_TYPE_FORCED_SET_MODE`, `battle.h:50`; `bcp.c:4143-4146`). A "2-vs-1" type: not found.

**Per battler or per side.** Per battler: the `BattleMon` with its volatiles, `turnFlags` and
`selfTurnFlags` (`context.h:216-217`), `speedRand` (:293), `battlerActions` (:256),
`selectedPartySlot` (:252), `moveHitBattler` (:273), `lastHitByBattler` (:287), and Wish and Future
Sight (`FieldConditions`, `context.h:28-36`). Per side: `sideConditions[NUM_BATTLE_SIDES]` (:214),
which also stores which battler set each screen and Follow Me (:39-48), and `sideGetExpMask[2]`
(:176).

**Targets.** In doubles the player picks a target for every move except an ally move whose
partner is absent; in singles the target is the user for own-side ranges, else `battler ^ 1`
(`bcp.c:4342-4369`). The choice is stored per battler (`BATTLE_ACTION_CHOOSE_TARGET`).
`BattleSystem_Defender` (`lib.c:1600-1730`) resolves the defender by
range (the full list is appendix A). The usual single-target move goes to the Follow Me user if
alive, else the chosen target, else a random opponent (:1709-1727); spread ranges start at the
first living target in speed order (:1611-1650); ally moves go to the partner (:1670-1675).
Lightning Rod and Storm Drain then pull a single-target or random-opponent Electric or Water move
to their holder (`lib.c:1732-1790`). Which battlers each target menu
offers is not transcribed (appendix B).

**Spread moves.** After the first target the move loops: `LoopSpreadMoves` (`bcp.c:3829-3888`)
walks the speed order, sets the next living enemy (:3839-3862) or the next living battler other
than the user (:3865-3883) as defender and re-enters at `BATTLE_CONTROL_BEFORE_MOVE`. Damage is
`x 3/4` for an adjacent-opponents move when the defender's side has two alive (`lib.c:7035-7038`)
and for an all-adjacent move when at least two battlers other than the defender are alive
(:7040-7043; `BattleSystem_CountAliveBattlers`, `lib.c:2825-2851`). Reflect and Light Screen cut to
`2/3` instead of `1/2` when the defender's side has two alive (`lib.c:6982-6990`, `:7023-7031`).

**Turn order.** `CalcTurnOrder` (`bcp.c:707-790`): items and switches first in battler order
(:754-761), then fights (:763-770), then every pair with the same command is compared and swapped
(:772-786) by `BattleSystem_CompareBattlerSpeed` (`lib.c:1188`). Each comparison at equal speed
draws its own `RandNext() & 1` (`lib.c:1400-1447`), so four battlers can draw up to six tie bits,
and a fainted battler loses every comparison (:1204-1211). The end-of-turn order is the same pair
loop without priority (`BattleSystem_SortMonSpeedOrder`, `lib.c:3364-3383`); the end-of-turn checks
then run per battler in that order (`bcp.c:1289-1303`). "Has this battler moved" is per battler
(`Battler_MovedThisTurn`, `lib.c:5365-5368`).

**Replacement after a faint.** `ReplaceFainted` (`bcp.c:4070-4155`) loops all battlers. In shared-
party doubles, when both of a side's battlers are down the first slot is served first (:4083-4086);
the Pokemon the partner has selected is not available (:4103-4105; also `lib.c:3151-3172`); a
battler with nothing left to send is flagged absent (:4109-4111) and the fight goes on with one.
The AI's replacement pick excludes the partner's slot and scores against a random opponent
(`lib.c:7941-7949`).

**Ally-affecting mechanics.** Found so far: Helping Hand, Follow Me, Plus and Minus, Flower Gift,
Intimidate on both foes, Lightning Rod and Storm Drain, Mud Sport and Water Sport (citations in
appendix A). The list is not complete; a full sweep is part of the milestone.

**Trainer AI.** Doubles forces `AI_FLAG_TAG_STRATEGY` on (`ai.c:252-255`); a trainer whose own flags
include it runs the same script in singles, where `IfTargetIsPartner` is false (`ai.c:297-306`,
`:2630-2639`). In doubles each AI battler scores its moves once per possible target, its ally
included, picks the best target with random tie-breaks, and almost never aims at the ally
(`TrainerAI_MainDoubles`, `ai.c:352-471`; steps in appendix A). The scripts can name four battlers:
attacker, defender and both partners (`ai.c:2765-2783`). The Tag Strategy script is
`script.s:6637-7695`, about 1,060 lines. Whether one AI battler sees its partner's choice of the
same turn: not found.

## 3. How many fights of each kind

Source: `data/intermediate_kaizo/trainers.json` (the `format` field, `data/trainer.py:166-171`),
split rows only. The backlog table counts trainer rows, so a two-trainer battle counts twice.
"Battles" pairs each row with its `opponent_2`.

| Split | Rows | Single | Double | Two-trainer rows / battles | Multi rows / battles | Battles |
|---|---|---|---|---|---|---|
| Roark | 23 | 23 | 0 | 0 / 0 | 0 / 0 | 23 |
| Gardenia | 43 | 20 | 5 | 0 / 0 | 18 / 11 | 36 |
| Fantina | 41 | 37 | 0 | 4 / 2 | 0 / 0 | 39 |
| Maylene | 34 | 27 | 5 | 0 / 0 | 2 / 1 | 33 |
| Wake | 37 | 25 | 2 | 10 / 5 | 0 / 0 | 32 |
| Byron | 49 | 36 | 7 | 0 / 0 | 6 / 4 | 47 |
| Candice | 47 | 23 | 0 | 0 / 0 | 24 / 12 | 35 |
| Galactic | 102 | 40 | 10 | 38 / 19 | 14 / 7 | 76 |
| Volkner | 25 | 9 | 4 | 12 / 6 | 0 / 0 | 19 |
| Elite Four | 19 | 18 | 1 | 0 / 0 | 0 / 0 | 19 |
| **Total** | **420** | **258** | **34** | **64 / 32** | **64 / 35** | **359** |

So 162 of 420 rows (39 %) but 101 of 359 battles (28 %) are not singles: 34 plain doubles, 32
against two trainers, 35 beside a partner. Multi battle counts are approximate: 4 Gardenia rows
have no `opponent_2`, and 14 multi rows have no known partner team. Also relevant now: 15 singles
rows carry `AI_FLAG_TAG_STRATEGY` (0x80) in their own flags.

## 4. Recommended state design

**Battler axis with a per-fight battler count.** A static `Format` (part of the per-fight
specialisation `design.md` §4.2 already uses) gives `B` (2 or 4), the number of parties `P` and a
table `party_of[battler]`. Battler `b` has side `b & 1` and partner `b ^ 2`.

- Per-battler fields `[B]`: `active` (party slot, or "absent"), stages, every volatile, `qc_roll`,
  Wish and Future Sight, and the per-turn facts (moved, protecting, Helping Hand).
- Per-side fields `[2]`: hazards, screens with their user, Tailwind, Mist, Safeguard, Lucky Chant,
  Follow Me.
- Party fields `[P, 6]`: P = 2 for singles and plain doubles, 3 for two trainers, 4 for multi.
- Fields that point at a battler (infatuation source, binder, Mean Look, Lock-On, last attacker)
  store `1 + battler`.
- AI knowledge per observed battler, per AI trainer; deaths, protection and experience stay on
  party 0, the player's.

**Why not four slots always.** It is one program instead of two, but singles would execute four
masked actors, four end-of-turn passes and double-width outcomes on every step. A per-fight count
keeps singles at today's shapes and cost (the only added work is reading the defender from a
variable) and fits the existing per-fight compile. Doubles then costs about twice the actors per
step, a longer compile and a far wider chance tree. None of this is measured. A mask is still
needed inside `B = 4`, because a side can be down to one battler (`bcp.c:4109-4111`).

**Actions and targets.** Each battler's action is today's scalar plus a target battler. In singles
the target follows from the move's range, as in the game (`bcp.c:4363-4367`), so the singles action
space is unchanged. In doubles the player's decision is the pair of both battlers' actions: at most
16 each (4 moves x up to 3 targets, plus up to 4 switches), so at most 256 joint actions before the
legality mask, which also forbids both switching to the same Pokemon. One resolver modelled on
`BattleSystem_Defender` turns (battler, move, chosen target) into the defender list.

**Chance and enumeration.** `Outcome` fields become `[B, ...]`. `speed_tie` and `end_speed_tie`
become one bit per compared pair (six for four battlers), consumed by the game's own swap loop.
New draws: random target, AI target tie-break. A spread move needs one set of hit draws per target;
the per-hit axis of 5 can carry them, since a spread move has at most three targets and is not
multi-hit (a proposal: which checks the per-target loop repeats is not yet traced). Exact
enumeration multiplies across four actors and across the AI's per-target scoring, so doubles turns
will often exceed the width cap; expect Monte Carlo first and exact search shallower than singles.

**Reference engine and tests.** The reference engine is restructured first, into battler, party and
side records, and can loop over battlers the way the decomp does. The differential generator gains
the format as a drawn input; the singles tests are the `B = 2` case of the same code and must stay
green through the re-index. A recorded doubles fight becomes a golden trace, as for each gym.

## 5. Conventions for the two singles streams, starting now

1. **Pass the other Pokemon in; never compute `1 - side` inside a mechanic.** A new hook or effect
   takes the acting battler and, where another Pokemon is involved, that battler as an argument.
   Reason: in doubles the defender is chosen data (`lib.c:1709-1727`). Example: `effects._Hit`
   already carries `side`, `target_side`, `user`, `target` (`effects.py:261-277`); `hooks.pluck_berry`
   does `foe = 1 - side` (`hooks.py:294`) and is the pattern to stop copying.
2. **Keep battler state and side state apart.** For each new field, take from the decomp struct
   whether it is per battler (`BattleMon`, `turnFlags`, any `[MAX_BATTLERS]` array) or per side
   (`SideConditions`), say so in `STATE.md`, and index side fields with the side of the battler,
   not with the battler variable. Example: `order.effective_speed` reads `stages[side]` (battler)
   and `tailwind[side]` (side) through one variable (`order.py:126-146`).
3. **A field that names another Pokemon stores the battler (`1 + battler`), not a yes/no.** Reason:
   the game stores the battler (Attract: `script.c:5738`; `lockOnTarget`, `meanLookTarget`,
   `bindTarget`, `lastHitByBattler`). Example: `infatuated` is 0/1 today (`state.py:39-42`); the
   coming Lock-On, Mean Look, Counter and Mirror Coat fields should not repeat that.
4. **"Every Pokemon on the field" is a loop over battlers; "an ally has X" is a count over the
   battler's side.** Reason: the game loops `maxBattlers` in speed order and counts per side
   (`lib.c:3795-3808`, `:6802-6809`). Example: `damage.weather_suppressed` already loops
   (`damage.py:103-117`); `hooks.on_switch_in` hand-writes "me, then the foe" (`hooks.py:344-345`).
5. **State turn-order facts about a named battler, not as "first" or "second".** Use "the target
   has moved", "the target is protecting"; never "I am the second actor". Reason:
   `Battler_MovedThisTurn(battler)` (`lib.c:5365-5368`), `turnFlags[battler]` (`context.h:216`).
   Example: `step.py:1089-1091` derive both from `position == 1`; new readers (Payback, Avalanche,
   Sucker Punch) should take the fact for their target as an input.
6. **Read a move's range through the shared helpers only.** New effects use `targets_user`,
   `self_defender`, `defender_always` (`effects.py:812-948`), never their own range test, so the
   milestone replaces one resolver. New chance fields are indexed by the acting battler like the
   existing `[2]` fields, never a scalar meaning "player versus opponent" like `speed_tie`
   (`outcome.py:63`).
7. **AI stream: resolve script battlers in one place and keep partner commands in the
   transcription.** Reason: the scripts name four battlers (`ai.c:2765-2783`), and Tag Strategy
   also runs in singles for trainers that carry the flag (15 singles rows). Example: new commands
   go through `commands._side` (`ai/commands.py:565`) instead of `if battler == Battler.ATTACKER`
   branches (:634, :765, :935); `if_target_is_partner` stays a command (:596) rather than a
   dropped line.

Not on the list on purpose: `enumerate.py`'s two-order branch structure and the `step` turn
skeleton. New mechanics have to extend them as they are; shaping them for four battlers now would
be guessing. They are accepted rework.

## 6. Size of the doubles milestone (estimate) and open decisions

Estimate: **15 to 20 batches**, of the size the streams use now. This is a judgement from the
survey, not a measurement.

| Batch group | Batches | What drives it |
|---|---|---|
| Re-index state and reference engine to battlers, no behaviour change | 1-2 | over 100 "other side" sites; singles tests are the safety net |
| Targets, actions, legality, Follow Me and redirection | 1-2 | the defender resolver and the target menus |
| Four-actor turn order, action loop, faints and replacement | 2 | pair-wise ties; Pursuit; one-battler sides |
| Spread moves and the doubles damage modifiers | 1 | per-target loop |
| Ally mechanics by family | 1-2 | the sweep of section 2's incomplete list |
| Outcomes, enumeration, observations | 2-3 | `enumerate.py`; the largest unknown |
| AI: per-target scoring, Tag Strategy script, partner arguments | 2 | ~1,060 script lines |
| Two-trainer format (three parties, two AI trainers and bags) | 1 | 32 battles |
| Multi format (AI partner on the player's side) | 1-2 | 35 battles; partner teams |
| Search, prep, line player and CLI with two actives | 2-3 | joint actions, two leads |
| A recorded doubles fight as golden trace | 1 | emulator recording |

**Decisions that are the user's.**

1. Per-fight battler count (recommended) or four slots in every fight.
2. Whether the first group (the re-index with no behaviour change) is pulled forward, after the
   current singles batches land, instead of waiting for the milestone. It is the step that stops
   new rework from accumulating; it also touches every engine file both streams are editing.
3. Whether doubles may start as Monte Carlo only, with exact search added where it fits.
4. Order and scope of the three kinds: plain doubles (34), two trainers (32), multi (35). Multi is
   the most expensive and 14 of its rows have no known partner team; that needs the user's
   knowledge of the hack or a recording.
5. What a partner's fainted Pokemon counts as in a multi battle (today only the player's deaths
   are counted).
6. Whether the backlog table should count battles instead of trainer rows.

## Appendix A: decomp detail

**Defender by move range** (`BattleSystem_Defender`, `lib.c:1600-1730`):

- Adjacent opponents (Acid, Blizzard): first living enemy in speed order (:1611-1633); all adjacent
  (Earthquake, Surf): first living battler other than the user (:1634-1650).
- Opponent side (Spikes, Stealth Rock) and Me First: a random opponent (:1661-1664;
  `BattleSystem_RandomOpponent`, `lib.c:4177-4198`, one `RandNext() & 1`, the other if it fainted).
- User, user side, field, Counter-type: the user (:1665-1669). Ally (Helping Hand): the partner in
  doubles, else the user (:1670-1675). User-or-ally: the chosen battler, the user if it fainted
  (:1676-1684).
- Random opponent (Outrage): the Follow Me user if alive, else a coin flip between two living
  enemies (:1685-1707).
- Single target: the Follow Me user if alive, else the chosen target, else a random opponent
  (:1709-1727).

**Ally-affecting mechanics.** Helping Hand: flags the partner if it is alive and has not
moved (`script.c:6297-6305`), power `x 1.5` (`lib.c:6694-6696`). Follow Me: per side, with its
user (`script.c:6261-6267`), cleared each turn (`lib.c:2251-2252`). Plus and Minus need the other
ability alive on the same side (`lib.c:6802-6809`); Flower Gift boosts its whole side in sun
(`lib.c:6931-6939`). Intimidate runs per unannounced holder in speed order (`lib.c:3795-3808`) and
lowers every living battler on the other side
(`res/battle/scripts/subscripts/subscript_intimidate.s:10-21`). Mud Sport and Water Sport read
every battler (`lib.c:6812`).

**AI in doubles.** Each AI battler runs `TrainerAI_MainDoubles` (`ai.c:352-471`): for every
other living battler, ally included, it re-initialises (four new damage-roll draws, `ai.c:234-241`,
`:369`), runs every flag script with that battler as the target, and keeps the best move with a random
tie-break (:362-428); a best score under 100 against the ally becomes -1 (:431-436);
the target is drawn among the best-scoring ones (:440-457); a user-or-ally move aimed at the
player's side and a non-Ghost Curse go to the user (:461-469). Script battler arguments include the
two partners (`ai.c:2765-2783`; partner types `ai.c:965-983`). The Tag Strategy script is
`script.s:6637-7695` (about 1,060 lines; table entry `script.s:26`). Switching: the Wonder Guard
check returns "no" in doubles (`ai.c:3294-3296`), and the switch fallback looks up the
partner battler outside tag and 2vs2 fights (`ai.c:4004-4011`). Whether one AI battler sees its partner's choice of the same
turn: not found.

## Appendix B: not verified

- Which accuracy, crit and damage draws a spread move repeats per target (`BattleSystem_SetupLoop`,
  `lib.c:3357`, not traced).
- Which battlers each target menu allows (layout tables in `battle_subscreen.c`).
- Experience sharing with two player battlers (`sideGetExpMask[2]`, `context.h:176`, not traced).
- AI item use in doubles and whether an AI battler reads its partner's same-turn choice.
- The meaning of `TAG` versus `2vs2` is read from the party lookup (`sys.c:124-133`) and the
  comments at `battle.h:29-30`, not from a traced battle; how the hack's "two-trainer" and "multi"
  rows map onto those flags is not checked.
- All performance statements; the multi battle count; the completeness of the ally-mechanics list.
- `main`'s own `data/intermediate_kaizo/trainers.json` is older than `main`'s code (it has no
  terrain field), so the counts were read from the copy in
  `.claude/worktrees/bold-mendeleev-477a50/data/`; the `format` counts are identical in both.
