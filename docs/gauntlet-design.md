# Gauntlet chaining and growth decisions: design v5

Decomp paths are relative to `external/pokeplatinum`. `bs.c` = `src/battle/battle_script.c`, `bcp.c` = `src/battle/battle_controller_player.c`. Repo cites are at 8b89ca3c. "rom:" cites read the Kaizo ROM (`tools/rom/readers/maps.py`; "scr N @X" = `scr_seq.narc` member N, byte X; "header N" = Kaizo map header N, `debug/cb_edit/d_test.narc`).

Principle:
- The repo stores game facts only: what the ROM and the decomp force or allow between fights, each with its cite.
- Every between-fight choice of the player (heal, bag, PC, team, order, swaps) and every Nuzlocke rule is a user setting. No user setting appears in any tracked file, in `trainers.json` or in the built tables.
- Each setting has a default (section 13, from the user's answers). An unset setting takes its default; the report names every value and where it came from.

## 1. Fights, gaps and groups

- A gauntlet group is a list of trainer ids (`data/kaizo/overrides.toml:482-550`).
- A fight is one battle: one trainer id, or two ids that are each other's `opponent_2` (two trainers or multi). The loader groups ids by the resolved `opponent_2` field. Wayward Cave's 12 ids (`overrides.toml:548`) are 6 fights.
- Fights are numbered 1..n per group in the order played (section 2, `orders`). Gap k sits after the k-th fight played, before the next (1 <= k <= n-1).
- `Trainer.gauntlet` becomes `(group, n)` plus the trainer's fight id (the first trainer id of its fight); both ids of a pair share it. Today `tools/kaizo/resolve.py:522-527` numbers ids, not fights; that numbering is replaced.
- Every setting that names a member names a fight by its first trainer id.
- Totals at 8b89ca3c (`docs/kaizo/mechanics_backlog.md:38`, `:90-131`): 27 battles need a gauntlet group; for 20 the gauntlet is the last blocker. The other 7 stay blocked by E2 formats: 15 Liv & Liz (double, AI_FLAG_TAG_STRATEGY) and the 6 Wayward Cave multis. Expected whole-game runnable: 230 -> 250 of 387. The 5 E4 + Champion fights (866-870) join as a new group with no new blocker and stay runnable.

## 2. Game facts per group (tracked data)

Stored in `overrides.toml` per `[[gauntlet]]` and resolved into the tables. Only these keys:
- `source`: the citation, checked by `_SOURCE` as today (`tools/kaizo/overrides.py:93-98`; `_source` `:163-170`). E4 + Champion: `source = "rom:scr 188/190/192/194/196"`. No separate cite or kind field: `check_gauntlet_members` already skips every non-sheet source (`overrides.py:630`, `_SHEET.fullmatch`).
- `forced`: a list of `{ gap = <k | "all">, <option> = <value>, ..., source = "<cite>" }`, only where the game decides. A user setting that contradicts a `forced` value is refused, with that entry's source in the message.
- `orders`: the fight orders the map allows (a list of trainer-id lists), with `orders_source`. One order means the game fixes it.
- `leave_gaps`: gaps after which the player can leave the group's map without entering an undefeated member's sight line. `escape`: whether Escape Rope and Dig work on the map. `pc_gaps`: gaps with a reachable PC. `heal_gaps`: gaps with a reachable free full heal. Each with its source.
- `rules = {}` and `RULE_KEYS` (`overrides.py:71`) are removed. `resolve.py` copies only these facts (today `:527` copies `g.rules` and `:594` writes `gauntlet_rules` into every trainer row). `trainers.json` gains `gauntlet_facts` in place of `gauntlet_rules`.
- The map facts (`orders`, `leave_gaps`, `pc_gaps`, `heal_gaps`) come from a read-only ROM tool added in GT-2 (`tools/kaizo/gauntlet_map.py`), so they can be re-derived. The values below are its first results.

### Shared facts

- F1. After every object-trainer fight the player gets field control back: common battle script scr 1114, `StartTrainerBattle` @3873, `ReleaseAll` @3900. The start menu, party menu and bag open from there.
- F2. Field item use: healing items through `UseHealingItemFromMenu` (`src/item_use_functions.c:142`, `:295`), TMs and HMs through `UseTMHMFromMenu` (`:147`, `:526`). An item's effect on a Pokemon is `Pokemon_ApplyItemEffects` (`src/item_use_pokemon.c:237-391`), driven by the item's `ITEM_PARAM_*` fields: HP restore and revive (`:158`, `:164`, `:271-275`), status cures (`:150-154`), PP restore (`:193-200`, `:326`), PP Up and PP Max (`:185`), Rare Candy level-up (`:159`), stone evolution (`:178`), EVs and friendship (`:217-224`, `:366-381`). A Revive works in the field; keeping fainted members out is a Nuzlocke setting (`run.revived_dead`).
- F3. No `HealParty` in any gauntlet map's own script file or in the E4 room and lift files (rom: scr 215, 431, 277, 73, 80, 81, 459, 318, 188-196; heal sites from a walk of every scr_seq member).
- F4. Kaizo script command 0x3F (sets a party status) runs only in headers 104, 113, 130, 148, 158, 160, 216, 257, 301, 302, 306, 321, 348, 384, 415, 440, 514 (`data/kaizo/rom_diff.toml:232`). None is a gauntlet header (199, 346, 256, 67, 73, 74, 373, 284, 175-185).
- F5. Field poison: one tick per 4 player steps (`Field_UpdatePoison`, `src/overlay005/field_control.c:895-905`). A tick takes 1 HP from each poisoned or badly poisoned member that can battle, never below 1 (`Pokemon_DoPoisonDamage`, `src/unk_02054884.c:177-212`); at 1 HP the status is cleared (`Pokemon_TrySurvivePoison`, `:215-222`) with a friendship loss (`:197-199`; `src/pokemon.c:2630`). Field poison never faints a member. The step count of a gap is not in the data.
- F6. Escape Rope and Dig work only on a cave map whose header allows escape (`CanUseEscapeRope`, `src/item_use_functions.c:941-947`; `FieldMoves_CheckDig`, `src/field_move_tasks.c:754-760`; map type and escape bits of the header's u16 at offset 22, `include/map_header.h:25-29`; Kaizo reads them from d_test.narc, `data/kaizo/rom_diff.toml:122-124`, `:142`).
- F7. Sight: a trainer of type NORMAL spots the player on its facing line within its sight range, unless a tile between them blocks (`GetTrainerDistToPlayer`, `src/trainer_encounter.c:182-229`; `IsPathInterrupted`, `:310-344`). A LOOK_* movement type turns the trainer among its listed directions.

### Map reads

Method: Kaizo ROM, per group map: terrain collision bit, object events (item balls and Rock Smash rocks treated as removable), warps (intra-map warp panels followed), each undefeated member's sight line over every direction its movement type can face (F7), Surf allowed. Not modelled: one-way ledges (none appears on the tiles read for Route 204 North and Route 213), elevation, dynamic map features. An order is allowed when each next member's sight line or a tile next to it is reachable, from the group's arrival tile or from the last opponent, without entering another undefeated member's sight line.

| Group (header) | Map-allowed orders | Leave on foot | Escape Rope / Dig (F6) | Heal or PC mid-group |
|---|---|---|---|---|
| Oreburgh Mine (199) | 195 then 196 only. From the stairs arrival (15,2) the only way to Worker Mason (2,24) passes (10,29), in Worker Colin's sight (LOOK_SOUTH, range 3, at (10,28)). The sheet order 196, 195 is not possible. | after 195: yes, stairs (15,1) -> B1F | yes: map type 3 (cave), escape bit set (rom header 199; vanilla `include/data/map_headers.h:4437`, `:4441`) | not read past leaving |
| Route 204 North (346) | 15, 13, 14 only, from the Ravaged Path arrival (180,699) | after 15 and after 13: yes, Ravaged Path warp (180,698) and the south edge | no: map type 2 | not read past leaving |
| Route 204 North B2B (256) | 296, 297, scripted | no: no field control | n/a | no |
| Eterna Gym trainers (67) | not settled: the gym trainers have trainer type NONE and start from coord events (19,13), (3,13) and talk entries; the clock layout is a dynamic map feature (`InitPersistedMapFeaturesForEternaGym`, rom: scr 73 entry 1 @46) the read does not model | not settled | no: map type 4 | no |
| Eterna Building #1 (73) | 831 then 421 only. From the 1F stairs arrival (13,4) every path to Galactic Amalthia (8,6) crosses row 9, x 13-15, in Scientist Galileo's sight (facing W, range 4, at (16,9)). The sheet order 421, 831 is not possible. | after 831: yes, 1F stairs (13,3) | no: map type 4 | not read |
| Eterna Building #2 (74) | 410, 423 only | after 410: yes, stairs (9,11) -> 2F | no | not read |
| Eterna Building #3 (74) | free: 412, 422 or 422, 412 | after 422 first: yes (warp panels, then 1F warp (23,12)); after 412 first: no | no | not read |
| Route 213 (373) | free: all 120 orders | after every fight: yes, lobby door (706,818) -> header 376. After a swimmer (106, 103, 105) the player is surfing. | no: escape bit clear | after every fight: the lobby healer, header 376 obj 1 at (11,4), scr 462 entry 2 `HealParty` @74, no payment check |
| Wayward Cave (284) | not read (multi pairs; E2-blocked) | not read | not read | partner heal after every won fight |
| Elite Four + Champion (175-185) | fixed by the game | no: doors | n/a | PC in every lift; no healer |

Tracked facts that follow:

| Group | `forced` | Other facts | Cite |
|---|---|---|---|
| Oreburgh Mine | order 195, 196 | `leave_gaps = [1]`, `escape = true` | rom header 199 ev 197 objs 5/6, stairs warp (15,1) |
| Route 204 North | order 15, 13, 14 | `leave_gaps = [1, 2]` | rom header 346 ev 332, warp (180,698) |
| Route 204 North B2B | gap 1: heal none, pc_heal off, team none, swap none, bag none, growth items none; order 296, 297 | | rom:scr 277 entry 2 `LockAll` @36, battle 296 @44, battle 297 @73, `ReleaseAll` @106 |
| Eterna Gym trainers | none | none settled | rom:scr 73 entries 5-9, each ending in `ReleaseAll` (@359, @471, @593, @702, @815) |
| Eterna Building #1 | order 831, 421 | `leave_gaps = [1]` | rom header 73 ev 72 |
| Eterna Building #2 | order 410, 423 | `leave_gaps = [1]` | rom header 74 ev 73 |
| Eterna Building #3 | none | `orders` both; `leave_gaps` after 422 first only | rom header 74 ev 73 |
| Route 213 | none | `orders` all; `leave_gaps` all; `heal_gaps` all | rom header 373 ev 359; warps (706,818), (706,813) -> header 376; scr 462 @74 |
| Wayward Cave | gap all: partner heal. After every won battle while the partner flag is set, the game heals the player's party: HP to max for every member, fainted ones included, status cleared, PP restored. | | rom:scr 318 entry 2 `SetHasPartner` @171; `src/encounter.c:197-199`; `Party_HealAllMembers`, `src/item_use_pokemon.c:603-628` |
| Elite Four + Champion | order 866, 867, 868, 869, 870 | `pc_gaps = [1, 2, 3, 4]`; no heal, tutor or relearner | below |

Elite Four facts:
- Order and doors. After a win the room script removes the exit door (`res/field/scripts/scripts_pokemon_league_aaron_room.s:24`) and returns control (`:30`). The exit warp leads to the next lift (`res/field/events/events_pokemon_league_aaron_room.json`: (8,2) -> `ELEVATOR_TO_BERTHA_ROOM`), and the lift's warp to the next room (`events_pokemon_league_elevator_to_bertha_room.json`: (4,2) -> `BERTHA_ROOM`). Entering a room adds the entrance door behind the player (`scripts_pokemon_league_aaron_room.s:62-69`). The Kaizo files have the same structure (rom: scr 188-196, init entry 2 `AddObject` @142, @180). The order is fixed and the player cannot walk back.
- Bag and party menu. Each fight ends in `ReleaseAll` (rom: scr 188/190/194 @118, scr 192 @156), so the bag and the party menu are open at every gap (F1, F2). No script in scr 188-196 or the lift files disables the start menu.
- PC. Each Kaizo lift has a tile with behaviour 0x83 `TILE_BEHAVIOR_PC` (`include/constants/field/map_tile_behaviors.h:142`): land 306 (6,11), 307 (6,11), 308 (6,19), used by headers 176, 178, 180, 182, 184 (rom scan, re-run for v5). Vanilla `res/field/maps/data/map_data_306.bin`-`308.bin` hold 0 at those tiles: Kaizo added them. Facing it north runs `CommonScript_PC` (`src/overlay005/field_control.c:654-655`; `res/field/scripts/scripts_common.s:47`).
- No healer: the rooms (land 260-264) have no PC tile; room events hold only the member and two doors; lifts have no objects; no script heals between rooms (F3). The last nurse is in header 175 (North Pokemon Center 1F), before Aaron.
- No tutor or relearner object in any room or lift (same events). TMs and HMs in the bag remain usable (F2).

## 3. User settings (local or per run, never tracked)

- Gauntlet options are entered per run: flags, or `--rules FILE` naming a file the user keeps outside the tracked tree. No gauntlet choice is stored by default (answer 1).
- Standing Nuzlocke settings may sit in `[run]` of `data/local.toml`, the existing git-ignored local inputs file (`.gitignore:15`; `docs/kaizo/pipeline.md:10`). `data/local.example.toml` gets no gauntlet or run entries; this section is the schema.
- Precedence per value: `forced` > flag > `--rules FILE` > `data/local.toml` > default (section 13). The report prints every value with its origin and says when a default was used.
- Input errors: unknown key, field or value; gap outside 1..n-1; an order not in the group's `orders`; a value that contradicts `forced`. Where the map read settles a gap: `pc_heal = on` outside `pc_gaps`, or a walk-out heal (`heal = full` outside `heal_gaps` and outside `leave_gaps`). Nothing else is rejected.

Schema; choices are listed strictest first.

```toml
[run]                          # Nuzlocke rules
deaths_scope = "chain" | "fight"
cap_overflow = "sit_out_after_fight" | "fight"
cap_sit_out_in_party = "allowed" | "forbidden"
revived_dead = "stay_dead" | "revived"
shedinja = "forbidden" | "allowed"
field_poison = "worst_case" | "ignore"

[gauntlet."<group name>"]      # per run
heal = "none" | "bag" | "full"
bag_uses = <int >= 0> | -1                                 # -1 = only what the bag holds
pc_heal = "off" | "on"
growth_items = "forbidden" | "allowed"                     # Rare Candy and stones between fights
team = "none" | "reorder" | "moves"
move_sources = ["bag"] | ["bag", "tutor", "relearner"]
swap = { bench = <int >= 0 | -1>, budget = <int >= 0 | -1> }
order = "sheet" | [<first trainer id of each fight>, ...]   # one of the group's `orders`
partner_heal = "game" | "stricter"
time = "day" | "night"

[gauntlet."<group name>".gap.<k>]                          # overrides the group value at gap k
heal = ...; pc_heal = ...; team = ...; growth_items = ...
swap_cap = <int >= 0>; bag_cap = <int >= 0>                # most spent at this gap; 0 closes it
```

Option kinds (`src/nuzline/data/gauntlet.py`, `OPTIONS` registry; loader, CLI, coverage and report iterate it):
- Level (ordered, per gap): `heal`, `pc_heal`, `team`, `growth_items`.
- Counted (whole remaining chain, spent at gaps, optional per-gap cap): `swap` (`bench`, `budget`), `bag_uses`.
- Set (whole chain): `order`, `move_sources`.
- Plain choice: the `[run]` keys, `partner_heal`, `time`.
- Adding an option: its `OPTIONS` entry and its transition in `prep/gauntlet.py`.

Meaning:
- `heal = bag`: bag medicine and PP items may be used (F2); each use is charged to `Cost` and to `bag_uses`. Ethers and Elixirs count like any other bag item. `full`: HP, PP and status reset on living members.
- `pc_heal = on`: at a gap, a member may be deposited and withdrawn: PP restored at deposit (`src/pc_boxes.c:82`; `BoxPokemon_RestorePP`, `src/pokemon.c:5109-5122`), HP to max and status cleared at withdraw (`Pokemon_FromBoxPokemon`, `src/pokemon.c:4031-4055`; `:633-645`). The withdrawn member goes to the end of the party (`Party_AddPokemon`, `src/party.c:50-59`; `src/applications/pc_boxes/box_app_manager.c:3963`). Under `team = none` the order is kept by withdrawing in party order.
- `swap`: `bench` = box members chosen before fight 1; `budget` = members swapped in over the chain; `-1` = unlimited. A swapped-in member arrives with full HP, PP and status, at the end of the party.
- `order = "sheet"`: the sheet order when it is in `orders`, else the single map order (Oreburgh Mine, Building #1); the report says which. A list picks one of `orders`.
- `partner_heal = game`: the `forced` heal applies. `stricter`: the user's `heal` applies instead, a house rule; the report says the game heals there.
- `cap_overflow = sit_out_after_fight`: a member that passes the next fight's level cap (`config/level_caps.toml`, `data/kaizo/level_caps.toml`) finishes the current fight, then takes no part in any later fight of the chain. `cap_sit_out_in_party = allowed`: it may stay in the party unused when it cannot be deposited. A fight where only sit-out members can still battle counts as lost.
- `field_poison = worst_case`: at every gap each poisoned living member drops to 1 HP and is cured, with the friendship loss (F5). `ignore`: no change.
- `time`: day and night evolutions are offered only with `time` set. Not read for the E4 + Champion group (answer 17).
- CLI: `nuzline prep|coverage --rule heal=bag --rule swap=bench:1,budget:1 --gap-rule 2:heal=full --rules FILE`; `--trainer-id` of any member selects the group and fight; `--beaten <ids>` with a chosen order; `--deaths` per `deaths_scope`; `--swaps-left N` and `--bench <box ids>` for a re-run mid-gauntlet (user input; the save does not hold them); `--horizon N`; `--growth-cap N`; `--time`; `--next <trainer-id>`.

## 4. Run-level scope: per boot

`BootState` sits outside every per-gauntlet field and holds values fixed for one boot of the console.

- `boot_weather_ball_byte`: the AI's untyped Weather Ball byte, the low byte of a heap pointer set by the per-boot heap shift; a multiple of 4 in 0..252 (64 values). E1 owns its rule and its decomp cite; this design states no rule for it.
- Scope: one value per boot. It stays fixed across every battle, gauntlet fight, overworld step, save and load until a soft reset or power cycle, and it is never restored from a save. Only a reset or boot event in the run plan replaces it. No battle, heal, PC, warp or team change touches it.
- Planning holds it hidden: a `uint64` mask of the possible values (bit v/4 set = value v possible), so an "observed AI behaviour narrows it" update can clear bits. A played or sampled battle takes one concrete `int8` value, written into the battle state at fight start as a read-only batch input in both engines, like the other run inputs.
- Scoring: a line is scored over every value that changes the AI's choice; the uniform prior only breaks ties. The search details are E1's.

## 5. Prep search

**Carry** (end of fight k = start of the next): section 7 per party-0 slot; party order; bag counts; dead members (out of the pool); deaths used; per Counted option the amount left; the bench; the swap plan; the growth plan (section 6); sit-out members; the fights beaten. The boot scope (section 4) rides beside it, unchanged by every gap.

**Start of the next fight** (`prep/states.make_states` from a Carry): the carry is written into party 0 of the fight's layout; the opponent and any partner party are built from trainer data (section 7). When the opponent is a Gym Leader, Elite Four or Champion class, every member then gains the battle-start friendship (+3/+2/+1 by band, Luxury Ball and met-location +1: `src/battle/battle_main.c:1360-1372`; `src/pokemon.c:2626`, `:2662-2668`). It runs before the battlers are built, so it changes Return and Frustration in that fight; it applies to single Gym fights too. Not in the engine today.

**Transition at gap k**, in order:
0. Learn prompts left at a battle-ending KO (G1) and evolution prompts (G6), answered by the growth plan. They run before field control returns (`src/battle/battle_main.c:256-297`).
1. `forced` values replace the settings for this gap.
2. Wayward partner heal (`forced`, unless `partner_heal = stricter`): HP to max on every member, status cleared, PP restored (`item_use_pokemon.c:603-628`). Members dead under `revived_dead = stay_dead` stay out of the pool.
3. `heal = full`: HP, PP and status reset on living members.
4. `pc_heal = on`: members the plan deposits and withdraws get full HP, PP and status; party order follows.
5. `heal = bag`: medicine and PP items (F2) per the plan, each charged to `Cost` and `bag_uses`.
6. `growth_items = allowed`: Rare Candies and stones per the growth plan (G7).
7. `team >= reorder`: party order and held items. `team = moves`: also TMs and HMs from the bag (+1 friendship per TM or HM, `FRIENDSHIP_EVENT_LEARN_TMHM`, `src/pokemon.c:2627`), and tutor or relearner moves when `move_sources` includes them.
8. `swap`: the plan's swap at gap k runs if budget and cap allow.
9. `field_poison` (F5).

**Search: receding horizon with the chain objective.**
- A setup's score is P(win every remaining fight with deaths within `--deaths` per `deaths_scope`). A chain playout runs the fight, applies the transition and continues to the last fight. Funnel, ladder and Stage B use this score.
- The answer commits only to the current fight: team, lead, variants, bag use, the swap at the gap before it, the next opponent (where `orders` allows a choice), the fight's learn answers (Stage B learn nodes) and the evolution answers right after it. At the start it also commits the bench. After each fight the user saves and re-runs prep from that save.
- Where control never returns (296/297), the Stage B tree spans the boundary through the transition.
- No lookahead cap by default; `--horizon N` caps it and the report says `lookahead capped at N of M remaining fights`.
- The report prints the chain cost per gauntlet: wall time and playouts for screening and Stage B, beside the single-fight figures.

**Swap timing.** A plan is a list of (gap, out-member, in-member), at most `budget` long, part of the setup at the root and of the carry in playouts.
- Bench candidates: the 3 box members with the best single-fight funnel screen against any remaining fight, per bench slot.
- Full plans are enumerated when they number at most `MAX_SWAP_PLANS = 64` per (setup, bench). E4 start, bench 1, budget 1: 1 + 4 gaps x 6 out x 1 in = 25.
- Above the cap the root enumerates only the current gap's swap; later swaps follow the lookahead policy.

**Chosen order** (a group whose `orders` holds more than one order and whose `order` setting leaves a choice).
- The next opponent is a root choice among the fights the allowed orders put next. The funnel screens each setup once per candidate: x r for r candidates. Stage B runs on the best (setup, next opponent) pairs.
- Lookahead order: all allowed remaining orders are screened when they number at most `MAX_ORDERS = 24`; above that, the next opponent at each later gap is chosen greedily by single-fight screen, and the report says so.

**Screening order.** The funnel screens setups with no future swap, the base growth plan and the fixed or greedy order. The ladder's survivors are screened against every (bench, swap plan, growth plan, order). Stage B runs on the best tuples.

**Bound per setup on the ladder:** growth plans (at most `--growth-cap`, default 64) x swap plans (at most 64; 25 at the E4) x bench choices (3 per bench slot) x next-opponent candidates r x screened lookahead orders (at most 24). E4 start: 64 x 25 x 3 = 4,800 (order fixed, r = 1). Route 213 at its start, order chosen: x 5 x 24.

**Lookahead policy** at later gaps:
- Planned swap: runs at its gap even if a member died earlier.
- Policy swap (above the cap): at a gap where a death or a sit-out left a slot and budget remains, bring in the bench member with the best precomputed screen score against the next fight.
- `heal = bag`: top every living member to full HP and PP and cure status with the smallest items that do it, oldest gap first, until `bag_uses` or the bag runs out.
- `pc_heal = on`: every living member is deposited and withdrawn in party order.
- `team >= reorder`: lead with the next living, not sat-out member in setup order. `team = moves`: no TM or tutor change.
- Growth: every prompt is answered by the committed growth plan.
- Level cap: per `cap_overflow` and `cap_sit_out_in_party` (section 3).
- The planned swap, bag and reorder policies under-estimate the chain; the growth plan is followable, so its score is achievable.

**Rejected:** fight by fight (trades the next fight's win chance for this one's); one joint tree over all fights (size is the product of per-fight setup spaces, up to `MAX_TEAM_SETUPS = 25_000` each, `src/nuzline/prep/setups.py:58`); swap timing, order or growth as a search choice at every playout gap.

**Report line:** `Gauntlet <name>, fights i-n; settings: heal=bag (flag), swap bench 1 budget 1 (flag), order 195,196 (forced: <source>), field_poison worst_case (default)`, then the next opponent, the bench, the swap plan, the growth answers for this fight and the gap after it, and the planned between-fight steps. The ladder states that it covers the whole remaining chain.

## 6. Growth decisions

Level-ups, learn prompts and evolutions are decisions of the line, in single battles and in gauntlets. None is declined by default.

### Decomp facts

- **G1 When EXP is paid.** A faint sets the payout flag (`bs.c:1800`). `BattleControllerPlayer_AnyExpPayout` (`bcp.c:4449-4469`) runs `subscript_grant_exp.s` (lines 5-7). Callers: `MoveEnd` (`bcp.c:3968`), `CheckFieldConditions` (`:941`), `CheckMonConditions` (`:1296`), `TurnEnd` (`:1846`), each before `CheckBattleOver`; `TurnEnd` pays before `ReplaceFainted` (`:1848`); Pursuit pays in its script (`subscript_pursuit.s:108`). A KO that ends the battle also pays, and its prompts are answered.
- **G2 Who gains.** Only party 0, the player's: the payout loops over `BATTLER_US`'s party (`bs.c:2200-2202`) and the task sets `expBattler = 0` (`bs.c:9701-9704`). Members on the field while the foe was (`sideGetExpMask`, `battle_lib.c:1461-1470`) and Exp. Share holders (`bs.c:9709`, `:9745-9746`). A fainted or level-100 member gains nothing (`bs.c:9740`). Trainer x1.5 (`bs.c:9753-9754`), Lucky Egg x1.5 (`:9749-9750`), traded x1.5 or x1.7 (`:9757-9762`). Base yield x level / 7, split by participants (`bs.c:2216-2239`).
- **G3 Level-up in battle.** Per level: friendship (`bs.c:9857`), `Pokemon_CalcStats` (`bs.c:9858`); current HP rises by the max-HP gain (`src/pokemon.c:633-645`). The active learner is reloaded (`bs.c:9860-9865`; `battle_lib.c:173-181`): stats, level, friendship, HP; stat stages untouched. The stat copy runs even under Transform; moves, PP and EXP reload only when not transformed and never into a Mimic slot (`battle_lib.c:183-193`). A benched member levels on its party copy only.
- **G4 Effect on the turn.** New stats are read at once. Turn order is computed once per turn (`bcp.c:698`, `:707`), so a Speed gain counts from the next turn. A selected move is re-read from its slot when it runs (`bcp.c:1889-1895`); a new move is selectable from the next turn.
- **G5 Learn prompt.** Each level gained is scanned (`bs.c:10013`, `:9824-9836`). `Pokemon_LevelUpMove` offers each learnset entry of exactly the current level (`src/pokemon.c:3935-3966`). An empty slot learns with no prompt (`src/pokemon.c:3846`; `bs.c:10021-10035`). With four moves the player replaces a slot or gives up (`bs.c:10069-10135`). An HM cannot be forgotten (`src/battle_sub_menus/battle_party.c:718`, `:1724-1735`). The new move gets max PP and PP Ups 0 (`bs.c:10169-10171`; `src/pokemon.c:3925-3932`). The active learner is reloaded (`bs.c:10173-10177`).
- **G6 Evolution after a battle.** Only on a win, capture or flee (`src/battle/battle_system.c:829-833`); only members that levelled (`leveledUpMonsMask`, `bs.c:9867` -> `battle_system.c:835-849`). HP is not checked: a member that levelled and then fainted is offered too (`battle_system.c:826-850`). It runs before field control returns (`battle_main.c:261`, `:274-297`). B cancels (flag 0x1, `battle_main.c:283`; `src/evolution.c:283-286`); a cancelled member is offered again only after its next level-up.
- **G7 Evolution between fights.** A Rare Candy level-up runs the same check (`src/applications/party_menu/callbacks.c:816-829`) and can be cancelled (`src/start_menu.c:1081-1090`, `:1740` flag 0x1). Kaizo's change there only keeps the menu in Rare Candy mode (`data/kaizo/rom_diff.toml:195-197`). A stone evolution cannot be cancelled (`src/start_menu.c:1069-1080`, `:1742`); the decision is whether to use the stone. Trade evolutions (`EVO_CLASS_BY_TRADE`, `src/pokemon.c:3727`) are not offered.
- **G8 Conditions** (`Pokemon_GetEvolutionTargetSpecies`, `src/pokemon.c:3528-3725`; first match wins, `:3721-3723`). Everstone blocks every level evolution (`:3543-3547`). Methods: level; friendship >= 220 (`include/constants/pokemon.h:25`), raised at the level-up before the check (`bs.c:9857`); day or night (`IsNight`, `src/rtc.c:134`); held item by day or night; known move; species in party; gender; Attack vs Defense; personality; map (`mapEvolutionMethod`, `src/field_battle_data_transfer.c:264`). Data: `Learnsets.evolutions` (`src/nuzline/data/learnsets.py:98`).
- **G9 On evolution.** Species set, ability from the new species and personality bit 0 (`src/evolution.c:435-436`; `src/pokemon.c:4746-4763`), stats recalculated with HP raised by the max-HP gain (`src/evolution.c:437`; `src/pokemon.c:633-645`). The new species' entries of exactly the current level are offered (`src/evolution.c:461-480`; `src/pokemon.c:3950`). A Nincada evolution adds a Shedinja when the party has a free slot and the bag a Poke Ball, which it uses up (`src/evolution.c:723-740`).
- **G10 Nincada in Kaizo.** The Kaizo encounter data holds Nincada (290) only in Eterna Forest's Poke Radar slots: `fielddata/encountdata/pl_enc_data.narc` member 8 (`res/field/encounters/encounters.order` line 9, Eterna Forest), bytes 124-136 = `radarEncounters[0..3]` (`include/overlay006/wild_encounters.h:31-36`; read at `src/overlay006/wild_encounters.c:696-699`). It is obtainable, so the Shedinja case stays. Gift, trade, honey tree and egg sources were not scanned.

### Decisions

- **In-fight learn answer** (G1, G5): replace slot 0-3 or give up. Stage B: the existing learn decision (`search/stage_b.py`; paused prompt `learn_site`, `engine/state.py:625`). Playouts: the growth plan.
- **Between-fight learn answer** (G5 after Rare Candies, G9 after evolution): same choices, from the growth plan.
- **Evolve or cancel** (G6, G7): per eligible living member per prompt. A dead member's answer is cancel and changes nothing.
- **Shedinja** (G9, G10): added only with `shedinja = allowed`, a free slot and a Poke Ball; otherwise the plan does not count it.
- **Pre-fight growth** (single battles and the first fight): Rare Candy levels to the cap (`prep/leveling.py`) with every prompt answered; stone use where the bag holds the stone.
- **Growth plan** (root, per member): one answer per offered move and an evolution gap (eligible gap or never). It holds in every playout.
- **Base plan** (funnel): learn into an empty slot, else give up; never evolve. This is today's playout behaviour (`search/learn.py:29-33`, `GIVE_UP`).
- **Time of day:** day and night evolutions are offered only with `time` set; not read at the E4 + Champion.
- **Single battle, post-fight evolution.** Evolution runs after the battle ends (G6), so it never changes the fight just planned. With `--next <trainer-id>`, both answers are scored by the single-fight screen against that trainer and the better is reported. Without `--next` the report recommends cancel, because a cancelled evolution is offered again at the next level-up (G6, G7). The report always says an evolution cannot be undone.

### Bound

- In-fight prompt: up to 5 answers; already a Stage B max node.
- Learn plan per member: 5^m for m offered moves before the last cap. Evolution per member: G_e + 1 options, x5 per post-evolution offered move. Per setup: the product over members.
1. **Commit per fight.** Only the current fight's prompts and the evolution prompts after it are committed. The current fight's prompts stay exact Stage B nodes and are never capped.
2. **Never-better pruning (exact), before the cap.** Give up only, for an offered move whose effect is unsupported (masked, D10) or that has the same effect id, type and category as a known move with power, accuracy and PP no higher. Evolve dominates cancel when the evolved species keeps types and ability, no base stat falls, and no learnset entry before the last cap is lost; then only the earliest eligible gap.
3. **Cap.** When the non-dominated plans per setup number at most `--growth-cap` (default `MAX_GROWTH_PLANS = 64`, one constant; `-1` = none), all are screened. Above it, the root enumerates only the current fight's prompts and the evolution after it; each later prompt takes the member's best answer by single-fight screen. The report says plainly that options were removed, gives `<kept> of <total>`, and names `--growth-cap -1`.

### Owners and status

Owners: E1 = abilities, items, conditions, AI flags; E2 = doubles and state shape; E3 = move effects, gauntlet and prep.

| Piece | Status at 8b89ca3c | Files | Owner |
|---|---|---|---|
| EXP payout, EVs, level-up stats and HP, friendship in battle | modelled | `engine/levelup.py` (`pay_exp` :145, `pay_exp_leveled` :151) | E3 consumer; exp items E1 |
| Learn prompt in battle | modelled | `engine/levelup.py` (`needs_learn` :292, `advance_learns` :400, `learn_options` :430, `learn` :465); `engine/state.py:614-615`, `:625` | E2 for the fields |
| Mid-turn learn prompt at its payout | modelled (E3-B2-21) | `search/stage_b.py`, `engine/state.py:625` | E3 |
| Learn answer in playouts | always give up | `search/learn.py:29-33` | E3 (GT-16) |
| Pre-fight Rare Candy levels | modelled; never evolved | `prep/leveling.py:21` | E3 |
| Evolution check, conditions, Everstone, apply, post-evolution learn, Shedinja | missing; coverage says never modelled (`engine/coverage.py:1115`) | `prep/growth.py` (new) | E3; tell E1 (Everstone, evolution held items) |
| Battle-start friendship for Gym, E4, Champion | missing | `prep/states.py` | E3 |
| Field item effects (medicine, PP items, Rare Candy, stones) | missing; `Resources` holds Heart Scales, TMs, HMs and holdable items only (`prep/problem.py:51-58`, `:86-97`) | `tools/build_tables.py`, `prep/problem.py`, `prep/gauntlet.py` | E3 |

## 7. Carried-field list (for E2 sign-off)

Shape: in the engine, P is the party axis and B the battler axis (`engine/state.py:392`, `:866-871`; `docs/doubles-state-design.md` section 4). Only party 0 carries: experience goes only to party 0 (G2), and every tag fight rebuilds the partner's party from trainer data (`src/encounter.c:753` sets `trainerIDs[BATTLER_PLAYER_2]`; `Trainer_Encounter` builds a party per set id, `src/trainer_data.c:25-47`). `Party_HealAllMembers` heals only the save party (`src/encounter.c:198`). So `Carry` holds one entry per party-0 slot: `[6]` per field (`[6, 4]` for moves), plus a leading batch axis named `R` when vmapped. At each fight's start the carry is written into party 0 of that fight's layout.

The battle reaches the party only through `BtlIOCmd_UpdatePartyMon` (`src/battle/battle_io_command.c:575-608`, from `BattleController_EmitUpdatePartyMon`, `src/battle/battle_controller.c:1462-1505`), plus the EXP task and the level-up path (`bs.c:9767-9779`, `:9857-9867`, `:10007`). At exit the battle party goes to the save (`battle_main.c:703-705`; `src/encounter.c:151-158`).

`carry_out(terminal state) -> Carry` reads, per party-0 slot s. A battler is "on s" when it is a party-0 battler with `active[b] == s`; the read loops over battlers, never "battler 0".

| Field | Source in the terminal state | Rule | Decomp cite |
|---|---|---|---|
| species | fight-start carry | never written by the battle | `battle_io_command.c:581-604` (fields written) |
| form | fight-start carry, except the written changes: Shaymin Sky -> Land, Giratina Origin -> Altered (`state.form`); Burmy by terrain if it took part | Castform, Cherrim, Arceus revert | `battle_io_command.c:597-599`; `battle_controller.c:1489-1499`; `battle_lib.c:6305-6400`; `battle_system.c:926-965` |
| ability | `party_ability` where set (`state.py:422`), else from species and personality | battle changes (Skill Swap, Worry Seed, Gastro Acid) revert | `battle_io_command.c:601-604` |
| level, exp, evs | `state.level[0, s]`, `state.exp[s]`, `state.evs[0, s]` | persist | `bs.c:9767-9779`, `:9857-9867` |
| stats, max_hp | recomputed from species, level, EVs, IVs, nature | not copied from battlers (Transform holds the target's) | `src/pokemon.c:633-645` |
| hp | `state.hp[0, s]` | persists; 0 when fainted | `battle_controller.c:1470`; `battle_io_command.c:594` |
| status | `state.status[0, s]`; 0 when fainted | persists | `battle_controller.c:1482-1486` |
| status_counter | sleep turns kept; Toxic counter 0 (Toxic bit kept) | | `battle_controller.c:1482`; restart at `bcp.c:1411-1415` |
| moves, pp | `state.moves`, `state.pp`; while a battler on s is `transformed`, all four from `shadow_moves` / `shadow_pp`; Mimic slots (`mimic_slots`) from the shadows | Transform skips the whole move block; Mimic slots are skipped | `battle_io_command.c:581-588`; `state.py:534-542` |
| pp_ups | `state.pp_ups[0, s]` | never written by the battle, except that a move learned in battle sets its slot to 0 | `battle_io_command.c:581-604` (no PP Ups write); `bs.c:10170` |
| max PP | not carried | derived from move and PP Ups, not stored | `MoveTable_CalcMaxPP`, `src/pokemon.c:961-965`; the setter ignores it, `:1519-1523` |
| held item | `party_held_item(state, 0, s)` (E2 D3-8b) | the party's item after the battle | `battle_io_command.c:590-592` |
| friendship | `party_friendship[0, s]` (`state.py:417`) | persists; the party value wraps past 255 | `bs.c:9857`; faint `bs.c:12080-12094`; `src/pokemon.c:1859-1874` |
| party order | fight-start order | in-battle switches never reorder the party: `partyOrder` is a BattleContext index map (`battle_lib.c:6465-6507`), and the party is copied back slot by slot | `battle_main.c:703-705` |
| dead | `state.deaths[s]` | out of the pool per `revived_dead` | |
| levelled | level at end > level at start | the `leveledUpMonsMask` bit | `bs.c:9867` |
| pending learn work | `learn_level`, `learn_cursor` | answered by the growth plan before the carry (G1) | `bs.c:10013` |
| personality, ivs, nature, gender | constant, from the save | | |
| pokerus | from the save | the end-of-fight spread is random and only adds EVs; not modelled (under-estimates) | `bcp.c:4041-4048` |

Not carried (reset at every fight start): volatiles, stat stages, types, battle forms above, abilities changed in battle, the Toxic counter. The boot scope (section 4) is not party data and is not in `Carry`.

## 8. Coverage and backlog

- The gauntlet refusal (`src/nuzline/engine/coverage.py:991-998`, `_check_kaizo_fields`) becomes a note line: fight i of n, the `forced` values with sources, the map facts.
- `Support.NONE` remains only for a stored `forced` that the engine cannot apply. Settings are never a coverage row.
- A fight is supported only when every remaining fight of its chain is. A later fight's refusal is listed as `chain: <fight>: <reason>` under that fight's own blocker group, so it is not counted twice.
- The evolution line (`coverage.py:1115`) is replaced when evolution lands.
- Backlog (`tools/kaizo/backlog.py`, regenerated): the 33 per-position gauntlet rows go to 0; 20 battles leave; 7 stay under their E2 format blockers; the 5 E4 + Champion fights enter as members with no new blocker.

## 9. State and doubles-readiness

- `BattleState` does not change except the boot input `boot_weather_ball_byte` (section 4), which E1 adds with its rule. The Carry and the transition are prep-side.
- `docs/doubles-state-design.md` section 5:
  1. No mechanic computes `1 - side`: the transition acts on party 0 by name; opponents come from trainer data.
  2. Battler and party data stay apart: the Carry is party data `[6]`; battler fields (shadows, `transformed`, `mimic_slots`) are read through the battlers on each slot (section 7).
  3. No Carry field names another Pokemon.
  4. "Every battler on slot s" is a loop over party-0 battlers, so a doubles fight with two actives reads both.
  5-7. No turn-order, move-range or AI field.
- Doubles and multi fights of a chain run when E2's formats land; the Carry already fits them (party 0 only).

## 10. Build plan (one batch, 18 groups, family: gauntlet chaining plus growth)

| # | Group | Sections | Needed by |
|---|---|---|---|
| GT-1 | `data/gauntlet.py`: `OPTIONS`, kinds, per-gap values and caps, precedence, defaults; `GauntletFacts` replaces `Trainer.gauntlet_rules` | 1, 3 | all |
| GT-2 | Group facts: fight numbering from `opponent_2`, `forced`, `orders`, `leave_gaps`, `escape`, `pc_gaps`, `heal_gaps`, sources; `tools/kaizo/gauntlet_map.py` (the map read); loader checks; E4 + Champion group; `rules`/`RULE_KEYS` removed; table field; vanilla and Kaizo rebuild | 1, 2 | all |
| GT-3 | Settings: flags, `--rules FILE`, `data/local.toml [run]`, errors with the `forced` source, report of origins | 3 | all |
| GT-4 | `carry_out` per section 7, calling `party_held_item` | 7 | all |
| GT-5 | Fight start from a Carry; Gym/E4/Champion battle-start friendship; level-cap sit-out | 3, 5, 7 | all |
| GT-6 | Gap transition steps 0-9; item field-effect table from the ROM item data (`ITEM_PARAM_*`, F2); `Resources` gains medicine, PP items, Rare Candies, stones; field poison | 5 | all |
| GT-7 | Swaps: bench, budget, plans (`MAX_SWAP_PLANS = 64`), deposit and withdraw effects, order change, policy swap | 3, 5 | user swaps |
| GT-8 | Chosen order: next opponent as a root choice within `orders`, `MAX_ORDERS = 24`, greedy beyond | 5 | Route 213, Building #3 |
| GT-9 | Chain search: chain playout, chain score, funnel / ladder / Stage B on it, Stage B across 296/297, `--horizon` | 5 | all |
| GT-10 | Lookahead policy (bag top-up incl. PP, PC, reorder lead, planned and policy swaps, cap sit-out) | 5 | all |
| GT-11 | Report and record: settings with origin, forced sources, chain cost | 3, 5 | all |
| GT-12 | Evolution conditions (G8), Everstone, `time` | 6 | chains, singles |
| GT-13 | Evolution applied (G9), post-evolution learn, Shedinja per setting | 6 | same |
| GT-14 | Evolution prompts: after a won fight (G6, dead members included), Rare Candy (cancellable), stones; pre-fight evolution in `prep/leveling.py` | 6 | same |
| GT-15 | Growth plan, never-better pruning, `--growth-cap` with the plain report line | 6 | same |
| GT-16 | Playout learn answer from the growth plan (`search/learn.py:29-33`) | 6 | same |
| GT-17 | Single battle post-fight evolution: `--next`, recommend cancel without it, "cannot be undone" | 6 | singles |
| GT-18 | Coverage and backlog: note line, `chain:` lines, forced refusal, evolution line, regenerate (33 -> 0) | 8 | all |

Owners named by the 18 Nov rulings: playout learn answer GT-16 = E3; evolution GT-12..14 = E3 (tell E1 about Everstone and evolution held items, read only by `src/pokemon.c:3543-3547`); species and ability in the carry GT-4 = E3, field list signed off by E2 (section 7). `boot_weather_ball_byte` in the battle state: E1.

## 11. Files touched

| File | Change |
|---|---|
| `docs/gauntlet-design.md` | this design |
| `src/nuzline/data/gauntlet.py` | new: GT-1, GT-3 loader, `BootState` |
| `src/nuzline/data/trainer.py:254-255`, `:347-348` | `gauntlet` = (group, n, fight id); `gauntlet_rules` -> `gauntlet_facts` |
| `tools/kaizo/overrides.py` | `RULE_KEYS` :71 removed; `Gauntlet` :137-141 gains the fact keys; `_gauntlets` :224-; drafting |
| `tools/kaizo/resolve.py:522-527`, `:592-596` | fight numbering; copy facts only |
| `tools/kaizo/gauntlet_map.py` | new: the read-only map read (orders, leave, PC, heal) |
| `data/kaizo/overrides.toml:478-550` | fact keys with sources; `rules` removed; E4 + Champion group |
| `tools/build_tables.py` | trainer field; item field-effect table; MANIFEST_VERSION bump; vanilla and Kaizo rebuild |
| `src/nuzline/engine/coverage.py` | `_check_kaizo_fields` :991-998; evolution line :1115 |
| `tools/kaizo/backlog.py`, `docs/kaizo/mechanics_backlog.md` | `chain:` lines; regenerate |
| `src/nuzline/prep/gauntlet.py` | new: Carry, `carry_out`, transition, swaps, order, chain playout |
| `src/nuzline/prep/growth.py` | new: evolution, growth plans, pruning |
| `src/nuzline/prep/leveling.py:21` | pre-fight evolution |
| `src/nuzline/prep/states.py:38` `make_states` | start from a Carry; battle-start friendship |
| `src/nuzline/prep/problem.py:51-58`, `:398` | `Resources` item kinds; `build_problem` settings, `bag_uses`, bench pool |
| `src/nuzline/prep/setups.py`, `screen.py`, `search.py` (`PrepSettings` :153, `_Funnel` :325, `run_prep` :901) | chain score; setup gains bench, swap plan, growth plan, next opponent |
| `src/nuzline/prep/report.py` (`render` :361, `record` :449) | GT-11, GT-15, GT-17 lines |
| `src/nuzline/prep/caps.py` | next fight's cap |
| `src/nuzline/search/learn.py:29-33` | `learn_policy` reads the growth plan |
| `src/nuzline/search/playout.py`, `search/stage_b.py` | chain playout hook; Stage B across 296/297 |
| `src/nuzline/cli.py` | GT-3 flags, `--growth-cap`, `--time`, `--next` |
| `docs/kaizo/e3-ledger.md` | rulings |
| tests, new | `tests/data/test_gauntlet_settings.py`, `tests/prep/test_gauntlet*.py`, `tests/prep/test_growth.py`, `tests/kaizo/test_gauntlet_map.py`; synthetic groups and settings only, no test pins a user choice |
| tests changed by the schema | `tests/data/test_trainer.py:207-209`, `:302-303`, `:319`; `tests/engine/test_coverage.py:373-392`; `tests/kaizo/test_overrides.py:51`, `:124`, `:131-136`; `tests/kaizo/test_resolve.py:583-585` |

No change to the JAX step or the reference engine beyond E1's boot input. `levelup.py` is read, not edited.

Tests per run: the four guards, `tests/test_differential.py`, the prep, growth and map tests, and the coverage, overrides, resolve and trainer tests above. The tables change, so the rebuild runs first and the `fullset` tests run on the pod.

## 12. Overlaps and merge order

| With | Overlap | Resolution |
|---|---|---|
| E2 D3-8b (7f391925 on `e2-d3`) | `party_held_item(state, party, slot)`: JAX `engine/state.py:1017`, reference `reference/engine.py:2145`; new `knocked_item` field | `carry_out` calls it directly. E3 batch 3 code starts after D3-8b is on main. Any open ruling on items after Trick, Switcheroo, Thief, Covet or Knock Off is E2's inside D3-8b; this design reads the helper and depends on no outcome. |
| E2 D3-8 (`pay_exp_leveled(battler=None)`, `step._pay_all`) | EXP, level and levelled slots read by the carry | read only; no edit |
| E2 coverage (`FOUR_TURN_CLAUSES`) | `coverage.py` | different functions; text merge |
| E2 D11 (search, prep and CLI with two actives) | prep and search files of this batch | D11 starts after E3 batch 3 merges and extends it |
| E1 (`boot_weather_ball_byte`) | the boot scope (section 4) | E1 owns the rule and the state field; this batch carries the mask and passes the value |
| E1 batch H (Exp. Share, Lucky Egg, Soothe Bell) | holders gain EXP and friendship; benched Exp. Share holders level and become evolution-eligible | no shared file; any order; E3 tests of item holders evolving go after H, or use plain participants |
| E1 (Everstone, evolution held items) | read only by evolution | built in `prep/growth.py`; tell E1 |
| E1 (party friendship wrap) | carry keeps the party value | `carry_out` reads `party_friendship` |

Merge order: E2 D3-8b, then E3 batch 3, then E2 D11. E1 batch H in any order.

## 13. User answers (defaults)

Every answer is a default; each stays a setting (section 3).

| # | Question | Default | Setting |
|---|---|---|---|
| 1 | E4 choices: saved or per run | entered per run; nothing stored | flags or `--rules FILE` |
| 2 | Rules not set | strictest choice of each option, and the report says so | all gauntlet options |
| 3 | E4 bag-healing limit | only what the player carries | `bag_uses = -1` |
| 4 | Ethers and Elixirs | bag items like any other; no use limit | `bag_uses` |
| 5 | Walk-out groups | the map read (section 2) where settled; else strictest | `heal`, `pc_heal`; checked against `leave_gaps`, `heal_gaps`, `pc_gaps` |
| 6 | Route 213 healer | the lobby healer is reachable after every fight (section 2); the heal still follows `heal`, whose default is strictest | `heal` for Route 213 |
| 7 | Wayward Cave heal | as in the game | `partner_heal = game` |
| 8 | Revived fainted members | stay dead | `run.revived_dead = stay_dead` |
| 9 | Fight order | the sheet order where the map allows it, else the single map order | `order` |
| 10 | Rare Candies and stones with healing off | no | `growth_items = forbidden` |
| 11 | Move changes between fights | none | `team = none` |
| 12 | Over the level cap mid-gauntlet | finishes the current fight, then takes no part in later fights | `run.cap_overflow = sit_out_after_fight` |
| 13 | Over the cap and cannot be deposited | may stay in the party unused | `run.cap_sit_out_in_party = allowed` |
| 14 | Deaths allowance | per-run setting; default strictest (whole chain) | `run.deaths_scope = chain` |
| 15 | Poison between fights | worst case, down to 1 HP (F5) | `run.field_poison = worst_case` |
| 16 | Shedinja | Nincada is obtainable in Kaizo (G10), so the case stays; default strictest | `run.shedinja = forbidden` |
| 17 | Time of day | not read at the E4 + Champion; an input for earlier groups | `time` |

Still open (game facts, not user questions):
- Eterna Gym trainers: order and walk-out depend on the clock map feature, which the read does not model.
- Wayward Cave: order and walk-out not read (multi pairs; the group stays E2-blocked).
- Oreburgh Mine and Building #1: the map read gives the reverse of the sheet order. The sheet ordering is to be checked against the ROM read before the facts are committed in GT-2.
