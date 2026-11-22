# Gauntlet chaining and growth decisions: design v5.4

Decomp paths are relative to `external/pokeplatinum`. `bs.c` = `src/battle/battle_script.c`, `bcp.c` = `src/battle/battle_controller_player.c`. Repo cites are at 8b89ca3c unless marked. "rom:" cites read the Kaizo ROM (`tools/rom/readers/maps.py`; "scr N @X" = `scr_seq.narc` member N, byte X; "header N" = Kaizo map header N, `debug/cb_edit/d_test.narc`).

Principle:
- The repo stores game facts only: what the ROM and the decomp force or allow between fights, each with its cite.
- Every between-fight choice of the player (heal, bag, PC, team, order, swaps) and every Nuzlocke rule is a user setting. No user setting appears in any tracked file, in `trainers.json` or in the built tables.
- An unset setting takes its code default: the strictest value. For a choice the game leaves open (`order`) the strictest value is `worst`: the score is the minimum over every open choice (section 13). No default is a user's answer. The report names every value and where it came from.

## 1. Fights, gaps and groups

- A gauntlet group is a list of trainer ids (`data/kaizo/overrides.toml:482-550`).
- A fight is one battle: one trainer id, or two ids fought together (two-trainer double or multi). The fights of a group are the entries of its `orders` fact (section 2): each order is a list of fights, each fight a list of one or two trainer ids. Wayward Cave's 12 ids are 6 fights, or 7 in the split order (Wayward Cave facts).
- The loader checks every two-id fight against the resolved `opponent_2` pairs after `trainer_record` (`tools/kaizo/resolve.py:571-583`), so numbering never needs `opponent_2` at context time (`:518-528`). A pair may appear as two one-id fights only where an `orders` entry with its source says so.
- Fights are numbered 1..n along the resolved order; n is per order. Gap k sits after the k-th fight played (1 <= k <= n-1).
- `Trainer.gauntlet` stays a pair: `(group, fight_id)`, where `fight_id` is the first trainer id of the trainer's pair as the sheet lists it (both ids of a pair share it). Positions and n come from `gauntlet_facts` per resolved order. Today `tools/kaizo/resolve.py:522-527` writes a position; that is replaced. Readers: `src/nuzline/cli.py:1637-1639` prints `fight <id>` in place of `#<position>`; `tests/data/test_trainer.py:391-394` and `tests/test_cli.py:977` keep the pair shape.
- Every setting that names a member names a fight by its first trainer id.
- Totals at 8b89ca3c (`docs/kaizo/mechanics_backlog.md:38`, `:90-131`): 27 battles need a gauntlet group; for 20 the gauntlet is the last blocker. The other 7 stay blocked by E2 formats: 15 Liv & Liz (double, AI_FLAG_TAG_STRATEGY) and the 6 Wayward Cave multis. Expected whole-game runnable: 230 -> 250 of 387. The E4 + Champion group (866-870) is not in batch 3: batch 4 adds it with the chain (B4-15, B4-19), so the five fights stay runnable as single fights until then (manager ruling 21 Nov); its facts are kept in section 2 for that. GT-18 re-counts.

## 2. Game facts per group (tracked data)

Stored in `overrides.toml` per `[[gauntlet]]` and resolved into the tables. Only these keys, each with its source:
- `source`: the citation, checked by `_SOURCE` as today (`tools/kaizo/overrides.py:93-98`; `_source` `:163-170`). E4 + Champion (B4-19): `source = "rom:scr 188/190/192/194/196"`. No separate cite or kind field: `check_gauntlet_members` skips every non-sheet source (`overrides.py:630`, `_SHEET.fullmatch`).
- `orders`: the fight orders the map allows, a list of orders, each a list of fights (lists of trainer ids), with `orders_source`. One order means the game fixes it; there is no separate forced order.
- `leave_gaps`: gaps after which the player can leave the group's map on foot without entering an undefeated member's sight line; one list per order where they differ (Building #1).
- `escape`: whether the header allows Escape Rope and Dig (F6). A partner blocks both (Wayward Cave facts).
- `pc_gaps`: gaps with a reachable PC inside the group's maps. `heal_gaps`: gaps with a reachable free full heal inside the group's maps.
- `locked_gaps`: gaps where the game gives no field control and takes no steps (B2B). No player action and no field poison happen there.
- `partner` (Wayward Cave only): `{ trainer = 612, join_before = [1], source }`: the one fight before which the game makes the partner join (a game fact, not a choice; Wayward Cave facts).
- Every `source` string is a valid `_SOURCE` string (`rom:...` or a sheet cite); GT-2 lists the exact strings.
- `rules = {}` and `RULE_KEYS` (`overrides.py:71`) are removed. `resolve.py` copies only these facts (today `:527` copies `g.rules` and `:594` writes `gauntlet_rules` into every trainer row). `trainers.json` gains `gauntlet_facts` in place of `gauntlet_rules`.
- The map facts come from a read-only ROM tool added in GT-2 (`tools/kaizo/gauntlet_map.py`), so they can be re-derived. The values below are its oracle's results (`_scratch/e3/rules/b3g/facts-final.md`, git-ignored, holds the run log).

Game behaviour that holds at every gap is a rule, not a stored value: the partner heal (Wayward Cave facts), the partner's block on Escape Rope and Dig, the lead rule (section 5).

### Shared facts

- F1. After every object-trainer fight the player gets field control back: common battle script scr 1114, `StartTrainerBattle` @3873, `ReleaseAll` @3900. The start menu, party menu and bag open from there.
- F2. Field item use: healing items through `UseHealingItemFromMenu` (`src/item_use_functions.c:142`, `:295`), TMs and HMs through `UseTMHMFromMenu` (`:147`, `:526`). An item's effect on a Pokemon is `Pokemon_ApplyItemEffects` (`src/item_use_pokemon.c:237-391`), driven by the item's `ITEM_PARAM_*` fields; the item table transcribes its param branches, not item classes. Checks (`:155-175`): revive when `REVIVE` and not `LEVEL_UP` (`:158-162`), HP restore (`:163-167`), Rare Candy below level 100 (`:171-175`). Apply: a `REVIVE && LEVEL_UP` item revives a fainted member (`:271-279`); HP restore (`:280-286`); Rare Candy: EXP to the next level, stats recalculated, and a fainted member revived with HP equal to the max-HP gain (`:291-299`); status cures (`:150-154`), PP restore (`:193-200`, `:326`), PP Up and PP Max (`:185`), stone evolution (`:178`), EVs (`:217-224`); friendship by tier from `ITEM_PARAM_FRIENDSHIP_LOW/MED/HIGH` through `UpdatePokemonFriendship` (`:363-387`, `:565-596`): Soothe Bell x150/100 first (`include/constants/items.h:50`), then +1 for a Luxury Ball and +1 when the egg location equals the caller's `location`, clamp 0..255. A Revive works in the field; keeping fainted members out is a Nuzlocke setting (`run.revived_dead`).
- F3. No `HealParty` in any gauntlet map's own script file or in the E4 room and lift files (rom: scr 215, 431, 277, 73, 80, 81, 459, 318, 188-196; heal sites from a walk of every scr_seq member). Route 213's healer is in the lobby (header 376, scr 462).
- F4. Kaizo script command 0x3F (sets a party status) runs only in headers 104, 113, 130, 148, 158, 160, 216, 257, 301, 302, 306, 321, 348, 384, 415, 440, 514 (`data/kaizo/rom_diff.toml:232`). None is a gauntlet header (199, 346, 256, 67, 72-75, 373, 376, 284, 175-186).
- F5. Field poison: one tick per 4 player steps (`Field_UpdatePoison`, `src/overlay005/field_control.c:895-905`). A tick takes 1 HP from each poisoned or badly poisoned member that can battle, never below 1 (`Pokemon_DoPoisonDamage`, `src/unk_02054884.c:177-212`); at 1 HP the status is cleared (`Pokemon_TrySurvivePoison`, `:215-222`) with a friendship loss (`:197-199`; `src/pokemon.c:2630`). Field poison never faints a member. The step count of a gap is not in the data; a locked gap takes no steps.
- F6. Escape Rope and Dig work only on a cave map whose header allows escape and only without a partner (`CanUseEscapeRope`, `src/item_use_functions.c:941-948`, partner at `:943-945`; `FieldMoves_CheckDig`, `src/field_move_tasks.c:754-764`; map type and escape bits of the header's u16 at offset 22, `include/map_header.h:25-29`; Kaizo reads them from d_test.narc, `data/kaizo/rom_diff.toml:122-124`, `:142`).
- F7. Sight: a trainer of type NORMAL spots the player on its facing line within its sight range, unless a tile between them blocks (`GetTrainerDistToPlayer`, `src/trainer_encounter.c:182-229`; `IsPathInterrupted`, `:310-344`). A LOOK_* movement type turns the trainer among its listed directions at random times (`sub_020645C0`, `src/unk_0206450C.c`), so passing it or being seen is the player's timing choice: the walk takes both. The first seer in object order fights; with two live Pokemon or a partner set (`field_control.c:181-186`) a singles seer and the next one fight as one double (`APPROACH_TYPE_VS2`, `trainer_encounter.c:94-106`); a double-battle trainer fights with both objects, and with one live Pokemon neither by sight nor by talk (`trainer_encounter.c:108-118`; `res/field/scripts/scripts_battles.s:19-22`). The check runs on every field-input frame (`FieldInput_Process`, `src/overlay005/field_control.c:181-188`, calling `FieldSystem_CheckForTrainersWantingBattle`, `src/trainer_encounter.c:84-100`), so it also fires on the tile a warp lands on.
- F8. Lead: the battle's first member is the first party slot with a species, not an egg and HP > 0 (`src/battle/battle_main.c:1335-1348`); `BattleSystem_InitPartyOrder` swaps it to the front of `partyOrder` (`src/battle/battle_lib.c:6465-6473`, `BattleSystem_SwitchSlots` `:6476-6507`, target slot 0 in singles `:6495`). The save party's order is unchanged (`battle_main.c:703-705`).

### Map reads

Method: Kaizo ROM, per group map: terrain collision bit, object events (item balls removable; Cut trees, Rock Smash rocks and Strength boulders removable only with the field move's badge, `FieldMoves.for_split`: the badges of the splits before the group's own), warps (intra-map warp panels followed), door tiles (behaviour 0x69 `TILE_BEHAVIOR_DOOR`, `include/constants/field/map_tile_behaviors.h:116`) entered by walking into them (`Field_CheckMapTransition`, `src/overlay005/field_control.c:572-592`), objects behind a table tile (0x80, `map_tile_behaviors.h:139`) talked to across it (`src/unk_0203C954.c:58-80`), each undefeated member's sight line over every direction its movement type can face (F7), checked on every tile entered and on every warp's arrival tile (F7). Tile behaviours (`player_move.c:1865-1907`, `map_object_move.c:706-737`): ledges 0x38-0x3B jump two tiles one way; water (the SURFABLE set) needs Surf (Fen Badge, no partner: `field_control.c:690`, `scripts_field_moves.s:302-303`); directional tiles 0x30-0x37, 0x49, 0x4A block one side for the walk and for sight lines; spin tiles 0x40-0x43 carry the player with no field input (`ov5_021E1154.c`; the trainer check of the frame before the spin, `field_control.c:187`, still runs on the tile stepped onto); ice 0x20 slides to the next collision; Rock Climb (Icicle) and Waterfall (Beacon) need their badge; bridges 0x70-0x7D keep the elevated state. Height: each tile's BDHC plate nearest the current height is the player's next height; a step whose heights differ by 20 or more collides (`terrain_collision_manager.c:249-305`); sight and talk need the same height unit (`grid_y`). Coord events on the path are checked against their var; a var below 0x4000 is a literal (`FieldSystem_TryGetVar`, `script_manager.c:432-433`), so a var-0 coord runs only when its value is 0. Each fight's map header is recorded (`fight_headers`); the Escape Rope header is the last fight's. Dynamic features: the Eterna Gym clock per state (Eterna Gym facts). Hidden flags: an object is created on a map load only while its hidden flag is clear (`sub_020620C4`, `src/map_object.c:607-624`; a script-less object always, flag 0 is never set, `src/vars_flags.c:52-56`); the walk carries the set flags (`Walker.flags`) and adds those a header's OnTransition script sets on every path when a warp lands there (`Walker.on_enter`; `FieldMapChange_UpdateGameData`, `src/field_map_change.c:254-299`, runs before `FieldMapChange_CreateObjects`, `:341-351`). The flags set at each group come from every SetFlag/ClearFlag of the flag in the ROM (`gauntlet_map.FLAG_WRITERS`, `GROUP_FLAGS`, `ENTER_FLAGS`); hidden at the group: the B2B works key (256 obj 3), on Route 213 the grunt, Looker (373 objs 28, 31) and the lobby's Looker (376 obj 2); trainer 696's pair and its spectators (373 objs 33-38) are hidden once the lobby is entered (scr 462 entry 1 @37, @152). Objects outside a group's own headers change no fact (ROM test, both extremes). Live count: the walk is done for two live Pokemon (or a partner) and for one; the one-live facts are `one_live` (two seers of one tile fight in turn, a double-battle trainer is not fought); only Route 204 North differs (15, the twins, is skipped: 13, 14, leave after 13). The planner picks by the party. Return visits: returning after the group's own split (its badges would widen the walk) is run-level planning's: the gauntlet is defined at its split. Roark's rock removal (scr 215 `RemoveObject 0` @70, `RemoveObject 1` @197) runs after both Oreburgh fights, so it does not change the walk. Tile kinds present in the gauntlet headers (ROM test): ledges in 199, 346, 256; water and Rock Climb in 373; no directional, spin, ice, bridge or waterfall tile. An order is allowed when each next member's sight line or a tile next to it is reachable, from any entry warp of the group's map or from the last opponent, without entering another undefeated member's sight line. The walk masks the matrix to the group's headers: Route 204 North is entered from the Ravaged Path side only (assumption; a Floaroma-side arrival is not read).

v5.1's read had no arrival check and no door warps. Re-run of every group, 20 Nov (before -> after):
- Eterna Building #3: free order, leave after 422 first -> order 412, 422 only, no leave gap. Every warp panel lands on (25,20), inside 412's sight.
- Route 213: unchanged values, but the lobby door was a wall in the old read; leave and heal now come from the read instead of by hand.
- Oreburgh Mine, Route 204 North, Eterna Gym, Building #1, Building #2, Wayward Cave, E4 PC tiles: unchanged.

Walk-fix round (22 Nov, areas review): Oreburgh Mine 195, 196 with leave [1] -> 196, 195 with no leave gap (rocks before the Coal Badge, one-way ledge); Route 213 120 orders, heal at every gap -> 2 orders, no heal gap (no Surf in the Wake split). Every other group unchanged (`_scratch/e3/walk_fix_diff.md`).

Rulings round (22 Nov): hidden flags and the live count are modelled (above); Route 204 North gains `one_live` [[13], [14]], leave [1]; every other fact unchanged. Header 256's battles are literal ids: scr 277 entry 2 `StartTrainerBattle` @44 (296, 0), @73 (297, 0) (opcode 0xE5, `include/data/scripts/scrcmd.h:237`, `asm/macros/scrcmd.inc:1427-1431`; operands read by `ScriptContext_GetVar`, `src/scrcmd_trainer.c:129-130`: a number below 0x4000 is itself). `_scratch/e3/walk_fix_diff.md`.

v5.4 (20 Nov): the read now applies every coord's var condition. In Wayward Cave the walk-out coord (85,8)-(85,10) runs while Mira has not joined and closes the start pocket, so no fight starts before she joins: `partner.join_before = [1]`, and the `partner_from` setting and the partnerless formats are dropped.

| Group (header) | Map-allowed orders | Leave on foot | Escape Rope / Dig (F6) | Heal or PC mid-group |
|---|---|---|---|---|
| Oreburgh Mine (199) | 196 then 195 only. Before the Coal Badge the Rock Smash rocks objs 1 and 2 at (19,28)-(19,29) block the east way (`scripts_field_moves.s:134`); the only way west is the ledge row (2-4,19) (behaviour 0x3B, one-way), into 196's line at row 24, then 195's tile (10,29). Events 197: 195 = obj 5 (scr 3194) at (10,28), LOOK_SOUTH, range 3, sight (10,29) only ((10,30) has terrain collision); 196 = obj 6 (scr 3195) at (2,24), LOOK_EAST, range 5, sight (3,24)-(5,24). From the stairs warp (15,1), rows 28-29 west of x = 19 are entered only through x = 19 (Rock Smash rocks objs 1 and 2, or Roark's script: scr 215 `RemoveObject 0` @70, `RemoveObject 1` @197), and x = 10 is crossed only at (10,29), 195's sight tile. | no: the stairs are not reached again before both fights | yes: map type 3, escape bit set (rom header 199) | none |
| Route 204 North (346) | 15, 13, 14 only, from the Ravaged Path arrival (180,699) | after 15 and after 13: yes, warp (180,698) -> header 254 | no: map type 2 | none |
| Route 204 North B2B (256) | 296, 297, scripted | no: no field control | n/a | no |
| Eterna Gym trainers (67) | the guide talk, then 324, 259, 260 only (Eterna Gym facts) | no | no: map type 4, escape bit 0 | no |
| Eterna Building #1 (73) | both: 421, 831 and 831, 421. Events 72: 421 = obj 0 (scr 3420) at (8,6), LOOK_NORTH, range 3, sight (8,5)-(8,3); 831 = obj 2 (scr 3830) at (16,9), facing west, range 4. 1F (header 72) warp 1 (14,6) -> 2F (8,3), an arrival in 421's sight; warp 3 (20,6) -> 2F (13,3), from where the only way south crosses 831's row-9 sight. | order 831, 421: after 831 yes (-> header 65). Order 421, 831: no | no: map type 4, escape bit 0 | none |
| Eterna Building #2 (74) | 410, 423 only. 410 = obj 0 (scr 3409) at (9,14), facing north, range 3; 2F warp 1 (16,3) -> (9,11), an arrival in 410's sight | after 410: yes | no | none |
| Eterna Building #3 (74) | 412, 422 only. 412 = obj 2 (scr 3411) at (27,20), facing west, range 3, sight (26,20)-(24,20); every warp panel leads to warp 4 = (25,20), inside it. 422 = obj 4 (scr 3421) at (23,18), facing east, range 4, reachable only after 412 | no | no | none |
| Route 213 (373) | 106, 103, 105, 111, 110 from the lobby end, or the reverse from the west end: no Surf before the Fen Badge, so the beach is one line | after every fight: yes, lobby door (706,818) -> header 376 | no: escape bit 0 | in no gap of every order: the lobby healer (header 376 obj 1 at (11,4), across the table tile (11,5); scr 462 entry 2 `HealParty` @74) is reached at every gap from the lobby end only |
| Wayward Cave (284) | 383+443, 380+458, 450+496, 459+460, 454+455, 749+750; or 459 and 460 split (Wayward Cave facts) | no (the walk-out runs only before Mira joins, before fight 1) | header yes; refused at every gap (the partner is set from fight 1) | partner heal after every won fight; no PC tile |
| Elite Four + Champion (175-186; group added by B4-19) | fixed by the game | no: doors | no: map type 4 | PC in the lifts between rooms; no healer |

Across the three Eterna Building groups (headers 72-75): 410 is reachable before any Building #1 fight; 423 only after 410; 412 only after 423. After 421 alone the only continuation is 831; after 412 alone it is 422. Every other point where another building group's fight is reachable is a leave gap or lies before a group's first fight, so the groups stay separate chains. A fight of another group taken at a leave gap is the player's choice; the next re-run starts from the save (section 5).

Tracked facts that follow:

| Group | `orders` | Other facts | Cite |
|---|---|---|---|
| Oreburgh Mine | [[196],[195]] | `leave_gaps = []`, `escape = true` | `rom:header 199 ev 197` objs 1/2 (rocks), 5/6, ledges (2-4,19), stairs warp (15,1); sight `src/trainer_encounter.c:182-229`, `:245-306`, `:310-344`; escape `src/item_use_functions.c:941-948` |
| Route 204 North | [[15],[13],[14]] | `leave_gaps = [1, 2]`, `escape = false` | `rom:header 346 ev 332`, warp (180,698) |
| Route 204 North B2B | [[296],[297]] | `locked_gaps = [1]`, `leave_gaps = []` | `rom:scr 277 entry 2` `LockAll` @36, battle 296 @44, battle 297 @73, `ReleaseAll` @106 |
| Eterna Gym trainers | [[324],[259],[260]] | `leave_gaps = []`, `escape = false` | Eterna Gym facts |
| Eterna Building #1 | [[421],[831]], [[831],[421]] | `leave_gaps`: 421,831 -> []; 831,421 -> [1]; `escape = false` | `rom:header 73 ev 72`; header 72 ev 71 warps 1 (14,6) and 3 (20,6) |
| Eterna Building #2 | [[410],[423]] | `leave_gaps = [1]`, `escape = false` | `rom:header 74 ev 73`; header 73 warp 1 (16,3) |
| Eterna Building #3 | [[412],[422]] | `leave_gaps = []`, `escape = false` | `rom:header 74 ev 73` obj 2, warp 4 (25,20) and the panels leading to it |
| Route 213 | [[106],[103],[105],[111],[110]] and its reverse | `leave_gaps` all, `heal_gaps = []`, `escape = false` | `rom:header 373 ev 359`; water 0x15 without Surf; door (706,818) -> header 376; `rom:scr 462 entry 2` `HealParty` @74 |
| Wayward Cave | [[383,443],[380,458],[450,496],[459,460],[454,455],[749,750]] and the same with [459],[460] in place of [459,460] | `leave_gaps = []`, `escape = true` (header), `pc_gaps = []`, `heal_gaps = []`, `partner = { trainer = 612, join_before = [1] }` | `rom:header 284 ev 273`; `rom:scr 318 entry 2` `SetHasPartner` @171; coord script 6 (85,8)-(85,10); Wayward Cave facts |
| Elite Four + Champion (B4-19) | [[866],[867],[868],[869],[870]] | `pc_gaps = [1, 2, 3, 4]`, `leave_gaps = []`, `escape = false`; no heal, tutor or relearner | Elite Four facts |

Eterna Gym facts (rom header 67, events 66; scr 73):
- No member is a sight trainer: objects 2 (19,12), 3 (3,12), 4 (2,8) have trainer type NONE. Fights start from scripts: coord (19,13) -> entry 7 (324); coord (3,13) or talking to obj 3 -> entry 8 (259); talking to obj 4 -> entry 5 (260). The coord events use var 0, which always matches (`sub_0203CC14`, `src/unk_0203C954.c:218`; `FieldSystem_TryGetVar`, `src/script_manager.c:432-433`, `:447-448`).
- Two vars. 0x40AE `VAR_ETERNA_GYM_TRAINERS_BEATEN` (`generated/vars_flags.txt:4285`) gates the fights: entry 7 runs while it is <= 1 (@475), entry 8 while <= 2 (@597), entry 5 while < 5 (@301). 0x404B `VAR_ETERNA_GYM_FLOWER_CLOCK_STATE` (`:4186`) is the clock state.
- The clock has 5 states, 0-4 (`enum EternaClockState`, `include/overlay008/gym_features.h:15-22`), read at map load (`EternaGym_DynamicMapFeaturesInit`, `src/overlay008/gym_features.c:2513`) and raised by one per `AdvanceEternaGymClock` (`EternaGym_AdvanceClockState`, `:2903-2914`). Collision per state: `sEternaGymClockCollision` (`:2333-2410`), `sEternaGymFountainCollision` (`:2412-2418`), applied by `EternaGym_DynamicMapFeaturesCheckCollision` (`:2534-2559`); hour-hand jump tiles per state `sEternaGymHourHandJumpTiles` (`:2308-2331`). Kaizo leaves overlay 8 unchanged (`data/kaizo/rom_diff.toml` has no overlay-8 entry).
- Clock advances: the gym guide's first talk (obj 0, entry 2 `AdvanceEternaGymClock` @105, once, gated by var 0x40AB @77); the first win over 324 (entry 7 @586 -> @1177); the win over 260 (entry 5 @361); the leader (@1065).
- Walk per state, from the door (11,26): state 0: only the guide is reachable. State 1: only coord (19,13) -> 324. State 2 (after 324, player on (19,13)): only coord (3,13) or obj 3 -> 259; the door is not reachable. 259's win keeps state 2 (entry 8 has no clock advance): obj 4 -> 260 is reachable, the door is not. State 3 (after 260): the door and the leader are reachable.
- So the order is fixed: the guide talk (no fight), then 324, 259, 260. No walk-out at gap 1 or 2. Escape: map type 4, escape bit 0, fly bit 0.
- Out of the model: the Sudowoodo (obj 7 at (19,20)), talked to from the south while 0x40AE is 2-4, resets 0x40AE to 1 (entry 10 @1238-1314), which re-opens entry 7 and forces a second fight with 324. The player stands south of it only after leaving the gym, and no gap allows leaving (a lost fight ends the chain). Nothing is stored for it.

Wayward Cave facts (rom header 284, events 273; scr 318):
- Partner: talking to Mira (obj 4 at (72,13), entry 2) sets 0x4091 `VAR_WAYWARD_CAVE_1F_FOLLOWER_MIRA_STATE` = 1 (@152; `generated/vars_flags.txt:4256`) and 0x403F `VAR_PARTNER_TRAINER_ID` = 612 (@165; `:4174`), and runs `SetHasPartner` (@171). The main entrance leads into coord row 48 (x 39-42, 0x4091 == 0): entry 5 warps the player to (84,9) (@199-232), Mira's area.
- When Mira joins: before fight 1. Header 284 has three coords, all on 0x4091: row 47 (x 39-42, value 1, entry 4: 749+750), row 48 (x 39-42, value 0, entry 5: back to (84,9)) and (85,8)-(85,10) (value 0, entry 6). Entry 6 asks "Ignore the cry for help?" (@277, `ShowYesNoMenu` @280); yes (`MENU_YES` = 0, `include/constants/menu.h:4`) warps the player out of the cave to header 350 (310,608), Route 206 at the cave mouth (@399); no walks the player 2 tiles west (@299, movement @1098). Those three tiles are the only way east of the start pocket, so from (84,9) the walk without her reaches 76 tiles, her tile and no trainer. No fight can start before she joins: `partner.join_before = [1]`. Walking out before talking to her skips Wayward Cave (then flag 864 stays unset, which decides Spear Pillar's format: `rom:scr 236 @1118-1129`, `@1411-1438`).
- Formats: with the partner the field counts the party as able to fight two (`src/overlay005/field_control.c:182-186`), so a tile both members of a pair see starts one battle with both (`FieldSystem_CheckForTrainersWantingBattle`, `src/trainer_encounter.c:84-100`), a multi with the partner (`BATTLE_TYPE_TRAINER_WITH_AI_PARTNER`, `Encounter_NewVsTrainer`, `src/encounter.c:730-735`; `ScrCmd_StartTrainerBattle`, `src/scrcmd_trainer.c:124-142`). The partner is set at every fight, so every pair is a multi. One id is a single beside the partner: with one enemy id `Encounter_NewVsTrainer` picks `BATTLE_TYPE_TRAINER` (`src/encounter.c:738-739`), and `Trainer_Encounter` ORs in only the trainer headers' `battleType` (`src/trainer_data.c:50`), 0 for 459, 460 and 612 (rom `poketool/trainer/trdata.narc`, byte 16). `trainers.json` stores the stored case (multi, partner 612); the chain derives each fight's format from these rules.
- Split: 459 can be talked to alone from (34,28) or (35,27), outside 460's sight (from the ROM script, not observed in play). Talking runs the common battle script with one id (rom: scr 1114 `StartTrainerBattle` @3873; `asm/macros/scrcmd.inc:1427-1431`, enemy2 = 0), a single; 460 then spots the player at (35,29), a second single (scr 1114 @4275, enemy2 = 0). Both are checked with Mira joined and every coord applied. No other pair member has a reachable tile outside both sight lines.
- Order: 383+443, 380+458, 450+496, 459+460, 454+455, then 749+750, a scripted tag battle at the exit (coord row 47, x 39-42, 0x4091 == 1 -> entry 4 -> @369 -> @470, `StartTagBattle 612, 750, 749` @667). 459+460 is reachable only past the Strength boulders at (26,16) and (28,16) (`FieldMoves_CheckStrength` needs `BADGE_ID_MINE`, `src/field_move_tasks.c:457-470`). No pair can be skipped. The arrival check changes nothing here.
- No walk-out at any gap: with 0x4091 == 1 neither entry 6 nor row 48 runs. The header allows Escape Rope and Dig (map type 3, escape bit 1), but the partner is set at every gap, so both are refused at every gap (`src/item_use_functions.c:943-945`; Dig `src/field_move_tasks.c:764`; Teleport `:717`).
- Heal: after every won battle (the partner flag is set at every fight), `Party_HealAllMembers` on the save party (`src/encounter.c:197-199`; `src/item_use_pokemon.c:603-628`): HP to max for every member, fainted ones included, status cleared, PP restored. This is game behaviour, applied by the transition (section 5), not a setting. No PC tile (behaviour 0x83) in header 284.

Elite Four facts (kept for batch 4: B4-19 restores the `[[gauntlet]]` entry in `data/kaizo/overrides.toml` and adds `league_facts` (`tools/kaizo/gauntlet_map.py`) to `derive_groups`. The entry: `name = "Elite Four + Champion"`, `trainers = [866, 867, 868, 869, 870]`, `source = orders_source = "rom:scr 188/190/192/194/196"`, `orders = [[[866], [867], [868], [869], [870]]]`, `leave_gaps = []`, `escape = false`, `pc_gaps = [1, 2, 3, 4]`, `pc_gaps_source = "rom:headers 178, 180, 182 (6,11), 184 (6,19) tile behaviour 0x83"`, `heal_gaps = []`, `locked_gaps = []`):
- Order and doors. After a win the room script removes the exit door (`res/field/scripts/scripts_pokemon_league_aaron_room.s:24`) and returns control (`:30`). The exit warp leads to the next lift (`res/field/events/events_pokemon_league_aaron_room.json`: (8,2) -> `ELEVATOR_TO_BERTHA_ROOM`), and the lift's warp to the next room (`events_pokemon_league_elevator_to_bertha_room.json`: (4,2) -> `BERTHA_ROOM`). Entering a room adds the entrance door behind the player (`scripts_pokemon_league_aaron_room.s:62-69`). The Kaizo files have the same structure (rom: scr 188-196, init entry 2 `AddObject` @142, @180). The order is fixed and the player cannot walk back.
- Bag and party menu. Each fight ends in `ReleaseAll` (rom: scr 188/190/194 @118, scr 192 @156), so the bag and the party menu are open at every gap (F1, F2). No script in scr 188-196 or the lift files disables the start menu.
- PC. The lifts between rooms each have a tile with behaviour 0x83 `TILE_BEHAVIOR_PC` (`include/constants/field/map_tile_behaviors.h:142`), reachable from the lift's arrival: headers 178, 180, 182 at (6,11) and 184 at (6,19) (gaps 1-4; rom scan re-run 20 Nov; header 176 is the lift before Aaron). Vanilla `res/field/maps/data/map_data_306.bin`-`308.bin` hold 0 at those tiles: Kaizo added them. Facing it north runs `CommonScript_PC` (`src/overlay005/field_control.c:654-655`; `res/field/scripts/scripts_common.s:47`).
- No healer: the rooms have no PC tile; room events hold only the member and two doors; lifts have no objects; no script heals between rooms (F3). The last nurse is in header 175 (North Pokemon Center 1F), before Aaron.
- No tutor or relearner object in any room or lift (same events). TMs and HMs in the bag remain usable (F2).

## 3. User settings (local or per run, never tracked)

- Gauntlet options are entered per run: flags, or `--rules FILE` naming a file the user keeps outside the tracked tree.
- Standing Nuzlocke settings may sit in `[run]` of `data/local.toml`, the existing git-ignored local inputs file (`.gitignore:14`; `docs/kaizo/pipeline.md:10`). No tool or session writes the user's `data/local.toml`.
- `[run]` lives in `data/local.toml`, not a second file. `tools/rom/inputs.py:64-79` (`_read_toml`) today raises `RomInputError` on any top-level key outside `_ENV`; GT-3 makes it skip the `run` table (and only that table), with tests in `tests/tools/rom/test_inputs.py`. Reason: one git-ignored local file keeps one place for the user's machine-local inputs, the change is one line plus a test, and a `local.toml` without `[run]` loads exactly as before. `data/local.example.toml` gets a documented `[run]` block holding the strictest values (no personal answers); `test_example_config_parses` (`tests/tools/rom/test_inputs.py:57-58`) then covers the skip. The gauntlet loader reads only `[run]` with `tomllib`; `src` does not import `tools`.
- Precedence per value: `locked` (a locked gap) > flag > `--rules FILE` > `data/local.toml` > default. The report prints every value with its origin and says when a default was used.
- Input errors: unknown key, field or value; a `[run]` key in a gauntlet table or the reverse; gap outside 1..n-1 of the resolved order; an order not in the group's `orders`; a `--dead` party slot outside the save party or box id not in the save's box; a `--bench` box id also in `--dead`; a `--dead` member at HP 0 (already dead); an explicit gap value other than the strictest at a locked gap (the message gives the `locked_gaps` source). Walk-out gaps of a resolved order = its `leave_gaps`, plus every gap where `escape` holds and no partner is set (the partner who joins before fight J is set at gaps J and later). Where the map read settles a gap: `heal = full` outside `heal_gaps` and outside the walk-out gaps; `pc_heal = on`, or a swap, outside `pc_gaps` and outside the walk-out gaps. Nothing else is rejected. A value merely inherited from the group, a rules file or a default is never an error at a locked gap; it is not applied there and the report says so.

Schema; choices are listed strictest first; the default is the first value unless marked.

```toml
[run]                          # Nuzlocke rules
deaths_scope = "chain" | "fight"
cap_overflow = "sit_out_after_fight" | "fight"
cap_sit_out_in_party = "forbidden" | "allowed"
revived_dead = "stay_dead" | "revived"
shedinja = "forbidden" | "allowed"
field_poison = "worst_case" | "ignore"

[gauntlet."<group name>"]      # per run
heal = "none" | "bag" | "full"
bag_uses = <int >= 0> | -1                                 # default 0; -1 = only what the bag holds
pc_heal = "off" | "on"
growth_items = "forbidden" | "allowed"                     # Rare Candy and stones between fights
team = "none" | "reorder" | "moves"
move_sources = ["bag"] | ["bag", "tutor", "relearner"]
swap = { bench = <int >= 0 | -1>, budget = <int >= 0 | -1> }   # default 0, 0
order = "worst" | "sheet" | [<first trainer id of each fight>, ...] | "best"
time = "day" | "night"                                     # no default: unset offers no day/night evolution

[gauntlet."<group name>".gap.<k>]                          # overrides the group value at gap k
heal = ...; pc_heal = ...; team = ...; growth_items = ...
swap_cap = <int >= 0>; bag_cap = <int >= 0>                # most spent at this gap; 0 closes it
```

Option kinds (`src/nuzline/data/gauntlet.py`, `OPTIONS` registry; loader, CLI, coverage and report iterate it):
- Level (ordered, per gap): `heal`, `pc_heal`, `team`, `growth_items`.
- Counted (whole remaining chain, spent at gaps, optional per-gap cap): `swap` (`bench`, `budget`), `bag_uses`.
- Set (whole chain): `order`, `move_sources`.
- Plain choice: the `[run]` keys, `time`.
- Adding an option: its `OPTIONS` entry and its transition in `prep/gauntlet.py`; the loader refuses an option without a transition.

Meaning:
- `heal = bag`: bag medicine and PP items may be used (F2); each use is charged to `Cost` and to `bag_uses`. Ethers and Elixirs count like any other bag item. `full`: HP, PP and status reset on living members.
- `pc_heal = on`: at a gap, a member may be deposited and withdrawn: PP restored at deposit (`src/pc_boxes.c:82`; `BoxPokemon_RestorePP`, `src/pokemon.c:5109-5122`), Shaymin set to Land form at deposit (`src/pc_boxes.c:83`), HP to max and status cleared at withdraw (`Pokemon_FromBoxPokemon`, `src/pokemon.c:4031-4055`; `:633-645`). The withdrawn member goes to the end of the party (`Party_AddPokemon`, `src/party.c:50-59`; `src/applications/pc_boxes/box_app_manager.c:3963`). Under `team = none` the order is kept by withdrawing in party order.
- `swap`: `bench` = box members chosen before fight 1; `budget` = members swapped in over the chain; `-1` = unlimited. A swapped-in member arrives with full HP, PP and status, at the end of the party. A deposited Shaymin is in Land form (`src/pc_boxes.c:83`).
- `order = "worst"` (default): every order in `orders` that `--beaten` still allows is open, and a setup's score is its minimum over them; the report names the worst order (section 5, "Chosen order"). It is a lower bound under any order the user plays, so it is the strictest value. `"sheet"`: the sheet order when it is in `orders`, else `worst`, and the report says which. A list picks one of `orders`. `"best"`: the search picks the order, and the answer names the next opponent.
- `cap_overflow = sit_out_after_fight`: a member that passes the next fight's level cap (`config/level_caps.toml`, `data/kaizo/level_caps.toml`) finishes the current fight, then takes no part in any later fight of the chain. `cap_sit_out_in_party = forbidden`: it must be deposited at that gap (a PC gap, a walk-out gap or a swap), else the playout row counts as lost; `allowed`: it may stay in the party unused (barred, section 5). A fight where only barred members can still battle counts as lost.
- `field_poison = worst_case`: at every gap that is not locked, each poisoned living member drops to 1 HP and is cured, with the friendship loss (F5). `ignore`: no change.
- `time`: day and night evolutions are offered only with `time` set. At the E4 + Champion a level-100 member gains no EXP (`bs.c:9740`) and so is never offered a level evolution; `time` matters there only for members below 100.
- CLI: `nuzline prep|coverage --rule heal=bag --rule swap=bench:1,budget:1 --gap-rule 2:heal=full --rules FILE`; `--trainer-id` of any member selects the group; the current fight follows from `--beaten` and `order`; `--beaten <ids>` in played order; `--deaths` per `deaths_scope`, and on a re-run (`--beaten` non-empty) the deaths still allowed, not the run's total; `--dead <party slots or box ids>`, `--swaps-left N` and `--bench <box ids>` for a re-run mid-gauntlet (user input per run, nothing tracked; the save holds no Nuzlocke deaths, swaps or bench); `--horizon N`; `--growth-cap N`; `--time`; `--next <trainer-id>`.

## 4. Run-level scope

None. The AI's untyped Weather Ball byte is per-battle state: E1 adds it to `BattleState` as a hidden, read-only battle input with its rule, and every gauntlet fight draws its own unknown value. Nothing in this design carries it between fights.

## 5. Prep search

**Carry** (end of fight k = start of the next): section 7 per save-party slot; party order; bag counts; dead members; deaths used; per Counted option the amount left; the bench; the swap plan; the growth plan (section 6); sit-out members; the fights beaten; whether the partner is set.

**Root of a run.**
- First fight of a group (no `--beaten`): today's path. The pool is party plus box at the cap, healed (`prep/pool.py:84`; `prep/leveling.py:18-31`), because the player can heal before entering.
- Re-run mid-chain (`--beaten` non-empty): the root is the save's party as it stands: its members and order, HP, PP, status, levels, held items, and the bag's counts. No `level_to_cap`, no healing, no free Rare Candy levels. The root's setup space is what the settings allow at the current gap (gap k = fights beaten along the resolved order): `heal`, `pc_heal`, `team`, Rare Candies and stones from the bag under `growth_items = allowed`, swaps from `--bench` within `--swaps-left`. The bench comes from the box as the save holds it.
- Nuzlocke state of the root. The save holds no Nuzlocke-dead flag and no count of deaths used, and in Wayward Cave the partner heal restores fainted members to full HP after every won fight (`src/encounter.c:197-199`; `src/item_use_pokemon.c:603-628`), so a save can show a dead member as healthy. So: the root `start.dead` = party members at HP 0, plus the members `--dead` names (party slots, or box ids, which also leave the bench pool). Deaths used at the root = 0: on a re-run `--deaths` is the deaths still allowed, and a member already dead adds no new death. Root sit-outs: members whose level exceeds the current fight's cap (`prep/caps.py:86`), per `cap_overflow` and `cap_sit_out_in_party` (section 3); a `forbidden` sit-out must be deposited at the current gap, else the setup scores 0. On a first fight (no `--beaten`) `--dead` members are left out of the pool, as `--exclude` does.

**Start of the next fight** (`write_carry`, below; for the root of a first fight, `make_states` from specs as today).
- Battle party = the whole save party in its order, fainted members included (`battle_main.c:703-705`; `src/encounter.c:151-158`). The lead is the first member with HP > 0 (F8). `slot_of[6]` maps engine party slot -> carry slot: identity with the lead swapped to slot 0, as `BattleSystem_InitPartyOrder` does (F8).
- Barred members: a member that may not fight under a Nuzlocke setting (dead under `revived_dead = stay_dead` but at HP > 0 after the partner heal or a Rare Candy (F2); sit-out under `cap_sit_out_in_party = allowed`) stays in the battle party, as the game keeps it. The player side gets a per-row `barred[6]` input (a playout and Stage B input, not a `BattleState` field): a barred member is never a switch target; while one is active (it led, or a forced switch-in such as Roar brought it in), the player's legal actions are its switches, and its own actions only when no switch is legal. Under `team >= reorder` the transition moves barred members behind the others. No new death is counted for a member already dead.
- Format per fight: one id -> single, with the partner set too (Wayward Cave facts); two ids with the partner set -> multi with the partner; two ids without a partner -> two-trainer double if two party members can battle (`src/encounter.c:731-732`), else the two ids as singles in turn (`src/overlay005/field_control.c:182-186`; `src/trainer_encounter.c:94-100`). The partner's party is rebuilt from trainer data every fight (`src/encounter.c:753`; `Trainer_Encounter`, `src/trainer_data.c:25-47`).
- Battle-start friendship: when enemy 1's or enemy 2's class is a Gym Leader, Elite Four member or Champion, every member gains `{3, 2, 1}` by band (`src/pokemon.c:2626`, `:2652-2660`), +1 for a Luxury Ball (`:2664-2666`), +1 when `MON_DATA_EGG_LOCATION` equals the battle's `mapHeader` (`:2668-2670`), which is the map label text id of the fight's header (`battleSys->mapHeader = dto->mapLabelTextID`, `src/battle/battle_main.c:1054`; `dto->mapLabelTextID = MapHeader_GetMapLabelTextID(mapHeaderID)`, `src/field_battle_data_transfer.c:263`), then Soothe Bell x150/100 and the clamp (`src/pokemon.c:2672-2686`); loop `src/battle/battle_main.c:1360-1372`, classes `:1376-1397`. It runs before the battlers are built, so it changes Return and Frustration in that fight; it applies to single Gym fights too. The partner party's members get it too (`battle_main.c:1368-1371`); no tag fight in the game is against these classes, so prep applies it to party 0 only and says so.

**Transition at gap k**, in order:
0. Learn prompts left at a battle-ending KO (G1) and evolution prompts (G6), answered by the growth plan. They run before field control returns (`src/battle/battle_main.c:256-297`).
1. Partner heal (game behaviour): when the partner is set and fight k was won, `Party_HealAllMembers`: HP to max on every member, fainted ones included, status cleared, PP restored (`src/encounter.c:197-199`; `src/item_use_pokemon.c:603-628`). No setting removes it; it charges nothing to `Cost` or `bag_uses`; the player's own settings govern only the player's own actions after it. Members dead under `revived_dead = stay_dead` stay dead and become barred.
2. Locked gap: nothing else runs (no player action, no steps, no field poison); go to the next fight.
3. `heal = full`: HP, PP and status reset on living members.
4. `pc_heal = on`: members the plan deposits and withdraws get full HP, PP and status; party order follows.
5. `heal = bag`: medicine and PP items (F2) per the plan, each charged to `Cost` and `bag_uses`.
6. `growth_items = allowed`: Rare Candies and stones per the growth plan (G7).
7. `team >= reorder`: party order and held items. `team = moves`: also TMs and HMs from the bag (+1/+1/+0 friendship by band per TM or HM, `FRIENDSHIP_EVENT_LEARN_TMHM`, `src/pokemon.c:2627`), and tutor or relearner moves when `move_sources` includes them.
8. `swap`: the plan's swap at gap k runs if budget and cap allow. A `cap_sit_out_in_party = forbidden` sit-out is deposited here or in step 4; if neither is open the row counts as lost.
9. `field_poison` (F5).

**Chain playout (host-side gaps).** A playout row plays fight k on the device, the host runs the gap, the device plays fight k + 1.
- Device: `batched_playout_states` (new in `search/playout.py`) plays fight k for all R rows, jitted and vmapped as today. `_run` (`search/playout.py:630-697`) returns `(result, s)`; `playout` and `batched_playout` (:723) drop `s`, `batched_playout_states` keeps it. The per-row `barred[6]` is an argument of `_run`, passed to `_policy` and to the replacement policy: they never pick a switch to a barred slot and, while a barred member is active, pick only its switches when one is legal. Stage B: the playouts that value its leaves and forced nodes (`evaluate_leaves`, `search/stage_b.py`) are chain playouts started from the node's state; Stage B returns a value, not terminal states.
- Host, numpy, vectorised over R (no per-row Python loop): `jax.device_get` of the party-0 and battler fields `carry_out` reads; `carry_out`; `transition`; the chain record per row (won, deaths used, deaths left, ended).
- Fight k + 1 states: `make_states` once for fight k + 1 with a one-row template team (`prep/states.py:38`); `write_carry(template, carries, slot_of)` broadcasts it to R rows and writes each row's party 0 from its carry, recomputing stats as `Pokemon_CalcStats` (`src/pokemon.c:633-645`); then the existing exported switch-in program (`_switch_in`, `prep/states.py:34-35`, chunked as `make_states` does at `:94-109`) runs on the device. Contract: row i equals `make_states(tables, [specs of carry i in slot_of order], opp, ...)` byte for byte; a test pins it.
- Fight k + 1 inputs: `opp`, `opp_items`, `start_condition`, `terrain`, `gravity`, `trick_room` from fight k + 1's trainer data, as `cli._prep` and `build_problem` (`prep/problem.py:398`) take them for a single fight of that trainer; `chatot_cry` from the save as for fight 1; format from the rule above; `budget` = deaths left (`deaths_scope = chain`: `--deaths` minus deaths used in that row since the root, `--deaths` on a re-run already excluding earlier deaths; `fight`: `--deaths`); `protected` = the setup's protected members through `slot_of` (a protected member that died ends the row as lost); per-row `barred[6]` and the growth plan's learn answers (GT-16) as batch inputs.
- Ended rows (lost, or past the death allowance) stay in the batch with a fixed result, so the shapes and the compiled programs stay the same.
- Cost: the gap's host work is O(R x 6) numpy over a few dozen fields plus one transfer of the party fields (about 2 KB per row; 8 MB at R = 4,096), milliseconds against seconds of device play per fight. A chain of m remaining fights costs about m single-fight playouts per row. Funnel and ladder keep their playouts per setup (safety over speed), so their wall time grows about m-fold (E4 start: 5x); `--horizon` bounds it and the report prints the chain cost. GT-9 records the measured cost on the E4 chain.

**Search: receding horizon with the chain objective.**
- A setup's score is P(win every remaining fight with deaths within `--deaths` per `deaths_scope`): the mean over rows of the chain result, for one resolved order. Where `order` is `worst`, the setup's score is its minimum over every open order; under `best`, the maximum. Funnel, ladder and Stage B use this score.
- The answer commits only to the current fight: team, lead, variants, bag use, the swap at the gap before it, the next opponent (only under `best`; under `worst` the setup holds against every open next opponent, the report gives each one's score and names the worst, and the user picks), the fight's learn answers (Stage B learn nodes) and the evolution answers right after it. At the start it also commits the bench. After each fight the user saves and re-runs prep from that save.
- Where control never returns (296/297), the Stage B tree spans the boundary through the transition at the locked gap.
- No lookahead cap by default; `--horizon N` caps it and the report says `lookahead capped at N of M remaining fights`.
- The report prints the chain cost per gauntlet: wall time and playouts for screening and Stage B, beside the single-fight figures.

**Swap timing.** A plan is a list of (gap, out-member, in-member), at most `budget` long, part of the setup at the root and of the carry in playouts.
- Bench candidates: the 3 box members with the best single-fight funnel screen against any remaining fight, per bench slot.
- Full plans are enumerated when they number at most `MAX_SWAP_PLANS = 64` per (setup, bench). E4 start, bench 1, budget 1: 1 + 4 gaps x 6 out x 1 in = 25.
- Above the cap the root enumerates only the current gap's swap; later swaps follow the lookahead policy.

**Chosen order** (a group whose `orders` holds more than one order, and whose settings leave a choice: `worst`, `best`, or `sheet` falling back to `worst`).
- Candidates: the next opponent among what the open orders put next. The funnel screens each setup once per candidate: x r for r candidates. Under `worst` a setup's funnel score is the minimum over its candidates, under `best` the maximum. Stage B runs per candidate on the best setups: under `worst` a setup's Stage B value is the minimum over its candidates; under `best` Stage B runs on the best (setup, candidate) pairs.
- Lookahead under `best`: all open remaining orders are screened when they number at most `MAX_ORDERS = 24` per candidate; above that, the next opponent at each later gap is chosen greedily by single-fight screen, and the report says so. A greedy pick scores one order the user can play, so it under-estimates the best.
- Lookahead under `worst`: every open order is screened, with no greedy cut, because a minimum over a subset over-estimates (safety over speed). Cost: `worst` screens the same set `best` screens below the cap; the minimum and the maximum are taken over the same r x (remaining orders) scores, so the bound below is unchanged wherever the orders fit `MAX_ORDERS` per candidate. Every group the facts allow fits: Route 213, Wayward Cave and Building #1 have 2 orders each, the most. A group above it would be screened in full, with the count and the cost in the report; `--horizon N` cuts it to the orders of the next N fights.

**Screening order.** The funnel screens setups with no future swap, the base growth plan and, per candidate, the fixed order or (above the cap) the greedy one; under `worst` the greedy pick takes the lowest single-fight screen. The funnel only filters: the ladder's survivors are screened against every (bench, swap plan, growth plan, order) and take the minimum (`worst`) or the maximum (`best`) over the open orders. Stage B runs on the best tuples.

**Bound per setup on the ladder:** growth plans (at most `--growth-cap`, default 64) x swap plans (at most 64; 25 at the E4) x bench choices (3 per bench slot) x next-opponent candidates r x screened lookahead orders (at most 24 under `best`; all under `worst`, at most 24 in every group today). E4 start: 64 x 25 x 3 = 4,800 (order fixed, r = 1). Route 213 at its start, order open (`worst` or `best`): x 2 x 1.

**Lookahead policy** at later gaps:
- Planned swap: runs at its gap even if a member died earlier.
- Policy swap (above the cap): at a gap where a death or a sit-out left a slot and budget remains, bring in the bench member with the best precomputed screen score against the next fight.
- `heal = bag`: top every living member to full HP and PP and cure status with the smallest items that do it, oldest gap first, until `bag_uses` or the bag runs out.
- `pc_heal = on`: every living member is deposited and withdrawn in party order.
- `team >= reorder`: lead with the next living, not barred member in setup order. `team = moves`: no TM or tutor change.
- Growth: every prompt is answered by the committed growth plan.
- Level cap: per `cap_overflow` and `cap_sit_out_in_party` (section 3).
- The planned swap, bag and reorder policies under-estimate the chain; the growth plan is followable, so its score is achievable.

**Rejected:** fight by fight (trades the next fight's win chance for this one's); one joint tree over all fights (size is the product of per-fight setup spaces, up to `MAX_TEAM_SETUPS = 25_000` each, `src/nuzline/prep/setups.py:58`); swap timing, order or growth as a search choice at every playout gap; device-side gaps (a JAX `carry_out`, transition and party writer: the host path costs milliseconds per gap, and a second implementation of the transition would have to match the host one forever).

**Report line:** `Gauntlet <name>, fights i-n; settings: heal=bag (flag), swap bench 1 budget 1 (flag), order 195,196 (map: <source>), field_poison worst_case (default)`, then the next opponent (under `worst`: each open next opponent's score and the worst order), the bench, the swap plan, the growth answers for this fight and the gap after it, and the planned between-fight steps. At a locked gap: `gap k locked (<source>): no player action`. At a partner heal: `the game heals here (<source>)`. The ladder states that it covers the whole remaining chain.

## 6. Growth decisions

Level-ups, learn prompts and evolutions are decisions of the line, in single battles and in gauntlets. None is declined by default.

### Decomp facts

- **G1 When EXP is paid.** A faint sets the payout flag (`bs.c:1800`). `BattleControllerPlayer_AnyExpPayout` (`bcp.c:4449-4469`) runs `subscript_grant_exp.s` (lines 5-7). Callers: `MoveEnd` (`bcp.c:3968`), `CheckFieldConditions` (`:941`), `CheckMonConditions` (`:1296`), `TurnEnd` (`:1846`), each before `CheckBattleOver`; `TurnEnd` pays before `ReplaceFainted` (`:1848`); Pursuit pays in its script (`subscript_pursuit.s:108`). A KO that ends the battle also pays, and its prompts are answered. E2's SF3 adds `learn_site` values for the Pursuit and switch payouts; every consumer handles `learn_site` values generically.
- **G2 Who gains.** Only party 0, the player's: the payout loops over `BATTLER_US`'s party (`bs.c:2200-2202`) and the task sets `expBattler = 0` (`bs.c:9701-9704`). Members on the field while the foe was (`sideGetExpMask`, `battle_lib.c:1461-1470`) and Exp. Share holders (`bs.c:9709`, `:9745-9746`). A fainted or level-100 member gains nothing (`bs.c:9740`). Trainer x1.5 (`bs.c:9753-9754`), Lucky Egg x1.5 (`:9749-9750`), traded x1.5 or x1.7 (`:9757-9762`). Base yield x level / 7, split by participants (`bs.c:2216-2239`).
- **G3 Level-up in battle.** Per level: friendship (`FRIENDSHIP_EVENT_LEVEL_UP` `{5, 3, 2}`, `src/pokemon.c:2623`; called at `bs.c:9857`), `Pokemon_CalcStats` (`bs.c:9858`); current HP rises by the max-HP gain (`src/pokemon.c:633-645`). The active learner is reloaded (`bs.c:9860-9865`; `battle_lib.c:173-181`): stats, level, friendship, HP; stat stages untouched. The stat copy runs even under Transform; moves, PP and EXP reload only when not transformed and never into a Mimic slot (`battle_lib.c:183-193`). A benched member levels on its party copy only.
- **G4 Effect on the turn.** New stats are read at once. Turn order is computed once per turn (`bcp.c:698`, `:707`), so a Speed gain counts from the next turn. A selected move is re-read from its slot when it runs (`bcp.c:1889-1895`); a new move is selectable from the next turn.
- **G5 Learn prompt.** Each level gained is scanned (`bs.c:10013`, `:9824-9836`). `Pokemon_LevelUpMove` offers each learnset entry of exactly the current level (`src/pokemon.c:3935-3966`). An empty slot learns with no prompt (`src/pokemon.c:3846`; `bs.c:10021-10035`). With four moves the player replaces a slot or gives up (`bs.c:10069-10135`). An HM cannot be forgotten (`src/battle_sub_menus/battle_party.c:718`, `:1724-1735`). The new move gets max PP and PP Ups 0 (`bs.c:10169-10171`; `src/pokemon.c:3925-3932`). The active learner is reloaded (`bs.c:10173-10177`).
- **G6 Evolution after a battle.** Only on a win, capture or flee (`src/battle/battle_system.c:829-833`); only members that levelled (`leveledUpMonsMask`, `bs.c:9867` -> `battle_system.c:835-849`). HP is not checked: a member that levelled and then fainted is offered too (`battle_system.c:826-850`). It runs before field control returns (`battle_main.c:261`, `:274-297`). B cancels (flag 0x1, `battle_main.c:283`; `src/evolution.c:283-286`); a cancelled member is offered again only after its next level-up.
- **G7 Evolution between fights.** A Rare Candy level-up runs the same check (`src/applications/party_menu/callbacks.c:816-829`) and can be cancelled (`src/start_menu.c:1081-1090`, `:1740` flag 0x1). Kaizo's change there only keeps the menu in Rare Candy mode (`data/kaizo/rom_diff.toml:195-197`). The Rare Candy itself is F2's (`src/item_use_pokemon.c:171-175`, `:291-299`, friendship `:363-387`), including the revive of a fainted member. A stone evolution cannot be cancelled (`src/start_menu.c:1069-1080`, `:1742`); the decision is whether to use the stone. Trade evolutions (`EVO_CLASS_BY_TRADE`, `src/pokemon.c:3727`) are not offered.
- **G8 Conditions** (`Pokemon_GetEvolutionTargetSpecies`, `src/pokemon.c:3528-3725`; first match wins, `:3721-3723`). Everstone blocks every level evolution (`:3543-3547`). Methods: level; friendship >= 220 (`include/constants/pokemon.h:25`), raised at the level-up before the check (`bs.c:9857`); day or night (`IsNight`, `src/rtc.c:134`); held item by day or night; known move; species in party; gender; Attack vs Defense; personality; beauty (`MON_DATA_BEAUTY`, carried); map (`mapEvolutionMethod`, `src/field_battle_data_transfer.c:264`, from `MapHeader_GetMapEvolutionMethod`, `src/map_header.c`). Data: `Learnsets.evolutions` (`src/nuzline/data/learnsets.py:98`).
- **G9 On evolution.** Species set, ability from the new species and personality bit 0 (`src/evolution.c:435-436`; `src/pokemon.c:4746-4763`), stats recalculated with HP raised by the max-HP gain (`src/evolution.c:437`; `src/pokemon.c:633-645`). The new species' entries of exactly the current level are offered (`src/evolution.c:461-480`; `src/pokemon.c:3950`). A Nincada evolution adds a Shedinja when the party has a free slot and the bag a Poke Ball, which it uses up (`src/evolution.c:723-740`). Held-item evolutions (`EVO_LEVEL_WITH_HELD_ITEM_DAY`, `_NIGHT`, and trade with held item) use up the held item: it is set to none (`src/evolution.c:782-787`).
- **G10 Nincada in Kaizo.** The Kaizo encounter data holds Nincada (290) only in Eterna Forest's Poke Radar slots: `fielddata/encountdata/pl_enc_data.narc` member 8 (`res/field/encounters/encounters.order` line 9, Eterna Forest), bytes 124-136 = `radarEncounters[0..3]` (`include/overlay006/wild_encounters.h:31-36`; read at `src/overlay006/wild_encounters.c:696-699`). It is obtainable, so the Shedinja case stays. Gift, trade, honey tree and egg sources were not scanned.

### Decisions

- **In-fight learn answer** (G1, G5): replace slot 0-3 or give up. Stage B: the existing learn decision (`search/stage_b.py`; paused prompt `learn_site`, `engine/state.py:625`). Playouts: the growth plan.
- **Between-fight learn answer** (G5 after Rare Candies, G9 after evolution): same choices, from the growth plan.
- **Evolve or cancel** (G6, G7): per eligible living member per prompt. A dead member's answer is cancel and changes nothing.
- **Shedinja** (G9, G10): added only with `shedinja = allowed`, a free slot and a Poke Ball; otherwise the plan does not count it.
- **Pre-fight growth** (single battles and the first fight): Rare Candy levels to the cap (`prep/leveling.py`) with every prompt answered; stone use where the bag holds the stone.
- **Growth plan** (root, per member): one answer per offered move and an evolution gap (eligible gap or never). It holds in every playout.
- **Base plan** (funnel): learn into an empty slot, else give up; never evolve. This is today's playout behaviour (`search/learn.py:29-33`, `GIVE_UP`).
- **Time of day:** day and night evolutions are offered only with `time` set.
- **Single battle, post-fight evolution.** Evolution runs after the battle ends (G6), so it never changes the fight just planned. With `--next <trainer-id>`, both answers are scored by the single-fight screen against that trainer and the better is reported. Without `--next` the report recommends cancel, because a cancelled evolution is offered again at the next level-up (G6, G7). The report always says an evolution cannot be undone.

### Bound

- In-fight prompt: up to 5 answers; already a Stage B max node.
- Learn plan per member: 5^m for m offered moves before the last cap. Evolution per member: G_e + 1 options, x5 per post-evolution offered move. Per setup: the product over members.
1. **Commit per fight.** Only the current fight's prompts and the evolution prompts after it are committed. The current fight's prompts stay exact Stage B nodes and are never capped.
2. **Never-better pruning (exact), before the cap.** Give up only, for an offered move whose effect is unsupported (masked, D10) or that has the same effect id, type and category as a known move with power, accuracy and PP no higher. Evolve dominates cancel only when the evolved form is the same in every battle rule (`growth.evolve_dominates`): the same types, ability and species ability pair (`trainer_ai.c:1191-1192`); equal base stats (a rise is not provably better: Trick Room, Gyro Ball, the confusion self-hit and the AI's damage scores read them); the same gender for the personality (`src/pokemon.c:2696-2705`), the same experience group and the same Low Kick / Grass Knot weight bracket (`battle_script.c:6808-6826`); neither species is species-keyed (held items such as Lucky Punch, form changes, Chatter, Burmy); the method uses up no held item and adds no Shedinja; and every offer `(level, move)` of the cancel path is offered at the same level on the evolved path. Then only the earliest eligible gap is kept.
3. **Cap.** When the non-dominated plans per setup number at most `--growth-cap` (default `MAX_GROWTH_PLANS = 64`, one constant; `-1` = none), all are screened. Above it, the root enumerates only the current fight's prompts and the evolution after it; each later prompt takes the member's best answer by single-fight screen. The report says plainly that options were removed, gives `<kept> of <total>`, and names `--growth-cap -1`.

### Owners and status

Owners: E1 = abilities, items, conditions, AI flags; E2 = doubles and state shape; E3 = move effects, gauntlet and prep.

| Piece | Status at 8b89ca3c | Files | Owner |
|---|---|---|---|
| EXP payout, EVs, level-up stats and HP, friendship in battle | modelled | `engine/levelup.py` (`pay_exp` :145, `pay_exp_leveled` :151) | E3 consumer; exp items E1 |
| Learn prompt in battle | modelled | `engine/levelup.py` (`needs_learn` :292, `advance_learns` :400, `learn_options` :430, `learn` :465); `engine/state.py:614-615`, `:625` | E2 for the fields |
| Mid-turn learn prompt at its payout | modelled (E3-B2-21); E2 SF3 adds the Pursuit and switch sites | `search/stage_b.py`, `engine/state.py:625` | E3, E2 |
| Learn answer in playouts | always give up | `search/learn.py:29-33` | E3 (GT-16) |
| Pre-fight Rare Candy levels | modelled; never evolved | `prep/leveling.py:21` | E3 |
| Evolution check, conditions, Everstone, apply, post-evolution learn, Shedinja | missing; coverage says never modelled (`engine/coverage.py:1115`) | `prep/growth.py` (new) | E3; tell E1 (Everstone, evolution held items) |
| Battle-start friendship for Gym, E4, Champion | missing | `prep/states.py` callers | E3 |
| Field item effects (medicine, PP items, Rare Candy, stones) | missing; `Resources` holds Heart Scales, TMs, HMs and holdable items only (`prep/problem.py:51-58`, `:86-97`) | `tools/build_tables.py`, `prep/problem.py`, `prep/gauntlet.py` | E3 |

## 7. Carried-field list (for E2 sign-off)

Shape: in the engine, P is the party axis and B the battler axis (`engine/state.py:392`, `:866-871`; `docs/doubles-state-design.md` section 4). Only party 0 carries: experience goes only to party 0 (G2), and every tag fight rebuilds the partner's party from trainer data (`src/encounter.c:753` sets `trainerIDs[BATTLER_PLAYER_2]`; `Trainer_Encounter` builds a party per set id, `src/trainer_data.c:25-47`). `Party_HealAllMembers` heals only the save party (`src/encounter.c:198`). So `Carry` holds one entry per save-party slot: `[6]` per field (`[6, 4]` for moves), plus a leading batch axis named `R` on the host path.

The battle reaches the party only through `BtlIOCmd_UpdatePartyMon` (`src/battle/battle_io_command.c:575-608`, from `BattleController_EmitUpdatePartyMon`, `src/battle/battle_controller.c:1462-1505`), plus the EXP task and the level-up path (`bs.c:9767-9779`, `:9857-9867`, `:10007`). At exit the battle party goes to the save (`battle_main.c:703-705`; `src/encounter.c:151-158`).

`carry_out(tables, start: Carry, slot_of, terminal) -> Carry`. `start` is the fight-start carry; `slot_of[s]` gives the carry slot of engine party-0 slot s (section 5). Every carry slot is written from engine slot `slot_of^-1`; the whole save party is in the battle, so no member is dropped. A battler is "on s" when it is a party-0 battler with `active[b] == s`; the read loops over battlers, never "battler 0".

| Field | Source | Rule | Decomp cite |
|---|---|---|---|
| species | `start` | never written by the battle | `battle_io_command.c:581-604` (fields written) |
| form | `start`, except the written changes: Shaymin Sky -> Land, Giratina Origin -> Altered (`state.form`); Burmy by terrain if it took part | Castform, Cherrim, Arceus revert | `battle_io_command.c:597-599`; `battle_controller.c:1489-1499`; `battle_lib.c:6305-6400`; `battle_system.c:926-965` |
| ability | `party_ability` where set (`state.py:422`), else from species and personality | battle changes (Skill Swap, Worry Seed, Gastro Acid) revert | `battle_io_command.c:601-604` |
| level, exp, evs | `state.level[0, s]`, `state.exp[s]`, `state.evs[0, s]` | persist | `bs.c:9767-9779`, `:9857-9867` |
| stats, max_hp | recomputed from species, level, EVs, IVs, nature | not copied from battlers (Transform holds the target's) | `src/pokemon.c:633-645` |
| hp | `state.hp[0, s]` | persists; 0 when fainted | `battle_controller.c:1470`; `battle_io_command.c:594` |
| status | `state.status[0, s]`; 0 when fainted | persists | `battle_controller.c:1482-1486` |
| status_counter | sleep turns kept; Toxic counter 0 (Toxic bit kept) | | `battle_controller.c:1482`; restart at `bcp.c:1411-1415` |
| moves, pp | `state.moves`, `state.pp`; while a battler on s is `transformed`, all four from `shadow_moves` / `shadow_pp`; Mimic slots (`mimic_slots`) from the shadows | Transform skips the whole move block; Mimic slots are skipped | `battle_io_command.c:581-588`; `state.py:534-542` |
| pp_ups | `state.pp_ups[0, s]` | never written by the battle, except that a move learned in battle sets its slot to 0 | `battle_io_command.c:581-604`; `bs.c:10170` |
| max PP | not carried | derived from move and PP Ups | `MoveTable_CalcMaxPP`, `src/pokemon.c:961-965`; the setter ignores it, `:1519-1523` |
| held item | `party_held_item(state, 0, s)` (E2 D3-8b) | the party's item after the battle | `battle_io_command.c:590-592` |
| friendship | `party_friendship[0, s]` (`state.py:417`) | persists; the party value wraps past 255 | `bs.c:9857`; faint `bs.c:12080-12094`; `src/pokemon.c:1859-1874` |
| party order | `start` | in-battle switches never reorder the party: `partyOrder` is a BattleContext index map (`battle_lib.c:6465-6507`), and the party is copied back slot by slot | `battle_main.c:703-705` |
| dead | `start.dead` OR `state.deaths[s]` | `state.deaths` means fainted in this battle (`engine/state.py:617`); earlier deaths come from `start`. Out of the pool per `revived_dead` | |
| levelled | level at end > level at start | the `leveledUpMonsMask` bit | `bs.c:9867` |
| pending learn work | `learn_level`, `learn_cursor` | answered by the growth plan before the carry (G1) | `bs.c:10013` |
| personality, ivs, nature, gender, beauty, ball, egg location | constant, from the save | | |
| pokerus | from the save | the end-of-fight spread is random and only adds EVs; not modelled (under-estimates) | `bcp.c:4041-4048` |

Not carried (reset at every fight start): volatiles, stat stages, types, battle forms above, abilities changed in battle, the Toxic counter.

Both engines: the reference state type differs (`RefState`; `reference/engine.py:2145` takes it), so `carry_out` has a reference twin that calls the reference `party_held_item`; a differential test compares them.

## 8. Coverage and backlog

- The gauntlet refusal (`src/nuzline/engine/coverage.py:991-998`, `_check_kaizo_fields`) becomes a note line: fight i of n, the map facts with sources (`orders`, `leave_gaps`, `escape`, `pc_gaps`, `heal_gaps`, `locked_gaps`, `partner`).
- No `Support.NONE` row for gauntlet facts: the loader refuses an option without a transition (section 3), so no stored fact can lack one. Settings are never a coverage row.
- A fight is supported only when every remaining fight of its chain is, in every form the facts allow (Wayward: multis, and 459, 460 as singles beside the partner). A later fight's refusal is listed as `chain: <fight>: <reason>` under that fight's own blocker group, so it is not counted twice.
- The evolution line (`coverage.py:1115`) is replaced when evolution lands.
- Backlog (`tools/kaizo/backlog.py`, regenerated): the 33 per-position gauntlet rows go to 0; 20 battles leave; 7 stay under their E2 format blockers; the 5 E4 + Champion fights enter as members with no new blocker (B4-19 adds the group).

## 9. State and doubles-readiness

- `BattleState` does not change in this batch. The Carry, the transition, `barred` and the learn answers are prep and search inputs.
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
| GT-1 | `data/gauntlet.py`: `OPTIONS`, kinds, per-gap values and caps, precedence, defaults; `GauntletFacts` replaces `Trainer.gauntlet_rules`; `Trainer.gauntlet` readers | 1, 3 | all |
| GT-2 | Group facts: `orders` as fight lists, `leave_gaps`, `escape`, `pc_gaps`, `heal_gaps`, `locked_gaps`, `partner`, sources; `tools/kaizo/gauntlet_map.py` (the map read, with arrival check, doors, tables, coords); loader checks; E4 + Champion facts (`league_facts`; the group entry moved to B4-19); `rules`/`RULE_KEYS` removed; table field; vanilla and Kaizo rebuild | 1, 2 | all |
| GT-3 | Settings: flags, `--rules FILE`, `data/local.toml [run]`; `tools/rom/inputs.py` skips `[run]`; `data/local.example.toml` `[run]` block; re-run inputs `--dead`, `--swaps-left`, `--bench`; errors; report of origins | 3 | all |
| GT-4 | `carry_out` per section 7, calling `party_held_item`; reference twin | 7 | all |
| GT-5 | Fight start: `slot_of`, lead rule, `barred`, format per fight, `write_carry`; re-run root from the save with `--dead`, deaths left and root sit-outs; battle-start friendship; level-cap sit-out | 3, 5, 7 | all |
| GT-6 | Gap transition steps 0-9; item field-effect table from the ROM item params (F2); `Resources` gains medicine, PP items, Rare Candies, stones; field poison | 5 | all |
| GT-7 | Swaps: bench, budget, plans (`MAX_SWAP_PLANS = 64`), deposit and withdraw effects, order change, policy swap | 3, 5 | user swaps |
| GT-8 | Chosen order: next opponent within `orders`; `worst` (default) = minimum over every open combination, `best` = search picks, `MAX_ORDERS = 24` with greedy beyond under `best` only | 5 | Route 213, Building #1, Wayward Cave |
| GT-9 | Chain search: host-side chain playout, `batched_playout_states`, chain score, funnel / ladder / Stage B on it, Stage B across 296/297, `--horizon` | 5 | all |
| GT-10 | Lookahead policy (bag top-up incl. PP, PC, reorder lead, planned and policy swaps, cap sit-out) | 5 | all |
| GT-11 | Report and record: settings with origin, locked gaps and partner heal with sources, chain cost | 3, 5 | all |
| GT-12 | Evolution conditions (G8), Everstone, `time`, map method reader | 6 | chains, singles |
| GT-13 | Evolution applied (G9), post-evolution learn, Shedinja per setting | 6 | same |
| GT-14 | Evolution prompts: after a won fight (G6, dead members included), Rare Candy (cancellable, revive, field friendship), stones; pre-fight evolution in `prep/leveling.py` | 6 | same |
| GT-15 | Growth plan, never-better pruning, `--growth-cap` with the plain report line | 6 | same |
| GT-16 | Playout learn answer from the growth plan (`search/learn.py:29-33`) | 6 | same |
| GT-17 | Single battle post-fight evolution: `--next`, recommend cancel without it, "cannot be undone" | 6 | singles |
| GT-18 | Coverage and backlog: note line, `chain:` lines, Wayward forms, evolution line, regenerate (33 -> 0) | 8 | all |

Owners named by the 18 Nov rulings: playout learn answer GT-16 = E3; evolution GT-12..14 = E3 (tell E1 about Everstone and evolution held items, read only by `src/pokemon.c:3543-3547`); species and ability in the carry GT-4 = E3, field list signed off by E2 (section 7). GT-4 and everything that needs it (GT-5 except the friendship part, GT-6, GT-7, GT-9, GT-10) wait until E2's D3-8b is on main.

## 11. Files touched

| File | Change |
|---|---|
| `docs/gauntlet-design.md` | this design |
| `src/nuzline/data/gauntlet.py` | new: GT-1, GT-3 loader |
| `src/nuzline/data/trainer.py:254-255`, `:347-348` | `gauntlet` = (group, fight id); `gauntlet_rules` -> `gauntlet_facts` |
| `src/nuzline/cli.py:1637-1639` | prints the fight id |
| `tools/rom/inputs.py:64-79` | `_read_toml` skips the `run` table |
| `data/local.example.toml` | documented `[run]` block, strictest values |
| `tools/kaizo/overrides.py` | `RULE_KEYS` :71 removed; `Gauntlet` :137-141 gains the fact keys; `_gauntlets` :224-; drafting |
| `tools/kaizo/resolve.py:522-527`, `:592-596` | fight numbering from `orders`; pair check after `trainer_record`; copy facts only |
| `tools/kaizo/gauntlet_map.py` | new: the read-only map read |
| `data/kaizo/overrides.toml:478-550` | fact keys with sources; `rules` removed; E4 + Champion group (B4-19) |
| `tools/build_tables.py` | trainer field; item field-effect table; MANIFEST_VERSION bump; vanilla and Kaizo rebuild |
| `src/nuzline/engine/coverage.py` | `_check_kaizo_fields` :991-998; evolution line :1115 |
| `tools/kaizo/backlog.py`, `docs/kaizo/mechanics_backlog.md` | `chain:` lines; regenerate |
| `src/nuzline/prep/gauntlet.py` | new: Carry, `carry_out`, `write_carry`, transition, swaps, order, chain playout |
| `src/nuzline/prep/growth.py` | new: evolution, growth plans, pruning |
| `src/nuzline/prep/leveling.py:21` | pre-fight evolution |
| `src/nuzline/prep/states.py:38` `make_states` | callers add battle-start friendship; template for `write_carry` |
| `src/nuzline/prep/problem.py:51-58`, `:398` | `Resources` item kinds; `build_problem` settings, `bag_uses`, bench pool, re-run root from the save |
| `src/nuzline/prep/setups.py`, `screen.py`, `search.py` (`PrepSettings` :153, `_Funnel` :325, `run_prep` :901) | chain score; setup gains bench, swap plan, growth plan, next opponent |
| `src/nuzline/prep/report.py` (`render` :361, `record` :449) | GT-11, GT-15, GT-17 lines |
| `src/nuzline/prep/caps.py` | next fight's cap |
| `src/nuzline/search/learn.py:29-33` | `learn_policy` reads the growth plan |
| `src/nuzline/search/playout.py`, `search/stage_b.py` | `batched_playout_states`; `barred` input; Stage B across 296/297 |
| `docs/kaizo/e3-ledger.md` | rulings |
| tests, new | `tests/data/test_gauntlet_settings.py`, `tests/prep/test_gauntlet*.py`, `tests/prep/test_growth.py`, `tests/kaizo/test_gauntlet_map.py`; synthetic groups and settings only, no test pins a user choice |
| tests changed by the schema | `tests/data/test_trainer.py:207-209`, `:302-303`, `:319`; `tests/engine/test_coverage.py:373-392`; `tests/kaizo/test_overrides.py:51`, `:124`, `:131-136`; `tests/kaizo/test_resolve.py:583-585`; `tests/tools/rom/test_inputs.py` |

No change to the JAX step or the reference engine's battle rules. `levelup.py` is read, not edited.

Tests per run: the four guards, `tests/test_differential.py`, the prep, growth and map tests, and the coverage, overrides, resolve, inputs and trainer tests above. The tables change, so the rebuild runs first and the `fullset` tests run on the pod. Tests that need the ROM (map facts, item table) are marked `slow` (the existing marker for ROM-bound tests, `pyproject.toml:39`); the pod holds no ROM, so the stream runs them locally with the ROM once per change to GT-2 or the item table, under the local test rules (CLAUDE.md).

## 12. Overlaps and merge order

| With | Overlap | Resolution |
|---|---|---|
| E2 D3-8b (7f391925 on `e2-d3`) | `party_held_item(state, party, slot)`: JAX `engine/state.py:1017`, reference `reference/engine.py:2145`; new `knocked_item` field | `carry_out` calls it directly. GT-4 starts after D3-8b is on main. Any open ruling on items after Trick, Switcheroo, Thief, Covet or Knock Off is E2's inside D3-8b; this design reads the helper and depends on no outcome. |
| E2 D3-8 (`pay_exp_leveled(battler=None)`, `step._pay_all`) | EXP, level and levelled slots read by the carry | read only; no edit |
| E2 SF3 (after the merge train) | new `learn_site` values for the Pursuit and switch payouts | playout, Stage B and growth code handle every `learn_site` value generically, never by an enumerated list |
| E2 coverage (`FOUR_TURN_CLAUSES`) | `coverage.py` | different functions; text merge |
| E2 D11 (search, prep and CLI with two actives) | prep and search files of this batch | D11 starts after E3 batch 3 merges and extends it |
| E1 batch H (Exp. Share, Lucky Egg, Soothe Bell) | holders gain EXP and friendship; benched Exp. Share holders level and become evolution-eligible | no shared file; any order; E3 tests of item holders evolving go after H, or use plain participants |
| E1 (Everstone, evolution held items) | read only by evolution | built in `prep/growth.py`; tell E1 |
| E1 (party friendship wrap) | carry keeps the party value | `carry_out` reads `party_friendship` |
| E1 (Weather Ball byte) | per-battle state | E1's; nothing here holds or carries it |

Merge order: E2 D3-8b, then E3 batch 3, then E2 D11. E1 batch H in any order.

## 13. Defaults policy

- Unset = the strictest value of each option, as listed first in section 3: the value whose score is a lower bound under every user rule. For a choice the game leaves open (`order`) that is `worst`: the minimum over every order the game allows, with the worst one reported. The game leaves bag healing and Rare Candies open too, and those default to their strictest value; orders are the player's choices in the same way. `best` (the search picks) and a fixed order are settings the user names. When Mira joins is not a choice: the game makes her join before fight 1 (Wayward Cave facts). `time` has no default: unset offers no day/night evolution. The report says when a default was used.
- These are code policy, not anyone's preferences. The user's own choices go in `data/local.toml [run]` (Nuzlocke rules) and in their rules files or flags (gauntlet options); nothing tracked holds them.
- Fixed by policy, not settings: Ethers and Elixirs are bag items like any other (`bag_uses`); Route 213's healer is a map fact, used only under `heal = full`; the Shedinja case stays because Nincada is obtainable (G10); a walk-out gap allows what the map outside allows.

## 14. Game data vs user belief

Where the map data and the user's record (the sheet) differ, the data is the game fact and the user's rule is a setting on top (`order`, `heal`).
- Route 204 North: Ravaged Path (header 254) is reachable on foot after fights 15 and 13 (rom header 346, warp (180,698)).
- Oreburgh Mine: 195 is fought first, not the sheet's 196 (Map reads).
- Wayward Cave: 450+496 come before 459+460 (Wayward Cave facts).
- Wayward Cave: the sheet records six multis. The game also allows 459, 460 as two singles beside Mira (459 talked to alone, Wayward Cave facts); that is a run choice (`order`), not a contradiction. No fight can be played before Mira joins.
- Eterna Building #3: v5.1 said "free order; leave after 422 first". With the arrival check the order is 412, 422 only and there is no leave gap. This is the sheet's order; the old fact came from the map-read bug, not from a user belief.
