# Decomp source locations

Where each piece of game data and battle logic lives in the
[`pret/pokeplatinum`](https://github.com/pret/pokeplatinum) decompilation.
Every extraction tool and every decomp-derived test reads this file.

- **Pinned commit:** `579d2d5c65e736a1a4268fb7acbfd70712177c6c`
  (`build: Feed the AI script through enumproc (#1315)`). `bash tools/clone_decomp.sh`
  clones it into `external/pokeplatinum`, which is gitignored.
- All paths below are relative to `external/pokeplatinum`. Line numbers hold
  at the pinned commit only. Re-check them if the pin moves.
- You do not need to build the ROM. Every table below is either a source JSON
  file or a plain-text constant list under `generated/`.
- **Id convention.** Each `generated/<name>.txt` file lists one constant per
  line. A constant's numeric id is its 0-based line number, so line 1 is id 0
  (`SPECIES_NONE`, `MOVE_NONE`, `ITEM_NONE`, `ABILITY_NONE`, `TRAINER_NONE`).
  Each per-entry JSON directory or file takes the constant's name, minus its
  prefix, in lowercase: `SPECIES_NIDORAN_F` -> `res/pokemon/nidoran_f/`,
  `MOVE_ROCK_THROW` -> `res/moves/rock_throw/`, `ITEM_SITRUS_BERRY` ->
  `res/items/data/sitrus_berry.json`, `TRAINER_LEADER_ROARK` ->
  `res/trainers/data/leader_roark.json`.
- **Stat order.** In memory the decomp orders stats HP, Atk, Def, **Spe**, SpA,
  SpD (see `generated/pokemon_stats.txt`, `generated/battle_stats.txt` and the
  field order of `BattleMon`). Our tables use HP, Atk, Def, SpA, SpD, Spe
  (index = `ids.Stat`), so the extractor reorders them.
- **Struct rows.** Each of the three "Battle-state struct" rows has exactly one
  header path in the *Path* column. The struct name is the first backticked
  token in the *Format* column. Tests rely on this layout to parse the rows.

| What | Path in external/pokeplatinum | Format | Notes |
|---|---|---|---|
| Species data (base stats, types, abilities, gender ratio, exp rate, weight) | `res/pokemon/<species>/data.json`; ids in `generated/species.txt` | JSON per species: `base_stats` {hp, attack, defense, speed, special_attack, special_defense}, `types` [2 x `TYPE_*`], `abilities` [2 x `ABILITY_*`; the second may be `ABILITY_NONE`], `gender_ratio`, `exp_rate`, `pokedex_data.weight_pounds`, `pokedex_data.en.name` | 496 constants: `SPECIES_NONE` (0), the 493 species, `SPECIES_EGG` and `SPECIES_BAD_EGG`. A mono-typed species lists its type twice. Alternate forms live in `res/pokemon/form_registry.json` (M1 does not need them). |
| Move data | `res/moves/<move>/data.json`; ids in `generated/moves.txt` | JSON per move: `name`, `class` (`CLASS_PHYSICAL`/`CLASS_SPECIAL`/`CLASS_STATUS`), `type`, `power`, `accuracy`, `pp`, `effect.type` (`BATTLE_EFFECT_*`), `effect.chance`, `range`, `priority`, `flags` | 469 constants, including `MOVE_NONE`. Every move's `script.s` contains only `GoToEffectScript`, so `effect.type` alone selects the behaviour. In battle the data is read through `MOVE_DATA(move)` (`include/battle/common.h:25`). |
| Type chart | `src/battle/battle_lib.c:2399` (`sTypeMatchupMultipliers`); type ids in `generated/pokemon_types.txt` | C array of {attacking type, defending type, multiplier} rows. The multiplier is `TYPE_MULTI_IMMUNE` 0, `TYPE_MULTI_NOT_VERY_EFF` 5 or `TYPE_MULTI_SUPER_EFF` 20, in tenths (`include/constants/battle.h:144`). Any pair not listed is neutral | 18 type ids, including `TYPE_MYSTERY` (???) at id 9, so the dense chart is 18x18. The rows after the `0xFE` separator (the Normal->Ghost and Fighting->Ghost immunities) are skipped under Foresight or Scrappy. `0xFF` ends the table. `BattleSystem_ApplyTypeChart` (`battle_lib.c:2560`) applies the chart once per defender type and skips type 2 when it equals type 1. |
| Natures | `src/pokemon.c:2415` (`sNatureStatAffinities`); ids in `generated/natures.txt` | C array `s8[25][5]` of +1/0/-1 per stat, in the order Atk, Def, Spe, SpA, SpD | `Pokemon_GetNatureStatValue` (`pokemon.c:2593`) applies it as x110/100 or x90/100 with integer division. Nature = `personality % 25` (`Pokemon_GetNatureOf`, `pokemon.c:2410`). |
| Items | `res/items/data/<item>.json`; ids in `generated/items.txt`; hold-effect ids in `generated/item_hold_effects.txt` | JSON per item: `name`, `holdEffect` (`HOLD_EFFECT_*`), `effectParam` (hold-effect power), `naturalGiftPower`/`naturalGiftType`, `flingPower`, `teachesMove` (TMs/HMs) | Sitrus Berry is `HOLD_EFFECT_HP_PCT_RESTORE` with param 25 (25% of max HP). Oran Berry is `HOLD_EFFECT_HP_RESTORE` with param 10. In-battle triggers: `BattleSystem_TriggerHeldItem` (`battle_lib.c:4576`), `BattleSystem_TriggerHeldItemOnStatus` (`battle_lib.c:4967`), `BattleSystem_TriggerLeftovers` (`battle_lib.c:4922`). |
| Abilities | ids in `generated/abilities.txt`; names in `res/text/ability_names.json` | Plain constant list (124 entries, including `ABILITY_NONE`). The names JSON has `messages[i].en_US`, indexed by ability id | Abilities exist only as code, spread across `src/battle/battle_lib.c`: switch-in effects in `BattleSystem_TriggerEffectOnSwitch` (`:3649`), damage modifiers inside `BattleSystem_CalcMoveDamage` (`:6601`), status immunity in `Ability_ForbidsStatus` (`:4454`). When a species has a second ability, `personality & 1` picks the slot (`sub_02073E18`, `src/pokemon.c:405`). |
| Level-up learnsets | `res/pokemon/<species>/data.json`, key `learnset.by_level` | JSON list of `[level, "MOVE_*"]` in learn order | A freshly created mon's default moveset is the last four distinct moves learned at or below its level (`BoxPokemon_SetDefaultMoves`, `src/pokemon.c:3819`). Tutor moves are under `learnset.by_tutor`. |
| TM/HM compatibility | `res/pokemon/<species>/data.json`, key `learnset.by_tm`; TM->move map in `res/items/data/tm01.json` ... `hm08.json`, key `teachesMove` | JSON list of `"TMnn"`/`"HMnn"` strings. Each TM item JSON names the move it teaches | `ITEM_TM01` is item id 328 and `ITEM_HM01` is id 420 (`generated/items.txt`). |
| Trainer parties | `res/trainers/data/<trainer>.json`; ids in `generated/trainers.txt`; binary layout in `include/struct_defs/trainer_data.h` | JSON per trainer: `name`, `class`, `items` (bag items the AI may use), `ai_flags`, `double_battle`, `party` [{`species`, `form`, `level`, `item`, `moves` (may hold fewer than 4 or be absent), `iv_scale` 0-255, `ball_seal`}] | Roark is `res/trainers/data/leader_roark.json`, `TRAINER_LEADER_ROARK`, id 246: Geodude L12 (Stealth Rock, Rock Throw), Onix L12 (Stealth Rock, Rock Throw, Screech), Cranidos L14 (Headbutt, Pursuit, Leer). All three have `iv_scale` 50 and no held item. His bag holds 2x Potion. His AI flags are BASIC, EVAL_ATTACK and EXPERT. |
| Trainer AI flags and scripts | Flags: `ai_flags` in the trainer JSON, ids in `generated/ai_flags.txt`. Script: `src/battle/trainer_ai/script.s`. Interpreter: `src/battle/trainer_ai/trainer_ai.c`. Opcodes: `asm/macros/aicmd.inc`, `include/data/scripts/aicmd.h`. AI state: `include/battle/ai_context.h` | The flags form a bitmask: bit n is line n+2 of `ai_flags.txt`, and `AI_FLAG_NONE` on line 1 is not a bit. `script.s` opens with `FlagTable`, one label per flag bit (`Basic_Main`, `EvalAttack_Main`, `Expert_Main`, ...), followed by bytecode with named labels written in `aicmd.inc` macros | Call chain: `TrainerAI_Main` (`trainer_ai.c:258`) -> `TrainerAI_MainSingles` (`:286`) -> `TrainerAI_EvalMoves` (`:483`). That last step runs each set flag's script on every move slot and adds the results to the move scores. Switch and item decisions: `TrainerAI_PickCommand` (`:3989`), `TrainerAI_ShouldSwitch` (`:3894`), `TrainerAI_ShouldUseItem` (`:4056`). AI damage estimate: `TrainerAI_CalcDamage` (`:2868`). The AI is called from `src/battle/battle_display.c:3411` and `:3590` and draws from the battle RNG (`BattleSystem_RandNext`). |
| Function that builds a trainer's party (IVs from difficulty, PID) | `src/trainer_data.c:176` (`TrainerData_BuildParty`) | C | For each mon: `LCRNG_SetSeed(iv_scale + level + species + trainerID)`, then call `LCRNG_Next()` once per unit of the trainer class id (`header.trainerType`). PID = `(rnd << 8) + genderMod`, where genderMod is 120 for female trainer classes and 136 otherwise (`TrainerClass_Gender`, `:164`; table in `include/data/trainer_class_genders.h`). All six IVs = `iv_scale * 31 / 255` (`MAX_IV_SCALE` 255, `include/struct_defs/trainer_data.h:15`), and EVs are 0. `Pokemon_InitWith` (`src/pokemon.c:381`) calls `sub_02073E18` (`:405`), which sets the IVs, the ability (from PID bit 0), the gender (from the PID) and the default moves. Any moves listed in the JSON then overwrite those slots. For Roark this gives IVs of 50*31/255 = 6. |
| Battle-state struct (per-Pokémon) | `include/battle/battle_mon.h` | C struct `BattleMon` (nests `MoveEffectsData`) | Holds the species, the five non-HP stats (in Atk, Def, Spe, SpA, SpD order), moves, IVs, `statBoosts[8]` (HP, Atk, Def, Spe, SpA, SpD, Acc, Eva; 6 = neutral), types, ability, PP, level, `curHP`/`maxHP`, `status` (non-volatile; bits in `include/constants/battle/condition.h`), `statusVolatile`, held item, and `moveEffectsMask` + `moveEffectsData` (counters for Disable, Encore, Taunt, the Protect chain, Substitute HP, ...). `BattleSystem_InitBattleMon` (`src/battle/battle_lib.c:70`) builds it from the party mon. Field accessor ids are in `generated/battle_mon_params.txt`. |
| Battle-state struct (per-side) | `include/battle/battle_context.h` | C struct `SideConditions` | Holds turns and users for Reflect, Light Screen, Mist and Safeguard, plus Follow Me, the knocked-off item mask, and the Spikes and Toxic Spikes layer counts. The side's on/off flags, including `SIDE_CONDITION_STEALTH_ROCK` (`include/constants/battle/condition.h:95`), are stored separately in `BattleContext.sideConditionsMask[2]`. |
| Battle-state struct (field/context) | `include/battle/battle_context.h` | C struct `BattleContext` (embeds `FieldConditions`, `SideConditions[2]`, `TurnFlags`, `SelfTurnFlags`, `MoveFailFlags`, `BattleMon battleMons[4]`) | Field-wide state: `fieldConditionsMask` (weather, Trick Room, Gravity, fog; bits in `condition.h`) and `FieldConditions` (weather turns, Future Sight, Wish). It also tracks the turn (`totalTurns`, action order, `moveLockedInto`, `movePrevByBattler`, scratch damage values). Most of the remaining fields belong to the script interpreter, IO and UI. |
| Damage calculation function | `src/battle/battle_lib.c:6601` (`BattleSystem_CalcMoveDamage`) | C | Base damage is `A*stage * power * (L*2/5+2) / D*stage / 50` (`:6947`-`:6976`). The code divides by D before 50; for non-negative integers that equals `.. // 50 // D`. Burn halving, screens, weather and Flash Fire follow, and `+ 2` comes last (`:7078`). A critical hit ignores the attacker's negative stages and the defender's positive ones. Simple doubles the effective stage at calc time, clamped to ±6 (`:6850`-`:6882`). The surrounding pipeline runs in this order: (1) crit roll, `BattleSystem_CalcCriticalMulti` (`:7105`; rates in `sCriticalStageRates`, `:7097`). (2) `BattleScript_CalcMoveDamage` (`src/battle/battle_script.c:1323`) multiplies by `criticalMul` and applies Life Orb and Metronome. (3) random roll, `BattleSystem_CalcDamageVariance` (`battle_lib.c:7081`): `damage * (100 - rand%16) / 100`, minimum 1. (4) STAB x15/10, then the type chart for each defender type, in `BattleSystem_ApplyTypeChart` (`battle_lib.c:2560`). Step 4 runs in `BattleControllerPlayer_TryMove`, state `TRY_MOVE_STATE_CHECK_TYPE_CHART` (`src/battle/battle_controller_player.c:3272`), after the move script has computed the damage. Stage ratios are in `sStatStageBoosts` (`battle_lib.c:6550`). |
| Stat calculation function | `src/pokemon.c:564` (`Pokemon_CalcStats`) | C | HP = `(2*B + IV + EV/4) * L / 100 + L + 10` (Shedinja always 1). Every other stat = `(2*B + IV + EV/4) * L / 100 + 5`, then the nature's x110/100 or x90/100. In battle, stat stages are applied when a stat is used and are never written back into the stat. |
| End-of-turn effect order | `src/battle/battle_controller_player.c`: `enum FieldCondCheckState` (`:897`), `enum MonCondCheckState` (`:1256`), `enum SideCondCheckState` (`:1736`) | C state machines, run in the order `BattleControllerPlayer_CheckFieldConditions` (`:932`) -> `CheckMonConditions` (`:1289`) -> `CheckSideConditions` (`:1746`) -> `TurnEnd` (`:1844`) | Field checks: Reflect, Light Screen, Mist, Safeguard, Tailwind, Lucky Chant, Wish, rain, sandstorm, sun, hail, fog, Gravity. Mon checks run one battler at a time in speed order (`monSpeedOrder`), finishing every check for a battler before moving on: Ingrain, Aqua Ring, ability (e.g. Speed Boost, Rain Dish), held-item use, Leftovers, Leech Seed, poison, toxic, burn, Nightmare, Curse, bind, Bad Dreams, Uproar, Thrash, Disable, Encore, Lock-On, Charge, Taunt, Magnet Rise, Heal Block, Embargo, Yawn, item status cure, detrimental items. Side checks: Future Sight, Perish Song, Trick Room. Faint checks run between steps through `BattleControllerPlayer_AnyFainted`. |
| Move effect id list | `generated/move_battle_effects.txt`; scripts in `res/battle/scripts/effects/effect_script_NNNN.s`; subscripts in `res/battle/scripts/subscripts/*.s` | Constant list of 277 `BATTLE_EFFECT_*` entries (id = 0-based line). Effect script NNNN implements effect id NNNN, dispatched by `BattleScript_Jump(..., BE_SEQ, CURRENT_MOVE_DATA.effect)` (`src/battle/battle_script.c:2084`) | Example: id 0, `BATTLE_EFFECT_HIT`, is `CalcCrit; CalcDamage`. The script opcodes (`BtlCmd_*`) are implemented in `src/battle/battle_script.c`, with macros in `asm/macros/btlcmd.inc`. Stealth Rock damage on switch-in comes from `BtlCmd_CheckStealthRock` (`battle_script.c:8302`), called from `subscript_hazards_check.s`. |
| Overworld-to-battle weather/fog hand-off | `src/field_battle_data_transfer.c:267` (`dto->fieldWeather = FieldOverworldState_GetWeather(...)`) -> `src/battle/battle_main.c:1059` -> `BattleSystem_TriggerEffectOnSwitch`, state `SWITCH_IN_CHECK_STATE_FIELD_WEATHER` (`src/battle/battle_lib.c:3663`) | C | Each map's weather is the `.weather` field of its header in `include/data/map_headers.h`, using constants from `include/constants/overworld_weather.h`. Mapping: rain, heavy rain and thunderstorm -> rain; snow, heavy snow and blizzard -> hail; sandstorm -> sand; fog and deep fog -> fog; harsh sun -> sun; Trick Room weather -> Trick Room. Oreburgh Gym (`MAP_HEADER_OREBURGH_CITY_GYM`, `map_headers.h:1078`) is `OVERWORLD_WEATHER_CLEAR`, so Roark's fight starts with no weather. |
| Trainer locations and date-dependent weather (start conditions) | `res/field/events/<eventsArchiveID>.json` and `res/field/scripts/<scriptsArchiveID>.s` (the archive ids are fields of each map header in `include/data/map_headers.h`); `src/overlay005/vs_seeker.c` (`gVsSeekerRematchData`); `src/field_overworld_weather.c` (`sYearlyWeather`, `FieldSystem_GetWeather`) | Events JSON: `object_events[].script` is `TRAINER_*` for a trainer standing on the map. Scripts: `StartTrainerBattle`, `StartTagBattle`, `StartFirstBattle` name trainers directly or through a `SetVar VAR_*, TRAINER_*` line. Rematch rows list a base trainer then its `_REMATCH_n` ids (`_` = none). `sYearlyWeather` is 366 rows of 5 columns, one per `OVERWORLD_WEATHER_YEARLY_START + n` constant | A rematch is fought where its base trainer stands. Five maps have yearly weather: Route 212 South (rain except Apr 30), Route 213 (clear or cloudy except 10 rainy days), Route 216 and Acuity Lakefront (always snow, so hail), Snowpoint City (`OVERWORLD_WEATHER_HAILING` on 10 days, which battle code does not map to hail). Overworld fog is cleared when Defog is active (`src/field_map_change.c:275`). Trainers on the two date-dependent routes are resolved in `tools/start_conditions.json`. |
| Turn order (supporting) | `src/battle/battle_controller_player.c:707` (`BattleControllerPlayer_CalcTurnOrder`); `src/battle/battle_lib.c:1188` (`BattleSystem_CompareBattlerSpeed`) | C | Switches go before moves. Moves are then ordered by priority (`battle_lib.c:1397`), then Quick Claw / Custap / Lagging Tail / Stall, then Trick Room, then speed. Speed includes stages (doubled for Simple, `:1171`), is quartered by paralysis (`:1259`) and halved by Slow Start. A speed tie is broken by `BattleSystem_RandNext(battleSys) & 1`. |
| Accuracy check (supporting) | `src/battle/battle_controller_player.c:2865` (`BattleControllerPlayer_CheckMoveHitAccuracy`); table in `include/data/hit_rate_stages.h` (`HitRateByStage`) | C | Combined stage = attacker accuracy stage + (6 - defender evasion stage), doubled for Simple and clamped. `hitRate = accuracy * num / den`, followed by item, Micle and Gravity modifiers. The move misses when `(rand % 100) + 1 > hitRate`. Overrides such as Protect and No Guard are in `BattleControllerPlayer_CheckMoveHitOverrides` (`:3001`). |
| Battle RNG (supporting) | `src/battle/battle_system.c:1308` (`BattleSystem_RandNext`) | C | LCRNG on `battleSys->seedRandNext` that returns the high 16 bits. We do not replay seeds. `step` takes explicit outcome inputs instead. |

## Completeness

Checked at the pinned commit by searching for address-named symbols
(`sub_XXXXXXXX`, `ovNN_XXXXXXXX`, `unk_`/`Unk_`) and by reading the
functions listed above.

- **The battle engine is fully decompiled and named on the paths we use.**
  `src/battle/battle_lib.c` (8204 lines),
  `src/battle/battle_controller_player.c` (4862 lines),
  `src/battle/trainer_ai/trainer_ai.c` and `src/trainer_data.c` contain no
  address-named function definitions or calls. The one hit in
  `battle_lib.c` is `#include "unk_0208C098.h"`, a header whose named
  `HealthBar_Color` is used for cry modulation at `:6214`.
  `src/battle/battle_script.c` contains address-named calls in only two
  places: `BattleScript_CatchMonTask` (`:10354`; the ball-throw animation
  and Pokédex screen, `ov12_*`/`ov21_*`/`sub_02015738`) and
  `BattleScript_LoadPartyLevelUpIcon` / `BattleScript_FreePartyLevelUpIcon`
  (`:11952`/`:12039`; font OAM, `sub_020127*`/`sub_02012898`). Neither is on
  a damage, end-of-turn or AI path. In `src/battle/battle_system.c`,
  `sub_0202F868` / `sub_0202F884` appear only in battle-recording playback.
  The battle structs in `include/battle/battle_mon.h` and
  `include/battle/battle_context.h` have no `unk` fields, only explicit
  `padding*` bits.
- **The battle scripts are fully disassembled.**
  `res/battle/scripts/effects/*.s` and `res/battle/scripts/subscripts/*.s` are
  written in named `btlcmd.inc` macros. The subscript files have names, but
  the jump targets inside them use numeric local labels (`_000`, `_012`, ...).
  Those labels carry no meaning but hide nothing.
- **The AI script interpreter is fully decompiled and named.**
  `src/battle/trainer_ai/trainer_ai.c` (4199 lines) has no address-named
  symbols, and every opcode handler is an `AICmd_*` function.
  `src/battle/trainer_ai/script.s` (8108 lines) uses named macros and 1011
  named labels, with no numeric-only labels.
- **Unnamed functions on the damage, end-of-turn or AI paths:**
  - None on the damage path.
  - None on the end-of-turn path.
  - None in the AI interpreter or the AI script.
  - The trainer-party build path has one: `sub_02073E18`
    (`src/pokemon.c:405`), the `BoxPokemon` initializer called by
    `Pokemon_InitWith`. Only its name is missing. The body is fully
    decompiled C that sets the PID, IVs, ability slot (PID bit 0), gender
    and default moves.
- Two `TODO` notes remain on these paths. One concerns Me First in
  `BattleScript_CalcMoveDamage` (`battle_script.c:1355`; Me First is out of
  scope). The other is a field-rename note at
  `battle_controller_player.c:4110`. Neither hides any behaviour.
