# Tag Strategy (AI flag bit 7, `TagStrategy_Main`) at B = 2

Source: `src/battle/trainer_ai/script.s` (`s`), `src/battle/trainer_ai/trainer_ai.c` (`ai`), `battle_system.c`
(`sys`), relative to `external/pokeplatinum`. Code: `policy._tag_strategy` (`script.s:6637-7338`).
Rules sheet: `_scratch/e4/rules/A-8.md` (sections 1-4, 7). Commands: `docs/ai/script_commands.md`.
`R@L(N)`: `IfRandomLessThan N` at line L, jump w.p. N/256; the change below is made when it does NOT jump.

Scope. Exact at B = 2 (singles): 16 singles split rows carry bit 7 (`ai:297-309` runs it after bit 6, before
bit 8). A doubles battle adds bit 7 to every thinking mask (`ai:252-255`); B = 4 is refused: the format gate
(`coverage.IMPLEMENTED_FORMATS`: `double`, `double_two_trainer`, `multi`) and `policy._refused`
(a B = 4 state whose `thinking_mask` has bit 7), until E4-A models the B = 4 partner reads. Unused labels
`TagStrategy_Unused_1` / `_2` (`s:7116-7125`) have no jump to them: not transcribed. `TagStrategy_Partner`
(`s:7339`) is reached only through `IfTargetIsPartner` (`ai:2630-2639`), never at B = 2.

## Partner-battler constants at B = 2

| item | value | cite |
|---|---|---|
| attacker / defender | battler 1 / 0 | `ai:264-265` |
| `ATTACKER_PARTNER` / `DEFENDER_PARTNER` | battler 3 / 2 (`arg ^ 2`) | `ai:2765-2782`, `trainer_ai.h:28-31` |
| `battleMons[2]`, `[3]` | all zero (context cleared, mons built for `i < maxBattlers`) | `battle_controller_player.c:180-182`, `:223-227` |
| hp% of a partner arg | 0 (`0 / 0` returns the dividend) | `ai:596`, `:611`, `:626`; ruling G-53 |
| `CheckBattlerAbility` partner arg | `UNKNOWN` (2): raw ability 0 / species 0 abilities 0, 0 | `ai:1248-1260` |
| `FlagBattlerIsType` partner arg | FALSE (types 0 = Normal; no block tests Normal) | `ai:999-1003` |
| `IfMoveEffect` partner arg | never jumps (mask 0) | |
| `IfMoveKnown ATTACKER_PARTNER` | never jumps (`curHP == 0`) | `ai:1650-1653` |
| `GetPartner` (`LoadTypeFrom ATTACKER_PARTNER_TYPE_1/_2`) | the attacker itself | `sys:400-402`, `ai:965-978` |
| `IfTargetIsPartner` | never | `ai:2630-2639` |
| `IfBattlerFainted` partner arg | jumps | |

## B = 2 effect per block

| block (`s` lines) | fires at B = 2 | score law |
|---|---|---|
| Main (6637-6656) | `CalcAllDamage` draws (Psywave / Magnitude ids), then HALF / QUARTER effectiveness to TryScoreMinus | no change by itself |
| TryScoreMinus1 / 2 (6658-6680) | step 2 `IfHPPercentEqualTo DEFENDER_PARTNER 0` always jumps | no -1 / -2; R@6666, R@6678 never drawn |
| ScoreMove (6682-6704) | `CheckIfHighestDamageWithPartner` (the attacker's own moves, 1 or 2 sweeps); HIGHEST: PRIORITY_1 R@6702(50) then R@6694(128) | +1 (P 206/256 under PRIORITY_1, else 128/256); a jump runs CheckBeforeScoring |
| CheckBeforeScoring (6706-6730) | not-HIGHEST or a jumped roll; flat-damage effects skip | DOUBLE R@6722(100): +1 (156/256); QUADRUPLE R@6728(64): +1 (192/256) |
| CheckSpecialScoring (6732-6752) | dispatch by move id, then table type (Electric, Fire, Water) | Helping Hand test never jumps |
| RainDance (6754-6786) | own Hydration (no status) / Dry Skin | +2; partner UNKNOWN: nothing |
| SunnyDay (6788-6855) | own Leaf Guard (no status, hp% >= 30), Flower Gift, Dry Skin, Solar Power | +2 / +2 / -2 / Solar Power: hp% >= 50 +1 then R@6820(128) -2 (1/2), hp% < 50 R@6820(128) -2 (1/2) |
| Hail (6857-6884) | own Ice Body, Snow Cloak, knows Blizzard | +2 |
| Sandstorm (6886-6916) | own Sand Veil; own Rock types, read again as the "partner" | +2 per hit: Rock +4, Sand Veil +2, both +4 |
| Gravity (6918-6977) | Gravity already on: -30; own Levitate / Flying / Magnet Rise; defender Levitate (known) / Flying / Magnet Rise | -30 under Gravity; own -5; defender R@6959(64): +3 (192/256) |
| TrickRoom (6979-7030) | `IfHPPercentEqualTo ATTACKER_PARTNER 0` | -30 always, no sort, no roll |
| FollowMe (7032-7100) | a = own hp%, q = 0 | a > 90 +3 (R@7095); 30 < a <= 90 +2 (R@7090); a <= 30 -5 (R@7053, 192/256); each roll's change 192/256 |
| Earthquake / Magnitude id (7127-7148) | ATTACKER_PARTNER tests all fail | -3 always |
| PartnerKnowsHelpingHand (7102-7114) | never | +1 not reached |
| FutureSight / Doom Desire (7150-7193) | `IfHPPercentEqualTo ATTACKER_PARTNER 0` ends | 0 |
| SkillSwap (7195-7213) | as at B = 4 | +5 own Truant / Slow Start / Stall / Klutz; +2 defender Shadow Tag, Pure Power, Huge Power, Mold Breaker, Solid Rock, Filter, Flower Gift (ability coin `ai:1195` for a two-ability unknown defender) |
| Electric (7215-7263) | Discharge only | Discharge -3 (SpreadElectric reads battler 3: Normal, no -10 / +3 case), other Electric 0 |
| Water (7265-7302) | `CheckBattlerAbility DEFENDER_PARTNER STORM_DRAIN` is UNKNOWN: -1 | every non-Surf Water move -1; Surf -3 |
| Fire (7304-7337) | `IfActivatedFlashFire ATTACKER` | +1 with Flash Fire on; Lava Plume -3 more (`s:7308`'s comment says +3, the script scores -3) |

Never at B = 2: Helping Hand +1, the defender-partner and attacker-partner score changes, both speed-rank
blocks (`LoadBattlerSpeedRank` in TrickRoom and FutureSight), `IfBattlerFainted` and `IfMoveKnown
ATTACKER_PARTNER` jumps. Each add clamps at 0 (`ai:583-585`).

## The 21 roll sites (`outcome.SITE_N`)

| line | N | label | B = 2 |
|---|---|---|---|
| 6666 | 64 | TryScoreMinus1 (-1 when it does not jump) | never reached |
| 6678 | 64 | TryScoreMinus2 (-2) | never reached |
| 6694 | 128 | ScoreMove (+1) | yes |
| 6702 | 50 | TryScorePlus1, PRIORITY_1 (+1) | yes |
| 6722 | 100 | TryPrioritizingDoubleEffective (+1) | yes |
| 6728 | 64 | TryPrioritizingQuadEffective (+1) | yes |
| 6820 | 128 | SunnyDay_SelfTryScoreMinus2 (-2) | yes |
| 6851 | 128 | SunnyDay_PartnerTryScoreMinus2 | never reached |
| 6959 | 64 | Gravity_TargetTryScorePlus3 (+3) | yes |
| 6972 | 64 | Gravity_TargetPartnerTryScorePlus3 | never reached |
| 7012 | 64 | TrickRoom_SelfMovesThird | never reached |
| 7022 | 64 | TrickRoom_SelfMovesLast | never reached |
| 7053 | 64 | FollowMe, own hp% <= 30 (-5) | yes |
| 7075 | 64 | FollowMe_TryScoreMinus1 | partner hp% > 90 only: never |
| 7080 | 64 | FollowMe_TryScoreMinus2 | never |
| 7085 | 64 | FollowMe_TryScorePlus1 | never |
| 7090 | 64 | FollowMe_TryScorePlus2 | yes |
| 7095 | 64 | FollowMe_TryScorePlus3 | yes |
| 7173 | 128 | FutureSight_SelfMovesThird | never reached |
| 7181 | 128 | FutureSight_SelfMovesSecond | never reached |
| 7187 | 128 | FutureSight_SelfMovesFirst | never reached |

The speed-rank tie draws (`s:6986-7018`, `:7162-7188`) are no `IfRandomLessThan` lines (A-10 section 2).
