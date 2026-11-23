# Tag Strategy (AI flag bit 7, `TagStrategy_Main`) at B = 2

Source: `src/battle/trainer_ai/script.s` (`s`), `src/battle/trainer_ai/trainer_ai.c` (`ai`), `battle_system.c`
(`sys`), relative to `external/pokeplatinum`. Code: `policy._tag_strategy` (`script.s:6637-7338`).
Rules sheet: `_scratch/e4/rules/A-8.md` (sections 1-4, 7). Commands: `docs/ai/script_commands.md`.
`R@L(N)`: `IfRandomLessThan N` at line L, jump w.p. N/256; the change below is made when it does NOT jump.

Scope. Exact at B = 2 (singles): 16 singles split rows carry bit 7 (`ai:297-309` runs it after bit 6, before
bit 8). Six are opponents (49, 50, 187, 646, 725, 898); the other ten (607, 608, 609, 610, 612, 619, 620, 861,
863, 865) are tag partners (each is a `partner` of multi rows in `data/intermediate_kaizo/trainers.json`, e.g.
608 Cheryl, `res/field/scripts/scripts_eterna_forest.s:57`), so they run Tag Strategy only at B = 4 beside the
player, still refused. A doubles battle adds bit 7 to every thinking mask (`ai:252-255`); B = 4 is refused: the format gate
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
| Fire (7304-7337) | `IfActivatedFlashFire ATTACKER` | +1 with Flash Fire on; Lava Plume -3 more (`s:7334`: the partner tests all fail). A Dry Skin partner scores -3 (`s:7322-7323`, `ScoreMinus3`) where `s:7308`'s comment says +3; unreachable at B = 2 (the partner's ability is UNKNOWN) |

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

# Partner reads at B = 4 (E4-A W2: 2.2, 2.5.5, 2.6-2.14, 2.16, 2.18-2.20, section 5 of A-8)

Rules sheet `_scratch/e4/rules/A-8.md` sections 1, 2, 5. One scoring is one attacker `a` (static) against one
enemy target `d`; the four script battlers are `a`, `d`, `a ^ 2`, `d ^ 2` (`ai:2765-2782`, `commands._battler`).
No block here changes with the code at B = 2: the blocks were written over the partner arguments, which at B = 2
name the zeroed battlers 2 and 3 (table above) and at B = 4 name the real ones. Tests: `tests/ai/test_tag_doubles.py`
(the scoring is run directly on test-built B = 4 contexts; `CheckIfHighestDamageWithPartner` runs with its B = 4
sweep, A-7's, and the attacker's comparator move keeps slot 0 NOT_HIGHEST, the module docstring).

## What each partner argument reads at B = 4

| command, argument | read | cite |
|---|---|---|
| `IfHPPercentEqualTo/GreaterThan/LessThan X_PARTNER` | `curHP * 100 / maxHP` of the real battler: 0 when fainted, and 0 when `curHP * 100 < maxHP` | `ai:596`, `:611`, `:626` |
| `CheckBattlerAbility ATTACKER_PARTNER` | raw `battleMons[a ^ 2].ability`, no AI knowledge (suppressed: NONE, so UNKNOWN) | `ai:1221-1222`, `:1255-1256` |
| `CheckBattlerAbility DEFENDER_PARTNER` | the AI's record of `d ^ 2`, else a trapping ability, else the species' two abilities (neither is the tested one: ability 1; one is: UNKNOWN; one ability: it) | `ai:1223-1253` |
| `FlagBattlerIsType X_PARTNER` | the real battler's current types, no HP test | `ai:999-1003` |
| `IfMoveEffect X_PARTNER MAGNET_RISE` | that battler's `moveEffectsMask` bit | `ai:704-714` |
| `IfMoveKnown ATTACKER_PARTNER` | its true moves, no jump while its `curHP` is 0 | `ai:1650-1664` |
| `IfNotStatus / IfStatus ATTACKER_PARTNER` | its `status` | `ai:648-674` |
| `LoadTypeFrom ATTACKER_PARTNER_TYPE_1/_2` | `BattleSystem_GetPartner(a)` = `a ^ 2` | `sys:394-411`, `ai:965-978` |
| `LoadBattlerSpeedRank` | one read per call, its own sort (A-10 section 2) | `ai:2056-2094` |
| `IfTargetIsPartner` | `(a & 1) == (d & 1)`: the ally part (A-9), not these blocks | `ai:2630-2639` |

MULTI and TWO_TRAINERS read the same way: the partner of battler 2 is battler 0 (the player's Pokemon), of
battler 1 and of battler 3 the other trainer's or the same trainer's Pokemon; a partner on the attacker's side is
read as true data, never through the AI's record (`ai:1255-1256`, `:1650-1664`, `:1925-1926`).

## B = 4 score laws

`P` = the change is made when the roll does NOT jump (`R@L(N)` jumps w.p. N/256), so P = (256 - N) / 256.

| block (`s`) | partner reads | law |
|---|---|---|
| TryScoreMinus1 / 2 (6658-6680) | `IfHPPercentEqualTo DEFENDER_PARTNER 0` | partner hp% != 0 and the move does not kill: -1 / -2 with P 192/256 (R@6666, R@6678); hp% 0: no change, no roll |
| RainDance (6754-6786) | partner Hydration (statused) or Dry Skin | +2 each of self and partner; no roll |
| SunnyDay (6788-6855) | partner Leaf Guard (no status, hp% >= 30) / Flower Gift +2; Dry Skin -2; Solar Power hp% >= 50 +1 then R@6851(128) -2 (1/2), hp% < 50 R@6851(128) -2 (1/2) | first HAVE of the four |
| Hail (6857-6884) | partner Ice Body, Snow Cloak, knows Blizzard (alive) | +2 |
| Sandstorm (6886-6916) | partner Sand Veil, type 1 or 2 Rock (its own types, not the attacker's) | +2 once |
| Gravity (6918-6977) | partner and defender partner: Levitate (HAVE), Flying, Magnet Rise | partner -5; defender partner R@6972(64) +3 (192/256), defender R@6959(64) +3 |
| TrickRoom (6979-7030) | hp% 0 of partner, defender partner, defender: -30; ranks r = rank(a), q = rank(a ^ 2), one sort per call | r0: q 1 or 0 -30 else -5; r1: q 0 -30 else -5; r2: q != 3 -5, else R@7012(64) -5 / +5 (192/256); r3: q != 2 -5, else R@7022(64) |
| FollowMe (7032-7100) | partner hp% bands (strict `>` 90, 50, 30) | as the table in A-8 2.12, each change P 192/256 (R@7075-7095) |
| Earthquake, id 222 (7127-7148) | partner Magnet Rise, Levitate, Flying +2; Fire, Electric, Poison, Rock -10; else -3 | first match, no HP test, no roll |
| PartnerKnowsHelpingHand (7102-7114) | `IfMoveKnown ATTACKER_PARTNER HELPING_HAND` (alive partner) | damaging non-flat move not matched above: +1 |
| FutureSight (7150-7193) | partner alive and knows Future Sight or Doom Desire; ranks r, q, q' (a new sort per call) | r3 -3; r2: q 0 or 1 -3, R@7173(128), q' == 2 -3; r1: q 0 -3, R@7181(128), q' == 1 -3; r0: R@7187(128), q' == 0 -3 |
| Electric (7215-7263) | defender partner Lightning Rod (HAVE only) -1, and -8 more unless it is not Ground; partner Lightning Rod -10; Discharge: SpreadElectric | Motor Drive / Volt Absorb +3, Water / Flying -10, Ground +3, else -3 |
| Water (7265-7302) | defender partner Storm Drain NOT_HAVE: 0, else (HAVE or UNKNOWN) -1; partner Storm Drain -10; Surf: SpreadWater | Dry Skin / Water Absorb +3, Ground / Fire -10, else -3 |
| Fire (7304-7337) | Lava Plume: SpreadFire | Dry Skin -3 (`s:7322-7323`), Flash Fire +3, Grass / Steel / Ice / Bug -10, else -3 |

Each add clamps at 0 (`ai:583-585`).

## A fainted partner at B = 4 (A-8 section 5)

`BattleSystem_CleanupFaintedMon` (`lib:2156-2226`) leaves species, types, ability, moves, item and `status` in
`battleMons`, clears stat stages, `moveEffectsMask`, `moveEffectsData` (Flash Fire, Magnet Rise) and the AI record
of that battler (`lib:2223-2225`; engine: `state.ai_record_clear`). So: hp% reads 0 (Trick Room -30, Future Sight
End, Follow Me with q = 0, the TryScoreMinus skip); ability and type blocks still score it (Rain, Sun, Hail,
Sandstorm, Gravity Levitate / Flying, Earthquake, Discharge, Surf, Lava Plume, Lightning Rod, Storm Drain of the
attacker's partner); its Magnet Rise reads false; `IfMoveKnown ATTACKER_PARTNER` does not jump; a fainted defender
partner's `CheckBattlerAbility` takes the guess path (its record is gone, so a recorded Lightning Rod or Storm Drain
no longer reads HAVE).

## Roll sites reached at B = 4

6666, 6678, 6851, 6972, 7012, 7022, 7075, 7080, 7085, 7173, 7181, 7187 join the B = 2 sites (the table above:
"never reached" at B = 2). Reads per call of `LoadBattlerSpeedRank`: at most 2 (A-10 section 2, decision 6).

# The ally part at B = 4 (`TagStrategy_Partner`, script.s:7339-7694; `policy._tag_strategy_partner`)

Entered by `IfTargetIsPartner` (s:6638 from bit 7, s:7697 from bit 8, so a CHECK_HP trainer runs it twice, both
score changes and both draw sets). Never entered at B = 2 (the target is `attacker ^ 1`). The ally is
`ATTACKER_PARTNER` and `DEFENDER`. Scores from 100; "P" is the probability of the change; every path ends.

| Move at the ally | Law (decomp lines) |
|---|---|
| ally down (`IfBattlerFainted`), or a damaging move of no listed type, or any status move not listed, Fling | -30 (:7339-7350, :7455-7456) |
| Fire move, Flash Fire ally | +3; -30 once Flash Fire is activated; -30 without the ability (:7352-7362) |
| Electric move, Motor Drive ally | R@7383(160) jump: nothing; else -30 at Speed +6, else +3 (:7380-7386) |
| Electric move, Volt Absorb ally / Water move, Water Absorb or Dry Skin ally | hp% 100: -10; > 90: nothing; > 75 / > 50 / else R@7395 / 7399 / 7403 (Water 7430 / 7434 / 7438) with N = 64 / 128 / 192, **the jump is +3**; any other ability -30 (:7388-7442) |
| Skill Swap | ally ability (knowledge branch, one coin per guessing read, 7467 / 7474): Truant, Slow Start +10; attacker Levitate: ally Levitate -30, type 1 not Electric to the accuracy block (so -30), type 1 Electric +1 then (type 2 not Electric) the accuracy block (net -29), both Electric +2; accuracy block: attacker Compound Eyes / No Guard and the ally knows an inaccurate move (14 ids) +3, else -30 (:7458-7511) |
| Will-O-Wisp | Flash Fire ally: the Fire ladder; else +5 for Guts, no status, no Fire type, no Flame / Toxic Orb, hp% >= 81; else -30 (:7513-7542) |
| Thunder Wave | Ground ally -30; Motor Drive / Volt Absorb: the Electric ladders; else -30 (:7544-7559) |
| Toxic / Poison moves | +5 for Poison Heal, ally unstatused, no Toxic Orb, hp% <= 91 (the comment says 81); else -30 (:7561-7580) |
| Helping Hand | hp% 0 -30; hp% > 50 or the ally's speed rank 0 (one call): R@7596(64) jump -1, else +2; else nothing (:7582-7600) |
| Swagger | Persim / Lum Berry holder: +3 below Attack +2 (decomp stage <= 7), else nothing; no berry -30 (:7602-7619) |
| Trick / Switcheroo | nothing (:7621-7622) |
| Gastro Acid | suppressed -30; Truant / Slow Start +5; else nothing (:7624-7644) |
| Acupressure | not Simple: a stat at +6 -30; Simple: a stat at +3 or more -10; then hp% < 51: -1; > 90: +2 with 176/256 (R@7682(80)); else R@7679(128) then R@7682: +2 with 88/256 (:7646-7690) |

New roll sites: 7383, 7395, 7399, 7403, 7430, 7434, 7438, 7596, 7679, 7682 (`outcome.SITE_N`, appended; B = 2 outputs
and the generated tables are unchanged: the flat layout reads them through an unused column). Id rows:
`MOVE_IDS` (Will-O-Wisp, Swagger, Gastro Acid, Acupressure, Fling and the 14 inaccurate moves), `MOVE_TRICK`,
`MOVE_SWITCHEROO` (Kaizo names ids 271 / 415 Hidden Power Psychic / Flying), `ABILITY_IDS["Compoundeyes"]`, four item ids.
