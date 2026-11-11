# Kaizo Roark's trainer AI, transcribed from the decomp

This is the specification for the Kaizo policy of the trainer AI in `src/nuzline/ai`
(milestone K3c). It is built the way `docs/ai/roark.md` is, and it states only what
differs from that vanilla transcription; every rule not restated here is roark.md's.
Paths are relative to `external/pokeplatinum`. `script.s` means
`src/battle/trainer_ai/script.s` and `trainer_ai.c` means
`src/battle/trainer_ai/trainer_ai.c`. Kaizo data comes from the Kaizo tables
(`data/intermediate_kaizo/*.json`, built into `data/gen4_platinum_kaizo`).

K3c is written task by task: sections 1-2 come from Task 2, the later sections from the
tasks that own them (draws and the outcome layout: Task 3; commands: Task 4; per-flag
paths: Tasks 5-6; switching: Task 7; the rest: Task 8).

## 1. Flags

- `data/intermediate_kaizo/trainers.json`, trainer 246 (Leader Roark): `ai_flags = 23`,
  `0b10111`.
- Line n+2 of `generated/ai_flags.txt` is bit n (roark.md 1), so 23 = BASIC (bit 0) +
  EVAL_ATTACK (bit 1) + EXPERT (bit 2) + RISKY (bit 4, `AI_FLAG_RISKY`, line 6).
- `FlagTable` (`script.s:18-22`) maps bit 4 to `Risky_Main` (`script.s:6521`).
- `TrainerAI_MainSingles` runs the flags in bit order (`trainer_ai.c:297-310`): BASIC,
  EVAL_ATTACK, EXPERT, then RISKY.
- The Kaizo policy runs exactly these four scripts (plan K3c Ruling 10). Until K3c Task 6
  registers RISKY, coverage refuses the flag (`engine/coverage.py`
  `IMPLEMENTED_AI_FLAGS`), so no Kaizo Roark fight is graded yet.

## 2. Differences from vanilla

**Where they live.** The game's `Tables` carry `ai_policy` (0 vanilla, 1 kaizo; built
from `game_constants.POLICY`, whose Kaizo entry is `"kaizo"`). `nuzline.ai.layout.layout_of`
reads it on concrete tables and gives the game's `AiLayout`. The AI's traced functions
take the layout as a static argument. The public functions of `ai.policy` resolve it
themselves. Under a trace without a layout they take `VANILLA` and refuse any other
policy's tables (`_Refused.policy`).

**The AI is the vanilla script plus one word.** overlay14, which carries `gTrainerAITable`,
is the same size in both ROMs. Its one changed script range is the word at 0x022249CC.
Every other change in overlay14 is a relocated call into overlay16 (K3a, overlay14
summary sections 1 and 3; `docs/kaizo/rom_diff.md`). So no script command, damage
estimate, switch rule or item rule of the AI's C code differs. Kaizo's different
behaviour comes from that word and from Kaizo's *data*: its moves carry their own power,
type and battle effect, read through the same code.

**2.1 The Dry Skin word.**
- Vanilla: `Basic_CheckForImmunity`'s last compare (`script.s:78`) tests
  `ABILITY_LEVITATE` (26) a second time, after `:77` already took Levitate to
  `Basic_CheckGroundAbsorption`. It never matches (the decomp marks it as a bug).
- Kaizo tests `ABILITY_DRY_SKIN` (87) there (`data/kaizo/rom_diff.toml`,
  `overlay14 0x022249CC-0x022249CD`, now `modelled`).
- On a match `Basic_CheckWaterAbsorption2` (`script.s:106-109`) runs. If
  `LoadTypeFrom LOAD_MOVE_TYPE` is Water, the move scores -12 (`ScoreMinus12`).
- The ability compared is the one `LoadBattlerAbility AI_BATTLER_DEFENDER`
  (`script.s:71`) loaded. That is `AICmd_LoadBattlerAbility` (`trainer_ai.c:1170-1198`): the
  known ability, else a coin-flip guess between the species' two abilities (roark.md 9.2).
  It is not `AICmd_LoadAbility` (`trainer_ai.c:2654-2661`, the true ability), as the K3a
  overlay14 notes and the old `rom_diff.toml` note said.
- Code: `AiLayout.water2_ability` (26 vanilla, 87 Kaizo), read by `policy._basic`. Test:
  `tests/ai/test_layout.py::test_dry_skin_compare_per_policy`.

**2.2 Transcribed moves by battle effect.**
- Vanilla keeps its refusal set by move id (`commands.MOVE_EFFECTS`).
- The Kaizo policy reads a move's script paths by its battle effect,
  `MOVE_DATA(move).effect` (`tables.move_ai_effect`; `IfCurrentMoveEffectEqualTo`,
  `trainer_ai.c:1441`). `commands.move_effect(tables, move)` reads that array in both
  policies. Kaizo refuses an opponent move whose effect is not in `KAIZO.transcribed`.
- That set starts as the effects of the vanilla-transcribed moves: HIT (0), DEF_DOWN (19),
  FLINCH_HIT (31), DEF_DOWN_2 (59), HIT_BEFORE_SWITCH (128) and STEALTH_ROCK (266).
  Tasks 5 and 6 add Roark's.
- The script-roll sites are keyed by battle effect too (section 8, K3c Task 3), so a move is
  transcribed by its effect alone. Example: Accelerock (205, effect HIT) takes Rock Throw's
  paths and its EvalAttack quad roll (`script.s:6383`).
- RISKY (`script.s:6521-6531`) scores only the effects of `Risky_RiskyEffects`
  (`script.s:6536-6561`). None of the six effects above is listed there, so the Kaizo
  policy's scores equal the three-script vanilla transcription's until Task 6 adds
  Selfdestruct (HALVE_DEFENSE, 7) and AncientPower (RAISE_ALL_STATS_HIT, 140).

**2.3 No damage estimate for the no-calc effects.** These hold in both policies; only
Kaizo data reaches them on a transcribed path.
- `TrainerAI_CalcAllDamage` (`trainer_ai.c:2827-2828`), `FlagMoveDamageScore`
  (`:1035-1036`) and `IfCurrentMoveKills` (`:1553-1554`) estimate damage only for a move
  that meets two tests: it has power greater than 1, and its effect is not in
  `sNoDamageCalcMoveEffects` (`trainer_ai.c:31-46`).
- A move that fails them estimates 0. It gets `AI_NO_COMPARISON_MADE` and never counts as
  the highest damage, and it never "kills".
- Kaizo Selfdestruct (120, HALVE_DEFENSE, power 200) and Head Smash (457, RECOIL_HALF,
  power 150) are in that table. So Kaizo Cranidos's highest-damage move is Crunch or Zen
  Headbutt, not Head Smash (plan K3c Review Focus 1).
- Code: `commands.is_damaging` and `commands.NO_DAMAGE_CALC_EFFECTS`. Test:
  `test_kaizo_highest_damage_ignores_head_smash`.
- No vanilla-transcribed move has such an effect, so vanilla is unchanged
  (`tests/ai/test_policy.py::test_vanilla_distribution_unchanged`).

**2.4 Refusals that only Kaizo data reaches.** Both policies refuse an opponent move that
the AI's damage estimate or move type does not take from the move table:
- an effect in `sAltPowerMoveEffects` (`trainer_ai.c:48-61`; `commands.ALT_POWER_EFFECTS`).
  Such a move is damage-estimated whatever its power (`:1035`, `:1553`, `:2827`). Kaizo
  Triple Axel (149) and Fury Cutter (210) carry RANDOM_DAMAGE_1_TO_150_LEVEL (88);
- a move id that `TrainerAI_CalcDamage` takes by case (`trainer_ai.c:2885-3056`;
  `commands.CALC_DAMAGE_MOVES`). The case applies whatever the move's effect. Kaizo
  Triple Axel sits on MOVE_PSYWAVE's id (`:3014`, which also draws `RandNext`). Kaizo
  Return (216) and Frustration (218) have effect 0 but still take their friendship
  cases (`:3018-3026`);
- a move id that `TrainerAI_MoveType` takes by case (`trainer_ai.c:3132-3222`;
  `commands.MOVE_TYPE_MOVES`): Natural Gift, Judgment, Hidden Power, Weather Ball.

In vanilla every such move is already refused as untranscribed, so the vanilla refused set
is unchanged.

**2.5 Ids.** The script's and `trainer_ai.c`'s ids are the same numbers in both ROMs (one
script word changed; the C code only relocated).
- So the AI's id lists keep their decomp ids in Kaizo. `game_constants` checks each one
  against the Kaizo names (scope `"kaizo_policy"`):
  - Fissure 90, Horn Drill 32, U-turn 369, Roar 46, Whirlwind 18;
  - the 11 Soundproof ids (`script.s:121-131`);
  - Gyro Ball 360 and Trick Room 433 (`script.s:3472-3473`);
  - the switching move ids;
  - the site table's ids;
  - Dry Skin 87.
- A Kaizo-renamed slot on one of these lists fails `load_game`.
- The id cases of 2.4 are compared by id whatever a game names the slot. They are checked
  against vanilla names only (Kaizo's 149 is Triple Axel, and it still takes Psywave's
  case).

## 8. Random draws: the per-slot layout

The draws themselves are roark.md 8's: `BattleSystem_RandNext` is a uniform `u16`, every
`IfRandomLessThan N` jumps with probability N/256 (`trainer_ai.c:528-538`), every
`LoadBattlerAbility DEFENDER` guess is a coin flip (`trainer_ai.c:1191-1198`), and the
tie-break is `RandNext % n` (`trainer_ai.c:338`). What differs is how `AiOutcome` lays them
out (`nuzline.ai.outcome`; `AiLayout.per_slot_sites`).

**8.1 Fields.**
- `sim_roll`: fixed at 100, as roark.md 8.1. Every damage read on the transcribed paths
  passes `USE_MAX_DAMAGE`.
- `ability_guess`: one scalar for the whole decision (8.4).
- `script_rolls: int8[MOVES, KAIZO.slot_sites]`. Column k of slot m is the k-th
  `IfRandomLessThan` site of the slot's move, in script order (flag order BASIC,
  EVAL_ATTACK, EXPERT, then RISKY, `trainer_ai.c:297-310`; then line order). 1 means the
  jump is taken.
- `tiebreak`: `RandNext % 12`, as roark.md 8.4.
- `switch_stop`: as roark.md 8.5, over `KAIZO.switch_sites` (K3c Task 7 sets them for the
  six-Pokemon party).

**8.2 The site table, keyed by battle effect.** A move reaches a site through
`IfCurrentMoveEffectEqualTo` (`trainer_ai.c:1441`), so the table is keyed by the move's
effect (`outcome.EFFECT_SITE_LINES`, carried as `KAIZO.effect_sites`), not its id. Each
site is named by its `script.s` line, and `outcome.SITE_N` gives its N:

| Effect | Sites (line: N) |
|---|---|
| HIT (0) | `:6383`: 80 |
| DEF_DOWN (19) | `:6383`: 80, `:2328`: 50 |
| FLINCH_HIT (31) | `:6383`: 80 |
| DEF_DOWN_2 (59) | `:6383`: 80, `:2328`: 50 |
| HIT_BEFORE_SWITCH (128) | `:6383`: 80, `:3726`: 128, `:3731`: 128 |
| STEALTH_ROCK (266) | `:6383`: 80, `:6274`: 128 |

- `:6383` is `EvalAttack_TryScorePlus2`. It is reached by any move whose
  `IfMoveEffectivenessEquals` is QUADRUPLE (`script.s:6378`), whatever its effect, so every
  effect has it. The vanilla table left it out of the Normal-type moves, but a table keyed
  by effect cannot know the type. An extra site that is never reached only costs width.
- The Expert sites follow `Expert_Main`'s dispatch (`script.s:1623-1811`): DEF_DOWN `:1642`
  and DEF_DOWN_2 `:1671` go to `Expert_StatusDefenseDown` (`:2318`, roll `:2328`);
  HIT_BEFORE_SWITCH `:1706` to `Expert_Pursuit` (`:3706`, rolls `:3726`, `:3731`);
  STEALTH_ROCK `:1805` to `Expert_StealthRock` (`:6268`, roll `:6274`). HIT and FLINCH_HIT
  have no entry and end at `:1811`.
- `KAIZO.transcribed` is exactly the table's effects, so the effect check that refuses an
  untranscribed move also guarantees the move has its row.
- `KAIZO.slot_sites` is 8, the most sites one Kaizo Roark move reaches: Selfdestruct
  (HALVE_DEFENSE), with EvalAttack `:6374` and `:6383`, Expert_Explosion `:1859`, `:1865`,
  `:1870`, `:1875` and `:1880`, and Risky `:6530`. Tasks 5 and 6 add Roark's effects to the
  table.

**8.3 The exact distribution, factored per slot** (`policy._slot_move_distribution`).
- `TrainerAI_EvalMoves` scores the move slots one by one (`trainer_ai.c:483-525`).
  Slot m's score reads only its own script rolls and the guess. `FlagMoveDamageScore`
  compares the damage estimates of all four moves, but those are deterministic
  (`trainer_ai.c:1012-1062`), not draws.
- So, for each guess: `_scores` runs once per local bit pattern p (2^8 = 256 patterns,
  the same p in every slot's columns), and slot m's score with its bits p is read off.
  Each slot's histogram over score values 0-255 weights p by its probability in that slot,
  `prod_k` of N_k/256 where bit k is 1, else 1 - N_k/256.
- The pick (`trainer_ai.c:315-338`): slot m with value v is picked when every other
  candidate ties at v or is below it. For each subset T of the other candidates that ties,
  m's share is `P(RandNext % (|T| + 1) == m's place in the tied list)`.
- The two guesses are averaged. The switch comes first and the item next, as
  `_distribution` does (roark.md 2). Struggle and the locked slot are unchanged.

**8.4 The one-scalar guess (plan K3c Ruling 6).** The decomp draws a fresh coin at every
`LoadBattlerAbility DEFENDER` (`trainer_ai.c:1191-1198`). One scalar is exact when at most
one such read in a decision can change a score. On the transcribed paths the reads are
`script.s:71` (`Basic_CheckForImmunity`), `:117` (`Basic_CheckSoundproof`) and `:470`
(`Basic_CheckClearBodyEffect`). `tests/ai/test_kaizo_outcome.py
test_one_scalar_guess_is_exact_for_roark` checks this as data: for every species with two
abilities, against each of Roark's Pokemon (Kaizo's six, vanilla's three), the loose
oracle (8.6) finds at most one read whose coin changes a score. It re-runs with every
read a later task adds.
- This holds for Roark's movesets, not for every moveset. A Pokemon with both Leer and
  Screech against Tentacool (Kaizo abilities 44 / Clear Body) has two `:470` reads that
  each change a score; the scalar moves them together and the decomp does not. Such a
  moveset is not on Roark's team. A later trainer's must pass the same data test.

**8.5 Width** (plan K3c Ruling 12). `policy.ai_width` counts the combinations the factored
distribution merges: the switch stops plus, per guess, `sum_slots 2^(sites of the slot)`.
For Kaizo Roark's six it is at most 10 + 2 x 4 x 2^8 = 2058 for any state (checked at
import, `policy._check_width`), against `WIDTH_CAP` 131072. `policy.ai_flat_width` reports
the flat product, `2 x 2^(sum of sites) x 12 x 10`, which is not used. Section 14 has the
measured numbers.

**8.6 The loose oracle** (tests only; `policy.ai_action_distribution_loose`). Every
`RandNext` read of the scoring is its own independent draw, as in the decomp: each
`IfRandomLessThan` with its own N (the literal at its call), each `LoadBattlerAbility
DEFENDER` its own coin, the tie-break `RandNext % n` over a `u16`. It reads neither the site
table nor the scalar guess. It finds every read on every path by enumerating the reads
found so far until no new one appears, then enumerates the slots jointly when they have at
most 18 reads in all, and per slot otherwise (exact in loose, where no two slots share a
draw). The gates of the exact layout are each checked against it, with a test that fails
when the gate is removed (`tests/ai/test_kaizo_outcome.py` docstring):
- per-slot factoring and the tie subsets, against a joint enumeration of every slot's bits
  and the 12 tiebreaks;
- the path-only sites (a move has only its effect's sites);
- the one-scalar guess (8.4).

The fourth gate, the dropped speed-tie draw (plan K3c Ruling 3: at equal speed
`BattleSystem_CompareBattlerSpeed` returns FASTER or TIE, never SLOWER,
`battle_lib.c:1401-1448`, while every compare on Roark's paths tests SLOWER), arrives with
`IfSpeedCompareEqualTo` in Task 4, which gives its tie its own loose bit.

**8.7 RNG read order.** As roark.md 8.7, with RISKY after EXPERT (slots 0-3: its roll as
reached), and the speed compare's tie draw where an `IfSpeedCompareEqualTo` meets equal
speeds (no `AiOutcome` field, Ruling 3; K3d's trace replay needs the count).

## 14. Width and cost, measured (K3c Task 3)

Measured 2026-11-11 on one CPU device (20 logical CPUs), warm, with Kaizo Roark's AI still
refused (Tasks 5-7): a refused row costs what an accepted one does, since the program
computes everything and then masks it. Scripts: the Task 3 report.

**Width.**
- `ai_width` (the combinations merged, 8.5) for Kaizo Roark's six against the
  `kaizo_save_pre_roark` pool: 18 to 26. Its bound for any state is 2058; `WIDTH_CAP` is 131072.
- `ai_flat_width` (reported, not used): 240 to 1920.
- Vanilla: `ai_width` 42, flat 3840 (`ENUMERATION_WIDTH`).

**Cost per state.** `policy._distribution`, vmapped over 1000 states, no padding:
- vanilla: 10.2 µs;
- Kaizo: 162 µs. The Kaizo layout runs `_scores` 2 x 2^8 = 512 times per state, vanilla 32.

**`ai_rows` buckets.** `expand.ai_rows` pads each call to a power-of-two bucket of its rows
(256 minimum, `CHUNK` 16384 maximum; `expand.bucket_rows`). Before that every call paid for
16384 rows: 2.39 s Kaizo and 0.24 s vanilla, for one state as for a thousand.

| bucket | vanilla warm | Kaizo warm | cold (compile + export) |
|---|---|---|---|
| 256 | 0.005 s | 0.039 s | not separated (compiled by an earlier step of the run) |
| 512 | 0.008 s | 0.077 s | 3.5 s each |
| 1024 | 0.013 s | 0.176 s | not separated |
| 16384 | 0.248 s | 2.34 s | 3.8 s vanilla, 5.1 s Kaizo |

The padding rows are copies of the call's row 0 and are dropped. A row's result does not
depend on the bucket: `tests/search/test_expand.py test_ai_rows_do_not_depend_on_the_bucket`
and `tests/ai/test_kaizo_outcome.py test_kaizo_ai_rows_do_not_depend_on_the_bucket` compare
the bytes.

**AI share of an expansion.** This is `ai_rows` over (`ai_rows` + `expand`) for one level.
The Kaizo AI is refused, so its legal move actions stand in for the actions with mass.

| | states | pairs | ai_rows | expand | AI share |
|---|---|---|---|---|---|
| Kaizo, depth 1 (one root) | 1 | 36 | 0.040 s | 0.218 s | 15.5% |
| Kaizo, depth 1, all 81 legal pairs | 1 | 81 | 0.039 s | 0.237 s | 14.0% |
| Kaizo, depth 2 (the root's decision children) | 399 | 14244 | 0.079 s | 13.6 s | 0.6% |
| vanilla, depth 1 | 1 | 18 | 0.005 s | 0.193 s | 2.5% |
| vanilla, depth 2 | 146 | 1867 | 0.005 s | 1.12 s | 0.5% |
| vanilla `nuzline solve`, depth 2, 2000 playouts (`stage_b.solve`) | 179 in 2 calls | | 0.012 s | 1.87 s | 0.6% (of a 58 s solve) |

**Review round (I2).** Kaizo `ai_rows` now maps its rows one at a time (`lax.map`,
`expand._ai`), so a row's float32 result no longer depends on the call size. Under `vmap`,
XLA fused the combine step per batch shape, and that moved some Kaizo rows by 1 ulp. Kaizo
re-measured warm after the change:
- bucket 256: 0.024 s;
- bucket 512: 0.046 s;
- bucket 1024: 0.095 s;
- bucket 16384: 1.49 s;
- cold: 2.7 s (512) and 4.2 s (16384);
- depth-1 AI share: 10%.

Vanilla keeps `vmap`, with the numbers above.

The spec 6 stop condition (an AI share above 50%) is not met. Before bucketing the Kaizo
depth-1 share was 91% and the vanilla one 56%, both from the 16384-row padding.
