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
  (`script.s:76`) loaded. That is `AICmd_LoadBattlerAbility` (`trainer_ai.c:1170-1198`): the
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
- Until Task 3 keys the script-roll sites by effect, the site table is still keyed by move
  id (`outcome._SITES`). So Kaizo also refuses a move that table does not list. Example:
  Accelerock (205, effect HIT) stays refused until then.
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
