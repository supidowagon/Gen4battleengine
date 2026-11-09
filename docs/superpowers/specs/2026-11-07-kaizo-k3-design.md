# K3: Kaizo Roark coverage, design

Status: draft for the user's review. Date: 2026-11-07.
Builds on: `docs/design.md`, `docs/superpowers/specs/2026-11-03-kaizo-data-design.md` (§10
step 1 and §11 define K3), `docs/kaizo/pipeline.md`, `docs/kaizo/rom_diff.md`,
`docs/tracing.md`, `docs/ai/roark.md`.

## 1. Intent

This command must run and give an answer the user can rely on:

```
uv run nuzline coverage --game kaizo --save <pre-Roark save> --trainer roark
```

Today it refuses the fight. The user plays a Kaizo Nuzlocke and ranks safety over speed: a
line nuzline calls safe has to be safe in the real game. So K3 removes every refusal by
modelling the real behaviour. It never removes a refusal by approximating or by quietly
leaving things out. Prep and solve on Kaizo Roark come next (K4, the exhaustive gate), and
they depend on everything here being right.

The save is `G:\Farry\Downloads\Platinum Kaizo Base.dsv`, configured through
`kaizo_save_pre_roark` in `data/local.toml` or `NUZLINE_KAIZO_SAVE_PRE_ROARK`. It is used
in place and is never copied or committed.

## 2. Goals, non-goals, anti-goals

**Goals**

1. `coverage --game kaizo --save <pre-Roark save> --trainer roark` prints `Support: EXACT`
   for the fight and for the whole pool, and exits 0.
2. Everything in the user's current pool is supported, not only Roark's side. The user
   said "do everything": the party, the box, every Move Relearner and bag-TM move, every
   held item already held, and every held item in the bag that prep could assign (§4.4).
3. Every one of the 103 unclassified Kaizo ROM changes is classified, with evidence (§5).
4. The Kaizo trainer AI for Roark is transcribed from the patched ROM: the Basic,
   EvalAttack, Expert and Risky flags, and script paths for every move either side can
   bring (§6).
5. A golden trace of Kaizo Roark, recorded by the user in DeSmuME from a copy of a save
   state, replays turn by turn through the engine with no mismatch (§7).

**Non-goals**

- Anything Roark's fight cannot reach: doubles, gauntlets, Trick Room, Gravity, other
  weather, and moves, abilities and items that neither side can bring.
- The exhaustive prep gate on Kaizo Roark (K4), and the exp-cap warning.
- Pokémon the user catches after this save. They are follow-up work, done the same way.
- Approving the K2b discrepancy report. That review is still open, and K3 does not wait
  for it.

**Anti-goals** (failure even if the command runs)

- A refusal removed by approximating a mechanic, or a ROM change marked irrelevant
  without evidence.
- A pool option that prep quietly drops, so that coverage says EXACT while prep cannot
  actually use it.
- A vanilla fact used for Kaizo without a check that Kaizo kept it (id lists, AI
  constants, item parameters, move data).
- A vanilla behaviour that changes. The vanilla suite stays green, and vanilla Roark keeps
  its exact results: coverage, enumerated probabilities and Stage B values. My call:
  seeded Monte Carlo numbers (Stage A estimates) may shift when the shared battle
  `Outcome` gains fields such as Thunder Fang's second secondary, Roar's target or
  Present. They are estimates, the code version already keys checkpoints and the compile
  cache, and the new draws stay inside the existing confidence intervals. Vanilla's
  `AiOutcome` layout does not change at all (§6).

## 3. Where things stand (2026-11-07, main at 5454b2b)

Running the command on the save gives `Support: NONE` with 54 missing on the fight and 38
missing in the pool (cap 16, 12 Pokémon).

**Roark's side** (trainer 246; the party is in kaizo-data-design §10):

- Moves: Self-Destruct, Brick Break, AncientPower, Recover, Thunder Fang, Light Screen,
  Toxic, Curse, Sand Tomb.
- Abilities: Solid Rock, Sand Veil.
- Held items: Focus Sash, Big Root, Chesto Berry, Rindo Berry, Berry Juice, Lum Berry.
- Battle start: Sandstorm, permanent (the map's weather).
- AI: no transcribed Kaizo policy, `AI_FLAG_RISKY`, and the script paths of all 19
  distinct moves of his.
- `Kaizo ROM: 103 code changes unclassified`.

**The player's side** (the party, box 1, the relearner moves and bag TMs at cap 16):

- Abilities: Rough Skin, Flame Body, Swift Swim, Limber, Snow Cloak, Cute Charm,
  Adaptability.
- Moves: Toxic (TM06, open to the whole pool), Swift, Shock Wave, Roar, Pay Day, Aqua
  Ring, Flail, Nature Power, Camouflage, Protect, Fake Out, Present, Attract, DoubleSlap,
  Aurora Beam, Gust.
- Held: Quick Claw (Prinplup).
- In the bag, as candidates for held items: White Herb, Lagging Tail, Pecha Berry, Chilan
  Berry (Oran Berry is supported already). The pool check does not list bag items today;
  see §4.4.

The engine supports 3 held-item effects (none, Oran, Sitrus), 21 abilities and 29 move
effects. The state already has fields for the screens (`reflect`, `light_screen`) and the
weather. The vanilla golden trace for Roark has never been recorded; the recording script
still has addresses marked VERIFY (`docs/tracing.md`).

## 4. The mechanics

**Status (K3b1 done, 2026-11-08).** Landed in both engines: §4.2 permanent sandstorm, field
pass, Light Screen and Brick Break; §4.3 Toxic, Sand Tomb, Swift family, Pay Day, Gust,
Recover, Fake Out, AncientPower, Curse, DoubleSlap, Self-Destruct, Flail, Aurora Beam; §4.4
Solid Rock, Sand Veil, Rough Skin, Flame Body, Swift Swim, Limber, Snow Cloak, Adaptability,
Focus Sash, Big Root, Chesto/Rindo/Berry Juice/Lum/Pecha/Chilan berries, White Herb, Lagging
Tail, Aqua Ring; obedience (§5). Still open for K3b2: Thunder Fang, Protect, Present, Attract,
Cute Charm, Quick Claw, Roar, Nature Power, Camouflage, terrain. Per-rule audit:
`docs/kaizo/k3b-rule-audit.md`.

### 4.1 Rules for every new mechanic

These are the existing project rules, restated because K3 roughly doubles the engine:

- Behaviour comes from the decomp (`external/pokeplatinum`), cited by file and line. Where
  Kaizo changed the code or data, it comes from the patched ROM: a decoded diff of the
  changed bytes, or the ROM tables K2b extracted. PKCalc and the spreadsheet are pointers
  and cross-checks, never the source.
- Each mechanic goes into the JAX engine and the reference engine, with differential
  tests. Damage-related mechanics are also checked against the damage-array oracle
  (`tools/smogon_calc`) where the oracle models them.
- Each one is registered in `engine/coverage.py` as `EXACT`, carries its confidence class
  (decomp-verified, ROM-verified, golden-traced), and adds its random draws to `Outcome`
  in the documented layout.
- A mechanic that depends on HP adds its cut to `prep/variants.py` (`hp_cuts`). Examples:
  Focus Sash at full HP, Flail's bands, the pinch and berry thresholds.

### 4.2 Field: permanent sandstorm and screens

- **The field-condition pass.** Add the decomp's `CheckFieldConditions` step to the end of
  the turn in Platinum's order, for weather and the screen timers. Today the engine has no
  such pass, which is why every weather is refused.
- **Sandstorm**: end-of-turn damage of 1/16 max HP (the decomp's rounding), skipping
  Rock, Ground and Steel types, Sand Veil, and Pokémon underground from Dig. Rock types
  get ×1.5 Special Defense. Map weather never runs out. Cloud Nine (already supported)
  suppresses weather.
- **Weather-reading effects in the pool**: Sand Veil's evasion; Swift Swim and Snow Cloak
  do nothing in sandstorm, but must be modelled exactly enough to be `EXACT`, not
  waved through.
- **Light Screen**: 5 turns, with the decomp's singles reduction and the crit bypass;
  Brick Break removes it.
- Only sandstorm becomes `EXACT` in `IMPLEMENTED_WEATHER`. Other weathers stay refused
  unless a move in the pool can set them; none can.

### 4.3 Moves

Roark: Self-Destruct, Brick Break, AncientPower, Recover, Thunder Fang, Light Screen,
Toxic, Curse (the non-Ghost form), Sand Tomb. Player: Toxic, Swift, Shock Wave, Roar, Pay
Day, Aqua Ring, Flail, Nature Power, Camouflage, Protect, Fake Out, Present, Attract,
DoubleSlap, Aurora Beam, Gust.

Notes for the builder:

- **Kaizo move data comes from the ROM**: power, accuracy, priority, effect id and effect
  chance. Water Pulse never misses, Accelerock and Stealth Rock have +1 priority. Where
  the ROM effect id is a repurposed vanilla effect, the mechanic is that effect's script.
  Sand Tomb decodes to Whirlpool's effect (261); check the trap and the dive interaction
  against the decomp.
- **Any move whose battle script Kaizo changed** (`be_seq` members 48, 88, 104, 141, 182)
  stays unsupported unless its member is classified and modelled (§5). Head Smash is
  already supported; confirm its effect script is not one of these.
- **Nature Power and Camouflage** depend on the battle's terrain. Take Roark's terrain
  from the ROM (his map's battle background), not from vanilla.
- **Roar** in a trainer battle forces a random switch-in. Its target draw goes into
  `Outcome`.
- **Present's** damage-or-heal draw is already named in the `Outcome` layout in
  `design.md`. Thunder Fang has two separate 10% secondaries (paralysis and flinch).
  AncientPower's raise-all-stats effect is a secondary.
- **Flail** reads the user's HP band. Its bands become HP cuts in prep.

### 4.4 Abilities and held items

Abilities: Solid Rock, Sand Veil (Roark); Rough Skin, Flame Body, Swift Swim, Limber, Snow
Cloak, Cute Charm, Adaptability (player). Contact procs (Rough Skin, Flame Body, Cute
Charm) are draws already reserved in `Outcome`. Check Kaizo's ability changes against the
K2b ability table.

Held items: Focus Sash, Big Root, Chesto Berry, Rindo Berry, Berry Juice, Lum Berry
(Roark); Quick Claw, White Herb, Lagging Tail, Pecha Berry, Chilan Berry (player, held or
in the bag). Item parameters come from the Kaizo item table (K2b), since Kaizo changed 124
item records.

**My call: the pool check also lists held items from the bag.** Prep assigns held items
from the bag, so an unsupported bag item is an option prep cannot use. "Everything" has to
include them: `check_pool` gains one row per bag item that has a battle hold effect in
the game's item table. Items with no hold effect (Honey, Stardust, shards, Pearl, Moon
Stone, Heart Scale, Rare Candy, medicine) do nothing when held. Prep leaves them out of
held-item assignment and they are not rows, so the vanilla item table is unchanged.
Today's bag gives four rows: White Herb, Lagging Tail, Pecha Berry and Chilan Berry.

### 4.5 The prep side of "everything"

Prep search excludes an unsupported candidate with a warning (design.md §3). After K3,
running it on this save should exclude nothing. One test runs the pool check on the save
(skipped when the save is absent) and asserts it is empty.

## 5. The 103 ROM changes

The count is 95 code entries in `data/kaizo/rom_diff.toml` (43 arm9, 2 overlay5, 13
overlay13, 32 overlay14, 5 overlay16) plus 8 changed battle-script members (`be_seq` 48,
88, 104, 141, 182; `sub_seq` 238, 251, 297). Today one `unknown` refuses every Kaizo
fight (K2b Ruling 13).

**Each change is decoded, then classified.** Decoding means diffing the patched bytes
against the vanilla function, naming what changed (a constant, a branch, a table entry, a
call), and writing that into the entry's `note` with the decomp citation. A classification
without a decoded note is not allowed.

**Statuses (my call: a fourth status).**

- `modelled`: the engine does what the patched code does, and a test says so.
- `irrelevant`: provably cannot change any battle outcome or AI choice. Examples are map
  headers, field scripts, menus and cursors, wallpapers, prize money, and the overlay13
  bag and party cursor code. The note says why.
- `not_reached` (new): battle-relevant but not modelled, with a `when` condition that
  coverage evaluates per fight. Examples: `TrainerAI_MainDoubles` with `when = "format !=
  single"`; `sub_seq` 238 and 251 with `when = "area_condition gravity"` and `"trick_room"`;
  `be_seq` 88 with `when = "move_effect 88"`. A fight is refused only when one of these
  conditions holds for it. The conditions are a small, fixed vocabulary (format, area
  condition, a move effect, an ability, an item, a map weather), so they can be checked by
  code. A free-text condition is not allowed.
- `unknown` keeps today's meaning: it refuses every Kaizo fight.

**Groups the builder will meet** (the first reading; each is confirmed by decoding):

- arm9 map headers, field scripts, encounters, wallpapers, the TM list, the item archive
  ids, hidden items, spawn points: most likely `irrelevant` to battle. `sTMHMMoves` and
  `sTeachableMoves`/`sSpeciesLearnsetsByTutor` (overlay5) feed the pool's TM and tutor
  moves, so they are checked against the K2b learnset tables, not just waved through.
- `TrainerData_BuildParty` (1033 bytes) was decoded in K2b (`rom_diff.md`, "decoded").
  Its classification follows that decoding: `modelled` by the K2b party builder.
- `PartyMenuCB_LevelUp`, `HandleInput_Main`, `SetMonDataFromMon`: these call the injected
  code in `weather_sys.narc` member 9. What that code does must be found before they are
  classified, because it may touch Pokémon data such as level caps or stats.
- overlay14 (32 routines, mostly 1-2 bytes): the Kaizo AI's command handlers, including
  `TrainerAI_CalcDamage`, `TrainerAI_ShouldSwitch` and `TrainerAI_ShouldUseItem`. Each
  becomes `modelled` in the Kaizo policy (§6) or `not_reached`.
- overlay16: `CheckMoveHitAccuracy` and `CheckMoveHitOverrides` (accuracy, which
  matters to every move), `CheckObedience`, and `BattleSystem_TriggerEffectOnSwitch`
  (210 bytes: map weather and the field conditions at switch-in). All four are on
  Roark's path; the K3a outcome below settles them: `CheckObedience` stays `unknown` (K3b),
  the two accuracy routines are `not_reached` on `move_effect` 141, and
  `BattleSystem_TriggerEffectOnSwitch` is `modelled`.
- `be_seq` and `sub_seq` members: `not_reached` with a move-effect or area-condition
  condition, unless a pool move uses them.

**K3a outcome (2026-11-08): where decoding moved work.**

- overlay14: the AI script (gTrainerAITable) equals vanilla's except one word (Dry Skin 87 in
  the second Levitate compare of `Basic_CheckForImmunity`); every other overlay14 change is a
  re-encoded call. K3c needs almost nothing beyond Dry Skin.
- overlay16: only `CheckObedience` stays `unknown` (always obeys, no RNG draw; K3b).
- `be_seq` 48, 88, 104, 141, 182: `not_reached` with `move_effect`; Roark and the pre-Roark
  pool use none of them.
- Battle-RAM offsets are statically verified against both ROMs (`tools/rom/offsets.py`); the
  in-emulator probe is pending.

## 6. The Kaizo trainer AI

- **A per-game policy.** The Kaizo policy reuses the vanilla script transcription where
  the bytes are unchanged and replaces exactly what the overlay14 diff changed. The
  builder first confirms where the AI script bytes live in the ROM and whether Kaizo
  changed them (`battle/tr_ai/tr_ai_seq.narc` is not among the changed files).
- **Flags**: Basic, EvalAttack, Expert and Risky (Roark's 23). Risky is transcribed from
  `script.s` like the other three, with its random draws added to `AiOutcome`.
- **Script paths** for all 19 of Roark's moves, and for every player move the scripts
  branch on (`player_move_refusal`, `player_ability_refusal`).
- **Id-keyed lists** (sound moves, the punch list, the `TrainerAI_CalcDamage` cases for
  Psywave, Magnitude, Return and Frustration, and the switching move ids) are read per
  game from the Kaizo ROM, never copied from vanilla. Repurposed slots matter: Psywave's
  slot is Triple Axel and Magnitude's is Bulldoze.
- **The spreadsheet's "AI Changes"** (the Dry Skin fix, the recoil routines, Risky for
  Triple Axel and Fury Cutter, Hurricane) are checked against the decoded overlay14
  changes. Any that touch Roark's fight are modelled; the rest are `not_reached`.
- **Six-Pokémon switching** goes through the patched `TrainerAI_ShouldSwitch`.
- **Width (my call: a per-game AI outcome layout, planned from the start).**
  `ENUMERATION_WIDTH` is 3840 of 4096 today (1 × 2 × 2^4 × 10 × 12). One more script-roll
  site doubles it to 7680, and Expert plus Risky over 19 moves will almost certainly add
  one. So overflow is expected, not hypothetical. The sampling-only grade would make
  coverage print `SAMPLING_ONLY`, which fails Goal 1, and it would give up exact plan
  trees on a fight where safety matters most. So K3c builds the layout per game: the
  `AiOutcome` fields and their ranges come from the trainer's AI flags and movesets, the
  vanilla layout stays exactly as it is, and the Kaizo Roark layout is sized from his
  actual script-roll sites. Draws that cannot change the chosen action are merged, as
  today. If the merged exact width still exceeds what Stage B can handle in time, the
  builder stops and reports measured numbers (width, Stage B time per node) before
  touching `MAX_ENUMERATED` or the width cap. Never approximate.
- `docs/ai/kaizo_roark.md` is written the way `docs/ai/roark.md` was: every rule cited.

## 7. The golden trace

- **Preparing the kit (builder, before the user plays).** Check `tools/trace_battle.lua`
  against Kaizo from the ROM: the BattleContext layout (overlay16 grew by 32 bytes) and
  how the script finds the battle. Make `trace_to_state` load the Kaizo data set. Write
  `docs/tracing_kaizo.md`: one page of exact steps for the user.
- **Address check (user, in K3a; decided by the user 2026-11-07: "Short check early").**
  The builder cannot run DeSmuME, and the addresses marked VERIFY in `docs/tracing.md`
  have never been confirmed. The builder writes a small probe script (`tools/
  probe_battle.lua`) that prints the values at each VERIFY address with what each is
  expected to hold. The user starts any battle on a save-state copy, runs the probe for
  about 10 minutes, and sends the output. The builder fixes any wrong address before
  K3d, so a bad address never costs a full Roark recording.
- **Recording (user, once the engine can replay the fight).** Load a save state made from
  a copy of the pre-Roark save, start the script, walk in and fight Roark with any lead
  and moves. Whether the run is won or lost does not matter, and the real run is never
  touched. Raw traces stay in git-ignored `traces/`; only the converted bundle
  (`tests/golden/kaizo_roark_01.json`) is committed.
- **Acceptance.** Every recorded turn replays through `step` with the realized outcome and
  matches on every compared field. The trace also checks the AI: each recorded opponent
  choice must have nonzero probability under the Kaizo policy.
- **A second recording** is requested only when the first one leaves something
  load-bearing unexercised: Focus Sash, Sand Tomb, a Risky choice, a switch, sandstorm
  chip, or Light Screen. `docs/tracing_kaizo.md` lists which mechanics the trace
  exercised.
- A mismatch is a bug in the engine or the AI. It is fixed at its source, never patched
  over in the converter.

## 8. Order of work

K3 is about as big as M1, so it is split into four parts. Each part ends with the suite
green, and each part's plan is written when the previous part lands.

- **K3a: the ROM changes and the recording kit.** Decode and classify the 103 changes, add
  `not_reached` with `when`, make coverage evaluate it per fight, check the recording kit
  against Kaizo, and end with the user's address check. This comes first because a decoding may move work into the later
  parts, for example something in overlay16 or the injected code.
- **K3b: the field and the mechanics.** The field-condition pass, sandstorm, screens, and
  every move, ability and item in §4. Each one goes through both engines with its tests.
- **K3c: the Kaizo AI.** The per-game policy, Risky, script paths, id lists, switching,
  the width check, and `docs/ai/kaizo_roark.md`.
- **K3d: the trace and acceptance.** The user records, the bundle is converted and
  replayed, the exit test runs on the save, and `docs/kaizo/pipeline.md` and
  `design.md` are updated.

The user is needed three times: to approve this design, for the 10-minute address check
at the end of K3a, and to record the trace at K3d. The only other planned stop is if even
the exact per-game AI width is too slow (§6).

## 9. Done means

1. `uv run nuzline coverage --game kaizo --save <pre-Roark save> --trainer roark` exits 0,
   with `Support: EXACT` for the fight and for the pool (bag held items included).
2. `data/kaizo/rom_diff.toml` has no `unknown` entries, and every entry has a decoded
   note.
3. `tests/golden/kaizo_roark_01.json` replays with no mismatch, and its AI choices are all
   possible under the Kaizo policy.
4. The whole suite passes, vanilla included, with no vanilla result changed.
5. `docs/kaizo/pipeline.md` "What is refused" and `design.md` §9 describe the new state.

## 10. Calls I made (please check)

- K3 covers the user's whole current pool, including holdable bag items, because "do
  everything" was the answer and prep would otherwise drop options silently.
- A new ROM-change status, `not_reached`, with a machine-checked per-fight condition,
  instead of modelling all 103 changes. Changes only doubles, Gravity, Trick Room or
  specific moves can reach do not block Roark, and they still block the fights that reach
  them.
- K3 is split into K3a to K3d, with one plan per part. The plan written with this design
  is K3a's.
- The trace is recorded from a copy of a save state with the user's real pool, not a
  scripted team. Any lead or line is fine, and a second recording happens only when
  mechanics are left unexercised.
- A per-game AI outcome layout is built in K3c from the start, because Roark's Kaizo AI
  almost certainly overflows the shared one. Work stops only if even the exact per-game
  width is too slow for Stage B.
- Only bag items with a battle hold effect count as pool options.
- "No vanilla change" means the same exact results. Seeded Monte Carlo estimates may
  shift.

## 11. For the builder to settle from the ROM (no user input needed)

- What the injected code in `weather_sys.narc` member 9 does.
- Roark's battle terrain (for Nature Power and Camouflage), and whether his map weather
  is a plain sandstorm (map weather 3) or a Kaizo variant.
- Whether Kaizo changed any of the moves, abilities or items in §4 beyond the table data.
- Where the AI script bytes live and whether they changed; what each overlay14 byte change
  does.
- Whether Kaizo's BattleContext layout matches vanilla's (for the trace).
