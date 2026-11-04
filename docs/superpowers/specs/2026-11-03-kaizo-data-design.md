# Platinum Kaizo as a second data set — design

Status: approved by the user, 2026-11-03. Parent: `docs/design.md` (§2 scope, §4.1 data layer, §6
caches, §9 extensibility). Related: `docs/superpowers/specs/2026-10-31-prep-search-design.md`
(milestone 4, whose non-goals deferred "reading data from a ROM, and difficulty hacks" to this
work).

This is a data design. It adds Pokémon Platinum Kaizo ("Nightmare Update Final") beside vanilla
Platinum US, sources every Kaizo fact, says how each is verified, and defines how nuzline switches
between the two games. Engine work that Kaizo needs (new effects, AI flags, permanent field
conditions, doubles) is listed so it can be planned, but each engine piece is its own milestone.

## 1. Intent

The user plays Kaizo with nuzline: a real Kaizo nuzlocke, every fight, not a test fixture. Vanilla
Platinum stays fully supported. Kaizo is also the harder proving ground nuzline lacks: milestone
4's vanilla Roark exhaustive gate failed to discriminate because nearly every team wins 98–100%;
Kaizo Roark (six Pokémon, IV 30, items, permanent sandstorm, Risky AI) spreads win rates out.

The user's priorities carry over unchanged: safety first, and accuracy of data over speed. A Kaizo
fight the engine cannot model exactly is refused with the missing piece named, never approximated.

The user is happy to develop as they play: gaps are found and fixed fight by fight, which the
fail-loudly rule makes safe.

## 2. Goals, non-goals, anti-goals

**Goals**
- A complete Kaizo data set: species (with forms), moves, learnsets (level-up, TM, tutor), TM and
  tutor contents and costs, items (held effects), abilities, trainers (party, items, IVs,
  nature/gender/ability, AI flags, battle format, partner), start conditions (permanent weather,
  fog, permanent Trick Room, permanent Gravity), gauntlet groups, level caps, exp yields.
- Every fact sourced from the patched ROM, PKCalc or the Kaizo docs spreadsheet, with a recorded
  source and a disagreement report; nothing silently picked.
- A per-game switch that keeps vanilla untouched and makes every run reproducible.
- Reading the user's Kaizo saves.
- Kaizo Roark as the first Kaizo coverage target, then a second exhaustive validation gate.

**Non-goals (this design)**
- Engine support for doubles, two-trainer doubles, multi/tag battles and gauntlets. The data
  records them in full; each format gets its own design later (§11).
- Implementing the new engine mechanics. They are listed (§5, §11) so milestones can be cut.
- Planning across fights, other ROM hacks, other Kaizo versions. The pipeline is written so a
  later version or hack is a new manifest, not a rewrite.

**Anti-goals**
- Silently using vanilla data, vanilla start conditions or vanilla AI behaviour for a Kaizo fight.
  202 of 454 id-joined trainers have different start weather from vanilla (§5.8); vanilla's
  hard-coded AI move lists point at repurposed slots (§5.10).
- Committing or distributing ROM bytes, or anything that only makes sense with the ROM.
- Treating a format the engine does not model as a single battle (§5.8 notes vanilla has this bug
  today).

## 3. Decisions made with the user

| Topic | Decision |
|---|---|
| Purpose | Full Kaizo support for the user's own run, plus a harder validation fight. |
| Formats | nuzline is meant to handle all formats eventually. This design captures every format in data; engine support comes in later projects, one format at a time. |
| Order | Kaizo data and singles first, Kaizo Roark as the first target; formats later, each with its own check fight (Volkner for doubles, a tag battle, the gauntlet gym). |
| Source priority | PKCalc beats the spreadsheet ("most current and accurate"; the spreadsheet "may have missing information"). The ROM beats PKCalc on anything the game executes: effect ids, AI scripts, trainer structures, start conditions. Disagreements are reviewed, never resolved silently. |
| Field effects | Permanent weather, fog, Trick Room and Gravity are set per fight. They never expire and moves never replace them. Weather **abilities** (Drought, Drizzle, Sand Stream, Snow Warning) do replace permanent weather, as in vanilla. |
| Fog | As vanilla: accuracy-checked moves have hit chance × 0.6 (`outcome.py`, `rate*6//10`); No Guard and never-miss moves always hit. |
| Moves inside permanent effects | Not a concern for the player: the player can never obtain those moves or abilities. Opponent cases found in data are in §5.8. |
| Level caps | The user's table (§5.12). Soft caps: the game never stops exp; the cap is the player's rule. |
| ROM | User supplied the patch (`Pokemon Platinum Kaizo Nightmare Update Final.xdelta`) and a clean vanilla ROM; the ROM may override PKCalc on game logic. |
| Saves | `Platinum Kaizo Nightmare(Maylene).sav` (after Maylene) is the Kaizo save-reading fixture. The user will supply a pre-Roark save for the gate; do not synthesise one. |
| Unmarked weather | "(RAIN)"/"(HAIL)" sections without PERMANENT are assumed permanent. |
| Rivals, optional trainers | Only the rival variant for the save's starter is shown; optional trainers are marked skippable. |
| Bag | Prep search uses only what the save's bag holds; Kaizo item locations are not needed. |
| Gauntlets | Rules vary per gauntlet (heal allowed or not, team changes allowed or not); recorded when play reaches each. |
| Checking | The user checks mechanics in-game as play reaches them; the decomp toolchain may be installed. |

## 4. Sources and authority

| Source | What it is | Authority |
|---|---|---|
| **Patched ROM** | Vanilla Platinum US (`CPUE`, rev 1, SHA-1 `0862ec35b24de5c7e2dcb88c9eea0873110d755c`) + the Nightmare Update Final VCDIFF patch. The vanilla SHA-1 equals the decomp's build target (`external/pokeplatinum/INSTALL.md:334`), so every changed byte maps onto a named decomp file or function. | Highest for executed behaviour: move/species/trainer/item/encounter NARCs, effect and AI scripts, ARM9/overlay code. |
| **Golden traces** | `tools/trace_battle.lua` run on the patched ROM in an emulator (`docs/tracing.md`). | Final word on behaviour the ROM data can mislead about (§5.10, "decoy" effect ids). |
| **PKCalc** | The official Kaizo damage calculator (`PKCalc-main.zip`; a fork of @smogon/calc 0.10.0). Data: `src/js/data/dex/{species,moves,abilities,items,locations,types,natures}.js`, `sets.js` (2112 trainer sets), `party_order.js`, `trainer_flags.js` (AI flags), `flags.js` (battle formats, weather, fog, hidden trainers). Mechanics: `calc/src/data/dex/moves.ts` (`PK_PATCH`, 174 entries), `calc/src/mechanics/gen4.ts`, `calc/src/items.ts`. UI rules: `src/js/shared_controls.js` (Gravity for Champion Cynthia, line 611). | Primary for names, descriptions, numbers, trainer sets, AI flags, formats. |
| **Kaizo docs spreadsheet** | `Platinum Kaizo Docs.xlsx` (39 sheets). Useful: `Moves` and `Move Changes` (effect labels), `Personal`, `Ability Changes`, `Evolutions`, `Level-Up Learnsets`, `TM Learnsets`, `Items` (hold effect and parameter; join by name, ids are packed), `AI Changes`, the ten `* Split` sheets (section headers carry PERMANENT RAIN/SUN/SANDSTORM/TRICK ROOM/GRAVITY, FOG, DOUBLE/MULTI/TAG, GAUNTLET, OPTIONAL, Left Side), `RAW TRAINER DATA`/`RAW TRAINER POKEMON` (ROM trainer ids, difficulty, class, Ability Number), `RAW Field Enc`/`RAW Water Enc`/`Swarm, Day, Night Encounters`. Not Kaizo: `Field Encounters` and `Plat Kaizo PokEditor Tester` are unmodified vanilla exports. | Secondary: cross-check, and the only source for Trick Room areas, gauntlets, item hold parameters and effect labels. |
| **Vanilla decomp** | `external/pokeplatinum` at the commit in `data/gen4_platinum/manifest.json`. | Layouts and semantics of every ROM structure; the vanilla baseline the diff is taken against. |
| **User** | Rules, level caps, field-effect semantics (§3). | Rules and intent. |

Conflict rule: ROM (and traces) > PKCalc > spreadsheet, per field, and every conflict is a line in
`docs/kaizo/discrepancies.md` with the chosen value and why. A conflict on a field
the engine executes is a hard build failure until it has a resolution entry (§6.5).

## 5. What Kaizo changes, and how each is sourced and verified

Counts below come from a cross-source audit run on 2026-11-03 (scripts and outputs saved outside
the repo in the session notes folder, `kaizo-audit/`). They describe the data as supplied; the
pipeline recomputes them from the ROM.

### 5.1 Ids

Kaizo keeps every vanilla id space: 493 species (+ forms at personal indices 496–507), 467 moves,
123 abilities, the vanilla item numbering, 928 trainer ids, 25 natures, the vanilla 17 types +
`???` with an unchanged type chart (0 of 324 cells differ). No Fairy type, no new abilities.
Slots are **repurposed**: 26 moves renamed (Comet Punch → Water Ball, Rollout → Accelerock,
Psywave → Triple Axel, Magnitude → Bulldoze, Splash → HP Water, …), 5 items renamed (Shoal Salt →
Berserk Gene, Shoal Shell → RageCandyBar, …), and 347 of 928 trainer slots reused for different
trainers.

Consequence: everything keyed by **name** must be re-keyed by **id** per game, and everything keyed
by **id** must be re-checked, because an id can now mean a different thing (§5.10, §7.3).

### 5.2 Moves

PKCalc differs from vanilla on 332 moves: power 156, accuracy 124, PP 117, effect chance 44,
type 34, target 33, **priority 30**, category 27, name 26.

Priority changes (PKCalc vs vanilla): Teleport, Transform, Spider Web, Spikes, Accelerock, Mean
Look, Sleep Talk, Baton Pass, Gravity, Me First, Copycat, Toxic Spikes, Magnet Rise, Stealth Rock,
Defend Order, Lunar Dance 0→+1; Tail Glow, Lucky Chant 0→+2; ExtremeSpeed and Sucker Punch →+2;
Fake Out +1→+3; Tailwind 0→+5; Block 0→+7; **Trick Room −7→+7**; Bide +1→−1; Constrict 0→−1;
Metal Burst 0→−2; Revenge and Avalanche −4→0; Snatch slot (now Drill Run) +4→0. Priority is read
from the per-game table; nothing assumes vanilla priorities. The engine's priority range must hold
−7..+7 (int8 does; the builder checks every consumer, including AI priority logic).

Effects. The spreadsheet's "Additional Effect" column is PokEditor's label for the effect id (in
the vanilla export, 257 labels map one-to-one onto the 257 effect ids). Reading the labels back
recovers a Kaizo effect id for 90 of the 101 effect-changed moves, and almost all are **reused
vanilla effects**: Whirlpool (261) for ViceGrip/Wrap/Submission/Fire Spin/Clamp/Crush Grip/…;
Volt Tackle's recoil+paralysis (262) for Thrash, Sky Attack, Wild Charge, Frenzy Plant; Flare
Blitz's recoil+burn (253) for Eruption, Blast Burn, Overheat; Blizzard's hail accuracy (260) for
Ice Ball and Sheer Cold; Metal Burst (227) for Bide; Triple Kick (104) for Triple Axel, Fury Cutter
and Rock Wrecker; bypass-accuracy (17) for Aurora Beam; plain HIT (0) for 27 "No Effect" moves.
Eleven moves carry six labels that are new and mark genuinely new behaviour: "1/2 damage dealt
recoil, fixed AI" (six moves: Hyper Beam, Outrage, Superpower, Hydro Cannon, Giga Impact, Draco
Meteor), Head Smash "VANILLA AI", Hurricane (always hits in rain, own effect: the vanilla
never-miss check is keyed on `BATTLE_EFFECT_THUNDER`, `battle_controller_player.c:3025-3030`),
party-wide Magnet Rise, gender-free Captivate, Minimize.

Known conflicts that only the ROM or a trace settles:
- Triple Kick family per-hit power. Spreadsheet: Triple Axel 40, Fury Cutter 30, Triple Kick 30,
  Rock Wrecker 40, "+10 each hit". PKCalc data: 20/20/30/30; PKCalc `gen4.ts`: 20/40/60 for Axel
  and Cutter, 30/60/90 for Kick and Wrecker. Vanilla `effect_script_0104.s:7` adds a hard-coded
  10 per hit regardless of base power.
- Roar of Time: recharge (spreadsheet label) or 1/3 recoil + 10% paralysis (PKCalc). Hurricane
  confusion 10% (sheet) or 30% (PKCalc). Kinesis: accuracy (sheet label, a known template bug for
  id 62) or Sp. Def −2 (PKCalc). Swords Dance PP 30 (sheet) or 2 (PKCalc). Constrict 100 BP Ice
  (sheet) or 255 BP Psychic −1 priority (PKCalc). 19 PKCalc/spreadsheet disagreements in all.
- Secondary effects with chance 0 (e.g. Ice Ball, decoded as Blizzard's effect 260 with chance 0)
  never fire. `build_tables.py` raises on chance 0 for effects mapped to a secondary-effect class
  outside `CERTAIN_SECONDARY_EFFECTS`; once 260 and similar effects are mapped, that check must
  accept chance 0 as "no secondary".

Sourcing: ROM `poketool/waza/pl_waza_tbl.narc` for every numeric field and the effect id; PKCalc
for names and descriptions; spreadsheet labels as an independent decode of effect ids.
Verification: three-way field diff; ROM-vs-label effect id diff; traces for the conflicts above
and for every new effect (§9).

### 5.3 Species, forms and learnsets

Vs vanilla (base species): base stats 208, abilities 335, level-up lists 487, TM compatibility
bits 119 (476 species can learn a different set of moves by TM, mostly because six TMs now teach
other moves), tutor compatibility 21 on the shared tutor moves (479 counting the swapped tutor
moves), evolutions 104, catch rate 452, exp yield 15, growth rate 11, gender ratio 2; types and
weight unchanged. Forms: 12 form entries. The 3 Deoxys forms (HP 90, Magic Guard) and 5 Rotom
appliances (Spe 91, own level-up lists) change stats; Wormadam-Sandy and -Trash change only their
ability (Snow Cloak); Giratina-Origin and Shaymin-Sky are unchanged. PKCalc and the spreadsheet
agree on forms.

TM contents changed in 6 slots: TM07 Hail → Aurora Beam, TM08 Bulk Up → Vital Throw, TM12 Taunt →
Faint Attack, TM15 Hyper Beam → Snore, TM17 Protect → Tackle, TM92 Trick Room → Miracle Eye.
Tutors: 38 slots, 6 swapped (out: Snore, Mud-Slap, Uproar, Trick, Dive, Air Cutter; in:
Earthquake, Sky Attack, Role Play, Hyper Voice, Tickle, Aqua Ring), 31 of the 32 shared moves
re-priced; PKCalc places all at Route 212. Egg moves are in neither source.

Sourcing: ROM `poketool/personal/pl_personal.narc` (stats, types, abilities, exp yield, growth,
TM compatibility bits), `poketool/personal/wotbl.narc` (level-up), `poketool/personal/evo.narc`,
the tutor table and TM→move table from ARM9/overlay data (located through the decomp), egg moves
from the ROM's egg-move table. PKCalc and the spreadsheet cross-check. Verification: three-way
diff; spreadsheet disagreements (catch rate on 121 species, 27 level-up lists, 6 growth rates, 13
exp yields, Fearow Sp. Atk 31 vs 61, 3 evolutions, 12 stale `Ability Changes` rows) resolve to
the ROM.

### 5.4 Trainers

928 ids kept; 581 keep their vanilla name, 537 parties changed; 178 trainers have six Pokémon
(vanilla: 23). PKCalc has 485 real trainers (2112 sets) plus 10 empty rival templates; 473 map
uniquely to ROM ids by species order, 12 have no spreadsheet counterpart (Barry #6, Sinister
Hooded Figure, Lady Emi, Riley @ Iron Island, …). EVs are 0 everywhere. No PKCalc-matched trainer
has bag items. Rival and partner sets vary with the player's starter.

**The vanilla PID routine still holds.** Re-running nuzline's `TrainerData_BuildParty` logic
(`src/nuzline/data/trainer.py`) on the ROM-style inputs reproduces PKCalc's nature and gender for
1816 of 1936 Pokémon with Ability Number 0 (93.8%) and the ability for 1933; 424 of 473 trainers
reproduce completely, **Kaizo Roark 6/6**. Of the 120 nature/gender misses, 54 are recovered by
a different difficulty value that still gives PKCalc's IVs, and 3 trainers by a changed class
(the spreadsheet predates PKCalc's ROM build); 66 remain unexplained and are settled by the ROM.

**Kaizo adds a per-Pokémon override** that vanilla lacks: the spreadsheet's Ability Number byte
(values 16, 32, 2 on 111 Pokémon) forces ability slot 1, slot 2, or female. On those Pokémon
PKCalc's natures do not follow the vanilla PID and no tested variant explains them. Vanilla's
trainer struct has a `u16 ivScale` (`include/struct_defs/trainer_data.h:35`), so Kaizo changed the
struct and `TrainerData_BuildParty`.

Sourcing: ROM `poketool/trainer/trdata.narc` and `trpoke.narc` (party, difficulty, override byte,
class, battle type, AI flags, bag), parsed with the Kaizo struct layout, which is confirmed by
diffing the patched `TrainerData_BuildParty`. Nature, gender and ability are computed by the
patched routine, transcribed; until that is done they are **explicit per-Pokémon data from PKCalc**
(`PokemonSpec` already carries them), and the PID reproduction is a cross-check. Verification: ROM
vs PKCalc for every set; the PID check over all trainers; a trace or in-game check of one override
Pokémon (e.g. Maylene's).

### 5.5 Items

Same numbering. Hold-effect changes: Lava Cookie and Old Gateau become Lum clones; Shoal Salt
becomes Berserk Gene (pinch Attack, param 0); Shoal Shell becomes RageCandyBar (heals 100 HP at
50%). Parameter changes: all 16 Plates 20 → 50 (×1.5), weather rocks 3 → 7, Jaboca/Rowap 1/8 →
1/6, BrightPowder/Lax Incense 10 → 11, Ganlon/Apicot 4 → 0 ("trigger immediately"). Fling power:
Town Map 0 → 120, Bicycle 0 → 150, slot 92 (Nugget, renamed Big Nugget by PKCalc) 30 → 200 (sheet)
or 250 (PKCalc). PKCalc has no numeric hold
fields; the spreadsheet does. Neither source says where items, TMs or shards are obtained.

Sourcing: ROM `itemtool/itemdata/pl_item_data.narc`; spreadsheet `Items` (joined by name);
PKCalc for names and descriptions. Pinch berries with param 0 need an explicit rule (the vanilla
formula divides by it), confirmed by trace. Verification: three-way diff; traces for Plates,
param-0 pinch berries and RageCandyBar.

### 5.6 Abilities, natures, type chart

All 123 ability names match vanilla, and PKCalc's descriptions and `gen4.ts` ability code read as
stock Gen 4: no evidence of changed ability effects. This is "no evidence", not proof: the ARM9
diff (§6.2) is what rules it out. Natures and the type chart are identical to vanilla.

### 5.7 Wild, gift and static availability

Kaizo encounters are in `RAW Field Enc`, `RAW Water Enc` and the swarm sheet: 183 vanilla tables
plus 19 appended. PKCalc `locations.js` matches 106 of 117 grass and 59 of 60 water locations
exactly and lacks swarms, Poké Radar, dual-slot, statics, fossils and gifts. nuzline plans from
what the save holds, so availability is planning data only. Sourcing: ROM
`fielddata/encountdata/pl_enc_data.narc` plus the spreadsheet's `Non-Wild Encounters` and
`Universal Encounters` for gifts, trades and statics. Not needed for the first milestones.

### 5.8 Start conditions and battle formats

Start conditions come from the spreadsheet's split headers and PKCalc `flags.js`, then the ROM.
PKCalc's weather list (235 names) is "what the calc starts with": of 234 resolvable entries, 113
match a sheet area tag, 34 are fog, 78 are weather set by a lead's or partner's ability, and 9 come
from double-battle second leads or duplicate listings. The design separates
**area condition** (what the fight starts with) from **ability weather** (computed by the engine
from the leads).

| Condition | Sources | Semantics (user-decided, §3) |
|---|---|---|
| Permanent rain/sun/sand | Sheet split headers (PERMANENT …), PKCalc `flags.js` weather | Never expires; moves do not replace it; weather abilities do. |
| Hail, and rain/hail without PERMANENT | Sheet headers "(HAIL)"/"(RAIN)" only — no sheet section says PERMANENT HAIL; PKCalc weather | Assumed permanent (user, §12.1). |
| Fog | Sheet (FOG), PKCalc `weather.fog` | Vanilla fog: accuracy × 0.6, No Guard and never-miss moves hit. |
| Permanent Trick Room | Sheet only: Veilstone Warehouse (26 trainer headers) and Galactic HQ (25, incl. Cyrus #2 and Saturn #2), 37 encounters, plus the Sinister Hooded Figure; Galactic Rosalind is listed both in the Warehouse and under Coronet Highlands hail, to settle | Never expires. |
| Permanent Gravity | Champion Cynthia (PKCalc `shared_controls.js:611`, sheet) | Never expires. |

Opponent interactions found in data: weather abilities inside permanent areas (e.g. Aroma Lady
Jenna's Numel Drought in Eterna Gym rain — the ability wins); opponents using Gravity inside
permanent Trick Room fights (Galactic Bianca #2, Galactic Kari, Sinister Hooded Figure) — the
Kaizo move text says Gravity's effect is permanent, confirmed by trace; Cheryl (tag partner) with
Sunny Day in a hail area. Weather labelled only "(HAIL)"/"(RAIN)" without PERMANENT (Snowpoint
Gym, Routes 215–217, Coronet Highlands Outside 1 and 2, Coronet Highlands Room 3) is assumed
permanent (user, §12.1).

Vanilla start conditions cannot be reused: 202 of 454 id-joined trainers have a different PKCalc
start weather from vanilla's `start_condition` (e.g. ids 397, 399, 400: Route 210 fog in vanilla,
Lake Valor rain grunts in Kaizo). That count includes ability weather, so it overstates changed
area conditions; the ROM pipeline recomputes area conditions directly.

Formats (PKCalc, 389 opponent encounters): 282 singles (counting rival-starter variants; tag
partners excluded), 36 one-trainer doubles, 32 two-trainer doubles, 39 multi battles with an NPC
partner; 32 trainers in GAUNTLET chains (sheet). Doubles include Leader Volkner, Mars #1, Jupiter
#2, Saturn #2, Cyrus #3 and the Sinister Hooded Figure. Battle-type disagreements: `RAW TRAINER
DATA` says single where PKCalc says double for Volkner, Cyrus #3 and Mercury #3 (split headers
agree with PKCalc) and for Saturn #2 and Dora & Diego (PKCalc only); RAW says double where PKCalc
says single (Worker Conrad, id 846) or multi (Castle Valet Darach / Lady Caitlin, 749/750). The
ROM decides.

Every trainer record carries: `format` (single, double, double_two_trainer, multi),
`opponent_2`, `partner` (with starter variant), `gauntlet_group` and position, `optional`,
`left_side` AI, `area_condition`, and the source of each.

**Vanilla bug found in passing:** `data/intermediate/trainers.json` drops the decomp's
`double_battle` flag (28 vanilla trainers have it), so a vanilla double battle would be simulated
as a single without refusal. Fixing it is a separate, small change (§13).

### 5.9 Mechanics

Kaizo-specific code in PKCalc's calc (`calc/src/mechanics/gen4.ts`, diffed against upstream
@smogon/calc): the triple-hit ramps above; Trump Card fixed 200 BP; Wring Out works like Brine
(×2 at ≤50% HP), Crush Grip flat 150; Eruption loses HP scaling (Water Spout keeps it); Revenge
doubles when moving second, like Payback; Pursuit/Rage ×2 on a switching target; Plates ×1.5;
Natural Gift and RageCandyBar placeholder. Several are battle-script or ASM changes the ROM diff
must find and the engine must implement per game. The ARM9/overlay diff (§6.2), mapped to decomp
functions, is the complete list; PKCalc is a pointer, not the list.

### 5.10 Trainer AI

PKCalc renders AI flags (`trainer_flags.js`, 11 flags) and has no switching AI (the `.switch-ai`
container is a stub), so it cannot cross-check nuzline's AI. Kaizo flags differ from vanilla for
473 of 928 trainers. Bosses: Roark 23 (Basic, EvalAttack, Expert, Risky), Gardenia 15, Fantina
271, Wake 15, Candice 15, Volkner 1039, Aaron 279, Bertha 15, Lucian 351, Cynthia 7. Across PKCalc:
Setup 175, TagStrategy 139, Harassment 107, Risky 105, CheckHP 95, PrioritizeExtremes 68, BatonPass
30, Weather 19. nuzline implements Basic+EvalAttack+Expert only (`engine/coverage.py:92-94`), and transcribes
AI move paths only for vanilla Roark's moves.

`AI Changes` (spreadsheet), each mapped to the vanilla script nuzline transcribes:
1. Dry Skin fix: `trainer_ai/script.s:78` tests Levitate where Dry Skin was meant; nuzline keeps
   that quirk on purpose (`ai/policy.py:150`). Kaizo policy must not.
2. 1/2-recoil moves except Head Smash follow the intended recoil routine (in vanilla their effects
   are in the no-damage-calc list, so the AI never rates their damage).
3. 1/4-recoil moves (Brave Bird, Close Combat, Take Down) get a new three-line routine; PKCalc
   calls it "vanilla Superpower AI". Whether Superpower's Attack-stage check is included is open.
4. Triple Axel and Fury Cutter count as Risky; Triple Kick and Rock Wrecker do not, despite one
   shared effect label.
5. Hurricane does not use the Thunder routine.
6. "If exported directly from the game, certain moves may read as having effects they do not.
   This is intentional." The AI and battle code read a move's effect id; Kaizo may set an effect
   id for AI purposes that differs from what the move does.

Also hard-coded by id in vanilla and now pointing at repurposed slots: `TrainerAI_CalcDamage`
cases for `MOVE_PSYWAVE` (now Triple Axel), `MOVE_MAGNITUDE` (Bulldoze), `MOVE_RETURN` and
`MOVE_FRUSTRATION` (now fixed 121 BP); `script.s:6737` Magnitude → TagStrategy; `battle_lib.c`
punch list (slot 4, now Water Ball), Gravity list (slot 150, now HP Water), Heal Block list (slot
361, now HP Electric). nuzline's own copies — `ai/commands.py:131` (`MOVE_IDS`, the Soundproof list
`SOUND_MOVES`), `ai/switching.py:106`, `ai/policy.py:325` — are vanilla facts.

Design consequences: a Kaizo move has two effect columns, `ai_effect` (the ROM id the AI scripts
and id-keyed checks read) and `mechanic` (what the move does), equal unless the ROM diff or a
trace shows otherwise; every id-keyed list becomes per-game data extracted from that game's ROM
code, never a constant; Kaizo AI scripts are taken from the patched `battle/tr_ai/tr_ai_seq.narc`
and the patched `trainer_ai.c` code, diffed against vanilla, and transcribed as a Kaizo policy
variant.

### 5.11 Saves

Kaizo keeps the vanilla save layout as far as the sources show; species, move and item ids are
the vanilla spaces. The current parser already reads the Maylene save without error using the
vanilla tables (checked 2026-11-03); what remains is reading it with Kaizo names and tables.
Forms: the parser refuses non-zero forms of the six species whose forms change stats, types or
abilities (Deoxys, Wormadam, Rotom, Giratina, Shaymin, Arceus); Kaizo has playable forms (Rotom,
…), which needs form-indexed tables (§7.1). Verification: parse `Platinum Kaizo Nightmare(Maylene).sav`, check
checksums and that every species/move/item is valid in the Kaizo tables, and diff against PKHeX
as for vanilla (`docs/saves.md`).

### 5.12 Level caps and exp

Kaizo level caps (user's table): Roark 16, Gardenia 28, Fantina 38, Barry #3 42, Maylene 47, Wake
54, Barry #5 63, Byron 65, Candice 74, Volkner 84, Coronet/Elite Four 100. Barry caps apply only to
that Barry battle; every other fight uses its split's gym-leader cap. The split comes from the
spreadsheet's `* Split` sheets. Gym-leader caps equal each leader's highest level in PKCalc; the
Barry caps are above his highest level, so the table is stored as given, not derived.

Caps are soft: the game never stops exp, so the cap is the player's rule. nuzline:
- excludes a Pokémon above the fight's cap;
- offers level-to-cap only up to the cap;
- warns when a line's exp from the fight would put a participant over the **next** fight's cap.
  This needs Kaizo exp yields (15 changed), growth rates (11 changed) and the Gen 4 exp formula,
  computed per participant (and Exp. Share if held).

## 6. Pipeline

### 6.1 Inputs (local only)

A local, git-ignored config (`data/local.toml` or an environment variable; builder's call) names
the vanilla ROM, the patch and the PKCalc and spreadsheet copies. The build:
1. checks the vanilla ROM SHA-1 is `0862ec35…755c` and the patch hash is the recorded one
   (SHA-256 `b787e743b5fa78e1b31bae9277e656482b7248b82e490a6e95236606ae92f433`); any mismatch
   stops the build;
2. applies the VCDIFF patch in memory or to a git-ignored path, and records the patched ROM hash;
3. never writes ROM bytes, NARC files or code into the repo.

The patch's VCDIFF header names the source and target files it was built from
(`PlatKaizoFinal2.nds` from `Vanilla Platinum - Copy (4).nds`); the hashes are the check, not the
names.

### 6.2 ROM diff (the change inventory)

Unpack both ROMs' file systems (NDS FAT/FNT, NARCs; ARM9 and overlays, decompressing as needed)
and diff file by file. Output `docs/kaizo/rom_diff.md`: every changed file, and for code, every
changed address range mapped to the decomp symbol containing it. The decomp builds the vanilla ROM
byte-exact, so a build's map file gives the symbols; K1 builds the decomp once for this (the user
agreed to installing its toolchain, §12.6). This is the authoritative list of what
Kaizo changed; each code change gets an entry: "understood and modelled", "irrelevant to battle"
or "unknown — refuse fights that can reach it".

### 6.3 Extraction

`tools/extract_rom.py` reads the patched ROM's NARCs with the layouts the decomp defines
(`pl_personal`, `pl_waza_tbl`, `wotbl`, `evo`, `trdata`/`trpoke` with Kaizo's struct, `pl_item_data`,
`pl_enc_data`, `tr_ai_seq`, `be_seq`/`sub_seq`), plus ARM9 data that is not in a NARC (map headers
with their weather, `src/map_header.c`; the TM and tutor tables) and the field scripts that tie
trainers to maps (as `extract_decomp.py`'s `_trainer_maps` does). It writes
`data/intermediate_kaizo/*.json` in the same schemas as `data/intermediate/`, extended as §7
describes. Its own test: run on the vanilla ROM, it must reproduce what the decomp extractor
produces at the same schema version, with `tools/start_conditions.json` applied the same way.

`tools/extract_pkcalc.py` evaluates PKCalc's data files in a sandbox (data only, no PKCalc code
run) and `tools/extract_kaizo_docs.py` reads the spreadsheet, each into comparison JSON.

### 6.4 Overlays the ROM cannot supply

Some facts are not in ROM data: area conditions that come from scripts rather than map headers
(permanent Trick Room, Cynthia's Gravity), gauntlet groups, optional trainers, the user's level
caps, the fight-to-split mapping, and PKCalc names. These go in a committed, hand-reviewed
`data/kaizo/overrides.toml`, each entry citing its source (sheet and line, PKCalc file and line,
or "user, 2026-11-03"). Once a trace or the ROM diff confirms an entry, the citation is upgraded.

### 6.5 Cross-check and resolution

`tools/crosscheck_kaizo.py` diffs ROM, PKCalc and spreadsheet field by field and writes
`docs/kaizo/discrepancies.md`. Policy:
- **Every disagreement is listed** in the report, with all three values and the value used.
- **ROM vs PKCalc on an executed field** (power, accuracy, PP, priority, effect, type, category,
  target, chance, stats, abilities, learnsets, trainer sets, AI flags, format, hold effects) fails
  the build until `data/kaizo/resolutions.toml` (committed) has an entry: value, reason, evidence.
  The builder writes these entries. Grouped entries are allowed when one cause explains many rows
  (for example, a PKCalc naming convention, or a class of decoy effect ids identified in the ROM
  diff).
- **Spreadsheet-only disagreements** (spreadsheet differs, ROM and PKCalc agree) are listed and
  need no entry.
- The default is the ROM value; a resolution against the ROM needs a trace.
- The user reviews the report and the resolutions once, at the end of K2, as one review — not
  entry by entry during the build.

### 6.6 Tables

`tools/build_tables.py --game kaizo` builds `data/gen4_platinum_kaizo/tables.npz` and
`manifest.json` from `data/intermediate_kaizo/` and the overrides, with the same encodings as
vanilla plus §7's additions. Adding the pipeline changes nothing in vanilla; the §7.1 schema
extensions change vanilla deliberately, once, as their own versioned step (§11, K2a), which bumps
the vanilla table version and so invalidates vanilla caches and any stored gate results built on
the old version.

## 7. Data model changes

### 7.1 New or changed fields (both games)

- **Forms:** species tables indexed by (species, form) through a personal index, as the ROM does
  (forms at 496–507). Vanilla gains them too (the save parser refuses forms today).
- **Learnsets:** level-up, TM (with the per-game TM → move table), tutor (with location and shard
  costs), egg moves; evolutions. Needed by the moveset toggle in either game.
- **Moves:** `ai_effect` alongside `mechanic` effect; chance-0 secondaries allowed.
- **Items:** hold effect and parameter per game; names per game.
- **Trainers:** `format`, `opponent_2`, `partner` (+ starter variant), `gauntlet_group`,
  `optional`, `left_side`, `area_condition` (weather, fog, trick_room, gravity, each with
  `permanent`), per-mon `nature`/`gender`/`ability_slot` overrides, and the source of each field.
  Vanilla `trainers.json` gains `format` from the decomp's `double_battle`.
- **Species:** exp yield (and growth rate, already present) for the exp-cap warning.
- **Level caps:** per-game cap table and fight → split map.

### 7.2 Game constants become data

Every vanilla id list in code (AI move/ability/item/type id lists, Soundproof and punch lists,
`MOVE_STRUGGLE`, item ids in `ai/commands.py`, `ai/switching.py`, `ai/policy.py`,
`engine/ids.py`, `effect_map.py`) is classified as either a true engine constant (same in every
game, asserted per game at load) or game data (moved into the manifest, extracted per ROM). A test
per game fails if a constant's meaning differs from that game's ROM.

### 7.3 Name lookups

`Names` resolves per game. A Kaizo save, trainer or CLI argument is resolved with Kaizo names
("Accelerock", "Water Ball"); the vanilla name for a repurposed slot ("Rollout" in Kaizo) is an
error pointing at the Kaizo name, not a silent match.

## 8. The data-set switch

- **One directory per game:** `data/gen4_platinum/` (vanilla, unchanged) and
  `data/gen4_platinum_kaizo/`, each with `manifest.json` and `tables.npz`; intermediates in
  `data/intermediate/` and `data/intermediate_kaizo/`.
- **Manifest provenance:** a `game` id (`platinum_us`, `platinum_kaizo_nightmare_final`); vanilla
  keeps `decomp_commit`; Kaizo records the vanilla ROM SHA-1, patch SHA-256, patched ROM SHA-1,
  PKCalc archive hash and spreadsheet hash, decomp commit (for layouts) and the overrides and
  resolutions file hashes.
- **Selection:** chosen once per run with `--game vanilla|kaizo` (default vanilla, so nothing
  changes for existing commands) or a `game` field in the run config. `tables.DEFAULT` becomes a
  function of the game. Nothing infers the game from a save (a Kaizo save looks like a vanilla
  save); a save whose contents are impossible in the chosen game (unknown move for a species'
  learnset, unknown item name) warns loudly.
- **Run records** store the game id next to the table version.
- **Cache keys:** `jaxcache.table_version()` already hashes `manifest.json` + `tables.npz`. The
  cache directory becomes `<game>/<table_version><code_version>`, and `prune_other_versions`
  prunes only within the same game, so switching games no longer deletes the other game's
  compiled programs.
- **Generated Kaizo tables stay local:** they are derived from the user's ROM, and the repo has a
  GitHub remote (`origin`, pushed before). So `data/gen4_platinum_kaizo/`, `data/intermediate_kaizo/`
  and the patched ROM path are git-ignored from the start, and the build regenerates them.
  Overrides, resolutions, the discrepancy report and the ROM diff report contain no ROM bytes and
  are committed.

## 9. Verification plan

1. **Reader test:** the ROM reader on the vanilla ROM reproduces `data/intermediate/` exactly.
2. **Three-way diff:** ROM vs PKCalc vs spreadsheet, every disagreement resolved (§6.5).
3. **ROM diff inventory:** every changed code range classified (§6.2).
4. **PID check:** every Kaizo trainer's nature, gender and ability reproduced by the transcribed
   Kaizo routine; until then, by explicit data with the vanilla routine as a cross-check.
5. **Golden traces on the Kaizo ROM** (`tools/trace_battle.lua`; first confirm the Kaizo
   BattleContext offsets match vanilla). One recording each for:
   - permanent sand (Oreburgh Gym), permanent rain (Eterna Gym, including Jenna's Drought),
     fog (a Byron-split fog trainer with an opposing No Guard user, e.g. Galactic Thebe or
     Callisto #2), permanent Trick Room (Galactic HQ),
     permanent Gravity (Cynthia) — checking the field-condition bits never decrement and a Gravity
     move becomes permanent;
   - per-hit power and accuracy re-rolls for Triple Kick, Triple Axel, Fury Cutter, Rock Wrecker;
   - recoil fractions and recharge for the 1/2, 1/3 and 1/4 recoil moves and Roar of Time;
   - Hurricane in rain, Plates, param-0 pinch berries, RageCandyBar;
   - one override trainer Pokémon (nature, gender, ability) — `trace_battle.lua` does not emit
     personality, nature, gender or ability today, so it is extended to, or this is checked
     in-game;
   - AI choices of a Risky trainer (Kaizo Roark) against the transcribed Kaizo policy.
6. **Save test:** the Maylene save parses (§5.11).
7. **Fidelity classes:** each Kaizo fact carries its confidence class (ROM-verified,
   golden-traced, PKCalc-derived, spreadsheet-only), and the UI shows the weakest in play, as
   design.md §3 requires.

## 10. Kaizo Roark: coverage target, then the second gate

Kaizo Roark (trainer id 246): permanent sandstorm, AI 23 (adds Risky), no bag items, six Pokémon at
IV 30:

| Pokémon | Item | Nature / ability | Moves |
|---|---|---|---|
| Bonsly L15 | Focus Sash | Impish / Rock Head | Stealth Rock (+1), Self-Destruct, Brick Break, Accelerock (+1) |
| Lileep L13 | Big Root | Naive / Solid Rock | Sludge Bomb, Giga Drain, AncientPower, Recover |
| Gible L14 | Chesto Berry | Gentle / Sand Veil | DragonBreath, Flamethrower, Thunder Fang, Earth Power |
| Corsola L14 | Rindo Berry | Quirky / Solid Rock | Recover, Water Pulse (never misses), Light Screen, Accelerock |
| Shuckle L13 | Berry Juice | Naive / Solid Rock | Toxic, Accelerock, Curse, Sand Tomb |
| Cranidos L16 | Lum Berry | Adamant / Rock Head | Head Smash, Zen Headbutt, Crunch, Rock Polish |

Today nuzline refuses him on four counts (sandstorm, the Risky flag, moves, items/abilities), as
it should. Supported already: 11 of 21 moves, Rock Head. Sand Tomb's Kaizo effect decodes to
Whirlpool (261), unmapped today; it is likely a cheap map onto the existing trap effect plus
Whirlpool's dive interaction, to be confirmed against the decomp. Missing: Self-Destruct, Brick Break, AncientPower, Recover, Thunder Fang, Water Pulse,
Light Screen, Toxic, Curse; Solid Rock, Sand Veil; all six items (Berry Juice is close to Oran);
permanent sandstorm with end-of-turn chip and the field-condition pass; the Risky AI flag and AI
script paths for ~19 moves.

**Step 1 — coverage milestone.** Make the fight accepted: the items above, plus a Kaizo golden
trace of Roark. Check whether the AI outcome width still fits (`ai/outcome.py`, 3840 ≤ 4096) with
Risky rolls and six-Pokémon switching; if not, that is a finding, not a reason to approximate.

**Step 2 — exhaustive gate.** On the user's pre-Roark Kaizo save (to be supplied), run the
milestone 4 exhaustive check (`benchmarks/exhaustive_roark.py` on `claude/m4-prep-search`) against
Kaizo Roark, beside the vanilla gate. The player side is unchanged; each evaluation is longer and
branchier (hazards, Sash, Recover/Curse stall), so the overnight budget is re-measured and a
smaller CI pool kept. The gate passes when prep search's best line matches the exhaustive best
within the milestone 4 tolerance.

## 11. Milestones (proposed)

The implementation plan written from this design covers **K1 and K2 only**. K3 is engine work and
gets its own design (as the non-goals say); K4 follows it.

- **K1 ROM pipeline:** local inputs and hash checks, patch application, file-system unpacking, ROM
  diff inventory, ROM reader with the vanilla round-trip test. New files only; can start from
  `main` now.
- **K2a vanilla schema extensions:** forms, full learnsets, egg moves, evolutions, `ai_effect`,
  trainer `format` (fixes the dropped `double_battle`), exp yields — in the decomp extractor and
  the ROM reader together, as one deliberate vanilla table-version bump. Lands after milestone 4
  (`claude/m4-prep-search`) is merged, because both touch `tools/extract_decomp.py`,
  `tools/effect_map.py` and the AI constants.
- **K2b Kaizo tables:** Kaizo intermediates, PKCalc and spreadsheet extractors, cross-check report
  and resolutions, overrides file, per-game constants (§7.2), the data-set switch and per-game
  caches, level caps, Kaizo save parsing on the Maylene save. Exit: the user's one review of the
  discrepancy report (§6.5).
- **K3 Kaizo Roark coverage (own design):** §10 step 1. Its golden trace needs the user's
  pre-Roark save.
- **K4 Kaizo Roark gate:** §10 step 2, on the same save.

Golden traces are a **per-fight gate**, not a K1/K2 exit criterion: each fight's trace is
recorded when play reaches it (or earlier, if a save for it exists), and until then that fight
carries the lower confidence class (§9.7). K1 and K2 need no traces.
- **Later, as play reaches them:** further Kaizo fights gym by gym (milestone 5's approach), the
  exp-cap warning in prep search, then doubles (Volkner), multi/tag battles, gauntlets — each its
  own design.

## 12. Open questions

Answered by the user (2026-11-03):
1. Weather marked only "(RAIN)"/"(HAIL)" without PERMANENT (Snowpoint Gym, Routes 215–217,
   Coronet Highlands): **assume permanent**, like the PERMANENT sections. An emulator test can
   later confirm it.
2. Rival and optional trainers: **show only the rival variant matching the save's starter; mark
   optional trainers as skippable.**
3. Items, TMs and shards: **use only what the save's bag holds** (as today); no Kaizo item
   locations are needed.
4. Gauntlets **vary**: some only forbid changing Pokémon but allow healing; others are strictly
   back to back with no healing or changes. Each gauntlet's rules are recorded in
   `data/kaizo/overrides.toml` when play reaches it (`gauntlet_group` gains a `rules` field:
   `heal_between`, `change_team`).
5. Golden traces: the user **checks moves and mechanics as play reaches them** ("when I get that
   move I can check"). Traces stay a per-fight gate (§11). Still open: whether emulator save
   states or other players' saves are acceptable starting points.
6. Decomp toolchain: **installing it is fine.** K1 builds `external/pokeplatinum` once to get the
   vanilla map file for the ROM diff (§6.2).

Settled by the ROM or traces (no user input needed):
7. Triple Kick family per-hit power and per-hit accuracy.
8. Natures of the 111 override Pokémon; the Kaizo trainer struct.
9. The "fixed AI" recoil effect id; the 1/4-recoil AI routine's exact checks; how Triple Axel and
   Fury Cutter become Risky.
10. Roar of Time, Hurricane chance, Kinesis, Swords Dance PP, Constrict, Big Nugget Fling, and the
    other PKCalc/spreadsheet conflicts.
11. Battle type for Volkner, Cyrus #3, Mercury #3, Dora & Diego, Saturn #2, Worker Conrad, Darach /
    Caitlin; Galactic Rosalind's area.
12. Whether Hurricane also gets Thunder's 50% accuracy in sun; whether Gravity used as a move is
    permanent; whether fog interacts with Lock-On, Gravity, Cloud Nine/Air Lock as in vanilla.
13. Whether Kaizo patched `TrainerAI_CalcDamage`'s Psywave/Magnitude/Return/Frustration cases and
    the `battle_lib.c` slot-id lists.
14. Whether Kaizo changed switching or replacement AI (`TrainerAI_ShouldSwitch`).
15. Whether the Kaizo BattleContext RAM layout matches vanilla (needed for traces).
16. Which AI-flag source is right for the 44 PKCalc/spreadsheet disagreements (default PKCalc,
    then ROM).
17. Egg moves and tutor locations (only matter for the moveset toggle).

## 13. Calls made in this design (for the user to check)

- ROM wins over PKCalc on executed data; PKCalc stays the reference for names and descriptions.
- Trick Room, Gravity, gauntlets and level caps live in a committed overrides file with citations
  until the ROM or a trace confirms them.
- Nature/gender/ability for Kaizo trainers are explicit data until the patched routine is
  transcribed.
- Two effect columns (`ai_effect`, `mechanic`) per move, equal by default.
- `--game` defaults to vanilla; the game is never inferred from a save.
- Forms, full learnsets and the trainer `format` field are added for vanilla too.
- Kaizo Roark is a coverage target before it is a gate.
- The plan covers K1–K2; K1 starts from `main`, K2a waits for milestone 4 to merge.
- Discrepancy resolutions are written by the builder and reviewed by the user once, at K2's end.
- Golden traces gate individual fights, not the data milestones.
- Generated Kaizo data is git-ignored from the start, because the repo has a GitHub remote.
- Follow-up outside this design: vanilla `trainers.json` drops `double_battle`, so vanilla double
  battles are not refused today.

## Appendix: sources consulted

- `PKCalc-main.zip` → `src/js/data/**`, `calc/src/**`, `src/js/shared_controls.js`,
  `calc/package.json` (@smogon/calc 0.10.0 base).
- `Platinum Kaizo Docs.xlsx`, 39 sheets (listed in §4).
- `Pokemon Platinum Kaizo Nightmare Update Final.xdelta` (VCDIFF header: target
  `PlatKaizoFinal2.nds`, source `Vanilla Platinum - Copy (4).nds`).
- `4998 - Pokemon - Platinum Version (v01) (U).nds` (SHA-1 `0862ec35…755c`, header `CPUE`, rev 1).
- `external/pokeplatinum`: `INSTALL.md:334`, `include/struct_defs/trainer_data.h:35`,
  `res/battle/scripts/effects/effect_script_0104.s:7`, `src/battle/battle_controller_player.c`,
  `src/battle/battle_lib.c`, `src/battle/trainer_ai/trainer_ai.c`,
  `src/battle/trainer_ai/script.s`; ROM NARC paths as listed in §5 and §6.3.
- nuzline: `tools/extract_decomp.py`, `tools/build_tables.py`, `tools/effect_map.py`,
  `src/nuzline/data/{tables,trainer,spec,party}.py`, `src/nuzline/jaxcache.py`,
  `src/nuzline/ai/{commands,switching,policy,outcome}.py`, `src/nuzline/engine/coverage.py`,
  `docs/design.md`, `docs/tracing.md`.
