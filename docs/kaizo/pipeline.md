# Kaizo data pipeline

How the Platinum Kaizo data set (`--game kaizo`) is built from your local files, what ends
up in git, and how to correct it. Design: `docs/superpowers/specs/2026-11-03-kaizo-data-design.md`;
plan: `docs/superpowers/plans/2026-11-03-k2b-kaizo-tables.md`. The ROM reading itself is
described in `docs/rom_pipeline.md`.

## Inputs

All inputs are local files and are never committed. Each path comes from an environment
variable, else from `data/local.toml` (git-ignored; copy `data/local.example.toml`). Use
single-quoted strings for Windows paths; relative paths resolve against the repository root.

| `data/local.toml` key | Environment variable | What it is |
| --- | --- | --- |
| `vanilla_rom` | `NUZLINE_VANILLA_ROM` | Platinum US ROM; SHA-1 must be `0862ec35b24de5c7e2dcb88c9eea0873110d755c` |
| `kaizo_patch` | `NUZLINE_KAIZO_PATCH` | the Kaizo Nightmare Update Final xdelta; SHA-256 must be `b787e743...f433` |
| `pkcalc_zip` | `NUZLINE_PKCALC_ZIP` | the PKCalc repository zip (audited copy: SHA-256 `849733e6...2eb0f`) |
| `kaizo_docs` | `NUZLINE_KAIZO_DOCS` | the Kaizo docs spreadsheet (audited copy: SHA-256 `41c0c7fe...d74f`) |
| `kaizo_save` | `NUZLINE_KAIZO_SAVE` | optional: a Kaizo save (the tests use a Maylene-era save) |
| `kaizo_save_pre_roark` | `NUZLINE_KAIZO_SAVE_PRE_ROARK` | optional: an earlier Kaizo save (pre-Roark), `.sav` or DeSmuME `.dsv` |
| `kaizo_save_probe` | `NUZLINE_KAIZO_SAVE_PROBE` | optional: the trace probe's save (`docs/tracing_kaizo.md`), the party taken to Roark |

A ROM or patch hash mismatch stops the build. The PKCalc and spreadsheet hashes are recorded
in the report, not enforced; tests that pin audited counts skip on another copy. Tests that
need a missing input skip.

Also needed:

- the vanilla intermediates (`uv run python -m tools.extract_decomp --decomp
  external/pokeplatinum --out data/intermediate`);
- the decomp at `external/pokeplatinum` with its linker map (`docs/rom_pipeline.md`);
- `node` on `PATH` (PKCalc's data files are evaluated in a locked-down `vm`; no PKCalc code
  runs);
- the `rom` dependency group (`uv run --group rom ...`).

## Commands

Run in order, from the repository root:

```
uv run --group rom python -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom
uv run --group rom python -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
uv run --group rom python -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
uv run --group rom python -m tools.kaizo.overrides check
uv run --group rom python -m tools.build_tables --game kaizo
```

1. `extract_rom` patches the ROM (the patched copy is cached under `.cache/rom/`) and reads
   species, moves, items, learnsets, trainers, map weather and battle formats from it. It
   prints the trainers whose start condition or format is left to you (see "User calls").
   It also records which code the patch changed (`changed_code.json`) and the hashes of the
   committed files it read: `data/kaizo/start_conditions.json`,
   `data/kaizo/battle_formats.json` and the `[move_lists]` of `data/kaizo/overrides.toml`.
   After editing one of those, re-run `extract_rom`: the build refuses ROM intermediates
   older than them ("re-run tools.extract_rom --game kaizo").
2. `extract_pkcalc` and `extract_kaizo_docs` turn PKCalc and the spreadsheet into JSON for the
   cross-check.
3. `overrides check` re-validates every citation in `data/kaizo/overrides.toml` against the
   spreadsheet and the ROM extraction.
4. `build_tables --game kaizo` cross-checks ROM, PKCalc and spreadsheet field by field, writes
   `docs/kaizo/discrepancies.md`, and stops (exit 1) on an unresolved or stale row. Clean, it
   writes the resolved `data/intermediate_kaizo/*.json` and
   `data/gen4_platinum_kaizo/{tables.npz,manifest.json}`.

Then use it: every `nuzline` command takes `--game kaizo`, for example

```
uv run nuzline save show --game kaizo path/to/kaizo.sav
uv run nuzline trainers --game kaizo --save path/to/kaizo.sav
uv run nuzline coverage --game kaizo --save path/to/kaizo.sav --trainer roark
```

The game is never guessed from a save. Today `coverage` refuses every Kaizo fight (exit 2);
see "What is refused".

`python -m tools.rom.diff` regenerates `docs/kaizo/rom_diff.md` (the list of every code and
file the patch changed); it is only needed when the patch or the classifications change.

## What is committed

Committed (ids, names, citations and decisions, never ROM bytes):

- `data/kaizo/overrides.toml`: facts the ROM cannot settle alone (Trick Room, Gravity,
  gauntlet groups, trainer-id settlements, field corrections).
- `data/kaizo/resolutions.toml`: how each ROM-vs-PKCalc disagreement on a real game field is
  settled.
- `data/kaizo/start_conditions.json`, `data/kaizo/battle_formats.json`: trainers whose start
  weather or battle format the ROM leaves ambiguous.
- `data/kaizo/level_caps.toml`: your level caps (read by `prep` at run time).
- `data/kaizo/rom_diff.toml`, `data/kaizo/rom.toml`: the classification of each patch change,
  and the patched ROM's hashes.
- `docs/kaizo/discrepancies.md` (generated by the build; byte-stable when nothing changed),
  `docs/kaizo/rom_diff.md`, this guide.

Never committed (git-ignored): `data/local.toml`, ROMs, the patch, saves, PKCalc and
spreadsheet copies, `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` and `.cache/`.

## Correcting the data

**A disagreement fails the build.** The error and `docs/kaizo/discrepancies.md` ("unresolved")
name the table, key, field and the three values. Add an entry to `data/kaizo/resolutions.toml`:

```
[[resolution]]
table = "moves"
keys = ["14 Swords Dance"]   # exact keys from the report; no wildcards
field = "pp"
pkcalc = 2                   # the PKCalc value being overruled
use = "rom"                  # or "pkcalc", which needs evidence starting "trace:"
reason = "why the sources differ"
evidence = "where the ROM value comes from"
```

then rebuild. One entry may list many keys when one cause explains them all. An entry whose
disagreement goes away, or whose PKCalc value changes, is stale and fails the build: delete or
update it. Disagreements where only the spreadsheet differs are listed but need no entry.

**A fact the ROM does not hold** goes in `data/kaizo/overrides.toml`, keyed by ROM trainer id,
each with a `source` (`"<Sheet>:<row>"`, `"pkcalc:<file>:<line>"`, `"rom:<what>"` or
`"user, YYYY-MM-DD"`):

- `[area.<id>]`: `trick_room`, `gravity`, `weather` (`rain|sun|sandstorm|hail|fog|none`;
  replaces the ROM start condition). This is also how you settle an "unknown" start
  condition (see "Settling a user call").
- `[area_exception.<id>]`: a ROM Trick Room trainer deliberately left without Trick Room, with
  the `reason`.
- `[[gauntlet]]`: `name`, `trainers` (ids in fight order), `rules` (`heal_between`,
  `change_team`; fill in when play reaches the gauntlet).
- `[trainer_ids."<label or Sheet:row>"]`: `id`, when a PKCalc label or sheet header matches
  the wrong ROM trainer.
- `[fields.<id>]`: `split` (the trainer's sheet position is then unknown), `optional`,
  `left_side`, `player_starter`, or `format`: `"unknown"` to refuse a trainer, or your answer
  (see "Settling a user call").

`python -m tools.kaizo.overrides draft` writes a fresh draft to
`.cache/kaizo/overrides.draft.toml` to compare against; the committed file is the reviewed one.
Run `overrides check`, then rebuild.

**Settling a user call.** Each user call (below) is settled in `data/kaizo/overrides.toml`
with `source = "user, YYYY-MM-DD"`, then `overrides check` and a rebuild:

- *Unknown start condition* (a `start_conditions.json` "unknown", or no map placement
  found): add `[area.<id>]` with the `weather` the fight starts in (and `trick_room = true`
  if it has Trick Room). Where the trainer has an `[area_exception.<id>]`, replace it.
- *Trick Room on one placement only*: replace the trainer's `[area_exception.<id>]` with
  `[area.<id>]`, `source = "user, YYYY-MM-DD"`. Say which fight is meant: `trick_room = true`
  for the Trick Room map, or the other map's `weather` (`"none"` for none). You may name the
  placement with `map_header = N` (one of the trainer's headers in the report); the check
  then requires `trick_room = true` exactly when that header has Trick Room. A `rom:` or
  sheet citation cannot settle these trainers: the ROM holds both placements.
- *Unknown format*: add `[fields.<id>] format = "single"`, `"double"`,
  `"double_two_trainer"` (with `opponent_2 = <the other trainer's id>`) or `"multi"`
  (`opponent_2` and `partner = <the player's partner's trainer id>` optional), replacing a
  `format = "unknown"`. Only a `user,` source may set a format. Record a pair on both sides.
  For 521 and 527 keep their entries in `data/kaizo/battle_formats.json`: the ROM's script
  claims still conflict, and `extract_rom` stops on a conflict that file does not list; the
  `[fields]` answer replaces the refused format. The cross-check then compares the answer
  with PKCalc and the sheet; a disagreement needs a resolution as usual.
- *Weather where the ROM disagrees with PKCalc and the sheet*: add `[area.<id>] weather`
  with a `user,` source. The ROM-vs-PKCalc row compares the ROM map byte and keeps its
  `use = "rom"` resolution (say in its `reason` that the user's answer sets the start
  condition); the report then drops the call.

**Level caps**: edit `data/kaizo/level_caps.toml` (`[splits]` per split sheet, `[fights]` per
fight label prefix). No rebuild needed.

## Rules the build follows (plan Rulings 8-16, 21-22)

- **Start weather (8).** Weather and fog come from the map the trainer stands on, and map
  weather lasts the whole battle. In Kaizo, map weather 18 means permanent Trick Room (read
  from the patched battle code), so every trainer on such a map gets Trick Room. Trick Room
  that no map holds, and Cynthia's Gravity, come from `overrides.toml`. Maps whose weather
  depends on the date are settled in `start_conditions.json`.
- **Split, optional, left side (9).** Read from the spreadsheet on every build, with a
  citation; `overrides.toml` can correct them. When a trainer's sheet headers disagree on
  `optional` or `left_side`, the field is unknown and the report says so. Gauntlet groups
  are in `overrides.toml`, drafted from the spreadsheet's gauntlet and "B2B With" tags; their
  healing and team-change rules stay empty until you reach them.
- **Battle formats (10).** Single, double, two-trainer double or multi come from the ROM's
  field scripts. When PKCalc or the sheet pairs two trainers that the scripts do not, and both
  stand on the same map, they are a sight pair (both see you in one step): a two-trainer
  double. Every non-single format is refused by coverage today.
- **Rivals (11).** Each rival or partner fight is tied to the player's starter. With `--save`
  only the variant for your starter is used; without one, a rival fight needs `--save` or
  `--trainer-id`. A name that matches several rival fights (`barry`) must name the fight
  (`"barry #3"`) or use `--trainer-id`.
- **Resolutions (12).** As above: explicit keys, a reason and evidence; the ROM value is the
  default, and overruling the ROM needs an emulator trace.
- **ROM changes (13).** Every code change and changed battle-script member of the patch has
  an entry in `data/kaizo/rom_diff.toml` with a decoded note. Statuses: `modelled` (the
  engine does what the patched code does), `irrelevant` (cannot change a battle outcome or
  an AI choice; the note says why), `not_reached` (battle-relevant, not modelled, with a
  `when` condition) and `unknown`. `when` keys: `format`, `area_condition`, `move_effect`,
  `ability`, `held_item`, `map_weather`, `map_header`, `trainer_items`. Coverage evaluates
  `when` per fight (and per pool candidate), so a `not_reached` entry refuses only the
  fights that reach it. A `map_weather` or `map_header` condition also lists every trainer
  with a party that no map header places (safety over coverage: such a trainer may be fought
  on any map). An `unknown` entry refuses every Kaizo fight. A change with no entry
  still counts as `unknown`. A move whose battle script the patch changed is refused as
  unsupported only when its `be_seq` member is `not_reached` for that move effect.
- **AI (14).** Only the vanilla trainer AI is transcribed. Kaizo fights are refused on the AI
  until the Kaizo AI is transcribed (milestone K3).
- **Level caps (15).** From `data/kaizo/level_caps.toml`. The Galactic split cap is 100,
  confirmed by you (2026-11-05), although that split comes before Volkner (84).
- **Exp-cap warning (16).** The data for it ships now (exp yields, growth rates, next cap);
  the warning in `prep` is later work.
- **Chance-0 secondary effects (21).** A move whose side effect has a 0% chance in Kaizo
  never applies it: it is plain damage, or unsupported if the move also does something else.
- **Spreadsheet tags (22).** Weather written as just `(RAIN)`, `(HAIL)` or `(SUN)` counts as
  permanent, and bare `(Trick Room)` as permanent Trick Room; this is confirmed by you
  (2026-11-05). It is one constant, `BARE_WEATHER_PERMANENT` in `tools/kaizo/resolve.py`:
  set it to `False` and rebuild if play shows such weather ending. A tag that looks like a
  keyword but does not parse stops the extraction.

## User calls

These are refused until you decide them. The "User calls" section of
`docs/kaizo/discrepancies.md` is the current list, with the ids, counts and evidence of
each; "Settling a user call" above says how to enter an answer. The kinds:

- **Unknown start conditions**: trainers placed on maps that start different weather
  (`start_conditions.json`; 676 Corbin, unreachable, is the one left).
- **No map placement found**: trainers with a party that the ROM's trainer -> map join
  places on no map (for example 468 Raul and the postgame daily trainers 906-920). Their start
  condition is refused as unknown, never read as no weather; the report lists them as one
  call. The story-reachable Pokemon Center daily trainers (874-891, 894-896) are answered.
- **Trick Room on one placement only**: trainers placed both on a Trick Room map and on a
  map without it. Their start condition is refused as unknown until answered (all 35, among
  them Rosalind 509, Puck 559 and Kari 649, are answered: the Galactic HQ fight).
- **Unknown formats**: trainers whose scripts disagree (`battle_formats.json`) or that the
  ROM fights in two different structures on two maps (`[fields]` in `overrides.toml`). All
  are answered today (521 Mercury and 527 Venus: a multi beside Mira 864).
- **Weather where the ROM disagrees with PKCalc and the sheet**, listed until answered
  (Route 213 is rain and Idol Catherine #2 fog, by your answers).
- **Gravity** whose entry cites PKCalc while no sheet row claims it (Champion Cynthia cites
  the sheet row Elite Four Split:298).
- **Unclassified ROM changes** (Ruling 13): two remain, both `unknown` (see "What is refused").

## What is refused

`coverage --game kaizo` exits 2 and lists every reason. For Roark today that includes the
Kaizo AI lines (no policy, `AI_FLAG_RISKY`, per-move script paths), `Kaizo ROM: 1 code changes
unclassified`, and the K3b2 mechanics below. K3b1 (done) models, in both engines: obedience
(Kaizo: every Pokemon always obeys, no RNG draw), permanent sandstorm with the end-of-turn
field pass and Cloud Nine, Sand Veil, Snow Cloak, Swift Swim, Light Screen, Brick Break,
Toxic, Sand Tomb, the Swift family, Pay Day, Gust, Recover, Fake Out, AncientPower, Curse,
DoubleSlap, Self-Destruct, Flail, Rough Skin, Flame Body, Limber, Adaptability, Solid Rock,
Focus Sash, Big Root, Berry Juice, status berries, White Herb, Lagging Tail, Aqua Ring, Rindo
and Chilan berries, and bag-item rows in the pool check. Still refused for Roark (K3b2):
Thunder Fang, Protect, Present, Attract, Cute Charm, Quick Claw, Roar, Nature Power,
Camouflage, and the two terrain ROM conditions (`MapHeader_GetBattleBG`,
`MapHeader_GetMapMatrixID`). The remaining `unknown` ROM change is overlay14
0x022249CC-0x022249CD (the AI script's second Levitate compare in `Basic_CheckForImmunity`
tests Dry Skin 87; K3c). Known blind spot: Kaizo field-script command
0x3F (map headers 257 and 321, common script 9019) sets a party status that persists into
fights on any map; coverage sees it only if the save was taken after that script.
