# K2b Kaizo Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, cross-checked Platinum Kaizo ("Nightmare Update Final") data set beside vanilla — `data/gen4_platinum_kaizo/` built from the patched ROM, PKCalc and the Kaizo docs spreadsheet with every disagreement reported and resolved — selected per run with `--game vanilla|kaizo`, with per-game caches, per-game id constants, Kaizo level caps and Kaizo save reading; every Kaizo fight the engine cannot model is refused with the missing piece named.

**Architecture:** Raw source extractions land in `data/intermediate_kaizo/{rom,pkcalc,docs}/` (K1's `extract_rom` plus two new extractors). `tools/kaizo/` maps PKCalc trainers to ROM ids, loads the committed overrides and resolutions, cross-checks the three sources field by field (writing `docs/kaizo/discrepancies.md`) and writes the resolved intermediates `data/intermediate_kaizo/*.json` in the vanilla schemas plus the new trainer fields; `tools/build_tables.py --game kaizo` runs that resolve step and packs `data/gen4_platinum_kaizo/`. At runtime `nuzline.data.games` names the two games; the CLI, caches, level caps, saves and coverage take the chosen game, and coverage refuses what Kaizo adds that the engine lacks (permanent Trick Room/Gravity, gauntlets, unclassified ROM code changes, the untranscribed Kaizo AI; K2a already refuses two-trainer, tag and partner battles).

**Tech Stack:** Python 3.12, numpy, pytest, ruff; the `rom` uv dependency group (ndspy, openpyxl) for build tools only; node (already installed) to evaluate PKCalc data files in a `vm` context. Run everything with `~/bin/uv.exe run ...`; tools and tests that need the group use `~/bin/uv.exe run --group rom ...`.

**Spec:** `docs/superpowers/specs/2026-11-03-kaizo-data-design.md` (approved 2026-11-03; parent `docs/design.md` §4.1, §6, §9). Read it before any task; §N below are the spec's. This plan is K2b of spec §11; K1 (ROM pipeline) and K2a (vanilla schema extensions) are separate plans and must be merged first.

## Global Constraints

- Starts after K1 and K2a are merged into `main` (K2a itself waits for `claude/m4-prep-search`). File:line references to M4 code below are against `claude/m4-prep-search`; re-check each at the start of the task that touches it.
- Never commit ROM bytes, NARC members, code, the patched ROM, PKCalc or spreadsheet copies, or anything generated from them: `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` and `.cache/` are git-ignored (K1 adds the ignores; Task 1 checks them). Committed Kaizo files are only `data/kaizo/*.toml`, `data/kaizo/start_conditions.json` and `docs/kaizo/*.md`, which hold ids, names, citations and decisions, never bytes.
- Source priority (spec §4): ROM (and traces) > PKCalc > spreadsheet, per field; nothing is resolved silently; a ROM-vs-PKCalc conflict on an executed field fails the build until `data/kaizo/resolutions.toml` has an entry; a resolution against the ROM needs a trace.
- Vanilla stays untouched in behaviour (spec §6.6: "Adding the pipeline changes nothing in vanilla"): `--game` defaults to vanilla, every existing command and test passes unchanged, every vanilla `tables.npz` array stays byte-identical, and K2b does **not** bump `MANIFEST_VERSION` (K2a's single bump, now in `nuzline.data.tables`, stands). The only vanilla artefact change is the manifest's new `"game"` key (Task 1), which changes vanilla's `table_version` once, so vanilla compile caches recompile once; a manifest without the key reads as `platinum_us`, so K2a-built tables keep loading.
- Fail loudly: a Kaizo fact the engine cannot model exactly is a coverage refusal naming it, never an approximation; a game is never inferred from a save; a sheet tag that names a known keyword but does not parse is an extraction error.
- The runtime package `nuzline` gains no dependency and never imports ndspy or openpyxl; `tools/rom/*` and `tools/kaizo/*` import them lazily; tests needing them use `pytest.importorskip`. `nuzline.cli` still imports no jax at module import.
- Tests that need a local input skip when it is absent: ROM/patch/PKCalc/spreadsheet/save through `tools.rom.inputs.load_inputs()` (`RomInputError` or a `None` path → skip), built Kaizo tables through a `kaizo_tables` fixture (skip when `data/gen4_platinum_kaizo/tables.npz` is absent), node through `shutil.which("node")`. ROM-level tests live in K1's `tests/tools/rom/` and use K1's session fixtures (`rom_inputs`, `vanilla_files`, `kaizo_files`, `decomp`, `symbols`); source-level Kaizo tests live in `tests/kaizo/`.
- PKCalc code is never run: only the files listed in Task 6 are evaluated. Node's `vm` is **not** a security boundary; the protection rests on the fixed file list, the zip's SHA-256 being recorded and reviewed, and the hardened context (null-prototype global, no string code generation, no `require`/`process`/timers, time limit), in that order.
- Never touch any folder named `backup`. CLI output stays ASCII; exit codes 0 ok / 1 error / 2 refused.
- ruff `line-length = 100`; cite decomp `file:line` and spreadsheet `sheet:line` / PKCalc `file:line` as existing code cites the decomp. Decomp: `external/pokeplatinum` (main checkout, `../../../external/pokeplatinum` from a worktree).
- The full suite takes ~10 min: run it in the background; per-task runs use the task's test files.

## Rulings (controller's calls; the spec is silent, open or slightly off)

1. **Interfaces assumed from K1 and K2a.** From K1: `tools.rom.inputs` (`load_inputs`, `RomInputs`, `RomInputError`, `read_checked`, `VANILLA_SHA1`, `KAIZO_PATCH_SHA256`), `tools.rom.patch.kaizo_rom`, `tools.rom.nds.RomFiles` (`paths()` without a leading slash), `tools.rom.symbols.SymbolMap`, `tools.rom.diff` (`load_classification(path) -> dict[tuple[str, str], tuple[str, str]]` over `[[code]] {module, symbol, status, note}`, stale entries an error, K1 Ruling 16), `tools.rom.readers.trainers` (called `read_trainers(rom, ...)` here), `tools.rom.readers.items`, `tools.extract_rom` (`--game`, `--out`) and its `--game kaizo` refusal (K1 Ruling 13), which Task 8 removes. From K2a: `trainers.json` `"format"` (`single|double|double_two_trainer|multi`), `"opponent_2"` and `"partner"` with `nuzline.data.trainer.PartnerVariant(trainer, starter, player_gender)`, the script-based `battle_structures` that the ROM reader runs on the patched ROM too, and coverage's refusal of every non-single format; species `"exp_yield"` / `Tables.species_exp_yield`; moves `"ai_effect"`/`"mechanic_effect"` and `Tables.move_ai_effect`; form-indexed species rows (`Tables.species_form_row`); `<inter>/renamed.json` → manifest `renamed` and its `Names` error (K2a Ruling 14); `MANIFEST_VERSION` and `check_manifest` in `nuzline.data.tables`; `nuzline.data.learnsets.Learnsets` (level-up, TM, tutor, egg, evolutions) and the TM→move and tutor tables. **Task 1 Step 0** reads the merged K1/K2a code and, where a name differs, edits this plan's references (never those modules) before any code is written.
2. **Directory layout.** `extract_rom --game kaizo --out data/intermediate_kaizo/rom`; PKCalc → `data/intermediate_kaizo/pkcalc/`; spreadsheet → `data/intermediate_kaizo/docs/`. The resolved intermediates the runtime reads (`Game.intermediate_dir`) are `data/intermediate_kaizo/*.json`, written by the resolve step that `build_tables --game kaizo` runs before packing (spec §6.6's "from `data/intermediate_kaizo/` and the overrides", in one command).
3. **Names (spec §7.3).** Kaizo display names are PKCalc's (`moves.js`/`species.js`/`items.js`/`abilities.js` `name`, joined by `num`, forms by Ruling 4's table). Renamed slots use K2a's mechanism unchanged: Task 15 writes `inter/renamed.json` in K2a's format (`{"species"|"move"|"item"|"ability": {old lowercase vanilla name: new name}}`), and K2a's `Names` raises `KeyError("unknown move 'Rollout': in this game that slot is 'Accelerock'")`. K2b adds only **aliases**: ROM text-bank names that differ from the display name (the save's and the in-game spelling, e.g. "Water Ball", "HP Water"), in manifest `aliases: {"species"|"move"|"item"|"ability": {alias: id}}` (singular kinds, as K2a's). A vanilla name of a renamed slot is never an alias even when the ROM text still reads it (the report lists it); an alias equal to another id's name or to a `renamed` key is a build error.
4. **Forms.** PKCalc form keys map to decomp form numbers by this table (decomp `res/pokemon/<species>/forms/`): Deoxys attack 1, defense 2, speed 3; Wormadam sandy 1, trash 2; Rotom heat 1, wash 2, frost 3, fan 4, mow 5; Giratina origin 1; Shaymin sky 1. Personal indices follow `Pokemon_GetFormNarcIndex` (`src/pokemon.c:4933-4955`: Deoxys 496-498, Wormadam 499-500, Giratina 501, Shaymin 502, Rotom 503-507). Any other PKCalc form key is an extraction error.
5. **Trainer names.** `Trainer.name` stays the ROM text name (`resolve_trainer` unchanged); the new `Trainer.label` is the PKCalc label ("Leader Roark", "Pokémon Trainer Barry #3 [Empoleon]") or `None`; `--trainer` matches either, case-insensitively.
6. **Kaizo trainer struct.** Kaizo's party entries read vanilla's `u16 ivScale` (`include/struct_defs/trainer_data.h:35`) as `u8 difficulty` + `u8 ability_override` (the spreadsheet's "Ability Number": 16 forces ability slot 0, 32 slot 1, 2 female; spec §5.4). Task 8 confirms the layout on Kaizo Roark and on the override byte's value set, and against `docs/kaizo/rom_diff.md`'s entry for `TrainerData_BuildParty`; the patched routine is **not** transcribed in K2b. The spreadsheet predates PKCalc's ROM build (spec §5.4), so ROM-vs-sheet differences in difficulty or the override byte are report rows (Task 13), not test failures.
7. **Nature, gender, ability.** Kaizo `TrainerMon`s carry explicit `nature`/`gender`/`ability_slot` from PKCalc (spec §5.4, §13). A trainer gains `"pid_routine"`: `"vanilla"` (the default when the key is absent, so vanilla JSON is unchanged) or `null` (Kaizo: not transcribed). With `null`, a mon without explicit values cannot be built: `build_trainer_party` raises and coverage refuses the fight ("no PKCalc set: nature/gender/ability unknown"). The vanilla PID routine is run only as a cross-check report section (Task 13).
8. **Area conditions.** Weather and fog come from the ROM's map-header weather through the trainer → map join (K1's ROM reader, as `extract_decomp._trainer_maps`); map weather is permanent in battle (`FIELD_CONDITION_*_PERM`, `src/battle/battle_controller_player.c:1115-1190`). Date-dependent Kaizo maps are settled in a new committed `data/kaizo/start_conditions.json` (the format of `tools/start_conditions.json`, keyed by id, with `"trainer"` = the **Kaizo ROM text name** at that id, since vanilla `TRAINER_*` constants do not name Kaizo slots). `extract_decomp._start_condition_overrides` validates `trainer` against `generated/trainers.txt` and would reject these, so Task 8 gives it a `name_at: Callable[[int], str]` parameter (vanilla: the `TRAINER_*` list, unchanged; Kaizo: the ROM trainer-name bank), and a wrong name or a drifted id raises with the same message shape. Trick Room and Gravity, which no map header holds, live in `data/kaizo/overrides.toml` (spec §6.4, §13). A trainer claimed by two areas (Galactic Rosalind, spec §5.8) is settled by the ROM trainer → map join, recorded as an overrides entry citing it.
9. **Split, optional, left side: derived, not copied.** A trainer's split (which `* Split` sheet lists it, and its line), `optional` (an `OPTIONAL` tag in its header or section) and `left_side` (the split's "Left Side AI" column) are derived from the spreadsheet at every build with a per-field source citation in `trainers.json`, not hand-copied into `overrides.toml`; overrides may correct any of them. Gauntlet groups, Trick Room and Gravity **are** in `overrides.toml` (spec §13), generated as a reviewed draft from the sheet. A "B2B With X" pair is a two-trainer gauntlet group; a section tagged "GYM TRAINERS ARE A GAUNTLET" is a group of that section's trainers in sheet order, leader excluded. Every group's `rules` table stays empty until play reaches it (spec §12.4: `heal_between`, `change_team` unset).
10. **Formats come from the ROM scripts (K2a).** `format`, `opponent_2` and `partner` are read from the patched ROM's field scripts by K2a's `battle_structures` and coverage refuses every non-single format as K2a built it; K2b adds no field and no refusal for them. Task 13 cross-checks them as executed rows: ROM `double` vs PKCalc `flags.js` `battleType.trueDouble`; ROM `double_two_trainer` and its `opponent_2` vs PKCalc `battleType.double` pairs (both labels mapped to ids); ROM `multi` vs PKCalc `battleType.tag` (the enemy pair against `opponent_2`; the partner only as present/absent, since PKCalc's partner entries are template labels such as `Pokémon Trainer {RIVAL2_NAME} @ Jubilife City [{RIVAL2_STARTER_2}]` that name no id); and all three against the sheet's `DOUBLE`/`DOUBLE BATTLE`, `MULTI BATTLE with X` (X by name) and `TAG PARTNER`/`TAG BATTLE with X` tags.
11. **Rivals.** `Trainer.player_starter` is the player's starter species id (387 Turtwig, 390 Chimchar, 393 Piplup) a rival/partner variant is fought with, from the sheet's `(Player chose|picked|selected X)` / `(TAG PARTNER, if Player chose X)` headers, else from PKCalc's bracket, which names **Barry's** line: a bracket in the Chimchar line (Chimchar, Monferno, Infernape) means the player has Turtwig 387; the Piplup line (Piplup, Prinplup, Empoleon) means Chimchar 390; the Turtwig line (Turtwig, Grotle, Torterra) means Piplup 393. Template labels (`[{RIVAL_STARTER_n}]`, `{RIVAL2_*}`) are the empty rival templates and get no starter. Sheet and PKCalc must agree where both exist (a disagreement is an executed row). The save's starter is `VAR_PLAYER_STARTER` (`generated/vars_flags.txt:4159` = 0x4030, after `VAR_OBJ_GFX_ID_F` = 0x402F; `src/system_vars.c:67`). Without `--save`, a name matching several starter variants is an error listing their ids.
12. **Resolutions file.** Each `[[resolution]]` has `table`, `keys` (an explicit list: no wildcards, so a new disagreement never falls under an old entry), `field`, `pkcalc` (the PKCalc value it resolves), `use` (`"rom"` default, or `"pkcalc"`, which requires `evidence` starting `trace:`), `reason`, `evidence`. An entry whose key no longer disagrees, or whose `pkcalc` value no longer matches, is **stale** and fails the build. ROM vs sheet with PKCalc silent (hold params, effect labels, difficulty, the override byte, tutor costs) is listed, ROM used, no entry needed.
13. **Unclassified ROM changes refuse every Kaizo fight.** Every `data/kaizo/rom_diff.toml` entry still `"unknown"` is assumed reachable from every fight (no reachability analysis in K2b): Kaizo coverage adds one line naming the count and `docs/kaizo/rom_diff.md`. Task 8 extends K1's file with a second table kind, `[[member]] {narc, index, status, note}` (`narc` as `RomFiles.paths()` spells it, no leading slash), for the changed members of `battle/skill/be_seq.narc`, `battle/skill/sub_seq.narc` and `battle/tr_ai/tr_ai_seq.narc` (`platinum.us/filesys.csv:180,181,284`), read by a sibling loader `load_member_classification` with K1's rules (missing = unknown, an entry matching no changed member is stale and an error). A move whose battle effect script (`be_seq` member = effect id) changed has mechanic `UNSUPPORTED` in Kaizo unless its member entry is `"modelled"`.
14. **AI policy is per game.** The transcribed AI (`ai/policy.py`, `ai/switching.py`, `ai/outcome.py` site tables, `ai/commands.py` `MOVE_EFFECTS`) is the vanilla policy. Its id constants are vanilla-policy facts, asserted for vanilla only; for Kaizo coverage refuses the AI ("no transcribed AI policy for this game (spec 5.10)") until K3 transcribes one. Engine constants used for every game are asserted per game at load (Task 4). Switching `MOVE_EFFECTS` to read `tables.move_ai_effect` (K2a Ruling 9 hands it to §7.2) is **deferred to K3**: while the Kaizo AI is refused it changes no reachable result, K2a's test already pins the dict equal to `move_ai_effect` for vanilla, and the switch belongs with the Kaizo AI's own move set.
15. **Level caps.** `data/kaizo/level_caps.toml` holds the user's table as `[splits]` (split sheet → cap) and `[fights]` (PKCalc label prefix → cap, every starter variant). The user's "Coronet/Elite Four 100" is read as Galactic split 100 and Elite Four split 100, although the Galactic split is fought before Volkner (84): flagged for the user at the Task 19 review. Vanilla keeps `config/level_caps.toml`; `prep/caps.py` picks the file per game.
16. **Exp-cap warning: data now, warning later.** K2b ships Kaizo exp yields and growth rates (ROM, cross-checked) and `next_fight_cap`. The warning itself (spec §5.12 third bullet) is a later task, as spec §11 lists: M4's prep has no hook that knows which participants fought which opponent in a line (`prep.report.render` sees the chosen setup and the plan tree, not per-opponent participation, and `build_problem`'s warnings run before any line exists).
17. **`--game` is a per-subcommand option** (`nuzline coverage --game kaizo ...`), default `vanilla`. There is no run config file today, so spec §8's "or a `game` field in the run config" is met by recording `game` in every run record and checkpoint fingerprint; no config file is added.
18. **`tables.DEFAULT` stays the vanilla directory** for the existing callers and tests; game-aware code uses `get_game(name).tables_dir` (spec §8's "a function of the game").
19. **Source hashes are recorded, not pinned.** The manifest and report record the PKCalc zip and spreadsheet SHA-256; resolutions are re-validated each build (Ruling 12). Tests asserting audited counts skip unless the input hash equals `PKCALC_SHA256 = "849733e658fc0116aaf0fc2f27db71d416165d79107cf3a0e29effce7062eb0f"` / `KAIZO_DOCS_SHA256 = "41c0c7fe8bb2b5d2179670f6e92655431ff86752a45370a11605715be8b1d74f"` (the copies audited on 2026-11-03). Counts the spec computed on the spreadsheet's inputs (the PID section's 1816 of 1936) are not asserted on ROM inputs: the build recomputes them, the report records them, the commit message quotes them.
20. **Kaizo forms in saves** are lifted by K2a's form-indexed tables; K2b only tests them on the Maylene save.
21. **Chance-0 secondaries (spec §5.2, §7.1; K2a Ruling 10 hands them here).** A move whose mapped effect is a `SECONDARY_*` class, whose decomp effect is not in `CERTAIN_SECONDARY_EFFECTS` and whose chance is 0 never applies the secondary: `build_tables` packs it as `EffectId.PLAIN, (0, 0, 0)` when the mapping's params carry nothing but the secondary (`thaws_user` 0), else `UNSUPPORTED`; the manifest lists those move ids as `no_secondary_moves`. The existing check stays for the other direction (a certain effect with chance ≠ 0 raises). No vanilla move is affected (the current check would have raised), so vanilla arrays are unchanged. Ice Ball's effect 260 is unmapped today, so it is `UNSUPPORTED` before this rule applies.
22. **Spreadsheet header tags.** Parentheticals are split on commas; each item is matched against the tag grammar of Task 7; items that are pure location notes ("Middle Door", "Path 2", "outside of Pastoria Gym") are kept as `note`. Ability names (vanilla ability list, e.g. "SHADOW TAG") are notes, checked before keywords. An item that contains a tag keyword (`PERMANENT`, `RAIN`, `SUN`, `HAIL`, `SANDSTORM`, `FOG`, `TRICK ROOM`, `GRAVITY`, `DOUBLE`, `MULTI`, `TAG`, `GAUNTLET`, `B2B`, `OPTIONAL`, `picked`, `chose`, `selected`) but does not parse is an `ExtractError` naming the cell. Bare `(SUN)`, like bare `(RAIN)`/`(HAIL)`, is read as `assumed_permanent: true`, although spec §12.1 settles only rain and hail: flagged for the user at the Task 19 review. Bare `(Trick Room)` (Leader Cyrus, `Galactic Split:727`) is a Trick Room claim with `assumed_permanent: true`, checked against overrides like the others.
23. **Outside K2b:** spec §5.7 encounter tables (the spec: "not needed for the first milestones"); §9.7's fidelity-class display (coverage accepts no Kaizo fight in K2b; the per-field `sources` recorded in `trainers.json` and the report are what K3 derives the classes from); §9.5 golden traces (a per-fight gate, spec §11); the exp-cap warning (Ruling 16); `MOVE_EFFECTS` from `move_ai_effect` (Ruling 14).

## Review Focus

1. **`--game kaizo` with no Kaizo tables built, or a tables directory built for the other game**: one error line naming `python -m tools.build_tables --game kaizo` (or both game ids), exit 1; `load_game` checks for `tables.npz` and reads `manifest.json` with `json` **before** importing `nuzline.data.tables` (which imports `jax.numpy`), so jax is never loaded on that path; tests in Task 1 and Task 18 (a subprocess asserts `"jax" not in sys.modules` after the exit-1 path).
2. **Old cache layout left behind** (`.cache/jax/<32 hex>` from before this change) and switching games back and forth: the old flat directories are pruned once, and enabling one game never deletes the other game's directory; test in Task 3.
3. **`save show --game kaizo` on a box mon whose Kaizo growth rate changed** (11 species, e.g. Onix 95: Medium Fast → Slow): a synthetic box Onix with 100000 exp is level 46 under vanilla tables and 43 under Kaizo tables; test in Task 17.
4. **A PKCalc label with no ROM match, or two labels claiming one id**: build error listing them, never a silent pick; test in Task 9.
5. **A PKCalc update that makes a resolution stale**: the build fails naming the entry; test in Task 11.

---

### Task 1: Game registry and the manifest's `game`

**Files:**
- Create: `src/nuzline/data/games.py`
- Modify: `tools/build_tables.py` (`--game` argument; manifest key `game`), `data/gen4_platinum/manifest.json` (regenerated: gains `game` only), `.gitignore` (check K1's entries; add any of `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` missing)
- Test: `tests/data/test_games.py`, `tests/tools/test_build_tables_game.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Game(id: str, tables_dir: Path, intermediate_dir: Path)`; `GAMES: dict[str, Game]` with `"platinum_us"` (`data/gen4_platinum`, `data/intermediate`) and `"platinum_kaizo_nightmare_final"` (`data/gen4_platinum_kaizo`, `data/intermediate_kaizo`); `ALIASES = {"vanilla": "platinum_us", "kaizo": "platinum_kaizo_nightmare_final"}`.
  - `get_game(name: str) -> Game` (id or alias, case-insensitive; `ValueError` listing the choices).
  - `manifest_game(manifest: dict) -> str` (`manifest.get("game", "platinum_us")`).
  - `class GameDataError(ValueError)`; `load_game(game: Game) -> tuple[Tables, Names]`: first, without importing `nuzline.data.tables`, raises `GameDataError` when `tables.npz` is absent (message names `python -m tools.build_tables --game <alias>`) or when `manifest_game(json manifest) != game.id` (message names both ids); only then imports `nuzline.data.tables` and loads (K2a's `check_manifest` runs inside); runs Task 4's `check_constants` once Task 4 lands.
  - `build_tables.build(intermediate_dir, out_dir, game: str = "platinum_us")`; manifest `"game"`. `MANIFEST_VERSION` unchanged (Global Constraints).

- [ ] **Step 0: Reconcile with K1/K2a** (Ruling 1): read the merged `tools/rom/`, `tools/extract_rom.py`, K2a's `tools/extract_decomp.py`, `tools/build_tables.py`, `src/nuzline/data/{tables,trainer,learnsets}.py`, `src/nuzline/engine/coverage.py`; correct this plan's assumed names; note M4-branch line drift for `cli.py`, `prep/caps.py`, `jaxcache.py`, `engine/coverage.py`.
- [ ] **Step 1: Write failing tests**

```python
def test_aliases_and_ids():
    assert get_game("vanilla") is GAMES["platinum_us"] is get_game("PLATINUM_US")
    assert get_game("kaizo").tables_dir.name == "gen4_platinum_kaizo"
def test_unknown_game():
    with pytest.raises(ValueError, match="vanilla, kaizo"):
        get_game("emerald")
def test_missing_tables_names_the_build_command(tmp_path):
    g = Game("platinum_kaizo_nightmare_final", tmp_path, tmp_path)
    with pytest.raises(GameDataError, match="tools.build_tables --game kaizo"):
        load_game(g)
def test_manifest_of_other_game_refused(tmp_path):  # copy vanilla tables into tmp_path, load as kaizo
    with pytest.raises(GameDataError, match="platinum_us.*platinum_kaizo_nightmare_final"):
        load_game(Game("platinum_kaizo_nightmare_final", tmp_path, tmp_path))
def test_manifest_without_game_is_vanilla():
    assert manifest_game({"version": 8}) == "platinum_us"
def test_vanilla_manifest_game():  # committed manifest
    assert json.loads((TABLES_DIR / "manifest.json").read_text())["game"] == "platinum_us"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/data/test_games.py tests/tools/test_build_tables_game.py -q`
- [ ] **Step 3: Implement**; rebuild vanilla tables (`python -m tools.build_tables`), confirm with a one-off `np.load` comparison that every array is byte-identical, commit the regenerated `manifest.json`.
- [ ] **Step 4: Run, expect PASS**, then `tests/data tests/test_jaxcache.py -q`.
- [ ] **Step 5: Commit** `feat(data): game registry and the manifest's game id`

### Task 2: Per-game names: aliases and provenance fields

**Files:**
- Modify: `src/nuzline/data/tables.py` (`Names`)
- Test: `tests/data/test_names_per_game.py`

**Interfaces:**
- Consumes: manifest `aliases` (Ruling 3; written by Task 15; absent in vanilla = empty), K2a's `renamed` handling (unchanged).
- Produces: `Names.species/move/item/ability(name)` resolve display names, then aliases (same case-insensitive index); an alias colliding with a display name or a `renamed` key raises `ValueError` at `Names` construction naming the alias and both ids. `Names.game: str` (`manifest_game`) and `Names.unclassified_rom_changes: int` (manifest key, 0 when absent).

- [ ] **Step 1: Write failing tests** (synthetic manifests from K2a's `MINIMAL_MANIFEST` test helper, or the vanilla manifest with `dict` edits)

```python
def test_alias_resolves(vanilla_manifest):
    m = {**vanilla_manifest, "aliases": {"move": {"Water Ball": 4}}}
    assert Names(m).move("water ball") == 4 and Names(m).move("Comet Punch") == 4
def test_alias_colliding_with_a_name_refused(vanilla_manifest):
    m = {**vanilla_manifest, "aliases": {"move": {"Tackle": 4}}}
    with pytest.raises(ValueError, match="'Tackle'.*33.*4"):
        Names(m)
def test_renamed_slot_keeps_k2a_error(vanilla_manifest):  # Kaizo-shaped data through K2a's lookup
    names = vanilla_manifest["move_names"].copy(); names[205] = "Accelerock"
    m = {**vanilla_manifest, "move_names": names,
         "renamed": {"species": {}, "move": {"rollout": "Accelerock"}, "item": {}, "ability": {}}}
    with pytest.raises(KeyError, match="in this game that slot is 'Accelerock'"):
        Names(m).move("Rollout")
def test_vanilla_defaults(names):
    assert names.move("Rollout") == 205 and names.game == "platinum_us"
    assert names.unclassified_rom_changes == 0
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/data/test_names_per_game.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS** (`tests/data -q`)
- [ ] **Step 5: Commit** `feat(data): per-game name aliases and the manifest's game in Names`

### Task 3: Per-game compile caches

**Files:**
- Modify: `src/nuzline/jaxcache.py` (`enable_compile_cache`, module docstring), callers in `src/nuzline/cli.py` (M4 `:416`, `:499`, `:603`) and `benchmarks/exhaustive_roark.py:88`
- Test: `tests/test_jaxcache.py`

**Interfaces:**
- Produces: `enable_compile_cache(version: str, game: str = "platinum_us") -> Path` using `cache_root()/"jax"/game/version`; it prunes version siblings only inside `jax/<game>/` and, once, version-named directories directly under `jax/` (the pre-K2b layout). `cached_export` is unchanged: its key already covers argument shapes and `code_version()`, and tables are arguments.

- [ ] **Step 1: Write failing tests** (with `NUZLINE_CACHE_DIR` pointed at `tmp_path`, `set_compile_cache_dir` monkeypatched to a no-op)

```python
def test_game_dirs_do_not_prune_each_other(tmp_path, monkeypatch):
    a = enable_compile_cache("a" * 32, "platinum_us")
    b = enable_compile_cache("b" * 32, "platinum_kaizo_nightmare_final")
    assert a.is_dir() and b.is_dir() and a.parent != b.parent
def test_other_versions_of_same_game_pruned(tmp_path, monkeypatch):
    old = enable_compile_cache("a" * 32, "platinum_us"); new = enable_compile_cache("c" * 32, "platinum_us")
    assert not old.exists() and new.is_dir()
def test_legacy_flat_dirs_pruned(tmp_path, monkeypatch):
    legacy = tmp_path / "jax" / ("d" * 32); legacy.mkdir(parents=True)
    enable_compile_cache("a" * 32, "platinum_us")
    assert not legacy.exists()
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/test_jaxcache.py -q`
- [ ] **Step 3: Implement**; callers pass the game id (vanilla until Task 18 wires `--game`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(jaxcache): one compile-cache directory per game`

### Task 4: Game constants classified (spec §7.2)

**Files:**
- Create: `src/nuzline/data/game_constants.py`
- Modify: `src/nuzline/data/games.py` (`load_game` calls `check_constants`), `src/nuzline/engine/coverage.py` (Kaizo AI refusal, Ruling 14), `tools/extract_decomp.py` `extract_items` and K1's `tools/rom/readers/items.py` (each item row gains `"hp_restored"`: decomp `res/items/data/*.json` `itemUseParams.hpRestored`, else 0; ROM `ItemPartyParam.hpRestored`, `include/item.h:48`; K2a's round-trip test then covers it), `tools/effect_map.py` (module docstring: which lists are per-game now; Task 5 moves two of them)
- Test: `tests/data/test_game_constants.py`, `tests/engine/test_coverage.py`, `tests/tools/test_extract_decomp.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Constant(where: str, kind: str, id: int, name: str, scope: str, hold_effect: int | None = None, hp_restored: int | None = None)`; `kind` in `move|ability|item|type|species`; `scope` `"engine"` (asserted for every game) or `"vanilla_policy"` (asserted for games whose AI policy is the vanilla one).
  - `CONSTANTS: tuple[Constant, ...]`; `GAME_DATA: dict[str, str]` (`"module.ATTR"` → the intermediate file that replaced it, for Task 5's lists); `EXEMPT: dict[str, str]` (`"module.ATTR"` → why it is not a game id table, e.g. `engine/ids.py Stat`: engine index order).
  - `POLICY: dict[str, str | None] = {"platinum_us": "vanilla", "platinum_kaizo_nightmare_final": None}`.
  - `check_constants(game_id: str, names: Names, items: Sequence[dict]) -> list[str]` where `items` is the game's `intermediate_dir/items.json` rows (`hold_effect`, `hp_restored`), one line per mismatch: `"ai/switching.py MOVE_STRUGGLE: id 165 is 'X' in this game, expected 'Struggle'"`, `"engine/ids.py POTION_HP_RESTORED: Potion restores 30 in this game, expected 20"`.
  - coverage: when `POLICY[names.game] is None`, `_check_trainer` adds `"{trainer}: no transcribed AI policy for this game (spec 5.10)"` as an `AI policy` line.

Classification (each entry cites its decomp source in a comment; the tests below enforce completeness):

| Constant (M4 branch) | Class |
|---|---|
| `ai/commands.py` `ABILITY_IDS`, `TYPE_IDS`; `ai/switching.py` `ABILITY_IDS`, `_TYPE_FLYING`, `_TYPE_STEEL`; `engine/ids.py` `Type`, `MOVE_STRUGGLE` (`:208`); `engine/stats.py` `SPECIES_SHEDINJA`; `ai/switching.py` `MOVE_STRUGGLE`; `prep/problem.py` `HEART_SCALE` (by name) | engine, asserted by name |
| `ai/commands.py` `ITEM_EXPERT_BELT`; `ai/switching.py` `ITEM_IRON_BALL` | engine, asserted by name **and** hold effect |
| `engine/ids.py` `POTION_HP_RESTORED` (Potion's `hp_restored`) | engine, asserted by name **and** `hp_restored` |
| `ai/commands.py` `MOVE_EFFECTS`, `BattleEffect`, `MOVE_IDS`, `SOUND_MOVES` (Basic_CheckSoundproof, `script.s:121-131`); `ai/switching.py` `MOVE_IDS`, `_MOVES`, `PLAYER_*_REFUSALS`; `ai/policy.py` `_STATE_MOVE_IDS`, the Dry Skin quirk (`:150`); `ai/outcome.py` `_SITES` | vanilla_policy (`MOVE_EFFECTS` stays a dict, Ruling 14) |
| `engine/ids.py` `MoveTarget`, `EffectId`, `Status`, `Stat`, `MoveFlag`, `Weather`, `ItemId`, `TrainerItemId`, `AbilityId` | `EXEMPT`: engine encodings (decomp `RANGE_*` order for `MoveTarget`, `generated/move_ranges.txt`), not game ids; a Kaizo change to range or effect handling is a `rom_diff.toml` entry (Ruling 13) |
| `tools/effect_map.py` `SOUND_MOVES`, `PUNCH_MOVES` (battle_lib.c `sSoundMoves:3473`, `sPunchingMoves:6566`) | game data → `move_lists.json` (Task 5) |
| `tools/effect_map.py` `TYPE_CHART_STATUS_MOVES` (Thunder Wave immediate, `battle_controller_player.c:2357`) | engine, asserted by name (a Kaizo change to that function is a `rom_diff.toml` entry, Ruling 13) |
| `tools/effect_map.py` `DECOMP_ITEM_TO_ID`, `DECOMP_TRAINER_ITEM_TO_ID` | keyed by name; Task 15 also requires the game's hold effect to equal vanilla's, else `UNSUPPORTED` |
| `tools/effect_map.py` `DECOMP_EFFECT_TO_ID`, `CERTAIN_SECONDARY_EFFECTS`, `DECOMP_ABILITY_TO_ID`, `UNSUPPORTED_MOVES` | keyed by effect id / ability name: engine semantics of vanilla scripts; Kaizo script changes handled by Ruling 13 |

- [ ] **Step 1: Write failing tests**

```python
LISTED = ["nuzline.ai.commands.ABILITY_IDS", "nuzline.ai.commands.MOVE_IDS", ...]  # every table row above
MODULES = ["nuzline.ai.commands", "nuzline.ai.switching", "nuzline.ai.policy", "nuzline.engine.ids",
           "tools.effect_map"]
def test_every_id_constant_is_classified():
    assert set(LISTED) <= {c.where for c in CONSTANTS} | set(GAME_DATA) | set(EXEMPT)
def test_no_unlisted_id_tables():
    # ast scan of MODULES: every module-level IntEnum class, and every UPPER/_UPPER name assigned a
    # dict/set/tuple/frozenset literal (or frozenset()/tuple() call) holding an int or name->int
    found = scan_id_tables(MODULES)
    assert found - set(LISTED) - set(GAME_DATA) - set(EXEMPT) == set()
    assert "nuzline.engine.ids.MoveTarget" in found       # the scan sees enums
def test_hp_restored_extracted(decomp):
    assert next(i for i in extract_items(decomp) if i["name"] == "Potion")["hp_restored"] == 20
@pytest.mark.parametrize("game", ["platinum_us", "platinum_kaizo_nightmare_final"])
def test_constants_hold(game, request):   # Kaizo skips without the Kaizo tables
    names = request.getfixturevalue("names" if game == "platinum_us" else "kaizo_names")
    items = json.loads((GAMES[game].intermediate_dir / "items.json").read_text("utf-8"))
    assert check_constants(game, names, items) == []
def test_kaizo_style_mismatch_reported(vanilla_manifest, vanilla_items):
    names = vanilla_manifest["move_names"].copy(); names[165] = "Struggle Bug"
    potion = [dict(i, hp_restored=30) if i["name"] == "Potion" else i for i in vanilla_items]
    lines = check_constants("platinum_us", Names({**vanilla_manifest, "move_names": names}), potion)
    assert any("MOVE_STRUGGLE" in s for s in lines) and any("restores 30" in s for s in lines)
def test_kaizo_trainer_ai_refused(kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party):
    report = check_opponent(kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party, Weather.NONE)
    assert "Roark: no transcribed AI policy for this game (spec 5.10) (AI policy UNSUPPORTED)" in report.missing
```
(Fixtures `vanilla_items` (`data/intermediate/items.json`), `kaizo_tables`, `kaizo_names`, `kaizo_manifest`, `kaizo_learnsets`, `kaizo_trainers`, `kaizo_roark` and `kaizo_roark_party` are added to `tests/conftest.py` here, the Kaizo ones skipping when the Kaizo tables are absent; the trainer ones load through Task 14's `load_trainers`, so they also skip until Task 15 builds.)

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/data/test_game_constants.py tests/engine/test_coverage.py tests/tools/test_extract_decomp.py -q`
- [ ] **Step 3: Implement**; re-extract vanilla (`data/intermediate/items.json` gains `hp_restored`; tables and manifest unchanged, since `build_tables` does not pack it).
- [ ] **Step 4: Run, expect PASS** (Kaizo cases skip until Task 15), then `~/bin/uv.exe run --group rom pytest tests/tools/rom -q` for K2a's round trip.
- [ ] **Step 5: Commit** `feat(data): classify every game id constant; Kaizo AI refused until transcribed`

### Task 5: Sound and punch lists as per-game data

**Files:**
- Modify: `tools/extract_decomp.py` (new `extract_move_lists` → `move_lists.json` from `src/battle/battle_lib.c` `sSoundMoves`, `sPunchingMoves`), K1's ROM reader set (`tools/rom/readers/moves.py`: `read_move_lists(rom, vanilla_lists) -> dict[str, list[int]]`), `tools/extract_rom.py` (writes `move_lists.json`), `tools/build_tables.py` (`_move_flags` by id from `move_lists.json`), `tools/effect_map.py` (drop `SOUND_MOVES`/`PUNCH_MOVES`)
- Test: `tests/tools/test_extract_decomp.py`, `tests/tools/rom/test_move_lists.py`, `tests/tools/test_build_tables_game.py`

**Interfaces:**
- Produces: `move_lists.json` `{"sound": [ids], "punch": [ids]}`; `read_move_lists` finds each vanilla list as consecutive little-endian u16 in the ROM's ARM9 and overlays (decompressed): exactly one hit is required in the vanilla ROM; in a patched ROM the same bytes found exactly once mean the list is unchanged, otherwise `ExtractError("sPunchingMoves not found unchanged: transcribe it into data/kaizo/overrides.toml [move_lists]")`. Task 10's overrides loader supplies `[move_lists]` when present. No manifest key is added (the lists reach the runtime only through `move_flags`).

- [ ] **Step 1: Write failing tests**

```python
def test_decomp_lists(decomp):
    lists = extract_move_lists(decomp)
    assert 4 in lists["punch"] and len(lists["punch"]) == 15 and len(lists["sound"]) == 12
def test_vanilla_rom_lists_match_decomp(vanilla_files, decomp):  # tests/tools/rom; skips without ROM/ndspy
    lists = extract_move_lists(decomp)
    assert read_move_lists(vanilla_files, lists) == lists
def test_flags_by_id_unchanged_for_vanilla(tables):  # rebuilt vanilla: same move_flags as before
    assert int(tables.move_flags[4]) & MoveFlag.PUNCH   # Comet Punch
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_extract_decomp.py tests/tools/rom/test_move_lists.py tests/tools/test_build_tables_game.py -q`
- [ ] **Step 3: Implement**; re-extract and rebuild vanilla; `tables.npz` and `manifest.json` must be byte-identical to Task 1's: check with `np.load` and `cmp` in a one-off comparison.
- [ ] **Step 4: Run, expect PASS**, then `tests/tools tests/engine/test_abilities_items.py -q`
- [ ] **Step 5: Commit** `feat(data): Soundproof and punch lists by id, per game`

### Task 6: PKCalc extractor

**Files:**
- Create: `tools/kaizo/__init__.py`, `tools/kaizo/pkcalc_eval.js`, `tools/extract_pkcalc.py`
- Test: `tests/kaizo/__init__.py`, `tests/kaizo/conftest.py` (input fixtures), `tests/kaizo/test_extract_pkcalc.py`

**Interfaces:**
- Consumes: `load_inputs().pkcalc_zip`.
- Produces:
  - `pkcalc_eval.js FILE GLOBAL OUT`: reads `FILE`, rewrites top-level `const`/`let` to `var`, builds the context as `vm.createContext(Object.create(null), {codeGeneration: {strings: false, wasm: false}})`, runs the code with `vm.runInContext(code, ctx, {timeout: 10000})`, writes `JSON.stringify(ctx[GLOBAL])` to `OUT`; nothing else reaches the context (no `require`, `process`, timers). The file's header comment states the Global Constraints' wording: `vm` is not a security boundary.
  - `PKCALC_FILES: dict[str, str]` (member → global): `src/js/data/dex/{species,moves,abilities,items,natures,types}.js` → `SPECIES, MOVES, ABILITIES, ITEMS, NATURES, TYPES`; `src/js/data/{sets,party_order,trainer_flags,flags}.js` → `SETDEX_PK, PARTY_ORDER_PK, TRAINER_AI_FLAGS_PK, FLAGS_PK`. Members are matched under the zip's single top folder; any other member is never extracted; a member path containing `..` or absolute is an error.
  - `extract_pkcalc(zip_path: Path, out: Path) -> dict` writing `out/raw/<global>.json` and the normalised `out/pkcalc.json`: `sha256`; `moves[num] = {name, aliases: [shortName, calcName], type, category, power, accuracy, pp, chance, priority, target, effect_text, machine, tutor, price}`; `species[personal key] = {num, form, name, base_stats (ids.Stat order), types, abilities, catch_rate, exp_yield, growth_rate, gender_ratio, evolutions, level_up [(level, move num)], tm [move nums], tutor [move nums]}` (form by Ruling 4); `items[num] = {name, desc}`; `abilities[num]`; `trainers[label] = {party: [{species, form, level, item, ability, nature, gender, ivs (6), moves}]}` in `PARTY_ORDER_PK` order (empty orders = rival templates, kept with `template: true`); `ai_flags[label] = bit mask` with bits Basic 0, EvaluateAttack 1, Expert 2, Setup 3, Risky 4, DamagePriority 5, BatonPass 6, TagStrategy 7, CheckHP 8, Weather 9, Harassment 10 (`coverage.AI_FLAG_NAMES` order); `formats = {"true_double": [labels], "double": [[e1, e2]], "tag": [[e1, e2, partner label]]}`; `hidden: [labels]`; `weather[kind] = [labels]`; `tm_table[tm number] = move num`, `tutor_table = [{move, cost, location}]` (from the moves' `machine`/`tutor`/`price`).
  - CLI: `python -m tools.extract_pkcalc [--zip PATH] --out data/intermediate_kaizo/pkcalc`; `ExtractError` when node is missing (naming it).

- [ ] **Step 1: Write failing tests**

```python
def test_eval_has_no_host_globals(tmp_path):  # skips without node
    js = tmp_path / "x.js"; js.write_text("const X = {a: typeof require, b: typeof process, c: typeof setTimeout};")
    assert run_eval(js, "X") == {"a": "undefined", "b": "undefined", "c": "undefined"}
def test_constructor_escape_throws(tmp_path):
    js = tmp_path / "x.js"; js.write_text("const X = ({}).constructor.constructor('return process')();")
    with pytest.raises(ExtractError, match="EvalError|Code generation"):
        run_eval(js, "X")
def test_eval_time_limit(tmp_path):
    js = tmp_path / "x.js"; js.write_text("const X = 1; while (true) {}")
    with pytest.raises(ExtractError, match="timed out"):
        run_eval(js, "X")
def test_zip_slip_refused(tmp_path):   # a zip whose sets.js member is "../sets.js"
    with pytest.raises(ExtractError, match=r"\.\./sets\.js"):
        extract_pkcalc(make_zip(tmp_path, {"../sets.js": "var SETDEX_PK = {};"}), tmp_path / "out")
def test_kaizo_roark_set(pkcalc):     # pkcalc.json from the real zip; skips without it
    roark = pkcalc["trainers"]["Leader Roark"]["party"]
    assert [m["species"] for m in roark] == [438, 345, 443, 222, 213, 408]
    assert roark[0]["nature"] == "Impish" and roark[0]["ivs"] == [30] * 6 and roark[0]["level"] == 15
    assert pkcalc["ai_flags"]["Leader Roark"] == 23
def test_counts(pkcalc):  # only when pkcalc["sha256"] == PKCALC_SHA256
    real = [t for t in pkcalc["trainers"].values() if not t.get("template")]
    assert len(real) == 485 and sum(len(t["party"]) for t in real) == 2112
    assert pkcalc["moves"]["205"]["name"] == "Accelerock" and pkcalc["moves"]["4"]["name"] == "Weather Ball Water"
    assert pkcalc["tm_table"]["7"] == 62     # TM07 Aurora Beam (spec 5.3)
def test_forms(pkcalc):
    assert {(s["num"], s["form"]) for s in pkcalc["species"].values() if s["num"] == 479} == {(479, f) for f in range(6)}
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_extract_pkcalc.py -q`
- [ ] **Step 3: Implement** (extract members to `.cache/pkcalc/<sha256>/`, pass paths as arguments, run `node pkcalc_eval.js` from the repo, never from the extraction folder).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): PKCalc data extractor (data files only, hardened vm context)`

### Task 7: Kaizo docs spreadsheet extractor

**Files:**
- Create: `tools/extract_kaizo_docs.py`
- Test: `tests/kaizo/test_extract_kaizo_docs.py`

**Interfaces:**
- Consumes: `load_inputs().kaizo_docs`; vanilla `data/intermediate/moves.json` (effect ids for the label map) and `abilities.json` (ability names for Ruling 22).
- Produces: `extract_kaizo_docs(xlsx: Path, vanilla_moves: list[dict], vanilla_abilities: list[str], out: Path) -> dict` (openpyxl `read_only=True, data_only=True`), writing `out/docs.json` with every row carrying `"cite": "<sheet>:<1-based row>"`:
  - `effect_labels`: label → vanilla effect id, from `Plat Kaizo PokEditor Tester - M` (the unmodified vanilla export) joined by move id with the vanilla effects; must be one-to-one (257 labels).
  - `moves[id]` from `Moves` (name, label, decoded `effect` or `None` for a new label, category, power, type, accuracy, pp, chance, target, priority, flags); `move_changes` rows from `Move Changes`.
  - `species[id]` from `Personal`; `ability_changes`, `evolutions`, `level_up`, `tm` from their sheets; `items[name]` from `Items` (hold effect, parameter, fling power; joined by name, spec §5.5).
  - `ai_changes` (rows of `AI Changes`, text only).
  - `raw_trainers[id]` from `RAW TRAINER DATA` + `RAW TRAINER POKEMON` (name, class, battle type, AI columns, party with difficulty and Ability Number).
  - `splits`: for each of the ten `* Split` sheets in game order (Roark, Gardenia, Fantina, Maylene, Wake, Byron, Candice, Galactic, Volkner, Elite Four), sections (`name`, tags, `cite`) and trainer headers (`text`, the party's species/levels, AI flag cells incl. "Left Side AI", `cite`, parsed `tags`).
  - Header tag grammar (Ruling 22; case-insensitive; a parenthetical splits on commas, and an item may hold several space-separated tags, as in `OPTIONAL HAIL`, `OPTIONAL DOUBLE BATTLE`):
    - weather: `PERMANENT RAIN|SUN|SANDSTORM|HAIL` → `permanent: <kind>`; bare `RAIN|SUN|HAIL` → `weather: <kind>, assumed_permanent: true`; `FOG` → `fog: true`;
    - field: `PERMANENT TRICK ROOM` → `permanent: trick_room`; bare `TRICK ROOM` → `trick_room: true, assumed_permanent: true`; `PERMANENT GRAVITY`;
    - format: `DOUBLE` / `DOUBLE BATTLE` → `format: double`; `MULTI BATTLE with|With|WITH <name>` → `multi_with: <name>`; `TAG PARTNER` → `tag_partner: true`; `TAG BATTLE with <name>` → `tag_partner: true, multi_with: <name>`;
    - rivals: `[TAG PARTNER[,]] [if] Player chose|picked|selected Turtwig|Chimchar|Piplup` → `player_starter: 387|390|393`;
    - groups: `Gauntlet Start|Gauntlet|Gauntlet End` → `gauntlet: start|middle|end`; `B2B With <name>` → `b2b_with: <name>`; section `GYM TRAINERS ARE A GAUNTLET`; `OPTIONAL` → `optional: true`;
    - notes: an ability name, or an item with no keyword (locations, "Path 2", "Also Appears in Room 5", "SUPERBOSS") → appended to `note`;
    - anything else containing a keyword → `ExtractError("<sheet>:<row>: unparsed tag '<item>'")`.
  - CLI: `python -m tools.extract_kaizo_docs [--xlsx PATH] --out data/intermediate_kaizo/docs`.

- [ ] **Step 1: Write failing tests** (grammar unit tests on strings; real-workbook tests skip without it or openpyxl; counts only when the hash is `KAIZO_DOCS_SHA256`)

```python
@pytest.mark.parametrize("text, tags", [
    ("Barry #3 (Player picked Piplup)", {"player_starter": 393}),
    ("Barry #5 (Player picked Chimchar, on Canalave Bridge)", {"player_starter": 390, "note": "on Canalave Bridge"}),
    ("Dawn (TAG PARTNER if Player chose Piplup)", {"tag_partner": True, "player_starter": 393}),
    ("Grunt (TAG BATTLE with Galactic Uranus)", {"tag_partner": True, "multi_with": "Galactic Uranus"}),
    ("Lass (OPTIONAL HAIL)", {"optional": True, "weather": "hail", "assumed_permanent": True}),
    ("Ace (SUN, DOUBLE BATTLE)", {"weather": "sun", "assumed_permanent": True, "format": "double"}),
    ("Leader Cyrus (Trick Room)", {"trick_room": True, "assumed_permanent": True}),
    ("CYRUS (SHADOW TAG)", {"note": "SHADOW TAG"}),
])
def test_header_tags(text, tags):
    assert parse_header(text, ABILITIES) == tags
def test_keyword_that_does_not_parse_raises():
    with pytest.raises(ExtractError, match="unparsed tag 'PERMANENT FROST'"):
        parse_header("Ace (PERMANENT FROST)", ABILITIES, cite="Candice Split:9")
def test_label_map_is_one_to_one(docs):
    assert len(docs["effect_labels"]) == 257 and len(set(docs["effect_labels"].values())) == 257
def test_label_decode_recovers_effects(docs):
    assert docs["moves"]["209"]["effect"] == 262      # Wild Charge: Volt Tackle's recoil + paralysis
    assert docs["moves"]["149"]["effect"] == 104      # Triple Axel: Triple Kick
def test_split_tags(docs):
    assert section(docs, "Roark", "Oreburgh Gym")["tags"]["permanent"] == "sandstorm"
    hq = section(docs, "Galactic", "Galactic HQ")
    assert hq["tags"]["permanent"] == "trick_room" and len(hq["trainers"]) == 25
    assert section(docs, "Candice", "Snowpoint Gym")["tags"] == {"weather": "hail", "assumed_permanent": True}
def test_gauntlet_and_rival_tags(docs):
    assert tags(docs, "Roark", "Worker Mason")["gauntlet"] == "start"
    assert tags(docs, "Roark", "Barry #2 (Player chose Chimchar)")["player_starter"] == 390
    assert tags(docs, "Maylene", "Barry #3 (Player picked Piplup)")["player_starter"] == 393
def test_raw_roark(docs):
    assert [m["level"] for m in docs["raw_trainers"]["246"]["party"]] == [15, 13, 14, 14, 13, 16]
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_extract_kaizo_docs.py -q`
- [ ] **Step 3: Implement**; run on the real workbook; every `ExtractError` is a grammar gap: extend the grammar (and its parametrized test) rather than the note fallback.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): Kaizo docs spreadsheet extractor`

### Task 8: Kaizo ROM specifics: trainer struct, changed scripts, start conditions

**Files:**
- Modify: K1's `tools/rom/readers/trainers.py` (`layout` keyword: `"vanilla"` | `"kaizo"`), `tools/extract_rom.py` (remove K1's Kaizo `RomInputError` refusal, K1 Ruling 13; `--game kaizo` uses the Kaizo layout, `data/kaizo/start_conditions.json` and writes `changed_scripts.json`), `tools/rom/diff.py` (`load_member_classification`), `tools/extract_decomp.py` (`_start_condition_overrides(path, name_at)`, Ruling 8; vanilla callers pass the `TRAINER_*` list's `__getitem__`)
- Create: `tools/rom/scripts_diff.py`, `data/kaizo/start_conditions.json` (comment + entries Step 3 finds), `[[member]]` entries in `data/kaizo/rom_diff.toml` (appended to K1's file)
- Test: `tests/tools/rom/test_kaizo_rom.py` (fixtures `kaizo_rom_trainers`, `vanilla_rom_trainers`, `kaizo_rom_dir` added to `tests/tools/rom/conftest.py`), `tests/tools/test_extract_decomp.py`

**Interfaces:**
- Consumes: K1 `RomFiles`, `kaizo_rom`, `load_inputs`, `tools.rom.diff.load_classification`.
- Produces:
  - Kaizo party mons in `rom/trainers.json` carry `"difficulty"` (low byte) and `"ability_override"` (high byte) per Ruling 6; vanilla output unchanged (`ability_override` absent).
  - `changed_members(vanilla: RomFiles, patched: RomFiles, path: str) -> list[int]` (member index whose bytes differ, plus members only one side has).
  - `rom/changed_scripts.json`: `{"battle/skill/be_seq.narc": [...], "battle/skill/sub_seq.narc": [...], "battle/tr_ai/tr_ai_seq.narc": [...]}`.
  - `load_member_classification(path: Path, changed: dict[str, list[int]]) -> dict[tuple[str, int], tuple[str, str]]` ((narc, index) → (status, note)); K1's rules: an unknown status raises, an entry naming an unchanged member raises as stale, a changed member with no entry reads as `"unknown"`. `extract_rom --game kaizo` appends an explicit `unknown` `[[member]]` entry for each missing one and says so. K1's `load_classification` ignores `[[member]]` tables.
  - `data/kaizo/start_conditions.json` entries `{"<id>": {"trainer": "<Kaizo ROM name>", "condition": ...}}`, validated by `_start_condition_overrides(path, name_at=rom_trainer_names.__getitem__)`; `name_at` raising `KeyError`/`IndexError` means "not a trainer id".

- [ ] **Step 1: Write failing tests** (ROM tests skip without inputs/ndspy)

```python
def test_kaizo_roark_struct(kaizo_rom_trainers, docs):
    roark = kaizo_rom_trainers[246]["party"]
    assert [m["species"] for m in roark] == [438, 345, 443, 222, 213, 408]
    assert {m["difficulty"] * 31 // 255 for m in roark} == {30}
    raw = docs["raw_trainers"]["246"]["party"]
    assert [(m["difficulty"], m["ability_override"]) for m in roark] == [(r["difficulty"], r["ability_number"]) for r in raw]
def test_override_byte_values(kaizo_rom_trainers):
    mons = [m for t in kaizo_rom_trainers for m in t["party"]]
    assert {m["ability_override"] for m in mons} <= {0, 2, 16, 32}
    assert 100 <= sum(m["ability_override"] != 0 for m in mons) <= 122   # spec 5.4: 111 on the sheet
def test_vanilla_layout_unchanged(vanilla_rom_trainers, decomp_trainers):
    assert vanilla_rom_trainers == decomp_trainers
def test_changed_members_synthetic():
    a, b = FakeRom({"x.narc": [b"0", b"1", b"2", b"3"]}), FakeRom({"x.narc": [b"0", b"9", b"2", b"8", b"4"]})
    assert changed_members(a, b, "x.narc") == [1, 3, 4]
def test_member_classification_rules(tmp_path):
    p = write(tmp_path, '[[member]]\nnarc = "battle/skill/be_seq.narc"\nindex = 7\nstatus = "modelled"\nnote = "n"\n')
    got = load_member_classification(p, {"battle/skill/be_seq.narc": [7, 9]})
    assert got == {("battle/skill/be_seq.narc", 7): ("modelled", "n"), ("battle/skill/be_seq.narc", 9): ("unknown", "")}
    with pytest.raises(ValueError, match="be_seq.narc#7.*stale"):
        load_member_classification(p, {"battle/skill/be_seq.narc": [9]})
def test_every_changed_script_classified(kaizo_rom_dir):
    changed = json.loads((kaizo_rom_dir / "changed_scripts.json").read_text("utf-8"))
    entries = load_member_classification(Path("data/kaizo/rom_diff.toml"), changed)
    assert set(entries) == {(narc, i) for narc, ids in changed.items() for i in ids}
def test_start_condition_names_checked(tmp_path):   # tests/tools/test_extract_decomp.py
    p = write(tmp_path, '{"246": {"trainer": "Byron", "condition": "none"}}')
    with pytest.raises(ExtractError, match="id 246 is Roark, not Byron"):
        _start_condition_overrides(p, {246: "Roark"}.__getitem__)
    with pytest.raises(ExtractError, match="not a trainer id"):
        _start_condition_overrides(write(tmp_path, '{"999": {"trainer": "Roark", "condition": "none"}}'), {246: "Roark"}.__getitem__)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_kaizo_rom.py tests/tools/test_extract_decomp.py -q`
- [ ] **Step 3: Implement**; run `python -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom`; for every trainer the extractor reports as date-ambiguous, add an entry to `data/kaizo/start_conditions.json` citing the sheet header and PKCalc `flags.js` line, then rerun until clean.
- [ ] **Step 4: Run, expect PASS**, then `tests/tools -q` (vanilla extraction unchanged).
- [ ] **Step 5: Commit** `feat(rom): Kaizo trainer struct, changed battle scripts, Kaizo start conditions`

### Task 9: PKCalc trainer → ROM id mapping

**Files:**
- Create: `tools/kaizo/trainer_map.py`
- Test: `tests/kaizo/test_trainer_map.py`

**Interfaces:**
- Consumes: Task 6 `pkcalc["trainers"]`, Task 8 `rom/trainers.json`, `nuzline.data.trainer.trainer_pid`, `gender_of`, species gender ratios; Task 10's `overrides.trainer_ids` (label → id, applied first).
- Produces: `@dataclass(frozen=True) Mapping(label: str, id: int | None, how: str, candidates: tuple[int, ...])` (`how`: `override`, `exact`, `pid`, `name`, `unmapped`); `map_trainers(pkcalc_trainers, rom_trainers, species, overrides) -> list[Mapping]`; `class MappingError(ValueError)`.
  Algorithm: signature = `((species, form, level), ...)` in party order; candidates = ROM ids with an equal signature; one → `exact`; several → keep the candidates where the vanilla PID (difficulty = low byte, override-0 mons only) reproduces the most PKCalc natures and genders, one left → `pid`; still several → the candidate whose ROM name is contained in the label → `name`; else `unmapped`. Templates are skipped. Two labels on one id raise `MappingError` listing both, unless an override names that id for one of them.

- [ ] **Step 1: Write failing tests** (synthetic: `party(sig)` builds PKCalc and ROM parties from `((species, form, level), ...)`; `rom_trainer(id, name, sig, difficulty)`)

```python
P = ((74, 0, 12), (95, 0, 14))
def test_exact():
    m = map_trainers({"Leader X": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("Leader X", 5, "exact", (5,))]
def test_pid_tiebreak():   # ids 6 and 7 share P; only id 7's difficulty gives PKCalc's natures
    m = map_trainers({"Hiker Y": pk(P, natures=natures_for(7))}, [rom_trainer(6, "Y", P, 0), rom_trainer(7, "Y", P, 200)], SPECIES, {})
    assert (m[0].id, m[0].how, m[0].candidates) == (7, "pid", (6, 7))
def test_unmapped_is_reported_not_guessed():
    m = map_trainers({"Lady Emi": pk(((1, 0, 5),))}, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("Lady Emi", None, "unmapped", ())]
def test_two_labels_one_id_raises():
    with pytest.raises(MappingError, match="A.*B"):
        map_trainers({"A": pk(P), "B": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {})
def test_override_breaks_the_tie():
    m = {x.label: x for x in map_trainers({"A": pk(P), "B": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {"B": 5})}
    assert (m["B"].id, m["B"].how) == (5, "override") and m["A"].id is None
def test_real_mapping(pkcalc, kaizo_rom_trainers, species):  # skips without inputs
    m = {x.label: x for x in map_trainers(pkcalc["trainers"], kaizo_rom_trainers, species, load_overrides().trainer_ids)}
    assert m["Leader Roark"].id == 246
    assert len(m) == 485 and sum(x.id is not None for x in m.values()) >= 473
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_trainer_map.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): map PKCalc trainers to ROM ids by party order with the PID tie-break`

### Task 10: Overrides file: schema, draft and citation check

**Files:**
- Create: `tools/kaizo/overrides.py`, `data/kaizo/overrides.toml` (Step 3)
- Test: `tests/kaizo/test_overrides.py`

**Interfaces:**
- Consumes: Task 7 `docs.json` (cites), Task 9 mappings.
- Produces: `load_overrides(path: Path = REPO_ROOT / "data/kaizo/overrides.toml") -> Overrides` (frozen dataclass: `area: dict[int, AreaOverride]`, `gauntlets: tuple[Gauntlet, ...]`, `trainer_ids: dict[str, int]`, `fields: dict[int, dict[str, object]]` (corrections to derived fields: split, optional, left_side, player_starter), `move_lists: dict[str, list[int]]`); `AreaOverride(trick_room: bool, gravity: bool, weather: str | None, source)`; `Gauntlet(name, trainers: tuple[int, ...], rules: dict[str, bool], source)` with `rules` keys only `heal_between`, `change_team`; every entry needs `source` (`"<sheet>:<row>"`, `"pkcalc:<file>:<line>"`, `"rom:<what>"` or `"user, <date>"`), else `OverridesError`.
  `check_citations(overrides, docs) -> list[str]`: each sheet citation must still be a row whose text supports the entry (the trainer header or its section carries the claimed tag); `draft_overrides(docs, mappings) -> str`: TOML for every Trick Room trainer, every gauntlet group (Ruling 9) and Cynthia's Gravity (`source = "pkcalc:src/js/shared_controls.js:611"`), with conflicts (one trainer claimed by two areas) emitted commented out with both citations.
  CLI: `python -m tools.kaizo.overrides draft` (writes `.cache/kaizo/overrides.draft.toml`, git-ignored with `.cache/`, and prints the path) and `python -m tools.kaizo.overrides check`.

- [ ] **Step 1: Write failing tests**

```python
def test_entry_without_source_refused(tmp_path):
    with pytest.raises(OverridesError, match="area.246.*source"):
        load_overrides(write(tmp_path, "[area.246]\ntrick_room = true\n"))
def test_unknown_rule_key_refused(tmp_path):
    with pytest.raises(OverridesError, match="'heal'"):
        load_overrides(write(tmp_path, '[[gauntlet]]\nname = "g"\ntrainers = [1, 2]\nrules = {heal = true}\nsource = "user, 2026-11-03"\n'))
def test_rules_left_unset_by_draft(docs, mappings):
    assert "rules = {}" in draft_overrides(docs, mappings)
def test_stale_citation_detected(docs):   # entry cites Roark Split:7, whose header has no TRICK ROOM tag
    o = overrides_with(area={7: AreaOverride(True, False, None, "Roark Split:7")})
    assert check_citations(o, docs) == ["area.7: Roark Split:7 does not say Trick Room"]
def test_committed_file_checks_clean(docs):      # skips without the workbook
    assert check_citations(load_overrides(), docs) == []
def test_trick_room_and_gravity_present():
    o = load_overrides()
    assert sum(a.trick_room for a in o.area.values()) >= 51 and any(a.gravity for a in o.area.values())
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_overrides.py -q`
- [ ] **Step 3: Implement**; generate the draft, settle the commented conflicts from the ROM trainer → map join (Galactic Rosalind), add the Sinister Hooded Figure's Trick Room (sheet), write `data/kaizo/overrides.toml` with a header comment stating Rulings 8-9, run `check`.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): committed overrides with citations (Trick Room, Gravity, gauntlets)`

### Task 11: Cross-check engine, resolutions policy and report writer

**Files:**
- Create: `tools/kaizo/crosscheck.py` (generic), `data/kaizo/resolutions.toml` (header comment only)
- Test: `tests/kaizo/test_crosscheck.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Row(table: str, key: str, field: str, rom, pkcalc, sheet, executed: bool)` (`None` = the source has no value).
  - `@dataclass(frozen=True) Resolution(table, keys: tuple[str, ...], field, pkcalc, use: str, reason: str, evidence: str)`; `load_resolutions(path) -> list[Resolution]` (Ruling 12; `use = "pkcalc"` without `evidence` starting `trace:` → `CrossCheckError`).
  - `@dataclass(frozen=True) Outcome(row: Row, kind: str, used, resolution: Resolution | None)`, `kind` in `agree`, `sheet_only`, `rom_vs_sheet`, `unexecuted`, `resolved`, `unresolved`.
  - `@dataclass CrossCheck(outcomes: list[Outcome], unresolved: list[Row], stale: list[Resolution])`; `compare(rows: Iterable[Row], resolutions: Sequence[Resolution]) -> CrossCheck`; `CrossCheck.require_clean()` raises `CrossCheckError` listing every unresolved row and stale entry.
  - `write_report(check: CrossCheck, sections: dict[str, list[str]], sources: dict[str, str], path: Path) -> None`: `docs/kaizo/discrepancies.md` with the source hashes, a count table per table × kind, then one table per kind except `agree` (`| table | key | field | ROM | PKCalc | sheet | used | why |`), then the extra `sections` (mapping, PID cross-check, area conditions); rows sorted; byte-identical on reruns.

Policy (spec §6.5): ROM = PKCalc → `agree` (sheet equal or absent) or `sheet_only`; ROM ≠ PKCalc and `executed` → `resolved` with the matching resolution's `use` (default `rom`), else `unresolved`; not executed → `unexecuted`, used = PKCalc (names, descriptions); PKCalc `None` and ROM ≠ sheet → `rom_vs_sheet`, ROM used.

- [ ] **Step 1: Write failing tests**

```python
def test_executed_conflict_needs_resolution():
    check = compare([Row("moves", "14 Swords Dance", "pp", 20, 2, 30, True)], [])
    assert [r.key for r in check.unresolved] == ["14 Swords Dance"]
    with pytest.raises(CrossCheckError, match="Swords Dance"):
        check.require_clean()
def test_grouped_resolution_defaults_to_rom():
    res = Resolution("moves", ("1 A", "2 B"), "power", 5, "rom", "PKCalc rounding", "rom_diff.md")
    check = compare([Row("moves", "1 A", "power", 4, 5, None, True), Row("moves", "2 B", "power", 4, 5, None, True)], [res])
    assert {o.used for o in check.outcomes} == {4} and not check.unresolved
def test_stale_when_pkcalc_value_changed():
    res = Resolution("moves", ("1 A",), "power", 5, "rom", "r", "e")
    assert compare([Row("moves", "1 A", "power", 4, 6, None, True)], [res]).stale == [res]
def test_stale_when_no_longer_disagreeing():
    res = Resolution("moves", ("1 A",), "power", 5, "rom", "r", "e")
    check = compare([Row("moves", "1 A", "power", 5, 5, None, True)], [res])
    assert check.stale == [res] and check.outcomes[0].kind == "agree"
    with pytest.raises(CrossCheckError, match="stale.*1 A"):
        check.require_clean()
def test_pkcalc_use_needs_trace(tmp_path):
    p = write(tmp_path, '[[resolution]]\ntable = "moves"\nkeys = ["1 A"]\nfield = "power"\npkcalc = 5\n'
                        'use = "pkcalc"\nreason = "r"\nevidence = "Moves:15"\n')
    with pytest.raises(CrossCheckError, match="trace:"):
        load_resolutions(p)
def test_sheet_only_needs_no_entry():
    assert compare([Row("moves", "1 A", "pp", 10, 10, 15, True)], []).outcomes[0].kind == "sheet_only"
def test_report_is_deterministic(tmp_path):
    check = compare([Row("moves", "14 Swords Dance", "pp", 20, 20, 30, True)], [])
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    for p in (a, b):
        write_report(check, {"mapping": ["x"]}, {"pkcalc": PKCALC_SHA256, "kaizo_docs": KAIZO_DOCS_SHA256}, p)
    text = a.read_text("utf-8")
    assert a.read_bytes() == b.read_bytes() and PKCALC_SHA256 in text and KAIZO_DOCS_SHA256 in text
    assert "| moves | 14 Swords Dance | pp | 20 | 20 | 30 | 20 |" in text
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/kaizo/test_crosscheck.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): three-way cross-check with the resolutions policy`

### Task 12: Cross-check rows for moves, species, learnsets, TM/tutor tables and items

**Files:**
- Create: `tools/kaizo/rows_tables.py`
- Test: `tests/kaizo/test_rows_tables.py`

**Interfaces:**
- Consumes: `rom/` intermediates (K1/K2a schemas, incl. K2a's TM→move and tutor tables), Task 6 `pkcalc.json`, Task 7 `docs.json`.
- Produces: `move_rows(rom_moves, pkcalc, docs) -> list[Row]` (executed: power, accuracy, pp, priority, type, category, target, chance, effect (ROM vs sheet label decode; PKCalc has no id); unexecuted: name); `species_rows(...)` (executed: base stats, types, abilities, exp yield, growth rate, gender ratio; unexecuted: catch rate, name; per personal index incl. forms, Ruling 4); `learnset_rows(...)` (executed: level-up list, TM set, tutor set; `evolutions` vs sheet); `tm_rows(...)` (`tm_table`, key `"TM07"`: the move, executed, ROM vs PKCalc `machine` vs the sheet's TM sheet); `tutor_rows(...)` (`tutor_table`, key = tutor slot: move executed; cost and location unexecuted, ROM vs PKCalc `price`/location); `item_rows(...)` (executed: hold effect, hold param, fling power — ROM vs sheet only; unexecuted: name); `ability_rows(...)` (unexecuted: name, all 123 expected equal). Keys are `"<id> <PKCalc name>"`. PKCalc target strings map `normal`→`RANGE_SINGLE_TARGET`, `self`→`RANGE_USER`, `allAdjacentFoes`→`RANGE_ADJACENT_OPPONENTS`, `allAdjacent`→`RANGE_ALL_ADJACENT`, `randomNormal`→`RANGE_RANDOM_OPPONENT`, `all`→`RANGE_FIELD`, `allySide`→`RANGE_USER_SIDE`, `foeSide`→`RANGE_OPPONENT_SIDE`, `scripted`→`RANGE_SINGLE_TARGET_SPECIAL` (`ids.MoveTarget`); an unmapped string raises. Priorities outside −7..+7 raise (spec §5.2).

- [ ] **Step 1: Write failing tests** (synthetic three-source fixtures per function, plus real-input checks that skip)

```python
def test_trick_room_priority_row(real_rows):
    row = find(real_rows, "moves", "433 Trick Room", "priority")
    assert (row.rom, row.pkcalc) == (7, 7)
def test_known_sheet_conflicts_listed(real_check):  # spec 5.2: Swords Dance PP sheet 30 vs PKCalc 2
    assert any(o.row.key.endswith("Swords Dance") and o.row.field == "pp" for o in real_check.outcomes if o.kind != "agree")
def test_forms_compared(real_rows):  # Rotom-Heat is personal index 503 (Ruling 4)
    row = next(r for r in real_rows if r.table == "species" and r.key.startswith("503 ") and r.field == "base_stats")
    assert row.rom[Stat.SPE] == 91 and row.rom == row.pkcalc
def test_tm07_is_aurora_beam(real_rows):
    row = find(real_rows, "tm_table", "TM07", "move")
    assert (row.rom, row.pkcalc, row.executed) == (62, 62, True)
def test_tutor_cost_unexecuted(synthetic_sources):
    rows = tutor_rows(*synthetic_sources)
    assert {(r.field, r.executed) for r in rows} == {("move", True), ("cost", False), ("location", False)}
def test_unknown_target_raises():
    with pytest.raises(ValueError, match="adjacentAllyOrSelf"):
        move_rows(rom_moves(1), pkcalc_moves(1, target="adjacentAllyOrSelf"), empty_docs())
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_rows_tables.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): cross-check rows for moves, species, learnsets, TM/tutor tables and items`

### Task 13: Cross-check rows for trainers, AI flags, formats and areas; PID cross-check

**Files:**
- Create: `tools/kaizo/rows_trainers.py`
- Test: `tests/kaizo/test_rows_trainers.py`

**Interfaces:**
- Consumes: Task 8 `rom/trainers.json` (with K2a's `format`, `opponent_2`, `partner`), Task 9 mappings, Task 10 overrides, Task 6/7 outputs, `nuzline.data.trainer` (`trainer_pid`, `gender_of`, `default_moves`, `difficulty_to_ivs`).
- Produces:
  - `trainer_rows(rom, mappings, pkcalc, docs) -> list[Row]`: per mapped trainer, executed `party_size`, and per mon (`key = "<id> <label>/<slot>"`) `species`, `level`, `item`, `moves` (ROM's, or `default_moves` from the Kaizo level-up list), `ivs` (ROM `difficulty_to_ivs` vs PKCalc), and `ability_slot` where `ability_override` forces one; `difficulty` and `ability_override` (ROM vs the sheet's RAW rows; PKCalc silent → `rom_vs_sheet` where they differ, Ruling 6); `ai_flags` (bits 0-10 ROM vs PKCalc vs the split's AI cells; higher ROM bits reported in a section); per Ruling 10 `format`, `opponent_2` and `partner_present` (ROM from K2a's fields vs PKCalc `formats` vs sheet tags, all executed); `player_starter` (sheet tag vs PKCalc bracket, Ruling 11; ROM `None`, so a disagreement is resolved by an overrides `fields` entry); `area_weather` (ROM `start_condition` vs sheet tag vs PKCalc weather list, a PKCalc entry explained by a lead's or partner's weather ability dropped first).
  - `mapping_section(mappings) -> list[str]` (every non-`exact` mapping, every unmapped label and its candidates).
  - `pid_section(rom, mappings, pkcalc, species) -> list[str]`: the vanilla routine's nature/gender/ability per override-0 mon vs PKCalc, as the lines `"nature and gender reproduced for N of M override-0 Pokemon"`, `"ability reproduced for N of M"`, `"trainers reproduced completely: N of M"`, `"Leader Roark (246): 6/6"`, then every miss; override mons listed as `"not checked: override byte N"`. The counts are recomputed on ROM inputs and recorded (Ruling 19).
  - `area_section(...)`: format/partner/opponent_2 and Trick Room/Gravity claims by source.
  An unmapped PKCalc label is itself an executed `trainers/<label>/mapped` row (ROM `None`, PKCalc set) that needs an overrides `trainer_ids` entry or a resolution.

- [ ] **Step 1: Write failing tests**

```python
def test_roark_rows_agree(real_rows):
    assert not [r for r in real_rows if r.key.startswith("246 ") and r.executed and r.rom != r.pkcalc]
def test_pid_section_format(real_sections):
    lines = real_sections["pid"]
    assert "Leader Roark (246): 6/6" in lines
    assert re.fullmatch(r"nature and gender reproduced for \d+ of \d+ override-0 Pokemon", lines[0])
def test_ability_weather_not_an_area_claim():  # PKCalc's sun list holds a trainer whose lead has Drought
    rows = trainer_rows(*synthetic(lead_ability="Drought", pkcalc_weather={"sun": ["Ace X"]}, rom_condition="none"))
    row = find(rows, "trainers", "5 Ace X", "area_weather")
    assert (row.rom, row.pkcalc, row.sheet) == ("none", None, None)
def test_unmapped_label_row_is_executed():
    rows = trainer_rows(*synthetic(mappings=[Mapping("Lady Emi", None, "unmapped", ())]))
    row = find(rows, "trainers", "Lady Emi", "mapped")
    assert row.executed and row.rom is None
    assert [r.key for r in compare(rows, []).unresolved] == ["Lady Emi"]
def test_ai_flag_bits():
    row = find(trainer_rows(*synthetic(rom_ai=23, pkcalc_ai=7)), "trainers", "5 Ace X", "ai_flags")
    assert (row.rom, row.pkcalc, row.executed) == (23, 7, True)
def test_format_rows_from_rom_structure():  # ROM multi with opponent_2 9; PKCalc tag pair ["Ace X", "Ace Y"]
    rows = trainer_rows(*synthetic(rom_format="multi", rom_opponent_2=9, pkcalc_tag=[["Ace X", "Ace Y", "{RIVAL2_NAME}"]]))
    assert find(rows, "trainers", "5 Ace X", "format").pkcalc == "multi"
    assert find(rows, "trainers", "5 Ace X", "opponent_2").pkcalc == 9          # "Ace Y" mapped to 9
    assert find(rows, "trainers", "5 Ace X", "partner_present").pkcalc is True
def test_sheet_difficulty_is_rom_vs_sheet():
    out = compare(trainer_rows(*synthetic(rom_difficulty=200, sheet_difficulty=180)), []).outcomes
    assert next(o for o in out if o.row.field == "difficulty").kind == "rom_vs_sheet"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_rows_trainers.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(kaizo): trainer, AI flag, format and area cross-check; vanilla PID report`

### Task 14: Trainer fields at runtime, explicit natures, coverage refusals

**Files:**
- Modify: `src/nuzline/data/trainer.py` (`Trainer`, `TrainerMon`, `load_trainers`, `build_trainer_party`), `src/nuzline/engine/coverage.py` (`_check_trainer`, `check_opponent`)
- Test: `tests/data/test_trainer.py`, `tests/engine/test_coverage.py`

**Interfaces:**
- Consumes: K2a's `Trainer.format`, `opponent_2`, `partner: tuple[PartnerVariant, ...]` and its format refusal (unchanged; Ruling 10).
- Produces (the `trainers.json` schema Task 15 writes):
  - `TrainerMon` gains `nature: int | None = None`, `gender: int | None = None`, `ability_slot: int | None = None`, `ability_override: int = 0`.
  - `Trainer` gains (all defaulted, so vanilla JSON loads unchanged): `label: str | None = None`, `pid_routine: str | None = "vanilla"`, `player_starter: int | None = None`, `gauntlet: tuple[str, int] | None = None` (group, position), `gauntlet_rules: dict[str, bool]` (empty), `optional: bool = False`, `left_side: bool = False`, `area_condition: AreaCondition`, `split: str | None = None`, `split_order: tuple[int, int] | None = None`, `sources: dict[str, str]` (empty).
  - `@dataclass(frozen=True) AreaCondition(weather: str | None = None, fog: bool = False, trick_room: bool = False, gravity: bool = False, permanent: bool = True)`; when the JSON has no `area_condition` it is built from `start_condition`.
  - `build_trainer_party`: explicit values win; with `pid_routine is None` a mon missing any of the three raises `ValueError("trainer N slot k: no nature/gender/ability data")`.
  - Coverage lines (each `Support.NONE`, rendered by `_Collector` as `"<what> (<kind> UNSUPPORTED)"`): `trick_room` → `"Battle start: permanent Trick Room"` (`field condition`); `gravity` → `"Battle start: permanent Gravity"` (`field condition`); `gauntlet` → `"{trainer}: gauntlet {group} (position {p}; rules {rules or 'not recorded'})"` (`battle format`); a mon without nature data → `"{trainer} slot k: nature/gender/ability unknown (no PKCalc set)"` (`trainer data`); `names.game` Kaizo with `unclassified_rom_changes > 0` → `"Kaizo ROM: N code changes unclassified (docs/kaizo/rom_diff.md)"` (`ROM change`). Area weather and fog keep today's refusal through `start_weather` (`"Battle start: Sandstorm (weather UNSUPPORTED)"`); `optional` refuses nothing.

- [ ] **Step 1: Write failing tests**

```python
def test_vanilla_trainers_load_with_defaults(trainers):   # data/intermediate trainers.json
    assert all(t.label is None and t.pid_routine == "vanilla" and not t.optional and t.gauntlet is None
               and t.sources == {} for t in trainers)
    assert trainers[246].area_condition == AreaCondition()      # vanilla Roark starts in no weather
def test_explicit_nature_wins(inter):
    t = replace(roark_trainer(), party=(replace(roark_trainer().party[0], nature=3, gender=1, ability_slot=1),))
    assert build_trainer_party(t, inter)[0].nature == 3
def test_no_routine_without_data_raises(inter):
    t = replace(roark_trainer(), pid_routine=None)
    with pytest.raises(ValueError, match="trainer 246 slot 0: no nature/gender/ability data"):
        build_trainer_party(t, inter)
def test_trick_room_refused(tables, names):
    report = check_opponent(tables, names, with_area(roark_trainer(), trick_room=True), roark_party(), 0)
    assert "Battle start: permanent Trick Room (field condition UNSUPPORTED)" in report.missing
def test_gauntlet_refused_rules_not_recorded(tables, names):
    t = replace(roark_trainer(), gauntlet=("Oreburgh Gym", 2))
    report = check_opponent(tables, names, t, roark_party(), 0)
    assert "Roark: gauntlet Oreburgh Gym (position 2; rules not recorded) (battle format UNSUPPORTED)" in report.missing
def test_optional_not_refused(tables, names):
    plain = check_opponent(tables, names, roark_trainer(), roark_party(), 0)
    assert check_opponent(tables, names, replace(roark_trainer(), optional=True), roark_party(), 0).missing == plain.missing
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/data/test_trainer.py tests/engine/test_coverage.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**, then `tests/engine tests/ai tests/prep -q` in the background.
- [ ] **Step 5: Commit** `feat(data): Kaizo trainer fields; refuse permanent field effects and gauntlets`

### Task 15: Resolve and build the Kaizo tables

**Files:**
- Create: `tools/kaizo/resolve.py`
- Modify: `tools/build_tables.py` (`--game kaizo` → `resolve`, then pack into `data/gen4_platinum_kaizo`; Kaizo manifest keys; `aliases` from `kaizo.json`; item map by name and hold effect; changed `be_seq` members → `UNSUPPORTED` mechanic; chance-0 secondaries per Ruling 21), `tools/effect_map.py` (comment on the chance-0 rule beside `CERTAIN_SECONDARY_EFFECTS`), `data/kaizo/resolutions.toml` (entries Step 3 writes)
- Test: `tests/kaizo/test_resolve.py`, `tests/kaizo/test_kaizo_tables.py`, `tests/tools/test_build_tables_game.py`

**Interfaces:**
- Consumes: Tasks 5-14; K1 `load_inputs`, `kaizo_rom`; K2a's `renamed.json` reader.
- Produces:
  - `resolve(inputs: RomInputs, inter: Path = get_game("kaizo").intermediate_dir, report: Path = REPO_ROOT / "docs/kaizo/discrepancies.md") -> Path`: runs the cross-check (`require_clean`, `report` written either way), writes resolved `inter/*.json` in the vanilla schemas (values per `Outcome.used`; names per Ruling 3; `trainers.json` in Task 14's schema with every new field and a `sources` map per field, and `start_condition` set from the resolved area weather or fog so `start_weather` keeps working), `inter/renamed.json` in K2a's format (Ruling 3), and `inter/kaizo.json` (`aliases`, `split_order`, provenance).
  - Kaizo manifest adds: `game`, `vanilla_rom_sha1`, `patch_sha256`, `patched_rom_sha1`, `pkcalc_sha256`, `kaizo_docs_sha256`, `decomp_commit`, `overrides_sha256`, `resolutions_sha256`, `rom_diff_sha256`, `aliases`, `changed_effects` (effect ids made `UNSUPPORTED` by Ruling 13), `no_secondary_moves` (Ruling 21), `unclassified_rom_changes` (count of `"unknown"` `[[code]]` and `[[member]]` entries); K2a's `renamed` comes from `renamed.json`.
  - Item effect: an item maps to its `ItemId`/`TrainerItemId` only when its vanilla name's effect_map entry exists **and** its hold effect equals vanilla's for that name; a renamed or re-purposed item is `UNSUPPORTED` (Shoal Salt → Berserk Gene, Lava Cookie as a Lum clone).
  - CLI: `python -m tools.build_tables --game kaizo` (writes `docs/kaizo/discrepancies.md`; exit 1 on unresolved or stale rows).

- [ ] **Step 1: Write failing tests**

```python
def test_vanilla_build_unchanged(tmp_path, tables):
    build(GAMES["platinum_us"].intermediate_dir, tmp_path, game="platinum_us")
    new, old = np.load(tmp_path / "tables.npz"), np.load(TABLES_DIR / "tables.npz")
    assert new.files == old.files and all(np.array_equal(new[k], old[k]) for k in old.files)
def test_chance0_secondary_is_no_secondary():   # tests/tools/test_build_tables_game.py
    assert _move_effect(move(effect=5, chance=0)) == (EffectId.PLAIN, (0, 0, 0))       # FREEZE_HIT
    assert _move_effect(move(effect=125, chance=0))[0] == EffectId.UNSUPPORTED          # thaws_user param
    assert _move_effect(move(effect=5, chance=10))[0] == EffectId.SECONDARY_STATUS
    with pytest.raises(BuildError):
        _move_effect(move(effect=218, chance=30))                                       # certain, chance != 0
def test_unresolved_stops_the_build(fake_inputs, tmp_path):   # one executed conflict, empty resolutions
    with pytest.raises(CrossCheckError, match="unresolved"):
        resolve(fake_inputs, tmp_path, report=tmp_path / "report.md")
    assert "| unresolved |" in (tmp_path / "report.md").read_text("utf-8")   # written either way
def test_renamed_json_in_k2a_format(kaizo_intermediate):
    renamed = json.loads((kaizo_intermediate / "renamed.json").read_text("utf-8"))
    assert renamed["move"]["rollout"] == "Accelerock" and set(renamed) == {"species", "move", "item", "ability"}
def test_kaizo_tables(kaizo_tables, kaizo_names):              # real build; skips when absent
    assert kaizo_names.game == "platinum_kaizo_nightmare_final"
    assert kaizo_names.move_names[205] == "Accelerock" and kaizo_names.move("Water Ball") == 4
    assert int(kaizo_tables.move_priority[kaizo_names.move("Trick Room")]) == 7
    assert int(kaizo_tables.item_effect[kaizo_names.item("Berserk Gene")]) == ItemId.UNSUPPORTED
def test_kaizo_manifest_provenance(kaizo_manifest):
    assert kaizo_manifest["vanilla_rom_sha1"] == VANILLA_SHA1 and kaizo_manifest["patch_sha256"] == KAIZO_PATCH_SHA256
def test_changed_effect_unsupported(kaizo_tables, kaizo_manifest):
    for m in moves_with_effects(kaizo_manifest["changed_effects"]):
        assert int(kaizo_tables.move_effect[m]) == EffectId.UNSUPPORTED
def test_priority_consumers_take_plus_seven(tables):
    # engine/order.py:78 and engine/enumerate.py:228 cast move_priority to int32 (re-check lines at M4)
    t = with_priorities(tables, {TACKLE: 7, QUICK_ATTACK: 5})        # tests/engine/test_order.py helpers
    assert first_mover(t, slow_user(TACKLE), fast_user(QUICK_ATTACK)) == "slow_user"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/kaizo/test_resolve.py tests/kaizo/test_kaizo_tables.py tests/tools/test_build_tables_game.py -q`
- [ ] **Step 3: Implement**, then build for real: run Task 6-8's extractors, `python -m tools.build_tables --game kaizo`; for each unresolved row, decide from the ROM (default) and write a `[[resolution]]` (grouped where one cause explains many rows: a PKCalc naming convention, decoy effect ids, a spreadsheet template bug), citing the evidence; rerun until it builds. Record the resolution count by table and the PID section's counts in the commit message.
- [ ] **Step 4: Run, expect PASS**, then `tests/tools -q` (vanilla build still byte-identical).
- [ ] **Step 5: Commit** `feat(kaizo): build the Kaizo tables from the resolved sources` (with `data/kaizo/resolutions.toml` and `docs/kaizo/discrepancies.md`)

### Task 16: Level caps per game and the next fight's cap

**Files:**
- Create: `data/kaizo/level_caps.toml`
- Modify: `src/nuzline/prep/caps.py` (M4 file)
- Test: `tests/prep/test_caps.py`

**Interfaces:**
- Consumes: Task 14 `Trainer.split`, `split_order`, `label`, `optional`, `player_starter`.
- Produces: `CAPS_FILES = {"platinum_us": config/level_caps.toml, "platinum_kaizo_nightmare_final": data/kaizo/level_caps.toml}`; `default_cap(trainer: Trainer, path: Path | None = None, game: str = "platinum_us") -> int` (`path` keeps M4's meaning and overrides `CAPS_FILES`; vanilla path unchanged; Kaizo: a `[fights]` entry whose key prefixes `trainer.label`, else `[splits][trainer.split]`, else `ValueError` naming the file and `--cap`); `next_fight_cap(trainer, trainers: Sequence[Trainer], game: str, player_starter: int | None, path: Path | None = None) -> int | None` (the cap of the first later non-optional trainer in `split_order` whose `player_starter` is `None` or equal; `None` after the last fight).
  `data/kaizo/level_caps.toml` (user, 2026-11-03, spec §5.12): `[splits]` Roark 16, Gardenia 28, Fantina 38, Maylene 47, Wake 54, Byron 65, Candice 74, Galactic 100, Volkner 84, "Elite Four" 100 (Ruling 15 comment); `[fights]` `"Pokémon Trainer Barry #3" = 42`, `"Pokémon Trainer Barry #5" = 63`.

- [ ] **Step 1: Write failing tests** (`kt(label, split, order, optional=False, starter=None)` builds a synthetic Kaizo `Trainer`)

```python
KAIZO = "platinum_kaizo_nightmare_final"
def test_vanilla_unchanged():
    assert load_level_caps() == {246: ("Roark", 16)} and default_cap(roark_trainer()) == 16
def test_kaizo_split_cap():
    assert default_cap(kt("Ace Trainer X", "Fantina", (2, 4)), game=KAIZO) == 38
def test_kaizo_barry_fight_cap():
    assert default_cap(kt("Pokémon Trainer Barry #3 [Empoleon]", "Maylene", (3, 0)), game=KAIZO) == 42
def test_kaizo_no_split_errors():
    with pytest.raises(ValueError, match="data/kaizo/level_caps.toml.*--cap"):
        default_cap(kt("Lady Emi", None, None), game=KAIZO)
def test_next_fight_cap_crosses_split(tmp_path):
    caps = write(tmp_path, '[splits]\nRoark = 16\nGardenia = 28\n[fights]\n"Rival [Torterra]" = 20\n')
    a = kt("Leader Roark", "Roark", (0, 9)); b = kt("Lass B", "Gardenia", (1, 0), optional=True)
    c = kt("Rival [Torterra]", "Gardenia", (1, 1), starter=393); d = kt("Rival [Infernape]", "Gardenia", (1, 2), starter=387)
    ts = [a, b, c, d]
    assert next_fight_cap(a, ts, KAIZO, 393, caps) == 20     # skips optional b; c matches Piplup
    assert next_fight_cap(a, ts, KAIZO, 387, caps) == 28     # skips c; d by split
    assert next_fight_cap(d, ts, KAIZO, 387, caps) is None
def test_every_leader_cap_is_his_max_level(kaizo_trainers):  # spec 5.12: leader caps = leader's highest level
    for t in (t for t in kaizo_trainers if t.label and t.label.startswith("Leader ")):
        assert default_cap(t, game=KAIZO) == max(m.level for m in t.party), t.label
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/prep/test_caps.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(prep): Kaizo level caps by split, and the next fight's cap`

### Task 17: Kaizo saves

**Files:**
- Modify: `src/nuzline/saves/layout.py` (`VARS` offset, `VAR_PLAYER_STARTER_INDEX`), `src/nuzline/saves/save.py` (`SaveFile.player_starter`; `load_save(..., game: str = "platinum_us")` defaults tables/names from that game), `src/nuzline/saves/export.py` (game-aware names), `tests/saves/writer.py` (`build_image` gains `vars: dict[int, int]`), `tests/saves/test_pkhex_oracle.py` (`NUZLINE_ORACLE_GAME`, default vanilla)
- Create: `src/nuzline/saves/check.py`
- Test: `tests/saves/test_save.py`, `tests/saves/test_kaizo_save.py`

**Interfaces:**
- Produces:
  - `layout.VARS`: offset of `VarsFlags.vars` in the normal block (page order of `src/savedata/save_table.c`, entry `SAVE_TABLE_ENTRY_VARS_FLAGS` at `:53`; struct `include/vars_flags.h:12`; offsets accumulate as `SavePageInfo_Init` does, documented like the other constants); `VAR_PLAYER_STARTER_INDEX = 48` (`VAR_PLAYER_STARTER` = 0x4030 − `VARS_START` 0x4000, `generated/vars_flags.txt:4107,4159`; C enum values, where alias lines such as `MAP_LOCAL_VARS_END = VAR_MAP_LOCAL_0x1F` do not advance, and `VAR_UNUSED_0x4031` on the next line confirms it).
  - `SaveFile.player_starter: int` (0 before the choice).
  - `impossible_contents(save: SaveFile, tables, names, learnsets) -> list[str]`: per mon, a move outside the union of its species' and pre-evolutions' level-up, TM/HM, tutor and egg moves (Smeargle skipped), and a held or bag item whose name is empty; each line starts with the location. The CLI prints them on stderr whenever a save is loaded (spec §8: warns loudly, never infers the game).

- [ ] **Step 1: Write failing tests**

```python
def test_player_starter_round_trip(tmp_path):
    image = writer.build_image(vars={VAR_PLAYER_STARTER_INDEX: 390})
    assert load_save(write(tmp_path, image)).player_starter == 390
def test_maylene_save_kaizo(kaizo_tables, kaizo_names, kaizo_learnsets, rom_inputs):  # skips without the save
    save = load_save(rom_inputs.kaizo_save, game="platinum_kaizo_nightmare_final")
    assert save.load_result == "ok" and save.player_starter in (387, 390, 393)
    for mon in save.party: mon.to_spec(kaizo_tables)               # every party mon converts
    assert impossible_contents(save, kaizo_tables, kaizo_names, kaizo_learnsets) == []
def test_growth_rate_differs_between_games(tmp_path, kaizo_tables):  # Review Focus 3; Onix 95: Medium Fast -> Slow
    path = write(tmp_path, writer.build_image(box=[writer.box_mon(species=95, exp=100_000)]))
    assert load_save(path).boxes[0][0].level == 46                                   # 46^3 <= 100000 < 47^3
    assert load_save(path, game="platinum_kaizo_nightmare_final").boxes[0][0].level == 43  # 5*43^3/4 <= 100000
def test_impossible_move_warned(tables, names, learnsets):  # synthetic Bulbasaur with Surf
    lines = impossible_contents(save_with_party(bulbasaur(moves=(57,))), tables, names, learnsets)
    assert lines == ["party slot 0 Bulbasaur: Surf is not learnable"]
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/saves -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**; if the user supplies a PKHeX export of the Maylene save, run the oracle test with `NUZLINE_ORACLE_GAME=kaizo` (otherwise it skips).
- [ ] **Step 5: Commit** `feat(saves): Kaizo saves, the player's starter, impossible-content warnings`

### Task 18: `--game` on every command; rivals and optional trainers

**Files:**
- Modify: `src/nuzline/cli.py` (module docstring; `_parser`; `_fight`; `_coverage`, `_print_pool`, `_rank`, `_solve`, `_prep`, `_trainers`, `_save_show`, `_save_export`; `resolve_trainer`; `_source_hash`), `src/nuzline/prep/report.py` (`record` gains `game`), `src/nuzline/prep/checkpoint.py` (`fingerprint` gains `"game"`), `README.md` (one usage line)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 1-4, 14, 16; Task 17's `SaveFile.player_starter` (rival choice), `load_save(..., game=...)` and `impossible_contents`.
- Produces: every subcommand takes `--game {vanilla,kaizo}` (default vanilla). `_fight` loads `load_game(get_game(args.game))` and trainers from `game.intermediate_dir`; `cli.INTERMEDIATE` stays (vanilla) for `benchmarks/` and tests. `resolve_trainer(trainers, *, name=None, trainer_id=None, player_starter=None)`: matches `name` or `label`; drops variants whose `player_starter` differs from a given one; several starter variants left → `ValueError` listing their ids and "pass --save or --trainer-id". `nuzline trainers [--game G] [--save S]` prints `label` when present, `[optional]`, `[gauntlet <group> #p]`, the format when not single, and with `--save` only the matching starter variants. Run records (solve, prep) and the plan tree JSON carry `"game": game.id` next to `table_version`; `enable_compile_cache(..., game.id)`; every loaded save's `impossible_contents` lines go to stderr.

- [ ] **Step 1: Write failing tests**

```python
def test_default_game_is_vanilla(capsys):
    rc = main(["coverage", "--party", SAMPLE, "--trainer", "roark"]); plain = capsys.readouterr().out
    assert main(["coverage", "--game", "vanilla", "--party", SAMPLE, "--trainer", "roark"]) == rc
    assert capsys.readouterr().out == plain
def test_kaizo_without_tables_exit_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(games.GAMES, "platinum_kaizo_nightmare_final", Game("platinum_kaizo_nightmare_final", tmp_path, tmp_path))
    assert main(["coverage", "--game", "kaizo", "--party", SAMPLE, "--trainer", "roark"]) == 1
    assert "tools.build_tables --game kaizo" in capsys.readouterr().err
def test_kaizo_without_tables_loads_no_jax(tmp_path):   # Review Focus 1, in a subprocess
    code = ("import sys; from pathlib import Path; from nuzline.data import games; from nuzline.cli import main;"
            f"games.GAMES['platinum_kaizo_nightmare_final'] = games.Game('platinum_kaizo_nightmare_final', Path(r'{tmp_path}'), Path(r'{tmp_path}'));"
            f"rc = main(['coverage', '--game', 'kaizo', '--party', r'{SAMPLE}', '--trainer', 'roark']);"
            "assert rc == 1 and 'jax' not in sys.modules")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
def test_solve_record_has_game(tmp_path, capsys):   # the existing solve-record test's command, with --record
    record = json.loads(run_solve_with_record(tmp_path).read_text("utf-8"))
    assert record["game"] == "platinum_us" and "table_version" in record
def test_rival_variants_need_save():
    ts = [kt("Pokémon Trainer Barry #3 [Empoleon]", id=10, starter=390), kt("Pokémon Trainer Barry #3 [Torterra]", id=11, starter=393),
          kt("Pokémon Trainer Barry #3 [Infernape]", id=12, starter=387)]
    with pytest.raises(ValueError, match=r"10, 11, 12.*--save or --trainer-id"):
        resolve_trainer(ts, name="barry #3")
    assert resolve_trainer(ts, name="barry #3", player_starter=390).id == 10
def test_trainers_marks_optional(tmp_path, monkeypatch, capsys):  # synthetic Kaizo intermediate dir
    write_trainers(tmp_path, [kt("Lass B", optional=True), kt("Worker C", gauntlet=("Oreburgh Gym", 2))])
    monkeypatch.setitem(games.GAMES, "platinum_kaizo_nightmare_final", Game("platinum_kaizo_nightmare_final", tmp_path, tmp_path))
    assert main(["trainers", "--game", "kaizo"]) == 0
    out = capsys.readouterr().out
    assert "Lass B [optional]" in out and "Worker C [gauntlet Oreburgh Gym #2]" in out
def test_kaizo_roark_refused(capsys, kaizo_tables):  # coverage --game kaizo --trainer roark -> exit 2
    assert main(["coverage", "--game", "kaizo", "--party", SAMPLE, "--trainer", "roark"]) == 2
    out = capsys.readouterr().out
    assert "Battle start: Sandstorm (weather UNSUPPORTED)" in out
    assert "no transcribed AI policy for this game" in out and "code changes unclassified" in out
def test_cli_still_imports_without_jax():
    code = "import sys, nuzline.cli; assert 'jax' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/test_cli.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**, then the full suite in the background.
- [ ] **Step 5: Commit** `feat(cli): --game vanilla|kaizo on every command; rivals by starter`

### Task 19: Full pipeline, docs and the user's review

**Files:**
- Create: `docs/kaizo/pipeline.md` (inputs and `data/local.toml`, the command sequence, what is committed and what is not, how to add a resolution or override, Rulings 8-16 and 21-22 in user terms)
- Modify: `docs/design.md` (§9 extensibility: link the spec and this plan; mark K2b done when merged), `docs/kaizo/discrepancies.md` (regenerated), `README.md` (Kaizo build line)
- Test: none new; the whole suite and ruff

- [ ] **Step 1: Run the pipeline from clean** (in the background):

```bash
~/bin/uv.exe run --group rom python -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom
~/bin/uv.exe run --group rom python -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
~/bin/uv.exe run --group rom python -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
~/bin/uv.exe run --group rom python -m tools.kaizo.overrides check
~/bin/uv.exe run --group rom python -m tools.build_tables --game kaizo
~/bin/uv.exe run nuzline save show --game kaizo "<Maylene save>"
~/bin/uv.exe run nuzline coverage --game kaizo --save "<Maylene save>" --trainer roark
```
Expected: every step exits 0 except `coverage` (exit 2, refusal listing `Battle start: Sandstorm (weather UNSUPPORTED)`, the Kaizo AI line, Roark's unsupported moves/items and the unclassified ROM changes); `git status` shows no file under `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` or `.cache/`.
- [ ] **Step 2: Full suite and lint** — `~/bin/uv.exe run --group rom pytest -q` and `~/bin/uv.exe run ruff check .`, both clean (input-dependent tests run here, since the inputs exist on this machine).
- [ ] **Step 3: Write `docs/kaizo/pipeline.md`**, commit with the regenerated report: `docs(kaizo): pipeline guide and the discrepancy report`.
- [ ] **Step 4: User review (K2b exit, spec §6.5, §11).** Ask the user to review `docs/kaizo/discrepancies.md` together with `data/kaizo/resolutions.toml` and `data/kaizo/overrides.toml`, once, as one review; also ask them to confirm Ruling 15's Galactic-split cap, Ruling 9's gauntlet groupings and Ruling 22's reading of bare `(SUN)` and `(Trick Room)` as permanent. Apply their corrections as resolution/override edits, rebuild, recommit. K2b is done when they accept the report.
