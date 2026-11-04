# K2a Vanilla Schema Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vanilla Platinum's tables gain the spec §7.1 fields: form rows through the personal index, TM/tutor/egg learnsets with the TM and tutor tables, evolutions, the `ai_effect` column with chance-0 secondaries allowed, exp yield, and trainer `format`/`opponent_2`/`partner`. Both `tools/extract_decomp.py` and the K1 ROM reader produce every field, K1's vanilla round-trip test stays green, and the change ships as one deliberate table-version bump (`MANIFEST_VERSION` 7 -> 8).

**Architecture:** Each field lands in one commit across the decomp extractor, the ROM reader (`tools/rom/readers/`), `tools/build_tables.py` and the runtime loaders, so the round-trip test (spec §9.1) keeps proving that both sources agree. Code-resident tables (TM->move, tutors, egg moves) are read from ARM9/overlay bytes through K1's `SymbolMap.find` and `readers.symbol_bytes`. Battle structure defined in scripts (two-trainer and tag battles, walking partners) is parsed by both sources into one normalized op list, and one shared module evaluates it. Species-row arrays grow from 496 to 508 rows (forms at 496-507, `Pokemon_GetFormNarcIndex`), reached through `species_form_row[species, form]`.

**Tech Stack:** Python 3.12, numpy, JAX, pytest, hypothesis, ruff. `tools/rom` uses ndspy from the `rom` uv dependency group (K1). Commands that touch `tests/tools/rom`, and the full-suite runs, use `~/bin/uv.exe run --group rom pytest ... -rs`; everything else uses `~/bin/uv.exe run ...`.

**Spec:** `docs/superpowers/specs/2026-11-03-kaizo-data-design.md` §7.1, §6.6, §9.1 (context §5.3, §5.8, §5.10, §5.11, §7.3, §11 K2a). Read it before any task. §N below refers to the spec's sections.

**Plan family:** K1 (ROM pipeline, from `main`), K2a (this plan), K2b (Kaizo tables, depends on K1 and K2a). K2a starts only after `claude/m4-prep-search` is merged into `main` **and** K1 is merged. File:line references below are written against `claude/m4-prep-search` at `f2e5c2f`. Re-check each one at the merge, before starting (Task 1 Step 1).

## Global Constraints

- The decomp (`external/pokeplatinum`, pinned `579d2d5c65e736a1a4268fb7acbfd70712177c6c`) is authoritative for layouts and vanilla values. Cite file:line in docstrings, as the existing code does. Decomp line numbers below hold at that pin.
- Decomp tests in `tests/tools/test_extract_decomp.py` and `tests/tools/test_extract_trainers.py` use the module constant `DECOMP = Path("external/pokeplatinum")` with a module-level `skipif`, plus their `out` / `trainers` fixtures. New decomp tests elsewhere (e.g. `tests/data/test_learnsets.py`) carry `pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")`. In a worktree, link the main checkout's clone first (PowerShell: `New-Item -ItemType Junction -Path external\pokeplatinum -Target "G:\Farry\Claude Workspace\Pokemon Platinum Game Engine\external\pokeplatinum"`; `external/` is git-ignored).
- ROM tests use K1's session fixtures from `tests/tools/rom/conftest.py` by their exact names: `rom_inputs`, `vanilla_bytes`, `vanilla_files` (`RomFiles` of the vanilla ROM), `kaizo_files`, `decomp`, `charmap`, `symbols` (`SymbolMap` of `DEFAULT_MAP`). Task 1 adds one more, `decomp_inter`. All of them skip when their input or the `rom` group is absent.
- K1's round-trip test `tests/tools/rom/test_extract_rom.py::test_vanilla_round_trip` must pass at the end of every task that touches extraction. It compares `extract("vanilla", ...)` with `extract_static` + `extract_trainers` output, file by file, and new intermediate files are part of that comparison. On this machine the ROM, the decomp and the linker map are present, so a **skip of that test counts as a failure of the step**. Run it with `-rs` and read the skip lines.
- Never commit ROM bytes, NARCs, the patched ROM, `data/intermediate/` or `tables.npz`. Commit `data/gen4_platinum/manifest.json` after every rebuild.
- Rebuild after any change to `tools/extract_decomp.py`, `tools/build_tables.py` or `tools/effect_map.py`:
  `~/bin/uv.exe run python -m tools.extract_decomp --out data/intermediate` then `~/bin/uv.exe run python -m tools.build_tables`.
- Nothing is skipped silently. Unparsable or unexpected input raises an error that names the file or symbol: `ExtractError` (decomp), K1's `tools.rom.readers.ReaderError` (ROM; `ScriptDecodeError` for scripts) or `BuildError` (build).
- "A Kaizo fight the engine cannot model exactly is refused with the missing piece named, never approximated" (§1). So coverage refuses every format other than `single`, and every form whose in-battle form change the engine does not model (Ruling 8).
- `nuzline` gets no new runtime dependency. `nuzline.cli` still does not import jax at module import, and `import nuzline.saves` still does not import jax (`tests/saves/test_save.py::test_import_saves_does_not_import_jax`).
- Never create, edit or delete anything inside a folder named `backup` (any case, any depth).
- ruff `line-length = 100`. `~/bin/uv.exe run ruff check src tests tools` and `ruff format --check` must be clean. The full suite takes about 10 minutes, so run it in the background.

## Rulings (the controller's calls where the spec is silent or leaves a choice open)

1. **Battle format commit.** The header-format work is commit `929b08c` on branch `claude/bold-mendeleev-477a50`, based on `claude/m4-prep-search` at `f2e5c2f` and merged into neither `main` nor M4 when this plan was written. It adds `trainers.json` `format` ("single"/"double" from `double_battle`), `Trainer.format`, `coverage.IMPLEMENTED_FORMATS` and the refusal line `"<name>: a <format> battle (battle format UNSUPPORTED)"`, with tests `test_battle_format_follows_double_battle`, `test_double_battle_trainer_is_refused`, `test_roark_is_a_single_battle` and the `load_trainers` refusal. **If `929b08c` (or a merge of it) is in `main` when K2a starts** (`git merge-base --is-ancestor 929b08c main`), skip Task 2's decomp, loader and coverage steps and keep only its ROM step. Otherwise cherry-pick `929b08c` in Task 2 and resolve conflicts against the merged M4 code; do not reimplement it.
2. **Format values and precedence** (vanilla sources only): `multi` if the trainer is an enemy of a `StartTagBattle`, or an enemy placed on a walking-partner map (Ruling 3). Else `double_two_trainer` if it is an operand of a `StartTrainerBattle t1, t2` with two trainer constants. Else `double` if its header has `double_battle`. Else `single`. If two structures claim the same trainer (two pairs, two tag battles), that is an `ExtractError`. `opponent_2` is the other enemy of the pair or tag battle, else null. Vanilla values: 10 two-trainer pairs are `double_two_trainer` (20 trainers: 9 restaurant pairs and the Spear Pillar grunts). The 11th pair, `TRAINER_GALACTIC_GRUNT_IRON_ISLAND_1/_2` (`res/field/scripts/scripts_iron_island_b2f_left_room.s:171-172`), is `multi` with partner Riley, because it starts after `SetHasPartner` on Riley's map (`:56-57`). The 28 `double_battle` trainers stay `double`, since none of them is a pair or tag operand or placed on a partner map.
3. **Walking partners.** Some maps' script files set `VAR_PARTNER_TRAINER_ID` to a trainer and run `SetHasPartner` (`res/field/scripts/scripts_eterna_forest.s:57-58`; also Iron Island B2F left room, Stark Mountain room 2, Victory Road 1F room 2, Wayward Cave 1F). Every enemy trainer placed on such a map is `multi` with that partner, because `ScrCmd_StartTrainerBattle` adds the partner when `SystemFlag_CheckHasPartner` is set (`src/scrcmd_trainer.c:132`). **Placed enemies** come from the map's trainer object events, the enemy operands of the map script's `StartTrainerBattle` (both trainer operands) and `StartTagBattle` (operands 2-3; a var operand is resolved through the `SetVar` before it), and the Vs. Seeker rematches of those trainers (as `_trainer_maps` does). Two kinds of id are never placed enemies: the value of a `SetVar VAR_PARTNER_TRAINER_ID`, and the `SetVar` values that a partner routine returns. `_trainer_maps`' broader `SetVar` rule (`tools/extract_decomp.py:458-460`) stays as it is for `start_condition`. So the partners (Cheryl, Riley, Buck, Marley, Mira) keep `format == "single"`, `partner == []`. A trainer fought after the partner leaves is over-refused, which is the safe side. A `SetHasPartner` map that places enemy trainers but sets no partner id is an `ExtractError`.
4. **Not recorded:** two single-format trainers who spot the player in the same step (`res/field/scripts/scripts_battles.s:173`). This depends on runtime geometry. The gap is written in `docs/decomp_paths.md` next to the format rules.
5. **Partner variants** are found by evaluating the partner routine for player gender 0/1 and starters Turtwig 387, Chimchar 390 and Piplup 393, recording `VAR_0x8004` at `Return`. A dimension that never changes the result is recorded as null. Fight Area and Spear Pillar have 3 variants (by starter). Jubilife and Veilstone have 6: Dawn for a male player, Lucas for a female one, each by starter.
6. **Form rows.** Personal rows are species ids 0-495, then the data forms in `res/pokemon/form_registry.json` order: Deoxys Attack/Defense/Speed 496-498, Wormadam Sandy/Trash 499-500, Giratina Origin 501, Shaymin Sky 502, Rotom Heat/Wash/Frost/Fan/Mow 503-507. That order must equal `Pokemon_GetFormNarcIndex` (`src/pokemon.c:4929`). Species-row arrays (`species_*`) have 508 rows, and `Names.species_names` keeps 496 entries. The `species_form_row` grid is int16[496, 28] (28 = `UNOWN_FORM_COUNT`, the widest). Each cell holds the data row for a data form, the species itself for a form below the species' `*_FORM_COUNT` (`include/constants/forms.h`) that has no data (cosmetic, battle-only, Arceus), and -1 beyond the count. The grid is stored three times, with identical values: `Tables.species_form_row` (npz, for jax code), manifest `species_form_row: list[list[int]]` (for `Names`, `Learnsets` and the jax-free `nuzline.saves`), and intermediate `form_rows.json`. The host lookup is `nuzline.data.forms.form_row(grid, species, form) -> int`, which has no jax import. A form row takes its weight from the base species, because form `data.json` files have no `pokedex_data`. Form names are the registry keys title-cased ("Wash", "Origin"), and the base form is named after the species. Both extractors take the names from the decomp, as K1 does for the ability list.
7. **Keys.** Level-up, TM and tutor learnsets are keyed by personal row (`Pokemon_LoadLevelUpMovesOf`, `src/pokemon.c:4585`; `Pokemon_ReadMovesetMaskByte`, `src/overlay005/scrcmd_move_tutor.c:211`). Evolutions and egg moves are keyed by species id (`LoadSpeciesEvolutions`, `src/pokemon.c:4812`; `LoadSpeciesEggMoves`, `src/overlay005/daycare.c:445`).
8. **Forms the engine can and cannot play.** Rotom, Deoxys and Wormadam forms are no longer refused, since their form never changes in battle. Two forms do change in battle. Giratina-Origin reverts to Altered at battle start when it does not hold the Griseous Orb outside the Distortion World, or when it is Transformed (`src/battle/battle_lib.c:6412-6417`, `subscript_form_change`). Shaymin-Sky reverts to Land when frozen (`src/battle/battle_lib.c:4536`, `src/battle/battle_controller_player.c:3500`). Coverage refuses, for either side, Giratina form 1 without `ITEM_GRISEOUS_ORB` (112) with `"<where>: Origin Forme without the Griseous Orb (form change UNSUPPORTED)"`, and every Shaymin form 1 with `"<where>: Sky Forme reverts when frozen (form change UNSUPPORTED)"`. Giratina-Origin holding the Orb is exact: no Pokemon in the engine's scope Transforms into it, and the orb's power boost is an item effect that coverage already checks. Arceus forms resolve to the base row and stay refused by the save parser (its form is its plate type, and Multitype is unsupported). Castform and Cherrim are battle-only forms with no data row and are unchanged.
9. **Two effect columns (spec §7.1 names).** Intermediate `moves.json` keeps `effect`, the mechanic effect that `build_tables._move_effect` maps through `effect_map`, and gains `ai_effect`, the id the game stores and the AI scripts read. They are equal in vanilla (both from `MoveTable.effect`). `build_tables` packs `ai_effect` unchanged as `Tables.move_ai_effect` int16[M]. Switching `ai/commands.py` `MOVE_EFFECTS` to read `move_ai_effect` is §7.2 (K2b); K2a only pins that they agree. K2b consumes `ai_effect` and `effect` under exactly these names.
10. **Chance-0 secondaries (§7.1 "chance-0 secondaries allowed"; §5.3) are K2a's.** A move whose effect maps to a `SECONDARY` class (`tools/build_tables.py:72`) and is not in `CERTAIN_SECONDARY_EFFECTS`, with `effect_chance == 0`, never fires its secondary. `_move_effect` packs it as `EffectId.PLAIN` with params `(0, 0, 0)` (each `SECONDARY_*` class is "damage, then a proc", `engine/ids.py:33-35`, so dropping the proc leaves PLAIN), and `move_effect_chance` stays 0. `move_ai_effect` keeps the stored id, so the AI still sees the effect it scores. A certain effect with a nonzero chance is still a `BuildError`. The engine's "chance 0 = on every hit" reading (`engine/ids.py:34`) is unchanged, and so is every engine file. Vanilla has no such move, so vanilla tables do not change; synthetic moves test the rule (Task 4).
11. **Exp yield** is `Tables.species_exp_yield` int16[508] (`SpeciesData.baseExpReward`, `include/struct_defs/species.h:43`), next to `species_exp_rate`.
12. **Tutor, egg and evolution data** are `Learnsets` lookups only. Wiring them into prep's moveset variants is a decision for the prep design, not for K2a. A `by_tutor` move missing from `move_tutors.json` is an `ExtractError`, whereas speciesproc only warns (`tools/dataproc/src/speciesproc.c:672`). Vanilla has no such move.
13. **Code-resident tables** are located with K1's `SymbolMap.find(name) -> Symbol` (frozen dataclass with `module`, `address`, `size`, `name`, `obj`; `KeyError` if absent or ambiguous). Their bytes come from K1's `tools.rom.readers.symbol_bytes(rom, symbols, name)`. K2a adds no symbol API. Tables are validated by shape (lengths, terminators, id ranges), and there is no byte-pattern search fallback.
14. **Name lookups (§7.3) belong to K2b**, in its Task 2, with the manifest keys `aliases` and `renamed: {kind: {vanilla name: id}}` and its message. Vanilla renames no slot, and both keys being absent means empty, so K2a writes neither key and adds no `Names` code for them. K2a's part of §7.3 is the per-game TM->move table (Task 1) and the form names (Task 5).
15. **Version bump.** `MANIFEST_VERSION = 8` moves into `nuzline.data.tables` as the single source, and `build_tables` imports it. Every loader refuses any other version with a rebuild message. With no code change, the new table version invalidates the jax caches (keyed by `table_version`, pruned by `prune_other_versions`) and prep checkpoints (`test_resume_refuses_mismatch`). The M4 overnight gate result (`benchmarks/exhaustive_roark.json`, if committed by then) is marked stale in `benchmarks/README.md`. Re-running it needs the user's save and is the user's call. K2b's plan currently bumps `MANIFEST_VERSION` again (K2a's + 1) for its vanilla manifest additions. That would be a second vanilla bump against §6.6's "once"; it is flagged to the controller and is K2b's to align.
16. **Kaizo-only trainer fields** (`gauntlet_group`, `optional`, `left_side`, `area_condition`, per-mon `nature`/`gender`/`ability_slot` overrides, per-field sources) have no vanilla decomp source and belong to K2b. Vanilla keeps `start_condition`.
17. **Items are out of K2a's scope.** K2a produces no per-item array (no `item_hold_effect` or similar). Per-game item hold effects, parameters and names (§7.1 "Items") are K2b's, built on K1's `read_items`.

## Review Focus

1. **Tables built before K2a** (manifest v7 on disk, new code): `load_tables`/`load_names`/`load_learnsets` raise one "rebuild" message instead of a `KeyError` on a missing array. Test in Task 9.
2. **A save or party Pokemon with a form beyond its species' count** (corrupt save, Rotom form 6, typo in TOML): `RecordError`/`ValueError` naming the slot, never an index of -1 into the form grid. Tests in Task 6.
3. **A trainer on a walking-partner map** (Eterna Forest, with Cheryl; the Iron Island grunts, with Riley) is refused as `multi` with the partner named and is never simulated as a single, while the partner's own row stays `single`. Test in Task 3.
4. **A party TOML form name** that is unknown, or given for a species without data forms: `ValueError` listing the valid names. Test in Task 6.
5. **Giratina-Origin without the Griseous Orb, or Shaymin-Sky**, on either side, is refused by coverage as a form change and is never simulated with the form's stats. Test in Task 6.
6. **A chance-0 non-certain secondary** builds as PLAIN with no proc and does not raise. Test in Task 4.

---

### Task 1: ROM reader level with M4, and the TM table

M4 added `items.json` `teaches_move` and `tm_learnsets.json` after K1 branched from `main`. If K1's reader already produces both (the round-trip test passes after the M4 merge), do Steps 1-2 only and mark the rest done.

**Files:**
- Modify: `tools/rom/readers/personal.py` (TM masks -> `tm_learnsets.json`), `tools/rom/readers/items.py` (`teaches_move`), `tools/extract_rom.py`, `tests/tools/rom/conftest.py` (`decomp_inter`)
- Create: `tools/rom/readers/code_tables.py`
- Test: `tests/tools/rom/test_code_tables.py`

**Interfaces:**
- Consumes: K1 `SymbolMap.find -> Symbol`, `tools.rom.readers.symbol_bytes`, `ReaderError`, `RomFiles`; `tools.extract_decomp.extract_static`, `extract_trainers`.
- Produces:
  - Fixture `decomp_inter` (session, in `tests/tools/rom/conftest.py`): a `tmp_path_factory` directory filled by `extract_static(decomp, d)` and `extract_trainers(decomp, d)`. It depends on `decomp`, so it skips when the decomp is absent.
  - `code_tables.tm_moves(rom: RomFiles, symbols: SymbolMap) -> list[int]`: `sTMHMMoves`, defined in the generated `res/items/item_tm_move_map.h` (template `tools/dataproc/data/item_tm_move_map.h.template:6`, included at `src/item.c:17`, read by `Item_MoveForTMHM`, `src/item.c:296-304`). 100 u16, each `0 < m < len(moves)`, else `ReaderError` naming the symbol.

- [ ] **Step 1: Re-check references at the merge.** Open every file:line this plan cites in `tools/`, `src/nuzline/` and `tests/` on the merged `main`, and note any drift in the task you are on. Check that K1's names match Rulings 13 and Global Constraints (`SymbolMap.find`, `symbol_bytes`, fixture names, `read_trainers` signature, `tools.rom.readers.maps`).
- [ ] **Step 2: Run K1's suite** — `~/bin/uv.exe run --group rom pytest tests/tools/rom -q -rs`. `test_vanilla_round_trip` must pass, not skip.
- [ ] **Step 3: Write failing tests**

```python
def test_tm_table_symbol(symbols):
    s = symbols.find("sTMHMMoves")
    assert s.module == "arm9" and s.size == 200
def test_tm_moves_match_decomp(vanilla_files, symbols, decomp_inter):
    tms = tm_moves(vanilla_files, symbols)
    assert len(tms) == 100 and tms[0] == 264           # TM01 Focus Punch
    items = json.loads((decomp_inter / "items.json").read_text("utf-8"))
    assert tms == [i["teaches_move"] for i in items[328:428]]   # ITEM_TM01..ITEM_HM08
```

- [ ] **Step 4: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_code_tables.py -q -rs`
- [ ] **Step 5: Implement** `tm_moves` on `symbol_bytes`. The items reader sets `teaches_move` for ITEM_TM01..ITEM_HM08 from `tm_moves` and null elsewhere. The personal reader writes `tm_learnsets.json` from `tmLearnsetMasks` (bit n = TM index n, as in `extract_tm_learnsets`) for species rows 0-495.
- [ ] **Step 6: Run, expect PASS**, then `~/bin/uv.exe run --group rom pytest tests/tools/rom -q -rs` (round trip passes, no skip).
- [ ] **Step 7: Commit** `feat(rom): TM table and TM learnsets from the ROM`

### Task 2: Trainer battle format from the header

Ruling 1: if `929b08c` is in `main`, do only the ROM part (Steps 1-2 with the ROM test only, then 4-5). Otherwise `git cherry-pick 929b08c` first, resolve conflicts against the merged code, and add the ROM part.

**Files:**
- Modify: `tools/rom/readers/trainers.py`; via `929b08c` if absent: `tools/extract_decomp.py` (`extract_trainers`), `src/nuzline/data/trainer.py`, `src/nuzline/engine/coverage.py`
- Test: `tests/tools/rom/test_trainers.py` (K1's); via `929b08c`: `tests/tools/test_extract_trainers.py`, `tests/data/test_trainer.py`, `tests/engine/test_coverage.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: K1 `read_trainers(rom, decomp, symbols, charmap, start_conditions)`.
- Produces: the `929b08c` interfaces (Ruling 1). On the ROM side, `format` comes from `TrainerHeader.battleType` (`include/struct_defs/trainer_data.h:31`): 0 is single, `BATTLE_TYPE_DOUBLES` (`1 << 1`, `include/constants/battle.h:27`) is double, and anything else is a `ReaderError` naming the trainer id.

- [ ] **Step 1: Write the failing ROM test**

```python
def test_rom_trainer_format_from_battle_type(vanilla_files, decomp, symbols, charmap, decomp_inter):
    want = json.loads((decomp_inter / "trainers.json").read_text("utf-8"))
    rows = read_trainers(vanilla_files, decomp, symbols, charmap, [t["start_condition"] for t in want])
    assert sum(t["format"] == "double" for t in rows) == 28
    assert [t["format"] for t in rows] == [t["format"] for t in want]
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_trainers.py tests/tools/test_extract_trainers.py tests/data/test_trainer.py tests/engine/test_coverage.py -q -rs`
- [ ] **Step 3: Implement** the ROM `format` (and the cherry-pick, if needed).
- [ ] **Step 4: Rebuild, run Step 2's command plus `tests/tools/rom`, expect PASS** (round trip included, no skip).
- [ ] **Step 5: Commit** `feat(rom): trainer battle format from TrainerHeader.battleType` (plus the cherry-picked `929b08c` commit if it was absent)

### Task 3: Two-trainer, tag and walking-partner battles

**Files:**
- Create: `tools/battle_scripts.py` (normalized ops, evaluator, structure finder; shared by both extractors)
- Modify: `tools/extract_decomp.py` (`extract_trainers`, plus `_script_ops(path) -> dict[str, list[Op]]` and `_placed_enemies`), `tools/rom/readers/maps.py` (K1's; add `script_ops`), `tools/rom/readers/trainers.py`, `src/nuzline/data/trainer.py`, `docs/decomp_paths.md` (format rules and Ruling 4's gap)
- Test: `tests/tools/test_battle_scripts.py`, `tests/tools/test_extract_trainers.py` (including the edit to `929b08c`'s `test_battle_format_follows_double_battle`), `tests/engine/test_coverage.py`, `tests/tools/rom/test_maps.py`, `tests/tools/rom/test_trainers.py`

**Interfaces:**
- Consumes: K1 `tools.rom.readers.maps.command_table(decomp)`, `ScriptDecodeError`, its header-entry walk and map-event reading (used by `trainer_start_conditions`).
- Produces (`tools/battle_scripts.py`):
  - `@dataclass(frozen=True) Op(name: str, args: tuple[int | str, ...])`, with names `label`(name), `StartTrainerBattle`(t1, t2), `StartTagBattle`(partner, e1, e2), `SetVar`(var, value), `GoToIfEq`(var, value, label), `GoTo`(label), `Call`(label), `Return`, `End`, `GetPlayerGender`(var), `GetPlayerStarterSpecies`(var), `SetHasPartner`, and `other` for every other command. Numbers are ids (trainers, species, genders, vars). Labels are strings: decomp label names, or `f"@{offset}"` on the ROM side.
  - `var_ids(decomp: Path) -> dict[str, int]` from `generated/vars_flags.txt` (sequential enum; `A = B` aliases). It asserts `VAR_0x8004 == 0x8004` and `VAR_RESULT == 0x800C`.
  - `STARTERS = (387, 390, 393)`. `PartnerVariant` is imported from `nuzline.data.trainer` (below), not redefined.
  - `partner_variants(ops: list[Op], label: str, var: int) -> tuple[PartnerVariant, ...]` (Ruling 5). It raises on an `other` op inside the evaluated path, on a loop, and on a run that ends without setting `var`.
  - `@dataclass(frozen=True) BattleStructure(format: str, opponent_2: int | None, partner: tuple[PartnerVariant, ...])` and `battle_structures(scripts: dict[str, list[Op]], placed: dict[str, set[int]]) -> dict[int, BattleStructure]`. `scripts` maps each map header to its script's ops in file order, and `placed` maps each header to its placed enemies (Ruling 3). It applies Rulings 2-3. When a `StartTagBattle`'s partner operand is a var, its variants come from the `Call` immediately before it in the same label block.
- ROM op decoder (in K1's `tools/rom/readers/maps.py`): `script_ops(script: bytes, table) -> list[Op]`. It decodes every byte range reachable from the header entries, using K1's walk, and emits ops in address order. Each jump target and header entry gets a `label` op `f"@{offset}"` in front of it, so a run that falls off the end of one block into a labelled one (as `SpearPillar_Return` is reached) continues in the evaluator exactly as it does in the decomp text. Fusions: `CompareVar v, x` (or `CompareVarToValue`) followed directly by `GoToIf 1, off` (condition 1 = equal, `asm/macros/scrcmd.inc`) becomes `GoToIfEq v, x, "@off"`. Any other `GoToIf` condition, or a `CompareVar` not directly followed by `GoToIf`, is `other`. `SetVarFromValue` becomes `SetVar`. An operand inside the var ranges (`asm/macros/scrcmd.inc:310`) is a var. An unknown opcode raises K1's `ScriptDecodeError`. The test helper `assemble(table, *cmds) -> bytes` (in `tests/tools/rom/test_maps.py`) writes a one-entry header plus the commands, with jump operands as offsets relative to the next command, as `scrcmd.inc` encodes them.
- `trainers.json` gains `"opponent_2": int | null` and `"partner": [{"trainer", "starter", "player_gender"}]`, and `format` widens to `single | double | double_two_trainer | multi`. `nuzline.data.trainer` gains `@dataclass(frozen=True) PartnerVariant(trainer: int, starter: int | None, player_gender: int | None)`. `Trainer` gains `opponent_2: int | None = None` and `partner: tuple[PartnerVariant, ...] = ()`. `load_trainers` refuses rows missing either key with `929b08c`'s message shape ("... has no battle structure: re-run tools/extract_decomp.py").

- [ ] **Step 1: Write failing tests.** Also edit `929b08c`'s `test_battle_format_follows_double_battle`: its last line becomes `assert {t["format"] for t in trainers} == {"single", "double", "double_two_trainer", "multi"}`, and its `double` count stays `== len(doubles) == 28`.

```python
# tests/tools/test_battle_scripts.py (synthetic ops, no decomp needed)
def test_partner_variants_by_gender_and_starter():
    # Jubilife shape: gender branch, then a starter branch per counterpart
    assert set(partner_variants(JUBILIFE_OPS, "SetTeam", 0x8004)) == {
        PartnerVariant(t, s, g) for g, ts in ((0, (11, 12, 13)), (1, (21, 22, 23)))
        for t, s in zip(ts, (390, 393, 387))}
def test_partner_variants_collapse_gender():   # Spear Pillar shape: the same team for both genders
    assert set(partner_variants(SPEAR_PILLAR_OPS, "SetTeam", 0x8004)) == {
        PartnerVariant(t, s, None) for t, s in ((31, 390), (32, 393), (33, 387))}
def test_partner_routine_with_unknown_op_raises():   # an Op("other", ()) on the path -> ValueError naming it
def test_trainer_in_two_structures_raises():          # one trainer in two StartTrainerBattle pairs -> ValueError naming it
def test_fall_through_into_label():
    ops = [Op("label", ("@0",)), Op("SetVar", (0x8004, 5)), Op("label", ("@9",)), Op("Return", ())]
    assert partner_variants(ops, "@0", 0x8004) == (PartnerVariant(5, None, None),)

# tests/tools/test_extract_trainers.py
def test_battle_structures(trainers):
    consts = (DECOMP / "generated" / "trainers.txt").read_text(encoding="utf-8").split()
    by = {c: trainers[i] for i, c in enumerate(consts)}
    assert sum(t["format"] == "double_two_trainer" for t in trainers) == 20   # 10 pairs (Ruling 2)
    g1, g2 = by["TRAINER_GALACTIC_GRUNT_IRON_ISLAND_1"], by["TRAINER_GALACTIC_GRUNT_IRON_ISLAND_2"]
    riley = by["TRAINER_RILEY_IRON_ISLAND"]
    assert (g1["format"], g1["opponent_2"], g2["opponent_2"]) == ("multi", g2["id"], g1["id"])
    assert g1["partner"] == [{"trainer": riley["id"], "starter": None, "player_gender": None}]
    mars, jupiter = by["TRAINER_COMMANDER_MARS_SPEAR_PILLAR"], by["TRAINER_COMMANDER_JUPITER_SPEAR_PILLAR"]
    assert (mars["format"], mars["opponent_2"]) == ("multi", jupiter["id"])
    assert {(p["starter"], p["player_gender"]) for p in mars["partner"]} == {(387, None), (390, None), (393, None)}
    assert len(by["TRAINER_GALACTIC_GRUNT_JUBILIFE_CITY_1"]["partner"]) == 6
    jack, cheryl = by["TRAINER_BUG_CATCHER_JACK"], by["TRAINER_CHERYL_ETERNA_FOREST"]   # Eterna Forest
    assert jack["format"] == "multi"
    assert jack["partner"] == [{"trainer": cheryl["id"], "starter": None, "player_gender": None}]
    for partner in (cheryl, riley):                       # Ruling 3: partners are not placed enemies
        assert (partner["format"], partner["partner"], partner["opponent_2"]) == ("single", [], None)
    assert roark(trainers)["partner"] == [] and roark(trainers)["opponent_2"] is None

# tests/engine/test_coverage.py
@pytest.mark.parametrize(("name", "fmt"), [("Mars", "multi"), ("Ismael", "double_two_trainer"), ("Jack", "multi")])
def test_multi_and_two_trainer_battles_are_refused(tables, names, name, fmt):
    # Mars @ Spear Pillar (tag), Artist Ismael (restaurant pair), Bug Catcher Jack (Eterna Forest, Cheryl)
    t = next(t for t in load_trainers(INTERMEDIATE) if t.name == name and t.format == fmt)
    r = check_opponent(tables, names, t, build_trainer_party(t, _inter()), 0)
    assert f"{name}: a {fmt} battle (battle format UNSUPPORTED)" in r.missing

# tests/tools/rom/test_maps.py
def test_script_ops_fuse_compare_and_branch(decomp):   # hand-assembled with K1's command_table opcodes
    table = command_table(decomp)
    script = assemble(table, ("CompareVar", 0x8004, 3), ("GoToIf", 1, 2), ("End",), ("Return",))
    ops = script_ops(script, table)                        # offset 2 skips the 2-byte End -> the Return
    assert [o.name for o in ops] == ["label", "GoToIfEq", "End", "label", "Return"]
    assert ops[1].args == (0x8004, 3, ops[3].args[0])
def test_rom_battle_structures_match(vanilla_files, decomp, symbols, charmap, decomp_inter):
    want = json.loads((decomp_inter / "trainers.json").read_text("utf-8"))
    got = read_trainers(vanilla_files, decomp, symbols, charmap, [t["start_condition"] for t in want])
    key = lambda t: (t["format"], t["opponent_2"], t["partner"])
    assert [key(t) for t in got] == [key(t) for t in want]
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_battle_scripts.py tests/tools/test_extract_trainers.py tests/engine/test_coverage.py tests/tools/rom/test_maps.py -q -rs`
- [ ] **Step 3: Implement** `tools/battle_scripts.py` and the decomp `_script_ops`: text `.s` -> ops per file, with symbolic values resolved through `trainers.txt`, `species.txt`, `genders.txt`, `var_ids`, ints and TRUE/FALSE. An unknown symbol in a kept op is an `ExtractError`. Then add `_placed_enemies` (Ruling 3, reusing `_map_headers`, the object-event reading and the rematch expansion of `_trainer_maps`), their use in `extract_trainers`, and the loader fields.
- [ ] **Step 4: Implement the ROM side.** Add `script_ops` in `maps.py`, build `placed` from the map events and enemy operands as Ruling 3 says, and call `battle_structures` from `read_trainers`.
- [ ] **Step 5: Rebuild, run Step 2's command plus `tests/tools/rom`, expect PASS** (round trip included, no skip).
- [ ] **Step 6: Commit** `feat(data): two-trainer, tag and walking-partner battles, refused by coverage`

### Task 4: Exp yield, the AI effect column and chance-0 secondaries

**Files:**
- Modify: `tools/extract_decomp.py:154-178` (`extract_species`: `"exp_yield"`), `:213-244` (`extract_moves`: `"ai_effect"`), `tools/build_tables.py:109-124` (`_move_effect`, Ruling 10), `:191-230`, `src/nuzline/data/tables.py:19-46`, `tools/rom/readers/personal.py`, `tools/rom/readers/moves.py`
- Test: `tests/tools/test_extract_decomp.py`, `tests/tools/test_build_tables.py` (create), `tests/data/test_tables.py`, `tests/ai/test_policy.py`

**Interfaces:**
- Produces: `species.json` `"exp_yield"` (decomp `base_exp_reward`; ROM `baseExpReward`, byte 9 of `SpeciesData`). `moves.json` `"ai_effect"`, equal to `effect` in vanilla, with both read from `MoveTable.effect` (u16 at 0) on the ROM side. `Tables.species_exp_yield` int16[S]. `Tables.move_ai_effect` int16[M] (= intermediate `ai_effect`). `_move_effect(move) -> tuple[int, tuple[int, int, int]]` implements Ruling 10.

- [ ] **Step 1: Write failing tests**

```python
def test_exp_yield(out):                      # decomp extractor
    sp = json.loads((out / "species.json").read_text("utf-8"))
    assert (sp[1]["exp_yield"], sp[487]["exp_yield"]) == (64, 220)   # Bulbasaur, Giratina
def test_ai_effect_equals_effect_in_vanilla(out):
    assert all(m["ai_effect"] == m["effect"] for m in json.loads((out / "moves.json").read_text("utf-8")))
def test_ai_effect_and_exp_yield_arrays(tables, names):
    assert int(tables.species_exp_yield[names.species("Bulbasaur")]) == 64
    assert int(tables.move_ai_effect[names.move("Stealth Rock")]) == 266
def test_transcribed_move_effects_match_tables(tables):          # tests/ai/test_policy.py
    for move, effect in MOVE_EFFECTS.items():
        assert int(tables.move_ai_effect[move]) == effect

# tests/tools/test_build_tables.py (synthetic move dicts, no tables needed)
def test_chance_zero_secondary_is_plain():     # a move with Ember's effect, effect_chance 0
    assert _move_effect({**EMBER, "effect_chance": 0}) == (int(EffectId.PLAIN), (0, 0, 0))
def test_certain_secondary_with_chance_raises():   # effect 218 with effect_chance 30 -> BuildError naming the move
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_extract_decomp.py tests/tools/test_build_tables.py tests/data/test_tables.py tests/ai/test_policy.py tests/tools/rom -q -rs`
- [ ] **Step 3: Implement** in both extractors, `build_tables` (with the encoding lines in its docstring, Ruling 10's rule among them) and `Tables` (field comments).
- [ ] **Step 4: Rebuild, run, expect PASS** (round trip included, no skip). The rebuilt `move_effect` array equals the previous one, because vanilla has no chance-0 non-certain secondary.
- [ ] **Step 5: Commit** `feat(data): exp yield, the AI's effect id, chance-0 secondaries as plain`

### Task 5: Form rows in the data

**Files:**
- Create: `src/nuzline/data/forms.py` (no jax import)
- Modify: `tools/extract_decomp.py` (new `extract_forms`, `species_form_rows`; `extract_learnsets` and `extract_tm_learnsets` add rows 496-507; `extract_static` writes `forms.json` and `form_rows.json`), `tools/build_tables.py`, `src/nuzline/data/tables.py`, `src/nuzline/data/learnsets.py`, `tools/rom/readers/personal.py`, `tools/rom/readers/learnsets.py`, `tools/extract_rom.py`
- Test: `tests/tools/test_extract_decomp.py`, `tests/data/test_tables.py`, `tests/data/test_learnsets.py`, `tests/tools/rom/test_personal.py` (K1's)

**Interfaces:**
- Produces:
  - `extract_forms(decomp: Path) -> list[dict]`: one entry per data form in registry order, `{"species": id, "form": n, "row": 496 + i, "name": "Wash", **species_row_fields}`. The row fields are exactly `extract_species`' keys except `name`, with weight from the base species (Ruling 6).
  - `species_form_rows(decomp: Path) -> list[list[int]]`: Ruling 6's grid, written as `form_rows.json`. It is checked against `Pokemon_GetFormNarcIndex`'s bases 496/499/501/502/503 (`src/pokemon.c:4935-4959`).
  - `nuzline.data.forms.form_row(grid, species: int, form: int) -> int`: `grid[species][form]`, taking either the manifest's nested lists or a numpy array. It raises `ValueError(f"species {species} has no form {form}")` for -1, a negative form, or a form >= 28.
  - Every `species_*` array has 508 rows (species, then forms). New fields: `Tables.species_form_row` int16[496, 28]; manifest `species_form_row` (the same grid); manifest `species_abilities` and `species_base_friendship` with 508 entries; manifest `form_names: {"<species>": [base name, form 1 name, ...]}` for the five data-form species. `level_learnsets` / `tm_learnsets` are keyed by row.
  - `Names.form(species: int, name: str | int) -> int`, `Names.form_names(species: int) -> tuple[str, ...]` (`("Rotom", "Heat", ...)`, or `(species name,)` for a species without data forms), `Names.species_abilities(species, form=0)`, `Names.species_base_friendship(species, form=0)`. `Learnsets.level_up(species, form=0)`, `.relearner_moves(species, level, form=0)` and `.can_learn_tm(species, item, form=0)` resolve the row with `form_row(manifest["species_form_row"], ...)` and raise its `ValueError`.
  - ROM side: `tools.rom.readers.personal.read_forms(rom, decomp, charmap) -> list[dict]` (the `forms.json` shape) and personal and wotbl rows 496-507. Both NARCs must have exactly 508 members, else a `ReaderError`, and the reader checks that the decomp grid's data rows are exactly 496..507 and fit the member count. `forms.json` row fields are read from the ROM. Its `name` fields and `form_rows.json` come from the decomp extractor (Ruling 6), and `rom_sources.json` records `"form_rows.json": "decomp: form registry"` and `"forms.json": "rom; names: decomp"`.

- [ ] **Step 1: Write failing tests**

```python
def test_forms_registry_order_matches_narc_index():          # tests/tools/test_extract_decomp.py
    f = extract_forms(DECOMP)
    assert [(x["species"], x["form"], x["row"]) for x in f][:3] == [(386, 1, 496), (386, 2, 497), (386, 3, 498)]
    wash = next(x for x in f if (x["species"], x["form"]) == (479, 2))
    assert wash["row"] == 504 and wash["base_stats"] == [50, 65, 107, 105, 107, 86]
def test_form_grid():
    g = species_form_rows(DECOMP)
    assert g[479][:7] == [479, 503, 504, 505, 506, 507, -1] and g[201][27] == 201 and g[201][0] == 201
    assert g[493][17] == 493 and g[493][18] == -1 and g[1][1] == -1
def test_form_tables(tables, names):                         # tests/data/test_tables.py
    origin = form_row(tables.species_form_row, names.species("Giratina"), 1)
    assert origin == 501 and int(tables.species_abilities_decomp[origin, 0]) == names.ability("Levitate")
    assert tables.species_base_stats.shape[0] == 508 and len(names.species_names) == 496
    assert names.form(479, "wash") == 2 and names.form_names(479)[1] == "Heat"
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        names.species_abilities(479, 6)
@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")   # tests/data/test_learnsets.py
def test_form_rows_have_learnsets(learnsets, names):
    wash = json.loads((DECOMP / "res/pokemon/rotom/forms/wash/data.json").read_text("utf-8"))
    consts = (DECOMP / "generated" / "moves.txt").read_text("utf-8").split()
    assert [m for _, m in learnsets.level_up(479, form=2)] == [
        consts.index(n) for _, n in wash["learnset"]["by_level"]
    ]
    assert learnsets.can_learn_tm(479, names.item("TM17"), form=2) == ("TM17" in wash["learnset"]["by_tm"])
def test_personal_narcs_have_508_members(vanilla_files):     # tests/tools/rom/test_personal.py
    for path in ("poketool/personal/pl_personal.narc", "poketool/personal/wotbl.narc"):
        assert len(vanilla_files.narc(path)) == 508
def test_rom_forms_match_decomp(vanilla_files, decomp, charmap, decomp_inter):
    want = json.loads((decomp_inter / "forms.json").read_text("utf-8"))
    assert read_forms(vanilla_files, decomp, charmap) == want    # row fields read from the NARC members
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_extract_decomp.py tests/data tests/tools/rom -q -rs`
- [ ] **Step 3: Implement** the extractors, the build (concatenate species and form rows in row order), `forms.py` and the loaders.
- [ ] **Step 4: Rebuild, run Step 2's command, expect PASS** (round trip included, no skip).
- [ ] **Step 5: Commit** `feat(data): form rows through the personal index`

### Task 6: Form-aware consumers and form refusals

Every index into a `species_*` table, and every species-keyed lookup, goes through the form row.

**Files:**
- Modify: `src/nuzline/engine/state.py:281-310` (`_fill_slot`: base stats, types and abilities by row; a bad form raises), `src/nuzline/ai/commands.py:259, 273`, `src/nuzline/ai/policy.py:371, 435`, `src/nuzline/ai/switching.py:281, 372` (row = `tables.species_form_row[state.species, state.form]`), `src/nuzline/engine/coverage.py:155-178` (`_check_party`: abilities by row; Ruling 8's two form refusals), `src/nuzline/prep/pool.py:69-117`, `src/nuzline/prep/problem.py:93-123, 206`, `src/nuzline/prep/report.py:85, 207`, `src/nuzline/prep/variants.py:241, 287, 313-314`, `src/nuzline/saves/pkm.py:63-120, 158-162` (the refused set becomes `{493}` Arceus; form >= count -> `RecordError`), `src/nuzline/saves/export.py:47` (writes `form = "<name>"` when non-zero), `src/nuzline/data/party.py:49-85` (optional `form` key: name or int), `src/nuzline/data/trainer.py:100-190` (`IntermediateData` loads `forms.json`, `form_rows.json`; `build_trainer_party` uses the row for abilities, gender ratio, friendship and default moves)
- Test: `tests/engine/test_state.py`, `tests/engine/test_coverage.py`, `tests/ai/test_policy.py`, `tests/saves/test_to_spec.py:107-117`, `tests/saves/test_export.py`, `tests/data/test_party.py`, `tests/data/test_trainer.py`

**Interfaces:**
- Consumes: Task 5 `nuzline.data.forms.form_row`, `Tables.species_form_row`, manifest `species_form_row`, `Names.form`, `Names.form_names`, `Learnsets(..., form=)`.
- `src/nuzline/saves/pkm.py` resolves the row with `form_row(_as_np(tables.species_form_row), species, form)` from `nuzline.data.forms`, catching its `ValueError` as `RecordError` naming the slot. It must not import `nuzline.data.tables`, which imports `jax.numpy` at module import.

- [ ] **Step 1: Write failing tests**

```python
def test_rotom_wash_state_uses_its_row(tables, sample_spec):
    s = make_state(tables, [replace(sample_spec, species=479, form=2, ability_slot=0)], [sample_spec])
    assert s.types[0, 0].tolist() == [int(tables.species_types[504, 0]), int(tables.species_types[504, 1])]
    sp = sample_spec
    want = calc_stats(np.asarray(tables.species_base_stats[504]), np.asarray(sp.ivs), np.asarray(sp.evs),
                      sp.level, np.asarray(tables.nature_mods[sp.nature]), species=479)
    assert int(s.max_hp[0, 0]) == int(want[0])
def test_bad_form_raises(tables, sample_spec):
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        make_state(tables, [replace(sample_spec, species=479, form=6)], [sample_spec])
def test_ai_reads_the_form_row_ability(tables, names, sample_spec):   # tests/ai/test_policy.py
    s = make_state(tables, [sample_spec],
                   [replace(sample_spec, species=487, form=1, item=names.item("Griseous Orb"))])
    assert int(attacker_ability(tables, s, side=1)) == int(tables.species_abilities[501, 0])   # Levitate
def test_data_form_converts_arceus_form_raises(tables):              # replaces test_alternate_form_raises
    assert mon_record(species=479, form=2).to_spec(tables).form == 2
    with pytest.raises(RecordError, match="Arceus"):    mon_record(species=493, form=3).to_spec(tables)
    with pytest.raises(RecordError, match="party slot 0"): load_party_record(mon_record(species=479, form=6))
def test_party_toml_form(tmp_path, names):
    assert _party(tmp_path, names, 'species = "Rotom"\nform = "Wash"')[0].form == 2
    with pytest.raises(ValueError, match="Heat, Wash, Frost, Fan, Mow"):
        _party(tmp_path, names, 'species = "Rotom"\nform = "Blue"')
    with pytest.raises(ValueError, match="Bulbasaur has no forms"):
        _party(tmp_path, names, 'species = "Bulbasaur"\nform = "Sky"')
def test_export_round_trips_a_form(tmp_path, tables, names):         # tests/saves/test_export.py
    spec = mon_record(species=479, form=2).to_spec(tables)
    text = export_party([spec], names)
    assert 'form = "Wash"' in text and load_party(_write(tmp_path, text), names)[0] == spec
def test_trainer_form_uses_row(tables, names):                       # tests/data/test_trainer.py
    t = replace(roark_trainer(), party=(replace(roark_trainer().party[0], species=487, form=1),))
    mon = build_trainer_party(t, _inter())[0]
    assert names.species_abilities(487, 1)[mon.ability_slot] == names.ability("Levitate")
@pytest.mark.parametrize(("species", "form", "item", "line"), [
    (487, 1, "no item", "Giratina (slot 0): Origin Forme without the Griseous Orb (form change UNSUPPORTED)"),
    (492, 1, "Griseous Orb", "Shaymin (slot 0): Sky Forme reverts when frozen (form change UNSUPPORTED)"),
])
def test_in_battle_form_changes_are_refused(tables, names, species, form, item, line):   # Ruling 8
    mon = replace(sample_party(names)[0], species=species, form=form,
                  item=0 if item == "no item" else names.item(item))
    r = check_coverage(tables, names, [mon], roark_trainer(), roark_party(names), Weather.NONE)
    assert any(m.endswith(line) for m in r.missing)
    ok = replace(mon, species=487, form=1, item=names.item("Griseous Orb"))
    r = check_coverage(tables, names, [ok], roark_trainer(), roark_party(names), Weather.NONE)
    assert not any("form change" in m for m in r.missing)
```

(`mon_record`, `load_party_record`, `_party`, `_write`, `export_party`, `attacker_ability`: use the helpers these test files already have, under their actual names, and note any rename in the test.)

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/engine/test_state.py tests/engine/test_coverage.py tests/ai/test_policy.py tests/saves tests/data -q -rs`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify that no species-id index remains.** `rg -n "species_(base_stats|types|abilities|abilities_decomp|weight|gender|friendship|exp_rate|exp_yield)\)?\[" src benchmarks` must list only row-indexed sites (`row`, `species_form_row[...]`, `form_row(...)`), or sites with a comment saying why species is right. For example, `species_exp_rate` is the same for every form, so reading it before the form is known is fine.
- [ ] **Step 5: Run, expect PASS.** Then run `tests/engine tests/ai tests/prep tests/saves tests/data -q -rs`, and `tests/test_differential.py` in the background. `tests/saves/test_save.py::test_import_saves_does_not_import_jax` must pass.
- [ ] **Step 6: Commit** `feat: forms reach the engine, AI, prep and save parser through the form row`

### Task 7: Evolutions and egg moves

**Files:**
- Modify: `tools/extract_decomp.py` (`extract_evolutions`, `extract_egg_moves`), `tools/build_tables.py` (manifest `evolutions`, `evolution_methods`, `egg_moves`), `src/nuzline/data/learnsets.py`, `tools/extract_rom.py`
- Create: `tools/rom/readers/evolutions.py`; add `egg_moves` to `tools/rom/readers/code_tables.py`
- Test: `tests/tools/test_extract_decomp.py`, `tests/data/test_learnsets.py`, `tests/tools/rom/test_code_tables.py`

**Interfaces:**
- Produces:
  - `evolutions.json`: `{"<species>": [[method, param, target], ...]}` for species 0-495. `method` is the `generated/evolution_methods.txt` id, and `param` is packed as `ProcEvolutions` packs it (`tools/dataproc/src/speciesproc.c:391-468`: level, item id, move id, species id, or 0). There are at most `MAX_EVOLUTIONS` (7) entries, and method-0 slots are dropped. ROM: `poketool/personal/evo.narc`. Each member is `SpeciesEvolution data[MAX_EVOLUTIONS] ALIGN_4` (`speciesproc.c:388`), i.e. **44 bytes**: 7 x 6-byte `SpeciesEvolution` (u16 method, param, targetSpecies; `include/struct_defs/species.h:60-64`) plus 2 padding bytes. `tools.rom.readers.evolutions.read_evolutions(rom: RomFiles) -> dict[str, list[list[int]]]` reads it with K1's `fixed_members(members, 44, path)`, in the `evolutions.json` shape.
  - `egg_moves.json`: `{"<species>": [move, ...]}` for species with egg moves. ROM: `code_tables.egg_moves(rom, symbols) -> dict[int, list[int]]` from `sEggMoves` (marker `20000 + species`, terminator `0xFFFF`; `tools/dataproc/data/species_egg_moves.h.template:7-8`, read as `LoadSpeciesEggMoves` reads it).
  - `Learnsets.evolutions(species) -> tuple[tuple[int, int, int], ...]`, `Learnsets.egg_moves(species) -> tuple[int, ...]`. Manifest `evolution_methods` holds the constants' stems, lowercased.

- [ ] **Step 1: Write failing tests**

```python
def test_evolutions():                                          # tests/tools/test_extract_decomp.py
    ev = extract_evolutions(DECOMP)
    assert ev["1"] == [[4, 16, 2]]                       # EVO_LEVEL 16 -> Ivysaur
    assert len(ev["133"]) == 7 and ev["133"][0] == [25, 0, 470]   # Eevee: EVO_LEVEL_MOSS_ROCK -> Leafeon
def test_egg_moves():
    assert extract_egg_moves(DECOMP)["1"][:3] == [113, 130, 219]   # Light Screen, Skull Bash, Safeguard
def test_rom_egg_moves_match(vanilla_files, symbols, decomp_inter):   # tests/tools/rom/test_code_tables.py
    want = json.loads((decomp_inter / "egg_moves.json").read_text("utf-8"))
    assert egg_moves(vanilla_files, symbols) == {int(k): v for k, v in want.items()}
def test_rom_evolutions_match(vanilla_files, decomp_inter):
    want = json.loads((decomp_inter / "evolutions.json").read_text("utf-8"))
    assert read_evolutions(vanilla_files) == want
def test_learnsets_evolutions_and_egg_moves(learnsets):         # tests/data/test_learnsets.py
    assert learnsets.evolutions(1) == ((4, 16, 2),)
    assert learnsets.evolutions(3) == ()                         # Venusaur
    assert learnsets.egg_moves(1)[:3] == (113, 130, 219) and learnsets.egg_moves(3) == ()
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_extract_decomp.py tests/data/test_learnsets.py tests/tools/rom/test_code_tables.py -q -rs`
- [ ] **Step 3: Implement** both sides and the lookups.
- [ ] **Step 4: Rebuild, run Step 2's command plus `tests/tools/rom`, expect PASS** (round trip included, no skip).
- [ ] **Step 5: Commit** `feat(data): evolutions and egg moves`

### Task 8: Move tutors

**Files:**
- Modify: `tools/extract_decomp.py` (`extract_tutor_moves`, `extract_tutor_learnsets`, `_tutor_indices`), `tools/build_tables.py` (manifest `tutor_moves`, `tutor_learnsets`), `src/nuzline/data/learnsets.py`, `tools/rom/readers/code_tables.py`, `tools/extract_rom.py`
- Test: `tests/tools/test_extract_decomp.py`, `tests/data/test_learnsets.py`, `tests/tools/rom/test_code_tables.py`

**Interfaces:**
- Produces:
  - `tutor_moves.json`: a list in `res/pokemon/move_tutors.json` order of `{"move": id, "costs": {"red", "blue", "yellow", "green"}, "location": "route_212" | "survival_area" | "snowpoint_city"}` (`generated/tutor_locations.txt` stems). ROM: `code_tables.tutor_moves(rom, symbols) -> list[dict]` from `sTeachableMoves`, 12-byte `TeachableMove` (u16 move, four u8 costs, pad, u32 location; `include/tutor_movesets.h:7-14`).
  - `tutor_learnsets.json`: `{"<row>": [tutor index, ...]}` for every row, with `[]` for rows 0, 494 and 495 (`HasTutorSet`, `speciesproc.c:258`). ROM: `code_tables.tutor_learnsets(rom, symbols, n_tutor) -> dict[str, list[int]]` from `sSpeciesLearnsetsByTutor`: `(n_tutor + 7) // 8` bytes per moveset, moveset m at index m - 1; species s uses moveset s, and form row r >= 496 uses moveset `NATIONAL_DEX_COUNT + (r - 495)` (`include/constants/forms.h:6-19`, `scrcmd_move_tutor.c:211-259`).
  - `_tutor_indices(by_tutor: list[str], tutor_moves: list[str], path: Path) -> list[int]`: raises `ExtractError` naming `path` and the move for a move missing from the tutor list (Ruling 12).
  - `@dataclass(frozen=True) TutorMove(move: int, red: int, blue: int, yellow: int, green: int, location: str)` in `nuzline.data.learnsets`; `Learnsets.tutor_moves() -> tuple[TutorMove, ...]`, `Learnsets.can_learn_tutor(species, move, form=0) -> bool`.

- [ ] **Step 1: Write failing tests**

```python
def test_tutor_moves():                                          # tests/tools/test_extract_decomp.py
    t = extract_tutor_moves(DECOMP)
    assert len(t) == 38 and t[0] == {"move": 291, "costs": {"red": 2, "blue": 4, "yellow": 2, "green": 0}, "location": "route_212"}
    assert t[1]["move"] == 189 and t[1]["location"] == "survival_area"    # Mud-Slap
    assert t[-1]["move"] == 253 and t[-1]["location"] == "snowpoint_city"    # Uproar
    assert sorted(Counter(x["location"] for x in t).values()) == [8, 13, 17]
def test_tutor_learnsets_rotom_and_forms():
    tl, t = extract_tutor_learnsets(DECOMP), extract_tutor_moves(DECOMP)
    assert 1 in tl["479"] and len(tl["479"]) == 9                # Rotom: Mud-Slap (index 1) among its 9
    assert tl["504"] == tl["479"] and tl["0"] == tl["494"] == tl["495"] == []   # Wash = base list
def test_unlisted_tutor_move_raises():
    with pytest.raises(ExtractError, match="MOVE_TACKLE"):
        _tutor_indices(["MOVE_MUD_SLAP", "MOVE_TACKLE"], ["MOVE_DIVE", "MOVE_MUD_SLAP"], Path("x/data.json"))
def test_rom_tutor_tables_match(vanilla_files, symbols, decomp_inter):   # tests/tools/rom/test_code_tables.py
    want_m = json.loads((decomp_inter / "tutor_moves.json").read_text("utf-8"))
    want_l = json.loads((decomp_inter / "tutor_learnsets.json").read_text("utf-8"))
    assert tutor_moves(vanilla_files, symbols) == want_m
    assert tutor_learnsets(vanilla_files, symbols, len(want_m)) == want_l
def test_can_learn_tutor(learnsets):                              # tests/data/test_learnsets.py
    assert learnsets.can_learn_tutor(479, 189) is True             # Rotom, Mud-Slap
    assert learnsets.can_learn_tutor(479, 291) is False            # Rotom, Dive
    assert learnsets.can_learn_tutor(479, 189, form=2) is True     # Rotom-Wash
    assert learnsets.tutor_moves()[0] == TutorMove(291, 2, 4, 2, 0, "route_212")
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/test_extract_decomp.py tests/data/test_learnsets.py tests/tools/rom/test_code_tables.py -q -rs`
- [ ] **Step 3: Implement** both sides and the lookups.
- [ ] **Step 4: Rebuild, run Step 2's command plus `tests/tools/rom`, expect PASS** (round trip included, no skip).
- [ ] **Step 5: Commit** `feat(data): move tutors with costs and locations`

### Task 9: Manifest version 8 and its consequences

**Files:**
- Modify: `src/nuzline/data/tables.py` (`MANIFEST_VERSION = 8`, `check_manifest(manifest: dict, dir: Path) -> None` called by `load_tables` and `load_names` before the npz is opened), `src/nuzline/data/learnsets.py:62` (`load_learnsets` calls it), `tools/build_tables.py:59` (imports the constant), `data/gen4_platinum/manifest.json` (rebuilt), `benchmarks/README.md` (Ruling 15 note under "Milestone 4: exhaustive check"), `docs/decomp_paths.md` (every new source path from Tasks 2-8), `docs/design.md` §4.1 data-layer list, if it enumerates the intermediate files
- Test: `tests/data/test_tables.py`

**Interfaces:**
- Produces: `check_manifest` raises `ValueError(f"{dir}/manifest.json is version {v}; this nuzline needs {MANIFEST_VERSION}: re-run tools/extract_decomp.py and tools/build_tables.py")`, also when `version` is missing.

- [ ] **Step 1: Write failing tests**

```python
def test_manifest_version_is_8(names):  # the committed manifest; `names` skips when tables are absent
    assert json.loads((TABLES_DIR / "manifest.json").read_text("utf-8"))["version"] == MANIFEST_VERSION == 8
@pytest.mark.parametrize("manifest", ['{"version": 7}', "{}"])
def test_old_tables_are_refused(tmp_path, manifest):   # no npz needed: check_manifest runs first
    (tmp_path / "manifest.json").write_text(manifest)
    for load in (load_tables, load_names, load_learnsets):
        with pytest.raises(ValueError, match="re-run tools/extract_decomp.py"):
            load(tmp_path)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/data/test_tables.py -q -rs`
- [ ] **Step 3: Implement**, rebuild and commit the manifest.
- [ ] **Step 4: Full verification in the background** — `~/bin/uv.exe run --group rom pytest -q -rs`, `~/bin/uv.exe run ruff check src tests tools`, `~/bin/uv.exe run ruff format --check`. All must be clean. Read the skip summary: `test_vanilla_round_trip` and every `tests/tools/rom` test that uses the vanilla ROM must have run, not been skipped. `tests/prep/test_exhaustive.py` must pass on the new tables.
- [ ] **Step 5: Commit** `feat(data): table manifest version 8 (K2a schema), stale tables refused`
