# K3a: Kaizo ROM changes and the recording kit, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode and classify all 103 Kaizo ROM changes, teach coverage a per-fight
`not_reached` status, and make the golden-trace kit work on Kaizo, ending with the user's
10-minute address check.

**Architecture:** `data/kaizo/rom_diff.toml` gains a fourth status, `not_reached`, whose
`when` table is resolved at build time into concrete ids (moves, abilities, items, trainers)
and checked per fight by `engine/coverage.py`, for the fight and the pool. A local decoding
tool (`tools.rom.decode`, Capstone Thumb disassembly) produces the evidence that goes into
each entry's `note`; bytes never leave the machine. The trace converter learns
`--game kaizo` and `--save`, and a probe Lua script lets the user confirm the battle-RAM
addresses.

**Tech Stack:** Python 3.12 (uv), pytest, ndspy, Capstone (new, `rom` group), DeSmuME Lua.

**Spec:** `docs/superpowers/specs/2026-11-07-kaizo-k3-design.md` (§5, §7, §8 K3a).
Background: `docs/kaizo/pipeline.md`, `docs/kaizo/rom_diff.md`, `docs/tracing.md`.

## Global Constraints

- Never commit ROM bytes, disassembly listings, saves or raw traces. Notes hold names,
  addresses, decoded meanings and decomp citations only.
- Never copy or commit the save `<pre-Roark Kaizo save>`; read it by
  path through `kaizo_save_pre_roark` / `NUZLINE_KAIZO_SAVE_PRE_ROARK`.
- Never touch a folder named `backup` (any case, any depth).
- Vanilla output is unchanged: the vanilla suite passes and vanilla coverage text is
  byte-identical.
- Kaizo tables are rebuilt only with `uv run --group rom python -m tools.build_tables
  --game kaizo` (after `tools.extract_rom --game kaizo` when ROM intermediates change).
- `data/kaizo/*.toml` and `docs/kaizo/*.md` stay LF (`.gitattributes`).
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Commit on the working branch only; never push, never merge to main.
- A classification without a decoded note is not allowed (spec §5). An `irrelevant`
  claim must say why it cannot change a battle outcome or an AI choice.

## Review Focus

1. A `not_reached` condition matched only by a **pool** option (a box Pokémon's ability, a
   relearner move) must refuse that candidate in `check_pool`, not pass silently: Task 3.
2. A `when` naming something that does not resolve (`ability = ["Dry skin"]`, a move
   effect no move uses is fine, an unknown key) must fail the build, not match nothing:
   Task 2.
3. A fight whose `format` is `None` (partner present, format unknown) must count as
   matching every `format` condition: Task 3.
4. Vanilla manifests carry no conditions and coverage prints no ROM-change lines for
   vanilla: Task 3.
5. An entry downgraded from `not_reached` back to `unknown`, or a stale entry after the
   patch changes, still counts and still fails the build as today: Task 1.

---

### Task 1: The `not_reached` status and its `when` grammar

**Files:**
- Modify: `tools/rom/diff.py` (`STATUSES`, `load_classification`,
  `load_member_classification`, `classification_skeleton` header comment)
- Test: `tests/tools/rom/test_diff_report.py`

**Interfaces:**
- Produces: `STATUSES = ("modelled", "irrelevant", "not_reached", "unknown")`;
  `WHEN_KEYS = ("format", "area_condition", "move_effect", "ability", "held_item",
  "map_weather", "trainer_items")`;
  `load_conditions(path: Path) -> dict[str, dict[str, object]]`, keyed by label
  (`"<module> <symbol>"` for `[[code]]`, `"<narc>#<index>"` for `[[member]]`), holding
  each `not_reached` entry's validated `when`.

Grammar (all values lists unless noted): `format` ⊆ {"double", "double_two_trainer",
"multi"}; `area_condition` ⊆ {"trick_room", "gravity"}; `move_effect`: ints (the ROM
battle effect id, = `be_seq` member); `ability`: names; `held_item`: names;
`map_weather`: ints (the raw map-header weather byte); `trainer_items`: `true`. Several
keys mean OR.

- [ ] **Step 1: Write the failing tests**

```python
def test_not_reached_needs_when(tmp_path):
    p = _toml(tmp_path, '[[code]]\nmodule="overlay14"\nsymbol="X"\nstatus="not_reached"\nnote="n"\n')
    with pytest.raises(ValueError, match="not_reached needs a when"):
        load_conditions(p)

def test_when_only_on_not_reached(tmp_path):
    p = _toml(tmp_path, '[[code]]\nmodule="o"\nsymbol="X"\nstatus="irrelevant"\nnote="n"\n'
                        'when={format=["double"]}\n')
    with pytest.raises(ValueError, match="when is only allowed on not_reached"):
        load_conditions(p)

def test_when_rejects_unknown_key_and_value(tmp_path):
    for bad in ('{weather=["sun"]}', '{format=["triple"]}', '{trainer_items=false}', '{}'):
        p = _toml(tmp_path, f'[[member]]\nnarc="battle/skill/be_seq.narc"\nindex=88\n'
                            f'status="not_reached"\nnote="n"\nwhen={bad}\n')
        with pytest.raises(ValueError):
            load_conditions(p)

def test_conditions_keyed_by_label(tmp_path):
    p = _toml(tmp_path,
        '[[code]]\nmodule="overlay14"\nsymbol="TrainerAI_MainDoubles"\nstatus="not_reached"\n'
        'note="n"\nwhen={format=["double","double_two_trainer","multi"]}\n'
        '[[member]]\nnarc="battle/skill/be_seq.narc"\nindex=88\nstatus="not_reached"\n'
        'note="n"\nwhen={move_effect=[88]}\n')
    assert load_conditions(p) == {
        "overlay14 TrainerAI_MainDoubles": {"format": ["double", "double_two_trainer", "multi"]},
        "battle/skill/be_seq.narc#88": {"move_effect": [88]},
    }

def test_not_reached_kept_by_code_loader(tmp_path):
    p = _toml(tmp_path, '[[code]]\nmodule="overlay14"\nsymbol="X"\nstatus="not_reached"\n'
                        'note="n"\nwhen={format=["multi"]}\n')
    assert load_code_classification(p, [["overlay14", "X"]]) == {
        ("overlay14", "X"): ("not_reached", "n")}
```

(`_toml` writes the text to `tmp_path / "c.toml"` and returns the path; add it beside the
existing helpers.)

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/tools/rom/test_diff_report.py -k "not_reached or when or conditions" -v`
Expected: FAIL (`load_conditions` not defined).

- [ ] **Step 3: Implement** `STATUSES`, `WHEN_KEYS`, `load_conditions` in
  `tools/rom/diff.py`. Both existing loaders accept the new status; `load_conditions`
  re-reads the same file and does all `when` validation, so the existing loaders keep
  their `(status, note)` return type. Update the skeleton's comment line to list the four
  statuses.

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/tools/rom -v`
Expected: PASS.

- [ ] **Step 5: Commit** `feat(rom): not_reached status with a checked when condition`

### Task 2: Resolve conditions at build time

**Files:**
- Modify: `tools/build_tables.py` (`_kaizo_rom_changes`, `build_arrays` manifest block)
- Modify: `src/nuzline/data/tables.py` (`Names.rom_change_conditions`)
- Test: `tests/kaizo/test_kaizo_tables.py`, `tests/data/test_names_per_game.py`

**Interfaces:**
- Consumes: `load_conditions` (Task 1).
- Produces: manifest key `rom_change_conditions`: a list of
  `{"label": str, "formats": [str], "areas": [str], "moves": [int], "abilities": [int],
  "items": [int], "trainers": [int], "trainer_items": bool}`, sorted by label. `moves` are
  the move ids whose ROM `effect` is in `move_effect`; `abilities` and `items` are decomp
  ids resolved by exact name (case-insensitive) through the Kaizo names; `trainers` are the
  trainer ids placed on a map whose header weather byte is in `map_weather` (from the
  map rows `extract_rom` already reads). `Names.rom_change_conditions:
  tuple[RomCondition, ...]`, with `RomCondition` a frozen dataclass of those fields
  (tuples), defined in `nuzline.data.tables`. Absent key → `()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_conditions_resolve_names(tmp_path, kaizo_inter):
    toml = ('[[member]]\nnarc="battle/skill/be_seq.narc"\nindex=88\nstatus="not_reached"\n'
            'note="n"\nwhen={move_effect=[88], ability=["Dry Skin"]}\n')
    conds = resolve_conditions(kaizo_inter, _write(tmp_path, toml))
    (c,) = conds
    assert c["label"] == "battle/skill/be_seq.narc#88"
    assert MOVE_TRIPLE_AXEL_SLOT in c["moves"]          # slot 149, kaizo-data-design §5.10
    assert c["abilities"] == [ABILITY_DRY_SKIN]

def test_unresolved_name_fails_build(tmp_path, kaizo_inter):
    toml = ('[[member]]\nnarc="battle/skill/be_seq.narc"\nindex=88\nstatus="not_reached"\n'
            'note="n"\nwhen={ability=["Dry skinn"]}\n')
    with pytest.raises(BuildError, match="Dry skinn"):
        resolve_conditions(kaizo_inter, _write(tmp_path, toml))

def test_names_reads_conditions(vanilla_manifest):
    n = Names({**vanilla_manifest, "rom_change_conditions": [
        {"label": "x", "formats": ["double"], "areas": [], "moves": [], "abilities": [],
         "items": [], "trainers": [], "trainer_items": False}]})
    assert n.rom_change_conditions[0].formats == ("double",)
    assert Names(vanilla_manifest).rom_change_conditions == ()
```

`kaizo_inter` is the existing Kaizo-intermediates fixture of `tests/kaizo/conftest.py`
(skips when the local inputs are absent).

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/kaizo/test_kaizo_tables.py tests/data/test_names_per_game.py -k condition -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `resolve_conditions(inter: Path, rom_diff: Path = ROM_DIFF)
  -> list[dict]` in `tools/build_tables.py`, called from the Kaizo branch of
  `build_arrays`, writing `manifest["rom_change_conditions"]`; and `RomCondition` plus the
  `Names` field. The unclassified count stays "unknown only".

- [ ] **Step 4: Run** `uv run pytest tests/kaizo tests/data -v` — PASS.

- [ ] **Step 5: Commit** `feat(kaizo): resolve not_reached conditions into the manifest`

### Task 3: Coverage refuses only the fights a condition reaches

**Files:**
- Modify: `src/nuzline/engine/coverage.py` (new `rom_change_refusals`; call it in
  `_check_trainer`'s Kaizo branch beside the unclassified line; module docstring)
- Modify: `src/nuzline/prep/pool.py` (`check_pool` refuses a candidate a condition
  reaches)
- Test: `tests/engine/test_coverage.py`, `tests/prep/test_pool.py`

**Interfaces:**
- Consumes: `Names.rom_change_conditions` (Task 2).
- Produces: `rom_change_refusals(tables: Tables, names: Names, trainer: Trainer,
  specs: Sequence[PokemonSpec], extra_moves: Sequence[int] = ()) -> list[str]` — the
  labels of conditions that hold. A condition holds when any of: `trainer.format` in
  `formats`, or `trainer.format is None` and `formats` is non-empty; `trainer.area_condition`
  has a listed area; any spec's move (or `extra_moves`) in `moves`; any spec's decomp
  ability (`tables.species_abilities_decomp` at its personal row and ability slot) in
  `abilities`; any spec's `item` in `items`; `trainer.id` in `trainers`;
  `trainer_items` and any nonzero `trainer.items`. Coverage line:
  `f"Kaizo ROM: {label} is not modelled and this fight reaches it (docs/kaizo/rom_diff.md)"`,
  kind `"ROM change"`.

- [ ] **Step 1: Write the failing tests** (synthetic `Names` with one condition each, on
  the vanilla tables, the way `test_coverage.py:375` builds `kaizo`):

```python
@pytest.mark.parametrize("cond, reached", [
    ({"formats": ["double"]}, False),                 # Roark is single
    ({"areas": ["gravity"]}, False),
    ({"moves": [MOVE_ROCK_TOMB]}, True),              # a vanilla Roark move
    ({"trainer_items": True}, True),                  # vanilla Roark has Potions
    ({"trainers": [246]}, True),
])
def test_rom_change_refusals(tables, vanilla_manifest, roark, roark_party, sample, cond, reached):
    names = _kaizo_names(vanilla_manifest, [{**EMPTY_COND, "label": "c", **cond}])
    got = rom_change_refusals(tables, names, roark, sample + roark_party)
    assert got == (["c"] if reached else [])

def test_unknown_format_matches_format_conditions(tables, vanilla_manifest, roark, sample):
    names = _kaizo_names(vanilla_manifest, [{**EMPTY_COND, "label": "c", "formats": ["multi"]}])
    assert rom_change_refusals(tables, names, dataclasses.replace(roark, format=None), sample) == ["c"]

def test_vanilla_has_no_rom_lines(tables, vanilla_names, roark, sample, roark_party):
    assert rom_change_refusals(tables, vanilla_names, roark, sample + roark_party) == []

def test_pool_candidate_reached_by_relearner_move(tables, vanilla_manifest, roark, candidate):
    # candidate.relearner holds MOVE_X; the fight's own specs do not
    names = _kaizo_names(vanilla_manifest, [{**EMPTY_COND, "label": "c", "moves": [MOVE_X]}])
    report = check_pool(tables, names, [candidate], trainer=roark)
    assert report.missing == [f"{candidate.label}: Kaizo ROM: c is not modelled and this "
                              "fight reaches it (docs/kaizo/rom_diff.md) (ROM change UNSUPPORTED)"]
```

(`EMPTY_COND` has every list empty and `trainer_items: False`; `_kaizo_names` builds a
`Names` with `game = KAIZO`, `unclassified_rom_changes = 0` and the conditions.)

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/engine/test_coverage.py tests/prep/test_pool.py -k "rom_change or reached or rom_lines" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `rom_change_refusals` and wire it: `_check_trainer` (fight:
  `player + opp` specs) and `check_pool` (per candidate: `[cand.spec]` with
  `extra_moves = cand.relearner + tm moves`). `check_pool` gains `trainer: Trainer | None
  = None`; `cli._print_pool` passes it. The pool only evaluates spec-based terms.

- [ ] **Step 4: Run** `uv run pytest tests/engine tests/prep -v` and the vanilla
  coverage text check: `uv run nuzline coverage --trainer roark` output identical to
  before (diff against a capture taken before Step 3). PASS / no diff.

- [ ] **Step 5: Commit** `feat(coverage): refuse a Kaizo fight only where a not_reached change applies`

### Task 4: The decoding tool

**Files:**
- Modify: `pyproject.toml` (`rom` group: `capstone>=5,<6`), `uv.lock`
- Create: `tools/rom/decode.py`
- Test: `tests/tools/rom/test_decode.py`

**Interfaces:**
- Consumes: `tools.rom.inputs.load_inputs`, `tools.rom.patch.kaizo_rom`,
  `tools.rom.nds.RomFiles`, `tools.rom.symbols.SymbolMap`, `tools.rom.diff.diff_roms`.
- Produces: `decode_code(module: str, symbol: str) -> list[RangeDecode]` with
  `RangeDecode(start: int, end: int, vanilla: list[str], patched: list[str])` (Thumb
  disassembly lines of the whole instructions covering the range, ±4 instructions of
  context); `decode_member(narc: str, index: int) -> tuple[list[str], list[str]]` (battle
  script commands, vanilla and patched, via the existing script reader, else hex words);
  `disassemble_at(addr: int, count: int) -> list[str]` (injected code). CLI:
  `python -m tools.rom.decode code <module> <symbol> | member <narc> <index> | at <hex> <n>`
  prints to stdout only.

- [ ] **Step 1: Write the failing tests** (skip without the ROM, like `test_diff_kaizo.py`):

```python
def test_decode_trainer_ai_init(kaizo_rom_inputs):
    (r,) = decode_code("overlay14", "TrainerAI_Init")        # 1 range, 1 byte (rom_diff.md)
    assert r.vanilla != r.patched and len(r.vanilla) == len(r.patched)

def test_decode_member_be_seq_182(kaizo_rom_inputs):
    van, pat = decode_member("battle/skill/be_seq.narc", 182)
    assert van != pat

def test_no_bytes_written(tmp_path, kaizo_rom_inputs, monkeypatch):
    monkeypatch.chdir(tmp_path); main(["code", "overlay14", "TrainerAI_Init"])
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run them and see them fail.** `uv run --group rom pytest tests/tools/rom/test_decode.py -v` — FAIL.
- [ ] **Step 3: Implement** with Capstone `CS_ARCH_ARM, CS_MODE_THUMB` (the ARM-mode
  fallback when the symbol map marks a function ARM).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(rom): local decoding tool for Kaizo code and script changes`

### Tasks 5–8: Decode and classify

Every entry gets a `note` of the form
`"decoded: <what changed> (<vanilla> -> <patched>); <decomp citation>; <why this status>"`.
`not_reached` entries get a `when`. An entry whose behaviour K3b or K3c must model stays
`unknown`, and its note ends with `"-> K3b"` or `"-> K3c"`. Each task runs
`python -m tools.rom.diff` to regenerate `docs/kaizo/rom_diff.md`, then
`tools.build_tables --game kaizo`, and adds its group to the data test below.

**Shared test** (create in Task 5): `tests/kaizo/test_rom_classification.py`, data-only
(reads `data/kaizo/rom_diff.toml`, no ROM):

```python
CITE = re.compile(r"(\S+\.(c|s|h|json|toml):\d+|rom:\S+)")

def test_every_classified_entry_is_decoded():
    for e in _entries():
        if e["status"] != "unknown" or e["note"]:
            assert e["note"].startswith("decoded: "), e
            assert CITE.search(e["note"]), e

def test_unknown_entries_name_their_milestone():
    for e in _entries():
        if e["status"] == "unknown" and e["note"]:
            assert e["note"].endswith(("-> K3b", "-> K3c")), e

@pytest.mark.parametrize("group", DONE_GROUPS)   # each task appends its group
def test_group_has_no_undecoded_entries(group): ...
```

#### Task 5: arm9 field and menu code, overlay5, overlay13

**Files:** Modify `data/kaizo/rom_diff.toml`, `docs/kaizo/rom_diff.md` (regenerated);
create the shared test.

Entries: the 23 `MapHeader_*` getters, `NitroMain`, `Options_Init`, the `ScrCmd_*`,
`0x0204EB22-0x0204EB24`, `EncEffects_GetEffectPair`, `Trainer_Encounter`,
`PCBoxes_UnlockWallpaper`, `sub_0207AD40`, `OS_GetInitArenaLo`, `sSpawnLocations`,
`gHiddenItems`, `gFieldScriptCommands`, `sEncEffectsTable`, `sTrainerEncounterBGMs`,
`sItemArchiveIDs`, `@29`, and overlay13's 13 cursor routines. `sTMHMMoves` (arm9) and
overlay5's `sTeachableMoves`/`sSpeciesLearnsetsByTutor` are `modelled` only after a test
shows the K2b TM and tutor tables equal the decoded tables
(`test_tm_table_matches_rom`, `test_tutor_table_matches_rom` in
`tests/tools/rom/test_move_lists.py`); otherwise `unknown -> K3b`.

- [ ] Step 1: write the shared test plus the two table tests; run; see the group test
  and the table tests fail.
- [ ] Step 2: decode each entry with `tools.rom.decode`; write the notes and statuses.
- [ ] Step 3: regenerate the report, rebuild Kaizo tables, run `uv run pytest tests/kaizo tests/tools/rom -v` — PASS.
- [ ] Step 4: commit `docs(kaizo): classify the arm9 field, overlay5 and overlay13 changes`.

#### Task 6: The injected code, its callers and the party builder

Entries: `PartyMenuCB_LevelUp`, `HandleInput_Main`, `SetMonDataFromMon` (their BLs go to
`weather_sys.narc` member 9, `rom_diff.md` "Injected code"), `TrainerData_BuildParty`.

- [ ] Step 1: disassemble the three injected targets (`decode at 0x023C82E2 64`, `…452`,
  `…482`) and follow their calls until each routine's effect is named. Write the finding
  to the shared notes folder (`kaizo-k3a/injected-code.md`) in prose, no bytes.
- [ ] Step 2: classify. If the injected code touches battle-relevant Pokémon data
  (stats, level, moves, ability, nature) outside the menus, the callers are
  `unknown -> K3b` and the finding names exactly what K3b must model; else `irrelevant`
  with the reason. `TrainerData_BuildParty` is `modelled`, citing `rom_diff.md`
  "TrainerData_BuildParty, decoded" and the K2b party test
  (`tests/tools/rom/test_kaizo_trainer_party.py`).
- [ ] Step 3: regenerate, rebuild, test (as Task 5) — PASS.
- [ ] Step 4: commit `docs(kaizo): classify the injected level-up code and the party builder`.

#### Task 7: overlay16 and the battle script members

Entries: `BattleControllerPlayer_CheckObedience`, `…_CheckMoveHitAccuracy`,
`…_CheckMoveHitOverrides`, `BattleSystem_TriggerEffectOnSwitch`, `sTrainerClassPrizeMul`;
`be_seq` 48, 88, 104, 141, 182; `sub_seq` 238, 251, 297.

- [ ] Step 1: decode each. For the four battle routines name every changed branch and
  constant and the moves, abilities or weathers it affects.
- [ ] Step 2: classify. `sTrainerClassPrizeMul`: `irrelevant` (prize money). Battle
  routines on Roark's path: `unknown -> K3b`, with the note listing what to model. A
  change reachable only under a condition gets `not_reached` with that `when` (for
  example `map_weather = [12]` for `sub_seq` 297 if decoding confirms it is reached only
  from weather 12; `area_condition = ["gravity"]` for 238, `["trick_room"]` for 251;
  `move_effect = [N]` for each `be_seq` member N whose users are not in the pool).
  A `be_seq` member a pool or Roark move uses stays `unknown -> K3b`.
- [ ] Step 3: regenerate, rebuild, test — PASS. Check `uv run nuzline coverage --game
  kaizo --save <save> --trainer roark` no longer lists any change Task 7 made
  `not_reached` (the unclassified count drops accordingly).
- [ ] Step 4: commit `docs(kaizo): classify overlay16 and the changed battle scripts`.

#### Task 8: overlay14, the trainer AI

Entries: the 32 overlay14 changes.

- [ ] Step 1: first establish where the AI script bytes live in the ROM (overlay14 data
  or `tr_ai_seq.narc`) and record whether Kaizo changed them; this decides how K3c
  builds the Kaizo policy. Then decode each routine: for the many 1–2 byte changes, name
  the constant (a move id, effect id, type id, item effect, threshold) and its vanilla
  and Kaizo value.
- [ ] Step 2: classify. `TrainerAI_MainDoubles`, `AICmd_CheckIfHighestDamageWithPartner`
  and other doubles-only paths: `not_reached`, `when = {format = ["double",
  "double_two_trainer", "multi"]}`. `TrainerAI_ShouldUseItem`: `not_reached` with
  `trainer_items = true` if its changes affect bag use only (Kaizo Roark has no bag
  items). Everything Roark's singles decision can run: `unknown -> K3c`, the note naming
  the constant K3c must use. Write a summary table to the shared notes folder
  (`kaizo-k3a/overlay14.md`) for K3c's plan.
- [ ] Step 3: regenerate, rebuild, test — PASS.
- [ ] Step 4: commit `docs(kaizo): decode and classify the Kaizo trainer AI changes`.

### Task 9: The recording kit for Kaizo

**Files:**
- Modify: `tools/trace_to_state.py` (`--game`, `--save`; `_trainer` and `main` use the
  game's intermediates and tables)
- Create: `tools/probe_battle.lua`, `docs/tracing_kaizo.md`
- Create: `tools/rom/offsets.py` (static check of the battle-RAM offsets)
- Test: `tests/test_trace_to_state.py`, `tests/tools/rom/test_offsets.py`

**Interfaces:**
- Produces: `bundle_trace(raw, party, names, trainer_id, game="platinum_us")` where
  `party` is a TOML path or a `SaveFile`; `trace_to_state --game kaizo --save <path>`
  builds the party from the save's party (`SaveMon.to_spec`), so the user never edits a
  TOML. `check_offsets(game: str) -> list[OffsetCheck]`, `OffsetCheck(field: str,
  offset: int, found: tuple[int, ...], ok: bool)`: for each `CTX`/`MON`/`SYS` field of
  `tools/trace_battle.lua`, the immediate offsets of the load/store instructions in the
  decomp functions known to touch that field (listed in `offsets.py` with citations,
  e.g. `totalTurns` in `BattleController_TurnEnd`), disassembled from that game's ROM.

- [ ] **Step 1: Write the failing tests**

```python
def test_offsets_vanilla_and_kaizo(rom_inputs):
    for game in ("platinum_us", "platinum_kaizo_nightmare_final"):
        bad = [c for c in check_offsets(game) if not c.ok]
        assert bad == [], bad          # a failure names the field to fix in trace_battle.lua

def test_bundle_from_save(kaizo_save, kaizo_names):
    b = bundle_trace(RAW_FIXTURE, load_save(kaizo_save, game=KAIZO), kaizo_names, 246, KAIZO)
    assert [m["species"] for m in b["party"]] == [SANDSHREW, FLAAFFY, SHINX, VULPIX, GOLDEEN, MEOWTH]
```

`RAW_FIXTURE` is a two-record synthetic raw trace in `tests/fixtures/`.

- [ ] **Step 2: Run them; see them fail.**
- [ ] **Step 3: Implement** `check_offsets`, the `trace_to_state` options, then
  `tools/probe_battle.lua`: locate the structures exactly as `trace_battle.lua` does,
  then each turn print one line per VERIFY field with its address, value and what it
  should be (`totalTurns` = turn number from 0; `fieldConditionsMask` nonzero in
  sandstorm, 0 otherwise; `damage` = −(HP lost) at `UPDATE_HP`; `selectedPartySlot[1]`
  = the opponent's party slot; the party decode vs the summary). It writes
  `traces/probe.txt` and records nothing else. If `check_offsets` found a wrong offset,
  fix `trace_battle.lua`'s tables and `docs/tracing.md` in this step.
- [ ] **Step 4: Write `docs/tracing_kaizo.md`**: the address check (Task 10) and the
  K3d recording, as numbered steps a player can follow. It covers: making a save-state
  copy so the real run is never touched; opening the Lua console; setting
  `LOCATE_SPECIES`/`LOCATE_LEVEL` for the lead (Sandshrew 27, level 16 in this save); any
  battle for the probe; where `traces/probe.txt` lands; what to send back.
- [ ] **Step 5: Run** `uv run --group rom pytest tests/test_trace_to_state.py tests/tools/rom/test_offsets.py -v` — PASS.
- [ ] **Step 6: Commit** `feat(trace): Kaizo recording kit: offsets check, save party, probe script`

### Task 10: The user's address check (manual)

- [ ] **Step 1:** Ask the user to follow `docs/tracing_kaizo.md` "Address check" and send
  `traces/probe.txt` (or paste it). Stop until they do.
- [ ] **Step 2:** Compare each line to its expectation. For any wrong field, fix the
  offset in `tools/trace_battle.lua`, `tools/probe_battle.lua` and `docs/tracing.md`
  together, add the correction and its evidence to `docs/tracing.md`, and ask for one
  re-run of the probe only if a fix was needed.
- [ ] **Step 3:** Mark each VERIFY row in `docs/tracing.md` as "verified 2026-MM-DD
  (Kaizo probe)". Commit `docs(trace): battle-RAM offsets verified on Kaizo`.

### Task 11: Close K3a

**Files:** Modify `docs/kaizo/pipeline.md` (Ruling 13 text: the four statuses, `when`,
per-fight refusal; "What is refused" for Roark now), `docs/superpowers/specs/
2026-11-07-kaizo-k3-design.md` (§5: any group whose decoding moved work, one line each).

- [ ] **Step 1:** `uv run --group rom python -m tools.build_tables --game kaizo`, then
  `uv run pytest` — the whole suite passes (1870 + this plan's tests).
- [ ] **Step 2:** `uv run nuzline coverage --game kaizo --save <save> --trainer roark`:
  the ROM-change lines now name only `unknown -> K3b/K3c` entries; record the count and
  the list in the shared notes folder (`kaizo-k3a/handover.md`) for the K3b and K3c
  plans.
- [ ] **Step 3:** Update the docs above; commit `docs(kaizo): K3a done: ROM changes classified, kit verified`.
