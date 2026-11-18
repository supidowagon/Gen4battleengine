# K1 ROM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the user's local vanilla ROM and Kaizo patch: hash-checked inputs, the patched Kaizo ROM (cached, never committed), both ROMs' file systems unpacked, a committed change inventory `docs/kaizo/rom_diff.md` mapping every changed code range to a decomp symbol (all initially "unknown"), and `tools/extract_rom.py`, whose vanilla output reproduces `tools/extract_decomp.py`'s `data/intermediate/` exactly.

**Architecture:** A new package `tools/rom/`: `inputs.py` (local config, hashes), `vcdiff.py` (pure-Python VCDIFF + xdelta3 extensions), `patch.py` (apply and cache), `nds.py` (ndspy wrapper: FAT/FNT, NARCs, ARM9, overlays), `symbols.py` (the decomp build's linker map), `diff.py` (file, NARC-member and code diff, report, classification), `readers/` (one module per table, layouts from the decomp's `include/` structs), and the CLI `tools/extract_rom.py`. Nothing in `src/nuzline` changes. The decomp is built once in WSL to get the map file.

**Tech Stack:** Python 3.12, stdlib `lzma`/`zlib`/`hashlib`/`tomllib`/`difflib`, ndspy 4.x (new `rom` dependency group), pytest. Run everything with `~/bin/uv.exe run --group rom ...`. WSL Ubuntu for the one decomp build.

**Spec:** `docs/superpowers/specs/2026-11-03-kaizo-data-design.md` — §4 (sources), §6.1 (inputs), §6.2 (ROM diff), §6.3 (extraction, vanilla round trip), §8 (git-ignored generated data), §11 (K1). Read it before any task; §N below are the spec's.

## Global Constraints

- Vanilla ROM SHA-1 `0862ec35b24de5c7e2dcb88c9eea0873110d755c` (`CPUE`, rev 1); patch SHA-256 `b787e743b5fa78e1b31bae9277e656482b7248b82e490a6e95236606ae92f433`. Any mismatch stops the build (§6.1).
- Never commit ROM bytes, NARC members, code, the patched ROM, the linker map, or anything that only makes sense with the ROM (§2 anti-goals, §6.1.3). Committed outputs: `docs/kaizo/rom_diff.md`, `data/kaizo/rom_diff.toml`, `data/kaizo/rom.toml`, `data/local.example.toml`, a hand-cut linker-map excerpt (names and addresses only) as a test fixture.
- Runtime package `nuzline` gains no dependency. `pyproject.toml` gets a dependency group `rom = ["ndspy>=4.1,<5", "openpyxl>=3.1,<4"]` (ndspy 4.2.0, 2024-09-14, and openpyxl 3.1.5 are the current releases). Everything under `tools/rom` imports ndspy inside functions; tests that need it use `pytest.importorskip("ndspy")`.
- Tests that need a local input (ROM, patch, linker map, decomp) skip when it is absent, like `tests/tools/test_extract_decomp.py` (decomp) and `tests/conftest.py` (tables).
- New files only, plus `pyproject.toml`, `uv.lock`, `.gitignore`. Do not edit `tools/extract_decomp.py`, `tools/build_tables.py`, `tools/effect_map.py` or anything in `src/` (they change in K2a, after milestone 4 merges).
- The decomp (`external/pokeplatinum`, commit `579d2d5c65e736a1a4268fb7acbfd70712177c6c`, the manifest's `decomp_commit`) is authoritative for layouts: cite `file:line` in docstrings as `tools/extract_decomp.py` does.
- **Worktree setup (before Task 1):** `external/` is git-ignored, so a fresh K1 worktree cut from main has no decomp and every round-trip test would skip quietly. Give it one: from the worktree root in PowerShell, `New-Item -ItemType Junction -Path external\pokeplatinum -Target "<repo>\external\pokeplatinum"` (the main checkout's copy, as the `strange-feistel` worktree has), or run `bash tools/clone_decomp.sh`. Check: `git -C external/pokeplatinum rev-parse HEAD` prints the pinned commit.
- Fail loudly: any byte a reader cannot account for raises with the file, NARC member and offset; nothing is skipped or defaulted silently.
- Never touch a folder named `backup`. ruff `line-length = 100`.

## Rulings (planner's calls; the spec is silent, or the inputs say otherwise)

1. **The patched ROM's game code is `JAK7`, not `CPUE`.** A throwaway prototype decoder (all 16 windows' Adler-32 checksums verified) gives a 134217728-byte ROM with SHA-1 `e8d13c6504406324a4a9ea9daf8d6bd409f8a644`, title `POKEMON PL`, maker `01`, revision 1, game code `JAK7`. The task text expected `CPUE`. Task 3's test pins title, maker and revision to vanilla's and the game code and SHA-1 to `data/kaizo/rom.toml`, which records the measured values with this note; **tell the user** when Task 3 lands. If the real implementation disagrees with the prototype, stop and report; do not edit `rom.toml` to match.
2. **xdelta3 secondary compression (measured on this patch):** header indicator `0x05` (VCD_SECONDARY | VCD_APPHEADER), compressor id 2 (LZMA). There is one xz stream per section kind (data, instructions, addresses), continued across windows and never finished: a compressed section is a varint of its decoded length followed by the next chunk of that stream; only window 0's chunks start with the xz magic `FD 37 7A 58 5A 00`. Decode with one `lzma.LZMADecompressor(format=lzma.FORMAT_XZ)` per section kind for the whole file; each chunk must yield exactly the stated length. A section is secondary-compressed only when its bit is set in that window's Delta_Indicator (`0x01` data, `0x02` instructions, `0x04` addresses); a clear bit means the section is stored raw and its decompressor is not touched. If a section's decompressor has reached `eof` and another compressed chunk arrives, a new decompressor is started (this patch never does it; the test writer's finished streams do). Windows: 16 × 8 MiB, indicators `0x05` (VCD_SOURCE | VCD_ADLER32) and `0x04` (VCD_ADLER32, target-only); the Adler-32 is 4 bytes big-endian after the three section lengths, over the decoded target window. Default code table only (header bit `0x02` → `VcdiffError`). Prototype runtime ~12 s; the cache makes it a one-off.
3. **Symbols come from `build/main.nef.xMAP`** (the Metrowerks linker's map, from `-map closure,unused` at `meson.build:99`; the executable is `main.nef`, see `.github/calcrom/calcrom.py:40`). Overlay names map to overlay ids through the linker-generated `SDK_OVERLAY_<name>_ID` symbols in the same map (what `FS_OVERLAY_ID(battle)` resolves to, `src/unk_0203D1B8.c:238`). If the build produces no `.xMAP`: run `arm-none-eabi-readelf -SsW build/main.nef > main.nef.readelf.txt` and `ninja -C build overlay.map` (`meson.build:292-301`, `tools/debug/overlay_mapper.py`), add `SymbolMap.from_readelf(readelf: Path, overlay_map: Path)`, and pass the same Task 5 tests. If neither attributes symbols to overlays well enough for those tests, stop and report.
4. **The build runs in the WSL Linux filesystem**, not on `/mnt/g` (`INSTALL.md:66-69`), from a fresh GitHub clone at the pinned commit. The map and `main.lsf` are copied to `<repo>/.cache/decomp/579d2d5c65e736a1a4268fb7acbfd70712177c6c/` (git-ignored by `.cache/`). The `apt install` step needs the user's sudo password: the user runs it. Do not type it.
5. **Config precedence:** environment variable > `data/local.toml` > absent. Relative TOML paths resolve against `REPO_ROOT`; environment paths are used as given. `vanilla_rom` is required; the others may be absent (`None`). A TOML parse error is re-raised as `RomInputError` saying "use single-quoted literal strings for Windows paths".
6. **Patched ROM cache:** `<cache_dir>/kaizo-<patch sha256[:16]>.nds` plus a sidecar `kaizo-<patch sha256[:16]>.nds.sha1` (the file's SHA-1 as hex). Written to a temporary name and renamed. On a hit the file is re-hashed and must equal the sidecar, otherwise it is rebuilt. After building or loading, the SHA-1 must equal `data/kaizo/rom.toml` `sha1` when that file exists.
7. **Code diff:** modules (ARM9 sections with their RAM addresses from ndspy's `MainCodeFile.sections`, ITCM/DTCM included; each overlay at its y9 RAM address) are aligned with `difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)` over 4-byte words (measured: 1.0 s on overlay 16 against 33 s for halfwords), then non-equal blocks are narrowed to bytes. Equal-length modules skip alignment and compare positionally. Change ranges closer than 4 equal bytes merge; a range is split at symbol boundaries so each `CodeChange` names one symbol (or none). **Relocation-only changes** are counted per symbol, not listed: a changed Thumb `BL`/`BLX` pair, ARM `BL`, or aligned 32-bit word whose vanilla target address, mapped through the alignment, equals the patched target. Measured scale: overlay 16 grew by 0x20 bytes, and about 4,200 raw changed blocks were found there, many expected to be relocations; ARM9 1447 bytes, overlay 5 171, overlay 13 27, overlay 14 114.
8. **Loose files are ignored by readers.** The Kaizo ROM adds 1751 FNT entries (`msgdata/pl_msg/NNNN` ×725, `poketool/trgra/...` ×525, `wazaeffect/wearc/...` ×501, apparently left over from an editor's unpacked NARCs) and changes 38 named files. Readers read the NARCs the game opens (`platinum.us/filesys.csv` paths). The report lists added files collapsed per folder.
9. **Species weight** is member 1 of `application/zukanlist/zkn_data/zukan_data_gira.narc` (Altered Giratina, matching `species.json`), one u32 per species `0..493` (`speciesproc.c:963-976` `PackDexData`, `PackField` at `:903`). Ids 494 and 495 (Egg, Bad Egg) have no dex entry and get 0, as `species.json` has.
10. **Unused items** (`ITEM_UNUSED_*` in `generated/items.txt`) are emitted exactly as `extract_items` does (constant name, `HOLD_EFFECT_NONE`, 0). For any other id, the name comes from `TEXT_BANK_ITEM_NAMES` and the data from `pl_item_data.narc` member `sItemArchiveIDs[id].dataID` (`src/item.c:48-57`; item 0 and `ITEM_RETURN_ID` use member 0).
11. **Trainer bag:** the header's four `items` slots with zeros dropped (`trainers.json` has no 0 entries; checked).
12. **Start conditions from the ROM:** a trainer's maps come from `zone_event.narc` object events whose script is `3000 + id - 1` or `5000 + id - 1` (`tools/jsoncnv/convert.py:79-83`), from the battle commands reached by walking each map's `scr_seq.narc` scripts by control flow, and from `gVsSeekerRematchData` (overlay 5). Weather from `sMapHeaders` and `sYearlyWeather` (ARM9). The condition logic, `WEATHER_CONDITION` and `tools/start_conditions.json` are imported from `tools/extract_decomp.py` and applied the same way (§6.3). Every trainer operand of a reached battle command counts, as the decomp's regex does: both of `StartTrainerBattle`, all three of `StartTagBattle` (the partner included), the one of `StartFirstBattle`. An operand ≥ `0x4000` is a variable: it resolves to every `SetVarFromValue` of that variable in the same script file (the `SetVar` macro assembles to `SetVarFromValue` for a constant, `asm/macros/scrcmd.inc:309-315`). The vanilla round trip is the check; any trainer that differs is investigated, never whitelisted. **Stop rule:** `extract_decomp._trainer_maps` (`tools/extract_decomp.py:494-528`, regex `:425-427`) is a textual scan of every line of the script file, so it also counts unreachable code and any `SetVar` of a `TRAINER_` value, whatever the variable is later used for. If a remaining difference is caused by that over-approximation (the decomp side's only source for the trainer on that map is a line the walk proves unreachable, or a `SetVar` whose variable never reaches a battle command), stop and report the trainers, maps and lines to the user; do not change the walk to imitate the scan, and do not special-case trainers.
13. **`--game kaizo` refuses in K1 and writes nothing:** the readers' layouts and the reused tables are checked against the Kaizo ROM, then the CLI stops with `RomInputError("trainers: the Kaizo trainer layout is not confirmed (spec §5.4); Kaizo extraction lands in K2b")`. **Writes:** every output is first written to a temporary directory created in `out`'s parent (`tempfile.mkdtemp(dir=out.parent)`); only after every file is written is `out` created if missing and each file moved in with `os.replace` (atomic per file, same volume). Files in `out` that this run does not write are left untouched, so an existing `data/intermediate/` keeps whatever other steps put there. On any failure the temporary directory is removed and `out` is neither created nor changed. K2b removes the refusal.
14. **Reused tables** (`types.json`, `natures.json`, `charmap.json`, `abilities.json`) are copied from a fresh `extract_decomp.extract_static` run into a temporary directory, never from a possibly stale `data/intermediate/`. For a game other than vanilla, they are reused only when the diff shows `sTypeMatchupMultipliers` and `sNatureStatAffinities` without changes and `pl_msg.narc` members `TEXT_BANK_ABILITY_NAMES`, `TEXT_BANK_POKEMON_TYPE_NAMES` and `TEXT_BANK_NATURE_NAMES` unchanged. The charmap is the font encoding, which has no table in the ROM. `extract_rom` prints and writes `rom_sources.json` (`{file: "rom" | "decomp: unchanged in ROM diff"}`) beside the outputs.
15. **Milestone 4 merge:** `claude/m4-prep-search` adds `items.json` `teaches_move` and `tm_learnsets.json` to `extract_decomp.py`. The round trip compares both directions (a file or key only one side has fails), so whichever of K1 and milestone 4 merges second adds those two to the ROM readers (`pl_personal` TM masks at offset 28, the TM → move map behind `res/items/item_tm_move_map.h`). K2a re-checks this plan's `extract_decomp.py` line references on the merged file.
16. **`data/kaizo/rom_diff.toml`:** `[[code]]` tables `{module, symbol, status, note}`, where `symbol` is the symbol name or `"0xSTART-0xEND"` for a range outside every symbol, and `status` is `"modelled" | "irrelevant" | "unknown"`. Missing means unknown. An entry matching no current change is an error that names it (it went stale). The first run commits an explicit `unknown` entry for every changed symbol, so later reviews edit lines instead of adding them.
17. `tools/rom/symbols.py` `DECOMP_COMMIT = "579d2d5c65e736a1a4268fb7acbfd70712177c6c"`; a test pins it equal to `data/gen4_platinum/manifest.json` `decomp_commit` and `tools/clone_decomp.sh` `PINNED_COMMIT`.

## Review Focus

1. **A Windows path with backslashes in `data/local.toml`** (`vanilla_rom = "D:\roms\..."`) is a TOML escape error: expect a `RomInputError` that names the file and says to use single quotes, not a raw `TOMLDecodeError`. Test in Task 1.
2. **A wrong vanilla dump** (rev 0, `ce81046e…`, or a trimmed ROM): expect `RomInputError` naming the path, the expected and the actual SHA-1 before any patching. Test in Task 1 (`read_checked`) and Task 3 (`kaizo_rom`).
3. **An interrupted or foreign cache file** under `.cache/rom/` (truncated write, sidecar from another run): expect it to be re-verified and rebuilt, never used. Test in Task 3 (`test_cache_hit_and_corrupt_cache_rebuilt`, `test_foreign_sidecar_rebuilt`).
4. **A Kaizo NARC whose member size differs from the vanilla struct** (a changed layout, e.g. the trainer struct §5.4): expect the reader to raise naming the NARC, the member and both sizes, not to misparse. Fixed-size members: Task 9 (`pl_personal`, 44) and Task 10 (`pl_waza_tbl`, 16). Trainer parties: Task 11, where the real rule is `round_up_4(party_size × entry_size)` with zero padding (measured on vanilla), so a padded member is accepted and a short or over-long one raises.
5. **Re-running the diff after the classification was edited**, when a classified change disappeared or moved symbol: expect an error that names the stale entries, not silent loss. Test in Task 7 (`test_stale_entry_raises` for a vanished symbol, `test_moved_change_reports_old_symbol_stale` for one that moved from `Foo` to `Bar`).

---

### Task 1: Dependency group, ignores and local inputs

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups] rom = ["ndspy>=4.1,<5", "openpyxl>=3.1,<4"]`), `uv.lock` (`~/bin/uv.exe lock`), `.gitignore` (add `data/local.toml`, `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` under "Generated data and runs")
- Create: `tools/rom/__init__.py` (docstring only), `tools/rom/inputs.py`, `data/local.example.toml` (every key, single-quoted Windows-style example paths, a comment pointing at the env variables), `docs/rom_pipeline.md` (section "Inputs"; later tasks add sections)
- Test: `tests/tools/rom/__init__.py`, `tests/tools/rom/conftest.py`, `tests/tools/rom/test_inputs.py`

**Interfaces:**
- Produces (`tools/rom/inputs.py`):
  - `REPO_ROOT: Path` (`Path(__file__).resolve().parents[2]`), `VANILLA_SHA1`, `KAIZO_PATCH_SHA256` (Global Constraints values)
  - `class RomInputError(Exception)`
  - `@dataclass(frozen=True) RomInputs(vanilla_rom: Path, kaizo_patch: Path | None, pkcalc_zip: Path | None, kaizo_docs: Path | None, kaizo_save: Path | None)`
  - `load_inputs(config: Path = REPO_ROOT / "data" / "local.toml", environ: Mapping[str, str] = os.environ) -> RomInputs`: keys `vanilla_rom, kaizo_patch, pkcalc_zip, kaizo_docs, kaizo_save`; env `NUZLINE_VANILLA_ROM, NUZLINE_KAIZO_PATCH, NUZLINE_PKCALC_ZIP, NUZLINE_KAIZO_DOCS, NUZLINE_KAIZO_SAVE` (Ruling 5). Unknown TOML keys raise. Existence is not checked here.
  - `read_checked(path: Path, expected_sha: str, algo: str) -> bytes` (`algo` is a `hashlib` name; missing file or mismatch → `RomInputError` with path, expected, actual)
- Produces (`tests/tools/rom/conftest.py`, session scope): `rom_inputs` (skips with "no data/local.toml or NUZLINE_VANILLA_ROM" on `RomInputError` for a missing vanilla path, or when the file does not exist), `vanilla_bytes` (`read_checked(..., VANILLA_SHA1, "sha1")`), `decomp` (`REPO_ROOT / "external" / "pokeplatinum"`, skips when absent, as `tests/tools/test_extract_decomp.py` does), `charmap` (`extract_charmap(decomp)`).

- [ ] **Step 1: Write failing tests**

```python
def test_toml_and_env_precedence(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text("vanilla_rom = 'roms/v.nds'\nkaizo_patch = 'C:\\p\\k.xdelta'\n")
    got = load_inputs(cfg, environ={"NUZLINE_KAIZO_SAVE": "s.sav"})
    assert got.vanilla_rom == REPO_ROOT / "roms" / "v.nds"
    assert got.kaizo_patch == Path("C:\\p\\k.xdelta") and got.kaizo_save == Path("s.sav")
    assert got.pkcalc_zip is None and got.kaizo_docs is None
    assert load_inputs(cfg, environ={"NUZLINE_VANILLA_ROM": "x.nds"}).vanilla_rom == Path("x.nds")
def test_missing_vanilla_names_key_and_env(tmp_path):
    with pytest.raises(RomInputError, match="vanilla_rom.*NUZLINE_VANILLA_ROM"):
        load_inputs(tmp_path / "absent.toml", environ={})
def test_backslash_basic_string_explained(tmp_path):      # Review Focus 1
    cfg = tmp_path / "local.toml"; cfg.write_text('vanilla_rom = "D:\\roms\\v.nds"\n')
    with pytest.raises(RomInputError, match="single-quoted"):
        load_inputs(cfg, environ={})
def test_unknown_key_raises(tmp_path):
    cfg = tmp_path / "local.toml"; cfg.write_text("vanila_rom = 'v.nds'\n")
    with pytest.raises(RomInputError, match="vanila_rom"):
        load_inputs(cfg, environ={})
def test_read_checked(tmp_path):
    p = tmp_path / "f.bin"; p.write_bytes(b"abc")
    assert read_checked(p, hashlib.sha1(b"abc").hexdigest(), "sha1") == b"abc"
    with pytest.raises(RomInputError, match="ce81046e.*a9993e36"):   # Review Focus 2: expected, then actual
        read_checked(p, "ce81046eda7d232513069519cb2085349896dec7", "sha1")
    with pytest.raises(RomInputError, match="absent.bin"):
        read_checked(tmp_path / "absent.bin", "0" * 40, "sha1")
def test_example_config_parses():
    got = load_inputs(REPO_ROOT / "data" / "local.example.toml", environ={})
    assert got.vanilla_rom.suffix == ".nds" and got.kaizo_patch.suffix == ".xdelta"
    assert got.kaizo_save.suffix == ".sav"
def test_vanilla_rom_hash(vanilla_bytes):                  # skips without inputs
    assert len(vanilla_bytes) == 134217728 and vanilla_bytes[12:16] == b"CPUE"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_inputs.py -q`
- [ ] **Step 3: Implement** `inputs.py` with `tomllib`, add the dependency group and lock, the ignores, the example config, and `docs/rom_pipeline.md` "Inputs" (keys, env variables, hashes, "never commit").
- [ ] **Step 4: Run, expect PASS**, then the same command with the user's `data/local.toml` in place to see `test_vanilla_rom_hash` pass rather than skip. `git status` must not list `data/local.toml`.
- [ ] **Step 5: Commit** `feat(rom): local input config and hash checks for the ROM pipeline`

### Task 2: VCDIFF decoder

**Files:**
- Create: `tools/rom/vcdiff.py`, `tests/tools/rom/vcdiff_writer.py` (test-only encoder)
- Test: `tests/tools/rom/test_vcdiff.py`

**Interfaces:**
- Produces: `class VcdiffError(Exception)`; `apply_vcdiff(source: bytes, patch: bytes) -> bytes`, implementing RFC 3284 (magic `D6 C3 C4 00`, header indicator, default code table §5.6, near cache 4 / same cache 3 × 256 address modes §5.3, `VCD_SOURCE` and `VCD_TARGET` segments, target self-copies that overlap the write position) plus Ruling 2's xdelta3 extensions: application header (skipped), secondary LZMA, per-window Adler-32 (verified). Every error names the window index and byte offset: bad magic, unsupported indicator bit or compressor id, a section that decodes to the wrong length, a COPY outside the source segment plus target so far, a size mismatch, a checksum mismatch, trailing or truncated data.
- Produces (`vcdiff_writer.py`): `encode(windows: list[Window], *, app_header: bytes | None = None, secondary: bool = False, adler: bool | int = True) -> bytes` (an `int` is written as every window's checksum instead of the real one), where `Window(source_segment: tuple[int, int] | None, target_from_target: bool, ops: list[Op])` and `Op` is `("add", bytes) | ("run", byte, n) | ("copy", addr, n, mode)`. `addr` is the decoded address (in the window's source segment followed by its target); the writer simulates the near and same caches and encodes `addr` in the given `mode`, raising `ValueError` when the mode cannot reach it. It emits each op with the default table's single-instruction code whose size field is 0 (size follows as a varint), so every address mode is reachable; with `secondary=True` each section is one finished xz stream with its Delta_Indicator bit set (Python's `lzma` has no sync flush), so secondary tests use one window. Cross-window stream continuation is covered by Task 3's real-patch test.

- [ ] **Step 1: Write failing tests**

```python
SRC = bytes(range(256)) * 4
W = Window((0, 1024), False, [("add", b"hi"), ("copy", 10, 20, 0), ("run", 7, 5)])
WANT = b"hi" + SRC[10:30] + b"\x07" * 5
def test_add_run_copy_from_source():
    assert apply_vcdiff(SRC, encode([W])) == WANT
def test_overlapping_self_copy():
    assert apply_vcdiff(b"", encode([Window(None, False, [("add", b"ab"), ("copy", 0, 10, 0)])])) == b"ab" * 6
def test_every_address_mode():   # modes 0 self, 1 here, 2-5 near, 6-8 same; each address reachable given the caches so far
    plan = [(0, 10), (1, 40), (2, 12), (3, 44), (4, 600), (5, 300), (6, 10), (7, 300), (8, 600)]
    ops = [("add", b"x")] + [("copy", a, 4, m) for m, a in plan]
    assert apply_vcdiff(SRC, encode([Window((0, 1024), False, ops)])) == b"x" + b"".join(SRC[a:a + 4] for _, a in plan)
def test_two_windows_and_target_window():
    p = encode([Window(None, False, [("add", b"hello")]), Window((0, 5), True, [("copy", 0, 5, 0)])])
    assert apply_vcdiff(b"", p) == b"hellohello"
def test_app_header_and_secondary_lzma():   # one window: the writer finishes each xz stream
    p = encode([W], app_header=b"t.nds//s.nds/", secondary=True)
    assert p[4] == 0x05 and apply_vcdiff(SRC, p) == WANT
def test_bad_adler_raises():
    with pytest.raises(VcdiffError, match="window 0.*Adler"):
        apply_vcdiff(SRC, encode([W], adler=0x12345678))
def test_truncated_patch_raises():
    with pytest.raises(VcdiffError, match="window 0"):
        apply_vcdiff(SRC, encode([W])[:-3])
def test_custom_code_table_refused():
    p = encode([W]); p = p[:4] + bytes([p[4] | 0x02]) + p[5:]
    with pytest.raises(VcdiffError, match="code table"):
        apply_vcdiff(SRC, p)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_vcdiff.py -q`
- [ ] **Step 3: Implement.** Build the 256-entry default instruction table from §5.6's rules (assert its length). Write the target window into a `bytearray`; an overlapping copy repeats the already-written chunk (doubling), not byte by byte.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): pure-Python VCDIFF decoder with xdelta3 secondary LZMA and Adler-32`

### Task 3: Patch, cache, header check and `data/kaizo/rom.toml`

**Files:**
- Create: `tools/rom/patch.py`, `data/kaizo/rom.toml`
- Test: `tests/tools/rom/test_patch.py`; `tests/tools/rom/conftest.py` gains `kaizo_inputs` (session; `rom_inputs`, then skips with "no kaizo_patch configured" when `kaizo_patch` is `None` or the file does not exist) and `kaizo_bytes` (session; built on `kaizo_inputs`). Every test that needs the patch takes `kaizo_inputs`, `kaizo_bytes` or `kaizo_files`, never bare `rom_inputs`.

**Interfaces:**
- Consumes: Task 1 `RomInputs`, `read_checked`, `RomInputError`, `REPO_ROOT`; Task 2 `apply_vcdiff`.
- Produces:
  - `@dataclass(frozen=True) RomHeader(title: str, game_code: str, maker: str, revision: int)`; `rom_header(data: bytes) -> RomHeader` (bytes 0x00-0x0B, 0x0C-0x0F, 0x10-0x11, 0x1E of the NDS header)
  - `expected_patched(path: Path = REPO_ROOT / "data" / "kaizo" / "rom.toml") -> dict | None`
  - `kaizo_rom(inputs: RomInputs, cache_dir: Path = REPO_ROOT / ".cache" / "rom") -> tuple[Path, str]` (Ruling 6; checks the vanilla SHA-1 and the patch SHA-256 first, then the patched size and header: title, maker and revision equal to vanilla's, game code and SHA-1 equal to `rom.toml`)
- `data/kaizo/rom.toml`:

```toml
# Patched Kaizo ROM (vanilla Platinum US + Nightmare Update Final xdelta). Hashes only, no ROM bytes.
# game_code is JAK7, not vanilla's CPUE: measured 2026-11-03 (plan K1 Ruling 1); the patch changes it.
vanilla_sha1 = "0862ec35b24de5c7e2dcb88c9eea0873110d755c"
patch_sha256 = "b787e743b5fa78e1b31bae9277e656482b7248b82e490a6e95236606ae92f433"
sha1 = "e8d13c6504406324a4a9ea9daf8d6bd409f8a644"
size = 134217728
title = "POKEMON PL"
game_code = "JAK7"
maker = "01"
revision = 1
```

- [ ] **Step 1: Write failing tests**

```python
def test_rom_toml_matches_inputs_constants():
    exp = expected_patched()
    assert (exp["vanilla_sha1"], exp["patch_sha256"]) == (VANILLA_SHA1, KAIZO_PATCH_SHA256)
def test_wrong_vanilla_refused_before_patching(tmp_path):   # Review Focus 2
    v = tmp_path / "v.nds"; v.write_bytes(b"x"); p = tmp_path / "k.xdelta"; p.write_bytes(b"y")
    with pytest.raises(RomInputError, match="v.nds.*0862ec35"):
        kaizo_rom(RomInputs(v, p, None, None, None), tmp_path / "c")
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())
def test_missing_patch_refused(tmp_path):      # checked before the vanilla file is read
    with pytest.raises(RomInputError, match="kaizo_patch"):
        kaizo_rom(RomInputs(tmp_path / "v.nds", None, None, None, None), tmp_path)
def test_kaizo_rom_and_header(kaizo_inputs, tmp_path):
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    exp = expected_patched()
    h = rom_header(path.read_bytes())
    assert sha == exp["sha1"] and path.stat().st_size == exp["size"]
    assert (h.title, h.maker, h.revision) == ("POKEMON PL", "01", 1) and h.game_code == exp["game_code"]
def _count_builds(monkeypatch):
    calls = []; real = patch_mod.apply_vcdiff
    monkeypatch.setattr(patch_mod, "apply_vcdiff", lambda s, p: calls.append(1) or real(s, p))
    return calls
def test_cache_hit_and_corrupt_cache_rebuilt(kaizo_inputs, tmp_path, monkeypatch):   # Review Focus 3
    calls = _count_builds(monkeypatch)
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 1      # hit
    path.write_bytes(path.read_bytes()[:-1])                                          # truncated
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 2
    assert hashlib.sha1(path.read_bytes()).hexdigest() == sha
def test_foreign_sidecar_rebuilt(kaizo_inputs, tmp_path, monkeypatch):               # Review Focus 3
    calls = _count_builds(monkeypatch)
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    sidecar = path.with_name(path.name + ".sha1"); sidecar.write_text("0" * 40)      # from another run
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 2
    assert sidecar.read_text().strip() == sha
```

(`patch_mod` is `tools.rom.patch`; each build is about 12 s, so these two tests cost about a minute.)

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_patch.py -q`
- [ ] **Step 3: Implement** `patch.py` and add `kaizo_inputs` and `kaizo_bytes` to the conftest (`kaizo_bytes` reads the path `kaizo_rom` returns, using the default cache).
- [ ] **Step 4: Run, expect PASS** with the user's inputs configured (none skipped). If the SHA-1 or game code differs from `rom.toml`, stop (Ruling 1).
- [ ] **Step 5: Commit** `feat(rom): apply and cache the Kaizo patch; record the patched ROM's hash and header`. Report to the user that the patched game code is `JAK7`.

### Task 4: NDS file system

**Files:**
- Create: `tools/rom/nds.py`
- Test: `tests/tools/rom/test_nds.py`; conftest gains `vanilla_files`, `kaizo_files` (session, `RomFiles.open` of the bytes fixtures, `importorskip("ndspy")`)

**Interfaces:**
- Consumes: ndspy 4.x (`ndspy.rom.NintendoDSRom`, `.filenames` / `.files`, `.loadArm9()` → `MainCodeFile.sections`, `.loadArm9Overlays()` → `Overlay.data/.ramAddress/.ramSize/.bssSize`, `ndspy.narc.NARC(...).files`).
- Produces: `class RomFiles` with `RomFiles.open(data: bytes) -> RomFiles`; `.game_code: str`; `.paths() -> list[str]` (FNT paths, `/`-separated, no leading slash, sorted); `.file(path: str) -> bytes` (`KeyError` naming the path); `.narc(path: str) -> list[bytes]`; `.arm9() -> bytes` (decompressed); `.arm9_sections() -> list[tuple[int, bytes]]` (RAM address, data; static section first); `.overlay(i: int) -> bytes` (decompressed); `.overlay_ids() -> list[int]`; `.overlay_info(i: int) -> tuple[int, int, int]` (RAM address, RAM size, BSS size); `.header() -> bytes` (the 0x200-byte NDS header, for Task 6's header diff and Task 3's `rom_header`). ndspy is imported inside `open`.

- [ ] **Step 1: Write failing tests**

```python
def test_vanilla_layout(vanilla_files):
    assert vanilla_files.game_code == "CPUE" and len(vanilla_files.paths()) == 340
    assert len(vanilla_files.overlay_ids()) == 122 and len(vanilla_files.arm9()) == 0x1023F8
    assert vanilla_files.overlay_info(16)[:2] == (0x223B140, 0x35A80)
    assert len(vanilla_files.narc("poketool/personal/pl_personal.narc")[1]) == 44   # sizeof(SpeciesData)
def test_kaizo_layout(kaizo_files):
    assert kaizo_files.game_code == "JAK7" and len(kaizo_files.paths()) == 2091
    assert len(kaizo_files.overlay(16)) == 0x35AA0
def test_missing_path_names_it(vanilla_files):
    with pytest.raises(KeyError, match="nope/x.narc"): vanilla_files.file("nope/x.narc")
def test_rom_package_imports_without_ndspy():   # subprocess: sys.modules["ndspy"] = None, then import every tools.rom module except readers -> exit 0
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_nds.py -q`
- [ ] **Step 3: Implement** `nds.py`. Neither ROM here has a compressed ARM9 or overlay (measured: module params `compressed_static_end` is 0, and no overlay has the y9 compressed flag), but `arm9()`/`overlay()` still decompress through ndspy when flagged.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): unpack NDS file systems, NARCs, ARM9 and overlays through ndspy`

### Task 5: Build the decomp once; linker-map symbols

**Files:**
- Create: `tools/rom/symbols.py`, `tests/tools/rom/fixtures/xmap_excerpt.txt` (about 30 lines cut from the real map: one ARM9 function, one ITCM symbol, two overlays' sections, their `SDK_OVERLAY_*_ID` lines; names and addresses only)
- Modify: `docs/rom_pipeline.md` (section "Decomp build", the steps below verbatim)
- Test: `tests/tools/rom/test_symbols.py`; conftest gains `symbols` (session; skips when `DEFAULT_MAP` is absent)

**Interfaces:**
- Produces: `DECOMP_COMMIT` (Ruling 17); `DEFAULT_MAP = REPO_ROOT / ".cache" / "decomp" / DECOMP_COMMIT / "main.nef.xMAP"`; `@dataclass(frozen=True) Symbol(module: str, address: int, size: int, name: str, obj: str, section: str)` (`section` as the map names it, e.g. `.text`, `.rodata`, `.data`, `.bss`, `.itcm`); `class SymbolMap` with `SymbolMap(symbols: Iterable[Symbol])`, `SymbolMap.from_map(path: Path) -> SymbolMap`, `.lookup(module: str, address: int) -> str | None` (the symbol whose `[address, address + size)` contains it; module `"arm9"` or `"overlay<N>"`; zero-size labels are never returned by `lookup`, only by `find`; when several symbols contain the address (aliases, nested labels) the smallest wins, ties going to the first in map order), `.find(name: str) -> Symbol` (`KeyError` if absent or ambiguous), `.symbols(module: str) -> list[Symbol]` (sorted by address).

- [ ] **Step 1: Build the decomp in WSL** (`external/pokeplatinum/INSTALL.md`, WSL2 section and §2-3):
  0. Precheck, in PowerShell: `wsl -l -v` must list `Ubuntu` with `VERSION 2`. On WSL 1, `INSTALL.md` advises building on the Windows drive instead of the Linux filesystem; stop and ask the user rather than adapting the steps.
  1. The user runs, in `wsl -d Ubuntu` (sudo password; `INSTALL.md:72-75`):
     `sudo dpkg --add-architecture i386 && sudo apt update`
     `sudo apt install bison flex g++ gcc-arm-none-eabi git make ninja-build pkg-config wget python3 xz-utils nasm libc6:i386 libpng-dev`
  2. Then, in the Linux filesystem (`INSTALL.md:66-69, 315-318, 324-334`):
     `cd ~ && git clone https://github.com/pret/pokeplatinum && cd pokeplatinum && git -c advice.detachedHead=false checkout 579d2d5c65e736a1a4268fb7acbfd70712177c6c && make`
     `make` is `release check` (`Makefile:96-106`): it downloads metroskrew and meson 1.12.0, builds `build/pokeplatinum.us.nds`, and its "ROM Checksum" test must pass (SHA-1 `0862ec35…755c`, `INSTALL.md:334`).
  3. `ls build/*.xMAP`. Expected `build/main.nef.xMAP`. If absent, follow Ruling 3's fallback.
  4. Copy the map and `platinum.us/main.lsf` into the K1 worktree. In WSL, with `WT` set to the worktree's Windows path exactly as `git rev-parse --show-toplevel` prints it there (for example `<repo>/.claude/worktrees/k1-rom-pipeline`):
     ```bash
     DEST="$(wslpath -a "$WT")/.cache/decomp/579d2d5c65e736a1a4268fb7acbfd70712177c6c"
     mkdir -p "$DEST" && cp build/main.nef.xMAP platinum.us/main.lsf "$DEST/" && ls -l "$DEST"
     ```
  Expected: `sha1sum build/pokeplatinum.us.nds` prints `0862ec35b24de5c7e2dcb88c9eea0873110d755c`.
- [ ] **Step 2: Look at the map's format** (section headers, symbol lines, where `SDK_OVERLAY_*_ID` appears), and cut the fixture excerpt from it.
- [ ] **Step 3: Write failing tests**

```python
def test_excerpt_parses():                                   # fixture only, no skip
    m = SymbolMap.from_map(FIXTURES / "xmap_excerpt.txt")
    # assertions on the excerpt's own lines: one arm9 lookup, one overlay lookup, an address between symbols -> None
def test_decomp_commit_pinned():                             # == manifest decomp_commit == clone_decomp.sh PINNED_COMMIT
def test_known_symbols(symbols):
    assert symbols.find("TrainerData_BuildParty").module == "arm9"
    assert symbols.find("TrainerAI_CalcDamage").module == "overlay14"      # trainer_ai, main.lsf order
    assert symbols.find("sTypeMatchupMultipliers").module == "overlay16"   # battle
    for name in ("sMapHeaders", "sYearlyWeather", "sTrainerClassGender", "sItemArchiveIDs",
                 "sNatureStatAffinities"):
        assert symbols.find(name).module == "arm9"
    assert symbols.find("gVsSeekerRematchData").module == "overlay5"
def test_overlay_symbols_inside_rom_overlays(symbols, vanilla_files):
    for i in vanilla_files.overlay_ids():
        ram, size, bss = vanilla_files.overlay_info(i)
        assert all(ram <= s.address and s.address + s.size <= ram + size + bss
                   for s in symbols.symbols(f"overlay{i}"))
def test_arm9_symbols_inside_rom_sections(symbols, vanilla_files):   # the map against the ROM, not against itself
    ranges = [(ram, ram + len(data)) for ram, data in vanilla_files.arm9_sections()]
    for s in symbols.symbols("arm9"):
        if s.size and s.section not in (".bss", ".sbss"):
            assert any(lo <= s.address and s.address + s.size <= hi for lo, hi in ranges), s
def test_known_functions_start_with_thumb_push(symbols, vanilla_files):
    for name in ("TrainerData_BuildParty", "TrainerAI_CalcDamage"):
        s = symbols.find(name)
        first = _code_at(vanilla_files, s.module, s.address, 2)
        assert first[1] == 0xB5, (name, first.hex())     # Thumb push {..., lr}, little-endian
def test_zero_size_and_alias_lookup():                  # SymbolMap([...]) built in the test, no skip
    m = SymbolMap([Symbol("arm9", 0x100, 0, "Label", "a.o", ".text"),
                   Symbol("arm9", 0x100, 0x20, "Outer", "a.o", ".text"),
                   Symbol("arm9", 0x100, 0x10, "Inner", "a.o", ".text")])
    assert m.lookup("arm9", 0x104) == "Inner" and m.lookup("arm9", 0x118) == "Outer"
    assert m.lookup("arm9", 0x120) is None and m.find("Label").size == 0
```

(`_code_at(rom, module, address, n)` is a test helper: `n` bytes at a RAM address, from the ARM9 section containing it or from `overlay(N)` at `address - overlay_info(N)[0]`. If one of the two functions does not open with `push`, look at its first instructions in the decomp's disassembly before concluding the map is wrong.)

- [ ] **Step 4: Run, expect FAIL**, implement `symbols.py`, **run, expect PASS** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_symbols.py -q`
- [ ] **Step 5: Commit** `feat(rom): linker-map symbols from the vanilla decomp build`

### Task 6: ROM diff

**Files:**
- Create: `tools/rom/diff.py` (the diff part; Task 7 adds the report)
- Test: `tests/tools/rom/test_diff.py` (unit tests on a `FakeRom` class with the `RomFiles` methods, built in the test file), `tests/tools/rom/test_diff_kaizo.py` (real ROMs; skips)

**Interfaces:**
- Consumes: Task 4 `RomFiles` methods (duck-typed), Task 5 `SymbolMap.lookup`/`.symbols`.
- Produces:
  - `HEADER_FIELDS: dict[str, tuple[int, int, str]]` (name → offset, width, kind `"ascii" | "int"`), exactly these named scalars of the NDS header (GBATEK "DS Cartridge Header"): `title` 0x000/12 ascii, `game_code` 0x00C/4 ascii, `maker_code` 0x010/2 ascii, `unit_code` 0x012/1, `device_capacity` 0x014/1, `rom_version` 0x01E/1, `autostart` 0x01F/1, `arm9_rom_offset` 0x020/4, `arm9_entry` 0x024/4, `arm9_ram_address` 0x028/4, `arm9_size` 0x02C/4, `arm7_rom_offset` 0x030/4, `arm7_entry` 0x034/4, `arm7_ram_address` 0x038/4, `arm7_size` 0x03C/4, `fnt_offset` 0x040/4, `fnt_size` 0x044/4, `fat_offset` 0x048/4, `fat_size` 0x04C/4, `arm9_overlay_offset` 0x050/4, `arm9_overlay_size` 0x054/4, `arm7_overlay_offset` 0x058/4, `arm7_overlay_size` 0x05C/4, `icon_title_offset` 0x068/4, `secure_area_crc` 0x06C/2, `total_rom_size` 0x080/4, `header_size` 0x084/4, `logo_crc` 0x15C/2, `header_crc` 0x15E/2. Nothing else of the header is compared or printed (no logo, reserved areas, or secure area).
  - `@dataclass(frozen=True) HeaderChange(field: str, vanilla: str, patched: str)` (`field` a `HEADER_FIELDS` key; values are the ASCII text or `0x`-hex of the integer, always under 32 characters)
  - `@dataclass(frozen=True) FileChange(path: str, status: str, vanilla_size: int | None, patched_size: int | None, members_changed: tuple[int, ...] = (), member_counts: tuple[int, int] | None = None)` (`status` `"changed" | "added" | "removed"`; NARC members compared one by one)
  - `@dataclass(frozen=True) CodeChange(module: str, start: int, end: int, patched_start: int, patched_end: int, symbol: str | None)` (vanilla RAM addresses, half-open; an insertion has `start == end`)
  - `@dataclass(frozen=True) RomDiff(header: tuple[HeaderChange, ...], files: tuple[FileChange, ...], code: tuple[CodeChange, ...], relocations: dict[tuple[str, str | None], int], modules: dict[str, tuple[int, int]])` (`modules`: vanilla and patched size of every module that differs)
  - `diff_roms(vanilla: RomFiles, patched: RomFiles, symbols: SymbolMap | None) -> RomDiff` (Ruling 7; without symbols every `symbol` is `None` and no splitting happens)
  - `member_label(path: str, index: int, decomp: Path) -> str | None` (`pl_msg.narc` → `TEXT_BANK_*`; `pl_personal`/`wotbl`/`evo` → `SPECIES_*`; `pl_waza_tbl` → `MOVE_*`; `trdata`/`trpoke` → `TRAINER_*`; else `None`; names from `generated/*.txt` via `tools.extract_decomp._constants`)

- [ ] **Step 1: Write failing tests**

`fake(files=None, arm9=None, ov3=None, header=None)` builds a `FakeRom`: `files` maps a path to bytes, or to a list of members for a `.narc` path; `arm9` is a list of `(ram, data)` sections (default one empty section at `0x02000000`); `ov3` is overlay 3's code at `BASE = 0x02200000` (default `CODE = bytes(range(256))`); `header` the 0x200-byte header (default zeros with `game_code` `CPUE`). `poke(data, off, new)` returns `data` with `new` written at `off`. `SYMS = SymbolMap([Symbol("overlay3", BASE + 0x40, 0x20, "Foo", "a.o", ".text"), Symbol("overlay3", BASE + 0x60, 0x20, "Bar", "a.o", ".text")])`.

```python
def test_identical_roms_have_no_diff():
    r = fake(); assert diff_roms(r, r, None) == RomDiff((), (), (), {}, {})
def test_added_removed_changed_files_and_narc_members():
    v = fake(files={"a.bin": b"1", "gone.bin": b"2", "n.narc": [b"m"] * 5})
    p = fake(files={"a.bin": b"9", "new/x.bin": b"3", "n.narc": [b"m", b"m", b"M", b"m", b"m"]})
    by = {f.path: f for f in diff_roms(v, p, None).files}
    assert [by[k].status for k in ("a.bin", "gone.bin", "new/x.bin")] == ["changed", "removed", "added"]
    assert by["n.narc"].members_changed == (2,) and by["n.narc"].member_counts == (5, 5)
def test_in_place_code_change_named_by_symbol():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x44, b"\xff\xff")), SYMS)
    assert d.code == (CodeChange("overlay3", BASE + 0x44, BASE + 0x46, BASE + 0x44, BASE + 0x46, "Foo"),)
def test_change_spanning_two_symbols_is_split():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x5E, b"\xff" * 4)), SYMS)
    assert [(c.start - BASE, c.end - BASE, c.symbol) for c in d.code] == [(0x5E, 0x60, "Foo"), (0x60, 0x62, "Bar")]
def test_nearby_changes_merge():                     # 3 equal bytes between -> one range; 5 -> two
    near = poke(poke(CODE, 0x44, b"\xff"), 0x48, b"\xff")
    far = poke(poke(CODE, 0x44, b"\xff"), 0x4A, b"\xff")
    assert len(diff_roms(fake(), fake(ov3=near), SYMS).code) == 1
    assert len(diff_roms(fake(), fake(ov3=far), SYMS).code) == 2
def test_insertion_aligned():
    d = diff_roms(fake(), fake(ov3=CODE[:0x80] + b"\xee" * 8 + CODE[0x80:]), None)
    assert [(c.start - BASE, c.end - BASE, c.patched_start - BASE, c.patched_end - BASE) for c in d.code] == [(0x80, 0x80, 0x80, 0x88)]
    assert d.modules["overlay3"] == (0x100, 0x108)
def test_relocated_thumb_bl_counted_not_listed():
    # van/pat/syms built in the test: an 8-byte insertion inside "Grown"; a Thumb BL in "Caller",
    # before the insertion, targets "Callee" past it and is re-encoded 8 bytes further
    d = diff_roms(van, pat, syms)
    assert [c.symbol for c in d.code] == ["Grown"] and d.relocations[("overlay3", "Caller")] == 1
def test_relocated_literal_pointer_counted():        # same layout; "Holder" has a 32-bit word = Callee's absolute address
    d = diff_roms(van, pat, syms)
    assert [c.symbol for c in d.code] == ["Grown"] and d.relocations[("overlay3", "Holder")] == 1
def test_itcm_section_addresses():
    itcm = bytes(64); v = fake(arm9=[(0x02000000, CODE), (0x01FF8000, itcm)])
    p = fake(arm9=[(0x02000000, CODE), (0x01FF8000, poke(itcm, 0x10, b"\x01"))])
    assert diff_roms(v, p, None).code == (CodeChange("arm9", 0x01FF8010, 0x01FF8011, 0x01FF8010, 0x01FF8011, None),)
def test_header_only_named_fields():
    h = bytearray(fake().header()); h[0xC0] ^= 0xFF          # logo byte: not a named field
    assert diff_roms(fake(), fake(header=bytes(h)), None).header == ()
    h[0x0C:0x10] = b"JAK7"
    assert diff_roms(fake(), fake(header=bytes(h)), None).header == (HeaderChange("game_code", "CPUE", "JAK7"),)
def test_kaizo_diff_shape(vanilla_files, kaizo_files, symbols):   # test_diff_kaizo.py
    d = diff_roms(vanilla_files, kaizo_files, symbols)
    assert d.modules["overlay16"] == (0x35A80, 0x35AA0)
    assert {c.module for c in d.code} >= {"arm9", "overlay5", "overlay13", "overlay14", "overlay16"}
    assert len([f for f in d.files if f.status == "changed"]) == 38
    assert len([f for f in d.files if f.status == "added"]) == 1751
    assert HeaderChange("game_code", "CPUE", "JAK7") in d.header
    assert all(h.field in HEADER_FIELDS and len(h.vanilla) < 32 and len(h.patched) < 32 for h in d.header)
```

Also print the code ranges with `symbol is None`. Ranges outside every symbol are legal (they are classified by range, Ruling 16), but many of them inside one module mean the map lacks data symbols: report that to the user before Task 7.

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_diff.py tests/tools/rom/test_diff_kaizo.py -q`
- [ ] **Step 3: Implement.** Thumb `BL` decoding: halfwords `11110 hi11` + `11111 lo11` (`BLX`: `11101`), target `pc + 4 + (sext(hi11) << 12) + (lo11 << 1)`; ARM `BL` cond `1011` imm24. "Mapped through the alignment" means the vanilla target address translated by the matching blocks of the same module.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): file, NARC-member and symbol-mapped code diff between two ROMs`

### Task 7: Report, classification, CLI and the first Kaizo run

**Files:**
- Modify: `tools/rom/diff.py` (report, classification, `main`)
- Create: `docs/kaizo/rom_diff.md`, `data/kaizo/rom_diff.toml` (both generated by Step 4); `docs/rom_pipeline.md` section "ROM diff"
- Test: `tests/tools/rom/test_diff_report.py`

**Interfaces:**
- Consumes: Task 6 `RomDiff`, `member_label`; Task 5 `DECOMP_COMMIT`; Task 3 `kaizo_rom`, `expected_patched`; Task 1 `load_inputs`, `read_checked`, `VANILLA_SHA1`, `KAIZO_PATCH_SHA256`.
- Produces:
  - `load_classification(path: Path) -> dict[tuple[str, str], tuple[str, str]]` ((module, symbol-or-range) → (status, note); Ruling 16; an unknown status raises)
  - `classification_skeleton(diff: RomDiff) -> str` (TOML text with one `unknown` entry per changed (module, symbol), sorted by module then address)
  - `stale_entries(diff: RomDiff, classification: dict) -> list[tuple[str, str]]`
  - `default_provenance() -> dict[str, str]`: `{"vanilla_sha1": VANILLA_SHA1, "patch_sha256": KAIZO_PATCH_SHA256, "patched_sha1": expected_patched()["sha1"], "decomp_commit": DECOMP_COMMIT}` (no input files read).
  - `write_report(diff: RomDiff, classification: dict, path: Path, provenance: dict[str, str] | None = None, decomp: Path | None = None) -> None` (the fixed cross-plan call `write_report(diff, classification, path)` works: `provenance` defaults to `default_provenance()`; `decomp` enables `member_label`): raises `ValueError` listing `stale_entries` if there are any. Writes a deterministic, ASCII Markdown file: a provenance table (vanilla SHA-1, patch SHA-256, patched SHA-1, decomp commit), header changes, module sizes, changed files (NARCs with changed members as compressed index ranges plus labels), added files collapsed per folder with counts, and per module a table `symbol | ranges | bytes changed | relocations | status | note`. No ROM bytes, no file contents.
  - CLI `main(argv: list[str] | None = None) -> int`, run as `python -m tools.rom.diff [--symbols PATH] [--decomp DIR] [--classification data/kaizo/rom_diff.toml] [--out docs/kaizo/rom_diff.md] [--init-classification]` (`--init-classification` writes the skeleton when the file does not exist and refuses, with exit status 1 and before any input is read, to overwrite one that does)

- [ ] **Step 1: Write failing tests**

`D` is the diff of Task 6's `fake()` against `fake(ov3=poke(CODE, 0x44, b"\xff\xff"))` with `SYMS` (one change in `Foo`); `D_MOVED` the same with the change at `0x64` (now in `Bar`); `PROV = {"vanilla_sha1": "v", "patch_sha256": "p", "patched_sha1": "k", "decomp_commit": "c"}`.

```python
def foo_row(text): return next(l for l in text.splitlines() if l.startswith("| Foo |"))
def test_report_deterministic_and_ascii(tmp_path):
    for n in ("a.md", "b.md"): write_report(D, {}, tmp_path / n, PROV)
    a = (tmp_path / "a.md").read_bytes(); assert a == (tmp_path / "b.md").read_bytes() and a.isascii()
def test_default_status_is_unknown(tmp_path):
    write_report(D, {}, tmp_path / "r.md", PROV); assert "| unknown |" in foo_row((tmp_path / "r.md").read_text())
def test_fixed_signature_uses_default_provenance(tmp_path):   # K2b calls write_report(diff, classification, path)
    write_report(D, {}, tmp_path / "r.md"); assert DECOMP_COMMIT in (tmp_path / "r.md").read_text()
def test_classification_applied(tmp_path):
    write_report(D, {("overlay3", "Foo"): ("irrelevant", "menu")}, tmp_path / "r.md", PROV)
    assert "| irrelevant | menu |" in foo_row((tmp_path / "r.md").read_text())
def test_stale_entry_raises(tmp_path):                  # Review Focus 5: vanished symbol
    with pytest.raises(ValueError, match="overlay3.*Gone"):
        write_report(D, {("overlay3", "Gone"): ("unknown", "")}, tmp_path / "r.md", PROV)
    assert not (tmp_path / "r.md").exists()
def test_moved_change_reports_old_symbol_stale(tmp_path):   # Review Focus 5: change moved Foo -> Bar
    cls = {("overlay3", "Foo"): ("modelled", "x")}
    assert stale_entries(D_MOVED, cls) == [("overlay3", "Foo")]
    with pytest.raises(ValueError, match="Foo"):
        write_report(D_MOVED, cls, tmp_path / "r.md", PROV)
def test_skeleton_round_trips(tmp_path):
    p = tmp_path / "c.toml"; p.write_text(classification_skeleton(D))
    cls = load_classification(p)
    assert set(cls) == {(c.module, c.symbol) for c in D.code} and {s for s, _ in cls.values()} == {"unknown"}
def test_init_refuses_overwrite(tmp_path):
    p = tmp_path / "c.toml"; p.write_text("# mine\n")
    assert main(["--classification", str(p), "--init-classification"]) == 1 and p.read_text() == "# mine\n"
def test_diff_types_carry_no_bytes():
    for cls in (RomDiff, FileChange, CodeChange, HeaderChange):
        assert all("bytes" not in str(f.type) for f in dataclasses.fields(cls)), cls
```

(HeaderChange values are pinned under 32 characters on the real diff in Task 6's `test_kaizo_diff_shape`.)

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_diff_report.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS; then the first Kaizo run:**

```bash
~/bin/uv.exe run --group rom python -m tools.rom.diff --init-classification
~/bin/uv.exe run --group rom python -m tools.rom.diff
git status --short   # exactly two new paths: ?? data/kaizo/rom_diff.toml and ?? docs/kaizo/rom_diff.md
```

Expected: every code row's status is `unknown`; header changes include `game_code CPUE -> JAK7`; 38 changed and 1751 added files. Read the report once yourself, end to end, for anything that looks like content rather than names and addresses.
- [ ] **Step 5: Commit** `feat(rom): ROM diff report and classification; first Kaizo inventory (all unknown)` with `docs/kaizo/rom_diff.md` and `data/kaizo/rom_diff.toml`

### Task 8: Message banks

**Files:**
- Create: `tools/rom/readers/__init__.py` (shared helpers), `tools/rom/readers/text.py`
- Test: `tests/tools/rom/test_text.py`

**Interfaces:**
- Consumes: `RomFiles.narc`, `RomFiles.arm9_sections`, `RomFiles.overlay`; `SymbolMap.find`; `tools.extract_decomp._constants`, `_index`; `charmap.json` contents (code → character).
- Produces:
  - `readers/__init__.py`: `class ReaderError(RuntimeError)`; `symbol_bytes(rom: RomFiles, symbols: SymbolMap, name: str) -> bytes` (the symbol's bytes from its module; raises if the symbol falls outside the module); `fixed_members(members: list[bytes], size: int, path: str) -> list[bytes]` (Review Focus 4: raises naming path, member, expected and actual size)
  - `decode_bank(data: bytes, charmap: dict[int, str]) -> list[str]`: `MessageBank_Get` decryption (`src/message.c:10-60`: entry key `(seed * 765 * (i + 1)) & 0xFFFF` doubled into 32 bits over offset and length; characters XOR `(i + 1) * 596947` plus `18749` per character, truncated to u16), stops at `0xFFFF`. A code outside the charmap, or the `0xF100` compressed-string marker, raises `ReaderError` naming entry and code.
  - `message_bank(rom: RomFiles, decomp: Path, bank: str, charmap: dict[int, str]) -> list[str]` (`bank` a `TEXT_BANK_*` constant; member index = its line in `generated/text_banks.txt`, `res/text/meson.build:4-8`; NARC `msgdata/pl_msg.narc`)
- Test helper (in the test file): `encode_bank(entries: list[str | list[int]], seed, charmap) -> bytes`, the inverse (a `list[int]` entry is written as raw codes).

- [ ] **Step 1: Write failing tests**

```python
def test_round_trip_synthetic(charmap):
    assert decode_bank(encode_bank(["Abra", "Mr. Mime"], 0x1234, charmap), charmap) == ["Abra", "Mr. Mime"]
def test_unknown_code_raises(charmap):
    bad = encode_bank([[0x0FFE]], 0x1234, charmap)        # entry 0: one raw code outside the charmap
    assert 0x0FFE not in charmap
    with pytest.raises(ReaderError, match="entry 0.*0xffe"):
        decode_bank(bad, charmap)
def test_vanilla_banks_match_decomp(vanilla_files, decomp, charmap):
    names = message_bank(vanilla_files, decomp, "TEXT_BANK_SPECIES_NAME", charmap)
    species = extract_species(decomp)                 # upper case in Gen 4: names[1] == "BULBASAUR"
    assert names[:len(species)] == [s["name"] for s in species]
    assert message_bank(vanilla_files, decomp, "TEXT_BANK_ABILITY_NAMES", charmap) == _text_bank(decomp, "ability_names")
def test_kaizo_renamed_move(kaizo_files, decomp, charmap):   # TEXT_BANK_MOVE_NAMES[4] (vanilla Comet Punch) == "Water Ball" (§5.1)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_text.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): decrypt message banks from pl_msg.narc`

### Task 9: Species, experience tables and level-up learnsets

**Files:**
- Create: `tools/rom/readers/personal.py`, `tools/rom/readers/learnsets.py`
- Test: `tests/tools/rom/test_personal.py`

**Interfaces:**
- Consumes: Task 8 `message_bank`, `fixed_members`, `ReaderError`; `tools.extract_decomp.STAT_KEYS`, `_constants`, `_gender_ratios`.
- Produces:
  - `read_species(rom, decomp: Path, charmap) -> list[dict]`: `species.json` schema for ids `0..495` (the `generated/species.txt` count; forms at 496-507 are K2a) from `poketool/personal/pl_personal.narc`, 44-byte `SpeciesData` (`include/struct_defs/species.h:39-58`: stats HP, Atk, Def, Spe, SpA, SpD at 0-5, reordered to `STAT_KEYS`; types 6-7; gender ratio 16; base friendship 18; exp rate 19; abilities 22-23); name from `TEXT_BANK_SPECIES_NAME`; `weight_hg` per Ruling 9.
  - `read_exp_tables(rom) -> list[list[int]]`: `exp_tables.json` schema from `poketool/personal/pl_growtbl.narc` (rate order of `generated/exp_rates.txt`; layout from `tools/scripts/make_pl_growtbl.py`).
  - `read_learnsets(rom, decomp: Path) -> dict[str, list[list[int]]]`: `learnsets.json` schema from `poketool/personal/wotbl.narc`; entries u16 `move | level << 9` up to `0xFFFF` (`speciesproc.c:472-494`).

- [ ] **Step 1: Write failing tests**

```python
def test_species_match_decomp(vanilla_files, decomp, charmap):
    assert read_species(vanilla_files, decomp, charmap) == extract_species(decomp)
def test_exp_tables_match_decomp(vanilla_files, decomp):     # == extract_exp_tables(decomp)
def test_learnsets_match_decomp(vanilla_files, decomp):      # == extract_learnsets(decomp)
def test_member_size_mismatch_raises():                       # Review Focus 4: fixed_members([b"\0"*44, b"\0"*46], 44, "pl_personal.narc") -> ReaderError naming member 1, 44, 46
def test_kaizo_species_read(kaizo_files, decomp, charmap):    # len == 496; Lileep (345) has ABILITY_SOLID_ROCK (§10), which vanilla Lileep lacks
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_personal.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): read species, growth tables and level-up learnsets from the ROM`

### Task 10: Moves and items

**Files:**
- Create: `tools/rom/readers/moves.py`, `tools/rom/readers/items.py`
- Test: `tests/tools/rom/test_moves_items.py`

**Interfaces:**
- Consumes: Task 8 helpers; `SymbolMap.find` (via `symbol_bytes`); `tools.extract_decomp._constants`, `_ai_flag_bits` (as the model for `--no-auto` metang masks).
- Produces:
  - `read_moves(rom, decomp: Path, charmap) -> list[dict]`: `moves.json` schema from `poketool/waza/pl_waza_tbl.narc`, 16-byte `MoveTable` (`include/move_table.h:12-24`: effect u16 0, class 2, power 3, type 4, accuracy 5, pp 6, chance 7, range u16 8, priority s8 10, flags 11). `target` is the `generated/move_ranges.txt` name for the range value, which is a `--no-auto` mask (`generated/meson.build:72`: line 1 is 0, line n + 2 is bit n; any other value raises). `flags` lists set bits in `generated/move_flags.txt` order (line n is bit n). Name from `TEXT_BANK_MOVE_NAMES`.
  - `read_items(rom, decomp: Path, symbols: SymbolMap, charmap) -> list[dict]`: `items.json` schema per Ruling 10; `sItemArchiveIDs` entries are 4 × u16 (data, icon, palette, gen3); data member layout `include/item.h:56-77` (hold effect at 2, param at 3).

- [ ] **Step 1: Write failing tests**

```python
def test_moves_match_decomp(vanilla_files, decomp):   # == extract_moves(decomp)
def test_items_match_decomp(vanilla_files, decomp, symbols):   # == extract_items(decomp)
def test_unknown_range_value_raises():                # a synthetic member with range 0x3 -> ReaderError naming the move id
def test_move_member_size_mismatch_raises():          # Review Focus 4: a 15-byte pl_waza_tbl member
    with pytest.raises(ReaderError, match="pl_waza_tbl.narc.*member 1.*16.*15"):
        fixed_members([bytes(16), bytes(15)], 16, "poketool/waza/pl_waza_tbl.narc")
def test_kaizo_changes(kaizo_files, decomp, symbols, charmap):   # §5.2: move named "Trick Room" priority 7 (vanilla -7); §5.5: "Flame Plate" hold_param 50 (vanilla 20)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_moves_items.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): read moves and items from the ROM`

### Task 11: Trainers (party, class, names)

**Files:**
- Create: `tools/rom/readers/trainers.py`
- Test: `tests/tools/rom/test_trainers.py`

**Interfaces:**
- Consumes: Task 8 helpers; `symbol_bytes`; `tools.extract_decomp._constants`.
- Produces: `party_bytes(member: bytes, trainer_id: int, data_type: int, party_size: int) -> bytes` (the size rule below; returns the member without padding); `read_trainers(rom, decomp: Path, symbols: SymbolMap, charmap, start_conditions: list[str]) -> list[dict]`, the `trainers.json` schema (`extract_trainers` docstring) without derivation of its own: `start_conditions[id]` is passed in (Task 12). `trdata.narc` members are 20-byte `TrainerHeader`s (`include/struct_defs/trainer_data.h:24-32`: data type 0, class 1, party size 3, items u16×4 at 4 (Ruling 11), AI mask u32 at 12). `trpoke.narc` member size must be `round_up_4(party_size × {8, 16, 10, 18}[data type])` (entry sizes `:34-64`, `TRDATATYPE_*` `:17-22`) with every padding byte zero; a trainer with party size 0 has an 8-byte all-zero placeholder member (trainer 0). Any other size, or a non-zero padding byte, raises `ReaderError` naming the trainer id, both sizes and the data type. Measured on vanilla with ndspy 4.2.0: 43 of 928 members are padded (type 3 × 5 mons 92 not 90, ×22; type 3 × 3 56 not 54, ×16; type 2 × 1 12 not 10, ×2; type 2 × 3 32 not 30, ×1; type 3 × 1 20 not 18, ×1; trainer 0's 8-byte placeholder), all padding zero. Species = `species & 0x3FF`, form = `species >> TRAINER_MON_FORM_SHIFT` (`:11`, `src/trainer_data.c` `TrainerData_BuildParty`); moves `None` for data types without moves; item 0 for types without items. `class_gender` from `sTrainerClassGender` (u8 per class, `src/trainer_data.c:9,166`). Names from `TEXT_BANK_NPC_TRAINER_NAMES` (by id) and `TEXT_BANK_TRAINER_CLASS_NAMES` (by class).

- [ ] **Step 1: Write failing tests**

```python
def test_trainers_match_decomp_except_start(vanilla_files, decomp, symbols, charmap, tmp_path):
    extract_trainers(decomp, tmp_path); want = json.loads((tmp_path / "trainers.json").read_text("utf-8"))
    got = read_trainers(vanilla_files, decomp, symbols, charmap, [t["start_condition"] for t in want])
    assert got == want
def test_roark(vanilla_files, decomp, symbols, charmap):    # id 246
    t = read_trainers(vanilla_files, decomp, symbols, charmap, ["none"] * 928)[246]
    assert (t["name"], t["class_id"], t["ai_flags"], t["items"]) == ("Roark", 62, 7, [17, 17])
    assert [m["species"] for m in t["party"]] == [74, 95, 408]
def test_party_size_rule():                            # Review Focus 4, the measured cases; party_bytes(member, trainer_id, data_type, party_size)
    assert len(party_bytes(bytes(92), 1, 3, 5)) == 90 and len(party_bytes(bytes(56), 2, 3, 3)) == 54
    assert len(party_bytes(bytes(12), 3, 2, 1)) == 10 and party_bytes(bytes(8), 0, 0, 0) == b""
    with pytest.raises(ReaderError, match="trainer 4.*56.*48"):     # short: party 3 of type 3 needs 56 (54 padded), 48 given
        party_bytes(bytes(48), 4, 3, 3)
    with pytest.raises(ReaderError, match="trainer 5.*padding"):    # non-zero padding
        party_bytes(bytes(90) + b"\x01\x00", 5, 3, 5)
    with pytest.raises(ReaderError, match="trainer 6"):             # over-long
        party_bytes(bytes(96), 6, 3, 5)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_trainers.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): read trainer headers and parties from the ROM`

### Task 12: Start conditions from map headers, events and scripts

**Files:**
- Create: `tools/rom/readers/maps.py`
- Test: `tests/tools/rom/test_maps.py`

**Interfaces:**
- Consumes: Task 8 helpers, `symbol_bytes`; from `tools.extract_decomp`: `WEATHER_CONDITION`, `START_CONDITIONS_FILE`, `_start_condition`, `_start_condition_overrides`, `_constants`, `ExtractError`.
- Produces:
  - `class ScriptDecodeError(ReaderError)` (script file, offset, opcode)
  - `command_table(decomp: Path) -> dict[int, tuple[str, tuple[int, ...]]]`: opcode → (macro name, operand widths). Opcode = index of its `ScriptCommand(SCRCMD_*, ...)` line in `include/data/scripts/scrcmd.h`. Widths come from the `asm/macros/scrcmd.inc` macro whose body starts `.short SCRCMD_*` and continues with fixed `.byte`/`.short`/`.long` lines (833 such macros). The six variable bodies (`DoStrengthFunc, DoFlashFunc, DoDefogFunc, DoGroupConnectionAction, CallTVBroadcast, MysteryGiftGive`) get explicit decoders that read operands as their `src/scrcmd*.c` handlers do, citing each. Two macros for one opcode must agree, else raise.
  - `script_battles(script: bytes, table) -> set[int]`: trainer ids started by `StartTrainerBattle` (2 × u16, both counted), `StartTagBattle` (3 × u16, all three counted, partner included) and `StartFirstBattle` (1 × u16), walking from every header entry (u32 `label-.-4` offsets up to u16 `0xFD13`, `scrcmd.inc:5105-5116`). It follows `GoTo, Call, GoToIf, CallIf, GoToIfTargetTrainerDefeated` and `ScrCmd_Unused_017/018/019` targets; `ApplyMovement`'s offset is data and is not walked; a path ends at `End`, `Return` or `GoTo`. Operand 0 is no trainer; an operand ≥ `0x4000` resolves through `SetVarFromValue` (Ruling 12).
  - `trainer_start_conditions(rom, decomp: Path, symbols: SymbolMap, *, overrides: Path | None = START_CONDITIONS_FILE) -> list[str]`: one condition per trainer id. `sMapHeaders` is 24-byte `MapHeader`s (`include/map_header.h:10-31`: scripts u16 at 4, events u16 at 16, the map-label bitfield u16 at 18, weather u8 at 20). Events: `zone_event.narc` layout `tools/jsoncnv/event.py:25-85` (object events 32 bytes, script u16 at +10). Yearly weather: `sYearlyWeather` u8 `[366][OVERWORLD_WEATHER_YEARLY_COUNT]` (`src/field_overworld_weather.c:17`). Rematches: `gVsSeekerRematchData` (`src/overlay005/vs_seeker.c:79-81`). Weather ids come from `include/constants/overworld_weather.h`, as in `_weather_conditions`.

- [ ] **Step 1: Write failing tests**

```python
def test_command_table(decomp):
    t = command_table(decomp); by_name = {v[0]: (k, v[1]) for k, v in t.items()}
    assert by_name["StartTrainerBattle"][1] == (2, 2) and by_name["GoToIf"][1] == (1, 4)
def test_script_walk_synthetic(decomp):    # hand-assembled script: entry -> GoToIf -> StartTrainerBattle 246, dead bytes after End are not decoded
def test_setvar_resolution(decomp):        # SetVarFromValue 0x8000, 5; StartTrainerBattle 0x8000, 0 -> {5}
def test_unknown_opcode_raises(decomp):    # opcode past the table -> ScriptDecodeError naming the offset
def test_vanilla_start_conditions_match_decomp(vanilla_files, decomp, symbols, tmp_path):
    extract_trainers(decomp, tmp_path)
    want = [t["start_condition"] for t in json.loads((tmp_path / "trainers.json").read_text("utf-8"))]
    assert trainer_start_conditions(vanilla_files, decomp, symbols) == want   # 810 none, 41 rain, 36 fog, 29 hail, 12 sandstorm
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_maps.py -q`
- [ ] **Step 3: Implement.** If the last test differs, print the differing trainers with the maps each side found: 135 vanilla trainers are placed only by script lines (measured), so a walk that misses code shows up here. Fix the walk when it misses reachable code; do not special-case trainers. If what remains is the decomp scan's textual over-approximation (Ruling 12's stop rule: unreachable lines, or a `SetVar` of a `TRAINER_` value that never reaches a battle command), stop and report those trainers, maps and script lines to the user instead of changing the walk.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(rom): derive trainer start conditions from map headers, events and scripts`

### Task 13: `tools/extract_rom.py` and the vanilla round trip

**Files:**
- Create: `tools/extract_rom.py`
- Modify: `docs/rom_pipeline.md` (section "Extraction": commands, `rom_sources.json`, Ruling 13's refusal)
- Test: `tests/tools/rom/test_extract_rom.py`

**Interfaces:**
- Consumes: everything above; `tools.extract_decomp.extract_static`, `extract_trainers`.
- Produces: CLI `python -m tools.extract_rom --game vanilla|kaizo [--out DIR] [--decomp external/pokeplatinum] [--symbols DEFAULT_MAP] [--config data/local.toml]` (default `--out`: `data/intermediate` for vanilla, `data/intermediate_kaizo` for kaizo); `extract(game: str, out: Path, decomp: Path, symbols: Path, inputs: RomInputs) -> dict[str, str]` (returns the `rom_sources.json` mapping). It writes `species, moves, items, learnsets, exp_tables` (ROM), `trainers` (ROM, with Task 12's conditions), and `types, natures, charmap, abilities` (reused, Ruling 14), all with `extract_decomp._write`'s formatting. Writes go through a temporary directory beside `out` and are moved in file by file with `os.replace`; other files in `out` are kept (Ruling 13). Kaizo: diff, reuse check, reader layout checks, then the refusal.

- [ ] **Step 1: Write failing tests**

```python
def test_vanilla_round_trip(rom_inputs, decomp, symbols, tmp_path):     # spec §6.3, §9.1
    extract_static(decomp, tmp_path / "d"); extract_trainers(decomp, tmp_path / "d")
    sources = extract("vanilla", tmp_path / "r", decomp, DEFAULT_MAP, rom_inputs)   # symbols fixture ensures it exists
    names = sorted(p.name for p in (tmp_path / "d").glob("*.json"))
    assert sorted(p.name for p in (tmp_path / "r").glob("*.json") if p.name != "rom_sources.json") == names
    for n in names:
        assert json.loads((tmp_path / "r" / n).read_text("utf-8")) == json.loads((tmp_path / "d" / n).read_text("utf-8")), n
    assert sources["types.json"].startswith("decomp") and sources["species.json"] == "rom"
def test_kaizo_refused_and_nothing_written(kaizo_inputs, decomp, symbols, tmp_path):
    with pytest.raises(RomInputError, match="K2b"):
        extract("kaizo", tmp_path / "k", decomp, DEFAULT_MAP, kaizo_inputs)
    assert sorted(p.name for p in tmp_path.iterdir()) == []      # no out dir, no leftover temp dir
def test_existing_out_files_kept(rom_inputs, decomp, symbols, tmp_path):   # Ruling 13: per-file replace
    out = tmp_path / "r"; out.mkdir(); (out / "other.json").write_text("{}")
    extract("vanilla", out, decomp, DEFAULT_MAP, rom_inputs)
    assert (out / "other.json").read_text() == "{}" and (out / "species.json").exists()
def test_cli_vanilla_default_out_is_intermediate():   # argument parsing only: --game vanilla -> data/intermediate, kaizo -> data/intermediate_kaizo; bad --game -> SystemExit 2
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run --group rom pytest tests/tools/rom/test_extract_rom.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**, then the real thing and the full checks:

```bash
~/bin/uv.exe run --group rom python -m tools.extract_rom --game vanilla --out .cache/rom/intermediate_check
~/bin/uv.exe run --group rom pytest tests/tools/rom -q          # nothing skipped with the user's inputs
~/bin/uv.exe run pytest -q                                      # background, ~10 min; tools/rom tests skip without the rom group
~/bin/uv.exe run ruff check .
git status --short                                              # no .nds, .narc, map or intermediate files staged
```

- [ ] **Step 5: Commit** `feat(rom): extract_rom CLI; the ROM reader reproduces the decomp intermediates`

---

## Self-review notes (planner)

- Spec coverage: §4 sources (Tasks 1-3, provenance in Task 7), §6.1.1-3 (Tasks 1, 3; Global Constraints), §6.2 including the decomp build and map (Tasks 5-7), §6.3 ROM reader and vanilla round trip (Tasks 8-13), §8 ignores (Task 1), §12.6 toolchain (Task 5). Deferred by design to K2a/K2b, not dropped: forms, TM/tutor/egg learnsets, evolutions, `ai_effect`, `tr_ai_seq`, `be_seq`/`sub_seq` and `pl_enc_data` readers, Kaizo trainer layout, PKCalc and spreadsheet extractors, cross-check, overrides.
- Not placeable as written: the task text's "patched header game code CPUE" (the measured value is `JAK7`; Ruling 1).
- Review round 1 (2026-11-03): trpoke size rule re-measured on vanilla (43 padded members, all zero padding) and restated with `round_up_4`; `MapHeader.eventsArchiveID` is at offset 16 (re-read `include/map_header.h:10-31`); species names are upper case (`data/intermediate/species.json`); `write_report` keeps the fixed three-argument call; Ruling 12 gained a stop rule for the decomp scan's over-approximation (`tools/extract_decomp.py:425-427, 494-528`); Kaizo tests take `kaizo_inputs`; writes are per-file `os.replace`; header diff limited to `HEADER_FIELDS`; every test sketch now carries its assertions. Interfaces re-checked across tasks: `fixed_members`, `party_bytes`, `SymbolMap(symbols)`, `Symbol.section`, `HEADER_FIELDS`, `default_provenance`, `main(argv)`, `kaizo_inputs` are each defined where first produced and used with the same names and arguments later.
