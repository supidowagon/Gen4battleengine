# Save Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `nuzline.saves.load_save(path) -> SaveFile`: party, 18 PC boxes, bag and trainer from a Platinum US `.sav`/`.dsv`, checksums checked the way the game checks them, every Pokemon convertible to the engine's `PokemonSpec`; wired into the CLI.

**Architecture:** `saves/crypto.py` (CRC-16, LCRNG XOR, block shuffle), `saves/layout.py` (offsets and sizes as constants with their arithmetic), `saves/pkm.py` (one record -> `SaveMon`, `to_spec`), `saves/save.py` (container, copy selection, party/boxes/bag/trainer -> `SaveFile`). Box levels need two new table arrays (growth rate, experience table) and nicknames a charmap, both added to `tools/extract_decomp.py` / `tools/build_tables.py`. `PokemonSpec` gains `pp`, `status_counter`, `stats` so the save's state reaches the battle. A test-only writer builds synthetic saves.

**Tech Stack:** Python 3.12, numpy, pytest, hypothesis, ruff. Run everything with `~/bin/uv.exe run ...`.

**Spec:** `docs/superpowers/specs/2026-10-31-save-parser-design.md` (parent `docs/design.md` §4.1, §5, §8 item 3). Read it before any task; section numbers below (§N) are the spec's.

## Global Constraints

- No new runtime dependencies: numpy, jax and the standard library only.
- The decomp (`external/pokeplatinum`, in the main checkout at `../../../external/pokeplatinum` from this worktree) is authoritative; cite file:line in docstrings as existing code does.
- Never commit a save, `.dsv`, ROM or PKHeX export. Oracle files live outside the repo, found through `NUZLINE_ORACLE_DIR`.
- Never touch any folder named `backup`.
- `nuzline.cli` must not import jax at module import (`tests/test_cli.py`); `nuzline.saves` must not import jax at all (it uses `tables.to_numpy`-style numpy arrays: load with `np.load`, see Task 1).
- Errors name the place: `"box 3 slot 12: ..."` / `"party slot 2: ..."`. CLI output is ASCII; exit codes 0 ok / 1 error / 2 refused (unchanged).
- ruff `line-length = 100`; match surrounding docstring density.
- The full suite takes ~10 min: run it in the background; per-task runs use the task's test files.

## Rulings (controller's calls; the spec is silent or slightly off)

1. **Reference engine needs no change.** It is built from a `BattleState` (`reference/convert.py:22` `to_ref`), never from `PokemonSpec`, so spec §6's "reference engine takes the same fields" is satisfied by `_fill_slot` alone.
2. **Refused forms**: `to_spec` refuses `form != 0` for species `ROTOM, WORMADAM, GIRATINA, SHAYMIN, DEOXYS, ARCEUS` (the decomp's `res/pokemon/<species>/forms/` directories whose forms change stats, types or abilities; Burmy, Shellos, Gastrodon, Unown are cosmetic; Castform/Cherrim forms exist only in battle).
3. **`.dsv`**: cut at the first occurrence of `b"|<--Snip above here to create a raw sav by excluding this DeSmuME savedata footer:"`; what remains must be exactly `0x80000` bytes. No marker and length `0x80000` = raw `.sav`. Anything else: `SaveError("not a Platinum save: N bytes")`.
4. **Box level ties**: level = largest `L` in 1..100 with `exp_table[rate][L] <= exp` (`SpeciesData_GetLevelAt`, `src/pokemon.c:2378`).
5. **Empty records**: a box record whose 136 raw bytes are all zero is empty without decrypting; otherwise decrypt and check, and decrypted species 0 is empty. A checksum mismatch on a non-empty record raises.
6. **Bag** is returned as `dict[str, list[tuple[int, int]]]` keyed `items, key_items, tms_hms, mail, medicine, berries, poke_balls, battle_items`, each `(item id, quantity)` up to the first `item == 0`.

## Review Focus

1. **Real saves with a fainted party member** (`current_hp == 0`): must load and convert (the engine already accepts hp 0); test in Task 5.
2. **A box Pokemon whose stored PP exceeds the base max because of PP Ups**: must convert (max is `base * (5 + ups) // 5`); test in Task 5.
3. **Party `currentCount` of 0 or > 6** (garbage or new-game save): `SaveError`, not an IndexError; test in Task 6.
4. **An egg in party slot 0** with `--save`: the battle party drops it and `--lead 0` means the first non-egg; test in Task 7.
5. **Counter wrap** (`0xFFFFFFFF` vs `0`) picking the wrong copy; test in Task 6.

---

### Task 1: Growth tables and charmap in the data layer

**Files:**
- Modify: `tools/extract_decomp.py` (`extract_species` adds `"exp_rate"`; new `extract_exp_tables`, `extract_charmap`; `extract_static` writes `exp_tables.json`, `charmap.json`)
- Modify: `tools/build_tables.py` (arrays `species_exp_rate`, `exp_table`; manifest `charmap`; `MANIFEST_VERSION = 6`)
- Modify: `src/nuzline/data/tables.py` (`Tables` fields; `Names.charmap`)
- Test: `tests/tools/test_extract_decomp.py`, `tests/data/test_tables.py`

**Interfaces:**
- Produces: `Tables.species_exp_rate` int8[S] (index into `generated/exp_rates.txt` order: 0 medium_fast, 1 erratic, 2 fluctuating, 3 medium_slow, 4 fast, 5 slow); `Tables.exp_table` int32[8, 101] (row = rate, column = level, from `res/pokemon/.shared/exp_tables.csv`, whose column order is that same order); `Names.charmap: dict[int, str]` (code -> character; `{command}` entries and comments skipped).

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_extract_decomp.py
def test_species_exp_rate(decomp):
    species = {s["name"]: s for s in extract_species(decomp)}
    assert species["Abra"]["exp_rate"] == 3        # EXP_RATE_MEDIUM_SLOW
def test_exp_tables(decomp):
    t = extract_exp_tables(decomp)
    assert len(t) == 8 and all(len(r) == 101 for r in t)
    assert t[0][100] == 1_000_000 and t[3][100] == 1_059_860 and t[5][100] == 1_250_000
def test_charmap(decomp):
    cm = extract_charmap(decomp)
    assert cm[0x0121] == "0" and cm[0x0145] == "a" and cm[0x01AB] == "!"
```
(Use the existing decomp fixture/skip convention of that file.)

```python
# tests/data/test_tables.py
def test_growth_tables(tables, names):
    assert int(tables.species_exp_rate[names.species("Abra")]) == 3
    assert int(tables.exp_table[0, 100]) == 1_000_000
def test_charmap_loaded(names):
    assert names.charmap[0x0145] == "a"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/tools/test_extract_decomp.py tests/data/test_tables.py -q`

- [ ] **Step 3: Implement** the three extractors, the two arrays and manifest key, the `Tables` fields and `Names.charmap` (JSON keys are hex strings; convert to int). `exp_rate` maps through `generated/exp_rates.txt` with the existing `_constants`/`_index` helpers.

- [ ] **Step 4: Rebuild tables in this worktree and run, expect PASS**

```bash
~/bin/uv.exe run python -m tools.extract_decomp --decomp ../../../external/pokeplatinum --out data/intermediate
~/bin/uv.exe run python -m tools.build_tables
~/bin/uv.exe run pytest tests/tools/test_extract_decomp.py tests/data/test_tables.py -q
```

- [ ] **Step 5: Commit** `feat(data): growth rates, experience tables and charmap for the save parser`

### Task 2: Crypto primitives

**Files:**
- Create: `src/nuzline/saves/__init__.py` (empty for now), `src/nuzline/saves/crypto.py`
- Test: `tests/saves/__init__.py`, `tests/saves/test_crypto.py`

**Interfaces:**
- Produces:
  - `crc16_ccitt(data: bytes) -> int` (poly 0x1021, init 0xFFFF, no xorout)
  - `lcrng_xor(data: bytes, seed: int) -> bytes` (XOR each little-endian u16 with `(seed := seed*1103515245+24691 & 0xFFFFFFFF) >> 16`; length must be even)
  - `BLOCK_ORDERS: tuple[tuple[int, int, int, int], ...]` (32 entries: for shuffle value `v`, the stored position of blocks A, B, C, D, transcribed from `BoxPokemon_GetDataBlock`, `src/pokemon.c:4861`)
  - `record_checksum(blocks: bytes) -> int` (u16 sum of the 64 u16s of 128 decrypted, unshuffled block bytes)
  - `decrypt_record(raw: bytes) -> bytes` (136 or 236 bytes in; same length out with header unchanged, blocks decrypted and put in A, B, C, D order, party extension decrypted with the personality seed — PKHeX `.pk4` layout)
  - `encrypt_record(plain: bytes) -> bytes` (exact inverse; uses the checksum already in bytes 6-7)

- [ ] **Step 1: Write failing tests**

```python
def test_crc_check_value():
    assert crc16_ccitt(b"123456789") == 0x29B1
def test_lcrng_first_words():
    out = lcrng_xor(b"\0" * 4, 0)
    assert out == (24691 >> 16).to_bytes(2, "little") + (((24691 * 1103515245 + 24691) & 0xFFFFFFFF) >> 16).to_bytes(2, "little")
def test_block_orders_are_permutations_and_repeat():
    assert all(sorted(o) == [0, 1, 2, 3] for o in BLOCK_ORDERS)
    assert len(set(BLOCK_ORDERS[:24])) == 24 and BLOCK_ORDERS[24:] == BLOCK_ORDERS[:8]
    assert BLOCK_ORDERS[0] == (0, 1, 2, 3) and BLOCK_ORDERS[6] == (1, 0, 2, 3)
@given(st.binary(min_size=236, max_size=236), st.booleans())
def test_encrypt_decrypt_round_trip(plain, party):
    plain = plain if party else plain[:136]
    assert decrypt_record(encrypt_record(plain)) == plain
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/saves/test_crypto.py -q`
- [ ] **Step 3: Implement** with numpy u16 views for the XOR (a box holds 540 records; stay vectorized per record).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(saves): CRC-16, LCRNG stream and block shuffle`

### Task 3: Record decoding to `SaveMon`, and the test writer

**Files:**
- Create: `src/nuzline/saves/pkm.py`, `tests/saves/writer.py`
- Test: `tests/saves/test_pkm.py`

**Interfaces:**
- Consumes: Task 1 `Tables.species_exp_rate`, `Tables.exp_table`, `Names.charmap`; Task 2 `decrypt_record`, `record_checksum`, `encrypt_record`.
- Produces:
  - `class RecordError(ValueError)`
  - `SaveMon` frozen dataclass exactly as spec §5 lists it.
  - `level_from_exp(tables, species: int, exp: int) -> int` (Ruling 4)
  - `decode_mon(raw: bytes, location: tuple[str, int, int], tables, charmap: dict[int, str]) -> SaveMon | None` (Ruling 5; for 236-byte records also status, level, HP, stored stats; party level must equal `level_from_exp`, else `RecordError`). `tables` may be numpy or jax arrays; index with `int(...)`.
  - `tests/saves/writer.py`: `mon_record(*, species, personality=0, ot_id=0, exp=None, level=5, nature=None, ability=0, ivs=(0,)*6, evs=(0,)*6, moves=(1,0,0,0), pp=(35,0,0,0), pp_ups=(0,)*4, item=0, friendship=70, gender=0, form=0, is_egg=False, nickname_codes=(0xFFFF,), party=None) -> bytes` returning an **encrypted** record; `party` is `None` (136 bytes) or a dict `{status, level, hp, stats}` (236 bytes). `exp=None` means `exp_table[rate][level]`. `nature` given forces nothing; the caller picks `personality` with `personality % 25 == nature` (helper `personality_for(nature, shuffle)`).

Field offsets (spec §4; block offsets within each 32-byte block, `include/struct_defs/pokemon.h`): A: species 0x00, item 0x02, OT ID 0x04, exp 0x08, friendship 0x0C, ability 0x0D, EVs 0x10-0x15 (HP, Atk, Def, **Spe, SpA, SpD**); B: moves 0x00, PP 0x08, PP Ups 0x0C, IV word 0x10 (5 bits each HP, Atk, Def, **Spe, SpA, SpD**, bit 30 egg, bit 31 nickname), byte 0x18 (bit 0 fateful, bits 1-2 gender, bits 3-7 form); C: nickname 0x00 (11 u16, 0xFFFF-terminated); party extension at record 0x88: status u32 0x88, level 0x8C, HP 0x8E, max HP 0x90, Atk, Def, **Spe**, SpA, SpD 0x92-0x9A. Store everything in `ids.Stat` order (HP, Atk, Def, SpA, SpD, Spe).

- [ ] **Step 1: Write failing tests**

```python
def test_level_from_exp(tables, names):
    abra = names.species("Abra")                       # medium slow
    assert level_from_exp(tables, abra, 0) == 1
    assert level_from_exp(tables, abra, int(tables.exp_table[3, 37])) == 37
    assert level_from_exp(tables, abra, int(tables.exp_table[3, 37]) - 1) == 36
def test_all_zero_record_is_empty(tables, names):
    assert decode_mon(b"\0" * 136, ("box", 0, 0), tables, names.charmap) is None
@given(...)  # species from a fixed list of 5 ids, personality any u32, ivs 0-31, evs 0-255,
             # moves/pp/pp_ups, item, ability, gender, nature via personality
def test_box_round_trip(tables, names, ...):
    mon = decode_mon(mon_record(...), ("box", 2, 7), tables, names.charmap)
    assert (mon.species, mon.ivs, mon.evs, mon.moves, mon.pp, mon.pp_ups, mon.level,
            mon.nature, mon.location) == (..., personality % 25, ("box", 2, 7))
    assert mon.current_hp is None and mon.status_raw == 0 and mon.stored_stats is None
def test_party_fields(tables, names):  # status, hp, stored_stats decoded in ids.Stat order
def test_checksum_mismatch_raises(tables, names):
    raw = bytearray(mon_record(species=63)); raw[0x20] ^= 1
    with pytest.raises(RecordError, match="box 0 slot 3"):
        decode_mon(bytes(raw), ("box", 0, 3), tables, names.charmap)
def test_party_level_disagreeing_with_exp_raises(tables, names):
def test_nickname_decoded(tables, names):  # codes for "Abra" -> "Abra"
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/saves/test_pkm.py -q`
- [ ] **Step 3: Implement** `decode_mon`, `level_from_exp`, the writer (writer computes the checksum with `record_checksum` and encrypts with `encrypt_record`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(saves): decode Pokemon records into SaveMon`

### Task 4: `PokemonSpec` carries PP, sleep turns and stats

**Files:**
- Modify: `src/nuzline/data/spec.py`, `src/nuzline/engine/state.py:226-255` (`_fill_slot`), `src/nuzline/data/party.py`
- Test: `tests/engine/test_state.py` (or the existing test file covering `make_state`), `tests/data/test_party.py`

**Interfaces:**
- Produces: `PokemonSpec.pp: tuple[int,int,int,int] | None = None`, `PokemonSpec.status_counter: int = 0`, `PokemonSpec.stats: tuple[int,...] | None = None` (ids.Stat order, max HP first). Party TOML optional keys `pp` (4 ints, one per listed move slot, padded with 0), `sleep_turns` (int -> `status_counter`, requires `status = "sleep"`), `stats` (6 ints), and `status` (`none/sleep/poison/burn/freeze/paralysis/toxic`, default none) — the TOML has no status key today and `save export` needs it.

- [ ] **Step 1: Write failing tests**

```python
def test_make_state_uses_spec_pp_counter_and_stats(tables, sample_spec):
    spec = replace(sample_spec, pp=(3, 0, 1, 2), status=Status.SLEEP, status_counter=2,
                   stats=(50, 30, 31, 32, 33, 34), current_hp=20)
    s = make_state(tables, [spec], [sample_spec])
    assert s.pp[0, 0].tolist() == [3, 0, 1, 2] and int(s.status_counter[0, 0]) == 2
    assert int(s.max_hp[0, 0]) == 50 and s.stats[0, 0].tolist() == [30, 31, 32, 33, 34]
def test_make_state_defaults_unchanged(tables, sample_spec):  # pp=None -> move_pp, stats=None -> calc_stats
def test_current_hp_checked_against_override(tables, sample_spec):  # hp 51 with stats[0]=50 -> ValueError
def test_party_toml_new_keys(tmp_path, names):  # pp, status, sleep_turns, stats parse; sleep_turns without sleep -> ValueError
```

- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement.** Engine field order of `stats` is Atk..Spe (`state.py` `a["stats"] = stats[ATK:SPE+1]`).
- [ ] **Step 4: Run, expect PASS**, then `tests/engine tests/data -q` in the background to confirm nothing else moved.
- [ ] **Step 5: Commit** `feat(engine): PokemonSpec carries current PP, sleep turns and stored stats`

### Task 5: `SaveMon.to_spec`

**Files:**
- Modify: `src/nuzline/saves/pkm.py`; move `decode_status` (and its bit constants) from `tools/trace_to_state.py` into `src/nuzline/saves/status.py`, with `tools/trace_to_state.py` importing it
- Test: `tests/saves/test_to_spec.py`; existing `tests/test_trace_to_state.py` must still pass

**Interfaces:**
- Consumes: Task 3 `SaveMon`, Task 4 spec fields, `engine/stats.calc_stats`.
- Produces: `SaveMon.to_spec(self, tables) -> PokemonSpec` raising `RecordError` (message starts with the location) for: egg; Ruling 2 form; ability not in the species' two decomp abilities (`tables.species_abilities_decomp`); PP above `base * (5 + ups) // 5`; party stat outside the band `calc_stats(evs=0) <= stored <= calc_stats(evs)` per stat (§5). `status`/`status_counter` from `decode_status(status_raw)`, toxic counter forced to 0. Party: `stats=stored_stats`, `current_hp`; box: `stats=None`, `current_hp=None`.

- [ ] **Step 1: Write failing tests** (build `SaveMon`s with `dataclasses.replace` on a decoded writer record)

```python
def test_box_mon_converts(...)            # stats None, hp None, pp copied, ability_slot right
def test_party_mon_with_lagging_stats_converts(...)  # stored = calc(evs=0), evs nonzero -> ok
def test_party_stat_outside_band_raises(...)         # stored Atk = calc(evs)+1 -> "party slot 1: Atk"
def test_fainted_party_mon_converts(...)             # current_hp 0
def test_pp_with_pp_ups_converts_and_above_max_raises(...)  # Tackle 35, ups 3 -> 56 ok, 57 raises
def test_sleep_status_and_counter(...)               # status_raw 3 -> SLEEP, 3
def test_toxic_counter_dropped(...)                  # TOXIC | counter bits -> TOXIC, 0
def test_rotom_form_raises_unown_form_passes(...)
def test_egg_raises(...)
def test_foreign_ability_raises(...)
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/saves/test_to_spec.py tests/test_trace_to_state.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(saves): convert SaveMon to PokemonSpec with the stat band check`

### Task 6: Save container, copy selection and `load_save`

**Files:**
- Create: `src/nuzline/saves/layout.py`, `src/nuzline/saves/save.py`; export `load_save, SaveFile, SaveError, SaveMon, RecordError` from `src/nuzline/saves/__init__.py`
- Modify: `tests/saves/writer.py` (`build_image`)
- Test: `tests/saves/test_save.py`

**Interfaces:**
- Consumes: Tasks 2, 3.
- Produces:
  - `layout.py` constants (spec §1-3, with the arithmetic as comments): `SAVE_SIZE = 0x80000`, `SECTOR_SIZE = 0x1000`, `COPY_OFFSETS = (0, 64 * 0x1000)`, `NORMAL_SIZE = 0xCF2C`, `BOXES_OFFSET = 0xCF2C`, `BOXES_SIZE = 0x121E4`, `FOOTER_SIZE = 20`, `SIGNATURE = 0x20060623`, `TRAINER_NAME = 0x68`, `TRAINER_ID = 0x78`, `BADGES = 0x82`, `PARTY_COUNT = 0x9C`, `PARTY_MONS = 0xA0`, `PARTY_MON_SIZE = 0xEC`, `BAG = 0x630`, `POCKETS` (name, size) in Ruling 6 order with sizes 165, 50, 100, 12, 40, 64, 15, 30, `BOX_MONS = 4` (within the boxes block), `BOX_MON_SIZE = 0x88`, `BOX_COUNT = 18`, `BOX_SLOTS = 30`.
  - `class SaveError(ValueError)`
  - `@dataclass(frozen=True) SaveFile`: `trainer_name: str`, `tid: int`, `sid: int`, `badges: int`, `party: tuple[SaveMon, ...]`, `boxes: tuple[tuple[SaveMon | None, ...], ...]` (18 × 30), `bag: dict[str, list[tuple[int, int]]]`, `load_result: str` (`"ok"`/`"corrupt"`), `normal_copy: str`, `boxes_copy: str` (`"primary"`/`"backup"`), `save_counter: int`, `sha256: str` (of the file as read).
  - `read_image(data: bytes) -> bytes` (Ruling 3)
  - `select_copies(image: bytes) -> tuple[str, int, int]` — `(load_result in "ok"|"corrupt"|"error"|"empty", normal copy index, boxes copy index)`, a line-for-line port of `SaveData_LoadCheck` / `SaveCheckInfo_CompareSectors` / `SaveCheckInfo_CompareCounters` (`src/savedata.c:375-523`), including the final `else` branch.
  - `load_save(path: Path, *, allow_corrupt: bool = False, tables=None, names=None) -> SaveFile` (defaults: numpy arrays from `data/gen4_platinum/tables.npz` via `np.load` and `load_names()`; no jax import).
  - writer: `build_image(*, party: list[bytes] = (), boxes: dict[tuple[int, int], bytes] = {}, bag: dict = {}, trainer_name_codes=(), tid=0, sid=0, badges=0, counters=((1, 1, 1), None)) -> bytes` — `counters[i]` is `(save counter, normal block counter, boxes block counter)` for copy i or `None` for an erased (0xFF) copy; returns a 0x80000 image with valid footers; helper `corrupt(image, copy, block) -> bytes` flips one body byte.

- [ ] **Step 1: Write failing tests**

```python
def test_raw_and_dsv(tmp_path):         # same image loads from .sav and from image + marker + 122 bytes
def test_wrong_size_raises(tmp_path):    # 0x40000 bytes -> SaveError("not a Platinum save")
def test_party_boxes_bag_trainer(tmp_path, tables, names):
    # 2 party mons, box (0,0) and (17,29) filled, two bag items, trainer name, tid/sid, badges 0b101
def test_newer_backup_wins(tmp_path):    # counters ((5,5,5),(6,6,6)) -> normal_copy == "backup"
def test_counter_wrap(tmp_path):         # ((0xFFFFFFFF,)*3, (0,0,0)) -> backup
def test_one_copy_corrupt_is_ok(tmp_path):     # newest normal block corrupt, older valid: load_result "corrupt" refused by default, loads with allow_corrupt=True
def test_mismatched_counters_corrupt(tmp_path) # normal and boxes newest disagree on save counter -> "corrupt"
def test_empty_and_error(tmp_path):      # both copies erased -> SaveError("empty"); normal valid, boxes invalid in both -> SaveError
def test_party_count_out_of_range(tmp_path):   # currentCount 7 and 0 -> SaveError
def test_bad_record_names_location(tmp_path):  # corrupted box record -> RecordError "box 4 slot 9"
def test_layout_sizes_match_decomp_arithmetic():
    assert BOXES_SIZE == (lambda s: s + (4 - s % 4) + 4)(4 + 18*30*0x88 + 18*20*2 + 18 + 1) + 20
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/saves/test_save.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS** (`tests/saves -q`)
- [ ] **Step 5: Commit** `feat(saves): load_save with the game's copy selection`

### Task 7: CLI

**Files:**
- Modify: `src/nuzline/cli.py` (module docstring lists the new commands), `tests/conftest.py` only if a shared synthetic-save fixture is wanted
- Create: `src/nuzline/saves/export.py` (`party_toml(save, names) -> str`, `save_json(save, names) -> dict`, `show_lines(save, names) -> list[str]`)
- Test: `tests/test_cli.py`, `tests/saves/test_export.py`

**Interfaces:**
- Consumes: Task 6 `load_save`, Task 5 `to_spec`, Task 4 TOML keys.
- Produces: `nuzline save show SAVE [--json] [--allow-corrupt]`, `nuzline save export SAVE -o PATH [--allow-corrupt]`; `fight()` arguments become a required mutually exclusive `--party PATH | --save PATH` plus `--allow-corrupt`; `_fight` builds the player party from the save (eggs dropped, each reported on stderr as `"party slot N: egg left out"`; `--lead` indexes the egg-free party); the solve run record has `save_sha256` instead of `party_sha256` when `--save` is used. `SaveError`/`RecordError` -> one line on stderr, exit 1.

- [ ] **Step 1: Write failing tests** (synthetic save built with `tests/saves/writer.py` from the sample party's species/moves at levels whose stats the band accepts)

```python
def test_save_show(tmp_path, capsys)            # lists trainer, "copy: primary", party lines, ASCII only
def test_save_show_json(tmp_path, capsys)       # json.loads works; party[0]["species_name"]
def test_save_export_round_trips(tmp_path)      # load_party(exported) == [m.to_spec(t) for m in party]
def test_rank_with_save(tmp_path, capsys)       # rank --save S --trainer roark --n 20 -> exit 0
def test_save_egg_first_is_dropped(tmp_path, capsys)  # stderr names slot 0; --lead 0 is the first non-egg
def test_party_and_save_exclusive()              # both -> argparse error (SystemExit 2)
def test_corrupt_save_exit_1(tmp_path, capsys)
def test_cli_still_imports_without_jax()         # existing test keeps passing
```

- [ ] **Step 2: Run, expect FAIL** — `~/bin/uv.exe run pytest tests/test_cli.py tests/saves/test_export.py -q`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(cli): nuzline save show/export and --save on fight commands`

### Task 8: PKHeX oracle test and docs

**Files:**
- Create: `tests/saves/test_pkhex_oracle.py`, `docs/saves.md` (how to produce the oracle folder from PKHeX, how to run the test)
- Modify: `docs/design.md` §8 item 3 (link the spec, mark done when merged), `README.md` (one usage line for `nuzline save show`)

**Interfaces:**
- Consumes: Task 2 `decrypt_record`, Task 6 `load_save`, `layout` offsets.

The test skips unless `NUZLINE_ORACLE_DIR` is set. In that folder: exactly one `*.sav`/`*.dsv`, one `*.csv` (PKHeX Box Data Report), and party `*.pk4` files. Steps:
1. `load_save` succeeds with `load_result == "ok"`.
2. For each party slot, the decrypted 236- or 136-byte record (`decrypt_record` of the raw bytes at `PARTY_MONS + i*PARTY_MON_SIZE` in the selected copy) equals some `.pk4` file's bytes over its length (match by personality = first 4 bytes; PKHeX may write 136-byte `.pk4`s, compare the common prefix).
3. For each CSV row, map columns by header (case-insensitive, fail listing the headers found if any of box, slot, species, level, nature, ability, held item, move 1-4, IVs, EVs is missing) and compare with `save.boxes[box-1][slot-1]` via `names` (box/slot 1-based in PKHeX).

- [ ] **Step 1: Write the test and `docs/saves.md`.**
- [ ] **Step 2: Run without the variable** — `~/bin/uv.exe run pytest tests/saves/test_pkhex_oracle.py -q` → 1 skipped.
- [ ] **Step 3: Commit** `test(saves): PKHeX oracle test against a user-supplied export`
- [ ] **Step 4: Full suite in the background** — `~/bin/uv.exe run pytest -q` and `~/bin/uv.exe run ruff check .`; both clean. The oracle run itself waits for the user's export.
