# Save Parser (Milestone 3) — Design

Date: 2026-10-31 · Status: approved by the user 2026-10-31 · Parent spec: `docs/design.md` (§4.1, §5, §8 item 3)

## Why

Every command takes the player's team as a hand-written party TOML today
(`nuzline.data.party.load_party`). Prep search (milestone 4) has to choose a team from
the player's **box**, which nobody will type in by hand, and a nuzlocker's real party has
partial HP, spent PP and statuses that the TOML only half covers. Milestone 3 reads all of
it straight from the player's Platinum save: party, every PC box and the bag, exactly as
the engine and solver take them.

## Decisions taken with the user

- Our own pure-Python parser, not a wrapper around an existing tool. PKHeX is used only as
  the **correctness oracle** in tests (decided before this milestone).
- The design proposed in the session of 2026-10-31 was accepted as a whole ("Looks right,
  write spec"): package layout, `SaveMon` + `to_spec`, the `PokemonSpec` PP/sleep
  extension, refusing corrupt saves by default, eggs dropped from the battle party, the
  CLI commands and `--save` on the existing commands, and the PKHeX oracle test fed by
  `NUZLINE_ORACLE_DIR`.
- After the builder check (2026-10-31), two more answers from the user:
  - **Party Pokemon battle with their stored stats.** Platinum adds battle EVs without
    recalculating stats (`BattleScript_CalcEffortValues`, `src/battle/battle_script.c`,
    calls no `Pokemon_CalcStats`; stats are recomputed only on level-up, items,
    evolution or a trip through the box), and the game battles with the stored values.
    `PokemonSpec` gains a stats override (§6) and the party check becomes a band check
    (§5).
  - **Alternate forms are refused, not supported, in this milestone.** The engine's
    tables are per species; per-form tables move to coverage expansion (milestone 5).

## Goals

1. `nuzline.saves.load_save(path) -> SaveFile`: reads a raw `.sav` or a DeSmuME `.dsv`,
   validates both save copies the way the game does, and returns the party, all 18 boxes,
   the bag and the trainer (name, ID, badges).
2. Every Pokemon comes out as a `SaveMon` holding everything the save stores that the
   engine or prep search can use, and `SaveMon.to_spec(tables)` gives the engine's
   `PokemonSpec`, including current HP, current PP and status.
3. The CLI: `nuzline save show`, `nuzline save export`, and `--save` as an alternative to
   `--party` on `coverage`, `rank` and `solve`.
4. Tests: crypto vectors, an encode/decode round trip on synthetic saves, and a PKHeX
   oracle test on the user's own save (skipped when not supplied).

## Non-goals

- Writing or editing saves (the test suite has a writer for synthetic saves only; it is
  not exported as an API or command).
- HGSS or Diamond/Pearl layouts. Platinum US only (`docs/design.md` §2). The layout
  constants live in one module so another Gen 4 game is a new constants set later.
- The extra save area (Hall of Fame, Battle Frontier, recordings: `ExtraSaveTableEntryID`).
- Daycare Pokemon, the Underground, Pal Park, mail contents, ribbons, contest stats.
- Using the save to pick the fight or level caps (badges are reported, nothing reads
  them yet).

## Anti-goals (a failure even if it runs)

- **Silently wrong Pokemon.** A record whose checksum fails, a party member whose stored
  stats fall outside the band of §5, an alternate form the engine would treat as the base
  form, or an unknown species/move/item id is an error that names the box and slot, never
  a quietly skipped, zero-filled or base-form mon.
- **Loading the wrong copy.** Picking the stale save copy (the player's previous save)
  instead of the one the game itself would load.
- **The user's save or ROM in the repo.** `.gitignore` already excludes `*.sav`, `*.dsv`
  and `*.nds`; oracle files live outside the repo.
- **A new runtime dependency.** Standard library, numpy and jax only.

## How it works

### 1. File container (`saves/save.py`)

- Platinum's save is 512 KiB (`0x80000`). A DeSmuME `.dsv` is the same image followed by a
  footer that starts with the ASCII marker `|<--Snip above here to create a raw sav by
  excluding this DeSmuME savedata footer:`; the parser cuts at the marker. Any other length
  is an error. (Builder: confirm the exact marker text and footer length from a real
  `.dsv`; the rule "cut at the marker, then require 0x80000 bytes" is the decision.)
- The image holds two copies of the game's save, the primary at sector 0 and the backup at
  sector 64 (`PRIMARY_SECTOR_START` / `BACKUP_SECTOR_START`,
  `include/constants/savedata/savedata.h`; sector size `0x1000`,
  `include/constants/savedata/save_table.h`; `SaveData_SaveOffset`, `src/savedata.c:299`).

### 2. Blocks, footers and copy selection (`saves/layout.py`, `saves/save.py`)

- Each copy holds two blocks back to back: **normal** (`SAVE_BLOCK_ID_NORMAL`) at offset 0
  and **boxes** (`SAVE_BLOCK_ID_BOXES`) right after it (`SaveBlockInfo_Init`,
  `src/savedata.c:852`: offsets accumulate block sizes, they are not sector-aligned).
- A block's size is the sum of its save-table pages plus a 20-byte footer. A page's size is
  its struct size rounded as `size += 4 - size % 4; size += 4` (`SaveTableEntry_BodySize`,
  `src/savedata.c:813`; pages in `gSaveTable` order, `src/savedata/save_table.c:48`). For
  Platinum that gives normal = `0xCF2C`, boxes = `0x121E4` (`PCBoxes` is
  `4 + 18*30*0x88 + 18*20*2 + 18 + 1` = `0x121C7` bytes, `include/pc_boxes.h:17`).
  `layout.py` holds these as constants with the arithmetic in comments; the normal-block
  size is checked against real saves by the footer rule below and by the oracle test.
- The footer is `SaveBlockFooter` (`include/savedata.h:8`): `u32 saveCounter`,
  `u32 blockCounter`, `u32 size`, `u32 signature`, `u8 saveBlockID`, `u16 checksum` at
  byte 18 (20 bytes with padding). A block is **valid** when, as in
  `SaveBlockFooter_Validate` (`src/savedata.c:327`), `size` equals the block size,
  `signature == 0x20060623` (`SECTOR_SIGNATURE`), `saveBlockID` is the block's id, and
  `checksum` equals CRC-16 over the block minus its footer
  (`SaveData_CalculateFooterChecksum`, `src/savedata.c:294`).
- The checksum is `MATH_CalcCRC16CCITT` (`CalcCRC16Checksum`, `src/math_util.c:238`):
  CRC-16/CCITT, polynomial `0x1021`, initial value `0xFFFF`, no final XOR (CRC-16/
  CCITT-FALSE, check value `0x29B1` for `"123456789"`). Builder: the NitroSDK source is not
  in the decomp; the check value test plus a valid real save confirm the variant.
- **Copy selection mirrors `SaveData_LoadCheck` (`src/savedata.c:441`) exactly**,
  including `SaveCheckInfo_CompareCounters`' wrap rule (`0xFFFFFFFF` is older than `0`,
  `:375`) and `SaveCheckInfo_CompareSectors` (`:390`). Normal and boxes blocks are chosen
  independently and must agree on the global save counter. The result is one of the
  decomp's `LoadResult`s: `OK`, `CORRUPT` (it fell back to an older block), `ERROR` or
  `EMPTY`.
- `load_save` raises `SaveError` on `EMPTY` and `ERROR`, and on `CORRUPT` unless called with
  `allow_corrupt=True` (CLI `--allow-corrupt`), in which case it loads the blocks the game
  would load and the `SaveFile` records `load_result="corrupt"`. `SaveFile` always records
  which copy each block came from and the save counter.

### 3. Offsets inside the normal block (Platinum US)

Page locations follow from the page sizes (`SavePageInfo_Init`, `src/savedata.c:826`).
The decomp is built with mwcc for ARM, where `s64` is 4-byte aligned, which is what makes
`SystemData` 92 bytes:

| What | Offset | Source |
|---|---|---|
| `SystemData` page | `0x00` (page `0x64`) | `include/system_data.h:18` |
| `PlayerSave` page | `0x64` (page `0x34`) | `include/save_player.h:9` |
| trainer name (8 `charcode_t`, `0xFFFF`-terminated) | `0x68` | `TrainerInfo`, `include/trainer_info.h:9` |
| trainer ID (`u32`, low half = TID, high = SID) | `0x78` | same |
| badge mask (`u8`) | `0x82` | same |
| `Party` page: `capacity`, `currentCount` | `0x98`, `0x9C` | `include/party.h:11` |
| party Pokemon, 6 × `0xEC` | `0xA0` | `struct Pokemon`, `include/struct_defs/pokemon.h` |
| `Bag` page: 8 pockets of `{u16 item, u16 quantity}` | `0x630` | `include/bag.h:30-45` |

Pocket order and sizes are the struct's: items 165, key items 50, TMs/HMs 100, mail 12,
medicine 40, berries 64, Poke Balls 15, battle items 30 (`include/bag.h:9-16`); a pocket
ends at its first empty (`item == 0`) slot. The PC boxes are at the start of the boxes
block: `u32 currentBoxID`, then `BoxPokemon boxMons[18][30]` (`0x88` each), then the box
names (`include/pc_boxes.h:17`).

These offsets are derived, not taken from PKHeX; the oracle test is what proves them.

### 4. Pokemon records (`saves/crypto.py`, `saves/pkm.py`)

- A `BoxPokemon` is `0x88` bytes (`include/struct_defs/pokemon.h`): `u32 personality`,
  `u16` flags, `u16 checksum`, then four 32-byte data blocks A–D. A party `Pokemon` adds the
  `0x64`-byte `PartyPokemon` (status, level, HP, the six stats, mail, ball capsule).
- **Decryption**: the 128 bytes of blocks are XORed, one `u16` at a time, with the LCRNG
  stream seeded by the record's `checksum`, and the party extension with the stream seeded
  by `personality` (`Pokemon_EncryptData` → `EncodeData`, `src/pokemon.c:4817`,
  `src/math_util.c:216`; `seed = seed * 1103515245 + 24691`, output `seed >> 16`,
  `include/math_util.h:8-9`).
- **Checksum**: the 16-bit sum of the 64 decrypted block `u16`s must equal `checksum`
  (`Pokemon_GetDataChecksum`, `src/pokemon.c:4827`). A mismatch raises (anti-goal 1).
- **Shuffle**: `(personality & 0x3E000) >> 13` picks one of 24 block orders
  (`BoxPokemon_GetDataBlock`, `src/pokemon.c:4861`); `crypto.py` transcribes the table
  with the decomp's case numbers.
- **Empty slots**: a record whose decrypted species is 0 is an empty slot (box) or ignored
  past `currentCount` (party). An all-zero record decrypts to species 0 because the stored
  checksum seeds the stream (`BoxPokemon_Init`, `src/pokemon.c:299`).
- **Fields** (`PokemonDataBlockA`–`D`): species, held item, OT ID, experience, friendship,
  ability (decomp ability id), EVs (stored HP, Atk, Def, **Spe**, SpA, SpD — reordered to
  `ids.Stat` order), moves, current PP, PP Ups, IVs (packed 5-bit, same Spe-before-SpA
  order) with the egg and nickname bits, gender (2 bits), form (5 bits), nickname,
  met level, Poke Ball. Nature is `personality % 25` (`Pokemon_GetNatureOf`,
  `src/pokemon.c:2410`).

### 5. `SaveMon` and conversion to the engine (`saves/pkm.py`)

```python
@dataclass(frozen=True)
class SaveMon:
    location: tuple[str, int, int]  # ("party", 0, slot) or ("box", box, slot)
    personality: int
    ot_id: int
    species: int  # decomp ids throughout
    form: int
    nickname: str
    is_egg: bool
    experience: int
    level: int  # stored (party) or computed from experience (box)
    nature: int
    ability: int  # decomp ability id as stored
    gender: int  # 0 male, 1 female, 2 none (PokemonSpec convention)
    friendship: int
    item: int
    ivs: tuple[int, ...]  # ids.Stat order
    evs: tuple[int, ...]  # ids.Stat order
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]
    pp_ups: tuple[int, int, int, int]
    current_hp: int | None  # party only; None in a box
    status_raw: int  # MON_CONDITION_* word; 0 in a box
    stored_stats: tuple[int, ...] | None  # party only, ids.Stat order, max HP first
```

- **Level of boxed Pokemon** is computed from experience as
  `SpeciesData_GetLevelAt` does (`src/pokemon.c:2378`): the highest level whose
  experience-table entry is ≤ the mon's experience, using the species' `exp_rate`
  (`res/pokemon/<species>/data.json`) and `res/pokemon/.shared/exp_tables.csv`. The party's
  stored level is checked against the same computation.
- `tools/extract_decomp.py` / `tools/build_tables.py` gain two arrays:
  `species_exp_rate` (int8[S]) and `exp_table` (int32[8, 101], the CSV's columns in
  `ExpRate` order). They go in `tables.npz` and the `Tables` NamedTuple; the table version
  bumps as usual.
- **`to_spec(tables) -> PokemonSpec`**: `ability_slot` is the index of the stored ability
  in the species' two abilities (an ability the species cannot have is an error);
  `status`/`status_counter` decode the `MON_CONDITION_*` word as
  `tools/trace_to_state.decode_status` does (sleep turns in bits 0–2; that decoder moves
  into `nuzline.saves` and `trace_to_state` imports it). Party members pass
  `stats=stored_stats`; boxed Pokemon pass `stats=None` (the engine computes them, as the
  game does when a mon leaves the box). Eggs raise.
- **Forms**: `to_spec` raises for any non-zero `form` whose species has alternate forms
  that change stats, types or abilities (Rotom, Wormadam, Giratina, Shaymin, Deoxys; the
  builder takes the exact list from the decomp's form tables). Cosmetic forms (Unown,
  Burmy, Shellos/Gastrodon, Cherrim) pass. `save show` still lists every mon with its form.
- **Stat check (party only)**: stored stats were computed at the current level from some
  earlier EV vector no larger than today's, so each stored stat must lie between
  `calc_stats` (`engine/stats.py`) with all EVs 0 and with the current EVs (same IVs,
  nature, level). Outside that band raises with the slot and stat named; it catches a
  wrong level, IV unpacking, nature or EV order. Equality with the current EVs is not
  required.

### 6. Engine change: current PP and sleep turns reach the battle

`PokemonSpec` (`data/spec.py`) gains two optional fields, defaults unchanged for every
existing caller:

- `pp: tuple[int, int, int, int] | None = None` — `None` keeps today's behavior (base PP
  from `move_pp`, `engine/state.py:251`); otherwise `_fill_slot` uses it. A value above the
  move's maximum with its PP Ups (`base * (5 + ups) // 5`) is an error in `to_spec`.
- `status_counter: int = 0` — written to the state's existing `status_counter` field
  (`engine/state.py:76`) by `_fill_slot`.
- `stats: tuple[int, int, int, int, int, int] | None = None` — `ids.Stat` order, max HP
  first. `None` keeps `calc_stats`; otherwise `_fill_slot` uses it for `max_hp` and
  `stats` (and `current_hp` is checked against it).

The reference engine's state builder (`reference/convert.py` or wherever it reads
`PokemonSpec`) takes the same three fields so the differential tests still compare like with
like. The party TOML gains optional `pp` (four numbers), `sleep_turns` and `stats` (six
numbers) keys so `save export` round-trips.

### 7. CLI (`cli.py`)

- `nuzline save show SAVE [--json] [--allow-corrupt]`: the trainer, which copy loaded and
  the save counter, then the party and each non-empty box, one line per Pokemon (species,
  level, nature, ability, item, HP, status, moves with PP). `--json` prints the whole
  `SaveFile` as JSON (decomp ids plus English names).
- `nuzline save export SAVE -o party.toml [--allow-corrupt]`: writes the party (eggs
  dropped) in the `--party` TOML format, with `hp`, `pp`, `sleep_turns` and `stats`.
- `coverage`, `rank` and `solve` accept `--save SAVE` instead of `--party` (exactly one is
  required). The battle party is the save's party in order with eggs dropped, each dropped
  egg reported on stderr. The run record stores `save_sha256` in place of
  `party_sha256`.
- Errors (`SaveError`, bad record, stat mismatch) exit with code 1 and one readable line.

### 8. Tests

- **Crypto unit tests**: the CRC check value; the LCRNG stream's first outputs from a fixed
  seed; the shuffle table is a permutation for all 32 cases and matches the decomp's
  duplicate cases (24–31 repeat 0–7).
- **Synthetic saves**: a test-only writer (`tests/saves/writer.py`) that encrypts
  `SaveMon`-like records and builds a full 512 KiB image with valid footers. Round trips
  (hypothesis over species, IVs, EVs, moves, PP, personality, so every shuffle case is
  hit), plus the copy-selection cases of `SaveData_LoadCheck`: newer primary, newer
  backup, counter wrap, one copy corrupt, mismatched normal/box counters, both empty;
  and the `.dsv` footer.
- **Engine change**: a state built from a spec with `pp`, `status_counter` and `stats`
  carries them; existing tests unchanged.
- **Stat band and forms**: a synthetic party mon whose stored stats lag its EVs loads; one
  outside the band raises; a Rotom form raises in `to_spec`, an Unown letter does not.
- **CLI**: `save show`, `save export` and `rank --save` on a synthetic save.
- **PKHeX oracle** (`tests/saves/test_pkhex_oracle.py`): skipped unless
  `NUZLINE_ORACLE_DIR` points at a folder (outside the repo) holding:
  - the save (`*.sav` or `*.dsv`, exactly one);
  - PKHeX's **Box Data Report** exported as CSV;
  - the party exported as PKHeX `.pk4` files (decrypted, unshuffled records).

  It checks each party record's decrypted, unshuffled bytes against its `.pk4` byte for
  byte, and species, level, nature, ability, IVs, EVs, moves, item and box/slot of every
  boxed Pokemon against the CSV. Matching on the CSV's English names goes through the
  manifest's name lists. Builder: the CSV's exact column headers come from the user's
  export; the test should map columns by header name and fail loudly on a missing one.

## Left to the builder

- Module boundaries inside `src/nuzline/saves/` beyond the four named files.
- Character decoding for nicknames and the trainer name: use the decomp's
  `tools/msgenc/charmap.txt`, extracted into the manifest by `build_tables`; unmapped
  codes become `?`. Names are display only.
- The text layout of `save show`.
- Whether `SaveFile` exposes boxes as `list[list[SaveMon | None]]` or a flat list with
  locations, as long as box/slot are kept.
- How the reference engine is threaded through the `PokemonSpec` change.

## Calls made by the designer (not discussed with the user)

- Bag and trainer (name, ID, badges) are included: `docs/design.md` §4.1 lists the bag in
  this milestone and prep search needs held items and evolution stones.
- The toxic counter is not carried from the save: the battle resets it on switch-in, so
  only the sleep counter goes into `status_counter`.
- Boxed Pokemon load at full HP with no status (the box stores neither); their PP is
  whatever the record holds.
