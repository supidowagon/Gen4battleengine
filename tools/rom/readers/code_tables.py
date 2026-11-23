"""Tables that live in the ARM9/overlay code rather than in a NARC (Ruling 13)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import MAX_TUTOR_COST, _constants, _map_header_ids, _stem
from tools.rom.readers import ReaderError, symbol_bytes
from tools.rom.readers.personal import FORM_ROW_COUNT, ROW_COUNT, SPECIES_COUNT

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

TM_SYMBOL = "sTMHMMoves"
TM_COUNT = 100  # NUM_TMS (92) + NUM_HMS (8)
EGG_SYMBOL = "sEggMoves"
EGG_OFFSET = 20000  # EGG_MOVES_SPECIES_OFFSET
EGG_TERMINATOR = 0xFFFF  # EGG_MOVES_TERMINATOR
MAX_EGG_MOVES = 16  # include/constants/daycare.h:19
TUTOR_MOVES_SYMBOL = "sTeachableMoves"
TUTOR_SETS_SYMBOL = "sSpeciesLearnsetsByTutor"
TUTOR_ENTRY = struct.Struct(
    "<H4BHI"
)  # TeachableMove (2 pad bytes before the u32), include/tutor_movesets.h:7-14
# Personal rows with no tutor set (SPECIES_NONE, SPECIES_EGG, SPECIES_BAD_EGG; HasTutorSet,
# tools/dataproc/src/speciesproc.c:258).
TUTOR_NO_SET_ROWS = (0, SPECIES_COUNT - 2, SPECIES_COUNT - 1)
MAP_EVOLUTION_SYMBOL = (
    "mapEvolutionMethods$12747"  # the function's static table, src/map_header.c:253
)
NATIONAL_DEX_COUNT = SPECIES_COUNT - 3  # MAX_SPECIES - 2, include/constants/species.h:7


def tm_moves(rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None) -> list[int]:
    """Move id of ITEM_TM01..ITEM_HM08, from ``sTMHMMoves``: 100 u16 in
    ``res/items/item_tm_move_map.h`` (template ``tools/dataproc/data/item_tm_move_map.h.template:6``,
    included at ``src/item.c:17``, read by ``Item_MoveForTMHM``, ``src/item.c:296-304``).
    Each must be a real move: ``0 < m < len(moves)``. ``decomp`` defaults to
    ``external/pokeplatinum``."""
    if decomp is None:
        from tools.rom.inputs import REPO_ROOT

        decomp = REPO_ROOT / "external" / "pokeplatinum"
    n_moves = len(_constants(Path(decomp), "moves", "MOVE_"))
    data = symbol_bytes(rom, symbols, TM_SYMBOL)
    if len(data) != 2 * TM_COUNT:
        raise ReaderError(f"{TM_SYMBOL}: {len(data)} bytes, expected {2 * TM_COUNT}")
    out = list(struct.unpack(f"<{TM_COUNT}H", data))
    for i, m in enumerate(out):
        if not 0 < m < n_moves:
            raise ReaderError(f"{TM_SYMBOL}[{i}]: move {m} is out of range (1..{n_moves - 1})")
    return out


def egg_moves(
    rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None
) -> dict[int, list[int]]:
    """Species id -> egg moves from ``sEggMoves``: u16s, each species' marker
    ``20000 + species`` followed by its moves, the whole table ended by ``0xFFFF``
    (``tools/dataproc/data/species_egg_moves.h.template:7-8``; ``EmitEggMoves``,
    ``speciesproc.c:681-692``). ``LoadSpeciesEggMoves`` (``src/overlay005/daycare.c:445``)
    scans for the marker and reads up to 16 values until one exceeds 20000.

    Shape checks: a whole number of u16s ending in the terminator and containing it nowhere
    else, a marker for a real species (1..495) at most once, every move a real move
    (``0 < m < len(moves)``), and at most 16 moves per species. A species with no moves
    after its marker is not recorded, as the decomp side does."""
    if decomp is None:
        from tools.rom.inputs import REPO_ROOT

        decomp = REPO_ROOT / "external" / "pokeplatinum"
    n_moves = len(_constants(Path(decomp), "moves", "MOVE_"))
    data = symbol_bytes(rom, symbols, EGG_SYMBOL)
    if len(data) % 2 or len(data) < 2:
        raise ReaderError(f"{EGG_SYMBOL}: {len(data)} bytes is not a whole number of u16")
    vals = struct.unpack(f"<{len(data) // 2}H", data)
    if vals[-1] != EGG_TERMINATOR or EGG_TERMINATOR in vals[:-1]:
        raise ReaderError(f"{EGG_SYMBOL}: not terminated by a single 0xffff at the end")
    out: dict[int, list[int]] = {}
    cur: list[int] | None = None
    for i, v in enumerate(vals[:-1]):
        if v > EGG_OFFSET:
            species = v - EGG_OFFSET
            if not 0 < species < SPECIES_COUNT:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: marker for species {species}")
            if species in out:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: a second marker for species {species}")
            cur = out[species] = []
        else:
            if cur is None:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: move {v} before any species marker")
            if not 0 < v < n_moves:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: move {v} is out of range (1..{n_moves - 1})")
            cur.append(v)
            if len(cur) > MAX_EGG_MOVES:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: more than {MAX_EGG_MOVES} egg moves")
    return {k: v for k, v in out.items() if v}


def map_evolution_methods(
    rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None
) -> dict[int, int]:
    """Map header id -> evolution method from ``mapEvolutionMethods``: u16 (header, method)
    pairs scanned two at a time by ``MapHeader_GetMapEvolutionMethod``
    (``src/map_header.c:251-303``; the linker map names the symbol ``mapEvolutionMethods$12747``,
    0x54 bytes, 21 pairs). A header outside the table gives ``EVO_NONE``. ``method`` is the
    ``generated/evolution_methods.txt`` id.

    Shape checks: a whole number of pairs, every header a real map header (below
    ``MAP_HEADER_COUNT``), every method a real evolution method (below the number of methods),
    and no header twice."""
    decomp = _decomp_root(decomp)
    n_headers = len(_map_header_ids(decomp))
    n_methods = len(_constants(decomp, "evolution_methods", "EVO_"))
    data = symbol_bytes(rom, symbols, MAP_EVOLUTION_SYMBOL)
    if not data or len(data) % 4:
        raise ReaderError(
            f"{MAP_EVOLUTION_SYMBOL}: {len(data)} bytes is not a whole number of u16 pairs"
        )
    vals = struct.unpack(f"<{len(data) // 2}H", data)
    out: dict[int, int] = {}
    for i, (header, method) in enumerate(zip(vals[0::2], vals[1::2], strict=True)):
        if header >= n_headers:
            raise ReaderError(f"{MAP_EVOLUTION_SYMBOL}[{i}]: map header {header} is out of range")
        if method >= n_methods:
            raise ReaderError(
                f"{MAP_EVOLUTION_SYMBOL}[{i}]: evolution method {method} is out of range"
            )
        if header in out:
            raise ReaderError(f"{MAP_EVOLUTION_SYMBOL}[{i}]: map header {header} listed twice")
        out[header] = method
    return out


def _decomp_root(decomp: Path | None) -> Path:
    if decomp is None:
        from tools.rom.inputs import REPO_ROOT

        decomp = REPO_ROOT / "external" / "pokeplatinum"
    return Path(decomp)


def tutor_moves(rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None) -> list[dict]:
    """The move tutors' table from ``sTeachableMoves`` (``include/tutor_movesets.h:7-14``,
    ``res/pokemon/tutorable_moves.h``): 12-byte ``TeachableMove`` entries, a u16 move, four u8
    shard costs (red, blue, yellow, green), two pad bytes and a u32 ``TutorLocation``. Returned as
    ``extract_tutor_moves`` writes them, ``location`` being the
    ``generated/tutor_locations.txt`` stem.

    Shape checks: a whole number of entries (at least one), every move real
    (``0 < m < len(moves)``), every cost 0..``MAX_TUTOR_COST``, zero pad bytes and a location
    below the number of ``tutor_locations`` (the count is the decomp's)."""
    decomp = _decomp_root(decomp)
    n_moves = len(_constants(decomp, "moves", "MOVE_"))
    locations = _constants(decomp, "tutor_locations", "TUTOR_LOCATION_")
    data = symbol_bytes(rom, symbols, TUTOR_MOVES_SYMBOL)
    size = TUTOR_ENTRY.size
    if not data or len(data) % size:
        raise ReaderError(
            f"{TUTOR_MOVES_SYMBOL}: {len(data)} bytes is not a whole number of {size}-byte entries"
        )
    out = []
    for i, (move, *costs, pad, loc) in enumerate(TUTOR_ENTRY.iter_unpack(data)):
        if not 0 < move < n_moves:
            raise ReaderError(
                f"{TUTOR_MOVES_SYMBOL}[{i}]: move {move} is out of range (1..{n_moves - 1})"
            )
        if any(c > MAX_TUTOR_COST for c in costs):
            raise ReaderError(f"{TUTOR_MOVES_SYMBOL}[{i}]: cost {costs} is above {MAX_TUTOR_COST}")
        if pad:
            raise ReaderError(f"{TUTOR_MOVES_SYMBOL}[{i}]: pad bytes are {pad:#x}, expected 0")
        if loc >= len(locations):
            raise ReaderError(
                f"{TUTOR_MOVES_SYMBOL}[{i}]: location {loc} is out of range "
                f"(0..{len(locations) - 1})"
            )
        out.append(
            {
                "move": move,
                "costs": dict(zip(("red", "blue", "yellow", "green"), costs, strict=True)),
                "location": _stem(locations[loc], "TUTOR_LOCATION_"),
            }
        )
    return out


def tutor_learnsets(rom: RomFiles, symbols: SymbolMap, n_tutor: int) -> dict[str, list[int]]:
    """Personal row -> sorted tutor indices, from ``sSpeciesLearnsetsByTutor``: one
    ``(n_tutor + 7) // 8``-byte bit mask per moveset (``MovesetMask``,
    ``include/tutor_movesets.h:20-26``), bit ``j`` being ``sTeachableMoves[j]``. Moveset ``m``
    is at index ``m - 1``; species ``s`` uses moveset ``s`` and personal row ``r >= 496`` (a
    data form) uses ``NATIONAL_DEX_COUNT + (r - 495)`` (``include/constants/forms.h:6-19``,
    ``Pokemon_ReadMovesetMaskByte``, ``src/overlay005/scrcmd_move_tutor.c:211-259``). Rows 0,
    494 and 495 have no mask and get ``[]``.

    Shape checks: exactly one mask for each species 1..493 and each form row, and no bit set at
    or above ``n_tutor``."""
    if n_tutor < 1:
        raise ReaderError(f"{TUTOR_SETS_SYMBOL}: {n_tutor} tutor moves")
    width = (n_tutor + 7) // 8
    n_sets = NATIONAL_DEX_COUNT + FORM_ROW_COUNT
    data = symbol_bytes(rom, symbols, TUTOR_SETS_SYMBOL)
    if len(data) != width * n_sets:
        raise ReaderError(
            f"{TUTOR_SETS_SYMBOL}: {len(data)} bytes, expected {width * n_sets} "
            f"({n_sets} masks of {width})"
        )
    out: dict[str, list[int]] = {}
    for row in range(ROW_COUNT):
        if row in TUTOR_NO_SET_ROWS:
            out[str(row)] = []
            continue
        moveset = row if row < SPECIES_COUNT else NATIONAL_DEX_COUNT + (row - (SPECIES_COUNT - 1))
        mask = int.from_bytes(data[(moveset - 1) * width : moveset * width], "little")
        if mask >> n_tutor:
            raise ReaderError(
                f"{TUTOR_SETS_SYMBOL}: moveset {moveset} has a bit past tutor {n_tutor - 1}"
            )
        out[str(row)] = [j for j in range(n_tutor) if mask >> j & 1]
    return out


GYM_E4_CHAMPION_SYMBOL = "TrainerIsGymLeaderE4OrChampion"  # src/battle/battle_main.c:1376-1397
# The compiled switch (Thumb, overlay16): subs r0, #base; cmp r0, #n - 1; bhi FALSE; the
# halfword jump table (adds r0, r0, r0; add r0, pc; ldrh r0, [r0, #6]; lsls/asrs #16;
# add pc, r0; n entries), then TRUE (movs r0, #1; bx lr) and FALSE (movs r0, #0; bx lr).
_SWITCH_HEAD = (0x1800, 0x4478, 0x88C0, 0x0400, 0x1400, 0x4487)
_RETURN_TRUE, _RETURN_FALSE = (0x2001, 0x4770), (0x2000, 0x4770)


def gym_e4_champion_classes(rom: RomFiles, symbols: SymbolMap) -> frozenset[int]:
    """The trainer class ids for which ``TrainerIsGymLeaderE4OrChampion``
    (src/battle/battle_main.c:1376-1397) returns TRUE, read from the ROM's compiled switch:
    ``BattleSys_New`` raises the party's friendship against them (:1360-1372). The function is
    read at its linker-map address plus Kaizo's overlay16 shift (``tools.rom.offsets._shift``,
    0 below the patched routines); any code other than the switch shape above raises."""
    from nuzline.data.games import KAIZO, VANILLA
    from tools.rom.offsets import _shift

    sym = symbols.find(GYM_E4_CHAMPION_SYMBOL)
    if sym.module != "overlay16":
        raise ReaderError(f"{GYM_E4_CHAMPION_SYMBOL}: module {sym.module}, not overlay16")
    game = VANILLA if rom.game_code == "CPUE" else KAIZO
    address = sym.address + _shift(game, sym.address)
    ram, data = rom.overlay_info(16)[0], rom.overlay(16)
    code = data[address - ram : address - ram + sym.size]
    words = struct.unpack(f"<{len(code) // 2}H", code)

    def fail(what: str):
        return ReaderError(f"{GYM_E4_CHAMPION_SYMBOL} at {address:#x}: {what}")

    if len(code) != sym.size or len(words) < 9 or words[0] >> 8 != 0x38 or words[1] >> 8 != 0x28:
        raise fail("not subs r0, #base; cmp r0, #n - 1")
    base, n = words[0] & 0xFF, (words[1] & 0xFF) + 1
    if words[2] >> 8 != 0xD8 or words[3:9] != _SWITCH_HEAD:
        raise fail("not bhi and the halfword jump table dispatch")
    table, tail = words[9 : 9 + n], words[9 + n :]
    if len(table) != n or tail != _RETURN_TRUE + _RETURN_FALSE:
        raise fail("the table is not followed by return TRUE, return FALSE")
    after_table = address + 2 * (9 + n)
    false_at = after_table + 4
    bhi = words[2] & 0xFF
    if address + 4 + 4 + 2 * (bhi - 256 if bhi & 0x80 else bhi) != false_at:
        raise fail("bhi does not reach return FALSE")
    pc = address + 2 * 8 + 4  # add pc, r0 reads its own address + 4
    targets = [pc + (t - 0x10000 if t & 0x8000 else t) for t in table]
    if not set(targets) <= {after_table, false_at}:
        raise fail("a case reaches neither return")
    return frozenset(base + i for i, t in enumerate(targets) if t == after_table)
