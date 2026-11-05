"""Tables that live in the ARM9/overlay code rather than in a NARC (Ruling 13)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import MAX_TUTOR_COST, _constants, _stem
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
