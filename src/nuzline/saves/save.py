"""Load a Platinum ``.sav``/``.dsv`` into a :class:`SaveFile` (spec §1-3).

Copy selection is a port of ``SaveData_LoadCheck`` (``src/savedata.c:441``) with its
helpers ``SaveCheckInfo_CompareCounters`` (``:375``), ``SaveCheckInfo_CompareSectors``
(``:390``) and ``SaveBlockFooter_Validate`` (``:327``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from nuzline.saves import layout as L
from nuzline.saves.crypto import crc16_ccitt
from nuzline.saves.pkm import SaveMon, decode_mon, decode_text

__all__ = [
    "DSV_MARKER",
    "SaveError",
    "SaveFile",
    "default_tables_and_names",
    "load_save",
    "read_image",
    "select_copies",
]

DSV_MARKER = b"|<--Snip above here to create a raw sav by excluding this DeSmuME savedata footer:"

_COPY_NAMES = ("primary", "backup")

# enum SaveSectorResult / SaveSectorID (include/constants/savedata/savedata.h)
_SECTOR_RESULT_INVALID, _SECTOR_RESULT_PARTIAL_VALID, _SECTOR_RESULT_VALID = 0, 1, 2
_SECTOR_ID_PRIMARY, _SECTOR_ID_BACKUP, _SECTOR_ID_ERROR = 0, 1, 2

_BLOCKS = {"normal": (0, L.NORMAL_SIZE), "boxes": (L.BOXES_OFFSET, L.BOXES_SIZE)}


class SaveError(ValueError):
    """The file is not a loadable Platinum save."""


@dataclass(frozen=True)
class SaveFile:
    trainer_name: str
    tid: int
    sid: int
    badges: int  # badge bit mask
    party: tuple[SaveMon, ...]
    boxes: tuple[tuple[SaveMon | None, ...], ...]  # [box][slot], 18 x 30
    bag: dict[str, list[tuple[int, int]]]  # pocket -> [(item id, quantity)]
    load_result: str  # "ok" | "corrupt"
    normal_copy: str  # "primary" | "backup"
    boxes_copy: str
    save_counter: int
    sha256: str  # of the file as read


def read_image(data: bytes) -> bytes:
    """The 0x80000-byte flash image: a raw ``.sav``, or a ``.dsv`` cut at its marker."""
    cut = data.find(DSV_MARKER)
    image = data if cut < 0 else data[:cut]
    if len(image) != L.SAVE_SIZE:
        raise SaveError(f"not a Platinum save: {len(image)} bytes")
    return image


class _CheckInfo(NamedTuple):  # SaveCheckInfo
    valid: bool
    global_counter: int
    block_counter: int


def _u16(b: bytes, o: int) -> int:
    return int.from_bytes(b[o : o + 2], "little")


def _u32(b: bytes, o: int) -> int:
    return int.from_bytes(b[o : o + 4], "little")


def _block(image: bytes, copy: int, block: str) -> bytes:
    """The whole block (body + footer) of ``block`` in copy ``copy`` (0/1)."""
    offset, size = _BLOCKS[block]
    start = L.COPY_OFFSETS[copy] + offset
    return image[start : start + size]


def _footer_validate(data: bytes, block: str) -> bool:
    """``SaveBlockFooter_Validate`` (src/savedata.c:327)."""
    size = _BLOCKS[block][1]
    f = size - L.FOOTER_SIZE
    if _u32(data, f + 8) != size:
        return False
    if _u32(data, f + 12) != L.SIGNATURE:
        return False
    if data[f + 16] != L.BLOCK_IDS[block]:
        return False
    # SaveData_CalculateFooterChecksum (:294): CRC over the block minus its footer.
    return _u16(data, f + 18) == crc16_ccitt(data[:f])


def _check_info(image: bytes, copy: int, block: str) -> _CheckInfo:
    """``SaveBlockFooter_CheckInfo``: counters are read whether or not the block is valid."""
    data = _block(image, copy, block)
    f = _BLOCKS[block][1] - L.FOOTER_SIZE
    return _CheckInfo(_footer_validate(data, block), _u32(data, f), _u32(data, f + 4))


def _compare_counters(counter1: int, counter2: int) -> int:
    """``SaveCheckInfo_CompareCounters`` (:375): 0xFFFFFFFF is older than 0."""
    if counter1 == 0xFFFFFFFF and counter2 == 0:
        return -1
    elif counter1 == 0 and counter2 == 0xFFFFFFFF:  # noqa: SIM114 (kept line for line)
        return 1
    elif counter1 > counter2:
        return 1
    elif counter1 < counter2:
        return -1
    return 0


def _compare_sectors(info1: _CheckInfo, info2: _CheckInfo) -> tuple[int, int, int]:
    """``SaveCheckInfo_CompareSectors`` (:390) -> (result, current sector, stale sector).

    The decomp's ``GF_ASSERT``s compile to nothing in retail builds (no
    ``PM_KEEP_ASSERTS``, include/global/assert.h) and are not ported.
    """
    global_diff = _compare_counters(info1.global_counter, info2.global_counter)
    block_diff = _compare_counters(info1.block_counter, info2.block_counter)

    if info1.valid and info2.valid:
        if global_diff > 0:
            current, stale = _SECTOR_ID_PRIMARY, _SECTOR_ID_BACKUP
        elif global_diff < 0:
            current, stale = _SECTOR_ID_BACKUP, _SECTOR_ID_PRIMARY
        elif block_diff > 0:
            current, stale = _SECTOR_ID_PRIMARY, _SECTOR_ID_BACKUP
        elif block_diff < 0:
            current, stale = _SECTOR_ID_BACKUP, _SECTOR_ID_PRIMARY
        else:
            current, stale = _SECTOR_ID_PRIMARY, _SECTOR_ID_BACKUP
        return _SECTOR_RESULT_VALID, current, stale
    elif info1.valid and not info2.valid:
        return _SECTOR_RESULT_PARTIAL_VALID, _SECTOR_ID_PRIMARY, _SECTOR_ID_ERROR
    elif not info1.valid and info2.valid:
        return _SECTOR_RESULT_PARTIAL_VALID, _SECTOR_ID_BACKUP, _SECTOR_ID_ERROR
    else:
        return _SECTOR_RESULT_INVALID, _SECTOR_ID_ERROR, _SECTOR_ID_ERROR


def select_copies(image: bytes) -> tuple[str, int, int]:
    """``SaveData_LoadCheck`` (:441) -> (load result, normal copy, boxes copy).

    The result is ``"ok"``, ``"corrupt"``, ``"error"`` or ``"empty"``; the copy indices
    (0 primary, 1 backup) are those ``SaveData_SetBlockCheckInfo`` records, and are -1
    for ``"error"``/``"empty"``.
    """
    normal_info = [_check_info(image, c, "normal") for c in (0, 1)]
    box_info = [_check_info(image, c, "boxes") for c in (0, 1)]

    normal_result, curr_normal, stale_normal = _compare_sectors(*normal_info)
    box_result, curr_box, stale_box = _compare_sectors(*box_info)

    if normal_result == _SECTOR_RESULT_INVALID and box_result == _SECTOR_RESULT_INVALID:
        return "empty", -1, -1

    if normal_result == _SECTOR_RESULT_INVALID or box_result == _SECTOR_RESULT_INVALID:
        return "error", -1, -1

    if normal_result == _SECTOR_RESULT_VALID and box_result == _SECTOR_RESULT_VALID:
        if normal_info[curr_normal].global_counter == box_info[curr_box].global_counter:
            return "ok", curr_normal, curr_box
        else:
            return "corrupt", stale_normal, curr_box

    if normal_result == _SECTOR_RESULT_PARTIAL_VALID and box_result == _SECTOR_RESULT_VALID:
        if normal_info[curr_normal].global_counter == box_info[curr_box].global_counter:
            return "corrupt", curr_normal, curr_box
        elif normal_info[curr_normal].global_counter == box_info[stale_box].global_counter:
            return "corrupt", curr_normal, stale_box

        return "error", -1, -1

    if normal_result == _SECTOR_RESULT_VALID and box_result == _SECTOR_RESULT_PARTIAL_VALID:
        if normal_info[curr_normal].global_counter == box_info[curr_box].global_counter:
            return "ok", curr_normal, curr_box
        else:
            return "corrupt", stale_normal, curr_box

    if (
        normal_result == _SECTOR_RESULT_PARTIAL_VALID
        and box_result == _SECTOR_RESULT_PARTIAL_VALID
        and curr_normal == curr_box
    ):
        return "ok", curr_normal, curr_box
    else:
        return "corrupt", curr_normal, curr_box


def default_tables_and_names(tables, names, dir=None):
    """numpy tables from ``tables.npz`` and the manifest names. Imported here, not at
    module top, so ``import nuzline.saves`` stays free of jax (``nuzline.data.tables``
    imports it)."""
    import numpy as np

    from nuzline.data.tables import DEFAULT, Tables, _read_manifest, load_names

    dir = DEFAULT if dir is None else dir
    _read_manifest(dir)  # stale tables: one rebuild message, even when both are passed
    if tables is None:
        with np.load(dir / "tables.npz") as z:
            tables = Tables(**{f: z[f] for f in Tables._fields})
    if names is None:
        names = load_names(dir)
    return tables, names


def _party(normal: bytes, tables, charmap) -> tuple[SaveMon, ...]:
    count = int.from_bytes(normal[L.PARTY_COUNT : L.PARTY_COUNT + 4], "little", signed=True)
    if not 1 <= count <= L.PARTY_MAX:
        raise SaveError(f"party count {count} outside 1..{L.PARTY_MAX}")
    party = []
    for slot in range(count):  # records beyond currentCount are ignored
        o = L.PARTY_MONS + slot * L.PARTY_MON_SIZE
        mon = decode_mon(normal[o : o + L.PARTY_MON_SIZE], ("party", 0, slot), tables, charmap)
        if mon is None:
            raise SaveError(f"party slot {slot}: empty within party count {count}")
        party.append(mon)
    return tuple(party)


def _bag(normal: bytes) -> dict[str, list[tuple[int, int]]]:
    bag = {}
    o = L.BAG
    for name, size in L.POCKETS:
        pocket = []
        for j in range(size):
            item = _u16(normal, o + 4 * j)
            if item == 0:
                break
            pocket.append((item, _u16(normal, o + 4 * j + 2)))
        bag[name] = pocket
        o += 4 * size
    return bag


def _boxes(box_block: bytes, tables, charmap) -> tuple[tuple[SaveMon | None, ...], ...]:
    boxes = []
    for b in range(L.BOX_COUNT):
        row = []
        for s in range(L.BOX_SLOTS):
            o = L.BOX_MONS + (b * L.BOX_SLOTS + s) * L.BOX_MON_SIZE
            row.append(
                decode_mon(box_block[o : o + L.BOX_MON_SIZE], ("box", b, s), tables, charmap)
            )
        boxes.append(tuple(row))
    return tuple(boxes)


def load_save(path: Path, *, allow_corrupt: bool = False, tables=None, names=None) -> SaveFile:
    """Parse a Platinum save, picking the blocks the game would load.

    Raises :class:`SaveError` for a wrong size, ``"empty"``/``"error"`` load results,
    ``"corrupt"`` unless ``allow_corrupt``, or a bad party count; a record that fails its
    checks raises ``RecordError`` naming its location.
    """
    data = Path(path).read_bytes()
    image = read_image(data)
    result, normal_copy, boxes_copy = select_copies(image)
    if result in ("empty", "error"):
        raise SaveError(f"save does not load: {result}")
    if result == "corrupt" and not allow_corrupt:
        raise SaveError(
            "save is corrupt: the game would fall back to an older block "
            "(use allow_corrupt to load it)"
        )
    tables, names = default_tables_and_names(tables, names)
    charmap = names.charmap

    normal = _block(image, normal_copy, "normal")
    box_block = _block(image, boxes_copy, "boxes")
    trainer_id = _u32(normal, L.TRAINER_ID)
    return SaveFile(
        trainer_name=decode_text(normal, L.TRAINER_NAME, L.TRAINER_NAME_LEN, charmap),
        tid=trainer_id & 0xFFFF,
        sid=trainer_id >> 16,
        badges=normal[L.BADGES],
        party=_party(normal, tables, charmap),
        boxes=_boxes(box_block, tables, charmap),
        bag=_bag(normal),
        load_result=result,
        normal_copy=_COPY_NAMES[normal_copy],
        boxes_copy=_COPY_NAMES[boxes_copy],
        save_counter=_u32(normal, L.NORMAL_SIZE - L.FOOTER_SIZE),
        sha256=hashlib.sha256(data).hexdigest(),
    )
