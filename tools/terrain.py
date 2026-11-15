"""Battle terrain of a trainer battle, per map header and per trainer (K3b2 Task 9;
docs/kaizo/k3b2-rules.md §9).

Shared by ``tools.extract_decomp`` (vanilla, from the decomp's sources) and
``tools.rom.readers.maps`` (either game, from the ROM): both hand it map headers,
matrices and land data in the decomp's member layout and get the same rows back, which
the vanilla round trip (``tests/tools/rom/test_extract_rom.py``) compares. Every rule is
read from the decomp (pokeplatinum, ``external/pokeplatinum``):

* A trainer battle starts through ``Encounter_NewVsTrainer`` (src/encounter.c:724-758) or
  ``Encounter_NewVsFirstBattle`` (:653-667). Both call ``FieldBattleDTO_Init`` with the
  field system (:744, :657) and set no terrain afterwards. ``FieldBattleDTO_Init``
  (src/field_battle_data_transfer.c:279-283) runs ``FieldBattleDTO_InitFromGameState``,
  whose field-system branch calls ``SetBackgroundAndTerrain`` (:244-246).
* ``SetBackgroundAndTerrain`` (:496-505): the background is
  ``MapHeader_GetBattleBG(location->mapHeaderID)``, ``BACKGROUND_WATER`` while the player
  surfs, and the terrain is ``CalcTerrain(fieldSystem, background)``.
* ``CalcTerrain`` (:467-505, with its caller above) reads the behaviour of the tile under the player
  (``TerrainCollisionManager_GetTileBehavior`` at ``location->x, z``,
  src/terrain_collision_manager.c:229-240: the land data's terrain attribute u16 masked by
  ``TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK``; both collision managers read the tile from the
  land data member of the player's matrix cell, at ``TERRAIN_ATTRIBUTES_OFFSET``, :160-199
  and src/terrain_attributes.c:39-70). Its tile tests run in source order (``tile_tests``);
  without a match it returns ``sTerrainForBackground[background]`` (:441-465), and a
  background past that table asserts (:491-493).
* The matrix is loaded once per map change, from the header the player arrives in
  (``MapMatrix_Load(location->mapHeaderID, ...)``, src/field_map_change.c:380;
  src/map_matrix.c:84-92), and kept while the player walks. ``MapMatrixData_Load`` (:13-58)
  gives the member layout (``parse_matrix``); a matrix with no header section gives every
  cell the header that loaded it (:47-49).
* ``location->mapHeaderID`` then follows the label of the cell the player stands in:
  ``FieldMap_ChangeZone`` (src/overlay005/fieldmap.c:397-417) sets it from
  ``MapMatrix_GetMapHeaderIDAtCoords`` whenever the player's position changes
  (``FieldMap_Main``, :250-255), and never reloads the matrix. So a header ``h`` can be
  current on any cell labelled ``h`` of any matrix some arrival header loads, not only on
  ``h``'s own matrix. The Distortion World skips the zone change (:399-401); its matrices
  have no header section, so every cell is the loading header anyway.
* Before the first step after a warp, ``location`` is the warp's: the destination header
  and the position of its warp event (src/field_map_change.c:224-230), whatever that cell's
  label. The on-frame scripts and the trainer sight check run on every field frame, the
  first included (``FieldInput_Process``, src/overlay005/field_control.c:176-195), so a
  battle can start there, on that tile, under the destination header's background.
* ``FieldMapChange`` rewrites some land data ids by story flag (src/field_map_change.c:
  382-388; ``MapMatrix_RevealSpringPath`` and ``MapMatrix_RevealSeabreakPath``,
  src/map_matrix.c:194-230): both values are possible. Their guard compares the loaded
  matrix's ``u8 matrixID`` (include/map_matrix.h:18, set at src/map_matrix.c:89), so they
  apply to every matrix whose id is 0 modulo 256, where the written cell exists.
* While surfing, the player stands on a tile ``MapObject_IsOnWater`` accepts
  (src/map_object_move.c:740-751); the terrain is then computed with ``BACKGROUND_WATER``.

So a battle in header ``h`` starts on a tile of a cell labelled ``h`` (in any matrix some
header loads; every cell of ``h``'s own matrix when it has no header section) or on the
tile of one of ``h``'s warp events, and its terrain is ``CalcTerrain`` of that tile's behaviour with ``h``'s
``battleBG`` (or ``BACKGROUND_WATER`` on a tile a surfing player can stand on). A trainer's
terrain is the union over its map headers (``tools.rom.readers.maps.trainer_maps``) of the
terrains of the tiles its battle can start on there: the sight-line and talk tiles of its
object, or the rectangle of the coord event whose script starts it, when the decomp fixes
them (``narrowed_terrains``, rules §9.3 step 2), else the header's whole set. One value is
the terrain, several are "mixed" and no header (no map places the trainer) is "unknown";
both of the last two store ``TERRAIN_UNKNOWN``.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from tools.extract_decomp import ExtractError, _constants

TERRAIN_UNKNOWN = -1  # a mixed terrain, or a trainer no map places
# CalcTerrain's tile tests, the background table and the trainer battle path (module
# docstring). A game that changes any of them changes the terrain this module derives:
# tools.rom.readers.maps.check_terrain_code refuses a ROM diff that touches one.
TERRAIN_CODE = (
    "Encounter_NewVsTrainer",
    "Encounter_NewVsFirstBattle",
    "FieldBattleDTO_Init",
    "FieldBattleDTO_InitFromGameState",
    "SetBackgroundAndTerrain",
    "CalcTerrain",
    "sTerrainForBackground",
    "sTileBehaviorFlags",
    "TerrainCollisionManager_GetTileBehavior",
    "TerrainCollisionManager_Init",
    "GetTileAttributes",
    "GetSimpleTileAttributes",
    "TerrainAttributes_Get",
    "LandDataManager_GetLoadedMapTerrainAttributes",
    "MapMatrix_Load",
    "MapMatrixData_Load",
    "MapMatrix_GetMapHeaderIDAtCoords",
    "MapMatrix_RevealSpringPath",
    "MapMatrix_RevealSeabreakPath",
    "FieldMap_ChangeZone",
    "MapObject_IsOnWater",
)
# The battle's terrain readers and their tables (K3b2 Task 10): the engine holds the tables as
# nuzline.engine.ids.TERRAIN_MOVES / TERRAIN_TYPES (the decomp's include/data/terrain/), so a
# game that changes any of them changes what Nature Power and Camouflage do:
# check_terrain_code refuses that too.
BATTLE_TERRAIN_CODE = (
    "BattleSystem_GetTerrain",
    "BtlCmd_TryCamouflage",
    "BtlCmd_GetTerrainMove",
    "sTerrainMove",
    "sTerrainCamouflageType",
)

FBDT_C = Path("src") / "field_battle_data_transfer.c"
TILE_C = Path("src") / "map_tile_behavior.c"
MATRIX_C = Path("src") / "map_matrix.c"
OBJECT_MOVE_C = Path("src") / "map_object_move.c"
BEHAVIOURS_H = Path("include") / "constants" / "field" / "map_tile_behaviors.h"
MAP_H = Path("include") / "constants" / "field" / "map.h"


class TerrainError(ExtractError):
    """The decomp or the data cannot be read as the terrain rules need."""


class Matrix(NamedTuple):
    """A ``map_matrix.narc`` member as ``MapMatrixData_Load`` reads it."""

    width: int
    height: int
    headers: tuple[int, ...] | None  # None: no header section (every cell is the loader)
    lands: tuple[int, ...]


class TerrainRules(NamedTuple):
    """What ``CalcTerrain`` needs, read from the decomp (and the ROM's tables).

    ``tests``: ``(behaviours, terrain)`` in ``CalcTerrain``'s order; ``by_background``:
    ``sTerrainForBackground``; ``water``: ``BACKGROUND_WATER``; ``on_water``: the
    behaviours a surfing player can stand on; ``patches``: ``(matrix, x, z, land data
    id)`` story-flag rewrites; ``attributes``: (offset, count) of the land data's
    terrain attributes; ``mask``: the behaviour mask; ``invalid_land``:
    ``INVALID_LAND_DATA_ID``; ``tiles_x``: ``MAP_TILES_COUNT_X`` (tiles per cell row);
    ``collision``: ``TERRAIN_ATTRIBUTES_COLLISION_MASK`` (the attribute's collision bit,
    src/terrain_collision_manager.c:212-227); ``table``: the behaviours
    ``TileBehavior_IsTable`` accepts (a talk reaches across one, src/unk_0203C954.c:62-84)."""

    names: tuple[str, ...]
    backgrounds: tuple[str, ...]
    tests: tuple[tuple[frozenset[int], int], ...]
    by_background: tuple[int, ...]
    water: int
    on_water: frozenset[int]
    patches: tuple[tuple[int, int, int, int], ...]
    attributes: tuple[int, int]
    mask: int
    invalid_land: int
    tiles_x: int
    collision: int
    table: frozenset[int]


def battle_constants(decomp: Path, name: str, prefix: str) -> list[str]:
    """``generated/<name>.txt`` ids (``battle_terrains``: ``TERRAIN_*``,
    ``battle_backgrounds``: ``BACKGROUND_*``), the trailing ``<prefix>MAX`` included:
    ``CalcTerrain`` returns ``TERRAIN_MAX`` for a background past its table."""
    names = _constants(Path(decomp), name, prefix)
    if not names or names[-1] != f"{prefix}MAX":
        raise TerrainError(f"generated/{name}.txt does not end with {prefix}MAX")
    return names


def _read(decomp: Path, rel: Path) -> str:
    return (Path(decomp) / rel).read_text(encoding="utf-8")


def _function(src: str, path: Path, signature: str) -> str:
    """The body of the C function whose definition line matches ``signature``."""
    m = re.search(signature + r"[^;{]*\)\s*\{(.*?)\n\}", src, re.DOTALL)
    if not m:
        raise TerrainError(f"{path}: no definition matching {signature!r}")
    return m.group(1)


def _defines(text: str, path: Path, names: Iterable[str]) -> dict[str, int]:
    out = {}
    for name in names:
        m = re.search(rf"^#define {name}\s+(0x[0-9A-Fa-f]+|\d+)\s*$", text, re.MULTILINE)
        if not m:
            raise TerrainError(f"{path}: no numeric #define {name}")
        out[name] = int(m.group(1), 0)
    return out


def tile_behaviours(decomp: Path) -> dict[str, int]:
    """``enum TileBehavior`` (map_tile_behaviors.h): name -> value. Only the first member
    may carry an explicit value (0); any other initialiser raises."""
    text = _read(decomp, BEHAVIOURS_H)
    m = re.search(r"enum TileBehavior \{(.*?)\};", text, re.DOTALL)
    if not m:
        raise TerrainError(f"{BEHAVIOURS_H}: no enum TileBehavior")
    out: dict[str, int] = {}
    for raw in m.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        e = re.fullmatch(r"(TILE_BEHAVIOR_\w+)(?:\s*=\s*(\d+))?,", line)
        if not e or (e.group(2) is not None and (out or e.group(2) != "0")):
            raise TerrainError(f"{BEHAVIOURS_H}: unparsed enum line {raw!r}")
        out[e.group(1)] = len(out)
    return out


def _flag_values(decomp: Path) -> dict[str, int]:
    """``TILE_BEHAVIOR_FLAG_*`` (map_tile_behaviors.h): ``N``, ``(1 << n)`` or an ``|`` of
    other flags."""
    text = _read(decomp, BEHAVIOURS_H)
    out: dict[str, int] = {}
    for name, expr in re.findall(
        r"^#define (TILE_BEHAVIOR_FLAG_\w+)\s+(.+?)\s*$", text, re.MULTILINE
    ):
        value = 0
        for term in re.sub(r"^\((.*)\)$", r"\1", expr).split("|"):
            term = term.strip()
            if term in out:
                value |= out[term]
            elif s := re.fullmatch(r"\(?1 << (\d+)\)?", term):
                value |= 1 << int(s.group(1))
            elif term.isdecimal():
                value |= int(term)
            else:
                raise TerrainError(f"{BEHAVIOURS_H}: unparsed flag {name} = {expr!r}")
        out[name] = value
    return out


def decomp_behaviour_flags(decomp: Path) -> list[int]:
    """``sTileBehaviorFlags`` (src/map_tile_behavior.c:6-) by behaviour value."""
    behaviours, flags = tile_behaviours(decomp), _flag_values(decomp)
    src = _read(decomp, TILE_C)
    m = re.search(r"sTileBehaviorFlags\[\] = \{(.*?)\};", src, re.DOTALL)
    if not m:
        raise TerrainError(f"{TILE_C}: no sTileBehaviorFlags")
    table: dict[int, int] = {}
    for b, f in re.findall(r"\[(TILE_BEHAVIOR_\w+)\]\s*=\s*(TILE_BEHAVIOR_FLAG_\w+),", m.group(1)):
        if b not in behaviours or f not in flags:
            raise TerrainError(f"{TILE_C}: sTileBehaviorFlags entry {b} = {f}")
        table[behaviours[b]] = flags[f]
    if sorted(table) != list(range(len(table))):
        raise TerrainError(f"{TILE_C}: sTileBehaviorFlags is not indexed 0..{len(table) - 1}")
    return [table[i] for i in range(len(table))]


def _predicate(decomp: Path, name: str, flags_table: Sequence[int]) -> frozenset[int]:
    """The behaviours for which ``TileBehavior_Is*`` ``name`` (map_tile_behavior.c) is
    TRUE: an ``||`` of ``behavior == TILE_BEHAVIOR_*`` or ``(sTileBehaviorFlags[behavior]
    & FLAG) != 0`` over ``flags_table``."""
    behaviours, flags = tile_behaviours(decomp), _flag_values(decomp)
    body = _function(_read(decomp, TILE_C), TILE_C, rf"BOOL {name}\(").strip()
    body = re.sub(r"\s+", " ", body)
    if m := re.fullmatch(
        r"return \(sTileBehaviorFlags\[behavior\] & (TILE_BEHAVIOR_FLAG_\w+)\) != 0;", body
    ):
        bit = flags[m.group(1)]
        return frozenset(b for b, f in enumerate(flags_table) if f & bit)
    if re.fullmatch(
        r"return behavior == TILE_BEHAVIOR_\w+( \|\| behavior == TILE_BEHAVIOR_\w+)*;", body
    ):
        names = re.findall(r"TILE_BEHAVIOR_\w+", body)
        missing = [n for n in names if n not in behaviours]
        if missing:
            raise TerrainError(f"{TILE_C}: {name} tests unknown behaviours {missing}")
        return frozenset(behaviours[n] for n in names)
    raise TerrainError(f"{TILE_C}: {name} has an unparsed body {body!r}")


def tile_tests(decomp: Path, flags_table: Sequence[int]) -> tuple[tuple[frozenset[int], int], ...]:
    """``CalcTerrain``'s tile tests in source order: each ``if`` whose condition is an
    ``||`` of ``TileBehavior_Is*(behavior)`` returns its ``TERRAIN_*``; then the
    ``sTerrainForBackground`` fallback and ``return TERRAIN_MAX``. Any other shape raises."""
    terrains = battle_constants(decomp, "battle_terrains", "TERRAIN_")
    body = _function(_read(decomp, FBDT_C), FBDT_C, r"static int CalcTerrain\(")
    tests = []
    for cond, terrain in re.findall(r"if \(([^{]*?)\)\s*\{\s*return (TERRAIN_\w+);", body):
        preds = re.findall(r"(TileBehavior_Is\w+)\(behavior\)", cond)
        if " || ".join(f"{p}(behavior)" for p in preds) != re.sub(r"\s+", " ", cond.strip()):
            raise TerrainError(f"{FBDT_C}: CalcTerrain condition {cond!r} not parsed")
        if terrain not in terrains:
            raise TerrainError(f"{FBDT_C}: CalcTerrain returns unknown {terrain}")
        behaviours = frozenset().union(*(_predicate(decomp, p, flags_table) for p in preds))
        tests.append((behaviours, terrains.index(terrain)))
    fallback = re.search(
        r"if \(background < NELEMS\(sTerrainForBackground\)\)\s*\{\s*"
        r"return sTerrainForBackground\[background\];\s*\}\s*"
        r"GF_ASSERT\(FALSE\);\s*return TERRAIN_MAX;\s*$",
        body,
    )
    if not tests or not fallback or body.count("return ") != len(tests) + 2:
        raise TerrainError(f"{FBDT_C}: CalcTerrain is not tile tests then the background table")
    return tuple(tests)


def decomp_terrain_by_background(decomp: Path) -> list[int]:
    """``sTerrainForBackground`` (src/field_battle_data_transfer.c:441-465) by background."""
    terrains = battle_constants(decomp, "battle_terrains", "TERRAIN_")
    backgrounds = battle_constants(decomp, "battle_backgrounds", "BACKGROUND_")
    src = _read(decomp, FBDT_C)
    m = re.search(r"sTerrainForBackground\[\] = \{(.*?)\};", src, re.DOTALL)
    if not m:
        raise TerrainError(f"{FBDT_C}: no sTerrainForBackground")
    table = {}
    for bg, t in re.findall(r"\[(BACKGROUND_\w+)\]\s*=\s*(TERRAIN_\w+),", m.group(1)):
        if bg not in backgrounds or t not in terrains:
            raise TerrainError(f"{FBDT_C}: sTerrainForBackground entry {bg} = {t}")
        table[backgrounds.index(bg)] = terrains.index(t)
    if sorted(table) != list(range(len(table))):
        raise TerrainError(f"{FBDT_C}: sTerrainForBackground is not indexed 0..{len(table) - 1}")
    return [table[i] for i in range(len(table))]


def _on_water(decomp: Path, flags_table: Sequence[int]) -> frozenset[int]:
    """Behaviours ``MapObject_IsOnWater`` (map_object_move.c:740-751) can accept: the union
    of the ``TileBehavior_Is*`` it calls."""
    body = _function(_read(decomp, OBJECT_MOVE_C), OBJECT_MOVE_C, r"int MapObject_IsOnWater\(")
    preds = re.findall(r"(TileBehavior_Is\w+)\(tileBehavior\)", body)
    if not preds:
        raise TerrainError(f"{OBJECT_MOVE_C}: MapObject_IsOnWater calls no tile test")
    return frozenset().union(*(_predicate(decomp, p, flags_table) for p in preds))


def matrix_patches(decomp: Path) -> tuple[tuple[int, int, int, int], ...]:
    """``(matrix, x, z, land data id)`` of ``MapMatrix_RevealSpringPath`` and
    ``MapMatrix_RevealSeabreakPath`` (src/map_matrix.c:194-230): each returns unless the
    matrix id is the one it tests, then writes ``landDataIDs[x + width * z] = id``."""
    src = _read(decomp, MATRIX_C)
    out = []
    for fn in ("MapMatrix_RevealSpringPath", "MapMatrix_RevealSeabreakPath"):
        body = _function(src, MATRIX_C, rf"void {fn}\(")
        guard = re.search(r"if \(mapMatrix->matrixID != (\d+)\) \{\s*return;\s*\}", body)
        writes = re.findall(r"mapMatrixLandDataIDs\[([^\]]+)\] = (\d+);", body)
        if not guard or not writes:
            raise TerrainError(f"{MATRIX_C}: {fn} is not a matrix guard then land id writes")
        for index, land in writes:
            e = re.sub(r"\s+", "", index)
            m = re.fullmatch(r"(\d+)\+mapMatrixWidth\*(\d+)", e) or re.fullmatch(
                r"mapMatrixWidth\*(\d+)\+(\d+)", e
            )
            if not m:
                raise TerrainError(f"{MATRIX_C}: {fn} index {index!r} not parsed")
            x, z = (m.group(1), m.group(2)) if e[0].isdigit() else (m.group(2), m.group(1))
            out.append((int(guard.group(1)), int(x), int(z), int(land)))
    return tuple(out)


def terrain_rules(
    decomp: Path,
    flags_table: Sequence[int] | None = None,
    by_background: Sequence[int] | None = None,
) -> TerrainRules:
    """The rules, read from ``decomp``. ``flags_table`` (``sTileBehaviorFlags``) and
    ``by_background`` (``sTerrainForBackground``) default to the decomp's own; the ROM
    reader passes the ROM's."""
    decomp = Path(decomp)
    flags_table = decomp_behaviour_flags(decomp) if flags_table is None else list(flags_table)
    if by_background is None:
        by_background = decomp_terrain_by_background(decomp)
    backgrounds = battle_constants(decomp, "battle_backgrounds", "BACKGROUND_")
    terrains = battle_constants(decomp, "battle_terrains", "TERRAIN_")
    if any(not 0 <= t < len(terrains) for t in by_background):
        raise TerrainError(f"sTerrainForBackground holds a terrain outside 0..{len(terrains) - 1}")
    defs = _defines(
        _read(decomp, MAP_H),
        MAP_H,
        (
            "TERRAIN_ATTRIBUTES_OFFSET",
            "TERRAIN_ATTRIBUTES_SIZE",
            "TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK",
            "INVALID_LAND_DATA_ID",
            "MAP_TILES_COUNT_X",
            "MAP_TILES_COUNT_Z",
            "TERRAIN_ATTRIBUTES_COLLISION_MASK",
        ),
    )
    count = defs["TERRAIN_ATTRIBUTES_SIZE"] // 2  # TERRAIN_ATTRIBUTES_COUNT, sizeof(u16)
    if count != defs["MAP_TILES_COUNT_X"] * defs["MAP_TILES_COUNT_Z"]:
        raise TerrainError(f"{MAP_H}: TERRAIN_ATTRIBUTES_SIZE is not one u16 per tile")
    return TerrainRules(
        names=tuple(terrains),
        backgrounds=tuple(backgrounds),
        tests=tile_tests(decomp, flags_table),
        by_background=tuple(by_background),
        water=backgrounds.index("BACKGROUND_WATER"),
        on_water=_on_water(decomp, flags_table),
        patches=matrix_patches(decomp),
        attributes=(defs["TERRAIN_ATTRIBUTES_OFFSET"], count),
        mask=defs["TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK"],
        invalid_land=defs["INVALID_LAND_DATA_ID"],
        tiles_x=defs["MAP_TILES_COUNT_X"],
        collision=defs["TERRAIN_ATTRIBUTES_COLLISION_MASK"],
        table=_predicate(decomp, "TileBehavior_IsTable", flags_table),
    )


def parse_matrix(member: bytes, index: int) -> Matrix:
    """A ``map_matrix.narc`` member (``MapMatrixData_Load``, src/map_matrix.c:13-58): u8
    width, height, has-headers, has-altitudes, name length; the name; the u16 header ids
    (when present); the u8 altitudes (when present); the u16 land data ids. Trailing bytes
    raise."""
    if len(member) < 5:
        raise TerrainError(f"map matrix {index}: {len(member)} bytes")
    width, height, has_headers, has_altitudes, name_len = member[:5]
    cells = width * height
    pos = 5 + name_len
    headers = None
    if has_headers:
        headers = struct.unpack_from(f"<{cells}H", member, pos)
        pos += 2 * cells
    if has_altitudes:
        pos += cells
    lands = struct.unpack_from(f"<{cells}H", member, pos)
    pos += 2 * cells
    if pos != len(member):
        raise TerrainError(f"map matrix {index}: {len(member) - pos} trailing bytes")
    return Matrix(width, height, headers, lands)


def land_behaviours(rules: TerrainRules, member: bytes, index: int) -> frozenset[int]:
    """The distinct tile behaviours of a ``land_data.narc`` member: its terrain attribute
    u16s (``TERRAIN_ATTRIBUTES_OFFSET``, one per tile) masked by the behaviour mask. The
    member's own header (``LandDataHeader_Load``, src/overlay005/land_data.c:457-478) must
    give the same attribute size the field code reads at that offset."""
    offset, count = rules.attributes
    if len(member) < offset + 2 * count:
        raise TerrainError(f"land data {index}: {len(member)} bytes, no terrain attributes")
    if struct.unpack_from("<I", member, 0)[0] != 2 * count:
        raise TerrainError(f"land data {index}: terrain attribute size is not {2 * count}")
    return frozenset(a & rules.mask for a in struct.unpack_from(f"<{count}H", member, offset))


def land_attributes(rules: TerrainRules, member: bytes, index: int) -> tuple[int, ...]:
    """Each tile's terrain attribute u16 of a ``land_data.narc`` member (behaviour and
    collision bit), by tile index ``z * MAP_TILES_COUNT_X + x`` (``GetTileAttributes``,
    src/terrain_collision_manager.c:176-178), after ``land_behaviours``' checks."""
    offset, count = rules.attributes
    land_behaviours(rules, member, index)
    return struct.unpack_from(f"<{count}H", member, offset)


def land_tiles(rules: TerrainRules, member: bytes, index: int) -> tuple[int, ...]:
    """Each tile's behaviour of a ``land_data.narc`` member, by tile index (``land_attributes``
    masked by the behaviour mask)."""
    return tuple(a & rules.mask for a in land_attributes(rules, member, index))


def calc_terrain(rules: TerrainRules, behaviour: int, background: int) -> int:
    """``CalcTerrain`` of a tile behaviour and a battle background. A background past
    ``sTerrainForBackground`` gives ``TERRAIN_MAX`` (src/field_battle_data_transfer.c:
    491-493): the retail build keeps ``GF_ASSERT`` (meson.build:72, ``PM_KEEP_ASSERTS``),
    but ``ErrorHandling_AssertFail`` (src/error_handling.c:11-16) only resets during a link,
    so the function goes on to ``return TERRAIN_MAX``."""
    for behaviours, terrain in rules.tests:
        if behaviour in behaviours:
            return terrain
    if 0 <= background < len(rules.by_background):
        return rules.by_background[background]
    return rules.names.index("TERRAIN_MAX")


def _land_ids(m: Matrix, matrix_id: int, rules: TerrainRules, cells: Iterable[int]) -> set[int]:
    """The land data ids ``cells`` of ``m`` can hold: the matrix's own and the story-flag
    rewrites whose guard (the ``u8 matrixID``) matches ``matrix_id``."""
    cells = set(cells)
    ids = {m.lands[i] for i in cells}
    for guard, x, z, land in rules.patches:
        if matrix_id & 0xFF == guard and x + m.width * z in cells:
            ids.add(land)
    return ids


def header_terrains(
    rules: TerrainRules,
    headers: Sequence[tuple[int, int]],
    matrices: Sequence[Matrix],
    lands: Sequence[bytes],
    wanted: Iterable[int],
    arrivals: Mapping[int, Iterable[tuple[int, int]]] | None = None,
) -> dict[int, frozenset[int]]:
    """Map header id -> the terrains a battle in it can start with, for the ``wanted``
    headers. ``headers[h]`` is ``(mapMatrixID, battleBG)``; ``arrivals[h]`` are the ``(x,
    z)`` tiles of ``h``'s warp events, where a warp puts the player under header ``h``.

    The tiles are those of every cell labelled ``h`` in every matrix some header loads (the
    ones with a header section) and of every cell of ``h``'s own matrix when it has none,
    under every land data id the cell can hold, plus each arrival tile (in ``h``'s own
    matrix, which the warp loads) that lies in a cell labelled otherwise. A header with no
    cell anywhere gives ``{TERRAIN_UNKNOWN}``: it is current only right after an arrival,
    and the arrivals by Fly or by a script warp are not read. An arrival outside ``h``'s own
    matrix (Kaizo headers 610 and 613: matrix 3 is 1x1, their warp events are at matrix-0
    coordinates) adds ``TERRAIN_UNKNOWN``: what such a warp would stand on is not read."""
    arrivals = arrivals or {}
    side = rules.tiles_x
    loadable = sorted({matrix for matrix, _ in headers})
    for matrix in loadable:
        if not 0 <= matrix < len(matrices):
            raise TerrainError(f"matrix {matrix} is not a matrix member")
    cache: dict[int, tuple[int, ...]] = {}

    def tiles(land: int) -> tuple[int, ...]:
        if not 0 <= land < len(lands):
            raise TerrainError(f"land data {land} is not a member")
        if land not in cache:
            cache[land] = land_tiles(rules, lands[land], land)
        return cache[land]

    out = {}
    for h in sorted(set(wanted)):
        if not 0 <= h < len(headers):
            raise TerrainError(f"map header {h} is not a header id")
        own, background = headers[h]
        ids: set[int] = set()
        for matrix in loadable:
            m = matrices[matrix]
            if m.headers is not None:
                labelled = [i for i, label in enumerate(m.headers) if label == h]
            elif matrix == own:
                labelled = list(range(m.width * m.height))
            else:
                continue
            ids |= _land_ids(m, matrix, rules, labelled)
        if not ids:
            out[h] = frozenset({TERRAIN_UNKNOWN})
            continue
        behaviours = {b for land in ids if land != rules.invalid_land for b in tiles(land)}
        m = matrices[own]
        unread = False
        for x, z in arrivals.get(h, ()):
            if not (0 <= x // side < m.width and 0 <= z // side < m.height):
                unread = True  # a warp here would stand outside the loaded matrix
                continue
            cell = x // side + m.width * (z // side)
            if m.headers is None or m.headers[cell] == h:
                continue  # its cell is counted already
            for land in _land_ids(m, own, rules, [cell]) - {rules.invalid_land}:
                behaviours.add(tiles(land)[(z % side) * side + x % side])
        terrains = {calc_terrain(rules, b, background) for b in behaviours}
        terrains |= {calc_terrain(rules, b, rules.water) for b in behaviours & rules.on_water}
        if unread:
            terrains.add(TERRAIN_UNKNOWN)
        out[h] = frozenset(terrains)
    return out


def trainer_terrains(
    n_trainers: int,
    where: Mapping[int, Iterable[int]],
    by_header: Mapping[int, frozenset[int]],
    narrowed: Mapping[tuple[int, int], frozenset[int]] | None = None,
) -> list[tuple[int, list[int]]]:
    """Per trainer id: ``(terrain, possible terrains)``. The possible terrains are the union
    over the trainer's map headers ``where[id]`` of ``narrowed[(id, header)]`` when given
    (``narrowed_terrains``: the tiles its battle can start on there), else ``by_header``;
    one value is the terrain, none or several give ``TERRAIN_UNKNOWN``. ``TERRAIN_UNKNOWN``
    in the list means a header whose cells cannot be read."""
    narrowed = narrowed or {}
    out = []
    for tid in range(n_trainers):
        terrains = sorted(
            frozenset().union(*(narrowed.get((tid, h), by_header[h]) for h in where.get(tid, ())))
        )
        known = len(terrains) == 1 and terrains[0] != TERRAIN_UNKNOWN
        out.append((terrains[0] if known else TERRAIN_UNKNOWN, terrains))
    return out


def terrain_fields(terrain: int, terrains: list[int]) -> dict:
    """The ``trainers.json`` fields: ``terrain`` (a ``TERRAIN_*`` id, ``TERRAIN_UNKNOWN``
    for mixed or unknown) and ``terrains`` (every terrain the trainer's maps can give,
    sorted; empty when no map places it)."""
    return {"terrain": terrain, "terrains": terrains}


# --- Narrowing: the tiles a trainer's battle can start on (rules §9.3 step 2) ----------
#
# The whole-header set above is what any battle in a header can start with. For most
# trainers the decomp fixes far fewer tiles, and ``narrowed_terrains`` reads those (both
# extractors hand it the same events and script graphs):
#
# * An object trainer (an object event whose script is a trainer's battle script, 3000 +
#   id - 1 or 5000 + id - 1, ``Script_GetTrainerID``) starts its battle when it spots the
#   player or when the player talks to it. Spotting (``GetTrainerDistToPlayer``,
#   src/trainer_encounter.c:182-226): a ``TRAINER_TYPE_NORMAL`` trainer (and the FACE_* /
#   SPIN_* types ``GetTrainerType`` maps to it, :165-180) looks along its current facing, a
#   ``TRAINER_TYPE_VIEW_ALL_DIRECTIONS`` one along all four, any other type not at all; the
#   player must be 1..``data[0]`` tiles away in that line (``GetDistance*``, :243-305), and
#   ``IsPathInterrupted`` (:310-343) refuses a line whose tiles in between, or the player's
#   own tile, carry the tile collision bit (``sub_02063E18``, src/map_object_move.c:575-604,
#   through ``TerrainCollisionManager_WillMapObjectCollide``,
#   src/terrain_collision_manager.c:285-302, which reads the static bit); only the trainer
#   then walks (:646-662). So a facing's sight tiles run outwards from the object up to its
#   range and stop at the first collision tile. Talking (``sub_0203C9D4`` /
#   ``sub_0203CA40``, src/unk_0203C954.c:62-97, from src/overlay005/field_control.c:262-273):
#   the player faces the object from any of its four neighbours, or from two tiles away
#   across a ``TileBehavior_IsTable`` tile. The battle script (``Battles_Trainer``,
#   res/field/scripts/scripts_battles.s) moves no one before ``StartTrainerBattle``.
# * The facings an object can have come from its movement type (``FACINGS``) plus its
#   event's initial facing, which it keeps until the type first turns it. A rematch trainer
#   of a Vs. Seeker rematch row is fought only after the Vs. Seeker spins it
#   (``MOVEMENT_TYPE_VS_SEEKER_SPIN``, src/overlay005/vs_seeker.c:659-690, then
#   ``MOVEMENT_TYPE_LOOK_AROUND`` :599-614 or a ``LOOK_*`` type :780-810): all four facings
#   for the trainer types it spins (:536-542). A row's base keeps its own facings: its
#   terrain is the first battle's, a split's fight (ruling B4-19,
#   docs/kaizo/e2-mechanics-ledger.md); a Vs. Seeker rematch can bring the base's own id
#   back at rematch level 0 (:692-773) spun, but that rematch is not a split fight.
# * A battle a coord event's script starts (``Field_CheckCoordEvent`` / ``sub_0203CC14``,
#   src/overlay005/field_control.c:768-775, src/unk_0203C954.c:208-225) starts on a tile of
#   the event's rectangle when every command before the battle leaves the player where it
#   is (``ScriptFacts.safe``) and nothing else starts that script in the header.
#
# Not read, so the set is never smaller than the decomp's: elevation (the height checks of
# ``IsPathInterrupted`` and ``sub_0203C9B0``), objects standing in a sight line,
# ``sub_02064004``'s directional blocks, and whether the player can reach a tile (each only
# removes tiles). Object moves are read from the header's own script file, not from common
# scripts (scripts_common addresses objects of whatever map is current). On a talk
# or coord tile the player can stand on a collision tile only under map features that
# override collision (``TerrainCollisionManager_WillPlayerCollide``,
# src/terrain_collision_manager.c:305-335; ``sCheckCollisionFuncs``,
# src/dynamic_map_features.c:48-60), which only an ``InitPersistedMapFeaturesFor*`` command
# turns on (src/scrcmd.c:4700-4785) and a warp turns off (src/field_map_change.c:267-269):
# those tiles count only when a header the player can walk in from runs one. A trainer whose
# start is not one of these (a movement type ``FACINGS`` does not list, which includes every
# type that moves the object; a script command that moves or turns its object; a script
# battle not started only by a coord event of the header; a command before the battle that
# is not known to leave the player in place) keeps the whole-header set for that header.

# DIR_NORTH, DIR_SOUTH, DIR_WEST, DIR_EAST (include/constants/map_object.h:5-8) as (dx, dz)
# (sMapObjectDxDir / sMapObjectDzDir, src/map_object_move.c:818-845).
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ALL_FACINGS = frozenset(range(len(DIRECTIONS)))
_N, _S, _W, _E = range(4)
# Movement types that never move their object -> the facings they turn it to. A type's
# functions are Unk_020EE3A8[type] (src/unk_020EDBAC.c:559-628): NONE's do nothing
# (sub_020633E0..sub_020633EC, src/map_object.c:2487-2505); LOOK_NORTH .. LOOK_EAST face one
# way once (sub_0206493C..sub_02064960 and sub_02064918, src/unk_0206450C.c:444-486); the
# other LOOK_* types turn among one list of Unk_020EEB08 (sub_0206453C..sub_020645B4 pick it,
# :170-223; sub_020645C0, :225-245, and sub_02065448, :1337-1420, turn only to a listed
# facing; the lists, :1490-1560, :1758-1799). Any other type leaves the header unnarrowed.
FACINGS = {
    "MOVEMENT_TYPE_NONE": (),
    "MOVEMENT_TYPE_LOOK_AROUND": (_N, _S, _W, _E),
    "MOVEMENT_TYPE_LOOK_NORTH_AND_WEST": (_N, _W),
    "MOVEMENT_TYPE_LOOK_NORTH_AND_EAST": (_N, _E),
    "MOVEMENT_TYPE_LOOK_SOUTH_AND_WEST": (_S, _W),
    "MOVEMENT_TYPE_LOOK_SOUTH_AND_EAST": (_S, _E),
    "MOVEMENT_TYPE_LOOK_NORTH_SOUTH_AND_WEST": (_N, _S, _W),
    "MOVEMENT_TYPE_LOOK_NORTH_SOUTH_AND_EAST": (_N, _S, _E),
    "MOVEMENT_TYPE_LOOK_NORTH_WEST_AND_EAST": (_N, _W, _E),
    "MOVEMENT_TYPE_LOOK_SOUTH_WEST_AND_EAST": (_S, _W, _E),
    "MOVEMENT_TYPE_LOOK_NORTH": (_N,),
    "MOVEMENT_TYPE_LOOK_SOUTH": (_S,),
    "MOVEMENT_TYPE_LOOK_WEST": (_W,),
    "MOVEMENT_TYPE_LOOK_EAST": (_E,),
    "MOVEMENT_TYPE_LOOK_NORTH_AND_SOUTH": (_N, _S),
    "MOVEMENT_TYPE_LOOK_WEST_AND_EAST": (_W, _E),
}
# GetTrainerType (src/trainer_encounter.c:165-180) and GetTrainerDistToPlayer (:182-226).
SEES_FACING = (
    "TRAINER_TYPE_NORMAL",
    "TRAINER_TYPE_FACE_SIDES",
    "TRAINER_TYPE_FACE_COUNTERCLOCKWISE",
    "TRAINER_TYPE_FACE_CLOCKWISE",
    "TRAINER_TYPE_SPIN_COUNTERCLOCKWISE",
    "TRAINER_TYPE_SPIN_CLOCKWISE",
)
SEES_ALL = ("TRAINER_TYPE_VIEW_ALL_DIRECTIONS",)
VS_SEEKER_TYPES = SEES_FACING + SEES_ALL  # VsSeekerSystem_CollectViableNPCs (vs_seeker.c:536-542)
# Script commands that leave the player where it is, each with its handler (src/scrcmd.c
# unless named; read in full: none writes the player's position, its movement or a warp).
# ApplyMovement of another object and CallCommonScript of a common script made of these are
# safe too (ScriptFacts.safe). The tile CalcTerrain reads is the player's map position
# (PlayerAvatar_GetXPos / GetZPos), which MapObject_StepDir advances when a step starts
# (src/map_object_move.c:849-858), so letting a step already under way finish moves nothing.
_SAFE_HANDLERS = {
    "END": ":780-784, stops the script",
    "WAITTIME": ":786-799, sets a countdown variable and waits",
    "COMPAREVARTOVALUE": ":944-950",
    "COMPAREVARTOVAR": ":952-958",
    "RETURNCOMMONSCRIPT": ":996-1002, clears the sub-context flag",
    "GOTO": ":1004-1008",
    "CALL": ":1053-1057",
    "RETURN": ":1059-1063",
    "GOTOIF": ":1065-1073",
    "CALLIF": ":1075-1083",
    "SETFLAG": ":1085-1091",
    "CLEARFLAG": ":1093-1099",
    "CHECKFLAG": ":1101-1107",
    "SETTRAINERFLAG": ":1127-1133",
    "CLEARTRAINERFLAG": ":1135-1141",
    "CHECKTRAINERFLAG": ":1143-1149",
    "SETVARFROMVALUE": ":1165-1170",
    "SETVARFROMVAR": ":1172-1178",
    "MESSAGEINSTANT": ":1188-1194",
    "MESSAGE": ":1303-1311",
    "MESSAGENOSKIP": ":1346-1354",
    "WAITBUTTON": ":1418-1422",
    "OPENMESSAGE": ":1462-1472",
    "CLOSEMESSAGE": ":1474-1485",
    "CLOSEMESSAGEWITHOUTERASING": ":1487-1497",
    # Waits until the movement counter of ApplyMovement's tasks is 0 (:2117-2125); only the
    # movements ScriptFacts.safe already allowed (other objects') can be running.
    "WAITMOVEMENT": ":2111-2115",
    # Pauses every object (:2179-2181); with a partner it lets the follower finish its step
    # (:2183-2190); with a talked-to object, LockLastTalked (:2276-2314) pauses all and lets
    # the player, the object and the follower finish a step already under way (:2288-2303):
    # no new step for the player.
    "LOCKALL": ":2174-2198",
    "RELEASEALL": ":2316-2323",
    "LOCKOBJECT": ":2325-2333, pauses one object",
    "RELEASEOBJECT": ":2335-2343",
    # Turns the talked-to object to face the player (ov5_021ECDFC on SCRIPT_MANAGER_TARGET_OBJECT,
    # :2433-2447); the player is only read (PlayerAvatar_GetFacingDir).
    "FACEPLAYER": ":2433-2447",
    "GETPLAYERMAPPOS": ":2449-2460, reads the position",
    "GETPLAYERSTARTERSPECIES": ":3777-3783",
    "GETPLAYERGENDER": ":4498-4505",
    "PLAYMUSIC": "src/scrcmd_sound.c:29-33",
    "STOPMUSIC": "src/scrcmd_sound.c:35-41",
    "PLAYDEFAULTMUSIC": "src/scrcmd_sound.c:43-47",
    "FADEOUTBGM": "src/scrcmd_sound.c:55-63",
    "FADEINBGM": "src/scrcmd_sound.c:70-78",
    "SETBGM": "src/scrcmd_sound.c:89-93",
    "PLAYSE": "src/scrcmd_sound.c:101-105",
    "WAITSE": "src/scrcmd_sound.c:113-119",
    "PLAYFANFARE": "src/scrcmd_sound.c:147-151",
    "WAITFANFARE": "src/scrcmd_sound.c:153-157",
    "BUFFERPLAYERNAME": "src/scrcmd_strings.c:46-54",
    "BUFFERRIVALNAME": "src/scrcmd_strings.c:56-64",
    "BUFFERPLAYERSTARTERSPECIESNAME": "src/scrcmd_strings.c:278-289",
    "BUFFERRIVALSTARTERSPECIESNAME": "src/scrcmd_strings.c:291-302",
}
SAFE_COMMANDS = frozenset(f"SCRCMD_{c}" for c in _SAFE_HANDLERS)
APPLY_MOVEMENT = "SCRCMD_APPLYMOVEMENT"  # localID, movement (src/scrcmd.c:2029-2044)
CALL_COMMON = "SCRCMD_CALLCOMMONSCRIPT"  # scriptID (src/scrcmd.c:973-987)
STARTS_BY_ID = "SCRCMD_UNUSED_013"  # any script id (src/scrcmd.c:960-971)
BATTLE_COMMANDS = frozenset(
    {"SCRCMD_STARTTRAINERBATTLE", "SCRCMD_STARTTAGBATTLE", "SCRCMD_STARTFIRSTBATTLE"}
)
SET_VAR = "SCRCMD_SETVARFROMVALUE"
# Commands that move or turn an object given by local id, their first operand (src/scrcmd.c:
# ScrCmd_ApplyMovement, _SetMovementType, _SetObjectEventPos, _SetPosition,
# _SetObjectEventMovementType, _SetObjectEventDir, _18C).
MOVES_OBJECT = frozenset(
    {
        APPLY_MOVEMENT,
        "SCRCMD_SETMOVEMENTTYPE",
        "SCRCMD_SETOBJECTEVENTPOS",
        "SCRCMD_SETPOSITION",
        "SCRCMD_SETOBJECTEVENTMOVEMENTTYPE",
        "SCRCMD_SETOBJECTEVENTDIR",
        "SCRCMD_18C",
    }
)
FEATURES_PREFIX = "SCRCMD_INITPERSISTEDMAPFEATURESFOR"
DISTORTION_FEATURES = "SCRCMD_INITPERSISTEDMAPFEATURESFORDISTORTIONWORLD"
# Local script ids game code starts (tools.rom.readers.maps.C_STARTED_SCRIPTS cites each
# site). 2, 6, 7 and 8 come only from overlay 9 (src/overlay009/ov9_02249960.c), loaded only
# while the Distortion World's map features are on (src/overlay005/fieldmap.c:170-182,
# :369-380), which only InitPersistedMapFeaturesForDistortionWorld sets (src/scrcmd.c:4780-4785).
C_STARTED_SCRIPTS = (2, 3, 4, 5, 6, 7, 8)
OVERLAY9_ONLY_SCRIPTS = (2, 6, 7, 8)
VARS_START = 0x4000  # operands at or above it are variables (src/script_manager.c:428-452)
# SCRIPT_ID_OFFSET_COMMON_SCRIPTS up to SCRIPT_ID_OFFSET_BG_EVENTS (include/script_manager.h:
# 90-91) run scripts_common (src/script_manager.c:62, :228-254).
COMMON_SCRIPTS = (2000, 2500)
SCRIPT_STACK_SIZE = 20  # ScriptContext.stack (include/field_script_context.h:20)
UNKNOWN_TRAINER = -1  # a battle operand or partner value no constant gives
LOCALID_H = Path("include") / "constants" / "scrcmd.h"


class ObjectEvent(NamedTuple):
    """An object event as ``MapObject`` reads it (tools/jsoncnv/event.py:52-69): ``trainer``
    is the trainer id its battle script names (0: not a trainer object)."""

    local: int
    movement: int
    trainer_type: int
    direction: int
    sight: int  # data[0]
    x: int
    z: int
    trainer: int


class CoordEvent(NamedTuple):
    """A coord event (event.py:109-131): its local script id and rectangle."""

    script: int
    x: int
    z: int
    width: int
    length: int


class HeaderSites(NamedTuple):
    """One map header's events as the narrowing reads them: ``other_scripts`` are the local
    script ids its bg and object events and its init scripts run; ``file`` keys its script
    file in ``narrowed_terrains``' ``scripts``."""

    objects: tuple[ObjectEvent, ...]
    coords: tuple[CoordEvent, ...]
    other_scripts: frozenset[int]
    file: object


class Cmd(NamedTuple):
    """One script command: its ``SCRCMD_*`` constant (``"?<macro>"`` for a decomp macro the
    reader cannot expand), its operands (None: a value the reader does not resolve; a branch
    command's offset is ``target`` instead), and the command after it (``next``)."""

    const: str
    args: tuple
    target: object
    next: object


class FieldRules(NamedTuple):
    """``FACINGS`` and the trainer types by id (generated/movement_types.txt,
    generated/trainer_types.txt) and ``LOCALID_PLAYER`` (include/constants/scrcmd.h:15)."""

    facings: Mapping[int, frozenset[int]]
    sees_facing: frozenset[int]
    sees_all: frozenset[int]
    vs_seeker: frozenset[int]
    localid_player: int


def field_rules(decomp: Path) -> FieldRules:
    decomp = Path(decomp)
    movements = _constants(decomp, "movement_types", "MOVEMENT_TYPE_")
    types = _constants(decomp, "trainer_types", "TRAINER_TYPE_")
    missing = [n for n in [*FACINGS, *VS_SEEKER_TYPES] if n not in movements + types]
    if missing:
        raise TerrainError(f"generated movement/trainer types lack {missing}")
    player = _defines(_read(decomp, LOCALID_H), LOCALID_H, ("LOCALID_PLAYER",))
    return FieldRules(
        facings={movements.index(n): frozenset(f) for n, f in FACINGS.items()},
        sees_facing=frozenset(types.index(n) for n in SEES_FACING),
        sees_all=frozenset(types.index(n) for n in SEES_ALL),
        vs_seeker=frozenset(types.index(n) for n in VS_SEEKER_TYPES),
        localid_player=player["LOCALID_PLAYER"],
    )


def _is_var(v) -> bool:
    return v is None or v >= VARS_START


def explore(script, starts, step) -> None:
    """Walk ``script`` (``.cmd(node) -> Cmd``) from ``starts`` ((node, state)) by control
    flow, as ``tools.rom.readers.maps._walk`` does: ``Call``/``CallIf`` push the command
    after them, ``Return`` pops (an empty stack ends the path), ``End`` ends and ``GoTo``
    jumps, any other branch command may jump or fall through. ``step(cmd, state)`` gives the
    state after a command, or None to end the path there."""
    seen = set()
    work = [(node, (), state) for node, state in starts]
    while work:
        node, stack, state = work.pop()
        if node is None or (node, stack, state) in seen:
            continue
        seen.add((node, stack, state))
        cmd = script.cmd(node)
        state = step(cmd, state)
        if state is None:
            continue
        c = cmd.const
        if c == "SCRCMD_END":
            continue
        if c == "SCRCMD_RETURN":
            if stack:
                work.append((stack[-1], stack[:-1], state))
            continue
        if c in ("SCRCMD_CALL", "SCRCMD_CALLIF"):
            if len(stack) + 1 >= SCRIPT_STACK_SIZE:
                raise TerrainError(f"script call stack overflow at {node!r}")
            work.append((cmd.target, (*stack, cmd.next), state))
            if c == "SCRCMD_CALL":
                continue
        elif c == "SCRCMD_GOTO":
            work.append((cmd.target, stack, state))
            continue
        elif cmd.target is not None:
            work.append((cmd.target, stack, state))
        work.append((cmd.next, stack, state))


class ScriptFacts:
    """What the narrowing reads from one script file (``script``: ``.entries``, the nodes of
    its local ids 1.., ``.cmd(node)`` and ``.partner_var``) and the common scripts it calls
    (``common``, the same shape): per local id, the trainers its battles and partner variable
    can name (``placed``) and the trainers it starts with the player still where the script
    began (``started``); over the whole file, the constants it reaches (``consts``), the
    local ids it moves objects by (``moved``) and whether it can start any of its own entries
    by id (``any_entry``)."""

    def __init__(self, script, common, fr: FieldRules):
        self.script, self.common, self.fr = script, common, fr
        self._placed: dict[int, frozenset[int]] = {}
        self._started: dict[int, tuple[frozenset[int], frozenset[int]]] = {}
        self._common_safe: dict[int, bool] = {}
        reached: list[Cmd] = []

        def keep(cmd, state):
            reached.append(cmd)
            return state

        explore(script, [(e, 0) for e in script.entries], keep)
        self.consts = frozenset(c.const for c in reached)
        self.moved = frozenset(
            UNKNOWN_TRAINER if _is_var(c.args[0] if c.args else None) else c.args[0]
            for c in reached
            if c.const in MOVES_OBJECT
        )
        self.any_entry = any(
            c.const == STARTS_BY_ID
            or c.const.startswith("?")
            or (
                c.const == CALL_COMMON
                and (not c.args or _is_var(c.args[0]) or c.args[0] < COMMON_SCRIPTS[0])
            )
            for c in reached
        )

    def _entry(self, local: int) -> list:
        if not 1 <= local <= len(self.script.entries):
            return []
        return [(self.script.entries[local - 1], False)]

    def placed(self, local: int) -> frozenset[int]:
        """Every trainer a battle command reached from entry ``local`` names and every
        constant it gives the partner variable (``maps.script_battles``); ``UNKNOWN_TRAINER``
        for a variable operand or a command the reader cannot expand."""
        if local not in self._placed:
            out: set[int] = set()

            def step(cmd, state):
                if cmd.const in BATTLE_COMMANDS:
                    out.update(UNKNOWN_TRAINER if _is_var(a) else a for a in cmd.args)
                elif cmd.const == SET_VAR and cmd.args[0] in (self.script.partner_var, None):
                    out.add(UNKNOWN_TRAINER if cmd.args[1] is None else cmd.args[1])
                elif cmd.const.startswith("?"):
                    out.add(UNKNOWN_TRAINER)
                return state

            explore(self.script, self._entry(local), step)
            out.discard(0)
            self._placed[local] = frozenset(out)
        return self._placed[local]

    def started(self, local: int) -> tuple[frozenset[int], frozenset[int]]:
        """(the trainers a battle reached from entry ``local`` starts while every command
        before it is ``safe``, the trainers of every other battle and partner value)."""
        if local not in self._started:
            safe: set[int] = set()
            other: set[int] = set()

            def step(cmd, moved):
                if cmd.const in BATTLE_COMMANDS:
                    for a in cmd.args:
                        if _is_var(a):
                            other.add(UNKNOWN_TRAINER)
                        elif a:
                            (other if moved else safe).add(a)
                    return moved
                if cmd.const == SET_VAR and cmd.args[0] in (self.script.partner_var, None):
                    other.add(UNKNOWN_TRAINER if cmd.args[1] is None else cmd.args[1])
                return moved or not self.safe(cmd)

            explore(self.script, self._entry(local), step)
            self._started[local] = (frozenset(safe), frozenset(other))
        return self._started[local]

    def safe(self, cmd: Cmd) -> bool:
        """Whether ``cmd`` leaves the player where it is (``SAFE_COMMANDS``)."""
        if cmd.const in SAFE_COMMANDS:
            return True
        if cmd.const == APPLY_MOVEMENT:
            local = cmd.args[0] if cmd.args else None
            return not _is_var(local) and local != self.fr.localid_player
        if cmd.const == CALL_COMMON and cmd.args and not _is_var(cmd.args[0]):
            return self._common_is_safe(cmd.args[0])
        return False

    def _common_is_safe(self, script_id: int) -> bool:
        """Whether common script ``script_id`` (its own context, run until it ends,
        src/scrcmd.c:973-1002) is made of ``safe`` commands and starts no battle."""
        lo, hi = COMMON_SCRIPTS
        if self.common is None or not lo <= script_id < hi:
            return False
        if not 0 <= script_id - lo < len(self.common.entries):
            return False
        if script_id not in self._common_safe:
            self._common_safe[script_id] = False  # a call cycle back into it is not safe
            ok = [True]

            def step(cmd, state):
                if cmd.const in BATTLE_COMMANDS or not self.safe(cmd):
                    ok[0] = False
                    return None
                return state

            explore(self.common, [(self.common.entries[script_id - lo], 0)], step)
            self._common_safe[script_id] = ok[0]
        return self._common_safe[script_id]


class _HeaderTiles:
    """The terrain attribute u16s tile ``(x, z)`` can have while header ``h`` is current
    (``header_terrains``' tiles): on every matrix some header loads, a tile whose cell is
    labelled ``h`` (any cell of ``h``'s own matrix when it has no header section), under
    each land data id the cell can hold, or one of ``h``'s warp arrivals in its own matrix.
    ``unread`` marks an arrival outside that matrix."""

    def __init__(self, rules, headers, matrices, lands, h, arrivals, cache):
        self.rules, self.matrices, self.lands, self.h, self.cache = (
            rules,
            matrices,
            lands,
            h,
            cache,
        )
        self.own, self.background = headers[h]
        self.loadable = sorted({matrix for matrix, _ in headers})
        self.arrivals = set(arrivals.get(h, ()))

    def _attrs(self, land: int) -> tuple[int, ...]:
        if not 0 <= land < len(self.lands):
            raise TerrainError(f"land data {land} is not a member")
        if land not in self.cache:
            self.cache[land] = land_attributes(self.rules, self.lands[land], land)
        return self.cache[land]

    def at(self, x: int, z: int) -> tuple[set[int], bool]:
        side, out = self.rules.tiles_x, set()
        if x < 0 or z < 0:
            return out, False
        for matrix in self.loadable:
            m = self.matrices[matrix]
            if m.headers is None and matrix != self.own:
                continue
            if not (x // side < m.width and z // side < m.height):
                continue
            cell = x // side + m.width * (z // side)
            arrival = matrix == self.own and (x, z) in self.arrivals
            if m.headers is not None and m.headers[cell] != self.h and not arrival:
                continue
            for land in _land_ids(m, matrix, self.rules, [cell]) - {self.rules.invalid_land}:
                out.add(self._attrs(land)[(z % side) * side + x % side])
        own = self.matrices[self.own]
        unread = (x, z) in self.arrivals and not (x // side < own.width and z // side < own.height)
        return out, unread


def _sight_tiles(tiles: _HeaderTiles, obj: ObjectEvent, facings: Iterable[int]) -> set:
    """The tiles a trainer facing one of ``facings`` can spot the player on: 1..``sight``
    tiles out, up to the first tile whose every variant carries the collision bit."""
    out = set()
    for d in facings:
        dx, dz = DIRECTIONS[d]
        for k in range(1, obj.sight + 1):
            x, z = obj.x + k * dx, obj.z + k * dz
            attrs, _ = tiles.at(x, z)
            if attrs and all(a & tiles.rules.collision for a in attrs):
                break  # IsPathInterrupted
            out.add((x, z))
    return out


def _talk_tiles(tiles: _HeaderTiles, obj: ObjectEvent) -> set:
    """The tiles the player can talk to ``obj`` from (``sub_0203C9D4``): a neighbour when
    the object's own tile can be other than a table, the tile beyond a neighbour that can
    be one."""
    rules = tiles.rules
    out = set()
    own, _ = tiles.at(obj.x, obj.z)
    for dx, dz in DIRECTIONS:
        if not own or any((a & rules.mask) not in rules.table for a in own):
            out.add((obj.x + dx, obj.z + dz))
        between, _ = tiles.at(obj.x + dx, obj.z + dz)
        if any((a & rules.mask) in rules.table for a in between):
            out.add((obj.x + 2 * dx, obj.z + 2 * dz))
    return out


def _tile_terrains(tiles: _HeaderTiles, spots: Iterable, keep_collision: bool) -> set[int]:
    """``CalcTerrain`` over ``spots`` with the header's background, and the water background
    on a tile a surfing player stands on; ``TERRAIN_UNKNOWN`` for an unread arrival. A tile
    variant with the collision bit counts only when ``keep_collision``."""
    rules = tiles.rules
    out: set[int] = set()
    for x, z in spots:
        attrs, unread = tiles.at(x, z)
        if unread:
            out.add(TERRAIN_UNKNOWN)
        for a in attrs:
            if a & rules.collision and not keep_collision:
                continue
            b = a & rules.mask
            out.add(calc_terrain(rules, b, tiles.background))
            if b in rules.on_water:
                out.add(calc_terrain(rules, b, rules.water))
    return out


def narrowed_terrains(
    rules: TerrainRules,
    fr: FieldRules,
    headers: Sequence[tuple[int, int]],
    matrices: Sequence[Matrix],
    lands: Sequence[bytes],
    where: Mapping[int, Iterable[int]],
    arrivals: Mapping[int, Iterable[tuple[int, int]]],
    sites: Mapping[int, HeaderSites],
    scripts: Mapping[object, ScriptFacts],
    rematch_rows: Iterable[tuple[int, Iterable[int]]],
) -> dict[tuple[int, int], frozenset[int]]:
    """``(trainer, header) -> terrains`` for each pair of ``where`` (trainer -> headers)
    whose battle the notes above narrow to known tiles; the other pairs are left out, so
    ``trainer_terrains`` keeps the whole-header set for them. ``sites[h]``: header ``h``'s
    events; ``scripts[file]``: its script file's facts; ``rematch_rows``: ``(base,
    rematches)`` of ``gVsSeekerRematchData`` (a rematch is fought at its base's objects)."""
    bases: dict[int, set[int]] = {}
    spins: set[int] = set()
    firsts: set[int] = set()
    for base, rematches in rematch_rows:
        firsts.add(base)
        for t in rematches:
            spins.add(t)
            bases.setdefault(t, set()).add(base)
    # A base is first fought unspun (its own row's first battle, ruling B4-19); only a rematch
    # id that is no row's base is always a Vs. Seeker battle.
    spins -= firsts
    cache: dict[int, tuple[int, ...]] = {}
    tiles_of: dict[int, _HeaderTiles] = {}
    features: dict[int, bool] = {}
    loadable = sorted({matrix for matrix, _ in headers})

    def keeps_collision(h: int) -> bool:
        """Whether ``h`` or a header the player can walk into it from (another label of a
        matrix with a cell labelled ``h``, ``FieldMap_ChangeZone``) turns map features on."""
        if h not in features:
            near = {h}
            for matrix in loadable:
                labels = matrices[matrix].headers
                if labels is not None and h in labels:
                    near |= set(labels)
            features[h] = any(
                c.startswith(FEATURES_PREFIX)
                for g in near
                if g in sites
                for c in scripts[sites[g].file].consts
            )
        return features[h]

    out: dict[tuple[int, int], frozenset[int]] = {}
    for t in sorted(where):
        ids = {t} | bases.get(t, set())
        for h in sorted(set(where[t])):
            if h not in sites:
                continue
            site = sites[h]
            facts = scripts[site.file]
            if h not in tiles_of:
                tiles_of[h] = _HeaderTiles(rules, headers, matrices, lands, h, arrivals, cache)
            tiles = tiles_of[h]
            sight: set = set()
            stand: set = set()
            ok = True
            objects = [o for o in site.objects if o.trainer in ids]
            for o in objects:
                if (
                    o.movement not in fr.facings
                    or o.local in facts.moved
                    or (UNKNOWN_TRAINER in facts.moved)
                ):
                    ok = False
                    break
                facings = fr.facings[o.movement] | {o.direction}
                if t in spins and o.trainer_type in fr.vs_seeker:
                    facings = ALL_FACINGS
                if o.trainer_type in fr.sees_all:
                    facings = ALL_FACINGS
                elif o.trainer_type not in fr.sees_facing:
                    facings = frozenset()
                sight |= _sight_tiles(tiles, o, facings)
                stand |= _talk_tiles(tiles, o)
            entries = [
                e
                for e in range(1, len(facts.script.entries) + 1)
                if facts.placed(e) & (ids | {UNKNOWN_TRAINER})
            ]
            coord_scripts = {c.script for c in site.coords}
            for e in entries if ok else ():
                safe, other = facts.started(e)
                c_started = e in C_STARTED_SCRIPTS and (
                    e not in OVERLAY9_ONLY_SCRIPTS or DISTORTION_FEATURES in facts.consts
                )
                if (
                    facts.any_entry
                    or e not in coord_scripts
                    or e in site.other_scripts
                    or c_started
                    or other & (ids | {UNKNOWN_TRAINER})
                    or not safe & ids
                ):
                    ok = False
                    break
                for c in site.coords:
                    if c.script == e:
                        stand |= {
                            (c.x + i, c.z + j) for i in range(c.width) for j in range(c.length)
                        }
            if not ok or not (objects or entries):
                continue
            terrains = _tile_terrains(tiles, sight, False)
            terrains |= _tile_terrains(tiles, stand, keeps_collision(h))
            if terrains:
                out[(t, h)] = frozenset(terrains)
    return out


# --- The decomp's sources of the same data (vanilla; tools.extract_decomp) -------------

MATRICES = Path("res") / "field" / "matrices"
MATRICES_ORDER = MATRICES / "map_matrices.order"  # map_matrix.narc member order
LAND_DATA = Path("res") / "field" / "maps" / "data"
LAND_DATA_ORDER = LAND_DATA / "map_data.order"  # land_data.narc member order
EVENTS = Path("res") / "field" / "events"
MAP_HEADERS_H = Path("include") / "data" / "map_headers.h"
NO_SCRIPT = 65535  # an object event clone (ObjectEvent_HasNoScript, src/map_object.c:2322)


def _order(decomp: Path, rel: Path) -> list[str]:
    return [s.strip() for s in _read(decomp, rel).splitlines() if s.strip()]


def _valued_constants(decomp: Path, name: str) -> dict[str, int]:
    """``generated/<name>.txt`` with ``NAME = value`` lines: a bare line is one more than
    the line before it."""
    out: dict[str, int] = {}
    nxt = 0
    for raw in _read(decomp, Path("generated") / f"{name}.txt").splitlines():
        m = re.fullmatch(r"\s*([A-Z][A-Z0-9_]*)(?:\s*=\s*(\d+))?\s*", raw)
        if not m:
            raise TerrainError(f"generated/{name}.txt: unparsed line {raw!r}")
        nxt = int(m.group(2)) if m.group(2) is not None else nxt
        out[m.group(1)] = nxt
        nxt += 1
    return out


def header_names(decomp: Path) -> list[str]:
    """``generated/map_headers.txt`` up to ``MAP_HEADER_COUNT``: list index = header id."""
    out = []
    for raw in _read(decomp, Path("generated") / "map_headers.txt").splitlines():
        line = raw.strip()
        if line == "MAP_HEADER_COUNT":
            return out
        if not re.fullmatch(r"MAP_HEADER_[A-Z0-9_]+", line):
            raise TerrainError(f"generated/map_headers.txt: unexpected line {raw!r}")
        out.append(line)
    raise TerrainError("generated/map_headers.txt: no MAP_HEADER_COUNT")


def decomp_header_battle_fields(decomp: Path) -> list[tuple[int, int]]:
    """(mapMatrixID, battleBG) of each map header (include/data/map_headers.h), by header
    id (generated/map_headers.txt): the matrix is its ``map_matrices.order`` member."""
    decomp = Path(decomp)
    names = header_names(decomp)
    backgrounds = battle_constants(decomp, "battle_backgrounds", "BACKGROUND_")
    matrices = _order(decomp, MATRICES_ORDER)
    src = _read(decomp, MAP_HEADERS_H)
    fields = {}
    for name, body in re.findall(
        r"\n    \[(MAP_HEADER_\w+)\] = \{\n(.*?)\n    \},", src, re.DOTALL
    ):
        f = dict(re.findall(r"\.(\w+) = ([^,\n]+),", body))
        if f.get("mapMatrixID") not in matrices or f.get("battleBG") not in backgrounds:
            raise TerrainError(f"{MAP_HEADERS_H}: {name} matrix/battleBG not parsed")
        fields[name] = (matrices.index(f["mapMatrixID"]), backgrounds.index(f["battleBG"]))
    missing = [n for n in names if n not in fields]
    if missing:
        raise TerrainError(f"{MAP_HEADERS_H}: no header entries for {missing}")
    return [fields[n] for n in names]


def decomp_matrices(decomp: Path) -> list[Matrix]:
    """The ``map_matrix.narc`` members from their JSON (tools/jsoncnv/map_matrix.py packs
    ``maps``' row count as the height, the first row's length as the width, and the
    ``headers`` section only when it is non-empty)."""
    decomp = Path(decomp)
    header_ids = {n: i for i, n in enumerate(header_names(decomp))}
    land_ids = _valued_constants(decomp, "maps")
    out = []
    for stem in _order(decomp, MATRICES_ORDER):
        d = json.loads(_read(decomp, MATRICES / f"{stem}.json"))
        rows = d["maps"]
        width, height = (len(rows[0]) if rows else 0), len(rows)
        if any(len(r) != width for r in rows):
            raise TerrainError(f"{stem}.json: ragged maps")
        headers = None
        if d["headers"]:
            headers = tuple(header_ids[h] for r in d["headers"] for h in r)
            if len(headers) != width * height:
                raise TerrainError(f"{stem}.json: headers do not match maps")
        out.append(Matrix(width, height, headers, tuple(land_ids[m] for r in rows for m in r)))
    return out


def decomp_land_data(decomp: Path) -> list[bytes]:
    """The ``land_data.narc`` members (``map_data.order``), as built."""
    return [(Path(decomp) / LAND_DATA / f).read_bytes() for f in _order(decomp, LAND_DATA_ORDER)]


def decomp_warp_tiles(decomp: Path, events: str) -> list[tuple[int, int]]:
    """(x, z) of an events archive's warp events (``res/field/events/<events>.json``)."""
    d = json.loads(_read(decomp, EVENTS / f"{events}.json"))
    return [(w["x"], w["z"]) for w in d["warp_events"]]


def decomp_check_trainer_objects(decomp: Path, headers: Mapping[str, Mapping[str, str]]) -> None:
    """``tools.rom.readers.maps.check_trainer_objects`` on the decomp's sources
    (``headers``: MAP_HEADER_* -> its ``scriptsArchiveID`` and ``eventsArchiveID``): an
    object whose script is ``NO_SCRIPT`` clones the object with its local id (``clone_id``,
    else its index) in the header its ``hidden_flag`` names; a ``SetObjectFlagIsPersistent
    <id>, TRUE`` line of a header's script file naming one of its trainer objects (or not a
    ``LOCALID_*`` constant) makes it persistent."""
    objects = {}
    names = {}
    for header, h in headers.items():
        d = json.loads(_read(decomp, EVENTS / f"{h['eventsArchiveID']}.json"))
        objects[header] = [
            (o.get("clone_id", i), str(o["script"]), o["hidden_flag"])
            for i, o in enumerate(d["object_events"])
        ]
        names[header] = {
            o.get("id") for o in d["object_events"] if str(o["script"]).startswith("TRAINER_")
        }
    trainers = {
        (h, local)
        for h, objs in objects.items()
        for local, s, _ in objs
        if s.startswith("TRAINER_")
    }
    for h, objs in objects.items():
        for local, script, flag in objs:
            if script == str(NO_SCRIPT) and (flag, local) in trainers:
                raise TerrainError(
                    f"{h} clones trainer object {local} of {flag}: its battle can start in a "
                    "header the trainer -> map join does not list"
                )
    for header, h in headers.items():
        if not names[header]:
            continue
        text = _read(decomp, Path("res") / "field" / "scripts" / f"{h['scriptsArchiveID']}.s")
        for local in re.findall(
            r"^\s*SetObjectFlagIsPersistent (\w+), TRUE\s*$", text, re.MULTILINE
        ):
            if local in names[header] or not local.startswith("LOCALID_"):
                raise TerrainError(
                    f"{header}: makes object {local} persistent while the header has trainer "
                    "objects; the battle could start in another header"
                )


# --- The narrowing's inputs from the decomp's sources (vanilla; tools.extract_decomp) ----

SCRIPTS_DIR = Path("res") / "field" / "scripts"
SCRIPTS_ORDER = SCRIPTS_DIR / "scripts.order"  # scr_seq.narc member order
COMMON_FILE = "scripts_common"
SCRCMD_INC = Path("asm") / "macros" / "scrcmd.inc"
# The branch commands, whose last operand is an offset (tools.rom.readers.maps._BRANCHES).
BRANCH_COMMANDS = frozenset(
    {
        "SCRCMD_GOTO",
        "SCRCMD_CALL",
        "SCRCMD_GOTOIF",
        "SCRCMD_CALLIF",
        "SCRCMD_GOTOIFTARGETTRAINERDEFEATED",
        "SCRCMD_UNUSED_017",
        "SCRCMD_UNUSED_018",
        "SCRCMD_UNUSED_019",
    }
)
_VAR_TEST = re.compile(
    r"\.if \(\(\\(\w+) >= VARS_START && \\\1 <= VARS_END\) \|\| "
    r"\(\\\1 >= SCRIPT_LOCAL_VARS_START && \\\1 <= SCRIPT_LOCAL_VARS_END\)\)"
)


def _macros(decomp: Path) -> dict[str, tuple[list[tuple[str, str | None]], list[str]]]:
    """``asm/macros/scrcmd.inc``: macro -> (parameters with defaults, body lines)."""
    src = re.sub(r"/\*.*?\*/", "", _read(decomp, SCRCMD_INC), flags=re.DOTALL)
    out = {}
    for m in re.finditer(
        r"^[ \t]*\.macro[ \t]+(\w+)([^\n]*)\n(.*?)^[ \t]*\.endm\b", src, re.MULTILINE | re.DOTALL
    ):
        params = []
        for p in (s.strip() for s in m.group(2).split(",")):
            if p:
                name, _, default = p.partition("=")
                params.append((name.strip(), default.strip() or None))
        body = [re.split(r"//", s, maxsplit=1)[0].strip() for s in m.group(3).splitlines()]
        out[m.group(1)] = (params, [s for s in body if s])
    return out


class DecompScript:
    """A ``res/field/scripts/<file>.s`` file as commands: each line's macro expanded through
    ``scrcmd.inc`` down to its ``.short SCRCMD_*`` commands, as the assembler does (a
    ``.if`` other than ``SetVar``/``CompareVar``'s variable test leaves the line one
    ``"?<macro>"`` command). Nodes are command indices; ``entries`` are the ``ScriptEntry``
    labels' nodes."""

    def __init__(self, decomp: Path, file: str, macros, symbols: Mapping[str, int], partner_var):
        from tools.battle_scripts import is_var

        self.partner_var = partner_var
        path = Path(decomp) / SCRIPTS_DIR / f"{file}.s"
        text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
        defines: dict[str, str] = {}
        local: dict[str, int] = {}
        m = re.search(r'#include "res/field/events/(\w+)\.h"', text)
        if m:
            events = json.loads(_read(decomp, EVENTS / f"{m.group(1)}.json"))
            local = {
                o["id"]: o.get("clone_id", i)
                for i, o in enumerate(events["object_events"])
                if "id" in o
            }

        def value(token: str, depth: int = 0):
            """A constant, a name the file defines or ``symbols`` holds, or an expression of
            those with ``+ - * < > == !=`` (the assembler's); None when a name is unknown."""
            token = token.strip()
            if re.fullmatch(r"-?(\d+|0[xX][0-9a-fA-F]+)", token):
                return int(token, 0)
            if token in defines and depth < 8:
                return value(defines[token], depth + 1)
            if token in local:
                return local[token]
            if token in symbols:
                return symbols[token]
            parts = re.split(r"(\w+)", token)
            operators = "".join(parts[::2])
            if depth < 8 and operators.strip() and re.fullmatch(r"[\s()+\-*<>=!]*", operators):
                names = [value(n, depth + 1) if not n.isdigit() else int(n) for n in parts[1::2]]
                if None in names:
                    return None
                expr = "".join(p if i % 2 == 0 else str(names[i // 2]) for i, p in enumerate(parts))
                return int(eval(expr, {"__builtins__": {}}))  # digits and operators only
            return None

        def split_args(text: str) -> list[str]:
            """Macro arguments: separated by commas, or by blanks between plain words (the
            assembler accepts both, e.g. ``ShowArrowSign``'s body)."""
            out = []
            for piece in (a.strip() for a in text.split(",")) if text.strip() else ():
                if re.fullmatch(r"[\w.\\]+(\s+[\w.\\]+)+", piece):
                    out.extend(piece.split())
                else:
                    out.append(piece)
            return out

        def expand(name: str, args: list[str], depth: int = 0) -> list:
            if name not in macros or depth > 16:
                return [("?" + name, (), None)]
            params, body = macros[name]
            bind = {}
            for i, (p, default) in enumerate(params):
                bind[p] = args[i] if i < len(args) and args[i] != "" else (default or "")
            out: list = []
            live = [True]
            for line in body:
                if line.startswith(".if"):
                    if not all(live):
                        live.append(False)
                        continue
                    t = _VAR_TEST.fullmatch(line)
                    blank = re.fullmatch(r"\.if(n?)b\s+\\(\w+)", line)
                    if t:
                        # SetVar / CompareVar (scrcmd.inc:178-184, :309-315): a name the
                        # variable list does not hold is a value, not a variable.
                        v = value(bind.get(t.group(1), ""))
                        live.append(v is not None and is_var(v))
                    elif blank:
                        empty = not bind.get(blank.group(2), "").strip()
                        live.append(empty != (blank.group(1) == "n"))
                    else:
                        cond = re.sub(
                            r"\\(\w+)", lambda g: bind.get(g.group(1), g.group(0)), line[3:]
                        )
                        v = value(cond)
                        if v is None:
                            return [("?" + name, (), None)]
                        live.append(bool(v))
                    continue
                if line == ".else":
                    live[-1] = not live[-1]
                    continue
                if line == ".endif":
                    live.pop()
                    continue
                if not all(live):
                    continue
                line = re.sub(r"\\(\w+)", lambda g: bind.get(g.group(1), g.group(0)), line)
                d = re.fullmatch(r"\.(byte|short|long)\s+(.+)", line)
                if d:
                    for operand in (s.strip() for s in d.group(2).split(",")):
                        if d.group(1) == "short" and operand.upper().startswith("SCRCMD_"):
                            out.append([operand, [], None])
                        elif not out:
                            return [("?" + name, (), None)]
                        elif re.fullmatch(r"\w+-\.-4", operand):
                            label = operand[: -len("-.-4")]
                            if out[-1][0] in BRANCH_COMMANDS:
                                out[-1][2] = label
                            else:
                                out[-1][1].append(None)
                        else:
                            out[-1][1].append(value(operand))
                    continue
                if line.startswith((".error", ".set")):
                    continue
                inner = re.fullmatch(r"(\w+)\s*(.*)", line)
                if not inner:
                    return [("?" + name, (), None)]
                sub = split_args(inner.group(2))
                out.extend(list(c) for c in expand(inner.group(1), sub, depth + 1))
            return [(c, tuple(a), t) for c, a, t in out]

        cmds: list = []
        labels: dict[str, int] = {}
        entries: list[str] = []
        accept = True
        for raw in text.splitlines():
            line = re.split(r"//|@", raw, maxsplit=1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                d = re.fullmatch(r"#define\s+(\w+)\s+(.+)", line)
                if d:
                    defines[d.group(1)] = d.group(2).strip()
                continue
            lab = re.fullmatch(r"(\w+):", line)
            if lab:
                labels[lab.group(1)] = len(cmds)
                continue
            if line.startswith("."):
                continue  # .balign and data directives
            name, rest = re.fullmatch(r"(\S+)\s*(.*)", line).groups()
            if name == "ScriptEntry" and accept:
                entries.append(rest.strip())
                continue
            if name == "ScriptEntryEnd":
                accept = False
                continue
            cmds.extend(expand(name, split_args(rest)))
        missing = [e for e in entries if e not in labels]
        if missing:
            raise TerrainError(f"{path}: ScriptEntry labels {missing} not defined")
        self.cmds = [
            Cmd(
                c, a, labels.get(t) if t is not None else None, i + 1 if i + 1 < len(cmds) else None
            )
            for i, (c, a, t) in enumerate(cmds)
        ]
        self.entries = [labels[e] for e in entries]

    def cmd(self, node: int) -> Cmd:
        return self.cmds[node]


def _decomp_init_ids(decomp: Path, file: str) -> list[int]:
    """The script ids an init scripts file (``InitScriptEntry_*``,
    ``InitScriptGoToIfEqual``, scrcmd.inc:5121-5165) runs."""
    text = _read(decomp, SCRIPTS_DIR / f"{file}.s")
    ids = re.findall(
        r"^\s*InitScriptEntry_(?:OnTransition|OnResume|OnLoad|Fixed\s+\w+,)\s*(\w+)",
        text,
        re.MULTILINE,
    )
    ids += re.findall(r"^\s*InitScriptGoToIfEqual\s+[^,]+,\s*[^,]+,\s*(\w+)", text, re.MULTILINE)
    return [int(i, 0) for i in ids if re.fullmatch(r"\d+|0[xX][0-9a-fA-F]+", i)]


def decomp_narrowing_inputs(decomp: Path, trainers: Sequence[str]):
    """(``sites``, ``scripts``, ``rematch_rows``, ``FieldRules``) for ``narrowed_terrains``
    from the decomp's map headers (include/data/map_headers.h), events JSON and script
    sources; ``trainers`` is the ``TRAINER_*`` list (id = index)."""
    from tools.battle_scripts import var_ids
    from tools.extract_decomp import _rematches

    decomp = Path(decomp)
    fr = field_rules(decomp)
    movements = _constants(decomp, "movement_types", "MOVEMENT_TYPE_")
    types = _constants(decomp, "trainer_types", "TRAINER_TYPE_")
    trainer_id = {n: i for i, n in enumerate(trainers)}
    vars_ = var_ids(decomp)
    symbols = {"TRUE": 1, "FALSE": 0, **vars_, **trainer_id}
    symbols.update({n: i for i, n in enumerate(_constants(decomp, "time_of_day", "TIMEOFDAY_"))})
    symbols.update(
        _defines(
            _read(decomp, LOCALID_H),
            LOCALID_H,
            ("LOCALID_PLAYER", "LOCALID_CAMERA", "LOCALID_FOLLOWER"),
        )
    )
    partner = vars_["VAR_PARTNER_TRAINER_ID"]
    macros = _macros(decomp)
    names = header_names(decomp)
    src = _read(decomp, MAP_HEADERS_H)
    fields = {
        name: dict(re.findall(r"\.(\w+) = ([^,\n]+),", body))
        for name, body in re.findall(
            r"\n    \[(MAP_HEADER_\w+)\] = \{\n(.*?)\n    \},", src, re.DOTALL
        )
    }
    files: dict[str, DecompScript] = {}
    common = DecompScript(decomp, COMMON_FILE, macros, symbols, partner)

    def script_trainer(script) -> int:
        if isinstance(script, str):
            if script.startswith("TRAINER_"):
                return trainer_id[script]
            script = int(script, 0) if re.fullmatch(r"\d+", script) else 0
        return _battle_script_trainer(script)

    sites = {}
    for h, name in enumerate(names):
        f = fields[name]
        events = json.loads(_read(decomp, EVENTS / f"{f['eventsArchiveID']}.json"))
        objects = tuple(
            ObjectEvent(
                o.get("clone_id", i),
                movements.index(o["movement_type"]),
                types.index(o["trainer_type"]),
                int(o["initial_dir"]),
                int(o["data"][0]) if o["data"] else 0,
                int(o["x"]),
                int(o["z"]),
                script_trainer(o["script"]),
            )
            for i, o in enumerate(events["object_events"])
        )
        coords = tuple(
            CoordEvent(
                int(c["script"]), int(c["x"]), int(c["z"]), int(c["width"]), int(c["length"])
            )
            for c in events["coord_events"]
        )
        other = {int(b["script"]) for b in events["bg_events"]}
        other |= {o["script"] for o in events["object_events"] if isinstance(o["script"], int)}
        other |= set(_decomp_init_ids(decomp, f["initScriptsArchiveID"]))
        file = f["scriptsArchiveID"]
        if file not in files:
            files[file] = DecompScript(decomp, file, macros, symbols, partner)
        sites[h] = HeaderSites(objects, coords, frozenset(other), file)
    scripts = {k: ScriptFacts(v, common, fr) for k, v in files.items()}
    rows = [(trainer_id[b], [trainer_id[t] for t in ts]) for b, ts in _rematches(decomp).items()]
    # NoUniqueRematches(id) is the row {id, id, END...} (src/overlay005/vs_seeker.c:53): the
    # Vs. Seeker rematches the trainer as itself, so it can be fought spun too.
    vs_seeker = _read(decomp, Path("src") / "overlay005" / "vs_seeker.c")
    rows += [
        (trainer_id[t], [trainer_id[t]])
        for t in re.findall(r"^\s*NoUniqueRematches\((TRAINER_\w+)\),", vs_seeker, re.MULTILINE)
    ]
    return sites, scripts, rows, fr


def _battle_script_trainer(script: int) -> int:
    """The trainer a single (3000 + id - 1) or double (5000 + id - 1) battle script names
    (include/script_manager.h:93-95), else 0."""
    if 3000 <= script < 5000:
        return script - 3000 + 1
    if 5000 <= script < 7000:
        return script - 5000 + 1
    return 0


# --- The narrowing's inputs from a ROM (either game; tools.rom.readers.maps) ------------


class RomScript:
    """A ``scr_seq.narc`` member as commands (``tools.rom.readers.maps._decode``): nodes are
    offsets, ``entries`` the header's entry offsets."""

    def __init__(self, script: bytes, table, name: str):
        from tools.rom.readers.maps import script_entries

        self.script, self.table, self.name = script, table, name
        self.entries = script_entries(script, name)
        self.partner_var = table.partner_var
        self._cmds: dict[int, Cmd] = {}

    def cmd(self, pos: int) -> Cmd:
        from tools.rom.readers.maps import _decode

        if pos not in self._cmds:
            op, values, end = _decode(self.script, self.table, self.name, pos)
            const = self.table.constants[op]
            if const in BRANCH_COMMANDS:
                self._cmds[pos] = Cmd(const, tuple(values[:-1]), end + values[-1], end)
            else:
                self._cmds[pos] = Cmd(const, tuple(values), None, end)
        return self._cmds[pos]


def rom_narrowing_inputs(rom, symbols, decomp: Path, n_trainers: int):
    """(``sites``, ``scripts``, ``rematch_rows``, ``FieldRules``) for ``narrowed_terrains``
    from a ROM: each header's ``zone_event.narc`` member (bg events: script u16 at +0, 20
    bytes; objects: local id, graphics, movement type, trainer type, flag, script, facing
    u16s, ``data[0]`` at +14, x and z at +24, 32 bytes; coord events: script, x, z, width,
    length u16s, 16 bytes; tools/jsoncnv/event.py), its ``scr_seq.narc`` script file and
    init scripts, ``scripts_common`` (``SCRIPTS_ORDER``) and ``gVsSeekerRematchData``."""
    from tools.rom.readers.maps import (
        EVENT_RECORD_SIZES,
        EVENTS_NARC,
        SCRIPTS_NARC,
        _rematches,
        command_table,
        init_script_ids,
        map_header_rows,
        map_init_scripts,
    )

    decomp = Path(decomp)
    fr = field_rules(decomp)
    table = command_table(decomp)
    rows = map_header_rows(rom, symbols)
    inits = map_init_scripts(rom, symbols)
    events = rom.narc(EVENTS_NARC)
    narc = rom.narc(SCRIPTS_NARC)
    common_id = _order(decomp, SCRIPTS_ORDER).index(COMMON_FILE)
    common = RomScript(narc[common_id], table, f"{SCRIPTS_NARC} member {common_id}")
    files: dict[int, ScriptFacts] = {}
    sites = {}
    for h, (script_id, events_id, _) in enumerate(rows):
        member = events[events_id]
        pos, records = 0, []
        for size in EVENT_RECORD_SIZES:
            n = struct.unpack_from("<I", member, pos)[0]
            records.append(
                [member[pos + 4 + i * size : pos + 4 + (i + 1) * size] for i in range(n)]
            )
            pos += 4 + n * size
        bg, objs, _, coord = records
        objects = []
        other = {struct.unpack_from("<H", r, 0)[0] for r in bg}
        for r in objs:
            local, _, movement, ttype, _, script, direction = struct.unpack_from("<7H", r, 0)
            sight = struct.unpack_from("<H", r, 14)[0]
            x, z = struct.unpack_from("<2H", r, 24)
            objects.append(
                ObjectEvent(
                    local, movement, ttype, direction, sight, x, z, _battle_script_trainer(script)
                )
            )
            other.add(script)
        coords = tuple(CoordEvent(*struct.unpack_from("<5H", r, 0)) for r in coord)
        other |= set(init_script_ids(narc[inits[h]], inits[h]))
        if script_id not in files:
            script = RomScript(narc[script_id], table, f"{SCRIPTS_NARC} member {script_id}")
            files[script_id] = ScriptFacts(script, common, fr)
        sites[h] = HeaderSites(tuple(objects), coords, frozenset(other), script_id)
    return sites, files, _rematches(rom, symbols, n_trainers), fr
