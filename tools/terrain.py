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
* ``CalcTerrain`` (:467-494) reads the behaviour of the tile under the player
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
terrain is the union over its map headers (``tools.rom.readers.maps.trainer_maps``): one
value is the terrain, several are "mixed" and no header (no map places the trainer) is
"unknown"; both of the last two store ``TERRAIN_UNKNOWN``. The sight line and the tile next
to a talk-to trainer are not used to narrow the set (rules §9.3 step 2): that would only
turn some mixed trainers into known ones.
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
    ``INVALID_LAND_DATA_ID``; ``tiles_x``: ``MAP_TILES_COUNT_X`` (tiles per cell row)."""

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


def land_tiles(rules: TerrainRules, member: bytes, index: int) -> tuple[int, ...]:
    """Each tile's behaviour of a ``land_data.narc`` member, by tile index
    ``z * MAP_TILES_COUNT_X + x`` (``GetTileAttributes``, src/terrain_collision_manager.c:
    176-178), after ``land_behaviours``' checks."""
    offset, count = rules.attributes
    land_behaviours(rules, member, index)
    return tuple(a & rules.mask for a in struct.unpack_from(f"<{count}H", member, offset))


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
) -> list[tuple[int, list[int]]]:
    """Per trainer id: ``(terrain, possible terrains)``. The possible terrains are the union
    over the trainer's map headers ``where[id]`` (``by_header``); one value is the terrain,
    none or several give ``TERRAIN_UNKNOWN``. ``TERRAIN_UNKNOWN`` in the list means a header
    whose cells cannot be read."""
    out = []
    for tid in range(n_trainers):
        terrains = sorted(frozenset().union(*(by_header[h] for h in where.get(tid, ()))))
        known = len(terrains) == 1 and terrains[0] != TERRAIN_UNKNOWN
        out.append((terrains[0] if known else TERRAIN_UNKNOWN, terrains))
    return out


def terrain_fields(terrain: int, terrains: list[int]) -> dict:
    """The ``trainers.json`` fields: ``terrain`` (a ``TERRAIN_*`` id, ``TERRAIN_UNKNOWN``
    for mixed or unknown) and ``terrains`` (every terrain the trainer's maps can give,
    sorted; empty when no map places it)."""
    return {"terrain": terrain, "terrains": terrains}


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
