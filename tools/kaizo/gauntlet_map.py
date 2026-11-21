"""Read-only map read of the Kaizo gauntlet groups (design v5.3 section 2, E3-B3-2).

A Python port of the Node oracle in ``_scratch/e3/b3rules`` (``rom.js``, ``dump.js``, ``bfs2.js``,
``chain.js``, ``gym.js``, ``wayward2.js``, ``wayward_nomira.js``, ``wayward_talk.js``) on
``tools.rom.nds.RomFiles``. The build does not run it: its values are written to
``data/kaizo/overrides.toml`` with their ``rom:`` sources, and a ROM-backed test re-derives and
compares them (``tests/kaizo/test_gauntlet_map.py``).

Method (every rule cited to the decomp; paths relative to ``external/pokeplatinum``):

- Map header ``debug/cb_edit/d_test.narc``: u16 matrix @2, scripts @4, events @16, flags @22 (map
  type bits 0-6, escape bit 14; ``include/map_header.h:10-31``). Land data attributes at 0x10,
  32 x 32 u16, bit 15 collision, low byte tile behaviour (``include/constants/field/map.h:9``,
  ``:17``, ``:22``). Zone events: ``tools/jsoncnv/event.py:51-69`` (object), ``:84-91`` (warp),
  ``:107-118`` (coord). Trainer id = script - 3000 + 1 (``include/script_manager.h:93-95``).
- Sight (``src/trainer_encounter.c:166-181``, ``:182-229``, ``:240-306``, ``:310-344``): a NORMAL
  trainer (FACE_* and SPIN_* count as NORMAL) looks along the directions its movement type faces,
  a VIEW_ALL_DIRECTIONS trainer along all four; the player is spotted at distance <= range when
  no tile before the player's has terrain or object collision and the player's own tile holds only
  the object bit. Walking movement types (``generated/movement_types.txt``: wander 3-5, walk
  20-44) move the trainer and its sight line: the tool does not model them. It raises for a group
  member with one; a non-member with one is taken as beaten before the group (the player can fight
  it first, and a beaten trainer only frees tiles). Route 213's trainers 107 and 108 (types 41, 43)
  are such non-members; its facts are already the widest possible (every order, every gap a leave
  and heal gap), so the assumption cannot change them.
- ``FieldInput_Process`` runs ``FieldSystem_CheckForTrainersWantingBattle`` on every field-input
  frame (``src/overlay005/field_control.c:181-188``; ``src/trainer_encounter.c:84-100``): the sight
  check runs on every tile entered, on the arrival tile of every warp and on the start tile. A
  walk may not enter an undefeated trainer's sight tile, or arrive on one, except to start that
  fight.
- Warps: a collision tile with behaviour 0x69 ``TILE_BEHAVIOR_DOOR`` that holds a warp is entered
  by walking into it (``Field_CheckMapTransition``, ``src/overlay005/field_control.c:572-592``;
  ``include/constants/field/map_tile_behaviors.h:116``); the player never stands on it, so no
  sight check applies there. A directional entrance or stairs tile moves the player after a push
  in its direction (``:600-620``, ``:812-815``); a push is always possible once the player stands
  on the tile, so a warp tile is a step into the warp, as in the oracle. Intra-map warp panels
  are followed.
- Talking: the object in front of the player, or across a tile with behaviour 0x80
  ``TILE_BEHAVIOR_TABLE`` the object behind it (``sub_0203C9D4``, ``src/unk_0203C954.c:58-80``).
- Coord events: one whose var is 0 always matches (``sub_0203CC14``, ``src/unk_0203C954.c:218``;
  ``FieldSystem_TryGetVar``, ``src/script_manager.c:432-433``, ``:447-448``); another matches when
  the caller's ``vars`` say so. A matching coord on a tile walked in a ``strict`` header (the
  group's own) raises unless the caller's ``coord_rule`` models it; in another header, or when
  the caller does not know the var, it is recorded in ``Walk.crossed``.
- Removable: item balls (scripts 7000-7999), Rock Smash rocks (10001) and Strength boulders (10002;
  ``include/script_manager.h:112``) where Strength is usable (``FieldMoves_CheckStrength``,
  ``src/field_move_tasks.c:457-470``).
- The matrix is masked to the group's header: a tile outside is a wall.

``derive_groups`` returns per group the facts of ``data/kaizo/overrides.toml``.
"""

from __future__ import annotations

import re
import struct
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from tools.rom.nds import RomFiles
from tools.rom.readers.maps import (
    DOUBLE_BATTLE_SCRIPTS,
    EVENTS_NARC,
    HEADER_EVENTS,
    HEADER_FLAGS,
    HEADER_MATRIX,
    HEADER_SCRIPTS,
    KAIZO_MAP_HEADERS,
    LAND_DATA_NARC,
    MATRIX_NARC,
    SCRIPTS_NARC,
    SINGLE_BATTLE_SCRIPTS,
    VISIBLE_ITEM_SCRIPTS,
    _decode,
    command_table,
    script_entries,
)
from tools.terrain import TerrainError, parse_matrix

# --- tile and object constants ------------------------------------------------------------

ATTRIBUTES_OFFSET = 0x10  # TERRAIN_ATTRIBUTES_OFFSET, include/constants/field/map.h:17
ATTRIBUTES_SIZE = 0x800  # TERRAIN_ATTRIBUTES_SIZE, :18
TILES_PER_SIDE = 32  # MAP_TILES_COUNT_X / _Z, :9-10
COLLISION = 0x8000  # TERRAIN_ATTRIBUTES_COLLISION_MASK, :22
BEHAVIOUR_MASK = 0xFF  # TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK, :27
INVALID_LAND = 0xFFFF  # INVALID_LAND_DATA_ID, :14
WARP_ENTRANCE_SOUTH = 0x65  # TILE_BEHAVIOR_WARP_ENTRANCE_SOUTH, map_tile_behaviors.h:112
DOOR = 0x69  # TILE_BEHAVIOR_DOOR, include/constants/field/map_tile_behaviors.h:116
TABLE = 0x80  # TILE_BEHAVIOR_TABLE, :139
PC_TILE = 0x83  # TILE_BEHAVIOR_PC, :142
NO_WARP = 0xFFFF
ESCAPE_BIT = 14  # isEscapeRopeAllowed, include/map_header.h:28
MAP_TYPE_MASK = 0x7F  # mapType : 7
CAVE_MAP_TYPE = 3  # MapHeader_IsCave: rom header 199 / 284 map type 3 (facts-final)
ROCK_SMASH_SCRIPT = 10001  # field-move scripts start at 10000, include/script_manager.h:112
STRENGTH_SCRIPT = 10002

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # N S W E, MapObject_GetDxFromDir order
TRAINER_NORMAL = 1  # TRAINER_TYPE_NORMAL
TRAINER_ALL_DIRECTIONS = 2  # TRAINER_TYPE_VIEW_ALL_DIRECTIONS
TRAINER_FACE_FIRST, TRAINER_FACE_LAST = 4, 8  # FACE_SIDES .. SPIN_CLOCKWISE count as NORMAL
# Movement type -> the directions a trainer looks along (generated/movement_types.txt, 0-based;
# 0 NONE keeps the initial direction).
LOOK = {
    2: (0, 1, 2, 3),
    6: (0, 2),
    7: (0, 3),
    8: (1, 2),
    9: (1, 3),
    10: (0, 1, 2),
    11: (0, 1, 3),
    12: (0, 2, 3),
    13: (1, 2, 3),
    14: (0,),
    15: (1,),
    16: (2,),
    17: (3,),
    18: (0, 1, 2, 3),
    19: (0, 1, 2, 3),
    45: (0, 1),
    46: (2, 3),
}
MOVING = frozenset({3, 4, 5, *range(20, 45)})  # wander and walk types


class MapReadError(ValueError):
    """The ROM holds a map the read cannot model."""


class Pos(NamedTuple):
    header: int
    x: int
    z: int


@dataclass(frozen=True)
class MapObject:
    """One object event (``tools/jsoncnv/event.py:51-69``)."""

    index: int
    local_id: int
    movement: int
    trainer_type: int
    script: int
    direction: int
    sight: int
    x: int
    z: int

    @property
    def trainer(self) -> int | None:
        """The trainer id of a battle script (3000 + id - 1 single, 5000 + id - 1 double)."""
        if SINGLE_BATTLE_SCRIPTS <= self.script < VISIBLE_ITEM_SCRIPTS:
            base = DOUBLE_BATTLE_SCRIPTS if self.script >= DOUBLE_BATTLE_SCRIPTS else 3000
            return self.script - base + 1
        return None

    @property
    def removable(self) -> bool:
        """An item ball or a Rock Smash rock."""
        in_items = VISIBLE_ITEM_SCRIPTS <= self.script < VISIBLE_ITEM_SCRIPTS + 1000
        return in_items or self.script == ROCK_SMASH_SCRIPT

    @property
    def boulder(self) -> bool:
        return self.script == STRENGTH_SCRIPT


class Warp(NamedTuple):
    index: int
    x: int
    z: int
    dest: int
    dest_warp: int


class Coord(NamedTuple):
    script: int
    x: int
    z: int
    width: int
    length: int
    value: int
    var: int

    def covers(self, x: int, z: int) -> bool:
        return self.x <= x < self.x + self.width and self.z <= z < self.z + self.length


@dataclass
class Area:
    """The tiles and events of one map header (``tiles``: (x, z) -> terrain attribute u16)."""

    header: int
    tiles: Mapping[tuple[int, int], int]
    objects: tuple[MapObject, ...] = ()
    warps: tuple[Warp, ...] = ()
    coords: tuple[Coord, ...] = ()
    map_type: int = 0
    escape_bit: bool = False
    object_at: dict = field(init=False, repr=False, default_factory=dict)
    _pc: frozenset | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.object_at = {(o.x, o.z): o for o in self.objects}

    def warp_at(self, x: int, z: int) -> Warp | None:
        return next((w for w in self.warps if (w.x, w.z) == (x, z)), None)

    def pc_tiles(self) -> frozenset:
        """The tiles with behaviour 0x83 (PC)."""
        if self._pc is None:
            self._pc = frozenset(p for p, a in self.tiles.items() if a & BEHAVIOUR_MASK == PC_TILE)
        return self._pc


class StaticMaps:
    """Areas given directly (tests)."""

    def __init__(self, areas: Iterable[Area]):
        self._areas = {a.header: a for a in areas}

    def area(self, header: int) -> Area:
        try:
            return self._areas[header]
        except KeyError:
            raise MapReadError(f"no map header {header}") from None


# --- ROM reader -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MapHeader:
    header: int
    matrix: int
    scripts: int
    events: int
    map_type: int
    escape_bit: bool


def parse_events(
    member: bytes,
) -> tuple[tuple[MapObject, ...], tuple[Warp, ...], tuple[Coord, ...]]:
    """The object, warp and coord events of a ``zone_event.narc`` member (a u32 count then
    fixed-size records: bg 20, object 32, warp 12 and coord 16 bytes)."""
    pos = 0
    records: list[list[bytes]] = []
    for size in (20, 32, 12, 16):
        (n,) = struct.unpack_from("<I", member, pos)
        pos += 4
        records.append([member[pos + i * size : pos + (i + 1) * size] for i in range(n)])
        pos += n * size
    objects = []
    for i, r in enumerate(records[1]):
        v = struct.unpack_from("<8H", r)  # id gfx movement trainer-type flag script dir data0
        x, z = struct.unpack_from("<2H", r, 24)
        objects.append(MapObject(i, v[0], v[2], v[3], v[5], v[6], v[7], x, z))
    warps = [Warp(i, *struct.unpack_from("<4H", r)) for i, r in enumerate(records[2])]
    coords = []
    for r in records[3]:
        script, x, z, w, length, _y, value, var = struct.unpack_from("<8H", r)
        coords.append(Coord(script, x, z, w, length, value, var))
    return tuple(objects), tuple(warps), tuple(coords)


class RomMaps:
    """The map headers of a Kaizo ROM as ``Area`` objects (``dump.js``)."""

    def __init__(self, files: RomFiles):
        self._headers = files.narc(KAIZO_MAP_HEADERS)
        self._events = files.narc(EVENTS_NARC)
        self._matrices = files.narc(MATRIX_NARC)
        self._lands = files.narc(LAND_DATA_NARC)
        self._cache: dict[tuple[int, bool], Area] = {}
        self._land_cache: dict[int, tuple[int, ...]] = {}

    def header(self, header: int) -> MapHeader:
        try:
            b = self._headers[header]
            (matrix,) = struct.unpack_from("<H", b, HEADER_MATRIX)
            (scripts,) = struct.unpack_from("<H", b, HEADER_SCRIPTS)
            (events,) = struct.unpack_from("<H", b, HEADER_EVENTS)
            (flags,) = struct.unpack_from("<H", b, HEADER_FLAGS)
        except (IndexError, struct.error):
            raise MapReadError(f"no map header {header}") from None
        return MapHeader(
            header, matrix, scripts, events, flags & MAP_TYPE_MASK, bool((flags >> ESCAPE_BIT) & 1)
        )

    def _land(self, land: int) -> tuple[int, ...]:
        if land not in self._land_cache:
            member = self._lands[land]
            if struct.unpack_from("<I", member, 0)[0] != ATTRIBUTES_SIZE:
                raise MapReadError(f"land data {land}: terrain attribute size is not 0x800")
            self._land_cache[land] = struct.unpack_from(
                f"<{ATTRIBUTES_SIZE // 2}H", member, ATTRIBUTES_OFFSET
            )
        return self._land_cache[land]

    def area(self, header: int, masked: bool = True) -> Area:
        """The area of a header; ``masked`` keeps only the matrix cells the header labels
        (``dump.js`` ``grid(mx, h)``), else every cell (the Eterna Gym)."""
        key = (header, masked)
        if key in self._cache:
            return self._cache[key]
        h = self.header(header)
        try:
            matrix = parse_matrix(self._matrices[h.matrix], h.matrix)
            objects, warps, coords = parse_events(self._events[h.events])
            tiles: dict[tuple[int, int], int] = {}
            for k, land in enumerate(matrix.lands):
                if land == INVALID_LAND:
                    continue
                if masked and matrix.headers is not None and matrix.headers[k] != header:
                    continue
                x0 = (k % matrix.width) * TILES_PER_SIDE
                z0 = (k // matrix.width) * TILES_PER_SIDE
                for i, a in enumerate(self._land(land)):
                    tiles[(x0 + i % TILES_PER_SIDE, z0 + i // TILES_PER_SIDE)] = a
        except (IndexError, struct.error, TerrainError) as e:
            raise MapReadError(f"map header {header}: {e}") from e
        area = Area(header, tiles, objects, warps, coords, h.map_type, h.escape_bit)
        self._cache[key] = area
        return area


def escape(maps, header: int) -> bool:
    """Whether Escape Rope and Dig work on a header: a cave map type with the escape bit
    (``CanUseEscapeRope``, ``src/item_use_functions.c:941-948``; ``src/field_move_tasks.c:760``)."""
    a = maps.area(header)
    return a.map_type == CAVE_MAP_TYPE and a.escape_bit


# --- the walk ---------------------------------------------------------------------------------


class Sighting(NamedTuple):
    """Where the player stands when a trainer starts the fight: ``how`` is ``"sight"`` (a tile in
    the trainer's line), ``"arrival"`` (a warp arrival or the start tile) or ``"talk"``."""

    pos: Pos
    how: str


@dataclass
class Walk:
    """The result of one walk: the tiles reached (tile -> the tile before it), the undefeated
    trainers that can start a fight, the objects that can be talked to (``(header, local id)``
    -> the player's tile) and the coord events crossed."""

    seen: dict[Pos, Pos | None]
    spot: dict[int, Sighting] = field(default_factory=dict)
    talk: dict[tuple[int, int], Pos] = field(default_factory=dict)
    crossed: list[tuple[Pos, Coord]] = field(default_factory=list)

    def headers(self) -> set[int]:
        return {p.header for p in self.seen}


CoordRule = Callable[[int, Coord, "bool | None"], "str | None"]


class Walker:
    """The walk rules over ``maps`` (an object with ``area(header) -> Area``).

    ``members``: the trainer ids of the group (a member with a walking movement type raises).
    ``strength``: Strength boulders can be pushed away. ``ignore(header, obj)``: an object that
    never blocks and never sees (the partner who follows the player). ``vars``: the known values
    of coord variables. ``coord_rule(header, coord, matches)``: ``"pass"``, ``"block"`` (the
    script takes the player: the tile is not entered) or None for the default (a matching coord
    raises when its header is in ``strict``, every header when None; elsewhere it is recorded: the
    walk beyond a group's own headers only has to show where the player can go)."""

    def __init__(
        self,
        maps,
        members: Collection[int] = (),
        *,
        strength: bool = False,
        ignore: Callable[[int, MapObject], bool] | None = None,
        vars: Mapping[int, int] | None = None,
        coord_rule: CoordRule | None = None,
        strict: Collection[int] | None = None,
    ):
        self.maps = maps
        self.members = frozenset(members)
        self.strength = strength
        self.ignore = ignore
        self.vars = dict(vars or {})
        self.coord_rule = coord_rule
        self.strict = None if strict is None else frozenset(strict)
        self._sight: dict[tuple, dict[tuple[int, int], list[int]]] = {}
        self._walks: dict[tuple, Walk] = {}

    # -- terrain and objects

    def _blocked(self, area: Area, x: int, z: int) -> bool:
        a = area.tiles.get((x, z))
        return a is None or bool(a & COLLISION)

    def _obj_blocks(self, area: Area, x: int, z: int, removable: bool) -> bool:
        o = area.object_at.get((x, z))
        if o is None:
            return False
        if removable and o.removable:
            return False
        if self.strength and o.boulder:
            return False
        return not (self.ignore is not None and self.ignore(area.header, o))

    def _inert(self, obj: MapObject) -> bool:
        """A non-member trainer with a walking movement type: taken as beaten before the group."""
        t = obj.trainer
        if t is None or obj.movement not in MOVING:
            return False
        if t in self.members:
            raise MapReadError(
                f"trainer {t} (movement type {obj.movement}) walks: its sight line is not modelled"
            )
        return True

    # -- sight

    def sight_tiles(
        self, header: int, obj: MapObject, removable: bool = True
    ) -> list[tuple[int, int, int, int]]:
        """The tiles ``(x, z, direction, distance)`` where ``obj`` spots the player
        (``GetTrainerDistToPlayer`` :182-229, ``IsPathInterrupted`` :310-344)."""
        if obj.trainer is None:
            return []
        area = self.maps.area(header)
        tt = obj.trainer_type
        if tt == TRAINER_ALL_DIRECTIONS:
            dirs: tuple[int, ...] = (0, 1, 2, 3)
        elif tt == TRAINER_NORMAL or TRAINER_FACE_FIRST <= tt <= TRAINER_FACE_LAST:
            if self._inert(obj):
                return []
            dirs = LOOK.get(obj.movement) or (obj.direction,)
        else:
            return []
        out = []
        for d in dirs:
            dx, dz = DIRS[d]
            for k in range(1, obj.sight + 1):
                x, z = obj.x + dx * k, obj.z + dz * k
                if self._blocked(area, x, z) or self._obj_blocks(area, x, z, removable):
                    break
                out.append((x, z, d, k))
        return out

    def sight_at(
        self, header: int, x: int, z: int, defeated: frozenset[int], removable: bool = True
    ) -> list[int]:
        """The undefeated trainers that spot the player standing on ``(x, z)``."""
        key = (header, defeated, removable)
        table = self._sight.get(key)
        if table is None:
            table = {}
            area = self.maps.area(header)
            for o in area.objects:
                t = o.trainer
                if t is None or t in defeated:
                    continue
                if self.ignore is not None and self.ignore(header, o):
                    continue
                for sx, sz, _, _ in self.sight_tiles(header, o, removable):
                    table.setdefault((sx, sz), []).append(t)
            self._sight[key] = table
        return table.get((x, z), [])

    # -- coords

    def coord_matches(self, coord: Coord) -> bool | None:
        """Whether a coord event runs: var 0 always; else by the known ``vars`` (None: unknown)."""
        if coord.var == 0:
            return True
        if coord.var in self.vars:
            return self.vars[coord.var] == coord.value
        return None

    def _coord_step(self, pos: Pos, walk: Walk) -> bool:
        """Apply the coord events of a tile entered; False when a script takes the player."""
        area = self.maps.area(pos.header)
        for c in area.coords:
            if not c.covers(pos.x, pos.z):
                continue
            matches = self.coord_matches(c)
            action = self.coord_rule(pos.header, c, matches) if self.coord_rule else None
            if action is None:
                if matches and (self.strict is None or pos.header in self.strict):
                    raise MapReadError(
                        f"coord event script {c.script} at {pos} runs (var {c.var:#x}) and is "
                        "not modelled"
                    )
                action = "pass"
            if action == "block":
                return False
            walk.crossed.append((pos, c))
        return True

    # -- the walk

    def walk(self, start: Pos, defeated: Iterable[int] = (), removable: bool = True) -> Walk:
        """Breadth-first walk from ``start`` with ``defeated`` trainers beaten (``bfs2.js``
        ``run``): sight is checked on the start tile, every tile entered and every warp arrival."""
        defeated = frozenset(defeated)
        key = (start, defeated, removable)
        if key in self._walks:
            return self._walks[key]
        walk = Walk({start: None})
        self._walks[key] = walk
        arrival = self.sight_at(start.header, start.x, start.z, defeated, removable)
        if arrival:
            for t in arrival:
                walk.spot[t] = Sighting(start, "arrival")
            return walk
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            area = self.maps.area(pos.header)
            for dx, dz in DIRS:
                nx, nz = pos.x + dx, pos.z + dz
                attr = area.tiles.get((nx, nz))
                warp = area.warp_at(nx, nz)
                blocks = self._obj_blocks(area, nx, nz, removable)
                is_door = (
                    not blocks
                    and attr is not None
                    and attr & BEHAVIOUR_MASK == DOOR
                    and warp is not None
                )
                if not is_door and (self._blocked(area, nx, nz) or blocks):
                    self._talk(area, pos, dx, dz, defeated, walk)
                    continue
                here = Pos(pos.header, nx, nz)
                sighted = self.sight_at(pos.header, nx, nz, defeated, removable)
                if sighted:
                    for t in sighted:
                        walk.spot.setdefault(t, Sighting(here, "sight"))
                    continue
                target = here
                if warp is not None and warp.dest != NO_WARP:
                    landing = self._arrival(warp)
                    if landing is not None:
                        target = landing
                        sighted = self.sight_at(
                            target.header, target.x, target.z, defeated, removable
                        )
                        if sighted:
                            for t in sighted:
                                walk.spot.setdefault(t, Sighting(target, "arrival"))
                            continue
                if target in walk.seen:
                    continue
                if not self._coord_step(target, walk):
                    continue
                walk.seen[target] = pos
                queue.append(target)
                if warp is not None:
                    walk.seen[here] = pos
        return walk

    def _arrival(self, warp: Warp) -> Pos | None:
        """The tile a warp lands on, or None when its destination is not a readable map."""
        try:
            dest = self.maps.area(warp.dest)
        except MapReadError:
            return None
        if warp.dest_warp >= len(dest.warps):
            return None
        w = dest.warps[warp.dest_warp]
        return Pos(warp.dest, w.x, w.z)

    def _talk(
        self, area: Area, pos: Pos, dx: int, dz: int, defeated: frozenset[int], walk: Walk
    ) -> None:
        """The player on ``pos`` faces ``(dx, dz)``: the object in front, or across a table tile
        the one behind it, can be talked to."""
        tx, tz = pos.x + dx, pos.z + dz
        front = area.tiles.get((tx, tz))
        if front is not None and front & BEHAVIOUR_MASK == TABLE:
            tx, tz = tx + dx, tz + dz
        o = area.object_at.get((tx, tz))
        if o is None:
            return
        walk.talk.setdefault((area.header, o.local_id), pos)
        t = o.trainer
        if t is not None and t not in defeated and not self._inert(o):
            walk.spot.setdefault(t, Sighting(pos, "talk"))

    def talk_tiles(
        self,
        start: Pos,
        header: int,
        member: int,
        defeated: Iterable[int] = (),
        removable: bool = True,
    ) -> list[tuple[int, int]]:
        """The tiles, reached from ``start`` without entering any sight line, from which
        ``member`` can be talked to (next to it, or two tiles away across a table tile)."""
        area = self.maps.area(header)
        obj = next((o for o in area.objects if o.trainer == member), None)
        if obj is None:
            raise MapReadError(f"trainer {member} is not an object of header {header}")
        seen = self.walk(start, defeated, removable).seen
        out = []
        for dx, dz in DIRS:
            near = (obj.x + dx, obj.z + dz)
            tile = area.tiles.get(near)
            if tile is not None and tile & BEHAVIOUR_MASK == TABLE:
                far = (obj.x + 2 * dx, obj.z + 2 * dz)
                if Pos(header, *far) in seen:
                    out.append(far)
            elif Pos(header, *near) in seen:
                out.append(near)
        return sorted(out)


def pc_reachable(maps, walk: Walk, headers: Collection[int]) -> bool:
    """Whether a tile next to a PC tile (behaviour 0x83) of ``headers`` was reached."""
    for h in headers:
        for x, z in maps.area(h).pc_tiles():
            if any(Pos(h, x + dx, z + dz) in walk.seen for dx, dz in DIRS):
                return True
    return False


# --- chains (chain.js) --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    """A chain of undefeated trainers walked from ``entries``: ``members`` (every trainer of the
    chain), ``headers`` (the chain's own map headers; reaching another header is a leave)."""

    members: tuple[int, ...]
    headers: tuple[int, ...]
    entries: tuple[Pos, ...]


class Run(NamedTuple):
    """One full sequence of the chain and the walk after each prefix: ``states[k]`` is the walk
    from the tile of the k-th fight with the first k fights beaten (k = 0..n)."""

    sequence: tuple[int, ...]
    states: tuple[Walk, ...]


def runs(walker: Walker, spec: ChainSpec) -> list[Run]:
    """Every full sequence of ``spec.members`` the walk allows (``chain.js`` ``rec``)."""
    out: list[Run] = []
    members = frozenset(spec.members)

    def rec(pos: Pos, done: frozenset[int], seq: tuple[int, ...], states: tuple[Walk, ...]):
        w = walker.walk(pos, done)
        states = (*states, w)
        if done == members:
            out.append(Run(seq, states))
            return
        for t, s in list(w.spot.items()):
            if t in members and t not in done:
                rec(s.pos, done | {t}, (*seq, t), states)

    for e in spec.entries:
        rec(e, frozenset(), (), ())
    return out


def chain_facts(
    maps,
    walker: Walker,
    spec: ChainSpec,
    group: tuple[int, ...],
    *,
    healers: Collection[tuple[int, int]] = (),
    pc_headers: Collection[int] = (),
) -> dict:
    """The facts of the trainers ``group`` of a chain: the group's orders (each full sequence
    restricted to the group), per order the gaps where every run can leave the chain's headers,
    and the gaps where every run reaches a healer or a PC. ``orders``, ``leave_gaps`` (one list
    per order), ``heal_gaps`` and ``pc_gaps`` (one list)."""
    members = frozenset(group)
    leave: dict[tuple[int, ...], set[int]] = {}
    heal: set[int] | None = None
    pcs: set[int] | None = None
    for run in runs(walker, spec):
        idx = [i for i, t in enumerate(run.sequence) if t in members]
        order = tuple(run.sequence[i] for i in idx)
        gaps_leave: set[int] = set()
        gaps_heal: set[int] = set()
        gaps_pc: set[int] = set()
        for k, i in enumerate(idx[:-1], 1):
            state = run.states[i + 1]
            if state.headers() - set(spec.headers):
                gaps_leave.add(k)
            if any(key in state.talk for key in healers):
                gaps_heal.add(k)
            if pc_headers and pc_reachable(maps, state, pc_headers):
                gaps_pc.add(k)
        leave[order] = leave[order] & gaps_leave if order in leave else gaps_leave
        heal = gaps_heal if heal is None else heal & gaps_heal
        pcs = gaps_pc if pcs is None else pcs & gaps_pc
    if not leave:
        raise MapReadError(f"no full sequence of {group}")
    orders = sorted(leave)
    return {
        "orders": [[[t] for t in o] for o in orders],
        "leave_gaps": [sorted(leave[o]) for o in orders],
        "pc_gaps": sorted(pcs or ()),
        "heal_gaps": sorted(heal or ()),
    }


# --- scripts ------------------------------------------------------------------------------------


class ScriptReader:
    """Straight-line traces of a script entry (``scr.js``): the commands from the entry to the
    first End, Return or GoTo, branches not followed."""

    def __init__(self, files: RomFiles, decomp: Path):
        self._scripts = files.narc(SCRIPTS_NARC)
        self._table = command_table(Path(decomp))

    def trace(self, member: int, entry: int) -> list[tuple[int, str, list[int]]]:
        """``(offset, SCRCMD_* constant, fixed operands)`` of the 1-based ``entry`` of script
        file ``member``."""
        script = self._scripts[member]
        name = f"scr_seq {member}"
        pos = script_entries(script, name)[entry - 1]
        out = []
        while True:
            op, values, end = _decode(script, self._table, name, pos)
            const = self._table.constants[op]
            out.append((pos, const, values))
            if const in ("SCRCMD_END", "SCRCMD_RETURN", "SCRCMD_GOTO"):
                return out
            pos = end


def b2b_facts(maps: RomMaps, scripts: ScriptReader, header: int) -> dict:
    """Back-to-back battles of one script entry (Route 204 North B2B): the fights in script order
    and a locked gap wherever no ReleaseAll lies between two battles (the player has no field
    control and takes no steps)."""
    entry = {c.script for c in maps.area(header).coords}
    if len(entry) != 1:
        raise MapReadError(f"header {header}: no single coord script")
    trace = scripts.trace(maps.header(header).scripts, entry.pop())
    if not any(c == "SCRCMD_LOCKALL" for _, c, _ in trace):
        raise MapReadError(f"header {header}: the entry does not LockAll")
    fights: list[int] = []
    locked: list[int] = []
    released = False
    for _, const, values in trace:
        if const == "SCRCMD_STARTTRAINERBATTLE":
            if fights and not released:
                locked.append(len(fights))
            fights.append(values[0])
            released = False
        elif const == "SCRCMD_RELEASEALL":
            released = True
    return {"orders": [[[t] for t in fights]], "leave_gaps": [[]], "locked_gaps": locked}


# --- Eterna Gym (gym.js) ---------------------------------------------------------------------

GYM_HEADER = 67
GYM_GUIDE = 0  # object 0 at (13, 21): the first talk advances the clock (scr 73 entry 2)
CLOCK_STATES = 5  # ETERNA_CLOCK_MAX, include/overlay008/gym_features.h:17-22
CLOCK_ORIGIN = (5, 7)  # x 5-17, z 7-19 (sEternaGymClockCollision, gym_features.c:2333-2410)
CLOCK_SIDE = 13
FOUNTAIN_ROW = 19  # sEternaGymFountainCollision, :2412-2418 (x 1-21)
FOUNTAIN_X0, FOUNTAIN_WIDTH = 1, 21
# sEternaGymHourHandJumpTiles (gym_features.c:2308-2331): state -> (x, z, jump axis north-south)
JUMP_TILES = {
    1: (11, 18, True),
    2: (6, 13, False),
    3: (11, 8, True),
    4: (11, 8, True),
}
# The fights of scr 73: trainer id -> (coord script, object local id, clock steps on the win):
# entry 7 (324) from the coord (19, 13), entry 8 (259) from the coord (3, 13) or obj 3, entry 5
# (260) from obj 4; the clock advances after the first win and after 260 (entry 7 @559-586,
# entry 5 @361).
GYM_FIGHTS = {324: (7, None, 1), 259: (8, 3, 0), 260: (None, 4, 1)}


def eterna_clock_tables(decomp: Path) -> tuple[list[list[int]], list[list[int]]]:
    """The clock and fountain collision tables of ``src/overlay008/gym_features.c``."""
    src = (Path(decomp) / "src" / "overlay008" / "gym_features.c").read_text(encoding="utf-8")

    def blocks(name: str, size: int) -> list[list[int]]:
        text = src[src.index(f"{name}[ETERNA_CLOCK_MAX]") :]
        text = text[: text.index("};")]
        found = [
            [int(d) for d in re.findall(r"[01]", b)]
            for b in re.findall(r"\]\s*=\s*\{([^{}]*)\}", text)
        ]
        if len(found) != CLOCK_STATES or any(len(b) != size for b in found):
            raise MapReadError(f"{name}: expected {CLOCK_STATES} blocks of {size}")
        return found

    return (
        blocks("sEternaGymClockCollision", CLOCK_SIDE * CLOCK_SIDE),
        blocks("sEternaGymFountainCollision", FOUNTAIN_WIDTH),
    )


class GymReach(NamedTuple):
    seen: frozenset  # tiles (x, z)
    talk: dict  # object local id -> the player's tile
    coords: dict  # coord script -> its tile


def gym_reach(
    area: Area,
    tables: tuple[list[list[int]], list[list[int]]],
    state: int,
    start: tuple[int, int],
    passed: Collection[int] = (),
) -> GymReach:
    """The tiles, talkable objects and coord events reachable in clock ``state`` from ``start``
    (``gym.js`` ``reach``): the clock and fountain collision and the hour-hand jump tile.

    A coord tile runs its script when entered (var 0 always matches, ``sub_0203CC14``,
    ``src/unk_0203C954.c:218``): one whose script is not in ``passed`` starts its fight there,
    so it is recorded and the walk does not go on from it (``gym.js`` walked on, which let
    state 2 reach obj 4 past the coord (3, 13)). A script in ``passed`` (its fight won) is a
    message only: scr 73 entry 7 fights while 0x40AE <= 1 (@475-481), entry 8 while <= 2
    (@597-603), and each win raises 0x40AE past its gate (facts-final, Eterna Gym)."""
    clock, fountain = tables
    x0, z0 = CLOCK_ORIGIN

    def collide(x: int, z: int) -> bool:
        a = area.tiles.get((x, z))
        if a is None or a & COLLISION:
            return True
        in_clock = z0 <= z < z0 + CLOCK_SIDE and x0 <= x < x0 + CLOCK_SIDE
        if in_clock and clock[state][(z - z0) * CLOCK_SIDE + (x - x0)]:
            return True
        in_fountain = z == FOUNTAIN_ROW and FOUNTAIN_X0 <= x < FOUNTAIN_X0 + FOUNTAIN_WIDTH
        if in_fountain and fountain[state][x - FOUNTAIN_X0]:
            return True
        return (x, z) in area.object_at

    seen = {start}
    queue = deque([start])
    talk: dict[int, tuple[int, int]] = {}
    coords: dict[int, tuple[int, int]] = {}
    jump = JUMP_TILES.get(state)
    while queue:
        x, z = queue.popleft()
        for d, (dx, dz) in enumerate(DIRS):
            nx, nz = x + dx, z + dz
            if jump and (jump[0], jump[1]) == (nx, nz) and (d < 2) == jump[2]:
                nx, nz = nx + dx, nz + dz
                if collide(nx, nz):
                    continue
            elif collide(nx, nz):
                o = area.object_at.get((nx, nz))
                if o is not None:
                    talk.setdefault(o.local_id, (x, z))
                continue
            if (nx, nz) in seen:
                continue
            seen.add((nx, nz))
            fights = [c.script for c in area.coords if c.covers(nx, nz) and c.script not in passed]
            for script in fights:
                coords.setdefault(script, (nx, nz))
            if not fights:
                queue.append((nx, nz))
    return GymReach(frozenset(seen), talk, coords)


def gym_facts(maps: RomMaps, decomp: Path) -> dict:
    """The Eterna Gym trainers (E3-B3-2.c): the orders the clock states allow and the gaps where
    the door can be reached. State 0 from the door offers only the guide; each fight changes the
    reach as the clock advances (after the first win over 324 and after 260)."""
    area = maps.area(GYM_HEADER, masked=False)
    tables = eterna_clock_tables(decomp)
    # The way in and out: the south warp entrance (11, 27), rom header 67 warp 0 -> header 65,
    # tile behaviour 0x65. The player arrives on it and leaves by a push south from it
    # (``Field_CheckMapTransition``, ``src/overlay005/field_control.c:609-612``), so it is both
    # the start and the tile a leave needs.
    door = next(
        (
            w
            for w in area.warps
            if area.tiles.get((w.x, w.z), 0) & BEHAVIOUR_MASK == WARP_ENTRANCE_SOUTH
        ),
        None,
    )
    if door is None:
        raise MapReadError("the gym has no south warp entrance")
    start = front = (door.x, door.z)
    r0 = gym_reach(area, tables, 0, start)
    fight_objects = {obj for _, obj, _ in GYM_FIGHTS.values() if obj is not None}
    if r0.coords or fight_objects & set(r0.talk):
        raise MapReadError("a fight can start before the guide's first talk")
    if GYM_GUIDE not in r0.talk:
        raise MapReadError("the guide cannot be reached from the door")
    orders: dict[tuple[int, ...], list[bool]] = {}

    def rec(clock: int, pos: tuple[int, int], done: tuple[int, ...], doors: list[bool]):
        if len(done) == len(GYM_FIGHTS):
            orders[done] = doors
            return
        won = {s for f, (s, _, _) in GYM_FIGHTS.items() if f in done and s is not None}
        reach = gym_reach(area, tables, clock, pos, won)
        for fight, (script, obj, step) in GYM_FIGHTS.items():
            if fight in done:
                continue
            if script is not None and script in reach.coords:
                where = reach.coords[script]
            elif obj is not None and obj in reach.talk:
                where = reach.talk[obj]
            else:
                continue
            won = {
                s for f, (s, _, _) in GYM_FIGHTS.items() if f in (*done, fight) and s is not None
            }
            after = gym_reach(area, tables, clock + step, where, won)
            rec(clock + step, where, (*done, fight), [*doors, front in after.seen])

    rec(1, start, (), [])
    if not orders:
        raise MapReadError("no order of the gym trainers")
    keys = sorted(orders)
    return {
        "orders": [[[t] for t in o] for o in keys],
        "leave_gaps": [[k for k, d in enumerate(orders[o][:-1], 1) if d] for o in keys],
    }


# --- Wayward Cave (wayward2.js, wayward_nomira.js, wayward_talk.js) ----------------------------

WAYWARD_HEADER = 284
WAYWARD_START = Pos(284, 84, 9)  # the arrival of scr 318 entry 5's warp (@232)
WAYWARD_PARTNER_VAR = 0x4091  # set to 1 by talking to the partner (scr 318 entry 2 @152)
WAYWARD_PARTNER_ID_VAR = 0x403F  # the partner's trainer id (@165)
WAYWARD_MIRA = 4  # object local id of the partner at (72, 13)
WAYWARD_PAIRS = ((383, 443), (380, 458), (450, 496), (459, 460), (454, 455))
WAYWARD_EXIT_ROW = 4  # coord script of row 47 (0x4091 == 1): scr 318 entry 4, the fight 749+750
WAYWARD_BACK_ROW = 5  # coord script of row 48 (0x4091 == 0): scr 318 entry 5, warps back to start
# coord script of (85,8)-(85,10) (0x4091 == 0): scr 318 entry 6 @277 asks "Ignore the cry for
# help?"; yes @385-431 Warp 350 (310,608) (the walk-out), no @299 walks the player 2 west.
WAYWARD_WALKOUT = 6
# Every coord script of the header takes the player: a fight, a warp back, or the walk-out.
WAYWARD_TAKES = (WAYWARD_EXIT_ROW, WAYWARD_BACK_ROW, WAYWARD_WALKOUT)
WAYWARD_LAST = (749, 750)


def _wayward_walker(maps, joined: bool) -> Walker:
    """The walk with the partner following (she neither blocks nor sees; 0x4091 = 1) or still
    standing at (72, 13) as an obstacle (0x4091 = 0). Each coord runs by its var condition, as
    the field does (``coord_matches``): a running coord of ``WAYWARD_TAKES`` takes the player,
    so its tiles are not walked through; a coord whose var does not match is walked over; any
    other running coord raises (not modelled)."""

    def ignore(header: int, obj: MapObject) -> bool:
        return joined and header == WAYWARD_HEADER and obj.local_id == WAYWARD_MIRA

    def rule(header: int, coord: Coord, matches: bool | None) -> str | None:
        if header != WAYWARD_HEADER:
            return None
        if matches is None:
            raise MapReadError(f"Wayward Cave: coord {coord.script} reads var {coord.var:#x}")
        if not matches:
            return "pass"
        return "block" if coord.script in WAYWARD_TAKES else None

    return Walker(
        maps,
        members={t for pair in WAYWARD_PAIRS for t in pair},
        strength=True,
        ignore=ignore,
        vars={WAYWARD_PARTNER_VAR: 1 if joined else 0},
        strict={WAYWARD_HEADER},
        coord_rule=rule,
    )


def _wayward_next(walker: Walker, pos: Pos, done: frozenset[int]) -> tuple[Walk, dict]:
    """The walk and the pairs it can start: pair -> the first sighting (``wayward2.js`` step)."""
    w = walker.walk(pos, done)
    nxt: dict[tuple[int, int], Sighting] = {}
    for t, s in w.spot.items():
        pair = next((p for p in WAYWARD_PAIRS if t in p), None)
        if pair is not None and pair not in nxt:
            nxt[pair] = s
    return w, nxt


def wayward_facts(maps: RomMaps, scripts: ScriptReader) -> dict:
    """Wayward Cave: the fight lists, the partner's join points and the split of 459 + 460."""
    area = maps.area(WAYWARD_HEADER)
    mira = next((o for o in area.objects if o.local_id == WAYWARD_MIRA), None)
    if mira is None:
        raise MapReadError("Wayward Cave has no partner object")
    joined = _wayward_walker(maps, True)
    alone = _wayward_walker(maps, False)
    rows = [c for c in area.coords if c.script == WAYWARD_EXIT_ROW]
    if len(rows) != 1:
        raise MapReadError("Wayward Cave: expected one exit coord row")
    row = rows[0]
    exits = {(x, row.z - 1) for x in range(row.x, row.x + row.width)}

    sequences: list[tuple] = []

    def with_partner(pos: Pos, done: frozenset[int], seq: tuple):
        w, nxt = _wayward_next(joined, pos, done)
        if any(Pos(WAYWARD_HEADER, x, z) in w.seen for x, z in exits):
            sequences.append(seq)
        for pair, s in nxt.items():
            with_partner(s.pos, done | set(pair), (*seq, pair))

    with_partner(WAYWARD_START, frozenset(), ())
    if len(sequences) != 1 or sorted(sequences[0]) != sorted(WAYWARD_PAIRS):
        raise MapReadError(f"Wayward Cave: the exit is reachable after {sequences}")
    order = sequences[0]

    reached: dict[int, bool] = {}

    def without_partner(pos: Pos, done: frozenset[int], seq: tuple):
        w, nxt = _wayward_next(alone, pos, done)
        near = [(mira.x + dx, mira.z + dz) for dx, dz in DIRS]
        ok = any(Pos(WAYWARD_HEADER, x, z) in w.seen for x, z in near)
        reached[len(seq) + 1] = reached.get(len(seq) + 1, True) and ok
        for pair, s in nxt.items():
            without_partner(s.pos, done | set(pair), (*seq, pair))

    without_partner(WAYWARD_START, frozenset(), ())
    # Without the partner the walk-out coord (0x4091 == 0) closes the start pocket: no pair is
    # reached and her tile is, so the game makes her join before fight 1 and the walk-out
    # never runs at a gap. A fight reached without her would make the join point a choice.
    join_before = sorted(k for k, ok in reached.items() if ok)
    if join_before != [1]:
        raise MapReadError(f"Wayward Cave: the partner's join points are {reached}, not fight 1")

    # the split: a pair member with a talk tile outside both sight lines is fought alone
    orders = [[list(p) for p in order]]
    pos, done = WAYWARD_START, frozenset()
    for i, pair in enumerate(order):
        _, nxt = _wayward_next(joined, pos, done)
        for a, b in (pair, pair[::-1]):
            tiles = joined.talk_tiles(pos, WAYWARD_HEADER, a, done)
            if not tiles:
                continue
            tx, tz = tiles[0]
            if b not in joined.walk(Pos(WAYWARD_HEADER, tx, tz), done | {a}).spot:
                raise MapReadError(f"Wayward Cave: {b} is not reached after {a} alone")
            before = [list(p) for p in order[:i]]
            after = [list(p) for p in order[i + 1 :]]
            orders.append([*before, [a], [b], *after])
        pos, done = nxt[pair].pos, done | set(pair)
    orders = [[*o, list(WAYWARD_LAST)] for o in orders]

    trace = scripts.trace(maps.header(WAYWARD_HEADER).scripts, 2)
    partner = [
        v for _, c, v in trace if c == "SCRCMD_SETVARFROMVALUE" and v[0] == WAYWARD_PARTNER_ID_VAR
    ]
    if len(partner) != 1:
        raise MapReadError("Wayward Cave: the partner's trainer id is not set once")
    return {
        "orders": orders,
        "leave_gaps": [[] for _ in orders],
        "partner": {"trainer": partner[0][1], "join_before": join_before},
    }


# --- Elite Four and Champion ---------------------------------------------------------------------

LEAGUE_SCRIPTS = (188, 190, 192, 194, 196)  # the five rooms' script files (rom: scr 188/.../196)
LEAGUE_TRAINERS = (866, 867, 868, 869, 870)
LEAGUE_HEADERS = tuple(range(175, 187))


def league_facts(maps: RomMaps) -> dict:
    """The five rooms in script order; the PC gaps are the lifts after rooms 1..n-1 (warp 0 of a
    room leads to its lift) whose PC tile can be reached from the lift's arrival."""
    rooms = []
    for s in LEAGUE_SCRIPTS:
        found = [h for h in LEAGUE_HEADERS if maps.header(h).scripts == s]
        if len(found) != 1:
            raise MapReadError(f"script file {s} belongs to headers {found}")
        rooms.append(found[0])
    walker = Walker(maps, strict=())
    pc_gaps = []
    for k, room in enumerate(rooms[:-1], 1):
        exit_warp = maps.area(room).warps[0]
        arrival = walker._arrival(exit_warp)
        if arrival is None:
            raise MapReadError(f"room {room}: no lift arrival")
        if pc_reachable(maps, walker.walk(arrival, ()), [exit_warp.dest]):
            pc_gaps.append(k)
    return {
        "orders": [[[t] for t in LEAGUE_TRAINERS]],
        "leave_gaps": [[]],
        "pc_gaps": pc_gaps,
        "escape_header": rooms[0],
    }


# --- the groups ----------------------------------------------------------------------------------

BUILDINGS = ChainSpec(
    members=(421, 831, 410, 423, 412, 422),
    headers=(72, 73, 74, 75),
    entries=(Pos(72, 11, 14),),
)
ROUTE_213 = ChainSpec(
    members=(106, 103, 105, 111, 110),
    headers=(373,),
    entries=(Pos(373, 706, 819), Pos(373, 706, 814), Pos(373, 647, 812)),
)
OREBURGH = ChainSpec(members=(195, 196), headers=(199,), entries=(Pos(199, 15, 1),))
ROUTE_204 = ChainSpec(members=(15, 13, 14), headers=(346,), entries=(Pos(346, 180, 699),))
# The lobby healer of Route 213: header 376 object 1 at (11, 4), talked to across the table tile
# (11, 5); scr 462 entry 2 runs HealParty (@74).
ROUTE_213_HEALERS = ((376, 1),)
B2B_HEADER = 256

ESCAPE_HEADERS = {
    "Oreburgh Mine": 199,
    "Route 204 North": 346,
    "Route 204 North B2B": B2B_HEADER,
    "Eterna Gym gym trainers": GYM_HEADER,
    "Team Galactic Eterna Building #1": 73,
    "Team Galactic Eterna Building #2": 74,
    "Team Galactic Eterna Building #3": 74,
    "Route 213": 373,
    "Wayward Cave": WAYWARD_HEADER,
}


def _facts(orders, leave_gaps, escape_, **more) -> dict:
    return {
        "orders": orders,
        "leave_gaps": leave_gaps,
        "escape": escape_,
        "pc_gaps": [],
        "heal_gaps": [],
        "locked_gaps": [],
        "partner": None,
        **more,
    }


def derive_groups(files: RomFiles, decomp: Path) -> dict[str, dict]:
    """The facts of every gauntlet group, read from the ROM: ``orders`` (a list of orders, each
    a list of fights, each a list of trainer ids), ``leave_gaps`` (one list per order),
    ``escape``, ``pc_gaps``, ``heal_gaps``, ``locked_gaps`` and ``partner`` (None, or
    ``{"trainer", "join_before"}``)."""
    maps = RomMaps(files)
    scripts = ScriptReader(files, decomp)
    out: dict[str, dict] = {}

    def chain(name: str, spec: ChainSpec, group: tuple[int, ...], **kw) -> None:
        facts = chain_facts(
            maps, Walker(maps, spec.members, strict=spec.headers), spec, group, **kw
        )
        out[name] = _facts(
            facts["orders"],
            facts["leave_gaps"],
            escape(maps, ESCAPE_HEADERS[name]),
            pc_gaps=facts["pc_gaps"],
            heal_gaps=facts["heal_gaps"],
        )

    chain("Oreburgh Mine", OREBURGH, OREBURGH.members)
    chain("Route 204 North", ROUTE_204, ROUTE_204.members)
    for n, group in enumerate(((421, 831), (410, 423), (412, 422)), 1):
        chain(f"Team Galactic Eterna Building #{n}", BUILDINGS, group)
    chain("Route 213", ROUTE_213, ROUTE_213.members, healers=ROUTE_213_HEALERS)
    for h, local in ROUTE_213_HEALERS:
        obj = next(o for o in maps.area(h).objects if o.local_id == local)
        trace = scripts.trace(maps.header(h).scripts, obj.script)
        if not any(c == "SCRCMD_HEALPARTY" for _, c, _ in trace):
            raise MapReadError(f"header {h} object {local} does not heal the party")

    b2b = b2b_facts(maps, scripts, B2B_HEADER)
    out["Route 204 North B2B"] = _facts(
        b2b["orders"], b2b["leave_gaps"], escape(maps, B2B_HEADER), locked_gaps=b2b["locked_gaps"]
    )
    gym = gym_facts(maps, decomp)
    out["Eterna Gym gym trainers"] = _facts(
        gym["orders"], gym["leave_gaps"], escape(maps, GYM_HEADER)
    )
    way = wayward_facts(maps, scripts)
    out["Wayward Cave"] = _facts(
        way["orders"], way["leave_gaps"], escape(maps, WAYWARD_HEADER), partner=way["partner"]
    )
    league = league_facts(maps)
    out["Elite Four + Champion"] = _facts(
        league["orders"],
        league["leave_gaps"],
        escape(maps, league["escape_header"]),
        pc_gaps=league["pc_gaps"],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m tools.kaizo.gauntlet_map [rom] [decomp]``: print the derived facts as JSON."""
    import json
    import sys

    from tools.rom.inputs import REPO_ROOT

    argv = sys.argv[1:] if argv is None else argv
    roms = sorted((REPO_ROOT / ".cache" / "rom").glob("kaizo-*.nds"))
    rom = Path(argv[0]) if argv else (roms[0] if roms else None)
    decomp = Path(argv[1]) if len(argv) > 1 else REPO_ROOT / "external" / "pokeplatinum"
    if rom is None or not rom.exists():
        print("no Kaizo ROM (.cache/rom/kaizo-*.nds)", file=sys.stderr)
        return 2
    print(json.dumps(derive_groups(RomFiles.open(rom.read_bytes()), decomp), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
