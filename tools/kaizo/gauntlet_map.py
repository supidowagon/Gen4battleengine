"""Read-only map read of the Kaizo gauntlet groups (design v5.3 section 2, E3-B3-2).

A Python port of the Node oracle in ``_scratch/e3/b3rules`` (``rom.js``, ``dump.js``, ``bfs2.js``,
``chain.js``, ``gym.js``, ``wayward2.js``, ``wayward_nomira.js``, ``wayward_talk.js``) on
``tools.rom.nds.RomFiles``, extended by the walk-fix round (E3-B3 review of 22 Nov). The build
does not run it: its values are written to ``data/kaizo/overrides.toml`` with their ``rom:``
sources, and a ROM-backed test re-derives and compares them (``tests/kaizo/test_gauntlet_map.py``).

Method (every rule cited to the decomp; paths relative to ``external/pokeplatinum``):

- Map header ``debug/cb_edit/d_test.narc``: u16 matrix @2, scripts @4, events @16, flags @22 (map
  type bits 0-6, escape bit 14; ``include/map_header.h:10-31``). Land data: four u32 sizes
  (attributes, props, model, BDHC; ``LandDataHeader_Load``, ``src/overlay005/land_data.c:457-478``),
  attributes at 0x10, 32 x 32 u16, bit 15 collision, low byte tile behaviour
  (``include/constants/field/map.h:9``, ``:17``, ``:22``). Zone events:
  ``tools/jsoncnv/event.py:51-69`` (object, its fx32 y at byte 28), ``:84-91`` (warp),
  ``:107-118`` (coord). Trainer id = script - 3000 + 1 (``include/script_manager.h:93-95``).
- Height (BDHC, ``src/overlay005/bdhc.c``): a tile's height is the plate through the tile centre
  (``CalculateObjectHeight`` :152-235: the strip by binary search on z, its access list in order,
  at most 10 plates whose box holds the point, the one nearest the current height). A step whose
  new height differs from the current by 20 or more is a collision (``GetVerticalDirection``,
  ``src/terrain_collision_manager.c:249-276``, ``:278-305``); with no plate the height reads 0 for
  that test and the object keeps its height (``GetHeight`` :47-136,
  ``MapObject_RecalculatePositionHeight``, ``src/map_object_move.c:1036-1047``). The player starts
  a map at height 0 (``PlayerAvatar_New`` takes x and z only, ``src/player_avatar.c:39-57``), an
  object at its event y (``src/map_object.c:709-716``), both then on the nearest plate.
- Movement (``PlayerAvatar_CheckCollision``, ``src/player_move.c:1865-1907``; walking,
  ``PlayerAvatar_SetMovement_NormalOverworld`` :1016-1081): a step is blocked by terrain collision
  or height (above), by a directional tile (``sub_02064004``, ``src/map_object_move.c:706-737``:
  ``TileBehavior_BlocksMovement*`` of the current tile in the step direction, of the target in the
  opposite one), by an object within 2 height units (``sub_02063F00`` :616-672), by a water tile
  (``PLAYER_COLLISION_WATER``; ``MapObject_IsOnWater`` :739-750) and, on an elevated bridge, by a
  bike bridge (:2116-2160). A jump tile in the step direction jumps two tiles
  (``PlayerAvatar_WillJump`` :1956-1999, ``MOVEMENT_ACTION_JUMP_FAR``): one-way. Surfing, a step is
  allowed with no collision or only water (:1058). Surf starts facing water with the Fen Badge and
  a party Surf user, not on an elevated bridge, never with a partner (``Field_TileBehaviorToScript``,
  ``src/overlay005/field_control.c:686-694``; ``PlayerAvatar_CanUseSurf``,
  ``src/overlay005/ov5_021DFB54.c:641-655``; ``res/field/scripts/scripts_field_moves.s:299-303``).
  Rock Climb tiles are climbed along their axis with the Icicle Badge, no partner
  (``field_control.c:682-684``; ``scripts_field_moves.s:246-288``), a waterfall with the Beacon
  Badge (``src/field_move_tasks.c:584-598``). Spin tiles (``SLIDE_*``) carry the player tile by
  tile, turning on each spin tile, until a step collides, with no field input on the way
  (``ov5_021E1154``, ``src/overlay005/ov5_021E1154.c:35-152``, called from ``Field_ProcessStep``,
  ``field_control.c:712``); ice slides on in the step direction until a step collides
  (``PlayerAvatar_TileMove_Ice``, ``player_move.c:544-604``). The elevated-bridge state turns on
  at a bridge start and off off a bridge (``MapObject_DetermineElevatedBridgeStatus``,
  ``map_object_move.c:562-569``). Walking only: cycling is not part of the walk, and no gauntlet
  header holds a bike tile (ROM test).
- Field-move objects (``res/field/scripts/scripts_field_moves.s``): a Cut tree (script 10000)
  needs the Forest Badge (:29), a Rock Smash rock (10001) the Coal Badge (:134;
  ``FieldMoves_CheckRockSmash``, ``src/field_move_tasks.c:545``), a Strength boulder (10002) the
  Mine Badge (:191; ``src/field_move_tasks.c:463``); without it the object blocks. The badges held
  at a group are those of the splits before the group's split (``FieldMoves.for_split``).
- Sight (``src/trainer_encounter.c:166-181``, ``:182-229``, ``:240-306``, ``:310-344``): a NORMAL
  trainer (FACE_* and SPIN_* count as NORMAL) looks along its current facing, a
  VIEW_ALL_DIRECTIONS trainer along all four. The player is spotted at distance <= range when no
  tile before the player's collides for the trainer (terrain and height from the trainer's own
  height, a directional tile, an object at its height) and the player's own tile holds only the
  player within 2 height units (``IsPathInterrupted`` keeps only the object bit). A LOOK_* or
  ROTATE_* movement type turns the trainer: every 16 to 64 frames it faces a random direction of its
  type's set (``sub_020645C0``, ``src/unk_0206450C.c:221-243``, timers ``Unk_020EEA88`` :1482-1488);
  only a running player in its range makes it face him (``sub_0206537C`` :1271-1335). So whether a
  turning trainer faces the player when he crosses its line is the player's choice of timing: the
  walk takes both, the fight and the pass. A fixed facing (one direction) is always forced.
  Walking movement types (``generated/movement_types.txt``: wander 3-5, walk 20-44) move the trainer
  and its sight line: the tool does not model them. It raises for a group member with one; a
  non-member with one is taken as beaten before the group (the player can fight it first, and a
  beaten trainer only frees tiles). Route 213's trainers 107 and 108 (types 41, 43) are such
  non-members.
- Who fights (``FieldSystem_CheckForTrainersWantingBattle``, ``src/trainer_encounter.c:84-131``):
  the first undefeated trainer in object order that sees the player
  (``FindUndefeatedTrainerInSight`` :133-155, ``MapObjectMan_FindObjectWithStatus``,
  ``src/map_object.c:857-881``). A double-battle trainer (trainer data battle type,
  ``Script_IsTrainerDoubleBattle``, ``src/script_manager.c:512-515``) fights with its partner object
  when the party has two live Pokemon, and does not fight at all otherwise (:113-118). A singles
  trainer fights together with the next undefeated trainer in object order that also sees the
  player, as one double battle (``APPROACH_TYPE_VS2``), when the party has two live Pokemon or a
  partner (``src/overlay005/field_control.c:181-186``), else alone (:98-106). The walk takes the
  party's count (``Walker.two_alive``) and the partner (``Walker.partner``); talking to a
  double-battle trainer reads the party only (``ScrCmd_CheckHasTwoAliveMons``,
  ``src/scrcmd_trainer.c:283-289``, from ``res/field/scripts/scripts_battles.s:21-22``). The
  next-fight table by the count (``live_table``) switches the count at every fight end and, inside
  a gap, where a revive, a PC or a healer changes it.
- ``FieldInput_Process`` runs the sight check on every field-input frame (``field_control.c:
  181-196``): on every tile entered, on the arrival tile of every warp and on the start tile, not on
  the tiles a spin crosses. A walk may not enter a tile where a fixed-facing undefeated trainer
  spots the player, or arrive on one, except to start that fight.
- Warps: a collision tile with behaviour 0x69 ``TILE_BEHAVIOR_DOOR`` that holds a warp is entered
  by walking into it (``Field_CheckMapTransition``, ``field_control.c:572-592``;
  ``include/constants/field/map_tile_behaviors.h:116``); the player never stands on it, so no
  sight check applies there. A directional entrance or stairs tile moves the player after a push
  in its direction (``:600-620``, ``:812-815``); a push is always possible once the player stands
  on the tile, so a warp tile is a step into the warp, as in the oracle. Intra-map warp panels
  are followed.
- Talking: the object in front of the player, or across a tile with behaviour 0x80
  ``TILE_BEHAVIOR_TABLE`` the object behind it, at the player's height unit
  (``sub_0203C9D4``, ``sub_0203C9B0``, ``src/unk_0203C954.c:47-80``).
- Coord events: one runs when ``FieldSystem_TryGetVar(var) == value`` (``sub_0203CC14``,
  ``src/unk_0203C954.c:208-223``). A var below ``VARS_START`` (0x4000) reads as the number itself
  (``FieldSystem_TryGetVar``, ``src/script_manager.c:428-452``), so a coord on var 0 runs only when
  its value is 0. Another var matches when the caller's ``vars`` say so. Only the first matching
  coord of a tile, in event order, runs (:217-221). A matching coord on a tile
  walked in a ``strict`` header (the group's own, and the Route 213 healer's) raises unless the
  caller's ``coord_rule`` models it; one whose var the caller does not know is passed and
  recorded in ``Walk.crossed``. In another header a coord that matches or whose var is not known
  blocks: its script may take the player, so no heal, PC or leave beyond it is granted (fewer
  player paths only).
- Dynamic collision and height plates (``DynamicMapFeatures_CheckCollision``,
  ``src/dynamic_map_features.c:88-103``, from ``terrain_collision_manager.c:316``; the plates of
  ``src/overlay008/gym_features.c:656``, ``src/platform_lift.c:200``) act only while the persisted
  feature id is set. It is reset on every warp (``src/field_map_change.c:268-269``, ``:325-326``)
  and set only by ten script commands (``src/scrcmd.c:4710-4782``, ``:5325``, ``:5699``). The
  land height offset is non-zero only in the Distortion World (``src/overlay005/fieldmap.c:782-788``,
  ``land_data.c:2492-2498``); matrix altitudes move only the rendering (``land_data.c:1555-1575``),
  not ``GetHeight`` (``terrain_collision_manager.c:47-125``). The walk models neither; the ROM test
  requires that no header a group's walk reaches runs one of the ten, except the Eterna Gym's
  own (``gym_facts``).
- Item balls (scripts 7000-7999) can be picked up; the matrix is masked to the group's header: a
  tile outside is a wall.

``derive_groups`` returns per group the facts of ``data/kaizo/overrides.toml``.
"""

from __future__ import annotations

import re
import struct
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
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
    NO_SCRIPT,
    NOT_CONSTANT,
    SCRIPTS_NARC,
    SINGLE_BATTLE_SCRIPTS,
    VISIBLE_ITEM_SCRIPTS,
    _decode,
    _walk,
    command_table,
    script_entries,
)
from tools.rom.readers.trainers import TRDATA_NARC
from tools.terrain import TerrainError, parse_matrix

# --- tile and object constants ------------------------------------------------------------

ATTRIBUTES_OFFSET = 0x10  # TERRAIN_ATTRIBUTES_OFFSET, include/constants/field/map.h:17
ATTRIBUTES_SIZE = 0x800  # TERRAIN_ATTRIBUTES_SIZE, :18
TILES_PER_SIDE = 32  # MAP_TILES_COUNT_X / _Z, :9-10
COLLISION = 0x8000  # TERRAIN_ATTRIBUTES_COLLISION_MASK, :22
BEHAVIOUR_MASK = 0xFF  # TERRAIN_ATTRIBUTES_TILE_BEHAVIOR_MASK, :27
INVALID_LAND = 0xFFFF  # INVALID_LAND_DATA_ID, :14
# Tile behaviours (enum TileBehavior, include/constants/field/map_tile_behaviors.h:10-267).
ICE = 0x20
BLOCK_E, BLOCK_W, BLOCK_N, BLOCK_S = 0x30, 0x31, 0x32, 0x33
BLOCK_NE, BLOCK_NW, BLOCK_SE, BLOCK_SW = 0x34, 0x35, 0x36, 0x37
JUMP_E, JUMP_W, JUMP_N, JUMP_S = 0x38, 0x39, 0x3A, 0x3B
SLIDE_E, SLIDE_W, SLIDE_N, SLIDE_S = 0x40, 0x41, 0x42, 0x43
BLOCK_NS, BLOCK_EW = 0x49, 0x4A
ROCK_CLIMB_NS, ROCK_CLIMB_EW = 0x4B, 0x4C
JUMP_TWICE = frozenset({0x5A, 0x5B, 0x5C, 0x5D})  # Distortion World only (player_move.c:2001-2035)
WATERFALL = 0x13
WARP_ENTRANCE_SOUTH = 0x65  # TILE_BEHAVIOR_WARP_ENTRANCE_SOUTH, map_tile_behaviors.h:112
DOOR = 0x69  # TILE_BEHAVIOR_DOOR, :116
BRIDGE_START = 0x70
BRIDGES = frozenset(range(0x71, 0x7E))  # TileBehavior_IsBridge, src/map_tile_behavior.c:659-674
BIKE_BRIDGES = frozenset({0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D})  # :695-709
BRIDGE_OVER_WATER = frozenset({0x73, 0x78, 0x7C})  # TileBehavior_IsBridgeOverWater :676-681
# TILE_BEHAVIOR_FLAG_SURFABLE in sTileBehaviorFlags (src/map_tile_behavior.c:13-140; the test
# re-reads the table from the decomp).
SURFABLE = frozenset(
    {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x19, 0x22, 0x2A, 0x50, 0x51, 0x52, 0x53, 0x73, 0x78, 0x7C}
)
TABLE = 0x80  # TILE_BEHAVIOR_TABLE, :139
PC_TILE = 0x83  # TILE_BEHAVIOR_PC, :142
NO_WARP = 0xFFFF
ESCAPE_BIT = 14  # isEscapeRopeAllowed, include/map_header.h:28
MAP_TYPE_MASK = 0x7F  # mapType : 7
CAVE_MAP_TYPE = 3  # MapHeader_IsCave: rom header 199 / 284 map type 3 (facts-final)
CUT_SCRIPT = 10000  # field-move scripts start at 10000, include/script_manager.h:112
ROCK_SMASH_SCRIPT = 10001
STRENGTH_SCRIPT = 10002
VARS_START = 0x4000  # generated/vars_flags.txt:4107

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # N S W E, MapObject_GetDxFromDir order
OPPOSITE = (1, 0, 3, 2)
JUMP = (JUMP_N, JUMP_S, JUMP_W, JUMP_E)  # the jump tile of each step direction
SLIDE = {SLIDE_N: 0, SLIDE_S: 1, SLIDE_W: 2, SLIDE_E: 3}
# TileBehavior_BlocksMovement{North,South,West,East}ward (src/map_tile_behavior.c:574-604).
BLOCKS = (
    frozenset({BLOCK_N, BLOCK_NE, BLOCK_NW, BLOCK_NS}),
    frozenset({BLOCK_S, BLOCK_SE, BLOCK_SW, BLOCK_NS}),
    frozenset({BLOCK_W, BLOCK_NW, BLOCK_SW, BLOCK_EW}),
    frozenset({BLOCK_E, BLOCK_NE, BLOCK_SE, BLOCK_EW}),
)
ROCK_CLIMB = (ROCK_CLIMB_NS, ROCK_CLIMB_NS, ROCK_CLIMB_EW, ROCK_CLIMB_EW)  # by step direction

FX32_ONE = 4096
HEIGHT_STEP = 20 * FX32_ONE  # GetVerticalDirection, src/terrain_collision_manager.c:270
TILE_FX = 16 * FX32_ONE  # MAP_OBJECT_TILE_SIZE
BDHC_CANDIDATES = 10  # BDHC_NEW_OBJECT_HEIGHT_CANDIDATES_ARRAY_SIZE, bdhc.c:12

TRAINER_NORMAL = 1  # TRAINER_TYPE_NORMAL
TRAINER_ALL_DIRECTIONS = 2  # TRAINER_TYPE_VIEW_ALL_DIRECTIONS
TRAINER_FACE_FIRST, TRAINER_FACE_LAST = 4, 8  # FACE_SIDES .. SPIN_CLOCKWISE count as NORMAL
# Movement type -> the directions a trainer faces over time (generated/movement_types.txt,
# 0-based; the sets of Unk_020EEB08, src/unk_0206450C.c:1758-1798). Another type keeps the
# initial direction.
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

# Badge ids (generated/badges.txt, 0-based; the bit TrainerInfo_HasBadge tests).
COAL, FOREST, COBBLE, FEN, RELIC, MINE, ICICLE, BEACON = range(8)
# The splits in game order (tools/extract_kaizo_docs.py SPLITS) and the badge each split's last
# fight gives: res/field/scripts/scripts_oreburgh_city_gym.s:28 (Coal), scripts_eterna_city_gym.s:76
# (Forest), scripts_hearthome_city_gym_leader_room.s:71 (Relic), scripts_veilstone_city_gym.s:34
# (Cobble), scripts_pastoria_city_gym.s:58 (Fen), scripts_canalave_city_gym.s:31 (Mine),
# scripts_snowpoint_city_gym.s:35 (Icicle), scripts_sunyshore_city_gym_room_3.s:41 (Beacon); the
# Kaizo files hold the same GiveBadge (ROM test: scr 52, 73, 137, 125, 38, 176, 160; the Relic
# Badge moved to Kaizo's rebuilt Hearthome Gym, scr 96 and 101).
SPLIT_BADGES = (
    ("Roark", COAL),
    ("Gardenia", FOREST),
    ("Fantina", RELIC),
    ("Maylene", COBBLE),
    ("Wake", FEN),
    ("Byron", MINE),
    ("Candice", ICICLE),
    ("Galactic", None),
    ("Volkner", BEACON),
    ("Elite Four", None),
)


class MapReadError(ValueError):
    """The ROM holds a map the read cannot model."""


class Pos(NamedTuple):
    header: int
    x: int
    z: int


def grid_y(height: int) -> int:
    """The height unit of a map object: ``(pos.y >> 3) / FX32_ONE`` (``sub_020630DC``,
    ``src/map_object.c:2147-2153``; C division truncates toward zero)."""
    q = height >> 3
    return q // FX32_ONE if q >= 0 else -((-q) // FX32_ONE)


@dataclass(frozen=True)
class FieldMoves:
    """The field moves the walk may use: a badge held and no partner where the script refuses
    one (``res/field/scripts/scripts_field_moves.s``; the party knowing the move is the player's
    choice: HM06 Rock Smash ``scripts_oreburgh_gate_1f.s:30``, HM03 Surf
    ``scripts_celestic_town_cave.s:129``, HM04 Strength ``scripts_iron_island.s:98``)."""

    cut: bool = False
    rock_smash: bool = False
    strength: bool = False
    surf: bool = False
    rock_climb: bool = False
    waterfall: bool = False

    @classmethod
    def from_badges(cls, badges: Collection[int], partner: bool = False) -> FieldMoves:
        b = set(badges)
        return cls(
            cut=FOREST in b,  # scripts_field_moves.s:29; field_move_tasks.c:332
            rock_smash=COAL in b,  # :134; field_move_tasks.c:545
            strength=MINE in b,  # :191; field_move_tasks.c:463
            surf=FEN in b and not partner,  # field_control.c:690; :302-303 no partner
            rock_climb=ICICLE in b and not partner,  # :251, :253-254 no partner
            waterfall=BEACON in b,  # field_move_tasks.c:590
        )

    @classmethod
    def for_split(cls, split: str, partner: bool = False) -> FieldMoves:
        """The field moves of a group in ``split``: the badges of every earlier split (the split
        ends with the fight that gives its badge, ``SPLIT_BADGES``)."""
        names = [s for s, _ in SPLIT_BADGES]
        if split not in names:
            raise MapReadError(f"unknown split {split!r}")
        held = [b for s, b in SPLIT_BADGES[: names.index(split)] if b is not None]
        return cls.from_badges(held, partner)


NO_FIELD_MOVES = FieldMoves()


class MapObject(NamedTuple):
    """One object event (``tools/jsoncnv/event.py:51-69``; ``y`` the fx32 event height;
    ``flag`` the hidden flag, ``ObjectEvent.hiddenFlag``, ``include/map_header_data.h:20-33``)."""

    index: int
    local_id: int
    movement: int
    trainer_type: int
    script: int
    direction: int
    sight: int
    x: int
    z: int
    y: int = 0
    flag: int = 0

    @property
    def trainer(self) -> int | None:
        """The trainer id of a battle script (3000 + id - 1 single, 5000 + id - 1 double)."""
        if SINGLE_BATTLE_SCRIPTS <= self.script < VISIBLE_ITEM_SCRIPTS:
            base = DOUBLE_BATTLE_SCRIPTS if self.script >= DOUBLE_BATTLE_SCRIPTS else 3000
            return self.script - base + 1
        return None

    def cleared(self, moves: FieldMoves) -> bool:
        """An item ball (picked up) or a field-move object the player can remove."""
        if VISIBLE_ITEM_SCRIPTS <= self.script < VISIBLE_ITEM_SCRIPTS + 1000:
            return True
        return (
            (self.script == CUT_SCRIPT and moves.cut)
            or (self.script == ROCK_SMASH_SCRIPT and moves.rock_smash)
            or (self.script == STRENGTH_SCRIPT and moves.strength)
        )


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


def literal_coord(coord: Coord) -> bool | None:
    """Whether a coord on a var below ``VARS_START`` runs: the var reads as its own number
    (``FieldSystem_TryGetVar``, ``src/script_manager.c:444-452``; ``sub_0203CC14``,
    ``src/unk_0203C954.c:218``). None for a real var."""
    if coord.var < VARS_START:
        return coord.var == coord.value
    return None


@dataclass
class Area:
    """The tiles and events of one map header (``tiles``: (x, z) -> terrain attribute u16;
    ``heights``: (x, z) -> the fx32 heights of the BDHC plates through the tile centre, in the
    order ``CalculateObjectHeight`` meets them; a tile with none has no plate)."""

    header: int
    tiles: Mapping[tuple[int, int], int]
    objects: tuple[MapObject, ...] = ()
    warps: tuple[Warp, ...] = ()
    coords: tuple[Coord, ...] = ()
    map_type: int = 0
    escape_bit: bool = False
    heights: Mapping[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)
    object_at: dict = field(init=False, repr=False, default_factory=dict)
    _pc: frozenset | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.object_at = {}
        for o in self.objects:
            self.object_at.setdefault((o.x, o.z), []).append(o)

    def behaviour(self, x: int, z: int) -> int | None:
        a = self.tiles.get((x, z))
        return None if a is None else a & BEHAVIOUR_MASK

    def plate(self, x: int, z: int, current: int) -> int | None:
        """The height ``CalculateObjectHeight`` gives an object at ``current`` on the tile: the
        first candidate nearest ``current`` (``bdhc.c:206-224``), None with no plate."""
        cands = self.heights.get((x, z), ())
        if not cands:
            return None
        return min(cands, key=lambda h: abs(h - current))

    def object_height(self, o: MapObject) -> int:
        h = self.plate(o.x, o.z, o.y)
        return o.y if h is None else h

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
        (y,) = struct.unpack_from("<i", r, 28)
        objects.append(MapObject(i, v[0], v[2], v[3], v[5], v[6], v[7], x, z, y, v[4]))
    warps = [Warp(i, *struct.unpack_from("<4H", r)) for i, r in enumerate(records[2])]
    coords = []
    for r in records[3]:
        script, x, z, w, length, _y, value, var = struct.unpack_from("<8H", r)
        coords.append(Coord(script, x, z, w, length, value, var))
    return tuple(objects), tuple(warps), tuple(coords)


def _fx_mul(a: int, b: int) -> int:
    return (a * b + 0x800) >> 12  # FX_Mul: rounded fx32 product


def _fx_div(a: int, b: int) -> int:
    q = Fraction(a * FX32_ONE, b)  # FX_Div: the fx32 quotient, rounded to nearest
    return int(q + Fraction(1, 2)) if q >= 0 else -int(-q + Fraction(1, 2))


def bdhc_heights(member: bytes, land: int = 0) -> list[tuple[int, ...]]:
    """Per tile (row-major 32 x 32) the candidate plate heights of a land data member's BDHC
    (``BDHC_LoadHeader`` and the load order, ``src/overlay005/bdhc.c:239-330``; the tile centre at
    ``(c * 16 + 8 - 256, r * 16 + 8 - 256)`` relative to the land, ``GetHeight``,
    ``src/terrain_collision_manager.c:78-85``)."""
    sizes = struct.unpack_from("<4I", member, 0)
    off = 0x10 + sizes[0] + sizes[1] + sizes[2]
    b = member[off : off + sizes[3]]
    if b[:4] != b"BDHC":
        raise MapReadError(f"land data {land}: no BDHC section")
    n_pts, n_nrm, n_cst, n_pl, n_st, n_acc = struct.unpack_from("<6H", b, 4)
    pos = 16
    pts = [struct.unpack_from("<2i", b, pos + 8 * i) for i in range(n_pts)]
    pos += 8 * n_pts
    nrm = [struct.unpack_from("<3i", b, pos + 12 * i) for i in range(n_nrm)]
    pos += 12 * n_nrm
    cst = [struct.unpack_from("<i", b, pos + 4 * i)[0] for i in range(n_cst)]
    pos += 4 * n_cst
    plates = [struct.unpack_from("<4H", b, pos + 8 * i) for i in range(n_pl)]
    pos += 8 * n_pl
    strips = [struct.unpack_from("<iHH", b, pos + 8 * i) for i in range(n_st)]
    pos += 8 * n_st
    access = struct.unpack_from(f"<{n_acc}H", b, pos)

    def strip_index(scan: int) -> int | None:
        # BDHC_FindStripIndexByScanline, bdhc.c:113-150, step for step
        if n_st == 0:
            return None
        if n_st == 1:
            return 0
        low, high = 0, n_st - 1
        mid = high // 2
        while True:
            if strips[mid][0] > scan:
                if high - 1 > low:
                    high = mid
                    mid = (low + high) // 2
                else:
                    return mid
            elif low + 1 < high:
                low = mid
                mid = (low + high) // 2
            else:
                return mid + 1

    out: list[tuple[int, ...]] = []
    for r in range(TILES_PER_SIDE):
        z = (r * 16 + 8 - 256) * FX32_ONE
        si = strip_index(z)
        for c in range(TILES_PER_SIDE):
            x = (c * 16 + 8 - 256) * FX32_ONE
            cands: list[int] = []
            if si is not None and si < n_st:
                _, count, start = strips[si]
                for i in range(count):
                    p1, p2, ni, ci = plates[access[start + i]]
                    (x1, z1), (x2, z2) = pts[p1], pts[p2]
                    if min(x1, x2) <= x <= max(x1, x2) and min(z1, z2) <= z <= max(z1, z2):
                        nx, ny, nz = nrm[ni]
                        h = _fx_div(-(_fx_mul(nx, x) + _fx_mul(nz, z) + cst[ci]), ny)
                        cands.append(h)
                        if len(cands) >= BDHC_CANDIDATES:
                            break
            out.append(tuple(cands))
    return out


def read_double_trainers(files: RomFiles) -> frozenset[int]:
    """The trainers whose trainer data battle type is not singles (u32 at 16 of a
    ``trdata.narc`` member; ``Script_IsTrainerDoubleBattle``, ``src/script_manager.c:512-515``)."""
    out = set()
    for tid, head in enumerate(files.narc(TRDATA_NARC)):
        if len(head) >= 20 and struct.unpack_from("<I", head, 16)[0] != 0:
            out.add(tid)
    return frozenset(out)


class RomMaps:
    """The map headers of a Kaizo ROM as ``Area`` objects (``dump.js``)."""

    def __init__(self, files: RomFiles):
        self._headers = files.narc(KAIZO_MAP_HEADERS)
        self._events = files.narc(EVENTS_NARC)
        self._matrices = files.narc(MATRIX_NARC)
        self._lands = files.narc(LAND_DATA_NARC)
        self._cache: dict[tuple[int, bool], Area] = {}
        self._land_cache: dict[int, tuple[tuple[int, ...], list[tuple[int, ...]]]] = {}

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

    def _land(self, land: int) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        if land not in self._land_cache:
            member = self._lands[land]
            if struct.unpack_from("<I", member, 0)[0] != ATTRIBUTES_SIZE:
                raise MapReadError(f"land data {land}: terrain attribute size is not 0x800")
            attrs = struct.unpack_from(f"<{ATTRIBUTES_SIZE // 2}H", member, ATTRIBUTES_OFFSET)
            self._land_cache[land] = (attrs, bdhc_heights(member, land))
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
            heights: dict[tuple[int, int], tuple[int, ...]] = {}
            for k, land in enumerate(matrix.lands):
                if land == INVALID_LAND:
                    continue
                if masked and matrix.headers is not None and matrix.headers[k] != header:
                    continue
                x0 = (k % matrix.width) * TILES_PER_SIDE
                z0 = (k // matrix.width) * TILES_PER_SIDE
                attrs, plates = self._land(land)
                for i, a in enumerate(attrs):
                    p = (x0 + i % TILES_PER_SIDE, z0 + i // TILES_PER_SIDE)
                    tiles[p] = a
                    if plates[i]:
                        heights[p] = plates[i]
        except (IndexError, struct.error, TerrainError) as e:
            raise MapReadError(f"map header {header}: {e}") from e
        area = Area(header, tiles, objects, warps, coords, h.map_type, h.escape_bit, heights)
        self._cache[key] = area
        return area


def escape(maps, header: int) -> bool:
    """Whether Escape Rope and Dig work on a header: a cave map type with the escape bit
    (``CanUseEscapeRope``, ``src/item_use_functions.c:941-948``; ``src/field_move_tasks.c:760``)."""
    a = maps.area(header)
    return a.map_type == CAVE_MAP_TYPE and a.escape_bit


# --- the walk ---------------------------------------------------------------------------------

Fight = tuple[int, ...]


class Sighting(NamedTuple):
    """Where the player stands when a fight starts: ``how`` is ``"sight"`` (a tile in a
    trainer's line), ``"arrival"`` (a warp arrival or the start tile) or ``"talk"``; ``y`` the
    player's height there; ``flags`` the flags set then (``Walker.flags`` and ``on_enter``);
    ``bridge`` the elevated-bridge state. A battle keeps the field's map objects
    (``FieldTask_Encounter`` runs ``FieldTransition_FinishMap``, ``src/encounter.c:173-175``,
    ``src/field_transition.c:70-80``; objects are deleted only on a map change,
    ``FieldMapChange_DeleteObjects``, ``src/field_map_change.c:355-360``), so the player's
    status bit ``MAP_OBJ_STATUS_ON_ELEVATED_BRIDGE`` (``include/constants/map_object.h:43``)
    holds after the fight: the walk after it starts in this state."""

    pos: Pos
    how: str
    y: int = 0
    flags: frozenset = frozenset()
    bridge: bool = False

    @property
    def state(self) -> tuple:
        """The start state the walk after the fight depends on (``how`` aside)."""
        return (self.pos, self.y, self.bridge, self.flags)


class Seer(NamedTuple):
    """An undefeated trainer object that sees a tile: ``turning`` when its facing changes."""

    index: int
    trainer: int
    turning: bool


@dataclass
class Walk:
    """The result of one walk: the tiles reached (tile -> the tile before it), the fights that
    can start (a fight is a tuple of trainer ids, one or two, in the game's object order), the
    objects that can be talked to (``(header, local id)`` -> the player's tile) and the coord
    events crossed. ``spot`` keeps the first sighting of each fight; ``starts`` every distinct
    one: sight is checked on every field-input frame before movement
    (``src/overlay005/field_control.c:188`` < :198), so every tile of a trainer's line the
    player can stand on, and every talk tile, starts the fight with its own height, bridge
    state and flags, and the walk after the fight depends on which."""

    seen: dict[Pos, Pos | None]
    spot: dict[Fight, Sighting] = field(default_factory=dict)
    starts: dict[Fight, set[Sighting]] = field(default_factory=dict)
    talk: dict[tuple[int, int], Pos] = field(default_factory=dict)
    crossed: list[tuple[Pos, Coord]] = field(default_factory=list)
    # held: the start tile starts a fight on every facing, so the player has no input there
    # (the sight check runs before movement and the start menu, field_control.c:188 < :198,
    # :331-333); states: the player states reached (tile, height, bridge, flags)
    held: bool = False
    states: set = field(default_factory=set)

    def headers(self) -> set[int]:
        return {p.header for p in self.seen}

    def trainers(self) -> set[int]:
        """Every trainer of a fight that can start."""
        return {t for f in self.spot for t in f}

    def sighted(self, f: Fight, sight: Sighting) -> None:
        """Record a start of ``f``."""
        self.spot.setdefault(f, sight)
        self.starts.setdefault(f, set()).add(sight)


CoordRule = Callable[[int, Coord, "bool | None"], "str | None"]


class _State(NamedTuple):
    pos: Pos
    y: int
    bridge: bool
    flags: frozenset = frozenset()


class Walker:
    """The walk rules over ``maps`` (an object with ``area(header) -> Area``).

    ``members``: the trainer ids of the group (a member with a walking movement type raises).
    ``moves``: the field moves the walk may use (``FieldMoves``). ``ignore(header, obj)``: an
    object that never blocks and never sees (the partner who follows the player). ``vars``: the
    known values of coord variables. ``coord_rule(header, coord, matches)``: ``"pass"``,
    ``"block"`` (the script takes the player: the tile is not entered) or None for the default: in
    a ``strict`` header (every header when None) a matching coord raises and one on an unknown var
    blocks (its script may start a battle or move the player: Route 213's coords on 0x40FA,
    header 373 scripts 9, 10, 13 start battles 692+693, 694+695, 491; Building 75's on 0x407A
    starts 406), except ``UNKNOWN_COORDS_PASSED`` (open); in another header a matching or
    unknown-var coord blocks too, so the walk grants nothing a coord script could take away.
    ``doubles``: the double-battle trainers (``read_double_trainers``). ``two_alive``: the party
    has two live Pokemon (``Party_HasTwoAliveMons``); ``partner``: the partner flag is set. Sight
    takes two when either holds (``field_control.c:182-186``); a talk reads the party only
    (``scrcmd_trainer.c:283-289``). ``flags``: the flags set
    when the walk starts; an object whose hidden flag is set is not created (``hidden``).
    ``on_enter``: header -> the flags its ``INIT_SCRIPT_ON_TRANSITION`` script sets on every
    path; a warp onto the header adds them before its objects are created
    (``FieldMapChange_UpdateGameData``, ``src/field_map_change.c:254-299``, then
    ``FieldMapChange_CreateObjects`` :341-351), so they hold for every map loaded after."""

    def __init__(
        self,
        maps,
        members: Collection[int] = (),
        *,
        moves: FieldMoves = NO_FIELD_MOVES,
        ignore: Callable[[int, MapObject], bool] | None = None,
        vars: Mapping[int, int] | None = None,
        coord_rule: CoordRule | None = None,
        strict: Collection[int] | None = None,
        doubles: Collection[int] = (),
        two_alive: bool = True,
        partner: bool = False,
        flags: Collection[int] = (),
        on_enter: Mapping[int, Collection[int]] | None = None,
    ):
        self.maps = maps
        self.members = frozenset(members)
        self.moves = moves
        self.ignore = ignore
        self.vars = dict(vars or {})
        self.coord_rule = coord_rule
        self.strict = None if strict is None else frozenset(strict)
        self.doubles = frozenset(doubles)
        self.two_alive = two_alive
        self.partner = partner
        self.flags = frozenset(flags)
        self.on_enter = {h: frozenset(f) for h, f in (on_enter or {}).items()}
        self._sight: dict[tuple, dict[tuple[int, int], list[tuple[Seer, int]]]] = {}
        self._walks: dict[tuple, Walk] = {}

    def reached_headers(self) -> set[int]:
        """Every header a walk of this walker has reached."""
        return {h for w in self._walks.values() for h in w.headers()}

    # -- terrain and objects

    def hidden(self, o: MapObject, flags: frozenset | None = None) -> bool:
        """An object the map load does not create: one with a script whose hidden flag is set
        (``sub_020620C4``, ``src/map_object.c:607-624``: created when it has no script or
        ``FieldSystem_CheckFlag(hiddenFlag)`` is FALSE; flag 0 is never set,
        ``VarsFlags_GetFlagChunk``, ``src/vars_flags.c:52-56``). A script-less object's flag
        field is not a flag (``ObjectEvent_GetHiddenFlagNoScript``, ``map_object.c:2333-2337``)."""
        return object_hidden(o, self.flags if flags is None else flags)

    def _ignored(self, header: int, o: MapObject, flags: frozenset | None = None) -> bool:
        return self.hidden(o, flags) or (self.ignore is not None and self.ignore(header, o))

    def _object_at(
        self,
        area: Area,
        x: int,
        z: int,
        y: int,
        removable: bool,
        flags: frozenset | None = None,
    ) -> MapObject | None:
        """An object on the tile within 2 height units of ``y`` that blocks (``sub_02063F00``)."""
        for o in area.object_at.get((x, z), ()):
            if removable and o.cleared(self.moves):
                continue
            if self._ignored(area.header, o, flags):
                continue
            if abs(grid_y(area.object_height(o)) - grid_y(y)) < 2:
                return o
        return None

    def _collides(self, area: Area, frm: tuple[int, int], y: int, d: int) -> bool:
        """``MAP_OBJ_COLLISION_WILL_COLLIDE`` of a step from ``frm`` at height ``y`` in ``d``."""
        dx, dz = DIRS[d]
        return self._collides_from(area, frm, (frm[0] + dx, frm[1] + dz), y, d)

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

    def facing(self, obj: MapObject) -> tuple[tuple[int, ...], bool] | None:
        """The directions a trainer object looks along and whether it turns (several facings
        over time), or None for no sight."""
        tt = obj.trainer_type
        if tt == TRAINER_ALL_DIRECTIONS:
            return (0, 1, 2, 3), False
        if tt == TRAINER_NORMAL or TRAINER_FACE_FIRST <= tt <= TRAINER_FACE_LAST:
            if self._inert(obj):
                return None
            dirs = LOOK.get(obj.movement) or (obj.direction,)
            return dirs, len(dirs) > 1
        return None

    def sight_tiles(
        self,
        header: int,
        obj: MapObject,
        removable: bool = True,
        flags: frozenset | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """The tiles ``(x, z, direction, distance)`` where ``obj`` can spot the player
        (``GetTrainerDistToPlayer`` :182-229, ``IsPathInterrupted`` :310-344): every tile up to
        the range that does not collide for the trainer, before the first that does or that
        holds another object at its height."""
        if obj.trainer is None:
            return []
        look = self.facing(obj)
        if look is None:
            return []
        area = self.maps.area(header)
        h = area.object_height(obj)
        out = []
        for d in look[0]:
            dx, dz = DIRS[d]
            for k in range(1, obj.sight + 1):
                x, z = obj.x + dx * k, obj.z + dz * k
                # every tile is tested from the trainer's own tile and height (sub_02063E94)
                if self._collides_from(area, (obj.x, obj.z), (x, z), h, d):
                    break
                if self._object_at(area, x, z, h, removable, flags) is not None:
                    break
                out.append((x, z, d, k))
        return out

    def _collides_from(
        self, area: Area, origin: tuple[int, int], tile: tuple[int, int], y: int, d: int
    ) -> bool:
        """``MAP_OBJ_COLLISION_WILL_COLLIDE`` of an object on ``origin`` at height ``y`` facing
        ``d`` for ``tile``: no tile (the null behaviour), a height step of 20 or more, the
        collision bit, or a directional tile (``sub_02063E18``, ``map_object_move.c:577-604``;
        ``sub_02064004`` :706-722 tests the object's own tile)."""
        attr = area.tiles.get(tile)
        if attr is None:
            return True
        h = area.plate(*tile, y)
        if abs((0 if h is None else h) - y) >= HEIGHT_STEP or attr & COLLISION:
            return True
        return area.behaviour(*origin) in BLOCKS[d] or attr & BEHAVIOUR_MASK in BLOCKS[OPPOSITE[d]]

    def _sight_table(
        self, header: int, defeated: frozenset[int], removable: bool, flags: frozenset
    ):
        key = (header, defeated, removable, flags)
        table = self._sight.get(key)
        if table is None:
            table = {}
            area = self.maps.area(header)
            for o in area.objects:
                t = o.trainer
                if t is None or t in defeated or self._ignored(header, o, flags):
                    continue
                look = self.facing(o)
                if look is None:
                    continue
                seer = Seer(o.index, t, look[1])
                gy = grid_y(area.object_height(o))
                for sx, sz, _, _ in self.sight_tiles(header, o, removable, flags):
                    table.setdefault((sx, sz), []).append((seer, gy))
            self._sight[key] = table
        return table

    def sight_at(
        self,
        header: int,
        x: int,
        z: int,
        defeated: frozenset[int],
        removable: bool = True,
        y: int = 0,
        flags: frozenset | None = None,
    ) -> list[Seer]:
        """The undefeated trainers that can spot the player standing on ``(x, z)`` at height
        ``y``, in object order (``sub_02063F00``: the player within 2 height units)."""
        flags = self.flags if flags is None else flags
        table = self._sight_table(header, frozenset(defeated), removable, flags)
        seers = [s for s, gy in table.get((x, z), ()) if abs(grid_y(y) - gy) < 2]
        return sorted(set(seers))

    def encounters(self, seers: list[Seer]) -> tuple[set[Fight], bool]:
        """The fights the tile can start and whether the player can stand on it unspotted: each
        set of turning seers that faces the player (fixed ones always do) gives the game's fight
        (``FieldSystem_CheckForTrainersWantingBattle``, ``trainer_encounter.c:84-131``)."""
        fixed = [s for s in seers if not s.turning]
        turning = [s for s in seers if s.turning]
        fights: set[Fight] = set()
        passable = False
        for r in range(len(turning) + 1):
            for chosen in combinations(turning, r):
                f = self._fight(sorted([*fixed, *chosen]))
                if f is None:
                    passable = True
                else:
                    fights.add(f)
        return fights, passable

    def _fight(self, seeing: list[Seer]) -> Fight | None:
        if not seeing:
            return None
        first = seeing[0]
        two = self.two_alive or self.partner  # field_control.c:182-186
        if first.trainer in self.doubles:  # :108-118: both objects, or no fight
            return (first.trainer,) if two else None
        second = next((s for s in seeing[1:] if s.trainer != first.trainer), None)
        if two and second is not None:  # :98-106, APPROACH_TYPE_VS2
            return (first.trainer, second.trainer)
        return (first.trainer,)

    def _spotted(
        self,
        walk: Walk,
        pos: Pos,
        y: int,
        how: str,
        defeated: frozenset[int],
        removable: bool,
        flags: frozenset,
        bridge: bool,
    ) -> bool:
        """Record the fights the tile starts; True when the player cannot stand on it."""
        fights, passable = self.encounters(
            self.sight_at(pos.header, pos.x, pos.z, defeated, removable, y, flags)
        )
        for f in fights:
            walk.sighted(f, Sighting(pos, how, y, flags, bridge))
        return not passable

    # -- coords

    def coord_matches(self, coord: Coord) -> bool | None:
        """Whether a coord event runs: a literal var by its number (``literal_coord``); else by
        the known ``vars`` (None: unknown)."""
        literal = literal_coord(coord)
        if literal is not None:
            return literal
        if coord.var in self.vars:
            return self.vars[coord.var] == coord.value
        return None

    def _coord_step(self, pos: Pos, walk: Walk) -> bool:
        """Apply the coord events of a tile entered; False when a script takes the player.

        The game runs only the first coord of the tile, in event order, whose var matches
        (``sub_0203CC14``, ``src/unk_0203C954.c:217-221``; one script, ``field_control.c:769-773``):
        a known match ends the loop. A coord whose var is unknown may or may not be that one,
        so its action applies (by default it blocks) and the later coords are applied too."""
        area = self.maps.area(pos.header)
        for c in area.coords:
            if not c.covers(pos.x, pos.z):
                continue
            matches = self.coord_matches(c)
            if matches is False:
                continue
            action = self.coord_rule(pos.header, c, matches) if self.coord_rule else None
            if action is None:
                foreign = self.strict is not None and pos.header not in self.strict
                if foreign or (
                    matches is None and (pos.header, c.script) not in UNKNOWN_COORDS_PASSED
                ):
                    action = "block"
                elif matches:
                    raise MapReadError(
                        f"coord event script {c.script} at {pos} runs (var {c.var:#x}) and is "
                        "not modelled"
                    )
                else:
                    action = "pass"
            if action == "block":
                return False
            walk.crossed.append((pos, c))
            if matches:
                break
        return True

    # -- movement

    def _water(self, behaviour: int | None, bridge: bool) -> bool:
        """``MapObject_IsOnWater`` (``map_object_move.c:739-750``)."""
        if behaviour is None:
            return False
        if behaviour in BRIDGE_OVER_WATER:
            return not bridge
        return behaviour in SURFABLE

    def _land(self, area: Area, x: int, z: int, y: int, bridge: bool) -> tuple[int, bool]:
        """The height and elevated-bridge state after the player arrives on ``(x, z)``."""
        h = area.plate(x, z, y)
        b = area.behaviour(x, z)
        if b == BRIDGE_START:
            bridge = True
        elif bridge and b not in BRIDGES:
            bridge = False
        return (y if h is None else h), bridge

    def _player_collision(self, area: Area, s: _State, d: int, removable: bool) -> set[str]:
        """The ``PLAYER_COLLISION_*`` kinds of a step (``PlayerAvatar_CheckCollision``)."""
        dx, dz = DIRS[d]
        x, z = s.pos.x + dx, s.pos.z + dz
        out = set()
        if self._collides(area, (s.pos.x, s.pos.z), s.y, d):
            out.add("collide")
        if self._object_at(area, x, z, s.y, removable, s.flags) is not None:
            out.add("object")
        b = area.behaviour(x, z)
        if b == JUMP[d]:
            out.add("jump")
        if b in JUMP_TWICE:
            out.add("jump_twice")
        if self._water(b, s.bridge):
            out.add("water")
        if s.bridge and b in BIKE_BRIDGES:
            out.add("collide")
        return out

    def _step(self, area: Area, s: _State, d: int, removable: bool) -> list[_State] | None:
        """The tiles a step in ``d`` puts the player on, in order (the last is where he stops;
        a field-move traverse and a jump skip the tiles between), or None when he stays (the
        caller then tries talking). Doors are handled by the caller."""
        dx, dz = DIRS[d]
        x, z = s.pos.x + dx, s.pos.z + dz
        hit = self._player_collision(area, s, d, removable)
        surfing = self._water(area.behaviour(s.pos.x, s.pos.z), s.bridge)
        b = area.behaviour(x, z)
        if surfing:
            if hit <= {"water"}:
                return [self._at(area, s, x, z)]
            if b == WATERFALL and self.moves.waterfall:
                return self._traverse(area, s, d, {WATERFALL})
            return None
        if "jump" in hit:  # JUMP_FAR: two tiles, landing not tested (player_move.c:1024-1026)
            lx, lz = x + dx, z + dz
            if (lx, lz) not in area.tiles:
                return None
            return [self._at(area, s, lx, lz)]
        if "jump_twice" in hit:
            raise MapReadError(f"header {area.header}: a Distortion World jump tile at {(x, z)}")
        if not hit:
            return [self._at(area, s, x, z)]
        if hit == {"water"} or (hit == {"water", "collide"} and b in SURFABLE):
            on_bridge = area.behaviour(s.pos.x, s.pos.z) in BRIDGES | {BRIDGE_START}
            if self.moves.surf and not (on_bridge and s.bridge):
                return [self._at(area, s, x, z)]  # the surf mount jumps one tile (ov5:707-716)
            return None
        if b == ROCK_CLIMB[d] and self.moves.rock_climb and "object" not in hit:
            return self._traverse(area, s, d, {ROCK_CLIMB[d]})
        return None

    def _at(self, area: Area, s: _State, x: int, z: int) -> _State:
        y, bridge = self._land(area, x, z, s.y, s.bridge)
        return _State(Pos(area.header, x, z), y, bridge, s.flags)

    def _traverse(self, area: Area, s: _State, d: int, kinds: set[int]) -> list[_State] | None:
        """A field-move script that carries the player over the run of ``kinds`` tiles in ``d``
        to the first other tile."""
        dx, dz = DIRS[d]
        cur = s
        x, z = s.pos.x, s.pos.z
        while True:
            x, z = x + dx, z + dz
            if (x, z) not in area.tiles:
                return None
            cur = self._at(area, cur, x, z)
            if area.behaviour(x, z) not in kinds:
                return [cur]

    def _forced(self, area: Area, s: _State, d: int, removable: bool) -> list[_State]:
        """The tiles a spin or ice tile then carries the player over (empty: none)."""
        out: list[_State] = []
        b = area.behaviour(s.pos.x, s.pos.z)
        if b in SLIDE:
            d = SLIDE[b]
            seen = set()
            cur = s
            # the first move is made without a collision test (ov5_021E1154.c:91-99)
            dx, dz = DIRS[d]
            if (cur.pos.x + dx, cur.pos.z + dz) not in area.tiles:
                return out
            cur = self._at(area, cur, cur.pos.x + dx, cur.pos.z + dz)
            out.append(cur)
            while True:
                nb = area.behaviour(cur.pos.x, cur.pos.z)
                d = SLIDE.get(nb, d)
                if (cur.pos, d) in seen:
                    raise MapReadError(f"header {area.header}: an endless spin at {cur.pos}")
                seen.add((cur.pos, d))
                if self._player_collision(area, cur, d, removable):
                    return out
                dx, dz = DIRS[d]
                cur = self._at(area, cur, cur.pos.x + dx, cur.pos.z + dz)
                out.append(cur)
        if b == ICE:
            cur = s
            while area.behaviour(cur.pos.x, cur.pos.z) == ICE:
                if self._player_collision(area, cur, d, removable):
                    break
                dx, dz = DIRS[d]
                cur = self._at(area, cur, cur.pos.x + dx, cur.pos.z + dz)
                out.append(cur)
        return out

    # -- the walk

    def arrival_height(self, pos: Pos) -> int:
        """The height of the player arriving on a map at ``pos`` (from 0, then the plate)."""
        h = self.maps.area(pos.header).plate(pos.x, pos.z, 0)
        return 0 if h is None else h

    def walk(
        self,
        start: Pos,
        defeated: Iterable[int] = (),
        removable: bool = True,
        y: int | None = None,
        flags: Collection[int] | None = None,
        bridge: bool | None = None,
    ) -> Walk:
        """Breadth-first walk from ``start`` (at height ``y``, an arrival when None) with
        ``defeated`` trainers beaten and ``flags`` set (``Walker.flags`` when None; ``bfs2.js``
        ``run``), on the elevated bridge when ``bridge`` (when None: the start tile is a bridge
        start): sight is checked on the start tile, every tile entered and every warp
        arrival."""
        defeated = frozenset(defeated)
        flags = self.flags if flags is None else frozenset(flags)
        if y is None:
            y = self.arrival_height(start)
        if bridge is None:
            bridge = self.maps.area(start.header).behaviour(start.x, start.z) == BRIDGE_START
        key = (start, y, bridge, defeated, removable, flags)
        if key in self._walks:
            return self._walks[key]
        walk = Walk({start: None})
        self._walks[key] = walk
        if self._spotted(walk, start, y, "arrival", defeated, removable, flags, bridge):
            walk.held = True
            return walk
        s0 = _State(start, y, bridge, flags)
        walk.states.add(s0)
        return self._bfs(walk, deque([s0]), defeated, removable)

    def walk_from(
        self, seeds: Collection[_State], defeated: Iterable[int] = (), removable: bool = True
    ) -> Walk:
        """The walk from every state of ``seeds`` at once (states another walk reached), with
        the sight check on each seed tile first: the player stands there when the walker's
        count starts to hold (a revive or a PC on that tile). ``Walk.spot`` holds the fights of
        every seed's walk."""
        defeated = frozenset(defeated)
        key = ("from", frozenset(seeds), defeated, removable)
        if key in self._walks:
            return self._walks[key]
        walk = Walk({s.pos: None for s in seeds})
        self._walks[key] = walk
        queue: deque[_State] = deque()
        for s in sorted(seeds):
            if not self._spotted(walk, s.pos, s.y, "sight", defeated, removable, s.flags, s.bridge):
                walk.states.add(s)
                queue.append(s)
        return self._bfs(walk, queue, defeated, removable)

    def _bfs(
        self, walk: Walk, queue: deque[_State], defeated: frozenset[int], removable: bool
    ) -> Walk:
        states = walk.states
        while queue:
            s = queue.popleft()
            area = self.maps.area(s.pos.header)
            for d, (dx, dz) in enumerate(DIRS):
                nx, nz = s.pos.x + dx, s.pos.z + dz
                warp = area.warp_at(nx, nz)
                attr = area.tiles.get((nx, nz))
                door = (
                    warp is not None
                    and attr is not None
                    and attr & BEHAVIOUR_MASK == DOOR
                    and self._object_at(area, nx, nz, s.y, removable, s.flags) is None
                )
                path = [self._at(area, s, nx, nz)] if door else self._step(area, s, d, removable)
                if path is None:
                    self._talk(area, s, dx, dz, defeated, walk)
                    continue
                here = path[-1]
                if not door and self._spotted(
                    walk, here.pos, here.y, "sight", defeated, removable, here.flags, here.bridge
                ):
                    continue
                if not door:
                    forced = self._forced(self.maps.area(here.pos.header), here, d, removable)
                    if forced:
                        for f in forced[:-1]:
                            walk.seen.setdefault(f.pos, s.pos)
                        walk.seen.setdefault(here.pos, s.pos)
                        here = forced[-1]
                        if self._spotted(
                            walk, here.pos, here.y, "sight", defeated, removable, here.flags,
                            here.bridge,
                        ):  # fmt: skip
                            continue
                target = here
                warp = self.maps.area(here.pos.header).warp_at(here.pos.x, here.pos.z)
                if warp is not None and warp.dest != NO_WARP:
                    landing = self._arrival(warp)
                    if landing is not None:
                        ly = self.arrival_height(landing)
                        lf = here.flags | self.on_enter.get(landing.header, frozenset())
                        la = self.maps.area(landing.header)
                        lb = la.behaviour(landing.x, landing.z) == BRIDGE_START
                        if self._spotted(walk, landing, ly, "arrival", defeated, removable, lf, lb):
                            continue
                        target = _State(landing, ly, lb, lf)
                if target in states:
                    continue
                if not self._coord_step(target.pos, walk):
                    continue
                states.add(target)
                walk.seen.setdefault(target.pos, s.pos)
                queue.append(target)
                if target is not here:
                    walk.seen.setdefault(here.pos, s.pos)
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
        self, area: Area, s: _State, dx: int, dz: int, defeated: frozenset[int], walk: Walk
    ) -> None:
        """The player on ``s`` faces ``(dx, dz)``: the object in front, or across a table tile
        the one behind it, at the player's height unit, can be talked to."""
        pos = s.pos
        tx, tz = pos.x + dx, pos.z + dz
        if area.behaviour(tx, tz) == TABLE:
            tx, tz = tx + dx, tz + dz
        for o in area.object_at.get((tx, tz), ()):
            if grid_y(area.object_height(o)) != grid_y(s.y) or self.hidden(o, s.flags):
                continue
            walk.talk.setdefault((area.header, o.local_id), pos)
            t = o.trainer
            # a double-battle trainer talked to with one live Pokemon only prints a message
            # (res/field/scripts/scripts_battles.s:19-22, Battles_NotEnoughPokemonMessage): the
            # script reads the party (ScrCmd_CheckHasTwoAliveMons, src/scrcmd_trainer.c:283-289),
            # not the partner flag
            lone = t in self.doubles and not self.two_alive
            if t is not None and t not in defeated and not self._inert(o) and not lone:
                walk.sighted((t,), Sighting(pos, "talk", s.y, s.flags, s.bridge))
            return

    def talk_tiles(
        self,
        start: Pos,
        header: int,
        member: int,
        defeated: Iterable[int] = (),
        removable: bool = True,
    ) -> list[tuple[int, int]]:
        """The tiles, reached from ``start`` without being spotted, from which ``member`` can
        be talked to (next to it, or two tiles away across a table tile)."""
        area = self.maps.area(header)
        obj = next((o for o in area.objects if o.trainer == member), None)
        if obj is None:
            raise MapReadError(f"trainer {member} is not an object of header {header}")
        seen = self.walk(start, defeated, removable).seen
        out = []
        for dx, dz in DIRS:
            near = (obj.x + dx, obj.z + dz)
            if area.behaviour(*near) == TABLE:
                far = (obj.x + 2 * dx, obj.z + 2 * dz)
                if Pos(header, *far) in seen:
                    out.append(far)
            elif Pos(header, *near) in seen:
                out.append(near)
        return sorted(out)


def pc_reachable(maps, walk: Walk, headers: Collection[int]) -> bool:
    """Whether a tile next to a PC tile (behaviour 0x83) of ``headers`` was reached."""
    return bool(pc_spots(maps, walk, headers))


def pc_spots(maps, walk: Walk, headers: Collection[int]) -> set[Pos]:
    """The tiles next to a PC tile (behaviour 0x83) of ``headers`` where the player stands with
    input (``Walk.states``; ``Walk.seen`` also holds the tiles a spin, a slide or a jump only
    crosses, and the warp tiles left at once)."""
    stood = {s.pos for s in walk.states}
    out = set()
    for h in headers:
        for x, z in maps.area(h).pc_tiles():
            out |= {Pos(h, x + dx, z + dz) for dx, dz in DIRS} & stood
    return out


# --- chains (chain.js) --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    """A chain of undefeated trainers walked from ``entries``: ``members`` (every trainer of the
    chain), ``headers`` (the chain's own map headers; reaching another header is a leave).
    ``entry_flags``: per entry, the flag sets the player can arrive with besides the walker's
    own (the flags earlier visits set; empty: the walker's flags only)."""

    members: tuple[int, ...]
    headers: tuple[int, ...]
    entries: tuple[Pos, ...]
    entry_flags: tuple[tuple[frozenset, ...], ...] = ()


class Run(NamedTuple):
    """One full sequence of the chain's fights, the walks after each prefix and the map header
    of each fight (where the player stands when it starts). ``states[k]`` holds the walk from
    every start state of the k-th fight that some history of the first k fights reaches, with
    those fights beaten (``states[0]``: the walks from the entries), k = 0..n."""

    sequence: tuple[Fight, ...]
    states: tuple[tuple[Walk, ...], ...]
    headers: tuple[int, ...] = ()


def _chain_fights(walk: Walk, members: frozenset, group: frozenset, done: frozenset):
    """The fights of ``group`` the walk can start, not yet fought (fight -> its starts); a
    fight that mixes the chain's members with other trainers raises."""
    out: dict[Fight, set[Sighting]] = {}
    for f, sights in walk.starts.items():
        fs = set(f)
        if not fs & group or fs & done:
            continue
        if not fs <= members:
            raise MapReadError(f"fight {f} mixes the chain {sorted(members)} with others")
        out[f] = sights
    return out


def _start_walk(walker: Walker, s: Sighting, done: frozenset) -> Walk:
    return walker.walk(s.pos, done, y=s.y, flags=s.flags, bridge=s.bridge)


def chain_prefixes(
    walker: Walker, spec: ChainSpec
) -> tuple[dict[tuple[Fight, ...], tuple[Walk, ...]], dict[Fight, int]]:
    """The walks after every prefix of the chain's fights, over every start state (a DP over
    (prefix, start state)): the walks of a prefix p + f are the walks from every start of f
    (``Walk.starts``) in every walk of p, with p + f beaten. The walk after a fight depends only
    on its start state, the beaten trainers and the flags, so no history is enumerated. Also
    the map header of each fight; a fight that starts on two headers raises."""
    members = frozenset(spec.members)
    level: dict[tuple[Fight, ...], dict[int, Walk]] = {(): {}}
    for i, e in enumerate(spec.entries):
        extra = spec.entry_flags[i] if spec.entry_flags else (frozenset(),)
        for fl in extra:
            w = walker.walk(e, (), flags=walker.flags | fl)
            level[()][id(w)] = w
    walks: dict[tuple[Fight, ...], tuple[Walk, ...]] = {}
    heads: dict[Fight, int] = {}
    while level:
        nxt: dict[tuple[Fight, ...], dict[int, Walk]] = {}
        for p, ws in level.items():
            walks[p] = tuple(ws.values())
            done = frozenset(t for f in p for t in f)
            if done == members:
                continue
            for w in ws.values():
                for f, sights in _chain_fights(w, members, members, done).items():
                    bucket = nxt.setdefault((*p, f), {})
                    for s in sights:
                        if heads.setdefault(f, s.pos.header) != s.pos.header:
                            raise MapReadError(f"fight {f} starts on two map headers")
                        after = _start_walk(walker, s, done | set(f))
                        bucket[id(after)] = after
        level = nxt
    return walks, heads


def runs(walker: Walker, spec: ChainSpec) -> list[Run]:
    """Every full sequence of fights over ``spec.members`` the walk allows (``chain.js``
    ``rec``), with the walks of every start state after each prefix (``chain_prefixes``). A
    fight that mixes members and other trainers raises: the group is wrong."""
    members = frozenset(spec.members)
    walks, heads = chain_prefixes(walker, spec)
    out = []
    for p in walks:
        if frozenset(t for f in p for t in f) != members:
            continue
        states = tuple(walks[p[:k]] for k in range(len(p) + 1))
        out.append(Run(p, states, tuple(heads[f] for f in p)))
    return out


def start_counts(walker: Walker, spec: ChainSpec) -> tuple[int, int, int]:
    """The DP's size: prefixes, (prefix, start state) pairs, and the most start states of
    one prefix."""
    walks, _ = chain_prefixes(walker, spec)
    return len(walks), sum(len(w) for w in walks.values()), max(len(w) for w in walks.values())


def _gap_kinds(
    maps,
    walk: Walk,
    headers: Collection[int],
    healers: Collection[tuple[int, int]],
    pc_headers: Collection[int],
) -> set[str]:
    """The capabilities of one walk in a gap: ``leave`` (a header outside ``headers``
    reached), ``heal`` (a healer's talk tile reached), ``pc`` (a tile next to a PC of
    ``pc_headers`` stood on)."""
    kinds = set()
    if walk.headers() - set(headers):
        kinds.add("leave")
    if any(key in walk.talk for key in healers):
        kinds.add("heal")
    if pc_headers and pc_reachable(maps, walk, pc_headers):
        kinds.add("pc")
    return kinds


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
    restricted to the group's fights), per order the gaps where every history can leave the
    chain's headers, the gaps where every history reaches a healer or a PC, the gaps where some
    history has no input (``locked``: a restriction, so the union), and per order the map header
    of each fight. A history is a sequence with a start state per fight: a capability holds at
    a gap only when the walk from every start state of the prefix has it (``chain_prefixes``).
    ``orders``, ``leave_gaps``, ``fight_headers`` (one list per order), ``heal_gaps``,
    ``pc_gaps`` and ``locked_gaps`` (one list)."""
    members = frozenset(group)
    leave: dict[tuple[Fight, ...], set[int]] = {}
    where: dict[tuple[Fight, ...], tuple[int, ...]] = {}
    heal: set[int] | None = None
    pcs: set[int] | None = None
    locked: set[int] = set()
    kinds_of: dict[int, set[str]] = {}

    def kinds(walks: tuple[Walk, ...]) -> set[str]:
        out: set[str] | None = None
        for w in walks:
            k = kinds_of.get(id(w))
            if k is None:
                k = kinds_of[id(w)] = _gap_kinds(maps, w, spec.headers, healers, pc_headers)
            out = k if out is None else out & k
        return set(out or ())

    for run in runs(walker, spec):
        idx = []
        for i, f in enumerate(run.sequence):
            if set(f) & members:
                if not set(f) <= members:
                    raise MapReadError(f"fight {f} mixes the group {sorted(members)} with others")
                idx.append(i)
        order = tuple(run.sequence[i] for i in idx)
        heads = tuple(run.headers[i] for i in idx)
        if where.setdefault(order, heads) != heads:
            raise MapReadError(f"order {order}: a fight starts on two map headers")
        gaps_leave: set[int] = set()
        gaps_heal: set[int] = set()
        gaps_pc: set[int] = set()
        for k, i in enumerate(idx[:-1], 1):
            walks = run.states[i + 1]
            have = kinds(walks)
            if "leave" in have:
                gaps_leave.add(k)
            if "heal" in have:
                gaps_heal.add(k)
            if "pc" in have:
                gaps_pc.add(k)
            if any(w.held for w in walks):
                locked.add(k)
        leave[order] = leave[order] & gaps_leave if order in leave else gaps_leave
        heal = gaps_heal if heal is None else heal & gaps_heal
        pcs = gaps_pc if pcs is None else pcs & gaps_pc
    if not leave:
        raise MapReadError(f"no full sequence of {group}")
    orders = sorted(leave)
    return {
        "orders": [[list(f) for f in o] for o in orders],
        "leave_gaps": [sorted(leave[o]) for o in orders],
        "fight_headers": [list(where[o]) for o in orders],
        "pc_gaps": sorted(pcs or ()),
        "heal_gaps": sorted(heal or ()),
        "locked_gaps": sorted(locked),
    }


def live_table(
    maps,
    walkers: Mapping[int, Walker],
    spec: ChainSpec,
    doubles: Collection[int],
    *,
    healers: Collection[tuple[int, int]] = (),
    pc_headers: Collection[int] = (),
) -> list[dict]:
    """The next-fight table of a chain by the live count (``LiveRow`` rows as dicts;
    ``nuzline.data.gauntlet``). ``walkers[c]`` walks with the field count c (1 or 2).

    The walk after ``fought`` starts on a start state of the last fight (``Walk.starts``: every
    tile, height, bridge state and flag set it can start from) with those trainers beaten
    (``chain_prefixes``). Under count 1 the group is the members less the double-battle
    trainers, who are not fought (``trainer_encounter.c:110-116``; ``scripts_battles.s:19-22``),
    and come back when the count is two again. The recursion switches the count at every fight
    end (a faint, a forced heal: both act before the gap's first step, on the fight tile) and,
    inside a gap, where a revive (``bag``: any reached tile), a PC (``pc``: a reached tile next
    to a PC of ``pc_headers``, a deposit or a withdraw) or a healer (``heal``: a talk tile of
    ``healers``) changes it; each of those gives a change row whose fights are the plain row's
    plus those the walk under the new count starts from such a tile, and the walk goes on from
    each. Over the entries, entry flags, histories and start states that reach one key,
    ``next`` is the union, the capabilities (``leave``, ``heal``, ``pc``) the intersection and
    the restriction ``locked`` the union (``chain_facts``). A plain row's ``gap``: ``leave`` (a
    header outside ``spec.headers`` reached), ``heal`` (a healer's talk tile reached), ``pc``,
    ``locked`` (``Walk.held``: no input in the gap). A change row needs its tile in every
    history: none where some history is locked.

    A count-1 walker with the partner raises: sight then counts two whatever the party holds
    (``field_control.c:182-184``), so a double-battle trainer still fights by sight, while a
    talk reads the party only; the table does not model that split."""
    if walkers[1].partner:
        raise MapReadError(
            "live_table: a count-1 walk with the partner is not modelled (sight counts two, "
            "field_control.c:182-184)"
        )
    members = frozenset(spec.members)
    groups = {2: members, 1: members - frozenset(doubles)}
    nexts: dict[tuple, dict[Fight, int]] = {}
    gaps: dict[tuple, set[str]] = {}
    held: dict[tuple, bool] = {}
    explored: set[tuple] = set()

    def add(key: tuple, found: Mapping[Fight, set[Sighting]]) -> None:
        row = nexts.setdefault(key, {})
        for f, sights in found.items():
            for sight in sights:
                if row.setdefault(f, sight.pos.header) != sight.pos.header:
                    raise MapReadError(f"after {key[0]}: fight {f} starts on two map headers")

    def go(found: Mapping[Fight, set[Sighting]], done: frozenset, seq: tuple) -> None:
        for f, sights in found.items():
            for s in sorted(sights):
                rec(s.pos, s.y, s.flags, s.bridge, done | set(f), (*seq, f))

    def rec(
        pos: Pos, y: int | None, flags: frozenset, bridge: bool | None, done: frozenset, seq
    ) -> None:
        state = (pos, y, flags, bridge, done, seq)
        if state in explored:
            return
        explored.add(state)
        gap = bool(seq) and done != members
        for c in (1, 2):
            walk = walkers[c].walk(pos, done, y=y, flags=flags, bridge=bridge)
            found = _chain_fights(walk, members, groups[c], done)
            add((seq, c, c, ""), found)
            if gap:
                kinds = _gap_kinds(maps, walk, spec.headers, healers, pc_headers)
                talks = {walk.talk[k] for k in healers if k in walk.talk}
                spots = pc_spots(maps, walk, pc_headers) if pc_headers else set()
                key = (seq, c)
                gaps[key] = kinds if key not in gaps else gaps[key] & kinds
                held[key] = held.get(key, False) or walk.held
            go(found, done, seq)
            if not gap or walk.held:
                continue
            changes = []
            if c == 1:
                changes.append(("bag", 2, walk.states))
                changes.append(("heal", 2, {st for st in walk.states if st.pos in talks}))
            changes.append(("pc", 3 - c, {st for st in walk.states if st.pos in spots}))
            for at, to, tiles in changes:
                if not tiles:
                    continue
                key = (seq, c, to, at)
                add(key, found)
                more = _chain_fights(walkers[to].walk_from(tiles, done), members, groups[to], done)
                add(key, more)
                go(more, done, seq)

    for i, e in enumerate(spec.entries):
        extra = spec.entry_flags[i] if spec.entry_flags else (frozenset(),)
        for f in extra:
            rec(e, None, walkers[2].flags | f, None, frozenset(), ())
    rows = []
    for (seq, live, to, at), found in nexts.items():
        done = frozenset(t for f in seq for t in f)
        if done == members:
            continue
        order = sorted(found)
        row = {
            "fought": [list(f) for f in seq],
            "live": live,
            "next": [list(f) for f in order],
            "headers": [found[f] for f in order],
        }
        have = gaps.get((seq, live), set())
        if at:
            if held[(seq, live)] or (at == "pc" and "pc" not in have):
                continue  # a change needs the tile in every history (the gap facts' rule)
            if at == "heal" and "heal" not in have:
                continue
            row["to"] = to
            row["at"] = at
        elif seq:
            row["gap"] = [g for g in ("leave", "heal", "pc") if g in have]
            if held[(seq, live)]:
                row["gap"].append("locked")
        rows.append(row)
    return _rows(rows)


# --- scripts ------------------------------------------------------------------------------------


class ScriptReader:
    """Straight-line traces of a script entry (``scr.js``): the commands from the entry to the
    first End, Return or GoTo, branches not followed."""

    def __init__(self, files: RomFiles, decomp: Path):
        self._scripts = files.narc(SCRIPTS_NARC)
        self._table = command_table(Path(decomp))

    def battle_operands(self, member: int, entry: int) -> dict[tuple[int, int], set[int]]:
        """``(offset of a battle command, variable) -> the constants`` its variable operands
        hold on the paths from the 1-based ``entry`` (``tools.rom.readers.maps._walk``)."""
        script = self._scripts[member]
        pos = script_entries(script, f"scr_seq {member}")[entry - 1]
        return _walk(script, self._table, f"scr_seq {member}", [pos])[2]

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
    script = entry.pop()
    trace = scripts.trace(maps.header(header).scripts, script)
    if not any(c == "SCRCMD_LOCKALL" for _, c, _ in trace):
        raise MapReadError(f"header {header}: the entry does not LockAll")
    fights: list[list[int]] = []
    locked: list[int] = []
    released = False
    operands: dict | None = None
    for at, const, values in trace:
        if const == "SCRCMD_STARTTRAINERBATTLE":
            if fights and not released:
                locked.append(len(fights))
            # StartTrainerBattle enemy1 u16, enemy2 u16 (scrcmd.inc:1427-1431), each read by
            # ScriptContext_GetVar (src/scrcmd_trainer.c:129-130): below VARS_START the number
            # itself (FieldSystem_TryGetVar, src/script_manager.c:428-452), else the variable,
            # whose value is the constant its paths assign (one, or the fight is not fixed).
            ids = []
            for v in values[:2]:
                if v >= VARS_START:
                    if operands is None:
                        operands = scripts.battle_operands(maps.header(header).scripts, script)
                    got = operands.get((at, v), set())
                    if len(got) != 1 or NOT_CONSTANT in got:
                        raise MapReadError(f"header {header}: battle @{at} var {v:#x} is {got}")
                    v = got.pop()
                if v:
                    ids.append(v)
            fights.append(ids)
            released = False
        elif const == "SCRCMD_RELEASEALL":
            released = True
    return {
        "orders": [fights],
        "leave_gaps": [[]],
        "locked_gaps": locked,
        "fight_headers": [[header] * len(fights)],
    }


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


def _gym_coord_runs(coord: Coord) -> bool:
    runs_ = literal_coord(coord)
    if runs_ is None:
        raise MapReadError(f"Eterna Gym coord {coord.script} reads var {coord.var:#x}")
    return runs_


def gym_reach(
    area: Area,
    tables: tuple[list[list[int]], list[list[int]]],
    state: int,
    start: tuple[int, int],
    passed: Collection[int] = (),
    flags: Collection[int] = (),
) -> GymReach:
    """The tiles, talkable objects and coord events reachable in clock ``state`` from ``start``
    (``gym.js`` ``reach``): the clock and fountain collision and the hour-hand jump tile.

    A coord tile runs its script when entered and its var condition holds (``literal_coord``:
    the gym's coords are on var 0 with value 0, so they run): one whose script is not in
    ``passed`` starts its fight there,
    so it is recorded and the walk does not go on from it (``gym.js`` follows the same rule and
    gives the same reach: start and leave on the south warp entrance, stop at an unwon coord). A
    script in ``passed`` (its fight won) is a message only, neither recorded nor a stop: scr 73 entry 7 fights while 0x40AE <= 1 (@475-481), entry 8 while <= 2
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
        return bool(standing(x, z))

    def standing(x: int, z: int) -> list[MapObject]:
        return [o for o in area.object_at.get((x, z), ()) if not object_hidden(o, flags)]

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
                objs = standing(nx, nz)
                if objs:
                    talk.setdefault(objs[0].local_id, (x, z))
                continue
            if (nx, nz) in seen:
                continue
            seen.add((nx, nz))
            fights = [
                c.script
                for c in area.coords
                if c.covers(nx, nz) and c.script not in passed and _gym_coord_runs(c)
            ]
            for script in fights:
                coords.setdefault(script, (nx, nz))
            if not fights:
                queue.append((nx, nz))
    return GymReach(frozenset(seen), talk, coords)


def gym_facts(
    maps: RomMaps, decomp: Path, flags: Collection[int] = (), reached: set[int] | None = None
) -> dict:
    """The Eterna Gym trainers (E3-B3-2.c): the orders the clock states allow and the gaps where
    the door can be reached. State 0 from the door offers only the guide; each fight changes the
    reach as the clock advances (after the first win over 324 and after 260). ``flags``: the
    hidden flags set (``object_hidden``). The fights are coord scripts and talks of singles
    trainers (no gym trainer is a double-battle trainer: ROM test), so the live count changes
    nothing here. ``reached`` (when given) gains the gym and the door's destination."""
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
    if reached is not None:
        reached |= {GYM_HEADER, door.dest}
    r0 = gym_reach(area, tables, 0, start, (), flags)
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
        reach = gym_reach(area, tables, clock, pos, won, flags)
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
            after = gym_reach(area, tables, clock + step, where, won, flags)
            rec(clock + step, where, (*done, fight), [*doors, front in after.seen])

    rec(1, start, (), [])
    if not orders:
        raise MapReadError("no order of the gym trainers")
    keys = sorted(orders)
    return {
        "orders": [[[t] for t in o] for o in keys],
        "leave_gaps": [[k for k, d in enumerate(orders[o][:-1], 1) if d] for o in keys],
        "fight_headers": [[GYM_HEADER] * len(o) for o in keys],
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
WAYWARD_SPLIT = "Galactic"  # Galactic Split:105


def _wayward_walker(
    maps, joined: bool, doubles: Collection[int] = (), two_alive: bool = True
) -> Walker:
    """The walk with the partner following (she neither blocks nor sees; 0x4091 = 1; the field
    then counts two live Pokemon, ``field_control.c:183-185``, and refuses Surf and Rock Climb)
    or still standing at (72, 13) as an obstacle (0x4091 = 0). Each coord runs by its var
    condition, as the field does (``coord_matches``): a running coord of ``WAYWARD_TAKES`` takes
    the player, so its tiles are not walked through; a coord whose var does not match is walked
    over; any other running coord raises (not modelled). ``two_alive``: the party's live count;
    once she joins, sight counts two (``Walker.partner``), a talk still reads the party."""

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
        moves=FieldMoves.for_split(WAYWARD_SPLIT, partner=joined),
        ignore=ignore,
        vars={WAYWARD_PARTNER_VAR: 1 if joined else 0},
        strict={WAYWARD_HEADER},
        coord_rule=rule,
        doubles=doubles,
        two_alive=two_alive,
        partner=joined,
        flags=GROUP_FLAGS["Wayward Cave"],
    )


def _wayward_next(
    walker: Walker, start: Sighting | Pos, done: frozenset[int]
) -> tuple[Walk, dict[Fight, set[Sighting]]]:
    """The walk from ``start`` (a fight's start state, or the arrival tile) and the fights of
    pair members it can start (fight -> every start state, ``Walk.starts``)."""
    if isinstance(start, Sighting):
        w = _start_walk(walker, start, done)
    else:
        w = walker.walk(start, done)
    members = frozenset(t for pair in WAYWARD_PAIRS for t in pair)
    try:
        nxt = _chain_fights(w, members, members, done)
    except MapReadError as e:
        raise MapReadError(f"Wayward Cave: {e}") from None
    return w, nxt


def wayward_facts(
    maps: RomMaps,
    scripts: ScriptReader,
    doubles: Collection[int] = (),
    two_alive: bool = True,
    headers: set[int] | None = None,
) -> dict:
    """Wayward Cave: every fight sequence the walk with the partner allows up to the exit, the
    partner's join points and the partner's trainer id. ``headers`` (when given) gains every
    header the walks reach."""
    area = maps.area(WAYWARD_HEADER)
    mira = next((o for o in area.objects if o.local_id == WAYWARD_MIRA), None)
    if mira is None:
        raise MapReadError("Wayward Cave has no partner object")
    joined = _wayward_walker(maps, True, doubles, two_alive)
    alone = _wayward_walker(maps, False, doubles, two_alive)
    rows = [c for c in area.coords if c.script == WAYWARD_EXIT_ROW]
    if len(rows) != 1:
        raise MapReadError("Wayward Cave: expected one exit coord row")
    row = rows[0]
    exits = {(x, row.z - 1) for x in range(row.x, row.x + row.width)}
    members = frozenset(t for pair in WAYWARD_PAIRS for t in pair)

    sequences: dict[tuple[Fight, ...], frozenset[int]] = {}

    def by_level(walker: Walker, visit) -> None:
        # a DP over (prefix, start state): the starts of p + f are every start of f in every
        # walk of p (``Walk.starts``), so no history is enumerated
        level: dict[tuple[Fight, ...], set] = {(): {WAYWARD_START}}
        while level:
            nxt: dict[tuple[Fight, ...], set] = {}
            for seq, starts in level.items():
                done = frozenset(t for f in seq for t in f)
                for start in starts:
                    w, fights = _wayward_next(walker, start, done)
                    visit(w, seq, done)
                    for f, sights in fights.items():
                        nxt.setdefault((*seq, f), set()).update(sights)
            level = nxt

    def with_partner(w: Walk, seq: tuple[Fight, ...], done: frozenset[int]) -> None:
        if any(Pos(WAYWARD_HEADER, x, z) in w.seen for x, z in exits):
            sequences[seq] = done

    by_level(joined, with_partner)
    short = {seq for seq, done in sequences.items() if done != members}
    if not sequences or short:
        raise MapReadError(f"Wayward Cave: the exit is reachable after {sorted(short)}")

    reached: dict[int, bool] = {}

    def without_partner(w: Walk, seq: tuple[Fight, ...], done: frozenset[int]) -> None:
        near = [(mira.x + dx, mira.z + dz) for dx, dz in DIRS]
        ok = any(Pos(WAYWARD_HEADER, x, z) in w.seen for x, z in near)
        reached[len(seq) + 1] = reached.get(len(seq) + 1, True) and ok

    by_level(alone, without_partner)
    # Without the partner the walk-out coord (0x4091 == 0) closes the start pocket: no pair is
    # reached and her tile is, so the game makes her join before fight 1 and the walk-out
    # never runs at a gap. A fight reached without her would make the join point a choice.
    join_before = sorted(k for k, ok in reached.items() if ok)
    if join_before != [1]:
        raise MapReadError(f"Wayward Cave: the partner's join points are {reached}, not fight 1")
    if headers is not None:
        headers |= joined.reached_headers() | alone.reached_headers()

    keys = sorted(sequences, key=lambda s: (len(s), s))  # the all-pairs order first
    orders = [[*(list(f) for f in seq), list(WAYWARD_LAST)] for seq in keys]
    trace = scripts.trace(maps.header(WAYWARD_HEADER).scripts, 2)
    partner = [
        v for _, c, v in trace if c == "SCRCMD_SETVARFROMVALUE" and v[0] == WAYWARD_PARTNER_ID_VAR
    ]
    if len(partner) != 1:
        raise MapReadError("Wayward Cave: the partner's trainer id is not set once")
    return {
        "orders": orders,
        "leave_gaps": [[] for _ in orders],
        "fight_headers": [[WAYWARD_HEADER] * len(o) for o in orders],
        "partner": {"trainer": partner[0][1], "join_before": join_before},
    }


# --- Elite Four and Champion ---------------------------------------------------------------------

LEAGUE_SCRIPTS = (188, 190, 192, 194, 196)  # the five rooms' script files (rom: scr 188/.../196)
LEAGUE_TRAINERS = (866, 867, 868, 869, 870)
LEAGUE_HEADERS = tuple(range(175, 187))


def league_facts(maps: RomMaps) -> dict:
    """The five rooms in script order; the PC gaps are the lifts after rooms 1..n-1 (warp 0 of a
    room leads to its lift) whose PC tile can be reached from the lift's arrival. Not a group of
    ``derive_groups`` yet: batch 4 (B4-19) adds the Elite Four + Champion group with the chain
    rules (``docs/gauntlet-design.md``)."""
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
        "fight_headers": [rooms],
        "escape_header": rooms[0],
    }


# --- the groups ----------------------------------------------------------------------------------

BUILDINGS = ChainSpec(
    members=(421, 831, 410, 423, 412, 422),
    headers=(72, 73, 74, 75),
    entries=(Pos(72, 11, 14),),
)
# The two lobby-end entries are the tiles outside the lobby's doors (warps 3 (706,818) and 4
# (706,813) -> header 376): the player came out of the lobby, so its OnTransition set flag 482
# (``ENTER_FLAGS``). The west entry (from the Pastoria gate, header 374) holds either state: a
# lobby visit before the group is the player's history.
LOBBY_FLAGS = frozenset({482})
ROUTE_213 = ChainSpec(
    members=(106, 103, 105, 111, 110),
    headers=(373,),
    entries=(Pos(373, 706, 819), Pos(373, 706, 814), Pos(373, 647, 812)),
    entry_flags=((LOBBY_FLAGS,), (LOBBY_FLAGS,), (frozenset(), LOBBY_FLAGS)),
)
OREBURGH = ChainSpec(members=(195, 196), headers=(199,), entries=(Pos(199, 15, 1),))
ROUTE_204 = ChainSpec(members=(15, 13, 14), headers=(346,), entries=(Pos(346, 180, 699),))
# The lobby healer of Route 213: header 376 object 1 at (11, 4), talked to across the table tile
# (11, 5); scr 462 entry 2 runs HealParty (@74).
ROUTE_213_HEALERS = ((376, 1),)
# Open (needs a manager ruling): Route 213's coords on 0x40FA (header 373, scr 459 entries 9-13:
# value 3 battle 692+693 sets 4, value 4 battle 694+695 sets 5, value 5 a party check, value 7 a
# cutscene that moves the player and sets 8, value 8 battle 491 sets 9) lie on the chain's path:
# blocked, the chain has no full sequence. The var's value at the group's start is not read, so
# these pass as before (unsafe: the walk may skip a scripted battle or a forced move). Every other
# unknown-var coord in a strict header blocks; a ROM test pins the list.
UNKNOWN_COORDS_PASSED = frozenset({(373, 9), (373, 10), (373, 11), (373, 12), (373, 13)})
B2B_HEADER = 256

# The split of each group: the sheet section that lists it (the [[gauntlet]] ``source`` of
# data/kaizo/overrides.toml; the ROM test checks the two agree).
GROUP_SPLITS = {
    "Oreburgh Mine": "Roark",  # Roark Split:227
    "Route 204 North": "Gardenia",  # Gardenia Split:57
    "Route 204 North B2B": "Gardenia",
    "Eterna Gym gym trainers": "Gardenia",  # Gardenia Split:451
    "Team Galactic Eterna Building #1": "Fantina",  # Fantina Split:5
    "Team Galactic Eterna Building #2": "Fantina",
    "Team Galactic Eterna Building #3": "Fantina",
    "Route 213": "Wake",  # Wake Split:192
    "Wayward Cave": WAYWARD_SPLIT,  # Galactic Split:105
}


# --- hidden flags --------------------------------------------------------------------------------
#
# An object is created on a map load only while its hidden flag is clear (``Walker.hidden``).
# ``FLAG_WRITERS`` lists every SetFlag / ClearFlag of each hidden flag carried by an object of a
# group's own headers (and the Route 213 lobby, the healer's header) in the whole ROM script
# archive, as ``(script file, offset, "set" | "clear")`` (the ROM test re-reads the archive and
# requires exactly these). Map-local flags (1-0x40: the field-move objects) are cleared on every
# map change (``FieldSystem_ClearLocalFlags``, ``src/script_manager.c:475-481``, from
# ``FieldMapChange_UpdateGameData``, ``src/field_map_change.c:254``) and item balls are taken as
# picked up (``MapObject.cleared``): neither is listed. Script file 412 is
# ``scripts_init_new_game`` (``res/field/scripts/meson.build:419``), run once at a new game
# (``FieldSystem_InitNewGameState``, ``src/script_manager.c:691-694``). Flag names are the
# vanilla ones (``generated/vars_flags.txt``); Kaizo reuses 482 for six Route 213 objects.
FLAG_WRITERS: dict[int, tuple[tuple[int, int, str], ...]] = {
    394: (),  # Oreburgh Mine B2F Roark (header 199 obj 0)
    # header 256 obj 3 (works key): scr 277 entry 2 clears it after both fights
    414: ((412, 298, "set"), (277, 311, "clear")),
    415: (),  # header 256 obj 0
    416: (),  # header 256 obj 1
    545: (),  # Eterna Gym Gardenia (header 67 obj 1)
    509: ((82, 391, "set"),),  # the building's grunts: 4F (header 75) entry 2, after its boss
    443: ((412, 218, "set"), (79, 145, "clear"), (82, 395, "set")),  # 1F Looker (header 72 obj 3)
    450: ((79, 206, "clear"), (82, 399, "set")),  # 1F grunt-Looker (header 72 obj 2)
    710: (),  # 1F wall to Rotom's room (header 72 obj 4)
    511: (),  # 4F hostages (header 75 objs 1-3)
    538: ((412, 246, "set"), (459, 1263, "clear"), (459, 1286, "clear")),  # Route 213 obj 28
    488: ((412, 250, "set"), (459, 1300, "clear")),  # Route 213 obj 31
    482: ((462, 37, "set"), (462, 152, "set")),  # Route 213 objs 33-38 (trainer 696's pair)
    529: ((412, 254, "set"), (459, 1479, "clear"), (416, 2239, "set")),  # lobby obj 2
    484: (),  # Wayward Cave Mira (header 284 obj 4)
}
# The flags set when each group starts, from ``FLAG_WRITERS`` and the story before its split:
# - a flag no script writes is clear (394, 415, 416, 545, 710, 511, 484): those objects stand.
# - 414: set at a new game, cleared only by the B2B script itself after its fights: set.
# - 509, 450: set only by scr 82 entry 2 (4F, after the building): clear. 443: set at a new game,
#   cleared by scr 79 entry 5, the frame-table script of header 72 (init 579: entry 5 while var
#   0x411D == 0, which it sets to 1 @181) that runs on the first frame in the 1F, before any step
#   (``FieldSystem_RunInitScript(INIT_SCRIPT_ON_FRAME_TABLE)``, ``field_control.c:176``): the
#   group's walk starts in the 1F, so clear.
# - 538 (the Route 213 grunt): set at a new game; scr 459 entry 1 (the OnTransition, init 941)
#   clears it @1286 only past CheckBadgeAcquired BADGE_ID_FEN @827-839, and entry 2 @1263 is the
#   grunt's own talk script: no Fen Badge before the Wake split (``SPLIT_BADGES``), so set.
#   488 and 529: set at a new game, cleared only in entry 2 past the grunt (@1300, @1479): set.
# - 482: written only by the lobby's OnTransition (scr 462 entry 1, init 944), on both branches
#   (var 0x40FA == 9 @24 -> @152, else @37): set by entering header 376 (``ENTER_FLAGS``).
GROUP_FLAGS: dict[str, frozenset[int]] = {
    "Oreburgh Mine": frozenset(),
    "Route 204 North": frozenset(),
    "Route 204 North B2B": frozenset({414}),
    "Eterna Gym gym trainers": frozenset(),
    "Team Galactic Eterna Building #1": frozenset(),
    "Team Galactic Eterna Building #2": frozenset(),
    "Team Galactic Eterna Building #3": frozenset(),
    "Route 213": frozenset({538, 488, 529}),
    "Wayward Cave": frozenset(),
}
# header -> the flags its OnTransition script sets on every path (``Walker.on_enter``).
ENTER_FLAGS: dict[int, frozenset[int]] = {376: LOBBY_FLAGS}


def object_hidden(obj: MapObject, flags: Collection[int]) -> bool:
    """``Walker.hidden`` for a fixed flag set."""
    return obj.script != NO_SCRIPT and obj.flag != 0 and obj.flag in flags


def group_escape(maps, fight_headers: Iterable[Iterable[int]]) -> bool:
    """Escape Rope and Dig (``escape``) on the map headers the group's fights start on; the
    headers must agree."""
    values = {escape(maps, h) for heads in fight_headers for h in heads}
    if len(values) != 1:
        raise MapReadError(f"the group's fight headers disagree on escape: {values}")
    return values.pop()


def _facts(orders, leave_gaps, fight_headers, escape_, **more) -> dict:
    return {
        "orders": orders,
        "leave_gaps": leave_gaps,
        "fight_headers": fight_headers,
        "escape": escape_,
        "pc_gaps": [],
        "heal_gaps": [],
        "locked_gaps": [],
        "partner": None,
        "live_next": None,
        **more,
    }


def _rows(rows) -> list[dict]:
    key = lambda r: (len(r["fought"]), r["fought"], r["live"], r.get("at", ""), r.get("to", 0))
    return sorted(rows, key=key)


def chain_live_next(
    maps,
    walkers: Mapping[int, Walker],
    spec: ChainSpec,
    group: tuple[int, ...],
    doubles: Collection[int],
    **kw,
) -> list[dict] | None:
    """``live_next`` of a chain group: None when the chain's walk does not depend on the count
    (its ``live_table`` is the prefix table of its two-live orders, ``prefix_live_rows``),
    else the table. A chain shared by several groups (``group`` short of ``spec.members``)
    whose walk depends on the count raises: its table is not split into groups."""
    from nuzline.data.gauntlet import prefix_live_rows

    table = _rows(live_table(maps, walkers, spec, doubles, **kw))
    whole = chain_facts(maps, walkers[2], spec, spec.members, **kw)
    orders = [tuple(tuple(f) for f in o) for o in whole["orders"]]
    prefix = prefix_live_rows(
        orders,
        whole["leave_gaps"],
        whole["fight_headers"],
        whole["pc_gaps"],
        whole["heal_gaps"],
        whole["locked_gaps"],
    )
    if table == _rows(r.to_dict() for r in prefix):
        return None
    if set(group) != set(spec.members):
        raise MapReadError(
            f"the walk of the chain {sorted(spec.members)} depends on the live count and the "
            f"chain holds several groups ({sorted(group)} is one): not modelled"
        )
    return table


def gap_fields(facts: Mapping, live_next: list[dict] | None) -> dict:
    """``leave_gaps``, ``heal_gaps`` and ``pc_gaps`` of a group as stored: the two-live
    values of ``facts`` (``chain_facts``) where the group has no ``live_next``; with one, a
    field whose table gap facts depend on the order or the live count is None (absent:
    ``nuzline.data.gauntlet.flat_gaps``), and a flat one must be the two-live value."""
    from nuzline.data.gauntlet import FLAT_GAP_FIELDS, LiveRow, flat_gaps

    out = {k: facts[k] for k in FLAT_GAP_FIELDS}
    if live_next is None:
        return out
    rows = [LiveRow.from_dict(r) for r in live_next]
    for key, kind in FLAT_GAP_FIELDS.items():
        flat = flat_gaps(rows, kind)
        if flat is None:
            out[key] = None
            continue
        want = list(flat)
        if key == "leave_gaps":
            want = [[k for k in flat if k < len(o)] for o in facts["orders"]]
        if want != out[key]:
            raise MapReadError(
                f"{key}: the live table gives {want} for every order and count, the two-live "
                f"walk {out[key]}"
            )
    return out


def derive_groups(
    files: RomFiles, decomp: Path, reached: dict[str, set[int]] | None = None
) -> dict[str, dict]:
    """The facts of every gauntlet group, read from the ROM: ``orders`` (a list of orders, each
    a list of fights, each a list of trainer ids), ``leave_gaps`` and ``fight_headers`` (one
    list per order), ``escape``, ``pc_gaps``, ``heal_gaps``, ``locked_gaps``, ``partner``
    (None, or ``{"trainer", "join_before"}``) and ``live_next``.

    The walk depends on the party's live count (``Walker.two_alive``): with two live Pokemon
    (or a partner) two singles trainers who see the player at once fight as one double battle
    and a double-battle trainer fights; with one, the first sees him alone and the next one
    after it, and a double-battle trainer is not fought at all (``trainer_encounter.c:94-118``,
    ``field_control.c:181-186``; talked to, ``res/field/scripts/scripts_battles.s:19-22``).
    The facts above are the two-live ones; ``live_next`` is the next-fight table by the count
    (``live_table``), or None when the walk does not depend on it. With a table,
    ``leave_gaps``, ``heal_gaps`` and ``pc_gaps`` are None where its gap facts depend on the
    order or the count (``gap_fields``). The planner picks by the party.

    ``reached`` (when given) gains, per group, every header its walks reach (the ROM test of
    the persisted map features reads it)."""
    maps = RomMaps(files)
    seen: dict[str, set[int]] = {} if reached is None else reached
    scripts = ScriptReader(files, decomp)
    doubles = read_double_trainers(files)
    out: dict[str, dict] = {}

    def chain(name: str, spec: ChainSpec, group: tuple[int, ...], **kw) -> None:
        walkers = {
            c: Walker(
                maps,
                spec.members,
                moves=FieldMoves.for_split(GROUP_SPLITS[name]),
                # the healer's header too: a coord there decides whether the heal is reached
                strict=(*spec.headers, *(h for h, _ in kw.get("healers", ()))),
                doubles=doubles,
                two_alive=c == 2,
                flags=GROUP_FLAGS[name],
                on_enter=ENTER_FLAGS,
            )
            for c in (1, 2)
        }
        facts = chain_facts(maps, walkers[2], spec, group, **kw)
        live_next = chain_live_next(maps, walkers, spec, group, doubles, **kw)
        seen.setdefault(name, set()).update(*(w.reached_headers() for w in walkers.values()))
        gaps = gap_fields(facts, live_next)
        out[name] = _facts(
            facts["orders"],
            gaps["leave_gaps"],
            facts["fight_headers"],
            group_escape(maps, facts["fight_headers"]),
            pc_gaps=gaps["pc_gaps"],
            heal_gaps=gaps["heal_gaps"],
            locked_gaps=facts["locked_gaps"],
            live_next=live_next,
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

    # B2B: two scripted single battles (b2b_facts); the gym: coord scripts and talks of singles
    # trainers (gym_facts). Neither reads the count: live_next None.
    b2b = b2b_facts(maps, scripts, B2B_HEADER)
    seen["Route 204 North B2B"] = {B2B_HEADER}  # one script entry: no walk
    out["Route 204 North B2B"] = _facts(
        b2b["orders"],
        b2b["leave_gaps"],
        b2b["fight_headers"],
        group_escape(maps, b2b["fight_headers"]),
        locked_gaps=b2b["locked_gaps"],
    )
    gym = gym_facts(
        maps,
        decomp,
        GROUP_FLAGS["Eterna Gym gym trainers"],
        seen.setdefault("Eterna Gym gym trainers", set()),
    )
    out["Eterna Gym gym trainers"] = _facts(
        gym["orders"],
        gym["leave_gaps"],
        gym["fight_headers"],
        group_escape(maps, gym["fight_headers"]),
    )
    # Wayward Cave: the partner joins before fight 1 and counts two at every fight; the walk
    # under either party count must give the same facts.
    way_headers = seen.setdefault("Wayward Cave", set())
    way = wayward_facts(maps, scripts, doubles, True, way_headers)
    if wayward_facts(maps, scripts, doubles, False, way_headers) != way:
        raise MapReadError("Wayward Cave: the facts depend on the party's live count")
    out["Wayward Cave"] = _facts(
        way["orders"],
        way["leave_gaps"],
        way["fight_headers"],
        group_escape(maps, way["fight_headers"]),
        partner=way["partner"],
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
