"""``tools.kaizo.gauntlet_map`` (E3-B3-2): the map read on synthetic maps, and, ROM-backed (slow;
skips without the Kaizo ROM and the decomp), the tool's output for every group against the facts
in ``data/kaizo/overrides.toml``."""

import os
import random
import re
import struct
from itertools import pairwise
from pathlib import Path

import pytest

from nuzline.data.gauntlet import MoveCase, Skip, WildLevel
from tools.kaizo.gauntlet_map import (
    CLOCK_SIDE,
    CLOCK_STATES,
    ENCOUNTER,
    FOUNTAIN_WIDTH,
    ROAMING_HEADERS,
    SPLIT_BADGES,
    STRENGTH_ACTIVE,
    SURFABLE,
    WAYWARD_HEADER,
    WAYWARD_PARTNER_VAR,
    WAYWARD_START,
    WAYWARD_WALKOUT,
    Area,
    ChainSpec,
    Coord,
    FieldMoves,
    MapObject,
    MapReadError,
    Pos,
    Seer,
    Sighting,
    StaticMaps,
    Walker,
    Warp,
    _State,
    _wayward_next,
    _wayward_union,
    _wayward_walker,
    bdhc_heights,
    chain_facts,
    chain_group_facts,
    eterna_clock_tables,
    gap_fields,
    grid_y,
    gym_reach,
    live_table,
    move_subsets,
    pc_reachable,
    pc_spots,
    relevant_moves,
    runs,
    wild_levels,
)
from tools.rom.inputs import REPO_ROOT

# --- synthetic maps ---------------------------------------------------------------------------

TILE = {
    ".": 0,
    "#": 0x8000,
    "D": 0x8069,
    "T": 0x8080,
    "P": 0x8083,
    "W": 0x8000,
    "v": 0x803B,  # JUMP_SOUTH with the collision bit, as the ROM's ledges (header 199)
    "~": 0x15,  # WATER_SEA
    "e": 0x30,  # BLOCK_EASTWARD
    "n": 0x42,  # SLIDE_NORTHWARD (a spin tile)
    "i": 0x20,  # ICE
    "g": 0x02,  # TALL_GRASS (an encounter tile)
    "c": 0x08,  # CAVE_FLOOR (an encounter tile)
}
EAST, SOUTH, WEST, NORTH = 3, 1, 2, 0


def area(header, rows, objects=(), warps=(), coords=(), heights=None):
    """An area from ASCII rows (z down, x across): ``.`` free, ``#`` wall, ``D`` door (collision,
    behaviour 0x69), ``T`` table, ``P`` PC; a space is no tile."""
    tiles = {(x, z): TILE[c] for z, row in enumerate(rows) for x, c in enumerate(row) if c != " "}
    return Area(header, tiles, tuple(objects), tuple(warps), tuple(coords), heights=heights or {})


def trainer(tid, x, z, facing, sight, movement=0, trainer_type=1):
    return MapObject(tid, tid, movement, trainer_type, 3000 + tid - 1, facing, sight, x, z)


def npc(local_id, x, z, script=100):
    return MapObject(local_id, local_id, 0, 0, script, 0, 0, x, z)


def boulder(local_id, x, z):
    """A Strength boulder: script 10002, graphics 0x54 (as header 284's objects 19 and 26)."""
    return MapObject(local_id, local_id, 0, 0, 10002, 0, 0, x, z, gfx=0x54)


def test_sight_stops_at_a_wall():
    a = area(1, ["....#..."], [trainer(1, 0, 0, EAST, 6)])
    got = Walker(StaticMaps([a])).sight_tiles(1, a.objects[0])
    assert got == [(1, 0, EAST, 1), (2, 0, EAST, 2), (3, 0, EAST, 3)]


def test_sight_stops_at_an_object_but_not_at_a_removable_one():
    ball = npc(9, 3, 0, script=7001)
    a = area(1, ["......"], [trainer(1, 0, 0, EAST, 5), npc(8, 3, 0)])
    assert [t[0] for t in Walker(StaticMaps([a])).sight_tiles(1, a.objects[0])] == [1, 2]
    b = area(1, ["......"], [trainer(1, 0, 0, EAST, 5), ball])
    w = Walker(StaticMaps([b]))
    assert [t[0] for t in w.sight_tiles(1, b.objects[0])] == [1, 2, 3, 4, 5]
    assert [t[0] for t in w.sight_tiles(1, b.objects[0], removable=False)] == [1, 2]


def test_sight_follows_the_movement_type_and_range():
    a = area(1, ["....."] * 5, [trainer(1, 2, 2, SOUTH, 1, movement=12)])  # looks N, W and E
    tiles = {(x, z) for x, z, _, _ in Walker(StaticMaps([a])).sight_tiles(1, a.objects[0])}
    assert tiles == {(2, 1), (1, 2), (3, 2)}


def test_a_trainer_that_looks_all_around():
    a = area(1, ["....."] * 5, [trainer(1, 2, 2, SOUTH, 1, trainer_type=2)])
    tiles = {(x, z) for x, z, _, _ in Walker(StaticMaps([a])).sight_tiles(1, a.objects[0])}
    assert tiles == {(2, 1), (2, 3), (1, 2), (3, 2)}


def test_a_two_trainer_corridor_forces_the_order():
    # alcoves at (3, 0) and (8, 0): each trainer looks south onto the corridor
    rows = ["###.####.###", "............", "############"]
    a = area(1, rows, [trainer(1, 3, 0, SOUTH, 2), trainer(2, 8, 0, SOUTH, 2)])
    maps = StaticMaps([a])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 11, 1),))
    found = runs(Walker(maps, spec.members), spec)
    assert [r.sequence for r in found] == [((2,), (1,))]
    facts = chain_facts(maps, Walker(maps, spec.members), spec, (1, 2))
    assert facts["orders"] == [[[2], [1]]] and facts["leave_gaps"] == [[]]


def test_a_warp_out_of_the_chain_is_a_leave_gap():
    rows = ["###.####.#.#", "............", "############"]
    objs = [trainer(1, 3, 0, SOUTH, 2), trainer(2, 8, 0, SOUTH, 2)]
    exit_early = area(1, rows, objs, [Warp(0, 10, 0, 9, 0)])  # an alcove beside the start
    far = area(9, ["..."], warps=[Warp(0, 0, 0, 1, 0)])
    maps = StaticMaps([exit_early, far])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 11, 1),))
    facts = chain_facts(maps, Walker(maps, spec.members), spec, (1, 2))
    assert facts["leave_gaps"] == [[1]]
    behind = area(1, rows, objs, [Warp(0, 0, 1, 9, 0)])  # at the far end, behind trainer 1
    maps = StaticMaps([behind, far])
    facts = chain_facts(maps, Walker(maps, spec.members), spec, (1, 2))
    assert facts["leave_gaps"] == [[]]


def test_a_self_loop_warp_is_a_step_not_a_loop():
    a = area(1, ["........"], warps=[Warp(0, 3, 0, 1, 0)])
    seen = Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen
    assert Pos(1, 3, 0) in seen and Pos(1, 7, 0) in seen


def test_field_move_objects_block_until_the_badge_and_move_remove_them():
    rock, stone = npc(5, 3, 0, script=10001), boulder(6, 3, 0)
    tree = npc(7, 3, 0, script=10000)
    far = Pos(1, 6, 0)
    a, b = area(1, ["......."], [rock]), area(1, ["......."], [stone])
    c = area(1, ["......."], [tree])
    smash = FieldMoves(rock_smash=True)
    assert far in Walker(StaticMaps([a]), moves=smash).walk(Pos(1, 0, 0)).seen
    assert far not in Walker(StaticMaps([a]), moves=smash).walk(Pos(1, 0, 0), removable=False).seen
    assert far not in Walker(StaticMaps([b]), moves=smash).walk(Pos(1, 0, 0)).seen
    # Strength pushes the boulder, never removes it: it ends on the last tile, (6, 0)
    pushed = Walker(StaticMaps([b]), moves=FieldMoves(strength=True)).walk(Pos(1, 0, 0)).seen
    assert Pos(1, 5, 0) in pushed and far not in pushed
    assert far not in Walker(StaticMaps([c]), moves=smash).walk(Pos(1, 0, 0)).seen
    assert far in Walker(StaticMaps([c]), moves=FieldMoves(cut=True)).walk(Pos(1, 0, 0)).seen


STRONG = FieldMoves(strength=True)


def test_strength_pushes_a_boulder_into_a_free_tile():
    # facing the boulder turns Strength on (scripts_field_moves.s:183-208); each step into it
    # then pushes it one tile while the player walks on the spot (ov5_021DFB54.c:500-540, :572-578)
    a = area(1, ["......"], [boulder(6, 2, 0)])
    w = Walker(StaticMaps([a]), moves=STRONG).walk(Pos(1, 0, 0))
    on = frozenset({STRENGTH_ACTIVE})
    assert (Pos(1, 1, 0), on, ((1, 6, 3, 0, 0),)) in {(s.pos, s.flags, s.moved) for s in w.states}
    assert {p.x for p in w.seen} == {0, 1, 2, 3, 4}  # it stops against the map's end at (5, 0)
    assert max(s.moved for s in w.states if s.moved) == ((1, 6, 5, 0, 0),)
    # without the move the script does not turn Strength on and the boulder blocks
    plain = Walker(StaticMaps([a])).walk(Pos(1, 0, 0))
    assert {p.x for p in plain.seen} == {0, 1} and all(not s.moved for s in plain.states)
    # Strength on (a fight's start state) without the move still pushes: the flag decides
    held = Walker(StaticMaps([a])).walk(Pos(1, 1, 0), flags=on)
    assert Pos(1, 4, 0) in held.seen


def test_no_push_into_a_wall_or_an_object():
    on = frozenset({STRENGTH_ACTIVE})
    wall = area(1, ["...#."], [boulder(6, 2, 0)])
    w = Walker(StaticMaps([wall]), moves=STRONG).walk(Pos(1, 1, 0), flags=on)
    assert {p.x for p in w.seen} == {0, 1} and all(not s.moved for s in w.states)
    other = area(1, ["....."], [boulder(6, 2, 0), npc(7, 3, 0)])
    w = Walker(StaticMaps([other]), moves=STRONG).walk(Pos(1, 1, 0), flags=on)
    assert {p.x for p in w.seen} == {0, 1} and all(not s.moved for s in w.states)
    two = area(1, ["....."], [boulder(6, 2, 0), boulder(7, 3, 0)])  # a boulder is an object too
    w = Walker(StaticMaps([two]), moves=STRONG).walk(Pos(1, 1, 0), flags=on)
    assert {p.x for p in w.seen} == {0, 1} and all(not s.moved for s in w.states)


def test_a_pushed_boulder_blocks_a_corridor_and_a_sight_line():
    # the boulder pushed from (1, 1) to (2, 1) sits in the corridor: it frees its event tile and
    # blocks the player and trainer 9's line (an object, sub_02063F00 / IsPathInterrupted)
    rows = ["##.##", "....."]
    a = area(1, rows, [boulder(6, 1, 1)])
    moved = ((1, 6, 2, 1, 0),)
    w = Walker(StaticMaps([a])).walk(Pos(1, 0, 1), moved=moved)
    assert set(w.seen) == {Pos(1, 0, 1), Pos(1, 1, 1)}
    line = area(1, ["......"], [boulder(6, 1, 0), trainer(9, 5, 0, WEST, 5)])
    walker = Walker(StaticMaps([line]))
    assert [t[0] for t in walker.sight_tiles(1, line.objects[1])] == [4, 3, 2]
    blocked = walker.sight_tiles(1, line.objects[1], moved=((1, 6, 3, 0, 0),))
    assert [t[0] for t in blocked] == [4]


def test_re_entry_resets_the_boulders_and_strength():
    # a warp re-creates the objects at their event tiles (field_map_change.c:341-360) and clears
    # FLAG_STRENGTH_ACTIVE (field_map_change_flags.c:68): after the round trip the boulder is back
    # on (3, 0) and, without the move, no longer moves
    here = area(1, ["......"], [boulder(6, 3, 0)], [Warp(0, 0, 0, 2, 0)])
    there = area(2, ["..."], warps=[Warp(0, 0, 0, 1, 0)])
    on = frozenset({STRENGTH_ACTIVE})
    w = Walker(StaticMaps([here, there])).walk(Pos(1, 2, 0), flags=on)
    assert any(s.moved == ((1, 6, 4, 0, 0),) for s in w.states)
    assert all(s.moved == () and not s.flags for s in w.states if s.pos.header == 2)
    back = [s for s in w.states if s.pos == Pos(1, 0, 0)]
    assert any(s.moved == () and not s.flags for s in back)
    after = Walker(StaticMaps([here, there])).walk(Pos(1, 0, 0), flags=frozenset())
    assert {p.x for p in after.seen if p.header == 1} == {0, 1, 2}


def test_a_rock_before_the_coal_badge_blocks():
    # Oreburgh Mine comes before Roark: no Coal Badge, so a Rock Smash rock is a wall
    # (FieldMoves_CheckRockSmash, field_move_tasks.c:545; scripts_field_moves.s:134).
    a = area(1, ["......."], [npc(5, 3, 0, script=10001)])
    roark = Walker(StaticMaps([a]), moves=FieldMoves.for_split("Roark"))
    assert Pos(1, 6, 0) not in roark.walk(Pos(1, 0, 0)).seen
    after = Walker(StaticMaps([a]), moves=FieldMoves.for_split("Gardenia"))
    assert Pos(1, 6, 0) in after.walk(Pos(1, 0, 0)).seen


def test_a_warp_into_a_sight_line_starts_that_fight():
    # the arrival tile (2, 0) of header 2 lies in trainer 7's line
    here = area(1, ["........"], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["........"], [trainer(7, 6, 0, WEST, 4)], [Warp(0, 2, 0, 1, 0)])
    w = Walker(StaticMaps([here, there])).walk(Pos(1, 0, 0))
    assert w.spot[(7,)].how == "arrival" and w.spot[(7,)].pos == Pos(2, 2, 0)
    assert all(p.header == 1 for p in w.seen)  # the walk never stands in header 2


def test_the_start_tile_is_checked_too():
    a = area(1, ["......"], [trainer(1, 4, 0, WEST, 3)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 2, 0))
    # trainer 1 (object 1) walks one step up to the player: (4, 0) -> (3, 0)
    spotted = Sighting(Pos(1, 2, 0), "arrival", moved=((1, 1, 3, 0),))
    assert w.spot == {(1,): spotted} and set(w.seen) == {Pos(1, 2, 0)}


def test_a_defeated_trainer_no_longer_sees():
    a = area(1, ["......"], [trainer(1, 5, 0, WEST, 4)])
    w = Walker(StaticMaps([a]))
    assert (1,) in w.walk(Pos(1, 0, 0)).spot
    assert w.walk(Pos(1, 0, 0), {1}).spot == {} and Pos(1, 4, 0) in w.walk(Pos(1, 0, 0), {1}).seen


def test_a_door_tile_with_the_collision_bit_is_entered():
    here = area(1, ["...D..."], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["..D.."], warps=[Warp(0, 2, 0, 1, 0)])
    seen = Walker(StaticMaps([here, there])).walk(Pos(1, 0, 0)).seen
    assert Pos(2, 2, 0) in seen and Pos(2, 4, 0) in seen
    wall = area(1, ["...#..."], warps=[Warp(0, 3, 0, 2, 0)])  # a collision tile that is no door
    assert all(p.header == 1 and p.x < 3 for p in Walker(StaticMaps([wall, there])).walk(
        Pos(1, 0, 0)).seen)  # fmt: skip


def test_an_object_behind_a_table_tile_can_be_talked_to():
    a = area(1, ["..T.."], [npc(5, 3, 0)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 1, 0))
    assert w.talk == {(1, 5): Pos(1, 1, 0)}
    no_table = area(1, ["....."], [npc(5, 3, 0)])
    assert Walker(StaticMaps([no_table])).walk(Pos(1, 0, 0)).talk == {(1, 5): Pos(1, 2, 0)}


def test_a_trainer_behind_a_table_starts_a_talk_fight():
    a = area(1, ["..T.."], [trainer(9, 3, 0, SOUTH, 0, trainer_type=0)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 1, 0))
    assert w.spot[(9,)] == Sighting(Pos(1, 1, 0), "talk")
    assert Walker(StaticMaps([a])).talk_tiles(Pos(1, 0, 0), 1, 9) == [(1, 0)]


def test_a_walking_member_raises_and_a_walking_non_member_stands_for_its_box():
    walker = trainer(3, 0, 0, EAST, 4, movement=41)  # range 0: its box is its event tile
    a = area(1, ["......"], [walker])
    with pytest.raises(MapReadError, match="walks"):
        Walker(StaticMaps([a]), members={3}).sight_tiles(1, walker)
    other = Walker(StaticMaps([a]), members={4})
    assert other.sight_tiles(1, walker) == [] and other.sight_at(1, 1, 0, frozenset()) == []
    # unbeaten: its lines from the box (every facing) close (1-4, 0); beaten: only its tile
    assert other.walk(Pos(1, 4, 0)).held
    beaten = Walker(StaticMaps([a]), members={4}, beaten={3})
    assert Pos(1, 1, 0) in beaten.walk(Pos(1, 5, 0)).seen
    assert Pos(1, 0, 0) not in beaten.walk(Pos(1, 5, 0)).seen
    assert (3,) not in beaten.walk(Pos(1, 5, 0)).spot  # not even talked to


def test_a_matching_coord_raises_unless_the_caller_models_it():
    a = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 0, 0)])  # var 0 always matches
    with pytest.raises(MapReadError, match="not modelled"):
        Walker(StaticMaps([a])).walk(Pos(1, 0, 0))
    passed = Walker(StaticMaps([a]), coord_rule=lambda h, c, m: "pass")
    w = passed.walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in w.seen and w.crossed[0][1].script == 7
    blocked = Walker(StaticMaps([a]), coord_rule=lambda h, c, m: "block")
    assert Pos(1, 3, 0) not in blocked.walk(Pos(1, 0, 0)).seen


def test_only_the_first_matching_coord_of_a_tile_runs():
    # sub_0203CC14 (src/unk_0203C954.c:217-221) returns the first coord in event order whose
    # var matches; the second coord on the tile never runs
    def rule(h, c, m):
        return "pass" if c.script == 7 else "block"

    first = [Coord(7, 3, 0, 1, 1, 0, 0), Coord(8, 3, 0, 1, 1, 0, 0)]
    w = Walker(StaticMaps([area(1, ["......"], coords=first)]), coord_rule=rule)
    walk = w.walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in walk.seen and [c.script for _, c in walk.crossed] == [7]
    second = Walker(StaticMaps([area(1, ["......"], coords=first[::-1])]), coord_rule=rule)
    assert Pos(1, 3, 0) not in second.walk(Pos(1, 0, 0)).seen
    # a first coord on an unknown var may not run: the matching one after it still applies
    unknown = [Coord(7, 3, 0, 1, 1, 1, 0x4091), Coord(8, 3, 0, 1, 1, 0, 0)]
    w = Walker(StaticMaps([area(1, ["......"], coords=unknown)]), coord_rule=rule)
    assert Pos(1, 3, 0) not in w.walk(Pos(1, 0, 0)).seen


def test_a_coord_by_a_known_or_unknown_var():
    a = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 1, 0x4091)])
    # var not known: the script may run (a battle, a move of the player), so the tile blocks
    unknown = Walker(StaticMaps([a])).walk(Pos(1, 0, 0))
    assert Pos(1, 3, 0) not in unknown.seen and Pos(1, 5, 0) not in unknown.seen
    passed = Walker(StaticMaps([a]), coord_rule=lambda h, c, m: "pass").walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in passed.seen and len(passed.crossed) == 1
    off = Walker(StaticMaps([a]), vars={0x4091: 0}).walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in off.seen
    with pytest.raises(MapReadError):
        Walker(StaticMaps([a]), vars={0x4091: 1}).walk(Pos(1, 0, 0))


def test_a_spin_past_a_pc_tile_gives_no_pc_spot():
    # the spin north from (1, 3) crosses (1, 2), the only tile beside the PC at (0, 2), with no
    # input there: the player never stands next to the PC
    rows = ["...", "#n#", "Pn#", "#n#", "..."]
    maps = StaticMaps([area(1, rows)])
    w = Walker(maps).walk(Pos(1, 1, 4))
    assert Pos(1, 1, 2) in w.seen and Pos(1, 1, 0) in w.seen
    assert pc_spots(maps, w, [1]) == set()


def test_a_pc_tile_is_reached_from_a_neighbour():
    a = area(1, ["..P..", "....."])
    w = Walker(StaticMaps([a])).walk(Pos(1, 0, 1))
    assert pc_reachable(StaticMaps([a]), w, [1])
    walled = StaticMaps([area(1, [".#P#.", "#####"])])
    assert not pc_reachable(walled, Walker(walled).walk(Pos(1, 0, 0)), [1])


def test_a_pair_member_can_be_talked_to_outside_both_sight_lines():
    # 11 and 12 look south at one tile each; the tiles beside them are outside both lines
    a = area(
        1,
        ["......."] * 3,
        [trainer(11, 1, 0, SOUTH, 1), trainer(12, 4, 0, SOUTH, 1)],
    )
    w = Walker(StaticMaps([a]), members={11, 12})
    assert w.talk_tiles(Pos(1, 0, 2), 1, 11) == [(0, 0), (2, 0)]
    assert w.talk_tiles(Pos(1, 0, 2), 1, 12) == [(3, 0), (5, 0)]
    assert w.talk_tiles(Pos(1, 0, 2), 1, 11, {12}) == [(0, 0), (2, 0)]


def test_a_var_gated_coord_closes_the_only_way_out():
    # The Wayward walk-out: a coord that runs while the partner var is 0 sits on the only tile
    # out of the start pocket. It takes the player (leave or step back), so alone the walk ends
    # there; with the partner (var 1) it does not run and the trainer beyond is reached.
    walkout = Coord(WAYWARD_WALKOUT, 3, 0, 1, 1, 0, WAYWARD_PARTNER_VAR)
    maps = StaticMaps(
        [area(WAYWARD_HEADER, ["........"], [trainer(383, 7, 0, WEST, 1)], (), [walkout])]
    )
    start = Pos(WAYWARD_HEADER, 0, 0)
    alone = _wayward_walker(maps, False).walk(start)
    assert alone.spot == {} and set(alone.seen) == {Pos(WAYWARD_HEADER, x, 0) for x in range(3)}
    joined = _wayward_walker(maps, True).walk(start)
    assert (383,) in joined.spot and Pos(WAYWARD_HEADER, 3, 0) in joined.seen


def test_an_unknown_running_wayward_coord_raises():
    other = Coord(9, 3, 0, 1, 1, 0, WAYWARD_PARTNER_VAR)
    maps = StaticMaps([area(WAYWARD_HEADER, ["......"], coords=[other])])
    with pytest.raises(MapReadError, match="not modelled"):
        _wayward_walker(maps, False).walk(Pos(WAYWARD_HEADER, 0, 0))
    assert (
        Pos(WAYWARD_HEADER, 5, 0)
        in _wayward_walker(maps, True).walk(Pos(WAYWARD_HEADER, 0, 0)).seen
    )


# --- the walk-fix round (review of 22 Nov): each rule on a hand-built map -----------------------


def test_two_trainers_on_one_sight_tile_are_one_double_battle():
    # 1 (object 0) looks south, 2 (object 1) looks west; both see (2, 2), the only way east.
    rows = ["##.###", "##.###", "......", "##.###"]
    a = area(1, rows, [trainer(1, 2, 0, SOUTH, 2), trainer(2, 4, 2, WEST, 2)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 0, 2))
    assert set(w.spot) == {(1, 2)} and w.spot[(1, 2)].pos == Pos(1, 2, 2)  # one VS2 fight
    alone = Walker(StaticMaps([a]), two_alive=False).walk(Pos(1, 0, 2))
    assert set(alone.spot) == {(1,)}  # one live Pokemon: the first in object order, alone
    maps = StaticMaps([a])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 0, 2),))
    facts = chain_facts(maps, Walker(maps, spec.members), spec, (1, 2))
    assert facts["orders"] == [[[1, 2]]] and facts["fight_headers"] == [[1]]


def test_a_double_battle_trainer_fights_once_with_both_objects_or_not_at_all():
    twins = [trainer(5, 5, 0, WEST, 3), MapObject(1, 1, 0, 1, 5000 + 5 - 1, WEST, 3, 4, 0)]
    a = area(1, ["......"], twins)
    w = Walker(StaticMaps([a]), doubles={5}).walk(Pos(1, 0, 0))
    assert set(w.spot) == {(5,)}
    one = Walker(StaticMaps([a]), doubles={5}, two_alive=False).walk(Pos(1, 0, 0))
    assert one.spot == {} and Pos(1, 3, 0) in one.seen  # no fight: the player walks on


def test_one_live_pokemon_fights_two_seers_of_one_tile_as_singles_in_turn():
    # Ruling (2): the walk depends on the live count. Two live: one VS2 battle; one live: the
    # first seer in object order, then the second (still seeing the tile), two singles.
    rows = ["##.###", "##.###", "......", "##.###"]
    a = area(1, rows, [trainer(1, 2, 0, SOUTH, 2), trainer(2, 4, 2, WEST, 2)])
    maps = StaticMaps([a])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 0, 2),))
    two = chain_facts(maps, Walker(maps, spec.members), spec, (1, 2))
    one = chain_facts(maps, Walker(maps, spec.members, two_alive=False), spec, (1, 2))
    assert two["orders"] == [[[1, 2]]] and two["leave_gaps"] == [[]]
    assert one["orders"] == [[[1], [2]]] and one["fight_headers"] == [[1, 1]]
    after = Walker(maps, two_alive=False).walk(Pos(1, 2, 2), {1}, y=0)
    assert set(after.spot) == {(2,)} and after.spot[(2,)].pos == Pos(1, 2, 2)


def test_double_trainer_talk_reads_the_party_not_the_partner():
    # Talking to a double-battle trainer reads the party (ScrCmd_CheckHasTwoAliveMons,
    # src/scrcmd_trainer.c:283-289, from scripts_battles.s:21-22); sight takes the partner flag
    # as two (field_control.c:182-186).
    talk_only = area(1, ["......"], [trainer(5, 3, 0, SOUTH, 0, trainer_type=0)])
    maps = StaticMaps([talk_only])
    for two, partner, fights in ((True, False, True), (False, True, False), (False, False, False)):
        w = Walker(maps, doubles={5}, two_alive=two, partner=partner).walk(Pos(1, 0, 0))
        assert ((5,) in w.spot) is fights and (1, 5) in w.talk
    twins = [trainer(5, 5, 0, WEST, 3), MapObject(1, 1, 0, 1, 5000 + 5 - 1, WEST, 3, 4, 0)]
    seen = StaticMaps([area(1, ["......"], twins)])
    w = Walker(seen, doubles={5}, two_alive=False, partner=True).walk(Pos(1, 0, 0))
    assert set(w.spot) == {(5,)}


def test_live_table_zero_step_gap_is_locked():
    # One live Pokemon and two seers of one tile: 1, then 2 at once on the fight tile, before
    # any input (the sight check runs before movement and the menu, field_control.c:188 <
    # :198, :331-333): the gap is locked and offers no count change.
    rows = ["##.###", "##.###", "......", "##.###"]
    a = area(1, rows, [trainer(1, 2, 0, SOUTH, 2), trainer(2, 4, 2, WEST, 2)])
    maps = StaticMaps([a])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 0, 2),))
    walkers = {c: Walker(maps, spec.members, two_alive=c == 2) for c in (1, 2)}
    table = live_table(maps, walkers, spec, ())
    assert table == [
        {"fought": [], "live": 1, "next": [[1]], "headers": [1]},
        {"fought": [], "live": 2, "next": [[1, 2]], "headers": [1]},
        {"fought": [[1]], "live": 1, "next": [[2]], "headers": [1], "gap": ["locked"]},
        {"fought": [[1]], "live": 2, "next": [[2]], "headers": [1], "gap": ["locked"]},
    ]


def test_live_table_raises_for_a_count_one_walk_with_the_partner():
    # with the partner sight counts two whatever the party holds (field_control.c:182-184):
    # dropping the double-battle trainers at one live would be wrong
    maps = StaticMaps([area(1, ["......"], [trainer(1, 5, 0, WEST, 3)])])
    spec = ChainSpec((1,), (1,), (Pos(1, 0, 0),))
    walkers = {c: Walker(maps, spec.members, two_alive=c == 2, partner=True) for c in (1, 2)}
    with pytest.raises(MapReadError, match="partner"):
        live_table(maps, walkers, spec, ())


def test_live_table_revive_row_reopens_a_double_battle_trainer():
    # 1 guards the corridor; twins 5 (a double-battle trainer, objects at (6, 1) and (7, 1))
    # watch row 0 east of it. One live after 1: the twins are passed and the chain ends; a
    # revive on any tile reached (the start menu, field_control.c:331-333) reopens them.
    rows = ["........", "........"]
    objs = [
        trainer(1, 2, 1, NORTH, 1),
        trainer(5, 6, 1, NORTH, 1),
        MapObject(7, 7, 0, 1, 5000 + 5 - 1, NORTH, 1, 7, 1),
    ]
    maps = StaticMaps([area(1, rows, objs)])
    spec = ChainSpec((1, 5), (1,), (Pos(1, 0, 0),))
    walkers = {c: Walker(maps, spec.members, doubles={5}, two_alive=c == 2) for c in (1, 2)}
    table = {
        (tuple(map(tuple, r["fought"])), r["live"], r.get("at", "")): r["next"]
        for r in live_table(maps, walkers, spec, {5})
    }
    assert table[((), 1, "")] == table[((), 2, "")] == [[1]]
    assert table[(((1,),), 2, "")] == [[5]]
    assert table[(((1,),), 1, "")] == []  # one live: nothing left to fight
    assert table[(((1,),), 1, "bag")] == [[5]]


def test_two_start_tiles_split_by_a_ledge_give_two_histories():
    # Trainer 1 (all directions, sight 1) sees (3, 1) from the start side and (5, 1) from the
    # far side, which the player reaches only down the one-way ledge at (1, 2). After a fight
    # from (3, 1) the warp at (0, 1) is reached, after one from (5, 1) it is not (the ledge
    # cannot be climbed, trainer 1 stands on (4, 1)): the leave at gap 1 needs both.
    rows = ["#########", ".........", "#v######.", ".........", ".........", "######.##"]
    objs = [trainer(1, 4, 1, SOUTH, 1, trainer_type=2), trainer(2, 6, 5, NORTH, 1)]
    a = area(1, rows, objs, [Warp(0, 0, 1, 9, 0)])
    far = area(9, ["..."], warps=[Warp(0, 0, 0, 1, 0)])
    maps = StaticMaps([a, far])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 2, 1),))
    w = Walker(maps, spec.members)
    first = w.walk(Pos(1, 2, 1))
    assert {s.pos for s in first.starts[(1,)]} == {Pos(1, 3, 1), Pos(1, 5, 1)}
    after = {s.pos: w.walk(s.pos, {1}, y=s.y, flags=s.flags) for s in first.starts[(1,)]}
    assert 9 in after[Pos(1, 3, 1)].headers() and 9 not in after[Pos(1, 5, 1)].headers()
    facts = chain_facts(maps, w, spec, (1, 2))
    assert facts["orders"] == [[[1], [2]], [[2], [1]]]
    assert facts["leave_gaps"] == [[], []]  # the first sighting alone gave [[1], []]
    (run,) = [r for r in runs(w, spec) if r.sequence == ((1,), (2,))]
    assert len(run.states[1]) == 2  # the walks from both start tiles


def test_a_gap_is_locked_when_one_start_tile_is_held():
    # One live Pokemon. Trainer 1 (all directions, sight 1) sees (3, 1) and (5, 1); trainer 2
    # (south, sight 1) sees (5, 1) too, so from (5, 1) the sight check starts 2 at once after
    # 1 (object order, trainer_encounter.c:94-131): no input in the gap. From (3, 1) the
    # player walks round to (5, 1). Locked is a restriction: the union over the starts.
    rows = ["#####.###", ".........", "#.#####.#", "........."]
    objs = [trainer(1, 4, 1, SOUTH, 1, trainer_type=2), trainer(2, 5, 0, SOUTH, 1)]
    maps = StaticMaps([area(1, rows, objs)])
    spec = ChainSpec((1, 2), (1,), (Pos(1, 0, 1),))
    one = Walker(maps, spec.members, two_alive=False)
    starts = one.walk(Pos(1, 0, 1)).starts[(1,)]
    assert {s.pos for s in starts} == {Pos(1, 3, 1), Pos(1, 5, 1)}
    assert [one.walk(s.pos, {1}).held for s in sorted(starts)] == [False, True]
    facts = chain_facts(maps, one, spec, (1, 2))
    assert facts["orders"] == [[[1], [2]]] and facts["locked_gaps"] == [1]
    walkers = {c: Walker(maps, spec.members, two_alive=c == 2) for c in (1, 2)}
    table = live_table(maps, walkers, spec, ())
    after = [r for r in table if r["fought"] == [[1]]]
    assert after and all("locked" in r["gap"] for r in after)  # no change row: none has "at"


# A sighting trainer walks distance - 1 tiles and stops next to the player
# (src/trainer_encounter.c:596-630); a talk moves no one; a warp re-creates the objects at their
# event tiles (src/field_map_change.c:341-360). Rows of the next three tests: trainer 1 on
# (6, 0) looks west over (5..1, 0); the player comes up the west side to (0, 0) or the east side
# to (7, 0) (a talk tile).
APPROACH_ROWS = ["........", ".######.", "........"]


def _approach_walker(warps=(), extra=(), rows=APPROACH_ROWS):
    a = area(1, rows, [trainer(1, 6, 0, WEST, 5)], warps)
    return Walker(StaticMaps([a, *extra]), {1})


def test_a_sight_fight_frees_the_event_tile_and_blocks_the_tile_next_to_the_player():
    w = _approach_walker()
    starts = w.walk(Pos(1, 0, 2)).starts[(1,)]
    got = {(s.how, s.pos): s.moved for s in starts}
    assert got[("sight", Pos(1, 1, 0))] == ((1, 1, 2, 0),)  # 4 steps: (6, 0) -> (2, 0)
    assert got[("talk", Pos(1, 7, 0))] == ()  # a talk: the trainer stays on (6, 0)
    sight = next(s for s in starts if s.pos == Pos(1, 1, 0))
    after = w.walk(sight.pos, {1}, y=sight.y, flags=sight.flags, moved=sight.moved)
    assert Pos(1, 6, 0) in after.seen and Pos(1, 3, 0) in after.seen  # round by the east side
    assert Pos(1, 2, 0) not in after.seen  # the trainer stands there now
    talk = next(s for s in starts if s.how == "talk")
    still = w.walk(talk.pos, {1}, y=talk.y, flags=talk.flags, moved=talk.moved)
    assert Pos(1, 6, 0) not in still.seen and Pos(1, 2, 0) in still.seen
    # the DP carries it: the walks after fight 1 are one per start state, moved included
    spec = ChainSpec((1,), (1,), (Pos(1, 0, 2),))
    (run,) = runs(w, spec)
    assert sorted(Pos(1, 6, 0) in x.seen for x in run.states[1]) == [False, True]


def test_a_moved_trainer_blocks_a_line_and_its_freed_tile_opens_one():
    # Trainer 1 on (6, 1) spots the player on (1, 1) and stops on (2, 1), inside trainer 2's
    # line (north from (2, 4)), which now ends there; trainer 3's line (north from (6, 4)),
    # which ended at trainer 1, now runs on over (6, 1) and (6, 0).
    rows = ["........"] * 6
    objs = [trainer(1, 6, 1, WEST, 5), trainer(2, 2, 4, NORTH, 4), trainer(3, 6, 4, NORTH, 4)]
    a = area(1, rows, objs)
    w = Walker(StaticMaps([a]), {1, 2, 3})
    sight = next(
        s for s in w.walk(Pos(1, 0, 1)).starts[(1,)] if (s.how, s.pos) == ("sight", Pos(1, 1, 1))
    )
    assert sight.moved == ((1, 1, 2, 1),)
    line = {(x, z) for x, z, _, _ in w.sight_tiles(1, objs[1])}
    cut = {(x, z) for x, z, _, _ in w.sight_tiles(1, objs[1], moved=sight.moved)}
    assert line == {(2, 3), (2, 2), (2, 1), (2, 0)} and cut == {(2, 3), (2, 2)}
    line3 = {(x, z) for x, z, _, _ in w.sight_tiles(1, objs[2])}
    open3 = {(x, z) for x, z, _, _ in w.sight_tiles(1, objs[2], moved=sight.moved)}
    assert line3 == {(6, 3), (6, 2)} and open3 == {(6, 3), (6, 2), (6, 1), (6, 0)}
    after = w.walk(sight.pos, {1}, y=sight.y, flags=sight.flags, moved=sight.moved)
    assert Pos(1, 2, 0) not in {s.pos for s in after.starts.get((2,), ())}
    assert Pos(1, 6, 0) in {s.pos for s in after.starts[(3,)]}
    plain = w.walk(sight.pos, {1}, y=sight.y, flags=sight.flags)
    assert Pos(1, 2, 0) in {s.pos for s in plain.starts[(2,)]}
    assert Pos(1, 6, 0) not in {s.pos for s in plain.starts.get((3,), ())}


def test_leaving_and_re_entering_puts_the_trainer_back_on_its_event_tile():
    # a warp on (1, 3), off the way east, to header 2 and back: the arrival re-creates
    # trainer 1 on (6, 0)
    there = area(2, ["..."], warps=[Warp(0, 1, 0, 1, 0)])
    w = _approach_walker([Warp(0, 1, 3, 2, 0)], [there], [*APPROACH_ROWS, "...#####"])
    sight = next(s for s in w.walk(Pos(1, 0, 2)).starts[(1,)] if s.pos == Pos(1, 1, 0))
    after = w.walk(sight.pos, {1}, y=sight.y, flags=sight.flags, moved=sight.moved)
    moved = {st.pos for st in after.states if st.moved}
    back = {st.pos for st in after.states if not st.moved and st.pos.header == 1}
    assert Pos(1, 6, 0) in moved and Pos(1, 2, 0) not in moved
    assert back and Pos(1, 2, 0) in back and Pos(1, 6, 0) not in back
    assert all(not st.moved for st in after.states if st.pos.header == 2)


def test_a_vs2_fight_moves_both_and_a_double_trainers_partner_walks_alongside():
    # 1 looks east from (0, 0), 2 west from (6, 0): both see (3, 0) at distance 3 (VS2)
    rows = [".......", "###.###", "......."]
    a = area(1, rows, [trainer(1, 0, 0, EAST, 4), trainer(2, 6, 0, WEST, 4)])
    w = Walker(StaticMaps([a]), {1, 2})
    vs2 = {s.pos: s.moved for s in w.walk(Pos(1, 3, 2)).starts[(1, 2)]}
    assert vs2[Pos(1, 3, 0)] == ((1, 1, 2, 0), (1, 2, 4, 0))
    one = Walker(StaticMaps([a]), {1, 2}, two_alive=False)
    alone = {s.pos: s.moved for s in one.walk(Pos(1, 3, 2)).starts[(1,)]}
    assert alone[Pos(1, 3, 0)] == ((1, 1, 2, 0),)  # one live: trainer 1 alone walks up
    # twins (one trainer id, two objects 5 and 6): the partner walks the leader's distance
    rows = ["###.###", ".......", "......."]
    twins = [trainer(5, 6, 1, WEST, 4), MapObject(6, 6, 0, 1, 3000 + 5 - 1, WEST, 4, 6, 2)]
    t = Walker(StaticMaps([area(1, rows, twins)]), {5}, doubles={5})
    got = {s.pos: s.moved for s in t.walk(Pos(1, 3, 0)).starts[(5,)]}
    assert got[Pos(1, 3, 1)] == ((1, 5, 4, 1), (1, 6, 4, 2))


def test_gap_fields_are_absent_where_the_live_table_makes_them_order_dependent():
    # Two orders of 1, 2: a healer at the gap after 1 first only (order-dependent heal), a
    # leave at the gap after 2 first at one live only (count-dependent leave), no PC anywhere
    # (flat). The flat field must be the two-live value; a dependent one is None.
    def row(fought, live, nxt, gap):
        return {"fought": fought, "live": live, "next": nxt, "headers": [], "gap": gap}

    live_next = [
        {"fought": [], "live": 1, "next": [[1], [2]], "headers": []},
        {"fought": [], "live": 2, "next": [[1], [2]], "headers": []},
        row([[1]], 1, [[2]], ["heal"]),
        row([[1]], 2, [[2]], ["heal"]),
        row([[2]], 1, [[1]], ["leave"]),
        row([[2]], 2, [[1]], []),
    ]
    facts = {
        "orders": [[[1], [2]], [[2], [1]]],
        "leave_gaps": [[], []],
        "heal_gaps": [],
        "pc_gaps": [],
    }
    assert gap_fields(facts, None) == {"leave_gaps": [[], []], "heal_gaps": [], "pc_gaps": []}
    assert gap_fields(facts, live_next) == {"leave_gaps": None, "heal_gaps": None, "pc_gaps": []}
    with pytest.raises(MapReadError, match="pc_gaps"):
        gap_fields({**facts, "pc_gaps": [1]}, live_next)


def test_a_hidden_object_neither_blocks_nor_spots():
    # Ruling (1): an object whose hidden flag is set is not created on the map load
    # (map_object.c:607-624). Trainer 1 (flag 600) watches the corridor, an NPC (flag 601)
    # stands in it; a script-less object's flag field is no flag (map_object.c:2333-2337).
    objs = [
        trainer(1, 0, 1, EAST, 5)._replace(flag=600),
        npc(7, 3, 0)._replace(flag=601),
    ]
    rows = ["......", "#....."]
    a = area(1, rows, objs)
    shown = Walker(StaticMaps([a])).walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) not in shown.seen  # the NPC blocks row 0 ...
    assert (1,) in shown.spot  # ... and trainer 1 sees row 1
    hidden = Walker(StaticMaps([a]), flags={600, 601}).walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in hidden.seen and Pos(1, 5, 1) in hidden.seen and hidden.spot == {}
    assert (1, 7) not in hidden.talk
    none = MapObject(8, 8, 0, 0, 0xFFFF, 0, 0, 3, 0, 0, 601)  # no script: always created
    b = area(1, ["......"], [none])
    assert Pos(1, 5, 0) not in Walker(StaticMaps([b]), flags={601}).walk(Pos(1, 0, 0)).seen


def test_a_flag_set_on_entering_a_map_hides_objects_on_the_next_load():
    # The OnTransition script of header 2 sets flag 700 before header 1 is loaded again
    # (field_map_change.c:254-299, then :341-351): the NPC on (2, 0) is gone on the way back.
    one = area(1, ["....."], [npc(7, 2, 0)._replace(flag=700)], [Warp(0, 1, 0, 2, 0)])
    two = area(2, ["..."], warps=[Warp(0, 1, 0, 1, 0)])
    maps = StaticMaps([one, two])
    stay = Walker(maps).walk(Pos(1, 0, 0))
    assert Pos(1, 3, 0) not in stay.seen
    back = Walker(maps, on_enter={2: {700}}).walk(Pos(1, 0, 0))
    assert Pos(1, 3, 0) in back.seen and Pos(2, 0, 0) in back.seen
    # a fight started after the visit carries the flag on (Sighting.flags)
    guard = trainer(3, 4, 0, WEST, 1)
    one_t = area(1, ["....."], [npc(7, 2, 0)._replace(flag=700), guard], [Warp(0, 1, 0, 2, 0)])
    w = Walker(StaticMaps([one_t, two]), on_enter={2: {700}}).walk(Pos(1, 0, 0))
    assert w.spot[(3,)].flags == frozenset({700})


def test_a_ledge_is_one_way():
    a = area(1, ["...", "#v#", "..."])
    down = Walker(StaticMaps([a])).walk(Pos(1, 1, 0)).seen
    assert Pos(1, 1, 2) in down and Pos(1, 1, 1) not in down  # jumped over the ledge
    up = Walker(StaticMaps([a])).walk(Pos(1, 1, 2)).seen
    assert set(up) == {Pos(1, x, 2) for x in range(3)}  # no way back north


def test_water_needs_surf():
    a = area(1, [".~~."])
    assert Pos(1, 3, 0) not in Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen
    surf = Walker(StaticMaps([a]), moves=FieldMoves(surf=True)).walk(Pos(1, 0, 0)).seen
    assert Pos(1, 3, 0) in surf
    # the Fen Badge gives Surf; a partner refuses it (scripts_field_moves.s:302-303)
    assert FieldMoves.for_split("Byron").surf and not FieldMoves.for_split("Wake").surf
    assert not FieldMoves.for_split("Galactic", partner=True).surf


def test_a_directional_tile_blocks_one_side():
    # BLOCK_EASTWARD on (1, 0): leaving it eastward is blocked, and so is entering it westward
    a = area(1, [".e."])
    assert Pos(1, 2, 0) not in Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen
    assert Pos(1, 0, 0) in Walker(StaticMaps([a])).walk(Pos(1, 1, 0)).seen
    assert Pos(1, 1, 0) not in Walker(StaticMaps([a])).walk(Pos(1, 2, 0)).seen
    # a trainer's line stops at it too
    b = area(1, [".e..."], [trainer(1, 4, 0, WEST, 4)])
    assert [t[0] for t in Walker(StaticMaps([b])).sight_tiles(1, b.objects[0])] == [3, 2]


def test_a_spin_tile_carries_the_player_past_a_sight_line():
    # stepping north onto the spin tile (1, 3) carries the player to (1, 0); trainer 1 looks
    # at (1, 2) only, which the player crosses during the spin (no field input: the trainer
    # check, field_control.c:187, runs on the frame before the spin starts, :200 -> :712)
    rows = ["...", "#n#", "#n#", "#n#", "..."]
    a = area(1, rows, [trainer(1, 2, 2, WEST, 1)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 1, 4))
    assert Pos(1, 1, 0) in w.seen and (1,) not in w.spot
    # a spin tile entered by a step is checked first: a line onto (1, 3) spots the player
    b = area(1, rows, [trainer(1, 2, 3, WEST, 1)])
    assert (1,) in Walker(StaticMaps([b])).walk(Pos(1, 1, 4)).spot


def test_ice_slides_until_a_step_collides():
    a = area(1, [".iii#"])
    assert Pos(1, 3, 0) in Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen


def test_tiles_on_different_floors_are_not_connected():
    # (0, 0)-(1, 0) at height 0, (2, 0) at 20: adjacent but 20 apart (GetVerticalDirection)
    up = 20 * 4096
    floors = {(0, 0): (0,), (1, 0): (0,), (2, 0): (up,), (3, 0): (up,)}
    a = area(1, ["...."], heights=floors)
    assert Pos(1, 2, 0) not in Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen
    slope = {(0, 0): (0,), (1, 0): (19 * 4096,), (2, 0): (up + 18 * 4096,)}
    b = area(1, ["..."], heights=slope)
    assert Pos(1, 2, 0) in Walker(StaticMaps([b])).walk(Pos(1, 0, 0)).seen
    # a trainer on the upper floor does not see the player below on a two-plate bridge row
    t = MapObject(0, 0, 0, 1, 3000, WEST, 3, 3, 0, up)
    c = area(1, ["...."], [t], heights={**floors, (1, 0): (0, up), (2, 0): (0, up)})
    w = Walker(StaticMaps([c])).walk(Pos(1, 0, 0))
    assert w.spot == {} and Pos(1, 2, 0) in w.seen


def test_a_var_zero_coord_runs_only_when_its_value_is_zero():
    # FieldSystem_TryGetVar(0) is 0 (script_manager.c:444-452)
    zero = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 0, 0)])
    with pytest.raises(MapReadError, match="not modelled"):
        Walker(StaticMaps([zero])).walk(Pos(1, 0, 0))
    never = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 5, 0)])
    w = Walker(StaticMaps([never])).walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in w.seen and w.crossed == []
    assert Walker(StaticMaps([never])).coord_matches(Coord(7, 0, 0, 1, 1, 0x10, 0x10))


def test_a_turning_trainer_can_be_passed_or_fought():
    # 1 turns north and west (LOOK_NORTH_AND_WEST, 6): the player crosses its west line when it
    # faces north, or is spotted. A fixed trainer (LOOK_WEST, 16) always spots him.
    rows = ["#.#..", "....."]
    turning = area(1, rows, [trainer(1, 4, 1, WEST, 4, movement=6)])
    w = Walker(StaticMaps([turning])).walk(Pos(1, 1, 0))
    assert (1,) in w.spot and Pos(1, 0, 1) in w.seen
    fixed = area(1, rows, [trainer(1, 4, 1, WEST, 4, movement=16)])
    w = Walker(StaticMaps([fixed])).walk(Pos(1, 1, 0))
    assert (1,) in w.spot and Pos(1, 0, 1) not in w.seen


def test_a_turning_and_a_fixed_trainer_on_one_tile():
    # object 0 turns, object 1 is fixed: the tile starts 2 alone or the double, never a pass
    rows = ["##.##", "##.##", "....."]
    a = area(1, rows, [trainer(1, 2, 0, SOUTH, 2, movement=10), trainer(2, 4, 2, WEST, 2)])
    w = Walker(StaticMaps([a]))
    fights, passable = w.encounters(w.sight_at(1, 2, 2, frozenset()))
    assert fights == {(1, 2), (2,)} and not passable


def test_bdhc_heights_reads_the_plates_of_a_tile():
    # one land with two flat plates: x tiles 0-15 at height 0, 16-31 at height 32
    fx = 4096
    pts = [(-256 * fx, -256 * fx), (0, 256 * fx), (0, -256 * fx), (256 * fx, 256 * fx)]
    body = b"BDHC" + struct.pack("<6H", len(pts), 1, 2, 2, 1, 2)
    body += b"".join(struct.pack("<2i", *p) for p in pts)
    body += struct.pack("<3i", 0, fx, 0)
    body += struct.pack("<2i", 0, -32 * fx)
    body += struct.pack("<4H", 0, 1, 0, 0) + struct.pack("<4H", 2, 3, 0, 1)
    body += struct.pack("<iHH", 256 * fx, 2, 0)
    body += struct.pack("<2H", 0, 1)
    member = struct.pack("<4I", 0x800, 0, 0, len(body)) + bytes(0x800) + body
    h = bdhc_heights(member)
    assert h[0] == (0,) and h[31] == (32 * fx,) and h[5 * 32 + 20] == (32 * fx,)


def test_the_surfable_set_matches_the_decomp(decomp):
    src = (decomp / "src" / "map_tile_behavior.c").read_text(encoding="utf-8")
    hdr = decomp / "include" / "constants" / "field" / "map_tile_behaviors.h"
    body = hdr.read_text(encoding="utf-8").split("enum TileBehavior {")[1]
    names = re.findall(r"^\s+(TILE_BEHAVIOR_\w+)", body, re.MULTILINE)
    value = {n: i for i, n in enumerate(names)}
    flags = re.findall(r"\[(TILE_BEHAVIOR_\w+)\]\s*=\s*(TILE_BEHAVIOR_FLAG_\w+)", src)
    assert {value[n] for n, f in flags if "SURFABLE" in f} == SURFABLE
    assert value["TILE_BEHAVIOR_JUMP_SOUTH"] == 0x3B
    assert value["TILE_BEHAVIOR_SLIDE_EASTWARD"] == 0x40
    assert value["TILE_BEHAVIOR_BLOCK_EAST_AND_WEST"] == 0x4A


def test_the_encounter_set_matches_the_decomp(decomp):
    src = (decomp / "src" / "map_tile_behavior.c").read_text(encoding="utf-8")
    hdr = decomp / "include" / "constants" / "field" / "map_tile_behaviors.h"
    body = hdr.read_text(encoding="utf-8").split("enum TileBehavior {")[1]
    names = re.findall(r"^\s+(TILE_BEHAVIOR_\w+)", body, re.MULTILINE)
    value = {n: i for i, n in enumerate(names)}
    flags = re.findall(r"\[(TILE_BEHAVIOR_\w+)\]\s*=\s*(TILE_BEHAVIOR_FLAG_\w+)", src)
    assert {value[n] for n, f in flags if "ENCOUNTER" in f} == ENCOUNTER


def test_the_roaming_headers_match_the_decomp(decomp):
    src = (decomp / "src" / "roaming_pokemon.c").read_text(encoding="utf-8")
    table = src.split("RoamingPokemonRoutes[RI_MAX] = {")[1].split("};")[0]
    routes = re.findall(r"= (MAP_HEADER_\w+)", table)
    lines = (decomp / "generated" / "map_headers.txt").read_text(encoding="utf-8").split()
    assert len(routes) == 29
    assert {lines.index(r) for r in routes} == ROAMING_HEADERS  # 0-based: EVERYWHERE is 0


def _grass_area(rows, wild=None, objects=()):
    a = area(1, rows, objects)
    return Area(1, a.tiles, a.objects, wild={"grass": (10, 7)} if wild is None else wild)


def test_an_encounter_tile_needs_wild():
    # (1, 0) is tall grass; the walk without ``wild`` (wild = avoid) never ends a step on it
    a = _grass_area([".g."])
    plain = Walker(StaticMaps([a]))
    assert Pos(1, 2, 0) not in plain.walk(Pos(1, 0, 0)).seen
    assert plain.used == set() and plain.wild_tiles == {}  # a yes only is recorded
    w = Walker(StaticMaps([a]), moves=FieldMoves(wild=True))
    assert Pos(1, 2, 0) in w.walk(Pos(1, 0, 0)).seen
    assert w.wild_tiles == {(1, "grass"): {(1, 0)}}
    # a header without wild data (or a zero rate) has no encounter tile
    free = Walker(StaticMaps([_grass_area([".g."], wild={})]))
    assert Pos(1, 2, 0) in free.walk(Pos(1, 0, 0)).seen and free.used == set()
    # water is the surf table: grass data alone leaves it free for a surfer
    sea = Area(1, area(1, [".~."]).tiles, wild={"grass": (10, 7)})
    assert sea.encounter(1, 0) is None
    assert Area(1, sea.tiles, wild={"surf": (5, 30)}).encounter(1, 0) == "surf"


def test_wild_levels_name_the_tables_the_walk_steps_on():
    a = _grass_area(["gcg"])
    w = Walker(StaticMaps([a]), moves=FieldMoves(wild=True))
    w.walk(Pos(1, 0, 0))
    assert wild_levels(StaticMaps([a]), [w]) == [
        {
            "header": 1,
            "kind": "grass",
            "rate": 10,
            "max_level": 7,
            "tiles": 3,
            "examples": [[0, 0], [1, 0], [2, 0]],
        }
    ]


def test_a_chain_through_grass_has_an_avoid_case():
    # the corridor z 1 reaches both trainers only over the grass at (2, 1); the bypass z 3 has
    # no encounter tile but passes trainer 2 first
    rows = ["###.##.", "..g....", ".#####.", "......."]
    objs = [trainer(1, 3, 0, SOUTH, 1), trainer(2, 6, 0, SOUTH, 1)]
    maps = StaticMaps([_grass_area(rows, objects=objs)])
    facts = chain_group_facts(
        maps, _walkers_for(maps), FieldMoves(wild=True), ROCK_SPEC, (1, 2), ()
    )
    assert facts["field_moves"] == ["wild"]
    assert facts["orders"] == [[[1], [2]], [[2], [1]]]
    assert facts["wild_levels"][0]["header"] == 1 and facts["wild_levels"][0]["max_level"] == 7
    (case,) = facts["cases"]
    assert case["moves"] == [] and case["orders"] == [[[2], [1]]]


def test_the_split_badges_follow_the_split_order():
    from tools.extract_kaizo_docs import SPLITS

    assert [s for s, _ in SPLIT_BADGES] == list(SPLITS)
    # ``wild`` needs no badge: every split holds it (the setting decides, C8)
    assert FieldMoves.for_split("Roark") == FieldMoves(wild=True)
    assert FieldMoves.for_split("Wake") == FieldMoves(cut=True, rock_smash=True, wild=True)
    assert FieldMoves.for_split("Galactic", partner=True) == FieldMoves(
        cut=True, rock_smash=True, strength=True, wild=True
    )


# --- C5b: the fast walk equals the plain exact walk (brute-force references) ----------------------


class PlainWalker(Walker):
    """The reference: every sight table built afresh from the full ``moved`` (no key, no
    cache), as the exact search reads it."""

    def _sight_table(self, header, defeated, removable, flags, moved=()):
        moved = tuple(e for e in moved if e[0] == header)
        area = self.maps.area(header)
        table = {}
        for o in area.objects:
            t = o.trainer
            if t is None or t in defeated or self._ignored(header, o, flags):
                continue
            look = self.facing(o)
            if look is None:
                continue
            seer = Seer(o.index, t, look[1])
            gy = grid_y(area.object_height(o))
            for sx, sz, _, _ in self.sight_tiles(header, o, removable, flags, moved):
                table.setdefault((sx, sz), []).append((seer, gy))
        return table


def _random_map(rng, header, ids, warp_to=None):
    """A small cave: walls, two boulders, trainers ``ids`` (random facing and range, some
    turning), an npc; a free start tile (returned) and, with ``warp_to``, a warp there."""
    w, h = rng.randint(5, 8), rng.randint(4, 6)
    rows = ["".join("#" if rng.random() < 0.15 else "." for _ in range(w)) for _ in range(h)]
    free = [(x, z) for z, r in enumerate(rows) for x, c in enumerate(r) if c == "."]
    rng.shuffle(free)
    start, *spots = free
    objs = [boulder(19, *spots[0]), boulder(26, *spots[1])]
    for k, t in enumerate(ids):
        x, z = spots[2 + k]
        o = trainer(t, x, z, rng.randrange(4), rng.randint(1, 3), rng.choice((0, 0, 2, 6)))
        objs.append(o._replace(index=len(objs)))
    objs.append(npc(30, *spots[2 + len(ids)])._replace(index=len(objs)))
    objs = [o._replace(index=i) for i, o in enumerate(objs)]
    warps = [Warp(0, *spots[3 + len(ids)], warp_to, 0)] if warp_to is not None else []
    return area(header, rows, objs, warps), Pos(header, *start)


def _moved_trainer(rng, a, ids):
    """One trainer of ``ids`` walked 1..range tiles along its facing (a sight fight's
    approach) when the tiles are free, as a ``moved`` entry; or None."""
    t = rng.choice(ids)
    o = next(o for o in a.objects if o.trainer == t)
    dx, dz = ((0, -1), (0, 1), (-1, 0), (1, 0))[o.direction]
    k = rng.randint(1, max(1, o.sight))
    x, z = o.x + dx * k, o.z + dz * k
    if (x, z) not in a.tiles or a.object_at.get((x, z)):
        return None
    return (a.header, o.index, x, z), o.trainer


def _walk_facts(w):
    return (
        set(w.seen),
        set(w.states),
        {f: {s.state + (s.how,) for s in v} for f, v in w.starts.items()},
        set(w.talk),
        w.held,
        set(w.crossed),
    )


def _layout_facts(w):
    """What ``walk_layouts`` keeps of a walk (``layouts``: the number of states)."""
    n = w.layouts or len(w.states)
    starts = {f: {s.state + (s.how,) for s in v} for f, v in w.starts.items()}
    return set(w.seen), n, starts, set(w.talk), set(w.crossed)


def _random_case(rng, ids):
    """A random two-header case: the cave (header 1, a warp to header 2 and back, a blocking
    coord on an unknown var), Strength on or off, and maybe a moved trainer (defeated)."""
    a, start = _random_map(rng, 1, ids, warp_to=2)
    free = [p for p in a.tiles if not a.object_at.get(p) and p != (start.x, start.z)]
    coord = Coord(9, *rng.choice(free), 1, 1, 0, 0x4999)
    a = Area(1, a.tiles, a.objects, a.warps, (coord,))
    there = area(2, ["..."], warps=[Warp(0, 1, 0, 1, 0)])
    on = frozenset({STRENGTH_ACTIVE}) if rng.random() < 0.5 else frozenset()
    moved, defeated = (), ()
    m = _moved_trainer(rng, a, ids) if rng.random() < 0.6 else None
    if m is not None:
        moved, defeated = (m[0],), (m[1],)
    return StaticMaps([a, there]), start, on, moved, defeated


def test_c5b_the_cut_sight_key_equals_the_plain_walk_with_boulders_and_moved_trainers():
    # Random small maps with two Strength boulders, trainers (turning ones too) and a moved
    # trainer: the walk with the cut sight-table key and the shared tuples gives the same
    # states, tiles, fight starts, talks and coords as the reference that builds every table
    # from the full ``moved``, from the start and from fight starts.
    rng = random.Random(5)
    ids = (1, 2, 3)
    pushed = 0
    for _ in range(60):
        maps, start, on, moved, defeated = _random_case(rng, ids)
        for two in (True, False):
            fast = Walker(maps, ids, moves=STRONG, two_alive=two)
            plain = PlainWalker(maps, ids, moves=STRONG, two_alive=two)
            want = plain.walk(start, defeated, flags=on, moved=moved)
            assert _walk_facts(fast.walk(start, defeated, flags=on, moved=moved)) == _walk_facts(
                want
            )
            for f, sights in want.starts.items():
                for sg in sorted(sights, key=repr)[:2]:
                    d = set(defeated) | set(f)
                    kw = {"y": sg.y, "flags": sg.flags, "bridge": sg.bridge, "moved": sg.moved}
                    assert _walk_facts(fast.walk(sg.pos, d, **kw)) == _walk_facts(
                        plain.walk(sg.pos, d, **kw)
                    )
            assert fast.reached_headers() == plain.reached_headers()
            pushed += any(len(e) == 5 for st in want.states for e in st.moved)
    assert pushed > 20  # the boulders moved in many of them


def test_c5b_the_layout_walk_equals_the_walk_over_every_state():
    # ``walk_layouts`` (one label per tile state, the layouts as a bit mask) against
    # ``walk_from`` of the plain reference over the same seeds: the seeds are every fight start
    # of a first walk (each with its own layout: approaches, pushed boulders) plus the start.
    rng = random.Random(7)
    ids = (1, 2, 3)
    big = 0
    for _ in range(60):
        maps, start, on, moved, defeated = _random_case(rng, ids)
        plain = PlainWalker(maps, ids, moves=STRONG)
        first = plain.walk(start, defeated, flags=on, moved=moved)
        seeds = {s for s in sorted(first.states) if rng.random() < 0.3} or {
            _seed_of(plain, start, on, moved)
        }
        cases = [(seeds, frozenset(defeated))]
        for f, sights in first.starts.items():
            st = {_State(g.pos, g.y, g.bridge, g.flags, g.moved) for g in sights}
            cases.append((st, frozenset(defeated) | set(f)))
        for st, done in cases:
            fast = Walker(maps, ids, moves=STRONG)
            ref = PlainWalker(maps, ids, moves=STRONG)
            got = fast.walk_layouts(st, done)
            want = ref.walk_from(st, done)
            assert _layout_facts(got) == _layout_facts(want)
            assert got.states == set() and got.layouts == len(want.states)
            assert fast.reached_headers() == ref.reached_headers()
            big += len({s.moved for s in want.states}) > 3
    assert big > 20  # many layouts in many of them


def _seed_of(walker, pos, flags, moved):
    return _State(pos, walker.arrival_height(pos), False, flags, moved)


def test_c5b_one_walk_from_all_starts_equals_the_union_of_the_walks_per_start():
    # ``_wayward_union`` (one layout walk from every start of a prefix) against the union of
    # the plain walks per start (``_wayward_next``): tiles, states, fights with every start
    # state, talks and headers, on random Wayward-header maps with boulders and Strength.
    rng = random.Random(11)
    ids = (383, 443, 380)
    checked = 0
    for _ in range(40):
        a, start = _random_map(rng, WAYWARD_HEADER, ids)
        maps = StaticMaps([a])
        fast = _wayward_walker(maps, True)
        ref = _wayward_walker(maps, True)
        ref.__class__ = PlainWalker
        _, fights0 = _wayward_next(ref, start, frozenset())
        level = [  # up to 12 starts of each fight (with their layouts)
            (frozenset(f), set(rng.sample(sorted(sights, key=repr), min(12, len(sights)))))
            for f, sights in sorted(fights0.items())
        ]
        level.append((frozenset(), {start}))
        for done, starts in level:
            u, unext = _wayward_union(fast, starts, done)
            seen, n, talk, nxt = set(), set(), set(), {}
            for sg in starts:
                w, fights = _wayward_next(ref, sg, done)
                seen |= set(w.seen)
                n |= w.states
                talk |= set(w.talk)
                for g, ss in fights.items():
                    nxt.setdefault(g, set()).update(x.state for x in ss)
            assert set(u.seen) == seen and u.layouts == len(n) and set(u.talk) == talk
            assert {g: {x.state for x in ss} for g, ss in unext.items()} == nxt
            checked += 1
        assert fast.reached_headers() == ref.reached_headers()
    assert checked > 80


# --- the Eterna Gym tables -----------------------------------------------------------------------


def _gym_source(clock_marks=(), fountain_marks=()):
    def block(size, marks, state):
        return ", ".join("1" if (state, i) in marks else "0" for i in range(size))

    clock = ",\n".join(
        f"    [ETERNA_CLOCK_{s}] = {{\n        {block(CLOCK_SIDE**2, clock_marks, s)}\n    }}"
        for s in range(CLOCK_STATES)
    )
    fountain = ",\n".join(
        f"    [ETERNA_CLOCK_{s}] = {{ {block(FOUNTAIN_WIDTH, fountain_marks, s)} }}"
        for s in range(CLOCK_STATES)
    )
    return (
        "static ALIGN_4 const u8 sEternaGymClockCollision[ETERNA_CLOCK_MAX]"
        f"[ETERNA_CLOCK_DIAMETER * ETERNA_CLOCK_DIAMETER] = {{\n{clock}\n}};\n\n"
        "static const u8 sEternaGymFountainCollision[ETERNA_CLOCK_MAX][ETERNA_GYM_WIDTH] = "
        f"{{\n{fountain}\n}};\n"
    )


def test_the_clock_and_fountain_tables_are_read_from_the_source(tmp_path):
    path = tmp_path / "src" / "overlay008"
    path.mkdir(parents=True)
    (path / "gym_features.c").write_text(_gym_source({(2, 5), (4, 168)}, {(1, 0), (3, 20)}))
    clock, fountain = eterna_clock_tables(tmp_path)
    assert [sum(b) for b in clock] == [0, 0, 1, 0, 1] and clock[4][168] == 1
    assert [sum(b) for b in fountain] == [0, 1, 0, 1, 0] and fountain[3][20] == 1


def test_the_tables_must_have_five_blocks_of_the_right_size(tmp_path):
    path = tmp_path / "src" / "overlay008"
    path.mkdir(parents=True)
    (path / "gym_features.c").write_text(_gym_source().replace("0, 0, 0", "0, 0", 1))
    with pytest.raises(MapReadError):
        eterna_clock_tables(tmp_path)


def test_the_hour_hand_jump_tile_is_jumped_along_its_axis():
    column = Area(67, {(11, 17): 0, (11, 18): 0, (11, 19): 0})  # state 1's jump tile is (11, 18)
    zero = [[0] * CLOCK_SIDE**2] * CLOCK_STATES, [[0] * FOUNTAIN_WIDTH] * CLOCK_STATES
    r = gym_reach(column, zero, 1, (11, 19))
    assert r.seen == {(11, 19), (11, 17)}  # north-south jump over (11, 18)
    r0 = gym_reach(column, zero, 0, (11, 19))
    assert (11, 18) in r0.seen  # state 0 has no jump tile


def test_clock_collision_blocks_and_talk_and_coord_are_recorded():
    tiles = {(x, z): 0 for x in range(24) for z in range(24)}
    clock = [[0] * CLOCK_SIDE**2 for _ in range(CLOCK_STATES)]
    clock[0][(8 - 7) * CLOCK_SIDE + (5 - 5)] = 1  # tile (5, 8) blocked in state 0
    fountain = [[0] * FOUNTAIN_WIDTH for _ in range(CLOCK_STATES)]
    a = Area(
        67,
        tiles,
        (npc(3, 2, 2),),
        coords=(Coord(8, 4, 4, 1, 1, 0, 0),),
    )
    r = gym_reach(a, (clock, fountain), 0, (0, 0))
    assert (5, 8) not in r.seen and (5, 9) in r.seen
    assert set(r.talk) == {3} and r.coords == {8: (4, 4)}


def test_a_fight_coord_stops_the_gym_walk_until_its_fight_is_won():
    corridor = Area(67, {(x, 0): 0 for x in range(4)}, coords=(Coord(8, 2, 0, 1, 1, 0, 0),))
    zero = [[0] * CLOCK_SIDE**2] * CLOCK_STATES, [[0] * FOUNTAIN_WIDTH] * CLOCK_STATES
    r = gym_reach(corridor, zero, 0, (0, 0))
    assert r.coords == {8: (2, 0)} and (3, 0) not in r.seen
    won = gym_reach(corridor, zero, 0, (0, 0), {8})
    assert won.coords == {} and (3, 0) in won.seen


# --- field-move cases (B3 fix 3 C6) ---------------------------------------------------------
# A field move needs a party member that knows it, not only the badge (FindPartySlotWithMove,
# scripts_field_moves.s:27, :131, :189, :249, :318, :420; Party_HasMonWithMove,
# field_control.c:690-692): the facts come per case, the moves the walk uses.

# Trainer 1 sees (3, 1) and trainer 2 sees (6, 1) on the corridor z 1; the bypass z 3 is entered
# from the start only past the rock at (0, 2).
ROCK_ROWS = ["###.##.", ".......", ".#####.", "......."]
ROCK_SPEC = ChainSpec(members=(1, 2), headers=(1,), entries=(Pos(1, 0, 1),))


def _rock_maps():
    objs = [trainer(1, 3, 0, SOUTH, 1), trainer(2, 6, 0, SOUTH, 1), npc(5, 0, 2, script=10001)]
    return StaticMaps([area(1, ROCK_ROWS, objs)])


def _walkers_for(maps):
    return lambda moves: {c: Walker(maps, (1, 2), moves=moves, two_alive=c == 2) for c in (1, 2)}


def test_a_rock_between_two_trainers_gives_one_order_per_case():
    maps = _rock_maps()
    held = FieldMoves(cut=True, rock_smash=True)  # Cut held too: no tree, so not relevant
    facts = chain_group_facts(maps, _walkers_for(maps), held, ROCK_SPEC, (1, 2), ())
    assert facts["field_moves"] == ["rock_smash"]
    assert facts["orders"] == [[[1], [2]], [[2], [1]]]
    (case,) = facts["cases"]
    assert case["moves"] == [] and case["orders"] == [[[1], [2]]]
    assert case["leave_gaps"] == [[]] and case["locked_gaps"] == [] and case["live_next"] is None


def test_a_case_with_no_full_sequence_is_stored_empty():
    # the corridor is walled at (2, 1): only the bypass past the rock reaches either trainer
    rows = ["###.##.", "..#....", ".#####.", "......."]
    objs = [trainer(1, 3, 0, SOUTH, 1), trainer(2, 6, 0, SOUTH, 1), npc(5, 0, 2, script=10001)]
    maps = StaticMaps([area(1, rows, objs)])
    facts = chain_group_facts(
        maps, _walkers_for(maps), FieldMoves(rock_smash=True), ROCK_SPEC, (1, 2), ()
    )
    assert facts["orders"] == [[[2], [1]]]
    assert facts["cases"] == [
        {
            "moves": [],
            "orders": [],
            "leave_gaps": [],
            "fight_headers": [],
            "pc_gaps": [],
            "heal_gaps": [],
            "locked_gaps": [],
            "live_next": None,
        }
    ]
    with pytest.raises(MapReadError, match="no full sequence"):
        chain_group_facts(maps, _walkers_for(maps), FieldMoves(), ROCK_SPEC, (1, 2), ())


def test_relevant_moves_are_the_moves_a_walk_used():
    maps = _rock_maps()
    w = Walker(maps, (1, 2), moves=FieldMoves(cut=True, rock_smash=True, surf=True))
    w.walk(Pos(1, 0, 1))
    assert relevant_moves([w]) == {"rock_smash"}
    assert relevant_moves([Walker(maps, (1, 2))]) == set()
    held = FieldMoves.for_split("Wake")
    assert held.names == ("cut", "rock_smash", "wild")
    assert held.restrict(["rock_smash", "surf"]) == FieldMoves(rock_smash=True)
    assert move_subsets(("rock_smash", "cut")) == [
        (),
        ("cut",),
        ("rock_smash",),
        ("cut", "rock_smash"),
    ]
    with pytest.raises(MapReadError, match="unknown field moves"):
        held.restrict(["fly"])


def test_a_boulder_pushed_onto_an_unsmashed_rock_collides():
    # without Rock Smash in the case the rock at (4, 0) stays and stops the boulder at (3, 0)
    # (sub_02063EBC, map_object_move.c:614-621: an object on the target tile); with it the walk
    # takes the rock as smashed and the boulder goes on to the map's end
    on = frozenset({STRENGTH_ACTIVE})
    a = area(1, ["......"], [boulder(6, 2, 0), npc(7, 4, 0, script=10001)])
    for moves, last in ((STRONG, 3), (FieldMoves(strength=True, rock_smash=True), 5)):
        w = Walker(StaticMaps([a]), moves=moves).walk(Pos(1, 1, 0), flags=on)
        assert max(e[2] for s in w.states for e in s.moved) == last


# --- ROM-backed (slow) ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rom_files():
    roms = sorted((REPO_ROOT / ".cache" / "rom").glob("kaizo-*.nds"))
    if not roms:
        pytest.skip("no Kaizo ROM in .cache/rom")
    pytest.importorskip("ndspy")
    from tools.rom.nds import RomFiles

    return RomFiles.open(roms[0].read_bytes())


@pytest.fixture(scope="module")
def decomp():
    for path in (
        REPO_ROOT / "external" / "pokeplatinum",
        Path(os.environ.get("NUZLINE_DECOMP", ".")),
    ):
        if (path / "src" / "overlay008" / "gym_features.c").exists():
            return path
    pytest.skip("decomp not cloned (external/pokeplatinum or NUZLINE_DECOMP)")


@pytest.fixture(scope="module")
def derived_and_reached(rom_files, decomp):
    from tools.kaizo.gauntlet_map import derive_groups

    reached: dict[str, set[int]] = {}
    return derive_groups(rom_files, decomp, reached), reached


@pytest.fixture(scope="module")
def derived(derived_and_reached):
    return derived_and_reached[0]


def _persisted_feature_commands(decomp) -> set[str]:
    """The script commands that set the persisted map feature id: the ``ScrCmd_*`` handlers of
    ``src/scrcmd.c`` whose body calls a ``PersistedMapFeatures_InitFor*`` (``:4710-4782``,
    ``:5325``, ``:5699``), by their ``include/data/scripts/scrcmd.h`` constant."""
    src = (decomp / "src" / "scrcmd.c").read_text(encoding="utf-8")
    handlers = set()
    for chunk in src.split("\nstatic BOOL ")[1:]:
        name = chunk.split("(", 1)[0]
        body = chunk.split("\n}\n", 1)[0]
        if "{" in body and "PersistedMapFeatures_InitFor" in body:
            handlers.add(name)
    table = (decomp / "include" / "data" / "scripts" / "scrcmd.h").read_text(encoding="utf-8")
    constants = {
        h: c for c, h in re.findall(r"^ScriptCommand\((\w+),\s*(\w+)\)", table, re.MULTILINE)
    }
    return {constants[h] for h in handlers}


@pytest.mark.slow
def test_no_header_a_walk_reaches_sets_a_persisted_map_feature(
    rom_files, decomp, derived_and_reached
):
    # Dynamic collision (DynamicMapFeatures_CheckCollision, dynamic_map_features.c:88-103) and
    # the dynamic height plates act only while the persisted feature id is set; a warp resets it
    # (field_map_change.c:268-269, :325-326) and only these commands set it. The walk models
    # none of them, so no reached header's script may run one, except the Eterna Gym's own
    # (gym_facts models its clock).
    from tools.kaizo.gauntlet_map import GYM_HEADER, RomMaps
    from tools.rom.readers.maps import SCRIPTS_NARC, _walk, command_table

    ten = _persisted_feature_commands(decomp)
    assert len(ten) == 10, sorted(ten)
    eterna = "SCRCMD_INITPERSISTEDMAPFEATURESFORETERNAGYM"
    assert eterna in ten
    _, reached = derived_and_reached
    headers = set().union(*reached.values())
    # the leave targets and the healer's header (docs/gauntlet-design.md section 3 tables)
    assert {198, 254, 65, 376, GYM_HEADER} <= headers, sorted(headers)
    maps = RomMaps(rom_files)
    scripts = rom_files.narc(SCRIPTS_NARC)
    table = command_table(decomp)
    found = {}
    for h in sorted(headers):
        member = maps.header(h).scripts
        decoded, _, _ = _walk(scripts[member], table, f"scr_seq {member}")
        used = {table.constants[op] for op, _, _ in decoded.values()} & ten
        if used != ({eterna} if h == GYM_HEADER else set()):
            found[h] = sorted(used)
    assert found == {}


def _common_ranges(decomp) -> list[tuple[int, str]]:
    """``SCRIPT_RANGE_TABLE`` (``src/script_manager.c:32-62``) as (first id, file), highest
    first, the order ``ScriptContext_LoadAndOffsetID`` tests them (:228-254); the offsets from
    ``include/script_manager.h:90-119``."""
    src = (decomp / "src" / "script_manager.c").read_text(encoding="utf-8")
    rows = re.findall(r"Entry\((SCRIPT_ID_OFFSET_\w+),\s*(\w+),", src)
    head = (decomp / "include" / "script_manager.h").read_text(encoding="utf-8")
    offsets = {k: int(v) for k, v in re.findall(r"#define (SCRIPT_ID_OFFSET_\w+)\s+(\d+)", head)}
    assert len(rows) == 30 and all(k in offsets for k, _ in rows)
    return [(offsets[k], f) for k, f in rows]


def _common_file(ranges, script_id: int) -> str | None:
    """The common file a script id runs, or None for 1..1999 (the current map's own file,
    script_manager.c:243-246); id 0 runs ``scripts_unk_0402`` (:247-251)."""
    for first, name in ranges:
        if script_id >= first:
            return name
    return None if script_id >= 1 else "scripts_unk_0402"


@pytest.mark.slow
def test_no_common_script_a_reached_header_starts_sets_a_persisted_map_feature(
    rom_files, decomp, derived_and_reached
):
    # A reached header can start a common script by id: its bg, object and coord events
    # (map_script_ids' fields, any id), its init scripts (init_script_ids) and the commands
    # that start a script by id in its own file and in each common file so reached
    # (CallCommonScript, src/scrcmd.c:973-985; Unused_013, :960-971). Each common file is
    # walked whole (every entry): scripts_safari_game holds InitGreatMarshTram (scrcmd.c:5325).
    # A common file is a scr_seq member by its scr_seq.naix index, the scripts.order line.
    import struct as st

    from tools.kaizo.gauntlet_map import RomMaps
    from tools.rom.readers.maps import (
        CALLS_COMMON_SCRIPT,
        EVENTS_NARC,
        HEADER_INIT_SCRIPTS,
        KAIZO_MAP_HEADERS,
        SCRIPTS_NARC,
        STARTS_SCRIPT_BY_ID,
        _event_records,
        _walk,
        command_table,
        init_script_ids,
    )

    ten = _persisted_feature_commands(decomp)
    ranges = _common_ranges(decomp)
    order_file = decomp / "res" / "field" / "scripts" / "scripts.order"
    order = [s.strip() for s in order_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    _, reached = derived_and_reached
    headers = set().union(*reached.values())
    maps = RomMaps(rom_files)
    scripts = rom_files.narc(SCRIPTS_NARC)
    events = rom_files.narc(EVENTS_NARC)
    raw = rom_files.narc(KAIZO_MAP_HEADERS)
    table = command_table(decomp)
    starts = {table.opcode[CALLS_COMMON_SCRIPT], table.opcode[STARTS_SCRIPT_BY_ID]}
    scanned: dict[str, tuple[set, set]] = {}

    def scan(member: int, name: str) -> tuple[set, set]:
        decoded, _, _ = _walk(scripts[member], table, name)
        used = {table.constants[op] for op, _, _ in decoded.values()} & ten
        return used, {v[0] for op, v, _ in decoded.values() if op in starts}

    found = {}
    reached_files = set()
    for h in sorted(headers):
        head = maps.header(h)
        ids = set(_event_records(events[head.events], head.events)[2])
        init = st.unpack_from("<H", raw[h], HEADER_INIT_SCRIPTS)[0]
        ids |= set(init_script_ids(scripts[init], init))
        ids |= scan(head.scripts, f"scr_seq {head.scripts}")[1]
        todo = {f for f in (_common_file(ranges, i) for i in ids) if f is not None}
        done = set()
        while todo:
            name = todo.pop()
            if name in done:
                continue
            done.add(name)
            if name not in scanned:
                scanned[name] = scan(order.index(name), name)
            used, more = scanned[name]
            if used:
                found[(h, name)] = sorted(used)
            todo |= {f for f in (_common_file(ranges, i) for i in more) if f is not None}
        reached_files |= done
    assert "scripts_battles" in reached_files  # the trainers' own scripts are reached
    assert found == {}


@pytest.mark.slow
def test_the_unknown_var_coords_of_the_strict_headers_are_the_listed_ones(
    rom_files, derived_and_reached
):
    # In a group's strict headers (its own and the healer's) a coord on a var the walk does not
    # know blocks; Route 213's 0x40FA story coords read the var from the walk state
    # (ROUTE_213_COORDS: 7 from the lobby, unknown from the west entry). The list of such coords
    # in the headers the walks reach is pinned: a new one fails here.
    from tools.kaizo.gauntlet_map import (
        BUILDINGS,
        OREBURGH,
        ROUTE_204,
        ROUTE_213,
        ROUTE_213_COORDS,
        ROUTE_213_HEALERS,
        RomMaps,
        literal_coord,
    )

    _, reached = derived_and_reached
    maps = RomMaps(rom_files)
    groups = {
        "Oreburgh Mine": OREBURGH.headers,
        "Route 204 North": ROUTE_204.headers,
        "Team Galactic Eterna Building #1": BUILDINGS.headers,
        "Route 213": (*ROUTE_213.headers, *(h for h, _ in ROUTE_213_HEALERS)),
    }
    found = set()
    for name, strict in groups.items():
        for h in sorted(set(strict) & reached[name]):
            found |= {(h, c.script, c.var) for c in maps.area(h).coords if literal_coord(c) is None}
    assert found == {(75, 2, 0x407A)} | {(373, s, 0x40FA) for s in ROUTE_213_COORDS}


def _committed():
    from tools.kaizo.overrides import load_overrides

    return {g.name: g for g in load_overrides().gauntlets}


def _fights(order):
    return tuple(frozenset(f) for f in order)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_committed()))
def test_the_tool_re_derives_the_committed_facts(derived, name):
    f = _committed()[name].facts
    d = derived[name]
    if f.leave_gaps is None:
        assert d["leave_gaps"] is None
    else:
        assert {
            _fights(o): tuple(g) for o, g in zip(d["orders"], d["leave_gaps"], strict=True)
        } == {_fights(o): g for o, g in zip(f.orders, f.leave_gaps, strict=True)}
    assert d["escape"] is f.escape
    flat = lambda v: None if v is None else tuple(v)
    assert (flat(d["pc_gaps"]), flat(d["heal_gaps"]), tuple(d["locked_gaps"])) == (
        f.pc_gaps,
        f.heal_gaps,
        f.locked_gaps,
    )
    assert {_fights(o): tuple(h) for o, h in zip(d["orders"], d["fight_headers"], strict=True)} == {
        _fights(o): h for o, h in zip(f.orders, f.fight_headers, strict=True)
    }
    assert d["live_next"] == ([r.to_dict() for r in f.live_next] or None)
    # the field-move cases (chain_group_facts): the relevant moves and every other subset's facts
    assert d["field_moves"] == list(f.field_moves)
    assert [MoveCase.from_dict(c) for c in d["cases"]] == list(f.cases)
    # the skips behind a pushed boulder (wayward_skip)
    assert [Skip.from_dict(k) for k in d.get("skips", [])] == list(f.skips)
    assert d.get("requires_beaten", []) == list(f.requires_beaten)
    # the wild levels of the walk with wild (gauntlet_map.wild_levels)
    assert [WildLevel.from_dict(w) for w in d["wild_levels"]] == list(f.wild_levels)
    if f.partner is None:
        assert d["partner"] is None
    else:
        assert d["partner"] == {
            "trainer": f.partner.trainer,
            "join_before": list(f.partner.join_before),
        }
    assert set(derived) == set(_committed())


@pytest.mark.slow
def test_oreburgh_fights_196_over_the_ledge_then_195_with_no_leave(derived):
    # Before the Coal Badge the Rock Smash rocks (19,28)-(19,29) close the east way; the only
    # way west is the ledge (2-4,19) (JUMP_SOUTH), one-way, then 196's line at row 24, then
    # 195's tile (10,29). Roark's talk removes rock 1 only after both fights.
    d = derived["Oreburgh Mine"]
    assert d["orders"] == [[[196], [195]]] and d["leave_gaps"] == [[]] and d["escape"] is True
    assert d["fight_headers"] == [[199, 199]]


@pytest.mark.slow
def test_building_1_has_both_orders_and_only_one_leave_gap(derived):
    d = derived["Team Galactic Eterna Building #1"]
    assert dict(zip(map(str, d["orders"]), d["leave_gaps"], strict=True)) == {
        str([[421], [831]]): [],
        str([[831], [421]]): [1],
    }


@pytest.mark.slow
def test_building_3_is_forced_412_422_with_no_leave_gap(derived):
    d = derived["Team Galactic Eterna Building #3"]
    assert d["orders"] == [[[412], [422]]] and d["leave_gaps"] == [[]]


@pytest.mark.slow
def test_route_213_without_surf_is_a_beach_line_from_either_end(derived):
    # The Wake split holds no Fen Badge: no Surf, so the beach is one line, walked from the
    # lobby end or the west end. From the lobby end the way passes coord 12's cutscene, which
    # parks trainer 107 on (700, 832), the line's only tile back east (ROUTE_213_CUTSCENE): no
    # leave and no lobby heal after it. The west order leaves at every gap (the west exit).
    d = derived["Route 213"]
    assert {tuple(t for f in o for t in f) for o in d["orders"]} == {
        (106, 103, 105, 111, 110),
        (110, 111, 105, 103, 106),
    }
    leave = {o[0][0]: g for o, g in zip(d["orders"], d["leave_gaps"])}
    assert leave == {106: [], 110: [1, 2, 3, 4]}
    assert d["heal_gaps"] == [] and d["pc_gaps"] == [] and d["live_next"] is None
    # walking non-members 107 and 108 close the lobby end unbeaten (Walker.wander_tiles)
    assert d["requires_beaten"] == [107, 108]


@pytest.mark.slow
def test_the_eterna_gym_sequence_over_the_clock_states(rom_files, decomp):
    from tools.kaizo.gauntlet_map import GYM_HEADER, RomMaps

    maps = RomMaps(rom_files)
    a = maps.area(GYM_HEADER, masked=False)
    tables = eterna_clock_tables(decomp)
    door = (11, 27)  # the south warp entrance (tile behaviour 0x65): arrival and leave tile
    r0 = gym_reach(a, tables, 0, door)
    assert r0.coords == {} and 0 in r0.talk and not {3, 4} & set(r0.talk)  # only the guide
    r1 = gym_reach(a, tables, 1, door)
    assert set(r1.coords) == {7} and 3 not in r1.talk and 4 not in r1.talk  # only 324
    r2 = gym_reach(a, tables, 2, r1.coords[7], {7})
    assert (8 in r2.coords or 3 in r2.talk) and 4 not in r2.talk and door not in r2.seen
    where = r2.coords[8] if 8 in r2.coords else r2.talk[3]
    r2b = gym_reach(a, tables, 2, where, {7, 8})
    assert 4 in r2b.talk and door not in r2b.seen  # 259 won: 260 reachable, the door not
    r3 = gym_reach(a, tables, 3, r2b.talk[4], {7, 8})
    assert door in r3.seen and 1 in r3.talk  # 260 won: the door and the leader


@pytest.mark.slow
def test_wayward_order_join_points_and_the_459_talk_tiles(rom_files, derived):
    from tools.kaizo.gauntlet_map import RomMaps

    d = derived["Wayward Cave"]
    # 459+460 and 454+455 each as one VS2 fight or in turn (a moved trainer or the pushed
    # boulder can cut the second one's line)
    assert [len(o) for o in d["orders"]] == [6, 7, 7, 8]
    assert d["partner"] == {"trainer": 612, "join_before": [1]}
    assert d["leave_gaps"] == [[]] * 4 and d["escape"] is True
    # Strength (the boulders before 459+460), Rock Smash and wild (cave-floor encounter tiles
    # on every way) all needed: every smaller case has no full sequence
    assert d["field_moves"] == ["rock_smash", "strength", "wild"]
    assert [(c["moves"], c["orders"]) for c in d["cases"]] == [
        ([], []),
        (["rock_smash"], []),
        (["strength"], []),
        (["wild"], []),
        (["rock_smash", "strength"], []),
        (["rock_smash", "wild"], []),
        (["strength", "wild"], []),
    ]
    # 455 (obj 14 at (42,42), LOOK_WEST, sight 2: line (41,42), (40,42)) is skipped only with
    # boulder 26 pushed from (28,16) onto (41,42) while the player passes (40,42)
    assert [
        (k["skipped"], k["boulder"], k["boulder_from"], k["boulder_to"], k["hold"])
        for k in d["skips"]
    ] == [([455], 26, [28, 16], [41, 42], [[40, 42]])] * 2
    assert sorted(len(k["order"]) for k in d["skips"]) == [6, 7]
    maps = RomMaps(rom_files)
    joined = _wayward_walker(maps, True)
    starts, done = {WAYWARD_START}, frozenset()
    for pair in ((383, 443), (380, 458), (450, 496)):
        _, fights = _wayward_union(joined, starts, done)  # the walks from every start at once
        nxt = {s for f, sights in fights.items() if set(f) == set(pair) for s in sights}
        assert nxt
        starts, done = nxt, done | set(pair)
    for s in starts:  # from every start state of the third pair
        assert joined.talk_tiles(s.pos, 284, 459, done) == [(34, 28), (35, 27)]
        assert joined.talk_tiles(s.pos, 284, 460, done) == []


@pytest.mark.slow
def test_wayward_walkout_closes_the_start_without_the_partner(rom_files):
    from tools.kaizo.gauntlet_map import DIRS, RomMaps

    maps = RomMaps(rom_files)
    (c,) = [c for c in maps.area(284).coords if c.script == WAYWARD_WALKOUT]
    assert (c.x, c.z, c.width, c.length, c.var, c.value) == (85, 8, 1, 3, WAYWARD_PARTNER_VAR, 0)
    alone = _wayward_walker(maps, False).walk(WAYWARD_START, ())
    assert alone.spot == {}  # no trainer is reachable before the partner joins
    assert any(Pos(284, 72 + dx, 13 + dz) in alone.seen for dx, dz in DIRS)  # her tile is
    assert not any(p.x > 85 for p in alone.seen)
    joined = _wayward_walker(maps, True).walk(WAYWARD_START, ())
    assert any(p.x > 85 for p in joined.seen)  # with 0x4091 == 1 the coord does not run


@pytest.mark.slow
def test_elite_four_pc_gaps_are_the_four_lifts(rom_files):
    # The facts B4-19 restores the Elite Four + Champion group with; no group in batch 3.
    from tools.kaizo.gauntlet_map import RomMaps, escape, league_facts

    maps = RomMaps(rom_files)
    d = league_facts(maps)
    assert d["orders"] == [[[866], [867], [868], [869], [870]]] and d["pc_gaps"] == [1, 2, 3, 4]
    assert escape(maps, d["escape_header"]) is False


GAUNTLET_HEADERS = (199, 346, 72, 73, 74, 75, 373, 284, 256)


@pytest.mark.slow
def test_rom_tile_kinds_in_the_gauntlet_headers(rom_files):
    # What the walk meets in the Kaizo gauntlet headers. Kinds absent here are modelled anyway
    # (the synthetic tests above): directional, spin, ice, bridge, bike and waterfall tiles.
    from tools.kaizo.gauntlet_map import (
        BLOCKS,
        BRIDGE_START,
        BRIDGES,
        ICE,
        JUMP,
        JUMP_TWICE,
        ROCK_CLIMB_EW,
        ROCK_CLIMB_NS,
        SLIDE,
        WATERFALL,
        RomMaps,
    )

    maps = RomMaps(rom_files)
    kinds = {h: {a & 0xFF for a in maps.area(h).tiles.values()} for h in GAUNTLET_HEADERS}
    every = set().union(*kinds.values())
    absent = set().union(*BLOCKS, SLIDE, JUMP_TWICE, BRIDGES, {ICE, WATERFALL, BRIDGE_START})
    assert every & absent == set()
    assert {h for h, k in kinds.items() if k & set(JUMP)} == {199, 346, 256}  # ledges
    assert {h for h, k in kinds.items() if k & SURFABLE} == {373}  # water
    assert {h for h, k in kinds.items() if k & {ROCK_CLIMB_NS, ROCK_CLIMB_EW}} == {373}


@pytest.mark.slow
def test_rom_heights_of_the_gauntlet_headers(rom_files):
    # No walkable tile of a gauntlet header holds two plates (no floor over another; some
    # collision tiles do), every tile the building walk reaches has a plate, and header 73
    # climbs from 0 over 8 to 24 (steps < 20).
    from tools.kaizo.gauntlet_map import BUILDINGS, FieldMoves, RomMaps

    maps = RomMaps(rom_files)
    for h in GAUNTLET_HEADERS:
        area = maps.area(h)
        walkable = [p for p, t in area.tiles.items() if not t & 0x8000]
        assert all(len(area.heights.get(p, ())) <= 1 for p in walkable), h
    w = Walker(maps, BUILDINGS.members, moves=FieldMoves.for_split("Fantina"), strict=(72,))
    seen = w.walk(Pos(72, 11, 14), frozenset(BUILDINGS.members)).seen
    assert all(maps.area(p.header).heights.get((p.x, p.z)) for p in seen if p.header in (72, 73))
    floors = {maps.area(73).heights[(p.x, p.z)][0] // 4096 for p in seen if p.header == 73}
    assert floors == {0, 7, 8, 23, 24}


@pytest.mark.slow
def test_rom_shared_sight_tiles_are_only_the_wayward_pairs(rom_files):
    # Two members that can spot the player on one tile fight as one double (VS2). In the Kaizo
    # gauntlet headers only the Wayward pairs share sight tiles; no other group has one.
    from tools.kaizo.gauntlet_map import BUILDINGS, OREBURGH, ROUTE_204, ROUTE_213, RomMaps

    maps = RomMaps(rom_files)
    for spec in (OREBURGH, ROUTE_204, BUILDINGS, ROUTE_213):
        w = Walker(maps, spec.members)
        for h in spec.headers:
            for o in maps.area(h).objects:
                if o.trainer in spec.members:
                    tiles = {(x, z) for x, z, _, _ in w.sight_tiles(h, o)}
                    for p in maps.area(h).objects:
                        if p.trainer in spec.members and p.trainer != o.trainer:
                            assert not tiles & {(x, z) for x, z, _, _ in w.sight_tiles(h, p)}


@pytest.mark.slow
def test_rom_buildings_2_and_3_are_both_header_74(rom_files):
    # ESCAPE_HEADERS gave both 74 by hand (review); the escape now comes from the fight headers,
    # and the ROM places 410, 423, 412 and 422 in header 74 (events 73 objects 0, 1, 2, 4).
    from tools.kaizo.gauntlet_map import RomMaps

    maps = RomMaps(rom_files)
    in74 = {o.trainer for o in maps.area(74).objects if o.trainer is not None}
    assert {410, 423, 412, 422} <= in74
    assert not {o.trainer for o in maps.area(75).objects} - {None}


@pytest.mark.slow
def test_rom_gym_scripts_give_the_split_badges(rom_files, decomp):
    import struct as st

    from tools.kaizo.gauntlet_map import COAL, FOREST, ICICLE, MINE
    from tools.rom.readers.maps import SCRIPTS_NARC, command_table

    table = command_table(decomp)
    give = table.constants.index("SCRCMD_GIVEBADGE")
    check = table.constants.index("SCRCMD_CHECKBADGEACQUIRED")
    scr = rom_files.narc(SCRIPTS_NARC)
    members = {"Roark": (52,), "Gardenia": (73,), "Fantina": (96, 101), "Maylene": (137,),
               "Wake": (125,), "Byron": (38,), "Candice": (176,), "Volkner": (160,)}  # fmt: skip
    for split, badge in SPLIT_BADGES:
        if badge is None:
            continue
        found = [m for m in members[split] if st.pack("<HH", give, badge) in scr[m]]
        assert found, (split, badge)
    # the field-move scripts (scr 409) check Coal, Forest, Mine and Icicle
    for badge in (COAL, FOREST, MINE, ICICLE):
        assert st.pack("<HH", check, badge) in scr[409]


@pytest.mark.slow
def test_rom_group_splits_match_the_committed_sources():
    from tools.kaizo.gauntlet_map import GROUP_SPLITS

    for name, g in _committed().items():
        assert g.facts.source.startswith(f"{GROUP_SPLITS[name]} Split:")


def test_a_matching_coord_outside_the_strict_headers_blocks():
    # its script may take the player: the tile is not entered, nothing beyond it is reached
    here = area(1, ["......"], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["....."], warps=[Warp(0, 0, 0, 1, 0)], coords=[Coord(7, 2, 0, 1, 1, 0, 0)])
    maps = StaticMaps([here, there])
    w = Walker(maps, strict=[1]).walk(Pos(1, 0, 0))
    assert Pos(2, 1, 0) in w.seen and Pos(2, 2, 0) not in w.seen and Pos(2, 4, 0) not in w.seen
    assert w.crossed == []
    with pytest.raises(MapReadError):
        Walker(maps).walk(Pos(1, 0, 0))
    # the caller's rule still decides
    passed = Walker(maps, strict=[1], coord_rule=lambda h, c, m: "pass").walk(Pos(1, 0, 0))
    assert Pos(2, 4, 0) in passed.seen


def _healer_maps(coords):
    # header 1 (the group's) warps into header 2, a corridor to a healer (obj 5 at (4, 0))
    # behind a coord at (2, 0) on var 0x4100
    here = area(1, ["......"], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["....."], [npc(5, 4, 0)], [Warp(0, 0, 0, 1, 0)], coords)
    return StaticMaps([here, there])


def test_a_foreign_coord_on_an_unknown_var_blocks_the_healers_talk_tile():
    maps = _healer_maps([Coord(7, 2, 0, 1, 1, 1, 0x4100)])
    w = Walker(maps, strict=[1]).walk(Pos(1, 0, 0))
    assert (2, 5) not in w.talk and Pos(2, 3, 0) not in w.seen
    # in a strict header (the healer's header is strict in derive_groups) it blocks too: its
    # script may start a battle or move the player
    strict = Walker(maps, strict=[1, 2]).walk(Pos(1, 0, 0))
    assert strict.talk == {} and Pos(2, 3, 0) not in strict.seen
    known = Walker(maps, strict=[1, 2], vars={0x4100: 0}).walk(Pos(1, 0, 0))
    assert known.talk == {(2, 5): Pos(2, 3, 0)}


def test_a_foreign_coord_with_a_non_matching_var_does_not_block():
    maps = _healer_maps([Coord(7, 2, 0, 1, 1, 1, 0x4100)])
    w = Walker(maps, strict=[1], vars={0x4100: 0}).walk(Pos(1, 0, 0))
    assert w.talk == {(2, 5): Pos(2, 3, 0)}
    with pytest.raises(MapReadError, match="not modelled"):
        Walker(maps, strict=[1, 2], vars={0x4100: 1}).walk(Pos(1, 0, 0))


# --- hidden flags, the live count and the B2B operands (manager rulings of 22 Nov) -------------

# The group's own headers (and the Route 213 lobby, the healer's header).
OWN_HEADERS = {
    "Oreburgh Mine": (199,),
    "Route 204 North": (346,),
    "Route 204 North B2B": (256,),
    "Eterna Gym gym trainers": (67,),
    "Team Galactic Eterna Building #1": (72, 73, 74, 75),
    "Team Galactic Eterna Building #2": (72, 73, 74, 75),
    "Team Galactic Eterna Building #3": (72, 73, 74, 75),
    "Route 213": (373, 376),
    "Wayward Cave": (284,),
}


def _flagged(maps, header):
    """The objects of a header whose hidden flag matters: a script, a flag that is not map-local
    (1-0x40, cleared on every map change) and no item ball (``MapObject.cleared``)."""
    from tools.kaizo.gauntlet_map import NO_FIELD_MOVES

    return [
        o
        for o in maps.area(header).objects
        if o.script != 0xFFFF and o.flag > 0x40 and not o.cleared(NO_FIELD_MOVES)
    ]


@pytest.mark.slow
def test_rom_flag_writers_are_exactly_the_cited_ones(rom_files, decomp):
    # Every SetFlag / ClearFlag of a cited hidden flag in the whole script archive, by control
    # flow from each file's entries. The files the walk cannot read are init-script files (a
    # header's initScriptsArchiveID, include/map_header.h:14-15, or of the same shape: typed
    # entries and frame tables, no commands; init_script_ids reads them).
    import struct as st

    from tools.kaizo.gauntlet_map import FLAG_WRITERS
    from tools.rom.readers.maps import (
        HEADER_INIT_SCRIPTS,
        KAIZO_MAP_HEADERS,
        SCRIPTS_NARC,
        ScriptDecodeError,
        _walk,
        command_table,
        init_script_ids,
    )

    inits = {
        st.unpack_from("<H", b, HEADER_INIT_SCRIPTS)[0]
        for b in rom_files.narc(KAIZO_MAP_HEADERS)
        if len(b) >= HEADER_INIT_SCRIPTS + 2
    }

    table = command_table(decomp)
    ops = {table.opcode["SCRCMD_SETFLAG"]: "set", table.opcode["SCRCMD_CLEARFLAG"]: "clear"}
    scr = rom_files.narc(SCRIPTS_NARC)
    found: dict[int, set] = {f: set() for f in FLAG_WRITERS}
    for m in range(len(scr)):
        try:
            decoded = _walk(scr[m], table, f"scr {m}")[0]
        except ScriptDecodeError:
            assert m in inits or scr[m][0] <= 4, m
            init_script_ids(scr[m], m)
            continue
        for at, (op, values, _) in decoded.items():
            if op in ops and values[0] in found:
                found[values[0]].add((m, at, ops[op]))
    assert found == {f: set(w) for f, w in FLAG_WRITERS.items()}


@pytest.mark.slow
def test_rom_hidden_objects_per_group(rom_files):
    # Which objects the flags hide when each group starts (GROUP_FLAGS) and after the Route 213
    # lobby is entered (ENTER_FLAGS); every flag that matters in the group's headers is cited.
    from tools.kaizo.gauntlet_map import ENTER_FLAGS, FLAG_WRITERS, GROUP_FLAGS, RomMaps

    maps = RomMaps(rom_files)
    hidden = {}
    for name, heads in OWN_HEADERS.items():
        for h in heads:
            for o in _flagged(maps, h):
                assert o.flag in FLAG_WRITERS, (name, h, o.index, o.flag)
        hidden[name] = sorted(
            (h, o.index) for h in heads for o in _flagged(maps, h) if o.flag in GROUP_FLAGS[name]
        )
    assert hidden == {
        "Oreburgh Mine": [],
        "Route 204 North": [],
        "Route 204 North B2B": [(256, 3)],  # the Galactic works key, until both fights are won
        "Eterna Gym gym trainers": [],
        "Team Galactic Eterna Building #1": [],
        "Team Galactic Eterna Building #2": [],
        "Team Galactic Eterna Building #3": [],
        "Route 213": [(373, 28), (373, 31), (376, 2)],  # the grunt, Looker, the lobby's Looker
        "Wayward Cave": [],
    }
    lobby = sorted(o.index for o in _flagged(maps, 373) if o.flag in ENTER_FLAGS[376])
    assert lobby == [33, 34, 35, 36, 37, 38]  # trainer 696's pair and four spectators


def _chain_groups():
    from tools.kaizo.gauntlet_map import BUILDINGS, OREBURGH, ROUTE_204, ROUTE_213

    return [
        ("Oreburgh Mine", OREBURGH, OREBURGH.members, {}),
        ("Route 204 North", ROUTE_204, ROUTE_204.members, {}),
        *(
            (f"Team Galactic Eterna Building #{n}", BUILDINGS, g, {})
            for n, g in enumerate(((421, 831), (410, 423), (412, 422)), 1)
        ),
        ("Route 213", ROUTE_213, ROUTE_213.members, {"healers": ((376, 1),)}),
    ]


@pytest.mark.slow
def test_rom_objects_beyond_the_groups_headers_change_no_fact(rom_files):
    # The flags of objects outside a group's own headers are not derived: the facts are the
    # same with every such object standing and with every one hidden, and hiding objects only
    # widens a walk (no ice or spin tile in these headers), so any mix gives them too.
    from tools.kaizo.gauntlet_map import (
        ENTER_FLAGS,
        FLAG_WRITERS,
        GROUP_FLAGS,
        GROUP_SPLITS,
        FieldMoves,
        RomMaps,
        Walker,
        chain_facts,
        read_double_trainers,
        runs,
    )

    maps = RomMaps(rom_files)
    doubles = read_double_trainers(rom_files)
    for name, spec, group, kw in _chain_groups():

        def walker(flags, name=name, spec=spec):
            return Walker(
                maps,
                spec.members,
                moves=FieldMoves.for_split(GROUP_SPLITS[name]),
                strict=spec.headers,
                doubles=doubles,
                flags=flags,
                on_enter=ENTER_FLAGS,
            )

        start = GROUP_FLAGS[name]
        heads = {
            h for r in runs(walker(start), spec) for ws in r.states for w in ws for h in w.headers()
        }
        # a flag an own-header object carries too (509: the Eterna City grunts) is cited
        beyond = {
            o.flag
            for h in heads - set(OWN_HEADERS[name])
            for o in _flagged(maps, h)
            if o.flag not in FLAG_WRITERS
        }
        base = chain_facts(maps, walker(start), spec, group, **kw)
        assert chain_facts(maps, walker(start | beyond), spec, group, **kw) == base, name


@pytest.mark.slow
def test_rom_building_1f_first_visit_placement_changes_nothing(rom_files):
    # On the first 1F visit scr 79 entry 5 moves the grunt-Looker (local 4) to (12,15)
    # (SetObjectEventPos @210, AddObject @230) and removes Looker (local 5, @234); later loads
    # place both at their event tile (14,15). The building facts are the same with neither tile
    # blocked and with both blocked, so with either placement.
    from tools.kaizo.gauntlet_map import BUILDINGS, FieldMoves, RomMaps, Walker, chain_facts

    base = RomMaps(rom_files)

    class Both:
        def __init__(self):
            a = base.area(72)
            (o,) = [o for o in a.objects if o.index == 2]
            moved = o._replace(index=99, x=12, z=15, flag=0)
            self._a = Area(
                72, a.tiles, (*a.objects, moved), a.warps, a.coords, a.map_type,
                a.escape_bit, a.heights,
            )  # fmt: skip

        def area(self, h, masked=True):
            return self._a if h == 72 else base.area(h, masked)

        def header(self, h):
            return base.header(h)

    def facts(maps, flags):
        out = []
        for g in ((421, 831), (410, 423), (412, 422)):
            w = Walker(
                maps, BUILDINGS.members, moves=FieldMoves.for_split("Fantina"),
                strict=BUILDINGS.headers, flags=flags,
            )  # fmt: skip
            out.append(chain_facts(maps, w, BUILDINGS, g))
        return out

    neither = facts(base, {443, 450})
    assert facts(base, ()) == neither == facts(Both(), ())


@pytest.mark.slow
def test_rom_b2b_battle_operands_are_literal_trainer_ids(rom_files, decomp):
    # Ruling (5): header 256's coord script, scr 277 entry 2, StartTrainerBattle (opcode 0xE5,
    # include/data/scripts/scrcmd.h:237; two u16 operands, asm/macros/scrcmd.inc:1427-1431) at
    # @44 (296, 0) and @73 (297, 0): ScriptContext_GetVar (src/scrcmd_trainer.c:129-130) reads a
    # number below VARS_START as itself, so both are literal single battles.
    from tools.kaizo.gauntlet_map import B2B_HEADER, RomMaps, ScriptReader, b2b_facts

    maps = RomMaps(rom_files)
    scripts = ScriptReader(rom_files, decomp)
    assert maps.header(B2B_HEADER).scripts == 277
    battles = [(at, v) for at, c, v in scripts.trace(277, 2) if c == "SCRCMD_STARTTRAINERBATTLE"]
    assert battles == [(44, [296, 0]), (73, [297, 0])]
    assert b2b_facts(maps, scripts, B2B_HEADER)["orders"] == [[[296], [297]]]


@pytest.mark.slow
def test_rom_route_204_live_next_follows_the_count(rom_files, derived):
    # 15 (Twins Liv & Liz) is a double-battle trainer: with one live Pokemon it is not fought,
    # and a faint or a revive between fights switches the continuation. No gym trainer is one,
    # and the Wayward pairs fight with the partner, who counts two: only Route 204 North's walk
    # depends on the count. Route 204 North's leave depends on the count (after 13: at one
    # live, not at two).
    from tools.kaizo.gauntlet_map import GYM_FIGHTS, read_double_trainers

    doubles = read_double_trainers(rom_files)
    assert 15 in doubles and not set(GYM_FIGHTS) & doubles
    rows = {
        (tuple(map(tuple, r["fought"])), r["live"], r.get("at", "")): r["next"]
        for r in derived["Route 204 North"]["live_next"]
    }
    assert rows[((), 1, "")] == [[13]] and rows[((), 2, "")] == [[15]]
    assert rows[(((15,),), 1, "")] == [[13]]  # a faint in the twins' fight
    assert rows[(((13,),), 2, "")] == [[14], [15]] and rows[(((13,),), 1, "")] == [[14]]
    assert rows[(((13,), (14,)), 1, "")] == [] and rows[(((13,), (14,)), 1, "bag")] == [[15]]
    assert {n for n, d in derived.items() if d["live_next"] is not None} == {"Route 204 North"}
    absent = {
        (n, k)
        for n, d in derived.items()
        for k in ("leave_gaps", "heal_gaps", "pc_gaps")
        if d[k] is None
    }
    assert absent == {("Route 204 North", "leave_gaps")}


def test_a_move_met_but_never_needed_is_not_relevant():
    # changing_moves: Cut is used (a tree met) but no subset's facts change without it
    from tools.kaizo.gauntlet_map import changing_moves

    a, b = {"orders": [[[1], [2]]]}, {"orders": [[[1], [2]], [[2], [1]]]}
    by = {(): a, ("cut",): a, ("rock_smash",): b, ("cut", "rock_smash"): b}
    assert changing_moves(by, {"cut", "rock_smash"}) == ("rock_smash",)
    assert changing_moves(by, {"cut"}) == ()
    # a subset with no full sequence (None) differs from one with facts
    by = {(): None, ("strength",): a}
    assert changing_moves(by, {"strength"}) == ("strength",)


@pytest.mark.slow
def test_rom_writers_of_route_213_var_are_exactly_the_cited_ones(rom_files, decomp):
    # Ruling of 22 Nov: the value of 0x40FA at the group's start comes from its writers
    # (ROUTE_213_LOBBY_VAR); every command of the archive that names the var is a
    # SetVarFromValue of VAR_40FA_WRITERS or a compare.
    from tools.kaizo.gauntlet_map import ROUTE_213_VAR, VAR_40FA_WRITERS
    from tools.rom.readers.maps import SCRIPTS_NARC, ScriptDecodeError, _walk, command_table

    table = command_table(decomp)
    scr = rom_files.narc(SCRIPTS_NARC)
    writes, other = set(), set()
    for m in range(len(scr)):
        try:
            decoded = _walk(scr[m], table, f"scr {m}")[0]
        except ScriptDecodeError:
            continue  # init-script files hold no commands (test_rom_flag_writers_...)
        for at, (op, values, _) in decoded.items():
            if ROUTE_213_VAR not in values:
                continue
            name = table.constants[op]
            if name == "SCRCMD_SETVARFROMVALUE" and values[0] == ROUTE_213_VAR:
                writes.add((m, at, values[1]))
            elif name != "SCRCMD_COMPAREVARTOVALUE":
                other.add((m, at, name))
    assert writes == set(VAR_40FA_WRITERS) and other == set()


# --- Route 213's var in the walk state, walking non-members (B3 fix 3 C7) ---------------------


def test_a_var_carried_in_the_state_decides_a_coord():
    from tools.kaizo.gauntlet_map import var_flag

    maps = _healer_maps([Coord(7, 2, 0, 1, 1, 1, 0x4100)])
    # the value 0 in the start state's flags: the coord (value 1) does not run
    known = Walker(maps, strict=[1, 2], flags={var_flag(0x4100, 0)}).walk(Pos(1, 0, 0))
    assert known.talk == {(2, 5): Pos(2, 3, 0)}

    def block(header, coord, matches):
        return "block" if matches else None

    held = Walker(maps, strict=[1, 2], flags={var_flag(0x4100, 1)}, coord_rule=block)
    assert held.walk(Pos(1, 0, 0)).talk == {}
    # unknown (no pseudo flag, no vars): blocks in a strict header
    assert Walker(maps, strict=[1, 2]).walk(Pos(1, 0, 0)).talk == {}


def test_a_warp_writes_the_var_unless_it_holds_the_kept_value():
    from tools.kaizo.gauntlet_map import var_flag, var_value

    w = Walker(StaticMaps([area(2, ["."])]), enter_vars={2: ((0x40FA, 7, 9),)})
    assert var_value(w._enter_flags(2, frozenset({var_flag(0x40FA, 3)})), 0x40FA) == 7
    assert var_value(w._enter_flags(2, frozenset({var_flag(0x40FA, 9)})), 0x40FA) == 9
    assert var_value(w._enter_flags(2, frozenset()), 0x40FA) is None  # unknown may be 9
    assert var_value(w._enter_flags(3, frozenset({var_flag(0x40FA, 3)})), 0x40FA) == 3


def _cutscene_maps(warp=False):
    # header 1: a column coord at x 2 (script 7, var 0x4100 value 7) every path east crosses;
    # trainer 3 at (7, 2) looks east off the map; with ``warp`` a door at (0, 2) into header 2
    # and back
    rows = ["........", "........", "........"]
    warps = [Warp(0, 0, 2, 2, 0)] if warp else []
    one = area(1, rows, [trainer(3, 7, 2, EAST, 2)], warps, [Coord(7, 2, 0, 1, 3, 7, 0x4100)])
    two = area(2, ["..."], warps=[Warp(0, 0, 0, 1, 0)])
    return StaticMaps([one, two])


def _cutscene_rule(effect):
    from tools.kaizo.gauntlet_map import CoordEffect

    def rule(header, coord, matches):
        if coord.script == 7 and matches:
            return effect
        if coord.script == 8:
            return "block" if matches else None
        return None

    assert isinstance(effect, CoordEffect)
    return rule


def _walks(w, start, flags):
    # the plain walk and the walk over object layouts must agree on the tiles and fights
    plain = w.walk(start, flags=flags, y=0)
    lay = w.walk_layouts([_State(start, 0, False, frozenset(flags))])
    assert set(plain.seen) == set(lay.seen) and set(plain.spot) == set(lay.spot)
    return plain


def test_a_coord_effect_writes_the_var_and_the_next_coord_reads_it():
    from tools.kaizo.gauntlet_map import CoordEffect, var_flag

    rows = ["........"]
    coords = [Coord(7, 2, 0, 1, 1, 7, 0x4100), Coord(8, 5, 0, 1, 1, 8, 0x4100)]
    maps = StaticMaps([area(1, rows, coords=coords)])
    rule = _cutscene_rule(CoordEffect(writes=((0x4100, 8),)))
    w = Walker(maps, strict=[1], coord_rule=rule)
    # 7 -> the effect passes and writes 8; coord 8 then matches and blocks
    walk = _walks(w, Pos(1, 0, 0), {var_flag(0x4100, 7)})
    assert Pos(1, 4, 0) in walk.seen and Pos(1, 5, 0) not in walk.seen
    assert [c.script for _, c in walk.crossed] == [7]
    # a write to a value no coord reads passes both
    other = Walker(maps, strict=[1], coord_rule=_cutscene_rule(CoordEffect(writes=((0x4100, 9),))))
    assert Pos(1, 7, 0) in _walks(other, Pos(1, 0, 0), {var_flag(0x4100, 7)}).seen


def test_a_coord_effect_places_a_trainer_until_the_next_map_load():
    from tools.kaizo.gauntlet_map import CoordEffect, place_flag, placements, var_flag

    # trainer 3 placed on (5, 2) looking north (movement 0: its direction only), sight 2
    effect = CoordEffect(place=((3, 5, 2, 0, NORTH),))
    w = Walker(_cutscene_maps(), {3}, strict=[1], coord_rule=_cutscene_rule(effect))
    walk = _walks(w, Pos(1, 0, 0), {var_flag(0x4100, 7)})
    # its line (5, 1), (5, 0) and its tile close the way east
    seen_at = {x.pos: x for x in walk.starts[(3,)] if x.how == "sight"}
    assert set(seen_at) == {Pos(1, 5, 0), Pos(1, 5, 1)}
    assert not {Pos(1, 6, z) for z in range(3)} & set(walk.seen)
    assert placements(seen_at[Pos(1, 5, 0)].flags) == [(1, 3, 5, 2, 0, NORTH)]
    # the trainer walks up from its placed tile (5, 2): one step north to (5, 1) for (5, 0)
    assert (1, 3, 5, 1, 0, NORTH) in seen_at[Pos(1, 5, 0)].moved
    assert (1, 3, 5, 2, 0, NORTH) in seen_at[Pos(1, 5, 1)].moved
    # not run (the var holds 8): it stays on (7, 2) looking east, the way is open
    plain = _walks(w, Pos(1, 0, 0), {var_flag(0x4100, 8)})
    assert Pos(1, 6, 0) in plain.seen
    assert all(x.how == "talk" for x in plain.starts.get((3,), ()))
    # the placed object blocks its tile, not its event tile
    s0 = _State(Pos(1, 2, 0), 0, False, frozenset({place_flag(1, 3, 5, 2, 0, NORTH)}))
    placed = w.with_effects(s0, (effect,))
    assert (1, 3, 5, 2, 0, NORTH) in placed.moved
    a = w.maps.area(1)
    assert [o.index for o in w.objects_on(a, 5, 2, placed.moved)] == [3]
    assert w.objects_on(a, 7, 2, placed.moved) == []
    # beaten, the placed trainer neither sees nor fights when talked to: it only blocks
    beaten = Walker(_cutscene_maps(), {9}, strict=[1], coord_rule=w.coord_rule, beaten={3})
    walk = _walks(beaten, Pos(1, 0, 0), {var_flag(0x4100, 7)})
    assert walk.spot == {} and Pos(1, 6, 0) in walk.seen and Pos(1, 5, 2) not in walk.seen


def test_a_warp_ends_a_placement():
    from tools.kaizo.gauntlet_map import CoordEffect, placements, var_flag

    effect = CoordEffect(place=((3, 1, 1, 0, EAST),))
    w = Walker(_cutscene_maps(warp=True), {3}, strict=[1, 2], coord_rule=_cutscene_rule(effect))
    s0 = w.with_effects(_State(Pos(1, 2, 2), 0, False, frozenset({var_flag(0x4100, 8)})), (effect,))
    assert placements(s0.flags) and s0.moved
    # the door (0, 2) -> header 2 -> back: the landing's flags carry no placement, no moved
    out = w.walk_from([s0])
    back = {s for s in out.states if s.pos.header == 1 and s.pos != Pos(1, 2, 2)}
    landed = [s for s in back if not placements(s.flags)]
    assert landed and all(s.moved == () for s in landed)
    assert all(var_flag(0x4100, 8) in s.flags for s in landed)  # the var stays


def test_the_route_213_cutscene_coord_passes_with_its_effect():
    from tools.kaizo.gauntlet_map import (
        ROUTE_213_CUTSCENE,
        ROUTE_213_VAR,
        route_213_coord_rule,
    )

    c = Coord(12, 699, 832, 1, 1, 7, ROUTE_213_VAR)
    assert route_213_coord_rule(373, c, True) is ROUTE_213_CUTSCENE
    assert route_213_coord_rule(373, c, None) is None  # unknown: blocks by default
    assert route_213_coord_rule(373, c._replace(script=13, value=8), True) == "block"
    assert ROUTE_213_CUTSCENE.writes == ((ROUTE_213_VAR, 8),)
    assert ROUTE_213_CUTSCENE.place == ((4, 700, 832, 0, SOUTH),)


@pytest.mark.slow
def test_rom_route_213_cutscene_script_is_the_cited_one(rom_files, decomp):
    # scr 459 script 12 (coord 12): SetPosition of object 6 (668, 832) @477 and (668, 810)
    # facing 1 @629, of object 4 (700, 832) facing 1 @521, SetMovementType 4 -> 0 @533,
    # 0x40FA = 8 @643; no move of the player (local id 255); objects 4 and 6 of header 373 are
    # trainer 107 (movement 43) and the type-0 object of talk script 14 on (668, 810) facing 1
    from tools.kaizo.gauntlet_map import ROUTE_213_CUTSCENE, ROUTE_213_VAR, RomMaps
    from tools.rom.readers.maps import SCRIPTS_NARC, _walk, command_table

    table = command_table(decomp)
    decoded = _walk(rom_files.narc(SCRIPTS_NARC)[459], table, "scr 459")[0]
    cmds = {at: (table.constants[op], tuple(v)) for at, (op, v, _) in decoded.items()}
    run = {at: c for at, c in cmds.items() if 477 <= at <= 649}
    assert run[477] == ("SCRCMD_SETPOSITION", (6, 668, 0, 832, 3))
    assert run[521] == ("SCRCMD_SETPOSITION", (4, 700, 0, 832, 1))
    assert run[533] == ("SCRCMD_SETMOVEMENTTYPE", (4, 0))
    assert run[629] == ("SCRCMD_SETPOSITION", (6, 668, 0, 810, 1))
    assert run[643] == ("SCRCMD_SETVARFROMVALUE", (ROUTE_213_VAR, 8))
    assert cmds[649][0] == "SCRCMD_END"
    moved = {v[0] for n, v in run.values() if n == "SCRCMD_APPLYMOVEMENT"}
    assert moved == {6, 241}
    placed = {
        v[0] for n, v in run.values() if n in ("SCRCMD_SETPOSITION", "SCRCMD_SETMOVEMENTTYPE")
    }
    assert placed == {4, 6}
    a = RomMaps(rom_files).area(373)
    o4, o6 = a.object_by_index(4), a.object_by_index(6)
    assert (o4.local_id, o4.trainer, o4.movement, o4.x, o4.z) == (4, 107, 43, 704, 832)
    assert (o6.local_id, o6.trainer_type, o6.script, o6.x, o6.z, o6.direction) == (
        6, 0, 14, 668, 810, 1,
    )  # fmt: skip
    ((index, x, z, movement, direction),) = ROUTE_213_CUTSCENE.place
    assert (index, x, z, movement, direction) == (o4.index, 700, 832, 0, 1)
    coord = next(c for c in a.coords if c.script == 12)
    assert (coord.x, coord.z, coord.value, coord.var) == (699, 832, 7, ROUTE_213_VAR)


def _wanderer(rx=1, rz=0):
    # trainer 50, movement 41 (MOVEMENT_TYPE_WALK_NORTH_EAST_SOUTH_WEST), at (4, 0), sight 1
    return MapObject(50, 50, 41, 1, 3049, SOUTH, 1, 4, 0, range_x=rx, range_z=rz)


def test_a_walking_non_member_blocks_its_box_and_unbeaten_its_sight_area():
    maps = StaticMaps([area(1, [".........", ".........", "........."], [_wanderer()])])
    unbeaten = Walker(maps, members={9}).walk(Pos(1, 0, 2))
    # box (3-5, 0); its lines reach row 1 below it and (2, 0), (6, 0): row 2 stays open
    assert Pos(1, 8, 2) in unbeaten.seen
    assert not {Pos(1, x, 1) for x in range(3, 6)} & set(unbeaten.seen)
    assert Pos(1, 2, 0) not in unbeaten.seen and Pos(1, 4, 0) not in unbeaten.seen
    beaten = Walker(maps, members={9}, beaten={50}).walk(Pos(1, 0, 2))
    assert Pos(1, 4, 1) in beaten.seen and Pos(1, 2, 0) in beaten.seen
    assert Pos(1, 4, 0) not in beaten.seen  # the box itself


def test_a_walking_non_member_with_an_unbounded_range_raises():
    maps = StaticMaps([area(1, ["....."], [_wanderer(rx=-1)])])
    with pytest.raises(MapReadError, match="unbounded"):
        Walker(maps, members={9}).walk(Pos(1, 0, 0))


# --- C9: independent hand-derived ROM checks --------------------------------------------------
# Expectations by hand; a raw reader here, not RomMaps, Walker or ScriptReader. Layouts: the
# events member (pp tools/jsoncnv/event.py:24-130, written in that order: bg events 20 bytes,
# objects 32 (14 u16 + u32 y), warps 12 (x, z, header, warp, pad 4), coords 16 (8 u16), each
# list behind a u32 count); the Kaizo header member (debug/cb_edit/d_test.narc, 24 bytes:
# matrix u16 @2, scripts u16 @4, events u16 @16, include/map_header.h:10-31); the matrix
# (src/map_matrix.c:30-56: u8 width, height, has-headers, has-altitudes, name length, name,
# u16 headers, u8 altitudes, u16 land ids); the land member's tile attributes, u16 per tile
# from 0x10, index z * 32 + x (include/constants/field/map.h:9-27), bit 15 collision, low byte
# behaviour.


def _raw_member(rom_files, narc, index):
    return rom_files.narc(narc)[index]


def _raw_header(rom_files, header):
    b = _raw_member(rom_files, "debug/cb_edit/d_test.narc", header)
    fields = (("matrix", 2), ("scripts", 4), ("events", 16))
    return {k: struct.unpack_from("<H", b, at)[0] for k, at in fields}


def _raw_events(rom_files, header):
    events = _raw_header(rom_files, header)["events"]
    m = _raw_member(rom_files, "fielddata/eventdata/zone_event.narc", events)
    out, pos = [], 0
    for size, fmt in ((20, None), (32, "<14HI"), (12, "<4H"), (16, "<8H")):
        (n,) = struct.unpack_from("<I", m, pos)
        pos += 4
        out.append([struct.unpack_from(fmt, m, pos + size * i) for i in range(n)] if fmt else n)
        pos += size * n
    assert pos == len(m)
    _, objs, warps, coords = out
    return objs, warps, coords


def _raw_attr(rom_files, header):
    matrix = _raw_header(rom_files, header)["matrix"]
    mx = _raw_member(rom_files, "fielddata/mapmatrix/map_matrix.narc", matrix)
    width, height, has_headers, has_alt, name_len = mx[:5]
    cells = width * height
    pos = 5 + name_len + (2 * cells if has_headers else 0) + (cells if has_alt else 0)
    lands = struct.unpack_from(f"<{cells}H", mx, pos)
    land = rom_files.narc("fielddata/land_data/land_data.narc")

    def attr(x, z):
        member = land[lands[(z // 32) * width + x // 32]]
        return struct.unpack_from("<H", member, 0x10 + 2 * ((z % 32) * 32 + x % 32))[0]

    return attr


def _committed_row(name):
    import tomllib

    text = (REPO_ROOT / "data" / "kaizo" / "overrides.toml").read_text(encoding="utf-8")
    return next(g for g in tomllib.loads(text)["gauntlet"] if g["name"] == name)


# include/constants/field/map_tile_behaviors.h: the enum from line 11 at 0 (JUMP_SOUTH :70,
# WATER_SEA :32, SAND :44, TABLE :139); trainer id = script - 3000 + 1 below 5000
# (Script_GetTrainerID, src/script_manager.c:498-501); movement 15 LOOK_SOUTH, 17 LOOK_EAST
# (generated/movement_types.txt:16, :18, 0-based).
JUMP_SOUTH_B, WATER_SEA_B, SAND_B, TABLE_B = 0x3B, 0x15, 0x21, 0x80
LOOK_SOUTH_MV, LOOK_EAST_MV = 15, 17


@pytest.mark.slow
def test_rom_c9_b2b_operands_by_hand(rom_files, decomp):
    # (a) header 256 runs scr 277; its entry 2 (the header's ``.long label-.-4`` list up to
    # 0xFD13) starts at 36 and entry 3 at 110; StartTrainerBattle = the line index of
    # SCRCMD_STARTTRAINERBATTLE among the ScriptCommand lines of include/data/scripts/scrcmd.h
    # (:237 -> 0xE5), operands u16 trainer + u16 partner; both literal (below VARS_START a var
    # reads as itself, FieldSystem_TryGetVar, src/script_manager.c:428-452).
    assert _raw_header(rom_files, 256)["scripts"] == 277
    lines = (decomp / "include" / "data" / "scripts" / "scrcmd.h").read_text(encoding="utf-8")
    names = re.findall(r"^ScriptCommand\((\w+),", lines, re.MULTILINE)
    opcode = {n: i for i, n in enumerate(names)}
    assert opcode["SCRCMD_STARTTRAINERBATTLE"] == 0xE5
    s = _raw_member(rom_files, "fielddata/script/scr_seq.narc", 277)
    entries, pos = [], 0
    while struct.unpack_from("<H", s, pos)[0] != 0xFD13:
        entries.append(pos + 4 + struct.unpack_from("<i", s, pos)[0])
        pos += 4
    assert entries[1:3] == [36, 110]
    assert struct.unpack_from("<H", s, 36)[0] == opcode["SCRCMD_LOCKALL"]
    assert struct.unpack_from("<3H", s, 44) == (0xE5, 296, 0)
    assert struct.unpack_from("<3H", s, 73) == (0xE5, 297, 0)
    assert struct.unpack_from("<H", s, 106)[0] == opcode["SCRCMD_RELEASEALL"]
    row = _committed_row("Route 204 North B2B")
    assert row["orders"] == [[[296], [297]]] and row["leave_gaps"] == []


def _hand_reach(attr, blocked, start, sight):
    """Tiles reached from ``start`` by hand rules: a tile without the collision bit and not
    ``blocked``; a JUMP_SOUTH tile stepped on southward lands two south (PlayerAvatar_WillJump,
    src/player_move.c:1956-1999); a tile in ``sight`` is entered and ends the walk there (the
    trainer sees the player). Returns (reached, sight tiles entered)."""
    seen, todo, hit = {start}, [start], set()
    while todo:
        x, z = todo.pop()
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = (x + dx, z + dz)
            a = attr(*t)
            if (dx, dz) == (0, 1) and a & 0xFF == JUMP_SOUTH_B:
                t = (x, z + 2)
                a = attr(*t)
            if a & 0x8000 or t in blocked or t in seen:
                continue
            seen.add(t)
            if t in sight:
                hit.add(t)
                continue
            todo.append(t)
    return seen, hit


@pytest.mark.slow
def test_rom_c9_oreburgh_hand_path(rom_files):
    # (b) header 199: one warp (15,1) -> header 198; ledges (2-4,19) JUMP_SOUTH with the
    # collision bit, (5,19) a wall; Rock Smash rocks (script 10001) on (19,28), (19,29); 196 on
    # (2,24) looking east, sight 5; 195 on (10,28) looking south, sight 3. Rock Smash is not held
    # at the Roark split (no badge), so the rocks stand.
    objs, warps, coords = _raw_events(rom_files, 199)
    assert warps == [(15, 1, 198, 3)] and coords == []
    by_xz = {(o[12], o[13]): o for o in objs}
    assert {xz for xz, o in by_xz.items() if o[5] == 10001} == {(19, 28), (19, 29)}
    t196, t195 = by_xz[(2, 24)], by_xz[(10, 28)]
    assert (t196[5] - 3000 + 1, t196[3], t196[2], t196[7]) == (196, 1, LOOK_EAST_MV, 5)
    assert (t195[5] - 3000 + 1, t195[3], t195[2], t195[7]) == (195, 1, LOOK_SOUTH_MV, 3)
    assert sorted(o[5] - 3000 + 1 for o in objs if o[3] == 1) == [195, 196]
    attr = _raw_attr(rom_files, 199)
    assert [attr(x, 19) for x in (2, 3, 4)] == [0x8000 | JUMP_SOUTH_B] * 3
    assert attr(5, 19) & 0x8000
    # the hand path: the warp's south neighbour, down the shaft, west along row 17, the jump
    # (4,18) -> (4,20), into 196's line at (3,24); then on to 195's line at (10,29)
    path = [(15, z) for z in range(2, 10)] + [(16, 9), (16, 10), (16, 11), (15, 11)]
    path += [(15, z) for z in range(12, 18)] + [(x, 17) for x in range(14, 3, -1)] + [(4, 18)]
    path += [(4, 20), (3, 20), (2, 20), (2, 21), (2, 22), (2, 23), (3, 23), (3, 24)]
    after = [(4, 24), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28)]
    after += [(x, 28) for x in range(6, 10)] + [(9, 29), (10, 29)]
    walk = path + after
    for a, b in pairwise(walk):
        assert not attr(*b) & 0x8000, b
        step = (b[0] - a[0], b[1] - a[1])
        assert step in ((1, 0), (-1, 0), (0, 1), (0, -1)) or (a, b) == ((4, 18), (4, 20))
    sight196 = {(3, 24), (4, 24), (5, 24)}  # (6,24) has the collision bit
    sight195 = {(10, 29)}  # (10,30) has the collision bit
    assert attr(6, 24) & 0x8000 and attr(10, 30) & 0x8000
    assert path[-1] in sight196 and after[-1] in sight195
    objects = set(by_xz)
    # before any fight: every way to 195 (its line, or a talk tile beside it) passes 196's line
    talk195 = {(9, 28), (11, 28), (10, 29)}
    reach, hit = _hand_reach(attr, objects, (15, 2), sight196 | sight195)
    assert hit == {(3, 24)} and not (reach - hit) & talk195
    # after 196: 195 is next, and (15,2) beside the only warp is out of reach (leave [])
    reach, hit = _hand_reach(attr, objects, (3, 24), sight195)
    assert hit == {(10, 29)} and (15, 2) not in reach
    row = _committed_row("Oreburgh Mine")
    assert row["orders"] == [[[196], [195]]]
    assert row["leave_gaps"] == [] and row["heal_gaps"] == []


@pytest.mark.slow
def test_rom_c9_route_213_lobby_and_heal_by_hand(rom_files):
    # (c) header 373 doors 3 (706,818) and 4 (706,813) lead to header 376 warps 1 (8,11) and
    # 0 (8,2), and back; 376 obj 1 (the healer, script 2) on (11,4) behind (11,5), a TABLE tile
    # talked across. The lobby-end order: coord 12 (699,832) value 7 on var 0x40FA places 107 on
    # (700,832); west of it the beach is one row (z 831 walls, z 833 WATER_SEA, no Surf before
    # the Fen Badge), so with 107 standing there the lobby is behind it: no heal. The west
    # order: the var is not known there, the story coord 12 blocks, the lobby stays behind it:
    # no heal.
    _, warps373, coords = _raw_events(rom_files, 373)
    assert warps373[3] == (706, 818, 376, 1) and warps373[4] == (706, 813, 376, 0)
    objs376, warps376, _ = _raw_events(rom_files, 376)
    assert warps376 == [(8, 2, 373, 4), (8, 11, 373, 3)]
    assert (objs376[1][5], objs376[1][12], objs376[1][13]) == (2, 11, 4)
    assert _raw_attr(rom_files, 376)(11, 5) == 0x8000 | TABLE_B
    assert (12, 699, 832, 1, 1, 0, 7, 0x40FA) in coords
    attr = _raw_attr(rom_files, 373)
    assert attr(699, 832) == SAND_B and attr(700, 832) == SAND_B
    for x in range(693, 700):
        assert attr(x, 831) & 0x8000 and attr(x, 833) == WATER_SEA_B
    row = _committed_row("Route 213")
    assert row["heal_gaps"] == [] and row["leave_gaps"] == [[], [1, 2, 3, 4]]
