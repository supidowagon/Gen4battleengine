"""``tools.kaizo.gauntlet_map`` (E3-B3-2): the map read on synthetic maps, and, ROM-backed (slow;
skips without the Kaizo ROM and the decomp), the tool's output for every group against the facts
in ``data/kaizo/overrides.toml``."""

import os
import re
import struct
from pathlib import Path

import pytest

from tools.kaizo.gauntlet_map import (
    CLOCK_SIDE,
    CLOCK_STATES,
    FOUNTAIN_WIDTH,
    SPLIT_BADGES,
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
    Sighting,
    StaticMaps,
    Walker,
    Warp,
    _wayward_next,
    _wayward_walker,
    bdhc_heights,
    chain_facts,
    eterna_clock_tables,
    gap_fields,
    gym_reach,
    live_table,
    pc_reachable,
    pc_spots,
    runs,
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
    rock, boulder = npc(5, 3, 0, script=10001), npc(6, 3, 0, script=10002)
    tree = npc(7, 3, 0, script=10000)
    far = Pos(1, 6, 0)
    a, b = area(1, ["......."], [rock]), area(1, ["......."], [boulder])
    c = area(1, ["......."], [tree])
    smash = FieldMoves(rock_smash=True)
    assert far in Walker(StaticMaps([a]), moves=smash).walk(Pos(1, 0, 0)).seen
    assert far not in Walker(StaticMaps([a]), moves=smash).walk(Pos(1, 0, 0), removable=False).seen
    assert far not in Walker(StaticMaps([b]), moves=smash).walk(Pos(1, 0, 0)).seen
    assert far in Walker(StaticMaps([b]), moves=FieldMoves(strength=True)).walk(Pos(1, 0, 0)).seen
    assert far not in Walker(StaticMaps([c]), moves=smash).walk(Pos(1, 0, 0)).seen
    assert far in Walker(StaticMaps([c]), moves=FieldMoves(cut=True)).walk(Pos(1, 0, 0)).seen


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
    assert w.spot == {(1,): Sighting(Pos(1, 2, 0), "arrival")} and set(w.seen) == {Pos(1, 2, 0)}


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


def test_a_walking_member_raises_and_a_walking_non_member_is_taken_as_beaten():
    walker = trainer(3, 0, 0, EAST, 4, movement=41)
    a = area(1, ["......"], [walker])
    with pytest.raises(MapReadError, match="walks"):
        Walker(StaticMaps([a]), members={3}).sight_tiles(1, walker)
    other = Walker(StaticMaps([a]), members={4})
    assert other.sight_tiles(1, walker) == [] and other.sight_at(1, 1, 0, frozenset()) == []
    assert Pos(1, 5, 0) in other.walk(Pos(1, 4, 0)).seen
    assert (3,) not in other.walk(Pos(1, 4, 0)).spot  # not even talked to


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


def test_the_split_badges_follow_the_split_order():
    from tools.extract_kaizo_docs import SPLITS

    assert [s for s, _ in SPLIT_BADGES] == list(SPLITS)
    assert FieldMoves.for_split("Roark") == FieldMoves()
    assert FieldMoves.for_split("Wake") == FieldMoves(cut=True, rock_smash=True)
    assert FieldMoves.for_split("Galactic", partner=True) == FieldMoves(
        cut=True, rock_smash=True, strength=True
    )


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
    # know blocks, except UNKNOWN_COORDS_PASSED (Route 213's 0x40FA story coords, open). The
    # list of such coords in the headers the walks reach is pinned: a new one fails here.
    from tools.kaizo.gauntlet_map import (
        BUILDINGS,
        OREBURGH,
        ROUTE_204,
        ROUTE_213,
        ROUTE_213_HEALERS,
        UNKNOWN_COORDS_PASSED,
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
    assert found == {(75, 2, 0x407A)} | {(h, s, 0x40FA) for h, s in UNKNOWN_COORDS_PASSED}


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
    # lobby end or the west end; the lobby healer is reached at every gap only from the lobby end
    d = derived["Route 213"]
    assert {tuple(t for f in o for t in f) for o in d["orders"]} == {
        (106, 103, 105, 111, 110),
        (110, 111, 105, 103, 106),
    }
    # the healer's reach depends on the order: no flat heal_gaps (live_next holds it)
    assert d["leave_gaps"] == [[1, 2, 3, 4]] * 2 and d["heal_gaps"] is None
    assert d["pc_gaps"] == []


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
    assert [len(o) for o in d["orders"]] == [6, 7]
    assert d["partner"] == {"trainer": 612, "join_before": [1]}
    assert d["leave_gaps"] == [[], []] and d["escape"] is True
    maps = RomMaps(rom_files)
    joined = _wayward_walker(maps, True)
    starts, done = {WAYWARD_START}, frozenset()
    for pair in ((383, 443), (380, 458), (450, 496)):
        nxt = set()
        for start in starts:
            _, fights = _wayward_next(joined, start, done)
            nxt |= {s for f, sights in fights.items() if set(f) == set(pair) for s in sights}
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
    # depends on the count. Route 213's table is stored for its per-order heal (the lobby healer
    # is reached at every gap after 106 first, at none after 110 first; heal_gaps is absent).
    # Route 204 North's leave depends on the count (after 13: at one live, not at two).
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
    assert {n for n, d in derived.items() if d["live_next"] is not None} == {
        "Route 204 North",
        "Route 213",
    }
    heal = {
        tuple(map(tuple, r["fought"]))
        for r in derived["Route 213"]["live_next"]
        if "heal" in r.get("gap", ())
    }
    assert heal == {
        ((106,),),
        ((106,), (103,)),
        ((106,), (103,), (105,)),
        ((106,), (103,), (105,), (111,)),
    }
    absent = {
        (n, k)
        for n, d in derived.items()
        for k in ("leave_gaps", "heal_gaps", "pc_gaps")
        if d[k] is None
    }
    assert absent == {("Route 204 North", "leave_gaps"), ("Route 213", "heal_gaps")}
