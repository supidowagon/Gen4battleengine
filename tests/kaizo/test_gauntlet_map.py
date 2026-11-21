"""``tools.kaizo.gauntlet_map`` (E3-B3-2): the map read on synthetic maps, and, ROM-backed (slow;
skips without the Kaizo ROM and the decomp), the tool's output for every group against the facts
in ``data/kaizo/overrides.toml``."""

import os
from pathlib import Path

import pytest

from tools.kaizo.gauntlet_map import (
    CLOCK_SIDE,
    CLOCK_STATES,
    FOUNTAIN_WIDTH,
    WAYWARD_HEADER,
    WAYWARD_PARTNER_VAR,
    WAYWARD_START,
    WAYWARD_WALKOUT,
    Area,
    ChainSpec,
    Coord,
    MapObject,
    MapReadError,
    Pos,
    StaticMaps,
    Walker,
    Warp,
    _wayward_next,
    _wayward_walker,
    chain_facts,
    eterna_clock_tables,
    gym_reach,
    pc_reachable,
    runs,
)
from tools.rom.inputs import REPO_ROOT

# --- synthetic maps ---------------------------------------------------------------------------

TILE = {".": 0, "#": 0x8000, "D": 0x8069, "T": 0x8080, "P": 0x8083, "W": 0x8000}
EAST, SOUTH, WEST, NORTH = 3, 1, 2, 0


def area(header, rows, objects=(), warps=(), coords=()):
    """An area from ASCII rows (z down, x across): ``.`` free, ``#`` wall, ``D`` door (collision,
    behaviour 0x69), ``T`` table, ``P`` PC; a space is no tile."""
    tiles = {(x, z): TILE[c] for z, row in enumerate(rows) for x, c in enumerate(row) if c != " "}
    return Area(header, tiles, tuple(objects), tuple(warps), tuple(coords))


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
    assert [r.sequence for r in found] == [(2, 1)]
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


def test_rocks_and_boulders_block_until_removed():
    rock, boulder = npc(5, 3, 0, script=10001), npc(6, 3, 0, script=10002)
    far = Pos(1, 6, 0)
    a, b = area(1, ["......."], [rock]), area(1, ["......."], [boulder])
    assert far in Walker(StaticMaps([a])).walk(Pos(1, 0, 0)).seen
    assert far not in Walker(StaticMaps([a])).walk(Pos(1, 0, 0), removable=False).seen
    assert far not in Walker(StaticMaps([b])).walk(Pos(1, 0, 0)).seen
    assert far in Walker(StaticMaps([b]), strength=True).walk(Pos(1, 0, 0)).seen


def test_a_warp_into_a_sight_line_starts_that_fight():
    # the arrival tile (2, 0) of header 2 lies in trainer 7's line
    here = area(1, ["........"], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["........"], [trainer(7, 6, 0, WEST, 4)], [Warp(0, 2, 0, 1, 0)])
    w = Walker(StaticMaps([here, there])).walk(Pos(1, 0, 0))
    assert w.spot[7].how == "arrival" and w.spot[7].pos == Pos(2, 2, 0)
    assert all(p.header == 1 for p in w.seen)  # the walk never stands in header 2


def test_the_start_tile_is_checked_too():
    a = area(1, ["......"], [trainer(1, 4, 0, WEST, 3)])
    w = Walker(StaticMaps([a])).walk(Pos(1, 2, 0))
    assert w.spot == {1: (Pos(1, 2, 0), "arrival")} and set(w.seen) == {Pos(1, 2, 0)}


def test_a_defeated_trainer_no_longer_sees():
    a = area(1, ["......"], [trainer(1, 5, 0, WEST, 4)])
    w = Walker(StaticMaps([a]))
    assert 1 in w.walk(Pos(1, 0, 0)).spot
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
    assert w.spot[9] == (Pos(1, 1, 0), "talk")
    assert Walker(StaticMaps([a])).talk_tiles(Pos(1, 0, 0), 1, 9) == [(1, 0)]


def test_a_walking_member_raises_and_a_walking_non_member_is_taken_as_beaten():
    walker = trainer(3, 0, 0, EAST, 4, movement=41)
    a = area(1, ["......"], [walker])
    with pytest.raises(MapReadError, match="walks"):
        Walker(StaticMaps([a]), members={3}).sight_tiles(1, walker)
    other = Walker(StaticMaps([a]), members={4})
    assert other.sight_tiles(1, walker) == [] and other.sight_at(1, 1, 0, frozenset()) == []
    assert Pos(1, 5, 0) in other.walk(Pos(1, 4, 0)).seen
    assert 3 not in other.walk(Pos(1, 4, 0)).spot  # not even talked to


def test_a_matching_coord_raises_unless_the_caller_models_it():
    a = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 0, 0)])  # var 0 always matches
    with pytest.raises(MapReadError, match="not modelled"):
        Walker(StaticMaps([a])).walk(Pos(1, 0, 0))
    passed = Walker(StaticMaps([a]), coord_rule=lambda h, c, m: "pass")
    w = passed.walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in w.seen and w.crossed[0][1].script == 7
    blocked = Walker(StaticMaps([a]), coord_rule=lambda h, c, m: "block")
    assert Pos(1, 3, 0) not in blocked.walk(Pos(1, 0, 0)).seen


def test_a_coord_by_a_known_or_unknown_var():
    a = area(1, ["......"], coords=[Coord(7, 3, 0, 1, 1, 1, 0x4091)])
    unknown = Walker(StaticMaps([a])).walk(Pos(1, 0, 0))  # var not known: recorded only
    assert Pos(1, 5, 0) in unknown.seen and len(unknown.crossed) == 1
    off = Walker(StaticMaps([a]), vars={0x4091: 0}).walk(Pos(1, 0, 0))
    assert Pos(1, 5, 0) in off.seen
    with pytest.raises(MapReadError):
        Walker(StaticMaps([a]), vars={0x4091: 1}).walk(Pos(1, 0, 0))


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
    assert 383 in joined.spot and Pos(WAYWARD_HEADER, 3, 0) in joined.seen


def test_an_unknown_running_wayward_coord_raises():
    other = Coord(9, 3, 0, 1, 1, 0, WAYWARD_PARTNER_VAR)
    maps = StaticMaps([area(WAYWARD_HEADER, ["......"], coords=[other])])
    with pytest.raises(MapReadError, match="not modelled"):
        _wayward_walker(maps, False).walk(Pos(WAYWARD_HEADER, 0, 0))
    assert (
        Pos(WAYWARD_HEADER, 5, 0)
        in _wayward_walker(maps, True).walk(Pos(WAYWARD_HEADER, 0, 0)).seen
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
def derived(rom_files, decomp):
    from tools.kaizo.gauntlet_map import derive_groups

    return derive_groups(rom_files, decomp)


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
    assert {_fights(o): tuple(g) for o, g in zip(d["orders"], d["leave_gaps"], strict=True)} == {
        _fights(o): g for o, g in zip(f.orders, f.leave_gaps, strict=True)
    }
    assert d["escape"] is f.escape
    assert (tuple(d["pc_gaps"]), tuple(d["heal_gaps"]), tuple(d["locked_gaps"])) == (
        f.pc_gaps,
        f.heal_gaps,
        f.locked_gaps,
    )
    if f.partner is None:
        assert d["partner"] is None
    else:
        assert d["partner"] == {
            "trainer": f.partner.trainer,
            "join_before": list(f.partner.join_before),
        }
    assert set(derived) == set(_committed())


@pytest.mark.slow
def test_oreburgh_fights_195_then_196_and_can_leave_after_195(derived):
    d = derived["Oreburgh Mine"]
    assert d["orders"] == [[[195], [196]]] and d["leave_gaps"] == [[1]] and d["escape"] is True


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
def test_route_213_leaves_and_heals_at_every_gap(derived):
    d = derived["Route 213"]
    assert len(d["orders"]) == 120
    assert d["leave_gaps"] == [[1, 2, 3, 4]] * 120 and d["heal_gaps"] == [1, 2, 3, 4]


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
    pos, done = WAYWARD_START, frozenset()
    for pair in ((383, 443), (380, 458), (450, 496)):
        _, nxt = _wayward_next(joined, pos, done)
        pos, done = nxt[pair].pos, done | set(pair)
    assert joined.talk_tiles(pos, 284, 459, done) == [(34, 28), (35, 27)]
    assert joined.talk_tiles(pos, 284, 460, done) == []


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


def test_a_matching_coord_outside_the_strict_headers_is_only_recorded():
    here = area(1, ["......"], warps=[Warp(0, 3, 0, 2, 0)])
    there = area(2, ["....."], warps=[Warp(0, 0, 0, 1, 0)], coords=[Coord(7, 2, 0, 1, 1, 0, 0)])
    maps = StaticMaps([here, there])
    w = Walker(maps, strict=[1]).walk(Pos(1, 0, 0))
    assert Pos(2, 4, 0) in w.seen and [(p.header, c.script) for p, c in w.crossed] == [(2, 7)]
    with pytest.raises(MapReadError):
        Walker(maps).walk(Pos(1, 0, 0))
