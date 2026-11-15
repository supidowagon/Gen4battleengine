"""Battle terrain narrowed per trainer to the tiles its battle can start on
(``tools.terrain.narrowed_terrains``; docs/kaizo/k3b2-rules.md §9.3 step 2)."""

import pytest

from nuzline.data.trainer import TERRAIN_NAMES
from tools.terrain import (
    Cmd,
    CoordEvent,
    HeaderSites,
    Matrix,
    ObjectEvent,
    ScriptFacts,
    field_rules,
    narrowed_terrains,
    terrain_rules,
    tile_behaviours,
)

GRASS, PLAIN = TERRAIN_NAMES.index("TERRAIN_GRASS"), TERRAIN_NAMES.index("TERRAIN_PLAIN")
# The terrains only one fixture tile gives, so each rule's own tile shows in the result.
SAND, SNOW, ICE = (TERRAIN_NAMES.index(f"TERRAIN_{n}") for n in ("SAND", "SNOW", "ICE"))
SOUTH = 1
PARTNER_VAR = 0x4100  # any variable id; the fake scripts never set it


def _t(name: str) -> int:
    return TERRAIN_NAMES.index(f"TERRAIN_{name}")


class _Script:
    """A script file as ``ScriptFacts`` reads it: one command list per entry (local id 1..),
    each ``(SCRCMD_*, operands)``, run in order."""

    def __init__(self, *entries):
        self.cmds, self.entries, self.partner_var = [], [], PARTNER_VAR
        for cmds in entries:
            self.entries.append(len(self.cmds))
            for i, (const, args) in enumerate(cmds):
                nxt = len(self.cmds) + 1 if i + 1 < len(cmds) else None
                self.cmds.append(Cmd(const, tuple(args), None, nxt))

    def cmd(self, node):
        return self.cmds[node]


@pytest.fixture(scope="module")
def world(decomp):
    """One 32x32 PLAIN cell (header 0, BACKGROUND_PLAIN) with a trainer at (10, 10) facing
    south, sight 4: grass on (10, 11), (10, 12), a collision tile on (10, 13) and sand past it
    on (10, 14) (only a sight line that ignores the collision reaches it); east of it grass on
    (11, 10), plain to (14, 10) and snow on (15, 10), one tile past the sight range; talk
    tiles: grass at (11, 10), a collision plain tile at (9, 10), a table with the collision bit
    at (10, 9) and ice beyond it at (10, 8) (only the across-the-table talk rule reaches it).
    Grass at (3, 3), (3, 4) for a coord event."""
    rules = terrain_rules(decomp)
    fr = field_rules(decomp)
    b = tile_behaviours(decomp)
    grass, table = b["TILE_BEHAVIOR_TALL_GRASS"], b["TILE_BEHAVIOR_TABLE"]
    sand, snow, ice = b["TILE_BEHAVIOR_SAND"], b["TILE_BEHAVIOR_SNOW_DEEP"], b["TILE_BEHAVIOR_ICE"]
    side = rules.tiles_x
    attrs = [0] * (side * side)
    for x, z, a in (
        (10, 11, grass),
        (10, 12, grass),
        (10, 13, rules.collision),
        (10, 14, sand),
        (15, 10, snow),
        (11, 10, grass),
        (9, 10, rules.collision),
        (10, 9, table | rules.collision),
        (10, 8, ice),
        (3, 3, grass),
        (3, 4, grass),
    ):
        attrs[z * side + x] = a
    offset, count = rules.attributes
    land = (2 * count).to_bytes(4, "little") + bytes(offset - 4)
    land += b"".join(a.to_bytes(2, "little") for a in attrs)
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    return rules, fr, [(0, plain)], [Matrix(1, 1, None, (0,))], [land]


def _movement(decomp, name):
    from tools.extract_decomp import _constants

    return _constants(decomp, "movement_types", "MOVEMENT_TYPE_").index(name)


def _trainer_type(decomp, name):
    from tools.extract_decomp import _constants

    return _constants(decomp, "trainer_types", "TRAINER_TYPE_").index(name)


def _narrow(world, objects=(), coords=(), script=None, other=(), rows=(), where=None):
    rules, fr, headers, matrices, lands = world
    script = script or _Script()
    facts = ScriptFacts(script, _Script([("SCRCMD_STOPMUSIC", ()), ("SCRCMD_END", ())]), fr)
    sites = {0: HeaderSites(tuple(objects), tuple(coords), frozenset(other), "f")}
    where = where or {7: [0], 9: [0]}
    return narrowed_terrains(
        rules, fr, headers, matrices, lands, where, {}, sites, {"f": facts}, rows
    )


def _kaitlin_like(decomp, movement="MOVEMENT_TYPE_LOOK_SOUTH", local=2):
    return ObjectEvent(
        local,
        _movement(decomp, movement),
        _trainer_type(decomp, "TRAINER_TYPE_NORMAL"),
        SOUTH,
        4,
        10,
        10,
        7,
    )


def test_an_object_trainer_narrows_to_its_sight_and_talk_tiles(decomp, world):
    # Sight south stops at the collision tile (IsPathInterrupted, trainer_encounter.c:310-343),
    # so (10, 14) is never a battle tile; the collision talk tiles are dropped and the table
    # lets the player talk from (10, 8) (sub_0203C9D4, unk_0203C954.c:62-84): its ice is the
    # only ICE, and the sand past the collision never shows.
    got = _narrow(world, objects=[_kaitlin_like(decomp)])
    assert got == {(7, 0): frozenset({GRASS, ICE})}


def test_sight_stops_at_its_range(decomp, world):
    # Facing east too (LOOK_EAST plus its initial south): sight 4 reaches (14, 10), never the
    # snow on (15, 10) (GetDistance*, trainer_encounter.c:243-305: 1..data[0] tiles).
    east = _kaitlin_like(decomp, "MOVEMENT_TYPE_LOOK_EAST")
    assert _narrow(world, objects=[east]) == {(7, 0): frozenset({GRASS, PLAIN, ICE})}


def test_a_moving_or_moved_object_keeps_the_whole_header(decomp, world):
    wander = _kaitlin_like(decomp, "MOVEMENT_TYPE_WANDER_AROUND")
    assert _narrow(world, objects=[wander]) == {}
    moves = _Script([("SCRCMD_APPLYMOVEMENT", (2, 0)), ("SCRCMD_END", ())])
    assert _narrow(world, objects=[_kaitlin_like(decomp)], script=moves) == {}


def test_a_vs_seeker_rematch_spins_the_trainer(decomp, world):
    # A rematch-only id (base 5, rematch 7, fought at the base's objects) is fought after the
    # Vs. Seeker spins the trainer (vs_seeker.c:659-690), so it sees along every facing: east
    # of it are grass and PLAIN; north and west are blocked by collision tiles.
    got = _narrow(world, objects=[_kaitlin_like(decomp)], rows=[(5, [7])])
    assert got == {(7, 0): frozenset({GRASS, PLAIN, ICE})}


def test_a_rematch_rows_base_keeps_its_first_battles_facings(decomp, world):
    # NoUniqueRematches (vs_seeker.c:53) rematches the trainer as itself; its first battle,
    # the split's fight, is fought unspun (ruling B4-19).
    got = _narrow(world, objects=[_kaitlin_like(decomp)], rows=[(7, [7])])
    assert got == {(7, 0): frozenset({GRASS, ICE})}


def test_map_features_keep_collision_talk_tiles(decomp, world):
    # A header that turns on collision-overriding map features (scrcmd.c:4700-4785) may let
    # the player stand on a collision tile next to the trainer.
    features = _Script([("SCRCMD_INITPERSISTEDMAPFEATURESFORPASTORIAGYM", ()), ("SCRCMD_END", ())])
    got = _narrow(world, objects=[_kaitlin_like(decomp)], script=features)
    assert got == {(7, 0): frozenset({GRASS, PLAIN, ICE})}


def test_a_coord_event_battle_narrows_to_its_rectangle(decomp, world):
    coord = CoordEvent(1, 3, 3, 1, 2)
    battle = [
        ("SCRCMD_LOCKALL", ()),
        ("SCRCMD_APPLYMOVEMENT", (5, 0)),
        ("SCRCMD_CALLCOMMONSCRIPT", (2000,)),
        ("SCRCMD_STARTTRAINERBATTLE", (9, 0)),
        ("SCRCMD_END", ()),
    ]
    assert _narrow(world, coords=[coord], script=_Script(battle)) == {(9, 0): frozenset({GRASS})}
    # The player moved before the battle, an unknown command, or another starter of the
    # same script: the whole header.
    _, fr, *_ = world
    moved = [("SCRCMD_APPLYMOVEMENT", (fr.localid_player, 0)), *battle[3:]]
    assert _narrow(world, coords=[coord], script=_Script(moved)) == {}
    warp = [("SCRCMD_WARP", (1, 2, 3, 4)), *battle[3:]]
    assert _narrow(world, coords=[coord], script=_Script(warp)) == {}
    assert _narrow(world, coords=[coord], script=_Script(battle), other=[1]) == {}
    # A battle not started by the header's coord events is not narrowed.
    assert _narrow(world, coords=[CoordEvent(2, 3, 3, 1, 2)], script=_Script(battle)) == {}


def test_the_constants_follow_the_rom_reader():
    from tools.rom.readers import maps
    from tools.terrain import BRANCH_COMMANDS, C_STARTED_SCRIPTS, COMMON_SCRIPTS, VARS_START

    assert C_STARTED_SCRIPTS == maps.C_STARTED_SCRIPTS
    assert BRANCH_COMMANDS == maps._BRANCHES
    assert COMMON_SCRIPTS[0] == maps.COMMON_SCRIPTS and VARS_START == maps.VARS_START


# --- the games ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def decomp_rows(decomp):
    """Vanilla, from the decomp: (whole-header rows, narrowed rows) by trainer id."""
    from tools.extract_decomp import _constants, _map_headers, _trainer_maps, _trainer_terrains
    from tools.terrain import (
        decomp_header_battle_fields,
        decomp_land_data,
        decomp_matrices,
        decomp_warp_tiles,
        header_names,
        header_terrains,
        trainer_terrains,
    )

    trainers = _constants(decomp, "trainers", "TRAINER_")
    headers = _map_headers(decomp)
    where = _trainer_maps(decomp, headers)
    narrowed = _trainer_terrains(decomp, trainers, headers, where)
    hid = {n: i for i, n in enumerate(header_names(decomp))}
    ids = {trainers.index(t): {hid[h] for h in hs} for t, hs in where.items()}
    arrivals = {hid[n]: decomp_warp_tiles(decomp, h["eventsArchiveID"]) for n, h in headers.items()}
    by_header = header_terrains(
        terrain_rules(decomp),
        decomp_header_battle_fields(decomp),
        decomp_matrices(decomp),
        decomp_land_data(decomp),
        {h for hs in ids.values() for h in hs},
        arrivals,
    )
    whole = trainer_terrains(len(trainers), ids, by_header)
    return whole, [(r["terrain"], r["terrains"]) for r in narrowed]


def test_vanilla_narrowing_only_resolves_mixed_sets(decomp_rows):
    whole, narrowed = decomp_rows
    changed = [t for t, (w, n) in enumerate(zip(whole, narrowed, strict=True)) if w != n]
    for t in changed:
        (_, before), (_, after) = whole[t], narrowed[t]
        assert set(after) < set(before) and len(before) > 1, t  # only a mixed set shrinks
    resolved = [t for t in changed if narrowed[t][0] != -1]
    assert (len(changed), len(resolved)) == (262, 203)
    # Roark (246) was BUILDING already.
    assert narrowed[246] == whole[246] == (_t("BUILDING"), [_t("BUILDING")])


def _rom_rows(rom, decomp, symbols):
    from tools.extract_decomp import _constants
    from tools.rom.readers.maps import trainer_start_conditions, trainer_terrain_rows

    where: dict[int, list[int]] = {}
    trainer_start_conditions(rom, decomp, symbols, maps=where)
    n = len(_constants(decomp, "trainers", "TRAINER_"))
    rows = trainer_terrain_rows(rom, decomp, symbols, where, n)
    return where, [(r["terrain"], r["terrains"]) for r in rows]


def test_vanilla_rom_narrows_as_the_decomp(vanilla_files, decomp, symbols, decomp_rows):
    # The ROM reader reads the same events and scripts the decomp's sources assemble to.
    assert _rom_rows(vanilla_files, decomp, symbols)[1] == decomp_rows[1]


def test_kaizo_route_203_fights(kaizo_files, decomp, symbols):
    from tools.rom.readers.maps import (
        LAND_DATA_NARC,
        MATRIX_NARC,
        header_arrivals,
        map_header_battle_fields,
        rom_terrain_rules,
    )
    from tools.terrain import parse_matrix, rom_narrowing_inputs

    where, rows = _rom_rows(kaizo_files, decomp, symbols)
    # Route 203 (header 344, BACKGROUND_PLAIN) mixes PLAIN, GRASS and WATER tiles. Barry #2
    # (247-249) starts from the coord event at (196, 757), 1x4, of script 6, whose commands
    # before the battle leave the player in place: four PLAIN tiles.
    for tid in (247, 248, 249):
        assert where[tid] == [344]
        assert rows[tid] == (PLAIN, [PLAIN])
    # Lass Kaitlin (323): LOOK_SOUTH at (242, 755), sight 6, so her first battle starts on a
    # GRASS tile, and that first battle is the split's fight (ruling B4-19).
    # NoUniqueRematches(TRAINER_LASS_KAITLIN) (vs_seeker.c:252) lets the Vs. Seeker rematch her
    # as 323 once beaten, spun, which also sees PLAIN tiles north and east: a rematch row's
    # base keeps its own facings, so only a rematch-only id spins.
    assert where[323] == [344]
    assert rows[323] == (GRASS, [GRASS])
    sites, scripts, rematch_rows, fr = rom_narrowing_inputs(kaizo_files, symbols, decomp, 1000)
    assert any(r[0] == 323 and 323 in r[1] for r in rematch_rows)
    first = narrowed_terrains(
        rom_terrain_rules(kaizo_files, symbols, decomp),
        fr,
        map_header_battle_fields(kaizo_files, symbols),
        [parse_matrix(m, i) for i, m in enumerate(kaizo_files.narc(MATRIX_NARC))],
        kaizo_files.narc(LAND_DATA_NARC),
        {323: [344]},
        header_arrivals(kaizo_files, symbols),
        sites,
        scripts,
        rematch_rows,
    )
    assert first == {(323, 344): frozenset({GRASS})}
