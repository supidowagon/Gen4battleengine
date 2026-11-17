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
    clone_scripts,
    field_rules,
    narrowed_terrains,
    past_table_starts,
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


def _narrow(
    world, objects=(), coords=(), script=None, other=(), rows=(), where=None, extra=(), any_=False
):
    rules, fr, headers, matrices, lands = world
    script = script or _Script()
    facts = ScriptFacts(script, _Script([("SCRCMD_STOPMUSIC", ()), ("SCRCMD_END", ())]), fr)
    sites = {
        0: HeaderSites(tuple(objects), tuple(coords), frozenset(other), "f", tuple(extra), any_)
    }
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
    # A battle of an entry the header never runs (no event, init script or C-started id of
    # it; E1 ruling G-38) starts nothing there: the header gives the trainer no terrain.
    got = _narrow(world, coords=[CoordEvent(2, 3, 3, 1, 2)], script=_Script(battle))
    assert got == {(9, 0): frozenset()}


def test_a_tag_battles_variable_partner_operand_names_only_partners(decomp, world):
    # ScrCmd_StartTagBattle (scrcmd_trainer.c:146-161) reads the partner first. Entry 4 is
    # C-started (the start menu's retire, start_menu.c:1457; the file sets the Pal Park flag,
    # E1 ruling G-42) and its partner operand is a
    # variable no constant reaches: it names a partner, never trainer 9, so 9's coord battle
    # (entry 1) still narrows (E1 ruling G-38); before, the operand matched every trainer.
    coord = CoordEvent(1, 3, 3, 1, 2)
    battle = [("SCRCMD_STARTTRAINERBATTLE", (9, 0)), ("SCRCMD_END", ())]
    tag = [("SCRCMD_STARTTAGBATTLE", (0x8004, 414, 415)), ("SCRCMD_END", ())]
    stop = [("SCRCMD_SETCLEARINCATCHINGSHOWFLAG", (0,)), ("SCRCMD_END", ())]
    script = _Script(battle, stop, stop, tag)
    assert _narrow(world, coords=[coord], script=script) == {(9, 0): frozenset({GRASS})}
    # A trainer some script makes a partner (here through the partner variable) is matched.
    partner = [("SCRCMD_SETVARFROMVALUE", (PARTNER_VAR, 9)), ("SCRCMD_END", ())]
    script = _Script(battle, partner, stop, tag)
    assert _narrow(world, coords=[coord], script=script) == {}
    # A partner operand variable resolves to the constants its paths assign (as the
    # trainer -> map join resolves operands, maps.script_battles): 9 is that partner.
    tag9 = [
        ("SCRCMD_SETVARFROMVALUE", (0x8004, 9)),
        ("SCRCMD_STARTTAGBATTLE", (0x8004, 414, 415)),
        ("SCRCMD_END", ()),
    ]
    assert _narrow(world, coords=[coord], script=_Script(battle, stop, stop, tag9)) == {}


def test_retire_and_last_ball_run_only_where_the_pal_park_flag_is_set(decomp, world):
    # E1 ruling G-42: local ids 3 and 4 are started by game code only in the catching show
    # (encounter.c:636-637; the Retire option, start_menu.c:279-287, :385, :1452-1457) or by
    # overlay 9. A header whose file never runs SetClearInCatchingShowFlag
    # (scrcmd_catching_show.c:24-38) or the Distortion World's features runs neither, so
    # trainer 9, named only by entry 4, gets no terrain there; else the whole header.
    battle = [("SCRCMD_STARTTRAINERBATTLE", (9, 0)), ("SCRCMD_END", ())]
    stop = [("SCRCMD_END", ())]
    assert _narrow(world, script=_Script(stop, stop, stop, battle)) == {(9, 0): frozenset()}
    for gate in (
        "SCRCMD_SETCLEARINCATCHINGSHOWFLAG",
        "SCRCMD_INITPERSISTEDMAPFEATURESFORDISTORTIONWORLD",
    ):
        opened = [(gate, (0,)), ("SCRCMD_END", ())]
        assert _narrow(world, script=_Script(opened, stop, stop, battle)) == {}


def test_clone_objects_run_the_script_of_the_object_they_keep():
    # E1 ruling G-40: a clone (script NO_SCRIPT) of header 20's local 3 keeps that object
    # across the header change (map_object.c:772-801) and runs its script 7; a clone whose
    # flag header has no such object is built fresh from the clone event (map_object.c:
    # 236-255) and runs no local id.
    clone = (3, 20, 65535)
    assert clone_scripts({10: [clone], 20: [(3, 0, 7)]}) == {10: {7}}
    assert clone_scripts({10: [clone], 20: [(4, 0, 7)]}) == {10: set()}


def test_a_past_table_id_runs_its_table_words_target(decomp, world):
    # E1 ruling G-40: ScriptContext_JumpToOffsetID (script_manager.c:412-416) reads the word
    # at 4 * (id - 1) without a bound. A 1-entry, 16-byte file: id 2 reads 0 at 4 and runs
    # offset 8 (an extra start); id 3 jumps past the file and id 5's word lies past the file's
    # end, both into heap memory the decomp does not fix: any entry may run.
    script = bytes(4) + (0).to_bytes(4, "little") + (256).to_bytes(4, "little") + bytes(4)
    assert past_table_starts(script, [4], [1, 2, 2000]) == ((8,), False)
    assert past_table_starts(script, [4], [3]) == ((), True)
    assert past_table_starts(script, [4], [5]) == ((), True)
    # An extra start naming trainer 9 keeps the whole header; none, and 9 gets no terrain.
    battle = [("SCRCMD_STARTTRAINERBATTLE", (9, 0)), ("SCRCMD_END", ())]
    stop = [("SCRCMD_END", ())]
    script = _Script(stop, battle)
    assert _narrow(world, script=script) == {(9, 0): frozenset()}
    assert _narrow(world, script=script, extra=[script.entries[1]]) == {}
    assert _narrow(world, script=script, any_=True) == {}


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
    # E1 ruling G-40: Route 203's past-table id 22 may run any entry, so the rival's coord
    # battle (247-249) keeps the whole header.
    assert (len(changed), len(resolved)) == (259, 200)
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


@pytest.fixture(scope="module")
def kaizo_rows(kaizo_files, decomp, symbols):
    return _rom_rows(kaizo_files, decomp, symbols)


def test_kaizo_venus_and_luna_fight_in_jubilife(kaizo_rows):
    # E1 ruling G-42: Venus (414) and Luna (415)'s tag battle is entry 4 of Jubilife's file
    # (a coord event) and of Route 202's copy (scr_seq 428), where only the start menu's
    # Retire could run it, which needs the Pal Park flag Route 202 never sets: Route 202 gives
    # them no terrain, Jubilife's whole header TERRAIN_BUILDING.
    where, rows = kaizo_rows
    for tid in (414, 415):
        assert 343 in where[tid]
        assert rows[tid] == (_t("BUILDING"), [_t("BUILDING")])


def test_kaizo_clowns_fight_in_jubilife(kaizo_rows):
    # Clowns Ronald (252) and Moe (254) (E1 ruling G-38, closing G-36): the join places them
    # on Jubilife City (header 3, its objects run their entries 15 and 14) and on Route 202
    # (header 343), whose file (scr_seq 428, that header's alone) carries their battles as
    # entries 14 and 15, which no Route 202 event, init script or C-started id runs. Its
    # C-started entry 4 is a tag battle whose partner operand VAR_0x8004 names a partner
    # (scrcmd_trainer.c:146-161), not them. So Route 202 gives them no terrain, and Jubilife's
    # whole header gives TERRAIN_BUILDING.
    where, rows = kaizo_rows
    for tid in (252, 254):
        assert sorted(where[tid]) == [3, 343]
        assert rows[tid] == (_t("BUILDING"), [_t("BUILDING")])


def test_kaizo_route_203_fights(kaizo_files, decomp, symbols, kaizo_rows):
    from tools.rom.readers.maps import (
        LAND_DATA_NARC,
        MATRIX_NARC,
        header_arrivals,
        map_header_battle_fields,
        rom_terrain_rules,
    )
    from tools.terrain import parse_matrix, rom_narrowing_inputs

    where, rows = kaizo_rows
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
