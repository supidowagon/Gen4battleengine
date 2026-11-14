"""Battle terrain per trainer from the ROM (K3b2 Task 9, ``tools/terrain.py``,
``tools.rom.readers.maps.trainer_terrain_rows``; docs/kaizo/k3b2-rules.md §9)."""

import json
import re

import pytest

from nuzline.data.trainer import TERRAIN_NAMES, TERRAIN_UNKNOWN
from tools.terrain import (
    Matrix,
    TerrainError,
    calc_terrain,
    decomp_check_trainer_objects,
    header_terrains,
    terrain_rules,
    trainer_terrains,
)

ROARK = 246
BUILDING = TERRAIN_NAMES.index("TERRAIN_BUILDING")
# Mixed trainers (several terrains over the tiles their battles can start on) per game,
# after the per-trainer narrowing of rules §9.3 step 2 (415 and 371 over whole headers).
MIXED = {"CPUE": 217, "JAK7": 242}


def _t(name: str) -> int:
    return TERRAIN_NAMES.index(f"TERRAIN_{name}")


# --- the rules, read from the decomp ------------------------------------------------------


def test_terrain_names_are_the_decomp_enum(decomp):
    from tools.terrain import battle_constants

    assert tuple(battle_constants(decomp, "battle_terrains", "TERRAIN_")) == TERRAIN_NAMES
    terrain_h = (decomp / "include" / "constants" / "battle" / "terrain.h").read_text("utf-8")
    assert "#define TERRAIN_SPECIAL TERRAIN_AARON" in terrain_h


def test_version_pearl_is_the_decomp_enum(decomp):
    from tools.rom.readers.maps import VERSION_PEARL

    text = (decomp / "include" / "constants" / "versions.h").read_text("utf-8")
    body = text[text.index("enum Version {") :]
    names = re.findall(r"^\s+(VERSION_\w+)", body, re.MULTILINE)
    assert names.index("VERSION_PEARL") == VERSION_PEARL


def test_calc_terrain_follows_the_decomp_order(decomp):
    # CalcTerrain (src/field_battle_data_transfer.c:467-494): ice, grass, sand, snow, mud,
    # cave floor, surfable, then sTerrainForBackground; TERRAIN_MAX past the table.
    rules = terrain_rules(decomp)
    assert [rules.names[t] for _, t in rules.tests] == [
        "TERRAIN_ICE",
        "TERRAIN_GRASS",
        "TERRAIN_SAND",
        "TERRAIN_SNOW",
        "TERRAIN_GREAT_MARSH",
        "TERRAIN_CAVE",
        "TERRAIN_WATER",
    ]
    tall_grass, cave_floor, sea = 0x02, 0x08, 0x15  # map_tile_behaviors.h enum order
    indoors = rules.backgrounds.index("BACKGROUND_INDOORS_1")
    assert calc_terrain(rules, tall_grass, indoors) == _t("GRASS")
    assert calc_terrain(rules, cave_floor, indoors) == _t("CAVE")
    assert calc_terrain(rules, sea, indoors) == _t("WATER")
    assert calc_terrain(rules, 0, indoors) == BUILDING
    assert calc_terrain(rules, 0, rules.backgrounds.index("BACKGROUND_FOREST")) == _t("GRASS")
    assert calc_terrain(rules, 0, rules.water) == _t("WATER")
    assert calc_terrain(rules, 0, len(rules.by_background)) == _t("MAX")
    # A surfing player stands on MapObject_IsOnWater's tiles: surfable or a bridge over water.
    assert sea in rules.on_water and 0 not in rules.on_water
    # Spring Path and Seabreak Path rewrite eight matrix-0 cells (src/map_matrix.c:194-230).
    assert {(m, x, z) for m, x, z, _ in rules.patches} >= {(0, 23, 21), (0, 28, 15)}
    assert len(rules.patches) == 8


def _rules(decomp):
    return terrain_rules(decomp)


def _land(rules, behaviours: list[int]) -> bytes:
    offset, count = rules.attributes
    tiles = (behaviours * count)[:count]
    head = (2 * count).to_bytes(4, "little") + bytes(offset - 4)
    return head + b"".join(
        (b | 0x8000 * (i % 2)).to_bytes(2, "little") for i, b in enumerate(tiles)
    )


def test_header_terrains_read_only_the_headers_cells(decomp):
    rules = _rules(decomp)
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    lands = [_land(rules, [0]), _land(rules, [0x02]), _land(rules, [0x15])]
    # A 3x1 matrix: cells of headers 5, 6, 5 on land data 0, 1, 2.
    m = Matrix(3, 1, (5, 6, 5), (0, 1, 2))
    headers = [(0, plain)] * 7
    got = header_terrains(rules, headers, [m], lands, {5, 6})
    assert got[5] == {_t("PLAIN"), _t("WATER")}  # cell 1 (grass) belongs to header 6
    assert got[6] == {_t("GRASS")}
    # No header section: every cell is the loading header's.
    whole = Matrix(3, 1, None, (0, 1, 2))
    assert header_terrains(rules, [(0, plain)], [whole], lands, {0})[0] == {
        _t("PLAIN"),
        _t("GRASS"),
        _t("WATER"),
    }
    # A header none of its matrix's cells names: where it is entered cannot be read.
    assert header_terrains(rules, headers, [m], lands, {4})[4] == {TERRAIN_UNKNOWN}
    # INVALID_LAND_DATA_ID cells hold nothing to stand on; an unknown land id raises.
    holes = Matrix(2, 1, None, (rules.invalid_land, 0))
    assert header_terrains(rules, [(0, plain)], [holes], lands, {0})[0] == {_t("PLAIN")}
    with pytest.raises(TerrainError, match="land data 9"):
        header_terrains(rules, [(0, plain)], [Matrix(1, 1, None, (9,))], lands, {0})


def test_surfing_uses_the_water_background(decomp):
    # SetBackgroundAndTerrain (fbdt:496-505): BACKGROUND_WATER while surfing. In Platinum
    # every tile MapObject_IsOnWater accepts (the bridges over water included) carries the
    # surfable flag, so CalcTerrain already gives WATER there and surfing adds nothing.
    rules = _rules(decomp)
    water_test = next(s for s, t in rules.tests if t == _t("WATER"))
    assert rules.on_water <= water_test
    # The rule itself: without that tile test, a surfable tile adds the water background's.
    city = rules.backgrounds.index("BACKGROUND_CITY")
    bare = rules._replace(tests=tuple(x for x in rules.tests if x[1] != _t("WATER")))
    lands = [_land(rules, [min(rules.on_water)])]
    got = header_terrains(bare, [(0, city)], [Matrix(1, 1, None, (0,))], lands, {0})
    assert got[0] == {BUILDING, _t("WATER")}


def test_story_flag_land_data_counts_both_ways(decomp):
    rules = _rules(decomp)
    matrix, _, _, land = rules.patches[0]
    assert matrix == 0
    width = max(p[1] for p in rules.patches) + 1
    height = max(p[2] for p in rules.patches) + 1
    lands = [_land(rules, [0])] * (land + 1)
    lands[land] = _land(rules, [0x02])
    cells = tuple(0 for _ in range(width * height))
    m = Matrix(width, height, None, cells)
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    got = header_terrains(rules, [(0, plain)], [m], lands, {0})[0]
    assert got == {_t("PLAIN"), _t("GRASS")}


def test_trainer_terrain_one_mixed_or_unplaced():
    by_header = {1: frozenset({BUILDING}), 2: frozenset({BUILDING, 2}), 3: frozenset({0})}
    rows = trainer_terrains(5, {0: [1], 1: [1, 2], 2: [], 4: [1, 3]}, by_header)
    assert rows == [
        (BUILDING, [BUILDING]),
        (TERRAIN_UNKNOWN, [2, BUILDING]),
        (TERRAIN_UNKNOWN, []),
        (TERRAIN_UNKNOWN, []),
        (TERRAIN_UNKNOWN, [0, BUILDING]),
    ]
    unreadable = trainer_terrains(1, {0: [9]}, {9: frozenset({TERRAIN_UNKNOWN})})
    assert unreadable == [(TERRAIN_UNKNOWN, [TERRAIN_UNKNOWN])]


def test_clone_or_persistent_trainer_object_is_refused(tmp_path):
    # An object of another header survives a header change when it is cloned there
    # (src/map_object.c:772-801) or flagged persistent (:182, scrcmd.c:2501-2508).
    events = tmp_path / "res" / "field" / "events"
    scripts = tmp_path / "res" / "field" / "scripts"
    events.mkdir(parents=True)
    scripts.mkdir(parents=True)

    def write(name, objects, script=""):
        (events / f"{name}.json").write_text(json.dumps({"object_events": objects}), "utf-8")
        (scripts / f"{name}.s").write_text(script, "utf-8")

    headers = {
        "MAP_HEADER_A": {"eventsArchiveID": "a", "scriptsArchiveID": "a"},
        "MAP_HEADER_B": {"eventsArchiveID": "b", "scriptsArchiveID": "b"},
    }
    trainer = {"id": "LOCALID_X", "script": "TRAINER_X", "hidden_flag": "0"}
    write("a", [trainer])
    write("b", [{"clone_id": 0, "script": 65535, "hidden_flag": "MAP_HEADER_A"}])
    with pytest.raises(TerrainError, match="clones trainer object 0 of MAP_HEADER_A"):
        decomp_check_trainer_objects(tmp_path, headers)
    write("b", [{"clone_id": 1, "script": 65535, "hidden_flag": "MAP_HEADER_A"}])
    decomp_check_trainer_objects(tmp_path, headers)
    write("a", [trainer], "    SetObjectFlagIsPersistent LOCALID_X, TRUE\n")
    with pytest.raises(TerrainError, match="makes object LOCALID_X persistent"):
        decomp_check_trainer_objects(tmp_path, headers)
    write("a", [trainer], "    SetObjectFlagIsPersistent LOCALID_X, FALSE\n")
    decomp_check_trainer_objects(tmp_path, headers)
    write("a", [trainer], "    SetObjectFlagIsPersistent VAR_0x8004, TRUE\n")
    with pytest.raises(TerrainError, match="makes object VAR_0x8004 persistent"):
        decomp_check_trainer_objects(tmp_path, headers)


def test_a_header_reads_its_cells_in_every_loadable_matrix(decomp):
    # The matrix is loaded once, from the arrival header (src/field_map_change.c:380), and
    # FieldMap_ChangeZone only relabels the header (fieldmap.c:397-417): header 2's own
    # matrix 1 is one PLAIN cell, but it is also cell 1 of matrix 0, which header 3 loads.
    rules = _rules(decomp)
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    lands = [_land(rules, [0]), _land(rules, [0x20]), _land(rules, [0x02])]  # plain, ice, grass
    m0 = Matrix(2, 1, (3, 2), (2, 1))
    m1 = Matrix(1, 1, None, (0,))
    headers = [(1, plain), (1, plain), (1, plain), (0, plain)]
    got = header_terrains(rules, headers, [m0, m1], lands, {1, 2, 3})
    assert got[2] == {_t("PLAIN"), _t("ICE")}
    assert got[1] == {_t("PLAIN")}  # matrix 1 has no header section: header 1's own
    assert got[3] == {_t("GRASS")}
    # A matrix no header loads is never read.
    assert header_terrains(rules, [(1, plain)] * 3, [m0, m1], lands, {2})[2] == {_t("PLAIN")}


def test_a_warp_arrival_reads_its_own_tile(decomp):
    # Before the first step after a warp, the header is the warp's and the tile is its warp
    # event's (src/field_map_change.c:224-230), whatever that cell's label.
    rules = _rules(decomp)
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    side = rules.tiles_x
    tiles = [0] * side * side
    tiles[3 * side + 5] = 0x20  # ice at (5, 3)
    lands = [_land(rules, [0]), _land(rules, tiles)]
    m = Matrix(2, 1, (0, 1), (0, 1))  # cell 1 is labelled header 1
    headers = [(0, plain), (0, plain)]
    assert header_terrains(rules, headers, [m], lands, {0})[0] == {_t("PLAIN")}
    arrival = {0: [(side + 5, 3)]}
    assert header_terrains(rules, headers, [m], lands, {0}, arrival)[0] == {_t("PLAIN"), _t("ICE")}
    # The arrival's own cell is counted already; one outside the matrix cannot be read.
    assert header_terrains(rules, headers, [m], lands, {0}, {0: [(1, 1)]})[0] == {_t("PLAIN")}
    outside = header_terrains(rules, headers, [m], lands, {0}, {0: [(3 * side, 0)]})[0]
    assert outside == {_t("PLAIN"), TERRAIN_UNKNOWN}


def test_story_flag_guard_is_the_u8_matrix_id(decomp):
    # MapMatrix.matrixID is a u8 (include/map_matrix.h:18): matrix 256 passes the
    # `matrixID != 0` guard of the Spring/Seabreak rewrites (src/map_matrix.c:202, :223).
    rules = _rules(decomp)
    _, x, z, land = rules.patches[0]
    width, height = x + 1, z + 1
    lands = [_land(rules, [0])] * (land + 1)
    lands[land] = _land(rules, [0x02])
    m = Matrix(width, height, None, (0,) * (width * height))
    plain = rules.backgrounds.index("BACKGROUND_PLAIN")
    matrices = [m] * 257
    for matrix, want in ((256, {_t("PLAIN"), _t("GRASS")}), (255, {_t("PLAIN")})):
        assert header_terrains(rules, [(matrix, plain)], matrices, lands, {0})[0] == want


# --- the ROM -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rom_rows(vanilla_files, kaizo_files, decomp, symbols):
    """game code -> (trainer -> map join, terrain rows) of both ROMs."""
    from tools.extract_decomp import _constants
    from tools.rom.readers.maps import trainer_start_conditions, trainer_terrain_rows

    n = len(_constants(decomp, "trainers", "TRAINER_"))
    out = {}
    for rom in (vanilla_files, kaizo_files):
        where: dict[int, list[int]] = {}
        trainer_start_conditions(rom, decomp, symbols, maps=where)
        out[rom.game_code] = (where, trainer_terrain_rows(rom, decomp, symbols, where, n))
    return out


@pytest.mark.parametrize("game", ["CPUE", "JAK7"])
def test_roark_terrain_is_derived_as_building(
    game, rom_rows, vanilla_files, kaizo_files, decomp, symbols
):
    # Roark (246) is fought in header 47 (Oreburgh Gym): matrix 113, one cell of land data
    # 229, battleBG INDOORS_1; no tile matches a CalcTerrain test, so the background's
    # terrain, BUILDING, is the only one.
    from tools.rom.readers.maps import LAND_DATA_NARC, MATRIX_NARC, map_header_battle_fields
    from tools.terrain import land_behaviours, parse_matrix

    rom = vanilla_files if game == "CPUE" else kaizo_files
    rules = terrain_rules(decomp)
    where, rows = rom_rows[game]
    assert where[ROARK] == [47]
    matrix_id, background = map_header_battle_fields(rom, symbols)[47]
    assert (matrix_id, rules.backgrounds[background]) == (113, "BACKGROUND_INDOORS_1")
    m = parse_matrix(rom.narc(MATRIX_NARC)[matrix_id], matrix_id)
    assert (m.width, m.height, m.headers, m.lands) == (1, 1, None, (229,))
    behaviours = land_behaviours(rules, rom.narc(LAND_DATA_NARC)[229], 229)
    assert behaviours == {0x00, 0x30, 0x31, 0x65}
    assert not any(behaviours & s for s, _ in rules.tests)
    assert rows[ROARK] == {"terrain": BUILDING, "terrains": [BUILDING]}


@pytest.mark.parametrize("game", ["CPUE", "JAK7"])
def test_mixed_trainers_per_game(game, rom_rows):
    _, rows = rom_rows[game]
    mixed = [t for t, r in enumerate(rows) if r["terrain"] == TERRAIN_UNKNOWN and r["terrains"]]
    assert len(mixed) == MIXED[game]
    for r in rows:
        if r["terrain"] != TERRAIN_UNKNOWN:
            assert r["terrains"] == [r["terrain"]]
    # A header none of its matrix's cells names gives TERRAIN_UNKNOWN (headers 1, 206, 222,
    # ...; Kaizo adds 615-617). No trainer is placed on one: the ten Kaizo trainers whose
    # battles sit in the shared scripts_empty file are placed only where an event runs them
    # (tools/rom/readers/maps.py trainer_maps), not on its 136 headers.
    unreadable = {t for t, r in enumerate(rows) if TERRAIN_UNKNOWN in r["terrains"]}
    assert unreadable == set()


def test_vanilla_rom_inputs_equal_the_decomp(vanilla_files, decomp, symbols):
    from tools.rom.readers.maps import (
        LAND_DATA_NARC,
        MATRIX_NARC,
        map_header_battle_fields,
        rom_terrain_rules,
    )
    from tools.terrain import (
        decomp_header_battle_fields,
        decomp_land_data,
        decomp_matrices,
        parse_matrix,
    )

    assert rom_terrain_rules(vanilla_files, symbols, decomp) == terrain_rules(decomp)
    assert map_header_battle_fields(vanilla_files, symbols) == decomp_header_battle_fields(decomp)
    matrices = [parse_matrix(m, i) for i, m in enumerate(vanilla_files.narc(MATRIX_NARC))]
    assert matrices == decomp_matrices(decomp)
    assert vanilla_files.narc(LAND_DATA_NARC) == decomp_land_data(decomp)


def test_kaizo_map_header_getters(kaizo_files, vanilla_files, symbols):
    # check_kaizo_map_headers matches the patched MapHeader_GetMapMatrixID (u16 at 2, the
    # Pearl test kept) and MapHeader_GetBattleBG (bits 7-11 of the u16 at 22).
    from tools.rom.readers import ReaderError
    from tools.rom.readers.maps import check_kaizo_map_headers

    from .test_kaizo_rom import _Arm9Patched

    check_kaizo_map_headers(kaizo_files, symbols)
    with pytest.raises(ReaderError):
        check_kaizo_map_headers(vanilla_files, symbols)
    matrix = symbols.find("MapHeader_GetMapMatrixID").address & ~1
    bad = _Arm9Patched(kaizo_files, matrix + 32, (4).to_bytes(4, "little"))  # u16 at 4
    with pytest.raises(ReaderError, match="MapHeader_GetMapMatrixID"):
        check_kaizo_map_headers(bad, symbols)
    bg = symbols.find("MapHeader_GetBattleBG").address & ~1
    bad = _Arm9Patched(kaizo_files, bg + 16, bytes([0x40, 0x06]))  # lsls #25: bits 0-6
    with pytest.raises(ReaderError, match="MapHeader_GetBattleBG"):
        check_kaizo_map_headers(bad, symbols)


def test_terrain_code_guard(kaizo_rom_dir):
    from tools.rom.readers import ReaderError
    from tools.rom.readers.maps import check_terrain_code

    changed = json.loads((kaizo_rom_dir / "changed_code.json").read_text("utf-8"))
    check_terrain_code(changed)  # Kaizo leaves the terrain path alone
    with pytest.raises(ReaderError, match="arm9 CalcTerrain"):
        check_terrain_code([*changed, ["arm9", "CalcTerrain"]])
    with pytest.raises(ReaderError, match="TileBehavior_IsSand"):
        check_terrain_code([["arm9", "TileBehavior_IsSand"]])


def test_terrain_code_symbols_exist(symbols):
    from tools.terrain import BATTLE_TERRAIN_CODE, TERRAIN_CODE

    for name in (*TERRAIN_CODE, *BATTLE_TERRAIN_CODE):
        symbols.find(name)  # KeyError for a misspelt name, which would never match a change


def test_battle_terrain_code_guard():
    from tools.rom.readers import ReaderError
    from tools.rom.readers.maps import check_terrain_code

    with pytest.raises(ReaderError, match="overlay16 sTerrainMove"):
        check_terrain_code([["overlay16", "sTerrainMove"]])


def test_battle_terrain_tables_in_both_roms(vanilla_files, kaizo_files, symbols, kaizo_rom_dir):
    """The engine's terrain tables (K3b2 Task 10) are both ROMs': vanilla's
    ``sTerrainMove`` (u16 each) and ``sTerrainCamouflageType`` (u8 each) read at their
    linker-map addresses; Kaizo's overlay 16 is longer (its symbols moved), so its diff
    (``changed_code.json``) must list neither and each table's vanilla bytes must sit within
    0x40 bytes after its vanilla address."""
    import struct

    from nuzline.engine.ids import TERRAIN_MOVES, TERRAIN_TYPES
    from tools.rom.readers import symbol_bytes

    tables = {
        "sTerrainMove": struct.pack(f"<{len(TERRAIN_MOVES)}H", *TERRAIN_MOVES),
        "sTerrainCamouflageType": bytes(int(t) for t in TERRAIN_TYPES),
    }
    changed = json.loads((kaizo_rom_dir / "changed_code.json").read_text("utf-8"))
    assert not {name for _, name in changed} & set(tables)
    for name, want in tables.items():
        assert symbol_bytes(vanilla_files, symbols, name) == want, name
        sym = symbols.find(name)
        n = int(sym.module.removeprefix("overlay"))
        start = sym.address - kaizo_files.overlay_info(n)[0]
        assert want in kaizo_files.overlay(n)[start : start + len(want) + 0x40], name


def test_kaizo_trainers_json_carry_terrain(kaizo_rom_trainers):
    roark = kaizo_rom_trainers[ROARK]
    assert (roark["terrain"], roark["terrains"]) == (BUILDING, [BUILDING])
    mixed = [t for t in kaizo_rom_trainers if t["terrain"] == TERRAIN_UNKNOWN and t["terrains"]]
    assert len(mixed) == MIXED["JAK7"]
