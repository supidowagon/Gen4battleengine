"""Built-table check of the walking-partner rule (partnerfix sheet section 4).

A lone enemy placed on a walking-partner map is BATTLE_TYPE_TRAINER (src/encounter.c:738-739)
ORed with the walking partner's header type (src/trainer_data.c:31-50), a single; a scripted
pair or a tag on the map stays multi (``Encounter_NewVsTrainer``, encounter.c:725-758).
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VANILLA = ROOT / "data" / "intermediate" / "trainers.json"
KAIZO = ROOT / "data" / "intermediate_kaizo" / "trainers.json"

pytestmark = pytest.mark.fullset

# Kaizo lone walk enemies by walking partner (Cheryl 608, Buck 611, Riley 609, Marley 610).
KAIZO_LONE = {
    608: [202, 203, 411, 413],
    611: [570, 571, 579, 580, 584, 586, 589, 590],
    609: [679],
    610: [675],
}
VANILLA_LONE = {
    608: [201, 202, 203, 204, 205, 206, 395, 398],
    610: [373, 386, 389, 390, 393, 394, 396, 397, 399, 400, 446, 447, 499, 500],
    611: [556, 561, 562, 569, 570, 571, 572, 579, 580, 581, 582, 584, 586, 589, 590, 606],
    612: [379, 380, 383, 443, 450, 454, 455, 458, 459, 496],
    609: [372, 388, 392, 445, 452, 453, 505, 506],
}


def _rows(path):
    if not path.exists():
        pytest.skip(f"{path} not built (tools/ops/build_tables.sh --here)")
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def test_kaizo_lone_walk_enemies_are_singles_with_the_walking_partner():
    rows = _rows(KAIZO)
    assert sum(len(v) for v in KAIZO_LONE.values()) == 14
    for partner, ids in KAIZO_LONE.items():
        for i in ids:
            r = rows[i]
            assert (r["format"], r["opponent_2"], r["partner"]) == ("single", None, []), i
            assert r["walking_partner"] == [partner], i


def test_every_kaizo_multi_on_a_walk_map_has_a_second_opponent():
    for i, r in _rows(KAIZO).items():
        if r["walking_partner"] and r["format"] == "multi":
            assert r["opponent_2"] is not None, i


def test_vanilla_lone_walk_enemies_are_singles_and_the_iron_island_pair_is_multi():
    rows = _rows(VANILLA)
    for partner, ids in VANILLA_LONE.items():
        for i in ids:
            r = rows[i]
            assert (r["format"], r["opponent_2"], r["partner"]) == ("single", None, []), i
            assert r["walking_partner"] == [partner], i
    for a, b in ((835, 836), (836, 835)):
        r = rows[a]
        assert (r["format"], r["opponent_2"]) == ("multi", b)
        assert r["walking_partner"] == [609]
    on_walk = {i for i, r in rows.items() if r["walking_partner"]}
    assert on_walk == {i for ids in VANILLA_LONE.values() for i in ids} | {835, 836}
