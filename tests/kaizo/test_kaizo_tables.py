"""The built Kaizo tables (K2b Task 15). Every test skips until
``python -m tools.build_tables --game kaizo`` has written ``data/gen4_platinum_kaizo``."""

import json

import pytest

from nuzline.data.games import GAMES, KAIZO
from nuzline.engine.ids import EffectId, ItemId, TrainerItemId
from tools.rom.inputs import KAIZO_PATCH_SHA256, VANILLA_SHA1

PROVENANCE = (
    "game",
    "vanilla_rom_sha1",
    "patch_sha256",
    "patched_rom_sha1",
    "pkcalc_sha256",
    "kaizo_docs_sha256",
    "decomp_commit",
    "overrides_sha256",
    "resolutions_sha256",
    "rom_diff_sha256",
)


def moves_with_effects(effects):
    moves = json.loads((GAMES[KAIZO].intermediate_dir / "moves.json").read_text("utf-8"))
    return [i for i, m in enumerate(moves) if m["effect"] in set(effects)]


def test_kaizo_tables(kaizo_tables, kaizo_names):
    assert kaizo_names.game == "platinum_kaizo_nightmare_final"
    assert kaizo_names.move_names[205] == "Accelerock" and kaizo_names.move("Water Ball") == 4
    assert int(kaizo_tables.move_priority[kaizo_names.move("Trick Room")]) == 7
    assert int(kaizo_tables.item_effect[kaizo_names.item("Berserk Gene")]) == ItemId.UNSUPPORTED


def test_renamed_vanilla_name_points_at_the_kaizo_name(kaizo_names):
    with pytest.raises(KeyError, match="in this game that slot is 'Accelerock'"):
        kaizo_names.move("Rollout")
    assert kaizo_names.species("Nidoran♀") == kaizo_names.species("Nidoran-F") == 29


def test_kaizo_manifest_provenance(kaizo_manifest):
    assert kaizo_manifest["vanilla_rom_sha1"] == VANILLA_SHA1
    assert kaizo_manifest["patch_sha256"] == KAIZO_PATCH_SHA256
    for key in (*PROVENANCE, "aliases", "renamed", "changed_effects", "no_secondary_moves"):
        assert key in kaizo_manifest, key
    assert kaizo_manifest["unclassified_rom_changes"] > 0  # every Kaizo fight is refused


def test_changed_effect_unsupported(kaizo_tables, kaizo_manifest):
    moves = moves_with_effects(kaizo_manifest["changed_effects"])
    assert moves  # Triple Axel (149) runs the patched multi-hit member 88
    for m in moves:
        assert int(kaizo_tables.move_effect[m]) == EffectId.UNSUPPORTED


def test_repurposed_items_unsupported(kaizo_tables, kaizo_names):
    # Spec 5.5: Lava Cookie and Old Gateau became Lum clones; Shoal Shell is RageCandyBar.
    for name in ("Lava Cookie", "Old Gateau", "RageCandyBar", "Big Nugget"):
        i = kaizo_names.item(name)
        assert int(kaizo_tables.item_effect[i]) == ItemId.UNSUPPORTED, name
        assert int(kaizo_tables.trainer_item_effect[i]) == TrainerItemId.UNSUPPORTED, name


def test_kaizo_roark_party_from_the_rom_decode(kaizo_roark, kaizo_roark_party, kaizo_names):
    assert kaizo_roark.id == 246 and kaizo_roark.pid_routine is None
    assert all(p.friendship in (0, 255) for p in kaizo_roark_party)
    assert len(kaizo_roark_party) == 6
