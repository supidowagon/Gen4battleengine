"""The built Kaizo tables (K2b Task 15). Every test skips until
``python -m tools.build_tables --game kaizo`` has written ``data/gen4_platinum_kaizo``."""

import json

import pytest

from nuzline.data.games import GAMES, KAIZO
from nuzline.engine.ids import EffectId, ItemId, TrainerItemId
from tools.build_tables import BuildError, resolve_conditions
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


# Triple Axel's slot in the Kaizo move table (kaizo-data-design section 5.10); its ROM effect is
# be_seq member 88. Dry Skin is line 88 (id 87) of the decomp's generated/abilities.txt.
MOVE_TRIPLE_AXEL_SLOT = 149
ABILITY_DRY_SKIN = 87


def _write(tmp_path, text):
    path = tmp_path / "rom_diff.toml"
    path.write_bytes(text.encode("utf-8"))
    return path


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


MEMBER88 = '[[member]]\nnarc="battle/skill/be_seq.narc"\nindex=88\nstatus="not_reached"\nnote="n"\n'


def test_conditions_resolve_names(tmp_path, kaizo_intermediate):
    when = (
        'when={move_effect=[88], ability=["Dry Skin"], held_item=["Leftovers"],'
        ' format=["double"], area_condition=["gravity"], trainer_items=true}\n'
    )
    (c,) = resolve_conditions(kaizo_intermediate, _write(tmp_path, MEMBER88 + when))
    assert c["label"] == "battle/skill/be_seq.narc#88"
    assert MOVE_TRIPLE_AXEL_SLOT in c["moves"]
    assert c["abilities"] == [ABILITY_DRY_SKIN]
    assert len(c["items"]) == 1 and c["formats"] == ["double"] and c["areas"] == ["gravity"]
    assert c["trainer_items"] is True and c["trainers"] == []


def test_map_weather_resolves_to_trainers(tmp_path, kaizo_intermediate):
    # Kaizo map weather 18 runs the Trick Room subscript (tools/rom/readers/maps.py).
    (c,) = resolve_conditions(
        kaizo_intermediate, _write(tmp_path, MEMBER88 + "when={map_weather=[18]}\n")
    )
    rows = json.loads((kaizo_intermediate / "rom" / "rom_field_effects.json").read_text("utf-8"))
    assert c["trainers"] and {r["id"] for r in rows} <= set(c["trainers"])


def test_map_header_resolves_to_its_trainers(tmp_path, kaizo_intermediate):
    placed = json.loads((kaizo_intermediate / "rom" / "rom_trainer_maps.json").read_text("utf-8"))
    want = sorted(int(t) for t, headers in placed.items() if 293 in headers)
    (c,) = resolve_conditions(
        kaizo_intermediate, _write(tmp_path, MEMBER88 + "when={map_header=[293]}\n")
    )
    assert want and c["trainers"] == want and c["moves"] == []
    assert 246 not in c["trainers"]  # Roark is on header 47


def test_unresolved_name_fails_build(tmp_path, kaizo_intermediate):
    with pytest.raises(BuildError, match="Dry skinn"):
        resolve_conditions(
            kaizo_intermediate, _write(tmp_path, MEMBER88 + 'when={ability=["Dry skinn"]}\n')
        )


def test_manifest_conditions_are_the_not_reached_entries(kaizo_manifest):
    from tools.build_tables import ROM_DIFF
    from tools.rom.diff import load_conditions

    labels = [c["label"] for c in kaizo_manifest["rom_change_conditions"]]
    assert labels == sorted(load_conditions(ROM_DIFF))
    assert "arm9 MapHeader_GetBattleBG" in labels  # K3a Task 5: terrain moves
