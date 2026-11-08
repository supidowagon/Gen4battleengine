"""Per-game names: aliases, renamed slots and the manifest's game id."""

import pytest

from nuzline.data.tables import Names


def test_alias_resolves(vanilla_manifest):
    m = {**vanilla_manifest, "aliases": {"move": {"Water Ball": 4}}}
    assert Names(m).move("water ball") == 4 and Names(m).move("Comet Punch") == 4


def test_alias_colliding_with_a_name_refused(vanilla_manifest):
    m = {**vanilla_manifest, "aliases": {"move": {"Tackle": 4}}}
    with pytest.raises(ValueError, match="'Tackle'.*33.*4"):
        Names(m)


def _renamed(vanilla_manifest, **extra):
    names = list(vanilla_manifest["move_names"])
    names[205] = "Accelerock"
    return {
        **vanilla_manifest,
        "move_names": names,
        "renamed": {"species": {}, "move": {"rollout": "Accelerock"}, "item": {}, "ability": {}},
        **extra,
    }


def test_renamed_slot_raises(vanilla_manifest):  # Kaizo-shaped data
    m = _renamed(vanilla_manifest)
    with pytest.raises(KeyError, match="in this game that slot is 'Accelerock'"):
        Names(m).move("Rollout")
    assert Names(m).move("Accelerock") == 205


def test_alias_colliding_with_a_renamed_key_refused(vanilla_manifest):
    m = _renamed(vanilla_manifest, aliases={"move": {"Rollout": 4}})
    with pytest.raises(ValueError, match="'Rollout'.*205.*4"):
        Names(m)


def test_vanilla_defaults(names):
    assert names.move("Rollout") == 205 and names.game == "platinum_us"
    assert names.unclassified_rom_changes == 0


def test_game_and_unclassified_read_from_manifest(vanilla_manifest):
    m = {
        **vanilla_manifest,
        "game": "platinum_kaizo_nightmare_final",
        "unclassified_rom_changes": 3,
    }
    n = Names(m)
    assert n.game == "platinum_kaizo_nightmare_final" and n.unclassified_rom_changes == 3


def test_names_reads_conditions(vanilla_manifest):
    n = Names(
        {
            **vanilla_manifest,
            "rom_change_conditions": [
                {
                    "label": "x",
                    "formats": ["double"],
                    "areas": [],
                    "moves": [],
                    "abilities": [],
                    "items": [],
                    "trainers": [],
                    "trainer_items": False,
                }
            ],
        }
    )
    assert n.rom_change_conditions[0].formats == ("double",)
    assert Names(vanilla_manifest).rom_change_conditions == ()
