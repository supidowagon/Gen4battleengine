"""``build_tables`` writes the manifest's ``game`` (and only there)."""

import json
from pathlib import Path

import pytest

import tools.effect_map
from nuzline.engine.ids import ItemId, MoveFlag, TrainerItemId
from tools import build_tables

INTER = Path(__file__).resolve().parents[2] / "data" / "intermediate"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (INTER / "species.json").exists():
        pytest.skip("intermediates not built")
    out = tmp_path_factory.mktemp("built")
    build_tables.build(INTER, out)
    return out


def test_default_game_is_vanilla(built):
    assert (
        json.loads((built / "manifest.json").read_text(encoding="utf-8"))["game"] == "platinum_us"
    )


def test_game_is_not_in_the_npz(built):
    import numpy as np

    with np.load(built / "tables.npz") as z:
        assert "game" not in z.files


def test_unknown_game_refused(tmp_path):
    with pytest.raises(ValueError, match="vanilla, kaizo"):
        build_tables.build(INTER, tmp_path, game="emerald")


def test_flags_by_id_unchanged_for_vanilla(built):
    import numpy as np

    with np.load(built / "tables.npz") as z:
        assert int(z["move_flags"][4]) & MoveFlag.PUNCH  # Comet Punch; regression guard
        assert int(z["move_flags"][46]) & MoveFlag.SOUND  # Roar


def test_move_flags_follow_the_lists_by_id_not_the_name():
    lists = {"sound": [], "punch": [33]}
    tackle = {"name": "Tackle", "flags": []}
    comet = {"name": "Comet Punch", "flags": []}
    assert build_tables._move_flags(tackle, 33, lists) & MoveFlag.PUNCH
    assert not build_tables._move_flags(comet, 4, lists) & MoveFlag.PUNCH
    assert build_tables._move_flags(tackle, 33, {"sound": [33], "punch": []}) & MoveFlag.SOUND


def test_effect_map_no_longer_holds_the_lists():
    assert not hasattr(tools.effect_map, "PUNCH_MOVES")
    assert not hasattr(tools.effect_map, "SOUND_MOVES")


def test_vanilla_build_unchanged(tmp_path, tables):  # regression guard (passes before K2b)
    import numpy as np

    from tests.conftest import TABLES_DIR

    if not (INTER / "species.json").exists():
        pytest.skip("intermediates not built")
    build_tables.build(INTER, tmp_path, game="platinum_us")
    with np.load(tmp_path / "tables.npz") as new, np.load(TABLES_DIR / "tables.npz") as old:
        assert new.files == old.files
        assert all(np.array_equal(new[k], old[k]) for k in old.files)
    new_m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    old_m = json.loads((TABLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert new_m == old_m


def _item(name, hold_effect):
    return {"name": name, "hold_effect": hold_effect}


def test_kaizo_item_maps_need_the_vanilla_name_and_hold_effect():
    vanilla = [_item("Oran Berry", 3), _item("Sitrus Berry", 4), _item("Oran Berry", 3)]
    vanilla += [_item("Potion", 0), _item("Potion", 0)]
    kaizo = [_item("ORAN BERRY", 3), _item("Sitrus Berry", 9), _item("Lum Clone", 3)]
    kaizo += [_item("Potion", 0), _item("Elixir", 0)]
    hold, bag = build_tables._item_maps(kaizo, vanilla)
    assert hold[0] == int(ItemId.ORAN_BERRY)  # the same name up to spelling, same hold effect
    assert hold[1] == hold[2] == int(ItemId.UNSUPPORTED)  # re-purposed (hold effect), renamed
    assert bag[3] == int(TrainerItemId.POTION) and bag[4] == int(TrainerItemId.UNSUPPORTED)
    assert build_tables._item_maps(vanilla, None)[0][:2] == [ItemId.ORAN_BERRY, ItemId.SITRUS_BERRY]


def test_kaizo_bag_item_needs_vanilla_restore_amount():
    vanilla = [{"name": "Potion", "hold_effect": 0, "hp_restored": 20}]
    kaizo = [{"name": "Potion", "hold_effect": 0, "hp_restored": 30}]
    assert build_tables._item_maps(kaizo, vanilla)[1] == [int(TrainerItemId.UNSUPPORTED)]
    assert build_tables._item_maps(vanilla, vanilla)[1] == [int(TrainerItemId.POTION)]


# --- the Kaizo ROM-change gate counts against the diff, not the entries (final review I1) --

_BE_SEQ = "battle/skill/be_seq.narc"


def _gate(tmp_path, code_entries: str, changed_code):
    """``_kaizo_rom_changes`` over a synthetic diff: two changed code keys, one changed
    be_seq member (classified), and ``code_entries`` as the ``[[code]]`` tables."""
    rom = tmp_path / "inter" / "rom"
    rom.mkdir(parents=True)
    scripts = {_BE_SEQ: [7], "battle/skill/sub_seq.narc": [], "battle/tr_ai/tr_ai_seq.narc": []}
    (rom / "changed_scripts.json").write_text(json.dumps(scripts), encoding="utf-8")
    (rom / "changed_code.json").write_text(json.dumps(changed_code), encoding="utf-8")
    toml = tmp_path / "rom_diff.toml"
    member = f'[[member]]\nnarc = "{_BE_SEQ}"\nindex = 7\nstatus = "irrelevant"\nnote = ""\n'
    toml.write_text(code_entries + member, encoding="utf-8")
    return build_tables._kaizo_rom_changes(tmp_path / "inter", toml)


_TWO = [["arm9", "NitroMain"], ["overlay16", "0x02200000-0x02200010"]]


def _code(module, symbol, status="irrelevant"):
    return f'[[code]]\nmodule = "{module}"\nsymbol = "{symbol}"\nstatus = "{status}"\n\n'


def test_kaizo_gate_counts_classified_entries(tmp_path):
    entries = _code(*_TWO[0]) + _code(*_TWO[1], status="unknown")
    effects, unknown = _gate(tmp_path, entries, _TWO)
    assert unknown == 1 and effects == frozenset({7})  # irrelevant member: still unsupported


def test_kaizo_gate_deleting_a_code_entry_raises_the_count(tmp_path):
    _, unknown = _gate(tmp_path, _code(*_TWO[1], status="unknown"), _TWO)
    assert unknown == 2  # NitroMain lost its entry: it counts as unknown, never as classified
    assert _gate(tmp_path / "none", "", _TWO)[1] == 2


def test_kaizo_gate_stale_code_entry_raises(tmp_path):
    entries = _code(*_TWO[0]) + _code(*_TWO[1]) + _code("arm9", "GoneFunc")
    with pytest.raises(build_tables.BuildError, match="arm9 GoneFunc: stale"):
        _gate(tmp_path, entries, _TWO)


def test_kaizo_gate_needs_changed_code(tmp_path):
    rom = tmp_path / "inter" / "rom"
    rom.mkdir(parents=True)
    (rom / "changed_scripts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(build_tables.BuildError, match="re-run python -m tools.extract_rom"):
        build_tables._kaizo_rom_changes(tmp_path / "inter", tmp_path / "rom_diff.toml")


def test_kaizo_gate_on_the_real_diff(tmp_path):  # skips until extract_rom --game kaizo ran
    from nuzline.data.games import GAMES, KAIZO

    inter = GAMES[KAIZO].intermediate_dir
    if not (inter / "rom" / "changed_code.json").exists():
        pytest.skip("Kaizo ROM intermediates without changed_code.json")
    _, full = build_tables._kaizo_rom_changes(inter)
    text = build_tables.ROM_DIFF.read_text(encoding="utf-8")
    first = text.index("[[code]]")
    trimmed = tmp_path / "rom_diff.toml"
    trimmed.write_text(text[:first] + text[text.index("[[code]]", first + 1) :], "utf-8")
    assert build_tables._kaizo_rom_changes(inter, trimmed)[1] >= full  # never fewer unknowns
