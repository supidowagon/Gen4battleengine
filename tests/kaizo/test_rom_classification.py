"""Every classified Kaizo ROM change carries its decoding (K3a Tasks 5-8, spec 5).

Data-only: reads ``data/kaizo/rom_diff.toml``, never the ROM.

A note has the form ``"decoded: <what changed> (<vanilla> -> <patched>); <decomp citation>;
<why this status>"``, and its citation matches ``CITE`` (``path/file.c:LINE`` and friends, or
``rom:...``). An ``unknown`` entry with a note names the milestone that settles it.

**Groups.** ``DONE_GROUPS`` lists the groups whose entries are all decoded; each
classification task appends its own. A group is ``(label, keys)``: ``keys`` is a tuple of
``(module, symbol)`` for ``[[code]]`` entries (``symbol`` as the toml spells it, a
``0xSTART-0xEND`` range included) and ``(narc, index)`` for ``[[member]]`` entries. Every
key must name an entry of the toml, and every entry of a done group has a decoded note.

``K2B_NOTES`` holds the keys whose note K2b wrote before this format existed (the changed
battle-script members); the two whole-file tests pass over them until Task 8 rewrites those
notes and empties it. A done group may not contain one.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROM_DIFF = Path(__file__).resolve().parents[2] / "data" / "kaizo" / "rom_diff.toml"
CITE = re.compile(r"(\S+\.(c|s|h|json|toml):\d+|rom:\S+)")

_MAP_HEADER_GETTERS = (
    "IDBoundsCheck",
    "GetAreaDataArchiveID",
    "GetPreloadedMapObjectsArchiveID",
    "GetMapMatrixID",
    "GetMsgArchiveID",
    "GetScriptsArchiveID",
    "GetInitScriptsArchiveID",
    "GetDayMusicID",
    "GetNightMusicID",
    "HasWildEncounters",
    "GetWildEncountersArchiveID",
    "GetEventsArchiveID",
    "GetMapLabelTextID",
    "GetMapLabelWindowID",
    "GetWeatherType",
    "GetCameraType",
    "GetBattleBG",
    "IsEscapeRopeAllowed",
    "IsFlyAllowed",
    "IsBikeAllowed",
    "GetMapType",
)
_OVERLAY13 = (
    "InitializeCursor",
    "CleanupCursor",
    "BattlePartySprites_DisableCursor",
    "BattleBagTask_CatchTutorial",
    "BattleBagTask_GetSelectedPartySlot",
    "BattleBagSprites_DisableCursor",
    "InitializeCatchTutorialCursor",
    "CleanupCatchTutorialCursor",
    "BattleBagSprites_SetupCatchTutorialCursor",
    "SetBattleSubMenuCursorCurrentPosition",
    "SetBattleSubMenuCursorPositions",
    "IsCursorVisible",
    "BattleSubMenuCursorTick",
)
TASK5 = (
    "task 5: arm9 field and menu code, overlay5, overlay13",
    tuple(
        [("arm9", f"MapHeader_{g}") for g in _MAP_HEADER_GETTERS]
        + [
            ("arm9", s)
            for s in (
                "NitroMain",
                "Options_Init",
                "ScrCmd_GetWallpaperFromCustomMessageWords",
                "ScrCmd_Dummy088",
                "0x0204EB22-0x0204EB24",
                "EncEffects_GetEffectPair",
                "Trainer_Encounter",
                "PCBoxes_UnlockWallpaper",
                "sub_0207AD40",
                "OS_GetInitArenaLo",
                "sSpawnLocations",
                "gHiddenItems",
                "gFieldScriptCommands",
                "sEncEffectsTable",
                "sTrainerEncounterBGMs",
                "sTMHMMoves",
                "sItemArchiveIDs",
                "@29",
            )
        ]
        + [("overlay5", "sTeachableMoves"), ("overlay5", "sSpeciesLearnsetsByTutor")]
        + [("overlay13", s) for s in _OVERLAY13]
    ),
)

DONE_GROUPS = [TASK5]

K2B_NOTES = {
    ("battle/skill/be_seq.narc", 48),
    ("battle/skill/be_seq.narc", 88),
    ("battle/skill/be_seq.narc", 104),
    ("battle/skill/be_seq.narc", 141),
    ("battle/skill/be_seq.narc", 182),
    ("battle/skill/sub_seq.narc", 238),
    ("battle/skill/sub_seq.narc", 251),
    ("battle/skill/sub_seq.narc", 297),
}


def _data() -> dict:
    with ROM_DIFF.open("rb") as f:
        return tomllib.load(f)


def _entries() -> list[dict]:
    """Every ``[[code]]`` and ``[[member]]`` entry but ``K2B_NOTES``, ``note`` defaulting
    to ``""``."""
    return [e for key, e in _by_key().items() if key not in K2B_NOTES]


def _by_key() -> dict[tuple, dict]:
    data = _data()
    out = {}
    for e in data.get("code", []):
        out[(e["module"], e["symbol"])] = {"note": "", **e}
    for e in data.get("member", []):
        out[(e["narc"], e["index"])] = {"note": "", **e}
    return out


def test_every_classified_entry_is_decoded():
    for e in _entries():
        if e["status"] != "unknown" or e["note"]:
            assert e["note"].startswith("decoded: "), e
            assert CITE.search(e["note"]), e


def test_unknown_entries_name_their_milestone():
    for e in _entries():
        if e["status"] == "unknown" and e["note"]:
            assert e["note"].endswith(("-> K3b", "-> K3c")), e


@pytest.mark.parametrize("group", DONE_GROUPS, ids=[g[0] for g in DONE_GROUPS])
def test_group_has_no_undecoded_entries(group):
    _, keys = group
    entries = _by_key()
    assert len(set(keys)) == len(keys), "a key is listed twice"
    assert not set(keys) & K2B_NOTES, "a done group still has a K2b-format note"
    missing = [k for k in keys if k not in entries]
    assert not missing, f"no rom_diff.toml entry for {missing}"
    undecoded = [k for k in keys if not entries[k]["note"].startswith("decoded: ")]
    assert not undecoded, f"undecoded entries: {undecoded}"
