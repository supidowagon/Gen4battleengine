"""Kaizo docs extractor: the header tag grammar (strings) and the real workbook (skips)."""

import pytest

from tools.extract_decomp import ExtractError
from tools.extract_kaizo_docs import ai_flag, parse_header, split_header
from tools.rom.inputs import KAIZO_DOCS_SHA256

ABILITIES = ["-", "Drizzle", "Sand Stream", "Shadow Tag", "Snow Warning"]


@pytest.mark.parametrize(
    "text, tags",
    [
        # Plan Task 7 cases.
        ("Barry #3 (Player picked Piplup)", {"player_starter": 393}),
        (
            "Barry #5 (Player picked Chimchar, on Canalave Bridge)",
            {"player_starter": 390, "note": "on Canalave Bridge"},
        ),
        ("Dawn (TAG PARTNER if Player chose Piplup)", {"tag_partner": True, "player_starter": 393}),
        (
            "Grunt (TAG BATTLE with Galactic Uranus)",
            {"tag_partner": True, "multi_with": "Galactic Uranus"},
        ),
        ("Lass (OPTIONAL HAIL)", {"optional": True, "weather": "hail", "assumed_permanent": True}),
        (
            "Ace (SUN, DOUBLE BATTLE)",
            {"weather": "sun", "assumed_permanent": True, "format": "double"},
        ),
        ("Leader Cyrus (Trick Room)", {"trick_room": True, "assumed_permanent": True}),
        ("CYRUS (SHADOW TAG)", {"note": "SHADOW TAG"}),
        # The rest of the grammar, each on a real header.
        ("Oreburgh Gym (PERMANENT SANDSTORM)", {"permanent": "sandstorm"}),
        ("Champion Cynthia (Permanent Gravity)", {"permanent": "gravity"}),
        ("Veilstone Warehouse (PERMANENT TRICK ROOM)*", {"permanent": "trick_room"}),
        ("Ninja Boy Donny (FOG)", {"fog": True}),
        ("Ruin Maniac Calvin (RAIN)", {"weather": "rain", "assumed_permanent": True}),
        ("Twins Liv & Liz (DOUBLE) (Gauntlet Start)*", {"format": "double", "gauntlet": "start"}),
        ("Bug Catcher Brandon (Gauntlet)*", {"gauntlet": "middle"}),
        ("Sailor Paul (GAUNTLET END)*", {"gauntlet": "end"}),
        ("Galactic Mercury (B2B With Galactic Terra)", {"b2b_with": "Galactic Terra"}),
        (
            "Commander Saturn (RAIN, MULTI BATTLE WITH GALACTIC TITAN)",
            {"weather": "rain", "assumed_permanent": True, "multi_with": "GALACTIC TITAN"},
        ),
        (
            "Eterna Gym (PERMANENT RAIN, GYM TRAINERS ARE A GAUNTLET)",
            {"permanent": "rain", "gym_gauntlet": True},
        ),
        (
            "Barry (TAG PARTNER, if Player chose Turtwig)",
            {"tag_partner": True, "player_starter": 387},
        ),
        (
            "Trainer Dawn/Lucas (TAG PARTNER, Player selected Chimchar)",
            {"tag_partner": True, "player_starter": 390},
        ),
        (
            "Policeman Morton (FOG, Also Appears in Room 5)",
            {"fog": True, "note": "Also Appears in Room 5"},
        ),
        (
            "Galactic Hippocamp (MULTI BATTLE With Galactic Psamanthe, Right, Path 1)*",
            {"multi_with": "Galactic Psamanthe", "note": "Right, Path 1"},
        ),
        ("Belle & Pa Ava & Matt (DOUBLE BATTLE)     .", {"format": "double"}),
        ("Hiker J.J Ahern (CreamElDudJafar)", {"note": "CreamElDudJafar"}),
        ("Leader Roark", {}),
        ("Mount Coronet - Room 1", {}),
        # Grammar extensions found on the real workbook (Step 3).
        ("GalacticF Bebhionn (OPTIONAL Top Trainer)", {"optional": True, "note": "Top Trainer"}),
        (
            "Galactic Oberon (Optional Bottom Path, MULTI BATTLE with Galactic Puck)*",
            {"optional": True, "note": "Bottom Path", "multi_with": "Galactic Puck"},
        ),
        ("Trainer Marley (PARTNER)", {"tag_partner": True}),
        ("Galactic Mercury (MUTLI BATTLE with Galactic Venus)", {"multi_with": "Galactic Venus"}),
        ("TAG PARTNER MIRA*", {"tag_partner": True}),
        (
            "Wayward Cave - OPTIONAL Gauntlet With Mira*",
            {"optional": True, "gauntlet_with": "Mira"},
        ),
        (
            "Sinister Hooded Figure - (SUPERBOSS) (DOUBLE, PERMANENT TRICK ROOM)*",
            {"note": "SUPERBOSS", "format": "double", "permanent": "trick_room"},
        ),
    ],
)
def test_header_tags(text, tags):
    assert parse_header(text, ABILITIES) == tags


@pytest.mark.parametrize(
    "text, name",
    [
        ("Worker Mason (Gauntlet Start)*", "Worker Mason"),
        ("Wayward Cave - OPTIONAL Gauntlet With Mira*", "Wayward Cave"),
        ("Mount Coronet - Room 1", "Mount Coronet - Room 1"),
        ("TAG PARTNER MIRA*", "MIRA"),
        (
            "Sinister Hooded Figure - (SUPERBOSS) (DOUBLE, PERMANENT TRICK ROOM)*",
            "Sinister Hooded Figure",
        ),
        ("   CYRUS (SHADOW TAG)*", "CYRUS"),
    ],
)
def test_header_names(text, name):
    assert split_header(text)[0] == name


def test_keyword_that_does_not_parse_raises():
    with pytest.raises(ExtractError, match="unparsed tag 'PERMANENT FROST'"):
        parse_header("Ace (PERMANENT FROST)", ABILITIES, cite="Candice Split:9")


@pytest.mark.parametrize(
    "item",
    [
        "RAIN for 3 turns",
        "PERMANENT SUN until X",
        "DOUBLE Kick",
        "TRICK ROOM 5 turns",
        "Hail Mary",
        "OPTIONAL Top Trainer With Gauntlet",
    ],
)
def test_partly_parsed_item_raises(item):
    with pytest.raises(ExtractError, match=f"x:1: unparsed tag '{item}'"):
        parse_header(f"Ace ({item})", ABILITIES, cite="x:1")


def test_keyword_after_a_note_raises():
    with pytest.raises(ExtractError, match="Candice Split:9: unparsed tag 'Left of DOUBLE'"):
        parse_header("Ace (Left of DOUBLE)", ABILITIES, cite="Candice Split:9")


def test_conflicting_tags_raise():
    with pytest.raises(ExtractError, match="conflicting"):
        parse_header("Ace (PERMANENT RAIN, PERMANENT SUN)", ABILITIES, cite="x:1")


def test_unbalanced_parenthesis_raises():
    with pytest.raises(ExtractError, match="parenthes"):
        parse_header("Ace (RAIN", ABILITIES, cite="x:1")


@pytest.mark.parametrize(
    "formula, flag",
    [("=F$2", "Basic"), ("=$H$2", "Expert"), ("=Q2", "Left Side AI"), ("=J104", "Risky")],
)
def test_ai_flag_from_key_column(formula, flag):
    assert ai_flag(formula, "x:1") == flag


@pytest.mark.parametrize("formula", ["=A2", "=SUM(F2:G2)", "F$2", "=F$2+1"])
def test_ai_flag_rejects_other_formulas(formula):
    with pytest.raises(ExtractError, match="AI flag"):
        ai_flag(formula, "x:1")


# --- The real workbook (skips without it, openpyxl or the vanilla intermediates) -----


def audited(docs):
    if docs["sha256"] != KAIZO_DOCS_SHA256:
        pytest.skip("workbook is not the audited Platinum Kaizo Docs.xlsx")


def section(docs, split, name):
    found = [s for s in docs["splits"][split]["sections"] if s["name"] == name]
    assert len(found) == 1, (split, name, len(found))
    return found[0]


def tags(docs, split, name):
    found = [
        t
        for s in docs["splits"][split]["sections"]
        for t in s["trainers"]
        if name in (t["text"], t["name"])
    ]
    assert len(found) == 1, (split, name, len(found))
    return found[0]["tags"]


@pytest.mark.slow
def test_label_map_is_one_to_one(docs):
    audited(docs)
    assert len(docs["effect_labels"]) == 257 and len(set(docs["effect_labels"].values())) == 257


def test_label_decode_recovers_effects(docs):
    assert docs["moves"]["209"]["effect"] == 262  # Wild Charge: Volt Tackle's recoil + paralysis
    assert docs["moves"]["149"]["effect"] == 104  # Triple Axel: Triple Kick


def test_new_label_decodes_to_none(docs):
    new = [m for m in docs["moves"].values() if m["label"] not in docs["effect_labels"]]
    assert new and all(m["effect"] is None for m in new)


def test_split_tags(docs):
    assert section(docs, "Roark", "Oreburgh Gym")["tags"]["permanent"] == "sandstorm"
    hq = section(docs, "Galactic", "Galactic HQ")
    assert hq["tags"]["permanent"] == "trick_room"
    assert section(docs, "Candice", "Snowpoint Gym")["tags"] == {
        "weather": "hail",
        "assumed_permanent": True,
    }


def test_galactic_hq_count(docs):
    audited(docs)
    assert len(section(docs, "Galactic", "Galactic HQ")["trainers"]) == 25


def test_gauntlet_and_rival_tags(docs):
    assert tags(docs, "Roark", "Worker Mason")["gauntlet"] == "start"
    assert tags(docs, "Roark", "Barry #2 (Player chose Chimchar)")["player_starter"] == 390
    assert tags(docs, "Maylene", "Barry #3 (Player picked Piplup)")["player_starter"] == 393


def test_raw_roark(docs):
    assert [m["level"] for m in docs["raw_trainers"]["246"]["party"]] == [15, 13, 14, 14, 13, 16]
    assert {"difficulty", "ability_number"} <= docs["raw_trainers"]["246"]["party"][0].keys()


def test_splits_in_game_order(docs):
    assert list(docs["splits"]) == [
        "Roark", "Gardenia", "Fantina", "Maylene", "Wake",
        "Byron", "Candice", "Galactic", "Volkner", "Elite Four",
    ]  # fmt: skip


def test_split_trainer_party_and_ai(docs):
    roark = tags_trainer(docs, "Roark", "Leader Roark")
    assert roark["cite"] == "Roark Split:272"
    assert [m["level"] for m in roark["party"]] == [15, 13, 14, 14, 13, 16]
    tristan = tags_trainer(docs, "Roark", "Youngster Tristan")
    assert tristan["party"] == [
        {"species": "Zigzagoon", "level": 5},
        {"species": "Sentret", "level": 5},
        {"species": "Pidgey", "level": 5},
    ]
    assert tristan["ai_flags"] == ["Basic", "Eval Att", "Expert"]
    assert tristan["left_side"] is False


def test_unanchored_ai_cell_is_not_a_flag(docs):
    # Lass Natalie's D24 is "=O11" (copied without anchoring): it shows O11, not the icon.
    natalie = tags_trainer(docs, "Roark", "Lass Natalie")
    assert {"cell": "D24", "ref": "O11", "flag": "Weather", "on_key": False} in natalie["ai_cells"]
    assert "Weather" not in natalie["ai_flags"]


def test_trainer_without_ai_icons_is_unknown(docs):
    angel = next(
        t
        for s in docs["splits"]["Gardenia"]["sections"]
        for t in s["trainers"]
        if t["cite"] == "Gardenia Split:474"
    )
    assert angel["ai_known"] is False
    assert angel["ai_flags"] is None and angel["left_side"] is None
    assert tags_trainer(docs, "Roark", "Youngster Tristan")["ai_known"] is True


def tags_trainer(docs, split, name):
    return next(
        t for s in docs["splits"][split]["sections"] for t in s["trainers"] if t["name"] == name
    )


def test_every_row_is_cited(docs):
    def walk(x):
        if isinstance(x, dict):
            yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)

    for key in ("moves", "species", "items", "raw_trainers", "level_up", "tm", "evolutions"):
        assert all("cite" in row for row in docs[key].values()), key
    for key in ("move_changes", "ability_changes", "ai_changes"):
        assert docs[key] and all("cite" in row for row in docs[key]), key
    for split in docs["splits"].values():
        for s in split["sections"]:
            assert s["cite"].startswith(split["sheet"] + ":")
            assert all(t["cite"].startswith(split["sheet"] + ":") for t in s["trainers"])


def test_items_hold_fields(docs):
    assert docs["items"]["Flame Plate"]["hold_param"] == 50
    assert docs["items"]["Shoal Salt"]["hold_effect"] == 36
    assert docs["items"]["Town Map"]["fling_power"] == 120
