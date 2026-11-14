"""PKCalc tag partners fill the multis whose ROM partner is not evaluated (tools/kaizo/
resolve.py ``fill_pk_partners``, ``pk_partner_variants``, ``rival2_genders``). Pure."""

import pytest

from tools.kaizo.resolve import (
    ResolveError,
    fill_pk_partners,
    pk_partner_variants,
    rival2_genders,
)

LABELS = {
    608: "Pokémon Trainer Cheryl @ Eterna Forest",
    613: "Pokémon Trainer Lucas @ Jubilife City [Grotle]",
    616: "Pokémon Trainer Dawn @ Jubilife City [Grotle]",
    621: "Pokémon Trainer Lucas @ Veilstone City [Empoleon]",
    622: "Pokémon Trainer Lucas @ Veilstone City [Torterra]",
    624: "Pokémon Trainer Dawn @ Veilstone City [Empoleon]",
    625: "Pokémon Trainer Dawn @ Veilstone City [Torterra]",
    800: "Pokémon Trainer Lucas @ Lake Verity",
    801: "Pokémon Trainer Dawn @ Lake Verity",
    607: "Pokémon Trainer Barry @ Spear Pillar [Torterra]",
    619: "Pokémon Trainer Barry @ Spear Pillar [Infernape]",
}
STARTERS = {613: 390, 616: 390, 621: 387, 622: 390, 624: 387, 625: 390, 607: 393, 619: 387}
GENDERS = {"Lucas": 1, "Dawn": 0}


def rec(id, label, fmt="single", partner=(), starter=None):
    return {
        "id": id,
        "label": label,
        "format": fmt,
        "partner": list(partner) if partner is not None else None,
        "player_starter": starter,
        "sources": {},
    }


def test_plain_label_is_one_variant():
    got = pk_partner_variants(LABELS[608], LABELS, STARTERS, GENDERS)
    assert got == [{"trainer": 608, "starter": None, "player_gender": None}]


def test_rival2_name_only_gives_gender_variants():
    got = pk_partner_variants(
        "Pokémon Trainer {RIVAL2_NAME} @ Lake Verity", LABELS, STARTERS, GENDERS
    )
    assert got == [
        {"trainer": 801, "starter": None, "player_gender": 0},
        {"trainer": 800, "starter": None, "player_gender": 1},
    ]


def test_rival2_name_and_starter_give_both():
    got = pk_partner_variants(
        "Pokémon Trainer {RIVAL2_NAME} @ Veilstone City [{RIVAL2_STARTER_3}]",
        LABELS,
        STARTERS,
        GENDERS,
    )
    assert got == [
        {"trainer": 624, "starter": 387, "player_gender": 0},
        {"trainer": 625, "starter": 390, "player_gender": 0},
        {"trainer": 621, "starter": 387, "player_gender": 1},
        {"trainer": 622, "starter": 390, "player_gender": 1},
    ]


def test_rival_starter_only():
    got = pk_partner_variants(
        "Pokémon Trainer Barry @ Spear Pillar [{RIVAL_STARTER_3}]", LABELS, STARTERS, GENDERS
    )
    assert [(p["trainer"], p["starter"], p["player_gender"]) for p in got] == [
        (619, 387, None),
        (607, 393, None),
    ]


def test_no_match_and_missing_starter_raise():
    with pytest.raises(ResolveError, match="matches no trainer"):
        pk_partner_variants("Pokémon Trainer Nobody", LABELS, STARTERS, GENDERS)
    with pytest.raises(ResolveError, match="has no starter"):
        pk_partner_variants(
            "Pokémon Trainer {RIVAL2_NAME} @ Veilstone City [{RIVAL2_STARTER_3}]",
            LABELS,
            {},
            GENDERS,
        )


def test_rival2_genders_from_rom_variants_and_conflict():
    records = [
        rec(613, LABELS[613]),
        rec(616, LABELS[616]),
        rec(
            414,
            "Galactic Venus #1",
            "multi",
            [
                {"trainer": 616, "starter": 390, "player_gender": 0},
                {"trainer": 613, "starter": 390, "player_gender": 1},
            ],
        ),
    ]
    assert rival2_genders(records) == {"Dawn": 0, "Lucas": 1}
    records[2]["partner"].append({"trainer": 616, "starter": 393, "player_gender": 1})
    with pytest.raises(ResolveError, match="both player genders"):
        rival2_genders(records)


def test_fill_only_unevaluated_multis():
    pkcalc = {
        "formats": {
            "tag": [
                ["Galactic Terra #2", "Galactic Luna #2", LABELS[608]],
                ["Galactic Ophelia", "Galactic Bianca #1", LABELS[608]],
            ]
        }
    }
    records = [
        rec(608, LABELS[608]),
        rec(435, "Galactic Terra #2", "multi", None),
        rec(436, "Galactic Luna #2", "multi", None),
        rec(
            204,
            "Galactic Ophelia",
            "multi",
            [{"trainer": 1, "starter": None, "player_gender": None}],
        ),
        rec(999, "Galactic Bianca #1", "single", None),
    ]
    assert fill_pk_partners(records, pkcalc) == [435, 436]
    assert records[1]["partner"] == [{"trainer": 608, "starter": None, "player_gender": None}]
    assert records[1]["sources"]["partner"].startswith("pkcalc: flags.js battleType.tag")
    assert records[3]["partner"][0]["trainer"] == 1  # a ROM-evaluated partner stays
    assert records[4]["partner"] is None  # not a multi: untouched
