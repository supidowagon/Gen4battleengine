"""``[sheet_unmapped]`` in overrides.toml (tools/kaizo/overrides.py): every split-sheet header
that maps to no trainer id is settled as an ally, a duplicate or a user call, so no battle
drops out of its split (``check_sheet_unmapped``). Pure; temporary files only."""

import json
import re

import pytest

from tools.kaizo.overrides import OverridesError, check_sheet_unmapped, load_overrides
from tools.kaizo.trainer_map import Mapping

GOOD = """
[sheet_unmapped."Gardenia Split:7"]
kind = "ally"
reason = "tag partner"
source = "Gardenia Split:7"

[sheet_unmapped."Byron Split:202"]
kind = "duplicate"
of = 96
reason = "same trainer as Galactic Split:7"
source = "Byron Split:202"

[sheet_unmapped."Gardenia Split:111"]
kind = "user_call"
reason = "no ROM party: is this a fight?"
source = "Gardenia Split:111"
"""


def load(tmp_path, text):
    p = tmp_path / "overrides.toml"
    p.write_text(text, encoding="utf-8")
    return load_overrides(p)


def test_load_kinds(tmp_path):
    o = load(tmp_path, GOOD)
    assert {c: e.kind for c, e in o.sheet_unmapped.items()} == {
        "Gardenia Split:7": "ally",
        "Byron Split:202": "duplicate",
        "Gardenia Split:111": "user_call",
    }
    assert o.sheet_unmapped["Byron Split:202"].of == 96
    assert o.sheet_unmapped["Gardenia Split:7"].of is None


@pytest.mark.parametrize(
    "body, match",
    [
        ('kind = "maybe"\nreason = "x"', "kind must be one of"),
        ('kind = "ally"\nreason = ""', "reason must be"),
        ('kind = "duplicate"\nreason = "x"', "'of' is required"),
        ('kind = "ally"\nof = 3\nreason = "x"', "'of' is required"),
        ('kind = "ally"\nreason = "x"\nextra = 1', "extra"),
    ],
)
def test_load_refuses(tmp_path, body, match):
    text = f'[sheet_unmapped."A:1"]\n{body}\nsource = "A:1"\n'
    with pytest.raises(OverridesError, match=match):
        load(tmp_path, text)


def test_check_every_unmapped_header_is_settled(tmp_path):
    o = load(tmp_path, GOOD)
    sheet = [
        Mapping("Gardenia Split:7", None, "unmapped", ()),
        Mapping("Byron Split:202", None, "unmapped", ()),
        Mapping("Gardenia Split:111", None, "unmapped", ()),
        Mapping("Galactic Split:7", 96, "exact", (96,)),
    ]
    assert check_sheet_unmapped(o, sheet) == []
    unsettled = [*sheet, Mapping("Roark Split:59", None, "unmapped", ())]
    assert check_sheet_unmapped(o, unsettled) == [
        (
            "sheet header Roark Split:59 maps to no trainer id: map it ([trainer_ids]) or settle "
            "it ([sheet_unmapped])"
        )
    ]


def test_check_stale_entry_and_dangling_duplicate(tmp_path):
    o = load(tmp_path, GOOD)
    sheet = [
        Mapping("Gardenia Split:7", 613, "override", (613,)),  # now maps: entry is stale
        Mapping("Byron Split:202", None, "unmapped", ()),  # duplicate of 96: nothing maps 96
        Mapping("Gardenia Split:111", None, "unmapped", ()),
    ]
    assert check_sheet_unmapped(o, sheet) == [
        "sheet_unmapped.Gardenia Split:7: the header maps to a trainer id",
        "sheet_unmapped.Byron Split:202: duplicate of 96, which no sheet header maps to",
    ]


ALLY_REASON = re.compile(
    r"with (?P<rival>[\w.' -]+) \(player starter (?P<starter>\w+)\): ROM (?P<ids>\d+(?:/\d+)*)"
)


def test_ally_reasons_name_the_partner_rows_starter(kaizo_intermediate):
    """Each committed ally entry that names a player starter names it right: the ROM partner
    rows it cites carry the rival's Pokemon in their label and the player starter in
    ``player_starter``, and the sheet header lists that Pokemon."""
    rows = {
        t["id"]: t for t in json.loads((kaizo_intermediate / "trainers.json").read_text("utf-8"))
    }
    species = json.loads((kaizo_intermediate / "species.json").read_text("utf-8"))
    docs_path = kaizo_intermediate / "docs" / "docs.json"
    if not docs_path.exists():
        pytest.skip("Kaizo docs not extracted")
    docs = json.loads(docs_path.read_text("utf-8"))
    headers = {
        h["cite"]: h
        for sv in docs["splits"].values()
        for sec in sv["sections"]
        for h in sec["trainers"]
    }
    checked = 0
    for cite, e in load_overrides().sheet_unmapped.items():
        if e.kind != "ally" or "player starter" not in e.reason:
            continue
        m = ALLY_REASON.search(e.reason)
        assert m, (cite, e.reason)
        for tid in map(int, m["ids"].split("/")):
            row = rows[tid]
            assert row["label"].endswith(f"[{m['rival']}]"), (cite, tid, row["label"])
            assert species[row["player_starter"]]["name"] == m["starter"], (cite, tid)
        party = [p["species"].split(" ")[0] for p in headers[cite]["party"]]
        assert m["rival"] in party, (cite, party)
        checked += 1
    assert checked == 6
