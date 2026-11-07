"""Three-way cross-check engine, resolutions policy and report writer (Task 11)."""

import pytest

from tools.kaizo.crosscheck import (
    CrossCheckError,
    Resolution,
    Row,
    compare,
    load_resolutions,
    write_report,
)
from tools.rom.inputs import KAIZO_DOCS_SHA256, PKCALC_SHA256


def write(tmp_path, text):
    p = tmp_path / "resolutions.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_executed_conflict_needs_resolution():
    check = compare([Row("moves", "14 Swords Dance", "pp", 20, 2, 30, True)], [])
    assert [r.key for r in check.unresolved] == ["14 Swords Dance"]
    with pytest.raises(CrossCheckError, match="Swords Dance"):
        check.require_clean()


def test_grouped_resolution_defaults_to_rom():
    res = Resolution("moves", ("1 A", "2 B"), "power", 5, "rom", "PKCalc rounding", "rom_diff.md")
    rows = [
        Row("moves", "1 A", "power", 4, 5, None, True),
        Row("moves", "2 B", "power", 4, 5, None, True),
    ]
    check = compare(rows, [res])
    assert {o.used for o in check.outcomes} == {4} and not check.unresolved
    assert {o.kind for o in check.outcomes} == {"resolved"}
    check.require_clean()


def test_resolution_to_pkcalc_uses_pkcalc():
    res = Resolution("moves", ("1 A",), "power", 5, "pkcalc", "r", "trace:t1")
    check = compare([Row("moves", "1 A", "power", 4, 5, None, True)], [res])
    assert check.outcomes[0].used == 5 and check.outcomes[0].resolution is res


def test_stale_when_pkcalc_value_changed():
    res = Resolution("moves", ("1 A",), "power", 5, "rom", "r", "e")
    check = compare([Row("moves", "1 A", "power", 4, 6, None, True)], [res])
    assert check.stale == [res] and [r.key for r in check.unresolved] == ["1 A"]


def test_stale_when_no_longer_disagreeing():
    res = Resolution("moves", ("1 A",), "power", 5, "rom", "r", "e")
    check = compare([Row("moves", "1 A", "power", 5, 5, None, True)], [res])
    assert check.stale == [res] and check.outcomes[0].kind == "agree"
    with pytest.raises(CrossCheckError, match="stale.*1 A"):
        check.require_clean()


def test_resolution_does_not_cover_unexecuted_row():
    res = Resolution("moves", ("1 A",), "name", "X", "rom", "r", "e")
    check = compare([Row("moves", "1 A", "name", "Y", "X", None, False)], [res])
    assert check.stale == [res]


def test_pkcalc_use_needs_trace(tmp_path):
    p = write(
        tmp_path,
        '[[resolution]]\ntable = "moves"\nkeys = ["1 A"]\nfield = "power"\npkcalc = 5\n'
        'use = "pkcalc"\nreason = "r"\nevidence = "Moves:15"\n',
    )
    with pytest.raises(CrossCheckError, match="trace:"):
        load_resolutions(p)


def test_load_resolutions_roundtrip(tmp_path):
    p = write(
        tmp_path,
        '[[resolution]]\ntable = "moves"\nkeys = ["1 A", "2 B"]\nfield = "power"\npkcalc = 5\n'
        'use = "pkcalc"\nreason = "r"\nevidence = "trace:abc"\n',
    )
    assert load_resolutions(p) == [
        Resolution("moves", ("1 A", "2 B"), "power", 5, "pkcalc", "r", "trace:abc")
    ]
    assert load_resolutions(tmp_path / "absent.toml") == []


_OK = {
    "table": '"m"',
    "keys": '["a"]',
    "field": '"f"',
    "pkcalc": "1",
    "use": '"rom"',
    "reason": '"r"',
    "evidence": '"e"',
}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"keys": "[]"}, "keys"),
        ({"use": '"sheet"'}, "use"),
        ({"pkcalc": None}, "pkcalc"),
        ({"reason": '""'}, "reason"),
        ({"x": "1"}, "unknown"),
    ],
)
def test_resolution_schema_errors(tmp_path, change, match):
    fields = {**_OK, **change}
    body = "\n".join(f"{k} = {v}" for k, v in fields.items() if v is not None)
    with pytest.raises(CrossCheckError, match=match):
        load_resolutions(write(tmp_path, f"[[resolution]]\n{body}\n"))


def test_duplicate_resolution_rejected(tmp_path):
    body = "\n".join(f"{k} = {v}" for k, v in _OK.items())
    with pytest.raises(CrossCheckError, match="twice"):
        load_resolutions(write(tmp_path, f"[[resolution]]\n{body}\n[[resolution]]\n{body}\n"))


def test_sheet_only_needs_no_entry():
    out = compare([Row("moves", "1 A", "pp", 10, 10, 15, True)], []).outcomes[0]
    assert out.kind == "sheet_only"


def test_unexecuted_uses_pkcalc():
    out = compare([Row("moves", "1 A", "name", "Pound", "POUND", None, False)], []).outcomes[0]
    assert (out.kind, out.used) == ("unexecuted", "POUND")


def test_pkcalc_absent_sheet_absent_is_agree():
    out = compare([Row("trainers", "1 A", "tag", "none", None, None, True)], []).outcomes[0]
    assert (out.kind, out.used) == ("agree", "none")


def test_pkcalc_absent_rom_equals_sheet_is_agree():
    assert compare([Row("items", "1 A", "e", 3, None, 3, True)], []).outcomes[0].kind == "agree"


def test_pkcalc_absent_rom_differs_from_sheet():
    out = compare([Row("items", "1 A", "e", 3, None, 4, True)], []).outcomes[0]
    assert (out.kind, out.used) == ("rom_vs_sheet", 3)


def test_rom_absent_with_pkcalc_is_unresolved_when_executed():
    check = compare([Row("trainers", "1 A", "mapped", None, 7, None, True)], [])
    assert check.outcomes[0].kind == "unresolved" and len(check.unresolved) == 1


def test_require_clean_lists_everything():
    res = Resolution("moves", ("9 Z",), "pp", 1, "rom", "r", "e")
    check = compare([Row("moves", "1 A", "pp", 1, 2, None, True)], [res])
    with pytest.raises(CrossCheckError) as e:
        check.require_clean()
    assert "1 A" in str(e.value) and "stale" in str(e.value) and "9 Z" in str(e.value)


def test_report_is_deterministic(tmp_path):
    check = compare([Row("moves", "14 Swords Dance", "pp", 20, 20, 30, True)], [])
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    for p in (a, b):
        write_report(
            check,
            {"mapping": ["x"]},
            {"pkcalc": PKCALC_SHA256, "kaizo_docs": KAIZO_DOCS_SHA256},
            p,
        )
    text = a.read_text("utf-8")
    assert a.read_bytes() == b.read_bytes() and PKCALC_SHA256 in text and KAIZO_DOCS_SHA256 in text
    assert "| moves | 14 Swords Dance | pp | 20 | 20 | 30 | 20 |" in text


def test_report_order_independent_ascii_and_escaped(tmp_path):
    rows = [
        Row("moves", "2 B", "name", "Café|x", "Cafe", None, True),
        Row("items", "1 A", "hold", [1, "a"], [1, "a"], 5, True),
        Row("moves", "1 A", "pp", 1, 1, 2, True),
    ]
    paths = []
    for i, order in enumerate((rows, rows[::-1])):
        p = tmp_path / f"{i}.md"
        write_report(compare(order, []), {"mapping": ["café"]}, {"pkcalc": "h"}, p)
        paths.append(p)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    text = paths[0].read_text("ascii")
    assert "Caf\\xe9\\|x" in text and "[1, a]" in text and "caf\\xe9" in text
    assert "\r" not in text
    assert "## unresolved (1)" in text
