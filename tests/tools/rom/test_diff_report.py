"""Report, classification and CLI of tools.rom.diff, on synthetic ROMs."""

import dataclasses

import pytest

from tests.tools.rom.diff_helpers import CODE, SYMS, fake, poke
from tools.rom.diff import (
    CodeChange,
    FileChange,
    HeaderChange,
    RomDiff,
    classification_skeleton,
    diff_roms,
    load_classification,
    main,
    stale_entries,
    write_report,
)
from tools.rom.symbols import DECOMP_COMMIT

D = diff_roms(fake(), fake(ov3=poke(CODE, 0x44, b"\xff\xff")), SYMS)
D_MOVED = diff_roms(fake(), fake(ov3=poke(CODE, 0x64, b"\xff\xff")), SYMS)
PROV = {"vanilla_sha1": "v", "patch_sha256": "p", "patched_sha1": "k", "decomp_commit": "c"}


def foo_row(text):
    return next(line for line in text.splitlines() if line.startswith("| Foo |"))


def test_report_deterministic_and_ascii(tmp_path):
    for n in ("a.md", "b.md"):
        write_report(D, {}, tmp_path / n, PROV)
    a = (tmp_path / "a.md").read_bytes()
    assert a == (tmp_path / "b.md").read_bytes() and a.isascii()


def test_default_status_is_unknown(tmp_path):
    write_report(D, {}, tmp_path / "r.md", PROV)
    assert "| unknown |" in foo_row((tmp_path / "r.md").read_text())


def test_fixed_signature_uses_default_provenance(tmp_path):  # K2b calls write_report(d, c, p)
    write_report(D, {}, tmp_path / "r.md")
    assert DECOMP_COMMIT in (tmp_path / "r.md").read_text()


def test_classification_applied(tmp_path):
    write_report(D, {("overlay3", "Foo"): ("irrelevant", "menu")}, tmp_path / "r.md", PROV)
    assert "| irrelevant | menu |" in foo_row((tmp_path / "r.md").read_text())


def test_stale_entry_raises(tmp_path):  # a vanished symbol
    with pytest.raises(ValueError, match="overlay3.*Gone"):
        write_report(D, {("overlay3", "Gone"): ("unknown", "")}, tmp_path / "r.md", PROV)
    assert not (tmp_path / "r.md").exists()


def test_moved_change_reports_old_symbol_stale(tmp_path):  # the change moved Foo -> Bar
    cls = {("overlay3", "Foo"): ("modelled", "x")}
    assert stale_entries(D_MOVED, cls) == [("overlay3", "Foo")]
    with pytest.raises(ValueError, match="Foo"):
        write_report(D_MOVED, cls, tmp_path / "r.md", PROV)


def test_skeleton_round_trips(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(classification_skeleton(D))
    cls = load_classification(p)
    assert set(cls) == {(c.module, c.symbol) for c in D.code}
    assert {s for s, _ in cls.values()} == {"unknown"}


def test_init_refuses_overwrite(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("# mine\n")
    assert main(["--classification", str(p), "--init-classification"]) == 1
    assert p.read_text() == "# mine\n"


def test_diff_types_carry_no_bytes():
    for cls in (RomDiff, FileChange, CodeChange, HeaderChange):
        assert all("bytes" not in str(f.type) for f in dataclasses.fields(cls)), cls


# --- beyond the brief ---------------------------------------------------------------------


def test_unknown_status_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[code]]\nmodule = "overlay3"\nsymbol = "Foo"\nstatus = "maybe"\nnote = ""\n')
    with pytest.raises(ValueError, match="maybe"):
        load_classification(p)


def test_range_key_for_code_outside_symbols(tmp_path):
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x04, b"\xff")), SYMS)
    assert [c.symbol for c in d.code] == [None]
    p = tmp_path / "c.toml"
    p.write_text(classification_skeleton(d))
    assert set(load_classification(p)) == {("overlay3", "0x02200004-0x02200005")}


def test_missing_decomp_is_an_error(tmp_path, capsys):
    argv = [
        "--decomp",
        str(tmp_path / "nope"),
        "--classification",
        str(tmp_path / "c.toml"),
        "--out",
        str(tmp_path / "r.md"),
    ]
    assert main(argv) == 1
    assert "decomp not found" in capsys.readouterr().err
    assert not (tmp_path / "r.md").exists()


def test_report_explains_relocation_only_rows(tmp_path):
    write_report(D, {}, tmp_path / "r.md", PROV)
    assert "not stale-checked" in (tmp_path / "r.md").read_text()


def test_files_section(tmp_path):
    d = RomDiff(
        (HeaderChange("game_code", "CPUE", "JAK7"),),
        (
            FileChange("n.narc", "changed", 10, 12, (3, 4, 5, 9), (20, 20)),
            FileChange("c.narc", "changed", 10, 11, (), (4, 4)),
            FileChange("g.narc", "changed", 10, 11, (0,), (4, 5)),
            FileChange("x/a", "added", None, 5),
            FileChange("x/b", "added", None, 7),
            FileChange("gone", "removed", 3, None),
        ),
        (),
        {},
        {},
    )
    write_report(d, {}, tmp_path / "r.md", PROV)
    t = (tmp_path / "r.md").read_text()
    assert "game_code" in t and "CPUE" in t and "JAK7" in t
    assert "3-5, 9" in t
    assert "container changed, no member changed" in t
    assert "4 -> 5 members" in t
    assert "| x | 2 | 12 |" in t
    assert "gone" in t and t.isascii()


def test_relocation_only_symbol_is_listed(tmp_path):
    d = dataclasses.replace(D, relocations={("overlay3", "Bar"): 7, ("overlay3", "Foo"): 2})
    write_report(d, {}, tmp_path / "r.md", PROV)
    t = (tmp_path / "r.md").read_text()
    assert next(line for line in t.splitlines() if line.startswith("| Bar |")).split("|")[4:5] == [
        " 7 "
    ]
    assert foo_row(t).split("|")[4] == " 2 "
