"""Cross-check the parser against PKHeX's view of the user's own save (spec section 8).

Skipped unless ``NUZLINE_ORACLE_DIR`` names a folder holding exactly one ``.sav``/``.dsv``,
one PKHeX Box Data Report ``.csv`` and the party Pokemon as ``.pk4`` files (see
``docs/saves.md``). ``NUZLINE_ORACLE_GAME`` (``vanilla`` or ``kaizo``, default ``vanilla``)
picks the tables and names the save is read with. The PKHeX CSV column headers are not
known exactly, so columns are found by header name with a small alias table and the test
fails listing the headers it saw when a required one has no match.
"""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from pathlib import Path

import pytest

from nuzline.saves import layout as L
from nuzline.saves import load_save
from nuzline.saves.crypto import decrypt_record
from nuzline.saves.save import _block, default_tables_and_names, read_image, select_copies

_DIR = os.environ.get("NUZLINE_ORACLE_DIR")
_GAME = os.environ.get("NUZLINE_ORACLE_GAME", "vanilla")

pytestmark = pytest.mark.skipif(not _DIR, reason="NUZLINE_ORACLE_DIR not set")

# ids.Stat order: HP, Atk, Def, SpA, SpD, Spe. "spd" is read as Special Defense and
# "spe"/"speed" as Speed (the PKHeX report's convention is assumed, not verified).
_STAT_TOKENS = (
    ("hp",),
    ("atk", "attack"),
    ("def", "defense", "defence"),
    ("spa", "spatk", "spatt", "specialattack"),
    ("spd", "spdef", "specialdefense"),
    ("spe", "speed"),
)

_FIELD_ALIASES = {
    "box": ("box", "boxnumber", "boxno", "boxindex", "boxid"),
    "slot": ("slot", "boxslot", "position"),
    "species": ("species",),
    "level": ("level", "lvl"),
    "nature": ("nature",),
    "ability": ("ability",),
    "item": ("helditem", "item"),
    **{f"move{i}": (f"move{i}",) for i in range(1, 5)},
}


def _key(text: str) -> str:
    """Header/name comparison key: case-folded, ASCII letters and digits only, with the
    gender signs kept as f/m so "Nidoran♀" and "Nidoran♂" stay distinct. Drops spaces,
    hyphens, dots and apostrophes ("Mr. Mime", "Farfetch'd", "Double-Edge")."""
    text = text.replace("\u2640", "f").replace("\u2642", "m")
    # Fold accents ("Pokemon" with an e-acute -> "pokemon").
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _name_key(name: str) -> str:
    # PKHeX prints "(None)"/"None" for an empty move or item slot.
    key = _key(name)
    return "" if key in ("none", "") else key


def _find_columns(headers: list[str], rows: list[list[str]]) -> dict[str, int]:
    by_key = {}
    for i, h in enumerate(headers):
        by_key.setdefault(_key(h), i)
    # A custom box name ("Graveyard") makes the "Box" column non-numeric: use another
    # numeric alias column instead.
    box_aliases = _FIELD_ALIASES["box"]
    box_cols = [by_key[a] for a in box_aliases if a in by_key]
    numeric = [
        c
        for c in box_cols
        if all(re.search(r"\d", r[c]) for r in rows if c < len(r) and r[c].strip())
    ]
    if box_cols and not numeric:
        pytest.fail(
            f"no numeric box column (custom box names?); box-like columns {box_cols}, "
            f"headers found: {headers}"
        )
    if numeric:
        by_key["box"] = numeric[0]
    cols, missing = {}, []
    for field, aliases in _FIELD_ALIASES.items():
        hit = next((by_key[a] for a in aliases if a in by_key), None)
        if hit is None:
            missing.append(field)
        cols[field] = hit
    for kind in ("iv", "ev"):
        for stat, tokens in enumerate(_STAT_TOKENS):
            forms = [f"{kind}{t}" for t in tokens] + [f"{t}{kind}" for t in tokens]
            hit = next((by_key[f] for f in forms if f in by_key), None)
            if hit is None:
                missing.append(f"{kind}[{stat}]")
            cols[f"{kind}{stat}"] = hit
    if missing:
        pytest.fail(f"CSV has no column for {missing}; headers found: {headers}")
    return cols


def _int(text: str) -> int:
    m = re.search(r"\d+", text)
    if m is None:
        raise AssertionError(f"no number in {text!r}")
    return int(m.group())


def _oracle_files(folder: Path):
    saves = sorted([*folder.glob("*.sav"), *folder.glob("*.dsv")])
    csvs = sorted(folder.glob("*.csv"))
    assert len(saves) == 1, f"expected exactly one .sav/.dsv in {folder}, found {saves}"
    assert len(csvs) == 1, f"expected exactly one .csv in {folder}, found {csvs}"
    return saves[0], csvs[0], sorted(folder.glob("*.pk4"))


def _check_party(save_path: Path, pk4s: list[Path], party_len: int) -> None:
    image = read_image(save_path.read_bytes())
    _, normal_idx, _ = select_copies(image)
    normal = _block(image, normal_idx, "normal")
    by_pid = {}
    for p in pk4s:
        data = p.read_bytes()
        assert len(data) in (136, 236), f"{p.name}: {len(data)} bytes is not a .pk4"
        assert data[:4] not in by_pid, f"{p.name}: same personality as {by_pid[data[:4]][0]}"
        by_pid[data[:4]] = (p.name, data)
    assert pk4s, "no .pk4 files in the oracle folder"
    for i in range(party_len):
        o = L.PARTY_MONS + i * L.PARTY_MON_SIZE
        ours = decrypt_record(normal[o : o + L.PARTY_MON_SIZE])
        assert ours[:4] in by_pid, f"party slot {i}: no .pk4 with personality {ours[:4].hex()}"
        name, theirs = by_pid[ours[:4]]
        n = min(len(ours), len(theirs))  # PKHeX may write the 136-byte stored form only
        assert ours[:n] == theirs[:n], f"party slot {i} differs from {name}"


def _check_boxes(save, names, csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows, f"{csv_path.name} is empty"
    cols = _find_columns(rows[0], rows[1:])
    problems, checked = [], 0
    for line, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue
        cells = {f: (row[i].strip() if i < len(row) else "") for f, i in cols.items()}
        get = cells.__getitem__
        if not get("species"):
            continue
        box, slot = _int(get("box")), _int(get("slot"))  # 1-based in PKHeX
        where = f"csv line {line} (box {box} slot {slot})"
        if not (1 <= box <= len(save.boxes) and 1 <= slot <= len(save.boxes[0])):
            problems.append(f"{where}: box/slot out of range")
            continue
        mon = save.boxes[box - 1][slot - 1]
        if mon is None:
            problems.append(f"{where}: PKHeX has {get('species')}, we have an empty slot")
            continue
        checked += 1
        want = {
            "species": (_key(get("species")), _key(names.species_names[mon.species])),
            "level": (_int(get("level")), mon.level),
            "nature": (_key(get("nature")), _key(names.nature_names[mon.nature])),
            "ability": (_key(get("ability")), _key(names.ability_names[mon.ability])),
            "item": (
                _name_key(get("item")),
                _name_key(names.item_names[mon.item]) if mon.item else "",
            ),
        }
        for i in range(4):
            move = names.move_names[mon.moves[i]] if mon.moves[i] else ""
            want[f"move{i + 1}"] = (_name_key(get(f"move{i + 1}")), _name_key(move))
        want["ivs"] = (tuple(_int(get(f"iv{s}")) for s in range(6)), tuple(mon.ivs))
        want["evs"] = (tuple(_int(get(f"ev{s}")) for s in range(6)), tuple(mon.evs))
        problems += [
            f"{where}: {field} PKHeX={theirs!r} ours={ours!r}"
            for field, (theirs, ours) in want.items()
            if theirs != ours
        ]
    assert not problems, "\n".join(problems)
    return checked


def test_matches_pkhex():
    save_path, csv_path, pk4s = _oracle_files(Path(_DIR))
    save = load_save(save_path, game=_GAME)
    assert save.load_result == "ok"
    _check_party(save_path, pk4s, len(save.party))

    _, names = default_tables_and_names(None, None, game=_GAME)
    assert _check_boxes(save, names, csv_path) > 0, "the CSV listed no Pokemon"
