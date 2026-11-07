"""Extract the Platinum Kaizo docs workbook into ``docs.json`` (spec 4, 5; plan Task 7).

The workbook is data: it is opened with openpyxl ``read_only=True``, values with
``data_only=True`` (the cached cell values; nothing is recalculated or evaluated). The split
sheets' AI flag cells hold no cached value, only a reference to the sheet's AI flag key
(``=F$2``), so those cells' formula text is read too (``data_only=False``) and parsed by
``ai_flag``, which accepts nothing but a single cell reference.

Every row written carries ``"cite": "<sheet>:<1-based row>"``. ``docs.json``:

- ``sha256``: the workbook's SHA-256 (``KAIZO_DOCS_SHA256`` is the audited one).
- ``effect_labels``: PokEditor's "Additional Effect" label -> vanilla effect id, from the
  unmodified vanilla export ``Plat Kaizo PokEditor Tester - M`` joined by move id with the
  vanilla moves (one-to-one or ``ExtractError``); ``effect_label_cites``: label -> cite.
- ``moves[id]`` (``Moves``): ``name``, ``label``, ``effect`` (decoded; ``None`` for a label
  the vanilla export never uses), ``category``/``type`` (vanilla ids) and ``target``
  (``RANGE_*``), decoded through the same vanilla join, ``power``, ``accuracy``, ``pp``,
  ``chance``, ``priority``, ``flags`` (``MOVE_FLAG_*``).
- ``move_changes``, ``ability_changes``, ``ai_changes``: rows of their sheets.
- ``species[id]`` (``Personal``), ``evolutions[id]``, ``level_up[id]``, ``tm[id]`` plus
  ``tm_moves`` (``TM Learnsets`` row 2), ``items[name]`` (``Items``, joined by name: the
  sheet's ids are packed, spec 5.5).
- ``raw_trainers[id]`` (``RAW TRAINER DATA`` + ``RAW TRAINER POKEMON``); party mons are
  ``{species, species_name, form, level, difficulty, ability_number, item, moves,
  ball_seal}``.
- ``splits[<split>]`` for the ten ``* Split`` sheets in game order: ``sections`` (``name``,
  ``text``, ``tags``, ``cite``, ``trainers``), each trainer ``{text, name, tags, party
  [{species, level}], ai_cells, ai_known, ai_flags, left_side, cite}``. ``ai_cells`` are
  the AI area's formula cells (``{cell, ref, flag, on_key}``). ``ai_known`` is true when at
  least one cell points at the key row (``on_key``); then ``ai_flags`` lists those cells'
  key names and ``left_side`` says whether "Left Side AI" is among them. When it is false
  the sheet shows no AI icon for the trainer (the export lost most of them), so
  ``ai_flags`` and ``left_side`` are ``None``: unknown, never "off".
- Header ``tags`` (plan Ruling 22, ``GRAMMAR``) use only these keys: ``permanent``
  (``rain|sun|sandstorm|hail|trick_room|gravity``); ``weather`` (``rain|sun|hail``) and
  ``trick_room`` (``true``), each with ``assumed_permanent``; ``fog``; ``format``
  (``double``); ``multi_with``; ``tag_partner``; ``player_starter`` (387|390|393);
  ``gauntlet`` (``start|middle|end``); ``gauntlet_with`` ("Gauntlet With <name>");
  ``b2b_with``; ``gym_gauntlet`` ("GYM TRAINERS ARE A GAUNTLET"); ``optional``; ``note``.
  ``PARTNER`` maps to ``tag_partner`` and the sheet's typo ``MUTLI BATTLE with`` to
  ``multi_with``: neither adds a key.

Usage: python -m tools.extract_kaizo_docs [--xlsx PATH] [--out data/intermediate_kaizo/docs]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from tools.extract_decomp import ExtractError
from tools.rom.inputs import REPO_ROOT, RomInputError, load_inputs

DEFAULT_OUT = REPO_ROOT / "data" / "intermediate_kaizo" / "docs"
VANILLA_INTERMEDIATE = REPO_ROOT / "data" / "intermediate"

VANILLA_MOVES_SHEET = "Plat Kaizo PokEditor Tester - M"
# Split name -> sheet, in game order (the workbook lists Volkner before Galactic).
SPLITS = {
    name: f"{name} Split"
    for name in (
        "Roark", "Gardenia", "Fantina", "Maylene", "Wake",
        "Byron", "Candice", "Galactic", "Volkner", "Elite Four",
    )
}  # fmt: skip

# Moves sheet flag columns (L..S) -> decomp MOVE_FLAG_*; checked against vanilla on every run.
MOVE_FLAG_COLUMNS = (
    "MOVE_FLAG_MAKES_CONTACT",  # Contact Move
    "MOVE_FLAG_CAN_PROTECT",  # Blocked by Protect
    "MOVE_FLAG_CAN_MAGIC_COAT",  # Reflected by Magic Coat
    "MOVE_FLAG_CAN_SNATCH",  # Affected by Snatch
    "MOVE_FLAG_CAN_MIRROR_MOVE",  # Affected by Mirror Move
    "MOVE_FLAG_TRIGGERS_KINGS_ROCK",  # Triggers Kings Rock
    "MOVE_FLAG_HIDES_HP_GAUGES",  # Hide HP Bars
    "MOVE_FLAG_HIDES_SHADOWS",  # Remove Target's Shadow
)
MOVES_HEADER = (
    "ID Number", "Name", "Additional Effect", "Category", "Power", "Type", "Accuracy", "PP",
    "Additional Effect Chance (%)", "Target(s)", "Priority", "Contact Move",
    "Blocked by Protect", "Reflected by Magic Coat", "Affected by Snatch",
    "Affected by Mirror Move", "Triggers Kings Rock", "Hide HP Bars",
    "Remove Target's Shadow",
)  # fmt: skip
PERSONAL_HEADER = (
    "ID Number", "Name", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "BST",
    "Type 1", "Type 2", "Catch Rate", "Exp Drop", "Uncommon Held Item", "Rare Held Item",
    "Gender Ratio", "Hatch Multiplier", "Base Happiness", "Growth Rate", "Egg Group 1",
    "Egg Group 2", "Ability 1", "Ability 2", "Safari Zone Run Chance",
)  # fmt: skip
ITEMS_HEADER = ("ID Number", "Name", "Price", "Battle Script", "Power", "Pluck Effect",
                "Fling Effect", "Fling Power")  # fmt: skip
TRAINER_DATA_HEADER = (
    "ID Number", "Name", "Moves", "Held Item", "Trainer Class", "Battle Type",
    "Number of Pokemon", "Item 1", "Item 2", "Item 3", "Item 4",
)  # fmt: skip
# RAW TRAINER DATA columns L..Y: the trainer's AI flag bits, in the sheet's order.
TRAINER_AI_COLUMNS = (
    "Prioritize Effectiveness", "Evaluate Attacks", "Expert", "Prioritize Status",
    "Risky Attacks", "Prioritize Damage", "Partner", "Double Battle", "Prioritize Healing",
    "Utilize Weather", "Harassment", "Roaming Pokemon", "Safari Zone", "Catching Demo",
)  # fmt: skip
TRAINER_MON_HEADER = (
    "Difficulty Value", "Ability Number", "Level", "Species", "Form Number", "Held Item",
    "Move 1", "Move 2", "Move 3", "Move 4", "Ball Seal",
)  # fmt: skip
# Split sheets: row 3, columns F..Q, the AI flag key the AI cells point at (row 2).
AI_KEY = ("Basic", "Eval Att", "Expert", "Status", "Risky", "Damage Prio", "Baton P ass",
          "Tag AI", "Check HP", "Weather", "Harrassment", "Left Side AI")  # fmt: skip
AI_KEY_COLUMNS = "FGHIJKLMNOPQ"
LEFT_SIDE = "Left Side AI"
_AI_REF = re.compile(r"=\$?([A-Z]+)\$?(\d+)")
# Pictures in the split sheets' B..D: GETTRAINERSPRITE (a workbook LAMBDA building an image
# URL from the header) and IMAGE("https://...") cells.
_SPRITE = re.compile(r'=GETTRAINERSPRITE\(E\d+\)|=IMAGE\("https://[^"]*"(?:\s*,\s*\d+)*\)')
_LEVEL = re.compile(r"lv\.?\s*(\d+)\s+(.+?)\s*[♂♀]?\s*", re.IGNORECASE)
_PARTY_COLUMNS = range(4, 15)  # E..O
_BLANK_COLUMNS = range(1, 17)  # B..Q

# --- Header tag grammar (plan Ruling 22) ---------------------------------------------

STARTERS = {"turtwig": 387, "chimchar": 390, "piplup": 393}
_Tags = dict[str, object]
_Producer = Callable[[re.Match[str]], _Tags]


def _const(tags: _Tags) -> _Producer:
    return lambda m: dict(tags)


# An item containing one of these that the grammar does not consume raises (Ruling 22).
KEYWORDS = (
    "PERMANENT", "RAIN", "SUN", "HAIL", "SANDSTORM", "FOG", "TRICK ROOM", "GRAVITY", "DOUBLE",
    "MULTI", "MUTLI", "TAG", "PARTNER", "GAUNTLET", "B2B", "OPTIONAL", "picked", "chose",
    "selected",
)  # fmt: skip
_KEYWORD = re.compile(r"\b(?:" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)
_KEYWORD_FREE = r"(?:(?!\b(?:" + "|".join(KEYWORDS) + r")\b).)+"

# Each entry is matched case-insensitively at the start of what is left of an item, and must
# end at a word boundary; an item may hold several (``OPTIONAL DOUBLE BATTLE``) and must be
# consumed whole: a remainder raises ``unparsed tag``. Order matters where one pattern is a
# prefix of another. A ``(.+)`` name runs to the item's end. Free text is a note only as a
# whole item without a keyword, or through the explicit ``optional <text>`` entry.
GRAMMAR: tuple[tuple[str, _Producer], ...] = (
    (r"permanent (rain|sun|sandstorm|hail)", lambda m: {"permanent": m[1].lower()}),
    (r"permanent trick room", _const({"permanent": "trick_room"})),
    (r"permanent gravity", _const({"permanent": "gravity"})),
    (r"rain", _const({"weather": "rain", "assumed_permanent": True})),
    (r"hail", _const({"weather": "hail", "assumed_permanent": True})),
    # Bare (SUN) read as permanent: USER-CONFIRMED (plan Ruling 22, flagged for the Task 19
    # review). To read it otherwise, change this one entry.
    (r"sun", _const({"weather": "sun", "assumed_permanent": True})),
    (r"fog", _const({"fog": True})),
    (r"trick room", _const({"trick_room": True, "assumed_permanent": True})),
    (r"double(?: battle)?", _const({"format": "double"})),
    # "MUTLI" is the sheet's typo (Galactic Split:1143).
    (r"(?:multi|mutli) battle with (.+)", lambda m: {"multi_with": m[1].strip()}),
    (r"tag battle with (.+)", lambda m: {"tag_partner": True, "multi_with": m[1].strip()}),
    (r"tag partner", _const({"tag_partner": True})),
    # Bare (PARTNER): Trainer Marley at Lake Valor (Candice Split:7).
    (r"partner", _const({"tag_partner": True})),
    (
        r"(?:if )?player (?:chose|picked|selected) (turtwig|chimchar|piplup)",
        lambda m: {"player_starter": STARTERS[m[1].lower()]},
    ),
    (r"gym trainers are a gauntlet", _const({"gym_gauntlet": True})),
    (r"gauntlet start", _const({"gauntlet": "start"})),
    (r"gauntlet end", _const({"gauntlet": "end"})),
    # "OPTIONAL Gauntlet With Mira" (Galactic Split:105, a section).
    (r"gauntlet with (.+)", lambda m: {"gauntlet_with": m[1].strip()}),
    (r"gauntlet", _const({"gauntlet": "middle"})),
    (r"b2b with (.+)", lambda m: {"b2b_with": m[1].strip()}),
    # "OPTIONAL Top Trainer" (Galactic Split:432), "Optional Bottom Path" (Galactic
    # Split:472): the only free text a tag may carry, and only text without a keyword.
    (rf"optional ({_KEYWORD_FREE})$", lambda m: {"optional": True, "note": m[1].strip()}),
    (r"optional", _const({"optional": True})),
)
_GRAMMAR = tuple(
    (re.compile(rf"{pattern}(?:\b|$)\s*", re.IGNORECASE), produce) for pattern, produce in GRAMMAR
)
_PAREN = re.compile(r"\(([^()]*)\)")
_TAG_PARTNER_PREFIX = re.compile(r"tag partner\s+(.+)", re.IGNORECASE)


def split_header(text: str, cite: str = "?") -> tuple[str, list[str]]:
    """``(name, items)`` of a section or trainer header: the parentheticals' comma-separated
    items, plus a ``" - "`` suffix of the name that carries a keyword (``Wayward Cave -
    OPTIONAL Gauntlet With Mira``). A trailing ``*`` (a cell comment) or ``.`` is dropped."""
    body = re.sub(r"[\s.*]+$", "", text.strip())
    groups = _PAREN.findall(body)
    name = _PAREN.sub(" ", body)
    if "(" in name or ")" in name:
        raise ExtractError(f"{cite}: unbalanced parenthesis in header {text!r}")
    name = re.sub(r"\s+", " ", name).strip().rstrip(" -").strip()
    items = [item.strip() for group in groups for item in group.split(",") if item.strip()]
    prefix, sep, suffix = name.partition(" - ")
    if sep and _KEYWORD.search(suffix):
        name, items = prefix.strip(), [suffix.strip(), *items]
    prefixed = _TAG_PARTNER_PREFIX.fullmatch(name)
    if prefixed:  # "TAG PARTNER MIRA" (Galactic Split:1133)
        name, items = prefixed[1], ["TAG PARTNER", *items]
    return name, items


def _parse_item(item: str, abilities: set[str], cite: str) -> tuple[_Tags, str | None]:
    """Tags and the note (if any) of one item."""
    if item.casefold() in abilities:
        return {}, item
    tags: _Tags = {}
    rest = item
    while rest:
        for pattern, produce in _GRAMMAR:
            m = pattern.match(rest)
            if m:
                _merge(tags, produce(m), cite)
                rest = rest[m.end() :]
                break
        else:
            break
    if not rest:
        return tags, tags.pop("note", None)
    if rest == item and not _KEYWORD.search(item):
        return {}, item  # a whole item without a keyword: a location or remark
    raise ExtractError(f"{cite}: unparsed tag '{item}'")


def _merge(tags: _Tags, new: _Tags, cite: str) -> None:
    for key, value in new.items():
        if key in tags and tags[key] != value:
            raise ExtractError(f"{cite}: conflicting tags {key}={tags[key]!r} and {value!r}")
        tags[key] = value


def parse_header(text: str, abilities: Iterable[str], cite: str = "?") -> _Tags:
    """The tags of a section or trainer header (plan Ruling 22). Ability names and items
    without a keyword are notes (``note``, joined by ``", "``)."""
    names = {a.casefold() for a in abilities}
    tags: _Tags = {}
    notes = []
    for item in split_header(text, cite)[1]:
        item_tags, note = _parse_item(item, names, cite)
        _merge(tags, item_tags, cite)
        if note is not None:
            notes.append(note)
    if notes:
        tags["note"] = ", ".join(notes)
    return tags


def ai_flag(formula: str, cite: str) -> str:
    """The AI flag key column a split's AI cell refers to: a single cell reference into the
    key's columns F..Q. Only a reference to row 2 shows the key's icon; cells the sheet's
    authors copied without anchoring (``=J104``) show another cell instead, so ``_split``
    keeps them as ``ai_cells`` with ``on_key: false`` and leaves them out of ``ai_flags``."""
    m = _AI_REF.fullmatch(formula)
    if not m or m[1] not in AI_KEY_COLUMNS or len(m[1]) != 1:
        raise ExtractError(f"{cite}: AI flag cell {formula!r} is not a reference to the key")
    return AI_KEY[AI_KEY_COLUMNS.index(m[1])]


# --- Cell values ----------------------------------------------------------------------

_Rows = list[tuple]


def _cell(row: tuple, col: int):
    return row[col] if col < len(row) else None


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _int(value, cite: str, what: str) -> int:
    """An integer cell: a whole number, or text ``"92 (+10)"`` (the sheet's change note)."""
    if isinstance(value, bool):
        raise ExtractError(f"{cite}: {what} is a boolean ({value!r})")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(-?\d+)(?:\s*\([+-]?\d+\))?\s*", value)
        if m:
            return int(m[1])
    raise ExtractError(f"{cite}: {what} {value!r} is not an integer")


def _text(value, cite: str, what: str) -> str:
    if not isinstance(value, str):
        raise ExtractError(f"{cite}: {what} {value!r} is not text")
    return value.strip()


def _bool(value, cite: str, what: str) -> bool:
    if not isinstance(value, bool):
        raise ExtractError(f"{cite}: {what} {value!r} is not TRUE/FALSE")
    return value


def _optional_name(value, cite: str, what: str) -> str | None:
    """Item and move cells: ``None``/``-``/empty -> ``None``."""
    if _blank(value):
        return None
    name = _text(value, cite, what)
    return None if name in ("None", "-", "(None)") else name


def _check_header(rows: _Rows, sheet: str, expected: tuple[str, ...], row: int = 1) -> None:
    got = tuple(rows[row - 1][: len(expected)]) if len(rows) >= row else ()
    if got != expected:
        raise ExtractError(f"{sheet}:{row}: header {got!r}, expected {expected!r}")


def _data_rows(rows: _Rows, first: int = 2):
    """``(row number, row)`` for rows whose column A holds an id, from ``first``."""
    for n, row in enumerate(rows[first - 1 :], first):
        if row and not _blank(row[0]):
            yield n, row


def _by_id(sheet: str, rows: _Rows, build, first: int = 2) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for n, row in _data_rows(rows, first):
        cite = f"{sheet}:{n}"
        key = str(_int(row[0], cite, "id"))
        if key in out:
            raise ExtractError(f"{cite}: duplicate id {key} (first at {out[key]['cite']})")
        out[key] = {**build(row, cite), "cite": cite}
    return out


# --- Moves ----------------------------------------------------------------------------


def _join_map(pairs: list[tuple[str, object, str]], what: str) -> dict[str, object]:
    """``label -> vanilla value`` from ``(label, value, cite)`` pairs; one-to-one or raise."""
    out: dict[str, object] = {}
    first: dict[str, str] = {}
    for label, value, cite in pairs:
        if label in out and out[label] != value:
            raise ExtractError(
                f"{cite}: {what} label {label!r} means {value!r} here and {out[label]!r} at "
                f"{first[label]}"
            )
        out.setdefault(label, value)
        first.setdefault(label, cite)
    values = list(out.values())
    if len(set(map(repr, values))) != len(values):
        raise ExtractError(f"{VANILLA_MOVES_SHEET}: {what} labels are not one-to-one")
    return out


def _move_flags(row: tuple, cite: str) -> list[str]:
    return [
        flag
        for flag, col in zip(MOVE_FLAG_COLUMNS, range(11, 19), strict=True)
        if _bool(_cell(row, col), cite, flag)
    ]


def _vanilla_join(rows: _Rows, vanilla_moves: list[dict]) -> dict[str, dict]:
    """The label maps of the vanilla export, checked against the vanilla moves."""
    sheet = VANILLA_MOVES_SHEET
    _check_header(rows, sheet, MOVES_HEADER)
    pairs: dict[str, list] = {"effect": [], "category": [], "type": [], "target": []}
    cites: dict[str, str] = {}
    seen = set()
    for n, row in _data_rows(rows):
        cite = f"{sheet}:{n}"
        move_id = _int(row[0], cite, "id")
        if move_id >= len(vanilla_moves):
            continue  # the export's placeholder rows 468-470 have no vanilla move
        seen.add(move_id)
        vanilla = vanilla_moves[move_id]
        if _cell(row, 1) != vanilla["name"]:
            raise ExtractError(f"{cite}: move {move_id} is {row[1]!r}, vanilla {vanilla['name']!r}")
        for field, col in (("power", 4), ("accuracy", 6), ("pp", 7), ("priority", 10)):
            if _int(row[col], cite, field) != vanilla[field]:
                raise ExtractError(f"{cite}: {field} differs from vanilla move {move_id}")
        if set(_move_flags(row, cite)) != set(vanilla["flags"]):
            raise ExtractError(f"{cite}: flag columns differ from vanilla move {move_id}'s flags")
        label = _text(row[2], cite, "effect label")
        pairs["effect"].append((label, vanilla["effect"], cite))
        cites.setdefault(label, cite)
        for field, col in (("category", 3), ("type", 5), ("target", 9)):
            pairs[field].append((_text(row[col], cite, field), vanilla[field], cite))
    if seen != set(range(len(vanilla_moves))):
        raise ExtractError(
            f"{sheet}: moves {sorted(set(range(len(vanilla_moves))) - seen)} missing"
        )
    maps = {field: _join_map(p, field) for field, p in pairs.items()}
    return {**maps, "effect_cites": cites}


def _decode(maps: dict, field: str, value, cite: str):
    label = _text(value, cite, field)
    if label not in maps[field]:
        raise ExtractError(f"{cite}: {field} {label!r} is not in the vanilla export")
    return maps[field][label]


def _moves(rows: _Rows, maps: dict) -> dict[str, dict]:
    _check_header(rows, "Moves", MOVES_HEADER)

    def build(row, cite):
        name = row[1]
        label = _text(row[2], cite, "effect label")
        return {
            # Rows 468-470 are unnamed placeholders whose name cell holds their id.
            "name": str(_int(name, cite, "name")) if not isinstance(name, str) else name.strip(),
            "label": label,
            "effect": maps["effect"].get(label),
            "category": _decode(maps, "category", row[3], cite),
            "power": _int(row[4], cite, "power"),
            "type": _decode(maps, "type", row[5], cite),
            "accuracy": _int(row[6], cite, "accuracy"),
            "pp": _int(row[7], cite, "pp"),
            "chance": _int(row[8], cite, "chance"),
            "target": _decode(maps, "target", row[9], cite),
            "priority": _int(row[10], cite, "priority"),
            "flags": _move_flags(row, cite),
        }

    return _by_id("Moves", rows, build)


def _text_rows(sheet: str, rows: _Rows, col: int, first: int) -> list[dict]:
    return [
        {"text": row[col].strip(), "cite": f"{sheet}:{n}"}
        for n, row in enumerate(rows[first - 1 :], first)
        if not _blank(_cell(row, col))
    ]


# --- Species, learnsets, items ----------------------------------------------------------


def _species(rows: _Rows, types: dict) -> dict[str, dict]:
    _check_header(rows, "Personal", PERSONAL_HEADER)

    def build(row, cite):
        def n(col, what):
            return _int(row[col], cite, what)

        def t(col, what):
            return _text(row[col], cite, what)

        return {
            "name": t(1, "name"),
            "base_stats": [n(c, PERSONAL_HEADER[c]) for c in range(2, 8)],
            "types": [_decode({"type": types}, "type", row[c], cite) for c in (9, 10)],
            "catch_rate": n(11, "catch rate"),
            "exp_yield": n(12, "exp drop"),
            "items": [_optional_name(row[c], cite, "held item") for c in (13, 14)],
            "gender_ratio": n(15, "gender ratio"),
            "hatch_cycles": n(16, "hatch multiplier"),
            "base_happiness": n(17, "base happiness"),
            "growth_rate": t(18, "growth rate"),
            "egg_groups": [t(19, "egg group 1"), t(20, "egg group 2")],
            "abilities": [t(21, "ability 1"), t(22, "ability 2")],
            "safari_flee": n(23, "safari zone run chance"),
        }

    return _by_id("Personal", rows, build)


def _ability_changes(rows: _Rows) -> list[dict]:
    sheet = "Ability Changes"
    if tuple(rows[0][1:3]) != ("Ability 1", "Ability 2"):
        raise ExtractError(f"{sheet}:1: header {rows[0]!r}")
    out = []
    for n, row in enumerate(rows[1:], 2):
        if all(_blank(v) for v in row):
            continue
        cite = f"{sheet}:{n}"
        out.append(
            {
                "species": _text(row[0], cite, "species"),
                "abilities": [_text(row[1], cite, "ability 1"), _text(row[2], cite, "ability 2")],
                "cite": cite,
            }
        )
    return out


def _evolutions(rows: _Rows) -> dict[str, dict]:
    _check_header(rows, "Evolutions", ("ID Number", "Name", "Method", "Required", "Result"))

    def build(row, cite):
        evos = []
        for col in range(2, len(row), 3):
            method = _text(row[col], cite, "method")
            if method == "None":
                continue
            required = _cell(row, col + 1)
            if isinstance(required, float):
                required = _int(required, cite, "required")
            evos.append(
                {
                    "method": method,
                    "required": required,
                    "result": _text(_cell(row, col + 2), cite, "result"),
                }
            )
        return {"name": _text(row[1], cite, "name"), "evolutions": evos}

    return _by_id("Evolutions", rows, build)


def _level_up(rows: _Rows) -> dict[str, dict]:
    _check_header(rows, "Level-Up Learnsets", ("ID Number", "Name", "Move", "Level"))

    def build(row, cite):
        moves = []
        for col in range(2, len(row), 2):
            move, level = _cell(row, col), _cell(row, col + 1)
            if _blank(move) and level is None:
                continue
            moves.append([_text(move, cite, "move"), _int(level, cite, "level")])
        return {"name": _text(row[1], cite, "name"), "moves": moves}

    return _by_id("Level-Up Learnsets", rows, build)


def _tm(rows: _Rows) -> tuple[dict[str, dict], dict[str, dict]]:
    sheet = "TM Learnsets"
    header = list(rows[0])
    while header and header[-1] is None:
        header.pop()
    machines = [_text(h, f"{sheet}:1", "machine") for h in header[2:]]
    if machines[0] != "TM01" or machines[-1] != "HM08" or len(machines) != 100:
        raise ExtractError(f"{sheet}:1: machine columns {machines[:1]}..{machines[-1:]}")
    tm_moves = {
        m: {"move": _text(v, f"{sheet}:2", m), "cite": f"{sheet}:2"}
        for m, v in zip(machines, rows[1][2 : 2 + len(machines)], strict=True)
    }

    def build(row, cite):
        cells = list(row[2 : 2 + len(machines)])
        if len(cells) != len(machines) or any(not _blank(v) for v in row[2 + len(machines) :]):
            raise ExtractError(f"{cite}: expected {len(machines)} machine cells")
        return {
            "name": _text(row[1], cite, "name"),
            "machines": [m for m, v in zip(machines, cells, strict=True) if _bool(v, cite, m)],
        }

    return _by_id(sheet, rows, build, first=3), tm_moves


def _items(rows: _Rows) -> dict[str, dict]:
    sheet = "Items"
    _check_header(rows, sheet, ITEMS_HEADER)
    out: dict[str, dict] = {}
    for n, row in _data_rows(rows):
        cite = f"{sheet}:{n}"
        name = _text(row[1], cite, "name")
        if name in out:
            raise ExtractError(f"{cite}: duplicate item {name!r} (first at {out[name]['cite']})")
        out[name] = {
            "sheet_id": _int(row[0], cite, "id"),
            "hold_effect": _int(row[3], cite, "battle script"),
            "hold_param": _int(row[4], cite, "power"),
            "fling_power": _int(row[7], cite, "fling power"),
            "cite": cite,
        }
    return out


# --- RAW trainers ---------------------------------------------------------------------


def _raw_trainers(data: _Rows, mons: _Rows, species: dict[str, dict]) -> dict[str, dict]:
    _check_header(data, "RAW TRAINER DATA", TRAINER_DATA_HEADER)
    ai_cols = tuple(data[0][11:25])
    if ai_cols != TRAINER_AI_COLUMNS or data[0][25] != "Battle Type":
        raise ExtractError(f"RAW TRAINER DATA:1: AI columns {ai_cols!r}")
    _check_header(mons, "RAW TRAINER POKEMON", ("ID Number", "Name", *TRAINER_MON_HEADER))
    # Base species by name (Personal 0..493; 0 is "-----", in empty slots such as trainer
    # 62's); forms are the Form Number column.
    by_name: dict[str, int] = {}
    for k, s in species.items():
        if int(k) <= 493:
            if s["name"] in by_name:
                raise ExtractError(f"{s['cite']}: species name {s['name']!r} is not unique")
            by_name[s["name"]] = int(k)
    width = 2 + len(TRAINER_MON_HEADER)
    out: dict[str, dict] = {}
    for n, row in _data_rows(data):
        cite = f"RAW TRAINER DATA:{n}"
        tid = _int(row[0], cite, "id")
        mon_row = mons[n - 1] if n <= len(mons) else ()
        mon_cite = f"RAW TRAINER POKEMON:{n}"
        if not mon_row or _cell(mon_row, 0) != row[0] or _cell(mon_row, 1) != row[1]:
            raise ExtractError(f"{mon_cite}: does not match trainer {tid} at {cite}")
        size = _int(row[6], cite, "number of pokemon")
        party = []
        for slot in range(6):
            cells = tuple(mon_row[2 + 11 * slot : width + 11 * slot])
            filled = any(v is not None for v in cells)
            if slot >= size:
                if filled:
                    raise ExtractError(f"{mon_cite}: slot {slot + 1} beyond party size {size}")
                continue
            name = _text(cells[3], mon_cite, "species")
            if name not in by_name:
                raise ExtractError(f"{mon_cite}: species {name!r} is not in Personal")
            party.append(
                {
                    "species": by_name[name],
                    "species_name": name,
                    "form": _int(cells[4], mon_cite, "form"),
                    "level": _int(cells[2], mon_cite, "level"),
                    "difficulty": _int(cells[0], mon_cite, "difficulty"),
                    "ability_number": _int(cells[1], mon_cite, "ability number"),
                    "item": _optional_name(cells[5], mon_cite, "held item"),
                    "moves": [_optional_name(m, mon_cite, "move") for m in cells[6:10]],
                    "ball_seal": _int(cells[10], mon_cite, "ball seal"),
                }
            )
        out[str(tid)] = {
            "name": _text(row[1], cite, "name"),
            "has_moves": _bool(row[2], cite, "moves"),
            "has_items": _bool(row[3], cite, "held item"),
            "class": _int(row[4], cite, "trainer class"),
            "battle_type_value": _int(row[5], cite, "battle type"),
            "battle_type": _text(row[25], cite, "battle type"),
            "items": [_optional_name(row[c], cite, "item") for c in range(7, 11)],
            "ai": [
                c for c, v in zip(TRAINER_AI_COLUMNS, row[11:25], strict=True) if _bool(v, cite, c)
            ],
            "party": party,
            "cite": cite,
            "party_cite": mon_cite,
        }
    return out


# --- Split sheets ---------------------------------------------------------------------


def _party_cells(row: tuple) -> list[tuple[int, re.Match[str] | None]]:
    return [(c, _LEVEL.fullmatch(str(row[c]))) for c in _PARTY_COLUMNS if not _blank(_cell(row, c))]


def _split(
    split: str, sheet: str, rows: _Rows, formulas: dict[tuple[int, int], str], abilities: set
) -> dict:
    key = tuple(_cell(rows[2], c) for c in range(5, 17)) if len(rows) > 2 else ()
    if key != AI_KEY:
        raise ExtractError(f"{sheet}:3: AI flag key {key!r}")
    blank = [all(_blank(_cell(r, c)) for c in _BLANK_COLUMNS) for r in rows]

    # Party rows: a "Lv N Species" cell in E..O. The header is the text in E above it.
    party_rows: dict[int, int] = {}  # header row -> party row
    for i, row in enumerate(rows, 1):
        cells = _party_cells(row)
        if not any(m for _, m in cells):
            continue
        bad = [row[c] for c, m in cells if not m]
        if bad:
            raise ExtractError(f"{sheet}:{i}: party cells {bad!r} are not 'Lv N Species'")
        h = i - 1
        while h > 3 and blank[h - 1]:
            h -= 1
        if i - h > 4 or _blank(_cell(rows[h - 1], 4)):  # 4: Elite Four Split:248/252
            raise ExtractError(f"{sheet}:{i}: party row without a header in column E above it")
        party_rows[h] = i
    in_block = set()
    for p in party_rows.values():
        r = p
        while r <= len(rows) and not blank[r - 1]:
            in_block.add(r)
            r += 1

    sections: list[dict] = []
    owners: list[tuple[int, dict | None]] = []  # (row, trainer or None for a section)
    for i, row in enumerate(rows[3:], 4):
        if i in in_block or blank[i - 1]:
            continue
        cite = f"{sheet}:{i}"
        text = _cell(row, 4)
        if _blank(text) or any(not _blank(_cell(row, c)) for c in _BLANK_COLUMNS if c != 4):
            raise ExtractError(f"{cite}: unexpected cells outside a trainer block: {row!r}")
        text = _text(text, cite, "header")
        name, _ = split_header(text, cite)
        tags = parse_header(text, abilities, cite)
        if i in party_rows:
            if not sections:
                raise ExtractError(f"{cite}: trainer before the first section")
            trainer = {
                "text": text,
                "name": name,
                "tags": tags,
                "party": [
                    {"species": m[2], "level": int(m[1])}
                    for _, m in _party_cells(rows[party_rows[i] - 1])
                    if m
                ],
                "cite": cite,
            }
            sections[-1]["trainers"].append(trainer)
            owners.append((i, trainer))
        else:
            sections.append(
                {"name": name, "text": text, "tags": tags, "cite": cite, "trainers": []}
            )
            owners.append((i, None))

    # AI cells: every formula in B..D after a trainer header and before the next header,
    # except the sprites (trainer pictures beside headers and in the AI area).
    for (r, c), formula in sorted(formulas.items()):
        if r <= 3 or _SPRITE.fullmatch(formula):
            continue
        owner = [t for row, t in owners if row <= r]
        trainer = owner[-1] if owner else None
        cell = f"{'ABCD'[c - 1]}{r}"
        if trainer is None or r == int(trainer["cite"].rsplit(":", 1)[1]):
            raise ExtractError(f"{sheet}:{r}: formula {formula!r} in {cell} outside an AI area")
        flag = ai_flag(formula, f"{sheet}:{r}")
        on_key = _AI_REF.fullmatch(formula)[2] == "2"
        trainer.setdefault("ai_cells", []).append(
            {"cell": cell, "ref": formula[1:], "flag": flag, "on_key": on_key}
        )
    for _, trainer in owners:
        if trainer is not None:
            cells = trainer.setdefault("ai_cells", [])
            on_key = {x["flag"] for x in cells if x["on_key"]}
            trainer["ai_known"] = bool(on_key)
            trainer["ai_flags"] = [f for f in AI_KEY if f in on_key] if on_key else None
            trainer["left_side"] = (LEFT_SIDE in on_key) if on_key else None
    return {"sheet": sheet, "sections": sections}


def _split_formulas(xlsx: Path) -> dict[str, dict[tuple[int, int], str]]:
    """Formula text of columns B..D of the split sheets (never evaluated)."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=False)
    try:
        out: dict[str, dict[tuple[int, int], str]] = {}
        for sheet in SPLITS.values():
            cells = {}
            for row in wb[sheet].iter_rows(min_col=2, max_col=4):
                for cell in row:
                    v = getattr(cell, "value", None)
                    if isinstance(v, str) and v.startswith("="):
                        cells[(cell.row, cell.column)] = v
            out[sheet] = cells
        return out
    finally:
        wb.close()


def _read_values(xlsx: Path) -> dict[str, _Rows]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        return {
            ws.title: [tuple(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets
        }
    finally:
        wb.close()


# --- Entry points ---------------------------------------------------------------------


def extract_kaizo_docs(
    xlsx: Path, vanilla_moves: list[dict], vanilla_abilities: list[str], out: Path
) -> dict:
    """Read the workbook and write ``out/docs.json``; returns the document."""
    sha256 = hashlib.sha256(xlsx.read_bytes()).hexdigest()
    sheets = _read_values(xlsx)
    missing = [
        s
        for s in (
            VANILLA_MOVES_SHEET,
            "Moves",
            "Move Changes",
            "Personal",
            "Ability Changes",
            "Evolutions",
            "Level-Up Learnsets",
            "TM Learnsets",
            "Items",
            "AI Changes",
            "RAW TRAINER DATA",
            "RAW TRAINER POKEMON",
            *SPLITS.values(),
        )
        if s not in sheets
    ]
    if missing:
        raise ExtractError(f"{xlsx}: sheets {missing} are missing")
    maps = _vanilla_join(sheets[VANILLA_MOVES_SHEET], vanilla_moves)
    species = _species(sheets["Personal"], maps["type"])
    tm, tm_moves = _tm(sheets["TM Learnsets"])
    formulas = _split_formulas(xlsx)
    abilities = {a.casefold() for a in vanilla_abilities}
    docs = {
        "sha256": sha256,
        "effect_labels": maps["effect"],
        "effect_label_cites": maps["effect_cites"],
        "moves": _moves(sheets["Moves"], maps),
        "move_changes": _text_rows("Move Changes", sheets["Move Changes"], 1, 2),
        "species": species,
        "ability_changes": _ability_changes(sheets["Ability Changes"]),
        "evolutions": _evolutions(sheets["Evolutions"]),
        "level_up": _level_up(sheets["Level-Up Learnsets"]),
        "tm": tm,
        "tm_moves": tm_moves,
        "items": _items(sheets["Items"]),
        "ai_changes": _text_rows("AI Changes", sheets["AI Changes"], 0, 1),
        "raw_trainers": _raw_trainers(
            sheets["RAW TRAINER DATA"], sheets["RAW TRAINER POKEMON"], species
        ),
        "splits": {
            split: _split(split, sheet, sheets[sheet], formulas[sheet], abilities)
            for split, sheet in SPLITS.items()
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "docs.json").write_text(json.dumps(docs, indent=1, ensure_ascii=False), "utf-8")
    return docs


def load_vanilla(inter: Path = VANILLA_INTERMEDIATE) -> tuple[list[dict], list[str]]:
    """Vanilla ``moves.json`` rows and ``abilities.json`` names."""
    moves = json.loads((inter / "moves.json").read_text(encoding="utf-8"))
    abilities = [a["name"] for a in json.loads((inter / "abilities.json").read_text("utf-8"))]
    return moves, abilities


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--xlsx", type=Path, help="workbook (default: kaizo_docs of local.toml)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vanilla", type=Path, default=VANILLA_INTERMEDIATE)
    args = p.parse_args(argv)
    try:
        xlsx = args.xlsx or load_inputs().kaizo_docs
        if xlsx is None:
            raise RomInputError("kaizo_docs is not set (data/local.toml or NUZLINE_KAIZO_DOCS)")
        moves, abilities = load_vanilla(args.vanilla)
        docs = extract_kaizo_docs(xlsx, moves, abilities, args.out)
    except (RomInputError, ExtractError, OSError) as e:
        message = str(e).encode("ascii", "backslashreplace").decode()
        print(f"error: {message}", file=sys.stderr)
        return 1
    trainers = sum(len(s["trainers"]) for sp in docs["splits"].values() for s in sp["sections"])
    print(f"sha256: {docs['sha256']}")
    print(
        f"effect labels: {len(docs['effect_labels'])}; moves: {len(docs['moves'])}; "
        f"species: {len(docs['species'])}; items: {len(docs['items'])}; "
        f"raw trainers: {len(docs['raw_trainers'])}; split trainers: {trainers}"
    )
    print(f"wrote {args.out / 'docs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
