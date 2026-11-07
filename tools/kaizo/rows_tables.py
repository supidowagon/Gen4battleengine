"""Cross-check rows for moves, species, learnsets, the TM and tutor tables, items and
abilities (plan Task 12, spec 5.2-5.5, 6.5), fed to ``tools.kaizo.crosscheck.compare``.

Sources: the Kaizo ROM intermediates (``load_rom``; K1/K2a schemas), PKCalc's
``pkcalc.json`` (Task 6) and the docs workbook's ``docs.json`` (Task 7). ``None`` in a row
means that source has no value. Keys are ``"<id> <PKCalc name>"`` (the ROM's, then the
sheet's name when PKCalc has none); TM rows are keyed ``"TM07"``/``"HM01"``.

Value conventions (so ``compare`` sees like with like):

- Every list-valued field is a Python ``list`` (nested pairs too), never a tuple.
- ``target`` is the ``ids.MoveTarget`` name (``"RANGE_SINGLE_TARGET"``); PKCalc's strings go
  through ``PKCALC_TARGETS`` and an unmapped string raises.
- ``effect`` is ROM vs the sheet's label decode (PKCalc has no effect id); a label new to
  Kaizo (decode ``None``) is reported as ``"label: <text>"`` so it never reads as agreement.
- ``abilities`` is the effective pair: slot 2 = 0 means slot 1 (``src/pokemon.c:472-483``
  and ``:4753-4762`` pick slot 1 whenever ability 2 is ``ABILITY_NONE``), so the ROM's
  ``[65, 65]`` and PKCalc's padded ``[65, 0]`` agree.
- Species names compare upper-cased (the ROM's species text bank is upper case).
- Learnset ``tm`` and ``tutor`` sets are sorted move ids, each source going through its own
  table (ROM: ``items.json`` ``teaches_move`` / ``tutor_moves.json``; sheet: ``tm_moves``).
- ``tutor_table`` rows are keyed by move (preflight #32): ``move`` is membership (bool,
  executed); ``cost`` (the four-shard dict, the same shape in both sources) and
  ``location`` (lower-case alphanumerics: ``route_212`` = ``route212``) are compared only
  for moves in both tables, unexecuted.
- Items: the sheet's ``sheet_id`` is packed (no ``ITEM_UNUSED_*`` rows); it is unpacked to
  the item id and checked against the sheet name. ROM ``items.json`` has no fling power,
  so ``fling_power`` rows carry ROM ``None``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nuzline.engine.ids import MoveTarget
from tools.kaizo.crosscheck import Row

ROM_FILES = (
    "moves",
    "species",
    "forms",
    "learnsets",
    "tm_learnsets",
    "tutor_learnsets",
    "tutor_moves",
    "evolutions",
    "evolution_methods",
    "items",
    "abilities",
)

# PKCalc moves.js target strings -> ids.MoveTarget names. The audited moves.js uses only
# the first nine; the three ally/Me First strings follow preflight #33. 'allyTeam' and
# 'any' are left out: no PKCalc move uses them, so no decomp range pins them.
PKCALC_TARGETS = {
    "normal": "RANGE_SINGLE_TARGET",
    "self": "RANGE_USER",
    "allAdjacentFoes": "RANGE_ADJACENT_OPPONENTS",
    "allAdjacent": "RANGE_ALL_ADJACENT",
    "randomNormal": "RANGE_RANDOM_OPPONENT",
    "all": "RANGE_FIELD",
    "allySide": "RANGE_USER_SIDE",
    "foeSide": "RANGE_OPPONENT_SIDE",
    "scripted": "RANGE_SINGLE_TARGET_SPECIAL",
    "adjacentAlly": "RANGE_ALLY",
    "adjacentAllyOrSelf": "RANGE_USER_OR_ALLY",
    "adjacentFoe": "RANGE_SINGLE_TARGET_ME_FIRST",
}
assert set(PKCALC_TARGETS.values()) <= set(MoveTarget.__members__)

PRIORITY_MIN, PRIORITY_MAX = -7, 7  # spec 5.2

# The sheet's growth-rate strings -> EXP_RATE_* (decomp generated/exp_rates.txt order).
GROWTH_RATES = {
    "Medium Fast": 0,
    "Erratic": 1,
    "Fluctuating": 2,
    "Medium Slow": 3,
    "Fast": 4,
    "Slow": 5,
}

# The sheet's (PokEditor's) evolution method labels -> evolution_methods.json names. Read
# off the workbook by aligning each label with the ROM method in the same position; the
# PID labels are PokEditor's (Wurmple: "PID > 5" is the Silcoon entry, EVO_LEVEL_PID_LOW).
# "Use Item (Day)" (Happiny, Evolutions:442) has no Platinum method: it stays a label.
EVOLUTION_LABELS = {
    "Level Up": "level",
    "Happiness": "level_happiness",
    "Happiness (Day)": "level_happiness_day",
    "Happiness (Night)": "level_happiness_night",
    "Trade": "trade",
    "Trade (Held Item)": "trade_with_held_item",
    "Use Item": "use_item",
    "Use Item (Male)": "use_item_male",
    "Use Item (Female)": "use_item_female",
    "Level (Attack > Defense)": "level_atk_gt_def",
    "Level (Attack = Defense)": "level_atk_eq_def",
    "Level (Attack < Defense)": "level_atk_lt_def",
    "Level (PID > 5)": "level_pid_low",
    "Level (PID < 5)": "level_pid_high",
    "Level (1 of 2)": "level_ninjask",
    "Level (2 of 2)": "level_shedinja",
    "Max Beauty": "level_beauty",
    "Held Item (Day)": "level_with_held_item_day",
    "Held Item (Night)": "level_with_held_item_night",
    "Attack Known": "level_know_move",
    "Pokemon in Party": "level_species_in_party",
    "Level (Male)": "level_male",
    "Level (Female)": "level_female",
    "Level (Mt. Coronet)": "level_magnetic_field",
    "Level (Eterna Forest)": "level_moss_rock",
    "Level (Route 217)": "level_ice_rock",
    "Use Item (Day)": "label: Use Item (Day)",
}

NUM_TMS = 92  # TM01-TM92, then HM01-HM08 (PKCalc tm_table keys 93-100)


def load_rom(rom_dir: Path) -> dict[str, Any]:
    """The ROM intermediates the rows read, keyed by file stem."""
    return {
        name: json.loads((Path(rom_dir) / f"{name}.json").read_text(encoding="utf-8"))
        for name in ROM_FILES
    }


def table_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """Every Task 12 row, in table order."""
    return [
        *move_rows(rom["moves"], pkcalc, docs),
        *species_rows(rom, pkcalc, docs),
        *learnset_rows(rom, pkcalc, docs),
        *tm_rows(rom, pkcalc, docs),
        *tutor_rows(rom, pkcalc, docs),
        *item_rows(rom, pkcalc, docs),
        *ability_rows(rom, pkcalc, docs),
    ]


# --- helpers -----------------------------------------------------------------------------


def _ids(*tables: Mapping | list | None) -> list[int]:
    out: set[int] = set()
    for t in tables:
        if t is None:
            continue
        out.update(range(len(t)) if isinstance(t, list) else (int(k) for k in t))
    out.discard(0)
    return sorted(out)


def _get(table: Mapping | list | None, i: int) -> Any:
    if table is None:
        return None
    if isinstance(table, list):
        return table[i] if 0 <= i < len(table) else None
    return table.get(str(i))


def _field(entry: Mapping | None, name: str) -> Any:
    return None if entry is None else entry.get(name)


def _key(i: int, *names: str | None) -> str:
    return f"{i} {next((n for n in names if n), '?')}"


def _name_index(names: Mapping[int, str], what: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, name in names.items():
        if name in out and out[name] != i:
            raise ValueError(f"{what} name {name!r} is both {out[name]} and {i}")
        out[name] = i
    return out


def _lookup(index: Mapping[str, int], name: str, what: str, where: str) -> int:
    try:
        return index[name]
    except KeyError:
        raise ValueError(f"{where}: unknown {what} {name!r}") from None


def _ref(index: Mapping[str, int], name: str) -> int | str:
    """A sheet name's id: exact, else case-insensitive (the sheet writes "Solarbeam",
    "U-Turn"); a name no id carries (a vanilla name of a renamed slot, e.g. "Taunt") stays
    as ``"name: <text>"``, so it is reported as a disagreement instead of guessed."""
    if name in index:
        return index[name]
    folded = {n.casefold(): i for n, i in index.items()}
    return folded.get(name.casefold(), f"name: {name}")


def _sheet_move_index(docs: dict) -> dict[str, int]:
    return _name_index(
        {int(k): m["name"] for k, m in docs.get("moves", {}).items() if int(k)}, "sheet move"
    )


# --- moves -------------------------------------------------------------------------------

_MOVE_FIELDS = (  # (row field, ROM field, PKCalc/sheet field)
    ("power", "power", "power"),
    ("accuracy", "accuracy", "accuracy"),
    ("pp", "pp", "pp"),
    ("priority", "priority", "priority"),
    ("type", "type", "type"),
    ("category", "category", "category"),
    ("chance", "effect_chance", "chance"),
)


def _target(pk: dict | None, key: str) -> str | None:
    if pk is None:
        return None
    try:
        return PKCALC_TARGETS[pk["target"]]
    except KeyError:
        raise ValueError(f"moves {key!r}: unmapped PKCalc target {pk['target']!r}") from None


def _check_priority(key: str, source: str, value: object) -> None:
    if value is not None and not PRIORITY_MIN <= value <= PRIORITY_MAX:
        raise ValueError(
            f"moves {key!r}: {source} priority {value} is outside "
            f"{PRIORITY_MIN}..{PRIORITY_MAX} (spec 5.2)"
        )


def _sheet_effect(sheet: dict | None) -> object:
    if sheet is None:
        return None
    if sheet["effect"] is None:
        return f"label: {sheet['label']}"
    return sheet["effect"]


def move_rows(rom_moves: list[dict], pkcalc: dict, docs: dict) -> list[Row]:
    """Executed: power, accuracy, pp, priority, type, category, target, chance and effect
    (ROM vs the sheet's label decode); unexecuted: name. Every id any source has but 0."""
    pk_moves, sheet_moves = pkcalc.get("moves", {}), docs.get("moves", {})
    rows: list[Row] = []
    for i in _ids(rom_moves, pk_moves, sheet_moves):
        rom, pk, sheet = _get(rom_moves, i), _get(pk_moves, i), _get(sheet_moves, i)
        key = _key(i, _field(pk, "name"), _field(rom, "name"), _field(sheet, "name"))
        for source, entry in (("ROM", rom), ("PKCalc", pk), ("sheet", sheet)):
            _check_priority(key, source, _field(entry, "priority"))
        for field, rf, pf in _MOVE_FIELDS:
            rows.append(
                Row("moves", key, field, _field(rom, rf), _field(pk, pf), _field(sheet, pf), True)
            )
        rows.append(
            Row(
                "moves",
                key,
                "target",
                _field(rom, "target"),
                _target(pk, key),
                _field(sheet, "target"),
                True,
            )
        )
        rows.append(
            Row("moves", key, "effect", _field(rom, "effect"), None, _sheet_effect(sheet), True)
        )
        rows.append(
            Row(
                "moves",
                key,
                "name",
                _field(rom, "name"),
                _field(pk, "name"),
                _field(sheet, "name"),
                False,
            )
        )
    return rows


# --- species -----------------------------------------------------------------------------


def _personal(rom: Mapping[str, Any]) -> dict[int, dict]:
    """Personal rows by index: base species, then forms by their ``row`` (Ruling 4)."""
    out = {i: s for i, s in enumerate(rom["species"])}
    for f in rom.get("forms", []):
        if f["row"] in out:
            raise ValueError(f"form {f['species']}/{f['form']} row {f['row']} is taken")
        out[f["row"]] = f
    return out


def _effective(abilities: list[int] | None) -> list[int] | None:
    if abilities is None:
        return None
    a1, a2 = abilities
    return [a1, a2 or a1]


def _sheet_abilities(sheet: dict | None, index: Mapping[str, int], where: str) -> object:
    if sheet is None:
        return None
    ids = [0 if n == "-" else _ref(index, n) for n in sheet["abilities"]]
    return _effective(ids)


def _ability_index(rom: Mapping[str, Any], pkcalc: dict) -> dict[str, int]:
    names: dict[str, int] = {}
    for i, a in enumerate(rom.get("abilities", [])):
        if i:
            names.setdefault(a["name"], i)
    for k, a in pkcalc.get("abilities", {}).items():
        for n in (a["name"], *a.get("aliases", [])):
            if names.setdefault(n, int(k)) != int(k):
                raise ValueError(f"ability name {n!r} is both {names[n]} and {k}")
    return names


def _upper(name: str | None) -> str | None:
    return None if name is None else name.upper()


def species_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """Per personal index, forms included (Ruling 4). Executed: base stats, types,
    abilities (effective pair), exp yield, growth rate, gender ratio; unexecuted: catch
    rate (the ROM intermediate has none), name."""
    personal = _personal(rom)
    pk_species, sheet_species = pkcalc.get("species", {}), docs.get("species", {})
    abilities = _ability_index(rom, pkcalc)
    rows: list[Row] = []
    for i in _ids(personal, pk_species, sheet_species):
        r, pk, sh = personal.get(i), _get(pk_species, i), _get(sheet_species, i)
        key = _key(i, _field(pk, "name"), _field(r, "name"), _field(sh, "name"))
        where = f"species {key!r}"
        growth = _field(sh, "growth_rate")
        if growth is not None:
            growth = _lookup(GROWTH_RATES, growth, "growth rate", where)

        def row(
            field: str, rv: object, pv: object, sv: object, executed: bool = True, key: str = key
        ) -> Row:
            return Row("species", key, field, rv, pv, sv, executed)

        rows += [
            row("base_stats", _field(r, "base_stats"), _field(pk, "base_stats"),
                _field(sh, "base_stats")),
            row("types", _field(r, "types"), _field(pk, "types"), _field(sh, "types")),
            row("abilities", _effective(_field(r, "abilities")),
                _effective(_field(pk, "abilities")), _sheet_abilities(sh, abilities, where)),
            row("exp_yield", _field(r, "exp_yield"), _field(pk, "exp_yield"),
                _field(sh, "exp_yield")),
            row("growth_rate", _field(r, "exp_rate"), _field(pk, "growth_rate"), growth),
            row("gender_ratio", _field(r, "gender_ratio"), _field(pk, "gender_ratio"),
                _field(sh, "gender_ratio")),
            row("catch_rate", None, _field(pk, "catch_rate"), _field(sh, "catch_rate"), False),
            row("name", _upper(_field(r, "name")), _upper(_field(pk, "name")),
                _upper(_field(sh, "name")), False),
        ]  # fmt: skip
    return rows


# --- learnsets ---------------------------------------------------------------------------


def _tm_labels(items: list[dict]) -> list[tuple[int, str]]:
    """``(item id, "TM01")`` for the TM/HM items in machine-index order: from ``TM01`` on
    while the names read TMnn/HMnn (``ProcTMLearnset`` indexes from ``ITEM_TM01``)."""
    first = next((i for i, it in enumerate(items) if it["name"] == "TM01"), None)
    if first is None:
        raise ValueError("ROM items.json has no TM01")
    out = []
    for i in range(first, len(items)):
        name = items[i]["name"]
        if not re.fullmatch(r"[TH]M\d\d", name):
            break
        if items[i].get("teaches_move") is None:
            raise ValueError(f"ROM item {i} {name} teaches no move")
        out.append((i, name))
    return out


def _pk_tm_key(label: str) -> str:
    n = int(label[2:])
    return str(n if label.startswith("TM") else NUM_TMS + n)


def _sheet_evolutions(
    sheet: dict | None, names: Mapping[str, Mapping[str, int]], where: str
) -> object:
    if sheet is None:
        return None
    species, items, sheet_moves = names["species"], names["items"], names["moves"]
    out = []
    for e in sheet["evolutions"]:
        method = _lookup(EVOLUTION_LABELS, e["method"], "evolution method", where)
        req = e["required"]
        if req is None:
            param = 0
        elif isinstance(req, int):
            param = req
        elif method == "level_know_move":
            param = _ref(sheet_moves, req)
        elif method == "level_species_in_party":
            param = _ref(species, req.upper())
        else:
            param = _ref(items, req)
        out.append([method, param, _ref(species, e["result"].upper())])
    return out


def _pk_machines(
    moves: list[int], pk_machine: Mapping[int, str], order: Mapping[str, int], where: str
) -> list[str]:
    labels = [_lookup(pk_machine, m, "PKCalc TM move", where) for m in moves]
    return sorted(labels, key=order.__getitem__)


def learnset_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """Per personal index: executed ``level_up`` ([[level, move]] in learn order), ``tm``
    and ``tutor`` (sorted move ids); ``evolutions`` ([[method name, param, species]], ROM
    vs sheet, base species only: the ROM keys evolutions by species id)."""
    pk_species = pkcalc.get("species", {})
    sheet_moves = _sheet_move_index(docs)
    machines = [name for _, name in _tm_labels(rom["items"])]
    order = {name: n for n, name in enumerate(machines)}
    pk_machine = {
        move: label
        for label in machines
        if (move := pkcalc.get("tm_table", {}).get(_pk_tm_key(label))) is not None
    }
    tutor_moves = [t["move"] for t in rom["tutor_moves"]]
    methods = rom["evolution_methods"]
    personal = _personal(rom)
    names = {
        "moves": sheet_moves,
        "items": _sheet_item_ids(docs, rom["items"]),
        "species": _name_index(
            {
                int(k): s["name"].upper()
                for k, s in docs.get("species", {}).items()
                if 0 < int(k) < len(rom["species"])
            },
            "sheet species",
        ),
    }
    rows: list[Row] = []
    tables = (rom["learnsets"], rom["tm_learnsets"], rom["tutor_learnsets"], pk_species)
    for i in _ids(*tables):
        pk = _get(pk_species, i)
        sheet_name = _field(_get(docs.get("level_up"), i), "name")
        key = _key(i, _field(pk, "name"), _field(personal.get(i), "name"), sheet_name)
        where = f"learnsets {key!r}"

        def row(
            field: str, rv: object, pv: object, sv: object, executed: bool = True, key: str = key
        ) -> Row:
            return Row("learnsets", key, field, rv, pv, sv, executed)

        lv = _get(rom["learnsets"], i)
        sheet_lv = _get(docs.get("level_up"), i)
        if sheet_lv is not None:
            sheet_lv = [[level, _ref(sheet_moves, m)] for m, level in sheet_lv["moves"]]
        rows.append(
            row(
                "level_up",
                None if lv is None else [list(p) for p in lv],
                None if pk is None else [list(p) for p in pk["level_up"]],
                sheet_lv,
            )
        )
        tm = _get(rom["tm_learnsets"], i)
        sheet_tm = _get(docs.get("tm"), i)
        if sheet_tm is not None:
            sheet_tm = sorted(
                sheet_tm["machines"],
                key=lambda m: _lookup(order, m, "machine", where),
            )
        rows.append(
            row(
                "tm",
                None if tm is None else [machines[x] for x in sorted(tm)],
                None if pk is None else _pk_machines(pk["tm"], pk_machine, order, where),
                sheet_tm,
            )
        )
        tutor = _get(rom["tutor_learnsets"], i)
        rows.append(
            row(
                "tutor",
                None if tutor is None else sorted(tutor_moves[x] for x in tutor),
                None if pk is None else sorted(pk["tutor"]),
                None,
            )
        )
        if i < len(rom["species"]):
            evo = _get(rom["evolutions"], i)
            rom_evo = None if evo is None else [[methods[m], p, t] for m, p, t in evo]
            sheet_evo = _sheet_evolutions(_get(docs.get("evolutions"), i), names, where)
            rows.append(row("evolutions", rom_evo, None, sheet_evo, False))
    return rows


# --- TM and tutor tables ----------------------------------------------------------------


def tm_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """``tm_table``, key ``"TM07"``: the move the machine teaches, executed; ROM
    (``items.json`` ``teaches_move``) vs PKCalc ``tm_table`` (from ``machine``) vs the
    sheet's ``TM Learnsets`` row 2."""
    sheet_moves = _sheet_move_index(docs)
    pk_table = pkcalc.get("tm_table", {})
    sheet_table = docs.get("tm_moves", {})
    labels = [name for _, name in _tm_labels(rom["items"])]
    extra = sorted(set(sheet_table) - set(labels))
    if extra:
        raise ValueError(f"sheet TM labels with no ROM machine: {extra}")
    rows = []
    for i, label in _tm_labels(rom["items"]):
        sheet = sheet_table.get(label)
        sv = None if sheet is None else _ref(sheet_moves, sheet["move"])
        pv = pk_table.get(_pk_tm_key(label))
        rows.append(Row("tm_table", label, "move", rom["items"][i]["teaches_move"], pv, sv, True))
    known = {_pk_tm_key(label) for label in labels}
    stray = sorted(set(pk_table) - known, key=int)
    if stray:
        raise ValueError(f"PKCalc tm_table keys with no ROM machine: {stray}")
    return rows


def _location(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _by_move(table: list[dict], source: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for t in table:
        if t["move"] in out:
            raise ValueError(f"{source} tutor table lists move {t['move']} twice")
        out[t["move"]] = t
    return out


def tutor_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """``tutor_table``, keyed by move (preflight #32): ``move`` membership, executed;
    ``cost`` (shard dict) and ``location`` for moves in both tables, unexecuted."""
    rom_t = _by_move(rom["tutor_moves"], "ROM")
    pk_t = _by_move(pkcalc.get("tutor_table", []), "PKCalc")
    pk_moves = pkcalc.get("moves", {})
    rows = []
    for move in sorted(set(rom_t) | set(pk_t)):
        name = _field(_get(pk_moves, move), "name") or _field(_get(rom.get("moves"), move), "name")
        key = _key(move, name)
        r, p = rom_t.get(move), pk_t.get(move)
        rows.append(Row("tutor_table", key, "move", r is not None, p is not None, None, True))
        if r is None or p is None:
            continue
        rows.append(Row("tutor_table", key, "cost", dict(r["costs"]), dict(p["cost"]), None, False))
        rows.append(
            Row(
                "tutor_table",
                key,
                "location",
                _location(r["location"]),
                _location(p["location"]),
                None,
                False,
            )
        )
    return rows


# --- items and abilities ----------------------------------------------------------------


def _sheet_item_ids(docs: dict, rom_items: list[dict]) -> dict[str, int]:
    """Sheet item name -> item id. ``sheet_id`` counts the items without the
    ``ITEM_UNUSED_*`` rows (113-134), so it is unpacked through the ROM's used ids and
    checked against the ROM names (a renamed slot, e.g. Nugget -> Big Nugget, joins by id)."""
    items = docs.get("items", {})
    used = [i for i, it in enumerate(rom_items) if not it["name"].startswith("ITEM_UNUSED_")]
    rom_ids = _name_index({i: rom_items[i]["name"] for i in used}, "ROM item")
    out = {}
    for name, v in items.items():
        if not 0 <= v["sheet_id"] < len(used):
            raise ValueError(f"sheet item {name!r}: sheet_id {v['sheet_id']} is out of range")
        i = used[v["sheet_id"]]
        if name in rom_ids and rom_ids[name] != i:
            raise ValueError(
                f"sheet item {name!r}: sheet_id {v['sheet_id']} unpacks to {i}, "
                f"but the ROM names it at {rom_ids[name]}"
            )
        out[name] = i
    return out


def item_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """Executed hold effect, hold param and fling power (ROM vs sheet; PKCalc has no
    numeric hold fields); unexecuted name. ``ITEM_UNUSED_*`` ids are skipped."""
    rom_items = rom["items"]
    sheet_ids = _sheet_item_ids(docs, rom_items)
    sheet_by_id = {i: (n, docs["items"][n]) for n, i in sheet_ids.items()}
    if len(sheet_by_id) != len(sheet_ids):
        raise ValueError("two sheet items unpack to the same item id")
    pk_items = pkcalc.get("items", {})
    rows = []
    for i in _ids(rom_items, pk_items, {str(k): v for k, v in sheet_by_id.items()}):
        r, pk = _get(rom_items, i), _get(pk_items, i)
        sheet_name, sh = sheet_by_id.get(i, (None, None))
        if r is not None and r["name"].startswith("ITEM_UNUSED_") and pk is None and sh is None:
            continue
        key = _key(i, _field(pk, "name"), _field(r, "name"), sheet_name)
        rows += [
            Row("items", key, "hold_effect", _field(r, "hold_effect"), None,
                _field(sh, "hold_effect"), True),
            Row("items", key, "hold_param", _field(r, "hold_param"), None,
                _field(sh, "hold_param"), True),
            Row("items", key, "fling_power", _field(r, "fling_power"), None,
                _field(sh, "fling_power"), True),
            Row("items", key, "name", _field(r, "name"), _field(pk, "name"), sheet_name, False),
        ]  # fmt: skip
    return rows


def ability_rows(rom: Mapping[str, Any], pkcalc: dict, docs: dict) -> list[Row]:
    """Unexecuted ``name`` per ability id (ROM vs PKCalc; the sheet has no ability table)."""
    rom_ab, pk_ab = rom["abilities"], pkcalc.get("abilities", {})
    rows = []
    for i in _ids(rom_ab, pk_ab):
        r, pk = _get(rom_ab, i), _get(pk_ab, i)
        key = _key(i, _field(pk, "name"), _field(r, "name"))
        rows.append(
            Row("abilities", key, "name", _field(r, "name"), _field(pk, "name"), None, False)
        )
    return rows
