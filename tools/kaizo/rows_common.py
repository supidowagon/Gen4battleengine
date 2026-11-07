"""Helpers shared by the Kaizo cross-check row builders (``rows_tables``, Task 12;
``rows_trainers``, Task 13): id-or-name table access and the sheet's name indexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def get(table: Mapping | list | None, i: int) -> Any:
    if table is None:
        return None
    if isinstance(table, list):
        return table[i] if 0 <= i < len(table) else None
    return table.get(str(i))


class Names(dict):
    """Name -> id, with ``folded`` (case-folded name -> id) built alongside."""

    folded: dict[str, int]


def name_index(names: Iterable[tuple[int, str]], what: str) -> Names:
    """Name -> id from ``(id, name)`` pairs; a name (or two names that differ only by case)
    on two ids raises."""
    out = Names()
    out.folded = {}
    for i, name in names:
        for index, n in ((out, name), (out.folded, name.casefold())):
            if index.setdefault(n, i) != i:
                raise ValueError(f"{what} name {name!r} is both {index[n]} and {i}")
    return out


def ref(index: Names, name: str) -> int | str:
    """A sheet name's id: exact, else case-insensitive (the sheet writes "Solarbeam",
    "U-Turn"); a name no id carries (a vanilla name of a renamed slot, e.g. "Taunt") stays
    as ``"name: <text>"``, so it is reported as a disagreement instead of guessed."""
    if name in index:
        return index[name]
    return index.folded.get(name.casefold(), f"name: {name}")


def sheet_move_index(docs: dict) -> dict[str, int]:
    return name_index(
        ((int(k), m["name"]) for k, m in docs.get("moves", {}).items() if int(k)), "sheet move"
    )


def sheet_item_ids(docs: dict, rom_items: list[dict]) -> dict[str, int]:
    """Sheet item name -> item id. ``sheet_id`` counts the items without the
    ``ITEM_UNUSED_*`` rows (113-134), so it is unpacked through the ROM's used ids and
    checked against the ROM names (a renamed slot, e.g. Nugget -> Big Nugget, joins by id)."""
    items = docs.get("items", {})
    used = [i for i, it in enumerate(rom_items) if not it["name"].startswith("ITEM_UNUSED_")]
    rom_ids = name_index(((i, rom_items[i]["name"]) for i in used), "ROM item")
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
