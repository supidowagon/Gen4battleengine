"""Text and JSON renderings of a :class:`SaveFile` for the ``nuzline save`` commands
(spec section 7). ``names`` is the manifest's :class:`nuzline.data.tables.Names`; the
TOML written here is what ``nuzline.data.party.load_party`` reads back."""

from __future__ import annotations

import dataclasses
import json

from nuzline.saves.pkm import SaveMon
from nuzline.saves.save import SaveFile, default_tables_and_names
from nuzline.saves.status import decode_status

__all__ = ["party_toml", "save_json", "show_lines"]

_GENDERS = ("male", "female", "none")


def _ascii(text: str) -> str:
    """``text`` with every non-ASCII character escaped (a default Windows console is cp1252)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _move_name(names, move: int) -> str:
    return "" if move == 0 else names.move_names[move]


def _item_name(names, item: int) -> str:
    return "" if item == 0 else names.item_names[item]


def _status_name(mon: SaveMon) -> str:
    return decode_status(mon.status_raw)[0].name.lower()


def party_toml(save: SaveFile, names, tables=None) -> str:
    """The party (eggs dropped) in the ``--party`` TOML format, with ``hp``, ``pp``,
    ``sleep_turns`` and ``stats``. ``tables`` (default: the built tables) feed
    ``SaveMon.to_spec``'s checks."""
    tables, names = default_tables_and_names(tables, names)
    out = ["# Exported by `nuzline save export`; eggs are left out.", ""]
    for mon in save.party:
        if mon.is_egg:
            continue
        spec = mon.to_spec(tables)
        listed = max([i + 1 for i, m in enumerate(spec.moves) if m] or [1])
        ability = names.species_abilities(spec.species)[spec.ability_slot]
        lines = [
            "[[pokemon]]",
            f"species = {json.dumps(names.species_names[spec.species])}",
            f"level = {spec.level}",
            f"nature = {json.dumps(names.nature_names[spec.nature])}",
            f"ivs = {list(spec.ivs)}",
            f"evs = {list(spec.evs)}",
            f"moves = {json.dumps([_move_name(names, m) for m in spec.moves[:listed]])}",
            f"item = {json.dumps(_item_name(names, spec.item))}",
            f"ability = {json.dumps(names.ability_names[ability])}",
            f"gender = {json.dumps(_GENDERS[spec.gender])}",
            f"friendship = {spec.friendship}",
        ]
        if spec.current_hp is not None:
            lines.append(f"hp = {spec.current_hp}")
        if spec.pp is not None:
            lines.append(f"pp = {list(spec.pp[:listed])}")
        if spec.status:
            lines.append(f"status = {json.dumps(_status_name(mon))}")
            if _status_name(mon) == "sleep":
                lines.append(f"sleep_turns = {spec.status_counter}")
        if spec.stats is not None:
            lines.append(f"stats = {list(spec.stats)}")
        out += [*lines, ""]
    return "\n".join(out)


def _mon_json(mon: SaveMon, names) -> dict:
    d = dataclasses.asdict(mon)
    d["species_name"] = names.species_names[mon.species]
    d["nature_name"] = names.nature_names[mon.nature]
    d["ability_name"] = names.ability_names[mon.ability]
    d["item_name"] = _item_name(names, mon.item)
    d["move_names"] = [_move_name(names, m) for m in mon.moves]
    d["status"] = _status_name(mon)
    return d


def save_json(save: SaveFile, names) -> dict:
    """The whole save as JSON-ready data: decomp ids plus English names."""
    d = dataclasses.asdict(save)
    d["party"] = [_mon_json(m, names) for m in save.party]
    d["boxes"] = [[None if m is None else _mon_json(m, names) for m in box] for box in save.boxes]
    d["bag"] = {
        pocket: [
            {"item": item, "item_name": _item_name(names, item), "quantity": qty}
            for item, qty in entries
        ]
        for pocket, entries in save.bag.items()
    }
    return d


def _mon_line(mon: SaveMon, names) -> str:
    if mon.is_egg:
        return "egg"
    moves = ", ".join(f"{names.move_names[m]} {mon.pp[i]}PP" for i, m in enumerate(mon.moves) if m)
    if mon.stored_stats is not None:
        hp = f"HP {mon.current_hp}/{mon.stored_stats[0]}"
    else:
        hp = "HP -"
    status = _status_name(mon)
    parts = [
        f"{names.species_names[mon.species]} Lv{mon.level}",
        names.nature_names[mon.nature],
        names.ability_names[mon.ability],
        f"item {_item_name(names, mon.item) or '-'}",
        hp,
        *([status] if status != "none" else []),
        f"[{moves}]",
    ]
    return " | ".join(parts)


def show_lines(save: SaveFile, names) -> list[str]:
    """Human-readable summary, ASCII only: trainer, loaded copies and counter, the party
    and each non-empty box, one line per Pokemon."""
    lines = [
        f"Trainer: {save.trainer_name} (TID {save.tid}, SID {save.sid}, badges {save.badges:#04x})",
        (
            f"copy: {save.normal_copy} (boxes: {save.boxes_copy}), "
            f"save counter {save.save_counter}, load result {save.load_result}"
        ),
        f"Party ({len(save.party)}):",
    ]
    for slot, mon in enumerate(save.party):
        lines.append(f"  {slot}. {_mon_line(mon, names)}")
    for b, box in enumerate(save.boxes):
        filled = [(s, m) for s, m in enumerate(box) if m is not None]
        if not filled:
            continue
        lines.append(f"Box {b + 1} ({len(filled)}):")
        for s, mon in filled:
            lines.append(f"  {s}. {_mon_line(mon, names)}")
    return [_ascii(line) for line in lines]
