"""Pack the intermediate decomp JSON into ``tables.npz`` + ``manifest.json``.

Every array is indexed by the decomp's own ids (species, move, item, ability,
type, nature). Effects, abilities and items are mapped to engine ids through
``tools/effect_map.py``; anything it does not list becomes ``UNSUPPORTED``.
Every value is range-checked against its dtype, so nothing wraps silently.

Encodings:

* ``move_category``: decomp class id (``generated/move_classes.txt``), see
  ``manifest["category_names"]``.
* ``move_target``: index into ``MOVE_TARGETS`` / ``manifest["target_names"]``
  (the decomp stores ``RANGE_*`` as a bit mask, kept as names in the JSON).
* ``type_chart[attacker, defender]``: multiplier in tenths (0, 5, 10, 20).
* ``type_chart_order[attacker, defender]``: row index of the pair in the decomp's
  ``sTypeMatchupMultipliers``, -1 when unlisted (neutral); a dual type's two
  multipliers apply in row order (``BattleSystem_ApplyTypeChart``).
* ``nature_mods[nature, stat]``: +1 raised, -1 lowered, 0 otherwise (HP always 0).
* ``item_param[item]``: ``(decomp hold param, 0)`` for supported items, else zeros.
* ``trainer_item_effect[item]``: ``ids.TrainerItemId`` of the item used from a trainer's
  bag (``DECOMP_TRAINER_ITEM_TO_ID``).
* ``species_friendship`` is int16, not int8: base friendship reaches 140.

CLI: ``uv run python -m tools.build_tables --inter data/intermediate --out data/gen4_platinum``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, MoveTarget, TrainerItemId
from tools.effect_map import (
    CERTAIN_SECONDARY_EFFECTS,
    DECOMP_ABILITY_TO_ID,
    DECOMP_EFFECT_TO_ID,
    DECOMP_ITEM_TO_ID,
    DECOMP_TRAINER_ITEM_TO_ID,
    PUNCH_MOVES,
    SOUND_MOVES,
    TYPE_CHART_STATUS_MOVES,
    UNSUPPORTED_MOVES,
)

MANIFEST_VERSION = 4
CLONE_SCRIPT = Path(__file__).with_name("clone_decomp.sh")

# generated/move_classes.txt order.
MOVE_CATEGORIES = ("PHYSICAL", "SPECIAL", "STATUS")
# ``MoveTarget`` order (generated/move_ranges.txt without unused ranges).
MOVE_TARGETS = tuple(t.name for t in MoveTarget)
DECOMP_FLAG_BITS = {
    "MOVE_FLAG_MAKES_CONTACT": MoveFlag.CONTACT,
    "MOVE_FLAG_CAN_PROTECT": MoveFlag.PROTECT,
}
SECONDARY = (EffectId.SECONDARY_STAT_SELF, EffectId.SECONDARY_STAT_FOE, EffectId.SECONDARY_STATUS)


class BuildError(RuntimeError):
    """Raised when the intermediate data cannot be packed exactly."""


def _arr(values, dtype, what: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.int64)
    info = np.iinfo(dtype)
    if a.size and (a.min() < info.min or a.max() > info.max):
        raise BuildError(f"{what}: values {a.min()}..{a.max()} do not fit {np.dtype(dtype)}")
    return a.astype(dtype)


def _load(inter: Path, name: str):
    return json.loads((inter / f"{name}.json").read_text(encoding="utf-8"))


def _decomp_commit() -> str:
    m = re.search(r'^PINNED_COMMIT="([0-9a-f]{40})"', CLONE_SCRIPT.read_text(), re.MULTILINE)
    if not m:
        raise BuildError(f"{CLONE_SCRIPT}: PINNED_COMMIT not found")
    return m.group(1)


def _move_flags(move: dict) -> int:
    bits = 0
    for flag in move["flags"]:
        bits |= DECOMP_FLAG_BITS.get(flag, 0)
    if move["name"] in SOUND_MOVES:
        bits |= MoveFlag.SOUND
    if move["name"] in PUNCH_MOVES:
        bits |= MoveFlag.PUNCH
    return int(bits)


def _move_effect(move: dict) -> tuple[int, tuple[int, int, int]]:
    if move["name"] in UNSUPPORTED_MOVES:
        return int(EffectId.UNSUPPORTED), (0, 0, 0)
    eid, params = DECOMP_EFFECT_TO_ID.get(move["effect"], (EffectId.UNSUPPORTED, (0, 0, 0)))
    if eid in SECONDARY:
        certain = move["effect"] in CERTAIN_SECONDARY_EFFECTS
        if certain != (move["effect_chance"] == 0):
            raise BuildError(
                f"move {move['name']!r}: effect {move['effect']} certain={certain} "
                f"but effect_chance={move['effect_chance']}"
            )
    if move["name"] in TYPE_CHART_STATUS_MOVES:
        if eid != EffectId.STATUS_FOE:
            raise BuildError(f"move {move['name']!r}: type-checked but not STATUS_FOE")
        params = (params[0], 1, params[2])
    return int(eid), params


def build_arrays(inter: Path) -> tuple[dict[str, np.ndarray], dict]:
    species = _load(inter, "species")
    moves = _load(inter, "moves")
    items = _load(inter, "items")
    abilities = _load(inter, "abilities")
    types = _load(inter, "types")
    natures = _load(inter, "natures")

    move_names = {m["name"] for m in moves}
    missing = sorted(
        (SOUND_MOVES | PUNCH_MOVES | UNSUPPORTED_MOVES | TYPE_CHART_STATUS_MOVES) - move_names
    )
    if missing:
        raise BuildError(f"effect_map moves not in moves.json: {missing}")
    ability_names = [a["name"] for a in abilities]
    ability_ids = [int(DECOMP_ABILITY_TO_ID.get(n, AbilityId.UNSUPPORTED)) for n in ability_names]
    unknown = sorted(set(DECOMP_ABILITY_TO_ID) - set(ability_names))
    unknown += sorted(
        (set(DECOMP_ITEM_TO_ID) | set(DECOMP_TRAINER_ITEM_TO_ID)) - {i["name"] for i in items}
    )
    if unknown:
        raise BuildError(f"effect_map names absent from the data: {unknown}")
    for m in moves:
        if m["target"] not in MOVE_TARGETS:
            raise BuildError(f"move {m['name']!r}: unknown target {m['target']!r}")
        if not 0 <= m["category"] < len(MOVE_CATEGORIES):
            raise BuildError(f"move {m['name']!r}: unknown category {m['category']}")

    n_types = len(types["names"])
    chart, chart_order = types["chart"], types["chart_order"]
    for what, grid in (("type chart", chart), ("type chart order", chart_order)):
        if len(grid) != n_types or any(len(row) != n_types for row in grid):
            raise BuildError(f"{what} is not {n_types}x{n_types}")

    nature_mods = np.zeros((len(natures), 6), dtype=np.int64)
    for i, n in enumerate(natures):
        if n["up"] is not None:
            nature_mods[i, n["up"]] += 1
            nature_mods[i, n["down"]] -= 1

    effects = [_move_effect(m) for m in moves]
    item_ids = [int(DECOMP_ITEM_TO_ID.get(i["name"], ItemId.UNSUPPORTED)) for i in items]
    trainer_item_ids = [
        int(DECOMP_TRAINER_ITEM_TO_ID.get(i["name"], TrainerItemId.UNSUPPORTED)) for i in items
    ]
    supported_item = [e not in (ItemId.UNSUPPORTED, ItemId.NONE) for e in item_ids]

    arrays = {
        "species_base_stats": _arr([s["base_stats"] for s in species], np.int16, "base_stats"),
        "species_types": _arr([s["types"] for s in species], np.int8, "types"),
        "species_abilities": _arr(
            [[ability_ids[a] for a in s["abilities"]] for s in species], np.int16, "abilities"
        ),
        "species_weight": _arr([s["weight_hg"] for s in species], np.int32, "weight"),
        "species_gender": _arr([s["gender_ratio"] for s in species], np.int16, "gender"),
        "species_friendship": _arr([s["base_friendship"] for s in species], np.int16, "friendship"),
        "move_power": _arr([m["power"] for m in moves], np.int16, "power"),
        "move_type": _arr([m["type"] for m in moves], np.int8, "move type"),
        "move_category": _arr([m["category"] for m in moves], np.int8, "category"),
        "move_accuracy": _arr([m["accuracy"] for m in moves], np.int8, "accuracy"),
        "move_pp": _arr([m["pp"] for m in moves], np.int8, "pp"),
        "move_priority": _arr([m["priority"] for m in moves], np.int8, "priority"),
        "move_target": _arr([MOVE_TARGETS.index(m["target"]) for m in moves], np.int8, "target"),
        "move_effect": _arr([e for e, _ in effects], np.int16, "effect"),
        "move_effect_chance": _arr([m["effect_chance"] for m in moves], np.int8, "chance"),
        "move_param": _arr([p for _, p in effects], np.int16, "move param"),
        "move_flags": _arr([_move_flags(m) for m in moves], np.int16, "move flags"),
        "type_chart": _arr(chart, np.int8, "type chart"),
        "type_chart_order": _arr(chart_order, np.int16, "type chart order"),
        "nature_mods": _arr(nature_mods, np.int8, "nature mods"),
        "item_effect": _arr(item_ids, np.int16, "item effect"),
        "item_param": _arr(
            [
                [i["hold_param"] if ok else 0, 0]
                for i, ok in zip(items, supported_item, strict=True)
            ],
            np.int16,
            "item param",
        ),
        "trainer_item_effect": _arr(trainer_item_ids, np.int8, "trainer item effect"),
    }
    manifest = {
        "version": MANIFEST_VERSION,
        "decomp_commit": _decomp_commit(),
        "species_names": [s["name"] for s in species],
        # Decomp ability ids (not engine ids) per species slot, so a party file's
        # ability name resolves to its slot; and base friendship for its default.
        "species_abilities": [list(s["abilities"]) for s in species],
        "species_base_friendship": [s["base_friendship"] for s in species],
        "move_names": [m["name"] for m in moves],
        "item_names": [i["name"] for i in items],
        "ability_names": ability_names,
        "type_names": list(types["names"]),
        "nature_names": [n["name"] for n in natures],
        "category_names": list(MOVE_CATEGORIES),
        "target_names": list(MOVE_TARGETS),
    }
    return arrays, manifest


def build(intermediate_dir: Path, out_dir: Path) -> None:
    arrays, manifest = build_arrays(Path(intermediate_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "tables.npz", **arrays)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inter", type=Path, default=Path("data/intermediate"))
    p.add_argument("--out", type=Path, default=Path("data/gen4_platinum"))
    args = p.parse_args(argv)
    build(args.inter, args.out)


if __name__ == "__main__":
    main()
