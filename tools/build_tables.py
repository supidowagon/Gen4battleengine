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
* ``type_chart_foresight[attacker, defender]``: 1 for the pairs listed after the 0xFE
  separator (the Ghost immunities), which a Foresight-identified defender ignores
  (``BattleSystem_ApplyTypeChart``, battle_lib.c:2611).
* ``nature_mods[nature, stat]``: +1 raised, -1 lowered, 0 otherwise (HP always 0).
* ``item_param[item]``: ``(decomp hold param, 0)`` for supported items, else zeros.
* ``trainer_item_effect[item]``: ``ids.TrainerItemId`` of the item used from a trainer's
  bag (``DECOMP_TRAINER_ITEM_TO_ID``).
* ``species_exp_rate[species]``: growth rate id (``generated/exp_rates.txt`` order: 0 medium
  fast, 1 erratic, 2 fluctuating, 3 medium slow, 4 fast, 5 slow); ``exp_table[rate, level]``
  is the experience for ``level`` 0..100 (``exp_tables.csv``, same rate order).
* ``species_exp_yield[species]``: ``SpeciesData.baseExpReward`` (int16).
* ``move_ai_effect[move]``: the decomp battle effect id as stored (``MoveTable.effect``),
  unmapped, which is what the AI scripts read; ``move_effect`` is the engine's mapped id.
* A ``SECONDARY_*`` move that is not in ``CERTAIN_SECONDARY_EFFECTS`` and has
  ``effect_chance`` 0 never fires its proc (K2b Ruling 21), so ``move_effect`` packs it as
  ``PLAIN`` with params ``(0, 0, 0)`` when the mapped params carry nothing beyond the
  secondary (``params[2]``, e.g. effect 125's ``thaws_user``, is 0), else ``UNSUPPORTED``
  (``move_effect_chance`` stays 0, ``move_ai_effect`` keeps the id); the Kaizo manifest lists
  those moves as ``no_secondary_moves``. A certain effect with a nonzero chance is a
  ``BuildError``.
* Every ``species_*`` array has one row per personal row: species ids ``0..495``, then
  the data forms ``496..507`` (``forms.json``, Ruling 6). ``species_form_row[species,
  form]`` (int16[496, 28]) is the personal row of a form (``form_rows.json``; the species
  itself for a form without data, -1 beyond the species' form count), the same grid as
  ``manifest["species_form_row"]``. ``manifest["species_names"]`` stays per species id.
* ``manifest["charmap"]``: text code (hex string) -> character, for decoding nicknames.
* ``species_friendship`` is int16, not int8: base friendship reaches 140.
* ``species_abilities_decomp[species, slot]``: the decomp's own ability ids (0 =
  ``ABILITY_NONE``), next to ``species_abilities``' engine ids, so the trainer AI can
  name abilities the engine does not implement.

Kaizo (``--game kaizo``, K2b Task 15): ``tools.kaizo.resolve.resolve`` first cross-checks
the sources and writes the resolved ``data/intermediate_kaizo/*.json`` (and
``docs/kaizo/discrepancies.md``; exit 1 on an unresolved or stale row), then the same packing
runs with three Kaizo rules: an item maps to its ``ItemId``/``TrainerItemId`` only when the
slot keeps its vanilla name (up to spelling) and vanilla's hold effect, else ``UNSUPPORTED``
(Shoal Salt -> Berserk Gene, Lava Cookie as a Lum clone); a move whose battle effect script
(``be_seq`` member = effect id) changed is ``UNSUPPORTED`` unless ``rom_diff.toml`` marks the
member ``modelled`` (Ruling 13; manifest ``changed_effects``); and the manifest gains the
provenance hashes, ``aliases``, ``renamed``, ``no_secondary_moves`` and
``unclassified_rom_changes`` (the ``unknown`` ``[[code]]`` and ``[[member]]`` entries).

CLI: ``uv run python -m tools.build_tables [--game vanilla|kaizo]`` (default vanilla: reads
``data/intermediate``, writes ``data/gen4_platinum``)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from nuzline.data.games import GAMES, KAIZO, VANILLA, get_game
from nuzline.data.tables import MANIFEST_VERSION
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, MoveTarget, TrainerItemId
from tools.effect_map import (
    CERTAIN_SECONDARY_EFFECTS,
    DECOMP_ABILITY_TO_ID,
    DECOMP_EFFECT_TO_ID,
    DECOMP_ITEM_TO_ID,
    DECOMP_TRAINER_ITEM_TO_ID,
    TYPE_CHART_STATUS_MOVES,
    UNSUPPORTED_MOVES,
)

NUM_TMS = 92
NUM_HMS = 8
CLONE_SCRIPT = Path(__file__).with_name("clone_decomp.sh")
ROM_DIFF = Path(__file__).resolve().parents[1] / "data" / "kaizo" / "rom_diff.toml"
BE_SEQ = "battle/skill/be_seq.narc"  # member index = battle effect id (Ruling 13)

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


def _move_flags(move: dict, move_id: int, lists: dict[str, list[int]]) -> int:
    """The move's flag bits; SOUND and PUNCH come from ``move_lists.json`` by move id."""
    bits = 0
    for flag in move["flags"]:
        bits |= DECOMP_FLAG_BITS.get(flag, 0)
    if move_id in lists["sound"]:
        bits |= MoveFlag.SOUND
    if move_id in lists["punch"]:
        bits |= MoveFlag.PUNCH
    return int(bits)


def _no_secondary(move: dict) -> bool:
    """A mapped ``SECONDARY_*`` effect outside ``CERTAIN_SECONDARY_EFFECTS`` with chance 0:
    its proc never fires (Ruling 21)."""
    eid, _ = DECOMP_EFFECT_TO_ID.get(move["effect"], (EffectId.UNSUPPORTED, (0, 0, 0)))
    return (
        eid in SECONDARY
        and move["effect"] not in CERTAIN_SECONDARY_EFFECTS
        and move["effect_chance"] == 0
    )


def _move_effect(
    move: dict, changed_effects: frozenset[int] = frozenset()
) -> tuple[int, tuple[int, int, int]]:
    if move["name"] in UNSUPPORTED_MOVES or move["effect"] in changed_effects:
        return int(EffectId.UNSUPPORTED), (0, 0, 0)
    eid, params = DECOMP_EFFECT_TO_ID.get(move["effect"], (EffectId.UNSUPPORTED, (0, 0, 0)))
    if eid in SECONDARY:
        certain = move["effect"] in CERTAIN_SECONDARY_EFFECTS
        if certain and move["effect_chance"] != 0:
            raise BuildError(
                f"move {move['name']!r}: effect {move['effect']} is certain "
                f"but effect_chance={move['effect_chance']}"
            )
        if not certain and move["effect_chance"] == 0:
            # The proc never fires; params beyond the secondary (effect 125's thaws_user)
            # would still act, which PLAIN cannot say.
            if params[2] != 0:
                return int(EffectId.UNSUPPORTED), (0, 0, 0)
            return int(EffectId.PLAIN), (0, 0, 0)
    if move["name"] in TYPE_CHART_STATUS_MOVES:
        if eid != EffectId.STATUS_FOE:
            raise BuildError(f"move {move['name']!r}: type-checked but not STATUS_FOE")
        params = (params[0], 1, params[2])
    return int(eid), params


def _personal_rows(
    species: list[dict], forms: list[dict], form_rows: list[list[int]]
) -> list[dict]:
    """The personal rows: species ``0..S-1``, then the data forms by ``row`` (Ruling 6),
    checked against the ``form_rows`` grid so a row is never misplaced."""
    rows = list(species)
    for i, f in enumerate(forms):
        if f["row"] != len(species) + i:
            raise BuildError(f"forms.json entry {i}: row {f['row']}, expected {len(species) + i}")
        if not 0 <= f["species"] < len(species) or not 0 <= f["form"] < len(form_rows[0]):
            raise BuildError(f"forms.json entry {i}: species {f['species']} form {f['form']}")
        if form_rows[f["species"]][f["form"]] != f["row"]:
            raise BuildError(
                f"form_rows.json [{f['species']}][{f['form']}] is "
                f"{form_rows[f['species']][f['form']]}, forms.json says row {f['row']}"
            )
        rows.append({k: v for k, v in f.items() if k not in ("species", "form", "row")})
    if len(form_rows) != len(species):
        raise BuildError(f"form_rows.json has {len(form_rows)} species, expected {len(species)}")
    for sid, grid_row in enumerate(form_rows):
        for form, row in enumerate(grid_row):
            if not (row == -1 or row == sid or len(species) <= row < len(rows)):
                raise BuildError(f"form_rows.json [{sid}][{form}] = {row} is no personal row")
    return rows


def _form_names(species: list[dict], forms: list[dict]) -> dict[str, list[str]]:
    """``{species id: [species name, form 1 name, ...]}`` for species with data forms."""
    out: dict[str, list[str]] = {}
    for f in forms:
        names = out.setdefault(str(f["species"]), [species[f["species"]]["name"]])
        if f["form"] != len(names):
            raise BuildError(f"forms.json: species {f['species']} form {f['form']} out of order")
        names.append(f["name"])
    return out


def _item_maps(items: list[dict], vanilla: list[dict] | None) -> tuple[list[int], list[int]]:
    """``(ItemId, TrainerItemId)`` per item. Vanilla: by name. Kaizo (``vanilla`` given):
    by the slot's vanilla name, only when the slot keeps that name (up to spelling) and
    vanilla's hold effect (and, for the bag item, vanilla's ``hp_restored``); a renamed or
    re-purposed item is ``UNSUPPORTED``."""
    if vanilla is None:
        hold = [int(DECOMP_ITEM_TO_ID.get(i["name"], ItemId.UNSUPPORTED)) for i in items]
        bag = [
            int(DECOMP_TRAINER_ITEM_TO_ID.get(i["name"], TrainerItemId.UNSUPPORTED)) for i in items
        ]
        return hold, bag
    from tools.kaizo.resolve import spelling

    if len(vanilla) != len(items):
        raise BuildError(f"items: {len(items)} rows, vanilla has {len(vanilla)}")
    hold, bag = [], []
    for item, van in zip(items, vanilla, strict=True):
        same_name = spelling(item["name"]) == spelling(van["name"])
        same_hold = same_name and item["hold_effect"] == van["hold_effect"]
        # A bag item's effect also rests on its restore amount (Potion's 20, ids.py).
        same_bag = same_hold and item.get("hp_restored") == van.get("hp_restored")
        held = DECOMP_ITEM_TO_ID.get(van["name"], ItemId.UNSUPPORTED)
        used = DECOMP_TRAINER_ITEM_TO_ID.get(van["name"], TrainerItemId.UNSUPPORTED)
        hold.append(int(held if same_hold else ItemId.UNSUPPORTED))
        bag.append(int(used if same_bag else TrainerItemId.UNSUPPORTED))
    return hold, bag


def _kaizo_rom_changes(inter: Path, rom_diff: Path = ROM_DIFF) -> tuple[frozenset[int], int]:
    """(effect ids whose ``be_seq`` member changed and is not ``modelled``, the count of
    ``unknown`` changed code keys and script members). Both are counted against what the ROM
    diff changed (``rom/changed_code.json``, ``rom/changed_scripts.json``), not against the
    entries ``rom_diff.toml`` happens to hold: a changed key or member with no entry counts as
    unknown, and an entry for something the diff did not change is stale and raises. A member
    marked ``irrelevant`` is UNSUPPORTED too: deliberately conservative, since "irrelevant"
    says the change does not matter to the classifier, not that the engine models the new
    script."""
    from tools.rom.diff import load_code_classification, load_member_classification

    for name in ("changed_scripts", "changed_code"):
        if not (inter / "rom" / f"{name}.json").exists():
            raise BuildError(
                f"no {inter / 'rom' / name}.json: re-run python -m tools.extract_rom --game kaizo"
            )
    changed = _load(inter / "rom", "changed_scripts")
    try:
        members = load_member_classification(rom_diff, changed)
        code = load_code_classification(rom_diff, _load(inter / "rom", "changed_code"))
    except ValueError as e:
        raise BuildError(str(e)) from e
    effects = frozenset(
        i for (narc, i), (status, _) in members.items() if narc == BE_SEQ and status != "modelled"
    )
    unknown = sum(st == "unknown" for st, _ in code.values())
    unknown += sum(st == "unknown" for st, _ in members.values())
    return effects, unknown


def resolve_conditions(inter: Path, rom_diff: Path = ROM_DIFF) -> list[dict]:
    """The ``not_reached`` conditions of ``rom_diff`` as ids, sorted by label. ``moves`` are the
    move ids whose ROM ``effect`` (the be_seq index) is in ``move_effect``; ``abilities`` and
    ``items`` are the decomp ids of the named ability or held item (exact name, case-
    insensitive); ``trainers`` are the trainers placed on a map header whose weather byte is in
    ``map_weather`` (``rom/rom_trainer_maps.json`` joined with ``rom/rom_map_weather.json``).
    A name that matches nothing, or a missing intermediate, raises ``BuildError``."""
    from tools.rom.diff import load_conditions

    try:
        raw = load_conditions(rom_diff)
    except ValueError as e:
        raise BuildError(str(e)) from e
    moves = _load(inter, "moves")
    names = {
        kind: {n["name"].lower(): i for i, n in reversed(list(enumerate(_load(inter, file))))}
        for kind, file in (("ability", "abilities"), ("held_item", "items"))
    }
    out = []
    for label, when in sorted(raw.items()):
        effects, weathers = set(when.get("move_effect", ())), set(when.get("map_weather", ()))
        ids = {}
        for kind in ("ability", "held_item"):
            ids[kind] = []
            for name in when.get(kind, ()):
                if name.lower() not in names[kind]:
                    raise BuildError(f"{rom_diff}: {label}: no {kind} named {name!r}")
                ids[kind].append(names[kind][name.lower()])
        trainers = []
        if weathers:
            for name in ("rom_trainer_maps", "rom_map_weather"):
                if not (inter / "rom" / f"{name}.json").exists():
                    raise BuildError(
                        f"no {inter / 'rom' / name}.json: re-run python -m tools.extract_rom "
                        "--game kaizo"
                    )
            weather = _load(inter / "rom", "rom_map_weather")
            trainers = sorted(
                int(t)
                for t, headers in _load(inter / "rom", "rom_trainer_maps").items()
                if any(weather[h] in weathers for h in headers)
            )
        out.append(
            {
                "label": label,
                "formats": list(when.get("format", ())),
                "areas": list(when.get("area_condition", ())),
                "moves": [i for i, m in enumerate(moves) if m["effect"] in effects],
                "abilities": ids["ability"],
                "items": ids["held_item"],
                "trainers": trainers,
                "trainer_items": bool(when.get("trainer_items", False)),
            }
        )
    return out


def build_arrays(inter: Path, game: str = "platinum_us") -> tuple[dict[str, np.ndarray], dict]:
    kaizo = game == KAIZO
    species_only = _load(inter, "species")
    forms = _load(inter, "forms")
    form_rows = _load(inter, "form_rows")
    moves = _load(inter, "moves")
    move_lists = _load(inter, "move_lists")
    items = _load(inter, "items")
    abilities = _load(inter, "abilities")
    types = _load(inter, "types")
    natures = _load(inter, "natures")
    exp_tables = _load(inter, "exp_tables")
    charmap = _load(inter, "charmap")
    level_learnsets = _load(inter, "learnsets")
    tm_learnsets = _load(inter, "tm_learnsets")
    evolutions = _load(inter, "evolutions")
    egg_moves = _load(inter, "egg_moves")
    evolution_methods = _load(inter, "evolution_methods")
    tutor_moves = _load(inter, "tutor_moves")
    tutor_learnsets = _load(inter, "tutor_learnsets")
    species = _personal_rows(species_only, forms, form_rows)
    item_ids_by_name = {i["name"]: n for n, i in enumerate(items)}
    first_tm_item = item_ids_by_name["TM01"]
    tm_moves = []
    for n in range(NUM_TMS + NUM_HMS):
        item = items[first_tm_item + n]
        want = f"TM{n + 1:02d}" if n < NUM_TMS else f"HM{n - NUM_TMS + 1:02d}"
        if item["name"] != want or item["teaches_move"] is None:
            raise BuildError(f"item {first_tm_item + n} is {item['name']!r}, expected {want}")
        tm_moves.append(item["teaches_move"])

    vanilla_items = _load(GAMES[VANILLA].intermediate_dir, "items") if kaizo else None
    changed_effects, unclassified = _kaizo_rom_changes(inter) if kaizo else (frozenset(), 0)
    move_names = {m["name"] for m in moves}
    missing = sorted((UNSUPPORTED_MOVES | TYPE_CHART_STATUS_MOVES) - move_names)
    if missing:
        raise BuildError(f"effect_map moves not in moves.json: {missing}")
    ability_names = [a["name"] for a in abilities]
    ability_ids = [int(DECOMP_ABILITY_TO_ID.get(n, AbilityId.UNSUPPORTED)) for n in ability_names]
    unknown = sorted(set(DECOMP_ABILITY_TO_ID) - set(ability_names))
    unknown += sorted(
        (set(DECOMP_ITEM_TO_ID) | set(DECOMP_TRAINER_ITEM_TO_ID))
        - {i["name"] for i in (vanilla_items or items)}
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

    chart_foresight = np.zeros((n_types, n_types), dtype=np.int64)
    for a, d in types["foresight_immunities"]:
        chart_foresight[a, d] = 1

    nature_mods = np.zeros((len(natures), 6), dtype=np.int64)
    for i, n in enumerate(natures):
        if n["up"] is not None:
            nature_mods[i, n["up"]] += 1
            nature_mods[i, n["down"]] -= 1

    effects = [_move_effect(m, changed_effects) for m in moves]
    item_ids, trainer_item_ids = _item_maps(items, vanilla_items)
    supported_item = [e not in (ItemId.UNSUPPORTED, ItemId.NONE) for e in item_ids]

    arrays = {
        "species_base_stats": _arr([s["base_stats"] for s in species], np.int16, "base_stats"),
        "species_types": _arr([s["types"] for s in species], np.int8, "types"),
        "species_abilities": _arr(
            [[ability_ids[a] for a in s["abilities"]] for s in species], np.int16, "abilities"
        ),
        "species_abilities_decomp": _arr(
            [s["abilities"] for s in species], np.int16, "decomp abilities"
        ),
        "species_weight": _arr([s["weight_hg"] for s in species], np.int32, "weight"),
        "species_gender": _arr([s["gender_ratio"] for s in species], np.int16, "gender"),
        "species_friendship": _arr([s["base_friendship"] for s in species], np.int16, "friendship"),
        "species_exp_rate": _arr([s["exp_rate"] for s in species], np.int8, "exp rate"),
        "species_exp_yield": _arr([s["exp_yield"] for s in species], np.int16, "exp yield"),
        "species_form_row": _arr(form_rows, np.int16, "form rows"),
        "exp_table": _arr(exp_tables, np.int32, "exp table"),
        "move_power": _arr([m["power"] for m in moves], np.int16, "power"),
        "move_type": _arr([m["type"] for m in moves], np.int8, "move type"),
        "move_category": _arr([m["category"] for m in moves], np.int8, "category"),
        "move_accuracy": _arr([m["accuracy"] for m in moves], np.int8, "accuracy"),
        "move_pp": _arr([m["pp"] for m in moves], np.int8, "pp"),
        "move_priority": _arr([m["priority"] for m in moves], np.int8, "priority"),
        "move_target": _arr([MOVE_TARGETS.index(m["target"]) for m in moves], np.int8, "target"),
        "move_effect": _arr([e for e, _ in effects], np.int16, "effect"),
        "move_effect_chance": _arr([m["effect_chance"] for m in moves], np.int8, "chance"),
        "move_ai_effect": _arr([m["ai_effect"] for m in moves], np.int16, "ai effect"),
        "move_param": _arr([p for _, p in effects], np.int16, "move param"),
        "move_flags": _arr(
            [_move_flags(m, i, move_lists) for i, m in enumerate(moves)], np.int16, "move flags"
        ),
        "type_chart": _arr(chart, np.int8, "type chart"),
        "type_chart_order": _arr(chart_order, np.int16, "type chart order"),
        "type_chart_foresight": _arr(chart_foresight, np.int8, "type chart foresight"),
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
        # The game id (nuzline.data.games); manifest only, the npz arrays do not carry it.
        "game": game,
        "decomp_commit": _decomp_commit(),
        "species_names": [s["name"] for s in species_only],
        # Personal row (species, then data forms; nuzline.data.forms.form_row) per
        # [species][form], and each data-form species' form names (form 0 = the species).
        "species_form_row": form_rows,
        "form_names": _form_names(species_only, forms),
        # Decomp ability ids (not engine ids) per personal-row slot, so a party file's
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
        "charmap": charmap,
        # Host-only lookups (nuzline.data.learnsets): per personal row, the level-up
        # (level, move) pairs in learn order and the TM/HM indices it can learn;
        # tm_moves[index] is the move that TM/HM teaches; first_tm_item is ITEM_TM01
        # (TM01-TM92 then HM01-HM08 are consecutive ids).
        "level_learnsets": level_learnsets,
        "tm_learnsets": tm_learnsets,
        "tm_moves": tm_moves,
        "first_tm_item": first_tm_item,
        # Keyed by species id (LoadSpeciesEvolutions, src/pokemon.c:4812;
        # LoadSpeciesEggMoves, src/overlay005/daycare.c:445): evolutions are
        # [method, param, target] with method an index into evolution_methods (the
        # generated/evolution_methods.txt stems, lowercased); egg_moves lists the moves of
        # the species that have any.
        "evolutions": evolutions,
        "evolution_methods": evolution_methods,
        "egg_moves": egg_moves,
        # Move tutors (sTeachableMoves order): {move, costs: {red, blue, yellow, green},
        # location}; tutor_learnsets maps a personal row to indices into tutor_moves
        # (Pokemon_ReadMovesetMaskByte, src/overlay005/scrcmd_move_tutor.c:211-259).
        "tutor_moves": tutor_moves,
        "tutor_learnsets": tutor_learnsets,
    }
    if kaizo:
        meta = _load(inter, "kaizo")
        manifest.update(meta["provenance"])
        manifest["aliases"] = meta["aliases"]
        manifest["renamed"] = _load(inter, "renamed")
        # Effect ids made UNSUPPORTED by Ruling 13 (changed be_seq members, not modelled).
        manifest["changed_effects"] = sorted(changed_effects)
        # Moves whose chance-0 secondary never fires (Ruling 21).
        manifest["no_secondary_moves"] = [
            i
            for i, m in enumerate(moves)
            if m["effect"] not in changed_effects and _no_secondary(m)
        ]
        manifest["unclassified_rom_changes"] = unclassified
        # The not_reached entries as ids, for coverage to refuse a fight that reaches one.
        manifest["rom_change_conditions"] = resolve_conditions(inter)
    return arrays, manifest


def build(intermediate_dir: Path, out_dir: Path, game: str = "platinum_us") -> None:
    arrays, manifest = build_arrays(Path(intermediate_dir), get_game(game).id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The npz carries its own version too (nuzline.data.tables.check_npz): manifest.json
    # is tracked in git and tables.npz is not, so the manifest cannot vouch for the npz.
    np.savez_compressed(out_dir / "tables.npz", version=np.int16(MANIFEST_VERSION), **arrays)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="vanilla", help="vanilla or kaizo (or a full game id)")
    p.add_argument("--inter", type=Path, help="default: the game's intermediate_dir")
    p.add_argument("--out", type=Path, help="default: the game's tables_dir")
    args = p.parse_args(argv)
    game = get_game(args.game)
    inter = args.inter or game.intermediate_dir
    if game.id == KAIZO:
        from tools.kaizo.crosscheck import DEFAULT_REPORT, CrossCheckError
        from tools.kaizo.resolve import ResolveError, resolve
        from tools.rom.inputs import RomInputError, load_inputs

        try:
            resolve(load_inputs(), inter)
        except (BuildError, CrossCheckError, ResolveError, RomInputError, FileNotFoundError) as e:
            print(f"error: {_ascii(str(e))}", file=sys.stderr)
            print(f"report: {DEFAULT_REPORT}", file=sys.stderr)
            return 1
    try:
        build(inter, args.out or game.tables_dir, game.id)
    except BuildError as e:
        print(f"error: {_ascii(str(e))}", file=sys.stderr)
        return 1
    return 0


def _ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


if __name__ == "__main__":
    sys.exit(main())
