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
* ``item_hold_effect[item]``: the decomp hold effect as stored (``ItemData.holdEffect``; 0 = none),
  whether or not the engine supports it (``item_effect`` is the mapped engine id).
* ``item_param[item]``: ``(decomp hold param, 0)`` for supported items, else zeros; a
  ``RESIST_BERRY``'s column 1 is the type its hold effect resists
  (``effect_map.RESIST_BERRY_HOLD_EFFECTS``), a ``TYPE_BOOST``'s the type its hold effect
  boosts (``effect_map.TYPE_BOOST_HOLD_EFFECTS``), a ``PINCH_BERRY``'s the ``Stat`` its hold
  effect raises (``effect_map.PINCH_STAT_HOLD_EFFECTS``), a ``FLAVOR_BERRY``'s its ``Flavor``
  (``effect_map.FLAVOR_HOLD_EFFECTS``).
* ``item_effect[item]``: the ``ids.ItemId`` (``effect_map.held_item_ids``: by name, else by
  hold effect, else ``NONE`` / ``NO_EFFECT_BERRY`` for an item with no hold effect);
  ``UNSUPPORTED`` for an item dividing by a param below its minimum (``_pinch_divisor_ok``).
* ``item_hold_param[item]``: the decomp hold param as stored (``ItemData.effectParam``, which
  ``Item_LoadParam(item, ITEM_PARAM_EFFECT_PARAM)`` returns, ``src/item.c:160-161``), whether or
  not the engine supports the item. The level-up EV gain reads it as a Power item's
  ``itemPower`` (``BattleScript_CalcEffortValues``, ``src/battle/battle_script.c:10226-10228``).
* ``item_fling_power`` / ``item_fling_effect`` / ``item_gift_power`` / ``item_gift_type``
  ``[item]``: ``ItemData.flingPower``, ``.flingEffect``, ``.naturalGiftPower`` and
  ``.naturalGiftType`` as stored (include/item.h:61-64; 0x1F = no Natural Gift type), whether or
  not the engine supports the item: Fling (``Battler_ItemFlingPower`` / ``_ItemFlingEffect``,
  battle_lib.c:5482-5498) and Natural Gift (``Battler_NaturalGiftPower`` / ``_Type``, :5465-5480).
* ``trainer_item_effect[item]``: ``ids.TrainerItemId`` of the item used from a trainer's
  bag (``DECOMP_TRAINER_ITEM_TO_ID``).
* ``species_exp_rate[species]``: growth rate id (``generated/exp_rates.txt`` order: 0 medium
  fast, 1 erratic, 2 fluctuating, 3 medium slow, 4 fast, 5 slow); ``exp_table[rate, level]``
  is the experience for ``level`` 0..100 (``exp_tables.csv``, same rate order).
* ``species_exp_yield[species]``: ``SpeciesData.baseExpReward`` (int16).
* ``species_ev_yield[row, stat]``: ``SpeciesData.evYields`` (2-bit fields,
  ``include/struct_defs/species.h:25-32, 44``), per personal row (so per form), ``ids.Stat``
  order.
* ``learnset_move[row, i]`` / ``learnset_level[row, i]``: entry ``i`` of the personal row's
  level-up learnset (``learnsets.json``) in source order, never sorted: the game scans the
  whole list for each level (``Pokemon_LevelUpMove``, ``src/pokemon.c:3935-3966``), and vanilla
  row 354 is unsorted. ``LEARNSET_SLOTS`` = 21 columns: ``SpeciesLearnset.entries`` is
  ``MAX_LEARNSET_ENTRIES + 1`` = 21 u16 *including* the terminator (``include/struct_defs/
  species.h:10, 73-75``), but ``ALIGN_4`` (``include/pch/global_pch.h:27``) rounds its size to
  44 bytes = 22 u16, so 21 entries plus the terminator fit (Kaizo row 359 has 21). Both are
  0 after the last entry; a real entry has level 1..100, so level 0 marks the end.
* ``move_is_hm[move]``: 1 for the moves of HM01-HM08, ``tm_moves[92:100]``
  (``Item_IsHMMove``, ``src/item.c:306-315``: ``sTMHMMoves[NUM_TMS + i]``).
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
* ``ai_policy`` (int8 scalar): the game's trainer-AI policy, the index of
  ``game_constants.POLICY[game]`` in ``game_constants.AI_POLICIES`` (0 vanilla, 1 kaizo;
  ``nuzline.ai.layout.layout_of`` reads it). A game without a policy is a ``BuildError``.

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

from nuzline.data.game_constants import AI_POLICIES, POLICY
from nuzline.data.games import GAMES, KAIZO, VANILLA, get_game
from nuzline.data.tables import MANIFEST_VERSION
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, MoveTarget, TrainerItemId
from tools.effect_map import (
    BLIZZARD_EFFECT,
    CERTAIN_SECONDARY_EFFECTS,
    CHARGE_FLAG_EFFECTS,
    CRIT_HOLD_EFFECTS,
    DECOMP_ABILITY_TO_ID,
    DECOMP_EFFECT_TO_ID,
    DECOMP_ITEM_TO_ID,
    DECOMP_TRAINER_ITEM_TO_ID,
    DIVISOR_PARAM_MIN,
    FLAVOR_HOLD_EFFECTS,
    INVOKER_MOVES,
    KAIZO_EFFECTS,
    KAIZO_FLAG_EFFECTS,
    METRONOME_EXCLUDED_MOVES,
    MULTI_TURN_EFFECTS,
    PINCH_STAT_HOLD_EFFECTS,
    RESIST_BERRY_HOLD_EFFECTS,
    SCRIPT_FLAG_EFFECTS,
    SEMI_HIT_EFFECTS,
    SLEEP_TALK_EXCLUDED_MOVES,
    THUNDER_EFFECT,
    TYPE_BOOST_HOLD_EFFECTS,
    TYPE_CHART_STATUS_MOVES,
    UNSUPPORTED_MOVES,
    WEATHER_ROCK_HOLD_EFFECTS,
    held_item_ids,
)

NUM_TMS = 92
NUM_HMS = 8
# 21 entries + the terminator fit the ALIGN_4-rounded 44-byte buffer (species.h:10, 73-75;
# global_pch.h:27): see the module docstring.
LEARNSET_SLOTS = 21
MAX_LEVEL = 100
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
    # BattleControllerPlayer_CheckExtraFlinch (battle_controller_player.c:4642).
    "MOVE_FLAG_TRIGGERS_KINGS_ROCK": MoveFlag.KINGS_ROCK,
}
SECONDARY = (
    EffectId.SECONDARY_STAT_SELF,
    EffectId.SECONDARY_STAT_FOE,
    EffectId.SECONDARY_STATUS,
    EffectId.RAISE_ALL_STATS_HIT,
)


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


def _move_flags(move: dict, move_id: int, lists: dict[str, list[int]], kaizo: bool = False) -> int:
    """The move's flag bits; SOUND and PUNCH come from ``move_lists.json`` by move id;
    NO_SLEEP_TALK and NO_METRONOME from the battle code's id lists (``tools.effect_map``) and,
    for Move_IsMultiTurn, the move's own decomp effect. The battle code's compares by effect id
    (Move_IsMultiTurn, THUNDER, BLIZZARD) hold whatever the script; the bits read off a script
    (CHARGE_FLAG_EFFECTS, SEMI_HIT_EFFECTS, SCRIPT_FLAG_EFFECTS) do not: for Kaizo a modelled
    changed script (``KAIZO_EFFECTS``) takes ``KAIZO_FLAG_EFFECTS``' bits instead (B13-3)."""
    bits = 0
    for flag in move["flags"]:
        bits |= DECOMP_FLAG_BITS.get(flag, 0)
    if move_id in lists["sound"]:
        bits |= MoveFlag.SOUND
    if move_id in lists["punch"]:
        bits |= MoveFlag.PUNCH
    effect = move.get("effect")
    if (
        move_id in INVOKER_MOVES
        or move_id in SLEEP_TALK_EXCLUDED_MOVES
        or effect in MULTI_TURN_EFFECTS
    ):
        bits |= MoveFlag.NO_SLEEP_TALK
    if move_id in METRONOME_EXCLUDED_MOVES:
        bits |= MoveFlag.NO_METRONOME
    if effect == THUNDER_EFFECT:
        bits |= MoveFlag.THUNDER
    if effect == BLIZZARD_EFFECT:
        bits |= MoveFlag.BLIZZARD
    if kaizo and effect in KAIZO_EFFECTS:  # a modelled changed script: its own bits (B13-3)
        return int(bits | KAIZO_FLAG_EFFECTS.get(effect, 0))
    # The charge-turn bits and the SYSCTL_HIT_DURING_* bits, by decomp effect (B10-12); the
    # script commands added to another family's effect (B13-1, B13-2).
    bits |= CHARGE_FLAG_EFFECTS.get(effect, 0)
    bits |= SEMI_HIT_EFFECTS.get(effect, 0)
    bits |= SCRIPT_FLAG_EFFECTS.get(effect, 0)
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
    move: dict, changed_effects: frozenset[int] = frozenset(), kaizo: bool = False
) -> tuple[int, tuple[int, int, int]]:
    if move["name"] in UNSUPPORTED_MOVES or move["effect"] in changed_effects:
        return int(EffectId.UNSUPPORTED), (0, 0, 0)
    if kaizo and move["effect"] in KAIZO_EFFECTS:  # a modelled changed script (B9-4)
        _, eid, params = KAIZO_EFFECTS[move["effect"]]
        return int(eid), params
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
    if eid == EffectId.FLINCH and move["effect_chance"] == 0:
        # effect_script_0031.s is a plain indirect effect: BattleSystem_TriggerSecondaryEffect
        # asserts effectChance != 0 (battle_lib.c:1577), while the engine would read chance 0
        # as an ON_HIT flinch.
        raise BuildError(f"move {move['name']!r}: flinch effect {move['effect']} with chance 0")
    if eid == EffectId.FANG_PARALYZE_FLINCH and move["effect_chance"] == 0:
        # subscript_paralyze_or_flinch.s rolls through BtlCmd_CheckEffectActivation, which
        # asserts effectChance != 0 (battle_script.c:8375).
        raise BuildError(f"move {move['name']!r}: fang effect {move['effect']} with chance 0")
    if eid == EffectId.RECOIL_STATUS and move["effect_chance"] == 0:
        # The PROBABILISTIC branch asserts effectChance != 0 (battle_lib.c:1557), while the
        # engine would read chance 0 as a certain status (B10-1).
        raise BuildError(
            f"move {move['name']!r}: recoil-status effect {move['effect']} with chance 0"
        )
    if eid == EffectId.FAKE_OUT and move["effect_chance"] == 0:
        # effect_script_0158.s rolls RandNext() % 100 < chance: chance 0 never flinches, but
        # the engine reads chance 0 as an ON_HIT effect, and PLAIN would drop the failure.
        return int(EffectId.UNSUPPORTED), (0, 0, 0)
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


def _learnset_grid(learnsets: dict[str, list[list[int]]], n_rows: int) -> tuple[list, list]:
    """``(moves, levels)``, each ``n_rows`` lists of ``LEARNSET_SLOTS``: row ``r``'s entries in
    source order, then zeros. A missing row, a row longer than ``LEARNSET_SLOTS`` or a level
    outside ``1..MAX_LEVEL`` (0 is the end marker) is a ``BuildError``."""
    if sorted(learnsets, key=int) != [str(r) for r in range(n_rows)]:
        raise BuildError(f"learnsets.json rows are not 0..{n_rows - 1}")
    moves, levels = [], []
    for r in range(n_rows):
        entries = learnsets[str(r)]
        if len(entries) > LEARNSET_SLOTS:
            raise BuildError(
                f"learnsets.json row {r}: {len(entries)} entries, more than {LEARNSET_SLOTS}"
            )
        for level, _ in entries:
            if not 1 <= level <= MAX_LEVEL:
                raise BuildError(f"learnsets.json row {r}: level {level} is outside 1..100")
        pad = [0] * (LEARNSET_SLOTS - len(entries))
        moves.append([m for _, m in entries] + pad)
        levels.append([lv for lv, _ in entries] + pad)
    return moves, levels


def _form_names(species: list[dict], forms: list[dict]) -> dict[str, list[str]]:
    """``{species id: [species name, form 1 name, ...]}`` for species with data forms."""
    out: dict[str, list[str]] = {}
    for f in forms:
        names = out.setdefault(str(f["species"]), [species[f["species"]]["name"]])
        if f["form"] != len(names):
            raise BuildError(f"forms.json: species {f['species']} form {f['form']} out of order")
        names.append(f["name"])
    return out


def _item_type(item: dict, item_id: int) -> int:
    """``item_param[i, 1]``: the type a ``RESIST_BERRY`` resists
    (``subscript_type_resist_berry.s``) or a ``TYPE_BOOST`` boosts (``sTypeBoostingItems``,
    battle_lib.c:6514), the ``Stat`` a ``PINCH_BERRY`` raises (battle_lib.c:4773-4836), the
    ``Weather`` a ``WEATHER_ROCK`` extends (effect_script_0136.s etc.) or the ``Flavor`` of a
    ``FLAVOR_BERRY`` (battle_lib.c:4698-4771), by its own hold effect, else 0."""
    tables = {
        ItemId.RESIST_BERRY: (RESIST_BERRY_HOLD_EFFECTS, "resists"),
        ItemId.TYPE_BOOST: (TYPE_BOOST_HOLD_EFFECTS, "boosts"),
        ItemId.PINCH_BERRY: (PINCH_STAT_HOLD_EFFECTS, "raises"),
        ItemId.WEATHER_ROCK: (WEATHER_ROCK_HOLD_EFFECTS, "extends"),
        ItemId.CRIT_ITEM: (CRIT_HOLD_EFFECTS, "needs"),
        ItemId.FLAVOR_BERRY: (FLAVOR_HOLD_EFFECTS, "tastes of"),
    }
    if item_id not in tables:
        return 0
    table, verb = tables[ItemId(item_id)]
    if item["hold_effect"] not in table:
        raise BuildError(f"item {item['name']!r}: hold effect {item['hold_effect']} {verb} no type")
    return int(table[item["hold_effect"]][1])


def _pinch_divisor_ok(items: list[dict], ids: list[int]) -> list[int]:
    """``ids`` with every ``effect_map.DIVISOR_PARAM_MIN`` item whose hold param is below its
    minimum made ``UNSUPPORTED``: a pinch berry's trigger divides maxHP by the param, halved by
    Gluttony (battle_lib.c:4774-4778, :1289-1294), Sticky Barb divides maxHP and Shell Bell the
    damage dealt by it (battle_lib.c:5330-5336, bcp.c:4736-4752), so such a param divides by
    zero, which the decomp does not define (the ROM's division routine decides)."""
    return [
        int(ItemId.UNSUPPORTED)
        if e in DIVISOR_PARAM_MIN and i["hold_param"] < DIVISOR_PARAM_MIN[ItemId(e)]
        else e
        for i, e in zip(items, ids, strict=True)
    ]


def _item_maps(items: list[dict], vanilla: list[dict] | None) -> tuple[list[int], list[int]]:
    """``(ItemId, TrainerItemId)`` per item. Vanilla: the held item by
    ``effect_map.held_item_ids``, the bag item by name. Kaizo (``vanilla`` given): vanilla's
    ids of the slot, only when the slot keeps vanilla's name (up to spelling) and hold effect
    (and, for the bag item, vanilla's ``hp_restored``); a renamed or re-purposed item is
    ``UNSUPPORTED``. The hold param is the game's own (``item_param``)."""
    if vanilla is None:
        hold = [int(i) for i in held_item_ids(items)]
        bag = [
            int(DECOMP_TRAINER_ITEM_TO_ID.get(i["name"], TrainerItemId.UNSUPPORTED)) for i in items
        ]
        return hold, bag
    from tools.kaizo.resolve import spelling

    if len(vanilla) != len(items):
        raise BuildError(f"items: {len(items)} rows, vanilla has {len(vanilla)}")
    hold, bag = [], []
    vanilla_held = held_item_ids(vanilla)
    for item, van, held in zip(items, vanilla, vanilla_held, strict=True):
        same_name = spelling(item["name"]) == spelling(van["name"])
        same_hold = same_name and item["hold_effect"] == van["hold_effect"]
        # A bag item's effect also rests on its restore amount (Potion's 20, ids.py).
        same_bag = same_hold and item.get("hp_restored") == van.get("hp_restored")
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
    modelled = {
        i for (narc, i), (status, _) in members.items() if narc == BE_SEQ and status == "modelled"
    }
    # A modelled member needs its KAIZO_EFFECTS entry. Against the real rom_diff.toml the match
    # is exact (B9-15): an entry with no modelled member would override the vanilla mapping of an
    # unchanged script. A test's own small diff models a subset only, so it gets the subset check.
    exact = Path(rom_diff).resolve() == Path(ROM_DIFF).resolve()
    if not modelled <= set(KAIZO_EFFECTS) or (exact and modelled != set(KAIZO_EFFECTS)):
        raise BuildError(
            f"modelled be_seq members {sorted(modelled)} != effect_map.KAIZO_EFFECTS "
            f"{sorted(KAIZO_EFFECTS)}"
        )
    unknown = sum(st == "unknown" for st, _ in code.values())
    unknown += sum(st == "unknown" for st, _ in members.values())
    return effects, unknown


def _rom_intermediate(inter: Path, name: str):
    """``inter/rom/<name>.json``, or ``BuildError`` asking for the Kaizo extraction."""
    if not (inter / "rom" / f"{name}.json").exists():
        raise BuildError(
            f"no {inter / 'rom' / name}.json: re-run python -m tools.extract_rom --game kaizo"
        )
    return _load(inter / "rom", name)


def _check_range(
    rom_diff: Path, label: str, key: str, values: set[int], end: int, what: str
) -> None:
    """``BuildError`` naming ``label`` for a ``when.<key>`` value outside ``0..end-1``."""
    for v in sorted(values):
        if not 0 <= v < end:
            raise BuildError(f"{rom_diff}: {label}: when.{key} value {v} is outside {what}")


def resolve_conditions(inter: Path, rom_diff: Path = ROM_DIFF) -> list[dict]:
    """The ``not_reached`` conditions of ``rom_diff`` as ids, sorted by label. ``moves`` are the
    move ids whose ROM ``effect`` (the be_seq index) is in ``move_effect``; ``abilities`` and
    ``items`` are the decomp ids of the named ability or held item (exact name, case-
    insensitive); ``trainers`` are the trainers placed on a map header whose weather byte is in
    ``map_weather`` or whose id is in ``map_header`` (``rom/rom_trainer_maps.json``, joined with
    ``rom/rom_map_weather.json`` for the weather), plus, for any such map-keyed condition,
    every trainer with a party that no map header places (``trainers.json``).
    A name that matches nothing, an integer outside its ROM table (a ``move_effect`` not a
    ``be_seq`` member, ``rom/rom_script_counts.json``; a ``map_header`` not a header id; a
    ``map_weather`` above the largest weather byte any header holds), or a missing
    intermediate raises ``BuildError``."""
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
        map_headers = set(when.get("map_header", ()))
        ids = {}
        for kind in ("ability", "held_item"):
            ids[kind] = []
            for name in when.get(kind, ()):
                if name.lower() not in names[kind]:
                    raise BuildError(f"{rom_diff}: {label}: no {kind} named {name!r}")
                ids[kind].append(names[kind][name.lower()])
        if effects:
            count = _rom_intermediate(inter, "rom_script_counts")[BE_SEQ]
            _check_range(rom_diff, label, "move_effect", effects, count, f"{BE_SEQ} members")
        trainers = []
        if weathers or map_headers:
            weather = _rom_intermediate(inter, "rom_map_weather")
            placements = _rom_intermediate(inter, "rom_trainer_maps")
            _check_range(rom_diff, label, "map_header", map_headers, len(weather), "map headers")
            top = max(weather)
            _check_range(rom_diff, label, "map_weather", weathers, top + 1, f"0..{top}")
            # Final review I2: a trainer with a party that no map header places may be fought
            # on any map, so every map-keyed condition lists it (safety over coverage).
            unplaced = {
                t["id"]
                for t in _load(inter, "trainers")
                if t["party"] and str(t["id"]) not in placements
            }
            trainers = sorted(
                unplaced
                | {
                    int(t)
                    for t, placed in placements.items()
                    if any(weather[h] in weathers or h in map_headers for h in placed)
                }
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


def _ai_policy(game: str) -> int:
    """``Tables.ai_policy`` of ``game``: its ``POLICY`` as an ``AI_POLICIES`` index."""
    policy = POLICY.get(game)
    if policy not in AI_POLICIES:
        raise BuildError(f"game {game!r} has no AI policy in game_constants.POLICY")
    return AI_POLICIES.index(policy)


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

    learnset_moves, learnset_levels = _learnset_grid(level_learnsets, len(species))
    hm_moves = set(tm_moves[NUM_TMS:])
    effects = [_move_effect(m, changed_effects, kaizo) for m in moves]
    item_ids, trainer_item_ids = _item_maps(items, vanilla_items)
    item_ids = _pinch_divisor_ok(items, item_ids)
    supported_item = [e not in (ItemId.UNSUPPORTED, ItemId.NONE) for e in item_ids]
    item_params = [
        [i["hold_param"] if ok else 0, _item_type(i, e)]
        for i, ok, e in zip(items, supported_item, item_ids, strict=True)
    ]

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
        "species_ev_yield": _arr([s["ev_yield"] for s in species], np.int8, "ev yield"),
        "species_form_row": _arr(form_rows, np.int16, "form rows"),
        "exp_table": _arr(exp_tables, np.int32, "exp table"),
        "learnset_move": _arr(learnset_moves, np.int16, "learnset move"),
        "learnset_level": _arr(learnset_levels, np.int8, "learnset level"),
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
            [_move_flags(m, i, move_lists, kaizo) for i, m in enumerate(moves)],
            np.int32,
            "move flags",
        ),
        "move_is_hm": _arr([int(i in hm_moves) for i in range(len(moves))], np.int8, "move is hm"),
        "type_chart": _arr(chart, np.int8, "type chart"),
        "type_chart_order": _arr(chart_order, np.int16, "type chart order"),
        "type_chart_foresight": _arr(chart_foresight, np.int8, "type chart foresight"),
        "nature_mods": _arr(nature_mods, np.int8, "nature mods"),
        "item_effect": _arr(item_ids, np.int16, "item effect"),
        "item_param": _arr(item_params, np.int16, "item param"),
        "trainer_item_effect": _arr(trainer_item_ids, np.int8, "trainer item effect"),
        "item_hold_effect": _arr([i["hold_effect"] for i in items], np.int16, "item hold effect"),
        "item_hold_param": _arr([i["hold_param"] for i in items], np.int16, "item hold param"),
        "item_fling_power": _arr([i["fling_power"] for i in items], np.int16, "item fling power"),
        "item_fling_effect": _arr([i["fling_effect"] for i in items], np.int8, "fling effect"),
        "item_gift_power": _arr(
            [i["natural_gift_power"] for i in items], np.int16, "natural gift power"
        ),
        "item_gift_type": _arr([i["natural_gift_type"] for i in items], np.int8, "gift type"),
        "ai_policy": _arr(_ai_policy(game), np.int8, "ai policy"),
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
