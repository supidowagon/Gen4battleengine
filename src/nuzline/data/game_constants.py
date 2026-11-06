"""Every game id the engine and the AI hard-code, classified per game (spec §7.2).

Code names some decomp ids directly (``MOVE_STRUGGLE = 165``, the AI's ability tables). A
game that moves or renames those slots would silently change what that code means, so
each such constant is listed here and checked against the game's data when the game is
loaded (``nuzline.data.games.load_game``):

* ``CONSTANTS``: the ids, each with the name (and, for items, the hold effect or HP
  restored) it must have. ``scope`` ``"engine"`` is asserted for every game;
  ``"vanilla_policy"`` only for games whose AI policy is the vanilla transcription
  (``POLICY``), since a game with no transcribed policy is refused by coverage anyway.
* ``GAME_DATA``: lists that are game data and come from the game's intermediate files.
* ``EXEMPT``: tables that are not game ids (engine or AI-script encodings, or keyed by
  name or effect id), with the reason.

Every key is the dotted ``module.ATTR`` form (``nuzline.engine.ids.Stat``,
``tools.effect_map.SOUND_MOVES``); the file-path form (``engine/ids.py Stat``) appears
only in ``check_constants`` messages. ``tests/data/test_game_constants.py`` checks each
entry against the code and scans the modules for unclassified id tables.

This module imports neither jax nor ``nuzline.data.tables``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuzline.data.tables import Names

KINDS = ("move", "ability", "item", "type", "species")
SCOPES = ("engine", "vanilla_policy")


@dataclass(frozen=True)
class Constant:
    """Code at ``where`` uses decomp ``kind`` id ``id`` as ``name``."""

    where: str
    kind: str
    id: int
    name: str
    scope: str
    hold_effect: int | None = None
    hp_restored: int | None = None


def _table(where: str, kind: str, scope: str, entries: Mapping[str, int]) -> tuple[Constant, ...]:
    return tuple(Constant(where, kind, i, name, scope) for name, i in entries.items())


_E, _P = "engine", "vanilla_policy"
_TYPES = {
    "Normal": 0,
    "Fighting": 1,
    "Flying": 2,
    "Poison": 3,
    "Ground": 4,
    "Rock": 5,
    "Bug": 6,
    "Ghost": 7,
    "Steel": 8,
    "???": 9,
    "Fire": 10,
    "Water": 11,
    "Grass": 12,
    "Electric": 13,
    "Psychic": 14,
    "Ice": 15,
    "Dragon": 16,
    "Dark": 17,
}
# The moves whose AI script paths are transcribed (roark.md 4-5, plus the test fixture's
# Tackle): ai/commands.py MOVE_EFFECTS and ai/outcome.py _SITES.
_TRANSCRIBED_MOVES = {
    "Stealth Rock": 446,
    "Rock Throw": 88,
    "Pursuit": 228,
    "Screech": 103,
    "Leer": 43,
    "Headbutt": 29,
    "Tackle": 33,
}
# Basic_CheckSoundproof's IfMoveEqualTo list (script.s:121-131).
_SOUND_MOVES = {
    "Growl": 45,
    "Roar": 46,
    "Sing": 47,
    "Supersonic": 48,
    "Screech": 103,
    "Snore": 173,
    "Uproar": 253,
    "Metal Sound": 319,
    "GrassWhistle": 320,
    "Bug Buzz": 405,
    "Chatter": 448,
}
# ai/switching.py _MOVES: moves whose decomp handling the switch check does not model.
_SWITCH_MOVES = {
    "Magnet Rise": 393,
    "Roost": 355,
    "Miracle Eye": 357,
    "Ingrain": 275,
    "Perish Song": 195,
    "Hidden Power": 237,
    "Weather Ball": 311,
    "Natural Gift": 363,
    "Judgment": 449,
}
# ai/switching.py _ANY_SIDE_ABILITIES (ApplyTypeChart / CalcEffectiveness, trainer_ai.c).
_ANY_SIDE_ABILITIES = {"Levitate": 26, "Wonder Guard": 25, "Scrappy": 113, "Normalize": 96}

CONSTANTS: tuple[Constant, ...] = (
    # --- engine: asserted for every game -------------------------------------------
    # Abilities trainer_ai.c and the AI scripts test by id (ABILITY_* in
    # generated/abilities.txt, line N is id N-1).
    *_table(
        "nuzline.ai.commands.ABILITY_IDS",
        "ability",
        _E,
        {
            "Mold Breaker": 104,
            "Volt Absorb": 10,
            "Motor Drive": 78,
            "Water Absorb": 11,
            "Flash Fire": 18,
            "Wonder Guard": 25,
            "Levitate": 26,
            "Soundproof": 43,
            "Clear Body": 29,
            "White Smoke": 73,
            "Shadow Tag": 23,
            "Magnet Pull": 42,
            "Arena Trap": 71,
            "Normalize": 96,
            "Adaptability": 91,
            "Scrappy": 113,
            "Tinted Lens": 110,
            "Filter": 111,
            "Solid Rock": 116,
        },
    ),
    *_table(
        "nuzline.ai.switching.ABILITY_IDS",
        "ability",
        _E,
        {
            **_ANY_SIDE_ABILITIES,
            "Shadow Tag": 23,
            "Arena Trap": 71,
            "Magnet Pull": 42,
            "Flash Fire": 18,
            "Water Absorb": 11,
            "Volt Absorb": 10,
            "Natural Cure": 30,
        },
    ),
    # TYPE_* (generated/pokemon_types.txt order).
    *_table(
        "nuzline.ai.commands.TYPE_IDS",
        "type",
        _E,
        {k: _TYPES[k] for k in ("Ground", "Ghost", "Fire", "Water", "Electric", "Psychic")},
    ),
    Constant("nuzline.ai.switching._TYPE_FLYING", "type", 2, "Flying", _E),
    Constant("nuzline.ai.switching._TYPE_STEEL", "type", 8, "Steel", _E),
    *_table("nuzline.engine.ids.Type", "type", _E, _TYPES),
    # MOVE_STRUGGLE (generated/moves.txt line 166).
    Constant("nuzline.engine.ids.MOVE_STRUGGLE", "move", 165, "Struggle", _E),
    Constant("nuzline.ai.switching.MOVE_STRUGGLE", "move", 165, "Struggle", _E),
    # Thunder Wave: BattleControllerPlayer_CheckTypeChart's MOVE_THUNDER_WAVE
    # (src/battle/battle_controller_player.c:2357).
    Constant("tools.effect_map.TYPE_CHART_STATUS_MOVES", "move", 86, "Thunder Wave", _E),
    # SPECIES_SHEDINJA (src/pokemon.c:598, :634); SPECIES_GIRATINA, SPECIES_SHAYMIN
    # (include/constants/forms.h:94, :98).
    Constant("nuzline.engine.stats.SPECIES_SHEDINJA", "species", 292, "Shedinja", _E),
    Constant("nuzline.engine.coverage.SPECIES_GIRATINA", "species", 487, "Giratina", _E),
    Constant("nuzline.engine.coverage.SPECIES_SHAYMIN", "species", 492, "Shaymin", _E),
    # Items (generated/items.txt, line N is id N-1); prep looks Heart Scale up by name.
    Constant("nuzline.prep.problem.HEART_SCALE", "item", 93, "Heart Scale", _E),
    Constant("nuzline.engine.coverage.ITEM_GRISEOUS_ORB", "item", 112, "Griseous Orb", _E),
    # Hold effects (generated/item_hold_effects.txt, line N is id N-1):
    # HOLD_EFFECT_POWER_UP_SE (96) and HOLD_EFFECT_SPEED_DOWN_GROUNDED (106).
    Constant("nuzline.ai.commands.ITEM_EXPERT_BELT", "item", 268, "Expert Belt", _E, 96),
    Constant("nuzline.ai.switching.ITEM_IRON_BALL", "item", 278, "Iron Ball", _E, 106),
    # Potion's itemUseParams.hpRestored (res/items/data/potion.json), which
    # BattleSystem_UseBagItem (src/battle/battle_system.c:435) restores.
    Constant("nuzline.engine.ids.POTION_HP_RESTORED", "item", 17, "Potion", _E, hp_restored=20),
    # --- vanilla_policy: the transcribed vanilla AI scripts --------------------------
    *_table("nuzline.ai.commands.MOVE_EFFECTS", "move", _P, _TRANSCRIBED_MOVES),
    # BattleEffect's ids are the battle effects of the transcribed moves; asserted
    # through those moves (MOVE_EFFECTS maps each one to its member).
    # A classification marker: these entries hold move ids, not effect ids.
    *_table("nuzline.ai.commands.BattleEffect", "move", _P, _TRANSCRIBED_MOVES),
    *_table("nuzline.ai.outcome._SITES", "move", _P, _TRANSCRIBED_MOVES),
    # Basic_Main's OHKO and Roar checks (script.s:52), Expert's U-turn, and the
    # Basic_CheckSoundproof list.
    *_table(
        "nuzline.ai.commands.MOVE_IDS",
        "move",
        _P,
        {
            "Fissure": 90,
            "Horn Drill": 32,
            "U-turn": 369,
            "Roar": 46,
            "Whirlwind": 18,
            **{k: v for k, v in _SOUND_MOVES.items() if k != "Roar"},
        },
    ),
    *_table("nuzline.ai.commands.SOUND_MOVES", "move", _P, _SOUND_MOVES),
    *_table("nuzline.ai.switching.MOVE_IDS", "move", _P, _SWITCH_MOVES),
    *_table("nuzline.ai.switching._MOVES", "move", _P, _SWITCH_MOVES),
    *_table(
        "nuzline.ai.switching.PLAYER_MOVE_REFUSALS",
        "move",
        _P,
        {k: v for k, v in _SWITCH_MOVES.items() if k != "Hidden Power"},
    ),
    *_table("nuzline.ai.switching.PLAYER_ABILITY_REFUSALS", "ability", _P, _ANY_SIDE_ABILITIES),
    *_table(
        "nuzline.ai.policy._STATE_MOVE_IDS",
        "move",
        _P,
        {k: _SWITCH_MOVES[k] for k in ("Miracle Eye", "Roost", "Magnet Rise", "Ingrain")},
    ),
    # The Dry Skin quirk (script.s:78 tests Levitate a second time; policy.py _basic).
    Constant("nuzline.ai.policy._basic", "ability", 26, "Levitate", _P),
)

# Lists that are game data: "module.ATTR" -> the intermediate file that replaces it
# (Task 5 moves both to the game's move_lists.json).
GAME_DATA: dict[str, str] = {
    # src/battle/battle_lib.c sSoundMoves (:3473).
    "tools.effect_map.SOUND_MOVES": "move_lists.json",
    # src/battle/battle_lib.c sPunchingMoves (:6566).
    "tools.effect_map.PUNCH_MOVES": "move_lists.json",
}

_ENGINE_ENCODING = (
    "engine encoding, not a game id; a game that changes how the decomp handles it is a "
    "rom_diff.toml entry (Ruling 13)"
)
_AI_ENCODING = "decomp AI-script encoding (trainer_ai.h / trainer_ai.c argument), not a game id"
_KEYED_BY_EFFECT = (
    "keyed by decomp effect id: engine semantics of the vanilla effect scripts; a game's "
    "script changes are rom_diff.toml entries (Ruling 13)"
)
_KEYED_BY_NAME = "keyed by name, looked up in each game's own data at table build"

# "module.ATTR" -> why it is not a game id table.
EXEMPT: dict[str, str] = {
    "nuzline.engine.ids.MoveTarget": (
        "engine index order of the decomp's RANGE_* (generated/move_ranges.txt); "
        + _ENGINE_ENCODING
    ),
    "nuzline.engine.ids.EffectId": _ENGINE_ENCODING,
    "nuzline.engine.ids.Status": _ENGINE_ENCODING,
    "nuzline.engine.ids.Stat": "engine index order of every stat vector; " + _ENGINE_ENCODING,
    "nuzline.engine.ids.MoveFlag": _ENGINE_ENCODING,
    "nuzline.engine.ids.Weather": _ENGINE_ENCODING,
    "nuzline.engine.ids.ItemId": _ENGINE_ENCODING,
    "nuzline.engine.ids.TrainerItemId": _ENGINE_ENCODING,
    "nuzline.engine.ids.AbilityId": _ENGINE_ENCODING,
    "nuzline.ai.commands.Battler": _AI_ENCODING + " (AI_BATTLER_*)",
    "nuzline.ai.commands.LoadType": _AI_ENCODING + " (LOAD_*, trainer_ai.c:937)",
    "nuzline.ai.commands.SideCondition": _AI_ENCODING + " (SIDE_CONDITION_*)",
    "nuzline.ai.commands.TypeMulti": _AI_ENCODING + " (TYPE_MULTI_*, battle.h:144-153)",
    "nuzline.ai.switching.SwitchSite": "kind of an AI switch draw (roark.md 11.1), not a game id",
    "nuzline.ai.switching._CANDIDATE_TARGETS": (
        "party slot of each AI switch candidate (-1: keep the battler), an engine layout"
    ),
    "tools.effect_map.DECOMP_EFFECT_TO_ID": _KEYED_BY_EFFECT,
    "tools.effect_map._EFFECTS": _KEYED_BY_EFFECT,
    "tools.effect_map.DECOMP_EFFECT_NAMES": _KEYED_BY_EFFECT,
    "tools.effect_map.CERTAIN_SECONDARY_EFFECTS": _KEYED_BY_EFFECT,
    "tools.effect_map.DECOMP_ABILITY_TO_ID": (
        "keyed by ability name: engine semantics of the vanilla ability handling; a game's "
        "changes to it are rom_diff.toml entries (Ruling 13)"
    ),
    "tools.effect_map.UNSUPPORTED_MOVES": _KEYED_BY_NAME,
    "tools.effect_map.DECOMP_ITEM_TO_ID": (
        _KEYED_BY_NAME + "; Task 15 also requires the game's hold effect to equal vanilla's"
    ),
    "tools.effect_map.DECOMP_TRAINER_ITEM_TO_ID": (
        _KEYED_BY_NAME + "; Task 15 also requires the game's hold effect to equal vanilla's"
    ),
}

# Game id -> the AI policy its trainers' scripts are transcribed as (None: none yet, so
# coverage refuses every trainer of that game, spec 5.10).
POLICY: dict[str, str | None] = {
    "platinum_us": "vanilla",
    "platinum_kaizo_nightmare_final": None,
}


def message_path(where: str) -> str:
    """``nuzline.ai.switching.MOVE_STRUGGLE`` -> ``ai/switching.py MOVE_STRUGGLE``."""
    module, attr = where.rsplit(".", 1)
    module = module.removeprefix("nuzline.")
    return f"{module.replace('.', '/')}.py {attr}"


def _check(c: Constant, names: Names, items: Sequence[dict]) -> list[str]:
    where = message_path(c.where)
    table = getattr(names, f"{c.kind}_names")
    if not 0 <= c.id < len(table):
        span = f"{len(table)} {c.kind} ids"
        return [f"{where}: id {c.id} is out of range in this game ({span}), expected {c.name!r}"]
    lines = []
    if table[c.id].lower() != c.name.lower():
        lines.append(f"{where}: id {c.id} is {table[c.id]!r} in this game, expected {c.name!r}")
    if c.hold_effect is None and c.hp_restored is None:
        return lines
    if c.id >= len(items):
        return [*lines, f"{where}: item {c.id} is missing from this game's items.json"]
    row = items[c.id]
    if c.hold_effect is not None and row.get("hold_effect") != c.hold_effect:
        lines.append(
            f"{where}: {c.name} has hold effect {row.get('hold_effect')} in this game, "
            f"expected {c.hold_effect}"
        )
    if c.hp_restored is not None:
        hp = row.get("hp_restored")
        if hp is None:
            lines.append(f"{where}: this game's items.json has no hp_restored (re-extract it)")
        elif hp != c.hp_restored:
            lines.append(f"{where}: {c.name} restores {hp} in this game, expected {c.hp_restored}")
    return lines


def check_constants(game_id: str, names: Names, items: Sequence[dict]) -> list[str]:
    """One line per hard-coded id that ``game_id``'s data contradicts; ``items`` are the
    rows of the game's ``intermediate_dir/items.json``. Engine constants are checked for
    every game, the vanilla policy's only for a game whose ``POLICY`` is ``"vanilla"``."""
    policy = POLICY.get(game_id)
    lines: list[str] = []
    for c in CONSTANTS:
        if c.scope == "engine" or policy == "vanilla":
            lines += _check(c, names, items)
    return lines
