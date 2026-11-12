"""Every game id the engine and the AI hard-code, classified per game (spec §7.2).

Code names some decomp ids directly (``MOVE_STRUGGLE = 165``, the AI's ability tables). A
game that moves or renames those slots would silently change what that code means, so
each such constant is listed here and checked against the game's data when the game is
loaded (``nuzline.data.games.load_game``):

* ``CONSTANTS``: the ids, each with the name (and, for items, the hold effect, hold param or
  HP restored) it must have. ``scope`` ``"engine"`` is asserted for every game;
  ``"vanilla_policy"`` and ``"kaizo_policy"`` only for games whose AI policy (``POLICY``)
  reads them, since a game with no transcribed policy is refused by coverage anyway. A
  table both policies read is listed once per scope.
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
SCOPES = ("engine", "vanilla_policy", "kaizo_policy")
# The AI policies, in ``Tables.ai_policy`` order (``tools/build_tables.py`` packs the index;
# ``nuzline.ai.layout.layout_of`` reads it).
AI_POLICIES = ("vanilla", "kaizo")


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
    hold_param: int | None = None


def _table(where: str, kind: str, scope: str, entries: Mapping[str, int]) -> tuple[Constant, ...]:
    return tuple(Constant(where, kind, i, name, scope) for name, i in entries.items())


_E, _P, _K = "engine", "vanilla_policy", "kaizo_policy"
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
# ai/commands.py MOVE_IDS: Basic_Main's OHKO checks (script.s:57-58), Expert_Pursuit's
# U-turn (:3730), Expert_StealthRock's Roar and Whirlwind (:6276-6277), Expert_Curse's Gyro
# Ball and Trick Room (:3472-3473, read from K3c Task 6), and the Basic_CheckSoundproof list
# (:121-131).
_AI_MOVE_IDS = {
    "Fissure": 90,
    "Horn Drill": 32,
    "U-turn": 369,
    "Roar": 46,
    "Whirlwind": 18,
    "Gyro Ball": 360,
    "Trick Room": 433,
    **{k: v for k, v in _SOUND_MOVES.items() if k != "Roar"},
}
# ai/commands.py CALC_DAMAGE_MOVES: TrainerAI_CalcDamage's id cases (trainer_ai.c:2885-3056;
# MOVE_* of generated/moves.txt, line N is id N-1).
_CALC_DAMAGE_MOVES = {
    "Natural Gift": 363,
    "Judgment": 449,
    "Hidden Power": 237,
    "Gyro Ball": 360,
    "Dragon Rage": 82,
    "Seismic Toss": 69,
    "Night Shade": 101,
    "Psywave": 149,
    "Return": 216,
    "Frustration": 218,
    "Magnitude": 222,
    "SonicBoom": 49,
    "Low Kick": 67,
    "Grass Knot": 447,
}
# ai/commands.py MOVE_TYPE_MOVES: TrainerAI_MoveType's id cases (trainer_ai.c:3132-3222).
_MOVE_TYPE_MOVES = {"Natural Gift": 363, "Judgment": 449, "Hidden Power": 237, "Weather Ball": 311}


def _policy_tables(scope: str) -> tuple[Constant, ...]:
    """The id tables both AI policies read by id (the script's and trainer_ai.c's ids are the
    same numbers in both ROMs, docs/ai/kaizo_roark.md 2), asserted by name for ``scope``'s
    games."""
    return (
        *_table("nuzline.ai.commands.MOVE_IDS", "move", scope, _AI_MOVE_IDS),
        *_table("nuzline.ai.commands.SOUND_MOVES", "move", scope, _SOUND_MOVES),
        *_table("nuzline.ai.switching.MOVE_IDS", "move", scope, _SWITCH_MOVES),
        *_table("nuzline.ai.switching._MOVES", "move", scope, _SWITCH_MOVES),
        *_table(
            "nuzline.ai.switching.PLAYER_MOVE_REFUSALS",
            "move",
            scope,
            {k: v for k, v in _SWITCH_MOVES.items() if k != "Hidden Power"},
        ),
        *_table(
            "nuzline.ai.switching.PLAYER_ABILITY_REFUSALS", "ability", scope, _ANY_SIDE_ABILITIES
        ),
        *_table(
            "nuzline.ai.policy._STATE_MOVE_IDS",
            "move",
            scope,
            {k: _SWITCH_MOVES[k] for k in ("Miracle Eye", "Roost", "Magnet Rise", "Ingrain")},
        ),
    )


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
            "Damp": 6,
            "Simple": 86,
            "Immunity": 17,
            "Magic Guard": 98,
            "Poison Heal": 90,
            "Leaf Guard": 102,
            "Hydration": 93,
            "Rock Head": 69,
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
        {
            k: _TYPES[k]
            for k in ("Poison", "Ground", "Ghost", "Steel", "Fire", "Water", "Electric", "Psychic")
        },
    ),
    Constant("nuzline.ai.switching._TYPE_FLYING", "type", 2, "Flying", _E),
    Constant("nuzline.ai.switching._TYPE_STEEL", "type", 8, "Steel", _E),
    *_table("nuzline.engine.ids.Type", "type", _E, _TYPES),
    # MOVE_STRUGGLE (generated/moves.txt line 166).
    Constant("nuzline.engine.ids.MOVE_STRUGGLE", "move", 165, "Struggle", _E),
    # MOVE_FAKE_OUT (generated/moves.txt line 253): subscript_flinch_mon.s:18.
    Constant("nuzline.engine.ids.MOVE_FAKE_OUT", "move", 252, "Fake Out", _E),
    # MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE (generated/moves.txt lines 183, 198, 204):
    # BtlCmd_TryProtection's chain check (battle_script.c:5154-5156).
    Constant("nuzline.engine.ids.MOVE_PROTECT", "move", 182, "Protect", _E),
    Constant("nuzline.engine.ids.MOVE_DETECT", "move", 197, "Detect", _E),
    Constant("nuzline.engine.ids.MOVE_ENDURE", "move", 203, "Endure", _E),
    # ABILITY_MULTITYPE (generated/abilities.txt line 122): BtlCmd_TryCamouflage
    # (battle_script.c:8001-8004).
    Constant("nuzline.engine.ids.ABILITY_MULTITYPE", "ability", 121, "Multitype", _E),
    # sTerrainMove (include/data/terrain/to_move.h): the moves Nature Power calls
    # (BtlCmd_GetTerrainMove, battle_script.c:8036-8048); its type twin TERRAIN_TYPES holds
    # ids.Type members, asserted with ids.Type.
    *_table(
        "nuzline.engine.ids.TERRAIN_MOVES",
        "move",
        _E,
        {
            "Earthquake": 89,
            "Seed Bomb": 402,
            "Rock Slide": 157,
            "Blizzard": 59,
            "Hydro Pump": 56,
            "Ice Beam": 58,
            "Tri Attack": 161,
            "Mud Bomb": 426,
            "Air Slash": 403,
        },
    ),
    # The weather code's decomp abilities (generated/abilities.txt line N is id N-1).
    *_table(
        "nuzline.engine.ids.WEATHER_ABILITY_IDS",
        "ability",
        _E,
        {
            "Cloud Nine": 13,
            "Air Lock": 76,
            "Magic Guard": 98,
            "Ice Body": 115,
        },
    ),
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
    # Quick Claw's effectParam (res/items/data/quick_claw.json) with
    # HOLD_EFFECT_SOMETIMES_PRIORITY (52): the engine stores speedRand % (100 / param) == 0
    # (battle_lib.c:1278-1286) as one bit, exact only for this param.
    Constant(
        "nuzline.engine.ids.QUICK_CLAW_PARAM", "item", 217, "Quick Claw", _E, 52, hold_param=20
    ),
    # saves/check.py: Sketch's user and the special routes outside the learnset tables
    # (Rotom's form moves, Route 228's ultimate moves, Wilma's Draco Meteor for Dragon
    # types, the Pichu egg's Volt Tackle); each route is cited there.
    Constant("nuzline.saves.check.SPECIES_SMEARGLE", "species", 235, "Smeargle", _E),
    Constant("nuzline.saves.check.SPECIES_PICHU", "species", 172, "Pichu", _E),
    Constant("nuzline.saves.check.SPECIES_ROTOM", "species", 479, "Rotom", _E),
    Constant("nuzline.saves.check.TYPE_DRAGON", "type", 16, "Dragon", _E),
    Constant("nuzline.saves.check.MOVE_DRACO_METEOR", "move", 434, "Draco Meteor", _E),
    Constant("nuzline.saves.check.MOVE_VOLT_TACKLE", "move", 344, "Volt Tackle", _E),
    *_table(
        "nuzline.saves.check.ROTOM_FORM_MOVES",
        "move",
        _E,
        {
            "-": 0,
            "Overheat": 315,
            "Hydro Pump": 56,
            "Blizzard": 59,
            "Air Slash": 403,
            "Leaf Storm": 437,
        },
    ),
    # ULTIMATE_MOVES: its species keys, then the three moves its values name.
    *_table(
        "nuzline.saves.check.ULTIMATE_MOVES",
        "species",
        _E,
        {
            "Charizard": 6,
            "Typhlosion": 157,
            "Blaziken": 257,
            "Infernape": 392,
            "Blastoise": 9,
            "Feraligatr": 160,
            "Swampert": 260,
            "Empoleon": 395,
            "Venusaur": 3,
            "Meganium": 154,
            "Sceptile": 254,
            "Torterra": 389,
        },
    ),
    *_table(
        "nuzline.saves.check.ULTIMATE_MOVES",
        "move",
        _E,
        {"Blast Burn": 307, "Hydro Cannon": 308, "Frenzy Plant": 338},
    ),
    # --- vanilla_policy: the transcribed vanilla AI scripts --------------------------
    *_table("nuzline.ai.commands.MOVE_EFFECTS", "move", _P, _TRANSCRIBED_MOVES),
    *_policy_tables(_P),
    # The flat layout's script-roll sites, by move id (the per-slot layout keys them by battle
    # effect, ai/outcome.py EFFECT_SITE_LINES).
    *_table("nuzline.ai.outcome._SITES", "move", _P, _TRANSCRIBED_MOVES),
    # The decomp's C id cases: both policies compare these by id, whatever a game names the
    # slot (Kaizo's 149 is Triple Axel and still takes MOVE_PSYWAVE's case), so they are
    # name-checked against vanilla only.
    *_table("nuzline.ai.commands.CALC_DAMAGE_MOVES", "move", _P, _CALC_DAMAGE_MOVES),
    *_table("nuzline.ai.commands.MOVE_TYPE_MOVES", "move", _P, _MOVE_TYPE_MOVES),
    # script.s:78's compare (Basic_CheckForImmunity -> Basic_CheckWaterAbsorption2): the
    # decomp tests Levitate a second time, which never matches (quirk).
    Constant("nuzline.ai.layout.VANILLA", "ability", 26, "Levitate", _P),
    # --- kaizo_policy: the vanilla transcription, read by battle effect (ai/layout.py) ---
    *_policy_tables(_K),
    # Kaizo's one changed script word (overlay14 0x022249CC, data/kaizo/rom_diff.toml).
    Constant("nuzline.ai.layout.KAIZO", "ability", 87, "Dry Skin", _K),
)

# Lists that are game data: "module.ATTR" -> the intermediate file that replaces it.
# Both names were removed from tools/effect_map.py (K2b Task 5); the lists now come from each
# game's move_lists.json by move id, and these entries record where they went.
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
    "nuzline.ai.commands.MonCondition": _AI_ENCODING + " (MON_CONDITION_*, condition.h:7-25)",
    "nuzline.ai.commands.FieldCondition": _AI_ENCODING + " (FIELD_CONDITION_*, condition.h:144)",
    "nuzline.ai.commands.AiWeather": _AI_ENCODING + " (AI_WEATHER_*, trainer_ai.c:1414-1439)",
    "nuzline.ai.commands.TypeMulti": _AI_ENCODING + " (TYPE_MULTI_*, battle.h:144-153)",
    "nuzline.ai.commands.BattleEffect": (
        "decomp battle effect ids (BATTLE_EFFECT_*, generated/move_battle_effects.txt) that the "
        "AI script compares with MoveTable.effect (IfCurrentMoveEffectEqualTo, trainer_ai.c:1441), "
        "a script Kaizo did not change apart from one word (docs/ai/kaizo_roark.md 2); each "
        "game's moves carry their own effect"
    ),
    "nuzline.ai.commands.NO_DAMAGE_CALC_EFFECTS": (
        "decomp battle effect ids of trainer_ai.c's sNoDamageCalcMoveEffects (:31-46), compared "
        "with MoveTable.effect by the AI's C code, which Kaizo did not change "
        "(docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.commands.RISKY_EFFECTS": (
        "decomp battle effect ids of script.s's Risky_RiskyEffects (:6536-6562), compared with "
        "MoveTable.effect (IfLoadedNotInTable, trainer_ai.c:876-891) by a script Kaizo did not "
        "change apart from one word (docs/ai/kaizo_roark.md 2); each game's moves carry their "
        "own effect"
    ),
    "nuzline.ai.commands.ALT_POWER_EFFECTS": (
        "decomp battle effect ids of trainer_ai.c's sAltPowerMoveEffects (:48-61), compared "
        "with MoveTable.effect by the AI's C code, which Kaizo did not change "
        "(docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.outcome.EFFECT_SITE_LINES": (
        "decomp battle effect ids (keys) and script.s line numbers (values): the per-slot AI "
        "layout's script-roll sites, read through MoveTable.effect by IfCurrentMoveEffectEqualTo "
        "(trainer_ai.c:1441), whose script Kaizo did not change apart from one word "
        "(docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.commands.ReadKind": "kind of a loose-oracle read (tests only), not a game id",
    "nuzline.ai.outcome.KAIZO_EXPERT_LABELS": (
        "decomp battle effect ids (keys) and the script.s labels and lines of their Expert_Main "
        "entries (:1628-1808), a script Kaizo did not change apart from one word "
        "(docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.outcome.KAIZO_EXPERT_TRANSCRIBED": (
        "decomp battle effect ids whose Expert_Main label (script.s:1628-1808) the AI "
        "transcribes; MoveTable.effect is read by a script Kaizo did not change apart from one "
        "word (docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.policy._LOOSE_SHARE": (
        "tie-break probabilities P(RandNext() % n == i) (trainer_ai.c:338), not game ids"
    ),
    "nuzline.ai.outcome.SITE_N": (
        "script.s line numbers and IfRandomLessThan thresholds of the AI script, not game ids"
    ),
    "nuzline.ai.switching.SwitchSite": "kind of an AI switch draw (roark.md 11.1), not a game id",
    "nuzline.ai.switching._CANDIDATE_TARGETS": (
        "party slot of each AI switch candidate (-1: keep the battler), an engine layout"
    ),
    "tools.effect_map.DECOMP_EFFECT_TO_ID": _KEYED_BY_EFFECT,
    "tools.effect_map._EFFECTS": _KEYED_BY_EFFECT,
    "tools.effect_map.DECOMP_EFFECT_NAMES": _KEYED_BY_EFFECT,
    "tools.effect_map.CERTAIN_SECONDARY_EFFECTS": _KEYED_BY_EFFECT,
    "tools.effect_map.RESIST_BERRY_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off the vanilla subscript_type_resist_berry.s: "
        "each game's items carry their own hold effect (build_tables looks theirs up here), "
        "and a game's script changes are rom_diff.toml entries (Ruling 13)"
    ),
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

# Game id -> the AI policy its trainers' scripts are transcribed as, one of AI_POLICIES (None:
# none yet, so coverage refuses every trainer of that game, spec 5.10).
POLICY: dict[str, str | None] = {
    "platinum_us": "vanilla",
    "platinum_kaizo_nightmare_final": "kaizo",
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
    if c.hold_effect is None and c.hp_restored is None and c.hold_param is None:
        return lines
    if c.id >= len(items):
        return [*lines, f"{where}: item {c.id} is missing from this game's items.json"]
    row = items[c.id]
    if c.hold_effect is not None and row.get("hold_effect") != c.hold_effect:
        lines.append(
            f"{where}: {c.name} has hold effect {row.get('hold_effect')} in this game, "
            f"expected {c.hold_effect}"
        )
    if c.hold_param is not None and row.get("hold_param") != c.hold_param:
        lines.append(
            f"{where}: {c.name} has hold param {row.get('hold_param')} in this game, "
            f"expected {c.hold_param}"
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
    every game, a policy's (``"vanilla_policy"``, ``"kaizo_policy"``) only for a game whose
    ``POLICY`` it is."""
    policy = POLICY.get(game_id)
    lines: list[str] = []
    for c in CONSTANTS:
        if c.scope == "engine" or (policy is not None and c.scope == f"{policy}_policy"):
            lines += _check(c, names, items)
    return lines
