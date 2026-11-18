"""Every game id the engine and the AI hard-code, classified per game (spec §7.2).

Code names some decomp ids directly (``MOVE_STRUGGLE = 165``, the AI's ability tables). A
game that moves or renames those slots would silently change what that code means, so
each such constant is listed here and checked against the game's data when the game is
loaded (``nuzline.data.games.load_game``):

* ``CONSTANTS``: the ids, each with the name (and, for items, the hold effect or hold param)
  it must have. ``scope`` ``"engine"`` is asserted for every game;
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
# ai/switching.py MOVE_IDS; _MOVES is these less Hidden Power, Roost and Ingrain
# (_SWITCH_REFUSED): moves whose decomp handling the switch check does not model. Roost and
# Ingrain are read since E1 prep batch P2 (switching.defender_chart_flags).
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
_SWITCH_REFUSED = {
    k: v for k, v in _SWITCH_MOVES.items() if k not in ("Hidden Power", "Roost", "Ingrain")
}
# ai/switching.py _ANY_SIDE_ABILITIES (ApplyTypeChart / CalcEffectiveness, trainer_ai.c).
_ANY_SIDE_ABILITIES = {"Levitate": 26, "Scrappy": 113, "Normalize": 96}
# ai/commands.py MOVE_IDS: Basic_Main's OHKO checks (script.s:57-58), Expert_Pursuit's
# U-turn (:3730), Expert_StealthRock's Roar and Whirlwind (:6276-6277), Expert_Curse's Gyro
# Ball and Trick Room (:3472-3473, read from K3c Task 6), and the Basic_CheckSoundproof list
# (:121-131).
_AI_MOVE_IDS = {
    "Thunder Wave": 86,  # Basic_CheckCannotParalyze (script.s:620)
    "Psych Up": 244,  # Expert_Swagger (script.s:2852)
    "Icy Wind": 196,  # Expert_SpeedDownOnHit (script.s:2345-2347)
    "Rock Tomb": 317,
    "Mud Shot": 341,
    "Fissure": 90,
    "Horn Drill": 32,
    "U-turn": 369,
    "Roar": 46,
    "Whirlwind": 18,
    "Gyro Ball": 360,
    "Trick Room": 433,
    # Expert_Counter / _MirrorCoat (:3125, :3882), Expert_Hail (:4105), Expert_Protect
    # (:3536-3537).
    "Counter": 68,
    "Mirror Coat": 243,
    "Blizzard": 59,
    "Feint": 364,
    "Shadow Force": 467,
    "Focus Punch": 264,
    # BatonPass_Main (script.s:6581-6588, :6615-6618), E1 prep batch P3.
    "Swords Dance": 14,
    "Dragon Dance": 349,
    "Calm Mind": 347,
    "Nasty Plot": 417,
    "Baton Pass": 226,
    "Protect": 182,
    "Detect": 197,
    **{k: v for k, v in _SOUND_MOVES.items() if k != "Roar"},
}
# ai/commands.py CALC_DAMAGE_MOVES: TrainerAI_CalcDamage's id cases the AI damage does not
# model, refused (trainer_ai.c:2885-3071; MOVE_* of generated/moves.txt, line N is id N-1).
# Return, Frustration, Low Kick, Grass Knot, Dragon Rage, SonicBoom, Seismic Toss and Night
# Shade are modelled since E1 batch 13, Hidden Power since batch 14, Psywave (149) and
# Magnitude (222) with their draws since E1 ruling G-39 (engine.damage.AI_DRAWN_MOVES).
_CALC_DAMAGE_MOVES = {
    "Natural Gift": 363,
    "Judgment": 449,
    "Gyro Ball": 360,
}
# ai/commands.py MOVE_TYPE_MOVES: TrainerAI_MoveType's id cases (trainer_ai.c:3132-3222).
# Hidden Power's is modelled (engine.damage.ai_move_type, E1 batch 14).
_MOVE_TYPE_MOVES = {"Natural Gift": 363, "Judgment": 449, "Weather Ball": 311}
# ai/commands.py COPYCAT_ENCOURAGED_MOVES: Expert_Copycat_EncouragedMoves (script.s:5582-5630).
_COPYCAT_ENCOURAGED_MOVES = {
    "Sleep Powder": 79, "Lovely Kiss": 142, "Spore": 147, "Hypnosis": 95, "Sing": 47,
    "GrassWhistle": 320, "Shadow Punch": 325, "Sand-Attack": 28, "SmokeScreen": 108,
    "Toxic": 92, "Guillotine": 12, "Horn Drill": 32, "Fissure": 90, "Sheer Cold": 329,
    "Cross Chop": 238, "Aeroblast": 177, "Confuse Ray": 109, "Sweet Kiss": 186,
    "Screech": 103, "Cotton Spore": 178, "Scary Face": 184, "Fake Tears": 313,
    "Metal Sound": 319, "Thunder Wave": 86, "Glare": 137, "PoisonPowder": 77,
    "Shadow Ball": 247, "DynamicPunch": 223, "Hyper Beam": 63, "ExtremeSpeed": 245,
    "Thief": 168, "Covet": 343, "Attract": 213, "Swagger": 207, "Torment": 259,
    "Flatter": 260, "Trick": 271, "Superpower": 276, "Skill Swap": 285,
    "Psycho Shift": 375, "Power Swap": 384, "Guard Swap": 385, "Sucker Punch": 389,
    "Heart Swap": 391, "Switcheroo": 415, "Captivate": 445, "Dark Void": 464,
}  # fmt: skip


def _policy_tables(scope: str) -> tuple[Constant, ...]:
    """The id tables both AI policies read by id (the script's and trainer_ai.c's ids are the
    same numbers in both ROMs, docs/ai/kaizo_roark.md 2), asserted by name for ``scope``'s
    games."""
    return (
        *_table("nuzline.ai.commands.MOVE_IDS", "move", scope, _AI_MOVE_IDS),
        *_table("nuzline.ai.commands.SOUND_MOVES", "move", scope, _SOUND_MOVES),
        *_table("nuzline.ai.switching.MOVE_IDS", "move", scope, _SWITCH_MOVES),
        # Hidden Power's AI type is modelled on both sides since E1 batch 14: in MOVE_IDS
        # (the bench's IV type), refused in neither.
        *_table("nuzline.ai.switching._MOVES", "move", scope, _SWITCH_REFUSED),
        *_table("nuzline.ai.switching.PLAYER_MOVE_REFUSALS", "move", scope, _SWITCH_REFUSED),
        *_table(
            "nuzline.ai.switching.PLAYER_ABILITY_REFUSALS", "ability", scope, _ANY_SIDE_ABILITIES
        ),
        *_table(
            "nuzline.ai.policy._STATE_MOVE_IDS",
            "move",
            scope,
            {k: _SWITCH_MOVES[k] for k in ("Miracle Eye", "Magnet Rise")},
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
            "Insomnia": 15,
            "Vital Spirit": 72,
            "Own Tempo": 20,
            "Limber": 7,
            "Oblivious": 12,
            "Water Veil": 41,
            "Hyper Cutter": 52,
            "Speed Boost": 3,
            "No Guard": 99,
            "Swift Swim": 33,
            "Rain Dish": 44,
            "Ice Body": 115,
            "Sturdy": 5,
            "Flower Gift": 122,
            "Solar Power": 94,
            "Suction Cups": 21,
            "Stall": 100,
            "Sticky Hold": 60,
            "Keen Eye": 51,
        },
    ),
    *_table(
        "nuzline.ai.switching.ABILITY_IDS",
        "ability",
        _E,
        {
            **_ANY_SIDE_ABILITIES,
            "Wonder Guard": 25,
            "Mold Breaker": 104,
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
            for k in (
                "Poison",
                "Ground",
                "Ghost",
                "Steel",
                "Fire",
                "Grass",
                "Water",
                "Electric",
                "Psychic",
                "Normal",
                "Fighting",
                "Flying",
                "Rock",
                "Bug",
                "Ice",
                "Dragon",
                "Dark",
            )
        },
    ),
    Constant("nuzline.ai.switching._TYPE_FLYING", "type", 2, "Flying", _E),
    Constant("nuzline.ai.switching._TYPE_STEEL", "type", 8, "Steel", _E),
    *_table("nuzline.engine.ids.Type", "type", _E, _TYPES),
    # MOVE_STRUGGLE (generated/moves.txt line 166).
    Constant("nuzline.engine.ids.MOVE_STRUGGLE", "move", 165, "Struggle", _E),
    # MOVE_FAKE_OUT (generated/moves.txt line 253): subscript_flinch_mon.s:18.
    Constant("nuzline.engine.ids.MOVE_FAKE_OUT", "move", 252, "Fake Out", _E),
    # MOVE_SNORE, MOVE_SLEEP_TALK (generated/moves.txt lines 174, 215): the sleep check's
    # exceptions (battle_controller_player.c:2455), E2 batch 4.
    Constant("nuzline.engine.ids.MOVE_SNORE", "move", 173, "Snore", _E),
    Constant("nuzline.engine.ids.MOVE_SLEEP_TALK", "move", 214, "Sleep Talk", _E),
    # MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE (generated/moves.txt lines 183, 198, 204):
    # BtlCmd_TryProtection's chain check (battle_script.c:5154-5156).
    Constant("nuzline.engine.ids.MOVE_PROTECT", "move", 182, "Protect", _E),
    Constant("nuzline.engine.ids.MOVE_DETECT", "move", 197, "Detect", _E),
    Constant("nuzline.engine.ids.MOVE_ENDURE", "move", 203, "Endure", _E),
    # MOVE_HEAL_BELL (generated/moves.txt line 216): BtlCmd_TryPartyStatusRefresh's compare
    # (battle_script.c:5035), E2 batch 12.
    Constant("nuzline.engine.ids.MOVE_HEAL_BELL", "move", 215, "Heal Bell", _E),
    # MOVE_IMPRISON (generated/moves.txt line 287): DecrementPP's Pressure count for it
    # (battle_controller_player.c:2239-2240), ruling G-8.
    Constant("nuzline.engine.ids.MOVE_IMPRISON", "move", 286, "Imprison", _E),
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
    # E2 batch 3: the decomp id of every supported AbilityId (damage.decomp_abilities of a
    # battler that traced), compared by value (its keys are engine ids).
    *_table(
        "nuzline.engine.ids.ABILITY_DECOMP_IDS",
        "ability",
        _E,
        {
            "-": 0,
            "Sturdy": 5,
            "Rock Head": 69,
            "Mold Breaker": 104,
            "Blaze": 66,
            "Torrent": 67,
            "Overgrow": 65,
            "Intimidate": 22,
            "Rivalry": 79,
            "Keen Eye": 51,
            "Simple": 86,
            "Unaware": 109,
            "Natural Cure": 30,
            "Poison Point": 38,
            "Inner Focus": 39,
            "Static": 9,
            "Guts": 62,
            "Steadfast": 80,
            "Insomnia": 15,
            "Cloud Nine": 13,
            "Compoundeyes": 14,
            "Shed Skin": 61,
            "Sand Veil": 8,
            "Snow Cloak": 81,
            "Swift Swim": 33,
            "Rough Skin": 24,
            "Flame Body": 49,
            "Limber": 7,
            "Adaptability": 91,
            "Solid Rock": 116,
            "Cute Charm": 56,
            "Gluttony": 82,
            "Reckless": 120,
            "Technician": 101,
            "Truant": 54,
            "Dry Skin": 87,
            "Lightningrod": 31,
            "No Guard": 99,
            "Serene Grace": 32,
            "Drought": 70,
            "Honey Gather": 118,
            "Levitate": 26,
            "Quick Feet": 95,
            "Scrappy": 113,
            "Shell Armor": 75,
            "Battle Armor": 4,
            "Shield Dust": 19,
            "Solar Power": 94,
            "Super Luck": 105,
            "Tangled Feet": 77,
            "Thick Fat": 47,
            "Tinted Lens": 110,
            "Trace": 36,
            "Drizzle": 2,
            "Sand Stream": 45,
            "Snow Warning": 117,
            "Filter": 111,
            "Magic Guard": 98,
            "Sniper": 97,
            # E2 batch 9.
            "Chlorophyll": 34,
            "Huge Power": 37,
            "Pure Power": 74,
            "Swarm": 68,
            "Unburden": 84,
            "Flower Gift": 122,
            "Volt Absorb": 10,
            "Water Absorb": 11,
            "Rain Dish": 44,
            "Clear Body": 29,
            "Hyper Cutter": 52,
            "Aftermath": 106,
            "Poison Heal": 90,
            "Bad Dreams": 123,
            "Arena Trap": 71,
            "Shadow Tag": 23,
            "Magnet Pull": 42,
            "Pressure": 46,
            "Leaf Guard": 102,
            "Synchronize": 28,
            "Wonder Guard": 25,
            "Download": 88,
            "Ice Body": 115,
            "White Smoke": 73,
            "Vital Spirit": 72,
            "Speed Boost": 3,
            "Iron Fist": 89,
        },
    ),
    # ChooseTraceTarget's untraceable abilities (battle_lib.c:7750-7775) and its Forecast.
    *_table(
        "nuzline.engine.ids.UNTRACEABLE_ABILITIES",
        "ability",
        _E,
        {"Forecast": 59, "Trace": 36, "Multitype": 121},
    ),
    Constant("nuzline.engine.ids.ABILITY_FORECAST", "ability", 59, "Forecast", _E),
    # The weather code's decomp abilities (generated/abilities.txt line N is id N-1).
    *_table(
        "nuzline.engine.ids.WEATHER_ABILITY_IDS",
        "ability",
        _E,
        {
            "Cloud Nine": 13,
            "Air Lock": 76,
            "Magic Guard": 98,
        },
    ),
    # Yawn's decomp abilities (subscript_yawn.s:5-9, :14; subscript_fall_asleep.s:7-14), E2
    # batch 8.
    *_table(
        "nuzline.engine.ids.YAWN_ABILITY_IDS",
        "ability",
        _E,
        {
            "Insomnia": 15,
            "Soundproof": 43,
            "Vital Spirit": 72,
            "Leaf Guard": 102,
        },
    ),
    # E2 batch 12's decomp abilities: Captivate's Oblivious (effect_script_0265.s:5), Rest's
    # Insomnia, Vital Spirit, Soundproof (subscript_rest.s:7-10), Heal Bell's Soundproof
    # (battle_script.c:5038).
    *_table(
        "nuzline.engine.ids.STAT_MOVE_ABILITY_IDS",
        "ability",
        _E,
        {
            "Oblivious": 12,
            "Insomnia": 15,
            "Soundproof": 43,
            "Vital Spirit": 72,
        },
    ),
    # The status orbs' decomp abilities (subscript_badly_poison.s:14, subscript_burn.s:6, :9).
    *_table(
        "nuzline.engine.ids.ITEM_ABILITY_IDS",
        "ability",
        _E,
        {
            "Immunity": 17,
            "Water Veil": 41,
            "Leaf Guard": 102,
            # BtlCmd_TryKnockOff's Sticky Hold (battle_script.c:6614), B10-6.
            "Sticky Hold": 60,
            "Own Tempo": 20,
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
    # The crit species items' species (BattleSystem_CalcCriticalMulti, battle_lib.c:7127-7128).
    Constant("nuzline.engine.ids.SPECIES_CHANSEY", "species", 113, "Chansey", _E),
    Constant("nuzline.engine.ids.SPECIES_FARFETCHD", "species", 83, "Farfetch’d", _E),
    # The species of BattleSystem_CalcMoveDamage's item block (battle_lib.c:6730-6774).
    *(
        Constant(f"nuzline.engine.ids.SPECIES_{name.upper()}", "species", i, name, _E)
        for name, i in (
            ("Pikachu", 25),
            ("Cubone", 104),
            ("Marowak", 105),
            ("Ditto", 132),
            ("Clamperl", 366),
            ("Latias", 380),
            ("Latios", 381),
            ("Dialga", 483),
            ("Palkia", 484),
            ("Giratina", 487),
        )
    ),
    # Items (generated/items.txt, line N is id N-1); prep looks Heart Scale up by name.
    Constant("nuzline.prep.problem.HEART_SCALE", "item", 93, "Heart Scale", _E),
    Constant("nuzline.engine.coverage.ITEM_GRISEOUS_ORB", "item", 112, "Griseous Orb", _E),
    # Trace never fires for a Griseous Orb holder (battle_lib.c:3728).
    Constant("nuzline.engine.ids.ITEM_GRISEOUS_ORB", "item", 112, "Griseous Orb", _E),
    # saves/pkm.py: the Luxury Ball friendship bonus (PokemonSpec.friend_bonus).
    Constant("nuzline.saves.pkm.ITEM_LUXURY_BALL", "item", 11, "Luxury Ball", _E),
    # Hold effects (generated/item_hold_effects.txt, line N is id N-1):
    # HOLD_EFFECT_POWER_UP_SE (96) and HOLD_EFFECT_SPEED_DOWN_GROUNDED (106).
    Constant("nuzline.ai.commands.ITEM_EXPERT_BELT", "item", 268, "Expert Belt", _E, 96),
    # IfHeldItemEqualTo's items (Basic_CheckMetalBurst, Expert_ChargeTurnNoInvuln).
    Constant("nuzline.ai.commands.ITEM_SHINY_STONE", "item", 107, "Shiny Stone", _E),
    Constant("nuzline.ai.switching.ITEM_IRON_BALL", "item", 278, "Iron Ball", _E, 106),
    # HOLD_EFFECT_CHARGE_SKIP (99): Expert_ChargeTurn*'s IfHeldItemEqualTo (script.s:3956, B10-12).
    Constant("nuzline.ai.commands.ITEM_POWER_HERB", "item", 271, "Power Herb", _E, 99),
    # TrainerAI_ShouldUseItem's by-id test (trainer_ai.c:4099).
    Constant("nuzline.engine.ids.ITEM_FULL_RESTORE", "item", 23, "Full Restore", _E),
    # Quick Claw's effectParam (res/items/data/quick_claw.json) with
    # HOLD_EFFECT_SOMETIMES_PRIORITY (52): the engine stores speedRand % (100 / param) == 0
    # (battle_lib.c:1278-1286) as one bit, exact only for this param.
    Constant(
        "nuzline.engine.ids.QUICK_CLAW_PARAM", "item", 217, "Quick Claw", _E, 52, hold_param=20
    ),
    # Focus Band's effectParam (res/items/data/focus_band.json) with HOLD_EFFECT_MAYBE_ENDURE
    # (65): the engine stores RandNext() % 100 < param (bcp.c:3387, battle_script.c:9110) as
    # one bit (Outcome.focus_band / future_band), exact only for this param (ruling G-1).
    Constant(
        "nuzline.engine.ids.FOCUS_BAND_PARAM", "item", 230, "Focus Band", _E, 65, hold_param=10
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
    # The flat layout's script-roll sites, by move id (the per-slot layout keys its reads by
    # order on the path, ai/outcome.py).
    *_table("nuzline.ai.outcome._SITES", "move", _P, _TRANSCRIBED_MOVES),
    # The decomp's C id cases: both policies compare these by id, whatever a game names the
    # slot (Kaizo's 149 is Triple Axel and still takes MOVE_PSYWAVE's case), so they are
    # name-checked against vanilla only.
    *_table("nuzline.ai.commands.CALC_DAMAGE_MOVES", "move", _P, _CALC_DAMAGE_MOVES),
    *_table("nuzline.ai.commands.MOVE_TYPE_MOVES", "move", _P, _MOVE_TYPE_MOVES),
    *_table("nuzline.ai.commands.COPYCAT_ENCOURAGED_MOVES", "move", _P, _COPYCAT_ENCOURAGED_MOVES),
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
_CODE_MOVE_IDS = (
    "move ids compared in battle code (battle_script.c:4918-4922, battle_lib.c:3032-3045, "
    ":7152-7178), not move data: the Kaizo ROM leaves that code unchanged (rom_diff.md), so the "
    "ids hold in both games whatever a slot is named (Kaizo renamed Snatch, Covet, Trick and "
    "Switcheroo's slots, which stay excluded); tests/tools/test_effect_map.py checks them "
    "against the decomp"
)

# "module.ATTR" -> why it is not a game id table.
EXEMPT: dict[str, str] = {
    "nuzline.engine.ids.MoveTarget": (
        "engine index order of the decomp's RANGE_* (generated/move_ranges.txt); "
        + _ENGINE_ENCODING
    ),
    "nuzline.engine.ids.EffectId": _ENGINE_ENCODING,
    "nuzline.engine.ids.SECRET_POWER_EFFECTS": (
        "sTerrainSideEffect (include/data/terrain/to_secondary_effect.h) as engine EffectId / "
        "Stat / Status members, indexed by the decomp TERRAIN_* id; " + _ENGINE_ENCODING
    ),
    "nuzline.engine.ids.Status": _ENGINE_ENCODING,
    "nuzline.engine.ids.Stat": "engine index order of every stat vector; " + _ENGINE_ENCODING,
    "nuzline.engine.ids.MoveFlag": _ENGINE_ENCODING,
    "nuzline.engine.ids.SemiKind": _ENGINE_ENCODING,
    "nuzline.engine.ids.Weather": _ENGINE_ENCODING,
    "nuzline.engine.ids.ItemId": _ENGINE_ENCODING,
    "nuzline.engine.ids.ItemParam": "columns of tables.item_party_param; " + _ENGINE_ENCODING,
    "nuzline.engine.ids.ItemAiCategory": _ENGINE_ENCODING,
    "nuzline.engine.ids.AbilityId": _ENGINE_ENCODING,
    "nuzline.ai.commands.Battler": _AI_ENCODING + " (AI_BATTLER_*)",
    "nuzline.ai.commands.LoadType": _AI_ENCODING + " (LOAD_*, trainer_ai.c:937)",
    "nuzline.ai.commands.SideCondition": _AI_ENCODING + " (SIDE_CONDITION_*)",
    "nuzline.ai.commands.MonCondition": _AI_ENCODING + " (MON_CONDITION_*, condition.h:7-25)",
    "nuzline.ai.commands.FieldCondition": _AI_ENCODING + " (FIELD_CONDITION_*, condition.h:144)",
    "nuzline.ai.commands.AiWeather": _AI_ENCODING + " (AI_WEATHER_*, trainer_ai.c:1414-1439)",
    "nuzline.ai.commands.VolatileCondition": _AI_ENCODING
    + " (VOLATILE_CONDITION_* masks, include/constants/battle/condition.h)",
    "nuzline.ai.commands.MoveEffectMask": _AI_ENCODING
    + " (MOVE_EFFECT_* masks, include/constants/battle/moves.h)",
    "nuzline.ai.commands.Gender": _AI_ENCODING + " (GENDER_*, generated/genders.txt)",
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
    "nuzline.ai.commands.ENCORE_ENCOURAGED_EFFECTS": (
        "decomp battle effect ids of script.s's Expert_Encore_EncouragedMoveEffects (:3202-3285), "
        "compared with MoveTable.effect (IfLoadedNotInTable, trainer_ai.c:876-891) by a script "
        "Kaizo did not change apart from one word (docs/ai/kaizo_roark.md 2); each game's moves "
        "carry their own effect"
    ),
    "nuzline.ai.commands.SETUP_FIRST_TURN_EFFECTS": (
        "decomp battle effect ids of script.s's SetupFirstTurn_SetupEffects (:6435-6497), "
        "compared with MoveTable.effect (IfLoadedNotInTable, trainer_ai.c:876-891) by a script "
        "Kaizo did not change apart from one word (docs/ai/kaizo_roark.md 2); each game's moves "
        "carry their own effect"
    ),
    "nuzline.ai.commands.HARRASSMENT_EFFECTS": (
        "decomp battle effect ids of script.s's Harrassment_Effects (:8036-8071), compared with "
        "MoveTable.effect (IfLoadedNotInTable, trainer_ai.c:876-891) by a script Kaizo did not "
        "change apart from one word (docs/ai/kaizo_roark.md 2); each game's moves carry their "
        "own effect"
    ),
    "nuzline.ai.commands.CHECK_HP_TABLES": (
        "decomp battle effect ids of script.s's six CheckHP_*Discourage* tables (:7748-7977), "
        "compared with MoveTable.effect (IfLoadedInTable, trainer_ai.c:858-874) by a script "
        "Kaizo did not change apart from one word (docs/ai/kaizo_roark.md 2); each game's moves "
        "carry their own effect"
    ),
    "nuzline.ai.commands.ALT_POWER_EFFECTS": (
        "decomp battle effect ids of trainer_ai.c's sAltPowerMoveEffects (:48-61), compared "
        "with MoveTable.effect by the AI's C code, which Kaizo did not change "
        "(docs/ai/kaizo_roark.md 2); each game's moves carry their own effect"
    ),
    "nuzline.ai.outcome.KAIZO_TRANSCRIBED": (
        "decomp battle effect ids: the effects whose AI script paths the per-slot layout "
        "transcribes, read through MoveTable.effect by IfCurrentMoveEffectEqualTo "
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
    "tools.effect_map.MULTI_TURN_EFFECTS": _KEYED_BY_EFFECT,
    "tools.effect_map.CHARGE_FLAG_EFFECTS": _KEYED_BY_EFFECT,  # B10-12
    "tools.effect_map.SEMI_HIT_EFFECTS": _KEYED_BY_EFFECT,  # B10-12
    "tools.effect_map.SCRIPT_FLAG_EFFECTS": _KEYED_BY_EFFECT,  # B13-1, B13-2
    # E2 batch 5: move ids the battle code compares against (BtlCmd_TrySleepTalk,
    # Move_IsInvoker, sCannotMetronomeMoves), turned into MoveFlag bits at table build.
    "tools.effect_map.INVOKER_MOVES": _CODE_MOVE_IDS,
    "tools.effect_map.SLEEP_TALK_EXCLUDED_MOVES": _CODE_MOVE_IDS,
    "tools.effect_map.METRONOME_EXCLUDED_MOVES": _CODE_MOVE_IDS,
    # Move_CanBeEncored's sCannotEncoreMoves and Move_IsInvoker as the engine reads them (B10-9,
    # B10-11).
    "nuzline.engine.ids.CANNOT_ENCORE_MOVES": _CODE_MOVE_IDS,
    "nuzline.engine.ids.INVOKER_MOVES": _CODE_MOVE_IDS,
    "tools.effect_map.RESIST_BERRY_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off the vanilla subscript_type_resist_berry.s: "
        "each game's items carry their own hold effect (build_tables looks theirs up here), "
        "and a game's script changes are rom_diff.toml entries (Ruling 13)"
    ),
    "tools.effect_map.TYPE_BOOST_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off the vanilla sTypeBoostingItems "
        "(battle_lib.c:6514): each game's items carry their own hold effect (build_tables "
        "looks theirs up here), and a game's code changes are rom_diff.toml entries"
    ),
    "nuzline.engine.damage.PLATE_TYPES": (
        "keyed by decomp hold effect id, read off the vanilla effect_script_0268.s (Judgment, "
        "B13-12; unchanged in Kaizo's be_seq): the engine reads each game's own item hold "
        "effect (Tables.item_hold_effect); tests/engine/test_batch13.py reads it back"
    ),
    "tools.effect_map.FLAVOR_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off BattleSystem_TriggerHeldItem's flavor berries "
        "(battle_lib.c:4698-4771, ruling G-4): each game's items carry their own hold effect "
        "(build_tables looks theirs up here), and a game's code changes are rom_diff.toml entries"
    ),
    "nuzline.engine.ids.Flavor": "enum Flavor (include/constants/flavor.h); " + _ENGINE_ENCODING,
    "nuzline.engine.ids.NATURE_DISLIKED_FLAVOR": (
        "sNatureFlavorAffinities (src/pokemon.c:61-237) indexed by decomp nature id: a code table, "
        "not item or move data; tests/engine/test_e1_gardenia.py reads it back from the decomp"
    ),
    "tools.effect_map.PINCH_STAT_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off BattleSystem_TriggerHeldItem's stat pinch "
        "berries (battle_lib.c:4773-4836): each game's items carry their own hold effect "
        "(build_tables looks theirs up here), and a game's code changes are rom_diff.toml entries"
    ),
    "tools.effect_map.WEATHER_ROCK_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off the weather moves' scripts (effect_script_0115.s, "
        "_0136.s, _0137.s, _0164.s): each game's items carry their own hold effect (build_tables "
        "looks theirs up here), and a game's script changes are rom_diff.toml entries"
    ),
    "tools.effect_map.CRIT_HOLD_EFFECTS": (
        "keyed by decomp hold effect id, read off BattleSystem_CalcCriticalMulti "
        "(battle_lib.c:7123-7128); its species values are ids.SPECIES_CHANSEY / _FARFETCHD, "
        "checked by name: each game's items carry their own hold effect (build_tables looks "
        "theirs up here), and a game's code changes are rom_diff.toml entries"
    ),
    "tools.effect_map.HOLD_EFFECT_TO_ID": (
        "keyed by decomp hold effect id: engine semantics of the vanilla item code; each "
        "game's items carry their own hold effect (build_tables looks theirs up here), and a "
        "game's code changes are rom_diff.toml entries (Ruling 13)"
    ),
    "tools.effect_map.DECOMP_ABILITY_TO_ID": (
        "keyed by ability name: engine semantics of the vanilla ability handling; a game's "
        "changes to it are rom_diff.toml entries (Ruling 13)"
    ),
    "tools.effect_map.KAIZO_EFFECTS": (
        "keyed by the decomp effect id of a be_seq member the Kaizo ROM changed; it is used "
        "only for Kaizo and only for a member rom_diff.toml marks modelled (Ruling 13, B9-4)"
    ),
    "tools.effect_map.KAIZO_FLAG_EFFECTS": (
        "keyed by the decomp effect id of a be_seq member the Kaizo ROM changed: its MoveFlag "
        "bits, used only for Kaizo and only with its KAIZO_EFFECTS entry (Ruling 13, B13-3)"
    ),
    "tools.effect_map.KAIZO_THUNDER_EFFECTS": (
        "decomp effect ids the Kaizo ROM's patched CheckMoveHitAccuracy / CheckMoveHitOverrides "
        "compare beside BATTLE_EFFECT_THUNDER; used only for Kaizo, from its modelled "
        "rom_diff.toml overlay16 entries (E1 Batch C, G-50)"
    ),
    "tools.effect_map.DIVISOR_PARAM_MIN": (
        "keyed by engine ItemId, whose ints are the smallest hold params each item may divide "
        "by (battle_lib.c:4778, :1294, :5330-5336; bcp.c:4736-4752), not game ids; each "
        "game's own params are compared with them at table build"
    ),
    "tools.effect_map.UNSUPPORTED_MOVES": _KEYED_BY_NAME,
    "tools.effect_map.DECOMP_ITEM_TO_ID": (
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
    if c.hold_effect is None and c.hold_param is None:
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
