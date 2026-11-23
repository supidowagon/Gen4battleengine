"""Coverage registry: which mechanics the engine and the AI implement, and a per-fight
check that refuses a fight using anything else (spec §3: refuse loudly and name what
is missing; never approximate silently).

``Support`` grades a mechanic: ``EXACT`` is modelled as the decomp does it,
``SAMPLING_ONLY`` would be right in distribution only, ``NONE`` is not implemented. A
mechanic absent from a registry dict is ``NONE``.

**The sampling-only path is unimplemented.** Spec sections 4.2 and 10 reserve
``SAMPLING_ONLY`` for a trainer whose AI draws are too many to enumerate exactly: the
registry would mark that trainer's AI sampling-only, Stage A would keep ranking it by
``sample_ai_outcome`` alone, and Stage B would say so in its report. Nothing assigns the
grade today: no registry entry is ``SAMPLING_ONLY`` (``tests/engine/test_coverage.py``
guards this), and ``CoverageReport.sampling_only`` is therefore always empty. Each game's
AI layout (``ai.layout``) has a width bounded at import: the vanilla flat layout's 3840
(``ai.outcome.ENUMERATION_WIDTH``) against ``MAX_ENUMERATED`` (4096), and the Kaizo
per-slot layout's factored width (``ai.policy.ai_width``, at most 8237) against
``WIDTH_CAP`` (``ai.policy._check_width``). Roark's AI is exact in both games, so the grade
is not needed for this milestone. The first trainer whose draws do not fit its game's
layout needs, in its plan: a wider layout or a per-trainer width, a registry entry that
grades the trainer's AI ``SAMPLING_ONLY`` when its width exceeds the budget, and the
Stage A and B handling of a sampling-only opponent, before anything can be marked as such.

The registries list exactly what plan 1a (Tasks 8a-8e: move effects, abilities, held
items) and plan 1b (Task 2b: trainer items; Task 3: Roark's AI flags) implement:

* move effects: every ``ids.EffectId`` but ``UNSUPPORTED`` (a Ghost-type user's Curse is
  modelled since E2 batch 8, B8-5), except a physical ``SELF_DESTRUCT`` move (Selfdestruct,
  Explosion) facing a Pokemon whose Defense is below ``HALVED_DEFENSE_MIN``
  (``halved_defense_line``: the halved Defense at -6 makes the damage divisor 0), refused per move on both sides
  here, and in the pool and prep both ways: a player candidate's move against the
  opponent's lowest Defense, and an opponent's move against a candidate's own Defense.
  ``effect_script_0007.s``'s Damp branch is never met: Damp is an unsupported ability, so
  a Damp holder on either side is refused. ``subscript_infatuate.s``'s Oblivious branch
  (Attract, Cute Charm) and Oblivious' cure are modelled (E1 batch H, H-2); Mental
  Herb's cure and Destiny Knot are modelled (E1 batch H, H-4). Reflect and Light Screen are modelled
  (``EffectId.REFLECT``, ``EffectId.LIGHT_SCREEN``, ``effects._screen``), and so are the
  four weather moves (``EffectId.WEATHER``, ``effects._weather``). Moves the tables map to
  ``UNSUPPORTED`` include Tailwind and Trick Room;
* abilities, held items, trainer items: every member but ``UNSUPPORTED``. Held items are
  mapped by name or by decomp hold effect (``tools.effect_map.held_item_ids``), so an item with
  no hold effect is ``ItemId.NONE`` (or ``NO_EFFECT_BERRY``) and supported;
* AI flags, per AI policy (``IMPLEMENTED_AI_FLAGS``): vanilla bits 0-2 (``AI_FLAG_BASIC``,
  ``AI_FLAG_EVAL_ATTACK``, ``AI_FLAG_EXPERT``), Kaizo those and bit 4 (``AI_FLAG_RISKY``).
  The vanilla policy runs its three scripts together, so a vanilla trainer must have all
  three; the Kaizo policy runs each trainer's own flag set (``ai.layout.kaizo_layout``, E1
  Ruling R2), so a Kaizo trainer needs only every set flag implemented. The Kaizo layout's ability
  guess is one coin per guessing read, as the decomp draws it (E1 Ruling R1,
  docs/ai/kaizo_roark.md 8.4), so it needs no per-trainer check;
* weather: ``Weather.NONE``, ``SANDSTORM``, ``SUN``, ``RAIN`` and ``HAIL`` (E2 batch 3), and deep
  fog (``FOG``, E1 Batch C, ruling G-44: the accuracy x6/10 under ``NO_CLOUD_NINE``, bcp.c:
  2934-2936, ``outcome._rate``; SolarBeam's ``FIELD_CONDITION_SOLAR_DOWN``; no countdown and no
  turn-end HP, bcp.c:1213-1225; replaced by any weather move or ability). Every
  effect of each on the implemented set is modelled, all gated on Cloud Nine / Air Lock
  (``damage.weather_suppressed``): the damage modifiers (rain and sun on Fire / Water, the
  sandstorm Rock-type SpD boost, Solar Power), the accuracy abilities (Sand Veil, Snow Cloak),
  Swift Swim, the sun's freeze immunity (``effects.can_have_status``), the end-of-turn chip
  and the weather abilities' HP (``step._weather_effect``: sand and hail, Ice Body, Dry Skin,
  Solar Power), the status orbs' Leaf Guard test, the weather moves with their rocks and the
  switch-in weather abilities (Drizzle, Sand Stream, Drought, Snow Warning). The other
  decomp readers of a weather (Thunder, Blizzard, Chlorophyll, Rain Dish and Flower Gift since
  E2 batch 9, Kaizo's Hurricane and Weather Ball since batch 13) are modelled too; the rest
  belong to unsupported moves (SolarBeam, Moonlight), refused with their holder (Leaf Guard is
  modelled since the E1 Gardenia batch, G-14; Hydration and Forecast since E1 batch F, F-5, F-6).
  The field conditions are ``fieldConditionsMask`` bit for bit (``BattleState.field_mask``,
  ``engine.field``; FCM.md 4.5): weather and deep fog, the Uproar bits, the Gravity and Trick
  Room counters, each reader testing its own bits, so a mask with several groups or a borrowed
  counter reads as the decomp reads it. Acid Rain (a Pursuit KO subtracts ``faintedMon`` from
  the mask, subscript_pursuit.s:100, :115; ledger G-12) is modelled in every format: nothing is
  refused for it;
* a Trace copy the AI refuses on the holder's side: the AI reads the copy
  (``damage.decomp_abilities``, as ``LoadBattlerAbility`` does, trainer_ai.c:1170-1209), so it
  is refused like the holder's own ability would be (``trace_copy_refusals``, ruling B3-21).
  Two weather abilities at war (the slower one's weather in ``monSpeedOrder``, ruling G-26)
  and Trace in both parties (an armed holder copies the next newcomer, ruling G-25) are
  modelled;
* Metronome (ruling B5-2): ``BtlCmd_Metronome`` can call any move ``metronome_calls`` lists
  (ids 1-467 but the user's own moves and sCannotMetronomeMoves, battle_lib.c:7152-7178), so a
  holder is refused while any of them is refused as a move of that Pokemon
  (``move_effect_line``), and every per-move check of the fight (the terrain, the Kaizo ROM
  changes, the halved Defense) runs over them too (``fight_calls``);
* in-battle form changes (plan K2a Ruling 8), on either side: Giratina-Origin without the
  Griseous Orb reverts to Altered Forme at battle start (``battle_lib.c:6412-6417``,
  ``subscript_giratina_form_change``), which the engine does not model (the Orb cannot leave in
  battle, so an Origin that starts with it never reaches that branch: Knock Off, Trick /
  Switcheroo, Thief / Covet, Fling and Embargo fail on a held Griseous Orb,
  subscript_knock_off.s:7, subscript_exchange_items.s:9-10, battle_script.c:5111,
  effect_script_0233.s:6, subscript_embargo_start.s:9; Pluck eats only berries,
  ``BattleSystem_PluckBerry``, battle_lib.c:5539-5543; every other ``RemoveItem`` takes the
  holder's own consumed item; Sticky Barb and Recycle need an empty hand, battle_lib.c:5389,
  :7379, effect_script_0184.s:5); Castform, Cherrim, Arceus
  and the Transformed Giratina-Origin are modelled since E1 batch H (``hooks.form_change``,
  ``BattleSystem_TriggerFormChange``, battle_lib.c:6305-6451, F-6, H-5); Shaymin-Sky reverts to Land
  Forme when frozen (``subscript_check_shaymin_form.s``, ``battle_controller_player.c:3497-3505``)
  and is modelled since E1 batch F (``hooks.shaymin_land``, F-16), also for a Transformed
  battler that copied a Sky Shaymin: the script writes Land's ability into the Transform user's
  own party Pokemon (``BattleMon_CopyToParty`` with ``updateStats``, battle_script.c:3401-3407,
  battle_io_command.c:601-604), which keeps it for its later switch-ins (``InitBattleMon``,
  battle_lib.c:124): ``BattleState.party_ability``, modelled in E1 batch G (F-O2).
  Rotom, Deoxys and Wormadam forms never change in battle and are read by personal row
  (``Tables.species_form_row``), like Giratina-Origin holding the Orb. A trainer Pokemon in
  a data form is built as its game builds it (G-41, ``data.trainer.build_trainer_party``):
  the base species' ability and default moves, the form's types, and the form's stats in
  Kaizo or the base's in vanilla;
* battle format: only "single". A trainer whose header is a double battle, who is fought
  beside a second enemy trainer ("double_two_trainer") or who is fought while the player has a
  partner ("multi", tag battles and walking partners) is refused (``Trainer.format``,
  ``IMPLEMENTED_FORMATS``), so no prep, search or CLI path plays a four-battler battle to a
  result. Four-battler TURNS are no longer refused: ``FOUR_TURN_CLAUSES`` is empty (D3-16
  deleted the last clause), so ``four_turn_refusals``, which every four-battler caller of
  ``step`` still runs, refuses nothing, and ``step`` plays every four-battler turn. At B = 4
  coverage checks only this battle-level format refusal. It must stay until D5 lands the Bind /
  Attract source (``step._source_leaves`` ends only ``_foe(side)``'s Bind and Attract; the decomp
  loops every battler, battle_lib.c:2088-2098, :2168-2183), D11 builds the four-battler
  enumeration and distribution (``enumerate`` / ``outcome``), and E1 reads AI_CONTEXT.defender
  in the AI's damage estimates (trainer_ai.c:1109): the ledger's "OWED D5" and "OWED D11 / E1"
  lines;
* Kaizo trainer fields (plan K2b Task 14): a permanent Trick Room or Gravity area
  (``Trainer.area_condition``; both are modelled: Gravity since the E1 trial batch, G-67, Trick
  Room since E3-2-4, ``start_trick_room``), a gauntlet (a chain
  of fights, ``Trainer.gauntlet``), a start condition the extraction left to the user
  (``"unknown"``, never read as a weather), a party member with no nature, gender or
  ability data when the trainer has no PID routine (no PKCalc set; plan preflight #34)
  and, for the Kaizo game, ROM code changes not yet classified
  (``Names.unclassified_rom_changes``) and ``not_reached`` ROM changes whose resolved
  condition (``Names.rom_change_conditions``, ``rom_change_refusals``) holds for the fight:
  a battle format or area, a move, ability or held item on either side, a trainer or a
  trainer bag. A condition that does not hold refuses nothing. Area weather and fog are read
  through ``start_weather`` like vanilla's; ``Trainer.optional`` refuses nothing;
* battle terrain (K3b2 Task 9): a move whose decomp effect reads the terrain
  (``TERRAIN_EFFECTS``: Nature Power, Secret Power, Camouflage), on either side or a pool
  candidate's, is refused against a trainer whose ``Trainer.terrain`` is
  ``TERRAIN_UNKNOWN``: its start tiles give several terrains and the run has not named one
  (``cli.fight_terrain``, ``--terrain``: a run input, ledger G-18; the line names the
  choices) or its maps give none (``terrain_refusal``). That is a run input (manager ruling 1),
  not a coverage gap: with a named terrain all three are modelled (ledger G-21). Nature Power's
  call (``terrain_call``, ``ids.TERRAIN_MOVES``) also counts as a move of the fight for the
  Kaizo ROM-change conditions (``rom_change_refusals``).

``check_coverage`` also runs the AI's own refusals (``ai.policy.ai_refusals``: opponent
moves without transcribed script paths, too many script-roll sites, type-chart
modifiers the AI does not model, switching-rule refusals) on the fight's turn-0 state,
and refuses a bag that mixes item ids (the decomp applies the last zeroed slot's item,
docs/discrepancies.md). The AI's state-dependent refusals that can first appear later in
the fight (``AI_OnlyIneffectiveMoves``) are not visible at turn 0; ``ai_policy`` refuses them
when they occur. The Kaizo layout's 69 switch sites are every draw a party of six can make.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from types import SimpleNamespace

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.game_constants import POLICY
from nuzline.data.games import KAIZO
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import (
    TERRAIN_NAMES,
    TERRAIN_UNKNOWN,
    UNKNOWN_CONDITION,
    Trainer,
    missing_set_data,
)
from nuzline.engine.damage import PHYSICAL, decomp_abilities
from nuzline.engine.format import foe as _foe
from nuzline.engine.format import format_of
from nuzline.engine.ids import (
    GIRATINA_FORM_ORIGIN,
    GRAVITY_TURNS,
    METRONOME_DRAW,
    MOVE_CHATTER,
    MOVE_SKETCH,
    SKETCH_FORBIDDEN,
    SPECIES_CHATOT,
    SPEED_HALVING_HOLD_EFFECTS,
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    TRICK_ROOM_TURNS,
    UNTRACEABLE_ABILITIES,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    Stat,
    Weather,
)
from nuzline.engine.levelup import HOLD_EFFECT_EXP_UP, MAX_LEVEL
from nuzline.engine.state import BattleState, make_state, spec_exp, spec_stats

# include/constants/forms.h:94, :98; generated/species.txt and items.txt (line N is id N-1).
# Checked against each game's data at load (nuzline.data.game_constants).
SPECIES_GIRATINA = 487
ITEM_GRISEOUS_ORB = 112


# Decomp battle effects whose scripts read the battle terrain (BattleSystem_GetTerrain):
# Nature Power (173, BtlCmd_GetTerrainMove, battle_script.c:8036-8048), Secret Power (197,
# BtlCmd_GetTerrainSecondaryEffect, :8063-) and Camouflage (213, BtlCmd_TryCamouflage,
# :7997-8022). Matched against tables.move_ai_effect (the unmapped decomp effect).
TERRAIN_EFFECTS = (173, 197, 213)
NATURE_POWER_EFFECT = 173  # BATTLE_EFFECT_NATURE_POWER, the decomp effect id


class Support(IntEnum):
    NONE = 0
    SAMPLING_ONLY = 1
    EXACT = 2


IMPLEMENTED_EFFECTS: dict[EffectId, Support] = {
    e: Support.EXACT for e in EffectId if e != EffectId.UNSUPPORTED
}
IMPLEMENTED_ABILITIES: dict[AbilityId, Support] = {
    a: Support.EXACT for a in AbilityId if a != AbilityId.UNSUPPORTED
}
IMPLEMENTED_ITEMS: dict[ItemId, Support] = {
    i: Support.EXACT for i in ItemId if i != ItemId.UNSUPPORTED
}
# generated/ai_flags.txt, in bit order (the FlagTable of src/battle/trainer_ai/script.s).
AI_FLAG_NAMES = (
    "BASIC",
    "EVAL_ATTACK",
    "EXPERT",
    "SETUP_FIRST_TURN",
    "RISKY",
    "PRIORITIZE_EXTREMES",
    "BATON_PASS",
    "TAG_STRATEGY",
    "CHECK_HP",
    "WEATHER",
    "HARRASSMENT",
)
# Per AI policy (game_constants.POLICY): the flag bits ai.policy runs (policy._flags).
IMPLEMENTED_AI_FLAGS: dict[str, dict[int, Support]] = {
    "vanilla": {0: Support.EXACT, 1: Support.EXACT, 2: Support.EXACT},
    "kaizo": {
        0: Support.EXACT,
        1: Support.EXACT,
        2: Support.EXACT,
        3: Support.EXACT,
        4: Support.EXACT,
        5: Support.EXACT,
        6: Support.EXACT,  # AI_FLAG_BATON_PASS, E1 prep batch P3
        8: Support.EXACT,
        9: Support.EXACT,
        10: Support.EXACT,  # AI_FLAG_HARRASSMENT, E1 Roark close batch (G-37)
    },
}
# The vanilla policy runs these scripts together, so a vanilla trainer must have them all. The
# Kaizo policy runs each trainer's own set (E1 Ruling R2): every set flag must be implemented.
POLICY_AI_FLAGS = {"vanilla": 0b111}
IMPLEMENTED_WEATHER: dict[Weather, Support] = {
    Weather.NONE: Support.EXACT,
    Weather.SUN: Support.EXACT,
    Weather.RAIN: Support.EXACT,
    Weather.SANDSTORM: Support.EXACT,
    Weather.HAIL: Support.EXACT,
    Weather.FOG: Support.EXACT,  # E1 Batch C, G-44
}
# Trainer.format (trainers.json: TrainerHeader.battleType, then the field scripts).
IMPLEMENTED_FORMATS: dict[str, Support] = {
    "single": Support.EXACT,
    "double": Support.NONE,
    "double_two_trainer": Support.NONE,
    "multi": Support.NONE,
    # Kaizo: a format the field scripts leave to game state (data/kaizo/battle_formats.json,
    # tools/battle_scripts.py UNKNOWN_FORMAT), refused explicitly.
    "unknown": Support.NONE,
}


@dataclass(frozen=True)
class TurnClause:
    """One refusal of the four-battler turn (``four_turn_refusals``): the D3 ``group`` that
    models ``feature`` and deletes this clause in its commit; ``refuses(tables, state, actions)``
    is true where the turn uses it. ``tables``, ``state`` and ``actions`` are the arguments of
    ``step`` (``actions``: one int per battler)."""

    group: str
    feature: str
    refuses: Callable[[Tables, BattleState, np.ndarray], bool]


# The effects whose move runs another move: a caller may become any move at all.
CALLERS = frozenset(
    {
        EffectId.SLEEP_TALK,
        EffectId.METRONOME,
        EffectId.COPYCAT,
        EffectId.ASSIST,
        EffectId.ME_FIRST,
        EffectId.NATURE_POWER,
    }
)


def _present(state: BattleState, b: int) -> bool:
    """Battler ``b``'s active Pokemon has HP (an absent battler's slot sits at 0 HP)."""
    fmt = format_of(state)
    return bool(np.asarray(state.hp)[fmt.party_of[b], int(np.asarray(state.active)[b])] > 0)


def _selects(tables: Tables, state: BattleState, actions, effects) -> bool:
    """A present battler whose move action selects a move whose effect is in ``effects`` or is a
    caller (``CALLERS``; ``order.selected_move``)."""
    from nuzline.engine.actions import MOVE_ACTIONS
    from nuzline.engine.order import selected_move

    wanted = {int(e) for e in set(effects) | CALLERS}
    effect = np.asarray(tables.move_effect)
    for b in range(format_of(state).B):
        if not _present(state, b) or int(actions[b]) >= MOVE_ACTIONS:
            continue  # a switch or item action selects no move
        move = int(selected_move(tables, state, b, int(actions[b]))[0])
        if int(effect[move]) in wanted:
            return True
    return False


# Build order of the D3 groups; each group deletes its own clause, and the tuple is empty after
# D3-16 (tests/engine/test_coverage_four.py::OWED). Battlers 2 and 3 absent (0 HP, nothing left
# in their parties) is the supported turn: the decomp's turn is then the two-battler turn
# (``order.action_order`` puts them after battlers 0 and 1, where they never act).
FOUR_TURN_CLAUSES: tuple[TurnClause, ...] = ()  # D3-16 deleted the last clause


def four_turn_refusals(tables: Tables, state: BattleState, actions) -> list[str]:
    """The lines refusing a four-battler turn: ``[]`` at two battlers, else one line per
    ``FOUR_TURN_CLAUSES`` entry the turn trips. Every four-battler caller of ``step`` runs it
    first (the differential skips a draw that is refused)."""
    if format_of(state).B == 2:
        return []
    actions = np.asarray(actions)
    return [
        f"{c.group}: {c.feature} (four-battler turn UNSUPPORTED)"
        for c in FOUR_TURN_CLAUSES
        if c.refuses(tables, state, actions)
    ]


# Trainer.start_condition (trainers.json) as a Weather.
START_CONDITIONS = {
    "none": Weather.NONE,
    "sun": Weather.SUN,
    "rain": Weather.RAIN,
    "sandstorm": Weather.SANDSTORM,
    "hail": Weather.HAIL,
    "fog": Weather.FOG,
}


@dataclass
class CoverageReport:
    """``missing``: one human-readable line per unsupported mechanic the fight uses;
    ``sampling_only``: the same for ``SAMPLING_ONLY`` ones; ``level``: the lowest support
    over everything present."""

    missing: list[str] = field(default_factory=list)
    sampling_only: list[str] = field(default_factory=list)
    level: Support = Support.EXACT


class UnsupportedMechanics(ValueError):
    """The fight uses mechanics the engine does not implement; ``report`` lists them."""

    def __init__(self, report: CoverageReport):
        self.report = report
        super().__init__(
            f"{len(report.missing)} unsupported mechanic(s): " + "; ".join(report.missing)
        )


def start_gravity(trainer: Trainer) -> int:
    """The Gravity turns a trainer's fight starts with: ``ids.GRAVITY_TURNS`` for the Kaizo area
    condition Gravity (sub_seq 297 runs ``subscript_gravity_start`` at battle start; the Kaizo
    tables re-arm it for ever, ``Tables.gravity_rearm``), else 0 (G-67)."""
    return GRAVITY_TURNS if trainer.area_condition.gravity else 0


def start_trick_room(trainer: Trainer) -> int:
    """The Trick Room counter a trainer's fight starts with: ``ids.TRICK_ROOM_TURNS`` for the Kaizo
    area condition Trick Room (map weather 18: ``subscript_overworld_trick_room.s:14`` sets
    FIELD_CONDITION_TRICK_ROOM_INIT; the Kaizo tables re-arm it for ever,
    ``Tables.trick_room_rearm``), else 0 (E3-2-4)."""
    return TRICK_ROOM_TURNS if trainer.area_condition.trick_room else 0


def start_weather(
    trainer: Trainer, field_defog: bool = False, script_weather: bool | None = None
) -> Weather:
    """The weather a trainer's fight starts in (``Trainer.start_condition``). A start
    condition of ``"unknown"`` (a user call) is no weather: ``ValueError``, like any
    other unknown value; ``check_opponent`` refuses it instead.

    ``field_defog``: the run cleared the fight's fog with field Defog on that map visit (a
    run input, ledger G-15): FLAG_DEFOG_ACTIVE turns the overworld weather of a fog header
    (14) clear (field_map_change.c:275-280; ScrCmd_0C4 at once, scrcmd.c:3667-3674), and
    the battle starts in the overworld weather (field_battle_data_transfer.c:267), so in no
    weather. ``ValueError`` for a fight that is not on such a header
    (``AreaCondition.fog_headers``): its fog, if any, is deep fog (15) or not map fog.

    ``script_weather`` (ledger G-24): True, the fight starts while a map script's weather is
    on (``AreaCondition.script_weather``: the Kaizo command 0x9C set the overworld weather,
    field_battle_data_transfer.c:267); False, after it ended or before it began (the map's
    own weather, clear: ScrCmd_0C4, scrcmd.c:3667-3674; a map change,
    field_map_change.c:272-280); None, the start condition. ``ValueError`` for True or
    False on a fight no script weather reaches (``AreaCondition.script_window`` empty)."""
    if script_weather is not None:
        area = trainer.area_condition
        if area.script_weather is None:
            raise ValueError(
                f"trainer {trainer.id}: no map script sets the weather where this fight is met "
                "(rom_script_weather.json)"
            )
        # False is the header's own weather; the one script window's header, 373 (Route 213),
        # holds 0 (rom/rom_map_weather.json[373]; field_map_change.c:273-280). A window on a
        # header with its own weather must read it here.
        return START_CONDITIONS[area.script_weather] if script_weather else Weather.NONE
    if field_defog:
        if not trainer.area_condition.fog_clearable:
            raise ValueError(
                f"trainer {trainer.id}: field Defog clears only the fog of a map whose weather "
                "is OVERWORLD_WEATHER_FOG (src/field_map_change.c:275-277), and this fight "
                "starts on none"
            )
        return Weather.NONE
    if trainer.start_condition == UNKNOWN_CONDITION:
        raise ValueError(
            f"trainer {trainer.id}: start condition unknown is a pending user call, not a "
            "weather; coverage refuses it (check_opponent)"
        )
    try:
        return START_CONDITIONS[trainer.start_condition]
    except KeyError:
        raise ValueError(f"unknown start condition {trainer.start_condition!r}") from None


def unknown_start(trainer: Trainer) -> bool:
    """The fight's start condition is a pending user call ("unknown"), never a weather."""
    return UNKNOWN_CONDITION in (trainer.start_condition, trainer.area_condition.weather)


def display_name(trainer: Trainer) -> str:
    """How a report names ``trainer``: its label when it has one (Kaizo: "Leader Roark",
    "Pokemon Trainer Barry #3 [Empoleon]"), else its name. ``Trainer.name`` stays the ROM
    name for data and matching (plan K2b Ruling 5); vanilla has no labels."""
    return trainer.label or trainer.name


class _Collector:
    def __init__(self) -> None:
        self.report = CoverageReport()

    def add(self, support: Support, what: str, kind: str) -> None:
        """Record ``what`` (a ``kind`` of mechanic) at ``support``."""
        self.report.level = min(self.report.level, support)
        if support == Support.NONE:
            self.report.missing.append(f"{what} ({kind} UNSUPPORTED)")
        elif support == Support.SAMPLING_ONLY:
            self.report.sampling_only.append(f"{what} ({kind} SAMPLING_ONLY)")


# BattleSystem_CalcMoveDamage halves defenseStat for BATTLE_EFFECT_HALVE_DEFENSE
# (battle_lib.c:6943-6945) before the stage scales it into stageDivisor (:6963-6975): at
# -6 Defense (x2/8) a Defense below this gives stageDivisor 0, and ``damage /= 0`` is not
# modelled by the engines.
HALVED_DEFENSE_MIN = 8


def halves_defense(tables: Tables, move: int) -> bool:
    """``move`` is a physical ``SELF_DESTRUCT`` move: its damage reads the halved Defense
    (a special one, Kaizo's Memento, halves nothing it reads)."""
    return (
        int(np.asarray(tables.move_effect)[move]) == EffectId.SELF_DESTRUCT
        and int(np.asarray(tables.move_category)[move]) == PHYSICAL
    )


def defense_stat(tables: Tables, spec: PokemonSpec) -> int:
    """``spec``'s Defense as ``make_state`` builds it (``state.spec_stats``)."""
    return int(spec_stats(tables, spec)[Stat.DEF])


def low_defense(tables: Tables, foes: Sequence[PokemonSpec]) -> int | None:
    """The lowest Defense below ``HALVED_DEFENSE_MIN`` among ``foes`` that can be hit (not
    fainted), or None."""
    defenses = [defense_stat(tables, m) for m in foes if m.species and m.current_hp != 0]
    low = [d for d in defenses if d < HALVED_DEFENSE_MIN]
    return min(low) if low else None


def halved_defense_line(tables: Tables, move: int, foe_defense: int | None) -> str | None:
    """`` against Defense N`` when ``move`` halves Defense and a foe has Defense ``N``
    below ``HALVED_DEFENSE_MIN`` (``low_defense``): refused, the -6 stage would divide by
    0. Else None."""
    if foe_defense is None or not halves_defense(tables, move):
        return None
    return f" against Defense {foe_defense}"


# BtlCmd_CalcGyroBallPower (battle_script.c:7127) divides by the user's monSpeedValues, which
# CompareBattlerSpeed computes as Speed x the stage ratio (-6: 10/40), then / 4 for paralysis
# (battle_lib.c:1225, :1259-1261): below 16 Speed that can reach 0, a division by zero the decomp
# leaves to the ROM's division routine. A Gyro Ball user below it is refused (B13-14). A holder of
# a sSpeedHalvingItemEffects item (Iron Ball, Macho Brace, the Power items: / 2 between the stage
# and paralysis, battle_lib.c:1240-1246, G-56) reaches 0 below twice that.
GYRO_SPEED_MIN = 16


def gyro_speed_line(tables: Tables, move: int, spec: PokemonSpec) -> str | None:
    """`` with Speed N`` when ``move`` is Gyro Ball's (``EffectId.GYRO_BALL``) and ``spec``'s
    Speed ``N`` is below ``GYRO_SPEED_MIN``: refused, its power can divide by 0. Else None."""
    if int(np.asarray(tables.move_effect)[move]) != EffectId.GYRO_BALL:
        return None
    speed = int(spec_stats(tables, spec)[Stat.SPE])
    halved = int(np.asarray(tables.item_hold_effect)[spec.item]) in SPEED_HALVING_HOLD_EFFECTS
    return f" with Speed {speed}" if speed < GYRO_SPEED_MIN * (1 + halved) else None


def halving_moves(tables: Tables, foes: Sequence[PokemonSpec]) -> list[int]:
    """The distinct moves of ``foes`` that can act (not fainted) that halve Defense
    (``halves_defense``), ascending: each is refused against a Pokemon whose own Defense is
    below ``HALVED_DEFENSE_MIN`` (``halved_by_line``)."""
    return sorted(
        {
            move
            for m in foes
            if m.species and m.current_hp != 0
            for move in m.moves
            if move and halves_defense(tables, move)
        }
    )


def halved_by_line(
    tables: Tables, names: Names, spec: PokemonSpec, moves: Sequence[int]
) -> str | None:
    """``Defense N against the opponent's <move>`` when ``spec``'s own Defense ``N`` is below
    ``HALVED_DEFENSE_MIN`` and ``moves`` (``halving_moves`` of its foes) is not empty: the
    foe's halved Defense at -6 would divide by 0. Else None."""
    if not moves or (defense := defense_stat(tables, spec)) >= HALVED_DEFENSE_MIN:
        return None
    return f"Defense {defense} against the opponent's {names.move_names[moves[0]]}"


def move_effect_line(
    tables: Tables, names: Names, move: int, row: int, known: Sequence[int] = ()
) -> tuple[Support, str]:
    """The support of ``move`` for the Pokemon at personal ``row`` and its name as a report
    line shows it: by engine effect. Metronome (ruling B5-2) is refused while any move it can
    call for a Pokemon knowing ``known`` (``metronome_calls``) is."""
    name = names.move_names[move]
    effect = EffectId(int(np.asarray(tables.move_effect)[move]))
    support = IMPLEMENTED_EFFECTS.get(effect, Support.NONE)
    if effect == EffectId.METRONOME and support != Support.NONE:
        flags = np.asarray(tables.move_flags)
        calls = [
            m
            for m in range(1, METRONOME_DRAW + 1)
            if m not in known and m != move and not flags[m] & MoveFlag.NO_METRONOME
        ]
        if any(move_effect_line(tables, names, m, row)[0] == Support.NONE for m in calls):
            support = Support.NONE
    return support, name


# --- Experience, level-up and learnable moves (K3d-exp design §4) ---------------------------

# Decomp battle effects (generated/move_battle_effects.txt, line N is id N - 1), matched
# against tables.move_ai_effect.
# BATTLE_EFFECT_POWER_BASED_ON_FRIENDSHIP (Return) and ..._LOW_FRIENDSHIP (Frustration):
# effect_script_0121.s / 0123.s read the attacker's BATTLEMON_FRIENDSHIP.
FRIENDSHIP_POWER_EFFECTS = (121, 123)
# Ruling D9 (gone): the payout reads the party item (``levelup.party_item``), which every item
# move but Knock Off copies from the battle item at once (BattleMon_CopyToParty); Knock Off's
# kept party item is the state field ``knocked_item``. No item-moving move needs a refusal.
MAX_EXP_BOOST = 170  # the largest exp_boost (battle_script.c:9757-9763)


def reachable_level(tables: Tables, spec: PokemonSpec, opp: Sequence[PokemonSpec]) -> int:
    """An upper bound on the level the player's ``spec`` can reach in a fight against ``opp``:
    the level at ``exp + sum(bound)`` over the opponents that can faint (a species and HP),
    with ``bound = max(E, 2) * 3/2 * (Lucky Egg ? 3/2 : 1) * exp_boost / 100 + 1`` (design
    §4.1): every payout to this Pokemon alone. ``E = yield[species] * level / 7``
    (battle_script.c:2216-2217); one Pokemon gets ``g + h <= max(E, 2)`` (``g = max(1, ...)``,
    ``h = max(1, ...)``, :2219-2238); then the Lucky Egg, trainer and OT multipliers
    (:9749-9763), whose truncations only lower it; ``+ 1`` is slack. An unresolved
    ``exp_boost`` (0) counts as the largest, 170. The level of an exp ``x`` is the last
    ``L`` with ``table[L] <= x`` (``Pokemon_ShouldLevelUp``, pokemon.c:3501-3526), and a
    level-100 Pokemon gains nothing (battle_script.c:9740)."""
    if spec.level >= MAX_LEVEL:
        return spec.level
    yields = np.asarray(tables.species_exp_yield)
    lucky = int(np.asarray(tables.item_hold_effect)[spec.item]) == HOLD_EFFECT_EXP_UP
    boost = spec.exp_boost or MAX_EXP_BOOST
    total = 0
    for foe in opp:
        if foe.species == 0 or foe.current_hp == 0:
            continue
        e = int(yields[foe.species]) * foe.level // 7
        total += max(e, 2) * 3 * (3 if lucky else 2) * boost // (2 * 2 * 100) + 1
    table = np.asarray(tables.exp_table)[int(np.asarray(tables.species_exp_rate)[spec.species])]
    x = spec_exp(tables, spec) + total
    return max(spec.level, int(np.sum(table[1 : MAX_LEVEL + 1] <= x)))


def reachable_learn_moves(
    tables: Tables, spec: PokemonSpec, opp: Sequence[PokemonSpec]
) -> list[int]:
    """Every move in the learnset of ``spec``'s form row (``Pokemon_LevelUpMove`` loads it per
    species and form, pokemon.c:3943) at a level in ``(level, reachable_level]``, in list
    order without repeats: the moves an in-battle level-up can auto-learn or prompt
    (bs.c:9999-10187). A move already known is included: a prompt can replace it first."""
    top = reachable_level(tables, spec, opp)
    if top <= spec.level:
        return []
    row = personal_row(tables, spec.species, spec.form)
    levels = np.asarray(tables.learnset_level)[row]
    moves = np.asarray(tables.learnset_move)[row]
    out: list[int] = []
    for level, move in zip(levels.tolist(), moves.tolist(), strict=True):
        if spec.level < level <= top and move and move not in out:
            out.append(int(move))
    return out


def forced_learn_moves(tables: Tables, spec: PokemonSpec, opp: Sequence[PokemonSpec]) -> list[int]:
    """The learnable moves (``reachable_learn_moves``) a level-up could put into one of
    ``spec``'s empty move slots without asking: ``BoxPokemon_AddMove`` (pokemon.c:3852-3873)
    takes the first empty slot (an auto-learn, battle_script.c:10021-10027) and prompts only
    once all four are full (:10016-10018); no battle path empties a slot. So with ``e`` empty
    slots these are at most the first ``e`` moves not already known, in scan order (levels
    ascending, bs.c:9882; within a level, list order, pokemon.c:3945-3956). A gap before a
    known move (AddMove learns it again) only fills slots sooner, so this is a superset."""
    empty = sum(1 for m in spec.moves if not m)
    top = reachable_level(tables, spec, opp)
    if not empty or top <= spec.level:
        return []
    row = personal_row(tables, spec.species, spec.form)
    levels = np.asarray(tables.learnset_level)[row].tolist()
    moves = np.asarray(tables.learnset_move)[row].tolist()
    order = sorted(
        (level, i) for i, level in enumerate(levels) if spec.level < level <= top and moves[i]
    )
    out: list[int] = []
    for _, i in order:
        move = int(moves[i])
        if move not in spec.moves and move not in out:
            out.append(move)
        if len(out) == empty:
            break
    return out


def learn_masked(tables: Tables, move: int) -> bool:
    """``levelup.learn_options`` (and its reference mirror) never offers to learn ``move``:
    its effect is ``UNSUPPORTED`` (ruling D10). The player can always give it up."""
    return int(np.asarray(tables.move_effect)[move]) == EffectId.UNSUPPORTED


def learnable_moves(tables: Tables, spec: PokemonSpec, opp: Sequence[PokemonSpec]) -> list[int]:
    """The ``reachable_learn_moves`` coverage must grade (ruling D10): every one but the
    prompt-only moves the learn options mask (``learn_masked`` and not in
    ``forced_learn_moves``), which the player never takes in the engine. A masked move is
    then a legal game choice the search skips: plan values stay achievable lower bounds,
    (``STATE.md``, Experience). Moves refused for any other reason (the AI, terrain, a ROM change,
    a halved Defense) are not masked and stay graded (refused)."""
    forced = forced_learn_moves(tables, spec, opp)
    return [
        m
        for m in reachable_learn_moves(tables, spec, opp)
        if m in forced or m in spec.moves or not learn_masked(tables, m)
    ]


def _effect_moves(tables: Tables, moves: Sequence[int], effects: Sequence[int]) -> list[int]:
    effect = np.asarray(tables.move_ai_effect)
    return sorted({int(m) for m in moves if m and int(effect[m]) in effects})


def exp_refusals(
    tables: Tables,
    names: Names,
    spec: PokemonSpec,
    learnable: Sequence[int],
    fight_moves: Sequence[int],
    levels_up: bool,
) -> list[str]:
    """Why the exp payout or a level-up of the player's ``spec`` would not be exact (design
    §4.2-4.6), one line each. ``learnable``: its ``reachable_learn_moves``; ``fight_moves``:
    every move any Pokemon of the fight could use (both sides, learnable ones included);
    ``levels_up``: ``reachable_level`` exceeds its level.

    * §4.3: below level 100 with ``exp_boost == 0`` (the OT and language test unresolved).
    * §4.2 (ruling D9, gone): the payout reads the **party** item (battle_script.c:2210, :9709,
      :10226; ``MON_DATA_HELD_ITEM``), which Trick, Switcheroo, Thief and Covet copy from the
      battle item (UpdateMonData, :3078, :3407, through ``BattleMon_CopyToParty``,
      battle_lib.c:1819, and ``BattleMon_CopyToParty``'s message, battle_io_command.c:589-591)
      and Knock Off does not: its ``knockedOffItemsMask`` bit stops the copy (:590-592) and the
      party keeps the item. The engines model both (``levelup.party_item``, ``knocked_item``).
    * §4.4 (ruling D6): its egg location is nonzero, it can level, and a friendship-power move
      (``FRIENDSHIP_POWER_EFFECTS``) is in the fight: the level-up adds 1 when the egg
      location is the battle map (pokemon.c:2668-2670), which the engine does not know
      (Kaizo changed ``MapHeader_GetMapLabelTextID``, docs/kaizo/rom_diff.md:269).
    * §4.5 (gone: ``BattleSystem_ReloadPokemon``, battle_lib.c:169-194, is modelled for Power
      Trick, B12-10, Mimic, G-68, and Transform, G-69).
    * §4.6: an empty move slot with PP Ups: a move learned there inherits them
      (``BoxPokemon_SetMoveSlot``, pokemon.c:3929-3930), which the engine does not model."""
    out = []
    if spec.level < MAX_LEVEL and spec.exp_boost == 0:
        out.append(
            "exp multiplier unresolved (original trainer and language unknown: read the save "
            "with its trainer)"
        )
    if (
        levels_up
        and spec.egg_location != 0
        and (readers := _effect_moves(tables, fight_moves, FRIENDSHIP_POWER_EFFECTS))
    ):
        out.append(
            f"egg location {spec.egg_location} may add 1 friendship per level-up, which "
            f"{names.move_names[readers[0]]} reads"
        )
    junk = [
        i for i, (m, ups) in enumerate(zip(spec.moves, spec.pp_ups, strict=True)) if not m and ups
    ]
    if junk:
        out.append(
            f"empty move slot {junk[0]} has {spec.pp_ups[junk[0]]} PP Ups, which a move "
            "learned there inherits"
        )
    return out


def _party_abilities(tables: Tables, party: Sequence[PokemonSpec]) -> set[int]:
    """The ``AbilityId`` of every Pokemon of ``party`` (its personal row and ability slot)."""
    abilities = np.asarray(tables.species_abilities)
    return {
        int(abilities[personal_row(tables, m.species, m.ability_from), m.ability_slot])
        for m in party
        if m.species
    }


def trace_copy_refusals(fight: Tables, state, names: Names | None = None) -> list[str]:
    """Ruling B3-21: the AI's refusals of a Trace copy. ``subscript_trace.s`` writes the
    copied ability into the holder's ``BattleMon.ability`` and the AI context
    (``BattleAI_SetAbility``, battle_script.c:3401-3403, :12106-12109), and
    ``LoadBattlerAbility`` / ``CheckBattlerAbility`` read it (trainer_ai.c:1170-1209), as the
    AI does through ``damage.decomp_abilities``. So for every Trace holder of ``state`` (either
    side; a lead may have traced in ``make_state`` already) and every ability it can copy from
    the other party (``ChooseTraceTarget`` skips Forecast, Trace and Multitype,
    battle_lib.c:7750-7775), the lines ``ai.policy.ai_refusals`` gives with that copy in place
    and not with every holder holding Trace, each as ``"Trace copying <ability>: <line>"``.
    ``fight`` are the fight's tables (``ai.layout.fight_tables``)."""
    from nuzline.ai.policy import ai_refusals  # ai imports the engine

    species, form = np.asarray(state.species), np.asarray(state.ability_form)
    row = np.asarray(fight.species_form_row)[species, form]
    own = np.asarray(fight.species_abilities)[row, np.asarray(state.ability_slot)]
    holders = list(zip(*np.nonzero((species > 0) & (own == AbilityId.TRACE)), strict=True))
    if not holders:
        return []
    armed = state.ability
    for side, slot in holders:
        armed = armed.at[side, slot].set(AbilityId.TRACE)
    armed = state._replace(ability=armed)
    decomp = np.asarray(decomp_abilities(fight, armed))
    base = set(ai_refusals(fight, armed, names))
    out: list[str] = []
    for side, slot in holders:
        foe = _foe(side)
        # Raw BattleMon.ability: ChooseTraceTarget (battle_lib.c:7754-7776) and the copy
        # (subscript_trace.s:6-7) read it, not Battler_Ability.
        copies = {
            (int(armed.ability[foe, s]), int(decomp[foe, s]))
            for s in range(species.shape[1])
            if species[foe, s] and int(decomp[foe, s]) not in UNTRACEABLE_ABILITIES
        }
        for copy, copy_decomp in sorted(copies):
            copied = armed._replace(ability=armed.ability.at[side, slot].set(copy))
            label = names.ability_names[copy_decomp] if names is not None else str(copy_decomp)
            for line in ai_refusals(fight, copied, names):
                if line not in base and (line := f"Trace copying {label}: {line}") not in out:
                    out.append(line)
    return out


def trace_copy_line(
    tables: Tables,
    names: Names,
    trainer: Trainer | None,
    spec: PokemonSpec,
    opp: Sequence[PokemonSpec],
) -> str | None:
    """Prep's per-candidate form of ``trace_copy_refusals`` (ruling B3-20): ``spec`` alone
    against ``trainer``'s party ``opp``, the first line or None. Only when a Trace holder is
    on either side; without a ``trainer`` there is no fight AI to ask (None)."""
    if trainer is None or AbilityId.TRACE not in (
        _party_abilities(tables, [spec]) | _party_abilities(tables, opp)
    ):
        return None
    fight, error = _fight_ai_tables(
        tables, names, trainer, opp, lambda: foe_moves(tables, [spec], opp)
    )
    if error is not None:  # refused for the fight by check_opponent
        return None
    state = make_state(fight, [spec], list(opp), opp_items=trainer.items, terrain=trainer.terrain)
    lines = trace_copy_refusals(fight, state, names)
    return lines[0] if lines else None


def metronome_calls(tables: Tables, mon: PokemonSpec) -> list[int]:
    """The moves ``BtlCmd_Metronome`` (battle_script.c:4535-4562) can call for ``mon``: ids
    1-467 it does not know that ``Move_CanBeMetronomed`` allows (``MoveFlag.NO_METRONOME``;
    ``order.metronome_candidates``). Gravity's test is state (``actions.gravity_banned``): this
    is the superset; Heal Block, its other test, is owed with that move."""
    flags = np.asarray(tables.move_flags)
    return [
        m
        for m in range(1, METRONOME_DRAW + 1)
        if m not in mon.moves and not flags[m] & MoveFlag.NO_METRONOME
    ]


def fight_calls(tables: Tables, party: Sequence[PokemonSpec]) -> list[int]:
    """The moves ``party``'s Metronome holders can call (``metronome_calls``): moves of the
    fight for the checks that read every move (the terrain, the Kaizo ROM changes). Sleep Talk
    calls its user's own moves and Copycat a move used in the fight, moves of the fight
    already."""
    effect = np.asarray(tables.move_effect)
    out: set[int] = set()
    for mon in party:
        if any(m and int(effect[m]) == EffectId.METRONOME for m in mon.moves):
            out.update(metronome_calls(tables, mon))
    return sorted(out)


# A Pokemon that knows no move: ``metronome_calls``' superset for a Metronome gained in battle.
_NO_MOVES = SimpleNamespace(moves=(0, 0, 0, 0))


def reached_moves(tables: Tables, moves: Sequence[int]) -> list[int]:
    """``moves`` and, when one of them is Metronome, every move ``BtlCmd_Metronome`` can call
    (battle_script.c:4535-4562; ``metronome_calls`` of a Pokemon knowing none: its holder's own
    moves, which it skips, are in ``moves`` already), distinct: the moves a line holding
    ``moves`` can reach, for the checks that read every move of the fight (the Kaizo ROM
    changes, the terrain)."""
    effect = np.asarray(tables.move_effect)
    out = [int(m) for m in moves if m]
    if any(int(effect[m]) == EffectId.METRONOME for m in out):
        out += metronome_calls(tables, _NO_MOVES)
    return list(dict.fromkeys(out))


# Decomp battle effects (tables.move_ai_effect; generated/move_battle_effects.txt, line N is id
# N - 1) that lock the user into the move it is using, ``moveLockedInto = moveCur``
# (Battler_LockMoveChoice, battle_lib.c:1828-1832): BtlCmd_LockMoveChoice in Bide's
# subscript_bide_start.s:6 (26) and the charge moves' subscript_vanish_on_charge_turn.s:10 (39,
# 75, 145, 151, 155, 255, 256, 263, 272), BtlCmd_CalcRolloutPower (battle_script.c:5667-5669;
# 117, Rollout and Ice Ball), and ``BTLVAR_ATTACKER_LOCKED_MOVE = BTLVAR_CURRENT_MOVE`` in
# subscript_thrash.s:8 (27) and subscript_uproar.s:5 (159). A called move's ``moveCur`` is the
# called move (BtlCmd_GoToMoveScript, battle_script.c:2118), so the next turn ``moveTemp`` is it
# (bcp.c:1885-1888), and UpdateMoveBuffers writes it to ``moveSketched`` and ``movePrevByBattler``
# (bcp.c:3936-3954). The recharge turn (80, subscript_recharge_turn.s:6) locks too, but prints no
# attack message (subscript_recharging.s, bcp.c:2514-2522), so it writes neither: not listed.
LOCK_EFFECTS = (26, 27, 39, 75, 117, 145, 151, 155, 159, 255, 256, 263, 272)
# The invokers whose call can be a move the user does not know: Metronome (BtlCmd_Metronome,
# battle_script.c:4535-4562), Copycat (``movePrev``, :7333-7336), Mirror Move (``moveCopied``,
# bcp.c:4480-4485) and Me First (the target's chosen move, battle_script.c:7286-7312).
FOREIGN_INVOKERS = (EffectId.METRONOME, EffectId.COPYCAT, EffectId.MIRROR_MOVE, EffectId.ME_FIRST)


def locked_calls(tables: Tables, moves: Sequence[int]) -> list[int]:
    """The moves a holder of ``moves`` can be locked into without knowing them: every move with a
    ``LOCK_EFFECTS`` effect when one of ``moves`` is a ``FOREIGN_INVOKERS`` move, else none. Its
    next turn's ``moveTemp`` is the locked move (bcp.c:1885-1888), which becomes its
    ``moveSketched`` (Sketch's copy, battle_script.c:4839-4884) and its ``movePrevByBattler``
    (Mimic's copy, :4476-4524). A superset: any lock move, whichever invoker and chain calls it."""
    effect = np.asarray(tables.move_effect)
    if not any(m and int(effect[m]) in FOREIGN_INVOKERS for m in moves):
        return []
    ai_effect = np.asarray(tables.move_ai_effect)
    return [m for m in range(1, METRONOME_DRAW + 1) if int(ai_effect[m]) in LOCK_EFFECTS]


def sketch_calls(
    tables: Tables,
    known: Sequence[int],
    opp: Sequence[PokemonSpec],
    party_moves: Sequence[int] = (),
) -> list[int]:
    """The moves a player's Sketch can copy (``BtlCmd_TrySketch``, battle_script.c:4839-4884):
    ``known`` are the Pokemon's moves and its ``learnable_moves`` (empty unless one is Sketch); the
    copy is a foe's ``moveSketched``, the move in its slot when the attack message shows
    (``moveTemp``, bcp.c:3952-3954), so a move of ``opp`` that is neither forbidden
    (``SKETCH_FORBIDDEN``) nor already known. A foe that can copy a move of the player's party
    carries it in a slot too: Mimic's copy (``BtlCmd_TryMimic``, battle_script.c:4476-4524) and
    Transform's four moves (``BtlCmd_Transform``, :5346-5348), Transform also through Metronome
    (not in ``sCannotMetronomeMoves``, battle_lib.c:7152-7177) and, when the player's party has
    it, through Copycat (``movePrev``, the player's Transform, ``Move_CanBeMetronomed``,
    battle_script.c:7333-7336). Then ``party_moves`` (the moves of the player's party and their
    ``learnable_moves``) join, as the Sketch holder itself may not know them. A foe slot with a
    ``FOREIGN_INVOKERS`` move adds the moves it can be locked into (``locked_calls``): a called
    Outrage or SolarBeam is its next turn's ``moveSketched``. A superset: the foe may not use it.
    The checks that read every move of the fight (``move_effect_line``, the AI's
    ``player_move_refusal``, the holder's own ``halved_defense_line`` and ``gyro_speed_line``, its
    Metronome calls, the terrain and the ROM changes) run over them, as over the Metronome calls
    (``fight_calls``)."""
    if MOVE_SKETCH not in known:
        return []
    effect = np.asarray(tables.move_effect)
    copier = {EffectId.MIMIC, EffectId.TRANSFORM, EffectId.METRONOME}
    if any(m and int(effect[m]) == EffectId.TRANSFORM for m in party_moves):
        copier.add(EffectId.COPYCAT)
    copies = any(m and int(effect[m]) in copier for mon in opp for m in mon.moves)
    slots = [m for mon in opp for m in mon.moves] + (list(party_moves) if copies else [])
    pool = slots + locked_calls(tables, slots)
    banned = {*known, *SKETCH_FORBIDDEN}
    return sorted({m for m in pool if m and m not in banned})


def _check_party(
    c: _Collector,
    tables: Tables,
    names: Names,
    owner: str,
    party: Sequence[PokemonSpec],
) -> None:
    abilities = np.asarray(tables.species_abilities)
    item_effect = np.asarray(tables.item_effect)
    for slot, mon in enumerate(party):
        where = f"{owner} {names.species_names[mon.species]} (slot {slot})"
        row = personal_row(tables, mon.species, mon.form)  # ValueError: no such form
        for move in mon.moves:
            if move:
                support, name = move_effect_line(tables, names, move, row, mon.moves)
                c.add(support, f"{where}: {name}", "move effect")
        ability_row = personal_row(tables, mon.species, mon.ability_from)
        ability = AbilityId(int(abilities[ability_row, mon.ability_slot]))
        ability_name = names.ability_names[
            names.species_abilities(mon.species, mon.ability_from)[mon.ability_slot]
        ]
        c.add(
            IMPLEMENTED_ABILITIES.get(ability, Support.NONE), f"{where}: {ability_name}", "ability"
        )
        item = ItemId(int(item_effect[mon.item]))
        item_name = names.item_names[mon.item] if mon.item else "no item"
        c.add(IMPLEMENTED_ITEMS.get(item, Support.NONE), f"{where}: {item_name}", "held item")
        _check_form(c, where, mon)


def _check_form(c: _Collector, where: str, mon: PokemonSpec) -> None:
    """Ruling 8: a form the battle changes, which the engine does not model."""
    if (
        mon.species == SPECIES_GIRATINA
        and mon.form == GIRATINA_FORM_ORIGIN
        and mon.item != ITEM_GRISEOUS_ORB
    ):
        c.add(Support.NONE, f"{where}: Origin Forme without the Griseous Orb", "form change")


def rom_change_refusals(
    tables: Tables,
    names: Names,
    trainer: Trainer,
    specs: Sequence[PokemonSpec],
    extra_moves: Sequence[int] = (),
    *,
    spec_terms_only: bool = False,
) -> list[str]:
    """The labels of the ``not_reached`` ROM-change conditions
    (``Names.rom_change_conditions``) that hold for a fight against ``trainer`` with ``specs``
    on the field (both sides), plus ``extra_moves`` a pool candidate could learn. A condition
    holds when any of: the trainer's format is listed (an unknown format, ``"unknown"`` in
    the data or ``None``, matches any condition that lists one); the trainer's area condition has a listed area
    (``trick_room``/``gravity``, the ``tools/rom/diff.py`` area names); a move, a decomp
    ability (``species_abilities_decomp`` at the spec's ability row and ability slot) or a
    held item is listed; the trainer is listed; or the condition covers trainer items and the
    bag holds one. ``spec_terms_only`` keeps just the moves, abilities and items: the pool
    asks it per candidate, the fight-level terms being the fight's own."""
    area = trainer.area_condition
    areas = {
        name for name, on in (("trick_room", area.trick_room), ("gravity", area.gravity)) if on
    }
    moves = {int(m) for s in specs for m in s.moves if m} | {int(m) for m in extra_moves}
    # A Metronome's calls are moves of the fight too (``reached_moves``), known or gained.
    moves = set(reached_moves(tables, sorted(moves)))
    # Nature Power's call runs its own effect script and data (terrain_calls).
    moves |= set(terrain_calls(tables, trainer, sorted(moves)))
    items = {int(s.item) for s in specs}
    decomp = np.asarray(tables.species_abilities_decomp)
    abilities = {
        int(decomp[personal_row(tables, s.species, s.ability_from)][s.ability_slot]) for s in specs
    }
    bag = any(trainer.items)
    unknown_format = trainer.format in (None, "unknown")
    out: list[str] = []
    for cond in names.rom_change_conditions:
        fight_level = (
            (trainer.format in cond.formats or (unknown_format and bool(cond.formats)))
            or areas & set(cond.areas)
            or trainer.id in cond.trainers
            or (cond.trainer_items and bag)
        )
        if (
            moves & set(cond.moves)
            or abilities & set(cond.abilities)
            or items & set(cond.items)
            or (fight_level and not spec_terms_only)
        ):
            out.append(cond.label)
    return out


def terrain_call(terrain: int) -> int | None:
    """The move Nature Power calls on battle terrain ``terrain`` (``BtlCmd_GetTerrainMove``,
    battle_script.c:8036-8048: ``sTerrainMove[min(terrain, TERRAIN_SPECIAL)]``), or None for
    ``TERRAIN_UNKNOWN``."""
    if terrain == TERRAIN_UNKNOWN:
        return None
    return TERRAIN_MOVES[min(terrain, TERRAIN_SPECIAL)]


def terrain_calls(tables: Tables, trainer: Trainer, moves: Sequence[int]) -> list[int]:
    """The moves ``moves``' Nature Power calls against ``trainer`` (``terrain_call`` of its
    known terrain), as moves of the fight; empty when none is Nature Power or the terrain is
    unknown (``terrain_refusal`` refuses that)."""
    called = terrain_call(trainer.terrain)
    effect = np.asarray(tables.move_ai_effect)
    if called is None or not any(m and int(effect[m]) == NATURE_POWER_EFFECT for m in moves):
        return []
    return [called]


def terrain_refusal(
    tables: Tables, names: Names, trainer: Trainer, moves: Sequence[int]
) -> str | None:
    """Why ``moves`` (decomp move ids) cannot be played against ``trainer``: some read the
    battle terrain (``TERRAIN_EFFECTS``) and the trainer's terrain is unknown, either mixed
    and not named by the run (``cli.fight_terrain``; the line names every terrain its start
    tiles give and the three sources checked, ledger G-18) or unplaced; else None. Every
    move Nature Power calls is modelled (ledger G-21)."""
    effect = np.asarray(tables.move_ai_effect)
    reached = reached_moves(tables, moves)  # a Metronome's calls read the terrain too
    hit = sorted({m for m in reached if int(effect[m]) in TERRAIN_EFFECTS})
    if not hit or trainer.terrain != TERRAIN_UNKNOWN:
        return None
    if trainer.terrains:
        known = [TERRAIN_NAMES[t] for t in trainer.terrains if t != TERRAIN_UNKNOWN]
        unread = (
            " and a map whose tiles cannot be read" if TERRAIN_UNKNOWN in trainer.terrains else ""
        )
        # Manager ruling 1: the tile under the player is the run's route, so the terrain is
        # a run input; the ROM gives the choices, the calc repo and the docs sheet none.
        why = (
            f"the battle terrain is a run input: its start tiles give {', '.join(known)}{unread} "
            "(ROM; the calc repo and the docs sheet record none)"
        )
        ask = ": name it with --terrain"
    else:
        why, ask = "the battle terrain is unknown (no map places the trainer)", ""
    moves_ = ", ".join(names.move_names[m] for m in hit)
    return f"{display_name(trainer)}: {why}, and {moves_} would read it{ask}"


def rom_change_line(label: str) -> str:
    return f"Kaizo ROM: {label} is not modelled and this fight reaches it (docs/kaizo/rom_diff.md)"


def _check_trainer(
    c: _Collector,
    tables: Tables,
    names: Names,
    trainer: Trainer,
    specs: Sequence[PokemonSpec],
    extra_moves: Sequence[int] = (),
    extra_calls: Sequence[int] = (),
) -> None:
    """``extra_moves``: the player's moves beyond its known ones; ``extra_calls``: the calls of a
    Metronome the player gains in battle, read here only by the terrain and ROM-change lines."""
    if POLICY.get(names.game) is None:
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: no transcribed AI policy for this game (spec 5.10)",
            "AI policy",
        )
    support = IMPLEMENTED_FORMATS.get(trainer.format, Support.NONE)
    if trainer.format == "unknown":
        what = f"{display_name(trainer)}: battle format unknown (depends on game state)"
    else:
        what = f"{display_name(trainer)}: a {trainer.format} battle"
    c.add(support, what, "battle format")
    _check_kaizo_fields(c, names, trainer)
    if names.game == KAIZO:
        for label in rom_change_refusals(
            tables, names, trainer, specs, [*extra_moves, *extra_calls]
        ):
            c.add(Support.NONE, rom_change_line(label), "ROM change")
    moves = [m for s in specs for m in s.moves] + [*extra_moves, *extra_calls]
    moves += fight_calls(tables, specs)
    if (line := terrain_refusal(tables, names, trainer, moves)) is not None:
        c.add(Support.NONE, line, "battle terrain")
    # A bag item acts by its party params (engine.bag, E1 rulings G-43, F-15).
    bag = [item for item in trainer.items if item]
    for item in sorted(set(bag)):
        c.add(Support.EXACT, f"{display_name(trainer)}: {names.item_names[item]}", "trainer item")
    if len(set(bag)) > 1:
        mixed = ", ".join(names.item_names[i] for i in bag)
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: a bag of mixed items {mixed}",
            "multi-slot item use",
        )
    _check_ai_flags(c, POLICY.get(names.game), trainer)


def _check_ai_flags(c: _Collector, policy: str | None, trainer: Trainer) -> None:
    """Each flag bit of ``trainer`` against ``policy``'s ``IMPLEMENTED_AI_FLAGS``, and, for
    vanilla, the flag set against ``POLICY_AI_FLAGS``. The Kaizo policy runs the trainer's own
    set (``ai.layout.kaizo_layout``, E1 Ruling R2), so every set flag implemented is the rule.
    A game without a policy is refused already, by its own line."""
    if policy is None:
        return
    flags = int(trainer.ai_flags)
    implemented = IMPLEMENTED_AI_FLAGS[policy]
    for bit in range(flags.bit_length()):
        if flags >> bit & 1:
            name = AI_FLAG_NAMES[bit] if bit < len(AI_FLAG_NAMES) else f"UNUSED_{bit}"
            support = implemented.get(bit, Support.NONE)
            c.add(support, f"{display_name(trainer)}: AI_FLAG_{name}", "AI flag")
    if policy == "vanilla" and flags & POLICY_AI_FLAGS[policy] != POLICY_AI_FLAGS[policy]:
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: AI flags {flags:#x} without all of BASIC, EVAL_ATTACK and "
            "EXPERT, which the AI policy always runs together",
            "AI flag set",
        )


def _check_kaizo_fields(c: _Collector, names: Names, trainer: Trainer) -> None:
    """The Kaizo trainer fields the engine does not model (plan K2b Task 14)."""
    if unknown_start(trainer):
        c.add(Support.NONE, "Battle start: condition unknown (user call)", "start condition")
    if trainer.gauntlet is not None:
        group, position = trainer.gauntlet
        rules = ", ".join(f"{k}={str(v).lower()}" for k, v in trainer.gauntlet_rules.items())
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: gauntlet {group} (position {position}; "
            f"rules {rules or 'not recorded'})",
            "battle format",
        )
    for k in missing_set_data(trainer):
        c.add(
            Support.NONE,
            f"{display_name(trainer)} slot {k}: nature/gender/ability unknown (no PKCalc set)",
            "trainer data",
        )
    if names.game == KAIZO and names.unclassified_rom_changes > 0:
        c.add(
            Support.NONE,
            f"Kaizo ROM: {names.unclassified_rom_changes} code changes unclassified "
            "(docs/kaizo/rom_diff.md)",
            "ROM change",
        )


def _check_weather(c: _Collector, weathers: Sequence[int]) -> None:
    for w in dict.fromkeys(int(w) for w in weathers):
        try:
            weather = Weather(w)
        except ValueError:
            c.add(Support.NONE, f"Battle start: weather id {w}", "weather")
            continue
        support = IMPLEMENTED_WEATHER.get(weather, Support.NONE)
        c.add(support, f"Battle start: {weather.name.title()}", "weather")


def check_opponent(
    tables: Tables,
    names: Names,
    trainer: Trainer,
    opp: Sequence[PokemonSpec],
    start_condition: int,
    player: Sequence[PokemonSpec] = (),
    extra_moves: Sequence[int] = (),
    extra_calls: Sequence[int] = (),
) -> CoverageReport:
    """The trainer's side of ``check_coverage``: its party, bag, AI flags and the weather
    lines (``player``, when given, ``extra_moves``, the player's learnable and Sketch moves, and
    ``extra_calls``, a gained Metronome's calls, only widen the Kaizo ROM-change conditions and
    the terrain check; ``extra_moves`` also joins the AI's Mimic and Transform pool). It runs no
    ``ai_refusals``: those depend on the player's party, so a caller that needs them runs
    them per setup."""
    c = _Collector()
    _check_party(c, tables, names, display_name(trainer), opp)
    _check_trainer(c, tables, names, trainer, [*player, *opp], extra_moves, extra_calls)
    # An "unknown" start condition is refused by _check_trainer, never read as a weather.
    weathers = [start_condition]
    if not unknown_start(trainer):
        weathers.append(start_weather(trainer))
    _check_weather(c, weathers)
    _, error = _fight_ai_tables(
        tables, names, trainer, opp, lambda: foe_moves(tables, player, opp, extra_moves)
    )
    if error is not None:
        c.add(Support.NONE, f"{display_name(trainer)}: {error}", "trainer AI")
    return c.report


def foe_moves(
    tables: Tables,
    player: Sequence[PokemonSpec],
    opp: Sequence[PokemonSpec],
    extra: Sequence[int] = (),
) -> list[int]:
    """The moves the player's side may have in a fight against ``opp``: each ``player`` spec's
    moves and its ``learnable_moves``, then ``extra``; distinct. The AI's Mimic and Transform
    copy one of them, scored by its effect's label (``ai.layout.fight_tables``, G-68)."""
    out: list[int] = []
    for spec in player:
        out += [*spec.moves, *learnable_moves(tables, spec, opp)]
    return list(dict.fromkeys(m for m in (*out, *extra) if m))


def _fight_ai_tables(tables, names, trainer, opp, foe=()):
    """``(tables, error)``: the fight's tables (``ai.layout.fight_tables``) under the trainer's
    transcribed flags (a set with others is refused by its flag lines, ``_check_ai_flags``),
    and the reason its AI layout cannot be built, or None: a read bound past ``WIDTH_CAP``
    (E1 Ruling R6). ``foe``: the player's possible moves (``foe_moves``; a callable is called only
    when the AI copies moves)."""
    from nuzline.ai.layout import KAIZO_FLAG_BITS, fight_tables, layout_of

    flags = int(trainer.ai_flags)
    if POLICY.get(names.game) == "kaizo":
        flags &= sum(1 << bit for bit in KAIZO_FLAG_BITS)
    fight = fight_tables(tables, trainer, opp, flags, foe)
    try:
        layout_of(fight)
    except ValueError as err:
        return fight, str(err)
    return fight, None


def check_coverage(
    tables: Tables,
    names: Names,
    player: list[PokemonSpec],
    trainer: Trainer,
    opp: list[PokemonSpec],
    start_condition: int,
    leads: tuple[int, int] = (0, 0),
    chatot_cry: int | None = None,
) -> CoverageReport:
    """The support level of one fight: the player's party against ``trainer`` with
    party ``opp``, starting in weather ``start_condition`` (a ``Weather``; the
    trainer's own ``start_condition`` is checked too) with the active party slots
    ``leads`` (``make_state``), whose turn-0 state the AI's refusals are run on.
    ``chatot_cry``: the player's Chatot cry parameter 0-2 (``make_state``), None when it is not
    known: a player Chatot that knows or can learn Chatter is then refused, since its confusion
    chance is the save's (battle_script.c:8407-8426; data, not a mechanic: an enemy's is 0), G-70.

    Experience (K3d-exp design §4): each player Pokemon's ``learnable_moves`` (its
    ``reachable_learn_moves`` but the prompt-only ones the learn options mask, ruling D10) go
    through every check its current moves go through (``move_effect_line`` as a "level-up in
    battle" move, the AI's ``player_move_refusal``, ``halved_defense_line``,
    ``terrain_refusal`` and the Kaizo ``rom_change_refusals``), and ``exp_refusals`` adds its
    exp lines. Out of scope and refused elsewhere (§4.7): every format but singles
    (``IMPLEMENTED_FORMATS``; ``expBattler = 2`` is doubles-only, battle_script.c:9716-9720);
    the engine plays only trainer battles; evolution and other after-battle effects are never
    modelled. A level-100 Pokemon gains nothing (battle_script.c:9740) and is not refused."""
    from nuzline.ai.policy import ai_refusals, player_move_refusal  # ai imports the engine

    c = _Collector()
    _check_party(c, tables, names, "Player", player)
    learnable = [learnable_moves(tables, mon, opp) for mon in player]
    party_moves = [m for i, mon in enumerate(player) for m in (*mon.moves, *learnable[i])]
    sketched = [
        sketch_calls(tables, [*mon.moves, *learnable[i]], opp, party_moves)
        for i, mon in enumerate(player)
    ]
    # A Metronome gained in battle (a Sketch copy or a level-up move) calls for its holder too:
    # any move ``Move_CanBeMetronomed`` allows (battle_script.c:4535-4562), its known moves at the
    # call unknown here, so none excluded (the superset). The calls are the holder's: they go to
    # its per-mon lines and to the fight's terrain and ROM lines (``extra_calls``), as a known
    # Metronome's do (``fight_calls``), not into ``extra``. A call is never ``movePrevByBattler``,
    # which Mimic copies (DEFENDER_LAST_MOVE, battle_script.c:4481): that is ``moveTemp``, the
    # Metronome itself (bcp.c:3945-3946), which Mimic refuses (Move_CanBeMimicked, battle_lib.c:
    # 7182-7191); Transform copies slot moves (:5346-5348). Only a called lock move is its next
    # turn's ``moveTemp`` (``locked_calls``), so those join ``extra`` for the AI's Mimic pool.
    effect = np.asarray(tables.move_effect)
    gained_calls = [
        metronome_calls(tables, _NO_MOVES)
        if any(int(effect[m]) == EffectId.METRONOME for m in (*learnable[i], *sketched[i]) if m)
        else []
        for i in range(len(player))
    ]
    own = [m for moves in sketched for m in moves] + party_moves
    extra = sorted(
        {m for moves in (*learnable, *sketched) for m in moves} | {*locked_calls(tables, own)}
    )
    extra_calls = sorted({m for moves in gained_calls for m in moves})
    if chatot_cry is None:
        for slot, mon in enumerate(player):
            if mon.species == SPECIES_CHATOT and MOVE_CHATTER in (*mon.moves, *learnable[slot]):
                c.add(
                    Support.NONE,
                    f"Player {names.species_names[mon.species].upper()} (slot {slot}): Chatter's "
                    "confusion chance needs the save's Chatot cry",
                    "chatter",
                )
    theirs = check_opponent(
        tables, names, trainer, opp, start_condition, player, extra, extra_calls
    )
    c.report.missing.extend(theirs.missing)
    c.report.sampling_only.extend(theirs.sampling_only)
    c.report.level = min(c.report.level, theirs.level)
    fight_moves = [m for mon in (*player, *opp) for m in mon.moves if m] + extra
    for slot, mon in enumerate(player):
        where = f"Player {names.species_names[mon.species]} (slot {slot})"
        row = personal_row(tables, mon.species, mon.form)
        for move in learnable[slot]:
            if move in mon.moves:  # graded by _check_party and ai_refusals already
                continue
            support, name = move_effect_line(tables, names, move, row)
            c.add(support, f"{where}: {name}", "level-up in battle move effect")
            if support != Support.NONE and (reason := player_move_refusal(move)) is not None:
                c.add(Support.NONE, f"{where}: {name} (level-up in battle, {reason})", "trainer AI")
        for move in sketched[slot]:  # a Sketch copies any move of the fight (I-3)
            support, name = move_effect_line(tables, names, move, row)
            c.add(support, f"{where}: {name}", "Sketch move effect")
            if support != Support.NONE and (reason := player_move_refusal(move)) is not None:
                c.add(Support.NONE, f"{where}: {name} (Sketch, {reason})", "trainer AI")
        levels_up = reachable_level(tables, mon, opp) > mon.level
        for line in exp_refusals(tables, names, mon, learnable[slot], fight_moves, levels_up):
            c.add(Support.NONE, f"{where}: {line}", "experience")

    for owner, party, foes in (("Player", player, opp), (display_name(trainer), opp, player)):
        foe_defense = low_defense(tables, foes)
        for slot, mon in enumerate(party):
            extra_moves = [*learnable[slot], *sketched[slot]] if owner == "Player" else ()
            calls = fight_calls(tables, [mon]) + (gained_calls[slot] if owner == "Player" else [])
            for move in [*mon.moves, *(m for m in [*extra_moves, *calls] if m not in mon.moves)]:
                if move and (low := halved_defense_line(tables, move, foe_defense)):
                    where = f"{owner} {names.species_names[mon.species]} (slot {slot})"
                    c.add(Support.NONE, f"{where}: {names.move_names[move]}{low}", "move effect")
                if move and (slow := gyro_speed_line(tables, move, mon)):
                    where = f"{owner} {names.species_names[mon.species]} (slot {slot})"
                    c.add(Support.NONE, f"{where}: {names.move_names[move]}{slow}", "move effect")
    state = make_state(
        tables,
        player,
        opp,
        start_condition,
        opp_items=trainer.items,
        leads=leads,
        terrain=trainer.terrain,
        chatot_cry=chatot_cry or 0,
    )
    # The trainer's flag set (E1 Ruling R2). A set with a flag the AI does not transcribe is
    # refused by its flag lines (_check_ai_flags); its other AI lines are listed under the
    # flags it has that the AI does transcribe.
    fight, error = _fight_ai_tables(
        tables, names, trainer, opp, lambda: foe_moves(tables, player, opp, extra)
    )
    if error is not None:  # listed by check_opponent
        return c.report
    refusals = ai_refusals(fight, state, names)
    # A lead's battle-start copy is in ``state`` already: its lines are listed once.
    copies = [
        r for r in trace_copy_refusals(fight, state, names) if r.split(": ", 1)[1] not in refusals
    ]
    for reason in [*refusals, *copies]:
        c.add(Support.NONE, f"{display_name(trainer)}: {reason}", "trainer AI")
    return c.report


def require_coverage(
    tables: Tables,
    names: Names,
    player: list[PokemonSpec],
    trainer: Trainer,
    opp: list[PokemonSpec],
    start_condition: int,
    leads: tuple[int, int] = (0, 0),
    chatot_cry: int | None = None,
) -> CoverageReport:
    """``check_coverage``, raising ``UnsupportedMechanics`` (every missing item in its
    message) when anything is missing."""
    report = check_coverage(tables, names, player, trainer, opp, start_condition, leads, chatot_cry)
    if report.missing:
        raise UnsupportedMechanics(report)
    return report
