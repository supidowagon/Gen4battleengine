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
guards this), ``ai.outcome`` has one fixed field layout whose width (3840) is a constant
checked once at import against ``MAX_ENUMERATED`` (4096), and ``CoverageReport
.sampling_only`` is therefore always empty. Roark's AI is exact, so the grade is not
needed for this milestone. The first trainer whose draws do not fit that layout needs,
in its plan: a per-trainer width (a layout chosen from the trainer's AI flags and
moveset rather than a module constant), a registry entry that grades the trainer's AI
``SAMPLING_ONLY`` when its width exceeds the budget, and the Stage A and B handling of
a sampling-only opponent, before anything can be marked as such.

The registries list exactly what plan 1a (Tasks 8a-8e: move effects, abilities, held
items) and plan 1b (Task 2b: trainer items; Task 3: Roark's AI flags) implement:

* move effects: every ``ids.EffectId`` but ``UNSUPPORTED``. Moves the tables map to
  ``UNSUPPORTED`` include every move that sets weather, Reflect, Light Screen, Tailwind
  or Trick Room: the engines' ``CheckFieldConditions`` pass (``step.field_conditions``)
  counts screens and weather down, but those moves are not modelled yet (plan ruling);
* abilities, held items, trainer items: every member but ``UNSUPPORTED``;
* AI flags: bits 0-2 (``AI_FLAG_BASIC``, ``AI_FLAG_EVAL_ATTACK``, ``AI_FLAG_EXPERT``).
  ``ai.policy`` always runs those three scripts, so a trainer must have exactly them;
* weather: only ``Weather.NONE``. Rain, sun, sandstorm, hail and fog are refused until
  every weather effect they touch is modelled (plan ruling; the field-condition pass
  exists, ``step.field_conditions``);
* in-battle form changes (plan K2a Ruling 8), on either side: Giratina-Origin without the
  Griseous Orb reverts to Altered Forme at battle start (``battle_lib.c:6412-6417``,
  ``subscript_form_change``) and Shaymin-Sky reverts to Land Forme when frozen
  (``battle_lib.c:4536``, ``battle_controller_player.c:3500``); the engine models neither.
  Rotom, Deoxys and Wormadam forms never change in battle and are read by personal row
  (``Tables.species_form_row``), like Giratina-Origin holding the Orb;
* trainer Pokemon in a data form (controller ruling, K2a Task 6): the opponent's party
  member whose form has its own personal row (``species_form_row[species, form] !=
  species``) is refused. ``TrainerData_BuildParty`` builds it from the base form and sets
  the form afterwards (``src/trainer_data.c:216-218``), so it keeps the base form's stats,
  ability and moves while its types follow the form (``src/pokemon.c:1224``); Kaizo patched
  that function, so the exact rule is per game (K2b). Cosmetic and battle-only forms and
  Arceus, which have no data row, pass;
* battle format: only "single". The engine models one battler per side, so a trainer
  whose header is a double battle, who is fought beside a second enemy trainer
  ("double_two_trainer") or who is fought while the player has a partner ("multi", tag
  battles and walking partners) is refused (``Trainer.format``);
* Kaizo trainer fields (plan K2b Task 14): a permanent Trick Room or Gravity area
  (``Trainer.area_condition``; the engine has neither field effect), a gauntlet (a chain
  of fights, ``Trainer.gauntlet``), a start condition the extraction left to the user
  (``"unknown"``, never read as a weather), a party member with no nature, gender or
  ability data when the trainer has no PID routine (no PKCalc set; plan preflight #34)
  and, for the Kaizo game, ROM code changes not yet classified
  (``Names.unclassified_rom_changes``) and ``not_reached`` ROM changes whose resolved
  condition (``Names.rom_change_conditions``, ``rom_change_refusals``) holds for the fight:
  a battle format or area, a move, ability or held item on either side, a trainer or a
  trainer bag. A condition that does not hold refuses nothing. Area weather and fog stay
  refused through ``start_weather`` like vanilla's; ``Trainer.optional`` refuses nothing.

``check_coverage`` also runs the AI's own refusals (``ai.policy.ai_refusals``: opponent
moves without transcribed script paths, too many script-roll sites, type-chart
modifiers the AI does not model, switching-rule refusals) on the fight's turn-0 state,
and refuses a bag that mixes item ids (the decomp applies the last zeroed slot's item,
docs/discrepancies.md). The AI's state-dependent refusals that can first appear later in
the fight (more than ``SWITCH_SITES`` switch draws, ``AI_OnlyIneffectiveMoves``) are
not visible at turn 0; ``ai_policy`` refuses them when they occur.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.game_constants import POLICY
from nuzline.data.games import KAIZO, VANILLA
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import UNKNOWN_CONDITION, Trainer, missing_set_data
from nuzline.engine.ids import AbilityId, EffectId, ItemId, TrainerItemId, Weather
from nuzline.engine.state import make_state

# include/constants/forms.h:94, :98; generated/species.txt and items.txt (line N is id N-1).
# Checked against each game's data at load (nuzline.data.game_constants).
SPECIES_GIRATINA, GIRATINA_FORM_ORIGIN = 487, 1
SPECIES_SHAYMIN, SHAYMIN_FORM_SKY = 492, 1
ITEM_GRISEOUS_ORB = 112


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
IMPLEMENTED_TRAINER_ITEMS: dict[TrainerItemId, Support] = {
    i: Support.EXACT for i in TrainerItemId if i != TrainerItemId.UNSUPPORTED
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
IMPLEMENTED_AI_FLAGS: dict[int, Support] = {0: Support.EXACT, 1: Support.EXACT, 2: Support.EXACT}
# ai.policy runs exactly these scripts (policy._FLAGS), so they must all be set.
POLICY_AI_FLAGS = 0b111
IMPLEMENTED_WEATHER: dict[Weather, Support] = {
    Weather.NONE: Support.EXACT,
    Weather.SUN: Support.NONE,
    Weather.RAIN: Support.NONE,
    Weather.SANDSTORM: Support.NONE,
    Weather.HAIL: Support.NONE,
    Weather.FOG: Support.NONE,
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


def start_weather(trainer: Trainer) -> Weather:
    """The weather a trainer's fight starts in (``Trainer.start_condition``). A start
    condition of ``"unknown"`` (a user call) is no weather: ``ValueError``, like any
    other unknown value; ``check_opponent`` refuses it instead."""
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


def _check_party(
    c: _Collector,
    tables: Tables,
    names: Names,
    owner: str,
    party: Sequence[PokemonSpec],
    trainer_side: bool = False,
) -> None:
    move_effect = np.asarray(tables.move_effect)
    abilities = np.asarray(tables.species_abilities)
    item_effect = np.asarray(tables.item_effect)
    for slot, mon in enumerate(party):
        where = f"{owner} {names.species_names[mon.species]} (slot {slot})"
        for move in mon.moves:
            if move:
                support = IMPLEMENTED_EFFECTS.get(EffectId(int(move_effect[move])), Support.NONE)
                c.add(support, f"{where}: {names.move_names[move]}", "move effect")
        row = personal_row(tables, mon.species, mon.form)  # ValueError: no such form
        ability = AbilityId(int(abilities[row, mon.ability_slot]))
        ability_name = names.ability_names[
            names.species_abilities(mon.species, mon.form)[mon.ability_slot]
        ]
        c.add(
            IMPLEMENTED_ABILITIES.get(ability, Support.NONE), f"{where}: {ability_name}", "ability"
        )
        item = ItemId(int(item_effect[mon.item]))
        item_name = names.item_names[mon.item] if mon.item else "no item"
        c.add(IMPLEMENTED_ITEMS.get(item, Support.NONE), f"{where}: {item_name}", "held item")
        _check_form(c, where, mon)
        if trainer_side and row != mon.species:
            c.add(Support.NONE, f"{where}: {_data_form_reason(names)}", "form")


def _data_form_reason(names: Names) -> str:
    """Why a trainer Pokemon in a data form is refused, true for ``names``' game: vanilla's
    TrainerData_BuildParty builds it from the base form (src/trainer_data.c:216-218);
    Kaizo's patched builder computes its stats after the form (docs/kaizo/rom_diff.md)."""
    if names.game == VANILLA:
        return "trainer Pokemon in a data form keeps its base form's stats, ability and moves"
    return (
        "trainer Pokemon in a data form: this game's party builder applies the form before "
        "computing its stats, which the engine's trainer party build does not model"
    )


def _check_form(c: _Collector, where: str, mon: PokemonSpec) -> None:
    """Ruling 8: a form the battle changes, which the engine does not model."""
    if (
        mon.species == SPECIES_GIRATINA
        and mon.form == GIRATINA_FORM_ORIGIN
        and mon.item != ITEM_GRISEOUS_ORB
    ):
        c.add(Support.NONE, f"{where}: Origin Forme without the Griseous Orb", "form change")
    if mon.species == SPECIES_SHAYMIN and mon.form == SHAYMIN_FORM_SKY:
        c.add(Support.NONE, f"{where}: Sky Forme reverts when frozen", "form change")


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
    ability (``species_abilities_decomp`` at the spec's personal row and ability slot) or a
    held item is listed; the trainer is listed; or the condition covers trainer items and the
    bag holds one. ``spec_terms_only`` keeps just the moves, abilities and items: the pool
    asks it per candidate, the fight-level terms being the fight's own."""
    area = trainer.area_condition
    areas = {
        name for name, on in (("trick_room", area.trick_room), ("gravity", area.gravity)) if on
    }
    moves = {int(m) for s in specs for m in s.moves if m} | {int(m) for m in extra_moves}
    items = {int(s.item) for s in specs}
    decomp = np.asarray(tables.species_abilities_decomp)
    abilities = {
        int(decomp[personal_row(tables, s.species, s.form)][s.ability_slot]) for s in specs
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


def rom_change_line(label: str) -> str:
    return f"Kaizo ROM: {label} is not modelled and this fight reaches it (docs/kaizo/rom_diff.md)"


def _check_trainer(
    c: _Collector,
    tables: Tables,
    names: Names,
    trainer: Trainer,
    specs: Sequence[PokemonSpec],
) -> None:
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
        for label in rom_change_refusals(tables, names, trainer, specs):
            c.add(Support.NONE, rom_change_line(label), "ROM change")
    effect = np.asarray(tables.trainer_item_effect)
    bag = [item for item in trainer.items if item]
    for item in bag:
        support = IMPLEMENTED_TRAINER_ITEMS.get(TrainerItemId(int(effect[item])), Support.NONE)
        c.add(support, f"{display_name(trainer)}: {names.item_names[item]}", "trainer item")
    if len(set(bag)) > 1:
        mixed = ", ".join(names.item_names[i] for i in bag)
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: a bag of mixed items {mixed}",
            "multi-slot item use",
        )
    flags = int(trainer.ai_flags)
    for bit in range(flags.bit_length()):
        if flags >> bit & 1:
            name = AI_FLAG_NAMES[bit] if bit < len(AI_FLAG_NAMES) else f"UNUSED_{bit}"
            support = IMPLEMENTED_AI_FLAGS.get(bit, Support.NONE)
            c.add(support, f"{display_name(trainer)}: AI_FLAG_{name}", "AI flag")
    if flags & POLICY_AI_FLAGS != POLICY_AI_FLAGS:
        c.add(
            Support.NONE,
            f"{display_name(trainer)}: AI flags {flags:#x} without all of BASIC, EVAL_ATTACK and "
            "EXPERT, which the AI policy always runs together",
            "AI flag set",
        )


def _check_kaizo_fields(c: _Collector, names: Names, trainer: Trainer) -> None:
    """The Kaizo trainer fields the engine does not model (plan K2b Task 14)."""
    area = trainer.area_condition
    permanent = "permanent " if area.permanent else ""
    if area.trick_room:
        c.add(Support.NONE, f"Battle start: {permanent}Trick Room", "field condition")
    if area.gravity:
        c.add(Support.NONE, f"Battle start: {permanent}Gravity", "field condition")
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
) -> CoverageReport:
    """The trainer's side of ``check_coverage``: its party, bag, AI flags and the weather
    lines (``player``, when given, only widens the Kaizo ROM-change conditions). It runs no ``ai_refusals``: those depend on the player's party, so a caller
    that needs them runs them per setup."""
    c = _Collector()
    _check_party(c, tables, names, display_name(trainer), opp, trainer_side=True)
    _check_trainer(c, tables, names, trainer, [*player, *opp])
    # An "unknown" start condition is refused by _check_trainer, never read as a weather.
    weathers = [start_condition]
    if not unknown_start(trainer):
        weathers.append(start_weather(trainer))
    _check_weather(c, weathers)
    return c.report


def check_coverage(
    tables: Tables,
    names: Names,
    player: list[PokemonSpec],
    trainer: Trainer,
    opp: list[PokemonSpec],
    start_condition: int,
    leads: tuple[int, int] = (0, 0),
) -> CoverageReport:
    """The support level of one fight: the player's party against ``trainer`` with
    party ``opp``, starting in weather ``start_condition`` (a ``Weather``; the
    trainer's own ``start_condition`` is checked too) with the active party slots
    ``leads`` (``make_state``), whose turn-0 state the AI's refusals are run on."""
    c = _Collector()
    _check_party(c, tables, names, "Player", player)
    theirs = check_opponent(tables, names, trainer, opp, start_condition, player)
    c.report.missing.extend(theirs.missing)
    c.report.sampling_only.extend(theirs.sampling_only)
    c.report.level = min(c.report.level, theirs.level)
    from nuzline.ai.policy import ai_refusals  # nuzline.ai imports the engine

    state = make_state(tables, player, opp, start_condition, opp_items=trainer.items, leads=leads)
    for reason in ai_refusals(tables, state, names):
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
) -> CoverageReport:
    """``check_coverage``, raising ``UnsupportedMechanics`` (every missing item in its
    message) when anything is missing."""
    report = check_coverage(tables, names, player, trainer, opp, start_condition, leads)
    if report.missing:
        raise UnsupportedMechanics(report)
    return report
