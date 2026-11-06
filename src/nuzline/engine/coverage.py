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
  or Trick Room: the engines have no ``CheckFieldConditions`` pass (step.py), so those
  stay unsupported (plan ruling);
* abilities, held items, trainer items: every member but ``UNSUPPORTED``;
* AI flags: bits 0-2 (``AI_FLAG_BASIC``, ``AI_FLAG_EVAL_ATTACK``, ``AI_FLAG_EXPERT``).
  ``ai.policy`` always runs those three scripts, so a trainer must have exactly them;
* weather: only ``Weather.NONE``. Rain, sun, sandstorm, hail and fog need the
  field-condition pass and are refused (plan ruling);
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
  battles and walking partners) is refused (``Trainer.format``).

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
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import Trainer
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
    """The weather a trainer's fight starts in (``Trainer.start_condition``)."""
    try:
        return START_CONDITIONS[trainer.start_condition]
    except KeyError:
        raise ValueError(f"unknown start condition {trainer.start_condition!r}") from None


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
            c.add(
                Support.NONE,
                f"{where}: trainer Pokemon in a data form keeps its base form's stats, "
                "ability and moves",
                "form",
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


def _check_trainer(c: _Collector, tables: Tables, names: Names, trainer: Trainer) -> None:
    if POLICY.get(names.game) is None:
        c.add(
            Support.NONE,
            f"{trainer.name}: no transcribed AI policy for this game (spec 5.10)",
            "AI policy",
        )
    support = IMPLEMENTED_FORMATS.get(trainer.format, Support.NONE)
    c.add(support, f"{trainer.name}: a {trainer.format} battle", "battle format")
    effect = np.asarray(tables.trainer_item_effect)
    bag = [item for item in trainer.items if item]
    for item in bag:
        support = IMPLEMENTED_TRAINER_ITEMS.get(TrainerItemId(int(effect[item])), Support.NONE)
        c.add(support, f"{trainer.name}: {names.item_names[item]}", "trainer item")
    if len(set(bag)) > 1:
        mixed = ", ".join(names.item_names[i] for i in bag)
        c.add(Support.NONE, f"{trainer.name}: a bag of mixed items {mixed}", "multi-slot item use")
    flags = int(trainer.ai_flags)
    for bit in range(flags.bit_length()):
        if flags >> bit & 1:
            name = AI_FLAG_NAMES[bit] if bit < len(AI_FLAG_NAMES) else f"UNUSED_{bit}"
            support = IMPLEMENTED_AI_FLAGS.get(bit, Support.NONE)
            c.add(support, f"{trainer.name}: AI_FLAG_{name}", "AI flag")
    if flags & POLICY_AI_FLAGS != POLICY_AI_FLAGS:
        c.add(
            Support.NONE,
            f"{trainer.name}: AI flags {flags:#x} without all of BASIC, EVAL_ATTACK and "
            "EXPERT, which the AI policy always runs together",
            "AI flag set",
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
) -> CoverageReport:
    """The trainer's side of ``check_coverage``: its party, bag, AI flags and the weather
    lines. It runs no ``ai_refusals``: those depend on the player's party, so a caller
    that needs them runs them per setup."""
    c = _Collector()
    _check_party(c, tables, names, trainer.name, opp, trainer_side=True)
    _check_trainer(c, tables, names, trainer)
    _check_weather(c, [start_condition, start_weather(trainer)])
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
    theirs = check_opponent(tables, names, trainer, opp, start_condition)
    c.report.missing.extend(theirs.missing)
    c.report.sampling_only.extend(theirs.sampling_only)
    c.report.level = min(c.report.level, theirs.level)
    from nuzline.ai.policy import ai_refusals  # nuzline.ai imports the engine

    state = make_state(tables, player, opp, start_condition, opp_items=trainer.items, leads=leads)
    for reason in ai_refusals(tables, state):
        c.add(Support.NONE, f"{trainer.name}: {reason}", "trainer AI")
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
