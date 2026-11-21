"""Trainers from ``data/intermediate/trainers.json`` and their battle parties.

``build_trainer_party`` reproduces ``TrainerData_BuildParty``
(``src/trainer_data.c:176``) and the ``BoxPokemon`` initializer it calls
(``sub_02073E18``, ``src/pokemon.c:405``): IVs from the difficulty, EVs 0, a PID
from the LCRNG seeded per party member, and from the PID the nature, gender and
ability slot. Listed moves overwrite all four default move slots.

A party member in a data form (G-41) is built from its base species: ``TrainerData_BuildParty``
sets ``MON_DATA_FORM`` (src/trainer_data.c:216-218, :246-248, :272-274, :303-305) only
after ``Pokemon_InitWith`` has read the base species' friendship (src/pokemon.c:435-436),
ability (:473-485), gender ratio (``Pokemon_GetGenderOf``, :2707-2715) and level-up
learnset (``BoxPokemon_SetDefaultMoves`` reads ``MON_DATA_FORM``, still 0, :3819-3829), and
computed the stats from the base row (:402). Setting the form only stores it (:1608-1609);
its types follow the form (``MON_DATA_TYPE_1/2``, :1219-1225). Vanilla never recomputes the
stats (``PokemonSpec.stats_form`` 0); Kaizo's patched builder calls ``Pokemon_CalcStats``
after the form (docs/kaizo/rom_diff.md, "TrainerData_BuildParty, decoded", step 11), which
loads the form's row (pokemon.c:595) and keeps a full-HP Pokemon at full HP (:633-641).

Kaizo (spec 5.4, 7.1; plan K2b Task 14) patched ``TrainerData_BuildParty``: its trainer
struct gained a per-Pokemon override byte (``TrainerMon.ability_override``) and its PIDs
no longer follow the vanilla routine for every Pokemon. Until the patched routine is
transcribed, a Kaizo trainer has ``pid_routine`` None and each party member carries its
nature, gender and ability slot explicitly (from PKCalc); explicit values always win over
the routine, and the patched builder's friendship (255, or 0 for a Pokemon with
Frustration) is explicit too. Every Kaizo field defaults to vanilla's reading, so vanilla
``trainers.json`` loads unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nuzline.data.forms import form_row
from nuzline.data.games import KAIZO, game_of_intermediate
from nuzline.data.spec import PokemonSpec

MAX_IV = 31  # MAX_IVS_SINGLE_STAT
MAX_IV_SCALE = 255  # include/struct_defs/trainer_data.h:15

# include/math_util.h
LCRNG_MULTIPLIER = 1103515245
LCRNG_INCREMENT = 24691

# generated/genders.txt
GENDER_MALE, GENDER_FEMALE, GENDER_NONE = 0, 1, 2

# generated/gender_ratios.txt
GENDER_RATIO_MALE_ONLY = 0
GENDER_RATIO_FEMALE_ONLY = 254
GENDER_RATIO_NO_GENDER = 255

# TrainerData_BuildParty's genderMod: the PID's low byte.
GENDER_MOD_FEMALE_CLASS = 120
GENDER_MOD_OTHER_CLASS = 136

MOVE_SLOTS = 4

# generated/battle_terrains.txt (enum BattleTerrain), id = index; TERRAIN_MAX is reachable:
# CalcTerrain returns it for a battle background past its table
# (src/field_battle_data_transfer.c:491-493). Checked against the decomp by
# tests/tools/rom/test_terrain.py.
TERRAIN_NAMES = (
    "TERRAIN_PLAIN",
    "TERRAIN_SAND",
    "TERRAIN_GRASS",
    "TERRAIN_PUDDLE",
    "TERRAIN_MOUNTAIN",
    "TERRAIN_CAVE",
    "TERRAIN_SNOW",
    "TERRAIN_WATER",
    "TERRAIN_ICE",
    "TERRAIN_BUILDING",
    "TERRAIN_GREAT_MARSH",
    "TERRAIN_BRIDGE",
    "TERRAIN_AARON",
    "TERRAIN_BERTHA",
    "TERRAIN_FLINT",
    "TERRAIN_LUCIAN",
    "TERRAIN_CYNTHIA",
    "TERRAIN_DISTORTION_WORLD",
    "TERRAIN_BATTLE_TOWER",
    "TERRAIN_BATTLE_FACTORY",
    "TERRAIN_BATTLE_ARCADE",
    "TERRAIN_BATTLE_CASTLE",
    "TERRAIN_BATTLE_HALL",
    "TERRAIN_GIRATINA",
    "TERRAIN_MAX",
)
# Trainer.terrain when the trainer's maps give several terrains (mixed) or none (tools/
# terrain.py TERRAIN_UNKNOWN).
TERRAIN_UNKNOWN = -1


@dataclass(frozen=True)
class TrainerMon:
    species: int
    level: int
    difficulty: int  # the decomp's iv_scale, 0-255
    moves: tuple[int, int, int, int] | None  # None: the species' default moveset
    item: int
    form: int
    # Explicit values (Kaizo, from PKCalc) win over the PID routine; None: computed.
    nature: int | None = None
    gender: int | None = None
    ability_slot: int | None = None
    # Kaizo's per-Pokemon Ability Number byte (16: ability slot 1, 32: slot 2, 2: female;
    # spec 5.4); 0 in vanilla. The patched routine is not transcribed, so a member with a
    # non-zero override needs explicit nature, gender and ability slot (missing_set_data).
    ability_override: int = 0
    friendship: int | None = None  # None: the species' base friendship


# Area weathers (the sheet grammar's PERMANENT RAIN|SUN|SANDSTORM|HAIL and the bare
# (RAIN)/(SUN)/(HAIL), read as permanent) plus "unknown": a start condition the
# extraction leaves to the user (Kaizo trainers 318, 325, 343, 509, 559, 610, 649, 676,
# 746, 747), which is never read as a weather.
AREA_WEATHERS = ("rain", "sun", "sandstorm", "hail")
UNKNOWN_CONDITION = "unknown"
# Overworld weather ids (include/constants/overworld_weather.h:4, :18): the map header's
# weather byte and FieldOverworldState.weather. Fog (14) is the one weather field Defog
# clears (``AreaCondition.fog_headers``).
OVERWORLD_WEATHER_CLEAR = 0
OVERWORLD_WEATHER_FOG = 14


@dataclass(frozen=True)
class AreaCondition:
    """What a fight starts with, apart from ability weather (spec 5.8): the area's
    ``weather`` (one of ``AREA_WEATHERS``, ``"unknown"`` or None), fog, Trick Room and
    Gravity. ``permanent``: they never expire (every Kaizo area condition, and vanilla's
    overworld weather).

    ``fog_headers``: the map headers placing a fog fight whose header weather is
    OVERWORLD_WEATHER_FOG (14), sorted; empty for every other fight. Only that weather
    turns clear when FLAG_DEFOG_ACTIVE is set (FieldMapChange_UpdateGameData,
    src/field_map_change.c:275-280), and field Defog sets the flag and the overworld weather
    clear at once (res/field/scripts/scripts_field_moves.s:385-387; scrcmd.c:3667-3674), so
    on these headers the fog start is a run input (ledger G-15): the battle reads the
    overworld weather (src/field_battle_data_transfer.c:267). Deep fog (15) is never cleared.

    ``script_weather`` / ``script_window`` (Kaizo, ledger G-24): a map script the fight can
    meet leaves this weather (one of ``AREA_WEATHERS``) on the map: ``script_window`` =
    (map header, overworld weather id it sets, saved var, value) from
    ``rom_script_weather.json``. The Kaizo command 0x9C sets the overworld weather and the
    script then sets the var to the value (Route 213: scr_seq 459 entry 12, rain, VAR 0x40FA
    := 8); the rain lasts until ScrCmd_0C4 after the Chelle win (src/scrcmd.c:3667-3674) or a
    map change (src/field_map_change.c:272-280), and the header's own weather is clear, so
    the fight starts in this weather or in none: a run input (``cli.fight_start``), with
    ``weather`` the default."""

    weather: str | None = None
    fog: bool = False
    trick_room: bool = False
    gravity: bool = False
    permanent: bool = True
    fog_headers: tuple[int, ...] = ()
    script_weather: str | None = None
    script_window: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.weather is not None and self.weather not in (*AREA_WEATHERS, UNKNOWN_CONDITION):
            raise ValueError(f"area weather {self.weather!r} is not one of {AREA_WEATHERS}")
        object.__setattr__(self, "fog_headers", tuple(sorted(self.fog_headers)))
        if self.fog_headers and not self.fog:
            raise ValueError(f"fog headers {self.fog_headers} on a fight without fog")
        object.__setattr__(self, "script_window", tuple(self.script_window))
        if (self.script_weather is None) != (not self.script_window) or (
            self.script_window and len(self.script_window) != 4
        ):
            raise ValueError(
                f"script weather {self.script_weather!r} needs a window (header, overworld "
                f"weather, var, value), got {self.script_window}"
            )
        if self.script_weather is not None and self.script_weather not in AREA_WEATHERS:
            raise ValueError(f"script weather {self.script_weather!r} is not an area weather")

    @property
    def fog_clearable(self) -> bool:
        """Field Defog can clear this fight's start fog (``fog_headers``)."""
        return bool(self.fog_headers)

    @classmethod
    def from_start_condition(cls, start_condition: str) -> AreaCondition:
        """The area condition of a ``trainers.json`` ``start_condition``."""
        if start_condition == "none":
            return cls()
        if start_condition == "fog":
            return cls(fog=True)
        if start_condition in (*AREA_WEATHERS, UNKNOWN_CONDITION):
            return cls(weather=start_condition)
        raise ValueError(f"unknown start condition {start_condition!r}")


@dataclass(frozen=True)
class PartnerVariant:
    """One team the player's partner can field in a multi battle: ``trainer`` is the
    partner's trainer id for a player of gender ``player_gender`` (0 male, 1 female) whose
    starter is species ``starter``. A field the team does not depend on is None."""

    trainer: int
    starter: int | None
    player_gender: int | None


@dataclass(frozen=True)
class Trainer:
    """One ``trainers.json`` entry; ``trainer_class`` is the JSON's ``class``.

    ``format`` is the battle format: "single" or "double" from the trainer's header, or
    "double_two_trainer" (fought beside ``opponent_2`` against the player alone) or
    "multi" (the player fights beside a partner, whose possible teams are ``partner``;
    None: a partner is present but the scripts do not say which, Kaizo only) from the field
    scripts (``tools/battle_scripts.py``). ``opponent_2`` is the other
    enemy of a two-trainer or tag battle, else None. ``walking_partner`` is the trainer ids
    of the walking partner a map's script can set where the trainer is placed: the fight may
    be fought under the partner flag. The post-win heal (``encounter.c:197-199``) applies
    only while the escort is active (``SetHasPartner`` runs in its own script, e.g.
    ``scripts_eterna_forest.s:57-58``), a run input; empty off a walking-partner map.

    Kaizo fields (plan K2b Task 14; all defaulted to vanilla's reading): ``label`` names
    one of several trainers sharing a name; ``pid_routine`` is "vanilla" when
    ``build_trainer_party`` may compute nature, gender and ability from the PID, None when
    every party member must carry them; ``player_starter`` is the starter species a rival
    or partner variant is fought with; ``gauntlet`` is (group, position) in a chain of
    fights with ``gauntlet_rules`` (``heal_between``, ``change_team``; empty: not
    recorded); ``optional`` and ``left_side`` (the split's Left Side AI) come from the
    spreadsheet; ``area_condition`` is what the fight starts with (built from
    ``start_condition`` when the JSON has none); ``split`` and ``split_order`` (split
    index, line) place the fight in the level-cap splits; ``sources`` cites each field.

    ``terrain`` is the battle terrain (a ``TERRAIN_NAMES`` id) of the trainer's maps, or
    ``TERRAIN_UNKNOWN`` when ``terrains``, every terrain those maps can give, has several
    values (mixed) or none (no map places the trainer); tools/terrain.py, K3b2 Task 9.

    A ``Trainer`` is unhashable: ``gauntlet_rules`` and ``sources`` are dicts, so the
    derived ``__hash__`` raises ``TypeError``; key caches by ``Trainer.id``."""

    id: int
    name: str
    trainer_class: str
    class_id: int
    class_gender: int
    ai_flags: int
    items: tuple[int, ...]
    party: tuple[TrainerMon, ...]
    start_condition: str
    format: str
    opponent_2: int | None = None
    partner: tuple[PartnerVariant, ...] | None = ()
    walking_partner: tuple[int, ...] = ()
    label: str | None = None
    pid_routine: str | None = "vanilla"
    player_starter: int | None = None
    gauntlet: tuple[str, int] | None = None
    gauntlet_rules: dict[str, bool] = field(default_factory=dict)
    optional: bool = False
    left_side: bool = False
    area_condition: AreaCondition = AreaCondition()
    split: str | None = None
    split_order: tuple[int, int] | None = None
    sources: dict[str, str] = field(default_factory=dict)
    terrain: int = TERRAIN_UNKNOWN
    terrains: tuple[int, ...] = ()


def run_terrain(trainer: Trainer) -> bool:
    """The fight's battle terrain was named by the run (``cli.fight_terrain``, ledger G-18):
    its start tiles give several terrains and ``trainer.terrain`` is one of them."""
    known = [t for t in trainer.terrains if t != TERRAIN_UNKNOWN]
    return len(known) > 1 and trainer.terrain in known


def _mon(d: dict) -> TrainerMon:
    moves = d["moves"]
    return TrainerMon(
        species=d["species"],
        level=d["level"],
        difficulty=d["difficulty"],
        moves=None if moves is None else tuple(moves),
        item=d["item"],
        form=d["form"],
        nature=d.get("nature"),
        gender=d.get("gender"),
        ability_slot=d.get("ability_slot"),
        ability_override=d.get("ability_override", 0),
        friendship=d.get("friendship"),
    )


def _area(t: dict) -> AreaCondition:
    area = t.get("area_condition")
    if area is None:  # vanilla: the fog headers sit beside the start condition
        cond = AreaCondition.from_start_condition(t["start_condition"])
        return dataclasses.replace(cond, fog_headers=tuple(t.get("fog_headers", ())))
    return AreaCondition(**area)


def _pair(t: dict, key: str):
    value = t.get(key)
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"trainer {t['id']}: {key} {value!r} is not a pair")
    return tuple(value)


def _read(dir: Path, name: str):
    return json.loads((Path(dir) / f"{name}.json").read_text(encoding="utf-8"))


def load_trainers(dir: Path) -> list[Trainer]:
    """Every trainer in ``<dir>/trainers.json``, indexed by trainer id."""
    rows = _read(dir, "trainers")
    if rows and "format" not in rows[0]:
        raise ValueError(
            f"{Path(dir) / 'trainers.json'} has no battle format: re-run tools/extract_decomp.py"
        )
    if rows and not {"opponent_2", "partner"} <= rows[0].keys():
        raise ValueError(
            f"{Path(dir) / 'trainers.json'} has no battle structure: re-run tools/extract_decomp.py"
        )
    if rows and not {"terrain", "terrains"} <= rows[0].keys():
        raise ValueError(
            f"{Path(dir) / 'trainers.json'} has no battle terrain: re-run tools/extract_decomp.py"
        )
    if rows and "walking_partner" not in rows[0]:
        raise ValueError(
            f"{Path(dir) / 'trainers.json'} has no walking_partner: re-run tools/extract_decomp.py"
        )
    return [
        Trainer(
            id=t["id"],
            name=t["name"],
            trainer_class=t["class"],
            class_id=t["class_id"],
            class_gender=t["class_gender"],
            ai_flags=t["ai_flags"],
            items=tuple(t["items"]),
            party=tuple(_mon(m) for m in t["party"]),
            start_condition=t["start_condition"],
            format=t["format"],
            opponent_2=t["opponent_2"],
            partner=None
            if t["partner"] is None
            else tuple(
                PartnerVariant(p["trainer"], p["starter"], p["player_gender"]) for p in t["partner"]
            ),
            walking_partner=tuple(t["walking_partner"]),
            label=t.get("label"),
            pid_routine=t.get("pid_routine", "vanilla"),
            player_starter=t.get("player_starter"),
            gauntlet=_pair(t, "gauntlet"),
            gauntlet_rules=dict(t.get("gauntlet_rules", {})),
            optional=t.get("optional", False),
            left_side=t.get("left_side", False),
            area_condition=_area(t),
            split=t.get("split"),
            split_order=_pair(t, "split_order"),
            sources=dict(t.get("sources", {})),
            terrain=t["terrain"],
            terrains=tuple(t["terrains"]),
        )
        for t in rows
    ]


class IntermediateData:
    """Thin loader of the extractor's species, form and learnset JSON. ``species`` is indexed
    by species id, ``personal`` and ``learnsets`` by personal row (the species, then the
    data forms of ``forms.json``), and ``form_rows`` maps ``[species][form]`` to the row.
    ``game`` is the id of the game the intermediates belong to (by default the game whose
    ``Game.intermediate_dir`` is ``dir``, ``games.game_of_intermediate``). ``patched_builder``
    is True for Kaizo's: its trainers are built by the patched ``TrainerData_BuildParty``,
    which recalculates stats after the form (docs/kaizo/rom_diff.md, "TrainerData_BuildParty,
    decoded", step 11). It keys on the game id, not on a file in ``dir``."""

    def __init__(self, dir: Path, game: str | None = None):
        self.game = game_of_intermediate(dir) if game is None else game
        self.patched_builder = self.game == KAIZO
        self.species: list[dict] = _read(dir, "species")
        self.personal: list[dict] = self.species + _read(dir, "forms")
        self.form_rows: list[list[int]] = _read(dir, "form_rows")
        self.learnsets: dict[int, list[tuple[int, int]]] = {
            int(k): [(lvl, move) for lvl, move in v] for k, v in _read(dir, "learnsets").items()
        }

    def row(self, species: int, form: int = 0) -> int:
        """The personal row of ``species``' ``form``; ``ValueError`` if it has no such form."""
        return form_row(self.form_rows, species, form)


def difficulty_to_ivs(difficulty: int) -> int:
    return difficulty * MAX_IV // MAX_IV_SCALE


def lcrng_next(state: int) -> tuple[int, int]:
    """``LCRNG_Next`` (``src/math_util.c``): the new u32 state and its high 16 bits."""
    state = (state * LCRNG_MULTIPLIER + LCRNG_INCREMENT) & 0xFFFFFFFF
    return state, state >> 16


def trainer_pid(
    difficulty: int, level: int, species: int, trainer_id: int, class_id: int, *, female_class: bool
) -> int:
    """The PID ``TrainerData_BuildParty`` gives a party member."""
    state = rnd = (difficulty + level + species + trainer_id) & 0xFFFFFFFF
    for _ in range(class_id):
        state, rnd = lcrng_next(state)
    gender_mod = GENDER_MOD_FEMALE_CLASS if female_class else GENDER_MOD_OTHER_CLASS
    return ((rnd << 8) + gender_mod) & 0xFFFFFFFF


def gender_of(gender_ratio: int, pid: int) -> int:
    """``SpeciesData_GetGenderOf`` (``src/pokemon.c``)."""
    if gender_ratio == GENDER_RATIO_MALE_ONLY:
        return GENDER_MALE
    if gender_ratio == GENDER_RATIO_FEMALE_ONLY:
        return GENDER_FEMALE
    if gender_ratio == GENDER_RATIO_NO_GENDER:
        return GENDER_NONE
    return GENDER_FEMALE if gender_ratio > (pid & 0xFF) else GENDER_MALE


def default_moves(learnset: Sequence[tuple[int, int]], level: int) -> tuple[int, int, int, int]:
    """``BoxPokemon_SetDefaultMoves`` (``src/pokemon.c:3819``): walk the level-up
    learnset in order up to ``level``; a known move is skipped, a new one fills the
    first empty slot or, with all four full, is appended after dropping slot 0."""
    slots: list[int] = []
    for learn_level, move in learnset:
        if learn_level > level:
            break
        if move in slots:
            continue
        if len(slots) == MOVE_SLOTS:
            slots.pop(0)
        slots.append(move)
    return tuple(slots + [0] * (MOVE_SLOTS - len(slots)))  # type: ignore[return-value]


def missing_set_data(trainer: Trainer) -> list[int]:
    """The party slots ``build_trainer_party`` cannot build: a member lacking an explicit
    nature, gender or ability slot when the trainer has no PID routine (``pid_routine``
    None) or the member has a Kaizo ``ability_override`` (untranscribed routine)."""
    return [
        k
        for k, mon in enumerate(trainer.party)
        if (trainer.pid_routine is None or mon.ability_override != 0)
        and None in (mon.nature, mon.gender, mon.ability_slot)
    ]


def build_trainer_party(trainer: Trainer, inter: IntermediateData) -> list[PokemonSpec]:
    """The trainer's battle party. Explicit ``TrainerMon`` nature, gender, ability slot and
    friendship win; the rest follows the vanilla routine. With ``pid_routine`` None a
    member without all three of nature, gender and ability slot is a ``ValueError``, as
    is one with a non-zero ``ability_override`` (``missing_set_data``)."""
    missing = missing_set_data(trainer)
    if missing:
        raise ValueError(f"trainer {trainer.id} slot {missing[0]}: no nature/gender/ability data")
    party = []
    for mon in trainer.party:
        inter.row(mon.species, mon.form)  # ValueError: no such form
        # Pokemon_InitWith reads the base species (module docstring, G-41).
        species = inter.personal[inter.row(mon.species)]
        pid = trainer_pid(
            mon.difficulty,
            mon.level,
            mon.species,
            trainer.id,
            trainer.class_id,
            female_class=trainer.class_gender == GENDER_FEMALE,
        )
        # sub_02073E18: PID bit 0 picks the second ability only when the species has one.
        ability_slot = 1 if species["abilities"][1] != 0 and pid & 1 else 0
        if mon.ability_slot is not None:
            ability_slot = mon.ability_slot
        if species["abilities"][1] == 0:  # ABILITY_NONE in slot 2: slot 1 (pokemon.c:476-484)
            ability_slot = 0
        nature = pid % 25 if mon.nature is None else mon.nature  # Pokemon_GetNatureOf
        gender = gender_of(species["gender_ratio"], pid) if mon.gender is None else mon.gender
        friendship = species["base_friendship"] if mon.friendship is None else mon.friendship
        moves = mon.moves
        if moves is None:
            moves = default_moves(inter.learnsets[inter.row(mon.species)], mon.level)
        ivs = difficulty_to_ivs(mon.difficulty)
        party.append(
            PokemonSpec(
                species=mon.species,
                level=mon.level,
                nature=nature,
                ivs=(ivs,) * 6,
                evs=(0,) * 6,
                moves=moves,
                item=mon.item,
                ability_slot=ability_slot,
                gender=gender,
                friendship=friendship,
                form=mon.form,
                ability_form=0,
                # By game: vanilla's builder keeps the base row's stats, Kaizo's the form's.
                stats_form=None if inter.patched_builder else 0,
            )
        )
    return party
