"""Trainers from ``data/intermediate/trainers.json`` and their battle parties.

``build_trainer_party`` reproduces ``TrainerData_BuildParty``
(``src/trainer_data.c:176``) and the ``BoxPokemon`` initializer it calls
(``sub_02073E18``, ``src/pokemon.c:405``): IVs from the difficulty, EVs 0, a PID
from the LCRNG seeded per party member, and from the PID the nature, gender and
ability slot. Listed moves overwrite all four default move slots.

A party member's ability, gender ratio, base friendship and default moves are read from
its form's personal row (``form_rows.json``, ``forms.json``; plan K2a Task 6). The decomp
differs (open question, K2a Task 6 report): ``TrainerData_BuildParty`` sets
``MON_DATA_FORM`` only after ``Pokemon_InitWith`` has picked the ability and default moves
from the base species (``src/pokemon.c:473-489``) and computed the stats
(``src/pokemon.c:402``), and nothing recomputes them (``src/trainer_data.c:216-218``).
Vanilla's only data-form trainer Pokemon (Devon's Wormadam-Sandy and -Trash, trainer
491) has the same ability, gender ratio, friendship and level-24 default moves either way.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nuzline.data.forms import form_row
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


@dataclass(frozen=True)
class TrainerMon:
    species: int
    level: int
    difficulty: int  # the decomp's iv_scale, 0-255
    moves: tuple[int, int, int, int] | None  # None: the species' default moveset
    item: int
    form: int


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
    "multi" (the player fights beside a partner, whose possible teams are ``partner``)
    from the field scripts (``tools/battle_scripts.py``). ``opponent_2`` is the other
    enemy of a two-trainer or tag battle, else None."""

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
    partner: tuple[PartnerVariant, ...] = ()


def _mon(d: dict) -> TrainerMon:
    moves = d["moves"]
    return TrainerMon(
        species=d["species"],
        level=d["level"],
        difficulty=d["difficulty"],
        moves=None if moves is None else tuple(moves),
        item=d["item"],
        form=d["form"],
    )


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
            partner=tuple(
                PartnerVariant(p["trainer"], p["starter"], p["player_gender"]) for p in t["partner"]
            ),
        )
        for t in rows
    ]


class IntermediateData:
    """Thin loader of the extractor's species, form and learnset JSON. ``species`` is indexed
    by species id, ``personal`` and ``learnsets`` by personal row (the species, then the
    data forms of ``forms.json``), and ``form_rows`` maps ``[species][form]`` to the row."""

    def __init__(self, dir: Path):
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


def build_trainer_party(trainer: Trainer, inter: IntermediateData) -> list[PokemonSpec]:
    party = []
    for mon in trainer.party:
        row = inter.row(mon.species, mon.form)
        species = inter.personal[row]
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
        moves = mon.moves
        if moves is None:
            moves = default_moves(inter.learnsets[row], mon.level)
        ivs = difficulty_to_ivs(mon.difficulty)
        party.append(
            PokemonSpec(
                species=mon.species,
                level=mon.level,
                nature=pid % 25,  # Pokemon_GetNatureOf
                ivs=(ivs,) * 6,
                evs=(0,) * 6,
                moves=moves,
                item=mon.item,
                ability_slot=ability_slot,
                gender=gender_of(species["gender_ratio"], pid),
                friendship=species["base_friendship"],
                form=mon.form,
            )
        )
    return party
