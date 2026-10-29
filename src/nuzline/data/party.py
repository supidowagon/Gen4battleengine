"""Load the player's party from a TOML file.

Each ``[[pokemon]]`` table has ``species``, ``level``, ``nature``, ``ivs`` and
``evs`` (six values each, HP/Atk/Def/SpA/SpD/Spe), ``moves`` (up to four names),
``item`` (``""`` for none), ``ability`` (one of the species' abilities) and
``gender`` (``male``/``female``/``none``). Optional: ``hp`` (current HP; omitted
means full) and ``friendship`` (omitted means the species' base friendship).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names

GENDERS = {"male": 0, "female": 1, "none": 2}


def _six(entry: dict, key: str) -> tuple[int, int, int, int, int, int]:
    values = tuple(int(v) for v in entry[key])
    if len(values) != 6:
        raise ValueError(f"{entry['species']}: {key} needs 6 values, got {len(values)}")
    return values  # type: ignore[return-value]


def _spec(entry: dict, names: Names) -> PokemonSpec:
    species = names.species(entry["species"])
    moves = [names.move(m) for m in entry["moves"]]
    if not 1 <= len(moves) <= 4:
        raise ValueError(f"{entry['species']}: needs 1-4 moves, got {len(moves)}")
    ability = names.ability(entry["ability"])
    slots = names.species_abilities(species)
    if ability == 0 or ability not in slots:
        raise ValueError(f"{entry['species']} cannot have ability {entry['ability']!r}")
    natures = {n.lower(): i for i, n in enumerate(names.nature_names)}
    if entry["nature"].lower() not in natures:
        raise ValueError(f"unknown nature {entry['nature']!r}")
    if entry["gender"] not in GENDERS:
        raise ValueError(f"gender must be one of {sorted(GENDERS)}, got {entry['gender']!r}")
    return PokemonSpec(
        species=species,
        level=int(entry["level"]),
        nature=natures[entry["nature"].lower()],
        ivs=_six(entry, "ivs"),
        evs=_six(entry, "evs"),
        moves=tuple(moves + [0] * (4 - len(moves))),  # type: ignore[arg-type]
        item=names.item(entry["item"]),
        ability_slot=slots.index(ability),
        gender=GENDERS[entry["gender"]],
        friendship=int(entry.get("friendship", names.species_base_friendship(species))),
        current_hp=None if "hp" not in entry else int(entry["hp"]),
    )


def load_party(path: Path, names: Names) -> list[PokemonSpec]:
    with open(path, "rb") as f:
        entries = tomllib.load(f)["pokemon"]
    if not 1 <= len(entries) <= 6:
        raise ValueError(f"{path}: a party has 1-6 Pokemon, got {len(entries)}")
    return [_spec(e, names) for e in entries]
