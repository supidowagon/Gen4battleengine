"""Load the player's party from a TOML file.

Each ``[[pokemon]]`` table has ``species``, ``level``, ``nature``, ``ivs`` and
``evs`` (six values each, HP/Atk/Def/SpA/SpD/Spe), ``moves`` (up to four names),
``item`` (``""`` for none), ``ability`` (one of the species' abilities) and
``gender`` (``male``/``female``/``none``). Optional: ``hp`` (current HP; omitted
means full), ``friendship`` (omitted means the species' base friendship), ``status``
(``none``/``sleep``/``poison``/``burn``/``freeze``/``paralysis``/``toxic``, default
``none``), ``sleep_turns`` (sleep turns left;
needs ``status = "sleep"``), ``pp`` (current PP, one value per listed move, padded with 0
to four slots; omitted means the table PP) and ``stats`` (six stored stats, max HP first,
in the ``ivs`` order; omitted means computed from level, IVs, EVs and nature).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names
from nuzline.engine.ids import Status

GENDERS = {"male": 0, "female": 1, "none": 2}


def _six(entry: dict, key: str) -> tuple[int, int, int, int, int, int]:
    values = tuple(int(v) for v in entry[key])
    if len(values) != 6:
        raise ValueError(f"{entry['species']}: {key} needs 6 values, got {len(values)}")
    return values  # type: ignore[return-value]


def _status(entry: dict) -> tuple[int, int]:
    name = str(entry.get("status", "none")).lower()
    if name.upper() not in Status.__members__:
        raise ValueError(f"{entry['species']}: unknown status {entry['status']!r}")
    status = Status[name.upper()]
    if "sleep_turns" in entry and status != Status.SLEEP:
        raise ValueError(f'{entry["species"]}: sleep_turns needs status = "sleep"')
    return int(status), int(entry.get("sleep_turns", 0))


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
    status, status_counter = _status(entry)
    pp = None
    if "pp" in entry:
        values = [int(v) for v in entry["pp"]]
        if len(values) != len(moves):
            raise ValueError(
                f"{entry['species']}: pp needs {len(moves)} values (one per move), "
                f"got {len(values)}"
            )
        pp = tuple(values + [0] * (4 - len(values)))
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
        status=status,
        status_counter=status_counter,
        pp=pp,  # type: ignore[arg-type]
        stats=_six(entry, "stats") if "stats" in entry else None,
    )


def load_party(path: Path, names: Names) -> list[PokemonSpec]:
    with open(path, "rb") as f:
        entries = tomllib.load(f)["pokemon"]
    if not 1 <= len(entries) <= 6:
        raise ValueError(f"{path}: a party has 1-6 Pokemon, got {len(entries)}")
    return [_spec(e, names) for e in entries]
