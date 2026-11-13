"""Load the player's party from a TOML file.

Each ``[[pokemon]]`` table has ``species``, ``level``, ``nature``, ``ivs`` and
``evs`` (six values each, HP/Atk/Def/SpA/SpD/Spe), ``moves`` (up to four names),
``item`` (``""`` for none), ``ability`` (one of the species' abilities) and
``gender`` (``male``/``female``/``none``). Optional: ``hp`` (current HP; omitted
means full), ``form`` (a form name from ``Names.form_names``, case-insensitive, such as
``"Wash"`` for Rotom, or a form number; omitted means 0, the base form, whose name is the
species'), ``friendship`` (omitted means the form's base friendship), ``status``
(``none``/``sleep``/``poison``/``burn``/``freeze``/``paralysis``/``toxic``, default
``none``), ``sleep_turns`` (sleep turns left;
needs ``status = "sleep"``), ``pp`` (current PP, one value per listed move, padded with 0
to four slots; omitted means the table PP) and ``stats`` (six stored stats, max HP first,
in the ``ivs`` order; omitted means computed from level, IVs, EVs and nature).

Experience (``PokemonSpec``): ``exp`` (total experience, of the Pokemon's ``level``;
omitted means the base of the level), ``exp_boost`` (100, 150 or 170: the exp multiplier in
percent; omitted means 100, the user's own Pokemon), ``pokerus`` and ``friend_bonus`` (0 or 1;
omitted means 0). Coverage inputs (``PokemonSpec``): ``egg_location`` (a location id, omitted
means 0) and ``pp_ups`` (four values 0-3, one per move slot including empty ones; omitted means
none).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.engine.ids import Status

GENDERS = {"male": 0, "female": 1, "none": 2}


def _six(entry: dict, key: str) -> tuple[int, int, int, int, int, int]:
    values = tuple(int(v) for v in entry[key])
    if len(values) != 6:
        raise ValueError(f"{entry['species']}: {key} needs 6 values, got {len(values)}")
    if key == "stats" and any(not 1 <= v <= 999 for v in values):
        raise ValueError(f"{entry['species']}: stats must be 1-999, got {list(values)}")
    return values  # type: ignore[return-value]


EXP_BOOSTS = (100, 150, 170)  # BattleScript_GetExpTask, battle_script.c:9757-9763


def _exp_fields(entry: dict) -> dict:
    """``exp``, ``exp_boost``, ``pokerus``, ``friend_bonus``, ``egg_location`` and ``pp_ups``
    (module docstring)."""
    out: dict = {}
    if "exp" in entry:
        out["exp"] = int(entry["exp"])
        if out["exp"] < 0:
            raise ValueError(f"{entry['species']}: exp must be 0 or more, got {out['exp']}")
    out["exp_boost"] = int(entry.get("exp_boost", 100))
    if out["exp_boost"] not in EXP_BOOSTS:
        raise ValueError(
            f"{entry['species']}: exp_boost must be one of {list(EXP_BOOSTS)}, "
            f"got {out['exp_boost']}"
        )
    for key in ("pokerus", "friend_bonus"):
        out[key] = int(entry.get(key, 0))
        if out[key] not in (0, 1):
            raise ValueError(f"{entry['species']}: {key} must be 0 or 1, got {out[key]}")
    out["egg_location"] = int(entry.get("egg_location", 0))
    if not 0 <= out["egg_location"] <= 0xFFFF:
        raise ValueError(f"{entry['species']}: egg_location must be 0-65535")
    ups = tuple(int(v) for v in entry.get("pp_ups", (0, 0, 0, 0)))
    if len(ups) != 4 or any(not 0 <= v <= 3 for v in ups):
        raise ValueError(f"{entry['species']}: pp_ups needs 4 values 0-3, got {list(ups)}")
    out["pp_ups"] = ups
    return out


def _status(entry: dict) -> tuple[int, int]:
    name = str(entry.get("status", "none")).lower()
    if name.upper() not in Status.__members__:
        raise ValueError(f"{entry['species']}: unknown status {entry['status']!r}")
    status = Status[name.upper()]
    if "sleep_turns" in entry and status != Status.SLEEP:
        raise ValueError(f'{entry["species"]}: sleep_turns needs status = "sleep"')
    turns = int(entry.get("sleep_turns", 0))
    if "sleep_turns" in entry and not 1 <= turns <= 7:
        raise ValueError(f"{entry['species']}: sleep_turns must be 1-7, got {turns}")
    return int(status), turns


def _form(entry: dict, names: Names, species: int) -> int:
    """The ``form`` key's form number; ``ValueError`` naming the species as written and its
    valid forms."""
    if "form" not in entry:
        return 0
    value = entry["form"]
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        # A bad party file is a ValueError throughout this loader, whatever the cause.
        raise ValueError(  # noqa: TRY004
            f"{entry['species']}: form must be a name or a number, got {value!r}"
        )
    try:
        return names.form(species, value)
    except ValueError:
        pass
    try:
        names.form_row(species, 1)
    except ValueError:
        raise ValueError(f"{entry['species']} has no forms, got form {value!r}") from None
    valid = ", ".join(names.form_names(species))
    if len(names.form_names(species)) == 1:  # cosmetic forms only (Unown, Burmy, ...)
        valid += ", or a form number"
    raise ValueError(f"{entry['species']} has no form {value!r} (forms: {valid})")


def _spec(entry: dict, names: Names) -> PokemonSpec:
    species = names.species(entry["species"])
    form = _form(entry, names, species)
    moves = [names.move(m) for m in entry["moves"]]
    if not 1 <= len(moves) <= 4:
        raise ValueError(f"{entry['species']}: needs 1-4 moves, got {len(moves)}")
    ability = names.ability(entry["ability"])
    slots = names.species_abilities(species, form)
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
        if any(not 0 <= v <= 64 for v in values):
            raise ValueError(f"{entry['species']}: pp values must be 0-64, got {values}")
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
        friendship=int(entry.get("friendship", names.species_base_friendship(species, form))),
        form=form,
        current_hp=None if "hp" not in entry else int(entry["hp"]),
        status=status,
        status_counter=status_counter,
        pp=pp,  # type: ignore[arg-type]
        stats=_six(entry, "stats") if "stats" in entry else None,
        **_exp_fields(entry),
    )


def load_party(path: Path, names: Names, tables: Tables | None = None) -> list[PokemonSpec]:
    """The party in ``path``. With ``tables``, an ``exp`` that is not of the Pokemon's
    ``level`` is refused (``ValueError``, ``engine.state.spec_exp``); without them
    ``make_state`` refuses it."""
    with open(path, "rb") as f:
        entries = tomllib.load(f)["pokemon"]
    if not 1 <= len(entries) <= 6:
        raise ValueError(f"{path}: a party has 1-6 Pokemon, got {len(entries)}")
    specs = [_spec(e, names) for e in entries]
    if tables is not None:
        from nuzline.engine.state import spec_exp  # the engine imports jax

        for entry, spec in zip(entries, specs, strict=True):
            try:
                spec_exp(tables, spec)
            except ValueError as e:
                raise ValueError(f"{entry['species']}: {e}") from None
    return specs
