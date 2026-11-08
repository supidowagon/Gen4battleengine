"""Load the packed Gen 4 Platinum tables (built by ``tools/build_tables.py``).

Every array is indexed by the decomp's own ids; encodings are documented in
``tools/build_tables.py``. ``Names`` resolves English names to those ids. ``species_*``
arrays are indexed by personal row: species ids 0-495, then the data forms 496-507
(``nuzline.data.forms.form_row`` maps a species and form to its row).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from nuzline.data.forms import form_row
from nuzline.data.games import manifest_game

DEFAULT = Path(__file__).resolve().parents[3] / "data" / "gen4_platinum"

# The version ``tools/build_tables.py`` writes into ``manifest.json``. Bump it with any change
# to the packed arrays or manifest keys; every loader refuses tables of another version.
MANIFEST_VERSION = 15


def check_manifest(manifest: dict, dir: Path) -> None:
    """Raise ``ValueError`` unless ``manifest`` is table version ``MANIFEST_VERSION`` (a
    missing ``version`` counts as stale), so old tables give one rebuild message instead of
    a ``KeyError`` on whichever new array or key is read first."""
    v = manifest.get("version")
    if v != MANIFEST_VERSION:
        raise ValueError(
            f"{Path(dir) / 'manifest.json'} is version {v}; this nuzline needs "
            f"{MANIFEST_VERSION}: re-run tools/extract_decomp.py and tools/build_tables.py"
        )


def check_npz(z, dir: Path) -> None:
    """Raise ``ValueError`` unless the opened ``tables.npz`` ``z`` carries ``version``
    ``MANIFEST_VERSION``. ``manifest.json`` is tracked in git and ``tables.npz`` is not, so
    after a pull a current manifest can sit beside an old npz; only the npz's own
    ``version`` array (absent before version 8, read as stale) proves its version."""
    v = int(z["version"]) if "version" in z.files else None
    if v != MANIFEST_VERSION:
        raise ValueError(
            f"{Path(dir) / 'tables.npz'} is version {v}; this nuzline needs "
            f"{MANIFEST_VERSION}: re-run tools/extract_decomp.py and tools/build_tables.py"
        )


def _read_manifest(dir: Path) -> dict:
    manifest = json.loads((Path(dir) / "manifest.json").read_text(encoding="utf-8"))
    check_manifest(manifest, dir)
    return manifest


class Tables(NamedTuple):
    # S = 508 personal rows: species 0-495, then the data forms (Ruling 6).
    species_base_stats: jnp.ndarray  # int16[S,6], index = ids.Stat
    species_types: jnp.ndarray  # int8[S,2]
    species_abilities: jnp.ndarray  # int16[S,2], ids.AbilityId
    species_abilities_decomp: jnp.ndarray  # int16[S,2], decomp ability ids
    species_weight: jnp.ndarray  # int32[S], hectograms
    species_gender: jnp.ndarray  # int16[S], decomp gender ratio
    species_friendship: jnp.ndarray  # int16[S] (140 does not fit int8)
    species_exp_rate: jnp.ndarray  # int8[S], 0 medium fast .. 5 slow (exp_rates.txt order)
    species_exp_yield: jnp.ndarray  # int16[S], SpeciesData.baseExpReward
    species_form_row: jnp.ndarray  # int16[496,28], [species, form] -> personal row, -1 none
    exp_table: jnp.ndarray  # int32[8,101], [rate, level] -> experience
    move_power: jnp.ndarray  # int16[M]
    move_type: jnp.ndarray  # int8[M]
    move_category: jnp.ndarray  # int8[M], 0 physical, 1 special, 2 status
    move_accuracy: jnp.ndarray  # int8[M], 0 = never misses
    move_pp: jnp.ndarray  # int8[M]
    move_priority: jnp.ndarray  # int8[M]
    move_target: jnp.ndarray  # int8[M], index into manifest target_names
    move_effect: jnp.ndarray  # int16[M], ids.EffectId
    move_effect_chance: jnp.ndarray  # int8[M]; a chance-0 non-certain secondary packs as PLAIN
    move_ai_effect: jnp.ndarray  # int16[M], the decomp battle effect id the AI scripts read
    move_param: jnp.ndarray  # int16[M,3]
    move_flags: jnp.ndarray  # int16[M], ids.MoveFlag bits
    type_chart: jnp.ndarray  # int8[T,T], tenths
    type_chart_order: jnp.ndarray  # int16[T,T], decomp chart row index, -1 unlisted
    type_chart_foresight: jnp.ndarray  # int8[T,T], 1 = row after the 0xFE separator
    nature_mods: jnp.ndarray  # int8[25,6], -1/0/+1
    item_effect: jnp.ndarray  # int16[I], ids.ItemId
    item_param: jnp.ndarray  # int16[I,2]
    trainer_item_effect: jnp.ndarray  # int8[I], ids.TrainerItemId
    item_hold_effect: jnp.ndarray  # int16[I], the decomp hold effect id (0 = none)


def load_tables(dir: Path = DEFAULT) -> Tables:
    _read_manifest(dir)
    with np.load(Path(dir) / "tables.npz") as z:
        check_npz(z, dir)
        return Tables(**{f: jnp.asarray(z[f]) for f in Tables._fields})


def to_numpy(tables: Tables) -> Tables:
    """The same ``Tables`` with every field a numpy array (for the plain-Python
    reference engine, ``nuzline.reference``)."""
    return Tables(*(np.asarray(field) for field in tables))


def _index(names: list[str], what: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        key = name.lower()
        if key in out:
            raise ValueError(f"duplicate {what} name {name!r}")
        out[key] = i
    return out


@dataclass(frozen=True)
class RomCondition:
    """A ``not_reached`` ROM change (``data/kaizo/rom_diff.toml``) as ids: it can only matter in
    a fight where one of these holds (a format or area condition, a move, ability or held item
    on a side, a trainer, or any trainer item)."""

    label: str
    formats: tuple[str, ...]
    areas: tuple[str, ...]
    moves: tuple[int, ...]
    abilities: tuple[int, ...]
    items: tuple[int, ...]
    trainers: tuple[int, ...]
    trainer_items: bool


class Names:
    """Case-insensitive English name -> decomp id lookups.

    A game's manifest may carry ``aliases`` (``{kind: {alias: id}}``: extra names that
    resolve to an id) and ``renamed`` (``{kind: {old lowercase vanilla name: new name}}``:
    slots the game renamed, whose old name is refused with a pointer to the new one). Both
    are absent in vanilla, which means empty.
    """

    def __init__(self, manifest: dict):
        self.game: str = manifest_game(manifest)
        self.unclassified_rom_changes: int = int(manifest.get("unclassified_rom_changes", 0))
        self.rom_change_conditions: tuple[RomCondition, ...] = tuple(
            RomCondition(
                c["label"],
                tuple(c["formats"]),
                tuple(c["areas"]),
                tuple(c["moves"]),
                tuple(c["abilities"]),
                tuple(c["items"]),
                tuple(c["trainers"]),
                bool(c["trainer_items"]),
            )
            for c in manifest.get("rom_change_conditions", ())
        )
        self.species_names: tuple[str, ...] = tuple(manifest["species_names"])
        self.move_names: tuple[str, ...] = tuple(manifest["move_names"])
        self.item_names: tuple[str, ...] = tuple(manifest["item_names"])
        self.ability_names: tuple[str, ...] = tuple(manifest["ability_names"])
        self.type_names: tuple[str, ...] = tuple(manifest["type_names"])
        self.nature_names: tuple[str, ...] = tuple(manifest["nature_names"])
        # Text code -> character (JSON keys are hex strings).
        self.charmap: dict[int, str] = {int(k, 16): v for k, v in manifest["charmap"].items()}
        # Per personal row (species, then data forms).
        self._species_abilities = tuple(tuple(a) for a in manifest["species_abilities"])
        self._species_friendship = tuple(manifest["species_base_friendship"])
        self._form_rows = manifest["species_form_row"]
        self._form_names = {int(s): tuple(names) for s, names in manifest["form_names"].items()}
        self._species = _index(manifest["species_names"], "species")
        self._moves = _index(manifest["move_names"], "move")
        self._items = _index(manifest["item_names"], "item")
        self._abilities = _index(manifest["ability_names"], "ability")
        self._types = _index(manifest["type_names"], "type")
        indexes = {
            "species": self._species,
            "move": self._moves,
            "item": self._items,
            "ability": self._abilities,
        }
        self._renamed: dict[str, dict[str, str]] = {
            kind: {
                old.lower(): new for old, new in manifest.get("renamed", {}).get(kind, {}).items()
            }
            for kind in indexes
        }
        for kind, index in indexes.items():
            for alias, i in manifest.get("aliases", {}).get(kind, {}).items():
                key = alias.lower()
                if key in index:
                    raise ValueError(
                        f"{kind} alias {alias!r} collides with the name of id {index[key]}"
                        f" (alias id {i})"
                    )
                new = self._renamed[kind].get(key)
                if new is not None:
                    raise ValueError(
                        f"{kind} alias {alias!r} collides with the renamed name of id "
                        f"{index.get(new.lower())} (now {new!r}; alias id {i})"
                    )
                index[key] = int(i)

    def _lookup(self, index: dict[str, int], name: str, what: str) -> int:
        key = name.lower()
        try:
            return index[key]
        except KeyError:
            new = self._renamed.get(what, {}).get(key)
            if new is not None:
                raise KeyError(
                    f"unknown {what} {name!r}: in this game that slot is {new!r}"
                ) from None
            raise KeyError(f"unknown {what} {name!r}") from None

    @staticmethod
    def _get(index: dict[str, int], name: str, what: str) -> int:
        try:
            return index[name.lower()]
        except KeyError:
            raise KeyError(f"unknown {what} {name!r}") from None

    def species(self, name: str) -> int:
        return self._lookup(self._species, name, "species")

    def form_row(self, species: int, form: int = 0) -> int:
        """The personal row of ``species``' ``form``; ``ValueError`` if it has no such form."""
        return form_row(self._form_rows, species, form)

    def _species_id(self, species: int) -> int:
        """``species`` as an int; ``ValueError`` outside the species ids (a negative id
        would otherwise wrap round the name tuple)."""
        species = int(species)
        if not 0 <= species < len(self.species_names):
            raise ValueError(f"species {species} out of range")
        return species

    def form_names(self, species: int) -> tuple[str, ...]:
        """``(species name, form 1 name, ...)`` for a species with data forms, else
        ``(species name,)``. ``ValueError`` for a species id out of range."""
        species = self._species_id(species)
        return self._form_names.get(species, (self.species_names[species],))

    def form(self, species: int, name: str | int) -> int:
        """The form number of ``species`` named ``name`` (case-insensitive, from
        ``form_names``), or ``name`` itself when it is a form number the species has.
        Raises ``ValueError`` naming the valid forms (or the species id out of range)."""
        species = self._species_id(species)
        valid = self.form_names(species)
        if not isinstance(name, str):
            try:
                self.form_row(species, name)
            except ValueError:
                raise ValueError(
                    f"{self.species_names[species]} has no form {name!r} "
                    f"(forms: {', '.join(valid)})"
                ) from None
            return int(name)
        for i, n in enumerate(valid):
            if n.lower() == name.lower():
                return i
        raise ValueError(
            f"{self.species_names[species]} has no form {name!r} (forms: {', '.join(valid)})"
        )

    def species_abilities(self, species: int, form: int = 0) -> tuple[int, int]:
        """The form's two decomp ability ids; slot 1 is 0 (ABILITY_NONE) when absent."""
        return self._species_abilities[self.form_row(species, form)]

    def species_base_friendship(self, species: int, form: int = 0) -> int:
        return self._species_friendship[self.form_row(species, form)]

    def move(self, name: str) -> int:
        """Move id; ``""`` is 0 (no move)."""
        return 0 if name == "" else self._lookup(self._moves, name, "move")

    def move_slot(self, state, side: int, name: str) -> int:
        """Move slot (0-3) of ``name`` on ``side``'s active Pokemon in a ``BattleState``;
        raises ``ValueError`` when it does not know the move."""
        move = self.move(name)
        known = np.asarray(state.moves[side, int(state.active[side])]).tolist()
        if move == 0 or move not in known:
            raise ValueError(f"side {side}'s active Pokemon does not know {name!r}")
        return known.index(move)

    def item(self, name: str) -> int:
        """Item id; ``""`` is 0 (no item)."""
        return 0 if name == "" else self._lookup(self._items, name, "item")

    def ability(self, name: str) -> int:
        return self._lookup(self._abilities, name, "ability")

    def type(self, name: str) -> int:
        """Decomp type id, e.g. ``type("Fire")``; the Mystery type is ``"???"``."""
        return self._get(self._types, name, "type")


def load_names(dir: Path = DEFAULT) -> Names:
    return Names(_read_manifest(dir))
