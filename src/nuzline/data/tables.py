"""Load the packed Gen 4 Platinum tables (built by ``tools/build_tables.py``).

Every array is indexed by the decomp's own ids; encodings are documented in
``tools/build_tables.py``. ``Names`` resolves English names to those ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

DEFAULT = Path(__file__).resolve().parents[3] / "data" / "gen4_platinum"


class Tables(NamedTuple):
    species_base_stats: jnp.ndarray  # int16[S,6], index = ids.Stat
    species_types: jnp.ndarray  # int8[S,2]
    species_abilities: jnp.ndarray  # int16[S,2], ids.AbilityId
    species_weight: jnp.ndarray  # int32[S], hectograms
    species_gender: jnp.ndarray  # int16[S], decomp gender ratio
    species_friendship: jnp.ndarray  # int16[S] (140 does not fit int8)
    move_power: jnp.ndarray  # int16[M]
    move_type: jnp.ndarray  # int8[M]
    move_category: jnp.ndarray  # int8[M], 0 physical, 1 special, 2 status
    move_accuracy: jnp.ndarray  # int8[M], 0 = never misses
    move_pp: jnp.ndarray  # int8[M]
    move_priority: jnp.ndarray  # int8[M]
    move_target: jnp.ndarray  # int8[M], index into manifest target_names
    move_effect: jnp.ndarray  # int16[M], ids.EffectId
    move_effect_chance: jnp.ndarray  # int8[M]
    move_param: jnp.ndarray  # int16[M,3]
    move_flags: jnp.ndarray  # int16[M], ids.MoveFlag bits
    type_chart: jnp.ndarray  # int8[T,T], tenths
    type_chart_order: jnp.ndarray  # int16[T,T], decomp chart row index, -1 unlisted
    nature_mods: jnp.ndarray  # int8[25,6], -1/0/+1
    item_effect: jnp.ndarray  # int16[I], ids.ItemId
    item_param: jnp.ndarray  # int16[I,2]
    trainer_item_effect: jnp.ndarray  # int8[I], ids.TrainerItemId


def load_tables(dir: Path = DEFAULT) -> Tables:
    with np.load(Path(dir) / "tables.npz") as z:
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


class Names:
    """Case-insensitive English name -> decomp id lookups."""

    def __init__(self, manifest: dict):
        self.species_names: tuple[str, ...] = tuple(manifest["species_names"])
        self.move_names: tuple[str, ...] = tuple(manifest["move_names"])
        self.item_names: tuple[str, ...] = tuple(manifest["item_names"])
        self.ability_names: tuple[str, ...] = tuple(manifest["ability_names"])
        self.type_names: tuple[str, ...] = tuple(manifest["type_names"])
        self.nature_names: tuple[str, ...] = tuple(manifest["nature_names"])
        self._species_abilities = tuple(tuple(a) for a in manifest["species_abilities"])
        self._species_friendship = tuple(manifest["species_base_friendship"])
        self._species = _index(manifest["species_names"], "species")
        self._moves = _index(manifest["move_names"], "move")
        self._items = _index(manifest["item_names"], "item")
        self._abilities = _index(manifest["ability_names"], "ability")
        self._types = _index(manifest["type_names"], "type")

    @staticmethod
    def _get(index: dict[str, int], name: str, what: str) -> int:
        try:
            return index[name.lower()]
        except KeyError:
            raise KeyError(f"unknown {what} {name!r}") from None

    def species(self, name: str) -> int:
        return self._get(self._species, name, "species")

    def species_abilities(self, species: int) -> tuple[int, int]:
        """The species' two decomp ability ids; slot 1 is 0 (ABILITY_NONE) when absent."""
        return self._species_abilities[species]

    def species_base_friendship(self, species: int) -> int:
        return self._species_friendship[species]

    def move(self, name: str) -> int:
        """Move id; ``""`` is 0 (no move)."""
        return 0 if name == "" else self._get(self._moves, name, "move")

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
        return 0 if name == "" else self._get(self._items, name, "item")

    def ability(self, name: str) -> int:
        return self._get(self._abilities, name, "ability")

    def type(self, name: str) -> int:
        """Decomp type id, e.g. ``type("Fire")``; the Mystery type is ``"???"``."""
        return self._get(self._types, name, "type")


def load_names(dir: Path = DEFAULT) -> Names:
    return Names(json.loads((Path(dir) / "manifest.json").read_text(encoding="utf-8")))
