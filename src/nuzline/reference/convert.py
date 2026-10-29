"""Exact conversion between ``BattleState`` (JAX arrays) and ``RefState`` (Python)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.engine.state import LAYOUT, BattleState
from nuzline.reference.state import (
    FIELD_FIELDS,
    MON_FIELDS,
    SIDE_FIELDS,
    RefPokemon,
    RefSide,
    RefState,
)

if set(MON_FIELDS + SIDE_FIELDS + FIELD_FIELDS) != set(BattleState._fields):
    raise RuntimeError("RefPokemon, RefSide and RefState must hold every BattleState field")


def to_ref(state: BattleState) -> RefState:
    a = {name: np.asarray(getattr(state, name)).tolist() for name in BattleState._fields}
    sides = [
        RefSide(
            party=[
                RefPokemon(**{f: a[f][side][slot] for f in MON_FIELDS})
                for slot in range(len(a["species"][side]))
            ],
            **{f: a[f][side] for f in SIDE_FIELDS},
        )
        for side in (0, 1)
    ]
    return RefState(sides=sides, **{f: a[f] for f in FIELD_FIELDS})


def from_ref(ref: RefState) -> BattleState:
    """The ``BattleState`` with ``ref``'s values in each field's storage dtype; a value
    that does not fit raises ``OverflowError``."""
    values: dict[str, object] = {}
    for f in MON_FIELDS:
        values[f] = [[getattr(mon, f) for mon in side.party] for side in ref.sides]
    for f in SIDE_FIELDS:
        values[f] = [getattr(side, f) for side in ref.sides]
    for f in FIELD_FIELDS:
        values[f] = getattr(ref, f)
    out = {}
    for name in BattleState._fields:
        shape, dtype = LAYOUT[name]
        arr = np.asarray(values[name], dtype=dtype)
        if arr.shape != shape:
            raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
        out[name] = jnp.asarray(arr)
    return BattleState(**out)
