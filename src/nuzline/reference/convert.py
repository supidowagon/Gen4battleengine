"""Exact conversion between ``BattleState`` (JAX arrays) and ``RefState`` (Python)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.engine.format import format_of
from nuzline.engine.state import BattleState, layout
from nuzline.reference.state import (
    BATTLER_FIELDS,
    FIELD_FIELDS,
    MON_FIELDS,
    PARTY_FIELDS,
    SIDE_FIELDS,
    RefBattler,
    RefParty,
    RefPokemon,
    RefSideConds,
    RefState,
    format_of_ref,
)

if set(MON_FIELDS + PARTY_FIELDS + BATTLER_FIELDS + SIDE_FIELDS + FIELD_FIELDS) != set(
    BattleState._fields
):
    raise RuntimeError(
        "RefPokemon, RefParty, RefBattler, RefSideConds and RefState must hold every "
        "BattleState field"
    )


def to_ref(state: BattleState) -> RefState:
    fmt = format_of(state)
    a = {name: np.asarray(getattr(state, name)).tolist() for name in BattleState._fields}
    parties = [
        RefParty(
            mons=[
                RefPokemon(**{f: a[f][p][slot] for f in MON_FIELDS})
                for slot in range(len(a["species"][p]))
            ],
            **{f: a[f][p] for f in PARTY_FIELDS},
        )
        for p in range(fmt.P)
    ]
    battlers = [RefBattler(**{f: a[f][b] for f in BATTLER_FIELDS}) for b in range(fmt.B)]
    sides = [RefSideConds(**{f: a[f][side] for f in SIDE_FIELDS}) for side in (0, 1)]
    return RefState(
        parties=parties, battlers=battlers, sides=sides, **{f: a[f] for f in FIELD_FIELDS}
    )


def from_ref(ref: RefState) -> BattleState:
    """The ``BattleState`` with ``ref``'s values in each field's storage dtype; a value
    that does not fit raises ``OverflowError``."""
    shapes = layout(format_of_ref(ref))
    values: dict[str, object] = {}
    for f in MON_FIELDS:
        values[f] = [[getattr(mon, f) for mon in party.mons] for party in ref.parties]
    for f in PARTY_FIELDS:
        values[f] = [getattr(party, f) for party in ref.parties]
    for f in BATTLER_FIELDS:
        values[f] = [getattr(battler, f) for battler in ref.battlers]
    for f in SIDE_FIELDS:
        values[f] = [getattr(side, f) for side in ref.sides]
    for f in FIELD_FIELDS:
        values[f] = getattr(ref, f)
    out = {}
    for name in BattleState._fields:
        shape, dtype = shapes[name]
        arr = np.asarray(values[name], dtype=dtype)
        if arr.shape != shape:
            raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
        out[name] = jnp.asarray(arr)
    return BattleState(**out)
