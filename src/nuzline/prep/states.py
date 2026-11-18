"""Batched turn-0 states for the prep search (plan 4c Task 2).

``make_states`` builds the numpy arrays of every team on the host (``initial_arrays``,
with ``make_state``'s validation), then runs the leads' switch-in checks (``hooks.battle_start``) for a whole
chunk in one exported program (``"make_states"``): the host loop of ``make_state`` calls
costs a dispatch per team, the program one per ``STATE_CHUNK`` teams. Chunks have one
padded shape (copies of row 0, dropped from the output) and are sharded over the host
devices as ``search.leaves`` does."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState for jax.export
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.data.trainer import TERRAIN_UNKNOWN
from nuzline.engine.hooks import battle_start
from nuzline.engine.ids import Weather
from nuzline.engine.state import LAYOUT, BattleState, initial_arrays, protected_mask
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.rows import concat, take

__all__ = ["STATE_CHUNK", "make_states"]

STATE_CHUNK = 512  # teams per program call, before padding to the device count


def _switch_in(tables: Tables, states: BattleState) -> BattleState:
    return jax.vmap(lambda state: battle_start(tables, state))(states)


def make_states(
    tables: Tables,
    teams: Sequence[Sequence[PokemonSpec]],
    opp: Sequence[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    budget: int = 0,
    protected: Sequence[Sequence[int]] | None = None,
    terrain: int = TERRAIN_UNKNOWN,
    gravity: int = 0,
) -> BattleState:
    """The turn-0 states of ``teams`` (each a player party, lead in slot 0) against
    ``opp``, as a numpy batch with leading axis ``len(teams)``. Row ``i`` equals
    ``make_state(tables, list(teams[i]), list(opp), start_condition, opp_items,
    leads=(0, 0), terrain=terrain, gravity=gravity).with_budget(budget, protected[i])`` byte
    for byte.
    ``ValueError`` as ``make_state`` / ``with_budget`` for a bad party, a bad budget or a protected slot
    that is empty or fainted."""
    n = len(teams)
    if protected is not None and len(protected) != n:
        raise ValueError(f"protected has {len(protected)} rows for {n} teams")
    rows = []
    for i, team in enumerate(teams):
        a = initial_arrays(
            tables,
            list(team),
            list(opp),
            start_condition,
            opp_items,
            terrain=terrain,
            gravity=gravity,
        )
        a["faint_budget"][...] = budget
        a["protected"][...] = protected_mask(
            a["species"][0], a["hp"][0], budget, () if protected is None else protected[i]
        )
        rows.append(BattleState(**{name: a[name] for name in BattleState._fields}))
    if n == 0:
        return BattleState(
            **{name: np.zeros((0, *shape), dtype) for name, (shape, dtype) in LAYOUT.items()}
        )
    stacked = jax.tree.map(lambda *fs: np.stack(fs), *rows)
    devices = jax.device_count()
    size = -(-STATE_CHUNK // devices) * devices
    sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
    outs = []
    for start in range(0, n, STATE_CHUNK):
        idx = np.arange(start, min(start + STATE_CHUNK, n))
        pad = np.concatenate([idx, np.full(size - len(idx), idx[0])])
        chunk = take(stacked, pad)
        chunk = jax.device_put(chunk, sharding) if devices > 1 else jax.device_put(chunk)
        args = (tables, chunk)
        program = cached_export(
            _switch_in, args, name="make_states", static={}, version=code_version()
        )
        outs.append(take(jax.device_get(program(*args)), slice(0, len(idx))))
    return concat(outs)
