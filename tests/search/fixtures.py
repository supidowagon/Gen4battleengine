"""Shared Stage B test states (plan "Shared test fixtures"): constructed fights whose
chance nodes, observations and endgames the solver tests need. Each builds on
``tests.engine.helpers.roark_state``."""

from __future__ import annotations

from nuzline.data.tables import Names, Tables
from nuzline.engine.ids import Stat
from nuzline.engine.state import BattleState
from tests.engine.helpers import roark_state

_SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe


def shinx_vs_cranidos(tables: Tables, names: Names, slower_opp: bool = False) -> BattleState:
    """Shinx (Tackle, Leer, Spark) against Cranidos (the fixture moves, Tackle first):
    Spark connects on a pure Rock type with its 30 % paralysis. ``slower_opp`` sets
    Cranidos's Spe stat one below Shinx's, so Spark lands before Tackle and its
    paralysis can stop Tackle the same turn."""
    s = roark_state(tables, names, player_species="Shinx", opp_species="Cranidos")
    if slower_opp:
        s = s._replace(stats=s.stats.at[1, 0, _SPE].set(s.stats[0, 0, _SPE] - 1))
    return s


def spike_cannon_root(tables: Tables, names: Names) -> BattleState:
    """Spike Cannon (multi-hit, 2-5 hits) on both leads: one action pair's outcome
    product exceeds ``enumerate.WIDTH_CAP``."""
    return roark_state(tables, names, player_moves=["Spike Cannon"], opp_moves=["Spike Cannon"])


def one_v_one(tables: Tables, names: Names, player_move: str) -> BattleState:
    """Monferno with ``player_move`` in slot 0 against Geodude (fixture moves, Tackle
    first), one Pokemon a side."""
    return roark_state(tables, names, player_count=1, opp_count=1, player_moves=[player_move])


def spore_1v1(tables: Tables, names: Names) -> BattleState:
    """``one_v_one`` with Spore (sleep, ``Random 3, 2``: 2-5 turns)."""
    return one_v_one(tables, names, "Spore")


def confuse_ray_1v1(tables: Tables, names: Names) -> BattleState:
    """``one_v_one`` with Confuse Ray (confusion, 2-5 turns)."""
    return one_v_one(tables, names, "Confuse Ray")


def taunt_1v1(tables: Tables, names: Names) -> BattleState:
    """``one_v_one`` with Taunt (3-5 turns)."""
    return one_v_one(tables, names, "Taunt")


def bind_1v1(tables: Tables, names: Names) -> BattleState:
    """``one_v_one`` with Bind (``TRAP``, 75 % accuracy, 3-6 turns)."""
    return one_v_one(tables, names, "Bind")
