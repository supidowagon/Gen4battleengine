"""Shared Stage B test states (plan "Shared test fixtures"): constructed fights whose
chance nodes, observations and endgames the solver tests need. Each builds on
``tests.engine.helpers.roark_state``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.ai.policy import ai_action_distribution
from nuzline.data.tables import Names, Tables
from nuzline.engine.actions import MOVE_ACTIONS, legal_actions
from nuzline.engine.ids import Stat
from nuzline.engine.state import BattleState
from tests.ai.fixtures import trace_f
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
    product exceeds ``enumerate.WIDTH_CAP``. For enumerate/expand tests only: the
    opponent's Spike Cannon has no transcribed AI script, so the AI refuses this root
    and ruling 4 refuses a solve of it (``UnsupportedAi``);
    test_stage_b.test_dropped_root_action gives Spike Cannon to the player only."""
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


def switch_refusal_root(tables: Tables, names: Names) -> BattleState:
    """``tests.ai.fixtures.trace_f`` (Cranidos 40/43, ``move_hit`` = Wing Attack, vs
    Starly) with Geodude (party slot 0) at 1 HP: the AI switches to Geodude with mass
    > 0 and Starly's Tackle KOs it on entry, so the replacement in a turn the AI
    switched is refused (``playout.send_replacements``; roark.md 14 item 3)."""
    return trace_f(tables, names).replace_hp(1, 0, 1)


def empty_bag(state: BattleState) -> BattleState:
    """Roark's bag emptied: the AI has no item to use."""
    return state._replace(
        trainer_items=state.trainer_items.at[1].set(0),
        trainer_item_count=state.trainer_item_count.at[1].set(0),
    )


def endgame(
    tables: Tables,
    names: Names,
    player_moves: list[str],
    opp_moves: list[str],
    player_hp: int | None = None,
    opp_hp: int | None = None,
) -> BattleState:
    """Monferno against Geodude, one Pokemon a side, Roark's bag emptied, each side
    knowing only ``player_moves``/``opp_moves`` (the other slots' PP zeroed), at the
    given HP."""
    s = roark_state(
        tables, names, player_count=1, opp_count=1, player_moves=player_moves, opp_moves=opp_moves
    )
    s = empty_bag(s)
    s = s.with_pp(0, 0, [int(p) if i < len(player_moves) else 0 for i, p in enumerate(s.pp[0, 0])])
    s = s.with_pp(1, 0, [int(p) if i < len(opp_moves) else 0 for i, p in enumerate(s.pp[1, 0])])
    if player_hp is not None:
        s = s.replace_hp(0, 0, player_hp)
    if opp_hp is not None:
        s = s.replace_hp(1, 0, opp_hp)
    return s


def all_terminal_root(tables: Tables, names: Names) -> BattleState:
    """Monferno (Scratch, Ember, Mach Punch, Quick Attack, all accuracy 100) against
    Geodude at 1 HP, one Pokemon a side, Roark's bag emptied: every player move lands
    first and KOs, so every child of the root is a win."""
    s = roark_state(
        tables,
        names,
        player_count=1,
        opp_count=1,
        player_moves=["Scratch", "Ember", "Mach Punch", "Quick Attack"],
    )
    s = empty_bag(s).replace_hp(1, 0, 1)
    q = np.asarray(ai_action_distribution(tables, s))
    assert q[MOVE_ACTIONS:].sum() == 0, "the AI row has item or switch mass"
    legal = np.flatnonzero(np.asarray(legal_actions(tables, s, 0)))
    assert legal.tolist() == list(range(MOVE_ACTIONS)), legal
    return s


def pixel_endgame(tables: Tables, names: Names) -> BattleState:
    """Monferno (Water Gun: sure, 60-72 damage without a crit; Hydro Pump: 80 %
    accuracy, 168-196) against a Leer-only Geodude whose ``max_hp`` is 999 (an HP-bar
    pixel is ~20.8 HP; observe.hp_pixels), at 242 HP. A Hydro Pump hit on turn 1 leaves
    46-74 HP; 46-62 share pixel 2, so one observation holds states where Water Gun KOs
    surely (HP <= 60, Water Gun best on turn 2) and the 62 HP state where Hydro Pump's
    80 % beats Water Gun's ~77 %: peeking at the exact HP would choose per state. The
    fight ends within 2 turns or counts 0 at a 2-turn cap (Geodude cannot hurt
    Monferno)."""
    from tests.search.exhaustive import children  # the fixture's own check

    s = endgame(tables, names, ["Water Gun", "Hydro Pump"], ["Leer"])
    s = s._replace(max_hp=s.max_hp.at[1, 0].set(999)).replace_hp(1, 0, 242)
    assert int(s.max_hp[1, 0]) > 48
    hydro = names.move_slot(s, 0, "Hydro Pump")
    seen: dict[bytes, set[int]] = {}
    for after, refused, obs, _ in children(tables, s, hydro, 0):
        if not refused and int(after.over) == 0:
            seen.setdefault(obs, set()).add(int(after.hp[1, 0]))
    assert any(len(hps) >= 2 for hps in seen.values()), (
        "no two turn-1 children share an observation"
    )
    return s


def spore_endgame(tables: Tables, names: Names) -> BattleState:
    """Monferno at 1 HP (Spore; Hydro Pump, 80 %) against a Tackle-only Geodude at 20
    HP: Spore's sleep (2-5 turns, the counter hidden) puts four states in one
    observation; a Hydro Pump hit KOs, a Tackle hit KOs Monferno."""
    return endgame(tables, names, ["Spore", "Hydro Pump"], ["Tackle"], player_hp=1, opp_hp=20)


def low_hp_endgame(tables: Tables, names: Names) -> BattleState:
    """Monferno at 10 HP (Scratch; Mega Kick, 75 %) against Geodude at 6 HP (Tackle,
    Rock Throw): a plain low-HP 1v1 decided by rolls, crits and accuracy."""
    return endgame(
        tables,
        names,
        ["Scratch", "Mega Kick"],
        ["Tackle", "Rock Throw"],
        player_hp=10,
        opp_hp=6,
    )


def at_replacement(state: BattleState, opp_down: bool = False) -> BattleState:
    """``state`` at a Set-mode player replacement: the player's active Pokemon at 0 HP
    (``deaths`` and ``player_faints`` updated), the battle not over; with ``opp_down`` the
    opponent's active at 0 HP too. Built by hand: until the terminal rule goes a step
    never leaves the player's side here."""
    active = int(state.active[0])
    s = state.replace_hp(0, active, 0)
    s = s._replace(
        deaths=s.deaths.at[active].set(1),
        player_faints=(s.player_faints + 1).astype(s.player_faints.dtype),
        over=jnp.zeros_like(s.over),
    )
    if opp_down:
        s = s.replace_hp(1, int(s.active[1]), 0)
    return s


def dead_monferno(tables: Tables, names: Names, **kw) -> BattleState:
    """Prinplup active (slot 0) against Geodude, Monferno already dead: ``hp`` 0,
    ``deaths`` and ``player_faints`` recorded, the battle not over. ``kw`` go to
    ``roark_state``."""
    s = roark_state(tables, names, player_count=2, opp_count=1, player_species="Prinplup", **kw)
    monferno = next(i for i in range(2) if int(s.species[0, i]) == names.species("Monferno"))
    return s.replace_hp(0, monferno, 0)._replace(
        deaths=s.deaths.at[monferno].set(1), player_faints=s.player_faints + 1
    )
