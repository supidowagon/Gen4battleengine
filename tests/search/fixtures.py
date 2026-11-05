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
from nuzline.search.leaves import LeafValue
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


def certain_zero_leaf(states, weights, pairs=None) -> list[LeafValue]:
    """A ``leaf_fn`` worth 0 with a negligible interval (``n = 10**6``). Unlike
    ``test_plan_tree.zero_leaf`` (``n = 1``, an interval half-width of about 0.9 per unit of
    leaf weight) it cannot make a zero leaf's interval overlap a valued line's, and it
    reports no expected deaths, so the tie rule never reorders a solve by them."""
    return [LeafValue(0, 10**6, 0, 10**6, False) for _ in range(len(states.hp))]


def _knowing_only(state: BattleState, slot: int, move_slot: int) -> BattleState:
    """The player's party member ``slot`` with the PP of every move but ``move_slot`` zeroed."""
    pp = [int(p) if i == move_slot else 0 for i, p in enumerate(state.pp[0, slot])]
    return state.with_pp(0, slot, pp)


def _team_of_four(tables: Tables, names: Names) -> BattleState:
    """Monferno, Prinplup, (Grotle's slot emptied) and Starly against Geodude alone; Roark's
    bag emptied and Geodude knowing only Tackle."""
    s = empty_bag(roark_state(tables, names, player_count=4, opp_count=1))
    s = s._replace(species=s.species.at[0, 2].set(0), hp=s.hp.at[0, 2].set(0))
    return s.with_pp(1, 0, [int(p) if i == 0 else 0 for i, p in enumerate(s.pp[1, 0])])


def _bubble_slot(names: Names, state: BattleState) -> int:
    return np.asarray(state.moves[0, 1]).tolist().index(names.move("Bubble"))


def _facing(state: BattleState, slot: int) -> BattleState:
    """``state`` with the player's party member ``slot`` active (for ``damage_table``)."""
    return state._replace(active=state.active.at[0].set(slot))


def faint_pending_root(tables: Tables, names: Names) -> BattleState:
    """A player replacement decision: Monferno fainted (``deaths[0] = 1``, ``player_faints =
    1``) with Prinplup (Bubble only) and Starly to send in against a full-HP Geodude
    (Tackle only, slower than Prinplup), ``with_budget(k=1)``. Asserts the numbers the tests
    rely on: one Bubble always KOs Geodude, Starly cannot KO it in one hit (even critical),
    and Prinplup is faster."""
    from nuzline.engine.damage import damage_table

    s = at_replacement(_team_of_four(tables, names))
    s = _knowing_only(s, 1, _bubble_slot(names, s))
    hp = int(s.hp[1, 0])
    bubble = damage_table(tables, _facing(s, 1), 0, names.move("Bubble"), 0)
    assert int(np.asarray(bubble).min()) >= hp, "one Bubble does not always KO Geodude"
    for move in np.asarray(s.moves[0, 3]).tolist():
        if move > 0:
            table = damage_table(tables, _facing(s, 3), 0, move, 1)
            assert int(np.asarray(table).max()) < hp, "Starly can KO Geodude in one hit"
    assert int(s.stats[1, 0, _SPE]) < int(s.stats[0, 1, _SPE]), "Geodude is not slower"
    return s.with_budget(1)


_SACRIFICE_CHECKED: set[int] = set()


def sacrifice_endgame(tables: Tables, names: Names) -> BattleState:
    """The ``faint_pending_root`` team before the faint: no deaths, ``faint_budget`` 0 (set
    ``k`` with ``with_budget``), Monferno alive at 1 HP knowing only Leer with its Spe stat
    one below Geodude's, Prinplup at 4 HP knowing only Bubble, Geodude at full HP knowing
    only Tackle. A direct switch to Prinplup eats a Tackle that KOs it 95 % of the time;
    after Leer sacrifices Monferno, Prinplup enters free, moves first and wins. Asserts the
    speed order, both KO facts and, with ``exhaustive`` (cap 2), that the death budget is
    worth at least 0.3 here and that some root value lies strictly between 0 and 1."""
    from nuzline.engine.damage import damage_table
    from tests.search.exhaustive import exhaustive

    s = _team_of_four(tables, names)
    s = _knowing_only(s, 0, np.asarray(s.moves[0, 0]).tolist().index(names.move("Leer")))
    s = _knowing_only(s, 1, _bubble_slot(names, s))
    s = s.replace_hp(0, 0, 1).replace_hp(0, 1, 4)
    s = s._replace(stats=s.stats.at[0, 0, _SPE].set(s.stats[1, 0, _SPE] - 1))
    spe_monferno, spe_geodude, spe_prinplup = (
        int(s.stats[i, j, _SPE]) for i, j in ((0, 0), (1, 0), (0, 1))
    )
    assert spe_monferno < spe_geodude < spe_prinplup, "Spe order Monferno < Geodude < Prinplup"
    tackle = damage_table(tables, _facing(s, 1), 1, names.move("Tackle"), 0)
    assert int(np.asarray(tackle).min()) >= 4, "a Tackle that hits does not KO Prinplup"
    bubble = damage_table(tables, _facing(s, 1), 0, names.move("Bubble"), 0)
    assert int(np.asarray(bubble).min()) >= int(s.hp[1, 0]), "one Bubble does not KO Geodude"
    if id(tables) not in _SACRIFICE_CHECKED:
        exact = [exhaustive(tables, s.with_budget(k), 2)[0] for k in (0, 1)]
        assert max(exact[1].values()) - max(exact[0].values()) >= 0.3, "the budget is worth <0.3"
        assert any(0 < v < 1 for e in exact for v in e.values()), "every root value is 0 or 1"
        _SACRIFICE_CHECKED.add(id(tables))
    return s
