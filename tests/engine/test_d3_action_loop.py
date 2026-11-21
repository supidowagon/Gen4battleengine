"""D3-5: the four-actor action loop (sheet D3-5 4.5). ``execute_actions`` runs ``battlerActionOrder``
over ``fmt.B`` positions; each action reads the turn flags of its own defender (Protect, Endure,
Roost) and its item states follow the attacker, the defender, then every other battler
(bcp.c:873-895, :3007, :3684-3696, :3730-3756, lib.c:2540-2543).

The order is the real CalcTurnOrder (D3-3): each test names the order it wants and ``play`` gives
the battlers distinct speeds that produce it (``speed_ordered``; every battler FIGHTs, an absent one
goes last). The engine side is the action phase of ``step`` (``play``); the reference side is
``ref_step(..., actions_only=True)``, which plays the same ``_actions`` loop and stops before the
end of the turn, as ``execute_actions`` does. The full state is compared field by field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import chosen_targets
from nuzline.engine.format import DOUBLES, SINGLES
from nuzline.engine.ids import STARF_QUEUE, AbilityId, Stat, Status, Type
from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import (
    ACT_MOVE,
    _clear_stale_choice_locks,
    _execute_actions,
    _record_last_move,
    replacement_power_mul,
    resolve_starf,
)
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.helpers import as_species, assert_matches_reference
from tests.engine.test_pivot import _mon

TACKLE, SWORDS, EARTH_POWER = "Tackle", "Swords Dance", "Earth Power"


@jax.jit
def _engine(tables, state, actions, outcome, targets, order=None):
    """``step._action_phase``: the turn's prologue (the Starf draw, the stale locks, the stored
    targets, the last moves), the order and the action loop; ``(state, shown)``. ``order``: an
    explicit battler order in place of CalcTurnOrder's (``play_forced``)."""
    queue = jnp.asarray(outcome.starf)[:, jnp.asarray(STARF_QUEUE)]
    opened = (state.starf_open != 0).astype(state.starf_taken.dtype)
    state = resolve_starf(state, queue[:, 0])
    state = state._replace(starf_roll=queue.astype(state.starf_roll.dtype), starf_taken=opened)
    fresh = _clear_stale_choice_locks(tables, state, actions)
    fresh = fresh._replace(chosen_target=chosen_targets(tables, state, actions, targets))
    fresh = _record_last_move(tables, fresh, actions)
    if order is None:
        order = action_order(tables, fresh, actions, outcome)
    order = jnp.asarray(order).astype(jnp.int32)
    out, _, shown, _, _ = _execute_actions(
        tables,
        fresh,
        actions,
        outcome,
        order,
        jnp.bool_(False),
        replacement_power_mul(tables, state, actions),
    )
    # The turn's end puts the Starf draw back to 0 where nothing keeps it (``step_events``).
    return out._replace(
        starf_roll=jnp.zeros_like(out.starf_roll), starf_taken=jnp.zeros_like(out.starf_taken)
    ), shown


def speed_ordered(state, order):
    """``state`` with distinct speeds that put the battlers in ``order`` (fastest first)."""
    for rank, b in enumerate(order):
        p, slot = party_of(state, b), int(state.active[b])
        state = state._replace(stats=state.stats.at[p, slot, Stat.SPE - 1].set(400 - 40 * rank))
    return state


def play_forced(tables, state, actions, order, targets=(0, 0, 0, 0), outcome=None):
    """Engine only: the action phase in an ``order`` CalcTurnOrder cannot give (a -6 priority
    mover acting before a priority-0 one). The reference has no such seam; the caller asserts the
    expected values itself. Returns ``(state, shown)``."""
    fmt_b = int(state.active.shape[0])
    outcome = empty_outcome(DOUBLES if fmt_b == 4 else SINGLES) if outcome is None else outcome
    return _engine(
        tables, state, jnp.asarray(actions, jnp.int32), outcome, jnp.asarray(targets, jnp.int8),
        jnp.asarray(order, jnp.int32),
    )  # fmt: skip


def play(tables, state, actions, order, targets=(0, 0, 0, 0), outcome=None):
    """The action phase of ``actions`` on the engine, asserted equal to the reference's (every
    field); returns ``(state, shown)`` of the engine. The battlers get the speeds that make
    CalcTurnOrder give ``order`` (an absent battler, 0 HP, goes last whatever it says);
    ``order=None`` keeps the state's speeds, and the outcome's ``order_tie`` decides ties."""
    fmt_b = int(state.active.shape[0])
    outcome = empty_outcome(DOUBLES if fmt_b == 4 else SINGLES) if outcome is None else outcome
    acts = jnp.asarray(actions, jnp.int32)
    if order is not None:
        present = [int(state.hp[party_of(state, b), int(state.active[b])]) > 0 for b in order]
        wanted = [b for b, p in zip(order, present, strict=True) if p]
        wanted += [b for b, p in zip(order, present, strict=True) if not p]
        state = speed_ordered(state, wanted)
        assert [int(b) for b in action_order(tables, state, acts, outcome)] == wanted
    tg = jnp.asarray(targets, jnp.int8)
    out, shown = _engine(tables, state, acts, outcome, tg)
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in actions),
            outcome_to_py(outcome),
            [int(t) for t in targets],
            actions_only=True,
        )
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        if name == "flinched":
            continue  # SetupNextTurn clears every flinch (lib.c:2234): the end of the turn's, not played
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out, shown


def hp(state, b: int) -> int:
    return int(state.hp[party_of(state, b), int(state.active[b])])


def pp(state, b: int, slot: int) -> int:
    return int(state.pp[party_of(state, b), int(state.active[b]), slot])


def _set_item(state, b: int, item: int):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(item=state.item.at[p, slot].set(item))


def _set_status(state, b: int, status: int):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(status=state.status.at[p, slot].set(status))


def _set_types(state, b: int, types):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(types=state.types.at[p, slot].set(jnp.asarray(types, state.types.dtype)))


def _all_moves(state, names, move: str):
    """Every battler knows only ``move`` in slot 0."""
    for b in range(4):
        state = with_moves(state, names, b, [move])
    return state


def test_four_tackles_all_act_in_the_order_given(tables, names):
    """Order (2, 1, 0, 3): all four act, each PP spent once, ``last_move`` set for each."""
    s = _all_moves(four(tables, names), names, TACKLE)
    out, shown = play(tables, s, [0, 0, 0, 0], (2, 1, 0, 3), targets=(1, 0, 3, 2))
    tackle = names.move(TACKLE)
    for b in range(4):
        assert pp(out, b, 0) == pp(s, b, 0) - 1, b
        assert int(out.last_move[b]) == tackle, b
        assert int(shown["act_kind"][b]) == ACT_MOVE, b
    # each Tackle reached its chosen target (the four Bulbasaur are alike and no one fainted)
    assert all(hp(out, b) < hp(s, b) for b in range(4))


def test_protect_of_the_first_actor_blocks_its_foe_not_the_others(tables, names):
    """Protect by battler 1 (first): battler 0's Tackle at 1 fails, battler 2's Tackle at 3 hits."""
    s = _all_moves(four(tables, names), names, TACKLE)
    s = with_moves(s, names, 1, ["Protect"])
    out, shown = play(tables, s, [0, 0, 0, 0], (1, 0, 2, 3), targets=(1, 0, 3, 2))
    assert hp(out, 1) == hp(s, 1)  # protected
    assert hp(out, 3) < hp(s, 3)  # battler 2's Tackle landed
    assert int(shown["protect_ok"][1]) == 1


def test_roost_by_battler_three_is_read_through_each_defender(tables, names):
    """Roost by battler 3 (Flying), then battler 0's Ground move at 3 hits (the chart has no
    Flying), battler 2's Ground move at 1 (Flying, no Roost) does not."""
    s = four(tables, names)
    s = with_moves(s, names, 0, [EARTH_POWER])
    s = with_moves(s, names, 2, [EARTH_POWER])
    s = with_moves(s, names, 3, ["Roost"])
    s = with_moves(s, names, 1, [TACKLE])
    flying = (Type.FLYING, Type.FLYING)
    s = _set_types(_set_types(s, 3, flying), 1, flying)
    s = with_hp(s, 3, hp(s, 3) // 2)  # Roost works below full HP
    out, shown = play(tables, s, [0, 0, 0, 0], (3, 0, 2, 1), targets=(3, 0, 1, 0))
    assert int(shown["roosted"][3]) == 1
    assert hp(out, 1) == hp(s, 1)  # battler 2's Earth Power met Flying: no effect
    # Roost healed to full, then battler 0's Earth Power hit the Roosting battler
    assert hp(out, 3) < int(s.max_hp[party_of(s, 3), int(s.active[3])])


def test_the_defenders_contact_ability_is_on_the_defenders_row(tables, names):
    """Battler 0 hits battler 3 with a contact move against Static: the ``contact_ability`` event
    is on 3's row, not on the old foe's (battler 1)."""
    s = _all_moves(four(tables, names), names, SWORDS)
    s = with_moves(s, names, 0, [TACKLE])
    p3, slot3 = party_of(s, 3), int(s.active[3])
    s = as_species(tables, s, p3, slot3, names.species("Pikachu"))  # ability slot 0: Static
    assert int(s.ability[p3, slot3]) == AbilityId.STATIC
    o = empty_outcome(DOUBLES)
    o = o._replace(contact_proc=o.contact_proc.at[3, 0].set(1))
    out, shown = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=(3, 0, 0, 0), outcome=o)
    assert int(shown["contact_ability"][3]) != 0
    assert int(shown["contact_ability"][1]) == 0
    assert int(out.status[party_of(s, 0), int(s.active[0])]) == Status.PARALYSIS


def test_every_battlers_status_item_is_checked_at_an_action_end(tables, names):
    """Rule 4: battler 0 hits battler 1; battler 2, not involved, poisoned and holding a Pecha
    Berry, is cured by the end of the turn's actions (the pass over every battler after each action)."""
    s = _all_moves(four(tables, names), names, SWORDS)
    s = with_moves(s, names, 0, [TACKLE])
    s = _set_status(s, 2, Status.POISON)
    s = _set_item(s, 2, names.item("Pecha Berry"))
    out, _ = play(tables, s, [0, 0, 0, 0], (0, 2, 1, 3), targets=(1, 0, 0, 0))
    p2, slot2 = party_of(s, 2), int(s.active[2])
    assert int(out.status[p2, slot2]) == Status.NONE
    assert int(out.item[p2, slot2]) == 0


def test_b2_sitrus_berry_is_eaten_once_by_a_self_defender_move(tables, names):
    """B = 2 (sheet 4.4): the user at <= 1/2 HP holding a Sitrus Berry uses Swords Dance, the foe
    holds a Lum Berry: Sitrus eaten once, Lum untouched; ``step`` equals the reference."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Slowpoke", ["Swords Dance"], 30, "Sitrus Berry")],
        [_mon(n, "Geodude", ["Leer"], 30, "Lum Berry")],
    )
    s = s._replace(hp=s.hp.at[0, 0].set(int(s.max_hp[0, 0]) // 2))
    out = assert_matches_reference(tables, s, (0, 0), empty_outcome())
    assert int(out.item[0, 0]) == 0  # eaten
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) + int(s.max_hp[0, 0]) // 4  # once
    assert int(out.item[1, 0]) == n.item("Lum Berry")  # untouched


def test_me_first_copies_the_resolved_defenders_move(tables, names):
    """B = 4 Me First (ledger D3-5): battler 2 targets battler 3, which has not acted and chose
    Scratch; battler 1 chose a status move. The copy is battler 3's, engine equal to reference."""
    s = _all_moves(four(tables, names), names, SWORDS)
    s = with_moves(s, names, 2, ["Me First"])
    s = with_moves(s, names, 3, ["Scratch"])
    out, _ = play(tables, s, [0, 0, 0, 0], (2, 3, 0, 1), targets=(0, 0, 3, 2))
    assert hp(out, 3) < hp(s, 3)  # the copied Scratch hit its target
    assert int(out.last_move[2]) == names.move("Me First")
    # the same turn aimed at battler 1, whose chosen move is not a damaging one: the call fails
    other, _ = play(tables, s, [0, 0, 0, 0], (2, 3, 0, 1), targets=(0, 0, 1, 2))
    assert hp(other, 1) == hp(s, 1)
