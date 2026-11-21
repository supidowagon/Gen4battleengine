"""D3-8b: ``sideGetExpMask`` is one mask per enemy battler (``BattleContext.sideGetExpMask`` is
``u32[2]`` indexed ``(battler >> 1) & 1`` of the enemy battler, battle_context.h:176; lib.c:1458,
:1470; script.c:2205, :9700, :9709, :9741, :10191). Battler 1's payout reads and clears entry 0
only, so battler 3's payout in the same pass pays its own participants from entry 1.

All Pokemon are level-50 Bulbasaur (``d2_states.four``), so ``E`` is the same for every foe unless
a test changes a level; ``pay`` is the trainer 150 / 100 (script.c:9753-9755) at ``exp_boost``
100 (no OT multiplier)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS, format_of, party
from nuzline.engine.levelup import pay_exp
from nuzline.engine.state import layout
from nuzline.engine.step import _pay_all, _switch_in
from tests.engine.d2_states import four

HOLDER = 2  # party 0's Exp. Share holder (a bench slot)


def exp_e(tables, s, p: int, j: int) -> int:
    """``E = yield[species] * level / 7`` of party ``p`` slot ``j`` (script.c:2216-2217)."""
    return int(tables.species_exp_yield[int(s.species[p, j])]) * int(s.level[p, j]) // 7


def pay(t: int) -> int:
    return t * 150 // 100


def share_g(e: int, n: int, holders: int) -> int:
    """``gainedExp`` (script.c:2219-2238)."""
    return max(1, (e // 2) // n) if holders else max(1, e // n)


def share_h(e: int, holders: int) -> int:
    """``sharedExp`` (script.c:2219-2230)."""
    return max(1, (e // 2) // holders)


def both_foes_faint(s):
    """``(before, after)`` with the Pokemon of battlers 1 and 3 fainted in ``after``."""
    fmt = format_of(s)
    p1, p3 = party(fmt, 1), party(fmt, 3)
    hp = s.hp.at[p1, int(s.active[1])].set(0).at[p3, int(s.active[3])].set(0)
    return s, s._replace(hp=hp)


def gains(before, after):
    return (np.asarray(after.exp) - np.asarray(before.exp)).tolist()


def masks(s) -> list[int]:
    return np.asarray(s.exp_mask).tolist()


# ---------------------------------------------------------------- T0: the layout


def test_exp_mask_layout_per_format():
    assert layout(SINGLES)["exp_mask"] == ((), np.int8)
    assert layout(DOUBLES)["exp_mask"] == ((2,), np.int8)
    assert layout(TWO_TRAINERS)["exp_mask"] == ((2,), np.int8)
    assert layout(MULTI)["exp_mask"] == ((2,), np.int8)


def test_initial_exp_mask_is_the_player_leads_in_both_entries(tables, names):
    assert masks(four(tables, names, DOUBLES)) == [0b11, 0b11]
    assert masks(four(tables, names, TWO_TRAINERS)) == [0b11, 0b11]
    assert masks(four(tables, names, MULTI)) == [0b1, 0b1]  # battler 2's slot is never flagged
    assert int(four(tables, names, SINGLES).exp_mask) == 1


# ---------------------------------------------------------------- T1-T4: the payouts


def test_doubles_both_foes_faint_each_pays_its_own_participants(tables, names):
    s = four(tables, names, DOUBLES)
    before, after = both_foes_faint(s)
    out = _pay_all(tables, before, after)
    e1, e3 = exp_e(tables, s, 1, 0), exp_e(tables, s, 1, 1)
    want = pay(share_g(e1, 2, 0)) + pay(share_g(e3, 2, 0))
    assert gains(s, out) == [want, want, 0, 0, 0, 0]
    assert masks(out) == [0, 0]


def test_doubles_exp_share_holder_gets_h_from_each_payout(tables, names):
    s = four(tables, names, DOUBLES)
    s = s._replace(item=s.item.at[0, HOLDER].set(names.item("Exp. Share")))
    before, after = both_foes_faint(s)
    out = _pay_all(tables, before, after)
    e1, e3 = exp_e(tables, s, 1, 0), exp_e(tables, s, 1, 1)
    g = pay(share_g(e1, 2, 1)) + pay(share_g(e3, 2, 1))
    h = pay(share_h(e1, 1)) + pay(share_h(e3, 1))
    assert gains(s, out) == [g, g, h, 0, 0, 0]
    assert masks(out) == [0, 0]


def test_multi_pays_party_0_only_from_each_entry(tables, names):
    s = four(tables, names, MULTI)
    before, after = both_foes_faint(s)
    out = _pay_all(tables, before, after)
    e1, e3 = exp_e(tables, s, 1, 0), exp_e(tables, s, 3, 0)
    assert gains(s, out) == [pay(share_g(e1, 1, 0)) + pay(share_g(e3, 1, 0)), 0, 0, 0, 0, 0]
    assert masks(out) == [0, 0]


def test_two_trainers_both_foes_faint(tables, names):
    s = four(tables, names, TWO_TRAINERS)
    s = s._replace(level=s.level.at[2, 0].set(80))
    before, after = both_foes_faint(s)
    out = _pay_all(tables, before, after)
    e1, e3 = exp_e(tables, s, 1, 0), exp_e(tables, s, 2, 0)
    assert e1 != e3
    want = pay(share_g(e1, 2, 0)) + pay(share_g(e3, 2, 0))
    assert gains(s, out) == [want, want, 0, 0, 0, 0]
    assert masks(out) == [0, 0]


# ---------------------------------------------------------------- T5, T6: the writers


def test_send_outs_flag_and_clear_per_entry(tables, names):
    """A player send-out flags both entries; an enemy send-out clears and flags its own."""
    s = four(tables, names, DOUBLES)
    s = _switch_in(tables, s, 0, 2, 1)  # battler 0 to slot 2 (lib.c:1461-1479, both entries)
    assert masks(s) == [0b111, 0b111]
    s = _switch_in(tables, s, 3, 2, 1)  # battler 3 to party 1 slot 2 (clear, then flag entry 1)
    assert masks(s) == [0b111, 0b110]
    before, after = both_foes_faint(s)
    out = _pay_all(tables, before, after)
    e1, e3 = exp_e(tables, s, 1, 0), exp_e(tables, s, 1, 2)
    first = pay(share_g(e1, 3, 0))
    second = pay(share_g(e3, 2, 0))
    assert gains(s, out) == [first, first + second, first + second, 0, 0, 0]


def test_a_foe_without_hp_is_not_flagged(tables, names):
    s = four(tables, names, DOUBLES)
    s = s._replace(
        hp=s.hp.at[1, int(s.active[3])].set(0),
        exp_mask=jnp.zeros_like(s.exp_mask),
    )
    s = _switch_in(tables, s, 2, 2, 1)  # battler 2 to slot 2
    assert masks(s) == [0b101, 0]  # entry 0: slots 0 and 2; entry 1: battler 3 has no HP


def test_an_absent_player_battler_is_not_flagged(tables, names):
    s = four(tables, names, DOUBLES, bench=3)
    s = s._replace(hp=s.hp.at[0, 1].set(0).at[0, 2].set(0))  # battler 2: 0 HP, empty bench
    s = _switch_in(tables, s, 1, 2, 1)  # battler 1 to party 1 slot 2: clear, then flag entry 0
    assert masks(s) == [0b001, 0b11]


def test_multi_flags_only_battler_0s_slot(tables, names):
    s = four(tables, names, MULTI)
    s = s._replace(exp_mask=jnp.zeros_like(s.exp_mask))
    s = _switch_in(tables, s, 2, 1, 1)  # the partner (party 2) to slot 1
    assert masks(s) == [0b1, 0b1]


# ---------------------------------------------------------------- T7: B = 2


def test_singles_mask_stays_a_scalar(tables, names):
    s = four(tables, names, SINGLES)
    assert np.asarray(s.exp_mask).shape == () and s.exp_mask.dtype == jnp.int8


# ---------------------------------------------------------------- D9: the party item


def foe_faints(s, battler: int = 1):
    return s._replace(hp=s.hp.at[1, int(s.active[battler])].set(0))


def holding(tables, names, slot: int, item: str, knocked: bool):
    """DOUBLES, party 0 ``slot`` holds ``item``: on the battle copy, or (``knocked``) knocked off:
    the battle copy empty, the party keeping it (battle_io_command.c:590-592)."""
    s = four(tables, names, DOUBLES)
    held = names.item(item)
    if not knocked:
        return s._replace(item=s.item.at[0, slot].set(held))
    return s._replace(
        knocked_off=s.knocked_off.at[0, slot].set(1),
        knocked_item=s.knocked_item.at[0, slot].set(held),
    )


def test_a_knocked_off_lucky_egg_still_boosts(tables, names):
    """The payout reads ``MON_DATA_HELD_ITEM`` of the party (script.c:9706, :9749-9751): the
    Lucky Egg a Knock Off took from the battle copy still gives its 150 / 100."""
    held = holding(tables, names, 0, "Lucky Egg", False)
    knocked = holding(tables, names, 0, "Lucky Egg", True)
    plain = four(tables, names, DOUBLES)
    a = pay_exp(tables, held, foe_faints(held), battler=1)
    b = pay_exp(tables, knocked, foe_faints(knocked), battler=1)
    c = pay_exp(tables, plain, foe_faints(plain), battler=1)
    np.testing.assert_array_equal(np.asarray(a.exp), np.asarray(b.exp))
    np.testing.assert_array_equal(np.asarray(a.evs), np.asarray(b.evs))
    assert int(b.exp[0]) - int(knocked.exp[0]) > int(c.exp[0]) - int(plain.exp[0])
    e = exp_e(tables, plain, 1, 0)
    assert gains(knocked, b)[0] == share_g(e, 2, 0) * 150 // 100 * 150 // 100


def test_a_knocked_off_exp_share_still_pays_the_bench(tables, names):
    """A slot whose Exp. Share was knocked off is still a holder (script.c:2210-2212, :9709):
    ``s`` counts it and it is visited and paid ``h`` from the payout."""
    held = holding(tables, names, HOLDER, "Exp. Share", False)
    knocked = holding(tables, names, HOLDER, "Exp. Share", True)
    a = pay_exp(tables, held, foe_faints(held), battler=1)
    b = pay_exp(tables, knocked, foe_faints(knocked), battler=1)
    np.testing.assert_array_equal(np.asarray(a.exp), np.asarray(b.exp))
    e = exp_e(tables, knocked, 1, 0)
    assert gains(knocked, b) == [
        pay(share_g(e, 2, 1)),
        pay(share_g(e, 2, 1)),
        pay(share_h(e, 1)),
        0,
        0,
        0,
    ]
