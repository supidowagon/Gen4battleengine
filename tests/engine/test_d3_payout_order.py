"""D3-8: the payout order (sheet D3-8 sections 1 and 5). ``AnyExpPayout`` (bcp.c:4449-4469) pays
the lowest set battler bit of ``SYSCTL_PAYOUT_EXP`` per call, so the payouts of one batch come
in ascending battler order (not party-slot order); a player battler pays nothing
(script.c:2196); the payer is the fainted battler's own party (script.c:2216-2217).

``levelup.pay_exp(..., battler=)`` is the static per-battler payout and ``step._pay_all`` the
order. The field-condition pass is driven directly (``field_conditions``): ``play`` runs only the
action phase."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, SINGLES, TWO_TRAINERS
from nuzline.engine.ids import Stat, Weather
from nuzline.engine.levelup import BOOST_PERCENT, pay_exp
from nuzline.engine.step import _pay_all, field_conditions
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_grant_exp
from tests.engine.d2_states import four

HOLDER = 2  # the player's Exp. Share holder (party 0 slot)


def species_yielding(tables, stat: int) -> int:
    """A species whose form-0 EV yield is exactly 1 in ``stat`` and nothing else."""
    rows = np.asarray(tables.species_form_row[:, 0])
    yields = np.asarray(tables.species_ev_yield)
    for species in range(1, rows.shape[0]):
        y = yields[rows[species]]
        if y[stat] == 1 and y.sum() == 1:
            return species
    raise AssertionError(f"no species yields only stat {stat}")


def sandstorm_double_faint(tables, names):
    """DOUBLES, sandstorm, battlers 1 and 3 at 1 HP: battler 3's Pokemon is party 1's slot 0 (a
    Speed-EV yielder), battler 1's is slot 1 (an Attack-EV yielder); party 0's slot ``HOLDER``
    holds an Exp. Share at 509 total EVs (one EV left under the 510 cap)."""
    s = four(tables, names, DOUBLES)
    s = s._replace(active=s.active.at[1].set(1).at[3].set(0))
    species = s.species.at[1, 1].set(species_yielding(tables, Stat.ATK))
    species = species.at[1, 0].set(species_yielding(tables, Stat.SPE))
    evs = s.evs.at[0, HOLDER].set(jnp.asarray([100, 100, 100, 100, 100, 9], s.evs.dtype))
    return s._replace(
        species=species,
        hp=s.hp.at[1, 0].set(1).at[1, 1].set(1),
        item=s.item.at[0, HOLDER].set(names.item("Exp. Share")),
        evs=evs,
        exp_mask=jnp.zeros_like(s.exp_mask),
        weather=jnp.asarray(Weather.SANDSTORM, s.weather.dtype),
        weather_timer=jnp.asarray(5, s.weather_timer.dtype),
    )


def test_doubles_sandstorm_double_faint_pays_battler_1_first(tables, names):
    """Battler 1 (slot 1) pays before battler 3 (slot 0): the holder is capped by the Attack EV
    and the Speed yield finds the 510 total reached. Slot order would give the reverse."""
    s = sandstorm_double_faint(tables, names)
    out = jax.jit(field_conditions)(tables, s, jnp.asarray([3, 1], jnp.int32))
    assert int(out.hp[1, 0]) == 0 and int(out.hp[1, 1]) == 0
    assert np.asarray(out.evs[0, HOLDER]).tolist() == [100, 101, 100, 100, 100, 9]
    assert int(out.exp[HOLDER]) > int(s.exp[HOLDER])


def test_pay_all_is_battler_1_then_battler_3(tables, names):
    s = sandstorm_double_faint(tables, names)
    after = s._replace(hp=s.hp.at[1, 0].set(0).at[1, 1].set(0))
    out = _pay_all(tables, s, after)
    first = pay_exp(tables, s, after, battler=1)
    assert np.asarray(out.evs[0, HOLDER]).tolist() == [100, 101, 100, 100, 100, 9]
    swapped = pay_exp(tables, s, first, battler=3)
    for name, x, y in zip(out._fields, out, swapped, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def test_two_trainers_battler_3_pays_from_party_2(tables, names):
    """Battler 3 is party 2's: its species and level, not party 1's, give E = yield * level / 7."""
    s = four(tables, names, TWO_TRAINERS)
    machop = names.species("Machop")
    s = s._replace(
        species=s.species.at[2, 0].set(machop),
        level=s.level.at[2, 0].set(80).at[1, 0].set(10),
        exp_mask=jnp.full_like(s.exp_mask, 1),
    )
    after = s._replace(hp=s.hp.at[2, 0].set(0))
    out = pay_exp(tables, s, after, battler=3)
    exp_e = int(tables.species_exp_yield[machop]) * 80 // 7
    gained = exp_e * BOOST_PERCENT // 100
    boost = int(s.exp_boost[0])
    if boost != 100:
        gained = gained * boost // 100
    assert int(out.exp[0]) - int(s.exp[0]) == gained
    # Battler 1's party 1 did not faint: no payout for it.
    assert int(pay_exp(tables, s, after, battler=1).exp[0]) == int(s.exp[0])


def test_player_battlers_pay_nothing(tables, names):
    s = four(tables, names, DOUBLES)
    after = s._replace(hp=s.hp.at[1].set(0).at[0].set(0))
    for battler in (0, 2):
        out = pay_exp(tables, s, after, battler=battler)
        for name, x, y in zip(out._fields, out, after, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def test_singles_pay_exp_without_battler_is_the_singles_payout_on_20_faints(tables, names):
    """B = 2: no ``battler`` is today's path; the static ``battler=1`` agrees with it and with the
    reference's ``ref_grant_exp`` on 20 drawn faints (slot, level, mask, Exp. Share)."""
    rng = np.random.default_rng(8)
    np_tables = to_numpy(tables)
    base = four(tables, names, SINGLES)
    for _ in range(20):
        j = int(rng.integers(0, 6))
        s = base._replace(
            active=base.active.at[1].set(j),
            level=base.level.at[1, j].set(int(rng.integers(2, 100))),
            exp_mask=jnp.asarray(int(rng.integers(0, 64)), base.exp_mask.dtype),
            item=base.item.at[0, int(rng.integers(0, 6))].set(names.item("Exp. Share")),
        )
        after = s._replace(hp=s.hp.at[1, j].set(0))
        plain = pay_exp(tables, s, after)
        fixed = pay_exp(tables, s, after, battler=1)
        ref = from_ref(ref_grant_exp(np_tables, to_ref(after), j))
        for name, x, y in zip(plain._fields, plain, fixed, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
        np.testing.assert_array_equal(np.asarray(plain.exp), np.asarray(ref.exp))
        np.testing.assert_array_equal(np.asarray(plain.evs), np.asarray(ref.evs))
