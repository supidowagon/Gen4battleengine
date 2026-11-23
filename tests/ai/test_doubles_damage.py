"""``CheckIfHighestDamageWithPartner`` at B = 4 (A-7; ``trainer_ai.c:2369-2436``) and the doubles damage
refusal (E4-A plan decision 12).

The command is called directly: ``_result`` builds an ``AiContext`` for the attacker ``a`` against the target
``d`` (``commands.new_context``), enters slot ``slot`` and runs the command at ``USE_MAX_DAMAGE`` (the only
use, script.s:6684). The result is HIGHEST iff no move of the attacker and no move of ``GetPartner(attacker)``
(``attacker ^ 2``, battle_system.c:394-411) does strictly more damage to the target than the current slot
(``trainer_ai.c:2416-2431``). Damage here has no random draw (no Psywave / Magnitude), so the result is
certain and no read is made.

Every state is the four-battler world of ``test_tag_doubles`` (Bulbasaur everywhere, the target and the
partners Normal / Normal, so Tackle has STAB for them and not for the Grass / Poison attacker). Damage of
the moves used, against a Normal / Normal target, neutral: Tackle (35) < Body Slam (85) < Earthquake (100);
the partner's Tackle with STAB (x1.5) is still under the attacker's Earthquake.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai import policy
from nuzline.ai.layout import TAG_STRATEGY, VANILLA
from nuzline.ai.outcome import READ_DRAWS, no_calc_draws
from nuzline.engine.format import DOUBLES, MULTI, SINGLES
from tests.ai.fixtures import outcome
from tests.ai.test_tag_doubles import SLOT_READS, _faint, _hp, _World
from tests.engine.d2_states import four, with_moves

NO_COMPARISON, NOT_HIGHEST, HIGHEST = (
    c.AI_NO_COMPARISON_MADE,
    c.AI_NOT_HIGHEST_DAMAGE,
    c.AI_MOVE_IS_HIGHEST_DAMAGE,
)


@functools.partial(jax.jit, static_argnames=("a", "slot"))
def _run(t, s, d, a, slot):
    rolls = jnp.asarray(np.full((4, SLOT_READS), READ_DRAWS[0], np.uint8))
    ai_outcome = outcome()._replace(script_rolls=rolls, calc_draws=no_calc_draws((), SLOT_READS))
    ctx = c.new_context(t, s, ai_outcome, attacker=a, defender=d)
    ctx = c.start_run(c.at_slot(ctx, slot), TAG_STRATEGY)
    ctx = c.check_if_highest_damage_with_partner(ctx, c.USE_MAX_DAMAGE)
    return ctx.calc_temp, ctx.reads[slot]


def _result(world, s, a=1, d=0, slot=0, reads=0):
    """``calcTemp`` after the command for slot ``slot`` of attacker ``a`` against target ``d``."""
    result, made = _run(world.t, s, jnp.asarray(d, jnp.int32), a=a, slot=slot)
    assert int(made) == reads, "reads of the command"
    return int(result)


def _moves(world, s, b, *moves):
    return world.moves(s, b, *moves)


@pytest.fixture(scope="module")
def w(tables, names):
    """Attacker 1 (partner 3) against target 0 (partner 2), DOUBLES."""
    return _World(tables, names, DOUBLES, 1)


@pytest.fixture(scope="module")
def w_multi(tables, names):
    return _World(tables, names, MULTI, 2)  # the AI partner: its partner is the player's battler 0


def test_partner_stronger_not_highest(w):
    # sweep 2 (:2414, :2398-2411) is the partner's own moveset: its Earthquake (100) beats the attacker's
    # Tackle (35, slot 0): NOT_HIGHEST (:2428-2431), although the attacker's own sweep alone is HIGHEST.
    s = _moves(w, _moves(w, w.s, 1, "Tackle"), 3, "Earthquake")
    assert _result(w, s) == NOT_HIGHEST
    # the attacker's own stronger move in another slot ends it at sweep 1
    s = _moves(w, _moves(w, w.s, 1, "Tackle", "Earthquake"), 3, "Tackle")
    assert _result(w, s, slot=0) == NOT_HIGHEST
    assert _result(w, s, slot=1) == HIGHEST  # the best of both movesets
    # a partner move 85 against the attacker's Earthquake: not stronger
    s = _moves(w, _moves(w, w.s, 1, "Earthquake"), 3, "Body Slam")
    assert _result(w, s) == HIGHEST


def test_partner_weaker_highest(w):
    # no value of either sweep above the slot's: HIGHEST (:2426-2427), also at equal damage (strict >,
    # :2420-2424): Earthquake against Earthquake, neither has STAB, the two have the same stats.
    attacker = _moves(w, w.s, 1, "Earthquake")
    assert _result(w, _moves(w, attacker, 3, "Tackle")) == HIGHEST  # 35 * 1.5 STAB < 100
    assert _result(w, _moves(w, attacker, 3, "Earthquake")) == HIGHEST  # equal
    assert _result(w, _moves(w, attacker, 3)) == HIGHEST  # no move: damage 0
    assert _result(w, _moves(w, attacker, 3, "Growl", "Swords Dance")) == HIGHEST  # no damage calc


def test_status_no_comparison(w):
    # the gate is FlagMoveDamageScore's (:2381-2394, :2433-2435): a status current move makes no
    # comparison whatever the partner holds; the sweeps are not run
    s = _moves(w, _moves(w, w.s, 1, "Growl"), 3, "Earthquake")
    assert _result(w, s) == NO_COMPARISON
    # nor a power <= 1 move (Fling: vanilla power 1, effect FLING outside the alt-power list)
    assert _result(w, _moves(w, _moves(w, w.s, 1, "Fling"), 3, "Earthquake")) == NO_COMPARISON


def test_fainted_partner_counts(w):
    # no HP test (:2398-2432): the fainted partner's own battleMons moves are compared; its record is
    # cleared (ai_record_clear) but the party data stays (battle_lib.c:2156-2226)
    s = _moves(w, _moves(w, w.s, 1, "Tackle"), 3, "Earthquake")
    assert _result(w, _faint(s, 3)) == NOT_HIGHEST
    assert _result(w, _hp(s, 3, 1)) == NOT_HIGHEST  # and at 1 HP
    s = _moves(w, _moves(w, w.s, 1, "Earthquake"), 3, "Tackle")
    assert _result(w, _faint(s, 3)) == HIGHEST


def test_partner_gyro_reads_partner_record(w):
    # Gyro Ball's power is 1 + 25 * monSpeedValues[defender] / monSpeedValues[battler] (:2995-3003),
    # ``battler`` the sweep's own: the partner reads its own entry of the record in sweep 2, not the
    # attacker's. Against the attacker's Tackle (35): a partner record of 10 against the target's 100 gives
    # power 150 (NOT_HIGHEST); 100 gives 26 (HIGHEST). The attacker's own entry is not read by sweep 2.
    base = _moves(w, _moves(w, w.s, 1, "Tackle"), 3, "Gyro Ball")

    def record(attacker_speed, partner_speed):
        rec = base.speed_rec
        for b, v in ((0, 100), (1, attacker_speed), (2, 100), (3, partner_speed)):
            rec = rec.at[b].set(v)
        return base._replace(speed_rec=rec)

    assert _result(w, record(100, 10)) == NOT_HIGHEST
    assert _result(w, record(100, 100)) == HIGHEST
    assert _result(w, record(10, 100)) == HIGHEST  # a slow attacker entry changes nothing
    assert _result(w, record(100, 10), d=2) == NOT_HIGHEST  # the target 2: its entry is 100 too


def test_multi_partner_sweep(w_multi):
    # MULTI: the partner of battler 2 is the player's battler 0 (battle_system.c:394-411); the target is an
    # enemy (1). Its moves are compared with its true data.
    s = _moves(w_multi, _moves(w_multi, w_multi.s, 2, "Tackle"), 0, "Earthquake")
    assert _result(w_multi, s, a=2, d=1) == NOT_HIGHEST
    s = _moves(w_multi, _moves(w_multi, w_multi.s, 2, "Earthquake"), 0, "Tackle")
    assert _result(w_multi, s, a=2, d=1) == HIGHEST


# --- the doubles damage refusal ---------------------------------------------------------------------


def _plain(w):
    """Every battler knows Tackle only: no spread move anywhere."""
    s = w.s
    for b in range(4):
        s = _moves(w, s, b, "Tackle")
    return s


@pytest.mark.fullset
def test_spread_refused_until_d45_8(w, names):
    # E4-A plan decision 12: until E2's D45-8 puts the doubles factors (screens x2/3, spread x3/4,
    # battle_lib.c:6982-6990, :7023-7044) into ``ai_damage_for_roll`` the policy names a refusal for
    # a B = 4 state that needs them; the D45-8 rebase deletes this test with the refusal, for
    # ``test_partner_earthquake_x3_4``.
    t, plain = w.t, _plain(w)
    assert not bool(policy._doubles_damage(t, plain))
    # a damaging RANGE_ALL_ADJACENT move of an AI battler or of its partner, three or more living
    for b in (1, 3):
        for move in ("Earthquake", "Surf", "Discharge"):
            assert bool(policy._doubles_damage(t, _moves(w, plain, b, move))), (b, move)
    # a RANGE_ADJACENT_OPPONENTS move, a side of two living battlers
    assert bool(policy._doubles_damage(t, _moves(w, plain, 1, "Rock Slide")))
    assert bool(policy._doubles_damage(t, _moves(w, plain, 3, "Razor Leaf")))
    # a status move, or a move of the player's side only: no factor
    assert not bool(policy._doubles_damage(t, _moves(w, plain, 1, "Growl", "Protect")))
    assert not bool(policy._doubles_damage(t, _moves(w, plain, 0, "Earthquake")))
    assert not bool(policy._doubles_damage(t, _moves(w, plain, 2, "Rock Slide")))
    # ALL_ADJACENT needs a third living battler: with two down it does not apply
    quake = _moves(w, plain, 1, "Earthquake")
    assert bool(policy._doubles_damage(t, _faint(quake, 2)))
    assert not bool(policy._doubles_damage(t, _faint(_faint(quake, 2), 0)))
    # Reflect / Light Screen on a side with two living battlers
    for field in ("reflect", "light_screen"):
        for side in (0, 1):
            screened = plain._replace(**{field: getattr(plain, field).at[side].set(5)})
            assert bool(policy._doubles_damage(t, screened)), (field, side)
    screened = plain._replace(reflect=plain.reflect.at[0].set(5))
    assert not bool(policy._doubles_damage(t, _faint(screened, 2)))  # battler 0 alone: x1/2
    # the named reason is part of the scoring refusal
    refused = policy._refused(t, _moves(w, plain, 1, "Earthquake"))
    assert bool(refused.doubles_damage) and bool(refused.scoring)
    lines = policy._scoring_refusals(t, _moves(w, plain, 1, "Earthquake"), VANILLA, names)
    assert any("doubles damage factor (E2 D45-8)" in line for line in lines), lines
    # B = 2: no doubles factor (BATTLE_TYPE_DOUBLES unset)
    singles = with_moves(four(t, names, SINGLES), names, 1, ("Earthquake",))
    assert not bool(policy._doubles_damage(t, singles))
