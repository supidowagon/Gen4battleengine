"""D45-2: the AI's ability record (``aiContext.battlerAbilities``, ai_context.h:28) is written for
every kept battler (``state.ai_record_kept``): every battler at B = 4, battler 0 at B = 2
(``BattleAI_SetAbility``, battle_script.c:12106-12109, from :3073-3075, :3401-3404, :11795-11807).
Eager JAX against the reference; no jitted B = 4 ``step`` here (fast tier)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine import effects, hooks
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.effects import know_ability
from nuzline.engine.format import DOUBLES, MULTI
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Type
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import step
from nuzline.reference import engine as ref
from nuzline.reference.convert import from_ref, to_ref
from tests.engine.d2_states import four, with_ability, with_hp
from tests.engine.helpers import roark_state


def _known(s) -> list:
    return np.asarray(s.ai_ability_known).tolist()


def _id(ability) -> int:
    return int(ABILITY_DECOMP_IDS[ability])


def _turn(tables, fmt):
    return ref._Turn(to_numpy(tables), empty_outcome(), entry_turn=1, battler_count=fmt.B)


def test_switch_in_reveal_records_any_battler(tables, names):
    s = with_ability(four(tables, names, DOUBLES), 3, AbilityId.PRESSURE)
    out = hooks.reveal_ability(s, 3, s.active[3], hooks.SWITCH_IN, True)
    r = to_ref(s)
    ref._know_ability(r, 3)
    assert _known(out) == [0, 0, 0, _id(AbilityId.PRESSURE)] == r.ai_ability_known
    # Intimidate's "cuts" message is built in C and tags no ability (script.c:2955-2961):
    # it is not a reveal, so the row stays 0.
    s = with_ability(four(tables, names, DOUBLES), 3, AbilityId.INTIMIDATE)
    assert _known(hooks.reveal_ability(s, 3, s.active[3], hooks.SWITCH_IN, True)) == [0] * 4


def test_multi_partner_reveal_records_row_2(tables, names):
    s = with_ability(four(tables, names, MULTI), 2, AbilityId.PRESSURE)
    out = hooks.reveal_ability(s, 2, s.active[2], hooks.SWITCH_IN, True)
    r = to_ref(s)
    ref._know_ability(r, 2)
    assert _known(out) == [0, 0, _id(AbilityId.PRESSURE), 0] == r.ai_ability_known


def test_contact_reveal_records_the_enemy(tables, names):
    s = with_ability(four(tables, names, DOUBLES), 1, AbilityId.STATIC)
    want = [0, _id(AbilityId.STATIC), 0, 0]
    out = hooks.reveal_ability(s, 1, s.active[1], hooks.CONTACT_STATUS, True)
    assert _known(out) == want
    assert _known(know_ability(s, 1, s.active[1], True)) == want
    r = to_ref(s)
    ref._know_ability(r, 1)
    assert r.ai_ability_known == want
    # a phase the ability does not reveal at, and a false condition, write nothing
    assert _known(hooks.reveal_ability(s, 1, s.active[1], hooks.SWITCH_IN, True)) == [0] * 4
    assert _known(hooks.reveal_ability(s, 1, s.active[1], hooks.CONTACT_STATUS, False)) == [0] * 4


def test_trace_records_holder_and_target(tables, names):
    s = four(tables, names, DOUBLES)
    s = with_ability(with_ability(s, 1, AbilityId.TRACE), 0, AbilityId.STATIC)
    s = with_hp(s, 2, 0)  # battler 0 is the one Trace pick (ChooseTraceTarget, lib.c:7750-7781)
    out, traced = hooks.trace(tables, s, 1, 0)
    r = to_ref(s)
    assert ref._trace(to_numpy(tables), r, 1, 0)
    want = [_id(AbilityId.STATIC), _id(AbilityId.STATIC), 0, 0]
    assert bool(traced)
    assert _known(out) == want == r.ai_ability_known


def test_entry_hazards_immunity_records_enemy(tables, names):
    s = with_ability(four(tables, names, DOUBLES), 3, AbilityId.IMMUNITY)
    # Bulbasaur is a Poison type and would absorb the Toxic Spikes: make the newcomer Normal.
    slot = int(s.active[3])
    s = s._replace(
        types=s.types.at[1, slot].set(np.asarray([Type.NORMAL, Type.NORMAL], np.int8)),
        toxic_spikes=s.toxic_spikes.at[1].set(1),
    )
    out = effects.entry_hazards(tables, s, 3)
    r = to_ref(s)
    ref._entry_hazards(_turn(tables, DOUBLES), r, 3)
    assert _known(out) == [0, 0, 0, _id(AbilityId.IMMUNITY)] == r.ai_ability_known


def test_transform_settles_enemy_record(tables, names):
    """The user's "known, unnamed" record (-1) settles to its decomp id before the copy
    (``_transform``, rule 5): eager JAX ``effects._transform`` (no ``step`` compile) against the
    reference ``_transform``, battler 1 Transforming into battler 0."""
    s = with_ability(four(tables, names, DOUBLES), 1, AbilityId.STATIC)
    s = s._replace(ai_ability_known=s.ai_ability_known.at[1].set(-1))
    zero = jnp.asarray(0, jnp.int32)
    h = effects._Hit(**{f: zero for f in effects._Hit._fields})._replace(
        side=jnp.asarray(1, jnp.int32),
        target_side=zero,
        user=s.active[1],
        target=s.active[0],
        effect=jnp.asarray(effects._E.TRANSFORM, jnp.int32),
        landed=jnp.asarray(True),
    )
    out = effects._transform(tables, s, h, empty_outcome())
    r = to_ref(s)
    assert ref._transform(to_numpy(tables), r, 1, 0)
    assert _known(out) == r.ai_ability_known
    assert r.ai_ability_known[1] == _id(AbilityId.STATIC)
    assert np.asarray(out.ability).tolist() == np.asarray(from_ref(r).ability).tolist()


def test_singles_enemy_reveal_not_kept(tables, names):
    s = with_ability(roark_state(tables, names), 1, AbilityId.PRESSURE)
    out = hooks.reveal_ability(s, 1, s.active[1], hooks.SWITCH_IN, True)
    assert _known(out) == [0, 0]
    assert _known(know_ability(s, 1, s.active[1], True)) == [0, 0]
    r = to_ref(s)
    ref._know_ability(r, 1)
    assert r.ai_ability_known == [0, 0]
    s = with_ability(s, 0, AbilityId.PRESSURE)  # battler 0's reveals as today
    out = hooks.reveal_ability(s, 0, s.active[0], hooks.SWITCH_IN, True)
    r = to_ref(s)
    ref._know_ability(r, 0)
    assert _known(out) == [_id(AbilityId.PRESSURE), 0] == r.ai_ability_known


@pytest.mark.slow
def test_step_records_a_switched_in_enemy_pressure(tables, names):
    """B = 4 ``step`` against ``ref_step``: the enemy battler 3 switches to a Pressure Pokemon;
    its switch-in message records row 3 in both (subscript_pressure.s:6)."""
    s = four(tables, names, DOUBLES)
    s = s._replace(ability=s.ability.at[1, 2].set(int(AbilityId.PRESSURE)))
    acts = jnp.asarray([0, 0, 0, MOVE_ACTIONS + 2], jnp.int32)
    targets = [1, 0, 1, 0]
    o = empty_outcome(DOUBLES)
    out = step(tables, s, acts, o, jnp.asarray(targets, jnp.int8))
    ref_out = from_ref(
        ref.ref_step(
            to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o), targets
        )
    )
    assert int(out.active[3]) == 2
    assert _known(out)[3] == _id(AbilityId.PRESSURE)
    for name, a, b in zip(out._fields, out, ref_out, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
