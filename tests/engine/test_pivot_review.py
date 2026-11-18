"""E2 batch 6 review follow-ups (rulings B6-10.., docs/kaizo/e2-mechanics-ledger.md): the
resumed half's laws against the live threshold, and hand cases of the pivot each checked against
the reference engine (``assert_matches_reference``) on both halves of the turn."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import enumerate_turn
from nuzline.engine.ids import Stat, Status
from nuzline.engine.outcome import MAX_HITS, empty_outcome, live_hit_threshold
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()


def _resume(tables, paused, slot, outcome=O):
    side = int(paused.pivot_battler) - 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[side] = MOVE_ACTIONS + slot
    return assert_matches_reference(tables, paused, tuple(acts), outcome), tuple(acts)


@pytest.mark.parametrize("move", ["Baton Pass", "U-turn"])
def test_resumed_hit_law_reads_the_newcomers_evasion(tables, names, move):
    """B6-10: Baton Pass carries +2 evasion to the newcomer, U-turn does not; the foe's Tackle
    after the resume hits with the live threshold's probability in the enumeration (each row
    stepped, independent of ``sample_outcome``)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Ninjask", [move], 40), _mon(n, "Geodude", ["Growl"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 40)],
    ).with_stages(0, eva=2)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    outcomes, probs = enumerate_turn(tables, paused, acts)
    import jax

    from nuzline.engine.step import step

    after = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))(tables, paused, acts, outcomes)
    hit = np.asarray(after.hp)[:, 0, 1] < int(paused.hp[0, 1])
    newcomer = paused._replace(active=paused.active.at[0].set(1))
    eva = 2 if move == "Baton Pass" else 0
    newcomer = newcomer._replace(stages=jnp.zeros_like(newcomer.stages)).with_stages(0, eva=eva)
    live = int(live_hit_threshold(tables, newcomer, 1, n.move("Tackle")))
    assert math.isclose(float(np.sum(probs[hit])), live / 100, abs_tol=1e-9)
    assert (live < 95) == (move == "Baton Pass")


def test_destiny_bond_and_rough_skin_stop_the_switch(tables, names):
    """subscript_attack_then_switch_out.s:15-34 (Destiny Bond) and :48: a user that fainted
    switches nothing; its King's Rock still checks (``rock_now``)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 5), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0 and int(out.hp[0, 0]) == 0
    # Rough Skin KOs a 1 HP user: no switch.
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30, "King’s Rock"), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Carvanha", ["Tackle"], 30)],
    ).replace_hp(0, 0, 1)
    out = assert_matches_reference(
        tables, s, (0, 0), O._replace(flinch_item=jnp.ones((2, MAX_HITS), jnp.int8))
    )
    assert int(out.pivot_battler) == 0


def test_u_turn_ko_then_list_faint_check_and_payout(tables, names):
    """B6-4: the U-turn's KO is processed at the list's faint check, paid at MoveEnd; hazards
    can KO the newcomer."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 5), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.hp[1, 0]) == 0
    exp0 = int(np.asarray(paused.exp)[0])
    after, _ = _resume(tables, paused, 1)
    assert int(np.asarray(after.exp)[0]) > exp0  # the U-turn user was in the mask
    # Stealth Rock KOs a 1 HP newcomer at the list.
    s2 = s.replace_hp(0, 1, 1)._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    paused = assert_matches_reference(tables, s2, (0, 0), O)
    after, _ = _resume(tables, paused, 1)
    assert int(after.hp[0, 1]) == 0


def test_two_pauses_in_one_turn(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Ninjask", ["U-turn"], 30), _mon(n, "Onix", ["Tackle"], 30)],
    )
    first = assert_matches_reference(tables, s, (0, 0), O)
    assert int(first.pivot_battler) in (1, 2)
    second, _ = _resume(tables, first, 1)
    assert int(second.pivot_battler) != 0 and int(second.pivot_battler) != int(first.pivot_battler)
    done, _ = _resume(tables, second, 1)
    assert int(done.pivot_battler) == 0 and int(done.turn) == int(s.turn) + 1


def test_pursuit_kos_the_u_turn_user(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 25), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 35)],
    ).replace_hp(0, 0, 2)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0 and int(out.hp[0, 0]) == 0


def test_sleep_talk_u_turn_with_a_choice_item(tables, names):
    """B6-5: moveTemp is Sleep Talk: the Choice-holding newcomer locks into it."""
    n = names
    s = make_state(
        tables,
        [
            _mon(n, "Scyther", ["Sleep Talk", "U-turn"], 30),
            _mon(n, "Geodude", ["Sleep Talk", "Tackle"], 30, "Choice Band"),
        ],
        [_mon(n, "Bulbasaur", ["Growl"], 35)],
    ).with_status(0, 0, Status.SLEEP, 3)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    after, _ = _resume(tables, paused, 1)
    assert int(after.choice_lock[0]) == n.move("Sleep Talk")


def test_baton_pass_without_bench_and_natural_cure(tables, names):
    n = names
    s = make_state(
        tables, [_mon(n, "Ninjask", ["Baton Pass"], 40)], [_mon(n, "Bulbasaur", ["Tackle"], 20)]
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0
    # Natural Cure on the U-turn user (subscript :56-57): cured at the list.
    s = make_state(
        tables,
        [_mon(n, "Starmie", ["U-turn"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Growl"], 35)],
    )
    s = s._replace(ability=s.ability.at[0, 0].set(_natural_cure_id(tables, names)))
    s = s.with_status(0, 0, Status.POISON)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.status[0, 0]) == Status.NONE


def _natural_cure_id(tables, names):
    from nuzline.engine.ids import AbilityId

    return int(AbilityId.NATURAL_CURE)


@pytest.mark.parametrize("item,side_item", [("Shell Bell", 0), ("Sticky Barb", 1)])
def test_pivot_items(tables, names, item, side_item):
    """TriggerHeldItemOnPivotMove (battle_lib.c:7336-7386): Shell Bell heals the user before the
    list; the defender's Sticky Barb moves to the user."""
    n = names
    user_item = item if side_item == 0 else ""
    foe_item = item if side_item == 1 else ""
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30, user_item), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Growl"], 40, foe_item)],
    ).replace_hp(0, 0, 20)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    if side_item == 0:
        assert int(paused.hp[0, 0]) > 20
    else:
        assert int(paused.item[0, 0]) == n.item(item) and int(paused.item[1, 0]) == 0
    _resume(tables, paused, 1)


def test_roar_drag_runs_its_switch_in_effects(tables, names):
    """subscript_force_target_to_switch_or_flee.s:31: the dragged Pokemon's switch-in effects at
    AfterMoveEffects (the site the resume shares): an Intimidate newcomer lowers the user's
    Attack."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Roar"], 40)],
        [_mon(n, "Geodude", ["Harden"], 20), _mon(n, "Gyarados", ["Harden"], 20)],
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.active[1]) == 1 and int(out.stages[0, Stat.ATK]) == -1
