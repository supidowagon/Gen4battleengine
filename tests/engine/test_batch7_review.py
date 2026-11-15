"""E2 batch 7 follow-ups (rulings B7-12.., docs/kaizo/e2-mechanics-ledger.md), each hand case
checked against the reference engine (``assert_matches_reference``)."""

from __future__ import annotations

import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.ids import AbilityId
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()


@pytest.mark.parametrize("pivot", ["Baton Pass", "U-turn"])
def test_a_switch_never_runs_the_pivot_in_its_slot(tables, names, pivot):
    """B7-12 (the differential divergence found at f22b818, from 470fbef): a switch action to
    party slot 1 clips to move slot 3, and a Baton Pass or U-turn held there paused the action, so
    the switch ran again on the newcomer (its Magic Guard restored away, the hazards twice). The
    turn branches on the command picked (bcp.c:893): a switch runs no move script."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Pikachu", ["Tackle"], 19)],
        [
            _mon(n, "Chansey", ["Bide", "Disable", "Pursuit", pivot], 30),
            _mon(n, "Carvanha", ["Bite"], 20),
        ],
    )
    s = s._replace(
        ability=s.ability.at[1, 1].set(AbilityId.MAGIC_GUARD),
        stealth_rock=s.stealth_rock.at[1].set(1),
        spikes=s.spikes.at[1].set(2),
    )
    hp = int(s.hp[1, 1])
    out = assert_matches_reference(tables, s, (0, MOVE_ACTIONS + 1), O)
    assert int(out.pivot_battler) == 0 and int(out.active[1]) == 1
    assert int(out.ability[1, 1]) == AbilityId.MAGIC_GUARD  # never re-entered
    assert int(out.hp[1, 1]) < hp  # Pikachu's Tackle found its target


def test_sleep_talk_skips_a_sealed_slot(tables, names):
    """B7-13: BtlCmd_TrySleepTalk's CheckInvalidMoves (battle_script.c:4929) includes
    CHECK_INVALID_IMPRISONED (battle_lib.c:2287-2290): with the foe imprisoning and knowing
    Tackle, an asleep Sleep Talk user never calls Tackle, whatever its draw."""
    import jax.numpy as jnp

    from nuzline.engine.ids import Status
    from nuzline.engine.order import sleep_talk_slots

    n = names
    s = make_state(
        tables,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Ember"])],
        [_mon(n, "Geodude", ["Tackle"])],
    ).with_status(0, 0, Status.SLEEP, 3)
    s = s._replace(imprison=s.imprison.at[1].set(1))
    assert [bool(x) for x in sleep_talk_slots(tables, s, 0)] == [False, False, True, False]
    for call in range(4):
        o = O._replace(call=jnp.asarray([call, 0], jnp.int16))
        out = assert_matches_reference(tables, s, (0, 0), o)
        assert int(out.last_move[0]) == n.move("Sleep Talk")


def test_baton_pass_ingrain_grounds_a_levitate_newcomer(tables, names):
    """B7-14: subscript_hazards_check.s:8 reads CheckAbility, Battler_Ability, which hides an
    ingrained holder's Levitate (battle_lib.c:3100-3103); Baton Pass keeps the Ingrain
    (battle_lib.c:2079), so the Spikes reach the Bronzor it brings in."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Ninjask", ["Baton Pass"], 40), _mon(n, "Bronzor", ["Tackle"], 30)],
        [_mon(n, "Geodude", ["Growl"], 20)],
    )
    s = s._replace(ingrain=s.ingrain.at[0].set(1), spikes=s.spikes.at[0].set(3))
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    hp = int(paused.hp[0, 1])
    out = assert_matches_reference(tables, paused, (MOVE_ACTIONS + 1, 0), O)
    full = int(out.max_hp[0, 1])
    # The Spikes' quarter (3 layers), then the turn end's Ingrain heal (bcp.c:1311-1326).
    assert int(out.ingrain[0]) == 1 and int(out.hp[0, 1]) == hp - full // 4 + full // 16


def test_mean_look_target_keeps_its_stale_bits(tables, names):
    """B7-15: subscript_mean_look.s:10 ORs the attacker into meanLookTarget, which only the
    trapped battler's switch-in or faint zeroes: a stale target stays after the Mean Look ends."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Pikachu", ["Tackle"], 30)],
        [_mon(n, "Gastly", ["Mean Look"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    trapped = assert_matches_reference(tables, s, (0, 0), O)
    assert int(trapped.mean_look[0]) == 1 and int(trapped.mean_look_target[0]) == 2
    left = assert_matches_reference(tables, trapped, (0, MOVE_ACTIONS + 1), O)
    assert int(left.mean_look[0]) == 0 and int(left.mean_look_target[0]) == 2


def test_an_anchored_roar_target_draws_nothing(tables, names):
    """B7-16: an ingrained target with no other moveEffectsMask bit stops Roar
    (subscript_force_target_to_switch_or_flee.s:7, :86) before TryWhirlwind's draws."""
    import numpy as np

    from nuzline.engine.enumerate import outcome_laws

    n = names
    s = make_state(
        tables,
        [_mon(n, "Shinx", ["Roar"], 10)],
        [_mon(n, "Geodude", ["Growl"], 30), _mon(n, "Onix", ["Tackle"], 30)],
    )
    s = s._replace(ingrain=s.ingrain.at[1].set(1))
    laws = outcome_laws(tables, s, (0, 0))
    assert not np.any(np.asarray(laws.relevant.roar_level))
    assert not np.any(np.asarray(laws.relevant.roar_target))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.active[1]) == 0
