"""The end-of-turn order when the weather lock changes during the turn (K3b2 final review
C1).

Decomp (external/pokeplatinum):

* ``BattleSystem_CompareBattlerSpeed`` doubles a Swift Swim holder's speed in rain only
  ``if (NO_CLOUD_NINE)`` (src/battle/battle_lib.c:1228-1237);
* ``NO_CLOUD_NINE`` is ``BattleSystem_CountAbility(..., COUNT_ALIVE_BATTLERS, ...)`` of
  Cloud Nine and of Air Lock both FALSE (include/battle/common.h:63-64), and
  ``COUNT_ALIVE_BATTLERS`` counts the battlers with that ability and ``curHP != 0``
  (battle_lib.c:2951-2957);
* the end-of-turn order is sorted once, after the last action
  (``BattleSystem_SortMonSpeedOrder``, battle_controller_player.c:887, battle_lib.c:3364).

So the lock that decides Swift Swim at the end of the turn is the one on the field after
the actions. A Cloud Nine holder that switches in, is dragged in, is dragged out, switches
out or faints changes it. Each case below ties the two end speeds only through that
change, so ``end_speed_tie`` must be relevant; the end order decides which poisoned
1-HP Pokemon faints first. Each compares the gated enumeration against the brute force
(``loose``: ``end_speed_tie`` always relevant).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import WEATHER_ABILITY_IDS, AbilityId, Stat, Status, Weather
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import step
from tests.engine.helpers import roark_state
from tests.engine.test_enumerate import assert_equals_bruteforce, laws_of

X = 40  # the Swift Swim holder's Spe stat: 2X in rain without a lock


def _cloud_nine_slot(tables, names) -> tuple[int, int]:
    """Psyduck's species id and its Cloud Nine ability slot (``decomp_abilities``)."""
    psyduck = names.species("Psyduck")
    row = int(np.asarray(tables.species_form_row)[psyduck, 0])
    abilities = np.asarray(tables.species_abilities_decomp)[row]
    return psyduck, int(np.flatnonzero(abilities == WEATHER_ABILITY_IDS["Cloud Nine"])[0])


def _setup(tables, names, case):
    """``(state, actions)``. The player: one Swift Swim Pokemon at Spe X, poisoned, 1 HP.
    The opponent: a lead in slot 0 and a bench Pokemon in slot 1; permanent rain."""
    move = {"switch_in": "Growl", "switch_out": "Growl", "faint": "Tackle"}.get(case, "Roar")
    s = roark_state(tables, names, player_count=1, opp_count=2, player_moves=(move,))
    a = {n: np.asarray(getattr(s, n)).copy() for n in s._fields}
    psyduck, slot = _cloud_nine_slot(tables, names)
    a["ability"][0, 0] = AbilityId.SWIFT_SWIM
    a["stats"][0, 0, Stat.SPE - 1] = X
    a["status"][0, 0], a["hp"][0, 0] = Status.POISON, 1
    a["level"][0, 0] = 50  # Roar's level check passes without a draw
    a["level"][1, :2] = 5
    a["ability"][1, :2] = AbilityId.NONE
    a["stealth_rock"][:] = 0
    a["weather"][...], a["weather_timer"][...] = Weather.RAIN, 0
    # (Psyduck's slot, its Spe, the other slot's Spe, the slot on the field at the end):
    lock, lock_spe, other_spe, end = {
        "switch_in": (1, X, 3 * X, 1),  # the opponent switches Psyduck in: X vs X
        "drag_in": (1, X, 3 * X, 1),  # the player's Roar drags Psyduck in: X vs X
        "drag_out": (0, 3 * X, 2 * X, 1),  # the Roar drags Psyduck out: 2X vs 2X
        "switch_out": (0, 3 * X, 2 * X, 1),  # the opponent switches Psyduck out: 2X vs 2X
        "faint": (0, 2 * X, 3 * X, 0),  # the player's Tackle faints Psyduck: 2X vs 2X
    }[case]
    a["species"][1, lock], a["form"][1, lock] = psyduck, 0
    a["ability_slot"][1, lock], a["ability_slot"][1, 1 - lock] = slot, 0
    a["stats"][1, lock, Stat.SPE - 1] = lock_spe
    a["stats"][1, 1 - lock, Stat.SPE - 1] = other_spe
    # The opponent's Pokemon on the field at the end: poisoned at 1 HP (the faint case:
    # Psyduck at 1 HP, fainted by the Tackle, so no end-of-turn effect of its own).
    a["hp"][1, end] = 1
    a["status"][1, end] = Status.NONE if case == "faint" else Status.POISON
    s = s._replace(**{n: jnp.asarray(a[n]) for n in s._fields})
    switch = case in ("switch_in", "switch_out")
    opp = MOVE_ACTIONS + 1 if switch else names.move_slot(s, 1, "Leer")
    return s, jnp.asarray([names.move_slot(s, 0, move), opp], jnp.int32)


CASES = ("switch_in", "drag_in", "drag_out", "switch_out", "faint")


@pytest.mark.parametrize("case", CASES)
def test_end_speed_tie_is_relevant(tables, names, case):
    s, acts = _setup(tables, names, case)
    assert bool(laws_of(tables, s, acts).relevant.end_speed_tie)


@pytest.mark.parametrize("case", ("switch_in", "drag_in", "drag_out", "switch_out"))
def test_the_end_order_decides_the_state(tables, names, case):
    # The two end_speed_tie values step to different states: the tie is real and read.
    s, acts = _setup(tables, names, case)
    out = empty_outcome()
    after = [step(tables, s, acts, out._replace(end_speed_tie=jnp.int8(t))) for t in (0, 1)]
    assert any(
        not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(*after, strict=True)
    )


@pytest.mark.parametrize("case", CASES)
def test_gated_equals_bruteforce(tables, names, case):
    s, acts = _setup(tables, names, case)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width <= brute


def test_no_lock_holder_no_widening(tables, names):
    # Without a living Cloud Nine / Air Lock holder in either party the lock cannot change:
    # the drag-in case with Psyduck's other ability is not relevant (speeds X vs 2X).
    s, acts = _setup(tables, names, "drag_in")
    s = s._replace(ability_slot=s.ability_slot.at[1, 1].set(1 - int(s.ability_slot[1, 1])))
    assert not bool(laws_of(tables, s, acts).relevant.end_speed_tie)
