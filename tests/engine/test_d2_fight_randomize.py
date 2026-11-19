"""D2-9: when the fight command randomizes the target (BattleControllerPlayer_FightCommand,
bcp.c:1861-1903): ``targets.fight_randomizes`` and what the flag does to ``targets.resolve``."""

from __future__ import annotations

import pytest

from nuzline.engine import targets
from nuzline.engine.state import make_state
from tests.engine.d2_states import four, mon, with_hp
from tests.engine.test_targets import go

NONE = targets.NO_DEFENDER


def _b2(tables, names):
    return make_state(tables, [mon(names)], [mon(names)])


def _with(s, **fields):
    return s._replace(**{k: getattr(s, k).at[0].set(v) for k, v in fields.items()})


def _flag(s, struggling=False):
    return bool(targets.fight_randomizes(s, 0, struggling))


def test_fight_randomizes_branches(tables, names):
    s = _b2(tables, names)
    tackle = names.move("Tackle")
    # branch 1.1: Struggle
    assert _flag(s, True)
    # branch 1.2: encored, the slot holds the encored move
    assert int(s.moves[0, int(s.active[0]), 0]) == tackle
    assert _flag(_with(s, encore_move=tackle, encore_slot=0))
    # branch 1.3: encored, the slot no longer holds it
    assert _flag(_with(s, encore_move=names.move("Growl"), encore_slot=0))
    # branch 1.4: locked (no encore): no randomization
    locked = _with(s, locked_move=names.move("Rollout"))
    assert not _flag(locked)
    # the Encore branches come before the lock branch
    assert _flag(_with(locked, encore_move=tackle, encore_slot=0))
    # branch 1.6: plain
    assert not _flag(s)


@pytest.mark.parametrize("bit", [0, 1])
def test_b4_encored_battler_ignores_its_chosen_target(tables, names, bit):
    s = four(tables, names)
    s = s._replace(encore_move=s.encore_move.at[0].set(names.move("Tackle")))
    assert _flag(s)
    # chosen = a foe: both coin values give the two living foes
    got = go(tables, names, s, 0, "Tackle", chosen=3, bit=bit, randomize=_flag(s))
    assert got == ((3, 3, 0, True) if bit == 0 else (1, 1, 0, True))
    # a foe down: the other one, no draw
    one = with_hp(s, 3, 0)
    assert go(tables, names, one, 0, "Tackle", chosen=3, bit=bit, randomize=True) == (
        1,
        1,
        0,
        False,
    )


@pytest.mark.parametrize("bit", [0, 1])
def test_b4_struggling_battler_ignores_its_chosen_target(tables, names, bit):
    s = four(tables, names)
    got = go(tables, names, s, 0, "Tackle", chosen=1, bit=bit, randomize=_flag(s, True))
    assert got == ((3, 3, 0, True) if bit == 0 else (1, 1, 0, True))


def test_b4_locked_rollout_hits_its_chosen_foe_without_a_draw(tables, names):
    s = four(tables, names)
    s = s._replace(locked_move=s.locked_move.at[0].set(names.move("Rollout")))
    assert not _flag(s)
    for chosen in (1, 3):
        for bit in (0, 1):
            got = go(tables, names, s, 0, "Rollout", chosen=chosen, bit=bit, randomize=False)
            assert got == (chosen, chosen, 0, False)


@pytest.mark.parametrize("bit", [0, 1])
def test_b4_locked_battler_whose_chosen_target_fainted_retargets(tables, names, bit):
    s = four(tables, names)
    s = s._replace(locked_move=s.locked_move.at[0].set(names.move("Rollout")))
    dead = with_hp(s, 3, 0)
    got = go(tables, names, dead, 0, "Rollout", chosen=3, bit=bit, randomize=False)
    assert got == (1, 1, 0, True)  # RandomOpponent steps to the living foe


def test_b2_locked_rollout_reads_its_stored_user_unless_randomized(tables, names):
    # singles follow-up (ii), ruling D2-33: the lock branch does not randomize (bcp.c:1885-1888),
    # so branch 10 reads the stored target (lib.c:1711, :1718-1719): the user after a RANGE_MY_SIDE
    # selection; an Encore (the randomizing branches) takes the foe
    s = _b2(tables, names)
    s = _with(s, locked_move=names.move("Rollout"))
    assert not _flag(s)
    assert go(tables, names, s, 0, "Rollout", chosen=0, randomize=False) == (0, 0, 0, False)
    assert go(tables, names, s, 0, "Rollout", chosen=1, randomize=False) == (1, 1, 0, False)
    encored = _with(s, encore_move=names.move("Rollout"), encore_slot=0)
    assert _flag(encored)
    assert go(tables, names, encored, 0, "Rollout", chosen=0, randomize=True) == (1, 1, 0, False)
