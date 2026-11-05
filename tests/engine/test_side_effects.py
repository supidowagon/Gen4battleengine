"""Confusion (``BATTLE_EFFECT_CONFUSE_HIT``), Mud Sport and Bug Bite (``EAT_BERRY``).

* ``effect_script_0076.s`` + ``subscript_confuse.s``: a probabilistic indirect effect that
  confuses a living, unconfused target for ``Random 3, 2`` turns (``outcome.duration``).
* ``effect_script_0201.s``: Mud Sport sets ``MOVE_EFFECT_MUD_SPORT`` on the user (fails
  when set); ``BattleSystem_AnyBattlersWithMoveEffect`` (battle_lib.c:6812) halves Electric
  power while any active battler has it; a switch-out clears it.
* ``effect_script_0224.s`` + ``subscript_pluck.s`` + ``BattleSystem_PluckBerry``
  (battle_lib.c:5539): the user eats the target's Oran / Sitrus Berry, healing only when
  not at full HP; ``RemoveItem`` takes the berry whatever the heal did.
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import damage_table
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state

PROC = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))
DURATION_3 = PROC._replace(duration=jnp.array([3, 3], jnp.int8))


@pytest.fixture
def venonat(tables, names):
    """Venonat's Confusion against Geodude, which moves first and only uses Leer."""
    return roark_state(
        tables,
        names,
        player_species="Venonat",
        player_moves=("Confusion",),
        opp_species="Geodude",
        opp_faster=True,
    )


def confuse(tables, names, s, outcome):
    return step(tables, s, A("Confusion", "Leer", s, names), outcome)


def test_confusion_confuses_on_proc(tables, names, venonat):
    assert int(confuse(tables, names, venonat, DURATION_3).confusion[1]) == 3
    # No proc: Confusion deals its damage only.
    out = confuse(
        tables, names, venonat, DURATION_3._replace(secondary=jnp.zeros((2, 5), jnp.int8))
    )
    assert int(out.confusion[1]) == 0 and int(out.hp[1, 0]) < int(venonat.hp[1, 0])
    # An already confused target keeps its counter (4 drops to 3 as Geodude moves).
    s = venonat._replace(confusion=venonat.confusion.at[1].set(4))
    kept = confuse(tables, names, s, DURATION_3._replace(duration=jnp.array([5, 5], jnp.int8)))
    assert int(kept.confusion[1]) == 3
    # A KO'd target gets none (the roll is cleared for a fainted battler).
    ko = confuse(tables, names, venonat.replace_hp(1, 0, 1), DURATION_3)
    assert int(ko.hp[1, 0]) == 0 and int(ko.confusion[1]) == 0


def test_mud_sport_halves_electric_until_switch(tables, names):
    s = roark_state(
        tables, names, player_species="Geodude", player_moves=("Mud Sport",), opp_species="Monferno"
    )
    acts = A("Mud Sport", "Leer", s, names)
    once = step(tables, s, acts, empty_outcome())
    assert int(once.mud_sport[0]) == 1 and int(once.mud_sport[1]) == 0
    assert int(once.hp[1, 0]) == int(s.hp[1, 0])
    # A second use fails: the flag and every other effect field are unchanged (turn
    # bookkeeping, Leer's stage drop and the AI's memory aside), only the PP drops.
    twice = step(tables, once, A("Mud Sport", "Leer", once, names), empty_outcome())
    for name in ("mud_sport", "hp", "status", "confusion", "item", "water_sport", "foresight"):
        assert (getattr(twice, name) == getattr(once, name)).all(), name
    assert int(twice.pp[0, 0, 0]) == int(once.pp[0, 0, 0]) - 1

    # Shinx's Spark on Monferno: halved while either side has the flag.
    s = roark_state(
        tables, names, player_species="Shinx", player_moves=("Spark",), opp_species="Monferno"
    )
    spark = names.move("Spark")
    full = damage_table(tables, s, 0, spark, False)
    assert int(full[-1]) > 0
    for side in (0, 1):
        sport = s._replace(mud_sport=s.mud_sport.at[side].set(1))
        halved = damage_table(tables, sport, 0, spark, False)
        assert 0 < int(halved[-1]) < int(full[-1])
    # The flag holder switches out (to Cranidos): the flag is gone and Spark is full again.
    sport = s._replace(mud_sport=s.mud_sport.at[1].set(1))
    out = step(tables, sport, jnp.array([names.move_slot(sport, 0, "Spark"), MOVE_ACTIONS + 2]),
               empty_outcome())  # fmt: skip
    assert int(out.mud_sport[1]) == 0
    fresh = damage_table(tables, out, 0, spark, False)
    flagged = out._replace(mud_sport=out.mud_sport.at[0].set(1))
    assert int(fresh[-1]) > int(damage_table(tables, flagged, 0, spark, False)[-1]) > 0


@pytest.fixture
def kricketot(tables, names):
    """Kricketot's Bug Bite against Geodude holding an Oran Berry; the user is 10 HP
    below max."""
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_moves=("Bug Bite",),
        opp_species="Geodude",
        opp_faster=True,
    )
    oran = names.item("Oran Berry")
    s = s._replace(item=s.item.at[1, 0].set(oran))
    return s.replace_hp(0, 0, int(s.max_hp[0, 0]) - 10)


def bite(tables, names, s):
    return step(tables, s, A("Bug Bite", "Leer", s, names), empty_outcome())


def test_bug_bite_eats_berry(tables, names, kricketot):
    oran, sitrus = names.item("Oran Berry"), names.item("Sitrus Berry")
    s = kricketot
    out = bite(tables, names, s)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) + 10
    assert int(out.item[1, 0]) == 0 and int(out.item_consumed[1, 0]) == oran
    # Capped at max HP.
    near = s.replace_hp(0, 0, int(s.max_hp[0, 0]) - 4)
    assert int(bite(tables, names, near).hp[0, 0]) == int(near.max_hp[0, 0])
    # Sitrus heals a quarter of max HP.
    sit = s._replace(item=s.item.at[1, 0].set(sitrus)).replace_hp(0, 0, 1)
    healed = bite(tables, names, sit)
    assert int(healed.hp[0, 0]) == 1 + int(sit.max_hp[0, 0]) * 25 // 100
    assert int(healed.item[1, 0]) == 0 and int(healed.item_consumed[1, 0]) == sitrus
    # At full HP nothing is healed and the berry is still gone.
    full = s.replace_hp(0, 0, int(s.max_hp[0, 0]))
    out = bite(tables, names, full)
    assert int(out.hp[0, 0]) == int(full.max_hp[0, 0])
    assert int(out.item[1, 0]) == 0 and int(out.item_consumed[1, 0]) == oran
    # A KO'ing Bug Bite still eats it.
    # (Kricketot moves first, so Geodude's own Oran Berry check has not run yet.)
    fast = s._replace(stats=s.stats.at[0, 0, 4].set(500)).replace_hp(1, 0, 1)
    out = bite(tables, names, fast)
    assert int(out.hp[1, 0]) == 0 and int(out.item_consumed[1, 0]) == oran
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) + 10
    # A target with no item is untouched.
    bare = s._replace(item=s.item.at[1, 0].set(0))
    out = bite(tables, names, bare)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.item_consumed[1, 0]) == 0


def test_side_effects_match_reference(tables, names, venonat, kricketot):
    for outcome in (DURATION_3, empty_outcome()):
        assert_matches_reference(tables, venonat, A("Confusion", "Leer", venonat, names), outcome)
        ko = venonat.replace_hp(1, 0, 1)
        assert_matches_reference(tables, ko, A("Confusion", "Leer", ko, names), outcome)
    confused = venonat._replace(confusion=venonat.confusion.at[1].set(4))
    assert_matches_reference(tables, confused, A("Confusion", "Leer", confused, names), DURATION_3)

    mud = roark_state(
        tables, names, player_species="Geodude", player_moves=("Mud Sport",), opp_species="Monferno"
    )
    once = assert_matches_reference(tables, mud, A("Mud Sport", "Leer", mud, names), PROC)
    assert_matches_reference(tables, once, A("Mud Sport", "Leer", once, names), PROC)

    sitrus = names.item("Sitrus Berry")
    cases = [
        kricketot,
        kricketot.replace_hp(0, 0, int(kricketot.max_hp[0, 0])),
        kricketot.replace_hp(1, 0, 1),
        kricketot._replace(item=kricketot.item.at[1, 0].set(sitrus)),
        kricketot._replace(item=kricketot.item.at[1, 0].set(0)),
    ]
    for s in cases:
        assert_matches_reference(tables, s, A("Bug Bite", "Leer", s, names), empty_outcome())
