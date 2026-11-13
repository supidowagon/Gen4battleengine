"""Light Screen, Reflect and Brick Break (survey section 4.4).

* Light Screen / Reflect: ``BtlCmd_TryLightScreen`` / ``TryReflect`` (battle_script.c:4214,
  :4257) fail when the user's side has the screen, else set ``NUM_SCREEN_TURNS`` (5); the
  end-of-turn pass (battle_controller_player.c:975) then counts it down once the same turn.
* The screens halve non-crit damage (battle_lib.c:6982, :7023) unless the move is Brick Break.
* Brick Break: a direct BREAK_SCREENS effect (lib.c:1481-1510) clearing both screens of the
  defender's side after the hit, even on an immune target, but not on a miss.
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.damage import _pre_random, damage_table
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import A, assert_matches_reference, roark_state


def _put(s, field, side, value):
    return s._replace(**{field: getattr(s, field).at[side].set(value)})


def _miss():
    return empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8))


@pytest.mark.slow
def test_light_screen_sets_five_turns_and_fails_when_up(tables, names):
    s = roark_state(tables, names, player_moves=["Light Screen"])
    acts = A("Light Screen", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    # 5 turns set, counted down once by the end-of-turn pass of the same turn.
    assert int(out.light_screen[0]) == 4 and int(out.light_screen[1]) == 0
    # Already up: the move fails and the timer is not refreshed (only the countdown runs).
    up = _put(s, "light_screen", 0, 2)
    out = assert_matches_reference(tables, up, acts, empty_outcome())
    assert int(out.light_screen[0]) == 1


def test_reflect_sets_five_turns_and_fails_when_up(tables, names):
    s = roark_state(tables, names, player_moves=["Reflect"])
    acts = A("Reflect", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.reflect[0]) == 4 and int(out.reflect[1]) == 0
    out = assert_matches_reference(tables, _put(s, "reflect", 0, 2), acts, empty_outcome())
    assert int(out.reflect[0]) == 1


def test_light_screen_halves_special_noncrit(tables, names):
    s = roark_state(tables, names, player_moves=["Ember"])
    ember = names.move("Ember")
    plain = damage_table(tables, s, 0, ember, False).tolist()
    screened = damage_table(tables, _put(s, "light_screen", 1, 3), 0, ember, False).tolist()
    assert all(b < a for a, b in zip(plain, screened, strict=True))
    # Exactly the decomp's `damage /= 2` (lib.c:7030), before the `+ 2` (:7078).
    base = int(_pre_random(tables, s, 0, ember, False)[0]) - 2
    halved = int(_pre_random(tables, _put(s, "light_screen", 1, 3), 0, ember, False)[0]) - 2
    assert halved == base // 2
    # A crit ignores it (lib.c:7023 criticalMul == 1) and Reflect does not touch special.
    assert damage_table(tables, _put(s, "light_screen", 1, 3), 0, ember, True).tolist() == (
        damage_table(tables, s, 0, ember, True).tolist()
    )
    assert damage_table(tables, _put(s, "reflect", 1, 3), 0, ember, False).tolist() == plain


def test_brick_break_ignores_and_breaks_screens(tables, names):
    s = roark_state(tables, names, player_moves=["Brick Break"])
    brick = names.move("Brick Break")
    both = _put(_put(s, "light_screen", 1, 3), "reflect", 1, 4)
    assert damage_table(tables, both, 0, brick, False).tolist() == (
        damage_table(tables, s, 0, brick, False).tolist()
    )
    out = assert_matches_reference(
        tables, both, A("Brick Break", "Leer", both, names), empty_outcome()
    )
    assert int(out.light_screen[1]) == 0 and int(out.reflect[1]) == 0
    assert int(out.hp[1, 0]) < int(both.hp[1, 0])
    # The user's own side is untouched.
    mine = _put(both, "reflect", 0, 4)
    out = assert_matches_reference(
        tables, mine, A("Brick Break", "Leer", mine, names), empty_outcome()
    )
    assert int(out.reflect[0]) == 3  # the end-of-turn countdown only


def test_brick_break_breaks_on_immune_not_on_miss(tables, names):
    # Gastly (Ghost) is immune to Fighting: lib.c:1497-1499 still applies the break.
    s = roark_state(tables, names, player_moves=["Brick Break"], opp_species="Gastly")
    both = _put(_put(s, "light_screen", 1, 3), "reflect", 1, 4)
    acts = A("Brick Break", "Leer", both, names)
    out = assert_matches_reference(tables, both, acts, empty_outcome())
    assert int(out.hp[1, 0]) == int(both.hp[1, 0])  # immune: no damage
    assert int(out.light_screen[1]) == 0 and int(out.reflect[1]) == 0
    # A miss does not break them (only the end-of-turn countdown runs).
    out = assert_matches_reference(tables, both, acts, _miss())
    assert int(out.light_screen[1]) == 2 and int(out.reflect[1]) == 3


def test_brick_break_clears_screens_when_it_kos(tables, names):
    # Primary effects run before UpdateHP (lib.c:1494), so the break survives the KO.
    s = roark_state(tables, names, player_moves=["Brick Break"]).replace_hp(1, 0, 1)
    both = _put(_put(s, "light_screen", 1, 3), "reflect", 1, 4)
    out = assert_matches_reference(
        tables, both, A("Brick Break", "Leer", both, names), empty_outcome()
    )
    assert int(out.hp[1, 0]) == 0
    assert int(out.light_screen[1]) == 0 and int(out.reflect[1]) == 0


def test_screens_match_reference(tables, names):
    s = roark_state(tables, names, player_moves=["Light Screen", "Reflect", "Brick Break", "Ember"],
                    opp_moves=["Light Screen", "Reflect"])  # fmt: skip
    both = _put(_put(s, "light_screen", 1, 3), "reflect", 1, 4)
    for state in (s, both):
        for p in ("Light Screen", "Reflect", "Brick Break", "Ember"):
            for o in ("Light Screen", "Reflect", "Tackle"):
                for outcome in (empty_outcome(), _miss()):
                    assert_matches_reference(tables, state, A(p, o, state, names), outcome)
