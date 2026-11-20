"""SF2 (f): Acid Rain at B = 2, the interim guard. A Pursuit KO of the switcher runs
``fieldConditionsMask -= faintedMon`` (subscript_pursuit.s:100, :115; script.c:3282-3287), which
the engine cannot hold: ``acid_rain`` records the operand when it is nonzero while a field
condition is up, and the host entry points refuse the state (a refusal of a mechanic not modelled
yet, so ``fullset``). Each case runs through ``assert_matches_reference``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.coverage import UnsupportedMechanics
from nuzline.engine.enumerate import enumerate_turn
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId, Stat, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from nuzline.search.learn import learn_reason
from nuzline.search.playout import finish_turn
from tests.engine.d2_states import four, with_ability, with_moves
from tests.engine.helpers import assert_matches_reference, jit_step

pytestmark = pytest.mark.fullset

O = empty_outcome()
SWITCH = MOVE_ACTIONS + 1  # switch to party slot 1


def _mon(names, species, moves, level=30, item="", ability=None) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    slot = 0
    if ability is not None:
        slot = names.species_abilities(names.species(species)).index(names.ability(ability))
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=names.item(item) if item else 0,
        ability_slot=slot,
        gender=0,
        friendship=70,
    )


def _i8(x, v):
    return jnp.asarray(v, x.dtype)


def _fight(tables, names, player, opp, weather=Weather.NONE):
    return make_state(tables, player, opp, weather)


def _refused(tables, out, acts=(0, 0)):
    assert int(out.acid_rain) != 0
    with pytest.raises(UnsupportedMechanics, match="Acid Rain"):
        enumerate_turn(tables, out, acts)


def _chaser(n, **kw):
    return _mon(n, "Houndour", ["Pursuit"], 35, **kw)


def _weak(n, species="Bulbasaur", **kw):
    return _mon(n, species, ["Tackle"], 5, **kw)


def _bench(n):
    return _mon(n, "Geodude", ["Tackle"], 30)


def test_player_pursuit_kos_the_switching_foe(tables, names):
    n = names
    s = _fight(tables, n, [_chaser(n), _bench(n)], [_weak(n), _bench(n)], Weather.HAIL)
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.acid_rain) == 1
    _refused(tables, out)
    # No field condition: operand 1 but nothing to corrupt: no guard, play goes on.
    s = _fight(tables, n, [_chaser(n), _bench(n)], [_weak(n), _bench(n)])
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.acid_rain) == 0


def test_player_pursuit_kos_the_u_turning_foe_under_gravity(tables, names):
    n = names
    foe = _mon(n, "Scyther", ["U-turn"], 5)
    s = _fight(tables, n, [_chaser(n), _bench(n)], [foe, _bench(n)])
    s = s._replace(
        gravity=_i8(s.gravity, 5),
        stats=s.stats.at[1, 0, Stat.SPE - 1].set(_i8(s.stats, 500)),
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.acid_rain) == 1
    _refused(tables, out)


def test_foe_pursuit_kos_the_switcher_without_recoil(tables, names):
    n = names
    s = _fight(tables, n, [_weak(n), _bench(n)], [_chaser(n), _bench(n)], Weather.HAIL)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.acid_rain) == 0


def test_aftermath_recoil_fells_the_foe_pursuer(tables, names):
    n = names
    switcher = _weak(n, "Drifloon", ability="Aftermath")
    s = _fight(tables, n, [switcher, _bench(n)], [_chaser(n), _bench(n)], Weather.HAIL)
    s = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.hp[1, 0]) == 0
    assert int(out.acid_rain) == 1
    _refused(tables, out)


def test_life_orb_recoil_fells_the_foe_pursuer(tables, names):
    n = names
    s = _fight(
        tables, n, [_weak(n), _bench(n)], [_chaser(n, item="Life Orb"), _bench(n)], Weather.HAIL
    )
    s = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.hp[1, 0]) == 0
    assert int(out.acid_rain) == 1
    _refused(tables, out)


def test_player_pursuer_felled_by_its_life_orb_leaves_operand_zero(tables, names):
    n = names
    s = _fight(
        tables, n, [_chaser(n, item="Life Orb"), _bench(n)], [_weak(n), _bench(n)], Weather.HAIL
    )
    s = s.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.hp[0, 0]) == 0
    assert int(out.acid_rain) == 0


def test_destiny_bond_restores_the_switcher_as_operand(tables, names):
    n = names
    switcher = _weak(n, "Carvanha", ability="Rough Skin")
    s = _fight(tables, n, [_chaser(n), _bench(n)], [switcher, _bench(n)], Weather.HAIL)
    s = s.replace_hp(0, 0, 1)
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(_i8(s.destiny_bond, 1)))
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.hp[0, 0]) == 0
    assert int(out.acid_rain) == 1  # not the pursuer (0): Destiny Bond restores faintedMon
    _refused(tables, out)


def test_destiny_bond_faint_clears_the_pursuers_uproar_bit(tables, names):
    n = names
    # Uproar is the only field condition; the bond faint clears the pursuer's bit before :115
    # (battle_lib.c:2213): the mask is 0, 0 - 1 clamps to 0.
    switcher = _weak(n, "Carvanha")
    s = _fight(tables, n, [_chaser(n), _bench(n)], [switcher, _bench(n)])
    s = s._replace(
        destiny_bond=s.destiny_bond.at[1].set(_i8(s.destiny_bond, 1)),
        uproar=s.uproar.at[0].set(_i8(s.uproar, 2)),
    )
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.hp[0, 0]) == 0  # the bond took the pursuer
    assert int(out.acid_rain) == 0


def test_recoil_faint_keeps_the_pursuers_uproar_bit(tables, names):
    n = names
    switcher = _weak(n, "Drifloon", ability="Aftermath")
    s = _fight(tables, n, [switcher, _bench(n)], [_chaser(n), _bench(n)])
    s = s.replace_hp(1, 0, 1)
    s = s._replace(uproar=s.uproar.at[1].set(_i8(s.uproar, 2)))
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.hp[1, 0]) == 0
    assert int(out.acid_rain) == 1  # a recoil faint keeps the bit: 0x200 - 1
    _refused(tables, out)


def test_jitted_step_search_and_reason_agree(tables, names):
    n = names
    s = _fight(tables, n, [_chaser(n), _bench(n)], [_weak(n), _bench(n)], Weather.HAIL)
    acts = jnp.asarray((0, SWITCH), jnp.int32)
    plain, jitted = step(tables, s, acts, O), jit_step(tables, s, acts, O)
    assert int(plain.acid_rain) == int(jitted.acid_rain) == 1
    out, refused = finish_turn(tables, jitted, False, 10)
    assert bool(refused)
    assert "Acid Rain" in learn_reason(out)
    ok = _fight(tables, n, [_chaser(n), _bench(n)], [_weak(n), _bench(n)])
    ok_out = step(tables, ok, acts, O)
    assert learn_reason(ok_out) is None


def test_search_entry_points_refuse_an_acid_rain_state(tables, names):
    """The search's host entry points refuse a state holding Acid Rain before any work:
    ``rank_actions`` (Stage A), ``evaluate_players`` (line odds) and ``find_line`` (line find)."""
    import jax

    from nuzline.search.line_search import find_line
    from nuzline.search.player_eval import evaluate_players
    from nuzline.search.stage_a import rank_actions
    from tests.engine.helpers import roark_state

    s = roark_state(tables, names)
    s = s._replace(acid_rain=_i8(s.acid_rain, 1))
    key = jax.random.PRNGKey(0)
    with pytest.raises(UnsupportedMechanics, match="Acid Rain"):
        rank_actions(tables, names, s, 1, key)
    with pytest.raises(UnsupportedMechanics, match="Acid Rain"):
        evaluate_players(tables, s, None, [()], 1, key)
    with pytest.raises(UnsupportedMechanics, match="Acid Rain"):
        find_line(tables, names, s, key)


# --- B = 4 (D3-9): the write per pass of the per-pursuer site -------------------------------


SW4 = MOVE_ACTIONS + 2


def _hail4(tables, names, chasers, **kw):
    """DOUBLES under permanent hail: battlers in ``chasers`` know Pursuit, the rest Swords Dance."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Pursuit" if b in chasers else "Swords Dance"])
    return s._replace(
        weather=_i8(s.weather, Weather.HAIL), weather_timer=_i8(s.weather_timer, 0), **kw
    )


def _play4(tables, s, acts, order, targets=(0, 0, 0, 0)):
    from tests.engine.test_d3_action_loop import play

    return play(tables, s, acts, order, targets)[0]


def _hurt(s, party, slot, hp):
    return s._replace(hp=s.hp.at[party, slot].set(hp))


def test_four_permanent_hail_ko_by_battler_0_of_the_switching_battler_3(tables, names):
    s = _hail4(tables, names, (0,))
    s = _hurt(s, 1, int(s.active[3]), 1)
    out = _play4(tables, s, [0, 0, 0, SW4], (3, 0, 1, 2))
    assert int(out.acid_rain) == 3
    _refused(tables, out)
    # the same KO with no field condition up: 0, plays on
    calm = s._replace(weather=_i8(s.weather, Weather.NONE))
    assert int(_play4(tables, calm, [0, 0, 0, SW4], (3, 0, 1, 2)).acid_rain) == 0


def test_four_uproar_alone_battler_1_ko_by_battler_2(tables, names):
    s = _hail4(tables, names, (2,))
    s = s._replace(weather=_i8(s.weather, Weather.NONE))
    s = s._replace(uproar=s.uproar.at[0].set(_i8(s.uproar, 3)))
    s = _hurt(s, 1, int(s.active[1]), 1)
    out = _play4(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3))
    assert int(out.acid_rain) == 1
    _refused(tables, out)


def test_four_ko_without_recoil_leaves_operand_zero(tables, names):
    s = _hurt(_hail4(tables, names, (1,)), 0, 0, 1)
    out = _play4(tables, s, [SW4, 0, 0, 0], (0, 1, 2, 3))
    assert int(out.hp[0, 0]) == 0 and int(out.acid_rain) == 0


def test_four_rough_skin_fells_battler_3(tables, names):
    s = with_ability(_hail4(tables, names, (3,)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(_hurt(s, 0, 0, 1), 1, int(s.active[3]), 1)
    out = _play4(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2))
    assert int(out.acid_rain) == 3
    _refused(tables, out)


def test_four_life_orb_recoil_fells_battler_2(tables, names):
    s = _hail4(tables, names, (2,))
    s = s._replace(item=s.item.at[0, int(s.active[2])].set(names.item("Life Orb")))
    s = _hurt(_hurt(s, 1, int(s.active[1]), 1), 0, int(s.active[2]), 1)
    out = _play4(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3))
    assert int(out.hp[0, int(s.active[2])]) == 0
    assert int(out.acid_rain) == 2
    _refused(tables, out)


def test_four_aftermath_fells_battler_1(tables, names):
    s = with_ability(_hail4(tables, names, (1,)), 0, AbilityId.AFTERMATH)
    s = _hurt(_hurt(s, 0, 0, 1), 1, int(s.active[1]), 1)
    out = _play4(tables, s, [SW4, 0, 0, 0], (0, 1, 2, 3))
    assert int(out.hp[1, int(s.active[1])]) == 0
    assert int(out.acid_rain) == 1
    _refused(tables, out)


def test_four_destiny_bond_restores_the_switcher_as_operand_zero(tables, names):
    """Battler 0 holding Destiny Bond (and Rough Skin) switches under hail, battler 3's Pursuit
    KOs it: battler 3 is at 0 HP by the bond, no contact ability, no reveal, ``acid_rain == 0``
    (faintedMon restored to the switcher, battler 0)."""
    s = with_ability(_hail4(tables, names, (3,)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(s, 0, 0, 1)
    s = s._replace(destiny_bond=s.destiny_bond.at[0].set(_i8(s.destiny_bond, 1)))
    out = _play4(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2))
    assert int(out.hp[1, int(s.active[3])]) == 0
    assert int(out.acid_rain) == 0


def test_four_destiny_bond_clears_the_pursuers_uproar_bit(tables, names):
    """Battler 1 (Destiny Bond) KO'd by battler 2 with ``uproar[2] != 0`` the only condition: the
    bond's faint clears battler 2's bit before :115, mask 0: plays on."""
    s = _hail4(tables, names, (2,))
    s = s._replace(weather=_i8(s.weather, Weather.NONE))
    s = s._replace(uproar=s.uproar.at[2].set(_i8(s.uproar, 3)))
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(_i8(s.destiny_bond, 1)))
    s = _hurt(s, 1, int(s.active[1]), 1)
    out = _play4(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3))
    assert int(out.acid_rain) == 0


def test_four_the_lower_battler_hits_first_whatever_the_speeds(tables, names):
    """Battler 0 (Rough Skin) switches under hail, battlers 1 (1 HP) and 3 both ready, battler 3
    faster: battler 1 hits first (index order) and falls to the Rough Skin, battler 3 never
    pursues (the KO'd chain: the switcher is dead at 1 HP): 1 (by speed it would be 3)."""
    s = with_ability(_hail4(tables, names, (1, 3)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(_hurt(s, 0, 0, 1), 1, int(s.active[1]), 1)
    out = _play4(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2))
    assert int(out.acid_rain) == 1
    _refused(tables, out)
