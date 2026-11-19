"""SF2 (h): Destiny Bond before the contact ability in Pursuit's KO path (B = 2).

subscript_pursuit.s runs FAINT_CHECK_DESTINY_BOND (:85) before TriggerAbilityOnHit (:86-87), so a
KO'd switcher that holds Destiny Bond takes the pursuer down first and every contact ability then
fails on the attacker's HP (battle_lib.c:4215-4363): no message, no reveal, no draw. Each case runs
through ``assert_matches_reference``. SF2-roost: a lock that hits its own user after the foe
roosted reads the user's own types (BasicTypeMulApplies tests turnFlags[defender].roosting,
battle_lib.c:2540-2543)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine import enumerate as enum_mod
from nuzline.engine import outcome as outcome_mod
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import _paused_rock, enumerate_turn
from nuzline.engine.ids import Status, Weather
from nuzline.engine.outcome import MAX_HITS, empty_outcome, pivot_view
from nuzline.engine.state import make_state
from nuzline.engine.step import step_events
from tests.engine.helpers import assert_matches_reference, jit_step, step_rows

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


def _chaser(n):
    return _mon(n, "Houndour", ["Pursuit"], 35)


def _weak(n, species, ability):
    return _mon(n, species, ["Tackle"], 5, ability=ability)


def _bench(n):
    return _mon(n, "Geodude", ["Tackle"], 30)


def _bond(s, side):
    return s._replace(destiny_bond=s.destiny_bond.at[side].set(_i8(s.destiny_bond, 1)))


def _proc(v):
    return O._replace(contact_proc=jnp.full((2, MAX_HITS), v, jnp.int8))


def _events(tables, s, acts, o):
    _, events = jax.jit(step_events)(tables, s, jnp.asarray(acts), o)
    return {k: np.asarray(v) for k, v in events.items()}


def _player_pursuit(tables, n, ability, bond, pursuer_hp=None, species="Carvanha"):
    s = make_state(
        tables, [_chaser(n), _bench(n)], [_weak(n, species, ability), _bench(n)], Weather.NONE
    )
    if pursuer_hp is not None:
        s = s.replace_hp(0, 0, pursuer_hp)
    return _bond(s, 1) if bond else s


@pytest.mark.parametrize("pursuer_hp", [None, 1])
def test_bond_takes_the_pursuer_before_rough_skin(tables, names, pursuer_hp):
    n = names
    s = _player_pursuit(tables, n, "Rough Skin", True, pursuer_hp)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[0, 0]) == 0
    assert int(out.hp[1, 0]) == 0
    ev = _events(tables, s, (0, SWITCH), _proc(1))
    assert int(ev["contact_ability"][1]) == 0


@pytest.mark.parametrize("pursuer_hp", [None, 1])
def test_bond_leaves_the_players_switcher_ability_unrevealed(tables, names, pursuer_hp):
    # The AI tracks only the player's Pokemon: mirror the case with the foe chasing.
    n = names
    s = make_state(tables, [_weak(n, "Carvanha", "Rough Skin"), _bench(n)], [_chaser(n), _bench(n)])
    s = _bond(s, 0)
    if pursuer_hp is not None:
        s = s.replace_hp(1, 0, pursuer_hp)
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.hp[1, 0]) == 0
    assert int(out.ai_ability_known[0]) == 0
    # Without the bond Rough Skin reveals itself (test 4's contrast).
    s = make_state(tables, [_weak(n, "Carvanha", "Rough Skin"), _bench(n)], [_chaser(n), _bench(n)])
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.ai_ability_known[0]) != 0


def test_static_switcher_with_bond_paralyzes_nobody_for_either_draw(tables, names):
    n = names
    s = _player_pursuit(tables, n, "Static", True, species="Pikachu")
    outs = [assert_matches_reference(tables, s, (0, SWITCH), _proc(v)) for v in (0, 1)]
    for out in outs:
        assert int(out.hp[0, 0]) == 0
        assert int(out.status[0, 0]) == Status.NONE
    for x, y in zip(outs[0], outs[1], strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # The enumerator's rows all play to one state (the draw is merged away).
    rows, _ = enumerate_turn(tables, s, (0, SWITCH))
    states = set()
    for i in range(len(rows.hit)):
        row = jax.tree.map(lambda f, i=i: f[i], rows)
        out = jit_step(tables, s, jnp.asarray((0, SWITCH)), row)
        states.add(tuple(np.asarray(f).tobytes() for f in out))
    assert len(states) == 1


def test_rough_skin_switcher_without_bond_acts_as_before(tables, names):
    n = names
    s = _player_pursuit(tables, n, "Rough Skin", False)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[1, 0]) == 0
    # Rough Skin: maxHP / 8, at least 1, off the pursuer (battle_lib.c:4257-4266, :3599-3620). The ``contact_ability``
    # event shows only a status or infatuation change (step._hit_events), not this damage.
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(1, int(s.max_hp[0, 0]) // 8)


def test_surviving_bond_switcher_gets_no_take_down(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Pursuit"], 5), _bench(n)],
        [_mon(n, "Sharpedo", ["Tackle"], 60, ability="Rough Skin"), _bench(n)],
    )
    s = _bond(s, 1)
    out = assert_matches_reference(tables, s, (0, SWITCH), _proc(1))
    assert int(out.hp[1, 0]) > 0
    # Rough Skin ran as today: maxHP / 8, at least 1, off the pursuer (battle_lib.c:4257-4266, :3599-3620).
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(1, int(s.max_hp[0, 0]) // 8)


# SF2 (c): the paused view counts a pursuer as acted (battle_script.c:6931; battle_lib.c:5365-5368).


def _paused_u_turn(tables, n, foe_moves):
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 25, "King’s Rock"), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", foe_moves, 35)],
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    return paused


def test_pursuer_is_moved_in_the_paused_view(tables, names):
    paused = _paused_u_turn(tables, names, ["Pursuit"])
    assert int(paused.ts_pursued) == 2 and int(paused.ts_acted[1]) == 0
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    _, viewed = pivot_view(tables, paused, acts)
    assert int(viewed[1]) == MOVE_ACTIONS + int(paused.active[1])
    assert not np.any(np.asarray(_paused_rock(tables, paused)))


def test_paused_view_without_pursuit_keeps_the_foes_action(tables, names):
    paused = _paused_u_turn(tables, names, ["Tackle"])
    assert int(paused.ts_pursued) == 0
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    _, viewed = pivot_view(tables, paused, acts)
    assert int(viewed[1]) == int(paused.ts_actions[1]) == 0


def _merged(tables, paused, acts):
    outcomes, probs = enumerate_turn(tables, paused, acts)
    after = jax.tree.map(np.asarray, step_rows(tables, paused, acts, outcomes))
    merged = {}
    for i in range(len(probs)):
        key = tuple(np.asarray(f)[i].tobytes() for f in after)
        merged[key] = merged.get(key, 0.0) + float(probs[i])
    return merged


def test_enumeration_after_a_pursuit_is_the_old_views_distribution(tables, names, monkeypatch):
    paused = _paused_u_turn(tables, names, ["Pursuit"])
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    new = _merged(tables, paused, acts)
    old_mask = lambda s: s.ts_acted != 0
    monkeypatch.setattr(outcome_mod, "moved_mask", old_mask)
    monkeypatch.setattr(enum_mod, "moved_mask", old_mask)
    enum_mod._jit_laws.clear_cache()
    enum_mod._jit_call_laws.clear_cache()
    try:
        old = _merged(tables, paused, acts)
    finally:
        monkeypatch.undo()
        enum_mod._jit_laws.clear_cache()
        enum_mod._jit_call_laws.clear_cache()
    assert new.keys() == old.keys()
    for k in new:
        assert new[k] == pytest.approx(old[k], abs=1e-9)


# SF2 (j): the paused view counts a fainted foe as done (laws only; subscript_show_party_list.s:22-24,
# subscript_faint_mon.s:7, battle_script.c:1801).


def _ko_pause(tables, n, mirror, ko=True):
    rock = _mon(n, "Scyther", ["U-turn"], 25, "King’s Rock")
    bench = _mon(n, "Geodude", ["Tackle"], 30)
    victim = _mon(n, "Rattata", ["Tackle"], 3) if ko else _mon(n, "Houndour", ["Tackle"], 35)
    mine, theirs = ([victim, bench], [rock, bench]) if mirror else ([rock, bench], [victim, bench])
    s = make_state(tables, mine, theirs)
    pivot = 1 if mirror else 0
    acts = (0, 0)
    # The foe's faster U-turn: raise its speed order by making the player's Pokemon slow.
    paused = assert_matches_reference(tables, s, acts, O)
    assert int(paused.pivot_battler) == pivot + 1
    return paused, pivot


def _pick(pivot):
    acts = [0, 0]
    acts[pivot] = MOVE_ACTIONS + 1
    return jnp.asarray(acts, jnp.int32)


def _old_merged(tables, paused, acts, monkeypatch):
    old_mask = lambda s: (s.ts_acted != 0) | (
        ((s.ts_pursued.astype(jnp.int32) >> jnp.arange(2)) & 1) != 0
    )
    monkeypatch.setattr(outcome_mod, "moved_mask", old_mask)
    monkeypatch.setattr(enum_mod, "moved_mask", old_mask)
    enum_mod._jit_laws.clear_cache()
    enum_mod._jit_call_laws.clear_cache()
    try:
        return _merged(tables, paused, acts), len(enumerate_turn(tables, paused, acts)[1])
    finally:
        monkeypatch.undo()
        enum_mod._jit_laws.clear_cache()
        enum_mod._jit_call_laws.clear_cache()


@pytest.mark.parametrize("mirror", [False, True])
def test_fainted_foe_is_done_in_the_paused_view(tables, names, monkeypatch, mirror):
    from tests.engine.test_enumerate import assert_equals_bruteforce

    paused, pivot = _ko_pause(tables, names, mirror)
    other = 1 - pivot
    assert int(paused.hp[other, int(paused.active[other])]) == 0
    assert int(paused.ts_acted[other]) == 0
    acts = _pick(pivot)
    _, viewed = pivot_view(tables, paused, acts)
    assert int(viewed[other]) == MOVE_ACTIONS + int(paused.active[other])
    assert not np.any(np.asarray(_paused_rock(tables, paused)))
    assert_equals_bruteforce(tables, paused, acts)
    new = _merged(tables, paused, acts)
    n_new = len(enumerate_turn(tables, paused, acts)[1])
    old, n_old = _old_merged(tables, paused, acts, monkeypatch)
    assert new.keys() == old.keys()
    for k in new:
        assert new[k] == pytest.approx(old[k], abs=1e-9)
    assert n_new < n_old


@pytest.mark.parametrize("mirror", [False, True])
def test_surviving_foe_keeps_its_action_in_the_paused_view(tables, names, mirror):
    paused, pivot = _ko_pause(tables, names, mirror, ko=False)
    other = 1 - pivot
    assert int(paused.hp[other, int(paused.active[other])]) > 0
    _, viewed = pivot_view(tables, paused, _pick(pivot))
    assert int(viewed[other]) == int(paused.ts_actions[other]) == 0
    assert bool(np.asarray(_paused_rock(tables, paused))[pivot])


@pytest.mark.parametrize(("user", "immune"), [("Pidgey", True), ("Sandshrew", False)])
def test_dig_strike_at_its_user_after_a_foe_roost_reads_the_users_types(
    tables, names, user, immune
):
    # turn 1: Dig (1 PP) is encored over a Swords Dance selection (the user is stored,
    # bcp.c:4363-4367) and charges; the Encore ends at the turn's end (no PP). Turn 2: the faster
    # Flying foe roosts (subscript_roost.s:12 sets its turnFlags.roosting), then the strike reads
    # the stored user. The chart tests the defender's flag (battle_lib.c:2540-2543): the user's,
    # never set, so a Flying user keeps its Ground immunity and the foe's Roost changes nothing.
    s = make_state(
        tables,
        [_mon(names, user, ["Swords Dance", "Dig"], 30)],
        [_mon(names, "Starly", ["Encore", "Roost"], 80)],
    )
    s = s._replace(last_move=s.last_move.at[0].set(names.move("Dig")))
    s = s._replace(pp=s.pp.at[0, 0, 1].set(1))
    s = s._replace(hp=s.hp.at[1, 0].set(int(s.max_hp[1, 0]) // 2))
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.locked_move[0]) == names.move("Dig")
    assert int(one.chosen_target[0]) == 0
    assert int(one.encore_move[0]) == 0
    two = assert_matches_reference(tables, one, (0, 1), O)
    assert int(two.hp[1, 0]) > int(one.hp[1, 0])  # the Roost healed
    assert (int(two.hp[0, 0]) == int(one.hp[0, 0])) == immune
    assert int(two.locked_move[0]) == 0
