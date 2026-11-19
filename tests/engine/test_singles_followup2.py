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
from nuzline.engine.ids import AbilityId, Status, Weather
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


# SF2 (g): the `quick_claw` event is a bit field: bit 0 the held-item animation, bit 1 Custap's
# "let it move first!" message (subscript_check_quick_claw.s:12-19; the foe has not acted yet).


def _held_duel(tables, n, item, holder_hp, foe_moves, holder_level=40, foe_level=5):
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Tackle"], holder_level, item)],
        [_mon(n, "Rattata", foe_moves, foe_level)],
    )
    return s if holder_hp is None else s.replace_hp(0, 0, holder_hp)


def _claw_event(tables, s, acts=(0, 0)):
    return int(_events(tables, s, acts, O)["quick_claw"][0])


def test_custap_holder_acting_first_shows_animation_and_message(tables, names):
    s = _held_duel(tables, names, "Custap Berry", 20, ["Tackle"])
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.item[0, 0]) == 0
    assert _claw_event(tables, s) == 3


def test_custap_holder_acting_second_shows_animation_only(tables, names):
    s = _held_duel(tables, names, "Custap Berry", 20, ["Quick Attack"])
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.item[0, 0]) == 0
    assert _claw_event(tables, s) == 1


def test_quick_claw_holder_first_shows_the_animation(tables, names):
    s = _held_duel(tables, names, "Quick Claw", None, ["Tackle"])
    s = s._replace(qc_roll=s.qc_roll.at[0].set(1))
    assert_matches_reference(tables, s, (0, 0), O)
    assert _claw_event(tables, s) == 1


def test_quick_claw_holder_second_shows_nothing(tables, names):
    s = _held_duel(tables, names, "Quick Claw", None, ["Quick Attack"])
    s = s._replace(qc_roll=s.qc_roll.at[0].set(1))
    assert_matches_reference(tables, s, (0, 0), O)
    assert _claw_event(tables, s) == 0


def test_custap_holder_knocked_out_before_its_move_shows_nothing(tables, names):
    s = _held_duel(tables, names, "Custap Berry", 1, ["Quick Attack"])
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[0, 0]) == 0
    assert int(out.item[0, 0]) != 0  # the berry stays: the holder never reached its move
    assert _claw_event(tables, s) == 0


def test_custap_holders_u_turn_shows_its_events_at_the_head_only(tables, names):
    # The flag moves a Custap holder first (order.custap_flags), so the holder's own U-turn
    # pauses the turn at its head: bits 0 and 1 there (the foe has not acted); its resumed
    # step carries no flag of its own (``ts_custap & to_come``). A Custap holder acting after
    # a pause is unreachable: no pivot move has priority.
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 25, "Custap Berry"), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Tackle"], 35)],
    )
    s = s.replace_hp(0, 0, 5)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    assert int(_events(tables, s, (0, 0), O)["quick_claw"][0]) == 3
    acts = (MOVE_ACTIONS + 1, 0)
    assert int(_events(tables, paused, acts, O)["quick_claw"][0]) == 0
    assert_matches_reference(tables, paused, acts, O)


# --- SF2 (a): MoveEnd cures the attacker and the defender before the battle-over check ----------


def _end_duel(tables, n, foe_move, holder, foe_hp, foe_item="Life Orb"):
    """The foe's last Pokemon (Mold Breaker) attacks the player's ``holder`` of the forbidding
    ability; the player's Leer changes nothing."""
    s = make_state(
        tables,
        [_mon(n, holder[0], ["Leer"], 50, ability=holder[1]), _bench(n)],
        [_mon(n, "Cranidos", [foe_move], 10, item=foe_item, ability="Mold Breaker")],
        Weather.NONE,
    )
    return s if foe_hp is None else s.replace_hp(1, 0, foe_hp)


def _lands(side=1):
    sec = jnp.zeros((2, MAX_HITS), jnp.int8).at[side].set(1)
    return O._replace(secondary=sec)


# Immunity's cure (battle_lib.c:4383-4388) comes with the Immunity group (mechanics_backlog.md).
@pytest.mark.parametrize(
    ("foe_move", "holder", "status"),
    [("Body Slam", ("Persian", "Limber"), Status.PARALYSIS)],
)
def test_battle_ending_move_end_cures_the_defender(tables, names, foe_move, holder, status):
    # bcp.c:3965-3969: the cure runs before CheckBattleOver; the foe falls to Life Orb recoil.
    s = _end_duel(tables, names, foe_move, holder, 1)
    out = assert_matches_reference(tables, s, (0, 0), _lands())
    assert int(out.over) == 1
    assert int(out.hp[1, 0]) == 0
    assert int(out.status[0, 0]) == Status.NONE
    # The cure is real: the status landed (the same duel without the cure's holder keeps it).
    plain = _end_duel(tables, names, foe_move, ("Rattata", None), 1)
    assert int(assert_matches_reference(tables, plain, (0, 0), _lands()).status[0, 0]) == status


@pytest.mark.parametrize(
    ("foe_move", "holder"),
    [("Body Slam", ("Persian", "Limber"))],
)
def test_move_end_cure_while_the_battle_goes_on_is_as_before(tables, names, foe_move, holder):
    s = _end_duel(tables, names, foe_move, holder, None, foe_item="")
    out = assert_matches_reference(tables, s, (0, 0), _lands())
    assert int(out.over) == 0
    assert int(out.status[0, 0]) == Status.NONE


def test_battle_ending_move_end_cures_the_attacker(tables, names):
    s = make_state(
        tables,
        [_mon(names, "Persian", ["Body Slam"], 50, ability="Limber"), _bench(names)],
        [_mon(names, "Rattata", ["Tackle"], 5)],
        Weather.NONE,
    )
    s = s.with_status(0, 0, Status.PARALYSIS).replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.over) == 1
    assert int(out.status[0, 0]) == Status.NONE


def test_battle_ending_confusion_self_hit_cures_only_the_user(tables, names):
    # bcp.c:2629: a confusion self-hit makes the user the defender, so the battle-ending MoveEnd
    # (bcp.c:3965-3969) reads the fallen foe twice and the player's Limber holder stays paralyzed.
    s = make_state(
        tables,
        [_mon(names, "Persian", ["Leer"], 50, ability="Limber"), _bench(names)],
        [_mon(names, "Rattata", ["Tackle"], 50)],
        Weather.NONE,
    )
    s = s.with_status(0, 0, Status.PARALYSIS).replace_hp(1, 0, 1)
    s = s._replace(confusion=s.confusion.at[1].set(3))
    hurt = O._replace(confusion_hit=jnp.asarray([0, 1], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), hurt)
    assert int(out.over) == 1
    assert int(out.hp[1, 0]) == 0
    assert int(out.status[0, 0]) == Status.PARALYSIS


@pytest.mark.parametrize("change", ["worry_seed", "gastro_acid"])
def test_battle_ending_move_end_cure_reads_the_battle_ability(tables, names, change):
    # Owed 20: a paralysis given while Limber was replaced (Worry Seed, Skill Swap: BattleMon.ability)
    # or suppressed (Gastro Acid). The battle-ending MoveEnd cure (bcp.c:3965-3969) reads
    # Battler_Ability (battle_lib.c:4382), not the party's Limber that the battle's end puts back.
    s = make_state(
        tables,
        [_mon(names, "Persian", ["Scratch"], 50, ability="Limber"), _bench(names)],
        [_mon(names, "Rattata", ["Leer"], 50)],
        Weather.NONE,
    )
    s = s.with_status(0, 0, Status.PARALYSIS).replace_hp(1, 0, 1)
    if change == "worry_seed":
        s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.INSOMNIA))
    else:
        s = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.over) == 1
    assert int(out.hp[1, 0]) == 0
    assert int(out.status[0, 0]) == Status.PARALYSIS
    assert int(out.ability[0, 0]) == AbilityId.LIMBER


# SF2 (l): a Pursuit's crit armor is judged on the outgoing switcher (battle_lib.c:7135-7136;
# battle_script.c:6927).


_ARMOR_SPECIES = {"Battle Armor": "Kabuto", "Shell Armor": "Omanyte", None: "Geodude"}


def _armor_duel(tables, n, out_ability, new_ability, chaser="Houndour"):
    """Battler 1 FIGHTs Pursuit; the player's slot 0 (``out_ability``) is switched for slot 1."""

    def holder(ability):
        return _mon(n, _ARMOR_SPECIES[ability], ["Tackle"], 40, ability=ability)

    foe = [_mon(n, chaser, ["Pursuit"], 35), _bench(n)]
    return make_state(tables, [holder(out_ability), holder(new_ability)], foe)


def _crit_out():
    return O._replace(crit=jnp.ones((2, MAX_HITS), jnp.int8))


def test_pursuit_crit_armor_is_the_newcomers_no_longer(tables, names):
    from tests.engine.test_enumerate import (
        assert_equals_bruteforce,
        check_frequencies,
        children,
        enumerate_pair,
        laws_of,
        sampled,
    )

    s = _armor_duel(tables, names, None, "Battle Armor")
    acts = (SWITCH, 0)
    laws = laws_of(tables, s, acts)
    assert not bool(laws.no_crit[1])
    outcomes, probs = enumerate_pair(tables, s, acts)
    crit = np.asarray(outcomes.crit)[:, 1, 0]
    assert set(crit.tolist()) == {0, 1}
    assert abs(float(np.asarray(probs)[crit == 1].sum()) - 1 / 16) < 1e-9
    ev = _events(tables, s, acts, _crit_out())
    assert int(ev["crit"][1][0]) == 1
    assert_matches_reference(tables, s, acts, _crit_out())
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, 4000), 4000)
    assert_equals_bruteforce(tables, s, acts)


def test_pursuit_crit_is_blocked_by_the_switchers_armor(tables, names):
    s = _armor_duel(tables, names, "Shell Armor", None)
    acts = (SWITCH, 0)
    from tests.engine.test_enumerate import laws_of

    assert bool(laws_of(tables, s, acts).no_crit[1])
    ev = _events(tables, s, acts, _crit_out())
    assert int(ev["crit"][1][0]) == 0
    assert_matches_reference(tables, s, acts, _crit_out())


def test_mold_breaker_pursuer_ignores_the_switchers_armor(tables, names):
    from tests.engine.helpers import as_species
    from tests.engine.test_enumerate import laws_of

    n = names
    s = _armor_duel(tables, n, "Battle Armor", None)
    acts = (SWITCH, 0)
    assert bool(laws_of(tables, s, acts).no_crit[1])
    sp = n.species("Pinsir")
    slot = n.species_abilities(sp).index(n.ability("Mold Breaker"))
    s = as_species(tables, s, 1, 0, sp, slot)
    assert not bool(laws_of(tables, s, acts).no_crit[1])


def test_paused_state_has_no_pursuit_chase(tables, names):
    paused = _paused_u_turn(tables, names, ["Pursuit"])
    assert int(paused.ts_pursued) == 2
    acts = jnp.asarray([SWITCH, 0], jnp.int32)
    assert not np.any(np.asarray(outcome_mod.pursuit_chase(tables, paused, acts)[0]))
    view, view_acts = pivot_view(tables, paused, acts)
    assert not np.any(np.asarray(outcome_mod.pursuit_chase(tables, view, view_acts)[0]))
