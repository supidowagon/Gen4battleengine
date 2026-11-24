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
from nuzline.engine.ids import (
    ABILITY_DECOMP_IDS,
    AbilityId,
    FieldCondition,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.order import metronome_weights, with_selection
from nuzline.engine.outcome import MAX_HITS, empty_outcome, pivot_view
from nuzline.engine.state import make_state
from nuzline.engine.step import _pursuit_hit, step_events
from tests.engine.helpers import assert_matches_reference, jit_step, step_rows, trick_room_of

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


def _player_switcher(tables, n, species, ability, bond, pursuer_hp=None):
    # The AI tracks only the player's Pokemon (hooks.reveal_ability): the player's switcher
    # shows whether its contact ability ran.
    s = make_state(tables, [_weak(n, species, ability), _bench(n)], [_chaser(n), _bench(n)])
    if pursuer_hp is not None:
        s = s.replace_hp(1, 0, pursuer_hp)
    return _bond(s, 0) if bond else s


def _after_hit(tables, n, s):
    # The state right after battler 1's Pursuit hit on the switching battler 0, its faint check
    # and the switch still to come.
    choices = (jnp.asarray([0, n.move("Pursuit")], jnp.int32), jnp.zeros(2, bool))
    acts = jnp.asarray((SWITCH, 0), jnp.int32)
    s = with_selection(tables, s, acts)  # TEMP_VALUE for TryPursuit (E2-ML-8)
    return jax.jit(_pursuit_hit)(tables, s, 1, 0, acts, choices, _proc(1))[0]


@pytest.mark.parametrize("pursuer_hp", [None, 1])
def test_bond_takes_the_pursuer_before_rough_skin(tables, names, pursuer_hp):
    # The old order ran Rough Skin first: its message tagged the switcher's ability
    # (subscript_rough_skin.s:8, battle_script.c:11805-11806) and the AI learned it. Now the
    # take-down (subscript_pursuit.s:85) leaves Rough Skin's attacker-HP test failing
    # (battle_lib.c:4258). The record is read right after the hit: the switch goes on after the KO
    # (subscript_switch_pokemon.s:11-13, :22) and the newcomer's entry clears the battler's record
    # (SwitchAndUpdateMon, battle_script.c:2522; battle_lib.c:2151-2153), so after the turn it is 0
    # with and without the bond.
    n = names
    s = _player_switcher(tables, n, "Carvanha", "Rough Skin", True, pursuer_hp)
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.hp[0, 0]) == 0
    assert int(out.hp[1, 0]) == 0
    assert int(out.ai_ability_known[0]) == 0
    hit = _after_hit(tables, n, s)
    assert int(hit.hp[0, 0]) == 0  # the hit happened
    assert int(hit.ai_ability_known[0]) == 0
    ev = _events(tables, s, (SWITCH, 0), _proc(1))
    assert int(ev["contact_ability"][0]) == 0
    # Without the bond Rough Skin reveals itself, after the KO's own clear (:85 before :86).
    s = _player_switcher(tables, n, "Carvanha", "Rough Skin", False, pursuer_hp)
    hit = _after_hit(tables, n, s)
    assert int(hit.hp[0, 0]) == 0
    assert int(hit.ai_ability_known[0]) == ABILITY_DECOMP_IDS[AbilityId.ROUGH_SKIN]
    out = assert_matches_reference(tables, s, (SWITCH, 0), _proc(1))
    assert int(out.active[0]) == 1
    assert int(out.ai_ability_known[0]) == 0


def test_rough_skin_of_a_u_turn_user_felled_by_pursuit_stays_known(tables, names):
    # A U-turn user the Pursuit fells gets no list (subscript_attack_then_switch_out.s:54-55). Its
    # FAINT_MON cleared the record (subscript_pursuit.s:85 -> subscript_faint_check_destiny_bond.s:
    # 36 -> battle_script.c:1803, battle_lib.c:2223-2225) and Rough Skin's message wrote the ability
    # after it (subscript_pursuit.s:86-87, subscript_rough_skin.s:8, battle_script.c:11805-11806).
    # Nothing clears that before the replacement enters; the moves seen before the KO are gone.
    n = names
    s = make_state(
        tables,
        [_mon(n, "Carvanha", ["U-turn"], 30, ability="Rough Skin"), _bench(n)],
        [_chaser(n)],
    ).replace_hp(0, 0, 1)
    # Houndour acts after the U-turn and lives through it.
    stages = s.stages.at[0, Stat.ATK].set(_i8(s.stages, -6))
    s = s._replace(
        stages=stages.at[1, Stat.SPE].set(_i8(s.stages, -6)),
        ai_seen_moves=s.ai_seen_moves.at[0, 0].set(n.move("U-turn")),
    )
    out = assert_matches_reference(tables, s, (0, 0), _proc(1))
    assert int(out.pivot_battler) == 0
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) > 0
    assert int(out.ai_ability_known[0]) == ABILITY_DECOMP_IDS[AbilityId.ROUGH_SKIN]
    assert not np.asarray(out.ai_seen_moves[0]).any()


def test_static_switcher_with_bond_paralyzes_nobody_for_either_draw(tables, names):
    # The old order rolled Static first: on a proc it paralyzed the pursuer and revealed the
    # switcher's ability (``hooks.CONTACT_STATUS``), so the two draws gave two states. Now the
    # draw is never read. ``outcome_laws`` keeps ``contact_proc`` relevant (a superset: whether the
    # Pursuit KOs depends on the crit and roll draws), so the check is that every row plays to one
    # state.
    n = names
    s = _player_switcher(tables, n, "Pikachu", "Static", True)
    outs = [assert_matches_reference(tables, s, (SWITCH, 0), _proc(v)) for v in (0, 1)]
    for out in outs:
        assert int(out.hp[1, 0]) == 0
        assert int(out.status[1, 0]) == Status.NONE
    hit = _after_hit(tables, n, s)
    assert int(hit.hp[0, 0]) == 0  # the hit happened
    assert int(hit.ai_ability_known[0]) == 0  # before the newcomer's entry
    for x, y in zip(outs[0], outs[1], strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    rows, _ = enumerate_turn(tables, s, (SWITCH, 0))
    states = set()
    for i in range(len(rows.hit)):
        row = jax.tree.map(lambda f, i=i: f[i], rows)
        out = jit_step(tables, s, jnp.asarray((SWITCH, 0)), row)
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


@pytest.mark.slow  # enumerate_turn twice with cache clears
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
    old_mask = lambda s: (
        (s.ts_acted != 0) | (((s.ts_pursued.astype(jnp.int32) >> jnp.arange(2)) & 1) != 0)
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


@pytest.mark.slow  # brute force and enumerate_turn twice with cache clears
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
    # step carries no flag of its own (cleared at its move start). A Custap holder acting after
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


# H-2: Immunity (battle_lib.c:4383-4388), Water Veil (:4412-4417), Magma Armor (:4419-4424).
@pytest.mark.parametrize(
    ("foe_move", "holder", "status"),
    [
        ("Body Slam", ("Persian", "Limber"), Status.PARALYSIS),
        ("Poison Jab", ("Zangoose", "Immunity"), Status.POISON),
        ("Ember", ("Wailmer", "Water Veil"), Status.BURN),
        ("Ice Beam", ("Camerupt", "Magma Armor"), Status.FREEZE),
    ],
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
    [
        ("Body Slam", ("Persian", "Limber")),
        ("Poison Jab", ("Zangoose", "Immunity")),
        ("Ember", ("Wailmer", "Water Veil")),
        ("Ice Beam", ("Camerupt", "Magma Armor")),
    ],
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
    # (bcp.c:3965-3969) reads the fallen foe twice and the player's Limber holder stays paralyzed
    # through it (no cure, so no STATUS_CURE reveal); subscript_battle_won.s:6-7 then clears it
    # (Ability_ForbidsStatus on the party's Limber, battle_lib.c:4466-4470; H-2).
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
    assert int(out.ai_ability_known[0]) == 0
    assert int(out.status[0, 0]) == Status.NONE


@pytest.mark.parametrize("change", ["worry_seed", "gastro_acid"])
def test_battle_ending_move_end_cure_reads_the_battle_ability(tables, names, change):
    # Owed 20: a paralysis given while Limber was replaced (Worry Seed, Skill Swap: BattleMon.ability)
    # or suppressed (Gastro Acid). The battle-ending MoveEnd cure (bcp.c:3965-3969) reads
    # Battler_Ability (battle_lib.c:4382), not the party's Limber that the battle's end puts back:
    # no MoveEnd cure (no STATUS_CURE reveal). subscript_battle_won.s:6-7 then clears it through
    # the party's Limber (battle_script.c:9149, battle_lib.c:4466-4470; H-2).
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
    assert int(out.ai_ability_known[0]) == 0
    assert int(out.status[0, 0]) == Status.NONE
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


@pytest.mark.slow  # brute force (assert_equals_bruteforce)
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


# SF2 (b): Intimidate and a substitute the action hit, across the party-list pause and U-turn's
# Pursuit, and on the next turn. A hit on a substitute sets the defender's
# SELF_TURN_FLAG_SUBSTITUTE_HIT, broken or not (bcp.c:3359-3371); CheckSubstitute skips a foe
# with a substitute up or that flag (battle_script.c:9179-9182; subscript_intimidate.s:13). The
# flag clears at BattleContext_Init (battle_lib.c:2013-2016, bcp.c:3998), after the U-turn
# newcomer's switch-in states (AfterMoveEffects, bcp.c:3672-3682); U-turn's :70 clears only the
# attacker's flags (subscript_attack_then_switch_out.s:70). Built by E1 (F-4).


def _sub_u_turn(tables, n, substitute, foe=("Bulbasaur", "Tackle")):
    """Scyther's U-turn hits ``foe`` behind a ``substitute``-HP substitute; an Intimidate
    Gyarados waits on the bench (party slot 1)."""
    s = make_state(
        tables,
        [
            _mon(n, "Scyther", ["U-turn", "Quick Attack"], 30),
            _mon(n, "Gyarados", ["Tackle"], 30, ability="Intimidate"),
        ],
        [_mon(n, foe[0], [foe[1]], 35)],
    )
    return s._replace(substitute_hp=s.substitute_hp.at[1].set(_i8(s.substitute_hp, substitute)))


def _resume_slot1(tables, paused):
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[int(paused.pivot_battler) - 1] = SWITCH
    return assert_matches_reference(tables, paused, tuple(acts), O)


def test_broken_substitute_flag_waits_through_the_pause(tables, names):
    paused = assert_matches_reference(tables, _sub_u_turn(tables, names, 1), (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.substitute_hp[1]) == 0
    assert np.asarray(paused.ts_sub_hit).tolist() == [0, 1]
    out = _resume_slot1(tables, paused)
    assert int(out.active[0]) == 1
    assert np.asarray(out.ts_sub_hit).tolist() == [0, 0]
    assert int(out.stages[1, Stat.ATK]) == 0  # skipped


def test_substitute_that_holds_keeps_the_foes_attack(tables, names):
    paused = assert_matches_reference(tables, _sub_u_turn(tables, names, 200), (0, 0), O)
    assert int(paused.substitute_hp[1]) > 0
    out = _resume_slot1(tables, paused)
    assert int(out.substitute_hp[1]) > 0 and int(out.stages[1, Stat.ATK]) == 0


def test_broken_substitute_flag_survives_the_pursuit_chase(tables, names):
    s = _sub_u_turn(tables, names, 1, foe=("Houndour", "Pursuit"))
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_pursued) == 2
    assert int(paused.hp[0, 0]) > 0 and int(paused.substitute_hp[1]) == 0
    assert int(paused.ts_sub_hit[1]) == 1
    out = _resume_slot1(tables, paused)
    assert int(out.active[0]) == 1 and int(out.stages[1, Stat.ATK]) == 0


def test_next_turns_switch_in_finds_the_flag_clear(tables, names):
    s = _sub_u_turn(tables, names, 1)
    hit = assert_matches_reference(tables, s, (1, 0), O)  # Quick Attack breaks it
    assert int(hit.substitute_hp[1]) == 0 and np.asarray(hit.ts_sub_hit).tolist() == [0, 0]
    out = assert_matches_reference(tables, hit, (SWITCH, 0), O)
    assert int(out.active[0]) == 1 and int(out.stages[1, Stat.ATK]) == -1


# SF2 (i): Enigma Berry in the resumed U-turn tail (bcp.c:3698-3706; battle_lib.c:5374-5378,
# :5421-5428). The tail reads the defender's SUBSTITUTE_HIT (``ts_sub_hit``, a per-battler byte
# since E1 batch F-4) and moveStatusFlags & SUPER_EFFECTIVE (``ts_se``): the U-turn's, or the last
# Pursuit's when one ran (lib.c:3359, script.c:6964-6973).


def _enigma_duel(tables, n, user, foe, foe_moves, *, user_level=50, foe_level=40, user_hp=None,
                 foe_hp=None, user_moves=("U-turn",), substitute=0, user_types=None,
                 user_ability=None, foe_bench=False):  # fmt: skip
    """The player's ``user`` (U-turn) with a Geodude on the bench vs ``foe`` holding an Enigma
    Berry (``foe_hp``: its starting HP, else full)."""
    s = make_state(
        tables,
        [
            _mon(n, user, list(user_moves), user_level, ability=user_ability),
            _mon(n, "Geodude", ["Tackle"], 30),
        ],
        [_mon(n, foe, list(foe_moves), foe_level, "Enigma Berry")]
        + ([_mon(n, "Geodude", ["Tackle"], 30)] if foe_bench else []),
    )
    if foe_hp is not None:
        s = s._replace(hp=s.hp.at[1, 0].set(_i8(s.hp, foe_hp)))
    if user_hp is not None:
        s = s._replace(hp=s.hp.at[0, 0].set(_i8(s.hp, user_hp)))
    if user_types is not None:
        s = s._replace(types=s.types.at[0, 0].set(jnp.asarray(user_types, s.types.dtype)))
    if substitute:
        s = s._replace(substitute_hp=s.substitute_hp.at[1].set(_i8(s.substitute_hp, substitute)))
    return s


def _gain(tables, n, s):
    return max(int(s.max_hp[1, 0]) // int(tables.item_param[n.item("Enigma Berry"), 0]), 1)


def _heals_on_resume(tables, names, paused, heals):
    hp0 = int(paused.hp[1, 0])
    out = _resume_slot1(tables, paused)
    if heals:
        gain = _gain(tables, names, paused)
        assert hp0 + gain <= int(paused.max_hp[1, 0]) and int(out.hp[1, 0]) == hp0 + gain
        assert int(out.item[1, 0]) == 0
    else:
        assert int(out.hp[1, 0]) == hp0
        assert int(out.item[1, 0]) == names.item("Enigma Berry")
    assert int(out.ts_se) == 0
    return out


def test_enigma_heals_after_a_super_effective_u_turn(tables, names):
    s = _enigma_duel(tables, names, "Scyther", "Tangela", ["Tackle"], foe_level=70)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.hp[1, 0]) > 0
    assert int(paused.ts_se) == 1 and int(paused.item[1, 0]) == names.item("Enigma Berry")
    _heals_on_resume(tables, names, paused, True)


def test_enigma_keeps_after_a_resisted_u_turn(tables, names):
    s = _enigma_duel(tables, names, "Scyther", "Charmander", ["Tackle"], foe_level=70)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_se) == 0
    _heals_on_resume(tables, names, paused, False)


def test_enigma_does_not_heal_behind_a_substitute_the_u_turn_broke(tables, names):
    s = _enigma_duel(tables, names, "Scyther", "Tangela", ["Tackle"], foe_level=70, substitute=1)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.ts_se) == 1 and np.asarray(paused.ts_sub_hit).tolist() == [0, 1]
    _heals_on_resume(tables, names, paused, False)


def test_enigma_heals_on_the_pursuits_flag_when_the_u_turn_was_resisted(tables, names):
    s = _enigma_duel(tables, names, "Abra", "Charmander", ["Pursuit"], user_level=60, foe_level=20)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_pursued) == 2
    assert int(paused.hp[0, 0]) > 0 and int(paused.hp[1, 0]) > 0
    assert int(paused.ts_se) == 1  # Pursuit, Dark on Psychic; the U-turn (Bug on Fire) resisted
    _heals_on_resume(tables, names, paused, True)


def test_enigma_keeps_when_the_pursuit_is_neutral_after_a_super_effective_u_turn(tables, names):
    s = _enigma_duel(tables, names, "Scyther", "Tangela", ["Pursuit"], user_level=60, foe_level=70)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_pursued) == 2
    assert int(paused.ts_se) == 0
    _heals_on_resume(tables, names, paused, False)


def test_enigma_pursuer_that_falls_to_rough_skin_does_not_heal(tables, names):
    # Rough Skin (Carvanha), types made Psychic: Pursuit is super effective, and the pursuer is
    # left with 1 HP after the U-turn, which its contact does not survive.
    duel = {
        "user_level": 60,
        "foe_level": 45,
        "user_ability": "Rough Skin",
        "foe_bench": True,
        "user_types": (Type.PSYCHIC, Type.PSYCHIC),
    }
    full = _enigma_duel(tables, names, "Carvanha", "Charmander", ["Pursuit"], **duel)
    after = assert_matches_reference(tables, full, (0, 0), O)
    rough_skin = max(int(full.max_hp[1, 0]) // 16, 1)
    taken = int(full.max_hp[1, 0]) - int(after.hp[1, 0]) - rough_skin  # the U-turn's damage
    s = _enigma_duel(
        tables, names, "Carvanha", "Charmander", ["Pursuit"], foe_hp=taken + rough_skin, **duel
    )  # fmt: skip
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_se) == 1
    assert int(paused.hp[1, 0]) == 0 and int(paused.item[1, 0]) == names.item("Enigma Berry")
    out = _resume_slot1(tables, paused)
    assert int(out.hp[1, 0]) == 0 and int(out.ts_se) == 0


def test_enigma_heals_in_the_same_step_when_the_pursuit_kills_the_user(tables, names):
    s = _enigma_duel(
        tables, names, "Abra", "Charmander", ["Pursuit"], user_level=60, foe_level=20, user_hp=1
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0 and int(out.ts_se) == 0
    assert int(out.hp[0, 0]) == 0 and int(out.item[1, 0]) == 0
    assert int(out.hp[1, 0]) > 0


def test_enigma_keeps_in_the_same_step_when_the_killing_pursuit_is_neutral(tables, names):
    s = _enigma_duel(
        tables, names, "Scyther", "Tangela", ["Pursuit"], user_level=60, foe_level=70, user_hp=1
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0 and int(out.ts_se) == 0 and int(out.hp[0, 0]) == 0
    assert int(out.item[1, 0]) == names.item("Enigma Berry")


def test_baton_pass_pause_leaves_the_flag_clear_and_the_newcomer_untouched(tables, names):
    s = _enigma_duel(tables, names, "Scyther", "Tangela", ["Tackle"], user_moves=("Baton Pass",))
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_se) == 0
    out = _resume_slot1(tables, paused)
    assert int(out.item[1, 0]) == names.item("Enigma Berry")
    assert int(out.hp[1, 0]) == int(paused.hp[1, 0])


# SF2 (a2): the switch / U-turn goes on after a Pursuit that empties a side. Nothing between the
# Pursuit and the action's MoveEnd checks for a battle end (subscript_switch_pokemon.s:12-48,
# subscript_show_party_list.s:23-24); MoveEnd decides (bcp.c:3965-3969), after the cure, the exp
# payout and before TriggerEffectOnSwitch (:3973). A recoil-felled pursuer's exp is paid there too
# (AnyExpPayout), after the newcomer's hazards; a Destiny-Bond pursuer's at the Pursuit (R3).


def _a2_mon(n, species, ability=None, item="", level=50, moves=("Tackle",)):
    return _mon(n, species, list(moves), level, item=item, ability=ability)


def _a2_duel(tables, n, *, a=("Geodude", None, ""), b=("Geodude", None, ""), foe_hp=1,
             foe_item="Life Orb", foe_bench=False, rock=True, spikes=0, toxic=0, mask=None,
             b_hp=None, a_hp=None):  # fmt: skip
    """The foe's last Pokemon (Houndour, Pursuit, ``foe_hp``) against the player's A (slot 0) and
    B (slot 1); Stealth Rock, ``spikes`` layers and ``toxic`` layers on the player's side."""
    s = make_state(
        tables,
        [_a2_mon(n, *a), _a2_mon(n, *b)],
        [_mon(n, "Houndour", ["Pursuit"], 5, item=foe_item)] + ([_bench(n)] if foe_bench else []),
        Weather.NONE,
    )
    if foe_hp is not None:
        s = s.replace_hp(1, 0, foe_hp)
    if a_hp is not None:
        s = s.replace_hp(0, 0, a_hp)
    if b_hp is not None:
        s = s.replace_hp(0, 1, b_hp)
    s = s._replace(
        stealth_rock=s.stealth_rock.at[0].set(int(rock)),
        spikes=s.spikes.at[0].set(spikes),
        toxic_spikes=s.toxic_spikes.at[0].set(toxic),
    )
    if mask is not None:
        s = s._replace(exp_mask=_i8(s.exp_mask, mask))
    return s


def test_switch_goes_on_after_a_pursuit_recoil_empties_the_foe(tables, names):
    s = _a2_duel(tables, names)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.hp[1, 0]) == 0
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.active[0]) == 1
    assert int(out.hp[0, 1]) < int(s.hp[0, 1])  # Stealth Rock
    assert int(out.hp[0, 0]) > 0 and int(out.deaths[0]) == 0
    assert int(out.exp[0]) > int(s.exp[0])  # A, in the mask, is paid at MoveEnd


def test_destiny_bond_draw_is_a_loss(tables, names):
    # A holds Destiny Bond, the pursuer KOs it and is taken; B, the last other Pokemon, falls to
    # Stealth Rock: both sides are out = a draw = a loss (bcp.c:4255-4260, :4015-4016).
    s = _bond(_a2_duel(tables, names, foe_hp=None, a_hp=1, b_hp=1), 0)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) == 0 and int(out.hp[0, 1]) == 0
    assert int(out.over) == 1 and int(out.winner) == 1
    assert np.asarray(out.deaths).tolist()[:2] == [1, 1] and int(out.player_faints) == 2
    assert int(out.exp[0]) == int(s.exp[0])  # n = 0: nothing paid


def test_hazard_kill_of_the_newcomer_leaves_the_exp_to_the_switcher(tables, names):
    s = _a2_duel(tables, names, spikes=3, mask=3, b_hp=1)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.hp[0, 1]) == 0 and int(out.deaths[1]) == 1 and int(out.player_faints) == 1
    assert int(out.exp[0]) > int(s.exp[0]) and int(out.exp[1]) == int(s.exp[1])
    # n = 1 (the dead newcomer is not in the divisor): A alone gets the whole share.
    alone = assert_matches_reference(
        tables, _a2_duel(tables, names, spikes=3, mask=1, b_hp=1), (SWITCH, 0), O
    )
    assert int(alone.exp[0]) == int(out.exp[0])


def test_no_switch_in_effects_for_the_newcomer_at_the_battles_end(tables, names):
    # Toxic Spikes poison B; its Pecha Berry (TriggerEffectOnSwitch's HELD_ITEM_STATUS) and its
    # Intimidate never run (bcp.c:3969 returns before :3973).
    s = _a2_duel(tables, names, b=("Growlithe", "Intimidate", "Pecha Berry"), toxic=1, rock=False)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.over) == 1 and int(out.active[0]) == 1
    assert int(out.status[0, 1]) == Status.POISON
    assert int(out.item[0, 1]) == names.item("Pecha Berry")
    assert int(out.stages[1, Stat.ATK]) == 0


def test_the_newcomer_is_cured_by_move_end_at_the_battles_end(tables, names):
    s = _a2_duel(tables, names, b=("Persian", "Limber", ""), rock=False)
    s = s.with_status(0, 1, Status.PARALYSIS)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.over) == 1 and int(out.active[0]) == 1
    assert int(out.status[0, 1]) == Status.NONE


def test_recoil_felled_pursuer_pays_after_the_newcomers_hazards(tables, names):
    # A (Rough Skin) falls to the Pursuit; the pursuer, at 1 HP, falls to Rough Skin (the KO
    # path); B, in the mask, levels up after its Stealth Rock damage (subscript_pursuit.s:86,
    # bcp.c:3968).
    a = ("Carvanha", "Rough Skin", "")
    base = _a2_duel(tables, names, a=a, a_hp=1, mask=2)
    plain = assert_matches_reference(tables, base, (SWITCH, 0), _proc(1))
    assert int(plain.over) == 1 and int(plain.winner) == 2 and int(plain.hp[0, 0]) == 0
    assert int(plain.hp[1, 0]) == 0 and int(plain.active[0]) == 1
    assert int(plain.exp[1]) > int(base.exp[1]) and int(plain.level[0, 1]) == 50
    rate = int(tables.species_exp_rate[int(base.species[0, 1])])
    near = base._replace(exp=base.exp.at[1].set(int(tables.exp_table[rate, 51]) - 1))
    out = assert_matches_reference(tables, near, (SWITCH, 0), _proc(1))
    assert int(out.level[0, 1]) == 51
    rock = int(base.max_hp[0, 1]) - int(plain.hp[0, 1])
    assert rock > 0 and int(out.hp[0, 1]) == int(out.max_hp[0, 1]) - rock


def _u_turn_duel(tables, n, *, spikes=0):
    """The player's Scyther (U-turn) and a Growlithe (Intimidate) vs the foe's last Pokemon:
    Tyranitar with Pursuit and a Life Orb, left with 1 HP by the U-turn."""
    s = make_state(
        tables,
        [
            _a2_mon(n, "Scyther", moves=("U-turn",), level=50),
            _a2_mon(n, "Growlithe", "Intimidate"),
        ],
        [_mon(n, "Tyranitar", ["Pursuit"], 50, item="Life Orb")],
        Weather.NONE,
    )
    full = assert_matches_reference(tables, s, (0, 0), O)
    # The full-HP run lost the U-turn's damage and the Life Orb's recoil (max / 10).
    taken = int(s.hp[1, 0]) - int(full.hp[1, 0]) - int(s.max_hp[1, 0]) // 10
    s = s.replace_hp(1, 0, taken + 1)
    return s._replace(spikes=s.spikes.at[0].set(spikes))


def test_u_turn_pauses_after_a_pursuit_that_empties_the_foe(tables, names):
    s = _u_turn_duel(tables, names, spikes=1)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.hp[1, 0]) == 0 and int(paused.hp[0, 0]) > 0
    assert int(paused.pivot_battler) == 1 and int(paused.over) == 0
    assert int(paused.exp[0]) == int(s.exp[0])  # the recoil-felled pursuer waits for MoveEnd
    out = _resume_slot1(tables, paused)
    assert int(out.over) == 1 and int(out.winner) == 2 and int(out.active[0]) == 1
    assert int(out.hp[0, 1]) < int(paused.hp[0, 1])  # Spikes
    assert int(out.exp[0]) > int(s.exp[0])


def test_foe_switch_into_the_players_last_pursuer_is_a_loss(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Pursuit"], 5, item="Life Orb")],
        [_a2_mon(n, "Geodude"), _a2_mon(n, "Geodude")],
        Weather.NONE,
    )
    s = s.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(out.over) == 1 and int(out.winner) == 1
    assert int(out.active[1]) == 1 and int(out.deaths[0]) == 1 and int(out.hp[0, 0]) == 0


def test_the_battle_goes_on_with_a_foe_bench(tables, names):
    s = _a2_duel(tables, names, foe_bench=True, rock=False)
    out = assert_matches_reference(tables, s, (SWITCH, 0), O)
    assert int(out.over) == 0 and int(out.active[0]) == 1
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 1]) == int(s.hp[0, 1])
    assert int(out.exp[0]) > int(s.exp[0])


# --- SF2 (k): claw_flag / custap_flag persist across the turn end (B = 2) --------------------
# A Trick Room's MoveEnd re-sort sets the flags of the activated holders again (bcp.c:3991,
# lib.c:7276-7300); a U-turn's pursuer never runs BeforeMove (script.c:6931). Only the holder's
# own BeforeMove, a switch-in and a faint clear them (subscript_check_quick_claw.s:22-28,
# lib.c:2100-2104, :2185-2188).

_SPE = Stat.SPE - 1


def _tr_duel(tables, n, item, foe_moves, *, foe_item="", foe_hp=None, foe_bench=False):
    """The player's Houndour (``item``; Trick Room, Tackle) and a bench Geodude against a Rattata."""
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Trick Room", "Tackle"], 40, item), _bench(n)],
        [_mon(n, "Rattata", foe_moves, 5, foe_item)] + ([_bench(n)] if foe_bench else []),
    )
    return s if foe_hp is None else s.replace_hp(1, 0, foe_hp)


def _pinch_after_tackle(tables, s, acts=(0, 0), o=O):
    """``s`` with the player's lead at the HP that leaves it exactly at the Custap pinch (a
    quarter of its max HP) once the foe's Tackle has hit: above it when the turn starts."""
    probe = assert_matches_reference(tables, s, acts, o)
    taken = int(s.hp[0, 0]) - int(probe.hp[0, 0])
    assert taken > 0
    return s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 4 + taken)


def test_trick_room_sets_custap_again_and_the_next_turn_eats_it(tables, names):
    n = names
    s = _pinch_after_tackle(tables, _tr_duel(tables, n, "Custap Berry", ["Tackle"], foe_bench=True))
    assert np.asarray(s.custap_flag).tolist() == [0, 0]
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 4 and int(trick_room_of(out)) != 0
    assert int(out.item[0, 0]) == n.item("Custap Berry")
    assert np.asarray(out.custap_flag).tolist() == [1, 0]
    assert np.asarray(out.claw_flag).tolist() == [0, 0]
    # Next turn the foe switches; the holder (above its pinch) attacks: the standing flag eats
    # the berry (animation, no message: the foe has acted) and is gone.
    nxt = out.replace_hp(0, 0, int(out.max_hp[0, 0]))
    acts = (1, SWITCH)
    after = assert_matches_reference(tables, nxt, acts, O)
    assert int(after.item[0, 0]) == 0
    assert int(_events(tables, nxt, acts, O)["quick_claw"][0]) == 1
    assert np.asarray(after.custap_flag).tolist() == [0, 0]


def test_trick_room_sets_no_flag_when_the_foe_is_down_or_switching(tables, names):
    n = names
    # The foe's Tackle hurts the holder into its pinch, then its Life Orb recoil fells it: it is
    # at 0 HP at the Trick Room's MoveEnd (a bench keeps the battle going).
    s = _tr_duel(
        tables, n, "Custap Berry", ["Tackle"], foe_item="Life Orb", foe_hp=1, foe_bench=True
    )
    s = _pinch_after_tackle(tables, s)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[1, 0]) == 0 and int(trick_room_of(out)) != 0 and int(out.over) == 0
    assert int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 4
    assert np.asarray(out.custap_flag).tolist() == [0, 0]
    # The foe switched: no FIGHT pair, no compare, whatever the holder's HP.
    s = _tr_duel(tables, n, "Custap Berry", ["Tackle"], foe_bench=True)
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 4)
    out = assert_matches_reference(tables, s, (0, SWITCH), O)
    assert int(trick_room_of(out)) != 0
    assert np.asarray(out.custap_flag).tolist() == [0, 0]


def _claw_next(out):
    """The turn after: the claw draw of the turn before is 0, Trick Room is over, the foe fast."""
    off = jnp.asarray(~int(FieldCondition.TRICK_ROOM), out.field_mask.dtype)
    nxt = out._replace(qc_roll=jnp.zeros_like(out.qc_roll), field_mask=out.field_mask & off)
    return nxt._replace(stats=nxt.stats.at[1, 0, _SPE].set(999))


def test_trick_room_sets_quick_claw_again_and_it_guards_against_knock_off(tables, names):
    n = names
    s = _tr_duel(tables, n, "Quick Claw", ["Tackle", "Knock Off"])
    o = O._replace(quick_claw=jnp.asarray([1, 0], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(trick_room_of(out)) != 0
    assert np.asarray(out.claw_flag).tolist() == [1, 0]
    nxt = _claw_next(out)
    acts = (1, 1)  # the foe's Knock Off comes first, on the standing flag
    after = assert_matches_reference(tables, nxt, acts, O)
    assert int(after.item[0, 0]) == n.item("Quick Claw")
    assert np.asarray(after.claw_flag).tolist() == [0, 0]  # the holder's BeforeMove cleared it
    # Without the flag the Knock Off lands.
    bare = nxt._replace(claw_flag=jnp.zeros_like(nxt.claw_flag))
    assert int(assert_matches_reference(tables, bare, acts, O).item[0, 0]) == 0


def test_a_called_trick_room_guards_the_holder_in_the_same_turn(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Metronome", "Tackle"], 40, "Quick Claw")],
        [_mon(n, "Rattata", ["Knock Off"], 5)],
    )
    s = s._replace(qc_roll=jnp.asarray([1, 0], s.qc_roll.dtype))  # the claw moves the holder first
    weights = np.asarray(metronome_weights(tables, s, 0))
    room = n.move("Trick Room")
    assert weights[room] > 0
    rank = int(np.count_nonzero(weights[:room]))
    o = O._replace(quick_claw=jnp.asarray([1, 0], jnp.int8), call=jnp.asarray([rank, 0], jnp.int16))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(trick_room_of(out)) != 0
    assert int(out.item[0, 0]) == n.item("Quick Claw")  # the slower foe's Knock Off failed


def _pursuit_duel(tables, n):
    """The player's faster U-turner (Quick Claw) and a Knock Off bench against the foe's Pursuit
    holder (Quick Claw) and a bench; both claws active when the turn starts."""
    s = make_state(
        tables,
        [
            _a2_mon(n, "Scyther", item="Quick Claw", moves=("U-turn",), level=50),
            _a2_mon(n, "Growlithe", moves=("Knock Off",)),
        ],
        [_mon(n, "Tyranitar", ["Pursuit"], 50, item="Quick Claw"), _bench(n)],
        Weather.NONE,
    )
    return s._replace(qc_roll=jnp.asarray([1, 1], s.qc_roll.dtype))


def test_a_u_turns_pursuer_keeps_its_claw_flag(tables, names):
    n = names
    s = _pursuit_duel(tables, n)
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    assert np.asarray(paused.ts_claw).tolist() == [0, 1]  # the pursuer never ran BeforeMove
    out = _resume_slot1(tables, paused)
    assert int(out.active[0]) == 1
    assert np.asarray(out.claw_flag).tolist() == [0, 1]
    nxt = out._replace(qc_roll=jnp.zeros_like(out.qc_roll))
    nxt = nxt._replace(stats=nxt.stats.at[0, 1, _SPE].set(999))  # the Growlithe moves first
    acts = (0, 0)
    after = assert_matches_reference(tables, nxt, acts, O)
    assert int(after.item[1, 0]) == n.item("Quick Claw")  # the Knock Off failed on the flag
    bare = nxt._replace(claw_flag=jnp.zeros_like(nxt.claw_flag))
    assert int(assert_matches_reference(tables, bare, acts, O).item[1, 0]) == 0
    # The foe switches instead: the newcomer carries nothing.
    gone = assert_matches_reference(tables, nxt, (0, SWITCH), O)
    assert np.asarray(gone.claw_flag).tolist() == [0, 0]


def test_a_flagged_battler_that_faints_at_the_turn_end_keeps_no_flag(tables, names):
    n = names
    s = _tr_duel(tables, n, "Quick Claw", ["Leer"]).with_status(0, 0, Status.POISON)
    s = s.replace_hp(0, 0, 1)
    o = O._replace(quick_claw=jnp.asarray([1, 0], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.hp[0, 0]) == 0 and int(trick_room_of(out)) != 0
    assert np.asarray(out.claw_flag).tolist() == [0, 0]


def test_a_trick_room_turn_without_a_holder_leaves_no_flag(tables, names):
    s = _tr_duel(tables, names, "", ["Tackle"])
    out = assert_matches_reference(tables, s, (0, 0), O._replace(quick_claw=jnp.ones(2, jnp.int8)))
    assert int(trick_room_of(out)) != 0
    assert not np.any(np.asarray(out.claw_flag)) and not np.any(np.asarray(out.custap_flag))


def test_moved_mask_reads_ts_pursued_as_a_bitmask_at_b4(tables, names):
    """``ts_pursued`` is ``1 << pursuer`` per pursuer (BtlCmd_TryPursuit's loop,
    battle_script.c:6884-6935, subscript_pursuit.s:82): at B = 4 pursuer 2 is bit 2, and two
    pursuers are both read."""
    from nuzline.engine.outcome import moved_mask
    from tests.engine.d2_states import four

    s = four(tables, names)
    one = s._replace(ts_pursued=jnp.asarray(1 << 2, s.ts_pursued.dtype))
    assert np.asarray(moved_mask(one)).tolist() == [False, False, True, False]
    two = s._replace(ts_pursued=jnp.asarray((1 << 1) | (1 << 3), s.ts_pursued.dtype))
    assert np.asarray(moved_mask(two)).tolist() == [False, True, False, True]
