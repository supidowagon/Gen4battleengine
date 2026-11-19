"""Singles follow-up (ledger D2-33): (i) Follow Me redirects a non-Ghost Curse (lib.c:1714-1717),
(ii) a lock move aimed at the user after an Encore ended inside the lock (bcp.c:1885-1888,
lib.c:1709-1719), (iii) a randomized non-Ghost Curse takes the foe (lib.c:1706-1707). Every case
goes through ``assert_matches_reference`` (the JAX and the reference step agree on every field);
the assertions below pin the decomp's values. Each jitted step is over the fast tier's 4 GB."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.enumerate import enumerate_outcomes, outcome_laws
from nuzline.engine.ids import AbilityId, SemiKind, Stat, Status
from nuzline.engine.order import metronome_candidates
from nuzline.engine.outcome import empty_outcome, live_hit_threshold
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from tests.engine.d2_states import as_ghost, with_ability
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()
pytestmark = pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB


def _set(s, field, side, value):
    return s._replace(**{field: getattr(s, field).at[side].set(value)})


def _pp(s, side, slot):
    return int(s.pp[side, int(s.active[side]), slot])


def _hp(s, side):
    return int(s.hp[side, int(s.active[side])])


def _with_hit(outcome, side, value):
    return outcome._replace(hit=outcome.hit.at[side, 0].set(value))


def _curse_state(
    tables, names, user_level=50, foe="Togepi", foe_moves=("Follow Me",), foe_level=50
):
    return make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Curse", "Swords Dance"], user_level)],
        [_mon(names, foe, list(foe_moves), foe_level)],
    )


def _curse_stages(s):
    st = np.asarray(s.stages[0])
    return int(st[Stat.ATK]), int(st[Stat.DEF]), int(st[Stat.SPE])


# --- (i) Follow Me sends a non-Ghost Curse to the Follow Me user ---------------------------


def test_follow_me_redirects_a_non_ghost_curse_and_pressure_counts(tables, names):
    # lib.c:1714-1717: the Curse's defender is the living Follow Me user; DecrementPP's default
    # case adds one for a Pressure defender (bcp.c:2261-2266); UpdateFlagsWhenHit records the move
    # on the defender (bcp.c:4498-4503).
    s = with_ability(_curse_state(tables, names), 1, AbilityId.PRESSURE)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _curse_stages(out) == (1, 1, -1)
    assert _pp(s, 0, 0) - _pp(out, 0, 0) == 2
    assert int(out.move_hit[1]) == names.move("Curse")


def test_follow_me_without_pressure_costs_one_pp_but_records_the_curse(tables, names):
    s = _curse_state(tables, names)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _curse_stages(out) == (1, 1, -1)
    assert _pp(s, 0, 0) - _pp(out, 0, 0) == 1
    assert int(out.move_hit[1]) == names.move("Curse")


def test_a_slower_follow_me_leaves_the_curse_on_the_user(tables, names):
    # Sleep Talk (priority 0) calling Follow Me runs after the faster Curse: nothing is set yet
    # (the user's own defender, 1 PP, no record on the foe).
    s = _curse_state(tables, names, user_level=60, foe_moves=("Sleep Talk", "Follow Me"))
    s = with_ability(s, 1, AbilityId.PRESSURE).with_status(1, 0, Status.SLEEP, 5)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _curse_stages(out) == (1, 1, -1)
    assert _pp(s, 0, 0) - _pp(out, 0, 0) == 1
    assert int(out.move_hit[1]) == 0


def test_a_ghost_users_curse_aims_at_the_foe_with_or_without_follow_me(tables, names):
    # the control: the Ghost Curse takes its chosen foe anyway, so Follow Me changes nothing
    s = as_ghost(_curse_state(tables, names), 0)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.curse[1]) != 0
    assert _curse_stages(out) == (0, 0, 0)


# --- (iii) a randomized non-Ghost Curse takes the foe ---------------------------------------


def _encored_curse(tables, names, foe="Geodude", foe_moves=("Harden",), level=50, user_level=60):
    s = _curse_state(tables, names, user_level, foe, foe_moves, level)
    s = _set(s, "encore_move", 0, names.move("Curse"))
    s = _set(s, "encore_slot", 0, 0)
    return _set(s, "encore_turns", 0, 3)


def test_encored_curse_takes_the_foe_and_pressure_counts(tables, names):
    # bcp.c:1869-1873 randomizes, lib.c:1706-1707 gives the living foe; moveTemp is the Curse, so a
    # Pressure foe makes it cost 2 (bcp.c:2261-2266); the foe's moveHit is the Curse.
    s = _encored_curse(tables, names, level=90, user_level=30)  # the foe acts first
    s = with_ability(s, 1, AbilityId.PRESSURE)
    out = assert_matches_reference(tables, s, (1, 0), O)  # chose Swords Dance: the Encore wins
    assert _curse_stages(out) == (1, 1, -1)
    assert _pp(s, 0, 0) - _pp(out, 0, 0) == 2
    assert int(out.move_hit[1]) == names.move("Curse")


def test_encored_curse_misses_a_semi_invulnerable_foe(tables, names):
    # the foe is mid-Fly (it strikes later in the turn): the Curse aims at it, so
    # CheckMoveHitOverrides (bcp.c:3034-3041) misses it and the side effect does not fire
    # (lib.c:1492-1505); a user with No Guard hits through (bcp.c:3016-3022)
    s = _encored_curse(tables, names, foe_moves=("Fly",))
    s = _set(s, "locked_move", 1, names.move("Fly"))
    s = _set(s, "semi_invulnerable", 1, SemiKind.AIRBORNE)
    out = assert_matches_reference(tables, s, (1, 0), O)
    assert _curse_stages(out) == (0, 0, 0)
    sure = with_ability(s, 0, AbilityId.NO_GUARD)
    out = assert_matches_reference(tables, sure, (1, 0), O)
    assert _curse_stages(out) == (1, 1, -1)


def test_encored_curse_with_no_target_does_nothing(tables, names):
    # a faster foe's Explosion KOs it (it has a bench): BATTLER_NONE (lib.c:1706-1707), the PP is
    # 1 (no defender, bcp.c:2238) and subscript_no_target.s runs
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Curse", "Swords Dance"], 30)],
        [_mon(names, "Geodude", ["Explosion"], 90), _mon(names, "Geodude", ["Tackle"], 5)],
    )
    s = with_ability(s, 1, AbilityId.PRESSURE)
    s = s._replace(hp=s.hp.at[0, 0].set(9000), max_hp=s.max_hp.at[0, 0].set(9000))
    s = _set(s, "encore_move", 0, names.move("Curse"))
    s = _set(s, "encore_slot", 0, 0)
    s = _set(s, "encore_turns", 0, 3)
    out = assert_matches_reference(tables, s, (1, 0), O)
    assert _hp(out, 1) == 0
    assert _curse_stages(out) == (0, 0, 0)
    assert _pp(s, 0, 0) - _pp(out, 0, 0) == 1


def _asleep_talker(tables, names, foe_moves=("Harden",), foe_level=40):
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Sleep Talk", "Curse"], 30)],
        [_mon(names, "Geodude", list(foe_moves), foe_level)],
    )
    return s.with_status(0, 0, Status.SLEEP, 5)


def test_sleep_talk_calling_curse_takes_the_foe_and_stores_it(tables, names):
    # GoToMoveScript randomizes (script.c:2120) and stores the defender (:2123): the foe
    s = _asleep_talker(tables, names, foe_level=90)  # the foe acts first
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _curse_stages(out) == (1, 1, -1)
    assert int(out.chosen_target[0]) == 1
    assert int(out.move_hit[1]) == names.move("Curse")


def test_sleep_talk_calling_curse_misses_a_semi_invulnerable_foe(tables, names):
    s = _asleep_talker(tables, names, foe_moves=("Fly",))
    s = _set(s, "locked_move", 1, names.move("Fly"))
    s = _set(s, "semi_invulnerable", 1, SemiKind.AIRBORNE)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _curse_stages(out) == (0, 0, 0)


# --- (ii) a lock reads the stored target ----------------------------------------------------


def _locked_rollout(
    tables, names, foe="Geodude", foe_moves=("Harden",), stored=0, user_level=50, foe_level=50
):
    """Battler 0 is locked into Rollout (count 3), its stored target ``stored``."""
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Swords Dance", "Rollout"], user_level)],
        [_mon(names, foe, list(foe_moves), foe_level)],
    )
    s = _set(s, "locked_move", 0, names.move("Rollout"))
    s = _set(s, "rollout_count", 0, 3)
    return _set(s, "chosen_target", 0, stored)


def test_locked_rollout_reads_the_stored_user(tables, names):
    s = _locked_rollout(tables, names)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _hp(out, 0) < _hp(s, 0)
    assert _hp(out, 1) == _hp(s, 1)
    # stored on the foe: the usual hit
    foe = assert_matches_reference(tables, _locked_rollout(tables, names, stored=1), (0, 0), O)
    assert _hp(foe, 1) < _hp(s, 1)
    assert _hp(foe, 0) == _hp(s, 0)


def test_the_users_own_evasion_decides_the_lock_hit(tables, names):
    # CheckMoveHitAccuracy reads the defender's evasion stage (bcp.c:2874): +6 on the user takes
    # Rollout's 90 % to 30 %, so a class-50 roll that hits at the foe's threshold misses
    s = _locked_rollout(tables, names)
    roll = _with_hit(O, 0, 50)
    hits = assert_matches_reference(tables, s, (0, 0), roll)
    assert _hp(hits, 0) < _hp(s, 0)
    s = s._replace(stages=s.stages.at[0, Stat.EVA].set(6))
    miss = assert_matches_reference(tables, s, (0, 0), roll)
    assert _hp(miss, 0) == _hp(s, 0)
    assert _hp(miss, 1) == _hp(s, 1)
    # the foe's evasion is not read for it: +6 there changes nothing
    foe_eva = _locked_rollout(tables, names)
    foe_eva = foe_eva._replace(stages=foe_eva.stages.at[1, Stat.EVA].set(6))
    kept = assert_matches_reference(tables, foe_eva, (0, 0), roll)
    assert _hp(kept, 0) < _hp(foe_eva, 0)


def test_a_faster_follow_me_turns_the_lock_to_the_foe(tables, names):
    # lib.c:1714-1717 comes before the stored target
    s = _locked_rollout(tables, names, foe="Togepi", foe_moves=("Follow Me",))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _hp(out, 1) < _hp(s, 1)
    assert _hp(out, 0) == _hp(s, 0)


def test_a_faster_encore_turns_the_lock_to_the_foe(tables, names):
    # the Encore branch randomizes (bcp.c:1869-1873): the foe
    s = _locked_rollout(tables, names, foe_moves=("Encore",), user_level=30, foe_level=80)
    s = _set(s, "last_move", 0, names.move("Rollout"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.encore_move[0]) == names.move("Rollout")
    assert _hp(out, 1) < _hp(s, 1)
    assert _hp(out, 0) == _hp(s, 0)


def test_a_faster_protect_does_not_stop_a_lock_that_hits_its_user(tables, names):
    # the Protect check reads the defender's flag (bcp.c:3007): the user's own, never set
    s = _locked_rollout(tables, names, foe_moves=("Protect",))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert _hp(out, 0) < _hp(s, 0)
    assert int(out.protect_count[1]) == 1  # it did protect (a first actor)


def test_the_stored_user_survives_two_turns_of_an_ended_encore(tables, names):
    # turn 1: Swords Dance is selected (the user is stored, bcp.c:4363-4367), the faster foe's
    # Encore on Rollout (1 PP left) takes the action over and hits the foe; the encored move has
    # no PP left at the turn's end, so the Encore ends (bcp.c:1598-1601) inside the lock; turn 2:
    # the locked Rollout reads the stored user (bcp.c:1885-1888, lib.c:1711).
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Swords Dance", "Rollout"], 30)],
        [_mon(names, "Geodude", ["Encore", "Harden"], 80)],
    )
    s = _set(s, "last_move", 0, names.move("Rollout"))
    s = s._replace(pp=s.pp.at[0, 0, 1].set(1))
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.chosen_target[0]) == 0
    assert int(one.locked_move[0]) == names.move("Rollout")
    assert int(one.encore_move[0]) == 0
    assert _hp(one, 1) < _hp(s, 1)
    assert _hp(one, 0) == _hp(s, 0)
    two = assert_matches_reference(tables, one, (0, 1), O)
    assert _hp(two, 0) < _hp(one, 0)
    assert _hp(two, 1) == _hp(one, 1)


def test_metronome_calling_rollout_stores_the_foe_for_the_next_lock_turn(tables, names):
    # GoToMoveScript stores the called move's defender (script.c:2123): the foe, so the lock
    # turn that follows hits the foe, not the user the Metronome selection stored
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Metronome"], 50)],
        [_mon(names, "Geodude", ["Harden"], 50)],
    )
    cands = np.flatnonzero(np.asarray(metronome_candidates(tables, s, 0)))
    rank = int(np.flatnonzero(cands == names.move("Rollout"))[0])
    outcome = O._replace(call=O.call.at[0].set(rank))
    one = assert_matches_reference(tables, s, (0, 0), outcome)
    assert int(one.locked_move[0]) == names.move("Rollout")
    assert int(one.chosen_target[0]) == 1
    assert _hp(one, 1) < _hp(s, 1)
    two = assert_matches_reference(tables, one, (0, 0), O)
    assert _hp(two, 1) < _hp(one, 1)
    assert _hp(two, 0) == _hp(one, 0)


def test_a_fly_strike_after_an_ended_encore_hits_its_user(tables, names):
    # turn 1: Fly (1 PP) is encored over a Swords Dance selection and charges; the Encore ends at
    # the turn's end (no PP); turn 2: the strike reads the stored user, and the user's own
    # semi-invulnerable bit is cleared before the accuracy check (effect_script_0156.s:19-25)
    s = make_state(
        tables,
        [_mon(names, "Bulbasaur", ["Swords Dance", "Fly"], 30)],
        [_mon(names, "Geodude", ["Encore", "Harden"], 80)],
    )
    s = _set(s, "last_move", 0, names.move("Fly"))
    s = s._replace(pp=s.pp.at[0, 0, 1].set(1))
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.locked_move[0]) == names.move("Fly")
    assert int(one.semi_invulnerable[0]) == SemiKind.AIRBORNE
    assert int(one.encore_move[0]) == 0
    two = assert_matches_reference(tables, one, (0, 1), O)
    assert _hp(two, 0) < _hp(one, 0)
    assert _hp(two, 1) == _hp(one, 1)
    assert int(two.semi_invulnerable[0]) == 0
    assert int(two.locked_move[0]) == 0


def test_a_self_lock_keeps_its_accuracy_roll_whole_in_the_laws(tables, names):
    # the lock may hit the user (evasion +6: 30 %) or, after a faster Encore or Follow Me, the
    # foe (90 %): the enumeration cuts its accuracy roll at no class (outcome.self_lock), and the
    # mass of the rows that hurt the user is the user-target threshold's
    s = _locked_rollout(tables, names)
    s = s._replace(stages=s.stages.at[0, Stat.EVA].set(6))
    acts = jnp.asarray([0, 0], jnp.int32)
    laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
    assert bool(laws.hit_whole[0]) and not bool(laws.hit_whole[1])
    rows, probs = enumerate_outcomes(laws)
    assert sorted(set(np.asarray(rows.hit)[:, 0, 0].tolist())) == list(range(1, 101))
    fn = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
    hurt = []
    chunk = 1024
    for start in range(0, len(probs), chunk):
        idx = np.arange(start, min(start + chunk, len(probs)))
        pad = np.concatenate([idx, np.full(chunk - len(idx), idx[0])])
        part = jax.tree.map(lambda f, pad=pad: jnp.asarray(np.asarray(f)[pad]), rows)
        out = fn(tables, s, acts, part)
        hurt.append(np.asarray(out.hp[:, 0, 0] < s.hp[0, 0])[: len(idx)])
    hurt = np.concatenate(hurt)
    # the user-target threshold (its own evasion), not the foe's 90
    rate = int(live_hit_threshold(tables, s, 0, names.move("Rollout"), target=0))
    assert 0 < rate < 90
    assert abs(float(np.sum(np.asarray(probs)[hurt])) - rate / 100) < 1e-9


# --- Lucky Chant on the hit's own defender (review of D2-33) --------------------------------


def _with_crit(outcome, side):
    # crit value 1 crits at any odds ((1 - 1) % odds == 0, outcome.crit_lands)
    return outcome._replace(crit=outcome.crit.at[side, 0].set(1))


def test_lucky_chant_on_the_users_side_blocks_a_lock_that_hits_its_user(tables, names):
    # CalcCrit passes the defender's side conditions (battle_script.c:2158-2163) and
    # BattleSystem_CalcCriticalMulti blocks a crit under Lucky Chant (battle_lib.c:7137): the
    # lock's defender is its user, so the user's side's Lucky Chant counts, read at the hit
    s = _locked_rollout(tables, names)
    plain = assert_matches_reference(tables, s, (0, 0), O)
    crit = assert_matches_reference(tables, s, (0, 0), _with_crit(O, 0))
    assert _hp(crit, 0) < _hp(plain, 0)  # the control: the crit lands without Lucky Chant
    chant = _set(s, "lucky_chant", 0, 5)
    blocked = assert_matches_reference(tables, chant, (0, 0), _with_crit(O, 0))
    assert _hp(blocked, 0) == _hp(plain, 0)
    # the foe's side's Lucky Chant is not the defender's: the crit lands
    foe_chant = _set(s, "lucky_chant", 1, 5)
    lands = assert_matches_reference(tables, foe_chant, (0, 0), _with_crit(O, 0))
    assert _hp(lands, 0) == _hp(crit, 0)


def test_lucky_chant_on_the_foes_side_blocks_a_lock_turned_to_the_foe(tables, names):
    # a faster Follow Me turns the lock to the foe (lib.c:1714-1717): the foe's side's Lucky
    # Chant blocks the crit (battle_lib.c:7137), which the turn-start cut left in (self_lock)
    s = _locked_rollout(tables, names, foe="Togepi", foe_moves=("Follow Me",))
    plain = assert_matches_reference(tables, s, (0, 0), O)
    crit = assert_matches_reference(tables, s, (0, 0), _with_crit(O, 0))
    assert _hp(crit, 1) < _hp(plain, 1)
    chant = _set(s, "lucky_chant", 1, 5)
    blocked = assert_matches_reference(tables, chant, (0, 0), _with_crit(O, 0))
    assert _hp(blocked, 1) == _hp(plain, 1)
    assert _hp(blocked, 0) == _hp(s, 0)
