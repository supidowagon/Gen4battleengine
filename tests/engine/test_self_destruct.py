"""Selfdestruct and Explosion (``SELF_DESTRUCT``, survey section 6).

* ``effect_script_0007.s``: the Damp check (Damp is an unsupported ability, so a Damp holder
  is refused before the two meet), then ``UpdateMonData OPCODE_SET, BTLSCR_ATTACKER,
  BATTLEMON_CUR_HP, 0`` and only then ``CalcCrit``/``CalcDamage``. The script runs before
  ``BattleControllerPlayer_TryMove`` (accuracy, type chart; bcp.c:3206-3214), so the user
  faints on a miss and into a Ghost too; a move with no target fails at ``HasNoTarget``
  (bcp.c:3190) before the script, and the user survives.
* ``BattleSystem_CalcMoveDamage`` halves ``defenseStat`` for the effect (battle_lib.c:
  6943-6945), after the sandstorm SpD boost and before the stage (:6963-6975).
* The user faints at ``BATTLE_CONTROL_FAINT_AFTER_SELFDESTRUCT`` (bcp.c:3889), within the
  same action as the target, so the action's faint check sees both: both last is a draw,
  played as a loss.
* With its HP at 0 the user no longer counts for ``COUNT_ALIVE_BATTLERS``: its own Cloud
  Nine stops suppressing the weather for its accuracy check and its damage.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.learnsets import load_learnsets
from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import check_coverage
from nuzline.engine.damage import damage_table
from nuzline.engine.ids import AbilityId, EffectId, Stat, Weather
from nuzline.engine.outcome import MAX_HITS, empty_outcome, hit_threshold
from nuzline.engine.step import WINNER_OPPONENT, WINNER_PLAYER, step_events
from nuzline.prep.pool import Candidate, check_pool
from nuzline.prep.problem import levelled_moves, usable_pool
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import (
    A,
    as_species,
    assert_matches_reference,
    roark_party,
    roark_state,
    roark_trainer,
    sample_party,
)

MISS = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
_events = jax.jit(step_events)


def _with_effect(tables, move, effect):
    return tables._replace(move_effect=tables.move_effect.at[move].set(effect))


def _weather(s, weather):
    return s._replace(weather=jnp.asarray(weather, s.weather.dtype))


@pytest.mark.slow
def test_self_destruct_user_faints_on_hit_miss_and_ghost(tables, names):
    sd = names.move("Selfdestruct")
    assert int(tables.move_effect[sd]) == EffectId.SELF_DESTRUCT
    assert int(tables.move_effect[names.move("Explosion")]) == EffectId.SELF_DESTRUCT
    s = roark_state(tables, names, player_moves=["Selfdestruct"])
    acts = A("Selfdestruct", "Leer", s, names)
    slot = names.move_slot(s, 0, "Selfdestruct")
    # A hit: the foe takes damage, the user faints (a death, not the end).
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) < int(s.hp[1, 0])
    assert int(out.player_faints) == 1 and int(out.deaths[0]) == 1 and int(out.over) == 0
    assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot]) - 1
    # A miss: the HP was set to 0 in the script, before the accuracy check.
    out = assert_matches_reference(tables, s, acts, MISS)
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == int(s.hp[1, 0])
    _, events = _events(tables, s, acts, MISS)
    assert int(events["missed"][0]) == 1
    # Into a Ghost: the type chart comes after the script too.
    g = roark_state(tables, names, player_moves=["Selfdestruct"], opp_species="Gastly")
    out = assert_matches_reference(tables, g, A("Selfdestruct", "Leer", g, names), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == int(g.hp[1, 0])
    # No target (HasNoTarget, before the script): the faster foe Self-Destructs and misses,
    # so it faints; the player's Selfdestruct then fails and the player survives, its PP
    # spent and no move recorded.
    both = roark_state(tables, names, player_moves=["Selfdestruct"], opp_moves=["Selfdestruct"],
                       opp_faster=True)  # fmt: skip
    foe_misses = empty_outcome()._replace(hit=jnp.array([[1] * MAX_HITS, [0] * MAX_HITS], jnp.int8))
    acts = A("Selfdestruct", "Selfdestruct", both, names)
    out = assert_matches_reference(tables, both, acts, foe_misses)
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) == int(both.hp[0, 0])
    assert int(out.pp[0, 0, slot]) == int(both.pp[0, 0, slot]) - 1
    assert int(out.last_move[0]) == 0 and int(out.player_faints) == 0
    _, events = _events(tables, both, acts, foe_misses)
    assert int(events["no_target"][0]) == 1


def test_self_destruct_halves_defense(tables, names):
    sd = names.move("Selfdestruct")
    plain = _with_effect(tables, sd, EffectId.PLAIN)
    s = roark_state(tables, names, player_moves=["Selfdestruct"])
    defense = int(s.stats[1, 0, Stat.DEF - 1])
    assert defense % 2 == 1  # the floor of the halving matters
    for stage in (0, 2, -2):
        st = s.with_stages(1, def_=stage)
        half = st._replace(stats=st.stats.at[1, 0, Stat.DEF - 1].set(defense // 2))
        for crit in (False, True):
            ours = np.asarray(damage_table(tables, st, 0, sd, crit)).tolist()
            # Defense halved first, then the stage: the plain hit on half the Defense.
            assert ours == np.asarray(damage_table(plain, half, 0, sd, crit)).tolist()
            assert ours == ref_damage_table(to_numpy(tables), to_ref(st), 0, sd, crit)
    # Special: only defenseStat is halved, never spDefenseStat.
    gun = names.move("Water Gun")
    special = _with_effect(tables, gun, EffectId.SELF_DESTRUCT)
    for crit in (False, True):
        assert (
            np.asarray(damage_table(special, s, 0, gun, crit)).tolist()
            == np.asarray(damage_table(tables, s, 0, gun, crit)).tolist()
        )


def test_kaizo_memento_is_a_special_self_destruct(kaizo_tables, kaizo_names):
    # Kaizo gives Memento effect 7 (Dark, special, 255 power): the halving does not apply.
    memento = kaizo_names.move("Memento")
    assert int(kaizo_tables.move_effect[memento]) == EffectId.SELF_DESTRUCT
    assert int(kaizo_tables.move_category[memento]) == 1


def test_self_destruct_both_last_is_draw(tables, names):
    # The player's last Pokemon KOs the opponent's last: both out at one faint check is
    # a draw, played as a loss.
    s = roark_state(tables, names, player_count=1, opp_count=1, player_moves=["Selfdestruct"])
    s = s.replace_hp(1, 0, 1)
    acts = A("Selfdestruct", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.over) == 1 and int(out.winner) == WINNER_OPPONENT
    assert int(out.player_faints) == 1
    # With a second player Pokemon left, the user's faint is a death and the player wins.
    two = roark_state(tables, names, player_count=2, opp_count=1, player_moves=["Selfdestruct"])
    two = two.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, two, A("Selfdestruct", "Leer", two, names),
                                   empty_outcome())  # fmt: skip
    assert int(out.over) == 1 and int(out.winner) == WINNER_PLAYER
    assert int(out.deaths[0]) == 1 and int(out.player_faints) == 1
    # The opponent's last Self-Destructs into the player's last: the same draw.
    o = roark_state(tables, names, player_count=1, opp_count=1, opp_moves=["Selfdestruct"])
    o = o.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, o, A("Leer", "Selfdestruct", o, names),
                                   empty_outcome())  # fmt: skip
    assert int(out.over) == 1 and int(out.winner) == WINNER_OPPONENT
    # Both sides with Pokemon left: two faints, the battle goes on.
    more = roark_state(tables, names, player_count=2, opp_count=2, opp_moves=["Selfdestruct"])
    more = more.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, more, A("Leer", "Selfdestruct", more, names),
                                   empty_outcome())  # fmt: skip
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert int(out.player_faints) == 1


def test_self_destruct_vs_focus_sash(tables, names):
    # bcp.c:3386-3404: a full-HP Focus Sash holder survives the hit with 1 HP; the user
    # still faints (its HP was set to 0 by the script, not by damage).
    s = roark_state(tables, names, player_moves=["Selfdestruct"], opp_species="Starly")
    sd = names.move("Selfdestruct")
    assert int(damage_table(tables, s, 0, sd, False)[0]) >= int(s.hp[1, 0])
    s = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
    out = assert_matches_reference(tables, s, A("Selfdestruct", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 1


def test_self_destruct_cloud_nine_user_no_longer_suppresses(tables, names):
    # Psyduck (Cloud Nine) in sand: its own Self-Destruct runs with its HP at 0, so the
    # sandstorm acts on that move's accuracy (Sand Veil) and damage (Rock SpD x1.5).
    veil = roark_state(tables, names, player_species="Psyduck", player_ability="Cloud Nine",
                       player_moves=["Selfdestruct", "Scratch"], opp_species="Sandshrew")  # fmt: skip
    veil = _weather(veil, Weather.SANDSTORM)
    assert int(hit_threshold(tables, veil, A("Selfdestruct", "Leer", veil, names))[0]) == 80
    assert int(hit_threshold(tables, veil, A("Scratch", "Leer", veil, names))[0]) == 100
    # A special Self-Destruct (Kaizo's Memento is one) into Rock-type Geodude in sand.
    s = roark_state(tables, names, player_species="Psyduck", player_ability="Cloud Nine",
                    player_moves=["Confusion"])  # fmt: skip
    s = _weather(s, Weather.SANDSTORM)
    confusion = names.move("Confusion")
    damp = as_species(tables, s, 0, 0, int(s.species[0, 0]))  # Damp: no suppression
    boosted = int(damage_table(tables, damp, 0, confusion, False)[15])  # empty_outcome: roll 15
    suppressed = int(damage_table(tables, s, 0, confusion, False)[15])
    assert boosted < suppressed < int(s.hp[1, 0])
    special = _with_effect(tables, confusion, EffectId.SELF_DESTRUCT)
    out = assert_matches_reference(special, s, A("Confusion", "Leer", s, names), empty_outcome())
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == boosted and int(out.hp[0, 0]) == 0


@pytest.mark.fullset
def test_damp_meets_self_destruct_refused(tables, names):
    # effect_script_0007.s:_038 (Damp stops the move) is not modelled: Damp is an
    # unsupported ability, so a Damp holder on either side refuses the fight.
    psyduck = names.species("Psyduck")
    damp = names.ability("Damp")
    assert names.species_abilities(psyduck)[0] == damp
    assert int(tables.species_abilities[psyduck, 0]) == AbilityId.UNSUPPORTED
    sd = names.move("Selfdestruct")
    base = sample_party(names)
    holder = replace(base[1], species=psyduck, form=0, ability_slot=0)
    user = replace(base[0], moves=(sd, 0, 0, 0))
    report = check_coverage(tables, names, [user, holder, *base[2:]], roark_trainer(),
                            roark_party(names), 0)  # fmt: skip
    assert "Player PSYDUCK (slot 1): Damp (ability UNSUPPORTED)" in report.missing


def test_self_destruct_zero_divisor_refused(tables, names):
    # Defense halved, then -6 (x2/8): a Defense below 8 gives stageDivisor 0
    # (battle_lib.c:6943-6975), a division the engines do not model.
    sd = names.move("Selfdestruct")
    base = sample_party(names)
    user = replace(base[0], moves=(sd, 0, 0, 0))
    opp = roark_party(names)
    geodude = opp[0]
    for defense, refused in ((7, True), (8, False)):
        low = replace(geodude, stats=(30, 20, defense, 10, 10, 10))
        report = check_coverage(tables, names, [user, *base[1:]], roark_trainer(),
                                [low, *opp[1:]], 0)  # fmt: skip
        line = f"Player MONFERNO (slot 0): Selfdestruct against Defense {defense} (move effect UNSUPPORTED)"
        assert (line in report.missing) == refused, report.missing
        # The pool and prep refuse the move the same way.
        cand = Candidate(("party", 0, 0), "P", user, 0, (), ())
        missing = check_pool(tables, names, [cand], opp=[low, *opp[1:]]).missing
        assert (f"P: Selfdestruct against Defense {defense} (current, move effect UNSUPPORTED)"
                in missing) == refused  # fmt: skip
        pool, lines = usable_pool(tables, names, [cand], opp=[low, *opp[1:]])
        assert (pool == []) == refused, lines
    # The opponent's Selfdestruct against a low-Defense player Pokemon.
    foe = replace(geodude, moves=(sd, 0, 0, 0))
    frail = replace(base[0], stats=(30, 20, 5, 10, 10, 10))
    report = check_coverage(tables, names, [frail, *base[1:]], roark_trainer(),
                            [foe, *opp[1:]], 0)  # fmt: skip
    assert any("GEODUDE (slot 0): Selfdestruct against Defense 5" in ln for ln in report.missing)
    # A level-up Selfdestruct (Geodude, level 18) is dropped against a low-Defense foe.
    learnsets = load_learnsets()
    rock = replace(geodude, level=17, moves=(names.move("Tackle"), 0, 0, 0))
    cand = Candidate(("party", 0, 0), "G", rock, 0, (), ())
    for defense, refused in ((7, True), (8, False)):
        low = replace(geodude, stats=(30, 20, defense, 10, 10, 10))
        (learned,) = levelled_moves(tables, [cand], learnsets, 18, opp=[low])
        assert ((18, sd) not in learned) == refused, learned


@pytest.mark.fullset
def test_self_destruct_ai_still_refused(tables, names):
    # The AI's Self-Destruct script paths (Basic_CheckCannotExplode, Expert_Explosion) are
    # K3c: an opponent with it is still refused.
    from nuzline.ai.policy import ai_refusals

    s = roark_state(tables, names, opp_moves=["Selfdestruct"])
    reasons = ai_refusals(tables, s, names)
    assert any("move Selfdestruct are not transcribed" in r for r in reasons), reasons


def test_self_destruct_matches_reference(tables, names):
    s = roark_state(tables, names, player_moves=["Selfdestruct", "Explosion"],
                    opp_moves=["Selfdestruct", "Explosion"])  # fmt: skip
    hurt = s.replace_hp(0, 0, 5).replace_hp(1, 0, 7)
    sand = _weather(s.with_stages(1, def_=-3), Weather.SANDSTORM)
    faster = roark_state(tables, names, player_moves=["Selfdestruct"],
                         opp_moves=["Selfdestruct"], opp_faster=True)  # fmt: skip
    last = roark_state(tables, names, player_count=1, opp_count=1,
                       player_moves=["Selfdestruct"], opp_moves=["Selfdestruct"])  # fmt: skip
    crit = empty_outcome()._replace(crit=jnp.ones((2, MAX_HITS), jnp.int8))
    outcomes = [empty_outcome(), MISS, crit,
                empty_outcome()._replace(hit=jnp.array([[0] * MAX_HITS, [1] * MAX_HITS], jnp.int8))]  # fmt: skip
    for state in (s, hurt, sand, faster, last):
        for p in ("Selfdestruct", "Leer"):
            for o in ("Selfdestruct", "Leer"):
                if p not in [names.move_names[int(m)] for m in state.moves[0, 0]]:
                    continue
                if o not in [names.move_names[int(m)] for m in state.moves[1, 0]]:
                    continue
                for outcome in outcomes:
                    assert_matches_reference(tables, state, A(p, o, state, names), outcome)
