"""Chatter (G-70): the confusion roll of ``subscript_chatter.s`` (battle_script.c:8397-8436), its
per-battler Chatot cry parameter and the exact enumeration of its draw; each case against the
reference engine."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.enumerate import _LAWS, outcome_laws
from nuzline.engine.ids import CHATTER_CHANCES, SPECIES_CHATOT, EffectId
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import A, assert_matches_reference, drop_programs
from tests.engine.test_pivot import _mon


def _chat(tables, names, *, cry=0, enemy=False, species="Chatot", mine=("Chatter", "Leer")):
    """A ``species`` with Chatter (a Chatot by default; on side 1 with ``enemy``) against a bulky
    Geodude with Leer and Tackle (400 HP: no faint, so the target is alive for the confusion);
    ``cry`` is the player's Chatot cry parameter."""
    n = names
    user = _mon(n, species, list(mine), 30)
    foe = _mon(n, "Geodude", ["Leer", "Tackle"], 50)
    a, b = ([foe], [user]) if enemy else ([user, _mon(n, "Bulbasaur", ["Tackle"], 30)], [foe])
    s = make_state(tables, a, b, chatot_cry=cry)
    # the Geodude's side: 400 HP
    side = 0 if enemy else 1
    return s._replace(hp=s.hp.at[side, 0].set(400), max_hp=s.max_hp.at[side, 0].set(400))


def _roll(**fields):
    o = empty_outcome()
    return o._replace(**{k: jnp.asarray(v, jnp.int8) for k, v in fields.items()})


def test_chatter_is_mapped_and_its_table_chance_is_unread(tables, names, kaizo_tables, kaizo_names):
    from tools.effect_map import DECOMP_EFFECT_TO_ID

    assert DECOMP_EFFECT_TO_ID[267] == (EffectId.CHATTER, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        m = n.move("Chatter")
        assert int(t.move_effect[m]) == EffectId.CHATTER
        assert int(t.move_effect_chance[m]) == 0  # Kaizo's 50 is never read (G-70)
        assert int(t.move_effect[m]) == int(t.move_effect[448])
    assert SPECIES_CHATOT == names.species("Chatot")


def test_an_enemy_chatot_confuses_exactly_when_the_roll_passes(tables, names):
    drop_programs()  # the programs of the earlier tests stay resident otherwise
    s = _chat(tables, names, enemy=True)
    acts = A("Leer", "Chatter", s, names)
    hit = assert_matches_reference(tables, s, acts, _roll(chatter=(0, 1), duration=(2, 4)))
    assert int(hit.confusion[0]) in (3, 4) and int(hit.confusion[1]) == 0  # 4, or 3 once it moved
    miss = assert_matches_reference(tables, s, acts, _roll(chatter=(0, 0), duration=(2, 4)))
    assert int(miss.confusion[0]) == 0
    # The law: an enemy's cry is always 0: 1 % (battle_system.c:915-924, chatot_cry.c:13-25).
    laws = outcome_laws(tables, s, jnp.asarray(acts))
    assert _LAWS["chatter"](laws, 1) == ([0, 1], [99, 1], 100)
    assert int(laws.chatter_chance[1]) == 0 and bool(laws.reads.chatter[1])


@pytest.mark.parametrize("cry", [0, 1, 2])
def test_a_players_chatot_uses_its_cry_parameter(tables, names, cry):
    s = _chat(tables, names, cry=cry)
    assert int(s.chatot_cry[0]) == cry and int(s.chatot_cry[1]) == 0
    acts = A("Chatter", "Leer", s, names)
    laws = outcome_laws(tables, s, jnp.asarray(acts))
    chance = CHATTER_CHANCES[cry]
    assert _LAWS["chatter"](laws, 0) == ([0, 1], [99 - chance, chance + 1], 100)
    hit = assert_matches_reference(tables, s, acts, _roll(chatter=(1, 0), duration=(3, 2)))
    assert int(hit.confusion[1]) in (2, 3)  # the counter 3, or 2 once the target moved
    assert chance in (0, 10, 30)


def test_only_an_untransformed_chatot_confuses(tables, names):
    flying = _chat(tables, names, species="Starly")
    laws = outcome_laws(tables, flying, jnp.asarray(A("Chatter", "Leer", flying, names)))
    assert not bool(laws.reads.chatter[0]) and not bool(np.asarray(laws.relevant.chatter).any())
    out = assert_matches_reference(
        tables,
        flying,
        A("Chatter", "Leer", flying, names),
        _roll(chatter=(1, 1), duration=(3, 3)),
    )
    assert int(out.confusion[1]) == 0  # Chatter by a non-Chatot never confuses
    chatot = _chat(tables, names)
    copied = chatot._replace(transformed=chatot.transformed.at[0].set(1))
    out = assert_matches_reference(
        tables,
        copied,
        A("Chatter", "Leer", copied, names),
        _roll(chatter=(1, 1), duration=(3, 3)),
    )
    assert int(out.confusion[1]) == 0  # Transformed (battle_script.c:8404)


def test_chatter_confusion_stops_on_a_confused_target_a_substitute_or_safeguard(tables, names):
    s = _chat(tables, names)
    acts = A("Chatter", "Leer", s, names)
    roll = _roll(chatter=(1, 1), duration=(5, 5))
    confused = s._replace(confusion=s.confusion.at[1].set(2))
    assert int(assert_matches_reference(tables, confused, acts, roll).confusion[1]) <= 2
    sub = s._replace(substitute_hp=s.substitute_hp.at[1].set(200))
    assert int(assert_matches_reference(tables, sub, acts, roll).confusion[1]) == 0
    safe = s._replace(safeguard=s.safeguard.at[1].set(3))
    assert int(assert_matches_reference(tables, safe, acts, roll).confusion[1]) == 0
    plain = assert_matches_reference(tables, s, acts, roll)
    assert int(plain.confusion[1]) in (4, 5)


@pytest.mark.parametrize(("cry", "enemy"), [(2, False), (0, True), (1, False)])
def test_chatter_enumeration_equals_bruteforce(tables, names, cry, enemy):
    from tests.engine.test_enumerate import assert_equals_bruteforce

    drop_programs()
    s = _chat(tables, names, cry=cry, enemy=enemy)
    acts = A("Leer", "Chatter", s, names) if enemy else A("Chatter", "Tackle", s, names)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width <= brute


def test_the_chatot_cry_is_a_player_constant_checked_at_the_start(tables, names):
    n = names
    team = [_mon(n, "Chatot", ["Chatter"], 30)]
    foe = [_mon(n, "Bulbasaur", ["Leer"], 30)]
    with pytest.raises(ValueError, match="chatot_cry"):
        make_state(tables, team, foe, chatot_cry=3)
    s = make_state(tables, team, foe, chatot_cry=2)
    assert s.chatot_cry.tolist() == [2, 0]


def test_a_player_chatot_that_knows_chatter_needs_its_cry(tables, names):
    """Coverage (G-70): the player's Chatter confusion chance is the save's Chatot cry: unknown
    (None) refuses a Chatot that knows or can learn Chatter; any parameter 0-2 is fine; an enemy
    Chatter is never refused."""
    from nuzline.engine.coverage import check_coverage
    from tests.engine.helpers import roark_party, roark_trainer

    team = [_mon(names, "Chatot", ["Chatter", "Leer"], 30)]
    opp = roark_party(names)
    refusal = "Chatter's confusion chance needs the save's Chatot cry"

    def refused(cry):
        report = check_coverage(tables, names, team, roark_trainer(), opp, 0, chatot_cry=cry)
        return any(refusal in line for line in report.missing)

    assert refused(None)
    assert not any(refused(cry) for cry in (0, 1, 2))
    plain = [_mon(names, "Chatot", ["Leer", "Fury Attack"], 30)]
    report = check_coverage(tables, names, plain, roark_trainer(), opp, 0)
    assert not any(refusal in line for line in report.missing)
