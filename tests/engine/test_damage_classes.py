"""Damage-class compression (rules §10.3 item 5): per hit, the ``(crit, roll)`` pairs
that give the same crit flag and the same damage merge into one class, with summed
probability, when no earlier event of the turn can change that hit's damage inputs
(``enumerate.damage_inputs_can_change``). Every case equals the brute-force enumeration
(``outcome_laws(..., loose=True)``, which never compresses) merged by state and
observation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st

import nuzline.engine.enumerate as enumerate_module
from nuzline.data.spec import PokemonSpec
from nuzline.engine.damage import hit_damage_table
from nuzline.engine.enumerate import (
    INPUT_CLAUSE_ABILITIES,
    INPUT_CLAUSE_EFFECTS,
    INPUT_CLAUSE_ITEMS,
    INPUT_NEUTRAL_ABILITIES,
    INPUT_NEUTRAL_EFFECTS,
    INPUT_NEUTRAL_ITEMS,
    NON_TABLE_DAMAGE_EFFECTS,
    TABLE_DAMAGE_EFFECTS,
    SamplingOnly,
    damage_class_law,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat, Status
from nuzline.engine.state import BattleState, make_state
from tests.engine.helpers import A, roark_state
from tests.engine.test_enumerate import (
    BRUTE_MAX,
    HYPOTHESIS,
    assert_equals_bruteforce,
    laws_of,
    order_actions,
    turn_children,
)
from tests.strategies import outcome_law_state

_SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe


def _state(tables, names, player, opp, order="player", setup=()):
    """Venonat (``player``) against Cranidos (``opp``), slot 0 each; ``order`` says who is
    faster ("player", "opp") or "tie". ``setup`` names extra changes (below)."""
    s = roark_state(
        tables,
        names,
        player_species="Venonat",
        player_moves=(player,),
        opp_species="Cranidos",
        opp_moves=(opp,),
    )
    # Sturdy (no Mold Breaker, no contact ability) unless a setup says otherwise.
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.STURDY))
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    for name in setup:
        if name == "flame_body":
            s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.FLAME_BODY))
        elif name == "guts_asleep":
            s = s._replace(
                ability=s.ability.at[0, 0].set(AbilityId.GUTS),
                status=s.status.at[0, 0].set(Status.SLEEP),
                status_counter=s.status_counter.at[0, 0].set(1),
            )
        elif name == "focus_sash":
            # A full-HP sash holder whose HP sits inside the damage range.
            s = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
            table = np.asarray(hit_damage_table(tables, s, 0, names.move(player)))
            hp = int(table[0, 8])
            s = s._replace(hp=s.hp.at[1, 0].set(hp), max_hp=s.max_hp.at[1, 0].set(hp))
        elif name == "strong":  # Atk 400, no crits: a wide damage range, a small brute force
            s = s._replace(
                stats=s.stats.at[0, 0, 0].set(400), lucky_chant=s.lucky_chant.at[:].set(5)
            )
        elif name == "chilan_two_hits":
            # A Chilan Berry (any Normal hit) with HP two middle rolls can take.
            s = s._replace(item=s.item.at[1, 0].set(names.item("Chilan Berry")))
            table = np.asarray(hit_damage_table(tables, s, 0, names.move(player)))
            hp = int(table[0, 8]) * 2
            s = s._replace(
                hp=s.hp.at[1, 0].set(hp), max_hp=s.max_hp.at[1, 0].set(max(hp, int(s.max_hp[1, 0])))
            )
        elif name == "sash_two_hits":
            # A full-HP Focus Sash holder that two hits can KO.
            s = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
            table = np.asarray(hit_damage_table(tables, s, 0, names.move(player)))
            hp = int(table[0, 15]) + int(table[0, 0])
            s = s._replace(hp=s.hp.at[1, 0].set(hp), max_hp=s.max_hp.at[1, 0].set(hp))
        elif name == "lucky_chant_opp":  # no crits for the opponent: a smaller brute force
            s = s._replace(lucky_chant=s.lucky_chant.at[0].set(5))
        elif name == "herb":  # a White Herb on the player
            s = s._replace(item=s.item.at[0, 0].set(names.item("White Herb")))
        elif name == "herb_atk_low":  # ... restoring the Attack the player's hits read
            s = s._replace(
                item=s.item.at[0, 0].set(names.item("White Herb")),
                stages=s.stages.at[0, Stat.ATK].set(-1),
            )
        elif name == "herb_opp_def_low":  # ... restoring the Defense they hit
            s = s._replace(
                item=s.item.at[1, 0].set(names.item("White Herb")),
                stages=s.stages.at[1, Stat.DEF].set(-1),
            )
        elif name == "low_hp":  # Flail's power band moves with its user's HP
            s = s._replace(hp=s.hp.at[0, 0].set(int(s.max_hp[0, 0]) * 2 // 5))
        else:
            raise ValueError(name)
    return s


# (player move, opponent move, order, setup, compressed player hits per order branch)
# The last column is the expected ``compress[branch, 0, :hits]`` (branch 0 alone without
# a tie).
EXACT_CASES = [
    # Multi-hit (Double Kick, two fixed hits); Leer lowers the player's Defense, which is
    # not an input of the player's own hits.
    ("Double Kick", "Leer", "player", (), [[True, True]]),
    ("Double Kick", "Leer", "opp", (), [[True, True]]),
    # A Focus Sash holder at full HP inside the damage range.
    ("Tackle", "Leer", "player", ("focus_sash",), [[True]]),
    # Double Hit (Normal, two hits) at Atk 400 into a Chilan Berry (halves the first Normal
    # hit, then is gone) and into a full-HP Focus Sash that two hits can KO, both orders.
    ("Double Hit", "Leer", "player", ("strong", "chilan_two_hits"), [[True, True]]),
    ("Double Hit", "Leer", "opp", ("strong", "chilan_two_hits"), [[True, True]]),
    ("Double Hit", "Leer", "player", ("strong", "sash_two_hits"), [[True, True]]),
    ("Double Hit", "Leer", "opp", ("strong", "sash_two_hits"), [[True, True]]),
    # A high-crit move: crit odds 1 in 8, so the class weights are not those of Tackle.
    ("Slash", "Leer", "player", (), [[True]]),
    # Inputs that change before the hit: compression off for that hit.
    ("Tackle", "Growl", "opp", (), [[False]]),
    ("Tackle", "Growl", "player", (), [[True]]),
    ("Tackle", "Growl", "tie", (), [[True], [False]]),
    ("Tackle", "Reflect", "opp", (), [[False]]),
    ("Water Gun", "Reflect", "opp", (), [[True]]),
    ("Water Gun", "Light Screen", "opp", (), [[False]]),
    ("Tackle", "Defense Curl", "opp", (), [[False]]),
    ("Tackle", "Swords Dance", "opp", (), [[True]]),
    ("Tackle", "Ember", "opp", ("lucky_chant_opp",), [[False]]),
    ("Water Gun", "Ember", "opp", ("lucky_chant_opp",), [[True]]),
    ("Double Kick", "Leer", "player", ("flame_body",), [[True, False]]),
    ("Flail", "Tackle", "opp", ("low_hp", "lucky_chant_opp"), [[False]]),
    ("Flail", "Tackle", "player", ("low_hp", "lucky_chant_opp"), [[True]]),
    ("Tackle", "Leer", "player", ("guts_asleep",), [[False]]),
    ("Tackle", "Tackle", "opp", ("lucky_chant_opp",), [[True]]),
    # White Herb (Task 12): Leer lowers the herb holder's Defense, which the player's own
    # hits do not read, so they stay compressed; a lowered Attack (the player's herb) or
    # Defense (the foe's herb) is restored after the first hit, so the second is not.
    ("Double Kick", "Leer", "player", ("herb",), [[True, True]]),
    ("Double Kick", "Leer", "opp", ("herb",), [[True, True]]),
    ("Double Kick", "Leer", "player", ("herb_atk_low",), [[True, False]]),
    ("Double Kick", "Leer", "opp", ("herb_atk_low",), [[False, False]]),
    ("Double Kick", "Leer", "player", ("herb_opp_def_low",), [[True, False]]),
    # Tackle changes no stage: only the foe's Defense below 0 at the start keeps the second
    # hit out of the classes (the start term of the defender's herb clause).
    ("Double Kick", "Tackle", "player", ("herb_opp_def_low",), [[True, False]]),
    ("Double Kick", "Growl", "opp", ("herb",), [[False, False]]),
]


@pytest.mark.slow
@pytest.mark.parametrize(("player", "opp", "order", "setup", "expected"), EXACT_CASES)
def test_damage_classes_equal_bruteforce(tables, names, player, opp, order, setup, expected):
    s = _state(tables, names, player, opp, order, setup)
    acts = A(player, opp, s, names)
    laws = laws_of(tables, s, acts)
    hits = len(expected[0])
    branches = [0, 1] if order == "tie" else [0]
    if order == "tie":
        assert bool(laws.relevant.speed_tie)
        # Branch 0 (speed_tie 0) is the one where the player moves first.
        assert int(laws.reads.first[0]) == 0
    got = [np.asarray(laws.compress)[t, 0, :hits].tolist() for t in branches]
    assert got == expected
    assert not np.asarray(laws.compress)[:, 0, hits:].any()
    width, brute = assert_equals_bruteforce(tables, s, acts)
    if any(any(row) for row in expected):
        assert width < brute


def test_class_law_merges_equal_damage(tables, names):
    # Tackle into Cranidos: the classes are (crit, damage) pairs, each weighing the sum of
    # its members, with its lowest roll as the representative.
    s = _state(tables, names, "Tackle", "Leer")
    laws = laws_of(tables, s, A("Tackle", "Leer", s, names))
    values, weights, den, crits, rolls = damage_class_law(laws, 0)
    table = np.asarray(laws.damage)[0, 0]
    odds = int(laws.crit_variants[0, 0])
    assert den == odds * 16
    assert values == list(range(len(weights)))
    keys = {(c, int(table[c, r])) for c in (0, 1) for r in range(16)}
    assert len(weights) == len(keys)
    for v, w, c, r in zip(values, weights, crits, rolls):
        members = [q for q in range(16) if table[c, q] == table[c, r]]
        assert r == members[0]
        assert w == len(members) * (1 if c else odds - 1)
    assert sum(weights) == den


def _forced(laws):
    return laws._replace(compress=np.ones_like(np.asarray(laws.compress)))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("player", "opp", "order", "setup"),
    [
        # Growl lowers the Attack the player's Slash reads (Slash, not Tackle: Tackle's
        # 3-4 damage keeps the same classes after the change, so forcing would not show).
        ("Slash", "Growl", "opp", ()),
        # Iron Defense raises the Defense it hits; Reflect halves it.
        ("Slash", "Iron Defense", "opp", ()),
        ("Slash", "Reflect", "opp", ()),
        # Ember's burn halves it (on 10 % of the rows).
        ("Slash", "Ember", "opp", ("lucky_chant_opp",)),
        # Flame Body burns the attacker on the first hit: the second hit is halved.
        ("Double Kick", "Leer", "player", ("flame_body",)),
        # Cranidos's Tackle moves Flail's power band.
        ("Flail", "Tackle", "opp", ("low_hp", "lucky_chant_opp")),
        # A sleeping Guts user wakes before its move: its status, and the boost, are gone.
        ("Tackle", "Leer", "player", ("guts_asleep",)),
        # A White Herb restores the lowered Attack / Defense after the first hit (Task 12).
        ("Double Kick", "Leer", "player", ("herb_atk_low",)),
        ("Double Kick", "Tackle", "player", ("herb_opp_def_low",)),
    ],
)
def test_compression_disabled_when_inputs_can_change(tables, names, player, opp, order, setup):
    s = _state(tables, names, player, opp, order, setup)
    acts = A(player, opp, s, names)
    laws = laws_of(tables, s, acts)
    assert not np.asarray(laws.compress)[:, 0].all()
    assert_equals_bruteforce(tables, s, acts)
    # Compressing anyway (classes of the turn-start table) is wrong: it merges rolls that
    # deal different damage once the input changed.
    loose = laws_of(tables, s, acts, loose=True)
    brute = turn_children(tables, s, acts, *enumerate_outcomes(loose))
    wrong = turn_children(tables, s, acts, *enumerate_outcomes(_forced(laws)))
    assert set(wrong) != set(brute) or any(abs(wrong[k] - p) > 1e-12 for k, p in brute.items())


def test_compressed_rows_draw_crit_and_roll_together(tables, names):
    s = _state(tables, names, "Double Kick", "Leer")
    laws = laws_of(tables, s, A("Double Kick", "Leer", s, names))
    outcomes, probs, drawn = enumerate_drawn(laws)
    assert (np.asarray(drawn.crit)[:, 0, :2] == np.asarray(drawn.roll)[:, 0, :2]).all()
    assert abs(math.fsum(probs) - 1) < 1e-12
    # Every drawn (crit, roll) is its class representative: the lowest roll of its damage.
    table = np.asarray(laws.damage)[0, 0]
    crit = np.asarray(outcomes.crit)[:, 0, :2]
    roll = np.asarray(outcomes.roll)[:, 0, :2]
    read = np.asarray(drawn.roll)[:, 0, :2]
    for c, r in zip(crit[read], roll[read]):
        assert r == int(np.argmax(table[c] == table[c, r]))


def test_every_damage_input_id_classified():
    # Each later mechanic (Present, Camouflage, Protect, Roar, ...) must classify its ids
    # for damage_inputs_can_change: neutral, or handled by one of its clauses.
    for neutral, clause, enum in (
        (INPUT_NEUTRAL_EFFECTS, INPUT_CLAUSE_EFFECTS, EffectId),
        (INPUT_NEUTRAL_ABILITIES, INPUT_CLAUSE_ABILITIES, AbilityId),
        (INPUT_NEUTRAL_ITEMS, INPUT_CLAUSE_ITEMS, ItemId),
    ):
        assert not set(neutral) & set(clause), enum.__name__
        known = set(neutral) | set(clause) | {enum.UNSUPPORTED}
        assert known == set(enum), (enum.__name__, set(enum) - known)


def test_every_effect_classified_for_table_damage():
    # The attacker's own effect decides whether its hits can be damage classes at all: an
    # allowlist, so a new effect (Present's drawn power, Nature Power's chosen move) is
    # never compressed until it is classified.
    table, other = set(TABLE_DAMAGE_EFFECTS), set(NON_TABLE_DAMAGE_EFFECTS)
    assert not table & other
    assert table | other | {EffectId.UNSUPPORTED} == set(EffectId), set(EffectId) - table - other


def test_non_table_effects_never_compressed(tables, names):
    # Magnitude's drawn power and Pursuit (x2 against a switch) keep their crit and roll.
    for move in ("Magnitude", "Pursuit"):
        s = _state(tables, names, move, "Leer")
        laws = laws_of(tables, s, A(move, "Leer", s, names))
        assert bool(laws.relevant.crit[0, 0])
        assert not np.asarray(laws.compress)[:, 0].any(), move


# Moves and abilities that change (or leave) damage inputs mid-turn, for the generated
# cases below.
INPUT_MOVES = (
    "Tackle",
    "Water Gun",
    "Double Kick",
    "Slash",
    "Flail",
    "Ember",
    "Growl",
    "Leer",
    "Defense Curl",
    "Swords Dance",
    "Reflect",
    "Light Screen",
    "Mud Sport",
    "Thundershock",
    "Metal Claw",
    "Aurora Beam",
    "Absorb",
    "Take Down",
    "Brick Break",
    # E2 batch 4: the clause effects (Swagger's Attack, SmellingSalt's cure) and the defender's
    # power moves.
    "Swagger",
    "SmellingSalt",
    "Grass Knot",
    "False Swipe",
    "Brine",
    "Facade",
)
INPUT_ABILITIES = (
    AbilityId.GUTS,
    AbilityId.FLAME_BODY,
    AbilityId.STATIC,
    AbilityId.ROUGH_SKIN,
    AbilityId.TORRENT,
    AbilityId.BLAZE,
    AbilityId.SIMPLE,
)
INPUT_ITEMS = ("Lum Berry", "White Herb", "Sitrus Berry", "Focus Sash", "Chilan Berry")


@st.composite
def input_state(draw, tables, names) -> BattleState:
    """``outcome_law_state`` with an ``INPUT_MOVES`` move in each active slot 0 most of
    the time, sometimes an ``INPUT_ABILITIES`` ability or an ``INPUT_ITEMS`` item, a status
    sometimes, Lucky Chant on both sides half the time, and equal speeds half the time."""
    s = draw(outcome_law_state(tables, names))
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    move_pp = np.asarray(tables.move_pp)
    for side in (0, 1):
        mon = int(a["active"][side])
        if draw(st.integers(0, 3)) > 0:
            move = names.move(draw(st.sampled_from(INPUT_MOVES)))
            a["moves"][side, mon, 0] = move
            a["pp"][side, mon, 0] = move_pp[move]
            if a["disabled_move"][side] == move:
                a["disabled_move"][side], a["disable_turns"][side] = 0, 0
        if draw(st.integers(0, 2)) == 0:
            a["ability"][side, mon] = draw(st.sampled_from(INPUT_ABILITIES))
        if draw(st.integers(0, 2)) == 0:
            a["item"][side, mon] = names.item(draw(st.sampled_from(INPUT_ITEMS)))
        if draw(st.integers(0, 3)) == 0:
            a["status"][side, mon] = draw(
                st.sampled_from([Status.BURN, Status.SLEEP, Status.POISON])
            )
            a["status_counter"][side, mon] = 1 if a["status"][side, mon] == Status.SLEEP else 0
        if draw(st.integers(0, 2)) == 0:  # a third of max HP or less: pinch and Flail bands
            a["hp"][side, mon] = max(1, int(a["max_hp"][side, mon]) // draw(st.integers(2, 4)))
    if draw(st.booleans()):
        a["lucky_chant"][:] = 5
    if draw(st.booleans()):
        m0, m1 = int(a["active"][0]), int(a["active"][1])
        a["stats"][1, m1, _SPE] = a["stats"][0, m0, _SPE]
        a["stages"][:, Stat.SPE] = 0
        a["tailwind"][:] = 0
    return BattleState(**{name: np.asarray(a[name]) for name in BattleState._fields})


@pytest.mark.slow
@settings(max_examples=150, **HYPOTHESIS)
@given(data=st.data())
def test_damage_classes_equal_bruteforce_generated(tables, names, data):
    s = data.draw(input_state(tables, names))
    acts = data.draw(order_actions(tables, s))
    try:
        brute = enumerate_outcomes(laws_of(tables, s, acts, loose=True))[1]
    except SamplingOnly:
        assume(False)
    assume(len(brute) <= BRUTE_MAX)
    width, _ = assert_equals_bruteforce(tables, s, acts)
    laws = laws_of(tables, s, acts)
    event(f"compressed hits: {int(np.asarray(laws.compress).sum())}")
    event(f"narrower than brute force: {width < len(brute)}")


def _kaizo_spec(names, species, move, level):
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=(names.move(move), 0, 0, 0),
        item=0,
        ability_slot=0,
        gender=1,
        friendship=70,
    )


@pytest.mark.slow
def test_doubleslap_vs_corsola_water_pulse_fits(kaizo_tables, kaizo_names, monkeypatch):
    # Rules §10.2/§10.4: Skitty's DoubleSlap (Kaizo: two fixed hits, 90 %) against Corsola's
    # Water Pulse overflows the cap without damage classes. Skitty is faster (21 against 19
    # in the saves; set here). DoubleSlap does 2-3 (4-6 on a crit) to Corsola.
    t, n = kaizo_tables, kaizo_names
    skitty = _kaizo_spec(n, "Skitty", "DoubleSlap", 16)
    corsola = _kaizo_spec(n, "Corsola", "Water Pulse", 14)
    s = make_state(t, [skitty], [corsola])
    s = s._replace(stats=s.stats.at[0, 0, _SPE].set(21).at[1, 0, _SPE].set(19))
    acts = np.zeros(2, np.int32)
    laws = laws_of(t, s, acts)
    assert np.asarray(laws.compress)[0, 0, :2].all() and bool(laws.compress[0, 1, 0])
    outcomes, probs = enumerate_outcomes(laws)
    assert len(probs) <= enumerate_module.WIDTH_CAP
    assert abs(math.fsum(probs) - 1) < 1e-12
    # The brute force is above the cap: enumerate it with the cap lifted for this test.
    monkeypatch.setattr(enumerate_module, "WIDTH_CAP", 1 << 21)
    loose = laws_of(t, s, acts, loose=True)
    brute_outcomes, brute_probs = enumerate_outcomes(loose)
    assert len(brute_probs) > 131072
    exact = turn_children(t, s, acts, outcomes, probs)
    brute = turn_children(t, s, acts, brute_outcomes, brute_probs)
    assert set(exact) == set(brute)
    for key, p in brute.items():
        assert abs(exact[key] - p) < 1e-12
