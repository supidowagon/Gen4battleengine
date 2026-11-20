"""Mid-turn accuracy (K3b2 Task 1, docs/kaizo/k3b2-rules.md §0.2b-c): the decomp computes a
move's hit rate at its own check (``BattleControllerPlayer_CheckMoveHitAccuracy``,
battle_controller_player.c:2865-2979), with the stages of that moment, so a faster
Sand-Attack or Double Team changes a later move's hit chance in the same turn.
``outcome.hit[side, 0]`` is the roll's interval class over every threshold the move can
have this turn (``outcome.hit_thresholds``), and ``step`` compares it with the live one."""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.engine.enumerate import (
    SamplingOnly,
    enumerate_outcomes,
    hit_class_law,
    outcome_laws,
)
from nuzline.engine.field import with_weather
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat, Weather
from nuzline.engine.outcome import (
    MAX_HITS,
    Outcome,
    empty_outcome,
    hit_class,
    hit_threshold,
    sample_outcome,
)
from tests.engine.helpers import (
    BATCH,
    A,
    assert_matches_reference,
    jit_step,
    roark_state,
    sample_rows,
    step_batch,
)
from tests.strategies import legal_action_pair, outcome_law_state
from tools.effect_map import DECOMP_EFFECT_TO_ID

CHUNK = BATCH  # the suite's one padded row count (tests.engine.helpers)
HYPOTHESIS = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
}
_laws = jax.jit(outcome_laws)
_step = step_batch
_sample = sample_rows
# One compiled program for every example (an eager sample_outcome dispatches op by op and
# compiles its control flow anew on each call, so memory grew over the examples).
_sample_one = jax.jit(sample_outcome)


def laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, actions))


def hit_law(tables, state, actions, side) -> dict[int, int]:
    """``value -> weight`` (out of 100) of ``hit[side, 0]`` in the enumeration."""
    laws = laws_of(tables, state, actions)
    values, weights, den = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
    assert den == 100
    return dict(zip(values, weights, strict=True))


def stepped(tables, state, actions, outcomes: Outcome):
    """``step`` of every row of ``outcomes`` (numpy fields with a leading axis), in fixed
    ``CHUNK`` batches; the result's fields are numpy with the same leading axis."""
    n = len(outcomes.hit)
    parts = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = Outcome(*(np.asarray(f)[pad] for f in outcomes))
        out = _step(tables, state, actions, rows)
        parts.append([np.asarray(f)[: len(idx)] for f in out])
    return [np.concatenate(fs) for fs in zip(*parts)]


def state_keys(fields) -> list[bytes]:
    n = len(fields[0])
    flat = np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in fields], axis=1
    )
    return [flat[i].tobytes() for i in range(n)]


def merged(keys, probs) -> dict[bytes, Fraction | float]:
    out: dict = {}
    for k, p in zip(keys, probs):
        out[k] = out.get(k, 0) + p
    return out


def sand_attack_first(tables, names):
    """Roark's Geodude, faster, uses Sand-Attack; the player's Monferno then Scratches
    (accuracy 100)."""
    s = roark_state(tables, names, opp_moves=["Sand-Attack"], opp_faster=True)
    return s, A("Scratch", "Sand-Attack", s, names)


def hit_probability(tables, state, actions, side) -> float:
    """The exact probability that ``side``'s move connects (the foe loses HP)."""
    outcomes, probs = enumerate_outcomes(laws_of(tables, state, actions))
    out = stepped(tables, state, actions, outcomes)
    hp = out[state._fields.index("hp")]
    foe = 1 - side
    target = int(state.active[foe])
    landed = hp[:, foe, target] < int(state.hp[foe, target])
    return math.fsum(probs[landed])


# --- the bug ---------------------------------------------------------------------


@pytest.mark.slow
def test_sand_attack_lowers_a_later_move_this_turn(tables, names):
    s, acts = sand_attack_first(tables, names)
    # HitRateByStage[5] = 75/100 (include/data/hit_rate_stages.h) at accStages -1
    # (bcp.c:2873); Scratch's own rate was 100 at the start of the turn.
    assert int(hit_threshold(tables, s, acts)[0]) == 100
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12
    # The stage drop is there after the turn (the move hit: Sand-Attack has accuracy 100).
    out = jit_step(tables, s, acts, empty_outcome())
    assert int(out.stages[0, Stat.ACC]) == -1

    # A slower Sand-Attack comes after Scratch: the turn-start rate stands.
    slow = roark_state(tables, names, opp_moves=["Sand-Attack"])
    assert hit_law(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) == {1: 100}
    assert (
        abs(hit_probability(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) - 1) < 1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize("move", ["Sand-Attack", "Double Team"])
def test_accuracy_change_equals_bruteforce(tables, names, move):
    # K3b2 final review I1: the brute force cuts the hit law at every threshold variant
    # (outcome.hit_thresholds), not only the possible ones, so a variant the gated laws
    # wrongly drop shows as a difference here.
    from tests.engine.test_enumerate import assert_equals_bruteforce

    s = roark_state(tables, names, opp_moves=[move], opp_faster=True)
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))  # no crits: a smaller product
    width, brute = assert_equals_bruteforce(tables, s, A("Scratch", move, s, names))
    assert width <= brute


def test_double_team_raises_evasion_for_a_later_move(tables, names):
    s = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    acts = A("Scratch", "Double Team", s, names)
    # evaStages = 6 - statBoosts = -1 (bcp.c:2874): HitRateByStage[5] again.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12


@pytest.mark.slow
def test_nature_power_call_is_seen_by_hit_thresholds(tables, names):
    # On GREAT_MARSH a faster foe's Nature Power calls Mud Bomb (Mud-Slap's family:
    # SECONDARY_STAT_FOE, accuracy -1, effect_script_0073.s): the later Scratch can meet
    # stage -1 (hit_thresholds reads the called move, order.called_move).
    marsh = 10  # TERRAIN_GREAT_MARSH (nuzline.data.trainer.TERRAIN_NAMES)
    s = roark_state(tables, names, opp_moves=["Nature Power"], opp_faster=True)
    s = s._replace(terrain=jnp.asarray(marsh, s.terrain.dtype))
    acts = A("Scratch", "Nature Power", s, names)
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    o = empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, s, acts, o._replace(hit=o.hit.at[0, 0].set(76)))
    assert int(out.stages[0, Stat.ACC]) == -1 and int(out.hp[1, 0]) == int(s.hp[1, 0])


@pytest.mark.slow
def test_class_values_against_live_threshold(tables, names):
    s, acts = sand_attack_first(tables, names)
    for value, connects in ((1, True), (75, True), (76, False), (0, False)):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        out = jit_step(tables, s, acts, o)
        assert (int(out.hp[1, 0]) < int(s.hp[1, 0])) == connects, value
    # Without the Sand-Attack, 76 is a hit at 100.
    plain = roark_state(tables, names, opp_faster=True)
    o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(76))
    out = jit_step(tables, plain, A("Scratch", "Leer", plain, names), o)
    assert int(out.hp[1, 0]) < int(plain.hp[1, 0])


# --- the law ---------------------------------------------------------------------


def test_hit_class_law_examples():
    assert hit_class_law([95]) == ([0, 1], [5, 95], 100)
    assert hit_class_law([100]) == ([1], [100], 100)
    assert hit_class_law([166, 100]) == ([1], [100], 100)
    assert hit_class_law([100, 75]) == ([1, 76], [75, 25], 100)
    assert hit_class_law([95, 71]) == ([0, 1, 72], [5, 71, 24], 100)
    assert hit_class_law([0]) == ([0], [100], 100)


@pytest.mark.slow
@settings(max_examples=40, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_class_law_sums_to_one(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    for side in (0, 1):
        possible = laws.hit_possible[side]
        assert possible[0], "the turn-start threshold is always possible"
        assert int(laws.hit_thresholds[side, 0]) == int(hit_threshold(tables, s, acts)[side])
        values, weights, den = hit_class_law(laws.hit_thresholds[side][possible])
        assert sum(weights) == den == 100 and min(weights) > 0
        assert len(set(values)) == len(values)
    try:
        _, probs = enumerate_outcomes(laws)
    except SamplingOnly:
        return
    assert abs(math.fsum(probs) - 1) < 1e-12


def test_no_change_keeps_two_classes(tables, names):
    s = roark_state(tables, names)
    acts = A("Scratch", "Tackle", s, names)  # Tackle 95, Scratch 100, Leer-free turn
    assert hit_law(tables, s, acts, 1) == {0: 5, 1: 95}
    assert hit_law(tables, s, acts, 0) == {1: 100}
    # A faster Sand-Attack on the other side does not split a turn that cannot change it.
    slow = roark_state(tables, names, player_moves=["Sand-Attack"], opp_faster=True)
    acts = A("Sand-Attack", "Tackle", slow, names)
    assert hit_law(tables, slow, acts, 1) == {0: 5, 1: 95}


def test_white_herb_restores_the_stage_before_the_check(tables, names):
    s, acts = sand_attack_first(tables, names)
    mon = int(s.active[0])
    herb = int(np.flatnonzero(np.asarray(tables.item_effect) == ItemId.WHITE_HERB)[0])
    s = s._replace(item=s.item.at[0, mon].set(herb))
    # Over-approximated: the herb's reset to 0 is a threshold the move may have.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    for value in (1, 76):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        assert_matches_reference(tables, s, acts, o)


# --- both engines ----------------------------------------------------------------


def test_accuracy_classes_match_reference(tables, names):
    cases = [sand_attack_first(tables, names)]
    dt = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    cases.append((dt, A("Scratch", "Double Team", dt, names)))
    # The player's Sand-Attack first, then Geodude's Tackle (95): three classes.
    sa = roark_state(tables, names, player_moves=["Sand-Attack"])
    cases.append((sa, A("Sand-Attack", "Tackle", sa, names)))
    for s, acts in cases:
        laws = laws_of(tables, s, acts)
        for side in (0, 1):
            values, _, _ = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
            for value in [*values, 0, 1, 100]:
                o = empty_outcome()._replace(hit=empty_outcome().hit.at[side, 0].set(value))
                assert_matches_reference(tables, s, acts, o)
    assert hit_law(tables, sa, cases[-1][1], 1) == {0: 5, 1: 71, 72: 24}


# --- sampling --------------------------------------------------------------------


def test_sampled_hit_frequencies_match_law(tables, names):
    s = roark_state(tables, names, player_moves=["Sand-Attack"])
    acts = A("Sand-Attack", "Tackle", s, names)
    law = hit_law(tables, s, acts, 1)
    assert law == {0: 5, 1: 71, 72: 24}
    n = 20_000
    keys = jax.random.split(jax.random.key(3), n)
    drawn = np.asarray(_sample(keys, tables, s, acts).hit)[:, 1, 0]
    assert set(np.unique(drawn).tolist()) <= set(law)
    for value, weight in law.items():
        p = weight / 100
        assert abs(np.mean(drawn == value) - p) < 5 * math.sqrt(p * (1 - p) / n), value


def test_hit_class_maps_every_roll():
    thresholds = jnp.array([95, 71, 100, 30], jnp.int32)
    possible = jnp.array([True, True, True, False])
    values, weights, _ = hit_class_law([95, 71, 100])
    got = [int(hit_class(u, thresholds, possible)) for u in range(1, 101)]
    assert [got.count(v) for v in values] == weights
    # Each class value decides every possible threshold exactly as its rolls do.
    for u, v in zip(range(1, 101), got, strict=True):
        for t in (95, 71, 100):
            assert (u <= t) == (0 < v <= t)


# --- soundness: classes equal the raw rolls ----------------------------------------


@pytest.mark.slow
@settings(max_examples=30, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_classes_equal_raw_rolls(tables, names, data):
    """With every other field fixed, the class enumeration of both sides' rolls gives the
    same merged children as all 100 x 100 raw rolls ``u`` (stored as is: ``u <= t`` is
    ``0 < u <= t``). A threshold missing from ``hit_thresholds`` would split a class's
    rolls between different states and fail this."""
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    assume(laws.relevant.hit[:, 0].any())
    base = jax.tree.map(
        np.asarray, _sample_one(jax.random.key(data.draw(st.integers(0, 99))), tables, s, acts)
    )

    def rows(values0, values1):
        grid = [(a, b) for a in values0 for b in values1]
        hit = np.repeat(base.hit[None], len(grid), axis=0)
        hit[:, 0, 0] = [a for a, _ in grid]
        hit[:, 1, 0] = [b for _, b in grid]
        fields = {
            name: np.repeat(np.asarray(f)[None], len(grid), axis=0)
            for name, f in zip(Outcome._fields, base)
        }
        fields["hit"] = hit
        return Outcome(**fields), grid

    laws_side = [
        dict(zip(*hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])[:2]))
        for side in (0, 1)
    ]
    classes, grid = rows(list(laws_side[0]), list(laws_side[1]))
    probs = [Fraction(laws_side[0][a] * laws_side[1][b], 10_000) for a, b in grid]
    want = merged(state_keys(stepped(tables, s, acts, classes)), probs)
    raw, grid = rows(list(range(1, 101)), list(range(1, 101)))
    got = merged(state_keys(stepped(tables, s, acts, raw)), [Fraction(1, 10_000)] * len(grid))
    assert got == want


# (move the faster opponent uses, which side's stage, its value, White Herb, Simple)
FIXED_CASES = [
    *(
        ("Sand-Attack", 0, stage, herb, simple)
        for stage in (-6, -2, 0, 3)
        for herb, simple in ((False, False), (True, False), (True, True))
    ),
    *(("Double Team", 1, stage, True, False) for stage in (-3, 0, 6)),
]


@pytest.mark.parametrize(("move", "holder", "stage", "herb", "simple"), FIXED_CASES)
def test_accuracy_classes_equal_raw_rolls_fixed(tables, names, move, holder, stage, herb, simple):
    """Hand-written cases of ``test_accuracy_classes_equal_raw_rolls``: the faster
    opponent's Sand-Attack (on the player's accuracy stage) or Double Team (on its own
    evasion stage), from several starting stages, with a White Herb and Simple on the
    Pokemon whose stage changes; the player's Scratch then checks accuracy."""
    s = roark_state(tables, names, opp_moves=[move], opp_faster=True)
    s = s.with_stages(0, acc=stage) if holder == 0 else s.with_stages(1, eva=stage)
    mon = int(s.active[holder])
    if herb:
        white_herb = int(np.flatnonzero(np.asarray(tables.item_effect) == ItemId.WHITE_HERB)[0])
        s = s._replace(item=s.item.at[holder, mon].set(white_herb))
    if simple:
        s = s._replace(ability=s.ability.at[holder, mon].set(AbilityId.SIMPLE))
    acts = A("Scratch", move, s, names)
    law = hit_law(tables, s, acts, 0)

    def rows(values):
        base = empty_outcome()
        return Outcome(
            *(np.repeat(np.asarray(f)[None], len(values), axis=0) for f in base)
        )._replace(hit=np.repeat(np.asarray(base.hit)[None], len(values), axis=0).copy())

    classes, raw = rows(list(law)), rows(list(range(1, 101)))
    classes.hit[:, 0, 0] = list(law)
    raw.hit[:, 0, 0] = range(1, 101)
    want = merged(
        state_keys(stepped(tables, s, acts, classes)), [Fraction(w, 100) for w in law.values()]
    )
    got = merged(state_keys(stepped(tables, s, acts, raw)), [Fraction(1, 100)] * 100)
    assert got == want
    for value in [*law, 0]:
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        assert_matches_reference(tables, s, acts, o)


# Every EffectId when hit_thresholds was written, classified. A new one fails
# test_every_accuracy_or_evasion_writer_is_seen_by_hit_thresholds until it is added to
# one of these (and, if it writes Stat.ACC / Stat.EVA, to outcome.hit_thresholds).
_STAT_KINDS = frozenset(
    {
        EffectId.STAT_SELF,
        EffectId.STAT_FOE,
        EffectId.SECONDARY_STAT_SELF,
        EffectId.SECONDARY_STAT_FOE,
    }
)
# Effects that write no accuracy or evasion stage (effects.py; hooks.py's White Herb and
# Intimidate are an item and an ability): RAISE_ALL_STATS_HIT raises Atk, Def, Spe, SpA,
# SpD (subscript_boost_all_stats.s), CURSE changes Spe, Atk, Def (subscript_curse_normal.s),
# the rest change no stage. FANG_PARALYZE_FLINCH (K3b2 Task 4) paralyzes and flinches
# (subscript_paralyze_or_flinch.s): no stage. PRESENT (Task 5) damages or heals the target
# (effect_script_0122.s, subscript_recover_hp.s): no stage. PROTECT (Task 6) sets the
# user's protecting flag and chain (BtlCmd_TryProtection): no stage. ATTRACT (Task 7) sets
# the target's infatuated (BtlCmd_TryAttract): no stage. CAMOUFLAGE (Task 10) writes the
# user's types (BtlCmd_TryCamouflage) and TRI_ATTACK a burn, freeze or paralysis
# (effect_script_0036.s): no stage. NATURE_POWER writes none itself; the move it calls can
# (Mud Bomb's accuracy drop at GREAT_MARSH), and hit_thresholds reads that called move's
# effect (order.called_move): test_nature_power_call_is_seen_by_hit_thresholds. ROAR (Task 11)
# drags the target out (BtlCmd_TryWhirlwind): the newcomer's stages are fresh, and it runs no
# accuracy check this turn (UpdateAfterSwitch ends its action), nor does anyone after Roar.
_NO_ACC_EVA_WRITE = (frozenset(EffectId(i) for i in range(40)) - _STAT_KINDS) | {
    EffectId.FANG_PARALYZE_FLINCH,
    EffectId.PRESENT,
    EffectId.PROTECT,
    EffectId.ATTRACT,
    EffectId.CAMOUFLAGE,
    EffectId.NATURE_POWER,
    EffectId.TRI_ATTACK,
    EffectId.ROAR,
    # E2 batch 3: writes the weather, not a stage; hit_thresholds follows it (Sand Veil, Snow
    # Cloak) as its weather variant.
    EffectId.WEATHER,
    # E2 batch 4: no accuracy or evasion stage. Secret Power's terrain effect can lower accuracy
    # (or evasion), but hit_thresholds reads it resolved (order.move_effects): as
    # SECONDARY_STAT_FOE, a _SEEN pair.
    EffectId.DOUBLE_POWER,
    EffectId.HP_POWER,
    EffectId.WEIGHT_POWER,
    EffectId.HALVE_HP,
    EffectId.OHKO,
    EffectId.FALSE_SWIPE,
    EffectId.SWAGGER,
    EffectId.DRAGON_DANCE,
    EffectId.ROOST,
    EffectId.TAILWIND,
    EffectId.SPIKES,
    EffectId.TOXIC_SPIKES,
    EffectId.DESTINY_BOND,
    EffectId.SECRET_POWER,
    EffectId.COUNTER,
    # E2 batch 5: the invokers write none themselves; hit_thresholds reads the move they call
    # (outcome_laws' and sample_outcome's ``moves``, order.used_moves).
    EffectId.SLEEP_TALK,
    EffectId.METRONOME,
    EffectId.COPYCAT,
    EffectId.ASSIST,  # an invoker (B10-11)
    EffectId.ME_FIRST,  # an invoker (E3-2-3): the copied move's own effect decides
    EffectId.MIRROR_MOVE,  # an invoker (E3-B2-2): the copied move's own effect decides
    EffectId.CONVERSION_2,  # writes the user's types, no stage (E3-B2-3)
    EffectId.SPITE,  # writes the target's PP, no stage (E3-B2-4)
    EffectId.GRUDGE,  # sets a flag on its user, no stage (E3-B2-9)
    # sets the target's identified bit, no stage; hit_thresholds' ``free`` clause covers its change
    # of the evasion the later checks read (like Foresight, outcome.future_thresholds, E3-B2-10)
    EffectId.MIRACLE_EYE,
    # sets a flag on its user, no stage; the move it reflects is read as the holder's own
    # (order.used_moves, E3-B2-6)
    EffectId.MAGIC_COAT,
    EffectId.LUCKY_CHANT,  # a side's crit block: no stage, no accuracy input (E3-B2-7)
    # Uproar (a lock and a field bit) and Future Sight (a stored hit) write no stage.
    EffectId.UPROAR,
    EffectId.FUTURE_SIGHT,
    # E2 batch 6: U-turn and Baton Pass write no stage on the field mid-law: a Baton Pass
    # newcomer's passed stages arrive at the pause, whose resumed half draws its own laws from the
    # paused state (B6-6, B6-10).
    EffectId.U_TURN,
    EffectId.BATON_PASS,
    # E2 batches 7-8: traps, seals, a lock, counts; Lock-On sets no stage (it makes the locker's
    # next move skip the roll, outcome._rate), and its user has used its move this turn.
    EffectId.INGRAIN,
    EffectId.MEAN_LOOK,
    EffectId.IMPRISON,
    EffectId.LOCK_ON,
    EffectId.PERISH_SONG,
    EffectId.YAWN,
    # E2 batch 9: Follow Me and Helping Hand change nothing in singles; Belly Drum writes Attack
    # only (subscript_belly_drum.s). Bulk Up is DRAGON_DANCE (Attack and Defense); Minimize is
    # STAT_SELF on Stat.EVA, a _SEEN pair.
    EffectId.FOLLOW_ME,
    EffectId.HELPING_HAND,
    EffectId.BELLY_DRUM,
    # E2 batch 10: hits whose recoil, status or crash write no stage (B10-1, B10-2); Substitute
    # writes the user's HP and substitute only (B10-4).
    EffectId.RECOIL_STATUS,
    EffectId.CRASH,
    EffectId.SUBSTITUTE,
    # Endeavor and Pain Split write HP only (B10-5).
    EffectId.ENDEAVOR,
    EffectId.PAIN_SPLIT,
    # Knock Off takes an item only (B10-6), no stage; the Wide Lens it takes is
    # hit_thresholds' second half, as Embargo's (F-17).
    EffectId.KNOCK_OFF,
    EffectId.SAFEGUARD,  # a side condition (B10-7)
    EffectId.FOCUS_PUNCH,  # a plain hit (B10-8)
    EffectId.LAST_RESORT,  # a plain hit that can fail (E3-B2-11)
    EffectId.FEINT,  # a plain hit that can fail (E3-B2-13)
    EffectId.ENCORE,  # sets the encore only (B10-9)
    # E2 batch 12, the stat family: Tickle writes Attack and Defense (B12-4), Power / Guard Swap
    # Attack / Sp. Atk or Defense / Sp. Def (B12-7), Captivate Sp. Atk (B12-8): their params name
    # no accuracy or evasion (checked below). Focus Energy (B12-3) and Power Trick (B12-10) write
    # no stage.
    EffectId.TICKLE,
    EffectId.STAGE_SWAP,
    EffectId.CAPTIVATE,
    EffectId.FOCUS_ENERGY,
    EffectId.POWER_TRICK,
    # The healing family writes HP, status, sleep turns, a wish or a side's Mist only (B12-12..);
    # Mist blocks later drops (B12-23) but writes no stage.
    EffectId.REST,
    EffectId.WEATHER_HEAL,
    EffectId.WISH,
    EffectId.HEAL_BELL,
    EffectId.REFRESH,
    EffectId.PSYCHO_SHIFT,
    EffectId.MIST,
    # E2 batch 13: hits that write HP only (Triple Kick B13-5, Beat Up B13-9; the variable
    # power and type hits B13-12..B13-17, Assurance's flag besides).
    EffectId.TRIPLE_KICK,
    EffectId.BEAT_UP,
    EffectId.WEATHER_BALL,
    EffectId.GYRO_BALL,
    EffectId.ASSURANCE,
    EffectId.TRUMP_CARD,
    EffectId.PUNISHMENT,
    EffectId.JUDGMENT,
    # The held-item moves write their user's item and recycleItem, and Natural Gift HP
    # (B13-19, B13-20); no accuracy item can be consumed, so none comes back.
    EffectId.RECYCLE,
    EffectId.NATURAL_GIFT,
    # Embargo writes the defender's embargoTurns, no stage (B13-31); the Wide Lens it hides is
    # hit_thresholds' second half (outcome._threshold_grid).
    EffectId.EMBARGO,
    # Endure (effect_script_0116.s:5) runs subscript_protect (:5-7): BtlCmd_TryProtection
    # (battle_script.c:5149-5190) sets turnFlags.enduring and protectSuccessTurns, no stage.
    EffectId.ENDURE,
    # Magnet Rise (effect_script_0252.s:5-12) sets MOVE_EFFECT_MAGNET_RISE and
    # magnetRiseTurns on its user, no stage.
    EffectId.MAGNET_RISE,
    # Mimic (effect_script_0082.s:5) runs subscript_mimic (:5-11): BtlCmd_TryMimic
    # (battle_script.c:4476-4524) writes a move slot, no stage.
    EffectId.MIMIC,
    # Chatter (effect_script_0267.s:5-7) is a hit whose subscript_chatter (:5-6) confuses the
    # defender through subscript_confuse: a volatile status, no stage.
    EffectId.CHATTER,
    # E3 batch 1: Splash / Kaizo Thief (effect_script_0085.s:5) and Teleport (effect_script_0153.s)
    # write no stage: a used move's last_move and PP only.
    EffectId.DO_NOTHING,
    EffectId.FLEE,
    EffectId.TORMENT,  # E3-2-1: a statusVolatile bit, no stage
    EffectId.NIGHTMARE,  # G-31: a statusVolatile bit (subscript_nightmare_start.s:9), no stage
    EffectId.TRICK_ROOM,  # E3-2-4: a field condition bit, no stage
}
# Effects that write an accuracy or evasion stage outside the stat kinds, each followed by
# hit_thresholds (B12-31): Haze zeroes every stage (BtlCmd_ResetAllStatChanges,
# battle_script.c:3534-3549), Psych Up copies the side's stages onto its user
# (BtlCmd_CopyStatStages, :6042-6053), Acupressure can draw evasion (BtlCmd_BoostRandomStatBy2,
# :8932-8956): test_haze_psych_up_acupressure_are_seen_by_hit_thresholds.
# Gravity (effects._gravity) sets FIELD_CONDITION_GRAVITY, whose x10/6 the check applies
# (battle_controller_player.c:2972-2974); Transform (BtlCmd_Transform, battle_script.c:5330-5373) copies the
# defender's statBoosts and ability (battle_mon.h:75, :84) and reruns its switch-in checks:
# test_gravity_and_transform_are_seen_by_hit_thresholds. Worry Seed (E3-2-2) writes the target's
# ability (Compoundeyes, Hustle, ... lost before its check): tests/engine/test_e3_batch1.py
# test_worry_seed_is_seen_by_hit_thresholds_and_crit_odds. Skill Swap (E3-2-5) swaps the two
# battlers' abilities: tests/engine/test_e3_batch1.py test_skill_swap_is_seen_by_hit_thresholds. Role Play (E3-B2-5) gives its user the target's ability: tests/engine/test_e3_batch2.py test_role_play_is_seen_by_hit_thresholds_and_armor. Gastro Acid (E3-2-6) suppresses the target's
# ability (Compoundeyes, Hustle, ... read NONE at its check):
# tests/engine/test_e3_batch1.py test_gastro_acid_is_seen_by_hit_thresholds_and_crit_odds.
_FOLLOWED = frozenset(
    {
        EffectId.HAZE,
        EffectId.PSYCH_UP,
        EffectId.ACUPRESSURE,
        EffectId.GRAVITY,
        EffectId.TRANSFORM,
        EffectId.WORRY_SEED,
        EffectId.SKILL_SWAP,
        EffectId.ROLE_PLAY,
        EffectId.GASTRO_ACID,
        EffectId.FLING,
        EffectId.DEFOG,  # clears the deep fog the check reads (E3-B2-14)
    }
)
# Effects whose params name stats: they must name no accuracy or evasion.
_PARAM_STATS = {EffectId.TICKLE: (0, 1), EffectId.STAGE_SWAP: (0, 1), EffectId.CAPTIVATE: (0,)}
# (effect, stat) pairs hit_thresholds follows: they change a later foe move's threshold.
_SEEN = {
    (EffectId.STAT_FOE, Stat.ACC),
    (EffectId.SECONDARY_STAT_FOE, Stat.ACC),
    (EffectId.STAT_SELF, Stat.EVA),
    (EffectId.SECONDARY_STAT_SELF, Stat.EVA),
}
# Pairs that cannot: the user's own accuracy and the foe's evasion only feed the user's
# own move, which has already run (Sweet Scent lowers the foe's evasion).
_IRRELEVANT = {
    (EffectId.STAT_SELF, Stat.ACC),
    (EffectId.SECONDARY_STAT_SELF, Stat.ACC),
    (EffectId.STAT_FOE, Stat.EVA),
    (EffectId.SECONDARY_STAT_FOE, Stat.EVA),
}


@pytest.mark.fullset
def test_every_accuracy_or_evasion_writer_is_seen_by_hit_thresholds(tables):
    """``outcome.hit_thresholds`` follows the stat effects on ``Stat.ACC`` / ``Stat.EVA``
    (``move_param[m, 0]``). An effect added later that writes them (Minimize, Haze, Psych
    Up, ...) fails here until it is classified and, if it can, followed there."""
    assert set(EffectId) == _NO_ACC_EVA_WRITE | _STAT_KINDS | _FOLLOWED, (
        "a new EffectId: say whether it writes Stat.ACC / Stat.EVA (outcome.hit_thresholds)"
    )
    assert not (_NO_ACC_EVA_WRITE & _FOLLOWED)
    pairs = {
        (effect, params[0])
        for effect, params in DECOMP_EFFECT_TO_ID.values()
        if effect in _STAT_KINDS and params[0] in (Stat.ACC, Stat.EVA)
    }
    effect, param = np.asarray(tables.move_effect), np.asarray(tables.move_param)
    for move in range(len(effect)):
        e, stat = EffectId(int(effect[move])), int(param[move, 0])
        if e in _STAT_KINDS and stat in (Stat.ACC, Stat.EVA):
            pairs.add((e, stat))
        for col in _PARAM_STATS.get(e, ()):
            assert int(param[move, col]) not in (Stat.ACC, Stat.EVA), (move, e)
    assert pairs, "no accuracy / evasion stat effect found"
    assert pairs <= _SEEN | _IRRELEVANT, pairs - _SEEN - _IRRELEVANT
    assert (EffectId.STAT_FOE, Stat.ACC) in pairs and (EffectId.STAT_SELF, Stat.EVA) in pairs


def _b12_31_cases(tables, names):
    """(state, actions, law, live, outcome, stages_ok) per B12-31 case: a faster foe's Haze,
    Psych Up or Acupressure before the slower player move. ``law`` is the move's class law,
    ``live`` its hit rate at its own check (after the foe's move), ``outcome`` the turn's other
    draws and ``stages_ok`` the foe's stage write. Before B12-31 each law held only the
    turn-start rate (100 for all three), one class whose rolls ``live + 1``..100 miss."""
    rolled = empty_outcome()
    # Haze: the player's accuracy +1 (Rock Throw 90 * 133 / 100, past 100) goes back to 0 (90)
    # before Rock Throw (BtlCmd_ResetAllStatChanges, battle_script.c:3534-3549).
    haze = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Haze"], opp_faster=True
    )
    haze = haze._replace(stages=haze.stages.at[0, Stat.ACC].set(1))
    # Psych Up: Geodude copies the player's evasion +1 (75 for Scratch).
    psych = roark_state(tables, names, opp_moves=["Psych Up"], opp_faster=True)
    psych = psych._replace(stages=psych.stages.at[0, Stat.EVA].set(1))
    # Acupressure: draw 6 of 7 valid stats is evasion (ACUPRESSURE_STATS): +2, 60 for Scratch.
    acup = roark_state(tables, names, opp_moves=["Acupressure"], opp_faster=True)
    drawn = rolled._replace(acupressure=rolled.acupressure.at[1].set(6))
    return [
        (
            haze,
            A("Rock Throw", "Haze", haze, names),
            {1: 90, 91: 10},
            90,
            rolled,
            lambda out: int(out.stages[0, Stat.ACC]) == 0,
        ),
        (
            psych,
            A("Scratch", "Psych Up", psych, names),
            {1: 75, 76: 25},
            75,
            rolled,
            lambda out: int(out.stages[1, Stat.EVA]) == 1,
        ),
        (
            acup,
            A("Scratch", "Acupressure", acup, names),
            {1: 60, 61: 40},
            60,
            drawn,
            lambda out: int(out.stages[1, Stat.EVA]) == 2,
        ),
    ]


def test_haze_psych_up_acupressure_are_seen_by_hit_thresholds(tables, names):
    """B12-31: a faster foe's Haze, Psych Up or Acupressure changes the stages the slower
    move's check reads (battle_controller_player.c:2873-2874); hit_thresholds holds the new
    threshold, so the roll past the live one is its own class."""
    for s, acts, law, _, _, _ in _b12_31_cases(tables, names):
        assert hit_law(tables, s, acts, 0) == law


@pytest.mark.slow
def test_haze_psych_up_acupressure_classes_match_the_live_check(tables, names):
    """B12-31: each class of the move's law is decided alike by the live check
    (``roll > hitRate`` misses, battle_controller_player.c:2976-2977) at both its ends, in both
    engines, and the classes that hit weigh ``live`` of 100. The turn-start law (one class
    1..100) fails both: its roll 100 misses where its roll 1 hits."""
    for s, acts, _, live, outcome, stages_ok in _b12_31_cases(tables, names):
        law = hit_law(tables, s, acts, 0)
        bounds = sorted(law)
        hit_weight = 0
        for lo, hi in zip(bounds, [*bounds[1:], 101], strict=True):
            hits = []
            for roll in (lo, hi - 1):
                out = assert_matches_reference(
                    tables, s, acts, outcome._replace(hit=outcome.hit.at[0, 0].set(roll))
                )
                assert stages_ok(out)
                hits.append(bool(out.hp[1, 0] < s.hp[1, 0]))
            assert hits == [lo <= live] * 2, (lo, hi - 1, hits)
            hit_weight += law[lo] if hits[0] else 0
        assert hit_weight == live


def _gravity_transform_cases(tables, names):
    """(state, actions, law, live) per case: a faster foe's Gravity or Transform before the
    slower player move; ``live`` is the move's hit rate at its own check. Before the fix each law
    held only the turn-start rate, one class whose rolls past it miss."""
    # Gravity: Rock Throw 90 becomes 90 * 10 / 6 = 150, past 100 (bcp.c:2972-2974).
    grav = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Gravity"], opp_faster=True
    )
    # Transform copies the player's evasion +1 (statBoosts, battle_mon.h:75): Scratch 75.
    stage = roark_state(tables, names, opp_moves=["Transform"], opp_faster=True)
    stage = stage._replace(stages=stage.stages.at[0, Stat.EVA].set(1))
    # Transform copies the player's Sand Veil (battle_mon.h:84) in sand: Scratch x80/100
    # (bcp.c:2927-2929) against the transformed Geodude.
    veil = roark_state(tables, names, opp_moves=["Transform"], opp_faster=True)
    veil = with_weather(veil, Weather.SANDSTORM)._replace(
        ability=veil.ability.at[0, int(veil.active[0])].set(AbilityId.SAND_VEIL)
    )
    return [
        (grav, A("Rock Throw", "Gravity", grav, names), {1: 90, 91: 10}, 100),
        (stage, A("Scratch", "Transform", stage, names), {1: 75, 76: 25}, 75),
        (veil, A("Scratch", "Transform", veil, names), {1: 80, 81: 20}, 80),
    ]


def test_gravity_and_transform_are_seen_by_hit_thresholds(tables, names):
    """A faster foe's Gravity (x10/6 at the check, battle_controller_player.c:2972-2974) or
    Transform (the side's stages and ability copied, BtlCmd_Transform battle_script.c:5330-5373)
    changes the slower move's threshold: hit_thresholds holds the new one, and each class is
    decided alike by the live check at both its ends, in both engines."""
    for s, acts, law, live in _gravity_transform_cases(tables, names):
        assert hit_law(tables, s, acts, 0) == law
        bounds = sorted(law)
        hit_weight = 0
        for lo, hi in zip(bounds, [*bounds[1:], 101], strict=True):
            hits = []
            for roll in (lo, hi - 1):
                o = empty_outcome()
                out = assert_matches_reference(
                    tables, s, acts, o._replace(hit=o.hit.at[0, 0].set(roll))
                )
                hits.append(bool(out.hp[1, 0] < s.hp[1, 0]))
            assert hits == [lo <= live] * 2, (lo, hi - 1, hits)
            hit_weight += law[lo] if hits[0] else 0
        assert hit_weight == live


def _taken_and_copied_cases(tables, names):
    """(state, actions, law, live) per case: a faster foe's move before the slower player move
    that changes its threshold through an item or an ability (F-17, G-69 follow-ups)."""
    cases = []
    # Knock Off takes the player's Wide Lens (BtlCmd_TryKnockOff, battle_script.c:6622-6624):
    # Rock Throw 90 * 110 / 100 = 99 becomes 90 (bcp.c:2955-2957), F-17.
    lens = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Knock Off"], opp_faster=True
    )
    lens = lens._replace(item=lens.item.at[0, 0].set(names.item("Wide Lens")))
    law = {1: 90, 91: 9, 0: 1}  # class 0: roll 100 misses at every threshold
    cases.append((lens, A("Rock Throw", "Knock Off", lens, names), law, 90))
    # Transform copies the player's Drizzle (battle_mon.h:84), whose switch-in check reruns
    # (bcp.c:3672-3673): rain, so Thunder never misses (bcp.c:3025-3027).
    rain = roark_state(
        tables, names, player_moves=["Thunder"], opp_moves=["Transform"], opp_faster=True
    )
    rain = rain._replace(ability=rain.ability.at[0, 0].set(AbilityId.DRIZZLE))
    cases.append((rain, A("Thunder", "Transform", rain, names), {1: 70, 71: 30}, 100))
    # The transformed foe holds the player's Unaware: the player's accuracy +1 is ignored
    # (bcp.c:2882-2884), Rock Throw 90 * 133 / 100 = 119 back to 90.
    unaware = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Transform"], opp_faster=True
    )
    unaware = unaware._replace(
        ability=unaware.ability.at[0, 0].set(AbilityId.UNAWARE),
        stages=unaware.stages.at[0, Stat.ACC].set(1),
    )
    cases.append((unaware, A("Rock Throw", "Transform", unaware, names), {1: 90, 91: 10}, 90))
    # The foe loses its own No Guard (bcp.c:3016-3021): Rock Throw can miss again.
    guard = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Transform"], opp_faster=True
    )
    guard = guard._replace(ability=guard.ability.at[1, 0].set(AbilityId.NO_GUARD))
    cases.append((guard, A("Rock Throw", "Transform", guard, names), {1: 90, 91: 10}, 90))
    # The foe loses its own Cloud Nine (NO_CLOUD_NINE, bcp.c:3024): rain's Thunder never misses.
    # NO_CLOUD_NINE reads the decomp ability (damage.decomp_abilities): a Psyduck in its
    # Cloud Nine slot.
    nine = roark_state(
        tables,
        names,
        player_moves=["Thunder"],
        opp_species="Psyduck",
        opp_moves=["Transform"],
        opp_faster=True,
    )
    slot = names.species_abilities(names.species("Psyduck")).index(names.ability("Cloud Nine"))
    nine = with_weather(nine, Weather.RAIN)._replace(
        ability_slot=nine.ability_slot.at[1, 0].set(slot),
        ability=nine.ability.at[1, 0].set(AbilityId.CLOUD_NINE),
    )
    cases.append((nine, A("Thunder", "Transform", nine, names), {1: 70, 71: 30}, 100))
    # In sun the lost Cloud Nine lets Thunder's hitRate = 50 apply (bcp.c:2915-2917), F-25.
    sun = with_weather(nine, Weather.SUN)
    cases.append((sun, A("Thunder", "Transform", sun, names), {1: 50, 51: 20, 0: 30}, 50))
    return cases


@pytest.mark.slow
def test_taken_items_and_copied_or_lost_abilities_are_seen_by_hit_thresholds(tables, names):
    """Each case's law holds the live threshold, and each class is decided alike by the live
    check at both its ends, in both engines (as test_gravity_and_transform_are_seen_by_...)."""
    for s, acts, law, live in _taken_and_copied_cases(tables, names):
        assert hit_law(tables, s, acts, 0) == law, law
        # Class k > 0 holds the rolls k up to the next class; class 0 the top rolls that miss
        # at every threshold.
        bounds = sorted(k for k in law if k)
        top = 101 - law.get(0, 0)
        ranges = list(zip(bounds, [*bounds[1:], top], strict=True))
        ranges += [(top, 101)] if 0 in law else []
        hit_weight = 0
        for lo, hi in ranges:
            hits = []
            for roll in (lo, hi - 1):
                o = empty_outcome()
                out = assert_matches_reference(
                    tables, s, acts, o._replace(hit=o.hit.at[0, 0].set(roll))
                )
                hits.append(bool(out.hp[1, 0] < s.hp[1, 0]))
            assert hits == [lo <= live] * 2, (law, lo, hi - 1, hits)
            hit_weight += law[lo if lo < top else 0] if hits[0] else 0
        assert hit_weight == live


@pytest.mark.slow
def test_a_self_lock_keeps_the_accuracy_roll_whole(tables, names):
    """Singles follow-up (ii): a lock whose stored target is its user may read either battler's
    evasion, items and abilities (the user's own, or the foe's after a faster Encore or Follow
    Me), so ``outcome.self_lock`` keeps its roll whole: the law and the sampler give every roll 1-100
    its own class, and the foe's side keeps its thresholds' two classes."""
    from tests.engine.test_singles_followup import _locked_rollout

    s = _locked_rollout(tables, names)
    acts = jnp.asarray([0, 0], jnp.int32)
    laws = laws_of(tables, s, acts)
    assert bool(laws.hit_whole[0]) and not bool(laws.hit_whole[1])
    keys = jax.random.split(jax.random.key(7), 4000)
    hit = np.asarray(_sample(keys, tables, s, acts).hit)
    assert set(hit[:, 0, 0].tolist()) == set(range(1, 101))
    assert set(hit[:, 1, 0].tolist()) <= {0, 1}  # Harden: accuracy 0, never rolls
