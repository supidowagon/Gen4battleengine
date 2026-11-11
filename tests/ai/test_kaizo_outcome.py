"""The Kaizo AI outcome layout, its factored distribution and the loose oracle (K3c Task 3;
docs/ai/kaizo_roark.md 8).

Decomp paths are relative to ``external/pokeplatinum``: ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``, ``script.s`` is ``src/battle/trainer_ai/script.s``.

The exact Kaizo distribution merges draws through four gates; each is checked against an
oracle that does not use it, and each has a test that fails when the gate is removed:
- per-slot factoring and the tie subsets: the factored pick against a joint enumeration of
  every slot's bits and the 12 tiebreaks (``test_factored_equals_joint_bruteforce``);
- path-only script-roll sites (a move's sites come from its effect's row of the site
  table): exact against loose, where every ``IfRandomLessThan`` read is its own draw
  (``test_exact_equals_loose``, ``test_loose_catches_a_missing_site``);
- the one-scalar ability guess (Ruling 6): exact against loose, where every
  ``LoadBattlerAbility DEFENDER`` read is its own coin flip, on the states where at most
  one read can change a score; and the data test that this holds for Roark
  (``test_one_scalar_guess_is_exact_for_roark``, ``test_loose_catches_a_split_guess``).
The fourth gate, the dropped speed-tie draw (Ruling 3), is checked in
``tests/ai/test_kaizo_commands.py`` (``test_speed_tie_never_slower``): ``IfSpeedCompareEqualTo``
gives the tie its own loose read (``commands.ReadKind.SPEED_TIE``). Since K3c Task 6 its three
compares are on transcribed paths (Expert_Explosion, Expert_StatusSpeedUp and Expert_Recovery,
script.s:1864, :2070, :2650), and ``tests/ai/test_kaizo_policy.py
test_exact_equals_loose_at_equal_speed`` checks exact against loose at equal speed on each.

The tests run on the real ``KAIZO`` layout: since K3c Task 6 every Expert label of Kaizo
Roark's effects is transcribed, so no test-only layout is needed.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.stats import chisquare

from nuzline.ai import commands, policy
from nuzline.ai.layout import KAIZO, VANILLA, layout_of
from nuzline.ai.outcome import (
    EFFECT_SITE_LINES,
    ENUMERATION_WIDTH,
    KAIZO_EXPERT_LABELS,
    MAX_ENUMERATED,
    MAX_SCRIPT_ROLLS,
    SIM_ROLL_MAX,
    SITE_LINES,
    SITE_N,
    SWITCH_NONE,
    TIEBREAK_MOD,
    TIEBREAK_P,
    AiOutcome,
    enumerate_ai_outcomes,
    sample_ai_outcome,
    slot_roll_thresholds,
    slot_site_table,
)
from nuzline.engine.actions import NUM_OPP_ACTIONS
from nuzline.engine.enumerate import WIDTH_CAP
from nuzline.engine.ids import Stat
from nuzline.engine.state import MOVES, BattleState, make_state
from nuzline.search.rows import take
from tests.ai.fixtures import GEODUDE, outcome, versus

# Kaizo move ids (data/gen4_platinum_kaizo names; effects checked in test_slot_site_table).
ROCK_THROW, PURSUIT, LEER, SCREECH, HEADBUTT, TACKLE = 88, 228, 43, 103, 29, 33
STEALTH_ROCK, ACCELEROCK, WATER_GUN, U_TURN = 446, 205, 55, 369
SOUNDPROOF, CLEAR_BODY = 43, 29  # decomp ability ids (commands.ABILITY_IDS)
_SETTINGS = settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

_factored = jax.jit(policy._slot_move_distribution, static_argnames="layout")
_distribution = policy._distribution  # jitted, static layout: NaN on a refused state
_scores_batch = jax.jit(
    jax.vmap(policy._scores, in_axes=(None, None, 0, None)), static_argnames="layout"
)
_pick_batch = jax.jit(
    jax.vmap(jax.vmap(policy._pick, in_axes=(None, None, None, 0)), in_axes=(None, None, 0, None))
)


# --- States ------------------------------------------------------------------------------


def _with_player(t, s: BattleState, species: int, slot: int = 0, known: bool = False):
    """``s`` with the player's active Pokemon made ``species`` (its types and its ability
    ``slot``; the AI knows the ability when ``known``). Its stats are left as they were."""
    return s._replace(
        species=s.species.at[0, 0].set(species),
        form=s.form.at[0, 0].set(0),
        types=s.types.at[0, 0].set(t.species_types[species]),
        ability_slot=s.ability_slot.at[0, 0].set(slot),
        ability=s.ability.at[0, 0].set(t.species_abilities[species, slot]),
        ai_ability_known=s.ai_ability_known.at[0].set(int(known)),
    )


def _kaizo_state(t, party, moves, bench=((ACCELEROCK, HEADBUTT, 0, 0),)) -> BattleState:
    """Kaizo Roark's Bonsly with ``moves`` (Kaizo ids) and Cranidos with ``bench`` behind
    it, against Roark's Gible as the player's lead: every opponent move transcribed."""
    opp = [dataclasses.replace(party[0], moves=tuple(moves), pp=None)]
    opp += [dataclasses.replace(party[5], moves=tuple(b), pp=None) for b in bench]
    return make_state(t, [party[2]], opp)


def _bulky(s: BattleState, hp: int = 400) -> BattleState:
    """The player's active at ``hp``/``hp``: no transcribed move kills it."""
    return s._replace(hp=s.hp.at[0, 0].set(hp), max_hp=s.max_hp.at[0, 0].set(hp))


def _move_pool(t, layout=KAIZO) -> tuple[int, ...]:
    """Kaizo moves ``layout`` transcribes (effect in ``layout.transcribed`` and, of
    ``KAIZO_EXPERT_LABELS``, only those in ``layout.expert_transcribed``; no id case, no
    alternative power)."""
    effect = np.asarray(t.move_ai_effect)
    skip = set(commands.CALC_DAMAGE_MOVES) | set(commands.MOVE_TYPE_MOVES)
    pending = set(KAIZO_EXPERT_LABELS) - layout.expert_transcribed
    return tuple(
        m
        for m in range(1, len(effect))
        if int(effect[m]) in layout.transcribed
        and int(effect[m]) not in pending
        and m not in skip
        and int(effect[m]) not in commands.ALT_POWER_EFFECTS
    )


def _two_ability_species(t) -> tuple[int, ...]:
    """Species with two different, nonzero abilities: the ones the AI guesses between
    (trainer_ai.c:1191-1198)."""
    a = np.asarray(t.species_abilities_decomp)
    n = t.species_form_row.shape[0]  # species ids; the rows past them are forms
    return tuple(s for s in range(1, n) if a[s, 0] and a[s, 1] and a[s, 0] != a[s, 1])


@st.composite
def kaizo_states(draw, t, party) -> BattleState:
    """Random Kaizo decisions: Bonsly's moves drawn from the transcribed pool (slots 1-3
    may be empty or out of PP), any player species with two abilities, guessed or known,
    random HP on both sides, Defense stage, turn and Pursuit's first-turn counter, and a
    U-turn the AI may have seen."""
    pool = _move_pool(t)
    moves = [draw(st.sampled_from(pool))] + [draw(st.sampled_from((0, *pool))) for _ in range(3)]
    s = _kaizo_state(t, party, moves)
    species = draw(st.sampled_from(_two_ability_species(t)))
    s = _with_player(t, s, species, draw(st.integers(0, 1)), draw(st.booleans()))
    hp = draw(st.integers(20, 400))
    s = s._replace(hp=s.hp.at[0, 0].set(hp), max_hp=s.max_hp.at[0, 0].set(max(hp, 400)))
    opp_max = int(s.max_hp[1, 0])
    s = s._replace(hp=s.hp.at[1, 0].set(draw(st.integers(1, opp_max))))
    s = s._replace(
        stages=s.stages.at[0, Stat.DEF].set(draw(st.integers(-6, 6))),
        turn=jnp.asarray(draw(st.integers(0, 3)), s.turn.dtype),
        fake_out_turn=s.fake_out_turn.at[1].set(draw(st.integers(0, 3))),
        moves=s.moves.at[0, 0, 3].set(U_TURN),
        ai_seen_moves=s.ai_seen_moves.at[0, 3].set(int(draw(st.booleans()))),
    )
    empty_pp = draw(st.lists(st.booleans(), min_size=3, max_size=3))
    for k, empty in enumerate(empty_pp, start=1):
        if empty and draw(st.booleans()):
            s = s._replace(pp=s.pp.at[1, 0, k].set(0))
    return s


@st.composite
def site_layouts(draw):
    """``KAIZO`` with a synthetic site table: each transcribed effect gets any of
    the script's roll lines, in any order, with any N (lines its path never reads are
    columns that are never read)."""
    rows = []
    for effect in sorted(KAIZO.transcribed):
        lines = draw(st.lists(st.sampled_from(SITE_LINES), unique=True, max_size=4))
        rows.append((effect, tuple((line, draw(st.integers(1, 255))) for line in lines)))
    return KAIZO._replace(effect_sites=tuple(rows))


# --- Oracles -------------------------------------------------------------------------------


def _joint_moves(t, s, layout) -> np.ndarray:
    """``float64[MOVES]``: the move pick by a joint enumeration of every slot's used sites,
    both guesses and the 12 tiebreaks, through ``_scores`` and ``_pick`` (no factoring)."""
    thresholds = np.asarray(slot_roll_thresholds(t, s, layout))
    used = np.argwhere(thresholds > 0)
    bits = (np.arange(2 ** len(used))[:, None] >> np.arange(len(used))) & 1
    rolls = np.zeros((len(bits), MOVES, layout.slot_sites), np.int8)
    rolls[:, used[:, 0], used[:, 1]] = bits
    p = thresholds[used[:, 0], used[:, 1]] / 256
    weight = np.prod(np.where(bits == 1, p, 1 - p), axis=1)
    dist = np.zeros(MOVES)
    k = len(bits)
    for guess in range(2):
        outs = AiOutcome(
            sim_roll=jnp.full((k, MOVES), SIM_ROLL_MAX, jnp.int8),
            ability_guess=jnp.full(k, guess, jnp.int8),
            script_rolls=jnp.asarray(rolls),
            tiebreak=jnp.zeros(k, jnp.int8),
            switch_stop=jnp.full(k, SWITCH_NONE, jnp.int8),
        )
        scores = _scores_batch(t, s, outs, layout)
        picks = np.asarray(_pick_batch(t, s, scores, jnp.arange(TIEBREAK_MOD, dtype=jnp.int8)))
        share = weight[:, None] * TIEBREAK_P[None, :] / 2
        for m in range(MOVES):
            dist[m] += share[picks == m].sum()
    return dist


def _sensitive_reads(t, states: BattleState, layout) -> np.ndarray:
    """``int[N]``: per state (leading axis N), the number of ``LoadBattlerAbility DEFENDER``
    reads (slot, read) whose own coin flip changes the slot's score for some value of the
    slot's other reads (loose: every read its own draw)."""
    reads, _, kind = policy._loose_reads(t, states, layout)
    bits = jnp.asarray(policy._loose_patterns(reads))
    scores, _ = policy._loose_batch(t, states, bits, layout)
    scores = np.asarray(scores)  # [N, P, MOVES]
    count = np.zeros(scores.shape[0], np.int64)
    for m in range(MOVES):
        q = np.arange(2 ** len(reads[m]))
        for k, r in enumerate(reads[m]):
            if kind[m, r] != commands.ReadKind.ABILITY:
                continue
            changed = scores[:, q, m] != scores[:, q ^ (1 << k), m]
            count += changed.any(axis=1)
    return count


def _one(s: BattleState) -> BattleState:
    return jax.tree.map(lambda a: jnp.asarray(a)[None], s)


# --- The layout ----------------------------------------------------------------------------


def test_vanilla_layout_frozen(tables, names):
    """The flat layout is unchanged (plan K3c Global Constraints): its shapes, its width and
    Task 2's bit-equality fixtures (tests/ai/test_policy.py FROZEN_VANILLA)."""
    from tests.ai import test_policy

    assert (ENUMERATION_WIDTH, MAX_ENUMERATED, MAX_SCRIPT_ROLLS) == (3840, 4096, 4)
    assert not VANILLA.per_slot_sites and VANILLA.effect_sites == ()
    s = versus(tables, names, GEODUDE)
    o = sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    assert {f: tuple(np.shape(v)) for f, v in o._asdict().items()} == {
        "sim_roll": (4,),
        "ability_guess": (),
        "script_rolls": (4,),
        "tiebreak": (),
        "switch_stop": (),
    }
    outs, _ = enumerate_ai_outcomes(tables, s)
    assert outs.script_rolls.shape == (ENUMERATION_WIDTH, MAX_SCRIPT_ROLLS)
    assert policy.ai_width(tables, s) == 10 + 2 * 16
    assert policy.ai_flat_width(tables, s) == ENUMERATION_WIDTH
    test_policy.test_vanilla_distribution_unchanged(tables, names)


def test_kaizo_layout_shapes(kaizo_tables, kaizo_roark_party):
    t = kaizo_tables
    assert layout_of(t) is KAIZO and KAIZO.per_slot_sites
    # Sized for Selfdestruct (K3c Task 6): EvalAttack :6374 and :6383, Expert_Explosion
    # :1859, :1865, :1870, :1875, :1880, Risky :6530.
    assert KAIZO.slot_sites == 8
    assert KAIZO.transcribed == frozenset(EFFECT_SITE_LINES)
    assert max(len(sites) for _, sites in KAIZO.effect_sites) <= KAIZO.slot_sites
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, PURSUIT, LEER, STEALTH_ROCK))
    o = sample_ai_outcome(jax.random.PRNGKey(0), t, s, KAIZO)
    assert o.script_rolls.shape == (MOVES, KAIZO.slot_sites)
    with pytest.raises(ValueError, match="script_rolls has shape"):
        policy.ai_scores(t, s, outcome())  # a flat outcome under the per-slot layout


def test_site_lines_are_the_scripts_rolls():
    """Every line of ``SITE_N`` is an ``IfRandomLessThan N`` of script.s with that N."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = root / "external/pokeplatinum/src/battle/trainer_ai/script.s"
    if not script.exists():
        pytest.skip("decomp not present")
    lines = script.read_text(encoding="utf-8").splitlines()
    for line, n in SITE_N.items():
        assert lines[line - 1].split()[:2] == ["IfRandomLessThan", f"{n},"], line


def test_slot_site_table(kaizo_tables, kaizo_names):
    """The site table is keyed by battle effect (``IfCurrentMoveEffectEqualTo``,
    trainer_ai.c:1441), not by move id: Accelerock (Kaizo 205, effect HIT) gets the
    EvalAttack quad roll (:6383) as Rock Throw does. This replaces Task 2's id-keyed guard."""
    t, n = kaizo_tables, kaizo_names
    names = {ROCK_THROW: "Rock Throw", ACCELEROCK: "Accelerock", PURSUIT: "Pursuit"}
    for move, name in names.items():
        assert n.move_names[move] == name
    expected = {
        ROCK_THROW: [(6383, 80)],
        ACCELEROCK: [(6383, 80)],
        HEADBUTT: [(6383, 80)],
        LEER: [(6383, 80), (2328, 50)],
        SCREECH: [(6383, 80), (2328, 50)],
        PURSUIT: [(6383, 80), (3726, 128), (3731, 128)],
        STEALTH_ROCK: [(6383, 80), (6274, 128)],
        0: [],
        # Selfdestruct, HALVE_DEFENSE: EvalAttack_MaybeDeprioritize's 51 roll (script.s:6365
        # -> :6374) before the quad roll, then Expert_Explosion's (:1859, :1865, :1870,
        # :1875, :1880) and Risky_Main's (:6530): the 8 columns of KAIZO.slot_sites.
        120: [
            (6374, 51),
            (6383, 80),
            (1859, 128),
            (1865, 50),
            (1870, 128),
            (1875, 50),
            (1880, 50),
            (6530, 128),
        ],
    }
    for move, sites in expected.items():
        n_, lines = (np.asarray(a).tolist() for a in slot_site_table(t, KAIZO, move))
        assert list(zip(lines, n_, strict=True))[: len(sites)] == sites, move
        assert n_[len(sites) :] == [0] * (KAIZO.slot_sites - len(sites)), move


def test_every_transcribed_effect_has_a_site_row():
    """Task 2's id guard is gone: a Kaizo move is transcribed by its effect alone, so every
    transcribed effect needs its row of the site table (an effect without one would lose
    its sites silently)."""
    assert {e for e, _ in KAIZO.effect_sites} == KAIZO.transcribed
    for _, sites in KAIZO.effect_sites:
        assert (6383, 80) in sites  # EvalAttack's quad roll, reached by every effect
        assert all(SITE_N[line] == n for line, n in sites)
        # Script order: flag order (trainer_ai.c:297-310), EvalAttack_Main (:6353-6415), then
        # Expert_Main's labels (:1623-6352), then Risky_Main (:6521-6562); then line order.
        lines = [line for line, _ in sites]
        flag = lambda line: 0 if 6353 <= line <= 6415 else 1 if 1623 <= line <= 6352 else 2
        assert lines == sorted(lines, key=lambda line: (flag(line), line))


def test_kaizo_accelerock_is_transcribed_with_its_site(kaizo_tables, kaizo_roark_party):
    """Task 2 review m2: the id guard refused Accelerock; keyed by effect it is accepted,
    with its EvalAttack site (exact equals loose on a quad-effective Accelerock)."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ACCELEROCK, LEER, 0, 0))
    s = _bulky(_with_player(t, s, 146))  # Moltres: Fire/Flying, Rock is QUADRUPLE
    assert policy.ai_refusals(t, s) == []
    exact = np.asarray(_distribution(t, s, KAIZO))
    np.testing.assert_allclose(exact, policy.ai_action_distribution_loose(t, s), atol=1e-6)


def test_kaizo_effect_check_refuses_an_untranscribed_effect(kaizo_tables, kaizo_roark_party):
    """Task 2 review m3: with the id guard gone the effect check carries the refusal. Rock
    Throw given STATUS_SLEEP (1, untranscribed) is refused."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, LEER, 0, 0))
    assert policy.ai_refusals(t, s) == []
    assert 1 not in KAIZO.transcribed
    t7 = t._replace(move_ai_effect=t.move_ai_effect.at[ROCK_THROW].set(1))
    reasons = policy.ai_refusals(t7, s)
    assert any("Rock Throw are not transcribed" in r for r in reasons), reasons


# --- Width ---------------------------------------------------------------------------------


def test_kaizo_width_under_cap(kaizo_tables, kaizo_roark_party):
    """``ai_width`` (the combinations the factored distribution enumerates, Ruling 12) is
    within ``WIDTH_CAP`` for each of Kaizo Roark's six as the active, and so is its largest
    value for any state; the flat product is reported, not used."""
    t, party = kaizo_tables, kaizo_roark_party
    for j in range(len(party)):
        s = make_state(t, [party[2]], [party[j], *(m for k, m in enumerate(party) if k != j)])
        width = policy.ai_width(t, s)
        assert 10 + 2 * MOVES <= width <= WIDTH_CAP, j
        counts = np.asarray(slot_roll_thresholds(t, s, KAIZO) > 0).sum(axis=1)
        assert width == 10 + 2 * int(np.sum(2**counts))
        assert policy.ai_flat_width(t, s) == 2 * 2 ** int(counts.sum()) * 12 * 10
    assert 10 + 2 * MOVES * 2**KAIZO.slot_sites <= WIDTH_CAP


# --- Sampling ------------------------------------------------------------------------------


def test_kaizo_sampled_frequencies_match(kaizo_tables, kaizo_roark_party):
    """20 000 ``sample_ai_outcome`` draws through ``ai_policy`` match the factored
    distribution (chi-square, p > 1e-3)."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, LEER, SCREECH, PURSUIT))
    s = _bulky(_with_player(t, s, 72))  # Tentacool (Kaizo: ability 44 / Clear Body)
    s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1), hp=s.hp.at[1, 0].set(20))
    dist = np.asarray(_distribution(t, s, KAIZO), np.float64)
    assert np.count_nonzero(dist) >= 2
    keys = jax.random.split(jax.random.PRNGKey(7), 20_000)
    act = jax.jit(
        jax.vmap(lambda k: policy._policy(t, s, sample_ai_outcome(k, t, s, KAIZO), KAIZO))
    )(keys)
    counts = np.bincount(np.asarray(act), minlength=NUM_OPP_ACTIONS)
    support = dist > 0
    assert counts[~support].sum() == 0
    expected = dist[support] / dist[support].sum() * counts.sum()
    assert chisquare(counts[support], expected).pvalue > 1e-3


# --- Factoring -----------------------------------------------------------------------------


def test_shared_guess_couples_slots(kaizo_tables, kaizo_names, kaizo_roark_party):
    """One guess for the whole decision (the exact layout): against Tentacool (ability 44 /
    Clear Body), Leer and Screech both lose 10 on the Clear Body guess
    (Basic_CheckClearBodyEffect, script.s:469-472), so they move together. Factoring the
    slots with the guess marginalised per slot would split them."""
    t = kaizo_tables
    assert np.asarray(t.species_abilities_decomp)[72].tolist() == [44, CLEAR_BODY]
    s = _kaizo_state(t, kaizo_roark_party, (LEER, SCREECH, TACKLE, 0))
    s = _bulky(_with_player(t, s, 72))
    s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1))
    joint = _joint_moves(t, s, KAIZO)
    np.testing.assert_allclose(np.asarray(_factored(t, s, KAIZO)), joint, atol=1e-6)
    assert joint[0] > 0 and joint[1] > 0  # Leer and Screech tie on the other guess


def test_factored_full_distribution_matches_enumerated(kaizo_tables, kaizo_roark_party):
    """``_distribution`` (switch, item, factored moves) equals ``ai_policy`` over every
    joint outcome (``enumerate_ai_outcomes`` with the per-slot layout)."""
    t = kaizo_tables
    for moves, player in (
        ((ROCK_THROW, LEER, SCREECH, PURSUIT), 72),
        ((ACCELEROCK, STEALTH_ROCK, TACKLE, 0), 146),
        ((PURSUIT, HEADBUTT, 0, 0), 72),
    ):
        s = _kaizo_state(t, kaizo_roark_party, moves)
        s = _bulky(_with_player(t, s, player))
        s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1))
        exact = np.asarray(_distribution(t, s, KAIZO))
        enumerated = np.asarray(policy.ai_action_distribution_enumerated(t, s, KAIZO))
        np.testing.assert_allclose(exact, enumerated, atol=2e-6)


@pytest.mark.slow
@_SETTINGS
@given(data=st.data())
def test_factored_equals_joint_bruteforce(kaizo_tables, kaizo_roark_party, data):
    """On random Kaizo decisions and synthetic site tables, the factored move pick equals a
    joint enumeration over all slots' bits, both guesses and the 12 tiebreaks, when the
    moveset has at most 14 sites."""
    t = kaizo_tables
    s = data.draw(kaizo_states(t, kaizo_roark_party))
    layout = data.draw(site_layouts())
    assume(not bool(policy._refused(t, s, layout).scoring))
    assume(int(np.sum(np.asarray(slot_roll_thresholds(t, s, layout)) > 0)) <= 14)
    np.testing.assert_allclose(
        np.asarray(_factored(t, s, layout)), _joint_moves(t, s, layout), atol=1e-5
    )


# --- Exact against loose -------------------------------------------------------------------


_slot_histograms = jax.jit(policy._slot_histograms, static_argnames="layout")


def _exact_slot_scores(t, s, layout) -> np.ndarray:
    """``float64[MOVES, 256]``: each slot's exact score distribution, the guesses averaged
    (``policy._slot_histograms``)."""
    _, hist = _slot_histograms(t, s, layout)
    return np.asarray(hist, np.float64).mean(axis=0)


def _loose_slot_scores(t, s, layout) -> np.ndarray:
    """``float64[MOVES, 256]``: each slot's loose score distribution (every read its own
    draw)."""
    out = np.zeros((MOVES, 256))
    for m, values in enumerate(policy._loose_slot_scores(t, s, layout)):
        for v, p in values.items():
            out[m, v] += p
    return out


def _assert_exact_equals_loose(t, s, layout=KAIZO) -> None:
    """The exact and loose action distributions are NaN on the same states and otherwise
    equal; so is each slot's score distribution before the pick (which catches a wrong site
    or a wrong slot weight even where the pick hides it). The loose side enumerates the
    slots jointly (its default ``joint_bits``), so it does not lean on per-slot factoring."""
    exact = np.asarray(_distribution(t, s, layout), np.float64)
    loose = policy.ai_action_distribution_loose(t, s, layout)
    np.testing.assert_array_equal(np.isnan(exact), np.isnan(loose))
    np.testing.assert_allclose(exact, loose, atol=1e-5)
    if not np.isnan(exact).any():
        np.testing.assert_allclose(
            _exact_slot_scores(t, s, layout), _loose_slot_scores(t, s, layout), atol=1e-6
        )


@pytest.mark.slow
@settings(_SETTINGS, max_examples=20)
@given(data=st.data())
def test_exact_equals_loose(kaizo_tables, kaizo_roark_party, data):
    """On random Kaizo decisions where at most one ability read can change a score (the
    condition Ruling 6 needs, pinned for Roark by test_one_scalar_guess_is_exact_for_roark),
    the exact distribution and each slot's score distribution equal the loose ones, where
    every ``IfRandomLessThan`` and every ``LoadBattlerAbility DEFENDER`` is its own draw.
    Both are NaN on the same refused states, the voluntary-switch refusals included."""
    t = kaizo_tables
    s = data.draw(kaizo_states(t, kaizo_roark_party))
    assume(_sensitive_reads(t, _one(s), KAIZO)[0] <= 1)
    _assert_exact_equals_loose(t, s, KAIZO)


@pytest.mark.parametrize(
    ("species", "abilities", "name"),
    [("Whismur", [113, SOUNDPROOF], "Scrappy"), ("Magnezone", [26, 42], "Levitate")],
)
def test_loose_is_nan_on_a_switching_refusal(
    kaizo_tables, kaizo_names, kaizo_roark_party, species, abilities, name
):
    """Review I1: a player ability the switch check does not implement
    (``switching.PLAYER_ABILITY_REFUSALS``): Whismur's Scrappy, and Magnezone's Levitate
    (species 462, the full suite's failing draw). The exact distribution is NaN and the
    loose oracle is NaN too, rather than raising ``UnsupportedSwitching``."""
    t, n = kaizo_tables, kaizo_names
    sp = n.species(species)
    assert np.asarray(t.species_abilities_decomp)[sp].tolist() == abilities
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, LEER, 0, 0))
    s = _bulky(_with_player(t, s, sp, 0))
    assert any(name in r for r in policy.ai_refusals(t, s, n))
    assert np.isnan(np.asarray(_distribution(t, s, KAIZO))).all()
    assert np.isnan(policy.ai_action_distribution_loose(t, s, KAIZO)).all()


def test_slot_scores_catch_a_missing_expert_site(kaizo_tables, kaizo_roark_party):
    """Review m2: dropping DEF_DOWN's Expert_StatusDefenseDown roll (script.s:2328) from the
    exact table can leave the pick unchanged (Leer loses to Rock Throw either way) but not
    Leer's score distribution, which the loose oracle draws for itself."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, LEER, 0, 0))
    s = _bulky(_with_player(t, s, 146))
    # Bonsly below 70% HP (IfHPPercentLessThan AI_BATTLER_ATTACKER, 70, script.s:2324): the
    # :2328 roll is reached.
    s = s._replace(hp=s.hp.at[1, 0].set(5))
    _assert_exact_equals_loose(t, s)
    def_down = 19  # BATTLE_EFFECT_DEF_DOWN (Expert_Main's entry, script.s:1642)
    rows = tuple(
        (e, tuple(x for x in sites if x[0] != 2328) if e == def_down else sites)
        for e, sites in KAIZO.effect_sites
    )
    dropped = KAIZO._replace(effect_sites=rows)
    assert not np.allclose(
        _exact_slot_scores(t, s, dropped), _loose_slot_scores(t, s, KAIZO), atol=1e-3
    )


def test_loose_joint_equals_per_slot(kaizo_tables, kaizo_roark_party):
    """The loose oracle's two modes agree: every read enumerated jointly, and each slot on
    its own reads (exact in loose, where no two slots share a draw)."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, PURSUIT, 0, 0))
    s = _bulky(_with_player(t, s, 146))
    joint = policy.ai_action_distribution_loose(t, s, KAIZO, joint_bits=18)
    per_slot = policy.ai_action_distribution_loose(t, s, KAIZO, joint_bits=0)
    np.testing.assert_allclose(joint, per_slot, atol=1e-9)
    np.testing.assert_allclose(np.asarray(_distribution(t, s, KAIZO)), joint, atol=1e-6)


def test_loose_catches_a_missing_site(kaizo_tables, kaizo_roark_party):
    """The path-only sites gate has teeth: a site table that drops HIT's EvalAttack quad
    roll (script.s:6383) gives Rock Throw a certain +2 against a quad-weak target, where
    the decomp draws (80/256 to skip it). Loose reads its own draw and disagrees."""
    t = kaizo_tables
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, LEER, 0, 0))
    s = _bulky(_with_player(t, s, 146))  # Moltres, Fire/Flying
    loose = policy.ai_action_distribution_loose(t, s, KAIZO)
    np.testing.assert_allclose(np.asarray(_distribution(t, s, KAIZO)), loose, atol=1e-6)
    rows = tuple((e, sites[1:] if e == 0 else sites) for e, sites in KAIZO.effect_sites)
    dropped = KAIZO._replace(effect_sites=rows)
    assert not np.allclose(np.asarray(_distribution(t, s, dropped)), loose, atol=1e-3)


def test_refused_states_are_nan_on_both_sides(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Task 5 review m3: on a state the real ``KAIZO`` layout refuses, the exact and the loose
    distributions are both NaN (both read ``policy._refused``).
    - Real: Kaizo Roark's Shuckle [Toxic, Accelerock, Curse, Sand Tomb] made Ghost-type, so
      its Curse takes Basic_CheckCurse_GhostType (script.s:681-684 -> :700), not transcribed.
    - Synthetic: no Kaizo Roark state is refused for an Expert label since K3c Task 6, so the
      guard is opened by a layout that lacks Expert_Recovery (RESTORE_HALF_HP, script.s:1652):
      Corsola [Recover, Water Pulse, Light Screen, Accelerock] is refused under it and
      accepted under ``KAIZO``."""
    t, n, party = kaizo_tables, kaizo_names, kaizo_roark_party
    shuckle = make_state(t, [party[2]], [party[4], party[0]])
    assert policy.ai_refusals(t, shuckle, n) == []
    ghost = shuckle._replace(types=shuckle.types.at[1, 0, 1].set(7))  # TYPE_GHOST
    assert any("Basic_CheckCurse_GhostType" in r for r in policy.ai_refusals(t, ghost, n))
    corsola = make_state(t, [party[2]], [party[3], party[0]])
    assert policy.ai_refusals(t, corsola, n) == []
    no_recovery = KAIZO._replace(expert_transcribed=KAIZO.expert_transcribed - {32})
    assert any("Expert_Recovery" in r for r in policy.ai_refusals(t, corsola, n, no_recovery))
    for s, layout in ((ghost, KAIZO), (corsola, no_recovery)):
        assert np.isnan(np.asarray(_distribution(t, s, layout))).all()
        assert np.isnan(policy.ai_action_distribution_loose(t, s, layout)).all()
    assert not np.isnan(np.asarray(_distribution(t, corsola, KAIZO))).any()


def test_loose_catches_a_split_guess(kaizo_tables, kaizo_roark_party):
    """The one-scalar guess gate has teeth: a species with Soundproof and Clear Body (made
    up) against Screech has two reads that change its score, Basic_CheckSoundproof's
    (script.s:117) and Basic_CheckClearBodyEffect's (:470). One guess scores Screech 90 on
    either ability; the decomp's two draws leave it at 100 one time in four."""
    t = kaizo_tables
    species = 72
    abilities = t.species_abilities_decomp.at[species].set(
        jnp.asarray([SOUNDPROOF, CLEAR_BODY], t.species_abilities_decomp.dtype)
    )
    t2 = t._replace(species_abilities_decomp=abilities)
    s = _kaizo_state(t2, kaizo_roark_party, (SCREECH, TACKLE, 0, 0))
    s = _bulky(_with_player(t2, s, species))
    s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1))
    assert _sensitive_reads(t2, _one(s), KAIZO)[0] == 2
    exact = np.asarray(_distribution(t2, s, KAIZO))
    loose = policy.ai_action_distribution_loose(t2, s, KAIZO)
    assert not np.allclose(exact, loose, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("game", ["vanilla", "kaizo"])
def test_one_scalar_guess_is_exact_for_roark(game, tables, names, kaizo_tables, kaizo_roark_party):
    """Ruling 6 as data: for every species of the game's table with two abilities, against
    each of Roark's Pokemon as the active (Kaizo's six, vanilla's three), at most one
    ``LoadBattlerAbility DEFENDER`` read of a decision changes a score. Re-run whenever a
    task adds reads: the loose oracle finds every read on every path."""
    if game == "kaizo":
        t, party = kaizo_tables, kaizo_roark_party
        bases = [
            make_state(t, [party[2]], [party[j], *(m for k, m in enumerate(party) if k != j)])
            for j in range(len(party))
        ]
        layout = KAIZO
    else:
        t, layout = tables, VANILLA
        bases = [versus(t, names, j) for j in range(3)]
    species = _two_ability_species(t)
    for j, base in enumerate(bases):
        states = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *(_bulky(_with_player(t, base, sp), int(base.max_hp[0, 0])) for sp in species),
        )
        count = _sensitive_reads(t, states, layout)
        bad = [int(species[i]) for i in np.flatnonzero(count > 1)]
        assert not bad, (game, j, bad)


def _random_kaizo_rows(t, party, n: int, seed: int) -> BattleState:
    """``n`` random Kaizo decisions (numpy batch): Bonsly's moves from the transcribed pool,
    a two-ability player species, random HP on both sides and Defense stage."""
    rng = np.random.default_rng(seed)
    pool = _move_pool(t, KAIZO)  # ai_rows uses KAIZO: no move it refuses
    species = _two_ability_species(t)
    states = []
    for _ in range(n):
        moves = [int(rng.choice(pool))] + [int(rng.choice((0, *pool))) for _ in range(3)]
        s = _kaizo_state(t, party, moves)
        s = _with_player(t, s, int(rng.choice(species)), int(rng.integers(2)))
        s = _bulky(s, int(rng.integers(20, 400)))
        s = s._replace(
            hp=s.hp.at[1, 0].set(int(rng.integers(1, int(s.max_hp[1, 0]) + 1))),
            stages=s.stages.at[0, Stat.DEF].set(int(rng.integers(-6, 7))),
        )
        states.append(s)
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *states)


@pytest.mark.slow
def test_kaizo_ai_rows_do_not_depend_on_the_bucket(kaizo_tables, kaizo_roark_party):
    """Review I2: ``expand.ai_rows`` with the Kaizo layout gives bit-identical rows for 257
    random Kaizo decisions across the bucket boundary (255, 256 and 257 rows: buckets 256
    and 512), for calls of 3, 7 and 300 rows, and against one ``CHUNK``-row call: padding
    never reaches the output, and the per-row program does not change with the call size
    (``expand._ai``)."""
    from nuzline.search.expand import CHUNK, ai_rows

    t = kaizo_tables
    rows = _random_kaizo_rows(t, kaizo_roark_party, 257, seed=3)
    ref = ai_rows(t, rows)  # bucket 512
    assert (~np.isnan(ref).any(axis=1)).sum() > 100  # mostly accepted states
    for n in (255, 256):
        assert ai_rows(t, take(rows, np.arange(n))).tobytes() == ref[:n].tobytes(), n
    for per_call in (3, 7, 300, CHUNK):
        assert ai_rows(t, rows, per_call=per_call).tobytes() == ref.tobytes(), per_call


_DEVICES_SCRIPT = """
import hashlib, jax
from nuzline.data.games import GAMES, KAIZO
from nuzline.data.tables import load_tables
from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
from nuzline.search.expand import ai_rows
from tests.ai.test_kaizo_outcome import _random_kaizo_rows

assert jax.device_count() == DEVICES, jax.device_count()
d = GAMES[KAIZO]
t = load_tables(d.tables_dir)
roark = next(x for x in load_trainers(d.intermediate_dir) if x.name == "Roark")
party = build_trainer_party(roark, IntermediateData(d.intermediate_dir))
rows = _random_kaizo_rows(t, party, 40, seed=5)
print(hashlib.sha256(ai_rows(t, rows).tobytes()).hexdigest())
"""


def _kaizo_ai_rows_on_devices(devices: int) -> str:
    """The sha256 of ``ai_rows`` of 40 random Kaizo decisions in a subprocess with
    ``devices`` host devices (pattern: tests/search/test_expand.py)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
    }
    done = subprocess.run(
        [sys.executable, "-c", f"DEVICES = {devices}\n" + _DEVICES_SCRIPT],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_kaizo_ai_rows_do_not_depend_on_the_device_count(kaizo_tables):
    """Review m-R1: the Kaizo ``ai_rows`` (``lax.map`` per device under ``shard_map``) give
    the same bytes on 1 and on 3 host devices (3 divides no bucket, so the padding
    differs too)."""
    assert _kaizo_ai_rows_on_devices(1) == _kaizo_ai_rows_on_devices(3)
