"""The Kaizo AI outcome layout, its factored distribution and the loose oracle (K3c Task 3;
docs/ai/kaizo_roark.md 8).

Decomp paths are relative to ``external/pokeplatinum``: ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``, ``script.s`` is ``src/battle/trainer_ai/script.s``.

The exact Kaizo distribution merges draws through four gates; each is checked against an
oracle that does not use it:
- per-slot factoring and the tie subsets: the factored pick against a joint enumeration of
  every slot's reads and the 12 tiebreaks (``test_factored_equals_joint_bruteforce``);
- every random read keyed by its order on the slot's path, its probability taken from the
  read (E1 Rulings R1, R4): exact against loose, where every ``IfRandomLessThan`` and every
  ``LoadBattlerAbility DEFENDER`` is its own draw in the scripts' static order, on every
  state, two score-changing guesses in two slots and in one slot included
  (``test_exact_equals_loose``, ``test_each_slot_guesses_on_its_own``,
  ``test_two_guesses_in_one_slot``), and the number of distinct paths per slot equals the
  loose one (``test_kaizo_width_under_cap``);
- the read bound: a path past ``slot_reads`` is refused
  (``test_a_read_past_the_bound_is_refused``), and the bound is the data's
  (``test_slot_reads_fit_every_transcribed_effect``).
The fourth gate, the dropped speed-tie draw (Ruling 3), is checked in
``tests/ai/test_kaizo_commands.py`` (``test_speed_tie_never_slower``): ``IfSpeedCompareEqualTo``
gives the tie its own loose read (``commands.ReadKind.SPEED_TIE``). Since K3c Task 6 its three
compares are on transcribed paths (Expert_Explosion, Expert_StatusSpeedUp and Expert_Recovery,
script.s:1864, :2070, :2650), and ``tests/ai/test_kaizo_policy.py
test_exact_equals_loose_at_equal_speed`` checks exact against loose at equal speed on each.

The tests run on ``KAIZO`` restricted to the K3c transcription's 24 effects (``ROARK_EFFECTS``:
Kaizo Roark's party's and the K3c test moves'), the ``kaizo_tables`` fight's (E1 Ruling R7: a fight's programs trace its own moves' labels only), so no
test compiles a program wider than a real fight's (E1 batch 8: a program of every transcribed
effect reached 20 GB in one test module).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, target
from hypothesis import strategies as st
from scipy.stats import chisquare

from nuzline.ai import commands, policy
from nuzline.ai.commands import ReadKind
from nuzline.ai.layout import KAIZO as KAIZO_ALL
from nuzline.ai.layout import VANILLA, kaizo_layout, layout_of
from nuzline.ai.outcome import (
    ENUMERATION_WIDTH,
    KAIZO_EXPERT_LABELS,
    KAIZO_TRANSCRIBED,
    MAX_ENUMERATED,
    MAX_SCRIPT_ROLLS,
    READ_DRAWS,
    SIM_ROLL_MAX,
    SITE_N,
    SWITCH_NONE,
    TIEBREAK_MOD,
    TIEBREAK_P,
    AiOutcome,
    enumerate_ai_outcomes,
    sample_ai_outcome,
)
from nuzline.engine.actions import NUM_OPP_ACTIONS
from nuzline.engine.enumerate import WIDTH_CAP
from nuzline.engine.ids import Stat
from nuzline.engine.state import MOVES, BattleState, make_state
from nuzline.search.rows import take
from tests.ai.fixtures import GEODUDE, outcome, versus
from tests.conftest import K3C_EFFECTS

# Kaizo move ids (data/gen4_platinum_kaizo names).
ROCK_THROW, PURSUIT, LEER, SCREECH, HEADBUTT, TACKLE = 88, 228, 43, 103, 29, 33
STEALTH_ROCK, ACCELEROCK, WATER_GUN, U_TURN, TOXIC = 446, 205, 55, 369, 92
SELFDESTRUCT = 120
# The K3c transcription's effects (tests/conftest.py K3C_EFFECTS): the layout these tests
# ran on, the kaizo_tables fight's (test_kaizo_layout_shapes).
ROARK_EFFECTS = K3C_EFFECTS
KAIZO = kaizo_layout(KAIZO_ALL.flags, KAIZO_ALL.slot_reads, ROARK_EFFECTS)

GIBLE = 443  # species: Sand Veil / Rough Skin, Dragon/Ground
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
_weight_batch = jax.jit(
    jax.vmap(policy._path_weight, in_axes=(None, None, 0, None)), static_argnames="layout"
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


def layout_for(t, flags: int, s: BattleState):
    """The Kaizo layout of flag set ``flags`` for ``s``'s opponent party: the read bound and the
    transcribed effects of its moves (E1 Rulings R6, R7), as ``ai.layout.fight_tables`` gives
    a fight."""
    from nuzline.ai.layout import fight_layout

    return fight_layout(t, flags, np.asarray(s.moves[1]).ravel())


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


# --- Oracles -------------------------------------------------------------------------------


def _joint_reads(t, s, layout) -> list[tuple[int, int]]:
    """The (slot, read) of every read some path of the slot makes (``policy._path_records``):
    as many reads per slot as its longest path."""
    _, reads = policy._path_records(t, s, layout)
    columns = np.asarray(reads).max(axis=0)
    return [(m, k) for m in range(MOVES) for k in range(int(columns[m]))]


def _joint_moves(t, s, layout) -> np.ndarray:
    """``float64[MOVES]``: the move pick by a joint enumeration of every slot's reads
    (``_joint_reads``) and the 12 tiebreaks, through ``_scores`` and ``_pick`` (no factoring),
    each pattern weighted by its paths (``policy._path_weight``). NaN when a scoring is
    refused (-1)."""
    bits = _joint_reads(t, s, layout)
    on = (np.arange(2 ** len(bits))[:, None] >> np.arange(len(bits))) & 1
    k = len(on)
    rolls = np.full((k, MOVES, layout.slot_reads), READ_DRAWS[0], np.uint8)
    for i, (m, r) in enumerate(bits):
        rolls[:, m, r] = np.asarray(READ_DRAWS, np.uint8)[on[:, i]]
    outs = AiOutcome(
        sim_roll=jnp.full((k, MOVES), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.zeros(k, jnp.int8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.zeros(k, jnp.int8),
        switch_stop=jnp.full(k, SWITCH_NONE, jnp.int8),
    )
    scores = _scores_batch(t, s, outs, layout)
    if np.any(np.asarray(scores) < 0):
        return np.full(MOVES, np.nan)
    weight = np.asarray(_weight_batch(t, s, outs, layout), np.float64)
    picks = np.asarray(_pick_batch(t, s, scores, jnp.arange(TIEBREAK_MOD, dtype=jnp.int8)))
    share = weight[:, None] * TIEBREAK_P[None, :]
    dist = np.zeros(MOVES)
    for m in range(MOVES):
        dist[m] += share[picks == m].sum()
    return dist


def _loose_paths(t, s, layout=KAIZO) -> np.ndarray:
    """``int[MOVES]``: per slot, its distinct scoring paths by the loose oracle: the distinct
    (reads on the path, their bits) over every pattern of the slot's reads, counting the reads
    that draw a ``RandNext`` the scoring can split on: every ``IfRandomLessThan``, and a
    ``LoadBattlerAbility DEFENDER`` when the decision guesses (trainer_ai.c:1191-1198; loose
    flips a coin at every such read, which only resolves to the known ability otherwise). The
    speed tie of a SLOWER compare is left out (Ruling 3: it never changes it); a FASTER
    compare's is counted (E1 Ruling R8)."""
    batch = _one(s)
    reads, _, kind = policy._loose_reads(t, batch, layout)
    bits = policy._loose_patterns(reads)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), layout)
    guessed = bool(commands.ability_guessed(t, s))
    draws = (kind == commands.ReadKind.ROLL) | ((kind == commands.ReadKind.ABILITY) & guessed)
    draws |= kind == commands.ReadKind.SPEED_TIE_DRAWN
    live = np.asarray(lo.live)[0] & draws[None]  # [P, MOVES, LOOSE_READS]
    out = []
    for m in range(MOVES):
        paths = {
            (tuple(live[p, m]), tuple(bits[p, m] * live[p, m])) for p in range(2 ** len(reads[m]))
        }
        out.append(len(paths))
    return np.array(out)


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


def _equal_speed(s: BattleState) -> BattleState:
    """``s`` with the player's active Speed stat set to the AI's (``stats[..., Stat.SPE - 1]``)."""
    spe = Stat.SPE - 1
    ai = s.stats[1, int(s.active[1]), spe]
    return s._replace(stats=s.stats.at[0, int(s.active[0]), spe].set(ai))


# --- The layout ----------------------------------------------------------------------------


def test_vanilla_layout_frozen(tables, names):
    """The flat layout is unchanged (plan K3c Global Constraints): its shapes, its width and
    Task 2's bit-equality fixtures (tests/ai/test_policy.py FROZEN_VANILLA)."""
    from tests.ai import test_policy

    assert (ENUMERATION_WIDTH, MAX_ENUMERATED, MAX_SCRIPT_ROLLS) == (3840, 4096, 4)
    assert not VANILLA.per_slot_sites and VANILLA.slot_reads == 0
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
    # Roark's fight layout is the tests' (E1 Ruling R7): his party's effects only
    assert layout_of(t) is KAIZO and KAIZO.per_slot_sites
    assert KAIZO.transcribed == ROARK_EFFECTS < KAIZO_TRANSCRIBED
    s = _kaizo_state(t, kaizo_roark_party, (ROCK_THROW, PURSUIT, LEER, STEALTH_ROCK))
    o = sample_ai_outcome(jax.random.PRNGKey(0), t, s, KAIZO)
    assert o.script_rolls.shape == (MOVES, KAIZO.slot_reads) and o.script_rolls.dtype == jnp.uint8
    assert o.ability_guess.shape == ()
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


def test_kaizo_accelerock_is_transcribed_with_its_site(kaizo_tables, kaizo_roark_party):
    """Task 2 review m2: the id guard refused Accelerock; keyed by effect it is accepted,
    with its EvalAttack roll (exact equals loose on a quad-effective Accelerock)."""
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
    """``ai_width`` (the combinations the factored distribution merges, Ruling 12) is the
    switch stops plus each slot's distinct scoring paths, which equal the loose oracle's (its
    reads in static order: the read order keys one pattern per path, E1 Ruling R4), for each of
    Kaizo Roark's six as the active against a player the AI guesses (Roark's Gible, Sand Veil /
    Rough Skin) and one whose ability it knows. Its largest value for any state is within
    ``WIDTH_CAP``; the flat product is reported, not used."""
    t, party = kaizo_tables, kaizo_roark_party
    stops = KAIZO.switch_sites + 1  # 45: the six-party switch stops (K3c Task 7)
    for j in range(len(party)):
        s = make_state(t, [party[2]], [party[j], *(m for k, m in enumerate(party) if k != j)])
        known = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))
        for state in (s, known):
            paths = _loose_paths(t, state)
            width = policy.ai_width(t, state)
            assert width == stops + int(paths.sum()) <= WIDTH_CAP, j
            assert policy.ai_flat_width(t, state) == int(np.prod(paths)) * 12 * stops
    most = KAIZO.switch_sites + 1 + MOVES * 2**KAIZO.slot_reads
    assert most <= WIDTH_CAP


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


def test_each_slot_guesses_on_its_own(kaizo_tables, kaizo_names, kaizo_roark_party):
    """E1 Ruling R1, two slots: against Tentacool (Kaizo abilities 44 / Clear Body), Leer and
    Screech each lose 10 on a Clear Body guess (Basic_CheckClearBodyEffect, script.s:469-472).
    Each read is its own coin (trainer_ai.c:1194-1198), so the two slots guess independently:
    the factored pick equals the joint enumeration of both slots' guess columns and the loose
    oracle. (One guess for the decision moved them together: 1/6 each where the decomp gives
    5/24.)"""
    t = kaizo_tables
    assert np.asarray(t.species_abilities_decomp)[72].tolist() == [44, CLEAR_BODY]
    s = _kaizo_state(t, kaizo_roark_party, (LEER, SCREECH, TACKLE, 0))
    s = _bulky(_with_player(t, s, 72))
    s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1))
    assert _sensitive_reads(t, _one(s), KAIZO)[0] == 2
    joint = _joint_moves(t, s, KAIZO)
    np.testing.assert_allclose(np.asarray(_factored(t, s, KAIZO)), joint, atol=1e-6)
    np.testing.assert_allclose(joint, [5 / 24, 5 / 24, 14 / 24, 0], atol=1e-4)
    _assert_exact_equals_loose(t, s)


def test_factored_full_distribution_matches_enumerated(kaizo_tables, kaizo_roark_party):
    """``_distribution`` (switch, item, factored moves) equals ``ai_policy`` over every
    joint outcome (``enumerate_ai_outcomes`` with the per-slot layout)."""
    t = kaizo_tables
    for moves, player, known in (
        ((ROCK_THROW, LEER, SCREECH, PURSUIT), 72, True),
        ((ACCELEROCK, STEALTH_ROCK, TACKLE, 0), 146, True),
        ((PURSUIT, HEADBUTT, 0, 0), 72, True),
        ((LEER, SCREECH, 0, 0), 72, False),  # 4 sites and 4 guess columns reached
    ):
        s = _kaizo_state(t, kaizo_roark_party, moves)
        s = _bulky(_with_player(t, s, player, known=known))
        s = s._replace(stages=s.stages.at[0, Stat.DEF].set(-1))
        exact = np.asarray(_distribution(t, s, KAIZO))
        enumerated = np.asarray(policy.ai_action_distribution_enumerated(t, s, KAIZO))
        np.testing.assert_allclose(exact, enumerated, atol=2e-6)


@pytest.mark.slow
@_SETTINGS
@given(data=st.data())
def test_factored_equals_joint_bruteforce(kaizo_tables, kaizo_roark_party, data):
    """On random Kaizo decisions the factored move pick equals a joint enumeration over all
    slots' reads and the 12 tiebreaks, when those number at most 16 bits. The claim needs
    slots whose score depends on a draw, ideally several: ``target`` steers the search to
    more of them, and ``test_factored_equals_joint_with_two_random_slots`` is the fixed
    floor of two."""
    t = kaizo_tables
    s = data.draw(kaizo_states(t, kaizo_roark_party))
    if data.draw(st.booleans()):
        # Two slots whose score depends on a roll (the claim's case): Rock Throw, QUADRUPLE
        # into a Fire/Flying player (EvalAttack :6383), and Light Screen at full HP
        # (Expert_LightScreen :2718; script.s:2716-2719).
        light_screen = 113
        s = s._replace(
            moves=s.moves.at[1, 0, 2].set(ROCK_THROW).at[1, 0, 3].set(light_screen),
            pp=s.pp.at[1, 0, 2].set(5).at[1, 0, 3].set(5),
            types=s.types.at[0, 0].set(jnp.asarray([10, 2], s.types.dtype)),
            hp=s.hp.at[1, 0].set(s.max_hp[1, 0]),
        )
    assume(not bool(policy._refused(t, s, KAIZO).scoring))
    assume(len(_joint_reads(t, s, KAIZO)) <= 16)
    factored = np.asarray(_factored(t, s, KAIZO))
    assert not np.isnan(factored).any()
    random_slots = _random_slots(t, s, KAIZO)
    target(float(random_slots), label="slots whose score depends on a roll")
    np.testing.assert_allclose(factored, _joint_moves(t, s, KAIZO), atol=1e-5)


def _random_slots(t, s, layout) -> int:
    """The move slots whose exact score distribution has more than one value."""
    _, hist = _slot_histograms(t, s, layout)
    return int(np.sum(np.sum(np.asarray(hist) > 0, axis=1) > 1))


@pytest.mark.slow
def test_factored_equals_joint_with_two_random_slots(kaizo_tables, kaizo_roark_party):
    """The floor under ``test_factored_equals_joint_bruteforce``, independent of what
    Hypothesis draws: Bonsly's Selfdestruct (EvalAttack :6374, always reached, kaizo_roark.md
    5.2) and Rock Throw, QUADRUPLE and the highest damage into a Fire/Flying player
    (:6383), so two slots' scores depend on their own rolls. The factored pick equals the
    joint enumeration."""
    t = kaizo_tables
    charizard = 6
    assert [int(x) for x in t.species_types[charizard]] == [10, 2]  # Fire, Flying
    s = _kaizo_state(t, kaizo_roark_party, [ROCK_THROW, 120, 0, 0])
    s = _bulky(_with_player(t, s, charizard, known=True), 400)
    assert _random_slots(t, s, KAIZO) == 2
    factored = np.asarray(_factored(t, s, KAIZO))
    np.testing.assert_allclose(factored, _joint_moves(t, s, KAIZO), atol=1e-5)


# --- Exact against loose -------------------------------------------------------------------


_slot_histograms = jax.jit(policy._slot_histograms, static_argnames="layout")


def _exact_slot_scores(t, s, layout) -> np.ndarray:
    """``float64[MOVES, 256]``: each slot's exact score distribution
    (``policy._slot_histograms``)."""
    _, hist = _slot_histograms(t, s, layout)
    return np.asarray(hist, np.float64)


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
    equal; so is each slot's score distribution before the pick (which catches a wrong read
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
    """On random Kaizo decisions the exact distribution and each slot's score distribution
    equal the loose ones, where every ``IfRandomLessThan`` and every ``LoadBattlerAbility
    DEFENDER`` is its own draw (E1 Ruling R1: no condition on how many ability reads can
    change a score). Both are NaN on the same refused states, the voluntary-switch
    refusals included."""
    t = kaizo_tables
    s = data.draw(kaizo_states(t, kaizo_roark_party))
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


def test_two_guesses_in_one_slot(kaizo_tables, kaizo_roark_party):
    """E1 Ruling R1, one slot: a species with Soundproof and Clear Body (made up) against
    Screech has two reads that change its score, Basic_CheckSoundproof's (script.s:117) and
    Basic_CheckClearBodyEffect's (:470), each its own coin (trainer_ai.c:1194-1198). Screech
    keeps 100 one time in four (Clear Body at :117, Soundproof at :470), as loose has it; one
    guess for the decision scored it 90 on either ability."""
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
    _assert_exact_equals_loose(t2, s)
    scores = _exact_slot_scores(t2, s, KAIZO)[0]
    np.testing.assert_allclose(scores[[90, 100]], [0.75, 0.25], atol=1e-6)


@pytest.mark.parametrize(
    ("trainer_id", "index", "species", "moves"),
    [
        (196, 2, "Geodude", ["Selfdestruct", "Rock Throw", "Rock Smash"]),  # Worker Mason
        (245, 0, "Shieldon", ["Toxic"]),  # Ruin Maniac Darius
    ],
)
def test_exact_equals_loose_on_roark_split_pokemon(
    kaizo_tables, kaizo_names, kaizo_trainers, kaizo_roark_party, trainer_id, index, species, moves
):
    """Exact == loose beyond Roark's team (E1 batch 1): two Pokemon of other Roark-split
    trainers, with the moves the AI transcribes (Geodude's Sleep Talk and Shieldon's Metal
    Burst and Sleep Talk are not yet), Roark's Cranidos behind them, against two guessed
    players: Gible (Sand Veil / Rough Skin) and Tentacool (Rain Dish / Clear Body).
    Selfdestruct reads :117 and :304, Rock Throw and Rock Smash :71 and :117, Toxic :117,
    :529 and :539, each its own coin."""
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from tests.conftest import KAIZO_INTERMEDIATE

    t, n = kaizo_tables, kaizo_names
    trainer = next(x for x in kaizo_trainers if x.id == trainer_id)
    mon = build_trainer_party(trainer, IntermediateData(KAIZO_INTERMEDIATE))[index]
    assert n.species_names[mon.species] == species
    kept = [m for m in mon.moves if n.move_names[m] in moves]
    assert [n.move_names[m] for m in kept] == moves
    mon = dataclasses.replace(mon, moves=tuple(kept + [0] * (4 - len(kept))), pp=None)
    bench = dataclasses.replace(kaizo_roark_party[5], moves=(ACCELEROCK, HEADBUTT, 0, 0), pp=None)
    base = make_state(t, [kaizo_roark_party[2]], [mon, bench])
    for player in (GIBLE, 72):
        s = _bulky(_with_player(t, base, player), 200)
        assert policy.ai_refusals(t, s, n) == [] and bool(commands.ability_guessed(t, s))
        _assert_exact_equals_loose(t, s)


@pytest.mark.parametrize(
    ("trainer_id", "index", "species", "moves"),
    [
        (2, 2, "Wurmple", ["Poison Sting", "Tackle"]),  # Bug Catcher Logan, flags 0x1
        (3, 0, "Sunkern", ["Toxic", "Mega Drain"]),  # Lass Natalie, flags 0x5
        (4, 1, "Torchic", ["Pound", "Ember", "Rock Smash"]),  # Youngster Michael, flags 0x7
        (246, 0, "Bonsly", ["Stealth Rock", "Selfdestruct", "Brick Break", "Accelerock"]),
        # E1 batch 2b: SETUP_FIRST_TURN and WEATHER, on the first turn.
        (195, 1, "Rhyhorn", ["Rock Slide", "Rock Polish"]),  # Colin, flags 0xf
        (196, 3, "Bidoof", ["Curse", "Accelerock"]),  # Mason, flags 0xf
        (10, 2, "Psyduck", ["Water Gun", "Confusion"]),  # Youngster Tyler, flags 0x20f
        (14, 4, "Bellsprout", ["Vine Whip", "Poison Jab", "Wrap"]),  # Taylor, flags 0x20f
        # E1 batch 2c: PRIORITIZE_EXTREMES and CHECK_HP.
        (249, 5, "Piplup", ["Water Pulse", "Rock Smash"]),  # Rival Barry, flags 0x27
        (356, 1, "Cascoon", ["Poison Sting"]),  # Bug Catcher Sebastian, flags 0x10f
    ],
)
def test_exact_equals_loose_per_flag_set(
    kaizo_tables, kaizo_names, kaizo_trainers, kaizo_roark_party, trainer_id, index, species, moves
):
    """E1 Ruling R2: each Roark-split trainer runs its own flag set (0x1 BASIC; 0x5 BASIC and
    EXPERT; 0x7 BASIC, EVAL_ATTACK and EXPERT; Roark's 0x17 with RISKY; 0xf with
    SETUP_FIRST_TURN; 0x20f with SETUP_FIRST_TURN and WEATHER; 0x27 with PRIORITIZE_EXTREMES;
    0x10f with SETUP_FIRST_TURN and CHECK_HP; trainer_ai.c:297-308; the
    states are the battle's first turn, where SETUP_FIRST_TURN and WEATHER score),
    and exact == loose under each, on a Pokemon of the trainer with the moves the AI
    transcribes (the others, e.g. Bug Bite, Leech Seed, Protect, Pluck, are not yet), Roark's
    Cranidos behind it, against two guessed players (Gible, Tentacool)."""
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from tests.conftest import KAIZO_INTERMEDIATE

    t, n = kaizo_tables, kaizo_names
    trainer = next(x for x in kaizo_trainers if x.id == trainer_id)
    mon = build_trainer_party(trainer, IntermediateData(KAIZO_INTERMEDIATE))[index]
    assert n.species_names[mon.species] == species
    kept = [m for m in mon.moves if n.move_names[m] in moves]
    assert [n.move_names[m] for m in kept] == moves
    mon = dataclasses.replace(mon, moves=tuple(kept + [0] * (4 - len(kept))), pp=None)
    bench = dataclasses.replace(kaizo_roark_party[5], moves=(ACCELEROCK, HEADBUTT, 0, 0), pp=None)
    base = make_state(t, [kaizo_roark_party[2]], [mon, bench])
    layout = layout_for(t, trainer.ai_flags, base)
    for player in (GIBLE, 72):
        s = _bulky(_with_player(t, base, player), 200)
        assert policy.ai_refusals(t, s, n, layout) == []
        _assert_exact_equals_loose(t, s, layout)


def test_slot_reads_fit_every_transcribed_effect(kaizo_tables, kaizo_roark_party):
    """E1 Ruling R6 against the data, in small pieces (E1 batch 8). The proof that no path makes
    more reads than its fight's bound is the walk of script.s (tests/ai/test_read_bounds.py
    regenerates its table); this checks it on the transcription. For the eight transcribed
    effects whose walked bound under every flag is largest, and Selfdestruct's, a fight of one
    such move with Roark's Accelerock and Headbutt, over random players (any species with two
    abilities, any types, guessed or known), random HP, Defense and evasion stages and the
    first turn or a later one: the loose oracle's longest path, under every transcribed flag,
    makes at most the fight's bound (``layout.fight_slot_reads``; a SLOWER compare's tie is no
    read of the exact layout, a FASTER one's is, E1 Ruling R8). Tailwind is always sampled, and
    half the states put both sides at equal speed so that its tie is on the path. Each piece is a fight-sized program, and JAX's caches are cleared
    between pieces. Selfdestruct reaches Roark's bound by hand: :117 and :304 (two guesses),
    :6374, :1859 (evasion +4 or more), :1870 and :1875 (HP at most 30 %) (script.s:6365-6374,
    :1841-1884), and Risky :6530 (:6521-6534)."""
    from nuzline.ai.layout import KAIZO_FLAG_BITS, fight_layout, fight_slot_reads, move_read_rows
    from nuzline.ai.read_bounds import READ_BOUND_BITS, READ_BOUNDS

    t = kaizo_tables
    every = sum(1 << b for b in KAIZO_FLAG_BITS)
    effect = np.asarray(t.move_ai_effect)
    first = {}
    for move in _move_pool(t, KAIZO_ALL):
        first.setdefault(int(effect[move]), move)
    assert set(first) == KAIZO_ALL.transcribed

    def walked(e):
        return max(sum(READ_BOUNDS[(e, d, q)]) for d in (False, True) for q in (False, True))

    sample = sorted(first, key=lambda e: (-walked(e), e))[:8]
    sample = [*sample, *([7] if 7 not in sample else [])]  # HALVE_DEFENSE: Selfdestruct
    sample = [*sample, *([225] if 225 not in sample else [])]  # Tailwind: a FASTER tie
    assert len(READ_BOUND_BITS) == 8
    rng = np.random.default_rng(11)
    species = _two_ability_species(t)
    for e in sample:
        moves = [first[e], ACCELEROCK, HEADBUTT, 0]
        states = []
        for _ in range(4):
            s = _kaizo_state(t, kaizo_roark_party, moves)
            s = _with_player(t, s, int(rng.choice(species)), int(rng.integers(2)))
            s = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(int(rng.integers(4) == 0)))
            if rng.integers(2):
                s = s._replace(
                    types=s.types.at[0, 0].set(jnp.asarray(rng.integers(0, 17, 2), s.types.dtype))
                )
            hp = int(rng.integers(20, 400))
            s = s._replace(
                hp=s.hp.at[0, 0].set(int(rng.integers(1, hp + 1))), max_hp=s.max_hp.at[0, 0].set(hp)
            )
            s = s._replace(hp=s.hp.at[1, 0].set(int(rng.integers(1, int(s.max_hp[1, 0]) + 1))))
            stages = s.stages.at[0, Stat.DEF].set(int(rng.integers(-6, 7)))
            stages = stages.at[0, Stat.EVA].set(int(rng.integers(0, 7)))
            s = s._replace(stages=stages.at[1, Stat.DEF].set(int(rng.integers(-6, 7))))
            if rng.integers(2):  # equal speed: the tie is drawn
                s = _equal_speed(s)
            states.append(s._replace(turn=jnp.asarray(int(rng.integers(2)), s.turn.dtype)))
        batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        party = np.asarray(states[0].moves[1]).ravel()
        layout = fight_layout(t, every, party)
        reads, _, _ = policy._loose_reads(t, batch, layout)
        _, lo = policy._loose_batch(t, batch, jnp.asarray(policy._loose_patterns(reads)), layout)
        counted = np.asarray(lo.live) & (np.asarray(lo.kind) != ReadKind.SPEED_TIE)
        longest = int(counted[:, :, 0].sum(axis=2).max())
        bound = fight_slot_reads(move_read_rows(t, party), every)
        assert longest <= bound, (e, longest, bound)
        jax.clear_caches()
    # Roark's bound by hand: Selfdestruct, a guessed two-ability player, evasion +4 (decomp
    # stage 10, script.s:1857), Bonsly at 10 % HP.
    assert fight_slot_reads(move_read_rows(t, t.ai_moves), KAIZO.flags) == KAIZO.slot_reads == 7
    s = _kaizo_state(t, kaizo_roark_party, (SELFDESTRUCT, 0, 0, 0))
    s = _bulky(_with_player(t, s, GIBLE), 400)
    s = s._replace(
        stages=s.stages.at[0, Stat.EVA].set(4), hp=s.hp.at[1, 0].set(int(s.max_hp[1, 0]) // 10)
    )
    batch = _one(s)
    reads, _, _ = policy._loose_reads(t, batch, KAIZO)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(policy._loose_patterns(reads)), KAIZO)
    live = np.asarray(lo.live)[0, :, 0]
    longest = live.sum(axis=1).argmax()
    lines = np.asarray(lo.line)[0, longest, 0][live[longest]].tolist()
    assert lines == [0, 0, 6374, 1859, 1870, 1875, 6530] and len(lines) == KAIZO.slot_reads
    _assert_exact_equals_loose(t, s)


def test_a_read_past_the_bound_is_refused(kaizo_tables, kaizo_names, kaizo_roark_party):
    """A layout bound: a path that makes more random reads than ``slot_reads`` is refused,
    never given a reused draw (``AiContext.missing_read``). Toxic against a guessed Gible
    (Sand Veil / Rough Skin) reads :117, :529 and :539; under a layout with two reads the
    exact distribution is NaN, ``ai_scores`` raises naming the slot, and with the ability
    known (no read guesses) it is accepted."""
    t, n = kaizo_tables, kaizo_names
    s = _kaizo_state(t, kaizo_roark_party, (TOXIC, TACKLE, 0, 0))
    s = _bulky(_with_player(t, s, GIBLE))
    narrow = KAIZO._replace(slot_reads=2)
    _assert_exact_equals_loose(t, s, KAIZO)
    assert np.isnan(np.asarray(_distribution(t, s, narrow))).any()
    line = (
        "ai scoring: opponent slot 0: move Toxic makes more than 2 random reads on one path, "
        "the layout's slot_reads (docs/ai/kaizo_roark.md 8.3)"
    )
    assert policy.ai_path_refusals(t, s, n, narrow) == [line]
    o = sample_ai_outcome(jax.random.PRNGKey(0), t, s, narrow)
    with pytest.raises(policy.UnsupportedAi, match="more than 2 random reads"):
        policy.ai_scores(t, s, o, narrow)
    known = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))
    assert not np.isnan(np.asarray(_distribution(t, known, narrow))).any()


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
import hashlib, jax, numpy as np
from nuzline.data.games import GAMES, KAIZO
from nuzline.data.tables import load_tables
from nuzline.ai.layout import fight_tables
from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
from nuzline.engine.actions import NUM_OPP_ACTIONS
from nuzline.search.expand import ai_rows
from nuzline.search.rows import take
from tests.ai.test_kaizo_outcome import _random_kaizo_rows

assert jax.device_count() == DEVICES, jax.device_count()
d = GAMES[KAIZO]
t = load_tables(d.tables_dir)
roark = next(x for x in load_trainers(d.intermediate_dir) if x.name == "Roark")
party = build_trainer_party(roark, IntermediateData(d.intermediate_dir))
t = fight_tables(t, roark, party)
rows = _random_kaizo_rows(t, party, 40, seed=5)
empty = ai_rows(t, take(rows, np.arange(0)))  # K3c final review: no rows, any device count
assert empty.shape == (0, NUM_OPP_ACTIONS) and empty.dtype == np.float32, empty
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
    differs too). An empty batch gives a (0, NUM_OPP_ACTIONS) array on both (K3c final
    review: ``shard_map`` needs the leading axis to divide by the device count)."""
    assert _kaizo_ai_rows_on_devices(1) == _kaizo_ai_rows_on_devices(3)
