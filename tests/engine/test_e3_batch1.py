"""E3 batch 1 (rulings E3-1-1 .. E3-1-5, docs/kaizo/e3-ledger.md): Mist Ball (#71, verified
modelled), Splash / Kaizo Thief (#85), Teleport (#153) and Hammer Arm (#218), each case checked
against the reference engine. Disable (#86) has no new engine work (tests/engine/test_disable.py);
its AI blocks are tested in tests/ai/test_kaizo_policy.py.

Decomp (external/pokeplatinum): ``res/battle/scripts/effects/effect_script_{0071,0085,0153,0218}.s``,
``src/battle/battle_lib.c`` (``BattleSystem_TriggerSecondaryEffect`` :1524-1529 ON_HIT, :1570-1586
indirect proc; ``BattleContext_Init`` :1974-1976 damage 0), ``src/battle/battle_script.c``
(``BtlCmd_ChangeStatStage`` :2804-2976), ``src/battle/battle_controller_player.c`` (UpdateHP :3347
skips damage 0; :3212, :3945-3949 a used move)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import nuzline.ai.commands as c
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.actions import usable_moves
from nuzline.engine.damage import ai_damage_for_roll, damage_table, decomp_abilities
from nuzline.engine.enumerate import hit_class_law, outcome_laws
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS, make_state
from nuzline.engine.step import step_events
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _usable as ref_usable
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tools.effect_map import CERTAIN_SECONDARY_EFFECTS, DECOMP_EFFECT_TO_ID

SPE = 4  # stats[..., 0..4] are Atk..Spe


def _spec(n, species, level, moves):
    return PokemonSpec(
        species=n.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(n.move(m) for m in moves) + (0,) * (4 - len(moves)),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


def _fight(t, n, mine, opp=("Tackle",)):
    """A level 20 Monferno (``mine`` first) against a level 40 Geodude (``opp`` first), no
    abilities, one Pokemon each."""
    s = make_state(t, [_spec(n, "Monferno", 20, mine)], [_spec(n, "Geodude", 40, opp)])
    return s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))


def _slower(s):
    return s._replace(stats=s.stats.at[0, 0, SPE].set(1))


def _procs():
    return empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8))


def _stage(s, side, stat):
    return int(s.stages[side, stat])


# --- #71 Mist Ball: modelled; the proc, Shield Dust and Mist --------------------------------


def test_lower_sp_atk_hit_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """effect_script_0071.s:5-7 (CalcCrit, CalcDamage, an indirect SP_ATTACK_DOWN_1_STAGE with no
    ON_HIT bit): SECONDARY_STAT_FOE (Sp. Atk, -1), rolled at the move's chance (50 and, for
    Kaizo's Mystical Fire, 100); battle_lib.c:1570-1586."""
    assert DECOMP_EFFECT_TO_ID[71] == (EffectId.SECONDARY_STAT_FOE, (Stat.SPA, -1, 0))
    assert 71 not in CERTAIN_SECONDARY_EFFECTS
    for t, n, moves in (
        (tables, names, {"Mist Ball": 50}),
        (kaizo_tables, kaizo_names, {"Mist Ball": 50, "Mystical Fire": 100}),
    ):
        for move, chance in moves.items():
            m = n.move(move)
            assert int(t.move_effect[m]) == EffectId.SECONDARY_STAT_FOE, move
            assert tuple(int(x) for x in t.move_param[m]) == (Stat.SPA, -1, 0), move
            assert int(t.move_effect_chance[m]) == chance, move


def test_mist_ball_drops_special_attack_unless_blocked(kaizo_tables, kaizo_names):
    """A proc (``secondary`` 1): foe Sp. Atk -1 (bs.c:2969-2973). Shield Dust on the foe blocks
    the indirect drop (:2918-2920) and so does Mist on its side (``mistTurns``, :2876-2882);
    with no proc nothing changes. Mystical Fire (chance 100) is the same effect."""
    t, n = kaizo_tables, kaizo_names
    for move in ("Mist Ball", "Mystical Fire"):
        s = _fight(t, n, (move,))
        acts = A(move, "Tackle", s, n)
        out = assert_matches_reference(t, s, acts, _procs())
        assert _stage(out, 1, Stat.SPA) == -1, move
        dust = s._replace(ability=s.ability.at[1, 0].set(AbilityId.SHIELD_DUST))
        out = assert_matches_reference(t, dust, acts, _procs())
        assert _stage(out, 1, Stat.SPA) == 0, move
        misty = s._replace(mist=s.mist.at[1].set(3))
        out = assert_matches_reference(t, misty, acts, _procs())
        assert _stage(out, 1, Stat.SPA) == 0, move
    s = _fight(t, n, ("Mist Ball",))
    out = assert_matches_reference(t, s, A("Mist Ball", "Tackle", s, n), empty_outcome())
    assert _stage(out, 1, Stat.SPA) == 0


# --- #85 Splash / Kaizo Thief ---------------------------------------------------------------


def test_do_nothing_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 85 (effect_script_0085.s:5): vanilla Splash (power 0), Kaizo Thief (power 40)."""
    assert DECOMP_EFFECT_TO_ID[85] == (EffectId.DO_NOTHING, (0, 0, 0))
    splash, thief = names.move("Splash"), kaizo_names.move("Thief")
    assert int(tables.move_effect[splash]) == EffectId.DO_NOTHING
    assert int(kaizo_tables.move_effect[thief]) == EffectId.DO_NOTHING
    assert int(kaizo_tables.move_power[thief]) == 40


def test_thief_deals_no_damage_but_the_ai_rates_it_a_forty_power_hit(kaizo_tables, kaizo_names):
    """The script has no CalcDamage, so the engine deals 0 for every (crit, roll)
    (damage.fixed_damage); the AI's own calculation (trainer_ai.c:31-45 does not list effect 85,
    power 40 > 1) keeps the move's power: ``is_damaging`` and a positive estimate (E3-1-2)."""
    t, n = kaizo_tables, kaizo_names
    s = _fight(t, n, ("Thief",))
    thief = n.move("Thief")
    for crit in (False, True):
        got = damage_table(t, s, 0, thief, crit).tolist()
        assert got == [0] * 16 == ref_damage_table(to_numpy(t), to_ref(s), 0, thief, crit)
    assert bool(c.is_damaging(t, thief))
    assert int(ai_damage_for_roll(t, s, 1, thief, 100)) > 0


def test_thief_changes_no_hp_and_counts_as_a_used_move(kaizo_tables, kaizo_names):
    """A forced crit and the top roll change nothing: no HP loss for the foe, a Substitute left
    whole (UpdateHP skips damage 0, bcp.c:3347); the move is used (PP spent, ``last_move``,
    bcp.c:3212, :3945-3949) so Disable and Encore can see it."""
    t, n = kaizo_tables, kaizo_names
    s = _fight(t, n, ("Thief", "Scratch"))
    thief = n.move("Thief")
    acts = A("Thief", "Tackle", s, n)
    o = empty_outcome()
    o = o._replace(crit=o.crit.at[0].set(1))  # the Thief user's hits only
    shielded = s._replace(substitute_hp=s.substitute_hp.at[1].set(20))
    for state, sub in ((s, 0), (shielded, 20)):
        out = assert_matches_reference(t, state, acts, o)
        assert int(out.hp[1, 0]) == int(state.hp[1, 0])
        assert int(out.substitute_hp[1]) == sub
        assert int(out.last_move[0]) == thief
        assert int(out.pp[0, 0, 0]) == int(state.pp[0, 0, 0]) - 1
        assert int(out.hp[0, 0]) < int(state.hp[0, 0])  # the foe's Tackle landed
    _, ev = step_events(t, s, acts, o)
    assert not np.asarray(ev["crit"][0]).any()  # no CalcCrit: no crit message


# --- #153 Teleport --------------------------------------------------------------------------


def test_teleport_is_mapped_with_the_games_priorities(tables, names, kaizo_tables, kaizo_names):
    """Effect 153 (effect_script_0153.s:5, :9-10); priority 0 vanilla, +1 Kaizo."""
    assert DECOMP_EFFECT_TO_ID[153] == (EffectId.FLEE, (0, 0, 0))
    for t, n, priority in ((tables, names, 0), (kaizo_tables, kaizo_names, 1)):
        m = n.move("Teleport")
        assert int(t.move_effect[m]) == EffectId.FLEE
        assert int(t.move_priority[m]) == priority


def test_teleport_fails_in_a_trainer_battle_and_is_a_used_move(
    tables, names, kaizo_tables, kaizo_names
):
    """It fails (nothing else happens) but still spends its PP and is the user's last move; the
    Kaizo +1 priority puts a slower user first, the vanilla 0 does not."""
    for t, n, first in ((tables, names, 1), (kaizo_tables, kaizo_names, 0)):
        s = _slower(_fight(t, n, ("Teleport", "Scratch")))
        acts = A("Teleport", "Tackle", s, n)
        out = assert_matches_reference(t, s, acts, empty_outcome())
        assert int(out.hp[1, 0]) == int(s.hp[1, 0])
        assert int(out.last_move[0]) == n.move("Teleport")
        assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
        assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # the foe's Tackle landed
        _, ev = step_events(t, s, acts, empty_outcome())
        assert int(np.asarray(ev["order"])[0]) == first


# --- #218 Hammer Arm ------------------------------------------------------------------------


def test_hammer_arm_is_mapped_as_a_certain_self_speed_drop(kaizo_tables, kaizo_names):
    """effect_script_0218.s:5-7: ON_HIT | TO_ATTACKER SPEED_DOWN_1_STAGE (no chance roll)."""
    assert DECOMP_EFFECT_TO_ID[218] == (EffectId.SECONDARY_STAT_SELF, (Stat.SPE, -1, 0))
    assert 218 in CERTAIN_SECONDARY_EFFECTS
    m = kaizo_names.move("Hammer Arm")
    assert int(kaizo_tables.move_effect[m]) == EffectId.SECONDARY_STAT_SELF
    assert int(kaizo_tables.move_effect_chance[m]) == 0


def test_hammer_arm_lowers_the_users_speed_through_a_substitute_and_a_faint(
    kaizo_tables, kaizo_names
):
    """lib.c:1524-1529 has no HP, Substitute or chance test, and bs.c:2876 skips every guard for
    a drop on the user: -1 after a hit on a Substitute and after the foe faints, floored at -6
    (:2924-2937, :2969-2973)."""
    t, n = kaizo_tables, kaizo_names
    s = _fight(t, n, ("Hammer Arm",))
    acts = A("Hammer Arm", "Tackle", s, n)
    assert _stage(assert_matches_reference(t, s, acts, empty_outcome()), 0, Stat.SPE) == -1
    shielded = s._replace(substitute_hp=s.substitute_hp.at[1].set(200))
    out = assert_matches_reference(t, shielded, acts, empty_outcome())
    assert _stage(out, 0, Stat.SPE) == -1 and int(out.substitute_hp[1]) < 200
    dying = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(t, dying, acts, empty_outcome())
    assert int(out.hp[1, 0]) == 0 and _stage(out, 0, Stat.SPE) == -1
    floor = s._replace(stages=s.stages.at[0, Stat.SPE].set(-6))
    assert _stage(assert_matches_reference(t, floor, acts, empty_outcome()), 0, Stat.SPE) == -6


# --- Round 2 (E3-2-1 .. E3-2-4) ---------------------------------------------------------------
#
# #165 Torment: ``effect_script_0165.s:5``, ``subscript_torment_start.s:5-8``; the selection-time
# block is ``BattleSystem_CheckInvalidMoves`` CHECK_INVALID_TORMENTED (battle_lib.c:2275-2279).


def _legal(t, s, side=1):
    return np.asarray(usable_moves(t, s, side)).tolist()


def test_torment_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 165 (effect_script_0165.s:5); Torment is in both games' move tables."""
    assert DECOMP_EFFECT_TO_ID[165] == (EffectId.TORMENT, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Torment")]) == EffectId.TORMENT


def test_torment_sets_the_bit_through_a_substitute_and_not_on_a_miss(tables, names):
    """subscript_torment_start.s:5-8: no Substitute test, no ability test; a miss or a target that
    has the bit already fails (the bit stays 1). No count is kept (E3-2-1)."""
    s = _fight(tables, names, ("Torment",), ("Tackle", "Rock Throw"))
    acts = A("Torment", "Tackle", s, names)
    hit = empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, s, acts, hit)
    assert int(out.torment[1]) == 1 and int(out.torment[0]) == 0
    shielded = s._replace(substitute_hp=s.substitute_hp.at[1].set(30))
    out = assert_matches_reference(tables, shielded, acts, hit)
    assert int(out.torment[1]) == 1
    already = s._replace(torment=s.torment.at[1].set(1))
    assert int(assert_matches_reference(tables, already, acts, hit).torment[1]) == 1
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    assert int(assert_matches_reference(tables, s, acts, miss).torment[1]) == 0


def test_torment_blocks_only_the_move_used_last(tables, names):
    """CHECK_INVALID_TORMENTED (battle_lib.c:2275-2279): the slot holding ``movePrevByBattler``;
    ``last_move`` 0 (a move that did not succeed) blocks nothing; a lone known move that is blocked
    leaves no legal slot (Struggle, bcp.c:433). The reference mask agrees (E3-2-1)."""
    s = _fight(tables, names, ("Tackle",), ("Tackle", "Rock Throw"))
    tormented = s._replace(
        torment=s.torment.at[1].set(1), last_move=s.last_move.at[1].set(names.move("Tackle"))
    )
    got = _legal(tables, tormented)
    assert got == [False, True, False, False]
    ref = ref_usable(to_numpy(tables), to_ref(tormented), 1)
    assert [bool(x) for x in ref] == got
    fresh = tormented._replace(last_move=tormented.last_move.at[1].set(0))
    assert _legal(tables, fresh) == [True, True, False, False]
    clean = tormented._replace(torment=tormented.torment.at[1].set(0))
    assert _legal(tables, clean) == [True, True, False, False]
    lone = _fight(tables, names, ("Tackle",), ("Tackle",))
    lone = lone._replace(
        torment=lone.torment.at[1].set(1), last_move=lone.last_move.at[1].set(names.move("Tackle"))
    )
    assert not any(_legal(tables, lone))


def test_torment_is_read_at_selection_only_and_leaves_with_the_battler(tables, names):
    """No execution check (bcp.c:2543-2560 has Disable and Taunt states only): a move chosen
    before the Torment still runs. A switch clears the bit with the volatiles (lib.c:2071; Baton
    Pass keeps ``VOLATILE_CONDITION_BATON_PASSED`` only, :2075, condition.h:72-76)."""
    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    tormented = s._replace(
        torment=s.torment.at[1].set(1), last_move=s.last_move.at[1].set(names.move("Tackle"))
    )
    acts = A("Tackle", "Tackle", tormented, names)  # the tormented foe's raw action 0: Tackle
    out = assert_matches_reference(tables, tormented, acts, empty_outcome())
    assert int(out.hp[0, 0]) < int(tormented.hp[0, 0])
    assert "torment" in VOLATILE_FIELDS and "torment" not in BATON_PASSED_FIELDS
    two = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Tackle",))],
        [_spec(names, "Geodude", 40, ("Tackle",)), _spec(names, "Geodude", 40, ("Scratch",))],
    )
    two = two._replace(torment=two.torment.at[1].set(1))
    switch = jnp.array([names.move_slot(two, 0, "Tackle"), 4 + 1], jnp.int32)
    out = assert_matches_reference(tables, two, switch, empty_outcome())
    assert int(out.torment[1]) == 0


# --- #247 Worry Seed (E3-2-2) -----------------------------------------------------------------
#
# ``effect_script_0247.s:5`` runs ``subscript_give_target_insomnia.s``: fails (:5-9) on a miss,
# a semi-invulnerable target, a Substitute, a raw Truant or Multitype ability, a held Griseous
# Orb; else :11 writes ``BattleMon.ability`` = Insomnia, and the MoveEnd recovery
# (battle_controller_player.c:3965-3967, battle_lib.c:4404-4410) wakes a sleeper.


def hit_law(tables, state, actions, side) -> dict[int, int]:
    """``value -> weight`` (out of 100) of ``hit[side, 0]`` in the enumeration."""
    laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, state, actions))
    values, weights, den = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
    assert den == 100
    return dict(zip(values, weights, strict=True))


def _hits():
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))


def _seeded(t, n, **over):
    """Monferno (side 0) a sleeper with Tackle; the Geodude seeds it. ``over``: state fields."""
    s = _fight(t, n, ("Tackle",), ("Worry Seed",))
    s = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
        ability=s.ability.at[0, 0].set(AbilityId.BLAZE),
    )
    return s._replace(**over) if over else s


def test_worry_seed_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 247 (effect_script_0247.s:5); Worry Seed is in both games' move tables."""
    assert DECOMP_EFFECT_TO_ID[247] == (EffectId.WORRY_SEED, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Worry Seed")]) == EffectId.WORRY_SEED


def test_worry_seed_writes_insomnia_wakes_the_sleeper_and_the_ai_learns_it(
    tables, names, kaizo_tables, kaizo_names
):
    """:11 sets the battler's ability to Insomnia; ``damage.decomp_abilities`` (Battler_Ability)
    reads it; the sleeper is cured at the move's end; ``BattleAI_SetAbility`` records it
    (battle_script.c:3073-3075). The foe's own ability is untouched."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = _seeded(t, n)
        out = assert_matches_reference(t, s, A("Tackle", "Worry Seed", s, n), _hits())
        assert int(out.ability[0, 0]) == AbilityId.INSOMNIA
        assert int(out.ability[1, 0]) == int(s.ability[1, 0])
        assert int(decomp_abilities(t, out)[0, 0]) == 15  # ABILITY_INSOMNIA
        assert int(out.status[0, 0]) == Status.NONE
        assert int(out.ai_ability_known[0]) == 15
        # a miss writes nothing
        miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
        out = assert_matches_reference(t, s, A("Tackle", "Worry Seed", s, n), miss)
        assert int(out.ability[0, 0]) == AbilityId.BLAZE and int(out.ai_ability_known[0]) == 0
        # an Insomnia target succeeds again
        again = s._replace(ability=s.ability.at[0, 0].set(AbilityId.INSOMNIA))
        out = assert_matches_reference(t, again, A("Tackle", "Worry Seed", s, n), _hits())
        assert int(out.ability[0, 0]) == AbilityId.INSOMNIA


def test_worry_seed_fails_on_substitute_truant_multitype_and_griseous_orb(tables, names):
    """subscript_give_target_insomnia.s:6-9: CheckSubstitute, a raw Truant (:7) or Multitype (:8)
    ability, a held Griseous Orb (:9): the ability stays and the sleeper sleeps on."""
    s = _seeded(tables, names)
    acts = A("Tackle", "Worry Seed", s, names)
    shielded = s._replace(substitute_hp=s.substitute_hp.at[0].set(30))
    truant = s._replace(ability=s.ability.at[0, 0].set(AbilityId.TRUANT))
    orb = s._replace(item=s.item.at[0, 0].set(names.item("Griseous Orb")))
    for case, kept in (
        (shielded, AbilityId.BLAZE),
        (truant, AbilityId.TRUANT),
        (orb, AbilityId.BLAZE),
    ):
        out = assert_matches_reference(tables, case, acts, _hits())
        assert int(out.ability[0, 0]) == kept
    arceus = make_state(
        tables,
        [_spec(names, "Arceus", 20, ("Tackle",))],
        [_spec(names, "Geodude", 40, ("Worry Seed",))],
    )
    out = assert_matches_reference(
        tables, arceus, A("Tackle", "Worry Seed", arceus, names), _hits()
    )
    assert int(decomp_abilities(tables, out)[0, 0]) == 121  # ABILITY_MULTITYPE stays


def test_worry_seed_ends_with_the_battler(tables, names):
    """The write is the BattleMon's (lib.c:124 reloads the party Pokemon's ability on the next
    switch-in): ``_restore_ability`` puts the species' ability back when the battler leaves."""
    two = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Tackle",)), _spec(names, "Monferno", 20, ("Scratch",))],
        [_spec(names, "Geodude", 40, ("Worry Seed",))],
    )
    out = assert_matches_reference(tables, two, A("Tackle", "Worry Seed", two, names), _hits())
    assert int(out.ability[0, 0]) == AbilityId.INSOMNIA
    switch = jnp.array([4 + 1, names.move_slot(out, 1, "Worry Seed")], jnp.int32)
    out2 = assert_matches_reference(tables, out, switch, empty_outcome())
    assert int(out2.ability[0, 0]) == int(two.ability[0, 0])


def test_worry_seed_is_seen_by_hit_thresholds_and_crit_odds(tables, names):
    """A faster foe's Worry Seed takes the side's Compoundeyes before its check (bcp.c:2876-2947):
    Rock Throw's 90 is 117 at the turn start and 90 once seeded (hit_thresholds holds both);
    it also takes Super Luck (battle_lib.c:7126), so ``CRIT_ODDS_EFFECTS`` carries it."""
    s = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Worry Seed"], opp_faster=True
    )
    s = s._replace(ability=s.ability.at[0, int(s.active[0])].set(AbilityId.COMPOUNDEYES))
    acts = A("Rock Throw", "Worry Seed", s, names)
    assert hit_law(tables, s, acts, 0) == {1: 90, 91: 10}
    from nuzline.engine.outcome import CRIT_ODDS_EFFECTS

    assert EffectId.WORRY_SEED in CRIT_ODDS_EFFECTS
