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
from nuzline.engine.field import with_trick_room, with_weather
from nuzline.engine.ids import TRICK_ROOM_TURNS, AbilityId, EffectId, Stat, Status
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS, make_state
from nuzline.engine.step import step_events
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _usable as ref_usable
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    roark_state,
    trick_room_of,
    weather_of,
    weather_timer_of,
)
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


# --- #241 Me First (E3-2-3) -------------------------------------------------------------------
#
# ``effect_script_0241.s:5`` ``BtlCmd_TryMeFirst`` (battle_script.c:7286-7312), then GoToMoveScript;
# ``Move_MeFirstCanCopy`` (battle_lib.c:7242-7252) compares battle EFFECTS; the x1.5 is
# ``BattleScript_CalcMoveDamage`` (battle_script.c:1354-1364).


def _me_first_fight(t, n, foe_move, mine=("Me First", "Rock Throw"), fast=True):
    """A level 50 Monferno (``mine``) against a level 50 Geodude using ``foe_move`` on slot 0."""
    s = make_state(
        t, [_spec(n, "Monferno", 50, mine)], [_spec(n, "Geodude", 50, (foe_move, "Growl"))]
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    return s._replace(stats=s.stats.at[0, 0, SPE].set(500 if fast else 1))


def test_me_first_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 241 (effect_script_0241.s:5); Me First is in both games' move tables."""
    assert DECOMP_EFFECT_TO_ID[241] == (EffectId.ME_FIRST, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Me First")]) == EffectId.ME_FIRST


def test_me_first_runs_the_targets_move_at_one_and_a_half(tables, names):
    """The copied Rock Throw deals what the user's own Rock Throw deals with the Me First counter
    set (x1.5 after the crit and Life Orb, before the variance), more than without it; the counter
    ends up at the Me First turn number minus 1 (:1356-1358), and the user is not moved by the
    foe's move first (it was faster)."""
    s = _me_first_fight(tables, names, "Rock Throw")
    out = assert_matches_reference(tables, s, A("Me First", "Rock Throw", s, names), _hits())
    boosted = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert boosted > 0 and int(out.me_first[0]) == 1 and int(out.me_first_turn[0]) == 0
    seeded = s._replace(
        me_first=s.me_first.at[0].set(1), me_first_turn=s.me_first_turn.at[0].set(1)
    )
    own = assert_matches_reference(tables, seeded, A("Rock Throw", "Growl", s, names), _hits())
    assert int(s.hp[1, 0]) - int(own.hp[1, 0]) == boosted
    plain = assert_matches_reference(tables, s, A("Rock Throw", "Growl", s, names), _hits())
    assert int(s.hp[1, 0]) - int(plain.hp[1, 0]) < boosted


def test_me_first_fails_after_the_target_moved_on_status_and_on_banned_effects(
    tables, names, kaizo_tables, kaizo_names
):
    """TryMeFirst fails (:7305-7309) when the target has acted (DEFENDER_ACTION is MOVE_END), chose
    no damaging move, or the move shares a battle effect with Counter, Mirror Coat, Thief, Covet,
    Focus Punch or Chatter: vanilla Covet (effect 105) is refused, and Kaizo, whose Covet slot has
    effect 0, refuses every plain hit (Rock Throw)."""
    cases = (
        (tables, names, "Rock Throw", False, "Me First moves after the target"),
        (tables, names, "Growl", True, "a status move has no power"),
        (tables, names, "Covet", True, "effect 105 is Thief's"),
        (kaizo_tables, kaizo_names, "Rock Throw", True, "Kaizo's effect 0 is banned"),
    )
    for t, n, foe_move, fast, why in cases:
        s = _me_first_fight(t, n, foe_move, fast=fast)
        out = assert_matches_reference(t, s, A("Me First", foe_move, s, n), _hits())
        assert int(out.hp[1, 0]) == int(s.hp[1, 0]), why
        assert int(out.me_first[0]) == 0, why


class _CounterState:
    """The three fields ``damage.me_first_*`` read and write, one battler."""

    def __init__(self, turn, flag, number):
        self.turn = jnp.asarray(turn)
        self.me_first = jnp.asarray([flag], jnp.int8)
        self.me_first_turn = jnp.asarray([number], jnp.int16)

    def _replace(self, **kw):
        new = _CounterState(0, 0, 0)
        new.turn, new.me_first, new.me_first_turn = self.turn, self.me_first, self.me_first_turn
        for k, v in kw.items():
            setattr(new, k, v)
        return new


def test_me_first_counter_arithmetic():
    """``damage.me_first_active`` / ``me_first_after_calc`` (battle_script.c:1354-1364) with
    meFirstTurnOrder = turn + 1: boosted on the Me First turn and (after the first CalcDamage
    lowered the number) cleared at the next turn's first one; a copy that never calculated keeps
    the number, so the next turn's first calculation is boosted too and clears one turn later."""
    from nuzline.engine.damage import me_first_active, me_first_after_calc

    calc = me_first_after_calc
    st = _CounterState(0, 1, 1)
    assert bool(me_first_active(st, 0))
    st = calc(st, 0, True)  # number 1 -> 0, order 1: still boosted
    assert (int(st.me_first[0]), int(st.me_first_turn[0])) == (1, 0) and bool(
        me_first_active(st, 0)
    )
    nxt = st._replace(turn=jnp.asarray(1))  # order 2, number 0: 2 above
    assert not bool(me_first_active(nxt, 0))
    assert int(calc(nxt, 0, True).me_first[0]) == 0
    idle = _CounterState(1, 1, 1)  # the Me First turn made no calculation; order 2, number 1
    assert bool(me_first_active(idle, 0))
    assert int(calc(idle, 0, True).me_first[0]) == 1  # 2 - 1 < 2: stays
    later = _CounterState(2, 1, 1)
    assert not bool(me_first_active(later, 0)) and int(calc(later, 0, True).me_first[0]) == 0
    # A cleared flag keeps no turn number (equal states stay equal).
    for st in (nxt, later):
        cleared = calc(st, 0, True)
        assert (int(cleared.me_first[0]), int(cleared.me_first_turn[0])) == (0, 0)


def _not_boosted_next_turn(t, n, after, foe_move):
    """The user's Rock Throw on the turn after ``after`` deals what it deals with no Me First
    counter at all, and leaves both fields 0."""
    bare = after._replace(
        me_first=after.me_first.at[0].set(0), me_first_turn=after.me_first_turn.at[0].set(0)
    )
    acts = A("Rock Throw", foe_move, after, n)
    out = assert_matches_reference(t, after, acts, _hits())
    plain = assert_matches_reference(t, bare, acts, _hits())
    assert int(out.hp[1, 0]) == int(plain.hp[1, 0])
    assert (int(out.me_first[0]), int(out.me_first_turn[0])) == (0, 0)


def test_me_first_counter_moves_when_the_copy_misses(tables, names):
    """CalcDamage runs in the copied move's script (effect_script_0000.s:5-6), which BeforeMove
    loads before TRY_MOVE (battle_controller_player.c:3214-3216), so before the accuracy roll and
    CheckMoveHitOverrides (:3254-3270): a copied Rock Throw that misses still lowers the number
    (battle_script.c:1356-1358), and the next turn's Rock Throw is not boosted (:1360-1363)."""
    s = _me_first_fight(tables, names, "Rock Throw")
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, s, A("Me First", "Rock Throw", s, names), miss)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])  # the copy missed
    assert (int(out.me_first[0]), int(out.me_first_turn[0])) == (1, 0)
    _not_boosted_next_turn(tables, names, out, "Growl")


def test_me_first_counter_moves_against_protect_and_a_semi_invulnerable_foe(tables, names):
    """With the counter set this turn (number 1 at turn 0, as a Me First leaves it), a Rock Throw
    into a Protect or a foe underground (Dig) runs its CalcDamage before CheckMoveHitOverrides
    stops it (bcp.c:3001-3014, :3034-3041): the number drops, and the next turn's hit is plain."""
    from nuzline.engine.ids import SemiKind

    for foe_move in ("Protect", "Dig"):
        s = _me_first_fight(tables, names, foe_move, fast=False)
        s = s._replace(me_first=s.me_first.at[0].set(1), me_first_turn=s.me_first_turn.at[0].set(1))
        out = assert_matches_reference(tables, s, A("Rock Throw", foe_move, s, names), _hits())
        assert int(out.hp[1, 0]) == int(s.hp[1, 0]), foe_move  # blocked
        if foe_move == "Dig":
            assert int(out.semi_invulnerable[1]) == SemiKind.UNDERGROUND
        assert (int(out.me_first[0]), int(out.me_first_turn[0])) == (1, 0), foe_move
        _not_boosted_next_turn(tables, names, out, foe_move)


# --- #259 Trick Room (E3-2-4) -----------------------------------------------------------------
#
# ``effect_script_0259.s:5-6`` toggles FIELD_CONDITION_TRICK_ROOM_INIT (5 turns, condition.h:145);
# the side-condition pass counts it down (battle_controller_player.c:1822-1830) after Perish Song;
# the Kaizo end script (sub_seq 251) sets it again, so it never ends there; map weather 18
# (subscript_overworld_trick_room.s:14) starts a fight with it up.


def _trick(t, n, turns=0, mine=("Trick Room",), opp=("Growl",)):
    s = _fight(t, n, mine, opp)
    return with_trick_room(s, turns)


def test_trick_room_is_mapped_and_rearmed_only_in_kaizo(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[259] == (EffectId.TRICK_ROOM, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Trick Room")]) == EffectId.TRICK_ROOM
    assert (int(tables.trick_room_rearm), int(kaizo_tables.trick_room_rearm)) == (0, 1)


def test_trick_room_toggles_and_the_turn_pass_counts_it_once(tables, names):
    s = _trick(tables, names)
    out = assert_matches_reference(tables, s, A("Trick Room", "Growl", s, names), _hits())
    # Set to 5 by the move, then the turn's pass counts it once (up through turn T + 4).
    assert int(trick_room_of(out)) == TRICK_ROOM_TURNS - 1
    up = _trick(tables, names, turns=3)
    out = assert_matches_reference(tables, up, A("Trick Room", "Growl", up, names), _hits())
    assert int(trick_room_of(out)) == 0  # a second use ends it (the pass has nothing left to count)


def test_trick_room_ends_in_vanilla_and_is_rearmed_in_kaizo(
    tables, names, kaizo_tables, kaizo_names
):
    """The pass takes the last turn to 0: vanilla's end script only prints, Kaizo's sets the field
    condition again (``Tables.trick_room_rearm``)."""
    for t, n, after in ((tables, names, 0), (kaizo_tables, kaizo_names, TRICK_ROOM_TURNS)):
        s = _trick(t, n, turns=1, mine=("Growl",))
        out = assert_matches_reference(t, s, A("Growl", "Growl", s, n), _hits())
        assert int(trick_room_of(out)) == after
        s = _trick(t, n, turns=2, mine=("Growl",))
        out = assert_matches_reference(t, s, A("Growl", "Growl", s, n), _hits())
        assert int(trick_room_of(out)) == 1


def test_trick_room_reverses_the_order_and_an_area_start_sets_five(tables, names):
    """Under Trick Room the slower Geodude hits the 1 HP Monferno first, which never moves."""
    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    s = s._replace(hp=s.hp.at[0, 0].set(1), stats=s.stats.at[0, 0, SPE].set(500))
    acts = A("Tackle", "Tackle", s, names)
    normal = assert_matches_reference(tables, s, acts, _hits())
    assert int(normal.hp[1, 0]) < int(s.hp[1, 0])
    started = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Tackle",))],
        [_spec(names, "Geodude", 40, ("Tackle",))],
        trick_room=TRICK_ROOM_TURNS,
    )
    assert int(trick_room_of(started)) == TRICK_ROOM_TURNS
    reversed_ = with_trick_room(s, 3)
    out = assert_matches_reference(tables, reversed_, acts, _hits())
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    for bad in (-1, TRICK_ROOM_TURNS + 1):
        try:
            make_state(
                tables,
                [_spec(names, "Monferno", 20, ("Tackle",))],
                [_spec(names, "Geodude", 40, ("Tackle",))],
                trick_room=bad,
            )
        except ValueError:
            continue
        raise AssertionError(bad)


# --- #191 Skill Swap (E3-2-5) ---------------------------------------------------------------
#
# ``effect_script_0191.s:5`` runs ``subscript_exchange_abilities.s``: fails (:5-13) on a miss or a
# semi-invulnerable defender, a raw Wonder Guard or Multitype ability of either battler, a held
# Griseous Orb of either, or both abilities ABILITY_NONE; else (:17-24) both ``truant`` become
# ``totalTurns & 1`` and the two ``BattleMon.ability`` swap. A gained entry ability acts at the
# move's end (battle_controller_player.c:3973, battle_lib.c:3720-3851) once: its announced flag.


def _ss(t, n, mine, foe, mine_moves=("Skill Swap", "Tackle"), foe_moves=("Tackle",), **over):
    """Monferno (side 0) with ``mine`` against Geodude with ``foe``, at turn 1 (so a Truant write
    of ``totalTurns & 1`` reads 1); ``over``: state fields."""
    s = _fight(t, n, mine_moves, foe_moves)
    s = s._replace(
        ability=s.ability.at[0, 0].set(mine).at[1, 0].set(foe),
        turn=jnp.asarray(1, s.turn.dtype),
    )
    return s._replace(**over) if over else s


def _flags(s, field, values):
    return jnp.asarray(values, getattr(s, field).dtype)


def test_skill_swap_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 191 (effect_script_0191.s:5); Skill Swap is in both games' move tables."""
    assert DECOMP_EFFECT_TO_ID[191] == (EffectId.SKILL_SWAP, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Skill Swap")]) == EffectId.SKILL_SWAP


def test_skill_swap_swaps_abilities_sets_truant_and_the_ai_learns_both(
    tables, names, kaizo_tables, kaizo_names
):
    """:21-24 swap the BattleMon abilities, :17-20 write ``totalTurns & 1`` (1 at turn 1) to both
    ``truant``; both are tagged for the AI (battle_script.c:3402-3404): the player's new ability."""
    from nuzline.engine.ids import ABILITY_DECOMP_IDS

    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = _ss(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
        out = assert_matches_reference(t, s, A("Skill Swap", "Tackle", s, n), _hits())
        assert int(out.ability[0, 0]) == AbilityId.ROCK_HEAD
        assert int(out.ability[1, 0]) == AbilityId.BLAZE
        assert int(decomp_abilities(t, out)[0, 0]) == ABILITY_DECOMP_IDS[AbilityId.ROCK_HEAD]
        assert int(decomp_abilities(t, out)[1, 0]) == ABILITY_DECOMP_IDS[AbilityId.BLAZE]
        assert int(out.ai_ability_known[0]) == ABILITY_DECOMP_IDS[AbilityId.ROCK_HEAD]
        assert out.truant.tolist() == [1, 1]


def test_skill_swap_ends_with_the_battler(tables, names):
    """The swap is the BattleMon's (battle_lib.c:124 reloads the party Pokemon's ability on the
    next switch-in): ``_restore_ability`` puts the species' back when the user leaves; the
    defender, which stays, keeps what it got."""
    two = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Skill Swap",)), _spec(names, "Monferno", 20, ("Scratch",))],
        [_spec(names, "Geodude", 40, ("Tackle",))],
    )
    two = two._replace(ability=two.ability.at[1, 0].set(AbilityId.ROCK_HEAD))
    own = int(two.ability[0, 0])
    out = assert_matches_reference(tables, two, A("Skill Swap", "Tackle", two, names), _hits())
    assert int(out.ability[0, 0]) == AbilityId.ROCK_HEAD and int(out.ability[1, 0]) == own
    switch = jnp.array([4 + 1, names.move_slot(out, 1, "Tackle")], jnp.int32)
    out2 = assert_matches_reference(tables, out, switch, empty_outcome())
    assert int(out2.ability[0, 0]) == own
    assert int(out2.ability[1, 0]) == own  # the defender kept the user's ability


def test_skill_swap_fails_on_its_listed_conditions_and_goes_through_a_substitute(tables, names):
    """subscript_exchange_abilities.s:5-13: a semi-invulnerable defender, a raw Wonder
    Guard (:6-7) or Multitype (:8-9) ability of either battler, a held Griseous Orb of either
    (:10-11), both abilities none (:12-13): nothing swaps and ``truant`` stays. The user's none
    alone is fine, and no Substitute test (a Substitute does not stop it)."""
    orb = names.item("Griseous Orb")
    base = _ss(tables, names, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
    acts = A("Skill Swap", "Tackle", base, names)
    fails = {
        "wonder guard foe": base._replace(
            ability=base.ability.at[1, 0].set(AbilityId.WONDER_GUARD)
        ),
        "wonder guard user": base._replace(
            ability=base.ability.at[0, 0].set(AbilityId.WONDER_GUARD)
        ),
        "orb user": base._replace(item=base.item.at[0, 0].set(orb)),
        "orb foe": base._replace(item=base.item.at[1, 0].set(orb)),
        "both none": base._replace(ability=base.ability.at[:, 0].set(AbilityId.NONE)),
        "semi-invulnerable": base._replace(semi_invulnerable=base.semi_invulnerable.at[1].set(1)),
    }
    for label, case in fails.items():
        out = assert_matches_reference(tables, case, acts, _hits())
        assert out.ability[:, 0].tolist() == case.ability[:, 0].tolist(), label
        assert out.truant.tolist() == case.truant.tolist(), label
    # Accuracy 0: CheckMoveHitAccuracy returns before its roll (bcp.c:2901), so no
    # MOVE_STATUS_MISSED arises and a failing roll class still swaps.
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, base, acts, miss)
    assert out.ability[:, 0].tolist() == base.ability[::-1, 0].tolist()
    arceus = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Skill Swap",))],
        [_spec(names, "Arceus", 40, ("Tackle",))],
    )
    out = assert_matches_reference(
        tables, arceus, A("Skill Swap", "Tackle", arceus, names), _hits()
    )
    assert int(decomp_abilities(tables, out)[1, 0]) == 121  # ABILITY_MULTITYPE stays
    assert int(out.ability[0, 0]) == int(arceus.ability[0, 0])
    # The user's ABILITY_NONE alone, and behind a Substitute: both swap.
    none_user = _ss(tables, names, AbilityId.NONE, AbilityId.ROCK_HEAD)
    out = assert_matches_reference(tables, none_user, acts, _hits())
    assert (
        int(out.ability[0, 0]) == AbilityId.ROCK_HEAD and int(out.ability[1, 0]) == AbilityId.NONE
    )
    shielded = base._replace(substitute_hp=base.substitute_hp.at[1].set(30))
    out = assert_matches_reference(tables, shielded, acts, _hits())
    assert (
        int(out.ability[0, 0]) == AbilityId.ROCK_HEAD and int(out.ability[1, 0]) == AbilityId.BLAZE
    )


def test_skill_swap_hits_a_flying_foe_under_lock_on_or_no_guard(tables, names):
    """:5 reads the move status MISSED | SEMI_INVULNERABLE, not the defender's bit:
    CheckMoveHitOverrides returns for a Lock-On on the defender or either side's No Guard
    (battle_controller_player.c:3016-3021) before it sets MOVE_STATUS_SEMI_INVULNERABLE
    (:3034-3041), so a Skill Swap into a foe in the air swaps; without either it fails."""
    from nuzline.engine.ids import SemiKind

    def airborne(mine, foe, **over):
        s = _ss(tables, names, mine, foe, foe_moves=("Fly",), **over)
        # The user outspeeds the foe, and the foe's Fly cannot KO it (Attack 1).
        return s._replace(
            stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, 0].set(1),
            semi_invulnerable=s.semi_invulnerable.at[1].set(SemiKind.AIRBORNE),
            locked_move=s.locked_move.at[1].set(names.move("Fly")),
        )

    plain = airborne(AbilityId.BLAZE, AbilityId.ROCK_HEAD)
    cases = {
        "no guard user": airborne(AbilityId.NO_GUARD, AbilityId.ROCK_HEAD),
        "no guard foe": airborne(AbilityId.BLAZE, AbilityId.NO_GUARD),
        "lock-on": plain._replace(
            lock_on=plain.lock_on.at[1].set(2), lock_on_target=plain.lock_on_target.at[1].set(1)
        ),
    }
    acts = A("Skill Swap", "Fly", plain, names)
    for label, case in cases.items():
        out = assert_matches_reference(tables, case, acts, _hits())
        assert out.ability[:, 0].tolist() == case.ability[::-1, 0].tolist(), label
        assert out.truant.tolist() == [1, 1], label
    out = assert_matches_reference(tables, plain, acts, _hits())
    assert out.ability[:, 0].tolist() == plain.ability[:, 0].tolist()


def test_skill_swap_gives_the_new_truant_holder_a_fresh_parity(tables, names):
    """:17-20: both ``truant`` are ``totalTurns & 1`` (1 at turn 1); the user, now Truant, loafs
    on the next turn (0 != 1, ``order.truant_loafs``), the defender, now without it, never."""
    from nuzline.engine.order import truant_loafs

    s = _ss(tables, names, AbilityId.BLAZE, AbilityId.TRUANT)
    out = assert_matches_reference(tables, s, A("Skill Swap", "Tackle", s, names), _hits())
    assert int(out.ability[0, 0]) == AbilityId.TRUANT and int(out.turn) == 2
    assert out.truant.tolist() == [1, 1]
    assert bool(truant_loafs(out, 0)) and not bool(truant_loafs(out, 1))


def test_skill_swap_hands_over_an_entry_ability_that_acts_once_at_the_move_end(tables, names):
    """The Intimidate the defender announced at its entry (flag set) arrives at the user with the
    flag clear: MoveEnd's TriggerEffectOnSwitch (bcp.c:3973) fires it at once (lib.c:3799-3803),
    the foe's Attack -1; the next move end does not (the flag is set); an Intimidate the user had
    announced already stays quiet. Drought acts the same way (:3740-3790: permanent sun)."""
    from nuzline.engine.ids import Weather

    s = _ss(tables, names, AbilityId.BLAZE, AbilityId.INTIMIDATE)
    s = s._replace(intimidate_announced=_flags(s, "intimidate_announced", [0, 1]))
    acts = A("Skill Swap", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _hits())
    assert int(out.ability[0, 0]) == AbilityId.INTIMIDATE
    assert int(out.stages[1, Stat.ATK]) == -1 and int(out.stages[0, Stat.ATK]) == 0
    assert out.intimidate_announced.tolist() == [1, 1]
    again = assert_matches_reference(tables, out, A("Tackle", "Tackle", out, names), _hits())
    assert int(again.stages[1, Stat.ATK]) == -1
    spent = s._replace(intimidate_announced=_flags(s, "intimidate_announced", [1, 1]))
    out = assert_matches_reference(tables, spent, acts, _hits())
    assert int(out.stages[1, Stat.ATK]) == 0
    rain = _ss(tables, names, AbilityId.BLAZE, AbilityId.DROUGHT)
    rain = rain._replace(weather_announced=_flags(rain, "weather_announced", [0, 1]))
    out = assert_matches_reference(tables, rain, acts, _hits())
    assert int(weather_of(out)) == Weather.SUN and int(weather_timer_of(out)) == 0
    assert out.weather_announced.tolist() == [1, 1]


def test_the_announced_flags_are_set_at_entry_and_cleared_by_a_switch(tables, names):
    """``make_state`` runs the leads' switch-in checks: an Intimidate lead has its flag set (the
    foe's lead has no ability here), a switch clears the flags (``VOLATILE_FIELDS``,
    battle_lib.c:103-106) before the newcomer's own entry sets its own."""
    assert {"trace_announced", "weather_announced"} <= set(VOLATILE_FIELDS)
    assert {"intimidate_announced", "download_announced"} <= set(VOLATILE_FIELDS)
    assert not {"trace_announced", "intimidate_announced"} & set(BATON_PASSED_FIELDS)
    shinx = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert shinx.intimidate_announced.tolist() == [1, 0]  # set by the lead's entry check
    lead = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Tackle",)), _spec(names, "Monferno", 20, ("Tackle",))],
        [_spec(names, "Geodude", 40, ("Tackle",))],
    )
    lead = lead._replace(
        ability=lead.ability.at[0, 0].set(AbilityId.INTIMIDATE),
        intimidate_announced=lead.intimidate_announced.at[0].set(1),
    )
    out = assert_matches_reference(tables, lead, A("Tackle", "Tackle", lead, names), _hits())
    assert out.intimidate_announced.tolist() == [
        1,
        0,
    ]  # the Intimidate that never fired for Geodude
    switch = jnp.array([4 + 1, names.move_slot(out, 1, "Tackle")], jnp.int32)
    out2 = assert_matches_reference(tables, out, switch, empty_outcome())
    assert out2.intimidate_announced.tolist() == [0, 0]


def test_skill_swap_is_seen_by_hit_thresholds_and_crit_odds(tables, names):
    """A faster foe's Skill Swap takes the side's Compoundeyes (the foe's own ability is the
    side's after it) before its check (bcp.c:2876-2947): Rock Throw's 90 is 117 at the turn
    start and 90 swapped; it also moves Super Luck (battle_lib.c:7126), so ``CRIT_ODDS_EFFECTS``
    carries it (a swap of every battler's ability with its foe's)."""
    from nuzline.engine.outcome import CRIT_ODDS_EFFECTS

    s = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Skill Swap"], opp_faster=True
    )
    s = s._replace(ability=s.ability.at[0, int(s.active[0])].set(AbilityId.COMPOUNDEYES))
    acts = A("Rock Throw", "Skill Swap", s, names)
    assert hit_law(tables, s, acts, 0) == {1: 90, 91: 10}
    assert EffectId.SKILL_SWAP in CRIT_ODDS_EFFECTS
    swapped = CRIT_ODDS_EFFECTS[EffectId.SKILL_SWAP](s)
    assert int(swapped.ability[1, int(s.active[1])]) == AbilityId.COMPOUNDEYES
    assert int(swapped.ability[0, int(s.active[0])]) == int(s.ability[1, int(s.active[1])])


# --- #239 Gastro Acid (E3-2-6) --------------------------------------------------------------
#
# ``effect_script_0239.s:5`` runs ``subscript_suppress_target_ability.s``: fails (:5-8) behind a
# Substitute, on a defender that has MOVE_EFFECT_ABILITY_SUPPRESSED, on a miss or a
# semi-invulnerable defender, on a raw Multitype ability; else :10 sets the bit. Every
# ``Battler_Ability`` read then gives ABILITY_NONE (battle_lib.c:3088-3093) unless the raw ability
# is Multitype; the bit ends with a switch (battle_lib.c:2071) and Baton Pass keeps it (:2076).


def _gastro(t, n, mine=("Gastro Acid",), foe=("Tackle",), bit=(0, 0), **over):
    """``_fight`` with the ``ability_suppressed`` bits ``bit`` (side 0, side 1)."""
    s = _fight(t, n, mine, foe)
    s = s._replace(ability_suppressed=jnp.asarray(bit, s.ability_suppressed.dtype))
    return s._replace(**over) if over else s


def _hp_lost(before, after, side):
    return int(before.hp[side, 0]) - int(after.hp[side, 0])


def test_gastro_acid_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 239 (effect_script_0239.s:5); Gastro Acid is in both games' move tables."""
    assert DECOMP_EFFECT_TO_ID[239] == (EffectId.GASTRO_ACID, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Gastro Acid")]) == EffectId.GASTRO_ACID


def test_gastro_acid_sets_the_bit_and_leaves_the_ability_written(
    tables, names, kaizo_tables, kaizo_names
):
    """:10 sets the defender's bit; ``BattleMon.ability`` is not written (no AI tag), the user's
    own bit stays clear."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = _gastro(t, n)
        s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.ROCK_HEAD))
        out = assert_matches_reference(t, s, A("Gastro Acid", "Tackle", s, n), _hits())
        assert out.ability_suppressed.tolist() == [0, 1]
        assert int(out.ability[1, 0]) == AbilityId.ROCK_HEAD
        assert int(out.ai_ability_known[0]) == 0


def test_gastro_acid_fails_on_its_listed_conditions(tables, names):
    """subscript_suppress_target_ability.s:5-8: a Substitute (:5, the bit is not set behind it), a
    defender that has the bit (:6), a miss or a semi-invulnerable defender (:7), a raw Multitype
    ability (:8). The Multitype exception is also the accessor's (lib.c:3090)."""
    from nuzline.engine.damage import battler_decomp_ability

    s = _gastro(tables, names)
    acts = A("Gastro Acid", "Tackle", s, names)
    shielded = s._replace(substitute_hp=s.substitute_hp.at[1].set(30))
    out = assert_matches_reference(tables, shielded, acts, _hits())
    assert out.ability_suppressed.tolist() == [0, 0]
    flying = s._replace(semi_invulnerable=s.semi_invulnerable.at[1].set(1))
    out = assert_matches_reference(tables, flying, acts, _hits())
    assert out.ability_suppressed.tolist() == [0, 0]
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, s, acts, miss)
    assert out.ability_suppressed.tolist() == [0, 0]
    already = _gastro(tables, names, bit=(0, 1))
    out = assert_matches_reference(tables, already, acts, _hits())
    assert out.ability_suppressed.tolist() == [0, 1]
    arceus = make_state(
        tables,
        [_spec(names, "Monferno", 20, ("Gastro Acid",))],
        [_spec(names, "Arceus", 40, ("Tackle",))],
    )
    out = assert_matches_reference(
        tables, arceus, A("Gastro Acid", "Tackle", arceus, names), _hits()
    )
    assert out.ability_suppressed.tolist() == [0, 0]
    # A Multitype holder that has the bit (a hand-built state) still reads Multitype:
    # Battler_Ability's exception.
    forced = out._replace(ability_suppressed=jnp.asarray([0, 1], out.ability_suppressed.dtype))
    assert int(battler_decomp_ability(tables, forced, 1)) == 121  # ABILITY_MULTITYPE


def test_gastro_acid_hides_a_damage_ability_and_an_immunity(tables, names):
    """Battler_Ability gives NONE for a suppressed battler (lib.c:6653-6654: attacker and defender
    both): a suppressed Levitate stops granting Earthquake's immunity, a suppressed Huge Power
    stops doubling its holder's Attack."""
    s = _fight(tables, names, ("Earthquake",), ("Tackle",))
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.LEVITATE))
    acts = A("Earthquake", "Tackle", s, names)
    base = assert_matches_reference(tables, s, acts, _hits())
    assert _hp_lost(s, base, 1) == 0  # the Levitate immunity
    bit = s._replace(ability_suppressed=s.ability_suppressed.at[1].set(1))
    hit = assert_matches_reference(tables, bit, acts, _hits())
    assert _hp_lost(s, hit, 1) > 0
    power = _fight(tables, names, ("Tackle",), ("Tackle",))
    power = power._replace(ability=power.ability.at[0, 0].set(AbilityId.HUGE_POWER))
    acts = A("Tackle", "Tackle", power, names)
    full = assert_matches_reference(tables, power, acts, _hits())
    off = power._replace(ability_suppressed=power.ability_suppressed.at[0].set(1))
    weak = assert_matches_reference(tables, off, acts, _hits())
    assert _hp_lost(power, weak, 1) < _hp_lost(power, full, 1)


def test_gastro_acid_hides_a_speed_ability(tables, names):
    """Swift Swim in rain doubles its holder's speed (CompareBattlerSpeed, lib.c:1213-1214,
    :1228-1238, through ``Battler_Ability``); suppressed, the foe moves first."""
    from nuzline.engine.ids import Weather
    from nuzline.engine.order import action_order, effective_speed

    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    s = with_weather(s, Weather.RAIN)._replace(
        ability=s.ability.at[0, 0].set(AbilityId.SWIFT_SWIM),
        stats=s.stats.at[0, 0, SPE].set(100).at[1, 0, SPE].set(150),
    )
    acts = A("Tackle", "Tackle", s, names)
    assert int(effective_speed(tables, s, 0)) == 200
    assert int(action_order(tables, s, acts, empty_outcome())[0]) == 0
    off = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    assert int(effective_speed(tables, off, 0)) == 100
    assert int(action_order(tables, off, acts, empty_outcome())[0]) == 1
    assert_matches_reference(tables, off, acts, _hits())


def test_gastro_acid_hides_an_end_of_turn_ability(tables, names):
    """Speed Boost's +1 at the turn end reads ``Battler_Ability``: none while suppressed."""
    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.SPEED_BOOST))
    acts = A("Tackle", "Tackle", s, names)
    boosted = assert_matches_reference(tables, s, acts, _hits())
    assert _stage(boosted, 0, Stat.SPE) == 1
    off = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    quiet = assert_matches_reference(tables, off, acts, _hits())
    assert _stage(quiet, 0, Stat.SPE) == 0


def test_gastro_acid_makes_a_klutz_holders_item_live(tables, names):
    """``Battler_HeldItem`` hides the item while ``Battler_Ability`` is Klutz (lib.c:5352-5356);
    a suppressed Klutz is NONE, so the Leftovers heal at the turn end."""
    from nuzline.engine.state import held_item

    leftovers = names.item("Leftovers")
    s = _fight(tables, names, ("Tackle",), ("Growl",))
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.KLUTZ),
        item=s.item.at[0, 0].set(leftovers),
        hp=s.hp.at[0, 0].set(s.max_hp[0, 0] // 2),
    )
    assert int(held_item(s, 0)) == 0
    off = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    assert int(held_item(off, 0)) == leftovers
    acts = A("Tackle", "Growl", s, names)
    hidden = assert_matches_reference(tables, s, acts, _hits())
    live = assert_matches_reference(tables, off, acts, _hits())
    assert int(live.hp[0, 0]) - int(hidden.hp[0, 0]) == max(1, int(s.max_hp[0, 0]) // 16)


def test_gastro_acid_ends_on_a_switch_and_baton_pass_keeps_it(tables, names):
    """The bit is a volatile (switch-out clears the mask, lib.c:2071) that Baton Pass keeps
    (moves.h:46-48, lib.c:2076). A Monferno with Intimidate on the bench: a plain switch clears
    the bit, so its Intimidate (``Battler_Ability``) drops the foe's Attack; a Baton Pass keeps
    it, so the newcomer is suppressed and its Intimidate stays silent (an unsuppressed Baton Pass
    newcomer's does fire)."""
    from nuzline.engine.actions import MOVE_ACTIONS

    assert "ability_suppressed" in VOLATILE_FIELDS and "ability_suppressed" in BATON_PASSED_FIELDS
    n = names
    s = make_state(
        tables,
        [_spec(n, "Ninjask", 30, ("Baton Pass",)), _spec(n, "Monferno", 20, ("Tackle",))],
        [_spec(n, "Geodude", 20, ("Tackle",))],
    )
    s = s._replace(
        ability=s.ability.at[:, 0].set(AbilityId.NONE).at[0, 1].set(AbilityId.INTIMIDATE)
    )
    off = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    switch = jnp.array([4 + 1, n.move_slot(s, 1, "Tackle")], jnp.int32)
    out = assert_matches_reference(tables, off, switch, empty_outcome())
    assert out.ability_suppressed.tolist() == [0, 0]
    assert _stage(out, 1, Stat.ATK) == -1  # the newcomer's Intimidate fired

    def baton(state):
        acts = A("Baton Pass", "Tackle", state, n)
        paused = assert_matches_reference(tables, state, acts, _hits())
        assert int(paused.pivot_battler) == 1
        resume = np.asarray(paused.ts_actions).astype(int).tolist()
        resume[0] = MOVE_ACTIONS + 1
        return assert_matches_reference(tables, paused, tuple(resume), empty_outcome())

    passed = baton(off)
    assert int(passed.active[0]) == 1 and passed.ability_suppressed.tolist() == [1, 0]
    assert _stage(passed, 1, Stat.ATK) == 0
    control = baton(s)
    assert control.ability_suppressed.tolist() == [0, 0]
    assert _stage(control, 1, Stat.ATK) == -1


def test_gastro_acid_leaves_super_luck_and_frees_a_klutz_crit_item(tables, names):
    """CalcCrit reads the attacker's ability raw (lib.c:7122, :7126: Super Luck stays) and its item
    through ``Battler_HeldItem`` (:7117: a suppressed Klutz holder's Scope Lens counts)."""
    from nuzline.engine.outcome import side_crit_odds

    tackle = names.move("Tackle")
    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    luck = s._replace(ability=s.ability.at[0, 0].set(AbilityId.SUPER_LUCK))
    off = luck._replace(ability_suppressed=luck.ability_suppressed.at[0].set(1))
    plain = int(side_crit_odds(tables, s, 0, tackle))
    with_luck = int(side_crit_odds(tables, luck, 0, tackle))
    assert with_luck < plain  # Super Luck: +1 stage
    assert int(side_crit_odds(tables, off, 0, tackle)) == with_luck
    lens = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.KLUTZ),
        item=s.item.at[0, 0].set(names.item("Scope Lens")),
    )
    free = lens._replace(ability_suppressed=lens.ability_suppressed.at[0].set(1))
    assert int(side_crit_odds(tables, lens, 0, tackle)) == plain
    assert int(side_crit_odds(tables, free, 0, tackle)) < plain


def test_gastro_acid_stops_ingrain_anchoring_against_roar(tables, names):
    """Roar's Ingrain test compares the whole mask to MOVE_EFFECT_INGRAIN
    (subscript_force_target_to_switch_or_flee.s:7): the suppressed bit is another bit."""
    from nuzline.engine.effects import ingrain_anchors

    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    s = s._replace(ingrain=s.ingrain.at[1].set(1))
    assert bool(ingrain_anchors(tables, s, 1))
    off = s._replace(ability_suppressed=s.ability_suppressed.at[1].set(1))
    assert not bool(ingrain_anchors(tables, off, 1))


def test_gastro_acid_is_seen_by_hit_thresholds_and_crit_odds(tables, names):
    """A faster foe's Gastro Acid takes the side's Compoundeyes before its check (bcp.c:2876-2947):
    Rock Throw's 90 is 117 at the turn start and 90 suppressed (hit_thresholds holds both). Its
    write to the crit inputs is every battler's bit: Super Luck (raw) does not move, a Klutz
    holder's Scope Lens does (``crit_variants``). It is an input clause of the enumeration and
    deals no table damage."""
    from nuzline.engine.enumerate import INPUT_CLAUSE_EFFECTS, NON_TABLE_DAMAGE_EFFECTS
    from nuzline.engine.outcome import CRIT_ODDS_EFFECTS, crit_variants

    assert EffectId.GASTRO_ACID in INPUT_CLAUSE_EFFECTS
    assert EffectId.GASTRO_ACID in NON_TABLE_DAMAGE_EFFECTS
    s = roark_state(
        tables, names, player_moves=["Rock Throw"], opp_moves=["Gastro Acid"], opp_faster=True
    )
    mon = int(s.active[0])
    s = s._replace(ability=s.ability.at[0, mon].set(AbilityId.COMPOUNDEYES))
    acts = A("Rock Throw", "Gastro Acid", s, names)
    assert hit_law(tables, s, acts, 0) == {1: 90, 91: 10}
    assert EffectId.GASTRO_ACID in CRIT_ODDS_EFFECTS
    assert CRIT_ODDS_EFFECTS[EffectId.GASTRO_ACID](s).ability_suppressed.tolist() == [1, 1]
    bit = 1 << list(CRIT_ODDS_EFFECTS).index(EffectId.GASTRO_ACID)
    odds, possible = crit_variants(tables, s, acts)
    assert not bool(possible[0, bit])  # Super Luck / Compoundeyes: no crit input changes
    lens = s._replace(
        ability=s.ability.at[0, mon].set(AbilityId.KLUTZ),
        item=s.item.at[0, mon].set(names.item("Scope Lens")),
    )
    odds, possible = crit_variants(tables, lens, acts)
    assert bool(possible[0, bit]) and int(odds[0, bit]) < int(odds[0, 0])


def test_the_suppressed_bit_is_observed():
    """The "ability was suppressed" message (subscript_suppress_target_ability.s:12) is in the
    observation (``observe._NONZERO_SIDE``)."""
    from nuzline.engine.observe import OBS_FIELDS

    assert "ability_suppressed" in [name for name, _ in OBS_FIELDS]
