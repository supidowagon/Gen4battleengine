"""E1 batch H. H-1: the Future Sight / Doom Desire hit's accuracy law where an Acid Rain write
(subscript_pursuit.s:115, OPCODE_SUB_TO_ZERO battle_script.c:3282-3287; ledger G-12) can happen:
classes over the union of every mask the hit can read (``outcome.future_thresholds``' field
axis, ``outcome._future_fields``), and the roll kept whole (``loose``) only on a battler whose hit
is due (bcp.c:1767-1769, :1782; accuracy 0 draws nothing, bcp.c:2900-2902). H-2: Immunity, Own Tempo,
Water Veil, Magma Armor and Oblivious (blocks, MoveEnd cures, the switch-out and battle-won cures).
H-3: Color Change (per hit), Marvel Scale and Soundproof. H-4: the held items Quick Powder, Full
Incense, Shed Shell, Mental Herb, Destiny Knot, and the ids of the items with no in-battle hook. H-5:
``BattleSystem_TriggerFormChange``'s Cherrim, Arceus and Transformed Giratina-Origin blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import (
    ai_damage_for_roll,
    confusion_damage,
    damage_for_roll,
    hidden_power,
)
from nuzline.engine.effects import apply_move_effect
from nuzline.engine.enumerate import _future_hit, hit_class_law, outcome_laws
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Stat, Status, Type, Weather
from nuzline.engine.order import is_switch, move_effects
from nuzline.engine.outcome import (
    MAX_HITS,
    acid_rain_pairs,
    empty_outcome,
    future_thresholds,
    hit_class,
)
from tests.engine.d2_states import four, with_moves
from tests.engine.helpers import (
    A,
    as_species,
    assert_matches_reference,
    jit_step,
    roark_state,
    sample_rows,
    weather_of,
)
from tests.engine.test_field_mask import _hits, _masked, _pursuit_fight, _rearming


def _pending(s, names, defender, attacker, timer=1):
    """``s`` with a Future Sight of battler ``attacker`` on battler ``defender``: count ``timer``,
    5 stored damage."""
    return s._replace(
        future_timer=s.future_timer.at[defender].set(timer),
        future_damage=s.future_damage.at[defender].set(5),
        future_move=s.future_move.at[defender].set(names.move("Future Sight")),
        future_attacker=s.future_attacker.at[defender].set(1 + attacker),
        future_sight=s.future_sight.at[defender].set(1),
    )


def _accuracy(tables, names):
    return int(tables.move_accuracy[names.move("Future Sight")])


def _law(tables, s, acts, side):
    laws = outcome_laws(tables, s, acts)
    assert not bool(laws.future_loose[side])
    return _future_hit(laws, side)


def _same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize(
    "foe_action, start",
    [
        (0, Weather.HAIL),  # Growl: no chase
        (MOVE_ACTIONS + 1, Weather.HAIL),  # the switch's chase (subscript_switch_pokemon.s:11)
        (1, Weather.HAIL),  # U-turn's chase (subscript_attack_then_switch_out.s:54)
        (MOVE_ACTIONS + 1, Weather.NONE),  # 0 - 1 clamps to 0 at :115
    ],
)
def test_future_sight_law_is_classes_where_acid_rain_can_write(tables, names, foe_action, start):
    """T1. A due Future Sight on the foe (battler 1) while the player uses Pursuit: never kept
    whole for Acid Rain. On 0x80 the write leaves 0x7F (faintedMon 1) or 0x80 (faintedMon 0):
    no fog, no Gravity, so every row has the one threshold ``a`` (vanilla 90): the Growl row's
    law."""
    s = _pursuit_fight(
        tables, names, ("Pursuit",), ("Growl", "U-turn"), foe="Scyther", start_condition=start
    )
    s = _pending(s, names, 1, 0)
    acts = jnp.asarray([0, foe_action], jnp.int32)
    assert not bool(future_thresholds(tables, s, acts)[2][1])
    assert _law(tables, s, acts, 1) == hit_class_law([_accuracy(tables, names)])


def _fog_chase(tables, names, mask=0x8000, hp=1):
    """The foe's Geodude (battler 1, ``hp`` HP, None: full) switches to the Geodude behind it
    while the player's Pursuit chases it, on ``mask``; a Future Sight of the player's is due on
    battler 1."""
    s = _pursuit_fight(tables, names, ("Pursuit",), ("Growl",))
    s = s if hp is None else s.replace_hp(1, 0, hp)
    s = _pending(_masked(s, mask, 0), names, 1, 0)
    return s, A("Pursuit", MOVE_ACTIONS + 1, s, names)


def test_future_sight_law_on_deep_fog_covers_the_acid_rain_gravity(tables, names):
    """T2. 0x8000 (deep fog): a Pursuit KO of the switcher subtracts 1 (0x7FFF: Gravity 7, fog
    off; the turn end's count leaves 6, bcp.c:1229), so the hit reads x10/6 (bcp.c:2972-2974);
    no KO leaves the fog (x6/10, :2934); a newcomer's weather ability would clear the fog. Law
    over {a*6//10, a, a*10//6}: vanilla [1, 55, 91] / [54, 36, 10]. The step lands every roll
    after the KO and only rolls up to a*6//10 without it; each roll steps as its class. The chase
    has no accuracy check (subscript_pursuit.s:7-17: TryPursuit, CalcCrit, CalcDamage,
    ApplyTypeEffectiveness, no CheckMoveHit), so the no-KO row is a full-HP switcher."""
    a = _accuracy(tables, names)
    s, acts = _fog_chase(tables, names)
    assert _law(tables, s, acts, 1) == hit_class_law([a * 6 // 10, a, a * 10 // 6])
    for hp, lands in ((1, lambda u: True), (None, lambda u: u <= a * 6 // 10)):
        s, acts = _fog_chase(tables, names, hp=hp)
        thr, ok, _ = future_thresholds(tables, s, acts)
        # The baseline's hit roll 0 always misses (step._future_sight: ``roll > 0``); the
        # default roll 1 lands in every class.
        base = _hits()._replace(future_hit=jnp.zeros((2,), jnp.int8))
        missed = assert_matches_reference(tables, s, acts, base)
        assert (int(missed.hp[1, 0]) == 0) is (hp == 1)
        assert int(missed.active[1]) == 1 and int(missed.hp[1, 1]) > 5  # survives the hail
        for u in (a * 6 // 10, a * 6 // 10 + 1, a, a + 1, 100):
            o = base._replace(future_hit=jnp.asarray([0, u], jnp.int8))
            out = assert_matches_reference(tables, s, acts, o)
            assert (int(out.hp[1, 1]) == int(missed.hp[1, 1]) - 5) is lands(u), (hp, u)
            c = int(hit_class(u, thr[1], ok[1]))
            if c != u:
                o_c = o._replace(future_hit=jnp.asarray([0, c], jnp.int8))
                _same(jit_step(tables, s, acts, o_c), out)


def test_future_sight_law_on_deep_fog_covers_a_recoil_fainted_pursuer(tables, names):
    """T2 mirror. 0x8000: the player's 1 HP Monferno (battler 0) switches, the foe's 1 HP pursuer
    (Life Orb) KOs it and falls to its recoil (battle_lib.c:7357-7365): faintedMon is the
    pursuer, battler 1 (battle_script.c:1767-1770), 0x8000 - 1 = 0x7FFF. The foe's Future Sight
    due on battler 0 has the law over {a*6//10, a, a*10//6}; after the recoil it lands at a + 1."""
    a = _accuracy(tables, names)
    s = _pursuit_fight(tables, names, ("Growl",), ("Pursuit",), foe_item="Life Orb")
    s = _pending(_masked(s.replace_hp(0, 0, 1).replace_hp(1, 0, 1), 0x8000, 0), names, 0, 1)
    acts = A(MOVE_ACTIONS + 1, "Pursuit", s, names)
    assert _law(tables, s, acts, 0) == hit_class_law([a * 6 // 10, a, a * 10 // 6])
    left = []
    for u in (0, a + 1):
        o = _hits()._replace(future_hit=jnp.asarray([u, 0], jnp.int8))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
        left.append(int(out.hp[0, 1]))
    assert left[0] > 5 and left[1] == left[0] - 5


def test_future_sight_law_under_kaizo_gravity_follows_the_acid_rain_write(tables, names):
    """T3. Kaizo's Gravity rule (``gravity_rearm``, sub_seq 238, G-67) on 0x1000 (Gravity 1): no
    KO keeps it on (the count ends and re-arms, bcp.c:1227-1238), a KO of the switcher leaves
    0x0FFF (Gravity off). Law over {a, min(a*10//6, 100)}: roll a + 1 misses after the KO and
    lands without it. Under Gravity Pursuit (accuracy 100) cannot miss (hitRate * 10 / 6,
    bcp.c:2972-2978), so the no-KO row is a full-HP switcher, not a missed Pursuit. (The vanilla
    rule ends that Gravity: ``loose``.)"""
    t = _rearming(tables, "gravity_rearm")
    a = _accuracy(t, names)
    assert a < 100
    s, acts = _fog_chase(t, names, 0x1000)
    assert _law(t, s, acts, 1) == hit_class_law([a, min(a * 10 // 6, 100)])
    for hp, lands in ((1, False), (None, True)):
        s, acts = _fog_chase(t, names, 0x1000, hp)
        base = _hits()
        left = []
        for u in (0, a + 1):
            o = base._replace(future_hit=jnp.asarray([0, u], jnp.int8))
            left.append(int(assert_matches_reference(t, s, acts, o).hp[1, 1]))
        assert (left[1] == left[0] - 5) is lands


def test_only_a_due_hit_keeps_the_roll_whole(tables, names):
    """T4. An Embargo up keeps the hit's roll whole (B13-31), but only where the hit is due:
    with the count at 2 nothing is drawn this turn (bcp.c:1767-1769), so the stored value is the
    raw roll's class; at 1 it is the raw roll."""
    s = _pursuit_fight(tables, names, ("Growl",), ("Growl",))
    s = s._replace(embargo=s.embargo.at[0].set(1))
    acts = jnp.asarray([0, 0], jnp.int32)
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(8)])
    raw = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 28), (2,), 1, 101))(keys)
    for timer, loose in ((2, False), (1, True)):
        st = _pending(s, names, 1, 0, timer)
        thr, ok, whole = future_thresholds(tables, st, acts)
        assert bool(whole[1]) is loose
        stored = np.asarray(sample_rows(keys, tables, st, acts).future_hit)[:, 1]
        want = raw[:, 1] if loose else hit_class(raw[:, 1], thr[1], ok[1])
        assert np.array_equal(stored, np.asarray(want))


def test_four_battlers_keep_the_roll_whole_where_an_acid_pair_is_reachable(tables, names):
    """T5. B = 4: battler 2 switches while battler 3 (the other side) uses Pursuit. Only the pair
    [2, 3] is reachable; a due Future Sight is kept whole (a third battler can write the mask
    before the chase; the exact handling is the doubles milestone's), and without the Pursuit it
    is not."""
    s = with_moves(four(tables, names), names, 3, ("Pursuit",))
    s = _pending(s, names, 0, 1)
    acts = jnp.asarray([0, 0, MOVE_ACTIONS + 2, 0], jnp.int32)
    for pursuit in (names.move("Pursuit"), 0):
        moves = jnp.asarray([0, 0, 0, pursuit], jnp.int32)
        effect, _ = move_effects(tables, s, moves)
        uses = ~is_switch(acts)
        pairs = np.asarray(acid_rain_pairs(s, acts, uses, effect))
        want = np.zeros((4, 4), bool)
        want[2, 3] = bool(pursuit)
        assert np.array_equal(pairs, want)
        assert bool(future_thresholds(tables, s, acts, moves)[2][0]) is bool(pursuit)


# --- H-2: Immunity, Own Tempo, Water Veil, Magma Armor, Oblivious; the switch-out cure -------
# Rule sheets ability-immunity.md, ability-own-tempo.md, ability-cure-veil-armor-oblivious.md. The
# blocks: subscript_poison.s:13, subscript_badly_poison.s:31, subscript_burn.s:20,
# subscript_freeze.s:5, subscript_confuse.s:12, subscript_infatuate.s:5; the cures:
# BattleSystem_RecoverStatusByAbility (battle_lib.c:4378-4452, no HP test) at the move end
# (bcp.c:3965-3967) and Ability_ForbidsStatus (battle_lib.c:4454-4492) in
# BtlCmd_TryRestoreStatusOnSwitch (battle_script.c:9139-9160). A status here only exists where
# the game sets it: a Mold Breaker attacker (Cranidos) pierces the block, Gastro Acid suppresses
# the ability, Skill Swap moves it.

SWITCH_TO_1 = MOVE_ACTIONS + 1
DECOMP = {"Immunity": 17, "Own Tempo": 20, "Water Veil": 41, "Magma Armor": 40, "Oblivious": 12}


def _procs(player=0, opp=0, secondary=0):
    """``empty_outcome`` with the defenders' ``contact_proc`` and every ``secondary`` roll set."""
    rows = jnp.array([[player] * MAX_HITS, [opp] * MAX_HITS], jnp.int8)
    sec = jnp.full((2, MAX_HITS), secondary, jnp.int8)
    return empty_outcome()._replace(contact_proc=rows, secondary=sec)


def _holder(tables, names, species, ability, opp_moves, **kw):
    s = roark_state(
        tables,
        names,
        player_species=species,
        player_ability=ability,
        player_moves=kw.pop("player_moves", ("Leer",)),
        opp_moves=opp_moves,
        **kw,
    )
    assert int(s.ability[0, 0]) == int(AbilityId[ability.upper().replace(" ", "_")])
    return s


@pytest.mark.parametrize(
    "species, ability, move, status, outcome",
    [
        ("Zangoose", "Immunity", "Toxic", Status.TOXIC, empty_outcome()),
        ("Wailmer", "Water Veil", "Will-O-Wisp", Status.BURN, empty_outcome()),
    ],
)
def test_a_direct_status_is_blocked_and_revealed_a_mold_breaker_one_cured_at_move_end(
    tables, names, species, ability, move, status, outcome
):
    """The direct status move prints "prevents" (TAG_NICKNAME_ABILITY: the AI learns the decomp id,
    battle_script.c:11798-11810). Cranidos' Mold Breaker pierces the block (CheckIgnorableAbility);
    the defender's MoveEnd cure clears it again and reveals nothing (BTLSCR_MSG_TEMP,
    battle_script.c:11801-11802)."""
    s = _holder(tables, names, species, ability, (move,))
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), outcome)
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == DECOMP[ability]
    mb = _holder(tables, names, species, ability, (move,), opp_species="Cranidos")
    hit = apply_move_effect(
        tables, mb, 1, names.move(move), outcome, 0, connected=True, target_moved=False
    )
    assert int(hit.status[0, 0]) == status and int(hit.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, mb, A("Leer", move, mb, names), outcome)
    assert int(out.status[0, 0]) == Status.NONE and int(out.status_counter[0, 0]) == 0
    assert int(out.ai_ability_known[0]) == 0


@pytest.mark.parametrize(
    "species, ability, move",
    [
        ("Zangoose", "Immunity", "Poison Sting"),
        ("Wailmer", "Water Veil", "Ember"),
        ("Camerupt", "Magma Armor", "Ice Beam"),
    ],
)
def test_a_secondary_status_is_blocked_silently(tables, names, species, ability, move):
    """An INDIRECT side effect ends at the ability test with no message (poison :_177 -> :_322,
    burn :_264, freeze :_128 -> :_157)."""
    s = _holder(tables, names, species, ability, (move,))
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), _procs(secondary=1))
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0


def test_magma_armor_never_reveals_and_a_pierced_freeze_thaws_at_move_end(tables, names):
    """subscript_freeze.s:5 -> :_128 prints only the attack message; Cranidos' Ice Beam freezes
    Camerupt (Mold Breaker), MSGCOND_FREEZE cures it at the MoveEnd (battle_lib.c:4419-4424)."""
    s = _holder(tables, names, "Camerupt", "Magma Armor", ("Ice Beam",), opp_species="Cranidos")
    o = _procs(secondary=1)
    hit = apply_move_effect(tables, s, 1, names.move("Ice Beam"), o, 0, connected=True,
                            target_moved=False)  # fmt: skip
    assert int(hit.status[0, 0]) == Status.FREEZE
    out = assert_matches_reference(tables, s, A("Leer", "Ice Beam", s, names), o)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0


def test_psycho_shift_of_poison_onto_immunity_is_blocked_and_the_user_keeps_it(tables, names):
    """subscript_give_target_own_status.s:9-11 runs subscript_poison.s as a direct effect: Immunity
    prints "prevents poisoning"; the defender has no status, so the user keeps its own (:41-45)."""
    s = _holder(tables, names, "Zangoose", "Immunity", ("Psycho Shift",))
    s = s.with_status(1, 0, Status.POISON)
    out = assert_matches_reference(tables, s, A("Leer", "Psycho Shift", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.status[1, 0]) == Status.POISON
    assert int(out.ai_ability_known[0]) == DECOMP["Immunity"]


@pytest.mark.parametrize(
    "species, ability, opp_ability, status",
    [
        ("Zangoose", "Immunity", AbilityId.POISON_POINT, Status.POISON),
        ("Wailmer", "Water Veil", AbilityId.FLAME_BODY, Status.BURN),
    ],
)
def test_the_attackers_own_ability_stops_a_contact_status_silently(
    tables, names, species, ability, opp_ability, status
):
    """BattleSystem_TriggerAbilityOnHit's status runs on the attacker as an ABILITY side effect:
    its own Immunity / Water Veil ends it silently (subscript_poison.s:13 -> :_322,
    subscript_burn.s:20 -> :_264); the holder's message (and the reveal) never comes."""
    s = _holder(tables, names, species, ability, ("Leer",), player_moves=("Tackle",))
    s = s._replace(ability=s.ability.at[1, 0].set(opp_ability))
    out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), _procs(opp=1))
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0
    bare = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    out = assert_matches_reference(tables, bare, A("Tackle", "Leer", bare, names), _procs(opp=1))
    assert int(out.status[0, 0]) == status


def test_own_tempo_blocks_confuse_ray_and_swagger_and_reveals(tables, names):
    """subscript_confuse.s:12 -> :_141 (DIRECT) prints "prevents confusion" (:73). Swagger raises
    Attack first (subscript_swagger.s:10-12) and keeps the raise."""
    s = _holder(tables, names, "Spoink", "Own Tempo", ("Confuse Ray", "Swagger"))
    out = assert_matches_reference(tables, s, A("Leer", "Confuse Ray", s, names), empty_outcome())
    assert int(out.confusion[0]) == 0 and int(out.ai_ability_known[0]) == DECOMP["Own Tempo"]
    out = assert_matches_reference(tables, s, A("Leer", "Swagger", s, names), empty_outcome())
    assert int(out.confusion[0]) == 0 and int(out.ai_ability_known[0]) == DECOMP["Own Tempo"]
    assert int(out.stages[0, Stat.ATK]) == 2


def test_own_tempo_blocks_a_psybeam_confusion_silently_and_cures_a_pierced_one(tables, names):
    """An INDIRECT confusion ends at :_141 -> :_185 silently. Cranidos pierces Confuse Ray; the
    MoveEnd cure (MSGCOND_CONFUSION, battle_lib.c:4390-4395) clears it, no reveal."""
    s = _holder(tables, names, "Spoink", "Own Tempo", ("Psybeam",))
    out = assert_matches_reference(tables, s, A("Leer", "Psybeam", s, names), _procs(secondary=1))
    assert int(out.confusion[0]) == 0 and int(out.ai_ability_known[0]) == 0
    mb = _holder(tables, names, "Spoink", "Own Tempo", ("Confuse Ray",), opp_species="Cranidos")
    ray = names.move("Confuse Ray")
    hit = apply_move_effect(tables, mb, 1, ray, empty_outcome(), 0, connected=True,
                            target_moved=False)  # fmt: skip
    assert int(hit.confusion[0]) != 0
    out = assert_matches_reference(tables, mb, A("Leer", "Confuse Ray", mb, names), empty_outcome())
    assert int(out.confusion[0]) == 0 and int(out.ai_ability_known[0]) == 0


def test_oblivious_blocks_attract_and_reveals_and_cures_a_pierced_one(tables, names):
    """subscript_infatuate.s:5 -> :_055 prints "prevents romance" (:39). Cranidos pierces it; the
    MoveEnd cure (MSGCOND_INFATUATION, battle_lib.c:4426-4431; :_052) clears the infatuation."""
    kw = {"player_gender": "female", "opp_gender": "male"}
    s = _holder(tables, names, "Slowpoke", "Oblivious", ("Attract",), **kw)
    out = assert_matches_reference(tables, s, A("Leer", "Attract", s, names), empty_outcome())
    assert int(out.infatuated[0]) == 0 and int(out.ai_ability_known[0]) == DECOMP["Oblivious"]
    mb = _holder(tables, names, "Slowpoke", "Oblivious", ("Attract",), opp_species="Cranidos", **kw)
    hit = apply_move_effect(tables, mb, 1, names.move("Attract"), empty_outcome(), 0,
                            connected=True, target_moved=False)  # fmt: skip
    assert int(hit.infatuated[0]) != 0
    out = assert_matches_reference(tables, mb, A("Leer", "Attract", mb, names), empty_outcome())
    assert int(out.infatuated[0]) == 0 and int(out.ai_ability_known[0]) == 0


def test_cute_charm_cannot_infatuate_an_oblivious_attacker(tables, names):
    """subscript_infatuate.s:5 -> :_055 -> :_082 -> :_117: an ABILITY side effect ends silently."""
    s = _holder(tables, names, "Slowpoke", "Oblivious", ("Leer",), player_moves=("Tackle",),
                player_gender="female", opp_gender="male")  # fmt: skip
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))
    out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), _procs(opp=1))
    assert int(out.infatuated[0]) == 0
    bare = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    out = assert_matches_reference(tables, bare, A("Tackle", "Leer", bare, names), _procs(opp=1))
    assert int(out.infatuated[0]) != 0


@pytest.mark.parametrize(
    "species, ability, status",
    [
        ("Zangoose", "Immunity", Status.TOXIC),
        ("Spoink", "Own Tempo", None),
        ("Wailmer", "Water Veil", Status.BURN),
    ],
)
def test_skill_swap_gives_the_foe_the_ability_and_its_move_end_cures_it(
    tables, names, species, ability, status
):
    """subscript_exchange_abilities.s:21-24: the foe now holds the ability; the defender's MoveEnd
    RecoverStatusByAbility (bcp.c:3966-3967) clears its status (counter 0) or its confusion."""
    s = _holder(tables, names, species, ability, ("Leer",), player_moves=("Skill Swap",))
    if status is None:
        s = s._replace(confusion=s.confusion.at[1].set(3))
    else:
        s = s.with_status(1, 0, status, 2 if status == Status.TOXIC else 0)
    out = assert_matches_reference(tables, s, A("Skill Swap", "Leer", s, names), empty_outcome())
    assert int(out.ability[1, 0]) == int(s.ability[0, 0])
    assert int(out.status[1, 0]) == Status.NONE and int(out.status_counter[1, 0]) == 0
    assert int(out.confusion[1]) == 0


def _suppressed(tables, names, species, ability, move, outcome, **kw):
    """Gastro Acid on the holder, then ``move`` lands (Battler_Ability is NONE: no block, no MoveEnd
    cure, battle_lib.c:3088-3093)."""
    s = _holder(tables, names, species, ability, ("Gastro Acid", move, "Leer"),
                player_moves=("Leer", "Scratch"), **kw)  # fmt: skip
    s = assert_matches_reference(tables, s, A("Leer", "Gastro Acid", s, names), empty_outcome())
    return assert_matches_reference(tables, s, A("Leer", move, s, names), outcome)


@pytest.mark.parametrize(
    "species, ability, move, status, outcome",
    [
        ("Zangoose", "Immunity", "Toxic", Status.TOXIC, empty_outcome()),
        ("Persian", "Limber", "Thunder Wave", Status.PARALYSIS, empty_outcome()),
        ("Wailmer", "Water Veil", "Will-O-Wisp", Status.BURN, empty_outcome()),
        ("Camerupt", "Magma Armor", "Ice Beam", Status.FREEZE, _procs(secondary=1)),
    ],
)
def test_a_suppressed_holder_keeps_its_status_until_its_switch_out_cure(
    tables, names, species, ability, move, status, outcome
):
    """TryRestoreStatusOnSwitch (subscript_switch_pokemon.s:5-6) reads the party Pokemon's
    MON_DATA_ABILITY (battle_script.c:9149), not Battler_Ability: Ability_ForbidsStatus
    (battle_lib.c:4454-4492) clears it on the switch out."""
    s = _suppressed(tables, names, species, ability, move, outcome)
    assert int(s.status[0, 0]) == status and int(s.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, A(SWITCH_TO_1, "Leer", s, names), empty_outcome())
    assert int(out.active[0]) == 1
    assert int(out.status[0, 0]) == Status.NONE and int(out.status_counter[0, 0]) == 0


def test_the_switch_out_cure_reads_the_party_ability_after_skill_swap(tables, names):
    """Zangoose swaps its Immunity away (subscript_exchange_abilities.s:21-24 writes only the
    battler's), is badly poisoned, and its switch out still clears it (MON_DATA_ABILITY,
    battle_script.c:9149)."""
    s = _holder(tables, names, "Zangoose", "Immunity", ("Toxic", "Leer"),
                player_moves=("Skill Swap", "Leer"))  # fmt: skip
    s = assert_matches_reference(tables, s, A("Skill Swap", "Leer", s, names), empty_outcome())
    assert int(s.ability[0, 0]) != AbilityId.IMMUNITY
    s = assert_matches_reference(tables, s, A("Leer", "Toxic", s, names), empty_outcome())
    assert int(s.status[0, 0]) == Status.TOXIC
    out = assert_matches_reference(tables, s, A(SWITCH_TO_1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE


def test_the_battle_won_cure_clears_a_suppressed_holders_status(tables, names):
    """subscript_battle_won.s:6-7: TryRestoreStatusOnSwitch on the player's battler after the last
    KO; Gastro Acid left Zangoose badly poisoned."""
    s = _suppressed(tables, names, "Zangoose", "Immunity", "Toxic", empty_outcome(), opp_count=1)
    assert int(s.status[0, 0]) == Status.TOXIC
    s = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.status[0, 0]) == Status.NONE


# --- H-3: Color Change, Marvel Scale, Soundproof ----------------------------------------------
# Rule sheets ability-color-change.md, ability-marvel-scale.md, ability-soundproof.md. Color
# Change: BattleSystem_TriggerAbilityOnHit (battle_lib.c:4232-4253), subscript_color_change.s;
# each hit of a multi-hit move re-runs CHECK_TYPE_CHART (bcp.c:3272-3278, SYSCTL_MULTI_HIT_MOVE
# does not skip it; BattleSystem_SetupLoop clears moveStatusFlags, battle_lib.c:3357-3362).
# Marvel Scale: CalcMoveDamage's defenseStat x150/100 (battle_lib.c:6797-6800). Soundproof:
# TriggerImmunityAbility (battle_lib.c:3473-3486, :3523-3530) under MOVE_STATUS_DID_NOT_HIT too
# (bcp.c:2791-2792), the uproar wake-ups (bcp.c:2430-2439, :1505-1515), the uproar sleep test
# (subscript_fall_asleep.s:44) and TryPerishSong (battle_script.c:5474). Their Kaizo holders
# (Kecleon, Milotic, Whismur's line) are player Pokemon: the holder is the player's lead.

DECOMP_H3 = {"Color Change": 16, "Marvel Scale": 63, "Soundproof": 43}
_ROLL = 15  # empty_outcome's roll (100 %)


def _types(s, side=0, slot=0):
    return tuple(int(t) for t in s.types[side, slot])


def _lost(before, after, side=0, slot=0):
    return int(before.hp[side, slot]) - int(after.hp[side, slot])


def _dmg(tables, s, names, move, side=1, crit=False):
    """The engine's damage of ``side``'s ``move`` on the other side's lead at ``_ROLL``."""
    return int(damage_for_roll(tables, s, side, names.move(move), crit, _ROLL, def_side=1 - side))


def _kecleon(tables, names, opp_moves, **kw):
    return _holder(tables, names, "Kecleon", "Color Change", opp_moves, **kw)


def test_color_change_takes_the_hits_type_and_reveals_and_the_next_hit_reads_it(tables, names):
    """A Flamethrower makes Kecleon Fire / Fire (subscript_color_change.s:9-10) and tags its
    ability (:6, TAG_NICKNAME_ABILITY_TYPE: battle_script.c:11798-11810). The next turn's
    Flamethrower meets Fire (x0.5); Kecleon's own Flamethrower gets STAB (MON_HAS_TYPE,
    battle_lib.c:2592)."""
    s = _kecleon(tables, names, ("Flamethrower",), player_moves=("Leer", "Flamethrower"))
    out = assert_matches_reference(tables, s, A("Leer", "Flamethrower", s, names), empty_outcome())
    assert _types(out) == (Type.FIRE, Type.FIRE)
    assert int(out.ai_ability_known[0]) == DECOMP_H3["Color Change"]
    assert _lost(s, out) == _dmg(tables, s, names, "Flamethrower")
    again = assert_matches_reference(
        tables, out, A("Flamethrower", "Flamethrower", out, names), empty_outcome()
    )
    normal = out._replace(types=s.types)
    assert _lost(out, again) == _dmg(tables, out, names, "Flamethrower")
    assert _lost(out, again) < _dmg(tables, normal, names, "Flamethrower")
    stab = _dmg(tables, out, names, "Flamethrower", side=0)
    assert _lost(out, again, side=1) == stab > _dmg(tables, normal, names, "Flamethrower", side=0)


def test_double_kicks_second_hit_reads_the_type_its_first_hit_set(tables, names):
    """Manager order: hit 1 meets Normal (x2) and makes Kecleon Fighting; hit 2's own
    CHECK_TYPE_CHART (bcp.c:3272-3278 per hit, the loop re-runs TryMove: bcp.c:3780-3792) meets
    Fighting (x1)."""
    s = _kecleon(tables, names, ("Double Kick",))
    first = _dmg(tables, s, names, "Double Kick")
    fighting = s._replace(types=s.types.at[0, 0].set(Type.FIGHTING))
    second = _dmg(tables, fighting, names, "Double Kick")
    assert second < first
    out = assert_matches_reference(tables, s, A("Leer", "Double Kick", s, names), empty_outcome())
    assert _types(out) == (Type.FIGHTING, Type.FIGHTING)
    assert _lost(s, out) == first + second


@pytest.mark.parametrize(
    "case, move",
    [
        ("substitute", "Flamethrower"),  # :4209 returns on a substitute hit
        ("power 0", "Thunder Wave"),  # :4249 CURRENT_MOVE_DATA.power
        ("fainted", "Flamethrower"),  # :4243 DEFENDING_MON.curHP
        ("own type", "Tackle"),  # :4250-4251 Normal already
    ],
)
def test_color_change_does_nothing(tables, names, case, move):
    s = _kecleon(tables, names, (move,))
    if case == "substitute":
        s = s._replace(substitute_hp=s.substitute_hp.at[0].set(int(s.max_hp[0, 0]) // 4))
    if case == "fainted":
        s = s.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), empty_outcome())
    assert _types(out) == _types(s) and int(out.ai_ability_known[0]) == 0


def test_struggle_and_a_suppressed_holder_keep_the_type(tables, names):
    """:4245 moveCur != MOVE_STRUGGLE: a Fire Kecleon stays Fire. Gastro Acid makes
    Battler_Ability NONE (battle_lib.c:3088-3093): a Flamethrower changes nothing."""
    s = _kecleon(tables, names, ("Flamethrower", "Gastro Acid"))
    fire = assert_matches_reference(tables, s, A("Leer", "Flamethrower", s, names), empty_outcome())
    stuck = fire._replace(pp=fire.pp.at[1, 0].set(0))
    out = assert_matches_reference(tables, stuck, A("Leer", 0, stuck, names), empty_outcome())
    assert int(out.hp[0, 0]) < int(fire.hp[0, 0]) and _types(out) == (Type.FIRE, Type.FIRE)
    acid = assert_matches_reference(tables, s, A("Leer", "Gastro Acid", s, names), empty_outcome())
    out = assert_matches_reference(
        tables, acid, A("Leer", "Flamethrower", acid, names), empty_outcome()
    )
    assert _types(out) == _types(s) and int(out.ai_ability_known[0]) == 0


@pytest.mark.parametrize(
    "move, opp_species",
    [
        ("Flamethrower", "Cranidos"),  # Battler_Ability, not ignorable: Mold Breaker moot
        ("Seismic Toss", None),  # table power 1
        ("Hidden Power", None),  # battleCtx->moveType, battle_script.c:6018-6029
    ],
)
def test_color_change_reads_moveType_and_ignores_mold_breaker(tables, names, move, opp_species):
    s = _kecleon(tables, names, (move,), opp_species=opp_species)
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), empty_outcome())
    want = int(tables.move_type[names.move(move)])
    if move == "Hidden Power":
        want = int(hidden_power(s.ivs[1, 0])[0])
    assert _types(out) == (want, want)
    assert int(out.ai_ability_known[0]) == DECOMP_H3["Color Change"]


def test_a_switch_gives_kecleon_its_species_types_back(tables, names):
    """The send-out reloads the party types (BattleSystem_InitBattleMon)."""
    s = _kecleon(tables, names, ("Flamethrower", "Leer"))
    out = assert_matches_reference(tables, s, A("Leer", "Flamethrower", s, names), empty_outcome())
    out = assert_matches_reference(tables, out, A(SWITCH_TO_1, "Leer", out, names), empty_outcome())
    back = assert_matches_reference(
        tables, out, A(MOVE_ACTIONS, "Leer", out, names), empty_outcome()
    )
    assert int(back.active[0]) == 0 and _types(back) == _types(s)


def _milotic(tables, names, opp_moves, status=Status.NONE, counter=0, **kw):
    s = _holder(tables, names, "Milotic", "Marvel Scale", opp_moves, **kw)
    return s.with_status(0, 0, status, counter)


def _scaled(s, on=True):
    """``s`` with Milotic's ability gone and, where ``on``, its Defense stat x150/100: the
    damage CalcMoveDamage computes with the factor (battle_lib.c:6797-6800)."""
    out = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    d = Stat.DEF - 1
    if on:
        out = out._replace(stats=out.stats.at[0, 0, d].set(int(s.stats[0, 0, d]) * 150 // 100))
    return out


@pytest.mark.parametrize(
    "move, status, factor",
    [
        ("Tackle", Status.PARALYSIS, True),
        ("Tackle", Status.SLEEP, True),
        ("Tackle", Status.NONE, False),
        ("Ember", Status.PARALYSIS, False),  # Sp. Def never
    ],
)
def test_marvel_scale_raises_defense_of_a_statused_holder(tables, names, move, status, factor):
    s = _milotic(tables, names, (move,), status, counter=3 if status == Status.SLEEP else 0)
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), empty_outcome())
    assert _lost(s, out) == _dmg(tables, _scaled(s, factor), names, move)
    assert int(out.ai_ability_known[0]) == 0  # no message


def test_mold_breaker_and_gastro_acid_drop_marvel_scale(tables, names):
    """Battler_IgnorableAbility (battle_lib.c:3108-3121) and Gastro Acid's NONE (:3088-3093)."""
    s = _milotic(tables, names, ("Tackle",), Status.PARALYSIS, opp_species="Cranidos")
    out = assert_matches_reference(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert _lost(s, out) == _dmg(tables, _scaled(s, False), names, "Tackle")
    s = _milotic(tables, names, ("Gastro Acid", "Tackle"), Status.PARALYSIS)
    acid = assert_matches_reference(tables, s, A("Leer", "Gastro Acid", s, names), empty_outcome())
    out = assert_matches_reference(tables, acid, A("Leer", "Tackle", acid, names), empty_outcome())
    assert _lost(acid, out) == _dmg(tables, _scaled(acid, False), names, "Tackle")


def test_marvel_scale_survives_a_crit_and_comes_before_explosions_halving(tables, names):
    """A crit drops the holder's +2 Defense stage (battle_lib.c:6963-6973), not the stat factor;
    Explosion halves the scaled Defense (:6943-6945)."""
    s = _milotic(tables, names, ("Tackle", "Explosion"), Status.PARALYSIS)
    s = s.with_stages(0, def_=2)
    o = empty_outcome()._replace(crit=jnp.zeros((2, MAX_HITS), jnp.int8).at[1, 0].set(1))
    out = assert_matches_reference(tables, s, A("Leer", "Tackle", s, names), o)
    assert _lost(s, out) == _dmg(tables, _scaled(s), names, "Tackle", crit=True)
    assert _lost(s, out) < _dmg(tables, _scaled(s, False), names, "Tackle", crit=True)
    out = assert_matches_reference(tables, s, A("Leer", "Explosion", s, names), empty_outcome())
    hit = _dmg(tables, _scaled(s), names, "Explosion")
    assert _lost(s, out) == min(hit, int(s.hp[0, 0]))


def test_a_burned_holders_confusion_self_hit_reads_marvel_scale(tables, names):
    """CALC_SELF_HIT passes the battler as attacker and defender (include/battle/common.h:35-46,
    bcp.c:2632): Defense x1.5 and the burn's halving both apply; then the burn's 1/8 at the turn
    end."""
    s = _milotic(tables, names, ("Leer",), Status.BURN)
    s = s._replace(confusion=s.confusion.at[0].set(3))
    o = empty_outcome()._replace(confusion_hit=jnp.array([1, 0], jnp.int8))
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), o)
    hit = int(confusion_damage(tables, _scaled(s), 0, _ROLL))
    assert hit < int(confusion_damage(tables, _scaled(s, False), 0, _ROLL))
    assert _lost(s, out) == hit + max(int(s.max_hp[0, 0]) // 8, 1)


def test_twineedles_second_hit_reads_the_poison_its_first_set(tables, names):
    """The first hit's poison (MULTI_HIT_TRIGGER_SECONDARY, bcp.c:3555-3565) comes before the
    second hit's CalcMoveDamage: x1.5 on it; then the poison's 1/8 at the turn end."""
    s = _milotic(tables, names, ("Twineedle",))
    o = empty_outcome()._replace(secondary=jnp.zeros((2, MAX_HITS), jnp.int8).at[1, 0].set(1))
    out = assert_matches_reference(tables, s, A("Leer", "Twineedle", s, names), o)
    assert int(out.status[0, 0]) == Status.POISON
    first = _dmg(tables, s, names, "Twineedle")
    second = _dmg(tables, _scaled(s.with_status(0, 0, Status.POISON)), names, "Twineedle")
    assert second < first
    assert _lost(s, out) == first + second + max(int(s.max_hp[0, 0]) // 8, 1)


def test_the_ai_damage_estimate_reads_marvel_scale(tables, names):
    """TrainerAI_CalcDamage calls CalcMoveDamage with the real battlers (trainer_ai.c:3081-3091)."""
    s = _milotic(tables, names, ("Tackle",), Status.PARALYSIS)
    tackle = names.move("Tackle")
    ai = int(ai_damage_for_roll(tables, s, 1, tackle, _ROLL, def_side=0))
    assert ai == int(ai_damage_for_roll(tables, _scaled(s), 1, tackle, _ROLL, def_side=0))
    assert ai < int(ai_damage_for_roll(tables, _scaled(s, False), 1, tackle, _ROLL, def_side=0))


def _exploud(tables, names, opp_moves, **kw):
    return _holder(tables, names, "Exploud", "Soundproof", opp_moves, **kw)


@pytest.mark.parametrize("opp_species", [None, "Cranidos"])
def test_soundproof_blocks_hyper_voice_and_mold_breaker_pierces_it(tables, names, opp_species):
    s = _exploud(tables, names, ("Hyper Voice",), opp_species=opp_species)
    out = assert_matches_reference(tables, s, A("Leer", "Hyper Voice", s, names), empty_outcome())
    if opp_species is None:
        assert _lost(s, out) == 0 and int(out.ai_ability_known[0]) == DECOMP_H3["Soundproof"]
    else:
        assert _lost(s, out) > 0 and int(out.ai_ability_known[0]) == 0


def test_soundproof_reveals_on_a_missed_or_protected_sound_move(tables, names):
    """bcp.c:2791-2792: subscript_blocked_by_soundproof.s runs under MOVE_STATUS_DID_NOT_HIT."""
    s = _exploud(tables, names, ("Supersonic",), player_moves=("Protect", "Leer"))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = assert_matches_reference(tables, s, A("Leer", "Supersonic", s, names), miss)
    assert int(out.confusion[0]) == 0
    assert int(out.ai_ability_known[0]) == DECOMP_H3["Soundproof"]
    out = assert_matches_reference(tables, s, A("Protect", "Supersonic", s, names), empty_outcome())
    assert int(out.confusion[0]) == 0
    assert int(out.ai_ability_known[0]) == DECOMP_H3["Soundproof"]


@pytest.mark.parametrize("move", ["Roar", "Uproar"])
def test_soundproof_stops_roar_and_uproar(tables, names, move):
    """MOVE_STATUS_NO_MORE_WORK: no drag (effect_script_0028.s), no uproar (subscript_uproar.s)."""
    s = _exploud(tables, names, (move,))
    out = assert_matches_reference(tables, s, A("Leer", move, s, names), empty_outcome())
    assert int(out.active[0]) == 0 and int(out.uproar[1]) == 0 and int(out.locked_move[1]) == 0
    assert int(out.ai_ability_known[0]) == DECOMP_H3["Soundproof"]


def test_a_soundproof_holder_sleeps_through_its_own_uproar(tables, names):
    """Exploud's Uproar is up when Geodude's Hypnosis lands: its own Soundproof skips the uproar
    test (subscript_fall_asleep.s:44), the turn end's check spares it (bcp.c:1509-1511) and so
    does its next status check (bcp.c:2432-2433). Without the ability Hypnosis fails (:45)."""
    s = _exploud(tables, names, ("Hypnosis", "Leer"), player_moves=("Uproar", "Leer"))
    out = assert_matches_reference(tables, s, A("Uproar", "Hypnosis", s, names), empty_outcome())
    assert int(out.uproar[0]) > 0 and int(out.status[0, 0]) == Status.SLEEP
    counter = int(out.status_counter[0, 0])
    nxt = assert_matches_reference(tables, out, A("Uproar", "Leer", out, names), empty_outcome())
    assert int(nxt.status[0, 0]) == Status.SLEEP and int(nxt.status_counter[0, 0]) == counter - 1
    bare = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    out = assert_matches_reference(tables, bare, A("Uproar", "Hypnosis", bare, names),
                                   empty_outcome())  # fmt: skip
    assert int(out.status[0, 0]) == Status.NONE


@pytest.mark.parametrize(
    "user, opp_species, marked",
    [
        ("foe", None, (0, 1)),  # the holder skipped, no reveal (BTLSCR_MSG_TEMP)
        ("foe", "Cranidos", (1, 1)),  # the user's Mold Breaker pierces (:5474)
        ("holder", None, (0, 1)),  # the user skips itself
    ],
)
def test_perish_song_skips_a_soundproof_battler(tables, names, user, opp_species, marked):
    s = _exploud(tables, names, ("Perish Song", "Leer"), opp_species=opp_species,
                 player_moves=("Perish Song", "Leer"))  # fmt: skip
    moves = ("Perish Song", "Leer") if user == "holder" else ("Leer", "Perish Song")
    acts = A(*moves, s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert (int(out.perish_song[0]), int(out.perish_song[1])) == marked
    assert int(out.ai_ability_known[0]) == 0


# --- H-4: held items ---------------------------------------------------------------------------


def _item(s, names, item, side=0):
    """``side``'s active Pokemon holds ``item`` (a name)."""
    return s._replace(item=s.item.at[side, int(s.active[side])].set(names.item(item)))


def _speed(tables, s, side):
    from nuzline.data.tables import to_numpy
    from nuzline.engine.order import effective_speed
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _effective_speed

    got = int(effective_speed(tables, s, side))
    assert got == _effective_speed(to_numpy(tables), to_ref(s), side)
    return got


def test_quick_powder_doubles_a_dittos_speed_after_the_scarf_and_before_paralysis(tables, names):
    """battle_lib.c:1248-1261: Choice Scarf x15/10, Quick Powder x2 on a Ditto, then paralysis /4."""
    from nuzline.engine.ids import ItemId

    ditto = as_species(tables, roark_state(tables, names), 0, 0, names.species("Ditto"))
    base = _speed(tables, ditto, 0)
    powder = _item(ditto, names, "Quick Powder")
    assert int(tables.item_effect[names.item("Quick Powder")]) == ItemId.QUICK_POWDER
    assert _speed(tables, powder, 0) == base * 2
    para = powder._replace(status=powder.status.at[0, 0].set(Status.PARALYSIS))
    assert _speed(tables, para, 0) == base * 2 // 4
    # A non-Ditto holder, and a Klutz Ditto (Battler_HeldItemEffect), are unchanged.
    plain = roark_state(tables, names)
    assert _speed(tables, _item(plain, names, "Quick Powder"), 0) == _speed(tables, plain, 0)
    klutz = powder._replace(ability=powder.ability.at[0, 0].set(AbilityId.KLUTZ))
    assert _speed(tables, klutz, 0) == _speed(tables, klutz._replace(item=ditto.item), 0)


@pytest.mark.parametrize("item", ["Full Incense", "Lagging Tail"])
def test_full_incense_orders_its_holder_last_like_lagging_tail(tables, names, item):
    """battle_lib.c:1302-1304, :1412-1421: a lone PRIORITY_DOWN holder acts second (equal
    priority), a faster one too; the AI's speed compare (``battler_slower``) agrees."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.order import battler_slower
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _player_second

    s = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Tackle",))
    fast = s._replace(stats=s.stats.at[0, 0, Stat.SPE - 1].set(500))
    held = _item(fast, names, item)
    for state, second in ((fast, False), (held, True)):
        assert bool(battler_slower(tables, state, 0, 0, state.qc_roll)) is second
        ref = _player_second(to_numpy(tables), to_ref(state), (0, 0), 0, [0, 0])
        assert ref is second
    klutz = held._replace(ability=held.ability.at[0, 0].set(AbilityId.KLUTZ))
    assert not bool(battler_slower(tables, klutz, 0, 0, klutz.qc_roll))


def test_shed_shell_lets_a_trapped_holder_switch(tables, names):
    """battle_lib.c:5514-5520: Battler_HeldItemEffect == HOLD_EFFECT_SWITCH returns FALSE before
    Mean Look, Ingrain and the trapping abilities; Klutz hides it."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.actions import switch_targets, trapped
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _switch_targets, _trapped

    s = roark_state(tables, names)
    s = s._replace(mean_look=s.mean_look.at[0].set(1), ingrain=s.ingrain.at[0].set(1))
    held = _item(s, names, "Shed Shell")
    klutz = held._replace(ability=held.ability.at[0, 0].set(AbilityId.KLUTZ))
    for state, want in ((s, True), (held, False), (klutz, True)):
        assert bool(trapped(tables, state, 0)) is want
        assert _trapped(to_numpy(tables), to_ref(state), 0) is want
        assert bool(np.asarray(switch_targets(tables, state, 0)).any()) is (not want)
        assert (
            _switch_targets(to_numpy(tables), to_ref(state), 0)
            == np.asarray(switch_targets(tables, state, 0)).tolist()
        )


def _lovers(tables, names, foe_moves, *, player_item="", player_moves=("Leer",), **kw):
    """A female player lead against a male foe lead (the sample party's lead, Roark's Geodude)."""
    s = roark_state(
        tables, names, player_moves=player_moves, opp_moves=foe_moves,
        player_gender="female", opp_gender="male", **kw,
    )  # fmt: skip
    return _item(s, names, player_item) if player_item else s


def test_mental_herb_cures_infatuation_at_once_and_is_used_up(tables, names):
    """battle_lib.c:4889-4894 / :5102-5107 (VOLATILE_CONDITION_ATTRACT only), subscript_held_item_
    heal_infatuation.s:10-11: the foe's Attract lands, the holder is cured by the after-move checks
    and the item is spent (Recycle's slot), no AI reveal (BTLSCR_MSG_TEMP)."""
    herb = names.item("Mental Herb")
    s = _lovers(tables, names, ("Attract", "Leer"), player_item="Mental Herb")
    out = assert_matches_reference(tables, s, A("Leer", "Attract", s, names), empty_outcome())
    assert int(out.infatuated[0]) == 0 and int(out.item[0, 0]) == 0
    assert int(out.item_consumed[0, 0]) == herb and int(out.recycle_item[0]) == herb
    assert int(out.ai_item_known[0]) == 0
    # Not in love: the item is kept. Klutz: stays in love.
    kept = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(kept.item[0, 0]) == herb
    klutz = s._replace(ability=s.ability.at[0, 0].set(AbilityId.KLUTZ))
    out = assert_matches_reference(tables, klutz, A("Leer", "Attract", s, names), empty_outcome())
    assert int(out.infatuated[0]) != 0 and int(out.item[0, 0]) == herb


def test_destiny_knot_reflects_an_attract_and_the_ai_learns_the_item(tables, names):
    """battle_lib.c:4546-4552: the defender holds the knot, is sideEffectMon and flagged
    INFATUATED (subscript_infatuate.s:29): the attacker falls in love too (:37-41 tags the
    holder's item). Same gender: Attract itself fails and nothing changes; Klutz hides the knot."""
    knot = names.item("Destiny Knot")
    s = _lovers(tables, names, ("Attract",), player_item="Destiny Knot")
    out = assert_matches_reference(tables, s, A("Leer", "Attract", s, names), empty_outcome())
    assert int(out.infatuated[0]) != 0 and int(out.infatuated[1]) != 0
    assert int(out.ai_item_known[0]) == knot
    male = s._replace(gender=s.gender.at[0, 0].set(0))
    out = assert_matches_reference(tables, male, A("Leer", "Attract", male, names), empty_outcome())
    assert np.asarray(out.infatuated).tolist() == [0, 0] and int(out.ai_item_known[0]) == 0
    klutz = s._replace(ability=s.ability.at[0, 0].set(AbilityId.KLUTZ))
    out = assert_matches_reference(tables, klutz, A("Leer", "Attract", s, names), empty_outcome())
    assert int(out.infatuated[1]) == 0 and int(out.infatuated[0]) != 0


def _charmed(tables, names):
    """The knot-holding female player's Tackle (contact) into a male foe with Cute Charm."""
    s = _lovers(tables, names, ("Leer",), player_item="Destiny Knot", player_moves=("Tackle",))
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))
    return s._replace(stats=s.stats.at[0, 0, Stat.SPE - 1].set(500))  # the holder hits first


def test_destiny_knot_reflects_a_cute_charm_after_the_holders_hit(tables, names):
    """The attacker holds the knot: battle_lib.c:4553-4559 (sideEffectMon was moved to it by Cute
    Charm, :4344-4346). The foe falls in love too."""
    s = _charmed(tables, names)
    out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), _procs(opp=1))
    assert np.asarray(out.infatuated).tolist() == [1, 1]
    assert int(out.ai_item_known[0]) == names.item("Destiny Knot")
    # An Oblivious attacker is not charmed, so no knot flag exists.
    oblivious = s._replace(ability=s.ability.at[0, 0].set(AbilityId.OBLIVIOUS))
    out = assert_matches_reference(tables, oblivious, A("Tackle", "Leer", s, names), _procs(opp=1))
    assert np.asarray(out.infatuated).tolist() == [0, 0]


def test_a_cute_charm_that_cannot_land_raises_no_flag_and_the_engines_agree(tables, names):
    """battle_lib.c:4344-4346 moves sideEffectMon to the attacker before TryAttract, which fails
    here (both female): no INFATUATED flag (subscript_infatuate.s:29 is on the landing path), so the
    knot stays quiet. The moved sideEffectMon itself is read only by the Synchronize step, and only
    with SYSCTL_TRY_SYNCHRONIZE_STATUS set (battle_lib.c:4499-4511), which a status landing in this
    action sets (``effects.sync_landed``). Singles cannot tell the move apart: the defender's
    ability is Cute Charm, so the one status that can land before the step is a secondary on the
    defender, which then cannot take the attacker's passed status (``can_have_status``); with no
    status landed the flag is clear. Engine == reference with a Synchronize attacker holding its
    own burn (the step's attacker branch runs on the moved value and lands nothing)."""
    s = _charmed(tables, names)
    s = s._replace(gender=s.gender.at[1, 0].set(1))
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.SYNCHRONIZE),
        status=s.status.at[0, 0].set(Status.BURN),
    )
    out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), _procs(opp=1))
    assert np.asarray(out.infatuated).tolist() == [0, 0]


def test_a_knot_whose_reflection_fails_marks_the_move_failed(tables, names):
    """subscript_infatuate.s:6-7, :108-110: the reflected TryAttract fails on a receiver already in
    love (battle_script.c:5733) and sets MOVE_STATUS_FAILED (type HELD_ITEM, not ABILITY), one of
    MOVE_STATUS_NO_EFFECTS (moves.h:85). UpdateFlagsWhenHit then clears the defender's Conversion 2
    record (bcp.c:4511-4512, :4528-4531) and VerifyMetronomeCount takes the Metronome item's count
    back (battle_lib.c:6183-6192). The foe, already in love with the holder, uses Attract."""
    attract = names.move("Attract")
    s = _lovers(tables, names, ("Attract",), player_item="Destiny Knot")
    s = s._replace(item=s.item.at[1, 0].set(names.item("Metronome")))
    s = s._replace(
        metronome_move=s.metronome_move.at[1].set(attract),
        metronome_turns=s.metronome_turns.at[1].set(2),
    )
    acts = A("Leer", "Attract", s, names)
    ok = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(ok.infatuated[0]) != 0 and int(ok.infatuated[1]) != 0
    assert int(ok.conv2_move[0]) == attract
    loving = s._replace(infatuated=s.infatuated.at[1].set(1))
    out = assert_matches_reference(tables, loving, acts, empty_outcome())
    assert int(out.infatuated[0]) != 0
    assert int(out.conv2_move[0]) == 0 and int(out.conv2_battler[0]) == 0
    assert int(out.metronome_turns[1]) == int(ok.metronome_turns[1]) - 1


def test_the_resumed_u_turn_never_reads_its_users_cute_charm_flag(tables, names):
    """subscript_attack_then_switch_out.s:7 runs the foe's Cute Charm on the U-turn user
    (sideEffectMon = attacker, battle_lib.c:4351; subscript_infatuate.s:29 flags it), then :70 sets
    BTLVAR_ATTACKER_SELF_TURN_STATUS_FLAGS (selfTurnFlags[attacker], battle_script.c:9623-9624) to
    SELF_TURN_FLAG_CLEAR before the list: the resumed AfterMoveEffects finds the newcomer's knot
    with no INFATUATED flag (battle_lib.c:4553-4556), so the foe stays out of love, no reveal."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon, _resume

    n = names
    lead = replace(_mon(n, "Scyther", ["U-turn"], 30), gender=1)
    bench = replace(_mon(n, "Geodude", ["Tackle"], 30, item="Destiny Knot"), gender=1)
    foe = replace(_mon(n, "Bulbasaur", ["Leer"], 35), gender=0)
    s = make_state(tables, [lead, bench], [foe])
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))
    s = s._replace(stats=s.stats.at[0, 0, Stat.SPE - 1].set(500))
    paused = assert_matches_reference(tables, s, (0, 0), _procs(opp=1))
    assert int(paused.pivot_battler) == 1 and np.asarray(paused.ts_loved).tolist() == [0, 0]
    out = _resume(tables, paused, 1, _procs(opp=1))
    assert int(out.active[0]) == 1 and int(out.infatuated[1]) == 0
    assert int(out.ai_item_known[1]) == 0


def _pursued_u_turn(tables, names, bench="Geodude", **start):
    """The foe's female Cute Charm Scyther U-turns (``bench`` behind it, female) into the player's
    slower male Steelix holding a Destiny Knot, which pursues (Steel / Ground: the U-turn does not
    KO it, so CheckBlackOut at subscript_attack_then_switch_out.s:5 lets the list come)."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    n = names
    me = replace(_mon(n, "Steelix", ["Pursuit"], 40, item="Destiny Knot"), gender=0)
    lead = replace(_mon(n, "Scyther", ["U-turn"], 50), gender=1)
    back = replace(_mon(n, bench, ["Tackle"], 30), gender=1)
    s = make_state(tables, [me], [lead, back], **start)
    return s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))


@pytest.mark.parametrize(
    "case, reflects",
    [
        ("plain", True),
        # The list's CheckToxicSpikes makes the newcomer the sideEffectMon (battle_script.c:
        # 7638-7641) before AfterMoveEffects: the defender branch's test fails.
        ("toxic spikes", False),
        # A Castform newcomer in rain: TriggerFormChange (battle_lib.c:6314-6340) returns first
        # (:4536-4544), before the switch-in effects (bcp.c:3672): the knot is lost.
        ("castform", False),
        # H-5: the Cherrim block (battle_lib.c:6366-6389) is pending in sun (party form 0), not in
        # rain (Overcast already), and not under an alive Cloud Nine (target Overcast).
        ("cherrim sun", False),
        ("cherrim rain", True),
        ("cherrim sun cloud nine", True),
    ],
)
def test_the_pursuers_cute_charm_flag_reaches_the_resumed_knot(tables, names, case, reflects):
    """subscript_attack_then_switch_out.s:54 calls subscript_pursuit.s before the :70 clear,
    which is the attacker's slot only: the Pursuit's contact runs the U-turn user's Cute Charm on
    the pursuer (:40; SYSCTL_UTURN_ACTIVE is set only at :68), making it the sideEffectMon
    (battle_lib.c:4351) and flagging it (subscript_infatuate.s:29). The resumed AfterMoveEffects
    then takes the knot's defender branch (battle_lib.c:4546-4552): the newcomer falls in love with
    the pursuer and the AI learns the item (subscript_infatuate.s:37-41)."""
    from tests.engine.test_pivot import _resume

    if case == "castform":
        s = _pursued_u_turn(tables, names, "Castform", start_condition=Weather.RAIN)
    elif case.startswith("cherrim"):
        weather = Weather.RAIN if case == "cherrim rain" else Weather.SUN
        s = _pursued_u_turn(tables, names, "Cherrim", start_condition=weather)
        if case.endswith("cloud nine"):
            s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.CLOUD_NINE))
    else:
        s = _pursued_u_turn(tables, names)
    if case == "toxic spikes":
        s = s._replace(toxic_spikes=s.toxic_spikes.at[1].set(1))
    paused = assert_matches_reference(tables, s, (0, 0), _procs(opp=1))
    assert int(paused.pivot_battler) == 2 and int(paused.ts_pursued) == 1
    assert np.asarray(paused.ts_loved).tolist() == [1, 0] and int(paused.infatuated[0]) != 0
    out = _resume(tables, paused, 1, _procs(opp=1))
    assert int(out.active[1]) == 1
    assert (int(out.infatuated[1]) != 0) is reflects
    knot = names.item("Destiny Knot")
    assert int(out.ai_item_known[0]) == (knot if reflects else 0)
    if case.startswith("cherrim"):  # the switch-in check ran after the lost knot (bcp.c:3672)
        assert int(out.form_num[1]) == int(case == "cherrim sun")


def test_a_u_turn_user_felled_by_the_pursuit_keeps_its_flag_for_its_knot(tables, names):
    """subscript_attack_then_switch_out.s:55 ends the script when the Pursuit felled the user,
    before the :70 clear: AfterMoveEffects reads the user's INFATUATED flag from the foe's Cute
    Charm (:7) and its knot (battle_lib.c:4553-4559) infatuates the pursuer, the message tagging
    the item (subscript_infatuate.s:37-41). The love toward the fainted holder is cleared again
    at its faint (battle_lib.c:2175-2176); the reveal stays."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    n = names
    lead = replace(_mon(n, "Scyther", ["U-turn"], 30, item="Destiny Knot"), gender=1)
    bench = replace(_mon(n, "Geodude", ["Tackle"], 30), gender=1)
    foe = replace(_mon(n, "Steelix", ["Pursuit"], 40), gender=0)  # slower, survives the U-turn
    s = make_state(tables, [lead, bench], [foe]).replace_hp(0, 0, 1)
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))
    out = assert_matches_reference(tables, s, (0, 0), _procs(opp=1))
    assert int(out.hp[0, 0]) == 0
    assert int(out.ai_item_known[0]) == n.item("Destiny Knot")


def _chased_charm(tables, names, genders, holder, target):
    """H-M1 (B = 4): battler 2 (female, Cute Charm) U-turns at ``target``; battlers 1 and 3 chose
    Pursuit and both chase it before the list (index order, subscript_pursuit.s:79-82), each hit
    passing the Cute Charm trigger (``contact_proc`` of battler 2). ``genders``: battlers 1 and 3
    (0 male: charmed; 1 female: the trigger passes, TryAttract fails). ``holder`` holds a Destiny
    Knot; the newcomer (battler 2's party slot 2) is female."""
    from tests.engine.d2_states import party_of, with_ability
    from tests.engine.test_d3_pause import pivoting

    s = with_moves(pivoting(tables, names), names, 0, ["Swords Dance"])
    for b in (1, 3):
        s = with_moves(s, names, b, ["Pursuit"])
    s = with_moves(s, names, 2, ["U-turn"])
    s = with_ability(s, 2, AbilityId.CUTE_CHARM)
    p2 = party_of(s, 2)
    gender = s.gender.at[p2, int(s.active[2])].set(1).at[p2, 2].set(1)
    for b, g in zip((1, 3), genders, strict=True):
        gender = gender.at[party_of(s, b), int(s.active[b])].set(g)
    s = s._replace(gender=gender)
    knot = names.item("Destiny Knot")
    s = s._replace(item=s.item.at[party_of(s, holder), int(s.active[holder])].set(knot))
    procs = jnp.zeros((4, MAX_HITS), jnp.int8).at[2].set(1)
    from nuzline.engine.format import DOUBLES

    outcome = empty_outcome(DOUBLES)._replace(contact_proc=procs)
    return s, outcome, (0, 0, target, 0)


@pytest.mark.slow  # two B = 4 action phases, each engine and reference (over 60 s)
@pytest.mark.parametrize(
    "genders, holder, target, loved, reflects",
    [
        # Battler 3's trigger (battle_lib.c:4350) moved sideEffectMon off charmed battler 1, the
        # defender holding the knot: the knot's defender test fails (battle_lib.c:4547-4549).
        ((0, 1), 1, 1, [0, 2, 0, 0], False),
        ((0, 0), 1, 1, [0, 2, 0, 1], False),
        # Both charmed: the last, battler 3, is the sideEffectMon and the defender with the knot.
        ((0, 0), 3, 3, [0, 2, 0, 1], True),
        # Only battler 3 charmed (battler 1's failed TryAttract moved sideEffectMon first).
        ((1, 0), 3, 3, [0, 0, 0, 1], True),
    ],
)
def test_the_resumed_knot_reads_the_last_pursuers_cute_charm_trigger(
    tables, names, genders, holder, target, loved, reflects
):
    """H-M1: subscript_pursuit.s:40 runs the U-turn user's Cute Charm on each pursuer's hit, and
    every trigger sets sideEffectMon = attacker (battle_lib.c:4339-4350) whether or not
    subscript_infatuate.s's TryAttract lands (:7, battle_script.c:5732-5735); no other write
    comes before the list (subscript_attack_then_switch_out.s:52-69). The resumed AfterMoveEffects'
    knot step (battle_lib.c:4546-4552) needs the defender to be that sideEffectMon and flagged
    INFATUATED (subscript_infatuate.s:29): ``ts_loved`` 1 marks the flagged sideEffectMon, 2 a
    flagged pursuer a later trigger superseded. Engine == reference on both halves."""
    from tests.engine.test_d3_action_loop import play
    from tests.engine.test_d3_pause import resume

    s, outcome, targets = _chased_charm(tables, names, genders, holder, target)
    paused, _ = play(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), targets=targets, outcome=outcome)
    assert int(paused.pivot_battler) == 3 and int(paused.ts_pursued) == 0b1010
    assert np.asarray(paused.ts_loved).tolist() == loved
    out = resume(tables, paused, outcome=outcome)
    assert (int(out.infatuated[2]) != 0) is reflects
    assert not np.asarray(out.ts_loved).any()


@pytest.mark.slow  # a B = 4 action phase, engine and reference (over 60 s)
def test_a_u_turn_user_felled_by_a_later_pursuer_keeps_the_charmed_one_as_side_effect_mon(
    tables, names
):
    """H-M1, the MODE_PURSUIT tail (subscript_attack_then_switch_out.s:55): the U-turn's contact
    meets battler 1's Static (battle_lib.c:4214-4224: sideEffectMon = the user, which is paralyzed),
    battler 1's Pursuit hit then passes the user's Cute Charm trigger and is charmed (:4350,
    sideEffectMon = 1), battler 3's hit fells the user, and a hit that faints the defender passes no
    Cute Charm trigger (DEFENDING_MON.curHP, :4347): the sideEffectMon stays battler 1 at
    AfterMoveEffects, so the knot of the defender (battler 1, flagged) reflects
    (battle_lib.c:4546-4552) onto the felled user. Engine == reference."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.outcome import outcome_to_py
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_step
    from tests.engine.d2_states import party_of, with_ability, with_hp
    from tests.engine.test_d3_action_loop import hp, play, speed_ordered

    s, outcome, targets = _chased_charm(tables, names, (0, 0), 1, 1)
    s = speed_ordered(with_ability(s, 1, AbilityId.STATIC), [2, 0, 1, 3])
    outcome = outcome._replace(contact_proc=outcome.contact_proc.at[1].set(1))
    # The reference alone measures the two Pursuit hits (equal Bulbasaurs, no crit: equal damage).
    probe = from_ref(
        ref_step(
            to_numpy(tables), to_ref(s), (0, 0, 0, 0), outcome_to_py(outcome), list(targets),
            actions_only=True,
        )
    )  # fmt: skip
    both = int(s.hp[party_of(s, 2), int(s.active[2])]) - hp(probe, 2)
    s = with_hp(s, 2, both // 2 + 1)  # the first hit leaves it, the second fells it
    out, _ = play(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), targets=targets, outcome=outcome)
    assert hp(out, 2) == 0 and int(out.pivot_battler) == 0
    assert int(probe.status[party_of(s, 2), int(s.active[2])]) == Status.PARALYSIS
    assert int(out.infatuated[1]) != 0 and int(out.infatuated[2]) != 0, np.asarray(out.infatuated)


@pytest.mark.slow  # a singles step, engine and reference (52 s, peak 9.9 GB locally)
def test_singles_a_blocked_poison_point_trigger_on_the_felling_pursuit_stops_the_knot(
    tables, names
):
    """H-M2, B = 2, the MODE_PURSUIT tail (subscript_attack_then_switch_out.s:55): the knot-holding
    Poison Point U-turn user is charmed by the foe's Cute Charm on its hit (:7; sideEffectMon = the
    user, battle_lib.c:4350; subscript_infatuate.s:29 flags it). The foe's Pursuit fells it, and
    Poison Point, with no defender-HP test (:4303-4313; only Cute Charm has one, :4347), passes its
    trigger on that hit (subscript_pursuit.s:84-87): sideEffectMon = the foe, whose Steel type then
    stops the poison (subscript_poison.s:34-37). AfterMoveEffects' knot attacker branch
    (battle_lib.c:4553-4558) needs the attacker to be the sideEffectMon: no reflection, no item
    record. The old code kept sideEffectMon = the user and reflected. Engine == reference."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    n = names
    lead = replace(_mon(n, "Scyther", ["U-turn"], 30, item="Destiny Knot"), gender=1)
    bench = replace(_mon(n, "Geodude", ["Tackle"], 30), gender=1)
    foe = replace(_mon(n, "Steelix", ["Pursuit"], 40), gender=0)  # slower, survives the U-turn
    s = make_state(tables, [lead, bench], [foe]).replace_hp(0, 0, 1)
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CUTE_CHARM))
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.POISON_POINT))
    out = assert_matches_reference(tables, s, (0, 0), _procs(player=1, opp=1))
    assert int(out.hp[0, 0]) == 0 and int(out.status[1, 0]) == Status.NONE
    assert int(out.infatuated[1]) == 0 and int(out.ai_item_known[0]) == 0


@pytest.mark.slow  # a B = 4 action phase, engine and reference (over 60 s)
def test_a_blocked_flame_body_trigger_moves_side_effect_mon_off_a_synchronize_pursuer(
    tables, names
):
    """H-M2, B = 4, the MODE_PURSUIT tail (subscript_attack_then_switch_out.s:55): foe battler 1
    (Flame Body) U-turns at battler 0. Battler 0's Pursuit hit is burned by the Flame Body
    (sideEffectMon = 0, battle_lib.c:4331; SYSCTL_TRY_SYNCHRONIZE_STATUS on, subscript_burn.s:79-84);
    battler 2's (Water Veil) hit fells the user and passes the trigger too, so sideEffectMon = 2
    although subscript_burn.s:20-23 stops the burn. At AfterMoveEffects the defender, battler 0's
    Synchronize, is not the sideEffectMon (battle_lib.c:4499-4502): no burn passed to the felled
    user, no Synchronize message (subscript_burn.s:68, the AI's record). Before H-M2 both engines
    kept sideEffectMon = 0 and passed it. Engine == reference."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.format import DOUBLES
    from nuzline.engine.outcome import outcome_to_py
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_step
    from tests.engine.d2_states import party_of, with_ability, with_hp
    from tests.engine.test_d3_action_loop import hp, play, speed_ordered
    from tests.engine.test_d3_pause import base

    s = with_moves(base(tables, names), names, 1, ["U-turn"])
    for b in (0, 2):
        s = with_moves(s, names, b, ["Pursuit"])
    s = with_ability(with_ability(s, 1, AbilityId.FLAME_BODY), 0, AbilityId.SYNCHRONIZE)
    s = with_ability(s, 2, AbilityId.WATER_VEIL)
    s = speed_ordered(s, [1, 0, 2, 3])
    outcome = empty_outcome(DOUBLES)._replace(
        contact_proc=jnp.zeros((4, MAX_HITS), jnp.int8).at[1].set(1)
    )
    targets = (1, 0, 1, 0)
    probe = from_ref(
        ref_step(
            to_numpy(tables), to_ref(s), (0, 0, 0, 0), outcome_to_py(outcome), list(targets),
            actions_only=True,
        )
    )  # fmt: skip
    both = int(s.hp[party_of(s, 1), int(s.active[1])]) - hp(probe, 1)
    s = with_hp(s, 1, both // 2 + 1)  # the first hit leaves it, the second fells it
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=targets, outcome=outcome)
    assert hp(out, 1) == 0 and int(out.pivot_battler) == 0
    assert int(out.status[0, int(s.active[0])]) == Status.BURN
    assert int(out.ai_ability_known[int(s.active[0])]) == 0


def test_the_h4_items_have_their_own_ids_and_the_inert_ones_one_shared_set(tables, names):
    from nuzline.engine.coverage import IMPLEMENTED_ITEMS
    from nuzline.engine.ids import NO_IN_BATTLE_HOLD_EFFECT_ITEMS, ItemId

    expect = {
        "Smoke Ball": ItemId.SMOKE_BALL,
        "Exp. Share": ItemId.EXP_SHARE,
        "Soothe Bell": ItemId.SOOTHE_BELL,
        "Lucky Egg": ItemId.LUCKY_EGG,
        "Quick Powder": ItemId.QUICK_POWDER,
        "Full Incense": ItemId.FULL_INCENSE,
        "Destiny Knot": ItemId.DESTINY_KNOT,
        "Shed Shell": ItemId.SHED_SHELL,
        "Mental Herb": ItemId.MENTAL_HERB,
    }
    for name, item in expect.items():
        assert int(tables.item_effect[names.item(name)]) == item, name
        assert item in IMPLEMENTED_ITEMS
    assert set(NO_IN_BATTLE_HOLD_EFFECT_ITEMS) == {
        ItemId.SMOKE_BALL,
        ItemId.EXP_SHARE,
        ItemId.SOOTHE_BELL,
        ItemId.LUCKY_EGG,
    }


# --- H-5: BattleSystem_TriggerFormChange's Cherrim, Arceus and Transformed Giratina blocks ----


def _cherrim_state(tables, names, weather, timer=0, form_num=0):
    """Monferno protects; the foe's active slot is a Cherrim (no ability test: lib.c:6366-6368)
    showing ``form_num``, under ``weather``."""
    from tests.engine.test_e1_batch_f import _become, _weather
    from tests.engine.test_protect import _state

    s = _state(tables, names, mine=["Protect"])
    s = _weather(_become(tables, names, s, 1, "Cherrim"), weather, timer)
    return s._replace(form_num=s.form_num.at[1].set(form_num))


@pytest.mark.parametrize(
    ("weather", "timer", "shown", "expect"),
    [
        (Weather.SUN, 0, 0, 1),  # :6375-6379 sun and Overcast -> Sunshine
        (Weather.SUN, 0, 1, 1),  # already Sunshine: no change
        (Weather.SUN, 1, 1, 0),  # the sun ends at the turn end: :6369-6374, the next turn start
        (Weather.RAIN, 0, 1, 0),  # :6380-6384
        (Weather.HAIL, 0, 1, 0),  # :6385-6389
        (Weather.SANDSTORM, 0, 1, 0),  # FIELD_CONDITION_CASTFORM has no sandstorm bit
        (Weather.NONE, 0, 1, 0),
    ],
)
def test_cherrim_shows_sunshine_in_sun_and_overcast_otherwise(
    tables, names, weather, timer, shown, expect
):
    """battle_lib.c:6366-6389: ``BattleMon.formNum`` only (CHERRIM_FORM_SUNSHINE 1 / OVERCAST 0);
    the party's form is not written (no SYSCTL_FORM_CHANGE)."""
    s = _cherrim_state(tables, names, weather, timer, shown)
    out = assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), empty_outcome())
    assert int(out.form_num[1]) == expect
    assert int(out.form[1, 0]) == int(s.form[1, 0])


def test_an_alive_cloud_nine_turns_a_sunshine_cherrim_back_to_overcast(tables, names):
    """NO_CLOUD_NINE fails (include/battle/common.h:63-64): formNum 1 -> OVERCAST (:6391-6395),
    sun or not."""
    s = _cherrim_state(tables, names, Weather.SUN, form_num=1)
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.CLOUD_NINE))
    out = assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), empty_outcome())
    assert int(out.form_num[1]) == 0


def test_sunny_day_gives_a_cherrim_sunshine_at_the_move_end_and_the_lead_at_battle_start(
    tables, names
):
    """MoveEnd's TriggerEffectOnSwitch (bcp.c:3973) and the battle start's ShowBattleMon
    (bcp.c:256) both run SWITCH_IN_CHECK_STATE_FORM_CHANGE (battle_lib.c:4115)."""
    from nuzline.engine.hooks import battle_start

    s = _cherrim_state(tables, names, Weather.NONE)
    s = s._replace(moves=s.moves.at[0, 0, 0].set(names.move("Sunny Day")))
    out = assert_matches_reference(tables, s, A(0, "Tackle", s, names), empty_outcome())
    assert int(weather_of(out)) == Weather.SUN and int(out.form_num[1]) == 1
    sunny = _cherrim_state(tables, names, Weather.SUN)
    assert int(battle_start(tables, sunny).form_num[1]) == 1


def test_a_switched_in_cherrim_takes_its_party_form_then_its_check(tables, names):
    """BattleSystem_InitBattleMon reads the party's form (battle_lib.c:132): the outgoing
    battler's formNum is gone at the send-out, and the check that follows sets Sunshine in sun."""
    from nuzline.engine.field import with_weather
    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    mine = _mon(names, "Monferno", ["Protect"], 40)
    foe = [_mon(names, "Geodude", ["Tackle"], 30), _mon(names, "Cherrim", ["Tackle"], 30)]
    for weather, expect in ((Weather.SUN, 1), (Weather.NONE, 0)):
        s = with_weather(make_state(tables, [mine], foe), weather, 0)
        s = s._replace(form_num=s.form_num.at[1].set(1))  # the outgoing Geodude's, not the party's
        out = assert_matches_reference(tables, s, (0, MOVE_ACTIONS + 1), empty_outcome())
        assert int(out.active[1]) == 1 and int(out.form_num[1]) == expect


def _arceus(names, plate="Flame Plate"):
    from dataclasses import replace

    from tests.engine.test_pivot import _mon

    return replace(_mon(names, "Arceus", ["Protect"], 50, item=plate), gender=2)


def test_a_lead_arceus_takes_its_plates_form_at_battle_start(tables, names):
    """battle_lib.c:6398-6410: formNum := Pokemon_GetArceusTypeOf(plate) (pokemon.c:4319-4377;
    Flame Plate: TYPE_FIRE 10) when it differs from the party's form (0: a trainer Pokemon gets
    no Pokemon_SetArceusForm call); the party's form stays."""
    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    foe = [_mon(names, "Geodude", ["Tackle"], 30)]
    s = make_state(tables, [_arceus(names)], foe)
    assert int(s.form_num[0]) == Type.FIRE == 10 and int(s.form[0, 0]) == 0
    assert int(make_state(tables, [_arceus(names, "Sitrus Berry")], foe).form_num[0]) == 0


def test_a_pending_form_change_is_each_blocks_own_test_in_both_engines(tables, names):
    """``effects.form_change_pending`` / the reference ``_form_pending``: an Arceus whose formNum
    differs from its plate's type (H-4 ruling 1's skip), a Cherrim in sun showing Overcast; none
    once settled."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.effects import form_change_pending
    from nuzline.engine.state import make_state
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _form_pending
    from tests.engine.test_pivot import _mon

    foe = [_mon(names, "Geodude", ["Tackle"], 30)]
    arceus = make_state(tables, [_arceus(names)], foe)
    cherrim = _cherrim_state(tables, names, Weather.SUN)
    cases = [
        (arceus, False),  # the battle start settled it
        (arceus._replace(form_num=arceus.form_num.at[0].set(0)), True),
        (cherrim, True),
        (cherrim._replace(form_num=cherrim.form_num.at[1].set(1)), False),
    ]
    for state, pending in cases:
        assert bool(form_change_pending(tables, state)) is pending
        assert _form_pending(to_numpy(tables), to_ref(state)) is pending


def test_a_pending_arceus_form_loses_the_knot_a_settled_one_keeps_it(tables, names):
    """sync_step (battle_lib.c:4536-4544): the pending form change returns before the Destiny
    Knot, so a reflection that would fail on a genderless Arceus (TryAttract, battle_script.c:
    5727-5742: MOVE_STATUS_FAILED) never runs."""
    from dataclasses import replace

    from nuzline.engine.effects import sync_step
    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    knot = replace(_mon(names, "Monferno", ["Tackle"], 40, item="Destiny Knot"), gender=0)
    s = make_state(tables, [knot], [_arceus(names)])
    loved = jnp.asarray([True, False])
    for shown, failed in ((0, False), (int(Type.FIRE), True)):
        s2 = s._replace(form_num=s.form_num.at[1].set(shown))
        got = sync_step(tables, s2, 0, 1, False, 0, loved, True)[-1]
        assert bool(got) is failed


def test_a_transform_copy_of_an_arceus_takes_its_own_plates_form(tables, names):
    """Transform copies formNum with the BattleMon bytes up to ``ability`` (battle_script.c:
    5346-5348, battle_mon.h:57-84) but not the item: a Ditto holding a Spooky Plate that copies a
    Flame Plate Arceus keeps Multitype and changes to its own plate's form (Ghost 7) at the
    check after the move (:6398-6410); its types are Ghost already (F-3)."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    ditto = replace(_mon(names, "Ditto", ["Transform"], 50, item="Spooky Plate"), gender=2)
    s = make_state(tables, [ditto], [_arceus(names)])
    out = assert_matches_reference(tables, s, A("Transform", "Protect", s, names), empty_outcome())
    assert int(out.species[0, 0]) == int(out.species[1, 0]) and int(out.form_num[0]) == Type.GHOST
    assert int(out.form_num[1]) == Type.FIRE and out.types[0, 0].tolist() == [Type.GHOST] * 2


# Peak 8.51 GB with a cold step compile (above the fast tier's ~8.3 GB): slow tier.
@pytest.mark.slow
def test_a_transformed_giratina_origin_is_rebuilt_as_altered_with_pressure(tables, names):
    """battle_lib.c:6412-6449: a Ditto that copied an Origin (formNum 1, the Orb stays with the
    foe) takes Altered's stats from the foe's party Pokemon (IVs, EVs, level, nature), ability
    Pressure (Altered's slot 0), formNum 0 and its slot's form 0; the party's form (``shadow_form``)
    is written 0 (SYSCTL_FORM_CHANGE, battle_io_command.c:597-599); types, stages and HP stay. The
    rebuild runs at SYNCHRONIZE_STATUS (battle_lib.c:4536-4544), so the switch-in checks after it
    reach the PRESSURE state (:4096-4110; Transform cleared pressureAnnounced, battle_script.c:5358)
    and the AI learns Pressure (subscript_pressure.s, BattleAI_SetAbility, battle_script.c:11806).
    The switch-out restore is G-69's (``_restore_transform``)."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from nuzline.engine.stats import calc_stats_jnp
    from tests.engine.test_pivot import _mon

    n = names
    ditto = _mon(n, "Ditto", ["Transform"], 50)
    bench = _mon(n, "Geodude", ["Tackle"], 30)
    giratina = replace(_mon(n, "Giratina", ["Protect"], 50, item="Griseous Orb"), form=1)
    s = make_state(tables, [ditto, bench], [giratina])
    assert int(s.form_num[1]) == 1
    out = assert_matches_reference(tables, s, A("Transform", "Protect", s, n), empty_outcome())
    altered = int(tables.species_form_row[n.species("Giratina"), 0])
    expect = calc_stats_jnp(
        tables.species_base_stats[altered],
        s.ivs[1, 0],
        s.evs[1, 0],
        s.level[1, 0],
        tables.nature_mods[s.nature[1, 0]],
        n.species("Giratina"),
    )
    assert out.stats[0, 0].tolist() == expect[1:].tolist()
    assert out.stats[0, 0].tolist() != out.stats[1, 0].tolist()  # Origin's, which it copied
    assert int(out.ability[0, 0]) == int(AbilityId.PRESSURE)
    assert (int(out.form_num[0]), int(out.form[0, 0]), int(out.shadow_form[0])) == (0, 0, 0)
    assert int(out.form_num[1]) == 1 and int(out.species[0, 0]) == n.species("Giratina")
    assert out.types[0, 0].tolist() == out.types[1, 0].tolist()
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert int(out.ai_ability_known[0]) == ABILITY_DECOMP_IDS[AbilityId.PRESSURE]


# A step compile like the test above (peak above the fast tier's ~8.3 GB): slow tier.
@pytest.mark.slow
def test_knock_off_leaves_giratina_origin_its_griseous_orb_and_its_form(tables, names):
    """H-5 review fix: subscript_knock_off.s:7 ends the script on a held Griseous Orb before
    TryKnockOff (battle_script.c:6608-6630) runs, so the Origin keeps the Orb and never reaches
    the no-Orb revert (battle_lib.c:6412-6418): item, slot mark, ``form_num`` Origin, its stats
    and Levitate all stay. Both engines."""
    from dataclasses import replace

    from nuzline.engine.state import make_state
    from tests.engine.test_pivot import _mon

    n = names
    origin = replace(_mon(n, "Giratina", ["Growl"], 50, item="Griseous Orb"), form=1)
    s = make_state(tables, [_mon(n, "Sneasel", ["Knock Off"], 50)], [origin])
    assert int(s.form_num[1]) == 1 and int(s.ability[1, 0]) == int(AbilityId.LEVITATE)
    out = assert_matches_reference(tables, s, (0, 0), empty_outcome())
    assert int(out.item[1, 0]) == n.item("Griseous Orb") and int(out.knocked_off[1, 0]) == 0
    assert int(out.form_num[1]) == 1 and int(out.form[1, 0]) == 1
    assert out.stats[1, 0].tolist() == s.stats[1, 0].tolist()
    assert int(out.ability[1, 0]) == int(AbilityId.LEVITATE)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # the hit itself lands


@pytest.mark.parametrize("target", ["Golduck", "Giratina"])
def test_a_doubles_giratina_copy_rebuilds_from_its_chosen_targets_party_pokemon(
    tables, names, target
):
    """battle_lib.c:6420-6446 at B = 4: the target is the copier's chosen target
    (battlerActions[..][CHOOSE_TARGET], :6424-6425), and ``Pokemon_SetGiratinaFormByHeldItem``
    recalculates only a Giratina (pokemon.c:4397-4404). A chosen Golduck gives its own party stats
    and ability (Cloud Nine, slot 1), which turns the Sunshine Cherrim back to Overcast in the same
    check (the loop re-runs, :4115-4121; Cherrim is battler 0, before the copier). A chosen Giratina
    with an odd personality (slot 1) still gives Pressure: Altered's second slot is empty
    (``BoxPokemon_CalcAbility``, pokemon.c:4755-4763). Both engines."""
    from dataclasses import replace

    from nuzline.data.tables import to_numpy
    from nuzline.engine.format import DOUBLES
    from nuzline.engine.hooks import form_change
    from nuzline.engine.state import BattleState, initial_arrays
    from nuzline.engine.stats import calc_stats_jnp
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import _form_change
    from tests.engine.d2_states import mon

    n = names
    base = mon(n)
    cherrim = replace(base, species=n.species("Cherrim"))
    ditto = replace(base, species=n.species("Ditto"))
    origin = replace(base, species=n.species("Giratina"), form=1, item=n.item("Griseous Orb"))
    golduck = replace(base, species=n.species("Golduck"), ability_slot=1)
    altered_odd = replace(base, species=n.species("Giratina"), ability_slot=1, level=40)
    second = golduck if target == "Golduck" else altered_odd
    a = initial_arrays(
        DOUBLES,
        tables,
        [[cherrim, ditto], [origin, second]],
        start_condition=Weather.SUN,
        opp2_items=(),
    )
    s = BattleState(**{k: jnp.asarray(v) for k, v in a.items()})
    # Battler 2 (the player's slot 1 Ditto) is a copy of the Origin, its own values in the shadow.
    s = s._replace(
        shadow_species=s.shadow_species.at[2].set(s.species[0, 1]),
        shadow_stats=s.shadow_stats.at[2].set(s.stats[0, 1]),
        shadow_ivs=s.shadow_ivs.at[2].set(s.ivs[0, 1]),
        species=s.species.at[0, 1].set(s.species[1, 0]),
        form=s.form.at[0, 1].set(1),
        stats=s.stats.at[0, 1].set(s.stats[1, 0]),
        ability=s.ability.at[0, 1].set(s.ability[1, 0]),
        transformed=s.transformed.at[2].set(1),
        form_num=s.form_num.at[0].set(1).at[2].set(1),
        chosen_target=s.chosen_target.at[2].set(3),
    )
    out = form_change(tables, s)
    r = to_ref(s)
    _form_change(to_numpy(tables), r)
    for field in ("stats", "ability", "ability_slot", "form", "form_num", "shadow_form"):
        assert getattr(from_ref(r), field).tolist() == getattr(out, field).tolist(), field
    assert int(out.form_num[2]) == 0 and int(out.shadow_form[2]) == 0
    if target == "Golduck":
        assert int(out.ability[0, 1]) == AbilityId.CLOUD_NINE == int(s.ability[1, 1])
        assert out.stats[0, 1].tolist() == s.stats[1, 1].tolist()
        assert int(out.form_num[0]) == 0  # Cloud Nine alive: Overcast (:6391-6395)
    else:
        altered = int(tables.species_form_row[n.species("Giratina"), 0])
        expect = calc_stats_jnp(
            tables.species_base_stats[altered],
            s.ivs[1, 1],
            s.evs[1, 1],
            s.level[1, 1],
            tables.nature_mods[s.nature[1, 1]],
            n.species("Giratina"),
        )
        assert out.stats[0, 1].tolist() == expect[1:].tolist()
        assert (int(out.ability[0, 1]), int(out.ability_slot[0, 1])) == (AbilityId.PRESSURE, 0)
        assert int(out.form_num[0]) == 1
