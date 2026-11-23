"""E4-A-W1 steps 1-2: the battler context and the role commands at B = 4 (rules A-4, A-10 s6, A-14)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai.commands import AI_HAVE, AI_NOT_HAVE, AI_UNKNOWN, Battler, LoadType
from nuzline.ai.outcome import READ_DRAWS, no_calc_draws
from nuzline.data.tables import BATTLE_TYPES
from nuzline.engine.damage import ai_damage_for_roll, decomp_abilities
from nuzline.engine.field import with_trick_room
from nuzline.engine.format import DOUBLES, MULTI, format_of
from nuzline.engine.ids import Stat, Status, Type
from nuzline.engine.order import sort_mon_speed
from nuzline.engine.state import MOVES, ai_record_moves
from tests.ai.b2_snapshot import snapshot
from tests.ai.fixtures import outcome
from tests.engine.d2_states import four, with_hp, with_moves

_SNAPSHOT = Path(__file__).with_name("b2_snapshot.json")


def test_b2_outputs_are_byte_identical(tables, names):
    """Every B = 2 output of ``b2_snapshot`` equals the one the engine gave before the context took
    its battlers (tip 85dd4cd0). Two groups of partner-argument rows changed on purpose, both
    unreachable at B = 2 (Tag Strategy is refused there) and now as the decomp reads them:
    ``IfMoveEffectNotKnown`` with a partner argument has no case, so never jumps
    (trainer_ai.c:1739-1820; it jumped as the defender's), and ``LoadHeldItem`` of a battler the
    format lacks reads its zeroed ``battleMons`` (it read the AI's own item through a clamped
    index). A refused call has no row on either side (``b2_snapshot._run``); a row only ``got``
    has (a call modelled since, or a LoadType member added since) has no old value to compare."""
    got = json.loads(json.dumps(snapshot(tables, names)))
    want = json.loads(_SNAPSHOT.read_text())
    assert set(want) <= set(got)
    changed = {key: got[key] for key, old in want.items() if got[key] != old}
    for key, new in changed.items():
        name = key.split("/", 2)[2]
        assert "PARTNER" in name and name.startswith(("eff_notknown_", "held_item_")), key
        assert new == [0, 0], key


# --- B = 4 -----------------------------------------------------------------------------------

PAIRS = [(1, 0), (1, 2), (1, 3), (3, 0), (3, 1)]  # (attacker, defender)
ARGS = (Battler.DEFENDER, Battler.ATTACKER, Battler.DEFENDER_PARTNER, Battler.ATTACKER_PARTNER)
MOVESETS = (
    ("Tackle", "Swords Dance", "Earthquake", "Curse"),
    ("Surf", "Thunderbolt", "Protect", "Recover"),
    ("Flamethrower", "Ice Beam", "Toxic", "Rest"),
    ("Psychic", "Shadow Ball", "Bite", "Quick Attack"),
)
TYPES = (
    (Type.FIRE, Type.WATER),
    (Type.GRASS, Type.POISON),
    (Type.ELECTRIC, Type.FLYING),
    (Type.ICE, Type.GROUND),
)


def _resolve(arg, a, d):
    return {Battler.ATTACKER: a, Battler.DEFENDER: d, Battler.ATTACKER_PARTNER: a ^ 2}.get(
        arg, d ^ 2
    )


def _at(s, b):
    """``(party row, active slot)`` of battler ``b``."""
    return format_of(s).party_of[b], int(s.active[b])


def _build(tables, names, fmt):
    """A four-battler state in which every battler differs: moves, types, HP, status, stages, item,
    the AI record rows (true item, record item and record moves all differ) and ability (Geodude's
    two abilities, the record of battler 2 naming the second)."""
    s = four(tables, names, fmt)
    geodude = names.species("Geodude")
    abilities = [int(x) for x in np.asarray(tables.species_abilities_decomp)[geodude]]
    effects = {}
    for i in range(1, len(tables.item_hold_effect)):
        effect = int(tables.item_hold_effect[i])
        if 0 < effect < 60:
            effects.setdefault(effect, i)
    pool = list(effects.values())
    assert len(pool) >= 8
    for b in range(4):
        p, slot = _at(s, b)
        s = with_moves(s, names, b, MOVESETS[b])
        s = s._replace(
            species=s.species.at[p, slot].set(geodude),
            form=s.form.at[p, slot].set(0),
            ability_form=s.ability_form.at[p, slot].set(0),
            ability_slot=s.ability_slot.at[p, slot].set(0),
            types=s.types.at[p, slot].set(jnp.asarray([int(t) for t in TYPES[b]], s.types.dtype)),
            item=s.item.at[p, slot].set(pool[b]),
            status=s.status.at[p, slot].set(int(Status.BURN) if b == 2 else 0),
            stages=s.stages.at[b, Stat.ATK].set(b - 1).at[b, Stat.SPE].set(2 - b),
            last_move=s.last_move.at[b].set(names.move(MOVESETS[(b + 1) % 4][1])),
            ai_item_known=s.ai_item_known.at[b].set(pool[4 + b]),
            ai_seen_moves=s.ai_seen_moves.at[b, 0].set(names.move(MOVESETS[(b + 2) % 4][0])),
            ai_ability_known=s.ai_ability_known.at[b].set(0),
        )
        s = with_hp(s, b, 20 + 15 * b)
    s = s._replace(ai_ability_known=s.ai_ability_known.at[2].set(abilities[1]))
    return s, abilities, pool


@pytest.fixture(scope="module", params=["doubles", "multi"])
def world(request, tables, names):
    fmt = DOUBLES if request.param == "doubles" else MULTI
    s, abilities, pool = _build(tables, names, fmt)
    return {"s": s, "abilities": abilities, "pool": pool, "ctx": {}, "fmt": fmt}


def _slot_outcome(rolls=(), reads: int = 8):
    script = np.full((4, reads), READ_DRAWS[0], np.uint8)
    for k, r in enumerate(rolls):
        script[0, k] = r
    return outcome()._replace(script_rolls=jnp.asarray(script), calc_draws=no_calc_draws((), reads))


def _ctx(world, tables, a, d, rolls=()):
    key = (a, d, tuple(rolls))
    if key not in world["ctx"]:
        ctx = c.new_context(tables, world["s"], _slot_outcome(rolls), attacker=a, defender=d)
        world["ctx"][key] = c.at_slot(ctx, 0)
    return world["ctx"][key]


@pytest.mark.parametrize("a, d", PAIRS)
def test_resolver_and_context_fields(world, tables, a, d):
    # AIScript_Battler (trainer_ai.c:2765-2782): pure XOR, no format or HP test; B = 4 has no
    # cleared battler; IfTargetIsPartner (:2630-2639) compares the sides.
    s, ctx = world["s"], _ctx(world, tables, a, d)
    assert type(ctx.attacker) is c.StaticBattler and ctx.attacker == a and int(ctx.defender) == d
    assert [int(c._battler(ctx, arg)) for arg in ARGS] == [d, a, d ^ 2, a ^ 2]
    assert not any(c._is_cleared(ctx, arg) for arg in ARGS)
    assert bool(c.if_target_is_partner(ctx).jump) == ((a & 1) == (d & 1))
    # the attacker's moves, its usable slots, the defender's seen moves (RecordLastMove for a
    # target of the other side only, :372-375)
    p, slot = _at(s, a)
    assert int(ctx.move) == int(s.moves[p, slot, 0])
    assert np.asarray(ctx.scores).tolist() == [100] * 4
    rec = [int(x) for x in s.ai_seen_moves[d] if x]
    last = int(s.last_move[d])
    if (a & 1) != (d & 1) and last not in rec:
        rec.append(last)
    assert np.asarray(ctx.seen_moves).tolist() == rec + [0] * (4 - len(rec))


def test_the_attacker_is_static_and_the_defender_traced(world, tables):
    ctx = _ctx(world, tables, 1, 0)
    assert jax.tree_util.tree_leaves(ctx.attacker) == []
    out = jax.lax.fori_loop(0, 2, lambda i, x: x, ctx)  # a Python int through the loop
    assert type(out.attacker) is c.StaticBattler and out.attacker == 1
    targets = jnp.asarray([0, 2, 3], jnp.int32)  # one script over every target (vmap)
    jumps = jax.vmap(lambda d: c.if_target_is_partner(ctx._replace(defender=d)).jump)(targets)
    assert jumps.tolist() == [False, False, True]
    hp = jax.vmap(
        lambda d: c.if_hp_percent_less_than(ctx._replace(defender=d), Battler.DEFENDER, 40).jump
    )(targets)
    assert hp.shape == (3,)


def _true_ability(tables, s, b):
    p, slot = _at(s, b)
    return int(decomp_abilities(tables, s)[p, slot])


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
@pytest.mark.parametrize("bit", [0, 1])
def test_load_battler_ability(world, tables, a, d, arg, bit):
    # LoadBattlerAbility (trainer_ai.c:1170-1210): the record, a trapping ability or a coin
    # for a battler that is not the attacker, unless the argument is ATTACKER_PARTNER; else the
    # true ability. The coin is a per-slot read: an odd draw is ability 1 (:1195-1199).
    s, (first, second) = world["s"], world["abilities"][:2]
    roll = READ_DRAWS[1] if bit else READ_DRAWS[0]
    out = c.load_battler_ability(_ctx(world, tables, a, d, (roll,)), arg)
    b = _resolve(arg, a, d)
    record = int(s.ai_ability_known[b])
    if not (b != a and arg != Battler.ATTACKER_PARTNER):
        want, reads = _true_ability(tables, s, b), 0
    elif record:
        want, reads = record, 0
    else:
        want, reads = (first if bit else second), 1
    assert int(out.calc_temp) == want
    assert int(out.reads[0]) == reads


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
def test_check_battler_ability(world, tables, a, d, arg):
    # CheckBattlerAbility (:1212-1265): the knowledge branch for DEFENDER and DEFENDER_PARTNER
    # whatever battler they resolve to (the attacker too), the true ability for the others; no coin.
    s, (first, second) = world["s"], world["abilities"][:2]
    ctx = _ctx(world, tables, a, d)
    b = _resolve(arg, a, d)
    record = int(s.ai_ability_known[b])
    for expected in (first, second, 255):
        if arg in (Battler.DEFENDER, Battler.DEFENDER_PARTNER):
            if record:
                seen = record
            else:  # two abilities, neither the expected one: ability 1; else not known
                seen = first if expected not in (first, second) else 0
        else:
            seen = _true_ability(tables, s, b)
        want = AI_UNKNOWN if seen == 0 else (AI_HAVE if seen == expected else AI_NOT_HAVE)
        out = c.check_battler_ability(ctx, arg, expected)
        assert int(out.calc_temp) == want, (arg, expected)
        assert int(out.reads[0]) == 0


@pytest.mark.parametrize("a, d", PAIRS)
def test_move_known_commands(world, tables, names, a, d):
    # IfMoveKnown / NotKnown (:1627-1735): ATTACKER the true moves, ATTACKER_PARTNER the partner's
    # unless its curHP is 0, DEFENDER the record, DEFENDER_PARTNER no case. IfMoveEffectKnown /
    # NotKnown (:1739-1820): ATTACKER true, DEFENDER record, no other case.
    s = world["s"]
    ctx = _ctx(world, tables, a, d)
    partner = [names.move(m) for m in MOVESETS[a ^ 2]]
    mine = [names.move(m) for m in MOVESETS[a]]
    seen = [x for x in np.asarray(ctx.seen_moves).tolist() if x]
    for arg in ARGS:
        for move in (mine[0], partner[0], seen[0], 33):
            expect = {
                Battler.ATTACKER: move in mine,
                Battler.ATTACKER_PARTNER: move in partner,
                Battler.DEFENDER: move in seen,
            }.get(arg)
            for known, fn in ((True, c.if_move_known), (False, c.if_move_not_known)):
                got = bool(fn(ctx, arg, move).jump)
                assert got == (False if expect is None else expect == known), (arg, move, known)
    # a fainted partner: no jump either way (:1650-1653)
    down = with_hp(s, a ^ 2, 0)
    dctx = c.at_slot(c.new_context(tables, down, _slot_outcome(), attacker=a, defender=d), 0)
    for move in (partner[0], 33):
        assert not bool(c.if_move_known(dctx, Battler.ATTACKER_PARTNER, move).jump)
        assert not bool(c.if_move_not_known(dctx, Battler.ATTACKER_PARTNER, move).jump)

    def effect(m):
        return int(tables.move_ai_effect[m])

    for arg in ARGS:
        pool = {Battler.ATTACKER: mine, Battler.DEFENDER: seen}.get(arg)
        for move in (*mine, *partner, *seen):
            found = pool is not None and any(effect(m) == effect(move) for m in pool)
            assert bool(c.if_move_effect_known(ctx, arg, effect(move)).jump) == found
            not_found = pool is not None and not found
            assert bool(c.if_move_effect_not_known(ctx, arg, effect(move)).jump) == not_found


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
def test_held_item_commands(world, tables, a, d, arg):
    # LoadHeldItem (:1891): the true item of any battler. LoadHeldItemEffect (:1901-1913): the
    # record `battlerHeldItems` for every battler but the attacker, the ally included.
    # IfHeldItemEqualTo (:1915-1934): the true item on the attacker's side, the record otherwise.
    s = world["s"]
    ctx = _ctx(world, tables, a, d)
    b = _resolve(arg, a, d)
    p, slot = _at(s, b)
    true, record = int(s.item[p, slot]), int(s.ai_item_known[b])
    assert true != record
    assert int(c.load_held_item(ctx, arg).calc_temp) == true
    want = int(tables.item_hold_effect[true if b == a else record])
    assert int(c.load_held_item_effect(ctx, arg).calc_temp) == want
    same_side = (b & 1) == (a & 1)
    seen_item, other = (true, record) if same_side else (record, true)
    assert bool(c.if_held_item_equal_to(ctx, arg, seen_item).jump)
    assert not bool(c.if_held_item_equal_to(ctx, arg, other).jump)


@pytest.mark.parametrize("a, d", PAIRS)
def test_load_type_from(world, tables, a, d):
    # LoadTypeFrom (:937-988): the partner cases go through BattleSystem_GetPartner (b ^ 2 in
    # doubles); LOAD_DEFENDER_PARTNER_TYPE_2 reads BATTLEMON_TYPE_1 (:980-983), a decomp bug.
    s = world["s"]
    ctx = _ctx(world, tables, a, d)

    def types(b):
        p, slot = _at(s, b)
        return [int(x) for x in s.types[p, slot]]

    want = {
        LoadType.ATTACKER_TYPE_1: types(a)[0],
        LoadType.ATTACKER_TYPE_2: types(a)[1],
        LoadType.DEFENDER_TYPE_1: types(d)[0],
        LoadType.DEFENDER_TYPE_2: types(d)[1],
        LoadType.ATTACKER_PARTNER_TYPE_1: types(a ^ 2)[0],
        LoadType.ATTACKER_PARTNER_TYPE_2: types(a ^ 2)[1],
        LoadType.DEFENDER_PARTNER_TYPE_1: types(d ^ 2)[0],
        LoadType.DEFENDER_PARTNER_TYPE_2: types(d ^ 2)[0],
    }
    for target, value in want.items():
        assert int(c.load_type_from(ctx, target).calc_temp) == value, target
    assert want[LoadType.DEFENDER_PARTNER_TYPE_2] != types(d ^ 2)[1]


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
def test_plain_reads_follow_the_resolved_battler(world, tables, a, d, arg):
    # The commands that read `battleMons[battler]` plainly: HP percent, status, stages, previous
    # move, type test, all by the resolved battler.
    s = world["s"]
    ctx = _ctx(world, tables, a, d)
    b = _resolve(arg, a, d)
    p, slot = _at(s, b)
    pct = int(s.hp[p, slot]) * 100 // int(s.max_hp[p, slot])
    assert bool(c.if_hp_percent_equal_to(ctx, arg, pct).jump)
    assert bool(c.if_hp_percent_less_than(ctx, arg, pct + 1).jump)
    assert not bool(c.if_hp_percent_greater_than(ctx, arg, pct).jump)
    assert bool(c.if_status(ctx, arg, c.MonCondition.BURN).jump) == (b == 2)
    stage = int(s.stages[b, Stat.ATK]) + 6
    assert bool(c.if_stat_stage_equal_to(ctx, arg, Stat.ATK, stage).jump)
    assert int(c.load_battler_previous_move(ctx, arg).calc_temp) == int(s.last_move[b])
    assert int(c.flag_battler_is_type(ctx, arg, int(TYPES[b][1])).calc_temp) == 1
    assert int(c.flag_battler_is_type(ctx, arg, int(TYPES[(b + 1) % 4][0])).calc_temp) == 0
    positive = max(int(s.stages[b, Stat.ATK]), 0) + max(int(s.stages[b, Stat.SPE]), 0)
    assert int(c.sum_positive_stat_stages(ctx, arg).calc_temp) == positive


@pytest.mark.parametrize("a, d", PAIRS)
def test_battler_deals_more_damage(world, tables, a, d):
    # IfBattlerDealsMoreDamage (:2187-2240): the defender's previous move, calculated as the
    # defender with the ATTACKER's IVs against the defender (AI_CONTEXT.defender), against the
    # best of the attacker's own moves.
    s = world["s"]
    ctx = _ctx(world, tables, a, d)
    p, slot = _at(s, a)
    best = max(
        int(c.ai_damage(tables, s, int(m), c.SIM_ROLL_MAX, atk_side=a, def_side=d))
        for m in s.moves[p, slot]
    )
    theirs = int(
        ai_damage_for_roll(
            tables, s, d, int(s.last_move[d]), c.SIM_ROLL_MAX, def_side=d, ivs=s.ivs[p, slot]
        )
    )
    out = c.if_battler_deals_more_damage(ctx, Battler.DEFENDER, c.USE_MAX_DAMAGE)
    assert bool(out.jump) == (theirs > best)
    # a defender that last used nothing deals the chart's +2 at most: never more
    quiet = ctx._replace(state=s._replace(last_move=s.last_move.at[d].set(0)))
    assert not bool(c.if_battler_deals_more_damage(quiet, Battler.DEFENDER, c.USE_MAX_DAMAGE).jump)


# --- E4-A-W1 step 3: the party commands (A-6) ----------------------------------------------


def _tables4(tables, fmt):
    """``tables`` with the fight's ``battleType``: every B = 4 type has the DOUBLES bit (encounter.c:
    724-740, ``data.tables.BATTLE_TYPES``)."""
    kind = {DOUBLES: "double", MULTI: "multi"}[fmt]
    return tables._replace(battle_type=BATTLE_TYPES[kind])


def _party_ctx(tables, s, a=1, d=0):
    return c.at_slot(c.new_context(tables, s, _slot_outcome(), attacker=a, defender=d), 0)


def _put(s, p, slot, **fields):
    for name, value in fields.items():
        s = s._replace(**{name: getattr(s, name).at[p, slot].set(value)})
    return s


def _count(ctx, arg):
    return int(c.count_alive_party_battlers(ctx, arg).calc_temp)


def test_count_doubles_shared(tables, names):
    # DOUBLES: battlers 1 and 3 on slots 0 and 1 of the shared party 1 (trainer_ai.c:1138-1143):
    # both actives are excluded, whichever of the two asks; the player's side the same.
    t = _tables4(tables, DOUBLES)
    s = four(t, names, DOUBLES)
    assert s.active.tolist() == [0, 0, 1, 1]
    for a, d in ((1, 0), (3, 0), (1, 2)):
        ctx = _party_ctx(t, s, a, d)
        assert [_count(ctx, arg) for arg in ARGS] == [4, 4, 4, 4]
    # a fainted bench member does not count; a fainted partner's active is excluded anyway
    s2 = with_hp(_put(s, 1, 4, hp=0), 3, 0)
    ctx = _party_ctx(t, s2, 1, 0)
    assert _count(ctx, Battler.ATTACKER) == 3 and _count(ctx, Battler.ATTACKER_PARTNER) == 3
    # without the DOUBLES bit one slot is excluded (the singles rule, :1138-1143): the bit decides
    single = _party_ctx(tables, s, 1, 0)
    assert _count(single, Battler.ATTACKER) == 5


def test_count_multi_cross_party(tables, names):
    # MULTI: battler 1 (party 1) on slot 0, battler 3 (party 3) on slot 2. The decomp excludes
    # sel[partner] from battler 1's own party too (a quirk, A-6): party-1 slots 0 and 2.
    t = _tables4(tables, MULTI)
    s = four(t, names, MULTI)
    s = s._replace(active=s.active.at[3].set(2))
    ctx = _party_ctx(t, s, 1, 0)
    assert _count(ctx, Battler.ATTACKER) == 4
    # party 3's own count skips its slot 2 and battler 1's slot 0 (sel[partner] of battler 3)
    assert _count(ctx, Battler.ATTACKER_PARTNER) == 4
    assert (
        _count(_party_ctx(t, _put(s, 1, 2, hp=0), 1, 0), Battler.ATTACKER) == 4
    )  # excluded anyway
    assert _count(_party_ctx(t, _put(s, 1, 1, hp=0), 1, 0), Battler.ATTACKER) == 3


def test_count_partner_absent(tables, names):
    # A partner with no replacement is absent: selectedPartySlot = 6 (bcp.c:4109-4111), so its
    # (fainted) slot excludes nothing from the other party (MULTI: battler 3 on slot 2).
    t = _tables4(tables, MULTI)
    s = four(t, names, MULTI)
    s = s._replace(active=s.active.at[3].set(2), hp=s.hp.at[3].set(0))
    assert bool(c.absent_now(s, 3)) and not bool(c.absent_now(s, 1))
    assert _count(_party_ctx(t, s, 1, 0), Battler.ATTACKER) == 5  # only battler 1's own slot
    # the same partner fainted with a living bench is not absent: slot 2 is excluded again
    s = s._replace(hp=s.hp.at[3, 4].set(30))
    assert not bool(c.absent_now(s, 3))
    assert _count(_party_ctx(t, s, 1, 0), Battler.ATTACKER) == 4


@pytest.mark.parametrize("fmt", [DOUBLES, MULTI])
def test_status_skips_the_actives(tables, names, fmt):
    # IfPartyMemberStatus / NotStatus (:1348-1412): the same two slots are skipped as in the
    # count; a poisoned active (own or the partner's) never jumps, a poisoned bench member does.
    t = _tables4(tables, fmt)
    s = four(t, names, fmt)
    s = s._replace(active=s.active.at[3].set(1 if fmt == DOUBLES else 2))
    skipped = (0, 1) if fmt == DOUBLES else (0, 2)  # slots of party 1 the command excludes
    poison, burn = int(Status.POISON), c.MonCondition.POISON
    on_actives = s
    for slot in skipped:
        on_actives = _put(on_actives, 1, slot, status=poison)
    ctx = _party_ctx(t, on_actives, 1, 0)
    assert not bool(c.if_party_member_status(ctx, Battler.ATTACKER, burn).jump)
    assert bool(c.if_party_member_not_status(ctx, Battler.ATTACKER, burn).jump)
    bench = _put(s, 1, 4, status=poison)
    ctx = _party_ctx(t, bench, 1, 0)
    assert bool(c.if_party_member_status(ctx, Battler.ATTACKER, burn).jump)
    assert not bool(c.if_party_member_status(ctx, Battler.ATTACKER, c.MonCondition.BURN).jump)
    # a fainted poisoned bench member has no HP: never counted
    ctx = _party_ctx(t, _put(bench, 1, 4, hp=0), 1, 0)
    assert not bool(c.if_party_member_status(ctx, Battler.ATTACKER, burn).jump)
    # NotStatus is False when every considered member has the status
    everyone = s
    for slot in range(6):
        everyone = _put(everyone, 1, slot, status=poison)
    ctx = _party_ctx(t, everyone, 1, 0)
    assert not bool(c.if_party_member_not_status(ctx, Battler.ATTACKER, burn).jump)
    assert bool(c.if_party_member_status(ctx, Battler.ATTACKER, burn).jump)


def test_wounded_and_used_pp_by_battler(tables, names):
    # IsWounded / UsedPP (:1968-2012): own active skipped only; the partner's wounded active of a
    # shared party jumps, a fainted bench member jumps IsWounded; DEFENDER reads the player's party.
    t = _tables4(tables, DOUBLES)
    s = four(t, names, DOUBLES)
    ctx = _party_ctx(t, s, 1, 0)
    for arg in ARGS:
        assert not bool(c.if_any_party_member_is_wounded(ctx, arg).jump)
        assert not bool(c.if_any_party_member_used_pp(ctx, arg).jump)
    ctx = _party_ctx(t, _put(s, 1, 1, hp=5), 1, 0)  # battler 3's active, a wounded partner
    assert bool(c.if_any_party_member_is_wounded(ctx, Battler.ATTACKER).jump)
    assert not bool(c.if_any_party_member_is_wounded(ctx, Battler.DEFENDER).jump)
    ctx = _party_ctx(t, _put(s, 1, 0, hp=5), 1, 0)  # battler 1's own active is skipped
    assert not bool(c.if_any_party_member_is_wounded(ctx, Battler.ATTACKER).jump)
    assert bool(c.if_any_party_member_is_wounded(ctx, Battler.ATTACKER_PARTNER).jump)  # battler 3
    ctx = _party_ctx(t, _put(s, 1, 5, hp=0), 1, 0)  # a fainted bench member
    assert bool(c.if_any_party_member_is_wounded(ctx, Battler.ATTACKER).jump)
    used = _put(s, 1, 1, pp=s.pp[1, 1].at[2].add(-1))  # the partner's active used a PP
    ctx = _party_ctx(t, used, 1, 0)
    assert bool(c.if_any_party_member_used_pp(ctx, Battler.ATTACKER).jump)
    assert not bool(c.if_any_party_member_used_pp(ctx, Battler.DEFENDER).jump)
    # PP Ups raise the maximum (MoveTable_CalcMaxPP): a slot at its base PP with 3 ups has used some
    ups = _put(s, 1, 4, pp_ups=jnp.asarray([3, 0, 0, 0], s.pp_ups.dtype))
    assert bool(c.if_any_party_member_used_pp(_party_ctx(t, ups, 1, 0), Battler.ATTACKER).jump)
    # an absent battler skips no slot (selectedPartySlot 6): its own fainted slot is "wounded"
    dead = s._replace(hp=s.hp.at[1].set(0))
    assert bool(c.absent_now(dead, 1))
    ctx = _party_ctx(t, dead, 3, 0)
    assert bool(c.if_any_party_member_is_wounded(ctx, Battler.ATTACKER_PARTNER).jump)


def _with_mimic(s, names, b, slot, copy, party_move, party_pp, copy_pp=5):
    """Battler ``b`` Mimicked ``copy`` into move ``slot`` (BtlCmd_Mimic, battle_script.c:4500-4512:
    the BattleMon's slot is the copy at min(5, PP), the slot flagged), its party Pokemon keeping
    ``party_move`` at ``party_pp`` (BtlIOCmd_UpdatePartyMon skips the slot, battle_io_command.c:581)."""
    p, mon = _at(s, b)
    return s._replace(
        moves=s.moves.at[p, mon, slot].set(names.move(copy)),
        pp=s.pp.at[p, mon, slot].set(copy_pp),
        pp_ups=s.pp_ups.at[p, mon, slot].set(0),
        mimic_slots=s.mimic_slots.at[b].set(1 << slot),
        shadow_moves=s.shadow_moves.at[b, slot].set(names.move(party_move)),
        shadow_pp=s.shadow_pp.at[b, slot].set(party_pp),
    )


def _with_transform(s, names, b, copied, party_moves, party_pp):
    """Battler ``b`` is Transformed (BtlCmd_Transform, battle_script.c:5342-5370): its BattleMon
    holds ``copied`` at 5 PP; the party Pokemon keeps ``party_moves`` / ``party_pp``, its IVs,
    species and ability slot (the ``shadow_*``; UpdatePartyMon writes no move under Transform,
    battle_io_command.c:579)."""
    p, mon = _at(s, b)
    own = [names.move(m) for m in party_moves] + [0] * (4 - len(party_moves))
    s = s._replace(
        transformed=s.transformed.at[b].set(1),
        mimic_slots=s.mimic_slots.at[b].set(0xF),
        shadow_moves=s.shadow_moves.at[b].set(jnp.asarray(own, s.shadow_moves.dtype)),
        shadow_pp=s.shadow_pp.at[b].set(jnp.asarray(party_pp, s.shadow_pp.dtype)),
        shadow_species=s.shadow_species.at[b].set(s.species[p, mon]),
        shadow_ivs=s.shadow_ivs.at[b].set(s.ivs[p, mon]),
        shadow_ability_slot=s.shadow_ability_slot.at[b].set(s.ability_slot[p, mon]),
        shadow_ability_form=s.shadow_ability_form.at[b].set(s.ability_form[p, mon]),
        pp_ups=s.pp_ups.at[p, mon].set(0),
    )
    return with_moves(s, names, b, copied, pp=5)


def test_used_pp_reads_the_party_copy(tables, names):
    # MUST 1 of review_w1_92eb8c44: IfAnyPartyMemberUsedPP compares the party Pokemon's
    # MON_DATA_MOVE1_PP with MON_DATA_MOVE1_MAX_PP (trainer_ai.c:1999), so the DOUBLES partner's
    # active (not skipped, :1997) counts with its party moves and PP, not its Mimic / Transform copy.
    # Basic_CheckLunarDance jumps on it (script.s:1561).
    t = _tables4(tables, DOUBLES)
    s = four(t, names, DOUBLES)
    p, mon = _at(s, 3)
    assert (p, mon) == (1, 1) and not bool(c.if_any_party_member_used_pp(_party_ctx(t, s), 1).jump)
    used = Battler.ATTACKER
    # Mimicked Fire Blast: the copy is at 5 of 5 (unused), the party's Mimic at 9 of 10 (used)
    s1 = _with_mimic(s, names, 3, 2, "Fire Blast", "Mimic", 9)
    assert int(s1.pp[p, mon, 2]) == int(t.move_pp[names.move("Fire Blast")]) == 5
    assert bool(c.if_any_party_member_used_pp(_party_ctx(t, s1), used).jump)
    # the reverse: the copy used (4 of 5), the party's Mimic full (10 of 10)
    s2 = _with_mimic(s, names, 3, 2, "Fire Blast", "Mimic", 10, copy_pp=4)
    assert not bool(c.if_any_party_member_used_pp(_party_ctx(t, s2), used).jump)
    # Transformed: the copy's moves at 5 PP (Tackle's max is 35), the party's own set full, then used
    own = ("Surf", "Growl")
    full = [int(t.move_pp[names.move(m)]) for m in own] + [0, 0]
    s3 = _with_transform(s, names, 3, ("Tackle", "Ember"), own, full)
    assert not bool(c.if_any_party_member_used_pp(_party_ctx(t, s3), used).jump)
    s4 = _with_transform(s, names, 3, ("Tackle", "Ember"), own, [full[0] - 1, *full[1:]])
    assert bool(c.if_any_party_member_used_pp(_party_ctx(t, s4), used).jump)
    # battler 3 itself skips its own slot: its Mimic party copy is not read
    assert not bool(c.if_any_party_member_used_pp(_party_ctx(t, s1, 3, 0), used).jump)


def test_party_copy_values(tables, names):
    # _party_copy: IVs and ability of a Transformed partner are its party Pokemon's (the shadow;
    # BtlCmd_Transform writes the BattleMon only, battle_script.c:5342-5370), the held item of a
    # knocked-off slot is the party's kept one (battle_io_command.c:590-592), every other slot is
    # the engine's own value.
    t = _tables4(tables, DOUBLES)
    s = four(t, names, DOUBLES)
    p, mon = _at(s, 3)
    own_ivs = jnp.asarray([1, 2, 3, 4, 5, 6], s.ivs.dtype)
    s = s._replace(ivs=s.ivs.at[p, mon].set(own_ivs))
    s = _with_transform(s, names, 3, ("Tackle",), ("Surf",), [15, 0, 0, 0])
    own_ability = int(s.ability[p, mon])
    copied = jnp.asarray([31] * 6, s.ivs.dtype)
    s = s._replace(
        ivs=s.ivs.at[p, mon].set(copied),
        ability=s.ability.at[p, mon].set(own_ability + 1),
        knocked_off=s.knocked_off.at[1, 4].set(1),
        knocked_item=s.knocked_item.at[p, 4].set(7),
        item=s.item.at[p, 4].set(0),
    )
    mons = c._party_copy(t, s, 1)
    assert mons.ivs[mon].tolist() == own_ivs.tolist()
    assert int(mons.ability[mon]) == own_ability
    assert mons.moves[mon].tolist() == [names.move("Surf"), 0, 0, 0]
    assert int(mons.item[4]) == 7
    for slot in (2, 3, 5):
        assert mons.moves[slot].tolist() == s.moves[p, slot].tolist()
        assert mons.ivs[slot].tolist() == s.ivs[p, slot].tolist()
    # the other side's party: no battler of party 1 is in it
    assert c._party_copy(t, s, 0).ivs.tolist() == s.ivs[0].tolist()


def test_party_member_deals_more_reads_the_party_copy(tables, names):
    # MUST 2 of review_w1_92eb8c44: IfPartyMemberDealsMoreDamage scores each member's
    # MON_DATA_MOVE1 + j (trainer_ai.c:2149-2151), so the DOUBLES partner's active is scored with
    # its party moves, not the Transform or Mimic copy. Only battler 3's slot is a living candidate.
    t = _tables4(tables, DOUBLES)
    s = four(t, names, DOUBLES)
    s = with_moves(s, names, 1, ("Tackle",))
    for slot in range(2, 6):
        s = _put(s, 1, slot, hp=0)

    def jumps(state):
        return bool(
            c.if_party_member_deals_more_damage(_party_ctx(t, state), c.USE_MAX_DAMAGE).jump
        )

    weak, strong = ("Growl",), ("Earthquake",)
    assert jumps(with_moves(s, names, 3, strong)) and not jumps(with_moves(s, names, 3, weak))
    assert not jumps(_with_transform(s, names, 3, strong, weak, [40, 0, 0, 0]))
    assert jumps(_with_transform(s, names, 3, weak, strong, [10, 0, 0, 0]))
    base = with_moves(s, names, 3, weak)
    assert not jumps(_with_mimic(base, names, 3, 0, "Earthquake", "Growl", 40))
    assert jumps(_with_mimic(base, names, 3, 1, "Growl", "Earthquake", 10))


def test_party_commands_b2_values(tables, names):
    # B = 2: the singles rule, one slot excluded (the byte-identity snapshot covers the rest)
    from tests.ai.fixtures import versus

    s = versus(tables, names, 0)
    ctx = c.at_slot(c.new_context(tables, s, _slot_outcome()), 0)
    alive = int(((s.species[1] > 0) & (s.hp[1] > 0)).sum()) - 1
    assert _count(ctx, Battler.ATTACKER) == alive
    sick = _put(s, 1, 1, status=int(Status.BURN))
    ctx = c.at_slot(c.new_context(tables, sick, _slot_outcome()), 0)
    assert bool(c.if_party_member_status(ctx, Battler.ATTACKER, c.MonCondition.BURN).jump)
    assert not bool(c.if_party_member_status(ctx, Battler.DEFENDER, c.MonCondition.BURN).jump)


# --- E4-A-W1 step 4: FlagBattlerIsType, IfActivatedFlashFire, IfBattlerFainted at B = 4 ------


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
def test_flag_battler_is_type_b4(world, tables, a, d, arg):
    # FlagBattlerIsType (:991-1004): MON_HAS_TYPE of the resolved battler, type 1 or type 2, no
    # HP test (a fainted battler's types still read).
    s = world["s"]
    b = _resolve(arg, a, d)
    ctx = _ctx(world, tables, a, d)
    for t in range(4):
        for type_ in TYPES[t]:
            want = int(type_) in (int(TYPES[b][0]), int(TYPES[b][1]))
            assert int(c.flag_battler_is_type(ctx, arg, int(type_)).calc_temp) == int(want)
    down = c.new_context(tables, with_hp(s, b, 0), _slot_outcome(), attacker=a, defender=d)
    assert int(c.flag_battler_is_type(down, arg, int(TYPES[b][1])).calc_temp) == 1


@pytest.mark.parametrize("a, d", PAIRS)
@pytest.mark.parametrize("arg", ARGS)
def test_flash_fire_partner_b4(world, tables, a, d, arg):
    # IfActivatedFlashFire (:2641-2652): the resolved battler's moveEffectsData.flashFire.
    s = world["s"]
    b = _resolve(arg, a, d)
    assert not bool(c.if_activated_flash_fire(_ctx(world, tables, a, d), arg).jump)
    for owner in range(4):
        lit = s._replace(flash_fire=s.flash_fire.at[owner].set(1))
        out = c.new_context(tables, lit, _slot_outcome(), attacker=a, defender=d)
        assert bool(c.if_activated_flash_fire(out, arg).jump) == (owner == b)


@pytest.mark.parametrize("fmt", [DOUBLES, MULTI])
def test_battler_fainted_b4(tables, names, fmt):
    # IfBattlerFainted (:2438-2452): battlersSwitchingMask, i.e. actions.absent_now, not HP: a
    # fainted partner with a living bench is not absent and does not jump (partner args only).
    t = _tables4(tables, fmt)
    s = four(t, names, fmt)
    ctx = _party_ctx(t, s, 1, 0)
    assert not bool(c.if_battler_fainted(ctx, Battler.ATTACKER_PARTNER).jump)
    assert not bool(c.if_battler_fainted(ctx, Battler.DEFENDER_PARTNER).jump)
    p3 = format_of(s).party_of[3]
    down = with_hp(s, 3, 0)  # battler 3 fainted with a bench
    assert not bool(c.absent_now(down, 3))
    assert not bool(c.if_battler_fainted(_party_ctx(t, down, 1, 0), Battler.ATTACKER_PARTNER).jump)
    gone = down._replace(hp=down.hp.at[p3].set(0))  # its whole party down: absent
    if fmt == DOUBLES:  # the shared party 1 also holds battler 1: keep it alive
        gone = gone._replace(hp=gone.hp.at[1, 0].set(50))
    assert bool(c.absent_now(gone, 3))
    assert bool(c.if_battler_fainted(_party_ctx(t, gone, 1, 0), Battler.ATTACKER_PARTNER).jump)
    # DEFENDER_PARTNER of the target 1 is battler 3; of the target 0 it is battler 2 (alive)
    assert bool(c.if_battler_fainted(_party_ctx(t, gone, 0, 1), Battler.DEFENDER_PARTNER).jump)
    assert not bool(c.if_battler_fainted(_party_ctx(t, gone, 1, 0), Battler.DEFENDER_PARTNER).jump)


# --- LoadBattlerSpeedRank at B = 4 (decision 6, the manager's ruling): the one-read law -------

FASTER, SLOWER, TIE = c.COMPARE_SPEED_FASTER, c.COMPARE_SPEED_SLOWER, c.COMPARE_SPEED_TIE
ORDER_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))  # the decomp's i < j loop


def _speed_state(
    tables,
    names,
    speeds,
    fmt=DOUBLES,
    alive=(1, 1, 1, 1),
    trick_room=False,
    claw=(0, 0, 0, 0),
    lag=(0, 0, 0, 0),
):
    """A four-battler state whose battlers have the Speed stat ``speeds`` (all else equal) and HP
    only where ``alive``; a ``claw`` battler holds a Quick Claw whose ``qc_roll`` activates it, a
    ``lag`` battler a Lagging Tail."""
    s = four(tables, names, fmt)
    for b in range(4):
        p = format_of(s).party_of[b]
        slot = int(s.active[b])
        s = _put(s, p, slot, stats=s.stats[p, slot].at[Stat.SPE - 1].set(speeds[b]))
        if claw[b]:
            s = _put(s, p, slot, item=names.item("Quick Claw"))
            s = s._replace(qc_roll=s.qc_roll.at[b].set(1))
        if lag[b]:
            s = _put(s, p, slot, item=names.item("Lagging Tail"))
        if not alive[b]:
            s = with_hp(s, b, 0)
    return with_trick_room(s, 3) if trick_room else s


def _py_compare(speed, alive, trick, b1, b2, claw=(0, 0, 0, 0), lag=(0, 0, 0, 0)):
    """``BattleSystem_CompareBattlerSpeed(b1, b2, TRUE)`` (battle_lib.c:1204-1211, :1401-1447):
    ``(result, draws)``, ``result`` a function of the tie bit, ``draws`` whether the compare reaches
    ``RandNext() & 1`` (equal speeds, in a two-sided branch). ``claw`` marks an activated Quick Claw
    (``speedRand`` from ``qc_roll``, :1278-1286), ``lag`` a Lagging Tail holder (:1299-1301)."""
    if alive[b1] != alive[b2]:
        return (lambda bit: FASTER if alive[b1] else SLOWER), False
    s1, s2 = speed[b1], speed[b2]
    if claw[b1] and claw[b2]:  # both claws (:1402-1407): plain speed, Trick Room not read
        slower = s1 < s2
    elif claw[b1] or claw[b2]:  # one claw (:1408-1411): fixed, no draw
        return (lambda bit: FASTER if claw[b1] else SLOWER), False
    elif lag[b1] and lag[b2]:  # both Lagging Tails (:1412-1417): reversed speed
        slower = s1 > s2
    elif lag[b1] or lag[b2]:  # one Lagging Tail (:1418-1421): fixed, no draw
        return (lambda bit: SLOWER if lag[b1] else FASTER), False
    else:  # plain (:1432-1447)
        slower = s1 > s2 if trick else s1 < s2
    if s1 == s2:
        return (lambda bit: TIE if bit else FASTER), True
    return (lambda bit: SLOWER if slower else FASTER), False


def _py_rank_laws(speed, alive, trick=False, claw=(0, 0, 0, 0), lag=(0, 0, 0, 0)):
    """Brute force over the decomp's own draw process: the sort of ``AICmd_LoadBattlerSpeedRank``
    (trainer_ai.c:2056-2094) with a fresh ``RandNext() & 1`` at each compare that reaches the tie
    test, a branch tree of probability 1/2 per draw. ``{battler: [P(rank 0..3)]}`` (Fractions)."""
    from fractions import Fraction

    law = {b: [Fraction(0)] * 4 for b in range(4)}

    def go(order, pairs, weight):
        if not pairs:
            for pos, b in enumerate(order):
                law[b][pos] += weight
            return
        (i, j), rest = pairs[0], pairs[1:]
        result, draws = _py_compare(speed, alive, trick, order[i], order[j], claw, lag)
        for bit in (0, 1) if draws else (0,):
            o = list(order)
            if result(bit) != FASTER:
                o[i], o[j] = o[j], o[i]
            go(o, rest, weight / (2 if draws else 1))

    go([0, 1, 2, 3], list(ORDER_PAIRS), Fraction(1))
    return law


SPEED_CASES = [
    ((10, 20, 30, 40), (1, 1, 1, 1), False),  # distinct: no read
    ((30, 30, 10, 20), (1, 1, 1, 1), False),  # a two-way tie
    ((30, 10, 30, 10), (1, 1, 1, 1), False),  # two tied pairs
    ((50, 50, 50, 40), (1, 1, 1, 1), False),  # a three-way tie
    ((50, 50, 50, 50), (1, 1, 1, 1), False),  # a four-way tie
    ((50, 50, 50, 50), (1, 1, 1, 1), True),  # the same under Trick Room
    ((50, 50, 20, 20), (1, 1, 1, 1), True),
    ((50, 50, 50, 50), (1, 0, 1, 1), False),  # a dead battler: last, no draw with it
    ((50, 50, 50, 50), (0, 1, 1, 1), False),
    ((50, 50, 50, 50), (1, 0, 0, 1), False),  # two dead battlers tied: their order is drawn
    ((50, 50, 50, 50), (0, 0, 0, 1), False),
    ((50, 40, 50, 40), (1, 0, 1, 0), True),
]

_law = jax.jit(c.speed_rank_law)


def _assert_law_is_the_brute_force(t, s, want):
    total = np.zeros(4)
    for b in range(4):
        got = np.asarray(_law(t, s, b), np.float64)
        np.testing.assert_allclose(got, [float(x) for x in want[b]], atol=0, rtol=0)
        assert all(float(x * 64).is_integer() for x in got)  # every probability a multiple of 1/64
        assert abs(got.sum() - 1.0) < 1e-12
        total += got
    # the four battlers fill every position exactly once in each outcome
    np.testing.assert_allclose(total, 1.0)


@pytest.mark.parametrize("speeds, alive, trick", SPEED_CASES)
def test_speed_rank_law_equals_the_brute_force(tables, names, speeds, alive, trick):
    # speed_rank_law (the mean of the sort over the 2**6 assignments of the pair bits) equals the
    # decomp's draw process enumerated as a tree (a bit only at a compare that reaches the tie
    # test), for every battler: the joint law of every tie bit of the call, exactly.
    t = _tables4(tables, DOUBLES)
    s = _speed_state(t, names, speeds, alive=alive, trick_room=trick)
    _assert_law_is_the_brute_force(t, s, _py_rank_laws(speeds, alive, trick))


# W2-15 (A-10 2.6, 2.10 item 1): Quick Claw (activated from qc_roll) and Lagging Tail ladders,
# battle_lib.c:1402-1421. (speeds, alive, trick, claw, lag)
ITEM_SPEED_CASES = [
    ((50, 50, 30, 20), (1, 1, 1, 1), False, (1, 1, 0, 0), (0, 0, 0, 0)),  # claw vs claw tie: draws
    ((50, 50, 30, 20), (1, 1, 1, 1), True, (1, 1, 0, 0), (0, 0, 0, 0)),  # ... Trick Room not read
    ((40, 50, 50, 40), (1, 1, 1, 1), True, (1, 1, 0, 0), (0, 0, 0, 0)),  # claw pair by plain speed
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (1, 1, 1, 1), (0, 0, 0, 0)),  # four claws, all tied
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (1, 0, 0, 0), (0, 0, 0, 0)),  # claw vs none: no draw
    ((10, 50, 50, 50), (1, 1, 1, 1), False, (1, 0, 0, 0), (0, 0, 0, 0)),  # a slow lone claw first
    ((50, 50, 50, 50), (1, 1, 1, 1), True, (0, 0, 1, 0), (0, 0, 0, 0)),  # lone claw in Trick Room
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (0, 0, 0, 0), (1, 1, 0, 0)),  # Lagging Tail pair draws
    ((30, 40, 50, 50), (1, 1, 1, 1), False, (0, 0, 0, 0), (1, 1, 0, 0)),  # tail pair: slower first
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (0, 0, 0, 0), (0, 0, 1, 0)),  # one tail: last, no draw
    ((50, 50, 50, 50), (1, 1, 1, 1), True, (0, 0, 0, 0), (0, 1, 0, 0)),  # one tail, Trick Room
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (1, 0, 0, 0), (0, 0, 0, 1)),  # claw first, tail last
    ((50, 50, 50, 50), (1, 0, 1, 1), False, (1, 0, 0, 0), (0, 0, 1, 0)),  # claw, tail, one dead
]


@pytest.mark.parametrize("speeds, alive, trick, claw, lag", ITEM_SPEED_CASES)
def test_speed_rank_law_claws_and_lagging_tail(tables, names, speeds, alive, trick, claw, lag):
    # The two-sided branches (both claws :1402-1407, both tails :1412-1417) draw at equal speed;
    # the one-sided ones (:1408-1411, :1418-1421) never do; the claw bit is the state's qc_roll.
    t = _tables4(tables, DOUBLES)
    s = _speed_state(t, names, speeds, alive=alive, trick_room=trick, claw=claw, lag=lag)
    _assert_law_is_the_brute_force(t, s, _py_rank_laws(speeds, alive, trick, claw, lag))


def test_speed_rank_law_known_values(tables, names):
    t = _tables4(tables, DOUBLES)

    def law(s, b):
        return [float(x) for x in _law(t, s, b)]

    distinct = _speed_state(t, names, (10, 20, 30, 40))
    assert [law(distinct, b).index(1.0) for b in range(4)] == [3, 2, 1, 0]  # the fastest first
    tie2 = _speed_state(t, names, (30, 30, 10, 20))
    assert law(tie2, 0) == [0.5, 0.5, 0.0, 0.0] and law(tie2, 1) == [0.5, 0.5, 0.0, 0.0]
    assert law(tie2, 2) == [0.0, 0.0, 0.0, 1.0] and law(tie2, 3) == [0.0, 0.0, 1.0, 0.0]
    dead = _speed_state(t, names, (50, 50, 50, 50), alive=(1, 0, 1, 1))
    assert law(dead, 1) == [0.0, 0.0, 0.0, 1.0]  # a dead battler is last
    two_dead = _speed_state(t, names, (50, 50, 50, 50), alive=(1, 0, 0, 1))
    assert law(two_dead, 1) == [0.0, 0.0, 0.5, 0.5]  # their order is drawn
    trick = _speed_state(t, names, (10, 20, 30, 40), trick_room=True)
    assert [law(trick, b).index(1.0) for b in range(4)] == [0, 1, 2, 3]
    # Quick Claw from qc_roll (ignoreQuickClaw only skips the write): a lone activated holder is first
    p, slot = _at(distinct, 0)
    held = distinct._replace(item=distinct.item.at[p, slot].set(names.item("Quick Claw")))
    assert law(held, 0).index(1.0) == 3  # not activated: its own speed, last
    assert law(held._replace(qc_roll=held.qc_roll.at[0].set(1)), 0).index(1.0) == 0


def test_sort_four_matches_the_engine_sort_and_the_decomp(tables, names):
    # The loose oracle's sort (_sort_four, the loop of AICmd_LoadBattlerSpeedRank) equals the
    # engine's order.sort_mon_speed for every assignment of the pair bits, and the written set does
    # not depend on the bits, for every alive pattern (decision 6: the writes are not a read).
    t = _tables4(tables, DOUBLES)
    all_bits = jnp.asarray(list(itertools.product((0, 1), repeat=6)))

    @jax.jit
    def both(t, s):
        mine = jax.vmap(lambda bits: c._sort_four(t, s, lambda p, reads: bits[p]))(all_bits)
        theirs = jax.vmap(lambda bits: sort_mon_speed(t, s, s.qc_roll, bits)[0])(all_bits)
        return mine, theirs

    for speeds in ((50, 50, 50, 50), (50, 40, 50, 40), (10, 20, 30, 40)):
        for alive in itertools.product((1, 0), repeat=4):
            (order, written), want = both(t, _speed_state(t, names, speeds, alive=alive))
            assert np.asarray(order).tolist() == np.asarray(want).tolist(), (speeds, alive)
            sets = {tuple(bool(x) for x in row) for row in np.asarray(written)}
            # a battler is stored when some compare pairs it with one of its own class (alive or
            # dead); a lone member of its class is never paired past the short-circuit (:1204-1211)
            lone = {x for x in range(4) if sum(a == alive[x] for a in alive) == 1}
            assert sets == {tuple(b not in lone for b in range(4))}, (speeds, alive, sets)


def _rank_ctx(tables, s, a, d, rolls=(), loose_bits=None):
    loose = None if loose_bits is None else c.loose_draws(loose_bits)
    ctx = c.new_context(tables, s, _slot_outcome(rolls), attacker=a, defender=d, loose=loose)
    return c.at_slot(ctx, 0)


ONE_READ_CASES = [
    ((50, 50, 50, 50), (1, 1, 1, 1)),
    ((50, 50, 30, 20), (1, 1, 1, 1)),
    ((50, 50, 50, 50), (1, 0, 0, 1)),
]
RANK_ARGS = (Battler.ATTACKER, Battler.ATTACKER_PARTNER, Battler.DEFENDER, Battler.DEFENDER_PARTNER)


def _quick_claw(t, names, s, holders):
    """``s`` with every battler of ``holders`` holding an activated Quick Claw (``qc_roll``)."""
    for b in holders:
        p, slot = _at(s, b)
        s = s._replace(item=s.item.at[p, slot].set(names.item("Quick Claw")))
        s = s._replace(qc_roll=s.qc_roll.at[b].set(1))
    return s


# The enumerator's patterns of one call (policy._enumerated: READ_DRAWS[bit] at the slot's reads
# 0 and 1, READ_DRAWS[0] past them; 2 reads per call, decision 10).
_RANK_PATTERNS = np.asarray(READ_DRAWS, np.uint8)[
    np.asarray(list(itertools.product((0, 1), repeat=2)))
]


@jax.jit
def _rank_paths(t, s):
    """Per argument of ``RANK_ARGS`` (attacker 1, defender 0) and per pattern of
    ``_RANK_PATTERNS``: ``(calc_temp, reads, weight, canonical)``, ``canonical`` the pattern's
    draws past its reads all ``READ_DRAWS[0]`` (``policy._canonical``)."""
    base = _rank_ctx(t, s, 1, 0)

    def one(arg, draws):
        script = base.outcome.script_rolls.at[0, :2].set(draws)
        out = c.load_battler_speed_rank(
            base._replace(outcome=base.outcome._replace(script_rolls=script)), arg
        )
        reads = out.reads[0]
        canonical = jnp.all((jnp.arange(2) < reads) | (draws == READ_DRAWS[0]))
        return out.calc_temp, reads, out.weight[0], canonical

    return [
        jax.vmap(lambda d, arg=arg: one(arg, d))(jnp.asarray(_RANK_PATTERNS)) for arg in RANK_ARGS
    ]


RANK_LAW_CASES = [(speeds, alive, trick, ()) for speeds, alive, trick in SPEED_CASES] + [
    ((50, 50, 50, 50), (1, 1, 1, 1), False, (0,)),  # an activated Quick Claw holder in a tie
    ((50, 50, 50, 40), (1, 1, 1, 1), False, (0, 2)),  # two holders
]


@pytest.mark.parametrize("speeds, alive, trick, holders", RANK_LAW_CASES)
def test_speed_rank_patterns_give_the_law(tables, names, speeds, alive, trick, holders):
    # The exact layout read the way the enumerator builds its patterns (READ_DRAWS only, two bits
    # per call): the canonical patterns' path weights sum to 1 and, by rank, give speed_rank_law
    # (re-derived as exact by the brute force, test_speed_rank_law_equals_the_brute_force), with
    # at most 2 reads per call and none when the rank is certain. The call writes the battlers
    # some compare of the sort stored (a lone member of its alive-or-dead class is not).
    t = _tables4(tables, DOUBLES)
    s = _quick_claw(
        t, names, _speed_state(t, names, speeds, alive=alive, trick_room=trick), holders
    )
    for arg, (rank, reads, weight, canonical) in zip(RANK_ARGS, _rank_paths(t, s)):
        law = np.asarray(_law(t, s, _resolve(arg, 1, 0)), np.float64)
        got = np.zeros(4)
        for r, w, ok in zip(
            np.asarray(rank), np.asarray(weight, np.float64), np.asarray(canonical)
        ):
            if ok:
                got[r] += w
        np.testing.assert_allclose(got, law, atol=1e-6)
        assert abs(got.sum() - 1.0) < 1e-6
        reads = np.asarray(reads)
        assert reads.max() <= 2 and (reads.max() == 0) == (law.max() == 1.0)
    classes = [sum(a == alive[x] for a in alive) for x in range(4)]
    written = c.load_battler_speed_rank(_rank_ctx(t, s, 1, 0), Battler.ATTACKER).written
    assert written.tolist() == [n > 1 for n in classes]


def test_speed_rank_sampled_is_the_law(tables, names):
    # A sampled outcome (raw calc_draws, a uniform u8 per read): over the 256 first draws the
    # ranks have exactly speed_rank_law's frequencies (the second draw only counts), on a law of
    # three ranks (speeds 50, 50, 50, 40: [1/4, 3/8, 3/8, 0] for battler 0).
    t = _tables4(tables, DOUBLES)
    s = _speed_state(t, names, (50, 50, 50, 40))
    base = _rank_ctx(t, s, 1, 0)
    base = base._replace(
        outcome=base.outcome._replace(calc_draws=jnp.zeros((MOVES, 8, 1), jnp.uint16))
    )
    law = np.asarray(_law(t, s, 0), np.float64)
    assert law.tolist() == [0.25, 0.375, 0.375, 0.0]

    @jax.jit
    def sweep(base):
        def one(r):
            script = base.outcome.script_rolls.at[0, 0].set(r.astype(jnp.uint8))
            ctx = base._replace(outcome=base.outcome._replace(script_rolls=script))
            return c.load_battler_speed_rank(ctx, Battler.DEFENDER)

        return jax.vmap(one)(jnp.arange(256))

    out = sweep(base)
    ranks = np.asarray(out.calc_temp)
    np.testing.assert_allclose(np.bincount(ranks, minlength=4) / 256, law, atol=0, rtol=0)
    weight = np.asarray(out.weight[:, 0], np.float64)
    np.testing.assert_allclose(weight, law[ranks], rtol=1e-6)


@pytest.mark.parametrize("speeds, alive", ONE_READ_CASES + [((50, 50, 50, 40), (1, 1, 1, 1))])
def test_speed_rank_loose_law_equals_the_exact_reads(tables, names, speeds, alive):
    # The loose oracle draws the same two bits (SPEED_RANK): the pair, then which of it, at one of
    # two indices by pair, each live only in its pair; over every pattern, each live read weighing
    # its p, it gives speed_rank_law.
    t = _tables4(tables, DOUBLES)
    s = _speed_state(t, names, speeds, alive=alive)
    all_bits = np.zeros((8, MOVES, c.LOOSE_READS), np.int8)
    all_bits[:, 0, :3] = list(itertools.product((0, 1), repeat=3))
    for arg in (Battler.ATTACKER, Battler.DEFENDER, Battler.DEFENDER_PARTNER):
        b = _resolve(arg, 1, 0)
        base = _rank_ctx(t, s, 1, 0, loose_bits=all_bits[0])

        @jax.jit
        def sweep(bits, base=base, arg=arg):
            def one(grid):
                return c.load_battler_speed_rank(
                    base._replace(loose=base.loose._replace(bits=grid)), arg
                )

            return jax.vmap(one)(bits)

        out = sweep(jnp.asarray(all_bits))
        assert np.asarray(out.loose.count[:, 0]).tolist() == [3] * 8
        kinds = set(np.asarray(out.loose.kind[:, 0, :3]).ravel().tolist())
        assert kinds == {int(c.ReadKind.SPEED_RANK)}
        live = np.asarray(out.loose.live[:, 0, :3])
        p = np.asarray(out.loose.p[:, 0, :3], np.float64)
        bits = all_bits[:, 0, :3].astype(bool)
        assert live.sum(axis=1).max() <= 2
        canonical = ~np.any(bits & ~live, axis=1)  # a bit past the live reads repeats a shorter one
        law = np.zeros(4)
        for k in np.flatnonzero(canonical):
            w = np.prod(np.where(live[k], np.where(bits[k], p[k], 1 - p[k]), 1.0))
            law[int(out.calc_temp[k])] += w
        np.testing.assert_allclose(law, np.asarray(_law(t, s, b), np.float64), atol=1e-6)


def test_speed_rank_certain_and_masked(tables, names):
    t = _tables4(tables, DOUBLES)
    # distinct speeds: no read, every battler written, calcTemp the position
    s = _speed_state(t, names, (10, 20, 30, 40))
    out = c.load_battler_speed_rank(_rank_ctx(t, s, 1, 0), Battler.ATTACKER)
    assert int(out.calc_temp) == 2 and int(out.reads[0]) == 0 and float(out.weight[0]) == 1.0
    assert out.written.tolist() == [True] * 4 and int(out.first_cmp[0]) == int(out.flag)
    # a dead battler: its compares with the living are short-circuited, so it is not stored
    dead = _speed_state(t, names, (10, 20, 30, 40), alive=(1, 1, 0, 1))
    out = c.load_battler_speed_rank(_rank_ctx(t, dead, 1, 0), Battler.ATTACKER_PARTNER)
    assert out.written.tolist() == [True, True, False, True] and int(out.calc_temp) == 0
    out = c.load_battler_speed_rank(_rank_ctx(t, dead, 1, 2), Battler.DEFENDER)
    assert int(out.calc_temp) == 3  # the dead battler 2 is last
    # a masked call (not live) writes nothing
    off = _rank_ctx(t, s, 1, 0)._replace(live=jnp.asarray(False))
    out = c.load_battler_speed_rank(off, Battler.ATTACKER)
    assert not bool(out.written.any()) and int(out.first_cmp[0]) == c.NO_COMPARE


def test_speed_compare_record_gate_b4(tables, names):
    # A unit test of the record gate of compare_full: IfSpeedCompare* writes the two compared
    # battlers only when both or neither have HP (battle_lib.c:1204-1211 return first; :1372-1373
    # store). TrainerAI_MainDoubles never evaluates a defender without HP (trainer_ai.c:362-367),
    # so the fainted defender below is not a state the script reaches; the gate matters for
    # LoadBattlerSpeedRank, whose sort compares all four battlers. A Gyro Ball estimate that
    # follows a gated compare still reads the old speed_rec.
    t = _tables4(tables, DOUBLES)
    speeds = (50, 40, 30, 60)
    s = _speed_state(t, names, speeds)
    s = with_moves(s, names, 1, ("Gyro Ball", "Tackle", "Earthquake", "Curse"))
    s = s._replace(speed_rec=jnp.asarray([5, 7, 9, 11], s.speed_rec.dtype))  # not the speeds
    assert [int(x) for x in c.current_speeds(t, s)] == list(speeds)
    gyro = 0

    def estimate(rec, d):
        state = s._replace(speed_rec=jnp.asarray(rec, s.speed_rec.dtype))
        return int(
            c.estimate(t, state, c.AI_GYRO_BALL, c.SIM_ROLL_MAX, atk_side=1, def_side=d).value[0]
        )

    # alive defender: both entries written, Gyro Ball reads the current speeds
    ctx = _rank_ctx(t, s, 1, 0)
    out = c.if_speed_compare_equal_to(ctx, SLOWER)
    assert out.written.tolist() == [True, True, False, False]
    assert int(out.first_cmp[0]) == int(out.flag)
    assert int(c.seen_damage(out).value[gyro, 0]) == estimate([50, 40, 9, 11], 0)
    assert int(c.seen_damage(ctx).value[gyro, 0]) == estimate([5, 7, 9, 11], 0)
    assert estimate([50, 40, 9, 11], 0) != estimate([5, 7, 9, 11], 0)
    # the mixed patterns: only the attacker's entry (1) or only the target's (3) written
    for written in ([False, True, False, False], [False, False, False, True]):
        part = _rank_ctx(t, s, 1, 3)._replace(written=jnp.asarray(written))
        rec = [5, 40 if written[1] else 7, 9, 60 if written[3] else 11]
        assert int(c.seen_damage(part).value[gyro, 0]) == estimate(rec, 3), written
    # a fainted, unreplaced partner (battler 3: its whole party is down) as the compared battler
    gone = s._replace(hp=s.hp.at[1].set(0).at[1, 0].set(int(s.hp[1, 0])))
    gone = with_hp(gone, 3, 0)
    assert bool(c.absent_now(gone, 3))
    ctx = _rank_ctx(t, gone, 1, 3)
    out = c.if_speed_compare_equal_to(ctx, SLOWER)
    assert not bool(out.written.any()) and int(out.first_cmp[0]) == c.NO_COMPARE
    assert int(c.seen_damage(out).value[gyro, 0]) == int(ctx.damage.value[gyro, 0])
    assert int(c.seen_damage(out).value[gyro, 0]) == estimate([5, 7, 9, 11], 3)
    # the compare with a living battler still writes (control)
    ok = c.if_speed_compare_equal_to(_rank_ctx(t, gone, 1, 2), SLOWER)
    assert ok.written.tolist() == [False, True, True, False]


def test_speed_compare_b4_results(tables, names):
    # IfSpeedCompareEqualTo at B = 4 by (attacker, defender): SLOWER never reads the tie; FASTER
    # and TIE at equal speed are one read of probability 1/2
    t = _tables4(tables, DOUBLES)
    s = _speed_state(t, names, (50, 40, 30, 40))
    for a, d, slower in ((1, 0, True), (1, 2, False), (3, 0, True), (1, 3, False)):
        out = c.if_speed_compare_equal_to(_rank_ctx(t, s, a, d), SLOWER)
        assert bool(out.jump) == slower, (a, d)
        assert int(out.reads[0]) == 0
    for a, d in ((1, 3), (3, 1)):
        for r, tie in ((READ_DRAWS[1], True), (READ_DRAWS[0], False)):
            ctx = _rank_ctx(t, s, a, d, rolls=(r,))
            assert bool(c.if_speed_compare_equal_to(ctx, TIE).jump) == tie
            assert bool(c.if_speed_compare_equal_to(ctx, FASTER).jump) == (not tie)
            assert float(c.if_speed_compare_equal_to(ctx, TIE).weight[0]) == 0.5


def _traced_sweep(t, s, d):
    """The changed commands of E4-A W1 at attacker 1 and defender ``d`` (traced under vmap): the
    role commands by argument, the context's damage and Gyro Ball pattern pick, a traced
    ``def_side`` estimate, ``defender_ability`` and the AI record."""
    ctx = c.at_slot(c.new_context(t, s, _slot_outcome((READ_DRAWS[1], READ_DRAWS[0])), 1, d), 0)
    out = {"partner": c.if_target_is_partner(ctx).jump, "seen": ctx.seen_moves}
    for arg in ARGS:
        n = arg.name
        out[f"ability_{n}"] = c.load_battler_ability(ctx, arg).calc_temp
        out[f"item_{n}"] = c.load_held_item(ctx, arg).calc_temp
        out[f"turns_{n}"] = c.load_battler_turn_count(ctx, arg).calc_temp
        out[f"stages_{n}"] = c.sum_positive_stat_stages(ctx, arg).calc_temp
        out[f"alive_{n}"] = c.count_alive_party_battlers(ctx, arg).calc_temp
        out[f"wounded_{n}"] = c.if_any_party_member_is_wounded(ctx, arg).jump
        out[f"prev_{n}"] = c.load_battler_previous_move(ctx, arg).calc_temp
        out[f"flash_{n}"] = c.if_activated_flash_fire(ctx, arg).jump
        out[f"gender_{n}"] = c.load_gender(ctx, arg).calc_temp
        out[f"hp_{n}"] = c.if_hp_percent_less_than(ctx, arg, 50).jump
        rank = c.load_battler_speed_rank(ctx, arg)
        out[f"rank_{n}"] = jnp.stack([rank.calc_temp, rank.reads[0]])
        if arg in (Battler.ATTACKER_PARTNER, Battler.DEFENDER_PARTNER):
            out[f"fainted_{n}"] = c.if_battler_fainted(ctx, arg).jump
    for value in (FASTER, SLOWER, TIE):
        cmp = c.if_speed_compare_equal_to(ctx, value)
        out[f"cmp_{value}"] = jnp.concatenate([cmp.jump[None], cmp.written])
    out["damage"] = ctx.damage.value
    for written in itertools.product((False, True), repeat=4):  # the Gyro Ball pattern pick
        part = ctx._replace(written=jnp.asarray(written))
        out[f"gyro_{written}"] = c.seen_damage(part).value
    out["estimate"] = c.estimate(t, s, c.AI_GYRO_BALL, c.SIM_ROLL_MAX, atk_side=1, def_side=d).value
    out["def_ability"] = jnp.stack([c.defender_ability(t, s, g, d) for g in (0, 1)])
    out["record"] = ai_record_moves(s, d)
    return out


def test_traced_defender_jit_vmap_equals_eager(world, tables, names):
    # One jit + vmap over the traced defender in {0, 2, 3} at B = 4 equals the eager result of each
    # defender, for every changed command (a Python branch on the defender would fail to trace or
    # give one target's value to all).
    t = _tables4(tables, world["fmt"])
    s = with_moves(world["s"], names, 1, ("Gyro Ball", "Tackle", "Earthquake", "Curse"))
    targets = (0, 2, 3)
    traced = jax.jit(jax.vmap(lambda d: _traced_sweep(t, s, d)))(jnp.asarray(targets))
    for i, d in enumerate(targets):
        eager = _traced_sweep(t, s, d)
        for key, want in eager.items():
            np.testing.assert_array_equal(
                np.asarray(traced[key][i]), np.asarray(want), err_msg=f"{key} d={d}"
            )
