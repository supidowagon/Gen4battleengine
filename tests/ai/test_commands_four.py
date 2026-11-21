"""E4-A-W1 steps 1-2: the battler context and the role commands at B = 4 (rules A-4, A-10 s6, A-14)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai.commands import AI_HAVE, AI_NOT_HAVE, AI_UNKNOWN, Battler, LoadType
from nuzline.ai.outcome import READ_DRAWS, no_calc_draws
from nuzline.engine.damage import ai_damage_for_roll, decomp_abilities
from nuzline.engine.format import DOUBLES, MULTI, format_of
from nuzline.engine.ids import Stat, Status, Type
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
    index)."""
    got = json.loads(json.dumps(snapshot(tables, names)))
    want = json.loads(_SNAPSHOT.read_text())
    # the two LoadType members added since (DEFENDER_PARTNER_TYPE_1 / _2) have no old row
    assert set(got) - set(want) == {k for k in got if "load_type_DEFENDER_PARTNER_TYPE" in k}
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
