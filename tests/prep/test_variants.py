"""Member variants, costs, HP regions and the prune (plan 4c Task 4)."""

import ast
import re
from dataclasses import replace
from pathlib import Path

import jax

from nuzline.ai import commands
from nuzline.ai.commands import SIM_ROLL_MAX, USE_MAX_DAMAGE, ai_damage
from nuzline.ai.outcome import sample_ai_outcome
from nuzline.engine.ids import Status
from nuzline.engine.state import make_state
from nuzline.prep.problem import Resources
from nuzline.prep.variants import (
    AI_DEFENDER_HP_CHECKS,
    AI_HP_READERS,
    DEFAULT_WEIGHTS,
    LEVEL_UP,
    RELEARNER,
    Cost,
    Member,
    MoveChange,
    ai_kill_damage,
    base_member,
    designated_change,
    hp_regions,
    member_cost,
    member_spec,
    move_changes,
    prestatus_options,
    prune,
    single_changes,
)
from tests.engine.helpers import roark_party, sample_party
from tests.prep.fixtures import candidate, problem

AI_DIR = Path(__file__).resolve().parents[2] / "src" / "nuzline" / "ai"


def _spec(names, species, moves=(), **kw):
    moves = tuple(names.move(m) if isinstance(m, str) else m for m in moves)
    moves = (*moves, *([0] * (4 - len(moves))))
    return replace(sample_party(names)[0], species=names.species(species), moves=moves, **kw)


def _furret(names, moves=("Scratch", "Foresight", "Defense Curl", "Quick Attack"), **kw):
    return _spec(names, "Furret", moves, ability_slot=1, **kw)


def test_hp_regions(tables, names):
    monferno = sample_party(names)[0]
    assert hp_regions(tables, monferno, 48) == (48, 34, 16)
    assert hp_regions(tables, _furret(names), 57) == (57, 40)
    assert hp_regions(tables, monferno, 3) == (3, 2, 1)
    shedinja = _spec(names, "Shedinja", ("Tackle",))
    assert hp_regions(tables, shedinja, 1) == (1,)


def test_hp_regions_include_ai_kill_checks(tables, names):
    spec = sample_party(names)[0]
    party = roark_party(names)
    rock_throw = names.move("Rock Throw")
    # Roark's real Cranidos has no Rock Throw; Geodude (slot 0) and Onix do
    state = make_state(tables, [spec], party, leads=(0, 0))
    d = int(ai_damage(tables, state, rock_throw, SIM_ROLL_MAX))
    max_hp = int(state.max_hp[0, 0])
    assert 1 <= d < max_hp
    kill = ai_kill_damage(tables, spec, party)
    assert d in kill and list(kill) == sorted(set(kill)) and 0 not in kill
    assert d in hp_regions(tables, spec, max_hp, kill)

    def jumps(hp):
        s = state._replace(hp=state.hp.at[0, 0].set(hp))
        ctx = commands.new_context(tables, s, sample_ai_outcome(jax.random.PRNGKey(0), tables, s))
        ctx = ctx._replace(move=jax.numpy.asarray(rock_throw, jax.numpy.int32))
        return bool(commands.if_current_move_kills(ctx, USE_MAX_DAMAGE).jump)

    assert jumps(d)
    assert not jumps(d + 1)


def test_ai_hp_checks_pinned():
    sources = {p: p.read_text(encoding="utf-8") for p in sorted(AI_DIR.glob("*.py"))}
    found = set()
    for text in sources.values():
        for kind, percent in re.findall(
            r"if_hp_percent_(greater|less)_than\(ctx, _D, (\d+)\)", text
        ):
            found.add((kind, int(percent)))
    assert found == set(AI_DEFENDER_HP_CHECKS)
    readers = set()
    kill_calls = []
    for text in sources.values():
        tree = ast.parse(text)
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and any(
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Attribute)
                and n.value.attr == "hp"
                for n in ast.walk(fn)
            ):
                readers.add(fn.name)
            if isinstance(fn, ast.Call):
                f = fn.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name == "if_current_move_kills":
                    kill_calls.append(ast.unparse(fn.args[-1]))
    assert readers == set(AI_HP_READERS)
    assert kill_calls and all(arg.endswith("USE_MAX_DAMAGE") for arg in kill_calls)


def test_move_changes(names):
    bite = names.move("Bug Bite")
    kricketot = _spec(names, "Kricketot", ("Growl", "Bide"))
    assert move_changes(kricketot, bite, RELEARNER) == (MoveChange(2, bite, RELEARNER),)
    hidden = names.move("Hidden Power")
    assert move_changes(_furret(names), hidden, 5) == tuple(
        MoveChange(s, hidden, 5) for s in range(4)
    )
    assert move_changes(_furret(names), names.move("Scratch"), RELEARNER) == ()


def test_member_spec_and_cost(tables, names):
    tm10, hidden = names.item("TM10"), names.move("Hidden Power")
    furret = candidate(names, _furret(names, pp=(35, 40, 40, 30)), 0)
    prob = problem(tables, names, [furret], resources=Resources(0, ((tm10, 1),), (), ()))
    member = Member(0, (MoveChange(2, hidden, tm10),), hp=40, status=Status.POISON)
    spec = member_spec(prob, member, tables)
    scratch, foresight, quick = (names.move(m) for m in ("Scratch", "Foresight", "Quick Attack"))
    assert spec.moves == (scratch, foresight, hidden, quick)
    pp = list(furret.spec.pp)
    pp[2] = int(tables.move_pp[hidden])
    assert spec.pp == tuple(pp)
    assert (spec.current_hp, spec.status, spec.status_counter) == (40, Status.POISON, 0)
    assert member_cost(prob, member) == Cost(tms=(tm10,))
    assert member_cost(prob, Member(0, (MoveChange(2, hidden, RELEARNER),))) == Cost(heart_scales=1)
    hm = names.item("HM01")
    hm_prob = problem(tables, names, [furret], resources=Resources(0, (), (hm,), ()))
    assert member_cost(hm_prob, Member(0, (MoveChange(2, hidden, hm),))) == Cost(hm_moves=1)
    assert member_spec(prob, Member(0, item=-1), tables).item == 0
    assert member_spec(prob, Member(0), tables).item == furret.spec.item
    oran = names.item("Oran Berry")
    assert member_spec(prob, Member(0, item=oran), tables).item == oran
    assert member_cost(prob, Member(0, item=oran)) == Cost(held=(oran,))
    assert member_cost(prob, Member(0, item=-1)) == Cost()
    cost = Cost(tms=(tm10,)) + Cost(heart_scales=1)
    assert cost.weighted(DEFAULT_WEIGHTS) == 13


def test_levelled_moves_and_candies(tables, names):
    bite = names.move("Bug Bite")
    boxed = candidate(names, _spec(names, "Kricketot", ("Growl", "Bide"), level=16), 0, candies=11)
    prob = problem(tables, names, [boxed], levelled=[((16, bite),)])
    base = base_member(prob, 0)
    assert base.moves == (MoveChange(2, bite, LEVEL_UP),)
    assert member_cost(prob, base) == Cost(candies=11)
    full = candidate(
        names,
        _spec(names, "Kricketot", ("Growl", "Bide", "Tackle", "Leer"), level=16),
        0,
        candies=11,
    )
    prob = problem(tables, names, [full], levelled=[((16, bite),)])
    assert base_member(prob, 0) == Member(0)
    variants = single_changes(tables, prob, 0, entry_hp=False, items=False)
    assert [v.moves for v in variants] == [(MoveChange(s, bite, LEVEL_UP),) for s in range(4)]
    assert all(member_cost(prob, v) == Cost(candies=11) for v in variants)
    # Without Bug Bite pending the levelled move is not offered twice as a relearner move.
    relearn = candidate(names, full.spec, 0, relearner=(bite,), candies=11)
    prob = problem(
        tables, names, [relearn], levelled=[((16, bite),)], resources=Resources(1, (), (), ())
    )
    assert {m for v in single_changes(tables, prob, 0, entry_hp=False) for m in v.moves} == {
        MoveChange(s, bite, LEVEL_UP) for s in range(4)
    }


def test_fits_respects_quantities(names):
    tm10, hm01 = names.item("TM10"), names.item("HM01")
    one = Resources(1, ((tm10, 1),), (hm01,), ())
    assert Cost(tms=(tm10,)).fits(one)
    assert not Cost(tms=(tm10, tm10)).fits(one)
    assert not Cost(heart_scales=2).fits(one)
    assert Cost(hm_moves=2).fits(one)
    assert not Cost(held=(names.item("Oran Berry"),)).fits(one)


def test_prestatus_options(tables, names):
    declared = (Status.POISON, Status.BURN)
    assert prestatus_options(tables, _spec(names, "Spinarak"), declared) == (Status.BURN,)
    assert prestatus_options(tables, sample_party(names)[0], declared) == (Status.POISON,)
    both = (Status.POISON, Status.PARALYSIS)
    assert prestatus_options(tables, _spec(names, "Hitmonlee", ability_slot=0), both) == (
        Status.POISON,
    )
    assert prestatus_options(tables, _spec(names, "Snorlax", ability_slot=0), both) == (
        Status.PARALYSIS,
    )


def _geodude(names, moves, relearner):
    spec = _spec(names, "Geodude", moves)
    ids = tuple(names.move(m) for m in relearner)
    return candidate(names, spec, 0, relearner=ids)


def test_designated_change(tables, names):
    rp, rt = names.move("Rock Polish"), names.move("Rock Throw")
    cand = _geodude(
        names, ("Tackle", "Defense Curl", "Mud Sport"), ("Rock Polish", "Rock Throw", "Magnitude")
    )
    prob = problem(tables, names, [cand], resources=Resources(1, (), (), ()), levelled=[()])
    assert designated_change(tables, prob, 0) == MoveChange(3, rt, RELEARNER)
    cand = _geodude(
        names, ("Tackle", "Defense Curl", "Mud Sport", "Rock Polish"), ("Rock Throw", "Magnitude")
    )
    prob = problem(tables, names, [cand], resources=Resources(1, (), (), ()), levelled=[()])
    assert designated_change(tables, prob, 0) == MoveChange(1, rt, RELEARNER)
    assert rp  # the status move exists in the tables
    cand = _geodude(names, ("Tackle", "Defense Curl", "Mud Sport", "Rock Polish"), ())
    assert designated_change(tables, problem(tables, names, [cand], levelled=[()]), 0) is None


def test_single_changes_respect_resources(tables, names):
    bite, ember = names.move("Bug Bite"), names.move("Ember")
    full = _spec(names, "Kricketot", ("Growl", "Bide", "Tackle", "Leer"), level=16)
    cand = candidate(names, full, 0, relearner=(ember, names.move("Scratch")), candies=11)
    prob = problem(tables, names, [cand], levelled=[((16, bite),)])
    moves = {m for v in single_changes(tables, prob, 0, entry_hp=False) for m in v.moves}
    assert moves == {MoveChange(s, bite, LEVEL_UP) for s in range(4)}
    prob = problem(
        tables,
        names,
        [cand],
        levelled=[((16, bite),)],
        resources=Resources(1, (), (), ()),
    )
    designated = single_changes(tables, prob, 0, relearner="designated", entry_hp=False)
    relearned = [m for v in designated for m in v.moves if m.source == RELEARNER]
    assert len(relearned) == 1
    assert relearned[0] == designated_change(tables, prob, 0)
    none = single_changes(tables, prob, 0, relearner="none", entry_hp=False)
    assert not [m for v in none for m in v.moves if m.source == RELEARNER]
    with_hp = single_changes(tables, prob, 0, relearner="none")
    assert any(v.hp is not None for v in with_hp)
    assert not any(v.hp is not None for v in single_changes(tables, prob, 0, entry_hp=False))


def test_single_changes_items_and_tms(tables, names):
    oran, sitrus = names.item("Oran Berry"), names.item("Sitrus Berry")
    tm10, hm01 = names.item("TM10"), names.item("HM01")
    hidden = names.move("Hidden Power")
    spec = replace(_furret(names), item=sitrus)
    cand = candidate(names, spec, 0, tms=((tm10, hidden), (hm01, names.move("Cut"))))
    res = Resources(0, ((tm10, 1),), (), ((oran, 1), (sitrus, 1)))
    prob = problem(tables, names, [cand], resources=res)
    variants = single_changes(tables, prob, 0, entry_hp=False)
    assert {v.item for v in variants if v.item} == {-1, oran}
    tm_moves = [m for v in variants for m in v.moves]
    assert {m.source for m in tm_moves} == {tm10}  # the HM is not in the bag
    assert not single_changes(tables, prob, 0, tms=False, items=False, entry_hp=False)


def test_prune_drops_only_strict_dominance(tables, names):
    tackle, scratch, ember, mach = (
        names.move(m) for m in ("Tackle", "Scratch", "Ember", "Mach Punch")
    )
    full = _spec(names, "Kricketot", ("Growl", "Bide", "Leer", "Harden"))
    cand = candidate(names, full, 0, relearner=(tackle, scratch, ember, mach))
    prob = problem(tables, names, [cand], resources=Resources(9, (), (), ()))
    variants = single_changes(tables, prob, 0, entry_hp=False, items=False)
    kept, dropped = prune(tables, prob, variants)
    pairs = {(d.moves[0].move, w.moves[0].move, d.moves[0].slot) for d, w in dropped}
    for slot in range(4):
        assert (tackle, scratch, slot) in pairs
    assert not any(move == ember for move, _, _ in pairs)
    assert not any(move == mach for move, _, _ in pairs)
    assert not any(move == scratch for move, _, _ in pairs)
    assert len(kept) + len(dropped) == len(variants)
    # A TM variant is not dropped by a relearner variant (different cost components).
    tm10 = names.item("TM10")
    mixed = [
        Member(0, (MoveChange(0, tackle, tm10),)),
        Member(0, (MoveChange(0, scratch, RELEARNER),)),
    ]
    tm_prob = problem(tables, names, [cand], resources=Resources(1, ((tm10, 1),), (), ()))
    assert prune(tables, tm_prob, mixed)[1] == []
