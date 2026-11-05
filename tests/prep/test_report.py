"""The prep report: setup card, costed steps, odds ladder, render and run record (plan 4c
Task 9). Results are fabricated over fixture candidates; no search runs."""

import json
from dataclasses import replace

from nuzline.engine.ids import Status
from nuzline.prep.problem import Resources
from nuzline.prep.report import cost_line, ladder_lines, prep_steps, record, render, setup_card
from nuzline.prep.screen import ScreenStats
from nuzline.prep.search import (
    Finalist,
    PrepResult,
    PrepSettings,
    Rung,
    SearchStats,
    setup_label,
)
from nuzline.prep.setups import Setup
from nuzline.prep.variants import LEVEL_UP, RELEARNER, Member, MoveChange
from nuzline.search import deaths
from nuzline.search.stage_b import ActionValue, PlanNode, SolveResult, SolveSettings
from tests.engine.helpers import sample_party
from tests.prep.fixtures import candidate, problem

LEFT_OUT = "party slot 5 ABRA: left out: Teleport (move effect not implemented)"


def _spec(names, species, moves, **kw):
    moves = tuple(names.move(m) for m in moves)
    moves = (*moves, *([0] * (4 - len(moves))))
    return replace(sample_party(names)[0], species=names.species(species), moves=moves, **kw)


def fixture(tables, names):
    """Boxed Kricketot (Lv5 -> 16, Bug Bite on levelling), party Furret, Monferno, Zubat; one
    Heart Scale, TM10. Returns ``(problem, setup)``: Zubat is left in the party."""
    bite, hidden, fury = (names.move(m) for m in ("Bug Bite", "Hidden Power", "Fury Swipes"))
    tm10 = names.item("TM10")
    kricketot = replace(
        candidate(names, _spec(names, "Kricketot", ("Growl", "Bide"), level=16), 0, candies=11),
        location=("box", 0, 1),
        label="box 1 slot 1 KRICKETOT",
    )
    furret = candidate(
        names,
        _spec(
            names,
            "Furret",
            ("Scratch", "Foresight", "Defense Curl", "Quick Attack"),
            ability_slot=1,
            stats=(57, 40, 40, 40, 40, 40),
        ),
        1,
    )
    monferno = candidate(names, replace(sample_party(names)[0], stats=(48, 30, 30, 30, 30, 30)), 2)
    zubat = candidate(names, _spec(names, "Zubat", ("Leech Life",)), 3)
    prob = problem(
        tables,
        names,
        [kricketot, furret, monferno, zubat],
        resources=Resources(1, ((tm10, 1),), (), ()),
        levelled=[((16, bite),), (), (), ()],
    )
    setup = Setup(
        (
            Member(0, (MoveChange(2, bite, LEVEL_UP),)),
            Member(1, (MoveChange(1, fury, RELEARNER), MoveChange(2, hidden, tm10)), hp=40),
            Member(2, hp=16),
        )
    )
    return prob, setup


def test_same_names_carry_their_location(tables, names):
    """Two un-nicknamed Pokemon of one species: every step but Deposit/Withdraw (which print
    the location already) names which one."""
    a = candidate(names, _spec(names, "Geodude", ("Tackle",)), 0)
    b = replace(
        candidate(names, _spec(names, "Geodude", ("Tackle",), level=10), 1, candies=4),
        location=("box", 0, 0),
        label="box 1 slot 0 GEODUDE",
    )
    prob = problem(tables, names, [a, b], resources=Resources(0, (), (), ()), levelled=[(), ()])
    steps = prep_steps(prob, names, Setup((Member(1), Member(0))))
    assert "Withdraw GEODUDE (box 1 slot 0)." in steps
    assert any(t.startswith("Rare Candy x4 on GEODUDE (box 1 slot 0) ") for t in steps)
    assert steps[-1] == ("Party order: GEODUDE (box 1 slot 0) first, then GEODUDE (party slot 0).")


def test_prep_steps_text(tables, names):
    prob, setup = fixture(tables, names)
    assert prep_steps(prob, names, setup) == [
        "Deposit ZUBAT (party slot 3).",
        "Withdraw KRICKETOT (box 1 slot 1).",
        "Rare Candy x11 on KRICKETOT (Lv5 -> 16), cancelling any evolution (press B), then heal at a Pokemon Center.",
        "KRICKETOT learns Bug Bite at Lv16 while levelling.",
        "Move Relearner (1 Heart Scale): FURRET forgets Foresight, learns Fury Swipes.",
        "TM10 (Hidden Power) on FURRET: forgets Defense Curl.",
        "Bring FURRET to 40/57 HP (at most 70 % keeps the trainer AI's high-HP check off).",
        "Bring MONFERNO to 16/48 HP (at most 1/3: Blaze active).",
        "Party order: KRICKETOT first, then FURRET, MONFERNO.",
    ]


def test_prep_steps_items_status_and_alone(tables, names):
    prob, _ = fixture(tables, names)
    oran = names.item("Oran Berry")
    held = names.item("Oran Berry")  # a bag item and the item to take off are told apart
    spec = replace(prob.candidates[2].spec, item=names.item("Sitrus Berry"))
    cands = list(prob.candidates)
    cands[2] = replace(cands[2], spec=spec)
    prob = replace(prob, candidates=tuple(cands), resources=Resources(1, (), (), ((held, 1),)))
    lead = Member(1, item=oran, status=int(Status.POISON))
    steps = prep_steps(prob, names, Setup((lead, Member(2, item=-1))))
    assert "Give FURRET the Oran Berry." in steps
    assert "Take the Sitrus Berry from MONFERNO." in steps
    assert "Poison FURRET before the fight." in steps
    assert steps.index("Give FURRET the Oran Berry.") < steps.index(
        "Poison FURRET before the fight."
    )
    assert steps[-1] == "Party order: FURRET first, then MONFERNO."
    assert prep_steps(prob, names, Setup((Member(2),)))[-1] == "Party order: MONFERNO alone."


def test_cost_line(tables, names):
    prob, setup = fixture(tables, names)
    assert cost_line(prob, names, setup) == "Cost: 11 Rare Candies, 1 Heart Scale, TM10"
    members = list(setup.members)
    members[1] = replace(members[1], moves=members[1].moves[1:])
    assert cost_line(prob, names, Setup(tuple(members))) == "Cost: 11 Rare Candies, TM10"
    assert cost_line(prob, names, Setup((Member(2), Member(1)))) == "Cost: none"


def test_setup_card(tables, names):
    prob, setup = fixture(tables, names)
    card = setup_card(prob, names, setup)
    assert len(card) == 3
    assert card[0].startswith("* KRICKETOT (KRICKETOT) Lv16")
    assert card[1].startswith("  FURRET (FURRET) Lv")
    assert "HP 40/57" in card[1] and "HP 16/48" in card[2]
    assert "Bug Bite" in card[0] and "Hidden Power" in card[1] and "Fury Swipes" in card[1]
    assert "Foresight" not in card[1] and "healthy" in card[0]


def make_result(prob, setup, *, plan=True, other=None, zero=False, stopped=None, k=0):
    """A fabricated result: rung ``j`` at 0.85, 0.9, 0.95, 0.99 (rung 0 at 0 if ``zero``),
    rung 2's best setup ``other``, rung ``k`` from the plan."""
    rungs = []
    for j, p in enumerate((0.85, 0.9, 0.95, 0.99)):
        p = 0.0 if zero and j == 0 else p
        best = (other or setup) if j == 2 else setup
        source = "plan" if plan and j == k else "screen"
        rungs.append(Rung(j, best, p, p - 0.015, p + 0.014, source))
    dist = deaths.DeathDistribution.from_counts(
        [0 if zero else 85, 5, 5, 2, 1, 0, 1, 1]  # whole percents of 100 playouts
    )
    solve, finalists = None, ()
    if plan:
        action = ActionValue(
            0, "Tackle", 0.0 if zero else 0.85, 0.8, 0.9, 0.8, 0.0, 0.0, 0.0, False,
            0, 0.0, 0.0, dist,
        )  # fmt: skip
        node = PlanNode(None, 1.0, 0, {}, {}, "decision")
        solve = SolveResult(
            [action], {}, 1, node, 100, 0.0, SolveSettings(), leaf_budget=10, deaths_allowed=k
        )
        finalists = (Finalist(setup, solve),)
    return PrepResult(
        chosen=setup,
        screen=ScreenStats((0 if zero else 17, 1, 1, 0, 0, 0, 1, 0), 3, 5.0, 6.0, False),
        plan=solve,
        ladder=tuple(rungs),
        distribution=dist,
        finalists=finalists,
        excluded=(LEFT_OUT,),
        pruned=((Member(1), Member(1, hp=40)),),
        stats=SearchStats(40, 4000, 3, 2, 4, "finalists", stopped, 12.5),
    )


def test_ladder_lines(tables, names):
    prob, setup = fixture(tables, names)
    other = Setup((setup.members[1], setup.members[0]))
    lines = ladder_lines(make_result(prob, setup, other=other), prob, names)
    assert lines[:5] == [
        "Odds ladder (rung 0 planned):",
        "  deaths <= 0: 85.0 % [83.5, 86.4]  plan",
        "  deaths <= 1: 90.0 % [88.5, 91.4]  screen",
        "  deaths <= 2: 95.0 % [93.5, 96.4]  screen",
        "  (best for this rung: FURRET (Fury Swipes, Hidden Power, 40 HP), KRICKETOT (Bug Bite))",
    ]
    assert lines[5].startswith("  deaths <= 3: 99.0 %")
    assert sum("best for this rung" in line for line in lines) == 1
    assert len(lines) == 7
    assert lines[-1] == (
        "Deaths (plan): 0: 85.0 %  1: 5.0 %  2: 5.0 %  3+: 3.0 %  whiteout: 1.0 %  "
        "unfinished: 1.0 %  protected lost: 0.0 %"
    )


def test_render_with_plan(tables, names):
    prob, setup = fixture(tables, names)
    lines = render(make_result(prob, setup), prob, names)
    text = "\n".join(lines)
    assert lines[1] == "Setup:" and "Prep steps:" in lines
    assert "  1. Deposit ZUBAT (party slot 3)." in lines
    assert "Cost: 11 Rare Candies, 1 Heart Scale, TM10" in lines
    assert "Plan:" in lines and "No deathless line" not in text
    assert any(line.startswith("Search: 40 setups screened, 3 team rounds") for line in lines)
    assert lines[-2:] == ["Left out:", "  " + LEFT_OUT]
    assert text.isascii()


def test_report_without_a_deathless_line(tables, names):
    prob, setup = fixture(tables, names)
    prob = replace(prob, deaths=1)
    lines = render(make_result(prob, setup, zero=True, k=1), prob, names)
    assert "No deathless line found: every setup's 0-death chance is 0 %." in lines
    ladder = [line for line in lines if line.startswith("  deaths <=")]
    assert len(ladder) == 4 and ladder[0].startswith("  deaths <= 0: 0.0 %")
    assert "Odds ladder (rung 1 planned):" in lines
    assert "\n".join(lines).isascii()


def test_screen_only_report(tables, names):
    prob, setup = fixture(tables, names)
    lines = render(make_result(prob, setup, plan=False, stopped="budget"), prob, names)
    assert "screen only, no plan (budget)" in lines
    assert "Plan:" not in lines
    assert any(line.startswith("Deaths (screen): 0: 85.0 %") for line in lines)


def test_record_is_json(tables, names):
    prob, setup = fixture(tables, names)
    prob = replace(prob, protected=frozenset({1}))
    rec = record(
        make_result(prob, setup), prob, names, PrepSettings(), 7, "tv", reserved=("Oran Berry",)
    )
    assert json.loads(json.dumps(rec)) == rec
    assert set(rec) == {
        "trainer_id", "trainer", "cap", "deaths", "protected", "reserved", "settings", "seed",
        "fingerprint", "setup", "prep_steps", "cost", "ladder", "distribution", "plan",
        "finalists", "stats", "excluded", "pruned", "cheaper_alternative",
    }  # fmt: skip
    assert rec["protected"] == ["party slot 1 FURRET"] and rec["reserved"] == ["Oran Berry"]
    assert rec["setup"][0]["label"] == "box 1 slot 1 KRICKETOT"
    assert rec["setup"][1]["moves"][1:3] == ["Fury Swipes", "Hidden Power"]
    assert rec["cost"]["text"] == "Cost: 11 Rare Candies, 1 Heart Scale, TM10"
    assert rec["plan"]["deaths_allowed"] == 0 and rec["plan"]["table_version"] == "tv"
    assert len(rec["ladder"]) == 4 and rec["distribution"]["win_0"] == 0.85
    assert rec["finalists"][0]["value"] == 0.85
    assert rec["pruned"][0]["dominated_by"]["hp"] == 40
    assert rec["stats"]["setups_screened"] == 40
    bare = record(make_result(prob, setup, plan=False), prob, names, PrepSettings(), 7, "tv")
    assert bare["plan"] is None and bare["finalists"] == []


def test_cheaper_alternative_line(tables, names):
    prob, setup = fixture(tables, names)
    cheap = Setup((Member(2), Member(1)))
    assert cost_line(prob, names, cheap) == "Cost: none"
    assert "1 Heart Scale" in cost_line(prob, names, setup)
    base = make_result(prob, setup)
    solved = base.finalists[0].result
    top = replace(solved.ranking[0], e_deaths=0.0123)
    other = Finalist(cheap, replace(solved, ranking=[top, *solved.ranking[1:]]))
    result = replace(base, finalists=(*base.finalists, other), cheaper=other)
    lines = render(result, prob, names)
    found = [line for line in lines if line.startswith("Cheaper, tied within noise: ")]
    assert len(found) == 1 and "Cost: none" in found[0]
    assert "0.012 expected deaths" in found[0]
    assert "\n".join(lines).isascii()
    rec = record(result, prob, names, PrepSettings(), 7, "tv")
    assert json.loads(json.dumps(rec)) == rec
    assert rec["cheaper_alternative"]["label"] == setup_label(names, prob, cheap)
    assert rec["cheaper_alternative"]["cost"] == "Cost: none"
    assert rec["cheaper_alternative"]["e_deaths"] == 0.0123
    lines = render(base, prob, names)
    assert not any(line.startswith("Cheaper, tied within noise: ") for line in lines)
    assert record(base, prob, names, PrepSettings(), 7, "tv")["cheaper_alternative"] is None


def test_prep_steps_deposits_left_out_party(tables, names):
    prob, setup = fixture(tables, names)
    prob = replace(prob, left_out_party=((1, "Egg"), (4, "GEODUDE")))
    steps = prep_steps(prob, names, setup)
    assert steps[:4] == [
        "Deposit Egg (party slot 1).",
        "Deposit ZUBAT (party slot 3).",
        "Deposit GEODUDE (party slot 4).",
        "Withdraw KRICKETOT (box 1 slot 1).",
    ]
