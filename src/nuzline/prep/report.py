"""The prep answer in words: the setup card, the costed prep steps, the odds ladder, the plan
and the JSON run record (plan 4c Task 9; spec section 7).

Everything prints ASCII only. ``prep_steps`` is unnumbered (``render`` numbers it); a step is
written for each thing the setup changes, grouped by kind in the order a player does them:
deposits and withdrawals, Rare Candies, moves learned while levelling, the Move Relearner,
TMs and HMs, held items, entry HP, prestatus, then the party order.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.gauntlet import Settings
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import TERRAIN_NAMES, run_terrain
from nuzline.engine.coverage import start_gravity, start_trick_room
from nuzline.engine.ids import Status
from nuzline.engine.state import BattleState, make_state
from nuzline.jaxcache import code_version
from nuzline.prep.checkpoint import _plain, fingerprint
from nuzline.prep.gauntlet import Carry
from nuzline.prep.gauntlet import member_spec as carry_member_spec
from nuzline.prep.growth import Prompt, post_battle_prompts, resolve_post_battle
from nuzline.prep.problem import PrepProblem
from nuzline.prep.screen import ScreenStats, prep_order, scored, screen
from nuzline.prep.search import PrepResult, PrepSettings, setup_label
from nuzline.prep.setups import Setup, setup_cost, setup_specs
from nuzline.prep.variants import (
    _PINCH_ABILITIES,
    LEVEL_UP,
    RELEARNER,
    Member,
    ai_defender_hp_checks,
    ai_kill_damage,
    max_hp_of,
    member_spec,
    percent_cut,
)
from nuzline.search import plan_tree
from nuzline.text import decimals
from nuzline.text import tenths as _tenths

__all__ = [
    "CANNOT_UNDO",
    "PostFight",
    "PostFightAnswer",
    "cost_line",
    "ladder_lines",
    "next_fight_scorer",
    "post_fight_evolution",
    "post_fight_json",
    "post_fight_lines",
    "prep_steps",
    "record",
    "render",
    "setup_card",
]

HM_NOTE = "(an HM move needs the Move Deleter to remove)"


def _shown_name(problem: PrepProblem, names: Names, candidate: int) -> str:
    cand = problem.candidates[candidate]
    return cand.nickname or names.species_names[cand.spec.species]


def _where(location: tuple[str, int, int]) -> str:
    kind, box, slot = location
    return f"party slot {slot}" if kind == "party" else f"box {box + 1} slot {slot}"


def _nick(problem: PrepProblem, names: Names, candidate: int) -> str:
    """The candidate's nickname or species name, with its location when another candidate
    or a left-out party Pokemon shows the same name (``GEODUDE (box 1 slot 0)``), so every
    step names one Pokemon."""
    name = _shown_name(problem, names, candidate)
    others = [
        _shown_name(problem, names, j) for j in range(len(problem.candidates)) if j != candidate
    ]
    others += [who for _, who in problem.left_out_party]
    if name in others:
        return f"{name} ({_where(problem.candidates[candidate].location)})"
    return name


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _status_name(status: int) -> str:
    return Status(status).name.title()


def _hp_why(problem: PrepProblem, names: Names, member: Member, tables: Tables) -> str:
    """Why ``member.hp`` is an HP worth bringing: the check the region's top keeps off or on."""
    spec = problem.candidates[member.candidate].spec
    top = max_hp_of(tables, spec)
    hp = member.hp
    for kind, percent in ai_defender_hp_checks(tables):
        if hp == percent_cut(kind, percent, top) and kind == "greater":
            return f"at most {percent} % keeps the trainer AI's above-{percent} % HP check off"
        if hp == percent_cut(kind, percent, top) and kind == "less":
            return f"below {percent} % turns the trainer AI's below-{percent} % HP check on"
    row = personal_row(tables, spec.species, spec.ability_from)
    ability = int(np.asarray(tables.species_abilities)[row, spec.ability_slot])
    if hp == top // 3 and ability in _PINCH_ABILITIES:
        name = names.ability_names[
            names.species_abilities(spec.species, spec.ability_from)[spec.ability_slot]
        ]
        return f"at most 1/3: {name} active"
    kill = ai_kill_damage(
        tables,
        spec,
        problem.opp,
        problem.start_condition,
        problem.trainer.items,
        gravity=start_gravity(problem.trainer),
        trick_room=start_trick_room(problem.trainer),
    )
    if hp in kill:
        return f"at most {hp}: the trainer AI sees a kill"
    return "a region of the trainer AI's HP checks"


def prep_steps(problem: PrepProblem, names: Names, setup: Setup, tables: Tables) -> list[str]:
    """What to do before the fight, unnumbered, in the order of the module docstring.
    ``tables`` and ``names`` are the problem's game's (every function here takes both)."""
    brought = {m.candidate for m in setup.members}
    cands = problem.candidates
    steps: list[str] = []

    def nick(i: int) -> str:
        return _nick(problem, names, i)

    deposits = [
        (cand.location[2], _shown_name(problem, names, i))
        for i, cand in enumerate(cands)
        if cand.location[0] == "party" and i not in brought
    ]
    deposits += list(problem.left_out_party)
    for slot, who in sorted(deposits, key=lambda d: d[0]):
        steps.append(f"Deposit {who} (party slot {slot}).")
    for m in setup.members:
        kind, box, slot = cands[m.candidate].location
        if kind == "box":
            steps.append(
                f"Withdraw {_shown_name(problem, names, m.candidate)} (box {box + 1} slot {slot})."
            )
    for m in setup.members:
        cand = cands[m.candidate]
        if cand.candies:
            steps.append(
                f"Rare Candy x{cand.candies} on {nick(m.candidate)} "
                f"(Lv{cand.spec.level - cand.candies} -> {problem.cap}), "
                "cancelling any evolution (press B), then heal at a Pokemon Center."
            )
    for source in ("level", "relearner", "tm"):
        for m in setup.members:
            who = nick(m.candidate)
            original = cands[m.candidate].spec.moves
            for change in m.moves:
                old = original[change.slot]
                new = names.move_names[change.move]
                if source == "level" and change.source == LEVEL_UP:
                    level = next(
                        lv for lv, mv in problem.levelled[m.candidate] if mv == change.move
                    )
                    tail = f": forget {names.move_names[old]}" if old else ""
                    steps.append(f"{who} learns {new} at Lv{level} while levelling{tail}.")
                elif source == "relearner" and change.source == RELEARNER:
                    what = (
                        f"forgets {names.move_names[old]}, learns {new}"
                        if old
                        else f"learns {new} into an empty slot"
                    )
                    steps.append(f"Move Relearner (1 Heart Scale): {who} {what}.")
                elif source == "tm" and change.source not in (LEVEL_UP, RELEARNER):
                    item = names.item_names[change.source]
                    tail = f"forgets {names.move_names[old]}" if old else "fills an empty slot"
                    note = f" {HM_NOTE}" if change.source in problem.resources.hms else ""
                    steps.append(f"{item} ({new}) on {who}: {tail}.{note}")
    for m in setup.members:
        item = names.item_names
        if m.item > 0:
            steps.append(f"Give {nick(m.candidate)} the {item[m.item]}.")
        elif m.item < 0 and cands[m.candidate].spec.item:
            steps.append(f"Take the {item[cands[m.candidate].spec.item]} from {nick(m.candidate)}.")
    for m in setup.members:
        if m.hp is not None:
            top = max_hp_of(tables, cands[m.candidate].spec)
            steps.append(
                f"Bring {nick(m.candidate)} to {m.hp}/{top} HP ({_hp_why(problem, names, m, tables)})."
            )
    for m in setup.members:
        if m.status:
            steps.append(f"{_status_name(m.status)} {nick(m.candidate)} before the fight.")
    lead, *rest = (nick(m.candidate) for m in setup.members)
    steps.append(
        f"Party order: {lead} first, then {', '.join(rest)}."
        if rest
        else f"Party order: {lead} alone."
    )
    if problem.field_move:  # ledger G-15: after the last warp onto the map, so no PC between
        move = names.move_names[problem.field_move]
        steps.append(
            f"On the fight's map, after the last warp onto it, use {move} from the party menu "
            "(clears the fog); then walk to the fight without warping."
        )
    if problem.script_weather_warp:  # ledger G-24: the script does not run again
        steps.append(
            "Warp onto the fight's map before the fight (a door, Fly or a PC visit elsewhere): "
            "the warp ends the map script's weather the save shows, and it does not come back."
        )
    if run_terrain(problem.trainer):  # ledger G-18: the tile under the player at the start
        terrain = TERRAIN_NAMES[problem.trainer.terrain]
        steps.append(
            f"Start the fight standing on a tile whose battle terrain is {terrain} (the tile "
            "under the player decides it)."
        )
    return steps


def cost_line(problem: PrepProblem, names: Names, setup: Setup) -> str:
    """``Cost: 11 Rare Candies, 1 Heart Scale, TM10`` or ``Cost: none``."""
    cost = setup_cost(problem, setup)
    parts = []
    if cost.candies:
        parts.append(_plural(cost.candies, "Rare Candy", "Rare Candies"))
    if cost.heart_scales:
        parts.append(_plural(cost.heart_scales, "Heart Scale", "Heart Scales"))
    parts.extend(names.item_names[t] for t in cost.tms)
    parts.extend(
        names.item_names[c.source]
        for m in setup.members
        for c in m.moves
        if c.source in problem.resources.hms
    )
    parts.extend(names.item_names[h] for h in cost.held)
    return "Cost: " + (", ".join(parts) if parts else "none")


def _moves_text(names: Names, moves: Sequence[int]) -> str:
    return ", ".join(names.move_names[m] for m in moves if m) or "no moves"


def setup_card(problem: PrepProblem, names: Names, setup: Setup, tables: Tables) -> list[str]:
    """One line per member, the lead marked ``*``: nickname, species, level, ability, item, HP,
    status and the four moves."""
    out = []
    for i, m in enumerate(setup.members):
        spec = member_spec(problem, m, tables)
        top = max_hp_of(tables, spec)
        species = names.species_names[spec.species]
        ability = names.ability_names[
            names.species_abilities(spec.species, spec.ability_from)[spec.ability_slot]
        ]
        item = names.item_names[spec.item] if spec.item else "no item"
        hp = top if spec.current_hp is None else spec.current_hp
        status = _status_name(spec.status) if spec.status else "healthy"
        out.append(
            f"{'*' if i == 0 else ' '} {_nick(problem, names, m.candidate)} ({species}) "
            f"Lv{spec.level}  {ability}  {item}  HP {hp}/{top}  {status}  "
            f"{_moves_text(names, spec.moves)}"
        )
    return out


def _team_text(problem: PrepProblem, names: Names, setup: Setup) -> str:
    """Lead first, each with the moves, item, HP and status it was changed to."""
    parts = []
    for m in setup.members:
        changes = [names.move_names[c.move] for c in m.moves]
        if m.item > 0:
            changes.append(names.item_names[m.item])
        elif m.item < 0:
            changes.append("item off")
        if m.hp is not None:
            changes.append(f"{m.hp} HP")
        if m.status:
            changes.append(_status_name(m.status))
        nick = _nick(problem, names, m.candidate)
        parts.append(f"{nick} ({', '.join(changes)})" if changes else nick)
    return ", ".join(parts)


def _distribution_line(result: PrepResult) -> str:
    p = result.distribution.probs
    source = "plan" if result.plan is not None else "screen"
    cells = (
        ("0", p[0]),
        ("1", p[1]),
        ("2", p[2]),
        ("3+", p[3] + p[4]),
        ("whiteout", p[6]),
        ("unfinished", p[7]),
        ("protected lost", p[5]),
    )
    # The 0-deaths cell is a win chance, rounded down; every other cell is a loss, rounded up.
    return f"Deaths ({source}): " + "  ".join(f"{n}: {_tenths(v, up=n != '0')} %" for n, v in cells)


def ladder_lines(result: PrepResult, problem: PrepProblem, names: Names) -> list[str]:
    """The odds ladder for 0-3 deaths with each rung's source, the best setup of a rung that
    differs from the chosen one, and the death distribution."""
    out = [f"Odds ladder (rung {problem.deaths} planned):"]
    for rung in result.ladder:
        out.append(
            f"  deaths <= {rung.k}: {_percent(rung.p)} "
            f"[{_tenths(rung.ci_low)}, {_tenths(rung.ci_high)}]  {rung.source}"
        )
        if rung.setup != result.chosen:
            out.append(f"  (best for this rung: {_team_text(problem, names, rung.setup)})")
    out.append(_distribution_line(result))
    return out


def _root(problem: PrepProblem, setup: Setup, tables: Tables) -> BattleState:
    return make_state(
        tables,
        setup_specs(problem, setup, tables),
        list(problem.opp),
        problem.start_condition,
        problem.trainer.items,
        terrain=problem.trainer.terrain,
        gravity=start_gravity(problem.trainer),
        trick_room=start_trick_room(problem.trainer),
        chatot_cry=problem.chatot_cry,
    )


def _no_deathless(result: PrepResult) -> bool:
    """Whether no rung-0 number anywhere in the answer is above 0."""
    return (
        result.ladder[0].p == 0
        and result.screen.rung(0) == 0
        and result.distribution.rung(0) == 0
        and all(
            f.result.ranking[0].estimate == 0
            for f in result.finalists
            if f.result.deaths_allowed == 0
        )
    )


def _ascii(line: str) -> str:
    return line.encode("ascii", "replace").decode("ascii")


def _cheaper_lines(result: PrepResult, problem: PrepProblem, names: Names) -> list[str]:
    """The display-only "Cheaper, tied within noise" line (``PrepResult.cheaper``), or none."""
    f = result.cheaper
    if f is None:
        return []
    top = f.result.ranking[0]
    line = (
        f"Cheaper, tied within noise: {setup_label(names, problem, f.setup)} "
        f"{_tenths(top.estimate)}% ({_tenths(top.ci_low)}%-{_tenths(top.ci_high)}%), "
        f"{decimals(top.e_deaths, 3, up=True)} expected deaths,{cost_line(problem, names, f.setup)}"
    )
    return [line]


def _cheaper_json(result: PrepResult, problem: PrepProblem, names: Names) -> dict | None:
    f = result.cheaper
    if f is None:
        return None
    top = f.result.ranking[0]
    return {
        "label": setup_label(names, problem, f.setup),
        "value": top.estimate,
        "ci_low": top.ci_low,
        "ci_high": top.ci_high,
        "e_deaths": top.e_deaths,
        "cost": cost_line(problem, names, f.setup),
    }


CANNOT_UNDO = "An evolution cannot be undone."


class PostFightAnswer(NamedTuple):
    """One answer to the post-fight prompts scored against the next fight: ``evolve`` the carry
    slots that evolve (every other offered member cancels), the rung-``k`` chance with its
    interval and the expected deaths of the single-fight screen."""

    evolve: tuple[int, ...]
    p: float
    ci_low: float
    ci_high: float
    e_deaths: float


class PostFight(NamedTuple):
    """The evolutions a single battle offers after it (design v5.4 section 6, "Single battle,
    post-fight evolution"; ledger E3-B3-17). ``prompts`` are GT-14's and ``methods`` their method
    names, one each. ``answers`` are the scored answers best first (``prep.screen.prep_order``)
    against ``next_trainer``, empty without ``--next``; ``k`` the deaths allowed."""

    prompts: tuple[Prompt, ...]
    methods: tuple[str, ...]
    answers: tuple[PostFightAnswer, ...] = ()
    k: int = 0
    next_trainer: str | None = None


def post_fight_evolution(
    tables: Tables,
    learnsets: Learnsets,
    carry: Carry,
    levelled: Sequence[bool] | np.ndarray,
    result: int,
    map_method: int,
    time: str | None,
    settings: Settings,
    *,
    score: Callable[[Sequence[Carry]], Sequence[ScreenStats]] | None = None,
    k: int = 0,
    next_trainer: str | None = None,
) -> PostFight:
    """The evolution prompts after a single battle and, with ``score`` (``--next``), every answer
    scored.

    ``carry`` is the party as the battle left it (a single Carry) and ``levelled`` the party-0
    slot mask of the members that levelled up in it: both come from the caller, because the
    planned line's post-fight party is ``carry_out``'s (GT-4b, which waits for D3-8b) and this
    function never reads a BattleState. The prompts are :func:`post_battle_prompts` (battle
    result, level-up mask, HP not checked: ``src/battle/battle_system.c:826-850``). Evolution
    runs after the battle ends (``battle_main.c:261``, ``:274-297``), so it never changes the
    fight just planned and only the next fight sees it.

    ``score`` takes the Carries one answer each and returns their screens against the next
    trainer (see :func:`next_fight_scorer`). The answers are every subset of the living offered
    members evolving, post-evolution learn prompts all given up (the base plan; GT-15's plan
    belongs to the next fight's own search), in :func:`~nuzline.prep.screen.prep_order`; among
    tied answers fewer evolutions rank first (they cost less), so a cancel wins a tie. A fainted
    member is offered but its answer is cancel (design section 6). An answer whose screen was
    refused is left out."""
    prompts = tuple(
        post_battle_prompts(tables, learnsets, carry, levelled, result, map_method, time)
    )
    methods = tuple(learnsets.evolution_methods[p.method].replace("_", " ") for p in prompts)
    base = PostFight(prompts, methods, (), k, next_trainer)
    living = [p.slot for p in prompts if not p.dead]
    if not living or score is None:
        return base
    subsets = [c for r in range(len(living) + 1) for c in itertools.combinations(living, r)]
    carries = [
        resolve_post_battle(
            tables,
            learnsets,
            carry,
            levelled,
            result,
            map_method,
            time,
            settings,
            lambda prompt, chosen=chosen: {} if prompt.slot in chosen else None,
        )[0]
        for chosen in subsets
    ]
    stats = list(score(carries))
    if len(stats) != len(subsets):
        raise ValueError(f"score returned {len(stats)} screens for {len(subsets)} answers")
    kept = [(c, st) for c, st in zip(subsets, stats, strict=True) if not st.refused]
    rows = [scored(st, k, float(len(c)), i) for i, (c, st) in enumerate(kept)]
    answers = []
    for i in prep_order(rows):
        chosen, st = kept[i]
        low, high = st.interval(k)
        answers.append(PostFightAnswer(chosen, st.rung(k), low, high, st.e_deaths))
    return base._replace(answers=tuple(answers))


def next_fight_scorer(
    tables: Tables,
    opp: Sequence[PokemonSpec],
    trainer,
    start_condition: int,
    *,
    k: int,
    n: int,
    key,
    max_turns: int = 100,
    chatot_cry: int = 0,
) -> Callable[[Sequence[Carry]], list[ScreenStats]]:
    """The ``score`` of :func:`post_fight_evolution`: each Carry's party, in carry order, plays
    ``n`` playouts (even) of the single-fight screen against ``opp`` with ``k`` deaths allowed.
    ``tables`` are the next trainer's fight tables and ``trainer`` its data (items, terrain)."""
    from nuzline.prep.states import make_states

    def score(carries: Sequence[Carry]) -> list[ScreenStats]:
        teams = [[carry_member_spec(c, int(i)) for i in np.flatnonzero(c.present)] for c in carries]
        states = make_states(
            tables,
            teams,
            opp,
            start_condition,
            trainer.items,
            k,
            terrain=trainer.terrain,
            gravity=start_gravity(trainer),
            chatot_cry=chatot_cry,
            trick_room=start_trick_room(trainer),
        )
        return screen(tables, states, n, key, max_turns)

    return score


def _percent(p: float) -> str:
    return f"{_tenths(p)} %"


def _answer_text(post: PostFight, names: Names, answer: PostFightAnswer) -> str:
    by_slot = {p.slot: p for p in post.prompts}
    who = ", ".join(
        f"{names.species_names[by_slot[s].species]} to {names.species_names[by_slot[s].target]}"
        for s in answer.evolve
    )
    return f"evolve {who}" if who else "cancel"


def post_fight_lines(post: PostFight, names: Names) -> list[str]:
    """The post-fight evolution section as ASCII lines; empty when nothing evolves. With scored
    answers the better of evolving and cancelling is named with both numbers; without them
    cancel is recommended, and the report always says an evolution cannot be undone."""
    if not post.prompts:
        return []
    out = ["After the fight (evolution runs after the battle ends and never changes this fight):"]
    for p, method in zip(post.prompts, post.methods, strict=True):
        fainted = ", fainted: offered, the answer is cancel" if p.dead else ""
        out.append(
            f"  party slot {p.slot}: {names.species_names[p.species]} would be offered "
            f"{names.species_names[p.target]} ({method}{fainted})"
        )
    rung = f"{post.k} {'death' if post.k == 1 else 'deaths'} allowed"
    cancel = next((a for a in post.answers if not a.evolve), None)
    evolving = next((a for a in post.answers if a.evolve), None)
    if post.next_trainer is not None and cancel is not None:
        out.append(f"  Against {post.next_trainer}, the next fight ({rung}):")
        for answer in (evolving, cancel):
            if answer is not None:
                out.append(
                    f"    {_answer_text(post, names, answer)}: {_percent(answer.p)} "
                    f"({_percent(answer.ci_low)} to {_percent(answer.ci_high)}), "
                    f"{decimals(answer.e_deaths, 2, up=True)} expected deaths"
                )
        out.append(f"  Better answer: {_answer_text(post, names, post.answers[0])}.")
    elif post.next_trainer is not None:
        why = (
            "every offered Pokemon fainted"
            if all(p.dead for p in post.prompts)
            else "a playout of the next fight was refused"
        )
        out.append(f"  Against {post.next_trainer}: not scored ({why}). Recommended: cancel.")
    else:
        out.append(
            "  No --next given. Recommended answer: cancel. A cancelled evolution is offered "
            "again at that Pokemon's next level-up, but the evolved species' moves of the level "
            "it evolves at are not: its learn loop offers only the entries of exactly that level "
            "(src/evolution.c:461-480), so after a cancel a move the evolved species learns only "
            "at the level reached now is never offered. Give --next <trainer-id> to score both answers "
            "against the next fight."
        )
    out.append(f"  {CANNOT_UNDO}")
    return out


def post_fight_json(post: PostFight | None, names: Names) -> dict | None:
    """:func:`post_fight_lines` as JSON-ready data (``None`` when nothing evolves)."""
    if post is None or not post.prompts:
        return None
    return {
        "offered": [
            {
                "slot": p.slot,
                "species": names.species_names[p.species],
                "target": names.species_names[p.target],
                "method": m,
                "fainted": p.dead,
            }
            for p, m in zip(post.prompts, post.methods, strict=True)
        ],
        "next_trainer": post.next_trainer,
        "deaths_allowed": post.k,
        "answers": [
            {
                "answer": _answer_text(post, names, a),
                "evolve": list(a.evolve),
                "p": a.p,
                "ci_low": a.ci_low,
                "ci_high": a.ci_high,
                "e_deaths": a.e_deaths,
            }
            for a in post.answers
        ],
        "recommended": _answer_text(post, names, post.answers[0]) if post.answers else "cancel",
        "note": CANNOT_UNDO,
    }


def render(
    result: PrepResult,
    problem: PrepProblem,
    names: Names,
    show_depth: int = 1,
    *,
    tables: Tables,
    post_fight: PostFight | None = None,
) -> list[str]:
    """The whole answer as ASCII lines (module docstring; spec section 7). ``post_fight``: the
    evolutions the fight offers afterwards (:func:`post_fight_evolution`)."""
    from nuzline.cli import all_zero_footer

    k = problem.deaths
    trainer = problem.trainer
    who = ", ".join(_nick(problem, names, i) for i in sorted(problem.protected)) or "none"
    head = (
        f"{trainer.name or f'Trainer {trainer.id}'} (trainer {trainer.id}): level cap "
        f"{problem.cap}, {_plural(k, 'death', 'deaths')} allowed, protected: {who}"
    )
    out = [
        head,
        "Setup:",
        *(f"  {line}" for line in setup_card(problem, names, result.chosen, tables)),
        "Prep steps:",
        *(
            f"  {i}. {s}"
            for i, s in enumerate(prep_steps(problem, names, result.chosen, tables), 1)
        ),
        cost_line(problem, names, result.chosen),
        *_cheaper_lines(result, problem, names),
        *ladder_lines(result, problem, names),
    ]
    if _no_deathless(result):
        out.append("No deathless line found: every setup's 0-death chance is 0 %.")
    if result.plan is not None:
        out.append("Plan:")
        out.extend(
            plan_tree.render(
                result.plan,
                names,
                show_depth,
                root=_root(problem, result.chosen, tables),
                trainer=trainer.name or "Opponent",
            )
        )
        if all(r.estimate == 0 for r in result.plan.ranking):
            out.append(all_zero_footer(k))
    else:
        why = result.stats.stopped or "max depth 0"
        out.append(f"screen only, no plan ({why})")
    if post_fight is not None:
        out.extend(post_fight_lines(post_fight, names))
    s = result.stats
    out.append(
        f"Search: {s.setups_screened} setups screened, {s.team_rounds} team rounds, "
        f"{s.variant_rounds} variant expansions ({s.variant_halvings} halvings), "
        f"{s.playouts} playouts, {len(result.finalists)} finalists, {s.seconds:.1f} s, "
        f"stopped: {s.stopped or 'no'}"
    )
    if result.excluded:
        out.append("Left out:")
        out.extend(f"  {line}" for line in result.excluded)
    return [_ascii(line) for line in out]


def _member_json(problem: PrepProblem, names: Names, member: Member, tables: Tables) -> dict:
    spec = member_spec(problem, member, tables)
    return {
        "label": problem.candidates[member.candidate].label,
        "moves": [names.move_names[m] for m in spec.moves if m],
        "changes": [
            {
                "slot": c.slot,
                "move": names.move_names[c.move],
                "source": _source_text(names, c.source),
            }
            for c in member.moves
        ],
        "item": names.item_names[spec.item] if spec.item else None,
        "hp": member.hp,
        "status": _status_name(member.status) if member.status else None,
    }


def _source_text(names: Names, source: int) -> str:
    if source == LEVEL_UP:
        return "level-up"
    if source == RELEARNER:
        return "relearner"
    return names.item_names[source]


def _setup_json(problem: PrepProblem, names: Names, setup: Setup, tables: Tables) -> list[dict]:
    return [_member_json(problem, names, m, tables) for m in setup.members]


def record(
    result: PrepResult,
    problem: PrepProblem,
    names: Names,
    settings: PrepSettings,
    seed: int,
    table_version: str,
    reserved: Sequence[str] = (),
    game: str = "platinum_us",
    *,
    tables: Tables,
    post_fight: PostFight | None = None,
) -> dict:
    """The run as JSON-ready data (spec section 7). ``reserved``: the item names the user
    reserved (the problem keeps only what is left in the bag). ``game``: the game id, kept
    next to ``table_version`` in the record and its plan tree (plan K2b Task 18). ``pruned``
    lists every variant the prune dropped with the variant that dominated it (plan 4c ruling 8)."""
    cost = setup_cost(problem, result.chosen)
    trainer = problem.trainer
    plan = None
    if result.plan is not None:
        plan = plan_tree.to_json(
            result.plan,
            names,
            root=_root(problem, result.chosen, tables),
            trainer=trainer.name or "Opponent",
            table_version=table_version,
        )
        plan["game"] = game
    return {
        "game": game,
        "table_version": table_version,
        "trainer_id": trainer.id,
        "trainer": trainer.name,
        "cap": problem.cap,
        "deaths": problem.deaths,
        "protected": [problem.candidates[i].label for i in sorted(problem.protected)],
        "reserved": list(reserved),
        "settings": _plain(settings),
        "seed": seed,
        "fingerprint": fingerprint(problem, settings, seed, table_version, code_version(), game),
        "setup": _setup_json(problem, names, result.chosen, tables),
        "prep_steps": prep_steps(problem, names, result.chosen, tables),
        "cost": {
            "text": cost_line(problem, names, result.chosen),
            "candies": cost.candies,
            "heart_scales": cost.heart_scales,
            "tms": [names.item_names[t] for t in cost.tms],
            "hm_moves": cost.hm_moves,
            "held": [names.item_names[h] for h in cost.held],
        },
        "ladder": [
            {
                "k": r.k,
                "p": r.p,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "source": r.source,
                "setup": _setup_json(problem, names, r.setup, tables),
            }
            for r in result.ladder
        ],
        "distribution": result.distribution.as_dict(),
        "plan": plan,
        "finalists": [
            {
                "label": setup_label(names, problem, f.setup),
                "value": f.result.ranking[0].estimate if f.result.ranking else 0.0,
            }
            for f in result.finalists
        ],
        "stats": _plain(dataclasses.asdict(result.stats)),
        "excluded": list(result.excluded),
        "pruned": [
            {
                "dropped": _member_json(problem, names, dropped, tables),
                "dominated_by": _member_json(problem, names, by, tables),
            }
            for dropped, by in result.pruned
        ],
        "cheaper_alternative": _cheaper_json(result, problem, names),
        "post_fight_evolution": post_fight_json(post_fight, names),
    }
