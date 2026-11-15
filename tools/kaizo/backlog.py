"""Write the Kaizo mechanics backlog: what each split's trainers need that the engine lacks.

Usage: python tools/kaizo/backlog.py docs/kaizo/mechanics_backlog.md  (needs built Kaizo tables)

The counting helpers (``group_kind``, ``battles``, ``battle_format``, ``new_by_split``) are
pure and tested in ``tests/tools/test_backlog.py``; ``main`` loads the game and writes the file.
"""

import collections
import pathlib
import re
import sys

FORMATS = ("single", "double", "double_two_trainer", "multi")
FORMAT_NAMES = {
    "single": "Singles",
    "double": "Doubles",
    "double_two_trainer": "Two-trainer",
    "multi": "Multi",
}
KINDS = (
    "AI flags",
    "AI move effects",
    "move effects",
    "held items",
    "abilities",
    "battle conditions",
    "other",
)
_KIND_OF = {
    "move effect": "move effects",
    "level-up in battle move effect": "move effects",
    "held item": "held items",
    "held item effect": "held items",
    "ability": "abilities",
    "trainer AI": "AI move effects",
    "battle format": "battle conditions",
    "battle terrain": "battle conditions",
    "weather": "battle conditions",
    "field condition": "battle conditions",
    "start condition": "battle conditions",
}


def group_kind(k):
    """The report kind (one of ``KINDS``) of a missing group ``(kind, name)``. ``("AI", name)``
    is an AI flag when ``name`` is an ``AI_FLAG_*`` name, else an AI move-effect path (an
    untranscribed effect, an effect the policy refuses, or a move-id case of the AI's damage
    or type helpers); coverage kinds map by ``_KIND_OF``; the rest (ROM changes, forms, trainer
    items, setup refusals) is "other"."""
    kind, name = k
    if kind == "AI":
        return "AI flags" if name.startswith("AI_FLAG_") else "AI move effects"
    return _KIND_OF.get(kind, "other")


def battles(rows, by_id):
    """The battles of the split rows ``rows``, each a sorted tuple of trainer ids: a row joined
    with its ``opponent_2`` (both directions, transitively). A second opponent that is not a
    split row itself (no ``split`` in the sheet, e.g. trainers 446 and 505) still belongs to
    the battle, so its team is checked with it. ``by_id`` maps every trainer id to its row.
    A row with no ``opponent_2`` is a battle of its own, whatever its format (four Gardenia
    multi rows have none; each is counted as one battle). Raises ``ValueError`` when one battle
    would span two splits."""
    parent = {}

    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for t in rows:
        find(t.id)
        if t.opponent_2 is not None:
            parent[find(t.id)] = find(t.opponent_2)
    comps = collections.defaultdict(list)
    for i in parent:
        comps[find(i)].append(i)
    out = []
    for ids in comps.values():
        splits = {by_id[i].split for i in ids if by_id[i].split}
        if len(splits) != 1:
            raise ValueError(f"battle {sorted(ids)} spans splits {sorted(splits)}")
        out.append(tuple(sorted(ids)))
    return sorted(out)


def ally_ids(trainers):
    """The trainer ids that fight beside the player: every partner variant of a multi row.
    A split row among them is the player's tag partner (the sheet lists it as a header), not
    an opponent, so it is no battle of its own."""
    return {p.trainer for t in trainers for p in (t.partner or ())}


def sheet_cites(t):
    """The split-sheet cites of trainer ``t`` (its ``sources["split"]``, ``sheet <cite>, ...``)."""
    src = t.sources.get("split", "")
    return (
        [c for c in src.removeprefix("sheet ").split(", ") if c] if src.startswith("sheet ") else []
    )


def battle_split(ids, by_id):
    """The split of the battle ``ids`` (the split of its split rows)."""
    return next(by_id[i].split for i in ids if by_id[i].split)


def battle_format(ids, by_id):
    """The format (one of ``FORMATS``) shared by every row of the battle ``ids``; a battle
    whose rows disagree is a data error (``ValueError``)."""
    fmts = {by_id[i].format for i in ids}
    if len(fmts) != 1 or not fmts <= set(FORMATS):
        raise ValueError(f"battle {list(ids)} has formats {sorted(map(str, fmts))}")
    return fmts.pop()


def partner_unknown(ids, by_id):
    """A multi battle whose rows record no partner team (``partner`` is None): the battle is
    still counted, as multi, and its opponents are checked; only the ally's team is unknown."""
    return battle_format(ids, by_id) == "multi" and all(by_id[i].partner is None for i in ids)


def new_by_split(order, needs):
    """For each split in ``order``, the groups of ``needs[split]`` that no earlier split
    needs. Their union over ``order`` is the whole-game union."""
    seen, out = set(), {}
    for s in order:
        out[s] = needs[s] - seen
        seen |= needs[s]
    return out


def kind_counts(groups):
    """``{kind: number of groups}`` over ``KINDS``."""
    c = collections.Counter(group_kind(k) for k in groups)
    return {k: c[k] for k in KINDS}


# Evolution methods (evolution_methods.json names) whose parameter is the level the evolution
# needs. Every other method (friendship, trade, items, known move, location) is taken as
# reachable at any level: the player-side bound assumes it.
LEVEL_METHODS = frozenset(
    {
        "level",
        "level_atk_gt_def",
        "level_atk_eq_def",
        "level_atk_lt_def",
        "level_pid_low",
        "level_pid_high",
        "level_ninjask",
        "level_shedinja",
        "level_male",
        "level_female",
    }
)


def evolution_parents(evolutions, methods):
    """``{child species: [(parent species, method name, param), ...]}`` from
    ``evolutions.json`` (species -> ``[[method, param, target], ...]``) and the method names."""
    out = collections.defaultdict(list)
    for parent, evos in evolutions.items():
        for method, param, target in evos:
            if target:
                out[target].append((int(parent), methods[method], param))
    return dict(out)


def reachable_species(species, parents, cap):
    """``{species: assumed}`` for each of ``species`` the player can have at level ``cap``,
    upper bound: a species with no pre-evolution is reachable; an evolved one when a parent
    is and the step's level (a ``LEVEL_METHODS`` step) is at most ``cap``. ``assumed`` is True
    when every path needs a non-level step (friendship, trade, item, ...), which the bound
    takes as possible at any level."""
    memo = {}

    def reach(s, stack=()):
        if s in memo:
            return memo[s]
        if s in stack:  # no evolution cycles in the data; refuse rather than loop
            raise ValueError(f"evolution cycle through species {s}")
        best = None
        for parent, method, param in parents.get(s, ()):
            r = reach(parent, (*stack, s))
            if r is None:
                continue
            if method in LEVEL_METHODS:
                if param > cap:
                    continue
                step = r
            else:
                step = True
            best = step if best is None else best and step
        memo[s] = False if s not in parents else best
        return memo[s]

    return {s: r for s in species if (r := reach(s)) is not None}


def ancestors(s, parents):
    """``s`` and every species it evolves from."""
    out, todo = [], [s]
    while todo:
        x = todo.pop()
        if x not in out:
            out.append(x)
            todo.extend(p for p, _, _ in parents.get(x, ()))
    return out


def levelup_moves(learnset, cap):
    """The moves of a ``[[level, move], ...]`` learnset learned at level ``cap`` or lower."""
    return {move for level, move in learnset if level <= cap}


# Hold effects with no effect on the fight (generated/item_hold_effects.txt names): why, and
# the decomp line that uses the effect (file, line, a token that line or the next two hold).
# ``player_section`` re-checks every citation and that no battle code (src/battle,
# include/battle, res/battle) names the effect except at the cited line.
_TRADE_EVO = ("held for a trade evolution, outside the battle", "src/pokemon.c", 3735, "EVO_TRADE")
NO_BATTLE_EFFECT = {
    "HOLD_EFFECT_MONEY_UP": (
        "doubles the prize money (prizeMoneyMul) and nothing else",
        "src/battle/battle_lib.c",
        4127,
        "prizeMoneyMul",
    ),
    "HOLD_EFFECT_ENCOUNTERS_DOWN": (
        "read by item id in the field's wild-encounter code only",
        "src/overlay006/wild_encounters.c",
        925,
        "ITEM_CLEANSE_TAG",
    ),
    "HOLD_EFFECT_NO_EVOLVE": (
        "read by the evolution check only",
        "src/pokemon.c",
        3544,
        "HOLD_EFFECT_NO_EVOLVE",
    ),
    "HOLD_EFFECT_EVOLVE_SEADRA": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_PORYGON": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_RHYDON": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_ELECTABUZZ": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_MAGMAR": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_PORYGON2": _TRADE_EVO,
    "HOLD_EFFECT_EVOLVE_DUSCLOPS": _TRADE_EVO,
}


def _check_no_battle_effect(decomp):
    """Verify ``NO_BATTLE_EFFECT`` against the decomp; ``ValueError`` on a stale citation."""
    battle = [
        p
        for sub in ("src/battle", "include/battle", "res/battle")
        for p in (decomp / sub).rglob("*")
        if p.suffix in (".c", ".h", ".s")
    ]
    text = {p: p.read_text(encoding="utf-8", errors="replace").splitlines() for p in battle}
    for name, (_, path, line, token) in NO_BATTLE_EFFECT.items():
        lines = (decomp / path).read_text(encoding="utf-8", errors="replace").splitlines()
        if not any(token in x for x in lines[line - 1 : line + 2]):
            raise ValueError(f"{name}: {path}:{line} no longer holds {token!r}")
        for p, ls in text.items():
            for i, x in enumerate(ls, 1):
                cited = p == decomp / path and i == line
                if re.search(rf"\b{name}\b", x) and not cited:
                    raise ValueError(f"{name} is read by battle code at {p}:{i}")


def player_section(
    L, order, whole, by_id, split_rows, extra, check, key, tables, names, d, items, decomp
):
    """Append the player-side upper bound and the AI partners' line to ``L`` (see the
    section's own text for the assumptions)."""
    import json
    import tomllib

    import numpy as np

    from nuzline.ai.policy import player_ability_refusal, player_move_refusal
    from nuzline.engine.coverage import (
        IMPLEMENTED_ABILITIES,
        IMPLEMENTED_ITEMS,
        Support,
        move_effect_line,
    )
    from nuzline.engine.ids import AbilityId, ItemId

    def load(name):
        return json.loads((d / name).read_text(encoding="utf-8"))

    learn, tm_learn = load("learnsets.json"), load("tm_learnsets.json")
    tutor_learn, egg = load("tutor_learnsets.json"), load("egg_moves.json")
    tutor_moves, form_rows = load("tutor_moves.json"), load("form_rows.json")
    tm_moves = [i["teaches_move"] for i in items if i.get("teaches_move")]
    parents = evolution_parents(load("evolutions.json"), load("evolution_methods.json"))
    root = pathlib.Path(__file__).resolve().parents[2]
    caps = tomllib.loads((root / "data/kaizo/level_caps.toml").read_text(encoding="utf-8"))
    cap_of = {s: caps["splits"][s.removesuffix(" Split")] for s in order}
    # Species ids 1..n-1, n the Egg's id; the ids from the Egg on are not Pokemon.
    species = list(range(1, names.species_names.index("Egg")))
    rows_of = {s: [r for r in form_rows[s] if r >= 0] for s in species}

    pmembers = collections.defaultdict(set)
    move_cache = {}

    def move_groups(move, row):
        if (move, row) not in move_cache:
            out = set()
            support, name = move_effect_line(tables, names, move, row)
            if support == Support.NONE:
                k, member = key(f"you: {name} (move effect UNSUPPORTED)")
                out.add(k)
                pmembers[k].add(member or name)
            if (reason := player_move_refusal(move)) is not None:
                k = ("AI", f"reads your move: {reason}")
                out.add(k)
                pmembers[k].add(names.move_names[move])
            move_cache[move, row] = out
        return move_cache[move, row]

    sa = np.asarray(tables.species_abilities)
    sad = np.asarray(tables.species_abilities_decomp)

    def row_groups(row, s):
        """Groups of personal ``row`` (species ``s``) at any level: its abilities, TM/HM and
        tutor moves, and the egg moves of its family's root."""
        out = set()
        for slot in (0, 1):
            dec = int(sad[row, slot])
            aname = names.ability_names[dec]
            if IMPLEMENTED_ABILITIES.get(AbilityId(int(sa[row, slot])), Support.NONE) == (
                Support.NONE
            ):
                out.add(("ability", aname))
            if (reason := player_ability_refusal(dec)) is not None:
                k = ("AI", f"reads your ability: {reason}")
                out.add(k)
                pmembers[k].add(aname)
        moves = {tm_moves[i] for i in tm_learn.get(str(row), ())}
        moves |= {tutor_moves[i]["move"] for i in tutor_learn.get(str(row), ())}
        for a in ancestors(s, parents):
            moves |= set(egg.get(str(a), ()))
        for m in moves:
            out |= move_groups(m, row)
        return out

    fixed = {(s, r): row_groups(r, s) for s in species for r in rows_of[s]}

    _check_no_battle_effect(decomp)
    hold_names = (decomp / "generated/item_hold_effects.txt").read_text().split()
    ie, hold = np.asarray(tables.item_effect), np.asarray(tables.item_hold_effect)
    held, inert = set(), collections.defaultdict(list)
    for i in range(1, len(hold)):
        h = int(hold[i])
        if not h or IMPLEMENTED_ITEMS.get(ItemId(int(ie[i])), Support.NONE) != Support.NONE:
            continue
        if hold_names[h] in NO_BATTLE_EFFECT:
            inert[h].append(names.item_names[i])
            continue
        k = ("held item effect", f"#{h}")
        held.add(k)
        pmembers[k].add(names.item_names[i])

    needs, reach_n, assumed_n = {}, {}, {}
    for s_name in order:
        cap = cap_of[s_name]
        reach = reachable_species(species, parents, cap)
        reach_n[s_name], assumed_n[s_name] = len(reach), sum(reach.values())
        groups = set(held)
        for s in reach:
            lv = set()
            for a in ancestors(s, parents):
                for r in rows_of.get(a, ()):
                    lv |= levelup_moves(learn.get(str(r), ()), cap)
            for r in rows_of[s]:
                groups |= fixed[s, r]
                for m in lv:
                    groups |= move_groups(m, r)
        needs[s_name] = groups
    pwhole = set().union(*needs.values())
    only = {s: needs[s] - whole for s in order}
    only_new = new_by_split(order, only)
    pw_only = pwhole - whole

    def plabel(k):
        kind, name = k
        m = sorted(pmembers.get(k, ()))
        if m:
            shown = ", ".join(m[:6]) + (f", +{len(m) - 6} more" if len(m) > 6 else "")
            return f"{kind} {name}: {shown}"
        return f"{kind}: {name}"

    L.extend(
        [
            "",
            "## Your side (upper bound)",
            "",
            "What your own Pokemon could bring that the engine does not model, counted apart",
            "from the opponent side. The headline is **player-only**: groups no opposing",
            "trainer in any split needs (the opponent whole-game union above). It is an upper",
            "bound under these assumptions:",
            "",
            "- Every species without a pre-evolution can be obtained in every split. An evolved",
            "  species counts from the split whose cap reaches its evolution level; evolutions",
            "  by friendship, trade, item, known move or place are taken as reachable at any",
            "  level (the bracketed count). Evolved species met in the wild or as gifts before",
            "  their evolution level are not seen, nor any encounter limit or Nuzlocke rule.",
            "- Moves: level-up moves at or below the split's cap (data/kaizo/level_caps.toml)",
            "  of the species and its pre-evolutions, plus every TM/HM and tutor move it is",
            "  compatible with and its family's egg moves, in every split.",
            "- Abilities: both of the species' abilities. Held items: every item with a hold",
            "  effect, in every split (so held items all count from the first split).",
            "- Also counted: the AI's refusals when your Pokemon knows a move or has an ability",
            "  its scripts read but the engine does not carry (`player_move_refusal`,",
            "  `player_ability_refusal`).",
            "- Not counted: hold effects with no battle effect (listed below, from the decomp).",
            "",
            (
                "| Split | Cap | Species reachable (via a non-level step) | Your groups "
                "| Player-only | New player-only |"
            ),
            "|---|---|---|---|---|---|",
        ]
    )
    for s in order:
        L.append(
            f"| {s} | {cap_of[s]} | {reach_n[s]} ({assumed_n[s]}) | {len(needs[s])} | "
            f"{len(only[s])} | {len(only_new[s])} |"
        )
    L.append(f"| **Whole game** | | | {len(pwhole)} | {len(pw_only)} | {len(pw_only)} |")
    L.extend(
        [
            "",
            (
                f"Of your {len(pwhole)} groups, {len(pwhole & whole)} are also needed by opposing"
                f" trainers and {len(pw_only)} are player-only."
            ),
            "",
            "### New player-only groups per split, by kind",
            "",
            "| Split | " + " | ".join(KINDS) + " | All |",
            "|---|" + "---|" * (len(KINDS) + 1),
        ]
    )
    for s in order:
        c = kind_counts(only_new[s])
        L.append(f"| {s} | " + " | ".join(str(c[k]) for k in KINDS) + f" | {len(only_new[s])} |")
    c = kind_counts(pw_only)
    L.append(
        "| **Whole game** | "
        + " | ".join(f"**{c[k]}**" for k in KINDS)
        + f" | **{len(pw_only)}** |"
    )
    first = {k: s for s in order for k in only_new[s]}
    L.extend(
        [
            "",
            "### Player-only groups",
            "",
            "| Kind | Group | First split |",
            "|---|---|---|",
        ]
        + [
            f"| {group_kind(k)} | {plabel(k)} | {first[k]} |"
            for k in sorted(
                pw_only, key=lambda k: (KINDS.index(group_kind(k)), order.index(first[k]), k)
            )
        ]
    )
    L.extend(
        [
            "",
            "### Hold effects with no battle effect (not counted)",
            "",
            "Each is cited where the decomp reads it, and no battle code (src/battle,",
            "include/battle, res/battle) names it elsewhere; the tool re-checks both on every run.",
            "",
            "| Hold effect | Items | Why | Decomp |",
            "|---|---|---|---|",
        ]
        + [
            f"| #{h} {hold_names[h]} | {', '.join(sorted(inert[h]))} | "
            f"{NO_BATTLE_EFFECT[hold_names[h]][0]} | "
            f"{NO_BATTLE_EFFECT[hold_names[h]][1]}:{NO_BATTLE_EFFECT[hold_names[h]][2]} |"
            for h in sorted(inert)
        ]
    )

    # AI partners beside the player in multi battles: their own line, not in either count.
    pids = sorted(
        {
            p.trainer
            for t in [*split_rows, *(by_id[i] for i in extra)]
            if t.format == "multi" and t.partner
            for p in t.partner
        }
    )
    pmem, perrs = collections.defaultdict(set), []
    partner = set().union(*(check(by_id[i], pmem, perrs) for i in pids)) if pids else set()
    c = kind_counts(partner - whole)
    L.extend(
        [
            "",
            "## AI partners in multi battles",
            "",
            f"{len(pids)} partner trainer rows ({', '.join(str(i) for i in pids)}), checked like"
            f" opponents: {len(partner)} missing groups, {len(partner - whole)} not needed by"
            f" any opposing trainer, {len(partner - whole - pwhole)} also not on your side."
            + (
                " Not needed by opponents, by kind: "
                + ", ".join(f"{k} {c[k]}" for k in KINDS if c[k])
                + "."
                if partner - whole
                else ""
            )
            + (f" Refused before the check: {len(perrs)}." if perrs else ""),
        ]
    )


def metronome_unmodelled(tables, names, row):
    """``(n, m)`` for Metronome (ruling B5-2): of the ``m`` moves ``BtlCmd_Metronome`` can call
    for a Pokemon at personal ``row`` that knows only Metronome (``coverage.metronome_calls``),
    the ``n`` that ``coverage.move_effect_line`` refuses, each of which refuses Metronome
    (coverage.py ``move_effect_line``)."""
    from types import SimpleNamespace

    from nuzline.engine.coverage import Support, metronome_calls, move_effect_line

    metronome = names.move_names.index("Metronome")
    calls = metronome_calls(tables, SimpleNamespace(moves=(metronome, 0, 0, 0)))
    calls = [m for m in calls if m != metronome]
    n = sum(move_effect_line(tables, names, m, row)[0] == Support.NONE for m in calls)
    return n, len(calls)


def main(out_path):
    import argparse
    import json

    from nuzline import cli
    from nuzline.ai.layout import KAIZO, kaizo_layout
    from nuzline.ai.policy import opponent_move_refusals
    from nuzline.data.games import load_game
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import check_opponent, start_weather

    game = cli._game(argparse.Namespace(game="kaizo"))
    tables, names = load_game(game)
    trainers = cli._load_trainers(game)
    inter = IntermediateData(game.intermediate_dir)
    d = game.intermediate_dir
    moves = json.loads((d / "moves.json").read_text(encoding="utf-8"))
    items = json.loads((d / "items.json").read_text(encoding="utf-8"))
    mv = {m["name"].lower(): m["ai_effect"] for m in moves if m.get("name")}
    it = {i["name"].lower(): i["hold_effect"] for i in items if i.get("name")}

    # The AI script labels of each battle effect: Basic_ScoreMoveEffect's and Expert_Main's jump
    # tables (script.s:134-285, :1628-1808), by effect name (generated/move_battle_effects.txt,
    # line n + 1 is effect n).
    decomp = pathlib.Path(__file__).resolve().parents[2] / "external/pokeplatinum"
    effect_names = [
        e.removeprefix("BATTLE_EFFECT_")
        for e in (decomp / "generated/move_battle_effects.txt").read_text().split()
    ]
    script = (decomp / "src/battle/trainer_ai/script.s").read_text(encoding="utf-8").splitlines()
    ai_labels = collections.defaultdict(list)
    for lo, hi in ((134, 285), (1628, 1808)):
        for line in script[lo - 1 : hi]:
            hit = re.match(r"\s*IfCurrentMoveEffectEqualTo BATTLE_EFFECT_(\w+), (\w+)", line)
            if hit and hit.group(2) not in ai_labels[hit.group(1)]:
                ai_labels[hit.group(1)].append(hit.group(2))

    members = collections.defaultdict(set)

    def ai_move_groups(trainer, party, mem):
        """The AI's per-move refusals of the trainer's moves (``policy.opponent_move_refusals``
        under the trainer's flag set): ``check_opponent`` runs no ``ai_refusals`` (they need
        the player's party), so the opponent's untranscribed AI move paths are listed here,
        keyed by battle effect. Runs for every checked trainer row, in every split."""
        try:
            # the per-move refusals read no read bound (E1 Ruling R6): any bound serves
            layout = kaizo_layout(trainer.ai_flags, KAIZO.slot_reads)
        except ValueError:  # a flag the policy lacks: its own flag line; moves under Roark's set
            layout = KAIZO
        effect = tables.move_ai_effect
        out = set()
        for mon in party:
            ghost = 7 in [int(x) for x in tables.species_types[mon.species]]  # TYPE_GHOST
            for move in mon.moves:
                e = int(effect[move]) if move else 0
                name = effect_names[e] if move else ""
                for kind in opponent_move_refusals(tables, move, layout, ghost):
                    if kind == "not transcribed":
                        labels = " / ".join(ai_labels.get(name, ())) or "no Basic or Expert label"
                        k = ("AI", f"move effect #{e} {name} not transcribed ({labels})")
                    elif kind in ("calc damage", "move type"):
                        helper = "CalcDamage" if kind == "calc damage" else "MoveType"
                        k = ("AI", f"move-id case of TrainerAI_{helper}")
                    else:
                        k = ("AI", f"move effect #{e} {name}: {kind}")
                    out.add(k)
                    mem[k].add(names.move_names[move])
        return out

    def key(s):
        body, kind = s.rsplit(" (", 1)
        kind = kind.removesuffix(" UNSUPPORTED)")
        name = body.split(": ", 1)[1] if ": " in body else body
        if kind == "move effect" and name.lower() in mv:
            return ("move effect", f"#{mv[name.lower()]}"), name
        if kind == "held item" and name.lower() in it:
            return ("held item effect", f"#{it[name.lower()]}"), name
        if kind == "AI flag":
            return ("AI", name), None
        return (kind, name), None

    errors = []

    def check(t, mem=members, errs=errors):
        """The missing groups of trainer row ``t`` (its team, bag, AI and battle conditions)."""
        try:
            cli._check_buildable(tables, names, t, inter)
            party = build_trainer_party(t, inter)
            r = check_opponent(tables, names, t, party, start_weather(t))
            ks = ai_move_groups(t, party, mem)
            for m in r.missing:
                k, member = key(str(m))
                ks.add(k)
                if member:
                    mem[k].add(member)
        except Exception as e:  # noqa: BLE001  refused before the check
            ks = {("setup", str(e).split(":")[-1].strip()[:60])}
            errs.append((t.id, t.label, str(e)[:120]))
        return ks

    by_id = {t.id: t for t in trainers}
    # Tag partners are allies, not battles: each must be a sheet header tagged as one.
    docs = json.loads((d / "docs/docs.json").read_text(encoding="utf-8"))
    tags = {
        h["cite"]: h["tags"]
        for sv in docs["splits"].values()
        for sec in sv["sections"]
        for h in sec["trainers"]
    }
    allies = ally_ids(trainers)
    for t in trainers:
        if t.id in allies and t.split:
            untagged = [c for c in sheet_cites(t) if not tags.get(c, {}).get("tag_partner")]
            if untagged or not sheet_cites(t):
                raise ValueError(f"partner {t.id} {t.label}: sheet header not a tag partner")
    split_rows = [t for t in trainers if t.split and t.split_order and t.id not in allies]
    splits = {}
    for t in split_rows:
        splits.setdefault(t.split, t.split_order[0])
    order = sorted(splits, key=splits.get)
    fights = {t.id: (t, check(t)) for t in split_rows}
    bs = battles(split_rows, by_id)
    extra = sorted(i for b in bs for i in b if i not in fights)  # second opponents, no split
    row_groups = {i: fights[i][1] if i in fights else check(by_id[i]) for b in bs for i in b}
    battle_groups = {b: set().union(*(row_groups[i] for i in b)) for b in bs}
    split_battles = {s: [b for b in bs if battle_split(b, by_id) == s] for s in order}
    needs = {s: set().union(*(battle_groups[b] for b in split_battles[s])) for s in order}
    new = new_by_split(order, needs)
    whole = set().union(*needs.values())

    def label(k):
        kind, name = k
        m = sorted(members.get(k, ()))
        if m:
            shown = ", ".join(m[:6]) + (f", +{len(m) - 6} more" if len(m) > 6 else "")
            if k == ("move effect", f"#{mv['metronome']}"):  # Bulbasaur's row: not Ghost
                shown += "; {} of {} callable moves unmodelled".format(
                    *metronome_unmodelled(tables, names, 1)
                )
            return f"{kind} {name}: {shown}"
        return f"{kind}: {name}"

    L = [
        "# Kaizo mechanics backlog",
        "",
        "Generated by `tools/kaizo/backlog.py` from the opponent-side coverage check",
        "(`check_opponent`), plus the AI's per-move refusals of the trainers' moves",
        "(`policy.opponent_move_refusals`: AI move effects whose Basic/Expert script paths are",
        "not transcribed, keyed by battle effect), which `check_opponent` skips. Both run on every",
        "opposing trainer row of every split.",
        "It counts only what the trainers' teams, bags, AI and battle conditions need.",
        "Your own Pokemon's moves, abilities and items are not counted, nor are the teams of",
        "AI partners beside you in multi battles.",
        "Moves are grouped by the game's move-effect number and held items by hold-effect number,",
        "since one implementation covers the whole group.",
        "",
        "## By split",
        "",
        "A row is one opposing trainer (tag partners beside you are not counted); a battle is a",
        "row joined with its second opponent (`opponent_2`),",
        "so a two-trainer or multi battle against two trainers is one battle. A second opponent",
        "with no split of its own is checked with its battle:",
        ", ".join(f"{i} {by_id[i].label}" for i in extra) or "none",
        "(not in the Rows column). A row with no `opponent_2` is one battle. Multi battles whose",
        "rows record no partner team are counted as multi (the column in brackets); their",
        "opponents are checked as usual. Runnable: every row of the battle has no missing group.",
        "New groups: needed in this split and in no earlier one.",
        "",
        (
            "| Split | Rows | Battles | Singles | Doubles | Two-trainer | Multi (no known partner) "
            "| Runnable battles | Distinct missing groups | New groups |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot = collections.Counter()
    for s in order:
        sb = split_battles[s]
        fmt = collections.Counter(battle_format(b, by_id) for b in sb)
        unk = sum(partner_unknown(b, by_id) for b in sb)
        rows = sum(1 for t in split_rows if t.split == s)
        ok = sum(1 for b in sb if not battle_groups[b])
        n = dict(rows=rows, battles=len(sb), ok=ok, unk=unk, **{f: fmt[f] for f in FORMATS})
        tot.update(n)
        L.append(
            f"| {s} | {rows} | {len(sb)} | {fmt['single']} | {fmt['double']} | "
            f"{fmt['double_two_trainer']} | {fmt['multi']} ({unk}) | {ok} | {len(needs[s])} | "
            f"{len(new[s])} |"
        )
    L.append(
        f"| **Whole game** | {tot['rows']} | {tot['battles']} | {tot['single']} | "
        f"{tot['double']} | {tot['double_two_trainer']} | {tot['multi']} ({tot['unk']}) | "
        f"{tot['ok']} | {len(whole)} | {len(whole)} |"
    )

    def kind_table(title, intro, per_split, last):
        L.extend(["", f"### {title}", "", intro, ""])
        L.append("| Split | " + " | ".join(KINDS) + " | All |")
        L.append("|---|" + "---|" * (len(KINDS) + 1))
        for s in order:
            c = kind_counts(per_split[s])
            L.append(
                f"| {s} | " + " | ".join(str(c[k]) for k in KINDS) + f" | {len(per_split[s])} |"
            )
        c = kind_counts(last)
        L.append(
            "| **Whole game** | "
            + " | ".join(f"**{c[k]}**" for k in KINDS)
            + f" | **{len(last)}** |"
        )

    L.extend(
        [
            "",
            "## Whole game",
            "",
            f"{len(whole)} distinct missing groups across all ten splits ({tot['battles']} battles,",
            f"{tot['rows'] + len(extra)} opposing trainer rows). Kinds: AI flags are flag lines the",
            "policy lacks; AI move effects are the AI's per-move script paths (untranscribed",
            "effects, refused effects, move-id cases of its damage and type helpers); battle",
            "conditions are battle formats (including each unrecorded gauntlet position), terrain,",
            "weather, field and start conditions; other is ROM changes, forms, trainer items and",
            "setup refusals.",
        ]
    )
    kind_table(
        "New groups per split, by kind",
        "Groups first needed in that split (needed by no earlier split). The whole-game row is"
        " the union.",
        new,
        whole,
    )
    kind_table(
        "Groups needed in each split, by kind",
        "Every group the split's battles need, whether or not an earlier split needs it too.",
        needs,
        whole,
    )
    first = {k: s for s in order for k in new[s]}
    odd = sorted(
        (k for k in whole if group_kind(k) in ("battle conditions", "other")),
        key=lambda k: (group_kind(k), order.index(first[k]), k),
    )
    L.extend(
        [
            "",
            "### Battle conditions and other groups",
            "",
            "| Kind | Group | First split |",
            "|---|---|---|",
        ]
        + [f"| {group_kind(k)} | {label(k)} | {first[k]} |" for k in odd]
    )
    player_section(
        L, order, whole, by_id, split_rows, extra, check, key, tables, names, d, items, decomp
    )

    def table(title, sel, top=None):
        fs = [f for f in fights.values() if f[0].split in sel]
        cnt = collections.Counter(k for _, ks in fs for k in ks)
        L.extend(
            [
                "",
                f"## {title}",
                "",
                f"{len(fs)} fights (trainer rows). A group's count is the number of rows it blocks.",
                "",
                "| Fights blocked | Group |",
                "|---|---|",
            ]
        )
        for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:top]:
            L.append(f"| {v} | {label(k)} |")
        kinds = collections.Counter()
        for k in cnt:
            kinds[k[0]] += 1
        L.extend(["", "Groups by kind: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common())])
        return cnt

    table("Roark split", order[:1])
    table("Roark and Gardenia splits (up to the second gym)", order[:2])
    table("All ten splits (top 60)", order, 60)

    # greedy unlock order for the first two splits
    fs = {t.id: set(ks) for t, ks in fights.values() if t.split in order[:2]}
    done, steps = set(), []
    while any(ks - done for ks in fs.values()):
        cnt = collections.Counter()
        for ks in fs.values():
            rest = ks - done
            if rest:
                for k in rest:
                    cnt[k] += 1.0 / len(rest)
        k = max(cnt, key=lambda x: (cnt[x], x))
        done.add(k)
        steps.append((k, sum(1 for ks in fs.values() if ks <= done)))
    L.extend(
        [
            "",
            "## Build order for the first two splits",
            "",
            "Greedy: each step is the group that brings the most fights closest to runnable.",
            "",
            "| Step | Group | Fights runnable after |",
            "|---|---|---|",
        ]
    )
    for i, (k, n) in enumerate(steps, 1):
        L.append(f"| {i} | {label(k)} | {n} of {len(fs)} |")
    if errors:
        L.extend(
            ["", "## Fights refused before the check", ""]
            + [f"- {i} {lb}: {e}" for i, lb, e in errors[:40]]
        )
    pathlib.Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L[:30]))
    print(len(steps), "steps;", len(errors), "errors")


if __name__ == "__main__":
    main(sys.argv[1])
