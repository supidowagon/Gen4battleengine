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

    def ai_move_groups(trainer, party):
        """The AI's per-move refusals of the trainer's moves (``policy.opponent_move_refusals``
        under the trainer's flag set): ``check_opponent`` runs no ``ai_refusals`` (they need
        the player's party), so the opponent's untranscribed AI move paths are listed here,
        keyed by battle effect. Runs for every checked trainer row, in every split."""
        try:
            layout = kaizo_layout(trainer.ai_flags)
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
                    members[k].add(names.move_names[move])
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

    def check(t):
        """The missing groups of trainer row ``t`` (its team, bag, AI and battle conditions)."""
        try:
            cli._check_buildable(tables, names, t, inter)
            party = build_trainer_party(t, inter)
            r = check_opponent(tables, names, t, party, start_weather(t))
            ks = ai_move_groups(t, party)
            for m in r.missing:
                k, member = key(str(m))
                ks.add(k)
                if member:
                    members[k].add(member)
        except Exception as e:  # noqa: BLE001  refused before the check
            ks = {("setup", str(e).split(":")[-1].strip()[:60])}
            errors.append((t.id, t.label, str(e)[:120]))
        return ks

    by_id = {t.id: t for t in trainers}
    split_rows = [t for t in trainers if t.split and t.split_order]
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
        "A row is one trainer; a battle is a row joined with its second opponent (`opponent_2`),",
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
