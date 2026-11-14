"""Write the Kaizo mechanics backlog: what each split's trainers need that the engine lacks.

Usage: python tools/kaizo/backlog.py docs/kaizo/mechanics_backlog.md  (needs built Kaizo tables)
"""

import argparse
import collections
import json
import pathlib
import re
import sys

from nuzline import cli
from nuzline.data.games import load_game
from nuzline.data.trainer import IntermediateData, build_trainer_party
from nuzline.engine.coverage import check_opponent, start_weather

out_path = sys.argv[1]
game = cli._game(argparse.Namespace(game="kaizo"))
tables, names = load_game(game)
trainers = cli._load_trainers(game)
inter = IntermediateData(game.intermediate_dir)
d = game.intermediate_dir
moves = json.loads((d / "moves.json").read_text(encoding="utf-8"))
items = json.loads((d / "items.json").read_text(encoding="utf-8"))
mv = {m["name"].lower(): m["ai_effect"] for m in moves if m.get("name")}
it = {i["name"].lower(): i["hold_effect"] for i in items if i.get("name")}


def key(s):
    body, kind = s.rsplit(" (", 1)
    kind = kind.removesuffix(" UNSUPPORTED)")
    name = body.split(": ", 1)[1] if ": " in body else body
    if kind == "move effect" and name.lower() in mv:
        return ("move effect", f"#{mv[name.lower()]}"), name
    if kind == "held item" and name.lower() in it:
        return ("held item effect", f"#{it[name.lower()]}"), name
    if kind == "AI flag set":
        return ("AI", "flag set " + re.search(r"0x[0-9a-fA-F]+", name).group(0)), None
    if kind == "AI flag":
        return ("AI", name), None
    return (kind, name), None


splits, fights = {}, {}
members = collections.defaultdict(set)
errors = []
for t in trainers:
    if not t.split or not t.split_order:
        continue
    splits.setdefault(t.split, t.split_order[0])
    try:
        cli._check_buildable(tables, names, t, inter)
        r = check_opponent(tables, names, t, build_trainer_party(t, inter), start_weather(t))
        ks = set()
        for m in r.missing:
            k, member = key(str(m))
            ks.add(k)
            if member:
                members[k].add(member)
    except Exception as e:  # noqa: BLE001  refused before the check
        ks = {("setup", str(e).split(":")[-1].strip()[:60])}
        errors.append((t.id, t.label, str(e)[:120]))
    fights[t.id] = (t, ks)

order = sorted(splits, key=splits.get)


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
    "Generated from the opponent-side coverage check (`check_opponent`) on main after the K3",
    "merge (exp and level-ups). It counts only what the trainers' teams, bags, AI and battle conditions need.",
    "Your own Pokemon's moves, abilities and items are not counted.",
    "Moves are grouped by the game's move-effect number and held items by hold-effect number,",
    "since one implementation covers the whole group.",
    "",
    "## By split",
    "",
    "| Split | Fights | Runnable now | Singles | Distinct missing groups |",
    "|---|---|---|---|---|",
]
for s in order:
    fs = [f for f in fights.values() if f[0].split == s]
    ok = sum(1 for t, ks in fs if not ks)
    single = sum(1 for t, ks in fs if t.format == "single")
    groups = set().union(*[ks for _, ks in fs]) if fs else set()
    L.append(f"| {s} | {len(fs)} | {ok} | {single} | {len(groups)} |")


def table(title, sel, top=None):
    fs = [f for f in fights.values() if f[0].split in sel]
    cnt = collections.Counter(k for _, ks in fs for k in ks)
    L.extend(
        [
            "",
            f"## {title}",
            "",
            f"{len(fs)} fights. A group's count is the number of fights it blocks.",
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
        + [f"- {i} {l}: {e}" for i, l, e in errors[:40]]
    )
pathlib.Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:30]))
print(len(steps), "steps;", len(errors), "errors")
