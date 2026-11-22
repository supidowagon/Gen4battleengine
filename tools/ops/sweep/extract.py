"""Cut the Hypothesis falsifying draws out of the sweep logs into replay input folders.

Plain Python (no nuzline / jax import). Usage:
    python extract.py <log dir> <out dir>
Every log named <node>_<seed>.log in <log dir> that holds a "Failing test case" gives
<out dir>/<node>_<seed>/ (a second falsifying example in the same log gets the suffix _b, _c, ...):
  node.txt       test_mask_sound | test_mask_sound_with_a_trainer_item
  state.txt      Draw 1 (BattleState repr; for the trainer-item node the state BEFORE the item)
  test_mask_sound:                    acts.txt (Draw 2), outcome.txt (Draw 3), seed.txt (Draw 4)
  ..._with_a_trainer_item:            item.txt (Draw 2, the bag item name), choice.txt (Draw 3,
                                      the player action drawn from the legal list), outcome.txt
                                      (Draw 4), seed.txt (Draw 5)
  reproduce.txt  the @reproduce_failure line, when present
"""

import ast
import re
import sys
from pathlib import Path

LAYOUT = {
    "test_mask_sound": ("state", "acts", "outcome", "seed"),
    "test_mask_sound_with_a_trainer_item": ("state", "item", "choice", "outcome", "seed"),
}
DRAW = re.compile(r"^E?\s*Draw (\d+): (.*)$")
END = re.compile(r"^E?\s*(You can reproduce|Explanation:|Draw \d+:)")


def strip_e(line: str) -> str:
    return line.removeprefix("E")


def examples(lines: list[str]) -> list[dict[int, str]]:
    """Every run of Draw 1..k blocks; a new Draw 1 starts a new example."""
    out: list[dict[int, str]] = []
    i = 0
    while i < len(lines):
        m = DRAW.match(lines[i])
        if not m:
            i += 1
            continue
        n, text = int(m.group(1)), [m.group(2)]
        i += 1
        # Until the next draw; blank lines inside a multi-dim array repr belong to it, so a
        # blank line ends the draw only once its brackets are closed.
        while i < len(lines) and not END.match(lines[i]) and lines[i].startswith("E"):
            line = strip_e(lines[i]).rstrip()
            if not line.strip() and balanced("".join(text)):
                break
            text.append(line)
            i += 1
        if n == 1 or not out:
            out.append({})
        out[-1][n] = "\n".join(text).strip()
    return out


def balanced(s: str) -> bool:
    depth = 0
    for c in s:
        depth += c in "([{"
        depth -= c in ")]}"
        if depth < 0:
            return False
    return depth == 0


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: extract.py <log dir> <out dir>")
    logs, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    for log in sorted(logs.glob("*.log")):
        m = re.fullmatch(r"(test_\w+?)_(\d+)", log.stem)
        if not m or m.group(1) not in LAYOUT:
            continue
        node, layout = m.group(1), LAYOUT[m.group(1)]
        text = log.read_text(errors="replace")
        if "Failing test case" not in text and "Falsifying example" not in text:
            print("PASS/no falsifier:", log.name)
            continue
        lines = text.splitlines()
        repro = [ln.strip() for ln in lines if "@reproduce_failure(" in ln]
        for j, ex in enumerate(examples(lines)):
            name = log.stem + ("" if j == 0 else "_" + "abcdefgh"[j])
            problems = [f"missing Draw {k}" for k in range(1, len(layout) + 1) if k not in ex]
            problems += [f"Draw {k} truncated ('...')" for k, v in ex.items() if "..." in v]
            problems += [f"Draw {k} unbalanced" for k, v in ex.items() if not balanced(v)]
            if len(ex) > len(layout):
                problems.append(f"{len(ex)} draws, expected {len(layout)}")
            d = out_root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "node.txt").write_text(node + "\n")
            for k, part in enumerate(layout, 1):
                if k in ex:
                    (d / f"{part}.txt").write_text(ex[k] + "\n")
            if repro:
                (d / "reproduce.txt").write_text(repro[min(j, len(repro) - 1)] + "\n")
            summary = []
            for part in ("item", "choice", "acts", "seed"):
                if part in layout and layout.index(part) + 1 in ex:
                    v = ex[layout.index(part) + 1]
                    if part in ("item", "choice", "seed"):
                        v = repr(ast.literal_eval(v))
                    summary.append(f"{part}={v}")
            print(
                name,
                node,
                " ".join(summary),
                "PROBLEMS: " + "; ".join(problems) if problems else "ok",
            )


if __name__ == "__main__":
    main()
