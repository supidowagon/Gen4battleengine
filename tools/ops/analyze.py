# Merge per-part pytest outcomes from the pod runs and report failures and skips.
import re, collections, pathlib
L = pathlib.Path("logs")
coll = [l.strip() for l in open("collect.txt", encoding="utf-8") if "::" in l and l.startswith("tests/")]
def order_for(units):
    out = []
    for u in units:
        out += [t for t in coll if t == u or t.startswith(u + "::") or t.startswith(u + "[")]
    return out
def chars(p):
    s = ""
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if re.fullmatch(r"[.sFExX]+( +\[ *\d+%\])?", line):
            s += re.sub(r" *\[.*", "", line)
    return s
res = {}
def add(order, cs, tag):
    for t, c in zip(order, cs):
        res.setdefault(t, []).append((c, tag))
for i in range(10):
    p = f"part_{i:02d}"
    if (L/"rb"/f"order_{p[5:]}.txt").exists():
        order = [l.strip() for l in open(L/"rb"/f"order_{p[5:]}.txt", encoding="utf-8")]
        cs = chars(L/"out"/f"{p}.log")
    else:
        order = order_for([u.strip() for u in open(f"pod/parts/{p}.txt") if u.strip()])
        cs = chars(L/"out"/f"{p}.log")
        if len(order) != len(cs): print("MISMATCH", p, len(order), len(cs))
    add(order, cs, "out/" + p)
for i in range(10):
    p = f"part_{i:02d}"
    order = [l.rstrip("\n") for l in open(L/"parts2"/f"{p}.txt", encoding="utf-8")]
    cs = chars(L/"out2"/f"{p}.log")
    if len(order) != len(cs): print("MISMATCH2", p, len(order), len(cs))
    add(order, cs, "out2/" + p)
cnt = collections.Counter(v[-1][0] for v in res.values())
print("tests with outcome", len(res), "of", len(coll), dict(cnt))
dups = [t for t, v in res.items() if len(v) > 1]
print("run twice", len(dups))
missing = [t for t in coll if t not in res]
print("missing", len(missing), missing[:5])
for t, v in res.items():
    if "clowns_fight" in t: print("CLOWN", t, v)
    if v[-1][0] in "FE": print("FAIL", t, v)
sk = collections.Counter(t.split("::")[0] for t, v in res.items() if v[-1][0] == "s")
print("skipped by file:", sk.most_common())
