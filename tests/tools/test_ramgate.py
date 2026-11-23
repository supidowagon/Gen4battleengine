"""tools/ops/ramgate.py: weight lookup, id selection, the decision, the pytest gate. No real RAM
read, no real waiting: free RAM, the clock and the sleep are injected."""

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ramgate_under_test", ROOT / "tools/ops/ramgate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["ramgate_under_test"] = rg
_spec.loader.exec_module(rg)

TABLE = {
    "tests/a": 3.0,
    "tests/a/test_x.py": 5.0,
    "tests/a/test_x.py::test_one": 9.0,
    "tests/b/test_y.py": 4.0,
    "tests/heavy.py": 30.0,
}


class Clock:
    """A fake clock whose sleep advances it."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _fixture_from(module):
    def fixture():
        pass

    fixture.__module__ = module
    return fixture


ROM_DEF = SimpleNamespace(func=_fixture_from("tests.conftest"))  # a ROM fixture of tests/conftest.py


def item(nodeid, *fixtures, direct=(), indirect=(), local=()):
    """A fake item. Fixtures come from tests/conftest.py. `direct`/`indirect`: parametrize argnames
    (pytest gives a direct one a pseudo fixture def from _pytest.python, an indirect one keeps its
    real fixture def); `local`: a fixture defined in the test module itself."""
    names = [*fixtures, *direct, *indirect, *local]
    defs = {n: [ROM_DEF] for n in (*fixtures, *indirect)}
    defs.update({n: [SimpleNamespace(func=_fixture_from("_pytest.python"))] for n in direct})
    defs.update({n: [ROM_DEF, SimpleNamespace(func=_fixture_from(nodeid.split("::")[0]))] for n in local})
    params = {n: 1 for n in (*direct, *indirect)}
    return SimpleNamespace(
        nodeid=nodeid,
        fixturenames=["request", *names],
        callspec=SimpleNamespace(params=params) if params else None,
        _fixtureinfo=SimpleNamespace(name2fixturedefs=defs),
    )


ENGINE = ["os", "jax", "nuzline.engine.step"]
NO_ENGINE = ["os", "jax", "nuzline.data.tables"]
ROM = item("tests/tools/rom/test_nds.py::test_a", "vanilla_bytes")
PLAIN = item("tests/a/test_x.py::test_b", "tmp_path")  # weight 5
ENG = item("tests/a/test_e.py::test_c")  # weight 3


class FreeLock:
    """A machine lock nobody else holds."""

    path = "fake.lock"
    held = False

    def acquire(self, kind="pytest"):
        self.held = True
        return True

    def holder(self):
        return None, "", None

    def held_by_lineage(self, kind=None):
        return False

    def in_lineage(self, pid, start=None):
        return pid == 4242  # the fake gated parent run


def gate(items, free=100.0, modules=NO_ENGINE, env=None, rom_table=None, **kw):
    clock = kw.pop("clock", None) or Clock()
    lines = kw.pop("lines", [])
    return rg.pytest_gate(
        items,
        modules=modules,
        env={} if env is None else env,
        table=TABLE,
        rom_table={} if rom_table is None else rom_table,
        pod_only=kw.pop("pod_only", ()),
        lock=kw.pop("lock", None) or FreeLock(),
        free=free if callable(free) else (lambda: free),
        sleep=clock.sleep,
        clock=clock.now,
        say=lines.append,
        **kw,
    )


def test_weight_exact_and_prefixes():
    assert rg.weight("tests/a/test_x.py", TABLE) == 5.0
    assert rg.weight("tests/a/test_x.py::test_two", TABLE) == 5.0
    assert rg.weight("tests/a/test_x.py::test_one[p]", TABLE) == 9.0
    assert rg.weight("tests/a/other.py", TABLE) == 3.0
    assert rg.weight("tests/b/test_y.py[x]", TABLE) == 4.0


def test_weight_longest_wins_and_no_match():
    assert rg.weight("tests/a/test_x.py::test_one", TABLE) == 9.0
    assert rg.weight("tests/ab/test_z.py", TABLE) == 20.0  # "tests/a" must end at a separator
    assert rg.weight("tests/zzz.py", TABLE) == 20.0


def test_weight_backslashes():
    assert rg.weight("tests\\a\\test_x.py", TABLE) == 5.0


def test_load_weights(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("# c\ntests/a.py 3 run1\nbad x\n\ntests/b.py 4.5\n", encoding="utf-8")
    assert rg.load_weights(f) == {"tests/a.py": 3.0, "tests/b.py": 4.5}
    assert rg.load_weights() and all(v > 0 for v in rg.load_weights().values())


def test_decide_boundaries():
    assert rg.decide(7.99, 100) == "GO"
    assert rg.decide(8, 100) == "POD"
    assert rg.decide(20, 0) == "POD"
    assert rg.decide(4, 10) == "WAIT"  # free == 6 + w waits
    assert rg.decide(4, 10.01) == "GO"
    assert rg.decide(7.99, 13.99) == "WAIT"


def test_decide_rom_gate_allows_heavy_but_still_waits():
    assert rg.decide(30, 100, rom_gate=True) == "GO"
    assert rg.decide(30, 36, rom_gate=True) == "WAIT"


def test_is_rom_only_by_fixture_and_by_path():
    assert rg.is_rom_only(ROM)
    assert rg.is_rom_only(item("tests/x.py::t", "tmp_path", "pkcalc"))
    assert not rg.is_rom_only(PLAIN)
    assert not rg.is_rom_only(ENG)
    path_test = (
        "tests/kaizo/test_sheet_unmapped.py::test_ally_reasons_name_the_partner_rows_starter"
    )
    assert rg.is_rom_only(item(path_test + "[a]"))
    for f in ("rom_inputs", "species", "kaizo_save_probe_path", "docs", "symbols"):
        assert f in rg.ROM_FIXTURES


def test_engine_loaded_prefix_only():
    assert rg.engine_loaded(["nuzline.engine", "nuzline.engine.step"]) and rg.engine_loaded(ENGINE)
    assert not rg.engine_loaded(NO_ENGINE + ["nuzline.engineering", "jax"])


def test_the_light_engine_modules_do_not_count_as_the_engine_loaded():
    # tools.kaizo.rows_tables imports nuzline.engine.ids (COORD.md 22 Nov 20:01): enums only
    assert not rg.engine_loaded(NO_ENGINE + ["nuzline.engine", "nuzline.engine.ids"])
    assert rg.engine_loaded(["nuzline.engine", "nuzline.engine.ids", "nuzline.engine.state"])
    for name in rg.LIGHT_ENGINE_MODULES:  # each light module imports only the standard library
        tail = "/__init__.py" if name == rg.ENGINE_PREFIX else ".py"
        path = ROOT / "src" / (name.replace(".", "/") + tail)
        heads = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", path.read_text(encoding="utf-8"), re.M)
        assert {h.split(".")[0] for h in heads} <= {"os", "sys", "enum", "__future__"}, (name, heads)


def test_gate_engine_free_run_goes():
    assert gate([PLAIN, ENG], modules=NO_ENGINE)[0] == "GO"  # nothing imported the engine


def test_gate_all_rom_only_goes_even_with_engine_loaded():
    assert gate([ROM, ROM], modules=ENGINE)[0] == "GO"


def test_gate_mixed_run_is_refused_naming_the_quick_lane():
    items = [ROM] + [item(f"tests/e/test_{n}.py::t") for n in range(7)]
    verdict, msg = gate(items, modules=ENGINE)
    assert verdict == "REFUSE"
    assert "quick lane" in msg and "q-<stream>-<sha7>" in msg and "section 5" in msg
    assert "tests/e/test_0.py::t" in msg and "tests/e/test_4.py::t" in msg
    assert "tests/e/test_5.py::t" not in msg and "and 2 more" in msg
    assert "test_nds" not in msg  # the ROM-only test is not listed


def test_rom_gate_lifts_nothing_in_the_pytest_gate():
    for value in ("1", "0"):  # ROM-only items need no override; engine tests are refused anyway
        assert gate([ENG], modules=ENGINE, env={rg.ENV_ROM_GATE: value})[0] == "REFUSE"
    assert gate([ROM], modules=ENGINE, env={rg.ENV_ROM_GATE: "1"})[0] == "GO"


def test_gate_collect_only_allowed_without_waiting():
    clock = Clock()
    assert gate([ENG], free=0.0, modules=ENGINE, collect_only=True, clock=clock)[0] == "GO"
    assert clock.sleeps == []


def test_gate_skipped_on_the_linux_pod_and_inside_a_live_gated_ancestor():
    assert gate([ENG], modules=ENGINE, env={rg.ENV_ON_POD: "1"}, platform="linux")[0] == "SKIP"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_DONE: "4242"})[0] == "SKIP"


def test_the_env_switches_do_not_turn_the_gate_off_on_the_pc():
    on_pc = {"modules": ENGINE, "platform": "win32"}
    assert gate([ENG], env={rg.ENV_ON_POD: "1"}, **on_pc)[0] == "REFUSE"  # the pods are Linux
    for done in ("1", "77", "x", ""):  # not the pid of a live gated ancestor
        assert gate([ENG], env={rg.ENV_DONE: done}, **on_pc)[0] == "REFUSE"
    assert gate([ENG], env={"PYTEST_CURRENT_TEST": "x::y (call)"}, **on_pc)[0] == "REFUSE"
    assert not rg.on_pod({rg.ENV_ON_POD: "1"}, "win32") and rg.on_pod({rg.ENV_ON_POD: "1"}, "linux")


def test_gate_weight_is_the_heaviest_collected_item_not_the_args():
    unknown = item("tests/zzz.py::t", "tmp_path")  # no key: 20 with the engine loaded, else 6
    assert rg.item_weight(unknown, TABLE, {}) == 20.0
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([PLAIN, unknown], free=11.9, env=env)[0] == "TIMEOUT"  # 6 > 5: needs > 12
    assert gate([PLAIN, unknown], free=12.1, env=env)[0] == "GO"
    assert gate([PLAIN], free=11.5)[0] == "GO"  # weight 5 -> needs > 11
    assert gate([PLAIN], free=10.9, env=env)[0] == "TIMEOUT"


def test_an_engine_free_item_without_a_key_weighs_six():
    new_tool_test = item("tests/tools/test_new.py::t", "tmp_path")  # no weights.txt line
    assert rg.ENGINE_FREE_WEIGHT == 6.0
    assert rg.item_weight(new_tool_test, TABLE, {}, engine_free=True) == 6.0
    assert rg.item_weight(new_tool_test, TABLE, {}) == 20.0  # the engine can load: 20
    assert rg.item_weight(PLAIN, TABLE, {}, engine_free=True) == 5.0  # a key still wins
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([new_tool_test], free=12.1, modules=NO_ENGINE, env=env)[0] == "GO"  # > 6 + 6
    assert gate([new_tool_test], free=11.9, modules=NO_ENGINE, env=env)[0] == "TIMEOUT"


def test_rom_only_items_weigh_six_until_measured():
    light = item("tests/b/test_y.py::t[x]", "kaizo_inputs")  # pod table says 4
    assert rg.ROM_UNMEASURED_GB == 6.0
    assert rg.item_weight(light, TABLE, {}, engine_free=True) == 6.0
    assert rg.item_weight(item("tests/zzz.py::t", "docs"), TABLE, {}, engine_free=True) == 6.0  # no pod key
    assert rg.item_weight(item("tests/heavy.py::t", "docs"), TABLE, {}, engine_free=True) == 30.0
    assert rg.item_weight(item("tests/b/test_y.py::t", "tmp_path"), TABLE, {}) == 4.0
    measured = {"tests/b/test_y.py": 7.0, "tests/b/test_y.py::t": 9.0}
    assert rg.item_weight(light, TABLE, measured) == 9.0  # longest prefix wins
    assert rg.item_weight(item("tests/b/test_y.py::u", "docs"), TABLE, measured) == 7.0
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([light], free=12.1, rom_table={}, env=env)[0] == "GO"  # needs > 12
    assert gate([light], free=11.9, rom_table={}, env=env)[0] == "TIMEOUT"
    assert gate([light], free=13.1, rom_table={"tests/b/test_y.py": 7.0}, env=env)[0] == "GO"


def test_a_non_rom_only_item_of_a_measured_file_weighs_the_file_figure_when_engine_free():
    # tests/tools/rom/test_terrain.py::test_battle_terrain_code_guard holds no ROM fixture, so the
    # pod key (6) outranked the file's rom_weights.txt line (4)
    plain = item("tests/b/test_y.py::t", "tmp_path")  # pod table says 4
    assert rg.item_weight(plain, TABLE, {"tests/b/test_y.py": 3.0}, engine_free=True) == 3.0
    assert rg.item_weight(plain, TABLE, {"tests/b/test_y.py": 3.0}) == 4.0  # engine loaded: pod key
    assert rg.item_weight(plain, TABLE, {}, engine_free=True) == 4.0  # no measured line: pod key
    assert rg.item_weight(plain, TABLE, {"tests/other.py": 3.0}, engine_free=True) == 4.0


def test_the_committed_lines_weigh_their_measured_figures():
    rom, pod = rg.load_rom_weights(), rg.load_weights()
    for ident, gb in (
        ("tests/tools/rom/test_terrain.py::test_battle_terrain_code_guard", 4.0),
        ("tests/tools/rom/test_terrain.py::test_terrain_code_guard", 4.0),
        ("tests/kaizo/test_overrides.py::test_rules_left_unset_by_draft", 3.0),
        ("tests/tools/rom/test_code_tables.py::test_tm_moves_match_decomp", 3.0),
        ("tests/tools/rom/test_extract_rom.py::test_vanilla_round_trip", 3.0),
        ("tests/tools/rom/test_offsets.py::test_no_field_accepts_another_fields_offset[platinum_us]", 3.0),
    ):
        assert rg.item_weight(item(ident, "tmp_path"), pod, rom, engine_free=True) == gb, ident


def test_an_unmeasured_rom_only_item_weighs_fourteen_once_the_engine_is_loaded():
    light = item("tests/b/test_y.py::t[x]", "kaizo_inputs")  # pod table says 4
    assert rg.ROM_ENGINE_UNMEASURED_GB == 14.0
    assert rg.item_weight(light, TABLE, {}) == 14.0
    assert rg.item_weight(item("tests/heavy.py::t", "docs"), TABLE, {}) == 30.0  # a higher pod key wins
    assert rg.item_weight(light, TABLE, {"tests/b/test_y.py": 3.0}) == 3.0  # measured wins
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([light], free=19.9, modules=ENGINE, env=env)[0] == "TIMEOUT"  # needs > 20
    assert gate([light], free=20.1, modules=ENGINE, env=env)[0] == "GO"


def test_load_rom_weights(tmp_path):
    f = tmp_path / "rom_weights.txt"
    f.write_text("# c\ntests/a.py 7 local-measure-abc1234-20261122\n", encoding="utf-8")
    assert rg.load_rom_weights(f) == {"tests/a.py": 7.0}
    assert rg.load_rom_weights(tmp_path / "missing.txt") == {}
    assert rg.ROM_WEIGHTS.parent.name == "day" and rg.ROM_WEIGHTS.name == "rom_weights.txt"


def test_species_as_a_parametrize_name_is_not_the_fixture():
    direct = item("tests/data/test_learnsets.py::t[1]", "species", direct=("species",))
    indirect = item("tests/saves/test_to_spec.py::t[1]", "species", indirect=("species",))
    plain_fixture = item("tests/x.py::t", "species")
    assert not rg.is_rom_only(direct)
    assert rg.is_rom_only(indirect)
    assert rg.is_rom_only(plain_fixture)
    assert rg.is_rom_only(item("tests/x.py::t[a]", "tmp_path", "species", direct=("which",)))


def test_a_rom_fixture_counts_only_from_the_three_conftests():
    assert rg.ROM_FIXTURE_MODULES == {
        "tests.conftest", "tests.tools.rom.conftest", "tests.kaizo.conftest"
    }  # fmt: skip
    # a test module's own fixture named like a ROM fixture overrides it: not ROM-only
    assert not rg.is_rom_only(item("tests/engine/test_x.py::t", "tmp_path", local=("docs",)))
    no_defs = item("tests/x.py::t", "tmp_path")
    no_defs.fixturenames.append("symbols")  # a name with no fixturedef at all
    assert not rg.is_rom_only(no_defs)
    for module in sorted(rg.ROM_FIXTURE_MODULES):
        it = item("tests/x.py::t", "tmp_path")
        it.fixturenames.append("pkcalc")
        it._fixtureinfo.name2fixturedefs["pkcalc"] = [SimpleNamespace(func=_fixture_from(module))]
        assert rg.is_rom_only(it)


def test_getfixturevalue_save_export_ids_are_rom_only_for_every_param():
    base = "tests/test_cli.py::test_kaizo_save_export_checks_kaizo_tables"
    for which in ("kaizo_save_pre_roark_path", "kaizo_save_path"):
        assert rg.is_rom_only(item(f"{base}[{which}]", "tmp_path", "request"))


def test_gate_waits_then_goes():
    clock = Clock()
    seq = iter([5.0, 9.0, 11.5])
    lines = []
    verdict, _ = gate([PLAIN], free=lambda: next(seq), clock=clock, lines=lines)
    assert verdict == "GO"  # weight 5: needs > 11
    assert clock.sleeps == [60.0, 60.0]
    assert len(lines) == 2 and "need > 11" in lines[0]


def test_gate_wait_timeout_exits():
    clock = Clock()
    verdict, msg = gate([PLAIN], free=7.0, env={rg.ENV_TIMEOUT: "3"}, clock=clock)
    assert verdict == "TIMEOUT"
    assert clock.sleeps == [60.0] * 3
    assert "3 min" in msg


OK_CHILD = [sys.executable, "-c", "pass"]


def test_cli_pod_rom_gate_and_build(monkeypatch, tmp_path):
    monkeypatch.delenv(rg.ENV_ROM_GATE, raising=False)
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    out = []

    def run(argv, free=100.0):
        lock = rg.MachineLock(tmp_path / "ramgate.lock")
        return rg.main(
            argv, free=lambda: free, sleep=lambda s: None, say=out.append, lock=lock,
            platform="win32",
        )  # fmt: skip

    assert run(["tests/a/test_x.py"]) == 0 and out[-1] == "GO 5"
    assert run(["tests/heavy.py"]) == 13 and out[-1] == "POD 30"
    assert run(["tests/heavy.py", "--rom-gate"]) == 0 and out[-1] == "GO 30"
    assert run(["tools.kaizo.x", "--weight", "9"]) == 13 and out[-1] == "POD 9"
    # a build never goes to the pod: 6 without a tools/build_tables.py line, only the RAM wait applies
    assert run(["--build", "--", *OK_CHILD]) == 0 and out[-1] == "GO 6"
    assert run(["--build", "--weight", "12", "--", *OK_CHILD]) == 0 and out[-1] == "GO 12"
    monkeypatch.setattr(rg, "load_weights", lambda: {**TABLE, "tools/build_tables.py": 11.0})
    assert run(["--build", "--", *OK_CHILD]) == 0 and out[-1] == "GO 11"  # the measured line
    assert run(["--build", "--weight", "12", "--", *OK_CHILD]) == 0 and out[-1] == "GO 12"
    assert rg.MachineLock(tmp_path / "ramgate.lock").holder()[0] is None  # released each time


def test_cli_weight_without_a_value_is_a_usage_error():
    out = []
    assert rg.main(["tests/a.py", "--weight"], say=out.append) == 2
    assert rg.main(["--weight", "x"], say=out.append) == 2
    assert "usage" in out[0]


def test_cli_build_needs_a_command_and_only_build_takes_one():
    out = []
    assert rg.main(["--build"], say=out.append) == 2 and "needs a command" in out[-1]
    assert rg.main(["tests/a.py", "--", "x"], say=out.append) == 2


def test_cli_honours_the_timeout_env(monkeypatch, tmp_path):
    monkeypatch.setenv(rg.ENV_TIMEOUT, "2")
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    clock = Clock()
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=lambda: 1.0, sleep=clock.sleep, clock=clock.now,
        say=out.append, lock=rg.MachineLock(tmp_path / "l"), platform="win32",
    )  # fmt: skip
    assert code == 14 and clock.sleeps == [60.0, 60.0] and out[-1] == "TIMEOUT 6"


def test_cli_build_waits_for_ram(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    seq = iter([10.0, 11.5, 12.5])  # need > 6 + 6 (the unlisted-build default)
    sleeps = []
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=lambda: next(seq), sleep=sleeps.append,
        say=out.append, lock=rg.MachineLock(tmp_path / "l"), platform="win32",
    )  # fmt: skip
    assert code == 0 and sleeps == [60.0, 60.0] and out[-1] == "GO 6"
    assert not any(m.startswith("POD") for m in out)


def test_cli_build_runs_the_command_as_its_child_under_the_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    lock_file = tmp_path / "ramgate.lock"
    seen = tmp_path / "seen.txt"
    child = [
        sys.executable, "-c",
        "import sys; open(sys.argv[1], 'w').write(open(sys.argv[2]).read()); sys.exit(7)",
        str(seen), str(lock_file),
    ]  # fmt: skip
    code = rg.main(
        ["--build", "--", *child], free=lambda: 100.0, say=lambda m: None,
        lock=rg.MachineLock(lock_file), platform="win32",
    )  # fmt: skip
    assert code == 7  # the child's exit code
    import os

    assert seen.read_text().split()[:2] == [str(os.getpid()), "build"]  # it ran under my lock
    assert rg.MachineLock(lock_file).holder()[0] is None and not rg.MachineLock(lock_file)._busy()


def test_cli_build_waits_for_the_lock_within_the_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv(rg.ENV_TIMEOUT, "1")
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    lock_file = tmp_path / "ramgate.lock"
    procs, born = {1: 0, 4242: 1, 100: 1}, {1: 0.0, 4242: 1.0, 100: 2.0}  # 4242: no ancestor
    other = rg.MachineLock(lock_file, procs=lambda: procs, started=born.get, me=4242)
    assert other.acquire("pytest")
    clock = Clock()
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=lambda: 100.0, sleep=clock.sleep, clock=clock.now,
        say=out.append, platform="win32",
        lock=rg.MachineLock(lock_file, procs=lambda: procs, started=born.get, me=100),
    )  # fmt: skip
    assert code == 14 and out[-1] == "TIMEOUT 6"
    assert clock.sleeps == [10.0] * 6  # 10 s polls, up to the 1 minute timeout
    told = [m for m in out if "holds the lock" in m]
    assert len(told) == 1 and "pid 4242" in told[0] and str(lock_file) in told[0]
    assert "end pid 4242" in told[0]  # how to clear it
    assert other.owned and other.holder()[:2] == (4242, "pytest")  # never touched
    other.release()


def test_cli_build_on_the_pod_or_off_windows_runs_the_command_directly(monkeypatch, tmp_path):
    def no_ram():
        raise AssertionError("free RAM read")

    lock_file = tmp_path / "_scratch" / "ramgate.lock"
    child = [sys.executable, "-c", "import sys; sys.exit(5)"]
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    for platform, pod in (("linux", "1"), ("linux", ""), ("darwin", "")):
        monkeypatch.setenv(rg.ENV_ON_POD, pod)
        out = []
        code = rg.main(
            ["--build", "--", *child], free=no_ram, say=out.append, platform=platform,
            lock=rg.MachineLock(lock_file),
        )  # fmt: skip
        assert code == 5 and out == []  # no GO line: no wait, no lock, no watchdog
        assert not lock_file.exists() and not lock_file.parent.exists()
    monkeypatch.setenv(rg.ENV_ON_POD, "1")  # ignored on win32: the gate applies
    out = []
    code = rg.main(
        ["--build", "--", *child], free=lambda: 100.0, say=out.append, platform="win32",
        lock=rg.MachineLock(lock_file),
    )  # fmt: skip
    assert code == 5 and out[-1].startswith("GO ")


POD_DIR = ROOT / "tools/ops/day/pod"
_PYTEST_START = re.compile(
    r"-m\s+pytest\b"  # python -m pytest
    r"|\buv\s+run(\s+--?[\w=-]+)*\s+pytest\b"  # uv run [opts] pytest
    r"|(^|[;&|`]|\$\(|^\s*\(|\b(exec|then|do|else)\b)\s*(\w+=\S*\s+)*pytest(\s|$)"  # a command
    r"|\S*/pytest(\s|[\"']|$)"  # a path to the binary: .venv/bin/pytest, "$repo"/.venv/bin/pytest
    r"|(^|[\s;&|(])pytest\s+-"  # behind any word with options: timeout 900 pytest -q, nohup pytest -x
)
_SEARCHES = re.compile(r"\b(pgrep|pkill|grep|egrep|fgrep)\b[^;&|]*")  # a pattern, not a start
_UNSETS = re.compile(r"\bunset\s+([\w\s]*\s)?NUZLINE_ON_POD\b|\benv\s+(-\S+\s+)*-u\s*NUZLINE_ON_POD\b|\benv\s+(-\S+\s+)*-i\b|\benv\s+--ignore-environment\b|\benv\s+--unset[= ]NUZLINE_ON_POD\b")
_PY_SCRIPT = re.compile(r"\bpython3?\S*\"?\s+\"?[^\s\"]*?([\w-]+\.py)\b")
_PY_INLINE = re.compile(r"\bpython3?\S*\"?\s+(-c\s|-\s|-$)|\"\$PY\"\s+-")
_MARKER = "NUZLINE_ON_POD=1"


def _code(line):
    """The part of a shell or Python line that runs: no comment line, nothing after ` #`."""
    return "" if line.lstrip().startswith("#") else line.split(" #")[0]


def _imports_engine(text):
    return bool(re.search(r"^\s*(from|import)\s+(nuzline\.engine|tests\.engine)\b", text, re.M))


def _inline_block(lines, i):
    """The inline Python a `python -c '...'` / `python - <<'TAG'` on line i runs."""
    tag = re.search(r"<<-?'?(\w+)'?", lines[i])
    end = i + 1
    while end < len(lines) and end - i < 400:
        if (tag and lines[end].strip() == tag.group(1)) or (not tag and "'" in lines[end]):
            break
        end += 1
    return "\n".join(lines[i : end + 1])


def pod_marker_problems(path, find_script=None):
    """Lines of a pod script that start pytest, or python importing the engine, while the pod
    marker is not in effect: neither an earlier top-level `export NUZLINE_ON_POD=1` nor the
    marker on that line itself. Also every line that drops the marker (`unset NUZLINE_ON_POD`,
    `env -u NUZLINE_ON_POD`, `env -i`). find_script(name) -> text of a python script it runs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    exported_at = next(
        (n for n, ln in enumerate(lines) if re.match(r"export\s.*\b" + _MARKER + r"\b", _code(ln))),
        None,
    )
    bad = []
    for n, line in enumerate(lines):
        code = _code(line)
        if not code.strip():
            continue
        if _UNSETS.search(code):
            bad.append(f"{path.name}:{n + 1}: drops the marker: {line.strip()[:100]}")
            continue
        starts = bool(_PYTEST_START.search(_SEARCHES.sub(" ", code)))
        if not starts and _PY_INLINE.search(code):
            starts = _imports_engine(_inline_block(lines, n))
        if not starts:
            m = _PY_SCRIPT.search(code)
            text = find_script(m.group(1)) if (m and find_script) else None
            starts = bool(text) and _imports_engine(text)
        if path.suffix == ".py" and _imports_engine(code):
            starts = True  # a pod Python file that imports the engine itself
        marked = _MARKER in code or (exported_at is not None and exported_at < n)
        if starts and not marked:
            bad.append(f"{path.name}:{n + 1}: {line.strip()[:120]}")
    return bad


def _repo_script(name):
    hits = [p for p in (ROOT / "tools").rglob(name) if p.is_file()]
    return "\n".join(p.read_text(encoding="utf-8") for p in hits) or None


def test_every_pod_script_that_starts_pytest_or_the_engine_sets_the_pod_marker():
    """Generic: every file under tools/ops/day/pod/, so a new pod script cannot miss it."""
    files = sorted(p for p in POD_DIR.iterdir() if p.is_file() and p.suffix in (".sh", ".py"))
    assert {"unit.sh", "setup.sh", "quick.sh", "sweep.sh"} <= {p.name for p in files}
    bad = [b for p in files for b in pod_marker_problems(p, _repo_script)]
    assert bad == [], "pod scripts without NUZLINE_ON_POD=1 in effect:\n" + "\n".join(bad)


def test_the_pod_marker_check_catches_each_way_to_miss_it(tmp_path):
    def problems(text, scripts=None):
        f = tmp_path / "x.sh"
        f.write_text(text, encoding="utf-8")
        return pod_marker_problems(f, (scripts or {}).get)

    run = 'uv run --frozen pytest -q "$@"\n'
    assert problems("#!/bin/bash\n" + run)  # no marker at all
    assert problems("# export NUZLINE_ON_POD=1\n" + run)  # commented out
    assert problems("echo x  # NUZLINE_ON_POD=1\n" + run)  # only in a comment
    assert problems(run + "export NUZLINE_ON_POD=1\n")  # set after the start
    assert problems('"$py" -m pytest -x "$node"\n')
    assert problems('.venv/bin/pytest -q tests\n')
    assert problems('"$repo/.venv/bin/python" "$tools/replay.py" a\n', {"replay.py": "import nuzline.engine.step\n"})
    assert problems("python - <<'PY'\nfrom nuzline.engine import ids\nPY\n")
    assert problems("x=$(python -c '\nimport tests.engine.helpers\n')\n")
    # fine: marker exported first, or on the line itself; engine-free python; no start at all
    assert not problems("export NUZLINE_ON_POD=1\n" + run)
    assert not problems("export A=1 NUZLINE_ON_POD=1 B=2\n" + run)
    assert not problems("NUZLINE_ON_POD=1 " + run)
    assert not problems("bash -c 'export NUZLINE_ON_POD=1; exec uv run pytest -q'\n")
    assert not problems('python "$t/extract.py" a\n', {"extract.py": "import re\n"})
    assert not problems("python -c 'from nuzline.jaxcache import jax_version'\n")
    assert not problems('echo "pytest rc $PRC"\npgrep -f "/.venv/bin/pytest"\n')
    assert not problems('grep -c pytest "$D/quick.pytest.log"\necho "outlived pytest: killed"\n')
    assert not problems('echo "QUICK_FAIL rc=$RC (pytest rc $PRC) $SUM"\n')


@pytest.mark.parametrize(
    "start",
    [
        "pytest tests/test_x.py\n",
        "  pytest -q tests\n",
        "exec pytest tests\n",
        "A=1 B=2 pytest tests/x.py::t\n",
        "cd r && pytest tests\n",
        "x=$(pytest --co -q tests)\n",
        "(pytest tests)\n",
        ".venv/bin/pytest tests\n",
        '"$repo/.venv/bin/pytest" tests\n',
        '"$repo"/.venv/bin/pytest\n',
        "nice -n 5 /work/r/.venv/bin/pytest -x\n",
        "timeout 900 pytest -q tests\n",
        "if pytest -q tests; then echo ok; fi\n",
        "nohup pytest -x tests &\n",
    ],
)
def test_the_pod_marker_check_sees_every_way_to_start_pytest(tmp_path, start):
    f = tmp_path / "x.sh"
    f.write_text(start, encoding="utf-8")
    assert pod_marker_problems(f)
    f.write_text("export NUZLINE_ON_POD=1\n" + start, encoding="utf-8")
    assert not pod_marker_problems(f)


@pytest.mark.parametrize(
    "drop",
    [
        "unset NUZLINE_ON_POD\n",
        "unset A NUZLINE_ON_POD\n",
        "env -u NUZLINE_ON_POD bash x.sh\n",
        "env -uNUZLINE_ON_POD bash x.sh\n",
        "env -i PATH=/usr/bin bash x.sh\n",
        "env --ignore-environment bash x.sh\n",
        "env --unset=NUZLINE_ON_POD bash x.sh\n",
    ],
)
def test_a_pod_script_may_not_drop_the_marker(tmp_path, drop):
    f = tmp_path / "x.sh"
    f.write_text("export NUZLINE_ON_POD=1\n" + drop + "uv run pytest -q\n", encoding="utf-8")
    assert any("drops the marker" in b for b in pod_marker_problems(f))


def test_blocker_blocks_engine_imports_only():
    b = rg.EngineBlocker()
    for name in ("nuzline.engine", "nuzline.engine.x", "nuzline.engine.step.y"):
        with pytest.raises(ImportError, match="quick lane"):
            b.find_spec(name)
    for name in ("nuzline.engineering", "nuzline.data.tables", "jax", "os"):
        assert b.find_spec(name) is None


def test_blocker_in_a_fake_run_stops_the_import_and_is_removed(monkeypatch):
    monkeypatch.delenv(rg.ENV_ENGINE_BLOCKED, raising=False)  # restored at teardown
    path = []
    b = rg.install_blocker(path)
    assert path == [b]
    # the real import machinery, with the finder first on a copy of the path
    saved = sys.meta_path
    sys.meta_path = [b, *saved]
    try:
        with pytest.raises(ImportError, match="quick lane"):
            importlib.import_module("nuzline.engine.fake_ramgate_probe")
        assert "nuzline.engine.fake_ramgate_probe" not in sys.modules
        assert importlib.import_module("json")
    finally:
        sys.meta_path = saved
    rg.remove_blocker(b, path)
    assert path == []
    rg.remove_blocker(b, path)  # twice is harmless


def test_blocker_wanted_for_engine_free_runs_and_unmeasured_rom_only_runs():
    listed = {ROM.nodeid: 3.0}
    no = lambda: False  # noqa: E731
    assert rg.blocker_wanted([PLAIN, ROM], {}, listed, no)
    assert rg.blocker_wanted([PLAIN], {rg.ENV_ROM_GATE: "1"}, listed, no)  # no override any more
    assert not rg.blocker_wanted([ROM, ROM], {}, listed, no)  # measured: the engine may load
    assert rg.blocker_wanted([ROM], {}, {}, no)  # unmeasured: its 6 GB assumes no engine
    assert not rg.blocker_wanted([ROM], {}, {}, lambda: True)  # the measuring harness runs it


def test_blocker_sets_the_child_env_and_removal_unsets_it(monkeypatch):
    monkeypatch.delenv(rg.ENV_ENGINE_BLOCKED, raising=False)
    path = []
    b = rg.install_blocker(path)
    assert rg.os.environ[rg.ENV_ENGINE_BLOCKED] == "1"
    rg.remove_blocker(b, path)
    assert rg.ENV_ENGINE_BLOCKED not in rg.os.environ


def _child_env(**extra):
    """The environment of a child process with none of the gate's own variables inherited."""
    import os

    drop = (rg.ENV_ENGINE_BLOCKED, rg.ENV_ON_POD, rg.ENV_DONE, "PYTEST_CURRENT_TEST", "PYTEST_ADDOPTS")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    return {**env, "PYTHONPATH": str(ROOT / "src"), **extra}


def test_engine_package_refuses_a_child_process_under_the_env():
    import subprocess

    def child(value):
        env = _child_env() if value is None else _child_env(**{rg.ENV_ENGINE_BLOCKED: value})
        return subprocess.run(
            [sys.executable, "-c", "import nuzline.engine"],
            env=env, capture_output=True, text=True, check=False,
        )  # fmt: skip

    bad = child("1")
    assert bad.returncode != 0 and "ImportError" in bad.stderr
    assert "NUZLINE_ENGINE_BLOCKED is set" in bad.stderr and "quick lane" in bad.stderr
    assert "unset it" in bad.stderr
    for value in (None, "", "0"):  # unset, empty and "0" are unset, as ramgate._flag reads them
        ok = child(value)
        assert ok.returncode == 0, (value, ok.stderr)


_ENGINE_FILE = "tests/engine/test_field_pass.py"


@pytest.mark.skipif(sys.platform != "win32", reason="the check guards the PC; the pods are Linux")
@pytest.mark.parametrize(
    "args, addopts",
    [
        (["--noconftest"], None),
        (["--confcutdir", "tests"], None),
        (["-p", "no:" + str(ROOT / "conftest.py")], None),
        ([], "--confcutdir tests"),
    ],
    ids=["noconftest", "confcutdir", "p-no-root-conftest", "pytest-addopts"],
)
def test_a_pytest_without_the_gate_cannot_import_the_engine(args, addopts):
    """Each flag keeps the root conftest (the gate) from loading; the engine package then refuses.
    Collect-only: if the check were missing this only imports the engine, it runs no test.
    win32-only, so no pod run covers it: run this file locally once per change to
    src/nuzline/engine/__init__.py or the root conftest.py (docs/process.md section 5)."""
    import subprocess

    env = _child_env(**({"PYTEST_ADDOPTS": addopts} if addopts else {}))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args,
         _ENGINE_FILE],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    out = run.stdout + run.stderr
    assert run.returncode != 0, out[-2000:]
    # pytest 9.2 itself refuses `-p no:<conftest path>` as a usage error; either way nothing runs
    refused = ("pytest runs without the local test gate", "Blocking conftest files using -p")
    assert any(r in out for r in refused), out[-2000:]


def _body_fetching_a_rom_fixture(request):
    return request.getfixturevalue("kaizo_save_path")


def _body_fetching_a_plain_fixture(request):
    return request.getfixturevalue("tmp_path"), request.getfixturevalue(request.param)


def test_a_literal_getfixturevalue_of_a_rom_fixture_is_rom_only():
    fetching = item("tests/x.py::t", "request")
    fetching.function = _body_fetching_a_rom_fixture
    plain = item("tests/x.py::t", "request")
    plain.function = _body_fetching_a_plain_fixture
    assert rg.is_rom_only(fetching)
    assert not rg.is_rom_only(plain)  # a name held in a variable needs ROM_ONLY_IDS
    plain.function = print  # no source for a builtin: not ROM-only, no crash
    assert not rg.is_rom_only(plain)


def test_the_dynamic_getfixturevalue_tests_are_listed():
    assert len(rg.ROM_ONLY_IDS) == 6
    for name in sorted(rg.ROM_ONLY_IDS):  # every listed id names a real test
        file, _, test = name.partition("::")
        assert f"def {test}(" in (ROOT / file).read_text(encoding="utf-8")


def _body_with_a_conditional_fetch(request, g):
    return request.getfixturevalue("tables" if g else "kaizo_save_path")


def test_a_conditional_getfixturevalue_counts_every_literal():
    it = item("tests/x.py::t", "request")
    it.function = _body_with_a_conditional_fetch
    assert rg.is_rom_only(it)


def test_timeout_message_names_the_env_and_the_weights_files():
    verdict, msg = gate([PLAIN], free=1.0, env={rg.ENV_TIMEOUT: "0"})
    assert verdict == "TIMEOUT"
    assert rg.ENV_TIMEOUT in msg and "weights.txt" in msg and "rom_weights.txt" in msg


def test_refusal_says_ids_never_a_file_path_and_names_no_override():
    _, msg = gate([ENG], modules=ENGINE)
    assert "never a file path" in msg and "--rom-ids" in msg
    assert rg.ENV_ROM_GATE not in msg and "last resort" not in msg


def test_rom_ids_group_by_file_and_print_ready_commands():
    a1 = item("tests/test_cli.py::test_a[x y]", "kaizo_save_path")
    a2 = item("tests/test_cli.py::test_engine", "tmp_path")
    a3 = item("tests/test_cli.py::test_b", "docs")
    b1 = item("tests/kaizo/test_z.py::test_c", "pkcalc")
    by_file = rg.rom_ids_by_file([a1, a2, a3, b1])
    assert by_file == {
        "tests/test_cli.py": ["tests/test_cli.py::test_a[x y]", "tests/test_cli.py::test_b"],
        "tests/kaizo/test_z.py": ["tests/kaizo/test_z.py::test_c"],
    }
    lines = rg.rom_commands(by_file)
    assert lines[0] == (
        "python -m pytest -p no:cacheprovider -q 'tests/test_cli.py::test_a[x y]' tests/test_cli.py::test_b"
    )
    assert len(lines) == 2 and "test_engine" not in lines[0]


def test_no_wait_counts_only_under_the_measuring_harness_lock(tmp_path):
    procs = {1: 0, 50: 1, 51: 50, 60: 1}  # 50 = harness, 51 = its pytest; 60 unrelated
    born = {1: 0.0, 50: 1.0, 51: 2.0, 60: 3.0}.get
    path = tmp_path / "ramgate.lock"
    harness = rg.MachineLock(path, procs=lambda: procs, started=born, me=50)
    assert harness.acquire("measure")
    clock = Clock()
    env = {rg.ENV_NO_WAIT: "1"}
    child = rg.MachineLock(path, procs=lambda: procs, started=born, me=51)
    assert gate([PLAIN], free=0.0, env=env, clock=clock, lock=child)[0] == "GO"
    assert clock.sleeps == []
    assert gate([ENG], free=0.0, modules=ENGINE, env=env, lock=child)[0] == "REFUSE"
    stranger = rg.MachineLock(path, procs=lambda: procs, started=born, me=60)  # NO_WAIT: ignored
    lines = []
    env_t = {**env, rg.ENV_TIMEOUT: "0"}
    assert gate([PLAIN], free=0.0, env=env_t, lock=stranger, lines=lines)[0] == "TIMEOUT"
    assert any("ignored" in m for m in lines)
    harness.release()
    assert harness.acquire("pytest")  # a plain run's lock is not the harness's
    child2 = rg.MachineLock(path, procs=lambda: procs, started=born, me=51)
    assert gate([PLAIN], free=0.0, env=env_t, lock=child2)[0] == "TIMEOUT"
    harness.release()


def test_the_committed_rom_weights_file_parses():
    lines = [
        ln.split()
        for ln in rg.ROM_WEIGHTS.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines and all(len(p) == 3 and float(p[1]) > 0 and p[2] for p in lines)
    table = rg.load_rom_weights()
    assert len(table) == len(lines) and all(k.startswith("tests/") for k in table)
    assert rg.lookup("tests/tools/rom/test_text.py::test_x[1]", table) == 3.0


PROCS = {1: 0, 10: 1, 11: 10, 12: 11, 20: 1, 30: 1}  # 12 -> 11 -> 10 -> 1; 20 and 30 unrelated
BORN = {1: 0.0, 10: 1.0, 11: 2.0, 12: 3.0, 20: 1.5, 30: 1.5}  # every parent started first


def _lock(tmp_path, me, procs=PROCS, born=BORN):
    return rg.MachineLock(
        tmp_path / "_scratch" / "ramgate.lock", procs=lambda: procs, started=born.get, me=me
    )


def test_lock_is_exclusive_and_released(tmp_path):
    a, b = _lock(tmp_path, 10), _lock(tmp_path, 20)
    assert a.acquire() and a.owned
    assert a.holder() == (10, "pytest", 1.0)
    assert not b.acquire() and not b.held  # an OS lock held by another open file
    a.release()
    assert a.holder() == (None, "", None) and a.path.exists()  # the file stays, empty
    assert b.acquire("build") and b.holder() == (20, "build", 1.5)
    b.release()


def test_a_line_naming_a_dead_holder_is_no_lock(tmp_path):
    a = _lock(tmp_path, 10)
    a.path.parent.mkdir(parents=True)
    a.path.write_text("999 pytest 7.0\n", encoding="utf-8")  # its writer died: no OS lock
    assert a.acquire() and a.owned and a.holder() == (10, "pytest", 1.0)
    a.release()


def test_lock_held_by_an_ancestor_is_inherited_and_left_alone(tmp_path):
    parent, child = _lock(tmp_path, 10), _lock(tmp_path, 12)
    assert parent.acquire()
    assert child.acquire() and child.inherited and not child.owned
    assert child.held_by_lineage() and child.held_by_lineage("pytest")
    assert not child.held_by_lineage("measure")
    child.release()
    assert parent.owned and parent.holder()[0] == 10  # the child leaves its ancestor's lock alone
    assert not _lock(tmp_path, 30).acquire() and not _lock(tmp_path, 30).held_by_lineage()
    parent.release()
    assert not child.held_by_lineage()  # a line alone is no hold: the OS lock is free


def test_a_live_pid_with_another_start_time_counts_as_gone(tmp_path):
    born = dict(BORN)
    parent = _lock(tmp_path, 10, born=born)
    assert parent.acquire()  # its line: 10 pytest 1.0
    born[10] = 9.0  # pid 10 now belongs to a process that started later (a reused pid)
    child = _lock(tmp_path, 12, born=born)
    assert not child.acquire() and not child.held and not child.held_by_lineage()
    assert not child.in_lineage(10) and rg.lineage(12, PROCS, born.get) == [12, 11]
    parent.release()


def test_lineage_checks_start_times():
    assert rg.lineage(12, PROCS, BORN.get) == [12, 11, 10, 1]
    late = {**BORN, 11: 5.0}  # 11 started after its child 12: a reused parent pid
    assert rg.lineage(12, PROCS, late.get) == [12]
    unknown = {**BORN, 10: None}
    assert rg.lineage(12, PROCS, unknown.get) == [12, 11]
    assert rg.lineage(1, {1: 2, 2: 1}, lambda p: 0.0) == [1, 2]  # a pid cycle ends


def test_inside_gated_run_needs_a_start_time_checked_ancestor(tmp_path):
    born = dict(BORN)
    child = _lock(tmp_path, 12, born=born)
    assert rg.inside_gated_run({rg.ENV_DONE: "10"}, child)
    born[11] = 8.0  # the recorded parent of 12 is a reused pid now
    assert not rg.inside_gated_run({rg.ENV_DONE: "10"}, child)


def test_the_lock_is_freed_when_its_holder_dies_without_unconfigure(tmp_path):
    import subprocess

    path = tmp_path / "_scratch" / "ramgate.lock"
    code = (
        "import importlib.util, sys, time; "
        f"s = importlib.util.spec_from_file_location('r', {str(ROOT / 'tools/ops/ramgate.py')!r}); "
        "r = importlib.util.module_from_spec(s); s.loader.exec_module(r); "
        f"lk = r.MachineLock({str(path)!r}); assert lk.acquire('build'); "
        "print('LOCKED', flush=True); time.sleep(120)"
    )
    holder = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        mine = rg.MachineLock(path)
        assert not mine.acquire() and mine.holder()[1] == "build"
        assert "holds the lock" in mine.busy_message()
    finally:
        holder.kill()  # no release, no unconfigure: a crash
        holder.wait(30)
    import time

    # the kernel frees it when the holder ends; on Windows a venv python.exe is a launcher whose
    # real python child ends a moment after the launcher is killed
    for _ in range(100):
        if mine.acquire():
            break
        time.sleep(0.1)
    assert mine.owned
    mine.release()


def test_gate_waits_for_the_lock_then_goes_and_holds_it(tmp_path):
    other, lock = _lock(tmp_path, 20), _lock(tmp_path, 12)
    assert other.acquire()
    clock = Clock()

    def sleep(s):
        clock.sleep(s)
        other.release()  # the other run ends during the first wait

    lines = []
    verdict, _ = rg.pytest_gate(
        [PLAIN], modules=NO_ENGINE, env={}, table=TABLE, rom_table={}, pod_only=(), lock=lock,
        free=lambda: 100.0, sleep=sleep, clock=clock.now, say=lines.append,
    )  # fmt: skip
    assert verdict == "GO" and lock.owned and lock.holder() == (12, "pytest", 3.0)
    assert clock.sleeps == [10.0] and "holds the lock" in lines[0] and "pid 20" in lines[0]
    lock.release()


def test_gate_lock_wait_times_out_with_a_message_naming_the_lock(tmp_path):
    other, lock = _lock(tmp_path, 20), _lock(tmp_path, 12)
    assert other.acquire()
    verdict, msg = gate([PLAIN], env={rg.ENV_TIMEOUT: "0"}, lock=lock)
    assert verdict == "TIMEOUT" and "ramgate.lock" in msg and not lock.held
    other.release()


def test_process_table_and_lineage_of_this_process():
    import os

    procs = rg.process_table()
    me = rg.lineage(os.getpid(), procs)
    assert me[0] == os.getpid() and len(me) >= 2  # at least the process that started pytest


def test_the_machine_lock_lives_in_the_main_checkout(tmp_path):
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n", encoding="utf-8")
    wt = tmp_path / "main" / "_scratch" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n", encoding="utf-8")
    assert rg.lock_path(wt) == (main / "_scratch" / "ramgate.lock").resolve()
    assert rg.lock_path(tmp_path / "plain") == tmp_path / "plain" / "_scratch" / "ramgate.lock"


def test_xdist_is_refused_off_the_pod():
    opt = SimpleNamespace
    assert rg.xdist_refusal(opt(numprocesses=2, dist="load"), {}, "win32")
    assert rg.xdist_refusal(opt(numprocesses="auto"), {}, "win32")
    assert rg.xdist_refusal(opt(dist="loadfile"), {}, "linux")  # Linux without the pod marker
    assert "pytest-xdist" in rg.xdist_refusal(opt(numprocesses=1), {}, "win32")
    assert rg.xdist_refusal(opt(numprocesses=2), {rg.ENV_ON_POD: "1"}, "win32")  # ignored on win32
    assert rg.xdist_refusal(opt(numprocesses=2), {rg.ENV_ON_POD: "1"}, "linux") is None
    assert rg.xdist_refusal(opt(), {}, "win32") is None  # xdist not installed: no such options
    assert rg.xdist_refusal(opt(numprocesses=None, dist="no"), {}, "win32") is None


def test_pod_only_tests_are_refused_on_the_pc_even_engine_free():
    keys = {"tests/test_spike_step.py", "tests/x.py::t"}
    spike = item("tests/test_spike_step.py::test_jit[a]", "tmp_path")
    verdict, msg = gate([PLAIN, spike], modules=NO_ENGINE, pod_only=keys)
    assert verdict == "REFUSE" and "pod_only.txt" in msg and "test_spike_step" in msg
    assert "quick lane" in msg and "tests/a/test_x.py" not in msg
    env = {rg.ENV_ROM_GATE: "1"}
    assert gate([spike], modules=NO_ENGINE, pod_only=keys, env=env)[0] == "REFUSE"
    assert gate([item("tests/x.py::tt")], modules=NO_ENGINE, pod_only=keys)[0] == "GO"
    assert gate([spike], pod_only=keys, collect_only=True)[0] == "GO"  # collecting is fine


def test_a_rom_and_engine_pod_only_test_gets_no_quick_lane_hint():
    # review_q3_merge_85fd5160 SHOULD: the quick lane has no ROM, so it cannot run these
    reasons = {"tests/test_spike_step.py": "jit", "tests/k/test_roark.py::t": "rom-and-engine-split-close"}
    roark = item("tests/k/test_roark.py::t", "docs")
    verdict, msg = gate([roark], modules=NO_ENGINE, pod_only=reasons)
    assert verdict == "REFUSE" and "test_roark" in msg and "ROM + engine" in msg
    assert "quick lane, run name" not in msg and "owner question" in msg
    spike = item("tests/test_spike_step.py::test_jit[a]", "tmp_path")
    verdict, msg = gate([spike, roark], modules=NO_ENGINE, pod_only=reasons)
    assert verdict == "REFUSE" and "quick lane, run name" in msg and "ROM + engine" in msg
    committed = rg.load_pod_only_reasons()
    assert set(committed) == rg.load_pod_only()
    assert sum(r.startswith(rg.ROM_AND_ENGINE_REASON) for r in committed.values()) == 5


def test_the_committed_pod_only_list():
    keys = rg.load_pod_only()
    assert {"tests/test_spike_step.py", "tests/test_jaxcache.py"} <= keys
    assert "tests/tools/test_ramgate_collect.py" in keys
    assert all((ROOT / k.split("::")[0]).exists() for k in keys)


def test_exit_codes_are_apart_from_pytests_own():
    import pytest as _pytest

    ours = {rg.EXIT_REFUSE, rg.EXIT_TIMEOUT, rg.EXIT_NO_RAM_READ}
    assert len(ours) == 3 and not ours & {int(c) for c in _pytest.ExitCode}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "x", ""])
def test_a_timeout_that_is_not_a_finite_count_of_minutes_is_the_default(value):
    assert rg.timeout_minutes({rg.ENV_TIMEOUT: value}) == rg.DEFAULT_TIMEOUT_MIN
    assert rg.timeout_minutes({rg.ENV_TIMEOUT: "2.5"}) == 2.5
    assert rg.timeout_minutes({}) == rg.DEFAULT_TIMEOUT_MIN
    clock = Clock()
    verdict, msg = gate([PLAIN], free=1.0, env={rg.ENV_TIMEOUT: value}, clock=clock)
    assert verdict == "TIMEOUT" and len(clock.sleeps) == 30  # 30 min, not endless


def test_an_unreadable_free_ram_figure_is_an_error_not_a_crash(monkeypatch, tmp_path):
    def broken():
        raise OSError("GlobalMemoryStatusEx failed")

    verdict, msg = gate([PLAIN], free=broken)
    assert verdict == "ERROR" and "cannot read free RAM" in msg
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=broken, say=out.append,
        lock=rg.MachineLock(tmp_path / "l"), platform="win32",
    )  # fmt: skip
    assert code == rg.EXIT_NO_RAM_READ and out[-1].startswith("ERROR")


def test_watchdog_fires_below_four_gb_kills_the_children_and_calls_on_fire():
    readings = iter([10.0, 4.0, 3.9, 1.0])
    killed, fired, lines = [], [], []
    wd = rg.Watchdog(
        pid=77, free=lambda: next(readings), kill=lambda pid: killed.append(pid) or [78, 79],
        on_fire=fired.append, say=lines.append,
    )  # fmt: skip
    assert not wd.check() and not wd.check()  # 10 and 4.0 are not below 4
    assert wd.check() and wd.fired and killed == [77] and fired == [wd]
    assert "3.9 GB free" in lines[0] and "killed 2" in lines[0] and "exit 15" in lines[0]
    assert wd.check() and killed == [77]  # fires once


def test_watchdog_ignores_an_unreadable_figure_and_stops_when_halted():
    def broken():
        raise OSError("x")

    wd = rg.Watchdog(pid=1, free=broken, kill=lambda pid: [], poll_s=0.01)
    assert not wd.check()
    wd.start()
    wd.halt()
    wd.join(5)
    assert not wd.is_alive() and not wd.fired


def test_kill_tree_kills_below_the_pid_and_skips_a_reused_parent_pid():
    procs = {1: 0, 10: 1, 11: 10, 12: 11, 13: 10, 20: 1, 30: 10}
    born = {1: 0, 10: 5, 11: 6, 12: 7, 13: 8, 20: 1, 30: 2}  # 30 is older than 10: not its child
    sent = []
    killed = rg.kill_tree(10, procs, started=born.get, kill=lambda pid, sig: sent.append(pid))
    assert sorted(killed) == [11, 12, 13] and 10 not in sent and 30 not in sent
    assert sent.index(12) < sent.index(11)  # leaves first


def test_kill_tree_kills_a_real_process_tree():
    import subprocess
    import time

    code = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(p.pid, flush=True); time.sleep(120)"
    )
    middle = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        grandchild = int(middle.stdout.readline())
        killed = rg.kill_tree(middle.pid)
        assert grandchild in killed and middle.pid not in killed
        for _ in range(100):
            if grandchild not in rg.process_table():
                break
            time.sleep(0.1)
        assert grandchild not in rg.process_table()
    finally:
        middle.kill()
        middle.wait(30)


def test_the_measuring_harness_figures_and_reads_rss():
    import os

    spec = importlib.util.spec_from_file_location("ramgate_measure_t", ROOT / "tools/ops/ramgate_measure.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.figure(2.0) == 4 and harness.figure(10.0) == 13 and harness.figure(0.5) == 3
    assert harness.rss_bytes(os.getpid()) > 10_000_000  # this pytest process
    assert harness.tree_pids(os.getpid())[0] == os.getpid()
    assert harness.START_FREE_GB == 20.0  # fixed: an unmeasured test may load the engine
    text = (ROOT / "tools/ops/ramgate_measure.py").read_text(encoding="utf-8")
    assert 'kind="measure"' in text and "rg.ENV_NO_WAIT" in text


def test_the_measuring_harness_gives_a_figure_only_when_every_id_ran():
    spec = importlib.util.spec_from_file_location("ramgate_measure_w", ROOT / "tools/ops/ramgate_measure.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    def line(rc, summary, figure=4):
        return harness.weight_line("tests/a.py", {"rc": rc, "summary": summary, "figure": figure}, "abc1234-20261122")

    assert line(0, "3 passed in 1.0s") == "tests/a.py 4 local-measure-abc1234-20261122"
    assert line(0, "2 passed, 1 deselected in 1.0s") == "tests/a.py 4 local-measure-abc1234-20261122"
    assert line(0, "2 passed, 1 skipped in 1.0s") == "tests/a.py 14 unmeasured-skipped-abc1234-20261122"
    assert line(0, "3 skipped in 0.1s") == "tests/a.py 14 unmeasured-skipped-abc1234-20261122"
    assert line(0, ["no summary"]) == "tests/a.py 14 unmeasured-skipped-abc1234-20261122"
    assert line(1, "1 failed, 2 passed in 1.0s") == "tests/a.py 14 unmeasured-failed-rc1-abc1234-20261122"
    assert line(5, "no tests ran in 0.1s") == "tests/a.py 14 unmeasured-failed-rc5-abc1234-20261122"
    assert line("killed", "pod-measure needed", 20) == "tests/a.py 20 pod-measure-needed"


def test_free_gb_reads_this_machine():
    assert 0.0 < rg.free_gb() < 10_000.0


def test_meminfo_available():
    text = "MemTotal:  131072000 kB\nMemFree: 1000 kB\nMemAvailable:   97656250 kB\n"
    assert rg.meminfo_available_gb(text) == pytest.approx(100.0)
    with pytest.raises(OSError, match="MemAvailable"):
        rg.meminfo_available_gb("MemTotal: 1 kB\n")


# A throwaway project with copies of the root conftest.py and ramgate.py: real child pytest runs
# through the real hook. Its "engine" is a fake module entry, so nothing heavy loads; its ROM
# fixture `docs` is defined in its own tests/conftest.py (module tests.conftest).
MINI_FILES = {
    "pytest.ini": "[pytest]\nmarkers =\n    rom: a ROM-only test\n",
    "tools/ops/day/pod/weights.txt": "tests 0.5 mini\n",
    "tools/ops/day/rom_weights.txt": "tests 0.5 mini\n",
    "tools/ops/day/pod_only.txt": "tests/test_podonly.py mini\n",
    "tests/__init__.py": "",
    "tests/conftest.py": "import pytest\n\n\n@pytest.fixture\ndef docs():\n    return 1\n",
    "tests/test_mix.py": (
        "import sys\nimport types\n\nimport pytest\n\n"
        "sys.modules.setdefault('nuzline.engine.step', types.ModuleType('nuzline.engine.step'))\n\n\n"
        "@pytest.mark.rom\ndef test_rom(docs):\n    assert docs == 1\n\n\n"
        "def test_eng():\n    pass\n"
    ),
    "tests/test_free.py": (
        "import os\nimport sys\nfrom pathlib import Path\n\n\n"
        "def test_blocker_env_and_lock_while_running():\n"
        "    assert any(type(f).__name__ == 'EngineBlocker' for f in sys.meta_path)\n"
        "    assert os.environ['NUZLINE_ENGINE_BLOCKED'] == '1'\n"
        "    assert os.environ['NUZLINE_RAMGATE_DONE'] == str(os.getpid())\n"
        "    lock = Path(__file__).parents[1] / '_scratch' / 'ramgate.lock'\n"
        "    assert lock.read_text().split()[:2] == [str(os.getpid()), 'pytest']\n"
    ),
    "tests/test_podonly.py": "def test_jit():\n    pass\n",
}

AFTER = (
    "import os, sys, pytest; rc = pytest.main(sys.argv[1:]); "
    "print('AFTER', any(type(f).__name__ == 'EngineBlocker' for f in sys.meta_path), "
    "os.environ.get('NUZLINE_ENGINE_BLOCKED'), os.environ.get('NUZLINE_RAMGATE_DONE'), "
    "open(os.path.join('_scratch', 'ramgate.lock')).read().strip() or 'EMPTY'); "
    "sys.exit(int(rc))"
)


@pytest.fixture
def mini(tmp_path):
    import shutil

    for rel, text in MINI_FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    shutil.copy(ROOT / "conftest.py", tmp_path / "conftest.py")
    shutil.copy(ROOT / "tools/ops/ramgate.py", tmp_path / "tools/ops/ramgate.py")
    return tmp_path


def _run(mini, *args, code=None):
    import subprocess

    env = _child_env(**{rg.ENV_TIMEOUT: "0"})
    head = [sys.executable, "-c", code] if code else [sys.executable, "-m", "pytest"]
    run = subprocess.run(
        [*head, "-q", "-p", "no:cacheprovider", *args],
        cwd=mini, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return run.returncode, run.stdout + run.stderr


def test_hook_refuses_a_mixed_run_with_the_engine_loaded(mini):
    rc, out = _run(mini, "tests/test_mix.py")
    assert rc == rg.EXIT_REFUSE, out
    assert "no engine tests on the PC" in out and "tests/test_mix.py::test_eng" in out
    assert not (mini / "_scratch" / "ramgate.lock").exists()


def test_hook_allows_collect_only(mini):
    rc, out = _run(mini, "--collect-only", "tests/test_mix.py")
    assert rc == 0, out
    assert "test_eng" in out and "test_rom" in out


def test_hook_sees_only_what_minus_m_selected(mini):
    rc, out = _run(mini, "-m", "rom", "tests/test_mix.py")  # test_eng deselected: all ROM-only
    assert rc == 0 and "1 passed" in out, out


def test_hook_installs_the_blocker_and_lock_and_removes_them_at_the_end(mini):
    rc, out = _run(mini, "tests/test_free.py", code=AFTER)
    assert rc == 0 and "1 passed" in out, out
    assert "AFTER False None None EMPTY" in out


def test_hook_refuses_a_pod_only_test(mini):
    rc, out = _run(mini, "tests/test_podonly.py")
    assert rc == rg.EXIT_REFUSE and "pod_only.txt" in out, out


def test_hook_waits_for_a_live_lock_and_runs_once_its_holder_is_killed(mini):
    import subprocess

    lock = mini / "_scratch" / "ramgate.lock"
    code = (
        "import importlib.util, sys, time; "
        "s = importlib.util.spec_from_file_location('r', 'tools/ops/ramgate.py'); "
        "r = importlib.util.module_from_spec(s); s.loader.exec_module(r); "
        "lk = r.MachineLock(); assert lk.acquire('pytest'); print('LOCKED', flush=True); "
        "time.sleep(120)"
    )
    other = subprocess.Popen(
        [sys.executable, "-c", code], cwd=mini, stdout=subprocess.PIPE, text=True
    )  # a gated run of another session
    try:
        assert other.stdout.readline().strip() == "LOCKED"
        rc, out = _run(mini, "tests/test_free.py")
        assert rc == rg.EXIT_TIMEOUT and "ramgate.lock" in out, out
    finally:
        other.kill()  # dies without unconfigure
        other.wait(30)
    rc, out = _run(mini, "tests/test_free.py")  # the kernel freed the lock: the run goes
    assert rc == 0 and "1 passed" in out, out
    assert lock.read_text(encoding="utf-8").strip() == ""


def test_rom_ids_prints_one_command_with_the_rom_only_ids(mini):
    import subprocess

    run = subprocess.run(
        [sys.executable, "tools/ops/ramgate.py", "--rom-ids", "tests/test_mix.py"],
        cwd=mini, env=_child_env(), capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert run.returncode == 0, run.stdout + run.stderr
    cmds = [ln for ln in run.stdout.splitlines() if ln.startswith("python -m pytest")]
    assert cmds == ["python -m pytest -p no:cacheprovider -q tests/test_mix.py::test_rom"]


def _fake_proc(root, rows):
    """A /proc tree: rows = {pid: (state, ppid)}, plus a non-pid entry and an unreadable one."""
    for pid, (state, ppid) in rows.items():
        (root / str(pid)).mkdir(parents=True)
        (root / str(pid) / "stat").write_text(
            f"{pid} (py thon) {state} {ppid} {pid} {pid} 0 -1 4194560 1 2 3 4 5 6 7 8 20 0 1 0 1234\n",
            encoding="utf-8",
        )
    (root / "self").mkdir()
    (root / "99").mkdir()  # no stat file: skipped
    (root / "98").mkdir()
    (root / "98" / "stat").write_text("98 (cut", encoding="utf-8")  # no ")": skipped


def test_a_zombie_counts_as_gone_in_the_process_table(tmp_path):
    _fake_proc(tmp_path, {1: ("S", 0), 10: ("S", 1), 11: ("Z", 10), 12: ("R", 10), 13: ("X", 10)})
    table = rg.proc_table(tmp_path)
    assert table == {1: 0, 10: 1, 12: 10}  # 11 (zombie) and 13 (dead) have ended
    born = {1: 0.0, 10: 1.0, 11: 2.0, 12: 3.0}.get
    sent = []
    killed = rg.kill_tree(10, table, started=born, kill=lambda pid, sig: sent.append(pid))
    assert killed == [12] and sent == [12]  # a zombie is never killed again
    assert rg.lineage(11, table, born) == []  # nor anyone's ancestor
    lock = rg.MachineLock(tmp_path / "l", procs=lambda: table, started=born, me=12)
    assert not lock.in_lineage(11) and lock.in_lineage(10)


def test_the_lock_wait_says_once_when_no_holder_is_named(tmp_path):
    path = tmp_path / "ramgate.lock"
    other = rg.MachineLock(path)
    assert other.acquire()
    path.write_bytes(b"")  # held, but its line is gone: no holder named
    lines = []
    clock = Clock()
    mine = rg.MachineLock(path, me=1)
    ok = rg.wait_for_ram(
        1.0, timeout_min=1, lock=mine, free=lambda: 100.0, sleep=clock.sleep, clock=clock.now,
        say=lines.append,
    )  # fmt: skip
    assert not ok and len(clock.sleeps) == 6 and len(lines) == 1
    assert "an unnamed process" in lines[0]
    other.release()
