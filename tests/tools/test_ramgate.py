"""tools/ops/ramgate.py: weight lookup, id selection, the decision, the pytest gate. No real RAM
read, no real waiting: free RAM, the clock and the sleep are injected."""

import importlib
import importlib.util
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
        return None, ""

    def held_by_lineage(self, kind=None):
        return False

    def in_lineage(self, pid):
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
    assert rg.engine_loaded(["nuzline.engine"]) and rg.engine_loaded(ENGINE)
    assert not rg.engine_loaded(NO_ENGINE + ["nuzline.engineering", "jax"])


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


def test_gate_override_allows_but_still_waits():
    env = {rg.ENV_ROM_GATE: "1"}
    assert gate([ENG], modules=ENGINE, env=env)[0] == "GO"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_ROM_GATE: "0"})[0] == "REFUSE"
    verdict, _ = gate([ENG], free=7.0, modules=ENGINE, env={**env, rg.ENV_TIMEOUT: "1"})
    assert verdict == "TIMEOUT"  # weight 5 needs > 11 free


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
    unknown = item("tests/zzz.py::t", "tmp_path")  # no key: 20 with the engine loaded
    env = {rg.ENV_TIMEOUT: "0", rg.ENV_ROM_GATE: "1"}
    assert gate([PLAIN, unknown], free=25.9, modules=ENGINE, env=env)[0] == "TIMEOUT"
    assert gate([PLAIN, unknown], free=26.1, modules=ENGINE, env=env)[0] == "GO"
    env = {rg.ENV_TIMEOUT: "0"}
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


def test_rom_only_items_weigh_fourteen_until_measured():
    light = item("tests/b/test_y.py::t[x]", "kaizo_inputs")  # pod table says 4
    assert rg.ROM_UNMEASURED_GB == 14.0
    assert rg.item_weight(light, TABLE, {}) == 14.0
    assert rg.item_weight(item("tests/zzz.py::t", "docs"), TABLE, {}) == 14.0  # no pod key
    assert rg.item_weight(item("tests/heavy.py::t", "docs"), TABLE, {}) == 30.0
    assert rg.item_weight(item("tests/b/test_y.py::t", "tmp_path"), TABLE, {}) == 4.0
    measured = {"tests/b/test_y.py": 7.0, "tests/b/test_y.py::t": 9.0}
    assert rg.item_weight(light, TABLE, measured) == 9.0  # longest prefix wins
    assert rg.item_weight(item("tests/b/test_y.py::u", "docs"), TABLE, measured) == 7.0
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([light], free=20.1, rom_table={}, env=env)[0] == "GO"  # needs > 20
    assert gate([light], free=19.9, rom_table={}, env=env)[0] == "TIMEOUT"
    assert gate([light], free=13.1, rom_table={"tests/b/test_y.py": 7.0}, env=env)[0] == "GO"


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
        return rg.main(argv, free=lambda: free, sleep=lambda s: None, say=out.append, lock=lock)

    assert run(["tests/a/test_x.py"]) == 0 and out[-1] == "GO 5"
    assert run(["tests/heavy.py"]) == 3 and out[-1] == "POD 30"
    assert run(["tests/heavy.py", "--rom-gate"]) == 0 and out[-1] == "GO 30"
    assert run(["tools.kaizo.x", "--weight", "9"]) == 3 and out[-1] == "POD 9"
    # a build never goes to the pod: 14 without a tools/build_tables.py line, only the RAM wait applies
    assert run(["--build", "--", *OK_CHILD]) == 0 and out[-1] == "GO 14"
    assert run(["--build", "--weight", "12", "--", *OK_CHILD]) == 0 and out[-1] == "GO 12"
    monkeypatch.setattr(rg, "load_weights", lambda: {**TABLE, "tools/build_tables.py": 11.0})
    assert run(["--build", "--", *OK_CHILD]) == 0 and out[-1] == "GO 11"  # the measured line
    assert run(["--build", "--weight", "12", "--", *OK_CHILD]) == 0 and out[-1] == "GO 12"
    assert not (tmp_path / "ramgate.lock").exists()  # released after each build


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
        say=out.append, lock=rg.MachineLock(tmp_path / "l"),
    )  # fmt: skip
    assert code == 4 and clock.sleeps == [60.0, 60.0] and out[-1] == "TIMEOUT 14"


def test_cli_build_waits_for_ram(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    seq = iter([10.0, 19.0, 20.5])
    sleeps = []
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=lambda: next(seq), sleep=sleeps.append,
        say=out.append, lock=rg.MachineLock(tmp_path / "l"),
    )  # fmt: skip
    assert code == 0 and sleeps == [60.0, 60.0] and out[-1] == "GO 14"
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
        lock=rg.MachineLock(lock_file),
    )  # fmt: skip
    assert code == 7  # the child's exit code
    import os

    assert seen.read_text().split() == [str(os.getpid()), "build"]  # it ran under my lock
    assert not lock_file.exists()


def test_cli_build_waits_for_the_lock_within_the_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv(rg.ENV_TIMEOUT, "1")
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    lock_file = tmp_path / "ramgate.lock"
    lock_file.write_text("4242 pytest\n", encoding="utf-8")
    procs = {1: 0, 4242: 1, 100: 1}  # 4242 is alive and not an ancestor of 100
    clock = Clock()
    out = []
    code = rg.main(
        ["--build", "--", *OK_CHILD], free=lambda: 100.0, sleep=clock.sleep, clock=clock.now,
        say=out.append, lock=rg.MachineLock(lock_file, procs=lambda: procs, me=100),
    )  # fmt: skip
    assert code == 4 and out[-1] == "TIMEOUT 14"
    assert clock.sleeps == [10.0] * 6  # 10 s polls, up to the 1 minute timeout
    assert sum("holds" in m and "4242" in m for m in out) == 1  # told once per holder
    assert lock_file.read_text(encoding="utf-8") == "4242 pytest\n"  # never touched


@pytest.mark.parametrize("name", ["unit.sh", "setup.sh", "quick.sh"])
def test_pod_entry_points_export_the_pod_marker(name):
    text = (ROOT / "tools/ops/day/pod" / name).read_text(encoding="utf-8")
    assert rg.ENV_ON_POD + "=1" in text


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


def test_blocker_wanted_only_for_engine_free_runs_without_override():
    assert rg.blocker_wanted([PLAIN, ROM], {})
    assert rg.blocker_wanted([PLAIN], {rg.ENV_ROM_GATE: "0"})
    assert not rg.blocker_wanted([ROM, ROM], {})
    assert not rg.blocker_wanted([PLAIN], {rg.ENV_ROM_GATE: "1"})


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
    Collect-only: if the check were missing this only imports the engine, it runs no test."""
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
    for name in (
        "tests/test_cli.py::test_kaizo_save_export_checks_kaizo_tables",
        "tests/test_cli.py::test_kaizo_roark_coverage_on_the_saves_is_exact",
        "tests/kaizo/test_roark_ai_audit.py::test_roark_ai_against_the_pool",
        "tests/kaizo/test_roark_widths.py::test_roark_pairs_fit_the_width_cap",
    ):
        assert name in rg.ROM_ONLY_IDS
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


def test_refusal_says_ids_never_a_file_path_and_rom_gate_is_a_last_resort():
    _, msg = gate([ENG], modules=ENGINE)
    assert "never a file path" in msg and "--rom-ids" in msg and "last resort" in msg


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
    path = tmp_path / "ramgate.lock"
    harness = rg.MachineLock(path, procs=lambda: procs, me=50)
    assert harness.acquire("measure")
    clock = Clock()
    env = {rg.ENV_NO_WAIT: "1"}
    child = rg.MachineLock(path, procs=lambda: procs, me=51)
    assert gate([PLAIN], free=0.0, env=env, clock=clock, lock=child)[0] == "GO"
    assert clock.sleeps == []
    assert gate([ENG], free=0.0, modules=ENGINE, env=env, lock=child)[0] == "REFUSE"
    stranger = rg.MachineLock(path, procs=lambda: procs, me=60)  # NO_WAIT alone: ignored
    lines = []
    env_t = {**env, rg.ENV_TIMEOUT: "0"}
    assert gate([PLAIN], free=0.0, env=env_t, lock=stranger, lines=lines)[0] == "TIMEOUT"
    assert any("ignored" in m for m in lines)
    harness.release()
    assert harness.acquire("pytest")  # a plain run's lock is not the harness's
    assert gate([PLAIN], free=0.0, env=env_t, lock=rg.MachineLock(path, procs=lambda: procs, me=51))[0] == "TIMEOUT"
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


def _lock(tmp_path, me, procs=PROCS, **kw):
    return rg.MachineLock(tmp_path / "_scratch" / "ramgate.lock", procs=lambda: procs, me=me, **kw)


def test_lock_is_exclusive_and_released(tmp_path):
    a, b = _lock(tmp_path, 10), _lock(tmp_path, 20)
    assert a.acquire() and a.owned
    assert a.path.read_text(encoding="utf-8") == "10 pytest\n"
    assert not b.acquire() and not b.held  # 10 is alive
    a.release()
    assert not a.path.exists()
    assert b.acquire("build") and b.holder() == (20, "build")
    b.release()


def test_lock_is_stale_when_its_pid_is_gone(tmp_path):
    a = _lock(tmp_path, 10)
    a.path.parent.mkdir(parents=True)
    a.path.write_text("999 pytest\n", encoding="utf-8")  # 999 is not running
    assert a.acquire() and a.holder() == (10, "pytest")


def test_lock_held_by_an_ancestor_is_inherited_and_left_alone(tmp_path):
    parent, child = _lock(tmp_path, 10), _lock(tmp_path, 12)
    assert parent.acquire()
    assert child.acquire() and child.inherited and not child.owned
    assert child.held_by_lineage() and child.held_by_lineage("pytest")
    assert not child.held_by_lineage("measure")
    child.release()
    assert parent.path.exists()  # the child never removes its ancestor's lock
    assert not _lock(tmp_path, 30).held_by_lineage()
    parent.release()


def test_an_empty_lock_file_waits_then_goes_stale(tmp_path):
    now = [0.0]
    a = _lock(tmp_path, 10, now=lambda: now[0])
    a.path.parent.mkdir(parents=True)
    a.path.write_text("", encoding="utf-8")
    mtime = a.path.stat().st_mtime
    now[0] = mtime + 1
    assert not a.acquire()  # its writer may be between create and write
    now[0] = mtime + rg.LOCK_EMPTY_STALE_S + 1
    assert a.acquire() and a.owned


def test_a_lock_of_another_pid_is_never_released_by_me(tmp_path):
    a = _lock(tmp_path, 10)
    assert a.acquire()
    a.path.write_text("20 pytest\n", encoding="utf-8")  # someone replaced it
    a.release()
    assert a.path.read_text(encoding="utf-8") == "20 pytest\n"


def test_gate_waits_for_the_lock_then_goes_and_holds_it(tmp_path):
    lock = _lock(tmp_path, 12, procs=dict(PROCS))
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("20 pytest\n", encoding="utf-8")
    procs = lock._procs()
    clock = Clock()

    def sleep(s):
        clock.sleep(s)
        procs.pop(20, None)  # the other run ends during the first wait

    lines = []
    verdict, _ = rg.pytest_gate(
        [PLAIN], modules=NO_ENGINE, env={}, table=TABLE, rom_table={}, lock=lock,
        free=lambda: 100.0, sleep=sleep, clock=clock.now, say=lines.append,
    )  # fmt: skip
    assert verdict == "GO" and lock.owned and lock.holder() == (12, "pytest")
    assert clock.sleeps == [10.0] and "holds" in lines[0]
    lock.release()


def test_gate_lock_wait_times_out_with_a_message_naming_the_lock(tmp_path):
    lock = _lock(tmp_path, 12)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("20 pytest\n", encoding="utf-8")
    verdict, msg = gate([PLAIN], env={rg.ENV_TIMEOUT: "0"}, lock=lock)
    assert verdict == "TIMEOUT" and "ramgate.lock" in msg and not lock.held


def test_process_table_and_lineage_of_this_process():
    import os

    procs = rg.process_table()
    me = rg.lineage(os.getpid(), procs)
    assert me[0] == os.getpid() and len(me) >= 2  # at least the process that started pytest
    assert rg.lineage(1, {1: 2, 2: 1}) == [1, 2]  # a pid cycle ends


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
