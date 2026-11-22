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


def item(nodeid, *fixtures):
    return SimpleNamespace(nodeid=nodeid, fixturenames=["request", *fixtures])


ENGINE = ["os", "jax", "nuzline.engine.step"]
NO_ENGINE = ["os", "jax", "nuzline.data.tables"]
ROM = item("tests/tools/rom/test_nds.py::test_a", "vanilla_bytes")
PLAIN = item("tests/tools/test_x.py::test_b", "tmp_path")
ENG = item("tests/search/test_leaves.py::test_c")


def gate(items, free=20.0, modules=NO_ENGINE, env=None, args=("tests/a/test_x.py",), **kw):
    clock = kw.pop("clock", None) or Clock()
    lines = kw.pop("lines", [])
    return rg.pytest_gate(
        items,
        args,
        ROOT,
        modules=modules,
        env={} if env is None else env,
        table=TABLE,
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


def test_selected_ids_drop_options_and_values():
    args = [
        "-m",
        "not slow and not fullset",
        "-q",
        "-k",
        "foo",
        "-p",
        "no:cacheprovider",
        "--durations",
        "40",
        "-o",
        "addopts=",
        "--maxfail=2",
        "tests\\a\\test_x.py::test_one",
        "./tests/b/",
    ]
    assert rg.selected_ids(args, ROOT) == ["tests/a/test_x.py::test_one", "tests/b"]


def test_selected_ids_absolute_and_invocation_dir():
    assert rg.selected_ids([str(ROOT / "tests" / "a.py")], ROOT) == ["tests/a.py"]
    assert rg.selected_ids(["a.py"], ROOT, ROOT / "tests") == ["tests/a.py"]
    assert rg.selected_ids(["-x", "-q"], ROOT) == []


def test_run_weight_heaviest_and_whole_tree():
    assert rg.run_weight(["tests/a/test_x.py", "tests/b/test_y.py"], TABLE) == 5.0
    assert rg.run_weight(["tests/a/test_x.py", "tests/new.py"], TABLE) == 20.0
    assert rg.run_weight([], TABLE) == 30.0


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


def test_gate_skipped_on_pod_and_in_nested_runs():
    assert gate([ENG], modules=ENGINE, env={rg.ENV_ON_POD: "1"})[0] == "SKIP"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_DONE: "1"})[0] == "SKIP"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_NESTED: "x::y (call)"})[0] == "SKIP"


def test_gate_weight_unknown_id_is_twenty_and_whole_tree_is_heaviest():
    assert (
        gate([PLAIN], free=25.9, args=["tests/zzz.py"], env={rg.ENV_TIMEOUT: "0"})[0] == "TIMEOUT"
    )
    assert gate([PLAIN], free=26.1, args=["tests/zzz.py"])[0] == "GO"
    assert gate([PLAIN], free=35.0, args=[], env={rg.ENV_TIMEOUT: "0"})[0] == "TIMEOUT"  # 30 + 6


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


def test_cli_pod_rom_gate_and_build(monkeypatch):
    monkeypatch.delenv(rg.ENV_ROM_GATE, raising=False)
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    out = []

    def run(argv, free=100.0):
        return rg.main(argv, free=lambda: free, sleep=lambda s: None, say=out.append)

    assert run(["tests/a/test_x.py"]) == 0 and out[-1] == "GO 5"
    assert run(["tests/heavy.py"]) == 3 and out[-1] == "POD 30"
    assert run(["tests/heavy.py", "--rom-gate"]) == 0 and out[-1] == "GO 30"
    assert run(["tools.kaizo.x", "--weight", "9"]) == 3 and out[-1] == "POD 9"
    # a build never goes to the pod: weight 20 by default, only the RAM wait applies
    assert run(["--build"]) == 0 and out[-1] == "GO 20"
    assert run(["--build", "--weight", "12"]) == 0 and out[-1] == "GO 12"


def test_cli_build_waits_for_ram():
    seq = iter([10.0, 25.0, 26.5])
    sleeps = []
    out = []
    code = rg.main(["--build"], free=lambda: next(seq), sleep=sleeps.append, say=out.append)
    assert code == 0 and sleeps == [60.0, 60.0] and out[-1] == "GO 20"
    assert not any(m.startswith("POD") for m in out)


@pytest.mark.parametrize("name", ["unit.sh", "setup.sh"])
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


def test_blocker_in_a_fake_run_stops_the_import_and_is_removed():
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
