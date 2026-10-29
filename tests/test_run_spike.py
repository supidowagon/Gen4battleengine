import json

from benchmarks.run_spike import measure, run


def test_measure_separates_compile_from_steady_state():
    r = measure(batch=64, n_steps=5)
    assert r["compile_s"] > 0 and r["steps_per_s"] > 0
    assert r["batch"] == 64 and r["n_steps"] == 5


def test_run_writes_unique_file_each_call(tmp_path):
    p1 = run([16], 2, tmp_path)
    p2 = run([16], 2, tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()
    data = json.loads(p1.read_text())
    assert data["results"][0]["batch"] == 16
