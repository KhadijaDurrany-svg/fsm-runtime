from __future__ import annotations

import json

from fsmckpt.bench import run_benchmark
from fsmckpt.cli import main
from fsmckpt.manifest import build_manifest, git_sha


def test_benchmark_runs_and_has_structure():
    r = run_benchmark(n_graphs=30, bootstrap_resamples=500, permutations=1000)
    assert r["dataset"]["n_graphs"] == 30
    assert "checksum" in r["dataset"]
    assert set(r["replay_determinism"]) >= {"logged_replay", "naive_replay"}
    assert set(r["resume_correctness"]["per_strategy"]) == {
        "wal-fsync",
        "wal-group4",
        "wal-group8",
        "rerun",
    }


def test_benchmark_logged_replay_is_perfect_and_beats_naive():
    r = run_benchmark(n_graphs=40, bootstrap_resamples=500, permutations=1000)
    rd = r["replay_determinism"]
    assert rd["logged_replay"]["rate"] == 1.0
    assert rd["naive_replay"]["rate"] < rd["logged_replay"]["rate"]


def test_benchmark_wal_beats_rerun():
    r = run_benchmark(n_graphs=40, bootstrap_resamples=500, permutations=1000)
    rc = r["resume_correctness"]["per_strategy"]
    assert rc["wal-fsync"]["rate"] >= rc["wal-group4"]["rate"] >= rc["rerun"]["rate"]
    assert rc["wal-fsync"]["rate"] > rc["rerun"]["rate"]


def test_benchmark_is_reproducible():
    a = run_benchmark(n_graphs=25, bootstrap_resamples=300, permutations=500)
    b = run_benchmark(n_graphs=25, bootstrap_resamples=300, permutations=500)
    assert a["dataset"]["checksum"] == b["dataset"]["checksum"]
    assert (
        a["resume_correctness"]["per_strategy"]["rerun"]["rate"]
        == b["resume_correctness"]["per_strategy"]["rerun"]["rate"]
    )


def test_manifest_fields_present():
    m = build_manifest("abc123", {"n_graphs": 1})
    assert m["dataset_checksum"] == "abc123"
    assert m["python"]
    assert m["git_sha"]  # "n/a" or a real sha, never empty
    assert isinstance(git_sha(), str)


def test_cli_bench_no_write(capsys):
    rc = main(["bench", "--graphs", "20", "--resamples", "300", "--permutations", "500", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fsmckpt benchmark" in out
    assert "not persisted" in out


def test_cli_bench_writes_files(tmp_path, capsys):
    out_dir = tmp_path / "bench_out"
    rc = main(["bench", "--graphs", "20", "--resamples", "300", "--permutations", "500", "--out", str(out_dir)])
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["dataset"]["checksum"] == manifest["dataset_checksum"]


def test_cli_demo(capsys):
    rc = main(["demo", "--seed", "3"])
    assert rc == 0
    assert "final state hash" in capsys.readouterr().out
