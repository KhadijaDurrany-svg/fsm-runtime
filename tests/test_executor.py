from __future__ import annotations

from fsmckpt.executor import (
    STRATEGIES,
    determinism_trial,
    reference_run,
    replay_from_journal,
    resume_run,
)
from fsmckpt.generate import build_graph, generate_spec, initial_state


def _graph(seed: int):
    return build_graph(generate_spec(seed))


def test_reference_run_terminates_and_hashes_align():
    g = _graph(1)
    trace = reference_run(g, initial_state(), 1)
    assert trace.length >= 1
    # hash_sequence is init + one per step.
    assert len(trace.hash_sequence) == trace.length + 1


def test_reference_run_is_repeatable():
    g = _graph(2)
    a = reference_run(g, initial_state(), 42)
    b = reference_run(g, initial_state(), 42)
    assert a.hash_sequence == b.hash_sequence
    assert a.ledger_signature == b.ledger_signature


def test_logged_replay_matches_reference():
    g = _graph(3)
    ref = reference_run(g, initial_state(), 5)
    replay = replay_from_journal(g, initial_state(), 5, ref.journal.durable_snapshot())
    assert replay.hash_sequence == ref.hash_sequence


def test_determinism_trial_logged_always_ok():
    for seed in range(20):
        out = determinism_trial(_graph(seed), initial_state(), seed + 100)
        assert out.logged_ok, f"logged replay diverged for seed {seed}"


def test_wal_fsync_resume_is_correct_for_every_crash_step():
    g = _graph(7)
    ref = reference_run(g, initial_state(), 11)
    for k in range(ref.length):
        out = resume_run(g, initial_state(), 11, ref, k, "wal-fsync")
        assert out.correct, f"wal-fsync failed at crash_step {k}"
        assert out.extra_node_executions == 0


def test_rerun_duplicates_side_effects_when_prefix_has_writes():
    # Find a graph + crash point where the prefix contains a write, then show
    # the naive re-run double-applies it (ledger differs from reference).
    for seed in range(40):
        g = _graph(seed)
        ref = reference_run(g, initial_state(), seed + 7)
        write_steps = [s.step for s in ref.steps if s.did_write]
        if not write_steps:
            continue
        k = max(write_steps)  # crash after a write happened
        out = resume_run(g, initial_state(), seed + 7, ref, k, "rerun")
        if out.ledger_signature != ref.ledger_signature:
            assert not out.correct
            return
    raise AssertionError("expected at least one graph where rerun duplicates writes")


def test_all_strategies_run_without_error():
    g = _graph(13)
    ref = reference_run(g, initial_state(), 99)
    k = ref.length // 2
    for strat in STRATEGIES:
        out = resume_run(g, initial_state(), 99, ref, k, strat)
        assert isinstance(out.correct, bool)


def test_rerun_redoes_more_work_than_wal():
    g = _graph(21)
    ref = reference_run(g, initial_state(), 3)
    k = ref.length - 1
    wal = resume_run(g, initial_state(), 3, ref, k, "wal-fsync")
    rerun = resume_run(g, initial_state(), 3, ref, k, "rerun")
    assert rerun.extra_node_executions >= wal.extra_node_executions
