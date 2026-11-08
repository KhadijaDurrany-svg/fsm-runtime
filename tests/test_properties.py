"""Property-based tests (Hypothesis) over seeded synthetic graphs."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fsmckpt.executor import reference_run, replay_from_journal, resume_run
from fsmckpt.generate import build_graph, generate_spec, initial_state
from fsmckpt.state import WorkflowState, hash_obj

_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_replay_determinism_property(seed: int):
    """Journal-based replay reproduces the reference state-hash sequence exactly."""
    g = build_graph(generate_spec(seed))
    ref = reference_run(g, initial_state(), seed)
    replay = replay_from_journal(g, initial_state(), seed, ref.journal.durable_snapshot())
    assert replay.hash_sequence == ref.hash_sequence


@given(seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_reference_run_is_reproducible_property(seed: int):
    g = build_graph(generate_spec(seed))
    a = reference_run(g, initial_state(), seed)
    b = reference_run(g, initial_state(), seed)
    assert a.hash_sequence == b.hash_sequence
    assert a.ledger_signature == b.ledger_signature


@given(seed=st.integers(min_value=0, max_value=5_000), k_frac=st.floats(0.0, 1.0))
@_SETTINGS
def test_wal_fsync_resume_equals_uninterrupted_property(seed: int, k_frac: float):
    """Write-ahead checkpoint resume matches the uninterrupted run for any crash point."""
    g = build_graph(generate_spec(seed))
    ref = reference_run(g, initial_state(), seed)
    k = min(ref.length - 1, int(k_frac * ref.length))
    out = resume_run(g, initial_state(), seed, ref, k, "wal-fsync")
    assert out.final_hash == ref.final_hash
    assert out.ledger_signature == ref.ledger_signature
    assert out.correct


@given(
    data=st.dictionaries(
        st.text(min_size=1, max_size=4),
        st.integers(),
        max_size=6,
    )
)
@_SETTINGS
def test_dict_hash_order_invariant_property(data: dict[str, int]):
    import random

    items = list(data.items())
    random.Random(len(items)).shuffle(items)
    shuffled = dict(items)
    assert hash_obj(data) == hash_obj(shuffled)


@given(
    acc=st.integers(min_value=-1000, max_value=1000),
    trail=st.lists(st.text(max_size=3), max_size=5),
)
@_SETTINGS
def test_state_hash_matches_equal_state_property(acc: int, trail: list[str]):
    a = WorkflowState(accumulator=acc, trail=list(trail))
    b = WorkflowState(accumulator=acc, trail=list(trail))
    assert a.state_hash() == b.state_hash()
