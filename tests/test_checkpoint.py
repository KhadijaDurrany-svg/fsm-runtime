from __future__ import annotations

from fsmckpt.checkpoint import Checkpoint, CheckpointStore
from fsmckpt.state import WorkflowState


def _cp(step: int) -> Checkpoint:
    state = WorkflowState(step_count=step + 1, accumulator=step)
    return Checkpoint(
        step=step,
        next_node=f"n{step + 1}",
        state=state,
        state_hash=state.state_hash(),
        completed_writes=frozenset(range(step + 1)),
    ).seal()


def test_sealed_checkpoint_is_intact():
    cp = _cp(0)
    assert cp.is_intact()


def test_torn_write_is_not_intact():
    cp = _cp(0)
    cp.committed = False
    assert not cp.is_intact()


def test_corrupt_checksum_is_not_intact():
    cp = _cp(0)
    cp.checksum = "ffff"
    assert not cp.is_intact()


def test_latest_valid_returns_highest_step():
    store = CheckpointStore()
    for i in range(4):
        store.save(_cp(i))
    assert store.latest_valid().step == 3


def test_latest_valid_falls_back_over_torn_write():
    store = CheckpointStore()
    for i in range(4):
        store.save(_cp(i))
    store.tear_last_write()
    assert store.latest_valid().step == 2


def test_latest_valid_falls_back_over_corrupt_write():
    store = CheckpointStore()
    for i in range(3):
        store.save(_cp(i))
    store.corrupt_last_write()
    assert store.latest_valid().step == 1


def test_latest_valid_none_when_empty():
    assert CheckpointStore().latest_valid() is None


def test_checksum_depends_on_state():
    a = _cp(0)
    b = _cp(0)
    assert a.checksum == b.checksum
    b.state.accumulator += 1
    assert b.checksum != b.compute_checksum()
