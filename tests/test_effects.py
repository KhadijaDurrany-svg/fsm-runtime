from __future__ import annotations

from fsmckpt.effects import (
    EffectJournal,
    ExecutionContext,
    ExternalSource,
    SideEffectLedger,
)


def test_first_read_differs_from_second_read_same_key():
    src = ExternalSource(run_seed=1)
    v0 = src.read(0, "n0", "a")
    v1 = src.read(0, "n0", "a")
    assert v0 != v1  # re-execution of the same key is a fresh physical read


def test_independent_sources_same_seed_first_read_matches():
    a = ExternalSource(run_seed=7)
    b = ExternalSource(run_seed=7)
    assert a.read(1, "n1", "a") == b.read(1, "n1", "a")


def test_journal_replays_recorded_value():
    src = ExternalSource(run_seed=3)
    j = EffectJournal()
    ledger = SideEffectLedger()
    ctx = ExecutionContext(0, "n0", src, j, ledger)
    first = ctx.read_int("a")
    j.fsync()

    # A replay context backed by the durable journal returns the same value
    # without ever touching the (advanced) source.
    src2 = ExternalSource(run_seed=3)
    src2.read(0, "n0", "a0")  # advance so a fresh draw would differ
    j2 = EffectJournal.from_durable(j.durable_snapshot())
    ctx2 = ExecutionContext(0, "n0", src2, j2, ledger)
    assert ctx2.read_int("a") == first


def test_journal_crash_loses_unflushed_tail():
    j = EffectJournal()
    j.record(0, "n0", "a", 111)
    j.fsync()
    j.record(1, "n1", "a", 222)  # buffered, not fsync'd
    j.crash()
    assert j.lookup(0, "n0", "a") == 111
    assert j.lookup(1, "n1", "a") is None


def test_ledger_records_in_order():
    ledger = SideEffectLedger()
    ledger.emit(0, "n0", 10)
    ledger.emit(1, "n1", 20)
    assert ledger.signature() == ((0, "n0", 10), (1, "n1", 20))


def test_ledger_clone_is_independent():
    ledger = SideEffectLedger()
    ledger.emit(0, "n0", 10)
    clone = ledger.clone()
    clone.emit(1, "n1", 20)
    assert len(ledger.entries) == 1
    assert len(clone.entries) == 2


def test_context_read_count_advances_tag():
    src = ExternalSource(run_seed=9)
    ctx = ExecutionContext(0, "n0", src, EffectJournal(), SideEffectLedger())
    a = ctx.read_int("r")
    b = ctx.read_int("r")
    # Two reads in one node use different tags, so they are distinct keys.
    assert a != b
