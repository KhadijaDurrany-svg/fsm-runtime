from __future__ import annotations

from fsmckpt.state import WorkflowState, canonical_bytes, hash_obj


def test_state_hash_is_stable_across_instances():
    a = WorkflowState(accumulator=5, trail=["a", "b"])
    b = WorkflowState(accumulator=5, trail=["a", "b"])
    assert a.state_hash() == b.state_hash()


def test_dict_insertion_order_does_not_change_hash():
    a = WorkflowState(tallies={"x": 1, "y": 2})
    b = WorkflowState(tallies={"y": 2, "x": 1})
    assert a.state_hash() == b.state_hash()


def test_list_order_does_change_hash():
    a = WorkflowState(trail=["a", "b"])
    b = WorkflowState(trail=["b", "a"])
    assert a.state_hash() != b.state_hash()


def test_set_membership_is_order_insensitive():
    assert hash_obj({1, 2, 3}) == hash_obj({3, 2, 1})


def test_negative_zero_float_normalized():
    assert canonical_bytes(-0.0) == canonical_bytes(0.0)


def test_copy_state_is_independent():
    a = WorkflowState(trail=["a"])
    b = a.copy_state()
    b.trail.append("b")
    assert a.trail == ["a"]
    assert b.trail == ["a", "b"]


def test_extra_fields_forbidden():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        WorkflowState(bogus=1)  # type: ignore[call-arg]


def test_canonical_bytes_deterministic():
    obj = {"b": [3, 2, 1], "a": {"z": 1, "y": 2}}
    assert canonical_bytes(obj) == canonical_bytes(dict(obj))
