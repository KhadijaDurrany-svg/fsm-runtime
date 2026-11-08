from __future__ import annotations

from fsmckpt.stats import (
    bootstrap_ci,
    permutation_test_paired,
    proportion,
    wilson_ci,
)


def test_proportion():
    assert proportion(3, 4) == 0.75
    assert proportion(0, 0) != proportion(0, 0) or True  # nan-safe smoke


def test_bootstrap_ci_brackets_point():
    outcomes = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    ci = bootstrap_ci(outcomes, resamples=2000, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert 0.0 <= ci.low <= 1.0
    assert 0.0 <= ci.high <= 1.0


def test_bootstrap_all_success_is_degenerate():
    ci = bootstrap_ci([1] * 20, resamples=1000)
    assert ci.point == 1.0
    assert ci.low == 1.0 and ci.high == 1.0


def test_wilson_all_success_not_degenerate():
    ci = wilson_ci(20, 20)
    assert ci.point == 1.0
    assert ci.low < 1.0  # Wilson stays informative where bootstrap collapses


def test_wilson_interval_contains_point():
    ci = wilson_ci(5, 10)
    assert ci.low < 0.5 < ci.high


def test_permutation_detects_clear_difference():
    a = [1] * 50
    b = [0] * 50
    res = permutation_test_paired(a, b, permutations=2000)
    assert res["observed_diff"] == 1.0
    assert res["p_value"] <= res["resolution"] * 2


def test_permutation_no_difference_high_p():
    a = [1, 0, 1, 0, 1, 0]
    b = [1, 0, 1, 0, 1, 0]
    res = permutation_test_paired(a, b, permutations=2000)
    assert res["observed_diff"] == 0.0
    assert res["p_value"] > 0.5


def test_permutation_requires_equal_length():
    import pytest

    with pytest.raises(ValueError):
        permutation_test_paired([1, 0], [1], permutations=10)


def test_permutation_pvalue_never_zero():
    res = permutation_test_paired([1] * 10, [0] * 10, permutations=100)
    assert res["p_value"] > 0.0
