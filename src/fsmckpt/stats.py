"""Small, dependency-free statistics for honest benchmark reporting.

We report three things for every rate:

* the point estimate (a proportion of successes),
* a bootstrap 95% confidence interval (percentile method), and
* a Wilson score interval, which -- unlike the bootstrap -- stays informative
  when the sample is all-success or all-failure (the bootstrap collapses to a
  single point there, and we say so in the report).

For comparing two paired strategies we use a sign-flip permutation test on the
per-trial difference, which makes no distributional assumption.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}


def proportion(successes: int, n: int) -> float:
    return successes / n if n else float("nan")


def bootstrap_ci(
    outcomes: list[int],
    *,
    resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Interval:
    """Percentile bootstrap CI for the mean of a 0/1 vector."""
    n = len(outcomes)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    point = sum(outcomes) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        s = 0
        for _ in range(n):
            s += outcomes[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(point, lo, hi)


def wilson_ci(successes: int, n: int, *, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def permutation_test_paired(
    a: list[int],
    b: list[int],
    *,
    permutations: int = 20000,
    seed: int = 777,
) -> dict[str, float]:
    """Two-sided sign-flip permutation test on paired 0/1 outcomes.

    Null: the per-trial difference ``a - b`` is symmetric about 0. Returns the
    observed mean difference and a p-value with explicit resolution.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [ai - bi for ai, bi in zip(a, b, strict=True)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    at_least = 0
    abs_obs = abs(observed)
    for _ in range(permutations):
        s = 0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / len(diffs)) >= abs_obs - 1e-12:
            at_least += 1
    # Add-one smoothing so the p-value is never an impossible exact 0.
    p = (at_least + 1) / (permutations + 1)
    return {
        "observed_diff": observed,
        "p_value": p,
        "permutations": float(permutations),
        "resolution": 1.0 / (permutations + 1),
    }
