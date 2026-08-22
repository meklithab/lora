"""Paired deltas, minimum detectable effect, and effect sizes. BUILD_SPEC.md 4.8.

No parametric significance tests: at n=3 the Wilcoxon signed-rank floor is p=0.25, at n=5 it's
p=0.0625 -- the test cannot return significance regardless of effect size, so this module reports
what can actually be claimed instead: paired deltas with every individual seed pair, the noise
floor, the effect size we'd need to be able to detect, and an *exact* sign-flip permutation p-value
that is reported alongside its own attainable floor so the reader can see when a "p" is pinned at
the design's limit rather than measuring anything.

Two things this module is careful about, both of which previously produced misleading verdicts:

  - `minimum_detectable_effect` is a two-independent-groups formula. It must be fed a sigma pooled
    across *both* arms of the comparison (`pooled_two_sample_sd`), not the baseline arm's sigma
    alone. Using the baseline alone understates the threshold whenever the treatment arm is noisier,
    which is exactly the regime a capacity-reallocation experiment produces.
  - the design is only nominally paired. LoRA A-initialisation consumes RNG per module in an amount
    that depends on that module's rank, so two conditions at the same seed do not share an
    initialisation; the seed indexes a replicate, not a matched pair. The independent-groups MDE is
    therefore the right form, and `paired_minimum_detectable_effect` is provided only for designs
    that genuinely do match nuisance factors across conditions.
"""
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from scipy import stats as _stats


@dataclass
class PairedDeltaResult:
    mean_delta: float
    n: int
    pairs: List[dict] = field(default_factory=list)  # [{"seed": s, "a": .., "b": .., "delta": ..}, ...]


def paired_delta(a: Dict[int, float], b: Dict[int, float]) -> PairedDeltaResult:
    """a, b: {seed: metric_value}, e.g. by_seed(df, 'gradnorm_prop', ...) vs by_seed(df, 'uniform', ...).
    Only seeds present in both are paired -- delta = a - b for each.
    """
    common_seeds = sorted(set(a) & set(b))
    pairs = [{"seed": s, "a": a[s], "b": b[s], "delta": a[s] - b[s]} for s in common_seeds]
    mean_delta = sum(p["delta"] for p in pairs) / len(pairs) if pairs else float("nan")
    return PairedDeltaResult(mean_delta=mean_delta, n=len(pairs), pairs=pairs)


def pooled_sd(values: Sequence[float]) -> float:
    values = list(values)
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def pooled_two_sample_sd(a: Sequence[float], b: Sequence[float]) -> float:
    """sqrt of the df-weighted pooled variance of two samples -- the sigma an independent-groups MDE
    is defined against. Falls back to whichever arm has >= 2 observations if the other does not.
    """
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 and n2 < 2:
        return float("nan")
    if n1 < 2:
        return pooled_sd(b)
    if n2 < 2:
        return pooled_sd(a)
    var1, var2 = pooled_sd(a) ** 2, pooled_sd(b) ** 2
    return (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5


def minimum_detectable_effect(sigma: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """(t_{1-alpha/2,df} + t_{power,df}) * sigma * sqrt(2/n), for two independent groups of size n.

    `sigma` must be pooled across both arms (see pooled_two_sample_sd); passing the baseline arm's
    sigma alone silently shrinks the threshold whenever the treatment arm is noisier.
    """
    if n < 2 or not math.isfinite(sigma):
        return float("nan")
    df = 2 * n - 2
    t_alpha = _stats.t.ppf(1 - alpha / 2, df)
    t_power = _stats.t.ppf(power, df)
    return (t_alpha + t_power) * sigma * (2 / n) ** 0.5


def paired_minimum_detectable_effect(sigma_diff: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """(t_{1-alpha/2,n-1} + t_{power,n-1}) * sigma_diff / sqrt(n), for a genuinely paired design.

    Only valid when the pairing controls a real shared nuisance factor. In this pipeline it does
    not (see the module docstring), so this is exported for completeness and is not what analyze.py
    gates on.
    """
    if n < 2 or not math.isfinite(sigma_diff):
        return float("nan")
    df = n - 1
    t_alpha = _stats.t.ppf(1 - alpha / 2, df)
    t_power = _stats.t.ppf(power, df)
    return (t_alpha + t_power) * sigma_diff / (n**0.5)


def sign_flip_test(deltas: Sequence[float]) -> dict:
    """Exact two-sided sign-flip permutation test on paired differences.

    Enumerates all 2^n sign assignments, so it is exact rather than asymptotic and is valid at n=3.
    Returns the p-value together with `p_floor` = 2^(1-n), the smallest value the test can return at
    this n. Reporting them together is the point: at n=5 the floor is 0.0625, so a p of 0.0625 means
    "every seed agreed and the design cannot say more", not "nearly significant".
    """
    d = [float(x) for x in deltas]
    n = len(d)
    if n == 0:
        return {"p_value": float("nan"), "p_floor": float("nan"), "n": 0}
    observed = abs(sum(d) / n)
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        m = abs(sum(s * x for s, x in zip(signs, d)) / n)
        if m >= observed - 1e-15:
            count += 1
    return {"p_value": count / (2**n), "p_floor": 2.0 ** (1 - n), "n": n}


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    mean1, mean2 = sum(a) / n1, sum(b) / n2
    var1 = sum((x - mean1) ** 2 for x in a) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in b) / (n2 - 1)
    pooled = (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5
    if pooled == 0:
        return float("nan")
    d = (mean1 - mean2) / pooled
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)  # small-sample bias correction
    return d * j


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (more - less) / (n1 * n2)
