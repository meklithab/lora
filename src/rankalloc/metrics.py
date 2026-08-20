"""Paired deltas, minimum detectable effect, and effect sizes. BUILD_SPEC.md §4.8.

No significance tests: at n=3 the Wilcoxon signed-rank floor is p=0.25, at n=5 it's p=0.0625 -- the
test cannot return significance regardless of effect size, so this module reports what can actually
be claimed instead: paired deltas with every individual seed pair, the noise floor, and the effect
size we'd need to be able to detect.
"""
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


def minimum_detectable_effect(sigma: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """(t_{1-alpha/2,df} + t_{power,df}) * sigma * sqrt(2/n), for two independent groups of size n."""
    if n < 2:
        return float("nan")
    df = 2 * n - 2
    t_alpha = _stats.t.ppf(1 - alpha / 2, df)
    t_power = _stats.t.ppf(power, df)
    return (t_alpha + t_power) * sigma * (2 / n) ** 0.5


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
