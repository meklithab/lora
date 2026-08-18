"""Budget solver and rank-allocation strategies -- the core contribution. BUILD_SPEC.md §4.4.

solve_allocation has no knowledge of the LoRA scaling mode (alpha computation is modeling.py's job,
P3) -- it answers exactly one question: given per-module weights and a parameter budget, what integer
rank does each module get, and how close does the realised spend land to the budget.
"""
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

STRATEGIES = (
    "uniform",
    "gradnorm_prop",
    "gradnorm_inverse",
    "random",
    "early_heavy",
    "late_heavy",
)


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    d_in: int
    d_out: int
    layer_idx: int = 0

    @property
    def c(self) -> int:
        return self.d_in + self.d_out


@dataclass
class Allocation:
    rank_pattern: Dict[str, int]
    budget: float
    params_total: int
    abs_error: float
    rel_error: float
    weights: Dict[str, float]  # raw (pre-sharpen) weights actually used
    strategy: Optional[str] = None
    temperature: float = 1.0
    signal: Optional[str] = None
    probe_id: Optional[str] = None
    per_module: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def budget_for_uniform_rank(modules: List[ModuleSpec], reference_rank: int) -> float:
    return reference_rank * sum(m.c for m in modules)


def strategy_weights(
    strategy: str,
    modules: List[ModuleSpec],
    *,
    signal: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    lambda_decay: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if strategy == "uniform":
        return {m.name: float(m.c) for m in modules}

    if strategy in ("gradnorm_prop", "gradnorm_inverse"):
        if signal is None:
            raise ValueError(f"strategy={strategy!r} requires a signal dict")
        missing = [m.name for m in modules if m.name not in signal]
        if missing:
            raise ValueError(f"signal missing entries for modules: {missing}")
        if strategy == "gradnorm_prop":
            return {m.name: float(signal[m.name]) for m in modules}
        return {m.name: 1.0 / (float(signal[m.name]) + eps) for m in modules}

    if strategy == "random":
        if seed is None:
            raise ValueError("strategy='random' requires a seed")
        sorted_names = sorted(m.name for m in modules)
        rng = np.random.default_rng(seed)
        draws = rng.dirichlet(np.ones(len(sorted_names)))
        return dict(zip(sorted_names, (float(d) for d in draws)))

    if strategy in ("early_heavy", "late_heavy"):
        layer_indices = [m.layer_idx for m in modules]
        max_layer = max(layer_indices) if layer_indices else 0
        sign = -1.0 if strategy == "early_heavy" else 1.0
        weights = {}
        for m in modules:
            frac = (m.layer_idx / max_layer) if max_layer > 0 else 0.0
            weights[m.name] = math.exp(sign * lambda_decay * frac)
        return weights

    raise ValueError(f"Unknown strategy: {strategy!r}")


def _sharpen(weights: Dict[str, float], temperature: float) -> Dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    sharpened = {k: v ** (1.0 / temperature) for k, v in weights.items()}
    total = sum(sharpened.values())
    if total <= 0:
        raise ValueError("weights sum to zero after sharpening")
    return {k: v / total for k, v in sharpened.items()}


def solve_allocation(
    modules: List[ModuleSpec],
    weights: Dict[str, float],
    budget: float,
    r_min: int = 1,
    r_max: int = 128,
    temperature: float = 1.0,
) -> Allocation:
    if not modules:
        raise ValueError("modules must be non-empty")
    if r_min < 0 or r_max < r_min:
        raise ValueError(f"invalid rank clamp range: r_min={r_min}, r_max={r_max}")
    names = [m.name for m in modules]
    if set(weights) != set(names):
        raise ValueError("weights keys must exactly match module names")
    c = {m.name: m.c for m in modules}

    norm_w = _sharpen(weights, temperature)

    ideal = {n: budget * norm_w[n] / c[n] for n in names}
    remainder = {n: ideal[n] - math.floor(ideal[n]) for n in names}

    rank = {n: min(max(int(math.floor(ideal[n])), r_min), r_max) for n in names}
    spent = sum(rank[n] * c[n] for n in names)

    # spend up: recover budget lost to flooring and to downward r_max clamps
    while True:
        eligible = [n for n in names if rank[n] < r_max and c[n] <= budget - spent]
        if not eligible:
            break
        pick = min(eligible, key=lambda n: (-remainder[n], n))
        rank[pick] += 1
        spent += c[pick]

    # spend down: give back budget added by upward r_min clamps
    while spent > budget:
        eligible = [n for n in names if rank[n] > r_min]
        if not eligible:
            break
        pick = min(eligible, key=lambda n: (remainder[n], n))
        rank[pick] -= 1
        spent -= c[pick]

    abs_error = abs(spent - budget)
    rel_error = abs_error / budget if budget else 0.0

    per_module = [
        {
            "name": n,
            "c": c[n],
            "weight": weights[n],
            "normalized_weight": norm_w[n],
            "ideal_rank": ideal[n],
            "rank": rank[n],
            "params": rank[n] * c[n],
        }
        for n in names
    ]

    return Allocation(
        rank_pattern=rank,
        budget=budget,
        params_total=spent,
        abs_error=abs_error,
        rel_error=rel_error,
        weights=dict(weights),
        temperature=temperature,
        per_module=per_module,
    )
