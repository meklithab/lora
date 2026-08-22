"""Budget solver and rank-allocation strategies -- the core contribution. BUILD_SPEC.md 4.4.

solve_allocation has no knowledge of the LoRA scaling mode (alpha computation is modeling.py's job,
P3) -- it answers exactly one question: given per-module weights and a parameter budget, what integer
rank does each module get, and how close does the realised spend land to the budget.

The solver is a two-stage apportionment (see solve_allocation's docstring for the derivation):

  1. water-fill a multiplier lambda so that budget freed by r_min / r_max clamps is returned to the
     unclamped modules *in proportion to their own demand*, and
  2. integerise by largest remainder with each module eligible for at most one +1,

which together guarantee the quota property  floor(rho_m) <= r_m <= ceil(rho_m)  for every module.
That guarantee is the whole point: without it, surplus budget can be funnelled into whichever module
happens to have the largest fractional remainder, and the resulting allocation stops being
attributable to the signal that was supposed to produce it.
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

# Float slack used when comparing a continuous target against its floor/ceil. The water-fill stage
# solves lambda by exact division rather than bisection precisely so that an integral target (the
# uniform baseline's rho_m = budget_rank) lands on the integer exactly; this tolerance only absorbs
# the last-ulp noise of the multiply-back.
_QUOTA_TOL = 1e-9


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    d_in: int
    d_out: int
    layer_idx: int = 0

    @property
    def c(self) -> int:
        return self.d_in + self.d_out

    @property
    def max_useful_rank(self) -> int:
        """rank(dW) <= min(d_in, d_out) for any dW = B @ A, so rank beyond this is provably wasted
        parameters rather than extra capacity. Binds hard under grouped-query attention:
        Qwen2.5-0.5B gives k_proj/v_proj d_out=128, so r=128 there is a *dense* reparameterisation
        of the frozen weight, not a low-rank adaptation.
        """
        return min(self.d_in, self.d_out)


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
    r_min: int = 1
    r_max: int = 128
    quota_max_deviation: float = 0.0  # max_m |r_m - rho_m|; must be < 1 by construction
    n_clamped_low: int = 0
    n_clamped_high: int = 0
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
    """Importance weights in *parameter-share* space: a module's target share of the budget is
    proportional to its weight. The uniform strategy therefore returns c_m (equal share per unit of
    cost, hence equal rank), which is what makes it the r=budget_rank baseline.
    """
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


def _sharpen(values: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Raise to the power 1/temperature and renormalise to sum 1.

    Applied to the *value density* v_m = w_m / c_m (score per parameter), never to the raw weight.
    That distinction matters for two reasons:

      - it keeps the uniform baseline uniform at every temperature (uniform has w_m = c_m, hence
        v_m = 1, hence a flat allocation for any T -- sharpening the raw weight instead would give
        rho_m proportional to c_m^(1/T)/c_m and quietly turn the *baseline* into a width-driven
        allocation), and
      - it is the form implied by treating temperature as a spectral-decay exponent: if the marginal
        value of the k-th rank unit in module m is v_m * k^(-T), then equalising marginal value per
        unit cost gives r_m proportional to (w_m / c_m)^(1/T), i.e. cost inside the exponent.

    At T = 1 this is algebraically identical to normalising the raw weights, so the tier-1 protocol
    is unchanged; only T != 1 behaves differently (and correctly).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    sharpened = {k: v ** (1.0 / temperature) for k, v in values.items()}
    total = sum(sharpened.values())
    if total <= 0:
        raise ValueError("weights sum to zero after sharpening")
    return {k: v / total for k, v in sharpened.items()}


def _water_fill(names, c, u, budget, r_min, r_cap):
    """Find the continuous allocation rho_m = clip(lambda * u_m, r_min, r_cap_m) with
    sum_m c_m rho_m == budget.

    sum_m c_m clip(lambda u_m, ...) is continuous and non-decreasing in lambda, so the root is
    unique whenever r_min*sum(c) <= budget <= sum(c_m r_cap_m). We solve it by active set rather
    than bisection: given a guess at which modules are clamped, lambda follows by exact division, so
    an integral target lands on the integer exactly instead of at 15.999999997 (which would then
    floor to 15 and silently corrupt the uniform baseline).

    Each pass can only *add* modules to a clamp set, so the loop runs at most len(names) times.
    """
    free = set(names)
    clamped_low, clamped_high = set(), set()
    lam = 0.0

    for _ in range(len(names) + 2):
        fixed = sum(c[n] * r_min for n in clamped_low) + sum(c[n] * r_cap[n] for n in clamped_high)
        denom = sum(c[n] * u[n] for n in free)
        if not free or denom <= 0:
            break
        lam = (budget - fixed) / denom
        newly_low = {n for n in free if lam * u[n] < r_min}
        newly_high = {n for n in free if lam * u[n] > r_cap[n]}
        if not newly_low and not newly_high:
            break
        clamped_low |= newly_low
        clamped_high |= newly_high
        free -= newly_low | newly_high

    rho = {}
    for n in names:
        if n in clamped_low:
            rho[n] = float(r_min)
        elif n in clamped_high:
            rho[n] = float(r_cap[n])
        else:
            rho[n] = min(max(lam * u[n], float(r_min)), float(r_cap[n]))
    return rho, len(clamped_low), len(clamped_high)


def solve_allocation(
    modules: List[ModuleSpec],
    weights: Dict[str, float],
    budget: float,
    r_min: int = 1,
    r_max: int = 128,
    temperature: float = 1.0,
) -> Allocation:
    """Integer ranks under a fixed parameter budget.

    Formally: choose r_m in [r_min, min(r_max, d_in, d_out)] integer, minimising distortion from the
    continuous target rho while spending at most `budget`, where rho is the budget-feasible
    projection of the sharpened value density. The returned allocation satisfies the *quota
    property*:

        floor(rho_m) <= r_m <= ceil(rho_m)   for every module m

    which is what makes the allocation attributable to `weights`: no module can receive rank the
    signal did not ask for. The residual under-spend is bounded by the cost of the cheapest module
    that still failed to fit, hence rel_error is O(max_m c_m / budget).
    """
    if not modules:
        raise ValueError("modules must be non-empty")
    if r_min < 0 or r_max < r_min:
        raise ValueError(f"invalid rank clamp range: r_min={r_min}, r_max={r_max}")
    names = [m.name for m in modules]
    if set(weights) != set(names):
        raise ValueError("weights keys must exactly match module names")
    bad = {n: w for n, w in weights.items() if not math.isfinite(w) or w < 0}
    if bad:
        raise ValueError(
            f"non-finite or negative weight(s): {bad} -- if these came from a probe signal, "
            "the probe's gradient stats are corrupted (e.g. an fp16 overflow step that wasn't "
            "excluded), not a solve_allocation bug"
        )

    c = {m.name: m.c for m in modules}
    r_cap = {m.name: min(r_max, m.max_useful_rank) for m in modules}
    too_tight = {n: r_cap[n] for n in names if r_cap[n] < r_min}
    if too_tight:
        raise ValueError(f"r_min={r_min} exceeds the maximum useful rank of module(s): {too_tight}")

    min_spend = sum(r_min * c[n] for n in names)
    max_spend = sum(r_cap[n] * c[n] for n in names)
    if budget < min_spend:
        raise ValueError(
            f"budget {budget} cannot cover r_min={r_min} for every module (needs {min_spend})"
        )
    if budget > max_spend:
        raise ValueError(
            f"budget {budget} exceeds the maximum spendable {max_spend} at r_max={r_max}"
        )

    density = {n: weights[n] / c[n] for n in names}
    u = _sharpen(density, temperature)
    rho, n_low, n_high = _water_fill(names, c, u, budget, r_min, r_cap)

    # Integerise: floor, then a single largest-remainder pass. Each module is eligible for at most
    # one +1, which is exactly what bounds the deviation by the quota property. (The pass does not
    # break early: a cheap module further down the remainder order may still fit after an expensive
    # one did not, and awarding it tightens the residual without breaking quota.)
    rank = {n: min(max(int(math.floor(rho[n] + _QUOTA_TOL)), r_min), r_cap[n]) for n in names}
    resid = {n: rho[n] - rank[n] for n in names}
    spent = sum(rank[n] * c[n] for n in names)

    for n in sorted(names, key=lambda k: (-resid[k], k)):
        if resid[n] <= _QUOTA_TOL or rank[n] >= r_cap[n]:
            continue
        if c[n] <= budget - spent:
            rank[n] += 1
            spent += c[n]

    quota_dev = max(abs(rank[n] - rho[n]) for n in names)
    assert quota_dev < 1.0 + _QUOTA_TOL, (
        f"quota property violated: max|r_m - rho_m| = {quota_dev:.4f} >= 1. This means budget was "
        "assigned to a module the signal did not ask for -- the allocation is no longer "
        "attributable to `weights`."
    )
    assert spent <= budget + _QUOTA_TOL, f"overspent: {spent} > {budget}"

    abs_error = abs(spent - budget)
    rel_error = abs_error / budget if budget else 0.0

    per_module = [
        {
            "name": n,
            "c": c[n],
            "weight": weights[n],
            "value_density": density[n],
            "normalized_weight": u[n],  # sharpened value density, normalised to sum 1
            "ideal_rank": rho[n],  # budget-feasible continuous target (post-clamp)
            "residual": resid[n],
            "r_cap": r_cap[n],
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
        r_min=r_min,
        r_max=r_max,
        quota_max_deviation=quota_dev,
        n_clamped_low=n_low,
        n_clamped_high=n_high,
        per_module=per_module,
    )
