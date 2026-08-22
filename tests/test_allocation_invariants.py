"""Regression tests for the allocation defects found in the P6 audit.

Each test here corresponds to a specific way the solver previously produced an allocation that was
not attributable to the signal that was supposed to produce it. The originals in
test_allocation.py all passed against the buggy solver, because they only ever exercised weight
distributions mild enough that r_max never bound -- which is precisely the regime where the defect
was invisible.
"""
import math

import numpy as np
import pytest

from rankalloc.allocation import (
    ModuleSpec,
    _sharpen,
    budget_for_uniform_rank,
    solve_allocation,
    strategy_weights,
)

HIDDEN = 896
KV_DIM = 128  # grouped-query attention: 2 KV heads x 64, vs 14 query heads
INTERMEDIATE = 4864
N_LAYERS = 8


def build_modules(n_layers=N_LAYERS):
    modules = []
    for layer in range(n_layers):
        specs = [
            ("q_proj", HIDDEN, HIDDEN),
            ("k_proj", HIDDEN, KV_DIM),
            ("v_proj", HIDDEN, KV_DIM),
            ("o_proj", HIDDEN, HIDDEN),
            ("gate_proj", HIDDEN, INTERMEDIATE),
            ("up_proj", HIDDEN, INTERMEDIATE),
            ("down_proj", INTERMEDIATE, HIDDEN),
        ]
        for name, d_in, d_out in specs:
            modules.append(ModuleSpec(name=f"layers.{layer}.{name}", d_in=d_in, d_out=d_out, layer_idx=layer))
    return modules


def heavy_tailed_weights(modules, seed):
    """Dirichlet(0.3) -- deliberately far more skewed than uniform(0.001, 1.0). This is the regime
    that drives modules into the r_max clamp and therefore generates surplus budget to redistribute;
    without it the old largest-remainder defect never fires.
    """
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.full(len(modules), 0.3))
    return {m.name: float(d) for m, d in zip(sorted(modules, key=lambda x: x.name), draws)}


# ---------------------------------------------------------------------------
# The quota property: no module may receive rank the signal did not ask for.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial_seed", range(60))
def test_quota_property_holds_under_heavy_tailed_weights(trial_seed):
    """floor(rho_m) <= r_m <= ceil(rho_m) for every module.

    The pre-fix solver reused a fractional remainder that was computed once and never updated, so a
    single module could absorb the entire r_max surplus: on the real GSM8K probe, a module whose
    continuous target was rank 6 came out at rank 128, and 13.6% of the total budget was assigned by
    rounding order rather than by the gradient signal.
    """
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    weights = heavy_tailed_weights(modules, trial_seed)
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=128)

    assert alloc.quota_max_deviation < 1.0
    for row in alloc.per_module:
        rho, r = row["ideal_rank"], row["rank"]
        assert math.floor(rho) - 1e-9 <= r <= math.ceil(rho) + 1e-9, (
            f"{row['name']}: rank {r} outside quota [{math.floor(rho)}, {math.ceil(rho)}] "
            f"of continuous target {rho:.3f}"
        )


@pytest.mark.parametrize("trial_seed", range(30))
def test_no_module_exceeds_its_own_max_useful_rank(trial_seed):
    """rank(dW) <= min(d_in, d_out), so rank beyond that is provably wasted parameters.

    Under GQA, k_proj/v_proj have d_out=128; a global r_max of 128 hid this by coincidence rather
    than enforcing it. In the shipped gradnorm_prop allocation, 17 modules sat at or above full rank.
    """
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    weights = heavy_tailed_weights(modules, trial_seed)
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=1024)
    by_name = {m.name: m for m in modules}
    for name, r in alloc.rank_pattern.items():
        m = by_name[name]
        assert r <= min(m.d_in, m.d_out), f"{name}: rank {r} exceeds max useful rank {min(m.d_in, m.d_out)}"


@pytest.mark.parametrize("trial_seed", range(30))
def test_rank_is_monotone_in_value_density(trial_seed):
    """Higher sharpened value density must not receive strictly less rank, beyond one unit of
    rounding slack. This is the ordering the whole experiment rests on: if it fails, "rank tracks
    the signal" is false.
    """
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    weights = heavy_tailed_weights(modules, trial_seed)
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=128)
    rows = sorted(alloc.per_module, key=lambda row: -row["normalized_weight"])
    for hi, lo in zip(rows, rows[1:]):
        # Only meaningful where neither is pinned at a clamp.
        if hi["rank"] >= hi["r_cap"] or lo["rank"] >= lo["r_cap"]:
            continue
        if hi["ideal_rank"] < lo["ideal_rank"]:
            continue
        assert hi["rank"] >= lo["rank"] - 1, (
            f"{hi['name']} (density {hi['normalized_weight']:.3e}, rank {hi['rank']}) ranked below "
            f"{lo['name']} (density {lo['normalized_weight']:.3e}, rank {lo['rank']})"
        )


# ---------------------------------------------------------------------------
# Temperature must not deform the baseline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 5.0, 50.0])
def test_uniform_is_exactly_uniform_at_every_temperature(temperature):
    """The uniform strategy expresses "equal rank" as weight = c_m. Sharpening the raw weight would
    give rho_m ~ c_m^(1/T)/c_m, quietly turning the *baseline* into a width-driven allocation at any
    T != 1. Sharpening the value density w_m/c_m instead leaves it flat at every T.
    """
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    alloc = solve_allocation(modules, strategy_weights("uniform", modules), budget, temperature=temperature)
    assert set(alloc.rank_pattern.values()) == {16}
    assert alloc.abs_error == 0
    assert alloc.rel_error == 0


def test_temperature_one_matches_raw_weight_normalisation_when_no_clamp_binds():
    """Backwards-compatibility guard: with no clamp active, sharpening the value density at T=1 is
    algebraically identical to normalising the raw weights, so the tier-1 protocol (T=1) is
    unchanged by the fix.

    The equivalence is deliberately scoped to the unclamped case. As soon as any module hits r_min
    or r_max the two differ, and that difference *is* the fix: water-filling returns the clamped
    module's unspendable budget to the others in proportion to their demand, whereas the legacy
    normalisation simply orphaned it.
    """
    modules = build_modules(n_layers=4)
    budget = budget_for_uniform_rank(modules, 16)
    rng = np.random.default_rng(0)
    # keep every target near rank 16 so neither clamp can bind
    weights = {m.name: m.c * float(rng.uniform(0.8, 1.25)) for m in modules}
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=128, temperature=1.0)
    assert alloc.n_clamped_low == 0 and alloc.n_clamped_high == 0, "test needs an unclamped regime"

    total_w = sum(weights.values())
    for row in alloc.per_module:
        legacy_rho = budget * (weights[row["name"]] / total_w) / row["c"]
        assert row["ideal_rank"] == pytest.approx(legacy_rho, rel=1e-9)


# ---------------------------------------------------------------------------
# Budget fidelity and clamp handling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial_seed", range(40))
def test_never_overspends_and_residual_is_bounded(trial_seed):
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    weights = heavy_tailed_weights(modules, trial_seed)
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=128)
    assert alloc.params_total <= budget
    assert alloc.abs_error <= max(m.c for m in modules)
    assert alloc.rel_error < 0.005  # the README's budget-fidelity bar
    recomputed = sum(alloc.rank_pattern[m.name] * m.c for m in modules)
    assert recomputed == alloc.params_total


def test_rmax_surplus_is_returned_proportionally_not_to_one_module():
    """When many modules clamp at r_max, the freed budget must flow back to the unclamped modules in
    proportion to their own demand. The pre-fix greedy sent essentially all of it to whichever
    module had the largest static fractional remainder.
    """
    modules = build_modules()
    budget = budget_for_uniform_rank(modules, 16)
    # a handful of modules demand far more than r_max; the rest are near-equal
    weights = {m.name: 1.0 for m in modules}
    for m in modules[:10]:
        weights[m.name] = 500.0
    alloc = solve_allocation(modules, weights, budget, r_min=1, r_max=64)

    assert alloc.n_clamped_high > 0, "test needs the r_max clamp to actually bind"
    # among the low-weight modules, ranks must stay tightly clustered: no single winner
    low = [row for row in alloc.per_module if row["weight"] == 1.0]
    ranks = [row["rank"] for row in low]
    assert max(ranks) - min(ranks) <= 1, f"surplus concentrated: low-weight ranks span {min(ranks)}..{max(ranks)}"


def test_infeasible_budget_raises_rather_than_silently_clipping():
    modules = build_modules(n_layers=1)
    tiny = sum(m.c for m in modules) * 0.5  # cannot even afford r_min=1 everywhere
    with pytest.raises(ValueError, match="cannot cover r_min"):
        solve_allocation(modules, strategy_weights("uniform", modules), tiny, r_min=1)

    huge = sum(min(128, m.d_in, m.d_out) * m.c for m in modules) * 2
    with pytest.raises(ValueError, match="exceeds the maximum spendable"):
        solve_allocation(modules, strategy_weights("uniform", modules), huge, r_max=128)


def test_sharpen_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        _sharpen({"a": 1.0}, 0.0)
    with pytest.raises(ValueError):
        _sharpen({"a": 1.0}, -1.0)
