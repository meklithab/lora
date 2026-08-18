import math

import numpy as np
import pytest

from rankalloc.allocation import (
    ModuleSpec,
    budget_for_uniform_rank,
    solve_allocation,
    strategy_weights,
)

HIDDEN = 896
KV_DIM = 128
INTERMEDIATE = 4864
N_LAYERS = 4


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


def max_c(modules):
    return max(m.c for m in modules)


# ---------------------------------------------------------------------------
# solve_allocation invariants (BUILD_SPEC.md §7, §4.4)
# ---------------------------------------------------------------------------


def test_uniform_weights_integral_r_gives_exact_ranks():
    modules = build_modules()
    R = 16
    budget = budget_for_uniform_rank(modules, R)
    weights = strategy_weights("uniform", modules)
    alloc = solve_allocation(modules, weights, budget)
    assert all(r == R for r in alloc.rank_pattern.values())
    assert alloc.abs_error == 0
    assert alloc.params_total == budget


@pytest.mark.parametrize("trial_seed", range(200))
def test_budget_error_bounded_by_max_module_cost(trial_seed):
    modules = build_modules()
    R = 16
    budget = budget_for_uniform_rank(modules, R)
    rng = np.random.default_rng(trial_seed)
    weights = {m.name: float(rng.uniform(0.001, 1.0)) for m in modules}
    alloc = solve_allocation(modules, weights, budget)
    assert alloc.abs_error <= max_c(modules)


def test_clamps_respected_under_extreme_weights():
    modules = build_modules()
    R = 16
    budget = budget_for_uniform_rank(modules, R)
    weights = {m.name: 0.0001 for m in modules}
    # one module gets nearly all the weight
    dominant = modules[0].name
    weights[dominant] = 0.999
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    r_min, r_max = 1, 128
    alloc = solve_allocation(modules, weights, budget, r_min=r_min, r_max=r_max)
    for r in alloc.rank_pattern.values():
        assert r_min <= r <= r_max


def test_permutation_invariance():
    modules = build_modules()
    weights = strategy_weights("uniform", modules)
    budget = budget_for_uniform_rank(modules, 16)
    alloc_a = solve_allocation(modules, weights, budget)

    shuffled = list(reversed(modules))
    alloc_b = solve_allocation(shuffled, weights, budget)

    assert alloc_a.rank_pattern == alloc_b.rank_pattern
    assert alloc_a.params_total == alloc_b.params_total


def test_determinism_across_repeated_calls():
    modules = build_modules()
    weights = strategy_weights("gradnorm_prop", modules, signal={m.name: (hash(m.name) % 97) + 1 for m in modules})
    budget = budget_for_uniform_rank(modules, 16)
    alloc_a = solve_allocation(modules, weights, budget)
    alloc_b = solve_allocation(modules, weights, budget)
    assert alloc_a.rank_pattern == alloc_b.rank_pattern
    assert alloc_a.params_total == alloc_b.params_total


def test_temperature_high_approaches_uniform():
    # temperature acts on the *weights* (normalized_weight), not directly on rank -- ranks still
    # differ by module cost c_m even under perfectly uniform weights (see the uniform-strategy test
    # above), so the thing that should flatten out under temperature->inf is normalized_weight.
    modules = build_modules(n_layers=1)
    skewed = {m.name: 1.0 for m in modules}
    skewed[modules[0].name] = 100.0  # one module heavily favoured
    budget = budget_for_uniform_rank(modules, 16)

    alloc_hot = solve_allocation(modules, skewed, budget, temperature=1e6)
    n = len(modules)
    normalized = {row["name"]: row["normalized_weight"] for row in alloc_hot.per_module}
    for w in normalized.values():
        assert abs(w - 1.0 / n) < 1e-3


def test_temperature_low_concentrates():
    modules = build_modules(n_layers=1)
    skewed = {m.name: 1.0 for m in modules}
    dominant = modules[0].name
    skewed[dominant] = 2.0  # only a mild edge, but temperature->0 should still blow it up
    budget = budget_for_uniform_rank(modules, 16)

    alloc_cold = solve_allocation(modules, skewed, budget, temperature=0.02, r_max=1_000_000)
    total_rank = sum(alloc_cold.rank_pattern.values())
    assert alloc_cold.rank_pattern[dominant] / total_rank > 0.9


def test_params_total_matches_recomputed_from_rank_pattern():
    modules = build_modules()
    weights = strategy_weights("random", modules, seed=0)
    budget = budget_for_uniform_rank(modules, 16)
    alloc = solve_allocation(modules, weights, budget)
    c = {m.name: m.c for m in modules}
    recomputed = sum(alloc.rank_pattern[n] * c[n] for n in alloc.rank_pattern)
    assert recomputed == alloc.params_total


def test_weights_keys_must_match_modules():
    modules = build_modules()
    weights = strategy_weights("uniform", modules)
    del weights[modules[0].name]
    with pytest.raises(ValueError):
        solve_allocation(modules, weights, budget_for_uniform_rank(modules, 16))


def test_temperature_must_be_positive():
    modules = build_modules()
    weights = strategy_weights("uniform", modules)
    with pytest.raises(ValueError):
        solve_allocation(modules, weights, budget_for_uniform_rank(modules, 16), temperature=0)


# ---------------------------------------------------------------------------
# strategy_weights
# ---------------------------------------------------------------------------


def test_gradnorm_prop_orders_by_signal():
    modules = build_modules(n_layers=1)
    signal = {m.name: float(i + 1) for i, m in enumerate(modules)}
    weights = strategy_weights("gradnorm_prop", modules, signal=signal)
    assert weights == {n: float(v) for n, v in signal.items()}


def test_gradnorm_inverse_is_inverse_ordering():
    modules = build_modules(n_layers=1)
    signal = {m.name: float(i + 1) for i, m in enumerate(modules)}
    weights = strategy_weights("gradnorm_inverse", modules, signal=signal)
    prop = strategy_weights("gradnorm_prop", modules, signal=signal)
    ordered_prop = sorted(prop, key=prop.get)
    ordered_inverse = sorted(weights, key=weights.get)
    assert ordered_prop == list(reversed(ordered_inverse))


def test_random_strategy_reproducible_with_same_seed():
    modules = build_modules(n_layers=1)
    w1 = strategy_weights("random", modules, seed=123)
    w2 = strategy_weights("random", modules, seed=123)
    assert w1 == w2


def test_random_strategy_differs_across_seeds():
    modules = build_modules(n_layers=1)
    w1 = strategy_weights("random", modules, seed=1)
    w2 = strategy_weights("random", modules, seed=2)
    assert w1 != w2


def test_random_strategy_permutation_invariant():
    modules = build_modules(n_layers=1)
    w1 = strategy_weights("random", modules, seed=7)
    w2 = strategy_weights("random", list(reversed(modules)), seed=7)
    assert w1 == w2


def test_early_heavy_decreasing_with_layer_idx():
    modules = build_modules()
    weights = strategy_weights("early_heavy", modules, lambda_decay=2.0)
    q_by_layer = {m.layer_idx: weights[m.name] for m in modules if m.name.endswith("q_proj")}
    layers_sorted = sorted(q_by_layer)
    values = [q_by_layer[l] for l in layers_sorted]
    assert values == sorted(values, reverse=True)


def test_late_heavy_increasing_with_layer_idx():
    modules = build_modules()
    weights = strategy_weights("late_heavy", modules, lambda_decay=2.0)
    q_by_layer = {m.layer_idx: weights[m.name] for m in modules if m.name.endswith("q_proj")}
    layers_sorted = sorted(q_by_layer)
    values = [q_by_layer[l] for l in layers_sorted]
    assert values == sorted(values)


def test_unknown_strategy_raises():
    modules = build_modules(n_layers=1)
    with pytest.raises(ValueError):
        strategy_weights("not_a_real_strategy", modules)
