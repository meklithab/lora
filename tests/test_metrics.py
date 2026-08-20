import math

from rankalloc.metrics import cliffs_delta, hedges_g, minimum_detectable_effect, paired_delta, pooled_sd


def test_paired_delta_matches_by_seed_only():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {0: 0.5, 1: 1.5, 3: 9.0}  # seed 3 only in b, seed 2 only in a -- both dropped
    result = paired_delta(a, b)
    assert result.n == 2
    assert result.mean_delta == 0.5
    assert {p["seed"] for p in result.pairs} == {0, 1}


def test_paired_delta_empty_when_no_overlap():
    result = paired_delta({0: 1.0}, {1: 2.0})
    assert result.n == 0
    assert math.isnan(result.mean_delta)


def test_pooled_sd_known_value():
    # values 2,4,4,4,5,5,7,9 -> sample sd = 2.13809...
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert abs(pooled_sd(values) - 2.138089935) < 1e-6


def test_pooled_sd_needs_at_least_two():
    assert math.isnan(pooled_sd([1.0]))


def test_mde_decreases_with_more_seeds():
    mde_small_n = minimum_detectable_effect(sigma=1.0, n=3)
    mde_large_n = minimum_detectable_effect(sigma=1.0, n=20)
    assert mde_large_n < mde_small_n


def test_mde_scales_with_sigma():
    mde_1 = minimum_detectable_effect(sigma=1.0, n=5)
    mde_2 = minimum_detectable_effect(sigma=2.0, n=5)
    assert abs(mde_2 - 2 * mde_1) < 1e-9


def test_hedges_g_zero_for_identical_groups():
    g = hedges_g([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(g) < 1e-9


def test_hedges_g_positive_when_a_larger():
    g = hedges_g([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
    assert g > 0


def test_cliffs_delta_all_a_greater_is_one():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0


def test_cliffs_delta_all_a_less_is_minus_one():
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_cliffs_delta_identical_distributions_near_zero():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
