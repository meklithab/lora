"""Regression tests for the statistics and LoRA-scaling defects found in the P6 audit."""
import math

import pytest

from rankalloc.metrics import (
    minimum_detectable_effect,
    paired_minimum_detectable_effect,
    pooled_sd,
    pooled_two_sample_sd,
    sign_flip_test,
)
from rankalloc.modeling import compute_alpha_pattern


# ---------------------------------------------------------------------------
# MDE must reflect BOTH arms' variance.
# ---------------------------------------------------------------------------


def test_pooled_sd_reflects_the_noisier_arm():
    """The shipped analysis used the uniform baseline's sigma for every comparison. On the real
    tier-1 data the gradnorm_prop arm's sigma was 6.3x the baseline's, so the threshold every
    verdict was gated on was roughly 4.5x too small.
    """
    tight = [0.4836, 0.4840, 0.4840, 0.4842, 0.4845]      # uniform-like
    loose = [0.4857, 0.4862, 0.4860, 0.4855, 0.4904]      # gradnorm_prop-like, one outlier seed
    sd_tight, sd_loose = pooled_sd(tight), pooled_sd(loose)
    assert sd_loose > 4 * sd_tight

    pooled = pooled_two_sample_sd(loose, tight)
    assert sd_tight < pooled < sd_loose
    assert minimum_detectable_effect(pooled, 5) > minimum_detectable_effect(sd_tight, 5)


def test_mde_grows_with_the_treatment_arm_variance():
    base = [1.000, 1.001, 0.999, 1.000, 1.001]
    calm = [1.010, 1.011, 1.009, 1.010, 1.011]
    wild = [1.010, 1.050, 0.970, 1.010, 1.011]
    mde_calm = minimum_detectable_effect(pooled_two_sample_sd(calm, base), 5)
    mde_wild = minimum_detectable_effect(pooled_two_sample_sd(wild, base), 5)
    assert mde_wild > mde_calm


def test_pooled_two_sample_sd_handles_degenerate_arms():
    assert math.isnan(pooled_two_sample_sd([1.0], [2.0]))
    assert pooled_two_sample_sd([1.0], [1.0, 2.0, 3.0]) == pytest.approx(pooled_sd([1.0, 2.0, 3.0]))


def test_independent_and_paired_mde_are_different_formulas():
    """They are not interchangeable. The pipeline's pairing is nominal (LoRA A-init consumes RNG per
    module in an amount that depends on rank, so two conditions at the same seed share no
    initialisation), which is why analyze.py gates on the independent-groups form.
    """
    sigma, n = 0.01, 5
    assert minimum_detectable_effect(sigma, n) != pytest.approx(paired_minimum_detectable_effect(sigma, n))


# ---------------------------------------------------------------------------
# The exact permutation test must report its own floor.
# ---------------------------------------------------------------------------


def test_sign_flip_reports_attainable_floor():
    """At n=5 the smallest two-sided p the design can return is 0.0625. Reporting it without the
    floor invites reading 'p=0.0625' as 'nearly significant' rather than 'every seed agreed and the
    design cannot say more'.
    """
    all_same_direction = [0.0012, 0.0026, 0.0017, 0.0015, 0.0064]
    res = sign_flip_test(all_same_direction)
    assert res["p_value"] == pytest.approx(2.0 ** (1 - 5))
    assert res["p_value"] == pytest.approx(res["p_floor"])
    assert sign_flip_test([1.0, 1.0, 1.0])["p_floor"] == pytest.approx(0.25)


def test_sign_flip_is_symmetric_and_bounded():
    d = [0.3, -0.1, 0.2]
    a, b = sign_flip_test(d), sign_flip_test([-x for x in d])
    assert a["p_value"] == pytest.approx(b["p_value"])
    for deltas in ([0.0, 0.0, 0.0], [5.0, -5.0], [1.0, 2.0, 3.0, 4.0]):
        p = sign_flip_test(deltas)["p_value"]
        assert 0.0 <= p <= 1.0


def test_sign_flip_returns_one_for_a_null_effect():
    assert sign_flip_test([0.0, 0.0, 0.0, 0.0])["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LoRA scaling: each mode must be the r-exponent it claims to be.
# ---------------------------------------------------------------------------


def _scaling(mode, ranks, alpha_ratio=2, fixed_alpha=32, reference_rank=16):
    """The multiplier s(r) that PEFT actually applies to B @ A x."""
    pattern = compute_alpha_pattern({f"m{r}": r for r in ranks}, mode, alpha_ratio, fixed_alpha, reference_rank)
    out = {}
    for r in ranks:
        alpha = pattern[f"m{r}"]
        out[r] = alpha / math.sqrt(r) if mode == "rslora" else alpha / r
    return out


RANKS = [1, 4, 16, 64, 128]


def test_all_scaling_modes_agree_at_the_reference_rank():
    """Anchoring at r = budget_rank is what makes the modes comparable: they must differ only in how
    they extrapolate away from R, not by an overall constant that would confound 'which r-exponent'
    with 'how strong is the adapter'.
    """
    for mode in ("constant_ratio", "rslora", "fixed_alpha"):
        assert _scaling(mode, RANKS)[16] == pytest.approx(2.0), mode


def test_rslora_scaling_decreases_as_one_over_sqrt_r():
    """Regression on a real bug: passing alpha = alpha_ratio * r together with use_rslora=True gave
    s(r) = alpha_ratio * sqrt(r) -- scaling that GROWS with rank, the exact opposite of the
    alpha/sqrt(r) rule rsLoRA specifies.
    """
    s = _scaling("rslora", RANKS)
    assert s[128] < s[16] < s[1]
    assert s[64] / s[16] == pytest.approx(math.sqrt(16 / 64))


def test_scaling_mode_r_exponents_are_ordered_as_documented():
    """s(r) ~ r^k with k = 0 (constant_ratio), -1/2 (rslora), -1 (fixed_alpha)."""
    expected = {"constant_ratio": 0.0, "rslora": -0.5, "fixed_alpha": -1.0}
    for mode, k in expected.items():
        s = _scaling(mode, RANKS)
        slope = math.log(s[128] / s[1]) / math.log(128 / 1)
        assert slope == pytest.approx(k, abs=1e-9), f"{mode}: exponent {slope}, expected {k}"


def test_constant_ratio_is_the_only_mode_that_does_not_shrink_with_rank():
    """Under AdamW from B=0 the adapter's contribution scales as s(r) * r^theta with
    theta in [1/2, 1], so rank-neutrality needs an s-exponent in [-1, -1/2]. constant_ratio sits at
    0, outside that bracket, which means it makes high-rank modules adapt *faster*, not equally --
    the opposite of the isolation it is assumed to provide.
    """
    assert _scaling("constant_ratio", RANKS)[128] == pytest.approx(_scaling("constant_ratio", RANKS)[1])
    assert _scaling("rslora", RANKS)[128] < _scaling("rslora", RANKS)[1]
    assert _scaling("fixed_alpha", RANKS)[128] < _scaling("fixed_alpha", RANKS)[1]


def test_unknown_scaling_mode_raises():
    with pytest.raises(ValueError):
        compute_alpha_pattern({"m": 8}, "not_a_mode", 2, 32, 16)
