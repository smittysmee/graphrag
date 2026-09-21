"""Chapter 3's primitives, each against an answer that is known before the code runs.

Three kinds of known answer are used here and nothing else. *Analytic*: a perfectly linear pair
correlates at exactly 1.0, a fair coin carries exactly 1 bit, two identical labellings share all
their information and none of their distance. *Textbook*: the binomial tail is the one scipy's
survival function computes, and Bonferroni and Holm on a hand-written vector are the numbers the
1936 and 1979 papers define. *Planted*: a Pareto sample drawn with a known shape must come back
heavy-tailed, and a normal one must not, because that is the whole job of the flag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from graphrag.sna.stats import (
    HEAVY_TAIL_RATIO,
    MIN_CORRELATION_SAMPLE,
    binomial_p,
    compare_partitions,
    correct,
    describe,
    empirical_p,
    entropy,
    fit_distributions,
    kendall,
    mean,
    mutual_information,
    normalized_mutual_information,
    pearson,
    permutation_p,
    spearman,
    std,
    variation_of_information,
    z_score,
)

# A Pareto sample with shape 1.5 and a normal one, both large enough for a 99th percentile to
# mean something. numpy's `pareto` is the Lomax form, so +1 makes it the Pareto of §3.2.
_RNG = np.random.default_rng(20250915)
PARETO = (_RNG.pareto(1.5, 4000) + 1.0).tolist()
NORMAL = _RNG.normal(170.0, 7.0, 4000).tolist()


# ------------------------------------------------------------------ §3.1 summary statistics


def test_describe_reports_the_textbook_summary_of_a_known_sample() -> None:
    summary = describe([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert summary.count == 9
    assert summary.mean == pytest.approx(5.0)
    assert summary.median == pytest.approx(5.0)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(9.0)
    assert summary.q1 == pytest.approx(3.0)
    assert summary.q3 == pytest.approx(7.0)
    assert summary.iqr == pytest.approx(4.0)
    # A symmetric sample: variance is the mean squared deviation, skewness is zero.
    assert summary.variance == pytest.approx(60.0 / 9.0)
    assert summary.std == pytest.approx(math.sqrt(60.0 / 9.0))
    assert summary.skew == pytest.approx(0.0)


def test_describe_flags_a_pareto_sample_and_clears_a_normal_one() -> None:
    """The one judgement in §3.1 that a report depends on: is the mean the typical case?"""
    fat = describe(PARETO)
    thin = describe(NORMAL)
    assert fat.heavy_tailed is True
    assert fat.tail_ratio >= HEAVY_TAIL_RATIO
    assert fat.mean > fat.median  # the wealth picture of Figure 3.1(b)
    assert thin.heavy_tailed is False
    assert thin.tail_ratio < HEAVY_TAIL_RATIO
    assert thin.mean == pytest.approx(thin.median, abs=0.5)  # Figure 3.1(a)


def test_the_caveat_is_present_exactly_when_the_tail_is_heavy() -> None:
    fat = describe(PARETO)
    assert "standard deviations from the average" in fat.caveat
    assert "§3.1" in fat.caveat
    assert "p. 383" in fat.caveat
    assert describe(NORMAL).caveat == ""


def test_a_sample_too_small_to_judge_says_so_rather_than_claiming_a_light_tail() -> None:
    summary = describe([1.0, 2.0, 3.0])
    assert summary.heavy_tailed is False
    assert "too few to tell" in summary.caveat


def test_a_mostly_constant_sample_with_outliers_is_heavy_tailed() -> None:
    """A degree sequence of mostly ones: the middle half is one value, the top is not."""
    summary = describe([1.0] * 90 + [2.0] * 5 + [50.0] * 5)
    assert math.isinf(summary.tail_ratio)
    assert summary.heavy_tailed is True
    assert "middle half of the sample is a single value" in summary.caveat


def test_describe_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        describe([])


# ------------------------------------------------------------------ §3.2 distributions


def test_a_normal_sample_fits_the_normal_and_not_the_power_law() -> None:
    fits = fit_distributions(NORMAL)
    assert fits["normal"].params["mu"] == pytest.approx(170.0, abs=0.5)
    assert fits["normal"].params["sigma"] == pytest.approx(7.0, abs=0.5)
    assert fits["normal"].p_value > 0.05
    assert fits["normal"].ks_statistic < fits["powerlaw"].ks_statistic


def test_a_pareto_sample_recovers_its_exponent_and_beats_the_normal_fit() -> None:
    """Drawn with shape 1.5, so p(x) ∝ x^-2.5 and the MLE of §3.2 must land there."""
    fits = fit_distributions(PARETO)
    assert fits["powerlaw"].params["alpha"] == pytest.approx(2.5, abs=0.15)
    assert fits["powerlaw"].params["xmin"] == pytest.approx(1.0, abs=0.01)
    assert fits["powerlaw"].ks_statistic < fits["normal"].ks_statistic
    assert fits["normal"].p_value < 1e-6


def test_fits_undefined_on_a_sample_are_left_out_rather_than_faked() -> None:
    fits = fit_distributions([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert "normal" in fits
    assert "lognormal" not in fits
    assert "powerlaw" not in fits
    assert "exponential" not in fits


def test_fitting_needs_a_sample_worth_fitting() -> None:
    with pytest.raises(ValueError, match="at least 8 values"):
        fit_distributions([1.0, 2.0, 3.0])


# ------------------------------------------------------------------ §3.3 p-values


def test_an_observation_above_every_null_sample_scores_one_over_n_plus_one() -> None:
    null = [float(i) for i in range(99)]
    assert empirical_p(1000.0, null, "right") == pytest.approx(1.0 / 100.0)
    assert empirical_p(-1.0, null, "left") == pytest.approx(1.0 / 100.0)
    assert empirical_p(1000.0, null, "two") == pytest.approx(2.0 / 100.0)


def test_an_observation_in_the_middle_of_the_null_is_unremarkable() -> None:
    null = [float(i) for i in range(101)]
    assert empirical_p(50.0, null, "right") == pytest.approx(52.0 / 102.0)
    assert empirical_p(50.0, null, "two") == pytest.approx(1.0)


def test_empirical_p_refuses_an_empty_null() -> None:
    with pytest.raises(ValueError, match="at least one null sample"):
        empirical_p(1.0, [])


def test_a_planted_group_difference_is_found_and_an_absent_one_is_not() -> None:
    def gap(a: object, b: object) -> float:
        return float(np.mean(np.asarray(a, dtype=float)) - np.mean(np.asarray(b, dtype=float)))

    low = [float(i) for i in range(20)]
    high = [value + 50.0 for value in low]
    # Separated by more than two sample ranges: no re-split of the pooled values comes near the
    # observed gap, so the test bottoms out at the floor, which the two-tailed default doubles.
    assert permutation_p(gap, low, high, 199, seed=1) == pytest.approx(2.0 / 200.0)
    assert permutation_p(gap, low, high, 199, seed=1, tail="left") == pytest.approx(1.0 / 200.0)
    # The same numbers dealt alternately into two groups: they differ by one, which is well
    # inside what re-dealing them produces, so there is nothing to find.
    assert permutation_p(gap, low[::2], low[1::2], 199, seed=1) > 0.05


def test_a_paired_permutation_sees_the_pairing_a_pooled_one_cannot() -> None:
    def correlation(a: object, b: object) -> float:
        return pearson(list(a), list(b)).coefficient  # type: ignore[arg-type]

    x = [float(i) for i in range(30)]
    y = [3.0 * value + 1.0 for value in x]
    assert permutation_p(
        correlation, x, y, 199, seed=3, tail="right", paired=True
    ) == pytest.approx(1.0 / 200.0)


def test_a_paired_permutation_needs_pairs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        permutation_p(lambda a, b: 0.0, [1.0, 2.0], [1.0], 10, paired=True)


def test_binomial_p_is_the_survival_function_of_the_same_urn() -> None:
    """Seven greens out of ten fair draws, the §3.2 urn game read as the §3.3 test."""
    assert binomial_p(7, 10, 0.5) == pytest.approx(float(scipy_stats.binom.sf(6, 10, 0.5)))
    assert binomial_p(7, 10, 0.5) == pytest.approx(0.171875)
    assert binomial_p(7, 10, 0.5, "left") == pytest.approx(float(scipy_stats.binom.cdf(7, 10, 0.5)))
    assert binomial_p(7, 10, 0.5, "two") == pytest.approx(0.34375)
    # Every outcome is at least as extreme as zero successes on the left, and at least as
    # extreme as ten on the right.
    assert binomial_p(0, 10, 0.5, "left") == pytest.approx(1.0 / 1024.0)
    assert binomial_p(0, 10, 0.5, "right") == pytest.approx(1.0)


def test_binomial_p_refuses_impossible_counts() -> None:
    with pytest.raises(ValueError, match="between 0 and 10"):
        binomial_p(11, 10, 0.5)
    with pytest.raises(ValueError, match="at least one trial"):
        binomial_p(0, 0, 0.5)


def test_bonferroni_and_holm_on_a_known_vector() -> None:
    """Four tests, so Bonferroni multiplies by four; Holm steps 4, 3, 2, 1 down the sorted p."""
    p_values = [0.01, 0.04, 0.03, 0.005]
    assert correct(p_values, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    # sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> held at 0.06.
    assert correct(p_values, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # Holm is uniformly no more conservative than Bonferroni, which is the reason to prefer it.
    assert all(
        h <= b
        for h, b in zip(correct(p_values, "holm"), correct(p_values, "bonferroni"), strict=True)
    )


def test_benjamini_hochberg_on_the_same_vector() -> None:
    p_values = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4/1 = 0.02, 0.01*4/2 = 0.02, 0.03*4/3 = 0.04, 0.04*4/4 = 0.04.
    assert correct(p_values, "fdr_bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_a_correction_never_exceeds_one_and_never_reorders() -> None:
    p_values = [0.5, 0.6, 0.9, 0.95]
    for method in ("bonferroni", "holm", "fdr_bh"):
        adjusted = correct(p_values, method)
        assert all(value <= 1.0 for value in adjusted)
        assert sorted(range(4), key=lambda i: adjusted[i]) == sorted(
            range(4), key=lambda i: p_values[i]
        )
    assert correct([], "holm") == []


def test_correct_rejects_a_bad_method_and_a_bad_p_value() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        correct([0.1], "sidak")
    with pytest.raises(ValueError, match="between 0 and 1"):
        correct([1.4], "holm")


# ------------------------------------------------------------------ §3.4 correlation


def test_pearson_of_an_exactly_linear_sequence_is_one() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pearson(x, [2.0 * v + 7.0 for v in x]).coefficient == pytest.approx(1.0)
    assert pearson(x, [-2.0 * v + 7.0 for v in x]).coefficient == pytest.approx(-1.0)
    assert pearson(x, x).method == "pearson"
    assert pearson(x, x).n == 5


def test_spearman_survives_a_monotone_transform_that_pearson_does_not() -> None:
    """§3.4's whole argument for the rank correlation, on a cubic."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    cubed = [v**3 for v in x]
    assert spearman(x, cubed).coefficient == pytest.approx(1.0)
    assert spearman(x, [math.exp(v) for v in x]).coefficient == pytest.approx(1.0)
    assert pearson(x, cubed).coefficient < 0.95
    assert kendall(x, cubed).coefficient == pytest.approx(1.0)


def test_a_u_shape_correlates_at_zero_however_obvious_it_is() -> None:
    x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    parabola = [v**2 for v in x]
    assert pearson(x, parabola).coefficient == pytest.approx(0.0)
    assert spearman(x, parabola).coefficient == pytest.approx(0.0)


def test_a_correlation_is_undefined_without_variation_or_without_pairs() -> None:
    with pytest.raises(ValueError, match="is constant"):
        pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=f"at least {MIN_CORRELATION_SAMPLE} pairs"):
        spearman([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="paired values"):
        kendall([1.0, 2.0, 3.0], [1.0, 2.0])


# ------------------------------------------------------------------ §3.5 information


def test_entropy_counts_the_bits_the_chapter_counts() -> None:
    assert entropy([0, 1]) == pytest.approx(1.0)  # the coin flip of §3.5
    assert entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
    assert entropy(["a"] * 9) == pytest.approx(0.0)
    assert entropy(["x", "x", "x", "y"]) == pytest.approx(0.8112781244591328)


def test_identical_labellings_share_everything_and_differ_by_nothing() -> None:
    labels = [0, 0, 1, 1]
    assert mutual_information(labels, labels) == pytest.approx(1.0)  # = the entropy, in bits
    assert normalized_mutual_information(labels, labels) == pytest.approx(1.0)
    assert variation_of_information(labels, labels) == pytest.approx(0.0)
    # Renaming the groups is not a different partition.
    assert normalized_mutual_information(labels, ["b", "b", "a", "a"]) == pytest.approx(1.0)
    assert variation_of_information(labels, ["b", "b", "a", "a"]) == pytest.approx(0.0)


def test_independent_labellings_share_nothing_and_differ_by_both_entropies() -> None:
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert mutual_information(a, b) == pytest.approx(0.0)
    assert normalized_mutual_information(a, b) == pytest.approx(0.0)
    assert variation_of_information(a, b) == pytest.approx(2.0)  # 1 bit each, nothing shared


def test_a_partly_informative_labelling_saves_a_known_number_of_bits() -> None:
    """x determines half of y: knowing x saves exactly one of y's two bits."""
    x = ["p", "p", "p", "p", "q", "q", "q", "q"]
    y = ["a", "a", "b", "b", "c", "c", "d", "d"]
    assert entropy(y) == pytest.approx(2.0)
    assert mutual_information(x, y) == pytest.approx(1.0)
    assert variation_of_information(x, y) == pytest.approx(1.0)  # 2 + 1 - 2*1


def test_information_measures_need_two_labellings_of_the_same_elements() -> None:
    with pytest.raises(ValueError, match="same elements"):
        mutual_information([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="non-empty"):
        variation_of_information([], [])
    with pytest.raises(ValueError, match="at least one label"):
        entropy([])


def test_compare_partitions_scores_identical_and_unrelated_labellings() -> None:
    a = [0, 0, 0, 1, 1, 1]
    assert compare_partitions(a, a)["adjusted_rand_index"] == pytest.approx(1.0)
    assert compare_partitions(a, a)["normalized_mutual_information"] == pytest.approx(1.0)
    unrelated = compare_partitions(a, [0, 1, 0, 1, 0, 1])
    assert unrelated["adjusted_rand_index"] < 0.1
    assert unrelated["normalized_mutual_information"] < 0.1


# ------------------------------------------------------------------ null-sample helpers


def test_the_z_score_is_the_distance_from_the_null_mean_in_null_deviations() -> None:
    null = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean 3, population sd sqrt(2)
    assert mean(null) == pytest.approx(3.0)
    assert std(null) == pytest.approx(math.sqrt(2.0))
    assert z_score(3.0 + 2.0 * math.sqrt(2.0), null) == pytest.approx(2.0)
    assert z_score(3.0, null) == pytest.approx(0.0)


def test_the_helpers_print_a_number_when_there_was_no_null_to_build() -> None:
    """A graph too small to rewire yields no samples; the report still has to render."""
    assert mean([]) == 0.0
    assert std([]) == 0.0
    assert z_score(0.9, []) == 0.0
    assert z_score(0.9, [0.5, 0.5, 0.5]) == 0.0  # no spread, so no distance to measure


def test_the_z_score_and_the_empirical_p_agree_about_a_planted_effect() -> None:
    """The two currencies of §3.3 pointing the same way on one null distribution."""
    null = _RNG.normal(0.0, 1.0, 999).tolist()
    assert z_score(6.0, null) > 3.0
    assert empirical_p(6.0, null, "right") == pytest.approx(1.0 / 1000.0)
    assert abs(z_score(0.0, null)) < 2.0
    assert empirical_p(0.0, null, "two") > 0.05
