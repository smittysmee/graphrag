"""The statistics chapter of the Atlas, in one module, so every later test quotes the same one.

Chapter 3 of *The Atlas for the Aspiring Network Scientist* is the toolbox the rest of the book
spends itself on: summary statistics (§3.1), the distributions a network scientist meets (§3.2),
p-values and what they do not mean (§3.3), correlation coefficients (§3.4), and mutual
information (§3.5). Nothing here is about networks. Everything here is called by something that
is: the null models, the assortativity, the backbone filters, the community comparisons.

Four things the chapter insists on, which shape the signatures below.

*A mean is not always the typical case* (§3.1). The book's sharpest statement of this is at
p. 383, arguing against naive edge-weight thresholds: a fat-tailed distribution "lacks of a
well-defined average value and has undefined variance… You cannot motivate your threshold choice
by saying that it is 'x standard deviations from the average'". :func:`describe` therefore still
returns the mean and the standard deviation -- refusing to compute them would only push the
caller to ``numpy`` -- but it decides whether the sample is heavy-tailed and hands back a
``caveat`` string that says so, in words, next to the numbers. A report that prints the mean
prints the caveat.

*A p-value is not an effect size* (§3.3). It is the probability of an observation at least this
extreme under the null, and nothing else: not how likely you are to be right, not how strong the
effect is, not how much evidence you have. The functions here return p-values and never verdicts;
the verdict sentence belongs to the report that also printed n and named the null.

*Many tests need a correction* (§3.3). Run a hundred tests at p < 0.01 and you expect one false
rejection; the book's own arithmetic is that at least one lands with probability 63%. Any code
here that produces more than one p-value is expected to pass them through :func:`correct`.

*Correlation is a family, not a number* (§3.4). Pearson measures a linear relationship, Spearman
a monotone one, Kendall a concordance of ranks. Which one a report used is part of the finding.

Base-2 logarithms throughout the information measures, because §3.5 counts in bits.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score

__all__ = [
    "CORRECTIONS",
    "DISTRIBUTIONS",
    "HEAVY_TAIL_RATIO",
    "TAILS",
    "Correlation",
    "FitResult",
    "Summary",
    "binomial_p",
    "compare_partitions",
    "correct",
    "describe",
    "empirical_p",
    "entropy",
    "fit_distributions",
    "kendall",
    "likelihood_ratio",
    "mean",
    "mutual_information",
    "normalized_mutual_information",
    "pearson",
    "permutation_p",
    "spearman",
    "std",
    "variation_of_information",
    "z_score",
]

#: The tails a one-sample test can be taken on. ``right`` asks whether the observation is
#: unusually large, ``left`` unusually small, ``two`` unusual in either direction.
TAILS: tuple[str, ...] = ("right", "left", "two")

#: The multiple-comparison corrections of §3.3, plus Benjamini-Hochberg, which the book does not
#: name but which is what a report with dozens of edges or nodes to test actually needs.
CORRECTIONS: tuple[str, ...] = ("bonferroni", "holm", "fdr_bh")

#: The distributions of §3.2 that :func:`fit_distributions` will fit. Uniform, binomial, Poisson
#: and hypergeometric are in the chapter too but are not fitted here: see that function.
DISTRIBUTIONS: tuple[str, ...] = ("normal", "exponential", "lognormal", "powerlaw")

#: How many interquartile ranges above the median the 99th percentile has to sit before
#: :func:`describe` calls a sample heavy-tailed. See that function for where 4.0 comes from.
HEAVY_TAIL_RATIO = 4.0

#: What that same ratio is for a normal sample, quoted in the caveat so a reader can see the
#: scale. (P99 - P50) / IQR = 2.326 / 1.349 for a Gaussian, whatever its mean and variance.
NORMAL_TAIL_RATIO = 1.72

#: Below this many values the shape of a tail is not a measurement, so :func:`describe` says it
#: cannot tell rather than guessing. Eight is where the 99th percentile stops being the maximum
#: and one outlier stops being able to move every quartile.
MIN_TAIL_SAMPLE = 8

#: Below this many values a distribution fit is arithmetic, not evidence.
MIN_FIT_SAMPLE = 8

#: Pearson, Spearman and Kendall all need at least three pairs: with two, every monotone pair is
#: a perfect correlation and the p-value is undefined.
MIN_CORRELATION_SAMPLE = 3

_LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- §3.1 summary


@dataclass(frozen=True)
class Summary:
    """One sample described, with the warning that decides whether to quote the mean.

    ``heavy_tailed`` and ``caveat`` are the point of this dataclass. Everything else is
    arithmetic ``numpy`` would also do.
    """

    count: int
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    q1: float
    q3: float
    maximum: float
    skew: float
    #: (P99 - median) / IQR: how far the top of the sample sits above its middle, measured in
    #: units of the middle's own spread. ``inf`` when the middle half is a single value.
    tail_ratio: float
    heavy_tailed: bool
    caveat: str

    @property
    def iqr(self) -> float:
        """The interquartile range: the spread of the middle half, which survives a fat tail."""
        return self.q3 - self.q1


def describe(values: Sequence[float]) -> Summary:
    """Summary statistics for one sample, and whether its mean is worth printing (§3.1).

    Count, mean, median, standard deviation, variance, min/max, quartiles and skewness, plus the
    judgement the book asks for: the mean is "what you'd expect to see if you were to measure the
    height of a random person" only when the sample distributes like height. Wealth does not, and
    neither do most quantities in a network -- degrees (§9.3) and edge weights (§27.1) least of
    all.

    **The rule used here, stated so it can be argued with.** The sample is called heavy-tailed
    when the 99th percentile sits at least :data:`HEAVY_TAIL_RATIO` interquartile ranges above
    the median. A normal sample puts it at about :data:`NORMAL_TAIL_RATIO` whatever its mean and
    variance, a uniform sample at 1.0, a Poisson sample between about 1.5 and 2.5 across its
    rate, an exponential sample at about 3.6; a lognormal sample lands near 6 and a Pareto
    sample between 7 and 20. The measure
    is built from order statistics alone, deliberately: a criterion for "the moments are not
    trustworthy" cannot itself be a moment. When the middle half of the sample is a single value
    the ratio is infinite, and the sample counts as heavy-tailed if anything at all sits above
    that value -- which is the case a degree sequence of mostly-ones is in.

    When the flag is set, ``caveat`` says in words that the mean and the standard deviation are
    still returned but are not the typical case and do not settle down as the sample grows, and
    that no threshold may be motivated as "x standard deviations from the average" (p. 383).
    Below :data:`MIN_TAIL_SAMPLE` values the flag is left false and ``caveat`` says the tail is
    not measurable at this n rather than that it is light.

    ``skew`` is the Fisher-Pearson moment coefficient: 0 for a symmetric sample, positive when
    the long side is to the right. It is itself a third moment, so on a genuinely fat-tailed
    sample it is as undefined as the variance, and it is reported for the light-tailed case and
    as an order-of-magnitude hint otherwise. It is 0.0 when every value is identical.

    Undefined for an empty sample, which raises ``ValueError``: there is no typical value of
    nothing.
    """
    if not values:
        msg = "describe() needs at least one value; an empty sample has no summary"
        raise ValueError(msg)
    sample = np.asarray(values, dtype=np.float64)
    q1, median, q3, p99 = (float(v) for v in np.percentile(sample, [25, 50, 75, 99]))
    deviation = float(np.std(sample))
    iqr = q3 - q1
    ratio = (p99 - median) / iqr if iqr > 0 else (math.inf if p99 > median else 0.0)
    heavy = len(sample) >= MIN_TAIL_SAMPLE and ratio >= HEAVY_TAIL_RATIO
    average = float(np.mean(sample))
    return Summary(
        count=len(sample),
        mean=average,
        median=median,
        std=deviation,
        variance=float(np.var(sample)),
        minimum=float(np.min(sample)),
        q1=q1,
        q3=q3,
        maximum=float(np.max(sample)),
        skew=float(scipy_stats.skew(sample)) if deviation > 0 else 0.0,
        tail_ratio=ratio,
        heavy_tailed=heavy,
        caveat=_tail_caveat(len(sample), heavy, ratio, average, deviation, median),
    )


def _tail_caveat(
    count: int, heavy: bool, ratio: float, average: float, deviation: float, median: float
) -> str:
    """The sentence a report prints next to a mean it should not have trusted."""
    if count < MIN_TAIL_SAMPLE:
        return (
            f"{count} value(s): too few to tell a heavy tail from a light one, so the mean "
            f"({average:.4g}) and standard deviation ({deviation:.4g}) below are provisional "
            f"rather than light-tailed (§3.1 needs at least {MIN_TAIL_SAMPLE})."
        )
    if not heavy:
        return ""
    where = (
        "the middle half of the sample is a single value"
        if math.isinf(ratio)
        else (
            f"the 99th percentile sits {ratio:.1f} interquartile ranges above the median, where a "
            f"normal sample puts it at about {NORMAL_TAIL_RATIO}"
        )
    )
    return (
        f"Heavy-tailed sample (§3.1): {where}. The mean ({average:.4g}) and standard deviation "
        f"({deviation:.4g}) are reported because they were asked for, but they are not the "
        "typical case and they do not settle down as n grows -- such a distribution 'lacks of a "
        "well-defined average value and has undefined variance' (p. 383), so nothing here may "
        "be motivated as 'x standard deviations from the average'. Quote the median "
        f"({median:.4g}) and the quartiles instead."
    )


# --------------------------------------------------------------------------- §3.2 distributions


@dataclass(frozen=True)
class FitResult:
    """One distribution fitted to one sample, with the goodness-of-fit test beside it."""

    name: str
    params: dict[str, float] = field(default_factory=dict)
    #: The Kolmogorov-Smirnov statistic: the largest gap between the sample's empirical
    #: cumulative distribution and the fitted one. 0 is a perfect fit; it is a distance, so
    #: smaller is better and it can be compared across the fits of one sample.
    ks_statistic: float = 0.0
    #: The KS p-value under the null "the sample was drawn from this fitted distribution". Low
    #: means the shape is wrong. High does **not** mean the shape is right (§3.3), and this one
    #: is optimistic besides: see :func:`fit_distributions`.
    p_value: float = 0.0
    log_likelihood: float = 0.0


def fit_distributions(values: Sequence[float]) -> dict[str, FitResult]:
    """Fit the continuous distributions of §3.2 to one sample and test each fit.

    Returns one :class:`FitResult` per distribution in :data:`DISTRIBUTIONS` that is defined on
    this sample, keyed by name, in that order. A distribution whose support does not contain the
    data is left out rather than fitted to nonsense: the lognormal and the power law need every
    value strictly positive, the exponential needs every value non-negative. So an empty result
    for "powerlaw" means "this sample has a value at or below zero", not "the fit was poor".

    What is fitted, and with what:

    ``normal``
        maximum likelihood over mean and standard deviation. Params ``mu``, ``sigma``.
    ``exponential``
        the memoryless distribution of §3.2, anchored at zero. Param ``rate`` (1/scale).
    ``lognormal``
        the distribution of a variable whose logarithm is normal, anchored at zero. Params
        ``mu`` and ``sigma``, both in log space.
    ``powerlaw``
        p(x) ∝ x^-alpha for x >= xmin, with xmin taken as the sample minimum and alpha from the
        closed-form maximum-likelihood estimator 1 + n / Σ ln(x/xmin). Params ``alpha``, ``xmin``.

    Two limits worth stating wherever this is printed. First, the KS p-values are optimistic,
    because the parameters were estimated from the very sample being tested; the honest version
    resamples from the fitted distribution and refits each time, which
    :func:`graphrag.sna.degree.fit_power_law` does for degree sequences (§9.4), together with
    the ``xmin`` sweep and the discrete maximum-likelihood exponent this function does not
    attempt -- ``xmin`` here is simply the sample minimum. Second, the book is explicit that a
    power law and a lognormal are "very tricky to tell apart"; comparing their KS statistics
    here is a hint, not a verdict, and the verdict needs a likelihood-ratio test between the
    two, which is :func:`likelihood_ratio` below and which
    :func:`graphrag.sna.degree.compare_tails` runs over a fitted tail.

    The discrete distributions of §3.2 -- uniform, binomial, Poisson, hypergeometric -- are not
    fitted here. Binomial and hypergeometric are not shapes one fits to a sample of measurements
    but models of an extraction whose parameters the caller already knows; they appear in this
    module as :func:`binomial_p` and in the backboning chapter instead.

    Undefined for fewer than :data:`MIN_FIT_SAMPLE` values, which raises ``ValueError``.
    """
    if len(values) < MIN_FIT_SAMPLE:
        msg = (
            f"fit_distributions() needs at least {MIN_FIT_SAMPLE} values to fit anything "
            f"meaningful, got {len(values)}"
        )
        raise ValueError(msg)
    sample = np.asarray(values, dtype=np.float64)
    smallest = float(np.min(sample))
    fits: dict[str, FitResult] = {}
    mu, sigma = (float(p) for p in scipy_stats.norm.fit(sample))
    if sigma > 0:
        fits["normal"] = _fit(
            sample, "normal", {"mu": mu, "sigma": sigma}, scipy_stats.norm(mu, sigma)
        )
    if smallest >= 0:
        scale = float(np.mean(sample))
        if scale > 0:
            fits["exponential"] = _fit(
                sample,
                "exponential",
                {"rate": 1.0 / scale},
                scipy_stats.expon(loc=0.0, scale=scale),
            )
    if smallest > 0:
        shape, _, log_scale = scipy_stats.lognorm.fit(sample, floc=0.0)
        if float(shape) > 0:
            fits["lognormal"] = _fit(
                sample,
                "lognormal",
                {"mu": math.log(float(log_scale)), "sigma": float(shape)},
                scipy_stats.lognorm(float(shape), loc=0.0, scale=float(log_scale)),
            )
        spread = float(np.sum(np.log(sample / smallest)))
        if spread > 0:
            alpha = 1.0 + len(sample) / spread
            fits["powerlaw"] = _fit(
                sample,
                "powerlaw",
                {"alpha": alpha, "xmin": smallest},
                scipy_stats.pareto(alpha - 1.0, loc=0.0, scale=smallest),
            )
    return {name: fits[name] for name in DISTRIBUTIONS if name in fits}


def _fit(sample: np.ndarray, name: str, params: dict[str, float], frozen: Any) -> FitResult:
    """One fitted distribution, KS-tested against the sample it was fitted to.

    ``frozen`` is a scipy distribution with its parameters already bound, so the cumulative
    function the KS test needs and the log-density the likelihood needs come from one object and
    cannot drift apart.
    """
    test = scipy_stats.kstest(sample, frozen.cdf)
    return FitResult(
        name=name,
        params=params,
        ks_statistic=float(test.statistic),
        p_value=float(test.pvalue),
        log_likelihood=float(np.sum(frozen.logpdf(sample))),
    )


def likelihood_ratio(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float]:
    """Which of two fitted models explains a sample better, and whether the answer is noise.

    Takes the **per-observation log-likelihoods** of two models fitted to the same sample -- not
    their totals, because the test is about how consistently one model wins, point by point --
    and returns ``(ratio, statistic, p_value)``:

    ``ratio``
        the summed log-likelihood of ``first`` minus that of ``second``. Positive favours the
        first model. On its own it is unreadable: any two models differ by *something*.
    ``statistic``
        that ratio normalised by its own spread, ``R / (sqrt(n) * sd)``, which is Vuong's test.
        This is the number to quote: it says how large the difference is compared with how much
        it varies across the sample.
    ``p_value``
        two-sided, from the standard normal the statistic follows when the two models are
        equally far from the truth. **A large p-value means the test cannot tell them apart**,
        which is not evidence that they are equally good and is certainly not evidence for
        either (§3.3: a p-value is not an effect size).

    The models must be fitted to the same observations in the same order, and neither may be
    nested inside the other -- the normalisation assumes two non-nested candidates, which is the
    case §9.4 needs it for: a power law against a lognormal, or against an exponential, on one
    tail. Both models having estimated their parameters from this sample is expected and is why
    the test is a comparison rather than a goodness of fit; :func:`fit_distributions` is where
    the goodness of fit lives, and :func:`graphrag.sna.degree.fit_power_law` is where the honest
    version of it lives for degrees.

    ``statistic`` is 0.0 and ``p_value`` 1.0 when the two log-likelihood vectors are identical
    point for point, and also when their difference is constant: there is nothing to normalise
    by, so nothing can be preferred, however large ``ratio`` is. Raises ``ValueError`` on
    vectors of different lengths or on an empty sample.
    """
    if not first or not second:
        msg = "likelihood_ratio() needs at least one observation; an empty sample compares nothing"
        raise ValueError(msg)
    if len(first) != len(second):
        msg = (
            "likelihood_ratio() needs the two models' log-likelihoods over the same "
            f"observations, got {len(first)} and {len(second)}"
        )
        raise ValueError(msg)
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    ratio = float(np.sum(difference))
    spread = float(np.std(difference))
    # A difference that is constant to floating-point precision has nothing to normalise by:
    # one model beats the other by the same amount at every point, which is a preference nobody
    # can size against its own variation. The tolerance is relative, because "no spread" on a
    # per-point difference of -2.3 is not the same number as on one of -2.3e6.
    if spread <= 1e-12 * max(1.0, abs(float(np.mean(difference)))):
        return (ratio, 0.0, 1.0)
    statistic = ratio / (math.sqrt(difference.size) * spread)
    return (ratio, statistic, math.erfc(abs(statistic) / math.sqrt(2.0)))


# --------------------------------------------------------------------------- §3.3 p-values


def empirical_p(observed: float, null_samples: Sequence[float], tail: str = "right") -> float:
    """How unusual ``observed`` is against a null distribution you generated (§3.3).

    This is the p-value of Figure 3.8, computed rather than looked up: you have a bag of values
    the null hypothesis produced -- rewired graphs, shuffled labels, resampled edges -- and you
    ask how often the null did at least as well as the thing you measured.

    **Convention.** The count is ``(1 + hits) / (1 + n)``, adding the observation to its own null
    sample. An observation more extreme than every one of n null samples therefore scores
    ``1/(n+1)`` and never 0: a test with 200 samples cannot say anything finer than p = 0.005,
    and reporting p = 0 for it would claim a precision the sampling does not have. The corollary
    is that ``n`` belongs next to the p-value in any report.

    ``tail`` is ``right`` (the null rarely goes this high), ``left`` (this low) or ``two``
    (either), where the two-tailed value is twice the smaller one-tailed value, capped at 1.0.

    Undefined for an empty null sample, which raises ``ValueError``: with nothing to compare
    against there is no p-value, and a report should say the null could not be built.
    """
    if tail not in TAILS:
        msg = f"tail must be one of {', '.join(TAILS)}, got {tail!r}"
        raise ValueError(msg)
    if not null_samples:
        msg = "empirical_p() needs at least one null sample; with none there is no p-value"
        raise ValueError(msg)
    samples = np.asarray(null_samples, dtype=np.float64)
    total = len(samples) + 1
    right = (1 + int(np.sum(samples >= observed))) / total
    left = (1 + int(np.sum(samples <= observed))) / total
    if tail == "right":
        return right
    if tail == "left":
        return left
    return min(1.0, 2.0 * min(right, left))


def permutation_p(
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    a: Sequence[float],
    b: Sequence[float],
    permutations: int = 1000,
    seed: int | None = None,
    *,
    tail: str = "two",
    paired: bool = False,
) -> float:
    """A p-value for a two-sample statistic, from shuffling rather than from a table (§3.3).

    ``statistic`` is any function of the two samples -- a difference of means, a correlation, an
    assortativity -- and the null is built by destroying the structure the statistic claims to
    measure, then recomputing it ``permutations`` times.

    Which structure gets destroyed is the ``paired`` flag, and the two modes answer different
    questions:

    ``paired=False`` (default)
        the two samples are pooled and re-split into groups of the original sizes. The null is
        "these two groups are one population", so this is the test for a difference *between*
        groups. The samples may differ in length.
    ``paired=True``
        ``b`` is shuffled against a fixed ``a``, so the null is "the pairing carries no
        information" and this is the test for an association *within* pairs. The samples must
        be the same length.

    The finest p-value obtainable is ``1/(permutations + 1)`` (see :func:`empirical_p`), which
    is why the default is 1000 rather than the 200 an assortativity null gets away with: a
    corrected p-value from :func:`correct` over twenty tests needs headroom below 0.05.

    ``tail`` defaults to ``two`` because a permutation test is usually run without having
    committed to a direction; pass ``right`` when the hypothesis genuinely is "larger than
    chance".

    Undefined when either sample is empty, or when ``paired`` is set and the lengths differ;
    both raise ``ValueError``.
    """
    if not a or not b:
        msg = "permutation_p() needs both samples to be non-empty"
        raise ValueError(msg)
    if paired and len(a) != len(b):
        msg = f"a paired permutation needs equal lengths, got {len(a)} and {len(b)}"
        raise ValueError(msg)
    observed = float(statistic(a, b))
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    null: list[float] = []
    if paired:
        shuffled = list(b)
        for _ in range(max(permutations, 0)):
            rng.shuffle(shuffled)
            null.append(float(statistic(a, shuffled)))
    else:
        pooled = [*a, *b]
        split = len(a)
        for _ in range(max(permutations, 0)):
            rng.shuffle(pooled)
            null.append(float(statistic(pooled[:split], pooled[split:])))
    return empirical_p(observed, null, tail)


def binomial_p(successes: int, trials: int, p: float = 0.5, tail: str = "right") -> float:
    """The binomial p-value of §3.2's urn game, used as the test of §3.3.

    ``trials`` draws each succeed with probability ``p``, independently and with replacement;
    this is the probability of a result at least as extreme as ``successes``. It is the test
    behind "did this edge appear more often than chance", which is why §27.5's noise-corrected
    backbone and the hypergeometric of §3.2 are its neighbours in the book.

    ``tail`` is ``right`` by default and means P(X >= successes), the question a network
    scientist almost always has. ``left`` is P(X <= successes). ``two`` is the exact two-sided
    binomial test, which sums both tails by probability rather than by distance and is therefore
    not simply twice the smaller one.

    Undefined when ``trials`` is zero, when ``successes`` is outside ``0..trials``, or when ``p``
    is outside ``0..1``; all raise ``ValueError``.
    """
    if tail not in TAILS:
        msg = f"tail must be one of {', '.join(TAILS)}, got {tail!r}"
        raise ValueError(msg)
    if trials <= 0:
        msg = f"binomial_p() needs at least one trial, got {trials}"
        raise ValueError(msg)
    if not 0 <= successes <= trials:
        msg = f"successes must be between 0 and {trials}, got {successes}"
        raise ValueError(msg)
    if not 0.0 <= p <= 1.0:
        msg = f"p must be a probability between 0 and 1, got {p}"
        raise ValueError(msg)
    if tail == "right":
        return float(scipy_stats.binom.sf(successes - 1, trials, p))
    if tail == "left":
        return float(scipy_stats.binom.cdf(successes, trials, p))
    return float(scipy_stats.binomtest(successes, trials, p).pvalue)


def correct(p_values: Sequence[float], method: str = "bonferroni") -> list[float]:
    """Adjust a family of p-values for having asked more than one question (§3.3).

    The book's arithmetic: run 100 tests at the p < 0.01 standard and the probability that at
    least one clears it by accident is 1 - 0.99^100, about 63%. Every place in this package that
    tests many things at once -- every edge of a backbone, every node's role, every attribute of
    a corpus -- produces a family and has to come through here before any of it is called
    significant.

    Returns adjusted p-values in the input order, each capped at 1.0, so the caller keeps
    comparing against their original threshold rather than juggling two.

    ``bonferroni``
        multiply each by the number of tests (§3.3, Bonferroni 1936 / Dunn 1961). Controls the
        probability of *any* false rejection and is the most conservative of the three; with
        hundreds of tests it rejects almost nothing.
    ``holm``
        the Holm-Bonferroni step-down (1979): sort ascending and scale the i-th by ``m - i``,
        then enforce monotonicity. Controls the same error rate as Bonferroni and is uniformly
        more powerful, so prefer it wherever Bonferroni was the reflex.
    ``fdr_bh``
        Benjamini-Hochberg. Controls the *expected share of false discoveries* among the
        rejections rather than the chance of any, which is the right target when the output is a
        ranked list of candidates to look at (link prediction, backbone edges) rather than a
        single claim. The book does not name it; it is included because with a network's worth
        of tests the other two leave nothing.

    An empty family returns an empty list. A p-value outside ``0..1`` raises ``ValueError``,
    because it is a bug upstream and silently clipping it would hide the bug.
    """
    if method not in CORRECTIONS:
        msg = f"method must be one of {', '.join(CORRECTIONS)}, got {method!r}"
        raise ValueError(msg)
    if not p_values:
        return []
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        msg = f"p-values must lie between 0 and 1, got {list(p_values)}"
        raise ValueError(msg)
    m = len(p_values)
    if method == "bonferroni":
        return [min(1.0, p * m) for p in p_values]
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    if method == "holm":
        # Step down: each p is scaled by how many tests were still in play when it came up, and
        # then held at least as high as the one before it, so the sequence never decreases.
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (m - rank) * p_values[index])
            adjusted[index] = min(1.0, running)
        return adjusted
    # fdr_bh steps *up*, from the largest p down, so each is held no higher than the next one.
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, m * p_values[index] / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------- §3.4 correlation


@dataclass(frozen=True)
class Correlation:
    """One correlation coefficient, its p-value, and how many pairs went into both."""

    method: str
    #: Between -1 (perfect anticorrelation) and +1 (perfect correlation), 0 for no relationship
    #: *of the kind this coefficient measures*.
    coefficient: float
    #: Under the null "the two variables are independent". A p-value, with everything §3.3 says
    #: about p-values: it is not the strength of the relationship, which is the coefficient.
    p_value: float
    n: int


def pearson(x: Sequence[float], y: Sequence[float]) -> Correlation:
    """Pearson's correlation: the covariance of two variables, normalised by their spreads (§3.4).

    +1 when the two rise together on a straight line, -1 when one falls as the other rises, 0
    when neither. Normalising is what makes it readable: a covariance of 5.5 means nothing
    without units, and changes if you measure height in metres instead of centimetres, while the
    correlation does not.

    Two failure modes the book spells out, and they are failures of the question rather than of
    the arithmetic. Pearson returns zero for a U-shaped or A-shaped relationship however obvious
    the relationship is, because it is not monotone. And for a monotone but curved relationship
    it understates: log-transform one or both variables first, or use :func:`spearman`.

    Undefined when there are fewer than :data:`MIN_CORRELATION_SAMPLE` pairs, when the two
    sequences differ in length, or when either variable is constant -- a variable that does not
    vary cannot covary. All raise ``ValueError`` rather than returning ``nan``.
    """
    left, right = _paired(x, y, "pearson")
    result = scipy_stats.pearsonr(left, right)
    return Correlation("pearson", float(result.statistic), float(result.pvalue), len(left))


def spearman(x: Sequence[float], y: Sequence[float]) -> Correlation:
    """Spearman's rank correlation: Pearson computed on the ranks instead of the values (§3.4).

    Use it when the relationship is monotone but not linear, or when outliers would drag a
    Pearson coefficient around. Because only the order of the values matters, the coefficient is
    unchanged by any monotone transform of either variable -- taking logs, squaring positives,
    rescaling -- which is exactly the robustness §3.4 recommends it for. It still cannot see a
    U shape, since that is not monotone either.

    Ties are handled by averaging the tied ranks, so a variable with heavy ties (a degree
    sequence of mostly ones) has less to work with than its n suggests.

    Undefined in the same cases as :func:`pearson`, and raises the same way.
    """
    left, right = _paired(x, y, "spearman")
    result = scipy_stats.spearmanr(left, right)
    return Correlation("spearman", float(result.statistic), float(result.pvalue), len(left))


def kendall(x: Sequence[float], y: Sequence[float]) -> Correlation:
    """Kendall's tau-b: the share of pairs that agree on order, minus the share that disagree.

    A rank correlation like :func:`spearman` and an alternative to it (§3.4). Where Spearman is
    a Pearson coefficient of ranks, tau counts concordant against discordant pairs directly, so
    it has a plain reading -- "pick two observations at random; how much more often do they
    agree on direction than disagree" -- and it is less moved by a single badly-ranked point.
    Its values run smaller than Spearman's on the same data, so never compare the two numbers as
    if they were the same scale. The tau-b variant is used, which corrects for ties.

    Undefined in the same cases as :func:`pearson`, and raises the same way.
    """
    left, right = _paired(x, y, "kendall")
    result = scipy_stats.kendalltau(left, right)
    return Correlation("kendall", float(result.statistic), float(result.pvalue), len(left))


def _paired(x: Sequence[float], y: Sequence[float], method: str) -> tuple[np.ndarray, np.ndarray]:
    """The two variables as arrays, having refused the cases where a coefficient is undefined."""
    if len(x) != len(y):
        msg = f"{method}() needs paired values, got {len(x)} and {len(y)}"
        raise ValueError(msg)
    if len(x) < MIN_CORRELATION_SAMPLE:
        msg = (
            f"{method}() needs at least {MIN_CORRELATION_SAMPLE} pairs, got {len(x)}: with two "
            "points every monotone relationship is perfect"
        )
        raise ValueError(msg)
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    for name, column in (("x", left), ("y", right)):
        if float(np.ptp(column)) == 0.0:
            msg = f"{method}() is undefined: {name} is constant, so it cannot covary with anything"
            raise ValueError(msg)
    return left, right


# --------------------------------------------------------------------------- §3.5 information


def entropy(labels: Sequence[Hashable]) -> float:
    """Shannon's information entropy of a labelling, in bits (§3.5).

    How many bits it takes on average to encode which value each element holds:
    ``H = -Σ p_i log2(p_i)``. A fair coin is 1 bit, four equally likely outcomes are 2 bits, and
    a sequence with only one distinct value is 0 bits, because nothing about it has to be
    transmitted.

    Undefined for an empty sequence, which raises ``ValueError``.
    """
    if not labels:
        msg = "entropy() needs at least one label"
        raise ValueError(msg)
    total = len(labels)
    return -sum(
        (count / total) * math.log2(count / total) for count in Counter(labels).values() if count
    )


def mutual_information(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """How many bits knowing one labelling saves you about the other (§3.5).

    ``MI = Σ p_ij log(p_ij / (p_i p_j))`` over the joint distribution of the two labellings of
    the same elements. The meat of it, as the book puts it, is the comparison of the joint
    probability with what independence would give: when the two are independent every term is
    ``log(1) = 0`` and the mutual information is 0 bits. When one determines the other it equals
    that one's entropy -- knowing x saves you all of y.

    Reported in bits (base-2 logarithms), as §3.5 counts. It is symmetric, never negative, and
    never larger than the smaller of the two entropies.

    It is *not* normalised, so it grows with the number of distinct labels and two mutual
    informations over different labellings are not comparable. Use
    :func:`normalized_mutual_information` for that, and read the caution there.

    Undefined for empty sequences or sequences of different length; both raise ``ValueError``.
    """
    _same_length(a, b, "mutual_information")
    return float(mutual_info_score(list(a), list(b))) / _LOG2


def normalized_mutual_information(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Mutual information divided by the mean of the two entropies: 0 to 1 (§3.5).

    1.0 when the two labellings are the same partition (whatever the labels are called, since
    this measures the partition and not the names), 0.0 when knowing one tells you nothing about
    the other. Being a ratio, it is the same in bits or nats.

    The caution that must travel with it: NMI is **not corrected for chance**. Two random
    labellings score above zero, and the more clusters they have the higher they score, so an
    NMI of 0.4 over 30 clusters can be less agreement than an NMI of 0.2 over 3. Quote it beside
    a chance-corrected index -- :func:`compare_partitions` returns the adjusted Rand index with
    it for exactly this reason -- or beside a baseline from shuffling one of the labellings.

    Undefined for empty or unequal-length sequences, which raise ``ValueError``. When one
    labelling has a single value and the other does not, there is no shared information and the
    value is 0.0; when both do, it is 1.0 by convention, which says more about the convention
    than about the data.
    """
    _same_length(a, b, "normalized_mutual_information")
    return float(normalized_mutual_info_score(list(a), list(b)))


def variation_of_information(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """The information-theoretic distance between two labellings, in bits (§3.5).

    ``VI = H(a) + H(b) - 2 MI(a, b)``: the information in one that is not in the other, plus the
    information in the other that is not in the one. 0.0 when they are the same partition, and
    larger the further apart they are. Unlike :func:`normalized_mutual_information` it is a true
    metric -- it obeys the triangle inequality -- which is what makes it the right thing to
    average or to cluster over when comparing many partitions rather than two.

    It is unnormalised and measured in bits, so it is bounded above by ``log2(n)`` and a VI of
    1.5 bits means different things over 10 nodes and over 10,000. Print n beside it.

    Undefined for empty or unequal-length sequences, which raise ``ValueError``.
    """
    _same_length(a, b, "variation_of_information")
    distance = entropy(a) + entropy(b) - 2.0 * mutual_information(a, b)
    return max(0.0, distance)  # the identity is exact; floating point is not


def _same_length(a: Sequence[Hashable], b: Sequence[Hashable], caller: str) -> None:
    """Both labellings describe the same elements, or there is nothing to compare."""
    if not a or not b:
        msg = f"{caller}() needs two non-empty labellings"
        raise ValueError(msg)
    if len(a) != len(b):
        msg = f"{caller}() needs two labellings of the same elements, got {len(a)} and {len(b)}"
        raise ValueError(msg)


def compare_partitions(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """Agreement between two labellings of the same nodes (§3.5).

    The adjusted Rand index is corrected for chance: 0 is what random labellings score, 1 is
    identical. Normalised mutual information is not chance-corrected and drifts upward as the
    number of clusters grows, so quote both.
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
        "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
    }


# --------------------------------------------------------------------------- null-sample helpers


def mean(values: Sequence[float]) -> float:
    """The arithmetic mean, or 0.0 for an empty sample.

    The report-safe companion to :func:`describe`, for the null distributions that feed a
    ``float`` field printed whether or not the null could be built: a graph too small to rewire
    yields no samples, and the report says so with its sample count rather than with a ``nan``.
    Where the question is "what does this sample look like", use :func:`describe` instead, which
    refuses an empty sample and says whether the mean is the typical case (§3.1).
    """
    return float(np.mean(values)) if values else 0.0


def std(values: Sequence[float]) -> float:
    """The population standard deviation (divisor n), or 0.0 for an empty sample.

    Divisor n rather than n-1 because these samples are null distributions generated to a
    requested size, not estimates of a wider population's spread. Same empty-sample convention
    as :func:`mean`, and the same §3.1 caveat applies: on a heavy-tailed sample this number does
    not settle down, so :func:`describe` is what decides whether it can be quoted.
    """
    return float(np.std(values)) if values else 0.0


def z_score(observed: float, null_samples: Sequence[float]) -> float:
    """How many null standard deviations ``observed`` sits above the null mean (§3.1, §3.3).

    The z-score is the companion of :func:`empirical_p` and answers the same question in a
    different currency: the p-value counts how often the null did this well, the z-score says
    how far out it is in units of the null's own spread. 2 is suggestive and 3 is solid, by the
    convention this package uses throughout.

    It is 0.0 when the null sample is empty or has no spread, so that a report can print it
    unconditionally; the sample count printed beside it is what tells a reader which case they
    are in. And it assumes the null distribution is roughly symmetric: when the null is itself
    heavy-tailed (:func:`describe` will say so), the z-score overstates and the empirical
    p-value is the one to quote.
    """
    deviation = std(null_samples)
    return (observed - mean(null_samples)) / deviation if deviation > 0 else 0.0
