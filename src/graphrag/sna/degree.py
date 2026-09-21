"""Degree, its distribution, and whether that distribution is a power law (Atlas ch. 9).

The chapter is short on formulas and long on discipline, and the discipline is the point of this
module. *"Just because something looks like a straight line in a log-log plot, it doesn't mean
it's a power law"* (§9.4). So nothing here reads a slope off a plot, and nothing here calls a
network scale free because its degrees are unequal.

What the chapter asks for, and where it lives:

§9.1 *Degree variants*
    :func:`degree_sequence`. A degree is the number of connections of a node; on a directed
    network it splits into the in-degree (arrow heads) and the out-degree (arrow tails), and on
    a weighted network the sum of the incident weights is the *node strength*, which the book
    keeps as a separate quantity rather than as a replacement -- *"one can have a node with
    enormous strength but low degree"*. A bipartite network has two degree sequences, one per
    mode, so ``mode`` selects one of them instead of mixing them.

§9.2 *Degree distributions*
    :func:`degree_distribution`. The scatter plot, the log-binned histogram and the CCDF, which
    is the one to read: *"the most common way to visualize degrees is by drawing cumulative
    distributions"*, because it is a function rather than a scattergram and *"helps us when we
    need to fit it"*. The log binning is there because equal-size bins lose the head of the
    distribution and keep the fat tail; it is offered with the book's own warning that choosing
    a bin size *"opens you to the possibility of tricking yourself into seeing a pattern that is
    not there"*.

§9.3 *Power laws*
    p(k) ~ k^-alpha, with alpha near 2 in most real networks and below 3 in the majority --
    which is exactly the regime where *"the degree distribution has a well defined mean, but not
    a well defined variance"*. :class:`PowerLawFit` therefore carries its own reminder that the
    mean degree of such a network is not the typical node.

§9.4 *Testing power laws*
    :func:`fit_power_law` and :func:`compare_tails`, which are the Clauset-Shalizi-Newman
    procedure the book points at: the exponent by maximum likelihood rather than by regression
    on the log-log plane, ``xmin`` chosen as the value that minimises the Kolmogorov-Smirnov
    distance, a goodness-of-fit p-value from a parametric bootstrap, and likelihood-ratio tests
    against the lognormal and the exponential over the same tail. The book's order of business
    is kept: first rule out the exponential -- *"if you think a distribution might be an
    exponential, then it's definitely not a power law"* -- then try to prefer the power law over
    the lognormal, and expect to fail, because *"having a significant difference between the
    power law and the lognormal model is extremely hard"*.

:func:`degree_report` puts them together and refuses the word. Its verdict says "scale-free"
only when the bootstrap does not reject the power law *and* the lognormal is not significantly
better; otherwise it names what the data support, which is usually the chapter's own closing
position: a broad, unequal degree distribution is the interesting finding, and it is *"not
crucial empirically"* whether a power law generated it.

What this module does **not** do, and where the rest of §9.1 lives.

*Shifted and truncated power laws* (§9.4) are not fitted as models of their own; exercise 6 does
it with ``scipy.optimize.curve_fit``. ``xmin`` is where the shift is handled, by excluding the
head the law does not hold on, and an exponential cutoff would need a fourth model in the
comparison and a verdict branch of its own.

*The data generating process* is not decided here. The book is explicit that when the lognormal
cannot be excluded statistically you have to argue from cumulative advantage instead, and no
arrangement of numbers can make that argument for you.

*The multilayer degree* (§9.1) is chapter 7's object, not this one's. Interlayer couplings are
not part of a degree -- *"eight out of ten cats say no"* -- and the quantities the section
defines over layers, the per-layer degree and neighbour set, the exclusive neighbours
``N^XOR_{u,l}`` and the layer relevance ``|N_{u,l}| / |N_u|``, are all about a node's position
*within* a layer. :mod:`graphrag.sna.layers` holds the layered network (ATL-07), and its
per-layer measures and the multilayer community work belong with it (ATL-40). What this module
does do is refuse to be silent about it: a report built on a flattening says so, because every
degree in it counts an edge whichever layer it came from.

*The multigraph distinction* (§9.1) is reported rather than modelled: where parallel edges
exist, the degree counts connections and the neighbour set counts neighbours, so
``k_u >= |N_u|``. ``networkx`` counts parallel edges in ``degree`` and this module inherits
that; a report on a multigraph says which of the two it printed.

*The hypergraph degree* (§9.1) is deferred exactly where the book defers it, to §34.1: with one
hyperedge able to join the whole network, the neighbour count has no relationship to the
connection count and even ``2|E|/|V|`` stops meaning anything. §34.1 answers it by closing the
hypergraph into a simplicial complex first and generalising the degree over its faces, and that
is where it now lives: :func:`graphrag.sna.highorder.simplicial_degree` is ``k_(d,m)``, *"the
number of d dimensional simplices incident on an m-face"* (p. 475), with ``k_(2,0)`` -- how many
filled triangles a node sits in -- printed by ``graphrag sna highorder``.
:func:`graphrag.sna.layers.hyperedge_sizes` remains the answer to "how big are the hyperedges",
which is the distribution to read before any of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np
from scipy import optimize, special
from scipy import stats as scipy_stats

from graphrag.sna.export import describe
from graphrag.sna.matrices import node_order
from graphrag.sna.sampling import reweighted_degree_distribution, sample_record
from graphrag.sna.stats import FitResult, Summary, empirical_p, fit_distributions, likelihood_ratio
from graphrag.sna.stats import describe as describe_sample

__all__ = [
    "CHAPTER",
    "CSN_LEVEL",
    "DEFAULT_BOOTSTRAP",
    "DEGREE_KINDS",
    "KIND_MEANING",
    "KS_LEVEL",
    "LOG_BIN_FACTOR",
    "LR_LEVEL",
    "MIN_DEGREE_SAMPLE",
    "MIN_TAIL_SIZE",
    "MIN_TAIL_VALUES",
    "VERDICTS",
    "DegreeDistribution",
    "DegreeReport",
    "LogBin",
    "ModeDegrees",
    "PowerLawFit",
    "TailComparison",
    "compare_tails",
    "degree_distribution",
    "degree_payload",
    "degree_report",
    "degree_sequence",
    "fit_power_law",
    "render_degree",
]

#: The variants of §9.1. ``degree`` is the plain count of connections and is defined on every
#: network here; ``in``/``out`` need a directed one; the ``weighted`` three are node strengths,
#: the sum of the incident weights, which the book keeps beside the count rather than instead of
#: it.
DEGREE_KINDS: tuple[str, ...] = ("degree", "in", "out", "weighted", "weighted_in", "weighted_out")

#: What each variant answers, printed above the numbers so a reader knows which question the
#: distribution below is a distribution of.
KIND_MEANING: dict[str, str] = {
    "degree": "how many distinct others a node is connected to (§9.1).",
    "in": "how many arrow heads point at the node: how often it is named by others (§9.1).",
    "out": "how many arrow tails leave the node: how often it names others (§9.1).",
    "weighted": (
        "node strength: the total weight incident to a node, so repeated co-appearances count "
        "(§9.1). A node can carry enormous strength on few edges, or little on many."
    ),
    "weighted_in": "incoming node strength: the total weight of the edges that point at it.",
    "weighted_out": "outgoing node strength: the total weight of the edges it points along.",
}

#: Below this many nodes with a positive degree, no fit is attempted at all. It is the sample
#: size Clauset, Shalizi and Newman call the minimum for their estimator to be trustworthy, and
#: it is also the book's own point about truncated power laws: a system that is not big enough
#: cannot show scale-free behaviour, so measuring one is measuring its size.
MIN_DEGREE_SAMPLE = 50

#: And below this many nodes *in the tail* (degree >= xmin), a candidate ``xmin`` is not
#: considered: the same rule of thumb, applied to the part of the data the fit actually uses.
MIN_TAIL_SIZE = 50

#: And a candidate ``xmin`` whose tail carries fewer than this many *distinct* degree values is
#: not considered either. A regular graph -- every node of degree 59 -- otherwise fits a power
#: law perfectly at xmin = 59, because a single value is a single point and every curve goes
#: through it; §9.2's whole argument is that a degree distribution worth fitting spans orders of
#: magnitude. Four is the smallest tail on which the KS distance can distinguish two shapes at
#: all.
MIN_TAIL_VALUES = 4

#: Synthetic data sets behind the goodness-of-fit p-value. 200 resamples put the standard error
#: of a p near 0.1 at about 0.02, which is enough to place it against the thresholds below;
#: Clauset et al. use 2,500 when they want two decimal places, and ``--bootstrap`` raises it.
DEFAULT_BOOTSTRAP = 200

#: At or below this p-value the power law is rejected outright: the observed Kolmogorov-Smirnov
#: distance is so much larger than the fit's own samples produce that the shape is wrong. This
#: is the conventional level, and it is the *lower* of the two bars here; it decides which
#: negative verdict is printed, never whether the word "scale-free" may be.
KS_LEVEL = 0.05

#: **The bar the verdict uses.** Clauset, Shalizi and Newman's own rule, which is what §9.4
#: sends the reader to: a power law is a plausible hypothesis only when p > 0.1, and at or below
#: that it is ruled out. Between :data:`KS_LEVEL` and this the fit is not rejected but the
#: evidence is too weak to carry the word, and the verdict says exactly that rather than
#: rounding up to "scale-free". A textbook Barabasi-Albert network of 3,000 nodes lands in that
#: band, because its degrees follow 2m(m+1)/(k(k+1)(k+2)) -- a *shifted* power law whose head a
#: KS test over a large tail detects -- and reading it as "broad, exponential ruled out,
#: lognormal not excluded" is §9.4's closing position, not a failure of the test.
CSN_LEVEL = 0.10

#: The level at which a likelihood ratio between two fitted tails counts as a preference rather
#: than as noise. The book warns to expect noise: *"be prepared for the fact that having a
#: significant difference between the power law and the lognormal model is extremely hard"*.
LR_LEVEL = 0.05

#: How far the tabulated CCDF behind the bootstrap's sampler runs above ``xmin``. See
#: :func:`_power_law_sampler`.
SAMPLER_SPAN = 100_000

#: The growth factor of the log-binned histogram of §9.2: the first bin is one degree wide and
#: each next bin is 10% wider than the last, the book's own example.
LOG_BIN_FACTOR = 1.1

#: The search range for the exponent. Below 1 the discrete power law has no normalisation at
#: all; above 10 the tail is steeper than any power law a network has ever been reported with
#: (§9.3 puts real networks near 2, and the majority below 3), so an estimate that reaches the
#: ceiling is reported as "steeper than the search range" rather than as a measurement.
ALPHA_BOUNDS = (1.05, 10.0)

#: The chapter every number in this module implements, printed next to them.
CHAPTER = "Atlas ch. 9 (§9.1-§9.4), power-law test after Clauset, Shalizi and Newman (2009)"

#: The sentence that names the null model behind the goodness-of-fit p-value. A KS distance on
#: its own says nothing: the question is how large that distance is for data that really were
#: drawn from the fitted power law, which is what the bootstrap answers.
NULL_MODEL = (
    "{bootstrap} synthetic data sets of the same size, each drawn from the fitted power law "
    "above xmin and resampled from the observed values below it, each refitted from scratch "
    "(its own xmin and alpha); p is the share whose KS distance is at least the observed one."
)

#: Every verdict :func:`degree_report` can reach, so a caller can branch on one without matching
#: prose. The reading printed beside each is built per report.
VERDICTS: tuple[str, ...] = (
    "too few nodes to test",
    "too few nodes in the tail to test",
    "not counts, so the discrete test does not apply",
    "two modes, measured separately",
    "exponential/Poisson-like",
    "lognormal rather than a power law",
    "heavy-tailed but not distinguishable from lognormal",
    "heavy-tailed but not a power law",
    "heavy-tailed, power law not rejected but evidence weak",
    "neither a power law nor a heavy tail",
    "scale-free",
)

#: How many rows of the CCDF and of the log-binned histogram a rendered report prints. The full
#: curves are in the JSON payload; a markdown table of 400 degree values is not a report.
TABLE_ROWS = 20


# ------------------------------------------------------------------------- §9.1 degree variants


def degree_sequence(
    graph: nx.Graph, kind: str = "degree", *, mode: str | None = None
) -> list[int | float]:
    """The degree of every node, in one of the variants of §9.1, in node order.

    Returns one value per node, ordered by :func:`graphrag.sna.matrices.node_order` -- the
    sorted node ids, the same order every matrix in this package uses. The book asks for exactly
    that discipline where a network has two sequences: *"you can have two sequences, but you
    need to make sure that the nth positions of the two sequences refer to the same node"*, so
    ``degree_sequence(g, "in")[i]`` and ``degree_sequence(g, "out")[i]`` are the two halves of
    one node's degree.

    ``kind`` is one of :data:`DEGREE_KINDS`. ``in`` and ``out`` are defined only on a directed
    network and raise on an undirected one, where they are the same question. On a directed
    network ``degree`` is the total, in + out, which is what ``networkx`` calls the degree and
    what §9.1 calls "the number of connections of a node"; read the two halves separately when
    the direction is the question. The ``weighted`` variants are node strengths: the sum of the
    incident weights, falling back to 1.0 for an edge that carries none, so on an unweighted
    network they reduce to the count exactly as the book says they should.

    ``mode`` keeps only the nodes of one side of a bipartite network -- ``speaker`` or
    ``entity`` as the builders write it onto each node. §9.1: a bipartite network has two degree
    sequences, one per node type, and they are not comparable with each other; both sum to |E|.
    Mixing them into one distribution and fitting it describes neither side.

    Self-loops are counted the way ``networkx`` counts them, twice in an undirected degree,
    because the corpus networks have none and inventing a third convention here would only be a
    trap for a network read in from a file.
    """
    if kind not in DEGREE_KINDS:
        msg = f"kind must be one of {', '.join(DEGREE_KINDS)}, got {kind!r}"
        raise ValueError(msg)
    if kind in {"in", "out", "weighted_in", "weighted_out"} and not graph.is_directed():
        msg = (
            f"{kind!r} is defined only on a directed network; this one is undirected, where in "
            "and out are the same thing. Use 'degree' or 'weighted'."
        )
        raise ValueError(msg)
    nodes = [
        n for n in node_order(graph) if mode is None or str(graph.nodes[n].get("mode", "")) == mode
    ]
    weight = "weight" if kind.startswith("weighted") else None
    view: Any
    if kind in {"in", "weighted_in"}:
        view = graph.in_degree(nodes, weight=weight)
    elif kind in {"out", "weighted_out"}:
        view = graph.out_degree(nodes, weight=weight)
    else:
        view = graph.degree(nodes, weight=weight)
    values = dict(view)
    return [float(values[n]) if weight else int(values[n]) for n in nodes]


# --------------------------------------------------------------------- §9.2 degree distributions


@dataclass(frozen=True)
class LogBin:
    """One bin of the power-binned histogram of §9.2: ``[low, high)``, and what fell in it."""

    low: float
    high: float
    count: int
    #: The count divided by the sample size *and* by the bin's width, which is what makes bins
    #: of different widths comparable. Without it a wider bin looks more populated for being
    #: wider.
    density: float


@dataclass(frozen=True)
class DegreeDistribution:
    """A degree sequence as the three plots of §9.2, aligned on one list of degree values.

    ``values`` are the distinct degrees observed, ascending; ``pmf`` and ``ccdf`` are aligned
    with it position by position. ``ccdf[i]`` is p(K >= values[i]), so it starts at 1.0 and
    never rises -- the shape the chapter says to fit, rather than the scatter it says not to.
    """

    n: int
    values: tuple[float, ...]
    counts: tuple[int, ...]
    pmf: tuple[float, ...]
    ccdf: tuple[float, ...]
    bins: tuple[LogBin, ...]
    factor: float


def degree_distribution(
    sequence: Sequence[float], *, factor: float = LOG_BIN_FACTOR
) -> DegreeDistribution:
    """The pmf, the CCDF and the log-binned histogram of one degree sequence (§9.2).

    The pmf is the degree scatter plot with its y axis normalised by the number of nodes, *"the
    probability of a node to have a degree equal to k, not simply the node count"*, which is
    what makes two networks of different sizes comparable.

    The CCDF is p(K >= k). It is the one to read and the one to fit: a fat tail spreads the top
    of a degree sequence over hundreds of values carried by one node each, which makes the
    scatter plot unreadable exactly where the interesting nodes are, while the CCDF stays a
    function. Note that its exponent is not the distribution's: *"the CCDF of a power law is
    also a power law, but with a different exponent"*, one lower, which is why nothing in this
    module fits the CCDF -- :func:`fit_power_law` works from the values themselves.

    The log binning is the third option of §9.2, offered with its warning. The first bin is one
    degree wide and each next is ``factor`` times wider, so the head keeps its resolution while
    the tail is grouped into bins wide enough to hold something; ``density`` divides by the bin
    width so the bins remain comparable. The bin size is a *choice*, and the book is blunt about
    what a choice costs: it *"opens you to the possibility of tricking yourself into seeing a
    pattern that is not there"*. It is here to be looked at, not to be fitted.

    Undefined for an empty sequence, which raises ``ValueError``: there is no distribution of no
    nodes. A sequence of zeros is allowed and produces a single point at 0.
    """
    if len(sequence) == 0:
        msg = "degree_distribution() needs at least one node; an empty network has no degrees"
        raise ValueError(msg)
    if factor <= 1.0:
        msg = f"factor must be greater than 1.0 for the bins to grow, got {factor}"
        raise ValueError(msg)
    sample = np.asarray(sequence, dtype=np.float64)
    n = int(sample.size)
    values, counts = np.unique(sample, return_counts=True)
    pmf = counts / n
    # p(K >= k) at each observed value: the mass at this value and everything above it.
    ccdf = np.cumsum(pmf[::-1])[::-1]
    return DegreeDistribution(
        n=n,
        values=tuple(float(v) for v in values),
        counts=tuple(int(c) for c in counts),
        pmf=tuple(float(p) for p in pmf),
        ccdf=tuple(min(1.0, float(p)) for p in ccdf),
        bins=_log_bins(sample, factor),
        factor=factor,
    )


def _log_bins(sample: np.ndarray, factor: float) -> tuple[LogBin, ...]:
    """The power binning of §9.2: first bin one wide, each next ``factor`` times wider."""
    low, high = float(np.min(sample)), float(np.max(sample))
    edges = [low]
    width = 1.0
    while edges[-1] <= high:
        edges.append(edges[-1] + width)
        width *= factor
    n = int(sample.size)
    bins: list[LogBin] = []
    for index, (start, end) in enumerate(pairwise(edges)):
        # Half-open bins, so no node is counted twice; the last one closes so the largest
        # degree is inside a bin rather than beyond the final edge.
        inside = (sample >= start) & (sample < end)
        if index == len(edges) - 2:
            inside = (sample >= start) & (sample <= end)
        count = int(np.count_nonzero(inside))
        bins.append(LogBin(low=start, high=end, count=count, density=count / (n * (end - start))))
    return tuple(bins)


# ---------------------------------------------------------------------- §9.4 fitting and testing


@dataclass(frozen=True)
class PowerLawFit:
    """A discrete power law fitted to the tail of a degree sequence, and its own test (§9.4).

    ``alpha`` is the scaling factor of p(k) ~ k^-alpha, estimated by maximum likelihood over the
    values at or above ``xmin`` -- never by a regression on the log-log plane, which the chapter
    rules out in as many words. ``xmin`` is the value that minimises ``ks_statistic``: the head
    of a real degree distribution rarely follows the law (*"getting the first kmin connections
    is easy"*), so the fit says where the law starts holding instead of pretending it holds
    everywhere.

    ``p_value`` is the goodness of fit, not a significance: it is the share of synthetic data
    sets drawn from this very fit whose KS distance was at least ``ks_statistic``. Small means
    the data are a worse fit to the power law than the power law's own samples are, so the
    hypothesis is rejected. Large means it could not be rejected, which is not the same as true
    -- the lognormal usually cannot be rejected either, which is what :func:`compare_tails` is
    for.

    ``standard_error`` is the (alpha - 1)/sqrt(n_tail) of Clauset et al.: the uncertainty on the
    exponent from the tail size alone, before any question of whether a power law is the right
    shape.

    ``at_bound`` is set when the likelihood wanted an exponent outside :data:`ALPHA_BOUNDS`. It
    means the tail is steeper than any power law in the search range -- an exponential in
    disguise, usually -- and ``alpha`` is then a ceiling rather than an estimate.
    """

    alpha: float
    xmin: float
    n: int
    n_tail: int
    ks_statistic: float
    p_value: float
    bootstrap: int
    standard_error: float
    seed: int | None = None
    at_bound: bool = False

    @property
    def infinite_variance(self) -> bool:
        """Whether §9.3's warning applies: with alpha <= 3 the variance is undefined.

        *"This is rather unfortunate, because it means that the degree distribution has a well
        defined mean, but not a well defined variance."* Below 2 the mean goes as well.
        """
        return self.alpha <= 3.0

    @property
    def rejected(self) -> bool:
        """Whether the bootstrap rejects the power law outright, at :data:`KS_LEVEL`."""
        return self.p_value <= KS_LEVEL

    @property
    def plausible(self) -> bool:
        """Whether p clears :data:`CSN_LEVEL`, which is the bar the verdict's word rests on.

        Clauset, Shalizi and Newman rule a power law out at p <= 0.1 rather than at 0.05, and
        §9.4 sends the reader to them, so this -- not :attr:`rejected` -- is what
        :func:`degree_report` requires before printing "scale-free". A fit between the two
        levels is reported as a broad distribution whose power law could not be ruled out and
        could not be established either.
        """
        return self.p_value > CSN_LEVEL


def fit_power_law(
    sequence: Sequence[float],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int | None = None,
) -> PowerLawFit | None:
    """Fit a discrete power law to a degree sequence and test it, per §9.4.

    The procedure is Clauset, Shalizi and Newman's, which is the one the chapter sends you to
    (and which the ``powerlaw`` package it names implements):

    1. For every distinct value in the sequence that leaves at least :data:`MIN_TAIL_SIZE`
       nodes and :data:`MIN_TAIL_VALUES` distinct degrees at or above it, estimate ``alpha`` by
       maximising the discrete likelihood ``-n ln zeta(alpha, xmin) - alpha sum(ln k)`` over
       :data:`ALPHA_BOUNDS`. This is the
       *discrete* estimator, over the Hurwitz zeta function: degrees are counts, and the
       continuous estimator that closed-forms to ``1 + n / sum(ln(k/kmin))`` is only its
       approximation.
    2. Keep the ``xmin`` whose fit has the smallest Kolmogorov-Smirnov distance to the data at
       or above it.
    3. Compute the goodness of fit by a parametric bootstrap: ``bootstrap`` synthetic data sets
       of the same size, each one drawn from the fitted power law above ``xmin`` and resampled
       with replacement from the observed values below it, each refitted from scratch with its
       own ``xmin``. ``p_value`` is the share whose KS distance is at least the observed one.

    Returns ``None`` -- not a fit with a caveat -- when the sequence has fewer than
    :data:`MIN_DEGREE_SAMPLE` positive values, or when no candidate ``xmin`` leaves a tail of
    :data:`MIN_TAIL_SIZE` nodes over :data:`MIN_TAIL_VALUES` distinct degrees. All of those are
    the same refusal: the shape of a tail that thin is not a measurement, and the book's own
    reading of a network too small to show scale-free behaviour is that you are looking at its
    size.

    Zero and negative values are dropped before fitting: the discrete power law is defined for
    k >= 1, and an isolated node has no place in a distribution about how connections
    concentrate. The count of what was dropped is in the report, never silently.

    ``seed`` fixes the bootstrap, so the same sequence gives the same p-value twice. With the
    default 200 resamples that p carries a standard error of about 0.02 near 0.1, which is the
    precision the thresholds here are read at; raise ``bootstrap`` for a published number.

    Undefined for non-integral values, which raise ``ValueError``: this is the discrete
    estimator, and a weighted degree of 2.5 is not a count of anything. Fit such a sequence with
    :func:`graphrag.sna.stats.fit_distributions`, whose power law is the continuous one.
    """
    sample = np.asarray(sequence, dtype=np.float64)
    if sample.size and not np.all(np.isfinite(sample)):
        msg = "fit_power_law() needs finite values; got a nan or an infinity"
        raise ValueError(msg)
    if not np.all(sample == np.rint(sample)):
        msg = (
            "fit_power_law() is the discrete estimator of §9.4 and needs integer counts; this "
            "sequence has fractional values. Use stats.fit_distributions() for a continuous fit."
        )
        raise ValueError(msg)
    values = sample[sample > 0].astype(np.int64)
    if values.size < MIN_DEGREE_SAMPLE:
        return None
    best = _best_xmin(values)
    if best is None:
        return None
    ks, alpha, xmin, n_tail = best
    p_value = _bootstrap_p(values, ks, alpha, xmin, bootstrap, seed)
    return PowerLawFit(
        alpha=alpha,
        xmin=float(xmin),
        n=int(values.size),
        n_tail=n_tail,
        ks_statistic=ks,
        p_value=p_value,
        bootstrap=bootstrap,
        standard_error=(alpha - 1.0) / math.sqrt(n_tail),
        seed=seed,
        at_bound=alpha >= ALPHA_BOUNDS[1] - 1e-3,
    )


def _alpha_mle(tail: np.ndarray, xmin: int) -> float:
    """The discrete maximum-likelihood exponent for this tail and this ``xmin``.

    There is no closed form for the discrete case, so the negative log-likelihood
    ``n ln zeta(alpha, xmin) + alpha sum(ln k)`` is minimised numerically over
    :data:`ALPHA_BOUNDS`. ``zeta`` here is the Hurwitz zeta, the normalisation of the discrete
    power law on k >= xmin.
    """
    size = int(tail.size)
    logs = float(np.sum(np.log(tail)))

    def negative_log_likelihood(alpha: float) -> float:
        return size * math.log(float(special.zeta(alpha, xmin))) + alpha * logs

    result = optimize.minimize_scalar(
        negative_log_likelihood,
        bounds=ALPHA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def _ks_distance(tail: np.ndarray, xmin: int, alpha: float) -> float:
    """The largest gap between the tail's empirical CDF and the fitted power law's."""
    uniques, counts = np.unique(tail, return_counts=True)
    empirical = np.cumsum(counts) / tail.size
    normaliser = float(special.zeta(alpha, xmin))
    model = 1.0 - special.zeta(alpha, uniques + 1) / normaliser
    return float(np.max(np.abs(empirical - model)))


def _best_xmin(values: np.ndarray) -> tuple[float, float, int, int] | None:
    """Sweep the candidate ``xmin`` values and keep the one with the smallest KS distance."""
    best: tuple[float, float, int, int] | None = None
    for candidate in np.unique(values):
        tail = values[values >= candidate]
        if tail.size < MIN_TAIL_SIZE or np.unique(tail).size < MIN_TAIL_VALUES:
            continue
        xmin = int(candidate)
        alpha = _alpha_mle(tail, xmin)
        distance = _ks_distance(tail, xmin, alpha)
        if best is None or distance < best[0]:
            best = (distance, alpha, xmin, int(tail.size))
    return best


def _power_law_sampler(alpha: float, xmin: int) -> Callable[[np.random.Generator, int], np.ndarray]:
    """A draw function for the discrete power law with this exponent, by inverse transform.

    The CCDF ``zeta(alpha, k) / zeta(alpha, xmin)`` is tabulated once -- here, not per draw --
    over :data:`SAMPLER_SPAN` values and inverted by binary search, rather than using the
    continuous approximation of Clauset et al.'s appendix, which rounds a Pareto draw and is
    noticeably off for a small ``xmin``, precisely the case a degree sequence lands in. The
    table stops at ``xmin + SAMPLER_SPAN``, which for any exponent in :data:`ALPHA_BOUNDS`
    leaves less than 1e-5 of the mass above it; a network with a hub that large does not exist.
    """
    support = np.arange(xmin, xmin + SAMPLER_SPAN, dtype=np.int64)
    ccdf = special.zeta(alpha, support) / float(special.zeta(alpha, xmin))

    def draw(rng: np.random.Generator, size: int) -> np.ndarray:
        # searchsorted needs an ascending array, and a CCDF descends: negate both sides.
        index = np.clip(np.searchsorted(-ccdf, -rng.random(size), side="left"), 1, SAMPLER_SPAN) - 1
        return support[index]

    return draw


def _bootstrap_p(
    values: np.ndarray, ks: float, alpha: float, xmin: int, bootstrap: int, seed: int | None
) -> float:
    """The goodness-of-fit p-value of §9.4: how often the fit's own samples fit it this badly.

    Each synthetic data set has the size of the original and is built the way Clauset et al.
    build theirs: each point comes from the fitted power law with probability n_tail/n, and
    otherwise is drawn with replacement from the observed values *below* ``xmin``, so the head
    the fit excluded is reproduced rather than modelled. Every synthetic set is then refitted
    from scratch -- its own ``xmin``, its own ``alpha`` -- because the observed KS distance is
    the distance of a fit that chose its own ``xmin`` too, and comparing it against fits held to
    somebody else's would flatter it.

    The share itself goes through :func:`graphrag.sna.stats.empirical_p`, so this p-value has the
    same convention as every other empirical p in the package: ``(k + 1) / (samples + 1)``, which
    is never exactly zero because 200 resamples cannot resolve a probability below 1/201.
    """
    if bootstrap <= 0:
        msg = f"bootstrap must be at least 1 resample, got {bootstrap}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    draw = _power_law_sampler(alpha, xmin)
    below = values[values < xmin]
    size = int(values.size)
    tail_share = (size - below.size) / size
    distances: list[float] = []
    for _ in range(bootstrap):
        from_tail = rng.random(size) < tail_share
        drawn = int(np.count_nonzero(from_tail))
        synthetic = np.empty(size, dtype=np.int64)
        synthetic[from_tail] = draw(rng, drawn)
        synthetic[~from_tail] = (
            rng.choice(below, size - drawn, replace=True) if below.size else draw(rng, size - drawn)
        )
        refit = _best_xmin(synthetic)
        if refit is not None:
            distances.append(refit[0])
    return empirical_p(ks, distances, tail="right") if distances else 1.0


@dataclass(frozen=True)
class TailComparison:
    """A likelihood-ratio test between the fitted power law and one alternative, over one tail.

    ``ratio`` is the log-likelihood of the power law minus that of the alternative, summed over
    the tail: positive favours the power law, negative the alternative. ``statistic`` is that
    ratio normalised by its own standard deviation (Vuong), and ``p_value`` is the two-sided
    probability of a ratio at least this large in absolute value when the two models are equally
    far from the truth. A large ``p_value`` therefore means *the test cannot tell them apart*,
    which the book expects to be the usual outcome, and which is not evidence for either.
    """

    other: str
    ratio: float
    statistic: float
    p_value: float
    params: dict[str, float] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """Whether the ratio is large enough, against its own spread, to be a preference."""
        return self.p_value < LR_LEVEL

    @property
    def favours(self) -> str:
        """``"power law"``, the alternative's name, or ``"neither"`` when the test cannot tell."""
        if not self.significant:
            return "neither"
        return "power law" if self.ratio > 0 else self.other

    def sentence(self) -> str:
        """The comparison in the words a report prints."""
        if not self.significant:
            return (
                f"power law vs {self.other}: R = {self.ratio:.3g} (normalised "
                f"{self.statistic:.2f}, p = {self.p_value:.3g}) — the two cannot be told apart "
                "on this tail."
            )
        winner = "the power law" if self.ratio > 0 else f"the {self.other}"
        return (
            f"power law vs {self.other}: R = {self.ratio:.3g} (normalised {self.statistic:.2f}, "
            f"p = {self.p_value:.3g}) — {winner} fits significantly better."
        )


def compare_tails(
    sequence: Sequence[float], fit: PowerLawFit | None = None, **kwargs: Any
) -> tuple[TailComparison, ...]:
    """Likelihood-ratio tests of the fitted power law against the lognormal and the exponential.

    §9.4 gives the order and the reasoning. *"First, make sure that your observations cannot be
    explained with an exponential. Confusion between a power law and some other distribution
    such as an exponential is hard. If you think a distribution might be an exponential, then
    it's definitely not a power law. Second, try to see if you can statistically prefer a power
    law model over a lognormal."*

    Both alternatives are fitted **on the same tail** as the power law -- the values at or above
    the fit's ``xmin`` -- because a likelihood ratio between models fitted to different data is
    not a comparison of models. Both are discrete, to match the power law:

    exponential
        the geometric distribution, which is what a discrete exponential is: p(k) = (1-q)
        q^(k-xmin), fitted in closed form from the tail's mean. This is the *"random"*
        alternative the chapter contrasts with cumulative advantage.
    lognormal
        the continuous lognormal integrated over each unit interval and renormalised on
        k >= xmin, fitted by numerical maximum likelihood. §9.4's Figure 9.20(b) is the warning
        this test exists to answer: a power law and a lognormal *"can yield extremely similar
        results"* and *"you cannot really tell which of the two functions fits the data better"*
        by eye.

    The test itself is Vuong's, via :func:`graphrag.sna.stats.likelihood_ratio`, which is the
    form Clauset et al. use: the summed log-likelihood difference normalised by its own standard
    deviation, so a ratio is only a preference when it is large compared with how much it varies
    from point to point.

    ``fit`` is the fit to compare; when it is ``None`` the sequence is fitted here with
    ``kwargs`` (``bootstrap``, ``seed``) passed through. Returns an empty tuple when there is no
    fit to compare against -- a network too small to fit is too small to compare.
    """
    fitted = fit if fit is not None else fit_power_law(sequence, **kwargs)
    if fitted is None:
        return ()
    sample = np.asarray(sequence, dtype=np.float64)
    tail = sample[sample >= fitted.xmin].astype(np.int64)
    xmin = int(fitted.xmin)
    power_law = -fitted.alpha * np.log(tail) - math.log(float(special.zeta(fitted.alpha, xmin)))
    comparisons: list[TailComparison] = []
    for name, (log_likelihood, params) in (
        ("lognormal", _discrete_lognormal(tail, xmin)),
        ("exponential", _discrete_exponential(tail, xmin)),
    ):
        ratio, statistic, p_value = likelihood_ratio(list(power_law), list(log_likelihood))
        comparisons.append(
            TailComparison(
                other=name, ratio=ratio, statistic=statistic, p_value=p_value, params=params
            )
        )
    return tuple(comparisons)


def _discrete_exponential(tail: np.ndarray, xmin: int) -> tuple[np.ndarray, dict[str, float]]:
    """The geometric distribution on k >= xmin, fitted in closed form, and its log-likelihood.

    A discrete exponential *is* a geometric: p(k) = (1 - q) q^(k - xmin) with q = e^-lambda. The
    mean of that distribution is xmin + q/(1-q), so the moment estimator is also the maximum
    likelihood one and no optimisation is needed.
    """
    mean = float(np.mean(tail))
    shifted = max(mean - xmin, 1e-12)
    q = min(max(shifted / (1.0 + shifted), 1e-12), 1.0 - 1e-12)
    rate = -math.log(q)
    return (
        math.log(1.0 - q) + (tail - xmin) * math.log(q),
        {"rate": rate, "q": q},
    )


def _discrete_lognormal(tail: np.ndarray, xmin: int) -> tuple[np.ndarray, dict[str, float]]:
    """The lognormal discretised onto the integers at or above ``xmin``, fitted numerically.

    p(k) is the lognormal's mass over ``[k - 0.5, k + 0.5)`` divided by the mass above
    ``xmin - 0.5``, which is a genuine probability mass function on the integers (it sums to one
    by construction) rather than a density evaluated at integer points. Its two parameters are
    fitted by maximising that likelihood directly, from the moments of the logged tail as a
    starting point.
    """
    logs = np.log(tail)
    start = np.array([float(np.mean(logs)), max(float(np.std(logs)), 1e-2)])

    def masses(mu: float, sigma: float) -> np.ndarray:
        frozen = scipy_stats.lognorm(sigma, scale=math.exp(mu))
        remaining = float(1.0 - frozen.cdf(xmin - 0.5))
        if remaining <= 0.0:
            return np.full(tail.shape, 1e-300)
        mass = frozen.cdf(tail + 0.5) - frozen.cdf(tail - 0.5)
        return np.clip(np.asarray(mass, dtype=np.float64), 1e-300, None) / remaining

    def negative_log_likelihood(params: np.ndarray) -> float:
        sigma = abs(float(params[1]))
        if sigma < 1e-6:
            return float("inf")
        return -float(np.sum(np.log(masses(float(params[0]), sigma))))

    result = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead")
    mu, sigma = float(result.x[0]), abs(float(result.x[1]))
    return np.log(masses(mu, sigma)), {"mu": mu, "sigma": sigma}


# --------------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class ModeDegrees:
    """One side of a two-mode network, measured on its own (§9.1).

    A bipartite network has two degree sequences and they are not comparable: in a
    speakers-entities network a speaker's degree counts the entities they named and an entity's
    degree counts the speakers who named it, and the two have different means by construction
    (they sum to the same |E| over different numbers of nodes). Fitting their union finds the
    *gap between the modes* rather than the shape of either tail -- the ``xmin`` sweep will
    happily put its cut between the two means -- so each side gets its own fit, its own
    comparisons and its own verdict, and the report prints both.
    """

    mode: str
    nodes: int
    isolated: int
    mean: float
    summary: Summary
    distribution: DegreeDistribution
    fit: PowerLawFit | None
    comparisons: tuple[TailComparison, ...]
    verdict: str
    reading: str


@dataclass(frozen=True)
class DegreeReport:
    """One degree section: the sampling frame, the distribution, the fit, and the verdict."""

    kind: str
    frame: str
    nodes: int
    edges: int
    directed: bool
    bipartite: bool
    weighted: bool
    isolated: int
    summary: Summary
    distribution: DegreeDistribution
    continuous_fits: dict[str, FitResult]
    fit: PowerLawFit | None
    comparisons: tuple[TailComparison, ...]
    verdict: str
    reading: str
    #: One entry per mode of a two-mode network, each with its own fit and verdict; empty on a
    #: one-mode network. When this is non-empty, :attr:`fit` is ``None`` and :attr:`verdict` is
    #: "two modes, measured separately": the answer is in here, twice.
    modes: tuple[ModeDegrees, ...] = ()
    sample: dict[str, Any] | None = None
    reweighted: dict[int, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def comparison(self, other: str) -> TailComparison | None:
        """The comparison against one named alternative, or ``None`` when it was not run."""
        return next((c for c in self.comparisons if c.other == other), None)


def degree_report(
    graph: nx.Graph,
    kind: str = "degree",
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int | None = None,
) -> DegreeReport:
    """Everything chapter 9 asks of one network's degrees, with the verdict it allows.

    Builds the sequence (§9.1), the distribution (§9.2), the continuous fits of §3.2 through
    :func:`graphrag.sna.stats.fit_distributions`, the discrete power-law fit and its bootstrap
    (§9.4) and the two likelihood ratios, then decides what may be said. The ladder, in order:

    0. the network has two modes: *two modes, measured separately*. §9.1 gives a bipartite
       network one degree sequence per mode, and they are not comparable, so each is fitted and
       read on its own and the mixed sequence is not fitted at all. The per-mode verdicts are
       in ``modes``; the whole-network reading names them.
    1. fewer than :data:`MIN_DEGREE_SAMPLE` nodes with a degree, or no tail long enough to fit:
       *too few nodes to test*. The book's reading of a system too small to show scale-free
       behaviour is that its size is what you are measuring.
    2. the exponential fits significantly better: *exponential/Poisson-like*. This is checked
       first because §9.4 checks it first.
    3. the lognormal fits significantly better: *lognormal rather than a power law*.
    4. the bootstrap rejects the power law at :data:`KS_LEVEL`: *heavy-tailed but not
       distinguishable from lognormal* when the sample is heavy-tailed by §3.1's own test and
       the lognormal could not be excluded, *heavy-tailed but not a power law* when there was no
       comparison to make, and *neither a power law nor a heavy tail* otherwise.
    5. the bootstrap does not reject it but p is at or below :data:`CSN_LEVEL`: *heavy-tailed,
       power law not rejected but evidence weak*. This is the band a textbook preferential-
       attachment network of a few thousand nodes lands in, and the word is withheld there.
    6. p clears :data:`CSN_LEVEL` and no alternative is significantly better: *scale-free* --
       and even then the reading says that the lognormal was not excluded, because it almost
       never is, and that §9.4 asks for an argument from cumulative advantage before the word is
       used in print.

    When ``graph`` carries a sampling record (``graphrag sna sample``), the re-weighted degree
    distribution of §29.3 is computed beside the raw one and the report says which is which: a
    random walk lands on a node in proportion to its degree, so the raw distribution of such a
    sample is the population's tilted by one factor of k. The fit is still run on the raw
    sequence, because the correction estimates a distribution and does not produce a corrected
    sample -- *"if what you need was the sample rather than the estimation of a simple measure,
    you're out of luck"*.
    """
    sequence = degree_sequence(graph, kind)
    shape = describe(graph)
    isolated = len([v for v in sequence if v <= 0])
    notes: list[str] = []
    fits = _continuous_fits(sequence) if sequence else {}
    fit, comparisons, summary, verdict, reading, note = _measure(
        sequence, kind, bootstrap=bootstrap, seed=seed
    )
    if note:
        notes.append(note)

    modes = _mode_reports(graph, kind, bootstrap=bootstrap, seed=seed)
    if modes:
        # §9.1: the union of two mode sequences is not a degree distribution of anything. The
        # per-mode fits above are the answer; the mixed one is not run at all.
        fit, comparisons = None, ()
        verdict, reading = _two_mode_reading(modes)

    record = sample_record(graph)
    reweighted: dict[int, float] = {}
    if record is not None and kind in {"degree", "in", "out"}:
        reweighted = reweighted_degree_distribution(int(v) for v in sequence)
        notes.append(
            f"This network is a {record['method']} sample. The distribution printed as "
            "'re-weighted' is the RWRW correction of §29.3, which is the estimate of the "
            "population's degree distribution; the raw one beside it is the sample's own. The "
            "power-law fit ran on the raw sequence, because the correction estimates a "
            "distribution rather than producing a corrected sample."
        )
    elif record is not None:
        notes.append(
            f"This network is a {record['method']} sample and the §29.3 correction is defined "
            "for the degree, not for the strength, so nothing below is corrected for the "
            "sampler."
        )
    if shape.bipartite:
        notes.append(
            "This is a two-mode network, and §9.1 gives it two degree sequences, one per mode, "
            "which both sum to the number of edges and are not comparable with each other. The "
            "fit and the verdict are therefore per mode; the distribution printed for the "
            "network as a whole is their union, and its shape is partly the gap between the "
            "two means. --project rebuilds the network as one mode if that is the question."
        )
    if graph.graph.get("layers"):
        notes.append(
            "This network is the flattening of a multilayer one, so every degree here counts an "
            "edge whichever layer it came from. §9.1's question on a multilayer network is the "
            "degree *per layer* -- and its layer relevance, the share of a node's neighbours "
            "reachable through one layer -- which is chapter 7's object: build one layer with "
            "--layers and measure that, rather than reading this distribution per layer."
        )
    if graph.is_multigraph():
        notes.append(
            "This is a multigraph, so a degree here counts connections and not neighbours: "
            "§9.1's k_u >= |N_u|, with equality only where no pair is joined twice."
        )
    if graph.is_directed() and kind == "degree":
        notes.append(
            "On a directed network §9.1 makes the degree sequence a list of pairs. `degree` "
            "here is the total, in + out; run --kind in and --kind out for the two halves, "
            "which are different questions and can have different shapes."
        )
    if isolated:
        notes.append(
            f"{isolated:,} node(s) have a degree of zero. They are counted in n and in the "
            "distribution, and excluded from the fit: the discrete power law is defined for "
            "k >= 1."
        )
    return DegreeReport(
        kind=kind,
        frame=str(graph.graph.get("frame", "")),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        directed=bool(graph.is_directed()),
        bipartite=shape.bipartite,
        weighted=shape.weighted,
        isolated=isolated,
        summary=summary,
        distribution=degree_distribution(sequence if sequence else [0.0]),
        continuous_fits=fits,
        fit=fit,
        comparisons=comparisons,
        verdict=verdict,
        reading=reading,
        modes=modes,
        sample=record,
        reweighted=reweighted,
        notes=tuple(notes),
    )


def _continuous_fits(sequence: Sequence[float]) -> dict[str, FitResult]:
    """§3.2's fits of the same sequence, for context, or nothing when it is too small for them."""
    try:
        return fit_distributions(sequence)
    except ValueError:
        return {}


def _measure(
    sequence: Sequence[float], kind: str, *, bootstrap: int, seed: int | None
) -> tuple[PowerLawFit | None, tuple[TailComparison, ...], Summary, str, str, str]:
    """Fit one degree sequence, compare its tails and read the result (§9.4).

    The whole-network path and each mode of a two-mode network go through here, so a per-mode
    verdict is reached by exactly the same ladder as a one-mode network's. Returns the fit, its
    comparisons, §3.1's summary of the sequence, the verdict, its reading, and a note for the
    caller's list (empty when there is nothing to say).
    """
    if not sequence:
        summary = describe_sample([0.0])
        verdict, reading = _verdict(0, summary, None, ())
        return (
            None,
            (),
            summary,
            verdict,
            reading,
            "This network is empty: there are no degrees to distribute.",
        )
    summary = describe_sample(sequence)
    if kind.startswith("weighted") and any(float(v) != round(float(v)) for v in sequence):
        return (
            None,
            (),
            summary,
            "not counts, so the discrete test does not apply",
            "These strengths are not whole numbers, and §9.4's estimator is the discrete one, "
            "over counts. Nothing was fitted here: §3.2's continuous fits are printed for "
            "context, and the unweighted degree of the same network is what the chapter's test "
            "is defined on.",
            "These strengths are not whole numbers, so the discrete estimator of §9.4 does not "
            "apply and no power law was fitted. The continuous fits are from §3.2; refit the "
            "unweighted degree for the chapter's test.",
        )
    fit = fit_power_law(sequence, bootstrap=bootstrap, seed=seed)
    comparisons = compare_tails(sequence, fit) if fit is not None else ()
    verdict, reading = _verdict(len([v for v in sequence if v > 0]), summary, fit, comparisons)
    return (fit, comparisons, summary, verdict, reading, "")


def _mode_reports(
    graph: nx.Graph, kind: str, *, bootstrap: int, seed: int | None
) -> tuple[ModeDegrees, ...]:
    """One measured report per mode of a two-mode network (§9.1), empty on a one-mode one."""
    modes = sorted({str(data.get("mode", "")) for _, data in graph.nodes(data=True)} - {""})
    if len(modes) < 2:
        return ()
    rows: list[ModeDegrees] = []
    for mode in modes:
        values = degree_sequence(graph, kind, mode=mode)
        fit, comparisons, summary, verdict, reading, _ = _measure(
            values, kind, bootstrap=bootstrap, seed=seed
        )
        rows.append(
            ModeDegrees(
                mode=mode,
                nodes=len(values),
                isolated=len([v for v in values if v <= 0]),
                mean=float(np.mean(values)) if values else 0.0,
                summary=summary,
                distribution=degree_distribution(values if values else [0.0]),
                fit=fit,
                comparisons=comparisons,
                verdict=verdict,
                reading=reading,
            )
        )
    return tuple(rows)


def _two_mode_reading(modes: Sequence[ModeDegrees]) -> tuple[str, str]:
    """The whole-network verdict on a two-mode network: the answer is per mode, and here it is."""
    per_mode = "; ".join(
        f"{row.mode} ({row.nodes:,} nodes, mean degree {row.mean:.3g}): {row.verdict}"
        for row in modes
    )
    return (
        "two modes, measured separately",
        "§9.1 gives a bipartite network one degree sequence per mode, and they are not "
        "comparable with each other: they sum to the same number of edges over different "
        "numbers of nodes, so their means differ by construction and the union of the two has a "
        "gap in it that no fit should be reading. Each mode was therefore fitted and tested on "
        f"its own, and those verdicts are the answer -- {per_mode}. The distribution printed "
        "for the network as a whole is the union of the two, and is here to be looked at rather "
        "than fitted.",
    )


def _verdict(
    positive: int,
    summary: Summary,
    fit: PowerLawFit | None,
    comparisons: Sequence[TailComparison],
) -> tuple[str, str]:
    """What the data support, in one of :data:`VERDICTS`, and the paragraph that reads it."""
    if fit is None:
        if positive < MIN_DEGREE_SAMPLE:
            return (
                "too few nodes to test",
                f"{positive:,} node(s) have a degree above zero, and §9.4's procedure needs at "
                f"least {MIN_DEGREE_SAMPLE} for an exponent to mean anything. Read the quartiles "
                "and the CCDF as a description of this network and fit nothing.",
            )
        return (
            "too few nodes in the tail to test",
            f"No value of xmin leaves {MIN_TAIL_SIZE} nodes over {MIN_TAIL_VALUES} distinct "
            "degrees at or above it, so there is no tail to fit. The degrees are concentrated "
            "on a few values, which is itself the finding: a distribution worth fitting spans "
            "orders of magnitude (§9.2).",
        )
    lognormal = next((c for c in comparisons if c.other == "lognormal"), None)
    exponential = next((c for c in comparisons if c.other == "exponential"), None)
    tail = (
        f"alpha = {fit.alpha:.3f} (± {fit.standard_error:.3f}) above xmin = {fit.xmin:g}, over "
        f"{fit.n_tail:,} of {fit.n:,} nodes; KS = {fit.ks_statistic:.4f}, bootstrap p = "
        f"{fit.p_value:.3f}."
    )
    if exponential is not None and exponential.significant and exponential.ratio < 0:
        return (
            "exponential/Poisson-like",
            f"{tail} The exponential fits this tail significantly better than the power law "
            f"({exponential.sentence()}), and §9.4 is explicit that this settles it: 'if you "
            "think a distribution might be an exponential, then it's definitely not a power "
            "law'. Degrees this shape are what a network wired at random produces (ch. 16).",
        )
    if lognormal is not None and lognormal.significant and lognormal.ratio < 0:
        return (
            "lognormal rather than a power law",
            f"{tail} The lognormal fits this tail significantly better ({lognormal.sentence()}). "
            "A lognormal is what multiplying many independent positive quantities produces, with "
            "no cumulative advantage anywhere in it, so the tail is broad without being "
            "scale-free (§9.4).",
        )
    if fit.rejected:
        if lognormal is not None and summary.heavy_tailed:
            return (
                "heavy-tailed but not distinguishable from lognormal",
                f"{tail} p is at or below {KS_LEVEL}, so data drawn from this very fit almost "
                "never sit this far from it: the power law is rejected. The sample is still "
                f"heavy-tailed by §3.1's test, and {lognormal.sentence()} So the honest "
                "statement is the chapter's own: the degrees are broad and unequal, and which "
                "broad distribution generated them is not settled here.",
            )
        if summary.heavy_tailed:
            return (
                "heavy-tailed but not a power law",
                f"{tail} p is at or below {KS_LEVEL}, so the power law is rejected, while §3.1's "
                "test still calls the sample heavy-tailed. Report the median, the quartiles and "
                "the CCDF; the mean is not the typical node.",
            )
        return (
            "neither a power law nor a heavy tail",
            f"{tail} The power law is rejected by the bootstrap and §3.1 does not call the "
            "sample heavy-tailed either. The degrees concentrate around their middle, which is "
            "what a randomly wired network looks like (ch. 16).",
        )
    undecided = (
        " The lognormal could not be excluded, which §9.4 warns is the usual outcome, so the "
        "word needs an argument from cumulative advantage -- the more connections a node has, "
        "the more likely the next one attaches to it -- and not only these numbers."
        if lognormal is not None and not lognormal.significant
        else ""
    )
    variance = (
        " With alpha at or below 3 the variance of this distribution is undefined (§9.3), so no "
        "reading of it may be motivated as 'x standard deviations from the average'."
        if fit.infinite_variance
        else ""
    )
    if not fit.plausible:
        return (
            "heavy-tailed, power law not rejected but evidence weak",
            f"{tail} p is above {KS_LEVEL}, so the bootstrap does not reject the power law, but "
            f"it is at or below the {CSN_LEVEL} that Clauset, Shalizi and Newman ask for before "
            "calling one plausible, so the word is withheld. What the data do support is the "
            "chapter's own closing position: the degrees are broad and unequal, spanning orders "
            f"of magnitude.{undecided}{variance}",
        )
    return (
        "scale-free",
        f"{tail} The bootstrap does not reject the power law at the {CSN_LEVEL} Clauset, "
        "Shalizi and Newman ask for, and no alternative fits significantly better."
        f"{undecided}{variance}",
    )


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    return f"{value:,.{places}f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _sampled_rows(values: Sequence[Any], limit: int = TABLE_ROWS) -> list[int]:
    """Evenly spaced indices into ``values``, so a long curve prints as a readable table."""
    if len(values) <= limit:
        return list(range(len(values)))
    step = (len(values) - 1) / (limit - 1)
    return sorted({round(i * step) for i in range(limit)})


def render_degree(report: DegreeReport) -> list[str]:
    """The degree section of a report: n and the frame first, then the fit, then the verdict."""
    distribution = report.distribution
    lines = [
        "## Degree",
        "",
        f"**{CHAPTER}.** {KIND_MEANING[report.kind]}",
        "",
        f"**Sampling frame.** {report.frame or 'not recorded'}",
        "",
        f"n = {report.nodes:,} nodes and {report.edges:,} edges; {report.isolated:,} node(s) "
        f"with no edge at all. Degree variant: `{report.kind}`.",
        "",
    ]
    if report.modes:
        lines += [
            "§9.1: a two-mode network has one degree sequence per mode, and they are not "
            "comparable with each other. Each is fitted and read on its own below; the "
            "distribution printed for the whole network is their union.",
            "",
        ]
        lines += _table(
            ["mode", "nodes", "mean degree", "verdict"],
            [[row.mode, f"{row.nodes:,}", _num(row.mean, 2), row.verdict] for row in report.modes],
        )
    lines += [
        "### The distribution (§9.2)",
        "",
        f"- mean {_num(report.summary.mean, 3)}, median {_num(report.summary.median, 3)}, "
        f"quartiles {_num(report.summary.q1, 3)} and {_num(report.summary.q3, 3)}, "
        f"maximum {_num(report.summary.maximum, 3)}",
        f"- distinct degree values: {len(distribution.values):,}",
        "",
    ]
    if report.summary.caveat:
        lines += [f"> {report.summary.caveat}", ""]
    rows = _sampled_rows(distribution.values)
    lines += ["The CCDF, p(K >= k) — the shape §9.2 says to read and to fit:", ""]
    lines += _table(
        ["k", "p(k)", "p(K >= k)"],
        [
            [
                _num(distribution.values[i], 3),
                f"{distribution.pmf[i]:.4g}",
                f"{distribution.ccdf[i]:.4g}",
            ]
            for i in rows
        ],
    )
    if len(distribution.values) > len(rows):
        lines += [
            f"({len(rows)} of {len(distribution.values):,} distinct degrees, evenly spaced; the "
            "whole curve is in the JSON payload.)",
            "",
        ]
    bin_rows = _sampled_rows(distribution.bins)
    lines += [
        f"Log-binned histogram (§9.2), first bin one degree wide and each next "
        f"{distribution.factor:g}x wider. The bin size is a choice, and a choice is what lets "
        "you see a pattern that is not there — read the CCDF above for anything that matters:",
        "",
    ]
    lines += _table(
        ["from", "to", "nodes", "density"],
        [
            [
                _num(distribution.bins[i].low, 2),
                _num(distribution.bins[i].high, 2),
                f"{distribution.bins[i].count:,}",
                f"{distribution.bins[i].density:.4g}",
            ]
            for i in bin_rows
        ],
    )
    if report.reweighted:
        lines += [
            "### Corrected for the sampler (§29.3)",
            "",
            "This network is a sample, so the distribution above is the sample's own. The RWRW "
            "correction below re-weights each sampled node by 1/k to estimate the population's "
            "degree distribution:",
            "",
        ]
        corrected = sorted(report.reweighted.items())
        picked = _sampled_rows(corrected)
        lines += _table(
            ["k", "corrected p(k)", "sampled p(k)"],
            [
                [
                    f"{corrected[i][0]:,}",
                    f"{corrected[i][1]:.4g}",
                    f"{_sampled_pmf(distribution, corrected[i][0]):.4g}",
                ]
                for i in picked
            ],
        )
    lines += _render_fit(report.fit, report.comparisons, report.continuous_fits)
    for row in report.modes:
        lines += [
            f"### Mode `{row.mode}` on its own (§9.1)",
            "",
            f"{row.nodes:,} node(s), {row.isolated:,} of them with no edge; mean degree "
            f"{_num(row.mean, 3)}, median {_num(row.summary.median, 3)}, maximum "
            f"{_num(row.summary.maximum, 3)}; {len(row.distribution.values):,} distinct degree "
            "value(s).",
            "",
        ]
        lines += _render_fit(
            row.fit, row.comparisons, None, f"#### Power law for `{row.mode}` (§9.3-§9.4)"
        )
        lines += [f"**{row.mode}: {row.verdict}.** {row.reading}", ""]
    lines += [
        "### Verdict",
        "",
        f"**{report.verdict}.** {report.reading}",
        "",
    ]
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return lines


def _sampled_pmf(distribution: DegreeDistribution, value: float) -> float:
    """p(k) as the sample measured it, for the row beside a corrected one."""
    for observed, probability in zip(distribution.values, distribution.pmf, strict=True):
        if observed == value:
            return probability
    return 0.0


def _render_fit(
    fit: PowerLawFit | None,
    comparisons: Sequence[TailComparison],
    continuous_fits: Mapping[str, FitResult] | None = None,
    heading: str = "### Power law: fit, test, compare (§9.3-§9.4)",
) -> list[str]:
    """The fitting and testing block of §9.4, or the reason there is not one.

    Takes the fit rather than the report, because a two-mode network has one of these per mode
    and the two have to be printed the same way as a one-mode network's.
    """
    lines = [heading, ""]
    if fit is None:
        lines += [
            "No fit. A power law was not fitted to this sequence, so nothing below claims one; "
            "the verdict says why.",
            "",
        ]
    else:
        lines += [
            f"- alpha = {fit.alpha:.4f} ± {fit.standard_error:.4f} (maximum likelihood, discrete, "
            "never a regression on the log-log plane)",
            f"- xmin = {fit.xmin:g}, chosen as the value with the smallest KS distance; the fit "
            f"describes the {fit.n_tail:,} node(s) at or above it, "
            f"{fit.n_tail / max(fit.n, 1):.0%} of those with a degree",
            f"- KS distance = {fit.ks_statistic:.4f}",
            f"- goodness of fit p = {fit.p_value:.3f} over {fit.bootstrap} bootstrap resamples"
            + (f", seed {fit.seed}" if fit.seed is not None else ", unseeded")
            + f" (rejected outright at p <= {KS_LEVEL}; the verdict's own bar is Clauset, "
            f"Shalizi and Newman's p > {CSN_LEVEL}, below which the word is withheld)",
            "",
            f"**Null model.** {NULL_MODEL.format(bootstrap=fit.bootstrap)}",
            "",
        ]
        if fit.at_bound:
            lines += [
                f"> The exponent reached the top of the search range ({ALPHA_BOUNDS[1]:g}): this "
                "tail is steeper than any power law a network has been reported with, so alpha "
                "is a ceiling rather than an estimate.",
                "",
            ]
        if comparisons:
            lines += [f"- {comparison.sentence()}" for comparison in comparisons]
            lines += [""]
    if continuous_fits:
        lines += [
            "For context, §3.2's continuous fits of the whole sequence — the same KS statistic "
            "for each shape, with parameters estimated from the sample being tested, so their "
            "p-values are optimistic and none of them is the test above:",
            "",
        ]
        lines += _table(
            ["distribution", "parameters", "KS", "log-likelihood"],
            [
                [
                    name,
                    ", ".join(f"{key}={value:.4g}" for key, value in result.params.items()),
                    f"{result.ks_statistic:.4f}",
                    f"{result.log_likelihood:.1f}",
                ]
                for name, result in continuous_fits.items()
            ],
        )
    return lines


def _distribution_payload(distribution: DegreeDistribution) -> dict[str, Any]:
    """One §9.2 distribution as JSON: the whole curve, not the rows the markdown samples."""
    return {
        "values": list(distribution.values),
        "counts": list(distribution.counts),
        "pmf": list(distribution.pmf),
        "ccdf": list(distribution.ccdf),
        "log_bin_factor": distribution.factor,
        "bins": [
            {"low": b.low, "high": b.high, "count": b.count, "density": b.density}
            for b in distribution.bins
        ],
    }


def _fit_payload(fit: PowerLawFit | None) -> dict[str, Any] | None:
    """One §9.4 fit as JSON, with both levels and the one the verdict rests on named."""
    if fit is None:
        return None
    return {
        "alpha": fit.alpha,
        "xmin": fit.xmin,
        "n": fit.n,
        "n_tail": fit.n_tail,
        "ks_statistic": fit.ks_statistic,
        "p_value": fit.p_value,
        "bootstrap": fit.bootstrap,
        "standard_error": fit.standard_error,
        "seed": fit.seed,
        "at_bound": fit.at_bound,
        "rejected": fit.rejected,
        "plausible": fit.plausible,
        "null_model": NULL_MODEL.format(bootstrap=fit.bootstrap),
        # Two bars, and which one the word rests on: p <= reject_level rejects the power law
        # outright, and only p > plausible_level -- Clauset, Shalizi and Newman's own rule, which
        # is the one the verdict uses -- allows "scale-free".
        "reject_level": KS_LEVEL,
        "plausible_level": CSN_LEVEL,
        "verdict_level": CSN_LEVEL,
    }


def _comparison_payload(comparison: TailComparison) -> dict[str, Any]:
    """One §9.4 likelihood ratio as JSON, with the direction it points already resolved."""
    return {
        "other": comparison.other,
        "ratio": comparison.ratio,
        "statistic": comparison.statistic,
        "p_value": comparison.p_value,
        "significant": comparison.significant,
        "favours": comparison.favours,
        "params": comparison.params,
    }


def degree_payload(report: DegreeReport) -> dict[str, Any]:
    """The same section as plain JSON-able data, for the ``--json`` output."""
    payload: dict[str, Any] = {
        "chapter": CHAPTER,
        "kind": report.kind,
        "meaning": KIND_MEANING[report.kind],
        "frame": report.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "isolated": report.isolated,
        "directed": report.directed,
        "bipartite": report.bipartite,
        "weighted": report.weighted,
        "modes": [
            {
                "mode": row.mode,
                "nodes": row.nodes,
                "isolated": row.isolated,
                "mean_degree": row.mean,
                "median_degree": row.summary.median,
                "heavy_tailed": row.summary.heavy_tailed,
                "distribution": _distribution_payload(row.distribution),
                "power_law": _fit_payload(row.fit),
                "comparisons": [_comparison_payload(c) for c in row.comparisons],
                "verdict": row.verdict,
                "reading": row.reading,
            }
            for row in report.modes
        ],
        "summary": {
            "count": report.summary.count,
            "mean": report.summary.mean,
            "median": report.summary.median,
            "std": report.summary.std,
            "q1": report.summary.q1,
            "q3": report.summary.q3,
            "maximum": report.summary.maximum,
            "heavy_tailed": report.summary.heavy_tailed,
            "caveat": report.summary.caveat,
        },
        "distribution": _distribution_payload(report.distribution),
        "continuous_fits": {
            name: {
                "params": result.params,
                "ks_statistic": result.ks_statistic,
                "p_value": result.p_value,
                "log_likelihood": result.log_likelihood,
            }
            for name, result in report.continuous_fits.items()
        },
        "power_law": _fit_payload(report.fit),
        "comparisons": [_comparison_payload(c) for c in report.comparisons],
        "verdict": report.verdict,
        "reading": report.reading,
        "notes": list(report.notes),
    }
    if report.sample is not None:
        payload["sample"] = report.sample
        payload["reweighted"] = {str(k): v for k, v in report.reweighted.items()}
    return payload
