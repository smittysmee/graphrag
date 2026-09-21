"""Does this network join like number to like number? (Atlas ch. 31, pp. 441-449)

Chapter 30 asks whether edges join nodes with the same *label*. This one asks the same question
of a *quantity*, and the change of question changes the arithmetic: ``"1 and 2 are different
values, but that still counts more towards assortativity than connecting a node of value 1 to a
node of value 100"`` (p. 441). A label is either shared or not; a number is near or far, so the
measure is a correlation rather than a matching rate, and the book keeps the word
**assortativity** for the quantitative case and **homophily** for the qualitative one (p. 441).

The most studied quantity is the degree, and the chapter's own advice is that everything it says
about degree holds for any other number a node carries: *"any technique for the estimation of
degree assortativity can be employed to estimate any other quantitative attribute's
assortativity"* (p. 441). So one module answers both, and ``--by <numeric key>`` and the degree
section of every report run the same code over different values.

Four things come out of it.

*The endpoint correlation* (§31.1, p. 443). Every edge is one observation: put the value of one
endpoint in ``x`` and of the other in ``y``, *"note that each edge contributes two entries to
this vector -- unless your network is directed"*, and take the Pearson correlation. For the
degree that is exactly Newman's degree assortativity coefficient, and
:func:`numeric_assortativity` reproduces ``networkx``'s ``degree_assortativity_coefficient`` to
the last digit on the karate club (-0.4756). Spearman is computed beside it because the values
are usually broad and Pearson is the linear question (§3.4).

*The neighbour-average curve* (§31.1, p. 443). The second strategy plots one point per node --
its own value against the mean of its neighbours' -- and for a real network aggregates every node
with the same value into one point, *"otherwise, we would have an unreadable cloud of points at
low degree values"* (p. 444). The book then asks for a power fit on the logged axes, whose
exponent *"tells you whether the network is degree assortative (if it's positive), disassortative
(if it's negative), or non assortative (if it's statistically indistinguishable from zero)"*
(p. 443).

*The friendship paradox* (§31.2). Your friends have more friends than you do, and the chapter is
precise about why: *"a node with degree k appears in k other nodes' averages, and hence is
'over-counted' by an amount equal to how much larger it is than the network's mean degree"*
(p. 447). :func:`friendship_paradox` reports both sides of the inequality and the share of nodes
it holds for, and states the identity it follows from.

*The attribute against the degree* (§31.3). A quantitative attribute usually correlates with the
degree, and then it inherits the paradox: *"everything that correlates with degree -- be it
happiness, income, or tax fraud -- will get its own paradox for free"* (p. 449). So the numeric
section bins the nodes by degree and describes the attribute inside each bin, with §3.1's
heavy-tail caveat attached to every mean it prints.

**Which null, and why it differs by key.** For an attribute, the null is the label shuffle of
§19.1: the same graph, the same numbers, dealt out again. For the *degree* that null is
meaningless, because the values are the graph -- dealing degrees out to random nodes measures
nothing that happened. The null the chapter itself names is the configuration model: *"if we were
to generate a random version of the co-authorship network respecting its degree distribution --
for instance via a configuration model -- we would obtain a degree disassortative network"*
(p. 445). A broad degree distribution cannot be assortative at the top, because there are not
enough hubs to go around, so the honest question for degree is not "is r different from zero" but
"is r different from what a random wiring of these same degrees produces". That is what
:data:`NUMERIC_NULLS` selects between, and the default is chosen by the key.

**What §31.3 describes and this does not compute.** The chapter's own worked example (Figure
31.8, the business-to-business trustworthiness network) measures assortativity a third way: for
each node, the mean difference in the attribute between it and its *neighbours* against the mean
difference between it and its *non-neighbours*, and counts how many nodes sit above the identity
line. It is not implemented here, and the reason is the frame rather than the arithmetic. The
non-neighbour set of a node in a projected corpus network is the complement of a dense
projection, so the comparison is a pass over all |V|^2 pairs -- hundreds of millions on a corpus
of tens of thousands of entities -- and the honest alternative, a seeded sample of non-neighbour
pairs, is a different measurement with a sampling frame, an n and a null of its own, which this
section would then have to print and defend. The endpoint correlation and the neighbour-average
curve above answer the same question on the same data with an exact frame, so the third method
was left out rather than approximated. Anyone adding it should treat it as its own section.

**What a number here is not.** ``mentions``, ``documents`` and ``chunks`` are quantities every
network in this package carries, and they are counted from the very memberships the edges were
projected from: a speaker in many documents meets many speakers by construction. They are
measurable and they are not attributes of the thing, and the report says so beside them. A
*borrowed* value -- one an entity took from the documents its passages sit in -- is worse and is
refused the verdict outright, exactly as :mod:`graphrag.sna.attributes` refuses a borrowed label:
a borrowed number is still borrowed.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import networkx as nx
import numpy as np

from graphrag.extract.attributes import AttributeTable, as_number
from graphrag.sna.attributes import (
    CONFOUNDED_SHARE,
    MIN_NODES,
    NULL_SAMPLES,
    PERMUTATIONS,
    attribute_labels,
    attribute_sources,
)
from graphrag.sna.export import INHERITED, OWN
from graphrag.sna.measures import undirected_view
from graphrag.sna.null import (
    HOLDS_FIXED,
    Significance,
    configuration,
    label_permutation,
    significance,
)
from graphrag.sna.stats import (
    MIN_CORRELATION_SAMPLE,
    Correlation,
    Summary,
    describe,
    pearson,
    spearman,
)

__all__ = [
    "BUILTIN_NUMERIC",
    "CHAPTER",
    "NUMERIC_NULLS",
    "CurvePoint",
    "DegreeBin",
    "DegreeCorrelations",
    "EndpointCorrelation",
    "FriendshipParadox",
    "NeighbourCurve",
    "NumericReport",
    "attribute_by_degree",
    "default_null",
    "degree_correlations",
    "degree_correlations_payload",
    "friendship_paradox",
    "is_numeric",
    "neighbour_average_curve",
    "numeric_assortativity",
    "numeric_payload",
    "numeric_report",
    "numeric_values",
    "render_degree_correlations",
    "render_numeric",
]

#: The numbers every network here already carries, with what each one counts. They are node data
#: rather than declared attributes, so no vocabulary declares them and no sidecar writes them --
#: and none of them is a property of the thing the node stands for. See :func:`_builtin_note`.
BUILTIN_NUMERIC: dict[str, str] = {
    "degree": (
        "how many distinct others this node is joined to in this network, which is the quantity "
        "ch. 31 is mostly about"
    ),
    "mentions": "how many passages named this entity",
    "documents": "how many documents this node was recorded in",
    "chunks": "how many passages this speaker wrote",
}

#: Which null the endpoint correlation is measured against. ``permutation`` deals the same
#: numbers out to the same nodes over the same graph (§19.1); ``configuration`` rewires the edges
#: and keeps every degree (§18.1, §19.1), which is the null p. 445 names for the degree because a
#: heavy-tailed degree distribution is disassortative before anything else happens to it.
NUMERIC_NULLS: tuple[str, ...] = ("permutation", "configuration")

#: The node-data key the permutation null shuffles. Values reach a graph as strings under many
#: different keys -- ``attr_<key>`` for a declared attribute, ``mentions`` for a count, nothing
#: at all for the degree -- so every one of them is copied onto one private key first and the
#: null shuffles that. Nothing outside this module reads it.
VALUE_KEY = "attr__quantity"

#: Below this many aggregated points the log-log fit of §31.1 is arithmetic rather than a fit:
#: three points always admit a line, and :func:`graphrag.sna.stats.pearson` refuses fewer.
MIN_CURVE_POINTS = MIN_CORRELATION_SAMPLE

#: How many rows of the neighbour-average curve a report prints. The curve can have as many
#: points as there are distinct values; the shape is what matters, so the table is sampled and
#: says that it was.
TABLE_ROWS = 20

#: What the chapter is, for the line every section prints next to its numbers.
CHAPTER = "Atlas ch. 31 (§31.1 degree correlations, §31.2 friendship paradox, §31.3 attributes)"


# --------------------------------------------------------------------------- reading the values


def numeric_values(graph: nx.Graph, key: str) -> dict[str, float]:
    """Each node's number for ``key``, leaving out every node that has none.

    Three kinds of key are read, and the first that matches wins:

    ``degree``
        computed from the graph rather than read off it, so every node has one.
    one of the other :data:`BUILTIN_NUMERIC` keys
        the node data the builders write (``mentions``, ``documents``, ``chunks``). A network
        whose builder does not write that key has no values for it, which is reported as "no
        node carries this" rather than as zeros -- a count of zero and an absent count are
        different claims.
    a declared attribute
        the string under ``attr_<key>``, parsed back to a float. A value that does not parse is
        dropped with its node, the same way an unlabelled node is left out of a categorical
        measure: there is nothing to correlate.

    Undefined for nothing. An empty dict means no node carried a usable number, and every
    measure below reports that rather than dividing by zero.
    """
    if key == "degree":
        return {node: float(degree) for node, degree in graph.degree()}
    if key in BUILTIN_NUMERIC:
        values: dict[str, float] = {}
        for node, data in graph.nodes(data=True):
            number = as_number(str(data.get(key, "")))
            if number is not None:
                values[node] = number
        return values
    parsed: dict[str, float] = {}
    for node, text in attribute_labels(graph, key).items():
        number = as_number(text)
        if number is not None:
            parsed[node] = number
    return parsed


def is_numeric(graph: nx.Graph, key: str, table: AttributeTable | None = None) -> bool:
    """Whether ``key`` is a quantity on this graph, so ch. 31's measures are the ones to run.

    Asked in three places, in this order.

    The key is one of :data:`BUILTIN_NUMERIC`: it is a count, and a count is a number.

    The persona's vocabulary declares it (``table``): its declaration decides, because that is
    where somebody said what the key means. ``type: number`` is quantitative and a key with
    ``values:`` is not, even when every one of those values happens to be a numeral -- a closed
    list of codes is a set of labels that look like numbers.

    Nothing declares it: the values decide. A key whose every value parses as a finite number,
    over at least two distinct numbers, is read as a quantity. This is the case for a graph read
    in from a file and for a persona that has not written its vocabulary down yet, and it is the
    only case where this function can be wrong: an undeclared key holding ``1`` and ``2`` as
    codes will be correlated rather than counted. Declaring it with ``values:`` says so.
    """
    if key in BUILTIN_NUMERIC:
        return True
    spec = table.spec(key) if table is not None else None
    if spec is not None:
        return spec.numeric
    labels = attribute_labels(graph, key)
    numbers = [as_number(text) for text in labels.values()]
    if not numbers or any(number is None for number in numbers):
        return False
    return len({number for number in numbers if number is not None}) >= 2


def _flat(graph: nx.Graph) -> nx.Graph:
    """The undirected view §31.1's curve and §31.2's paradox are defined on.

    Both are statements about neighbourhoods rather than about direction: "the average degree of
    my neighbours" has no in- or out- version that the chapter defines, and reading only the
    successors of a node on a directed network would answer a question nobody asked. The endpoint
    correlation keeps the direction instead -- there each edge is one ordered observation -- which
    is the one place §31.1 distinguishes the two cases (p. 443).
    """
    return undirected_view(graph)[0]


# --------------------------------------------------------- §31.1 the endpoint correlation


@dataclass(frozen=True)
class EndpointCorrelation:
    """§31.1's first strategy: one observation per edge, the two endpoints' values as the pair."""

    key: str
    #: How many nodes carried a number, and how many edges joined two of them. The frame.
    nodes: int
    edges: int
    #: How many (x, y) pairs went into the coefficient: twice the edges on an undirected network,
    #: because "each edge contributes two entries to this vector" (p. 443), once each on a
    #: directed one.
    pairs: int
    pearson: Correlation | None = None
    spearman: Correlation | None = None
    #: The same coefficient over the null sample, with the null named. ``None`` when no null was
    #: asked for or none could be built.
    null: Significance | None = None
    null_model: str = ""
    #: :func:`graphrag.sna.stats.describe` of the node values themselves, which is where the
    #: heavy-tail caveat that governs how r may be read comes from (§3.1).
    values: Summary | None = None
    directed: bool = False
    #: Why there is no coefficient, in words, when there is none.
    undefined: str = ""

    @property
    def r(self) -> float | None:
        """Newman's coefficient when the key is the degree; the Pearson over endpoints always."""
        return self.pearson.coefficient if self.pearson else None

    @property
    def rho(self) -> float | None:
        return self.spearman.coefficient if self.spearman else None

    @property
    def z(self) -> float:
        return self.null.z if self.null and self.null.testable else 0.0

    @property
    def significant(self) -> bool:
        """Whether the observation sits more than two null standard deviations from the null."""
        return bool(self.null and self.null.testable and abs(self.null.z) >= 2)

    @property
    def verdict(self) -> str:
        """Assortative, disassortative or neither, read against the null that was run.

        The three cases are the chapter's own (p. 443), and the sentence names the null because
        for the degree "not different from zero" and "not different from the null" are different
        claims -- p. 445's structural disassortativity puts the null well below zero.
        """
        return _assortativity_verdict(self)


def numeric_assortativity(
    graph: nx.Graph,
    key: str,
    *,
    permutations: int = PERMUTATIONS,
    samples: int = NULL_SAMPLES,
    null: str | None = None,
    seed: int | None = None,
) -> EndpointCorrelation:
    """Correlate ``key`` across the endpoints of every edge (§31.1, p. 443).

    *"You iterate over all the edges in the network and put into two vectors the degrees of the
    nodes at the two endpoints. Note that each edge contributes two entries to this vector --
    unless your network is directed. So, if your network only contains a single edge connecting
    nodes 1 and 2, your two vectors are x = [k1, k2] and y = [k2, k1]... Then, assortativity is
    simply the Pearson correlation coefficient of these two vectors."* That is done literally
    here, which is why it reproduces ``nx.degree_assortativity_coefficient`` exactly for
    ``key="degree"``: the mixing-matrix formula and this one are the same number, and subtracting
    one from every degree to get the excess degree does not move a Pearson coefficient.

    Spearman is computed on the same pairs, because §3.4 says which coefficient was used is part
    of the finding and the values in these networks are usually broad: Pearson answers "do the
    two rise together on a straight line", Spearman "do they rise together at all".

    ``null`` is one of :data:`NUMERIC_NULLS` and defaults by key -- ``configuration`` for the
    degree, ``permutation`` for anything else. See this module's docstring for why the two cannot
    be swapped: for the degree the values *are* the graph, so only a rewiring is a null, and
    p. 445's structural disassortativity is a property of the degree distribution that a label
    shuffle prices at zero.

    Undefined, with the reason in ``undefined`` rather than as an exception, when fewer than
    three endpoint pairs survive or when every endpoint carries the same number: a variable that
    does not vary cannot covary (§3.4). A perfectly assortative network is one whose every
    component is a clique (p. 443), and such a network reaches r = 1 only when its cliques differ
    in size -- one clique on its own has a constant degree and no coefficient at all.

    **What the book defines and what this adds.** Both sentences above are the chapter's. The
    null, the Spearman coefficient and the refusal cases are this package's contract (§3.3,
    §3.4, §19.1); on a directed network §31.1 (p. 445-446) asks for *four* coefficients --
    in-in, in-out, out-in, out-out -- and this reports one, the ordered source-to-target
    correlation of whatever ``key`` names, with ``directed`` set so the report can say so.
    """
    model = null if null is not None else default_null(key)
    if model not in NUMERIC_NULLS:
        msg = f"null must be one of {', '.join(NUMERIC_NULLS)}, got {model!r}"
        raise ValueError(msg)
    values = numeric_values(graph, key)
    sub: nx.Graph = graph.subgraph(values).copy()
    left, right = _endpoint_pairs(sub, values)
    base = EndpointCorrelation(
        key=key,
        nodes=len(values),
        edges=sub.number_of_edges(),
        pairs=len(left),
        values=describe(sorted(values.values())) if values else None,
        directed=bool(graph.is_directed()),
        null_model=model,
    )
    if len(left) < MIN_CORRELATION_SAMPLE:
        return _undefined(
            base,
            f"{len(left)} endpoint pair(s) over {sub.number_of_edges()} edge(s) between nodes "
            f"carrying '{key}': §3.4 needs at least {MIN_CORRELATION_SAMPLE} pairs, since with "
            "two points every monotone relationship is perfect.",
        )
    if len(set(left)) == 1 or len(set(right)) == 1:
        return _undefined(
            base,
            f"Every endpoint carries the same value for '{key}', so there is nothing to "
            "correlate: a variable that does not vary cannot covary (§3.4). A regular network "
            "has no degree assortativity, which is a fact about it rather than a failure.",
        )
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    observed = pearson(left, right)
    sampled = _null_scores(graph, key, model, permutations=permutations, samples=samples, rng=rng)
    return EndpointCorrelation(
        key=base.key,
        nodes=base.nodes,
        edges=base.edges,
        pairs=base.pairs,
        pearson=observed,
        spearman=spearman(left, right),
        # Two-sided: disassortativity is as much a finding as assortativity, and p. 442 spends as
        # long on the protein network that hubs avoid each other in as on the coauthorship one.
        null=significance(
            observed.coefficient,
            sampled,
            null="configuration" if model == "configuration" else "label_permutation",
            tail="two",
        ),
        null_model=model,
        values=base.values,
        directed=base.directed,
    )


def default_null(key: str) -> str:
    """Which of :data:`NUMERIC_NULLS` this key is read against unless the caller says otherwise."""
    return "configuration" if key == "degree" else "permutation"


def _undefined(base: EndpointCorrelation, reason: str) -> EndpointCorrelation:
    """The same frame with no coefficient and the reason there is none."""
    return EndpointCorrelation(
        key=base.key,
        nodes=base.nodes,
        edges=base.edges,
        pairs=base.pairs,
        null_model=base.null_model,
        values=base.values,
        directed=base.directed,
        undefined=reason,
    )


def _endpoint_pairs(
    graph: nx.Graph, values: Mapping[str, float]
) -> tuple[list[float], list[float]]:
    """The two vectors of p. 443: each undirected edge twice, each directed edge once.

    Self-loops are skipped. They would contribute the same value to both vectors and inflate any
    coefficient towards +1; none of the corpus networks has one, and a network read in from a
    file should not acquire a convention here that the book does not state.
    """
    left: list[float] = []
    right: list[float] = []
    for u, v in graph.edges():
        if u == v:
            continue
        first, second = values[u], values[v]
        left.append(first)
        right.append(second)
        if not graph.is_directed():
            left.append(second)
            right.append(first)
    return left, right


def _null_scores(
    graph: nx.Graph,
    key: str,
    model: str,
    *,
    permutations: int,
    samples: int,
    rng: random.Random,
) -> list[float]:
    """The same Pearson over the null sample, however that null is built."""
    if model == "configuration":
        return _rewired_scores(graph, key, samples, rng)
    return _permuted_scores(graph, key, permutations, rng)


def _rewired_scores(graph: nx.Graph, key: str, samples: int, rng: random.Random) -> list[float]:
    """r on degree-preserving rewirings of the same network (§18.1, §19.1).

    The values are read again from each sample, so for ``key="degree"`` the null carries the
    observed degree *sequence* and nothing else -- which is the comparison p. 445 asks for, since
    a broad degree distribution is disassortative in a random wiring and the observation only
    means something against that baseline. For any other key the values travel with their nodes
    and only the edges move.
    """
    scores: list[float] = []
    for sample in configuration(graph, max(samples, 0), seed=rng):
        values = numeric_values(sample, key)
        left, right = _endpoint_pairs(sample.subgraph(values), values)
        score = _coefficient(left, right)
        if score is not None:
            scores.append(score)
    return scores


def _permuted_scores(
    graph: nx.Graph, key: str, permutations: int, rng: random.Random
) -> list[float]:
    """r with the same numbers dealt out to the same nodes again (§19.1).

    :func:`graphrag.sna.null.label_permutation` shuffles one node-data key, so the numbers are
    copied onto :data:`VALUE_KEY` first and read back as floats. It holds the network and the
    multiset of values fixed and moves only which node holds which, which is the null for "is
    this pairing of numbers with positions one the same numbers could have made by accident".
    """
    values = numeric_values(graph, key)
    working: nx.Graph = graph.subgraph(values).copy()
    for node, value in values.items():
        working.nodes[node][VALUE_KEY] = repr(float(value))
    scores: list[float] = []
    for sample in label_permutation(working, max(permutations, 0), key=VALUE_KEY, seed=rng):
        shuffled = {
            node: float(data[VALUE_KEY])
            for node, data in sample.nodes(data=True)
            if VALUE_KEY in data
        }
        left, right = _endpoint_pairs(sample, shuffled)
        score = _coefficient(left, right)
        if score is not None:
            scores.append(score)
    return scores


def _coefficient(left: Sequence[float], right: Sequence[float]) -> float | None:
    """One Pearson coefficient, with §3.4's undefined cases turned into ``None``."""
    try:
        result = pearson(left, right)
    except ValueError:
        return None
    return None if math.isnan(result.coefficient) else result.coefficient


# ------------------------------------------------------------ §31.1 the neighbour-average curve


@dataclass(frozen=True)
class CurvePoint:
    """One point of Figure 31.3: a value, and the mean neighbour value of the nodes that hold it."""

    value: float
    neighbour_mean: float
    nodes: int


@dataclass(frozen=True)
class NeighbourCurve:
    """§31.1's second strategy, aggregated as p. 444 says real networks have to be."""

    key: str
    points: tuple[CurvePoint, ...] = ()
    #: The exponent of the power fit: the slope of the least-squares line through the logged
    #: points. Positive is assortative, negative disassortative, indistinguishable from zero
    #: non-assortative (p. 443). ``None`` when there were too few points to fit.
    slope: float | None = None
    intercept: float | None = None
    #: How well that line describes the aggregated points -- not how well it describes the
    #: network. See :func:`neighbour_average_curve`.
    r_squared: float = 0.0
    p_value: float = math.nan
    fitted_points: int = 0
    nodes: int = 0
    undefined: str = ""

    @property
    def reading(self) -> str:
        """What the exponent says, in the chapter's own three cases (p. 443)."""
        if self.slope is None:
            return self.undefined or "No fit."
        if math.isnan(self.p_value) or self.p_value > 0.05:
            return (
                f"The fitted exponent is {self.slope:+.3f} but the fit is not distinguishable "
                "from a flat line at the 0.05 level, which is the chapter's third case: "
                "non-assortative. A node's value says little about its neighbours'."
            )
        if self.slope > 0:
            return (
                f"The fitted exponent is {self.slope:+.3f}: positive, so the curve rises and the "
                "network is assortative -- the more a node has, the more its neighbours have."
            )
        return (
            f"The fitted exponent is {self.slope:+.3f}: negative, so the curve falls and the "
            "network is disassortative -- high values attach to low ones, which is what "
            "preferential attachment produces (p. 442)."
        )


def neighbour_average_curve(graph: nx.Graph, key: str) -> NeighbourCurve:
    """Every node's value against the mean of its neighbours', aggregated and fitted (§31.1).

    Figure 31.3's plot: *"Rather than plotting each edge, we plot each node. We compare a node's
    degree with the average degree of its neighbors. In a degree assortative network, we expect
    to see a positive correlation."* For a real network the points are aggregated first --
    *"we usually aggregate in the same data point all nodes with the same degree... otherwise, we
    would have an unreadable cloud of points at low degree values, since most nodes in real world
    networks have a degree equal to one or two"* (p. 444) -- so each returned point is one value
    and the mean, over the nodes holding it, of their own neighbour averages.

    The fit is the power fit the chapter asks for: least squares on log(value) against
    log(neighbour mean), whose slope is the exponent. Three things it is not, all of which the
    report prints beside it. It is a fit to the **aggregated** points, so a value held by one node
    weighs as much as a value held by a thousand -- that is what the book plots, and it is why
    ``nodes`` travels with every point. Points at zero on either axis are dropped, because a
    logarithm needs a positive number, and how many were dropped is the difference between
    ``nodes`` and the fitted points. And an r-squared from a regression on logged values is the
    statistic §9.2 warns about: it tells you a line fits these points, not that the relationship
    is a power law. Read the curve; quote :func:`numeric_assortativity`'s r.

    Undefined, with the reason recorded, when fewer than :data:`MIN_CURVE_POINTS` positive points
    survive or when every node holds the same value.
    """
    flat = _flat(graph)
    values = numeric_values(flat, key)
    averages = _neighbour_averages(flat, values)
    if not averages:
        return NeighbourCurve(
            key=key,
            undefined=(
                f"No node carrying '{key}' has a neighbour that carries it, so there is no "
                "neighbourhood to average over."
            ),
        )
    grouped: dict[float, list[float]] = defaultdict(list)
    for node, average in averages.items():
        grouped[values[node]].append(average)
    points = tuple(
        CurvePoint(value=value, neighbour_mean=float(np.mean(group)), nodes=len(group))
        for value, group in sorted(grouped.items())
    )
    fittable = [p for p in points if p.value > 0 and p.neighbour_mean > 0]
    curve = NeighbourCurve(key=key, points=points, nodes=len(averages))
    if len(fittable) < MIN_CURVE_POINTS:
        return NeighbourCurve(
            key=curve.key,
            points=curve.points,
            nodes=curve.nodes,
            # Set even though no fit ran: it is the count of points a fit *could* have used, and
            # the report's n line prints it, so leaving it at zero would contradict the sentence
            # below it.
            fitted_points=len(fittable),
            undefined=(
                f"{len(fittable)} point(s) sit at a positive value on both axes, and a log-log "
                f"fit needs at least {MIN_CURVE_POINTS}. The points above are the whole finding."
            ),
        )
    log_x = [math.log(p.value) for p in fittable]
    log_y = [math.log(p.neighbour_mean) for p in fittable]
    try:
        correlation = pearson(log_x, log_y)
    except ValueError as exc:
        return NeighbourCurve(
            key=curve.key,
            points=curve.points,
            nodes=curve.nodes,
            fitted_points=len(fittable),
            undefined=f"No fit: {exc}",
        )
    slope, intercept = (float(c) for c in np.polyfit(log_x, log_y, 1))
    return NeighbourCurve(
        key=curve.key,
        points=curve.points,
        nodes=curve.nodes,
        slope=slope,
        intercept=intercept,
        r_squared=correlation.coefficient**2,
        p_value=correlation.p_value,
        fitted_points=len(fittable),
    )


def _neighbour_averages(graph: nx.Graph, values: Mapping[str, float]) -> dict[str, float]:
    """For each valued node with at least one valued neighbour, the mean of their values."""
    averages: dict[str, float] = {}
    for node in values:
        neighbours = [values[other] for other in graph.neighbors(node) if other in values]
        if neighbours:
            averages[node] = float(np.mean(neighbours))
    return averages


# --------------------------------------------------------------------- §31.2 friendship paradox


@dataclass(frozen=True)
class FriendshipParadox:
    """Your friends have more friends than you (§31.2), for the degree or for any number.

    Every field is computed over the nodes with at least one neighbour: an isolated node has no
    neighbour average, and including it in one mean and not the other would compare two different
    populations. ``isolated`` says how many were left out.
    """

    key: str
    nodes: int
    isolated: int
    #: The mean value over the frame. For the degree, 2m/n over the non-isolated nodes.
    mean_value: float
    #: The mean, over nodes, of each node's neighbour average: the y axis of Figure 31.6 against
    #: its identity line.
    mean_neighbour_value: float
    #: The value of the node you reach by walking one randomly chosen edge: sum(k_u v_u)/sum(k_u).
    #: For the degree this is the <k^2>/<k> of the exact statement below.
    edge_weighted_value: float
    #: How many nodes are above the identity line of Figure 31.6: their neighbours out-do them.
    outnumbered: int

    @property
    def share(self) -> float:
        """The share of nodes whose neighbours out-do them, which is the paradox's own number."""
        return self.outnumbered / self.nodes if self.nodes else 0.0

    @property
    def excess(self) -> float:
        """How much the random neighbour beats the average node by: the size of the paradox."""
        return self.edge_weighted_value - self.mean_value

    @property
    def holds(self) -> bool:
        return self.excess > 0

    @property
    def statement(self) -> str:
        """The inequality, exactly, and what makes it exact rather than empirical."""
        if self.key == "degree":
            return (
                "Exactly: the degree of the node at the end of a randomly chosen edge averages "
                "<k^2>/<k> = <k> + var(k)/<k>, which is at least <k> for every network and equal "
                "to it only when every node has the same degree. The mean over nodes of the "
                "neighbour average obeys the same inequality, because summing k_v/k_u + k_u/k_v "
                "over the edges is at least 2 per edge. p. 447 states the mechanism rather than "
                "the algebra: 'a node with degree k appears in k other nodes' averages, and "
                "hence is over-counted by an amount equal to how much larger it is than the "
                "network's mean degree'. The only escape is a network where the degree is nearly "
                "constant, such as a small-world network with a low rewiring probability."
            )
        return (
            f"The excess here is cov(k, {self.key})/<k>: the value at the end of a random edge "
            f"beats the average node exactly when '{self.key}' correlates with the degree. That "
            "is §31.3's 'everything that correlates with degree -- be it happiness, income, or "
            "tax fraud -- will get its own paradox for free' (p. 449), stated as an identity. "
            "Unlike the degree's, this inequality can go either way, and which way it goes is "
            "the finding."
        )


def friendship_paradox(graph: nx.Graph) -> FriendshipParadox:
    """Do this network's nodes have fewer friends than their friends? (§31.2)

    *"That's the friendship paradox: your friends are, on average, more popular than you! This
    means that, for the average node, its degree is lower than the average degree of their
    neighbors."* (p. 446-447.) Three numbers come back and they are three different readings of
    the same identity: the mean degree, the mean over nodes of each node's neighbour average, and
    the degree of the node at the end of a randomly chosen edge. The last one is the exact form --
    <k^2>/<k> = <k> + var(k)/<k> -- and it is why the paradox is not a curiosity about some
    networks but a theorem about all of them: the excess is a variance, so it is zero only for a
    regular network and positive for every other.

    ``outnumbered`` counts the nodes above the identity line of Figure 31.6, which is the version
    a person feels. It is *not* implied by the inequality: the mean can be dragged up by a few
    nodes with enormous neighbours while most sit below the line. In a broad degree distribution
    it is most of them -- *"there are way more nodes above the identity line than below"*
    (p. 446) -- and on the karate club it is 85%.

    Undefined for a network with no edges, which has no neighbourhood to average: every field is
    then zero and ``nodes`` says so.
    """
    return _paradox(graph, "degree")


def _paradox(graph: nx.Graph, key: str) -> FriendshipParadox:
    """§31.2 for the degree and §31.3's free paradox for any other number, in one computation."""
    flat = _flat(graph)
    values = numeric_values(flat, key)
    degrees = {node: float(degree) for node, degree in flat.degree()}
    averages = _neighbour_averages(flat, values)
    frame = sorted(averages)
    if not frame:
        return FriendshipParadox(
            key=key,
            nodes=0,
            isolated=len(values),
            mean_value=0.0,
            mean_neighbour_value=0.0,
            edge_weighted_value=0.0,
            outnumbered=0,
        )
    weights = float(sum(degrees[node] for node in frame))
    return FriendshipParadox(
        key=key,
        nodes=len(frame),
        isolated=len(values) - len(frame),
        mean_value=float(np.mean([values[node] for node in frame])),
        mean_neighbour_value=float(np.mean([averages[node] for node in frame])),
        edge_weighted_value=(
            float(sum(degrees[node] * values[node] for node in frame) / weights) if weights else 0.0
        ),
        outnumbered=sum(1 for node in frame if averages[node] > values[node]),
    )


# ------------------------------------------------------- §31.3 the attribute against the degree


@dataclass(frozen=True)
class DegreeBin:
    """One degree band and :func:`graphrag.sna.stats.describe` of the attribute inside it."""

    low: int
    high: int
    nodes: int
    summary: Summary

    @property
    def label(self) -> str:
        return f"{self.low}" if self.low == self.high else f"{self.low}-{self.high}"


def attribute_by_degree(graph: nx.Graph, key: str) -> tuple[DegreeBin, ...]:
    """How ``key`` distributes inside each degree band (§31.3).

    §31.3's question -- is this quantity related to how connected a node is -- asked as a
    distribution rather than as one coefficient, because the coefficient hides the shape: an
    attribute can be flat in the middle of the degree range and extreme at both ends, and a
    Pearson over nodes would call that nothing.

    The bands are powers of two (0, 1, 2-3, 4-7, 8-15, ...), which is §9.2's log binning and is
    used for its reason: *"an issue of power-binning is that it forces you to make a choice"*, but
    equal-width bins on a broad degree distribution put almost every node in the first one. The
    bins partition the nodes carrying a value exactly, so their counts sum to n, and every
    ``summary`` carries §3.1's caveat -- a mean inside a bin is as untrustworthy as a mean
    anywhere else when the values are heavy-tailed, and these usually are.

    Returns an empty tuple when no node carries the key. A node whose degree is zero gets its own
    band rather than being dropped, because "the isolated nodes are the ones with the extreme
    values" is a finding this would otherwise hide.
    """
    flat = _flat(graph)
    values = numeric_values(flat, key)
    if not values:
        return ()
    degrees = {node: int(degree) for node, degree in flat.degree()}
    bins: dict[int, list[float]] = defaultdict(list)
    for node, value in values.items():
        bins[_bin_index(degrees.get(node, 0))].append(value)
    return tuple(
        DegreeBin(
            low=_bin_bounds(index)[0],
            high=_bin_bounds(index)[1],
            nodes=len(group),
            summary=describe(sorted(group)),
        )
        for index, group in sorted(bins.items())
    )


def _bin_index(degree: int) -> int:
    """Which power-of-two band a degree falls in; band 0 is degree 0 on its own."""
    return 0 if degree <= 0 else math.floor(math.log2(degree)) + 1


def _bin_bounds(index: int) -> tuple[int, int]:
    """The inclusive degree range of one band."""
    return (0, 0) if index == 0 else (2 ** (index - 1), 2**index - 1)


# ------------------------------------------------------------------------ the whole section


@dataclass(frozen=True)
class DegreeCorrelations:
    """The degree half of chapter 31, which every report prints whatever it was asked for."""

    nodes: int
    edges: int
    correlation: EndpointCorrelation
    curve: NeighbourCurve
    paradox: FriendshipParadox
    directed_note: str = ""

    @property
    def verdict(self) -> str:
        """Assortative, disassortative or neutral, against the null that was actually run."""
        return self.correlation.verdict


def degree_correlations(
    graph: nx.Graph, *, samples: int = NULL_SAMPLES, seed: int | None = None
) -> DegreeCorrelations:
    """Chapter 31 over the degree: r, the k_nn curve and the friendship paradox (§31.1, §31.2).

    Printed in every ``sna analyze`` report because the degree correlation is not an optional
    extra: *"degree assortativity is a super important property for your network. Degree
    correlations radically change many network dynamics"* (p. 444), and every centrality ranking
    above it is read differently depending on whether the hubs in it talk to each other.

    The null is the configuration model, not a label shuffle -- see this module's docstring and
    p. 445. ``samples`` rewirings are drawn; the report prints how many were actually built,
    because a small network admits none.
    """
    flat, note = undirected_view(graph)
    return DegreeCorrelations(
        nodes=flat.number_of_nodes(),
        edges=flat.number_of_edges(),
        correlation=numeric_assortativity(flat, "degree", samples=samples, seed=seed),
        curve=neighbour_average_curve(flat, "degree"),
        paradox=friendship_paradox(flat),
        directed_note=note,
    )


@dataclass(frozen=True)
class NumericReport:
    """One quantitative attribute measured against one network, with what ch. 31 asks for."""

    key: str
    meaning: str
    #: Where the decision to read this key as a number came from, in words: ``node data`` for one
    #: of :data:`BUILTIN_NUMERIC`, ``declared attribute`` for a key the persona's vocabulary
    #: declares ``type: number``, and ``undeclared attribute, read as a number from its values``
    #: for the fallback of :func:`is_numeric`. It decides which caveat the section carries: a
    #: count of the corpus is not a property of the thing, a borrowed attribute is not either,
    #: and a key nobody declared might be a set of codes that happen to look like numbers.
    source: str
    labelled: int
    total: int
    values: Summary | None
    correlation: EndpointCorrelation
    curve: NeighbourCurve
    paradox: FriendshipParadox
    bins: tuple[DegreeBin, ...]
    #: Whether a vocabulary declared this key ``type: number``. False both for a built-in count
    #: and for :func:`is_numeric`'s fallback, which the ``source`` above tells apart.
    declared: bool = False
    #: Degree correlations of the same network, so the reader can see whether this attribute is
    #: saying anything the degree was not already saying (§31.3).
    degree_correlation: float | None = None
    #: How many labelled nodes carry the value in their own right and how many borrowed it from
    #: their documents, keyed by :data:`graphrag.sna.export.OWN` and
    #: :data:`graphrag.sna.export.INHERITED`. Empty for a built-in count, which nobody recorded.
    sources: dict[str, int] = field(default_factory=dict)
    single_document: int = 0
    notes: tuple[str, ...] = ()

    @property
    def own(self) -> int:
        return self.sources.get(OWN, 0)

    @property
    def inherited(self) -> int:
        return self.sources.get(INHERITED, 0)

    @property
    def inherited_share(self) -> float:
        return self.inherited / self.labelled if self.labelled else 0.0

    @property
    def confounded(self) -> bool:
        """Whether enough values were borrowed to make the correlation circular.

        The same rule and the same threshold as the categorical case
        (:data:`graphrag.sna.attributes.CONFOUNDED_SHARE`), because it is the same problem: the
        edges come from the documents, and a value borrowed from those documents makes like join
        like whatever the number means. A borrowed number is still borrowed.
        """
        return self.inherited_share > CONFOUNDED_SHARE

    @property
    def unlabelled(self) -> int:
        return max(self.total - self.labelled, 0)

    @property
    def verdict(self) -> str:
        """What the correlation and its null support together, in one sentence."""
        if self.confounded:
            return (
                f"Not a reading of this attribute: {self.inherited:,} of {self.labelled:,} nodes "
                f"({self.inherited_share:.0%}) took their number from the very documents the "
                "edges were drawn from, so near joins near by construction and the shuffle null "
                "cannot subtract it. Record the attribute on the nodes themselves before reading "
                "anything off this coefficient."
            )
        return self.correlation.verdict


def _assortativity_verdict(correlation: EndpointCorrelation) -> str:
    """The chapter's three cases, read off the coefficient and the null that was run."""
    if correlation.r is None:
        return f"Not measurable. {correlation.undefined}"
    null = correlation.null
    against = (
        f"the {null.samples} {correlation.null_model} sample(s)"
        if null is not None and null.testable
        else "no null (none could be built)"
    )
    if null is None or not null.testable:
        return (
            f"r = {correlation.r:+.4f} over {correlation.pairs:,} endpoint pair(s), against "
            f"{against}. Report it as a description of this network and not as a finding (§19.1)."
        )
    if abs(null.z) < 2:
        return (
            f"r = {correlation.r:+.4f}, which is {null.z:+.2f} null standard deviations from "
            f"{against}: this network sorts by '{correlation.key}' about as much as the null "
            "does, so there is no quantitative assortativity here to report."
        )
    if correlation.r > null.null_mean:
        return (
            f"r = {correlation.r:+.4f} against a null mean of {null.null_mean:+.4f} "
            f"(z {null.z:+.2f} over {against}): edges join near to near more than the null does, "
            f"so this network is assortative by '{correlation.key}'."
        )
    return (
        f"r = {correlation.r:+.4f} against a null mean of {null.null_mean:+.4f} (z {null.z:+.2f} "
        f"over {against}): edges join high to low more than the null does, so this network is "
        f"disassortative by '{correlation.key}' -- hubs attach to the periphery (p. 442)."
    )


def numeric_report(
    graph: nx.Graph,
    key: str,
    *,
    permutations: int = PERMUTATIONS,
    samples: int = NULL_SAMPLES,
    null: str | None = None,
    seed: int | None = None,
    table: AttributeTable | None = None,
) -> NumericReport:
    """Chapter 31 over one quantitative key: the ``--by`` section for a number (§31.1-§31.3).

    The counterpart of :func:`graphrag.sna.attributes.analyse_attribute`, which answers the same
    question for a label. Everything is computed over the nodes carrying a usable number, never
    over the whole network, for the reason the categorical case gives: a node nobody tagged has
    nothing to correlate, and treating an absent number as zero would invent the largest group in
    most corpora.

    Provenance is read and reported exactly as it is for a label, and it decides whether the rest
    is evidence: a value the node owns is independent of the edges, one borrowed from the
    documents its passages sit in is not, and past
    :data:`graphrag.sna.attributes.CONFOUNDED_SHARE` the verdict refuses rather than reports. A
    built-in count (:data:`BUILTIN_NUMERIC`) is nobody's record at all and carries its own
    warning instead.

    ``table`` is the persona's vocabulary, and it is here so the section can say **why** this key
    was read as a number rather than as a label. Three answers, and the report prints whichever
    applies: it is one of the counts every network carries, a vocabulary declares it
    ``type: number``, or nothing declares it and :func:`is_numeric` read it off the values. The
    third is a guess -- a closed list of codes that happen to be numerals would be read the same
    way -- so it is said out loud in a note rather than left in the phrase "declared attribute".
    """
    values = numeric_values(graph, key)
    builtin = key in BUILTIN_NUMERIC
    spec = table.spec(key) if table is not None else None
    declared = bool(spec is not None and spec.numeric)
    sources = {} if builtin else _sources_of(graph, key, values)
    notes: list[str] = []
    correlation = numeric_assortativity(
        graph, key, permutations=permutations, samples=samples, null=null, seed=seed
    )
    report = NumericReport(
        key=key,
        meaning=BUILTIN_NUMERIC.get(key, f"the value each node carries for '{key}'"),
        source=_source_of(builtin, declared),
        declared=declared,
        labelled=len(values),
        total=graph.number_of_nodes(),
        values=describe(sorted(values.values())) if values else None,
        correlation=correlation,
        curve=neighbour_average_curve(graph, key),
        paradox=_paradox(graph, key),
        bins=attribute_by_degree(graph, key),
        # The same network's degree assortativity, so the section can say when an attribute is
        # restating the degree rather than describing anything of its own (§31.3, p. 449).
        degree_correlation=(None if key == "degree" else _coefficient(*_degree_pairs(graph))),
        sources=dict(Counter(sources.values())),
        single_document=sum(
            1
            for node, origin in sources.items()
            if origin == INHERITED and int(graph.nodes[node].get("documents", 0) or 0) == 1
        ),
    )
    if not values:
        return _with_notes(
            report,
            [
                f"No node in this network carries a number for '{key}'. Either nothing has been "
                "tagged with it, the key is spelled differently in the sidecars, or its values "
                "are not numbers -- in which case it is a categorical attribute and `--by` "
                f"measures it as one. {_builtin_hint(graph)}"
            ],
        )
    if len(values) < MIN_NODES:
        notes.append(
            f"{len(values)} node(s) carry a number for '{key}': too few for any of this to be a "
            "measurement. The values above are the whole finding."
        )
    if builtin:
        notes.append(_builtin_note(key))
    elif not declared:
        notes.append(_undeclared_note(key, table))
    if report.inherited:
        notes.append(
            f"{report.inherited} of {report.labelled} node(s) ({report.inherited_share:.0%}) "
            f"carry no value of their own for '{key}' and were given the one most of their "
            "documents carry. Those numbers come from the same documents as the edges, so they "
            "raise the correlation whether or not the attribute sorts anything"
            + (
                f"; {report.single_document} of them sit in a single document, where every edge "
                "is inside the document that supplied the number."
                if report.single_document
                else "."
            )
        )
    if report.unlabelled:
        notes.append(
            f"{report.unlabelled} of {report.total} node(s) carry no number for '{key}' and are "
            "in none of these measures. They are untagged, not a value of zero."
        )
    if report.values is not None and report.values.heavy_tailed:
        notes.append(
            f"The values themselves are heavy-tailed, so r is moved by a handful of nodes: a "
            "Pearson coefficient is a statement about squared deviations and those are where the "
            f"squares are. Read the Spearman coefficient beside it. {report.values.caveat}"
        )
    if report.degree_correlation is not None and abs(report.degree_correlation) > 0.5:
        notes.append(
            f"This network's degree assortativity is {report.degree_correlation:+.4f}, and "
            f"'{key}' is measured over the same edges. Check the degree bands below before "
            "reading the coefficient as a statement about the attribute rather than about how "
            "connected its holders are (§31.3)."
        )
    return _with_notes(report, notes)


def _degree_pairs(graph: nx.Graph) -> tuple[list[float], list[float]]:
    """The endpoint vectors for the degree, for the "is this just the degree" note."""
    degrees = numeric_values(graph, "degree")
    return _endpoint_pairs(graph, degrees)


def _sources_of(graph: nx.Graph, key: str, values: Mapping[str, float]) -> dict[str, str]:
    """Where each node's number came from, restricted to the nodes that have one."""
    sources = attribute_sources(graph, key)
    return {node: sources.get(node, OWN) for node in values}


def _source_of(builtin: bool, declared: bool) -> str:
    """Why this key was read as a number, in the words the section prints."""
    if builtin:
        return "node data"
    return "declared attribute" if declared else "undeclared attribute, read as a number"


def _undeclared_note(key: str, table: AttributeTable | None) -> str:
    """Said out loud whenever the routing was a guess rather than a declaration."""
    vocabulary = (
        "nothing in facets.yaml declares"
        if table is not None and bool(table)
        else "this persona has written no attribute vocabulary, so nothing declares"
    )
    return (
        f"{vocabulary} `{key}`. Every value of it parsed as a number, so it was read as a "
        "quantity and measured with chapter 31 rather than chapter 30. That is a guess about "
        "what the key means: if these are codes rather than quantities -- a band, a cohort, a "
        f"stage number -- declare `{key}` with `values:` in facets.yaml and the section becomes "
        "the categorical one. Declaring it `type: number` makes the reading here explicit."
    )


def _builtin_note(key: str) -> str:
    """The warning a count of the corpus carries and an attribute does not."""
    return (
        f"`{key}` is not an attribute anybody recorded about these nodes: it is "
        f"{BUILTIN_NUMERIC[key]}, counted from the same memberships this network was projected "
        "from. A node recorded in many documents meets many others *because* of that, so part of "
        "any correlation here is the projection describing itself. It is a real measurement of "
        "the corpus and a poor one of the world; read it beside the degree correlations, which "
        "are the same statement in the currency the chapter uses."
    )


def _builtin_hint(graph: nx.Graph) -> str:
    """Which built-in numbers this particular network does carry, since they differ per builder."""
    carried = sorted(
        name
        for name in BUILTIN_NUMERIC
        if name == "degree" or any(name in data for _, data in graph.nodes(data=True))
    )
    return f"This network carries: {', '.join(carried)}." if carried else ""


def _with_notes(report: NumericReport, notes: Sequence[str]) -> NumericReport:
    """The same report with its notes set. It is frozen, so every step builds a new one."""
    return replace(report, notes=tuple(notes))


# ----------------------------------------------------------------------------- rendering


def _num(value: float | None, places: int = 4) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _sampled(points: Sequence[CurvePoint], limit: int = TABLE_ROWS) -> list[CurvePoint]:
    """Up to ``limit`` points, evenly spaced through the curve, always including both ends."""
    if len(points) <= limit:
        return list(points)
    step = (len(points) - 1) / (limit - 1)
    indices = sorted({round(i * step) for i in range(limit)} | {0, len(points) - 1})
    return [points[i] for i in indices]


def _render_correlation(correlation: EndpointCorrelation, heading: str) -> list[str]:
    """§31.1's coefficient with its frame, its n, its null and its chapter beside it."""
    null = correlation.null
    endpoints = (
        "each edge contributing one ordered pair, source to target, because the network is directed"
        if correlation.directed
        else "each edge contributing two pairs, one per direction, as p. 443 requires"
    )
    lines = [
        heading,
        "",
        f"**Sampling frame.** The {correlation.nodes:,} node(s) carrying a number for "
        f"`{correlation.key}` and the {correlation.edges:,} edge(s) between them, "
        f"{endpoints}.",
        "",
        f"**n.** {correlation.pairs:,} endpoint pair(s).",
        "",
    ]
    if null is not None and null.testable:
        lines += [
            f"**Null model.** `{null.null}`, which holds fixed {HOLDS_FIXED.get(null.null, '')}. "
            f"{null.samples} sample(s).",
            "",
        ]
    elif correlation.r is None:
        # Two different absences, and saying the wrong one is a lie about the network: a
        # 10-cycle has a perfectly good configuration null and no coefficient to test with it.
        lines += [
            f"**Null model.** None was run. There is no coefficient here to test, so "
            f"`{correlation.null_model}` was never asked for a sample; the reason there is no "
            "coefficient is below.",
            "",
        ]
    else:
        lines += [
            f"**Null model.** `{correlation.null_model}`, which produced no usable sample -- "
            "either none was asked for, or this network is too small for the null to have a move "
            "to make. Either way the coefficient below is a description and not a test (§19.1).",
            "",
        ]
    lines += ["**Implements.** §31.1, p. 443 (the edge-centric strategy).", ""]
    if correlation.r is None:
        return [*lines, correlation.undefined, ""]
    lines += [
        f"- Pearson r: {_num(correlation.r)} "
        f"(p {_num(correlation.pearson.p_value if correlation.pearson else math.nan)})",
        f"- Spearman rho: {_num(correlation.rho)} "
        f"(p {_num(correlation.spearman.p_value if correlation.spearman else math.nan)})",
    ]
    if null is not None and null.testable:
        lines += [
            f"- null: mean {_num(null.null_mean)}, sd {_num(null.null_std)}",
            f"- z {_num(null.z, 2)}; empirical p (two-sided) {_num(null.p_value)}",
        ]
    lines += [""]
    if null is not None and null.caveat:
        lines += [null.caveat, ""]
    return lines


def _render_curve(curve: NeighbourCurve) -> list[str]:
    """Figure 31.3's plot as a table, with the power fit under it."""
    lines = [
        "### The neighbour-average curve",
        "",
        f"**Sampling frame.** The {curve.nodes:,} node(s) with at least one neighbour carrying "
        f"`{curve.key}`, aggregated into one point per distinct value as p. 444 does "
        '("we usually aggregate in the same data point all nodes with the same degree").',
        "",
        f"**n.** {len(curve.points):,} point(s); the fit ran on the {curve.fitted_points:,} that "
        "sit at a positive value on both axes.",
        "",
        "**Null model.** None: this is a description of the curve, and the coefficient above is "
        "the tested statement.",
        "",
        "**Implements.** §31.1, p. 443-444 (the node-centric strategy and its power fit).",
        "",
    ]
    if not curve.points:
        return [*lines, curve.undefined, ""]
    shown = _sampled(curve.points)
    lines += _table(
        ["value", "mean neighbour value", "nodes"],
        [[_num(p.value), _num(p.neighbour_mean), f"{p.nodes:,}"] for p in shown],
    )
    if len(shown) < len(curve.points):
        lines += [
            f"{len(shown)} of {len(curve.points)} points, evenly spaced; the shape is the point "
            "and the whole curve is in the JSON payload.",
            "",
        ]
    if curve.slope is None:
        return [*lines, curve.undefined, ""]
    lines += [
        f"- exponent (log-log least squares): {_num(curve.slope)}",
        f"- r-squared of that line: {_num(curve.r_squared)}; p {_num(curve.p_value)}",
        "",
        curve.reading,
        "",
        "The fit runs on the aggregated points, so a value one node holds weighs as much as a "
        "value a thousand hold, and an r-squared from a regression on logged values says a line "
        "fits these points rather than that the relationship is a power law (§9.2). The curve is "
        "here to be looked at; the coefficient above is the number to quote.",
        "",
    ]
    return lines


def _render_paradox(paradox: FriendshipParadox) -> list[str]:
    """§31.2, or §31.3's free paradox when the key is not the degree."""
    noun = "degree" if paradox.key == "degree" else paradox.key
    lines = [
        "### The friendship paradox" if paradox.key == "degree" else f"### The {noun} paradox",
        "",
        f"**Sampling frame.** The {paradox.nodes:,} node(s) with at least one neighbour carrying "
        f"a number; {paradox.isolated:,} node(s) have none and are outside every number here.",
        "",
        f"**n.** {paradox.nodes:,}.",
        "",
        "**Null model.** None, and none is needed: the inequality below is an identity, not an "
        "observation. What is observed is the share.",
        "",
        "**Implements.** §31.2, p. 446-447"
        + ("" if paradox.key == "degree" else ", generalised as §31.3 does on p. 449")
        + ".",
        "",
    ]
    if not paradox.nodes:
        return [*lines, "No node has a neighbour carrying a number, so there is no paradox.", ""]
    lines += [
        f"- mean {noun} of a node: {_num(paradox.mean_value)}",
        f"- mean {noun} of a node's neighbours, averaged over nodes: "
        f"{_num(paradox.mean_neighbour_value)}",
        f"- {noun} at the end of a randomly chosen edge: {_num(paradox.edge_weighted_value)}",
        f"- nodes whose neighbours out-do them: {paradox.outnumbered:,} of {paradox.nodes:,} "
        f"({paradox.share:.0%})",
        "",
        paradox.statement,
        "",
    ]
    return lines


def _render_bins(bins: Sequence[DegreeBin], key: str) -> list[str]:
    """§31.3: the attribute's distribution inside each degree band."""
    total = sum(b.nodes for b in bins)
    lines = [
        "### The attribute against the degree",
        "",
        f"**Sampling frame.** The {total:,} node(s) carrying a number for `{key}`, split into "
        "power-of-two degree bands (§9.2's log binning, because equal-width bands put almost "
        "every node in the first one). The bands partition those nodes, so the counts sum to n.",
        "",
        f"**n.** {total:,} over {len(bins)} band(s).",
        "",
        "**Null model.** None: a distribution per band is a description. The tested statement is "
        "the coefficient above.",
        "",
        "**Implements.** §31.3, p. 447-449.",
        "",
    ]
    if not bins:
        return [*lines, f"No node carries a number for `{key}`.", ""]
    lines += _table(
        ["degree", "nodes", "median", "mean", "sd", "min", "max", "heavy-tailed"],
        [
            [
                b.label,
                f"{b.nodes:,}",
                _num(b.summary.median),
                _num(b.summary.mean),
                _num(b.summary.std),
                _num(b.summary.minimum),
                _num(b.summary.maximum),
                "yes" if b.summary.heavy_tailed else "no",
            ]
            for b in bins
        ],
    )
    caveated = [b for b in bins if b.summary.heavy_tailed]
    if caveated:
        lines += [
            f"{len(caveated)} band(s) are heavy-tailed, so their means are arithmetic rather "
            f"than typical: {caveated[0].summary.caveat}",
            "",
        ]
    return lines


def render_degree_correlations(report: DegreeCorrelations) -> list[str]:
    """The degree section every report prints: r, the curve and the paradox (§31.1, §31.2)."""
    lines = [
        "## Degree correlations",
        "",
        "Whether the hubs in this network talk to each other or to the periphery. It is not a "
        'detail of the degree distribution: *"degree assortativity is a super important '
        "property for your network. Degree correlations radically change many network "
        'dynamics"* (p. 444), so every ranking above this section is read differently depending '
        "on which of the two it is.",
        "",
    ]
    if report.directed_note:
        lines += [report.directed_note, ""]
    lines += _render_correlation(report.correlation, "### Degree assortativity r")
    if report.correlation.r is not None:
        lines += [report.verdict, ""]
        lines += [
            "The null is the configuration model rather than a shuffle of the degrees, because "
            "for the degree the values *are* the graph. p. 445: a network with a broad degree "
            'distribution rewired at random comes out **disassortative** -- "the likelihood of '
            'connecting a hub to many small nodes seems too high" -- so r below zero is the '
            "expected state of a heavy-tailed network and only the distance from this null is a "
            'finding. A network that is both broad and assortative has "some non-trivial '
            "machinery driving their nodes' connections\".",
            "",
        ]
    lines += _render_curve(report.curve)
    lines += _render_paradox(report.paradox)
    return lines


def render_numeric(report: NumericReport) -> list[str]:
    """The ``--by`` section for a quantitative key: the whole of chapter 31 over one number."""
    lines = [
        f"## By attribute: {report.key} (quantitative)",
        "",
        f"{report.labelled:,} of {report.total:,} nodes carry a number for `{report.key}` "
        f"({report.source}): {report.meaning}. Chapter 31 asks of a number what chapter 30 asks "
        "of a label -- do edges join like to like -- but a number is near or far rather than "
        "same or different, so the measure is a correlation over the edges and the curve of a "
        "node's value against its neighbours'.",
        "",
    ]
    if report.values is not None:
        lines += _table(
            ["n", "median", "mean", "sd", "min", "q1", "q3", "max"],
            [
                [
                    f"{report.values.count:,}",
                    _num(report.values.median),
                    _num(report.values.mean),
                    _num(report.values.std),
                    _num(report.values.minimum),
                    _num(report.values.q1),
                    _num(report.values.q3),
                    _num(report.values.maximum),
                ]
            ],
        )
    if report.sources:
        lines += [
            f"Recorded about the node itself: {report.own:,}. Borrowed from the node's "
            f"documents: {report.inherited:,}"
            + (
                f", {report.single_document:,} of them from a single document"
                if report.single_document
                else ""
            )
            + ".",
            "",
        ]
    lines += _render_correlation(report.correlation, "### Assortativity over the edge endpoints")
    lines += [report.verdict, ""]
    lines += _render_curve(report.curve)
    lines += _render_paradox(report.paradox)
    lines += _render_bins(report.bins, report.key)
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return lines


# ----------------------------------------------------------------------------- payloads


def _correlation_payload(correlation: EndpointCorrelation) -> dict[str, Any]:
    null = correlation.null
    return {
        "key": correlation.key,
        "nodes": correlation.nodes,
        "edges": correlation.edges,
        "pairs": correlation.pairs,
        "directed": correlation.directed,
        "pearson": correlation.r,
        "pearson_p": correlation.pearson.p_value if correlation.pearson else None,
        "spearman": correlation.rho,
        "spearman_p": correlation.spearman.p_value if correlation.spearman else None,
        "null_model": correlation.null_model,
        "null": None
        if null is None
        else {
            "null": null.null,
            "holds_fixed": HOLDS_FIXED.get(null.null, ""),
            "samples": null.samples,
            "null_mean": null.null_mean,
            "null_std": null.null_std,
            "z": null.z,
            "p_value": null.p_value,
            "tail": null.tail,
            "caveat": null.caveat,
        },
        "undefined": correlation.undefined,
    }


def _curve_payload(curve: NeighbourCurve) -> dict[str, Any]:
    return {
        "key": curve.key,
        "points": [
            {"value": p.value, "neighbour_mean": p.neighbour_mean, "nodes": p.nodes}
            for p in curve.points
        ],
        "slope": curve.slope,
        "intercept": curve.intercept,
        "r_squared": curve.r_squared,
        "p_value": None if math.isnan(curve.p_value) else curve.p_value,
        "fitted_points": curve.fitted_points,
        "nodes": curve.nodes,
        "reading": curve.reading,
        "undefined": curve.undefined,
    }


def _paradox_payload(paradox: FriendshipParadox) -> dict[str, Any]:
    return {
        "key": paradox.key,
        "nodes": paradox.nodes,
        "isolated": paradox.isolated,
        "mean_value": paradox.mean_value,
        "mean_neighbour_value": paradox.mean_neighbour_value,
        "edge_weighted_value": paradox.edge_weighted_value,
        "excess": paradox.excess,
        "outnumbered": paradox.outnumbered,
        "share": paradox.share,
        "holds": paradox.holds,
        "statement": paradox.statement,
    }


def _bins_payload(bins: Sequence[DegreeBin]) -> list[dict[str, Any]]:
    return [
        {
            "low": b.low,
            "high": b.high,
            "nodes": b.nodes,
            "median": b.summary.median,
            "mean": b.summary.mean,
            "std": b.summary.std,
            "minimum": b.summary.minimum,
            "maximum": b.summary.maximum,
            "heavy_tailed": b.summary.heavy_tailed,
            "caveat": b.summary.caveat,
        }
        for b in bins
    ]


def degree_correlations_payload(report: DegreeCorrelations) -> dict[str, Any]:
    """The degree section as plain JSON-able data, for ``--json``."""
    return {
        "nodes": report.nodes,
        "edges": report.edges,
        "chapter": CHAPTER,
        "correlation": _correlation_payload(report.correlation),
        "curve": _curve_payload(report.curve),
        "paradox": _paradox_payload(report.paradox),
        "verdict": report.verdict,
        "directed_note": report.directed_note,
    }


def numeric_payload(report: NumericReport) -> dict[str, Any]:
    """The quantitative ``--by`` section as plain JSON-able data, for ``--json``."""
    return {
        "key": report.key,
        "meaning": report.meaning,
        "source": report.source,
        "declared": report.declared,
        "chapter": CHAPTER,
        "labelled": report.labelled,
        "total": report.total,
        "unlabelled": report.unlabelled,
        "values": None
        if report.values is None
        else {
            "count": report.values.count,
            "mean": report.values.mean,
            "median": report.values.median,
            "std": report.values.std,
            "minimum": report.values.minimum,
            "q1": report.values.q1,
            "q3": report.values.q3,
            "maximum": report.values.maximum,
            "heavy_tailed": report.values.heavy_tailed,
            "caveat": report.values.caveat,
        },
        "sources": report.sources,
        "own": report.own,
        "inherited": report.inherited,
        "inherited_share": report.inherited_share,
        "single_document": report.single_document,
        "confounded": report.confounded,
        "correlation": _correlation_payload(report.correlation),
        "curve": _curve_payload(report.curve),
        "paradox": _paradox_payload(report.paradox),
        "bins": _bins_payload(report.bins),
        "degree_correlation": report.degree_correlation,
        "verdict": report.verdict,
        "notes": list(report.notes),
    }
