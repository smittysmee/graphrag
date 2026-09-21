"""The family a network is compared against (ch. 19), and the one model that is not a null (§19.2).

You have one network. Whatever you measure on it -- clustering, modularity, assortativity, a
motif count -- comes out as a number, and a number on its own carries no evidence: "whenever you
observe a given property you don't have the statistical power to claim that what you're observing
is interesting" (§19). The evidence comes from a *family*: keep everything about the network
fixed except the property you are asking about, generate many networks from that family, and see
whether the observation is a typical member of it. That procedure is §19.1's four steps, and this
module is the first two of them; :mod:`graphrag.sna.stats` is the last two.

**What a null holds fixed is the whole question.** A shuffle does not test "is this network
random"; it tests "is this property explained by the things the shuffle kept". Swap that for a
different set of fixed properties and the same observation can turn from significant to ordinary,
so the null has to be named next to the number, and every sampler here says in one line what it
holds fixed (:data:`HOLDS_FIXED`):

``erdos_renyi``
    §16.1. Only the node count, and either the edge count (G(n,m)) or the density (G(n,p)).
    Degrees, clustering, components -- everything else is free, which is why it is almost always
    too weak a null for an observed network and is offered mainly as the baseline the book starts
    from.
``configuration``
    §18.1 and §19.1. Every node's degree, exactly, by repeated edge swaps -- and on a directed
    network every node's in-degree and out-degree, by the three-arc swap §19.1 asks for. This is
    the default null for anything that could be explained by "some nodes are simply busier".
``label_permutation``
    The network entirely; only which node holds which label moves. The null for a claim that a
    *node attribute* explains the edges -- and, when the label was borrowed from the same
    documents the edges were drawn from, the null that makes a circular result look strongest
    (see :mod:`graphrag.sna.attributes`).
``bipartite_preserving``
    §18.1 applied to the two-mode network underneath a projection: every left degree and every
    right degree, by curveball trades on the incidence matrix, re-projected afterwards. Every
    network this package builds is such a projection, so this is the null that keeps the
    projection's own structure -- the combinatorial weights, the clumping one big document
    causes -- instead of treating the projected graph as if it had been observed directly.

**One interface.** Each null is a generator of graphs: ``sampler(observed, n, *, seed, ...)``
yields up to ``n`` samples, and yields none at all when the observed network is too small for the
operation to mean anything (a graph with three nodes has no distinct double edge swap). That
"fewer than asked for" case is not an error and must be reported: :func:`significance` carries
the sample count, and a report prints it beside the z-score. ``NULLS`` maps the names above onto
the functions.

**The samples are for measuring, not for keeping.** A sampler may hand back one working copy
over and over, relabelled or rewired in place, so a caller that wants to keep a sample must copy
it. Nothing here ever mutates the observed network.

**Independent draws, not a chain.** §19.1 asks for independent shuffles, so every sample starts
again from the observed network rather than continuing where the last one stopped.

**Reading the result.** :func:`significance` puts §3.3's two currencies side by side: the z-score
(§19.1 "the number of standard deviations between the observation and the null average") and the
empirical p-value. The z-score assumes the null distribution is roughly normal, which §19.1 says
outright is the easy case and not the general one; when :func:`graphrag.sna.stats.describe` finds
the null sample heavy-tailed, the returned ``caveat`` says so and the empirical p is the one to
quote.

**And the thing that is not a null.** :func:`ergm` fits §19.2's exponential random graph model by
maximum pseudo-likelihood: a logistic regression over dyads whose predictors are the change
statistics of Figure 19.5's configurations -- an edge, a two-star, a three-star, a triangle --
plus reciprocity on a directed network and, optionally, an attribute match. It is the book's
"right way" in its cheapest form, and every fit carries the three warnings in
:data:`ERGM_CAVEATS`, which are part of the result rather than documentation.

**Documented, not built.** §19.2 closes with the ERGM variants recent research has moved to, and
none of them is here. Each is left out for a reason, not by oversight, and a question that needs
one of them needs a different tool rather than a looser reading of this one:

*Monte Carlo maximum likelihood* (§19.2, refs 10-11), the estimator that makes an ERGM an ERGM.
It needs an MCMC sampler and a convergence diagnostic, and a fit whose chain nobody checked is
worse than no fit; MPLE at least announces what it is. This is the gap that matters most, and
:data:`ERGM_CAVEATS` says so on every fit.

*Longitudinal ERGMs, TERGMs and their separable form* (refs 16-18), for networks observed more
than once. This package's networks are built from one corpus at one moment; ``--since``/``--until``
cut windows out of it, but two windows are two builds with different node sets, not two
observations of one evolving network, and feeding them to a temporal model would treat a change
in what was recorded as a change in the world.

*ERGMs for valued edges* (refs 19-20). Every weight here is a count of shared documents, and a
valued ERGM would model that count as if the corpus had sampled it; §26.1's warning about
combinatorial projected weights applies first, and backboning (ch. 27) is the honest thing to do
with such a weight before any model sees it.

*Multilayer ERGMs* (ref 21), which need the multilayer network ATL-07 owns. When the layers are a
first-class object, a model over them is a ticket, not an afterthought here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression

from graphrag.sna.matrices import Side, adjacency, incidence
from graphrag.sna.projection import (
    DEFAULT_LAMBDA,
    PAIRS,
    graph_from_incidence,
)
from graphrag.sna.stats import Summary, describe, empirical_p, mean, std, z_score

__all__ = [
    "ERGM_CAVEATS",
    "ERGM_TERMS",
    "HOLDS_FIXED",
    "NULLS",
    "ERGMFit",
    "ERGMTerm",
    "Significance",
    "bipartite_preserving",
    "configuration",
    "erdos_renyi",
    "ergm",
    "label_permutation",
    "pairs_of",
    "projected_side",
    "render_ergm",
    "significance",
]

#: A double edge swap needs four nodes to swap between and two edges to swap, so below this a
#: degree-preserving null has no move to make and the sampler yields nothing at all. The directed
#: swap rotates three arcs at a time (see :func:`configuration`), so it needs three of them.
MIN_SWAP_NODES = 4
MIN_SWAP_EDGES = 2
MIN_DIRECTED_SWAP_EDGES = 3

#: How many swaps a sample attempts, per edge. §19.1 calls this "a non trivial quantity to
#: evaluate" and leaves it open (Bottazzi and Pirino, *Measuring industry relatedness and
#: corporate coherence*, 2010), so one per edge is this package's convention and not a settled
#: answer: it is what both rewiring nulls used before they moved here, and raising it is the
#: first thing to try when a null distribution sits suspiciously close to the observation.
SWAPS_PER_EDGE = 1

#: How many rejected swaps ``networkx`` tolerates per edge before it gives up. Twenty
#: is what both of this package's rewiring nulls used before they moved here.
TRIES_PER_EDGE = 20

#: Curveball trades per sample, per row of the incidence matrix. Strona et al. found five trades
#: per row enough to lose the starting configuration on ecological matrices; the floor covers the
#: small networks a test plants, where five per row would be a handful of trades.
TRADES_PER_ROW = 5
MIN_TRADES = 20

#: Above this many dyads an MPLE fit is a regression over millions of rows, which is not what
#: §19.2 is for. ``ergm`` refuses instead, and says to take an ego network or a component.
MAX_ERGM_DYADS = 250_000

#: What each null holds fixed, in one line, so a report can name its null model next to the
#: number rather than saying "against a null".
HOLDS_FIXED: dict[str, str] = {
    "erdos_renyi": (
        "the node count and the edge count (G(n,m)), or the node count and the density "
        "(G(n,p)); degrees, clustering and components are all free (§16.1)"
    ),
    "configuration": (
        "every node's degree, exactly -- in-degree and out-degree separately when the network is "
        "directed -- by edge swaps over the observed edges (§18.1, §19.1)"
    ),
    "label_permutation": (
        "the network itself and how many nodes carry each value; only which node holds which "
        "value moves (§19.1)"
    ),
    "bipartite_preserving": (
        "every left and every right degree of the two-mode network the projection came from, "
        "by curveball trades on the incidence matrix (§18.1 on the two-mode case)"
    ),
}


def _random(seed: int | random.Random | None) -> random.Random:
    """The generator to draw from: the caller's own if they passed one, else one seeded here.

    Passing a ``random.Random`` in is how a report that draws several nulls keeps one stream, so
    the whole report is reproducible from a single ``--seed`` -- and it is what lets a caller
    interleave its own draws with a sampler's, since a generator only advances when it is asked
    for the next sample.
    """
    return seed if isinstance(seed, random.Random) else random.Random(seed)  # noqa: S311


def _weights(graph: nx.Graph) -> list[float]:
    """The observed edge weights, in edge order. A missing or zero weight counts as 1.0, because
    an edge is a shared something and these networks never mean "shared none of it"."""
    return [float(data.get("weight", 1.0) or 1.0) for _, _, data in graph.edges(data=True)]


def _carry_weights(sample: nx.Graph, weights: Sequence[float], rng: random.Random) -> None:
    """Deal the observed weights over a sample's edges, so a weighted measure has something to read.

    A rewired network has the observed *set* of weights but says nothing about which edge should
    carry which, so they are dealt at random. When the sample has a different number of edges --
    only G(n,p) does -- they are drawn with replacement instead, which keeps the weight
    distribution and gives up the exact multiset.
    """
    if not weights:
        return
    target = sample.number_of_edges()
    if target == len(weights):
        dealt = list(weights)
        rng.shuffle(dealt)
    else:
        dealt = [weights[rng.randrange(len(weights))] for _ in range(target)]
    for (u, v), weight in zip(sample.edges(), dealt, strict=True):
        sample[u][v]["weight"] = weight


# ----------------------------------------------------------------------------- §16.1 random graphs


def erdos_renyi(
    graph: nx.Graph,
    n: int,
    *,
    seed: int | random.Random | None = None,
    p: float | None = None,
    weights: bool = True,
) -> Iterator[nx.Graph]:
    """Random graphs on the observed node set: G(n,m), or G(n,p) when ``p`` is given (§16.1).

    G(n,m) draws uniformly from "all possible graphs with n nodes and m edges" and so matches the
    observed edge count exactly; G(n,p) tosses the same coin for every one of the ``n(n-1)/2``
    pairs, so its edge count varies around ``p·n(n-1)/2`` -- the book's own conversion between the
    two. Either way the edges are *independent*: this is the null in which "having a common friend
    increases the chance of being connected" is forbidden, which is what makes it the weakest null
    here and the one that flatters almost any observation.

    The sample carries the observed node ids so a measure can be read node by node against the
    observed network. ``weights`` deals the observed edge weights over the sample's edges; see
    :func:`_carry_weights` for what that does and does not preserve.

    Yields nothing when the network has fewer than two nodes or no edges: there is no family to
    draw from. Raises ``ValueError`` for a ``p`` outside [0, 1].
    """
    if p is not None and not 0.0 <= p <= 1.0:
        msg = f"p must be a probability in [0, 1], got {p!r}"
        raise ValueError(msg)
    nodes = sorted(graph.nodes)
    edges = graph.number_of_edges()
    if len(nodes) < 2 or edges == 0:
        return
    rng = _random(seed)
    observed = _weights(graph)
    relabel = dict(enumerate(nodes))
    for _ in range(max(n, 0)):
        draw = rng.randrange(1_000_000)
        sample = (
            nx.gnm_random_graph(len(nodes), edges, seed=draw)
            if p is None
            else nx.gnp_random_graph(len(nodes), p, seed=draw)
        )
        labelled: nx.Graph = nx.relabel_nodes(sample, relabel, copy=True)
        if weights:
            _carry_weights(labelled, observed, rng)
        yield labelled


# -------------------------------------------------------------- §18.1 / §19.1 degree preserving


def configuration(
    graph: nx.Graph,
    n: int,
    *,
    seed: int | random.Random | None = None,
    weights: bool = True,
) -> Iterator[nx.Graph]:
    """Degree-preserving rewirings of the observed network, by edge swaps (§19.1, §18.1).

    The edge swap of Figure 19.1: take two edges ``(1,2)`` and ``(3,4)`` and replace them with
    ``(1,3)`` and ``(2,4)``. Every node keeps its degree, and repeated enough times -- once per
    edge here -- the result is "quite different from your original one". ``networkx`` rejects a
    swap that would create a self-loop or a parallel edge, exactly as §19.1 requires.

    **Directed networks get the directed swap.** §19.1 says the procedure keeps the degree
    distribution "also in case of a directed network, provided that you always swap edges in the
    correct direction", and the two-edge swap cannot do that: reversing one edge to make the
    pairing work would change two nodes' in-degree and two others' out-degree. So a ``DiGraph``
    is rewired with ``nx.directed_edge_swap``, which rotates *three* arcs (``a->b->c->d`` becomes
    ``a->c->b->d``) and therefore holds every node's in-degree **and** out-degree fixed; an
    undirected graph keeps ``nx.double_edge_swap``. Neither holds reciprocity fixed: a directed
    swap can create or destroy a mutual pair, which is exactly why reciprocity is a thing worth
    testing against this null (and an ERGM term, :func:`ergm`). Three arcs per swap also means a
    directed network needs at least three edges before there is a move to make.

    §18.1 explains why this rather than a Molloy-Reed configuration model, although the two end up
    with the same degree sequence: Molloy-Reed builds a graph *like* yours from a degree sequence
    and may be forced into self-loops and parallel edges, while shuffling stays inside the family
    of simple graphs and stays close enough to the observation "because they need to be compared
    to it". When the comparison is with data, "the differences are crucial".

    What it does **not** hold fixed is everything a reader might assume it does: the number of
    connected components, the clustering, the communities. §19.1 lists those as further
    constraints you could add by rejecting more swaps; none of them is added here. A
    degree-preserving null is therefore not a community null, and a modularity that beats it has
    beaten "the degrees", not "chance".

    **How many swaps is not a settled question.** §19.1 says so outright -- "the number of swaps
    to perform before stopping is a non trivial quantity to evaluate", citing Bottazzi and Pirino
    -- and this package's convention is :data:`SWAPS_PER_EDGE` attempted swaps per edge, one
    each. That is a convention, not an answer: too few and the sample is the observation with a
    dent in it, so the null looks like the data and the z-score collapses. A caller who has
    reason to think the chain has not mixed should raise it and see whether the answer moves.

    ``weights`` deals the observed edge weights over the rewired edges, which is what a weighted
    measure needs: ``networkx`` gives a swapped edge no ``weight`` at all, since the pair it now
    joins never had one. Turning it off leaves that hole, so it is only for an unweighted measure.

    Yields nothing when the graph is too small for its swap to have a move -- fewer than
    :data:`MIN_SWAP_NODES` nodes, or fewer than :data:`MIN_SWAP_EDGES` edges undirected and
    :data:`MIN_DIRECTED_SWAP_EDGES` directed -- and skips a sample when the degree sequence admits
    no swap at all. Both cases are why a caller counts the samples it actually got rather than the
    number it asked for, and the second is not rare in the directed case: the three-arc swap needs
    a path ``a->b->c->d`` whose rotation does not duplicate an arc, so a network where one side
    only sends and the other only receives, or a small dense clique of arcs, can reject every
    attempt. A network like that has no degree-preserving null available and a report has to say
    so, which is what a zero sample count is for.
    """
    edges = graph.number_of_edges()
    directed = graph.is_directed()
    floor = MIN_DIRECTED_SWAP_EDGES if directed else MIN_SWAP_EDGES
    if graph.number_of_nodes() < MIN_SWAP_NODES or edges < floor:
        return
    swap = nx.directed_edge_swap if directed else nx.double_edge_swap
    rng = _random(seed)
    observed = _weights(graph)
    for _ in range(max(n, 0)):
        rewired = graph.copy()
        try:
            swap(
                rewired,
                nswap=edges * SWAPS_PER_EDGE,
                max_tries=edges * TRIES_PER_EDGE,
                seed=rng.randrange(1_000_000),
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue  # too few distinct degrees to rewire; this sample contributes nothing
        if weights:
            _carry_weights(rewired, observed, rng)
        yield rewired


# ----------------------------------------------------------------------------- label shuffling


def label_permutation(
    graph: nx.Graph,
    n: int,
    *,
    key: str,
    seed: int | random.Random | None = None,
) -> Iterator[nx.Graph]:
    """The same network with the values of one node attribute dealt out again at random.

    ``key`` is the node-data key itself (``attr_region``, not ``region``), so this stays a
    statement about graphs rather than about this package's attribute vocabulary. Nodes carrying
    no value for ``key`` are left out of the shuffle and keep having none: an untagged node is not
    a value (see :mod:`graphrag.sna.attributes`).

    This null holds the network *and* the mix of values fixed, so it tests exactly one claim: that
    the observed pairing of labels with positions is not one the same labels would have made by
    accident. It cannot test anything about the edges, and it is the wrong null when the labels
    were themselves derived from whatever produced the edges -- shuffling then destroys the very
    correlation that the borrowing created, so the observation looks stronger the more circular it
    is. What it also cannot see is a label that is really a *position*: if one value belongs to
    the busiest nodes, a fixed graph plus moved labels will always call that a finding.
    :func:`bipartite_preserving` is the null for that second case, because it keeps each node's
    activity while moving everything else. Neither null repairs the first case; only recording the
    attribute on the nodes themselves does (see :mod:`graphrag.sna.attributes`).

    The yielded graph is one working copy, relabelled in place each time; the observed graph is
    never touched. Yields nothing when fewer than two nodes carry a value.
    """
    nodes = sorted(node for node, data in graph.nodes(data=True) if str(data.get(key, "")).strip())
    if len(nodes) < 2:
        return
    rng = _random(seed)
    values = [str(graph.nodes[node][key]) for node in nodes]
    working = graph.copy()
    for _ in range(max(n, 0)):
        rng.shuffle(values)
        for node, value in zip(nodes, values, strict=True):
            working.nodes[node][key] = value
        yield working


# ------------------------------------------------------------- §18.1 on the two-mode network


def bipartite_preserving(
    pairs: Iterable[tuple[str, str]],
    n: int,
    *,
    side: Side = "left",
    scheme: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    seed: int | random.Random | None = None,
    trades: int | None = None,
) -> Iterator[nx.Graph]:
    """Re-projections of a two-mode network that keep every left and every right degree.

    The projected networks this package builds are not observations. What was observed is a set of
    memberships -- this speaker in that document, this entity in that passage -- and the graph is
    ``B @ B.T`` (§8.3, §26.1). Rewiring the *projection* with :func:`configuration` therefore
    invents graphs that no membership table could have produced: it breaks up the cliques a single
    document necessarily creates, and so calls the projection's own arithmetic a discovery. This
    null rewires the memberships instead and projects again, which keeps the arithmetic and
    randomises only who was in what.

    The rewiring is the **curveball** algorithm of Strona et al. (Strona, Nappo, Boccacci,
    Fattorini and San-Miguel-Ayanz, "A fast and unbiased procedure to randomize ecological binary
    matrices with fixed row and column totals", *Nature Communications* 5:4114, 2014), which is
    §18.1's fixed-degree-sequence idea applied to a binary matrix: take two rows, leave the columns
    they share alone, pool the columns only one of them has, and deal that pool back out so each
    row ends with the number it started with. Every row sum and every column sum is preserved
    exactly, and the trades sample the space of such matrices uniformly in the limit, which the
    naive swap-until-it-looks-different procedure does not.

    ``side`` is the mode to project onto, matching
    :func:`graphrag.sna.export.bipartite_projection`: ``left`` joins two left nodes that share a
    right node, ``right`` the other way round. ``scheme`` and ``lam`` are chapter 26's weighting,
    and they have to be the ones the observed network used: a hyperbolic projection compared
    against simple-weight re-projections is a comparison of two definitions, not a null. The
    caller reads them off the observed graph (:data:`graphrag.sna.projection.PROJECTION`), the
    sample carries them on itself, and the default is the default projection. Isolated nodes are
    kept, so every sample has the observed node set and a measure can be read against it.

    ``trades`` defaults to :data:`TRADES_PER_ROW` per row of the incidence matrix (floored at
    :data:`MIN_TRADES`); each sample starts again from the observed memberships rather than
    continuing the previous chain, because §19.1 asks for independent shuffles. Yields nothing
    when there are fewer than two rows to trade between.
    """
    memberships = sorted(set(pairs))
    if not memberships:
        return
    matrix, rows, columns = incidence(memberships)
    dense_matrix = np.asarray(matrix, dtype=np.float64)
    if len(rows) < 2:
        return
    row_sets = [set(np.flatnonzero(dense_matrix[index]).tolist()) for index in range(len(rows))]
    rng = _random(seed)
    swaps = trades if trades is not None else max(TRADES_PER_ROW * len(rows), MIN_TRADES)
    for _ in range(max(n, 0)):
        traded = [set(row) for row in row_sets]
        for _ in range(max(swaps, 0)):
            _curveball(traded, rng)
        sample = np.zeros_like(dense_matrix)
        for index, held in enumerate(traded):
            if held:
                sample[index, sorted(held)] = 1.0
        # The same builder the observation went through (``export.bipartite_projection``), so
        # the null's weights are made the way the observed weights were.
        projected = graph_from_incidence(sample, rows, columns, side=side, scheme=scheme, lam=lam)
        # The sample carries the memberships it came from, exactly as a real projection does
        # (``export.bipartite_projection``), so a null network is the same kind of object as an
        # observed one: its margins can be read back, and it can be nulled again in its turn.
        projected.graph[PAIRS] = sorted(
            (rows[index], columns[column]) for index, held in enumerate(traded) for column in held
        )
        yield projected


def _curveball(rows: list[set[int]], rng: random.Random) -> None:
    """One curveball trade between two randomly chosen rows, in place.

    The shared columns cannot move without changing a column total, so they stay put; the rest are
    pooled and dealt back, ``len(only_first)`` to the first row and the remainder to the second.
    Each pooled column goes to exactly one of the two rows, so both row sums and every column sum
    come out unchanged.
    """
    first, second = rng.sample(range(len(rows)), 2)
    shared = rows[first] & rows[second]
    only_first = sorted(rows[first] - shared)
    pool = only_first + sorted(rows[second] - shared)
    if not pool:
        return
    rng.shuffle(pool)
    rows[first] = shared | set(pool[: len(only_first)])
    rows[second] = shared | set(pool[len(only_first) :])


def pairs_of(graph: nx.Graph) -> list[tuple[str, str]] | None:
    """The two-mode memberships a projected graph was built from, or ``None`` if it has none.

    :func:`graphrag.sna.export.bipartite_projection` records them on the graph so that a null can
    go back to what was observed. A graph that never went through a projection -- the topic
    network, which reads stored co-occurrence edges, or any graph built by hand -- has none, and
    that is the case a caller has to refuse rather than work around.
    """
    stored = graph.graph.get(PAIRS)
    if not stored:
        return None
    return [(str(left), str(right)) for left, right in stored]


def projected_side(graph: nx.Graph, pairs: Sequence[tuple[str, str]]) -> Side | None:
    """Which mode ``graph`` is the projection of, read back from its own node ids.

    A projection keeps its memberships but not which side it collapsed onto, and the answer is in
    the node ids: a graph of left nodes cannot contain a right id. ``None`` when the nodes are in
    neither mode entirely -- a graph that was filtered until nothing recognisable was left, or one
    carrying somebody else's pairs.
    """
    nodes = set(graph.nodes)
    if not nodes:
        return None
    if nodes <= {left for left, _ in pairs}:
        return "left"
    if nodes <= {right for _, right in pairs}:
        return "right"
    return None


# ----------------------------------------------------------------------------- §19.1 step four


@dataclass(frozen=True)
class Significance:
    """One observation against one null sample: §19.1's step four, with the null named.

    ``z`` is the book's "number of standard deviations between the observation and the null
    average"; ``p_value`` is the empirical p of §3.3, counting how often the null did at least as
    well. They answer the same question in different currencies and disagree exactly when the null
    distribution is not pseudo-normal, which is what ``caveat`` is for.
    """

    observed: float
    null: str
    """Which null produced the sample, for the report line that has to name it."""
    samples: int
    null_mean: float
    null_std: float
    z: float
    p_value: float
    tail: str
    summary: Summary | None
    """:func:`graphrag.sna.stats.describe` of the null sample, or ``None`` when there was none."""
    caveat: str

    @property
    def testable(self) -> bool:
        return self.samples > 0


def significance(
    observed: float,
    samples: Sequence[float],
    *,
    null: str = "",
    tail: str = "right",
) -> Significance:
    """Compare an observation with the null distribution of the same measure (§19.1, §3.3).

    ``samples`` is the measure computed on each null network, and ``null`` names the model that
    produced them -- ``configuration``, ``bipartite_preserving``, or a sentence -- because "three
    standard deviations above the null" means nothing until a reader knows what the null held
    fixed. :data:`HOLDS_FIXED` has that sentence for each sampler here.

    Both currencies are returned. The z-score is §19.1's own test and reads straightforwardly when
    the null histogram is the "nice normal or pseudo-normal distribution" the chapter's figures
    show; the empirical p-value does not need that assumption and is bounded below by
    ``1/(n+1)``, so it cannot claim more resolution than the sample size bought. ``caveat`` fires
    when :func:`graphrag.sna.stats.describe` finds the null sample heavy-tailed, in which case the
    z-score overstates the distance and the p-value is the one to quote.

    A null that could not be built at all -- too small a network to rewire -- is reported rather
    than raised: ``samples`` is 0, ``z`` is 0.0, ``p_value`` is ``nan`` and the caveat says the
    test did not happen. Printing the sample count beside the z-score is what keeps that case
    visible.
    """
    values = list(samples)
    if not values:
        return Significance(
            observed=observed,
            null=null,
            samples=0,
            null_mean=0.0,
            null_std=0.0,
            z=0.0,
            p_value=math.nan,
            tail=tail,
            summary=None,
            caveat=(
                "No null sample could be built, so nothing here was tested. The network is too "
                "small for this null to have a move to make; report the observation as a "
                "description and stop (§19.1)."
            ),
        )
    sample_summary = describe(values)
    return Significance(
        observed=observed,
        null=null,
        samples=len(values),
        null_mean=mean(values),
        null_std=std(values),
        z=z_score(observed, values),
        p_value=empirical_p(observed, values, tail=tail),
        tail=tail,
        summary=sample_summary,
        caveat=_significance_caveat(sample_summary),
    )


def _significance_caveat(summary: Summary) -> str:
    """The sentence a heavy-tailed null distribution owes the reader (§19.1, §3.1)."""
    if not summary.heavy_tailed:
        return ""
    return (
        "The null distribution is heavy-tailed, so the z-score is not the test §19.1 describes: "
        "that test counts standard deviations and assumes the 'nice normal or pseudo-normal' "
        "null of Figure 19.2, and on this sample the standard deviation is not a stable unit. "
        f"Quote the empirical p-value with its sample count instead. {summary.caveat}"
    )


#: The samplers, by the name a report prints. Each takes ``(observed, n, *, seed, ...)`` and
#: yields graphs; see :data:`HOLDS_FIXED` for what each one keeps.
NULLS: dict[str, Callable[..., Iterator[nx.Graph]]] = {
    "erdos_renyi": erdos_renyi,
    "configuration": configuration,
    "label_permutation": label_permutation,
    "bipartite_preserving": bipartite_preserving,
}


# ----------------------------------------------------------------------------- §19.2 ERGM


ERGMTermName = Literal["edges", "reciprocity", "two_stars", "three_stars", "triangles", "homophily"]

#: The configurations ``g`` this fit can carry (§19.2), in the order Figure 19.5 prints them.
#: ``edges`` is the intercept and is always in the model -- "this is always going to be present in
#: any ERGM". ``two_stars`` is the chain of three nodes and the degree term of the worked example
#: (Figure 19.4); ``three_stars`` is the figure's star of four nodes; ``triangles`` is the
#: transitivity term ``τT(A')``; ``reciprocity`` is the ``p₁R(A')`` of the directed model and is
#: refused on an undirected network; ``homophily`` is the attribute match of a p2-style model.
ERGM_TERMS: tuple[ERGMTermName, ...] = (
    "edges",
    "reciprocity",
    "two_stars",
    "three_stars",
    "triangles",
    "homophily",
)

#: What a fit carries when the caller does not choose: Figure 19.5's structural terms, plus the
#: reciprocity term on a directed network, where §19.2 puts it in the model explicitly.
DEFAULT_ERGM_TERMS: tuple[ERGMTermName, ...] = ("edges", "two_stars", "triangles")
DEFAULT_DIRECTED_ERGM_TERMS: tuple[ERGMTermName, ...] = (
    "edges",
    "reciprocity",
    "two_stars",
    "triangles",
)

#: Printed with every fit, because each one changes what the coefficients below may be used for.
ERGM_CAVEATS: tuple[str, ...] = (
    "This is MPLE, not MLE. §19.2's model is fitted here by maximum *pseudo*-likelihood: each "
    "dyad is regressed on how much it would change the model's statistics, as if the dyads were "
    "independent. They are not -- the triangle term exists precisely because they are not -- so "
    "the coefficients are biased and the standard errors are too small, both worse the more the "
    "structural terms matter. Modern practice is Monte Carlo MLE (§19.2, refs 10-11); that needs "
    "MCMC, which this package does not carry. Read the coefficients as directions and orders of "
    "magnitude, never as measurements.",
    "Models with a triangle term are prone to degeneracy. §19.2 warns that these models 'can be "
    "very difficult to solve analytically for all but the simplest networks' and that a dense "
    "network can need an exponentially large number of samples to estimate its betas; the "
    "failure mode behind that is a fitted model whose probability mass sits almost entirely on "
    "the empty or the complete graph, so it describes no observable network at all. A large "
    "positive triangle coefficient is as often a symptom of that as a finding about closure. "
    "Simulate from the fit before believing it.",
    "A fitted model is P(x|theta) maximised. The estimate is the parameter value under which the "
    "observed network is most probable; it is never a statement that the model is probable, that "
    "the network was generated this way, or that any configuration caused any other. §19.2's own "
    "reading of a coefficient is 'more (or less) than chance occurrence of the pattern', and "
    "'chance' there means the rest of this same model.",
)


@dataclass(frozen=True)
class ERGMTerm:
    """One configuration's coefficient, with the pseudo-likelihood's own (optimistic) error."""

    name: str
    coefficient: float
    std_error: float
    z: float
    p_value: float

    @property
    def reading(self) -> str:
        """§19.2's reading: positive is more than chance, negative less, near zero neither."""
        if not math.isfinite(self.z) or abs(self.z) < 2.0:
            return "not distinguishable from chance under this model"
        return (
            "more likely than chance under this model"
            if self.coefficient > 0
            else "less likely than chance under this model"
        )


@dataclass(frozen=True)
class ERGMFit:
    """An exponential random graph model fitted by maximum pseudo-likelihood (§19.2)."""

    terms: tuple[ERGMTerm, ...]
    nodes: int
    edges: int
    dyads: int
    """How many rows the regression had: one per ordered pair when the network is directed,
    one per unordered pair when it is not."""
    directed: bool
    attribute: str
    converged: bool
    caveats: tuple[str, ...] = ERGM_CAVEATS

    def term(self, name: str) -> ERGMTerm:
        """One term by name; raises ``KeyError`` if the fit did not carry it."""
        for term in self.terms:
            if term.name == name:
                return term
        msg = f"{name!r} is not in this fit; it carried {', '.join(t.name for t in self.terms)}"
        raise KeyError(msg)

    def coefficient(self, name: str) -> float:
        return self.term(name).coefficient


def ergm(
    graph: nx.Graph,
    *,
    terms: Sequence[str] | None = None,
    attribute: str | None = None,
) -> ERGMFit:
    """Fit §19.2's exponential random graph model by maximum pseudo-likelihood.

    The model of §19.2 is ``Pr(A = A') = (1/B)·exp(Σ_g β_g·g(A'))``: the probability of a network
    is set by how many of each configuration ``g`` it contains, and ``β_g`` says whether that
    configuration appears more or less than chance. Fitting it exactly needs MCMC. What is done
    here is the book's own intuition, taken literally: "this is sort of similar to estimating a
    logistic regression... you have a binary outcome (edge present/absent) and a set of variables
    that might be able to predict its value". Each dyad is one row -- every *unordered* pair when
    the network is undirected, every *ordered* pair when it is directed, because ``u -> v`` and
    ``v -> u`` are then two separate variables. The outcome is whether the edge is there; the
    predictors are the *change statistics*, how much each configuration count would rise if that
    edge were added:

    ``edges``
        1 for every dyad, so its coefficient is the intercept. Negative means a sparse network:
        "two nodes are unlikely to be connected" (§19.2's reading of -4.27 in Figure 19.5).
    ``reciprocity``
        1 when the reverse arc is already there, which is §19.2's ``p₁R(A')``: "if you have a
        directed graph you can represent reciprocity with the probability p₁ of a node to
        reciprocate the connection". Positive means arcs come back more often than the rest of
        the model accounts for. Directed networks only; on an undirected one it raises, because
        every edge is its own reverse (§6.2).
    ``two_stars``
        Figure 19.5's chain of three nodes, and the degree term of the worked example, where "the
        degree of a node influences its likelihood of getting a connection": the two endpoints'
        degrees, excluding the dyad itself. Directed, it is the sender's out-degree plus the
        receiver's in-degree -- how active the one is and how popular the other -- which is the
        directed reading of the same sentence.
    ``three_stars``
        Figure 19.5's star of four nodes: ``C(k_u, 2) + C(k_v, 2)`` over the same degrees, since
        an edge into a node of degree k completes that many new three-stars there.
    ``triangles``
        the transitivity term ``τT(A')``: how many neighbours the two endpoints already share, so
        adding the edge closes that many triangles. Positive means "when you have a triad, it is
        more likely than chance to have the third edge". Directed, it counts the two-paths
        ``u -> w -> v`` the new arc closes into transitive triples.
    ``homophily``
        1 when the two endpoints carry the same value of ``attribute`` (pass the node-data key).
        Requires ``attribute``; nodes with no value are treated as not matching anyone, including
        each other, because an absent tag is not a value.

    **What the book defines and what this adds.** §19.2 gives the directed case as
    ``p|E'| + p₁R(A')`` and stops: the edge and reciprocity terms are the book's, the ordered-dyad
    design is the book's, and the directed readings of ``two_stars``, ``three_stars`` and
    ``triangles`` above are *not* in the chapter -- they are the standard directed statistics,
    written down here so that the convention is arguable rather than implicit. A fit that uses
    them on a directed network is using more model than §19.2 specifies; the coefficients for
    ``edges`` and ``reciprocity`` are the ones the chapter will back you on.

    ``terms`` defaults to :data:`DEFAULT_ERGM_TERMS`, or :data:`DEFAULT_DIRECTED_ERGM_TERMS` on a
    directed network. ``edges`` is in every model whether or not it is asked for.

    Standard errors come from the logistic model's own information matrix and are therefore
    *pseudo*-likelihood errors: they assume the dyads are independent, which the whole point of a
    triangle term denies. They understate. Everything else that must be said about a fit is in
    :data:`ERGM_CAVEATS`, which travels on the result and which :func:`render_ergm` prints.

    Undefined, and raising ``ValueError``, when: the graph has fewer than three nodes (no dyad has
    a triangle to close); every dyad has the same outcome, so no logistic model exists (a complete
    or an empty graph); ``homophily`` was asked for without an attribute, or ``reciprocity``
    without a directed network; or the graph has more than :data:`MAX_ERGM_DYADS` dyads, where a
    dyad-level regression stops being §19.2 and starts being a data-processing job -- take an ego
    network or one component instead.
    """
    directed = graph.is_directed()
    chosen = terms if terms is not None else _default_terms(directed)
    wanted = _ergm_terms(chosen, attribute, directed)
    nodes = sorted(graph.nodes)
    if len(nodes) < 3:
        msg = "ergm() needs at least three nodes; with two there is one dyad and no configuration"
        raise ValueError(msg)
    dyads = len(nodes) * (len(nodes) - 1) if directed else len(nodes) * (len(nodes) - 1) // 2
    if dyads > MAX_ERGM_DYADS:
        msg = (
            f"ergm() would regress over {dyads:,} dyads, past the {MAX_ERGM_DYADS:,} limit. Fit "
            "an ego network, one component, or a filtered network instead (§19.2)."
        )
        raise ValueError(msg)
    matrix, order = adjacency(graph, nodes=nodes, weight=None)
    binary = np.asarray(matrix, dtype=np.float64)
    rows, columns = _ergm_dyads(len(order), directed)
    outcome = binary[rows, columns]
    if outcome.min() == outcome.max():
        msg = (
            "ergm() cannot fit a graph whose dyads are all present or all absent: a logistic "
            "model needs both outcomes (§19.2)."
        )
        raise ValueError(msg)
    design, names = _ergm_design(binary, rows, columns, order, wanted, graph, attribute, directed)
    # No penalty (``C`` infinite): a ridge would shrink the coefficients towards zero, and a
    # shrunken beta is not the maximum of anything the chapter describes.
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000, fit_intercept=True)
    model.fit(design, outcome)
    coefficients = np.concatenate([np.asarray(model.intercept_), np.asarray(model.coef_).ravel()])
    errors = _ergm_errors(design, np.asarray(model.predict_proba(design))[:, 1])
    fitted = tuple(
        _ergm_term(name, float(coefficient), float(error))
        for name, coefficient, error in zip(["edges", *names], coefficients, errors, strict=True)
    )
    return ERGMFit(
        terms=fitted,
        nodes=len(order),
        edges=graph.number_of_edges(),
        dyads=dyads,
        directed=directed,
        attribute=attribute or "",
        converged=bool(np.all(np.asarray(model.n_iter_) < model.max_iter)),
    )


def _default_terms(directed: bool) -> tuple[ERGMTermName, ...]:
    """Figure 19.5's structural terms, plus reciprocity where §19.2 puts it in the model."""
    return DEFAULT_DIRECTED_ERGM_TERMS if directed else DEFAULT_ERGM_TERMS


def _ergm_terms(terms: Sequence[str], attribute: str | None, directed: bool) -> tuple[str, ...]:
    """The terms to fit, with ``edges`` forced in and the impossible ones refused."""
    unknown = [term for term in terms if term not in ERGM_TERMS]
    if unknown:
        msg = f"unknown ERGM term(s) {', '.join(unknown)}; known terms are {', '.join(ERGM_TERMS)}"
        raise ValueError(msg)
    if "homophily" in terms and not attribute:
        msg = "the homophily term needs attribute=<node-data key> to match on"
        raise ValueError(msg)
    if "reciprocity" in terms and not directed:
        msg = (
            "the reciprocity term needs a directed network: in an undirected one every edge is "
            "its own reverse, so R(A') is the edge count again and the two terms are the same "
            "column (§19.2, §6.2)"
        )
        raise ValueError(msg)
    # The edge term is the intercept and §19.2 has it in every model, so asking for it is
    # optional and leaving it out is not.
    return tuple(term for term in ERGM_TERMS if term in terms and term != "edges")


def _ergm_dyads(size: int, directed: bool) -> tuple[np.ndarray, np.ndarray]:
    """The dyads to regress: every ordered pair when directed, the upper triangle when not.

    A directed network has two variables per pair of nodes, and collapsing them to one -- reading
    only the upper triangle -- would drop every reciprocated arc's second half and then report the
    result as if it had been one dyad. The self-pairs are out either way: a self-loop is not a
    configuration any of these terms is about.
    """
    if not directed:
        rows, columns = np.triu_indices(size, k=1)
        return rows, columns
    grid = np.arange(size)
    rows = np.repeat(grid, size)
    columns = np.tile(grid, size)
    off_diagonal = rows != columns
    return rows[off_diagonal], columns[off_diagonal]


def _ergm_design(
    binary: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    order: Sequence[str],
    terms: Sequence[str],
    graph: nx.Graph,
    attribute: str | None,
    directed: bool,
) -> tuple[np.ndarray, list[str]]:
    """One row per dyad: the change statistic of each configuration if that edge were added.

    Directed, "the degree of a node" splits in two, so the star terms read the sender's out-degree
    and the receiver's in-degree; undirected, both are the same degree. Either way the dyad's own
    edge is subtracted first, because a change statistic counts what the edge would *add*.
    """
    present = binary[rows, columns]
    out_degree = binary.sum(axis=1)
    in_degree = binary.sum(axis=0) if directed else out_degree
    sender = out_degree[rows] - present
    receiver = in_degree[columns] - present
    built: list[np.ndarray] = []
    names: list[str] = []
    for term in terms:
        if term == "reciprocity":
            built.append(binary[columns, rows])
        elif term == "two_stars":
            built.append(sender + receiver)
        elif term == "three_stars":
            built.append(_pairs_of(sender) + _pairs_of(receiver))
        elif term == "triangles":
            built.append((binary @ binary)[rows, columns])
        else:  # homophily
            key = attribute or ""
            values = [str(graph.nodes[node].get(key, "")).strip() for node in order]
            labels = np.array(values, dtype=object)
            matched = (labels[rows] == labels[columns]) & (labels[rows] != "")
            built.append(matched.astype(np.float64))
        names.append(term)
    if not built:
        return np.zeros((len(rows), 0), dtype=np.float64), names
    return np.column_stack(built), names


def _pairs_of(degrees: np.ndarray) -> np.ndarray:
    """``C(k, 2)``: how many three-stars an edge into a node of degree ``k`` completes there."""
    return degrees * (degrees - 1.0) / 2.0


def _ergm_errors(design: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Standard errors from the logistic information matrix, intercept first.

    ``sqrt(diag((XᵀWX)⁻¹))`` with ``W = p(1-p)``, the textbook logistic errors -- and, under a
    pseudo-likelihood, the optimistic ones: see :data:`ERGM_CAVEATS`. A pseudo-inverse is used so
    that a collinear design (a term that never varies) returns an error rather than raising.
    """
    with_intercept = np.column_stack([np.ones(len(design)), design])
    weights = np.clip(probabilities * (1.0 - probabilities), 1e-12, None)
    information = with_intercept.T @ (with_intercept * weights[:, None])
    covariance = np.linalg.pinv(information)
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _ergm_term(name: str, coefficient: float, error: float) -> ERGMTerm:
    """One coefficient with its Wald z and two-sided p, or no test when it has no usable error."""
    if error <= 0 or not math.isfinite(error):
        return ERGMTerm(name, coefficient, math.nan, math.nan, math.nan)
    z = coefficient / error
    return ERGMTerm(name, coefficient, error, z, float(2.0 * scipy_stats.norm.sf(abs(z))))


def render_ergm(fit: ERGMFit, graph: nx.Graph | None = None) -> list[str]:
    """The fit as report lines: the frame first, the coefficients second, the warnings with them.

    The warnings are printed with every fit rather than kept for a footnote, because each one
    changes what a coefficient may be used for.

    ``graph`` is the network that was fitted, and is only read for its sampling frame. Pass it and
    the frame line is that network's own -- the persona and filters a builder in
    :mod:`graphrag.sna.export` recorded, or a legendary graph's published one. Leave it out, or
    pass a graph that carries none, and the line says no frame was recorded rather than inventing
    one: a fit of somebody else's graph is not a fit of this corpus, and a report that claimed it
    was would be wrong in the one place a reader trusts.
    """
    pair, tie = ("ordered pair", "arcs") if fit.directed else ("unordered pair", "edges")
    lines = [
        "## Exponential random graph (MPLE)",
        "",
        f"**Sampling frame.** Every {pair} of the {fit.nodes:,} nodes in this network is one "
        f"observation: n = {fit.dyads:,} dyads, of which {fit.edges:,} are {tie}. "
        + _frame_sentence(graph),
        "",
        "**Null model.** None, and that is what separates this section from every other one "
        "here: §19.2 fits a model rather than testing against a family of shuffles. Each "
        "coefficient is read against the rest of this same model -- 'more than chance' means "
        "more than the other terms already account for.",
        "",
        "**Implements.** §19.2 (Exponential Random Graphs), by maximum pseudo-likelihood."
        + (
            " The chapter's directed model is `p|E'| + p₁R(A')`: edges and reciprocity. Any "
            "star or triangle term on this directed fit is the standard directed statistic, "
            "not the chapter's, and is more model than §19.2 specifies."
            if fit.directed and any(term.name not in ("edges", "reciprocity") for term in fit.terms)
            else ""
        ),
        "",
        "| configuration | beta | std. error | z | p | reading |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {term.name} | {term.coefficient:.4f} | {term.std_error:.4f} | {term.z:.2f} | "
        f"{term.p_value:.4f} | {term.reading} |"
        for term in fit.terms
    ]
    lines += ["", f"Converged: {'yes' if fit.converged else 'no'}.", ""]
    lines += [f"- {caveat}" for caveat in fit.caveats]
    lines += [""]
    return lines


def _frame_sentence(graph: nx.Graph | None) -> str:
    """What this network is a network *of*, from the graph itself, or an admission that it is not
    recorded.

    A network built by :mod:`graphrag.sna.export` carries ``frame`` and ``persona_id``; a
    legendary graph carries ``frame`` alone; a graph somebody built in a notebook carries neither,
    and for that one the only honest sentence is that nobody wrote the frame down.
    """
    if graph is None:
        return "No sampling frame was recorded: this fit was handed a graph and not a corpus."
    frame = str(graph.graph.get("frame", "")).strip()
    persona = str(graph.graph.get("persona_id", "")).strip()
    if frame and persona:
        return (
            f"{frame} Built from persona `{persona}`, so it describes what was written down and "
            "never a population."
        )
    if frame:
        return frame
    return "No sampling frame was recorded on this network, so say where it came from by hand."
