"""Network backboning (ch. 27): deciding which edges of a too-dense network are worth keeping.

*"Network backboning is the problem of taking a network that is too dense and removing the
connections that are likely to be not significant -- or 'strong enough'. If you ever found
yourself in a situation thinking 'there are too many edges in this network, I'm going to filter
some out', then you performed network backboning"* (p. 381). Every network this package builds
is a co-occurrence network, so every one of them is dense by construction and has been filtered
by ``--min-weight`` since the beginning. That filter is §27.1, the naive threshold, and the book
spends the rest of the chapter explaining why it is the weakest option available.

**What backboning is not.** It is not summarization (pp. 381-382): nodes are never merged and as
many as possible are kept, because the point is to let the strong connections emerge while
every entity in the data can still be described individually. Chapter 46 (``sna summarize``,
ATL-46) is the other task. That is why :func:`backbone` removes edges only, and why
``keep_isolates`` decides the fate of a node the filter stranded rather than of a node that
arrived alone.

**Structural against statistical.** The book splits the methods in two (p. 382). The structural
ones -- ``naive`` (§27.1), ``doubly-stochastic`` (§27.2), ``high-salience`` (§27.3), ``convex``
(§27.4), plus ``mst`` and ``pmfg`` from §13.4 -- re-weight or re-select edges from the topology
and then threshold; every surviving edge carries the ``score`` it was selected on. The
statistical ones -- ``disparity`` (§27.5) and ``noise-corrected`` (§27.6) -- state a null model,
compute a p-value per edge and keep the edges least likely to have happened by chance; every
surviving edge carries its ``p_value``. A backbone is therefore never anonymous: the method, the
level, the null model and how much of the network survived are recorded in ``graph.graph`` and
printed in the frame.

**Which to reach for here.** Every weight this package produces is a count of shared documents
or shared passages, which is exactly the discrete-count case §27.6 was written for, so
``noise-corrected`` is the recommended default. The disparity filter keeps an edge when *either*
endpoint finds it significant, and the book is explicit about what that does to the result: it
"tends to create networks with high centralization, broad degree distributions, and weak
communities" (p. 391), because a hub's weakest link is still the strongest thing that ever
happened to the small node at the other end. Running Louvain over a DF backbone therefore
measures the filter as much as the corpus. NC asks both endpoints to agree and is "the most
punishing method for the central hubs" (p. 393), which is the bias one wants when the next
question is about communities.

None of this is free: a backbone changes the sampling frame. The network after it describes the
edges that survived a stated test, not the corpus, and the method has to be named next to every
number that follows it.

**What the chapter cites and this module does not build.** p. 382 says the noise-corrected urn
has a family around it and p. 391 that the disparity filter has better-behaved relatives; none of
the five is implemented, and the reason is different in each case:

``Tumminello et al.``, statistically validated networks in bipartite systems (p. 382)
    a hypergeometric test on the two-mode network *before* projecting. It is the right shape for
    the speakers-entities network, but it belongs with the projections of chapter 26 (ATL-26),
    not here: it filters the projection as it is built rather than afterwards.
``Marcaccioli and Livan``, the Pólya-urn filter (p. 382)
    the same urn as §27.6 with reinforcement, which fits a network where observing an edge makes
    the next observation of it more likely. Whether a corpus behaves that way is an empirical
    question nobody has asked of these networks, and a filter whose reinforcement parameter
    nobody can motivate is §27.1's problem again with more arithmetic.
``Radicchi et al.``, information filtering by a null version of the network (p. 382)
    generate a null network and test the observed weights against it. ATL-19 is building the
    named null models this package will use; doing it here would fix one null before that
    contract exists.
``Dianati``'s marginal-likelihood filter and ``Gemmetto, Cardillo and Garlaschelli``'s maximum-
entropy one (p. 391)
    the "collection of alternatives [that] take this additional piece of information into
    account and are thus less biased" than the disparity filter. They are less biased than DF in
    the same direction §27.6 is, and the book recommends NC for count weights, so they would add
    a third answer to a question ``noise-corrected`` already answers. Worth having if a network
    ever turns up here whose weights are not counts.

Also not built, deliberately: §13.4's **triangulated** maximally filtered graph, which the book
names beside the PMFG without giving a construction.

**Directed networks.** The ``relations`` network is a digraph, and the book defines the directed
form of both statistical methods: §27.5 tests an edge "either when compared to the
out-connections of the node sending the edge, or when compared to the in-connection weights of
the node receiving it" (p. 391), and §27.6 notes that "if your network is directed, w_uv != w_vu
and you'll get a different null expectation for either direction of the edge" (p. 392). Both are
implemented, as are the two naive thresholds, which never read a neighbourhood. The remaining
structural methods refuse a digraph and say whether the book defines a directed form at all --
see ``directed_note`` on each :class:`BackboneMethod`, which the refusal quotes.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from graphrag.sna.matrices import SPARSE_ABOVE_NODES, adjacency, dense, scale_to_unit_sums
from graphrag.sna.stats import CORRECTIONS, Summary, binomial_p, correct, describe

type Node = Any
"""A node id. This package's networks use ``str``; the legendary graphs use ``int``."""

type Edge = tuple[Node, Node]
"""One undirected edge, in the orientation ``graph.edges()`` yielded it."""


@dataclass(frozen=True)
class BackboneMethod:
    """One method of chapter 27, with the section it comes from and the null it assumes."""

    name: str
    section: str
    statistical: bool
    null: str
    """The null model (statistical methods) or the selection criterion (structural ones), in
    words, printed next to the numbers because a filtered network without one is a number."""
    scored: str
    """What the number left on every surviving edge means."""
    directed: bool
    """Whether this method accepts an :class:`networkx.DiGraph`."""
    directed_note: str
    """What the book says about the directed case: the definition used when there is one, the
    reason there is none when there is not. The refusal quotes it, so a caller is never told
    merely that something is unsupported."""


#: Every method, in the order the book presents them: §27.1-27.6, then §13.4.
BACKBONE_METHODS: tuple[BackboneMethod, ...] = (
    BackboneMethod(
        name="naive",
        section="§27.1",
        statistical=False,
        null="none: a single hard weight threshold, the same everywhere in the network",
        scored="score = the edge's own weight, which is what it was kept for",
        directed=True,
        directed_note=(
            "§27.1 thresholds a weight and reads no neighbourhood, so it applies to a directed "
            "edge unchanged: u->v and v->u are two edges, each judged on its own weight"
        ),
    ),
    BackboneMethod(
        name="naive-top",
        section="§27.1",
        statistical=False,
        null=(
            "none: every node keeps its n strongest edges, so the threshold is per node rather "
            "than global -- at the price of fixing the network's minimum degree at n"
        ),
        scored="score = the edge's own weight, which is what put it in somebody's top n",
        directed=True,
        directed_note=(
            "the book defines no directed form. The convention here is each node's n strongest "
            "*incident* edges whichever way they run, so a node's strongest tie survives in "
            "either direction; counting out-edges alone would strand every node that only "
            "receives"
        ),
    ),
    BackboneMethod(
        name="doubly-stochastic",
        section="§27.2",
        statistical=False,
        null=(
            "none: the adjacency matrix is normalised until rows and columns both sum to 1, "
            "which puts every edge on one scale, and the threshold is the highest one that "
            "leaves the network in as few components as it started with (Slater's rule)"
        ),
        scored="score = the edge's entry in the doubly stochastic matrix",
        directed=False,
        directed_note=(
            "Sinkhorn-Knopp is defined for any non-negative square matrix, so the arithmetic "
            "would run on a directed adjacency, but §27.2 discusses only the undirected case "
            "and its stopping rule is about undirected connectivity. Not guessed at"
        ),
    ),
    BackboneMethod(
        name="high-salience",
        section="§27.3",
        statistical=False,
        null=(
            "none: each node builds its own shortest-path tree over distance = 1/weight, and an "
            "edge is scored by the share of those trees it appears in"
        ),
        scored="score = the share of the network's shortest-path trees using this edge, in [0, 1]",
        directed=False,
        directed_note=(
            "§27.3 builds one shortest-path tree per node, and a directed network has two kinds "
            "-- the tree of paths out of a node and the tree of paths into it -- which the book "
            "does not choose between"
        ),
    ),
    BackboneMethod(
        name="convex",
        section="§27.4",
        statistical=False,
        null=(
            "none: edges are removed until the network is a tree of cliques, the shape in which "
            "every connected induced subgraph holds all the shortest paths between its nodes"
        ),
        scored="score = the size of the clique (block) the edge ended up in; 2 for a bridge",
        directed=False,
        directed_note=(
            "§27.4's convexity is Harary and Nieminen's, defined over the shortest paths of an "
            "undirected graph; directed convexity is a separate literature the book does not "
            "cite"
        ),
    ),
    BackboneMethod(
        name="disparity",
        section="§27.5",
        statistical=True,
        null=(
            "node-centric: each node's edge weights are spread uniformly at random over its "
            "neighbours, and an edge is kept when it is unlikely under that null from *either* "
            "endpoint's point of view"
        ),
        scored="p = the smaller of the two endpoints' p-values, the one that kept the edge",
        directed=True,
        directed_note=(
            "§27.5 defines it (p. 391): the edge is kept when it is significant against the "
            "out-connections of the node sending it, or against the in-connection weights of "
            "the node receiving it"
        ),
    ),
    BackboneMethod(
        name="noise-corrected",
        section="§27.6",
        statistical=True,
        null=(
            "edge-centric binomial: the network's total edge weight is drawn from an urn in "
            "which the chance of landing on (u, v) is the product of the two nodes' strengths "
            "over the square of that total, so the expected weight is the configuration-model "
            "one and both endpoints have to agree that the observed weight beats it"
        ),
        scored="p = P(a weight at least this large | the binomial null), identical from both ends",
        directed=True,
        directed_note=(
            "§27.6 defines it (p. 392): u->v and v->u are different edges with different null "
            "expectations, built from the sender's out-strength and the receiver's in-strength"
        ),
    ),
    BackboneMethod(
        name="mst",
        section="§13.4",
        statistical=False,
        null=(
            "none: the maximum spanning tree, the acyclic subgraph that touches every node with "
            "the largest total weight. Weights here are proximities, so the maximum is the one "
            "to want"
        ),
        scored="score = the edge's own weight",
        directed=False,
        directed_note=(
            "§13.4's spanning tree is defined for an undirected graph; the directed analogue is "
            "the optimum branching (arborescence), which the Atlas puts in chapter 33 with "
            "hierarchies and which now exists as graphrag.sna.hierarchy.arborescence -- a "
            "maximum spanning arborescence by Chu-Liu/Edmonds, reported as §33.4's "
            "hierarchicalness score rather than offered here as a backbone"
        ),
    ),
    BackboneMethod(
        name="pmfg",
        section="§13.4",
        statistical=False,
        null=(
            "none: edges are taken heaviest first and kept while the graph stays planar, up to "
            "the 3(n-2) edges a planar graph can hold"
        ),
        scored="score = the edge's own weight",
        directed=False,
        directed_note=(
            "planarity is a property of the underlying undirected graph, and §13.4 defines the "
            "PMFG for weighted undirected networks"
        ),
    ),
)

#: The method names, for a CLI option and for validation.
BACKBONES: tuple[str, ...] = tuple(method.name for method in BACKBONE_METHODS)

_BY_NAME: dict[str, BackboneMethod] = {method.name: method for method in BACKBONE_METHODS}

#: The significance level the book's two statistical methods are usually run at.
DEFAULT_ALPHA = 0.05

#: How many edges each node keeps under ``naive-top`` when no ``threshold`` is given. The book
#: names no n; 1 is the mildest form of the objection on p. 384, since every node that had an
#: edge keeps exactly its strongest one and the minimum degree it imposes is the smallest there
#: is. Any n is a choice, and the report says which one was made.
NAIVE_TOP_DEFAULT = 1

#: The book's own cap on Sinkhorn-Knopp normalisation attempts (§27.8 exercise 3): "if by 1,000
#: normalizations you don't have a doubly stochastic matrix, the calculation didn't converge".
SINKHORN_ITERATIONS = 1000

#: How close to 1 every row and column sum has to be before the normalisation counts as done.
SINKHORN_TOLERANCE = 1e-9

#: The default high-salience cut. §27.3's "horns" plot has a peak at 0 and a peak at 1 with
#: almost nothing between them, which is what makes HSS "almost parameter free": any threshold
#: in the empty middle selects the same skeleton. 0.5 is the middle of it. Grady et al. use 0.8.
SALIENCE_THRESHOLD = 0.5

#: Above this many nodes, :func:`high_salience` draws its Dijkstra sources from a sample rather
#: than paying for one shortest-path tree per node (ATL-F2). §27.3 states the cost -- "it
#: requires a lot of shortest path calculations -- which makes it computationally expensive"
#: (p. 387) -- but names no sampling procedure of its own; this is the same node count
#: :data:`graphrag.sna.matrices.SPARSE_ABOVE_NODES` already uses for switching this package's
#: dense linear algebra to a sparse path (ATL-ENT-3), reused here for the same reason: it is the
#: scale at which this package's own networks stop being fast to compute over in full, not a
#: number this method's chapter picked.
SALIENCE_SAMPLE_ABOVE_NODES = SPARSE_ABOVE_NODES


# ----------------------------------------------------------------------------- §27.1 naive


def prune_below(graph: nx.Graph, min_weight: int) -> nx.Graph:
    """Drop edges below ``min_weight`` **in place**, then the nodes that this isolates (§27.1).

    This is what ``--min-weight`` has always done and what every builder calls, kept here so
    the naive threshold lives beside the methods that answer it rather than in the exporter.
    It mutates and returns the graph it was given; :func:`backbone` works on a copy.

    The book's two objections to it, which its help text carries: real edge weights distribute
    broadly -- in the book's own projected Twitter network "82% of the edges have weight equal
    to one. The smallest possible hard threshold would remove 82% of the network, without
    allowing for any nuance" -- and such a distribution "lacks of a well-defined average value
    and has undefined variance", so "you cannot motivate your threshold choice by saying that it
    is 'x standard deviations from the average'" (p. 383). Edge weights are also locally
    correlated (pp. 384-385), so one threshold for the whole network flattens the dense corner
    and leaves the sparse one untouched.

    ``min_weight`` of 1 or less is a no-op, including on isolates: a speaker who shares no
    document with anyone is in the network on purpose.
    """
    if min_weight <= 1:
        return graph
    weak = [(u, v) for u, v, w in graph.edges(data="weight") if (w or 1) < min_weight]
    graph.remove_edges_from(weak)
    graph.remove_nodes_from([n for n, degree in graph.degree() if degree == 0])
    return graph


def _naive(
    graph: nx.Graph, edges: Sequence[Edge], threshold: float
) -> dict[Edge, dict[str, float]]:
    """§27.1 on a candidate edge set: keep the edges whose weight clears one global threshold."""
    kept: dict[Edge, dict[str, float]] = {}
    for u, v in edges:
        weight = _weight(graph, u, v)
        if weight >= threshold:
            kept[(u, v)] = {"score": weight}
    return kept


def _naive_top(graph: nx.Graph, edges: Sequence[Edge], n: int) -> dict[Edge, dict[str, float]]:
    """§27.1's other naive strategy: "simply pick the top n strongest connections for each node".

    The book offers it as the repair for the objections to a global threshold, and it is one:
    the threshold becomes local, so a node whose edges are all light keeps its own strongest
    ones instead of losing everything to the dense corner's scale (pp. 383-384). Then it takes
    the offer back. "However, by applying it you're effectively determining the minimum degree
    of the network to be n. This is a heinous crime against the God of power law degree
    distributions and, if you commit it, you will be tormented by scale free demons in network
    hell for all eternity" (p. 384).

    That is a joke about a real defect, and it is the reason this method is here rather than
    recommended. A corpus network's degree distribution is its most reported property (§9.3);
    cutting its left tail off at n makes every node look at least n-connected, so the degree
    distribution, the degree assortativity and anything read off the low-degree end of the
    network afterwards describe this filter. Whoever runs it should report the degree
    distribution of the backbone *and* of the network it came from.

    The rule is a union, not an intersection: an edge survives if it is in *either* endpoint's
    top n, so a node can end with more than n edges, and only a node that already had n or more
    ends with exactly n. Ties are broken by the neighbour's id, so two runs agree.
    """
    kept: dict[Edge, dict[str, float]] = {}
    # Incident, not outgoing: on a directed network a node's strongest tie counts whichever way
    # it runs, so a node that only ever receives keeps its edges too. The book defines no
    # directed form of this one; that is the convention chosen, and ``directed_note`` says so.
    incident: dict[Node, list[Edge]] = defaultdict(list)
    for edge in edges:
        incident[edge[0]].append(edge)
        incident[edge[1]].append(edge)
    keep = max(1, int(n))
    for candidates in incident.values():
        candidates.sort(key=lambda edge: (-_weight(graph, *edge), _key(edge)))
        for edge in candidates[:keep]:
            kept[edge] = {"score": _weight(graph, *edge)}
    return kept


# ------------------------------------------------------------------- §27.2 doubly stochastic


def doubly_stochastic_scores(
    graph: nx.Graph, *, sparse_above: int = SPARSE_ABOVE_NODES
) -> tuple[dict[Edge, float], bool]:
    """The doubly stochastic entry of every edge, and whether the normalisation converged (§27.2).

    Sinkhorn-Knopp: normalise the rows to sum to 1, then the columns, and repeat. "After you
    perform such normalization, the scale of all edges is the same, and you break local
    correlations" (p. 385), which is the one objection to §27.1 that a global threshold can
    actually answer.

    The second flag is not decoration. "Only strictly positive matrices can" be made doubly
    stochastic, and "since real world networks are sparse, they actually contain lots of zeros.
    So this solution cannot be always applied" (p. 385). A network whose normalisation has not
    settled after :data:`SINKHORN_ITERATIONS` passes returns ``False`` and the report says so
    next to the numbers rather than quietly thresholding scores that mean nothing. The book's
    escape -- adding a small epsilon everywhere -- is not taken here: as epsilon goes to 0 the
    normalisation of ``A + epsilon`` does not converge to the normalisation of ``A`` (p. 386),
    so the answer would depend on a number nobody could motivate.

    **On a two-mode network this method does not apply at all.** p. 386 proves it in one line: if
    every row sums to one then the entries sum to the number of rows, and if every column does
    then they sum to the number of columns, so "a doubly stochastic matrix must be square" and
    "you cannot apply the doubly stochastic backboning to bipartite networks, unless
    ``|V1| = |V2|``". What is normalised here is the square all-node adjacency, not the
    ``|V1| x |V2|`` biadjacency, so the arithmetic runs -- but the convergence it would have to
    reach does not exist for the object the book means, and in practice the iteration fails: a
    two-mode network's adjacency has a zero block on each mode's diagonal and no positive
    diagonal to converge onto. The failure is reported with that reason rather than the generic
    sparsity one, because it is not bad luck about this corpus, it is the shape of the network.

    Undefined for a node with no edges: its row cannot be scaled to 1. Such rows are left at
    zero, which makes the matrix sub-stochastic in those rows and is the honest statement of it.

    ``sparse_above`` picks the linear-algebra path (ATL-F2): a dense ``numpy`` array below it,
    a ``scipy.sparse`` one above -- the same threshold and the same reason
    :func:`graphrag.sna.matrices.eigenpairs` already switches on (ATL-ENT-3), since diagonal
    scaling costs a sparse matrix nothing a dense one does not also pay for, but a dense
    ``n x n`` array does not fit comfortably once ``n`` is in the thousands. Both paths run the
    identical formula through :func:`graphrag.sna.matrices.scale_to_unit_sums`, so a network
    below the threshold gets exactly the numbers it always did; a test asserts the two paths
    agree on a network forced above it.
    """
    sparse = graph.number_of_nodes() > sparse_above
    matrix, order = adjacency(graph, sparse=sparse)
    current = matrix if sparse else dense(matrix).copy()
    index = {node: position for position, node in enumerate(order)}
    converged = False
    for _ in range(SINKHORN_ITERATIONS):
        current = scale_to_unit_sums(current, axis=1)
        current = scale_to_unit_sums(current, axis=0)
        # ``.sum(axis=)`` on either array type, without densifying: the O(n x n) array a sparse
        # network is being routed here to avoid building would otherwise get rebuilt every one
        # of up to :data:`SINKHORN_ITERATIONS` passes just to check whether it converged.
        rows = np.asarray(current.sum(axis=1)).ravel()
        columns = np.asarray(current.sum(axis=0)).ravel()
        live = rows > 0
        if bool(
            np.all(np.abs(rows[live] - 1.0) < SINKHORN_TOLERANCE)
            and np.all(np.abs(columns[live] - 1.0) < SINKHORN_TOLERANCE)
        ):
            converged = True
            break
    return {
        (u, v): float(current[index[u], index[v]]) for u, v in graph.edges() if u != v
    }, converged


def _components(graph: nx.Graph) -> int:
    """How many pieces the network is in, weakly for a digraph.

    Weak rather than strong because the question every caller here asks is whether the backbone
    left the network in more pieces than it found it, and a directed corpus network is almost
    never strongly connected to begin with -- reciprocity is low, so strong components would
    count nearly every node as its own piece and the number would say nothing about the filter.
    """
    if graph.is_directed():
        return int(nx.number_weakly_connected_components(graph))
    return int(nx.number_connected_components(graph))


def _connectivity_threshold(scores: dict[Edge, float], graph: nx.Graph) -> float:
    """The highest score at which the backbone still has as few components as the network did.

    §27.2's own guideline: "You should pick the threshold that allows your graph to be a single
    connected component", and §27.8's exercise 4 asks for the backbone "including all nodes in
    the network in a single (weakly) connected component with the minimum number of edges". A
    network that arrived in several components cannot be made into one by removing edges, so
    the rule generalises to "no more components than it started with".
    """
    target = _components(graph)
    parent = {node: node for node in graph}
    components = graph.number_of_nodes()
    threshold = 0.0
    for edge, score in sorted(scores.items(), key=lambda item: (-item[1], _key(item[0]))):
        if _union(parent, *edge):
            components -= 1
            threshold = score
            if components <= target:
                break
    return threshold


# -------------------------------------------------------------------- §27.3 high salience


def high_salience(
    graph: nx.Graph,
    *,
    sources: Sequence[Node] | None = None,
    sample: int | None = None,
    seed: int | None = None,
    sample_above: int = SALIENCE_SAMPLE_ABOVE_NODES,
) -> tuple[dict[Edge, float], str]:
    """The share of the network's shortest-path trees each edge appears in (§27.3), in [0, 1].

    "To build an HSS we loop over the nodes and we build their shortest path tree: a tree
    originating from a node, touching all other nodes in the minimum number of hops possible and
    maximum amount of edge weight possible" (p. 386). The trees are summed, and an edge's score
    is the fraction of nodes whose tree uses it.

    **The cost function.** Our weights are proximities -- a weight of 6 means "shared six
    documents" -- so a heavier edge has to be a *shorter* hop for a path to prefer it. Every
    shortest path here is therefore computed over ``distance = 1/weight``, the same cost
    ``measures._with_distance`` writes for closeness and betweenness. An edge of weight 6
    costs 1/6 of an edge of weight 1, which is a choice and not a fact: it makes two shared
    documents exactly twice as close as one, and a different convention (``max(w) - w``, say)
    would give a different skeleton.

    This is the expensive method in the chapter -- one Dijkstra per node, so O(n(m + n log n))
    -- and the book says as much (p. 387). It differs from edge betweenness on purpose: a tree
    cannot contain a cycle, so each node's view is forced to choose, and the score counts what
    is salient "from each node's local perspective, rather than the network's global
    perspective" (p. 387).

    **Sampling the sources (ATL-F2, not in the book).** §27.3 states the cost and offers no
    remedy for it. Above ``sample_above`` nodes -- :data:`SALIENCE_SAMPLE_ABOVE_NODES` by
    default -- the trees are built from a sample of sources rather than every node, chosen
    degree-stratified (:func:`_degree_stratified_sample`) rather than uniformly at random,
    because §29.1's own warning about plain random node samples applies here word for word:
    "unbiased in the mean but, on a heavy tail, unlikely to catch a hub", and a hub's tree is
    exactly the one whose absence would most distort which edges look salient. The idea itself is
    the one Brandes and Pich use to approximate betweenness at scale -- a handful of pivots
    standing in for every source -- carried over to HSS's per-source tree count, which is a
    different measure but the identical cost shape (one traversal per source).

    The score is a **stratified mean**, not ``count / len(sources)``: the sample is split into
    bands by degree, one source drawn from each, and each drawn source's tree is weighted by its
    own band's share of all ``n`` nodes (``_degree_stratified_sample`` returns those weights, and
    they always sum to 1). Weighting by the *sample* share instead -- crediting every draw
    ``1 / len(sources)`` regardless of how large a band it stands for -- is only unbiased when
    every band holds exactly the same number of nodes, which an ``n`` that does not divide evenly
    by the sample size never gives; the stratified mean is unbiased for §27.3's population share
    whatever the split, which is what makes the sample size a free choice rather than one that
    has to divide ``n``. It is exact, not merely close, when ``sources`` covers every node -- so a
    network below the threshold, or a caller who passes every node explicitly, gets the identical
    number this function always returned. ``sources`` overrides the automatic choice outright (an
    explicit source list, e.g. for a caller who already sampled its own, or a test): such a list
    carries no band weights of its own, so it is scored as a plain mean, ``count / len(sources)``,
    which is what the caller should want unless it already knows better. ``sample`` fixes the
    sample size (below or above the threshold, forcing or skipping the sampling regardless of
    ``n``) and is what tests use to exercise this without a multi-thousand-node fixture. Both keep
    ``seed``, which is otherwise the degree-stratified draw's only source of randomness.

    The second return value is empty when nothing was sampled, and otherwise the sentence the
    report prints next to the scores: how many of how many nodes were used, the method and the
    seed.

    **On a network in several components this measure is not comparable across them.** The
    divisor is every node, as §27.3 defines it, but a tree rooted outside an edge's component
    cannot use that edge, so an edge in a component holding a tenth of the nodes can score at
    most 0.1 and no threshold worth the name will keep it. A corpus network that fragmented
    under ``--min-weight`` will therefore lose its small components entirely. Read the component
    count first; ``sna backbone`` prints it per method for this reason.

    Undefined for a graph with fewer than two nodes; it returns an empty mapping, since a node
    on its own has no tree to build.
    """
    canonical = _canonical(graph)
    nodes = list(graph)
    total = len(nodes)
    weights: dict[Node, float] | None
    if sources is not None:
        chosen = list(sources)
        weights = None  # a caller-given list carries no band of its own: scored as a plain mean
        method = "the sources given by the caller"
    elif sample is not None and sample < total:
        chosen, weights = _degree_stratified_sample(graph, sample, seed)
        method = "a degree-stratified sample"
    elif sample is None and total > sample_above:
        chosen, weights = _degree_stratified_sample(graph, sample_above, seed)
        method = "a degree-stratified sample"
    else:
        chosen = nodes
        weights = None
        method = ""
    if not chosen:
        return {}, ""
    scores: dict[Edge, float]
    if weights is not None:
        # Weighted (bands): sum each drawn source's tree, scaled by its band's population share.
        scores = dict.fromkeys(canonical.values(), 0.0)
        for source in chosen:
            weight = weights[source]
            for edge in _tree(graph, canonical, source):
                scores[edge] += weight
    else:
        # Unweighted: an integer count per edge, divided once at the end -- exact, the same
        # arithmetic §27.3's own count/n always used, rather than len(chosen) additions of
        # 1/len(chosen) whose rounding error a bridge's score of exactly 1.0 would otherwise show.
        counts: dict[Edge, int] = dict.fromkeys(canonical.values(), 0)
        for source in chosen:
            for edge in _tree(graph, canonical, source):
                counts[edge] += 1
        divisor = float(len(chosen))
        scores = {edge: count / divisor for edge, count in counts.items()}
    note = _salience_note(len(chosen), total, method, seed) if method else ""
    return scores, note


def _tree(graph: nx.Graph, canonical: dict[Edge, Edge], source: Node) -> set[Edge]:
    """One node's shortest-path tree, as the set of canonical edges it uses (§27.3).

    A tree, not a bag of paths: an edge near the root is on every path that leaves through it,
    and counting those separately would score it above 1. The union of the paths is the tree.
    """
    paths = nx.single_source_dijkstra_path(graph, source, weight=_distance)
    tree: set[Edge] = set()
    for target, path in paths.items():
        if target == source:
            continue
        tree.update(canonical[(path[step], path[step + 1])] for step in range(len(path) - 1))
    return tree


def _degree_stratified_sample(
    graph: nx.Graph, k: int, seed: int | None
) -> tuple[list[Node], dict[Node, float]]:
    """``k`` source nodes stratified by degree, and the weight each carries in a stratified mean.

    Not the book's own idea (§27.3 gives none): plain uniform-random nodes are, per §29.1,
    "unbiased in the mean but, on a heavy tail, unlikely to catch a hub" -- and a hub's
    shortest-path tree is exactly the one whose absence would most distort which edges the
    sample calls salient. Nodes are ranked by degree and cut into ``k`` bands from the top down,
    sized ``n // k`` or ``n // k + 1`` so that every band holds at least one node whenever
    ``k <= n`` -- never rounded to zero, so there is nothing to top up afterwards -- and exactly
    one node is drawn per band.

    The second return value is that draw's weight in a stratified mean of the population: the
    band's size over ``n``, which is what makes ``sum(weights.values()) == 1.0`` regardless of
    how unevenly ``n`` splits into ``k`` bands. Crediting every draw the same ``1 / k`` instead
    is only unbiased when the bands are exactly equal in size, which an arbitrary ``k`` almost
    never gives; the size-weighted version is unbiased for whatever the split turns out to be.

    Deterministic given ``seed``; a tie in degree is broken by node id so two runs agree on the
    ranking. Bands are disjoint slices of one sorted list, so the ``k`` nodes drawn are always
    distinct.
    """
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    ordered = sorted(graph.nodes, key=lambda n: (-graph.degree(n), _key(n)))
    n = len(ordered)
    k = max(0, min(k, n))
    if k == 0:
        return [], {}
    base, extra = divmod(n, k)
    sizes = [base + 1] * extra + [base] * (k - extra)  # every size >= 1 since k <= n
    chosen: list[Node] = []
    weights: dict[Node, float] = {}
    start = 0
    for size in sizes:
        stop = start + size
        node = ordered[rng.randrange(start, stop)]
        chosen.append(node)
        weights[node] = size / n
        start = stop
    return chosen, weights


def _salience_note(used: int, total: int, method: str, seed: int | None) -> str:
    """The sentence a sampled :func:`high_salience` run prints next to its scores."""
    share = used / total if total else 0.0
    seeded = "unseeded" if seed is None else f"seed {seed}"
    return (
        f"high-salience computed from {used:,} of {total:,} nodes' shortest-path trees "
        f"({share:.1%}), {method} ({seeded}), above the "
        f"{SALIENCE_SAMPLE_ABOVE_NODES:,}-node threshold this method samples at by default -- "
        "not in the book (§27.3 states the cost and offers no sampling procedure). Each edge's "
        "score is a stratified mean over the sampled sources (each weighted by its band's share "
        "of the population) rather than count over every node, which estimates the same [0, 1] "
        "share and is exact, not approximate, wherever the sample covers every node."
    )


def _distance(u: Node, v: Node, data: dict[str, Any]) -> float:
    """``1/weight``: what one affinity edge costs, as the shortest-path measures also read it."""
    weight = float(data.get("weight", 1.0)) or 1.0
    return 1.0 / weight


# ------------------------------------------------------------------------- §27.4 convex


def _convex(graph: nx.Graph, edges: Sequence[Edge]) -> dict[Edge, dict[str, float]]:
    """Reduce the network to a tree of cliques: §27.4's convex skeleton.

    "A subgraph of a network G is convex if it contains all shortest paths existing in the main
    network G between its nodes... A network is convex if all its connected induced subgraphs
    are convex" (p. 388). The book gives the two elementary convex shapes -- a tree and a clique
    (p. 389) -- and then the construction: "You can build an arbitrary convex network by
    stitching together trees and cliques. In practice, it's just stitching together cliques,
    because the 'tree-like' parts are nothing more than 2-cliques", and a skeleton is had by
    "finding the minimal set of edges to remove to reduce the network into a tree of cliques"
    (p. 389). A network in which every biconnected component is a clique is exactly such a tree
    of cliques.

    **What the book under-specifies, and what was chosen here.** It names the target shape but
    no algorithm, and finding the *minimal* edge set that reaches it is NP-hard in general (it
    contains the block-graph edge deletion problem). This is the minimal faithful version:

    1. take the network's own maximal cliques, heaviest and largest first;
    2. accept a clique whole when its nodes lie in pairwise different components of what has
       been accepted so far -- which is precisely the condition under which the result is still
       a tree of cliques, since two nodes already connected would close a cycle across blocks
       and the merged block would not be complete;
    3. then join what is left with the heaviest edges that connect two different components,
       so no node is stranded and each original component stays in one piece.

    Cliques are taken whole rather than an edge at a time deliberately: a greedy that adds
    single edges to a spanning tree can only ever grow a triangle, so it would dissolve the
    dense corners the method exists to preserve. The cost is that the result depends on the
    order -- two cliques overlapping in two nodes cannot both survive, and the heavier one wins.

    Convexity is destroyed by a single edge either way ("adding a single edge to a tree or
    removing a single edge from a clique completely destroys convexity", p. 389), so this is a
    drastic filter: expect it to keep on the order of a tree's worth of edges plus the cliques.
    """
    kept: dict[Edge, dict[str, float]] = {}
    canonical = _canonical(graph)
    parent = {node: node for node in graph}
    cliques = [sorted(clique, key=_key) for clique in nx.find_cliques(graph) if len(clique) > 2]
    cliques.sort(key=lambda clique: (-len(clique), -_clique_weight(graph, clique), _key(clique)))
    for clique in cliques:
        roots = {_find(parent, node) for node in clique}
        if len(roots) != len(clique):
            continue  # two of its nodes are already joined: the merged block would not be a clique
        for position, u in enumerate(clique):
            for v in clique[position + 1 :]:
                kept[canonical[(u, v)]] = {"score": float(len(clique))}
                _union(parent, u, v)
    # What is left is a forest problem: the heaviest edge that joins two blocks, over and over.
    for u, v in sorted(edges, key=lambda edge: (-_weight(graph, *edge), _key(edge))):
        if _union(parent, u, v):
            kept.setdefault(canonical[(u, v)], {"score": 2.0})
    return kept


def _clique_weight(graph: nx.Graph, clique: Sequence[Node]) -> float:
    """The total weight inside one clique, which decides which of two overlapping ones wins."""
    return sum(
        _weight(graph, u, v) for position, u in enumerate(clique) for v in clique[position + 1 :]
    )


# ------------------------------------------------------------------- §27.5 disparity filter


def disparity_p(graph: nx.Graph, u: Node, v: Node) -> float:
    """The disparity-filter p-value of the edge ``u-v`` **as ``u`` sees it** (§27.5).

    The formula on p. 390, with ``N_u`` for ``u``'s neighbours::

        p((u, v), u) = (1 - w_uv / sum_{v' in N_u} w_uv') ** (|N_u| - 1)

    It is the probability that a node spreading its total strength uniformly at random over its
    neighbours would give one of them at least this large a share. The argument order matters:
    "the p-values for the same edge u, v will be different depending whether we focus on u or on
    v" whenever the two endpoints differ in degree or in strength (p. 390). That asymmetry is
    the method, not a defect of this implementation, and it is what
    :func:`noise_corrected_p` does not have.

    **On a directed network** the perspective is still ``u``'s, and the direction of the edge
    decides which of ``u``'s two distributions it is compared against, which is exactly the rule
    p. 391 gives: the edge "must be significant either when compared to the out-connections of
    the node sending the edge, or when compared to the in-connection weights of the node
    receiving it". So for the edge ``u -> v``, ``disparity_p(g, u, v)`` is the sender's view,
    over ``u``'s out-strength and out-degree, and ``disparity_p(g, v, u)`` is the receiver's,
    over ``v``'s in-strength and in-degree. When both ``u -> v`` and ``v -> u`` exist they are
    two different edges (§27.6, p. 392) and this function takes the one leaving ``u``.

    Undefined when ``u`` has exactly one neighbour on the side being read: the exponent is zero
    and the base is zero, and a node with a single edge has no distribution of weights for that
    edge to be unusual against. It is reported as 1.0 -- not significant from that side -- so
    such an edge survives only if the node at the other end finds it significant. Serrano et al.
    keep those edges by convention to preserve connectivity; that convention is not applied
    here, because an edge kept by a test that could not be run is an edge nobody can defend, and
    ``keep_isolates`` already decides what happens to the node.

    Raises ``KeyError`` when the two nodes are not joined in either direction.
    """
    if graph.is_directed():
        if graph.has_edge(u, v):  # u sends: its out-connections are the comparison
            return _disparity(_weight(graph, u, v), _out_strength(graph, u), graph.out_degree(u))
        if graph.has_edge(v, u):  # u receives: its in-connections are
            return _disparity(_weight(graph, v, u), _in_strength(graph, u), graph.in_degree(u))
        msg = f"{u!r} and {v!r} are not joined in either direction, so there is no edge to test"
        raise KeyError(msg)
    if not graph.has_edge(u, v):
        msg = f"{u!r} and {v!r} are not joined, so there is no edge to test"
        raise KeyError(msg)
    return _disparity(_weight(graph, u, v), _strength(graph, u), graph.degree(u))


def _disparity(weight: float, strength: float, degree: int) -> float:
    """p. 390's formula itself: ``(1 - w / s) ** (k - 1)``, on whichever k and s were passed.

    Undirected, that is the node's degree and strength; directed, the sender's out-degree and
    out-strength or the receiver's in-degree and in-strength. Keeping the arithmetic in one
    place is what makes the three readings obviously the same test.
    """
    if int(degree) <= 1:
        return 1.0
    share = weight / strength if strength else 0.0
    return float(max(0.0, 1.0 - share) ** (int(degree) - 1))


# -------------------------------------------------------------------- §27.6 noise corrected


def noise_corrected_p(graph: nx.Graph, u: Node, v: Node) -> float:
    """The noise-corrected p-value of the edge ``u-v`` (§27.6): one number, both endpoints' view.

    The null is the urn of §3.2, stated on pp. 391-392. The number of trials is the network's
    total edge weight; the number of successes is the weight of this edge; and the chance that
    any one unit of weight lands on this edge is::

        p_uv = (sum_{v' in N_u} w_uv') * (sum_{u' in N_v} w_u'v) / (sum_{u',v'} w_u'v') ** 2

    The p-value is then ``P(X >= w_uv)`` for ``X ~ Binomial(trials, p_uv)``. The sums run over
    *ordered* pairs, as the book writes them, so ``trials`` is twice the sum of the undirected
    edge weights; that is what makes the expected weight of an edge equal ``s_u * s_v / 2W``,
    the configuration model's expectation, rather than twice it.

    **On a directed network** the ordered-pair sum is already the sum over the edges, so
    ``trials`` is the total weight undoubled, and the two factors part company: the first is the
    weight leaving ``u`` (its out-strength) and the second the weight arriving at ``v`` (its
    in-strength), giving the directed configuration model's expectation
    ``out(u) * in(v) / W``. The measure is then **not** symmetric, and it should not be: "if
    your network is directed, w_uv != w_vu and you'll get a different null expectation for
    either direction of the edge, because the u, v edge is different from the v, u edge"
    (p. 392). ``noise_corrected_p(g, u, v)`` is about ``u -> v`` and raises ``KeyError`` if that
    edge does not exist, even when ``v -> u`` does.

    "All the elements here are the same in the perspective of u and v, thus this measure is u, v
    specific, differently from the disparity filter" (p. 392). That is the whole point: NC
    requires both nodes to agree, so a hub's weakest link does not survive on the strength of
    the small node's enthusiasm, and the peripheral edges that carry the community structure do
    (p. 393).

    **It works only on counts.** "Given that we use a binomial as a null model, you can see that
    NC works only for discrete counts as edge weights, because the binomial is a discrete
    distribution" (p. 392). Every weight this package produces is a count of shared documents or
    shared passages, so this holds; a graph carrying a fractional weight raises ``ValueError``
    rather than being rounded into one the book would not recognise.

    Raises ``KeyError`` when the two nodes are not joined, and ``ValueError`` on a network with
    no weight at all, where the urn has nothing to draw.
    """
    if not graph.has_edge(u, v):
        joined = "joined" if not graph.is_directed() else f"joined {u!r} -> {v!r}"
        msg = f"{u!r} and {v!r} are not {joined}, so there is no edge to test"
        raise KeyError(msg)
    trials = _trials(graph)
    if graph.is_directed():
        weights = _out_strength(graph, u) * _in_strength(graph, v)
    else:
        weights = _strength(graph, u) * _strength(graph, v)
    prior = weights / float(trials) ** 2
    return binomial_p(round(_weight(graph, u, v)), trials, min(1.0, prior), tail="right")


def _trials(graph: nx.Graph) -> int:
    """The binomial's number of draws: the total edge weight, summed over ordered pairs.

    Undirected, every edge is two ordered pairs, so the sum of the weights is doubled. Directed,
    every edge is one, so it is not.
    """
    fractional = [
        (u, v, w)
        for u, v, w in graph.edges(data="weight", default=1.0)
        if not float(w).is_integer()
    ]
    if fractional:
        u, v, w = fractional[0]
        msg = (
            "the noise-corrected backbone (§27.6) is a binomial test, which is discrete, so it "
            f"needs count weights; edge {u!r}-{v!r} has weight {w}. Use --backbone disparity or "
            "a structural method on a network whose weights are not counts."
        )
        raise ValueError(msg)
    # Ordered pairs both times: on a digraph the edge list *is* the ordered-pair list, so the
    # undirected sum is the one that has to be doubled to match it (§27.6, p. 392).
    scale = 1 if graph.is_directed() else 2
    total = round(scale * sum(float(w) for _, _, w in graph.edges(data="weight", default=1.0)))
    if total <= 0:
        msg = "the noise-corrected backbone needs a network with some edge weight in it"
        raise ValueError(msg)
    return total


# --------------------------------------------------------------------- §13.4 trees and PMFG


def _mst(graph: nx.Graph, edges: Sequence[Edge]) -> dict[Edge, dict[str, float]]:
    """The maximum spanning tree (§13.4), or a spanning forest when the network is not connected.

    "Is an edge with a high weight expressing the cost of going from u to v, or is it saying how
    much u and v interact?" (p. 202). Here it is always the second: weights are proximities, so
    the tree to want is the maximum one. It has ``n - 1`` edges per component and is the sparsest
    thing that still touches every node, which makes it the floor of this module rather than a
    backbone one would read communities off.

    Rarely unique: "as soon as you have two edges with the same weight, you open the door to the
    possibility of having more than one minimum spanning tree" (p. 203) -- a possibility, not a
    certainty, and the rule the book gives for telling them apart is that distinct weights
    guarantee a unique tree. A co-occurrence network is mostly ties, so expect the door to be
    open: which of the equally good trees comes back is ``networkx``'s Kruskal ordering, and a
    caller who needs one answer should say which tree they got rather than assume there was only
    one.
    """
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    canonical = _canonical(graph)
    candidates = set(edges)
    return {
        canonical[(u, v)]: {"score": _weight(graph, u, v)}
        for u, v in tree.edges()
        if canonical[(u, v)] in candidates
    }


def _pmfg(graph: nx.Graph, edges: Sequence[Edge]) -> dict[Edge, dict[str, float]]:
    """The planar maximally filtered graph (§13.4): the heaviest planar subgraph, greedily.

    Edges are taken heaviest first and kept while the graph can still be drawn without
    crossings, so the result holds at most ``3(n - 2)`` edges -- "a planar maximally filtered
    graph must have 3(|V|-2) or fewer edges" (p. 203) -- against a spanning tree's ``n - 1``.
    It is a superset of the maximum spanning tree by construction and keeps triangles, which a
    tree cannot, so it is the least destructive of the structural methods here.

    Greedy, therefore not optimal: the true maximum-weight planar subgraph is NP-hard, and this
    is the standard construction from Tumminello et al. Planarity is decided by
    ``nx.check_planarity``. A motif that cannot be drawn flat -- a 5-clique, a 3,3-bipartite
    core (p. 203) -- cannot survive, so a network whose dense corner is a large clique loses
    most of it whatever the weights say.
    """
    limit = 3 * (graph.number_of_nodes() - 2) if graph.number_of_nodes() > 2 else len(edges)
    built = nx.Graph()
    built.add_nodes_from(graph)
    kept: dict[Edge, dict[str, float]] = {}
    for u, v in sorted(edges, key=lambda edge: (-_weight(graph, *edge), _key(edge))):
        built.add_edge(u, v)
        if not nx.check_planarity(built, counterexample=False)[0]:
            built.remove_edge(u, v)
            continue
        kept[(u, v)] = {"score": _weight(graph, u, v)}
        if built.number_of_edges() >= limit:
            break
    return kept


# ------------------------------------------------------------------------- the one interface


def backbone(
    graph: nx.Graph,
    method: str = "noise-corrected",
    *,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
    keep_isolates: bool = False,
    seed: int | None = None,
) -> nx.Graph:
    """Return a copy of ``graph`` with the edges chapter 27's ``method`` would remove taken out.

    ``alpha`` is the significance level of the two statistical methods (``disparity``,
    ``noise-corrected``): an edge survives when its p-value is at or below it. ``correction``
    is one of :data:`graphrag.sna.stats.CORRECTIONS` (or ``"none"``) and adjusts the family of
    edge p-values before that comparison, because a backbone tests every edge at once and "100
    tests at the p < 0.01 standard" leave a 63% chance of a false positive (§3.3). Neither is
    read by a structural method, which has no null and therefore no p-value; those take
    ``threshold`` instead, in the units of their own score, and each has a documented default:

    ``naive``
        the weight threshold ``--min-weight`` has always applied; 1 by default, which keeps
        everything.
    ``naive-top``
        how many of its strongest edges each node keeps, :data:`NAIVE_TOP_DEFAULT` by default.
        It is a count, so it is rounded down to a whole number and never below 1.
    ``doubly-stochastic``
        the highest threshold leaving the network in as few components as it started with
        (§27.2's own guideline).
    ``high-salience``
        :data:`SALIENCE_THRESHOLD`, the middle of the empty valley in §27.3's "horns".
    ``convex``, ``mst``, ``pmfg``
        no threshold: the edge set falls out of the construction.

    The two are not interchangeable: ``threshold`` is **ignored by** ``disparity`` and
    ``noise-corrected``, which cut on ``alpha``, and ``alpha`` and ``correction`` are ignored by
    every structural method. ``graph.graph`` records whichever one governed the run -- the level
    in the sentence names ``alpha`` for a statistical method and ``threshold`` for a structural
    one -- so a report never implies a level that did no work.

    Every surviving edge carries the number it survived on -- ``p_value`` for a statistical
    method, ``score`` for a structural one -- so a reader can see *why* each edge is there and
    re-cut at a different level without recomputing. A statistical method also leaves
    ``p_adjusted`` (equal to ``p_value`` when no correction was asked for), and the disparity
    filter leaves ``p_other``: the p-value of the endpoint that voted to delete the edge, which is
    what makes its bias visible. The name is ``p_value`` and not ``p`` because ``p`` is taken, by
    the one thing it could be confused with: :mod:`graphrag.sna.uncertain` writes each edge's
    *probability of existing* there (Atlas §28.2), and a p-value is very nearly its opposite --
    small means the edge is real, where a small ``p`` means it probably is not.

    ``keep_isolates`` decides the fate of a node the filter stranded. False, the default, drops
    it; True keeps it, which is what p. 382 asks for -- backboning "wants to keep as many [nodes]
    as possible" -- and what one wants before comparing two backbones over the same node set. A
    node that was already isolated before the filter ran is always kept, whichever way this is
    set: the backbone did not remove its edges, and dropping it would be a filter nobody asked
    for.

    ``seed`` is accepted so every entry point in this package takes one, and recorded in the
    provenance. No method here is randomised by the book's own construction; two runs over the
    same graph give the same backbone, with the one caveat that ties are broken by node order
    (``mst`` most of all, §13.4) -- and, since ATL-F2, ``high-salience``'s own draw of sources on
    a network above :data:`SALIENCE_SAMPLE_ABOVE_NODES` nodes, which is this package's addition
    and not the book's, and is the one place ``seed`` changes the answer rather than only the
    tie-break.

    The provenance of the run is in ``graph.graph``: ``backbone`` is the sentence the report
    prints, and ``backbone_method``, ``backbone_section``, ``backbone_alpha``,
    ``backbone_correction``, ``backbone_threshold``, ``backbone_null``,
    ``backbone_edges_before``/``_after`` and ``backbone_nodes_before``/``_after`` are the pieces
    of it. All of them are scalars, so the graph still writes to GraphML.

    A directed network is accepted by the methods the book defines a directed form for --
    ``disparity`` (§27.5), ``noise-corrected`` (§27.6) -- and by the two naive thresholds, which
    read no neighbourhood. The rest refuse it and quote their ``directed_note``, which says
    whether the book defines a directed form at all.

    Raises ``ValueError`` on an unknown method, an ``alpha`` outside (0, 1], an unknown
    correction, a directed graph given to an undirected-only method, or -- for
    ``noise-corrected`` -- a network whose weights are not counts.
    """
    if method not in _BY_NAME:
        msg = f"backbone method must be one of {', '.join(BACKBONES)}, got {method!r}"
        raise ValueError(msg)
    if not 0.0 < alpha <= 1.0:
        msg = f"alpha must be a significance level in (0, 1], got {alpha}"
        raise ValueError(msg)
    if correction != "none" and correction not in CORRECTIONS:
        msg = f"correction must be 'none' or one of {', '.join(CORRECTIONS)}, got {correction!r}"
        raise ValueError(msg)
    spec = _BY_NAME[method]
    if graph.is_directed() and not spec.directed:
        msg = (
            f"the {spec.name} backbone ({spec.section}) is undirected: {spec.directed_note}. "
            "Use --backbone disparity or noise-corrected, whose directed forms the book does "
            "define, or undirect the network first and say that you did."
        )
        raise ValueError(msg)
    # A self-loop is not an edge between two nodes, so no method in the chapter scores one.
    # They are left exactly as they are rather than silently removed. Note the asymmetry that
    # leaves in §27.5 and §27.6 should one ever arrive: ``_strength`` reads
    # ``graph.degree(weight=)``, which counts a self-loop twice, while ``_trials`` sums
    # ``graph.edges``, which counts it once. No builder in this package emits a self-loop --
    # every projection is over ``combinations`` of distinct nodes -- so this is documented
    # rather than handled.
    edges: list[Edge] = [(u, v) for u, v in graph.edges() if u != v]
    note = ""
    used = 0.0
    if not edges:
        kept: dict[Edge, dict[str, float]] = {}
    elif method == "naive":
        used = 1.0 if threshold is None else threshold
        kept = _naive(graph, edges, used)
    elif method == "naive-top":
        # n is a count of edges, so it is floored at 1: "keep each node's top 0 edges" is not a
        # backbone, and a recorded 0 would make the report's note say "0 strongest".
        used = float(max(1, int(NAIVE_TOP_DEFAULT if threshold is None else threshold)))
        kept = _naive_top(graph, edges, int(used))
    elif method == "doubly-stochastic":
        scores, converged = doubly_stochastic_scores(graph)
        used = _connectivity_threshold(scores, graph) if threshold is None else threshold
        kept = {edge: {"score": s} for edge, s in scores.items() if s >= used}
        note = "" if converged else _sinkhorn_note(graph)
    elif method == "high-salience":
        scores, note = high_salience(graph, seed=seed)
        used = SALIENCE_THRESHOLD if threshold is None else threshold
        kept = {edge: {"score": s} for edge, s in scores.items() if s >= used}
    elif method == "convex":
        kept = _convex(graph, edges)
    elif method == "mst":
        kept = _mst(graph, edges)
    elif method == "pmfg":
        kept = _pmfg(graph, edges)
    else:
        kept = _significant(graph, edges, method, alpha, correction)

    result = graph.copy()
    result.remove_edges_from([edge for edge in edges if edge not in kept])
    for (u, v), data in kept.items():
        result[u][v].update(data)
    if not keep_isolates:
        stranded = [node for node in result if result.degree(node) == 0 and graph.degree(node) > 0]
        result.remove_nodes_from(stranded)
    _record(
        result,
        graph,
        spec,
        alpha=alpha,
        correction=correction,
        threshold=used,
        note=note,
        seed=seed,
    )
    return result


def _significant(
    graph: nx.Graph, edges: Sequence[Edge], method: str, alpha: float, correction: str
) -> dict[Edge, dict[str, float]]:
    """§27.5 and §27.6: a p-value per edge, corrected as a family, then compared with alpha."""
    directed = bool(graph.is_directed())
    raw: list[float] = []
    other: list[float] = []
    for u, v in edges:
        if method == "disparity":
            if directed:
                # p. 391: "the edge must be significant either when compared to the
                # out-connections of the node sending the edge, or when compared to the
                # in-connection weights of the node receiving it". The pair is (u -> v), so the
                # two readings are u's out side and v's in side -- never the reverse edge, which
                # is a different edge with its own two readings.
                first = _disparity(
                    _weight(graph, u, v), _out_strength(graph, u), graph.out_degree(u)
                )
                second = _disparity(
                    _weight(graph, u, v), _in_strength(graph, v), graph.in_degree(v)
                )
            else:
                first, second = disparity_p(graph, u, v), disparity_p(graph, v, u)
            # p. 390: the edge is checked from both ends and one success is enough to keep it,
            # so the smaller p-value is the one that decides. The other is kept on the edge.
            raw.append(min(first, second))
            other.append(max(first, second))
        else:
            raw.append(noise_corrected_p(graph, u, v))
    adjusted = correct(raw, correction) if correction != "none" else list(raw)
    kept: dict[Edge, dict[str, float]] = {}
    for position, edge in enumerate(edges):
        if adjusted[position] > alpha:
            continue
        data = {"p_value": raw[position], "p_adjusted": adjusted[position]}
        if other:
            data["p_other"] = other[position]
        kept[edge] = data
    return kept


def _sinkhorn_note(graph: nx.Graph) -> str:
    """Why the normalisation did not settle: the shape of the network, or the sparsity of it.

    A two-mode network is the first case and is not bad luck: §27.2 proves a doubly stochastic
    matrix must be square and concludes "you cannot apply the doubly stochastic backboning to
    bipartite networks, unless |V1| = |V2|" (p. 386). Saying "sparse networks" there would
    invite the reader to try again with a bigger corpus, which cannot help.
    """
    # Imported here rather than at module scope: ``export`` imports this module for the naive
    # threshold, so a module-level import back into it would be a cycle. ``describe`` reads the
    # ``mode`` the builders put on each node, which is what makes a network two-mode (§6.4).
    from graphrag.sna.export import describe

    if describe(graph).bipartite:
        return (
            f"the Sinkhorn-Knopp normalisation did not converge in {SINKHORN_ITERATIONS} passes, "
            "and on this network it cannot: it is two-mode, and §27.2 shows a doubly stochastic "
            "matrix must be square, so 'you cannot apply the doubly stochastic backboning to "
            "bipartite networks, unless |V1| = |V2|' (p. 386). Project it onto one side first "
            "(--project), or read another method; the scores below are not doubly stochastic."
        )
    return (
        f"the Sinkhorn-Knopp normalisation did not converge in {SINKHORN_ITERATIONS} passes, "
        "which §27.2 warns happens on sparse networks ('this solution cannot be always applied', "
        "p. 385). The scores below are the iteration's last state and are not doubly stochastic; "
        "read another method."
    )


def _record(
    result: nx.Graph,
    before: nx.Graph,
    spec: BackboneMethod,
    *,
    alpha: float,
    correction: str,
    threshold: float,
    note: str,
    seed: int | None,
) -> None:
    """Write the run's provenance onto the graph, as scalars a GraphML file can hold."""
    edges_before, edges_after = before.number_of_edges(), result.number_of_edges()
    share = edges_after / edges_before if edges_before else 0.0
    if spec.statistical:
        adjusted = f", {correction}-corrected" if correction != "none" else ""
        level = f"alpha {alpha:g}{adjusted}"
    else:
        level = f"threshold {threshold:g}"
    sentence = (
        f"Backboned with the {spec.name} method ({spec.section}) at {level}: "
        f"{edges_after:,} of {edges_before:,} edges survive ({share:.1%}), "
        f"{result.number_of_nodes():,} of {before.number_of_nodes():,} nodes. "
        f"Null model: {spec.null}."
    )
    result.graph.update(
        backbone=sentence + (f" Note: {note}" if note else ""),
        backbone_method=spec.name,
        backbone_section=spec.section,
        backbone_alpha=alpha if spec.statistical else 0.0,
        backbone_correction=correction if spec.statistical else "",
        backbone_threshold=threshold,
        backbone_null=spec.null,
        backbone_note=note,
        backbone_seed="" if seed is None else str(seed),
        backbone_edges_before=edges_before,
        backbone_edges_after=edges_after,
        backbone_nodes_before=before.number_of_nodes(),
        backbone_nodes_after=result.number_of_nodes(),
    )


# ------------------------------------------------------------------------- §27's comparison


@dataclass(frozen=True)
class BackboneRow:
    """What one method kept, next to the null model that kept it."""

    method: str
    section: str
    null: str
    nodes: int
    edges: int
    edge_share: float
    """Surviving edges over the edges the method was given."""
    weight_share: float
    """Surviving edge weight over the total edge weight: how much of the corpus's co-occurrence
    the backbone still holds, which can be far higher than the edge share when the method keeps
    the heavy edges."""
    components: int
    weights: Summary | None
    """``stats.describe`` of the surviving weights, or ``None`` when nothing survived."""
    note: str = ""
    applicable: bool = True
    """False when the method refused this network -- §27.6 on non-count weights, say. The
    refusal is printed rather than the row being dropped."""


def compare_backbones(
    graph: nx.Graph,
    *,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
    methods: Sequence[str] | None = None,
    seed: int | None = None,
) -> list[BackboneRow]:
    """Run every method over one network at one alpha, and report what each kept (ch. 27).

    This is the chapter's own comparison: "these methods exist because they give very different
    results. It is up to you to decide which of their assumptions best fits the network you are
    analyzing and the type of things you want to say about the network" (p. 393). Nothing here
    picks a winner; the rows are the evidence for that decision.

    ``naive`` is run at **the smallest threshold that removes anything** -- the second-smallest
    distinct weight in the network -- because that is the row p. 383 is about: in the book's
    Twitter projection "82% of the edges have weight equal to one. The smallest possible hard
    threshold would remove 82% of the network, without allowing for any nuance". The note on
    that row says what the share is here.

    ``threshold``, when given, replaces every structural method's default -- including the
    ``naive`` row's -- in that method's own units. Left alone, each uses the default documented
    on :func:`backbone`, which is what makes the rows comparable at all.

    A method that refuses this network returns a row with ``applicable=False`` and the reason,
    so the refusal is part of the comparison rather than a gap in it. On a directed network that
    is most of them: the two naive thresholds and the two statistical methods run, and the five
    the book defines no directed form for say so by row rather than being left out.
    """
    weights = [float(w) for _, _, w in graph.edges(data="weight", default=1.0)]
    rows: list[BackboneRow] = []
    for name in methods or BACKBONES:
        spec = _BY_NAME[name]
        level = threshold
        derived = level is None and name == "naive"
        if derived:
            level = _smallest_hard_threshold(weights)
        try:
            filtered = backbone(
                graph,
                name,
                alpha=alpha,
                correction=correction,
                threshold=level,
                keep_isolates=True,
                seed=seed,
            )
        except ValueError as exc:
            rows.append(
                BackboneRow(
                    method=name,
                    section=spec.section,
                    null=spec.null,
                    nodes=0,
                    edges=0,
                    edge_share=0.0,
                    weight_share=0.0,
                    components=0,
                    weights=None,
                    note=str(exc),
                    applicable=False,
                )
            )
            continue
        kept = [float(w) for _, _, w in filtered.edges(data="weight", default=1.0)]
        note = str(filtered.graph.get("backbone_note", ""))
        if name == "naive":
            note = (note + " " + _naive_note(weights, level, derived=derived)).strip()
        if name == "naive-top":
            note = (note + " " + _naive_top_note(filtered)).strip()
        rows.append(
            BackboneRow(
                method=name,
                section=spec.section,
                null=spec.null,
                nodes=filtered.number_of_nodes() - _stranded(filtered),
                edges=filtered.number_of_edges(),
                edge_share=(len(kept) / len(weights)) if weights else 0.0,
                weight_share=(sum(kept) / sum(weights)) if sum(weights) else 0.0,
                components=_components(filtered),
                weights=describe(kept) if kept else None,
                note=note,
            )
        )
    return rows


def _stranded(graph: nx.Graph) -> int:
    """How many nodes the filter left with no edges (the comparison keeps them, and says so)."""
    return sum(1 for node in graph if graph.degree(node) == 0)


def _smallest_hard_threshold(weights: Sequence[float]) -> float:
    """The lowest threshold that removes any edge at all: §27.1's best case."""
    distinct = sorted(set(weights))
    if len(distinct) > 1:
        return distinct[1]
    return distinct[0] + 1.0 if distinct else 1.0


def _naive_note(weights: Sequence[float], threshold: float | None, *, derived: bool) -> str:
    """The p. 383 arithmetic for this network, in the units the book used.

    ``derived`` says whether ``threshold`` is the smallest hard threshold this function
    computed or a level the caller asked for. The book's sentence -- the smallest cut removes
    the whole bottom tier in one step -- is only true of the former; beside a level the caller
    chose, the note reports the share at the floor and what the chosen level did to it.
    """
    if not weights or threshold is None:
        return ""
    floor = min(weights)
    share = sum(1 for w in weights if w <= floor) / len(weights)
    if derived:
        return (
            f"{share:.0%} of the edges sit at the smallest weight ({floor:g}), so the smallest "
            f"possible hard threshold ({threshold:g}) removes that share of the network in one "
            "step, with no nuance available (p. 383). Whether any threshold can be motivated at "
            "all is settled by the weight distribution at the top of this report, not by this row."
        )
    fate = "keeps" if threshold <= floor else "removes"
    return (
        f"{share:.0%} of the edges sit at the smallest weight ({floor:g}); the threshold asked "
        f"for ({threshold:g}) {fate} that tier. The smallest possible hard threshold would remove "
        "it in one step, with no nuance available (p. 383). Whether any threshold can be "
        "motivated at all is settled by the weight distribution at the top of this report, not "
        "by this row."
    )


def _correction_text(correction: str, edges: int) -> str:
    """Whether the family of edge tests was corrected, for the report's own frame line."""
    if correction == "none":
        return ", uncorrected -- every edge is a test of its own (§3.3; see --correction)."
    return f", {correction}-corrected across all {edges:,} edge tests (§3.3)."


def _naive_top_note(filtered: nx.Graph) -> str:
    """p. 384's objection, measured on the result rather than only asserted."""
    degrees = [d for _, d in filtered.degree() if d]
    if not degrees:
        return ""
    return (
        f"Each node kept its {int(filtered.graph['backbone_threshold'])} strongest edge(s), so "
        f"the minimum degree of this backbone is {min(degrees)} by construction and its degree "
        'distribution has no left tail left to read -- "a heinous crime against the God of '
        'power law degree distributions" (p. 384). Read the degree distribution (§9.3) off the '
        "unfiltered network, never off this one."
    )


def render_backbones(
    graph: nx.Graph,
    rows: Sequence[BackboneRow],
    *,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
) -> str:
    """The comparison as markdown: what each method of chapter 27 keeps, and under which null."""
    edges = graph.number_of_edges()
    nodes = graph.number_of_nodes()
    weights = [float(w) for _, _, w in graph.edges(data="weight", default=1.0)]
    lines = [
        "# Backbone comparison",
        "",
        "**Implements:** *The Atlas for the Aspiring Network Scientist*, chapter 27 (network "
        "backboning), plus §13.4 for the spanning tree and the planar filtered graph.",
        "",
        f"**Sampling frame:** {graph.graph.get('frame', 'not recorded')}",
        "",
        f"**n before any backbone:** {nodes:,} nodes, {edges:,} edges, "
        f"{_components(graph):,} {'weakly ' if graph.is_directed() else ''}connected "
        "component(s). "
        f"**Significance level:** alpha {alpha:g}{_correction_text(correction, edges)} Nodes are "
        "counted before any isolate is dropped, so every method is compared over one node set.",
        "",
    ]
    if weights:
        summary = describe(weights)
        lines += [
            f"**The weights being filtered** (n={summary.count:,}): median {summary.median:g}, "
            f"quartiles {summary.q1:g}-{summary.q3:g}, max {summary.maximum:g}, "
            f"mean {summary.mean:.3g}.",
            "",
        ]
        if summary.caveat:
            lines += [f"> {summary.caveat}", ""]
    lines += [
        "| method | § | edges kept | share | weight share | nodes with edges | components |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row.applicable:
            lines.append(f"| `{row.method}` | {row.section} | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{row.method}` | {row.section} | {row.edges:,} | {row.edge_share:.1%} | "
            f"{row.weight_share:.1%} | {row.nodes:,} | {row.components:,} |"
        )
    lines += ["", "## What each null model assumed, and what survived it", ""]
    for row in rows:
        lines.append(f"**`{row.method}` ({row.section})** — null model: {row.null}.")
        lines.append("")
        lines.append(f"On every edge it keeps: {_BY_NAME[row.method].scored}.")
        lines.append("")
        if not row.applicable:
            lines += [f"Not run on this network: {row.note}", ""]
            continue
        if row.weights is None:
            lines.append("Nothing survived, so there is no weight distribution to describe.")
        else:
            # The full heavy-tail caveat is printed once, over the unfiltered weights at the top;
            # repeating it per method would bury the comparison it is meant to qualify.
            tail = (
                " Heavy-tailed (§3.1): quote the median, not the mean."
                if (row.weights.heavy_tailed)
                else ""
            )
            lines.append(
                f"Surviving weights (n={row.weights.count:,}): median {row.weights.median:g}, "
                f"quartiles {row.weights.q1:g}-{row.weights.q3:g}, max {row.weights.maximum:g}, "
                f"mean {row.weights.mean:.3g}.{tail}"
            )
        if row.note:
            lines.append("")
            lines.append(f"> {row.note}")
        lines.append("")
    lines += [
        "## Reading this",
        "",
        "A backbone is a new sampling frame, not a cleaner view of the old one: every number "
        "computed after one describes the edges that survived the stated test. Name the method "
        "and the level next to the number.",
        "",
        'Chapter 27 closes on the trade-off (p. 393): "A naive threshold fixes the same '
        "obstacle for all nodes no matter how strong, favoring the connections of the hub; HSS "
        "can include weaker links if they're the only path to a node; DF is similar to naive, "
        "but can recognize important weak edges; and NC overweights peripheries and communities: "
        'it is the most punishing method for the central hubs." Weights here are counts of '
        "shared documents or passages, which is the case §27.6 was written for, so "
        "`noise-corrected` is the one to reach for first -- and the one to prefer over "
        "`disparity` whenever the next question is about communities.",
    ]
    return "\n".join(lines)


def backbone_payload(
    rows: Sequence[BackboneRow], *, alpha: float = DEFAULT_ALPHA, correction: str = "none"
) -> dict[str, Any]:
    """``rows`` (ch. 27's comparison, from :func:`compare_backbones`) as the JSON `sna backbone`
    writes with ``--json`` and the ``sna_backbone`` MCP tool returns as ``payload`` -- the same
    fields :func:`render_backbones` reads, one dict per :class:`BackboneRow` (ATL-F1: one
    function both surfaces call, instead of each building this dict inline)."""
    from dataclasses import asdict

    return {"alpha": alpha, "correction": correction, "rows": [asdict(r) for r in rows]}


# ----------------------------------------------------------------------------- small helpers


def _weight(graph: nx.Graph, u: Node, v: Node) -> float:
    """The edge's weight, with an unweighted edge counting 1."""
    return float(graph[u][v].get("weight", 1.0))


def _strength(graph: nx.Graph, node: Node) -> float:
    """A node's strength: the total weight of its edges (§27.5's and §27.6's denominator)."""
    return float(graph.degree(node, weight="weight"))


def _out_strength(graph: nx.Graph, node: Node) -> float:
    """The weight leaving a node: the sending half of §27.5's and §27.6's directed forms."""
    return float(graph.out_degree(node, weight="weight"))


def _in_strength(graph: nx.Graph, node: Node) -> float:
    """The weight arriving at a node: the receiving half."""
    return float(graph.in_degree(node, weight="weight"))


def _canonical(graph: nx.Graph) -> dict[Edge, Edge]:
    """Both orientations of every edge, mapped to the one ``graph.edges()`` yields."""
    index: dict[Edge, Edge] = {}
    for u, v in graph.edges():
        if u == v:
            continue
        index[(u, v)] = (u, v)
        index[(v, u)] = (u, v)
    return index


def _key(value: Any) -> str:
    """A deterministic sort key for node ids of mixed type, so ties break the same way twice."""
    return str(value)


def _find(parent: dict[Node, Node], node: Node) -> Node:
    """Union-find with path compression, over the components accepted so far."""
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def _union(parent: dict[Node, Node], u: Node, v: Node) -> bool:
    """Join two components; False when they were already one, which is the cycle test."""
    a, b = _find(parent, u), _find(parent, v)
    if a == b:
        return False
    parent[b] = a
    return True
