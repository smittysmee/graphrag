"""Centrality, brokerage and whole-network summaries. Returns data; prints nothing.

One subtlety runs through this module. Our edge weights are *affinities*: a weight of 6 means
"shared six documents", so a heavier edge means the two nodes are closer. ``networkx`` shortest
paths read ``weight`` as a *distance*, where heavier means further apart. Betweenness and
closeness therefore run on a derived ``distance`` attribute of ``1 / weight``, while degree,
eigenvector and PageRank use the affinity directly. Getting this backwards silently inverts the
result, which is why it is done in one place here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from graphrag.sna.export import describe as describe_network
from graphrag.sna.matrices import adjacency, eigenpairs
from graphrag.sna.stats import Summary, describe

CENTRALITIES: tuple[str, ...] = (
    "degree",
    "weighted_degree",
    "betweenness",
    "closeness",
    "harmonic",
    "eigenvector",
    "pagerank",
    "reach",
    "coreness",
)

#: The same battery on a directed network. Degree splits in two, because on a directed graph
#: "how many others" is two different questions (§6.2): how many point at this node, and how
#: many it points at. The rest keep their names and read direction where the algorithm does.
#: HITS is here and nowhere else: §14.5 defines it on a directed network and it degenerates on
#: an undirected one, so :func:`hits` refuses that case rather than printing two equal columns.
DIRECTED_CENTRALITIES: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "betweenness",
    "closeness",
    "harmonic",
    "eigenvector",
    "pagerank",
    "reach",
    "hits_hub",
    "hits_authority",
    "coreness",
)

#: Centralities whose per-node cost does not need a shortest-path search: degree in any of its
#: four forms is one pass over the edges, coreness is one O(n + m) peel (§14.7), and eigenvector,
#: PageRank and HITS are each a fixed number of power-iteration sweeps over the adjacency. Chapter
#: 28's Monte Carlo (§28.2) calls the whole battery again on every sampled possible world, so
#: these are the ones ``analyze --uncertain`` samples by default (ATL-F2); see
#: :data:`EXPENSIVE_CENTRALITIES` for the rest and ``--uncertain-expensive`` for opting them in.
#: The two tuples partition :data:`CENTRALITIES` and :data:`DIRECTED_CENTRALITIES` exactly, which
#: a test holds them to.
CHEAP_CENTRALITIES: tuple[str, ...] = (
    "degree",
    "weighted_degree",
    "in_degree",
    "out_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "eigenvector",
    "pagerank",
    "hits_hub",
    "hits_authority",
    "coreness",
)

#: Centralities that search every node's shortest paths (betweenness, closeness, harmonic) or a
#: bounded BFS from it (reach) -- O(n(m + n log n)) or worse, the same cost shape §27.3 complains
#: about for high-salience. Re-running the whole battery once per sampled possible world (§28.2)
#: multiplies that cost by ``uncertain_samples``, which is what made ``analyze --uncertain`` slow
#: on a large network; these are skipped by default and opted in with ``--uncertain-expensive``.
EXPENSIVE_CENTRALITIES: tuple[str, ...] = (
    "betweenness",
    "closeness",
    "harmonic",
    "reach",
)

#: How many hops :func:`reach` counts by default. §14.3 defines reach centrality with no bound
#: -- everything a BFS from the node can touch -- and says in the same breath that it "doesn't
#: make much sense for undirected networks", because one connected component makes every node's
#: reach 1. Two hops is the bounded variant: the share of the network a node's neighbours'
#: neighbours cover, which distinguishes nodes on a connected undirected network where the
#: unbounded one cannot. ``reach(graph, hops=None)`` is the chapter's own measure.
REACH_HOPS = 2

CENTRALITY_MEANING: dict[str, str] = {
    "degree": "local prominence: how many distinct others a node sits with.",
    "weighted_degree": "volume: prominence with repeated co-appearances counted.",
    "in_degree": "being named: how many distinct others point at this node.",
    "out_degree": "naming: how many distinct others this node points at.",
    "weighted_in_degree": "being named, with every passage that says so counted.",
    "weighted_out_degree": "naming, with every passage that says so counted.",
    "betweenness": "brokerage: how often a node lies on the shortest path between two others.",
    "closeness": "reach: how near a node is to everyone else.",
    "harmonic": (
        "closeness that survives disconnection: the sum of inverse distances, which counts an "
        "unreachable node as 0 rather than as undefined (§14.6)."
    ),
    "eigenvector": "influence among the influential: being connected to well-connected nodes.",
    "pagerank": "influence with a damping factor, so a single huge hub cannot dominate.",
    "reach": (
        f"command: the share of the network a node touches within {REACH_HOPS} hops (§14.3)."
    ),
    "hits_hub": "hubbiness: pointing at the nodes everyone else points at (§14.5).",
    "hits_authority": "authority: being pointed at by the good hubs (§14.5).",
    "coreness": (
        "depth: the largest k whose k-core still holds this node, after everyone with fewer "
        "than k connections has been peeled away (§14.7). Many nodes share a value."
    ),
}

#: What the same measure means once the edges have a direction. A caption that does not change
#: with the network is a caption that is wrong on one of them, so these three are overridden and
#: :func:`centrality_meaning` is what the reports print.
DIRECTED_CENTRALITY_MEANING: dict[str, str] = {
    "degree": (
        "local prominence, both directions at once: on a directed network this is the *total* "
        "degree (in + out). Read in_degree and out_degree for the two halves."
    ),
    "weighted_degree": (
        "volume, both directions at once: the passages behind a node's ties whichever way they "
        "run (in + out). Read weighted_in_degree and weighted_out_degree for the two halves."
    ),
    "closeness": (
        "being reachable: how near everyone else is to this node. On a directed network this is "
        "*incoming* closeness, so it ranks the entities the rest of the network can get to, not "
        "the ones that can get everywhere."
    ),
    "harmonic": (
        "being reachable from everywhere, counting the unreachable as 0 (§14.6). Like closeness "
        "this is the *incoming* form on a directed network: the sum of 1/distance over the paths "
        "that arrive. It is the one shortest-path ranking here that stays defined when the "
        "network is in pieces, which a directed one usually is."
    ),
    "reach": (
        "command: the share of the network this node can get to in at most "
        f"{REACH_HOPS} hops, following the edges the way they point (§14.3). This is the one "
        "ranking here that reads *out* of a node while closeness and harmonic read into it, so "
        "a high reach beside a low closeness is an entity that names everything and is named "
        "by nothing."
    ),
    "coreness": (
        "depth, direction ignored: the k-core decomposition of §14.7 is defined on an "
        "undirected network, so this is computed on the flattened view. The directed analogue "
        "the chapter names -- D-cores, which split the peel into in- and out-degree -- is not "
        "computed here."
    ),
}

#: What a measure that cannot read direction had to do to this network, in the report's words.
#: Handed out by :func:`undirected_view` so every caller says the same thing.
FLATTENED = (
    "on the flattened undirected view of this directed network, where two entities joined in "
    "either direction become one undirected edge carrying the total of the directed weights, so "
    "direction, and with it reciprocity, is invisible to it (§6.2)"
)

#: The conventions ``networkx`` applies to two whole-network numbers once a graph is directed.
#: Both stay in the summary table under their undirected names, so the report has to say which
#: coefficient it printed rather than leave a reader to assume the undirected one.
DIRECTED_SUMMARY = (
    "On a directed network two numbers in the summary change definition without changing name: "
    "`degree_assortativity` is the out-in coefficient (the out-degree of the source correlated "
    "with the in-degree of the target, networkx's default for a digraph), and "
    "`average_clustering` is the directed (Fagiolo) coefficient, which counts all eight "
    "triangle orientations rather than the one an undirected triangle has. `transitivity` is "
    "the third case: §12.2's global coefficient is defined on undirected graphs and networkx "
    "returns 0.0 for a directed triangle rather than refusing, so it is computed on the "
    "flattened view and is not on the same footing as the `average_clustering` beside it."
)


def centralities_for(graph: nx.Graph) -> tuple[str, ...]:
    """The battery to run on this graph: the directed one when it has directions to read."""
    return DIRECTED_CENTRALITIES if graph.is_directed() else CENTRALITIES


def centrality_meaning(kind: str, graph: nx.Graph) -> str:
    """The caption a report prints under this ranking, for this network.

    Three measures mean something different once the edges point: total degree stops being "how
    many others", and closeness becomes a statement about being reached rather than reaching
    (§6.2). Everything else keeps its line.
    """
    if graph.is_directed() and kind in DIRECTED_CENTRALITY_MEANING:
        return DIRECTED_CENTRALITY_MEANING[kind]
    return CENTRALITY_MEANING[kind]


def undirected_view(graph: nx.Graph) -> tuple[nx.Graph, str]:
    """This graph as an undirected one, and a note saying what that cost -- or itself, and "".

    Several methods in this package are defined only for undirected graphs: Louvain and its
    degree-preserving null, the eigenvector solved with ``eigh`` (which needs a symmetric
    matrix), the participation coefficient, the spectral embedding. The Atlas is clear that a
    directed edge carries information an undirected one does not (§6.2), so handing them a
    directed network has to be a stated step rather than a silent one: this returns the note
    they print, and refusing outright would leave a directed network with no communities at all.

    Reciprocated pairs are summed, not averaged: ``A -> B`` weighted 2 beside ``B -> A``
    weighted 1 becomes one edge of weight 3, because the weights count passages and three
    passages spoke about that pair.
    """
    if not graph.is_directed():
        return graph, ""
    flat = nx.Graph()
    flat.graph.update(graph.graph)
    flat.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if flat.has_edge(u, v):
            flat[u][v]["weight"] += weight
        else:
            flat.add_edge(u, v, weight=weight)
    return flat, FLATTENED


def directed_notes(graph: nx.Graph) -> list[str]:
    """What a reader of a directed network's report has to be told, and nowhere else is.

    Two things: which measures could not read the direction at all, and which two summary
    numbers quietly changed definition when they met a digraph. Louvain says the first for
    itself, in :attr:`graphrag.sna.cluster.LouvainResult.note`, because it can name its own null
    model; this covers the measures with nowhere of their own to say it. Empty for an undirected
    network, where none of it arises.
    """
    if not graph.is_directed():
        return []
    return [
        "Every measure in this report that is defined only for undirected graphs -- the "
        "eigenvector centrality, the broker scores, any spectral features and the --by section "
        "-- is computed " + FLATTENED + ". The in- and out- rankings, the directed betweenness "
        "and closeness, and the reciprocity are what read the direction.",
        DIRECTED_SUMMARY,
    ]


def reciprocity(graph: nx.Graph) -> float:
    """The share of connected pairs whose connection runs both ways (§10.3).

    The book counts pairs, not edges: "we count the number of connected pairs of the network:
    pairs of nodes with at least one edge between them [...] Then we count the number of
    connected pairs that have both possible edges between them [...] Reciprocity is simply the
    second count over the first one." Its worked example has five connected pairs, two of them
    reciprocated, and reads 2/5.

    That is deliberately not ``networkx.overall_reciprocity``, which divides reciprocated edges
    by all edges and would call the same figure 4/7. Both are defensible; this one is the
    book's, and it is the one a report that cites §10.3 has to print.

    Undefined for an undirected graph, where every edge is reciprocal by construction, and for
    a directed graph with no connected pairs at all. Both return 0.0, because a table of
    measures has no room for a NaN -- the same convention degree assortativity uses here.
    """
    if not graph.is_directed() or graph.number_of_edges() == 0:
        return 0.0
    # One entry per connected pair, keyed on the unordered pair and holding one of the two
    # directions; a self loop is not a pair of nodes and is left out of both counts.
    pairs: dict[frozenset[Any], tuple[Any, Any]] = {}
    for u, v in graph.edges():
        if u != v:
            pairs.setdefault(frozenset((u, v)), (u, v))
    if not pairs:
        return 0.0
    return sum(1 for u, v in pairs.values() if graph.has_edge(v, u)) / len(pairs)


def _with_distance(graph: nx.Graph) -> nx.Graph:
    """A copy carrying ``distance = 1 / weight`` for the shortest-path measures."""
    copy = graph.copy()
    for u, v, data in copy.edges(data=True):
        weight = float(data.get("weight", 1.0)) or 1.0
        copy[u][v]["distance"] = 1.0 / weight
    return copy


def centrality(graph: nx.Graph, kind: str) -> dict[str, float]:
    """One centrality by name. ``centralities_for(graph)`` lists the ones this graph supports
    and :func:`centrality_meaning` says what each answers *on this graph*.

    ``degree`` and ``weighted_degree`` are defined on a directed network too, and there they are
    the **total** degree: in plus out, the two questions of §6.2 added together. They are not
    refused, because every ordering in the reports -- which members a community lists first,
    which nodes a comparison ranks -- is a total-degree ordering and would otherwise have to
    pick a side. Ask for ``in_degree`` / ``out_degree`` (or their weighted forms) when the
    direction is the question; the caption the report prints says which of the two it is.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if kind == "degree":
        return {n: float(d) for n, d in nx.degree_centrality(graph).items()}
    if kind == "weighted_degree":
        return {n: float(d) for n, d in graph.degree(weight="weight")}
    if kind in {"in_degree", "out_degree", "weighted_in_degree", "weighted_out_degree"}:
        return _directed_degree(graph, kind)
    if kind == "betweenness":
        return {
            n: float(v)
            for n, v in nx.betweenness_centrality(_with_distance(graph), weight="distance").items()
        }
    if kind == "closeness":
        return _closeness(graph)
    if kind == "harmonic":
        return harmonic(graph)
    if kind == "eigenvector":
        return _eigenvector(graph)
    if kind == "pagerank":
        return {n: float(v) for n, v in nx.pagerank(graph, weight="weight").items()}
    if kind == "reach":
        return reach(graph)
    if kind in {"hits_hub", "hits_authority"}:
        return _hits_role(graph, "hub" if kind == "hits_hub" else "authority")
    if kind == "coreness":
        return coreness(graph)
    msg = f"centrality must be one of {', '.join(centralities_for(graph))}, got {kind!r}"
    raise ValueError(msg)


def _closeness(graph: nx.Graph) -> dict[str, float]:
    """Closeness over the affinity-derived distances, and on a digraph the *incoming* one.

    ``networkx`` measures a node's closeness along the paths that arrive at it, so on a directed
    network this answers "how near is everyone else to this node" rather than "how near is this
    node to everyone else" (§6.2 -- with a direction the two stop being the same question).
    That convention is kept rather than reversed, because it is the one every reader of the
    networkx docs expects; for the other direction, pass ``graph.reverse()``, whose incoming
    closeness is this graph's outgoing closeness. Undirected, there is only one answer and this
    is it.
    """
    return {
        n: float(v)
        for n, v in nx.closeness_centrality(_with_distance(graph), distance="distance").items()
    }


def _directed_degree(graph: nx.Graph, kind: str) -> dict[str, float]:
    """In- or out-degree, raw or weighted, on a directed network (§6.2).

    Raw degrees are normalised by ``n - 1`` exactly as ``nx.degree_centrality`` normalises the
    undirected one, so the two are on the same scale and a report can put them in one table.
    The weighted variants are left as totals, because that is what ``weighted_degree`` is: a
    count of the passages behind a node's ties, and dividing it by anything hides the count.
    """
    if not graph.is_directed():
        msg = (
            f"{kind} is defined only on a directed network; this one is undirected, where "
            "in and out are the same thing. Use degree or weighted_degree."
        )
        raise ValueError(msg)
    weighted = kind.startswith("weighted_")
    incoming = "in_degree" in kind
    view = graph.in_degree if incoming else graph.out_degree
    if weighted:
        return {n: float(d) for n, d in view(weight="weight")}
    scale = max(graph.number_of_nodes() - 1, 1)
    return {n: float(d) / scale for n, d in view()}


def _eigenvector(graph: nx.Graph) -> dict[str, float]:
    """The principal eigenvector of the weighted adjacency, solved densely.

    ``networkx`` offers two eigenvector solvers and neither is usable in a report: the power
    iteration fails to converge on a graph with several components, which real corpora produce
    constantly, and the sparse solver starts from a random vector, so the same graph and the
    same seed give different numbers on two runs and it raises outright on graphs with only a
    few nodes. ``numpy.linalg.eigh`` is exact, reproducible and defined for every symmetric
    matrix, which is what a network's adjacency always is here -- once a directed network has
    been flattened, which this does and :func:`directed_notes` reports (§6.2).
    ``matrices.eigenpairs`` is that same solver, kept there for the same reason and reused
    rather than repeated.

    The matrix is laid out in ``list(graph.nodes)`` order rather than the sorted order
    ``matrices`` defaults to, because the callers of this function have always read the graph in
    that order. Node order changes which row is which, not what a row is worth (§8.1), and
    flattening preserves it, so an undirected graph comes back through here bit for bit.
    """
    if graph.number_of_edges() == 0:
        return dict.fromkeys(graph.nodes, 0.0)
    graph, _ = undirected_view(graph)  # eigh needs a symmetric matrix; the note is the caller's
    nodes = list(graph.nodes)
    matrix, order = adjacency(graph, nodes=nodes, weight="weight")
    _, vectors = eigenpairs(matrix, k=1, largest=True)
    principal = np.abs(vectors[:, 0])
    norm = float(np.linalg.norm(principal)) or 1.0
    return {node: float(value) for node, value in zip(order, principal / norm, strict=True)}


# ------------------------------------------------------- reach, harmonic, HITS, cores (ch. 14)


def reach(graph: nx.Graph, hops: int | None = REACH_HOPS) -> dict[str, float]:
    """The share of the network a node can get to, within ``hops`` steps (§14.3).

    "The local reach centrality of a node v is the fraction of nodes in a network that you can
    reach starting from v", computed the way the chapter computes it: "you start from node v and
    you explore the graph with a BFS strategy. Once you cannot explore any more, you stop. The
    number of nodes you touched divided by the number of nodes in the network is your reach
    centrality." On a directed network the BFS follows the edges the way they point, so this is
    the one ranking in the battery that reads *out* of a node -- "how much you can command in
    your network if you're node v" -- while closeness and harmonic read into it.

    **The origin is not one of the nodes it reached**, here or in the book: §14.3's wheel graph
    has a central node of reach 1 and a sink of reach 0, and both hold only if v is left out of
    the count on the top and the bottom. So this is reachable others over ``|V| - 1``, which is
    also ``networkx.local_reaching_centrality``'s convention for the unbounded case.

    **What the book defines and what this adds.** The chapter's measure is unbounded --
    ``hops=None`` -- and it warns in the same paragraph that it "doesn't make much sense for
    undirected networks": one connected component gives every node a reach of exactly 1, and so
    does one strongly connected component on a directed network. ``hops`` is the standard
    bounded variant (an *m-reach*): the share of the network within m steps, which still
    separates nodes on a connected undirected network. :data:`REACH_HOPS` is the default and the
    reports print it in the caption, because "reach" without its radius is two different numbers.

    Hops are counted as **edges, not weights**: the chapter's BFS has no notion of a heavy edge,
    and a co-occurrence weight of 6 does not make two entities six steps apart. Undefined for a
    network with fewer than two nodes, which has nothing to reach; that returns 0.0 per node.
    """
    if hops is not None and hops < 1:
        msg = (
            f"reach() needs at least one hop, or hops=None for §14.3's unbounded reach, got {hops}"
        )
        raise ValueError(msg)
    others = graph.number_of_nodes() - 1
    if others < 1:
        return dict.fromkeys(graph.nodes, 0.0)
    return {
        node: (len(nx.single_source_shortest_path_length(graph, node, cutoff=hops)) - 1) / others
        for node in graph.nodes
    }


def harmonic(graph: nx.Graph) -> dict[str, float]:
    """The sum of inverse distances: closeness that survives disconnection (§14.6).

    ``HC_v = sum_u 1 / |P_vu|``, and the chapter's whole point is the convention that makes it
    total: "the harmonic centrality handles unreachable nodes properly, based on the assumption
    that 1/inf = 0". Closeness, betweenness, reach and the random-walk measures are all "ill
    defined when your network has pairs of unreachable nodes"; this one is defined on every
    network, which is why it is in the battery beside closeness rather than instead of it.

    It is **not on closeness's scale**. This is the chapter's raw sum, not an average, so it runs
    from 0 to ``|V| - 1`` on an unweighted network instead of 0 to 1, and two networks of
    different sizes cannot be compared by it. Distances are the affinity-derived ``1 / weight``
    this module uses everywhere (see the module docstring), so on a network whose weights exceed
    1 a distance can be below 1 and the sum can exceed ``|V| - 1``; the ordering is what a report
    reads off it.

    On a directed network this is ``networkx``'s *incoming* harmonic centrality, the same
    direction :func:`_closeness` takes, for the same reason: it ranks the nodes the rest of the
    network can get to. §14.6 does not choose between the two, and the other one is
    ``harmonic(graph.reverse())``.

    Lin's centrality, the other repair §14.6 offers -- closeness multiplied by the size of the
    node's component, ``(|V_v| - 1)^2 / sum_u |P_vu|`` -- is not computed here. It needs a
    component to belong to, so it says nothing about the isolates a filtered corpus network is
    full of, while the harmonic sum scores them 0 without special-casing them.
    """
    return {
        n: float(v)
        for n, v in nx.harmonic_centrality(_with_distance(graph), distance="distance").items()
    }


def hits(graph: nx.Graph) -> tuple[dict[str, float], dict[str, float]]:
    """Hub and authority scores, as the two eigenvectors §14.5 defines them by.

    "A good hub is a hub that points to good authorities. A good authority is an authority which
    is pointed by good hubs." Solved as linear algebra rather than by iterating that definition:
    the authority vector is the principal eigenvector of ``A^T A`` and the hub vector the
    principal eigenvector of ``A A^T``, which is the "more efficiently, with clever linear
    algebra" the chapter points at. Both matrices are symmetric, so
    :func:`graphrag.sna.matrices.eigenpairs` solves them exactly with ``eigh`` -- the same choice
    :func:`_eigenvector` makes and for the same reasons: ``networkx.hits`` is a power iteration
    that raises ``PowerIterationFailedConvergence`` on graphs a corpus produces.

    Returns ``(hubs, authorities)``, each **normalised so the maximum score is 1**, which is the
    chapter's normalisation ("we then normalize so that the maximum hub and authority score is
    one", Figure 14.11). ``networkx`` normalises the sum to 1 instead; the ranking is the same
    either way, the numbers are not, and the ones printed here are the book's.

    Two departures from the text, both stated rather than hidden. The chapter's ``A`` is binary
    and this one carries the edge weights, because every other ranking in this module reads the
    affinity weights and a relation stated in nine passages is not the same edge as one stated
    once -- pass a graph whose weights are all 1 for the chapter's own matrix. And where the
    leading eigenvalue is repeated, which happens as soon as the network has two components that
    each contain a hub, the leading eigenvector is not unique: ``eigh`` returns a deterministic
    one, so two runs agree with each other, but the split between the tied components is an
    artefact of the solver rather than a statement about the network.

    **Refused on an undirected network**, which is the degeneracy §14.5 rests on: with a
    symmetric ``A``, ``A^T A`` and ``A A^T`` are the same matrix, so hubs and authorities are one
    vector printed twice, and the chapter's two roles -- "a person who knows the people who know
    them" against "she has mastered her own topic" -- cannot be told apart. Use ``eigenvector``
    there, which is what that single vector is.
    """
    return _hits_role(graph, "hub"), _hits_role(graph, "authority")


def _hits_role(graph: nx.Graph, role: str) -> dict[str, float]:
    """One of the two vectors of :func:`hits`, so the battery pays for one ``eigh`` and not two.

    ``centrality(graph, "hits_hub")`` asks for the hub scores and nothing else, and the two
    eigendecompositions are the whole cost of the measure. The refusal and the two empty cases
    live here so both doors give the same answer.
    """
    if not graph.is_directed():
        msg = (
            "hits() is defined only on a directed network (§14.5): with an undirected, symmetric "
            "adjacency matrix A^T A and A A^T are the same matrix, so the hub and the authority "
            "score are one vector printed twice. Use the eigenvector centrality, which is that "
            "vector."
        )
        raise ValueError(msg)
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    if graph.number_of_edges() == 0:
        return dict.fromkeys(nodes, 0.0)
    matrix, order = adjacency(graph, nodes=nodes, weight="weight")
    square = matrix @ matrix.T if role == "hub" else matrix.T @ matrix
    return _leading_unit(square, order)


def _leading_unit(square: np.ndarray, order: Sequence[Any]) -> dict[str, float]:
    """The principal eigenvector of a symmetric matrix, scaled so its largest entry is 1."""
    _, vectors = eigenpairs(square, k=1, largest=True)
    principal = np.abs(vectors[:, 0])
    peak = float(principal.max()) or 1.0
    return {node: float(value) for node, value in zip(order, principal / peak, strict=True)}


def coreness(graph: nx.Graph) -> dict[str, float]:
    """Each node's core number: the deepest k-core it survives into (§14.7).

    "A k-core in a network is a subset of its nodes in which all nodes have at least k
    connections to each other", and the decomposition of Figure 14.12 is the peeling that finds
    them: label and remove the nodes of degree 1, recursively, because removing a neighbour can
    drop another node to degree 1 as well; then do the same for 2, 3, and up "until there are no
    remaining nodes in the network". A node's coreness is the step at which it was peeled, which
    is ``networkx.core_number`` (Batagelj and Zaversnik's O(m) algorithm, the chapter's own
    footnote 33).

    Returned as floats because the battery is a ``dict[str, float]``; every value is a whole
    number, and a node of coreness 4 is not "twice as central" as one of coreness 2.

    **A coreness is a floor, not a rank**, which is the caveat that makes it different from
    everything else in this module: it is a property of a *set*, so a large network typically
    has a handful of distinct values and dozens of nodes sharing each of them. The karate club
    has core numbers 1 to 4 over 34 nodes, ten of them in the 4-core. Use it to say "this node
    is in the dense middle" and never "this node is the 7th most central".

    Weights are ignored, because the chapter counts connections rather than their strength, and
    a directed network is flattened first (§14.7 is stated for undirected networks; the chapter
    names D-cores, k-shells and k-coronas as the variants that go further, and none of them is
    computed here). Self loops are dropped: ``networkx`` refuses them outright, and a node is not
    one of its own k connections. :func:`core_shells` is the same decomposition seen as sizes.
    """
    flat, _ = _undirected_simple(graph)
    return {node: float(value) for node, value in nx.core_number(flat).items()}


def core_shells(graph: nx.Graph) -> dict[int, int]:
    """How many nodes sit in each shell: core number -> node count, ascending (§14.7).

    The shell decomposition the report prints under the coreness ranking, and the reason that
    ranking needs one: the k-shell is the set of nodes whose core number is *exactly* k, so this
    says how coarse the ranking is before a reader takes a position in it seriously. A network
    whose 34 nodes fall into four shells has four distinct scores, not 34.

    Empty for a graph with no nodes.
    """
    counts: dict[int, int] = {}
    for value in coreness(graph).values():
        counts[int(value)] = counts.get(int(value), 0) + 1
    return dict(sorted(counts.items()))


def k_truss(graph: nx.Graph, k: int) -> nx.Graph:
    """The k-truss: every edge that sits in at least ``k - 2`` triangles, peeled recursively.

    **What the book defines and what this adds.** §14.7 defines the k-core and then says the
    decomposition "is only the most famous among many similar which define a structure of
    interest and use it to define a centrality measure", naming D-cores, k-shells and k-coronas.
    The truss is the edge-side member of that family (Jonathan Cohen. Trusses: cohesive subgraphs
    for social network analysis. National Security Agency technical report, 2008) and the book
    does not name it: where the k-core asks every node for k neighbours, the k-truss asks every
    *edge* for ``k - 2`` triangles, so it is the cohesion the clustering coefficient of §12.2
    counts rather than the cohesion the degree counts. A k-truss is always inside the
    ``(k-1)``-core, never the other way round.

    It is the stricter of the two on a co-occurrence network, and usefully so: a document naming
    m entities makes an m-clique, so a single long passage inflates everybody's core number,
    while the truss keeps only the pairs whose triangles recur.

    Returns the induced subgraph ``networkx.k_truss`` returns, on the undirected, self-loop-free
    view (the definition counts triangles, which §12.1 defines on an undirected graph, and
    ``networkx`` refuses a digraph or a self loop). Empty for a ``k`` no edge reaches, which is
    an answer and not a failure. ``k`` below 2 is refused: a triangle count of ``k - 2`` is not
    defined below zero.
    """
    if k < 2:
        msg = f"k_truss() needs k of at least 2 (an edge in k-2 triangles), got {k}"
        raise ValueError(msg)
    flat, _ = _undirected_simple(graph)
    result: nx.Graph = nx.k_truss(flat, k)
    return result


def truss_number(graph: nx.Graph) -> dict[tuple[Any, Any], int]:
    """Per edge, the largest k whose k-truss still holds it -- the truss analogue of coreness.

    Keyed by the endpoint pair in sorted order, so ``(u, v)`` and ``(v, u)`` are one key however
    the edge was added, and a directed network's two arcs between the same pair are one entry
    (the truss is defined on the flattened, self-loop-free view -- see :func:`k_truss`).

    Every edge of a graph with at least one edge has a truss number of at least 2, because the
    2-truss asks for no triangles at all. Computed by raising k until the truss is empty, which
    is at most as many passes as the largest clique has nodes; the chapter defines no cost for
    this measure, since it does not define the measure.
    """
    flat, _ = _undirected_simple(graph)
    numbers: dict[tuple[Any, Any], int] = {}
    k = 2
    while True:
        truss = nx.k_truss(flat, k)
        if truss.number_of_edges() == 0:
            return numbers
        for u, v in truss.edges():
            first, second = _sorted_nodes((u, v))
            numbers[first, second] = k
        k += 1


def top_n(scores: dict[str, float], n: int = 20) -> list[tuple[str, float]]:
    """The ``n`` highest scores, ties broken by node id so the output is reproducible."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]


def brokers(
    graph: nx.Graph, communities: Sequence[Iterable[str]], n: int = 20
) -> list[tuple[str, float]]:
    """Nodes whose neighbours span the most communities, by participation coefficient.

    The coefficient is ``1 - sum_c (k_ic / k_i)^2``: zero when every neighbour sits in one
    community, approaching one when a node's ties are spread evenly across many. It answers a
    different question from betweenness, which counts shortest paths without knowing where the
    groups are; a node can broker between two groups without being on many shortest paths.

    A neighbour is a neighbour in either direction, so a directed network is flattened first
    (§6.2, reported by :func:`directed_notes`): otherwise the score would count only the ties a
    node draws itself, and call an entity that nothing but relations point at a non-broker.
    """
    graph, _ = undirected_view(graph)
    membership: dict[str, int] = {}
    for index, community in enumerate(communities):
        for node in community:
            membership[node] = index
    scores: dict[str, float] = {}
    for node in graph.nodes:
        per_community: dict[int, float] = {}
        total = 0.0
        for neighbour in graph.neighbors(node):
            weight = float(graph[node][neighbour].get("weight", 1.0))
            group = membership.get(neighbour, -1)
            per_community[group] = per_community.get(group, 0.0) + weight
            total += weight
        if total <= 0:
            scores[node] = 0.0
            continue
        scores[node] = 1.0 - sum((share / total) ** 2 for share in per_community.values())
    return top_n(scores, n)


def summary(graph: nx.Graph) -> dict[str, float]:
    """Whole-network shape: size, density, fragmentation, clustering, assortativity.

    Two of those are chapter 12's and are not interchangeable (§12.2). ``average_clustering`` is
    the mean of the per-node coefficients, **weighted** by Onnela's definition, so a triangle of
    weak ties counts for less than a triangle of strong ones; ``transitivity`` is the *global*
    coefficient, 3 x triangles / triads over the whole network and binary, so a hub with many
    unacquainted neighbours pulls it down while leaving the average alone. The book's own
    example has a global 0.5 beside an average 0.648 on one graph. The report's ``## Density``
    section prints both with the gap explained, and :func:`density_report` adds the *unweighted*
    average, which is the one directly comparable with the transitivity.

    ``density`` is likewise unreadable without ``nodes`` beside it (§12.1); :func:`density_note`
    is the sentence that says so and the report prints it.

    On a directed network the table gains ``reciprocity`` (§10.3) and its ``components`` are the
    *weakly* connected ones: the pieces a reader of an undirected network expects, direction
    ignored. Which nodes can actually reach which -- the strongly connected components of §10.4
    -- is a different question, answered by :func:`graphrag.sna.paths.components` and printed in
    every report's "Paths and components" section beside the weak ones, together with the
    condensation DAG and §10.3's dyad census behind this table's ``reciprocity``.

    Two more keep their names and change their definition there, which is worse than changing
    both, so :data:`DIRECTED_SUMMARY` states it and every report prints that line:
    ``degree_assortativity`` is networkx's out-in coefficient for a digraph (the source's
    out-degree against the target's in-degree), and ``average_clustering`` is the directed
    (Fagiolo) coefficient, which counts all eight orientations of a triangle rather than the
    single one an undirected triangle has.
    """
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    if nodes == 0:
        empty: dict[str, float] = {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "components": 0,
            "largest_component": 0,
            "largest_component_share": 0.0,
            "average_clustering": 0.0,
            "transitivity": 0.0,
            "degree_assortativity": 0.0,
            "average_weighted_degree": 0.0,
        }
        if graph.is_directed():
            empty["reciprocity"] = 0.0
        return empty
    components = list(
        nx.weakly_connected_components(graph)
        if graph.is_directed()
        else nx.connected_components(graph)
    )
    largest = max((len(c) for c in components), default=0)
    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        assortativity = float("nan")
    if math.isnan(assortativity):
        # Undefined when every node has the same degree; report 0.0 rather than a NaN in a table.
        assortativity = 0.0
    weighted = [float(d) for _, d in graph.degree(weight="weight")]
    stats: dict[str, float] = {
        "nodes": nodes,
        "edges": edges,
        "density": float(nx.density(graph)),
        "components": len(components),
        "largest_component": largest,
        "largest_component_share": largest / nodes,
        "average_clustering": float(nx.average_clustering(graph, weight="weight")),
        "transitivity": _network_clustering(graph, "global"),
        "degree_assortativity": assortativity,
        "average_weighted_degree": sum(weighted) / nodes,
    }
    if graph.is_directed():
        stats["reciprocity"] = reciprocity(graph)
    return stats


# ------------------------------------------------- density, clustering, cliques (ch. 12)

#: The four coefficients of §12.2, which are four different questions about the same graph.
#: ``local`` and ``weighted`` return one number per node, ``average`` and ``global`` one number
#: for the network.
CLUSTERING_KINDS: tuple[str, ...] = ("local", "average", "global", "weighted")

#: How many cliques :func:`cliques` enumerates before it stops and says it stopped. §12.3 says
#: only that a k-clique contains a clique of every smaller size; the cost that follows is not
#: in the chapter. A graph of n nodes can hold up to 3^(n/3) maximal cliques (J. W. Moon and
#: L. Moser. On cliques in graphs. *Israel Journal of Mathematics*, 3:23-28, 1965), so listing
#: them is exponential in the worst case and a report needs a budget rather than a promise.
CLIQUE_LIMIT = 20_000

#: How many cliques of the largest size, and how many members of the independent set, a report
#: lists by name before it gives up and prints the count.
MAX_LISTED = 8


def _undirected_simple(graph: nx.Graph) -> tuple[nx.Graph, str]:
    """The undirected, self-loop-free view chapter 12 is defined on, and what that cost.

    The chapter states its assumption once and keeps it: "from now on, I'll just assume the
    network is undirected, unipartite and monolayer, for simplicity" (§12.1). Triangles,
    cliques and independent sets are all defined that way, and ``networkx`` refuses
    ``find_cliques`` on a digraph outright -- ``transitivity`` does something worse, returning
    0.0 for a directed triangle rather than raising. So everything in this section runs on the
    flattened view and the report prints :data:`FLATTENED` beside the numbers. Self loops are
    dropped because none of the three objects has a place for one: a node is not its own
    neighbour, its own clique-mate or its own independent-set conflict.
    """
    flat, note = undirected_view(graph)
    loops = list(nx.selfloop_edges(flat))
    if loops:
        flat = flat.copy()
        flat.remove_edges_from(loops)
    return flat, note


def _sorted_nodes(nodes: Iterable[Any]) -> list[Any]:
    """Nodes in their own order where they have one, by their text where they do not.

    The legendary graphs are keyed by integers and a corpus network by strings, and a member
    list sorted by ``str`` puts node 13 between 1 and 2. Mixed-type ids -- which no builder
    here produces -- fall back to the text.
    """
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=str)


def _label(graph: nx.Graph, node: Any) -> str:
    """A node's display name, by the same rule the report uses: ``label``, ``name``, or its id."""
    data = graph.nodes.get(node, {})
    return str(data.get("label") or data.get("name") or node)


def density_note(graph: nx.Graph) -> str:
    """The sentence §12.1 says has to sit next to a density, with this network's numbers in it.

    The chapter's whole argument is that a density means nothing on its own. The denominator,
    ``|V|(|V|-1)/2`` undirected and ``|V|(|V|-1)`` directed, grows quadratically with the nodes;
    the numerator grows about linearly, because real networks keep adding a few edges per node
    and no more. So the density of a real network falls as it grows -- the book's own examples
    run from 0.05% for the power grid through 0.004% for arXiv citations to 0.003% for the
    Internet backbone -- and "0.004 is low" is not a statement about a network until n is beside
    it. Two networks of different sizes cannot be compared by their densities at all.

    Self loops are not counted in the numerator, because the chapter bans them from the
    denominator and says so twice ("we're banning self loops", "seriously, no self loops!"):
    the possible edges are the pairs of *distinct* nodes, and an edge from a node to itself is
    not one of them. It would otherwise be possible to print "8 of the 12 edges 4 nodes could
    carry" with one of the eight not among the twelve. The count is said out loud when any were
    dropped. Note that :func:`summary` reports ``networkx``'s density, which does count them;
    on every network this package builds the two are the same number, because no builder here
    relates a node to itself.

    Undefined for a network with fewer than two nodes, where there is no pair to connect; the
    note says so rather than dividing by zero.
    """
    nodes, edges, possible, loops = _density_counts(graph)
    if possible == 0:
        return (
            f"Density is undefined for {nodes} node(s): there is no pair of distinct nodes to "
            "connect, so the |V|(|V|-1) of §12.1 is zero."
        )
    formula = "|V|(|V|-1)" if graph.is_directed() else "|V|(|V|-1)/2"
    dropped = (
        f" {loops:,} self loop(s) are left out of the count, because §12.1 counts the pairs of "
        "distinct nodes and a loop is not one of them."
        if loops
        else ""
    )
    return (
        f"{edges:,} of the {possible:,} edges {nodes:,} nodes could carry ({formula}), a "
        f"density of {edges / possible:.4f}.{dropped} This number is only readable against n "
        "(§12.1): the possible edges grow quadratically with the nodes while a real network "
        "adds only a few edges per node, so density falls as a network grows. It is not on its "
        "own evidence that this corpus is sparsely connected, and it cannot be compared with "
        "the density of a network of a different size."
    )


def _density_counts(graph: nx.Graph) -> tuple[int, int, int, int]:
    """Nodes, edges between distinct nodes, edges those nodes could carry, and self loops.

    One place decides what the numerator and the denominator of a density are, so
    :func:`density_note` and :func:`density_report` cannot print a fraction and a percentage
    that disagree. The denominator is §12.1's: ``|V|(|V|-1)/2`` undirected, ``|V|(|V|-1)``
    directed, and on a directed network that is the count this section keeps -- density is the
    one number here that is *not* taken on the flattened view, because flattening a digraph
    would halve the denominator and change what was measured.
    """
    nodes = graph.number_of_nodes()
    loops = int(nx.number_of_selfloops(graph))
    edges = graph.number_of_edges() - loops
    possible = nodes * (nodes - 1) if graph.is_directed() else nodes * (nodes - 1) // 2
    return nodes, edges, possible, loops


def _local_clustering(graph: nx.Graph, *, weight: str | None = None) -> dict[str, float]:
    """Per-node local clustering on the flattened, self-loop-free view (§12.2)."""
    flat, _ = _undirected_simple(graph)
    return {node: float(value) for node, value in nx.clustering(flat, weight=weight).items()}


def clustering(graph: nx.Graph, kind: str = "local") -> dict[str, float] | float:
    """One of the four clustering coefficients of §12.2, by name.

    The chapter's point is that these are different questions, and reporting one as another is
    the mistake it names outright: "you have to remember that the global and the average
    clustering coefficient are two different things, and not to report one as the other".

    ``local`` (one number per node)
        The triangles the node belongs to over the triads centred on it, i.e. over the
        ``k(k-1)/2`` pairs of its neighbours. It is the share of a node's neighbours who know
        each other, so 1 means no structural hole around it and a low value means the node is
        what information has to pass through (§12.2). **Undefined for a node of degree 0 or 1**,
        which has no pair of neighbours to close: ``networkx`` and this function report 0.0
        there, which is not the same as "its neighbours are strangers".
    ``average`` (one number)
        The mean of the local coefficients, ``CC_avg = (1/|V|) sum_v CC_v``. Every node counts
        once however large its neighbourhood, so a mass of small tight neighbourhoods lifts it.
    ``global`` (one number)
        Transitivity: ``3 * triangles / triads`` over the whole network. The three is because
        one triangle closes three triads, one seen from each of its corners. Triads are counted
        as ``sum_v k_v(k_v-1)/2``, so a hub contributes quadratically and dominates the number.
        **Undefined for a graph with no triads at all**, reported as 0.0. The book's worked
        example: eight triangles and 48 triads give 3*8/48 = 0.5 on a graph whose *average*
        coefficient is 0.648.
    ``weighted`` (one number per node)
        Onnela's coefficient (Onnela, Saramäki, Kertész and Kaski, *Phys. Rev. E* 71:065103,
        2005; the variants are compared by Saramäki, Kivelä, Onnela, Kaski and Kertész, *Phys.
        Rev. E* 75:027105, 2007), which §12.2 points at for the weighted case. A triangle counts
        not as 1 but as the geometric mean of its three edge weights, normalised by the heaviest
        weight in the network, so a triangle of weak ties is worth less than a triangle of
        strong ones. This is what ``networkx`` implements for ``weight=``. Our weights count
        shared documents, so it answers "do the co-occurrences that close triangles happen
        often, or once each".

    On a **two-mode** network every one of them is 0 by construction: a triangle needs three
    mutually adjacent nodes and every edge there joins the two modes, so there are no triangles
    to count and these coefficients are not defined on it at all (§6.4). Project onto one mode
    before asking; :func:`density_report` says so rather than printing the zero as a finding.

    A directed graph is flattened first and every kind here is the undirected coefficient,
    which is §12.1's stated assumption. That is deliberately *not* what :func:`summary` does
    with ``average_clustering``, which stays on Fagiolo's directed coefficient (ATL-06), so the
    two can differ on a digraph and every report says which it printed.
    """
    if kind in {"local", "weighted"}:
        return _local_clustering(graph, weight="weight" if kind == "weighted" else None)
    if kind in {"average", "global"}:
        return _network_clustering(graph, kind)
    msg = f"clustering kind must be one of {', '.join(CLUSTERING_KINDS)}, got {kind!r}"
    raise ValueError(msg)


def _network_clustering(graph: nx.Graph, kind: str) -> float:
    """The two whole-network coefficients, as a ``float`` rather than a union (§12.2).

    :func:`clustering` is one door with four answers, two of them per-node, so its return type
    is a union and every caller inside this module would have to narrow it. This is the branch
    that always returns one number.
    """
    flat, _ = _undirected_simple(graph)
    if kind == "average":
        return float(nx.average_clustering(flat)) if flat.number_of_nodes() else 0.0
    return float(nx.transitivity(flat))


def local_clustering_summary(graph: nx.Graph) -> Summary:
    """The distribution behind the average coefficient, described by §3.1's rules (§12.2).

    The average clustering coefficient is a mean, and §3.1's warning applies to it like any
    other: a network of a few dense triangles hanging off a mass of degree-1 nodes has a mean
    that nobody's neighbourhood looks like. This hands back the median, the quartiles and the
    heavy-tail caveat to print next to it.

    The sample is every node, including the degree-0 and degree-1 ones whose coefficient is
    undefined and counted as 0.0 -- which is why the median can be 0.0 on a network that is
    visibly clustered. Undefined for a graph with no nodes, which raises ``ValueError``.
    """
    values = _local_clustering(graph)
    if not values:
        msg = "local_clustering_summary() needs at least one node; an empty graph has none"
        raise ValueError(msg)
    return describe(list(values.values()))


@dataclass(frozen=True)
class CliqueReport:
    """The complete subgraphs of one network (§12.3), and what listing them cost."""

    #: How many maximal cliques were found: cliques no further node can be added to.
    total: int
    #: Maximal clique size -> how many of that size. A 2 is an edge whose endpoints share no
    #: neighbour; the right tail of this distribution is the tightly knit groups.
    sizes: dict[int, int]
    largest_size: int
    #: Up to :data:`MAX_LISTED` of the maximal cliques of ``largest_size``, members sorted.
    largest: list[list[str]]
    #: How many maximal cliques have ``largest_size``, which can be more than were listed.
    largest_count: int
    #: The ``k`` that was asked for, or ``None`` when only the maximal cliques were wanted.
    k: int | None
    #: How many complete subgraphs of exactly ``k`` nodes exist, maximal or not.
    k_clique_count: int
    #: Up to :data:`MAX_LISTED` of them.
    k_cliques: list[list[str]]
    #: True when the enumeration hit its budget, which makes every count above a lower bound.
    truncated: bool
    note: str


def cliques(graph: nx.Graph, k: int | None = None, limit: int = CLIQUE_LIMIT) -> CliqueReport:
    """The maximal cliques of a network, their sizes, and the largest of them (§12.3).

    A clique is a set of k nodes carrying all ``k(k-1)/2`` possible edges: "when it comes to
    clustering and density, you cannot do any better than having all possible edges among the
    nodes". A *maximal* clique is one no further node can be added to, which is what is
    enumerated here (``networkx.find_cliques``, Bron-Kerbosch); the book is explicit that every
    k-clique contains cliques of every smaller size, so listing the non-maximal ones is mostly
    listing subsets.

    ``k`` asks the other question the chapter defines: how many complete subgraphs of exactly
    ``k`` nodes there are, maximal or not. That is the expensive one -- one maximal clique of
    size m holds ``C(m, k)`` of them, and ``networkx.enumerate_all_cliques`` reaches size k by
    enumerating every clique below it first -- so it is opt-in and budgeted.

    **The cost, which the chapter does not state.** §12.3 defines the objects and leaves it
    there. The bound is that a graph of n nodes can hold up to 3^(n/3) maximal cliques (Moon
    and Moser, 1965), so enumerating them is exponential in the worst case. ``limit`` caps how
    many cliques *either* walk visits; past it the walk stops, ``truncated`` is set, and every
    count becomes a lower bound. The budget covers the whole ``k`` walk and not only its
    size-``k`` finds, which matters because the walk reaches size k last: a budget spent on the
    smaller cliques leaves ``k_clique_count`` a lower bound that can be 0 on a graph that has
    such cliques. The note says so whenever the budget bit.

    On a co-occurrence network the cliques are also uninteresting before they are expensive --
    a passage naming m entities *is* an m-clique on its own -- so the largest clique of an
    entity network is usually one document. The report says so, and so does the reading rule.

    On a **two-mode** network every maximal clique is a single edge, because no three nodes are
    mutually adjacent there; the object §12.3 defines for that case is the biclique, which is
    not computed. :func:`render_density` says so rather than printing "largest clique: 2" as a
    finding.

    An empty graph has no cliques and reports zero of them; nothing here is undefined.
    """
    if k is not None and k < 1:
        msg = f"k must be at least 1 (a 1-clique is a node, a 2-clique an edge), got {k}"
        raise ValueError(msg)
    flat, _ = _undirected_simple(graph)
    sizes: dict[int, int] = {}
    largest: list[list[str]] = []
    largest_size = 0
    total = 0
    truncated = False
    for found in nx.find_cliques(flat) if flat.number_of_nodes() else ():
        total += 1
        size = len(found)
        sizes[size] = sizes.get(size, 0) + 1
        if size > largest_size:
            largest_size, largest = size, []
        if size == largest_size and len(largest) < MAX_LISTED:
            largest.append(_sorted_nodes(found))
        if total >= limit:
            truncated = True
            break
    k_cliques: list[list[str]] = []
    k_count = 0
    k_truncated = False
    if k is not None:
        # ``enumerate_all_cliques`` yields in increasing size, so the walk can stop at k + 1 --
        # but it has to visit every smaller clique to get there, and those are what the budget
        # is usually spent on. Counting every visit, not only the size-k ones, is what keeps a
        # k of 5 on a dense graph from running unbounded.
        visited = 0
        for found in nx.enumerate_all_cliques(flat):
            visited += 1
            if visited > limit:
                truncated = k_truncated = True
                break
            if len(found) > k:
                break
            if len(found) < k:
                continue
            k_count += 1
            if len(k_cliques) < MAX_LISTED:
                k_cliques.append(_sorted_nodes(found))
    return CliqueReport(
        total=total,
        sizes=dict(sorted(sizes.items())),
        largest_size=largest_size,
        largest=largest,
        largest_count=sizes.get(largest_size, 0),
        k=k,
        k_clique_count=k_count,
        k_cliques=k_cliques,
        truncated=truncated,
        note=_clique_note(truncated=truncated, k_truncated=k_truncated, limit=limit),
    )


def _clique_note(*, truncated: bool, k_truncated: bool, limit: int) -> str:
    """What a reader has to know about these counts before using them.

    §12.3 states neither the cost nor the bound; the 3^(n/3) is Moon and Moser's, and it is
    quoted here rather than left to a docstring because a truncated count printed in a report
    without its reason is a wrong number.
    """
    cost = (
        "A graph of n nodes can hold up to 3^(n/3) maximal cliques (Moon and Moser, 1965; the "
        f"chapter defines the objects and not the cost), so this walk is budgeted at {limit:,} "
        "cliques."
    )
    if not truncated:
        return f"{cost} It finished inside the budget, so these counts are complete."
    hit = (
        f"{cost} It hit that budget, so every count here is a lower bound and the largest "
        "clique may be larger than the one reported."
    )
    if not k_truncated:
        return hit
    return (
        f"{hit} The budget ran out inside the walk for cliques of exactly k nodes, which "
        "reaches that size last, so the count of those is a lower bound and can be 0 on a "
        "graph that has them."
    )


def independent_set(graph: nx.Graph) -> list[str]:
    """A large set of nodes no two of which are connected: an anti-clique (§12.4).

    "Just like matter has anti-matter, also cliques have anti-cliques": an independent set is a
    set of nodes none of which are connected to each other, which is a clique in the graph's
    complement. The set returned is **maximal** -- no node of the graph can be added to it
    without breaking independence.

    It is *not* the **maximum** one, and that difference is the caution of §12.4: "you should
    not confuse the maximal independent set with the maximum independent set". The maximum one
    is the largest such set in the network, and the chapter's exercise on it accepts an
    approximate answer. That finding it is NP-hard is not the chapter's statement but the
    standard result (R. M. Karp. Reducibility among combinatorial problems. In *Complexity of
    Computer Computations*, pages 85-103, 1972, where it is the clique problem on the
    complement). This is the classic greedy approximation: repeatedly take the node with the
    fewest surviving neighbours (ties broken by node id, so the answer is the same on every
    run) and delete it and its neighbours. The size is therefore a *lower* bound on the
    maximum, and a report must not read it as "the largest group in this corpus that never
    co-occurs".

    ``networkx.maximal_independent_set`` is the obvious alternative and is not used, for one
    reason: it picks each node with ``random.choice`` over a Python ``set``, so with string node
    ids the same graph gives a different set on two runs even at a fixed seed, and a report
    cannot print a number that moves. ``networkx.algorithms.approximation.maximum_independent_set``
    has the better O(n/log^2 n) guarantee but recurses once per node, which a corpus-sized
    network overflows, and it does not promise a maximal set.

    §12.4 also names the problem this generalises: covering every node with independent sets is
    graph colouring, so how many such sets it takes to exhaust a network is its chromatic
    number. That is not computed here. A directed network is flattened first, and a self loop is
    ignored -- it would otherwise make a node its own conflict.
    """
    flat, _ = _undirected_simple(graph)
    residual: dict[Any, set[Any]] = {node: set(flat.adj[node]) for node in flat}
    chosen: list[Any] = []
    while residual:
        node = min(residual, key=lambda n: (len(residual[n]), str(n)))
        chosen.append(node)
        removed = {node, *residual[node]}
        for gone in removed:
            residual.pop(gone, None)
            for neighbour in flat.adj[gone]:
                if neighbour in residual:
                    residual[neighbour].discard(gone)
    return _sorted_nodes(chosen)


@dataclass(frozen=True)
class DensityReport:
    """Chapter 12 over one network: how full it is, how clustered, and its densest parts."""

    nodes: int
    #: Edges between distinct nodes: a self loop is not one of the pairs §12.1 counts.
    edges: int
    #: ``edges`` over the ``|V|(|V|-1)/2`` (or ``|V|(|V|-1)``, directed) that §12.1 counts. This
    #: is the one number in the report that is **not** taken on the flattened view: a directed
    #: network keeps its directed denominator, because flattening would halve it and answer a
    #: different question. :func:`summary`'s ``density`` is ``networkx``'s, which counts a self
    #: loop as an edge; the two differ only on a network that has one, and none built here does.
    density: float
    #: The §12.1 sentence that has to be printed with the density, with this network's n in it.
    density_note: str
    #: The mean of the unweighted local coefficients: every node counts once.
    average_clustering: float
    #: Onnela's weighted coefficient, averaged. The number :func:`summary` prints.
    weighted_average_clustering: float
    #: Transitivity: 3 * triangles / triads over the whole network.
    global_clustering: float
    #: What the gap between the previous two means on *this* network.
    clustering_note: str
    #: The distribution behind the average, or ``None`` for a network with no nodes.
    local: Summary | None
    cliques: CliqueReport
    #: A maximal independent set (§12.4), greedy and therefore a lower bound on the maximum one.
    independent_set: list[str]
    independent_note: str
    #: Display names for every node this section lists, by the report's usual rule.
    labels: dict[str, str]
    #: The sampling frame carried on the graph, repeated here so the section can print it.
    frame: str
    #: :data:`FLATTENED` when the network was directed, ``""`` otherwise. It applies to the
    #: clustering coefficients, the cliques and the independent set, not to ``density``.
    flattened: str
    #: Whether the nodes come in two kinds (§6.4), by the ``mode`` the builders write. A
    #: two-mode network has no triangle and no clique above an edge by construction, so the
    #: notes say the zeros are structural rather than letting them read as findings.
    two_mode: bool = False


def density_report(graph: nx.Graph, k: int | None = None) -> DensityReport:
    """Everything chapter 12 asks of a network, in one pass, ready to render.

    The clustering coefficients, the cliques and the independent set are computed on the
    undirected, self-loop-free view the chapter assumes (§12.1), so on a directed network they
    are comparable with each other and none of them reads direction; ``flattened`` carries the
    note that says so.

    **The density is the exception**, and deliberately: it stays on the graph as given, with
    §12.1's directed denominator ``|V|(|V|-1)`` when the graph is directed. Flattening first
    would halve the denominator and report the density of a network nobody built. So on a
    directed network this section prints one directed number and four undirected ones, and
    :func:`render_density` says which is which rather than claiming all five are flattened.
    """
    flat, flattened = _undirected_simple(graph)
    nodes, edges, possible, _ = _density_counts(graph)
    average = _network_clustering(graph, "average")
    transitivity = _network_clustering(graph, "global")
    weighted = _local_clustering(graph, weight="weight")
    two_mode = describe_network(graph).bipartite
    clique_report = cliques(graph, k)
    independent = independent_set(graph)
    listed = {node for clique in clique_report.largest for node in clique}
    listed |= {node for clique in clique_report.k_cliques for node in clique}
    listed |= set(independent[:MAX_LISTED])
    return DensityReport(
        nodes=nodes,
        edges=edges,
        density=(edges / possible) if possible else 0.0,
        density_note=density_note(graph),
        average_clustering=average,
        weighted_average_clustering=(sum(weighted.values()) / nodes) if nodes else 0.0,
        global_clustering=transitivity,
        clustering_note=_clustering_note(average, transitivity, flat, two_mode=two_mode),
        local=local_clustering_summary(graph) if nodes else None,
        cliques=clique_report,
        independent_set=independent,
        independent_note=_independent_note(independent, flat),
        labels={str(node): _label(graph, node) for node in listed},
        frame=str(graph.graph.get("frame", "")),
        flattened=flattened,
        two_mode=two_mode,
    )


def _clustering_note(
    average: float, transitivity: float, graph: nx.Graph, *, two_mode: bool = False
) -> str:
    """Why the average and the global coefficient differ here, in the terms §12.2 sets out."""
    if two_mode:
        return (
            "Both coefficients are 0.0 by construction, not as a finding about this corpus: a "
            "triangle needs three mutually adjacent nodes, every edge of a two-mode network "
            "joins the two modes (§6.4), so no triad can close and §12.2's coefficients are "
            "not defined on it. Project onto one mode first -- `--project speakers` or "
            "`--project entities` -- and read the coefficient of the projection, remembering "
            "that a projection closes triangles by construction too: every document's entities "
            "become a clique."
        )
    definitions = (
        "The average counts every node once, so a mass of small tight neighbourhoods lifts it; "
        "the global one is 3 x triangles / triads over the whole network, so a hub whose many "
        "neighbours do not know each other contributes triads quadratically and drags it down "
        "(§12.2). They answer different questions and neither is the other."
    )
    if average == 0.0 and transitivity == 0.0:
        return (
            "No triad in this network closes into a triangle, so both coefficients are 0.0 -- "
            "and on a network with no triads at all they are undefined rather than zero. "
            + definitions
        )
    gap = average - transitivity
    if abs(gap) < 0.05:
        return (
            f"The two agree to within {abs(gap):.4f} here, which is not the usual case. "
            + definitions
        )
    if gap > 0:
        busiest = max((degree for _, degree in graph.degree()), default=0)
        return (
            f"The average sits {gap:.4f} above the global coefficient, which is the hub-heavy "
            "case: the many low-degree nodes each score their small neighbourhood in full, "
            f"while the busiest node's {busiest:,} neighbours open more triads than all of them "
            "close. Quote the one you mean, by name. " + definitions
        )
    return (
        f"The global coefficient sits {-gap:.4f} above the average, which happens when the "
        "triangles are concentrated among the high-degree nodes and the periphery is made of "
        "degree-1 nodes scoring 0.0. " + definitions
    )


def _independent_note(independent: Sequence[str], graph: nx.Graph) -> str:
    """What the size of an independent set does and does not say (§12.4)."""
    nodes = graph.number_of_nodes()
    share = len(independent) / nodes if nodes else 0.0
    return (
        f"{len(independent):,} of {nodes:,} nodes ({share:.0%}) can be picked with no edge "
        "between any two of them. §12.4 separates this *maximal* set, which no node can be "
        "added to, from the *maximum* one, which is the largest in the network: this is greedy "
        "and therefore a lower bound on that (finding it exactly is NP-hard -- Karp, 1972 -- "
        "which the chapter does not say, though its exercise accepts an approximate answer). A "
        "large one means the network is far from complete; how many such sets it takes to "
        "cover every node is its chromatic number (§12.4's graph colouring), which is not "
        "computed here."
    )


def render_density(report: DensityReport) -> list[str]:
    """The ``## Density`` section of a report: chapter 12 with its frame, its n and its nulls."""
    frame = f" Frame: {report.frame}" if report.frame else ""
    lines = [
        "## Density",
        "",
        f"*Atlas ch. 12, over the whole network: n = {report.nodes:,} nodes and "
        f"{report.edges:,} edges.{frame} No null model: §12.1-§12.4 define these as descriptive "
        "quantities, and the chapter's own \"150 times higher than you would expect if its "
        'edges were distributed randomly" reading needs a random-graph baseline, which is '
        "chapter 17's. `graphrag.sna.null` builds one; this section does not.*",
        "",
        f"- **Density** {report.density_note}",
        f"- **Average clustering** {report.average_clustering:.4f} (the mean of the per-node "
        f"coefficients) against **global clustering** {report.global_clustering:.4f} "
        f"(transitivity). {report.clustering_note}",
        f"- **Weighted average clustering** {report.weighted_average_clustering:.4f} — Onnela's "
        "coefficient, which counts a triangle by the geometric mean of its edge weights rather "
        "than as a 1. " + _weighted_row_note(report),
    ]
    local = report.local
    if local is not None:
        lines.append(
            (
                f"- **Local clustering** over {local.count:,} nodes: median {local.median:.4f}, "
                f"quartiles {local.q1:.4f}-{local.q3:.4f}, mean {local.mean:.4f}, maximum "
                f"{local.maximum:.4f}. A node of degree 0 or 1 has no triad to close and counts "
                f"as 0.0. {local.caveat}"
            ).rstrip()
        )
    lines.append(_clique_line(report))
    members = ", ".join(
        report.labels.get(str(node), str(node)) for node in report.independent_set[:MAX_LISTED]
    )
    more = len(report.independent_set) - MAX_LISTED
    lines.append(
        f"- **Independent set** {report.independent_note} Members: {members}"
        + (f", +{more:,} more" if more > 0 else "")
        + "."
    )
    if report.flattened:
        lines.append(
            "- **Direction.** The clustering coefficients, the cliques and the independent set "
            "above are computed " + report.flattened + ", because §12.1 defines all three on an "
            "undirected network and `networkx` reports a transitivity of 0.0 for a directed "
            "triangle rather than refusing it. **The density is not**: it is the directed one, "
            "counting the |V|(|V|-1) ordered pairs a directed network could join, because "
            "flattening first would halve the denominator and describe a network nobody built."
        )
    return [*lines, ""]


def _weighted_row_note(report: DensityReport) -> str:
    """Which of the two weighted averages the summary table is holding, which is not always this.

    On an undirected network the section's Onnela mean and :func:`summary`'s ``average_clustering``
    are the same number. On a directed one they are not: the summary keeps Fagiolo's directed
    coefficient on the graph as given (ATL-06), while this row is Onnela's on the flattened view,
    so saying "this is the summary's number" there would be false.
    """
    if not report.flattened:
        return (
            "This is the number the summary table above calls weighted average clustering "
            "(`average_clustering` in the JSON)."
        )
    return (
        "The summary table's weighted average clustering is **not** this number on a directed "
        "network: it is Fagiolo's directed coefficient, computed on the graph as given and "
        "counting all eight orientations of a triangle, while this row is Onnela's on the "
        "flattened view."
    )


def _clique_line(report: DensityReport) -> str:
    """The clique bullet: the largest ones by name, the distribution, and the caveat."""
    clique = report.cliques
    if clique.total == 0:
        return f"- **Cliques** none: this network has no nodes. {clique.note}"
    if report.two_mode:
        return (
            f"- **Cliques** every one of the {clique.total:,} maximal cliques here is a single "
            "edge, by construction rather than as a finding: no three nodes of a two-mode "
            "network are mutually adjacent (§6.4), so a clique of three is impossible. The "
            "object §12.3 defines for this case is the **biclique** -- every node of one mode "
            "joined to every node of the other, an n,m-clique -- which is not computed here. "
            "Project onto one mode and ask there, remembering that the projection makes a "
            "clique of every document's entities on its own."
        )
    listed = "; ".join(
        "{" + ", ".join(report.labels.get(str(node), str(node)) for node in members) + "}"
        for members in clique.largest
    )
    unlisted = clique.largest_count - len(clique.largest)
    sizes = ", ".join(f"{count:,} of size {size}" for size, count in clique.sizes.items())
    extra = (
        ""
        if clique.k is None
        else (
            f" Complete subgraphs of exactly {clique.k} nodes, maximal or not: "
            f"{clique.k_clique_count:,}."
        )
    )
    return (
        f"- **Cliques** the largest maximal clique has {clique.largest_size} nodes "
        f"({clique.largest_count:,} of that size): {listed}"
        + (f", +{unlisted:,} more" if unlisted > 0 else "")
        + f". Maximal cliques in all: {clique.total:,} ({sizes}). A clique in a co-occurrence "
        "network is often a single document: a passage naming m entities is an m-clique on its "
        f"own (§12.3). {clique.note}{extra}"
    )


def density_payload(report: DensityReport) -> dict[str, Any]:
    """The same section as plain JSON-able data."""
    local = report.local
    return {
        "chapter": "Atlas ch. 12",
        "nodes": report.nodes,
        "edges": report.edges,
        "frame": report.frame,
        "null_model": "none: chapter 12's measures are descriptive",
        "density": report.density,
        "density_note": report.density_note,
        "average_clustering": report.average_clustering,
        "weighted_average_clustering": report.weighted_average_clustering,
        "global_clustering": report.global_clustering,
        "clustering_note": report.clustering_note,
        "local_clustering": None
        if local is None
        else {
            "count": local.count,
            "mean": local.mean,
            "median": local.median,
            "q1": local.q1,
            "q3": local.q3,
            "maximum": local.maximum,
            "heavy_tailed": local.heavy_tailed,
            "caveat": local.caveat,
        },
        "cliques": {
            "total": report.cliques.total,
            "sizes": {str(size): count for size, count in report.cliques.sizes.items()},
            "largest_size": report.cliques.largest_size,
            "largest_count": report.cliques.largest_count,
            "largest": report.cliques.largest,
            "k": report.cliques.k,
            "k_clique_count": report.cliques.k_clique_count,
            "truncated": report.cliques.truncated,
            "note": report.cliques.note,
        },
        "independent_set": {
            "size": len(report.independent_set),
            "members": report.independent_set,
            "note": report.independent_note,
        },
        "labels": report.labels,
        "flattened": report.flattened,
        "two_mode": report.two_mode,
    }


# --------------------------------------------------------------- centralization (§14.8)

#: The one sentence every centralization here has to be read with. §14.8's denominator is "the
#: largest theoretical sum of differences in networks of comparable size", obtained by a star,
#: and a star carries no weights: doubling every weight in a network would double its closeness
#: values, double the numerator, and leave the star where it was, so the ratio would double
#: without the network changing shape. Every centralization in this package is therefore taken
#: on the **binary view** -- the same nodes and edges, every weight 1 -- and is a statement
#: about the shape of the network rather than about the volume behind its ties. It is the one
#: place in a report where the number is not a function of the ranking printed above it.
CENTRALIZATION_NOTE = (
    "Freeman's centralization (§14.8) is the sum of each node's distance from the most central "
    "node, over that same sum on the most centralized network of this size -- a star. A star "
    "has no weights, so both sums are taken on the binary view of this network (every edge "
    "weight 1): with weights the ratio would change when the weights were rescaled and the "
    "shape was not. 1.0 is a star, 0.0 is a network where every node scores the same (a cycle, "
    "a complete graph). It says one thing about the whole network and nothing about any node in "
    "it, and the value depends on which centrality it was computed from -- §14.8's own figure "
    "has degree and betweenness disagreeing about which of two networks is more centralized. "
    "Where a row reads `undefined`, a star is not the maximum for that ranking on this network, "
    "so the chapter's ratio has no denominator; the bullet under the table says which case it is."
)

#: Closed forms for the star's sum of differences, where one exists for the normalisation this
#: package uses. They are what :func:`star_spread` returns on an **undirected** network, and a
#: unit test holds each of them to the value computed from an actual star of the same size.
#: ``degree`` is over ``nx.degree_centrality``'s ``k / (n-1)``, so its star sums to ``n - 2``;
#: ``weighted_degree`` is over raw degrees on the binary view, which is Freeman's own
#: ``(|V|-1)(|V|-2)``; ``betweenness`` is over the normalised betweenness, where the star's
#: centre scores 1 and its leaves 0; ``closeness`` is over ``(n-1)/sum(d)``, whose leaves score
#: ``(n-1)/(2n-3)``. Everything else is computed from a star, because no closed form for this
#: package's normalisation is in print.
STAR_SPREAD: dict[str, Callable[[int], float]] = {
    "degree": lambda n: float(n - 2),
    "weighted_degree": lambda n: float((n - 1) * (n - 2)),
    "betweenness": lambda n: float(n - 1),
    "closeness": lambda n: (n - 1) * (n - 2) / (2 * n - 3),
}

#: A centralization needs a network with a top and a bottom: on two nodes every graph is a star
#: and every star's sum of differences is 0, so the ratio is 0/0.
MIN_CENTRALIZATION_NODES = 3


@dataclass(frozen=True)
class Centralization:
    """One row of §14.8: how centralized this network is, according to one centrality."""

    kind: str
    #: The ratio, or ``None`` when §14.8's denominator is zero for this kind on this network.
    value: float | None
    #: The observed sum of differences from the most central node, on the binary view.
    observed: float
    #: The same sum on a star of the same size: §14.8's "largest theoretical sum".
    star: float
    #: Why ``value`` is ``None``, or what the row has to be read with. Empty when neither.
    note: str = ""


def _binary(graph: nx.Graph) -> nx.Graph:
    """The same nodes and edges with every weight set to 1. See :data:`CENTRALIZATION_NOTE`."""
    plain = nx.DiGraph() if graph.is_directed() else nx.Graph()
    plain.graph.update(graph.graph)
    plain.add_nodes_from(graph.nodes(data=True))
    plain.add_edges_from((u, v, {"weight": 1.0}) for u, v, _ in graph.edges(data=True))
    return plain


def _stars(n: int, *, directed: bool) -> list[nx.Graph]:
    """The stars of ``n`` nodes §14.8 normalises against: one undirected, three directed.

    The chapter says the maximum is "usually" obtained by a star, and on a directed network that
    is one sentence short of a definition: an out-star (the centre points at everyone) maximises
    out-degree and hubbiness, an in-star maximises in-degree, authority and closeness, and
    neither maximises betweenness, because no shortest path passes through a centre that only
    sends or only receives. The reciprocal star -- every leaf pointing at the centre and the
    centre back -- is the one that does. So all three are built and :func:`star_spread` takes the
    largest, which is what "the largest theoretical sum of differences" asks for.
    """
    leaves = range(1, n)
    if not directed:
        undirected = nx.Graph()
        undirected.add_nodes_from(range(n))
        undirected.add_edges_from((0, leaf, {"weight": 1.0}) for leaf in leaves)
        return [undirected]
    stars: list[nx.Graph] = []
    for arcs in (
        [(0, leaf) for leaf in leaves],
        [(leaf, 0) for leaf in leaves],
        [(0, leaf) for leaf in leaves] + [(leaf, 0) for leaf in leaves],
    ):
        star = nx.DiGraph()
        star.add_nodes_from(range(n))
        star.add_edges_from((u, v, {"weight": 1.0}) for u, v in arcs)
        stars.append(star)
    return stars


def _spread(scores: dict[str, float]) -> float:
    """§14.8's first step: the sum of the differences between the top score and every score."""
    if not scores:
        return 0.0
    peak = max(scores.values())
    return float(sum(peak - value for value in scores.values()))


def star_spread(n: int, kind: str, *, directed: bool = False) -> float:
    """The denominator of §14.8: the sum of differences a star of ``n`` nodes produces.

    Read from :data:`STAR_SPREAD` where a closed form exists for this package's normalisation
    and the network is undirected; computed from an actual star otherwise, which is the general
    procedure the chapter describes ("you calculate what would be the largest theoretical sum of
    differences in networks of comparable size"). On a directed network it is the largest of the
    three orientations :func:`_stars` builds.

    Returns 0.0 when a star gives every node the same score -- coreness does, since a star is a
    1-core and nothing deeper, and so does the reach of an undirected star, where everybody
    reaches everybody within two hops. That is not a small denominator to be divided by
    carefully: it means the star is not the most centralized network for that measure, so
    §14.8's ratio is not defined at all, and :func:`centralization` raises rather than returning
    an infinity.

    Those two are answered without building a star, and not only for tidiness: a reach over a
    star of n nodes visits the whole star from every node, which is quadratic, and this is the
    one denominator a report on a large network still asks for. The unit test holds both against
    a star actually built, at sizes where building one is cheap.
    """
    if n < MIN_CENTRALIZATION_NODES:
        return 0.0
    if kind == "coreness":
        return 0.0  # every node of a star, oriented any way, sits in the 1-core and no deeper
    if kind == "reach":
        # The out-star is the maximum: its centre reaches everything and its leaves nothing.
        # Undirected, and reciprocally directed, everybody reaches everybody inside two hops.
        return float(n - 1) if directed else 0.0
    if not directed and kind in STAR_SPREAD:
        return STAR_SPREAD[kind](n)
    return max(_spread(centrality(star, kind)) for star in _stars(n, directed=directed))


def centralization(graph: nx.Graph, kind: str) -> float:
    """How centralized this network is, by Freeman's ratio over one centrality (§14.8).

    "You sum the centrality differences between the most central node in the network and all
    other nodes. Then you calculate what would be the largest theoretical sum of differences in
    networks of comparable size. Usually the maximum is obtained by a star graph with the same
    number of nodes of your original network. The ratio between the two is the degree of
    centralization."

    1.0 is a star and 0.0 is a network whose nodes all score the same. It is a number about the
    network, never about a node: a report that quotes it beside a ranking is saying how much of
    the ranking's spread sits at the top, not who is at the top.

    **Which copy of the network.** The binary one -- see :data:`CENTRALIZATION_NOTE` -- because
    the star it is divided by has no weights and the ratio would otherwise move when the weights
    were rescaled. On a directed network the star is the best-oriented of the three
    (:func:`_stars`).

    **Which centrality matters, and the chapter says so twice**: "depending on the centrality
    measure you picked, you're going to obtain different results", and Figure 14.14 has degree
    overstating centralization on one network and betweenness overstating it on another. Print
    the table (:func:`centralization_table`), not one row of it.

    §14.8's other route, the information-theoretic one -- turn the degree sequence into a
    probability distribution and take Shannon's entropy -- is **not** computed here, for the
    reason the chapter gives against it: "it's not immediately obvious how centralized a network
    is by simply looking at the entropy value", it grows with n, and normalising it needs the
    minimum possible entropy for a graph of that size, "which can be tricky".
    :func:`graphrag.sna.stats.entropy` is the entropy itself, for a caller who wants it -- but
    it counts the labels it is handed, so it returns §14.8's *degree* entropy only if each node
    is passed once per edge end (``[node] * degree``), which is what makes the chapter's
    "probability that an edge will be attached to this node", ``k_v / sum(k)``, the distribution
    being measured.

    Raises ``ValueError`` for a network of fewer than :data:`MIN_CENTRALIZATION_NODES` nodes,
    and for three cases where a star is not §14.8's maximum, which is not the same thing as a
    network that failed to be centralized: ``coreness``, where every node of a star sits in the
    1-core; ``reach`` on an undirected network, where every node of a star reaches everything;
    and ``eigenvector`` on a network in more than one piece, where :func:`_disconnected_note`
    explains what goes wrong. In all three the chapter's ratio has no denominator, and
    :func:`centralization_table` keeps the row with the reason in it.
    """
    nodes = graph.number_of_nodes()
    if nodes < MIN_CENTRALIZATION_NODES:
        msg = (
            f"centralization() needs at least {MIN_CENTRALIZATION_NODES} nodes (§14.8 divides by "
            f"a star of the same size, and below that every graph is a star), got {nodes}"
        )
        raise ValueError(msg)
    if kind == "eigenvector" and _components(graph) > 1:
        raise ValueError(_disconnected_note(graph))
    denominator = star_spread(nodes, kind, directed=graph.is_directed())
    if denominator <= 0.0:
        msg = (
            f"centralization() is undefined for {kind!r} on this network: every node of a star "
            f"of {nodes} nodes gets the same {kind} score, so §14.8's denominator -- the largest "
            "theoretical sum of differences -- is zero and a star is not the most centralized "
            "network for this measure."
        )
        raise ValueError(msg)
    return _spread(centrality(_binary(graph), kind)) / denominator


def _components(graph: nx.Graph) -> int:
    """How many pieces this network is in, direction ignored -- the count §14.8 cares about.

    The eigenvector is solved on the flattened view (:func:`_eigenvector`), so the connectivity
    that decides whether its centralization is defined is the flattened one: a directed network
    whose arcs all run one way is still one piece for this purpose.
    """
    flat, _ = undirected_view(graph)
    return int(nx.number_connected_components(flat)) if flat.number_of_nodes() else 0


def _disconnected_note(graph: nx.Graph) -> str:
    """Why §14.8 has no denominator for the eigenvector once the network is in pieces."""
    return (
        f"centralization() is undefined for 'eigenvector' on this network: it is in "
        f"{_components(graph):,} pieces, and the principal eigenvector of a disconnected "
        "adjacency matrix belongs to one component -- every node outside it scores ~0 once the "
        "vector is normalised to unit length. The observed sum of differences can then exceed "
        "the star's, which is §14.8's assumption failing rather than a network more centralized "
        "than a star: the chapter says the maximum is *usually* a star, and this is one of the "
        "cases where it is not. PageRank's teleportation is §14.4's own answer to exactly this "
        "problem, so read the `pagerank` row, or take the eigenvector centralization on the "
        "largest component alone and say that is what it describes."
    )


def centralization_table(
    graph: nx.Graph, kinds: Sequence[str] | None = None
) -> list[Centralization]:
    """§14.8 for every ranking this network supports, with the undefined ones saying why.

    ``kinds`` defaults to :func:`centralities_for`, so the table has one row per centrality the
    report printed. A kind §14.8's ratio is not defined for keeps its row with ``value=None`` and
    the reason in ``note``: dropping the row would leave a reader to wonder whether the measure
    had been forgotten. Such a row carries no ``star`` either -- there is no denominator to
    print, and computing one would be a number that does nothing but look like a maximum.

    The weighted degrees are not a second opinion here. On the binary view §14.8 requires,
    ``weighted_degree`` *is* the degree, so those rows repeat their unweighted counterparts by
    construction and the note says so.
    """
    rows: list[Centralization] = []
    nodes = graph.number_of_nodes()
    for kind in kinds if kinds is not None else centralities_for(graph):
        try:
            value = centralization(graph, kind)
        except ValueError as exc:
            rows.append(
                Centralization(kind=kind, value=None, observed=0.0, star=0.0, note=str(exc))
            )
            continue
        star = star_spread(nodes, kind, directed=graph.is_directed())
        rows.append(
            Centralization(
                kind=kind,
                value=value,
                observed=value * star,
                star=star,
                note=_centralization_note(kind),
            )
        )
    return rows


def _centralization_note(kind: str) -> str:
    """What a row of the table has to be read with, beyond :data:`CENTRALIZATION_NOTE`."""
    if kind.startswith("weighted_"):
        return (
            "On the binary view §14.8 is computed on, the weighted degree is the degree, so this "
            f"row and `{kind.removeprefix('weighted_')}` are the same number."
        )
    return ""
