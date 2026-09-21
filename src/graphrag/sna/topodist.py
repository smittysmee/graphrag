"""Topological distances (ch. 48): how similar two networks' topologies are, node alignment by
id, and similarity network fusion. Returns dataclasses and a graph; prints nothing on its own --
:func:`render_topology` is what ``sna compare --topology`` calls, and :func:`compare_topology`
is the one entry point a caller reaches for two whole graphs, whether they are two windows of one
persona network (what ``sna compare`` builds) or two different networks of a persona compared
directly (``sna export``'s speakers network against its entities network, say).

**Network similarity (§48.1).** The chapter surveys five families -- global property comparison,
pairwise node similarity, graph edit distance, holistic signature vectors, and information theory
-- and names, by citation, four concrete methods this ticket builds: **spectral distance**
(:func:`spectral_distance`, p. 705-706, comparing the eigenvalues of the two graphs' Laplacians),
**NetSimile** (:func:`netsimile_distance`, p. 704, a holistic signature vector), **DeltaCon**
(:func:`delta_con`, p. 702, a node-affinity comparison) and **portrait divergence**
(:func:`portrait_divergence`, the chapter's information-theoretic family generalised to the
per-node shortest-path-length distribution the book's own §13.3 histogram already describes).
The chapter gives the *shape* of each -- what it compares and why -- rather than every constant;
where a formula is not printed in extractable text, the closed form is the one the method's own
paper defines, and each function's docstring says so and cites it. None of the four is the "right"
one -- p. 699's own figure shows two networks with the same global properties and very different
clustering, and the book's verdict is that this is a *feature* if the statistics you chose matter
to your question and a trap if you are hunting for something "universal" -- so
:func:`compare_topology` runs all four and reports them side by side rather than picking a
winner, and :data:`TOPODIST_RULES` in :mod:`graphrag.sna.guide` says why that is the right way
to read them.

**Network alignment (§48.2).** Every method above except the affinity comparison is
alignment-free: it reduces each graph to a number or a vector without asking which node in one
graph is which node in the other. DeltaCon is not -- its affinity matrix is indexed by node, so
comparing two affinity matrices entry by entry needs the two graphs' rows to mean the same node.
:func:`align_by_id` is the "we already know" case the chapter opens the section with (p. 706):
two dated snapshots, or two networks built from a persona whose entities and speakers keep the
same id across builds, are aligned for free because their node ids already name the same real
thing. The harder case the chapter spends the rest of §48.2 on -- pairwise node similarity,
maximum common subgraph, graphlet-degree vectors, MAGNA's genetic search -- is for two networks
whose ids carry no relationship at all, and is not built here: :func:`align_by_id` says so in its
own note whenever the two graphs it was given share too little of their id space for identity to
be doing any of the work.

**Network fusion (§48.3).** :func:`fuse_networks` is the chapter's own worked example (Figure
48.10, p. 709): average an edge's weight across every aligned observation that carries it, and
keep the edge above a stated threshold. The chapter's citation (Wang et al. 2014, "Similarity
Network Fusion") is a substantially more sophisticated iterative kernel method built for genomic
data at a different scale of "multiple observations", and is not built here -- see
:func:`fuse_networks` for why the book's own toy algorithm is the one this ticket commits to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats

from graphrag.sna.matrices import Node, adjacency, laplacian
from graphrag.sna.matrices import eigenpairs as _eigenpairs
from graphrag.sna.measures import clustering, undirected_view

__all__ = [
    "Alignment",
    "DeltaConResult",
    "NetSimileDistance",
    "PortraitDivergenceResult",
    "SpectralDistance",
    "TopologicalDistances",
    "align_by_id",
    "compare_topology",
    "delta_con",
    "fuse_networks",
    "netsimile_distance",
    "netsimile_signature",
    "portrait_divergence",
    "render_topology",
    "spectral_distance",
]

#: The seven per-node features NetSimile aggregates (Berlingerio, Koutra, Eliassi-Rad and
#: Faloutsos, 2012, "NetSimile: A Scalable Approach to Size-Independent Network Similarity"),
#: in the order :func:`netsimile_signature` computes and reports them.
NETSIMILE_FEATURES: tuple[str, ...] = (
    "degree",
    "clustering",
    "avg_neighbor_degree",
    "avg_neighbor_clustering",
    "egonet_edges",
    "egonet_edges_leaving",
    "two_hop_neighbors",
)

#: The five moments each feature is aggregated into, the same five the NetSimile paper uses.
NETSIMILE_MOMENTS: tuple[str, ...] = ("median", "mean", "std", "skew", "kurtosis")

#: Below this shared share of the union of two node-id sets, identity alignment is mostly padding
#: rather than correspondence, and :func:`align_by_id` says so. Not a hard cutoff: even at 100%
#: overlap the ids may still name unrelated things, which is a judgement no share can make.
LOW_OVERLAP = 0.5


# ----------------------------------------------------------------------------- §48.2 alignment


@dataclass(frozen=True)
class Alignment:
    """A node-to-node mapping between two graphs, built the way §48.2 opens the section (p. 706):
    "two networks are aligned if we have a node to node mapping... Many networks are naturally
    aligned [...] since we have the same ids on the nodes, we have the alignment for free." This
    is that free case -- matching by shared id -- and nothing more; see the module docstring for
    what is not built.
    """

    #: The union of both graphs' node ids, sorted -- the row/column order a caller aligns to.
    nodes: tuple[Node, ...]
    #: The ids present in *both* graphs -- the only ones identity alignment actually maps.
    shared: tuple[Node, ...]
    #: ``len(shared) / len(nodes)``. 1.0 means the two graphs describe the same node set; 0.0
    #: means they share no id at all, so every row of an aligned matrix compares one graph's real
    #: entry against the other's zero-padding.
    overlap: float
    #: Empty above :data:`LOW_OVERLAP`; otherwise the caveat a report has to print next to
    #: anything built from this alignment.
    note: str


def _sorted_nodes(nodes: set[Node]) -> list[Node]:
    """``nodes`` sorted, falling back to sorting by ``str`` when the ids do not natively compare
    (two graphs whose node ids are of different types -- an ``int``-labelled synthetic graph and
    a ``str``-labelled persona network -- share no id either way, but Python's ``<`` refuses to
    even ask)."""
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=str)


def align_by_id(g1: nx.Graph, g2: nx.Graph) -> Alignment:
    """Align ``g1`` and ``g2`` by shared node id (§48.2's "we have the alignment for free" case).

    The union of both node sets becomes the common row order; a node one graph does not have
    still gets a row, padded with degree zero, so the two graphs compare over the same axes. This
    is the right alignment for two windows of one persona network, or two dated snapshots, where
    the ids *are* the same entities across builds -- it is the wrong one for two networks whose
    ids happen to collide (two corpora that both number their nodes 0, 1, 2, ...) or that name the
    same real entities under different ids, and :attr:`Alignment.note` is how a caller is told
    which situation it is looking at rather than trusting the number silently.
    """
    nodes1, nodes2 = set(g1.nodes), set(g2.nodes)
    shared = nodes1 & nodes2
    union = nodes1 | nodes2
    overlap = len(shared) / len(union) if union else 0.0
    note = ""
    if union and overlap == 0.0:
        note = (
            "the two graphs share no node id at all: identity alignment maps nothing, so every "
            "row of an aligned comparison is one graph's real entry against the other's zero "
            "padding. §48.2's harder alignment methods -- pairwise node similarity, maximum "
            "common subgraph, graphlet-degree vectors -- would be needed to find a real "
            "correspondence, and are not built here; without one, an id-aligned comparison of "
            "these two graphs answers 'how different are the node sets', not 'how different is "
            "the shared structure'."
        )
    elif union and overlap < LOW_OVERLAP:
        note = (
            f"only {overlap:.0%} of the union of node ids is shared between the two graphs: an "
            "id-aligned comparison is mostly comparing padding, not structure. Read it beside "
            "the node counts before treating it as a statement about shared topology."
        )
    return Alignment(
        nodes=tuple(_sorted_nodes(union)),
        shared=tuple(_sorted_nodes(shared)),
        overlap=overlap,
        note=note,
    )


def _padded(graph: nx.Graph, nodes: Sequence[Node]) -> nx.Graph:
    """``graph`` with every id in ``nodes`` present, the ones it lacked added as isolated nodes."""
    padded = graph.copy()
    padded.add_nodes_from(n for n in nodes if n not in padded)
    return padded


# ----------------------------------------------------------------------------- §48.1 spectral


@dataclass(frozen=True)
class SpectralDistance:
    """The Euclidean distance between two graphs' normalised-Laplacian spectra (p. 705-706:
    "comparing the eigenvectors of their Laplacians. Similar networks will experience similar
    spreading patterns, which are reflected in their spectra", citing Banerjee 2012)."""

    distance: float
    #: How many eigenvalues were compared after zero-padding the shorter spectrum.
    dims: int
    provenance: dict[str, Any]


def spectral_distance(g1: nx.Graph, g2: nx.Graph, *, k: int | None = None) -> SpectralDistance:
    """The Euclidean distance between the sorted, non-trivial eigenvalues of ``g1`` and ``g2``'s
    symmetric normalised Laplacians (§8.4's ``D^-1/2 L D^-1/2``, the same matrix
    :func:`graphrag.sna.embed.spectral` and :func:`graphrag.sna.cluster.spectral_embedding` use,
    so a spectral embedding and this distance always agree about which matrix answered).

    The Laplacian's smallest eigenvalue is always 0 (§8.4) and carries no information beyond "the
    graph has at least one node", so it is discarded on both sides the way ``embed.spectral``
    discards it; every eigenvalue after it is kept and compared ascending. ``k`` truncates each
    side to its ``k`` smallest non-trivial eigenvalues first, which is the useful move on a large
    network where only the coarse, global structure the smallest eigenvalues carry is the
    question; the default keeps the full spectrum, which is exact on the small networks the
    known-answer tests use and expensive on nothing this package's persona corpora produce.

    **Two graphs of different order are not refused.** The shorter spectrum is zero-padded to the
    longer's length before the distance is taken, which is a stated choice, not a workaround: a
    16-node graph missing eight of a 24-node graph's largest eigenvalues is missing real
    structure, and padding with zero scores that absence as difference rather than hiding it by
    truncating the larger graph down to size. This is exactly why the chapter's own caveat that
    size confounds every distance in this chapter applies here too (see
    :data:`graphrag.sna.guide.TOPODIST_RULES`) -- a bigger graph is *farther* from a smaller one
    under this measure for its size alone, before anything about its shape is considered.

    A directed graph is flattened first (§6.2), the same convention every Laplacian-based
    function in this package follows.

    Undefined, and raised on, for a graph with fewer than two nodes: there is no non-trivial
    eigenvalue to compare.
    """
    flat1, note1 = undirected_view(g1)
    flat2, note2 = undirected_view(g2)
    n1, n2 = flat1.number_of_nodes(), flat2.number_of_nodes()
    if n1 < 2 or n2 < 2:
        msg = "spectral_distance needs at least two nodes on each side to have a spectrum to read"
        raise ValueError(msg)
    lap1, _ = laplacian(flat1, kind="symmetric", weight=None)
    lap2, _ = laplacian(flat2, kind="symmetric", weight=None)
    values1, _ = _eigenpairs(lap1, largest=False)
    values2, _ = _eigenpairs(lap2, largest=False)
    nontrivial1 = values1[1:]
    nontrivial2 = values2[1:]
    if k is not None:
        nontrivial1 = nontrivial1[:k]
        nontrivial2 = nontrivial2[:k]
    dims = max(len(nontrivial1), len(nontrivial2))
    padded1 = np.zeros(dims)
    padded1[: len(nontrivial1)] = nontrivial1
    padded2 = np.zeros(dims)
    padded2[: len(nontrivial2)] = nontrivial2
    distance = float(np.linalg.norm(padded1 - padded2))
    provenance: dict[str, Any] = {
        "chapter": "§48.1",
        "matrix": "symmetric normalised Laplacian (D^-1/2 L D^-1/2, §8.4)",
        "source": "Banerjee 2012, Biosystems 107(3):186-196",
        "n_nodes": (n1, n2),
        "k_requested": k,
        "dims_compared": dims,
        "directed_flattened_notes": (note1, note2),
    }
    return SpectralDistance(distance=distance, dims=dims, provenance=provenance)


# ----------------------------------------------------------------------------- §48.1 NetSimile


def _egonet_features(
    graph: nx.Graph, degree: dict[Node, int], clustering_: dict[Node, float]
) -> dict[Node, list[float]]:
    """The seven NetSimile features (p. 704) for every node of ``graph``."""
    features: dict[Node, list[float]] = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        deg = degree[node]
        cc = clustering_[node]
        neighbor_degrees = [degree[n] for n in neighbors]
        neighbor_clustering = [clustering_[n] for n in neighbors]
        avg_neighbor_degree = float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0
        avg_neighbor_clustering = (
            float(np.mean(neighbor_clustering)) if neighbor_clustering else 0.0
        )
        ego_nodes = {node, *neighbors}
        ego_edges = sum(1 for u, v in graph.edges(ego_nodes) if u in ego_nodes and v in ego_nodes)
        leaving = sum(1 for u in ego_nodes for v in graph.neighbors(u) if v not in ego_nodes)
        two_hop = {n for u in neighbors for n in graph.neighbors(u) if n not in ego_nodes}
        features[node] = [
            float(deg),
            float(cc),
            avg_neighbor_degree,
            avg_neighbor_clustering,
            float(ego_edges),
            float(leaving),
            float(len(two_hop)),
        ]
    return features


def _moments(values: Sequence[float]) -> list[float]:
    """Median, mean, std, skew, kurtosis (§48.1's five, Berlingerio et al. 2012).

    ``skew`` and ``kurtosis`` are 0.0 for a constant feature -- every node of a star graph has
    clustering coefficient 0, and a third or fourth moment of a sample with no spread is not
    "undefined", it is the flat sample this package's own :func:`graphrag.sna.stats.describe`
    already calls 0.0 for the same reason. ``scipy`` returns ``nan`` there instead, which this
    guards against so a signature vector never carries one.
    """
    sample = np.asarray(values, dtype=np.float64)
    std = float(np.std(sample))
    skew = float(scipy_stats.skew(sample)) if std > 0 else 0.0
    kurtosis = float(scipy_stats.kurtosis(sample)) if std > 0 else 0.0
    return [float(np.median(sample)), float(np.mean(sample)), std, skew, kurtosis]


def netsimile_signature(graph: nx.Graph) -> np.ndarray:
    """``graph``'s NetSimile signature vector (p. 704): the seven per-node features of
    :data:`NETSIMILE_FEATURES`, each aggregated into the five moments of
    :data:`NETSIMILE_MOMENTS`, laid out feature-major (``feature * 5 + moment``) -- 35 entries.

    The seven features are computed directly here rather than through
    :func:`graphrag.sna.measures.centrality`, because that function's ``"degree"`` is *degree
    centrality* (``d / (n-1)``, §14.1) and NetSimile's own definition (p. 704) is the raw degree;
    :func:`graphrag.sna.measures.clustering` is reused as-is for the local clustering coefficient,
    since that one is not renormalised anywhere in this package. The egonet of a node is itself
    plus its direct neighbours (radius 1); "edges leaving the egonet" counts every edge with
    exactly one endpoint inside it, which on a self-loop-free simple graph is the same as counting
    each such edge once from the ego-set side.

    A directed graph is flattened first (§6.2). Undefined, and raised on, for a graph with no
    nodes: there is nothing to aggregate a moment over.
    """
    flat, _ = undirected_view(graph)
    if flat.number_of_nodes() == 0:
        msg = "netsimile_signature needs at least one node"
        raise ValueError(msg)
    degree = dict(flat.degree())
    # kind="local" always returns one value per node (see clustering()'s own docstring).
    clustering_ = cast("dict[Node, float]", clustering(flat, "local"))
    per_node = _egonet_features(flat, degree, clustering_)
    columns = zip(*per_node.values(), strict=True)
    signature: list[float] = []
    for column in columns:
        signature += _moments(column)
    return np.array(signature, dtype=np.float64)


@dataclass(frozen=True)
class NetSimileDistance:
    """The Canberra distance (p. 704, "the authors focus specifically on the Camberra distance")
    between two graphs' NetSimile signature vectors."""

    distance: float
    signature_a: np.ndarray
    signature_b: np.ndarray
    provenance: dict[str, Any]


def netsimile_distance(g1: nx.Graph, g2: nx.Graph) -> NetSimileDistance:
    """NetSimile's own distance (p. 704): the Canberra distance between the two graphs'
    :func:`netsimile_signature` vectors, ``sum(|a_i - b_i| / (|a_i| + |b_i|))`` over the 35
    entries, 0 where both entries are 0. Size-independent by the paper's own name -- every
    feature is a per-node quantity aggregated by a moment, never a sum of nodes -- so two graphs
    of very different order are compared exactly as directly as two of the same order, unlike
    :func:`spectral_distance` and :func:`portrait_divergence`.
    """
    sig1, sig2 = netsimile_signature(g1), netsimile_signature(g2)
    denom = np.abs(sig1) + np.abs(sig2)
    terms = np.divide(np.abs(sig1 - sig2), denom, out=np.zeros_like(denom), where=denom > 0)
    distance = float(terms.sum())
    provenance: dict[str, Any] = {
        "chapter": "§48.1",
        "source": "Berlingerio, Koutra, Eliassi-Rad and Faloutsos, 2012, arXiv:1209.2684",
        "features": NETSIMILE_FEATURES,
        "moments": NETSIMILE_MOMENTS,
        "n_nodes": (g1.number_of_nodes(), g2.number_of_nodes()),
    }
    return NetSimileDistance(
        distance=distance, signature_a=sig1, signature_b=sig2, provenance=provenance
    )


# ----------------------------------------------------------------------------- §48.1 DeltaCon


@dataclass(frozen=True)
class DeltaConResult:
    """DeltaCon (p. 702, citing Koutra, Vogelstein and Faloutsos 2013): each graph's node-affinity
    matrix under fast belief propagation, and the distance and similarity between them."""

    #: Root Euclidean (Matusita) distance between the two affinity matrices, >= 0.
    distance: float
    #: ``1 / (1 + distance)``: the paper's own similarity score, in ``(0, 1]``.
    similarity: float
    affinity_a: np.ndarray
    affinity_b: np.ndarray
    epsilon_a: float
    epsilon_b: float
    alignment: Alignment
    provenance: dict[str, Any]


def _fabp_affinity(graph: nx.Graph, nodes: Sequence[Node]) -> tuple[np.ndarray, float]:
    """The fast-belief-propagation affinity matrix ``S = [I + eps^2 D - eps A]^-1`` (Koutra,
    Vogelstein and Faloutsos, 2013, "DeltaCon: A Principled Massive-Graph Similarity Function"),
    with ``eps = 1 / (1 + max degree)`` and every id in ``nodes`` present, isolated where
    ``graph`` did not have it (§48.2's identity alignment)."""
    padded = _padded(graph, nodes)
    matrix, order = adjacency(padded, nodes=list(nodes), weight=None)
    degrees = matrix.sum(axis=1)
    max_degree = float(degrees.max()) if len(degrees) else 0.0
    epsilon = 1.0 / (1.0 + max_degree)
    identity = np.eye(len(order))
    belief = identity + (epsilon**2) * np.diag(degrees) - epsilon * matrix
    affinity = np.linalg.inv(belief)
    return affinity, epsilon


def delta_con(g1: nx.Graph, g2: nx.Graph) -> DeltaConResult:
    """DeltaCon's similarity between ``g1`` and ``g2`` (p. 702): the fast-belief-propagation
    affinity matrix of each, node-aligned by :func:`align_by_id`, compared by the Matusita
    (root Euclidean) distance ``sqrt(sum((sqrt(S1_ij) - sqrt(S2_ij))^2))`` the paper defines.

    Each graph's own ``epsilon`` is drawn from its own maximum degree, exactly as the paper
    computes it (the padded nodes an alignment adds have degree 0 and do not change either
    graph's own maximum). Two graphs that share no node id at all still return a number -- every
    row is then one graph's real affinity against the other's, which is the identity matrix's
    corresponding row, since an isolated node's own affinity to itself is 1 and to everyone else
    is 0 -- but :attr:`DeltaConResult.alignment` carries the note that says the number is reading
    the node sets, not the structure, in that case (see :func:`align_by_id`).

    Undefined, and raised on, for two graphs that together have no node at all.
    """
    alignment = align_by_id(g1, g2)
    if not alignment.nodes:
        msg = "delta_con needs at least one node between the two graphs"
        raise ValueError(msg)
    affinity1, epsilon1 = _fabp_affinity(g1, alignment.nodes)
    affinity2, epsilon2 = _fabp_affinity(g2, alignment.nodes)
    distance = float(np.sqrt(np.sum((np.sqrt(affinity1) - np.sqrt(affinity2)) ** 2)))
    similarity = 1.0 / (1.0 + distance)
    provenance: dict[str, Any] = {
        "chapter": "§48.1",
        "source": "Koutra, Vogelstein and Faloutsos, 2013, SDM'13",
        "n_nodes_aligned": len(alignment.nodes),
        "overlap": alignment.overlap,
    }
    return DeltaConResult(
        distance=distance,
        similarity=similarity,
        affinity_a=affinity1,
        affinity_b=affinity2,
        epsilon_a=epsilon1,
        epsilon_b=epsilon2,
        alignment=alignment,
        provenance=provenance,
    )


# ------------------------------------------------------------------- §48.1 portrait divergence


def _portrait(graph: nx.Graph) -> tuple[np.ndarray, int]:
    """The network portrait ``B`` (Bagrow and Bollt, 2019, "An Information-Theoretic, All-Scales
    Approach to Comparing Networks", Applied Network Science 4:45): ``B[l, k]`` is the number of
    nodes with exactly ``k`` other nodes at shortest-path distance ``l``, for ``l`` from 0 up to
    the graph's own diameter. Row 0 is always ``B[0, 1] = n``, because every node has exactly
    itself at distance 0 -- the pair a shortest-path search never records is filled in here
    rather than left as a hole. A node with no node at exactly distance ``l`` for some
    ``0 < l <= diameter`` (it ran out of graph before reaching that far) counts into ``B[l, 0]``
    the same way, so that **every row sums to n**: that invariant is what lets ``B[l, :] / n`` be
    read as a probability distribution over ``k`` conditional on the shell ``l``, which is the
    quantity :func:`portrait_divergence` compares. An extra final row, index ``diameter + 1``,
    counts *unreachable* pairs the same way -- ``B[diameter + 1, k]`` is the number of nodes with
    ``k`` nodes they cannot reach at all -- which is this function's own reading of a network the
    paper's own examples assume is connected (§13.3's own convention for a disconnected network:
    an unreachable pair's length is not a number, so it needs its own bin rather than a
    fabricated one). Returns ``(B, diameter)``; ``B`` has ``diameter + 2`` rows and ``n`` columns.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros((1, 0)), 0
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    diameter = 0
    per_node: list[tuple[dict[int, int], int]] = []
    for node in graph.nodes:
        reached = lengths.get(node, {})
        counts: dict[int, int] = {}
        for other, dist in reached.items():
            if other == node:
                continue
            counts[dist] = counts.get(dist, 0) + 1
            diameter = max(diameter, dist)
        unreachable = n - 1 - sum(counts.values())
        per_node.append((counts, unreachable))
    rows = diameter + 2
    portrait = np.zeros((rows, n))
    for counts, unreachable in per_node:
        # l = 0: every node has exactly itself at distance 0 (k = 1), which ``reached`` never
        # records because the self pair is skipped above.
        portrait[0, 1] += 1
        # l = 1 .. diameter: a node with no node at exactly this distance still has to be
        # counted, at k = 0 -- either because l sits beyond its own eccentricity (it has
        # already reached everyone it can by then) or because it sits in a smaller component.
        # Without this, a row would not sum to n and B[l, :] / n would not be a probability
        # distribution over k, which is what the paper's Pr(k | l) requires.
        for dist in range(1, diameter + 1):
            portrait[dist, counts.get(dist, 0)] += 1
        portrait[diameter + 1, unreachable] += 1
    return portrait, diameter


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits, ``0 log 0 := 0`` (§3.5)."""
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


@dataclass(frozen=True)
class PortraitDivergenceResult:
    """The Jensen-Shannon divergence, in bits, between two graphs' network portraits."""

    #: In ``[0, 1]`` bits: 0 for the same portrait, 1 for two portraits with disjoint support.
    divergence: float
    shells: int
    provenance: dict[str, Any]


def portrait_divergence(g1: nx.Graph, g2: nx.Graph) -> PortraitDivergenceResult:
    """The portrait divergence between ``g1`` and ``g2``: the Jensen-Shannon divergence between
    their two network portraits (:func:`_portrait`), read as joint distributions over (shell,
    count) the way the paper does -- ``P(l, k) = B[l, k] / (N * shells)``, uniform over each
    graph's own set of shells (its diameter plus one, plus the unreachable bin), conditional on
    the shell on ``B[l, k] / N``.

    The two portraits are padded to a common shape before comparison -- ``shells = max`` of the
    two graphs' own ``diameter + 2``, columns up to ``max(N1, N2)`` -- which is the same size
    confound :func:`spectral_distance` names: a bigger graph reaches more shells and holds more
    nodes at a given shell than a smaller one can, so part of the divergence between two
    differently-sized graphs is size before it is shape. This is the alignment-free member of the
    four: it reduces each graph to a distribution over path-length shells before comparing, and
    never asks which node in one graph is which node in the other.

    Undefined, and raised on, for a graph with no nodes.
    """
    flat1, note1 = undirected_view(g1)
    flat2, note2 = undirected_view(g2)
    if flat1.number_of_nodes() == 0 or flat2.number_of_nodes() == 0:
        msg = "portrait_divergence needs at least one node on each side"
        raise ValueError(msg)
    portrait1, diameter1 = _portrait(flat1)
    portrait2, diameter2 = _portrait(flat2)
    n1, n2 = flat1.number_of_nodes(), flat2.number_of_nodes()
    shells = max(portrait1.shape[0], portrait2.shape[0])
    columns = max(portrait1.shape[1], portrait2.shape[1])
    padded1 = np.zeros((shells, columns))
    padded1[: portrait1.shape[0], : portrait1.shape[1]] = portrait1
    padded2 = np.zeros((shells, columns))
    padded2[: portrait2.shape[0], : portrait2.shape[1]] = portrait2
    p1 = padded1 / (n1 * shells)
    p2 = padded2 / (n2 * shells)
    mixture = 0.5 * (p1 + p2)
    entropy_mixture = _shannon_entropy(mixture.ravel())
    entropy1 = _shannon_entropy(p1.ravel())
    entropy2 = _shannon_entropy(p2.ravel())
    divergence = entropy_mixture - 0.5 * entropy1 - 0.5 * entropy2
    provenance: dict[str, Any] = {
        "chapter": "§48.1",
        "source": "Bagrow and Bollt, 2019, Applied Network Science 4:45",
        "diameters": (diameter1, diameter2),
        "shells_compared": shells,
        "n_nodes": (n1, n2),
        "directed_flattened_notes": (note1, note2),
    }
    return PortraitDivergenceResult(
        divergence=max(divergence, 0.0), shells=shells, provenance=provenance
    )


# ----------------------------------------------------------------------------- gathered report


@dataclass(frozen=True)
class TopologicalDistances:
    """All four §48.1 distances between two graphs, gathered for one report section."""

    spectral: SpectralDistance
    netsimile: NetSimileDistance
    delta_con: DeltaConResult
    portrait: PortraitDivergenceResult
    notes: tuple[str, ...] = field(default_factory=tuple)


def compare_topology(g1: nx.Graph, g2: nx.Graph, *, k: int | None = None) -> TopologicalDistances:
    """Every §48.1 distance between ``g1`` and ``g2``, run once and reported together, because
    p. 699's own figure is the chapter's warning against reading any one of them alone: two
    networks can share every global property one method checks and disagree on the one it did
    not. ``k`` is forwarded to :func:`spectral_distance` only; the other three have no comparable
    knob.
    """
    spectral = spectral_distance(g1, g2, k=k)
    netsimile = netsimile_distance(g1, g2)
    delta = delta_con(g1, g2)
    portrait = portrait_divergence(g1, g2)
    notes = tuple(note for note in (delta.alignment.note,) if note)
    return TopologicalDistances(
        spectral=spectral, netsimile=netsimile, delta_con=delta, portrait=portrait, notes=notes
    )


def _num(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _size_caveat(n1: int, n2: int) -> str:
    """A short, inline disclosure for a row whose padding scores size as difference (see
    :data:`graphrag.sna.guide.TOPODIST_RULES`'s "size confounds every distance" rule) --
    empty when the two graphs are the same order, where the padding adds nothing."""
    if n1 == n2:
        return ""
    return f"; inflated by size alone, {n1} vs {n2} nodes"


def render_topology(result: TopologicalDistances) -> list[str]:
    """``## Topological distances`` as markdown lines (ch. 48): one row per method, what it
    compares, and the number, with the caveats every reading needs printed beside it rather than
    left for a caller to remember. The spectral-distance and portrait-divergence rows carry their
    own zero-padding disclosure inline, next to the number, whenever the two graphs being
    compared are not the same order -- NetSimile and DeltaCon need no such disclosure, because
    neither one's number grows just because a graph gained nodes (see
    :func:`netsimile_distance` and :func:`delta_con`)."""
    spectral_n1, spectral_n2 = result.spectral.provenance["n_nodes"]
    portrait_n1, portrait_n2 = result.portrait.provenance["n_nodes"]
    lines = [
        "## Topological distances",
        "",
        "How similar the two builds' whole topologies are, by four different definitions of "
        "'similar' (§48.1) -- none of them the one true answer (p. 699).",
        "",
        "| method | value | compares |",
        "|---|---|---|",
        f"| spectral distance | {_num(result.spectral.distance)} | the "
        f"{result.spectral.dims} smallest non-trivial Laplacian eigenvalues on each side, "
        f"zero-padded to match{_size_caveat(spectral_n1, spectral_n2)} |",
        f"| NetSimile (Canberra) | {_num(result.netsimile.distance)} | 35-entry signature "
        "vectors of per-node degree, clustering and egonet features |",
        f"| DeltaCon similarity | {_num(result.delta_con.similarity)} | node-affinity matrices "
        "under fast belief propagation, aligned by id |",
        f"| portrait divergence | {_num(result.portrait.divergence)} | the distribution of "
        f"shortest-path-length shells, in bits (0 to 1), both graphs' portraits padded to the "
        f"larger's shells and node count{_size_caveat(portrait_n1, portrait_n2)} |",
        "",
        "Higher is more different for every row except DeltaCon similarity, where higher is more "
        "alike.",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    if result.notes:
        lines.append("")
    return lines


# ----------------------------------------------------------------------------- §48.3 fusion


def fuse_networks(
    graphs: Sequence[nx.Graph], *, weight: str = "weight", min_weight: float | None = None
) -> nx.Graph:
    """Combine several observations of the same network into one fused summary (§48.3, Figure
    48.10, p. 709): align every graph by node id, average each edge's weight across the
    observations that carry it -- an observation that lacks an edge contributes 0, not "no
    opinion" -- and keep the edge if the resulting average clears ``min_weight`` (any positive
    average, by default).

    This is the chapter's own worked example, not the "much smarter and more sophisticated" (p.
    709) algorithm its citation (Wang et al. 2014, Similarity Network Fusion) actually runs: SNF
    iteratively cross-diffuses each observation's own k-nearest-neighbour affinity kernel through
    every other observation's kernel until they converge to a consensus, which needs a
    neighbourhood-size parameter and an iteration count the chapter never states and this ticket
    is not in a position to invent defaults for. The averaging-and-threshold version is exactly
    what Figure 48.10 works through by hand, and is what this function builds.

    Every input graph is flattened first (§6.2): a fused network answers "how strong is this tie
    across every observation", which is not a question with a direction until the observations
    agree on one, and this ticket does not have a rule for combining directions that differ.

    Undefined, and raised on, for an empty sequence of graphs.
    """
    if not graphs:
        msg = "fuse_networks needs at least one graph to fuse"
        raise ValueError(msg)
    flattened = [undirected_view(g)[0] for g in graphs]
    fused = nx.Graph()
    fused.add_nodes_from({node for g in flattened for node in g.nodes})
    totals: dict[tuple[Node, Node], float] = {}
    for g in flattened:
        for u, v, data in g.edges(data=True):
            key = (u, v) if str(u) <= str(v) else (v, u)
            totals[key] = totals.get(key, 0.0) + float(data.get(weight, 1.0))
    count = len(flattened)
    for (u, v), total in totals.items():
        average = total / count
        keep = average > 0.0 if min_weight is None else average >= min_weight
        if keep:
            fused.add_edge(u, v, weight=average)
    fused.graph["fused_from"] = len(flattened)
    fused.graph["min_weight"] = min_weight
    return fused
