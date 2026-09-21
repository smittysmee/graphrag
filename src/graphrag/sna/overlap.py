"""Overlapping coverage: Atlas ch. 38, where the classical community definition breaks.

*"Communities are groups of nodes densely connected to each other and sparsely connected to
nodes outside the community"* is the definition every earlier module in this package assumes.
Chapter 38 asks who the center person of a school-and-work social circle belongs to -- *"the more
reasonable answer is 'she belongs to both'"* (p. 543) -- and none of the disjoint methods can say
that: modularity's Kronecker delta double-counts a node placed in two communities and the standard
definition breaks down.

Three ways of finding an overlapping *cover* (a node may sit in more than one community) are
built here, and the module keeps every one of them a cover rather than a partition:

*§38.3, k-clique percolation* (:func:`k_clique_communities`). Communities are unions of
overlapping ``k``-cliques: two ``k``-cliques belong to the same community when they share
``k - 1`` nodes. This package hands the construction to ``networkx``
(``nx.community.k_clique_communities``), which is the same algorithm (Derenyi, Palla and Vicsek,
2005) the chapter names, rather than re-implementing clique percolation.

*§38.5, Hierarchical Link Clustering* (:func:`link_clustering`). Cluster the *edges* instead of
the nodes: two edges sharing a node are similar by the Jaccard coefficient of the neighbourhoods
of their two non-shared endpoints (p. 554), single-linkage clustering turns that into a
dendrogram, and the cut that maximises the chapter's own **partition density** (p. 555) is kept.
A node belongs to every community any of its edges ended up in.

*§38.5, "Ego Networks"* (:func:`ego_splitting`, called ``ego-splitting`` by this ticket). For
every node, its alters-only ego network (:mod:`graphrag.sna.ego`'s construction, minus the ego)
is partitioned by a disjoint community-discovery algorithm -- Louvain here, where the original
paper (Coscia et al., *DEMON*, 2012) uses label propagation, both being disjoint methods the
chapter treats as interchangeable (p. 555: *"this needs not to be the case"*) -- the ego is
unioned back into each local community it finds, and candidate communities across the whole
network are merged when their node-Jaccard passes a threshold (§38.9 exercise 3's own number,
0.1, ignoring singletons).

*§38.1, overlapping NMI* (:func:`overlapping_nmi`). The chapter's own recipe: represent each
cover as a ``|V| x C`` binary affiliation matrix (Figure 38.2) and compare two covers by pairing
each column on one side with its most similar column on the other, rather than comparing two
label vectors directly, which an overlapping cover cannot be flattened into without losing
information. *"These indexes share with their original counterpart the issue of non-zero values
for independent vectors"* (p. 545) -- this one more so, because matching each community to its
*best* partner on the other side is itself a source of chance agreement the book's own citation
names (Gates and Ahn, 2017): read "near zero" generously and never as chance-corrected.

*§38.7, the overlap paradox* (:func:`overlap_paradox`). If the nodes shared between two
communities avoid each other, the community is not internally dense; if they connect to each
other more than average, because they carry two communities' worth of ties, the shared region is
*denser* than either community and the "boundary" is in fact the core (Figure 38.15). Both
readings are checked, per pair of overlapping communities, against the density of each community
on its own.

**What every cover is put through anyway.** Chapter 36's battery (:mod:`graphrag.sna.evaluate`)
is defined on disjoint partitions only, and this package will not silently pretend a cover is
one. :func:`flatten_cover` assigns each node to the single largest community it belongs to --
the best-effort partition a reader who only has ch. 36's tools could build -- and the report says
exactly what that flattening threw away before printing the battery over what is left.

**What the book defines and what this adds.** The chapter reviews Mixed Membership Stochastic
Blockmodels, the community affiliation graph and BigClam, overlapping modularity, overlapping
Infomap and overlapping label propagation, and OSLOM (§38.2, §38.4, §38.6) without giving worked
formulas for any of them; none is built here, because each needs either a generative-model fit
this package has no vocabulary for yet or a bespoke objective the chapter itself says has "many
conflicting ways" to define (p. 545). Fuzzy clustering (Figure 38.3) -- belonging *coefficients*
rather than flat membership -- is named by the chapter as a harder, more precise alternative to
every method here; nothing in this module reports one, and every community below is read as "this
node fully belongs here", never "60% here and 40% there".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from graphrag.sna.cluster import louvain
from graphrag.sna.evaluate import (
    PartitionScores,
    evaluate_partition,
    evaluation_payload,
    render_evaluation,
)
from graphrag.sna.matrices import Node
from graphrag.sna.measures import undirected_view
from graphrag.sna.stats import normalized_mutual_information

__all__ = [
    "CHAPTER",
    "DEFAULT_JACCARD_THRESHOLD",
    "DEFAULT_K",
    "MAX_EGO_SPLIT_CANDIDATES",
    "MAX_LINK_CLUSTERING_EDGES",
    "METHODS",
    "CliqueCover",
    "EgoSplitCover",
    "LinkClustering",
    "OverlapParadox",
    "OverlapParadoxRow",
    "OverlapReport",
    "build_overlap_report",
    "ego_splitting",
    "flatten_cover",
    "k_clique_communities",
    "link_clustering",
    "overlap_paradox",
    "overlap_payload",
    "overlapping_nmi",
    "render_overlap",
]

#: The section header every part of this module answers to.
CHAPTER = "§38.1-38.7"

#: Derenyi, Palla and Vicsek's own worked cases use k=4 and k=5; 4 is the smaller and the one
#: that still needs a real 3-node overlap (k-1) to percolate, which is a demonstrable property on
#: a network of this package's size, unlike k=5 which needs a shared 4-clique.
DEFAULT_K = 4

#: §38.9 exercise 3's own number: "merge communities with a node Jaccard coefficient higher than
#: 0.1 (ignoring singletons: communities of a single node)".
DEFAULT_JACCARD_THRESHOLD = 0.1

#: Above this many edges, the dense E x E similarity matrix :func:`link_clustering` builds would
#: be a matrix of a hundred thousand-plus floats per side; §38.5 gives no bound of its own, so
#: this is this module's, sized for a corpus network already filtered by `--min-weight`/`--where`.
MAX_LINK_CLUSTERING_EDGES = 600

#: Above this many candidate persona-communities (one node can produce several, one per local
#: community in its ego network), the pairwise Jaccard merge below is a search over this many
#: squared comparisons; a bound for the same reason as the one above.
MAX_EGO_SPLIT_CANDIDATES = 4000

#: The three cover-discovery methods :func:`build_overlap_report` and `sna overlap` accept.
METHODS: tuple[str, ...] = ("clique", "link", "ego")


def _nodes(community: Iterable[Node]) -> list[str]:
    return sorted({str(node) for node in community})


def _density(graph: nx.Graph, nodes: Iterable[str]) -> float:
    """The induced density of ``nodes`` in ``graph`` (§12.1). ``nan`` below two nodes."""
    members = list(nodes)
    if len(members) < 2:
        return math.nan
    possible = len(members) * (len(members) - 1) / 2
    return float(graph.subgraph(members).number_of_edges()) / possible


def _edge(u: object, v: object) -> tuple[str, str]:
    """The two endpoints as strings, in a stable order, so an edge has one canonical key."""
    left, right = str(u), str(v)
    return (left, right) if left <= right else (right, left)


# --------------------------------------------------------------------------------- §38.3


@dataclass(frozen=True)
class CliqueCover:
    """k-clique percolation (§38.3): communities are unions of k-cliques sharing k-1 nodes."""

    k: int
    communities: list[list[str]]
    too_low_degree: list[str]
    """Nodes whose degree is below k-1, so they can never sit in a k-clique at all (p. 550)."""
    uncovered: list[str]
    """Nodes with degree k-1 or more that still fell in no community: not disconnected from the
    idea of a k-clique, just not part of one that percolated."""


def k_clique_communities(graph: nx.Graph, k: int = DEFAULT_K) -> CliqueCover:
    """Communities that are unions of overlapping k-cliques (§38.3, Derenyi, Palla and Vicsek 2005).

    *"The algorithm finds all cliques of size k... then it attempts to merge two communities in
    the same community if the two communities share at least a k-1 clique"* (p. 548-549). This
    hands the construction straight to ``networkx``'s own implementation of that algorithm rather
    than rebuilding clique enumeration.

    ``k`` must be at least 2 (a 2-clique is an edge, so k=2 percolates through shared nodes and
    gives the connected components -- a legitimate, if degenerate, use of the definition).

    The chapter's own limitation travels with the result rather than being left for a reader to
    discover: *"if you set your k relatively low, e.g. k=4, all nodes with degree equal to one
    cannot be part of any community"* (p. 550) -- more generally, a node of degree under k-1
    cannot sit in any k-clique, and :attr:`CliqueCover.too_low_degree` names them so a reader does
    not read their absence as a finding about the node.
    """
    if k < 2:
        msg = f"k must be at least 2 (a 2-clique is an edge), got {k}"
        raise ValueError(msg)
    view, _ = undirected_view(graph)
    view = nx.Graph(view)
    view.remove_edges_from(list(nx.selfloop_edges(view)))
    raw = nx.community.k_clique_communities(view, k)
    communities = sorted((_nodes(c) for c in raw), key=lambda c: (-len(c), c[0] if c else ""))
    covered = {node for community in communities for node in community}
    too_low_degree = sorted(
        str(node) for node, degree in view.degree() if degree < k - 1 and str(node) not in covered
    )
    uncovered = sorted(
        str(node)
        for node in view.nodes
        if str(node) not in covered and str(node) not in too_low_degree
    )
    return CliqueCover(
        k=k, communities=communities, too_low_degree=too_low_degree, uncovered=uncovered
    )


# --------------------------------------------------------------------------------- §38.5, links


def _partition_density(edges: Sequence[tuple[str, str]], labels: Sequence[int]) -> float:
    """§38.5's partition density: D = (1/|E|) sum_c |Ec| Dc, Dc = 0 when |Vc| <= 2 (p. 555)."""
    by_label: dict[int, list[tuple[str, str]]] = {}
    for edge, label in zip(edges, labels, strict=True):
        by_label.setdefault(int(label), []).append(edge)
    total = 0.0
    for community_edges in by_label.values():
        members = {node for edge in community_edges for node in edge}
        edge_count, node_count = len(community_edges), len(members)
        if node_count <= 2:
            density = 0.0
        else:
            denominator = node_count * (node_count - 1) / 2 - (node_count - 1)
            density = (edge_count - (node_count - 1)) / denominator if denominator > 0 else 0.0
        total += edge_count * density
    return total / len(edges) if edges else 0.0


@dataclass(frozen=True)
class LinkClustering:
    """Hierarchical Link Clustering (§38.5, Ahn, Bagrow and Lehmann 2010)."""

    communities: list[list[str]]
    edge_communities: list[list[tuple[str, str]]]
    partition_density: float
    """The chapter's own quality function (p. 555), maximised over every cut of the dendrogram."""
    threshold: float
    """The dendrogram height the best cut was made at, in 1 - similarity terms."""
    edges: int


def _edge_similarity(
    neighbours: dict[str, set[str]], shared: str, e1: tuple[str, str], e2: tuple[str, str]
) -> float:
    """S(u,k),(v,k): the Jaccard of the two non-shared endpoints' neighbourhoods (p. 554)."""
    u = e1[0] if e1[1] == shared else e1[1]
    v = e2[0] if e2[1] == shared else e2[1]
    if u == v:
        return 1.0
    left, right = neighbours[u], neighbours[v]
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def link_clustering(graph: nx.Graph) -> LinkClustering:
    """Cluster the edges instead of the nodes, then read the node cover off the result (§38.5).

    *"If two edges share no node, their similarity is zero. If the edges share a node, their
    similarity is the Jaccard coefficient of the neighborhoods of the two non-shared nodes"*
    (p. 554). Single-linkage hierarchical clustering over ``1 - similarity`` turns that into a
    dendrogram (:func:`scipy.cluster.hierarchy.linkage`), and every distinct merge height is tried
    as a cut, keeping the one that maximises :func:`_partition_density` -- the chapter's own
    criterion, because *"we cannot use modularity, because these are link communities, not node
    communities"* (p. 555).

    A node belongs to every node-community that any of its edges ended up in, which is how a
    single bridging edge -- one whose two endpoints sit in otherwise separate neighbourhoods --
    can end up its own small community and put both of its endpoints in it besides whichever
    community the rest of their edges join (Figure 38.12: *"node 4 belongs to three
    communities"*).

    Refuses above :data:`MAX_LINK_CLUSTERING_EDGES` edges, where the full edge-by-edge similarity
    matrix this method needs stops being a reasonable thing to hold in memory; filter the network
    first (`--min-weight`, `--where`, a time window) rather than raising the bound.
    """
    view, _ = undirected_view(graph)
    view = nx.Graph(view)
    view.remove_edges_from(list(nx.selfloop_edges(view)))
    edges: list[tuple[str, str]] = sorted({_edge(u, v) for u, v in view.edges()})
    count = len(edges)
    if count < 2:
        msg = f"link_clustering() needs at least two edges to compare, got {count}"
        raise ValueError(msg)
    if count > MAX_LINK_CLUSTERING_EDGES:
        msg = (
            f"this network has {count:,} edges, over the {MAX_LINK_CLUSTERING_EDGES:,} "
            "link_clustering() will hold as a dense similarity matrix (§38.5): filter the "
            "network first (--min-weight, --where, a time window)"
        )
        raise ValueError(msg)

    neighbours = {str(node): {str(n) for n in view.neighbors(node)} for node in view.nodes}
    incident: dict[str, list[int]] = {}
    for index, (u, v) in enumerate(edges):
        incident.setdefault(u, []).append(index)
        incident.setdefault(v, []).append(index)

    distance = np.ones((count, count), dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    for shared, indices in incident.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                similarity = _edge_similarity(neighbours, shared, edges[i], edges[j])
                cost = 1.0 - similarity
                if cost < distance[i, j]:
                    distance[i, j] = distance[j, i] = cost

    linkage_matrix = linkage(squareform(distance, checks=False), method="single")
    heights = sorted({0.0, *(float(h) for h in linkage_matrix[:, 2])})
    best_density, best_height, best_labels = -1.0, 0.0, np.arange(1, count + 1)
    for height in heights:
        labels = (
            np.arange(1, count + 1)
            if height <= 0.0
            else fcluster(linkage_matrix, t=height, criterion="distance")
        )
        density = _partition_density(edges, labels.tolist())
        if density > best_density:
            best_density, best_height, best_labels = density, height, labels

    groups: dict[int, list[tuple[str, str]]] = {}
    for edge, label in zip(edges, best_labels.tolist(), strict=True):
        groups.setdefault(int(label), []).append(edge)
    edge_communities = [sorted(group) for group in groups.values()]
    communities = sorted(
        (sorted({node for edge in group for node in edge}) for group in edge_communities),
        key=lambda c: (-len(c), c[0] if c else ""),
    )
    return LinkClustering(
        communities=communities,
        edge_communities=edge_communities,
        partition_density=best_density,
        threshold=best_height,
        edges=count,
    )


# --------------------------------------------------------------------------------- §38.5, ego


@dataclass(frozen=True)
class EgoSplitCover:
    """The "Ego Networks" method (§38.5, Coscia et al.'s DEMON): this ticket's "ego-splitting"."""

    communities: list[list[str]]
    candidates: int
    """How many persona-communities were proposed before merging: one per node per local
    community found in that node's ego network."""
    jaccard_threshold: float


def _merge_by_jaccard(
    candidates: Sequence[frozenset[str]], threshold: float
) -> list[frozenset[str]]:
    """Union-find over the candidate sets: merge whenever node-Jaccard exceeds ``threshold``.

    Built as connected components of a "similar enough" graph over the candidates rather than a
    greedy sequential pass, so the answer does not depend on the order the candidates arrived in.
    """
    unique = sorted({c for c in candidates if len(c) > 1}, key=sorted)
    parent = list(range(len(unique)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            union = unique[i] | unique[j]
            jaccard = len(unique[i] & unique[j]) / len(union) if union else 0.0
            if jaccard > threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

    merged: dict[int, set[str]] = {}
    for index, candidate in enumerate(unique):
        merged.setdefault(find(index), set()).update(candidate)
    return [frozenset(group) for group in merged.values()]


def ego_splitting(
    graph: nx.Graph,
    *,
    resolution: float = 1.0,
    seed: int | None = None,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> EgoSplitCover:
    """For every node, split it across the local communities of its own ego network (§38.5).

    *"First, we extract the ego network of a node, removing the ego itself... Then, we apply a
    disjoint community discovery algorithm to the ego network... we have a set of communities,
    and we can merge them according to some criterion"* (p. 555-556). This reuses
    :mod:`graphrag.sna.ego`'s construction -- the alters-only ego network -- and Louvain
    (:func:`graphrag.sna.cluster.louvain`) as the disjoint algorithm, where the original paper
    uses label propagation; the chapter treats the choice as open (*"this needs not to be the
    case"*, p. 556). For every local community the ego network yields, a candidate is formed from
    the ego plus that community, and candidates across the whole network are merged whenever
    their node-Jaccard exceeds ``jaccard_threshold`` -- §38.9 exercise 3's own number, 0.1,
    "ignoring singletons: communities of a single node".

    A node whose ego network is already disconnected without it -- two neighbour groups joined
    only through this node -- yields one candidate per component and so is placed in (at least)
    that many communities without any merging needed at all; a node whose ego network has
    internal structure that Louvain further subdivides is split further still.

    Refuses above :data:`MAX_EGO_SPLIT_CANDIDATES` candidates, where the merge step's pairwise
    comparison stops being cheap; narrow the network first.
    """
    if not 0.0 <= jaccard_threshold < 1.0:
        msg = f"jaccard_threshold must be in [0, 1), got {jaccard_threshold}"
        raise ValueError(msg)
    view, _ = undirected_view(graph)
    view = nx.Graph(view)
    view.remove_edges_from(list(nx.selfloop_edges(view)))

    candidates: list[frozenset[str]] = []
    for node in sorted(str(n) for n in view.nodes):
        alters = sorted(str(n) for n in view.neighbors(node))
        if len(alters) < 2:
            candidates.append(frozenset({node, *alters}))
            continue
        sub = view.subgraph(alters)
        if sub.number_of_edges() == 0:
            for alter in alters:
                candidates.append(frozenset({node, alter}))
            continue
        local = louvain(sub, resolution=resolution, seed=seed, runs=1)
        if not local.communities:
            candidates.append(frozenset({node, *alters}))
            continue
        for community in local.communities:
            candidates.append(frozenset({node, *community}))
        if len(candidates) > MAX_EGO_SPLIT_CANDIDATES:
            msg = (
                f"more than {MAX_EGO_SPLIT_CANDIDATES:,} candidate persona-communities were "
                "proposed before every node was even visited: narrow the network first "
                "(--min-weight, --where, a time window)"
            )
            raise ValueError(msg)

    merged = _merge_by_jaccard(candidates, jaccard_threshold)
    communities = sorted(
        (sorted(group) for group in merged), key=lambda c: (-len(c), c[0] if c else "")
    )
    return EgoSplitCover(
        communities=communities, candidates=len(candidates), jaccard_threshold=jaccard_threshold
    )


# --------------------------------------------------------------------------------- §38.1


def overlapping_nmi(cover_a: Sequence[Iterable[Node]], cover_b: Sequence[Iterable[Node]]) -> float:
    """Normalised mutual information extended to covers, by matching communities (§38.1).

    *"We don't compare the vectors directly. We compare two bipartite matrices... calculate the
    mutual information between the two matrices by pairing the columns such that we assign to
    each column on one side the ones on the other side that is the most similar to it"*
    (p. 544). Each community becomes a binary column over the union of both covers' nodes (1 if
    the node is a member, Figure 38.2), every column on each side is matched to its
    highest-:func:`graphrag.sna.stats.normalized_mutual_information` partner on the other, and
    the two directions (A matched against B, B matched against A) are averaged, because matching
    is not symmetric: A having a good partner for every column does not mean B does.

    1.0 when the two covers are the same partition of communities (a column always finds itself,
    perfectly, on both sides); the chapter's own caution about ordinary NMI applies more strongly
    here rather than less -- *"these indexes share with their original counterpart the issue of
    non-zero values for independent vectors"* (p. 545) -- and picking each column's *best* match
    among many is itself a source of chance agreement that grows with the number of communities on
    either side (Gates and Ahn, 2017, cited at p. 544): a "near zero" reading is not chance
    corrected and should be read against how many communities each side has, the way plain NMI is
    read against how many clusters it has (§36.4).

    **Where this departs from the book.** The chapter names six published variants without
    settling on one; this is the plain matching-and-average reading of the paragraph quoted
    above, not the Lancichinetti/McDaid papers' conditional-entropy formula, and not the
    Omega Index's chance correction (p. 545), which needs a null model this function does not
    build. Raises ``ValueError`` for an empty cover on either side.
    """
    left = [set(_nodes(community)) for community in cover_a if list(community)]
    right = [set(_nodes(community)) for community in cover_b if list(community)]
    if not left or not right:
        msg = "overlapping_nmi() needs two non-empty covers"
        raise ValueError(msg)
    universe = sorted({node for community in (*left, *right) for node in community})
    if len(universe) < 2:
        msg = "overlapping_nmi() needs at least two nodes across both covers"
        raise ValueError(msg)

    def column(community: set[str]) -> list[int]:
        return [1 if node in community else 0 for node in universe]

    def best_match(source: Sequence[set[str]], target: Sequence[set[str]]) -> float:
        target_columns = [column(community) for community in target]
        return sum(
            max(normalized_mutual_information(column(community), other) for other in target_columns)
            for community in source
        ) / len(source)

    return (best_match(left, right) + best_match(right, left)) / 2.0


# --------------------------------------------------------------------------------- §38.7


@dataclass(frozen=True)
class OverlapParadoxRow:
    """One pair of overlapping communities, read against §38.7's paradox."""

    a: int
    b: int
    overlap_size: int
    overlap_density: float
    density_a: float
    density_b: float
    verdict: str


@dataclass(frozen=True)
class OverlapParadox:
    """Every pair of overlapping communities, checked both ways the definition can break (§38.7)."""

    rows: list[OverlapParadoxRow]
    denser_than_both: int
    """Pairs where the shared nodes are more densely tied to each other than either community is
    internally -- the boundary is denser than the core, Figure 38.15."""
    sparser_than_both: int
    """Pairs where the shared nodes barely connect to each other at all -- "holes" in a
    stochastic blockmodel's reading of the communities, Figure 38.14."""


def overlap_paradox(graph: nx.Graph, communities: Sequence[Sequence[Node]]) -> OverlapParadox:
    """Is the overlap denser than the communities it sits between, or sparser (§38.7)?

    *"If the overlap nodes don't connect to each other, they have low connection probability,
    which contradicts the fact that they are part of the same community"* (holes, p. 558); *"since
    these nodes share not one but two communities... they are more likely to connect to each
    other than nodes sharing only one community... the overlap... is denser than the community
    itself"* (p. 558). Both readings are computed for every pair of communities that share at
    least two nodes -- one shared node has no possible edge between it and itself, so its density
    is undefined rather than zero.

    Neither reading is a defect to fix: the chapter's point is that the classical "dense inside,
    sparse outside" definition cannot survive overlap intact, whichever way it breaks.
    """
    view, _ = undirected_view(graph)
    sets = [set(_nodes(community)) for community in communities]
    rows: list[OverlapParadoxRow] = []
    denser = sparser = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if len(overlap) < 2:
                continue
            overlap_density = _density(view, overlap)
            density_a, density_b = _density(view, sets[i]), _density(view, sets[j])
            if math.isnan(density_a) or math.isnan(density_b):
                verdict = "undefined: one of the two communities has fewer than two members"
            elif overlap_density > max(density_a, density_b):
                verdict = "the overlap is denser than either community: the boundary is the core"
                denser += 1
            elif overlap_density < min(density_a, density_b):
                verdict = "the overlap is sparser than either community: a hole, not a boundary"
                sparser += 1
            else:
                verdict = (
                    "between the two communities' densities: neither paradox reading fires here"
                )
            rows.append(
                OverlapParadoxRow(
                    a=i,
                    b=j,
                    overlap_size=len(overlap),
                    overlap_density=overlap_density,
                    density_a=density_a,
                    density_b=density_b,
                    verdict=verdict,
                )
            )
    return OverlapParadox(rows=rows, denser_than_both=denser, sparser_than_both=sparser)


# --------------------------------------------------------------------------------- flattening


def flatten_cover(
    graph: nx.Graph, communities: Sequence[Sequence[Node]]
) -> tuple[list[list[str]], list[str]]:
    """The best-effort disjoint partition a cover can be read as, for ch. 36's battery.

    Every node goes to the single **largest** community it belongs to, ties broken by that
    community's position in ``communities``; every other membership it held is dropped. This is
    a reading tool, not a finding -- the cover above is the answer chapter 38 gives, and this is
    only what the disjoint tools of chapter 36 are able to look at.
    """
    sets = [_nodes(community) for community in communities]
    membership_counts: dict[str, int] = {}
    choice: dict[str, tuple[int, int]] = {}
    for index, community in enumerate(sets):
        size = len(community)
        for node in community:
            membership_counts[node] = membership_counts.get(node, 0) + 1
            key = (-size, index)
            if node not in choice or key < choice[node]:
                choice[node] = key
    grouped: dict[int, list[str]] = {}
    for node, (_, index) in choice.items():
        grouped.setdefault(index, []).append(node)
    partition = [sorted(grouped[index]) for index in sorted(grouped)]

    overlap_nodes = sorted(node for node, count in membership_counts.items() if count > 1)
    notes: list[str] = []
    if overlap_nodes:
        dropped = sum(membership_counts[node] - 1 for node in overlap_nodes)
        notes.append(
            f"{len(overlap_nodes)} of {len(membership_counts)} node(s) were in more than one "
            f"community; each was kept in its largest one and {dropped} other membership(s) "
            "were dropped so that chapter 36's disjoint battery has something to score. The "
            "cover above is the finding this module makes; the partition below is only what "
            "that battery can read."
        )
    else:
        notes.append("No node was in more than one community, so flattening changed nothing.")
    vanished = sum(1 for index in range(len(sets)) if index not in grouped)
    if vanished:
        notes.append(
            f"{vanished} of {len(sets)} communities lost every member to a larger overlapping "
            "one and do not appear in the flattened partition at all."
        )
    return partition, notes


# --------------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class OverlapReport:
    """One cover, its flattened evaluation, the overlap-paradox check and a disjoint baseline."""

    method: str
    section: str
    frame: str
    nodes: int
    edges: int
    parameters: dict[str, Any]
    communities: list[list[str]]
    method_notes: list[str]
    flattened: list[list[str]]
    flattening_notes: list[str]
    evaluation: PartitionScores
    paradox: OverlapParadox
    baseline_communities: int
    overlap_nmi_vs_baseline: float
    notes: list[str] = field(default_factory=list)

    @property
    def sizes(self) -> list[int]:
        return [len(community) for community in self.communities]

    @property
    def overlap_nodes(self) -> list[str]:
        counts: dict[str, int] = {}
        for community in self.communities:
            for node in community:
                counts[node] = counts.get(node, 0) + 1
        return sorted(node for node, count in counts.items() if count > 1)


def build_overlap_report(
    graph: nx.Graph,
    method: str = "clique",
    *,
    k: int = DEFAULT_K,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    resolution: float = 1.0,
    seed: int | None = None,
    link_prediction: bool = True,
) -> OverlapReport:
    """Run one of the three cover-discovery methods and put it through every check this module has.

    ``method`` is ``"clique"`` (§38.3), ``"link"`` (§38.5's Hierarchical Link Clustering) or
    ``"ego"`` (§38.5's Ego Networks / DEMON, the ticket's "ego-splitting"). Every path ends the
    same way: the cover is flattened (:func:`flatten_cover`) and scored by ch. 36's battery
    (:func:`graphrag.sna.evaluate.evaluate_partition`), checked for the overlap paradox
    (:func:`overlap_paradox`), and compared by :func:`overlapping_nmi` against this network's own
    Louvain partition read as a (non-overlapping) cover -- an always-available baseline that
    answers "how much did allowing overlap change the picture", without needing an outside truth
    this package has no way to obtain for a corpus network.
    """
    if method not in METHODS:
        msg = f"method must be one of {METHODS}, got {method!r}"
        raise ValueError(msg)
    view, flattened_note = undirected_view(graph)

    parameters: dict[str, Any]
    if method == "clique":
        clique = k_clique_communities(view, k)
        communities = clique.communities
        parameters = {"k": k}
        section = "§38.3"
        method_notes = [
            f"{len(clique.too_low_degree)} node(s) have degree under k-1={k - 1} and can never "
            f"join a {k}-clique at all (p. 550).",
            f"{len(clique.uncovered)} node(s) have enough degree but fell in no {k}-clique that "
            "percolated.",
        ]
    elif method == "link":
        links = link_clustering(view)
        communities = links.communities
        parameters = {"partition_density": links.partition_density, "threshold": links.threshold}
        section = "§38.5"
        method_notes = [
            f"Best partition density {links.partition_density:.4f} at dendrogram height "
            f"{links.threshold:.4f}, over {len(links.edge_communities)} edge communit"
            f"{'y' if len(links.edge_communities) == 1 else 'ies'} across {links.edges:,} edges.",
        ]
    else:
        ego = ego_splitting(
            view, resolution=resolution, seed=seed, jaccard_threshold=jaccard_threshold
        )
        communities = ego.communities
        parameters = {"jaccard_threshold": jaccard_threshold, "resolution": resolution}
        section = "§38.5"
        method_notes = [
            f"{ego.candidates} candidate persona-communit"
            f"{'y' if ego.candidates == 1 else 'ies'} proposed from every node's own ego "
            f"network, merged into {len(communities)} community/communities by node-Jaccard "
            f"> {jaccard_threshold} (§38.9 exercise 3's threshold).",
        ]

    partition, flattening_notes = flatten_cover(view, communities)
    evaluation = evaluate_partition(
        view, partition, link_prediction=link_prediction and view.number_of_edges() > 0, seed=seed
    )
    paradox = overlap_paradox(view, communities)

    notes: list[str] = []
    if flattened_note:
        notes.append(f"Every number here was computed {flattened_note} (§6.2).")
    if not communities:
        notes.append("This method found no community at all: nothing below is a measurement.")

    baseline: list[list[str]] = []
    overlap_nmi = math.nan
    if view.number_of_edges() > 0:
        baseline_result = louvain(view, seed=seed)
        baseline = [sorted(str(n) for n in community) for community in baseline_result.communities]
        if communities and baseline:
            overlap_nmi = overlapping_nmi(communities, baseline)
    else:
        notes.append(
            "No baseline: Louvain has nothing to partition on a network with no edges, so "
            "overlapping NMI against it is not computed."
        )

    return OverlapReport(
        method=method,
        section=section,
        frame=str(view.graph.get("frame", "")),
        nodes=view.number_of_nodes(),
        edges=view.number_of_edges(),
        parameters=parameters,
        communities=communities,
        method_notes=method_notes,
        flattened=partition,
        flattening_notes=flattening_notes,
        evaluation=evaluation,
        paradox=paradox,
        baseline_communities=len(baseline),
        overlap_nmi_vs_baseline=overlap_nmi,
        notes=notes,
    )


# --------------------------------------------------------------------------------- rendering


_METHOD_NAMES: dict[str, str] = {
    "clique": "k-clique percolation",
    "link": "Hierarchical Link Clustering",
    "ego": "Ego Networks (DEMON-style ego-splitting)",
}


def _num(value: float, places: int = 4) -> str:
    if math.isnan(value):
        return "-"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_overlap(report: OverlapReport) -> list[str]:
    """The `## Overlapping coverage` section: cover, flattening, paradox check and battery."""
    method_name = _METHOD_NAMES.get(report.method, report.method)
    lines: list[str] = [
        f"## Overlapping coverage: {method_name}",
        "",
        f"**Implements.** Atlas {CHAPTER}, this method at {report.section}. Chapter 38's own "
        'argument: the classical "dense inside, sparse outside" community cannot always be a '
        "*partition* -- a node may belong to more than one -- so this section returns a *cover* "
        "and never pretends it is disjoint.",
        "",
        f"**Sampling frame.** {report.frame or 'not recorded'}",
        "",
        f"**n.** {report.nodes:,} nodes and {report.edges:,} edges, covered by "
        f"{len(report.communities)} communit{'y' if len(report.communities) == 1 else 'ies'} "
        f"(sizes {', '.join(str(size) for size in report.sizes[:12])}"
        f"{', …' if len(report.sizes) > 12 else ''}); {len(report.overlap_nodes)} node(s) "
        "belong to more than one.",
        "",
        "**Null model.** None for the cover itself: k-clique percolation, link clustering and "
        "ego-splitting are structural constructions, not fits to a generative model, and the "
        "chapter proposes none for any of them. The flattened partition below carries "
        "chapter 36's own null models, and the overlapping-NMI comparison is read against 0 and "
        "1 the way ordinary NMI is (§36.4), not against a sampled null.",
        "",
    ]
    lines += [f"- {note}" for note in report.method_notes]
    lines += ["", "The communities, largest first:", ""]
    lines += _table(
        ["#", "size", "members (up to 12)"],
        [
            [
                str(index + 1),
                f"{len(community):,}",
                ", ".join(community[:12]) + (", …" if len(community) > 12 else ""),
            ]
            for index, community in enumerate(report.communities)
        ],
    )
    lines += [
        "### Flattened to a partition, for chapter 36 (§38.1's own reason this is necessary)",
        "",
        *[f"- {note}" for note in report.flattening_notes],
        "",
    ]
    lines += render_evaluation(report.evaluation)
    lines += _render_paradox(report.paradox)
    lines += [
        "### Against this network's own disjoint answer (§38.1)",
        "",
        f"Overlapping NMI against Louvain's {report.baseline_communities}-community disjoint "
        f"partition of the same network: {_num(report.overlap_nmi_vs_baseline, 3)}. Read this as "
        "how much allowing overlap changed the picture, not as agreement with a truth: Louvain's "
        "answer is one more community-discovery method's opinion, not a ground truth this corpus "
        "carries. 1.0 would mean the cover has no overlap and reproduces Louvain's groups "
        "exactly; a low value means the two disagree about which nodes go where, an overlap "
        "existing at all being one reason they would.",
        "",
    ]
    if report.notes:
        lines += [f"> {note}" for note in report.notes] + [""]
    return lines


def _render_paradox(paradox: OverlapParadox) -> list[str]:
    lines = [
        "### The overlap paradox (§38.7)",
        "",
        "**Implements.** §38.7: whether the nodes shared between two communities are internally "
        'sparse (a "hole" a stochastic blockmodel would not expect, Figure 38.14) or internally '
        "dense enough that the shared region outranks either community on its own (Figure "
        '38.15) -- either way, "dense inside, sparse outside" cannot hold for both the '
        "communities and their overlap at once.",
        "",
        f"**n.** {len(paradox.rows)} pair(s) of communities share at least two nodes and have a "
        "computable overlap density.",
        "",
        "**Null model.** None: this is a comparison of three observed densities on the same "
        "network, not a test against a random graph.",
        "",
    ]
    if not paradox.rows:
        lines += [
            "No two communities share as many as two nodes, so neither reading of the paradox "
            "has anything to check.",
            "",
        ]
        return lines
    lines += _table(
        [
            "community A",
            "community B",
            "overlap size",
            "overlap density",
            "density A",
            "density B",
            "reading",
        ],
        [
            [
                f"#{row.a + 1}",
                f"#{row.b + 1}",
                f"{row.overlap_size:,}",
                _num(row.overlap_density, 3),
                _num(row.density_a, 3),
                _num(row.density_b, 3),
                row.verdict,
            ]
            for row in paradox.rows
        ],
    )
    lines += [
        f"{paradox.denser_than_both} pair(s) have an overlap denser than both communities -- "
        "the boundary is the core -- and "
        f"{paradox.sparser_than_both} have an overlap sparser than both -- a hole a "
        "stochastic-blockmodel reading of the cover would not expect.",
        "",
    ]
    return lines


def overlap_payload(report: OverlapReport) -> dict[str, Any]:
    """The same section as plain JSON-able data, for `--json`."""
    return {
        "implements": f"Atlas {CHAPTER} ({report.section})",
        "method": report.method,
        "frame": report.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "parameters": report.parameters,
        "communities": report.communities,
        "sizes": report.sizes,
        "overlap_nodes": report.overlap_nodes,
        "method_notes": report.method_notes,
        "flattened_partition": report.flattened,
        "flattening_notes": report.flattening_notes,
        "evaluation": evaluation_payload(report.evaluation),
        "paradox": {
            "denser_than_both": report.paradox.denser_than_both,
            "sparser_than_both": report.paradox.sparser_than_both,
            "rows": [
                {
                    "a": row.a + 1,
                    "b": row.b + 1,
                    "overlap_size": row.overlap_size,
                    "overlap_density": None
                    if math.isnan(row.overlap_density)
                    else row.overlap_density,
                    "density_a": None if math.isnan(row.density_a) else row.density_a,
                    "density_b": None if math.isnan(row.density_b) else row.density_b,
                    "verdict": row.verdict,
                }
                for row in report.paradox.rows
            ],
        },
        "baseline_communities": report.baseline_communities,
        "overlap_nmi_vs_baseline": (
            None if math.isnan(report.overlap_nmi_vs_baseline) else report.overlap_nmi_vs_baseline
        ),
        "notes": report.notes,
    }
