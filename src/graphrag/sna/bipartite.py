"""Bipartite community discovery (Atlas ch. 39): communities found *on* the two-mode network.

"Today: speakers-entities is projected before it is grouped" (ATL-39's own words for what this
module replaces). Projecting first (§39.2, ATL-26) throws away exactly the structure a community
in a two-mode network is made of -- "there are a number of different structures in bipartite
networks that project into the same unipartite graphs" (p. 563) -- so a proper bipartite method
has to read the incidence directly. Four things live here, one per section of the chapter.

*Evaluation* (§39.1). Modularity's own null model changes on a two-mode network: "the expected
number of edges ... is kukv/|E|" rather than kukv/2|E|, and "the sum of modularity is made only
across pairs of nodes of unlike types, otherwise we would have negative modularity contributions
from nodes that cannot be connected" (p. 561-562). :func:`barber_modularity` is that amended
quality function (Michael J. Barber, "Modularity and community detection in bipartite networks",
Physical Review E, 76(6):066102, 2007), and it is the number every method below is optimised for
and reported against -- never the unamended modularity :mod:`graphrag.sna.evaluate` computes on
the same partition, which is a different, wrong quantity here (see
:data:`BIPARTITE_EVALUATION_NOTE`, which every caller of ``evaluate_partition`` on a two-mode
partition appends to that section for exactly this reason).

*Direct module detection* (§39.3). :func:`brim` is Barber's own alternating optimisation over the
row and column community-indicator matrices ("BRIM" in the ATL-39 ticket and in the literature
that follows Barber's paper); :func:`bilouvain` is a Louvain-style local-moving optimiser that
gets there by the classical route -- move one node at a time into whichever neighbouring
community raises :func:`barber_modularity` the most -- amended the same way Barber's own
modularity is: the expected term a node compares itself against is only the opposite-mode degree
already inside a candidate community, never the whole community's degree. Both search for the
partition directly on the incidence; neither ever builds the projection.

*Neighbour similarity* (§39.4). "The point of a community in a bipartite network is not that the
nodes connect densely to each other, but that they connect to the same neighbours" (p. 568); the
book's own route there is co-clustering, "cluster the rows and the columns of your matrix at the
same time" (p. 569, citing Dhillon 2001). :func:`neighbor_similarity` is spectral co-clustering
(``sklearn.cluster.SpectralCoclustering``, already a project dependency) on the incidence matrix,
which is exactly that: no projection, no modularity, a community is a block of mutually similar
rows and columns.

**The null, always.** Every modularity above is read against :func:`graphrag.sna.null.
bipartite_preserving`, the curveball algorithm that holds every row's and every column's degree
of the two-mode network fixed -- never :func:`graphrag.sna.null.configuration` run on a
projection, which would rewire a graph no membership table could have produced (§18.1, and see
:mod:`graphrag.sna.null`'s own docstring on why the projection is not the observation).
:func:`bipartite_significance` draws that null, re-optimises the *same* method on each sample --
the comparison that discriminates, as :func:`graphrag.sna.cluster.null_model_modularity` argues
for the unipartite case -- and reports the z-score and empirical p through the one
:func:`graphrag.sna.null.significance` contract every null in this package uses.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import adjusted_rand_score

from graphrag.sna.coreperiphery import two_mode_sides
from graphrag.sna.matrices import Node
from graphrag.sna.null import (
    HOLDS_FIXED,
    Significance,
    bipartite_preserving,
    pairs_of,
    significance,
)

__all__ = [
    "BIPARTITE_EVALUATION_NOTE",
    "DEFAULT_K_RANGE",
    "MAX_DENSE_CELLS",
    "NULL_SAMPLES",
    "RESTARTS",
    "BipartiteCommunities",
    "BipartiteReport",
    "barber_modularity",
    "bilouvain",
    "bipartite_payload",
    "bipartite_report",
    "bipartite_significance",
    "brim",
    "neighbor_similarity",
    "render_bipartite",
]

#: Multiple random restarts of BRIM, the same convention :data:`graphrag.sna.coreperiphery.
#: RESTARTS` uses for the discrete core search -- the book leaves the search procedure open and
#: this is this package's answer to it, not the chapter's.
RESTARTS = 8

#: How many curveball samples :func:`bipartite_significance` draws by default, matching every
#: other null in this package (:data:`graphrag.sna.coreperiphery.NULL_SAMPLES`).
NULL_SAMPLES = 50

#: The candidate community counts :func:`brim` and :func:`neighbor_similarity` search when the
#: caller does not pin ``k``, mirroring :mod:`graphrag.sna.cluster`'s ``k_range`` for K-means.
DEFAULT_K_RANGE: tuple[int, ...] = tuple(range(2, 9))

#: Above this many cells in the dense incidence matrix, :func:`brim` and
#: :func:`neighbor_similarity` refuse rather than allocate it -- the same guard
#: :func:`graphrag.sna.coreperiphery.nestedness` uses for the same reason.
MAX_DENSE_CELLS = 20_000_000

_MAX_ITER = 200
_EPS = 1e-9


@dataclass(frozen=True)
class BipartiteCommunities:
    """One partition of a two-mode network, found directly on the incidence (§39.3-39.4).

    ``communities`` mix both modes freely -- a community is a set of rows and columns, never one
    side alone -- which is the point §39.1 makes about why the ordinary modularity sum has to be
    restricted to unlike-type pairs in the first place.
    """

    communities: list[list[Node]]
    modularity: float
    """:func:`barber_modularity` of ``communities``, the quality every method here optimises."""
    method: str
    k: int
    resolution: float
    restarts: int
    """How many independent optimisation runs were compared to reach this partition -- BRIM's
    random restarts, biLouvain's seeded runs, or 1 for neighbour similarity, which is
    deterministic given its own random state."""
    stability: float
    """Mean pairwise adjusted Rand index between the ``restarts`` runs at the ``k`` this
    partition was kept from. 1.0 when every run agreed; see :attr:`graphrag.sna.cluster.
    LouvainResult.stability` for the same reading."""
    seeds: list[int]
    iterations: int = 0
    """How many alternating/local-moving passes the winning run took to converge. 0 for
    neighbour similarity, which has no such loop."""

    @property
    def sizes(self) -> list[int]:
        return [len(c) for c in self.communities]


@dataclass
class BipartiteReport:
    """§39.1's evaluation of one :class:`BipartiteCommunities`, with the note that goes with it."""

    frame: str
    nodes: int
    edges: int
    row_mode: str
    column_mode: str
    result: BipartiteCommunities
    significance: Significance
    notes: list[str] = field(default_factory=list)


def _left_set(graph: nx.Graph) -> tuple[str, str, set[Node], list[Node], list[Node]]:
    """``two_mode_sides`` plus the row set as a lookup, raising the one message every caller
    uses."""
    sides = two_mode_sides(graph)
    if sides is None:
        msg = (
            "bipartite community discovery needs a two-mode network (§6.4, §39.1): this graph "
            "has one kind of node, or an edge that does not cross between two. Build the "
            "speakers-entities network without --project -- projecting first is exactly what "
            "chapter 39 replaces (§39.2) -- or use louvain/kmeans/gmm for a one-mode network."
        )
        raise ValueError(msg)
    row_mode, column_mode, rows, columns = sides
    return row_mode, column_mode, set(rows), rows, columns


def _weighted_degree(graph: nx.Graph) -> dict[Node, float]:
    return {node: float(degree) for node, degree in graph.degree(weight="weight")}


# ------------------------------------------------------------------------------- §39.1 quality


def barber_modularity(
    graph: nx.Graph, communities: Sequence[Iterable[Node]], *, resolution: float = 1.0
) -> float:
    """Barber's bipartite modularity of ``communities`` (§39.1, p. 561-562).

    ``Q = (1/m) * sum_c [e_c - resolution * a_c * b_c / m]``, where ``m`` is the total edge
    weight, ``e_c`` is the edge weight internal to community ``c``, and ``a_c``/``b_c`` are the
    row-side and column-side weighted degree sums of ``c``'s own members. This is the algebraic
    rearrangement of the chapter's per-pair sum ``A_uv - k_u*k_v/m`` over every unlike-type pair
    sharing a community, computed in one pass over the edges instead of one over every pair --
    the same trick the standard modularity of Section 36.1 uses, with ``m`` in place of ``2m``
    because the chapter's sum runs once over each unlike-type pair rather than twice.

    Raises ``ValueError`` when ``graph`` is not two-mode (§6.4): the sum above is defined on the
    incidence matrix, and there is no amended null to compute on a graph where every pair is
    already of "unlike type" or none is. Returns 0.0 for an edgeless graph, where nothing is
    internal to anything.
    """
    _, _, left, _, _ = _left_set(graph)
    membership: dict[Node, int] = {
        node: index for index, community in enumerate(communities) for node in community
    }
    m = float(graph.size(weight="weight"))
    if m == 0.0:
        return 0.0
    degree = _weighted_degree(graph)
    row_degree: dict[int, float] = defaultdict(float)
    column_degree: dict[int, float] = defaultdict(float)
    for node, community in membership.items():
        if node in left:
            row_degree[community] += degree.get(node, 0.0)
        else:
            column_degree[community] += degree.get(node, 0.0)
    internal: dict[int, float] = defaultdict(float)
    for u, v, data in graph.edges(data=True):
        cu = membership.get(u)
        cv = membership.get(v)
        if cu is not None and cu == cv:
            internal[cu] += float(data.get("weight", 1.0))
    communities_seen = set(row_degree) | set(column_degree)
    total = sum(
        internal.get(c, 0.0) - resolution * row_degree.get(c, 0.0) * column_degree.get(c, 0.0) / m
        for c in communities_seen
    )
    return total / m


def _with_isolates(graph: nx.Graph, groups: list[list[Node]]) -> list[list[Node]]:
    """Every node not already in ``groups`` becomes its own singleton community.

    A node the optimiser never touched -- degree zero, or dropped by a size guard upstream --
    still has to be covered, exactly as Louvain's own communities cover isolated nodes as
    singletons (:func:`networkx.community.louvain_communities`).
    """
    covered = {node for group in groups for node in group}
    leftover = [[node] for node in graph.nodes if node not in covered]
    return groups + leftover


def _stability(labelled_runs: Sequence[dict[Node, int]], nodes: Sequence[Node]) -> float:
    """Mean pairwise adjusted Rand index between runs, the same reading as
    :attr:`graphrag.sna.cluster.LouvainResult.stability`."""
    if len(labelled_runs) < 2:
        return 1.0
    vectors = [[run.get(node, -1) for node in nodes] for run in labelled_runs]
    pairs = [
        float(adjusted_rand_score(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return float(np.mean(pairs)) if pairs else 1.0


def _grouped(membership: dict[Node, int]) -> list[list[Node]]:
    buckets: dict[int, list[Node]] = defaultdict(list)
    for node, community in membership.items():
        buckets[community].append(node)
    return [sorted(buckets[key], key=str) for key in sorted(buckets)]


def _check_dense(rows: Sequence[Node], columns: Sequence[Node], caller: str) -> None:
    cells = len(rows) * len(columns)
    if cells > MAX_DENSE_CELLS:
        msg = (
            f"{caller} would build a dense {len(rows):,} x {len(columns):,} matrix "
            f"({cells:,} cells), more than MAX_DENSE_CELLS ({MAX_DENSE_CELLS:,}). Raise "
            "--min-weight, filter the network, or take one component first."
        )
        raise ValueError(msg)


# --------------------------------------------------------------------------- §39.3 direct: BRIM


def _brim_once(
    incidence: np.ndarray, k: int, rng: random.Random, resolution: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """One BRIM run from a random row assignment: alternate column, then row, updates to
    convergence (Barber 2007's own algorithm), returning the row and column labels, the final
    trace-modularity, and how many alternations it took.
    """
    n_rows, n_columns = incidence.shape
    m = float(incidence.sum())
    row_degree = incidence.sum(axis=1)
    column_degree = incidence.sum(axis=0)
    modularity_matrix = incidence - resolution * np.outer(row_degree, column_degree) / m
    row_labels = np.array([rng.randrange(k) for _ in range(n_rows)])
    column_labels = np.zeros(n_columns, dtype=np.int64)
    previous = -np.inf
    iterations = 0
    for step in range(1, _MAX_ITER + 1):
        iterations = step
        row_indicator = np.zeros((n_rows, k))
        row_indicator[np.arange(n_rows), row_labels] = 1.0
        column_scores = row_indicator.T @ modularity_matrix  # k x n_columns
        column_labels = np.argmax(column_scores, axis=0)
        column_indicator = np.zeros((n_columns, k))
        column_indicator[np.arange(n_columns), column_labels] = 1.0
        row_scores = modularity_matrix @ column_indicator  # n_rows x k
        row_labels = np.argmax(row_scores, axis=1)
        quality = float(np.trace(row_indicator.T @ modularity_matrix @ column_indicator)) / m
        if quality <= previous + _EPS:
            break
        previous = quality
    return row_labels, column_labels, previous, iterations


def brim(
    graph: nx.Graph,
    *,
    k: int | None = None,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    resolution: float = 1.0,
    restarts: int = RESTARTS,
    seed: int | None = None,
) -> BipartiteCommunities:
    """Barber's BRIM: alternating row/column module assignment maximising Barber modularity.

    Barber's own algorithm (§39.3, and the paper the chapter cites for it): fix a guess at which
    module every column belongs to, assign every row to the module that maximises its own
    contribution to :func:`barber_modularity`, then fix the rows and do the same for the
    columns, repeating until the modularity stops rising. This is an EM-style search and gets
    stuck in local optima, so ``restarts`` random starts are compared and the best kept -- the
    convention :func:`graphrag.sna.coreperiphery.discrete_core` uses for the same reason -- and
    when ``k`` is not pinned, every value in ``k_range`` is tried the same way and the best of
    all of them is returned, because BRIM (unlike Louvain) needs the number of modules stated
    up front.

    ``stability`` is the mean pairwise adjusted Rand index between the ``restarts`` runs at the
    winning ``k``, read the same way as :attr:`graphrag.sna.cluster.LouvainResult.stability`: a
    low score means the modules named below are an artefact of the random start, not of the data.

    Raises ``ValueError`` on a network that is not two-mode, or one whose incidence matrix would
    not fit in memory (see :data:`MAX_DENSE_CELLS`); on such a network use :func:`bilouvain`
    instead, which never builds the dense matrix.
    """
    _, _, _left, rows, columns = _left_set(graph)
    _check_dense(rows, columns, "brim()")
    row_index = {node: i for i, node in enumerate(rows)}
    column_index = {node: i for i, node in enumerate(columns)}
    dense = np.zeros((len(rows), len(columns)))
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if u in row_index and v in column_index:
            dense[row_index[u], column_index[v]] = weight
        elif v in row_index and u in column_index:
            dense[row_index[v], column_index[u]] = weight
    if dense.sum() == 0.0 or not rows or not columns:
        return BipartiteCommunities(
            communities=_with_isolates(graph, []),
            modularity=0.0,
            method="brim",
            k=0,
            resolution=resolution,
            restarts=0,
            stability=1.0,
            seeds=[],
        )
    candidates = [
        c for c in ([k] if k is not None else k_range) if 2 <= c <= min(len(rows), len(columns))
    ]
    if not candidates:
        candidates = [min(2, min(len(rows), len(columns)))]
    base_seed = random.SystemRandom().randrange(1_000_000) if seed is None else seed
    best: tuple[int, float, list[np.ndarray], list[np.ndarray], list[int], int] | None = None
    for candidate_k in candidates:
        row_runs: list[np.ndarray] = []
        column_runs: list[np.ndarray] = []
        qualities: list[float] = []
        run_seeds: list[int] = []
        iterations_used = 0
        for restart in range(max(restarts, 1)):
            run_seed = base_seed + restart
            run_seeds.append(run_seed)
            row_labels, column_labels, quality, iterations = _brim_once(
                dense,
                candidate_k,
                random.Random(run_seed),  # noqa: S311 -- reproducibility, not secrecy
                resolution,
            )
            row_runs.append(row_labels)
            column_runs.append(column_labels)
            qualities.append(quality)
            iterations_used = max(iterations_used, iterations)
        top = int(np.argmax(qualities))
        if best is None or qualities[top] > best[1]:
            best = (candidate_k, qualities[top], row_runs, column_runs, run_seeds, iterations_used)
    assert best is not None  # candidates is never empty
    _, _, row_runs, column_runs, run_seeds, iterations_used = best
    labelled = [
        {
            **{rows[i]: int(label) for i, label in enumerate(row_labels)},
            **{columns[i]: int(label) for i, label in enumerate(column_labels)},
        }
        for row_labels, column_labels in zip(row_runs, column_runs, strict=True)
    ]
    top = int(
        np.argmax([barber_modularity(graph, _grouped(m), resolution=resolution) for m in labelled])
    )
    found = _grouped(labelled[top])
    communities = _with_isolates(graph, found)
    modularity = barber_modularity(graph, communities, resolution=resolution)
    stability = _stability(labelled, rows + columns)
    return BipartiteCommunities(
        communities=communities,
        modularity=modularity,
        method="brim",
        k=len(found),
        resolution=resolution,
        restarts=len(row_runs),
        stability=stability,
        seeds=run_seeds,
        iterations=iterations_used,
    )


# ---------------------------------------------------------------------- §39.3 direct: biLouvain


def bilouvain(
    graph: nx.Graph,
    *,
    resolution: float = 1.0,
    seed: int | None = None,
    runs: int = 10,
) -> BipartiteCommunities:
    """A Louvain-style local-moving optimiser for :func:`barber_modularity` (§39.3, "adapting
    classical approaches"), run directly on the incidence rather than on a projection.

    Standard Louvain moves one node at a time into whichever neighbouring community raises
    modularity the most, comparing the node's degree against the *whole* target community's
    degree. That comparison is exactly what §39.1 says is wrong for a two-mode network -- a row
    node has no edges to give a column-side community's row members, so their degree has to be
    excluded from what it is compared against. Here the gain of moving row node ``u`` into
    community ``c`` is ``weight(u, c) - resolution * k_u * b_c / m``, where ``b_c`` is only the
    *column-side* degree already in ``c`` (symmetrically for a column node and ``a_c``); this is
    Barber's own amendment applied to Louvain's gain formula rather than to a finished partition.
    A node may also start a fresh singleton community at zero gain, which is what keeps this
    procedure from ever lowering modularity, the same guarantee plain Louvain has.

    This is single-level: nodes move but communities are never coarsened into supernodes and
    re-optimised a second time, unlike the multi-pass Louvain :mod:`graphrag.sna.cluster` runs
    for unipartite networks. A second level would need supernodes that are themselves two-mode,
    which the book does not work through for this chapter; ``runs`` different random visiting
    orders substitute for it the way :func:`graphrag.sna.cluster.louvain`'s multiple seeds do,
    and the best of them is kept.

    Never builds a dense matrix, so it has no analogue of :data:`MAX_DENSE_CELLS` and is the
    method to reach for when :func:`brim` or :func:`neighbor_similarity` refuse a network as too
    large.
    """
    _, _, left, rows, columns = _left_set(graph)
    nodes = rows + columns
    m = float(graph.size(weight="weight"))
    if m == 0.0 or not nodes:
        return BipartiteCommunities(
            communities=_with_isolates(graph, []),
            modularity=0.0,
            method="bilouvain",
            k=0,
            resolution=resolution,
            restarts=0,
            stability=1.0,
            seeds=[],
        )
    degree = _weighted_degree(graph)
    neighbours: dict[Node, dict[Node, float]] = {node: {} for node in nodes}
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        neighbours.setdefault(u, {})[v] = neighbours.get(u, {}).get(v, 0.0) + weight
        neighbours.setdefault(v, {})[u] = neighbours.get(v, {}).get(u, 0.0) + weight

    def _one_run(rng: random.Random) -> tuple[dict[Node, int], int]:
        community: dict[Node, int] = {node: index for index, node in enumerate(nodes)}
        row_degree: dict[int, float] = defaultdict(float)
        column_degree: dict[int, float] = defaultdict(float)
        for node, c in community.items():
            (row_degree if node in left else column_degree)[c] += degree.get(node, 0.0)
        next_id = len(nodes)
        order = list(nodes)
        improved = True
        iterations = 0
        while improved and iterations < _MAX_ITER:
            improved = False
            iterations += 1
            rng.shuffle(order)
            for node in order:
                is_row = node in left
                own = row_degree if is_row else column_degree
                other = column_degree if is_row else row_degree
                old_c = community[node]
                own[old_c] -= degree.get(node, 0.0)
                weight_to: dict[int, float] = defaultdict(float)
                for neighbour, weight in neighbours.get(node, {}).items():
                    weight_to[community[neighbour]] += weight
                best_c, best_gain = next_id, 0.0
                for c, weight in weight_to.items():
                    gain = weight - resolution * degree.get(node, 0.0) * other.get(c, 0.0) / m
                    if gain > best_gain:
                        best_gain, best_c = gain, c
                community[node] = best_c
                own[best_c] += degree.get(node, 0.0)
                if best_c == next_id:
                    next_id += 1
                if best_c != old_c:
                    improved = True
        return community, iterations

    base_seed = random.SystemRandom().randrange(1_000_000) if seed is None else seed
    runs_out: list[dict[Node, int]] = []
    qualities: list[float] = []
    seeds: list[int] = []
    max_iterations = 0
    for run in range(max(runs, 1)):
        run_seed = base_seed + run
        seeds.append(run_seed)
        membership, iterations = _one_run(random.Random(run_seed))  # noqa: S311
        max_iterations = max(max_iterations, iterations)
        runs_out.append(membership)
        qualities.append(barber_modularity(graph, _grouped(membership), resolution=resolution))
    top = int(np.argmax(qualities))
    communities = _with_isolates(graph, _grouped(runs_out[top]))
    modularity = barber_modularity(graph, communities, resolution=resolution)
    stability = _stability(runs_out, nodes)
    return BipartiteCommunities(
        communities=communities,
        modularity=modularity,
        method="bilouvain",
        k=len(communities),
        resolution=resolution,
        restarts=len(runs_out),
        stability=stability,
        seeds=seeds,
        iterations=max_iterations,
    )


# --------------------------------------------------------------------- §39.4 neighbour similarity


def neighbor_similarity(
    graph: nx.Graph,
    *,
    k: int | None = None,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    seed: int | None = None,
) -> BipartiteCommunities:
    """Spectral co-clustering of rows and columns at once (§39.4, p. 568-569).

    "You can pick any of your favorite machine learning algorithms ... and interpret their
    clusters as communities" once a bipartite network is read as a data matrix; the book singles
    out co-clustering because it "will cluster the rows and the columns of your matrix at the
    same time" (p. 569, citing Dhillon, "Co-clustering documents and words using bipartite
    spectral graph partitioning", SIGKDD 2001) rather than clustering one side and having to map
    the result back onto the other by hand. ``sklearn.cluster.SpectralCoclustering`` is that
    algorithm: it reads the (binary) incidence matrix and returns one row cluster and one column
    cluster per module, which this function pairs up into the mixed-mode communities every other
    method here returns.

    This asks a different question from :func:`brim` and :func:`bilouvain`: not "which rows and
    columns connect more than a null model expects" but "which rows and columns have similar
    neighbourhoods", so its own :func:`barber_modularity` is a comparison against the other
    methods rather than the quantity it was fitted for -- co-clustering does not optimise it, and
    a lower score here is not a defect the way a lower BRIM score would be.

    Unlike ``incidence``'s general contract, isolated nodes (zero degree) are dropped before
    clustering and returned as singleton communities, because a zero row or column has no
    neighbourhood to be similar to anything else's.

    Raises ``ValueError`` on a network that is not two-mode, one too large for a dense matrix
    (:data:`MAX_DENSE_CELLS`), or when the requested/candidate ``k`` exceeds the smaller side's
    node count, which ``SpectralCoclustering`` cannot honour.
    """
    _, _, _, rows, columns = _left_set(graph)
    degree = _weighted_degree(graph)
    rows = [node for node in rows if degree.get(node, 0.0) > 0]
    columns = [node for node in columns if degree.get(node, 0.0) > 0]
    _check_dense(rows, columns, "neighbor_similarity()")
    if not rows or not columns:
        return BipartiteCommunities(
            communities=_with_isolates(graph, []),
            modularity=0.0,
            method="neighbor-similarity",
            k=0,
            resolution=1.0,
            restarts=0,
            stability=1.0,
            seeds=[],
        )
    row_index = {node: i for i, node in enumerate(rows)}
    column_index = {node: i for i, node in enumerate(columns)}
    dense = np.zeros((len(rows), len(columns)))
    for u, v in graph.edges():
        if u in row_index and v in column_index:
            dense[row_index[u], column_index[v]] = 1.0
        elif v in row_index and u in column_index:
            dense[row_index[v], column_index[u]] = 1.0
    ceiling = min(len(rows), len(columns))
    candidates = [c for c in ([k] if k is not None else k_range) if 2 <= c <= ceiling]
    if not candidates:
        msg = (
            f"neighbor_similarity() needs 2 <= k <= {ceiling} (the smaller side of the incidence "
            f"matrix); got {k if k is not None else list(k_range)}."
        )
        raise ValueError(msg)
    base_seed = random.SystemRandom().randrange(1_000_000) if seed is None else seed
    best: tuple[int, float, list[list[Node]]] | None = None
    for candidate_k in candidates:
        model = SpectralCoclustering(n_clusters=candidate_k, random_state=base_seed)
        model.fit(dense)
        buckets: dict[int, list[Node]] = defaultdict(list)
        for node, label in zip(rows, model.row_labels_, strict=True):
            buckets[int(label)].append(node)
        for node, label in zip(columns, model.column_labels_, strict=True):
            buckets[int(label)].append(node)
        groups = [sorted(buckets[key], key=str) for key in sorted(buckets) if buckets[key]]
        modularity = barber_modularity(graph, _with_isolates(graph, groups), resolution=1.0)
        if best is None or modularity > best[1]:
            best = (candidate_k, modularity, groups)
    assert best is not None  # candidates is never empty
    _, _, groups = best
    communities = _with_isolates(graph, groups)
    modularity = barber_modularity(graph, communities, resolution=1.0)
    return BipartiteCommunities(
        communities=communities,
        modularity=modularity,
        method="neighbor-similarity",
        k=len(groups),
        resolution=1.0,
        restarts=1,
        stability=1.0,
        seeds=[base_seed],
    )


# ------------------------------------------------------------------------------- §39.1 the null


def _refit(
    method: str, *, k: int, resolution: float, seed: int
) -> Callable[[nx.Graph], BipartiteCommunities]:
    """The same method, pinned to the ``k`` (or nothing, for biLouvain) the observed run found,
    with one restart -- the same "re-optimise on the null sample" cost
    :func:`graphrag.sna.cluster.rewired_modularity` pays for the unipartite null."""
    if method == "brim":
        return lambda rewired: brim(rewired, k=k, resolution=resolution, restarts=1, seed=seed)
    if method == "bilouvain":
        return lambda rewired: bilouvain(rewired, resolution=resolution, runs=1, seed=seed)
    if method == "neighbor-similarity":
        return lambda rewired: neighbor_similarity(rewired, k=k, seed=seed)
    msg = f"unknown bipartite method {method!r}"
    raise ValueError(msg)


def bipartite_significance(
    graph: nx.Graph,
    result: BipartiteCommunities,
    *,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
) -> Significance:
    """:func:`barber_modularity` against :func:`graphrag.sna.null.bipartite_preserving` (§39.1).

    Never the configuration model on a projection of this network -- there is no projection here
    to rewire -- and never on the raw two-mode graph either, which would break the very
    memberships (one document, one speaker's passage, one entity) the edges are made of. The
    curveball keeps every row's and every column's degree exactly and re-deals the rest
    (:data:`graphrag.sna.null.HOLDS_FIXED`\\ ``["bipartite_preserving"]``).

    Each sample is *re-optimised* with the same method at the same ``k`` (or, for biLouvain,
    with a fresh local-moving run), not scored with the observed partition transplanted onto it:
    :func:`graphrag.sna.cluster.null_model_modularity` makes the case for why the fixed-partition
    version calls a random graph "structured", and it applies here identically. One restart per
    null sample, not :data:`RESTARTS`, is the same cost trade every rewiring-based null in this
    package makes.
    """
    row_mode, column_mode, left, _, _ = _left_set(graph)
    observed = barber_modularity(graph, result.communities, resolution=result.resolution)
    pairs = [(u, v) if u in left else (v, u) for u, v in graph.edges()]
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    null_scores: list[float] = []
    for sample in bipartite_preserving(pairs, max(samples, 0), seed=rng):
        drawn = pairs_of(sample)
        if not drawn:
            continue
        rewired = nx.Graph()
        for u, v in drawn:
            rewired.add_node(u, mode=row_mode)
            rewired.add_node(v, mode=column_mode)
            rewired.add_edge(u, v, weight=1.0)
        run_seed = rng.randrange(1_000_000)
        rerun = _refit(
            result.method, k=max(result.k, 2), resolution=result.resolution, seed=run_seed
        )(rewired)
        null_scores.append(rerun.modularity)
    return significance(observed, null_scores, null="bipartite_preserving")


# ----------------------------------------------------------------------------- the whole report


#: What §36's battery does and does not mean once the partition mixes two node types. Every
#: caller that runs ``evaluate_partition`` on a bipartite partition appends this to
#: ``PartitionScores.notes`` rather than duplicating the battery's table here.
BIPARTITE_EVALUATION_NOTE = (
    "This is a two-mode partition (§39.1). The modularity in `### Modularity` above is "
    "nx.community.modularity's unamended formula, which gives every same-type pair inside a "
    "community a negative expected contribution it should not have (p. 561-562) -- the number "
    "to trust is Barber modularity in `## Bipartite community discovery` below, against the "
    "curveball null, never this section's. internal_density and cut_ratio divide by every "
    "possible pair including same-type ones that can never connect, so both read lower than "
    "they would on a one-mode partition of the same sizes; performance inherits the same defect "
    "for the same reason (it credits a partition for correctly calling a same-type pair a "
    "non-edge, which was never in question); triangle_participation is undefined by "
    "construction, because a bipartite network has no triangles at all (§39.1, p. 567). "
    "conductance, expansion, the out-degree fractions and coverage count only edges that exist "
    "and stay meaningful, as does the mutual-information section against a ground truth. The "
    "link-prediction AUC (§36.3) reads too high here: most of its sampled non-edges are "
    "same-type pairs that were never candidates for an edge, not missing links the partition "
    "failed to predict."
)


def bipartite_report(
    graph: nx.Graph,
    method: str,
    *,
    k: int | None = None,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    resolution: float = 1.0,
    restarts: int = RESTARTS,
    runs: int = 10,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
) -> BipartiteReport:
    """Run one of §39.3-39.4's methods and score it against §39.1's null: the whole chapter.

    ``method`` is one of ``"brim"``, ``"bilouvain"`` or ``"neighbor-similarity"``. This is the
    entry point :func:`graphrag.sna.analysis.run_analysis` calls for
    ``--network speakers-entities`` without ``--project``; the general ch. 36 battery still runs
    on the returned partition through the same ``evaluate_partition`` call every other method
    goes through, with :data:`BIPARTITE_EVALUATION_NOTE` appended by the caller.
    """
    row_mode, column_mode, _, _, _ = _left_set(graph)
    if method == "brim":
        result = brim(
            graph, k=k, k_range=k_range, resolution=resolution, restarts=restarts, seed=seed
        )
    elif method == "bilouvain":
        result = bilouvain(graph, resolution=resolution, seed=seed, runs=runs)
    elif method == "neighbor-similarity":
        result = neighbor_similarity(graph, k=k, k_range=k_range, seed=seed)
    else:
        msg = f"method must be one of 'brim', 'bilouvain', 'neighbor-similarity', got {method!r}"
        raise ValueError(msg)
    sig = bipartite_significance(graph, result, samples=samples, seed=seed)
    return BipartiteReport(
        frame=str(graph.graph.get("frame", "")),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        row_mode=row_mode,
        column_mode=column_mode,
        result=result,
        significance=sig,
    )


def _num(value: float, places: int = 4) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.{places}f}"


#: The section each method implements, for the ``## Bipartite community discovery`` frame line.
_METHOD_SECTION: dict[str, str] = {
    "brim": "§39.3 (BRIM)",
    "bilouvain": "§39.3 (biLouvain)",
    "neighbor-similarity": "§39.4 (neighbour similarity)",
}


def render_bipartite(report: BipartiteReport) -> list[str]:
    """The ``## Bipartite community discovery`` section: method, Barber modularity, the null."""
    result = report.result
    section = _METHOD_SECTION.get(result.method, "§39.3")
    lines = [
        "## Bipartite community discovery",
        "",
        f"**Implements.** §39.1 (Barber modularity and its null), and {section} for the "
        f"partition itself. {report.row_mode}s and {report.column_mode}s are grouped together, "
        "directly on the two-mode network -- nothing here was projected.",
        "",
        f"n = {report.nodes:,} nodes, {report.edges:,} edges, {len(result.communities)} "
        f"communit{'y' if len(result.communities) == 1 else 'ies'} (sizes "
        f"{', '.join(str(size) for size in result.sizes[:12])}"
        f"{', …' if len(result.sizes) > 12 else ''}).",
        "",
        f"- method: {result.method} (k={result.k}, resolution={_num(result.resolution, 2)})",
        f"- Barber modularity: {_num(result.modularity)}",
        f"- stability across {result.restarts} run(s): {_num(result.stability, 3)} (mean "
        "pairwise adjusted Rand index)",
    ]
    sig = report.significance
    if sig.testable:
        lines.append(
            f"- against {sig.samples} curveball rewiring(s) "
            f"(`bipartite_preserving`, holds fixed {HOLDS_FIXED['bipartite_preserving']}): "
            f"null mean {_num(sig.null_mean)}, sd {_num(sig.null_std)}, z={_num(sig.z, 2)}, "
            f"p={_num(sig.p_value, 4)}"
        )
        if sig.caveat:
            lines.append(f"  > {sig.caveat}")
    else:
        lines.append(f"- {sig.caveat}")
    lines.append("")
    return lines


def bipartite_payload(report: BipartiteReport) -> dict[str, Any]:
    result = report.result
    return {
        "frame": report.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "row_mode": report.row_mode,
        "column_mode": report.column_mode,
        "method": result.method,
        "k": result.k,
        "resolution": result.resolution,
        "modularity": result.modularity,
        "stability_adjusted_rand": result.stability,
        "restarts": result.restarts,
        "sizes": result.sizes,
        "communities": result.communities,
        "significance": {
            "null": report.significance.null,
            "samples": report.significance.samples,
            "null_mean": report.significance.null_mean,
            "null_std": report.significance.null_std,
            "z": report.significance.z,
            "p_value": report.significance.p_value,
            "caveat": report.significance.caveat,
        },
        "notes": report.notes,
    }
