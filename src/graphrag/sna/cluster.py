"""Community detection and clustering, with the checks that make a result reportable.

Two kinds of input are handled here and they are not interchangeable:

*Edges.* ``louvain`` partitions a graph directly by maximising modularity. It needs no ``k``,
but it is randomised, so it is run many times and the runs are compared with the adjusted Rand
index; and a high modularity is not by itself evidence of structure, so
``null_model_modularity`` scores the partition against degree-preserving rewirings.

*Vectors.* ``kmeans`` and ``gmm`` cluster one row per node. Those rows come either from
``spectral_embedding`` (the graph's own structure, turned into coordinates) or from the store's
mean text embeddings (what the nodes are *about*). The two answer different questions and a
report must say which was used.

Nothing here re-implements an algorithm: Louvain and modularity come from ``networkx``, K-means
and Gaussian mixtures from ``scikit-learn``, spectral embedding from
:func:`graphrag.sna.embed.spectral` (ATL-42's §42.1 contract over the ATL-08 Laplacian), and
every statistic -- the agreement indices, the null-model z-score -- from
:mod:`graphrag.sna.stats`, which is chapter 3 of the Atlas in one place.

*Hierarchies.* Chapter 37 asks a different question of the same partitions: not which one is
best, but whether several of them, at several scales, are all valid readings of the same network.
``louvain_levels`` and ``girvan_newman_dendrogram`` are the merging and splitting approaches of
§37.1, each returning a :class:`Dendrogram` cut by its own modularity profile; ``hrg_fit`` is the
Hierarchical Random Graph of §37.2, fitted by the Markov chain Monte Carlo search the chapter
describes rather than re-implemented from a library, because no dependency here has one; and
``directed_communities`` is §37.3's directed reading of the same Louvain algorithm, on the arrows
themselves rather than a flattened graph.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from graphrag.sna.embed import Embedding
from graphrag.sna.embed import spectral as _spectral_embedding
from graphrag.sna.matrices import adjacency, dense, stochastic
from graphrag.sna.measures import undirected_view
from graphrag.sna.null import configuration

# Re-exported rather than defined here: the agreement indices are statistics, not clustering,
# and live in ``stats`` with the rest of chapter 3 (§3.5). The name stays importable from
# this module because that is where every caller and every test already reaches for it.
from graphrag.sna.stats import compare_partitions as compare_partitions
from graphrag.sna.stats import mean, std, z_score
from graphrag.sna.walks import stationary_distribution

Labels = list[int]

COVARIANCES: tuple[str, ...] = ("full", "tied", "diag", "spherical")


@dataclass(frozen=True)
class LouvainResult:
    """A Louvain partition plus the evidence needed to trust it."""

    communities: list[list[str]]
    modularity: float
    resolution: float
    modularities: list[float]
    seeds: list[int]
    stability: float
    """Mean pairwise adjusted Rand index across the runs. 1.0 means every seed found the same
    partition; below about 0.6 the community boundaries are an artefact of the seed, not of the
    data, and individual memberships should not be quoted."""
    note: str = ""
    """What a directed network cost this partition, or "" when there was nothing to say. Louvain
    maximises a modularity defined on undirected graphs, so a directed one is flattened first
    (§6.2) and the report prints this sentence beside the communities."""

    @property
    def sizes(self) -> list[int]:
        return [len(c) for c in self.communities]

    def labels(self, nodes: Sequence[str]) -> Labels:
        return _label_vector(self.communities, nodes)


@dataclass(frozen=True)
class KMeansResult:
    k: int
    labels: Labels
    inertia: float
    silhouette: float
    seed: int


@dataclass(frozen=True)
class GMMResult:
    k: int
    labels: Labels
    responsibilities: list[list[float]]
    """Soft membership: row ``i`` is how much node ``i`` belongs to each component. This is the
    output to use when a node legitimately belongs to more than one group."""
    bic: float
    aic: float
    covariance: str
    seed: int
    converged: bool


@dataclass(frozen=True)
class KChoice:
    """One row per candidate ``k``, with the picks marked."""

    rows: list[dict[str, float]] = field(default_factory=list)
    best_silhouette_k: int | None = None
    elbow_k: int | None = None
    best_bic_k: int | None = None


@dataclass(frozen=True)
class NullModelResult:
    """The observed modularity against degree-preserving rewirings of the same graph."""

    observed: float
    null_mean: float
    null_std: float
    z_score: float
    samples: int

    @property
    def verdict(self) -> str:
        if self.samples == 0:
            # Either too small to have a swap, or -- the directed case -- big enough and still
            # unswappable, because every three-arc rotation would duplicate an arc. Both are "no
            # null was built", and neither may be printed as a z-score of 0.00.
            return "not testable: no rewiring of this network was possible"
        if self.z_score >= 3.0:
            return "structure well beyond chance (z >= 3)"
        if self.z_score >= 2.0:
            return "structure above chance (z >= 2)"
        return "not distinguishable from a random graph with the same degrees"


def _label_vector(communities: Sequence[Iterable[str]], nodes: Sequence[str]) -> Labels:
    membership = {node: index for index, community in enumerate(communities) for node in community}
    return [membership.get(node, -1) for node in nodes]


def _seeds(seed: int | None, runs: int) -> list[int]:
    """Deterministic when ``seed`` is given; otherwise drawn once so a run is still repeatable
    from the seeds printed in the report."""
    base = random.SystemRandom().randrange(1_000_000) if seed is None else seed
    return [base + i for i in range(max(runs, 1))]


def _run_louvain(
    graph: nx.Graph, resolution: float, seed: int | None, runs: int, note: str
) -> LouvainResult:
    """The multi-seed Louvain run shared by :func:`louvain` and :func:`directed_communities`.

    The two differ only in whether ``graph`` was flattened before it got here and in what
    ``note`` says about that; ``nx.community.louvain_communities`` and ``nx.community.modularity``
    both already dispatch on ``graph.is_directed()`` themselves (Dugué and Perez's directed
    modularity gain when it is, the standard one when it is not), so nothing else changes.

    A graph with nodes and no edges is answered without calling either: ``nx.community.
    modularity`` divides by ``2 * |E|`` and raises ``ZeroDivisionError`` on such a graph, but the
    right partition is not in doubt -- with no edges there is no co-membership evidence for any
    two nodes, so every node is its own community. That partition's modularity is exactly 0 (the
    empty sum), and every seed necessarily finds it, so stability is 1.0 rather than undefined.
    """
    if graph.number_of_edges() == 0:
        seeds = _seeds(seed, runs)
        communities = [[node] for node in sorted(graph.nodes, key=str)]
        return LouvainResult(
            communities=communities,
            modularity=0.0,
            resolution=resolution,
            modularities=[0.0] * len(seeds),
            seeds=seeds,
            stability=1.0,
            note=note,
        )
    seeds = _seeds(seed, runs)
    partitions: list[list[list[str]]] = []
    modularities: list[float] = []
    for run_seed in seeds:
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=resolution, seed=run_seed
        )
        ordered = sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0]))
        partitions.append(ordered)
        modularities.append(
            float(nx.community.modularity(graph, ordered, weight="weight", resolution=resolution))
        )
    nodes = list(graph.nodes)
    vectors = [_label_vector(p, nodes) for p in partitions]
    pairs = [
        float(adjusted_rand_score(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    best = int(np.argmax(modularities))
    return LouvainResult(
        communities=partitions[best],
        modularity=modularities[best],
        resolution=resolution,
        modularities=modularities,
        seeds=seeds,
        stability=float(np.mean(pairs)) if pairs else 1.0,
        note=note,
    )


def louvain(
    graph: nx.Graph, resolution: float = 1.0, seed: int | None = None, runs: int = 10
) -> LouvainResult:
    """Louvain community detection, run ``runs`` times and scored for stability.

    Louvain is greedy and randomised: a different seed can produce a different partition of the
    same graph. Running it once and reporting the communities hides that. Here every run is
    kept, the best-modularity partition is returned, and the mean pairwise adjusted Rand index
    across runs is reported as ``stability``.

    Two limits are worth stating in any write-up. ``resolution`` above 1.0 finds smaller
    communities and below 1.0 finds larger ones, so the number of communities is partly a
    choice, not a finding. And modularity has a resolution limit: in a large network, genuine
    small communities get absorbed into larger ones whatever the seed.

    A third applies to the ``relations`` network: modularity as used here is defined on
    undirected graphs, so a directed network is flattened before it is partitioned (§6.2) and
    the returned ``note`` says so. The communities are therefore about who is related to whom,
    never about which way the relation ran. Use :func:`directed_communities` to read the arrows
    themselves instead of discarding them (§37.3).
    """
    if graph.number_of_nodes() == 0:
        return LouvainResult([], 0.0, resolution, [], [], 1.0)
    graph, flattened = undirected_view(graph)
    note = f"Louvain and its degree-preserving null model ran {flattened}." if flattened else ""
    return _run_louvain(graph, resolution, seed, runs, note)


def directed_communities(
    graph: nx.Graph, resolution: float = 1.0, seed: int | None = None, runs: int = 10
) -> LouvainResult:
    """Louvain over the directed modularity of §37.3, without flattening the network first.

    §37.3 makes directed community discovery its own question: *"we can get to the scenario of
    directed community discovery... here we totally reject the idea that we should group
    communities into super communities"* and instead let the arrows themselves decide the
    groups, using *"one of the many competing versions of directed modularity"* the chapter
    already pointed to in §36.1. Leicht and Newman's version -- ``k_c^in * k_c^out`` in place of
    ``k_c^2`` -- is exactly what ``networkx``'s own ``louvain_communities`` and ``modularity``
    compute once handed a directed graph (Dugué and Perez, 2015, cited by that implementation),
    so this is :func:`louvain` with the flattening step removed, run and stabilised the same way.

    This can separate two blocks that :func:`louvain` -- which flattens first (§6.2) -- would
    merge: when almost every edge between two blocks runs the same way, the directed null model
    already expects that flow from each side's in- and out-degree, so it costs the partition
    little to keep the blocks apart; an undirected reading of the same edges counts them as
    ordinary co-membership evidence with no direction to explain away, and can find it cheaper
    to put both blocks in one community instead.

    Raises on an undirected graph, which has no direction to arrange communities by -- use
    :func:`louvain` there instead.
    """
    if not graph.is_directed():
        msg = (
            "directed_communities() needs a directed graph (§37.3): an undirected network has no "
            "direction for the communities to be arranged by. Use louvain() instead, or build "
            "--network relations, the one directed network this package exports."
        )
        raise ValueError(msg)
    if graph.number_of_nodes() == 0:
        return LouvainResult([], 0.0, resolution, [], [], 1.0)
    note = (
        "Modularity here is the directed form (§36.1, §37.3): k_c^in * k_c^out in place of "
        "k_c^2, computed without flattening the network."
    )
    return _run_louvain(graph, resolution, seed, runs, note)


def spectral_embedding_full(graph: nx.Graph, dims: int = 8, seed: int | None = None) -> Embedding:
    """The §42.1 contract behind :func:`spectral_embedding`: the same graph, the same clamped
    ``dims``, but the :class:`~graphrag.sna.embed.Embedding` itself rather than only its
    ``.vectors`` -- node order, method name and, in particular, the provenance that says whether
    an eigengap truncated ``dims`` further than the clamp already did (see
    :func:`graphrag.sna.embed.spectral`). ``analysis.py``'s report calls this one, not
    :func:`spectral_embedding`, because it needs that provenance to say why fewer columns came
    back than ``dims`` asked for.
    """
    flat, _ = undirected_view(graph)
    nodes = list(flat.nodes)
    if len(nodes) < 2:
        vectors = np.zeros((len(nodes), 1), dtype=np.float64)
        provenance: dict[str, object] = {"chapter": "§42.3", "n_nodes": len(nodes)}
        return Embedding(vectors=vectors, nodes=nodes, method="spectral", provenance=provenance)
    capped = max(1, min(dims, len(nodes) - 1))
    return _spectral_embedding(flat, capped, seed=seed, nodes=nodes)


def spectral_embedding(graph: nx.Graph, dims: int = 8, seed: int | None = None) -> np.ndarray:
    """Coordinates for each node from the normalised Laplacian, aligned to ``list(graph.nodes)``.

    This is the bridge from edges to vectors: once a graph has coordinates, K-means and Gaussian
    mixtures apply, and you can ask for exactly ``k`` groups.

    ``dims`` is clamped to ``n - 1``; an embedding cannot have more dimensions than the graph
    has nodes to separate. A directed network is flattened first (§6.2), since the Laplacian
    behind the embedding is defined on a symmetric adjacency; ``measures.directed_notes`` is
    where the report says so.

    The vectors themselves come from :func:`graphrag.sna.embed.spectral` (ATL-42), the §42.1
    embedding contract over the same symmetric normalised Laplacian this function always used
    (§8.4), by way of :func:`spectral_embedding_full`; only ``.vectors`` is returned here,
    because every caller of this function wants coordinates to feed to K-means or a Gaussian
    mixture, not the contract's node order, method name and provenance -- callers that want those
    call :func:`spectral_embedding_full`. ``dims`` is clamped rather than refused, unlike the
    contract's own bound, because these callers have already decided how many coordinates they
    want and raising would not change what happens next; ``seed`` is accepted for every caller's
    call signature and recorded but does nothing, since the underlying solver
    (``numpy.linalg.eigh``) is deterministic.

    **The vectors returned may have fewer than ``dims`` columns.** When ``dims`` would cut into a
    degenerate eigenspace -- an exactly regular subgraph, most commonly a clique, has no internal
    structure to spread ``dims`` eigenvectors across -- :func:`graphrag.sna.embed.spectral`
    truncates to the last dimension before that block rather than handing K-means a solver
    artefact dressed up as structure. A caller that must know how many columns came back reads
    ``.shape[1]``; a caller that must say why calls :func:`spectral_embedding_full` instead.
    """
    return spectral_embedding_full(graph, dims, seed).vectors


def _silhouette(features: np.ndarray, labels: Labels) -> float:
    """Silhouette, or 0.0 when it is undefined (one cluster, or one point per cluster)."""
    distinct = len(set(labels))
    if distinct < 2 or distinct >= len(labels):
        return 0.0
    return float(silhouette_score(features, labels))


def kmeans(features: np.ndarray, k: int, seed: int | None = None) -> KMeansResult:
    """K-means (k-means++ initialisation) over one row per node.

    Assumes clusters that are roughly spherical and similar in spread, and assigns every point
    to exactly one of them. When those assumptions hold it is the clearest thing to report;
    when clusters overlap or differ in shape, use ``gmm`` instead.
    """
    matrix = np.asarray(features, dtype=np.float64)
    run_seed = _seeds(seed, 1)[0]
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=run_seed)
    labels = [int(label) for label in model.fit_predict(matrix)]
    return KMeansResult(
        k=k,
        labels=labels,
        inertia=float(model.inertia_),
        silhouette=_silhouette(matrix, labels),
        seed=run_seed,
    )


def _elbow(ks: Sequence[int], inertias: Sequence[float]) -> int | None:
    """The ``k`` furthest from the straight line joining the first and last points of the
    inertia curve: the classic elbow, computed rather than eyeballed."""
    if len(ks) < 3:
        return None
    x = np.asarray(ks, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)
    span = y[0] - y[-1]
    if span == 0:
        return None
    y = (y - y[-1]) / span
    x = (x - x[0]) / (x[-1] - x[0])
    distances = np.abs(y - (1.0 - x))
    return int(ks[int(np.argmax(distances))])


def choose_k_kmeans(
    features: np.ndarray, k_range: Sequence[int], seed: int | None = None
) -> KChoice:
    """Silhouette and inertia for each candidate ``k``, with the elbow and best silhouette marked.

    The two criteria can disagree. Silhouette asks how well separated the clusters are; the
    elbow asks where adding another cluster stops buying much. When they disagree, prefer
    silhouette for a claim about groups and say that the elbow pointed elsewhere.
    """
    matrix = np.asarray(features, dtype=np.float64)
    rows: list[dict[str, float]] = []
    for k in k_range:
        if k < 2 or k >= len(matrix):
            continue
        result = kmeans(matrix, k, seed)
        rows.append({"k": float(k), "silhouette": result.silhouette, "inertia": result.inertia})
    if not rows:
        return KChoice()
    ks = [int(r["k"]) for r in rows]
    best = max(rows, key=lambda r: r["silhouette"])
    return KChoice(
        rows=rows,
        best_silhouette_k=int(best["k"]),
        elbow_k=_elbow(ks, [r["inertia"] for r in rows]),
    )


def gmm(
    features: np.ndarray, k: int, covariance: str = "full", seed: int | None = None
) -> GMMResult:
    """A Gaussian mixture over one row per node, returning hard and soft memberships.

    Use this when clusters may overlap, may differ in shape or spread, or when the useful answer
    is "this node is 70% group A and 30% group B". ``covariance`` is a modelling choice and has
    to be stated: ``full`` lets each component have its own shape and orientation and costs the
    most parameters, ``diag`` and ``spherical`` constrain it and need far fewer points. With few
    points per component a full covariance degenerates onto single points, which is why
    scikit-learn's regularisation is left at its default rather than turned off.
    """
    if covariance not in COVARIANCES:
        msg = f"covariance must be one of {', '.join(COVARIANCES)}, got {covariance!r}"
        raise ValueError(msg)
    matrix = np.asarray(features, dtype=np.float64)
    run_seed = _seeds(seed, 1)[0]
    model = GaussianMixture(
        n_components=k, covariance_type=covariance, n_init=5, random_state=run_seed
    )
    model.fit(matrix)
    labels = [int(label) for label in model.predict(matrix)]
    responsibilities = [[float(p) for p in row] for row in model.predict_proba(matrix)]
    return GMMResult(
        k=k,
        labels=labels,
        responsibilities=responsibilities,
        bic=float(model.bic(matrix)),
        aic=float(model.aic(matrix)),
        covariance=covariance,
        seed=run_seed,
        converged=bool(model.converged_),
    )


def choose_k_gmm(
    features: np.ndarray,
    k_range: Sequence[int],
    covariance: str = "full",
    seed: int | None = None,
) -> KChoice:
    """BIC and AIC for each candidate ``k``; the pick is the lowest BIC.

    BIC penalises parameters harder than AIC does, so it prefers simpler mixtures. It is the
    right default here because the alternative, adding components until the fit stops improving,
    always ends at "one component per node".
    """
    matrix = np.asarray(features, dtype=np.float64)
    rows: list[dict[str, float]] = []
    for k in k_range:
        if k < 1 or k > len(matrix):
            continue
        result = gmm(matrix, k, covariance, seed)
        rows.append({"k": float(k), "bic": result.bic, "aic": result.aic})
    if not rows:
        return KChoice()
    best = min(rows, key=lambda r: r["bic"])
    return KChoice(rows=rows, best_bic_k=int(best["k"]))


def rewired_modularity(
    rewired: nx.Graph, resolution: float = 1.0, seed: int | None = None
) -> float:
    """The best modularity Louvain finds on one degree-preserving rewiring: one null sample.

    Split out of :func:`null_model_modularity` so that a report which needs the same family of
    rewirings for something else can draw the family once and score each member here as it goes
    past. Drawing is the expensive half -- one edge swap per edge, per sample -- and two
    sections drawing their own families ask §19.1's question twice at twice the cost. The
    rewirings themselves cannot be kept in a list to share instead: fifty copies of a
    74,000-edge network is a gigabyte, so they are consumed one at a time.
    """
    found = nx.community.louvain_communities(
        rewired, weight="weight", resolution=resolution, seed=seed
    )
    return float(nx.community.modularity(rewired, found, weight="weight", resolution=resolution))


def null_model_modularity(
    graph: nx.Graph,
    communities: Sequence[Iterable[str]],
    samples: int = 50,
    seed: int | None = None,
    resolution: float = 1.0,
    scores: Sequence[float] | None = None,
) -> NullModelResult:
    """Score a partition against degree-preserving rewirings of the same graph.

    Any graph, including a purely random one, has some partition with positive modularity. The
    question is whether *this* modularity is higher than a graph with the same degree sequence
    could produce by chance. Each sample rewires the edges with ``nx.double_edge_swap``, which
    preserves every node's degree, reassigns the original weights at random over the rewired
    edges, and then *re-runs Louvain on the rewired graph*. The z-score is how many null
    standard deviations the observed modularity sits above that null mean; 2 is suggestive and
    3 is solid.

    The re-running matters and is easy to get wrong. Holding the observed partition fixed and
    scoring it on the rewired graphs measures how well a partition fitted to one graph
    transfers to another, which is always badly: that test calls a purely random graph
    "structured" with a z-score in the high single digits. Comparing best-achievable modularity
    to best-achievable modularity is the comparison that actually discriminates, and it costs
    one Louvain run per sample.

    The rewiring itself is :func:`graphrag.sna.null.configuration`, which is where §19.1's edge
    swap and everything it does *not* hold fixed are written down. Two things it does not: the
    number of connected components, and the clustering. So a modularity that beats this null has
    beaten the degree sequence, which is not the same as having beaten chance.

    A directed network is flattened first, because ``nx.double_edge_swap`` is defined only for
    undirected graphs and because the partition it is scoring was found on the flattened view
    anyway (§6.2). :attr:`LouvainResult.note` is where that is reported; saying it twice in one
    report would be noise.

    ``scores`` is for the caller who has already drawn this null family and scored it with
    :func:`rewired_modularity` -- the full report does, because its "against random" section
    (Atlas ch. 17) needs the same rewirings, at the same count and from the same seed. Passing
    them in skips the draw and only the observed modularity is computed here; passing ``None``
    draws and scores ``samples`` of them exactly as before. Either way the result is the same
    numbers, because the shared draw uses this function's own seeding order.
    """
    graph, _ = undirected_view(graph)
    partition = [sorted(c) for c in communities]
    observed = (
        float(nx.community.modularity(graph, partition, weight="weight", resolution=resolution))
        if partition and graph.number_of_edges()
        else 0.0
    )
    if not partition:
        return NullModelResult(observed, observed, 0.0, 0.0, 0)

    # Reproducibility, not secrecy: the same seed must produce the same null distribution. The
    # sampler draws from this same generator, so the whole test is one stream from one seed.
    rng = random.Random(seed)  # noqa: S311
    sampled = (
        list(scores)
        if scores is not None
        else [
            rewired_modularity(rewired, resolution, rng.randrange(1_000_000))
            for rewired in configuration(graph, samples, seed=rng)
        ]
    )
    if not sampled:
        return NullModelResult(observed, observed, 0.0, 0.0, 0)
    return NullModelResult(
        observed, mean(sampled), std(sampled), z_score(observed, sampled), len(sampled)
    )


# ==================================================================================================
# Chapter 37 -- Hierarchical community discovery: dendrograms with a cut chooser (§37.1), a
# Hierarchical Random Graph fit (§37.2), directed community discovery (§37.3, see
# ``directed_communities`` above) and the density-versus-hierarchy reading (§37.4, ``sna guide``).
# ==================================================================================================


def _mean_internal_density(graph: nx.Graph, partition: Sequence[Sequence[str]]) -> float:
    """The mean internal density (§12.1) across one partition's communities.

    §37.4's contrast needs a density number beside every modularity in a dendrogram: *"when we
    group everything in a community, there's some density even if modularity is zero... at the
    best partition we have agreement... but density is still high even with low modularity for
    a partition that puts together connected node pairs"* (Figure 37.11, p. 540-541). This is
    edges present over edges possible inside each community, unweighted across communities the
    way :func:`graphrag.sna.measures.density` reads one graph; a community of fewer than two
    nodes has nothing possible to be dense, and is skipped rather than counted as 0 or 1.
    """
    densities: list[float] = []
    for community in partition:
        members = list(community)
        n = len(members)
        if n < 2:
            continue
        possible = n * (n - 1) if graph.is_directed() else n * (n - 1) / 2
        internal = graph.subgraph(members).number_of_edges()
        densities.append(internal / possible if possible else 0.0)
    return float(np.mean(densities)) if densities else 0.0


@dataclass(frozen=True)
class DendrogramLevel:
    """One partition along a dendrogram, with the two numbers §37.4 asks be read together.

    ``modularity`` is standard modularity (§36.1) for this level's partition on the graph the
    dendrogram was built from. ``density`` is :func:`_mean_internal_density` for the same
    partition: the two profiles do not move together (§37.4), which is why both are kept on
    every level rather than only the one a cut was chosen by.
    """

    communities: list[list[str]]
    modularity: float
    density: float

    @property
    def size(self) -> int:
        """How many communities this level holds."""
        return len(self.communities)


@dataclass(frozen=True)
class Dendrogram:
    """A hierarchy of partitions, with the cut its own modularity profile picks (§37.1).

    Both the merging approach (:func:`louvain_levels`) and the splitting one
    (:func:`girvan_newman_dendrogram`) build one of these: a sequence of partitions from one
    community holding every node down to (at most) every node alone, each level scored by the
    modularity and density it would have on its own. *"Higher modularities are better
    partitions... the good cuts will appear as peaks in the modularity profile"* (p. 536), so
    ``cut`` is the level of highest modularity and ``peaks`` is every local maximum of that
    profile (:func:`_local_maxima`) -- *"multiple peaks are a clue of a hierarchical
    organization, because they identify good partitions with a very different number of
    communities"* (p. 536), which is a finding about the network and not only about where the
    single best cut happens to sit.
    """

    method: str
    levels: list[DendrogramLevel]
    cut: int
    peaks: list[int]
    resolution: float
    note: str = ""

    @property
    def communities(self) -> list[list[str]]:
        """The partition at the chosen cut, or ``[]`` for an empty dendrogram."""
        return self.levels[self.cut].communities if self.levels else []

    @property
    def modularity(self) -> float:
        return self.levels[self.cut].modularity if self.levels else 0.0

    @property
    def density(self) -> float:
        return self.levels[self.cut].density if self.levels else 0.0


def _local_maxima(values: Sequence[float]) -> list[int]:
    """Indices of the peaks in a profile (p. 536): a value at least as high as both neighbours
    and strictly higher than at least one of them, so a run of equal values counts once, at its
    first index, rather than once per member. A profile of one value is trivially its own peak."""
    n = len(values)
    if n == 0:
        return []
    peaks: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        left_ok = i == 0 or values[i] > values[i - 1]
        right_ok = j == n - 1 or values[i] > values[j + 1]
        if left_ok and right_ok:
            peaks.append(i)
        i = j + 1
    return peaks


def score_level(
    graph: nx.Graph, partition: Sequence[Iterable[str]], resolution: float = 1.0
) -> DendrogramLevel:
    """Modularity and mean internal density for one partition (§37.4's pairing, Figure 37.11).

    This is the scorer :func:`louvain_levels` and :func:`girvan_newman_dendrogram` run over
    every level of their own dendrograms; exposed on its own so that any partition -- one level
    plucked out of a dendrogram, or a partition that never came from one at all -- can be read
    the same way, the pairing §37.4 asks for rather than modularity alone.
    """
    ordered = sorted((sorted(c) for c in partition), key=lambda c: (-len(c), c[0]))
    modularity = (
        float(nx.community.modularity(graph, ordered, weight="weight", resolution=resolution))
        if ordered and graph.number_of_edges()
        else 0.0
    )
    return DendrogramLevel(ordered, modularity, _mean_internal_density(graph, ordered))


def _dendrogram(
    method: str,
    graph: nx.Graph,
    partitions: Sequence[Sequence[Iterable[str]]],
    resolution: float,
    note: str,
) -> Dendrogram:
    """Score every level of a partition sequence and pick the peak-modularity cut."""
    levels = [score_level(graph, partition, resolution) for partition in partitions]
    modularities = [level.modularity for level in levels]
    peaks = _local_maxima(modularities)
    cut = int(np.argmax(modularities)) if modularities else 0
    return Dendrogram(
        method=method, levels=levels, cut=cut, peaks=peaks, resolution=resolution, note=note
    )


def louvain_levels(graph: nx.Graph, resolution: float = 1.0, seed: int | None = None) -> Dendrogram:
    """The Louvain dendrogram itself (§37.1 "Merging"), not just the partition it settles on.

    :func:`louvain` returns the best of several seeded runs at the algorithm's own stopping
    point. This runs ``networkx``'s ``louvain_partitions`` once instead, which yields every
    intermediate level the merging phases pass through -- from the first pass's small,
    modularity-driven merges up to the coarsest partition ``louvain()`` would have returned --
    and scores each one, so a caller can see whether the algorithm's own stopping point is the
    only good cut or one of several (§37.4's multiple peaks).

    A directed network is flattened first, for the reason :func:`louvain` flattens it (§6.2):
    the merge step and the modularity that drives it are both defined on undirected graphs.
    """
    if graph.number_of_nodes() == 0:
        return Dendrogram("louvain", [], 0, [], resolution, "")
    flat, flattened = undirected_view(graph)
    note = f"Louvain and its dendrogram ran {flattened}." if flattened else ""
    if flat.number_of_edges() == 0:
        # ``louvain_partitions`` scores its own levels with ``nx.community.modularity``, which
        # divides by ``2 * |E|`` and raises on a graph with nodes and no edges. There is exactly
        # one level to report in that case -- every node alone, modularity 0 -- so it is built
        # directly rather than asking networkx for a merge sequence that has nothing to merge.
        partitions: list[list[list[str]]] = [[[node] for node in flat.nodes]]
        return _dendrogram("louvain", flat, partitions, resolution, note)
    partitions = list(
        nx.community.louvain_partitions(flat, weight="weight", resolution=resolution, seed=seed)
    )
    return _dendrogram("louvain", flat, partitions, resolution, note)


#: Above this many nodes, :func:`girvan_newman_dendrogram` refuses rather than pay the chapter's
#: own cost estimate of O(|V||E|^2): edge betweenness is recomputed after every edge it removes,
#: which the chapter calls unusable "on anything but trivially small networks" (p. 535).
GIRVAN_NEWMAN_MAX_NODES: int = 300


def girvan_newman_dendrogram(
    graph: nx.Graph,
    resolution: float = 1.0,
    max_nodes: int = GIRVAN_NEWMAN_MAX_NODES,
) -> Dendrogram:
    """The splitting dendrogram of §37.1, cut by the peak of its own modularity profile.

    The first level is the whole network as one community -- modularity 0 by construction
    (Figure 37.11(a)) -- and then ``networkx``'s ``girvan_newman`` removes the edge of highest
    betweenness one at a time until every node is alone, one level per split that changes the
    number of components. §37.1 is explicit that Girvan-Newman itself does not stop at a good
    cut: *"the algorithm will normally perform all the possible splits and returns you the full
    structure, rather than the cut that maximizes modularity. Thus you will have to calculate
    the modularity of each split yourself"* (p. 536) -- which is what :func:`_dendrogram` does
    here, the same way :func:`louvain_levels` does for the merging approach.

    Above ``max_nodes`` this raises rather than pay the O(|V||E|^2) the chapter warns about
    (p. 535): sample the network first (ch. 29, ``sna sample``) or take a backbone (ch. 27,
    ``sna backbone``) and run this on that instead. A directed network is flattened first, for
    the same reason as the modularity it is cut by (§6.2): edge betweenness here is undirected.
    """
    if graph.number_of_nodes() == 0:
        return Dendrogram("girvan-newman", [], 0, [], resolution, "")
    if graph.number_of_nodes() > max_nodes:
        msg = (
            f"girvan_newman_dendrogram() refuses a network of {graph.number_of_nodes():,} nodes: "
            f"above {max_nodes:,} the repeated edge-betweenness recomputation this method needs "
            'is the complexity the chapter calls unusable outside "trivially small networks" '
            "(§37.1, p. 535). Sample the network first (ch. 29: sna sample) or take a backbone "
            "(ch. 27: sna backbone) and run this on that instead."
        )
        raise ValueError(msg)
    flat, flattened = undirected_view(graph)
    note = f"Girvan-Newman and its modularity profile ran {flattened}." if flattened else ""
    if flat.number_of_edges() == 0:
        partitions: list[list[list[str]]] = [[[node] for node in flat.nodes]]
    else:
        partitions = [[list(flat.nodes)]]
        for split in nx.community.girvan_newman(flat):
            partitions.append([list(community) for community in split])
    return _dendrogram("girvan-newman", flat, partitions, resolution, note)


# --------------------------------------------------------------------------------- §37.2 HRG


def _hrg_term(edges: int, possible: int) -> float:
    """One internal node's contribution to the dendrogram's log-likelihood (§37.2).

    ``possible`` is ``L_r * R_r``, the pairs a leaf on one side of internal node ``r`` could form
    with a leaf on the other, and ``edges`` is how many of those pairs the graph actually joins;
    the maximum-likelihood ``p_r = edges / possible`` turns the log-likelihood into
    ``edges * log(p_r) + (possible - edges) * log(1 - p_r)``, which is 0 -- not ``nan`` -- when
    ``p_r`` is 0 or 1, since both of those limits are what a Bernoulli model assigns no surprise
    to a certain outcome. 0.0 when ``possible`` is 0, which cannot happen on the actual dendrogram
    (every internal node has at least one leaf on each side) but can while it is being built.
    """
    if possible <= 0 or edges <= 0 or edges >= possible:
        return 0.0
    p = edges / possible
    return edges * math.log(p) + (possible - edges) * math.log(1 - p)


def _count_edges(adjacency_matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> int:
    """How many of the graph's edges run between the leaves ``left`` marks and the ones ``right``
    marks, from the boolean adjacency matrix built once in :func:`hrg_fit`."""
    return int(adjacency_matrix[np.ix_(left, right)].sum())


@dataclass
class _HRGState:
    """A binary dendrogram over ``n`` leaves, mutated in place by :func:`_hrg_step`.

    Node ids ``0..n-1`` are the leaves, indexing into the node order :func:`hrg_fit` built this
    state from; ids ``n..2n-2`` are the ``n - 1`` internal nodes a full binary tree over ``n``
    leaves needs (§37.2, p. 537). ``mask[node]`` is which leaves sit under ``node``, as a
    boolean array over leaf indices, so the edges between two subtrees are one adjacency lookup
    (:func:`_count_edges`) rather than a rebuilt set.
    """

    adjacency_matrix: np.ndarray
    children: dict[int, tuple[int, int]]
    parent: dict[int, int]
    mask: dict[int, np.ndarray]
    edges: dict[int, int]
    counts: dict[int, tuple[int, int]]
    log_likelihood: float
    root: int


def _initial_tree(n: int, adjacency_matrix: np.ndarray, rng: random.Random) -> _HRGState:
    """A uniformly random full binary tree over ``n`` leaves, scored under §37.2's model.

    Every node is its own community at the start -- the merge order here carries no meaning and
    is only a starting point for :func:`_hrg_step` to rearrange -- built by repeatedly picking
    two random current subtrees and joining them under a new internal node, the standard way to
    draw an unbiased random binary tree topology.
    """
    children: dict[int, tuple[int, int]] = {}
    parent: dict[int, int] = {}
    mask: dict[int, np.ndarray] = {}
    for leaf in range(n):
        single = np.zeros(n, dtype=bool)
        single[leaf] = True
        mask[leaf] = single
    active = list(range(n))
    rng.shuffle(active)
    next_id = n
    while len(active) > 1:
        left = active.pop(rng.randrange(len(active)))
        right = active.pop(rng.randrange(len(active)))
        node_id = next_id
        next_id += 1
        children[node_id] = (left, right)
        parent[left] = node_id
        parent[right] = node_id
        mask[node_id] = mask[left] | mask[right]
        active.append(node_id)
    root = active[0]
    parent[root] = -1
    edges: dict[int, int] = {}
    counts: dict[int, tuple[int, int]] = {}
    log_likelihood = 0.0
    for node_id, (left, right) in children.items():
        e = _count_edges(adjacency_matrix, mask[left], mask[right])
        lr = (int(mask[left].sum()), int(mask[right].sum()))
        edges[node_id] = e
        counts[node_id] = lr
        log_likelihood += _hrg_term(e, lr[0] * lr[1])
    return _HRGState(
        adjacency_matrix=adjacency_matrix,
        children=children,
        parent=parent,
        mask=mask,
        edges=edges,
        counts=counts,
        log_likelihood=log_likelihood,
        root=root,
    )


def _hrg_step(state: _HRGState, rng: random.Random) -> None:
    """One Metropolis move over dendrogram space (§37.2, Clauset, Moore and Newman 2008).

    Pick a random internal node ``c`` with a parent ``p``; ``p``'s other child is ``c``'s
    sibling ``s`` and ``c``'s own two children are ``c1`` and ``c2``. The three ways to arrange
    ``{s, c1, c2}`` into that same two-internal-node shape are ``p=(s,(c1,c2))`` (the current
    one), ``p=(c1,(s,c2))`` and ``p=(c2,(s,c1))`` -- *"the resulting dendrogram would have been
    equally likely"* whichever of these is current (p. 538) -- so proposing one of the other two
    uniformly is its own reverse move with the same probability, and the acceptance ratio needs
    only the likelihoods either side of it. Only ``p`` and ``c`` change; every other internal
    node's leaf set is untouched, since the total leaf set under ``p`` is the same in every
    arrangement.
    """
    internal = [node for node in state.children if state.parent[node] != -1]
    if not internal:
        return
    c = rng.choice(internal)
    p = state.parent[c]
    c1, c2 = state.children[c]
    p_left, p_right = state.children[p]
    s = p_left if p_right == c else p_right
    moved_up, stays = (c1, c2) if rng.randrange(2) == 0 else (c2, c1)

    old = _hrg_term(state.edges[p], state.counts[p][0] * state.counts[p][1]) + _hrg_term(
        state.edges[c], state.counts[c][0] * state.counts[c][1]
    )

    new_c_mask = state.mask[s] | state.mask[stays]
    new_c_edges = _count_edges(state.adjacency_matrix, state.mask[s], state.mask[stays])
    new_c_counts = (int(state.mask[s].sum()), int(state.mask[stays].sum()))
    new_p_edges = _count_edges(state.adjacency_matrix, state.mask[moved_up], new_c_mask)
    new_p_counts = (int(state.mask[moved_up].sum()), int(new_c_mask.sum()))

    new = _hrg_term(new_p_edges, new_p_counts[0] * new_p_counts[1]) + _hrg_term(
        new_c_edges, new_c_counts[0] * new_c_counts[1]
    )
    delta = new - old
    if delta < 0 and rng.random() >= math.exp(delta):
        return  # reject: state is unchanged

    state.children[c] = (s, stays)
    state.children[p] = (moved_up, c)
    state.parent[s] = c
    state.parent[moved_up] = p
    state.mask[c] = new_c_mask
    state.edges[c] = new_c_edges
    state.counts[c] = new_c_counts
    state.edges[p] = new_p_edges
    state.counts[p] = new_p_counts
    state.log_likelihood += delta


@dataclass(frozen=True)
class HRGFit:
    """A Hierarchical Random Graph fitted to one network (§37.2): the best dendrogram an MCMC
    search over dendrogram space found, with the ``p_r`` probability its model assigns each
    internal node's two branches.

    ``order`` is leaf id to node name; ``children[node]`` is ``(left, right)`` for every internal
    node, where ids ``>= len(order)`` are internal and ids ``< len(order)`` index into ``order``.
    :func:`hrg_communities` is the reading of this structure into a flat partition; nothing here
    requires that reading, since the dendrogram itself -- which nodes share which internal
    ancestor, and how associative that ancestor is -- is the object §37.2 is fitting.
    """

    order: tuple[str, ...]
    children: dict[int, tuple[int, int]]
    probabilities: dict[int, float]
    counts: dict[int, tuple[int, int]]
    root: int
    log_likelihood: float
    samples: int
    restarts: int
    seed: int


def _hrg_probabilities(
    children: dict[int, tuple[int, int]], counts: dict[int, tuple[int, int]], edges: dict[int, int]
) -> dict[int, float]:
    return {
        node: (
            edges[node] / (counts[node][0] * counts[node][1])
            if counts[node][0] * counts[node][1]
            else 0.0
        )
        for node in children
    }


def hrg_fit(
    graph: nx.Graph,
    samples: int | None = None,
    restarts: int = 5,
    seed: int | None = None,
) -> HRGFit:
    """Fit a Hierarchical Random Graph to ``graph`` by Markov chain Monte Carlo (§37.2).

    *"Finding the dendrogram that most likely fits the data is just a choice of how you want to
    explore the space of all possible dendrograms. In the original paper, authors use a Markov
    chain Monte Carlo method, where each dendrogram is sampled proportionally to its likelihood
    value"* (p. 538). This runs that chain from ``restarts`` independent random starting trees
    (:func:`_initial_tree`) for ``samples`` steps each -- like :func:`louvain`, a randomised
    search is run more than once and the best is kept, here by log-likelihood rather than
    modularity, because one random starting tree can settle into a shallow local optimum the
    same search from another start escapes.

    ``samples`` defaults to ``max(2000, 300 * (n - 1))``, scaling with the number of internal
    nodes a full binary tree over ``n`` leaves has to place. A directed network is flattened
    first (§6.2): the model of p. 537 is a probability of connection between two leaves, which
    has no direction. Raises on fewer than two nodes, where no internal node exists to fit.
    """
    flat, _ = undirected_view(graph)
    nodes = sorted(flat.nodes, key=str)
    n = len(nodes)
    if n < 2:
        msg = f"hrg_fit() needs at least two nodes to fit a dendrogram to, got {n}"
        raise ValueError(msg)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency_matrix = np.zeros((n, n), dtype=bool)
    for u, v in flat.edges():
        if u != v:
            adjacency_matrix[index[u], index[v]] = True
            adjacency_matrix[index[v], index[u]] = True
    steps = samples if samples is not None else max(2000, 300 * (n - 1))
    run_seed = _seeds(seed, 1)[0]
    rng = random.Random(run_seed)  # noqa: S311

    best_children: dict[int, tuple[int, int]] = {}
    best_counts: dict[int, tuple[int, int]] = {}
    best_edges: dict[int, int] = {}
    best_root = n
    best_ll = float("-inf")
    for _ in range(max(restarts, 1)):
        state = _initial_tree(n, adjacency_matrix, rng)
        if state.log_likelihood > best_ll:
            best_ll = state.log_likelihood
            best_children, best_edges, best_counts, best_root = (
                dict(state.children),
                dict(state.edges),
                dict(state.counts),
                state.root,
            )
        for _ in range(steps):
            _hrg_step(state, rng)
            if state.log_likelihood > best_ll:
                best_ll = state.log_likelihood
                best_children, best_edges, best_counts, best_root = (
                    dict(state.children),
                    dict(state.edges),
                    dict(state.counts),
                    state.root,
                )
    return HRGFit(
        order=tuple(nodes),
        children=best_children,
        probabilities=_hrg_probabilities(best_children, best_counts, best_edges),
        counts=best_counts,
        root=best_root,
        log_likelihood=best_ll,
        samples=steps,
        restarts=max(restarts, 1),
        seed=run_seed,
    )


def _hrg_leaves(fit: HRGFit, node: int) -> list[str]:
    if node not in fit.children:
        return [fit.order[node]]
    left, right = fit.children[node]
    return _hrg_leaves(fit, left) + _hrg_leaves(fit, right)


def hrg_communities(fit: HRGFit, threshold: float | None = None) -> list[list[str]]:
    """Read an assortative partition off a fitted dendrogram (§37.2, tied to §12.1's density).

    Descends from the root and cuts at the first internal node on each path whose ``p_r``
    exceeds ``threshold``: that branch connects its members more densely than the threshold
    asks, and nothing above it in the tree needs to be checked once it has qualified. A leaf
    that is never inside such a branch is returned alone.

    ``threshold`` defaults to the root's own ``p_r``, which is exactly the graph's own density --
    the root's two branches between them span every leaf, so its ``edges / possible`` is the
    graph's ``possible`` and ``edges`` too -- making the default cut "denser than the network as
    a whole", the reading §37.4 puts beside modularity. Passing a higher ``threshold`` reads a
    deeper, more exacting level of the same fitted hierarchy instead of refitting it, since nested
    branches of a dendrogram are exactly nested readings of the same tree.
    """
    baseline = fit.probabilities[fit.root] if threshold is None else threshold
    communities: list[list[str]] = []

    def descend(node: int) -> None:
        if node not in fit.children:
            communities.append([fit.order[node]])
            return
        if fit.probabilities[node] > baseline:
            communities.append(sorted(_hrg_leaves(fit, node)))
            return
        left, right = fit.children[node]
        descend(left)
        descend(right)

    descend(fit.root)
    return communities


# ==================================================================================================
# Chapter 35 -- Graph partitions: the historical routes to a community, before chapter 36 asks how
# to score one and chapter 37 asks whether one scale is enough. §35.1 is the stochastic blockmodel
# (:func:`sbm_communities`), §35.2 the two random-walk readings (:func:`infomap_communities`,
# :func:`walktrap`), §35.3 label percolation (:func:`label_propagation`), §35.4 tracking a
# partition across snapshots (:func:`match_communities`) and §35.5 growing one from a single seed
# (:func:`local_community`).
# ==================================================================================================


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _communities_from_labels(order: Sequence[str], labels: Sequence[int]) -> list[list[str]]:
    buckets: dict[int, list[str]] = {}
    for node, label in zip(order, labels, strict=True):
        buckets.setdefault(int(label), []).append(node)
    return sorted((sorted(members) for members in buckets.values()), key=lambda c: (-len(c), c[0]))


# -------------------------------------------------------------------- §35.3 label percolation


@dataclass(frozen=True)
class LabelPropagationResult:
    """A label-percolation partition (§35.3), run several times because the algorithm is not
    deterministic: ties among a node's most common neighbour labels are broken at random."""

    communities: list[list[str]]
    seeds: list[int]
    #: Mean pairwise adjusted Rand index across the runs, read exactly as :attr:`LouvainResult.
    #: stability` is: below about 0.6 the boundaries are the tie-breaking, not the network.
    stability: float
    note: str = ""


def label_propagation(
    graph: nx.Graph, seed: int | None = None, runs: int = 10
) -> LabelPropagationResult:
    """Asynchronous label percolation (§35.3): each node repeatedly adopts the label most common
    among its neighbours, until nothing changes.

    *"We start with a network whose node labels are scattered randomly. Then each node looks at
    its neighbors and adopts the most common labels it sees ... labels will percolate through the
    network until we reach a state in which no more significant changes can happen"* (p. 499).
    This is ``networkx``'s own ``asyn_lpa_communities``: the *asynchronous* variant the chapter
    calls "the original formulation of the label propagation principle" (p. 500) -- at step ``i``
    a node uses whatever label its neighbours hold *right now*, some already updated this step and
    some not, rather than the semi-synchronous or synchronous alternatives the chapter also names.

    Non-determinism is the point rather than a bug (p. 500): tied labels are broken at random, so
    this runs ``runs`` times from ``runs`` seeds -- exactly as :func:`louvain` does -- and reports
    the mean pairwise adjusted Rand index as ``stability``. Unlike Louvain there is no modularity
    this algorithm is optimising, so the run kept as ``communities`` is the one of highest
    modularity among the ``runs`` found, which is a reading applied after the fact rather than
    what the algorithm itself is climbing.

    A directed network is flattened first (§6.2): "most common label among my neighbours" has no
    direction to it. An edgeless graph returns every node alone, for the same reason
    :func:`louvain` does -- there is no neighbour whose label to adopt.
    """
    if graph.number_of_nodes() == 0:
        return LabelPropagationResult([], [], 1.0, "")
    flat, flattened = undirected_view(graph)
    note = f"Label propagation ran {flattened}." if flattened else ""
    seeds = _seeds(seed, runs)
    if flat.number_of_edges() == 0:
        communities = [[node] for node in sorted(flat.nodes, key=str)]
        return LabelPropagationResult(communities, seeds, 1.0, note)
    nodes = list(flat.nodes)
    partitions = [
        sorted(
            (sorted(c) for c in nx.community.asyn_lpa_communities(flat, weight="weight", seed=s)),
            key=lambda c: (-len(c), c[0]),
        )
        for s in seeds
    ]
    vectors = [_label_vector(p, nodes) for p in partitions]
    pairs = [
        float(adjusted_rand_score(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    modularities = [float(nx.community.modularity(flat, p, weight="weight")) for p in partitions]
    best = int(np.argmax(modularities))
    return LabelPropagationResult(
        communities=partitions[best],
        seeds=seeds,
        stability=float(np.mean(pairs)) if pairs else 1.0,
        note=note,
    )


# --------------------------------------------------------------------------- §35.2 random walks


def _entropy(probabilities: np.ndarray, total: float) -> float:
    """Shannon entropy in bits of ``probabilities / total``, 0.0 where ``total`` is 0 (nothing to
    describe) rather than a division by zero. A term of 0 contributes 0 (the ``x log x -> 0``
    limit)."""
    if total <= 0:
        return 0.0
    terms = probabilities[probabilities > 0] / total
    return float(-np.sum(terms * np.log2(terms)))


def map_equation(
    graph: nx.Graph,
    communities: Sequence[Iterable[str]],
    _stationary: tuple[np.ndarray, list[str]] | None = None,
) -> float:
    """The map equation in bits per step (§35.2, Rosvall and Bergstrom 2008): how expensive
    ``communities`` makes it to describe a random walk on ``graph``.

    ``L(M) = q_curl * H(Q) + sum_i p_circle_i * H(P^i)`` (p. 497-499): ``pi`` is the walk's
    stationary distribution (:func:`graphrag.sna.walks.stationary_distribution`, closed-form
    degree share on an undirected graph, §11.1); ``q_i`` is module ``i``'s exit probability, the
    share of a step that crosses its boundary (the module's cut weight over twice the network's
    total weight); ``p_circle_i = q_i + sum of pi over i's members`` is the module's whole
    codebook usage; ``q_curl = sum_i q_i``. ``H(Q)`` is the entropy of the normalised exit
    probabilities across modules (the *index codebook*, naming which module a walker enters) and
    ``H(P^i)`` is the entropy of module ``i``'s own codebook (its members' visit rates plus its
    own exit code). Lower is a better description; the whole-network partition and the
    all-singleton partition are both legal inputs, at the two extremes p. 497-499 walks through.

    0.0 for an edgeless graph (there is no walk to describe) and for the all-in-one partition on
    a connected graph (nothing ever exits, so both terms vanish). A directed graph is flattened
    first (§6.2): the stationary distribution and the exit flow this package computes are both
    defined on the undirected walk every other random-walk quantity here uses.

    ``_stationary`` is an internal hook: :func:`infomap_communities`'s search evaluates this
    thousands of times over the same graph and passes its own ``(pi, order)`` in rather than
    have every candidate move recompute it.
    """
    flat, _ = undirected_view(graph)
    if flat.number_of_nodes() == 0 or flat.number_of_edges() == 0:
        return 0.0
    pi, order = _stationary if _stationary is not None else stationary_distribution(flat)
    index = {node: i for i, node in enumerate(order)}
    partition = [sorted(set(c) & set(order)) for c in communities]
    partition = [c for c in partition if c]
    if not partition:
        return 0.0
    membership = {node: m for m, members in enumerate(partition) for node in members}
    k = len(partition)
    cuts = np.zeros(k, dtype=np.float64)
    total_weight = 0.0
    for u, v, data in flat.edges(data=True):
        weight = float(data.get("weight", 1.0))
        total_weight += weight
        cu, cv = membership.get(u), membership.get(v)
        if cu is not None and cv is not None and cu != cv:
            cuts[cu] += weight
            cuts[cv] += weight
    two_m = 2.0 * total_weight
    q = cuts / two_m if two_m else np.zeros(k)
    p_module = np.array([sum(pi[index[n]] for n in members) for members in partition])
    p_circle = q + p_module
    q_curl = float(q.sum())
    h_q = _entropy(q, q_curl)
    module_terms = 0.0
    for i, members in enumerate(partition):
        member_pi = np.array([pi[index[n]] for n in members])
        within = np.concatenate(([q[i]], member_pi))
        module_terms += p_circle[i] * _entropy(within, p_circle[i])
    return q_curl * h_q + module_terms


@dataclass(frozen=True)
class InfomapResult:
    """The map equation's own search (§35.2): the lowest-code-length partition found, and the
    evidence for whether that search settled or is wandering."""

    communities: list[list[str]]
    code_length: float
    code_lengths: list[float]
    seeds: list[int]
    #: Mean pairwise adjusted Rand index across the runs, read as :attr:`LouvainResult.stability`.
    stability: float
    sweeps: int
    note: str = ""


#: Above this many nodes, :func:`infomap_communities` refuses. The node-moving search here
#: recomputes :func:`map_equation` -- an O(n + m) pass -- for every candidate move of every node
#: in every sweep, which is the cost of a single-level search with no multilevel aggregation on
#: top of it (the note below the docstring says so); past this size a report should sample (ch.
#: 29) or backbone (ch. 27) first.
INFOMAP_MAX_NODES = 200

#: Sweeps (one pass over every node) before :func:`infomap_communities` gives up on a seed even
#: if it is still moving nodes -- generous, since a sweep that moves nothing stops early anyway.
INFOMAP_MAX_SWEEPS = 100


def _infomap_search(
    graph: nx.Graph, run_seed: int, max_sweeps: int
) -> tuple[list[list[str]], float, int]:
    """One node-moving search minimising :func:`map_equation`, from every node in its own module.

    This is Rosvall and Bergstrom's own greedy heuristic *without* the multilevel aggregation
    real Infomap implementations add on top (repeatedly collapsing the found modules into
    super-nodes and re-running the search on those) -- a simplification stated here and in the
    ticket's report, made because that aggregation buys speed on large networks rather than a
    different answer on the small ones this package's known-answer tests use.
    """
    rng = random.Random(run_seed)  # noqa: S311 -- reproducibility, not secrecy
    pi, order = stationary_distribution(graph)
    membership = {node: i for i, node in enumerate(order)}
    neighbours = {node: list(graph.neighbors(node)) for node in order}
    sweep = 0
    for sweep in range(1, max_sweeps + 1):  # noqa: B007 -- the final count is the return value
        visiting = order[:]
        rng.shuffle(visiting)
        moved = False
        for node in visiting:
            current = membership[node]
            candidates = {membership[nb] for nb in neighbours[node]} | {current}
            best_module, best_length = current, None
            for candidate in candidates:
                trial = dict(membership)
                trial[node] = candidate
                communities = _communities_from_labels(order, [trial[n] for n in order])
                length = map_equation(graph, communities, _stationary=(pi, order))
                if best_length is None or length < best_length - 1e-12:
                    best_length, best_module = length, candidate
            if best_module != current:
                membership[node] = best_module
                moved = True
        if not moved:
            break
    communities = _communities_from_labels(order, [membership[n] for n in order])
    length = map_equation(graph, communities, _stationary=(pi, order))
    return communities, length, sweep


def infomap_communities(
    graph: nx.Graph,
    seed: int | None = None,
    runs: int = 5,
    max_sweeps: int = INFOMAP_MAX_SWEEPS,
    max_nodes: int = INFOMAP_MAX_NODES,
) -> InfomapResult:
    """The map equation approach (§35.2): find the partition that best compresses a random walk.

    Runs :func:`_infomap_search` from ``runs`` random node orderings -- itself non-deterministic,
    as the chapter says every random-walk-based method is (p. 498) -- and keeps the lowest code
    length, reporting the mean pairwise adjusted Rand index across runs as ``stability`` exactly
    as :func:`louvain` does. Raises above ``max_nodes``: see :data:`INFOMAP_MAX_NODES`.

    A directed network is flattened first (§6.2), and an edgeless graph returns every node alone
    at a code length of 0.0 -- there is no walk to describe either way.
    """
    if graph.number_of_nodes() == 0:
        return InfomapResult([], 0.0, [], [], 1.0, 0, "")
    flat, flattened = undirected_view(graph)
    note = f"Infomap and its map equation ran {flattened}." if flattened else ""
    if flat.number_of_nodes() > max_nodes:
        msg = (
            f"infomap_communities() refuses a network of {flat.number_of_nodes():,} nodes: this "
            f"single-level search recomputes the map equation for every candidate move, which is "
            f"unusable above {max_nodes:,} nodes (§35.2). Sample the network first (ch. 29: sna "
            "sample) or take a backbone (ch. 27: sna backbone) and run this on that instead."
        )
        raise ValueError(msg)
    seeds = _seeds(seed, runs)
    if flat.number_of_edges() == 0:
        communities = [[node] for node in sorted(flat.nodes, key=str)]
        return InfomapResult(communities, 0.0, [0.0] * len(seeds), seeds, 1.0, 0, note)
    runs_out = [_infomap_search(flat, s, max_sweeps) for s in seeds]
    partitions = [r[0] for r in runs_out]
    lengths = [r[1] for r in runs_out]
    sweeps = max(r[2] for r in runs_out)
    nodes = list(flat.nodes)
    vectors = [_label_vector(p, nodes) for p in partitions]
    pairs = [
        float(adjusted_rand_score(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    best = int(np.argmin(lengths))
    return InfomapResult(
        communities=partitions[best],
        code_length=lengths[best],
        code_lengths=lengths,
        seeds=seeds,
        stability=float(np.mean(pairs)) if pairs else 1.0,
        sweeps=sweeps,
        note=note,
    )


#: Above this many nodes, :func:`walktrap` refuses: it builds a dense n x n distance matrix and
#: takes a matrix power of a dense n x n transition matrix, both O(n^2) or worse in memory or time.
WALKTRAP_MAX_NODES = 2000


def walktrap_distance(graph: nx.Graph, steps: int = 4) -> tuple[np.ndarray, list[str]]:
    """Pons and Latapy's random-walk distance (§35.2, p. 497's family of walk-based methods):
    ``r_ij = sqrt(sum_k (P^t_ik - P^t_jk)^2 / d_k)``, and the node order the matrix is in.

    Two nodes are close when a walker starting at either one ends up, after ``t`` steps, with
    about the same chance of standing on every other node ``k`` -- weighted by ``1 / d_k`` so
    that agreeing about a low-degree node (a stronger coincidence) counts for more than agreeing
    about a hub. ``t`` trades off locality against noise: too small and only immediate neighbours
    look alike, too large and the distance washes out towards every node's stationary share
    regardless of community (§11.1's "two nodes -> same distribution" limit). ``t = 4`` is Pons
    and Latapy's own reported default.

    A directed network is flattened first (§6.2): ``P`` here is :func:`graphrag.sna.matrices.
    stochastic`'s row-normalised transition matrix, defined the same way as every other
    random-walk quantity in this package, none of which read direction.
    """
    flat, _ = undirected_view(graph)
    order = list(flat.nodes)
    p, _ = stochastic(flat, nodes=order, orientation="row")
    p_t = np.linalg.matrix_power(dense(p), max(steps, 1))
    degrees = np.array([flat.degree(n, weight="weight") for n in order], dtype=np.float64)
    inverse_degree = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    diff = p_t[:, None, :] - p_t[None, :, :]
    distance_sq = np.einsum("ijk,k->ij", diff * diff, inverse_degree)
    return np.sqrt(np.maximum(distance_sq, 0.0)), order


def _linkage_partitions(z: np.ndarray, leaves: Sequence[str]) -> list[list[list[str]]]:
    """Every partition a scipy linkage matrix passes through, from ``n`` singletons (index 0) to
    one cluster holding everyone (the last index) -- the same shape :func:`louvain_levels` and
    :func:`girvan_newman_dendrogram` hand :func:`_dendrogram`, so it scores this one identically."""
    clusters: dict[int, list[str]] = {i: [leaf] for i, leaf in enumerate(leaves)}
    partitions = [sorted((sorted(c) for c in clusters.values()), key=lambda c: (-len(c), c[0]))]
    next_id = len(leaves)
    for left, right, *_rest in z:
        merged = clusters.pop(int(left)) + clusters.pop(int(right))
        clusters[next_id] = merged
        next_id += 1
        partitions.append(
            sorted((sorted(c) for c in clusters.values()), key=lambda c: (-len(c), c[0]))
        )
    return partitions


def walktrap(
    graph: nx.Graph,
    steps: int = 4,
    resolution: float = 1.0,
    max_nodes: int = WALKTRAP_MAX_NODES,
) -> Dendrogram:
    """Walktrap (§35.2, Pons and Latapy 2006): agglomerate on the random-walk distance, cut by
    the peak of the merge sequence's own modularity profile -- the same cut rule §37.1 uses.

    Builds :func:`walktrap_distance` once, feeds it to ``scipy``'s Ward linkage (minimum-variance
    agglomeration, the merge rule the original paper uses), and scores every merge exactly as
    :func:`louvain_levels` scores every split of its own dendrogram, via :func:`_dendrogram`.
    Ward's method is defined for points in a Euclidean space; handed a distance matrix instead,
    ``scipy`` applies the equivalent Lance-Williams recursion, which is the standard way every
    distance-based Walktrap implementation (this package included) runs it.

    Raises above ``max_nodes`` (dense n x n distance matrix and a matrix power of a dense n x n
    transition matrix); flattens a directed graph first (§6.2), for the reason
    :func:`walktrap_distance` does.
    """
    if graph.number_of_nodes() == 0:
        return Dendrogram("walktrap", [], 0, [], resolution, "")
    flat, flattened = undirected_view(graph)
    note = f"Walktrap and its random-walk distance ran {flattened}." if flattened else ""
    if flat.number_of_nodes() > max_nodes:
        msg = (
            f"walktrap() refuses a network of {flat.number_of_nodes():,} nodes: it holds a dense "
            f"n x n distance matrix, unusable above {max_nodes:,} nodes. Sample the network first "
            "(ch. 29: sna sample) or take a backbone (ch. 27: sna backbone) and run this on that "
            "instead."
        )
        raise ValueError(msg)
    if flat.number_of_edges() == 0 or flat.number_of_nodes() < 2:
        partitions: list[list[list[str]]] = [[[node] for node in flat.nodes]]
        return _dendrogram("walktrap", flat, partitions, resolution, note)
    distance, order = walktrap_distance(flat, steps)
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method="ward")
    partitions = _linkage_partitions(z, order)
    return _dendrogram("walktrap", flat, partitions, resolution, note)


# --------------------------------------------------------------------------------- §35.1 SBM


def _one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    onehot = np.zeros((len(labels), k), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    return onehot


def _block_weights(matrix: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``e[r, s]``: total weight between block ``r`` and ``s`` from ``matrix`` (a diagonal entry
    counts an internal edge twice, since the symmetric matrix sums both its (i, j) and (j, i)),
    and ``sizes[r]``: how many nodes are in block ``r``. Works for a weighted or a 0/1 matrix."""
    onehot = _one_hot(labels, k)
    e = onehot.T @ matrix @ onehot
    sizes = onehot.sum(axis=0)
    return e, sizes


def _plain_score(e: np.ndarray, sizes: np.ndarray, k: int) -> float:
    """The book's own cartoon likelihood (§35.1, p. 494-495), generalised from two blocks to
    ``k``: each block pair gets its own maximum-likelihood connection probability ``p_rs = edges
    / possible``, and :func:`_hrg_term` -- the same Bernoulli log-likelihood term §37.2's HRG
    fits with -- scores it. ``e`` must come from a 0/1 adjacency matrix; on a weighted graph this
    ignores the weights, which is the Bernoulli model's own limit, not a bug introduced here."""
    total = 0.0
    for r in range(k):
        for s in range(r, k):
            n_r, n_s = sizes[r], sizes[s]
            if r == s:
                possible = n_r * (n_r - 1) / 2
                edges = e[r, r] / 2
            else:
                possible = n_r * n_s
                edges = e[r, s]
            total += _hrg_term(round(float(edges)), round(float(possible)))
    return total


def _dc_score(e: np.ndarray, _sizes: np.ndarray, k: int) -> float:
    """The degree-corrected blockmodel's own maximum-likelihood log-likelihood (Karrer and
    Newman 2011, cited at §35.1's ref 25), up to an additive constant that does not depend on the
    partition -- so only *differences* between two fits are meaningful, never the raw value.

    ``L(g) = 1/2 * sum_{r,s} e_rs * ln(e_rs / (kappa_r * kappa_s))``, summed over every ordered
    pair of blocks including ``r == s``, where ``kappa_r = sum_s e_rs`` is block ``r``'s total
    (weighted) degree. Unlike :func:`_plain_score` this needs no block sizes: a block's expected
    edges scale with its members' own degrees rather than with how many members it has, which is
    exactly the correction the plain model lacks (p. 495) -- a hub's high degree is explained by
    its own ``theta``, implicit here in ``kappa``, instead of by inflating its block's density.
    """
    kappa = e.sum(axis=1)
    total = 0.0
    for r in range(k):
        for s in range(k):
            if e[r, s] <= 0 or kappa[r] <= 0 or kappa[s] <= 0:
                continue
            total += e[r, s] * math.log(e[r, s] / (kappa[r] * kappa[s]))
    return 0.5 * total


def _move_delta(e: np.ndarray, contact_i: np.ndarray, a: int, b: int) -> np.ndarray:
    """The block matrix after moving one node from block ``a`` to block ``b``, given
    ``contact_i``: that node's total edge weight to every block, read off before the move.

    Node ``i``'s edges to the rest of ``a`` (``contact_i[a]``) were counted twice in ``e[a, a]``
    and become, after the move, edges from ``b`` to ``a`` -- counted once each way. Symmetrically
    for ``contact_i[b]``. Every other block ``s`` simply swaps which of ``a`` or ``b`` its cross
    weight with node ``i`` is credited to. This is the incremental form of rebuilding
    :func:`_block_weights` from scratch after the move, and is what lets the greedy search in
    :func:`_greedy_partition` try every candidate move in O(k) rather than O(n^2).
    """
    if a == b:
        return e
    new_e = e.copy()
    new_e[a, a] -= 2 * contact_i[a]
    new_e[b, b] += 2 * contact_i[b]
    cross = contact_i[a] - contact_i[b]
    new_e[a, b] += cross
    new_e[b, a] += cross
    for s in range(len(contact_i)):
        if s in (a, b):
            continue
        new_e[a, s] -= contact_i[s]
        new_e[s, a] -= contact_i[s]
        new_e[b, s] += contact_i[s]
        new_e[s, b] += contact_i[s]
    return new_e


def _greedy_partition(
    matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
    score: Callable[[np.ndarray, np.ndarray, int], float],
    adjacency_bool: np.ndarray,
    rng: random.Random,
    max_sweeps: int,
) -> tuple[np.ndarray, float, int]:
    """Greedy local search maximising ``score`` by moving one node at a time (§35.1's own
    Expectation-Maximization stand-in: the chapter names EM as the principled way to search the
    partition space and hands the actual search to later chapters' heuristics -- this is that
    heuristic, node-moving in the style :func:`_infomap_search` uses for a different objective).

    Every node starts at ``labels`` (its spectral-and-k-means seed) and, each sweep, is offered a
    move to any block a neighbour currently holds; the move that improves ``score`` the most is
    taken, ties broken by seed order. Stops at the first sweep that moves nothing, or at
    ``max_sweeps``. Returns the final labels, ``score`` there, and the sweep count.
    """
    n = matrix.shape[0]
    labels = labels.copy()
    e, sizes = _block_weights(matrix, labels, k)
    contact = matrix @ _one_hot(labels, k)
    sweep = 0
    for sweep in range(1, max_sweeps + 1):  # noqa: B007 -- the final count is the return value
        order = list(range(n))
        rng.shuffle(order)
        moved = False
        for i in order:
            a = int(labels[i])
            candidates = {int(labels[j]) for j in np.nonzero(adjacency_bool[i])[0]} | {a}
            current_score = score(e, sizes, k)
            best_b, best_score = a, current_score
            for b in candidates:
                if b == a:
                    continue
                trial_e = _move_delta(e, contact[i], a, b)
                trial_sizes = sizes.copy()
                trial_sizes[a] -= 1
                trial_sizes[b] += 1
                trial_score = score(trial_e, trial_sizes, k)
                if trial_score > best_score + 1e-9:
                    best_score, best_b = trial_score, b
            if best_b != a:
                e = _move_delta(e, contact[i], a, best_b)
                contact[:, a] -= matrix[:, i]
                contact[:, best_b] += matrix[:, i]
                sizes[a] -= 1
                sizes[best_b] += 1
                labels[i] = best_b
                moved = True
        if not moved:
            break
    return labels, score(e, sizes, k), sweep


@dataclass(frozen=True)
class SBMFit:
    """A stochastic blockmodel fitted to one network (§35.1): the partition its search settled
    on, the log-likelihood it reaches there, and which implementation produced it."""

    communities: list[list[str]]
    k: int
    degree_corrected: bool
    log_likelihood: float
    implementation: str
    #: One row per candidate ``k`` tried when ``k`` was not fixed: its best log-likelihood and
    #: the BIC used to choose among them (fewer free parameters is rewarded, as in
    #: :func:`choose_k_gmm`). Empty when ``k`` was given directly.
    choices: list[dict[str, float]]
    sweeps: int
    seed: int
    note: str = ""


#: Candidate block counts tried when ``k`` is not fixed, the same shape as :data:`COVARIANCES`'s
#: k-range convention elsewhere in this module.
SBM_K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

#: Above this many nodes :func:`sbm_communities` refuses: the greedy fit holds a dense n x n
#: matrix and restarts several times per candidate ``k``.
SBM_MAX_NODES = 400

SBM_MAX_SWEEPS = 50
SBM_RESTARTS = 5

#: What :func:`sbm_communities` reports when it did the fitting itself rather than delegating.
SPECTRAL_GREEDY = "spectral-initialised greedy likelihood fit"


def _weighted_matrix(graph: nx.Graph, order: Sequence[str]) -> np.ndarray:
    matrix, _ = adjacency(graph, nodes=order, weight="weight")
    return dense(matrix)


def _binary_matrix(graph: nx.Graph, order: Sequence[str]) -> np.ndarray:
    matrix, _ = adjacency(graph, nodes=order, weight=None)
    return (dense(matrix) > 0).astype(np.float64)


def _try_graph_tool(
    flat: nx.Graph, order: list[str], k: int | None, degree_corrected: bool, run_seed: int
) -> SBMFit | None:
    """Delegate to ``graph-tool``'s own Bayesian inference when it is importable, else ``None``.

    §35.1 names ``graph-tool`` (Peixoto's own package, ref 26) as the principled way to fit an
    SBM: minimum-description-length model selection over ``k`` by Markov chain Monte Carlo,
    rather than this module's BIC-over-a-greedy-search stand-in. It has no PyPI wheel -- it is
    built against Boost and CGAL, and is not something ``pip``/``uv`` can install at all; it is
    ordinarily installed from conda-forge (``conda install -c conda-forge graph-tool``) or a
    distribution's own package (e.g. ``apt install python3-graph-tool`` on Debian/Ubuntu with the
    deb.skewed.de repository added). This function only ever *detects* it on whatever interpreter
    already has it on its path -- there is deliberately no entry for it in ``pyproject.toml``,
    because declaring an unresolvable PyPI requirement there would make the whole project
    uninstallable via ``uv``/``pip``, not merely this optional path. No environment this ticket
    ran in has ``graph-tool`` installed, so this path is untested; any failure of it -- import or
    otherwise -- falls back to :func:`_greedy_partition` with a note, rather than letting an
    unverified integration take down a report that has a working default.
    """
    try:
        import graph_tool.all as gt  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        index = {node: i for i, node in enumerate(order)}
        g = gt.Graph(directed=False)
        g.add_vertex(len(order))
        weight_prop = g.new_edge_property("double")
        for u, v, data in flat.edges(data=True):
            edge = g.add_edge(index[u], index[v])
            weight_prop[edge] = float(data.get("weight", 1.0))
        state_args: dict[str, object] = {"deg_corr": degree_corrected, "recs": [weight_prop]}
        if k is not None:
            state = gt.minimize_blockmodel_dl(g, state_args=state_args, B_min=k, B_max=k)
        else:
            state = gt.minimize_blockmodel_dl(g, state_args=state_args)
        blocks = state.get_blocks()
        raw_labels = [int(blocks[i]) for i in range(len(order))]
        remap = {label: i for i, label in enumerate(sorted(set(raw_labels)))}
        labels = np.array([remap[label] for label in raw_labels])
        found_k = len(remap)
        matrix = _weighted_matrix(flat, order) if degree_corrected else _binary_matrix(flat, order)
        e, sizes = _block_weights(matrix, labels, found_k)
        ll = _dc_score(e, sizes, found_k) if degree_corrected else _plain_score(e, sizes, found_k)
        return SBMFit(
            communities=_communities_from_labels(order, labels.tolist()),
            k=found_k,
            degree_corrected=degree_corrected,
            log_likelihood=ll,
            implementation="graph-tool (minimize_blockmodel_dl)",
            choices=[],
            sweeps=0,
            seed=run_seed,
        )
    except Exception:
        return None


def _sbm_free_parameters(k: int, n: int, degree_corrected: bool) -> float:
    """How many parameters a fit at block count ``k`` spends, for the BIC ``sbm_communities``
    chooses ``k`` by.

    Every fit spends ``k * (k + 1) / 2``: one connection probability (plain) or one block
    affinity ``kappa_r * kappa_s`` term (degree-corrected) per unordered pair of blocks, the
    count :func:`_plain_score` and :func:`_dc_score` both sum over. The degree-corrected model
    spends ``n - k`` more on top of that: Karrer and Newman's own ``theta_i``, one per node,
    constrained to sum to 1 within each of the ``k`` blocks, which is ``n`` free values minus
    ``k`` constraints. The plain model has no ``theta`` at all, so it pays nothing extra.
    """
    block_pairs = k * (k + 1) / 2
    return block_pairs + (n - k) if degree_corrected else block_pairs


def sbm_communities(
    graph: nx.Graph,
    k: int | None = None,
    k_range: Sequence[int] = SBM_K_RANGE,
    degree_corrected: bool = True,
    seed: int | None = None,
    restarts: int = SBM_RESTARTS,
    max_sweeps: int = SBM_MAX_SWEEPS,
    max_nodes: int = SBM_MAX_NODES,
) -> SBMFit:
    """Fit a stochastic blockmodel to ``graph`` (§35.1) and return the partition it implies.

    Tries ``graph-tool`` first (:func:`_try_graph_tool`) when it is importable; otherwise --
    always, in every environment this ticket could test in -- runs the default path this ticket
    guarantees: spectral-embed the graph (:func:`spectral_embedding`), seed ``k``-means from it,
    then greedily move nodes between blocks to maximise the model's own log-likelihood
    (:func:`_greedy_partition`), from ``restarts`` seeded starts, keeping the best.

    ``degree_corrected`` switches between the two likelihoods §35.1 contrasts: the plain
    Bernoulli one (:func:`_plain_score`, the book's own cartoon example, generalised past two
    blocks) uses only which nodes connect, blind to how many edges each one has; the
    degree-corrected one (:func:`_dc_score`, Karrer and Newman 2011) explains a node's own degree
    away with its own parameter before asking what its block is. The practical difference the
    chapter and this ticket's tests both point at: on a network with one very high-degree hub in
    an otherwise ordinary block, the plain fit tends to carve the hub into a block of its own --
    it is the only way that model can explain a degree that looks nothing like its neighbours' --
    while the degree-corrected fit keeps it with the block its edges actually reach.

    ``k`` fixes the block count; left unset, every ``k`` in ``k_range`` (clamped to ``2 <= k <
    n``) is fit and the one with the lowest BIC is kept, with every candidate's numbers in
    ``choices`` (:func:`_sbm_free_parameters`). Both models spend ``k * (k + 1) / 2`` parameters,
    one connection probability or block affinity per unordered block pair; the degree-corrected
    model spends ``n - k`` more on top of that for its own per-node ``theta`` (Karrer and Newman
    2011), one per node constrained to sum to 1 within each block. Comparing a plain and a
    degree-corrected fit's BIC directly is therefore comparing two different penalties as well as
    two different likelihoods, which is expected: they are different models of the same data, not
    two settings of one model.

    A directed network is flattened first (§6.2): every blockmodel here is defined on a symmetric
    adjacency. Raises above ``max_nodes`` (:data:`SBM_MAX_NODES`); an edgeless graph returns
    every node as its own block, log-likelihood 0.0 (no edges to explain, so no partition is
    wrong), attributed to the default path since there was nothing for either implementation to
    fit.
    """
    if graph.number_of_nodes() == 0:
        return SBMFit([], 0, degree_corrected, 0.0, SPECTRAL_GREEDY, [], 0, _seeds(seed, 1)[0], "")
    flat, flattened = undirected_view(graph)
    note = f"The blockmodel fit ran {flattened}." if flattened else ""
    if flat.number_of_nodes() > max_nodes:
        msg = (
            f"sbm_communities() refuses a network of {flat.number_of_nodes():,} nodes: the "
            f"default fit holds a dense n x n matrix and restarts several searches over it, "
            f"unusable above {max_nodes:,} nodes. Sample the network first (ch. 29: sna sample) "
            "or take a backbone (ch. 27: sna backbone) and run this on that instead."
        )
        raise ValueError(msg)
    order = sorted(flat.nodes, key=str)
    run_seed = _seeds(seed, 1)[0]
    if flat.number_of_edges() == 0:
        communities = [[node] for node in order]
        return SBMFit(
            communities, len(order), degree_corrected, 0.0, SPECTRAL_GREEDY, [], 0, run_seed, note
        )

    delegated = _try_graph_tool(flat, order, k, degree_corrected, run_seed)
    if delegated is not None:
        return SBMFit(
            communities=delegated.communities,
            k=delegated.k,
            degree_corrected=delegated.degree_corrected,
            log_likelihood=delegated.log_likelihood,
            implementation=delegated.implementation,
            choices=delegated.choices,
            sweeps=delegated.sweeps,
            seed=delegated.seed,
            note=note,
        )

    weighted = _weighted_matrix(flat, order)
    binary = _binary_matrix(flat, order)
    scoring_matrix = weighted if degree_corrected else binary
    score_fn: Callable[[np.ndarray, np.ndarray, int], float] = (
        _dc_score if degree_corrected else (_plain_score)
    )
    adjacency_bool = binary > 0

    candidates = [k] if k is not None else [c for c in k_range if 2 <= c < len(order)]
    if not candidates:
        candidates = [min(2, len(order))]

    choices: list[dict[str, float]] = []
    best: tuple[int, np.ndarray, float, int, float] | None = None  # k, labels, ll, sweeps, bic
    for candidate_k in candidates:
        best_for_k: tuple[np.ndarray, float, int] | None = None
        for restart in range(max(restarts, 1)):
            restart_seed = run_seed + restart + candidate_k * 1000
            embedding = spectral_embedding(flat, dims=min(8, len(order) - 1), seed=restart_seed)
            init_labels = np.array(kmeans(embedding, candidate_k, restart_seed).labels)
            rng = random.Random(restart_seed)  # noqa: S311 -- reproducibility, not secrecy
            labels, ll, sweeps = _greedy_partition(
                scoring_matrix, init_labels, candidate_k, score_fn, adjacency_bool, rng, max_sweeps
            )
            if best_for_k is None or ll > best_for_k[1]:
                best_for_k = (labels, ll, sweeps)
        assert best_for_k is not None  # restarts is at least 1
        labels, ll, sweeps = best_for_k
        params = _sbm_free_parameters(candidate_k, len(order), degree_corrected)
        bic = -2 * ll + params * math.log(len(order))
        choices.append({"k": float(candidate_k), "log_likelihood": ll, "bic": bic})
        if best is None or bic < best[4]:
            best = (candidate_k, labels, ll, sweeps, bic)
    assert best is not None  # candidates is never empty
    found_k, labels, ll, sweeps, _bic = best
    return SBMFit(
        communities=_communities_from_labels(order, labels.tolist()),
        k=found_k,
        degree_corrected=degree_corrected,
        log_likelihood=ll,
        implementation=SPECTRAL_GREEDY,
        choices=choices if k is None else [],
        sweeps=sweeps,
        seed=run_seed,
        note=note,
    )


# --------------------------------------------------------------------------- §35.4 temporal


@dataclass(frozen=True)
class Transition:
    """One correspondence between a community at one window and one at the next (§35.4, Figures
    35.9-35.11): which label it was, which it became, and which of the book's named events that
    is. ``from_label`` is ``None`` for a birth and ``to_label`` is ``None`` for a death."""

    from_window: str
    to_window: str
    from_label: int | None
    to_label: int | None
    jaccard: float
    kind: str
    """One of "birth", "death", "continue", "grow", "shrink", "merge", "split"."""


@dataclass(frozen=True)
class TemporalCommunities:
    """A partition tracked across snapshots (§35.4): one static partition per window, joined by
    the :class:`Transition` that matches each window's communities to the next's."""

    windows: list[str]
    partitions: list[list[list[str]]]
    transitions: list[Transition]
    method: str
    threshold: float
    note: str = ""


#: The minimum Jaccard overlap for two communities in consecutive windows to be read as the same
#: lineage rather than an unrelated birth and death -- p. 502's own suggestion ("A possible
#: similarity criterion would be calculating the Jaccard coefficient"), given a cutoff because
#: the book leaves the threshold itself unspecified.
TEMPORAL_MATCH_THRESHOLD = 0.1


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def match_communities(
    snapshots: Sequence[tuple[str, nx.Graph]],
    threshold: float = TEMPORAL_MATCH_THRESHOLD,
    resolution: float = 1.0,
    seed: int | None = None,
    runs: int = 10,
) -> TemporalCommunities:
    """Track Louvain communities across ``snapshots`` (§35.4), the *"series of network
    snapshots"* approach the chapter opens with (p. 502) rather than evolutionary clustering's
    smoothed quality function -- the simpler of the two named approaches, and the one buildable
    without re-deriving a temporally-aware objective for every method in this module.

    ``snapshots`` is what :func:`graphrag.sna.layers.snapshots` returns: ``(label, graph)`` pairs
    in time order. Louvain runs on each independently (:func:`louvain`, with its own stability
    check per window), and each pair of consecutive partitions is joined by the Jaccard overlap
    of their communities (p. 502): a pair at or above ``threshold`` is a match, and every match is
    classified by the book's own taxonomy (Figures 35.9-35.11) -- **birth** (nothing before it
    matched), **death** (nothing after it matches), **continue**/**grow**/**shrink** (exactly one
    match on each side, by whether the community's size changed), **merge** (more than one
    earlier community matches the same later one) and **split** (the mirror: one earlier
    community matches more than one later one).

    The chapter is explicit that real evolution is messier than these six labels (p. 503: *"don't
    assume you're going to be able to say, unequivocally, something like 'community C split in C1
    and C2 at time t'"*) -- a community that both split and partly merged with another in the
    same window gets whichever label its own matches individually earn per pair, which is why
    :class:`Transition` is one row per *pair* rather than one row per community.

    Raises on fewer than two snapshots (nothing to track across) and on ``threshold`` outside
    ``[0, 1]``, a fraction the Jaccard coefficient cannot exceed either end of.
    """
    if len(snapshots) < 2:
        msg = "match_communities() needs at least two snapshots to track anything across"
        raise ValueError(msg)
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be between 0 and 1, got {threshold}"
        raise ValueError(msg)
    labels = [window for window, _ in snapshots]
    partitions = [
        louvain(graph, resolution=resolution, seed=seed, runs=runs).communities
        for _, graph in snapshots
    ]

    transitions: list[Transition] = []
    for t in range(1, len(partitions)):
        prev_sets = [set(c) for c in partitions[t - 1]]
        curr_sets = [set(c) for c in partitions[t]]
        scored = [
            (i, j, _jaccard(prev_sets[i], curr_sets[j]))
            for i in range(len(prev_sets))
            for j in range(len(curr_sets))
        ]
        matched = [(i, j, s) for i, j, s in scored if s >= threshold]
        out_counts = Counter(i for i, _, _ in matched)
        in_counts = Counter(j for _, j, _ in matched)
        matched_prev = {i for i, _, _ in matched}
        matched_curr = {j for _, j, _ in matched}
        for i, j, s in matched:
            if in_counts[j] > 1:
                kind = "merge"
            elif out_counts[i] > 1:
                kind = "split"
            else:
                size_change = len(curr_sets[j]) - len(prev_sets[i])
                kind = "grow" if size_change > 0 else "shrink" if size_change < 0 else "continue"
            transitions.append(Transition(labels[t - 1], labels[t], i, j, s, kind))
        transitions += [
            Transition(labels[t - 1], labels[t], i, None, 0.0, "death")
            for i in range(len(prev_sets))
            if i not in matched_prev
        ]
        transitions += [
            Transition(labels[t - 1], labels[t], None, j, 0.0, "birth")
            for j in range(len(curr_sets))
            if j not in matched_curr
        ]
    return TemporalCommunities(
        windows=labels,
        partitions=partitions,
        transitions=transitions,
        method="louvain",
        threshold=threshold,
    )


# --------------------------------------------------------------------------- §35.5 local


@dataclass(frozen=True)
class LocalCommunity:
    """A community grown outward from one seed node (§35.5), without ever looking at the rest of
    the network.

    ``trajectory`` is every node in the order the greedy expansion accepted it, ``seed`` first,
    and ``members`` is the same list -- growth stops the moment nothing left in the frontier
    would hold ``local_modularity``, so nothing is ever grown past the community reported.
    """

    seed: str
    members: list[str]
    trajectory: list[str]
    local_modularity: float
    stopped: str
    """Why growth stopped: "local modularity optimum" (every remaining neighbour would lower
    ``R``), "explored the whole component" (the frontier ran out first) or "reached the
    exploration limit" (``max_size``)."""
    note: str = ""


def _local_modularity(graph: nx.Graph, members: set[str]) -> float:
    """Clauset's ``R`` (§35.5 ref 65, p. 505-506): of every edge touching ``members``, the share
    with both ends inside it. 0.0 when ``members`` touches no edge at all."""
    boundary = 0
    internal = 0
    seen: set[frozenset[str]] = set()
    for node in members:
        for neighbour in graph.neighbors(node):
            edge = frozenset((node, neighbour))
            if edge in seen:
                continue
            seen.add(edge)
            boundary += 1
            if neighbour in members:
                internal += 1
    return internal / boundary if boundary else 0.0


def local_community(
    graph: nx.Graph, seed_node: str, max_size: int | None = None, seed: int | None = None
) -> LocalCommunity:
    """Grow a community outward from ``seed_node`` (§35.5), one node at a time, never looking
    beyond the frontier it has explored.

    *"You start from a seed node v0 ... All of its neighbors are part of the unexplored node set
    U. You iterate over all members of U, trying to find the one that would maximize some
    community quality function ... We add to C the v1 node with the most edges to the local
    community"* (p. 505) -- the quality function used here is Clauset's local modularity ``R``
    (:func:`_local_modularity`, ref 65), the concrete instance the chapter leaves as "some
    community quality function". Ties among candidates giving the same ``R`` are broken at random
    (seeded), matching the book's own tie-breaking for the plain edge-count version.

    The chapter's own stopping rule is external -- *"we explored the number of nodes we wanted to
    test, or we ran out of time, or we actually explored all nodes in the component"* (p. 505) --
    but on a network small enough to explore in full, "all nodes in the component" is not a useful
    community: ``R`` returns to 1.0 the moment ``C`` becomes the whole of a connected component,
    since every edge then has both ends inside it, so hunting for that value's *global* peak
    would always end at "the whole component" and never at a boundary. What Clauset's own
    algorithm does instead, and what this does, is stop the first time growing would make ``R``
    go down: each step takes the single best candidate the frontier offers (accepting a tie, so a
    plateau -- every candidate equally good, as inside an unbroken clique -- keeps growing rather
    than stopping on the first step), and refuses it once the best available ``R`` is below the
    community's current one, which is exactly the point where the next node has fewer ties back
    into ``C`` than out of it.

    Raises when ``seed_node`` is not in the network. A directed network is flattened first (§6.2):
    ``R`` is defined on undirected boundary edges, as every other measure in this module is.
    """
    flat, flattened = undirected_view(graph)
    note = f"Local community discovery ran {flattened}." if flattened else ""
    if seed_node not in flat:
        msg = f"local_community() was given a seed not in the network: {seed_node!r}"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    trajectory = [seed_node]
    member_set = {seed_node}
    frontier = set(flat.neighbors(seed_node)) - member_set
    current_r = _local_modularity(flat, member_set)
    limit = max_size if max_size is not None else flat.number_of_nodes()
    stopped = "explored the whole component"
    while frontier and len(member_set) < limit:
        best_r: float | None = None
        best_candidates: list[str] = []
        for candidate in sorted(frontier):
            trial_r = _local_modularity(flat, member_set | {candidate})
            if best_r is None or trial_r > best_r + 1e-12:
                best_r, best_candidates = trial_r, [candidate]
            elif abs(trial_r - best_r) <= 1e-12:
                best_candidates.append(candidate)
        if best_r is not None and best_r < current_r - 1e-12:
            stopped = "local modularity optimum"
            break
        chosen = rng.choice(best_candidates) if len(best_candidates) > 1 else best_candidates[0]
        member_set.add(chosen)
        trajectory.append(chosen)
        frontier.discard(chosen)
        frontier |= set(flat.neighbors(chosen)) - member_set
        current_r = best_r if best_r is not None else current_r
    if stopped == "explored the whole component" and len(member_set) >= limit and frontier:
        stopped = "reached the exploration limit"
    return LocalCommunity(
        seed=seed_node,
        members=sorted(trajectory),
        trajectory=trajectory,
        local_modularity=current_r,
        stopped=stopped,
        note=note,
    )


def render_local_community(result: LocalCommunity) -> list[str]:
    """``sna community --seed`` as markdown: the community §35.5 grew, and why it stopped."""
    lines = [
        f"## Local community: {result.seed}",
        "",
        f"Grown outward from `{result.seed}` (§35.5), never looking past its own frontier: "
        f"{len(result.members)} node(s), local modularity R = {result.local_modularity:.4f}.",
        f"Stopped: {result.stopped}.",
        "",
    ]
    if result.note:
        lines += [result.note, ""]
    lines += ["Members, in the order they were added:", "", ", ".join(result.trajectory), ""]
    return lines


def local_community_payload(result: LocalCommunity) -> dict[str, object]:
    return {
        "implements": "Atlas §35.5 (local community discovery, Clauset's local modularity)",
        "seed": result.seed,
        "members": result.members,
        "trajectory": result.trajectory,
        "local_modularity": result.local_modularity,
        "stopped": result.stopped,
        "note": result.note,
    }


def render_temporal_communities(tracked: TemporalCommunities) -> list[str]:
    """``sna community --temporal`` as markdown: one partition per window and the transitions
    joining them (§35.4)."""
    lines = [
        "## Temporal communities",
        "",
        f"Method: {tracked.method}. Match threshold: {tracked.threshold} (Jaccard). "
        f"{len(tracked.windows)} windows.",
        "",
    ]
    lines += _table(
        ["window", "communities"],
        [
            [window, str(len(partition))]
            for window, partition in zip(tracked.windows, tracked.partitions, strict=True)
        ],
    )
    lines += ["### Transitions", ""]
    lines += _table(
        ["from window", "from", "to window", "to", "jaccard", "kind"],
        [
            [
                t.from_window,
                str(t.from_label) if t.from_label is not None else "-",
                t.to_window,
                str(t.to_label) if t.to_label is not None else "-",
                f"{t.jaccard:.3f}",
                t.kind,
            ]
            for t in tracked.transitions
        ],
    )
    if tracked.note:
        lines += [tracked.note, ""]
    return lines


def temporal_communities_payload(tracked: TemporalCommunities) -> dict[str, object]:
    return {
        "implements": "Atlas §35.4 (temporal communities, matched by Jaccard overlap)",
        "method": tracked.method,
        "threshold": tracked.threshold,
        "windows": tracked.windows,
        "partitions": tracked.partitions,
        "transitions": [
            {
                "from_window": t.from_window,
                "to_window": t.to_window,
                "from_label": t.from_label,
                "to_label": t.to_label,
                "jaccard": t.jaccard,
                "kind": t.kind,
            }
            for t in tracked.transitions
        ],
        "note": tracked.note,
    }
