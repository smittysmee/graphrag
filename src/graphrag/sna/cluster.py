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

Nothing here re-implements an algorithm: Louvain and modularity come from ``networkx``,
K-means, Gaussian mixtures, spectral embedding and the agreement indices from ``scikit-learn``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

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
            return "not testable: the network is too small to rewire"
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
    """
    if graph.number_of_nodes() == 0:
        return LouvainResult([], 0.0, resolution, [], [], 1.0)
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
    )


def spectral_embedding(graph: nx.Graph, dims: int = 8, seed: int | None = None) -> np.ndarray:
    """Coordinates for each node from the normalised Laplacian, aligned to ``list(graph.nodes)``.

    This is the bridge from edges to vectors: once a graph has coordinates, K-means and Gaussian
    mixtures apply, and you can ask for exactly ``k`` groups. The weighted adjacency is handed
    to scikit-learn as a precomputed affinity, so heavier edges pull nodes together.

    ``dims`` is clamped to ``n - 1``; an embedding cannot have more dimensions than the graph
    has nodes to separate.
    """
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return np.zeros((len(nodes), 1), dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=np.float64)
    components = max(1, min(dims, len(nodes) - 1))
    embedder = SpectralEmbedding(n_components=components, affinity="precomputed", random_state=seed)
    embedded: np.ndarray = embedder.fit_transform(adjacency)
    return embedded


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


def compare_partitions(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """Agreement between two labellings of the same nodes.

    The adjusted Rand index is corrected for chance: 0 is what random labellings score, 1 is
    identical. Normalised mutual information is not chance-corrected and drifts upward as the
    number of clusters grows, so quote both.
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
        "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
    }


def null_model_modularity(
    graph: nx.Graph,
    communities: Sequence[Iterable[str]],
    samples: int = 50,
    seed: int | None = None,
    resolution: float = 1.0,
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
    """
    partition = [sorted(c) for c in communities]
    observed = (
        float(nx.community.modularity(graph, partition, weight="weight", resolution=resolution))
        if partition and graph.number_of_edges()
        else 0.0
    )
    edges = graph.number_of_edges()
    if graph.number_of_nodes() < 4 or edges < 2 or not partition:
        return NullModelResult(observed, observed, 0.0, 0.0, 0)

    # Reproducibility, not secrecy: the same seed must produce the same null distribution.
    rng = random.Random(seed)  # noqa: S311
    weights = [float(d.get("weight", 1.0)) for _, _, d in graph.edges(data=True)]
    scores: list[float] = []
    for _ in range(samples):
        rewired = graph.copy()
        try:
            nx.double_edge_swap(
                rewired, nswap=edges, max_tries=edges * 20, seed=rng.randrange(1_000_000)
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue  # too few distinct degrees to rewire; this sample contributes nothing
        shuffled = list(weights)
        rng.shuffle(shuffled)
        for (u, v), weight in zip(rewired.edges(), shuffled, strict=True):
            rewired[u][v]["weight"] = weight
        found = nx.community.louvain_communities(
            rewired, weight="weight", resolution=resolution, seed=rng.randrange(1_000_000)
        )
        scores.append(
            float(nx.community.modularity(rewired, found, weight="weight", resolution=resolution))
        )
    if not scores:
        return NullModelResult(observed, observed, 0.0, 0.0, 0)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    z = (observed - mean) / std if std > 0 else 0.0
    return NullModelResult(observed, mean, std, float(z), len(scores))
