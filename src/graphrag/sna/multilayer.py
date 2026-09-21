"""Chapter 40's community discovery over the multilayer networks of :mod:`graphrag.sna.layers`.

Three answers to "what is a community, here, across layers" (§40.1-§40.3), one ambiguity chapter
40 refuses to resolve for you (§40.4), and one number this module computes so that the choice
among the three is not a coin flip (§40.5). All three read the same :class:`graphrag.sna.layers.
Multilayer`, and every partition any of them produces is scored by :func:`graphrag.sna.evaluate.
evaluate_partition`, the same battery every other grouping in this package goes through -- so the
three are read on the same terms rather than three unrelated numbers.

**Flattening (§40.1)** is :func:`graphrag.sna.layers.flatten`, already built for ATL-07 and
already how ``sna analyze --layers`` runs a grouping; nothing here rebuilds it. What this module
adds is running :func:`graphrag.sna.cluster.louvain` on the result and putting it beside the
other two methods, because "flatten it and cluster it" is only a *finding* about the corpus once
it can be compared with an answer that never assumed every layer mattered equally.

**Layer-by-layer with cross-layer matching (§40.2)** is Figure 40.2's own procedure, followed to
the letter rather than approximated: Louvain on each layer separately, an overlap graph over the
per-layer communities (an edge when two of them share at least ``threshold`` nodes), and the
*maximal cliques* of that graph as the multilayer communities -- which is what makes one
per-layer community able to sit in two of them at once, exactly as the book's own worked example
does (p. 574: "C1L2 is part of two maximal sets"). A node's final affiliation is the intersection
of every constituent community's membership (Figure 40.2c); a node satisfying more than one
maximal set is assigned to the largest for the sake of :func:`evaluate_partition`'s disjointness
requirement, and the report says how often that happened -- the book allows the true overlap and
this module keeps it in :attr:`LayerByLayerResult.multilayer_communities`, only flattening it for
scoring.

**Multilayer modularity over the supra-adjacency (§40.3)** is Mucha et al.'s ``B`` matrix, built
on top of :func:`graphrag.sna.layers.supra_adjacency`: every diagonal (within-layer) block gets
its own configuration-model null subtracted, every off-diagonal (coupling) block is left alone,
because the coupling term of the formula carries no null of its own. Maximising it is a greedy
local-moving pass over the (node, layer) supra-nodes -- the first phase of Louvain, iterated to
convergence, without the hierarchical aggregation phase that the full "generalised Louvain"
(GenLouvain, Jutla, Jeub and Mucha) adds; see the module's own admission of that in
:func:`optimize_multilayer_modularity`. ``omega`` is Cvsr, a parameter the corpus does not state
(§7.2), and :func:`omega_sweep` is what makes its effect visible rather than assumed: at 0 the
layers cannot influence each other and the supra-partition degenerates to running each layer on
its own; as it grows every node is pulled toward "pillar" communities (Figure 40.5a) that ignore
what it actually connects to in favour of what it *is* in every layer.

**Multilayer density (§40.4)** is deliberately two numbers, because the chapter says there is no
one answer: :func:`redundancy` (every pair joined through every layer at once) and
:func:`complementarity` (variety x exclusivity x homogeneity: the pairs joined through *some*
layer, spread evenly across many of them). Both are exact reproductions of the chapter's formulas
except the third term of complementarity, where the chapter's own worked example cannot be
reproduced from what it states; see :data:`DEPARTURE_HOMOGENEITY`.

**§40.5** is not a method, it is the chapter's own closing argument -- "you can intend
'communities' in complex networks in a thousand different ways" -- turned into one number:
:func:`layer_agreement` scores how much the per-layer partitions already agree (mean adjusted
Rand index over shared nodes) and recommends flattening when they do and layer-by-layer /
low-omega supra modularity when they do not, because that is exactly the axis every method above
threatens to paper over (§40.2, p. 575: "these methods have the downside of relying more or less
on the same assumption: that the layers are correlated to each other").
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations

import networkx as nx
import numpy as np

from graphrag.sna.cluster import LouvainResult, NullModelResult, louvain, null_model_modularity
from graphrag.sna.evaluate import PartitionScores, evaluate_partition, render_evaluation
from graphrag.sna.layers import Multilayer, flatten, supra_adjacency
from graphrag.sna.measures import undirected_view
from graphrag.sna.null import Significance, configuration, significance
from graphrag.sna.stats import compare_partitions, mean

__all__ = [
    "AGREEMENT_HIGH",
    "CHAPTER",
    "DEFAULT_MIN_SHARED",
    "DEFAULT_NULL_SAMPLES",
    "DEFAULT_OMEGA_SWEEP",
    "DEPARTURE_HOMOGENEITY",
    "Complementarity",
    "LayerAgreement",
    "LayerByLayerResult",
    "LayerCommunity",
    "MultilayerCommunity",
    "MultilayerCommunityReport",
    "OmegaSweepRow",
    "Redundancy",
    "SupraModularityResult",
    "analyze_multilayer_communities",
    "complementarity",
    "density_table",
    "layer_agreement",
    "layer_by_layer_communities",
    "modularity_matrix",
    "multilayer_communities_payload",
    "multilayer_modularity_value",
    "null_model_multilayer_modularity",
    "omega_sweep",
    "optimize_multilayer_modularity",
    "redundancy",
    "render_multilayer_communities",
    "supra_network",
]

#: What the section header says it implements.
CHAPTER = "§40.1-40.5"

#: Figure 40.2's own worked threshold: two layer-communities merge when they share at least this
#: many nodes. A corpus network should scale it -- three nodes is a lot to ask of a network with
#: eight -- so the CLI exposes it rather than hard-coding the book's example.
DEFAULT_MIN_SHARED = 3

#: A representative spread from "layers cannot influence each other" to "one shared answer",
#: printed beside whatever single ``--omega`` a report asked for. Not the book's numbers -- the
#: book states no default -- chosen only so the shape of the sweep (Figure 40.5) is visible.
DEFAULT_OMEGA_SWEEP: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)

#: Null-model rewirings for §40.3's significance test. Smaller than evaluate.py's 20-50 because
#: each sample here costs one local-moving pass over every layer at once, not one Louvain call.
DEFAULT_NULL_SAMPLES = 20

#: This module's own cutoff for "the layers agree enough to flatten" (§40.5), stated as a number
#: rather than left to a reader's eye. Chosen, not measured: it is the same threshold the book's
#: own example modularities (p. 512-513) treat as "clearly more than chance" would sit above.
AGREEMENT_HIGH = 0.3

DEPARTURE_HOMOGENEITY = (
    "§40.4 defines homogeneity as 1 - sigma_c/sigma_max_c, 'normalized by its theoretical "
    "maximum', but never writes that maximum down, and its own worked example (p. 583) cannot be "
    "reproduced from the two numbers it states: sigma_c = sqrt(32/9) = 1.886 for the vector "
    "[5, 1, 5] is exact, and the stated ratio of two thirds implies sigma_max_c = 2*sqrt(2) = "
    "2.8284, which is not the population standard deviation of [11, 0, 0] -- the same total edge "
    "count (11) concentrated into one of the three layers, which is the natural theoretical "
    "maximum -- and comes to 5.185. No other concentration of the same total this module tried "
    "reaches 2.8284 either. Rather than present an unreproducible formula silently, this module "
    "states its own: sigma_max_c is the population standard deviation of the community's total "
    "edge count concentrated entirely into one layer, sigma_max_c = e_total * sqrt(L-1) / L. "
    "Homogeneity computed this way is a different number from the book's 0.33 on the book's own "
    "example, and every report that prints it says so."
)


def _safe_louvain(
    graph: nx.Graph, resolution: float = 1.0, seed: int | None = None, runs: int = 10
) -> LouvainResult:
    """:func:`graphrag.sna.cluster.louvain`, or an edgeless graph's own trivial partition.

    **Where this departs from the book, and why it exists at all.** ``cluster.louvain()`` scores
    every run by ``nx.community.modularity``, unconditionally, which divides by the graph's total
    degree -- 0 on a graph with nodes but no edges. :mod:`graphrag.sna.layers` documents that
    case as a real, expected layer -- "a layer where two nodes have many common neighbors...
    holds a node and no edge at all -- which is the case worth having a row for" -- and §40.2
    calls Louvain once per layer, so an edgeless layer is not a hypothetical here. This ticket
    does not touch ``cluster.py``, so the guard lives here: an edgeless graph's only sensible
    partition is every node on its own, modularity 0 (undefined made concrete, since there is no
    edge to be inside or outside of any grouping), and perfectly stable across every seed because
    there is nothing for a seed to disagree about.
    """
    if graph.number_of_edges() == 0:
        communities = [[node] for node in sorted(graph.nodes, key=str)]
        return LouvainResult(
            communities=communities,
            modularity=0.0,
            resolution=resolution,
            modularities=[0.0],
            seeds=[seed if seed is not None else 0],
            stability=1.0,
            note="",
        )
    return louvain(graph, resolution=resolution, seed=seed, runs=runs)


# ----------------------------------------------------------------------------- §40.2 layer by layer


@dataclass(frozen=True)
class LayerCommunity:
    """One Louvain community from one layer, before any cross-layer matching (§40.2)."""

    layer: str
    index: int
    members: frozenset[str]

    @property
    def label(self) -> str:
        """``layer:index``, the book's ``C1L1`` notation with this corpus's own layer names."""
        return f"{self.layer}:{self.index + 1}"


@dataclass(frozen=True)
class MultilayerCommunity:
    """One maximal set of layer-communities that overlap enough to merge (Figure 40.2b).

    ``members`` is the intersection of every constituent's own members -- Figure 40.2c's rule,
    "a node is part of a multidimensional community if it is part of all communities composing
    it" -- so a node can qualify for more than one :class:`MultilayerCommunity` at once. That is
    the book's own overlap (p. 574: "we are ok if a community gets merged in different sets"),
    kept here and only flattened into a disjoint partition by :attr:`LayerByLayerResult.partition`
    for the sake of scoring it.
    """

    constituents: tuple[LayerCommunity, ...]
    members: frozenset[str]


@dataclass(frozen=True)
class LayerByLayerResult:
    """§40.2 end to end: one Louvain partition per layer, matched into multilayer communities."""

    threshold: int
    per_layer: dict[str, LouvainResult]
    layer_communities: tuple[LayerCommunity, ...]
    multilayer_communities: tuple[MultilayerCommunity, ...]
    unmerged: tuple[LayerCommunity, ...]
    """Layer-communities that shared ``threshold`` nodes with nothing: never part of a maximal
    set of size two or more, so they stay single-layer answers."""
    assignment: dict[str, int]
    """Node to the index of the :class:`MultilayerCommunity` it was finally assigned, after the
    largest-clique tie-break described in :attr:`overlap_note`."""
    unaffiliated: tuple[str, ...]
    """Nodes that qualified for no multilayer community at all -- Figure 40.2's node 10, "a weird
    combination of communities" that the book explicitly allows to be left unaffiliated."""
    overlap_note: str

    @property
    def partition(self) -> list[list[str]]:
        """The disjoint grouping :func:`graphrag.sna.evaluate.evaluate_partition` can score.

        Unaffiliated nodes are left out entirely, the same way a feature-based clustering that
        had nothing to say about a node leaves it out (§36's own contract): a node with no
        multilayer community is not a community of one.
        """
        groups: dict[int, list[str]] = {}
        for node, index in self.assignment.items():
            groups.setdefault(index, []).append(node)
        return [sorted(members, key=str) for members in groups.values()]


def layer_by_layer_communities(
    ml: Multilayer,
    *,
    threshold: int = DEFAULT_MIN_SHARED,
    resolution: float = 1.0,
    seed: int | None = None,
    runs: int = 10,
) -> LayerByLayerResult:
    """Figure 40.2 over ``ml``: Louvain per layer, then matched into multilayer communities.

    Two layer-communities merge when they share at least ``threshold`` nodes -- the book's own
    example uses 3 (p. 574) -- and the multilayer communities are the *maximal cliques* of the
    overlap graph this builds over the per-layer communities, which is what reproduces the book's
    figure exactly: a triangle of three mutually-overlapping layer-communities is one maximal
    set, and a layer-community that overlaps two unrelated ones sits in two maximal sets rather
    than merging them into one.

    Two layer-communities of the *same* layer never merge with each other: Louvain already made
    them disjoint, so an "overlap" there would only be the empty set.
    """
    per_layer: dict[str, LouvainResult] = {}
    layer_communities: list[LayerCommunity] = []
    for name in ml.names:
        result = _safe_louvain(ml.graphs[name], resolution=resolution, seed=seed, runs=runs)
        per_layer[name] = result
        layer_communities.extend(
            LayerCommunity(layer=name, index=index, members=frozenset(community))
            for index, community in enumerate(result.communities)
        )

    overlap = nx.Graph()
    overlap.add_nodes_from(range(len(layer_communities)))
    for i, a in enumerate(layer_communities):
        for j in range(i + 1, len(layer_communities)):
            b = layer_communities[j]
            if a.layer == b.layer:
                continue
            if len(a.members & b.members) >= threshold:
                overlap.add_edge(i, j)

    cliques = sorted(
        (tuple(sorted(clique)) for clique in nx.find_cliques(overlap) if len(clique) >= 2),
        key=lambda clique: (-len(clique), clique),
    )
    multilayer_communities = tuple(
        MultilayerCommunity(
            constituents=tuple(layer_communities[i] for i in clique),
            members=frozenset.intersection(*(layer_communities[i].members for i in clique)),
        )
        for clique in cliques
    )
    merged = {i for clique in cliques for i in clique}
    unmerged = tuple(lc for i, lc in enumerate(layer_communities) if i not in merged)

    assignment: dict[str, int] = {}
    overlap_ties = 0
    for node in ml.nodes:
        qualifies = [i for i, mlc in enumerate(multilayer_communities) if node in mlc.members]
        if not qualifies:
            continue
        if len(qualifies) > 1:
            overlap_ties += 1
        assignment[node] = qualifies[0]  # sorted largest-first above: a deterministic tie-break
    unaffiliated = tuple(sorted(set(ml.nodes) - set(assignment), key=str))

    overlap_note = (
        f"{overlap_ties} node(s) qualified for more than one multilayer community; each was "
        "assigned to the largest (most constituent layer-communities) so the partition below "
        "stays disjoint, which is what evaluate_partition needs. §40.2 allows the true overlap; "
        "read multilayer_communities for it."
        if overlap_ties
        else "No node qualified for more than one multilayer community here."
    )
    return LayerByLayerResult(
        threshold=threshold,
        per_layer=per_layer,
        layer_communities=tuple(layer_communities),
        multilayer_communities=multilayer_communities,
        unmerged=unmerged,
        assignment=assignment,
        unaffiliated=unaffiliated,
        overlap_note=overlap_note,
    )


# --------------------------------------------------------------------------- §40.3 supra modularity


def _undirected_multilayer(ml: Multilayer) -> tuple[Multilayer, str]:
    """Every layer flattened to undirected (§6.2), or ``ml`` unchanged and "" when it already is.

    §40.3's null term needs a symmetric within-layer degree; a directed layer is flattened the
    same way :func:`graphrag.sna.cluster.louvain` flattens a directed graph before finding its
    communities, for the same reason.
    """
    if not ml.directed:
        return ml, ""
    graphs = {name: undirected_view(ml.graphs[name])[0] for name in ml.names}
    note = (
        "Every layer was flattened to undirected (§6.2) before the supra-adjacency was built, "
        "because §40.3's modularity null term is defined on symmetric within-layer degrees."
    )
    return replace(ml, graphs=graphs), note


def modularity_matrix(
    ml: Multilayer, omega: float | None = None, gamma: float = 1.0
) -> tuple[np.ndarray, list[tuple[str, str]], float]:
    """The multilayer modularity matrix B and 2*mu (Mucha et al. 2010, §40.3).

    ``B[(v,s), (u,r)] = (A_vus - gamma_s * k_vs*k_us / 2|Es|) * delta_sr + C_vsr * delta_uv``:
    every diagonal (within-layer) block of :func:`graphrag.sna.layers.supra_adjacency` gets its
    own configuration-model null subtracted -- ``gamma`` is the book's per-layer importance,
    applied here as one shared value because a corpus gives no reason to prefer one layer over
    another -- and every off-diagonal (coupling) block is left exactly as the supra-adjacency
    built it, because the coupling term of the formula carries no null of its own.

    Returns the matrix, the ``(node, layer)`` order :func:`supra_adjacency` uses, and
    ``2*mu = |E| + |C|`` doubled -- the sum of every entry of the *raw* supra-adjacency, never of
    ``B``, which already carries the subtracted null and would double-count it.

    A layer with no edges contributes nothing to subtract: there is no chance to beat where
    nothing was observed.
    """
    supra, order = supra_adjacency(ml, omega)
    two_mu = float(supra.sum())
    count = len(ml.nodes)
    matrix = supra.copy()
    for index in range(len(ml.names)):
        start = index * count
        block = matrix[start : start + count, start : start + count]
        degree = block.sum(axis=1)
        two_m = float(degree.sum())
        if two_m > 0:
            matrix[start : start + count, start : start + count] = (
                block - gamma * np.outer(degree, degree) / two_m
            )
    return matrix, order, two_mu


def multilayer_modularity_value(
    matrix: np.ndarray,
    order: Sequence[tuple[str, str]],
    labels: Mapping[tuple[str, str], int],
    two_mu: float,
) -> float:
    """Q for one assignment of every (node, layer) supra-node to a community (§40.3).

    ``labels`` is a community index *per supra-node*, not per node: a multilayer partition can,
    and often does, put the same node in different communities in different layers. The special
    case where it never does -- Figure 40.5a's "pillar communities" -- is what
    :attr:`SupraModularityResult.consistency` measures.

    0.0 when ``two_mu`` is 0: a network with no edges and no coupling has nothing for any
    partition to be better than.
    """
    if two_mu <= 0:
        return 0.0
    codes = np.array([labels[key] for key in order])
    total = 0.0
    for label in np.unique(codes):
        idx = np.flatnonzero(codes == label)
        total += float(matrix[np.ix_(idx, idx)].sum())
    return total / two_mu


def _local_moving(
    matrix: np.ndarray, seed: int | None = None, max_passes: int = 100
) -> tuple[np.ndarray, int]:
    """One node-moving pass to convergence: the first phase of Louvain, without aggregation.

    **Where this departs from the book.** Mucha et al.'s own solver, and the "generalised
    Louvain" (GenLouvain) tooling built for it, alternate this local-moving phase with
    hierarchical aggregation -- coarsening each community into one super-node and repeating --
    which finds better optima on large networks. This is the single-level version: correct for
    the same modularity function, cheaper to implement over a dense supra-adjacency matrix that
    is already this package's own choice (§40.3's docstring in ``layers.py``), and adequate for
    a corpus network's handful of layers, but it can settle into a worse local optimum than the
    full multi-level algorithm would on a large one. It is iterated until no node's move
    improves Q, or ``max_passes`` sweeps, whichever comes first.
    """
    n = matrix.shape[0]
    neighbours = [
        [(int(j), float(matrix[i, j])) for j in np.flatnonzero(matrix[i]) if j != i]
        for i in range(n)
    ]
    labels = list(range(n))
    order = list(range(n))
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        rng.shuffle(order)
        for i in order:
            gains: dict[int, float] = {}
            for j, weight in neighbours[i]:
                gains[labels[j]] = gains.get(labels[j], 0.0) + weight
            if not gains:
                continue
            current_gain = gains.get(labels[i], 0.0)
            best_label, best_gain = max(gains.items(), key=lambda item: item[1])
            if best_gain > current_gain + 1e-9:
                labels[i] = best_label
                improved = True
        passes += 1
    return np.array(labels), passes


@dataclass(frozen=True)
class SupraModularityResult:
    """One §40.3 partition of the supra-adjacency, at one omega."""

    omega: float
    gamma: float
    modularity: float
    communities: list[list[tuple[str, str]]]
    """Each community as a list of ``(node, layer)`` supra-nodes -- the graph
    :func:`multilayer_modularity_value` and :func:`supra_network` both use."""
    node_labels: dict[tuple[str, str], int]
    iterations: int
    consistency: float
    """Share of nodes assigned the same community in every layer they appear in: 1.0 is the
    fully "pillar" case (Figure 40.5a), read beside omega rather than alone, since flat
    communities can coincide by chance too. ``nan`` on a single-layer network, where every node
    is trivially consistent with itself."""


def _pillar_consistency(ml: Multilayer, labels: Mapping[tuple[str, str], int]) -> float:
    if len(ml.names) < 2 or not ml.nodes:
        return math.nan
    consistent = sum(1 for node in ml.nodes if len({labels[node, name] for name in ml.names}) == 1)
    return consistent / len(ml.nodes)


def optimize_multilayer_modularity(
    ml: Multilayer,
    omega: float | None = None,
    gamma: float = 1.0,
    seed: int | None = None,
) -> SupraModularityResult:
    """Find communities over the supra-adjacency by greedily maximising §40.3's modularity.

    ``ml`` should already be undirected (:func:`_undirected_multilayer`); this function does not
    convert it, so that a caller running many omegas over the same network -- :func:`omega_sweep`
    -- pays that cost once.
    """
    coupling = ml.omega if omega is None else omega
    matrix, order, two_mu = modularity_matrix(ml, coupling, gamma)
    raw_labels, passes = _local_moving(matrix, seed=seed)
    labels: dict[tuple[str, str], int] = {order[i]: int(raw_labels[i]) for i in range(len(order))}
    modularity = multilayer_modularity_value(matrix, order, labels, two_mu)
    groups: dict[int, list[tuple[str, str]]] = {}
    for key, label in labels.items():
        groups.setdefault(label, []).append(key)
    communities = [sorted(members) for members in groups.values()]
    return SupraModularityResult(
        omega=coupling,
        gamma=gamma,
        modularity=modularity,
        communities=communities,
        node_labels=labels,
        iterations=passes,
        consistency=_pillar_consistency(ml, labels),
    )


def supra_network(ml: Multilayer, omega: float | None = None) -> nx.Graph:
    """The supra-adjacency (§8.1) as a plain weighted graph, node ``(node, layer)``.

    This is what §40.3 actually partitions: :func:`graphrag.sna.evaluate.evaluate_partition`
    needs a graph and a disjoint partition of its nodes and has no opinion about what a node
    *is*, so handing it this graph and a :class:`SupraModularityResult`'s ``communities`` scores
    the multilayer partition through the same battery every other partition in this package
    goes through.
    """
    matrix, order = supra_adjacency(ml, omega)
    graph: nx.Graph = nx.from_numpy_array(matrix)
    graph = nx.relabel_nodes(graph, dict(enumerate(order)))
    graph.graph["frame"] = (
        f"{ml.frame} Read over the supra-adjacency (Atlas §8.1): one node per (node, layer) "
        f"pair, at omega={(omega if omega is not None else ml.omega):g}."
    )
    return graph


@dataclass(frozen=True)
class OmegaSweepRow:
    omega: float
    modularity: float
    communities: int
    consistency: float


def omega_sweep(
    ml: Multilayer,
    omegas: Sequence[float] = DEFAULT_OMEGA_SWEEP,
    gamma: float = 1.0,
    seed: int | None = None,
) -> list[OmegaSweepRow]:
    """§40.3's omega at a spread of values, so Figure 40.5's effect is read off numbers.

    ``ml.omega`` is always included even when it is not in ``omegas``, so the sweep never omits
    the value the rest of the report used. ``ml`` should already be undirected -- see
    :func:`optimize_multilayer_modularity`.
    """
    values = sorted({*omegas, ml.omega})
    rows = []
    for value in values:
        result = optimize_multilayer_modularity(ml, omega=value, gamma=gamma, seed=seed)
        rows.append(
            OmegaSweepRow(
                omega=value,
                modularity=result.modularity,
                communities=len(result.communities),
                consistency=result.consistency,
            )
        )
    return rows


def null_model_multilayer_modularity(
    ml: Multilayer,
    omega: float | None = None,
    gamma: float = 1.0,
    samples: int = DEFAULT_NULL_SAMPLES,
    seed: int | None = None,
) -> Significance:
    """The observed §40.3 modularity against every layer independently rewired.

    §40.3 states no null model of its own; this is §19.1's own procedure -- Comparing best
    achievable modularity to best achievable modularity, never a fixed partition against a
    resampled graph -- carried across every layer at once, the multilayer generalisation of
    :func:`graphrag.sna.cluster.null_model_modularity`. Each null sample keeps every layer's own
    degree sequence (one :func:`graphrag.sna.null.configuration` rewiring per layer) and leaves
    the identity coupling untouched, because "this row and that row are the same actor" is a fact
    about the node mapping, not something chance could have produced differently.

    ``ml`` should already be undirected -- see :func:`optimize_multilayer_modularity`.
    """
    observed = optimize_multilayer_modularity(ml, omega=omega, gamma=gamma, seed=seed).modularity
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    drawn: list[float] = []
    for _ in range(max(samples, 0)):
        rewired_graphs = {}
        for name in ml.names:
            graph = ml.graphs[name]
            # A layer too small for a swap to have a move (graphrag.sna.null.configuration's own
            # threshold, §19.1) yields nothing; that layer sits out this sample unrewired rather
            # than raising, exactly as it would if it were the only network being tested.
            rewired_graphs[name] = next(
                configuration(graph, 1, seed=rng.randrange(1_000_000)), graph
            )
        rewired = replace(ml, graphs=rewired_graphs)
        drawn.append(
            optimize_multilayer_modularity(
                rewired, omega=omega, gamma=gamma, seed=rng.randrange(1_000_000)
            ).modularity
        )
    return significance(
        observed,
        drawn,
        null="every layer independently degree-preserving rewired, the identity coupling fixed",
    )


# ----------------------------------------------------------------------------- §40.4 density


@dataclass(frozen=True)
class Redundancy:
    """rho_c (§40.4, p. 582): the share of a community's (pair, layer) slots that are filled.

    1.0 means every pair of the community is joined in *every* layer -- "connected through all
    layers at the same time" (p. 581). ``nan`` for a community of fewer than two members or a
    multilayer network with no layers, where the ratio has no denominator.
    """

    community_size: int
    layers: int
    connected_pairs: int
    """Pairs joined in at least one layer -- for reading beside the ratio, not part of it."""
    numerator: int
    denominator: int
    value: float


def redundancy(ml: Multilayer, community: Iterable[str]) -> Redundancy:
    """rho_c = sum_{u,v in Pc} |{l : (u,v,l) in E}| / (|L| * |Pc|) (§40.4).

    Every node of ``community`` that is not in ``ml.nodes`` is dropped silently -- the same
    contract :func:`graphrag.sna.evaluate.evaluate_partition` uses for a partition that does not
    cover the network.
    """
    members = sorted(set(community) & set(ml.nodes), key=str)
    layers = len(ml.names)
    if len(members) < 2 or layers == 0:
        return Redundancy(len(members), layers, 0, 0, 0, math.nan)
    pairs = list(combinations(members, 2))
    numerator = sum(1 for u, v in pairs for name in ml.names if ml.graphs[name].has_edge(u, v))
    connected = sum(1 for u, v in pairs if any(ml.graphs[name].has_edge(u, v) for name in ml.names))
    denominator = layers * len(pairs)
    return Redundancy(
        len(members), layers, connected, numerator, denominator, numerator / denominator
    )


@dataclass(frozen=True)
class Complementarity:
    """variety x exclusivity x homogeneity (§40.4, p. 582): the "spread across layers" reading.

    ``nan`` for a community of fewer than two members or a network of fewer than two layers,
    where variety has no denominator. See :data:`DEPARTURE_HOMOGENEITY` for the one term this
    module could not reproduce from the chapter's own worked example.
    """

    community_size: int
    layers_present: int
    layers_total: int
    variety: float
    exclusivity: float
    homogeneity: float
    value: float
    edge_counts: dict[str, int]


def complementarity(ml: Multilayer, community: Iterable[str]) -> Complementarity:
    """The three terms of §40.4's complementarity, each computed from the same pair counts.

    *Variety*: ``(|Lc|-1) / (|L|-1)``, the share of the network's layers this community actually
    uses. *Exclusivity*: the share of the community's pairs joined in exactly one layer, over
    every possible pair (joined or not). *Homogeneity*: 1 minus the standard deviation of the
    per-layer edge counts over its theoretical maximum -- see :data:`DEPARTURE_HOMOGENEITY`.
    """
    members = sorted(set(community) & set(ml.nodes), key=str)
    layers_total = len(ml.names)
    if len(members) < 2 or layers_total < 2:
        return Complementarity(
            len(members), 0, layers_total, math.nan, math.nan, math.nan, math.nan, {}
        )
    pairs = list(combinations(members, 2))
    total_pairs = len(pairs)
    edge_counts = {
        name: sum(1 for u, v in pairs if ml.graphs[name].has_edge(u, v)) for name in ml.names
    }
    present = [name for name, count in edge_counts.items() if count > 0]
    exclusive_pairs = sum(
        1 for u, v in pairs if sum(1 for name in ml.names if ml.graphs[name].has_edge(u, v)) == 1
    )
    variety = (len(present) - 1) / (layers_total - 1)
    exclusivity = exclusive_pairs / total_pairs if total_pairs else math.nan
    counts = list(edge_counts.values())
    total_edges = sum(counts)
    sigma = float(np.std(counts))
    sigma_max = total_edges * math.sqrt(layers_total - 1) / layers_total if total_edges else 0.0
    homogeneity = 1.0 if sigma_max == 0 else 1.0 - sigma / sigma_max
    value = variety * exclusivity * homogeneity if total_pairs else math.nan
    return Complementarity(
        len(members),
        len(present),
        layers_total,
        variety,
        exclusivity,
        homogeneity,
        value,
        edge_counts,
    )


def density_table(
    ml: Multilayer, communities: Sequence[Sequence[str]]
) -> list[tuple[Redundancy, Complementarity]]:
    """Redundancy and complementarity for every community of a partition, in the order given."""
    return [
        (redundancy(ml, community), complementarity(ml, community)) for community in communities
    ]


# ----------------------------------------------------------------------------- §40.5 which to use


@dataclass(frozen=True)
class LayerAgreement:
    """§40.5's mopping-up rule, made numeric: do the layers agree enough to flatten?"""

    pairs: tuple[tuple[str, str, float, int], ...]
    """``(layer_a, layer_b, adjusted_rand_index, shared_nodes)`` for every pair of layers that
    shared at least two nodes with a Louvain community each."""
    mean_agreement: float
    recommended: str
    """``flatten`` or ``layer-by-layer`` -- never a third value, since supra modularity at a low
    omega is the numeric equivalent of layer-by-layer and at a high one of flattening (§40.3's
    own "pillar" versus "flat" language), so the recommendation is read onto whichever omega an
    analyst chooses rather than naming a fourth method."""
    reason: str


def layer_agreement(
    ml: Multilayer,
    resolution: float = 1.0,
    seed: int | None = None,
    runs: int = 10,
) -> LayerAgreement:
    """Score whether the layers' own communities agree, and recommend a method from the answer.

    Every method that combines layers -- flattening, ensemble merging, a large interlayer
    coupling -- rests on the assumption §40.2 states outright: "these methods have the downside
    of relying more or less on the same assumption: that the layers are correlated to each
    other... disassortative layers exist and might represent a problem" (p. 575). This runs
    Louvain on each layer separately and scores every pair of layers by the adjusted Rand index
    of their communities over the nodes both layers share (§3.5) -- the same agreement chapter 3
    defines for any two partitions. :data:`AGREEMENT_HIGH` is this module's own cutoff for
    "agree enough", not the chapter's: the chapter states the axis and refuses to fix a number
    on it, so this module fixes one and says it did.
    """
    results = {
        name: _safe_louvain(ml.graphs[name], resolution=resolution, seed=seed, runs=runs)
        for name in ml.names
    }
    rows: list[tuple[str, str, float, int]] = []
    for a, b in combinations(ml.names, 2):
        shared = sorted(set(ml.graphs[a].nodes) & set(ml.graphs[b].nodes), key=str)
        if len(shared) < 2:
            continue
        ari = compare_partitions(results[a].labels(shared), results[b].labels(shared))[
            "adjusted_rand_index"
        ]
        rows.append((a, b, ari, len(shared)))

    mean_agreement = mean([ari for _, _, ari, _ in rows]) if rows else math.nan
    if not rows:
        recommended, reason = (
            "layer-by-layer",
            (
                "no two layers shared two or more nodes to compare, so nothing here measures "
                "whether the layers agree; layer-by-layer with matching is the safer default "
                "because it never assumes they do."
            ),
        )
    elif mean_agreement >= AGREEMENT_HIGH:
        recommended, reason = (
            "flatten",
            (
                f"the layers' own Louvain partitions agree at a mean adjusted Rand index of "
                f"{mean_agreement:.3f}, at or above this module's cutoff of {AGREEMENT_HIGH:g}, so "
                "flattening's cost -- treating every layer as equally important -- is small: the "
                "layers were largely already saying the same thing."
            ),
        )
    else:
        recommended, reason = (
            "layer-by-layer",
            (
                f"the layers' own Louvain partitions agree at a mean adjusted Rand index of only "
                f"{mean_agreement:.3f}, below this module's cutoff of {AGREEMENT_HIGH:g}. §40.2's "
                "own warning applies: flattening, or a large interlayer coupling, would force "
                "these disagreeing layers into one answer and hide exactly the disagreement "
                "that is here to read. Layer-by-layer with matching, or the supra-adjacency "
                "modularity at a low omega, keeps each layer's own communities visible."
            ),
        )
    return LayerAgreement(tuple(rows), mean_agreement, recommended, reason)


# ----------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class MultilayerCommunityReport:
    """§40.1-§40.5 together, over one :class:`graphrag.sna.layers.Multilayer` network."""

    frame: str
    flattened: nx.Graph
    flatten_result: LouvainResult
    flatten_null: NullModelResult
    flatten_scores: PartitionScores
    layer_by_layer: LayerByLayerResult
    layer_by_layer_scores: PartitionScores | None
    supra: SupraModularityResult
    supra_graph: nx.Graph
    supra_scores: PartitionScores
    supra_null: Significance
    sweep: list[OmegaSweepRow]
    density: list[tuple[Redundancy, Complementarity]]
    agreement: LayerAgreement
    notes: list[str] = field(default_factory=list)


def analyze_multilayer_communities(
    ml: Multilayer,
    *,
    threshold: int = DEFAULT_MIN_SHARED,
    omega: float | None = None,
    gamma: float = 1.0,
    resolution: float = 1.0,
    runs: int = 10,
    samples: int = DEFAULT_NULL_SAMPLES,
    omegas: Sequence[float] = DEFAULT_OMEGA_SWEEP,
    seed: int | None = None,
) -> MultilayerCommunityReport:
    """Run §40.1, §40.2 and §40.3 over ``ml``, score every one of them, and add §40.4 and §40.5.

    The layer-by-layer partition (§40.2) is evaluated against the *flattened* graph, not against
    any one layer, on purpose: it gives every method the same nodes, the same edges and the same
    modularity denominator, so the numbers in this report are directly comparable to each other.
    A layer's own fit to its own communities is already printed above it, in that layer's Louvain
    modularity from :attr:`LayerByLayerResult.per_layer`.

    §40.4's density is read on the flattening's own communities: it is the one partition every
    other method in this report can be compared with, and multilayer density is defined on any
    community regardless of which method found it -- :func:`density_table` takes any partition.
    """
    notes: list[str] = []
    flat = flatten(ml)
    flatten_result = _safe_louvain(flat, resolution=resolution, seed=seed, runs=runs)
    if flatten_result.note:
        notes.append(flatten_result.note)
    flatten_null = null_model_modularity(
        flat, flatten_result.communities, samples=samples, seed=seed, resolution=resolution
    )
    flatten_scores = evaluate_partition(
        flat, flatten_result.communities, resolution=resolution, seed=seed
    )

    lbl = layer_by_layer_communities(
        ml, threshold=threshold, resolution=resolution, seed=seed, runs=runs
    )
    partition = lbl.partition
    lbl_scores = (
        evaluate_partition(flat, partition, resolution=resolution, seed=seed) if partition else None
    )
    if not partition:
        notes.append("Layer-by-layer with matching found no multilayer community to evaluate.")

    undirected_ml, directed_note = _undirected_multilayer(ml)
    if directed_note:
        notes.append(directed_note)
    coupling = ml.omega if omega is None else omega
    supra_result = optimize_multilayer_modularity(
        undirected_ml, omega=coupling, gamma=gamma, seed=seed
    )
    supra_graph = supra_network(undirected_ml, coupling)
    supra_scores = evaluate_partition(
        supra_graph, supra_result.communities, resolution=resolution, seed=seed
    )
    supra_null = null_model_multilayer_modularity(
        undirected_ml, omega=coupling, gamma=gamma, samples=samples, seed=seed
    )
    sweep = omega_sweep(undirected_ml, omegas=omegas, gamma=gamma, seed=seed)

    density = density_table(ml, flatten_result.communities)
    agreement = layer_agreement(ml, resolution=resolution, seed=seed, runs=runs)

    return MultilayerCommunityReport(
        frame=ml.frame,
        flattened=flat,
        flatten_result=flatten_result,
        flatten_null=flatten_null,
        flatten_scores=flatten_scores,
        layer_by_layer=lbl,
        layer_by_layer_scores=lbl_scores,
        supra=supra_result,
        supra_graph=supra_graph,
        supra_scores=supra_scores,
        supra_null=supra_null,
        sweep=sweep,
        density=density,
        agreement=agreement,
        notes=notes,
    )


# ----------------------------------------------------------------------------- rendering


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


def render_multilayer_communities(report: MultilayerCommunityReport) -> list[str]:
    """The whole of chapter 40 as one markdown report, over one multilayer network."""
    lines: list[str] = [
        "## Multilayer community discovery",
        "",
        f"**Implements.** Atlas {CHAPTER} -- flattening, layer-by-layer with cross-layer "
        "matching, multilayer modularity over the supra-adjacency, multilayer density, and the "
        "chapter's own closing rule on which of them to trust.",
        "",
        f"**Sampling frame.** {report.frame}",
        "",
        f"n = {report.flattened.number_of_nodes():,} nodes, "
        f"{report.flattened.number_of_edges():,} edges once flattened, over "
        f"{report.flattened.graph.get('layers', '')!s}.",
        "",
    ]

    lines += [
        "### Flattening (§40.1)",
        "",
        f"- {report.flatten_result.communities and len(report.flatten_result.communities)} "
        f"communities, modularity {_num(report.flatten_result.modularity)}, stability (ARI) "
        f"{_num(report.flatten_result.stability, 3)} over {len(report.flatten_result.seeds)} seeds",
        f"- null model: degree-preserving rewiring, z = {_num(report.flatten_null.z_score, 2)} "
        f"({report.flatten_null.verdict})",
        "",
    ]
    lines += render_evaluation(report.flatten_scores)

    lines += ["### Layer by layer with cross-layer matching (§40.2)", ""]
    lines += _table(
        ["layer", "communities", "modularity", "stability (ARI)"],
        [
            [
                name,
                str(len(result.communities)),
                _num(result.modularity),
                _num(result.stability, 3),
            ]
            for name, result in report.layer_by_layer.per_layer.items()
        ],
    )
    lbl = report.layer_by_layer
    lines += [
        f"{len(lbl.layer_communities)} per-layer communities, merged at threshold "
        f"{lbl.threshold} shared node(s) into {len(lbl.multilayer_communities)} multilayer "
        f"community(-ies); {len(lbl.unmerged)} never merged with anything.",
        "",
    ]
    if lbl.multilayer_communities:
        lines += _table(
            ["#", "constituents", "members"],
            [
                [str(i + 1), ", ".join(c.label for c in mlc.constituents), str(len(mlc.members))]
                for i, mlc in enumerate(lbl.multilayer_communities)
            ],
        )
    lines += [
        f"{len(lbl.unaffiliated)} node(s) belong to no multilayer community (Figure 40.2's own "
        "'weird combination' case, left unaffiliated rather than forced into one).",
        "",
        f"> {lbl.overlap_note}",
        "",
    ]
    if report.layer_by_layer_scores is not None:
        lines += render_evaluation(report.layer_by_layer_scores)
    else:
        lines += ["Nothing to evaluate: the matched partition is empty.", ""]

    lines += [
        "### Multilayer modularity over the supra-adjacency (§40.3)",
        "",
        f"- omega = {report.supra.omega:g}, gamma = {report.supra.gamma:g}: "
        f"{len(report.supra.communities)} communities, modularity "
        f"{_num(report.supra.modularity, 4)}",
        f"- consistency (share of nodes in the same community in every layer, 1.0 = fully "
        f"'pillar', Figure 40.5a): {_num(report.supra.consistency, 3)}",
        f"- {report.supra.iterations} local-moving pass(es); this is single-level, without "
        "GenLouvain's hierarchical aggregation -- see optimize_multilayer_modularity",
        f"- null model: {report.supra_null.null}, {report.supra_null.samples} sample(s) -- "
        f"z = {_num(report.supra_null.z, 2)}, p = {_num(report.supra_null.p_value, 3)}",
        "",
        "**The omega sweep (§40.3, Figure 40.5).** Interlayer coupling is a parameter, not a "
        "measurement; this is its effect on the same network.",
        "",
    ]
    lines += _table(
        ["omega", "modularity", "communities", "consistency"],
        [
            [
                f"{row.omega:g}",
                _num(row.modularity, 4),
                str(row.communities),
                _num(row.consistency, 3),
            ]
            for row in report.sweep
        ],
    )
    lines += render_evaluation(report.supra_scores)

    lines += [
        "### Multilayer density (§40.4)",
        "",
        "Two readings of the same flattening communities, over the layers rather than the "
        "flattened edges: redundancy asks whether a community's pairs are joined through "
        "*every* layer; complementarity asks whether they are joined through *many*, each "
        "mostly its own.",
        "",
    ]
    lines += _table(
        ["#", "size", "redundancy", "variety", "exclusivity", "homogeneity", "complementarity"],
        [
            [
                str(i + 1),
                str(red.community_size),
                _num(red.value, 3),
                _num(comp.variety, 3),
                _num(comp.exclusivity, 3),
                _num(comp.homogeneity, 3),
                _num(comp.value, 3),
            ]
            for i, (red, comp) in enumerate(report.density)
        ],
    )
    lines += [f"> {DEPARTURE_HOMOGENEITY}", ""]

    lines += [
        "### Which to use (§40.5)",
        "",
        "\"You can intend 'communities' in complex networks in a thousand different ways\" -- "
        "this is not a verdict, only whether the layers agree enough for the methods above that "
        "assume they do.",
        "",
    ]
    lines += _table(
        ["layer a", "layer b", "adjusted Rand index", "shared nodes"],
        [[a, b, _num(ari, 3), str(n)] for a, b, ari, n in report.agreement.pairs],
    )
    lines += [
        f"mean agreement: {_num(report.agreement.mean_agreement, 3)}; recommended: "
        f"**{report.agreement.recommended}**",
        "",
        f"> {report.agreement.reason}",
        "",
    ]
    if report.notes:
        lines += [f"> {note}" for note in report.notes] + [""]
    return lines


def multilayer_communities_payload(report: MultilayerCommunityReport) -> dict[str, object]:
    """The report as plain JSON-able data. ``nan`` becomes ``null``, never 0."""

    def _clean(value: float) -> float | None:
        return None if math.isnan(value) else value

    return {
        "implements": f"Atlas {CHAPTER} (multilayer community discovery)",
        "frame": report.frame,
        "flattening": {
            "communities": len(report.flatten_result.communities),
            "modularity": report.flatten_result.modularity,
            "stability": report.flatten_result.stability,
            "null_z": report.flatten_null.z_score,
            "null_verdict": report.flatten_null.verdict,
        },
        "layer_by_layer": {
            "threshold": report.layer_by_layer.threshold,
            "per_layer_communities": {
                name: len(result.communities)
                for name, result in report.layer_by_layer.per_layer.items()
            },
            "multilayer_communities": len(report.layer_by_layer.multilayer_communities),
            "unmerged": len(report.layer_by_layer.unmerged),
            "unaffiliated": len(report.layer_by_layer.unaffiliated),
        },
        "supra_modularity": {
            "omega": report.supra.omega,
            "gamma": report.supra.gamma,
            "communities": len(report.supra.communities),
            "modularity": report.supra.modularity,
            "consistency": _clean(report.supra.consistency),
            "null_z": report.supra_null.z,
            "null_p": _clean(report.supra_null.p_value),
            "sweep": [
                {
                    "omega": row.omega,
                    "modularity": row.modularity,
                    "communities": row.communities,
                    "consistency": _clean(row.consistency),
                }
                for row in report.sweep
            ],
        },
        "density": [
            {
                "size": red.community_size,
                "redundancy": _clean(red.value),
                "variety": _clean(comp.variety),
                "exclusivity": _clean(comp.exclusivity),
                "homogeneity": _clean(comp.homogeneity),
                "complementarity": _clean(comp.value),
            }
            for red, comp in report.density
        ],
        "recommendation": {
            "mean_agreement": _clean(report.agreement.mean_agreement),
            "recommended": report.agreement.recommended,
            "reason": report.agreement.reason,
        },
        "notes": report.notes,
    }
