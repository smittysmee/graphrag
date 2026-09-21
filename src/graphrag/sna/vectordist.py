"""How far apart are two node vectors on the same network? (Atlas ch. 47, pp. 679-696)

Chapter 13 answers "how far apart are two *nodes*". This chapter answers a different question:
given a network and two vectors ``p`` and ``q`` that each put some amount of value on every
node -- praise counts, mentions from a region, a window's degree -- how far apart are the
*vectors*? "Is the total red hue distributed closer to the blue color, or to the green? How much
does it matter that the darker nodes carry more weight" (p. 679).

**The resolver.** A vector has to come from somewhere. :func:`resolve_vector` reads a small spec
language rather than a raw array, because every number this package can put on a node already has
a name and a provenance, and inventing a second way to hand one in would let a caller smuggle in
a vector nobody can trace back to a source:

- ``stance:<praise|complaint|substitution|neutral>`` -- annotated mention counts per entity
  (:mod:`graphrag.sna.stances`), only on the ``entities`` network.
- ``attr:<key>`` -- a numeric node attribute or built-in count (mentions, documents, chunks,
  degree), the same reading :func:`graphrag.sna.assortativity.numeric_values` gives ``--by``.
- ``centrality:<kind>`` -- one of :func:`graphrag.sna.measures.centrality`'s rankings.
- ``window:<since>:<until>:<kind>`` -- a centrality computed on one time window's network
  (:func:`graphrag.sna.layers.window_graph`), read as an occupancy snapshot: a node the window
  did not reach carries 0, not "no data".

**What is built, section by section.**

§47.1 (non-network baselines)
    :func:`euclidean_distance`, :func:`cosine_distance`, :func:`correlation_distance`: plain
    vector distances that know nothing about the graph, kept here as the thing the rest of the
    chapter improves on.

§47.2 (generalized Euclidean)
    :func:`generalized_euclidean_distance` replaces the identity matrix of a Euclidean distance
    with the Moore-Penrose pseudoinverse of the graph Laplacian (:func:`graphrag.sna.walks.
    laplacian_pseudoinverse`), so two nodes joined by an edge count as close and two nodes with
    no path between them do not collapse onto "equally far" the way plain Euclidean does. Built:
    the Laplacian route of the three the chapter names. The Markov Chain and Annihilation routes
    are documented, not built -- see the module's closing report for why.

§47.3 (shortest-path based)
    :func:`shortest_path_linkage_distance` (single, complete, average linkage, the chapter's
    "non-optimized" family) and :func:`earth_mover_distance` (the "optimized" family, solved
    exactly as a transportation linear program with HiGHS through ``scipy.optimize.linprog`` --
    the same solver :func:`graphrag.sna.hierarchy.agony` uses for a different LP). Both rescale
    the lighter vector up first, the rule p. 688 states, and both use hop counts, the chapter's
    own "number of edges separating node u to v" (p. 686).

§47.4 (graph Fourier transform)
    :func:`graph_fourier_distance` -- ``‖L(p-q)‖₂``, p. 692's own formula worked out in full in
    its docstring. :func:`graph_fourier_smoothness` -- the Laplacian quadratic form ``xᵀLx``, the
    Dirichlet energy -- is **not** ch. 47's formula; it is kept as a per-vector diagnostic a
    report can print beside the distance, labelled plainly as borrowed from the wider
    graph-signal-processing literature rather than as something the chapter itself defines.

§47.5 (network-space statistics)
    :func:`network_variance` (p. 693's ``var(x) = ½ Σ x_u x_v d²_uv``, with ``d`` read as
    effective resistance, the chapter's own preference over shortest paths because it is a
    proper metric) and :func:`network_correlation`, a cosine similarity in the inner product the
    Laplacian pseudoinverse defines -- the book states no closed form for this one, and neither
    mean-centres its vector nor sits on the same scale as ``network_variance``'s pairwise-Ω sum;
    see its own docstring for the disclosure this choice needs.

Every function here takes arrays or mappings and a graph; none of them read a persona, a store,
or print anything. ``graphrag.cli``'s ``sna distance`` command is the only caller that resolves a
spec, builds the network, and renders a report.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array

from graphrag.graph.store import GraphStore
from graphrag.sna.matrices import Node, dense, laplacian, node_order
from graphrag.sna.stats import pearson
from graphrag.sna.walks import MAX_NODES, laplacian_pseudoinverse, resistance_matrix

__all__ = [
    "CHAPTER",
    "LINKAGES",
    "MAX_NODES",
    "MAX_SUPPORT",
    "SPEC_HELP",
    "Baselines",
    "GeneralizedEuclidean",
    "NetworkStatistics",
    "NodeVector",
    "ShortestPathDistances",
    "Smoothness",
    "VectorDistanceReport",
    "baseline_distances",
    "compare_vectors",
    "correlation_distance",
    "cosine_distance",
    "distance_payload",
    "earth_mover_distance",
    "euclidean_distance",
    "generalized_euclidean_distance",
    "graph_fourier_distance",
    "graph_fourier_smoothness",
    "network_correlation",
    "network_variance",
    "render_distance",
    "resolve_vector",
    "shortest_path_linkage_distance",
]

#: What every section of the report cites.
CHAPTER = (
    "Atlas ch. 47 (§47.1 non-network baselines, §47.2 generalized Euclidean, §47.3 shortest-path "
    "and earth-mover, §47.4 graph Fourier transform, §47.5 network-space statistics)"
)

#: The three aggregations of §47.3's "non-optimized" shortest-path family.
LINKAGES: tuple[str, ...] = ("single", "complete", "average")

#: A linkage or an earth-mover LP has one candidate, or one LP variable, per (support-of-p,
#: support-of-q) pair, so both cost grows with the *product* of the two supports. A vector like
#: a centrality ranking is nonzero on almost every node, so this refuses rather than let a report
#: silently take minutes: narrow the vectors (an attribute few nodes carry, an annotated stance)
#: or raise this deliberately.
MAX_SUPPORT: int = 60

#: Below this, ``p_u * q_v`` is treated as "no mass here" rather than as a rounding artefact of
#: the greedy exchange in :func:`shortest_path_linkage_distance`.
_EPS = 1e-9

SPEC_HELP = (
    "a vector spec must be one of: 'stance:<praise|complaint|substitution|neutral>' (annotated "
    "mention counts, --network entities only); 'attr:<key>' (a numeric node attribute or a "
    "built-in count -- mentions, documents, chunks, degree); 'centrality:<kind>' (one of "
    "graphrag.sna.measures.centrality's rankings); or "
    "'window:<since>:<until>:<kind>' (a centrality computed on one time window's network, ISO "
    "dates, e.g. window:2025-01-01:2025-06-30:weighted_degree)"
)


# ----------------------------------------------------------------------------- vector resolution


@dataclass(frozen=True)
class NodeVector:
    """One vector, resolved from a spec, before it is aligned to a node order.

    ``values`` holds only the nodes this vector's source actually spoke about; a node absent from
    it is not "zero by observation", it is a node the source said nothing about, and
    :meth:`array` fills it with 0.0 as the occupancy reading of ch. 47 does ("if I occupy nodes 1
    and 2..."). :meth:`present` counts how many of a network's nodes were spoken about at all, so
    a report can say how much of the 0.0s are silence rather than measurement.
    """

    spec: str
    values: dict[Node, float]
    source: str

    def array(self, order: Sequence[Node]) -> np.ndarray:
        """This vector's values, aligned to ``order``, absent nodes filled with 0.0."""
        return np.array([float(self.values.get(node, 0.0)) for node in order], dtype=np.float64)

    def present(self, order: Sequence[Node]) -> int:
        """How many nodes of ``order`` this vector's source actually gave a value for."""
        return sum(1 for node in order if node in self.values)


def resolve_vector(
    spec: str,
    graph: nx.Graph,
    *,
    store: GraphStore | None = None,
    persona_id: str | None = None,
    network: str = "entities",
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
) -> NodeVector:
    """Read one vector spec (see :data:`SPEC_HELP`) into a :class:`NodeVector` on ``graph``.

    ``store`` and ``persona_id`` are needed by ``stance:`` and ``window:``, which read the
    annotation layer or rebuild a window's network; ``attr:`` and ``centrality:`` need only
    ``graph`` and work on any graph this package or a caller built. Raises ``ValueError`` -- with
    :data:`SPEC_HELP` in the message -- for anything the spec language does not name.
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "stance" and len(parts) == 2:
        return _resolve_stance(
            parts[1], network=network, store=store, persona_id=persona_id, source_id=source_id
        )
    if kind == "attr" and len(parts) == 2:
        return _resolve_attribute(parts[1], graph)
    if kind == "centrality" and len(parts) == 2:
        return _resolve_centrality(parts[1], graph)
    if kind == "window" and len(parts) == 4:
        return _resolve_window(
            since=parts[1],
            until=parts[2],
            centrality_kind=parts[3],
            store=store,
            persona_id=persona_id,
            network=network,
            source_id=source_id,
            min_weight=min_weight,
            types=types,
            facets=facets,
            where=where,
        )
    msg = f"{spec!r} is not a vector spec this module reads. {SPEC_HELP}"
    raise ValueError(msg)


def _resolve_stance(
    stance: str,
    *,
    network: str,
    store: GraphStore | None,
    persona_id: str | None,
    source_id: str | None,
) -> NodeVector:
    from graphrag.sna.export import STANCES
    from graphrag.sna.stances import build_stance_report

    if stance not in STANCES:
        msg = f"stance:{stance} is not one of the annotation layer's stances: {', '.join(STANCES)}"
        raise ValueError(msg)
    if network != "entities":
        msg = (
            "stance:<name> is counted per entity (graphrag.sna.stances), so it only makes sense "
            f"on --network entities; this comparison asked for --network {network}"
        )
        raise ValueError(msg)
    if store is None or persona_id is None:
        msg = "stance:<name> needs a persona and a store to read the annotation layer"
        raise ValueError(msg)
    report = build_stance_report(store, persona_id, source_id=source_id)
    values = {entity.entity_id: float(entity.counts.get(stance, 0)) for entity in report.entities}
    return NodeVector(
        spec=f"stance:{stance}",
        values=values,
        source=(
            f"annotated {stance} counts per entity (graphrag.sna.stances), "
            f"{report.annotations:,} annotations read"
        ),
    )


def _resolve_attribute(key: str, graph: nx.Graph) -> NodeVector:
    from graphrag.sna.assortativity import numeric_values

    values = numeric_values(graph, key)
    if not values:
        msg = (
            f"attr:{key} has no numeric value on any node of this network: check the key against "
            "the persona's facets.yaml, or use one of the built-in counts (mentions, documents, "
            "chunks, degree)"
        )
        raise ValueError(msg)
    return NodeVector(
        spec=f"attr:{key}",
        values=values,
        source=f"the numeric attribute {key!r} (graphrag.sna.assortativity.numeric_values)",
    )


def _resolve_centrality(kind: str, graph: nx.Graph) -> NodeVector:
    from graphrag.sna.measures import centrality

    values = centrality(graph, kind)  # raises ValueError naming the valid kinds itself
    return NodeVector(
        spec=f"centrality:{kind}",
        values={node: float(value) for node, value in values.items()},
        source=f"the {kind} centrality (graphrag.sna.measures.centrality)",
    )


def _resolve_window(
    *,
    since: str,
    until: str,
    centrality_kind: str,
    store: GraphStore | None,
    persona_id: str | None,
    network: str,
    source_id: str | None,
    min_weight: int | None,
    types: Sequence[str] | None,
    facets: Sequence[str] | None,
    where: Mapping[str, str] | None,
) -> NodeVector:
    from graphrag.sna.layers import window_graph
    from graphrag.sna.measures import centrality

    if store is None or persona_id is None:
        msg = "window:<since>:<until>:<kind> needs a persona and a store to build the window"
        raise ValueError(msg)
    window = window_graph(
        store,
        persona_id,
        network,
        since=since or None,
        until=until or None,
        source_id=source_id,
        min_weight=min_weight,
        types=types,
        facets=facets,
        where=where,
    )
    values = centrality(window, centrality_kind)
    return NodeVector(
        spec=f"window:{since}:{until}:{centrality_kind}",
        values={node: float(value) for node, value in values.items()},
        source=(
            f"the {centrality_kind} centrality on the {since}..{until} window of the {network} "
            f"network (§7.4), {window.number_of_nodes():,} nodes reached; a node this window "
            "never touched carries 0.0, read as occupancy rather than as missing data"
        ),
    )


# ----------------------------------------------------------------------------- §47.1 baselines


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    """The Euclidean distance ``((p-q)ᵀ(p-q))^½`` (p. 680): every dimension counts equally.

    The measure §47.2 improves on: it cannot tell "mass moved between two far-apart nodes" from
    "mass moved between two adjacent ones", because it has no idea the nodes are in a network at
    all -- see :func:`generalized_euclidean_distance`.
    """
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


def cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    """``1 - cos(angle between p and q)`` (p. 682): direction only, magnitude ignored.

    Two points on the same ray from the origin are at distance 0 "even if they're infinitely
    farther apart" by Euclidean distance (p. 682): this is p. 682's own example of a distance
    that does not respect the triangle inequality, which the chapter uses to make the point that
    a "distance" need not be a metric in the strict sense.

    Undefined, and raised on, when either vector is the zero vector: there is no angle to a point
    with no direction.
    """
    left, right = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        msg = (
            "cosine_distance is undefined for a zero vector: there is no angle to a point with "
            "no direction"
        )
        raise ValueError(msg)
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 - max(-1.0, min(1.0, cosine))  # clamp float error outside [-1, 1]


@dataclass(frozen=True)
class CorrelationDistance:
    """A baseline correlation distance (§47.1), with the coefficient it was built from."""

    distance: float
    coefficient: float
    p_value: float
    n: int


def correlation_distance(p: np.ndarray, q: np.ndarray) -> CorrelationDistance:
    """``1 - pearson(p, q)`` (§47.1, "correlation distances", p. 682, no formula given).

    Reuses :func:`graphrag.sna.stats.pearson`, so the p-value tests the same null it always
    does -- "these two are independent" -- and the same failure modes apply: fewer than 3 nodes,
    or either vector constant across the network, both raise rather than return a number.
    """
    result = pearson(list(p), list(q))
    return CorrelationDistance(
        distance=1.0 - result.coefficient,
        coefficient=result.coefficient,
        p_value=result.p_value,
        n=result.n,
    )


@dataclass(frozen=True)
class Baselines:
    """Every §47.1 distance that ran, and why one that did not."""

    euclidean: float
    cosine: float | None
    cosine_note: str
    correlation: CorrelationDistance | None
    correlation_note: str


def baseline_distances(p: np.ndarray, q: np.ndarray) -> Baselines:
    """All three §47.1 baselines at once, catching each one's own undefined case."""
    cosine: float | None
    cosine_note = ""
    try:
        cosine = cosine_distance(p, q)
    except ValueError as exc:
        cosine, cosine_note = None, str(exc)
    correlation: CorrelationDistance | None
    correlation_note = ""
    try:
        correlation = correlation_distance(p, q)
    except ValueError as exc:
        correlation, correlation_note = None, str(exc)
    return Baselines(
        euclidean=euclidean_distance(p, q),
        cosine=cosine,
        cosine_note=cosine_note,
        correlation=correlation,
        correlation_note=correlation_note,
    )


# ------------------------------------------------------------------- §47.2 generalized Euclidean


def generalized_euclidean_distance(
    pseudoinverse: np.ndarray, p: np.ndarray, q: np.ndarray
) -> float:
    """``((p-q)ᵀ L† (p-q))^½`` (§47.2, p. 684): Euclidean distance warped by the graph's topology.

    ``pseudoinverse`` is ``L†``, the Moore-Penrose pseudoinverse of the combinatorial Laplacian
    -- built by :func:`graphrag.sna.walks.laplacian_pseudoinverse`, the same primitive
    :func:`graphrag.sna.walks.effective_resistance` inverts to get ``Ω``. For two indicator
    vectors ``p = e_u``, ``q = e_v`` **in the same connected component** this is exactly
    ``√Ω(u,v)``: the more paths and the shorter they are between two nodes, the smaller the
    warped distance between "all the mass on u" and "all the mass on v", which is precisely
    §47.2's motivating example (Figure 47.5) -- a Euclidean distance cannot tell a network's
    neighbours from its strangers, and this can.

    ``L†`` is block-diagonal across connected components (:func:`graphrag.sna.walks.
    laplacian_pseudoinverse`'s own contract): a *cross* term ``L†[u,v]`` between two nodes in
    different components is 0, never a large or infinite one. That does **not** mean two
    vectors supported on different components read as 0 apart -- each side still contributes its
    own diagonal term ``L†[u,u]``, which is generally positive (it is how far ``u`` sits from its
    own component's "centre" in the warped space), so the distance is the sum of the two sides'
    own spread rather than anything resembling ``Ω(u,v)`` (which :func:`graphrag.sna.walks.
    resistance_matrix` reports as ``inf`` for the same pair). Read a number here between two
    different components as "how spread out each side's own mass is", not as "how far apart the
    two components are" -- the warped space has no route between them to measure that with.

    A tiny negative value from floating-point error in an otherwise non-negative quadratic form
    (``L†`` is positive semi-definite) is clamped to 0 before the square root.
    """
    diff = np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    value = float(diff @ pseudoinverse @ diff)
    return math.sqrt(max(value, 0.0))


@dataclass(frozen=True)
class GeneralizedEuclidean:
    """§47.2's distance next to the plain Euclidean it replaces, so a report can compare them."""

    value: float
    plain_euclidean: float
    components: int
    max_nodes: int


# ----------------------------------------------------------- §47.3 shortest-path and earth-mover


def _support(values: Mapping[Node, float]) -> dict[Node, float]:
    """The nonzero entries of a vector, the only ones §47.3's mass-moving methods read."""
    return {node: float(value) for node, value in values.items() if value > 0}


def _rescale_to_equal_mass(
    p: Mapping[Node, float], q: Mapping[Node, float]
) -> tuple[dict[Node, float], dict[Node, float]]:
    """Scale the lighter vector up so ``Σp = Σq`` (p. 688's rule), both already positive-mass."""
    sum_p, sum_q = sum(p.values()), sum(q.values())
    if sum_p <= 0 or sum_q <= 0:
        msg = "both vectors need positive total mass to compare how it is distributed"
        raise ValueError(msg)
    if math.isclose(sum_p, sum_q):
        return dict(p), dict(q)
    if sum_p < sum_q:
        scale = sum_q / sum_p
        return {node: value * scale for node, value in p.items()}, dict(q)
    scale = sum_p / sum_q
    return dict(p), {node: value * scale for node, value in q.items()}


def _refuse_oversized_support(
    p: Mapping[Node, float], q: Mapping[Node, float], max_support: int
) -> None:
    if len(p) > max_support or len(q) > max_support:
        msg = (
            f"this method builds one candidate per (p, q) support pair; got {len(p)} x {len(q)} "
            f"nonzero nodes, above max_support={max_support}. Narrow the vectors to fewer "
            "nonzero nodes (an annotated stance, a tagged attribute) or raise max_support "
            "deliberately"
        )
        raise ValueError(msg)


def _hop_distances(
    graph: nx.Graph, sources: Sequence[Node], targets: Sequence[Node]
) -> dict[tuple[Node, Node], float]:
    """Shortest-path length in hops from every source to every target (§47.3's ``|P_u,v|``).

    Only the pairs the two supports actually need, never the whole network's distance matrix --
    §47.3 names this convention explicitly (p. 686) to avoid solving all-pairs shortest paths
    when most of it is never read. ``inf`` when no path joins the two: no current network method
    here raises on that, because a disconnected network is a legitimate answer, not a mistake.
    """
    distances: dict[tuple[Node, Node], float] = {}
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target in targets:
            distances[(source, target)] = float(lengths[target]) if target in lengths else math.inf
    return distances


def shortest_path_linkage_distance(
    graph: nx.Graph,
    p: Mapping[Node, float],
    q: Mapping[Node, float],
    linkage: Literal["single", "complete", "average"] = "single",
    *,
    max_support: int = MAX_SUPPORT,
) -> float:
    """The "non-optimized" shortest-path family of §47.3 (p. 687-688), one of three linkages.

    Both vectors are restricted to their nonzero support and rescaled to equal total mass first
    (p. 688). ``single`` and ``complete`` repeat the same greedy exchange -- at each step, find
    the closest (``single``) or farthest (``complete``) still-active pair of origin and
    destination nodes and move the most mass they can (``min(p_u, q_v)``) between them, until
    every unit of mass has moved. ``average`` is the closed form of p. 688: the weighted mean
    path length over every origin-destination pair, ``Σ p_u q_v |P_u,v| / Σp``.

    Returns ``inf`` rather than raising when the mass genuinely cannot all move: for ``single``,
    once every remaining pair is disconnected; for ``complete``, the moment the *farthest*
    still-active pair is disconnected, which can happen before most of the mass has moved --
    "complete" means the farthest pair decides, and an unreachable pair is infinitely farther
    than any reachable one. ``average`` returns ``inf`` if any pair with positive ``p_u q_v`` is
    disconnected, since the sum it computes then has an infinite term in it.

    Raises when the vectors carry no positive mass, or when ``len(support) x len(support)``
    exceeds :data:`MAX_SUPPORT` — see :func:`_refuse_oversized_support`.
    """
    if linkage not in LINKAGES:
        msg = f"linkage must be one of {', '.join(LINKAGES)}, got {linkage!r}"
        raise ValueError(msg)
    p_support, q_support = _support(p), _support(q)
    if not p_support or not q_support:
        msg = "shortest_path_linkage_distance needs positive mass in each vector"
        raise ValueError(msg)
    _refuse_oversized_support(p_support, q_support, max_support)
    p_scaled, q_scaled = _rescale_to_equal_mass(p_support, q_support)
    p_nodes, q_nodes = sorted(p_scaled, key=str), sorted(q_scaled, key=str)
    dist = _hop_distances(graph, p_nodes, q_nodes)

    if linkage == "average":
        total_mass = sum(p_scaled.values())
        numerator = sum(p_scaled[u] * q_scaled[v] * dist[(u, v)] for u in p_nodes for v in q_nodes)
        return numerator / total_mass

    remaining_p, remaining_q = dict(p_scaled), dict(q_scaled)
    active_p = {node for node, value in remaining_p.items() if value > _EPS}
    active_q = {node for node, value in remaining_q.items() if value > _EPS}
    pick = min if linkage == "single" else max
    total = 0.0
    while active_p and active_q:
        u, v = pick(((a, b) for a in active_p for b in active_q), key=lambda pair: dist[pair])
        if not math.isfinite(dist[(u, v)]):
            return math.inf
        amount = min(remaining_p[u], remaining_q[v])
        total += amount * dist[(u, v)]
        remaining_p[u] -= amount
        remaining_q[v] -= amount
        if remaining_p[u] <= _EPS:
            active_p.discard(u)
        if remaining_q[v] <= _EPS:
            active_q.discard(v)
    return total


def earth_mover_distance(
    graph: nx.Graph,
    p: Mapping[Node, float],
    q: Mapping[Node, float],
    *,
    max_support: int = MAX_SUPPORT,
) -> float:
    """The "optimized" shortest-path family of §47.3 (p. 688-690): the Optimal Transportation
    Problem, solved exactly as a transportation linear program.

    Minimises ``Σ m_u,v d_u,v`` over the amount ``m_u,v`` moved from each origin ``u`` to each
    destination ``v``, subject to ``Σ_v m_u,v = p_u`` and ``Σ_u m_u,v = q_v`` (p. 689), with
    ``d_u,v`` the shortest-path length in hops. Solved with HiGHS through
    ``scipy.optimize.linprog`` -- the general-purpose solver p. 689 cites a family of
    approximation papers for; an exact LP is cheap here because the program has one variable per
    (support-of-p, support-of-q) pair, not per node of the network. Both vectors are restricted
    to their nonzero support and rescaled to equal mass first, the same rule
    :func:`shortest_path_linkage_distance` follows.

    Not built: §47.3's constrained variants -- Multi-Agent Path Finding, where edges and nodes
    have finite capacity, and pursuit-evasion -- both of which the chapter itself introduces as
    "another variant" rather than as the base problem (pp. 690-692); see the module's closing
    report.

    Returns ``inf`` when no feasible transport plan exists at all -- some node in one support has
    no finite path to *any* node in the other, so the LP is infeasible rather than merely
    expensive. Raises on non-positive mass or an oversized support, as
    :func:`shortest_path_linkage_distance` does.
    """
    p_support, q_support = _support(p), _support(q)
    if not p_support or not q_support:
        msg = "earth_mover_distance needs positive mass in each vector"
        raise ValueError(msg)
    _refuse_oversized_support(p_support, q_support, max_support)
    p_scaled, q_scaled = _rescale_to_equal_mass(p_support, q_support)
    p_nodes, q_nodes = sorted(p_scaled, key=str), sorted(q_scaled, key=str)
    cost = _hop_distances(graph, p_nodes, q_nodes)

    n_p, n_q = len(p_nodes), len(q_nodes)
    # One column per finite (u, v) pair; an infinite-cost pair gets no column at all, which is
    # the same as forbidding it, and is what lets the LP itself say "infeasible" rather than this
    # function guessing at a large finite cost to stand in for infinity.
    columns: list[tuple[int, int]] = [
        (i, j)
        for i in range(n_p)
        for j in range(n_q)
        if math.isfinite(cost[(p_nodes[i], q_nodes[j])])
    ]
    if not columns:
        return math.inf
    objective = np.array([cost[(p_nodes[i], q_nodes[j])] for i, j in columns])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for column, (i, j) in enumerate(columns):
        rows += [i, n_p + j]
        cols += [column, column]
        values += [1.0, 1.0]
    a_eq = coo_array((values, (rows, cols)), shape=(n_p + n_q, len(columns)), dtype=np.float64)
    b_eq = np.array([p_scaled[u] for u in p_nodes] + [q_scaled[v] for v in q_nodes])
    solution = linprog(objective, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if solution.status == 2:  # infeasible: some support node has no finite route to the other side
        return math.inf
    if not solution.success:
        msg = f"the transportation program did not solve: {solution.message}"
        raise ValueError(msg)
    return float(solution.fun)


@dataclass(frozen=True)
class ShortestPathDistances:
    """Every §47.3 distance, and the support sizes and rescaling that went into them."""

    single: float
    complete: float
    average: float
    earth_mover: float
    support_p: int
    support_q: int


# ------------------------------------------------------------------- §47.4 graph Fourier transform


def graph_fourier_distance(
    graph: nx.Graph, p: np.ndarray, q: np.ndarray, order: Sequence[Node]
) -> float:
    """``δ = Euclidean(pΛΦᵀ, qΛΦᵀ) = ‖L(p-q)‖₂`` (§47.4, p. 692), exactly as printed.

    P. 692 builds this in two steps: the forward transform ``p̂ = Φᵀp`` turns the node-domain
    signal into spectral coefficients (``Φ``'s columns are ``L``'s eigenvectors); "filtering...
    multiplying it with the diagonal matrix of the Laplacian's eigenvalues Λ" weights each
    coefficient by its eigenvalue, ``Λp̂ = ΛΦᵀp``; and comparing two *filtered* signals with a
    plain Euclidean distance needs them back in the node domain, ``Φ(ΛΦᵀp) = ΦΛΦᵀp``. Since
    ``L = ΦΛΦᵀ`` exactly (the definition of the eigendecomposition ``Λ`` and ``Φ`` build), that
    round trip is just ``Lp`` -- so ``δ = Euclidean(Lp, Lq) = ‖L(p-q)‖₂``, whichever consistent
    order the (row- or column-vector) matrix product is read in. Computed here as ``‖L(p-q)‖₂``
    directly rather than by building ``Φ`` and ``Λ`` and multiplying them out, which is the same
    number without the eigendecomposition's cost or its sign ambiguity.

    Reuses :func:`graphrag.sna.matrices.laplacian` (``combinatorial``, the plain ``D - A``), so
    it refuses a directed graph exactly the way that function does (§8.4: a directed node has
    two degrees, and neither Laplacian is more correct than the other).

    P. 692's own caveats, worth repeating next to any number this returns: *"this is not one,
    but a family of measures"* -- an off-the-shelf distance other than Euclidean could replace
    the outer comparison, since the filtered vectors already carry the network's topology -- and
    *"the transformation proposed here might not be the optimal one"*, since graph Fourier
    filtering has uses (signal cleaning, frequency analysis, sampling, interpolation, trend
    filtering) this chapter does not explore.
    """
    matrix, _ = laplacian(graph, order, kind="combinatorial")
    diff = np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    return float(np.linalg.norm(dense(matrix) @ diff))


def graph_fourier_smoothness(graph: nx.Graph, x: np.ndarray, order: Sequence[Node]) -> float:
    """``xᵀLx`` -- the graph total variation / Dirichlet energy, in coordinates
    ``Σ_(u,v)∈E w(u,v) (x_u - x_v)²``. **Not ch. 47's own formula**: p. 692 defines a *distance*
    between two filtered signals (:func:`graph_fourier_distance`), never a "smoothness" of one
    vector on its own. This is the standard graph-signal-processing quantity that name usually
    refers to, kept here as a diagnostic a report can print beside each vector individually --
    read it as a substitute borrowed from the wider GSP literature, not as a number ch. 47 asks
    for.

    0 exactly when ``x`` is constant on every connected component -- every edge then joins two
    equal values -- and grows with how sharply ``x`` changes across an edge, weighted by that
    edge's weight. Always defined and always ``>= 0`` (``L`` is positive semi-definite); a tiny
    negative value from floating-point error is clamped to 0.

    Reuses :func:`graphrag.sna.matrices.laplacian` (``combinatorial``, the plain ``D - A``), so
    it refuses a directed graph exactly the way that function does (§8.4: a directed node has
    two degrees, and neither Laplacian is more correct than the other).
    """
    matrix, _ = laplacian(graph, order, kind="combinatorial")
    values = np.asarray(x, dtype=np.float64)
    return max(float(values @ dense(matrix) @ values), 0.0)


@dataclass(frozen=True)
class Smoothness:
    """§47.4's number (``distance``, p. 692's own formula) next to a GSP diagnostic (``p``,
    ``q``: each vector's own Dirichlet energy, not something ch. 47 itself defines -- see
    :func:`graph_fourier_smoothness`)."""

    p: float
    q: float
    distance: float


# ----------------------------------------------------------------- §47.5 network-space statistics


def network_variance(
    graph: nx.Graph, x: np.ndarray, order: Sequence[Node], *, max_nodes: int = MAX_NODES
) -> float:
    """``var(x) = ½ Σ_u,v x_u x_v d²_uv`` (§47.5, p. 693), ``d`` read as effective resistance.

    ``x`` is normalised to sum to 1 first -- the formula assumes "a distribution x" (p. 693), and
    a raw count vector is not one. §47.5 prefers effective resistance to shortest-path distance
    here explicitly, "because it is a proper metric and it uses random walks rather than shortest
    paths", which makes it "less sensitive to small changes in the structure" (p. 693) -- the
    same reason :func:`graphrag.sna.walks.effective_resistance` gives.

    0 for a vector concentrated on one node (there is only the ``u=v`` term, and ``d_uu=0``), and
    grows as the mass spreads to nodes farther apart by effective resistance. ``inf`` if the
    vector's support spans more than one connected component, since resistance between them is
    infinite and some pair contributes an infinite term; computed only over the support (the
    nonzero entries) so a zero-mass node paired with an unreachable one never has to multiply
    ``0 * inf`` into a spurious ``nan``.

    Raises when ``x`` has non-positive total mass, which cannot be normalised into a distribution.
    """
    values = np.asarray(x, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        msg = (
            "network_variance needs a vector with positive total mass to normalise into a "
            "distribution"
        )
        raise ValueError(msg)
    normalised = values / total
    support = np.flatnonzero(normalised)
    if support.size == 0:
        return 0.0
    omega, _ = resistance_matrix(graph, order, max_nodes=max_nodes)
    accumulator = 0.0
    for i in support:
        for j in support:
            weight = float(normalised[i] * normalised[j])
            if weight == 0.0:
                continue
            d = float(omega[i, j])
            if not math.isfinite(d):
                return math.inf
            accumulator += weight * d * d
    return 0.5 * accumulator


def network_correlation(
    graph: nx.Graph,
    p: np.ndarray,
    q: np.ndarray,
    order: Sequence[Node],
    *,
    max_nodes: int = MAX_NODES,
) -> float:
    """A network-aware Pearson correlation (§47.5, p. 694): cosine similarity in the inner
    product ``⟨x, y⟩ = xᵀL†y`` the Laplacian pseudoinverse defines.

    **Why this formula, when the book states none.** §47.5 describes network correlation only
    qualitatively -- neighbours should influence a node's contribution, weaker the farther they
    are, "usually estimated via -- again -- the effective resistance matrix" (p. 694) -- and
    cites two papers (Coscia 2021; Coscia and Devriendt 2024) rather than printing a formula, the
    way §47.2 and the network-variance formula both do. ``L†`` is what §11.4 builds effective
    resistance *from* (``Ω[u,v] = L†[u,u] + L†[v,v] - 2L†[u,v]``), so treating it as the
    "covariance" of a vector space on the network keeps this consistent with §47.5's own pointer
    to resistance, reuses the same primitive §47.2 already needs, and gives exactly Pearson's own
    identities: ``rho(x, x) = 1`` and ``rho(x, -x) = -1``, both by direct cancellation in the
    quadratic form.

    **Two things this is not.** It does not mean-centre ``p`` or ``q`` first: ordinary Pearson
    correlation is a cosine similarity of the two vectors *after* each has had its own mean
    subtracted, and the two cited papers generalise that centred version -- this module's
    ``xᵀL†y`` has no centring step, so a pair of vectors that share a large constant offset will
    correlate more strongly here than a mean-centred formula would say. And its number is not on
    the scale of anything else in this report: it is a bounded ``[-1, 1]`` cosine in the ``L†``
    inner product, while :func:`network_variance` sums ``Ω²`` (unbounded, in whatever units the
    vector's own values carry) -- the two are not two readings of the same quantity and should
    never be compared to each other directly. This function is this package's choice among many
    possible readings of an unstated formula, not a reproduction of the cited papers' own result.

    Undefined, and raised on, when either vector is constant on every connected component: it
    then carries no ``L†``-energy (``xᵀL†x = 0``, since ``L†``'s kernel is the constant vectors
    per component), the same way a constant sequence has no variance for
    :func:`graphrag.sna.stats.pearson` to divide by.
    """
    pinv, _ = laplacian_pseudoinverse(graph, order, max_nodes=max_nodes)
    left, right = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    left_energy = float(left @ pinv @ left)
    right_energy = float(right @ pinv @ right)
    if left_energy <= 0 or right_energy <= 0:
        msg = (
            "network_correlation is undefined when a vector is constant on every connected "
            "component: it carries no L-dagger energy to correlate"
        )
        raise ValueError(msg)
    cross = float(left @ pinv @ right)
    value = cross / math.sqrt(left_energy * right_energy)
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class NetworkStatistics:
    """§47.5's numbers: each vector's own network variance, and the correlation between them."""

    variance_p: float
    variance_q: float
    variance_note: str
    correlation: float | None
    correlation_note: str


# ----------------------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class VectorDistanceReport:
    """Everything ``sna distance`` computed for one pair of vectors on one network."""

    persona_id: str
    network: str
    frame: str
    n: int
    p: NodeVector
    q: NodeVector
    p_present: int
    q_present: int
    baselines: Baselines
    generalized_euclidean: GeneralizedEuclidean
    shortest_path: ShortestPathDistances
    smoothness: Smoothness
    statistics: NetworkStatistics
    notes: tuple[str, ...] = field(default_factory=tuple)


def compare_vectors(
    graph: nx.Graph,
    p: NodeVector,
    q: NodeVector,
    *,
    persona_id: str = "",
    max_nodes: int = MAX_NODES,
    max_support: int = MAX_SUPPORT,
) -> VectorDistanceReport:
    """Run every §47.1-47.5 distance for ``p`` and ``q`` on ``graph``, catching what is undefined.

    A section that raises for its own reason (an oversized support, a constant vector, a
    non-positive total mass) is caught and folded into ``notes`` rather than failing the whole
    report -- one undefined number is not a reason to withhold the other four.
    """
    order = node_order(graph)
    array_p, array_q = p.array(order), q.array(order)
    notes: list[str] = []

    baselines = baseline_distances(array_p, array_q)

    components = nx.number_connected_components(graph.to_undirected())
    pinv, _ = laplacian_pseudoinverse(graph, order, max_nodes=max_nodes)
    generalized = GeneralizedEuclidean(
        value=generalized_euclidean_distance(pinv, array_p, array_q),
        plain_euclidean=baselines.euclidean,
        components=components,
        max_nodes=max_nodes,
    )

    try:
        shortest_path = ShortestPathDistances(
            single=shortest_path_linkage_distance(
                graph, p.values, q.values, "single", max_support=max_support
            ),
            complete=shortest_path_linkage_distance(
                graph, p.values, q.values, "complete", max_support=max_support
            ),
            average=shortest_path_linkage_distance(
                graph, p.values, q.values, "average", max_support=max_support
            ),
            earth_mover=earth_mover_distance(graph, p.values, q.values, max_support=max_support),
            support_p=len(_support(p.values)),
            support_q=len(_support(q.values)),
        )
    except ValueError as exc:
        notes.append(f"§47.3 shortest-path and earth-mover distances: {exc}")
        shortest_path = ShortestPathDistances(
            single=math.nan,
            complete=math.nan,
            average=math.nan,
            earth_mover=math.nan,
            support_p=len(_support(p.values)),
            support_q=len(_support(q.values)),
        )

    smoothness = Smoothness(
        p=graph_fourier_smoothness(graph, array_p, order),
        q=graph_fourier_smoothness(graph, array_q, order),
        distance=graph_fourier_distance(graph, array_p, array_q, order),
    )

    variance_note = ""
    try:
        variance_p = network_variance(graph, array_p, order, max_nodes=max_nodes)
        variance_q = network_variance(graph, array_q, order, max_nodes=max_nodes)
    except ValueError as exc:
        variance_p = variance_q = math.nan
        variance_note = str(exc)
        notes.append(f"§47.5 network variance: {exc}")
    correlation: float | None
    correlation_note = ""
    try:
        correlation = network_correlation(graph, array_p, array_q, order, max_nodes=max_nodes)
    except ValueError as exc:
        correlation, correlation_note = None, str(exc)
    statistics = NetworkStatistics(
        variance_p=variance_p,
        variance_q=variance_q,
        variance_note=variance_note,
        correlation=correlation,
        correlation_note=correlation_note,
    )

    return VectorDistanceReport(
        persona_id=persona_id,
        network=graph.graph.get("network", ""),
        frame=str(graph.graph.get("frame", "")),
        n=graph.number_of_nodes(),
        p=p,
        q=q,
        p_present=p.present(order),
        q_present=q.present(order),
        baselines=baselines,
        generalized_euclidean=generalized,
        shortest_path=shortest_path,
        smoothness=smoothness,
        statistics=statistics,
        notes=tuple(notes),
    )


def _num(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "∞"
    return f"{value:,.4f}"


def render_distance(report: VectorDistanceReport) -> list[str]:
    """``report`` as markdown lines, ready for a console or a file."""
    lines = [
        f"# Node vector distance: {report.persona_id}",
        "",
        f"**Sampling frame.** {report.frame or f'the {report.network} network'}. n={report.n:,} "
        f"nodes. {CHAPTER}.",
        "",
        f"- **p** — `{report.p.spec}`: {report.p.source}. {report.p_present:,}/{report.n:,} "
        "nodes carried a value; the rest are read as 0.",
        f"- **q** — `{report.q.spec}`: {report.q.source}. {report.q_present:,}/{report.n:,} "
        "nodes carried a value; the rest are read as 0.",
        "",
        "**Null model.** None for any number below: these are exact geometric or algebraic "
        "quantities of the two vectors as resolved and this network as built, never a test "
        "against a random population. The one exception is the baseline correlation's p-value, "
        "which is against the null 'these two are independent' (§3.4).",
        "",
        "## Non-network baselines (§47.1)",
        "",
        f"- Euclidean: {_num(report.baselines.euclidean)}",
    ]
    if report.baselines.cosine is not None:
        lines.append(f"- Cosine: {_num(report.baselines.cosine)}")
    else:
        lines.append(f"- Cosine: undefined — {report.baselines.cosine_note}")
    if report.baselines.correlation is not None:
        corr = report.baselines.correlation
        lines.append(
            f"- Correlation (1 - Pearson r): {_num(corr.distance)} (r={corr.coefficient:.4f}, "
            f"p={corr.p_value:.4f}, n={corr.n})"
        )
    else:
        lines.append(f"- Correlation: undefined — {report.baselines.correlation_note}")
    lines += [
        "",
        "## Generalized Euclidean (§47.2, Laplacian pseudoinverse)",
        "",
        f"- Generalized Euclidean: {_num(report.generalized_euclidean.value)}, against a plain "
        f"Euclidean of {_num(report.generalized_euclidean.plain_euclidean)}",
        f"- {report.generalized_euclidean.components:,} connected component(s); L-dagger is "
        "block-diagonal across them, so mass on different components contributes no cross term "
        "-- read a number here as each side's own spread, not as a distance between components",
        "",
        "## Shortest-path and earth-mover (§47.3)",
        "",
    ]
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
    else:
        sp = report.shortest_path
        lines += [
            f"- Support: {sp.support_p:,} nonzero node(s) in p, {sp.support_q:,} in q",
            f"- Single linkage: {_num(sp.single)}",
            f"- Complete linkage: {_num(sp.complete)}",
            f"- Average linkage: {_num(sp.average)}",
            f"- Earth-mover (optimal transport): {_num(sp.earth_mover)}",
        ]
    lines += [
        "",
        "## Graph Fourier transform (§47.4)",
        "",
        f"- Distance (p. 692's own formula, ‖L(p-q)‖₂): {_num(report.smoothness.distance)}",
        f"- Dirichlet energy of p (a GSP diagnostic, not ch. 47's own formula): "
        f"{_num(report.smoothness.p)}",
        f"- Dirichlet energy of q (same diagnostic): {_num(report.smoothness.q)}",
        "",
        "## Network-space statistics (§47.5)",
        "",
    ]
    if report.statistics.variance_note:
        lines.append(f"- Network variance: undefined — {report.statistics.variance_note}")
    else:
        lines += [
            f"- Network variance of p: {_num(report.statistics.variance_p)}",
            f"- Network variance of q: {_num(report.statistics.variance_q)}",
        ]
    if report.statistics.correlation is not None:
        lines.append(f"- Network correlation: {_num(report.statistics.correlation)}")
        lines.append(
            "  (this package's own choice of formula, since §47.5 states none: not "
            "mean-centred, and not on network variance's Ω² scale -- see network_correlation's "
            "docstring)"
        )
    else:
        lines.append(f"- Network correlation: undefined — {report.statistics.correlation_note}")
    return lines


def distance_payload(report: VectorDistanceReport) -> dict[str, object]:
    """``report`` as JSON-safe primitives, for a caller that wants the numbers, not the prose."""
    correlation = report.baselines.correlation
    return {
        "persona_id": report.persona_id,
        "network": report.network,
        "n": report.n,
        "p": {"spec": report.p.spec, "present": report.p_present},
        "q": {"spec": report.q.spec, "present": report.q_present},
        "baselines": {
            "euclidean": report.baselines.euclidean,
            "cosine": report.baselines.cosine,
            "correlation": correlation.distance if correlation is not None else None,
            "correlation_r": correlation.coefficient if correlation is not None else None,
            "correlation_p": correlation.p_value if correlation is not None else None,
        },
        "generalized_euclidean": report.generalized_euclidean.value,
        "shortest_path": {
            "single": report.shortest_path.single,
            "complete": report.shortest_path.complete,
            "average": report.shortest_path.average,
            "earth_mover": report.shortest_path.earth_mover,
        },
        "smoothness": {
            "p": report.smoothness.p,
            "q": report.smoothness.q,
            "distance": report.smoothness.distance,
        },
        "statistics": {
            "variance_p": None if report.statistics.variance_note else report.statistics.variance_p,
            "variance_q": None if report.statistics.variance_note else report.statistics.variance_q,
            "correlation": report.statistics.correlation,
        },
        "notes": list(report.notes),
        "chapter": CHAPTER,
    }
