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
from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np

CENTRALITIES: tuple[str, ...] = (
    "degree",
    "weighted_degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
)

CENTRALITY_MEANING: dict[str, str] = {
    "degree": "local prominence: how many distinct others a node sits with.",
    "weighted_degree": "volume: prominence with repeated co-appearances counted.",
    "betweenness": "brokerage: how often a node lies on the shortest path between two others.",
    "closeness": "reach: how near a node is to everyone else.",
    "eigenvector": "influence among the influential: being connected to well-connected nodes.",
    "pagerank": "influence with a damping factor, so a single huge hub cannot dominate.",
}


def _with_distance(graph: nx.Graph) -> nx.Graph:
    """A copy carrying ``distance = 1 / weight`` for the shortest-path measures."""
    copy = graph.copy()
    for u, v, data in copy.edges(data=True):
        weight = float(data.get("weight", 1.0)) or 1.0
        copy[u][v]["distance"] = 1.0 / weight
    return copy


def centrality(graph: nx.Graph, kind: str) -> dict[str, float]:
    """One centrality by name. See ``CENTRALITIES`` for the list and ``CENTRALITY_MEANING``
    for what each one answers."""
    if graph.number_of_nodes() == 0:
        return {}
    if kind == "degree":
        return {n: float(d) for n, d in nx.degree_centrality(graph).items()}
    if kind == "weighted_degree":
        return {n: float(d) for n, d in graph.degree(weight="weight")}
    if kind == "betweenness":
        return {
            n: float(v)
            for n, v in nx.betweenness_centrality(_with_distance(graph), weight="distance").items()
        }
    if kind == "closeness":
        return {
            n: float(v)
            for n, v in nx.closeness_centrality(_with_distance(graph), distance="distance").items()
        }
    if kind == "eigenvector":
        return _eigenvector(graph)
    if kind == "pagerank":
        return {n: float(v) for n, v in nx.pagerank(graph, weight="weight").items()}
    msg = f"centrality must be one of {', '.join(CENTRALITIES)}, got {kind!r}"
    raise ValueError(msg)


def _eigenvector(graph: nx.Graph) -> dict[str, float]:
    """The principal eigenvector of the weighted adjacency, solved densely.

    ``networkx`` offers two eigenvector solvers and neither is usable in a report: the power
    iteration fails to converge on a graph with several components, which real corpora produce
    constantly, and the sparse solver starts from a random vector, so the same graph and the
    same seed give different numbers on two runs and it raises outright on graphs with only a
    few nodes. ``numpy.linalg.eigh`` is exact, reproducible and defined for every symmetric
    matrix, which is what a network's adjacency always is here.
    """
    if graph.number_of_edges() == 0:
        return dict.fromkeys(graph.nodes, 0.0)
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=np.float64)
    _, vectors = np.linalg.eigh(adjacency)
    principal = np.abs(vectors[:, -1])  # eigh sorts ascending, so the last column is the top one
    norm = float(np.linalg.norm(principal)) or 1.0
    return {node: float(value) for node, value in zip(nodes, principal / norm, strict=True)}


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
    """
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
    """Whole-network shape: size, density, fragmentation, clustering, assortativity."""
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    if nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "components": 0,
            "largest_component": 0,
            "largest_component_share": 0.0,
            "average_clustering": 0.0,
            "degree_assortativity": 0.0,
            "average_weighted_degree": 0.0,
        }
    components = list(nx.connected_components(graph))
    largest = max((len(c) for c in components), default=0)
    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        assortativity = float("nan")
    if math.isnan(assortativity):
        # Undefined when every node has the same degree; report 0.0 rather than a NaN in a table.
        assortativity = 0.0
    weighted = [float(d) for _, d in graph.degree(weight="weight")]
    return {
        "nodes": nodes,
        "edges": edges,
        "density": float(nx.density(graph)),
        "components": len(components),
        "largest_component": largest,
        "largest_component_share": largest / nodes,
        "average_clustering": float(nx.average_clustering(graph, weight="weight")),
        "degree_assortativity": assortativity,
        "average_weighted_degree": sum(weighted) / nodes,
    }
