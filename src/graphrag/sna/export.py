"""Build ``networkx`` graphs out of a ``GraphStore``, and write them to disk.

Every function here is read-only. The store methods behind them page their Cypher and take
parameters, so nothing interpolates a persona id into a query and nothing reads the environment.

Each returned graph carries its provenance in ``graph.graph``: the persona, the network kind,
the sampling frame in words, and the filters that were applied. The report prints those next to
the numbers, because a co-participation network describes who was recorded, not a population.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal

import networkx as nx

from graphrag.graph.store import GraphStore

Network = Literal["speakers", "entities", "topics"]
NETWORKS: tuple[Network, ...] = ("speakers", "entities", "topics")

Side = Literal["left", "right"]

FRAMES: dict[Network, str] = {
    "speakers": (
        "Speakers who appear in the same document. Describes who was recorded together in this "
        "corpus, not who talks to whom in the world."
    ),
    "entities": (
        "Entities mentioned in the same passage. Describes what the corpus discusses together, "
        "and therefore inherits whatever the extraction pass did and did not name."
    ),
    "topics": (
        "Topics that co-occur on the same document, as computed at ingestion. Describes how the "
        "corpus was labelled, not how the subject matter divides."
    ),
}


def bipartite_projection(pairs: Iterable[tuple[str, str]], side: Side = "left") -> nx.Graph:
    """Project a two-mode edge list onto one side.

    ``pairs`` are ``(left, right)`` memberships, for example (speaker, document) or
    (entity, passage). Two nodes on the chosen side are joined when they share an opposite-side
    node, and the edge weight counts how many they share. Both projections use this one
    function, so the speaker and the entity networks are built the same way.

    Nodes are added in sorted order rather than in the order they turn up. That is not
    cosmetic: Louvain shuffles the node list, and ``networkx`` builds its matrices in insertion
    order, so a graph whose nodes arrive in a different order produces a different partition and
    different eigenvector scores from the very same data and the very same ``--seed``.
    """
    members: dict[str, set[str]] = defaultdict(set)
    for left, right in pairs:
        key, value = (right, left) if side == "left" else (left, right)
        members[key].add(value)

    graph = nx.Graph()
    graph.add_nodes_from(sorted({value for group in members.values() for value in group}))
    for group in members.values():
        for a, b in combinations(sorted(group), 2):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)
    return graph


def _prune(graph: nx.Graph, min_weight: int) -> nx.Graph:
    """Drop edges below ``min_weight``, then drop the nodes that this isolates."""
    if min_weight <= 1:
        return graph
    weak = [(u, v) for u, v, w in graph.edges(data="weight") if (w or 1) < min_weight]
    graph.remove_edges_from(weak)
    graph.remove_nodes_from([n for n, degree in graph.degree() if degree == 0])
    return graph


def speaker_co_participation(
    store: GraphStore, persona_id: str, source_id: str | None = None, min_weight: int = 1
) -> nx.Graph:
    """Speakers joined when they appear in the same document; weight = shared documents.

    Node attributes: ``documents`` (how many of the persona's documents the speaker appears in)
    and ``chunks`` (how many passages they hold).
    """
    rows = store.speaker_document_pairs(persona_id, source_id)
    graph = bipartite_projection([(r.speaker, r.doc_id) for r in rows], side="left")
    documents: Counter[str] = Counter()
    chunks: Counter[str] = Counter()
    for row in rows:
        documents[row.speaker] += 1
        chunks[row.speaker] += row.chunks
    graph.add_nodes_from(sorted(documents))  # keeps speakers who share no document with anyone
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["documents"] = documents.get(name, 0)
        graph.nodes[name]["chunks"] = chunks.get(name, 0)
    _prune(graph, min_weight)
    graph.graph.update(
        network="speakers",
        persona_id=persona_id,
        source_id=source_id or "",
        min_weight=min_weight,
        frame=FRAMES["speakers"],
        unit="documents shared",
    )
    return graph


def entity_co_mention(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    min_weight: int = 2,
    types: Sequence[str] | None = None,
) -> nx.Graph:
    """Entities joined when both are mentioned in the same passage; weight = shared passages.

    ``min_weight`` defaults to 2 because a single shared passage is usually coincidence. Node
    attributes: ``name``, ``type``, ``mentions`` (passages mentioning the entity) and
    ``documents``.
    """
    rows = store.entity_chunk_pairs(persona_id, source_id, types)
    graph = bipartite_projection([(r.entity_id, r.chunk_id) for r in rows], side="left")
    mentions: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, tuple[str, str]] = {}
    for row in rows:
        mentions[row.entity_id] += 1
        documents[row.entity_id].add(row.doc_id)
        labels[row.entity_id] = (row.name, row.type)
    graph.add_nodes_from(sorted(mentions))
    for node in graph.nodes:
        name, kind = labels.get(node, (node, "other"))
        graph.nodes[node]["label"] = name
        graph.nodes[node]["name"] = name
        graph.nodes[node]["type"] = kind
        graph.nodes[node]["mentions"] = mentions.get(node, 0)
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
    _prune(graph, min_weight)
    graph.graph.update(
        network="entities",
        persona_id=persona_id,
        source_id=source_id or "",
        min_weight=min_weight,
        types=",".join(types) if types else "",
        frame=FRAMES["entities"],
        unit="passages shared",
    )
    return graph


def topic_co_occurrence(store: GraphStore, persona_id: str, min_weight: int = 2) -> nx.Graph:
    """The persisted topic co-occurrence edges for one persona, with document counts on nodes."""
    edges = store.topic_edges(persona_id, min_weight)
    graph = nx.Graph()
    # Nodes first and in sorted order, for the same reason as in ``bipartite_projection``.
    graph.add_nodes_from(sorted({name for e in edges for name in (e.source, e.target)}))
    for edge in edges:
        graph.add_edge(edge.source, edge.target, weight=edge.weight)
    counts = {t.topic: t.count for t in store.list_topics(persona_id, limit=1_000_000)}
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["documents"] = counts.get(name, 0)
    graph.graph.update(
        network="topics",
        persona_id=persona_id,
        source_id="",
        min_weight=min_weight,
        frame=FRAMES["topics"],
        unit="documents shared",
    )
    return graph


def build_network(
    store: GraphStore,
    network: str,
    persona_id: str,
    *,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
) -> nx.Graph:
    """Dispatch to the right builder. ``min_weight`` defaults per network (1, 2, 2)."""
    if network == "speakers":
        return speaker_co_participation(
            store, persona_id, source_id, 1 if min_weight is None else min_weight
        )
    if network == "entities":
        return entity_co_mention(
            store, persona_id, source_id, 2 if min_weight is None else min_weight, types
        )
    if network == "topics":
        return topic_co_occurrence(store, persona_id, 2 if min_weight is None else min_weight)
    msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
    raise ValueError(msg)


def ego(graph: nx.Graph, node: str, radius: int = 1) -> nx.Graph:
    """The neighbourhood around one node, out to ``radius`` hops."""
    if node not in graph:
        msg = f"{node!r} is not in this network"
        raise KeyError(msg)
    return nx.ego_graph(graph, node, radius=radius)


def write_graph(graph: nx.Graph, path: Path) -> Path:
    """Write GraphML (``.graphml``) or node-link JSON (``.json``), chosen by suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".graphml":
        nx.write_graphml(graph, path)
    elif suffix == ".json":
        data = nx.node_link_data(graph, edges="links")
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    else:
        msg = f"unsupported export format {path.suffix!r}; use .graphml or .json"
        raise ValueError(msg)
    return path
