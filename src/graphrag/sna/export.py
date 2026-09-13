"""Build ``networkx`` graphs out of a ``GraphStore``, and write them to disk.

Every function here is read-only. The store methods behind them page their Cypher and take
parameters, so nothing interpolates a persona id into a query and nothing reads the environment.

Each returned graph carries its provenance in ``graph.graph``: the persona, the network kind,
the sampling frame in words, and the filters that were applied. The report prints those next to
the numbers, because a co-participation network describes who was recorded, not a population.

Three filters change what the network *means*, not merely how big it is, so each one appends a
sentence to the frame rather than quietly shrinking the graph:

``--stance``
    only mentions the annotation layer marked with that stance count toward an edge, so an
    edge stops meaning "discussed together" and starts meaning "praised together" (or
    complained about together). It is a different claim and has to be reported as one.
``--facet``
    only passages carrying that facet contribute. The speaker network is built per document,
    so there the unit is a document holding such a passage; the topic network is recomputed
    over the surviving documents instead of reading the co-occurrence stored at ingestion.
``--since`` / ``--until``
    the speaker network filters its own dated ``SPOKE`` edges. The other networks have no date
    of their own, so they keep the documents that hold a dated passage inside the window --
    which means an undated document drops out of every network once a window is set.
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
from graphrag.models import EntityChunk, EntityMention, Stance

Network = Literal["speakers", "entities", "topics", "speakers-entities"]
NETWORKS: tuple[Network, ...] = ("speakers", "entities", "topics", "speakers-entities")

Side = Literal["left", "right"]
Project = Literal["speakers", "entities"]
PROJECTIONS: tuple[Project, ...] = ("speakers", "entities")

STANCES: tuple[Stance, ...] = ("praise", "complaint", "substitution", "neutral")

#: Enough to cover any persona's document list; the store pages, so this is one read either way.
_ALL_DOCUMENTS = 1_000_000
#: How many opposite-mode partners a bipartite node records, for the cross-mode report section.
_TOP_PARTNERS = 8

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
    "speakers-entities": (
        "Speakers and the entities mentioned in the passages they wrote, as a two-mode network. "
        "Describes who wrote about what in this corpus; it says nothing about whether a speaker "
        "had an opinion, only that their passage named the thing."
    ),
}

#: Appended to the frame when a filter changes the question the network answers.
STANCE_FRAME = (
    "Only mentions annotated {stances} count toward an edge, so an edge means 'appears in the "
    "same {stances} passage', not 'discussed together'. Unannotated mentions are absent, not "
    "neutral."
)
FACET_FRAME = (
    "Only passages annotated with the facet(s) {facets} contribute, so the network describes "
    "what was said about those functions of the subject and nothing else."
)
WINDOW_FRAME = (
    "Restricted to {window}, read from dated passages, so anything the attribution pass could "
    "not date is outside the network entirely."
)


def bipartite_projection(pairs: Iterable[tuple[str, str]], side: Side = "left") -> nx.Graph:
    """Project a two-mode edge list onto one side.

    ``pairs`` are ``(left, right)`` memberships, for example (speaker, document) or
    (entity, passage). Two nodes on the chosen side are joined when they share an opposite-side
    node, and the edge weight counts how many they share. Every projection here uses this one
    function, so the speaker, entity and speaker-entity networks are built the same way.

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


# ----------------------------------------------------------------------------- filters


def _window_text(since: str | None, until: str | None) -> str:
    """The window in words, for the frame line. Empty when no window was set."""
    if since and until:
        return f"passages dated {since} to {until} inclusive"
    if since:
        return f"passages dated {since} or later"
    if until:
        return f"passages dated {until} or earlier"
    return ""


def _facet_ids(
    store: GraphStore, persona_id: str, source_id: str | None, facets: Sequence[str]
) -> tuple[set[str], set[str]]:
    """The passages carrying any of ``facets``, and the documents those passages belong to."""
    wanted = {f.strip() for f in facets if f.strip()}
    rows = [r for r in store.chunk_facets(persona_id, source_id) if wanted & set(r.facets)]
    return {r.chunk_id for r in rows}, {r.doc_id for r in rows}


def _window_doc_ids(
    store: GraphStore, persona_id: str, source_id: str | None, since: str | None, until: str | None
) -> set[str]:
    """Documents holding at least one passage dated inside the window.

    The entity and topic networks have no date of their own. Rather than invent one from the
    document's publication date -- which for a captured thread is when the file was written, not
    when anyone posted -- they inherit the window from the dated ``SPOKE`` edges the attribution
    pass wrote. A document nobody dated is therefore outside every window, which is the same
    rule the speaker network already applies to an undated post.
    """
    rows = store.speaker_document_pairs(persona_id, source_id, since=since, until=until)
    return {r.doc_id for r in rows}


def _frame(
    network: Network,
    *,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> str:
    """The sampling frame, plus a sentence for every filter that changed the question."""
    parts = [FRAMES[network]]
    if stances:
        parts.append(STANCE_FRAME.format(stances=" or ".join(stances)))
    if facets:
        parts.append(FACET_FRAME.format(facets=", ".join(facets)))
    window = _window_text(since, until)
    if window:
        parts.append(WINDOW_FRAME.format(window=window))
    return " ".join(parts)


def _meta(
    graph: nx.Graph,
    network: Network,
    persona_id: str,
    *,
    source_id: str | None,
    min_weight: int,
    unit: str,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    project: str = "",
) -> None:
    """Record the provenance every report prints next to the numbers."""
    graph.graph.update(
        network=network,
        persona_id=persona_id,
        source_id=source_id or "",
        min_weight=min_weight,
        types=",".join(types) if types else "",
        stances=",".join(stances) if stances else "",
        facets=",".join(facets) if facets else "",
        since=since or "",
        until=until or "",
        project=project,
        frame=_frame(network, stances=stances, facets=facets, since=since, until=until),
        unit=unit,
    )


# ----------------------------------------------------------------------------- networks


def speaker_co_participation(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    min_weight: int = 1,
    *,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> nx.Graph:
    """Speakers joined when they appear in the same document; weight = shared documents.

    Node attributes: ``documents`` (how many of the persona's documents the speaker appears in)
    and ``chunks`` (how many passages they hold). With a window, only dated passages count, so a
    speaker credited on a document but never dated disappears rather than being assumed present.
    """
    rows = store.speaker_document_pairs(persona_id, source_id, since=since, until=until)
    if facets:
        _, docs = _facet_ids(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.doc_id in docs]
    graph = bipartite_projection([(r.speaker, r.doc_id) for r in rows], side="left")
    documents: Counter[str] = Counter()
    chunks: Counter[str] = Counter()
    for row in rows:
        documents[row.speaker] += 1
        chunks[row.speaker] += row.chunks
    graph.add_nodes_from(sorted(documents))  # keeps speakers who share no document with anyone
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "speaker"
        graph.nodes[name]["documents"] = documents.get(name, 0)
        graph.nodes[name]["chunks"] = chunks.get(name, 0)
    _prune(graph, min_weight)
    _meta(
        graph,
        "speakers",
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        unit="documents shared",
        facets=facets,
        since=since,
        until=until,
    )
    return graph


def _entity_rows(
    store: GraphStore,
    persona_id: str,
    source_id: str | None,
    types: Sequence[str] | None,
    stances: Sequence[str] | None,
    facets: Sequence[str] | None,
    since: str | None,
    until: str | None,
) -> list[EntityChunk] | list[EntityMention]:
    """The entity-passage edges left after the filters, from whichever read carries them.

    ``entity_chunk_pairs`` is the cheaper read and is used whenever no stance is asked for;
    a stance filter needs the annotation on the mention, which only ``entity_mention_rows``
    carries. Both rows expose the same five fields the projection uses.
    """
    passages = _facet_ids(store, persona_id, source_id, facets)[0] if facets else None
    windowed = since is not None or until is not None
    docs = _window_doc_ids(store, persona_id, source_id, since, until) if windowed else None

    def keep(row: EntityChunk | EntityMention) -> bool:
        if passages is not None and row.chunk_id not in passages:
            return False
        return docs is None or row.doc_id in docs

    if stances:
        wanted = set(stances)
        return [
            r
            for r in store.entity_mention_rows(persona_id, source_id, types)
            if r.stance in wanted and keep(r)
        ]
    return [r for r in store.entity_chunk_pairs(persona_id, source_id, types) if keep(r)]


def entity_co_mention(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    min_weight: int = 2,
    types: Sequence[str] | None = None,
    *,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> nx.Graph:
    """Entities joined when both are mentioned in the same passage; weight = shared passages.

    ``min_weight`` defaults to 2 because a single shared passage is usually coincidence. Node
    attributes: ``name``, ``type``, ``mentions`` (passages mentioning the entity) and
    ``documents``.

    With ``stances`` the network is *signed*: only mentions carrying one of those stances count,
    so two entities are joined because they were praised in the same passage, or complained
    about in the same passage, rather than merely named in it. The two sub-networks are not
    complements of each other -- a mention nobody annotated is in neither.
    """
    rows = _entity_rows(store, persona_id, source_id, types, stances, facets, since, until)
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
        graph.nodes[node]["mode"] = "entity"
        graph.nodes[node]["type"] = kind
        graph.nodes[node]["mentions"] = mentions.get(node, 0)
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
    _prune(graph, min_weight)
    _meta(
        graph,
        "entities",
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        unit="passages shared",
        types=types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
    )
    return graph


def topic_co_occurrence(
    store: GraphStore,
    persona_id: str,
    min_weight: int = 2,
    *,
    source_id: str | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> nx.Graph:
    """The topic co-occurrence edges for one persona, with document counts on nodes.

    Unfiltered, these are the edges written at ingestion. A facet or a window cannot be applied
    to a stored aggregate, so with either one the co-occurrence is recomputed over the documents
    that survive the filter. The two paths count the same thing -- documents sharing a pair of
    topics -- but the recomputed one is scoped, so its weights are smaller and are not
    comparable with the stored ones.
    """
    filtered = bool(facets) or since is not None or until is not None or source_id is not None
    if filtered:
        doc_ids = store.document_ids(persona_id, source_id)
        if facets:
            _, faceted = _facet_ids(store, persona_id, source_id, facets)
            doc_ids &= faceted
        if since is not None or until is not None:
            doc_ids &= _window_doc_ids(store, persona_id, source_id, since, until)
        graph, counts = _topics_from_documents(store, persona_id, doc_ids, min_weight)
    else:
        edges = store.topic_edges(persona_id, min_weight)
        graph = nx.Graph()
        # Nodes first and in sorted order, for the same reason as in ``bipartite_projection``.
        graph.add_nodes_from(sorted({name for e in edges for name in (e.source, e.target)}))
        for edge in edges:
            graph.add_edge(edge.source, edge.target, weight=edge.weight)
        counts = {t.topic: t.count for t in store.list_topics(persona_id, limit=_ALL_DOCUMENTS)}
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "topic"
        graph.nodes[name]["documents"] = counts.get(name, 0)
    _meta(
        graph,
        "topics",
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        unit="documents shared",
        facets=facets,
        since=since,
        until=until,
    )
    return graph


def _topics_from_documents(
    store: GraphStore, persona_id: str, doc_ids: set[str], min_weight: int
) -> tuple[nx.Graph, dict[str, int]]:
    """Recompute topic co-occurrence over one subset of a persona's documents."""
    every = store.list_documents(persona_id, limit=_ALL_DOCUMENTS)
    documents = [d for d in every if d.id in doc_ids]
    weights: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    for document in documents:
        topics = sorted(set(document.topics))
        counts.update(topics)
        weights.update(combinations(topics, 2))
    graph = nx.Graph()
    kept = {pair: w for pair, w in weights.items() if w >= min_weight}
    graph.add_nodes_from(sorted({name for pair in kept for name in pair}))
    for (source, target), weight in kept.items():
        graph.add_edge(source, target, weight=weight)
    return graph, dict(counts)


def speaker_entity_bipartite(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    min_weight: int = 1,
    types: Sequence[str] | None = None,
    *,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    project: Project | None = None,
) -> nx.Graph:
    """Speakers and the entities their passages mention, as one two-mode network.

    Left with ``project=None`` the graph has both modes: a speaker node carries
    ``mode='speaker'``, an entity node ``mode='entity'``, and an edge between them weighs how
    many *documents* hold a passage by that speaker mentioning that entity. Documents rather
    than passages, because chunk boundaries are an artefact of the chunker: a long post split
    in three would otherwise triple a speaker's apparent interest in whatever it names.

    With ``project`` the two-mode network is collapsed onto one side, and the weight changes
    meaning with it: projected onto ``speakers``, two speakers are joined by how many entities
    they both wrote about; projected onto ``entities``, two entities are joined by how many
    speakers wrote about both. Projected weights inflate -- one document naming twenty entities
    contributes 190 entity pairs -- so read them as co-membership, and raise ``--min-weight``
    before reading any ranking off them.

    Every node keeps ``partners`` and ``partner_weights``: its heaviest opposite-mode
    neighbours, which survive the projection and are what the report's cross-mode section reads.
    """
    rows = store.entity_mention_rows(persona_id, source_id, types)
    if stances:
        keep = set(stances)
        rows = [r for r in rows if r.stance in keep]
    if facets:
        passages, _ = _facet_ids(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.chunk_id in passages]
    if since is not None or until is not None:
        docs = _window_doc_ids(store, persona_id, source_id, since, until)
        rows = [r for r in rows if r.doc_id in docs]

    shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for row in rows:
        labels[row.entity_id] = row.name
        kinds[row.entity_id] = row.type
        for speaker in row.speakers:
            shared[(speaker, row.entity_id)].add(row.doc_id)
    pairs = sorted(shared)

    if project is None:
        graph = nx.Graph()
        graph.add_nodes_from(sorted({name for pair in pairs for name in pair}))
        for speaker, entity_id in pairs:
            graph.add_edge(speaker, entity_id, weight=len(shared[(speaker, entity_id)]))
    else:
        graph = bipartite_projection(pairs, side="left" if project == "speakers" else "right")

    _label_bipartite(graph, pairs, shared, labels, kinds, project)
    _prune(graph, min_weight)
    _meta(
        graph,
        "speakers-entities",
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        unit=(
            "documents shared"
            if project is None
            else ("entities shared" if project == "speakers" else "speakers shared")
        ),
        types=types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        project=project or "",
    )
    return graph


def _label_bipartite(
    graph: nx.Graph,
    pairs: Sequence[tuple[str, str]],
    shared: dict[tuple[str, str], set[str]],
    labels: dict[str, str],
    kinds: dict[str, str],
    project: Project | None,
) -> None:
    """Put the mode, the counts and the top opposite-mode partners on every node.

    ``partners`` and ``partner_weights`` are ``;``-joined strings rather than lists so the graph
    still writes to GraphML, which has no list type. They are the only trace of the other mode
    left after a projection, so the report can still say which entities a speaker community
    wrote about.
    """
    partners: dict[str, Counter[str]] = defaultdict(Counter)
    documents: dict[str, set[str]] = defaultdict(set)
    for speaker, entity_id in pairs:
        docs = shared[(speaker, entity_id)]
        partners[speaker][labels.get(entity_id, entity_id)] += len(docs)
        partners[entity_id][speaker] += len(docs)
        documents[speaker] |= docs
        documents[entity_id] |= docs
    for node in graph.nodes:
        is_entity = node in labels and (project != "speakers")
        graph.nodes[node]["label"] = labels.get(node, node) if is_entity else node
        graph.nodes[node]["mode"] = "entity" if is_entity else "speaker"
        if is_entity:
            graph.nodes[node]["name"] = labels.get(node, node)
            graph.nodes[node]["type"] = kinds.get(node, "other")
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
        top = partners.get(node, Counter()).most_common(_TOP_PARTNERS)
        graph.nodes[node]["partners"] = "; ".join(name for name, _ in top)
        graph.nodes[node]["partner_weights"] = "; ".join(str(count) for _, count in top)


def build_network(
    store: GraphStore,
    network: str,
    persona_id: str,
    *,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    project: Project | None = None,
) -> nx.Graph:
    """Dispatch to the right builder. ``min_weight`` defaults per network (1, 2, 2, 1)."""
    if network not in NETWORKS:
        msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
        raise ValueError(msg)
    if stances and network in {"speakers", "topics"}:
        msg = (
            "--stance reads the annotation on a mention, which the speakers and topics networks "
            "do not have. Use --network entities or --network speakers-entities."
        )
        raise ValueError(msg)
    if project is not None and network != "speakers-entities":
        msg = "--project applies to --network speakers-entities, which is the only two-mode one"
        raise ValueError(msg)
    if network == "speakers":
        return speaker_co_participation(
            store,
            persona_id,
            source_id,
            1 if min_weight is None else min_weight,
            facets=facets,
            since=since,
            until=until,
        )
    if network == "entities":
        return entity_co_mention(
            store,
            persona_id,
            source_id,
            2 if min_weight is None else min_weight,
            types,
            stances=stances,
            facets=facets,
            since=since,
            until=until,
        )
    if network == "topics":
        return topic_co_occurrence(
            store,
            persona_id,
            2 if min_weight is None else min_weight,
            source_id=source_id,
            facets=facets,
            since=since,
            until=until,
        )
    return speaker_entity_bipartite(
        store,
        persona_id,
        source_id,
        1 if min_weight is None else min_weight,
        types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        project=project,
    )


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
