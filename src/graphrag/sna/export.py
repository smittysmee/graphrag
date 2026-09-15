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

``--where key=value``
    keeps only nodes carrying that attribute value. A speaker is matched on what an attribution
    pass recorded about the speaker; an entity, a topic or a document is matched on what an
    annotation pass recorded about the document the passage belongs to. A node with no value
    for a filtered key is out of the network entirely, the same rule a window applies to an
    undated post: a missing tag is not a value, and treating it as one would put every untagged
    node in whichever group was asked for. The two-mode network holds both kinds of node at
    once, so each key is routed to whichever side ever carries it -- a speaker attribute filters
    the speakers, everything else filters the passages, and therefore the entities and the
    edges -- rather than asked of both sides for every key.

Every node also carries the attributes themselves, as ``attr_<key>``, so ``sna analyze --by``
can partition the network by one of them without rebuilding it. A speaker's value is its own; an
entity's or a topic's is the value most of its passages' documents carry, with the number of
distinct values it saw kept beside it in ``attrn_<key>`` so a report can say how mixed it was.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
WHERE_FRAME = (
    "Restricted to nodes attributed {where}: a speaker by what an attribution pass recorded "
    "about them, everything else by the attributes of the document its passage belongs to. A "
    "node nobody gave a value for that key is outside the network, not in some other group."
)
#: The two-mode network holds both kinds of node in one graph, so each ``--where`` key is
#: routed to whichever side actually carries it rather than asked of both at once -- see
#: ``_split_bipartite_where``.
BIPARTITE_WHERE_FRAME = (
    "Restricted to nodes attributed {where}. Each key is answered by the side that carries it, "
    "never by both: {detail}. A node missing a value for its own key is outside the network "
    "entirely, and the side a key never applies to keeps only the nodes that stay connected "
    "once the other side is filtered."
)

#: How a node's attributes are named in ``graph.nodes``: the value, and how many distinct values
#: the node's passages carried. Prefixed rather than bare so an attribute called ``type`` or
#: ``documents`` cannot overwrite what the builders already put on the node.
ATTR = "attr_"
ATTR_COUNT = "attrn_"


def attr_key(key: str) -> str:
    """The node-data key holding the value of attribute ``key``."""
    return f"{ATTR}{key}"


def attr_count_key(key: str) -> str:
    """The node-data key holding how many distinct values this node's passages carried."""
    return f"{ATTR_COUNT}{key}"


def matches(attributes: Mapping[str, str], where: Mapping[str, str] | None) -> bool:
    """Whether these attributes satisfy every ``key=value`` asked for.

    A key the node has no value for never matches. That is the whole rule, and it is the one
    worth stating in a report: filtering a corpus on ``region=north`` answers a question about
    the nodes somebody tagged, never about the corpus.
    """
    if not where:
        return True
    return all(attributes.get(key) == value for key, value in where.items())


#: What each node's attributes were seen to be: node -> attribute key -> value -> how often.
Seen = dict[str, dict[str, Counter[str]]]


def _seen(seen: Seen, node: str, attributes: Mapping[str, str]) -> None:
    """Record one more sighting of a node's attributes, from a speaker or from a document."""
    held = seen.setdefault(node, {})
    for key, value in attributes.items():
        held.setdefault(key, Counter())[value] += 1


def _label_attributes(graph: nx.Graph, seen: Seen) -> None:
    """Put one value per attribute key on each node, with how many values it actually saw.

    A speaker carries exactly one value per key, because that is how the attribution layer
    writes it. An entity or a topic is spread over many documents, which can disagree, so the
    value written is the one most of its passages' documents carry -- ties broken alphabetically
    so a rebuild produces the same label -- and the count of distinct values goes beside it. A
    report that partitions by such a key has to be able to say how many of its nodes were mixed.
    """
    for node in graph.nodes:
        for key, counts in seen.get(node, {}).items():
            if not counts:
                continue
            graph.nodes[node][attr_key(key)] = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            graph.nodes[node][attr_count_key(key)] = len(counts)


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
    where: Mapping[str, str] | None = None,
    where_detail: str = "",
) -> str:
    """The sampling frame, plus a sentence for every filter that changed the question.

    ``where_detail``, when given, names which side of a two-mode network answered each
    ``--where`` key and replaces the generic :data:`WHERE_FRAME` sentence with
    :data:`BIPARTITE_WHERE_FRAME`; every other network keeps the generic sentence.
    """
    parts = [FRAMES[network]]
    if stances:
        parts.append(STANCE_FRAME.format(stances=" or ".join(stances)))
    if facets:
        parts.append(FACET_FRAME.format(facets=", ".join(facets)))
    window = _window_text(since, until)
    if window:
        parts.append(WINDOW_FRAME.format(window=window))
    if where and where_detail:
        parts.append(BIPARTITE_WHERE_FRAME.format(where=where_text(where), detail=where_detail))
    elif where:
        parts.append(WHERE_FRAME.format(where=where_text(where)))
    return " ".join(parts)


def where_text(where: Mapping[str, str] | None) -> str:
    """``region=north, size=small``: the attribute filter as the report prints it."""
    return ", ".join(f"{key}={value}" for key, value in sorted((where or {}).items()))


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
    where: Mapping[str, str] | None = None,
    where_detail: str = "",
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
        where=where_text(where),
        project=project,
        frame=_frame(
            network,
            stances=stances,
            facets=facets,
            since=since,
            until=until,
            where=where,
            where_detail=where_detail,
        ),
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
    where: Mapping[str, str] | None = None,
) -> nx.Graph:
    """Speakers joined when they appear in the same document; weight = shared documents.

    Node attributes: ``documents`` (how many of the persona's documents the speaker appears in)
    and ``chunks`` (how many passages they hold). With a window, only dated passages count, so a
    speaker credited on a document but never dated disappears rather than being assumed present.

    ``where`` is matched against the speaker's own attributes, not the document's: the nodes
    here are people, and the question an attribute filter asks of this network is which people
    were tagged that way, whatever they happened to write in.
    """
    rows = store.speaker_document_pairs(persona_id, source_id, since=since, until=until)
    if facets:
        _, docs = _facet_ids(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.doc_id in docs]
    rows = [r for r in rows if matches(r.speaker_attributes, where)]
    graph = bipartite_projection([(r.speaker, r.doc_id) for r in rows], side="left")
    documents: Counter[str] = Counter()
    chunks: Counter[str] = Counter()
    seen: Seen = {}
    for row in rows:
        documents[row.speaker] += 1
        chunks[row.speaker] += row.chunks
        _seen(seen, row.speaker, row.speaker_attributes)
    graph.add_nodes_from(sorted(documents))  # keeps speakers who share no document with anyone
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "speaker"
        graph.nodes[name]["documents"] = documents.get(name, 0)
        graph.nodes[name]["chunks"] = chunks.get(name, 0)
    _label_attributes(graph, seen)
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
        where=where,
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
    where: Mapping[str, str] | None = None,
    doc_attrs: Mapping[str, Mapping[str, str]] | None = None,
) -> list[EntityChunk] | list[EntityMention]:
    """The entity-passage edges left after the filters, from whichever read carries them.

    ``entity_chunk_pairs`` is the cheaper read and is used whenever no stance is asked for;
    a stance filter needs the annotation on the mention, which only ``entity_mention_rows``
    carries. Both rows expose the same five fields the projection uses.

    ``where`` is answered from the document the passage belongs to. An entity has no attributes
    of its own -- it is named by passages, and what those passages are is what an annotation
    pass recorded about their documents -- so a passage whose document was never given a value
    for a filtered key drops out, taking that mention with it.
    """
    passages = _facet_ids(store, persona_id, source_id, facets)[0] if facets else None
    windowed = since is not None or until is not None
    docs = _window_doc_ids(store, persona_id, source_id, since, until) if windowed else None
    attributes = doc_attrs or {}

    def keep(row: EntityChunk | EntityMention) -> bool:
        if passages is not None and row.chunk_id not in passages:
            return False
        if not matches(attributes.get(row.doc_id, {}), where):
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
    where: Mapping[str, str] | None = None,
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
    doc_attrs = store.document_attributes(persona_id)
    rows = _entity_rows(
        store, persona_id, source_id, types, stances, facets, since, until, where, doc_attrs
    )
    graph = bipartite_projection([(r.entity_id, r.chunk_id) for r in rows], side="left")
    mentions: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, tuple[str, str]] = {}
    seen: Seen = {}
    for row in rows:
        mentions[row.entity_id] += 1
        documents[row.entity_id].add(row.doc_id)
        labels[row.entity_id] = (row.name, row.type)
        _seen(seen, row.entity_id, doc_attrs.get(row.doc_id, {}))
    graph.add_nodes_from(sorted(mentions))
    for node in graph.nodes:
        name, kind = labels.get(node, (node, "other"))
        graph.nodes[node]["label"] = name
        graph.nodes[node]["name"] = name
        graph.nodes[node]["mode"] = "entity"
        graph.nodes[node]["type"] = kind
        graph.nodes[node]["mentions"] = mentions.get(node, 0)
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
    _label_attributes(graph, seen)
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
        where=where,
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
    where: Mapping[str, str] | None = None,
) -> nx.Graph:
    """The topic co-occurrence edges for one persona, with document counts on nodes.

    Unfiltered, these are the edges written at ingestion. A facet, a window or an attribute
    cannot be applied to a stored aggregate, so with any of them the co-occurrence is recomputed
    over the documents that survive the filter. The two paths count the same thing -- documents
    sharing a pair of topics -- but the recomputed one is scoped, so its weights are smaller and
    are not comparable with the stored ones.
    """
    filtered = (
        bool(facets)
        or bool(where)
        or since is not None
        or until is not None
        or source_id is not None
    )
    doc_attrs = store.document_attributes(persona_id)
    if filtered:
        doc_ids = store.document_ids(persona_id, source_id)
        if facets:
            _, faceted = _facet_ids(store, persona_id, source_id, facets)
            doc_ids &= faceted
        if since is not None or until is not None:
            doc_ids &= _window_doc_ids(store, persona_id, source_id, since, until)
        if where:
            doc_ids = {d for d in doc_ids if matches(doc_attrs.get(d, {}), where)}
        graph, counts, seen = _topics_from_documents(
            store, persona_id, doc_ids, min_weight, doc_attrs
        )
    else:
        edges = store.topic_edges(persona_id, min_weight)
        graph = nx.Graph()
        # Nodes first and in sorted order, for the same reason as in ``bipartite_projection``.
        graph.add_nodes_from(sorted({name for e in edges for name in (e.source, e.target)}))
        for edge in edges:
            graph.add_edge(edge.source, edge.target, weight=edge.weight)
        counts = {t.topic: t.count for t in store.list_topics(persona_id, limit=_ALL_DOCUMENTS)}
        seen = _topic_attributes(store, persona_id, None, doc_attrs)
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "topic"
        graph.nodes[name]["documents"] = counts.get(name, 0)
    _label_attributes(graph, seen)
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
        where=where,
    )
    return graph


def _topic_attributes(
    store: GraphStore,
    persona_id: str,
    doc_ids: set[str] | None,
    doc_attrs: Mapping[str, Mapping[str, str]],
) -> Seen:
    """What the documents carrying each topic were attributed, so a topic node can be labelled."""
    if not doc_attrs:
        return {}
    seen: Seen = {}
    for document in store.list_documents(persona_id, limit=_ALL_DOCUMENTS):
        if doc_ids is not None and document.id not in doc_ids:
            continue
        attributes = doc_attrs.get(document.id, {})
        for topic in set(document.topics):
            _seen(seen, topic, attributes)
    return seen


def _topics_from_documents(
    store: GraphStore,
    persona_id: str,
    doc_ids: set[str],
    min_weight: int,
    doc_attrs: Mapping[str, Mapping[str, str]],
) -> tuple[nx.Graph, dict[str, int], Seen]:
    """Recompute topic co-occurrence over one subset of a persona's documents."""
    every = store.list_documents(persona_id, limit=_ALL_DOCUMENTS)
    documents = [d for d in every if d.id in doc_ids]
    weights: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    seen: Seen = {}
    for document in documents:
        topics = sorted(set(document.topics))
        counts.update(topics)
        weights.update(combinations(topics, 2))
        for topic in topics:
            _seen(seen, topic, doc_attrs.get(document.id, {}))
    graph = nx.Graph()
    kept = {pair: w for pair, w in weights.items() if w >= min_weight}
    graph.add_nodes_from(sorted({name for pair in kept for name in pair}))
    for (source, target), weight in kept.items():
        graph.add_edge(source, target, weight=weight)
    return graph, dict(counts), seen


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
    where: Mapping[str, str] | None = None,
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

    ``where`` has two sides here because the graph does: a key is answered by whichever side
    actually carries it, never by both at once. A key any speaker was ever given a value for
    filters the speakers -- an edge whose speaker does not match is dropped, and an entity left
    with no matching speaker drops with it. Every other key filters the passages by the document
    each belongs to, which is therefore also what filters the entity side and the edges: an
    entity's mentions are cut down to the ones in matching documents, and a speaker who wrote
    only outside those documents drops the same way. Asking both sides for the same key -- the
    older, broken behaviour -- would zero out whichever side never recorded it: a document-only
    key has no value for any speaker, and "no value" never matches, so every speaker would drop
    for a key that was never about them.
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
    speaker_attrs = store.speaker_attributes(persona_id)
    speaker_where, doc_where = _split_bipartite_where(where, speaker_attrs)
    if doc_where:
        rows = [r for r in rows if matches(r.document_attributes, doc_where)]

    shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    seen: Seen = {}
    for row in rows:
        labels[row.entity_id] = row.name
        kinds[row.entity_id] = row.type
        for speaker in row.speakers:
            if not matches(speaker_attrs.get(speaker, {}), speaker_where):
                continue
            shared[(speaker, row.entity_id)].add(row.doc_id)
            _seen(seen, speaker, speaker_attrs.get(speaker, {}))
            _seen(seen, row.entity_id, row.document_attributes)
    pairs = sorted(shared)

    if project is None:
        graph = nx.Graph()
        graph.add_nodes_from(sorted({name for pair in pairs for name in pair}))
        for speaker, entity_id in pairs:
            graph.add_edge(speaker, entity_id, weight=len(shared[(speaker, entity_id)]))
    else:
        graph = bipartite_projection(pairs, side="left" if project == "speakers" else "right")

    _label_bipartite(graph, pairs, shared, labels, kinds, project)
    _label_attributes(graph, seen)
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
        where=where,
        where_detail=_bipartite_where_detail(speaker_where, doc_where),
        project=project or "",
    )
    return graph


def _split_bipartite_where(
    where: Mapping[str, str] | None, speaker_attrs: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Route each ``--where`` clause to the side of the two-mode network that carries it.

    A key is a speaker attribute if any speaker was ever given a value for it, whatever this
    particular query asks -- that is what makes it a thing speakers *have*, as opposed to a
    thing this document happens to be tagged with under the same name. Everything else is
    answered as a document attribute, the same rule the single-mode networks already use for
    "everything that is not a speaker".
    """
    if not where:
        return {}, {}
    speaker_keys = {key for attrs in speaker_attrs.values() for key in attrs}
    speaker_where = {k: v for k, v in where.items() if k in speaker_keys}
    doc_where = {k: v for k, v in where.items() if k not in speaker_keys}
    return speaker_where, doc_where


def _bipartite_where_detail(speaker_where: Mapping[str, str], doc_where: Mapping[str, str]) -> str:
    """Which side answered each clause, for the frame line ``_meta`` prints."""
    parts = []
    if speaker_where:
        parts.append(f"{where_text(speaker_where)} by the speakers' own attribute")
    if doc_where:
        parts.append(f"{where_text(doc_where)} by the document each passage belongs to")
    return "; ".join(parts)


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
    where: Mapping[str, str] | None = None,
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
            where=where,
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
            where=where,
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
            where=where,
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
        where=where,
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
