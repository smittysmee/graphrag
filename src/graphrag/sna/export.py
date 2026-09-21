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
    pass recorded about the speaker, an entity on what an extraction pass recorded about the
    entity and otherwise on the document its passage belongs to, and a topic on the document. A
    node with no value for a filtered key is out of the network entirely, the same rule a window
    applies to an undated post: a missing tag is not a value, and treating it as one would put
    every untagged node in whichever group was asked for.

    The two-mode network holds both kinds of node at once, so each key is routed to
    whichever side ever carries it -- a speaker attribute filters the speakers, everything
    else filters the passages, and therefore the entities and the edges -- rather than asked
    of both sides for every key.

``--backbone``
    chapter 27's filters, applied after ``min_weight`` and never unless asked for. The naive
    threshold ``min_weight`` applies is itself §27.1, and the chapter is an argument against it;
    :mod:`graphrag.sna.backbone` holds the defensible ones. Each leaves on every surviving edge
    the ``p_value`` or ``score`` it survived on and appends its own sentence to the frame,
    because a backboned network is a different sample and not a tidier view of the same one.

Every node also carries the attributes themselves, as ``attr_<key>``, so ``sna analyze --by``
can partition the network by one of them without rebuilding it. Beside each value sit two more
keys: ``attrn_<key>``, how many distinct values the node was seen under, and ``attrsrc_<key>``,
where the value came from.

That last one decides whether a number computed from the attribute means anything, so it is
recorded rather than assumed:

``node``
    the attribute was recorded about this node -- a speaker by an attribution pass, an entity by
    an extraction pass. The label is a property of the node and is independent of the edges.
``document``
    no such record exists, so the value is the one most of the documents holding the node's
    passages carry, ties broken alphabetically so a rebuild produces the same label.

A borrowed label is not a worse label, but it cannot be correlated with the edges of a network
whose edges are drawn from the same documents. Two entities are joined here *because* they share
a passage, so they share a document, so they are handed the same borrowed value: the edge and
both its labels have one source, and the assortativity that follows measures the projection, not
the entities. Permuting the labels does not remove it -- the permutation null holds the graph
fixed and only the observed value is inflated -- which is why the shortest way to describe the
defect is that the measure was never about the attribute. :mod:`graphrag.sna.attributes` reads
these keys back and says so next to the number; an entity given its own attributes is how the
question becomes answerable rather than merely reportable.

The last section of this module is interchange (Atlas §53.1-53.2): ``write_graph`` and
``read_graph`` move a network between here and GraphML, node-link JSON, GEXF (Gephi), Pajek,
a weighted edge list and Cytoscape's node/edge CSV pair, and ``FORMATS`` says what each format
can and cannot carry -- the provenance above is the first thing a format drops. ``to_igraph``
and ``to_graph_tool`` hand the same network to the two libraries §53.1 recommends for the work
networkx is slow at; neither library is a dependency, and both are imported only when called.
"""

from __future__ import annotations

import csv
import importlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from graphrag.graph.store import GraphStore
from graphrag.models import EntityChunk, EntityMention, RelationRow, Stance
from graphrag.sna.backbone import DEFAULT_ALPHA, prune_below
from graphrag.sna.backbone import backbone as extract_backbone
from graphrag.sna.projection import (
    DEFAULT_LAMBDA,
    PAIRS,
    PROJECTION,
    PROJECTION_LAMBDA,
    SCHEME_NOTES,
    SCHEMES,
    project,
)
from graphrag.sna.uncertain import (
    CERTAIN_RULE,
    EDGE_PROBABILITY,
    PROBABILITY_RULE,
    RELATION_RULE,
    STATED_PROBABILITY,
    TIER_RULE,
    TIERS,
    combine,
    format_tier_counts,
    support,
    tier_probability,
)

Network = Literal["speakers", "entities", "topics", "speakers-entities", "relations"]
NETWORKS: tuple[Network, ...] = (
    "speakers",
    "entities",
    "topics",
    "speakers-entities",
    "relations",
)

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
    "relations": (
        "Entities an extraction pass said are related, as a directed network: an edge runs from "
        "the entity doing the relating to the one related to, and weighs how many distinct "
        "passages state it. Describes what the corpus asserts, not what is the case, and it "
        "exists only where an extraction pass wrote a relation down: an absent edge is an "
        "absent sentence, never a denial."
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
RELATION_TYPE_FRAME = (
    "Only relations typed {types} are kept, so the network is about that kind of tie alone and "
    "an entity related to another in some other way is absent from it entirely."
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
#: Named on every projected network, including the default one. Chapter 26 exists because the
#: weight is a choice, so a report that does not say which choice was made is not readable, and
#: "simple" is as much a choice as the rest -- the one the chapter spends its first page warning
#: about.
PROJECTION_FRAME = (
    "Edge weights come from the {scheme} projection ({note}): {caveat} The scheme is a modelling "
    "choice (Atlas ch. 26), not a measurement, and weights from two schemes are not comparable. "
    "`graphrag sna projections` runs all of them on this network and reports where they disagree."
)
#: The half-sentence the frame above puts in ``{caveat}``: what this scheme does to a hub.
_SIMPLE_CAVEAT = (
    "one document naming twenty entities makes a 190-edge clique on its own, so these weights "
    "are combinatorial rather than additive (§26.1)."
)
_DISCOUNTED_CAVEAT = (
    "a shared node is discounted by how many others it touches, so a large document counts for "
    "less here than under `simple` and the weights are fractions rather than counts (§26.3-26.5)."
)
_VECTOR_CAVEAT = (
    "the weight is a vector similarity between the two nodes' rows of the incidence matrix, "
    "which also reads the memberships they share the absence of (§26.2)."
)
#: The one scheme whose caveat is about the *book* rather than about the weight. It is stated in
#: the frame, next to the numbers, because a reader checking these against p. 374 will find a
#: formula that does not produce them and needs to know why before doubting the code.
_HYPERBOLIC_CAVEAT = (
    "a large document counts for less here than under `simple` and the weights are fractions "
    "rather than counts (§26.3). Note which formula that is, because the book gives two: p. 374 "
    "displays `w = sum of 1/(k_z - 1)` and argues for the minus one, but its own figure 26.7, "
    "that "
    "figure's caption and §26.4's matrix description all compute `sum of 1/k_z` -- .46 is "
    "1/3 + 1/8 and .79 is 1/3 + 1/3 + 1/8, neither of which the displayed formula gives. The "
    "figure's arithmetic is what is implemented here, because it is also what makes §26.4's "
    "'ProbS is the same as the hyperbolic projection, but you normalize differently' come out."
)
_CAVEATS: dict[str, str] = {
    "simple": _SIMPLE_CAVEAT,
    "jaccard": (
        "the count is divided by the union of both neighbourhoods, so two nodes with one "
        "membership each, shared, score the maximum (§26.1)."
    ),
    "cosine": _VECTOR_CAVEAT,
    "pearson": _VECTOR_CAVEAT,
    "euclidean": _VECTOR_CAVEAT,
    "hyperbolic": _HYPERBOLIC_CAVEAT,
}
BACKBONE_FRAME = (
    "{sentence} Everything below therefore describes the edges that survived that test (ch. 27), "
    "not the corpus, so name the method and the level next to every number."
)

#: How a node's attributes are named in ``graph.nodes``: the value, how many distinct values the
#: node's passages carried, and where the value came from. Prefixed rather than bare so an
#: attribute called ``type`` or ``documents`` cannot overwrite what the builders already put on
#: the node.
ATTR = "attr_"
ATTR_COUNT = "attrn_"
ATTR_SOURCE = "attrsrc_"

#: Where it keeps what each right-mode node was tagged with -- the attributes of the document a
#: passage belongs to. A borrowed label *is* this table, read through the memberships, so a null
#: that rewires the memberships has to re-derive the labels from it rather than carry them along.
RIGHT_ATTRIBUTES = "right_attributes"

#: The keys above are the inputs a projection was built from rather than part of the network, and
#: nothing that leaves this package can hold them: a list of pairs has no GraphML type, and
#: igraph and graph-tool want scalars for a graph attribute. Every export drops them --
#: :func:`write_graph`, :func:`to_igraph`, :func:`to_graph_tool` -- and nothing reads them back.
#: :data:`graphrag.sna.projection.PROJECTION` is *not* among them: it is a scalar, and a file
#: whose weights came from a scheme nobody recorded cannot be read at all.
_UNWRITTEN_GRAPH_KEYS = (PAIRS, RIGHT_ATTRIBUTES)

#: The attribute was recorded about this node, so the label is independent of the edges.
OWN = "node"
#: No such record: the value is borrowed from the documents the node's passages sit in.
INHERITED = "document"


def attr_key(key: str) -> str:
    """The node-data key holding the value of attribute ``key``."""
    return f"{ATTR}{key}"


def attr_count_key(key: str) -> str:
    """The node-data key holding how many distinct values this node's passages carried."""
    return f"{ATTR_COUNT}{key}"


def attr_source_key(key: str) -> str:
    """The node-data key saying whether this node's value is its own or borrowed.

    Either :data:`OWN` or :data:`INHERITED`. Read by :mod:`graphrag.sna.attributes`, which
    cannot report an assortativity honestly without it: see this module's docstring.
    """
    return f"{ATTR_SOURCE}{key}"


def resolve(own: Mapping[str, str], inherited: Mapping[str, str]) -> dict[str, str]:
    """One node's attributes, its own values winning over the ones it would borrow.

    Used for filtering, where the question is what the node *is*: an entity an extraction pass
    put in a sector is in that sector whichever document the passage came from, and falling back
    to the document for a key nobody recorded on the entity is what keeps ``--where`` working on
    a corpus that has only ever tagged documents.
    """
    return {**inherited, **own}


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


def _label_attributes(
    graph: nx.Graph, seen: Seen, own: Mapping[str, Mapping[str, str]] | None = None
) -> None:
    """Put one value per attribute key on each node, with how many values it saw and its source.

    A value recorded about the node itself wins outright and is marked :data:`OWN`: it came from
    one writer, so it saw one value and nothing can outvote it. Otherwise the node is spread over
    many documents, which can disagree, so the value written is the one most of its passages'
    documents carry -- ties broken alphabetically so a rebuild produces the same label -- it is
    marked :data:`INHERITED`, and the count of distinct values goes beside it.

    Both halves are needed downstream and for different reasons. The count says how mixed a node
    was, so a report partitioning by the key can say how many of its nodes were a majority rather
    than a fact. The source says whether the label was drawn from the same documents as the
    edges, which is what decides whether correlating the two measures anything at all.
    """
    held = own or {}
    for node in graph.nodes:
        intrinsic = held.get(node, {})
        for key, value in intrinsic.items():
            graph.nodes[node][attr_key(key)] = value
            graph.nodes[node][attr_count_key(key)] = 1
            graph.nodes[node][attr_source_key(key)] = OWN
        for key, counts in seen.get(node, {}).items():
            if not counts or key in intrinsic:
                continue
            graph.nodes[node][attr_key(key)] = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            graph.nodes[node][attr_count_key(key)] = len(counts)
            graph.nodes[node][attr_source_key(key)] = INHERITED


def bipartite_projection(
    pairs: Iterable[tuple[str, str]],
    side: Side = "left",
    *,
    scheme: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    probabilities: Mapping[tuple[str, str], float] | None = None,
) -> nx.Graph:
    """Project a two-mode edge list onto one side (ch. 26), pricing each edge (ch. 28).

    ``pairs`` are ``(left, right)`` memberships, for example (speaker, document) or
    (entity, passage). Two nodes on the chosen side are joined when they share an opposite-side
    node. Every projection here uses this one function, so the speaker, entity and
    speaker-entity networks are built the same way.

    ``scheme`` is the weighting, and it is a modelling choice rather than a measurement: the
    default ``simple`` is §26.1's ``|N(u) ∩ N(v)|``, the count of shared opposite-side nodes,
    and :mod:`graphrag.sna.projection` holds the ten others chapter 26 offers along with what
    each one is for. The scheme is recorded on the graph, because a weight of 0.46 means nothing
    until it is named, and every report prints it in the frame.

    Nodes are added in sorted order rather than in the order they turn up. That is not
    cosmetic: Louvain shuffles the node list, and ``networkx`` builds its matrices in insertion
    order, so a graph whose nodes arrive in a different order produces a different partition and
    different eigenvector scores from the very same data and the very same ``--seed``.

    The memberships themselves are kept on the graph under :data:`PAIRS`. A projection is not an
    observation -- the memberships are -- so a null model that rewires the projected edges invents
    graphs no membership table could produce; :func:`graphrag.sna.null.bipartite_preserving` reads
    these back, rewires what was actually observed, and re-projects under this same scheme.

    ``probabilities`` prices each *membership* -- how sure we are that this left node really
    belongs to that right node -- and the projected edge gets a probability of existing from them
    (Atlas §28.2, :mod:`graphrag.sna.uncertain`). One shared opposite-mode node is worth the
    weaker of the two memberships through it (:func:`~graphrag.sna.uncertain.support`: both have
    to hold, and the same pass produced both, so the minimum and not the product), and the shares
    combine as ``1 - prod(1 - p)`` (:func:`~graphrag.sna.uncertain.combine`: the edge is there if
    any one of them holds). Left out, every membership is certain and every edge gets ``p = 1.0``,
    which is what the speaker network's documents are: a record, not a match. The price is the
    same under every scheme, because every scheme places an edge exactly where the simple count
    is non-zero (§26.6) and the evidence for an edge is the memberships behind it, not its weight.
    """
    graph = project(pairs, side, scheme=scheme, lam=lam)
    members: dict[str, set[str]] = defaultdict(set)
    for left, right in graph.graph[PAIRS]:
        key, value = (right, left) if side == "left" else (left, right)
        members[key].add(value)
    priced = probabilities or {}

    def membership_p(key: str, value: str) -> float:
        """``p`` of the membership behind one side of a shared node, whichever way it projects."""
        pair = (value, key) if side == "left" else (key, value)
        return float(priced.get(pair, 1.0))

    shares: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, group in members.items():
        for a, b in combinations(sorted(group), 2):
            shares[(a, b)].append(support(membership_p(key, a), membership_p(key, b)))
    for (a, b), evidence in shares.items():
        graph[a][b][EDGE_PROBABILITY] = combine(evidence)
    return graph


def _backboned(
    graph: nx.Graph,
    method: str | None,
    alpha: float,
    correction: str,
    threshold: float | None = None,
) -> nx.Graph:
    """Apply one of chapter 27's backbones, after ``--min-weight`` and before the frame is written.

    ``None`` -- the default everywhere -- leaves the network alone: a backbone changes what the
    network is a sample of, so it is never applied unless it was asked for. When one is applied
    the returned graph is a new one and carries the run in ``graph.graph`` for :func:`_meta` to
    put in the frame.
    """
    if not method:
        return graph
    return extract_backbone(graph, method, alpha=alpha, correction=correction, threshold=threshold)


def default_min_weight(network: str, scheme: str = "simple") -> int:
    """The ``min_weight`` a network uses when the caller named none.

    The counts are the ones this package has always used -- 1 for speakers, 2 for entities and
    topics, 1 for the two-mode and relations networks -- and they are counts: "two shared
    passages, not one". A non-simple projection scheme does not produce counts, so the same
    number would delete every edge; there the default is 1, which prunes nothing, and the
    threshold has to be chosen against the weights the scheme actually produced.
    """
    if scheme != "simple":
        return 1
    return {"entities": 2, "topics": 2}.get(network, 1)


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
    relation_types: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    where_detail: str = "",
    projection: str = "",
    backbone: str = "",
) -> str:
    """The sampling frame, plus a sentence for every filter that changed the question.

    ``where_detail``, when given, names which side of a two-mode network answered each
    ``--where`` key and replaces the generic :data:`WHERE_FRAME` sentence with
    :data:`BIPARTITE_WHERE_FRAME`; every other network keeps the generic sentence.
    """
    parts = [FRAMES[network]]
    if projection:
        parts.append(
            PROJECTION_FRAME.format(
                scheme=projection,
                note=SCHEME_NOTES[projection],
                caveat=_CAVEATS.get(projection, _DISCOUNTED_CAVEAT),
            )
        )
    if stances:
        parts.append(STANCE_FRAME.format(stances=" or ".join(stances)))
    if relation_types:
        parts.append(RELATION_TYPE_FRAME.format(types=" or ".join(relation_types)))
    if facets:
        parts.append(FACET_FRAME.format(facets=", ".join(facets)))
    window = _window_text(since, until)
    if window:
        parts.append(WINDOW_FRAME.format(window=window))
    if where and where_detail:
        parts.append(BIPARTITE_WHERE_FRAME.format(where=where_text(where), detail=where_detail))
    elif where:
        parts.append(WHERE_FRAME.format(where=where_text(where)))
    if backbone:
        parts.append(BACKBONE_FRAME.format(sentence=backbone))
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
    relation_types: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    where_detail: str = "",
    project: str = "",
    projection: str = "",
    lam: float = DEFAULT_LAMBDA,
    p_rule: str = CERTAIN_RULE,
    tiers: Mapping[str, int] | None = None,
) -> None:
    """Record the provenance every report prints next to the numbers.

    ``projection`` is the chapter-26 weighting scheme, and it is empty for the networks that are
    not projections of a membership table -- ``relations``, and the topic network when it is read
    from the co-occurrence edges stored at ingestion. Naming a scheme on those would be a claim
    about how their weights were made that is not true of them.

    ``p_rule`` is how this network's edges were priced (Atlas §28.1) and ``tiers`` is the census
    of the mention tiers behind them, when the network was built from mentions. Both are scalars
    on ``graph.graph``, so they survive into GraphML beside the frame, and
    :func:`graphrag.sna.uncertain.evidence_report` reads them back rather than guessing.
    """
    graph.graph.update(
        network=network,
        persona_id=persona_id,
        source_id=source_id or "",
        min_weight=min_weight,
        types=",".join(types) if types else "",
        stances=",".join(stances) if stances else "",
        facets=",".join(facets) if facets else "",
        relation_types=",".join(relation_types) if relation_types else "",
        since=since or "",
        until=until or "",
        where=where_text(where),
        project=project,
        frame=_frame(
            network,
            stances=stances,
            facets=facets,
            relation_types=relation_types,
            since=since,
            until=until,
            where=where,
            where_detail=where_detail,
            projection=projection,
            backbone=str(graph.graph.get("backbone", "")),
        ),
        unit=unit,
        **{PROBABILITY_RULE: p_rule, TIERS: format_tier_counts(tiers or {})},
    )
    if projection:
        graph.graph[PROJECTION] = projection
        graph.graph[PROJECTION_LAMBDA] = lam


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
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
) -> nx.Graph:
    """Speakers joined when they appear in the same document; weight = shared documents.

    Node attributes: ``documents`` (how many of the persona's documents the speaker appears in)
    and ``chunks`` (how many passages they hold). With a window, only dated passages count, so a
    speaker credited on a document but never dated disappears rather than being assumed present.

    ``where`` is matched against the speaker's own attributes, not the document's: the nodes
    here are people, and the question an attribute filter asks of this network is which people
    were tagged that way, whatever they happened to write in.

    Every edge carries ``p = 1.0`` (Atlas §28.2): a speaker is on a document because the loader
    or an attribution pass put them there, which is a record and not a match, so this pipeline
    has nothing to put below one. It is not a claim that the attribution is right -- §28.1's
    missing and duplicated nodes are exactly the errors it cannot see.
    """
    rows = store.speaker_document_pairs(persona_id, source_id, since=since, until=until)
    if facets:
        _, docs = _facet_ids(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.doc_id in docs]
    rows = [r for r in rows if matches(r.speaker_attributes, where)]
    graph = bipartite_projection(
        [(r.speaker, r.doc_id) for r in rows], side="left", scheme=projection, lam=lam
    )
    documents: Counter[str] = Counter()
    chunks: Counter[str] = Counter()
    # Every row carries the speaker's own attributes, so nothing here is inherited: a speaker is
    # what an attribution pass said it is, in whatever document it turned up in.
    own: dict[str, Mapping[str, str]] = {}
    for row in rows:
        documents[row.speaker] += 1
        chunks[row.speaker] += row.chunks
        own.setdefault(row.speaker, row.speaker_attributes)
    graph.add_nodes_from(sorted(documents))  # keeps speakers who share no document with anyone
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "speaker"
        graph.nodes[name]["documents"] = documents.get(name, 0)
        graph.nodes[name]["chunks"] = chunks.get(name, 0)
    _label_attributes(graph, {}, own)
    prune_below(graph, min_weight)
    graph = _backboned(graph, backbone, alpha, correction, threshold)
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
        projection=projection,
        lam=lam,
        p_rule=CERTAIN_RULE,
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
    entity_attrs: Mapping[str, Mapping[str, str]] | None = None,
) -> list[EntityChunk] | list[EntityMention]:
    """The entity-passage edges left after the filters, from whichever read carries them.

    ``entity_chunk_pairs`` is the cheaper read and is used whenever no stance is asked for;
    a stance filter needs the annotation on the mention, which only ``entity_mention_rows``
    carries. Both rows expose the same five fields the projection uses.

    ``where`` is answered from the entity first and from the document the passage belongs to
    only for the keys the entity carries no value of its own for. An entity an extraction pass
    put in a sector is in that sector in every passage that names it, so filtering on it either
    keeps the entity or drops it. A key recorded only on documents still filters per mention:
    a passage whose document was never given a value for it drops out, taking that mention with
    it, which is the rule a corpus that tags documents and not entities has always had.
    """
    passages = _facet_ids(store, persona_id, source_id, facets)[0] if facets else None
    windowed = since is not None or until is not None
    docs = _window_doc_ids(store, persona_id, source_id, since, until) if windowed else None
    attributes = doc_attrs or {}
    own = entity_attrs or {}

    def keep(row: EntityChunk | EntityMention) -> bool:
        if passages is not None and row.chunk_id not in passages:
            return False
        if not matches(resolve(own.get(row.entity_id, {}), attributes.get(row.doc_id, {})), where):
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


def entity_passage_rows(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    *,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
) -> list[EntityChunk] | list[EntityMention]:
    """The entity-passage memberships :func:`entity_co_mention` projects, before the projection.

    The same rows, the same filters and the same attribute resolution, handed out so that
    :mod:`graphrag.sna.layers` can read a passage as a hyperedge (Atlas §7.3) or as a dated
    occurrence (§7.4) rather than as the edges it stands for. Kept here rather than copied there
    because ``where`` on an entity is decided by a rule -- the entity's own value, then its
    documents' -- that must mean one thing in every reading of the same data.
    """
    return _entity_rows(
        store,
        persona_id,
        source_id,
        types,
        stances,
        facets,
        since,
        until,
        where,
        store.document_attributes(persona_id),
        store.entity_attributes(persona_id),
    )


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
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
) -> nx.Graph:
    """Entities joined when both are mentioned in the same passage; weight = shared passages.

    ``min_weight`` defaults to 2 because a single shared passage is usually coincidence. Node
    attributes: ``name``, ``type``, ``mentions`` (passages mentioning the entity) and
    ``documents``.

    With ``stances`` the network is *signed*: only mentions carrying one of those stances count,
    so two entities are joined because they were praised in the same passage, or complained
    about in the same passage, rather than merely named in it. The two sub-networks are not
    complements of each other -- a mention nobody annotated is in neither.

    Every edge also carries ``p``, its probability of existing (Atlas §28.2), built from the
    tiers of the mentions behind it: a passage is worth the weaker of its two mentions
    (:data:`graphrag.sna.uncertain.TIER_PROBABILITY`) and the passages combine as
    ``1 - prod(1 - p)``. An edge two exact passages support is all but certain; one resting on a
    single loose match is not, and ``analyze`` counts those before it prints anything else. The
    stance is deliberately not part of that number: it rides on only one of the two reads above,
    so folding it in would make ``p`` depend on which flag was passed rather than on the evidence.
    """
    doc_attrs = store.document_attributes(persona_id)
    entity_attrs = store.entity_attributes(persona_id)
    rows = _entity_rows(
        store,
        persona_id,
        source_id,
        types,
        stances,
        facets,
        since,
        until,
        where,
        doc_attrs,
        entity_attrs,
    )
    graph = bipartite_projection(
        [(r.entity_id, r.chunk_id) for r in rows],
        side="left",
        scheme=projection,
        lam=lam,
        probabilities={(r.entity_id, r.chunk_id): tier_probability(r.tier) for r in rows},
    )
    tiers: Counter[str] = Counter(r.tier for r in rows)
    mentions: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, tuple[str, str]] = {}
    seen: Seen = {}
    for row in rows:
        mentions[row.entity_id] += 1
        documents[row.entity_id].add(row.doc_id)
        labels[row.entity_id] = (row.name, row.type)
        _seen(seen, row.entity_id, doc_attrs.get(row.doc_id, {}))
    # What each passage was tagged with, which is what the borrowed labels below are made of: a
    # node with no value of its own takes the one most of these carry. Keeping the table lets
    # ``sna analyze --null bipartite`` re-derive those labels on a rewired corpus instead of
    # carrying them across, which is the only way a null can price the borrowing in.
    graph.graph[RIGHT_ATTRIBUTES] = {
        row.chunk_id: dict(doc_attrs.get(row.doc_id, {})) for row in rows
    }
    graph.add_nodes_from(sorted(mentions))
    for node in graph.nodes:
        name, kind = labels.get(node, (node, "other"))
        graph.nodes[node]["label"] = name
        graph.nodes[node]["name"] = name
        graph.nodes[node]["mode"] = "entity"
        graph.nodes[node]["type"] = kind
        graph.nodes[node]["mentions"] = mentions.get(node, 0)
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
    _label_attributes(graph, seen, entity_attrs)
    prune_below(graph, min_weight)
    graph = _backboned(graph, backbone, alpha, correction, threshold)
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
        projection=projection,
        lam=lam,
        p_rule=TIER_RULE,
        tiers=tiers,
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
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
) -> nx.Graph:
    """The topic co-occurrence edges for one persona, with document counts on nodes.

    Unfiltered, these are the edges written at ingestion. A facet, a window or an attribute
    cannot be applied to a stored aggregate, so with any of them the co-occurrence is recomputed
    over the documents that survive the filter. The two paths count the same thing -- documents
    sharing a pair of topics -- but the recomputed one is scoped, so its weights are smaller and
    are not comparable with the stored ones.

    A ``projection`` other than ``simple`` forces the recomputed path, whatever the filters: the
    stored edge is a count somebody wrote at ingestion and there is no incidence matrix behind
    it to weigh any other way (ch. 26). The recomputed path *is* a projection of documents onto
    topics, so every scheme applies there; the weights are then no longer the counts the stored
    edges hold, which is why the frame names the scheme.
    Every edge carries ``p = 1.0`` (Atlas §28.2). A topic is on a document because the loader
    read it off the front matter, and the stored co-occurrence is a count nobody matched
    fuzzily, so there is no evidence here to be less than sure of. What the number does not say
    is whether the labelling was any good: that is the sampling frame's job, and it says the
    network describes how the corpus was labelled rather than how the subject matter divides.
    """
    scheme = projection or "simple"
    filtered = (
        scheme != "simple"
        or bool(facets)
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
            store, persona_id, doc_ids, min_weight, doc_attrs, scheme=scheme, lam=lam
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
        # Read from the store rather than projected here, so no scheme is claimed for it.
        scheme = ""
    for name in graph.nodes:
        graph.nodes[name]["label"] = name
        graph.nodes[name]["mode"] = "topic"
        graph.nodes[name]["documents"] = counts.get(name, 0)
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 1.0
    _label_attributes(graph, seen)
    graph = _backboned(graph, backbone, alpha, correction, threshold)
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
        projection=scheme,
        lam=lam,
        p_rule=CERTAIN_RULE,
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
    *,
    scheme: str = "simple",
    lam: float = DEFAULT_LAMBDA,
) -> tuple[nx.Graph, dict[str, int], Seen]:
    """Recompute topic co-occurrence over one subset of a persona's documents.

    The pairs are counted here rather than projected, because the counting also produces the
    per-topic document totals and the attribute votes the caller needs. A non-simple ``scheme``
    re-weighs those same pairs through :func:`graphrag.sna.projection.project`, keeping the pair
    set and the order the counting produced: the edge list of a projection is where the count is
    non-zero, whatever the scheme, so only the numbers on the edges change.

    ``min_weight`` of 1 or less keeps every pair, which is
    :func:`graphrag.sna.backbone.prune_below`'s rule and has to be this one too. Applying
    ``w >= min_weight`` unconditionally comes to the same thing on the simple scheme, whose
    weights are counts of at least 1, and deletes the whole network under every other scheme,
    whose weights are fractions -- which is exactly the case :func:`default_min_weight` hands a
    ``min_weight`` of 1 so that nothing would be pruned.
    """
    every = store.list_documents(persona_id, limit=_ALL_DOCUMENTS)
    documents = [d for d in every if d.id in doc_ids]
    counted: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    seen: Seen = {}
    memberships: list[tuple[str, str]] = []
    for document in documents:
        topics = sorted(set(document.topics))
        counts.update(topics)
        counted.update(combinations(topics, 2))
        memberships += [(topic, document.id) for topic in topics]
        for topic in topics:
            _seen(seen, topic, doc_attrs.get(document.id, {}))
    weights: dict[tuple[str, str], float] = dict(counted)
    if scheme != "simple":
        projected = project(memberships, "left", scheme=scheme, lam=lam)
        weights = {
            pair: float(projected.edges[pair]["weight"]) if projected.has_edge(*pair) else 0.0
            for pair in weights
        }
    graph = nx.Graph()
    kept = (
        weights if min_weight <= 1 else {pair: w for pair, w in weights.items() if w >= min_weight}
    )
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
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
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

    ``projection`` weighs the projected edges by one of chapter 26's schemes and is read only
    with ``project``: without it this *is* the two-mode network and nothing has been projected,
    so the frame names no scheme and the weight stays the observed count of documents.

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

    Unlike :func:`entity_co_mention`, this network stores no :data:`RIGHT_ATTRIBUTES` table. Its
    two modes are speakers and entities; the documents are what an edge *weighs*, not a mode. A
    borrowed label here comes from those documents, so no rewiring of speakers against entities
    can re-derive one, and ``sna analyze --null bipartite`` keeps the labels as they are and says
    so. On this network the bipartite null therefore tests the projection's shape only, which is
    still more than a label shuffle can test, and is less than it tests on the entity network.

    ``p`` on an edge is the mention evidence behind it (Atlas §28.2): the speaker's side is a
    record and certain, so a (speaker, entity) edge is priced from the tiers of the mentions in
    the passages the speaker wrote, combined as ``1 - prod(1 - p)``. Projected onto either side,
    those memberships are what the projection prices its own edges from -- the weaker of the two
    memberships through each shared node, combined the same way.
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
    entity_attrs = store.entity_attributes(persona_id)
    speaker_where, doc_where = _split_bipartite_where(where, speaker_attrs)
    if doc_where:
        rows = [
            r
            for r in rows
            if matches(resolve(entity_attrs.get(r.entity_id, {}), r.document_attributes), doc_where)
        ]

    shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    # The mention tiers behind each (speaker, entity) pair, one per passage, combined below.
    # Passages rather than documents, which is what the weight counts: ``1 - prod(1 - p)`` is
    # associative, so grouping the passages by document first would give the same number.
    evidence: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    seen: Seen = {}
    tiers: Counter[str] = Counter()
    for row in rows:
        labels[row.entity_id] = row.name
        kinds[row.entity_id] = row.type
        for speaker in row.speakers:
            if not matches(speaker_attrs.get(speaker, {}), speaker_where):
                continue
            shared[(speaker, row.entity_id)].add(row.doc_id)
            evidence[(speaker, row.entity_id)].append(tier_probability(row.tier))
            tiers[row.tier] += 1
            _seen(seen, row.entity_id, row.document_attributes)
    pairs = sorted(shared)
    probabilities = {pair: combine(values) for pair, values in evidence.items()}

    if project is None:
        # Nothing has been projected, so no scheme applies: the weight is what was observed,
        # the documents holding a speaker's passage that names an entity.
        projection = ""
        graph = nx.Graph()
        graph.add_nodes_from(sorted({name for pair in pairs for name in pair}))
        for speaker, entity_id in pairs:
            graph.add_edge(
                speaker,
                entity_id,
                weight=len(shared[(speaker, entity_id)]),
                **{EDGE_PROBABILITY: probabilities[(speaker, entity_id)]},
            )
    else:
        graph = bipartite_projection(
            pairs,
            side="left" if project == "speakers" else "right",
            scheme=projection,
            lam=lam,
            probabilities=probabilities,
        )

    _label_bipartite(graph, pairs, shared, labels, kinds, project)
    # A speaker's attributes are already its own, so the two maps never claim the same node:
    # a speaker is absent from one and an entity from the other.
    _label_attributes(graph, seen, {**speaker_attrs, **entity_attrs})
    prune_below(graph, min_weight)
    graph = _backboned(graph, backbone, alpha, correction, threshold)
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
        projection=projection,
        lam=lam,
        project=project or "",
        p_rule=TIER_RULE,
        tiers=tiers,
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


def entity_relations(
    store: GraphStore,
    persona_id: str,
    source_id: str | None = None,
    min_weight: int = 1,
    *,
    relation_types: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
) -> nx.DiGraph:
    """Entities an extraction pass related to each other, as a directed graph (§6.2).

    This is the only network here that is not a projection of co-membership. Every other one
    joins two nodes because a document or a passage held both, which is symmetric by
    construction: if A shares a passage with B then B shares it with A. A relation is an
    assertion with a subject and an object, so ``(u, v)`` is not ``(v, u)`` (§6.2) and the graph
    is an :class:`networkx.DiGraph`. Reading it with an undirected measure answers a different
    question, which is why :func:`describe` prints the type at the top of every report and the
    measures that cannot read direction say that they flattened it.

    The weight is how many *distinct passages* state the relation (§6.3, count weights): a
    proximity, not a distance, the same convention the rest of the package uses. ``min_weight``
    defaults to 1 rather than to the entity network's 2, because a single passage stating "A
    acquired B" is evidence of exactly that, where a single shared passage is mostly
    coincidence.

    Edge data: ``type`` is the relation type most passages state for that ordered pair, ties
    broken alphabetically; ``types`` lists every type seen between them, ``;``-joined. A simple
    directed graph holds one edge per ordered pair, so two types between the same two entities
    are folded into one edge and recorded there -- parallel edges are a multigraph, which is
    chapter 7's subject and ATL-07's job, not something to fake here.

    ``where`` has to hold at *both* ends, as in the two-mode network: an edge whose target the
    filter excluded would otherwise stay in the network with one endpoint nobody asked for. An
    entity is matched on what an extraction pass recorded about the entity, falling back to the
    attributes of the document the stating passage belongs to.

    ``p`` is built from the same passages the weight counts (Atlas §28.2): each is worth
    :data:`graphrag.sna.uncertain.STATED_PROBABILITY`, because an extraction pass wrote the
    relation *from* that passage and quoted it, and they combine as ``1 - prod(1 - p)``. So a
    relation two passages state is surer than one a single passage states, which is the ordering
    §28.1 asks a report to make visible. There is no tier here to read: the anchor a relation
    landed on is not recorded, so an edge whose quoted evidence matched no passage is priced like
    one whose evidence was found.
    """
    rows = store.relation_rows(persona_id, source_id)
    if relation_types:
        wanted = {r.strip() for r in relation_types if r.strip()}
        rows = [r for r in rows if r.type in wanted]
    if facets:
        passages, _ = _facet_ids(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.chunk_id in passages]
    if since is not None or until is not None:
        dated = _window_doc_ids(store, persona_id, source_id, since, until)
        rows = [r for r in rows if r.doc_id in dated]
    doc_attrs = store.document_attributes(persona_id)
    entity_attrs = store.entity_attributes(persona_id)
    if where:
        rows = [r for r in rows if _relation_matches(r, where, doc_attrs, entity_attrs)]

    graph = nx.DiGraph()
    # Sorted, for the reason ``bipartite_projection`` sorts: insertion order decides matrix
    # order, and a network that is not built identically twice cannot be seeded.
    graph.add_nodes_from(sorted({name for r in rows for name in (r.source_id, r.target_id)}))
    stated: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    labels: dict[str, tuple[str, str]] = {}
    touching: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    seen: Seen = {}
    for row in rows:
        pair = (row.source_id, row.target_id)
        stated[pair][row.type] += 1
        evidence[pair].add(row.chunk_id)
        labels[row.source_id] = (row.source_name, row.source_type)
        labels[row.target_id] = (row.target_name, row.target_type)
        attributes = doc_attrs.get(row.doc_id, {})
        for node in pair:
            touching[node] += 1
            documents[node].add(row.doc_id)
            _seen(seen, node, attributes)
    for (source, target), types in sorted(stated.items()):
        stating = len(evidence[(source, target)])
        graph.add_edge(
            source,
            target,
            weight=stating,
            type=min(types.items(), key=lambda item: (-item[1], item[0]))[0],
            types="; ".join(sorted(types)),
            **{EDGE_PROBABILITY: combine([STATED_PROBABILITY] * stating)},
        )
    for node in graph.nodes:
        name, kind = labels.get(node, (node, "other"))
        graph.nodes[node]["label"] = name
        graph.nodes[node]["name"] = name
        graph.nodes[node]["mode"] = "entity"
        graph.nodes[node]["type"] = kind
        graph.nodes[node]["relations"] = touching.get(node, 0)
        graph.nodes[node]["documents"] = len(documents.get(node, ()))
    _label_attributes(graph, seen, entity_attrs)
    prune_below(graph, min_weight)
    # ``backbone`` is accepted here so that asking for one on this network is refused rather
    # than ignored: §27.5 and §27.6 both have directed forms and neither is implemented, so
    # :func:`graphrag.sna.backbone.backbone` raises rather than flattening the direction away.
    graph = _backboned(graph, backbone, alpha, correction, threshold)
    _meta(
        graph,
        "relations",
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        unit="passages stating the relation",
        relation_types=relation_types,
        facets=facets,
        since=since,
        until=until,
        where=where,
        p_rule=RELATION_RULE,
    )
    return graph


def _relation_matches(
    row: RelationRow,
    where: Mapping[str, str],
    doc_attrs: Mapping[str, Mapping[str, str]],
    entity_attrs: Mapping[str, Mapping[str, str]],
) -> bool:
    """Whether both ends of a relation carry the attributes asked for."""
    inherited = doc_attrs.get(row.doc_id, {})
    return all(
        matches(resolve(entity_attrs.get(entity_id, {}), inherited), where)
        for entity_id in (row.source_id, row.target_id)
    )


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
    relation_types: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
    project: Project | None = None,
) -> nx.Graph:
    """Dispatch to the right builder. ``min_weight`` defaults per network (1, 2, 2, 1, 1).

    ``projection`` names one of chapter 26's weighting schemes
    (:data:`graphrag.sna.projection.SCHEMES`) for the networks that are projections of a
    membership table -- every one but ``relations``. It defaults to ``simple``, §26.1's count of
    shared documents or passages, so the numbers are what they have always been until a scheme
    is asked for, and the frame names whichever was used. Only the simple scheme produces
    counts, so a caller who names another and no ``min_weight`` gets 1 rather than the network's
    usual count threshold, which would delete every fractional edge
    (:func:`default_min_weight`).

    ``backbone`` names one of chapter 27's filters (:data:`graphrag.sna.backbone.BACKBONES`) and
    is applied *after* ``min_weight``, on the network the filters produced: the naive threshold
    is itself §27.1, so the two compose in the order the book would run them. ``alpha`` and
    ``correction`` are read only by the statistical methods. No backbone is applied unless one
    is named, and when one is, the frame says so. Chapters 26 and 27 compose in the book's own
    order too: project first, threshold second (§26.6).
    """
    if network not in NETWORKS:
        msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
        raise ValueError(msg)
    if projection not in SCHEMES:
        msg = f"--projection must be one of {', '.join(SCHEMES)}, got {projection!r}"
        raise ValueError(msg)
    if projection != "simple" and network == "relations":
        msg = (
            "--projection weighs a bipartite projection, and the relations network is not one: "
            "its edges are relations an extraction pass stated, with a direction and a type, "
            "not two entities that shared a passage. Use --network entities."
        )
        raise ValueError(msg)
    if projection != "simple" and network == "speakers-entities" and project is None:
        msg = (
            "--projection weighs the edges a projection creates, and without --project this is "
            "the two-mode network itself. Add --project speakers or --project entities."
        )
        raise ValueError(msg)
    if stances and network in {"speakers", "topics", "relations"}:
        msg = (
            "--stance reads the annotation on a mention, which the speakers, topics and "
            "relations networks do not have. Use --network entities or "
            "--network speakers-entities."
        )
        raise ValueError(msg)
    if project is not None and network != "speakers-entities":
        msg = "--project applies to --network speakers-entities, which is the only two-mode one"
        raise ValueError(msg)
    if relation_types and network != "relations":
        msg = "--relation-type applies to --network relations, the only network with typed edges"
        raise ValueError(msg)
    if network == "relations":
        return entity_relations(
            store,
            persona_id,
            source_id,
            1 if min_weight is None else min_weight,
            relation_types=relation_types,
            facets=facets,
            since=since,
            until=until,
            where=where,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
        )
    if network == "speakers":
        return speaker_co_participation(
            store,
            persona_id,
            source_id,
            default_min_weight(network, projection) if min_weight is None else min_weight,
            facets=facets,
            since=since,
            until=until,
            where=where,
            projection=projection,
            lam=lam,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
        )
    if network == "entities":
        return entity_co_mention(
            store,
            persona_id,
            source_id,
            default_min_weight(network, projection) if min_weight is None else min_weight,
            types,
            stances=stances,
            facets=facets,
            since=since,
            until=until,
            where=where,
            projection=projection,
            lam=lam,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
        )
    if network == "topics":
        return topic_co_occurrence(
            store,
            persona_id,
            default_min_weight(network, projection) if min_weight is None else min_weight,
            source_id=source_id,
            facets=facets,
            since=since,
            until=until,
            where=where,
            projection=projection,
            lam=lam,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
        )
    return speaker_entity_bipartite(
        store,
        persona_id,
        source_id,
        default_min_weight(network, projection) if min_weight is None else min_weight,
        types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        where=where,
        projection=projection,
        lam=lam,
        backbone=backbone,
        alpha=alpha,
        correction=correction,
        threshold=threshold,
        project=project,
    )


# ----------------------------------------------------------------------------- network type


@dataclass(frozen=True)
class NetworkDescription:
    """What kind of network this is, in the Atlas's vocabulary (ch. 6).

    The book builds its types by modifying one definition at a time: a *simple graph* is nodes
    and edges and nothing else, with no parallel edges and no self loops (§6.1); a *directed
    graph* is one where ``(u, v)`` is not ``(v, u)`` (§6.2); a *weighted graph* is one whose
    edges carry a quantity (§6.3); and §6.4 sorts networks by whether their metadata are
    *fundamental* -- whether knowing what the metadata mean changes how the analysis is done.
    Direction, weight and two-mode structure all are: each one changes which measures are
    defined and how their results read, which is why the report prints this line above the
    numbers rather than leaving a reader to infer it.

    ``multilayer`` and ``dynamic`` are the two types of chapter 7 (§7.2, §7.4). Nothing in this
    module sets them; they are read from ``graph.graph``, which is how
    :func:`graphrag.sna.layers.flatten` and :func:`graphrag.sna.layers.snapshots` declare what
    they built without this function learning about either.
    """

    directed: bool
    weighted: bool
    bipartite: bool
    multigraph: bool
    self_loops: int
    multilayer: bool = False
    dynamic: bool = False

    @property
    def simple(self) -> bool:
        """§6.1 literally: nodes and edges, one per pair, no direction, no weight, no loops."""
        return not (
            self.directed
            or self.weighted
            or self.bipartite
            or self.multigraph
            or self.multilayer
            or self.dynamic
            or self.self_loops
        )

    @property
    def text(self) -> str:
        """The type in the book's words, e.g. ``directed, weighted``."""
        if self.simple:
            return "simple"
        parts = [
            "directed" if self.directed else "undirected",
            "weighted" if self.weighted else "unweighted",
        ]
        if self.bipartite:
            parts.append("bipartite")
        if self.multigraph:
            parts.append("multigraph")
        if self.multilayer:
            parts.append("multilayer")
        if self.dynamic:
            parts.append("dynamic")
        if self.self_loops:
            parts.append(f"{self.self_loops} self loop(s)")
        return ", ".join(parts)

    @property
    def sentence(self) -> str:
        """The type and the one thing it changes about reading the numbers below it."""
        if self.simple:
            return f"{self.text} (Atlas §6.1): nodes and edges, and nothing else to interpret."
        if self.directed:
            return (
                f"{self.text} (Atlas §6.2): an edge runs from source to target and (u, v) is "
                "not (v, u), so in- and out- measures answer different questions and "
                "reciprocity says how often both directions were stated (§10.3). Any measure "
                "defined only for undirected graphs says below that it flattened this one."
            )
        if self.bipartite:
            return (
                f"{self.text} (Atlas §6.4): two kinds of node, so every edge crosses between "
                "them and a one-mode reading is a projection, never this network."
            )
        return (
            f"{self.text} (Atlas §6.3): weights are proximities, not distances -- a heavier "
            "edge means the two nodes are closer, the opposite of what a shortest path assumes "
            "of an attribute called weight."
        )


def describe(graph: nx.Graph) -> NetworkDescription:
    """Name this network's type per chapter 6, from the graph itself.

    ``bipartite`` is read from the ``mode`` the builders put on each node, not from
    2-colourability: the question §6.4 asks is whether the nodes are of two *kinds* -- speakers
    and the entities they wrote about -- and plenty of one-mode networks happen to be
    2-colourable without that meaning anything. ``weighted`` is likewise a question about the
    data model: whether the edges carry a weight at all, not whether the weights vary.
    """
    modes = {str(data.get("mode", "")) for _, data in graph.nodes(data=True)}
    return NetworkDescription(
        directed=bool(graph.is_directed()),
        weighted=any("weight" in data for _, _, data in graph.edges(data=True)),
        bipartite=len(modes - {""}) > 1,
        multigraph=bool(graph.is_multigraph()),
        self_loops=int(nx.number_of_selfloops(graph)),
        multilayer=bool(graph.graph.get("layers")),
        dynamic=bool(graph.graph.get("dynamic")),
    )


def ego(graph: nx.Graph, node: str, radius: int = 1) -> nx.Graph:
    """The neighbourhood around one node, out to ``radius`` hops."""
    if node not in graph:
        msg = f"{node!r} is not in this network"
        raise KeyError(msg)
    return nx.ego_graph(graph, node, radius=radius)


# ----------------------------------------------------------------------------- interchange


@dataclass(frozen=True)
class GraphFormat:
    """One interchange format, and exactly what it carries (Atlas §53.2).

    The flags are not decoration. A network built here means nothing without the provenance in
    ``graph.graph`` and the attributes on its nodes, so a format that cannot hold them has to
    say so before it is used, not after somebody reads a partition out of a file that lost the
    labels. ``write_graph`` lists the losses, and ``tests/unit/test_sna_interop.py`` asserts
    each one: a documented loss, never a silent one.
    """

    name: str
    suffixes: tuple[str, ...]
    node_attributes: bool
    """Whether ``attr_*``/``attrn_*``/``attrsrc_*``, labels, modes and counts survive."""
    edge_weights: bool
    edge_attributes: bool
    """Whether edge data *other than* ``weight`` survives."""
    provenance: bool
    """Whether ``graph.graph`` -- persona, network, frame, filters -- survives."""
    direction: bool
    """Whether the file records that the network was directed."""
    isolates: bool
    """Whether a node with no edges is still in the file."""
    typed: bool
    """Whether an *attribute* value comes back as the type it went in as. A CSV cell and an
    edge-list field are text, so they come back as whatever the text reads as -- the string
    ``"2"`` returns as the number 2 -- and GEXF and Pajek return weights as floats.

    It says nothing about node ids, which are text in every format here except node-link JSON:
    a graph whose nodes are integers, which every legendary fixture in ``tests/legendary.py``
    is, comes back keyed by ``"0"`` rather than ``0``. Only ``.json`` returns the ids
    themselves. A test that compares an exported fixture with a fresh one has to expect that,
    and a node id is the one thing that cannot be coerced back safely: ``"0042"`` and ``42``
    are different nodes."""
    note: str


FORMATS: tuple[GraphFormat, ...] = (
    GraphFormat(
        name="GraphML",
        suffixes=(".graphml",),
        node_attributes=True,
        edge_weights=True,
        edge_attributes=True,
        provenance=True,
        direction=True,
        isolates=True,
        typed=True,
        note=(
            "Attribute-lossless, typed, and what Cytoscape reads (§53.2). The format to use "
            "whenever the file will be read back rather than only looked at. Node ids come "
            "back as text, so an integer-keyed graph -- a legendary fixture, say -- returns "
            'keyed by ``"0"``; reading also adds the two ``node_default``/``edge_default`` '
            "keys networkx writes."
        ),
    ),
    GraphFormat(
        name="node-link JSON",
        suffixes=(".json",),
        node_attributes=True,
        edge_weights=True,
        edge_attributes=True,
        provenance=True,
        direction=True,
        isolates=True,
        typed=True,
        note=(
            "Lossless, down to the node ids: the only format here that returns an integer-keyed "
            "graph as integer-keyed. It is also the shape a notebook or a D3 drawing wants."
        ),
    ),
    GraphFormat(
        name="GEXF",
        suffixes=(".gexf",),
        node_attributes=True,
        edge_weights=True,
        edge_attributes=True,
        provenance=False,
        direction=True,
        isolates=True,
        typed=False,
        note=(
            "What Gephi saves and opens (§53.2). GEXF declares attributes for nodes and edges "
            "and has nowhere to put graph-level data, so the persona, the frame and the filters "
            "are lost: keep a GraphML beside it. Reading gives every node a ``label`` (its own "
            "id, when it had none) and every edge an ``id``, returns weights as floats, and -- "
            "as in GraphML, Pajek and the CSV pair -- returns node ids as text."
        ),
    ),
    GraphFormat(
        name="weighted edge list",
        suffixes=(".edgelist", ".txt"),
        node_attributes=False,
        edge_weights=True,
        edge_attributes=False,
        provenance=False,
        direction=False,
        isolates=False,
        typed=False,
        note=(
            "Three tab-separated fields, ``source target weight``: the lowest common "
            "denominator every tool reads. It carries the edges and nothing else -- no node "
            "attributes, no provenance, no direction, and a node with no edges is not in the "
            "file at all. Read it back with ``directed=True`` if the network was directed."
        ),
    ),
    GraphFormat(
        name="Pajek",
        suffixes=(".net",),
        node_attributes=False,
        edge_weights=True,
        edge_attributes=False,
        provenance=False,
        direction=True,
        isolates=True,
        typed=False,
        note=(
            "Pajek's own format (§53.2), which Pajek, networkx and igraph all read. It carries "
            "the names, the edges, the weights and -- through ``*Arcs`` -- the direction. Node "
            "attributes are dropped rather than written through networkx's non-standard "
            "key/value extension, which Pajek itself does not read and which silently discards "
            "every non-string value anyway; the drawing coordinates Pajek stores are not read "
            "back; and parallel edges collapse, which our networks never have."
        ),
    ),
    GraphFormat(
        name="CSV pair",
        suffixes=(".csv",),
        node_attributes=True,
        edge_weights=True,
        edge_attributes=True,
        provenance=False,
        direction=False,
        isolates=True,
        typed=False,
        note=(
            "``<stem>.nodes.csv`` and ``<stem>.edges.csv``, the node table plus edge table "
            "Cytoscape imports. Two tables have nowhere to put graph-level data, so the "
            "provenance and the direction are lost; an empty cell is a missing value rather "
            "than an empty string; and values come back as text unless they read as a number, "
            "so an attribute whose value is the string ``0042`` comes back as the number 42."
        ),
    ),
)

#: Which node keys Pajek writes for drawing rather than for data, and which reading drops.
_PAJEK_DRAWING = ("id", "x", "y", "shape")
#: Tab, not space: speaker names have spaces in them and would split into two nodes.
_EDGELIST_DELIMITER = "\t"


def graph_format(path: Path) -> GraphFormat:
    """The format ``path``'s suffix names, or a ``ValueError`` listing the ones there are."""
    suffix = path.suffix.lower()
    for candidate in FORMATS:
        if suffix in candidate.suffixes:
            return candidate
    known = ", ".join(name for candidate in FORMATS for name in candidate.suffixes)
    msg = f"unsupported export format {path.suffix!r}; use one of {known}"
    raise ValueError(msg)


def write_graph(graph: nx.Graph, path: Path) -> Path:
    """Write a network in the format ``path``'s suffix names, per Atlas §53.2.

    ``.graphml`` and ``.json`` are lossless and are what to write when the file will be read
    back. ``.gexf`` is for Gephi, ``.net`` for Pajek, ``.csv`` for Cytoscape's node/edge table
    pair, and ``.edgelist``/``.txt`` for everything else. What each one drops is recorded on
    ``FORMATS`` and repeated here, because the loss is the part that changes an answer:

    - **GEXF** loses ``graph.graph``: the persona, the network, the sampling frame and the
      filters. A GEXF on its own cannot say what it is a network *of*.
    - **Pajek** loses node attributes as well, so ``--by`` cannot be run on what comes back.
    - **CSV** keeps every attribute but loses the provenance and the direction.
    - **The edge list** keeps only the edges and their weights: no node attributes, no
      provenance, no direction, and every isolated node disappears.

    Every format also drops the two keys in :data:`_UNWRITTEN_GRAPH_KEYS`: the two-mode
    memberships a projection was built from (:data:`PAIRS`) and the right-mode attribute table
    beside them (:data:`RIGHT_ATTRIBUTES`). They are the projection's *input*, not part of the
    network, and no format here has a type for them -- GraphML raises on a list of pairs. The
    in-memory graph keeps them; a graph read back from a file has none, so the bipartite null
    has to be run on a network this package built rather than on one that made a round trip.

    The CSV pair writes two files, ``<stem>.nodes.csv`` and ``<stem>.edges.csv``, and returns
    the first; every other format returns the single path it wrote.
    """
    fmt = graph_format(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = {key: graph.graph.pop(key) for key in _UNWRITTEN_GRAPH_KEYS if key in graph.graph}
    try:
        if fmt.name == "GraphML":
            nx.write_graphml(graph, path)
        elif fmt.name == "node-link JSON":
            data = nx.node_link_data(graph, edges="links")
            path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        elif fmt.name == "GEXF":
            nx.write_gexf(graph, path)
        elif fmt.name == "Pajek":
            nx.write_pajek(_structure_only(graph), path)
        elif fmt.name == "CSV pair":
            return _write_csv(graph, path)
        else:
            _write_edgelist(graph, path)
    finally:
        graph.graph.update(kept)
    return path


def read_graph(path: Path, *, directed: bool = False) -> nx.Graph:
    """Read a network back, the inverse of :func:`write_graph` (Atlas §53.2, §53.1).

    The format is chosen by suffix, and what comes back is what that format could carry: a
    GraphML or a node-link JSON returns the graph with its attributes and its provenance, while
    a Pajek file returns names, edges and weights alone. Check ``graph_format(path)`` before
    trusting an attribute to be there; ``FORMATS`` is the table.

    Only ``.json`` returns the node ids themselves. Every other format here stores them as
    text, so a graph keyed by integers comes back keyed by ``"0"``, ``"1"``, ... and a lookup
    by the original id raises ``KeyError``. Nothing coerces them back, because ``"0042"`` and
    ``42`` are different nodes and guessing between them would merge or split rows silently.

    ``directed`` is consulted only by the two formats that cannot record direction themselves,
    the edge list and the CSV pair, and is ignored by the four that can. An edge list of a
    directed network read without it comes back undirected, and a reciprocated pair of arcs
    silently becomes one edge -- which is why a directed network belongs in GraphML.

    One thing is never restored, because it was never written: the two-mode memberships and the
    right-mode attribute table a projection carried (:data:`_UNWRITTEN_GRAPH_KEYS`). A network
    that made a round trip is the projection alone, so ``sna analyze --null bipartite`` on it
    refuses rather than rewiring memberships it no longer has. Rebuild from the store for that.
    """
    fmt = graph_format(path)
    if fmt.name == "GraphML":
        graph: nx.Graph = nx.read_graphml(path, node_type=str)
        return graph
    if fmt.name == "node-link JSON":
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded: nx.Graph = nx.node_link_graph(payload, edges="links")
        return loaded
    if fmt.name == "GEXF":
        from_gexf: nx.Graph = nx.read_gexf(path, node_type=str)
        return from_gexf
    if fmt.name == "Pajek":
        return _read_pajek(path)
    if fmt.name == "CSV pair":
        return _read_csv(path, directed=directed)
    return _read_edgelist(path, directed=directed)


def _empty(directed: bool) -> nx.Graph:
    """An empty graph of the right kind, for the formats that are handed their direction."""
    return nx.DiGraph() if directed else nx.Graph()


def _value(text: str) -> str | int | float:
    """A CSV or edge-list field as the number it reads as, or as the text it is."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _structure_only(graph: nx.Graph) -> nx.Graph:
    """The same nodes and edges with the weights alone: what Pajek's format can hold."""
    _refuse(
        (node for node in graph.nodes if '"' in str(node)),
        "contain a double quote, which Pajek uses to quote a name and strips on the way back, "
        "so the node would come back under a different name",
    )
    plain = _empty(graph.is_directed())
    plain.add_nodes_from(graph.nodes)
    plain.add_edges_from(
        (source, target, {"weight": data.get("weight", 1)})
        for source, target, data in graph.edges(data=True)
    )
    return plain


def _read_pajek(path: Path) -> nx.Graph:
    """Pajek back as a simple graph: networkx reads ``.net`` as a multigraph, ours never are."""
    multi = nx.read_pajek(path)
    graph = _empty(multi.is_directed())
    graph.add_nodes_from(
        (node, {key: value for key, value in data.items() if key not in _PAJEK_DRAWING})
        for node, data in multi.nodes(data=True)
    )
    for source, target, data in multi.edges(data=True):
        graph.add_edge(source, target, **data)
    return graph


def _refuse(bad: Iterable[object], reason: str) -> None:
    """Stop before writing a name the format would mangle, and say where to write it instead.

    A format that cannot hold a name has two honest options: refuse it, or document that it
    mangles it. These are the names where the mangling would be silent on the way back -- a
    lost edge, or a node under a different name -- so they are refused.
    """
    names = sorted(str(node) for node in bad)
    if not names:
        return
    msg = f"{len(names)} node name(s) {reason} (first: {names[0]!r}); write .graphml instead"
    raise ValueError(msg)


def _write_edgelist(graph: nx.Graph, path: Path) -> None:
    """``source<TAB>target<TAB>weight``, one edge per line and nothing else."""
    _refuse(
        (node for node in graph.nodes if _EDGELIST_DELIMITER in str(node)),
        "contain a tab, which an edge list uses as its delimiter",
    )
    _refuse(
        (node for node in graph.nodes if str(node).startswith("#")),
        "start with '#', which an edge list reads as a comment: the node and every edge it "
        "holds would be missing from what comes back",
    )
    lines = (
        _EDGELIST_DELIMITER.join((str(source), str(target), str(data.get("weight", 1))))
        for source, target, data in graph.edges(data=True)
    )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _read_edgelist(path: Path, *, directed: bool) -> nx.Graph:
    graph = _empty(directed)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(_EDGELIST_DELIMITER)
        if len(fields) == 2:
            graph.add_edge(fields[0], fields[1])
        elif len(fields) == 3:
            graph.add_edge(fields[0], fields[1], weight=_value(fields[2]))
        else:
            msg = f"{path}:{number}: expected 'source<TAB>target[<TAB>weight]', got {line!r}"
            raise ValueError(msg)
    return graph


def _csv_paths(path: Path) -> tuple[Path, Path]:
    """The node and edge tables for ``path``, whichever of the three names it was given."""
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    for tail in (".nodes", ".edges"):
        stem = stem.removesuffix(tail)
    return path.with_name(f"{stem}.nodes.csv"), path.with_name(f"{stem}.edges.csv")


def _write_csv(graph: nx.Graph, path: Path) -> Path:
    """The Cytoscape pair: one row per node with every attribute, one row per edge."""
    nodes_path, edges_path = _csv_paths(path)
    node_keys = sorted({key for _, data in graph.nodes(data=True) for key in data})
    with nodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *node_keys])
        for node, data in graph.nodes(data=True):
            writer.writerow([node, *(data.get(key, "") for key in node_keys)])
    edge_keys = sorted({key for _, _, data in graph.edges(data=True) for key in data})
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", *edge_keys])
        for source, target, data in graph.edges(data=True):
            writer.writerow([source, target, *(data.get(key, "") for key in edge_keys)])
    return nodes_path


def _read_csv(path: Path, *, directed: bool) -> nx.Graph:
    nodes_path, edges_path = _csv_paths(path)
    graph = _empty(directed)
    with nodes_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node = row.pop("id")
            graph.add_node(node, **{k: _value(v) for k, v in row.items() if v})
    with edges_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source, target = row.pop("source"), row.pop("target")
            graph.add_edge(source, target, **{k: _value(v) for k, v in row.items() if v})
    return graph


def to_igraph(graph: nx.Graph) -> Any:
    """The same network as an ``igraph.Graph``, for the analyses igraph is faster at (§53.1).

    The Atlas recommends igraph exactly where networkx runs out of patience -- label
    propagation and fast-greedy community discovery on a hundred thousand nodes, Figure 53.1 --
    so this is a bridge, not a port: nothing in this package needs igraph, and it is imported
    only when this function is called. Node ids become the ``name`` vertex attribute, node and
    edge attributes carry over by key with ``None`` where a node had no value, and
    ``graph.graph`` becomes igraph's graph attributes, so the provenance survives the crossing --
    all but :data:`_UNWRITTEN_GRAPH_KEYS`, the projection's own inputs, which are not graph
    attributes in anybody's sense and stay behind.

    Raises ``ImportError``, saying how to install it, when igraph is absent.
    """
    igraph = _optional(
        "igraph",
        "igraph is not installed. It is an optional bridge, never a dependency of this package "
        "and not an extra of it: install it with `pip install igraph` (published as "
        "`python-igraph` before 0.10, and that name still works), then call this again.",
    )
    nodes = list(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    node_keys = sorted({key for _, data in graph.nodes(data=True) for key in data})
    edge_keys = sorted({key for _, _, data in graph.edges(data=True) for key in data})
    edges = [(index[source], index[target]) for source, target in graph.edges]
    converted = igraph.Graph(
        n=len(nodes),
        edges=edges,
        directed=graph.is_directed(),
        vertex_attrs={
            "name": [str(node) for node in nodes],
            **{key: [graph.nodes[node].get(key) for node in nodes] for key in node_keys},
        },
        edge_attrs={
            key: [data.get(key) for _, _, data in graph.edges(data=True)] for key in edge_keys
        },
    )
    for key, value in _exportable_provenance(graph):
        converted[key] = value
    return converted


def to_graph_tool(graph: nx.Graph) -> Any:
    """The same network as a ``graph_tool.Graph``, for the heavy end of §53.1.

    graph-tool is the fast, parallel alternative the Atlas pairs with networkx, and the one it
    warns is hard to install -- it is not on PyPI at all. So it is a bridge on the same terms as
    :func:`to_igraph`: imported on call, never required. Node ids become the ``name`` vertex
    property; every other attribute becomes a property map typed by the value it holds (int,
    float, or string for anything else), and ``graph.graph`` becomes graph properties -- all but
    :data:`_UNWRITTEN_GRAPH_KEYS`, which stay behind for the reason :func:`to_igraph` gives.

    Raises ``ImportError``, saying where to get it, when graph-tool is absent.
    """
    gt = _optional(
        "graph_tool.all",
        "graph-tool is not installed, and it is not on PyPI: install it from "
        "https://graph-tool.skewed.de (conda-forge, or your distribution's package), then call "
        "this again. It is an optional bridge, never a dependency of this package.",
    )
    converted = gt.Graph(directed=graph.is_directed())
    vertices = {node: converted.add_vertex() for node in graph.nodes}
    names = converted.new_vertex_property("string")
    for node, vertex in vertices.items():
        names[vertex] = str(node)
    converted.vertex_properties["name"] = names

    node_keys = sorted({key for _, data in graph.nodes(data=True) for key in data})
    for key in node_keys:
        values = {node: data[key] for node, data in graph.nodes(data=True) if key in data}
        prop = converted.new_vertex_property(_gt_type(values.values()))
        for node, value in values.items():
            prop[vertices[node]] = value
        converted.vertex_properties[key] = prop

    edge_keys = sorted({key for _, _, data in graph.edges(data=True) for key in data})
    props = {
        key: converted.new_edge_property(
            _gt_type(data[key] for _, _, data in graph.edges(data=True) if key in data)
        )
        for key in edge_keys
    }
    for source, target, data in graph.edges(data=True):
        edge = converted.add_edge(vertices[source], vertices[target])
        for key, value in data.items():
            props[key][edge] = value
    for key, prop in props.items():
        converted.edge_properties[key] = prop

    for key, value in _exportable_provenance(graph):
        prop = converted.new_graph_property(_gt_type([value]))
        prop[converted] = value
        converted.graph_properties[key] = prop
    return converted


def _optional(module: str, message: str) -> Any:
    """Import a bridge library on demand, or explain how to install it and stop."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(message) from exc


def _exportable_provenance(graph: nx.Graph) -> list[tuple[str, Any]]:
    """``graph.graph`` minus the keys no export carries (:data:`_UNWRITTEN_GRAPH_KEYS`).

    The persona, the network, the frame and the filters cross to igraph and graph-tool, because a
    network without them cannot say what it is a network *of*. The projection's inputs do not: a
    membership table is not a graph attribute in either library, and both would either refuse it
    or flatten it to a string nothing can read back.
    """
    return [(key, value) for key, value in graph.graph.items() if key not in _UNWRITTEN_GRAPH_KEYS]


def _gt_type(values: Iterable[object]) -> str:
    """The graph-tool property type for a column: ``int``, ``double`` or ``string``."""
    seen = list(values)
    if seen and all(isinstance(value, int) for value in seen):  # bool is an int, and fits too
        return "int"
    if seen and all(isinstance(value, int | float) for value in seen):
        return "double"
    return "string"
