from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Protocol

from graphrag.embed.base import Matrix, Vector
from graphrag.models import (
    Chunk,
    ChunkFacets,
    Document,
    Enrichment,
    Entity,
    EntityChunk,
    EntityCollision,
    EntityMention,
    GraphStats,
    MentionStance,
    PersonaSpec,
    RelatedTopic,
    RelationRow,
    ScoredChunk,
    SpeakerCount,
    SpeakerDocument,
    Stance,
    TopicCount,
    TopicEdge,
)

AttributeKind = Literal["speaker", "entity", "document"]
"""Which sidecar owns an attribute: attribution files say what a speaker is, extraction files what
an entity is, annotation files what a document is."""


class GraphStore(Protocol):
    """Every operation the pipeline, retriever, CLI and MCP server need from the graph."""

    # schema / lifecycle
    def ensure_schema(self, dim: int) -> None: ...
    def close(self) -> None: ...

    # writes
    def upsert_persona(self, persona: PersonaSpec) -> None: ...
    def upsert_documents(self, documents: Sequence[Document]) -> None: ...
    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None: ...
    def upsert_topic_cooccurrence(self, pairs: Sequence[tuple[str, str, int]]) -> None: ...
    # Write entities, mentions and relations. An incoming entity whose id already names a node
    # under a different spelling (compared case- and whitespace-folded, and against the node's
    # recorded `aliases`) never renames it: the node stands, its mentions and relations still
    # attach, and the incoming name comes back as an `EntityCollision` for the caller to report.
    # Saying two spellings are one thing is the alias table's job, not a writer's.
    # The exception is a stale node -- no mention and no recorded alias, all that is left of an
    # entity whose passages a re-ingest deleted. It holds nothing but its id, so the incoming
    # name takes it over instead of a spelling nobody stands behind outliving its evidence.
    # A mention carries the tier that found its passage (`MentionTier`: exact, loose, or none for
    # the fallback anchor). An incoming `unknown` never erases a recorded tier, on the rule the
    # stance already follows, so re-importing a file or a snapshot written before tiers existed
    # does not cost the evidence a later pass measured; reads give `unknown` back for a mention
    # nobody recorded one on.
    def upsert_enrichment(self, enrichment: Enrichment) -> list[EntityCollision]: ...
    # Fold the alias spellings of one entity into a single canonical node: re-point this
    # persona's MENTIONS and RELATED_TO edges, merge the mentions, record `aliases` on the
    # canonical node and delete the alias nodes nothing else holds. Returns mentions moved.
    def merge_entities(self, persona_id: str, canonical: str, aliases: Sequence[str]) -> int: ...
    def delete_persona(self, persona_id: str) -> None: ...
    def delete_documents(self, doc_ids: Sequence[str]) -> None: ...
    # Delete this persona's Entity nodes that no passage mentions any more, taking their
    # RELATED_TO edges with them, and return how many went. `dry_run` counts the same nodes and
    # writes nothing. A re-ingest replaces a source's documents, which deletes the mentions on
    # their passages; without this the entity nodes those mentions were the only evidence for
    # stand for ever, holding their ids against the next extraction pass.
    # Entity nodes are shared between personas, so a node another persona's passage still
    # relates to is left alone even when nothing mentions it.
    def delete_orphan_entities(self, persona_id: str, *, dry_run: bool = False) -> int: ...

    # search
    def vector_search(
        self, vector: Vector, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]: ...
    def fulltext_search(
        self, query: str, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]: ...

    # reads
    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...
    def neighbors(self, chunk_id: str, hops: int = 1) -> list[Chunk]: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def list_documents(
        self,
        persona_id: str | None = None,
        *,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 50,
    ) -> list[Document]: ...
    def document_chunks(self, doc_id: str, start: int = 0, count: int = 5) -> list[Chunk]: ...
    def document_ids(self, persona_id: str, source_id: str | None = None) -> set[str]: ...
    def list_topics(self, persona_id: str | None = None, limit: int = 50) -> list[TopicCount]: ...
    def related_topics(
        self, topic: str, persona_id: str | None = None, limit: int = 10
    ) -> list[RelatedTopic]: ...
    def list_speakers(
        self, persona_id: str | None = None, limit: int = 50
    ) -> list[SpeakerCount]: ...
    def list_personas(self) -> list[PersonaSpec]: ...
    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> list[Entity]: ...
    # every entity this persona mentions anywhere, so a name can be resolved to a node without
    # knowing which passage it was extracted from
    def persona_entities(self, persona_id: str) -> list[Entity]: ...
    def enriched_doc_ids(self, persona_id: str) -> set[str]: ...
    # documents of one source that already carry at least one entity mention
    def enriched_document_ids(self, persona_id: str, source_id: str) -> set[str]: ...

    # bulk (snapshots)
    def iter_documents(self, persona_id: str) -> Iterator[Document]: ...
    def iter_chunks(self, persona_id: str) -> Iterator[tuple[Chunk, Vector]]: ...
    def enrichment_for_persona(self, persona_id: str) -> Enrichment: ...

    # misc
    def stats(self) -> GraphStats: ...
    # ATL-ENT-3: a cheap, one-round-trip summary of a persona's data, for the network cache
    # (graphrag.sna.cache) to fold into its key so a write the CLI did not make -- Cypher run by
    # hand, a foreign client -- still misses. Returns (documents, chunks, mentions,
    # attributed_nodes): the last is how many Document/Entity/Speaker nodes this persona's own
    # data carries an attribute property on, so a property-only reimport (same document and
    # mention counts, new values) still moves it.
    def persona_fingerprint(self, persona_id: str) -> tuple[int, int, int, int]: ...
    def run_readonly_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    # attribution: speakers for documents whose loader parsed no speaker turns
    # Idempotent: re-attaching the same speaker to the same passage changes nothing. When given,
    # `posted_at` (ISO YYYY-MM-DD), `role` and `score` land on the SPOKE edge.
    def attach_speaker(
        self,
        doc_id: str,
        chunk_id: str,
        speaker: str,
        *,
        posted_at: str | None = None,
        role: str | None = None,
        score: int | None = None,
    ) -> None: ...
    # documents of one source that already carry at least one speaker
    def attributed_document_ids(self, persona_id: str, source_id: str) -> set[str]: ...

    # node attributes: what kind of speaker this is, and what kind of document that is
    # Speaker attributes are scoped to one persona, because a Speaker node is shared between
    # personas and a reading written for one corpus is not a fact about the other.
    # First value written wins: a key this persona already holds under a different value is
    # left exactly as it is and its name comes back, for the caller to report. Saying which of
    # two readings is right is the annotator's job, not a writer's.
    def set_speaker_attributes(
        self, persona_id: str, speaker: str, attributes: Mapping[str, str]
    ) -> list[str]: ...
    # Merge into the document's attributes, incoming value winning per key. One annotation file
    # holds one document, so there is no second writer to disagree with.
    def set_document_attributes(self, doc_id: str, attributes: Mapping[str, str]) -> None: ...
    # Entity attributes are scoped to one persona for the reason speaker attributes are: an
    # Entity node is shared between personas and "which sector this company is in" is a reading
    # of one corpus. Same first-value-wins rule, same refused-key return: one entity is named by
    # many documents, so many extraction files can claim it, and which claim is right is the
    # annotator's question rather than the last writer's.
    def set_entity_attributes(
        self, persona_id: str, entity_id: str, attributes: Mapping[str, str]
    ) -> list[str]: ...
    # Every speaker of this persona that carries attributes, every document that does, and every
    # entity that does. Read by the importers (to report conflicts), by the snapshot, and by the
    # networks, which need one lookup rather than one read per node.
    def speaker_attributes(self, persona_id: str) -> dict[str, dict[str, str]]: ...
    def document_attributes(self, persona_id: str) -> dict[str, dict[str, str]]: ...
    def entity_attributes(self, persona_id: str) -> dict[str, dict[str, str]]: ...
    # Forget every value of one kind this persona holds, and say how many nodes lost one.
    # First-value-wins means a corrected sidecar cannot replace a stored value, so a full
    # refresh clears the values that kind of sidecar owns before it re-reads them all. Another
    # persona's values on the same shared node are untouched.
    def clear_attributes(self, persona_id: str, kind: AttributeKind) -> int: ...

    # annotation: what a passage says about an entity, and which functions it is about
    # Both are idempotent, and both ignore a chunk that is not a passage of `doc_id`.
    def annotate_mention(self, doc_id: str, chunk_id: str, entity: str, stance: Stance) -> None: ...
    def annotate_chunk(self, doc_id: str, chunk_id: str, facets: Sequence[str]) -> None: ...
    # documents of one source that already carry at least one stance, facet, or document
    # attribute -- an annotation file with nothing but a top-level ``attributes`` block and no
    # ``annotations`` entries still counts, since it wrote something the graph now holds
    def annotated_document_ids(self, persona_id: str, source_id: str) -> set[str]: ...
    # the annotated edges a signed entity network and a facet filter are built from
    def mention_stances(
        self, persona_id: str, source_id: str | None = None
    ) -> list[MentionStance]: ...
    def chunk_facets(self, persona_id: str, source_id: str | None = None) -> list[ChunkFacets]: ...

    # network analysis (graphrag.sna)
    # Bipartite edges the speaker and entity networks are projected from, plus the persisted
    # topic co-occurrence edges. Read-only, paged, and scoped to one persona.
    # `since`/`until` are inclusive ISO dates read from the SPOKE edge; when either is given,
    # an edge with no date is left out, because an undated post cannot be put in a window.
    # Each row carries the speaker's attributes and the document's, so an attribute filter
    # never costs a second read.
    def speaker_document_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[SpeakerDocument]: ...
    # Each row carries the mention's tier, which is the evidence its edge rests on and what
    # `graphrag.sna.export` turns into the edge probability `p` (Atlas §28.1).
    def entity_chunk_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityChunk]: ...
    def topic_edges(self, persona_id: str, min_weight: int = 1) -> list[TopicEdge]: ...
    # The directed, typed relations an extraction pass wrote between entities, scoped through
    # this persona's passages exactly as `entity_chunk_pairs` is: the Entity nodes are shared,
    # so "this persona's relations" only ever means the ones its passages state.
    # One row per (relation, passage), so a relation stated in three passages comes back three
    # times and the network weighs it 3. A row whose passage or whose endpoints the graph no
    # longer holds is not returned at all.
    def relation_rows(self, persona_id: str, source_id: str | None = None) -> list[RelationRow]: ...
    # Mean chunk embedding per document or per entity, L2-normalised, for clustering by content
    # rather than by graph structure. Returns the key order and a matrix aligned to it.
    def mean_embeddings(
        self, persona_id: str, level: str = "document"
    ) -> tuple[list[str], Matrix]: ...
    # The same bipartite edges as `entity_chunk_pairs`, carrying the annotation layer's stance
    # and the passage's speakers. Read by the signed entity network (`--stance`) and by the
    # speakers-entities bipartite network, which cannot be assembled from the two reads
    # separately because both need the mention and the speaker on the *same* passage.
    def entity_mention_rows(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityMention]: ...
