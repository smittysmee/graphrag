"""In-memory ``GraphStore``. Used by unit tests and as an executable spec of the store contract."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from graphrag.embed.base import Matrix, Vector
from graphrag.graph.vectors import stack_means
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
    Mention,
    MentionStance,
    PersonaSpec,
    RelatedTopic,
    Relation,
    ScoredChunk,
    SpeakerCount,
    SpeakerDocument,
    SpeakerPost,
    Stance,
    TopicCount,
    TopicEdge,
    merge_entity,
)

TOKEN = re.compile(r"[a-z0-9]+")


def _fold(name: str) -> str:
    """Lower case with runs of whitespace collapsed: how two spellings are compared."""
    return " ".join(name.split()).casefold()


def _in_window(posted_at: str | None, since: str | None, until: str | None) -> bool:
    """Whether an ISO date falls in an inclusive window. An undated post never does."""
    if posted_at is None:
        return False
    return (since is None or posted_at >= since) and (until is None or posted_at <= until)


def _holds(chunk: Chunk, name: str, since: str | None, until: str | None, windowed: bool) -> bool:
    """Whether ``name`` holds this passage, inside the window when one is set."""
    if not windowed:
        return name in chunk.speakers
    return any(
        p.speaker == name and _in_window(p.posted_at, since, until) for p in chunk.speaker_posts
    )


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.personas: dict[str, PersonaSpec] = {}
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.cooccurrence: dict[tuple[str, str], int] = {}
        self.entities: dict[str, Entity] = {}
        self.mentions: list[Mention] = []
        self.relations: list[Relation] = []
        #: (persona_id, speaker) -> attributes. Keyed on the persona because a speaker name is
        #: shared between personas exactly as the Neo4j ``Speaker`` node is.
        self.speaker_attrs: dict[tuple[str, str], dict[str, str]] = {}
        self.dim: int | None = None

    # ------------------------------------------------------------- lifecycle
    def ensure_schema(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            msg = f"store initialised with dim={self.dim}, got {dim}"
            raise ValueError(msg)
        self.dim = dim

    def close(self) -> None:
        return None

    # ------------------------------------------------------------- writes
    def upsert_persona(self, persona: PersonaSpec) -> None:
        self.personas[persona.id] = persona

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        for doc in documents:
            self.documents[doc.id] = doc

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None:
        if len(chunks) != len(embeddings):
            msg = f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            raise ValueError(msg)
        for chunk, vec in zip(chunks, embeddings, strict=True):
            if self.dim is not None and vec.shape[0] != self.dim:
                msg = f"embedding dim {vec.shape[0]} != schema dim {self.dim}"
                raise ValueError(msg)
            self.chunks[chunk.id] = chunk
            self.embeddings[chunk.id] = np.asarray(vec, dtype=np.float32)

    def upsert_topic_cooccurrence(self, pairs: Sequence[tuple[str, str, int]]) -> None:
        for a, b, weight in pairs:
            key = (a, b) if a <= b else (b, a)
            self.cooccurrence[key] = weight

    def upsert_enrichment(self, enrichment: Enrichment) -> list[EntityCollision]:
        collisions: list[EntityCollision] = []
        counts = Counter(m.entity_id for m in self.mentions)
        for entity in enrichment.entities:
            held = self.entities.get(entity.id)
            resolved, collision = merge_entity(
                held, entity, held_mentions=None if held is None else counts[entity.id]
            )
            self.entities[entity.id] = resolved
            if collision is not None:
                collisions.append(collision)
        existing = {(m.chunk_id, m.entity_id): i for i, m in enumerate(self.mentions)}
        for mention in enrichment.mentions:
            key = (mention.chunk_id, mention.entity_id)
            index = existing.get(key)
            if index is None:
                self.mentions.append(mention)
                existing[key] = len(self.mentions) - 1
            elif mention.stance is not None:
                # A re-import must not erase an annotation, but may set one that was missing.
                self.mentions[index] = self.mentions[index].model_copy(
                    update={"stance": mention.stance}
                )
        seen = {(r.source_id, r.target_id, r.type, r.chunk_id) for r in self.relations}
        for rel in enrichment.relations:
            if (rel.source_id, rel.target_id, rel.type, rel.chunk_id) not in seen:
                self.relations.append(rel)
        return collisions

    def merge_entities(self, persona_id: str, canonical: str, aliases: Sequence[str]) -> int:
        """Fold every alias spelling of ``canonical`` into one node. Returns mentions moved.

        Scoped to one persona on purpose: entity nodes are shared, so another persona's mentions
        of an alias spelling are left where they are and the alias node survives. Only a node
        nothing points at any more is deleted.
        """
        spellings = {_fold(a) for a in aliases if a.strip()} | {_fold(canonical)}
        matched = [e for e in self.entities.values() if _fold(e.name) in spellings]
        if not matched:
            return 0
        chunk_ids = {c.id for c in self.chunks.values() if c.persona_id == persona_id}
        counts = Counter(m.entity_id for m in self.mentions if m.chunk_id in chunk_ids)
        best = max(
            matched,
            key=lambda e: (_fold(e.name) == _fold(canonical), counts[e.id], e.id),
        )
        target_id = Entity.make_id(canonical, best.type)
        held = self.entities.get(target_id)
        self.entities[target_id] = Entity(
            id=target_id,
            name=canonical,
            type=best.type,
            description=(held.description if held is not None else "") or best.description,
            aliases=sorted(
                {a for e in matched for a in e.aliases}
                | {a.strip() for a in aliases if a.strip() and _fold(a) != _fold(canonical)}
            ),
        )
        stale = {e.id for e in matched if e.id != target_id}
        if not stale:
            return 0

        moved = 0
        kept: list[Mention] = []
        stances: dict[tuple[str, str], Stance | None] = {}
        for mention in self.mentions:
            if mention.entity_id in stale and mention.chunk_id in chunk_ids:
                moved += 1
                mention = mention.model_copy(update={"entity_id": target_id})
            key = (mention.chunk_id, mention.entity_id)
            if mention.entity_id == target_id:
                if key in stances:
                    stances[key] = stances[key] or mention.stance
                    continue
                stances[key] = mention.stance
            kept.append(mention)
        self.mentions = [
            m.model_copy(update={"stance": stances[(m.chunk_id, m.entity_id)]})
            if m.entity_id == target_id
            else m
            for m in kept
        ]

        relations: list[Relation] = []
        seen: set[tuple[str, str, str, str]] = set()
        for rel in self.relations:
            if rel.chunk_id not in chunk_ids:
                relations.append(rel)
                continue
            source = target_id if rel.source_id in stale else rel.source_id
            dest = target_id if rel.target_id in stale else rel.target_id
            if source == dest:
                continue  # two spellings of one thing cannot be related to each other
            edge = (source, dest, rel.type, rel.chunk_id)
            if edge in seen:
                continue
            seen.add(edge)
            relations.append(rel.model_copy(update={"source_id": source, "target_id": dest}))
        self.relations = relations

        alive = {m.entity_id for m in self.mentions} | {
            e for r in self.relations for e in (r.source_id, r.target_id)
        }
        for entity_id in stale - alive:
            self.entities.pop(entity_id, None)
        return moved

    def delete_orphan_entities(self, persona_id: str, *, dry_run: bool = False) -> int:
        """Drop the entity nodes nothing mentions any more. Returns how many went.

        Scoped the way ``merge_entities`` is: entity nodes are shared, so a node another
        persona's passage still relates to survives even with no mention on it. A relation whose
        passage is gone holds nothing, so it does not keep its endpoints alive.
        """
        mentioned = {m.entity_id for m in self.mentions}
        foreign = {
            entity_id
            for r in self.relations
            if (chunk := self.chunks.get(r.chunk_id)) is not None and chunk.persona_id != persona_id
            for entity_id in (r.source_id, r.target_id)
        }
        orphans = set(self.entities) - mentioned - foreign
        if dry_run or not orphans:
            return len(orphans)
        for entity_id in orphans:
            del self.entities[entity_id]
        self.relations = [
            r for r in self.relations if r.source_id not in orphans and r.target_id not in orphans
        ]
        return len(orphans)

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        wanted = set(doc_ids)
        chunk_ids = {c for c, ch in self.chunks.items() if ch.doc_id in wanted}
        for c in chunk_ids:
            del self.chunks[c]
            self.embeddings.pop(c, None)
        for d in wanted:
            self.documents.pop(d, None)
        self.mentions = [m for m in self.mentions if m.chunk_id not in chunk_ids]
        self.relations = [r for r in self.relations if r.chunk_id not in chunk_ids]

    def delete_persona(self, persona_id: str) -> None:
        doc_ids = {d for d, doc in self.documents.items() if doc.persona_id == persona_id}
        chunk_ids = {c for c, ch in self.chunks.items() if ch.persona_id == persona_id}
        for d in doc_ids:
            del self.documents[d]
        for c in chunk_ids:
            del self.chunks[c]
            self.embeddings.pop(c, None)
        self.mentions = [m for m in self.mentions if m.chunk_id not in chunk_ids]
        self.relations = [r for r in self.relations if r.chunk_id not in chunk_ids]
        self.personas.pop(persona_id, None)
        self.speaker_attrs = {k: v for k, v in self.speaker_attrs.items() if k[0] != persona_id}
        self.delete_orphan_entities(persona_id)

    # ------------------------------------------------------------- search
    def _persona_chunk_ids(self, persona_id: str | None) -> list[str]:
        if persona_id is None:
            return list(self.chunks)
        return [cid for cid, ch in self.chunks.items() if ch.persona_id == persona_id]

    def vector_search(
        self, vector: Vector, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]:
        ids = [cid for cid in self._persona_chunk_ids(persona_id) if cid in self.embeddings]
        if not ids:
            return []
        matrix = np.stack([self.embeddings[cid] for cid in ids])
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk_id=ids[i], score=float(scores[i])) for i in order]

    def fulltext_search(
        self, query: str, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]:
        terms = set(TOKEN.findall(query.lower()))
        if not terms:
            return []
        scored: list[ScoredChunk] = []
        for cid in self._persona_chunk_ids(persona_id):
            tokens = Counter(TOKEN.findall(self.chunks[cid].text.lower()))
            hits = sum(tokens[t] for t in terms)
            distinct = sum(1 for t in terms if tokens[t])
            if hits:
                scored.append(ScoredChunk(chunk_id=cid, score=distinct + hits / 100.0))
        scored.sort(key=lambda s: -s.score)
        return scored[:k]

    # ------------------------------------------------------------- reads
    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self.chunks[c] for c in chunk_ids if c in self.chunks]

    def neighbors(self, chunk_id: str, hops: int = 1) -> list[Chunk]:
        chunk = self.chunks.get(chunk_id)
        if chunk is None or hops <= 0:
            return []
        return sorted(
            (
                c
                for c in self.chunks.values()
                if c.doc_id == chunk.doc_id
                and c.id != chunk_id
                and abs(c.ordinal - chunk.ordinal) <= hops
            ),
            key=lambda c: c.ordinal,
        )

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def list_documents(
        self,
        persona_id: str | None = None,
        *,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 50,
    ) -> list[Document]:
        docs = [
            d
            for d in self.documents.values()
            if (persona_id is None or d.persona_id == persona_id)
            and (topic is None or topic in d.topics)
            and (speaker is None or speaker in d.speakers)
        ]
        docs.sort(key=lambda d: (d.published or "", d.title), reverse=True)
        return docs[:limit]

    def document_ids(self, persona_id: str, source_id: str | None = None) -> set[str]:
        return {
            doc_id
            for doc_id, doc in self.documents.items()
            if doc.persona_id == persona_id and (source_id is None or doc.source_id == source_id)
        }

    def document_chunks(self, doc_id: str, start: int = 0, count: int = 5) -> list[Chunk]:
        chunks = sorted(
            (c for c in self.chunks.values() if c.doc_id == doc_id), key=lambda c: c.ordinal
        )
        return chunks[start : start + count]

    def list_topics(self, persona_id: str | None = None, limit: int = 50) -> list[TopicCount]:
        counter: Counter[str] = Counter()
        for doc in self.documents.values():
            if persona_id is None or doc.persona_id == persona_id:
                counter.update(doc.topics)
        return [TopicCount(topic=t, count=n) for t, n in counter.most_common(limit)]

    def related_topics(
        self, topic: str, persona_id: str | None = None, limit: int = 10
    ) -> list[RelatedTopic]:
        out = [
            RelatedTopic(topic=b if a == topic else a, weight=w)
            for (a, b), w in self.cooccurrence.items()
            if topic in (a, b)
        ]
        out.sort(key=lambda r: -r.weight)
        return out[:limit]

    def list_speakers(self, persona_id: str | None = None, limit: int = 50) -> list[SpeakerCount]:
        counter: Counter[str] = Counter()
        for doc in self.documents.values():
            if persona_id is None or doc.persona_id == persona_id:
                counter.update(doc.speakers)
        return [SpeakerCount(speaker=s, documents=n) for s, n in counter.most_common(limit)]

    def list_personas(self) -> list[PersonaSpec]:
        return sorted(self.personas.values(), key=lambda p: p.id)

    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> list[Entity]:
        wanted = set(chunk_ids)
        ids = {m.entity_id for m in self.mentions if m.chunk_id in wanted}
        return sorted((self.entities[e] for e in ids if e in self.entities), key=lambda e: e.name)

    def persona_entities(self, persona_id: str) -> list[Entity]:
        chunk_ids = {c.id for c in self.chunks.values() if c.persona_id == persona_id}
        ids = {m.entity_id for m in self.mentions if m.chunk_id in chunk_ids}
        return sorted((self.entities[e] for e in ids if e in self.entities), key=lambda e: e.id)

    def enriched_doc_ids(self, persona_id: str) -> set[str]:
        chunk_to_doc = {c.id: c.doc_id for c in self.chunks.values() if c.persona_id == persona_id}
        return {chunk_to_doc[m.chunk_id] for m in self.mentions if m.chunk_id in chunk_to_doc}

    def enriched_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        wanted = self.document_ids(persona_id, source_id)
        chunk_to_doc = {c.id: c.doc_id for c in self.chunks.values() if c.doc_id in wanted}
        # Neo4j can only hold a MENTIONS edge to an Entity node, so an entity-less mention
        # does not count here either.
        return {
            chunk_to_doc[m.chunk_id]
            for m in self.mentions
            if m.chunk_id in chunk_to_doc and m.entity_id in self.entities
        }

    # ------------------------------------------------------------- bulk
    def iter_documents(self, persona_id: str) -> Iterator[Document]:
        yield from sorted(
            (d for d in self.documents.values() if d.persona_id == persona_id), key=lambda d: d.id
        )

    def iter_chunks(self, persona_id: str) -> Iterator[tuple[Chunk, Vector]]:
        for chunk in sorted(
            (c for c in self.chunks.values() if c.persona_id == persona_id),
            key=lambda c: (c.doc_id, c.ordinal),
        ):
            yield chunk, self.embeddings[chunk.id]

    def enrichment_for_persona(self, persona_id: str) -> Enrichment:
        chunk_ids = {c.id for c in self.chunks.values() if c.persona_id == persona_id}
        mentions = [m for m in self.mentions if m.chunk_id in chunk_ids]
        relations = [r for r in self.relations if r.chunk_id in chunk_ids]
        entity_ids = {m.entity_id for m in mentions} | {
            e for r in relations for e in (r.source_id, r.target_id)
        }
        entities = [self.entities[e] for e in sorted(entity_ids) if e in self.entities]
        return Enrichment(entities=entities, mentions=mentions, relations=relations)

    # ------------------------------------------------------------- misc
    def stats(self) -> GraphStats:
        per: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "chunks": 0})
        for doc in self.documents.values():
            per[doc.persona_id]["documents"] += 1
        for chunk in self.chunks.values():
            per[chunk.persona_id]["chunks"] += 1
        speakers = {s for d in self.documents.values() for s in d.speakers}
        topics = {t for d in self.documents.values() for t in d.topics}
        return GraphStats(
            personas=len(self.personas),
            sources=sum(len(p.sources) for p in self.personas.values()),
            documents=len(self.documents),
            chunks=len(self.chunks),
            speakers=len(speakers),
            topics=len(topics),
            entities=len(self.entities),
            per_persona=dict(per),
        )

    def run_readonly_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        msg = "Cypher is only available on the Neo4j store"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------- attribution
    def attach_speaker(
        self,
        doc_id: str,
        chunk_id: str,
        speaker: str,
        *,
        posted_at: str | None = None,
        role: str | None = None,
        score: int | None = None,
    ) -> None:
        """Record that ``speaker`` wrote the passage ``chunk_id`` of document ``doc_id``.

        Stored on the ``speakers`` lists rather than in a side table, because that is what the
        Neo4j store builds its ``Speaker`` nodes from on upsert -- so an attributed document
        survives a snapshot export and load with its speakers intact. The singular ``speaker``
        field is left alone: it means "the voice of this turn", and one passage of a thread can
        hold posts by several people.

        ``posted_at``, ``role`` and ``score`` describe the post rather than the person, so they
        go on ``speaker_posts``, which is the passage's copy of what Neo4j keeps on ``SPOKE``.
        Re-attaching replaces that record rather than adding a second, because one speaker holds
        one passage once however many times a file says so.
        """
        chunk = self.chunks.get(chunk_id)
        document = self.documents.get(doc_id)
        if chunk is None or document is None or chunk.doc_id != doc_id:
            return
        update: dict[str, Any] = {}
        if speaker not in chunk.speakers:
            update["speakers"] = [*chunk.speakers, speaker]
        if posted_at is not None or role is not None or score is not None:
            post = SpeakerPost(speaker=speaker, posted_at=posted_at, role=role, score=score)
            update["speaker_posts"] = [
                *(p for p in chunk.speaker_posts if p.speaker != speaker),
                post,
            ]
        if update:
            self.chunks[chunk_id] = chunk.model_copy(update=update)
        if speaker not in document.speakers:
            self.documents[doc_id] = document.model_copy(
                update={"speakers": [*document.speakers, speaker]}
            )

    def attributed_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        wanted = self.document_ids(persona_id, source_id)
        return {doc_id for doc_id in wanted if self.documents[doc_id].speakers}

    # ------------------------------------------------------------- node attributes
    def set_speaker_attributes(
        self, persona_id: str, speaker: str, attributes: Mapping[str, str]
    ) -> list[str]:
        """Record what this persona's corpus says the speaker is. Returns the keys it refused.

        First value written wins: a key already held under a different value is left alone and
        comes back for the caller to report. An identical value is not a conflict, so
        re-importing the same file is the no-op every other import here is.
        """
        held = self.speaker_attrs.setdefault((persona_id, speaker), {})
        refused = [key for key, value in attributes.items() if held.get(key, value) != value]
        for key, value in attributes.items():
            held.setdefault(key, value)
        return refused

    def set_document_attributes(self, doc_id: str, attributes: Mapping[str, str]) -> None:
        document = self.documents.get(doc_id)
        if document is None:
            return
        merged = {**document.attributes, **attributes}
        if merged != document.attributes:
            self.documents[doc_id] = document.model_copy(update={"attributes": merged})

    def speaker_attributes(self, persona_id: str) -> dict[str, dict[str, str]]:
        return {
            speaker: dict(attrs)
            for (pid, speaker), attrs in self.speaker_attrs.items()
            if pid == persona_id and attrs
        }

    def document_attributes(self, persona_id: str) -> dict[str, dict[str, str]]:
        return {
            doc.id: dict(doc.attributes)
            for doc in self.documents.values()
            if doc.persona_id == persona_id and doc.attributes
        }

    # ------------------------------------------------------------- annotation
    def _mention_index(self, doc_id: str, chunk_id: str, entity: str) -> int | None:
        """Where the ``MENTIONS`` edge this annotation is about sits, or ``None``.

        An entity the passage does not already mention has no edge here to annotate, and a
        passage that is not this document's is not this document's to annotate either.
        """
        chunk = self.chunks.get(chunk_id)
        if chunk is None or chunk.doc_id != doc_id:
            return None
        wanted = _fold(entity)
        for index, mention in enumerate(self.mentions):
            if mention.chunk_id != chunk_id:
                continue
            found = self.entities.get(mention.entity_id)
            if found is not None and (_fold(found.name) == wanted or found.id == entity.strip()):
                return index
        return None

    def annotate_mention(self, doc_id: str, chunk_id: str, entity: str, stance: Stance) -> None:
        index = self._mention_index(doc_id, chunk_id, entity)
        if index is not None:
            self.mentions[index] = self.mentions[index].model_copy(update={"stance": stance})

    def annotate_chunk(self, doc_id: str, chunk_id: str, facets: Sequence[str]) -> None:
        chunk = self.chunks.get(chunk_id)
        if chunk is None or chunk.doc_id != doc_id:
            return
        merged = [*chunk.facets, *(f for f in facets if f not in chunk.facets)]
        if merged != chunk.facets:
            self.chunks[chunk_id] = chunk.model_copy(update={"facets": merged})

    def annotated_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        wanted = self.document_ids(persona_id, source_id)
        chunk_to_doc = {c.id: c.doc_id for c in self.chunks.values() if c.doc_id in wanted}
        annotated = {
            chunk_to_doc[c.id] for c in self.chunks.values() if c.id in chunk_to_doc and c.facets
        }
        annotated |= {
            chunk_to_doc[m.chunk_id]
            for m in self.mentions
            if m.stance is not None and m.chunk_id in chunk_to_doc
        }
        # An annotation file that carries only a top-level ``attributes`` block, with no
        # ``annotations`` entries, still landed something: the document's attributes. Without
        # this, that import looks identical to one that never ran.
        annotated |= {doc_id for doc_id in wanted if self.documents[doc_id].attributes}
        return annotated

    def mention_stances(self, persona_id: str, source_id: str | None = None) -> list[MentionStance]:
        wanted = self.document_ids(persona_id, source_id)
        rows = []
        for mention in self.mentions:
            chunk = self.chunks.get(mention.chunk_id)
            entity = self.entities.get(mention.entity_id)
            if mention.stance is None or chunk is None or entity is None:
                continue
            if chunk.doc_id not in wanted:
                continue
            rows.append(
                MentionStance(
                    entity_id=entity.id,
                    name=entity.name,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    stance=mention.stance,
                    speakers=list(chunk.speakers),
                )
            )
        rows.sort(key=lambda r: (r.entity_id, r.chunk_id))
        return rows

    def chunk_facets(self, persona_id: str, source_id: str | None = None) -> list[ChunkFacets]:
        wanted = self.document_ids(persona_id, source_id)
        rows = [
            ChunkFacets(chunk_id=c.id, doc_id=c.doc_id, facets=list(c.facets))
            for c in self.chunks.values()
            if c.facets and c.doc_id in wanted
        ]
        rows.sort(key=lambda r: r.chunk_id)
        return rows

    # ------------------------------------------------------------- network analysis
    def speaker_document_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[SpeakerDocument]:
        wanted = self.document_ids(persona_id, source_id)
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunks.values():
            if chunk.doc_id in wanted:
                by_doc[chunk.doc_id].append(chunk)
        windowed = since is not None or until is not None
        rows: list[SpeakerDocument] = []
        for doc_id in sorted(wanted):
            chunks = by_doc.get(doc_id, [])
            doc = self.documents[doc_id]
            # A speaker counts as present if the document credits them or if they hold a passage.
            names = set(doc.speakers) | {s for c in chunks for s in c.speakers}
            if windowed:
                # Only dated passages can be placed in a window, so a document credit on its own
                # is not enough and an undated post drops out rather than being assumed in range.
                names = {
                    p.speaker
                    for c in chunks
                    for p in c.speaker_posts
                    if _in_window(p.posted_at, since, until)
                }
            for name in sorted(names):
                spoken = sum(1 for c in chunks if _holds(c, name, since, until, windowed))
                rows.append(
                    SpeakerDocument(
                        speaker=name,
                        doc_id=doc_id,
                        chunks=spoken,
                        speaker_attributes=dict(self.speaker_attrs.get((persona_id, name), {})),
                        document_attributes=dict(doc.attributes),
                    )
                )
        rows.sort(key=lambda r: (r.speaker, r.doc_id))
        return rows

    def entity_chunk_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityChunk]:
        wanted = self.document_ids(persona_id, source_id)
        kinds = set(types) if types else None
        rows: list[EntityChunk] = []
        for mention in self.mentions:
            chunk = self.chunks.get(mention.chunk_id)
            entity = self.entities.get(mention.entity_id)
            if chunk is None or entity is None or chunk.doc_id not in wanted:
                continue
            if kinds is not None and entity.type not in kinds:
                continue
            rows.append(
                EntityChunk(
                    entity_id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                )
            )
        rows.sort(key=lambda r: (r.entity_id, r.chunk_id))
        return rows

    def topic_edges(self, persona_id: str, min_weight: int = 1) -> list[TopicEdge]:
        present = {
            t for d in self.documents.values() if d.persona_id == persona_id for t in d.topics
        }
        rows = [
            TopicEdge(source=a, target=b, weight=w)
            for (a, b), w in self.cooccurrence.items()
            if w >= min_weight and a in present and b in present
        ]
        rows.sort(key=lambda r: (r.source, r.target))
        return rows

    def mean_embeddings(self, persona_id: str, level: str = "document") -> tuple[list[str], Matrix]:
        sums: dict[str, np.ndarray] = {}
        counts: Counter[str] = Counter()

        def add(key: str, chunk_id: str) -> None:
            vec = self.embeddings.get(chunk_id)
            if vec is None:
                return
            sums[key] = sums.get(key, np.zeros_like(vec)) + vec
            counts[key] += 1

        if level == "document":
            for chunk in self.chunks.values():
                if chunk.persona_id == persona_id:
                    add(chunk.doc_id, chunk.id)
        elif level == "entity":
            for mention in self.mentions:
                mentioned = self.chunks.get(mention.chunk_id)
                if mentioned is not None and mentioned.persona_id == persona_id:
                    add(mention.entity_id, mentioned.id)
        else:
            msg = f"level must be 'document' or 'entity', got {level!r}"
            raise ValueError(msg)
        return stack_means(sums, counts)

    def entity_mention_rows(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityMention]:
        wanted = self.document_ids(persona_id, source_id)
        kinds = set(types) if types else None
        rows: list[EntityMention] = []
        for mention in self.mentions:
            chunk = self.chunks.get(mention.chunk_id)
            entity = self.entities.get(mention.entity_id)
            if chunk is None or entity is None or chunk.doc_id not in wanted:
                continue
            if kinds is not None and entity.type not in kinds:
                continue
            rows.append(
                EntityMention(
                    entity_id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    stance=mention.stance,
                    speakers=list(chunk.speakers),
                    document_attributes=dict(self.documents[chunk.doc_id].attributes),
                )
            )
        rows.sort(key=lambda r: (r.entity_id, r.chunk_id))
        return rows
