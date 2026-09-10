"""In-memory ``GraphStore``. Used by unit tests and as an executable spec of the store contract."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from graphrag.embed.base import Matrix, Vector
from graphrag.models import (
    Chunk,
    Document,
    Enrichment,
    Entity,
    GraphStats,
    Mention,
    PersonaSpec,
    RelatedTopic,
    Relation,
    ScoredChunk,
    SpeakerCount,
    TopicCount,
)

TOKEN = re.compile(r"[a-z0-9]+")


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

    def upsert_enrichment(self, enrichment: Enrichment) -> None:
        for entity in enrichment.entities:
            self.entities[entity.id] = entity
        existing = {(m.chunk_id, m.entity_id) for m in self.mentions}
        for mention in enrichment.mentions:
            if (mention.chunk_id, mention.entity_id) not in existing:
                self.mentions.append(mention)
                existing.add((mention.chunk_id, mention.entity_id))
        seen = {(r.source_id, r.target_id, r.type, r.chunk_id) for r in self.relations}
        for rel in enrichment.relations:
            if (rel.source_id, rel.target_id, rel.type, rel.chunk_id) not in seen:
                self.relations.append(rel)

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

    def enriched_doc_ids(self, persona_id: str) -> set[str]:
        chunk_to_doc = {c.id: c.doc_id for c in self.chunks.values() if c.persona_id == persona_id}
        return {chunk_to_doc[m.chunk_id] for m in self.mentions if m.chunk_id in chunk_to_doc}

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
