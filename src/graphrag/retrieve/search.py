"""Hybrid retrieval: vector + full-text fused with reciprocal-rank fusion, then graph expansion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from graphrag.embed.base import Embedder
from graphrag.graph.store import GraphStore
from graphrag.models import (
    ChunkHit,
    ContextPack,
    Document,
    PersonaSpec,
    ScoredChunk,
    SearchMode,
)

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[ScoredChunk]], k: int = RRF_K
) -> list[tuple[str, float, list[str]]]:
    """Fuse several rankings. Returns ``(chunk_id, fused_score, methods)`` best first."""
    scores: dict[str, float] = defaultdict(float)
    methods: dict[str, list[str]] = defaultdict(list)
    for method, ranked in ranked_lists.items():
        for rank, item in enumerate(ranked, start=1):
            scores[item.chunk_id] += 1.0 / (k + rank)
            methods[item.chunk_id].append(method)
    return sorted(
        ((cid, score, methods[cid]) for cid, score in scores.items()),
        key=lambda row: (-row[1], row[0]),
    )


class Retriever:
    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        *,
        persona_id: str | None = None,
        k: int = 8,
        mode: SearchMode = "hybrid",
        expand: int = 0,
    ) -> list[ChunkHit]:
        query = query.strip()
        if not query:
            return []
        rankings: dict[str, Sequence[ScoredChunk]] = {}
        oversample = k * 2
        if mode in ("hybrid", "vector"):
            vec = self._embedder.embed_query(query)
            rankings["vector"] = self._store.vector_search(vec, oversample, persona_id)
        if mode in ("hybrid", "fulltext"):
            rankings["fulltext"] = self._store.fulltext_search(query, oversample, persona_id)
        fused = reciprocal_rank_fusion(rankings)[:k]
        if not fused:
            return []

        chunk_ids = [cid for cid, _, _ in fused]
        chunks = {c.id: c for c in self._store.get_chunks(chunk_ids)}
        docs: dict[str, Document] = {}
        hits: list[ChunkHit] = []
        for cid, score, methods in fused:
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            if chunk.doc_id not in docs:
                doc = self._store.get_document(chunk.doc_id)
                if doc is None:
                    continue
                docs[chunk.doc_id] = doc
            hits.append(
                ChunkHit(
                    chunk=chunk,
                    document=docs[chunk.doc_id],
                    score=score,
                    methods=methods,
                    neighbors=self._store.neighbors(cid, expand) if expand > 0 else [],
                )
            )
        entity_map = {e.id: e for e in self._store.entities_for_chunks([h.chunk.id for h in hits])}
        if entity_map:
            for hit in hits:
                hit.entities = [
                    e for e in self._store.entities_for_chunks([hit.chunk.id]) if e.id in entity_map
                ]
        return hits

    def context(
        self,
        query: str,
        *,
        persona: PersonaSpec | None = None,
        k: int | None = None,
        expand: int | None = None,
        mode: SearchMode | None = None,
    ) -> ContextPack:
        retrieval = persona.retrieval if persona else None
        hits = self.search(
            query,
            persona_id=persona.id if persona else None,
            k=k or (retrieval.top_k if retrieval else 8),
            mode=mode or (retrieval.mode if retrieval else "hybrid"),
            expand=expand
            if expand is not None
            else (retrieval.expand_neighbors if retrieval else 1),
        )
        topics: list[str] = []
        for hit in hits:
            for topic in hit.document.topics:
                if topic not in topics:
                    topics.append(topic)
        entities = {e.id: e for hit in hits for e in hit.entities}
        return ContextPack(
            query=query,
            persona_id=persona.id if persona else None,
            persona_name=persona.name if persona else None,
            role_prompt=persona.role_prompt if persona else None,
            hits=hits,
            topics=topics[:12],
            entities=sorted(entities.values(), key=lambda e: e.name),
        )
