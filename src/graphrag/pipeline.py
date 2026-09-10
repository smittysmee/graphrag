"""Ingest pipeline: load -> chunk -> embed -> upsert graph -> topic co-occurrence."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graphrag.embed.base import Embedder
from graphrag.extract.topics import topic_cooccurrence
from graphrag.graph.store import GraphStore
from graphrag.ingest.chunker import ChunkerConfig, chunk_document
from graphrag.ingest.loaders import load_source
from graphrag.models import Chunk, Document, PersonaSpec, SourceSpec

Progress = Callable[[str], None]


@dataclass
class IngestReport:
    persona_id: str
    source_id: str
    documents: int = 0
    chunks: int = 0
    seconds: float = 0.0
    embedding_model: str = ""
    skipped: list[str] = field(default_factory=list)


class IngestPipeline:
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        *,
        chunker: ChunkerConfig | None = None,
        doc_batch: int = 8,
        progress: Progress | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunker = chunker or ChunkerConfig()
        self._doc_batch = doc_batch
        self._progress = progress or (lambda _msg: None)

    def ingest(self, root: Path, persona: PersonaSpec, source: SourceSpec) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(
            persona_id=persona.id, source_id=source.id, embedding_model=self._embedder.model_name
        )
        self._store.ensure_schema(self._embedder.dim)
        self._store.upsert_persona(persona)

        pending_docs: list[Document] = []
        pending_chunks: list[Chunk] = []
        for loaded in load_source(root, source, persona.id):
            chunks = chunk_document(loaded, self._chunker)
            if not chunks:
                report.skipped.append(loaded.document.path)
                continue
            pending_docs.append(loaded.document)
            pending_chunks.extend(chunks)
            if len(pending_docs) >= self._doc_batch:
                self._flush(pending_docs, pending_chunks, report)
                pending_docs, pending_chunks = [], []
        if pending_docs:
            self._flush(pending_docs, pending_chunks, report)

        all_docs = list(self._store.iter_documents(persona.id))
        self._store.upsert_topic_cooccurrence(topic_cooccurrence(all_docs))
        report.seconds = time.perf_counter() - started
        return report

    def _flush(self, docs: list[Document], chunks: list[Chunk], report: IngestReport) -> None:
        matrix = self._embedder.embed_documents([c.text for c in chunks])
        if matrix.shape != (len(chunks), self._embedder.dim):
            msg = f"embedder returned {matrix.shape}, expected {(len(chunks), self._embedder.dim)}"
            raise ValueError(msg)
        self._store.delete_documents([d.id for d in docs])  # re-ingest replaces, never merges
        self._store.upsert_documents(docs)
        self._store.upsert_chunks(chunks, np.asarray(matrix, dtype=np.float32))
        report.documents += len(docs)
        report.chunks += len(chunks)
        self._progress(f"{report.documents} documents / {report.chunks} chunks ({docs[-1].title})")
