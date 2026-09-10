"""Portable graph snapshots: what gets committed to git so teammates skip ingest + embedding.

Layout of ``data/snapshots/<persona>/``::

    manifest.json        model name, dim, counts, source commit
    documents.jsonl.gz   one Document per line
    chunks.jsonl.gz      one Chunk per line (no vectors), in the same order as embeddings.npy
    embeddings.npy       float32 (n_chunks, dim)
    enrichment.json.gz   entities / mentions / relations (may be empty)
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from graphrag import __version__
from graphrag.extract.topics import topic_cooccurrence
from graphrag.graph.store import GraphStore
from graphrag.models import Chunk, Document, Enrichment, PersonaSpec, SnapshotManifest

MANIFEST = "manifest.json"
DOCUMENTS = "documents.jsonl.gz"
CHUNKS = "chunks.jsonl.gz"
EMBEDDINGS = "embeddings.npy"
ENRICHMENT = "enrichment.json.gz"
LOAD_BATCH = 1000


class SnapshotError(RuntimeError):
    pass


def snapshot_dir(root: Path, persona_id: str) -> Path:
    return root / persona_id


def read_manifest(path: Path) -> SnapshotManifest:
    manifest_file = path / MANIFEST
    if not manifest_file.exists():
        msg = f"no snapshot manifest at {manifest_file}"
        raise SnapshotError(msg)
    return SnapshotManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))


def list_snapshots(root: Path) -> list[SnapshotManifest]:
    if not root.exists():
        return []
    out: list[SnapshotManifest] = []
    for child in sorted(root.iterdir()):
        if (child / MANIFEST).exists():
            out.append(read_manifest(child))
    return out


def export_snapshot(
    store: GraphStore,
    persona: PersonaSpec,
    root: Path,
    *,
    embedding_model: str,
    embedding_dim: int,
    source_commit: str | None = None,
) -> SnapshotManifest:
    target = snapshot_dir(root, persona.id)
    target.mkdir(parents=True, exist_ok=True)
    if source_commit is None and (target / MANIFEST).exists():
        source_commit = read_manifest(target).source_commit

    documents = list(store.iter_documents(persona.id))
    with gzip.open(target / DOCUMENTS, "wt", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc.model_dump_json() + "\n")

    vectors: list[np.ndarray] = []
    n_chunks = 0
    with gzip.open(target / CHUNKS, "wt", encoding="utf-8") as fh:
        for chunk, vec in store.iter_chunks(persona.id):
            fh.write(chunk.model_dump_json() + "\n")
            vectors.append(np.asarray(vec, dtype=np.float32))
            n_chunks += 1
    matrix = np.stack(vectors) if vectors else np.zeros((0, embedding_dim), dtype=np.float32)
    if matrix.shape[1] != embedding_dim:
        msg = f"stored vectors have dim {matrix.shape[1]}, expected {embedding_dim}"
        raise SnapshotError(msg)
    np.save(target / EMBEDDINGS, matrix)

    enrichment = store.enrichment_for_persona(persona.id)
    with gzip.open(target / ENRICHMENT, "wt", encoding="utf-8") as fh:
        fh.write(enrichment.model_dump_json())

    manifest = SnapshotManifest(
        persona_id=persona.id,
        created_at=datetime.now(tz=UTC),
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        document_count=len(documents),
        chunk_count=n_chunks,
        entity_count=len(enrichment.entities),
        sources=persona.sources,
        source_commit=source_commit,
        tool_version=__version__,
    )
    (target / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def check_embedding_compat(manifest: SnapshotManifest, model_name: str, dim: int) -> None:
    if manifest.embedding_model != model_name or manifest.embedding_dim != dim:
        msg = (
            f"snapshot '{manifest.persona_id}' was built with {manifest.embedding_model} "
            f"({manifest.embedding_dim} dims) but the configured embedder is {model_name} "
            f"({dim} dims). Query vectors must come from the same model: set "
            f"GRAPHRAG_EMBEDDING_MODEL={manifest.embedding_model} and "
            f"GRAPHRAG_EMBEDDING_DIM={manifest.embedding_dim}, or re-ingest."
        )
        raise SnapshotError(msg)


def load_snapshot(
    store: GraphStore,
    persona: PersonaSpec,
    root: Path,
    *,
    embedding_model: str,
    embedding_dim: int,
    on_progress: Callable[[str], None] | None = None,
) -> SnapshotManifest:
    target = snapshot_dir(root, persona.id)
    manifest = read_manifest(target)
    check_embedding_compat(manifest, embedding_model, embedding_dim)

    store.ensure_schema(embedding_dim)
    store.upsert_persona(persona)

    documents: list[Document] = []
    with gzip.open(target / DOCUMENTS, "rt", encoding="utf-8") as fh:
        documents.extend(Document.model_validate_json(line) for line in fh if line.strip())
    store.upsert_documents(documents)
    if on_progress:
        on_progress(f"{persona.id}: {len(documents)} documents")

    matrix = np.load(target / EMBEDDINGS)
    if matrix.shape[0] != manifest.chunk_count:
        msg = f"embeddings.npy has {matrix.shape[0]} rows, manifest says {manifest.chunk_count}"
        raise SnapshotError(msg)
    batch: list[Chunk] = []
    offset = 0
    with gzip.open(target / CHUNKS, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            batch.append(Chunk.model_validate_json(line))
            if len(batch) == LOAD_BATCH:
                store.upsert_chunks(batch, matrix[offset : offset + len(batch)])
                offset += len(batch)
                batch = []
                if on_progress:
                    on_progress(f"{persona.id}: {offset}/{manifest.chunk_count} chunks")
    if batch:
        store.upsert_chunks(batch, matrix[offset : offset + len(batch)])
        offset += len(batch)
    if offset != manifest.chunk_count:
        msg = f"loaded {offset} chunks, manifest says {manifest.chunk_count}"
        raise SnapshotError(msg)

    store.upsert_topic_cooccurrence(topic_cooccurrence(documents))

    enrichment_file = target / ENRICHMENT
    if enrichment_file.exists():
        with gzip.open(enrichment_file, "rt", encoding="utf-8") as fh:
            enrichment = Enrichment.model_validate_json(fh.read())
        store.upsert_enrichment(enrichment)
    if on_progress:
        on_progress(f"{persona.id}: loaded")
    return manifest


def manifest_summary(manifest: SnapshotManifest) -> dict[str, object]:
    data: dict[str, object] = json.loads(manifest.model_dump_json())
    return data
