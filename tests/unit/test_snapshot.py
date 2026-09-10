from pathlib import Path

import numpy as np
import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec
from graphrag.pipeline import IngestReport


def test_export_then_load_into_fresh_store(
    tmp_path: Path,
    ingested: IngestReport,
    memory_store: InMemoryGraphStore,
    persona: PersonaSpec,
    hash_embedder: HashEmbedder,
) -> None:
    root = tmp_path / "snapshots"
    manifest = snap.export_snapshot(
        memory_store,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=hash_embedder.dim,
        source_commit="abc",
    )
    assert manifest.chunk_count == ingested.chunks
    assert (root / persona.id / "embeddings.npy").exists()
    assert snap.list_snapshots(root)[0].persona_id == persona.id

    fresh = InMemoryGraphStore()
    progress: list[str] = []
    loaded = snap.load_snapshot(
        fresh,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=hash_embedder.dim,
        on_progress=progress.append,
    )
    assert loaded.document_count == 3
    assert fresh.stats().chunks == memory_store.stats().chunks
    assert progress[-1].endswith("loaded")
    q = hash_embedder.embed_query("leaky bucket")
    assert fresh.vector_search(q, 1)[0].chunk_id == memory_store.vector_search(q, 1)[0].chunk_id
    assert fresh.related_topics("retention")
    for cid, vec in memory_store.embeddings.items():
        assert np.allclose(vec, fresh.embeddings[cid])


def test_load_refuses_other_embedding_model(
    tmp_path: Path, ingested: IngestReport, memory_store: InMemoryGraphStore, persona: PersonaSpec
) -> None:
    root = tmp_path / "snapshots"
    snap.export_snapshot(memory_store, persona, root, embedding_model="hash-test", embedding_dim=64)
    with pytest.raises(snap.SnapshotError, match="GRAPHRAG_EMBEDDING_MODEL=hash-test"):
        snap.load_snapshot(
            InMemoryGraphStore(),
            persona,
            root,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )


def test_missing_manifest(tmp_path: Path, persona: PersonaSpec) -> None:
    with pytest.raises(snap.SnapshotError):
        snap.read_manifest(tmp_path / "nope")
    assert snap.list_snapshots(tmp_path / "nope") == []


def test_export_preserves_source_commit(
    tmp_path: Path, ingested: IngestReport, memory_store: InMemoryGraphStore, persona: PersonaSpec
) -> None:
    root = tmp_path / "snapshots"
    snap.export_snapshot(
        memory_store,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=64,
        source_commit="abc",
    )
    again = snap.export_snapshot(
        memory_store, persona, root, embedding_model="hash-test", embedding_dim=64
    )
    assert again.source_commit == "abc"
