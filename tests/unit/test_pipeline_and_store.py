from pathlib import Path

import numpy as np
import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec, SourceSpec
from graphrag.pipeline import IngestPipeline, IngestReport


def test_ingest_populates_graph(ingested: IngestReport, memory_store: InMemoryGraphStore) -> None:
    assert ingested.documents == 3
    assert ingested.chunks >= 3
    s = memory_store.stats()
    assert s.documents == 3 and s.chunks == ingested.chunks and s.personas == 1
    assert s.per_persona["test-pm"]["documents"] == 3
    assert s.speakers == 4  # three guests + host
    assert "retention" in {t.topic for t in memory_store.list_topics("test-pm")}


def test_vector_search_finds_relevant_chunk(
    ingested: IngestReport, memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder
) -> None:
    hits = memory_store.vector_search(
        hash_embedder.embed_query("retention curve flattens leaky bucket"),
        k=3,
        persona_id="test-pm",
    )
    top = memory_store.get_chunks([hits[0].chunk_id])[0]
    assert top.doc_id.endswith("ada-north")
    assert "leaky bucket" in top.text


def test_fulltext_search_scopes_to_persona(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    assert memory_store.fulltext_search("roadmap review", k=5, persona_id="test-pm")
    assert memory_store.fulltext_search("roadmap review", k=5, persona_id="other") == []


def test_neighbors_and_document_chunks(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    doc = memory_store.list_documents("test-pm", speaker="Ada North")[0]
    chunks = memory_store.document_chunks(doc.id, 0, 100)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    if len(chunks) > 1:
        assert memory_store.neighbors(chunks[0].id, 1)[0].ordinal == 1
    assert memory_store.neighbors("missing", 1) == []


def test_topic_cooccurrence_and_filters(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    related = memory_store.related_topics("retention", "test-pm")
    assert {r.topic for r in related} >= {"product market fit", "roadmap"}
    assert len(memory_store.list_documents("test-pm", topic="onboarding")) == 2
    assert memory_store.list_documents("test-pm")[0].published is not None
    assert memory_store.list_speakers("test-pm")[0].speaker == "Lenny Rachitsky"


def test_delete_persona_removes_everything(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    memory_store.delete_persona("test-pm")
    s = memory_store.stats()
    assert s.documents == 0 and s.chunks == 0 and s.personas == 0


def test_documents_loader_pipeline(
    sample_corpus: Path,
    docs_persona: PersonaSpec,
    documents_source: SourceSpec,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
) -> None:
    report = IngestPipeline(memory_store, hash_embedder).ingest(
        sample_corpus, docs_persona, documents_source
    )
    assert report.documents == 2
    hits = memory_store.fulltext_search("enrollment period", k=2, persona_id="test-docs")
    assert hits


def test_store_rejects_dim_mismatch(memory_store: InMemoryGraphStore, persona: PersonaSpec) -> None:
    memory_store.ensure_schema(8)
    with pytest.raises(ValueError, match="dim"):
        memory_store.ensure_schema(16)


def test_pipeline_rejects_bad_embedder_shape(
    sample_corpus: Path,
    persona: PersonaSpec,
    transcript_source: SourceSpec,
    memory_store: InMemoryGraphStore,
) -> None:
    class BadEmbedder(HashEmbedder):
        def embed_documents(self, texts):  # type: ignore[override]
            return np.zeros((len(texts), self.dim + 1), dtype=np.float32)

    with pytest.raises(ValueError, match="embedder returned"):
        IngestPipeline(memory_store, BadEmbedder(dim=8)).ingest(
            sample_corpus, persona, transcript_source
        )


def test_reingest_replaces_chunks_instead_of_merging(
    sample_corpus: Path,
    persona: PersonaSpec,
    transcript_source: SourceSpec,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
) -> None:
    pipeline = IngestPipeline(memory_store, hash_embedder)
    first = pipeline.ingest(sample_corpus, persona, transcript_source)
    path = sample_corpus / "episodes" / "ada-north" / "transcript.md"
    text = path.read_text()
    path.write_text(text[: text.index("Lenny Rachitsky (00:00:41)")])  # shorter episode
    second = pipeline.ingest(sample_corpus, persona, transcript_source)
    assert second.documents == first.documents == 3
    assert memory_store.stats().chunks == second.chunks <= first.chunks
    assert not any("Charge early" in c.text for c in memory_store.chunks.values())
