from pathlib import Path

import numpy as np
import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention, PersonaSpec, SourceSpec
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


def test_document_ids_are_scoped_by_persona_and_source(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """What `graphrag sync` compares the raw files against."""
    ids = memory_store.document_ids("test-pm")
    assert len(ids) == 3
    assert all(i.startswith("test-pm:test-podcast:") for i in ids)
    assert memory_store.document_ids("test-pm", "test-podcast") == ids
    assert memory_store.document_ids("test-pm", "nope") == set()
    assert memory_store.document_ids("other-persona") == set()


def test_enriched_document_ids_answer_only_after_an_extraction_lands(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """What `graphrag sync` compares those ids against to find documents with no entities.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    doc_id = sorted(memory_store.document_ids("test-pm", "test-podcast"))[0]
    assert memory_store.enriched_document_ids("test-pm", "test-podcast") == set()

    chunk_id = memory_store.document_chunks(doc_id, 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="metric:retention", name="Retention", type="metric")],
            mentions=[Mention(chunk_id=chunk_id, entity_id="metric:retention")],
        )
    )

    assert memory_store.enriched_document_ids("test-pm", "test-podcast") == {doc_id}
    assert memory_store.enriched_document_ids("test-pm", "nope") == set()
    assert memory_store.enriched_document_ids("other-persona", "test-podcast") == set()


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


def test_attach_speaker_and_attributed_document_ids(
    thread_document: str, memory_store: InMemoryGraphStore
) -> None:
    """What `graphrag attribution-import` writes, and what `graphrag sync` reads back.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    assert memory_store.attributed_document_ids("test-docs", "threads") == set()
    chunks = memory_store.document_chunks(thread_document, 0, 100)

    memory_store.attach_speaker(thread_document, chunks[0].id, "quill-maker")
    memory_store.attach_speaker(thread_document, chunks[-1].id, "ledger-ann")
    memory_store.attach_speaker(thread_document, chunks[0].id, "quill-maker")  # idempotent

    assert memory_store.attributed_document_ids("test-docs", "threads") == {thread_document}
    assert memory_store.attributed_document_ids("test-docs", "docs") == set()
    assert memory_store.attributed_document_ids("other-persona", "threads") == set()
    assert memory_store.documents[thread_document].speakers == ["quill-maker", "ledger-ann"]
    assert memory_store.document_chunks(thread_document, 0, 1)[0].speakers == ["quill-maker"]
    assert memory_store.list_documents("test-docs", speaker="ledger-ann") != []
    assert memory_store.stats().speakers == 2


def test_network_reads_return_the_edges_the_sna_package_projects_from(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """The store contract behind `graphrag sna`.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    pairs = memory_store.speaker_document_pairs("test-pm")
    assert [(p.speaker, p.doc_id) for p in pairs] == sorted(
        (p.speaker, p.doc_id) for p in pairs
    )  # stable order, so a projection is reproducible
    host = [p for p in pairs if p.speaker == "Lenny Rachitsky"]
    assert len(host) == 3 and all(p.chunks >= 1 for p in host)
    assert len({p.doc_id for p in pairs}) == 3
    assert memory_store.speaker_document_pairs("test-pm", "test-podcast") == pairs
    assert memory_store.speaker_document_pairs("test-pm", "absent") == []
    assert memory_store.speaker_document_pairs("other-persona") == []

    doc_id = sorted(memory_store.document_ids("test-pm"))[0]
    chunk_id = memory_store.document_chunks(doc_id, 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="metric:retention", name="Retention", type="metric"),
                Entity(id="concept:onboarding", name="Onboarding", type="concept"),
            ],
            mentions=[
                Mention(chunk_id=chunk_id, entity_id="metric:retention"),
                Mention(chunk_id=chunk_id, entity_id="concept:onboarding"),
            ],
        )
    )
    mentions = memory_store.entity_chunk_pairs("test-pm")
    assert {m.entity_id for m in mentions} == {"metric:retention", "concept:onboarding"}
    assert all(m.chunk_id == chunk_id and m.doc_id == doc_id for m in mentions)
    assert [m.name for m in mentions if m.entity_id == "metric:retention"] == ["Retention"]
    typed = memory_store.entity_chunk_pairs("test-pm", types=["metric"])
    assert [m.entity_id for m in typed] == ["metric:retention"]
    assert memory_store.entity_chunk_pairs("test-pm", "absent") == []
    assert memory_store.entity_chunk_pairs("other-persona") == []

    edges = memory_store.topic_edges("test-pm", min_weight=1)
    assert edges and all(e.source < e.target and e.weight >= 1 for e in edges)
    assert memory_store.topic_edges("test-pm", min_weight=99) == []
    assert memory_store.topic_edges("other-persona") == []


def test_mean_embeddings_are_one_unit_vector_per_document_or_entity(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """What K-means and Gaussian mixtures cluster when asked for content rather than structure."""
    doc_ids, matrix = memory_store.mean_embeddings("test-pm")
    assert doc_ids == sorted(memory_store.document_ids("test-pm"))
    assert matrix.shape == (3, 64)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    chunk_id = memory_store.document_chunks(doc_ids[0], 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="metric:retention", name="Retention", type="metric")],
            mentions=[Mention(chunk_id=chunk_id, entity_id="metric:retention")],
        )
    )
    keys, entity_matrix = memory_store.mean_embeddings("test-pm", level="entity")
    assert keys == ["metric:retention"]
    assert np.allclose(entity_matrix[0], memory_store.embeddings[chunk_id], atol=1e-5)

    assert memory_store.mean_embeddings("other-persona")[0] == []
    with pytest.raises(ValueError, match="level must be"):
        memory_store.mean_embeddings("test-pm", level="chunk")


def test_attach_speaker_dates_the_edge_so_a_network_can_be_cut_to_a_window(
    thread_document: str, memory_store: InMemoryGraphStore
) -> None:
    """What `graphrag attribution-import` passes through, and what `graphrag sna` filters on.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.attach_speaker(
        thread_document, chunks[0].id, "quill-maker", posted_at="2025-02-03", role="op", score=12
    )
    memory_store.attach_speaker(
        thread_document, chunks[-1].id, "ledger-ann", posted_at="2025-02-06", role="reply"
    )

    post = memory_store.document_chunks(thread_document, 0, 1)[0].speaker_posts[0]
    assert (post.speaker, post.posted_at, post.role, post.score) == (
        "quill-maker",
        "2025-02-03",
        "op",
        12,
    )
    everyone = memory_store.speaker_document_pairs("test-docs")
    assert [p.speaker for p in everyone] == ["ledger-ann", "quill-maker"]

    early = memory_store.speaker_document_pairs("test-docs", until="2025-02-04")
    assert [(p.speaker, p.chunks) for p in early] == [("quill-maker", 1)]
    late = memory_store.speaker_document_pairs("test-docs", since="2025-02-04")
    assert [p.speaker for p in late] == ["ledger-ann"]
    assert (
        memory_store.speaker_document_pairs("test-docs", since="2025-02-01", until="2025-02-28")
        == everyone
    )
    assert memory_store.speaker_document_pairs("test-docs", since="2030-01-01") == []


def test_an_undated_speaker_drops_out_of_a_window_rather_than_being_assumed_in_range(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """Transcripts carry no post dates, so a window over them has to come back empty."""
    assert memory_store.speaker_document_pairs("test-pm") != []
    assert memory_store.speaker_document_pairs("test-pm", since="2000-01-01") == []


def test_annotations_write_a_stance_on_a_mention_and_facets_on_a_passage(
    thread_document: str, memory_store: InMemoryGraphStore
) -> None:
    """What `graphrag annotations-import` writes, and what `graphrag sync` reads back.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="concept:handbook", name="Handbook", type="concept")],
            mentions=[Mention(chunk_id=chunks[0].id, entity_id="concept:handbook")],
        )
    )
    assert memory_store.annotated_document_ids("test-docs", "threads") == set()

    memory_store.annotate_mention(thread_document, chunks[0].id, "Handbook", "complaint")
    memory_store.annotate_chunk(thread_document, chunks[0].id, ["handover", "access"])
    memory_store.annotate_chunk(thread_document, chunks[0].id, ["handover"])  # idempotent

    assert memory_store.annotated_document_ids("test-docs", "threads") == {thread_document}
    assert memory_store.annotated_document_ids("test-docs", "docs") == set()
    assert memory_store.annotated_document_ids("other-persona", "threads") == set()
    stance = memory_store.mention_stances("test-docs")[0]
    assert (stance.name, stance.stance, stance.doc_id) == ("Handbook", "complaint", thread_document)
    assert memory_store.chunk_facets("test-docs")[0].facets == ["handover", "access"]
    assert memory_store.chunk_facets("test-docs", "docs") == []
    assert memory_store.mention_stances("other-persona") == []

    # an entity the passage does not mention, and a passage of another document, write nothing
    memory_store.annotate_mention(thread_document, chunks[0].id, "Nobody", "praise")
    memory_store.annotate_chunk("other-doc", chunks[0].id, ["ghost"])
    assert len(memory_store.mention_stances("test-docs")) == 1
    assert memory_store.chunk_facets("test-docs")[0].facets == ["handover", "access"]


def test_persona_entities_answer_with_every_spelling_a_node_holds(
    thread_document: str, memory_store: InMemoryGraphStore
) -> None:
    """What the annotation fallback resolves a name against.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    chunk_id = memory_store.document_chunks(thread_document, 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(
                    id="concept:handover",
                    name="Handover",
                    type="concept",
                    aliases=["handbook"],
                ),
                Entity(id="concept:unmentioned", name="Unmentioned", type="concept"),
            ],
            mentions=[Mention(chunk_id=chunk_id, entity_id="concept:handover")],
        )
    )

    entities = memory_store.persona_entities("test-docs")

    assert [e.id for e in entities] == ["concept:handover"]  # an entity nobody mentions is not the
    assert entities[0].aliases == ["handbook"]  # persona's, whatever else holds it
    assert memory_store.persona_entities("other-persona") == []


def test_entity_mention_rows_carry_the_stance_and_the_passage_speakers(
    layered: InMemoryGraphStore,
) -> None:
    """The store contract behind `--stance` and the speakers-entities network.

    The two facts have to arrive together. A stance sits on the mention and a speaker sits on
    the passage, so joining two separate reads afterwards would pair a speaker with a stance
    that belongs to a different passage of the same document.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    rows = layered.entity_mention_rows("test-layers")
    assert [(r.entity_id, r.chunk_id) for r in rows] == sorted(
        (r.entity_id, r.chunk_id) for r in rows
    )
    keyed = {(r.entity_id, r.doc_id.rsplit(":", 1)[1]): r for r in rows}
    praised = keyed[("product:alpha", "post-1")]
    assert (praised.name, praised.type, praised.stance) == ("Alpha", "product", "praise")
    assert praised.speakers == ["ana"]
    assert keyed[("product:alpha", "post-2")].stance == "complaint"
    assert keyed[("product:alpha", "post-2")].speakers == ["bo"]
    assert keyed[("product:gamma", "post-3")].stance == "neutral"

    # A mention the annotation pass never reached carries no stance, which is not "neutral".
    layered.upsert_enrichment(
        Enrichment(
            mentions=[
                Mention(chunk_id="test-layers:posts:post-2#0", entity_id="product:beta"),
            ]
        )
    )
    fresh = layered.entity_mention_rows("test-layers")
    unannotated = [r for r in fresh if r.entity_id == "product:beta" and "post-2" in r.doc_id]
    assert [r.stance for r in unannotated] == [None]

    assert layered.entity_mention_rows("test-layers", types=["person"]) == []
    assert [r.entity_id for r in layered.entity_mention_rows("test-layers", types=["product"])] == [
        r.entity_id for r in fresh
    ]
    assert layered.entity_mention_rows("test-layers", "absent") == []
    assert layered.entity_mention_rows("other-persona") == []
