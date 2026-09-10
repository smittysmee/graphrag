from graphrag.models import PersonaSpec, ScoredChunk
from graphrag.pipeline import IngestReport
from graphrag.retrieve.search import Retriever, reciprocal_rank_fusion


def test_rrf_prefers_items_in_both_lists() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [ScoredChunk(chunk_id="a", score=0.9), ScoredChunk(chunk_id="b", score=0.8)],
            "fulltext": [
                ScoredChunk(chunk_id="b", score=3.0),
                ScoredChunk(chunk_id="c", score=1.0),
            ],
        }
    )
    assert fused[0][0] == "b"
    assert fused[0][2] == ["vector", "fulltext"]


def test_hybrid_search_returns_cited_hits(ingested: IngestReport, retriever: Retriever) -> None:
    hits = retriever.search("retention curve", persona_id="test-pm", k=3, expand=1)
    assert hits
    assert hits[0].document.title
    assert "Ada North" in hits[0].citation() or "Ben Oduya" in hits[0].citation()
    assert "youtube.com" in hits[0].citation()
    assert retriever.search("   ", persona_id="test-pm") == []


def test_modes(ingested: IngestReport, retriever: Retriever) -> None:
    v = retriever.search("pricing signal of value", persona_id="test-pm", k=2, mode="vector")
    f = retriever.search("pricing signal of value", persona_id="test-pm", k=2, mode="fulltext")
    assert v and f
    assert v[0].methods == ["vector"] and f[0].methods == ["fulltext"]


def test_context_pack_prompt(
    ingested: IngestReport, retriever: Retriever, persona: PersonaSpec
) -> None:
    pack = retriever.context("how should I run a roadmap review", persona=persona)
    prompt = pack.to_prompt()
    assert prompt.startswith("# Persona: Test Product Leader")
    assert persona.role_prompt in prompt
    assert "## [1]" in prompt
    assert len(pack.hits) <= persona.retrieval.top_k
    assert "roadmap" in pack.topics
