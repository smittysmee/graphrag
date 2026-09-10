"""FastMCP server exposing the graph to agents.

Tools are thin: they validate arguments, call the same ``Retriever``/``GraphStore`` the CLI uses,
and return JSON-serialisable dicts. ``AppContext`` is created lazily so importing this module
(e.g. in tests) never opens a Neo4j connection.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from graphrag.app import AppContext
from graphrag.config import Settings
from graphrag.graph import snapshot as snap
from graphrag.models import SearchMode
from graphrag.personas.brief import build_brief
from graphrag.personas.registry import SDLC_STAGES

INSTRUCTIONS = """graphrag: persona-grounded knowledge graph.
Start with `list_personas`, then `persona_brief(persona_id)` to adopt a persona, then
`context(query, persona_id)` for cited passages before answering. `search` is the raw hybrid
search; `documents`/`topics`/`speakers` browse the corpus; `read_document` pages through a
source; `cypher` runs read-only Cypher for anything else."""


class ServerState:
    def __init__(self, settings: Settings | None = None, context: AppContext | None = None) -> None:
        self._settings = settings
        self._context = context

    @property
    def ctx(self) -> AppContext:
        if self._context is None:
            self._context = AppContext.build(self._settings)
        return self._context


def create_server(state: ServerState | None = None) -> FastMCP:
    st = state or ServerState()
    mcp = FastMCP("graphrag", instructions=INSTRUCTIONS)

    # ------------------------------------------------------------- personas
    @mcp.tool
    def list_personas() -> list[dict[str, Any]]:
        """Personas available in the graph, with their SDLC stages and source descriptions."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "sdlc_stages": p.sdlc_stages,
                "sources": [s.description or s.url or s.id for s in p.sources],
            }
            for p in st.ctx.registry.all()
        ]

    @mcp.tool
    def persona_brief(persona_id: str) -> str:
        """Role prompt + grounding summary to adopt before answering as a persona."""
        return build_brief(st.ctx.registry.get(persona_id), st.ctx.store)

    @mcp.tool
    def recommend_personas(sdlc_stage: str) -> dict[str, Any]:
        """Which personas apply to an SDLC stage (discovery, requirements, design, testing, ...)."""
        matches = st.ctx.registry.recommend(sdlc_stage)
        return {
            "stage": sdlc_stage,
            "known_stages": SDLC_STAGES,
            "personas": [
                {"id": p.id, "name": p.name, "description": p.description} for p in matches
            ],
        }

    # ------------------------------------------------------------- retrieval
    @mcp.tool
    def context(
        query: str,
        persona_id: str | None = None,
        k: int | None = None,
        expand: int | None = None,
    ) -> dict[str, Any]:
        """Cited context pack for a question. Returns prompt-ready text plus structured hits."""
        persona = st.ctx.registry.get(persona_id) if persona_id else None
        pack = st.ctx.retriever.context(query, persona=persona, k=k, expand=expand)
        return {
            "prompt": pack.to_prompt(),
            "hits": [
                {
                    "n": i,
                    "citation": h.citation(),
                    "doc_id": h.document.id,
                    "chunk_id": h.chunk.id,
                    "speaker": h.chunk.speaker,
                    "url": h.chunk.url or h.document.url,
                    "score": round(h.score, 5),
                    "text": h.passage(),
                }
                for i, h in enumerate(pack.hits, start=1)
            ],
            "topics": pack.topics,
            "entities": [e.model_dump() for e in pack.entities],
        }

    @mcp.tool
    def search(
        query: str, persona_id: str | None = None, k: int = 8, mode: SearchMode = "hybrid"
    ) -> list[dict[str, Any]]:
        """Raw hybrid (vector + full-text) search over passages."""
        hits = st.ctx.retriever.search(query, persona_id=persona_id, k=k, mode=mode)
        return [
            {
                "chunk_id": h.chunk.id,
                "doc_id": h.document.id,
                "title": h.document.title,
                "speaker": h.chunk.speaker,
                "start_ts": h.chunk.start_ts,
                "url": h.chunk.url or h.document.url,
                "score": round(h.score, 5),
                "methods": h.methods,
                "text": h.chunk.text,
            }
            for h in hits
        ]

    # ------------------------------------------------------------- browsing
    @mcp.tool
    def documents(
        persona_id: str | None = None,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """List documents, optionally filtered by topic or speaker."""
        return [
            {
                "id": d.id,
                "title": d.title,
                "published": d.published.isoformat() if d.published else None,
                "url": d.url,
                "speakers": d.speakers,
                "topics": d.topics,
                "description": d.description[:300],
            }
            for d in st.ctx.store.list_documents(
                persona_id, topic=topic, speaker=speaker, limit=limit
            )
        ]

    @mcp.tool
    def get_document(doc_id: str) -> dict[str, Any]:
        """Full metadata for one document."""
        doc = st.ctx.store.get_document(doc_id)
        if doc is None:
            return {"error": f"unknown document {doc_id}"}
        return doc.model_dump(mode="json")

    @mcp.tool
    def read_document(doc_id: str, start: int = 0, count: int = 5) -> list[dict[str, Any]]:
        """Page through a document's passages in order (start = passage ordinal)."""
        return [
            {
                "ordinal": c.ordinal,
                "speaker": c.speaker,
                "start_ts": c.start_ts,
                "url": c.url,
                "text": c.text,
            }
            for c in st.ctx.store.document_chunks(doc_id, start, count)
        ]

    @mcp.tool
    def topics(persona_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Topics ranked by how many documents carry them."""
        return [t.model_dump() for t in st.ctx.store.list_topics(persona_id, limit)]

    @mcp.tool
    def related_topics(
        topic: str, persona_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Topics that co-occur with a topic (graph adjacency), strongest first."""
        return [t.model_dump() for t in st.ctx.store.related_topics(topic, persona_id, limit)]

    @mcp.tool
    def speakers(persona_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Speakers/authors ranked by number of documents."""
        return [s.model_dump() for s in st.ctx.store.list_speakers(persona_id, limit)]

    # ------------------------------------------------------------- escape hatches
    @mcp.tool
    def cypher(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run read-only Cypher (MATCH/RETURN). Labels: Persona, Source, Document, Chunk, Speaker,
        Topic, Entity. Relationships: GROUNDED_BY, CONTAINS, HAS_CHUNK, NEXT, SPOKE, FEATURES,
        ABOUT, CO_OCCURS, MENTIONS, RELATED_TO. Max 500 rows."""
        return st.ctx.store.run_readonly_cypher(query, params)

    @mcp.tool
    def stats() -> dict[str, Any]:
        """Node counts overall and per persona, plus committed snapshots."""
        return {
            "graph": st.ctx.store.stats().model_dump(),
            "snapshots": [
                snap.manifest_summary(m) for m in snap.list_snapshots(st.ctx.snapshots_dir)
            ],
            "embedding": {
                "backend": st.ctx.settings.embedding.backend,
                "model": st.ctx.settings.embedding.model,
                "dim": st.ctx.settings.embedding.dim,
            },
        }

    # ------------------------------------------------------------- resources & prompts
    @mcp.resource("graphrag://personas")
    def personas_resource() -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in st.ctx.registry.all()]

    @mcp.resource("graphrag://persona/{persona_id}")
    def persona_resource(persona_id: str) -> str:
        return build_brief(st.ctx.registry.get(persona_id), st.ctx.store)

    @mcp.prompt
    def assume_persona(persona_id: str, task: str) -> str:
        """System-style prompt: adopt a persona and work on a task with grounded citations."""
        brief = build_brief(st.ctx.registry.get(persona_id), st.ctx.store)
        return (
            f"{brief}\n\n# Task\n{task}\n\n"
            "Before answering, call the `context` tool with this task and the persona id, then "
            "answer in the persona's voice citing [n]."
        )

    return mcp


def run_server(transport: str = "http", settings: Settings | None = None) -> None:
    cfg = settings or Settings()
    server = create_server(ServerState(cfg))
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "http":
        server.run(transport="http", host=cfg.mcp_host, port=cfg.mcp_port)
    else:
        msg = f"unknown transport {transport!r}; use http or stdio"
        raise ValueError(msg)


def main() -> None:
    run_server()


if __name__ == "__main__":
    main()
