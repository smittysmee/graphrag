import asyncio
from typing import Any

from fastmcp import Client

from graphrag.app import AppContext
from graphrag.mcp_server import ServerState, create_server
from graphrag.pipeline import IngestReport


def _call(server: Any, tool: str, **args: Any) -> Any:
    async def run() -> Any:
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            return result.data if result.data is not None else result.content

    return asyncio.run(run())


def _tools(server: Any) -> set[str]:
    async def run() -> set[str]:
        async with Client(server) as client:
            return {t.name for t in await client.list_tools()}

    return asyncio.run(run())


def test_tools_are_registered(app_context: AppContext) -> None:
    server = create_server(ServerState(context=app_context))
    assert {
        "list_personas",
        "persona_brief",
        "context",
        "search",
        "documents",
        "read_document",
        "topics",
        "related_topics",
        "speakers",
        "cypher",
        "stats",
        "recommend_personas",
        "get_document",
    } <= _tools(server)


def test_tools_answer_from_graph(app_context: AppContext, ingested: IngestReport) -> None:
    server = create_server(ServerState(context=app_context))
    personas = _call(server, "list_personas")
    assert {p["id"] for p in personas} == {"test-docs", "test-pm"}

    brief = _call(server, "persona_brief", persona_id="test-pm")
    assert "Indexed: 3 documents" in str(brief)

    pack = _call(server, "context", query="retention curve", persona_id="test-pm")
    assert pack["hits"] and "## [1]" in pack["prompt"]
    assert pack["hits"][0]["url"].startswith("https://www.youtube.com")

    hits = _call(server, "search", query="roadmap review", persona_id="test-pm", k=2)
    assert hits and hits[0]["title"]

    docs = _call(server, "documents", persona_id="test-pm", topic="onboarding")
    assert len(docs) == 2
    first = _call(server, "read_document", doc_id=docs[0]["id"], start=0, count=1)
    assert first[0]["ordinal"] == 0
    assert _call(server, "get_document", doc_id="missing")["error"]

    assert _call(server, "topics", persona_id="test-pm")[0]["count"] >= 1
    assert _call(server, "related_topics", topic="retention")
    assert _call(server, "speakers", persona_id="test-pm")[0]["speaker"] == "Lenny Rachitsky"
    rec = _call(server, "recommend_personas", sdlc_stage="compliance")
    assert rec["personas"][0]["id"] == "test-docs"
    stats = _call(server, "stats")
    assert stats["graph"]["documents"] == 3 and stats["embedding"]["backend"] == "hash"
