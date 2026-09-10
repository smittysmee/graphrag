"""Standalone embedding server: ``POST /v1/embeddings`` (OpenAI-compatible), ``GET /health``.

Run this on the machine with the fastest hardware (e.g. a GB10) and point every other node at it
with ``GRAPHRAG_EMBEDDING_BACKEND=http GRAPHRAG_EMBEDDING_BASE_URL=http://host:8766/v1``.
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from graphrag.config import Settings, load_settings
from graphrag.embed.base import Embedder, build_embedder


def create_app(embedder: Embedder) -> Starlette:
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model": embedder.model_name, "dim": embedder.dim})

    async def models(_: Request) -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": embedder.model_name, "object": "model"}]}
        )

    async def embeddings(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        raw = body.get("input")
        texts: list[str] = [raw] if isinstance(raw, str) else [str(t) for t in (raw or [])]
        if not texts:
            return JSONResponse({"error": "input must be a string or list of strings"}, 400)
        matrix = embedder.embed_documents(texts)
        data = [
            {"object": "embedding", "index": i, "embedding": row.tolist()}
            for i, row in enumerate(matrix)
        ]
        tokens = sum(len(t.split()) for t in texts)
        return JSONResponse(
            {
                "object": "list",
                "model": embedder.model_name,
                "data": data,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/embeddings", embeddings, methods=["POST"]),
        ]
    )


def main(settings: Settings | None = None) -> None:
    import uvicorn

    cfg = settings or load_settings()
    embedder = build_embedder(cfg.embedding)
    uvicorn.run(create_app(embedder), host=cfg.embed_server_host, port=cfg.embed_server_port)


if __name__ == "__main__":
    main()
