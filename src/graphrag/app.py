"""Composition root shared by the CLI and the MCP server. Dependencies are injected so tests can
swap the store/embedder for in-memory fakes without patching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from graphrag.config import Settings, load_settings
from graphrag.embed.base import Embedder, build_embedder
from graphrag.graph.store import GraphStore
from graphrag.personas.registry import PersonaRegistry
from graphrag.retrieve.search import Retriever


@dataclass
class AppContext:
    settings: Settings
    store: GraphStore
    registry: PersonaRegistry
    # Built on first use, not at construction: creating the schema or loading a snapshot needs
    # only the configured model name and dimension, so a machine with no model cached can still
    # run `setup` and then import a bundle that supplies one.
    embedder: Embedder | None = None

    def require_embedder(self) -> Embedder:
        """The embedder, constructed on demand. Only call this when actually embedding."""
        if self.embedder is None:
            self.embedder = build_embedder(self.settings.embedding)
        return self.embedder

    @property
    def retriever(self) -> Retriever:
        return Retriever(self.store, self.require_embedder())

    @property
    def snapshots_dir(self) -> Path:
        return self.settings.snapshots_dir

    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        store: GraphStore | None = None,
        embedder: Embedder | None = None,
    ) -> AppContext:
        cfg = settings or load_settings()
        if store is None:
            from graphrag.graph.neo4j_store import Neo4jGraphStore

            store = Neo4jGraphStore(cfg.neo4j)
        return cls(
            settings=cfg,
            store=store,
            embedder=embedder,
            registry=PersonaRegistry(cfg.personas_dir),
        )

    def close(self) -> None:
        self.store.close()
