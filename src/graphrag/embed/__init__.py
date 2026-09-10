"""Embedding backends behind one Protocol so compute placement is a configuration choice."""

from graphrag.embed.base import Embedder, build_embedder

__all__ = ["Embedder", "build_embedder"]
