from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from graphrag.config import EmbeddingSettings

Vector = np.ndarray  # shape (dim,), float32, L2-normalised
Matrix = np.ndarray  # shape (n, dim), float32, L2-normalised


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into normalised float32 vectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


def normalise(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result: np.ndarray = (arr / norms).astype(np.float32)
    return result


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    """Factory keyed on ``GRAPHRAG_EMBEDDING_BACKEND``."""
    if settings.backend == "hash":
        from graphrag.embed.hashing import HashEmbedder

        return HashEmbedder(model_name=settings.model, dim=settings.dim)
    if settings.backend == "http":
        from graphrag.embed.http_embedder import HttpEmbedder

        if not settings.base_url:
            msg = "GRAPHRAG_EMBEDDING_BACKEND=http requires GRAPHRAG_EMBEDDING_BASE_URL"
            raise ValueError(msg)
        return HttpEmbedder(
            base_url=settings.base_url,
            model_name=settings.model,
            dim=settings.dim,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            batch_size=settings.batch_size,
            timeout_seconds=settings.timeout_seconds,
        )
    from graphrag.embed.fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder(
        model_name=settings.model,
        dim=settings.dim,
        batch_size=settings.batch_size,
        cuda=settings.cuda,
    )
