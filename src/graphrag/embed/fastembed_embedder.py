"""In-process ONNX embeddings via fastembed (default backend; offline after the first download)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from graphrag.embed.base import Matrix, Vector, normalise


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        batch_size: int = 64,
        cuda: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if cuda:
            kwargs["cuda"] = True
        self._model = TextEmbedding(**kwargs)
        self._model_name = model_name
        self._dim = dim
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts), batch_size=self._batch_size))
        matrix = normalise(np.stack(vectors))
        self._check_dim(matrix.shape[1])
        return matrix

    def embed_query(self, text: str) -> Vector:
        vectors = list(self._model.query_embed(text))
        matrix = normalise(np.stack(vectors))
        self._check_dim(matrix.shape[1])
        return np.asarray(matrix[0], dtype=np.float32)

    def _check_dim(self, actual: int) -> None:
        if actual != self._dim:
            msg = (
                f"embedding model {self._model_name!r} produced {actual} dims but "
                f"GRAPHRAG_EMBEDDING_DIM={self._dim}"
            )
            raise ValueError(msg)
