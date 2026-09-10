"""In-process ONNX embeddings via fastembed.

**This backend never downloads a model unless you explicitly allow it.** fastembed would
otherwise fetch from Hugging Face the first time it is constructed, which many environments
forbid and which would happen silently at query time. The default is therefore
``allow_download=False``: if the model is not already in the cache, construction fails with
instructions instead of touching the network.

Supply the model without downloading by importing a persona bundle exported ``--with-model``,
or with ``make model-import FILE=model.tar.gz``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from graphrag.embed.base import Matrix, Vector, normalise


class EmbeddingModelUnavailableError(RuntimeError):
    """The local model is not cached and downloading it is not permitted."""


def _forbid_downloads() -> None:
    """Make the Hugging Face stack fail locally rather than fetch.

    Written rather than read, so this is not a config source: it is the only lever these
    libraries expose for "do not use the network".
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        batch_size: int = 64,
        cuda: bool = False,
        cache_dir: str | None = None,
        allow_download: bool = False,
    ) -> None:
        from fastembed import TextEmbedding

        if not allow_download:
            _forbid_downloads()

        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if cuda:
            kwargs["cuda"] = True

        try:
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:
            if allow_download:
                raise
            msg = (
                f"the embedding model {model_name!r} is not in the local cache"
                f"{f' ({cache_dir})' if cache_dir else ''} and downloading is disabled.\n"
                "Supply it without touching the network by either:\n"
                "  - importing a persona bundle exported with --with-model, or\n"
                "  - restoring a cache: make model-import FILE=model.tar.gz\n"
                "Or, if your environment permits fetching it, set "
                "GRAPHRAG_EMBEDDING_ALLOW_DOWNLOAD=true (then `make model`)."
            )
            raise EmbeddingModelUnavailableError(msg) from exc

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
