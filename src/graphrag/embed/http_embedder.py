"""OpenAI-compatible ``POST {base_url}/embeddings`` client.

Speaks the wire format used by this project's own ``graphrag-embed-server``, Ollama, vLLM,
Hugging Face text-embeddings-inference, llama.cpp server and LM Studio, so the embedding
compute can live on any machine on the network (for example a GB10) with one env change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from graphrag.embed.base import Matrix, Vector, normalise


class HttpEmbedder:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        dim: int,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )
        self._model_name = model_name
        self._dim = dim
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def _post(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.post(
            "/embeddings", json={"model": self._model_name, "input": list(texts)}
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        rows = sorted(payload["data"], key=lambda item: int(item["index"]))
        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dim:
            msg = f"remote embedder returned shape {matrix.shape}, expected (n, {self._dim})"
            raise ValueError(msg)
        return matrix

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        batches = [
            self._post(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        return normalise(np.concatenate(batches))

    def embed_query(self, text: str) -> Vector:
        return np.asarray(self.embed_documents([text])[0], dtype=np.float32)

    def ping(self) -> float:
        """Round-trip latency in seconds for a one-token embed (used by ``graphrag doctor``)."""
        import time

        start = time.perf_counter()
        self.embed_query("ping")
        return time.perf_counter() - start
