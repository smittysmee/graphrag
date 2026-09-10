"""Deterministic bag-of-words hashing embedder. Tests only: no model download, stable vectors,
and lexical overlap still yields higher cosine similarity, so retrieval tests are meaningful."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from graphrag.embed.base import Matrix, Vector, normalise

TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, model_name: str = "hash-test", dim: int = 64) -> None:
        self._model_name = model_name
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        return np.asarray(self.embed_documents([text])[0], dtype=np.float32)
