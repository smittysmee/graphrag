"""Aggregating chunk vectors into one vector per document or per entity.

Shared by both stores so ``mean_embeddings`` means the same thing everywhere: the arithmetic
mean of a key's chunk vectors, re-normalised to unit length. Because the embedder emits
L2-normalised vectors and search scores them by dot product, re-normalising the mean makes
Euclidean distance between two keys a monotone function of their cosine similarity, which is
what K-means and Gaussian mixtures assume when they are handed text embeddings.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from graphrag.embed.base import Matrix, normalise


def stack_means(sums: dict[str, np.ndarray], counts: Counter[str]) -> tuple[list[str], Matrix]:
    """Turn per-key vector sums into a key order and a matrix of unit-length means."""
    keys = sorted(sums)
    if not keys:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack([sums[k] / max(counts[k], 1) for k in keys])
    return keys, normalise(matrix)
