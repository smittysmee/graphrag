from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import combinations

from graphrag.models import Document


def topic_cooccurrence(documents: Iterable[Document]) -> list[tuple[str, str, int]]:
    """Count how often two topics are tagged on the same document. Undirected, sorted pairs."""
    counter: Counter[tuple[str, str]] = Counter()
    for doc in documents:
        for a, b in combinations(sorted(set(doc.topics)), 2):
            counter[(a, b)] += 1
    return [(a, b, n) for (a, b), n in sorted(counter.items())]
