"""Integration fixtures: a real Neo4j from the compose `test` profile. Skips when unreachable."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from graphrag.config import Neo4jSettings
from graphrag.graph.neo4j_store import Neo4jGraphStore

TEST_PERSONA_PREFIX = "it-"


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings()


@pytest.fixture(scope="session")
def neo4j_store(neo4j_settings: Neo4jSettings) -> Iterator[Neo4jGraphStore]:
    store = Neo4jGraphStore(neo4j_settings)
    try:
        store.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable at {neo4j_settings.uri}: {exc}")
    yield store
    store.close()


@pytest.fixture
def clean_store(neo4j_store: Neo4jGraphStore) -> Iterator[Neo4jGraphStore]:
    """Removes every `it-*` persona after the test (post-yield cleanup keyed on a prefix)."""
    yield neo4j_store
    for persona in neo4j_store.list_personas():
        if persona.id.startswith(TEST_PERSONA_PREFIX):
            neo4j_store.delete_persona(persona.id)
