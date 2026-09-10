from graphrag.config import EmbeddingSettings, Settings


def test_empty_optional_strings_become_none() -> None:
    emb = EmbeddingSettings(backend="http", base_url="", api_key="")
    assert emb.base_url is None
    assert emb.api_key is None


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_BACKEND", "http")
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_BASE_URL", "http://gb10:8766/v1")
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_DIM", "1024")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("GRAPHRAG_MCP_PORT", "9999")
    cfg = Settings()
    assert cfg.embedding.backend == "http"
    assert cfg.embedding.base_url == "http://gb10:8766/v1"
    assert cfg.embedding.dim == 1024
    assert cfg.neo4j.password.get_secret_value() == "secret"
    assert cfg.mcp_port == 9999
