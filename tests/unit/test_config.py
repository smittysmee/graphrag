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


def test_rendered_env_uses_only_keys_the_models_actually_read() -> None:
    """Guards against drift: a renamed field must not leave a stale key behind."""
    from graphrag.config import EmbeddingSettings, Neo4jSettings, Settings, env_name, render_dotenv

    text = render_dotenv(Settings())
    written = {line.split("=", 1)[0] for line in text.splitlines() if line and "=" in line}
    readable = set()
    for cls in (Neo4jSettings, EmbeddingSettings, Settings):
        readable |= {env_name(cls, field) for field in cls.model_fields}
    assert written <= readable, f"keys nothing reads: {written - readable}"


def test_rendered_env_round_trips_through_settings(tmp_path, monkeypatch) -> None:
    """Write a .env from a Settings object, read it back, get the same values."""
    from pydantic import SecretStr

    from graphrag.config import EmbeddingSettings, Neo4jSettings, Settings, render_dotenv

    for key in (
        "NEO4J_PASSWORD",
        "NEO4J_URI",
        "GRAPHRAG_EMBEDDING_BACKEND",
        "GRAPHRAG_EMBEDDING_BASE_URL",
        "GRAPHRAG_EMBEDDING_ALLOW_DOWNLOAD",
        "GRAPHRAG_MCP_PORT",
    ):
        monkeypatch.delenv(key, raising=False)

    original = Settings().model_copy(
        update={
            "neo4j": Neo4jSettings(uri="bolt://box:7687", password=SecretStr("s3cretpass")),
            "embedding": EmbeddingSettings(
                backend="http", base_url="http://gb10.local:8766/v1", allow_download=True
            ),
        }
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(render_dotenv(original), encoding="utf-8")

    loaded = Settings()
    assert loaded.neo4j.uri == "bolt://box:7687"
    assert loaded.neo4j.password.get_secret_value() == "s3cretpass"
    assert loaded.embedding.backend == "http"
    assert loaded.embedding.base_url == "http://gb10.local:8766/v1"
    assert loaded.embedding.allow_download is True
