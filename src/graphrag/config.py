"""Typed configuration. Everything comes from the environment (or .env); nothing is hard-coded.

Prefixes:
    NEO4J_*              connection to Neo4j Community
    GRAPHRAG_EMBEDDING_* which embedder runs, and where (in-process, or a machine on the network)
    GRAPHRAG_*           paths, MCP server, embed server, enrichment model
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackend = Literal["fastembed", "http", "hash"]

_ENV_FILE = ".env"


class Neo4jSettings(BaseSettings):
    """Bolt connection settings for Neo4j."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=_ENV_FILE, extra="ignore")

    uri: str = "bolt://neo4j:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("graphrag-local-password")
    database: str = "neo4j"


class EmbeddingSettings(BaseSettings):
    """Which embedding model to use and where the compute runs.

    ``backend="fastembed"`` runs an ONNX model inside this process (default, offline).
    ``backend="http"`` calls an OpenAI-compatible ``/v1/embeddings`` endpoint on another machine
    (for example a GB10 running ``graphrag-embed-server``, Ollama, vLLM, or TEI).
    ``backend="hash"`` is a deterministic fake for tests.
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAPHRAG_EMBEDDING_", env_file=_ENV_FILE, extra="ignore"
    )

    backend: EmbeddingBackend = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    batch_size: int = Field(default=64, ge=1, le=2048)
    cuda: bool = False
    base_url: str | None = None
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)

    @field_validator("base_url", mode="before")
    @classmethod
    def _empty_url_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(env_prefix="GRAPHRAG_", env_file=_ENV_FILE, extra="ignore")

    personas_dir: Path = Path("personas")
    snapshots_dir: Path = Path("data/snapshots")
    enrichment_dir: Path = Path("data/enrichment")
    model_cache_dir: Path = Path("/models")
    skills_dir: Path = Path(".claude/skills")

    mcp_host: str = "0.0.0.0"  # noqa: S104 - bound inside a container on purpose
    mcp_port: int = 8765
    embed_server_host: str = "0.0.0.0"  # noqa: S104
    embed_server_port: int = 8766

    enrich_model: str = "claude-opus-5"
    enrich_max_chunk_chars: int = 6000

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


def load_settings() -> Settings:
    """Build settings from the environment. Kept as a function so tests can construct their own."""
    return Settings()
