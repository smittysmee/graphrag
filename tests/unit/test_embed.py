import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from graphrag.config import EmbeddingSettings
from graphrag.embed.base import Embedder, build_embedder
from graphrag.embed.hashing import HashEmbedder
from graphrag.embed.http_embedder import HttpEmbedder
from graphrag.embed.server import create_app


def test_hash_embedder_is_deterministic_and_normalised(hash_embedder: HashEmbedder) -> None:
    a = hash_embedder.embed_documents(["retention curve flattens", "pricing signal"])
    b = hash_embedder.embed_documents(["retention curve flattens", "pricing signal"])
    assert a.shape == (2, hash_embedder.dim)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert isinstance(hash_embedder, Embedder)


def test_hash_embedder_lexical_overlap_scores_higher(hash_embedder: HashEmbedder) -> None:
    q = hash_embedder.embed_query("retention curve")
    docs = hash_embedder.embed_documents(
        ["the retention curve flattens", "design reviews are short"]
    )
    assert docs[0] @ q > docs[1] @ q


def test_build_embedder_http_requires_base_url() -> None:
    with pytest.raises(ValueError, match="BASE_URL"):
        build_embedder(EmbeddingSettings(backend="http", base_url=None))


def test_embed_server_speaks_openai_format(hash_embedder: HashEmbedder) -> None:
    client = TestClient(create_app(hash_embedder))
    assert client.get("/health").json()["dim"] == hash_embedder.dim
    resp = client.post("/v1/embeddings", json={"model": "x", "input": ["a b", "c"]})
    body = resp.json()
    assert resp.status_code == 200
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert len(body["data"][0]["embedding"]) == hash_embedder.dim
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 400


def test_http_embedder_round_trips_through_server(hash_embedder: HashEmbedder) -> None:
    """HttpEmbedder -> MockTransport -> embed server app: the same vectors as in-process."""
    server = TestClient(create_app(hash_embedder))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        payload = json.loads(request.content)
        assert payload["model"] == "hash-test"
        r = server.post("/v1/embeddings", json=payload)
        return httpx.Response(r.status_code, json=r.json())

    remote = HttpEmbedder(
        base_url="http://gb10:8766/v1",
        model_name="hash-test",
        dim=hash_embedder.dim,
        api_key="k",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    )
    texts = ["one two", "three", "four five six"]
    assert np.allclose(remote.embed_documents(texts), hash_embedder.embed_documents(texts))
    assert remote.embed_query("one two").shape == (hash_embedder.dim,)
    assert remote.ping() >= 0


def test_http_embedder_rejects_wrong_dim(hash_embedder: HashEmbedder) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    remote = HttpEmbedder("http://x/v1", "m", dim=64, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="expected"):
        remote.embed_documents(["a"])


def test_download_is_off_by_default() -> None:
    """Policy default: the fastembed backend must not fetch weights unless asked."""
    assert EmbeddingSettings().allow_download is False


def test_missing_model_fails_with_instructions_instead_of_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache miss must raise locally, and must have marked the HF stack offline first."""
    from graphrag.embed.fastembed_embedder import EmbeddingModelUnavailableError, FastEmbedEmbedder

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(EmbeddingModelUnavailableError) as exc:
        FastEmbedEmbedder(
            model_name="definitely/not-a-real-model",
            dim=384,
            cache_dir=str(tmp_path / "empty"),
        )
    assert "downloading is disabled" in str(exc.value)
    assert "make model-import" in str(exc.value)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
