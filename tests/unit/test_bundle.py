import json
import tarfile
from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec
from graphrag.personas.bundle import (
    BundleError,
    export_bundle,
    import_bundle,
    read_bundle_manifest,
)
from graphrag.personas.registry import PersonaRegistry
from graphrag.pipeline import IngestReport


@pytest.fixture
def exported(
    settings: Settings,
    registry: PersonaRegistry,
    persona: PersonaSpec,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    ingested: IngestReport,
    tmp_path: Path,
) -> Path:
    """A bundle of the test persona, graph and one extraction file."""
    snap.export_snapshot(
        memory_store,
        persona,
        settings.snapshots_dir,
        embedding_model=hash_embedder.model_name,
        embedding_dim=hash_embedder.dim,
    )
    settings.enrichment_dir.mkdir(parents=True, exist_ok=True)
    (settings.enrichment_dir / "mine.json").write_text(
        json.dumps(
            {"doc_id": f"{persona.id}:test-podcast:ada-north", "entities": [], "relations": []}
        )
    )
    (settings.enrichment_dir / "theirs.json").write_text(
        json.dumps({"doc_id": "other-persona:src:doc", "entities": [], "relations": []})
    )
    out = tmp_path / "bundle.tar.gz"
    export_bundle(
        persona,
        out,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        enrichment_dir=settings.enrichment_dir,
    )
    return out


def test_export_records_what_it_contains(exported: Path, ingested: IngestReport) -> None:
    m = read_bundle_manifest(exported)
    assert m.persona_id == "test-pm"
    assert m.includes_graph is True
    assert m.document_count == 3
    assert m.chunk_count == ingested.chunks
    assert m.embedding_model == "hash-test"
    assert m.enrichment_files == 1  # the other persona's file is not swept in


def test_round_trip_into_a_fresh_installation(
    exported: Path, tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    """What the recipient gets: persona, graph and extractions, loadable into an empty store."""
    fresh = tmp_path / "buddy"
    report = import_bundle(
        exported,
        personas_dir=fresh / "personas",
        snapshots_dir=fresh / "snapshots",
        enrichment_dir=fresh / "enrichment",
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
    )
    assert report.persona_path.exists()
    assert report.snapshot_path is not None
    assert [p.name for p in report.enrichment_paths] == ["mine.json"]

    spec = PersonaRegistry(fresh / "personas").get("test-pm")
    store = InMemoryGraphStore()
    snap.load_snapshot(
        store,
        spec,
        fresh / "snapshots",
        embedding_model=hash_embedder.model_name,
        embedding_dim=hash_embedder.dim,
    )
    assert store.stats().documents == 3
    assert store.vector_search(hash_embedder.embed_query("leaky bucket"), 1)


def test_import_refuses_a_different_embedder(exported: Path, tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="GRAPHRAG_EMBEDDING_MODEL=hash-test"):
        import_bundle(
            exported,
            personas_dir=tmp_path / "p",
            snapshots_dir=tmp_path / "s",
            enrichment_dir=tmp_path / "e",
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
        )


def test_no_graph_import_ignores_the_embedder(exported: Path, tmp_path: Path) -> None:
    report = import_bundle(
        exported,
        personas_dir=tmp_path / "p",
        snapshots_dir=tmp_path / "s",
        enrichment_dir=tmp_path / "e",
        model_name="something-else",
        dim=999,
        include_graph=False,
    )
    assert report.persona_path.exists()
    assert report.snapshot_path is None


def test_overwrite_replaces_a_modified_persona(
    exported: Path, settings: Settings, hash_embedder: HashEmbedder
) -> None:
    """--overwrite is the escape hatch when the local copy really has diverged."""
    path = settings.personas_dir / "test-pm" / "persona.yaml"
    path.write_text(path.read_text() + "\n# local change\n")
    report = import_bundle(
        exported,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        enrichment_dir=settings.enrichment_dir,
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
        overwrite=True,
    )
    assert report.persona_path.exists()
    assert "# local change" not in path.read_text()


def test_export_without_graph_or_enrichment(
    settings: Settings, registry: PersonaRegistry, persona: PersonaSpec, tmp_path: Path
) -> None:
    out = tmp_path / "def-only.tar.gz"
    m = export_bundle(
        persona,
        out,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        include_graph=False,
        include_enrichment=False,
        notes="definition only",
    )
    assert m.includes_graph is False
    assert m.chunk_count == 0
    assert m.notes == "definition only"


def test_rejects_files_that_are_not_bundles(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="no such bundle"):
        read_bundle_manifest(tmp_path / "nope.tar.gz")
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a tarball")
    with pytest.raises(BundleError, match="not a readable persona bundle"):
        read_bundle_manifest(junk)
    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz") as tar:
        placeholder = tmp_path / "x.txt"
        placeholder.write_text("x")
        tar.add(placeholder, arcname="x.txt")
    with pytest.raises(BundleError, match="not a readable persona bundle"):
        read_bundle_manifest(empty)


@pytest.fixture
def fake_model_cache(tmp_path: Path) -> Path:
    """A stand-in for the ONNX cache: same shape, none of the weight."""
    cache = tmp_path / "models"
    (cache / "models--vendor--model" / "blobs").mkdir(parents=True)
    (cache / "models--vendor--model" / "blobs" / "abc123").write_bytes(b"x" * 2048)
    (cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55")
    return cache


def test_export_with_model_records_and_carries_it(
    settings: Settings,
    registry: PersonaRegistry,
    persona: PersonaSpec,
    fake_model_cache: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "with-model.tar.gz"
    m = export_bundle(
        persona,
        out,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        include_graph=False,
        model_cache_dir=fake_model_cache,
    )
    assert m.includes_model is True
    assert m.model_bytes > 2000
    with tarfile.open(out) as tar:
        assert any(n.startswith("model-cache/") for n in tar.getnames())


def test_import_restores_the_model_into_an_empty_cache(
    settings: Settings,
    registry: PersonaRegistry,
    persona: PersonaSpec,
    fake_model_cache: Path,
    tmp_path: Path,
    hash_embedder: HashEmbedder,
) -> None:
    out = tmp_path / "with-model.tar.gz"
    export_bundle(
        persona,
        out,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        include_graph=False,
        model_cache_dir=fake_model_cache,
    )
    fresh = tmp_path / "buddy"
    report = import_bundle(
        out,
        personas_dir=fresh / "personas",
        snapshots_dir=fresh / "snapshots",
        enrichment_dir=fresh / "enrichment",
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
        model_cache_dir=fresh / "models",
    )
    assert report.model_restored is True
    assert (
        fresh / "models" / "models--vendor--model" / "blobs" / "abc123"
    ).read_bytes() == b"x" * 2048


def test_import_never_clobbers_a_cached_model(
    settings: Settings,
    registry: PersonaRegistry,
    persona: PersonaSpec,
    fake_model_cache: Path,
    tmp_path: Path,
    hash_embedder: HashEmbedder,
) -> None:
    out = tmp_path / "with-model.tar.gz"
    export_bundle(
        persona,
        out,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        include_graph=False,
        model_cache_dir=fake_model_cache,
    )
    existing = tmp_path / "theirs"
    (existing / "models--vendor--model").mkdir(parents=True)
    (existing / "models--vendor--model" / "keep").write_text("mine")
    import_bundle(
        out,
        personas_dir=tmp_path / "p2",
        snapshots_dir=tmp_path / "s2",
        enrichment_dir=tmp_path / "e2",
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
        model_cache_dir=existing,
    )
    assert (existing / "models--vendor--model" / "keep").read_text() == "mine"


def test_mismatch_message_mentions_a_bundled_model(exported: Path) -> None:
    from graphrag.personas.bundle import BundleManifest, check_embedder

    m = read_bundle_manifest(exported).model_copy(update={"includes_model": True})
    with pytest.raises(BundleError, match="This bundle ships that model"):
        check_embedder(m, "other-model", 384)
    assert isinstance(m, BundleManifest)


def test_compact_halves_the_vectors_without_moving_results(
    settings: Settings,
    registry: PersonaRegistry,
    persona: PersonaSpec,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    ingested: IngestReport,
    tmp_path: Path,
) -> None:
    """--compact stores float16. Vectors shrink; the top hit stays the top hit."""
    import numpy as np

    snap.export_snapshot(
        memory_store,
        persona,
        settings.snapshots_dir,
        embedding_model=hash_embedder.model_name,
        embedding_dim=hash_embedder.dim,
    )
    full, small = tmp_path / "full.tar.gz", tmp_path / "small.tar.gz"
    for out, compact in ((full, False), (small, True)):
        export_bundle(
            persona,
            out,
            personas_dir=settings.personas_dir,
            snapshots_dir=settings.snapshots_dir,
            compact=compact,
        )
    assert small.stat().st_size < full.stat().st_size
    assert read_bundle_manifest(small).embedding_precision == "float16"
    assert read_bundle_manifest(full).embedding_precision == "float32"

    fresh = tmp_path / "buddy"
    import_bundle(
        small,
        personas_dir=fresh / "personas",
        snapshots_dir=fresh / "snapshots",
        enrichment_dir=fresh / "enrichment",
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
    )
    stored = np.load(fresh / "snapshots" / persona.id / snap.EMBEDDINGS)
    assert stored.dtype == np.float16

    store = InMemoryGraphStore()
    spec = PersonaRegistry(fresh / "personas").get(persona.id)
    snap.load_snapshot(
        store,
        spec,
        fresh / "snapshots",
        embedding_model=hash_embedder.model_name,
        embedding_dim=hash_embedder.dim,
    )
    query = hash_embedder.embed_query("retention curve leaky bucket")
    assert (
        store.vector_search(query, 1)[0].chunk_id
        == memory_store.vector_search(query, 1)[0].chunk_id
    )


def test_import_allows_replacing_an_identical_persona(
    exported: Path, settings: Settings, hash_embedder: HashEmbedder
) -> None:
    """A repo may ship a persona as an example; re-importing its own bundle is not a clobber."""
    report = import_bundle(
        exported,
        personas_dir=settings.personas_dir,
        snapshots_dir=settings.snapshots_dir,
        enrichment_dir=settings.enrichment_dir,
        model_name=hash_embedder.model_name,
        dim=hash_embedder.dim,
    )
    assert report.persona_path.exists()


def test_import_still_refuses_to_clobber_a_modified_persona(
    exported: Path, settings: Settings, hash_embedder: HashEmbedder
) -> None:
    path = settings.personas_dir / "test-pm" / "persona.yaml"
    path.write_text(path.read_text() + "\n# a local change worth keeping\n")
    with pytest.raises(BundleError, match="differs from the one in the bundle"):
        import_bundle(
            exported,
            personas_dir=settings.personas_dir,
            snapshots_dir=settings.snapshots_dir,
            enrichment_dir=settings.enrichment_dir,
            model_name=hash_embedder.model_name,
            dim=hash_embedder.dim,
        )
