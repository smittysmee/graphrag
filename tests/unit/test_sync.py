"""``graphrag sync``: work out which sources are behind, ingest those, restore their entities.

Everything here runs against the in-memory store and the hash embedder. The embedder is passed
as a factory, so a test can prove no ingest happened by passing one that refuses to build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag.embed.base import Embedder
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Mention, PersonaSpec, SourceSpec
from graphrag.pipeline import IngestPipeline
from graphrag.sync import (
    SourceReport,
    SyncReport,
    UnknownSourceError,
    expected_document_ids,
    index_extractions,
    summary_lines,
    sync_persona,
)
from tests.conftest import write_sample_corpus

PERSONA_ID = "test-multi"
TRANSCRIPTS = "test-podcast"
DOCS = "docs"


def no_embedder() -> Embedder:
    """An embedder factory that fails the test if anything tries to ingest."""
    msg = "the embedder was built, so something ingested when it should not have"
    raise AssertionError(msg)


@pytest.fixture
def multi_persona(transcript_source: SourceSpec, documents_source: SourceSpec) -> PersonaSpec:
    """A persona with two sources, which is where `graphrag ingest` alone stops being enough."""
    return PersonaSpec(
        id=PERSONA_ID,
        name="Test Multi Source",
        sources=[transcript_source, documents_source],
    )


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return write_sample_corpus(tmp_path / "raw" / PERSONA_ID)


@pytest.fixture
def enrichment_root(tmp_path: Path) -> Path:
    path = tmp_path / "enrichment"
    path.mkdir()
    return path


@pytest.fixture
def synced(
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
) -> None:
    """Both sources fully ingested: the state sync is supposed to leave alone."""
    pipeline = IngestPipeline(memory_store, hash_embedder)
    for source in multi_persona.sources:
        pipeline.ingest(raw_root, multi_persona, source)


def write_extraction(directory: Path, doc_id: str, name: str = "Retention") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.lower()}.json"
    payload = {
        "doc_id": doc_id,
        "entities": [{"name": name, "type": "metric", "description": "Users who come back."}],
        "relations": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run(
    store: InMemoryGraphStore,
    persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    *,
    embedder: object = None,
    **kwargs: object,
) -> SyncReport:
    return sync_persona(
        persona,
        store=store,
        embedder=embedder or no_embedder,  # type: ignore[arg-type]
        raw_root=raw_root,
        enrichment_root=enrichment_root,
        **kwargs,  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------------- nothing missing


def test_a_complete_persona_is_left_alone(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """No ingest, no embedder, no writes: `no_embedder` would raise if the pipeline was built."""
    before = dict(memory_store.documents)
    report = run(memory_store, multi_persona, raw_root, enrichment_root)

    assert report.missing_total == 0
    assert report.stale_sources == ()
    assert {s.source_id for s in report.sources} == {TRANSCRIPTS, DOCS}
    assert memory_store.documents == before


# ----------------------------------------------------------------------------- one source behind


def test_a_new_file_re_ingests_only_its_own_source(
    synced: None,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    (raw_root / "docs" / "new-note.md").write_text(
        "---\ntitle: New Note\n---\n\n# New Note\n\nA note written after the last ingest.\n",
        encoding="utf-8",
    )
    # A mention on the untouched source: re-ingesting would delete its chunk and so this too.
    transcript_chunk = next(
        c for c in memory_store.chunks.values() if f":{TRANSCRIPTS}:" in c.doc_id
    )
    memory_store.mentions.append(
        Mention(chunk_id=transcript_chunk.id, entity_id="metric:untouched")
    )

    report = run(
        memory_store, multi_persona, raw_root, enrichment_root, embedder=lambda: hash_embedder
    )

    by_source = {s.source_id: s for s in report.sources}
    assert by_source[TRANSCRIPTS].stale is False
    assert by_source[DOCS].missing == (f"{PERSONA_ID}:{DOCS}:new-note",)
    assert by_source[DOCS].ingested_documents == 3  # the whole source, not just the new file
    assert f"{PERSONA_ID}:{DOCS}:new-note" in memory_store.document_ids(PERSONA_ID, DOCS)
    assert any(m.entity_id == "metric:untouched" for m in memory_store.mentions)


def test_enrichment_is_reimported_because_a_re_ingest_drops_mentions(
    synced: None,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))
    write_extraction(enrichment_root / PERSONA_ID / TRANSCRIPTS, doc_ids[0])
    memory_store.delete_documents([doc_ids[1]])  # as if a transcript never reached the graph

    report = run(
        memory_store, multi_persona, raw_root, enrichment_root, embedder=lambda: hash_embedder
    )

    transcripts = next(s for s in report.sources if s.source_id == TRANSCRIPTS)
    assert transcripts.missing == (doc_ids[1],)
    assert transcripts.enrichment_files == 1
    assert transcripts.enrichment_errors == ()
    assert memory_store.document_ids(PERSONA_ID, TRANSCRIPTS) == set(doc_ids)
    assert [e.name for e in memory_store.entities.values()] == ["Retention"]
    assert any(m.entity_id == "metric:retention" for m in memory_store.mentions)


def test_an_unimportable_extraction_file_is_reported_not_raised(
    synced: None,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, DOCS))
    write_extraction(enrichment_root / PERSONA_ID / DOCS, doc_ids[0])
    broken = enrichment_root / PERSONA_ID / DOCS / "broken.json"
    broken.write_text(json.dumps({"doc_id": f"{PERSONA_ID}:{DOCS}:gone"}), encoding="utf-8")
    memory_store.delete_documents([doc_ids[0]])

    report = run(
        memory_store, multi_persona, raw_root, enrichment_root, embedder=lambda: hash_embedder
    )

    docs = next(s for s in report.sources if s.source_id == DOCS)
    assert docs.enrichment_files == 1
    assert len(docs.enrichment_errors) == 1
    assert "unknown document" in docs.enrichment_errors[0]
    assert report.errors == docs.enrichment_errors


# ----------------------------------------------------------------------------- dry run


def test_dry_run_reports_and_writes_nothing(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))
    write_extraction(enrichment_root / PERSONA_ID / TRANSCRIPTS, doc_ids[0])
    memory_store.delete_documents([doc_ids[1]])
    before = dict(memory_store.documents)

    report = run(memory_store, multi_persona, raw_root, enrichment_root, dry_run=True)

    transcripts = next(s for s in report.sources if s.source_id == TRANSCRIPTS)
    assert transcripts.missing == (doc_ids[1],)
    assert transcripts.enrichment_files == 1  # what it *would* re-import
    assert transcripts.ingested_documents == 0
    assert memory_store.documents == before
    assert memory_store.entities == {}


# ----------------------------------------------------------------------------- source selection


def test_source_limits_the_run(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    report = run(
        memory_store, multi_persona, raw_root, enrichment_root, source_id=DOCS, dry_run=True
    )
    assert [s.source_id for s in report.sources] == [DOCS]


def test_an_unknown_source_is_rejected_with_the_known_ones(
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    with pytest.raises(UnknownSourceError, match=TRANSCRIPTS):
        run(memory_store, multi_persona, raw_root, enrichment_root, source_id="nope")


# ----------------------------------------------------------------------------- expected ids


def test_expected_ids_come_from_the_loader_so_duplicates_do_not_count(
    multi_persona: PersonaSpec, raw_root: Path, transcript_source: SourceSpec
) -> None:
    """An episode archived twice is one document; a file-count comparison would loop forever."""
    original = raw_root / "episodes" / "ada-north" / "transcript.md"
    duplicate = raw_root / "episodes" / "ada-north-reupload"
    duplicate.mkdir()
    (duplicate / "transcript.md").write_text(original.read_text(encoding="utf-8"), encoding="utf-8")

    ids = expected_document_ids(raw_root, multi_persona, transcript_source)
    assert len(ids) == 3
    assert f"{PERSONA_ID}:{TRANSCRIPTS}:ada-north" in ids


def test_an_absent_source_directory_is_not_a_finding(
    memory_store: InMemoryGraphStore, multi_persona: PersonaSpec, tmp_path: Path
) -> None:
    """A source whose files are not checked out (a git submodule) must not look un-ingested."""
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    report = run(memory_store, multi_persona, empty, tmp_path / "enrichment", dry_run=True)
    assert report.missing_total == 0


# ----------------------------------------------------------------------------- extraction index


def test_extractions_are_grouped_by_doc_id_not_by_directory(tmp_path: Path) -> None:
    """Both corpus layouts exist; the id inside the file is what ties it to a source."""
    root = tmp_path / "enrichment"
    nested = write_extraction(root / "alpha" / "one", "alpha:one:doc-a")
    flat = write_extraction(root, "alpha:two:doc-b", name="Onboarding")
    (root / "not-json.md").write_text("ignore me", encoding="utf-8")
    (root / "no-doc-id.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
    (root / "unreadable.json").write_text("{ broken", encoding="utf-8")

    grouped = index_extractions(root)

    assert grouped == {"alpha:one:": [nested], "alpha:two:": [flat]}


def test_indexing_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert index_extractions(tmp_path / "absent") == {}


# ----------------------------------------------------------------------------- summary_lines


def test_summary_lines_say_nothing_to_sync_when_everything_is_present() -> None:
    report = SyncReport(persona_id="p", sources=(SourceReport(source_id="a"),))
    assert summary_lines(report) == ["a: up to date", "p: nothing to sync"]


def test_summary_lines_report_a_dry_run_in_the_conditional() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(SourceReport(source_id="a", missing=("p:a:x",), enrichment_files=2),),
        dry_run=True,
    )
    lines = summary_lines(report)
    assert lines[0] == "a: 1 documents missing; would re-ingest and re-import 2 extraction files"
    assert lines[-1] == "p: 1 documents would be ingested across 1 source(s)"


def test_summary_lines_report_what_a_real_run_did() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(
            SourceReport(
                source_id="a",
                missing=("p:a:x",),
                ingested_documents=4,
                ingested_chunks=9,
                enrichment_files=2,
                enrichment_errors=("bad.json: invalid",),
            ),
            SourceReport(source_id="b"),
        ),
    )
    lines = summary_lines(report)
    assert lines[0] == (
        "a: 1 missing -> ingested 4 documents / 9 chunks, re-imported 2 extraction files"
    )
    assert lines[1] == "  bad.json: invalid"
    assert lines[2] == "b: up to date"
    assert lines[-1] == "p: synced 1 source(s), 1 documents were missing"
