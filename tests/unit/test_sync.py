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
from graphrag.models import Enrichment, Entity, Mention, PersonaSpec, SourceSpec
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
from tests.conftest import THREAD_POSTS, write_sample_corpus

PERSONA_ID = "test-multi"
TRANSCRIPTS = "test-podcast"
DOCS = "docs"


def no_embedder() -> Embedder:
    """An embedder factory that fails the test if anything tries to ingest."""
    msg = "the embedder was built, so something ingested when it should not have"
    raise AssertionError(msg)


CAPTURE_FRONT_MATTER = (
    "fetched_at: '2025-01-01T00:00:00Z'\n"
    "fetched_by: test-harness\n"
    "retrieval: manual\n"
    "content_fidelity: verbatim\n"
    "document_type: research_note\n"
    "category: test\n"
)


def make_capture_compliant(path: Path, *, filler_words: int = 0) -> None:
    """Give a `documents` source file the front-matter keys the corpus contract requires.

    This corpus fixture predates that contract (it is illustrative content, not a real capture),
    and sync now validates a `documents` source before re-ingesting it, exactly as `ingest`
    already did. `filler_words` tops up a short body past the contract's minimum word count.
    """
    text = path.read_text(encoding="utf-8")
    text = text.replace("---\n", f"---\n{CAPTURE_FRONT_MATTER}", 1)
    if filler_words:
        text += "\n" + " ".join(["filler"] * filler_words) + "\n"
    path.write_text(text, encoding="utf-8")


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
    root = write_sample_corpus(tmp_path / "raw" / PERSONA_ID)
    # `docs` and `threads` are `documents`-loader sources, so sync now validates them (like
    # `ingest`) before re-ingesting; give the fixture's illustrative files a compliant capture.
    make_capture_compliant(root / "docs" / "plan-guide.md", filler_words=12)
    make_capture_compliant(root / "threads" / "onboarding-thread.md")
    return root


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
    """One extraction file, named after its document so a test can write several at once."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc_id.rsplit(':', 1)[-1]}.json"
    payload = {
        "doc_id": doc_id,
        "entities": [{"name": name, "type": "metric", "description": "Users who come back."}],
        "relations": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def give_entities(store: InMemoryGraphStore, doc_id: str) -> None:
    """Put one entity mention on a document, which is what makes sync count it as enriched."""
    chunk = store.document_chunks(doc_id, 0, 1)[0]
    store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="metric:existing", name="Existing", type="metric")],
            mentions=[Mention(chunk_id=chunk.id, entity_id="metric:existing")],
        )
    )


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
    note = raw_root / "docs" / "new-note.md"
    note.write_text(
        "---\ntitle: New Note\n---\n\n# New Note\n\nA note written after the last ingest.\n",
        encoding="utf-8",
    )
    make_capture_compliant(note, filler_words=32)
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


# ----------------------------------------------------------------------------- corpus contract


def test_an_invalid_documents_source_is_refused_while_the_other_still_syncs(
    synced: None,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """A bad capture is invisible once it is a chunk, so sync gates a `documents` source on the
    same corpus contract `ingest` does -- without stopping a source that has nothing wrong."""
    (raw_root / "docs" / "bad-note.md").write_text(
        "---\ntitle: Bad Note\n---\n\nToo short.\n", encoding="utf-8"
    )
    new_episode = raw_root / "episodes" / "extra-guest"
    new_episode.mkdir()
    (new_episode / "transcript.md").write_text(
        "---\nguest: Extra Guest\ntitle: Extra Guest\n---\n\n"
        "## Transcript\n\nExtra Guest (00:00:00):\nWe measure everything twice.\n",
        encoding="utf-8",
    )

    report = run(
        memory_store, multi_persona, raw_root, enrichment_root, embedder=lambda: hash_embedder
    )

    by_source = {s.source_id: s for s in report.sources}
    docs = by_source[DOCS]
    assert docs.invalid is True
    assert docs.missing == ()  # refused, not counted as something that would be re-ingested
    assert docs.invalid_documents == 1
    assert any("bad-note.md" in p for p in docs.problems)
    assert f"{PERSONA_ID}:{DOCS}:bad-note" not in memory_store.document_ids(PERSONA_ID, DOCS)
    transcripts = by_source[TRANSCRIPTS]
    assert transcripts.invalid is False
    assert transcripts.ingested_documents == 4  # the untouched source still synced
    assert f"{PERSONA_ID}:{TRANSCRIPTS}:extra-guest" in memory_store.document_ids(
        PERSONA_ID, TRANSCRIPTS
    )
    assert report.errors == ()
    lines = summary_lines(report)
    docs_line = next(line for line in lines if line.startswith(f"{DOCS}: "))
    assert docs_line == (
        f"{DOCS}: refused -- 3 problem(s) in 2 captured file(s), 1 document(s) not ingested; "
        "fix the files or pass --allow-invalid"
    )
    assert any("bad-note.md" in line for line in lines)


def test_allow_invalid_re_ingests_despite_a_bad_capture(
    synced: None,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    (raw_root / "docs" / "bad-note.md").write_text(
        "---\ntitle: Bad Note\n---\n\nToo short.\n", encoding="utf-8"
    )

    report = run(
        memory_store,
        multi_persona,
        raw_root,
        enrichment_root,
        embedder=lambda: hash_embedder,
        allow_invalid=True,
    )

    docs = next(s for s in report.sources if s.source_id == DOCS)
    assert docs.invalid is False
    assert f"{PERSONA_ID}:{DOCS}:bad-note" in memory_store.document_ids(PERSONA_ID, DOCS)


def test_malformed_front_matter_is_reported_not_raised(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """A YAML parse failure used to crash while sync was only working out what is missing, before
    it ever validated anything. The loader now tolerates it; the explicit check names the file."""
    (raw_root / "docs" / "broken.md").write_text(
        "---\ntitle: [unterminated\n---\n\nToo short.\n", encoding="utf-8"
    )

    report = run(memory_store, multi_persona, raw_root, enrichment_root, dry_run=True)

    docs = next(s for s in report.sources if s.source_id == DOCS)
    assert docs.invalid is True
    assert any("broken.md" in p and "front-matter" in p for p in docs.problems)


def test_summary_lines_report_a_refused_source() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(
            SourceReport(
                source_id="a",
                invalid=True,
                invalid_files=2,
                invalid_documents=1,
                problems=("NEAR-EMPTY (2 words): bad.md",),
            ),
            SourceReport(source_id="b"),
        ),
    )
    lines = summary_lines(report)
    assert lines[0] == (
        "a: refused -- 1 problem(s) in 2 captured file(s), 1 document(s) not ingested; "
        "fix the files or pass --allow-invalid"
    )
    assert lines[1] == "  NEAR-EMPTY (2 words): bad.md"
    assert lines[2] == "b: up to date"


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


# ------------------------------------------------------------------- ingested but not extracted


def test_an_up_to_date_source_imports_files_for_documents_with_no_entities(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """Ingest and extraction are separate steps, so a complete source can still lack entities."""
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))
    give_entities(memory_store, doc_ids[0])
    for doc_id in doc_ids[:2]:
        write_extraction(enrichment_root / PERSONA_ID / TRANSCRIPTS, doc_id)

    report = run(memory_store, multi_persona, raw_root, enrichment_root)

    transcripts = next(s for s in report.sources if s.source_id == TRANSCRIPTS)
    assert transcripts.stale is False  # `no_embedder` proves nothing was re-ingested
    assert transcripts.enrichment_files == 1  # the file for doc_ids[1] only
    assert transcripts.enrichment_errors == ()
    assert memory_store.enriched_document_ids(PERSONA_ID, TRANSCRIPTS) == set(doc_ids[:2])
    assert any(m.entity_id == "metric:retention" for m in memory_store.mentions)
    assert report.wrote is True
    assert summary_lines(report)[0] == (
        f"{TRANSCRIPTS}: up to date, imported 1 extraction files for documents without entities"
    )


def test_a_document_that_already_has_entities_is_not_imported_again(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """Otherwise every run would re-import the whole corpus for nothing."""
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, DOCS))
    for doc_id in doc_ids:
        give_entities(memory_store, doc_id)
        write_extraction(enrichment_root / PERSONA_ID / DOCS, doc_id)
    before = list(memory_store.mentions)

    report = run(memory_store, multi_persona, raw_root, enrichment_root)

    docs = next(s for s in report.sources if s.source_id == DOCS)
    assert docs.enrichment_files == 0
    assert memory_store.mentions == before
    assert set(memory_store.entities) == {"metric:existing"}  # the files' entity never landed
    assert report.wrote is False
    assert summary_lines(report)[-1] == f"{PERSONA_ID}: nothing to sync"


def test_refresh_extraction_re_imports_documents_that_already_have_entities(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """The flag is how an entity attribute added to an extraction file reaches a graph that
    already has that document's entities, which the gap check alone never re-reads."""
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, DOCS))
    for doc_id in doc_ids:
        give_entities(memory_store, doc_id)
        write_extraction(enrichment_root / PERSONA_ID / DOCS, doc_id)
    memory_store.set_entity_attributes(PERSONA_ID, "metric:existing", {"region": "south"})

    report = run(memory_store, multi_persona, raw_root, enrichment_root, refresh_extraction=True)

    docs = next(s for s in report.sources if s.source_id == DOCS)
    assert docs.enrichment_files == len(doc_ids)
    assert memory_store.entity_attributes(PERSONA_ID) == {}  # cleared, and no file restates it
    assert summary_lines(report)[-1] == (
        f"{PERSONA_ID}: nothing to ingest, re-imported {len(doc_ids)} extraction files"
    )


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


def test_dry_run_counts_files_for_documents_with_no_entities_without_importing_them(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))
    write_extraction(enrichment_root / PERSONA_ID / TRANSCRIPTS, doc_ids[0])

    report = run(memory_store, multi_persona, raw_root, enrichment_root, dry_run=True)

    transcripts = next(s for s in report.sources if s.source_id == TRANSCRIPTS)
    assert transcripts.stale is False
    assert transcripts.enrichment_files == 1
    assert memory_store.entities == {}
    assert memory_store.mentions == []
    assert report.wrote is False
    assert summary_lines(report)[0] == (
        f"{TRANSCRIPTS}: up to date, would import 1 extraction files for documents without entities"
    )


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


# ----------------------------------------------------------------------------- attribution

THREADS = "threads"


@pytest.fixture
def thread_persona(thread_source: SourceSpec) -> PersonaSpec:
    """One source of prose documents: ingesting it gives the graph no speakers at all."""
    return PersonaSpec(id=PERSONA_ID, name="Test Threads", sources=[thread_source])


@pytest.fixture
def attribution_root(tmp_path: Path) -> Path:
    path = tmp_path / "attribution"
    path.mkdir()
    return path


@pytest.fixture
def threaded(
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    thread_persona: PersonaSpec,
    raw_root: Path,
) -> str:
    """The ingested thread's document id: the state sync is supposed to find a speaker gap in."""
    IngestPipeline(memory_store, hash_embedder).ingest(
        raw_root, thread_persona, thread_persona.sources[0]
    )
    return next(iter(memory_store.document_ids(PERSONA_ID, THREADS)))


def write_attribution(directory: Path, doc_id: str, speaker: str = "quill-maker") -> Path:
    """One attribution file whose single anchor is the thread's opening sentence."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc_id.rsplit(':', 1)[-1]}.json"
    payload = {
        "doc_id": doc_id,
        "posts": [{"speaker": speaker, "anchor": THREAD_POSTS[0][2][:60], "role": "op"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_up_to_date_source_imports_files_for_documents_with_no_speakers(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """Ingest and attribution are separate steps, exactly as ingest and extraction are."""
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded)

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.stale is False  # `no_embedder` proves nothing was re-ingested
    assert threads.attribution_files == 1
    assert threads.attribution_errors == ()
    assert memory_store.documents[threaded].speakers == ["quill-maker"]
    assert memory_store.attributed_document_ids(PERSONA_ID, THREADS) == {threaded}
    assert report.wrote is True
    assert summary_lines(report)[0] == (
        f"{THREADS}: up to date, imported 1 attribution files for documents without speakers"
    )
    assert summary_lines(report)[-1] == (
        f"{PERSONA_ID}: nothing to ingest, imported 1 attribution files for documents "
        f"without speakers"
    )


def test_a_document_that_already_has_speakers_is_not_imported_again(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """Otherwise every run would re-attribute the whole corpus for nothing."""
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded, speaker="from-the-file")
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "already-here"
    )

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.attribution_files == 0
    assert memory_store.documents[threaded].speakers == ["already-here"]
    assert report.wrote is False
    assert summary_lines(report)[-1] == f"{PERSONA_ID}: nothing to sync"


def test_refresh_attribution_re_imports_a_document_that_already_has_speakers(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """The case the gap test cannot see: a file rewritten with new fields for a document that
    already has speakers. Nothing in the graph says the file changed, so only the flag does."""
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded, speaker="from-the-file")
    write_attribution(attribution_root / PERSONA_ID / THREADS, f"{PERSONA_ID}:{THREADS}:ghost")
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "already-here"
    )

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        refresh_attribution=True,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.stale is False  # `no_embedder` proves nothing was re-ingested
    assert threads.attribution_files == 1  # the file for a document the graph lacks stays out
    assert threads.attribution_errors == ()
    assert "from-the-file" in memory_store.documents[threaded].speakers
    assert report.wrote is True
    assert summary_lines(report)[0] == f"{THREADS}: up to date, re-imported 1 attribution files"
    assert summary_lines(report)[-1] == (
        f"{PERSONA_ID}: nothing to ingest, re-imported 1 attribution files"
    )


def test_a_refresh_is_how_node_attributes_reach_a_graph_that_already_has_the_layer(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
    annotation_root: Path,
) -> None:
    """A tagging pass rewrites sidecars for documents that already have speakers and stances.

    Nothing in the graph says those files changed, so the two refresh flags are the only thing
    that carries the new attributes in; what the vocabulary refuses is said out loud on the way.
    """
    from graphrag.extract.attributes import load_attributes

    vocabulary = raw_root.parent / "facets.yaml"
    vocabulary.write_text("attributes:\n  region:\n    values: [north, south]\n", encoding="utf-8")
    directory = attribution_root / PERSONA_ID / THREADS
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{threaded.rsplit(':', 1)[-1]}.json").write_text(
        json.dumps(
            {
                "doc_id": threaded,
                "posts": [
                    {
                        "speaker": "quill-maker",
                        "anchor": THREAD_POSTS[0][2][:60],
                        "role": "op",
                        "attributes": {"region": "north", "colour": "green"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    readings = annotation_root / PERSONA_ID / THREADS
    readings.mkdir(parents=True, exist_ok=True)
    (readings / f"{threaded.rsplit(':', 1)[-1]}.json").write_text(
        json.dumps({"doc_id": threaded, "attributes": {"region": "south"}}), encoding="utf-8"
    )
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "quill-maker"
    )
    said: list[str] = []

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        annotation_root=annotation_root,
        attributes=load_attributes(vocabulary),
        refresh_attribution=True,
        refresh_annotations=True,
        progress=said.append,
    )

    assert report.errors == ()
    assert memory_store.speaker_attributes(PERSONA_ID) == {"quill-maker": {"region": "north"}}
    assert memory_store.documents[threaded].attributes == {"region": "south"}
    assert any("attribute invalid: colour" in line for line in said)


def _one_post_file(attribution_root: Path, threaded: str, region: str) -> None:
    directory = attribution_root / PERSONA_ID / THREADS
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{threaded.rsplit(':', 1)[-1]}.json").write_text(
        json.dumps(
            {
                "doc_id": threaded,
                "posts": [
                    {
                        "speaker": "quill-maker",
                        "anchor": THREAD_POSTS[0][2][:60],
                        "role": "op",
                        "attributes": {"region": region},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_a_refresh_lets_a_corrected_file_replace_the_value_its_old_version_wrote(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """First value wins, so a corrected attribution file used to be reported as a conflict with
    its own earlier version and change nothing. A persona-wide refresh clears the stored speaker
    values first; another persona's reading of the same shared speaker is left alone."""
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "quill-maker"
    )
    memory_store.set_speaker_attributes(PERSONA_ID, "quill-maker", {"region": "south"})
    memory_store.set_speaker_attributes("someone-else", "quill-maker", {"region": "west"})
    _one_post_file(attribution_root, threaded, "north")
    said: list[str] = []

    run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        refresh_attribution=True,
        progress=said.append,
    )

    assert memory_store.speaker_attributes(PERSONA_ID) == {"quill-maker": {"region": "north"}}
    assert memory_store.speaker_attributes("someone-else") == {"quill-maker": {"region": "west"}}
    assert "cleared stored speaker attributes on 1 node(s) before re-reading" in said
    assert not any("attribute conflict" in line for line in said)


def test_a_one_source_refresh_clears_nothing(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """The other sources' files set values too and are not re-read, so a refresh limited to one
    source keeps what is stored and reports the disagreement instead."""
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "quill-maker"
    )
    memory_store.set_speaker_attributes(PERSONA_ID, "quill-maker", {"region": "south"})
    _one_post_file(attribution_root, threaded, "north")
    said: list[str] = []

    run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        refresh_attribution=True,
        source_id=THREADS,
        progress=said.append,
    )

    assert memory_store.speaker_attributes(PERSONA_ID) == {"quill-maker": {"region": "south"}}
    assert any("attribute conflict" in line for line in said)


def test_without_the_flag_a_rewritten_attribution_file_is_left_alone(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """Which is the behaviour the flag exists to override, asserted beside it."""
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded, speaker="from-the-file")
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "already-here"
    )

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
    )

    assert next(s for s in report.sources if s.source_id == THREADS).attribution_files == 0
    assert memory_store.documents[threaded].speakers == ["already-here"]


def test_a_dry_run_counts_a_refresh_without_importing_it(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded, speaker="from-the-file")
    memory_store.attach_speaker(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, "already-here"
    )

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        refresh_attribution=True,
        dry_run=True,
    )

    assert next(s for s in report.sources if s.source_id == THREADS).attribution_files == 1
    assert memory_store.documents[threaded].speakers == ["already-here"]
    assert report.wrote is False
    assert summary_lines(report)[0] == (
        f"{THREADS}: up to date, would re-import 1 attribution files"
    )


def test_attribution_is_reimported_because_a_re_ingest_drops_speakers(
    threaded: str,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    """SPOKE edges hang off chunks, so a re-ingest costs a document its attributed speakers."""
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded)
    memory_store.delete_documents([threaded])  # as if the thread never reached the graph

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        embedder=lambda: hash_embedder,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.missing == (threaded,)
    assert threads.attribution_files == 1
    assert memory_store.documents[threaded].speakers == ["quill-maker"]
    assert summary_lines(report)[0].endswith(
        "re-imported 0 extraction files and 1 attribution files"
    )


def test_an_unimportable_attribution_file_is_reported_not_raised(
    threaded: str,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    broken = attribution_root / PERSONA_ID / THREADS / "broken.json"
    broken.parent.mkdir(parents=True)
    broken.write_text(json.dumps({"doc_id": f"{PERSONA_ID}:{THREADS}:gone"}), encoding="utf-8")
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded)
    memory_store.delete_documents([threaded])

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        embedder=lambda: hash_embedder,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.attribution_files == 1
    assert len(threads.attribution_errors) == 1
    assert "unknown document" in threads.attribution_errors[0]
    assert report.errors == threads.attribution_errors
    assert summary_lines(report)[1] == f"  broken.json: {threads.attribution_errors[0][13:]}"


def test_a_dry_run_counts_attribution_files_without_importing_them(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path,
) -> None:
    write_attribution(attribution_root / PERSONA_ID / THREADS, threaded)

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        attribution_root=attribution_root,
        dry_run=True,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.attribution_files == 1
    assert memory_store.documents[threaded].speakers == []
    assert report.wrote is False
    assert summary_lines(report)[0] == (
        f"{THREADS}: up to date, would import 1 attribution files for documents without speakers"
    )


def test_no_attribution_directory_is_not_a_finding(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """Most personas have none, and their sync output must read as it always did."""
    report = run(memory_store, thread_persona, raw_root, enrichment_root)

    assert report.backfilled_attribution_files == 0
    assert summary_lines(report) == [f"{THREADS}: up to date", f"{PERSONA_ID}: nothing to sync"]


def test_summary_lines_report_both_layers_when_both_did_something() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(SourceReport(source_id="a", enrichment_files=2, attribution_files=3),),
    )
    assert summary_lines(report)[0] == (
        "a: up to date, imported 2 extraction files for documents without entities, "
        "imported 3 attribution files for documents without speakers"
    )
    assert summary_lines(report)[-1] == (
        "p: nothing to ingest, imported 2 extraction files for documents without entities, "
        "imported 3 attribution files for documents without speakers"
    )


# ----------------------------------------------------------------------------- annotation


@pytest.fixture
def annotation_root(tmp_path: Path) -> Path:
    path = tmp_path / "annotations"
    path.mkdir()
    return path


def write_annotation(directory: Path, doc_id: str, facet: str = "handover") -> Path:
    """One annotation file whose single anchor is the thread's opening sentence."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc_id.rsplit(':', 1)[-1]}.json"
    payload = {
        "doc_id": doc_id,
        "annotations": [{"anchor": THREAD_POSTS[0][2][:60], "facets": [facet]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_up_to_date_source_imports_files_for_documents_with_no_annotations(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    """Ingest and annotation are separate steps, exactly as ingest and extraction are."""
    write_annotation(annotation_root / PERSONA_ID / THREADS, threaded)

    report = run(
        memory_store, thread_persona, raw_root, enrichment_root, annotation_root=annotation_root
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.stale is False  # `no_embedder` proves nothing was re-ingested
    assert (threads.annotation_files, threads.annotation_errors) == (1, ())
    assert memory_store.chunk_facets(PERSONA_ID)[0].facets == ["handover"]
    assert report.wrote is True
    assert summary_lines(report)[0] == (
        f"{THREADS}: up to date, imported 1 annotation files for documents without annotations"
    )


def test_a_document_that_already_has_annotations_is_not_imported_again(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    """Otherwise every run would re-annotate the whole corpus for nothing."""
    write_annotation(annotation_root / PERSONA_ID / THREADS, threaded, facet="from-the-file")
    memory_store.annotate_chunk(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, ["already-here"]
    )

    report = run(
        memory_store, thread_persona, raw_root, enrichment_root, annotation_root=annotation_root
    )

    assert next(s for s in report.sources if s.source_id == THREADS).annotation_files == 0
    assert memory_store.chunk_facets(PERSONA_ID)[0].facets == ["already-here"]
    assert report.wrote is False


def test_refresh_annotations_re_imports_a_document_that_already_has_annotations(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    """The annotation layer's half of the same problem: a reading added to a document that
    already carries one is invisible to a check that only asks whether the layer is there."""
    write_annotation(annotation_root / PERSONA_ID / THREADS, threaded, facet="from-the-file")
    memory_store.annotate_chunk(
        threaded, memory_store.document_chunks(threaded, 0, 1)[0].id, ["already-here"]
    )

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        annotation_root=annotation_root,
        refresh_annotations=True,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.annotation_files == 1
    assert threads.annotation_errors == ()
    assert "from-the-file" in memory_store.chunk_facets(PERSONA_ID)[0].facets
    assert report.wrote is True
    assert summary_lines(report)[0] == f"{THREADS}: up to date, re-imported 1 annotation files"


def test_annotations_are_reimported_because_a_re_ingest_drops_them(
    threaded: str,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    """Facets sit on a passage and a stance on a mention, and a re-ingest deletes both."""
    write_annotation(annotation_root / PERSONA_ID / THREADS, threaded)
    run(memory_store, thread_persona, raw_root, enrichment_root, annotation_root=annotation_root)
    memory_store.delete_documents([threaded])
    assert memory_store.chunk_facets(PERSONA_ID) == []

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        embedder=lambda: hash_embedder,
        annotation_root=annotation_root,
    )

    threads = next(s for s in report.sources if s.source_id == THREADS)
    assert threads.stale is True and threads.annotation_files == 1
    assert memory_store.chunk_facets(PERSONA_ID)[0].facets == ["handover"]


def test_an_unimportable_annotation_file_is_reported_not_raised(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    folder = annotation_root / PERSONA_ID / THREADS
    folder.mkdir(parents=True)
    (folder / "bad.json").write_text(
        json.dumps({"doc_id": threaded, "annotations": [{"anchor": "  "}]}), encoding="utf-8"
    )

    report = run(
        memory_store, thread_persona, raw_root, enrichment_root, annotation_root=annotation_root
    )

    assert len(report.errors) == 1 and report.errors[0].startswith("bad.json: invalid")
    assert "  bad.json: invalid" in summary_lines(report)[1]


def test_a_dry_run_counts_annotation_files_without_importing_them(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    annotation_root: Path,
) -> None:
    write_annotation(annotation_root / PERSONA_ID / THREADS, threaded)

    report = run(
        memory_store,
        thread_persona,
        raw_root,
        enrichment_root,
        annotation_root=annotation_root,
        dry_run=True,
    )

    assert report.backfilled_annotation_files == 1
    assert report.wrote is False
    assert memory_store.chunk_facets(PERSONA_ID) == []
    assert summary_lines(report)[0].endswith(
        "would import 1 annotation files for documents without annotations"
    )


def test_no_annotation_directory_is_not_a_finding(
    threaded: str,
    memory_store: InMemoryGraphStore,
    thread_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
) -> None:
    """Most personas have none, and their sync output must read as it always did."""
    report = run(memory_store, thread_persona, raw_root, enrichment_root)

    assert report.backfilled_annotation_files == 0
    assert summary_lines(report) == [f"{THREADS}: up to date", f"{PERSONA_ID}: nothing to sync"]


# ----------------------------------------------------------------------------- aliases


def test_aliases_are_applied_once_the_whole_graph_has_landed(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    tmp_path: Path,
) -> None:
    """Two spellings that arrived in different files can only be folded together afterwards.

    They are in the graph here because they were imported before anyone wrote the table, which
    is the situation the pass exists for.
    """
    from graphrag.extract.aliases import load_aliases

    doc_ids = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))
    for doc_id, entity in zip(
        doc_ids,
        [
            Entity(id="metric:retention", name="Retention", type="metric"),
            Entity(id="metric:retention-rate", name="retention rate", type="metric"),
        ],
        strict=False,
    ):
        chunk = memory_store.document_chunks(doc_id, 0, 1)[0]
        memory_store.upsert_enrichment(
            Enrichment(
                entities=[entity], mentions=[Mention(chunk_id=chunk.id, entity_id=entity.id)]
            )
        )
    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  Retention: [retention rate]\n", encoding="utf-8")

    report = run(memory_store, multi_persona, raw_root, enrichment_root, aliases=load_aliases(path))

    assert report.aliases is not None and report.aliases.mentions_moved == 1
    assert "metric:retention-rate" not in memory_store.entities
    assert memory_store.entities["metric:retention"].aliases == ["retention rate"]
    assert report.wrote is True  # folding the graph is a change worth a new snapshot
    assert "aliases: folded 1 mentions onto 1 canonical names" in summary_lines(report)


def test_an_import_during_sync_writes_the_canonical_name_in_the_first_place(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    tmp_path: Path,
) -> None:
    """So the alias node the pass would clean up is never created by a synced import."""
    from graphrag.extract.aliases import load_aliases

    doc_id = sorted(memory_store.document_ids(PERSONA_ID, TRANSCRIPTS))[0]
    write_extraction(enrichment_root / PERSONA_ID / TRANSCRIPTS, doc_id, name="retention rate")
    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  Retention: [retention rate]\n", encoding="utf-8")

    run(memory_store, multi_persona, raw_root, enrichment_root, aliases=load_aliases(path))

    assert list(memory_store.entities) == ["metric:retention"]
    assert memory_store.entities["metric:retention"].aliases == ["retention rate"]


def test_a_dry_run_never_folds_the_graph(
    synced: None,
    memory_store: InMemoryGraphStore,
    multi_persona: PersonaSpec,
    raw_root: Path,
    enrichment_root: Path,
    tmp_path: Path,
) -> None:
    from graphrag.extract.aliases import load_aliases

    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  Retention: [retention rate]\n", encoding="utf-8")

    report = run(
        memory_store,
        multi_persona,
        raw_root,
        enrichment_root,
        aliases=load_aliases(path),
        dry_run=True,
    )

    assert report.aliases is None
    assert summary_lines(report)[-1] == f"{PERSONA_ID}: nothing to sync"


def test_summary_lines_report_all_three_layers_when_all_three_did_something() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(
            SourceReport(
                source_id="a", enrichment_files=2, attribution_files=3, annotation_files=4
            ),
        ),
    )
    assert summary_lines(report)[0] == (
        "a: up to date, imported 2 extraction files for documents without entities, "
        "imported 3 attribution files for documents without speakers, "
        "imported 4 annotation files for documents without annotations"
    )
    assert summary_lines(report)[-1] == (
        "p: nothing to ingest, imported 2 extraction files for documents without entities, "
        "imported 3 attribution files for documents without speakers, "
        "imported 4 annotation files for documents without annotations"
    )


def test_summary_lines_tell_a_refresh_apart_from_a_backfill() -> None:
    """The two answer different questions, so a reader should not have to guess which ran."""
    report = SyncReport(
        persona_id="p",
        sources=(
            SourceReport(
                source_id="a",
                enrichment_files=2,
                attribution_files=3,
                annotation_files=4,
                attribution_refreshed=True,
            ),
        ),
    )
    assert summary_lines(report)[0] == (
        "a: up to date, imported 2 extraction files for documents without entities, "
        "re-imported 3 attribution files, "
        "imported 4 annotation files for documents without annotations"
    )
    assert summary_lines(report)[-1] == (
        "p: nothing to ingest, imported 2 extraction files for documents without entities, "
        "re-imported 3 attribution files, "
        "imported 4 annotation files for documents without annotations"
    )


def test_a_stale_source_line_names_every_layer_it_re_imported() -> None:
    report = SyncReport(
        persona_id="p",
        sources=(
            SourceReport(
                source_id="a",
                missing=("p:a:one",),
                ingested_documents=1,
                ingested_chunks=2,
                enrichment_files=1,
                attribution_files=1,
                annotation_files=1,
            ),
        ),
    )
    assert summary_lines(report)[0] == (
        "a: 1 missing -> ingested 1 documents / 2 chunks, re-imported 1 extraction files "
        "and 1 attribution files and 1 annotation files"
    )
