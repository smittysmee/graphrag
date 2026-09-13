"""``graphrag layers check``: which layers a captured document has, and what came loose.

Everything runs against the in-memory store and the synthetic corpus from ``tests/conftest.py``,
so the ids here are invented ones and no real document is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.extract.layers import (
    SourceNotFoundError,
    check_documents,
    doc_id_for_file,
    index_sidecars,
    summary_lines,
)
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec, SourceSpec
from graphrag.pipeline import IngestReport
from tests.conftest import THREAD_POSTS

runner = CliRunner()

FIRST = THREAD_POSTS[0][2][:60]
LAST = THREAD_POSTS[-1][2][:60]

#: A name that occurs verbatim in the first episode, so its mention anchors exactly.
VERBATIM = "Product-market fit"
#: Not verbatim, but every token of it occurs: the importer places it loosely.
TOKENS = "retention curve flattening"
#: No token of it occurs anywhere, so the mention falls back to the first passage.
ABSENT = "Zzyzx Protocol"


def write_extraction(path: Path, doc_id: str, names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "entities": [
                    {"name": name, "type": "concept", "description": f"{name}, as discussed."}
                    for name in names
                ],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_attribution(path: Path, doc_id: str, posts: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, "posts": posts}), encoding="utf-8")
    return path


def write_annotations(path: Path, doc_id: str, annotations: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, "annotations": annotations}), encoding="utf-8")
    return path


@pytest.fixture
def episode_id(memory_store: InMemoryGraphStore, ingested: IngestReport) -> str:
    """The id of the one episode whose text the extraction fixtures name."""
    return "test-pm:test-podcast:ada-north"


# ----------------------------------------------------------------------------- doc ids


def test_a_documents_source_slugifies_the_path_under_it(docs_persona: PersonaSpec) -> None:
    doc_id, source = doc_id_for_file(
        docs_persona,
        Path("/repo/data/raw/test-docs"),
        Path("/repo/data/raw/test-docs/docs/faq.txt"),
    )
    assert (doc_id, source.id) == ("test-docs:docs:faq", "docs")


def test_a_transcripts_source_slugifies_the_containing_folder(persona: PersonaSpec) -> None:
    """One folder is one recording, so the file name inside it never reaches the id."""
    doc_id, _source = doc_id_for_file(
        persona,
        Path("/repo/data/raw/test-pm"),
        Path("/repo/data/raw/test-pm/episodes/ada-north/transcript.md"),
    )
    assert doc_id == "test-pm:test-podcast:ada-north"


def test_a_nested_path_keeps_its_folders_in_the_slug(docs_persona: PersonaSpec) -> None:
    doc_id, _source = doc_id_for_file(
        docs_persona,
        Path("/repo/data/raw/test-docs"),
        Path("/repo/data/raw/test-docs/docs/notes/2026-01-02-a-note.md"),
    )
    assert doc_id == "test-docs:docs:notes-2026-01-02-a-note"


def test_the_deeper_source_claims_a_path_two_sources_cover() -> None:
    """A source rooted inside another one is the more specific answer, not an ambiguity."""
    spec = PersonaSpec(
        id="test-two",
        name="Two Sources",
        role_prompt="x",
        sources=[
            SourceSpec(id="outer", path="", loader="documents", glob="**/*.md"),
            SourceSpec(id="inner", path="notes", loader="documents", glob="**/*.md"),
        ],
    )
    doc_id, source = doc_id_for_file(
        spec, Path("/repo/data/raw/test-two"), Path("/repo/data/raw/test-two/notes/one.md")
    )
    assert (doc_id, source.id) == ("test-two:inner:one", "inner")


def test_a_path_no_source_covers_is_an_error(docs_persona: PersonaSpec) -> None:
    with pytest.raises(SourceNotFoundError):
        doc_id_for_file(docs_persona, Path("/repo/data/raw/test-docs"), Path("/elsewhere/stray.md"))


# ----------------------------------------------------------------------------- disk index


def test_sidecars_are_keyed_on_the_id_inside_them_not_the_directory(tmp_path: Path) -> None:
    write_extraction(tmp_path / "flat.json", "test-pm:test-podcast:ada-north", [VERBATIM])
    write_extraction(tmp_path / "deep" / "nested" / "x.json", "test-pm:test-podcast:ben-oduya", [])
    write_extraction(tmp_path / "other.json", "other-persona:src:doc", [])

    found = index_sidecars(tmp_path, "test-pm")

    assert set(found) == {"test-pm:test-podcast:ada-north", "test-pm:test-podcast:ben-oduya"}


def test_a_missing_sidecar_directory_is_not_a_finding(tmp_path: Path) -> None:
    assert index_sidecars(tmp_path / "nothing-here", "test-pm") == {}
    assert index_sidecars(None, "test-pm") == {}


# ----------------------------------------------------------------------------- the check


def test_a_clean_extraction_file_reports_every_layer_it_has(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, episode_id: str, tmp_path: Path
) -> None:
    write_extraction(tmp_path / "enrichment" / "one.json", episode_id, [VERBATIM])

    report = check_documents(
        memory_store,
        persona,
        enrichment_root=tmp_path / "enrichment",
        doc_ids=[episode_id],
    )

    document = report.documents[0]
    assert document.in_graph and document.has_extraction
    assert document.state("extraction") == "file"  # a dry run writes nothing
    assert document.state("attribution") == "graph"  # the transcript loader parsed its turns
    assert document.state("annotation") == "-"
    assert report.total_loose == 0
    assert report.ok


def test_loose_and_unmatched_entity_names_are_counted_by_reason(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, episode_id: str, tmp_path: Path
) -> None:
    write_extraction(tmp_path / "enrichment" / "one.json", episode_id, [TOKENS, ABSENT])

    report = check_documents(
        memory_store, persona, enrichment_root=tmp_path / "enrichment", doc_ids=[episode_id]
    )

    totals = report.loose_totals
    assert totals["extraction/loose"] == 1
    assert totals["extraction/unmatched"] == 1
    assert not report.ok


def test_a_document_with_no_extraction_json_fails_the_check(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, episode_id: str, tmp_path: Path
) -> None:
    report = check_documents(
        memory_store, persona, enrichment_root=tmp_path / "enrichment", doc_ids=[episode_id]
    )

    assert report.without_extraction == (episode_id,)
    assert not report.ok
    assert "no extraction JSON: 1" in "\n".join(summary_lines(report))


def test_a_sidecar_naming_a_document_the_graph_lacks_is_an_error(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, ingested: IngestReport, tmp_path: Path
) -> None:
    """The case that makes a capture look finished and be empty: JSON for a file never ingested."""
    write_extraction(tmp_path / "enrichment" / "ghost.json", "test-pm:test-podcast:ghost", [])

    report = check_documents(
        memory_store, persona, enrichment_root=tmp_path / "enrichment", doc_ids=None
    )

    ghost = next(d for d in report.documents if d.doc_id.endswith(":ghost"))
    assert not ghost.in_graph
    assert ghost.errors and "unknown document" in ghost.errors[0]
    assert not report.ok


def test_attribution_and_annotation_sidecars_are_dry_run_too(
    memory_store: InMemoryGraphStore,
    docs_persona: PersonaSpec,
    thread_document: str,
    tmp_path: Path,
) -> None:
    write_extraction(tmp_path / "enrichment" / "t.json", thread_document, ["Handbook"])
    write_attribution(
        tmp_path / "attribution" / "t.json",
        thread_document,
        [
            {"speaker": "quill-maker", "anchor": FIRST, "role": "op"},
            {"speaker": "nobody", "anchor": "words that appear in no passage at all here"},
        ],
    )
    write_annotations(
        tmp_path / "annotations" / "t.json",
        thread_document,
        [{"anchor": LAST, "facets": ["access"]}],
    )

    report = check_documents(
        memory_store,
        docs_persona,
        enrichment_root=tmp_path / "enrichment",
        attribution_root=tmp_path / "attribution",
        annotation_root=tmp_path / "annotations",
        doc_ids=[thread_document],
    )

    document = report.documents[0]
    assert set(document.sidecars) == {"extraction", "attribution", "annotation"}
    assert report.loose_totals["attribution/anchor"] == 1
    assert not memory_store.documents[thread_document].speakers  # nothing was written
    assert not report.ok


def test_all_walks_the_persona_and_a_source_narrows_it(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, ingested: IngestReport, tmp_path: Path
) -> None:
    report = check_documents(
        memory_store, persona, enrichment_root=tmp_path / "enrichment", doc_ids=None
    )
    assert len(report.documents) == 3

    narrowed = check_documents(
        memory_store,
        persona,
        enrichment_root=tmp_path / "enrichment",
        doc_ids=None,
        source_id="nothing-like-this",
    )
    assert narrowed.documents == ()


def test_summary_lines_name_each_layer_and_the_loose_reasons(
    memory_store: InMemoryGraphStore, persona: PersonaSpec, episode_id: str, tmp_path: Path
) -> None:
    write_extraction(tmp_path / "enrichment" / "one.json", episode_id, [VERBATIM, ABSENT])

    lines = summary_lines(
        check_documents(
            memory_store, persona, enrichment_root=tmp_path / "enrichment", doc_ids=[episode_id]
        )
    )

    assert lines[0] == "test-pm: 1 documents checked"
    assert any("extraction: 1 sidecars on disk" in line for line in lines)
    assert any("entity name in no passage" in line for line in lines)
    assert lines[-1] == "layers incomplete"


# ----------------------------------------------------------------------------- the command


def test_the_command_exits_1_when_a_document_has_no_extraction(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    result = runner.invoke(
        app, ["layers", "check", "test-pm", "--doc-id", "test-pm:test-podcast:ada-north"]
    )

    assert result.exit_code == 1, result.output
    assert "no extraction JSON" in result.output


def test_the_command_exits_0_when_every_layer_is_complete(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    write_extraction(
        cli_context.settings.enrichment_dir / "one.json",
        "test-pm:test-podcast:ada-north",
        [VERBATIM],
    )

    result = runner.invoke(
        app, ["layers", "check", "test-pm", "--doc-id", "test-pm:test-podcast:ada-north"]
    )

    assert result.exit_code == 0, result.output
    assert "every layer complete" in result.output


def test_the_command_takes_a_raw_file_and_derives_the_id(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    raw = cli_context.settings.raw_dir / "test-pm" / "episodes" / "ada-north" / "transcript.md"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("# a transcript\n", encoding="utf-8")
    write_extraction(
        cli_context.settings.enrichment_dir / "one.json",
        "test-pm:test-podcast:ada-north",
        [VERBATIM],
    )

    result = runner.invoke(app, ["layers", "check", "test-pm", "--file", str(raw)])

    # A derived id that missed would have found no extraction file and exited 1.
    assert result.exit_code == 0, result.output
    assert "test-pm: 1 documents checked" in result.output


def test_the_command_asks_for_a_scope_when_given_none(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["layers", "check", "test-pm"])

    assert result.exit_code == 2
    assert "--doc-id" in result.output


def test_the_command_rejects_a_file_outside_the_personas_sources(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["layers", "check", "test-pm", "--file", "/elsewhere/stray.md"])

    assert result.exit_code == 2
    assert "no source of test-pm" in result.output
