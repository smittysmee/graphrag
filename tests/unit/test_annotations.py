"""``graphrag annotations-import``: anchoring a reading to a passage, and to a mention on it.

Everything runs against the in-memory store and the sample thread from ``tests/conftest.py``,
which is long enough that a reading has somewhere wrong to land.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag.app import AppContext
from graphrag.extract.aliases import load_aliases
from graphrag.extract.annotations import (
    EntityIndex,
    FacetError,
    import_annotation_file,
    load_facets,
    load_persona_facets,
    loose_summary,
    loose_totals,
    report_lines,
)
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention
from tests.conftest import THREAD_POSTS

# The opening words of the first and last posts, which the chunker puts in different passages.
FIRST = THREAD_POSTS[0][2][:60]
LAST = THREAD_POSTS[-1][2][:60]

FACETS = """
facets:
  handover: Moving work or knowledge from one person to another.
  access: Getting the accounts and permissions a job needs.
"""


def write_annotations(path: Path, doc_id: str, annotations: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"doc_id": doc_id, "annotations": annotations}), encoding="utf-8")
    return path


def annotation(anchor: str, **extra: object) -> dict[str, object]:
    return {"anchor": anchor, **extra}


def write_facets(root: Path, persona_id: str, text: str = FACETS) -> Path:
    folder = root / persona_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "facets.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def annotated_thread(memory_store: InMemoryGraphStore, thread_document: str) -> dict[str, str]:
    """The thread with one entity extracted into its first passage, and one into its last.

    A stance annotates a mention that exists, so the entity layer has to be there first.
    """
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="concept:handbook", name="Handbook", type="concept"),
                Entity(id="concept:checklist", name="Checklist", type="concept"),
            ],
            mentions=[
                Mention(chunk_id=chunks[0].id, entity_id="concept:handbook"),
                Mention(chunk_id=chunks[-1].id, entity_id="concept:checklist"),
            ],
        )
    )
    return {"first": chunks[0].id, "last": chunks[-1].id}


# ----------------------------------------------------------------------------- facets


def test_a_facet_file_declares_the_vocabulary_and_an_absent_one_declares_nothing(
    tmp_path: Path,
) -> None:
    """An empty table and no table are different: no table means anything is accepted."""
    table = load_facets(write_facets(tmp_path, "test-docs"))

    assert table.known("handover") and table.known("access")
    assert not table.known("invented")
    assert load_persona_facets(tmp_path, "test-docs") is not None
    assert load_persona_facets(tmp_path, "no-such-persona") is None


def test_a_malformed_facet_file_is_named_rather_than_guessed_at(tmp_path: Path) -> None:
    with pytest.raises(FacetError, match="expected a mapping"):
        load_facets(write_facets(tmp_path, "test-docs", "- a\n- list\n"))
    with pytest.raises(FacetError, match="must be a mapping"):
        load_facets(write_facets(tmp_path, "test-docs", "facets: [a, b]\n"))
    with pytest.raises(FacetError, match="cannot read"):
        load_facets(tmp_path / "absent.yaml")


# ----------------------------------------------------------------------------- importing


def test_a_stance_lands_on_the_mention_in_the_passage_the_anchor_is_in(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation(FIRST, entity="Handbook", stance="complaint", facets=["handover"]),
            annotation(LAST, entity="Checklist", stance="praise", facets=["access", "handover"]),
        ],
    )

    result = import_annotation_file(memory_store, file)

    assert result.ok
    assert (result.annotations, result.applied, result.stances, result.facets) == (2, 2, 2, 3)
    assert result.entities == ("Handbook", "Checklist")
    stances = memory_store.mention_stances("test-docs")
    assert [(s.name, s.stance, s.chunk_id) for s in stances] == [
        ("Checklist", "praise", annotated_thread["last"]),
        ("Handbook", "complaint", annotated_thread["first"]),
    ]
    facets = memory_store.chunk_facets("test-docs")
    assert {f.chunk_id: f.facets for f in facets} == {
        annotated_thread["first"]: ["handover"],
        annotated_thread["last"]: ["access", "handover"],
    }
    assert memory_store.annotated_document_ids("test-docs", "threads") == {thread_document}
    assert memory_store.annotated_document_ids("test-docs", "docs") == set()
    assert memory_store.annotated_document_ids("other-persona", "threads") == set()


def test_a_stance_carries_the_speakers_who_wrote_the_passage(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """So a signed network can say who praised a thing without a second read."""
    memory_store.attach_speaker(thread_document, annotated_thread["first"], "quill-maker")
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Handbook", stance="complaint")],
    )

    import_annotation_file(memory_store, file)

    assert memory_store.mention_stances("test-docs")[0].speakers == ["quill-maker"]


def test_an_anchor_in_no_passage_costs_only_its_own_annotation(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation("words this thread does not contain anywhere", facets=["access"]),
            annotation(FIRST, entity="Handbook", stance="praise"),
        ],
    )

    result = import_annotation_file(memory_store, file)

    assert result.ok  # a loose anchor is a report, not a failure
    assert (result.applied, result.stances) == (1, 1)
    assert [(item.reason, item.detail) for item in result.loose] == [
        ("anchor", "words this thread does not contain anywhere")
    ]
    assert result.loose_by_reason == {"anchor": 1, "unknown-entity": 0, "entity-absent": 0}
    assert memory_store.chunk_facets("test-docs") == []


def test_an_entity_whose_name_is_nowhere_in_the_passage_is_loose_not_guessed_at(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """The fallback needs the passage as evidence; the annotation asserting it is not enough."""
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation(FIRST, entity="Checklist", stance="praise", facets=["access"]),
            annotation(LAST, entity="Checklist", stance="praise"),
        ],
    )

    result = import_annotation_file(memory_store, file)

    assert result.ok and (result.applied, result.stances, result.created) == (1, 1, 0)
    assert [item.reason for item in result.loose] == ["entity-absent"]
    assert result.loose[0].detail.startswith("Checklist at ")
    assert str(result.loose[0]).startswith("entity not in the passage: Checklist at ")
    # the whole annotation was skipped, so its facets did not land either
    assert [f.chunk_id for f in memory_store.chunk_facets("test-docs")] == []
    assert [s.chunk_id for s in memory_store.mention_stances("test-docs")] == [
        annotated_thread["last"]
    ]


def test_an_entity_the_persona_does_not_have_at_all_stays_loose(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """Creating entities is the extraction pass's job, so this reason means extend that."""
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Pairing", stance="praise")],
    )

    result = import_annotation_file(memory_store, file)

    assert result.ok and (result.applied, result.created) == (0, 0)
    assert [item.reason for item in result.loose] == ["unknown-entity"]
    assert str(result.loose[0]).startswith("entity unknown to the persona: Pairing at ")
    assert memory_store.mention_stances("test-docs") == []


def test_an_entity_the_persona_has_and_the_passage_names_gets_its_mention_created(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """Extraction and annotation read a document separately, so one can name what the other did
    not list in that passage. The reading is still true, and the passage is the evidence."""
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="concept:onboarding", name="Onboarding", type="concept")],
            mentions=[Mention(chunk_id=chunks[-1].id, entity_id="concept:onboarding")],
        )
    )
    assert "onboarding" in chunks[0].text.lower()
    assert "concept:onboarding" not in {
        e.id for e in memory_store.entities_for_chunks([chunks[0].id])
    }
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Onboarding", stance="complaint", facets=["handover"])],
    )

    result = import_annotation_file(memory_store, file)

    assert result.ok and result.loose == ()
    assert (result.applied, result.stances, result.created) == (1, 1, 1)
    assert result.entities == ("Onboarding",)
    stance = next(s for s in memory_store.mention_stances("test-docs") if s.name == "Onboarding")
    assert (stance.chunk_id, stance.stance) == (annotated_thread["first"], "complaint")


def test_the_fallback_matches_a_spelling_recorded_on_the_entity(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """An alias pass folded the spelling onto the node, so the passage still names the thing."""
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(
                    id="concept:handover",
                    name="Handover",
                    type="concept",
                    aliases=["handbook"],
                )
            ],
            mentions=[Mention(chunk_id=chunks[-1].id, entity_id="concept:handover")],
        )
    )
    assert "handover" not in chunks[0].text.lower() and "handbook" in chunks[0].text.lower()
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Handover", stance="praise")],
    )

    result = import_annotation_file(memory_store, file)

    assert (result.created, result.stances, result.loose) == (1, 1, ())
    stance = memory_store.mention_stances("test-docs")[0]
    assert (stance.name, stance.chunk_id) == ("Handover", annotated_thread["first"])


def test_the_fallback_writes_nothing_on_a_dry_run(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="concept:onboarding", name="Onboarding", type="concept")],
            mentions=[Mention(chunk_id=chunks[-1].id, entity_id="concept:onboarding")],
        )
    )
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Onboarding", stance="complaint")],
    )

    result = import_annotation_file(memory_store, file, dry_run=True)

    assert result.created == 1  # reported, so a reviewer sees it coming
    assert memory_store.mention_stances("test-docs") == []
    assert "concept:onboarding" not in {
        e.id for e in memory_store.entities_for_chunks([chunks[0].id])
    }


def test_a_prebuilt_index_and_a_lazy_one_agree(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """The CLI builds the index once for a whole run; a single-file call builds its own."""
    index = EntityIndex.build(memory_store, "test-docs")

    assert index.find("handbook") is not None
    assert index.find("HAND  BOOK") is None
    assert index.find("concept:checklist") is not None
    assert index.find("nothing of the sort") is None

    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Checklist", stance="praise")],
    )
    passed = import_annotation_file(memory_store, file, entities=index, dry_run=True)
    lazy = import_annotation_file(memory_store, file, dry_run=True)

    assert [item.reason for item in passed.loose] == [item.reason for item in lazy.loose]


def test_the_reasons_are_counted_across_a_run_in_a_fixed_order(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """The three send a reviewer to different files, so the summary has to tell them apart."""
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation("not a sentence this thread holds", facets=["x"]),
            annotation(FIRST, entity="Pairing", stance="praise"),
            annotation(FIRST, entity="Checklist", stance="praise"),
            annotation(LAST, entity="Checklist", stance="praise"),
        ],
    )

    totals = loose_totals([import_annotation_file(memory_store, file, dry_run=True)])

    assert totals == {"anchor": 1, "unknown-entity": 1, "entity-absent": 1}
    assert loose_summary(totals) == (
        "1 anchor not found, 1 entity unknown to the persona, 1 entity not in the passage"
    )
    assert loose_summary({"anchor": 0, "unknown-entity": 2, "entity-absent": 0}) == (
        "2 entity unknown to the persona"
    )


def test_a_facet_outside_the_vocabulary_costs_the_slug_not_the_file(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    table = load_facets(write_facets(tmp_path, "test-docs"))
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, facets=["access", "invented", "Hand Over"])],
    )

    result = import_annotation_file(memory_store, file, facets=table)

    assert result.ok
    assert result.unknown_facets == ("invented", "hand-over")
    assert memory_store.chunk_facets("test-docs")[0].facets == ["access"]


def test_with_no_vocabulary_any_facet_is_accepted(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    file = write_annotations(
        tmp_path / "thread.json", thread_document, [annotation(FIRST, facets=["Whatever It Is"])]
    )

    result = import_annotation_file(memory_store, file, facets=None)

    assert result.unknown_facets == ()
    assert memory_store.chunk_facets("test-docs")[0].facets == ["whatever-it-is"]


def test_an_alias_spelling_in_an_annotation_finds_the_canonical_node(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """An annotation names the entity the way its passage does; the table does the rest."""
    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  Handbook: [the handbook, hand book]\n", encoding="utf-8")
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="the handbook", stance="complaint")],
    )

    result = import_annotation_file(memory_store, file, aliases=load_aliases(path))

    assert result.stances == 1 and result.loose == ()
    assert memory_store.mention_stances("test-docs")[0].name == "Handbook"


def test_re_importing_the_same_file_changes_nothing(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    """`make sync` re-imports these files, so a second pass must not duplicate a facet."""
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [annotation(FIRST, entity="Handbook", stance="praise", facets=["handover", "handover"])],
    )

    first = import_annotation_file(memory_store, file)
    before = dict(memory_store.chunks)
    second = import_annotation_file(memory_store, file)

    assert first.facets == second.facets == 1  # the repeat inside the file counted once
    assert memory_store.chunks == before
    assert memory_store.chunk_facets("test-docs")[0].facets == ["handover"]
    assert len(memory_store.mention_stances("test-docs")) == 1


def test_a_dry_run_reports_without_writing(
    memory_store: InMemoryGraphStore,
    thread_document: str,
    annotated_thread: dict[str, str],
    tmp_path: Path,
) -> None:
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation(FIRST, entity="Handbook", stance="praise", facets=["handover"]),
            annotation("nowhere in the thread at all", facets=["access"]),
        ],
    )

    result = import_annotation_file(memory_store, file, dry_run=True)

    assert (result.applied, result.stances, result.facets, len(result.loose)) == (1, 1, 1, 1)
    assert memory_store.mention_stances("test-docs") == []
    assert memory_store.chunk_facets("test-docs") == []


def test_bad_input_is_reported_per_file_rather_than_raised(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """One malformed file must not stop a run over hundreds of them."""
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert import_annotation_file(memory_store, broken).error.startswith("invalid:")

    blank = write_annotations(tmp_path / "blank.json", thread_document, [annotation("  ")])
    assert import_annotation_file(memory_store, blank).error.startswith("invalid:")

    stanceless = write_annotations(
        tmp_path / "stanceless.json", thread_document, [annotation(FIRST, stance="praise")]
    )
    result = import_annotation_file(memory_store, stanceless)
    assert "stance requires an entity" in result.error

    invented = write_annotations(
        tmp_path / "invented.json",
        thread_document,
        [annotation(FIRST, entity="Handbook", stance="delighted")],
    )
    assert import_annotation_file(memory_store, invented).error.startswith("invalid:")

    gone = write_annotations(tmp_path / "gone.json", "test-docs:threads:missing", [])
    assert "unknown document" in import_annotation_file(memory_store, gone).error

    assert import_annotation_file(memory_store, tmp_path / "absent.json").error.startswith(
        "invalid:"
    )


def test_a_file_with_no_annotations_is_valid_and_does_nothing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    file = write_annotations(tmp_path / "empty.json", thread_document, [])

    result = import_annotation_file(memory_store, file)

    assert result.ok and (result.annotations, result.applied) == (0, 0)
    assert memory_store.annotated_document_ids("test-docs", "threads") == set()


# ----------------------------------------------------------------------------- the command


def test_the_cli_reports_a_dry_run_before_it_writes(
    cli_context: AppContext, thread_document: str, annotated_thread: dict[str, str], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from graphrag.cli import app

    runner = CliRunner()
    write_facets(cli_context.settings.personas_dir, "test-docs")
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation(FIRST, entity="Handbook", stance="complaint", facets=["handover"]),
            annotation(LAST, entity="Checklist", stance="praise", facets=["invented"]),
            annotation("not a line this thread contains", facets=["access"]),
        ],
    )

    dry = runner.invoke(app, ["annotations-import", "test-docs", str(file), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "2/3 annotations" in dry.stdout
    assert "1 loose" in dry.stdout
    assert "1 unknown facets" in dry.stdout
    assert "loose: anchor not found: not a line this thread" in dry.output
    assert "1 loose (1 anchor not found)" in dry.stdout
    assert "0 mentions created" in dry.stdout
    assert cli_context.store.mention_stances("test-docs") == []

    done = runner.invoke(app, ["annotations-import", "test-docs", str(file)])
    assert done.exit_code == 0, done.output
    assert "2 stances" in done.stdout
    assert len(cli_context.store.mention_stances("test-docs")) == 2
    assert (cli_context.settings.snapshots_dir / "test-docs" / "manifest.json").exists()

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"doc_id": "nope", "annotations": []}))
    assert runner.invoke(app, ["annotations-import", "test-docs", str(missing)]).exit_code == 2


def test_each_files_loose_lines_print_under_its_own_file(
    cli_context: AppContext, thread_document: str, annotated_thread: dict[str, str], tmp_path: Path
) -> None:
    """A loose line used to go to stderr while its file's line went to stdout.

    Nothing orders one stream against the other, so with several files the first loose line of
    one could surface under the line of the file before it, and a reviewer would open the wrong
    JSON. One block on one stream is the fix, so the two files here are asserted in order.
    """
    from typer.testing import CliRunner

    from graphrag.cli import app

    clean = write_annotations(
        tmp_path / "clean.json",
        thread_document,
        [annotation(FIRST, entity="Handbook", stance="praise")],
    )
    loose = write_annotations(
        tmp_path / "loose.json",
        thread_document,
        [annotation("no sentence of this thread reads like this", facets=["access"])],
    )

    for argv in (
        ["annotations-import", "test-docs", str(clean), str(loose), "--dry-run"],
        ["annotations-import", "test-docs", str(clean), str(loose), "--no-export"],
    ):
        result = CliRunner().invoke(app, argv)

        assert result.exit_code == 0, result.output
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines[0] == f"{thread_document}: 1/1 annotations, 1 stances, 0 facets"
        assert lines[1] == f"{thread_document}: 0/1 annotations, 0 stances, 0 facets; 1 loose"
        assert lines[2] == "  loose: anchor not found: no sentence of this thread reads like this"


def test_a_files_report_is_one_block_a_reader_can_attribute(
    cli_context: AppContext, thread_document: str, tmp_path: Path
) -> None:
    """The rendering on its own: pure, so the ordering is asserted without a terminal."""
    result = import_annotation_file(
        cli_context.store,
        write_annotations(
            tmp_path / "one.json",
            thread_document,
            [
                annotation(FIRST, facets=["handover"]),
                annotation("nothing in the thread says this", facets=["handover"]),
            ],
        ),
        dry_run=True,
        facets=load_facets(write_facets(tmp_path / "personas", "test-docs")),
    )

    assert report_lines(result) == [
        f"{thread_document}: 1/2 annotations, 0 stances, 1 facets; 1 loose",
        "  loose: anchor not found: nothing in the thread says this",
    ]


def test_the_summary_line_breaks_the_loose_count_into_its_three_reasons(
    cli_context: AppContext, thread_document: str, annotated_thread: dict[str, str], tmp_path: Path
) -> None:
    """A reviewer reads this line to know whether to fix anchors, the alias file or extraction."""
    from typer.testing import CliRunner

    from graphrag.cli import app

    chunks = cli_context.store.document_chunks(thread_document, 0, 100)
    cli_context.store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="concept:onboarding", name="Onboarding", type="concept")],
            mentions=[Mention(chunk_id=chunks[-1].id, entity_id="concept:onboarding")],
        )
    )
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        [
            annotation(FIRST, entity="Onboarding", stance="complaint"),
            annotation("no sentence of this thread reads like this", facets=["access"]),
            annotation(FIRST, entity="Pairing", stance="praise"),
            annotation(FIRST, entity="Checklist", stance="praise"),
        ],
    )

    result = CliRunner().invoke(app, ["annotations-import", "test-docs", str(file)])

    assert result.exit_code == 0, result.output
    assert (
        "imported 1 files: 1 annotations, 1 stances, 0 facets, 1 mentions created, 3 loose "
        "(1 anchor not found, 1 entity unknown to the persona, 1 entity not in the passage)"
    ) in result.stdout
    assert "1 mentions created" in result.stdout
    assert {s.name for s in cli_context.store.mention_stances("test-docs")} == {"Onboarding"}
