"""Node attributes: the vocabulary, the two importers that write them, and what the stores hold.

The attribute keys and values here are invented for the tests (``region: north | south``, a
couple of free-text ones), for the same reason the rest of the suite invents its corpus: the
rules being checked are about the mechanism, not about any corpus that happens to use it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.extract.annotations import import_annotation_file, load_persona_facets
from graphrag.extract.attributes import (
    EMPTY_ATTRIBUTES,
    AttributeTable,
    AttributeTableError,
    load_attributes,
    load_persona_attributes,
)
from graphrag.extract.attribution import import_attribution_file, report_lines
from graphrag.graph.memory_store import InMemoryGraphStore
from tests.conftest import THREAD_POSTS

FIRST = THREAD_POSTS[0][2][:60]
SECOND = THREAD_POSTS[1][2][:60]
LAST = THREAD_POSTS[-1][2][:60]

VOCABULARY = """
facets:
  handover: Moving work or knowledge from one person to another.
attributes:
  region:
    values: [north, south]
    description: Which half of the network this belongs to.
  team:
    description: The team a speaker says they are on, as written.
"""


def write_vocabulary(root: Path, persona_id: str, text: str = VOCABULARY) -> Path:
    folder = root / persona_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "facets.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_attribution(path: Path, doc_id: str, posts: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"doc_id": doc_id, "posts": posts}), encoding="utf-8")
    return path


def write_annotations(path: Path, doc_id: str, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps({"doc_id": doc_id, **payload}), encoding="utf-8")
    return path


@pytest.fixture
def table(tmp_path: Path) -> AttributeTable:
    return load_attributes(write_vocabulary(tmp_path / "personas", "test-docs"))


# ----------------------------------------------------------------------------- the vocabulary


def test_a_closed_key_rejects_a_value_it_does_not_declare(table: AttributeTable) -> None:
    assert table.check_attribute("region", "north") is None
    problem = table.check_attribute("region", "west")
    assert problem is not None
    assert "not one of north, south" in problem


def test_a_key_without_values_takes_any_text(table: AttributeTable) -> None:
    assert table.check_attribute("team", "The Pairing Crew") is None
    assert table.check_attribute("team", "  ") == "team: blank value"


def test_a_key_the_file_does_not_declare_is_refused(table: AttributeTable) -> None:
    problem = table.check_attribute("colour", "green")
    assert problem is not None
    assert "not declared in facets.yaml" in problem


def test_a_key_that_is_not_a_slug_is_refused_whatever_the_file_says(table: AttributeTable) -> None:
    for key in ("Region", "two words", "1st"):
        problem = table.check_attribute(key, "north")
        assert problem is not None
        assert "not a usable attribute key" in problem


def test_an_undeclared_vocabulary_accepts_anything() -> None:
    assert not EMPTY_ATTRIBUTES
    assert EMPTY_ATTRIBUTES.check_attribute("anything", "at all") is None
    assert EMPTY_ATTRIBUTES.check_attribute("anything", "") == "anything: blank value"


def test_check_splits_a_sidecar_into_what_to_write_and_what_to_report(
    table: AttributeTable,
) -> None:
    kept, problems = table.check({"region": "north", "team": " Blue ", "colour": "green"})
    assert kept == {"region": "north", "team": "Blue"}
    assert len(problems) == 1 and "colour" in problems[0]


def test_a_persona_with_no_file_and_one_with_no_section_both_accept_anything(
    tmp_path: Path,
) -> None:
    personas = tmp_path / "personas"
    assert load_persona_attributes(personas, "absent") is EMPTY_ATTRIBUTES
    write_vocabulary(personas, "facets-only", "facets:\n  handover: Moving work.\n")
    assert not load_persona_attributes(personas, "facets-only")
    # The same file still answers for facets, which is the point of keeping them together.
    facets = load_persona_facets(personas, "facets-only")
    assert facets is not None and facets.known("handover")


def test_a_malformed_vocabulary_is_an_error_rather_than_an_empty_table(tmp_path: Path) -> None:
    personas = tmp_path / "personas"
    write_vocabulary(personas, "bad-list", "attributes:\n  region: [north, south]\n")
    with pytest.raises(AttributeTableError, match="must be a mapping or a description string"):
        load_persona_attributes(personas, "bad-list")
    write_vocabulary(personas, "bad-values", "attributes:\n  region:\n    values: north\n")
    with pytest.raises(AttributeTableError, match="must declare 'values' as a list"):
        load_persona_attributes(personas, "bad-values")
    write_vocabulary(personas, "bad-key", "attributes:\n  Region:\n    values: [north]\n")
    with pytest.raises(AttributeTableError, match="must be lower case"):
        load_persona_attributes(personas, "bad-key")
    write_vocabulary(personas, "empty-values", "attributes:\n  region:\n    values: []\n")
    with pytest.raises(AttributeTableError, match="empty 'values' list"):
        load_persona_attributes(personas, "empty-values")


# ----------------------------------------------------------------------------- attribution


def test_a_post_writes_its_speakers_attributes_onto_the_speaker(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [
            {"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}},
            {"speaker": "ledger-ann", "anchor": LAST, "attributes": {"region": "south"}},
        ],
    )

    result = import_attribution_file(memory_store, file, attributes=table)

    assert result.ok and result.attributes == 2
    assert result.attribute_problems == () and result.conflicts == ()
    assert memory_store.speaker_attributes("test-docs") == {
        "quill-maker": {"region": "north"},
        "ledger-ann": {"region": "south"},
    }


def test_a_value_the_vocabulary_refuses_is_dropped_and_the_post_still_lands(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "west"}}],
    )

    result = import_attribution_file(memory_store, file, attributes=table)

    assert result.attached == 1  # the post itself is not the casualty
    assert result.attributes == 0
    assert len(result.attribute_problems) == 1
    assert "not one of north, south" in result.attribute_problems[0]
    assert memory_store.speaker_attributes("test-docs") == {}


def test_two_posts_that_disagree_keep_the_first_value_and_report_the_second(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [
            {"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}},
            {"speaker": "quill-maker", "anchor": SECOND, "attributes": {"region": "south"}},
        ],
    )

    result = import_attribution_file(memory_store, file, attributes=table)

    assert result.conflicts == ("attribute conflict: quill-maker region north vs south",)
    assert memory_store.speaker_attributes("test-docs")["quill-maker"] == {"region": "north"}
    assert "attribute conflict: quill-maker region north vs south" in "\n".join(
        report_lines(result)
    )


def test_a_second_file_that_disagrees_with_the_stored_value_reports_it_too(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    first = write_attribution(
        tmp_path / "one.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}}],
    )
    import_attribution_file(memory_store, first, attributes=table)
    second = write_attribution(
        tmp_path / "two.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "south"}}],
    )

    result = import_attribution_file(memory_store, second, attributes=table)

    assert result.conflicts == ("attribute conflict: quill-maker region north vs south",)
    assert memory_store.speaker_attributes("test-docs")["quill-maker"] == {"region": "north"}


def test_re_importing_the_same_attributes_is_not_a_conflict(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}}],
    )
    import_attribution_file(memory_store, file, attributes=table)

    again = import_attribution_file(memory_store, file, attributes=table)

    assert again.conflicts == () and again.attributes == 1


def test_a_dry_run_reports_the_conflict_it_would_have_hit_and_writes_nothing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    held = write_attribution(
        tmp_path / "one.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}}],
    )
    import_attribution_file(memory_store, held, attributes=table)
    incoming = write_attribution(
        tmp_path / "two.json",
        thread_document,
        [
            {"speaker": "ledger-ann", "anchor": LAST, "attributes": {"region": "south"}},
            {"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "south"}},
        ],
    )

    result = import_attribution_file(memory_store, incoming, dry_run=True, attributes=table)

    assert result.conflicts == ("attribute conflict: quill-maker region north vs south",)
    assert result.attributes == 1  # ledger-ann's, which a real run would have written
    assert "ledger-ann" not in memory_store.speaker_attributes("test-docs")


def test_attributes_on_a_post_whose_anchor_is_loose_are_not_written(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """A post nobody can place is not evidence of anything, including of who its speaker is."""
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [
            {
                "speaker": "quill-maker",
                "anchor": "no sentence of this thread reads anything like this one",
                "attributes": {"region": "north"},
            }
        ],
    )

    result = import_attribution_file(memory_store, file, attributes=table)

    assert len(result.loose) == 1 and result.attributes == 0
    assert memory_store.speaker_attributes("test-docs") == {}


# ----------------------------------------------------------------------------- annotation


def test_an_annotation_file_sets_the_documents_attributes(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        {"attributes": {"region": "south"}, "annotations": []},
    )

    result = import_annotation_file(memory_store, file, attributes=table)

    assert result.ok and result.attributes == 1
    assert memory_store.documents[thread_document].attributes == {"region": "south"}
    assert memory_store.document_attributes("test-docs") == {thread_document: {"region": "south"}}


def test_a_file_can_carry_attributes_and_no_annotations_at_all(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """A document nobody has a reading of still belongs to a part of the corpus."""
    file = write_annotations(
        tmp_path / "thread.json", thread_document, {"attributes": {"region": "north"}}
    )

    result = import_annotation_file(memory_store, file, attributes=table)

    assert result.annotations == 0 and result.attributes == 1
    assert memory_store.documents[thread_document].attributes == {"region": "north"}


def test_an_invalid_document_attribute_is_reported_and_dropped(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_annotations(
        tmp_path / "thread.json",
        thread_document,
        {"attributes": {"region": "west", "colour": "green"}},
    )

    result = import_annotation_file(memory_store, file, attributes=table)

    assert result.attributes == 0
    assert len(result.attribute_problems) == 2
    assert memory_store.documents[thread_document].attributes == {}


def test_a_dry_run_of_an_annotation_file_writes_no_attributes(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_annotations(
        tmp_path / "thread.json", thread_document, {"attributes": {"region": "north"}}
    )

    result = import_annotation_file(memory_store, file, dry_run=True, attributes=table)

    assert result.attributes == 1
    assert memory_store.documents[thread_document].attributes == {}


def test_a_second_import_merges_keys_rather_than_replacing_the_map(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    one = write_annotations(
        tmp_path / "one.json", thread_document, {"attributes": {"region": "north"}}
    )
    import_annotation_file(memory_store, one)
    two = write_annotations(
        tmp_path / "two.json", thread_document, {"attributes": {"team": "Blue"}}
    )

    import_annotation_file(memory_store, two)

    assert memory_store.documents[thread_document].attributes == {
        "region": "north",
        "team": "Blue",
    }


# ----------------------------------------------------------------------------- the commands


def test_the_import_commands_read_the_personas_vocabulary_and_report_what_it_refused(
    cli_context: AppContext, thread_document: str, tmp_path: Path
) -> None:
    write_vocabulary(cli_context.registry.directory, "test-docs")
    posts = write_attribution(
        tmp_path / "posts.json",
        thread_document,
        [
            {"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}},
            {"speaker": "ledger-ann", "anchor": LAST, "attributes": {"region": "west"}},
        ],
    )
    readings = write_annotations(
        tmp_path / "readings.json", thread_document, {"attributes": {"region": "south"}}
    )
    runner = CliRunner()

    attributed = runner.invoke(app, ["attribution-import", "test-docs", str(posts), "--no-export"])
    annotated = runner.invoke(
        app, ["annotations-import", "test-docs", str(readings), "--no-export"]
    )

    assert attributed.exit_code == 0, attributed.output
    assert "1 speaker attributes" in attributed.stdout
    assert "1 attribute problems" in attributed.stdout
    assert "not one of north, south" in attributed.stdout
    assert annotated.exit_code == 0, annotated.output
    assert "1 document attributes" in annotated.stdout
    store = cli_context.store
    assert store.speaker_attributes("test-docs") == {"quill-maker": {"region": "north"}}
    assert store.documents[thread_document].attributes == {"region": "south"}


def test_a_dry_run_of_either_command_writes_nothing(
    cli_context: AppContext, thread_document: str, tmp_path: Path
) -> None:
    write_vocabulary(cli_context.registry.directory, "test-docs")
    posts = write_attribution(
        tmp_path / "posts.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}}],
    )
    runner = CliRunner()

    result = runner.invoke(app, ["attribution-import", "test-docs", str(posts), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "validated" in result.stdout
    assert cli_context.store.speaker_attributes("test-docs") == {}


def test_layers_check_reports_an_out_of_vocabulary_value_as_loose(
    cli_context: AppContext, thread_document: str
) -> None:
    """The gate a tagging pass runs into: a value the persona never declared, named as such."""
    write_vocabulary(cli_context.registry.directory, "test-docs")
    settings = cli_context.settings
    enrichment = settings.enrichment_dir / "test-docs"
    enrichment.mkdir(parents=True, exist_ok=True)
    (enrichment / "thread.json").write_text(
        json.dumps(
            {
                "doc_id": thread_document,
                "entities": [
                    {"name": "Handbook", "type": "concept", "description": "The written one."}
                ],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )
    posts = settings.attribution_dir / "test-docs"
    posts.mkdir(parents=True, exist_ok=True)
    write_attribution(
        posts / "thread.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "west"}}],
    )
    readings = settings.annotations_dir / "test-docs"
    readings.mkdir(parents=True, exist_ok=True)
    write_annotations(
        readings / "thread.json", thread_document, {"attributes": {"colour": "green"}}
    )

    result = CliRunner().invoke(app, ["layers", "check", "test-docs", "--doc-id", thread_document])

    assert result.exit_code == 1, result.output
    assert "1 speaker attribute invalid" in result.stdout
    assert "1 document attribute invalid" in result.stdout
    assert "layers incomplete" in result.stdout
    # A dry run: the refused values reached neither node.
    assert cli_context.store.speaker_attributes("test-docs") == {}
    assert cli_context.store.documents[thread_document].attributes == {}


def test_two_files_that_disagree_about_one_speaker_print_the_conflict(
    cli_context: AppContext, thread_document: str, tmp_path: Path
) -> None:
    """The case a real corpus produces: one handle tagged one way here and another way there."""
    write_vocabulary(cli_context.registry.directory, "test-docs")
    first = write_attribution(
        tmp_path / "one.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": FIRST, "attributes": {"region": "north"}}],
    )
    second = write_attribution(
        tmp_path / "two.json",
        thread_document,
        [{"speaker": "quill-maker", "anchor": SECOND, "attributes": {"region": "south"}}],
    )
    runner = CliRunner()

    kept = runner.invoke(app, ["attribution-import", "test-docs", str(first), "--no-export"])
    clash = runner.invoke(app, ["attribution-import", "test-docs", str(second), "--no-export"])

    assert kept.exit_code == 0, kept.output
    assert "1 speaker attributes" in kept.stdout
    assert clash.exit_code == 0, clash.output
    # The per-file line, then the conflict under it, naming both values in the order they arrived.
    assert "attribute conflict: quill-maker region north vs south" in clash.stdout
    assert "1 attribute conflicts" in clash.stdout
    assert cli_context.store.speaker_attributes("test-docs")["quill-maker"] == {"region": "north"}
