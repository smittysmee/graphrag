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
from graphrag.extract.aliases import load_aliases
from graphrag.extract.annotations import import_annotation_file, load_persona_facets
from graphrag.extract.attributes import (
    EMPTY_ATTRIBUTES,
    AttributeTable,
    AttributeTableError,
    load_attributes,
    load_persona_attributes,
)
from graphrag.extract.attribution import import_attribution_file, report_lines
from graphrag.extract.importer import import_extraction_file
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
  tenure_years:
    type: number
    min: 0
    max: 60
    description: How long the speaker says they have been doing this.
  founded_year:
    type: number
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


def test_a_numeric_key_takes_a_number_and_refuses_anything_else(table: AttributeTable) -> None:
    """Atlas ch. 31 needs a value with a sorting, so the vocabulary has to promise one."""
    spec = table.spec("tenure_years")
    assert spec is not None and spec.numeric and not spec.closed

    assert table.check_attribute("tenure_years", "3.5") is None
    assert table.check_attribute("tenure_years", "0") is None
    assert table.check_attribute("tenure_years", "60") is None

    problem = table.check_attribute("tenure_years", "abc")
    assert problem is not None and "not a number" in problem
    for outside in ("-1", "60.5"):
        refused = table.check_attribute("tenure_years", outside)
        assert refused is not None and "outside the declared range" in refused
        assert "between 0 and 60" in refused
    # nan and inf parse as floats and measure nothing, so they are refused as not numbers.
    for junk in ("nan", "inf"):
        assert table.check_attribute("tenure_years", junk) is not None


def test_a_numeric_key_without_bounds_takes_any_finite_number(table: AttributeTable) -> None:
    assert table.check_attribute("founded_year", "-800") is None
    assert table.check_attribute("founded_year", "1994") is None
    spec = table.spec("founded_year")
    assert spec is not None and spec.minimum is None and spec.maximum is None
    assert spec.bounds == "any finite number"


def test_a_numeric_value_is_stored_as_the_string_the_sidecar_wrote(table: AttributeTable) -> None:
    """Which is why `--where tenure_years=3.5` matches by exact string and 3.50 does not."""
    kept, problems = table.check({"tenure_years": " 3.5 ", "founded_year": "1994"})
    assert kept == {"tenure_years": "3.5", "founded_year": "1994"}
    assert not problems


def test_a_malformed_numeric_declaration_is_an_error(tmp_path: Path) -> None:
    personas = tmp_path / "personas"
    write_vocabulary(personas, "bad-type", "attributes:\n  age:\n    type: integer\n")
    with pytest.raises(AttributeTableError, match="not one of text, number"):
        load_persona_attributes(personas, "bad-type")
    write_vocabulary(
        personas, "both", "attributes:\n  age:\n    type: number\n    values: ['1', '2']\n"
    )
    with pytest.raises(AttributeTableError, match="declare one or the other"):
        load_persona_attributes(personas, "both")
    write_vocabulary(personas, "range-on-text", "attributes:\n  team:\n    min: 1\n")
    with pytest.raises(AttributeTableError, match="range is meaningless"):
        load_persona_attributes(personas, "range-on-text")
    write_vocabulary(personas, "bad-min", "attributes:\n  age:\n    type: number\n    min: soon\n")
    with pytest.raises(AttributeTableError, match="which is not a number"):
        load_persona_attributes(personas, "bad-min")
    write_vocabulary(
        personas, "inverted", "attributes:\n  age:\n    type: number\n    min: 9\n    max: 2\n"
    )
    with pytest.raises(AttributeTableError, match="above max"):
        load_persona_attributes(personas, "inverted")


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


# ----------------------------------------------------------------------------- extraction


def write_extraction(path: Path, doc_id: str, entities: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"doc_id": doc_id, "entities": entities, "relations": []}), encoding="utf-8"
    )
    return path


def test_an_extraction_file_writes_its_entities_attributes_onto_the_entity(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """The point of the whole layer: an entity carries what it is, not what named it.

    ``onboarding`` is mentioned by a document the test never tags, so the value on the node can
    only have come from the extraction file.
    """
    file = write_extraction(
        tmp_path / "thread.json",
        thread_document,
        [
            {"name": "onboarding", "type": "concept", "attributes": {"region": "north"}},
            {"name": "handbook", "type": "concept", "attributes": {"region": "south"}},
        ],
    )

    result = import_extraction_file(memory_store, file, attributes=table)

    assert result.ok and result.attributes == 2
    assert result.attribute_problems == () and result.attribute_conflicts == ()
    assert memory_store.entity_attributes("test-docs") == {
        "concept:onboarding": {"region": "north"},
        "concept:handbook": {"region": "south"},
    }


def test_an_entity_value_the_vocabulary_refuses_is_dropped_and_the_entity_still_lands(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    file = write_extraction(
        tmp_path / "thread.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "west"}}],
    )

    result = import_extraction_file(memory_store, file, attributes=table)

    assert result.entities == 1  # the entity itself is not the casualty
    assert result.attributes == 0
    assert len(result.attribute_problems) == 1
    assert "not one of north, south" in result.attribute_problems[0]
    assert memory_store.entity_attributes("test-docs") == {}


def test_two_documents_that_disagree_about_one_entity_keep_the_first_and_report_the_second(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """The speaker rule, for the reason the speaker rule exists.

    One entity is named by many documents, so many extraction files can claim it. Which of two
    readings is right is a question for whoever wrote them, not for whichever file was imported
    last.
    """
    first = write_extraction(
        tmp_path / "first.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "north"}}],
    )
    second = write_extraction(
        tmp_path / "second.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "south"}}],
    )

    assert import_extraction_file(memory_store, first, attributes=table).attribute_conflicts == ()
    result = import_extraction_file(memory_store, second, attributes=table)

    assert result.attribute_conflicts == (
        "attribute conflict: concept:onboarding region north vs south",
    )
    assert result.attributes == 0
    assert memory_store.entity_attributes("test-docs")["concept:onboarding"] == {"region": "north"}


def test_one_file_that_disagrees_with_itself_reports_it_rather_than_racing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """Two spellings of one thing inside one file fold together, first value winning."""
    file = write_extraction(
        tmp_path / "thread.json",
        thread_document,
        [
            {"name": "onboarding", "type": "concept", "attributes": {"region": "north"}},
            {"name": "Onboarding", "type": "concept", "attributes": {"region": "south"}},
        ],
    )

    result = import_extraction_file(memory_store, file, attributes=table)

    assert memory_store.entity_attributes("test-docs")["concept:onboarding"] == {"region": "north"}
    assert result.attributes == 1


def test_re_importing_the_same_entity_attributes_is_not_a_conflict(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """Re-import has to be a no-op: ``make sync`` re-runs these files after every re-ingest."""
    file = write_extraction(
        tmp_path / "thread.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "north"}}],
    )

    import_extraction_file(memory_store, file, attributes=table)
    again = import_extraction_file(memory_store, file, attributes=table)

    assert again.attribute_conflicts == ()
    assert memory_store.entity_attributes("test-docs") == {
        "concept:onboarding": {"region": "north"}
    }


def test_a_dry_run_reports_the_entity_conflict_it_would_have_hit_and_writes_nothing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    first = write_extraction(
        tmp_path / "first.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "north"}}],
    )
    import_extraction_file(memory_store, first, attributes=table)
    second = write_extraction(
        tmp_path / "second.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "south"}}],
    )

    dry = import_extraction_file(memory_store, second, dry_run=True, attributes=table)

    assert dry.attribute_conflicts == (
        "attribute conflict: concept:onboarding region north vs south",
    )
    assert memory_store.entity_attributes("test-docs")["concept:onboarding"] == {"region": "north"}


def test_an_entity_attribute_follows_its_name_onto_the_canonical_node(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path, table: AttributeTable
) -> None:
    """A claim about an alias is a claim about whatever node that spelling folds onto.

    Otherwise the attribute would land on an id with no mentions on it, and the entity the
    mentions did land on would be the one the analysis reports as untagged.
    """
    alias_file = tmp_path / "aliases.yaml"
    alias_file.write_text("aliases:\n  Onboarding Programme: [onboarding]\n", encoding="utf-8")
    aliases = load_aliases(alias_file)
    file = write_extraction(
        tmp_path / "thread.json",
        thread_document,
        [{"name": "onboarding", "type": "concept", "attributes": {"region": "north"}}],
    )

    result = import_extraction_file(memory_store, file, aliases=aliases, attributes=table)

    assert result.ok
    assert memory_store.entity_attributes("test-docs") == {
        "concept:onboarding-programme": {"region": "north"}
    }
