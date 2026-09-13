"""The alias layer: reading a persona's table, folding an extraction, folding the graph.

Everything runs against the in-memory store, the sample transcripts and a table written in the
test, so nothing here depends on what any real corpus happens to be spelled like.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag.app import AppContext
from graphrag.extract.aliases import (
    AliasError,
    alias_lines,
    apply_alias_table,
    apply_aliases,
    load_aliases,
    load_persona_aliases,
    preview_alias_table,
)
from graphrag.extract.importer import import_extraction_file
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention, Relation
from graphrag.pipeline import IngestReport

TABLE = """
aliases:
  Northwind Ledger: [Northwind, north wind ledger, "Ledger, Northwind"]
  Quill Editor: [quill]
"""


def write_table(root: Path, persona_id: str, text: str = TABLE) -> Path:
    folder = root / persona_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "aliases.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- loading


def test_a_table_maps_every_spelling_to_its_canonical(tmp_path: Path) -> None:
    table = load_aliases(write_table(tmp_path, "test-pm"))

    assert bool(table)
    assert table.canonical("Northwind") == "Northwind Ledger"
    assert table.canonical("NORTH  WIND   LEDGER") == "Northwind Ledger"  # case and whitespace
    assert table.canonical("Ledger, Northwind") == "Northwind Ledger"
    assert table.canonical("Northwind Ledger") == "Northwind Ledger"  # a canonical maps to itself
    assert table.canonical("Something Else") == "Something Else"  # silence leaves a name alone
    assert table.aliases_for("Northwind Ledger") == (
        "Northwind",
        "north wind ledger",
        "Ledger, Northwind",
    )
    assert table.aliases_for("Not A Canonical") == ()


def test_one_alias_under_two_canonicals_is_a_contradiction_not_a_race(tmp_path: Path) -> None:
    """Resolving it by file order would make the graph depend on how the YAML was sorted."""
    path = write_table(
        tmp_path, "test-pm", "aliases:\n  First Thing: [shared]\n  Second Thing: [Shared]\n"
    )

    with pytest.raises(AliasError, match="claimed by both"):
        load_aliases(path)


def test_a_name_cannot_be_a_canonical_and_someone_elses_alias(tmp_path: Path) -> None:
    path = write_table(
        tmp_path, "test-pm", "aliases:\n  First Thing: [Second Thing]\n  Second Thing: [other]\n"
    )

    with pytest.raises(AliasError, match="both a canonical name and an alias"):
        load_aliases(path)


def test_malformed_tables_are_named_rather_than_guessed_at(tmp_path: Path) -> None:
    blank = write_table(tmp_path, "test-pm", "aliases:\n  Thing: ['']\n")
    with pytest.raises(AliasError, match="blank alias"):
        load_aliases(blank)

    scalar = write_table(tmp_path, "test-pm", "aliases:\n  Thing: not-a-list\n")
    with pytest.raises(AliasError, match="must be a list"):
        load_aliases(scalar)

    wrong = write_table(tmp_path, "test-pm", "- just\n- a list\n")
    with pytest.raises(AliasError, match="expected a mapping"):
        load_aliases(wrong)

    with pytest.raises(AliasError, match="cannot read"):
        load_aliases(tmp_path / "absent.yaml")


def test_an_empty_or_absent_table_is_not_a_finding(tmp_path: Path) -> None:
    """Most personas keep no alias file, and that has to read as normal rather than broken."""
    assert not load_aliases(write_table(tmp_path, "test-pm", "aliases: {}\n"))
    assert not load_aliases(write_table(tmp_path, "test-pm", "\n"))
    assert not load_persona_aliases(tmp_path, "no-such-persona")
    assert load_persona_aliases(tmp_path, "test-pm").canonical("Northwind") == "Northwind"


# ----------------------------------------------------------------------------- folding a file


def test_folding_an_extraction_rewrites_ids_names_mentions_and_relations(tmp_path: Path) -> None:
    table = load_aliases(write_table(tmp_path, "test-pm"))
    enrichment = Enrichment(
        entities=[
            Entity(id="product:northwind", name="Northwind", type="product", description="A tool."),
            Entity(id="product:northwind-ledger", name="Northwind Ledger", type="product"),
            Entity(id="concept:pricing", name="Pricing"),
        ],
        mentions=[
            Mention(chunk_id="c1", entity_id="product:northwind"),
            Mention(chunk_id="c1", entity_id="product:northwind-ledger"),
            Mention(chunk_id="c2", entity_id="concept:pricing"),
        ],
        relations=[
            Relation(
                source_id="product:northwind",
                target_id="concept:pricing",
                type="USES",
                chunk_id="c1",
            ),
            Relation(
                source_id="product:northwind",
                target_id="product:northwind-ledger",
                type="SAME_AS",
                chunk_id="c1",
            ),
        ],
    )

    folded = apply_aliases(enrichment, table)

    assert [e.id for e in folded.entities] == ["product:northwind-ledger", "concept:pricing"]
    ledger = folded.entities[0]
    assert ledger.name == "Northwind Ledger"
    assert ledger.aliases == ["Northwind"]  # the surface form is recorded, not lost
    assert ledger.description == "A tool."  # kept from whichever spelling carried one
    assert [(m.chunk_id, m.entity_id) for m in folded.mentions] == [
        ("c1", "product:northwind-ledger"),
        ("c2", "concept:pricing"),
    ]
    assert [(r.source_id, r.target_id) for r in folded.relations] == [
        ("product:northwind-ledger", "concept:pricing")
    ]  # the SAME_AS became a self-relation and says nothing now


def test_an_empty_table_leaves_an_extraction_exactly_as_it_was() -> None:
    enrichment = Enrichment(entities=[Entity(id="concept:x", name="X")])

    assert apply_aliases(enrichment, load_persona_aliases(Path("nowhere"), "nobody")) == enrichment


def test_importing_with_a_table_never_creates_the_alias_node(
    ingested: IngestReport, memory_store: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The name a passage uses is still what the mention is matched on; only the node changes."""
    table = load_aliases(
        write_table(tmp_path, "test-pm", "aliases:\n  Customer Retention: [retention]\n")
    )
    doc = memory_store.list_documents("test-pm", speaker="Ada North")[0]
    file = tmp_path / "ada.json"
    file.write_text(
        json.dumps(
            {
                "doc_id": doc.id,
                "entities": [{"name": "retention", "type": "metric"}],
                "relations": [],
            }
        )
    )

    result = import_extraction_file(memory_store, file, aliases=table)

    assert result.ok and result.renamed == ("retention -> Customer Retention",)
    assert list(memory_store.entities) == ["metric:customer-retention"]
    assert memory_store.entities["metric:customer-retention"].aliases == ["retention"]
    # matched on the surface form, so the mention landed on a passage that says "retention"
    assert result.mentions == 1
    landed = memory_store.get_chunks([memory_store.mentions[0].chunk_id])[0]
    assert "retention" in landed.text.lower()


# ----------------------------------------------------------------------------- folding the graph


@pytest.fixture
def two_spellings(ingested: IngestReport, memory_store: InMemoryGraphStore) -> list[str]:
    """One thing in the graph twice, mentioned in two different documents."""
    doc_ids = sorted(memory_store.document_ids("test-pm"))
    first = memory_store.document_chunks(doc_ids[0], 0, 1)[0].id
    second = memory_store.document_chunks(doc_ids[1], 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="product:northwind", name="Northwind", type="product"),
                Entity(
                    id="product:northwind-ledger",
                    name="Northwind Ledger",
                    type="product",
                    description="The ledger.",
                ),
                Entity(id="concept:pricing", name="Pricing"),
            ],
            mentions=[
                Mention(chunk_id=first, entity_id="product:northwind"),
                Mention(chunk_id=second, entity_id="product:northwind-ledger"),
                Mention(chunk_id=second, entity_id="concept:pricing"),
            ],
            relations=[
                Relation(
                    source_id="product:northwind",
                    target_id="concept:pricing",
                    type="USES",
                    chunk_id=first,
                )
            ],
        )
    )
    return [first, second]


def test_merging_moves_the_mentions_records_the_spellings_and_drops_the_node(
    memory_store: InMemoryGraphStore, two_spellings: list[str]
) -> None:
    moved = memory_store.merge_entities("test-pm", "Northwind Ledger", ["Northwind", "northwind!"])

    assert moved == 1
    assert "product:northwind" not in memory_store.entities
    canonical = memory_store.entities["product:northwind-ledger"]
    assert canonical.name == "Northwind Ledger"
    assert canonical.description == "The ledger."
    assert canonical.aliases == ["Northwind", "northwind!"]
    assert sorted(m.entity_id for m in memory_store.mentions) == [
        "concept:pricing",
        "product:northwind-ledger",
        "product:northwind-ledger",
    ]
    assert memory_store.relations[0].source_id == "product:northwind-ledger"


def test_merging_again_moves_nothing_so_sync_can_run_it_every_time(
    memory_store: InMemoryGraphStore, two_spellings: list[str]
) -> None:
    memory_store.merge_entities("test-pm", "Northwind Ledger", ["Northwind"])
    before = list(memory_store.mentions)

    assert memory_store.merge_entities("test-pm", "Northwind Ledger", ["Northwind"]) == 0
    assert memory_store.mentions == before


def test_merging_a_canonical_the_graph_has_never_seen_creates_it_and_renames_nothing_else(
    memory_store: InMemoryGraphStore, two_spellings: list[str]
) -> None:
    """The graph holds only alias spellings, so the canonical node has to be made."""
    moved = memory_store.merge_entities("test-pm", "Pricing Model", ["Pricing"])

    assert moved == 1
    assert "concept:pricing" not in memory_store.entities
    assert memory_store.entities["concept:pricing-model"].name == "Pricing Model"
    assert memory_store.entities["concept:pricing-model"].aliases == ["Pricing"]


def test_merging_two_spellings_of_one_passage_leaves_one_mention(
    memory_store: InMemoryGraphStore, two_spellings: list[str]
) -> None:
    """Both spellings in the same passage must not become two edges to the same node."""
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="product:northwind", name="Northwind", type="product")],
            mentions=[Mention(chunk_id=two_spellings[1], entity_id="product:northwind")],
        )
    )

    moved = memory_store.merge_entities("test-pm", "Northwind Ledger", ["Northwind"])

    assert moved == 2  # both alias edges were re-pointed
    ledger = [m for m in memory_store.mentions if m.entity_id == "product:northwind-ledger"]
    assert len(ledger) == 2  # one per passage, not per spelling
    assert len({m.chunk_id for m in ledger}) == 2


def test_a_name_no_spelling_of_which_is_in_the_graph_moves_nothing(
    memory_store: InMemoryGraphStore, two_spellings: list[str]
) -> None:
    assert memory_store.merge_entities("test-pm", "Absent Thing", ["nowhere"]) == 0


def test_another_personas_mentions_of_a_spelling_are_left_where_they_are(
    memory_store: InMemoryGraphStore, two_spellings: list[str], thread_document: str
) -> None:
    """Entity nodes are shared, so a merge for one persona must not reach into another's."""
    other = memory_store.document_chunks(thread_document, 0, 1)[0].id
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="product:northwind", name="Northwind", type="product")],
            mentions=[Mention(chunk_id=other, entity_id="product:northwind")],
        )
    )

    memory_store.merge_entities("test-pm", "Northwind Ledger", ["Northwind"])

    assert "product:northwind" in memory_store.entities  # still held by the other persona
    assert [m.chunk_id for m in memory_store.mentions if m.entity_id == "product:northwind"] == [
        other
    ]


# ----------------------------------------------------------------------------- the pass


def test_applying_a_table_reports_per_canonical_and_a_preview_matches_it(
    memory_store: InMemoryGraphStore, two_spellings: list[str], tmp_path: Path
) -> None:
    table = load_aliases(write_table(tmp_path, "test-pm"))

    preview = preview_alias_table(memory_store, "test-pm", table)
    applied = apply_alias_table(memory_store, "test-pm", table)

    assert preview.dry_run and not applied.dry_run
    assert preview.mentions_moved == applied.mentions_moved == 1
    assert [m.canonical for m in applied.applied] == ["Northwind Ledger"]
    assert alias_lines(preview) == [
        "Northwind Ledger: would move 1 mentions from 3 spellings",
        "test-pm: would move 1 mentions onto 1 canonical names",
    ]
    assert alias_lines(applied)[-1] == "test-pm: moved 1 mentions onto 1 canonical names"
    assert alias_lines(apply_alias_table(memory_store, "test-pm", table)) == [
        "test-pm: no alias spellings found in the graph"
    ]


def test_the_cli_reports_a_dry_run_before_it_writes(
    cli_context: AppContext, two_spellings: list[str], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from graphrag.cli import app

    runner = CliRunner()
    write_table(cli_context.settings.personas_dir, "test-pm")

    dry = runner.invoke(app, ["aliases", "apply", "test-pm", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would move 1 mentions" in dry.stdout
    assert "product:northwind" in cli_context.store.entities  # a dry run wrote nothing

    done = runner.invoke(app, ["aliases", "apply", "test-pm"])
    assert done.exit_code == 0, done.output
    assert "moved 1 mentions onto 1 canonical names" in done.stdout
    assert "product:northwind" not in cli_context.store.entities
    assert (cli_context.settings.snapshots_dir / "test-pm" / "manifest.json").exists()


def test_a_persona_with_no_table_is_told_so_rather_than_failing(cli_context: AppContext) -> None:
    from typer.testing import CliRunner

    from graphrag.cli import app

    result = CliRunner().invoke(app, ["aliases", "apply", "test-pm"])

    assert result.exit_code == 0
    assert "has no alias table" in result.output


def test_a_broken_table_stops_the_command_with_the_file_named(cli_context: AppContext) -> None:
    from typer.testing import CliRunner

    from graphrag.cli import app

    write_table(
        cli_context.settings.personas_dir,
        "test-pm",
        "aliases:\n  A: [shared]\n  B: [shared]\n",
    )

    result = CliRunner().invoke(app, ["aliases", "apply", "test-pm"])

    assert result.exit_code == 2
    assert "claimed by both" in result.output
