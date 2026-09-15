"""`graphrag sna` end to end against the in-memory context, the way the other CLI tests run."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport

runner = CliRunner()


@pytest.fixture
def enriched(cli_context: AppContext, ingested: IngestReport) -> AppContext:
    """Put entities on the sample corpus so the entity network has something to cluster.

    Each document gets a different three of the four entities, so the co-mention weights
    differ and the projection is not a single clique.
    """
    store = cli_context.store
    entities = [
        Entity(id="metric:retention", name="Retention", type="metric"),
        Entity(id="concept:onboarding", name="Onboarding", type="concept"),
        Entity(id="concept:pricing", name="Pricing", type="concept"),
        Entity(id="concept:roadmap", name="Roadmap", type="concept"),
    ]
    per_document = [
        ["metric:retention", "concept:onboarding", "concept:pricing"],
        ["metric:retention", "concept:onboarding", "concept:roadmap"],
        ["metric:retention", "concept:pricing", "concept:roadmap"],
    ]
    mentions: list[Mention] = []
    for doc_id, entity_ids in zip(sorted(store.document_ids("test-pm")), per_document, strict=True):
        for chunk in store.document_chunks(doc_id, 0, 100):
            mentions += [Mention(chunk_id=chunk.id, entity_id=e) for e in entity_ids]
    store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))
    return cli_context


def test_sna_guide_prints_the_selection_rules(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["sna", "guide"])
    assert result.exit_code == 0, result.output
    assert "Which method" in result.stdout
    assert "Louvain" in result.stdout and "Gaussian mixture" in result.stdout


def test_sna_export_writes_graphml_and_json(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    target = tmp_path / "speakers.graphml"
    result = runner.invoke(
        app, ["sna", "export", "test-pm", "--network", "speakers", "--out", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "4 nodes" in result.stdout
    graph = nx.read_graphml(target)
    assert "Lenny Rachitsky" in graph

    as_json = tmp_path / "topics.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-pm",
            "--network",
            "topics",
            "--min-weight",
            "1",
            "-o",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(as_json.read_text())["nodes"]


def test_sna_export_rejects_unknown_networks_and_personas(
    cli_context: AppContext, tmp_path: Path
) -> None:
    bad_net = runner.invoke(
        app, ["sna", "export", "test-pm", "--network", "people", "-o", str(tmp_path / "x.json")]
    )
    assert bad_net.exit_code == 2
    assert (
        runner.invoke(app, ["sna", "export", "nope", "-o", str(tmp_path / "y.json")]).exit_code != 0
    )


def test_sna_analyze_louvain_writes_a_report_with_its_checks(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    report = tmp_path / "note.md"
    payload = tmp_path / "note.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "speakers",
            "--method",
            "louvain",
            "--seed",
            "1",
            "--runs",
            "3",
            "--samples",
            "10",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "# Network analysis: test-pm / speakers" in text
    assert "Louvain because the input is a graph" in text
    assert "## Network summary" in text
    assert "### Stability across seeds" in text
    assert "### Null model" in text
    assert "## Centrality" in text
    assert "betweenness" in text
    assert "## Caveats" in text
    assert "Report the sampling frame" in text

    data = json.loads(payload.read_text())
    assert data["persona_id"] == "test-pm" and data["method"] == "louvain"
    assert data["seed"] == 1
    assert data["summary"]["nodes"] == 4
    assert "stability_adjusted_rand" in data["louvain"]
    assert data["null_model"]["samples"] >= 0
    assert data["caveats"]


def test_sna_analyze_is_deterministic_when_seeded(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    def run(name: str) -> dict[str, object]:
        out = tmp_path / f"{name}.json"
        result = runner.invoke(
            app,
            [
                "sna",
                "analyze",
                "test-pm",
                "--seed",
                "5",
                "--runs",
                "4",
                "--samples",
                "5",
                "--out",
                str(tmp_path / f"{name}.md"),
                "--json",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        data: dict[str, object] = json.loads(out.read_text())
        data.pop("generated_at")
        return data

    assert run("a") == run("b")


def test_sna_analyze_kmeans_and_gmm_on_spectral_features(
    enriched: AppContext, tmp_path: Path
) -> None:
    for method in ("kmeans", "gmm"):
        report = tmp_path / f"{method}.md"
        payload = tmp_path / f"{method}.json"
        result = runner.invoke(
            app,
            [
                "sna",
                "analyze",
                "test-pm",
                "--network",
                "entities",
                "--method",
                method,
                "--features",
                "spectral",
                "--min-weight",
                "1",
                "--k-range",
                "2-3",
                "--seed",
                "1",
                "--out",
                str(report),
                "--json",
                str(payload),
            ],
        )
        assert result.exit_code == 0, result.output
        text = report.read_text()
        assert "### Choosing k" in text
        assert "Clusters" in text
        assert "**Features.** `spectral`" in text
        data = json.loads(payload.read_text())
        assert data["features"] == "spectral"
        assert data["groups"]
        assert data["choose_k"]["rows"]
    assert "responsibilities" in json.loads((tmp_path / "gmm.json").read_text())["gmm"]


def test_sna_analyze_can_cluster_entities_by_what_they_are_about(
    enriched: AppContext, tmp_path: Path
) -> None:
    payload = tmp_path / "content.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--method",
            "kmeans",
            "--features",
            "embedding",
            "--min-weight",
            "1",
            "--k-range",
            "2-3",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "content.md"),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(payload.read_text())
    assert data["features"] == "embedding"
    assert data["feature_dims"] > 0
    assert "about the same" in (tmp_path / "content.md").read_text()


def test_embedding_features_are_refused_where_there_is_no_text(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "speakers",
            "--method",
            "kmeans",
            "--features",
            "embedding",
            "--out",
            str(tmp_path / "no.md"),
        ],
    )
    assert result.exit_code == 2
    assert "only the entities network" in result.output


def test_sna_analyze_rejects_an_unknown_method_and_a_bad_k_range(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    bad_method = runner.invoke(
        app, ["sna", "analyze", "test-pm", "--method", "vibes", "--out", str(tmp_path / "n.md")]
    )
    assert bad_method.exit_code == 2
    bad_range = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--method",
            "kmeans",
            "--k-range",
            "two to ten",
            "--out",
            str(tmp_path / "n.md"),
        ],
    )
    assert bad_range.exit_code == 2


def test_an_empty_network_reports_itself_instead_of_crashing(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    report = tmp_path / "empty.md"
    result = runner.invoke(
        app,
        ["sna", "analyze", "test-pm", "--network", "entities", "--seed", "1", "--out", str(report)],
    )
    assert result.exit_code == 0, result.output
    assert "The network is empty" in report.read_text()


def test_a_network_with_no_edges_reports_itself_instead_of_crashing(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """bo and cy are both attributed to the south but never appear in the same document, so
    filtering the speaker network down to the south leaves two nodes and no edges -- exactly
    what used to raise a ``ZeroDivisionError`` inside networkx's modularity."""
    report = tmp_path / "no-edges.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "speakers",
            "--where",
            "region=south",
            "--seed",
            "1",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "2 node(s) and no edges" in text
    assert "none of them ran" in text
    assert "## Communities" not in text
    assert "### Null model" not in text


# ------------------------------------------------------- the stance, facet and window options


def test_sna_export_takes_a_window_a_stance_and_a_facet(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    target = tmp_path / "complaints.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--stance",
            "complaint",
            "--min-weight",
            "1",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert {n["id"] for n in payload["nodes"]} == {"product:alpha", "product:gamma"}
    assert "praised together" not in payload["graph"]["frame"]
    assert "complaint" in payload["graph"]["stances"]

    windowed = tmp_path / "early.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "speakers",
            "--since",
            "2025-01-01",
            "--until",
            "2025-03-01",
            "-o",
            str(windowed),
        ],
    )
    assert result.exit_code == 0, result.output
    assert {n["id"] for n in json.loads(windowed.read_text())["nodes"]} == {"ana", "bo"}

    faceted = tmp_path / "outage.json"
    result = runner.invoke(
        app,
        ["sna", "export", "test-layers", "--facet", "outage", "-o", str(faceted)],
    )
    assert result.exit_code == 0, result.output
    assert {n["id"] for n in json.loads(faceted.read_text())["nodes"]} == {"bo", "cy"}


def test_sna_export_writes_the_bipartite_network_and_its_projections(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    both = tmp_path / "two-mode.graphml"
    result = runner.invoke(
        app,
        ["sna", "export", "test-layers", "--network", "speakers-entities", "-o", str(both)],
    )
    assert result.exit_code == 0, result.output
    assert "6 nodes" in result.stdout
    graph = nx.read_graphml(both)  # the partner strings have to survive GraphML
    assert graph.nodes["ana"]["mode"] == "speaker"
    assert "Beta" in graph.nodes["ana"]["partners"]

    onto = tmp_path / "speakers.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "speakers-entities",
            "--project",
            "speakers",
            "-o",
            str(onto),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(onto.read_text())
    assert {n["id"] for n in payload["nodes"]} == {"ana", "bo", "cy"}
    assert payload["graph"]["project"] == "speakers"


def test_sna_rejects_a_stance_a_projection_and_a_network_it_cannot_serve(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = str(tmp_path / "x.json")
    bad_stance = runner.invoke(
        app, ["sna", "export", "test-layers", "--stance", "grumpy", "-o", out]
    )
    assert bad_stance.exit_code == 2
    assert "--stance must be one of" in bad_stance.output

    wrong_network = runner.invoke(
        app,
        ["sna", "export", "test-layers", "--network", "speakers", "--stance", "praise", "-o", out],
    )
    assert wrong_network.exit_code == 2
    assert "--stance reads the annotation" in wrong_network.output

    bad_project = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "speakers-entities",
            "--project",
            "topics",
            "-o",
            out,
        ],
    )
    assert bad_project.exit_code == 2
    assert "--project must be one of" in bad_project.output


def test_sna_analyze_reports_the_filters_and_the_two_modes(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "two-mode.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "speakers-entities",
            "--facet",
            "outage",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "3",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "facets=outage" in text
    assert "## Across the two modes" in text
    assert "| # | size | speakers | entities |" in text
    assert "Alpha" in text and "bo" in text


def test_sna_stances_writes_the_report_and_names_an_entity_it_could_not_find(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "stances.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "stances",
            "test-layers",
            "--entity",
            "Alpha",
            "--entity",
            "Nope",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3 annotated mentions over 1 entities" in result.stdout
    assert "no annotated mentions for: Nope" in result.output
    text = out.read_text()
    assert "| entity | praise | complaint | total | documents |" in text
    assert "| Alpha | 1 | 2 | 3 | 3 |" in text
    assert "> Alpha went down on the busiest morning" in text


def test_sna_compare_writes_a_two_window_report(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "compare.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-layers",
            "--network",
            "speakers",
            "--until",
            "2025-03-01",
            "--since2",
            "2025-06-01",
            "--seed",
            "1",
            "--runs",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 nodes in up to 2025-03-01" in result.stdout
    assert "1 in both, 1 entered, 1 left" in result.stdout
    text = out.read_text()
    assert "## Who entered and who left" in text
    assert "Largest rank changes (weighted_degree)" in text

    bad = runner.invoke(
        app,
        ["sna", "compare", "test-layers", "--centrality", "charisma", "--out", str(out)],
    )
    assert bad.exit_code == 2
    assert "centrality must be one of" in bad.output


def test_sna_export_takes_an_attribute_filter(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    target = tmp_path / "north.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "speakers",
            "--where",
            "region=north",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert {n["id"] for n in payload["nodes"]} == {"ana"}
    assert payload["graph"]["where"] == "region=north"
    assert "Restricted to nodes attributed region=north" in payload["graph"]["frame"]
    # The value rides on the node, so a downstream tool can colour by it.
    assert payload["nodes"][0]["attr_region"] == "north"


def test_sna_rejects_a_where_it_cannot_read(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = str(tmp_path / "x.json")
    malformed = runner.invoke(app, ["sna", "export", "test-layers", "--where", "region", "-o", out])
    assert malformed.exit_code == 2
    assert "must look like key=value" in malformed.output

    twice = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--where",
            "region=north",
            "--where",
            "region=south",
            "-o",
            out,
        ],
    )
    assert twice.exit_code == 2
    assert "given twice" in twice.output


def test_sna_analyze_by_attribute_adds_the_section_to_the_report_and_the_json(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "by-region.md"
    as_json = tmp_path / "by-region.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--by",
            "region",
            "--permutations",
            "20",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## By attribute: region" in text
    assert "hypothesis about where this network divides" in text
    payload = json.loads(as_json.read_text())
    assert payload["attribute"]["key"] == "region"
    assert payload["attribute"]["counts"] == {"south": 2, "north": 1}


def test_sna_stances_can_be_cut_to_one_population(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "north-stances.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "stances",
            "test-layers",
            "--where",
            "region=north",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "where=region=north" in text
    # The north posts praise Alpha and Beta and read Gamma neutrally; nobody complains there.
    assert "complaint" not in text.split("## By entity")[1].split("###")[0]


def test_sna_compare_can_build_two_populations_instead_of_two_windows(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "north-vs-south.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--where",
            "region=north",
            "--where2",
            "region=south",
            "--seed",
            "1",
            "--runs",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "region=north against region=south" in text
    assert "Sampling frame, region=south" in text
    assert "3 nodes in region=north" in result.stdout


def test_the_cli_default_for_permutations_matches_the_analysis_default() -> None:
    """The CLI cannot import the analysis package at module scope, so it repeats the number."""
    from graphrag.cli import WHERE_PERMUTATIONS
    from graphrag.sna.attributes import PERMUTATIONS

    assert WHERE_PERMUTATIONS == PERMUTATIONS
