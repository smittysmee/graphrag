"""`graphrag sna` end to end against the in-memory context, the way the other CLI tests run."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
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
