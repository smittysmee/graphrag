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


def test_analyze_stances_compare_and_multilayer_communities_print_without_out(
    cli_context: AppContext, ingested: IngestReport, layered: InMemoryGraphStore
) -> None:
    """ATL-F1: `--out` is now optional on every report command; these four used to require it
    and print only a one-line summary, so this is the console-only mode that is new for them.
    """
    analyze = runner.invoke(
        app, ["sna", "analyze", "test-pm", "--network", "speakers", "--seed", "1", "--samples", "2"]
    )
    assert analyze.exit_code == 0, analyze.output
    assert "# Network analysis: test-pm / speakers" in analyze.stdout
    assert "## Provenance" in analyze.stdout

    stances = runner.invoke(app, ["sna", "stances", "test-layers"])
    assert stances.exit_code == 0, stances.output
    assert "# Stances: test-layers" in stances.stdout
    assert "## Caveats" in stances.stdout
    assert "annotated mentions over" in stances.stdout

    compare = runner.invoke(
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
        ],
    )
    assert compare.exit_code == 0, compare.output
    assert "## Who entered and who left" in compare.stdout

    multilayer = runner.invoke(
        app,
        [
            "sna",
            "multilayer-communities",
            "test-layers",
            "--network",
            "entities",
            "--layers",
            "stance",
            "--min-weight",
            "1",
            "--threshold",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "2",
        ],
    )
    assert multilayer.exit_code == 0, multilayer.output
    assert "## Multilayer community discovery" in multilayer.stdout
    assert "recommended:" in multilayer.stdout


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
    # The partition's checks sit under the partition's own heading, never under whichever
    # top-level section (chapter 28's evidence, say) happens to be rendered before them.
    before_checks = text[: text.index("### Stability across seeds")]
    nearest_heading = [line for line in before_checks.splitlines() if line.startswith("## ")][-1]
    assert nearest_heading.startswith("## Communities (")
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


def test_sna_analyze_prints_chapter_fourteens_rankings_and_their_stability(
    enriched: AppContext, tmp_path: Path
) -> None:
    """The new rankings reach the report and the payload, and so does the section reading them.

    Chapter 14's additions are only built if ``analyze`` prints them: harmonic, reach and
    coreness join the centrality tables, §14.8's centralization lands under them with its star
    denominators, and the stability half says how many resamples it took and of what.
    """
    out = tmp_path / "ranking.md"
    as_json = tmp_path / "ranking.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "5",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    for kind in ("harmonic", "reach", "coreness"):
        assert f"### {kind}" in text
    assert "## Ranking stability" in text
    assert "### Centralization (§14.8)" in text
    assert "Shells (§14.7)" in text
    assert "5 x `bootstrap-edges`" in text

    payload = json.loads(as_json.read_text())
    assert set(payload["centrality"]) >= {"harmonic", "reach", "coreness"}
    ranking = payload["ranking"]
    assert ranking["chapter"] == "Atlas ch. 14"
    assert ranking["nodes"] == payload["summary"]["nodes"]
    assert "bootstrap-edges" in ranking["null_model"]
    assert ranking["reach_hops"] == 2
    kinds = {row["kind"] for row in ranking["centralization"]}
    assert {"degree", "betweenness", "closeness", "harmonic", "coreness"} <= kinds
    undefined = {row["kind"] for row in ranking["centralization"] if row["value"] is None}
    assert "coreness" in undefined  # a star is a 1-core, so §14.8 has no denominator for it
    assert {row["kind"] for row in ranking["stability"]} == kinds
    assert all(row["method"] == "bootstrap-edges" for row in ranking["stability"])


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
        # ATL-ENT-4's provenance block carries a second, independent timestamp.
        data["provenance"].pop("generated_at")  # type: ignore[union-attr]
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
        # k=3 is the top of --k-range 2-3, so choosing it must come with the range-edge warning;
        # k=2 is the floor no search can go below, so it never does.
        chosen = json.loads(payload.read_text())[method]["k"]
        assert ("is the largest k searched (2-3)" in text) == (chosen == 3)
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


@pytest.mark.parametrize(
    "method",
    ["louvain-levels", "girvan-newman", "hrg", "sbm", "infomap", "walktrap", "label-propagation"],
)
def test_sna_analyze_runs_every_chapter_35_and_37_method(
    enriched: AppContext, tmp_path: Path, method: str
) -> None:
    """Every one of ATL-35's and ATL-37's methods must be reachable from `--method`, print its
    own diagnostics section, and go through the chapter 36 evaluation like any other partition."""
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
            "--min-weight",
            "1",
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
    assert "## Communities" in text
    assert "## Community evaluation" in text
    data = json.loads(payload.read_text())
    assert data["groups"]
    assert "evaluation" in data


def test_sna_analyze_directed_method_reads_the_relations_network(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    report = tmp_path / "directed.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "relations",
            "--method",
            "directed",
            "--seed",
            "1",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "directed modularity" in text.lower() or "§37.3" in text


def test_sna_analyze_degree_corrected_flag_only_affects_sbm(
    enriched: AppContext, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--method",
            "sbm",
            "--min-weight",
            "1",
            "--no-degree-corrected",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "plain_sbm.md"),
            "--json",
            str(tmp_path / "plain_sbm.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "plain_sbm.json").read_text())
    assert data["sbm"]["degree_corrected"] is False


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


def test_sna_compare_topology_adds_the_ch48_section(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "compare-topology.md"
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
            "--topology",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "topology: spectral" in result.stdout
    text = out.read_text()
    assert "## Topological distances" in text
    assert "DeltaCon similarity" in text


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


def test_sna_analyze_can_ask_for_the_bipartite_null(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """`--null bipartite` rewires the memberships the network was projected from.

    The entity network is a projection and carries them, so the section gains a second null. The
    topic network reads stored co-occurrence edges and never went through a projection, so the
    command says so and exits 2 rather than quietly reporting the shuffle under the other name.
    """
    out = tmp_path / "by-region.md"
    common = [
        "sna",
        "analyze",
        "test-layers",
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
        "--null",
        "bipartite",
        "--min-weight",
        "1",
    ]
    result = runner.invoke(app, [*common, "--network", "entities", "--out", str(out)])
    assert result.exit_code == 0, result.output
    # Three labelled entities is below the floor for measuring anything, so the section reports
    # its counts and stops; what the block looks like when there is enough to measure is pinned
    # in tests/unit/test_sna_null.py.
    assert "## By attribute: region" in out.read_text()

    unknown = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--null",
            "vibes",
            "--out",
            str(tmp_path / "nope.md"),
        ],
    )
    assert unknown.exit_code == 2
    assert "--null must be one of" in unknown.output


def test_sna_analyze_refuses_the_bipartite_null_on_a_network_that_was_never_projected(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    """The topic network reads the co-occurrence edges stored at ingestion, so there are no
    memberships to rewire. Saying so beats reporting the label shuffle under the other name."""
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "topics",
            "--min-weight",
            "1",
            "--by",
            "region",
            "--null",
            "bipartite",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "topics.md"),
        ],
    )

    assert result.exit_code == 2
    assert "carries none" in result.output


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


def test_sna_export_and_analyze_the_directed_relations_network(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The fifth network reaches the CLI as a DiGraph, and ``--relation-type`` cuts it."""
    target = tmp_path / "relations.graphml"
    result = runner.invoke(
        app, ["sna", "export", "test-layers", "--network", "relations", "-o", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "3 nodes, 4 edges (relations)" in result.stdout
    graph = nx.read_graphml(target)
    assert graph.is_directed()
    assert graph["product:alpha"]["product:beta"]["weight"] == 2
    assert not graph.has_edge("product:gamma", "product:alpha")

    typed = tmp_path / "competes.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "relations",
            "--relation-type",
            "competes_with",
            "-o",
            str(typed),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(typed.read_text())
    assert {(link["source"], link["target"]) for link in payload["links"]} == {
        ("product:alpha", "product:gamma"),
        ("product:gamma", "product:beta"),
    }

    report = tmp_path / "relations.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "relations",
            "--seed",
            "1",
            "--runs",
            "3",
            "--samples",
            "5",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "**Network type.** directed, weighted" in text
    assert "### in_degree" in text and "### out_degree" in text
    assert "reciprocity" in text
    # §14.5's two roles exist only where the edges point, so they are on this report and on no
    # other: a hub points at the good authorities, an authority is pointed at by the good hubs.
    assert "### hits_hub" in text and "### hits_authority" in text
    # Chapter 33 is on this report for the same reason and on no other (see the undirected
    # assertion below): alpha and beta relate to each other, so the network has a two-cycle.
    assert "## Hierarchy" in text
    # alpha -> beta -> alpha and alpha -> gamma -> beta -> alpha: all three entities are in one
    # strongly connected component, which §33.2 scores at zero hierarchicalness by definition.
    assert "### Type: strongly connected (§33.1)" in text
    assert "### Agony: the arrows that point up (§33.5)" in text

    wrong_network = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--relation-type",
            "competes_with",
            "-o",
            str(tmp_path / "x.json"),
        ],
    )
    assert wrong_network.exit_code == 2
    assert "--relation-type applies to" in wrong_network.output


def test_sna_layers_prints_the_per_layer_table_and_the_supra_adjacency_shape(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """`sna layers` is the table the flattening hides (Atlas §7.2), plus §8.1's matrix shape."""
    out = tmp_path / "layers.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "layers",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "layered by stance" in result.stdout
    assert "supra-adjacency (Atlas §8.1): 9 by 9" in result.stdout
    assert "3 node(s) times 3 layer(s), omega=1" in result.stdout
    assert "cannot see which layer an edge came from" in result.stdout
    text = out.read_text()
    assert "## Layers (3)" in text
    assert "| praise | 2 | 1 | 1 | 1 |" in text

    bad = runner.invoke(app, ["sna", "layers", "test-layers", "--layers", "colour"])
    assert bad.exit_code == 2
    assert "--layers must be one of" in bad.output


def test_sna_multilayer_communities_runs_flattening_layer_by_layer_and_supra_modularity(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """`sna multilayer-communities` (Atlas ch. 40) end to end, on the shared planted corpus."""
    out = tmp_path / "multilayer.md"
    payload = tmp_path / "multilayer.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "multilayer-communities",
            "test-layers",
            "--network",
            "entities",
            "--layers",
            "stance",
            "--min-weight",
            "1",
            "--threshold",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "2",
            "-o",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "recommended:" in result.stdout
    text = out.read_text()
    assert "## Multilayer community discovery" in text
    assert "### Flattening (§40.1)" in text
    assert "### Layer by layer with cross-layer matching (§40.2)" in text
    assert "### Multilayer modularity over the supra-adjacency (§40.3)" in text
    assert "### Multilayer density (§40.4)" in text
    assert "### Which to use (§40.5)" in text
    data = json.loads(payload.read_text())
    assert data["implements"].startswith("Atlas §40")
    assert data["recommendation"]["recommended"] in {"flatten", "layer-by-layer"}

    # A single-source persona has one "source" value, so --layers source cannot build two layers.
    too_few = runner.invoke(
        app, ["sna", "multilayer-communities", "test-layers", "--layers", "source", "-o", str(out)]
    )
    assert too_few.exit_code == 2
    assert "needs at least two layers" in too_few.output


def test_sna_analyze_with_layers_runs_on_the_flattening_and_says_so(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "facets.md"
    payload = tmp_path / "facets.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "entities",
            "--layers",
            "facet",
            "--min-weight",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "**Network type.** undirected, weighted, multilayer" in text
    assert "## Layers (2)" in text
    assert "| outage | 2 | 1 | 2 | 1 |" in text
    assert "cannot see which layer an edge came from" in text
    layers = json.loads(payload.read_text())["layers"]
    assert layers["layering"] == "facet"
    assert [row["name"] for row in layers["layers"]] == ["outage", "quoting"]


def test_sna_export_can_write_the_flattened_multilayer_network(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    target = tmp_path / "layered.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--layers",
            "facet",
            "--min-weight",
            "1",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    graph = nx.node_link_graph(json.loads(target.read_text()), edges="links")
    assert graph.graph["layers"] == "outage,quoting"
    assert graph["product:beta"]["product:gamma"]["layers"] == "quoting"


def test_sna_compare_takes_a_window_grid_instead_of_two_date_pairs(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """`--window` is the general form of `--since2`/`--until2` (Atlas §7.4)."""
    out = tmp_path / "grid.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-layers",
            "--network",
            "speakers",
            "--since",
            "2025-01-01",
            "--until",
            "2025-06-30",
            "--window",
            "2M",
            "--seed",
            "1",
            "--runs",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    # January against May-June: ana and bo, then ana and cy.
    assert "2 nodes in 2025-01-01 to 2025-02-28" in result.stdout
    text = out.read_text()
    assert "The window grid holds 3 windows and only its ends are compared" in text
    assert "2025-03-01 to 2025-04-30" in text

    clash = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-layers",
            "--since",
            "2025-01-01",
            "--until",
            "2025-06-30",
            "--window",
            "2M",
            "--since2",
            "2025-04-01",
            "--out",
            str(out),
        ],
    )
    assert clash.exit_code == 2
    assert "cannot be given with --since2" in clash.output


def test_the_cli_default_for_permutations_matches_the_analysis_default() -> None:
    """The CLI cannot import the analysis package at module scope, so it repeats the number."""
    from graphrag.cli import WHERE_PERMUTATIONS
    from graphrag.sna.attributes import PERMUTATIONS

    assert WHERE_PERMUTATIONS == PERMUTATIONS


# ----------------------------------------------------------------- backboning (Atlas ch. 27)


def test_sna_export_applies_a_backbone_and_writes_the_p_value_on_every_edge(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    target = tmp_path / "backboned.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--backbone",
            "noise-corrected",
            "--alpha",
            "0.5",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert payload["graph"]["backbone_method"] == "noise-corrected"
    assert "27.6" in payload["graph"]["frame"]
    for link in payload["links"]:
        assert link["p_value"] <= 0.5


def test_sna_export_refuses_a_backbone_it_does_not_have(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-pm",
            "--backbone",
            "magic",
            "-o",
            str(tmp_path / "x.json"),
        ],
    )
    assert result.exit_code == 2
    assert "noise-corrected" in result.output


def test_sna_backbone_compares_every_method_of_the_chapter(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "backbones.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "backbone",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = out.read_text()
    for method in (
        "naive",
        "naive-top",
        "doubly-stochastic",
        "high-salience",
        "convex",
        "disparity",
        "noise-corrected",
        "mst",
        "pmfg",
    ):
        assert f"`{method}`" in report
    assert "chapter 27" in report
    assert "Sampling frame:" in report
    assert "null model:" in report
    assert "p. 383" in report  # the naive row carries the book's critique of the threshold


def test_sna_backbone_says_so_when_there_is_nothing_to_filter(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    result = runner.invoke(
        app, ["sna", "backbone", "test-pm", "--network", "entities", "--min-weight", "9"]
    )
    assert result.exit_code == 2
    assert "nothing to backbone" in result.output


def test_the_min_weight_help_carries_the_books_objection_to_it() -> None:
    """p. 383: a fat-tailed weight distribution cannot motivate a threshold."""
    from graphrag.cli import MIN_WEIGHT_HELP

    assert "fat-tailed" in MIN_WEIGHT_HELP
    assert "standard deviations from the average" in MIN_WEIGHT_HELP
    assert "--backbone" in MIN_WEIGHT_HELP


def test_backboning_the_directed_relations_network_end_to_end(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """§27.5 and §27.6 have directed forms, so the fifth network can be backboned (p. 391-392)."""
    target = tmp_path / "relations-nc.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "relations",
            "--backbone",
            "noise-corrected",
            "--alpha",
            "0.9",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert payload["directed"] is True
    assert payload["graph"]["backbone_method"] == "noise-corrected"
    assert "27.6" in payload["graph"]["frame"]
    for link in payload["links"]:
        assert link["p_value"] <= 0.9

    report = tmp_path / "relations-df.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "relations",
            "--backbone",
            "disparity",
            "--alpha",
            "0.9",
            "--seed",
            "1",
            "--runs",
            "3",
            "--samples",
            "5",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "disparity" in report.read_text()


def test_sna_backbone_on_a_digraph_runs_what_applies_and_names_the_rest(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The table over a directed network: four methods run, five say why they cannot."""
    out = tmp_path / "relations-backbones.md"
    result = runner.invoke(
        app,
        ["sna", "backbone", "test-layers", "--network", "relations", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = out.read_text()
    assert "weakly connected component(s)" in report
    for method in ("naive", "naive-top", "disparity", "noise-corrected"):
        assert f"`{method}`" in report
    for method, because in (
        ("doubly-stochastic", "square matrix"),
        ("high-salience", "shortest-path tree"),
        ("convex", "Harary"),
        ("mst", "arborescence"),
        ("pmfg", "planarity"),
    ):
        assert f"`{method}`" in report
        assert because in report


def test_a_backbone_and_a_layering_cannot_be_asked_for_at_once(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """A backbone filters one network; a layering builds several and sums them."""
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--layers",
            "stance",
            "--backbone",
            "noise-corrected",
            "-o",
            str(tmp_path / "both.json"),
        ],
    )
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_the_threshold_option_reaches_a_structural_method(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """docs/SNA.md promises --threshold as the structural companion to --alpha."""
    target = tmp_path / "top.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--backbone",
            "naive-top",
            "--threshold",
            "2",
            "-o",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert payload["graph"]["backbone_threshold"] == 2.0
    assert "threshold 2" in payload["graph"]["backbone"]


def test_sna_walks_prints_the_pairwise_numbers_of_chapter_eleven(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`sna walks` is one pair of nodes, six numbers and the frame, n and chapter above them.

    The fixture's entity network is four entities co-mentioned across three documents, so every
    pair is connected and the numbers are finite. What is asserted here is the contract of the
    command -- the chapter, the frame, the asymmetry of the hitting times, and §11.4's bound
    against §11.5's cut -- rather than the arithmetic, which ``test_sna_walks.py`` holds to
    hand-computed answers.
    """
    result = runner.invoke(
        app,
        [
            "sna",
            "walks",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--from",
            "metric:retention",
            "--to",
            "concept:roadmap",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Atlas ch. 11 (random walks)" in result.stdout
    assert "Null model: none." in result.stdout
    assert "n=4 nodes, m=6 edges" in result.stdout
    assert "hitting time out" in result.stdout
    assert "hitting time back" in result.stdout
    assert "effective resistance" in result.stdout
    assert "minimum cut (= max flow)" in result.stdout
    assert "expensive to cut is cheap to walk across" in result.stdout

    missing = runner.invoke(
        app,
        ["sna", "walks", "test-pm", "--from", "metric:retention", "--to", "nobody:at:all"],
    )
    assert missing.exit_code == 2
    assert "not in the entities network: nobody:at:all" in missing.output

    same = runner.invoke(
        app,
        ["sna", "walks", "test-pm", "--from", "metric:retention", "--to", "metric:retention"],
    )
    assert same.exit_code == 2
    assert "two different nodes" in same.output


def test_sna_distance_compares_two_stance_vectors_on_the_entity_network(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    """`sna distance` is ch. 47: five sections, each its own answer, none of them "the" distance.

    The fixture's entity network (Alpha, Beta, Gamma) is a full triangle at ``--min-weight 1``:
    Alpha has one praise (post-1) and two complaints (post-2, post-4), Gamma has two complaints
    and one neutral, Beta has two praises. ``stance:praise`` and ``stance:complaint`` are two
    different, non-trivial vectors over the same three nodes, which is what every section below
    needs to have something to compare -- the exact numbers are held by
    ``test_sna_vectordist.py``'s known answers, not here.
    """
    result = runner.invoke(
        app,
        [
            "sna",
            "distance",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--vectors",
            "stance:praise",
            "stance:complaint",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Atlas ch. 47" in result.stdout
    assert "n=3 nodes" in result.stdout
    assert "stance:praise" in result.stdout
    assert "stance:complaint" in result.stdout
    assert "Null model" in result.stdout
    assert "## Non-network baselines (§47.1)" in result.stdout
    assert "## Generalized Euclidean (§47.2, Laplacian pseudoinverse)" in result.stdout
    assert "## Shortest-path and earth-mover (§47.3)" in result.stdout
    assert "## Graph Fourier transform (§47.4)" in result.stdout
    assert "## Network-space statistics (§47.5)" in result.stdout
    assert "Network correlation" in result.stdout


def test_sna_distance_reads_a_built_in_attribute_and_a_centrality(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    """``attr:mentions`` and ``centrality:weighted_degree`` need no annotation layer at all."""
    result = runner.invoke(
        app,
        [
            "sna",
            "distance",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--vectors",
            "attr:mentions",
            "centrality:weighted_degree",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "attr:mentions" in result.stdout
    assert "centrality:weighted_degree" in result.stdout


def test_sna_distance_reads_two_time_windows_as_occupancy_vectors(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    """``window:<since>:<until>:<kind>`` builds each window's own network and reads a centrality
    off it, so the fixture's January posts and June posts give two different degree vectors."""
    result = runner.invoke(
        app,
        [
            "sna",
            "distance",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--vectors",
            "window:2025-01-01:2025-01-31:weighted_degree",
            "window:2025-06-01:2025-06-30:weighted_degree",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "window:2025-01-01:2025-01-31:weighted_degree" in result.stdout
    assert "window:2025-06-01:2025-06-30:weighted_degree" in result.stdout
    assert "occupancy" in result.stdout


def test_sna_distance_refuses_an_unreadable_spec_and_a_stance_off_the_entity_network(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    bad_spec = runner.invoke(
        app,
        ["sna", "distance", "test-layers", "--vectors", "nonsense:thing", "attr:mentions"],
    )
    assert bad_spec.exit_code == 2
    assert "not a vector spec" in bad_spec.output

    wrong_network = runner.invoke(
        app,
        [
            "sna",
            "distance",
            "test-layers",
            "--network",
            "speakers",
            "--vectors",
            "stance:praise",
            "stance:complaint",
        ],
    )
    assert wrong_network.exit_code == 2
    assert "--network entities" in wrong_network.output


def test_sna_ego_prints_one_neighbourhood_with_and_without_the_ego(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`sna ego` is Atlas §30.1: the two densities, the clustering identity, and the brokerage.

    The fixture's entity network is four entities co-mentioned across three documents, so every
    entity is joined to the other three and each ego network is the whole thing. That is itself
    one of the section's warnings, and the command prints it rather than hiding it.
    """
    report = tmp_path / "ego.md"
    payload = tmp_path / "ego.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "ego",
            "test-pm",
            "metric:retention",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "## Ego network: Retention" in text
    assert "**Implements.** §30.1" in text
    assert "**Null model.** None." in text
    assert "local clustering coefficient" in text
    assert "Brokerage" in text

    data = json.loads(payload.read_text())
    assert data["node"] == "metric:retention"
    assert data["alters"] == 3
    # Every other entity is an alter and all three know each other, so the neighbourhood is
    # closed: density 1.0 without the ego, which is the ego's local clustering coefficient.
    assert data["density_without_ego"] == pytest.approx(1.0)
    assert data["local_clustering"] == pytest.approx(1.0)
    assert data["identity_holds"] is True
    assert data["brokerage_pairs"] == 0

    without = runner.invoke(
        app,
        ["sna", "ego", "test-pm", "metric:retention", "--min-weight", "1", "--without-ego"],
    )
    assert without.exit_code == 0, without.output
    assert "Reported without the ego." in without.stdout

    missing = runner.invoke(app, ["sna", "ego", "test-pm", "nobody:at:all"])
    assert missing.exit_code == 2
    assert "not in the entities network: nobody:at:all" in missing.output

    bad_radius = runner.invoke(app, ["sna", "ego", "test-pm", "metric:retention", "--radius", "0"])
    assert bad_radius.exit_code == 2
    assert "--radius must be at least 1 hop" in bad_radius.output


def test_an_analysis_carries_the_weak_ties_section(enriched: AppContext, tmp_path: Path) -> None:
    """§30.3 rides along with every analyze run, with its frame, its n and its absent null."""
    report = tmp_path / "note.md"
    payload = tmp_path / "note.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "5",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "## Weak ties" in text
    assert "**Implements.** §30.3" in text
    assert "Mean overlap" in text

    data = json.loads(payload.read_text())
    assert data["ties"]["measured"] == data["ties"]["edges"]
    assert "reading" in data["ties"]


def test_the_cli_default_for_the_walk_guard_matches_the_walks_module() -> None:
    """Same reason as the permutations default: a signature default is evaluated at import time
    and ``graphrag.sna.walks`` pulls in numpy, so the CLI repeats the number instead."""
    from graphrag.cli import WALK_MAX_NODES
    from graphrag.sna.walks import MAX_NODES

    assert WALK_MAX_NODES == MAX_NODES


def test_the_cli_defaults_for_the_distance_guards_match_the_vectordist_module() -> None:
    """Same reason as the walks guard: a signature default is evaluated at import time and
    ``graphrag.sna.vectordist`` pulls in numpy and scipy, so the CLI repeats the numbers instead."""
    from graphrag.cli import DISTANCE_MAX_NODES, DISTANCE_MAX_SUPPORT
    from graphrag.sna.vectordist import MAX_NODES, MAX_SUPPORT

    assert DISTANCE_MAX_NODES == MAX_NODES
    assert DISTANCE_MAX_SUPPORT == MAX_SUPPORT


def test_sna_analyze_takes_a_projection_scheme_and_names_it_in_the_frame(
    enriched: AppContext, tmp_path: Path
) -> None:
    """Atlas ch. 26: the weight on a projected edge is a choice, so the report says which."""
    report = tmp_path / "hyperbolic.md"
    payload = tmp_path / "hyperbolic.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--projection",
            "hyperbolic",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "5",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "hyperbolic projection" in text
    assert "modelling choice (Atlas ch. 26)" in text
    assert "projection=hyperbolic" in text
    # A hyperbolic weight is a fraction, so the count threshold the entity network defaults to
    # is not applied: the network is still here.
    assert json.loads(payload.read_text())["summary"]["nodes"] == 4

    simple = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--seed",
            "1",
            "-o",
            str(tmp_path / "simple.md"),
        ],
    )
    assert simple.exit_code == 0, simple.output
    assert "simple projection" in (tmp_path / "simple.md").read_text()

    bad = runner.invoke(
        app,
        ["sna", "analyze", "test-pm", "--projection", "astrology", "-o", str(tmp_path / "x.md")],
    )
    assert bad.exit_code == 2
    assert "--projection must be one of" in bad.output


def test_sna_export_refuses_a_projection_on_a_network_that_is_not_one(
    enriched: AppContext, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            "test-pm",
            "--network",
            "speakers-entities",
            "--projection",
            "probs",
            "-o",
            str(tmp_path / "two-mode.json"),
        ],
    )
    assert result.exit_code == 2
    assert "Add --project speakers or --project entities" in result.output


def test_sna_projections_compares_every_scheme_on_one_network(
    enriched: AppContext, tmp_path: Path
) -> None:
    """Atlas §26.6: one observation, every scheme, and where they disagree."""
    from graphrag.sna.projection import SCHEMES

    out = tmp_path / "projections.md"
    payload = tmp_path / "projections.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "projections",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "Implements." in text and "26.6" in text
    assert "## Edge-weight distributions" in text
    assert "## Rank agreement between schemes" in text
    assert "## What survives a threshold" in text
    assert "**Null model.** None" in text
    for scheme in (
        "simple",
        "jaccard",
        "cosine",
        "pearson",
        "euclidean",
        "hyperbolic",
        "resource",
        "probs",
        "heats",
        "hybrid",
        "randomwalk",
    ):
        assert f"`{scheme}`" in text

    data = json.loads(payload.read_text())
    assert data["implements"] == "26.6"
    assert [row["scheme"] for row in data["schemes"]] == list(SCHEMES)
    assert data["nodes"] == 4
    assert data["correlations"]


def test_sna_projections_refuses_a_network_that_was_never_projected(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    result = runner.invoke(app, ["sna", "projections", "test-pm", "--network", "topics"])
    assert result.exit_code == 2
    assert "two-mode memberships" in result.output

    bad = runner.invoke(app, ["sna", "projections", "test-pm", "--scheme", "astrology"])
    assert bad.exit_code == 2
    assert "--projection must be one of" in bad.output


def test_sna_projections_min_weight_narrows_the_memberships_it_compares(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """`--min-weight` prunes the simple-weight network, and the memberships of the nodes it
    strands have to go with it.

    The `test-layers` entity network is three entities over eight memberships; at `--min-weight 2`
    `product:beta` loses both its edges and is pruned out, leaving two. The membership table on
    the graph still holds all eight, because pruning touches edges and nodes and not the table it
    was projected from -- so without the filter the comparison would silently run over three
    entities' memberships while reporting a two-node network, and every discounting scheme would
    divide by the unpruned opposite-mode degrees.
    """

    def run(min_weight: str, name: str) -> dict[str, object]:
        target = tmp_path / f"{name}.json"
        result = runner.invoke(
            app,
            [
                "sna",
                "projections",
                "test-layers",
                "--network",
                "entities",
                "--min-weight",
                min_weight,
                "--json",
                str(target),
            ],
        )
        assert result.exit_code == 0, result.output
        return dict(json.loads(target.read_text()))

    wide, narrow = run("1", "loose"), run("2", "tight")
    assert wide["nodes"] == 3 and wide["memberships"] == 8
    assert narrow["nodes"] == 2
    assert int(narrow["memberships"]) < int(wide["memberships"])
    assert "--min-weight 2" in str(narrow["frame"])

    empty = runner.invoke(
        app,
        ["sna", "projections", "test-layers", "--network", "entities", "--min-weight", "99"],
    )
    assert empty.exit_code == 2
    assert "is empty before any scheme runs" in empty.output


def test_sna_compare_applies_one_projection_scheme_to_both_builds(
    enriched: AppContext, tmp_path: Path
) -> None:
    """Two schemes would be two definitions of an edge weight, so `--projection` is one option
    for both builds and both frames name it."""
    out = tmp_path / "compare.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--projection",
            "hyperbolic",
            "--seed",
            "1",
            "--runs",
            "2",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text().count("hyperbolic projection") >= 1

    bad = runner.invoke(
        app,
        ["sna", "compare", "test-pm", "--projection", "astrology", "-o", str(tmp_path / "x.md")],
    )
    assert bad.exit_code == 2
    assert "--projection must be one of" in bad.output

    bad_lambda = runner.invoke(
        app,
        [
            "sna",
            "compare",
            "test-pm",
            "--projection",
            "hybrid",
            "--lambda",
            "3",
            "-o",
            str(tmp_path / "y.md"),
        ],
    )
    assert bad_lambda.exit_code == 2
    assert "--lambda must be between 0" in bad_lambda.output


def test_sna_roles_prints_the_two_halves_of_chapter_fifteen(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`sna roles` is §15.1's role table and §15.2's similarity over one network.

    The fixture's entity network is four entities co-mentioned across three documents, so it is
    small and dense; what is asserted here is the contract of the command -- the chapter, the
    frame, the partition the roles are relative to, the absence of a null model, and the
    parameters of the similarity that was chosen -- rather than the arithmetic, which
    ``test_sna_roles.py`` holds to the book's own worked example and to planted graphs.
    """
    out = tmp_path / "roles.md"
    payload = tmp_path / "roles.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "roles",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--method",
            "simrank",
            "-k",
            "2",
            "--seed",
            "1",
            "--runs",
            "2",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "chapter 15 (node" in text
    assert "**Null model:** none." in text
    assert "**Partition:** Louvain at resolution 1" in text
    assert "decay=0.8" in text, "SimRank's decay is a choice, so it is printed"
    assert "ultra-peripheral" in text and "provincial hub" in text
    assert "Positions and the image matrix" in text
    assert "§15.3" in text

    data = json.loads(payload.read_text())
    assert data["implements"].startswith("Atlas ch. 15")
    assert data["similarity"]["method"] == "simrank"
    assert data["similarity"]["parameters"]["decay"] == 0.8
    assert len(data["roles"]) == data["nodes"] == 4
    assert data["blockmodel"]["k"] == 2
    assert data["null_model"].startswith("none")


def test_sna_roles_refuses_a_similarity_it_does_not_have_and_a_k_it_cannot_reach(
    enriched: AppContext,
) -> None:
    bad_method = runner.invoke(
        app, ["sna", "roles", "test-pm", "--network", "entities", "--method", "euclidean"]
    )
    assert bad_method.exit_code == 2
    assert "--method must be one of" in bad_method.output

    bad_k = runner.invoke(
        app,
        ["sna", "roles", "test-pm", "--network", "entities", "--min-weight", "1", "-k", "99"],
    )
    assert bad_k.exit_code == 2
    assert "-k must be between 1 and the 4 nodes" in bad_k.output


def test_sna_roles_says_so_when_the_network_has_no_edges(cli_context: AppContext) -> None:
    """No edges, no neighbours, so no node has a structural role to report (§15.2)."""
    result = runner.invoke(app, ["sna", "roles", "test-pm", "--network", "entities"])
    assert result.exit_code == 2
    assert "no node has a structural role" in result.output


def test_sna_analyze_carries_a_short_roles_section_next_to_its_communities(
    enriched: AppContext, tmp_path: Path
) -> None:
    """Chapter 15 in `analyze` is the short form: counts, hubs, and the same-role pairs.

    It appears only because a partition exists -- both of §15.1's coordinates are defined against
    the modules -- so the section names the grouping it was computed against and repeats the
    ceiling that grouping imposes on P.
    """
    out = tmp_path / "analysis.md"
    as_json = tmp_path / "analysis.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "3",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## Roles" in text
    assert "Atlas §15.1" in text
    assert "Null model: none" in text
    assert "P cannot exceed" in text

    data = json.loads(as_json.read_text())
    assert data["roles"]["implements"].startswith("Atlas §15.1")
    assert sum(data["roles"]["counts"].values()) == data["summary"]["nodes"]
    assert data["roles"]["same_role_never_cooccur"]["similarity"] == "jaccard"


def test_sna_community_grows_a_local_community_from_a_seed(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`sna community --seed` is §35.5: it must at least reach the seed's own neighbourhood and
    print the chapter 36 evaluation the wave's own rule asks for on every partition."""
    report = tmp_path / "community.md"
    payload = tmp_path / "community.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "community",
            "test-pm",
            "--seed",
            "metric:retention",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--random-seed",
            "1",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "## Local community: metric:retention" in text
    assert "Stopped:" in text
    assert "#### Community evaluation" in text
    data = json.loads(payload.read_text())
    assert data["implements"].startswith("Atlas §35.5")
    assert "metric:retention" in data["members"]
    assert "evaluation" in data


def test_sna_community_rejects_seed_and_temporal_together(enriched: AppContext) -> None:
    result = runner.invoke(
        app, ["sna", "community", "test-pm", "--seed", "metric:retention", "--temporal"]
    )
    assert result.exit_code == 2
    assert "two different modes" in result.output


def test_sna_community_needs_seed_or_temporal(enriched: AppContext) -> None:
    result = runner.invoke(app, ["sna", "community", "test-pm"])
    assert result.exit_code == 2
    assert "--seed" in result.output and "--temporal" in result.output


def test_sna_community_temporal_refuses_an_undated_corpus(enriched: AppContext) -> None:
    """`--temporal` needs `graphrag.sna.layers.snapshots`'s own dates; the fixture has none, and
    the command should turn that into a clean exit 2 rather than a traceback."""
    result = runner.invoke(app, ["sna", "community", "test-pm", "--temporal"])
    assert result.exit_code == 2
    assert "dated" in result.output


def test_sna_summarize_runs_all_four_techniques_and_the_evaluation(
    enriched: AppContext, tmp_path: Path
) -> None:
    """Atlas ch. 46 end to end: aggregation, compression, simplification, the influence summary,
    and -- the wave's own rule -- the chapter 36 battery on the `--by community` grouping."""
    report = tmp_path / "summary.md"
    payload = tmp_path / "summary.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "summarize",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--by",
            "community",
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
    assert "# Graph summarization" in text
    assert "backboning" in text
    assert "## Aggregation (§46.1)" in text
    assert "## Compression (§46.2)" in text
    assert "## Simplification (§46.3)" in text
    assert "## Influence-based summary (§46.4)" in text
    data = json.loads(payload.read_text())
    assert data["implements"].startswith("Atlas §46.1-46.4")
    assert data["by"] == "community"
    assert data["aggregation"]["supernodes"]
    assert "total_bits" in data["compression"]
    assert data["simplification"]["ranked"]
    assert data["influence"]["ranked"]


def test_sna_summarize_says_so_when_there_is_nothing_to_summarize(
    enriched: AppContext,
) -> None:
    result = runner.invoke(
        app, ["sna", "summarize", "test-pm", "--network", "entities", "--min-weight", "99"]
    )
    assert result.exit_code == 2
    assert "nothing to summarize" in result.output


def test_sna_hierarchy_reads_the_relations_network_and_refuses_the_others(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """Atlas ch. 33: the section stands on its own, and only a directed network has one."""
    section = tmp_path / "hierarchy.md"
    payload = tmp_path / "hierarchy.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "hierarchy",
            "test-layers",
            "--samples",
            "5",
            "--seed",
            "1",
            "--out",
            str(section),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = section.read_text()
    assert "**n.** 3 nodes and 4 arcs, directed." in text
    assert "**Implements.** §33.1" in text
    assert "Null model." in text and "configuration model" in text
    # alpha -> beta -> alpha, and alpha -> gamma -> beta -> alpha: every arc is inside the one
    # strongly connected component, which is §33.2's "hierarchicalness of zero by definition".
    data = json.loads(payload.read_text())
    assert data["cycles"]["on_cycles"] == 4
    assert data["cycles"]["flow_hierarchy"] == 0.0
    assert data["type"]["name"] == "strongly connected"
    assert data["global_reach"]["grc"] == 0.0  # everybody reaches everybody
    assert data["agony"]["exact"] is True
    # One boss each still leaves a tree over the knot: two of the four arcs survive.
    assert data["arborescence"]["kept"] == 2
    assert len(data["arborescence"]["roots"]) == 1
    assert data["layout"]["depth"] == 1  # one component, so one layer

    # One relation type is a narrower network, and the frame says so.
    typed = runner.invoke(
        app,
        [
            "sna",
            "hierarchy",
            "test-layers",
            "--relation-type",
            "competes_with",
            "--samples",
            "0",
        ],
    )
    assert typed.exit_code == 0, typed.output
    assert "3 nodes and 2 arcs" in typed.output

    # Chapter 33 assumes direction, and the four other networks have none.
    undirected = runner.invoke(
        app, ["sna", "hierarchy", "test-layers", "--network", "entities", "--min-weight", "1"]
    )
    assert undirected.exit_code == 2
    assert "directed" in undirected.output


def test_the_analyze_report_has_no_hierarchy_section_without_directions(
    enriched: AppContext, tmp_path: Path
) -> None:
    """The four undirected networks get no chapter 33 section, because it would be degenerate."""
    report = tmp_path / "entities.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "entities",
            "--seed",
            "1",
            "--runs",
            "2",
            "--samples",
            "5",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Hierarchy" not in report.read_text()


def test_the_cli_default_for_the_hierarchy_null_matches_the_hierarchy_module() -> None:
    """Same reason as the walks guard: a signature default is evaluated at import time and
    ``graphrag.sna.hierarchy`` pulls in scipy, so the CLI repeats the number instead."""
    from graphrag.cli import HIERARCHY_SAMPLES
    from graphrag.sna.hierarchy import NULL_SAMPLES

    assert HIERARCHY_SAMPLES == NULL_SAMPLES


def test_the_cli_default_for_the_similarity_guard_matches_the_roles_module() -> None:
    """Same reason as the walks guard: a signature default is evaluated at import time and
    ``graphrag.sna.roles`` pulls in numpy and sklearn, so the CLI repeats the number instead."""
    from graphrag.cli import ROLES_MAX_NODES
    from graphrag.sna.roles import MAX_SIMILARITY_NODES

    assert ROLES_MAX_NODES == MAX_SIMILARITY_NODES


def test_sna_robustness_prints_the_criterion_the_curves_and_the_cascade(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`sna robustness` end to end (Atlas ch. 22): §22.1's criterion, §22.2's curves, §22.3's
    cascade. The fixture's four-entity network is small and dense; what is asserted is the
    command's contract -- the chapter, the frame, the null model, and the sections `--cascade`
    adds -- rather than the arithmetic, which ``test_sna_robustness.py`` holds to a star graph
    and a planted coupling.
    """
    out = tmp_path / "robustness.md"
    payload = tmp_path / "robustness.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "robustness",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--strategy",
            "degree",
            "--steps",
            "4",
            "--runs",
            "2",
            "--seed",
            "1",
            "--cascade",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "Atlas ch. 22" in text
    assert "## Critical fraction (§22.1)" in text
    assert "## Removal curves" in text
    assert "## Cascade (§22.3)" in text
    assert "Null model.** Random removal" in text

    data = json.loads(payload.read_text())
    assert data["chapter"] == 22
    assert data["nodes"] == 4
    assert data["interdependent"] is None


def test_sna_robustness_couples_two_of_the_personas_networks(
    enriched: AppContext, tmp_path: Path
) -> None:
    """`--couple` (§22.4) needs a second network built the same way, and names it in the report."""
    out = tmp_path / "robustness-coupled.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "robustness",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--steps",
            "4",
            "--runs",
            "1",
            "--seed",
            "1",
            "--couple",
            "speakers-entities",
            "--coupling",
            "same-id",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## Interdependent networks (§22.4)" in text
    assert "speakers-entities" in text


def test_sna_robustness_refuses_an_unknown_strategy_load_or_coupling(
    enriched: AppContext,
) -> None:
    bad_strategy = runner.invoke(
        app,
        ["sna", "robustness", "test-pm", "--network", "entities", "--strategy", "not-a-thing"],
    )
    assert bad_strategy.exit_code == 2
    assert "--strategy must be one of" in bad_strategy.output

    bad_load = runner.invoke(
        app,
        ["sna", "robustness", "test-pm", "--network", "entities", "--cascade", "--load", "traffic"],
    )
    assert bad_load.exit_code == 2
    assert "--load must be one of" in bad_load.output

    bad_coupling = runner.invoke(
        app,
        [
            "sna",
            "robustness",
            "test-pm",
            "--network",
            "entities",
            "--couple",
            "speakers-entities",
            "--coupling",
            "nearest",
        ],
    )
    assert bad_coupling.exit_code == 2
    assert "--coupling must be one of" in bad_coupling.output


def test_sna_robustness_refuses_zero_steps_and_a_network_too_large_for_max_nodes(
    enriched: AppContext,
) -> None:
    bad_steps = runner.invoke(
        app, ["sna", "robustness", "test-pm", "--network", "entities", "--steps", "0"]
    )
    assert bad_steps.exit_code == 2
    assert "--steps and --runs must be at least 1" in bad_steps.output

    too_big = runner.invoke(
        app,
        [
            "sna",
            "robustness",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--max-nodes",
            "1",
        ],
    )
    assert too_big.exit_code == 2
    assert "above --max-nodes" in too_big.output


def test_sna_robustness_says_so_when_the_network_has_no_nodes(cli_context: AppContext) -> None:
    """No entities enriched onto the corpus yet, so the entity network is empty (§22.1)."""
    result = runner.invoke(app, ["sna", "robustness", "test-pm", "--network", "entities"])
    assert result.exit_code == 2
    assert "is empty, so there is nothing to remove" in result.output


def test_sna_motifs_prints_the_census_and_the_profile(enriched: AppContext, tmp_path: Path) -> None:
    """`sna motifs` is chapter 41's top-down half: the census always, the profile beside it.

    ``test_sna_motifs.py`` holds the arithmetic to planted graphs with a known answer; what is
    asserted here is the command's own contract -- the sections it prints, the null it names,
    and the shape of the JSON payload -- against the fixture's small entity network.
    """
    out = tmp_path / "motifs.md"
    payload = tmp_path / "motifs.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "motifs",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--samples",
            "20",
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "chapter 41" in text
    assert "## The three-node census (§41.2)" in text
    assert "## Motif profile at 3 nodes (§41.2)" in text
    assert "configuration" in text

    data = json.loads(payload.read_text())
    assert data["implements"].startswith("Atlas ch. 41")
    assert sum(data["census"]["counts"].values()) == data["census"]["triples"]
    assert data["profile"]["null_model"]
    assert data["profile"]["asked"] == 20


def test_sna_motifs_mine_reports_both_frequent_subgraph_sections(enriched: AppContext) -> None:
    """`--mine` adds §41.4 (only on `--network entities`, its own database) and §41.5.

    The fixture's three documents each name three of its four entities, which is dense enough
    for both sections to have something in them, so this checks that opting in actually reaches
    both code paths rather than one silently standing in for the other.
    """
    result = runner.invoke(
        app,
        [
            "sna",
            "motifs",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--mine",
            "--min-support",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Frequent subgraphs across documents (§41.4)" in result.output
    assert "## Frequent subgraphs inside this network (§41.5)" in result.output
    assert "no transactional section" not in result.output


def test_sna_motifs_says_so_when_there_are_fewer_than_three_nodes(
    cli_context: AppContext,
) -> None:
    result = runner.invoke(app, ["sna", "motifs", "test-pm", "--network", "entities"])
    assert result.exit_code == 2
    assert "fewer than three nodes" in result.output


def test_the_cli_defaults_for_overlap_match_the_overlap_module() -> None:
    """Same reason as the walk guard: a signature default is evaluated at import time and
    ``graphrag.sna.overlap`` pulls in scipy, so the CLI repeats the numbers instead."""
    from graphrag.cli import OVERLAP_DEFAULT_K, OVERLAP_JACCARD_THRESHOLD
    from graphrag.sna.overlap import DEFAULT_JACCARD_THRESHOLD, DEFAULT_K

    assert OVERLAP_DEFAULT_K == DEFAULT_K
    assert OVERLAP_JACCARD_THRESHOLD == DEFAULT_JACCARD_THRESHOLD


@pytest.mark.parametrize("method", ["clique", "link", "ego"])
def test_sna_overlap_runs_every_method_and_prints_every_section(
    enriched: AppContext, method: str, tmp_path: Path
) -> None:
    """`sna overlap` is ch. 38: a cover, its ch. 36 evaluation, the paradox check and a baseline.

    The fixture's entity network is a 4-node clique (every entity co-mentioned with every
    other), so `--method clique -k 4` finds the whole network as one community and the other two
    methods have something non-trivial to chew on too; what is asserted here is the command's own
    contract, not a known answer -- ``test_sna_overlap.py`` holds those.
    """
    out = tmp_path / "overlap.md"
    payload = tmp_path / "overlap.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "overlap",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--method",
            method,
            "--seed",
            "3",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## Overlapping coverage" in text
    assert "## Community evaluation" in text
    assert "### The overlap paradox (§38.7)" in text
    assert "### Against this network's own disjoint answer (§38.1)" in text

    data = json.loads(payload.read_text())
    assert data["method"] == method
    assert data["communities"]
    assert "evaluation" in data and "paradox" in data


def test_sna_overlap_rejects_an_unknown_method(enriched: AppContext) -> None:
    result = runner.invoke(app, ["sna", "overlap", "test-pm", "--method", "nope"])
    assert result.exit_code == 2
    assert "--method must be one of" in result.output


def test_sna_overlap_says_so_when_the_network_has_no_edges(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["sna", "overlap", "test-pm", "--network", "entities"])
    assert result.exit_code == 2
    assert "no edges" in result.output


# --------------------------------------------------------------------------- sna draw (ATL-49)


def test_the_cli_default_canvas_matches_the_draw_module() -> None:
    """Same reason as the walk guard: a signature default is evaluated at import time and
    ``graphrag.sna.draw`` pulls in networkx, so the CLI repeats the numbers instead."""
    from graphrag.cli import DRAW_HEIGHT, DRAW_WIDTH
    from graphrag.sna.draw import CANVAS_HEIGHT, CANVAS_WIDTH

    assert DRAW_WIDTH == CANVAS_WIDTH
    assert DRAW_HEIGHT == CANVAS_HEIGHT


@pytest.mark.parametrize("layout", ["force", "circular", "arc", "matrix"])
def test_sna_draw_writes_svg_and_html_for_every_undirected_layout(
    enriched: AppContext, layout: str, tmp_path: Path
) -> None:
    out = tmp_path / "network.svg"
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-pm",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--layout",
            layout,
            "--size",
            "degree",
            "--color",
            "community",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Visualization" in result.output
    svg = out.read_text()
    assert svg.startswith("<svg")
    html = out.with_suffix(".html").read_text()
    assert svg in html


def test_sna_draw_layered_needs_the_directed_relations_network(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "hierarchy.svg"
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-layers",
            "--network",
            "relations",
            "--layout",
            "layered",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    undirected = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-layers",
            "--network",
            "entities",
            "--layout",
            "layered",
            "--out",
            str(tmp_path / "nope.svg"),
        ],
    )
    assert undirected.exit_code == 2
    assert "directed" in undirected.output


def test_sna_draw_writes_json_and_report(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "by-count.svg"
    report = tmp_path / "by-count.md"
    payload = tmp_path / "by-count.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--layout",
            "force",
            "--seed",
            "1",
            "--out",
            str(out),
            "--report",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Visualization" in report.read_text()
    data = json.loads(payload.read_text())
    assert data["layout"] == "force"
    assert set(data["positions"])  # at least one node positioned


def test_sna_draw_color_by_a_categorical_attribute(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The entity network borrows `region` from its documents (see `--by region` above);
    `--color attr:region` should read it the same way and colour it categorically."""
    out = tmp_path / "by-region.svg"
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--color",
            "attr:region",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "attribute 'region' (categorical)" in result.output


def test_sna_draw_rejects_an_unknown_layout(enriched: AppContext, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["sna", "draw", "test-pm", "--layout", "spiral", "--out", str(tmp_path / "x.svg")],
    )
    assert result.exit_code == 2
    assert "--layout must be one of" in result.output


def test_sna_draw_rejects_an_unknown_size(enriched: AppContext, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            "test-pm",
            "--network",
            "entities",
            "--size",
            "popularity",
            "--out",
            str(tmp_path / "x.svg"),
        ],
    )
    assert result.exit_code == 2
    assert "centrality" in result.output


def test_sna_degree_carries_a_provenance_block_and_cite_adds_the_references(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    """ATL-ENT-4: every report prints `## Provenance`, and `--cite` -- off by default -- adds
    `## References`, naming the book once and exactly the chapters `provenance.chapters` lists,
    no more and no fewer."""
    out = tmp_path / "degree.md"
    payload_path = tmp_path / "degree.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            "test-pm",
            "--network",
            "speakers",
            "--bootstrap",
            "5",
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Provenance" in out.read_text()
    assert "## References" not in out.read_text()
    payload = json.loads(payload_path.read_text())
    assert payload["provenance"]["method"] == "degree"
    assert payload["provenance"]["chapters"] == [9]
    assert payload["provenance"]["seed"] == 1
    assert payload["provenance"]["tool_version"]

    cited = tmp_path / "degree-cited.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            "test-pm",
            "--network",
            "speakers",
            "--bootstrap",
            "5",
            "--out",
            str(cited),
            "--cite",
        ],
    )
    assert result.exit_code == 0, result.output
    text = cited.read_text()
    assert "## References" in text
    assert "Coscia" in text
    assert "Chapter 9 -- Degree" in text
    # Exactly the chapters this report's own provenance names -- not every chapter degree.py
    # could in principle touch, and not zero.
    assert text.count("Chapter 9") == 1
    assert "Chapter 10" not in text


def test_sna_backbone_cite_lists_chapters_13_and_27(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    out = tmp_path / "backbone.md"
    result = runner.invoke(
        app,
        ["sna", "backbone", "test-pm", "--network", "speakers", "--out", str(out), "--cite"],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "Chapter 13" in text
    assert "Chapter 27 -- Network Backboning" in text


def test_sna_export_cite_prints_the_references_to_the_console(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    """`sna export` has no markdown report at all -- `--cite` still adds the block, to the
    console, because provenance rides on `graph.graph` rather than on a report string."""
    target = tmp_path / "speakers.json"
    result = runner.invoke(
        app,
        ["sna", "export", "test-pm", "--network", "speakers", "-o", str(target), "--cite"],
    )
    assert result.exit_code == 0, result.output
    assert "## References" in result.output
    assert "Coscia" in result.output
    payload = json.loads(target.read_text())
    provenance = json.loads(payload["graph"]["provenance"])
    assert provenance["method"] == "export"
    assert provenance["chapters"] == [6, 7, 53]
