import json
from pathlib import Path

from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.pipeline import IngestReport
from tests.conftest import write_sample_corpus

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.stdout.strip()


def test_persona_commands(cli_context: AppContext) -> None:
    assert "test-pm" in runner.invoke(app, ["persona", "list"]).stdout
    show = runner.invoke(app, ["persona", "show", "test-pm"])
    assert show.exit_code == 0 and json.loads(show.stdout)["id"] == "test-pm"
    rec = runner.invoke(app, ["persona", "recommend", "discovery"])
    assert "test-pm" in rec.stdout
    new = runner.invoke(app, ["persona", "new", "Support Engineer", "-d", "Runbooks"])
    assert new.exit_code == 0 and "support-engineer" in new.stdout


def test_ingest_search_context_stats_and_snapshot(
    cli_context: AppContext, sample_corpus: Path
) -> None:
    ingest = runner.invoke(app, ["ingest", str(sample_corpus), "--persona", "test-pm", "--export"])
    assert ingest.exit_code == 0, ingest.output
    assert "ingested 3 documents" in ingest.stdout
    assert (cli_context.settings.snapshots_dir / "test-pm" / "manifest.json").exists()

    search = runner.invoke(app, ["search", "roadmap review", "--persona", "test-pm", "-k", "2"])
    assert search.exit_code == 0 and "Ben Oduya" in search.stdout

    ctx = runner.invoke(app, ["context", "retention", "--persona", "test-pm", "--json"])
    assert ctx.exit_code == 0 and json.loads(ctx.stdout)["persona_id"] == "test-pm"

    stats = runner.invoke(app, ["stats"])
    assert stats.exit_code == 0 and "test-pm" in stats.stdout

    listing = runner.invoke(app, ["snapshot", "list"])
    assert listing.exit_code == 0 and "test-pm" in listing.stdout

    skills = runner.invoke(app, ["persona", "export-skill", "--all"])
    assert skills.exit_code == 0
    assert (cli_context.settings.skills_dir / "persona-test-pm" / "SKILL.md").exists()


def test_setup_loads_snapshots_and_writes_skills(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    export = runner.invoke(app, ["snapshot", "export", "test-pm"])
    assert export.exit_code == 0
    cli_context.store.delete_persona("test-pm")
    setup = runner.invoke(app, ["setup"])
    assert setup.exit_code == 0, setup.output
    assert "1 snapshot(s) loaded" in setup.stdout
    assert cli_context.store.stats().documents == 3
    again = runner.invoke(app, ["setup"])
    assert "already loaded" in again.stdout


def test_ingest_errors(cli_context: AppContext, sample_corpus: Path) -> None:
    assert runner.invoke(app, ["ingest", str(sample_corpus), "--persona", "nope"]).exit_code != 0
    bad_src = runner.invoke(
        app, ["ingest", str(sample_corpus), "--persona", "test-pm", "--source", "x"]
    )
    assert bad_src.exit_code == 2


def test_doctor_reports_backend(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "backend=hash" in result.stdout


def test_sync_ingests_what_is_missing_then_goes_quiet(cli_context: AppContext) -> None:
    """`make sync` is one command, so the CLI behind it has to be safe to re-run."""
    write_sample_corpus(cli_context.settings.raw_dir / "test-pm")

    dry = runner.invoke(app, ["sync", "test-pm", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "3 documents missing" in dry.stdout
    assert cli_context.store.stats().documents == 0  # a dry run writes nothing

    done = runner.invoke(app, ["sync", "test-pm"])
    assert done.exit_code == 0, done.output
    assert cli_context.store.stats().documents == 3
    assert (cli_context.settings.snapshots_dir / "test-pm" / "manifest.json").exists()

    again = runner.invoke(app, ["sync", "test-pm"])
    assert again.exit_code == 0, again.output
    assert "nothing to sync" in again.stdout

    unknown = runner.invoke(app, ["sync", "test-pm", "--source", "nope"])
    assert unknown.exit_code == 2


def test_enrich_import_from_agent_json(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    doc = cli_context.store.list_documents("test-pm", speaker="Ada North")[0]
    payload = {
        "doc_id": doc.id,
        "entities": [
            {"name": "Retention", "type": "metric", "description": "Users who keep coming back."},
            {"name": "Onboarding", "type": "concept"},
        ],
        "relations": [
            {
                "source": "Onboarding",
                "target": "Retention",
                "type": "IMPROVES",
                "evidence": "fix onboarding first",
            }
        ],
    }
    file = tmp_path / "ada.json"
    file.write_text(json.dumps(payload))
    result = runner.invoke(app, ["enrich-import", "test-pm", str(file)])
    assert result.exit_code == 0, result.output
    assert "2 entities" in result.stdout
    assert cli_context.store.stats().entities == 2
    assert (cli_context.settings.snapshots_dir / "test-pm" / "enrichment.json.gz").exists()
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"doc_id": "nope", "entities": [], "relations": []}))
    assert runner.invoke(app, ["enrich-import", "test-pm", str(missing)]).exit_code == 2
    sloppy = tmp_path / "sloppy.json"
    sloppy.write_text(
        json.dumps(
            {
                "doc_id": doc.id,
                "entities": [{"name": "Not In Text", "type": "concept"}],
                "relations": [{"source": "Not In Text", "target": "Ghost", "type": "X"}],
            }
        )
    )
    dry = runner.invoke(app, ["enrich-import", "test-pm", str(sloppy), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "1 unmatched names" in dry.stdout
    assert "1 dangling relations" in dry.stdout
    assert cli_context.store.stats().entities == 2  # dry run wrote nothing
