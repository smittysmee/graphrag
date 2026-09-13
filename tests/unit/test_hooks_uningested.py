"""The Stop-hook un-ingested check, exercised against a fake project and a fake client."""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from graphrag.hooks import uningested
from graphrag.hooks.mcp_client import McpUnavailable

PERSONA_YAML = """\
id: demo-persona
name: Demo Persona
sources:
  - id: notes
    path: notes
    loader: documents
    glob: "**/*.md"
  - id: talks
    path: talks
    loader: transcripts
    glob: "*/transcript.md"
"""


class FakeClient:
    """A stand-in for ``McpClient``: answers the ``cypher`` tool from a canned id table."""

    def __init__(self, doc_ids_by_prefix: dict[str, list[str]] | None = None) -> None:
        self._doc_ids_by_prefix = doc_ids_by_prefix or {}
        self.calls: list[dict[str, Any]] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert name == "cypher"
        self.calls.append(arguments)
        params = arguments["params"]
        prefix, skip = params["prefix"], params["skip"]
        page = sorted(self._doc_ids_by_prefix.get(prefix, []))[skip : skip + 500]
        return [{"d.id": doc_id} for doc_id in page]


class FailingClient:
    """Every ``cypher`` call fails, like a server that answered the handshake then dropped."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise McpUnavailable("down")


def _write_snapshot(root: Path, persona_id: str, doc_ids: list[str]) -> None:
    snap_dir = root / "data" / "snapshots" / persona_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "manifest.json").write_text(
        json.dumps({"persona_id": persona_id}), encoding="utf-8"
    )
    with gzip.open(snap_dir / "documents.jsonl.gz", "wt", encoding="utf-8") as fh:
        for doc_id in doc_ids:
            fh.write(json.dumps({"id": doc_id}) + "\n")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """One persona: a ``documents`` source with two files, a ``transcripts`` source with two."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")
    persona_dir = tmp_path / "personas" / "demo-persona"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona.yaml").write_text(PERSONA_YAML, encoding="utf-8")

    notes = tmp_path / "data" / "raw" / "demo-persona" / "notes"
    notes.mkdir(parents=True)
    (notes / "one.md").write_text("# One\n", encoding="utf-8")
    (notes / "two.md").write_text("# Two\n", encoding="utf-8")

    talks = tmp_path / "data" / "raw" / "demo-persona" / "talks"
    for episode in ("ep1", "ep2"):
        directory = talks / episode
        directory.mkdir(parents=True)
        (directory / "transcript.md").write_text("# Talk\n", encoding="utf-8")

    return tmp_path


# ----------------------------------------------------------------------------- find_uningested


def test_snapshot_fallback_reports_missing_files_and_a_zero_ingested_source(root: Path) -> None:
    _write_snapshot(root, "demo-persona", ["demo-persona:notes:one"])
    report = uningested.find_uningested(root, None)

    assert report.source == "snapshot"
    by_source = {(f.persona_id, f.source_id): f for f in report.findings}
    assert by_source[("demo-persona", "notes")].missing_files == ("two.md",)
    assert by_source[("demo-persona", "talks")].missing_files == ()
    assert by_source[("demo-persona", "talks")].raw_count == 2
    assert report.total_files == 3
    assert report.by_persona == {"demo-persona": 3}
    assert report.raw_roots["demo-persona"] == str(root / "data" / "raw" / "demo-persona")


def test_transcripts_source_is_not_flagged_once_any_document_exists(root: Path) -> None:
    """The loader dedupes archived re-uploads, so folder count > document count is normal."""
    _write_snapshot(
        root,
        "demo-persona",
        ["demo-persona:notes:one", "demo-persona:notes:two", "demo-persona:talks:ep1"],
    )
    report = uningested.find_uningested(root, None)
    assert report.findings == ()


def test_live_graph_is_used_and_pages_past_five_hundred_rows(root: Path) -> None:
    padding = [f"demo-persona:notes:aaa{i:03d}" for i in range(500)]
    client = FakeClient(
        {
            "demo-persona:notes:": [*padding, "demo-persona:notes:one"],
            "demo-persona:talks:": ["demo-persona:talks:ep1"],
        }
    )
    report = uningested.find_uningested(root, client)  # type: ignore[arg-type]

    assert report.source == "graph"
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert (finding.persona_id, finding.source_id) == ("demo-persona", "notes")
    assert finding.missing_files == ("two.md",)

    skips = {
        c["params"]["skip"] for c in client.calls if c["params"]["prefix"] == "demo-persona:notes:"
    }
    assert skips == {0, 500}  # the id that matters only appears on the second page


def test_a_failing_source_query_is_skipped_not_raised(root: Path) -> None:
    report = uningested.find_uningested(root, FailingClient())  # type: ignore[arg-type]
    assert report.source == "graph"
    assert report.findings == ()


def test_only_document_suffixes_are_considered(root: Path) -> None:
    (root / "data" / "raw" / "demo-persona" / "notes" / "ignored.json").write_text(
        "{}", encoding="utf-8"
    )
    report = uningested.find_uningested(root, None)
    finding = next(f for f in report.findings if f.source_id == "notes")
    assert "ignored.json" not in finding.missing_files


# ----------------------------------------------------------------------------- summary_line


def test_summary_line_is_empty_when_nothing_is_missing() -> None:
    assert uningested.summary_line(uningested.Report(source="graph")) == ""


def test_summary_line_matches_the_documented_shape() -> None:
    report = uningested.Report(
        source="graph",
        findings=(
            uningested.Finding(
                "demo-persona", "notes", "documents", missing_files=("a.md", "b.md", "c.md")
            ),
        ),
    )
    assert uningested.summary_line(report) == "not yet ingested: 3 files (demo-persona: 3)"


def test_summary_line_uses_the_singular_for_one_file() -> None:
    report = uningested.Report(
        source="graph",
        findings=(
            uningested.Finding("demo-persona", "notes", "documents", missing_files=("a.md",)),
        ),
    )
    assert uningested.summary_line(report) == "not yet ingested: 1 file (demo-persona: 1)"


def test_summary_line_notes_the_snapshot_fallback() -> None:
    report = uningested.Report(
        source="snapshot",
        findings=(
            uningested.Finding("demo-persona", "notes", "documents", missing_files=("a.md",)),
        ),
    )
    assert uningested.summary_line(report).endswith("(vs committed snapshot)")


# ----------------------------------------------------------------------------- render_message


def test_render_message_is_none_when_nothing_is_missing() -> None:
    assert uningested.render_message(uningested.Report(source="graph")) is None


def test_render_message_names_files_and_gives_the_ingest_command() -> None:
    report = uningested.Report(
        source="graph",
        findings=(
            uningested.Finding(
                "demo-persona", "notes", "documents", missing_files=("a.md", "b.md")
            ),
        ),
        raw_roots={"demo-persona": "/repo/data/raw/demo-persona"},
    )
    message = uningested.render_message(report)
    assert message == (
        "graphrag: 2 raw files not in the graph: a.md, b.md. "
        "Ingest with `make ingest PERSONA=demo-persona SRC=/repo/data/raw/demo-persona` "
        "then re-import enrichment JSON."
    )
    assert message is not None
    assert len(message) <= 500


def test_render_message_shows_at_most_three_names_then_a_count() -> None:
    files = tuple(f"{i}.md" for i in range(5))
    report = uningested.Report(
        source="graph",
        findings=(uningested.Finding("demo-persona", "notes", "documents", missing_files=files),),
        raw_roots={"demo-persona": "/repo"},
    )
    message = uningested.render_message(report)
    assert message is not None
    assert "0.md, 1.md, 2.md (+2 more)" in message


def test_render_message_stays_under_the_budget_with_long_names() -> None:
    files = tuple(f"a-very-long-descriptive-research-note-file-name-{i}.md" for i in range(40))
    report = uningested.Report(
        source="graph",
        findings=(uningested.Finding("demo-persona", "notes", "documents", missing_files=files),),
        raw_roots={"demo-persona": "/repo/data/raw/demo-persona"},
    )
    message = uningested.render_message(report)
    assert message is not None
    assert len(message) <= 500


def test_render_message_labels_a_transcripts_zero_ingested_finding() -> None:
    report = uningested.Report(
        source="graph",
        findings=(uningested.Finding("demo-persona", "talks", "transcripts", raw_count=2),),
        raw_roots={"demo-persona": "/repo/data/raw/demo-persona"},
    )
    message = uningested.render_message(report)
    assert message is not None
    assert "talks/ (0 of 2 ingested)" in message


def test_render_message_picks_the_persona_with_the_most_missing_files() -> None:
    report = uningested.Report(
        source="graph",
        findings=(
            uningested.Finding("alpha", "notes", "documents", missing_files=("a.md",)),
            uningested.Finding("beta", "notes", "documents", missing_files=("b.md", "c.md")),
        ),
        raw_roots={"alpha": "/repo/alpha", "beta": "/repo/beta"},
    )
    message = uningested.render_message(report)
    assert message is not None
    assert "PERSONA=beta SRC=/repo/beta" in message


def test_render_message_notes_the_snapshot_fallback() -> None:
    report = uningested.Report(
        source="snapshot",
        findings=(
            uningested.Finding("demo-persona", "notes", "documents", missing_files=("a.md",)),
        ),
        raw_roots={"demo-persona": "/repo"},
    )
    message = uningested.render_message(report)
    assert message is not None
    assert "(vs committed snapshot)" in message


# ----------------------------------------------------------------------------- main()


def _run_main(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert uningested.main() == 0
    return capsys.readouterr().out


def test_main_prints_a_system_message_when_files_are_missing(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(
        monkeypatch,
        {"session_id": "x", "cwd": str(root), "hook_event_name": "Stop", "stop_hook_active": False},
        capsys,
    )
    payload = json.loads(out)
    assert payload["systemMessage"].startswith("graphrag:")


def test_main_is_silent_when_stop_hook_active(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(
        monkeypatch,
        {"session_id": "x", "cwd": str(root), "hook_event_name": "Stop", "stop_hook_active": True},
        capsys,
    )
    assert out == ""


def test_main_is_silent_outside_a_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    stray = tmp_path / "stray"
    stray.mkdir()
    monkeypatch.chdir(stray)  # a bogus cwd must not fall back to the real project's cwd
    out = _run_main(
        monkeypatch,
        {
            "session_id": "x",
            "cwd": str(stray),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        },
        capsys,
    )
    assert out == ""


def test_main_never_raises_on_garbage_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    stray = tmp_path / "stray"
    stray.mkdir()
    monkeypatch.chdir(stray)
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    assert uningested.main() == 0
    assert capsys.readouterr().out == ""


def test_main_is_silent_when_nothing_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")
    (tmp_path / "personas").mkdir()
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(
        monkeypatch,
        {
            "session_id": "x",
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        },
        capsys,
    )
    assert out == ""


# ----------------------------------------------------------------------------- wrapper script


def test_wrapper_script_runs_within_the_timeout(root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / ".claude" / "hooks" / "uningested.sh"
    payload = json.dumps(
        {"session_id": "x", "cwd": str(root), "hook_event_name": "Stop", "stop_hook_active": False}
    )
    started = time.monotonic()
    result = subprocess.run(
        [str(script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_PROJECT_DIR": str(root), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert '"systemMessage"' in result.stdout
    assert elapsed < 3.0
