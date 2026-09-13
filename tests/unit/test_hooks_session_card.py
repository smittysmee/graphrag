"""The SessionStart status card, exercised against a fake project and a fake client."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from graphrag.hooks import session_card
from graphrag.hooks.mcp_client import McpUnavailable

NOW = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)  # noqa: UP017 - hooks target py3.10


class FakeClient:
    """A tiny stand-in for ``McpClient``: no sockets, just a canned ``call_tool``."""

    def __init__(
        self, stats: dict[str, Any] | None = None, url: str = "http://localhost:8765/mcp"
    ) -> None:
        self.url = url
        self._stats = stats
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self._stats is None:
            raise McpUnavailable("down")
        if name == "stats":
            return self._stats
        raise AssertionError(f"unexpected tool call {name!r}")


def stats_payload(per_persona: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {"graph": {"per_persona": per_persona}, "snapshots": [], "embedding": {}}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """One persona with a snapshot and a couple of raw files, another with no snapshot yet."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")

    handbook = tmp_path / "personas" / "handbook"
    handbook.mkdir(parents=True)
    (handbook / "persona.yaml").write_text(
        "id: handbook\nname: Handbook\nsources:\n  - id: notes\n    path: notes\n",
        encoding="utf-8",
    )

    fresh = tmp_path / "personas" / "fresh"
    fresh.mkdir(parents=True)
    (fresh / "persona.yaml").write_text("id: fresh\nname: Fresh\n", encoding="utf-8")

    notes = tmp_path / "data" / "raw" / "handbook" / "notes"
    notes.mkdir(parents=True)
    (notes / "one.md").write_text("# One\n", encoding="utf-8")
    (notes / "two.md").write_text("# Two\n", encoding="utf-8")

    snapshot = tmp_path / "data" / "snapshots" / "handbook"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "persona_id": "handbook",
                "created_at": "2026-09-09T20:57:22Z",
                "document_count": 2,
                "chunk_count": 9,
                "entity_count": 4,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


# ----------------------------------------------------------------------------- render_card


def test_render_card_lists_every_persona(root: Path) -> None:
    card = session_card.render_card(root, None, NOW)
    assert card.startswith("graphrag knowledge graph\n")
    assert "- fresh (Fresh): unknown (no snapshot; server not reachable)" in card
    assert "handbook" in card


def test_render_card_uses_the_manifest_when_the_server_is_down(root: Path) -> None:
    card = session_card.render_card(root, None, NOW)
    assert "- handbook (Handbook): 2 documents, 9 chunks (snapshot), snapshot 2026-09-09" in card
    assert "MCP server: not reachable at" in card


def test_render_card_prefers_live_counts_when_the_server_answers(root: Path) -> None:
    client = FakeClient(stats_payload({"handbook": {"documents": 5, "chunks": 40}}))
    card = session_card.render_card(root, client, NOW)  # type: ignore[arg-type]
    assert "- handbook (Handbook): 5 documents, 40 chunks, snapshot 2026-09-09" in card
    assert "- fresh (Fresh): not ingested" in card
    assert "MCP server: up at http://localhost:8765/mcp" in card
    assert [name for name, _ in client.calls] == ["stats"]


def test_render_card_falls_back_to_the_manifest_when_the_stats_call_itself_fails(
    root: Path,
) -> None:
    """The connection is up (``client`` is not ``None``) even if one tool call errors."""
    client = FakeClient(stats=None)
    card = session_card.render_card(root, client, NOW)  # type: ignore[arg-type]
    assert "- handbook (Handbook): 2 documents, 9 chunks (snapshot)" in card
    assert "MCP server: up at http://localhost:8765/mcp" in card


def test_render_card_lists_newest_raw_files(root: Path) -> None:
    card = session_card.render_card(root, None, NOW)
    assert "newest raw files:" in card
    assert "data/raw/handbook/notes/one.md (" in card
    assert "data/raw/handbook/notes/two.md (" in card


def test_render_card_omits_newest_files_section_when_there_are_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")
    (tmp_path / "personas").mkdir()
    card = session_card.render_card(tmp_path, None, NOW)
    assert "newest raw files:" not in card
    assert "no personas found under personas/" in card


def test_render_card_includes_the_uningested_line_when_given(root: Path) -> None:
    card = session_card.render_card(
        root, None, NOW, uningested_line="graphrag: 3 raw files not in the graph."
    )
    assert "graphrag: 3 raw files not in the graph." in card


def test_render_card_omits_the_uningested_section_when_none_is_given(root: Path) -> None:
    card = session_card.render_card(root, None, NOW, uningested_line=None)
    assert "not in the graph" not in card


def test_render_card_ends_with_the_tool_pointer(root: Path) -> None:
    card = session_card.render_card(root, None, NOW)
    assert card.rstrip().splitlines()[-1] == (
        "Use the graphrag MCP tools (context, search, read_document); "
        "persona skills are under .claude/skills/."
    )


def test_render_card_is_compact(root: Path) -> None:
    card = session_card.render_card(
        root, None, NOW, uningested_line="graphrag: 1 raw file not in the graph."
    )
    assert len(card.splitlines()) < 20


# ----------------------------------------------------------------------------- main()


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "argv", ["session_card.py", *argv])
    assert session_card.main() == 0
    return capsys.readouterr().out


def test_main_prints_the_card_for_a_known_root(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(
        monkeypatch,
        {
            "session_id": "x",
            "cwd": str(root),
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
        [],
        capsys,
    )
    assert out.startswith("graphrag knowledge graph\n")


def test_main_prints_nothing_outside_a_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    stray = tmp_path / "stray"
    stray.mkdir()
    out = _run_main(
        monkeypatch,
        {
            "session_id": "x",
            "cwd": str(stray),
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
        [],
        capsys,
    )
    assert out == ""


def test_main_json_flag_emits_hook_specific_output(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(
        monkeypatch,
        {
            "session_id": "x",
            "cwd": str(root),
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
        ["--json"],
        capsys,
    )
    payload = json.loads(out)
    output = payload["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart"
    assert output["additionalContext"].startswith("graphrag knowledge graph\n")


def test_main_never_raises_on_garbage_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    stray = tmp_path / "stray"
    stray.mkdir()
    monkeypatch.chdir(stray)
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    monkeypatch.setattr(sys, "argv", ["session_card.py"])
    assert session_card.main() == 0
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------------- wrapper script


def test_wrapper_script_runs_within_the_timeout(root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / ".claude" / "hooks" / "session-card.sh"
    payload = json.dumps(
        {
            "session_id": "x",
            "cwd": str(root),
            "hook_event_name": "SessionStart",
            "source": "startup",
        }
    )
    result = subprocess.run(
        [str(script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_PROJECT_DIR": str(root), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("graphrag knowledge graph\n")
