"""The PreToolUse subagent-publish guard, exercised via stdin payloads."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from graphrag.hooks import guard


def _payload(
    command: str,
    *,
    agent_id: str | None = None,
    agent_type: str | None = None,
    tool_name: str = "Bash",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "session_id": "x",
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": command},
    }
    if agent_id is not None:
        data["agent_id"] = agent_id
    if agent_type is not None:
        data["agent_type"] = agent_type
    return data


# ----------------------------------------------------------------------------- build_decision


def test_main_session_git_push_is_allowed() -> None:
    """No ``agent_id`` means the lead session; publishing is its call, not the guard's."""
    assert guard.build_decision(_payload("git push")) is None


def test_subagent_git_push_is_denied() -> None:
    decision = guard.build_decision(_payload("git push origin main", agent_id="agent-1"))
    assert decision == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "subagents may not publish; ask the lead session to push or open the PR"
            ),
        }
    }


def test_subagent_publish_after_a_chained_command_is_denied() -> None:
    decision = guard.build_decision(
        _payload("git status && git push", agent_id="agent-1", agent_type="Explore")
    )
    assert decision is not None
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_subagent_non_publish_command_is_allowed() -> None:
    assert guard.build_decision(_payload("git pull", agent_id="agent-1")) is None


def test_non_bash_tool_is_left_undecided() -> None:
    payload = _payload("git push", agent_id="agent-1", tool_name="Write")
    assert guard.build_decision(payload) is None


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --title x",
        "gh pr merge 1",
        "gh release create v1",
        "gh repo create foo",
        "git remote add origin https://example.com/x.git",
        "npm publish",
        "twine upload dist/*",
        "docker push example/image:latest",
    ],
)
def test_each_publish_pattern_is_denied(command: str) -> None:
    assert guard.build_decision(_payload(command, agent_id="agent-1")) is not None


def test_lookalike_command_is_not_denied() -> None:
    """Word boundaries keep ``git pushx`` and similar near-misses from matching."""
    assert guard.build_decision(_payload("git pushx origin", agent_id="agent-1")) is None


def test_missing_tool_input_is_allowed() -> None:
    payload = {"tool_name": "Bash", "agent_id": "agent-1"}
    assert guard.build_decision(payload) is None


# ----------------------------------------------------------------------------- main()


def _run_main(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert guard.main() == 0
    return capsys.readouterr().out


def test_main_is_silent_for_the_main_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _run_main(monkeypatch, _payload("git push"), capsys)
    assert out == ""


def test_main_denies_for_a_subagent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _run_main(monkeypatch, _payload("git push origin main", agent_id="agent-1"), capsys)
    decision = json.loads(out)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_main_allows_a_subagent_git_pull(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _run_main(monkeypatch, _payload("git pull", agent_id="agent-1"), capsys)
    assert out == ""


def test_main_never_raises_on_garbage_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    assert guard.main() == 0
    assert capsys.readouterr().out == ""


def test_main_is_silent_on_empty_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert guard.main() == 0
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------------- wrapper script


def test_wrapper_script_denies_a_subagent_publish() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / ".claude" / "hooks" / "guard-subagent.sh"
    payload = json.dumps(_payload("git push origin main", agent_id="agent-1"))
    started = time.monotonic()
    result = subprocess.run(
        [str(script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_PROJECT_DIR": str(repo_root), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    decision = json.loads(result.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert elapsed < 3.0


def test_wrapper_script_is_silent_for_the_main_session() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / ".claude" / "hooks" / "guard-subagent.sh"
    payload = json.dumps(_payload("git push"))
    result = subprocess.run(
        [str(script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_PROJECT_DIR": str(repo_root), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
