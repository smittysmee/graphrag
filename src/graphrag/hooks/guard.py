"""PreToolUse hook: stop a subagent from publishing anything outside the repo.

Subagent reports already come back neutralised by the harness, but a subagent that still has
tool access can act on injected text directly -- for example a data source that talks it into
running ``git push`` or ``gh pr create``. This hook denies a small set of publish-shaped Bash
commands (see :data:`_PUBLISH_PATTERNS`) whenever the stdin payload carries an ``agent_id``,
meaning the call originates inside a subagent rather than the lead session. The lead session
(no ``agent_id``) and any non-publish command are left untouched.

Fail-silent like the other hooks here: any error, or stdin that is not the JSON shape expected,
exits 0 with no output, which allows the call through the normal permission flow. A guard that
can wedge or falsely deny an unrelated command would be worse than no guard at all.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

__all__ = ["build_decision", "is_publish_command", "main"]

# Word-boundary patterns for commands that publish something outside the repo. Matched anywhere
# in the command string, so ``git status && git push`` and ``git push origin main`` both hit.
_PUBLISH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgit\s+push\b",
        r"\bgh\s+pr\s+create\b",
        r"\bgh\s+pr\s+merge\b",
        r"\bgh\s+release\b",
        r"\bgh\s+repo\s+create\b",
        r"\bgit\s+remote\s+add\b",
        r"\bnpm\s+publish\b",
        r"\btwine\s+upload\b",
        r"\bdocker\s+push\b",
    )
)

_DENY_REASON = "subagents may not publish; ask the lead session to push or open the PR"


def is_publish_command(command: str) -> bool:
    """Whether ``command`` matches one of the publish patterns anywhere in the string."""
    return any(pattern.search(command) for pattern in _PUBLISH_PATTERNS)


def build_decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The ``PreToolUse`` deny payload for this call, or ``None`` to leave it undecided.

    ``None`` covers the main session (no ``agent_id``), any non-``Bash`` tool, a payload
    missing the fields this hook needs, and a ``Bash`` command that is not publish-shaped.
    """
    if payload.get("tool_name") != "Bash":
        return None
    agent_id = payload.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command")
    if not isinstance(command, str) or not command:
        return None
    if not is_publish_command(command):
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": _DENY_REASON,
        }
    }


def _read_stdin_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    """Read the ``PreToolUse`` stdin payload; print a deny decision when it applies."""
    try:
        payload = _read_stdin_payload()
        decision = build_decision(payload)
        if decision is not None:
            print(json.dumps(decision))
    except Exception:  # a broken guard must never block or crash a tool call
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
