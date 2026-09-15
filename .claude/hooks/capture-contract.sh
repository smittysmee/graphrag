#!/usr/bin/env bash
# PostToolUse hook: after a file is written under data/raw, states the document id it will get,
# the sidecar files it still owes the graph, and the command that checks them.
# See src/graphrag/hooks/capture_contract.py for the implementation. Fails silently: this runs
# after every Write/Edit, so it must never wedge, block, or spam stderr.

command -v python3 >/dev/null 2>&1 || exit 0

if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    root="$CLAUDE_PROJECT_DIR"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
    root="$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd -P)"
fi

exec env PYTHONPATH="$root/src" python3 -m graphrag.hooks.capture_contract "$@"
