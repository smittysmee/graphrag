#!/usr/bin/env bash
# SessionStart hook: prints a short graphrag status card.
# See src/graphrag/hooks/session_card.py for the implementation. Fails silently: this runs on
# every session start (and from an agent's frontmatter), so it must never wedge or spam stderr.

command -v python3 >/dev/null 2>&1 || exit 0

if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    root="$CLAUDE_PROJECT_DIR"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
    root="$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd -P)"
fi

exec env PYTHONPATH="$root/src" python3 -m graphrag.hooks.session_card "$@"
