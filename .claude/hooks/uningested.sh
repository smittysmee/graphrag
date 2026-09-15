#!/usr/bin/env bash
# Stop hook: warns when files under data/raw have not reached the graph.
# See src/graphrag/hooks/uningested.py for the implementation. Fails silently: this runs
# after every turn, so it must never wedge, block, or spam stderr.

command -v python3 >/dev/null 2>&1 || exit 0

if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    root="$CLAUDE_PROJECT_DIR"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
    root="$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd -P)"
fi

exec env PYTHONPATH="$root/src" python3 -m graphrag.hooks.uningested "$@"
