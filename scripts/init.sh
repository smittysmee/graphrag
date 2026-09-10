#!/usr/bin/env bash
# Interactive first-run setup. Asks only what it cannot work out for itself, writes .env,
# then runs the right steps in the right order.
#
#   ./scripts/init.sh          interactive
#   ./scripts/init.sh --yes    accept every default, ask nothing (CI, scripted installs)
#
# Defaults are the safe ones: embeddings run locally, and nothing downloads model weights.
set -euo pipefail
cd "$(dirname "$0")/.."

ASSUME_YES=0
case "${1:-}" in --yes|-y) ASSUME_YES=1 ;; esac

# True when we can actually prompt: not in --yes mode, and a terminal exists to read from.
interactive() { [ "$ASSUME_YES" = 0 ] && [ -e /dev/tty ] && [ -r /dev/tty ]; }

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
dim()  { printf '\033[2m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
ok()   { printf '\033[32m%s\033[0m\n' "$1"; }

# Ask a multiple-choice question. Usage: choose <default-index> <prompt> <option>...
# Answer lands in $REPLY_INDEX. Non-interactive runs take the default.
choose() {
  local default="$1"; shift
  local prompt="$1"; shift
  local n=$#
  printf '\n'; bold "$prompt"
  local i=1
  for opt in "$@"; do
    if [ "$i" = "$default" ]; then printf '  %d) %s  (default)\n' "$i" "$opt"
    else printf '  %d) %s\n' "$i" "$opt"; fi
    i=$((i + 1))
  done
  if ! interactive; then
    REPLY_INDEX="$default"; printf '  → %s (default)\n' "$default"; return
  fi
  local ans
  while true; do
    printf '  > '
    read -r ans < /dev/tty 2>/dev/null || { REPLY_INDEX="$default"; printf '%s\n' "$default"; return; }
    ans="${ans:-$default}"
    if [ "$ans" -ge 1 ] 2>/dev/null && [ "$ans" -le "$n" ]; then REPLY_INDEX="$ans"; return; fi
    warn "  pick 1-$n"
  done
}

# A Neo4j password travels as NEO4J_AUTH=neo4j/<password>, so a slash silently produces an
# invalid value and the container crash-loops. Compose also interpolates $ in .env values.
valid_password() {
  case "$1" in ""|*/*|*' '*|*'$'*) return 1 ;; esac
  [ "${#1}" -ge 8 ]
}

valid_url() { case "$1" in http://*|https://*) return 0 ;; *) return 1 ;; esac; }

# Ask for free text with a default, re-asking until it validates. Answer lands in $REPLY_TEXT.
# Usage: ask <prompt> <default> [validator] [hint]
ask() {
  local prompt="$1" default="$2" validator="${3:-}" hint="${4:-}"
  if ! interactive; then
    REPLY_TEXT="$default"
    if [ -n "$validator" ] && ! "$validator" "$default"; then
      warn "  refusing an invalid default for '$prompt': $hint"; exit 1
    fi
    return
  fi
  while true; do
    printf '  %s [%s]: ' "$prompt" "$default"
    local ans; read -r ans < /dev/tty 2>/dev/null || ans=""
    ans="${ans:-$default}"
    if [ -z "$validator" ] || "$validator" "$ans"; then REPLY_TEXT="$ans"; return; fi
    warn "  $hint"
  done
}

model_cached() {
  command -v docker >/dev/null 2>&1 || return 1
  docker info >/dev/null 2>&1 || return 1
  ./scripts/model.sh verify >/dev/null 2>&1
}

printf '\n'; bold "graph-rag setup"
dim   "Everything runs in Docker. This writes .env and runs the right steps."

if [ -f .env ]; then
  choose 1 "You already have a .env." \
    "Keep it and just run setup" \
    "Reconfigure from scratch"
  [ "$REPLY_INDEX" = "1" ] && KEEP_ENV=1 || KEEP_ENV=0
else
  KEEP_ENV=0
fi

GOAL=3
if [ "$KEEP_ENV" = 0 ]; then
  choose 3 "What do you want to do first?" \
    "Import a persona someone sent me" \
    "Build a graph from my own documents" \
    "Just set up the platform; I'll decide later"
  GOAL="$REPLY_INDEX"

  choose 1 "Where should embeddings run?" \
    "On this machine (free, offline, uses CPU)" \
    "On another host: a GPU box, Ollama, vLLM, text-embeddings-inference"
  if [ "$REPLY_INDEX" = "2" ]; then
    BACKEND=http
    ask "Endpoint URL" "http://gb10.local:8766/v1" valid_url \
        "must start with http:// or https://"
    BASE_URL="$REPLY_TEXT"
  else
    BACKEND=fastembed
    BASE_URL=""
  fi

  # Only worth asking when a local model is actually needed and not already present.
  ALLOW_DOWNLOAD=false
  if [ "$BACKEND" = "fastembed" ]; then
    if model_cached; then
      ok "  the pinned embedding model is already cached; nothing to download"
    else
      choose 1 "The embedding model (~64 MB) is not cached. How should it get here?" \
        "I'll supply it offline (a persona bundle, or make model-import)" \
        "Download it now from Hugging Face"
      [ "$REPLY_INDEX" = "2" ] && ALLOW_DOWNLOAD=true
    fi
  else
    dim "  a remote embedder needs no local model"
  fi

  ask "Neo4j password" "graphrag-local-password" valid_password \
      "at least 8 characters, and no spaces, slashes or \$"
  NEO4J_PASSWORD="$REPLY_TEXT"

  printf '\n'; bold "Writing .env"
  [ -f .env ] && cp .env ".env.backup.$(date +%s)" && dim "  previous .env backed up"
  {
    echo "# Written by scripts/init.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo "# See .env.example for every setting and what it does."
    echo
    echo "NEO4J_URI=bolt://neo4j:7687"
    echo "NEO4J_USER=neo4j"
    echo "NEO4J_PASSWORD=${NEO4J_PASSWORD}"
    echo
    echo "GRAPHRAG_EMBEDDING_BACKEND=${BACKEND}"
    echo "GRAPHRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5"
    echo "GRAPHRAG_EMBEDDING_DIM=384"
    echo "GRAPHRAG_EMBEDDING_ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD}"
    [ -n "$BASE_URL" ] && echo "GRAPHRAG_EMBEDDING_BASE_URL=${BASE_URL}"
    echo
    echo "GRAPHRAG_MCP_HOST=0.0.0.0"
    echo "GRAPHRAG_MCP_PORT=8765"
  } > .env
  ok "  .env written (backend=${BACKEND}, downloads=${ALLOW_DOWNLOAD})"
fi

printf '\n'; bold "Building and starting"
dim   "  first run builds images; a few minutes"
make setup

if [ "$KEEP_ENV" = 0 ] && [ "${ALLOW_DOWNLOAD:-false}" = "true" ]; then
  printf '\n'; bold "Fetching the embedding model"
  make model
fi

printf '\n'; bold "Ready"
echo "  MCP server   http://localhost:8765/mcp"
echo "  Neo4j        http://localhost:7474"
case "$GOAL" in
  1) printf '\n'; echo "  Next, import what you were sent:"
     echo "    make persona-import FILE=path/to/bundle.tar.gz" ;;
  2) printf '\n'; echo "  Next, define a persona and ingest your documents:"
     echo "    docker compose run --rm graphrag graphrag persona new \"My Persona\""
     echo "    # put files under data/raw/<id>/, add a sources: entry, then"
     echo "    make ingest PERSONA=<id> SRC=data/raw/<id>" ;;
  *) printf '\n'; echo "  Try the docs: docs/USAGE.md" ;;
esac
printf '\n'; echo "  Check anything looks wrong with: make doctor"
