#!/usr/bin/env bash
# Manage the pinned embedding model inside the `models` Docker volume.
#   ./scripts/model.sh ensure   warm the cache (downloads on first run) and verify the pin
#   ./scripts/model.sh verify   verify only; non-zero exit if the pin does not match
#   ./scripts/model.sh export F write the whole cache to tarball F (for air-gapped machines)
#   ./scripts/model.sh import F restore a cache tarball into the volume
set -euo pipefail
cd "$(dirname "$0")/.."

pin() { grep -E "^$1[[:space:]]*=" vendor/model.lock | cut -d= -f2- | xargs; }

# Which embedder is configured. Only "fastembed" needs a model on this machine; the http backend
# calls out to another host. Checks the environment first, then .env, as the app itself does.
backend() {
  if [ -n "${GRAPHRAG_EMBEDDING_BACKEND:-}" ]; then echo "$GRAPHRAG_EMBEDDING_BACKEND"; return; fi
  if [ -f .env ]; then
    b=$(grep -E "^GRAPHRAG_EMBEDDING_BACKEND=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"' " )
    if [ -n "$b" ]; then echo "$b"; return; fi
  fi
  echo fastembed
}
SHA="$(pin onnx_sha256)"; REPO="$(pin hf_repo)"; REV="$(pin hf_revision)"; FILE="$(pin onnx_file)"
CACHE="models--${REPO//\//--}"
# Compose prefixes volumes with the project name; override if yours differs.
VOL="${GRAPHRAG_MODEL_VOLUME:-${COMPOSE_PROJECT_NAME:-graph-rag}_models}"

verify() {
  docker run --rm -v "$VOL":/models alpine sh -c "
    f=\$(find /models/$CACHE -type f -name '$SHA' 2>/dev/null | head -1)
    [ -n \"\$f\" ] || { echo 'model blob $SHA not in cache'; exit 1; }
    echo \"verified \$f\"
  "
}

case "${1:-ensure}" in
  ensure)
    if [ "$(backend)" != "fastembed" ]; then
      echo "model: not needed (GRAPHRAG_EMBEDDING_BACKEND=$(backend) does not use a local model)"
      exit 0
    fi
    if verify >/dev/null 2>&1; then echo "model: present and pinned ($REPO@${REV:0:7})"; exit 0; fi
    # Nothing downloads a model implicitly. `make model` sets ALLOW_DOWNLOAD; everything else
    # reports and moves on, so an environment that forbids fetching weights is never surprised.
    if [ -z "${ALLOW_DOWNLOAD:-}" ]; then
      echo "model: NOT present, and nothing will download it."
      echo "  supply it offline:  make model-import FILE=model.tar.gz"
      echo "                      or import a persona bundle exported --with-model"
      echo "  or fetch it:        make model    (~64 MB from Hugging Face)"
      exit 0
    fi
    echo "warming model cache ($REPO@${REV:0:7}); downloading ~64 MB..."
    docker compose run --rm -T graphrag python -c \
      "from graphrag.config import load_settings; from graphrag.embed.base import build_embedder; \
       print('dims', build_embedder(load_settings().embedding).embed_query('warm').shape[0])"
    verify
    ;;
  verify) verify ;;
  export)
    out="${2:?usage: model.sh export <file.tar.gz>}"
    docker run --rm -v "$VOL":/models -v "$(cd "$(dirname "$out")" && pwd)":/out alpine \
      tar czf "/out/$(basename "$out")" -C /models .
    echo "wrote $out ($(du -h "$out" | cut -f1))"
    ;;
  import)
    in="${2:?usage: model.sh import <file.tar.gz>}"
    docker volume create "$VOL" >/dev/null
    docker run --rm -v "$VOL":/models -v "$(cd "$(dirname "$in")" && pwd)":/in alpine \
      tar xzf "/in/$(basename "$in")" -C /models
    verify
    ;;
  *) echo "unknown command: $1" >&2; exit 2 ;;
esac
