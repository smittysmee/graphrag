#!/usr/bin/env bash
# Manage the pinned embedding model inside the `models` Docker volume.
#   ./scripts/model.sh ensure   warm the cache (downloads on first run) and verify the pin
#   ./scripts/model.sh verify   verify only; non-zero exit if the pin does not match
#   ./scripts/model.sh export F write the whole cache to tarball F (for air-gapped machines)
#   ./scripts/model.sh import F restore a cache tarball into the volume
set -euo pipefail
cd "$(dirname "$0")/.."

pin() { grep -E "^$1[[:space:]]*=" vendor/model.lock | cut -d= -f2- | xargs; }
SHA="$(pin onnx_sha256)"; REPO="$(pin hf_repo)"; REV="$(pin hf_revision)"; FILE="$(pin onnx_file)"
CACHE="models--${REPO//\//--}"
VOL=graph-rag_models

verify() {
  docker run --rm -v "$VOL":/models alpine sh -c "
    f=\$(find /models/$CACHE -type f -name '$SHA' 2>/dev/null | head -1)
    [ -n \"\$f\" ] || { echo 'model blob $SHA not in cache'; exit 1; }
    echo \"verified \$f\"
  "
}

case "${1:-ensure}" in
  ensure)
    if verify >/dev/null 2>&1; then echo "model already present and pinned ($REPO@${REV:0:7})"; exit 0; fi
    echo "warming model cache ($REPO@${REV:0:7}); first run downloads ~64 MB..."
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
