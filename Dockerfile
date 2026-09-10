# syntax=docker/dockerfile:1.7
# Single image, two targets:
#   runtime : MCP server / CLI / embed server (no dev tools)
#   dev     : + ruff, mypy, pytest, LSP; used by `make check`, git hooks, and editors
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    IN_CONTAINER=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# make (task runner) and git (records the source commit in snapshot manifests)
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached layer), project second.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Optional GPU build for the embedding server (e.g. on a GB10 / DGX Spark):
#   docker build --build-arg EMBED_GPU=1 --target runtime -t graphrag:gpu .
ARG EMBED_GPU=0
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$EMBED_GPU" = "1" ]; then \
      uv pip install --python /app/.venv/bin/python "fastembed-gpu>=0.8,<0.9"; \
    fi

# ------------------------------------------------------------------ runtime
FROM base AS runtime
COPY personas ./personas
COPY data ./data
EXPOSE 8765 8766
CMD ["graphrag", "serve", "--transport", "http"]

# ------------------------------------------------------------------ dev
FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --group lsp
COPY tests ./tests
COPY Makefile ./
COPY personas ./personas
COPY data ./data
CMD ["make", "check-local"]
