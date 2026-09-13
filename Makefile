# Every target runs inside Docker. On the host, targets delegate to the `dev` service;
# inside the container (IN_CONTAINER=1) the *-local targets run the tools directly.
SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE     := docker compose
DEV         := $(COMPOSE) --profile dev run --rm -T dev
PERSONA     ?= product-leader
SRC         ?= data/raw/$(PERSONA)
Q           ?= how do I find product market fit
K           ?= 5

.PHONY: help build up down logs setup doctor hooks check check-local fmt lint type test \
        test-integration test-integration-local ingest sync snapshot-export snapshot-load search \
        context stats shell lsp embed-up enrich clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ lifecycle
build: ## Build runtime + dev images
	$(COMPOSE) --profile dev --profile test build

up: ## Start Neo4j + MCP server (http://localhost:8765/mcp)
	$(COMPOSE) up -d --wait neo4j graphrag

down: ## Stop everything (keeps volumes)
	$(COMPOSE) --profile dev --profile test --profile embed down

logs: ## Tail app + neo4j logs
	$(COMPOSE) logs -f graphrag neo4j

init: ## Interactive first-run setup: asks a few questions, writes .env, runs setup
	@./scripts/init.sh

model: ## Fetch the pinned embedding model. Explicit opt-in: nothing else downloads it.
	@ALLOW_DOWNLOAD=1 ./scripts/model.sh ensure

model-check: ## Report whether the pinned model is present, without fetching anything
	@./scripts/model.sh ensure

model-export: ## Tar the model cache for an air-gapped machine (FILE=model.tar.gz)
	./scripts/model.sh export $(or $(FILE),model.tar.gz)

model-import: ## Restore a model cache tarball (FILE=model.tar.gz)
	./scripts/model.sh import $(or $(FILE),model.tar.gz)

sources: ## Fetch the pinned source archives (only needed to ingest them yourself)
	git submodule update --init --recursive

setup: hooks build model-check ## One-shot: hooks, images, Neo4j, snapshots, MCP up (no download)
	$(COMPOSE) up -d --wait neo4j
	$(COMPOSE) run --rm -T graphrag graphrag setup
	$(COMPOSE) up -d --wait graphrag
	@echo "MCP server ready at http://localhost:8765/mcp  (Neo4j browser: http://localhost:7474)"

doctor: ## Check Neo4j, embedder backend, snapshots
	$(COMPOSE) run --rm -T graphrag graphrag doctor

hooks: ## Enable the versioned git hooks (.githooks)
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "git hooks enabled (core.hooksPath=.githooks)"

# ------------------------------------------------------------------ quality
ifeq ($(IN_CONTAINER),1)
check: check-local
test: test-local
fmt: fmt-local
lint: lint-local
type: type-local
else
check: ## Format check + lint + mypy + unit tests (in Docker)
	$(DEV) make check-local
test: ## Unit tests (in Docker)
	$(DEV) make test-local
fmt: ## Auto-format with ruff (in Docker)
	$(DEV) make fmt-local
lint: ## Ruff lint (in Docker)
	$(DEV) make lint-local
type: ## mypy --strict (in Docker)
	$(DEV) make type-local
endif

check-local: ## (container) ruff format --check, ruff check, mypy, pytest
	ruff format --check src tests
	ruff check src tests
	mypy
	pytest -q -m "not integration and not embedding"

fmt-local:
	ruff format src tests
	ruff check --fix src tests

lint-local:
	ruff check src tests

type-local:
	mypy

test-local:
	pytest -q -m "not integration and not embedding"

test-integration: ## Integration tests against a throwaway Neo4j (in Docker)
	$(COMPOSE) --profile test run --rm -T test make test-integration-local; \
	status=$$?; $(COMPOSE) --profile test rm -sf neo4j-test test >/dev/null 2>&1; exit $$status
	# NOTE: never `down -v` here; that would delete the main neo4j_data and models volumes.

test-integration-local:
	pytest -q -m "integration"

# ------------------------------------------------------------------ data
ingest: ## Ingest SRC into PERSONA (SRC=repo-relative path PERSONA=id [SOURCE=id]) and export
	$(COMPOSE) run --rm -T graphrag graphrag ingest /app/$(SRC) --persona $(PERSONA) \
		$(if $(SOURCE),--source $(SOURCE)) --export

sync: ## Ingest whatever is missing for PERSONA and re-import its enrichment JSON
	$(COMPOSE) run --rm -T graphrag graphrag sync $(PERSONA) $(if $(SOURCE),--source $(SOURCE))

snapshot-export: ## Export PERSONA from Neo4j to data/snapshots/PERSONA
	$(COMPOSE) run --rm -T graphrag graphrag snapshot export $(PERSONA)

snapshot-load: ## Load committed snapshots into Neo4j
	$(COMPOSE) run --rm -T graphrag graphrag snapshot load --all

# Bundles live under data/exports because only ./data is mounted into the container.
EXPORT_DIR := data/exports

persona-export: ## Pack a persona into data/exports/ (PERSONA=id [COMPACT=1] [WITH_MODEL=1])
	@mkdir -p $(EXPORT_DIR)
	$(COMPOSE) run --rm -T graphrag graphrag persona export $(PERSONA) \
		-o /app/$(EXPORT_DIR)/$(or $(FILE),$(PERSONA)-persona.tar.gz) \
		$(if $(COMPACT),--compact,) $(if $(WITH_MODEL),--with-model,)
	@echo "→ $(EXPORT_DIR)/$(or $(FILE),$(PERSONA)-persona.tar.gz)"

persona-import: ## Install a bundle and load it (FILE=path [OVERWRITE=1])
	$(COMPOSE) run --rm -T graphrag graphrag persona import /app/$(FILE) --load $(if $(OVERWRITE),--overwrite,)

enrich: ## Claude entity/claim enrichment for PERSONA (LIMIT=n documents)
	$(COMPOSE) run --rm -T graphrag graphrag enrich $(PERSONA) --limit $(or $(LIMIT),10)

search: ## Hybrid search: make search Q="..." PERSONA=... K=5
	$(COMPOSE) run --rm -T graphrag graphrag search "$(Q)" --persona $(PERSONA) -k $(K)

context: ## Context pack for an agent: make context Q="..." PERSONA=...
	$(COMPOSE) run --rm -T graphrag graphrag context "$(Q)" --persona $(PERSONA)

stats: ## Graph statistics
	$(COMPOSE) run --rm -T graphrag graphrag stats

# ------------------------------------------------------------------ dev ergonomics
shell: ## Shell in the dev container
	$(COMPOSE) --profile dev run --rm dev bash

lsp: ## Run the Python LSP (pylsp) over stdio for your editor
	@$(COMPOSE) --profile dev run --rm -i lsp

embed-up: ## Start the standalone embedding server (:8766). On a GB10 add -f docker-compose.gpu.yml
	$(COMPOSE) --profile embed up -d --wait embed

clean: ## Remove caches
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
