"""FastMCP server exposing the graph to agents.

Tools are thin: they validate arguments, call the same ``Retriever``/``GraphStore`` the CLI uses,
and return JSON-serialisable dicts. ``AppContext`` is created lazily so importing this module
(e.g. in tests) never opens a Neo4j connection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from graphrag.app import AppContext
from graphrag.config import Settings
from graphrag.graph import snapshot as snap
from graphrag.models import SearchMode
from graphrag.personas.brief import build_brief
from graphrag.personas.registry import SDLC_STAGES
from graphrag.sna.arguments import (
    DEFAULT_MAX_SECONDS,
    parse_max_seconds,
    parse_sample_params,
    parse_types,
    parse_where,
)


#: ``sna analyze --k-range`` in a form a tool caller can hand in natively: ``"2-10"`` or
#: ``"3,5,8"``. Kept here rather than imported from :mod:`graphrag.cli` because that module's own
#: parser raises ``typer.BadParameter`` (a ``click`` exception, not a ``ValueError``), and every
#: refusal below is turned into a structured error by catching ``ValueError`` alone.
def _parse_k_range(text: str) -> list[int]:
    try:
        if "-" in text:
            low, high = (int(part) for part in text.split("-", 1))
            return list(range(low, high + 1))
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"k_range must look like 2-10 or 3,5,8, got {text!r}"
        raise ValueError(msg) from exc


#: ``sna spread --seeds``, native to a tool caller: names, comma separated, or a bare count drawn
#: at random. Mirrors ``graphrag.cli._seeds_or_exit`` for the same reason ``_parse_k_range`` does.
def _parse_seeds(value: str | None) -> list[str] | int | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        msg = "seeds was empty: name the patient zeros, or give a count"
        raise ValueError(msg)
    if len(names) == 1 and names[0].isdigit():
        return int(names[0])
    return names


INSTRUCTIONS = """graphrag: persona-grounded knowledge graph.
Start with `list_personas`, then `persona_brief(persona_id)` to adopt a persona, then
`context(query, persona_id)` for cited passages before answering. `search` is the raw hybrid
search; `documents`/`topics`/`speakers` browse the corpus; `read_document` pages through a
source; `cypher` runs read-only Cypher for anything else."""


class ServerState:
    def __init__(self, settings: Settings | None = None, context: AppContext | None = None) -> None:
        self._settings = settings
        self._context = context

    @property
    def ctx(self) -> AppContext:
        if self._context is None:
            self._context = AppContext.build(self._settings)
        return self._context


def create_server(state: ServerState | None = None) -> FastMCP:
    st = state or ServerState()
    mcp = FastMCP("graphrag", instructions=INSTRUCTIONS)

    # ------------------------------------------------------------- personas
    @mcp.tool
    def list_personas() -> list[dict[str, Any]]:
        """Personas available in the graph, with their SDLC stages and source descriptions."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "sdlc_stages": p.sdlc_stages,
                "sources": [s.description or s.url or s.id for s in p.sources],
            }
            for p in st.ctx.registry.all()
        ]

    @mcp.tool
    def persona_brief(persona_id: str) -> str:
        """Role prompt + grounding summary to adopt before answering as a persona."""
        return build_brief(st.ctx.registry.get(persona_id), st.ctx.store)

    @mcp.tool
    def recommend_personas(sdlc_stage: str) -> dict[str, Any]:
        """Which personas apply to an SDLC stage (discovery, requirements, design, testing, ...)."""
        matches = st.ctx.registry.recommend(sdlc_stage)
        return {
            "stage": sdlc_stage,
            "known_stages": SDLC_STAGES,
            "personas": [
                {"id": p.id, "name": p.name, "description": p.description} for p in matches
            ],
        }

    # ------------------------------------------------------------- retrieval
    @mcp.tool
    def context(
        query: str,
        persona_id: str | None = None,
        k: int | None = None,
        expand: int | None = None,
    ) -> dict[str, Any]:
        """Cited context pack for a question. Returns prompt-ready text plus structured hits."""
        persona = st.ctx.registry.get(persona_id) if persona_id else None
        pack = st.ctx.retriever.context(query, persona=persona, k=k, expand=expand)
        return {
            "prompt": pack.to_prompt(),
            "hits": [
                {
                    "n": i,
                    "citation": h.citation(),
                    "doc_id": h.document.id,
                    "chunk_id": h.chunk.id,
                    "speaker": h.chunk.speaker,
                    "url": h.chunk.url or h.document.url,
                    "score": round(h.score, 5),
                    "text": h.passage(),
                }
                for i, h in enumerate(pack.hits, start=1)
            ],
            "topics": pack.topics,
            "entities": [e.model_dump() for e in pack.entities],
        }

    @mcp.tool
    def search(
        query: str, persona_id: str | None = None, k: int = 8, mode: SearchMode = "hybrid"
    ) -> list[dict[str, Any]]:
        """Raw hybrid (vector + full-text) search over passages."""
        hits = st.ctx.retriever.search(query, persona_id=persona_id, k=k, mode=mode)
        return [
            {
                "chunk_id": h.chunk.id,
                "doc_id": h.document.id,
                "title": h.document.title,
                "speaker": h.chunk.speaker,
                "start_ts": h.chunk.start_ts,
                "url": h.chunk.url or h.document.url,
                "score": round(h.score, 5),
                "methods": h.methods,
                "text": h.chunk.text,
            }
            for h in hits
        ]

    # ------------------------------------------------------------- browsing
    @mcp.tool
    def documents(
        persona_id: str | None = None,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """List documents, optionally filtered by topic or speaker."""
        return [
            {
                "id": d.id,
                "title": d.title,
                "published": d.published.isoformat() if d.published else None,
                "url": d.url,
                "speakers": d.speakers,
                "topics": d.topics,
                "description": d.description[:300],
            }
            for d in st.ctx.store.list_documents(
                persona_id, topic=topic, speaker=speaker, limit=limit
            )
        ]

    @mcp.tool
    def get_document(doc_id: str) -> dict[str, Any]:
        """Full metadata for one document."""
        doc = st.ctx.store.get_document(doc_id)
        if doc is None:
            return {"error": f"unknown document {doc_id}"}
        return doc.model_dump(mode="json")

    @mcp.tool
    def read_document(doc_id: str, start: int = 0, count: int = 5) -> list[dict[str, Any]]:
        """Page through a document's passages in order (start = passage ordinal)."""
        return [
            {
                "ordinal": c.ordinal,
                "speaker": c.speaker,
                "start_ts": c.start_ts,
                "url": c.url,
                "text": c.text,
            }
            for c in st.ctx.store.document_chunks(doc_id, start, count)
        ]

    @mcp.tool
    def topics(persona_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Topics ranked by how many documents carry them."""
        return [t.model_dump() for t in st.ctx.store.list_topics(persona_id, limit)]

    @mcp.tool
    def related_topics(
        topic: str, persona_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Topics that co-occur with a topic (graph adjacency), strongest first."""
        return [t.model_dump() for t in st.ctx.store.related_topics(topic, persona_id, limit)]

    @mcp.tool
    def speakers(persona_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Speakers/authors ranked by number of documents."""
        return [s.model_dump() for s in st.ctx.store.list_speakers(persona_id, limit)]

    # ------------------------------------------------------------- escape hatches
    @mcp.tool
    def cypher(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run read-only Cypher (MATCH/RETURN). Labels: Persona, Source, Document, Chunk, Speaker,
        Topic, Entity. Relationships: GROUNDED_BY, CONTAINS, HAS_CHUNK, NEXT, SPOKE, FEATURES,
        ABOUT, CO_OCCURS, MENTIONS, RELATED_TO. Max 500 rows."""
        return st.ctx.store.run_readonly_cypher(query, params)

    @mcp.tool
    def stats() -> dict[str, Any]:
        """Node counts overall and per persona, plus committed snapshots."""
        return {
            "graph": st.ctx.store.stats().model_dump(),
            "snapshots": [
                snap.manifest_summary(m) for m in snap.list_snapshots(st.ctx.snapshots_dir)
            ],
            "embedding": {
                "backend": st.ctx.settings.embedding.backend,
                "model": st.ctx.settings.embedding.model,
                "dim": st.ctx.settings.embedding.dim,
            },
        }

    # ------------------------------------------------------------- sna
    #
    # One tool per `graphrag sna <command>` (docs/ATLAS_PLAN.md, ATL-ENT-1). Each tool below calls
    # exactly the functions that command's own body calls -- the same `graphrag.sna.export.
    # build_network`, the same report/`render_*`/`*_payload` triplet -- so nothing here is a
    # second implementation of a number the CLI prints. Two differences from the CLI, both because
    # a tool call is not a terminal session that exits right after:
    #
    # - A CLI command opens its own `AppContext` and closes it when it returns. This server holds
    #   one `AppContext` open across every call (`st.ctx`), the way every other tool in this file
    #   already does, so every `sna_*` tool reads `st.ctx.store` directly and never closes it --
    #   none of them go through `graphrag.sna.facade.Network.build`, whose `AppContext.build(...)
    #   / ctx.close()` pattern would close the shared store out from under the next call.
    # - `--out`/`--json`/`--html`/`--report` are optional here even where the CLI requires them
    #   (`sna export`, `sna analyze`, `sna draw`, `sna multilayer-communities`): the point of a
    #   tool call is the returned `markdown`/`payload`, computed and returned regardless of
    #   whether a path was given. Passing one *also* writes the file, exactly as the CLI would,
    #   for a caller that has filesystem access to the container.
    #
    # A refusal the CLI prints in red and exits 2 for -- an unknown network, method or persona, a
    # filter combination `build_network` rejects, `graphrag[gnn]`'s torch missing -- comes back
    # here as ``{"error": "..."}`` rather than an exception.

    def _err(message: str) -> dict[str, Any]:
        return {"error": message}

    def _persona_error(persona_id: str) -> dict[str, Any] | None:
        try:
            st.ctx.registry.get(persona_id)
        except KeyError as exc:
            return _err(str(exc))
        return None

    def _maybe_write(path: Path | None, text: str) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def _maybe_write_json(path: Path | None, payload: Any) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _with_provenance(markdown: str, provenance: Any) -> str:
        """``markdown`` with `## Provenance` appended (ATL-ENT-4): no `--cite` here, unlike the
        CLI -- a tool caller already has :attr:`Provenance.chapters` in the payload and can look
        the references up itself; the block still has to match the CLI's for the same call, so
        the render step is the one line this and :func:`_payload_with_provenance` share.
        """
        from graphrag.sna.provenance import render_provenance

        lines = [markdown.rstrip(), "", *render_provenance(provenance)]
        return "\n".join(lines).rstrip() + "\n"

    def _payload_with_provenance(payload: dict[str, Any], provenance: Any) -> dict[str, Any]:
        from graphrag.sna.provenance import provenance_payload

        payload["provenance"] = provenance_payload(provenance)
        return payload

    def _snapshot_commit(persona_id: str) -> str | None:
        from graphrag.sna.provenance import snapshot_commit_for

        return snapshot_commit_for(st.ctx.settings, persona_id)

    def _build(
        network: str,
        persona_id: str,
        *,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        backbone: str | None = None,
        alpha: float = 0.05,
        correction: str = "none",
        threshold: float | None = None,
    ) -> Any:
        from graphrag.sna.cache import build_network_cached

        return build_network_cached(
            st.ctx.store,
            network,
            persona_id,
            settings=st.ctx.settings,
            source_id=source,
            min_weight=min_weight,
            types=parse_types(types),
            stances=stance,
            facets=facet,
            relation_types=relation_type,
            since=since,
            until=until,
            where=parse_where(where),
            projection=projection,
            lam=lam,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
            project=project,
        )

    @mcp.tool
    def sna_sample(
        persona: str,
        size: int,
        network: str = "speakers",
        method: str = "random-walk",
        params: dict[str, Any] | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        types: list[str] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna sample`: take a smaller network out of a bigger one and say what that
        cost (Atlas ch. 29): `graphrag.sna.sampling.sample`, `bias_report`, `completion`,
        `render_sample`, `sample_payload` -- the same functions the CLI calls.

        `params` is `--param name=value`, repeated, already parsed into a mapping (`k`, `p`,
        `restart`, `rounds`, `neighbors`; see the CLI's own docstring for which sampler reads
        which). Unlike `graphrag sna sample`, this never writes the sampled graph to a file --
        there is no `--out` to name a format with here; export the sample's own node set with
        `sna_export` (built with the same filters and a `--where`/id filter on what the sample
        kept) if a file is wanted.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.sampling import (
            SAMPLE_METHODS,
            bias_report,
            completion,
            render_sample,
            sample_payload,
        )
        from graphrag.sna.sampling import sample as take_sample

        if method not in SAMPLE_METHODS:
            return _err(f"method must be one of {', '.join(SAMPLE_METHODS)}, got {method!r}")
        try:
            population = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
            )
            sampled = take_sample(population, method, size, seed=seed, **(params or {}))
        except ValueError as exc:
            return _err(str(exc))
        bias = bias_report(population, sampled)
        estimate = completion(sampled)
        prov = make_provenance(
            method="sample",
            parameters={
                "persona": persona,
                "size": size,
                "network": network,
                "method": method,
                "param": parse_sample_params(params),
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "types": parse_types(types),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["sample"],
            snapshot_commit=_snapshot_commit(persona),
        )
        return {
            "markdown": _with_provenance(render_sample(bias, estimate), prov),
            "payload": _payload_with_provenance(sample_payload(bias, estimate), prov),
        }

    @mcp.tool
    def sna_export(
        persona: str,
        network: str = "speakers",
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        backbone: str | None = None,
        alpha: float = 0.05,
        correction: str = "none",
        threshold: float | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        layers: str | None = None,
        omega: float = 1.0,
        out: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna export`: one of the graph's networks as node-link JSON.

        `payload` is `networkx.node_link_data(graph, edges="links")` -- the exact structure the
        CLI writes for a `.json` target -- so a caller that wants `.graphml`/`.gexf`/`.net`/
        `.csv`/`.edgelist` instead should pass `out` with that suffix, which also writes the file
        (`graphrag.sna.export.write_graph`, the CLI's own writer); `markdown` is the CLI's
        one-line confirmation. `layers` composes as the CLI's `--layers` does (the flattening of
        the multilayer build, `graphrag.sna.layers.multilayer`/`flatten`) and cannot be combined
        with `backbone`, which filters one network rather than building several.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        import networkx as nx

        from graphrag.sna.export import _UNWRITTEN_GRAPH_KEYS, write_graph
        from graphrag.sna.layers import flatten, multilayer
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, provenance_payload

        try:
            if layers is not None:
                if backbone is not None:
                    msg = (
                        "backbone filters one network; layers builds one per value and sums "
                        "them, so the two cannot be combined. Export the flattening first, then "
                        "backbone it"
                    )
                    return _err(msg)
                stack = multilayer(
                    st.ctx.store,
                    persona,
                    network,
                    layers,  # type: ignore[arg-type]
                    source_id=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    relation_types=relation_type,
                    since=since,
                    until=until,
                    where=parse_where(where),
                    project=project,  # type: ignore[arg-type]
                    omega=omega,
                )
                graph = flatten(stack)
            else:
                graph = _build(
                    network,
                    persona,
                    source=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    backbone=backbone,
                    alpha=alpha,
                    correction=correction,
                    threshold=threshold,
                    types=parse_types(types),
                    stance=stance,
                    facet=facet,
                    relation_type=relation_type,
                    since=since,
                    until=until,
                    where=parse_where(where),
                    project=project,
                )
        except ValueError as exc:
            return _err(str(exc))
        prov = make_provenance(
            method="export",
            parameters={
                "persona": persona,
                "network": network,
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "backbone": backbone,
                "alpha": alpha,
                "correction": correction,
                "threshold": threshold,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
                "layers": layers,
                "omega": omega,
            },
            chapters=COMMAND_CHAPTERS["export"],
            snapshot_commit=_snapshot_commit(persona),
        )
        # Scalar, not the dict `provenance_payload` returns: GraphML/GEXF attributes cannot be
        # nested (see `_UNWRITTEN_GRAPH_KEYS`'s own reason), so this rides along as a JSON string
        # a caller reads back with `json.loads` -- matching the CLI's own `sna export`.
        graph.graph["provenance"] = json.dumps(provenance_payload(prov), sort_keys=True)
        if out is not None:
            write_graph(graph, out)
        prefix = f"wrote {out}: " if out is not None else ""
        markdown = (
            f"{prefix}{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges "
            f"({network})"
        )
        # `write_graph`'s node-link branch pops the projection's own inputs (the two-mode pairs,
        # the right-mode attribute table) before serialising, because no export format has a
        # type for them; mirrored here so `payload` is exactly what a `.json` --out would hold.
        kept = {key: graph.graph.pop(key) for key in _UNWRITTEN_GRAPH_KEYS if key in graph.graph}
        payload = nx.node_link_data(graph, edges="links")
        # `payload["graph"]` is `graph.graph` itself, not a copy (networkx hands the same dict
        # back) -- detach it before restoring the popped keys below, or restoring would mutate
        # the payload right back to holding them.
        payload["graph"] = dict(payload["graph"])
        graph.graph.update(kept)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_analyze(
        persona: str,
        network: str = "speakers",
        method: str = "louvain",
        k: int | None = None,
        k_range: str = "2-10",
        resolution: float = 1.0,
        runs: int = 10,
        features: str = "spectral",
        dims: int = 8,
        p: float = 1.0,
        q: float = 1.0,
        covariance: str = "full",
        samples: int = 50,
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        backbone: str | None = None,
        alpha: float = 0.05,
        correction: str = "none",
        threshold: float | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        by: str | None = None,
        permutations: int = 200,
        null: str = "permutation",
        project: str | None = None,
        sample_method: str | None = None,
        sample_size: int | None = None,
        layers: str | None = None,
        omega: float = 1.0,
        uncertain: bool = False,
        uncertain_samples: int = 200,
        uncertain_expensive: bool = False,
        uncertain_max_seconds: float = DEFAULT_MAX_SECONDS,
        degree_corrected: bool = True,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna analyze`: measure a network, group it, check the grouping (the CLI's
        flagship report). Calls `graphrag.sna.analysis.run_analysis`, then `render_markdown` for
        `markdown` and `to_payload` for `payload` -- see the CLI command's own docstring
        (`graphrag sna analyze --help`) for what every section and every flag means; this tool
        takes the identical set, at the identical defaults.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.extract.attributes import load_persona_attributes
        from graphrag.sna.analysis import METHODS, render_markdown, run_analysis, to_payload
        from graphrag.sna.attributes import ATTRIBUTE_NULLS
        from graphrag.sna.layers import Multilayer, flatten, multilayer
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.sampling import SAMPLE_METHODS
        from graphrag.sna.sampling import sample as take_sample

        if method not in METHODS:
            return _err(f"method must be one of {', '.join(METHODS)}, got {method!r}")
        if sample_method is not None:
            if sample_method not in SAMPLE_METHODS:
                return _err(f"sample_method must be one of {', '.join(SAMPLE_METHODS)}")
            if sample_size is None or sample_size < 1:
                return _err("sample_method needs sample_size, a node count of at least 1")
        if null not in ATTRIBUTE_NULLS:
            return _err(f"null must be one of {', '.join(ATTRIBUTE_NULLS)}")

        try:
            k_values = _parse_k_range(k_range)
            stack: Multilayer | None = None
            if layers is not None:
                stack = multilayer(
                    st.ctx.store,
                    persona,
                    network,
                    layers,  # type: ignore[arg-type]
                    source_id=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    relation_types=relation_type,
                    since=since,
                    until=until,
                    where=parse_where(where),
                    project=project,  # type: ignore[arg-type]
                    omega=omega,
                )
            graph = (
                flatten(stack)
                if stack is not None
                else _build(
                    network,
                    persona,
                    source=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    backbone=backbone,
                    alpha=alpha,
                    correction=correction,
                    threshold=threshold,
                    types=parse_types(types),
                    stance=stance,
                    facet=facet,
                    relation_type=relation_type,
                    since=since,
                    until=until,
                    where=parse_where(where),
                    project=project,
                )
            )
            if sample_method is not None and sample_size is not None:
                graph = take_sample(graph, sample_method, sample_size, seed=seed)
            analysis = run_analysis(
                st.ctx.store,
                graph,
                persona_id=persona,
                network=network,
                method=method,
                k=k,
                k_range=k_values,
                resolution=resolution,
                runs=runs,
                features=features,  # type: ignore[arg-type]
                dims=dims,
                p=p,
                q=q,
                covariance=covariance,
                samples=samples,
                by=by.strip() if by else None,
                permutations=permutations,
                null=null,
                seed=seed,
                multilayer=stack,
                vocabulary=load_persona_attributes(st.ctx.registry.directory, persona),
                uncertain=uncertain,
                uncertain_samples=uncertain_samples,
                uncertain_expensive=uncertain_expensive,
                uncertain_max_seconds=parse_max_seconds(uncertain_max_seconds),
                sbm_degree_corrected=degree_corrected,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="analyze",
            parameters={
                "persona": persona,
                "network": network,
                "method": method,
                "k": k,
                "k_range": k_range,
                "resolution": resolution,
                "runs": runs,
                "features": features,
                "dims": dims,
                "p": p,
                "q": q,
                "covariance": covariance,
                "samples": samples,
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "backbone": backbone,
                "alpha": alpha,
                "correction": correction,
                "threshold": threshold,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "by": by,
                "permutations": permutations,
                "null": null,
                "project": project,
                "sample_method": sample_method,
                "sample_size": sample_size,
                "layers": layers,
                "omega": omega,
                "uncertain": uncertain,
                "uncertain_samples": uncertain_samples,
                "uncertain_expensive": uncertain_expensive,
                "uncertain_max_seconds": parse_max_seconds(uncertain_max_seconds),
                "degree_corrected": degree_corrected,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["analyze"],
            null_model=(
                "degree-preserving rewiring (configuration model)"
                if analysis.null_model is not None
                else None
            ),
            null_model_samples=(
                analysis.null_model.samples if analysis.null_model is not None else None
            ),
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_markdown(analysis), prov)
        payload = _payload_with_provenance(to_payload(analysis), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_stances(
        persona: str,
        entity: list[str] | None = None,
        source: str | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        quotes: int = 3,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna stances`: what the corpus says about each entity, not merely how often
        it names it -- `graphrag.sna.stances.build_stance_report`/`render_stances`/
        `stances_payload`, the same functions the CLI calls (ATL-F1: `--out`/`--json` write the
        same `markdown`/`payload` this tool returns).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.stances import build_stance_report, render_stances, stances_payload

        report = build_stance_report(
            st.ctx.store,
            persona,
            source_id=source,
            entities=entity or [],
            facets=facet or [],
            where=parse_where(where),
            quotes_per_stance=max(0, quotes),
        )
        prov = make_provenance(
            method="stances",
            parameters={
                "persona": persona,
                "entity": entity,
                "source": source,
                "facet": facet,
                "where": parse_where(where),
                "quotes": quotes,
            },
            chapters=COMMAND_CHAPTERS["stances"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_stances(report), prov)
        _maybe_write(out, markdown)
        payload = _payload_with_provenance(stances_payload(persona, report), prov)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_compare(
        persona: str,
        network: str = "speakers",
        since: str | None = None,
        until: str | None = None,
        since2: str | None = None,
        until2: str | None = None,
        window: str | None = None,
        step: str | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        backbone: str | None = None,
        alpha: float = 0.05,
        correction: str = "none",
        threshold: float | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        where2: dict[str, str] | None = None,
        project: str | None = None,
        centrality: str = "weighted_degree",
        resolution: float = 1.0,
        runs: int = 10,
        seed: int | None = None,
        topology: bool = False,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna compare`: build one network over two windows (or two `where` populations)
        and report what changed (`graphrag.sna.compare.compare_windows`/`render_comparison`/
        `compare_payload`, the same functions the CLI calls; ATL-F1: `--out`/`--json` write the
        same `markdown`/`payload` this tool returns).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.compare import compare_payload, compare_windows, render_comparison
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            comparison = compare_windows(
                st.ctx.store,
                persona,
                network,
                since=since,
                until=until,
                since2=since2,
                until2=until2,
                window=window,
                step=step,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stance,
                facets=facet,
                where=parse_where(where),
                where2=parse_where(where2, "--where2"),
                projection=projection,
                lam=lam,
                backbone=backbone,
                alpha=alpha,
                correction=correction,
                threshold=threshold,
                project=project,  # type: ignore[arg-type]
                centrality_kind=centrality,
                resolution=resolution,
                runs=runs,
                seed=seed,
                topology=topology,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="compare",
            parameters={
                "persona": persona,
                "network": network,
                "since": since,
                "until": until,
                "since2": since2,
                "until2": until2,
                "window": window,
                "step": step,
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "backbone": backbone,
                "alpha": alpha,
                "correction": correction,
                "threshold": threshold,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
                "where2": parse_where(where2, "--where2"),
                "project": project,
                "centrality": centrality,
                "resolution": resolution,
                "runs": runs,
                "topology": topology,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["compare"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_comparison(comparison), prov)
        _maybe_write(out, markdown)
        payload = _payload_with_provenance(compare_payload(comparison), prov)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_layers(
        persona: str,
        network: str = "entities",
        layers: str = "stance",
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        types: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        omega: float = 1.0,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna layers`: read one network as a multilayer network -- one layer per value
        of `layers`, and what each holds (Atlas §7.2, §8.1). Nothing here has a null model: this
        is a description of what `graphrag.sna.layers.multilayer` built, not a test.

        `graphrag.sna.layers.multilayer`/`render_layers`/`layers_payload`, the same functions the
        CLI calls (ATL-F1: `--json` writes the same `payload` this tool returns).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.layers import layers_payload, multilayer, render_layers
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            stack = multilayer(
                st.ctx.store,
                persona,
                network,
                layers,  # type: ignore[arg-type]
                source_id=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
                facets=facet,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,  # type: ignore[arg-type]
                omega=omega,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="layers",
            parameters={
                "persona": persona,
                "network": network,
                "layers": layers,
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "types": parse_types(types),
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
                "omega": omega,
            },
            chapters=COMMAND_CHAPTERS["layers"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_layers(stack)), prov)
        _maybe_write(out, markdown)
        payload = _payload_with_provenance(layers_payload(persona, network, layers, stack), prov)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_multilayer_communities(
        persona: str,
        network: str = "entities",
        layers: str = "stance",
        threshold: int = 3,
        omega: float = 1.0,
        gamma: float = 1.0,
        resolution: float = 1.0,
        runs: int = 10,
        samples: int = 20,
        source: str | None = None,
        min_weight: int | None = None,
        projection: str = "simple",
        lam: float = 0.5,
        types: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna multilayer-communities`: chapter 40 over one multilayer network --
        flattening, layer-by-layer, supra modularity, and which to prefer.
        `graphrag.sna.multilayer.analyze_multilayer_communities`/`render_multilayer_communities`/
        `multilayer_communities_payload`, the same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.layers import multilayer
        from graphrag.sna.multilayer import (
            analyze_multilayer_communities,
            multilayer_communities_payload,
            render_multilayer_communities,
        )
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            stack = multilayer(
                st.ctx.store,
                persona,
                network,
                layers,  # type: ignore[arg-type]
                source_id=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
                facets=facet,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,  # type: ignore[arg-type]
                omega=omega,
            )
            if len(stack) < 2:
                return _err(f"multilayer-communities needs at least two layers; got {len(stack)}")
            report = analyze_multilayer_communities(
                stack,
                threshold=threshold,
                omega=omega,
                gamma=gamma,
                resolution=resolution,
                runs=runs,
                samples=samples,
                seed=seed,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="multilayer-communities",
            parameters={
                "persona": persona,
                "network": network,
                "layers": layers,
                "threshold": threshold,
                "omega": omega,
                "gamma": gamma,
                "resolution": resolution,
                "runs": runs,
                "samples": samples,
                "source": source,
                "min_weight": min_weight,
                "projection": projection,
                "lam": lam,
                "types": parse_types(types),
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["multilayer-communities"],
            null_model=report.supra_null.null or "per-layer rewiring null (§40.3)",
            null_model_samples=report.supra_null.samples,
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_multilayer_communities(report)), prov)
        payload = _payload_with_provenance(multilayer_communities_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_projections(
        persona: str,
        network: str = "entities",
        project: str | None = None,
        source: str | None = None,
        min_weight: int | None = 1,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        scheme: list[str] | None = None,
        lam: float = 0.5,
        keep_top: float = 0.1,
        threshold: float | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna projections`: project one network every way chapter 26 offers, and report
        where the schemes disagree (§26.6). `graphrag.sna.projection.compare_schemes`/
        `render_projections`/`projections_payload`, the same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.null import pairs_of, projected_side
        from graphrag.sna.projection import SCHEMES, compare_schemes, projections_payload
        from graphrag.sna.projection import render_projections as render_projections_of
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        names = [s for s in (scheme or []) if s] or list(SCHEMES)
        unknown = [s for s in names if s not in SCHEMES]
        if unknown:
            return _err(f"scheme must be one of {', '.join(SCHEMES)}; got {', '.join(unknown)}")

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))

        pairs = pairs_of(graph)
        if pairs is None:
            return _err(
                "sna projections needs the two-mode memberships a projection was built from, "
                "and this network carries none. The speakers and entities networks have them, "
                "as does speakers-entities with project set; the topic network reads stored "
                "co-occurrence edges and the relations network is not a projection at all"
            )
        if not graph.number_of_nodes():
            return _err(
                f"the {network} network is empty before any scheme runs, so there is nothing to "
                f"project (min_weight={min_weight}). Lower it"
            )
        mode = projected_side(graph, pairs)
        if mode is None:
            return _err("could not tell which mode this network is the projection of")
        kept_nodes = set(graph.nodes)
        pairs = [
            (left, right)
            for left, right in pairs
            if (left if mode == "left" else right) in kept_nodes
        ]
        frame = str(graph.graph.get("frame", ""))
        if min_weight is not None and min_weight > 1:
            frame += (
                f" Restricted to the nodes surviving min_weight={min_weight} on the "
                "simple-weight projection, so every scheme below is compared over that "
                "sub-corpus and the opposite-mode degrees are its own."
            )
        try:
            comparison = compare_schemes(
                pairs,
                mode,
                schemes=names,
                lam=lam,
                keep_top=keep_top,
                threshold=threshold,
                persona_id=persona,
                network=network,
                unit="document" if network == "speakers" else "opposite-mode node",
                frame=frame,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="projections",
            parameters={
                "persona": persona,
                "network": network,
                "project": project,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "scheme": scheme,
                "lam": lam,
                "keep_top": keep_top,
                "threshold": threshold,
            },
            chapters=COMMAND_CHAPTERS["projections"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_projections_of(comparison), prov)
        payload = _payload_with_provenance(projections_payload(comparison), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_walks(
        persona: str,
        source_node: str,
        target_node: str,
        network: str = "entities",
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        unweighted: bool = False,
        max_nodes: int = 2_000,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna walks`: what a random walker between two nodes costs -- hitting,
        commute, resistance, cut (Atlas ch. 11). None of these has a null model: they are exact
        properties of the network as built, not claims about a population.

        `graphrag.sna.walks.walks_report`/`render_walks`/`walks_payload`, the same functions the
        CLI calls (ATL-F1: `--out`/`--json` write the same `markdown`/`payload` this tool
        returns).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.measures import undirected_view
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.walks import render_walks, walks_payload, walks_report

        try:
            built = _build(
                network, persona, source=source, min_weight=min_weight, types=parse_types(types)
            )
        except ValueError as exc:
            return _err(str(exc))

        graph, flattened = undirected_view(built)
        for node in (source_node, target_node):
            if node not in graph:
                return _err(f"not in the {network} network: {node}")
        if source_node == target_node:
            return _err("source_node and target_node must name two different nodes")

        try:
            report = walks_report(
                graph,
                source_node,
                target_node,
                persona_id=persona,
                network=network,
                flattened_note=flattened,
                unweighted=unweighted,
                max_nodes=max_nodes,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="walks",
            parameters={
                "persona": persona,
                "network": network,
                "from": source_node,
                "to": target_node,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "unweighted": unweighted,
                "max_nodes": max_nodes,
            },
            chapters=COMMAND_CHAPTERS["walks"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_walks(report)), prov)
        _maybe_write(out, markdown)
        payload = _payload_with_provenance(walks_payload(report), prov)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_distance(
        persona: str,
        vector_a: str,
        vector_b: str,
        network: str = "entities",
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        max_nodes: int = 2_000,
        max_support: int = 60,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna distance`: compare two node vectors on one network, five different ways
        (Atlas ch. 47). `vector_a`/`vector_b` are the CLI's `--vectors a b`, split into two
        arguments; see `graphrag.sna.vectordist.SPEC_HELP` for the spec language (`stance:praise`,
        `attr:<key>`, `centrality:<kind>`, `window:<since>:<until>:<kind>`).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.vectordist import compare_vectors, distance_payload, resolve_vector
        from graphrag.sna.vectordist import render_distance as render_distance_of

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                facet=facet,
                where=parse_where(where),
            )
            p = resolve_vector(
                vector_a,
                graph,
                store=st.ctx.store,
                persona_id=persona,
                network=network,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                facets=facet,
                where=parse_where(where),
            )
            q = resolve_vector(
                vector_b,
                graph,
                store=st.ctx.store,
                persona_id=persona,
                network=network,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                facets=facet,
                where=parse_where(where),
            )
        except ValueError as exc:
            return _err(str(exc))

        if graph.number_of_nodes() == 0:
            return _err("this network has no nodes, so there is nothing to compare")
        try:
            report = compare_vectors(
                graph, p, q, persona_id=persona, max_nodes=max_nodes, max_support=max_support
            )
        except ValueError as exc:
            return _err(str(exc))
        prov = make_provenance(
            method="distance",
            parameters={
                "persona": persona,
                "network": network,
                "vectors": [vector_a, vector_b],
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "facet": facet,
                "where": parse_where(where),
                "max_nodes": max_nodes,
                "max_support": max_support,
            },
            chapters=COMMAND_CHAPTERS["distance"],
            snapshot_commit=_snapshot_commit(persona),
        )
        full_markdown = _with_provenance("\n".join(render_distance_of(report)), prov)
        _maybe_write(out, full_markdown)
        payload = _payload_with_provenance(distance_payload(report), prov)
        _maybe_write_json(as_json, payload)
        # `markdown` itself keeps no trailing newline (unlike every other tool's, and unlike
        # what `--out` just wrote above): the CLI prints this one straight to the console, and
        # its own parity test strips stdout's trailing newline rather than the string this
        # returns -- see `test_sna_distance_matches_cli`.
        return {"markdown": full_markdown.rstrip("\n"), "payload": payload}

    @mcp.tool
    def sna_backbone(
        persona: str,
        network: str = "speakers",
        alpha: float = 0.05,
        correction: str = "none",
        threshold: float | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna backbone`: run every backboning method of Atlas chapter 27 over one
        network and report what each keeps -- run before choosing a `backbone` for `sna_analyze`.
        `graphrag.sna.backbone.compare_backbones`/`render_backbones`/`backbone_payload`, the same
        functions the CLI calls (ATL-F1: `--json` writes the same `payload` this tool returns).
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.backbone import backbone_payload, compare_backbones, render_backbones
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_edges() == 0:
            return _err("this network has no edges, so there is nothing to backbone")
        rows = compare_backbones(
            graph, alpha=alpha, correction=correction, threshold=threshold, seed=seed
        )
        prov = make_provenance(
            method="backbone",
            parameters={
                "persona": persona,
                "network": network,
                "alpha": alpha,
                "correction": correction,
                "threshold": threshold,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["backbone"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(
            render_backbones(graph, rows, alpha=alpha, correction=correction), prov
        )
        _maybe_write(out, markdown)
        payload = _payload_with_provenance(
            backbone_payload(rows, alpha=alpha, correction=correction), prov
        )
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_summarize(
        persona: str,
        by: str = "community",
        network: str = "entities",
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        resolution: float = 1.0,
        runs: int = 10,
        simplify_method: str = "degree",
        simplify_share: float = 0.5,
        influence_top: int = 15,
        influence_beta: float = 1.0,
        influence_attempts: int = 1,
        influence_runs: int = 1,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna summarize`: reduce a network to a smaller one that stands in for it
        (Atlas ch. 46) -- aggregation, compression, simplification, an influence-based summary,
        over the `by` grouping (`community`, Louvain, or `attr:<key>`).
        `graphrag.sna.summarize.summarize`/`render_summary`/`summary_payload`, the same functions
        the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.summarize import render_summary, summarize, summary_payload

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_edges() == 0:
            return _err("this network has no edges, so there is nothing to summarize")
        try:
            report = summarize(
                graph,
                by=by,
                resolution=resolution,
                community_runs=runs,
                simplify_method=simplify_method,
                simplify_share=simplify_share,
                influence_top=influence_top,
                influence_beta=influence_beta,
                influence_attempts=influence_attempts,
                influence_runs=influence_runs,
                seed=seed,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="summarize",
            parameters={
                "persona": persona,
                "by": by,
                "network": network,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
                "project": project,
                "resolution": resolution,
                "runs": runs,
                "simplify_method": simplify_method,
                "simplify_share": simplify_share,
                "influence_top": influence_top,
                "influence_beta": influence_beta,
                "influence_attempts": influence_attempts,
                "influence_runs": influence_runs,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["summarize"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_summary(report)), prov)
        payload = _payload_with_provenance(summary_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_degree(
        persona: str,
        network: str = "entities",
        kind: str = "degree",
        bootstrap: int = 200,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        sample_method: str | None = None,
        sample_size: int | None = None,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna degree`: the degree distribution, and whether it is a power law (Atlas
        ch. 9). `graphrag.sna.degree.degree_report`/`render_degree`/`degree_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.degree import DEGREE_KINDS, degree_payload, degree_report, render_degree
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.sampling import SAMPLE_METHODS
        from graphrag.sna.sampling import sample as take_sample

        if kind not in DEGREE_KINDS:
            return _err(f"kind must be one of {', '.join(DEGREE_KINDS)}, got {kind!r}")
        if bootstrap < 1:
            return _err("bootstrap must be at least 1 resample")
        if sample_method is not None:
            if sample_method not in SAMPLE_METHODS:
                return _err(f"sample_method must be one of {', '.join(SAMPLE_METHODS)}")
            if sample_size is None or sample_size < 1:
                return _err("sample_method needs sample_size, a node count of at least 1")

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
            if sample_method is not None and sample_size is not None:
                graph = take_sample(graph, sample_method, sample_size, seed=seed)
            report = degree_report(graph, kind, bootstrap=bootstrap, seed=seed)
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="degree",
            parameters={
                "persona": persona,
                "network": network,
                "kind": kind,
                "bootstrap": bootstrap,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
                "sample_method": sample_method,
                "sample_size": sample_size,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["degree"],
            null_model="power-law bootstrap (synthetic data drawn from the fitted exponent, §9.4)",
            null_model_samples=bootstrap,
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_degree(report)), prov)
        payload = _payload_with_provenance(degree_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_roles(
        persona: str,
        network: str = "entities",
        method: str = "jaccard",
        k: int | None = None,
        decay: float = 0.8,
        minimum: float = 0.5,
        resolution: float = 1.0,
        runs: int = 10,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        max_nodes: int = 2_000,
    ) -> dict[str, Any]:
        """`graphrag sna roles`: what each node *is* in the network (Atlas ch. 15) -- z/P roles,
        structural-equivalence similarity, and the "same role, never together" section.
        `graphrag.sna.roles.build_roles_report`/`render_roles`/`roles_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.roles import SIMILARITIES, build_roles_report, render_roles, roles_payload

        if method not in SIMILARITIES:
            return _err(f"method must be one of {', '.join(SIMILARITIES)}, got {method!r}")

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_edges() == 0:
            return _err("this network has no edges, so no node has a structural role")
        if k is not None and not 1 <= k <= graph.number_of_nodes():
            return _err(f"k must be between 1 and the {graph.number_of_nodes()} nodes")
        try:
            report = build_roles_report(
                graph,
                persona_id=persona,
                network=network,
                method=method,
                k=k,
                decay=decay,
                minimum=minimum,
                resolution=resolution,
                runs=runs,
                seed=seed,
                max_nodes=max_nodes,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="roles",
            parameters={
                "persona": persona,
                "network": network,
                "method": method,
                "k": k,
                "decay": decay,
                "minimum": minimum,
                "resolution": resolution,
                "runs": runs,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
                "max_nodes": max_nodes,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["roles"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_roles(report), prov)
        payload = _payload_with_provenance(roles_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_ego(
        persona: str,
        node: str,
        network: str = "entities",
        radius: int = 1,
        without_ego: bool = False,
        by: str | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna ego`: one node's neighbourhood, with and without the node (Atlas §30.1).
        `graphrag.sna.ego.ego_report`/`render_ego`/`ego_payload`, the same functions the CLI
        calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.ego import ego_payload, ego_report, render_ego
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if radius < 1:
            return _err("radius must be at least 1 hop")
        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        try:
            report = ego_report(graph, node, radius=radius, with_ego=not without_ego, by=by)
        except KeyError:
            return _err(f"not in the {network} network: {node}")

        prov = make_provenance(
            method="ego",
            parameters={
                "persona": persona,
                "node": node,
                "network": network,
                "radius": radius,
                "without_ego": without_ego,
                "by": by,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
            },
            chapters=COMMAND_CHAPTERS["ego"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_ego(report)), prov)
        payload = _payload_with_provenance(ego_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_community(
        persona: str,
        seed_node: str | None = None,
        max_size: int | None = None,
        temporal: bool = False,
        network: str = "entities",
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        window: str = "6M",
        step: str | None = None,
        since: str | None = None,
        until: str | None = None,
        cumulative: bool = False,
        threshold: float = 0.1,
        resolution: float = 1.0,
        runs: int = 10,
        random_seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna community`: community discovery that does not look at the whole network
        at once (Atlas §35.4-§35.5) -- `seed_node` grows a local community outward from one node;
        `temporal` tracks Louvain communities across dated snapshots instead. Exactly one of the
        two must be given. `graphrag.sna.cluster.local_community`/`match_communities` and their
        `render_*`/`*_payload` pairs, plus `graphrag.sna.evaluate.evaluate_partition` for the
        `## Community evaluation` section(s) -- the same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        if seed_node is not None and temporal:
            return _err("seed_node and temporal are two different modes; pass only one")
        if seed_node is None and not temporal:
            return _err(
                "give seed_node for a local community, or temporal=true to track across snapshots"
            )

        from graphrag.sna.cluster import (
            local_community,
            local_community_payload,
            match_communities,
            render_local_community,
            render_temporal_communities,
            temporal_communities_payload,
        )
        from graphrag.sna.evaluate import evaluate_partition, evaluation_payload, render_evaluation
        from graphrag.sna.layers import snapshots
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        lines: list[str] = []
        payload: dict[str, Any] = {}
        try:
            if temporal:
                windows = snapshots(
                    st.ctx.store,
                    persona,
                    network,
                    window=window,
                    step=step,
                    since=since,
                    until=until,
                    cumulative=cumulative,
                    source_id=source,
                    min_weight=min_weight,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    where=parse_where(where),
                    project=project,  # type: ignore[arg-type]
                )
                tracked = match_communities(
                    windows,
                    threshold=threshold,
                    resolution=resolution,
                    seed=random_seed,
                    runs=runs,
                )
                lines = render_temporal_communities(tracked)
                payload = temporal_communities_payload(tracked)
                evaluations: dict[str, Any] = {}
                windows_by_label = dict(windows)
                for window_label, partition in zip(
                    tracked.windows, tracked.partitions, strict=True
                ):
                    graph_for_window = windows_by_label[window_label]
                    if partition and graph_for_window.number_of_edges():
                        scores = evaluate_partition(
                            graph_for_window, partition, resolution=resolution, seed=random_seed
                        )
                        lines += [f"#### Community evaluation: {window_label}", ""]
                        lines += render_evaluation(scores)
                        evaluations[window_label] = evaluation_payload(scores)
                if evaluations:
                    payload["evaluation"] = evaluations
            else:
                assert seed_node is not None
                graph = _build(
                    network,
                    persona,
                    source=source,
                    min_weight=min_weight,
                    types=parse_types(types),
                    stance=stance,
                    facet=facet,
                    where=parse_where(where),
                    project=project,
                )
                result = local_community(graph, seed_node, max_size=max_size, seed=random_seed)
                lines = render_local_community(result)
                payload = local_community_payload(result)
                if len(result.members) > 1:
                    scores = evaluate_partition(graph, [result.members], seed=random_seed)
                    lines += ["#### Community evaluation", ""]
                    lines += render_evaluation(scores)
                    payload["evaluation"] = evaluation_payload(scores)
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="community",
            parameters={
                "persona": persona,
                "seed_node": seed_node,
                "max_size": max_size,
                "temporal": temporal,
                "network": network,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
                "project": project,
                "window": window,
                "step": step,
                "since": since,
                "until": until,
                "cumulative": cumulative,
                "threshold": threshold,
                "resolution": resolution,
                "runs": runs,
            },
            seed=random_seed,
            chapters=COMMAND_CHAPTERS["community"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(lines), prov)
        payload = _payload_with_provenance(payload, prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_overlap(
        persona: str,
        network: str = "entities",
        method: str = "clique",
        k: int = 4,
        jaccard_threshold: float = 0.1,
        resolution: float = 1.0,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna overlap`: nodes that belong to more than one community (Atlas ch. 38) --
        `method` clique|link|ego. `graphrag.sna.overlap.build_overlap_report`/`render_overlap`/
        `overlap_payload`, the same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.overlap import METHODS, build_overlap_report, overlap_payload
        from graphrag.sna.overlap import render_overlap as render_overlap_of
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if method not in METHODS:
            return _err(f"method must be one of {', '.join(METHODS)}, got {method!r}")

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_edges() == 0:
            return _err("this network has no edges, so there is nothing to find a cover of")
        try:
            report = build_overlap_report(
                graph,
                method,
                k=k,
                jaccard_threshold=jaccard_threshold,
                resolution=resolution,
                seed=seed,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="overlap",
            parameters={
                "persona": persona,
                "network": network,
                "method": method,
                "k": k,
                "jaccard_threshold": jaccard_threshold,
                "resolution": resolution,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["overlap"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_overlap_of(report)), prov)
        payload = _payload_with_provenance(overlap_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_hierarchy(
        persona: str,
        network: str = "relations",
        samples: int = 50,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        relation_type: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        seed: int | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna hierarchy`: whether the network is an organisation chart, by four
        measures tested against `samples` degree-preserving rewirings (Atlas ch. 33). Needs a
        directed network (`network="relations"`, the only one this package builds).
        `graphrag.sna.hierarchy.analyse_hierarchy`/`render_hierarchy`/`hierarchy_payload`, the
        same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.hierarchy import analyse_hierarchy, hierarchy_payload, render_hierarchy
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if samples < 0:
            return _err("samples cannot be negative")
        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                relation_type=relation_type,
                facet=facet,
                since=since,
                until=until,
                where=parse_where(where),
            )
            report = analyse_hierarchy(graph, samples=samples, seed=seed)
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="hierarchy",
            parameters={
                "persona": persona,
                "network": network,
                "samples": samples,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "relation_type": relation_type,
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["hierarchy"],
            null_model="directed degree-preserving rewiring (in- and out-degree fixed)",
            null_model_samples=samples,
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_hierarchy(report)), prov)
        payload = _payload_with_provenance(hierarchy_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_highorder(
        persona: str,
        network: str = "entities",
        max_dim: int = 3,
        damping: float = 0.85,
        exact: bool = False,
        source: str | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna highorder`: the passages as simplices, and a walk that remembers where it
        came from (Atlas ch. 34) -- the simplicial complex (§34.1) and the memory network
        (§34.2-§34.3). Fixed to `network="entities"`, the only projection of the passage
        hypergraph this package builds. `graphrag.sna.highorder.highorder_reports`/
        `render_simplicial`/`simplicial_payload`/`render_memory`/`memory_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.highorder import (
            highorder_reports,
            memory_payload,
            render_memory,
            render_simplicial,
            simplicial_payload,
        )
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if network != "entities":
            return _err(
                "network must be entities: chapter 34's structures are read off the passage "
                "hypergraph, and only the entity network is a projection of one"
            )
        if max_dim < 0:
            return _err("max_dim must be at least 0 — a 0-simplex is a node")
        if not exact and not 0.0 < damping <= 1.0:
            return _err("damping must be above 0 and at most 1 (§11.1's teleport)")

        try:
            simplicial, memory = highorder_reports(
                st.ctx.store,
                persona,
                source_id=source,
                types=parse_types(types),
                stances=stance,
                facets=facet,
                since=since,
                until=until,
                where=parse_where(where),
                max_dim=max_dim,
                damping=None if exact else damping,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="highorder",
            parameters={
                "persona": persona,
                "network": network,
                "max_dim": max_dim,
                "damping": damping,
                "exact": exact,
                "source": source,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
            },
            chapters=COMMAND_CHAPTERS["highorder"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(
            "\n".join([*render_simplicial(simplicial), *render_memory(memory)]), prov
        )
        payload = _payload_with_provenance(
            {"simplicial": simplicial_payload(simplicial), "memory": memory_payload(memory)}, prov
        )
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_predict_eval(
        persona: str,
        network: str = "entities",
        temporal: str | None = None,
        share: float = 0.1,
        negatives: float = 1.0,
        k: int = 10,
        seed: int | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna predict-eval`: build a link-prediction experiment and run the baselines
        (preferential attachment, chance) through it (Atlas ch. 25) -- the report every prediction
        in this package comes back through. `temporal="2025-01-01"` splits by date instead of
        deleting a random `share` of edges. `graphrag.sna.experiment.holdout`/`temporal_holdout`/
        `evaluate_predictor`/`render_experiment`/`experiment_payload`, the same functions the CLI
        calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.experiment import (
            evaluate_predictor,
            experiment_payload,
            holdout,
            preferential_attachment,
            random_scores,
            render_experiment,
            temporal_holdout,
        )
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            if temporal is not None:
                split = temporal_holdout(
                    st.ctx.store,
                    persona,
                    network,
                    split_at=temporal,
                    seed=seed,
                    negatives=negatives,
                    source_id=source,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    where=parse_where(where),
                )
            else:
                graph = _build(
                    network,
                    persona,
                    source=source,
                    min_weight=min_weight,
                    types=parse_types(types),
                    stance=stance,
                    facet=facet,
                    where=parse_where(where),
                )
                split = holdout(graph, share=share, seed=seed, negatives=negatives)
        except ValueError as exc:
            return _err(str(exc))

        if not split.positives or not split.negatives:
            return _err(
                f"this split has {len(split.positives):,} positive(s) and "
                f"{len(split.negatives):,} negative(s); §25.2's measures need both classes. "
                "Widen the network, lower min_weight, or move the split date."
            )

        reports = [
            evaluate_predictor(split, preferential_attachment, k=k),
            evaluate_predictor(split, lambda train: random_scores(train, seed=seed), k=k),
        ]
        prov = make_provenance(
            method="predict-eval",
            parameters={
                "persona": persona,
                "network": network,
                "temporal": temporal,
                "share": share,
                "negatives": negatives,
                "k": k,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["predict-eval"],
            snapshot_commit=_snapshot_commit(persona),
        )
        report = _with_provenance(render_experiment(split, reports), prov)
        markdown = f"{persona} / {network}: {split.method} holdout\n{report}"
        payload = _payload_with_provenance(experiment_payload(split, reports), prov)
        _maybe_write(out, report)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_predict(
        persona: str,
        method: str = "cn",
        top: int = 10,
        share: float = 0.1,
        negatives: float = 1.0,
        k: int = 10,
        seed: int | None = None,
        beta: float | None = None,
        hrg_samples: int | None = None,
        hrg_restarts: int = 5,
        min_support: int = 2,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        signed: bool = False,
        sign_share: float = 0.2,
        layers: str | None = None,
        target_layer: str | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna predict`: rank link-prediction hypotheses on the entity network, never
        writing one to the graph (Atlas ch. 23) -- always the ch. 25 evaluation first, then the
        ranked hypotheses with their evidence passages. `method` is one of pa|cn|aa|ra|jaccard|
        katz|hrg|rules. `signed` adds §24.1's balance-theory sign prediction; `layers` switches to
        §24.2's generalized multilayer prediction for `target_layer` instead. Fixed to the entity
        network. Calls the exact functions `graphrag.cli.sna_predict`'s body does
        (`graphrag.sna.predict`, `graphrag.sna.experiment`), unabridged.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.experiment import (
            evaluate_predictor,
            holdout,
            preferential_attachment,
            random_scores,
            render_experiment,
        )
        from graphrag.sna.export import entity_passage_rows
        from graphrag.sna.layers import multilayer
        from graphrag.sna.motifs import document_graphs
        from graphrag.sna.predict import (
            DEFAULT_EVIDENCE_LIMIT,
            METHOD_NAMES,
            METHOD_SECTIONS,
            PREDICT_METHODS,
            QUOTE,
            adamic_adar,
            association_rule_scores,
            common_neighbors,
            cross_layer_common_neighbours,
            documents_without,
            evaluate_sign_predictor,
            hrg_scores,
            hypotheses_payload,
            jaccard,
            katz,
            passages_by_entity,
            rank_hypotheses,
            render_hypotheses,
            render_predicted_signs,
            render_sign_evaluation,
            resource_allocation,
            sign_holdout,
            signed_graph,
        )
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.stances import build_stance_report
        from graphrag.textutil import sanitize_inline

        if method not in PREDICT_METHODS:
            return _err(f"method must be one of {', '.join(PREDICT_METHODS)}, got {method!r}")
        if top < 1:
            return _err("top must be at least 1")

        if layers is not None:
            try:
                ml = multilayer(
                    st.ctx.store,
                    persona,
                    "entities",
                    layers,  # type: ignore[arg-type]
                    source_id=source,
                    min_weight=min_weight,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    where=parse_where(where),
                )
                target = target_layer or ml.names[0]
                if target not in ml.names:
                    return _err(
                        f"target_layer must be one of {', '.join(ml.names)}, got {target!r}"
                    )
                target_graph = ml.graphs[target]
                split = holdout(target_graph, share=share, seed=seed, negatives=negatives)
                if not split.positives or not split.negatives:
                    return _err(
                        f"this split has {len(split.positives):,} positive(s) and "
                        f"{len(split.negatives):,} negative(s); §25.2's measures need both "
                        "classes. Widen the layer or lower min_weight."
                    )
                reports = [
                    evaluate_predictor(
                        split,
                        lambda train: cross_layer_common_neighbours(ml, target, train),
                        name=f"cross-layer common neighbours ({target})",
                        k=k,
                    ),
                    evaluate_predictor(
                        split,
                        common_neighbors,
                        name=f"common neighbours ({target}, single layer)",
                        k=k,
                    ),
                    evaluate_predictor(split, preferential_attachment, k=k),
                    evaluate_predictor(split, lambda train: random_scores(train, seed=seed), k=k),
                ]
                report = render_experiment(
                    split, reports, title=f"Multilayer link prediction ({target}, Atlas §24.2)"
                )
            except ValueError as exc:
                return _err(str(exc))
            prov = make_provenance(
                method="predict",
                parameters={
                    "persona": persona,
                    "network": "entities",
                    "layers": layers,
                    "target_layer": target,
                    "share": share,
                    "negatives": negatives,
                    "k": k,
                    "source": source,
                    "min_weight": min_weight,
                    "types": parse_types(types),
                    "stance": stance,
                    "facet": facet,
                    "where": parse_where(where),
                },
                seed=seed,
                chapters=COMMAND_CHAPTERS["predict"],
                snapshot_commit=_snapshot_commit(persona),
            )
            report = _with_provenance(report, prov)
            markdown = f"{persona} / entities, layers={layers}: target={target}\n{report}"
            _maybe_write(out, markdown)
            return {"markdown": markdown, "payload": _payload_with_provenance({}, prov)}

        try:
            graph = _build(
                "entities",
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                where=parse_where(where),
            )
            documents = (
                document_graphs(
                    st.ctx.store,
                    persona,
                    source_id=source,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    where=parse_where(where),
                )
                if method == "rules"
                else []
            )
            split = holdout(graph, share=share, seed=seed, negatives=negatives)
            rows = entity_passage_rows(
                st.ctx.store,
                persona,
                source,
                types,
                stances=stance,
                facets=facet,
                where=parse_where(where),
            )
        except ValueError as exc:
            return _err(str(exc))

        if not split.positives or not split.negatives:
            return _err(
                f"this split has {len(split.positives):,} positive(s) and "
                f"{len(split.negatives):,} negative(s); §25.2's measures need both classes. "
                "Widen the network or lower min_weight."
            )
        if method == "rules" and len(documents) < 2:
            return _err("method rules needs at least two documents to fit a rule from (§41.4)")

        def scorer_for(train: Any) -> Any:
            if method == "pa":
                return preferential_attachment(train)
            if method == "cn":
                return common_neighbors(train)
            if method == "aa":
                return adamic_adar(train)
            if method == "ra":
                return resource_allocation(train)
            if method == "jaccard":
                return jaccard(train)
            if method == "katz":
                return katz(train, beta)
            if method == "hrg":
                return hrg_scores(train, samples=hrg_samples, restarts=hrg_restarts, seed=seed)
            removed = frozenset(split.positives) if train is split.train else frozenset()
            return association_rule_scores(
                train, documents_without(documents, removed), min_support=min_support
            )

        try:
            reports = [evaluate_predictor(split, scorer_for, name=METHOD_NAMES[method], k=k)]
            if method != "pa":
                reports.append(evaluate_predictor(split, preferential_attachment, k=k))
            reports.append(
                evaluate_predictor(split, lambda train: random_scores(train, seed=seed), k=k)
            )
        except ValueError as exc:
            return _err(str(exc))
        evaluation = render_experiment(
            split, reports, title=f"Link prediction evaluation ({METHOD_NAMES[method]}, ch. 25)"
        )

        try:
            live_scores = scorer_for(graph)
        except ValueError as exc:
            return _err(str(exc))
        passages = passages_by_entity(rows)
        hypotheses = rank_hypotheses(
            live_scores,
            graph,
            passages,
            method=METHOD_NAMES[method],
            section=METHOD_SECTIONS[method],
            top=top,
            evidence_limit=DEFAULT_EVIDENCE_LIMIT,
        )
        chunk_ids = sorted({ref.chunk_id for h in hypotheses for ref in h.evidence.passages})
        excerpts = {
            c.id: sanitize_inline(c.text, QUOTE) for c in st.ctx.store.get_chunks(chunk_ids)
        }

        sign_section = ""
        if signed:
            signed_pairs = build_stance_report(
                st.ctx.store,
                persona,
                source_id=source,
                facets=facet or (),
                where=parse_where(where),
            ).pairs
            signed_net = signed_graph(signed_pairs)
            if signed_net.number_of_edges() < 2:
                sign_section = (
                    "## Signed link prediction (balance theory)\n\n"
                    f"Only {signed_net.number_of_edges():,} signed co-mention edge(s) "
                    f"({signed_net.graph.get('ambiguous', 0):,} ambiguous pair(s) dropped); "
                    "§24.1's holdout needs at least 2. Annotate more praise/complaint mentions "
                    "first.\n"
                )
            else:
                try:
                    sign_split = sign_holdout(signed_net, share=sign_share, seed=seed)
                except ValueError as exc:
                    return _err(str(exc))
                sign_eval = evaluate_sign_predictor(sign_split, seed=seed)
                sign_section = render_sign_evaluation(
                    sign_split, sign_eval
                ) + render_predicted_signs(hypotheses, signed_net)

        frame = (
            f"{persona} / entities network, {graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges. {METHOD_NAMES[method]} "
            f"({METHOD_SECTIONS[method]}) scored on the network as it stands today, not on the "
            "evaluation's holdout."
        )
        null_model = (
            "the same two baselines as the evaluation above -- a scorer that ignores the "
            "network and preferential attachment -- plus, for `rules`, the network's own global "
            "clustering coefficient (§12.2), since a rule that closes triangles no faster than "
            "the network already does is not a finding."
        )
        hypothesis_section = render_hypotheses(
            hypotheses,
            frame=frame,
            null_model=null_model,
            title=f"Link prediction hypotheses ({METHOD_NAMES[method]}, ch. 23)",
            excerpts=excerpts,
        )
        report = f"{evaluation}\n{hypothesis_section}"
        if sign_section:
            report = f"{report}\n{sign_section}"
        prov = make_provenance(
            method="predict",
            parameters={
                "persona": persona,
                "network": "entities",
                "method": method,
                "top": top,
                "share": share,
                "negatives": negatives,
                "k": k,
                "beta": beta,
                "hrg_samples": hrg_samples,
                "hrg_restarts": hrg_restarts,
                "min_support": min_support,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
                "signed": signed,
                "sign_share": sign_share,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["predict"],
            null_model=null_model,
            snapshot_commit=_snapshot_commit(persona),
        )
        report = _with_provenance(report, prov)
        markdown = f"{persona} / entities: {method}\n{report}"
        # `hypotheses_payload` is a list, not a dict -- the one report whose payload is not
        # already keyed, so it is the one wrapped rather than given a "provenance" key directly.
        payload = _payload_with_provenance({"hypotheses": hypotheses_payload(hypotheses)}, prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_complete(
        persona: str,
        key: str,
        method: str = "gcn",
        hidden_dims: str = "16",
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 0,
        val_share: float = 0.2,
        top: int = 25,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        where: dict[str, str] | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna complete`: suggest `key` for entities that carry no value of their own,
        by a two-layer GCN or GraphSAGE trained on the entities this persona already tagged
        (Atlas ch. 44). Nothing is written to the graph; every suggestion is a row to paste into
        an extraction sidecar, or to discard. Needs the `graphrag[gnn]` extra (CPU-only `torch`);
        returns a structured error naming the install command when it is not present, rather than
        raising. `graphrag.sna.gnn.complete_attribute`/`render_completion`/`completion_payload`,
        the same functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna import gnn
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if method not in gnn.METHODS:
            return _err(f"method must be one of {gnn.METHODS}, got {method!r}")
        try:
            dims = tuple(int(d.strip()) for d in hidden_dims.split(",") if d.strip())
        except ValueError:
            return _err(
                f"hidden_dims must be a comma separated list of integers, got {hidden_dims!r}"
            )
        if not dims:
            return _err("hidden_dims needs at least one layer width")
        if not gnn.available():
            return _err(gnn.INSTALL_HINT)

        try:
            graph = _build(
                "entities",
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                where=parse_where(where),
            )
            text = st.ctx.store.mean_embeddings(persona, level="entity")
            report = gnn.complete_attribute(
                graph,
                key,
                method=method,
                features=text,
                hidden_dims=dims,
                epochs=epochs,
                lr=lr,
                seed=seed,
                val_share=val_share,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="complete",
            parameters={
                "persona": persona,
                "key": key,
                "method": method,
                "hidden_dims": hidden_dims,
                "epochs": epochs,
                "lr": lr,
                "val_share": val_share,
                "top": top,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": parse_where(where),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["complete"] if method == "gat" else (44,),
            snapshot_commit=_snapshot_commit(persona),
        )
        rendered = _with_provenance(gnn.render_completion(report, top=top), prov)
        markdown = f"{persona} / entities: completing {key!r} ({method})\n{rendered}"
        payload = _payload_with_provenance(gnn.completion_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_robustness(
        persona: str,
        network: str = "entities",
        strategy: list[str] | None = None,
        recompute: bool = False,
        steps: int = 20,
        runs: int = 20,
        seed: int | None = None,
        cascade: bool = False,
        tolerance: list[float] | None = None,
        load: str = "degree",
        couple: str | None = None,
        coupling: str = "same-id",
        max_nodes: int = 2000,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        out: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna robustness`: how much of this network survives losing its nodes --
        random, targeted, cascade, coupled (Atlas ch. 22). `graphrag.sna.robustness.
        robustness_report`/`render_robustness`/`robustness_payload`, the same functions the CLI
        calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.robustness import (
            CASCADE_TOLERANCES,
            COUPLINGS,
            LOADS,
            TARGETED_STRATEGIES,
            render_robustness,
            robustness_payload,
            robustness_report,
            strategies_for,
        )

        if load not in LOADS:
            return _err(f"load must be one of {', '.join(LOADS)}")
        if coupling not in COUPLINGS:
            return _err(f"coupling must be one of {', '.join(COUPLINGS)}")
        if steps < 1 or runs < 1:
            return _err("steps and runs must be at least 1")
        tolerances = tuple(tolerance) if tolerance else CASCADE_TOLERANCES
        if any(value < 0 for value in tolerances):
            return _err("tolerance must be at least 0")

        try:
            graph = _build(
                network, persona, source=source, min_weight=min_weight, types=parse_types(types)
            )
            other = (
                _build(
                    couple, persona, source=source, min_weight=min_weight, types=parse_types(types)
                )
                if couple is not None
                else None
            )
        except ValueError as exc:
            return _err(str(exc))

        if graph.number_of_nodes() == 0:
            return _err(f"the {network} network is empty, so there is nothing to remove")
        if graph.number_of_nodes() > max_nodes:
            return _err(
                f"{graph.number_of_nodes():,} nodes is above max_nodes {max_nodes:,}: every "
                "curve removes the network one step at a time and recomputes the components, so "
                "the cost grows with n. Raise max_nodes, or lower steps and runs"
            )
        chosen = [s.strip() for s in strategy or []] or list(TARGETED_STRATEGIES)
        allowed = strategies_for(graph)
        unknown = [s for s in chosen if s not in allowed or s == "random"]
        if unknown:
            return _err(
                f"strategy must be one of {', '.join(s for s in allowed if s != 'random')}; got "
                f"{', '.join(unknown)}. Random removal always runs as the null"
            )

        try:
            report = robustness_report(
                graph,
                network=network,
                persona_id=persona,
                strategies=chosen,
                recompute=recompute,
                steps=steps,
                runs=runs,
                seed=seed,
                cascades=cascade,
                tolerances=tolerances,
                load=load,
                other=other,
                coupling=coupling,
                coupled_network=couple or "",
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="robustness",
            parameters={
                "persona": persona,
                "network": network,
                "strategy": strategy,
                "recompute": recompute,
                "steps": steps,
                "runs": runs,
                "cascade": cascade,
                "tolerance": tolerance,
                "load": load,
                "couple": couple,
                "coupling": coupling,
                "max_nodes": max_nodes,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["robustness"],
            null_model="random removal order (§22.1)",
            null_model_samples=runs,
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_robustness(report), prov)
        payload = _payload_with_provenance(robustness_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_motifs(
        persona: str,
        network: str = "entities",
        size: int = 3,
        samples: int = 50,
        seed: int | None = None,
        mine: bool = False,
        min_support: int = 3,
        max_size: int = 4,
        out: Path | None = None,
        as_json: Path | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
        max_nodes: int = 500,
    ) -> dict[str, Any]:
        """`graphrag sna motifs`: what small shapes this network is built out of, and which are
        surprising (Atlas ch. 41) -- the census, the motif profile against `samples`
        degree-preserving rewirings, and (`mine=true`) frequent-subgraph mining.
        `graphrag.sna.motifs.build_motifs_report`/`render_motifs`/`motifs_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.motifs import build_motifs_report, document_graphs, motifs_payload
        from graphrag.sna.motifs import render_motifs as render_motifs_of
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
            documents = (
                document_graphs(
                    st.ctx.store,
                    persona,
                    source_id=source,
                    types=parse_types(types),
                    stances=stance,
                    facets=facet,
                    since=since,
                    until=until,
                    where=parse_where(where),
                )
                if mine and network == "entities"
                else None
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_nodes() < 3:
            return _err("fewer than three nodes, so there is no three-node shape to count")
        try:
            report = build_motifs_report(
                graph,
                persona_id=persona,
                network=network,
                size=size,
                samples=samples,
                seed=seed,
                documents=documents,
                mine=mine,
                min_support=min_support,
                max_size=max_size,
                max_nodes=max_nodes,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="motifs",
            parameters={
                "persona": persona,
                "network": network,
                "size": size,
                "samples": samples,
                "mine": mine,
                "min_support": min_support,
                "max_size": max_size,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
                "max_nodes": max_nodes,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["motifs"],
            null_model="degree-preserving rewiring (§19.1)",
            null_model_samples=samples,
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_motifs_of(report), prov)
        payload = _payload_with_provenance(motifs_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_spread(
        persona: str,
        model: str = "si",
        seeds: str | None = None,
        beta: float = 0.2,
        mu: float = 0.1,
        threshold: float = 2.0,
        attempts: int = 1,
        steps: int = 50,
        runs: int = 20,
        share: float | None = None,
        strategy: str = "random",
        seed: int | None = None,
        network: str = "entities",
        out: Path | None = None,
        as_json: Path | None = None,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        stance: list[str] | None = None,
        facet: list[str] | None = None,
        relation_type: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna spread`: what would spread on this network, if anything ever did (Atlas
        ch. 20-21) -- a what-if simulation, never a forecast. `model` is si|sis|sir|threshold|
        limited; `seeds` is comma-separated patient-zero names, or a bare count drawn at random.
        `graphrag.sna.spread.build_spread_report`/`render_spread`/`spread_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance
        from graphrag.sna.spread import (
            IMMUNISATION_STRATEGIES,
            SPREAD_MODELS,
            build_spread_report,
            render_spread,
            spread_payload,
        )

        if model not in SPREAD_MODELS:
            return _err(f"model must be one of {', '.join(SPREAD_MODELS)}, got {model!r}")
        if strategy not in IMMUNISATION_STRATEGIES:
            return _err(f"strategy must be one of {', '.join(IMMUNISATION_STRATEGIES)}")

        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                stance=stance,
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
                project=project,
            )
        except ValueError as exc:
            return _err(str(exc))
        if graph.number_of_nodes() == 0:
            return _err("this network has no nodes, so there is nothing to spread on")
        try:
            report = build_spread_report(
                graph,
                persona_id=persona,
                network=network,
                model=model,
                seeds=_parse_seeds(seeds),
                beta=beta,
                mu=mu,
                threshold=threshold,
                attempts=attempts,
                steps=steps,
                runs=runs,
                share=share,
                strategy=strategy,
                seed=seed,
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="spread",
            parameters={
                "persona": persona,
                "network": network,
                "model": model,
                "seeds": seeds,
                "beta": beta,
                "mu": mu,
                "threshold": threshold,
                "attempts": attempts,
                "steps": steps,
                "runs": runs,
                "share": share,
                "strategy": strategy,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "relation_type": relation_type,
                "since": since,
                "until": until,
                "where": parse_where(where),
                "project": project,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["spread"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance(render_spread(report), prov)
        payload = _payload_with_provenance(spread_payload(report), prov)
        _maybe_write(out, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_draw(
        persona: str,
        network: str = "entities",
        layout: str = "force",
        size: str | None = None,
        color: str | None = None,
        seed: int | None = None,
        width: float = 960.0,
        height: float = 720.0,
        source: str | None = None,
        min_weight: int | None = None,
        types: list[str] | None = None,
        relation_type: list[str] | None = None,
        facet: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        where: dict[str, str] | None = None,
        out: Path | None = None,
        html: Path | None = None,
        report: Path | None = None,
        as_json: Path | None = None,
    ) -> dict[str, Any]:
        """`graphrag sna draw`: lay out and style one network for an SVG and a self-contained
        interactive HTML (Atlas ch. 49-51). `size` names a centrality (ch. 14); `color` is
        `community` (Louvain) or `attr:<key>`. `payload` is `draw_payload` -- the positions,
        sizes and colours the picture used -- not the SVG/HTML markup itself; pass `out` (and
        optionally `html`) to also write the picture to the container's filesystem, as the CLI's
        mandatory `--out` would. `graphrag.sna.draw.draw`/`render_draw`/`draw_payload`, the same
        functions the CLI calls.
        """
        persona_error = _persona_error(persona)
        if persona_error is not None:
            return persona_error
        from graphrag.sna.draw import LAYOUTS, draw, draw_payload
        from graphrag.sna.draw import render_draw as render_draw_of
        from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance

        if layout not in LAYOUTS:
            return _err(f"layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
        try:
            graph = _build(
                network,
                persona,
                source=source,
                min_weight=min_weight,
                types=parse_types(types),
                facet=facet,
                relation_type=relation_type,
                since=since,
                until=until,
                where=parse_where(where),
            )
            result = draw(
                graph,
                layout=layout,
                size=size,
                color=color,
                seed=seed,
                width=width,
                height=height,
                title=f"{persona} · {network} · {layout}",
            )
        except ValueError as exc:
            return _err(str(exc))

        prov = make_provenance(
            method="draw",
            parameters={
                "persona": persona,
                "network": network,
                "layout": layout,
                "size": size,
                "color": color,
                "width": width,
                "height": height,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "relation_type": relation_type,
                "facet": facet,
                "since": since,
                "until": until,
                "where": parse_where(where),
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["draw"],
            snapshot_commit=_snapshot_commit(persona),
        )
        markdown = _with_provenance("\n".join(render_draw_of(result)), prov)
        payload = _payload_with_provenance(draw_payload(result), prov)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.svg, encoding="utf-8")
            html_path = html if html is not None else out.with_suffix(".html")
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(result.html, encoding="utf-8")
        _maybe_write(report, markdown)
        _maybe_write_json(as_json, payload)
        return {"markdown": markdown, "payload": payload}

    @mcp.tool
    def sna_guide() -> dict[str, Any]:
        """`graphrag sna guide`: the method-selection rules -- which network, which method, which
        centrality (`graphrag.sna.guide.render_guide`, the same rules `docs/SNA.md`'s generated
        block quotes)."""
        from graphrag.sna.guide import render_guide

        return {"markdown": render_guide(), "payload": {}}

    # ------------------------------------------------------------- resources & prompts
    @mcp.resource("graphrag://personas")
    def personas_resource() -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in st.ctx.registry.all()]

    @mcp.resource("graphrag://persona/{persona_id}")
    def persona_resource(persona_id: str) -> str:
        return build_brief(st.ctx.registry.get(persona_id), st.ctx.store)

    @mcp.prompt
    def assume_persona(persona_id: str, task: str) -> str:
        """System-style prompt: adopt a persona and work on a task with grounded citations."""
        brief = build_brief(st.ctx.registry.get(persona_id), st.ctx.store)
        return (
            f"{brief}\n\n# Task\n{task}\n\n"
            "Before answering, call the `context` tool with this task and the persona id, then "
            "answer in the persona's voice citing [n]."
        )

    return mcp


def run_server(transport: str = "http", settings: Settings | None = None) -> None:
    cfg = settings or Settings()
    server = create_server(ServerState(cfg))
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "http":
        server.run(transport="http", host=cfg.mcp_host, port=cfg.mcp_port)
    else:
        msg = f"unknown transport {transport!r}; use http or stdio"
        raise ValueError(msg)


def main() -> None:
    run_server()


if __name__ == "__main__":
    main()
