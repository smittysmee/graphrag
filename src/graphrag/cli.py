"""``graphrag`` command line: setup, ingest, snapshots, search, personas, enrichment, servers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from graphrag import __version__
from graphrag.app import AppContext
from graphrag.config import Settings, load_settings
from graphrag.graph import snapshot as snap
from graphrag.models import SearchMode
from graphrag.personas.brief import build_brief
from graphrag.personas.registry import SDLC_STAGES
from graphrag.personas.skill_export import export_persona_skill
from graphrag.pipeline import IngestPipeline

app = typer.Typer(
    help="Persona-grounded Graph RAG on Neo4j (CLI). Every command reads settings from the env.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
snapshot_app = typer.Typer(help="Export/load the committed graph snapshots.", no_args_is_help=True)
persona_app = typer.Typer(help="Manage personas.", no_args_is_help=True)
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(persona_app, name="persona")

console = Console(soft_wrap=True)
err = Console(stderr=True, soft_wrap=True)


class State:
    """Holds an injectable context so tests can run commands against in-memory fakes."""

    factory: Any = None  # Callable[[], AppContext]

    @classmethod
    def context(cls) -> AppContext:
        if cls.factory is not None:
            ctx: AppContext = cls.factory()
            return ctx
        return AppContext.build()


def _settings() -> Settings:
    return State.context().settings if State.factory else load_settings()


def _progress(msg: str) -> None:
    err.print(f"[dim]{msg}[/dim]")


def _warn_if_ephemeral(target: Path) -> None:
    """In a container, only bind-mounted paths survive. Writing elsewhere reports success and
    then silently discards the file when the container exits, so say so loudly."""
    if not Path("/.dockerenv").exists() and not os.environ.get("IN_CONTAINER"):
        return
    mounted = (Path("/app/data"), Path("/app/personas"), Path("/app/.claude"))
    resolved = target.resolve()
    if any(resolved.is_relative_to(m) for m in mounted):
        return
    err.print(
        f"[yellow]warning:[/yellow] {resolved} is inside the container but not on a mounted "
        "volume, so the file will vanish when this command exits. Write it under /app/data "
        "instead, or use `make persona-export`."
    )


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# ----------------------------------------------------------------------------- lifecycle


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Print version and exit.", is_eager=True)
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit


@app.command()
def setup(
    skip_snapshots: Annotated[bool, typer.Option(help="Do not load committed snapshots.")] = False,
) -> None:
    """Create the schema, load every committed snapshot, and export persona skills."""
    ctx = State.context()
    try:
        ctx.store.ensure_schema(ctx.embedder.dim)
        console.print(f"schema ready (dim={ctx.embedder.dim}, model={ctx.embedder.model_name})")
        loaded = 0
        if not skip_snapshots:
            for manifest in snap.list_snapshots(ctx.snapshots_dir):
                try:
                    persona = ctx.registry.get(manifest.persona_id)
                except KeyError as exc:
                    err.print(f"[yellow]skip {manifest.persona_id}: {exc}[/yellow]")
                    continue
                stats = ctx.store.stats().per_persona.get(persona.id, {})
                if stats.get("chunks", 0) == manifest.chunk_count and manifest.chunk_count > 0:
                    console.print(f"{persona.id}: already loaded ({manifest.chunk_count} chunks)")
                    continue
                snap.load_snapshot(
                    ctx.store,
                    persona,
                    ctx.snapshots_dir,
                    embedding_model=ctx.embedder.model_name,
                    embedding_dim=ctx.embedder.dim,
                    on_progress=_progress,
                )
                loaded += 1
                console.print(f"{persona.id}: loaded {manifest.document_count} documents")
        for persona in ctx.registry.all():
            ctx.store.upsert_persona(persona)
            path = export_persona_skill(
                persona, build_brief(persona, ctx.store), ctx.settings.skills_dir
            )
            console.print(f"skill written: {path}")
        console.print(f"[green]setup complete[/green] ({loaded} snapshot(s) loaded)")
    finally:
        ctx.close()


@app.command()
def doctor() -> None:
    """Check Neo4j, the embedding backend, personas and snapshots."""
    import time

    cfg = _settings()
    ok = True
    console.print(f"graphrag {__version__}")
    console.print(f"neo4j: {cfg.neo4j.uri} (user {cfg.neo4j.user})")
    try:
        ctx = State.context()
    except Exception as exc:
        err.print(f"[red]cannot build context: {exc}[/red]")
        raise typer.Exit(1) from exc
    try:
        try:
            ctx.store.stats()
            console.print("  [green]connected[/green]")
        except Exception as exc:
            ok = False
            err.print(f"  [red]unreachable: {exc}[/red]")
        emb = cfg.embedding
        where = emb.base_url if emb.backend == "http" else "in-process"
        console.print(f"embedder: backend={emb.backend} model={emb.model} dim={emb.dim} at {where}")
        try:
            start = time.perf_counter()
            vec = ctx.embedder.embed_query("doctor probe")
            ms = (time.perf_counter() - start) * 1000
            status = "[green]ok[/green]" if vec.shape[0] == emb.dim else "[red]dim mismatch[/red]"
            console.print(f"  {status} ({vec.shape[0]} dims, {ms:.0f} ms round trip)")
            ok = ok and vec.shape[0] == emb.dim
        except Exception as exc:
            ok = False
            err.print(f"  [red]failed: {exc}[/red]")
        personas = ctx.registry.all()
        console.print(f"personas: {', '.join(p.id for p in personas) or '(none)'}")
        for manifest in snap.list_snapshots(ctx.snapshots_dir):
            compat = manifest.embedding_model == emb.model and manifest.embedding_dim == emb.dim
            flag = "[green]compatible[/green]" if compat else "[red]embedder mismatch[/red]"
            console.print(
                f"snapshot {manifest.persona_id}: {manifest.document_count} docs, "
                f"{manifest.chunk_count} chunks, {manifest.embedding_model} — {flag}"
            )
            ok = ok and compat
    finally:
        ctx.close()
    if not ok:
        raise typer.Exit(1)
    console.print("[green]all good[/green]")


@app.command()
def stats() -> None:
    """Node counts, overall and per persona."""
    ctx = State.context()
    try:
        s = ctx.store.stats()
    finally:
        ctx.close()
    table = Table(title="graph")
    table.add_column("label")
    table.add_column("count", justify="right")
    for label in ("personas", "sources", "documents", "chunks", "speakers", "topics", "entities"):
        table.add_row(label, str(getattr(s, label)))
    console.print(table)
    for pid, counts in s.per_persona.items():
        console.print(f"  {pid}: {counts['documents']} documents, {counts['chunks']} chunks")


# ----------------------------------------------------------------------------- ingest


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Root folder that contains the source files.")],
    persona: Annotated[str, typer.Option("--persona", "-p", help="Persona id to ground.")],
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Source id from persona.yaml.")
    ] = None,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = False,
    replace: Annotated[bool, typer.Option(help="Delete the persona's graph first.")] = False,
) -> None:
    """Load, chunk, embed and write one persona source into the graph."""
    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        if not spec.sources:
            err.print(f"[red]persona {persona} has no sources in persona.yaml[/red]")
            raise typer.Exit(2)
        if source is None:
            if len(spec.sources) > 1:
                err.print("[red]--source is required when a persona has several sources[/red]")
                raise typer.Exit(2)
            src = spec.sources[0]
        else:
            matches = [s for s in spec.sources if s.id == source]
            if not matches:
                err.print(f"[red]unknown source {source!r} for persona {persona}[/red]")
                raise typer.Exit(2)
            src = matches[0]
        if replace:
            ctx.store.delete_persona(spec.id)
        pipeline = IngestPipeline(ctx.store, ctx.embedder, progress=_progress)
        report = pipeline.ingest(path, spec, src)
        console.print(
            f"[green]ingested[/green] {report.documents} documents / {report.chunks} chunks "
            f"in {report.seconds:.0f}s with {report.embedding_model}"
        )
        if report.skipped:
            err.print(f"[yellow]skipped (empty): {len(report.skipped)} files[/yellow]")
        if export:
            manifest = snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.embedder.model_name,
                embedding_dim=ctx.embedder.dim,
                source_commit=_git_commit(path),
            )
            console.print(f"snapshot exported to {snap.snapshot_dir(ctx.snapshots_dir, spec.id)}")
            console.print(json.dumps(snap.manifest_summary(manifest), indent=2))
    finally:
        ctx.close()


@app.command()
def enrich(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    limit: Annotated[int, typer.Option(help="Max documents to enrich this run.")] = 10,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
) -> None:
    """Extract entities/relations with Claude (structured outputs) for un-enriched documents."""
    import os

    from graphrag.extract.llm import AnthropicExtractionClient, enrich_documents

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            err.print("[red]ANTHROPIC_API_KEY is not set; add it to .env to run enrichment[/red]")
            raise typer.Exit(2)
        done = ctx.store.enriched_doc_ids(spec.id)
        todo = [d for d in ctx.store.iter_documents(spec.id) if d.id not in done][:limit]
        if not todo:
            console.print("nothing to enrich")
            return
        client = AnthropicExtractionClient(model=ctx.settings.enrich_model)
        result = enrich_documents(
            ctx.store,
            client,
            spec,
            todo,
            max_chars=ctx.settings.enrich_max_chunk_chars,
            on_progress=_progress,
        )
        console.print(
            f"[green]enriched[/green] {len(todo)} documents: {len(result.entities)} entities, "
            f"{len(result.mentions)} mentions, {len(result.relations)} relations"
        )
        if export:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.embedder.model_name,
                embedding_dim=ctx.embedder.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


@app.command("enrich-import")
def enrich_import(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    files: Annotated[list[Path], typer.Argument(help="JSON files: {doc_id, entities, relations}.")],
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and report only; write nothing.")
    ] = False,
) -> None:
    """Load entity/relation extractions produced by an agent (no API key needed).

    Reports entity names that do not occur verbatim in any passage (their mention falls back
    to the first passage), so reviewers can spot non-canonical names.
    """
    from graphrag.extract.llm import DocumentExtraction, match_chunks, result_to_enrichment

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        total_entities = total_mentions = total_relations = 0
        problems = 0
        for file in files:
            try:
                payload = DocumentExtraction.model_validate_json(file.read_text(encoding="utf-8"))
            except ValueError as exc:
                err.print(f"[red]{file}: invalid: {exc}[/red]")
                problems += 1
                continue
            chunks = ctx.store.document_chunks(payload.doc_id, 0, 100_000)
            if not chunks:
                err.print(f"[red]{file}: unknown document {payload.doc_id}[/red]")
                problems += 1
                continue
            tiers = {e.name: match_chunks(e.name.strip(), chunks)[1] for e in payload.entities}
            unmatched = [n for n, tier in tiers.items() if tier == "none"]
            loose = [n for n, tier in tiers.items() if tier == "loose"]
            names = {e.name.strip().lower() for e in payload.entities}
            dangling = [
                f"{r.source} -> {r.target}"
                for r in payload.relations
                if r.source.strip().lower() not in names or r.target.strip().lower() not in names
            ]
            enrichment = result_to_enrichment(payload, chunks)
            if not dry_run:
                ctx.store.upsert_enrichment(enrichment)
            total_entities += len(enrichment.entities)
            total_mentions += len(enrichment.mentions)
            total_relations += len(enrichment.relations)
            console.print(
                f"{payload.doc_id}: {len(enrichment.entities)} entities, "
                f"{len(enrichment.mentions)} mentions, {len(enrichment.relations)} relations"
                + (f"; {len(loose)} loosely matched" if loose else "")
                + (f"; [yellow]{len(unmatched)} unmatched names[/yellow]" if unmatched else "")
                + (f"; [yellow]{len(dangling)} dangling relations[/yellow]" if dangling else "")
            )
            for name in loose:
                err.print(f"  loose: {name}")
            for name in unmatched:
                err.print(f"  unmatched: {name}")
            for rel in dangling:
                err.print(f"  dangling: {rel}")
        verb = "validated" if dry_run else "imported"
        console.print(
            f"[green]{verb}[/green] {len(files) - problems} files: {total_entities} entities, "
            f"{total_mentions} mentions, {total_relations} relations"
        )
        if problems:
            raise typer.Exit(2)
        if export and not dry_run:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.embedder.model_name,
                embedding_dim=ctx.embedder.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- snapshots


@snapshot_app.command("list")
def snapshot_list() -> None:
    """List committed snapshots."""
    cfg = _settings()
    manifests = snap.list_snapshots(cfg.snapshots_dir)
    if not manifests:
        console.print(f"no snapshots under {cfg.snapshots_dir}")
        return
    table = Table(title="snapshots")
    for col in ("persona", "documents", "chunks", "entities", "model", "created"):
        table.add_column(col)
    for m in manifests:
        table.add_row(
            m.persona_id,
            str(m.document_count),
            str(m.chunk_count),
            str(m.entity_count),
            m.embedding_model,
            m.created_at.strftime("%Y-%m-%d"),
        )
    console.print(table)


@snapshot_app.command("export")
def snapshot_export(persona: Annotated[str, typer.Argument(help="Persona id.")]) -> None:
    """Write data/snapshots/<persona>/ from the live graph."""
    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        manifest = snap.export_snapshot(
            ctx.store,
            spec,
            ctx.snapshots_dir,
            embedding_model=ctx.embedder.model_name,
            embedding_dim=ctx.embedder.dim,
        )
        console.print(json.dumps(snap.manifest_summary(manifest), indent=2))
    finally:
        ctx.close()


@snapshot_app.command("load")
def snapshot_load(
    persona: Annotated[str | None, typer.Argument(help="Persona id.")] = None,
    load_all: Annotated[bool, typer.Option("--all", help="Load every snapshot.")] = False,
) -> None:
    """Load committed snapshot(s) into Neo4j."""
    ctx = State.context()
    try:
        ids = (
            [m.persona_id for m in snap.list_snapshots(ctx.snapshots_dir)]
            if load_all
            else ([persona] if persona else [])
        )
        if not ids:
            err.print("[red]give a persona id or --all[/red]")
            raise typer.Exit(2)
        for pid in ids:
            spec = ctx.registry.get(pid)
            manifest = snap.load_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.embedder.model_name,
                embedding_dim=ctx.embedder.dim,
                on_progress=_progress,
            )
            console.print(
                f"[green]{pid}[/green]: {manifest.document_count} docs, "
                f"{manifest.chunk_count} chunks"
            )
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- query


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    persona: Annotated[str | None, typer.Option("--persona", "-p")] = None,
    k: Annotated[int, typer.Option("-k")] = 5,
    mode: Annotated[str, typer.Option(help="hybrid | vector | fulltext")] = "hybrid",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Hybrid search; prints cited passages."""
    ctx = State.context()
    try:
        search_mode: SearchMode = mode  # type: ignore[assignment]
        hits = ctx.retriever.search(query, persona_id=persona, k=k, mode=search_mode)
    finally:
        ctx.close()
    if as_json:
        console.print_json(json.dumps([h.model_dump(mode="json") for h in hits]))
        return
    if not hits:
        console.print("no results")
        return
    for i, hit in enumerate(hits, start=1):
        console.rule(f"[{i}] {hit.citation()}  score={hit.score:.4f} via {','.join(hit.methods)}")
        console.print(hit.chunk.text[:700] + ("…" if len(hit.chunk.text) > 700 else ""))


@app.command()
def context(
    query: Annotated[str, typer.Argument()],
    persona: Annotated[str | None, typer.Option("--persona", "-p")] = None,
    k: Annotated[int | None, typer.Option("-k")] = None,
    expand: Annotated[int | None, typer.Option(help="Neighbour chunks to include.")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build the context pack an agent would receive (prompt-ready text)."""
    ctx = State.context()
    try:
        spec = ctx.registry.get(persona) if persona else None
        pack = ctx.retriever.context(query, persona=spec, k=k, expand=expand)
    finally:
        ctx.close()
    if as_json:
        console.print_json(pack.model_dump_json())
    else:
        console.print(pack.to_prompt(), markup=False)


# ----------------------------------------------------------------------------- personas


@persona_app.command("list")
def persona_list() -> None:
    """Personas defined under personas/."""
    cfg = _settings()
    from graphrag.personas.registry import PersonaRegistry

    table = Table(title="personas")
    for col in ("id", "name", "sources", "sdlc stages"):
        table.add_column(col)
    for p in PersonaRegistry(cfg.personas_dir).all():
        table.add_row(p.id, p.name, ", ".join(s.id for s in p.sources), ", ".join(p.sdlc_stages))
    console.print(table)


@persona_app.command("show")
def persona_show(persona: Annotated[str, typer.Argument()]) -> None:
    """Print persona.yaml as JSON."""
    cfg = _settings()
    from graphrag.personas.registry import PersonaRegistry

    console.print_json(PersonaRegistry(cfg.personas_dir).get(persona).model_dump_json())


@persona_app.command("brief")
def persona_brief(persona: Annotated[str, typer.Argument()]) -> None:
    """Print the brief an agent uses to assume the persona (with live graph stats)."""
    ctx = State.context()
    try:
        console.print(build_brief(ctx.registry.get(persona), ctx.store), markup=False)
    finally:
        ctx.close()


@persona_app.command("new")
def persona_new(
    name: Annotated[str, typer.Argument(help="Display name, e.g. 'Support Engineer'.")],
    description: Annotated[str, typer.Option("--description", "-d")] = "",
    persona_id: Annotated[str | None, typer.Option("--id")] = None,
) -> None:
    """Scaffold personas/<id>/persona.yaml + README with the add-sources runbook."""
    cfg = _settings()
    from graphrag.personas.registry import PersonaRegistry

    spec = PersonaRegistry(cfg.personas_dir).scaffold(name, description, persona_id)
    console.print(f"created {cfg.personas_dir / spec.id / 'persona.yaml'}")


@persona_app.command("export-skill")
def persona_export_skill(
    persona: Annotated[str | None, typer.Argument()] = None,
    export_all: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Write .claude/skills/persona-<id>/SKILL.md so Claude Code can assume the persona."""
    ctx = State.context()
    try:
        specs = ctx.registry.all() if export_all else [ctx.registry.get(persona or "")]
        for spec in specs:
            path = export_persona_skill(spec, build_brief(spec, ctx.store), ctx.settings.skills_dir)
            console.print(f"wrote {path}")
    finally:
        ctx.close()


@persona_app.command("export")
def persona_export(
    persona: Annotated[str, typer.Argument(help="Persona id to pack.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output file.")] = None,
    include_graph: Annotated[
        bool, typer.Option("--graph/--no-graph", help="Include the built graph.")
    ] = True,
    include_enrichment: Annotated[
        bool, typer.Option("--enrichment/--no-enrichment", help="Include extraction JSON.")
    ] = True,
    compact: Annotated[
        bool,
        typer.Option(
            "--compact",
            help="Store vectors as float16: ~35% smaller, no measurable effect on ranking.",
        ),
    ] = False,
    with_model: Annotated[
        bool,
        typer.Option(
            "--with-model",
            help="Also carry the embedding model (~63 MB). Use for air-gapped or archival copies.",
        ),
    ] = False,
    notes: Annotated[str, typer.Option(help="Free-text note for the recipient.")] = "",
) -> None:
    """Pack a persona (and its graph) into one file you can send to someone."""
    from graphrag.personas.bundle import BundleError, export_bundle

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        target = out or Path(f"{spec.id}-persona.tar.gz")
        _warn_if_ephemeral(target)
        try:
            manifest = export_bundle(
                spec,
                target,
                personas_dir=ctx.settings.personas_dir,
                snapshots_dir=ctx.snapshots_dir,
                enrichment_dir=ctx.settings.enrichment_dir,
                include_graph=include_graph,
                include_enrichment=include_enrichment,
                model_cache_dir=ctx.settings.model_cache_dir if with_model else None,
                compact=compact,
                notes=notes,
            )
        except BundleError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        size_mb = target.stat().st_size / 1e6
        console.print(f"[green]wrote[/green] {target} ({size_mb:.1f} MB)")
        if manifest.includes_graph:
            console.print(
                f"  graph: {manifest.document_count:,} documents, "
                f"{manifest.chunk_count:,} passages, {manifest.entity_count:,} entities"
            )
            console.print(
                f"  built with {manifest.embedding_model} ({manifest.embedding_dim}d) — "
                "the recipient needs the same embedder"
            )
            if manifest.embedding_precision == "float16":
                console.print("  vectors stored as float16 (--compact)")
        else:
            console.print("  definition only, no graph")
        if manifest.enrichment_files:
            console.print(f"  {manifest.enrichment_files} extraction files")
        if manifest.includes_model:
            console.print(
                f"  embedding model included ({manifest.model_bytes / 1e6:.0f} MB) — "
                "the recipient needs no network"
            )
        elif manifest.includes_graph:
            console.print("  no model: they run `make model` (or `--with-model` next time)")
        console.print(f"\nThey run: [bold]graphrag persona import {target.name} --load[/bold]")
    finally:
        ctx.close()


@persona_app.command("import")
def persona_import(
    bundle: Annotated[Path, typer.Argument(help="Bundle file you were sent.")],
    load: Annotated[bool, typer.Option("--load", help="Load the graph into Neo4j too.")] = False,
    include_graph: Annotated[
        bool, typer.Option("--graph/--no-graph", help="Unpack the bundled graph.")
    ] = True,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing persona.")
    ] = False,
) -> None:
    """Install a persona someone sent you, optionally loading its graph."""
    from graphrag.personas.bundle import BundleError, import_bundle

    ctx = State.context()
    try:
        try:
            report = import_bundle(
                bundle,
                personas_dir=ctx.settings.personas_dir,
                snapshots_dir=ctx.snapshots_dir,
                enrichment_dir=ctx.settings.enrichment_dir,
                model_name=ctx.embedder.model_name,
                dim=ctx.embedder.dim,
                model_cache_dir=ctx.settings.model_cache_dir,
                include_graph=include_graph,
                overwrite=overwrite,
            )
        except BundleError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

        m = report.manifest
        console.print(f"[green]installed[/green] {m.persona_id} ({m.persona_name})")
        if m.notes:
            console.print(f"  note from sender: {m.notes}")
        if report.snapshot_path:
            console.print(
                f"  graph: {m.document_count:,} documents, {m.chunk_count:,} passages, "
                f"{m.entity_count:,} entities"
            )
        if report.enrichment_paths:
            console.print(f"  {len(report.enrichment_paths)} extraction files")
        if report.model_restored:
            console.print(f"  embedding model restored ({m.embedding_model})")

        spec = ctx.registry.get(m.persona_id)
        if load and report.snapshot_path:
            snap.load_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.embedder.model_name,
                embedding_dim=ctx.embedder.dim,
                on_progress=_progress,
            )
            report.loaded = True
            console.print("  [green]loaded into Neo4j[/green]")
        path = export_persona_skill(spec, build_brief(spec, ctx.store), ctx.settings.skills_dir)
        console.print(f"  skill written: {path}")
        if report.snapshot_path and not load:
            console.print(f"\nRun [bold]graphrag snapshot load {m.persona_id}[/bold] to query it.")
    finally:
        ctx.close()


@persona_app.command("recommend")
def persona_recommend(stage: Annotated[str, typer.Argument(help="SDLC stage.")]) -> None:
    """Which personas apply to an SDLC stage (discovery, requirements, design, testing, ...)."""
    cfg = _settings()
    from graphrag.personas.registry import PersonaRegistry

    matches = PersonaRegistry(cfg.personas_dir).recommend(stage)
    if not matches:
        console.print(f"no persona covers {stage!r}; known stages: {', '.join(SDLC_STAGES)}")
        return
    for p in matches:
        console.print(f"{p.id}: {p.description}")


# ----------------------------------------------------------------------------- servers


@app.command()
def serve(
    transport: Annotated[str, typer.Option(help="http | stdio")] = "http",
) -> None:
    """Run the MCP server (FastMCP)."""
    from graphrag.mcp_server import run_server

    run_server(transport=transport, settings=_settings())


@app.command("embed-server")
def embed_server() -> None:
    """Run the OpenAI-compatible embedding server (for a GPU box such as a GB10)."""
    from graphrag.embed.server import main as embed_main

    embed_main(_settings())


if __name__ == "__main__":
    app()
