"""``graphrag`` command line: setup, ingest, snapshots, search, personas, enrichment, servers."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.table import Table

from graphrag import __version__
from graphrag.app import AppContext
from graphrag.config import Settings, load_settings
from graphrag.graph import snapshot as snap
from graphrag.models import SearchMode, SourceSpec
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

#: Default label shuffles behind ``sna analyze --by``. Repeated from
#: :data:`graphrag.sna.attributes.PERMUTATIONS` rather than imported, because a default in a
#: signature is evaluated at import time and the analysis package pulls in numpy and sklearn;
#: a unit test holds the two to the same number.
WHERE_PERMUTATIONS = 200


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


FLAGGED_LINES = 20


def _print_flagged(flagged: list[tuple[str, str]]) -> None:
    """Report files whose raw text looked like an injection attempt. Advisory: they were
    ingested. Excerpts are printed with markup off so bracketed chat tokens survive."""
    if not flagged:
        return
    err.print(f"[yellow]flagged {len(flagged)} file(s) carrying untrusted-text patterns[/yellow]")
    for path, reason in flagged[:FLAGGED_LINES]:
        err.print(f"  {path}: {reason}", style="yellow", markup=False, highlight=False)
    if len(flagged) > FLAGGED_LINES:
        err.print(f"  (+{len(flagged) - FLAGGED_LINES} more)", style="yellow", markup=False)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        ctx.store.ensure_schema(ctx.settings.embedding.dim)
        console.print(
            f"schema ready (dim={ctx.settings.embedding.dim}, model={ctx.settings.embedding.model})"
        )
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
                    embedding_model=ctx.settings.embedding.model,
                    embedding_dim=ctx.settings.embedding.dim,
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
            vec = ctx.require_embedder().embed_query("doctor probe")
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
def validate(
    path: Annotated[Path, typer.Argument(help="Root folder of a captured corpus.")],
    merge: Annotated[
        bool,
        typer.Option("--merge", help="Write provenance/provenance.jsonl from the manifests."),
    ] = False,
) -> None:
    """Check a captured corpus against the front-matter, body-length and provenance contract.

    Every captured document must open with parsable YAML front-matter carrying `title`,
    `fetched_at`, `fetched_by`, `retrieval`, `content_fidelity`, `document_type` and `category`,
    plus a `source_url` unless it is a research note or internal context, and a body of 40 to
    6,000 words. Every provenance line must be valid JSON, and one that claims a successful
    capture must name a file that exists.

    Exits 1 when anything is wrong, 2 when the path does not exist.
    """
    from graphrag.ingest.validate import merge_provenance, validate_corpus

    if not path.is_dir():
        err.print(f"[red]not a directory: {path}[/red]")
        raise typer.Exit(2)
    report = validate_corpus(path)
    for line in report.problem_lines():
        console.print(line, markup=False, highlight=False)
    if merge:
        out = merge_provenance(path, report.provenance)
        console.print(f"merged {len(report.provenance)} provenance records -> {out}")
    for line in report.summary_lines():
        console.print(line, markup=False, highlight=False)
    if not report.ok:
        raise typer.Exit(1)


def _validate_or_exit(path: Path, source: SourceSpec) -> None:
    """Refuse to ingest a `documents` source whose captured files break the corpus contract.

    A bad capture is invisible once it is in the graph: it becomes a weak chunk that dilutes
    every answer rather than an error anyone sees. Catching it here is the only cheap moment.
    """
    from graphrag.ingest.loaders import iter_source_files
    from graphrag.ingest.validate import DOCUMENT_SUFFIXES, validate_files

    base = path / source.path if source.path else path
    files = [p for p in iter_source_files(path, source) if p.suffix.lower() in DOCUMENT_SUFFIXES]
    report = validate_files(files, root=base)
    if report.ok:
        return
    err.print(
        f"[red]{len(report.problems)} problem(s) in {len(files)} captured file(s) of source "
        f"{source.id!r}[/red]"
    )
    for line in report.problem_lines():
        err.print(f"  {line}", markup=False, highlight=False)
    err.print("[red]refusing to ingest; fix the files or pass --allow-invalid[/red]")
    raise typer.Exit(2)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Root folder that contains the source files.")],
    persona: Annotated[str, typer.Option("--persona", "-p", help="Persona id to ground.")],
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Source id from persona.yaml.")
    ] = None,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = False,
    replace: Annotated[bool, typer.Option(help="Delete the persona's graph first.")] = False,
    allow_invalid: Annotated[
        bool,
        typer.Option("--allow-invalid", help="Ingest a `documents` source without validating it."),
    ] = False,
) -> None:
    """Load, chunk, embed and write one persona source into the graph.

    A `documents` source is validated first and the ingest is refused with exit 2 if any captured
    file breaks the contract; `--allow-invalid` skips that check. `transcripts` sources are not
    validated, because the archive that produced them owns their front-matter.
    """
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
        if src.loader == "documents" and not allow_invalid:
            _validate_or_exit(path, src)
        if replace:
            ctx.store.delete_persona(spec.id)
        pipeline = IngestPipeline(ctx.store, ctx.require_embedder(), progress=_progress)
        report = pipeline.ingest(path, spec, src)
        orphans = (
            f", removed {report.orphans_removed} orphaned entities"
            if report.orphans_removed
            else ""
        )
        console.print(
            f"[green]ingested[/green] {report.documents} documents / {report.chunks} chunks "
            f"in {report.seconds:.0f}s with {report.embedding_model}{orphans}"
        )
        if report.skipped:
            err.print(f"[yellow]skipped (empty): {len(report.skipped)} files[/yellow]")
        _print_flagged(report.flagged)
        if export:
            manifest = snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
                source_commit=_git_commit(path),
            )
            console.print(f"snapshot exported to {snap.snapshot_dir(ctx.snapshots_dir, spec.id)}")
            console.print(json.dumps(snap.manifest_summary(manifest), indent=2))
    finally:
        ctx.close()


@app.command()
def sync(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Only this source id.")
    ] = None,
    refresh_attribution: Annotated[
        bool,
        typer.Option(
            "--refresh-attribution",
            help="Re-import every attribution file, not only those filling a gap.",
        ),
    ] = False,
    refresh_annotations: Annotated[
        bool,
        typer.Option(
            "--refresh-annotations",
            help="Re-import every annotation file, not only those filling a gap.",
        ),
    ] = False,
    allow_invalid: Annotated[
        bool,
        typer.Option(
            "--allow-invalid", help="Re-ingest a `documents` source without validating it."
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what is missing; write nothing.")
    ] = False,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
) -> None:
    """Ingest whatever the graph is missing for a persona and re-import its extraction JSON.

    One command instead of three: it compares the document ids the loaders would produce for
    the persona's raw files with the ids already in the graph, re-ingests only the sources that
    are behind, then re-imports their extraction files, since a re-ingest drops entity mentions.
    Sources that were already complete are still checked for documents the graph holds without
    any entity, and the extraction files for those are imported too.

    A `documents` source behind on documents is validated first, exactly as `ingest` validates
    one, and refused with exit 2 if any captured file breaks the contract; `--allow-invalid`
    skips that check. A refused source is reported and left alone; every other source, and this
    source's own backfill of documents already in the graph, still runs.

    That check asks whether a layer is missing, so a rewritten sidecar is invisible to it: an
    attribution file that gained posted dates names a document that already has speakers.
    `--refresh-attribution` and `--refresh-annotations` re-import every sidecar of that kind
    instead, whatever the graph already holds, and each source's line says which it did. A
    tagging pass that adds node attributes to files the graph already has speakers and stances
    for is exactly that case, so those flags are how attributes reach the graph.
    """
    from graphrag.sync import UnknownSourceError, summary_lines, sync_persona

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        try:
            report = sync_persona(
                spec,
                store=ctx.store,
                embedder=ctx.require_embedder,
                raw_root=ctx.settings.raw_dir / spec.id,
                enrichment_root=ctx.settings.enrichment_dir,
                attribution_root=ctx.settings.attribution_dir,
                annotation_root=ctx.settings.annotations_dir,
                aliases=_alias_table(ctx, spec.id),
                facets=_facet_table(ctx, spec.id),
                attributes=_attribute_table(ctx, spec.id),
                source_id=source,
                refresh_attribution=refresh_attribution,
                refresh_annotations=refresh_annotations,
                allow_invalid=allow_invalid,
                dry_run=dry_run,
                progress=_progress,
            )
        except UnknownSourceError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        for line in summary_lines(report):
            console.print(line, markup=False, highlight=False)
        if report.errors:
            err.print(f"[yellow]{len(report.errors)} files failed to import[/yellow]")
        if report.invalid_sources:
            err.print(
                f"[red]{len(report.invalid_sources)} source(s) refused; fix the files or pass "
                "--allow-invalid[/red]"
            )
        if export and report.wrote:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
            )
            console.print(f"snapshot exported to {snap.snapshot_dir(ctx.snapshots_dir, spec.id)}")
        if report.errors or report.invalid_sources:
            raise typer.Exit(2)
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
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
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
    from graphrag.extract.importer import import_extraction_file

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        table = _alias_table(ctx, spec.id)
        total_entities = total_mentions = total_relations = 0
        problems = 0
        for file in files:
            result = import_extraction_file(ctx.store, file, dry_run=dry_run, aliases=table)
            if not result.ok:
                err.print(f"[red]{file}: {result.error}[/red]")
                problems += 1
                continue
            total_entities += result.entities
            total_mentions += result.mentions
            total_relations += result.relations
            console.print(
                f"{result.doc_id}: {result.entities} entities, "
                f"{result.mentions} mentions, {result.relations} relations"
                + (f"; {len(result.loose)} loosely matched" if result.loose else "")
                + (
                    f"; [yellow]{len(result.unmatched)} unmatched names[/yellow]"
                    if result.unmatched
                    else ""
                )
                + (
                    f"; [yellow]{len(result.dangling)} dangling relations[/yellow]"
                    if result.dangling
                    else ""
                )
                + (
                    f"; [yellow]{len(result.collisions)} id collisions[/yellow]"
                    if result.collisions
                    else ""
                )
            )
            for name in result.loose:
                err.print(f"  loose: {name}")
            for name in result.unmatched:
                err.print(f"  unmatched: {name}")
            for rel in result.dangling:
                err.print(f"  dangling: {rel}")
            for renamed in result.renamed:
                err.print(f"  alias: {renamed}")
            for collision in result.collisions:
                err.print(f"  collision: {collision}")
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
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
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
            embedding_model=ctx.settings.embedding.model,
            embedding_dim=ctx.settings.embedding.dim,
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
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
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
                model_name=ctx.settings.embedding.model,
                dim=ctx.settings.embedding.dim,
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
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
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


config_app = typer.Typer(help="Inspect and write configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write; - for stdout.")
    ] = Path(".env"),
    backend: Annotated[str, typer.Option(help="fastembed | http | hash")] = "fastembed",
    base_url: Annotated[str | None, typer.Option(help="Endpoint for the http backend.")] = None,
    neo4j_password: Annotated[str | None, typer.Option(help="Neo4j password.")] = None,
    allow_download: Annotated[bool, typer.Option(help="Permit fetching model weights.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a .env. Keys come from the settings models, so they cannot drift from the code."""
    from graphrag.config import EmbeddingSettings, Neo4jSettings, Settings, render_dotenv

    to_stdout = str(output) == "-"
    if not to_stdout and output.exists() and not force:
        err.print(f"[red]{output} exists; pass --force to overwrite[/red]")
        raise typer.Exit(2)

    neo4j = Neo4jSettings()
    if neo4j_password:
        neo4j = neo4j.model_copy(update={"password": SecretStr(neo4j_password)})
    embedding = EmbeddingSettings().model_copy(
        update={"backend": backend, "base_url": base_url, "allow_download": allow_download}
    )
    settings = Settings().model_copy(update={"neo4j": neo4j, "embedding": embedding})

    rendered = render_dotenv(settings, header=f"Written by `graphrag config init` on {_utc_now()}.")
    if to_stdout:
        # Bare print: the caller is redirecting this into a file, so no markup or wrapping.
        print(rendered, end="")
        return
    output.write_text(rendered, encoding="utf-8")
    console.print(f"[green]wrote[/green] {output} (backend={backend}, downloads={allow_download})")


@config_app.command("show")
def config_show() -> None:
    """Print the effective configuration, secrets redacted."""
    cfg = _settings()
    redacted = "***"
    data = json.loads(cfg.model_dump_json())
    data["neo4j"]["password"] = redacted
    if data["embedding"].get("api_key"):
        data["embedding"]["api_key"] = redacted
    console.print_json(json.dumps(data))


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


# ----------------------------------------------------------------------------- attribution


@app.command("attribution-import")
def attribution_import(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    files: Annotated[list[Path], typer.Argument(help="JSON files: {doc_id, posts}.")],
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and report only; write nothing.")
    ] = False,
) -> None:
    """Attach speakers to the passages they wrote, from JSON an agent produced.

    For documents whose loader parsed no speaker turns, such as a captured discussion thread:
    each post names a speaker and quotes its opening words, and those words are looked for in
    the document's passages. Reports posts whose anchor occurs in no passage (they are skipped,
    never guessed at), so a reviewer can fix the quote before writing.

    A post may also carry `attributes`, which describe the speaker rather than the post and go
    on the speaker: `{"region": "north"}`. They are checked against the `attributes:` section of
    `personas/<id>/facets.yaml` when the persona declares one; a key or value outside it is
    reported and dropped. The first value written for a speaker stands, and a later post that
    disagrees is reported as a conflict rather than overwriting it.
    """
    from graphrag.extract.attributes import AttributeTableError
    from graphrag.extract.attribution import import_attribution_file, report_lines

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        try:
            vocabulary = _attribute_table(ctx, spec.id)
        except AttributeTableError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        total_posts = total_attached = total_loose = 0
        total_attributes = total_invalid = total_conflicts = 0
        voices: set[str] = set()
        problems = 0
        for file in files:
            result = import_attribution_file(
                ctx.store, file, dry_run=dry_run, attributes=vocabulary
            )
            if not result.ok:
                err.print(f"[red]{file}: {result.error}[/red]")
                problems += 1
                continue
            total_posts += result.posts
            total_attached += result.attached
            total_loose += len(result.loose)
            total_attributes += result.attributes
            total_invalid += len(result.attribute_problems)
            total_conflicts += len(result.conflicts)
            voices.update(result.speakers)
            for line in report_lines(result):
                console.print(line, markup=False, highlight=False)
        verb = "validated" if dry_run else "imported"
        console.print(
            f"[green]{verb}[/green] {len(files) - problems} files: {total_attached}/{total_posts} "
            f"posts attached, {len(voices)} speakers, {total_loose} loose anchors"
            + (f", {total_attributes} speaker attributes" if total_attributes else "")
            + (f", {total_invalid} attribute problems" if total_invalid else "")
            + (f", {total_conflicts} attribute conflicts" if total_conflicts else "")
        )
        if problems:
            raise typer.Exit(2)
        if export and not dry_run:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- network analysis


sna_app = typer.Typer(
    help=(
        "Build, measure and cluster the graph's networks (speakers, entities, topics, "
        "speakers-entities), filtered by stance, facet or date window."
    ),
    no_args_is_help=True,
)
app.add_typer(sna_app, name="sna")


def _parse_k_range(text: str) -> list[int]:
    """``2-10`` or ``3,5,8`` into a list of candidate k values."""
    try:
        if "-" in text:
            low, high = (int(part) for part in text.split("-", 1))
            return list(range(low, high + 1))
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"--k-range must look like 2-10 or 3,5,8, got {text!r}"
        raise typer.BadParameter(msg) from exc


def _network_or_exit(network: str) -> str:
    from graphrag.sna.export import NETWORKS

    if network not in NETWORKS:
        err.print(f"[red]--network must be one of {', '.join(NETWORKS)}[/red]")
        raise typer.Exit(2)
    return network


def _stances_or_exit(values: list[str] | None) -> list[str] | None:
    """``--stance`` may be repeated; every value must be one the annotation layer writes."""
    from graphrag.sna.export import STANCES

    if not values:
        return None
    cleaned = [v.strip().lower() for v in values if v.strip()]
    unknown = [v for v in cleaned if v not in STANCES]
    if unknown:
        err.print(
            f"[red]--stance must be one of {', '.join(STANCES)}; got {', '.join(unknown)}[/red]"
        )
        raise typer.Exit(2)
    return cleaned or None


def _project_or_exit(value: str | None) -> Any:
    from graphrag.sna.export import PROJECTIONS

    if value is None:
        return None
    if value not in PROJECTIONS:
        err.print(f"[red]--project must be one of {', '.join(PROJECTIONS)}[/red]")
        raise typer.Exit(2)
    return value


def _clean(values: list[str] | None) -> list[str] | None:
    """A repeated string option with the blanks dropped, or ``None`` when nothing was given."""
    cleaned = [v.strip() for v in (values or []) if v.strip()]
    return cleaned or None


def _where_or_exit(values: list[str] | None, option: str = "--where") -> dict[str, str] | None:
    """``--where key=value``, repeatable, into a mapping. Exits 2 on anything it cannot read.

    A key given twice is refused rather than resolved: a node holds one value per key, so
    ``--where region=north --where region=south`` can only ever match nothing, and silently
    keeping the last one would answer a question nobody asked.
    """
    if not values:
        return None
    where: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            err.print(f"[red]{option} must look like key=value, got {item!r}[/red]")
            raise typer.Exit(2)
        if key in where:
            err.print(
                f"[red]{option} {key} given twice; a node holds one value per key, so no node "
                "could match both[/red]"
            )
            raise typer.Exit(2)
        where[key] = value
    return where


def _network_or_bad_parameter(build: Callable[[], Any]) -> Any:
    """Build a network, turning the builder's rejection of a filter combination into exit 2."""
    try:
        return build()
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@sna_app.command("export")
def sna_export(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Target .graphml or .json file.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities")
    ] = "speakers",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Drop edges below this weight.")
    ] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(
            "--where",
            help="Keep only nodes whose attribute matches, as key=value; repeatable.",
        ),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(help="Collapse the two-mode network onto speakers | entities."),
    ] = None,
) -> None:
    """Write one of the graph's networks to GraphML or node-link JSON.

    `--where key=value` keeps only nodes attributed that way: a speaker by what an attribution
    pass recorded about them, an entity or a topic by the attributes of the document its passage
    belongs to. A node with no value for that key is left out rather than treated as a group of
    its own. Every node carries its attributes into the file as `attr_<key>`.
    """
    from graphrag.sna.export import build_network, write_graph

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    ctx = State.context()
    try:
        ctx.registry.get(persona)  # fail fast on an unknown persona
        graph = _network_or_bad_parameter(
            lambda: build_network(
                ctx.store,
                network,
                persona,
                source_id=source,
                min_weight=min_weight,
                types=[t.strip() for t in types.split(",") if t.strip()] if types else None,
                stances=stances,
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    _warn_if_ephemeral(out)
    write_graph(graph, out)
    console.print(
        f"[green]wrote[/green] {out}: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges ({network})"
    )


@sna_app.command("analyze")
def sna_analyze(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Markdown report to write.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities")
    ] = "speakers",
    method: Annotated[str, typer.Option(help="louvain | kmeans | gmm")] = "louvain",
    k: Annotated[int | None, typer.Option("-k", help="Force k; otherwise it is chosen.")] = None,
    k_range: Annotated[str, typer.Option("--k-range", help="Candidates, e.g. 2-10.")] = "2-10",
    resolution: Annotated[float, typer.Option(help="Louvain resolution.")] = 1.0,
    runs: Annotated[int, typer.Option(help="Louvain seeds to compare for stability.")] = 10,
    features: Annotated[str, typer.Option(help="spectral | embedding")] = "spectral",
    dims: Annotated[int, typer.Option(help="Spectral embedding dimensions.")] = 8,
    covariance: Annotated[str, typer.Option(help="GMM covariance: full|tied|diag|spherical.")] = (
        "full"
    ),
    samples: Annotated[int, typer.Option(help="Null-model rewirings.")] = 50,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight")] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(
            "--where",
            help="Keep only nodes whose attribute matches, as key=value; repeatable.",
        ),
    ] = None,
    by: Annotated[
        str | None,
        typer.Option(
            "--by",
            help="Also report whether the network divides along this node attribute.",
        ),
    ] = None,
    permutations: Annotated[
        int, typer.Option(help="Label shuffles for the --by assortativity null.")
    ] = WHERE_PERMUTATIONS,
    project: Annotated[
        str | None,
        typer.Option(help="Collapse the two-mode network onto speakers | entities."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Fix every random seed.")] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
) -> None:
    """Measure a network, group it, check the grouping, and write a markdown report.

    `--by <key>` adds an attribute section: how many nodes carry each value, how far edges join
    like to like (assortativity, against a null that shuffles the labels), how the partition by
    that attribute scores against the best Louvain grouping and against a degree-preserving
    null, and how far the groups this run found agree with the attribute. It answers whether
    the network divides along something already known about its nodes, which is a question the
    grouping itself cannot be asked.

    `--where key=value` narrows the network to nodes attributed that way before any of it runs.
    """
    from graphrag.sna.analysis import METHODS, render_markdown, run_analysis, to_payload
    from graphrag.sna.export import build_network

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if method not in METHODS:
        err.print(f"[red]--method must be one of {', '.join(METHODS)}[/red]")
        raise typer.Exit(2)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network(
                ctx.store,
                network,
                persona,
                source_id=source,
                min_weight=min_weight,
                types=[t.strip() for t in types.split(",") if t.strip()] if types else None,
                stances=stances,
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
        try:
            analysis = run_analysis(
                ctx.store,
                graph,
                persona_id=persona,
                network=network,
                method=method,
                k=k,
                k_range=_parse_k_range(k_range),
                resolution=resolution,
                runs=runs,
                features=features,  # type: ignore[arg-type]
                dims=dims,
                covariance=covariance,
                samples=samples,
                by=by.strip() if by else None,
                permutations=permutations,
                seed=seed,
            )
        except ValueError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
    finally:
        ctx.close()

    _warn_if_ephemeral(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(analysis), encoding="utf-8")
    console.print(
        f"[green]wrote[/green] {out}: {int(analysis.summary['nodes']):,} nodes, "
        f"{int(analysis.summary['edges']):,} edges, {len(analysis.groups)} "
        f"{analysis.group_noun}(s)"
    )
    if analysis.louvain_result is not None:
        console.print(
            f"  modularity {analysis.louvain_result.modularity:.4f}, "
            f"stability (ARI) {analysis.louvain_result.stability:.3f}"
        )
    if analysis.null_model is not None:
        console.print(
            f"  null model z={analysis.null_model.z_score:.2f} — {analysis.null_model.verdict}"
        )
    for note in analysis.notes:
        err.print(f"  note: {note}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        as_json.write_text(json.dumps(to_payload(analysis), indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("stances")
def sna_stances(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Markdown report to write.")],
    entity: Annotated[
        list[str] | None,
        typer.Option("--entity", help="Limit to this entity by name; repeatable."),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(
            "--where",
            help="Keep only documents whose attribute matches, as key=value; repeatable.",
        ),
    ] = None,
    quotes: Annotated[int, typer.Option(help="Verbatim passages to quote per stance.")] = 3,
) -> None:
    """Report what the corpus says about each entity, not merely how often it names it.

    Reads the annotation layer: per entity, how many passages praise it, complain about it,
    replace it or mention it neutrally, who wrote each kind, which facets those passages carry,
    and up to `--quotes` verbatim passages per stance. Ends with a signed co-mention table, so
    entities praised together are separated from entities complained about together.

    Every count is a count of annotations. An entity with no complaints has no annotated
    complaints, which is not the same as no complaints.

    `--where key=value` keeps only the annotations whose document is attributed that way, which
    is how one population's reading of a thing is separated from another's. Run it once per
    value and compare the two reports; the filtered table and the unfiltered one have different
    denominators and do not subtract.
    """
    from graphrag.sna.stances import build_stance_report, render_stances

    filters = _where_or_exit(where)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        report = build_stance_report(
            ctx.store,
            persona,
            source_id=source,
            entities=_clean(entity) or [],
            facets=_clean(facet) or [],
            where=filters,
            quotes_per_stance=max(0, quotes),
        )
    finally:
        ctx.close()

    _warn_if_ephemeral(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_stances(report), encoding="utf-8")
    console.print(
        f"[green]wrote[/green] {out}: {report.annotations:,} annotated mentions over "
        f"{len(report.entities):,} entities, {len(report.pairs):,} co-mentioned pairs"
    )
    for name in report.missing:
        err.print(f"  no annotated mentions for: {name}", markup=False, highlight=False)


@sna_app.command("compare")
def sna_compare(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Markdown report to write.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities")
    ] = "speakers",
    since: Annotated[str | None, typer.Option(help="First window starts, ISO.")] = None,
    until: Annotated[str | None, typer.Option(help="First window ends, ISO.")] = None,
    since2: Annotated[str | None, typer.Option("--since2", help="Second window starts.")] = None,
    until2: Annotated[str | None, typer.Option("--until2", help="Second window ends.")] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight")] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(
            "--where",
            help="Attribute filter for the first build, as key=value; repeatable.",
        ),
    ] = None,
    where2: Annotated[
        list[str] | None,
        typer.Option(
            "--where2",
            help="Attribute filter for the second build; defaults to --where. Repeatable.",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    centrality: Annotated[
        str, typer.Option(help="Which ranking the rank changes are read from.")
    ] = "weighted_degree",
    resolution: Annotated[float, typer.Option(help="Louvain resolution.")] = 1.0,
    runs: Annotated[int, typer.Option(help="Louvain seeds to compare for stability.")] = 10,
    seed: Annotated[int | None, typer.Option(help="Fix every random seed.")] = None,
) -> None:
    """Build one network over two time windows and report what changed between them.

    Reports n for each window, which nodes entered and which left, how far the two partitions
    agree over the nodes both windows hold (adjusted Rand index and normalised mutual
    information), and the ten largest centrality rank changes among those nodes.

    Community numbers are not comparable between two Louvain runs, so nothing here says
    "community 2 grew"; a window is read from dated passages, so an undated document is in
    neither window.

    The two builds need not differ by a window. `--where key=value` and `--where2 key=other`
    build the same network over two populations of one corpus and report them exactly as two
    windows are reported: counts first, then structure. `--where2` defaults to `--where`, so a
    window comparison inside one population takes a single filter.
    """
    from graphrag.sna.compare import compare_windows, render_comparison

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    filters2 = _where_or_exit(where2, "--where2")
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        comparison = _network_or_bad_parameter(
            lambda: compare_windows(
                ctx.store,
                persona,
                network,
                since=since,
                until=until,
                since2=since2,
                until2=until2,
                source_id=source,
                min_weight=min_weight,
                types=[t.strip() for t in types.split(",") if t.strip()] if types else None,
                stances=stances,
                facets=_clean(facet),
                where=filters,
                where2=filters2,
                project=side,
                centrality_kind=centrality,
                resolution=resolution,
                runs=runs,
                seed=seed,
            )
        )
    finally:
        ctx.close()

    _warn_if_ephemeral(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_comparison(comparison), encoding="utf-8")
    console.print(
        f"[green]wrote[/green] {out}: {int(comparison.a.summary['nodes']):,} nodes in "
        f"{comparison.a.label}, {int(comparison.b.summary['nodes']):,} in {comparison.b.label}; "
        f"{len(comparison.shared):,} in both, {len(comparison.entered):,} entered, "
        f"{len(comparison.left):,} left"
    )
    if comparison.similarity:
        console.print(
            f"  partition similarity ARI {comparison.similarity['adjusted_rand_index']:.3f}, "
            f"NMI {comparison.similarity['normalized_mutual_information']:.3f}"
        )
    for note in comparison.notes:
        err.print(f"  note: {note}")


@sna_app.command("guide")
def sna_guide() -> None:
    """Print the method-selection rules: which network, which method, which centrality."""
    from graphrag.sna.guide import render_guide

    console.print(render_guide(), markup=False, highlight=False)


# ----------------------------------------------------------------------------- aliases


aliases_app = typer.Typer(
    help="Fold the spellings a corpus uses into one canonical entity per thing.",
    no_args_is_help=True,
)
app.add_typer(aliases_app, name="aliases")


def _alias_table(ctx: AppContext, persona_id: str) -> Any:
    """The persona's alias table, or an empty one. Absence is normal, not a warning."""
    from graphrag.extract.aliases import load_persona_aliases

    return load_persona_aliases(ctx.registry.directory, persona_id)


def _facet_table(ctx: AppContext, persona_id: str) -> Any:
    """The persona's facet vocabulary, or ``None`` when it declares none and anything goes."""
    from graphrag.extract.annotations import load_persona_facets

    return load_persona_facets(ctx.registry.directory, persona_id)


def _attribute_table(ctx: AppContext, persona_id: str) -> Any:
    """The persona's attribute vocabulary, out of the same file. Empty means anything goes."""
    from graphrag.extract.attributes import load_persona_attributes

    return load_persona_attributes(ctx.registry.directory, persona_id)


@aliases_app.command("apply")
def aliases_apply(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would move; write nothing.")
    ] = False,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
) -> None:
    """Fold alias spellings of an entity into one node, from `personas/<id>/aliases.yaml`.

    Extraction keeps the spelling each passage uses, so one thing arrives as several nodes and
    every count over it is low. This re-points that persona's mentions and relations onto the
    canonical node, records the spellings on it and deletes the nodes nothing holds any more.
    """
    from graphrag.extract.aliases import (
        ALIAS_FILE,
        AliasError,
        alias_lines,
        apply_alias_table,
        preview_alias_table,
    )

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        try:
            table = _alias_table(ctx, spec.id)
        except AliasError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        if not table:
            path = ctx.registry.directory / spec.id / ALIAS_FILE
            err.print(f"[yellow]{spec.id} has no alias table at {path}[/yellow]")
            return
        report = (
            preview_alias_table(ctx.store, spec.id, table)
            if dry_run
            else apply_alias_table(ctx.store, spec.id, table)
        )
        for line in alias_lines(report):
            console.print(line, markup=False, highlight=False)
        if export and not dry_run and report.mentions_moved:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- annotations


@app.command("annotations-import")
def annotations_import(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    files: Annotated[list[Path], typer.Argument(help="JSON files: {doc_id, annotations}.")],
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and report only; write nothing.")
    ] = False,
) -> None:
    """Record what a passage says about an entity, and which functions it is about.

    Each annotation quotes a verbatim anchor, which is looked for in the document's passages the
    way attribution looks for a post. The stance goes on the mention and the facets go on the
    passage. When the passage does not mention the entity yet but the persona has it and one of
    its spellings is in the passage, the mention is created; an entity the persona does not have
    is left alone, because creating entities is the extraction pass's job.

    Skipped annotations are counted by reason -- anchor not found, entity unknown to the persona,
    entity not in the passage -- so a reviewer knows whether to fix the anchors, extend the alias
    file or extend the extraction. Facets outside `personas/<id>/facets.yaml`, when the persona
    keeps one, are reported and dropped.

    A file may also carry a top-level `attributes` object, which describes the document rather
    than any passage of it and goes on the document: `{"region": "north"}`. It is checked
    against the `attributes:` section of the same `facets.yaml`, and a file may carry attributes
    and no annotations at all.
    """
    from graphrag.extract.annotations import (
        AnnotationResult,
        EntityIndex,
        FacetError,
        import_annotation_file,
        loose_summary,
        loose_totals,
        report_lines,
    )

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        try:
            table = _alias_table(ctx, spec.id)
            vocabulary = _facet_table(ctx, spec.id)
            attributes = _attribute_table(ctx, spec.id)
        except (FacetError, ValueError) as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        # Built once: the annotation pass never creates an entity, so this cannot go stale.
        known = EntityIndex.build(ctx.store, spec.id)
        results: list[AnnotationResult] = []
        problems = 0
        for file in files:
            result = import_annotation_file(
                ctx.store,
                file,
                dry_run=dry_run,
                aliases=table,
                facets=vocabulary,
                attributes=attributes,
                entities=known,
            )
            if not result.ok:
                err.print(f"[red]{file}: {result.error}[/red]")
                problems += 1
                continue
            results.append(result)
            for line in report_lines(result):
                console.print(line, markup=False, highlight=False)
        verb = "validated" if dry_run else "imported"
        totals = loose_totals(results)
        loose = sum(totals.values())
        console.print(
            f"[green]{verb}[/green] {len(files) - problems} files: "
            f"{sum(r.applied for r in results)} annotations, "
            f"{sum(r.stances for r in results)} stances, {sum(r.facets for r in results)} facets, "
            + (
                f"{sum(r.attributes for r in results)} document attributes, "
                if any(r.attributes for r in results)
                else ""
            )
            + f"{sum(r.created for r in results)} mentions created, {loose} loose"
            + (f" ({loose_summary(totals)})" if loose else "")
            + (
                f", {sum(len(r.attribute_problems) for r in results)} attribute problems"
                if any(r.attribute_problems for r in results)
                else ""
            )
        )
        if problems:
            raise typer.Exit(2)
        if export and not dry_run:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- entities


entities_app = typer.Typer(help="Maintain the entity layer.", no_args_is_help=True)
app.add_typer(entities_app, name="entities")


@entities_app.command("prune")
def entities_prune(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would go; write nothing.")
    ] = False,
    export: Annotated[bool, typer.Option(help="Export the snapshot afterwards.")] = True,
) -> None:
    """Delete the entity nodes no passage mentions any more, with their relations.

    A re-ingest replaces a source's documents and deletes the mentions on their passages. An
    entity those mentions were the only evidence for is left holding its id and nothing else,
    and the next extraction pass lands on the stale node instead of creating its own. Ingest
    prunes as it goes; this runs the same sweep over a graph that was loaded from a snapshot or
    edited by hand. A node another persona's passage still relates to is left alone.
    """
    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        removed = ctx.store.delete_orphan_entities(spec.id, dry_run=dry_run)
        verb = "would remove" if dry_run else "removed"
        console.print(f"[green]{verb}[/green] {removed} orphaned entities")
        if export and not dry_run and removed:
            snap.export_snapshot(
                ctx.store,
                spec,
                ctx.snapshots_dir,
                embedding_model=ctx.settings.embedding.model,
                embedding_dim=ctx.settings.embedding.dim,
            )
            console.print("snapshot re-exported")
    finally:
        ctx.close()


# ----------------------------------------------------------------------------- layers


layers_app = typer.Typer(
    help="Check that captured documents carry every layer: entities, speakers, annotations.",
    no_args_is_help=True,
)
app.add_typer(layers_app, name="layers")


@layers_app.command("check")
def layers_check(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    doc_id: Annotated[
        list[str] | None, typer.Option("--doc-id", help="Check this document id (repeatable).")
    ] = None,
    file: Annotated[
        list[Path] | None,
        typer.Option("--file", help="Check the document this raw file becomes (repeatable)."),
    ] = None,
    all_documents: Annotated[
        bool, typer.Option("--all", help="Check every document of the persona.")
    ] = False,
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Only this source id (with --all).")
    ] = None,
) -> None:
    """Report which layers a captured document has on disk and in the graph, and what came loose.

    Writing a document is the first of four passes: the text is ingested, an extraction file
    names its entities, an attribution file names the voices in it, an annotation file records
    what its passages say. Each can be skipped silently, so this checks all four at once. Every
    sidecar found is re-run through its own importer as a dry run, which writes nothing and
    reports exactly what a real import would place and what it would leave loose.

    Exits 1 when a listed document has no extraction JSON, when a sidecar fails to import, or
    when anything came loose; 0 when every layer is complete. A node attribute the persona's
    vocabulary does not declare counts as loose, under `attribute invalid`.
    """
    from graphrag.extract.layers import (
        LAYERS,
        SourceNotFoundError,
        check_documents,
        doc_id_for_file,
        summary_lines,
    )

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        raw_root = ctx.settings.raw_dir / spec.id
        wanted = list(doc_id or [])
        for path in file or []:
            try:
                wanted.append(doc_id_for_file(spec, raw_root, path)[0])
            except SourceNotFoundError as exc:
                err.print(f"[red]{exc}[/red]")
                raise typer.Exit(2) from exc
        if not wanted and not all_documents and source is None:
            err.print(
                "[red]name documents with --doc-id or --file, or pass --all "
                "(optionally with --source)[/red]"
            )
            raise typer.Exit(2)
        report = check_documents(
            ctx.store,
            spec,
            enrichment_root=ctx.settings.enrichment_dir,
            attribution_root=ctx.settings.attribution_dir,
            annotation_root=ctx.settings.annotations_dir,
            doc_ids=wanted or None,
            source_id=None if wanted else source,
            aliases=_alias_table(ctx, spec.id),
            facets=_facet_table(ctx, spec.id),
            attributes=_attribute_table(ctx, spec.id),
        )
        table = Table(title=f"layers: {spec.id}")
        # Folded rather than truncated: a document id a reader cannot finish reading is the one
        # column this table exists for, and it is longer than a default 80-column terminal.
        table.add_column("document", overflow="fold")
        table.add_column("source", overflow="fold")
        for col in ("graph", *LAYERS, "loose"):
            table.add_column(col)
        for document in report.documents:
            table.add_row(
                document.doc_id,
                document.source_id,
                "yes" if document.in_graph else "[red]no[/red]",
                *(_layer_cell(document.state(layer)) for layer in LAYERS),
                str(len(document.loose)) if document.loose else "",
            )
        console.print(table)
        for line in summary_lines(report):
            console.print(line, markup=False, highlight=False)
        if not report.ok:
            raise typer.Exit(1)
    finally:
        ctx.close()


def _layer_cell(state: str) -> str:
    """Colour one layer cell: on disk and in the graph, one of the two, or neither."""
    return {
        "ok": "[green]ok[/green]",
        "file": "[yellow]file[/yellow]",
        "graph": "[yellow]graph[/yellow]",
    }.get(state, "[red]-[/red]")


if __name__ == "__main__":
    app()
