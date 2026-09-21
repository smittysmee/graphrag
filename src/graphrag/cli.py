"""``graphrag`` command line: setup, ingest, snapshots, search, personas, enrichment, servers."""

from __future__ import annotations

import json
import os
import shlex
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
from graphrag.sna.arguments import (
    DEFAULT_MAX_SECONDS,
    parse_max_seconds,
    parse_sample_params,
    parse_types,
    parse_where,
)

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

#: Possible worlds drawn behind ``sna analyze --uncertain`` (Atlas §28.2). Repeated from
#: :data:`graphrag.sna.uncertain.SAMPLES` for the reason ``WHERE_PERMUTATIONS`` is, and held to
#: the same number by a unit test.
UNCERTAIN_SAMPLES = 200

#: The largest connected component ``sna walks`` will pseudo-invert (Atlas §11.4). Repeated
#: from :data:`graphrag.sna.walks.MAX_NODES` for the same reason as ``WHERE_PERMUTATIONS``, and
#: held to the same number by a unit test.
WALK_MAX_NODES = 2_000

#: The largest network ``sna roles`` will build a dense similarity matrix over (Atlas §15.2).
#: Repeated from :data:`graphrag.sna.roles.MAX_SIMILARITY_NODES` for the same reason as
#: ``WALK_MAX_NODES``, and held to the same number by a unit test.
ROLES_MAX_NODES = 2_000

#: The largest connected component ``sna distance`` will pseudo-invert (Atlas §47.2). Repeated
#: from :data:`graphrag.sna.walks.MAX_NODES` for the same reason as ``WALK_MAX_NODES``, and held
#: to the same number by a unit test.
DISTANCE_MAX_NODES = 2_000

#: The largest (support-of-p, support-of-q) pair ``sna distance`` will build a shortest-path
#: linkage or an earth-mover LP over (Atlas §47.3). Repeated from
#: :data:`graphrag.sna.vectordist.MAX_SUPPORT` for the same reason as ``WALK_MAX_NODES``, and
#: held to the same number by a unit test.
DISTANCE_MAX_SUPPORT = 60

#: Degree-preserving rewirings the four hierarchy scores of ``sna hierarchy`` are tested
#: against (Atlas §33.8). Repeated from :data:`graphrag.sna.hierarchy.NULL_SAMPLES` for the same
#: reason as ``WALK_MAX_NODES``, and held to the same number by a unit test.
HIERARCHY_SAMPLES = 50

#: How far ``sna highorder`` closes the passage hypergraph downward (Atlas §34.1). Repeated from
#: :data:`graphrag.sna.highorder.DEFAULT_MAX_DIM` for the same reason as ``WALK_MAX_NODES``, and
#: held to the same number by a unit test.
HIGHORDER_MAX_DIM = 3

#: The teleport factor both of ``sna highorder``'s walks run with (§11.1, §34.3). Repeated from
#: :data:`graphrag.sna.highorder.DEFAULT_DAMPING` for the same reason, and pinned by the same test.
HIGHORDER_DAMPING = 0.85

#: The largest network ``sna spread`` will simulate on (Atlas ch. 20-21): both the step and the
#: spectral threshold are dense in n. Repeated from
#: :data:`graphrag.sna.spread.MAX_SPREAD_NODES` for the same reason as ``WALK_MAX_NODES``, and
#: held to the same number by a unit test.
SPREAD_MAX_NODES = 3_000

#: ``sna spread``'s defaults, repeated from :mod:`graphrag.sna.spread` for the same reason as
#: ``WALK_MAX_NODES`` -- a signature default is evaluated at import time and that module pulls in
#: numpy -- and held to the module's numbers by a unit test. beta and the threshold are the
#: values the book's own exercises use (§20.5, §21.6); the rest are this package's.
SPREAD_BETA = 0.2
SPREAD_MU = 0.1
SPREAD_THRESHOLD = 2.0
SPREAD_ATTEMPTS = 1
SPREAD_STEPS = 50
SPREAD_RUNS = 20

#: The defaults of ``sna motifs`` (Atlas ch. 41): null rewirings behind the motif profile, the
#: support a mined pattern needs, the largest pattern grown (in edges), and the largest network
#: ``--mine`` will enumerate every subgraph of. All four are repeated from
#: :mod:`graphrag.sna.motifs` for the reason ``WHERE_PERMUTATIONS`` is, and held to the same
#: numbers by a unit test.
MOTIF_SAMPLES = 50
MOTIF_MIN_SUPPORT = 3
MOTIF_MAX_SIZE = 4
MOTIFS_MAX_NODES = 500

#: ``sna overlap``'s defaults (Atlas ch. 38): the clique size k-clique percolation uses, and the
#: node-Jaccard merge threshold ego-splitting uses. Repeated from :mod:`graphrag.sna.overlap` for
#: the reason ``WALK_MAX_NODES`` is, and held to the same numbers by a unit test.
OVERLAP_DEFAULT_K = 4
OVERLAP_JACCARD_THRESHOLD = 0.1

#: The holdout defaults of ``sna predict-eval`` (Atlas §25.1-25.2): the book's own 90/10 split,
#: its balanced test set of one sampled non-edge per real one, and the length of a ranked list a
#: person reads. Repeated from :mod:`graphrag.sna.experiment` for the same reason as
#: ``WHERE_PERMUTATIONS``, and held to the same numbers by a unit test.
PREDICT_SHARE = 0.1
PREDICT_NEGATIVES = 1.0
PREDICT_K = 10

#: ``sna predict``'s own defaults (Atlas ch. 23): how many hypotheses to rank, the association
#: rule's minimum support (§41.4 counts graphs, so at least 2), and the HRG restarts
#: :func:`graphrag.sna.cluster.hrg_fit` itself defaults to. Repeated from
#: :mod:`graphrag.sna.predict` and :mod:`graphrag.sna.cluster` for the same reason as
#: ``WHERE_PERMUTATIONS``, and held to the same numbers by a unit test.
PREDICT_TOP = 10
PREDICT_MIN_SUPPORT = 2
PREDICT_HRG_RESTARTS = 5

#: ``sna predict --signed``'s own default (Atlas §24.1): the share of signed edges whose sign is
#: hidden for the held-out-sign accuracy. Repeated from :data:`graphrag.sna.predict.
#: DEFAULT_SIGN_SHARE` for the reason ``WHERE_PERMUTATIONS`` is, and held to the same number by a
#: unit test.
PREDICT_SIGN_SHARE = 0.2

#: Default synthetic data sets behind the power-law goodness-of-fit p-value of ``sna degree``.
#: Repeated from :data:`graphrag.sna.degree.DEFAULT_BOOTSTRAP` for the same reason, and held to
#: it by the same unit test.
DEGREE_BOOTSTRAP = 200

#: ``sna draw``'s canvas, in pixels (Atlas ch. 49-51). Repeated from
#: :mod:`graphrag.sna.draw` for the same reason as ``WALK_MAX_NODES``, and held to the module's
#: numbers by a unit test.
DRAW_WIDTH = 960.0
DRAW_HEIGHT = 720.0

#: ``--min-weight`` is the Atlas's naive backbone (section 27.1), and chapter 27 is largely an
#: argument against it, so the help text carries the argument rather than only the mechanics.
MIN_WEIGHT_HELP = (
    "Drop edges below this weight. This is the Atlas's naive backbone (27.1) and the book "
    "argues against it: co-occurrence weights are fat-tailed -- most edges sit at 1 -- so a "
    "single threshold has no defensible value and cannot be motivated as 'x standard "
    "deviations from the average' (p. 383). Use --backbone instead."
)
#: The defensible filters of chapter 27, named in the order the book presents them.
BACKBONE_HELP = (
    "Apply one of chapter 27's backbones after --min-weight: naive, naive-top, "
    "doubly-stochastic, high-salience, convex, disparity, noise-corrected, mst, pmfg. "
    "noise-corrected is the one for count weights; disparity manufactures hub-and-spoke "
    "structure (p. 391) and naive-top fixes the minimum degree (p. 384)."
)
ALPHA_HELP = "Significance level for --backbone disparity or noise-corrected."
#: The structural methods have no null and so no alpha; they cut on a score, in their own units.
THRESHOLD_HELP = (
    "Level for a structural --backbone method, in that method's own units, replacing its "
    "default: a weight for naive, a count of edges per node for naive-top, a score in [0, 1] "
    "for doubly-stochastic and high-salience. Ignored by disparity and noise-corrected, which "
    "cut on --alpha, and by convex, mst and pmfg, which take no level."
)
CORRECTION_HELP = (
    "Multiple-test correction across the backbone's edge p-values, since every edge is a test "
    "of its own (3.3): none, bonferroni, holm, fdr_bh."
)
#: Chapter 26: the weight on a projected edge is a choice, so the help text names what each
#: choice does rather than only listing them, and says what it costs --min-weight.
PROJECTION_HELP = (
    "How a projected edge is weighted (chapter 26): simple (shared documents or passages, "
    "counted -- the default), jaccard, cosine, pearson, euclidean, hyperbolic (a shared node "
    "discounted by its degree), resource/probs, heats, hybrid (--lambda), randomwalk. Only "
    "simple produces counts, so --min-weight then applies to a fraction and defaults to 1 "
    "instead of the network's usual threshold. Run `graphrag sna projections` to compare them."
)
LAMBDA_HELP = (
    "The hybrid projection's interpolation between HeatS (0) and ProbS (1), 26.4. Read only "
    "with --projection hybrid."
)
#: Every command that builds a network takes this one option (ATL-ENT-3), rather than each
#: writing its own cache-bypass flag: routed through `graphrag.sna.cache.build_network_cached`.
NO_CACHE_OPTION = typer.Option(
    "--no-cache", help="Skip the on-disk network cache: rebuild from the store and re-cache it."
)
#: Every report-producing `sna` command takes this one option (ATL-ENT-4): append a `##
#: References` block naming the book once and, per Atlas chapter the report touched, a
#: `Chapter N -- Title (pp. x-y)` line from `graphrag.sna.provenance.CHAPTER_TITLES`.
CITE_OPTION = typer.Option(
    "--cite", help="Also print the book's reference for each Atlas chapter this report touches."
)


def _with_provenance(section: str, provenance: Any, *, cite: bool) -> str:
    """``section``'s markdown with `## Provenance` appended, and `## References` too if ``cite``.

    The one call every `sna` command's render step makes (ATL-ENT-4), so the block a report
    prints and the block its JSON payload carries (:func:`_payload_with_provenance`) are always
    built from the same :class:`~graphrag.sna.provenance.Provenance`.
    """
    from graphrag.sna.provenance import render_provenance, render_references

    lines = [section.rstrip(), "", *render_provenance(provenance)]
    if cite:
        lines += render_references(provenance.chapters)
    return "\n".join(lines).rstrip() + "\n"


def _payload_with_provenance(payload: dict[str, Any], provenance: Any) -> dict[str, Any]:
    """``payload`` with its `"provenance"` key set from the same `Provenance` the markdown used."""
    from graphrag.sna.provenance import provenance_payload

    payload["provenance"] = provenance_payload(provenance)
    return payload


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


def _invalidate_network_cache(settings: Settings, persona_id: str) -> None:
    """Clear ``persona_id``'s entries in the on-disk network cache (ATL-ENT-3).

    Called as the last step of every command that writes to a persona's graph -- ingest, sync,
    enrich, the sidecar importers, a snapshot load, an alias merge, an orphan prune -- so a build
    made before the write is never served after it. The cache's own fingerprint check
    (`graphrag.sna.cache._identity`) is a second line of defence for a write this function was
    not called after (Cypher run by hand, a foreign client); this call is the cheap, exact one,
    and costs nothing when the persona has nothing cached.
    """
    from graphrag.sna.cache import NetworkCache

    NetworkCache(settings.sna_cache_dir).clear(persona_id)


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
                _invalidate_network_cache(ctx.settings, persona.id)
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
        _invalidate_network_cache(ctx.settings, spec.id)
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
    refresh_extraction: Annotated[
        bool,
        typer.Option(
            "--refresh-extraction",
            help="Re-import every extraction file, not only those filling a gap.",
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
    force_lock: Annotated[
        bool,
        typer.Option(
            "--force-lock",
            help="Take the persona's sync lock even if another run appears to hold it live.",
        ),
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
    `--refresh-attribution`, `--refresh-extraction` and `--refresh-annotations` re-import every
    sidecar of that kind instead, whatever the graph already holds, and each source's line says
    which it did. A tagging pass that adds node attributes to files the graph already has
    speakers, entities and stances for is exactly that case, so those flags are how attributes
    reach the graph. Over the whole persona (no `--source`), each refresh first clears the stored
    attribute values its kind of sidecar owns, because the first value written wins and a
    corrected file could not otherwise replace one.

    Before touching the graph, this takes an advisory lock for the persona under
    `Settings.sync_lock_dir`, so a second `sync` against the same persona refuses instead of
    racing the first one for hours and risking a half-written document if one is killed. A lock
    whose heartbeat is older than `Settings.sync_lock_stale_seconds` is treated as abandoned and
    replaced; `--force-lock` takes a live one anyway. `--dry-run` never takes the lock, and only
    reports whether one is currently held.
    """
    from graphrag.sync import (
        SyncLockError,
        UnknownSourceError,
        acquire_sync_lock,
        read_sync_lock,
        refresh_sync_lock,
        release_sync_lock,
        summary_lines,
        sync_persona,
    )

    ctx = State.context()
    try:
        spec = ctx.registry.get(persona)
        lock_dir = ctx.settings.sync_lock_dir
        stale_after = ctx.settings.sync_lock_stale_seconds

        if dry_run:
            # dry-run never takes the lock: it only reports what it saw.
            held = read_sync_lock(lock_dir, spec.id)
            if held is not None:
                state = "stale" if held.is_stale(stale_after) else "live"
                err.print(
                    f"[yellow]{spec.id}: lock held ({state}) -- {held.held_message()}[/yellow]"
                )

        command_parts = ["graphrag", "sync", persona]
        if source:
            command_parts += ["--source", source]
        if refresh_attribution:
            command_parts.append("--refresh-attribution")
        if refresh_annotations:
            command_parts.append("--refresh-annotations")
        if allow_invalid:
            command_parts.append("--allow-invalid")
        if force_lock:
            command_parts.append("--force-lock")

        lock = None
        if not dry_run:
            try:
                lock = acquire_sync_lock(
                    lock_dir,
                    spec.id,
                    command=shlex.join(command_parts),
                    stale_after_seconds=stale_after,
                    force=force_lock,
                    progress=_progress,
                )
            except SyncLockError as exc:
                err.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc

        def _progress_and_heartbeat(msg: str) -> None:
            _progress(msg)
            if lock is not None:
                refresh_sync_lock(lock_dir, lock)

        try:
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
                    refresh_extraction=refresh_extraction,
                    allow_invalid=allow_invalid,
                    dry_run=dry_run,
                    progress=_progress_and_heartbeat,
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
            if report.wrote:
                _invalidate_network_cache(ctx.settings, spec.id)
            if export and report.wrote:
                snap.export_snapshot(
                    ctx.store,
                    spec,
                    ctx.snapshots_dir,
                    embedding_model=ctx.settings.embedding.model,
                    embedding_dim=ctx.settings.embedding.dim,
                )
                console.print(
                    f"snapshot exported to {snap.snapshot_dir(ctx.snapshots_dir, spec.id)}"
                )
            if report.errors or report.invalid_sources:
                raise typer.Exit(2)
        finally:
            if lock is not None:
                release_sync_lock(lock_dir, spec.id)
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
        _invalidate_network_cache(ctx.settings, spec.id)
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
        vocabulary = _attribute_table(ctx, spec.id)
        total_entities = total_mentions = total_relations = total_attributes = 0
        problems = 0
        for file in files:
            result = import_extraction_file(
                ctx.store, file, dry_run=dry_run, aliases=table, attributes=vocabulary
            )
            if not result.ok:
                err.print(f"[red]{file}: {result.error}[/red]")
                problems += 1
                continue
            total_entities += result.entities
            total_mentions += result.mentions
            total_relations += result.relations
            total_attributes += result.attributes
            console.print(
                f"{result.doc_id}: {result.entities} entities, "
                f"{result.mentions} mentions, {result.relations} relations"
                + (f", {result.attributes} entity attributes" if result.attributes else "")
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
                + (
                    f"; [yellow]{len(result.attribute_problems)} attribute problems[/yellow]"
                    if result.attribute_problems
                    else ""
                )
                + (
                    f"; [yellow]{len(result.attribute_conflicts)} attribute conflicts[/yellow]"
                    if result.attribute_conflicts
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
            for problem in result.attribute_problems:
                err.print(f"  attribute invalid: {problem}")
            for conflict in result.attribute_conflicts:
                err.print(f"  {conflict}")
        verb = "validated" if dry_run else "imported"
        console.print(
            f"[green]{verb}[/green] {len(files) - problems} files: {total_entities} entities, "
            f"{total_mentions} mentions, {total_relations} relations"
            + (f", {total_attributes} entity attributes" if total_attributes else "")
        )
        if not dry_run:
            _invalidate_network_cache(ctx.settings, spec.id)
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
            _invalidate_network_cache(ctx.settings, pid)
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
            _invalidate_network_cache(ctx.settings, spec.id)
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
        if not dry_run:
            _invalidate_network_cache(ctx.settings, spec.id)
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
        "speakers-entities, relations), filtered by stance, facet, relation type or date "
        "window, or read as layers of one multilayer network."
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


def _backbone_or_exit(value: str | None, layering: Any = None) -> str | None:
    """``--backbone`` must name one of chapter 27's methods; nothing is applied without it.

    Refused together with ``--layers``: a backbone filters one network, and a layering builds
    several and sums them, so there is no single edge set for the test to run over. Filter the
    layering's flattening by exporting it and running ``sna backbone`` on the result.
    """
    from graphrag.sna.backbone import BACKBONES

    if value is None:
        return None
    if value not in BACKBONES:
        err.print(f"[red]--backbone must be one of {', '.join(BACKBONES)}[/red]")
        raise typer.Exit(2)
    if layering is not None:
        err.print(
            "[red]--backbone filters one network; --layers builds one per value and sums them, "
            "so the two cannot be combined. Export the flattening first, then backbone it[/red]"
        )
        raise typer.Exit(2)
    return value


def _projection_or_exit(value: str, lam: float) -> str:
    """``--projection`` must name one of chapter 26's schemes, and ``--lambda`` must be a λ.

    Both are checked here rather than in the builder so that a typo costs exit 2 rather than a
    traceback, and so that the λ is refused even on the schemes that ignore it: a command that
    silently drops a number the user typed is worse than one that says it was not read.
    """
    from graphrag.sna.projection import SCHEMES

    if value not in SCHEMES:
        err.print(f"[red]--projection must be one of {', '.join(SCHEMES)}[/red]")
        raise typer.Exit(2)
    if not 0.0 <= lam <= 1.0:
        err.print("[red]--lambda must be between 0 (HeatS) and 1 (ProbS)[/red]")
        raise typer.Exit(2)
    return value


def _correction_or_exit(value: str) -> str:
    """``--correction`` adjusts the family of edge p-values a backbone produces (chapter 3.3)."""
    from graphrag.sna.stats import CORRECTIONS

    if value != "none" and value not in CORRECTIONS:
        err.print(f"[red]--correction must be none or one of {', '.join(CORRECTIONS)}[/red]")
        raise typer.Exit(2)
    return value


def _clean(values: list[str] | None) -> list[str] | None:
    """A repeated string option with the blanks dropped, or ``None`` when nothing was given."""
    cleaned = [v.strip() for v in (values or []) if v.strip()]
    return cleaned or None


def _where_or_exit(values: list[str] | None, option: str = "--where") -> dict[str, str] | None:
    """``--where key=value``, repeatable, into the canonical mapping (`sna.arguments.parse_where`,
    ATL-F1). Exits 2 rather than raising, so a malformed filter costs a refusal, not a traceback.
    """
    try:
        return parse_where(values, option)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _seeds_or_exit(value: str | None) -> list[str] | int | None:
    """``--seeds`` is either the patient zeros by name or how many to draw at random.

    A comma-separated list names them, and the same list runs in every simulation, which is what
    makes two runs comparable. A bare number draws that many nodes uniformly per run, which is
    what §20.5 asks for ("pick a random node and place it in the Infected state"); ``None`` means
    one. A node id that happens to be a number cannot be told from a count, so a single numeric
    token is read as a count -- name it with a second seed, or filter the network, if the id is
    what was meant.
    """
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        err.print("[red]--seeds was empty: name the patient zeros, or give a count[/red]")
        raise typer.Exit(2)
    if len(names) == 1 and names[0].isdigit():
        return int(names[0])
    return names


def _layering_or_exit(value: str | None) -> Any:
    """``--layers`` names the qualitative property that becomes the edge type (Atlas §7.2)."""
    from graphrag.sna.layers import LAYERINGS

    if value is None:
        return None
    if value not in LAYERINGS:
        err.print(f"[red]--layers must be one of {', '.join(LAYERINGS)}[/red]")
        raise typer.Exit(2)
    return value


def _network_or_bad_parameter(build: Callable[[], Any]) -> Any:
    """Build a network, turning the builder's rejection of a filter combination into exit 2."""
    try:
        return build()
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _degree_kind_or_exit(kind: str) -> str:
    """``--kind`` names the degree variant of Atlas §9.1: plain, directed or weighted."""
    from graphrag.sna.degree import DEGREE_KINDS

    if kind not in DEGREE_KINDS:
        err.print(f"[red]--kind must be one of {', '.join(DEGREE_KINDS)}[/red]")
        raise typer.Exit(2)
    return kind


def _sampler_or_exit(method: str) -> str:
    from graphrag.sna.sampling import SAMPLE_METHODS

    if method not in SAMPLE_METHODS:
        err.print(f"[red]--method must be one of {', '.join(SAMPLE_METHODS)}[/red]")
        raise typer.Exit(2)
    return method


def _sample_params_or_exit(values: list[str] | None) -> dict[str, Any]:
    """``--param k=3``, repeatable, into the sampler's own parameters (`sna.arguments.
    parse_sample_params`, ATL-F1). Exits 2 rather than raising, so a malformed ``--param`` costs
    a refusal, not a traceback.
    """
    try:
        return parse_sample_params(values)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _sampled_or_exit(graph: Any, method: str, size: int, seed: int | None, params: Any) -> Any:
    """Take the sample, turning a sampler's rejection of a size or a parameter into exit 2."""
    from graphrag.sna.sampling import sample

    try:
        return sample(graph, method, size, seed=seed, **params)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@sna_app.command("sample")
def sna_sample(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the sample.")],
    size: Annotated[int, typer.Option("--size", help="How many nodes the sample holds.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "speakers",
    method: Annotated[
        str,
        typer.Option(
            help=(
                "induced | edge | ties | bfs | snowball | forest-fire | random-walk | "
                "metropolis-hastings | neighbor-reservoir"
            )
        ),
    ] = "random-walk",
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help=(
                "A sampler parameter as name=value (k, p, restart, rounds, neighbors); repeatable."
            ),
        ),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Drop edges below this weight.")
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Fix the sampler's randomness.")] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the bias report as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Take a smaller network out of one of the graph's networks, and say what that cost.

    Atlas ch. 29. Each method distorts a different measurement in a direction the book states:
    `bfs` and `snowball` reach the hubs first and inflate clustering, `random-walk` oversamples
    hubs because its stationary distribution is proportional to degree, `metropolis-hastings`
    corrects that walk, `forest-fire` is the BFS variant that estimates clustering properly,
    `induced` picks nodes fairly and then shatters the connectivity, `edge` and `ties` draw
    edges instead of nodes and so land on hubs, and `neighbor-reservoir` swaps nodes in and out
    of a walked core while refusing to break it apart. The bias table printed here puts the
    observed direction next to the predicted one, and the completion estimate (§29.5) says how
    much of the network the sample did not see.

    `--param` carries a sampler's own settings: `k=5` (snowball), `p=0.3` (forest fire),
    `restart=0.2` (random walk), `rounds=20` (neighbor reservoir), `neighbors=true` (induced,
    for the whole neighbourhood of each selected node, which returns more nodes than --size).

    The sample is written through the same writer `sna export` uses, so the suffix picks the
    format; only `.graphml` and `.json` carry the sampling record back out of the file.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.export import write_graph
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.sampling import bias_report, completion, render_sample, sample_payload

    network = _network_or_exit(network)
    method = _sampler_or_exit(method)
    projection = _projection_or_exit(projection, lam)
    params = _sample_params_or_exit(param)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        population = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
            )
        )
    finally:
        ctx.close()
    sampled = _sampled_or_exit(population, method, size, seed, params)
    bias = bias_report(population, sampled)
    estimate = completion(sampled)
    prov = make_provenance(
        method="sample",
        parameters={
            "persona": persona,
            "network": network,
            "method": method,
            "param": params,
            "source": source,
            "min_weight": min_weight,
            "projection": projection,
            "lam": lam,
            "types": parse_types(types),
            "size": size,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["sample"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )

    _warn_if_ephemeral(out)
    write_graph(sampled, out)
    console.print(
        f"[green]wrote[/green] {out}: {sampled.number_of_nodes():,} of "
        f"{population.number_of_nodes():,} nodes, {sampled.number_of_edges():,} of "
        f"{population.number_of_edges():,} edges ({network}, {method})"
    )
    body = _with_provenance(render_sample(bias, estimate), prov, cite=cite)
    console.print(body)
    for check in bias.disagreements:
        err.print(
            f"  note: {check.measure} went {check.observed} where {bias.section} predicts "
            f"{check.predicted}."
        )
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(sample_payload(bias, estimate), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("export")
def sna_export(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Target .graphml or .json file.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "speakers",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option("--min-weight", help=MIN_WEIGHT_HELP),
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help=BACKBONE_HELP),
    ] = None,
    alpha: Annotated[
        float,
        typer.Option("--alpha", help=ALPHA_HELP),
    ] = 0.05,
    correction: Annotated[
        str,
        typer.Option("--correction", help=CORRECTION_HELP),
    ] = "none",
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help=THRESHOLD_HELP),
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
    relation_type: Annotated[
        list[str] | None,
        typer.Option(
            "--relation-type",
            help="Keep only relations of this type (--network relations); repeatable.",
        ),
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
    layers: Annotated[
        str | None,
        typer.Option(
            "--layers",
            help="Build one layer per stance | facet | source | relation-type and flatten them.",
        ),
    ] = None,
    omega: Annotated[float, typer.Option(help="Interlayer coupling recorded with --layers.")] = 1.0,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Write one of the graph's networks to a file, in the format its suffix names.

    `.graphml` and `.json` (node-link) carry everything; `.gexf` (Gephi), `.net` (Pajek), a
    tab-separated `.edgelist`/`.txt`, and a `.csv` node/edge pair (Cytoscape) each drop what
    the format cannot hold, and `graphrag.sna.FORMATS` says exactly what. Only node-link JSON
    returns integer node ids as integers; the rest read them back as text.

    `--where key=value` keeps only nodes attributed that way: a speaker by what an attribution
    pass recorded about them, an entity by what an extraction pass recorded about it and
    otherwise by the attributes of the document its passage belongs to. A node with no value
    for that key is left out rather than treated as a group of its own. Every node carries its
    attributes into the file as `attr_<key>`.

    `--layers <key>` writes the *flattening* of a multilayer network (Atlas §7.2): one layer per
    value, summed, with every edge recording which layers contributed and how much. Run
    `graphrag sna layers` for the per-layer table the flattening hides.

    `--backbone <method>` filters the network with one of Atlas chapter 27's methods after
    `--min-weight`, writing the `p` or `score` each surviving edge cleared onto the edge and the
    method, the level and the survival rate into the frame. Run `graphrag sna backbone` first to
    see what each of them would keep. It filters one network, so it cannot be combined with
    `--layers`, which builds several and sums them.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.export import write_graph
    from graphrag.sna.layers import flatten, multilayer
    from graphrag.sna.provenance import (
        COMMAND_CHAPTERS,
        make_provenance,
        provenance_payload,
        render_references,
        snapshot_commit_for,
    )

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    layering = _layering_or_exit(layers)
    projection = _projection_or_exit(projection, lam)
    entity_types = parse_types(types)
    method = _backbone_or_exit(backbone, layering)
    correction = _correction_or_exit(correction)
    ctx = State.context()
    try:
        ctx.registry.get(persona)  # fail fast on an unknown persona
        graph = _network_or_bad_parameter(
            lambda: (
                flatten(
                    multilayer(
                        ctx.store,
                        persona,
                        network,
                        layering,
                        source_id=source,
                        min_weight=min_weight,
                        projection=projection,
                        lam=lam,
                        types=entity_types,
                        stances=stances,
                        facets=_clean(facet),
                        relation_types=_clean(relation_type),
                        since=since,
                        until=until,
                        where=filters,
                        project=side,
                        omega=omega,
                    )
                )
                if layering is not None
                else build_network_cached(
                    ctx.store,
                    network,
                    persona,
                    settings=ctx.settings,
                    no_cache=no_cache,
                    source_id=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    types=entity_types,
                    stances=stances,
                    facets=_clean(facet),
                    relation_types=_clean(relation_type),
                    since=since,
                    until=until,
                    where=filters,
                    backbone=method,
                    alpha=alpha,
                    correction=correction,
                    threshold=threshold,
                    project=side,
                )
            )
        )
    finally:
        ctx.close()
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
            "where": filters,
            "project": project,
            "layers": layers,
            "omega": omega,
        },
        chapters=COMMAND_CHAPTERS["export"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    # `write_graph`'s formats each carry what they can of a network's own attributes (some drop
    # everything but nodes and edges); putting provenance on `graph.graph` rather than as a
    # payload-only key lets it ride along the same way, and lets the JSON/GraphML/GEXF writers
    # keep it without a format-specific case for this one key. It has to be a string, not the
    # dict `provenance_payload` returns: GraphML and GEXF attributes are scalar (the same reason
    # `_UNWRITTEN_GRAPH_KEYS` exists), so a caller reads it back with `json.loads`.
    graph.graph["provenance"] = json.dumps(provenance_payload(prov), sort_keys=True)
    _warn_if_ephemeral(out)
    write_graph(graph, out)
    console.print(
        f"[green]wrote[/green] {out}: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges ({network})"
    )
    if cite:
        console.print("\n".join(render_references(prov.chapters)))


@sna_app.command("analyze")
def sna_analyze(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "speakers",
    method: Annotated[
        str,
        typer.Option(
            help="louvain | kmeans | gmm | directed | louvain-levels | girvan-newman | hrg | "
            "sbm | infomap | walktrap | label-propagation | brim | bilouvain | "
            "neighbor-similarity (the last three group a two-mode network directly, ch. 39)"
        ),
    ] = "louvain",
    k: Annotated[int | None, typer.Option("-k", help="Force k; otherwise it is chosen.")] = None,
    k_range: Annotated[str, typer.Option("--k-range", help="Candidates, e.g. 2-10.")] = "2-10",
    resolution: Annotated[float, typer.Option(help="Louvain resolution.")] = 1.0,
    runs: Annotated[int, typer.Option(help="Louvain seeds to compare for stability.")] = 10,
    features: Annotated[
        str,
        typer.Option(
            help="spectral | embedding | node2vec | metapath2vec (ch. 42-43; metapath2vec needs "
            "--network speakers-entities without --project)"
        ),
    ] = "spectral",
    dims: Annotated[int, typer.Option(help="Embedding dimensions.")] = 8,
    p: Annotated[
        float, typer.Option("--p", help="node2vec return parameter: 1/p to backtrack.")
    ] = 1.0,
    q: Annotated[
        float,
        typer.Option("--q", help="node2vec in-out parameter: 1/q away from what was just seen."),
    ] = 1.0,
    covariance: Annotated[str, typer.Option(help="GMM covariance: full|tied|diag|spherical.")] = (
        "full"
    ),
    samples: Annotated[int, typer.Option(help="Null-model rewirings.")] = 50,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option("--min-weight", help=MIN_WEIGHT_HELP),
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help=BACKBONE_HELP),
    ] = None,
    alpha: Annotated[
        float,
        typer.Option("--alpha", help=ALPHA_HELP),
    ] = 0.05,
    correction: Annotated[
        str,
        typer.Option("--correction", help=CORRECTION_HELP),
    ] = "none",
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help=THRESHOLD_HELP),
    ] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option(
            "--relation-type",
            help="Keep only relations of this type (--network relations); repeatable.",
        ),
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
            help="Also report whether the network divides along this node attribute. A "
            "quantitative key gets Atlas ch. 31's correlation instead of ch. 30's homophily: "
            "degree | mentions | documents | chunks, a key declared `type: number`, or an "
            "undeclared key whose every value parses as a number.",
        ),
    ] = None,
    permutations: Annotated[
        int, typer.Option(help="Label shuffles for the --by assortativity null.")
    ] = WHERE_PERMUTATIONS,
    null: Annotated[
        str,
        typer.Option(
            "--null",
            help=(
                "Second null for --by: permutation (shuffle the labels) or bipartite (rewire "
                "the memberships this network was projected from)."
            ),
        ),
    ] = "permutation",
    project: Annotated[
        str | None,
        typer.Option(help="Collapse the two-mode network onto speakers | entities."),
    ] = None,
    sample_method: Annotated[
        str | None,
        typer.Option(
            "--sample",
            help="Analyse a sample of the network instead: induced | edge | ties | bfs | "
            "snowball | forest-fire | random-walk | metropolis-hastings | neighbor-reservoir "
            "(Atlas ch. 29).",
        ),
    ] = None,
    sample_size: Annotated[
        int | None, typer.Option("--sample-size", help="How many nodes the sample holds.")
    ] = None,
    layers: Annotated[
        str | None,
        typer.Option(
            "--layers",
            help="Build one layer per stance | facet | source | relation-type; analyse the "
            "flattening and report the per-layer table.",
        ),
    ] = None,
    omega: Annotated[float, typer.Option(help="Interlayer coupling reported with --layers.")] = 1.0,
    uncertain: Annotated[
        bool,
        typer.Option(
            "--uncertain",
            help="Also report every ranking as a mean and an interval over sampled possible "
            "worlds, each edge existing with its own probability (Atlas ch. 28).",
        ),
    ] = False,
    uncertain_samples: Annotated[
        int, typer.Option("--uncertain-samples", help="Possible worlds to sample for --uncertain.")
    ] = UNCERTAIN_SAMPLES,
    uncertain_expensive: Annotated[
        bool,
        typer.Option(
            "--uncertain-expensive",
            help="Also sample the shortest-path centralities under --uncertain (betweenness, "
            "closeness, harmonic, reach): one more shortest-path search per node per sampled "
            "world, which is what made --uncertain slow on a large network. Off by default; "
            "degree, eigenvector, pagerank, coreness and the rest always run.",
        ),
    ] = False,
    uncertain_max_seconds: Annotated[
        float,
        typer.Option(
            "--max-seconds",
            help="Stop drawing --uncertain realisations once this many seconds have been spent "
            "across every measure, and report the interval over whatever finished (Atlas §28.2). "
            "Stops the next realisation from starting, not one already running, so a single "
            "expensive call can still push the total past this on a large network. 0 for no "
            "budget.",
        ),
    ] = DEFAULT_MAX_SECONDS,
    degree_corrected: Annotated[
        bool,
        typer.Option(
            "--degree-corrected/--no-degree-corrected",
            help="For --method sbm only: fit the degree-corrected blockmodel (default) or the "
            "plain one (§35.1). No effect on any other method.",
        ),
    ] = True,
    seed: Annotated[int | None, typer.Option(help="Fix every random seed.")] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Measure a network, group it, check the grouping, and write a markdown report.

    `--by <key>` adds an attribute section: how many nodes carry each value, how far edges join
    like to like (assortativity, against a null that shuffles the labels), how the partition by
    that attribute scores against the best Louvain grouping and against a degree-preserving
    null, and how far the groups this run found agree with the attribute. It answers whether
    the network divides along something already known about its nodes, which is a question the
    grouping itself cannot be asked.

    What that section contains depends on what kind of value the key holds, because the book
    asks two different questions. A **label** gets chapter 30's homophily, as above. A **number**
    gets chapter 31's quantitative assortativity instead. Three things count as a number: one of
    the counts every network carries (`degree`, `mentions`, `documents`, `chunks`); a key the
    persona declared `type: number` in `facets.yaml`; and a key **nothing declares** whose every
    value parses as a finite number over at least two distinct numbers, which is a guess the
    section states in a note -- declare such a key with `values:` if they are codes rather than
    quantities. A declared key with `values:` stays categorical however numeric it looks. The
    quantitative section prints: the Pearson and Spearman correlation between the numbers
    at the two ends of every edge, the curve of a node's value against the mean of its
    neighbours' with its log-log exponent, the paradox that "your friends have more friends than
    you", and the attribute's distribution inside each degree band. The `--null` flag applies to
    the categorical section only; chapter 31's section picks its own null and says which.

    Chapter 31 over the *degree* is printed on every network whether or not `--by` was asked
    for, under `## Degree correlations`: whether the hubs in this network talk to each other or
    to the periphery changes how every ranking above it reads. `--by degree` therefore adds
    nothing and says so rather than printing the same numbers twice.

    That section also says where each label came from, and the answer decides whether the rest
    of it is a reading of the attribute. A value recorded about the node itself is independent
    of the edges; one borrowed from the documents the node's passages sit in is not, because the
    edges come from those same documents, and the label shuffle inflates such a result rather
    than catching it. When most labels were borrowed the verdict says so instead of reporting
    the coefficient: record the attribute on the entities and ask again.

    `--null bipartite` adds a second null to that section: instead of shuffling the labels over a
    fixed graph, it rewires the two-mode memberships this network was projected from -- keeping
    every node's number of documents and every document's size -- and measures the same labels on
    each re-projection. It is what catches an attribute that is really "was recorded a lot", and
    it needs a projected network, so it exits 2 on the topic network. Both nulls are reported.

    `--where key=value` narrows the network to nodes attributed that way before any of it runs.

    `--sample <method> --sample-size <n>` runs the whole analysis on a sample of the network
    instead of the network (Atlas ch. 29). The report's sampling frame then says which sampler
    took it, n of N, the bias that sampler is known to have, and how much of the network it did
    not see -- because every number under it is a number about the sampler as much as about the
    network. `graphrag sna sample` prints the full bias table.
    `--layers <key>` builds one layer per value of that key (Atlas §7.2) and runs everything
    above on their flattening, which is blind to which layer an edge came from. The report says
    so and prints the per-layer n and m above the numbers; `graphrag sna layers` prints the same
    table on its own, with the supra-adjacency shape.

    `--backbone <method>` filters the edges with one of Atlas chapter 27's methods, after
    `--min-weight` and before anything is measured, and the report's frame names it. Prefer
    `noise-corrected` over `disparity` whenever the grouping matters: the disparity filter keeps
    an edge whenever either endpoint likes it, which manufactures hub-and-spoke structure and
    weak communities (p. 391). It filters one network, so it is refused with `--layers`, which
    builds several; a backbone and a sample compose in that order, and the frame says both.

    Every report already says what its edges rest on (Atlas §28.1): how many hang on a single
    passage, how many of those on a loose match, and what share of the total weight they carry.
    That section is not a flag, because an edge is a claim with evidence and the size of the
    thinnest evidence decides how everything under it reads.

    `--uncertain` adds the rest of chapter 28. Each edge is given its probability of existing
    from that evidence, `--uncertain-samples` possible worlds are drawn from it (§28.2), and
    every centrality and the partition's modularity come back as a mean and a 95% interval beside
    the observed value, with the adjacent ranks whose intervals overlap named as ties. Expected
    degree, expected edges and expected density are exact rather than sampled (§28.3). It costs
    one evaluation of every measure per sampled world, so it is asked for rather than assumed.

    Two flags keep that cost bounded on a large network. `--uncertain-expensive` opts betweenness,
    closeness, harmonic and reach into the sampling; they are left out by default, since each
    needs its own shortest-path search per node per realisation, which is what makes `--uncertain`
    slow without them. `--max-seconds` caps the wall clock the whole `--uncertain` section may
    spend, shared across every measure rather than reset for each one; whichever measure is
    running when it runs out finishes its current realisation and stops there, and the report
    says, next to that measure's numbers, how many of `--uncertain-samples` it actually got. It
    defaults to five minutes; `--max-seconds 0` runs every realisation however long that takes.
    """
    from graphrag.sna.analysis import METHODS, render_markdown, run_analysis, to_payload
    from graphrag.sna.attributes import ATTRIBUTE_NULLS
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.layers import Multilayer, flatten, multilayer
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    layering = _layering_or_exit(layers)
    projection = _projection_or_exit(projection, lam)
    entity_types = parse_types(types)
    backbone = _backbone_or_exit(backbone, layering)
    correction = _correction_or_exit(correction)
    if method not in METHODS:
        err.print(f"[red]--method must be one of {', '.join(METHODS)}[/red]")
        raise typer.Exit(2)
    if sample_method is not None:
        sample_method = _sampler_or_exit(sample_method)
        if sample_size is None or sample_size < 1:
            err.print("[red]--sample needs --sample-size, a node count of at least 1[/red]")
            raise typer.Exit(2)
    if null not in ATTRIBUTE_NULLS:
        err.print(f"[red]--null must be one of {', '.join(ATTRIBUTE_NULLS)}[/red]")
        raise typer.Exit(2)

    ctx = State.context()
    stack: Multilayer | None = None
    try:
        ctx.registry.get(persona)
        if layering is not None:
            stack = _network_or_bad_parameter(
                lambda: multilayer(
                    ctx.store,
                    persona,
                    network,
                    layering,
                    source_id=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    types=entity_types,
                    stances=stances,
                    facets=_clean(facet),
                    relation_types=_clean(relation_type),
                    since=since,
                    until=until,
                    where=filters,
                    project=side,
                    omega=omega,
                )
            )
        graph = (
            flatten(stack)
            if stack is not None
            else _network_or_bad_parameter(
                lambda: build_network_cached(
                    ctx.store,
                    network,
                    persona,
                    settings=ctx.settings,
                    no_cache=no_cache,
                    source_id=source,
                    min_weight=min_weight,
                    projection=projection,
                    lam=lam,
                    types=entity_types,
                    stances=stances,
                    facets=_clean(facet),
                    relation_types=_clean(relation_type),
                    since=since,
                    until=until,
                    where=filters,
                    backbone=backbone,
                    alpha=alpha,
                    correction=correction,
                    threshold=threshold,
                    project=side,
                )
            )
        )
        if sample_method is not None and sample_size is not None:
            graph = _sampled_or_exit(graph, sample_method, sample_size, seed, {})
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
                p=p,
                q=q,
                covariance=covariance,
                samples=samples,
                by=by.strip() if by else None,
                permutations=permutations,
                null=null,
                seed=seed,
                multilayer=stack,
                # The persona's vocabulary is what says whether --by names a number or a label,
                # since a value reaches the graph as a string either way.
                vocabulary=_attribute_table(ctx, persona),
                uncertain=uncertain,
                uncertain_samples=uncertain_samples,
                uncertain_expensive=uncertain_expensive,
                uncertain_max_seconds=parse_max_seconds(uncertain_max_seconds),
                sbm_degree_corrected=degree_corrected,
            )
        except ValueError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
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
            "where": filters,
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
        snapshot_commit=snapshot_commit,
    )

    body = _with_provenance(render_markdown(analysis), prov, cite=cite)
    console.print(body, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
    summary = (
        f"{int(analysis.summary['nodes']):,} nodes, "
        f"{int(analysis.summary['edges']):,} edges, {len(analysis.groups)} "
        f"{analysis.group_noun}(s)"
    )
    console.print(f"[green]wrote[/green] {out}: {summary}" if out is not None else summary)
    if analysis.louvain_result is not None:
        console.print(
            f"  modularity {analysis.louvain_result.modularity:.4f}, "
            f"stability (ARI) {analysis.louvain_result.stability:.3f}"
        )
    if analysis.null_model is not None:
        console.print(
            f"  null model z={analysis.null_model.z_score:.2f} — {analysis.null_model.verdict}"
        )
    if analysis.dendrogram is not None:
        console.print(
            f"  {analysis.dendrogram.method} dendrogram: {len(analysis.dendrogram.levels)} "
            f"level(s), cut at level {analysis.dendrogram.cut}, "
            f"modularity {analysis.dendrogram.modularity:.4f}"
        )
    if analysis.hrg is not None:
        console.print(f"  HRG log-likelihood {analysis.hrg.log_likelihood:.4f}")
    if analysis.sbm is not None:
        console.print(
            f"  SBM ({analysis.sbm.implementation}): k={analysis.sbm.k}, "
            f"degree-corrected={analysis.sbm.degree_corrected}, "
            f"log-likelihood {analysis.sbm.log_likelihood:.4f}"
        )
    if analysis.infomap is not None:
        console.print(
            f"  Infomap code length {analysis.infomap.code_length:.4f} bits, "
            f"stability (ARI) {analysis.infomap.stability:.3f}"
        )
    if analysis.label_propagation_result is not None:
        console.print(
            f"  label propagation stability (ARI) {analysis.label_propagation_result.stability:.3f}"
        )
    if analysis.evidence is not None:
        console.print(
            f"  evidence: {analysis.evidence.single:,} of {analysis.evidence.edges:,} edges rest "
            f"on one piece of evidence, {analysis.evidence.single_weak:,} of those on a loose "
            "match or weaker"
        )
    for note in analysis.notes:
        err.print(f"  note: {note}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(to_payload(analysis), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("stances")
def sna_stances(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
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
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    cite: Annotated[bool, CITE_OPTION] = False,
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
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.stances import build_stance_report, render_stances, stances_payload

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
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
    prov = make_provenance(
        method="stances",
        parameters={
            "persona": persona,
            "entity": entity,
            "source": source,
            "facet": facet,
            "where": filters,
            "quotes": quotes,
        },
        chapters=COMMAND_CHAPTERS["stances"],
        snapshot_commit=snapshot_commit,
    )

    body = _with_provenance(render_stances(report), prov, cite=cite)
    console.print(body, markup=False, highlight=False)
    summary = (
        f"{report.annotations:,} annotated mentions over {len(report.entities):,} entities, "
        f"{len(report.pairs):,} co-mentioned pairs"
    )
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}: {summary}")
    else:
        console.print(summary)
    for name in report.missing:
        err.print(f"  no annotated mentions for: {name}", markup=False, highlight=False)
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(stances_payload(persona, report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("compare")
def sna_compare(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "speakers",
    since: Annotated[str | None, typer.Option(help="First window starts, ISO.")] = None,
    until: Annotated[str | None, typer.Option(help="First window ends, ISO.")] = None,
    since2: Annotated[str | None, typer.Option("--since2", help="Second window starts.")] = None,
    until2: Annotated[str | None, typer.Option("--until2", help="Second window ends.")] = None,
    window: Annotated[
        str | None,
        typer.Option(
            "--window",
            help="Cut --since..--until into windows of this width (90, 90D, 6M, 2Y) and compare "
            "the first with the last, instead of naming both pairs by hand.",
        ),
    ] = None,
    step: Annotated[
        str | None,
        typer.Option(
            "--step",
            help="How far each window starts after the last; default --window. Wider than "
            "--window throws the time between windows away, and the report says so.",
        ),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option("--min-weight", help=MIN_WEIGHT_HELP),
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help=BACKBONE_HELP),
    ] = None,
    alpha: Annotated[
        float,
        typer.Option("--alpha", help=ALPHA_HELP),
    ] = 0.05,
    correction: Annotated[
        str,
        typer.Option("--correction", help=CORRECTION_HELP),
    ] = "none",
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help=THRESHOLD_HELP),
    ] = None,
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
    topology: Annotated[
        bool,
        typer.Option(
            "--topology",
            help="Add ch. 48's topological-distance section: spectral distance, NetSimile, "
            "DeltaCon and portrait divergence between the two builds' whole graphs, node "
            "overlap or none at all.",
        ),
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    cite: Annotated[bool, CITE_OPTION] = False,
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

    `--projection` applies the same chapter 26 weighting to *both* builds, as `--backbone`
    applies the same filter to both: two schemes are two definitions of an edge weight, so a
    difference across them would be reporting the definition rather than the corpus. Both
    builds' frames name it.

    `--window 6M` is the general form of the two date pairs (Atlas §7.4): it cuts
    `--since`..`--until` into a grid and compares the first window with the last, naming every
    window in between in the notes rather than dropping it quietly. `--step` narrower than
    `--window` makes the grid slide instead of tiling. A range that makes only one window is
    refused: a build cannot be its own before.

    `--topology` adds ch. 48's whole-graph comparison: spectral distance, NetSimile, DeltaCon and
    portrait divergence between the two builds, computed over their whole topologies rather than
    the nodes they share. It needs no overlapping nodes at all, and is skipped -- with a note --
    when either build has fewer than two nodes.
    """
    from graphrag.sna.compare import compare_payload, compare_windows, render_comparison
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    filters2 = _where_or_exit(where2, "--where2")
    projection = _projection_or_exit(projection, lam)
    method = _backbone_or_exit(backbone)
    correction = _correction_or_exit(correction)
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
                window=window,
                step=step,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                where=filters,
                where2=filters2,
                projection=projection,
                lam=lam,
                backbone=method,
                alpha=alpha,
                correction=correction,
                threshold=threshold,
                project=side,
                centrality_kind=centrality,
                resolution=resolution,
                runs=runs,
                seed=seed,
                topology=topology,
            )
        )
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
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
            "where": filters,
            "where2": filters2,
            "project": project,
            "centrality": centrality,
            "resolution": resolution,
            "runs": runs,
            "topology": topology,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["compare"],
        snapshot_commit=snapshot_commit,
    )

    body = _with_provenance(render_comparison(comparison), prov, cite=cite)
    console.print(body, markup=False, highlight=False)
    summary = (
        f"{int(comparison.a.summary['nodes']):,} nodes in {comparison.a.label}, "
        f"{int(comparison.b.summary['nodes']):,} in {comparison.b.label}; "
        f"{len(comparison.shared):,} in both, {len(comparison.entered):,} entered, "
        f"{len(comparison.left):,} left"
    )
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}: {summary}")
    else:
        console.print(summary)
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(compare_payload(comparison), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")
    if comparison.similarity:
        console.print(
            f"  partition similarity ARI {comparison.similarity['adjusted_rand_index']:.3f}, "
            f"NMI {comparison.similarity['normalized_mutual_information']:.3f}"
        )
    if comparison.topology is not None:
        console.print(
            f"  topology: spectral {comparison.topology.spectral.distance:.3f}, NetSimile "
            f"{comparison.topology.netsimile.distance:.3f}, DeltaCon similarity "
            f"{comparison.topology.delta_con.similarity:.3f}, portrait divergence "
            f"{comparison.topology.portrait.divergence:.3f}"
        )
    for note in comparison.notes:
        err.print(f"  note: {note}")


@sna_app.command("layers")
def sna_layers(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    layers: Annotated[
        str, typer.Option("--layers", help="stance | facet | source | relation-type")
    ] = "stance",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Applied inside each layer.")
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    omega: Annotated[float, typer.Option(help="Interlayer coupling for the supra-adjacency.")] = (
        1.0
    ),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the table as markdown here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Read one network as a multilayer network: one layer per value, and what each holds.

    Prints the sampling frame, one row per layer (its nodes, its edges, the weight it carries
    and how many nodes it leaves isolated) and the shape of the supra-adjacency matrix at
    `--omega` — Atlas §7.2 for the layers, §8.1 for the matrix.

    Nothing here is a test, so nothing here has a null model: this is a description of what was
    built. `sna analyze --layers` runs the measures, on the flattening, and carries its own.

    The layers do not have to add up to the unlayered network. A facet layering splits passages,
    so it does; a stance layering splits mentions inside a passage, so a pair named once
    approvingly and once critically in one passage is in no layer at all.
    """
    from graphrag.sna.layers import (
        flatten,
        layer_table,
        layers_payload,
        multilayer,
        render_layers,
        supra_adjacency,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    layering = _layering_or_exit(layers)
    projection = _projection_or_exit(projection, lam)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        stack = _network_or_bad_parameter(
            lambda: multilayer(
                ctx.store,
                persona,
                network,
                layering,
                source_id=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
                project=side,
                omega=omega,
            )
        )
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
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
            "where": filters,
            "project": project,
            "omega": omega,
        },
        chapters=COMMAND_CHAPTERS["layers"],
        snapshot_commit=snapshot_commit,
    )

    console.print(f"[bold]{persona} / {network}[/bold], layered by {layering}")
    console.print(stack.frame, markup=False, highlight=False)
    table = Table(title=f"layers ({len(stack)})")
    table.add_column("layer")
    table.add_column("nodes", justify="right")
    table.add_column("edges", justify="right")
    table.add_column("total weight", justify="right")
    table.add_column("no edge here", justify="right")
    for row in layer_table(stack):
        table.add_row(
            row.name, f"{row.nodes:,}", f"{row.edges:,}", f"{row.weight:,.0f}", f"{row.isolated:,}"
        )
    console.print(table)

    matrix, order = supra_adjacency(stack)
    flattened = flatten(stack)
    console.print(
        f"supra-adjacency (Atlas §8.1): {matrix.shape[0]:,} by {matrix.shape[1]:,} over "
        f"{len(order):,} (node, layer) pairs, {len(stack.nodes):,} node(s) times "
        f"{len(stack):,} layer(s), omega={omega:g}"
    )
    console.print(
        f"flattened: {flattened.number_of_nodes():,} nodes, {flattened.number_of_edges():,} edges "
        "— every measure runs on this, so it cannot see which layer an edge came from"
    )
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = _with_provenance("\n".join(render_layers(stack)), prov, cite=cite)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(layers_payload(persona, network, layering, stack), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("multilayer-communities")
def sna_multilayer_communities(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    layers: Annotated[
        str, typer.Option("--layers", help="stance | facet | source | relation-type")
    ] = "stance",
    threshold: Annotated[
        int,
        typer.Option(
            "--threshold", help="Shared nodes needed to merge two layer-communities (§40.2)."
        ),
    ] = 3,
    omega: Annotated[
        float, typer.Option(help="Interlayer coupling for the supra-adjacency (§40.3).")
    ] = 1.0,
    gamma: Annotated[
        float, typer.Option(help="Per-layer resolution in the multilayer modularity null (§40.3).")
    ] = 1.0,
    resolution: Annotated[float, typer.Option(help="Louvain resolution.")] = 1.0,
    runs: Annotated[int, typer.Option(help="Louvain seeds to compare for stability.")] = 10,
    samples: Annotated[int, typer.Option(help="Null-model rewirings.")] = 20,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Applied inside each layer.")
    ] = None,
    projection: Annotated[
        str,
        typer.Option("--projection", help=PROJECTION_HELP),
    ] = "simple",
    lam: Annotated[
        float,
        typer.Option("--lambda", help=LAMBDA_HELP),
    ] = 0.5,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Fix every random seed.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Chapter 40 over one multilayer network: flattening, layer-by-layer, supra modularity.

    Builds the network once per `--layers` value (Atlas §7.2, the same as `sna layers`), then
    runs and scores three community-discovery methods on it side by side: flattening it and
    running Louvain (§40.1), running Louvain on each layer separately and matching the results
    into multilayer communities (§40.2), and maximising modularity over the supra-adjacency at
    `--omega` with an omega sweep printed beside it (§40.3). Every partition goes through the
    same evaluation battery (`## Community evaluation`, Atlas ch. 36). `--threshold` is Figure
    40.2's own merge rule -- how many nodes two layer-communities must share to merge -- and
    defaults to the book's own worked example, 3.

    Also prints multilayer density (redundancy and complementarity, §40.4) over the flattening's
    communities, and a `## Which to use` section (§40.5) recommending flattening when the
    layers' own communities already agree and layer-by-layer (or a low `--omega`) when they do
    not, scored by the adjusted Rand index between every pair of layers.
    """
    from graphrag.sna.layers import multilayer
    from graphrag.sna.multilayer import (
        analyze_multilayer_communities,
        multilayer_communities_payload,
        render_multilayer_communities,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    layering = _layering_or_exit(layers)
    projection = _projection_or_exit(projection, lam)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        stack = _network_or_bad_parameter(
            lambda: multilayer(
                ctx.store,
                persona,
                network,
                layering,
                source_id=source,
                min_weight=min_weight,
                projection=projection,
                lam=lam,
                types=parse_types(types),
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
                project=side,
                omega=omega,
            )
        )
        if len(stack) < 2:
            err.print(
                f"[red]multilayer-communities needs at least two layers; got {len(stack)}[/red]"
            )
            raise typer.Exit(2)
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
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
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
            "where": filters,
            "project": project,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["multilayer-communities"],
        null_model=report.supra_null.null or "per-layer rewiring null (§40.3)",
        null_model_samples=report.supra_null.samples,
        snapshot_commit=snapshot_commit,
    )

    body = _with_provenance("\n".join(render_multilayer_communities(report)), prov, cite=cite)
    console.print(body, markup=False, highlight=False)
    summary = (
        f"flatten {len(report.flatten_result.communities)} "
        f"communities (Q={report.flatten_result.modularity:.4f}), layer-by-layer "
        f"{len(report.layer_by_layer.multilayer_communities)}, supra "
        f"{len(report.supra.communities)} (omega={report.supra.omega:g}, "
        f"Q={report.supra.modularity:.4f}) -- recommended: {report.agreement.recommended}"
    )
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}: {summary}")
    else:
        console.print(summary)
    for note in report.notes:
        err.print(f"  note: {note}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(multilayer_communities_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("projections")
def sna_projections(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities")
    ] = "entities",
    project: Annotated[
        str | None,
        typer.Option(help="Which mode to project the two-mode network onto: speakers | entities."),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option(
            "--min-weight",
            help=(
                "Prune the simple-weight network first, then compare the schemes over the "
                "memberships of the nodes that survived. It narrows the corpus the comparison "
                "runs on; it is not a threshold on any scheme's own weights, which is what "
                "--keep-top and --threshold are for."
            ),
        ),
    ] = 1,
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
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    scheme: Annotated[
        list[str] | None,
        typer.Option(
            "--scheme",
            help="Limit the comparison to these schemes; repeatable. Default is all of them.",
        ),
    ] = None,
    lam: Annotated[float, typer.Option("--lambda", help=LAMBDA_HELP)] = 0.5,
    keep_top: Annotated[
        float,
        typer.Option(
            "--keep-top",
            help=(
                "Share of the edges each scheme keeps at the threshold step, so every scheme "
                "keeps the same count and the comparison is of which edges."
            ),
        ),
    ] = 0.1,
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help=(
                "Use an absolute weight cut for every scheme instead of --keep-top. Only "
                "readable for schemes on the same scale, which most of these are not."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report as markdown here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Project one network every way chapter 26 offers, and report where the schemes disagree.

    Atlas §26.6. The memberships are read once — this speaker in that document, this entity in
    that passage — and projected under every scheme, so the node set and the edge set are the
    same in every row and only the weights move. Three things come out:

    the **edge-weight distribution** per scheme (§26.6's figure 26.10), with the entropy of the
    weights as a summary of how much of the network's total weight sits on a few edges;

    the **rank agreement** between every pair of schemes (figure 26.11), by Spearman, because the
    scales are unrelated and only the order can be compared — and §26.6's finding is that schemes
    you would expect to agree need not: on the book's Twitter data HeatS and ProbS, one the
    transpose of the other, came out anti-correlated;

    and **what survives a threshold**, which is the step that decides whether there is a network
    to read at all: every scheme joins two nodes that share even one neighbour, so every
    projection is dense whatever the weighting. That cut is a plain weight threshold, not the
    backboning of chapter 27 — run `graphrag sna backbone` for those.

    Nothing here is a test and nothing here has a null model: a weighting scheme is a definition,
    so there is nothing for it to be unlikely against.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.null import pairs_of, projected_side
    from graphrag.sna.projection import (
        SCHEMES,
        compare_schemes,
        projections_payload,
        render_projections,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    names = _clean(scheme) or list(SCHEMES)
    for name in names:
        _projection_or_exit(name, lam)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
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

    pairs = pairs_of(graph)
    if pairs is None:
        err.print(
            "[red]sna projections needs the two-mode memberships a projection was built from, "
            "and this network carries none. The speakers and entities networks have them, as "
            "does speakers-entities with --project; the topic network reads stored co-occurrence "
            "edges and the relations network is not a projection at all[/red]"
        )
        raise typer.Exit(2)
    if not graph.number_of_nodes():
        err.print(
            f"[red]the {network} network is empty before any scheme runs, so there is nothing "
            f"to project (--min-weight {min_weight}). Lower it[/red]"
        )
        raise typer.Exit(2)
    mode = projected_side(graph, pairs)
    if mode is None:
        err.print("[red]could not tell which mode this network is the projection of[/red]")
        raise typer.Exit(2)
    # --min-weight pruned the simple-weight network, not the membership table it was projected
    # from, so the memberships of the nodes it stranded have to go too. Without this the
    # comparison would silently run over the whole corpus whatever was asked for, and the
    # opposite-mode degrees every discounting scheme divides by would be the unpruned ones.
    kept_nodes = set(graph.nodes)
    pairs = [
        (left, right) for left, right in pairs if (left if mode == "left" else right) in kept_nodes
    ]
    frame = str(graph.graph.get("frame", ""))
    if min_weight is not None and min_weight > 1:
        frame += (
            f" Restricted to the nodes surviving --min-weight {min_weight} on the simple-weight "
            "projection, so every scheme below is compared over that sub-corpus and the "
            "opposite-mode degrees are its own."
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

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
            "where": filters,
            "scheme": scheme,
            "lam": lam,
            "keep_top": keep_top,
            "threshold": threshold,
        },
        chapters=COMMAND_CHAPTERS["projections"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    report = _with_provenance(render_projections(comparison), prov, cite=cite)
    console.print(report, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(projections_payload(comparison), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("walks")
def sna_walks(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    source_node: Annotated[str, typer.Option("--from", help="The node the walker starts from.")],
    target_node: Annotated[str, typer.Option("--to", help="The node the walker is heading for.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Drop edges below this weight.")
    ] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    unweighted: Annotated[
        bool,
        typer.Option(
            "--unweighted", help="Ignore edge weights, so every edge is one 1-ohm resistor."
        ),
    ] = False,
    max_nodes: Annotated[
        int,
        typer.Option(
            "--max-nodes",
            help="Refuse a component larger than this: §11.4's formula is a dense O(n^3) inverse.",
        ),
    ] = WALK_MAX_NODES,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report as markdown here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """What a random walker between two nodes costs: hitting, commute, resistance, cut (ch. 11).

    Six numbers for one pair, each one the answer to a different question, and none of them a
    test — nothing here has a null model, because these are exact properties of the network as it
    was built rather than claims about a population.

    `hitting time` is the expected number of steps to arrive and it is **asymmetric**, so both
    directions are printed: §11.3's own example is one step from the end of a chain to its middle
    and three steps back. `commute time` is their sum. `effective resistance` is that commute
    time with the size of the network divided out (`C = 2|E|Ω`, §11.4), which is the number to
    compare across two networks; it is a proper metric, unlike a shortest path, and it moves less
    when one edge appears or vanishes. `minimum cut` is the cheapest set of edges that separates
    the two, equal to the maximum flow between them, and §11.5's reading is that a network
    expensive to cut is cheap to walk across — the resistance is never below `1 / cut`.

    Edge weights are read as **conductances**: a heavier edge is one the walker crosses more often
    and one with less resistance, which is the affinity reading the rest of the package gives
    them. `--unweighted` gives the book's plain 1-ohm resistors.

    Undefined, and said so rather than guessed, when the two nodes sit in different components:
    the walker never arrives, so the times are infinite and the cut is already made.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.measures import undirected_view
    from graphrag.sna.provenance import (
        COMMAND_CHAPTERS,
        make_provenance,
        render_provenance,
        render_references,
        snapshot_commit_for,
    )
    from graphrag.sna.walks import render_walks, volume_label, walks_payload, walks_report

    network = _network_or_exit(network)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
            )
        )
    finally:
        ctx.close()
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
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )

    graph, flattened = undirected_view(graph)
    for node in (source_node, target_node):
        if node not in graph:
            err.print(f"[red]not in the {network} network: {node}[/red]")
            raise typer.Exit(2)
    if source_node == target_node:
        err.print("[red]--from and --to must name two different nodes[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]{persona} / {network}[/bold]: {source_node} -> {target_node}")
    console.print(str(graph.graph.get("frame", "")), markup=False, highlight=False)

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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    console.print(
        f"Atlas ch. 11 (random walks), §11.3 times, §11.4 resistance, §11.5 cut. "
        f"n={report.n:,} nodes, m={report.m:,} edges, "
        f"{volume_label(report)}" + (f". {flattened}" if flattened else "")
    )
    console.print(
        "Null model: none. These are exact quantities of the network as built, not tests against "
        "a null, so there is no p-value to withhold them behind."
    )

    table = Table(title=f"{source_node} and {target_node}")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_column("what it means")
    table.add_row(
        "stationary probability",
        f"{report.source_probability:.5f} / {report.target_probability:.5f}",
        "§11.1: the share of an infinite walk spent on each — which undirected is degree / 2|E|",
    )
    table.add_row(
        "hitting time out",
        f"{report.hitting_out:,.2f}",
        f"§11.3: expected steps from {source_node} to first reach {target_node}",
    )
    table.add_row(
        "hitting time back",
        f"{report.hitting_back:,.2f}",
        f"§11.3: and from {target_node} back to {source_node}, which is not the same number",
    )
    table.add_row(
        "commute time", f"{report.commute_time:,.2f}", "§11.3: there and back; carries 2|E|"
    )
    table.add_row(
        "effective resistance",
        f"{report.effective_resistance:,.4f}",
        "§11.4: the commute time without 2|E|; comparable across networks",
    )
    table.add_row(
        "minimum cut (= max flow)",
        f"{report.cut.value:,.2f}",
        f"§11.5: {len(report.cut.source_side):,} nodes on one side, "
        f"{len(report.cut.target_side):,} on the other",
    )
    console.print(table)
    if report.connected:
        console.print(
            f"§11.4-11.5 check: resistance {report.effective_resistance:.4f} >= 1 / cut "
            f"{1 / report.cut.value:.4f} — a network that is expensive to cut is cheap to walk "
            "across."
        )
    else:
        console.print(
            "These two are in different components: the walker never arrives, so the times are "
            "infinite and §11.1 reads the two sides as two networks rather than one."
        )
    console.print("\n".join(render_provenance(prov)), markup=False, highlight=False)
    if cite:
        console.print("\n".join(render_references(prov.chapters)), markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = _with_provenance("\n".join(render_walks(report)), prov, cite=cite)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(walks_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("distance")
def sna_distance(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    vectors: Annotated[
        tuple[str, str],
        typer.Option(
            "--vectors",
            help=(
                "Two vector specs to compare, e.g. --vectors stance:praise stance:complaint. "
                "See graphrag.sna.vectordist.SPEC_HELP for every form."
            ),
        ),
    ],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Drop edges below this weight.")
    ] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    max_nodes: Annotated[
        int,
        typer.Option(
            "--max-nodes",
            help="Refuse a component larger than this: §47.2's L-dagger is a dense O(n^3) invert.",
        ),
    ] = DISTANCE_MAX_NODES,
    max_support: Annotated[
        int,
        typer.Option(
            "--max-support",
            help=(
                "Refuse a (support-of-p, support-of-q) pair larger than this for §47.3's "
                "linkages and earth-mover LP."
            ),
        ),
    ] = DISTANCE_MAX_SUPPORT,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report as markdown here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Compare two node vectors on one network: how far apart, five different ways (ch. 47).

    `--vectors a b` names two vectors with a small spec language -- `stance:praise` (annotated
    mention counts, `--network entities` only), `attr:<key>` (a numeric node attribute or a
    built-in count: mentions, documents, chunks, degree), `centrality:<kind>`, or
    `window:<since>:<until>:<kind>` (a centrality on one time window's network). A node the
    source said nothing about reads as 0, the occupancy reading ch. 47 opens with.

    Prints five sections, each its own answer to "how far": the non-network baselines (§47.1,
    Euclidean, cosine, correlation); the generalized Euclidean through the graph Laplacian's
    pseudoinverse (§47.2), which tells a swap between neighbours from a swap between strangers
    where plain Euclidean cannot; the shortest-path family (§47.3, single/complete/average
    linkage and the exact earth-mover optimum); the graph Fourier transform smoothness (§47.4);
    and network variance and network correlation (§47.5). None of these is "the" distance --
    each answers a different question, and the report prints all five rather than picking one.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.vectordist import (
        compare_vectors,
        distance_payload,
        render_distance,
        resolve_vector,
    )

    network = _network_or_exit(network)
    filters = _where_or_exit(where)
    parsed_types = parse_types(types)
    parsed_facets = _clean(facet)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parsed_types,
                facets=parsed_facets,
                where=filters,
            )
        )
        try:
            p = resolve_vector(
                vectors[0],
                graph,
                store=ctx.store,
                persona_id=persona,
                network=network,
                source_id=source,
                min_weight=min_weight,
                types=parsed_types,
                facets=parsed_facets,
                where=filters,
            )
            q = resolve_vector(
                vectors[1],
                graph,
                store=ctx.store,
                persona_id=persona,
                network=network,
                source_id=source,
                min_weight=min_weight,
                types=parsed_types,
                facets=parsed_facets,
                where=filters,
            )
        except ValueError as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
    finally:
        ctx.close()

    if graph.number_of_nodes() == 0:
        err.print("[red]this network has no nodes, so there is nothing to compare[/red]")
        raise typer.Exit(2)
    try:
        report = compare_vectors(
            graph, p, q, persona_id=persona, max_nodes=max_nodes, max_support=max_support
        )
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    prov = make_provenance(
        method="distance",
        parameters={
            "persona": persona,
            "network": network,
            "vectors": list(vectors),
            "source": source,
            "min_weight": min_weight,
            "types": parse_types(types),
            "facet": facet,
            "where": filters,
            "max_nodes": max_nodes,
            "max_support": max_support,
        },
        chapters=COMMAND_CHAPTERS["distance"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    body = _with_provenance("\n".join(render_distance(report)), prov, cite=cite)
    console.print(body, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(distance_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("backbone")
def sna_backbone(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "speakers",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    alpha: Annotated[float, typer.Option("--alpha", help=ALPHA_HELP)] = 0.05,
    correction: Annotated[str, typer.Option("--correction", help=CORRECTION_HELP)] = "none",
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help=THRESHOLD_HELP),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option("--min-weight", help="Naive threshold applied before every method (27.1)."),
    ] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Recorded in the report; no method here is randomised.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Run every backboning method of Atlas chapter 27 over one network and report what each keeps.

    One row per method: edges kept, the share of edges and of total weight that survived, how
    many nodes still have an edge, how many components are left, and the distribution of the
    surviving weights, each next to the null model or criterion that produced it. This is the
    chapter's own comparison -- the methods "exist because they give very different results"
    (p. 393) -- and it is meant to be run *before* choosing a `--backbone` for `sna analyze`.

    The `naive` row is run at the smallest threshold that removes anything at all, because that
    is the row p. 383 is about: when most edges sit at weight 1, the mildest hard threshold
    available already deletes most of the network in one step.

    Two methods are expensive on a large network: high-salience runs one Dijkstra per node
    (27.3) and convex enumerates the maximal cliques (27.4). Above
    `graphrag.sna.backbone.SALIENCE_SAMPLE_ABOVE_NODES` nodes, high-salience draws its sources
    from a degree-stratified sample instead of every node and says so, with the sample size,
    next to its row; doubly-stochastic switches its Sinkhorn normalisation to a sparse matrix
    above the same threshold `sna cluster`'s eigensolver already uses. Convex still enumerates
    every maximal clique in full. Narrow the network with `--min-weight` or a filter first if it
    is big.
    """
    from graphrag.sna.backbone import backbone_payload, compare_backbones, render_backbones
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    correction = _correction_or_exit(correction)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
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
    if graph.number_of_edges() == 0:
        err.print("[red]this network has no edges, so there is nothing to backbone[/red]")
        raise typer.Exit(2)
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
            "where": filters,
            "project": project,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["backbone"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    report = _with_provenance(
        render_backbones(graph, rows, alpha=alpha, correction=correction), prov, cite=cite
    )
    console.print(report, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(
            backbone_payload(rows, alpha=alpha, correction=correction), prov
        )
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("summarize")
def sna_summarize(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    by: Annotated[
        str,
        typer.Option(
            "--by",
            help="community (Louvain) or attr:<key> (the nodes sharing one value of an "
            "attribute) -- the grouping every technique below aggregates.",
        ),
    ] = "community",
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    resolution: Annotated[
        float, typer.Option(help="--by community only: Louvain resolution.")
    ] = 1.0,
    runs: Annotated[
        int, typer.Option(help="--by community only: Louvain seeds to compare for stability.")
    ] = 10,
    simplify_method: Annotated[
        str,
        typer.Option(
            "--simplify-method", help="Centrality (ch. 14) that ranks nodes for §46.3, e.g. degree."
        ),
    ] = "degree",
    simplify_share: Annotated[
        float, typer.Option("--simplify-share", help="Share of nodes §46.3 keeps.")
    ] = 0.5,  # graphrag.sna.summarize.SIMPLIFY_DEFAULT_SHARE, duplicated: sna modules load lazily
    influence_top: Annotated[
        int,
        typer.Option("--influence-candidates", help="§46.4: how many highest-degree seeds to try."),
    ] = 15,  # graphrag.sna.summarize.INFLUENCE_CANDIDATES, duplicated: sna modules load lazily
    influence_beta: Annotated[
        float, typer.Option("--influence-beta", help="§46.4: per-contact transmission chance.")
    ] = 1.0,
    influence_attempts: Annotated[
        int,
        typer.Option(
            "--influence-attempts", help="§46.4: steps a newly-infected node keeps trying."
        ),
    ] = 1,  # graphrag.sna.spread.DEFAULT_ATTEMPTS, duplicated: sna modules load lazily
    influence_runs: Annotated[
        int, typer.Option("--influence-runs", help="§46.4: simulated cascades per candidate seed.")
    ] = 1,
    seed: Annotated[
        int | None, typer.Option(help="Fix Louvain's tie-breaking and the cascade simulation.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report as markdown here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write JSON here.")] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Reduce a network to a smaller one that stands in for it (Atlas ch. 46).

    Four techniques over the `--by` grouping, in the chapter's own order: aggregation (§46.1,
    edge weights and within-group density of the meta-network the grouping collapses to),
    compression (§46.2, the description length of the network given that grouping as a model,
    against the plain edge list's), simplification (§46.3, the original nodes ranked by
    `--simplify-method` and only the top `--simplify-share` kept), and an influence-based summary
    (§46.4, a simulated independent cascade from the highest-degree candidates, keeping the edges
    any of them used and ranking the candidates by how fast their own cascade saturated).

    This is not `sna backbone`: that chapter drops edges and keeps every node it can; this one
    merges nodes into super nodes on purpose. The report's frame says so in the chapter's own
    words (p. 381-382), and the grouping `--by community` produces is scored by the chapter 36
    battery the same way every other partition this package reports is.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.summarize import render_summary, summarize, summary_payload

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    entity_types = parse_types(types)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=entity_types,
                stances=stances,
                facets=_clean(facet),
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    if graph.number_of_edges() == 0:
        err.print("[red]this network has no edges, so there is nothing to summarize[/red]")
        raise typer.Exit(2)
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

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
            "where": filters,
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
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    section = _with_provenance("\n".join(render_summary(report)), prov, cite=cite)
    console.print(section, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(summary_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("degree")
def sna_degree(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="degree | in | out | weighted | weighted_in | weighted_out (Atlas §9.1).",
        ),
    ] = "degree",
    bootstrap: Annotated[
        int, typer.Option(help="Synthetic data sets behind the goodness-of-fit p-value.")
    ] = DEGREE_BOOTSTRAP,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Keep one kind of stated relation; repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(help="Collapse the two-mode network onto speakers | entities."),
    ] = None,
    sample_method: Annotated[
        str | None,
        typer.Option("--sample", help="Measure a sample of the network instead (Atlas ch. 29)."),
    ] = None,
    sample_size: Annotated[
        int | None, typer.Option("--sample-size", help="How many nodes the sample holds.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Fix the bootstrap and any sampling.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the section as markdown here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the section as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """The degree distribution, and whether it is a power law — Atlas ch. 9.

    Prints the sampling frame and n, then the distribution as §9.2 asks for it (the pmf, the
    log-binned histogram and the CCDF, which is the one to read), then the §9.4 test: the
    exponent by maximum likelihood with `xmin` chosen by Kolmogorov-Smirnov distance, a
    goodness-of-fit p-value from `--bootstrap` synthetic data sets drawn from the fit itself, and
    likelihood-ratio tests against the lognormal and the exponential over that same tail.

    The verdict withholds the word. "Scale-free" is printed only when the bootstrap does not
    reject the power law and the lognormal does not fit significantly better; otherwise the
    report names what the data support — exponential-like, heavy-tailed but indistinguishable
    from a lognormal, or too few nodes to test at all. A straight line on a log-log plot is not
    evidence, and neither is a high R-squared from a regression on the logged values.

    `--kind` picks the variant of §9.1: `in` and `out` need a directed network (`--network
    relations`), and the `weighted` three are node strengths, the total weight incident to a
    node. `--sample` measures a sample instead, and the report then prints the §29.3 re-weighted
    estimate of the population's distribution beside the sample's own.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.degree import degree_payload, degree_report, render_degree
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    kind = _degree_kind_or_exit(kind)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if bootstrap < 1:
        err.print("[red]--bootstrap must be at least 1 resample[/red]")
        raise typer.Exit(2)
    if sample_method is not None:
        sample_method = _sampler_or_exit(sample_method)
        if sample_size is None or sample_size < 1:
            err.print("[red]--sample needs --sample-size, a node count of at least 1[/red]")
            raise typer.Exit(2)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    if sample_method is not None and sample_size is not None:
        graph = _sampled_or_exit(graph, sample_method, sample_size, seed, {})
    try:
        report = degree_report(graph, kind, bootstrap=bootstrap, seed=seed)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

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
            "where": filters,
            "project": project,
            "sample_method": sample_method,
            "sample_size": sample_size,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["degree"],
        null_model="power-law bootstrap (synthetic data drawn from the fitted exponent, §9.4)",
        null_model_samples=bootstrap,
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    section = _with_provenance("\n".join(render_degree(report)), prov, cite=cite)
    console.print(section, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(degree_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("roles")
def sna_roles(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    method: Annotated[
        str, typer.Option(help="jaccard | cosine | pearson | simrank | regular")
    ] = "jaccard",
    k: Annotated[
        int | None,
        typer.Option("-k", help="Also cluster the similarity into k blockmodel positions."),
    ] = None,
    decay: Annotated[
        float, typer.Option("--decay", help="SimRank's gamma (p. 338). It changes the answer.")
    ] = 0.8,
    minimum: Annotated[
        float,
        typer.Option("--minimum", help="Similarity floor for the same-role pairs. No null model."),
    ] = 0.5,
    resolution: Annotated[float, typer.Option(help="Louvain resolution for the partition.")] = 1.0,
    runs: Annotated[int, typer.Option(help="Louvain seeds to compare for stability.")] = 10,
    seed: Annotated[int | None, typer.Option(help="Louvain seed; the rest is deterministic.")] = (
        None
    ),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Relation type (--network relations); repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    max_nodes: Annotated[
        int,
        typer.Option(
            "--max-nodes",
            help="Refuse a network larger than this: §15.2's similarity matrix is dense n x n.",
        ),
    ] = ROLES_MAX_NODES,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """What each node *is* in the network, rather than how much of it there is (Atlas ch. 15).

    Two reports in one. §15.1 places every node by how well connected it is inside its own
    community (`z`) and how evenly its ties are spread across the others (`P`), then cuts that
    plane into seven roles — ultra-peripheral, peripheral, non-hub connector, non-hub kinless,
    provincial hub, connector hub, kinless hub — which are mapped back onto the four words the
    chapter itself uses: broker, gatekeeper, core, periphery. **A role is relative to the
    partition it was computed on**, so the Louvain run behind it, its resolution and its
    stability are printed above the counts; so is the ceiling `1 - 1/m`, because with two
    communities no node can reach the connector threshold however evenly its ties run.

    §15.2 is node similarity: `--method jaccard|cosine|pearson` compares the rows of the
    adjacency matrix, which is the book's own definition of structural equivalence (*"a vector of
    zeros and ones"*, p. 230); `simrank` and `regular` are its two recursive relaxations, whose
    parameters (`--decay`, and the alpha the recursion needs to converge) are printed because
    they change the answer. `-k` clusters the similarity into blockmodel positions and prints the
    image matrix — the density of edges between two positions — with the warning that a position
    is a cluster and inherits a clustering's instability.

    The section the chapter earns is the last one: **same role, never together**. Two nodes can
    be structurally equivalent without sharing an edge — the book's own illustration of a
    perfectly equivalent pair has none — so this lists the entities the corpus discusses in the
    same company and never in the same passage. Whether that is a gap in the corpus or a fact
    about the subject is not something the network can say.

    Nothing here is a test: no number in this report has a null model, because a role is a
    deterministic function of the network and the partition. §15.3 (node embeddings) is
    deliberately absent — the section defers to the graph-neural-network chapters and so does
    this package.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.roles import SIMILARITIES, build_roles_report, render_roles, roles_payload

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if method not in SIMILARITIES:
        err.print(f"[red]--method must be one of {', '.join(SIMILARITIES)}, got {method}[/red]")
        raise typer.Exit(2)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    if graph.number_of_edges() == 0:
        err.print("[red]this network has no edges, so no node has a structural role[/red]")
        raise typer.Exit(2)
    if k is not None and not 1 <= k <= graph.number_of_nodes():
        err.print(f"[red]-k must be between 1 and the {graph.number_of_nodes()} nodes[/red]")
        raise typer.Exit(2)
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
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
            "where": filters,
            "project": project,
            "max_nodes": max_nodes,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["roles"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance(render_roles(report), prov, cite=cite)
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(roles_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("ego")
def sna_ego(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    node: Annotated[
        str, typer.Argument(help="The node to centre on, by id as `sna export` writes it.")
    ],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    radius: Annotated[
        int, typer.Option("--radius", help="How many hops out the neighbourhood reaches (§30.1).")
    ] = 1,
    without_ego: Annotated[
        bool,
        typer.Option(
            "--without-ego",
            help="Lead with the view the ego has been removed from, which is §30.1's remedy.",
        ),
    ] = False,
    by: Annotated[
        str | None,
        typer.Option("--by", help="Also report what the alters are for this node attribute."),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Keep one kind of stated relation; repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(help="Collapse the two-mode network onto speakers | entities."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the section as markdown here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the section as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """One node's neighbourhood, with and without the node — Atlas §30.1.

    The ego network is the node, its alters and the edges among them. Most of what it shows is
    the construction rather than the node: the ego is joined to every alter by definition, so
    §30.1 warns that every ego network has one component and a diameter of two. This prints both
    views for that reason. The density **without** the ego is the number to read, and it is the
    ego's local clustering coefficient (§12.2) — the pairs of its neighbours that know each
    other over the pairs it could have closed — computed here from the whole network so the two
    can be checked against each other.

    What is left over is brokerage: the pairs of neighbours joined only through this node, which
    is `1 - clustering` in share terms and the position §30.3 calls a weak tie. On a corpus
    network a high one has two readings the number cannot separate — a thing that genuinely
    connects unrelated subjects, and a thing that was written about in unrelated contexts.

    `--by` adds §30.2 over the neighbourhood: what the alters carry for one attribute, against
    what the whole network carries, with the provenance of every label and the majority illusion
    of §30.4 when a value that is a minority in the corpus is a majority here. It is not a test;
    `sna analyze --by` is where the nulls are.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.ego import ego_payload, ego_report, render_ego
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if radius < 1:
        err.print("[red]--radius must be at least 1 hop[/red]")
        raise typer.Exit(2)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    try:
        report = ego_report(graph, node, radius=radius, with_ego=not without_ego, by=by)
    except KeyError:
        err.print(f"[red]not in the {network} network: {node}[/red]")
        raise typer.Exit(2) from None

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
            "where": filters,
            "project": project,
        },
        chapters=COMMAND_CHAPTERS["ego"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    section = _with_provenance("\n".join(render_ego(report)), prov, cite=cite)
    console.print(section, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(ego_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("community")
def sna_community(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    seed_node: Annotated[
        str | None,
        typer.Option(
            "--seed",
            help="Grow a local community outward from this node (§35.5) instead of tracking "
            "communities across snapshots.",
        ),
    ] = None,
    max_size: Annotated[
        int | None, typer.Option("--max-size", help="Stop local growth after this many nodes.")
    ] = None,
    temporal: Annotated[
        bool,
        typer.Option(
            "--temporal",
            help="Track Louvain communities across dated snapshots instead of growing a local "
            "one from --seed (§35.4). Needs --seed to be left out.",
        ),
    ] = False,
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None,
        typer.Option("--stance", help="Keep only mentions with this stance; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    window: Annotated[
        str, typer.Option("--window", help="--temporal only: snapshot width (90, 90D, 6M, 2Y).")
    ] = "6M",
    step: Annotated[
        str | None,
        typer.Option("--step", help="--temporal only: snapshot stride; default --window."),
    ] = None,
    since: Annotated[
        str | None, typer.Option(help="--temporal only: earliest date, ISO YYYY-MM-DD.")
    ] = None,
    until: Annotated[
        str | None, typer.Option(help="--temporal only: latest date, ISO YYYY-MM-DD.")
    ] = None,
    cumulative: Annotated[
        bool, typer.Option("--cumulative", help="--temporal only: every window starts at --since.")
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold", help="--temporal only: minimum Jaccard overlap to call a match (§35.4)."
        ),
    ] = 0.1,  # graphrag.sna.cluster.TEMPORAL_MATCH_THRESHOLD, duplicated: sna modules load lazily
    resolution: Annotated[
        float, typer.Option(help="--temporal only: Louvain resolution for each snapshot.")
    ] = 1.0,
    runs: Annotated[
        int, typer.Option(help="--temporal only: Louvain seeds per snapshot for stability.")
    ] = 10,
    random_seed: Annotated[
        int | None,
        typer.Option(
            "--random-seed",
            help="Fix the random tie-breaking and, for --temporal, Louvain's own seed.",
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the section as markdown here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the section as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Community discovery that does not look at the whole network at once (Atlas §35.4-§35.5).

    `--seed <node>` grows a local community outward from one node (§35.5): starting from the seed
    alone, it repeatedly admits whichever neighbour of the frontier most raises Clauset's local
    modularity `R` -- the share of edges touching the community that stay inside it -- and stops
    the first time every remaining neighbour would lower `R` instead, `--max-size` nodes in, or
    when the whole reachable component has been explored, whichever comes first. Nothing outside
    the frontier it explores is ever looked at, which is the point on a network too large to
    cluster whole.

    `--temporal` tracks communities across dated snapshots instead (§35.4): Louvain runs
    independently on each `--window`-wide slice of the corpus, and consecutive partitions are
    joined by the Jaccard overlap of their communities, each pair above `--threshold` labelled
    with one of the book's six events (birth, death, continue, grow, shrink, merge, split;
    Figures 35.9-35.11). Community numbers are not comparable across windows on their own, which
    is exactly what this command's transition table is for.

    Every partition produced -- the local community, or each snapshot's -- goes through the
    chapter 36 battery (`## Community evaluation`) the same way `sna analyze`'s does.
    """
    from graphrag.sna.cache import build_network_cached
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
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    if seed_node is not None and temporal:
        err.print("[red]--seed and --temporal are two different modes; pass only one[/red]")
        raise typer.Exit(2)
    if seed_node is None and not temporal:
        err.print(
            "[red]give --seed <node> for a local community, or --temporal to track across "
            "snapshots[/red]"
        )
        raise typer.Exit(2)

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    entity_types = parse_types(types)

    ctx = State.context()
    lines: list[str] = []
    payload: dict[str, Any] = {}
    try:
        ctx.registry.get(persona)
        if temporal:
            windows = _network_or_bad_parameter(
                lambda: snapshots(
                    ctx.store,
                    persona,
                    network,
                    window=window,
                    step=step,
                    since=since,
                    until=until,
                    cumulative=cumulative,
                    source_id=source,
                    min_weight=min_weight,
                    types=entity_types,
                    stances=stances,
                    facets=_clean(facet),
                    where=filters,
                    project=side,
                )
            )
            try:
                tracked = match_communities(
                    windows,
                    threshold=threshold,
                    resolution=resolution,
                    seed=random_seed,
                    runs=runs,
                )
            except ValueError as exc:
                err.print(f"[red]{exc}[/red]")
                raise typer.Exit(2) from exc
            lines = render_temporal_communities(tracked)
            payload = temporal_communities_payload(tracked)
            evaluations: dict[str, Any] = {}
            windows_by_label = dict(windows)
            for window_label, partition in zip(tracked.windows, tracked.partitions, strict=True):
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
            console.print(
                f"[green]tracked[/green] {len(tracked.windows)} windows, "
                f"{len(tracked.transitions)} transitions"
            )
        else:
            assert seed_node is not None
            graph = _network_or_bad_parameter(
                lambda: build_network_cached(
                    ctx.store,
                    network,
                    persona,
                    settings=ctx.settings,
                    no_cache=no_cache,
                    source_id=source,
                    min_weight=min_weight,
                    types=entity_types,
                    stances=stances,
                    facets=_clean(facet),
                    where=filters,
                    project=side,
                )
            )
            try:
                result = local_community(graph, seed_node, max_size=max_size, seed=random_seed)
            except ValueError as exc:
                err.print(f"[red]{exc}[/red]")
                raise typer.Exit(2) from exc
            lines = render_local_community(result)
            payload = local_community_payload(result)
            if len(result.members) > 1:
                scores = evaluate_partition(graph, [result.members], seed=random_seed)
                lines += ["#### Community evaluation", ""]
                lines += render_evaluation(scores)
                payload["evaluation"] = evaluation_payload(scores)
            console.print(
                f"[green]grown[/green] {len(result.members)} node(s) from {seed_node}, "
                f"stopped: {result.stopped}"
            )
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()
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
            "where": filters,
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
        snapshot_commit=snapshot_commit,
    )

    section = _with_provenance("\n".join(lines), prov, cite=cite)
    console.print(section, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(payload, prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("overlap")
def sna_overlap(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    method: Annotated[
        str, typer.Option(help="clique | link | ego — which cover-discovery method to run.")
    ] = "clique",
    k: Annotated[
        int, typer.Option("-k", help="Clique size for --method clique (§38.3).")
    ] = OVERLAP_DEFAULT_K,
    jaccard_threshold: Annotated[
        float,
        typer.Option(
            "--jaccard-threshold", help="Merge threshold for --method ego (§38.9 exercise 3)."
        ),
    ] = OVERLAP_JACCARD_THRESHOLD,
    resolution: Annotated[
        float, typer.Option(help="Louvain resolution used inside --method ego's ego networks.")
    ] = 1.0,
    seed: Annotated[
        int | None, typer.Option(help="Seed for --method ego and the baseline.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Relation type (--network relations); repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Nodes that belong to more than one community, and what that does to the definition (ch. 38).

    Three ways to find an overlapping *cover* rather than a disjoint partition. `--method clique`
    (§38.3) percolates k-cliques: two of them join the same community once they share k-1 nodes,
    and a node of degree under k-1 can never sit in any of them, which the report counts rather
    than silently dropping. `--method link` (§38.5) clusters the *edges* by the Jaccard
    similarity of their non-shared endpoints' neighbourhoods and cuts the resulting dendrogram at
    the height that maximises the chapter's own partition density; a node belongs to every
    community any of its edges landed in. `--method ego` (§38.5's "Ego Networks", DEMON) looks at
    one node's own ego network at a time, partitions it, unions the node back into each local
    community it finds, and merges candidates across the network whenever their node-Jaccard
    passes `--jaccard-threshold`.

    Every cover found is put through the same three checks. It is flattened to the best-effort
    disjoint partition chapter 36's battery can score (`## Community evaluation`), with the
    report saying exactly what the flattening threw away. It is checked against **the overlap
    paradox** (§38.7): communities are supposed to be dense inside and sparse outside, and
    overlap makes that impossible to keep on both counts at once — the shared nodes either barely
    connect to each other (a hole) or connect to each other more than either community does
    internally (the boundary is the core). And it is compared, by **overlapping NMI** (§38.1),
    against this same network's own Louvain partition read as a cover with no overlaps — not a
    ground truth, just this package's other opinion about the same edges.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.overlap import METHODS, build_overlap_report, overlap_payload, render_overlap
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if method not in METHODS:
        err.print(f"[red]--method must be one of {', '.join(METHODS)}, got {method}[/red]")
        raise typer.Exit(2)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    if graph.number_of_edges() == 0:
        err.print("[red]this network has no edges, so there is nothing to find a cover of[/red]")
        raise typer.Exit(2)
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
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
            "where": filters,
            "project": project,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["overlap"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance("\n".join(render_overlap(report)), prov, cite=cite)
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(overlap_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("hierarchy")
def sna_hierarchy(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="The network to read; only `relations` is directed.")
    ] = "relations",
    samples: Annotated[
        int, typer.Option(help="Degree-preserving rewirings the four scores are tested against.")
    ] = HIERARCHY_SAMPLES,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Keep one kind of stated relation; repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None,
        typer.Option("--facet", help="Keep only passages with this facet; repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Fix the rewirings behind the null model.")] = (
        None
    ),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the section as markdown here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the section as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Whether the network is an organisation chart, by four measures — Atlas ch. 33.

    Prints the frame and the n, then §33.1's classification of the shape with the checks that
    decided it (acyclic? one root? one boss per node? reciprocity?), then the four
    hierarchicalness scores the chapter says must be read together because each is wrong where
    the next is right: the share of arcs on a cycle (§33.2, too lenient — every DAG scores
    perfectly), global reach centrality (§33.3, too strict — only a star scores 1), the maximum
    spanning arborescence with its root and the arcs it discards (§33.4), and the agony of every
    arrow that points up (§33.5, exact by linear program). It closes with the layering a drawing
    would use (§33.6).

    Every score is also computed on `--samples` directed rewirings that hold each node's
    in-degree and out-degree fixed, which is the chapter's own exercise, so the report says
    whether the shape is more hierarchical than the degrees already force.

    Needs a directed network: chapter 33 says "from now on, we always assume that the network
    we're analyzing is directed", and on an undirected one every measure here is degenerate
    rather than inapplicable. `--network relations` is the only directed network this package
    builds; the *order* hierarchy of an undirected one is its centrality ranking (`sna analyze`)
    and the *nested* one is community discovery.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.hierarchy import analyse_hierarchy, hierarchy_payload, render_hierarchy
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    filters = _where_or_exit(where)
    if samples < 0:
        err.print("[red]--samples cannot be negative[/red]")
        raise typer.Exit(2)

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
            )
        )
    finally:
        ctx.close()

    try:
        report = analyse_hierarchy(graph, samples=samples, seed=seed)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

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
            "where": filters,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["hierarchy"],
        null_model="directed degree-preserving rewiring (in- and out-degree fixed)",
        null_model_samples=samples,
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    section = _with_provenance("\n".join(render_hierarchy(report)), prov, cite=cite)
    console.print(section, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(hierarchy_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("highorder")
def sna_highorder(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="entities — the passage hypergraph is the only high-order one.")
    ] = "entities",
    max_dim: Annotated[
        int,
        typer.Option("--max-dim", help="How far the downward closure goes (Atlas §34.1, §7.3)."),
    ] = HIGHORDER_MAX_DIM,
    damping: Annotated[
        float,
        typer.Option("--damping", help="Teleport factor of both walks (§11.1). It changes them."),
    ] = HIGHORDER_DAMPING,
    exact: Annotated[
        bool,
        typer.Option(
            "--exact/--damped",
            help="Run the undamped stationary distribution; fails if the chain has none.",
        ),
    ] = False,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """The passages as simplices, and a walk that remembers where it came from — Atlas ch. 34.

    Two reports over the same passages, answering two questions a single graph cannot.

    **§34.1, the simplicial complex.** A passage naming three entities is one relation between
    three things, and closing that downward — every sub-group of it is a face — is what separates
    a simplex from a hyperedge (*"a hyperedge with four nodes only contains those four nodes"*,
    §7.3 p. 112). Out of it come the generalized degree `k_(d,m)`, *"the number of d dimensional
    simplices incident on an m-face"* (p. 475), the incidence that shows this is not a manifold,
    and the number the chapter earns: the **closure**, how many triangles of the 1-skeleton the
    passages actually filled. An unfilled triangle is three entities this corpus paired up in
    three different passages and never named together — which is exactly what an analysis that
    promoted every triangle of the entity network to a simplex would have invented, and §34.1 is
    explicit that the two operations *"are not commutative"* (Figure 34.10, p. 481).

    **§34.2 and §34.3, the memory network.** A node is an ordered transition `A -> B` read off
    consecutive passages of one document, an edge is a path `A -> B -> C` the corpus actually
    shows, and the walker on it is the second-order walker of §34.3. Its stationary distribution,
    folded back onto entities, is printed beside the first-order one, with the entities that
    change rank. Both walks are damped by default (`--damping`, §11.1's teleport, the one Rosvall
    et al. compare) because a memory network is rarely strongly connected; `--exact` asks for the
    undamped π and fails where there is none. Read the difference and not the level, and do not
    attribute it to one cause: the two walks coincide only where the memory network's own flow is
    balanced — every transition continued as often as it is entered — which even a document that
    ends on the entity it began with does not give, so second-order structure inside a document,
    the document boundaries and the teleport all move the numbers.

    Document boundaries are hard stops, a passage that named nothing breaks the chain rather than
    being bridged, and the passage order is a transcript's running order — never a cause.
    """
    from graphrag.sna.highorder import (
        highorder_reports,
        memory_payload,
        render_memory,
        render_simplicial,
        simplicial_payload,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    if network != "entities":
        err.print(
            "[red]--network must be entities: chapter 34's structures are read off the passage "
            "hypergraph, and only the entity network is a projection of one[/red]"
        )
        raise typer.Exit(2)
    if max_dim < 0:
        err.print("[red]--max-dim must be at least 0 — a 0-simplex is a node[/red]")
        raise typer.Exit(2)
    if not exact and not 0.0 < damping <= 1.0:
        err.print("[red]--damping must be above 0 and at most 1 (§11.1's teleport)[/red]")
        raise typer.Exit(2)
    stances = _stances_or_exit(stance)
    filters = _where_or_exit(where)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        reports = _network_or_bad_parameter(
            lambda: highorder_reports(
                ctx.store,
                persona,
                source_id=source,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
                max_dim=max_dim,
                damping=None if exact else damping,
            )
        )
    finally:
        ctx.close()
    simplicial, memory = reports
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
            "where": filters,
        },
        chapters=COMMAND_CHAPTERS["highorder"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance(
        "\n".join([*render_simplicial(simplicial), *render_memory(memory)]), prov, cite=cite
    )
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(
            {
                "simplicial": simplicial_payload(simplicial),
                "memory": memory_payload(memory),
            },
            prov,
        )
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("predict-eval")
def sna_predict_eval(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    temporal: Annotated[
        str | None,
        typer.Option(
            "--temporal",
            help=(
                "Split at this ISO date: train on what was joined before it, test on what was "
                "joined after. The split the book prefers."
            ),
        ),
    ] = None,
    share: Annotated[
        float,
        typer.Option("--share", help="Share of edges to delete for the random holdout (§25.1)."),
    ] = PREDICT_SHARE,
    negatives: Annotated[
        float,
        typer.Option("--negatives", help="Sampled non-edges per positive; 1.0 is balanced."),
    ] = PREDICT_NEGATIVES,
    k: Annotated[
        int, typer.Option("--k", help="How many predictions precision@k scores.")
    ] = PREDICT_K,
    seed: Annotated[int | None, typer.Option(help="Fix the split and the random baseline.")] = None,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None,
        typer.Option(
            "--min-weight",
            help="Drop edges below this weight. The random split only: see the docstring.",
        ),
    ] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the evaluation as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Build a link-prediction experiment and run the baselines through it (Atlas ch. 25).

    This is the report every prediction in this package has to come back through, and it is
    deliberately available before any predictor is: chapter 25 exists because *"evaluating the
    performance of an oracle in getting things right is harder than it might seem"*.

    Two splits. `--temporal 2025-01-01` trains on the pairs the corpus had joined before that
    date and tests on the ones it joined after — §25.1's first choice, and the only one that
    leaves the training structure intact. Without it the split is a random deletion of `--share`
    of the edges, which is the k-fold branch of Figure 25.2 run once and which damages the
    network it then asks a predictor to reconstruct; the frame says so above the numbers.

    Negatives are sampled uniformly from the pairs the network never joined, `--negatives` of
    them per positive, because the pair space is too large to score whole (p. 358) and an
    unbalanced test set puts everything in the true-negative cell (p. 361). The report prints the
    imbalance of that unsampled space next to the balanced n, which is the number to read before
    the AUC.

    Two baselines run: preferential attachment (the degree product) and a scorer that ignores the
    network, whose AUC is the 45-degree line. `sna predict` (chapter 23) adds `--method` and
    reports through exactly this.

    One asymmetry to know about: `--min-weight` applies to the random split, which builds the
    network the usual way (so the entity network keeps its default of 2), and not to the temporal
    one, which reads §7.4's timestamped edge list where an edge is one dated activation. A
    temporal split is therefore taken over the unthresholded network, and its frame says how many
    activations it saw.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.experiment import (
        evaluate_predictor,
        experiment_payload,
        holdout,
        preferential_attachment,
        random_scores,
        render_experiment,
        temporal_holdout,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    filters = _where_or_exit(where)
    kinds = parse_types(types)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        if temporal is not None:
            split = _network_or_bad_parameter(
                lambda: temporal_holdout(
                    ctx.store,
                    persona,
                    network,
                    split_at=temporal,
                    seed=seed,
                    negatives=negatives,
                    source_id=source,
                    types=kinds,
                    stances=stances,
                    facets=_clean(facet),
                    where=filters,
                )
            )
        else:
            graph = _network_or_bad_parameter(
                lambda: build_network_cached(
                    ctx.store,
                    network,
                    persona,
                    settings=ctx.settings,
                    no_cache=no_cache,
                    source_id=source,
                    min_weight=min_weight,
                    types=kinds,
                    stances=stances,
                    facets=_clean(facet),
                    where=filters,
                )
            )
            split = _network_or_bad_parameter(
                lambda: holdout(graph, share=share, seed=seed, negatives=negatives)
            )
    finally:
        ctx.close()

    if not split.positives or not split.negatives:
        err.print(
            f"[red]this split has {len(split.positives):,} positive(s) and "
            f"{len(split.negatives):,} negative(s); §25.2's measures need both classes. Widen "
            "the network, lower --min-weight, or move the split date.[/red]"
        )
        raise typer.Exit(2)

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
            "where": filters,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["predict-eval"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    report = _with_provenance(render_experiment(split, reports), prov, cite=cite)
    console.print(f"[bold]{persona} / {network}[/bold]: {split.method} holdout")
    console.print(report, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(experiment_payload(split, reports), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("predict")
def sna_predict(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="pa | cn | aa | ra | jaccard | katz | hrg | rules (Atlas ch. 23).",
        ),
    ] = "cn",
    top: Annotated[
        int, typer.Option("--top", help="How many ranked hypotheses to print.")
    ] = PREDICT_TOP,
    share: Annotated[
        float,
        typer.Option("--share", help="Share of edges to delete for the evaluation (§25.1)."),
    ] = PREDICT_SHARE,
    negatives: Annotated[
        float,
        typer.Option("--negatives", help="Sampled non-edges per positive; 1.0 is balanced."),
    ] = PREDICT_NEGATIVES,
    k: Annotated[
        int, typer.Option("--k", help="How many predictions precision@k scores.")
    ] = PREDICT_K,
    seed: Annotated[int | None, typer.Option(help="Fix the split and every random draw.")] = None,
    beta: Annotated[
        float | None,
        typer.Option("--beta", help="Katz's beta; defaults to half of 1/lambda_max (§23.7)."),
    ] = None,
    hrg_samples: Annotated[
        int | None,
        typer.Option("--hrg-samples", help="MCMC steps per HRG restart; see hrg_fit's default."),
    ] = None,
    hrg_restarts: Annotated[
        int, typer.Option("--hrg-restarts", help="Independent HRG dendrogram searches.")
    ] = PREDICT_HRG_RESTARTS,
    min_support: Annotated[
        int,
        typer.Option("--min-support", help="Documents an association rule must hold in (§41.4)."),
    ] = PREDICT_MIN_SUPPORT,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    signed: Annotated[
        bool,
        typer.Option(
            "--signed",
            help="Also predict praise/complaint sign by balance theory (Atlas §24.1).",
        ),
    ] = False,
    sign_share: Annotated[
        float,
        typer.Option("--sign-share", help="Share of signed edges to hide for the §24.1 holdout."),
    ] = PREDICT_SIGN_SHARE,
    layers: Annotated[
        str | None,
        typer.Option(
            "--layers",
            help=(
                "stance | facet | source | relation-type: run generalized multilayer prediction "
                "(Atlas §24.2) for --target-layer instead of the single-network ranking above."
            ),
        ),
    ] = None,
    target_layer: Annotated[
        str | None,
        typer.Option(
            "--target-layer", help="Which layer's edges to predict; defaults to the first built."
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the ranking as JSON here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Rank link-prediction hypotheses on the entity network, never writing one to the graph
    (Atlas ch. 23).

    Two sections, always in this order (Wave 5's own rule: chapter 25 lands before chapter 23 so
    a score is never printed unevaluated). First the same evaluation `sna predict-eval` prints --
    `--method` against preferential attachment and chance on a random holdout of this network --
    so a reader can see whether the method beats both baselines before trusting anything it
    proposes. Then the ranked hypotheses themselves, scored on the network as it stands today
    (not the holdout, which deleted edges on purpose), each with the passages a reader would
    check before believing it: passages that already name both entities but did not reach an
    edge, or the passage naming a third entity with each of them when neighbourhood methods
    (`cn`, `aa`, `ra`, `jaccard`) are asked, or just where each is named alone when no such
    passage exists yet. Nothing here is written to the graph; a score is a claim about what the
    corpus would say if somebody looked.

    Fixed to `--network entities`: the evidence and `--method rules` both resolve through entity
    mentions, which only that network has.

    `--signed` adds Atlas §24.1's balance-theory reading: a held-out-sign accuracy against a
    measured coin flip, plus the sign each ranked hypothesis would carry from the entity
    network's praise/complaint co-mentions. `--layers` switches to §24.2's generalized multilayer
    prediction instead of the sections above: a layer of `--target-layer` is held out and scored
    by common neighbours blended with every other layer's own score, weighted by how correlated
    that layer is with the target (§24.4's own exercise), against the single-layer baseline.
    """
    from collections.abc import Mapping

    import networkx as nx

    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.experiment import (
        Pair,
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
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.stances import build_stance_report
    from graphrag.textutil import sanitize_inline

    if method not in PREDICT_METHODS:
        err.print(f"[red]--method must be one of {', '.join(PREDICT_METHODS)}[/red]")
        raise typer.Exit(2)
    if top < 1:
        err.print("[red]--top must be at least 1[/red]")
        raise typer.Exit(2)
    layering = _layering_or_exit(layers)
    stances = _stances_or_exit(stance)
    filters = _where_or_exit(where)
    kinds = parse_types(types)
    facets = _clean(facet)

    if layering is not None:
        ctx = State.context()
        try:
            ctx.registry.get(persona)
            ml = _network_or_bad_parameter(
                lambda: multilayer(
                    ctx.store,
                    persona,
                    "entities",
                    layering,
                    source_id=source,
                    min_weight=min_weight,
                    types=kinds,
                    stances=stances,
                    facets=facets,
                    where=filters,
                )
            )
            target = target_layer or ml.names[0]
            if target not in ml.names:
                err.print(
                    f"[red]--target-layer must be one of {', '.join(ml.names)}, got {target!r}"
                    "[/red]"
                )
                raise typer.Exit(2)
            target_graph = ml.graphs[target]
            split = _network_or_bad_parameter(
                lambda: holdout(target_graph, share=share, seed=seed, negatives=negatives)
            )
            if not split.positives or not split.negatives:
                err.print(
                    f"[red]this split has {len(split.positives):,} positive(s) and "
                    f"{len(split.negatives):,} negative(s); §25.2's measures need both classes. "
                    "Widen the layer or lower --min-weight.[/red]"
                )
                raise typer.Exit(2)
            reports = [
                evaluate_predictor(
                    split,
                    lambda train: cross_layer_common_neighbours(ml, target, train),
                    name=f"cross-layer common neighbours ({target})",
                    k=k,
                ),
                evaluate_predictor(
                    split, common_neighbors, name=f"common neighbours ({target}, single layer)", k=k
                ),
                evaluate_predictor(split, preferential_attachment, k=k),
                evaluate_predictor(split, lambda train: random_scores(train, seed=seed), k=k),
            ]
            report = render_experiment(
                split,
                reports,
                title=f"Multilayer link prediction ({target}, Atlas §24.2)",
            )
            snapshot_commit = snapshot_commit_for(ctx.settings, persona)
        finally:
            ctx.close()
        prov = make_provenance(
            method="predict",
            parameters={
                "persona": persona,
                "network": "entities",
                "layers": layering,
                "target_layer": target,
                "share": share,
                "negatives": negatives,
                "k": k,
                "source": source,
                "min_weight": min_weight,
                "types": parse_types(types),
                "stance": stance,
                "facet": facet,
                "where": filters,
            },
            seed=seed,
            chapters=COMMAND_CHAPTERS["predict"],
            snapshot_commit=snapshot_commit,
        )
        report = _with_provenance(report, prov, cite=cite)
        console.print(f"[bold]{persona} / entities, layers={layering}[/bold]: target={target}")
        console.print(report, markup=False, highlight=False)
        if out is not None:
            _warn_if_ephemeral(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report, encoding="utf-8")
            console.print(f"[green]wrote[/green] {out}")
        return

    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                "entities",
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=kinds,
                stances=stances,
                facets=facets,
                where=filters,
            )
        )
        documents = (
            document_graphs(
                ctx.store,
                persona,
                source_id=source,
                types=kinds,
                stances=stances,
                facets=facets,
                where=filters,
            )
            if method == "rules"
            else []
        )
        split = _network_or_bad_parameter(
            lambda: holdout(graph, share=share, seed=seed, negatives=negatives)
        )
        rows = entity_passage_rows(
            ctx.store, persona, source, kinds, stances=stances, facets=facets, where=filters
        )

        if not split.positives or not split.negatives:
            err.print(
                f"[red]this split has {len(split.positives):,} positive(s) and "
                f"{len(split.negatives):,} negative(s); §25.2's measures need both classes. "
                "Widen the network or lower --min-weight.[/red]"
            )
            raise typer.Exit(2)
        if method == "rules" and len(documents) < 2:
            err.print(
                "[red]--method rules needs at least two documents to fit a rule from (§41.4)[/red]"
            )
            raise typer.Exit(2)

        def scorer_for(train: nx.Graph) -> Mapping[Pair, float]:
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
                try:
                    return katz(train, beta)
                except ValueError as exc:
                    err.print(f"[red]{exc}[/red]")
                    raise typer.Exit(2) from exc
            if method == "hrg":
                return hrg_scores(train, samples=hrg_samples, restarts=hrg_restarts, seed=seed)
            removed = frozenset(split.positives) if train is split.train else frozenset()
            return association_rule_scores(
                train, documents_without(documents, removed), min_support=min_support
            )

        reports = [evaluate_predictor(split, scorer_for, name=METHOD_NAMES[method], k=k)]
        if method != "pa":
            reports.append(evaluate_predictor(split, preferential_attachment, k=k))
        reports.append(
            evaluate_predictor(split, lambda train: random_scores(train, seed=seed), k=k)
        )
        evaluation = render_experiment(
            split, reports, title=f"Link prediction evaluation ({METHOD_NAMES[method]}, ch. 25)"
        )

        live_scores = scorer_for(graph)
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
        excerpts = {c.id: sanitize_inline(c.text, QUOTE) for c in ctx.store.get_chunks(chunk_ids)}

        sign_section = ""
        if signed:
            signed_pairs = build_stance_report(
                ctx.store, persona, source_id=source, facets=facets or (), where=filters
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
                sign_split = _network_or_bad_parameter(
                    lambda: sign_holdout(signed_net, share=sign_share, seed=seed)
                )
                sign_eval = evaluate_sign_predictor(sign_split, seed=seed)
                sign_section = render_sign_evaluation(
                    sign_split, sign_eval
                ) + render_predicted_signs(hypotheses, signed_net)
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()

    frame = (
        f"{persona} / entities network, {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges. {METHOD_NAMES[method]} ({METHOD_SECTIONS[method]}) "
        "scored on the network as it stands today, not on the evaluation's holdout."
    )
    null_model = (
        "the same two baselines as the evaluation above -- a scorer that ignores the network "
        "and preferential attachment -- plus, for `rules`, the network's own global clustering "
        "coefficient (§12.2), since a rule that closes triangles no faster than the network "
        "already does is not a finding."
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
            "where": filters,
            "signed": signed,
            "sign_share": sign_share,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["predict"],
        null_model=null_model,
        snapshot_commit=snapshot_commit,
    )
    report = _with_provenance(report, prov, cite=cite)
    console.print(f"[bold]{persona} / entities[/bold]: {method}")
    console.print(report, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        # `hypotheses_payload` is a list, not a dict -- the one report whose payload is not
        # already keyed, so it is the one wrapped rather than given a "provenance" key directly.
        payload = _payload_with_provenance({"hypotheses": hypotheses_payload(hypotheses)}, prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("complete")
def sna_complete(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    key: Annotated[str, typer.Argument(help="Attribute key to complete (facets.yaml).")],
    method: Annotated[str, typer.Option(help="gcn | graphsage | gat")] = "gcn",
    hidden_dims: Annotated[
        str, typer.Option("--hidden-dims", help="Comma-separated hidden layer widths.")
    ] = "16",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 200,
    lr: Annotated[float, typer.Option("--lr", help="Adam learning rate.")] = 0.05,
    seed: Annotated[int, typer.Option(help="Seeds the weight init and the held-out split.")] = 0,
    gat_heads: Annotated[
        int,
        typer.Option(
            "--gat-heads",
            help=(
                "Attention heads per GAT layer (--method gat only, §45.1). A single head is "
                "GAT's minimal case and can underfit where gcn or graphsage would not at the "
                "same --epochs/--lr; try 2 or more before trusting a gat result over them."
            ),
        ),
    ] = 1,
    val_share: Annotated[
        float,
        typer.Option(
            "--val-share", help="Share of owned labels withheld to measure generalisation."
        ),
    ] = 0.2,
    top: Annotated[int, typer.Option("--top", help="How many ranked suggestions to print.")] = 25,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[
        int | None, typer.Option("--min-weight", help="Drop edges below this weight.")
    ] = None,
    types: Annotated[
        str | None, typer.Option(help="Entity types to keep, comma separated.")
    ] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the report here.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the suggestions as JSON here.")
    ] = None,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Suggest `key` for entities that carry no value of their own (Atlas ch. 44; §45.1 for
    `--method gat`).

    A two-layer GCN, GraphSAGE or GAT (`--method`), trained on the entities this persona's
    extraction sidecars already tagged with `key`, predicts a value for the ones they did not,
    from the entity's own text (the mean embedding of its passages) and the network's structure.
    Only entity-*owned* values are training signal -- a borrowed one is the majority vote of the
    entity's own documents, and training on it would fit that circularity rather than the
    attribute; `gnn.complete_attribute`'s own docstring says why at length.

    `--method gat` also prints, per suggestion, which neighbours the fitted attention leaned on
    most (`--gat-heads` attention heads per layer, averaged); the other two methods never learn a
    per-neighbour weight to report.

    Nothing here is written to the graph. Every suggestion is a row to paste into an entity's
    extraction sidecar under `attributes:`, or to discard; a person still decides.

    Needs the `graphrag[gnn]` extra (CPU-only `torch`); exits 2 with the install command when it
    is not present, before touching the persona at all.
    """
    from graphrag.sna import gnn

    # Every argument this command can validate on its own -- before the extra is even checked
    # for, so a typo in `--method` or `--hidden-dims` is reported the same way whether or not
    # `torch` happens to be installed here.
    if method not in gnn.METHODS:
        err.print(f"[red]method must be one of {gnn.METHODS}, got {method!r}[/red]")
        raise typer.Exit(2)
    try:
        dims = tuple(int(d.strip()) for d in hidden_dims.split(",") if d.strip())
    except ValueError:
        err.print(
            f"[red]--hidden-dims must be a comma separated list of integers, got "
            f"{hidden_dims!r}[/red]"
        )
        raise typer.Exit(2) from None
    if not dims:
        err.print("[red]--hidden-dims needs at least one layer width[/red]")
        raise typer.Exit(2)
    if gat_heads < 1:
        err.print(f"[red]--gat-heads must be at least 1, got {gat_heads}[/red]")
        raise typer.Exit(2)
    if not gnn.available():
        # markup=False: the hint names the `graphrag[gnn]` extra, and rich would otherwise read
        # the brackets as a (nonexistent) style tag and silently drop them.
        err.print(gnn.INSTALL_HINT, style="red", markup=False)
        raise typer.Exit(2)

    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    stances = _stances_or_exit(stance)
    filters = _where_or_exit(where)
    kinds = parse_types(types)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                "entities",
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=kinds,
                stances=stances,
                facets=_clean(facet),
                where=filters,
            )
        )
        text = ctx.store.mean_embeddings(persona, level="entity")
        snapshot_commit = snapshot_commit_for(ctx.settings, persona)
    finally:
        ctx.close()

    try:
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
            heads=gat_heads,
        )
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    prov = make_provenance(
        method="complete",
        parameters={
            "persona": persona,
            "key": key,
            "method": method,
            "hidden_dims": hidden_dims,
            "epochs": epochs,
            "lr": lr,
            "gat_heads": gat_heads,
            "val_share": val_share,
            "top": top,
            "source": source,
            "min_weight": min_weight,
            "types": parse_types(types),
            "stance": stance,
            "facet": facet,
            "where": filters,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["complete"] if method == "gat" else (44,),
        snapshot_commit=snapshot_commit,
    )
    rendered = _with_provenance(gnn.render_completion(report, top=top), prov, cite=cite)
    console.print(f"[bold]{persona} / entities[/bold]: completing {key!r} ({method})")
    console.print(rendered, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(gnn.completion_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("robustness")
def sna_robustness(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            help=(
                "Targeted removal order; repeatable. Any centrality (degree, betweenness, "
                "pagerank, coreness, ...). Random removal always runs as the null."
            ),
        ),
    ] = None,
    recompute: Annotated[
        bool,
        typer.Option("--recompute", help="Recompute the ranking on the survivors at every step."),
    ] = False,
    steps: Annotated[int, typer.Option(help="How many removal fractions to measure.")] = 20,
    runs: Annotated[int, typer.Option(help="Independent orders averaged per curve.")] = 20,
    seed: Annotated[int | None, typer.Option(help="Fix the removal orders.")] = None,
    cascade: Annotated[
        bool, typer.Option("--cascade", help="Also run §22.3's load-redistribution cascade.")
    ] = False,
    tolerance: Annotated[
        list[float] | None,
        typer.Option(
            "--tolerance",
            help="Cascade slack: capacity = (1 + tolerance) x initial load; repeatable.",
        ),
    ] = None,
    load: Annotated[str, typer.Option("--load", help="Cascade load: degree | betweenness.")] = (
        "degree"
    ),
    couple: Annotated[
        str | None,
        typer.Option("--couple", help="Also couple this persona's <network> to that one (§22.4)."),
    ] = None,
    coupling: Annotated[
        str, typer.Option("--coupling", help="same-id | random | degree.")
    ] = "same-id",
    max_nodes: Annotated[
        int,
        typer.Option("--max-nodes", help="Refuse a network larger than this: the curves are slow."),
    ] = 2000,
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """How much of this network survives losing its nodes: random, targeted, cascade, coupled.

    Atlas ch. 22. Nodes are removed a fraction at a time and what is left of the largest
    component is the measure, which is the chapter's own criterion (*"the share of nodes part of
    its largest component"*, p. 315). Random removal (§22.1) is the null every targeted curve is
    read against — a behavioural null, not a statistical one: it is what an accident does, so
    what an attacker does can be quoted as a difference. Beside the curves sits the critical
    fraction the degree distribution implies, κ = ⟨k²⟩/⟨k⟩ and f_c = 1 - 1/(kappa - 1), which is not
    in the chapter (it states the transition and cites Cohen et al. 2000 for the form).

    `--strategy` is §22.2's attacker, repeatable: degree is the chapter's own, and betweenness
    frequently does more damage for the same effort. `--recompute` asks again after every step,
    which is the stronger attack; without it the ranking is the one the original network had,
    and the report says which was run.

    `--cascade` is §22.3: a node's load is redistributed over its neighbours when it fails, and
    a neighbour whose new load exceeds its capacity fails in turn. The load (`--load`) and the
    slack (`--tolerance`, repeatable, capacity = (1 + tolerance) * the initial load) are
    assumptions, not measurements — nothing is actually carried between two entities that
    co-occur — so the sweep is read comparatively.

    `--couple <network>` is §22.4: two of this persona's networks made interdependent, node for
    node, with failures bouncing between them until the mutual giant component settles. The
    collapse is abrupt, so the whole curve is printed; `--coupling same-id` pairs the nodes the
    two networks share by name, `random` is the chapter's model assumption and `degree` its
    perfect correlation.

    What a removal is *not*: a failure. Nothing flows through a corpus network and no node in it
    can be attacked. These curves say how concentrated the evidence is — how much structure
    survives losing the best-connected records — which is a question about the corpus.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
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

    network = _network_or_exit(network)
    other_name = _network_or_exit(couple) if couple is not None else None
    if load not in LOADS:
        err.print(f"[red]--load must be one of {', '.join(LOADS)}[/red]")
        raise typer.Exit(2)
    if coupling not in COUPLINGS:
        err.print(f"[red]--coupling must be one of {', '.join(COUPLINGS)}[/red]")
        raise typer.Exit(2)
    if steps < 1 or runs < 1:
        err.print("[red]--steps and --runs must be at least 1[/red]")
        raise typer.Exit(2)
    tolerances = tuple(tolerance) if tolerance else CASCADE_TOLERANCES
    if any(value < 0 for value in tolerances):
        err.print("[red]--tolerance must be at least 0[/red]")
        raise typer.Exit(2)

    kinds = parse_types(types)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=kinds,
            )
        )
        other = (
            _network_or_bad_parameter(
                lambda: build_network_cached(
                    ctx.store,
                    other_name,
                    persona,
                    settings=ctx.settings,
                    no_cache=no_cache,
                    source_id=source,
                    min_weight=min_weight,
                    types=kinds,
                )
            )
            if other_name is not None
            else None
        )
    finally:
        ctx.close()

    if graph.number_of_nodes() == 0:
        err.print(f"[red]the {network} network is empty, so there is nothing to remove[/red]")
        raise typer.Exit(2)
    if graph.number_of_nodes() > max_nodes:
        err.print(
            f"[red]{graph.number_of_nodes():,} nodes is above --max-nodes {max_nodes:,}: every "
            "curve removes the network one step at a time and recomputes the components, so the "
            "cost grows with n. Raise --max-nodes, or lower --steps and --runs[/red]"
        )
        raise typer.Exit(2)
    chosen = [s.strip() for s in strategy or []] or list(TARGETED_STRATEGIES)
    allowed = strategies_for(graph)
    unknown = [s for s in chosen if s not in allowed or s == "random"]
    if unknown:
        err.print(
            f"[red]--strategy must be one of {', '.join(s for s in allowed if s != 'random')}; "
            f"got {', '.join(unknown)}. Random removal always runs as the null[/red]"
        )
        raise typer.Exit(2)

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
            coupled_network=other_name or "",
        )
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
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
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance(render_robustness(report), prov, cite=cite)
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(robustness_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("motifs")
def sna_motifs(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    size: Annotated[
        int, typer.Option(help="Motif size in nodes: 3 always, 4 on an undirected network.")
    ] = 3,
    samples: Annotated[int, typer.Option(help="Degree-preserving rewirings for the null.")] = (
        MOTIF_SAMPLES
    ),
    seed: Annotated[int | None, typer.Option(help="Seed for the null rewirings.")] = None,
    mine: Annotated[
        bool, typer.Option("--mine", help="Also mine frequent subgraphs (§41.4, §41.5). Slow.")
    ] = False,
    min_support: Annotated[
        int, typer.Option("--min-support", help="Support a pattern needs to be reported.")
    ] = MOTIF_MIN_SUPPORT,
    max_size: Annotated[
        int, typer.Option("--max-size", help="Largest pattern to grow, in edges.")
    ] = MOTIF_MAX_SIZE,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Relation type (--network relations); repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    max_nodes: Annotated[
        int,
        typer.Option(
            "--max-nodes",
            help="Refuse --mine on a network larger than this: §41.5 enumerates every subgraph.",
        ),
    ] = MOTIFS_MAX_NODES,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """What small shapes this network is built out of, and which of them are surprising (ch. 41).

    Three sections, and the first one is always printed. The **census** sorts every triple of
    nodes into its class — four classes undirected, the sixteen MAN types on `--network
    relations` — so a count is read against the number of triples that could have been anything
    else, the way §10.3's dyad census is read. Undirected, the last two classes are §12.2's own
    (a triad and a triangle, Figure 12.4) and the global clustering coefficient falls straight
    out of them.

    The **motif profile** is §41.2's four-step procedure: count each shape, rewire the network
    keeping every degree fixed (§19.1, in-degree and out-degree separately when it is directed),
    count again, and report how many null standard deviations apart the two are. That is what
    turns a count into a finding, and it is two-tailed because a shape the network *avoids* is a
    finding too. `--size 4` widens it to four-node shapes on an undirected network — the square
    and the diamond and the rest — and is refused on a directed one, where there are 199 of them.

    `--mine` adds the bottom-up half, where the data names the shapes instead of you. §41.4's
    **transactional mining** runs over one entity graph per document (`--network entities` only,
    because that is the database this corpus has) and its support counts *documents*: a triangle
    with support 5 was found in five episodes, however many triangles each held. §41.5's
    **single-graph mining** runs over the network itself and reports the minimum image support —
    for each node of the pattern, how many different network nodes ever play that role, then the
    smallest of those counts — which is deliberately smaller than what you could count by eye,
    because a support that can grow with the pattern makes the search impossible (p. 601).

    Mining is opt-in and bounded (`--max-size` in edges, `--max-nodes` on the network) because
    §41.2 says plainly that finding motifs in a large network is a hard problem; when a bound is
    hit the command says so and exits 2 rather than running for a week. And the caveat that
    outlasts every number here: every network but `relations` is a projection, so a passage
    naming three things is a triangle by construction, and a census of these shapes counts how
    the corpus was written down before it counts anything about the subject.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.motifs import (
        build_motifs_report,
        document_graphs,
        motifs_payload,
        render_motifs,
    )
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    entity_types = parse_types(types)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=entity_types,
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
        documents = (
            document_graphs(
                ctx.store,
                persona,
                source_id=source,
                types=entity_types,
                stances=stances,
                facets=_clean(facet),
                since=since,
                until=until,
                where=filters,
            )
            if mine and network == "entities"
            else None
        )
    finally:
        ctx.close()
    if graph.number_of_nodes() < 3:
        err.print("[red]fewer than three nodes, so there is no three-node shape to count[/red]")
        raise typer.Exit(2)
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if mine and documents is None:
        err.print(
            "[yellow]no transactional section: §41.4 needs a database of small graphs, and the "
            "one this corpus has is one entity graph per document, so it is offered on --network "
            "entities only[/yellow]"
        )
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
            "where": filters,
            "project": project,
            "max_nodes": max_nodes,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["motifs"],
        null_model="degree-preserving rewiring (§19.1)",
        null_model_samples=samples,
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance(render_motifs(report), prov, cite=cite)
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(motifs_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("spread")
def sna_spread(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    model: Annotated[str, typer.Option(help="si | sis | sir | threshold | limited")] = "si",
    seeds: Annotated[
        str | None,
        typer.Option(
            "--seeds",
            help="Patient zeros, comma separated. A bare number draws that many at random.",
        ),
    ] = None,
    beta: Annotated[
        float,
        typer.Option(help="Infection probability per infected neighbour per step (§20.1)."),
    ] = SPREAD_BETA,
    mu: Annotated[
        float, typer.Option(help="Recovery probability per step; sis and sir only (§20.2).")
    ] = SPREAD_MU,
    threshold: Annotated[
        float,
        typer.Option(
            help="--model threshold: kappa when >= 1 (§21.1), a fraction of neighbours below 1."
        ),
    ] = SPREAD_THRESHOLD,
    attempts: Annotated[
        int, typer.Option(help="--model limited: steps a new infection keeps trying (§21.2).")
    ] = SPREAD_ATTEMPTS,
    steps: Annotated[int, typer.Option(help="Steps per run.")] = SPREAD_STEPS,
    runs: Annotated[int, typer.Option(help="Runs to band the curve over.")] = SPREAD_RUNS,
    share: Annotated[
        float | None,
        typer.Option("--share", help="Immunise this share of the network and compare (§21.3)."),
    ] = None,
    strategy: Annotated[
        str, typer.Option(help="random | degree | betweenness | acquaintance (§21.3).")
    ] = "random",
    seed: Annotated[
        int | None, typer.Option(help="Random seed; without one the run cannot be reproduced.")
    ] = None,
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the markdown report here.")
    ] = None,
    as_json: Annotated[Path | None, typer.Option("--json", help="Also write the payload here.")] = (
        None
    ),
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    stance: Annotated[
        list[str] | None, typer.Option("--stance", help="Stance filter; repeatable.")
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Relation type (--network relations); repeatable."),
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    project: Annotated[
        str | None, typer.Option(help="Collapse the two-mode network onto speakers | entities.")
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """What would spread on this network, if anything ever did (Atlas ch. 20 and 21).

    **Nothing spread.** Chapter 20's models are dynamics you *embed* in a network — "edges don't
    change, but nodes can transition into different states" (p. 286) — and the network they are
    embedded in here joins two nodes because a passage named them together, not because anything
    passed between them. So this is a what-if over the structure: *if* something moved along the
    lines this corpus drew, at rate `--beta`, this is how far it would get and how fast. The
    report says so above the numbers, and a final size is a statement about connectedness, never
    a forecast.

    `--model si|sis|sir` are the three compartmental models of chapter 20; `threshold` is §21.1's
    complex contagion, where a node needs several infected neighbours at once (Granovetter's
    `kappa` at or above 1, Watts' cascade fraction below it); `limited` is §21.2's independent
    cascade, where a new infection gets `--attempts` steps to persuade its neighbours and then
    stops trying forever. The curve is banded over `--runs` seeded simulations; the band is the
    stochasticity of the process, and there is no null model here — a simulation is not a
    measurement that could have come out otherwise by chance.

    What plays that role is the **epidemic threshold** (§20.2), printed before the curve:
    `beta/mu` against `1/lambda_1` (exact for this network), `1/(k+1)` (what the chapter gives
    for a Gn,p graph) and `k/k^2` (what it gives for a preferential-attachment one, the form that
    "tends to zero" on a heavy tail). `--share` adds §21.3's intervention comparison — the same
    outbreak with and without immunisation, judged on both of the chapter's criteria, the drop in
    final size and the delay — and every report ends on §21.4's driver-node share, the smallest
    set of levers this network could be steered from, by maximum matching.

    Edge weights are ignored: beta is a per-contact probability and an edge is one contact. Use
    `--min-weight` if a single co-mention is too thin a contact to run a spread along.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for
    from graphrag.sna.spread import (
        IMMUNISATION_STRATEGIES,
        SPREAD_MODELS,
        build_spread_report,
        render_spread,
        spread_payload,
    )

    network = _network_or_exit(network)
    stances = _stances_or_exit(stance)
    side = _project_or_exit(project)
    filters = _where_or_exit(where)
    if model not in SPREAD_MODELS:
        err.print(f"[red]--model must be one of {', '.join(SPREAD_MODELS)}, got {model}[/red]")
        raise typer.Exit(2)
    if strategy not in IMMUNISATION_STRATEGIES:
        err.print(
            f"[red]--strategy must be one of {', '.join(IMMUNISATION_STRATEGIES)}, "
            f"got {strategy}[/red]"
        )
        raise typer.Exit(2)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                stances=stances,
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
                project=side,
            )
        )
    finally:
        ctx.close()
    if graph.number_of_nodes() == 0:
        err.print("[red]this network has no nodes, so there is nothing to spread on[/red]")
        raise typer.Exit(2)
    try:
        report = build_spread_report(
            graph,
            persona_id=persona,
            network=network,
            model=model,
            seeds=_seeds_or_exit(seeds),
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
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
            "where": filters,
            "project": project,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["spread"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    markdown = _with_provenance(render_spread(report), prov, cite=cite)
    console.print(markdown, markup=False, highlight=False)
    if out is not None:
        _warn_if_ephemeral(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]wrote[/green] {out}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(spread_payload(report), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("draw")
def sna_draw(
    persona: Annotated[str, typer.Argument(help="Persona id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="SVG file to write.")],
    layout: Annotated[
        str, typer.Option(help="force | circular | arc | matrix | layered (Atlas ch. 51).")
    ] = "force",
    size: Annotated[
        str | None,
        typer.Option(help="A centrality name (Atlas ch. 14); uniform radius when unset."),
    ] = None,
    color: Annotated[
        str | None,
        typer.Option("--color", help="community (Louvain) or attr:<key>; uniform fill when unset."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Fixes the force layout and the Louvain run.")
    ] = None,
    width: Annotated[float, typer.Option(help="Canvas width in px.")] = DRAW_WIDTH,
    height: Annotated[float, typer.Option(help="Canvas height in px.")] = DRAW_HEIGHT,
    html: Annotated[
        Path | None, typer.Option("--html", help="HTML file to write (default: --out, .html).")
    ] = None,
    report: Annotated[
        Path | None, typer.Option("--report", help="Also write the markdown section here.")
    ] = None,
    as_json: Annotated[
        Path | None,
        typer.Option("--json", help="Also write positions/sizes/colours as JSON here."),
    ] = None,
    network: Annotated[
        str, typer.Option(help="speakers | entities | topics | speakers-entities | relations")
    ] = "entities",
    source: Annotated[str | None, typer.Option(help="Limit to one source id.")] = None,
    min_weight: Annotated[int | None, typer.Option("--min-weight", help=MIN_WEIGHT_HELP)] = None,
    types: Annotated[str | None, typer.Option(help="Entity types, comma separated.")] = None,
    relation_type: Annotated[
        list[str] | None,
        typer.Option("--relation-type", help="Relation type (--network relations); repeatable."),
    ] = None,
    facet: Annotated[
        list[str] | None, typer.Option("--facet", help="Facet filter; repeatable.")
    ] = None,
    since: Annotated[str | None, typer.Option(help="Earliest post date, ISO YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Latest post date, ISO YYYY-MM-DD.")] = None,
    where: Annotated[
        list[str] | None,
        typer.Option("--where", help="Keep only nodes attributed key=value; repeatable."),
    ] = None,
    no_cache: Annotated[bool, NO_CACHE_OPTION] = False,
    cite: Annotated[bool, CITE_OPTION] = False,
) -> None:
    """Draw one network: an SVG and a self-contained interactive HTML (Atlas ch. 49-51).

    `--size <centrality>` maps the value onto node *area*, quasi-logged first so one hub cannot
    swamp the picture (§49.1); `--color community` runs Louvain (§35.6, imported rather than
    rerun by this command) and `--color attr:<key>` reads a node attribute an
    attribution/extraction sidecar wrote (§7.5), categorical or, if every value parses as a
    number, sequential. Both palettes are colour-blind-safe and stop at nine colours total
    (§49.2); `sna draw --color community` also prints `## Community evaluation` (ch. 36) for the
    partition it coloured by.

    Edges are styled from the same measure that set the node size when there is one -- edge
    weight normally, edge betweenness when `--size betweenness` (§50.1's own pairing) -- at a
    fixed transparency regardless (Figure 50.4(b)). `--layout layered` needs a directed network
    (`--network relations`) and reuses `sna hierarchy`'s own layering (§33.6); the other four
    lay out any network. `graphrag sna guide` prints the cautions this command already enforces
    (node size is area, not radius; no more than nine colours; a hairball is a layout failure,
    not evidence of nothing).

    The HTML fetches nothing external: hover a node for its name, drag one to move it (straight
    edges follow; curved ones do not), and a legend on the side names every colour. `--json`
    writes the same positions, sizes and colours the picture used, for `docs/SNA.md`'s §51.5
    recipes -- reading a force layout's coordinates back against `sna analyze`'s own centrality
    columns.
    """
    from graphrag.sna.cache import build_network_cached
    from graphrag.sna.draw import LAYOUTS, draw, draw_payload, render_draw
    from graphrag.sna.provenance import COMMAND_CHAPTERS, make_provenance, snapshot_commit_for

    network = _network_or_exit(network)
    filters = _where_or_exit(where)
    if layout not in LAYOUTS:
        err.print(f"[red]--layout must be one of {', '.join(LAYOUTS)}, got {layout}[/red]")
        raise typer.Exit(2)
    ctx = State.context()
    try:
        ctx.registry.get(persona)
        graph = _network_or_bad_parameter(
            lambda: build_network_cached(
                ctx.store,
                network,
                persona,
                settings=ctx.settings,
                no_cache=no_cache,
                source_id=source,
                min_weight=min_weight,
                types=parse_types(types),
                facets=_clean(facet),
                relation_types=_clean(relation_type),
                since=since,
                until=until,
                where=filters,
            )
        )
    finally:
        ctx.close()

    try:
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
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

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
            "where": filters,
        },
        seed=seed,
        chapters=COMMAND_CHAPTERS["draw"],
        snapshot_commit=snapshot_commit_for(ctx.settings, persona),
    )
    section = _with_provenance("\n".join(render_draw(result)), prov, cite=cite)
    console.print(section, markup=False, highlight=False)

    _warn_if_ephemeral(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.svg, encoding="utf-8")
    console.print(f"[green]wrote[/green] {out}")

    html_path = html if html is not None else out.with_suffix(".html")
    _warn_if_ephemeral(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(result.html, encoding="utf-8")
    console.print(f"[green]wrote[/green] {html_path}")

    if report is not None:
        _warn_if_ephemeral(report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(section, encoding="utf-8")
        console.print(f"[green]wrote[/green] {report}")
    if as_json is not None:
        _warn_if_ephemeral(as_json)
        as_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload_with_provenance(draw_payload(result), prov)
        as_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {as_json}")


@sna_app.command("guide")
def sna_guide() -> None:
    """Print the method-selection rules: which network, which method, which centrality."""
    from graphrag.sna.guide import render_guide

    console.print(render_guide(), markup=False, highlight=False)


cache_app = typer.Typer(
    help="The on-disk network cache every `sna` command reads through (ATL-ENT-3).",
    no_args_is_help=True,
)
sna_app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def sna_cache_clear() -> None:
    """Empty the on-disk network cache: the next command that builds a network rebuilds it."""
    from graphrag.sna.cache import NetworkCache

    settings = _settings()
    removed = NetworkCache(settings.sna_cache_dir).clear()
    console.print(
        f"[green]cleared[/green] {removed} cached network(s) from {settings.sna_cache_dir}"
    )


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
        if not dry_run and report.mentions_moved:
            _invalidate_network_cache(ctx.settings, spec.id)
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
        if not dry_run:
            _invalidate_network_cache(ctx.settings, spec.id)
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
        if not dry_run and removed:
            _invalidate_network_cache(ctx.settings, spec.id)
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
        for col in ("graph", *LAYERS, "attributes", "loose"):
            table.add_column(col)
        for document in report.documents:
            table.add_row(
                document.doc_id,
                document.source_id,
                "yes" if document.in_graph else "[red]no[/red]",
                *(_layer_cell(document.state(layer)) for layer in LAYERS),
                str(document.attribute_count) if document.attribute_count else "",
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
