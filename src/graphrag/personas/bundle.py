"""Persona bundles: pack a persona and everything that makes it work into one file.

A bundle is a gzipped tar carrying the persona definition, optionally its built graph, and
optionally the entity-extraction JSON. It is the unit you hand to someone else::

    graphrag persona export product-leader --out product-leader.tar.gz
    # ... send the file ...
    graphrag persona import product-leader.tar.gz --load

Deliberately *not* included: the embedding model (large, pinned, fetched by the recipient) and
any Docker images. The manifest records which embedder built the graph so the import side can
refuse a mismatch up front rather than returning nonsense at query time.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from graphrag import __version__
from graphrag.graph import snapshot as snap
from graphrag.models import PersonaSpec
from graphrag.personas.registry import PERSONA_FILE, PersonaRegistry

BUNDLE_MANIFEST = "bundle.json"
MODEL_CACHE_DIR = "model-cache"
FORMAT_VERSION = 1


class BundleError(RuntimeError):
    """Raised when a bundle cannot be written, read or safely applied."""


class BundleManifest(BaseModel):
    """What is inside a bundle, so the import side can check before unpacking anything."""

    format_version: int = FORMAT_VERSION
    persona_id: str
    persona_name: str
    exported_at: datetime
    tool_version: str = __version__
    includes_graph: bool = False
    embedding_model: str | None = None
    embedding_dim: int | None = None
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    enrichment_files: int = 0
    embedding_precision: str = "float32"
    includes_model: bool = False
    model_bytes: int = 0
    notes: str = ""


@dataclass
class ImportReport:
    manifest: BundleManifest
    persona_path: Path
    snapshot_path: Path | None = None
    enrichment_paths: list[Path] = field(default_factory=list)
    model_restored: bool = False
    loaded: bool = False


def _enrichment_files(enrichment_dir: Path, persona_id: str) -> list[Path]:
    """Extraction JSON belonging to one persona, matched on the doc_id prefix."""
    if not enrichment_dir.is_dir():
        return []
    prefix = f"{persona_id}:"
    out: list[Path] = []
    for path in sorted(enrichment_dir.glob("*.json")):
        try:
            doc_id = json.loads(path.read_text(encoding="utf-8")).get("doc_id", "")
        except (OSError, ValueError):
            continue
        if isinstance(doc_id, str) and doc_id.startswith(prefix):
            out.append(path)
    return out


def export_bundle(
    persona: PersonaSpec,
    out_file: Path,
    *,
    personas_dir: Path,
    snapshots_dir: Path,
    enrichment_dir: Path | None = None,
    include_graph: bool = True,
    include_enrichment: bool = True,
    model_cache_dir: Path | None = None,
    compact: bool = False,
    notes: str = "",
) -> BundleManifest:
    """Write ``out_file`` containing the persona and, when present, its graph and extractions."""
    persona_dir = personas_dir / persona.id
    if not (persona_dir / PERSONA_FILE).exists():
        msg = f"no persona definition at {persona_dir / PERSONA_FILE}"
        raise BundleError(msg)

    snapshot_dir = snap.snapshot_dir(snapshots_dir, persona.id)
    has_graph = include_graph and (snapshot_dir / snap.MANIFEST).exists()
    snapshot_manifest = snap.read_manifest(snapshot_dir) if has_graph else None

    extractions = (
        _enrichment_files(enrichment_dir, persona.id)
        if include_enrichment and enrichment_dir is not None
        else []
    )

    carry_model = model_cache_dir is not None and model_cache_dir.is_dir()
    model_bytes = _tree_bytes(model_cache_dir) if carry_model and model_cache_dir is not None else 0

    manifest = BundleManifest(
        persona_id=persona.id,
        persona_name=persona.name,
        exported_at=datetime.now(tz=UTC),
        includes_graph=has_graph,
        embedding_model=snapshot_manifest.embedding_model if snapshot_manifest else None,
        embedding_dim=snapshot_manifest.embedding_dim if snapshot_manifest else None,
        document_count=snapshot_manifest.document_count if snapshot_manifest else 0,
        chunk_count=snapshot_manifest.chunk_count if snapshot_manifest else 0,
        entity_count=snapshot_manifest.entity_count if snapshot_manifest else 0,
        enrichment_files=len(extractions),
        embedding_precision="float16" if (compact and has_graph) else "float32",
        includes_model=carry_model,
        model_bytes=model_bytes,
        notes=notes,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        manifest_file = Path(tmp) / BUNDLE_MANIFEST
        manifest_file.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        with tarfile.open(out_file, "w:gz") as tar:
            tar.add(manifest_file, arcname=BUNDLE_MANIFEST)
            tar.add(persona_dir, arcname=f"personas/{persona.id}")
            if has_graph:
                if compact:
                    staged = Path(tmp) / "snapshot"
                    _stage_compact_snapshot(snapshot_dir, staged)
                    tar.add(staged, arcname=f"data/snapshots/{persona.id}")
                else:
                    tar.add(snapshot_dir, arcname=f"data/snapshots/{persona.id}")
            for path in extractions:
                tar.add(path, arcname=f"data/enrichment/{path.name}")
            if carry_model and model_cache_dir is not None:
                tar.add(model_cache_dir, arcname=MODEL_CACHE_DIR)
    return manifest


def read_bundle_manifest(bundle_file: Path) -> BundleManifest:
    """Read the manifest without unpacking anything else."""
    if not bundle_file.exists():
        msg = f"no such bundle: {bundle_file}"
        raise BundleError(msg)
    try:
        with tarfile.open(bundle_file, "r:gz") as tar:
            member = tar.extractfile(BUNDLE_MANIFEST)
            if member is None:
                msg = f"{bundle_file} has no {BUNDLE_MANIFEST}; is it a persona bundle?"
                raise BundleError(msg)
            manifest = BundleManifest.model_validate_json(member.read().decode("utf-8"))
    except (tarfile.TarError, KeyError) as exc:
        msg = f"{bundle_file} is not a readable persona bundle: {exc}"
        raise BundleError(msg) from exc
    if manifest.format_version > FORMAT_VERSION:
        msg = (
            f"bundle format v{manifest.format_version} is newer than this build understands "
            f"(v{FORMAT_VERSION}); upgrade graphrag"
        )
        raise BundleError(msg)
    return manifest


def check_embedder(manifest: BundleManifest, model_name: str, dim: int) -> None:
    """Refuse a graph built with a different embedder: its vectors live in another space."""
    if not manifest.includes_graph:
        return
    if manifest.embedding_model != model_name or manifest.embedding_dim != dim:
        msg = (
            f"this bundle's graph was built with {manifest.embedding_model} "
            f"({manifest.embedding_dim} dims) but your embedder is {model_name} ({dim} dims). "
            f"Set GRAPHRAG_EMBEDDING_MODEL={manifest.embedding_model} and "
            f"GRAPHRAG_EMBEDDING_DIM={manifest.embedding_dim}, or import with --no-graph and "
            f"re-ingest the sources yourself."
        )
        if manifest.includes_model:
            msg += " This bundle ships that model, so you will not need to download it."
        raise BundleError(msg)


def import_bundle(
    bundle_file: Path,
    *,
    personas_dir: Path,
    snapshots_dir: Path,
    enrichment_dir: Path,
    model_name: str,
    dim: int,
    model_cache_dir: Path | None = None,
    include_graph: bool = True,
    include_model: bool = True,
    overwrite: bool = False,
) -> ImportReport:
    """Unpack a bundle into this installation. Does not touch the database."""
    manifest = read_bundle_manifest(bundle_file)
    if include_graph:
        check_embedder(manifest, model_name, dim)

    registry = PersonaRegistry(personas_dir)

    with TemporaryDirectory() as tmp:
        staging = Path(tmp)
        with tarfile.open(bundle_file, "r:gz") as tar:
            # filter="data" rejects absolute paths, traversal and special files: a bundle is
            # something someone else sent you.
            tar.extractall(staging, filter="data")

        report = ImportReport(
            manifest=manifest, persona_path=registry.path_for(manifest.persona_id)
        )

        src_persona = staging / "personas" / manifest.persona_id
        if not (src_persona / PERSONA_FILE).exists():
            msg = f"bundle is missing personas/{manifest.persona_id}/{PERSONA_FILE}"
            raise BundleError(msg)

        # Only refuse when replacing would actually lose something. A repo may ship a persona
        # definition as an example, and importing its own bundle over an untouched copy is not
        # a clobber worth blocking.
        existing = registry.path_for(manifest.persona_id)
        if (
            existing.exists()
            and not overwrite
            and not _same_file(existing, src_persona / PERSONA_FILE)
        ):
            msg = (
                f"persona {manifest.persona_id!r} already exists here and differs from the one "
                f"in the bundle; pass --overwrite to replace it"
            )
            raise BundleError(msg)
        _replace_tree(src_persona, personas_dir / manifest.persona_id)

        src_snapshot = staging / "data" / "snapshots" / manifest.persona_id
        if include_graph and src_snapshot.is_dir():
            target = snap.snapshot_dir(snapshots_dir, manifest.persona_id)
            _replace_tree(src_snapshot, target)
            report.snapshot_path = target

        src_model = staging / MODEL_CACHE_DIR
        if include_model and src_model.is_dir() and model_cache_dir is not None:
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            for item in src_model.iterdir():
                target_item = model_cache_dir / item.name
                if target_item.exists():
                    continue  # never clobber a cache entry the recipient already has
                if item.is_dir():
                    shutil.copytree(item, target_item)
                else:
                    shutil.copy2(item, target_item)
            report.model_restored = True

        src_enrichment = staging / "data" / "enrichment"
        if src_enrichment.is_dir():
            enrichment_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(src_enrichment.glob("*.json")):
                target_file = enrichment_dir / path.name
                target_file.write_bytes(path.read_bytes())
                report.enrichment_paths.append(target_file)

    return report


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _stage_compact_snapshot(snapshot_dir: Path, staged: Path) -> None:
    """Copy a snapshot, re-storing its vectors as float16.

    Vectors dominate a bundle and barely compress: they are high-entropy floats, so gzip only
    reaches ~90% of raw. Halving the width is worth far more than any codec. The cost is about
    2e-4 of cosine score, which does not move the top result: measured over 300 real queries,
    top-1 never changed and worst-case top-10 overlap was 9/10. Neo4j widens them back on load.
    """
    import numpy as np

    staged.mkdir(parents=True, exist_ok=True)
    for path in sorted(snapshot_dir.iterdir()):
        if path.name == snap.EMBEDDINGS:
            np.save(staged / path.name, np.load(path).astype(np.float16))
        elif path.is_file():
            shutil.copy2(path, staged / path.name)


def _tree_bytes(root: Path) -> int:
    """Size on disk, counting each file once. A Hugging Face cache hardlinks its blobs into
    snapshots/, so a naive sum reports roughly double the real size."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        st = path.stat()
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += st.st_size
    return total


def _replace_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
