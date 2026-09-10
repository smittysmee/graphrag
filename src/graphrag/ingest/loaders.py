"""Loader registry: a persona source names a loader; the loader yields ``LoadedDocument``s."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from graphrag.ingest.documents import load_document
from graphrag.ingest.transcript import load_transcript
from graphrag.models import LoadedDocument, LoaderName, SourceSpec

Loader = Callable[..., LoadedDocument]

LOADERS: dict[LoaderName, Loader] = {
    "transcripts": load_transcript,
    "documents": load_document,
}

DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_WS = re.compile(r"\s+")
_FINGERPRINT_SKIP = 3000  # skip the opening chars (intros/ads differ between re-uploads)


def content_fingerprint(loaded: LoadedDocument) -> str:
    """Hash of the normalised body text. Two files with the same fingerprint are the same
    document, whatever their metadata says (archives re-upload episodes with new ids)."""
    text = _WS.sub(" ", " ".join(t.text for t in loaded.turns)).strip().lower()
    if len(text) > 2 * _FINGERPRINT_SKIP:
        text = text[_FINGERPRINT_SKIP:]
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324


def iter_source_files(root: Path, source: SourceSpec) -> list[Path]:
    base = root / source.path if source.path else root
    files = sorted(p for p in base.glob(source.glob) if p.is_file())
    if source.loader == "documents":
        files = [p for p in files if p.suffix.lower() in DOCUMENT_SUFFIXES]
    return files


def load_source(root: Path, source: SourceSpec, persona_id: str) -> Iterator[LoadedDocument]:
    """Yield documents in sorted file order. Files whose body text was already seen are skipped
    (the same episode archived twice under different metadata); ids that collide (e.g. folders
    ``guest`` and ``guest_`` for a repeat guest, or two episodes sharing a URL by mistake) get a
    deterministic ``-2``, ``-3`` suffix."""
    loader = LOADERS[source.loader]
    base = root / source.path if source.path else root
    seen: dict[str, int] = {}
    seen_bodies: set[str] = set()
    for path in iter_source_files(root, source):
        loaded: LoadedDocument = loader(path, root=base, persona_id=persona_id, source_id=source.id)
        if loaded.turns:
            fingerprint = content_fingerprint(loaded)
            if fingerprint in seen_bodies:
                continue
            seen_bodies.add(fingerprint)
        doc_id = loaded.document.id
        if doc_id in seen:
            seen[doc_id] += 1
            unique = f"{doc_id}-{seen[doc_id]}"
            loaded = loaded.model_copy(
                update={"document": loaded.document.model_copy(update={"id": unique})}
            )
        else:
            seen[doc_id] = 1
        yield loaded
