"""Generic loader for prose documents (.md, .txt, .pdf): paragraphs become turns."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from graphrag.ingest.frontmatter import split_frontmatter
from graphrag.models import Document, LoadedDocument, Scalar, Turn, slugify

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def read_text(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() == ".pdf":
        return {}, _read_pdf(path)
    return split_frontmatter(path.read_text(encoding="utf-8", errors="replace"))


def paragraphs_to_turns(body: str) -> list[Turn]:
    """Split on blank lines; a markdown heading becomes a prefix of the paragraph that follows."""
    turns: list[Turn] = []
    pending_heading: str | None = None
    for block in PARAGRAPH_SPLIT.split(body):
        text = block.strip()
        if not text:
            continue
        heading = HEADING.match(text)
        if heading and "\n" not in text:
            pending_heading = heading.group(1).strip()
            continue
        if pending_heading:
            text = f"{pending_heading}\n{text}"
            pending_heading = None
        turns.append(Turn(text=text))
    return turns


def _title_from(meta: dict[str, Any], body: str, path: Path) -> str:
    if meta.get("title"):
        return str(meta["title"]).strip()
    for line in body.splitlines():
        heading = HEADING.match(line.strip())
        if heading:
            return heading.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").strip()


def load_document(path: Path, *, root: Path, persona_id: str, source_id: str) -> LoadedDocument:
    meta, body = read_text(path)
    turns = paragraphs_to_turns(body)
    rel = path.relative_to(root)
    doc_id = f"{persona_id}:{source_id}:{slugify(str(rel.with_suffix('')))}"
    raw_topics = meta.get("keywords") or meta.get("topics") or meta.get("tags") or []
    topics = sorted({slugify(str(k)).replace("-", " ") for k in raw_topics if str(k).strip()})
    metadata: dict[str, Scalar] = {
        k: (v if isinstance(v, str | int | float | bool) or v is None else str(v))
        for k, v in meta.items()
        if k not in {"title", "keywords", "topics", "tags", "description", "url", "source_url"}
    }
    document = Document(
        id=doc_id,
        persona_id=persona_id,
        source_id=source_id,
        title=_title_from(meta, body, path),
        path=str(rel),
        url=str(meta.get("url") or meta.get("source_url") or "") or None,
        description=str(meta.get("description") or "").strip(),
        topics=topics,
        metadata=metadata,
        word_count=sum(len(t.text.split()) for t in turns),
    )
    return LoadedDocument(document=document, turns=turns)
