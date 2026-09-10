"""Group turns into retrieval-sized chunks.

Turns are packed until ``target_words`` is reached; a single oversized turn is split on sentence
boundaries. Chunks never straddle a speaker change *and* a size boundary at the same time, which
keeps the ``speaker`` attribution on each chunk honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from graphrag.ingest.transcript import deep_link
from graphrag.models import Chunk, LoadedDocument, Turn

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkerConfig:
    target_words: int = 300
    max_words: int = 450
    min_words: int = 40


def _split_long_turn(turn: Turn, target_words: int) -> list[Turn]:
    words = turn.text.split()
    if len(words) <= target_words:
        return [turn]
    pieces: list[Turn] = []
    current: list[str] = []
    count = 0
    for sentence in SENTENCE_END.split(turn.text):
        n = len(sentence.split())
        if current and count + n > target_words:
            pieces.append(turn.model_copy(update={"text": " ".join(current)}))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(turn.model_copy(update={"text": " ".join(current)}))
    return pieces


def chunk_document(loaded: LoadedDocument, config: ChunkerConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkerConfig()
    doc = loaded.document
    units: list[Turn] = []
    for turn in loaded.turns:
        units.extend(_split_long_turn(turn, cfg.target_words))

    chunks: list[Chunk] = []
    group: list[Turn] = []
    group_words = 0

    def flush() -> None:
        nonlocal group, group_words
        if not group:
            return
        first = group[0]
        speakers: list[str] = []
        for t in group:
            if t.speaker and t.speaker not in speakers:
                speakers.append(t.speaker)
        text = "\n\n".join(
            (f"{t.speaker}: {t.text}" if t.speaker and len(speakers) > 1 else t.text) for t in group
        )
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.id}#{ordinal:04d}",
                doc_id=doc.id,
                persona_id=doc.persona_id,
                ordinal=ordinal,
                text=text,
                speaker=first.speaker,
                speakers=speakers,
                start_ts=first.start_ts,
                start_seconds=first.start_seconds,
                url=deep_link(doc.url, first.start_seconds),
                word_count=group_words,
            )
        )
        group, group_words = [], 0

    for unit in units:
        n = len(unit.text.split())
        if group and group_words + n > cfg.max_words:
            flush()
        group.append(unit)
        group_words += n
        if group_words >= cfg.target_words:
            flush()

    # Avoid a tiny trailing chunk: merge it into the previous one when possible.
    if group and group_words < cfg.min_words and chunks:
        last = chunks.pop()
        merged_text = last.text + "\n\n" + "\n\n".join(t.text for t in group)
        merged_speakers = list(last.speakers)
        for t in group:
            if t.speaker and t.speaker not in merged_speakers:
                merged_speakers.append(t.speaker)
        chunks.append(
            last.model_copy(
                update={
                    "text": merged_text,
                    "speakers": merged_speakers,
                    "word_count": last.word_count + group_words,
                }
            )
        )
        group, group_words = [], 0
    flush()
    return chunks
