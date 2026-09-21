"""Reproducibility and citation (ATL-ENT-4), held to the book's own table of contents.

The "known answer" here is not a legendary graph -- there is no network in this ticket -- it is
the Atlas's own outline (``scripts/atlas_chapter.py toc``, read once while building
:data:`graphrag.sna.provenance.CHAPTER_TITLES` and asserted against again here), and the version
string and timestamp :func:`graphrag.sna.provenance.make_provenance` is required to produce
without being told what to produce.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from graphrag import __version__
from graphrag.cli import sna_app
from graphrag.config import Settings
from graphrag.graph import snapshot as snap
from graphrag.models import SnapshotManifest
from graphrag.sna.provenance import (
    BOOK_CITATION,
    CHAPTER_TITLES,
    COMMAND_CHAPTERS,
    Provenance,
    chapter_citation,
    make_provenance,
    provenance_payload,
    render_provenance,
    render_references,
    snapshot_commit_for,
)

ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# --------------------------------------------------------------------------------- make_provenance


def test_make_provenance_fills_the_tool_version_and_a_real_timestamp_unasked() -> None:
    """The two fields a caller never supplies: not re-derived from anything the report computed,
    but not left to the caller to get right twice either."""
    prov = make_provenance(method="degree", parameters={"persona": "p"}, chapters=[9])
    assert prov.tool_version == __version__
    assert ISO_TIMESTAMP.match(prov.generated_at)


def test_make_provenance_keeps_the_exact_parameters_and_seed_it_was_given() -> None:
    """Not re-derived: the dict object's *values* ride straight through, unlike a report's own
    fields, which could in principle be recomputed and drift."""
    params = {"persona": "p", "network": "entities", "kind": "degree", "bootstrap": 5}
    prov = make_provenance(
        method="degree",
        parameters=params,
        chapters=(9,),
        seed=3,
        null_model="x",
        null_model_samples=5,
    )
    assert prov.parameters == params
    assert prov.parameters is not params  # make_provenance copies rather than aliasing
    assert prov.seed == 3
    assert prov.null_model == "x"
    assert prov.null_model_samples == 5
    assert prov.chapters == (9,)


def test_make_provenance_defaults_seed_and_null_model_to_none() -> None:
    prov = make_provenance(method="walks", parameters={}, chapters=(2, 11))
    assert prov.seed is None
    assert prov.null_model is None
    assert prov.null_model_samples is None
    assert prov.snapshot_commit is None


def test_make_provenance_takes_a_fixed_clock_for_a_test_that_wants_one() -> None:
    prov = make_provenance(
        method="degree", parameters={}, chapters=(9,), generated_at="2026-01-01T00:00:00Z"
    )
    assert prov.generated_at == "2026-01-01T00:00:00Z"


# ------------------------------------------------------------------------------- render_provenance


def test_render_provenance_prints_every_field_the_dataclass_holds() -> None:
    prov = Provenance(
        method="degree",
        parameters={"persona": "p", "bootstrap": 5},
        seed=3,
        tool_version="9.9.9",
        null_model="power-law bootstrap",
        null_model_samples=200,
        chapters=(9,),
        generated_at="2026-01-01T00:00:00Z",
        snapshot_commit="abc1234",
    )
    lines = render_provenance(prov)
    text = "\n".join(lines)
    assert text.startswith("## Provenance\n")
    assert "- method: `degree` (Atlas ch. 9)" in text
    assert '"bootstrap": 5' in text
    assert '"persona": "p"' in text
    assert "- seed: 3" in text
    assert "- null model: power-law bootstrap (200 samples)" in text
    assert "- tool version: graphrag 9.9.9" in text
    assert "- generated: 2026-01-01T00:00:00Z" in text
    assert "- snapshot commit: abc1234" in text


def test_render_provenance_names_every_chapter_it_touched() -> None:
    prov = make_provenance(method="analyze", parameters={}, chapters=(9, 12, 35))
    text = "\n".join(render_provenance(prov))
    assert "Atlas ch. 9, 12, 35" in text


def test_render_provenance_says_not_fixed_and_unknown_rather_than_none() -> None:
    """A bare ``None`` in the middle of a markdown sentence reads as a bug; the report always
    spells out what the missing value means."""
    prov = make_provenance(method="export", parameters={}, chapters=(6,))
    text = "\n".join(render_provenance(prov))
    assert "- seed: not fixed" in text
    assert "- null model: none for this report's headline claim" in text
    assert "- snapshot commit: unknown (no source commit recorded for this persona)" in text


def test_render_provenance_parameters_are_sorted_regardless_of_build_order() -> None:
    """A CLI command and its MCP tool rarely declare the same options in the same order; the
    printed parameters must agree even so, or otherwise-identical markdown would disagree on
    this line alone (this is what broke `sna_analyze`/`sna_sample` parity before it was fixed)."""
    a = make_provenance(method="x", parameters={"b": 1, "a": 2}, chapters=(1,))
    b = make_provenance(method="x", parameters={"a": 2, "b": 1}, chapters=(1,))
    a_line = next(line for line in render_provenance(a) if line.startswith("- parameters"))
    b_line = next(line for line in render_provenance(b) if line.startswith("- parameters"))
    assert a_line == b_line


def test_render_provenance_parameters_are_valid_json() -> None:
    prov = make_provenance(
        method="degree", parameters={"where": None, "types": ["a", "b"]}, chapters=(9,)
    )
    line = next(line for line in render_provenance(prov) if line.startswith("- parameters"))
    blob = line.split("`", 1)[1].rsplit("`", 1)[0]
    assert json.loads(blob) == {"where": None, "types": ["a", "b"]}


# ------------------------------------------------------------------------------- provenance_payload


def test_provenance_payload_round_trips_every_field() -> None:
    prov = make_provenance(
        method="roles",
        parameters={"persona": "p"},
        chapters=(15,),
        seed=7,
        null_model=None,
        snapshot_commit="deadbeef",
    )
    payload = provenance_payload(prov)
    assert payload == {
        "method": "roles",
        "parameters": {"persona": "p"},
        "seed": 7,
        "tool_version": __version__,
        "null_model": None,
        "null_model_samples": None,
        "chapters": [15],
        "generated_at": prov.generated_at,
        "snapshot_commit": "deadbeef",
    }


# --------------------------------------------------------------------------------- chapter_citation


@pytest.mark.parametrize(
    ("chapter", "expected"),
    [
        (9, "Chapter 9 -- Degree (pp. 134-153)."),
        (27, "Chapter 27 -- Network Backboning (pp. 381-394)."),
        (56, "Chapter 56 -- Bibliography (pp. 803-916)."),
    ],
)
def test_chapter_citation_matches_the_books_own_outline(chapter: int, expected: str) -> None:
    """The known answer: the book's own outline (`scripts/atlas_chapter.py toc`), read once
    while building :data:`CHAPTER_TITLES` and checked again here rather than trusted."""
    assert chapter_citation(chapter) == expected


def test_chapter_citation_reports_an_unknown_chapter_rather_than_raising() -> None:
    assert chapter_citation(999) == "Chapter 999 (page range not on file)."


def test_every_chapter_title_has_a_page_range_that_does_not_run_backwards() -> None:
    for chapter, (title, first, last) in CHAPTER_TITLES.items():
        assert 1 <= chapter <= 56
        assert title
        assert first <= last


# -------------------------------------------------------------------------------- render_references


def test_render_references_prints_the_book_once_then_each_chapter_sorted_and_deduped() -> None:
    lines = render_references([27, 9, 27])
    text = "\n".join(lines)
    assert text.startswith("## References\n")
    assert text.count(BOOK_CITATION) == 1
    order = [chapter_citation(9), chapter_citation(27)]
    assert order[0] in text
    assert order[1] in text
    assert text.index(order[0]) < text.index(order[1])
    assert text.count("Chapter 27") == 1


# ------------------------------------------------------------------------------ COMMAND_CHAPTERS


def test_every_command_chapters_entry_points_at_a_known_chapter() -> None:
    for command, chapters in COMMAND_CHAPTERS.items():
        assert chapters, f"{command} lists no chapters"
        for chapter in chapters:
            assert chapter in CHAPTER_TITLES, f"{command} cites unknown chapter {chapter}"


def test_command_chapters_covers_every_report_command() -> None:
    """`guide` and `cache *` are not reports (no method, no chapter, no payload to attach a
    provenance block to) and are the only `sna_app` commands not in this table -- the same
    exclusion `test_mcp_sna.py::test_tool_registry_covers_every_sna_command` makes, for the
    same reason."""
    cli_commands = typer.main.get_command(sna_app).commands  # type: ignore[attr-defined]
    report_commands = {
        name for name, command in cli_commands.items() if not hasattr(command, "commands")
    } - {"guide"}
    assert set(COMMAND_CHAPTERS) == report_commands


# ---------------------------------------------------------------------------- snapshot_commit_for


def _manifest(persona_id: str, *, source_commit: str | None) -> SnapshotManifest:
    return SnapshotManifest(
        persona_id=persona_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        embedding_model="hash-test",
        embedding_dim=8,
        document_count=1,
        chunk_count=1,
        source_commit=source_commit,
    )


def _write_manifest(snapshots_dir: Path, manifest: SnapshotManifest) -> None:
    target = snap.snapshot_dir(snapshots_dir, manifest.persona_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / snap.MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def test_snapshot_commit_for_is_none_without_a_manifest(settings: Settings) -> None:
    assert snapshot_commit_for(settings, "no-such-persona") is None


def test_snapshot_commit_for_reads_the_manifests_source_commit(settings: Settings) -> None:
    _write_manifest(settings.snapshots_dir, _manifest("p1", source_commit="abc123"))
    assert snapshot_commit_for(settings, "p1") == "abc123"


def test_snapshot_commit_for_is_none_when_the_manifest_never_recorded_one(
    settings: Settings,
) -> None:
    _write_manifest(settings.snapshots_dir, _manifest("p1", source_commit=None))
    assert snapshot_commit_for(settings, "p1") is None
