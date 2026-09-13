"""Host-side repository introspection, exercised against a project built in ``tmp_path``."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from graphrag.hooks import project

PERSONA_YAML = """\
id: handbook
name: Handbook
description: >-
  A short persona used by the tests: two lines of folded text
  that collapse into one paragraph.
role_prompt: |
  Line one.
  Line two.
voice:
  - Direct and practical; leads with the recommendation.
sdlc_stages: [design, retrospective]
sources:
  - id: notes            # a trailing comment
    kind: local
    path: notes
    loader: documents
    glob: "**/*.md"
    description: Field notes.
  - id: talks
    path: talks
    loader: transcripts
    glob: "*/transcript.md"
retrieval:
  top_k: 8
  mode: hybrid
tags: [alpha, beta]
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A miniature graph-rag checkout: one persona, one snapshot, a small raw corpus."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")
    persona_dir = tmp_path / "personas" / "handbook"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona.yaml").write_text(PERSONA_YAML, encoding="utf-8")

    notes = tmp_path / "data" / "raw" / "handbook" / "notes"
    (notes / "deep").mkdir(parents=True)
    (notes / "one.md").write_text("# One\n", encoding="utf-8")
    (notes / "deep" / "Two Words.md").write_text("# Two\n", encoding="utf-8")
    (notes / "ignored.json").write_text("{}", encoding="utf-8")
    talks = tmp_path / "data" / "raw" / "handbook" / "talks" / "opening"
    talks.mkdir(parents=True)
    (talks / "transcript.md").write_text("# Opening\n", encoding="utf-8")

    snapshot = tmp_path / "data" / "snapshots" / "handbook"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "persona_id": "handbook",
                "created_at": "2026-09-09T20:57:22.830608Z",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "document_count": 2,
                "chunk_count": 9,
                "entity_count": 4,
                "sources": [{"id": "notes", "path": "notes", "loader": "documents"}],
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(snapshot / "documents.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "handbook:notes:one", "title": "One"}) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"id": "handbook:notes:deep-two-words", "title": "Two"}) + "\n")
    return tmp_path


@pytest.fixture
def handbook(root: Path) -> project.PersonaInfo:
    return project.load_persona(root, "handbook")


# ----------------------------------------------------------------------------- root


def test_find_root_walks_up_from_a_subdirectory(root: Path) -> None:
    assert project.find_root(root / "data" / "raw" / "handbook" / "notes") == root


def test_find_root_prefers_the_claude_project_dir(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(root))
    assert project.find_root(elsewhere) == root


def test_find_root_ignores_a_bogus_claude_project_dir(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "nowhere"))
    assert project.find_root(root / "personas") == root


def test_find_root_returns_none_outside_a_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    stray = tmp_path / "stray"
    stray.mkdir()
    assert project.find_root(stray) is None


def test_cache_dir_is_created_under_data_cache(root: Path) -> None:
    path = project.cache_dir(root)
    assert path == root / "data" / "cache" / "hooks"
    assert path.is_dir()


# ----------------------------------------------------------------------------- personas


def test_persona_ids(root: Path) -> None:
    assert project.persona_ids(root) == ["handbook"]


def test_load_persona_reads_the_keys_the_hooks_need(handbook: project.PersonaInfo) -> None:
    assert handbook.id == "handbook"
    assert handbook.name == "Handbook"
    assert handbook.tags == ("alpha", "beta")
    assert handbook.description.startswith("A short persona used by the tests:")
    assert "\n" not in handbook.description
    assert [s.id for s in handbook.sources] == ["notes", "talks"]


def test_source_defaults_are_filled_in(handbook: project.PersonaInfo) -> None:
    notes, talks = handbook.sources
    assert (notes.kind, notes.loader, notes.glob) == ("local", "documents", "**/*.md")
    assert (talks.kind, talks.loader, talks.glob) == ("local", "transcripts", "*/transcript.md")
    assert notes.description == "Field notes."
    assert notes.url is None


def test_load_persona_missing_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        project.load_persona(root, "absent")


def test_load_personas_skips_unreadable_ones(root: Path) -> None:
    broken = root / "personas" / "broken"
    broken.mkdir()
    (broken / "persona.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    loaded = project.load_personas(root)
    assert "handbook" in loaded
    assert set(loaded) <= {"handbook", "broken"}


# ----------------------------------------------------------------------------- yaml fallback


def test_simple_yaml_parser_matches_the_persona_file() -> None:
    data = project.parse_simple_yaml(PERSONA_YAML)
    assert data["id"] == "handbook"
    assert data["tags"] == ["alpha", "beta"]
    assert data["sdlc_stages"] == ["design", "retrospective"]
    assert data["voice"] == ["Direct and practical; leads with the recommendation."]
    assert data["retrieval"] == {"top_k": 8, "mode": "hybrid"}
    assert data["sources"][0]["id"] == "notes"
    assert data["sources"][1]["glob"] == "*/transcript.md"


def test_simple_yaml_folded_and_literal_block_scalars() -> None:
    data = project.parse_simple_yaml(PERSONA_YAML)
    assert data["description"] == (
        "A short persona used by the tests: two lines of folded text "
        "that collapse into one paragraph."
    )
    assert data["role_prompt"] == "Line one.\nLine two.\n"


def test_simple_yaml_handles_comments_blanks_and_scalar_types() -> None:
    data = project.parse_simple_yaml(
        "# leading comment\n\nname: Demo  # trailing\nempty:\nflag: true\ncount: 3\n"
    )
    assert data == {"name": "Demo", "empty": None, "flag": True, "count": 3}


def test_persona_info_survives_a_file_without_any_sources(root: Path) -> None:
    bare = root / "personas" / "bare"
    bare.mkdir()
    (bare / "persona.yaml").write_text("id: bare\n", encoding="utf-8")
    info = project.load_persona(root, "bare")
    assert info == project.PersonaInfo(id="bare", name="bare")


# ----------------------------------------------------------------------------- snapshots


def test_load_manifests(root: Path) -> None:
    manifests = project.load_manifests(root)
    manifest = manifests["handbook"]
    assert manifest.document_count == 2
    assert manifest.chunk_count == 9
    assert manifest.entity_count == 4
    assert manifest.embedding_model == "BAAI/bge-small-en-v1.5"
    assert manifest.created_date == "2026-09-09"
    assert [s.id for s in manifest.sources] == ["notes"]
    assert manifest.path == root / "data" / "snapshots" / "handbook"


def test_load_manifests_skips_a_corrupt_file(root: Path) -> None:
    broken = root / "data" / "snapshots" / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    assert set(project.load_manifests(root)) == {"handbook"}


def test_load_manifests_without_a_snapshot_directory(tmp_path: Path) -> None:
    assert project.load_manifests(tmp_path) == {}


def test_snapshot_doc_ids(root: Path) -> None:
    assert project.snapshot_doc_ids(root, "handbook") == {
        "handbook:notes:one",
        "handbook:notes:deep-two-words",
    }


def test_snapshot_doc_ids_without_a_snapshot(root: Path) -> None:
    assert project.snapshot_doc_ids(root, "absent") == set()


# ----------------------------------------------------------------------------- raw corpus


def test_raw_files_applies_the_glob_and_the_suffix_filter(
    root: Path, handbook: project.PersonaInfo
) -> None:
    notes, talks = handbook.sources
    assert [p.name for p in project.raw_files(root, handbook, notes)] == ["Two Words.md", "one.md"]
    assert [p.name for p in project.raw_files(root, handbook, talks)] == ["transcript.md"]


def test_raw_files_for_a_missing_source_directory(
    root: Path, handbook: project.PersonaInfo
) -> None:
    absent = project.SourceInfo(id="absent", path="absent")
    assert project.raw_files(root, handbook, absent) == []


def test_expected_doc_id_matches_the_documents_loader(
    root: Path, handbook: project.PersonaInfo
) -> None:
    notes = handbook.sources[0]
    ids = {
        project.expected_doc_id(root, handbook, notes, p)
        for p in project.raw_files(root, handbook, notes)
    }
    assert ids == {"handbook:notes:one", "handbook:notes:deep-two-words"}
    assert ids == project.snapshot_doc_ids(root, "handbook")


def test_expected_doc_id_accepts_a_persona_id_string(
    root: Path, handbook: project.PersonaInfo
) -> None:
    notes = handbook.sources[0]
    path = root / "data" / "raw" / "handbook" / "notes" / "one.md"
    assert project.expected_doc_id(root, "handbook", notes, path) == "handbook:notes:one"


def test_raw_root_is_the_src_make_ingest_uses(root: Path) -> None:
    assert project.raw_root(root, "handbook") == root / "data" / "raw" / "handbook"


def test_rel_path(root: Path) -> None:
    assert project.rel_path(root, root / "data" / "raw" / "x.md") == "data/raw/x.md"
    assert project.rel_path(root, Path("/elsewhere/x.md")) == "/elsewhere/x.md"


def test_newest_raw_files_are_ordered_newest_first(root: Path) -> None:
    raw = root / "data" / "raw" / "handbook"
    ages = {
        raw / "notes" / "one.md": 1000.0,
        raw / "notes" / "deep" / "Two Words.md": 3000.0,
        raw / "talks" / "opening" / "transcript.md": 2000.0,
    }
    for path, mtime in ages.items():
        os.utime(path, (mtime, mtime))

    newest = project.newest_raw_files(root, 2)
    assert [p.name for p, _ in newest] == ["Two Words.md", "transcript.md"]
    assert [m for _, m in newest] == [3000.0, 2000.0]


def test_newest_raw_files_skips_non_corpus_files_and_hidden_directories(root: Path) -> None:
    hidden = root / "data" / "raw" / "handbook" / ".cache"
    hidden.mkdir()
    (hidden / "recent.md").write_text("x", encoding="utf-8")
    names = {p.name for p, _ in project.newest_raw_files(root, 20)}
    assert names == {"one.md", "Two Words.md", "transcript.md"}


def test_newest_raw_files_with_nothing_to_show(root: Path) -> None:
    empty = root / "empty-checkout"
    empty.mkdir()
    assert project.newest_raw_files(empty, 5) == []
    assert project.newest_raw_files(root, 0) == []
