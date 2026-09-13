"""The PostToolUse capture contract, exercised against a fake project and stdin payloads."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from graphrag.hooks import capture_contract

PERSONA_YAML = """\
id: demo-persona
name: Demo Persona
sources:
  - id: notes
    path: notes
    loader: documents
    glob: "**/*.md"
  - id: talks
    path: talks
    loader: transcripts
    glob: "*/transcript.md"
"""

ALIASES_YAML = """\
aliases:
  Ledger Console: [ledger console, LedgerConsole]
"""

FACETS_YAML = """\
facets:
  handover: Moving work or knowledge from one person to another.
"""

THREAD = """\
---
title: A captured thread
---

**quill-maker** wrote on 2025-02-03:
The first week decides how the next year goes, and nobody writes that down.

**ledger-ann** wrote on 2025-02-06:
Most of what goes wrong in a first month is access to accounts nobody created.
"""

PROSE = """\
---
title: An ordinary note
---

A paragraph of plain prose with nothing post-shaped about it, and no declared vocabulary.
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """One persona with a ``documents`` source and a ``transcripts`` source, and no sidecars."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "graphrag"\n', encoding="utf-8")
    persona_dir = tmp_path / "personas" / "demo-persona"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona.yaml").write_text(PERSONA_YAML, encoding="utf-8")
    (tmp_path / "data" / "raw" / "demo-persona" / "notes").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "demo-persona" / "talks").mkdir(parents=True)
    return tmp_path


def write_note(root: Path, name: str, body: str = PROSE) -> Path:
    path = root / "data" / "raw" / "demo-persona" / "notes" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def payload(path: Path, root: Path, *, tool_name: str = "Write", **extra: Any) -> dict[str, Any]:
    return {
        "session_id": "x",
        "cwd": str(root),
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {"file_path": str(path)},
        "tool_response": {"filePath": str(path), "type": "create"},
        **extra,
    }


# ----------------------------------------------------------------------------- resolution


def test_a_note_under_data_raw_resolves_to_its_document_id(root: Path) -> None:
    path = write_note(root, "2026-01-02-a-note.md")

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None
    assert target.doc_id == "demo-persona:notes:2026-01-02-a-note"
    assert (target.persona_id, target.source_id) == ("demo-persona", "notes")


def test_a_transcript_takes_the_id_of_its_folder(root: Path) -> None:
    path = root / "data" / "raw" / "demo-persona" / "talks" / "an-episode" / "transcript.md"
    path.parent.mkdir(parents=True)
    path.write_text(PROSE, encoding="utf-8")

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None and target.doc_id == "demo-persona:talks:an-episode"


def test_a_path_outside_data_raw_resolves_to_nothing(root: Path) -> None:
    path = root / "src" / "graphrag" / "cli.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n", encoding="utf-8")

    assert capture_contract.resolve_target(payload(path, root)) is None


def test_a_sidecar_json_under_data_is_not_a_corpus_document(root: Path) -> None:
    path = root / "data" / "enrichment" / "demo-persona" / "notes" / "a-note.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    assert capture_contract.resolve_target(payload(path, root)) is None


def test_a_suffix_no_loader_reads_resolves_to_nothing(root: Path) -> None:
    path = root / "data" / "raw" / "demo-persona" / "notes" / "sheet.csv"
    path.write_text("a,b\n", encoding="utf-8")

    assert capture_contract.resolve_target(payload(path, root)) is None


def test_a_persona_directory_that_does_not_exist_resolves_to_nothing(root: Path) -> None:
    path = root / "data" / "raw" / "not-a-persona" / "notes" / "x.md"
    path.parent.mkdir(parents=True)
    path.write_text(PROSE, encoding="utf-8")

    assert capture_contract.resolve_target(payload(path, root)) is None


def test_a_path_under_no_source_of_the_persona_resolves_to_nothing(root: Path) -> None:
    """The persona exists but the file sits beside its sources, not inside one."""
    path = root / "data" / "raw" / "demo-persona" / "stray.md"
    path.write_text(PROSE, encoding="utf-8")

    assert capture_contract.resolve_target(payload(path, root)) is None


@pytest.mark.parametrize("tool_name", ["Read", "Bash", "Grep"])
def test_a_tool_that_does_not_write_is_ignored(root: Path, tool_name: str) -> None:
    path = write_note(root, "a-note.md")

    assert capture_contract.resolve_target(payload(path, root, tool_name=tool_name)) is None


@pytest.mark.parametrize("tool_name", ["Write", "Edit", "MultiEdit"])
def test_every_writing_tool_is_covered(root: Path, tool_name: str) -> None:
    path = write_note(root, "a-note.md")

    assert capture_contract.resolve_target(payload(path, root, tool_name=tool_name)) is not None


# ----------------------------------------------------------------------------- which layers


def test_extraction_is_expected_of_every_document(root: Path) -> None:
    target = capture_contract.resolve_target(payload(write_note(root, "a-note.md"), root))

    assert target is not None and target.layers == ("extraction",)


def test_attributed_posts_make_the_attribution_layer_expected(root: Path) -> None:
    path = write_note(root, "a-thread.md", THREAD)

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None and "attribution" in target.layers
    assert "2 attributed posts" in target.reasons["attribution"]


def test_declared_vocabulary_makes_the_annotation_layer_expected(root: Path) -> None:
    (root / "personas" / "demo-persona" / "aliases.yaml").write_text(ALIASES_YAML, encoding="utf-8")
    path = write_note(root, "a-note.md", PROSE.replace("plain prose", "the Ledger Console"))

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None and "annotation" in target.layers
    assert "ledger console" in target.reasons["annotation"]


def test_a_facet_word_also_counts_as_declared_vocabulary(root: Path) -> None:
    (root / "personas" / "demo-persona" / "facets.yaml").write_text(FACETS_YAML, encoding="utf-8")
    path = write_note(root, "a-note.md", PROSE.replace("plain prose", "a handover problem"))

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None and "annotation" in target.layers


def test_a_persona_with_no_declared_vocabulary_is_never_told_to_annotate(root: Path) -> None:
    target = capture_contract.resolve_target(payload(write_note(root, "a-note.md"), root))

    assert target is not None and "annotation" not in target.layers
    assert "declares no alias or facet vocabulary" in target.reasons["annotation"]


def test_one_post_is_not_enough_to_call_a_document_attributed(root: Path) -> None:
    body = PROSE + "\n**someone** wrote on 2025-01-01:\nA single quoted line.\n"
    path = write_note(root, "a-note.md", body)

    target = capture_contract.resolve_target(payload(path, root))

    assert target is not None and "attribution" not in target.layers


# ----------------------------------------------------------------------------- the message


def test_the_context_names_the_id_the_sidecars_and_the_check(root: Path) -> None:
    path = write_note(root, "a-thread.md", THREAD)

    context = capture_contract.build_context(payload(path, root))

    assert context is not None
    assert "demo-persona:notes:a-thread" in context
    assert "data/enrichment/demo-persona/notes/a-thread.json" in context
    assert "data/attribution/demo-persona/notes/a-thread.json" in context
    assert "extraction, expected" in context
    assert "attribution, expected" in context
    assert "annotation, not indicated" in context
    assert "graphrag layers check demo-persona --file data/raw/demo-persona/notes/a-thread.md" in (
        context
    )
    assert ".claude/skills/graph-rag-capture/SKILL.md" in context


def test_a_sidecar_already_on_disk_is_reported_as_such(root: Path) -> None:
    path = write_note(root, "a-note.md")
    sidecar = root / "data" / "enrichment" / "demo-persona" / "notes" / "a-note.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"doc_id": "demo-persona:notes:a-note"}), encoding="utf-8")

    context = capture_contract.build_context(payload(path, root))

    assert context is not None and "already on disk" in context


def test_the_output_is_the_posttooluse_additional_context_shape(root: Path) -> None:
    path = write_note(root, "a-note.md")

    output = capture_contract.build_output(payload(path, root))

    assert output is not None
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == "PostToolUse"
    assert specific["additionalContext"].startswith("graph-rag capture contract.")
    assert set(output) == {"hookSpecificOutput"}


# ----------------------------------------------------------------------------- main()


def _run_main(
    monkeypatch: pytest.MonkeyPatch, data: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(data)))
    assert capture_contract.main() == 0
    return capsys.readouterr().out


def test_main_prints_the_contract_for_a_corpus_document(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = _run_main(monkeypatch, payload(write_note(root, "a-note.md"), root), capsys)

    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "PostToolUse"


def test_main_is_silent_for_a_path_outside_the_corpus(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    path = root / "README.md"
    path.write_text("# readme\n", encoding="utf-8")

    assert _run_main(monkeypatch, payload(path, root), capsys) == ""


def test_main_never_raises_on_garbage_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    assert capture_contract.main() == 0
    assert capsys.readouterr().out == ""


def test_main_is_silent_on_empty_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert capture_contract.main() == 0
    assert capsys.readouterr().out == ""


def test_main_is_silent_when_the_payload_has_no_file_path(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = {"tool_name": "Write", "cwd": str(root), "tool_input": {}}
    assert _run_main(monkeypatch, data, capsys) == ""


# ----------------------------------------------------------------------------- wrapper script


def test_wrapper_script_is_silent_for_a_path_outside_the_corpus() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / ".claude" / "hooks" / "capture-contract.sh"
    data = json.dumps(
        {
            "tool_name": "Write",
            "cwd": str(repo_root),
            "tool_input": {"file_path": str(repo_root / "README.md")},
        }
    )
    started = time.monotonic()
    result = subprocess.run(
        [str(script)],
        input=data,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_PROJECT_DIR": str(repo_root), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert time.monotonic() - started < 3.0


def test_a_hostile_vocabulary_term_is_flattened_before_it_reaches_the_context(root: Path) -> None:
    """Alias names come off disk and land in text a model reads, so they are flattened.

    The document names the entity in the same hidden-character form the alias file declares, so
    the term the reason echoes is the hostile one rather than a harmless spelling of it. Without
    the flattening the characters still reach the line, as the escape sequences ``repr`` prints.
    """
    hostile = "Ledger\u200bConsole\u202e"
    (root / "personas" / "demo-persona" / "aliases.yaml").write_text(
        f'aliases:\n  "{hostile}": [ledgerconsole]\n', encoding="utf-8"
    )
    path = write_note(root, "a-note.md", PROSE.replace("plain prose", f"the {hostile}"))

    context = capture_contract.build_context(payload(path, root))

    assert context is not None
    assert "annotation, expected" in context
    assert "such as 'ledgerconsole'" in context
    assert "\u200b" not in context and "\u202e" not in context
    assert "\\u200b" not in context and "\\u202e" not in context
    assert len(context) < 10_000  # the harness caps injected context at 10k characters
