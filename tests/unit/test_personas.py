from pathlib import Path

import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec
from graphrag.personas.brief import build_brief
from graphrag.personas.registry import PersonaRegistry
from graphrag.personas.skill_export import export_persona_skill
from graphrag.pipeline import IngestReport


def test_registry_roundtrip(registry: PersonaRegistry, persona: PersonaSpec) -> None:
    assert [p.id for p in registry.all()] == ["test-docs", "test-pm"]
    assert registry.get("test-pm") == persona
    with pytest.raises(KeyError, match="known: test-docs, test-pm"):
        registry.get("nope")


def test_registry_scaffold_and_recommend(registry: PersonaRegistry) -> None:
    spec = registry.scaffold("Support Engineer", "Answers from our runbooks.")
    assert spec.id == "support-engineer"
    assert (registry.directory / spec.id / "README.md").read_text().startswith("# Support Engineer")
    with pytest.raises(FileExistsError):
        registry.scaffold("Support Engineer")
    assert [p.id for p in registry.recommend("Requirements")] == [
        "support-engineer",
        "test-pm",
    ]
    assert [p.id for p in registry.recommend("compliance")] == ["test-docs"]
    assert registry.recommend("unknown-stage") == []


def test_registry_infers_id_from_folder(tmp_path: Path) -> None:
    (tmp_path / "folder-id").mkdir()
    (tmp_path / "folder-id" / "persona.yaml").write_text("name: Folder\n")
    assert PersonaRegistry(tmp_path).all()[0].id == "folder-id"


def test_brief_and_skill(
    ingested: IngestReport, memory_store: InMemoryGraphStore, persona: PersonaSpec, tmp_path: Path
) -> None:
    brief = build_brief(persona, memory_store)
    assert "# Test Product Leader" in brief
    assert "Indexed: 3 documents" in brief
    assert "retention" in brief
    assert "Lenny Rachitsky" in brief
    path = export_persona_skill(persona, brief, tmp_path / "skills")
    text = path.read_text()
    assert path.name == "SKILL.md" and path.parent.name == "persona-test-pm"
    assert text.startswith("---\nname: persona-test-pm")
    assert 'persona_brief("test-pm")' in text
