"""Persona definitions live in ``personas/<id>/persona.yaml`` and are the unit of sharing."""

from __future__ import annotations

from pathlib import Path

import yaml

from graphrag.models import PersonaSpec, slugify

PERSONA_FILE = "persona.yaml"

SDLC_STAGES = [
    "discovery",
    "requirements",
    "prioritization",
    "design",
    "implementation",
    "testing",
    "release",
    "operations",
    "support",
    "compliance",
    "retrospective",
]


class PersonaRegistry:
    def __init__(self, personas_dir: Path) -> None:
        self._dir = personas_dir

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, persona_id: str) -> Path:
        return self._dir / persona_id / PERSONA_FILE

    def all(self) -> list[PersonaSpec]:
        if not self._dir.exists():
            return []
        specs: list[PersonaSpec] = []
        for path in sorted(self._dir.glob(f"*/{PERSONA_FILE}")):
            specs.append(self._load(path))
        return specs

    def get(self, persona_id: str) -> PersonaSpec:
        path = self.path_for(persona_id)
        if not path.exists():
            known = ", ".join(p.id for p in self.all()) or "(none)"
            msg = f"unknown persona {persona_id!r}; known: {known}"
            raise KeyError(msg)
        return self._load(path)

    def save(self, spec: PersonaSpec) -> Path:
        path = self.path_for(spec.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = spec.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    def scaffold(
        self, name: str, description: str = "", persona_id: str | None = None
    ) -> PersonaSpec:
        pid = persona_id or slugify(name)
        if self.path_for(pid).exists():
            msg = f"persona {pid!r} already exists"
            raise FileExistsError(msg)
        spec = PersonaSpec(
            id=pid,
            name=name,
            description=description,
            role_prompt=(
                f"You are {name}. Ground every answer in the retrieved sources and cite them. "
                "When the sources are silent, say so."
            ),
            sdlc_stages=["requirements"],
        )
        self.save(spec)
        readme = self.path_for(pid).parent / "README.md"
        readme.write_text(
            f"# {name}\n\n{description}\n\n"
            "## Adding sources\n\n"
            "1. Put documents (`.md`, `.txt`, `.pdf`) under `data/raw/" + pid + "/<source-id>/`.\n"
            "2. Add a `sources:` entry to `persona.yaml` (loader `documents` or `transcripts`).\n"
            f"3. `make ingest PERSONA={pid} SRC=data/raw/{pid}` "
            f"then commit `data/snapshots/{pid}/`.\n",
            encoding="utf-8",
        )
        return spec

    def recommend(self, stage: str) -> list[PersonaSpec]:
        wanted = slugify(stage)
        return [p for p in self.all() if wanted in {slugify(s) for s in p.sdlc_stages}]

    @staticmethod
    def _load(path: Path) -> PersonaSpec:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "id" not in data:
            data["id"] = path.parent.name
        return PersonaSpec.model_validate(data)
