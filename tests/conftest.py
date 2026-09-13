# ruff: noqa: E501 - synthetic corpus text is clearer on one line
"""Shared fixtures. Everything external is replaced by an injected fake, never patched."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from graphrag.app import AppContext
from graphrag.config import EmbeddingSettings, Neo4jSettings, Settings
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec, RetrievalConfig, SourceSpec
from graphrag.personas.registry import PersonaRegistry
from graphrag.pipeline import IngestPipeline, IngestReport
from graphrag.retrieve.search import Retriever

DIM = 64

EPISODES = {
    "ada-north": {
        "guest": "Ada North",
        "title": "Finding product-market fit before you scale | Ada North",
        "video_id": "abc123",
        "publish_date": "2025-01-10",
        "keywords": ["product-market fit", "retention", "onboarding"],
        "turns": [
            (
                "Ada North",
                "00:00:00",
                "Product-market fit is when retention curves flatten instead of decaying to zero. Until the retention curve flattens you do not have a business, you have a leaky bucket.",
            ),
            ("Lenny Rachitsky", "00:00:41", "How do you know when the curve has flattened enough?"),
            (
                "Ada North",
                "00:01:02",
                "Look at cohorts by signup month. If month-six retention holds above forty percent for a consumer product, you are close. Onboarding is the lever that moves early retention most, so fix onboarding before you spend on acquisition.",
            ),
            (
                "Lenny Rachitsky",
                "00:02:15",
                "What about pricing while you are still searching for fit?",
            ),
            (
                "Ada North",
                "00:02:30",
                "Charge early. Pricing is a signal of value, and free users lie to you about what they want.",
            ),
        ],
    },
    "ben-oduya": {
        "guest": "Ben Oduya",
        "title": "Running a roadmap review that ships | Ben Oduya",
        "video_id": "def456",
        "publish_date": "2025-03-02",
        "keywords": ["roadmap", "prioritization", "retention"],
        "turns": [
            (
                "Ben Oduya",
                "00:00:00",
                "A roadmap review should be a prioritization argument, not a status meeting. Bring the three bets you would kill and the one you would double.",
            ),
            (
                "Lenny Rachitsky",
                "00:00:50",
                "How do you handle a retention bet versus a growth bet?",
            ),
            (
                "Ben Oduya",
                "00:01:10",
                "Retention bets compound; growth bets pay once. Score each bet on reach, confidence and effort, then argue about confidence only.",
            ),
        ],
    },
    "cleo-vance": {
        "guest": "Cleo Vance",
        "title": "Design reviews that engineers enjoy | Cleo Vance",
        "video_id": "ghi789",
        "publish_date": "2024-11-20",
        "keywords": ["design", "communication", "onboarding"],
        "turns": [
            (
                "Cleo Vance",
                "00:00:00",
                "The best design review starts with the user problem written in one sentence. If the sentence has an 'and' in it, you have two problems and two reviews.",
            ),
            ("Lenny Rachitsky", "00:00:33", "How long should the review be?"),
            (
                "Cleo Vance",
                "00:00:40",
                "Thirty minutes. Anything longer means the onboarding of reviewers into the context failed before the meeting.",
            ),
        ],
    },
}


"""A captured discussion thread: four posts by three handles, in one prose document.

The loader that reads it knows nothing about posts, so the graph holds it with no speakers at
all -- which is exactly the gap `graphrag attribution-import` fills. The bodies are long enough
that the chunker splits the thread across more than one passage, so an anchor has somewhere
wrong to land.
"""
THREAD_POSTS = [
    (
        "quill-maker",
        "2025-02-03",
        "I ran onboarding for three years and the part nobody warns you about is that the first "
        "week decides how the next year goes. We tried a long written handbook and nobody read "
        "it. What actually moved the needle was a single page listing the five things a new "
        "person has to be able to do by Friday, and then getting out of the way. If you cannot "
        "write that page, you do not understand the job well enough to hire for it yet. Write "
        "the page first, hire second, and revise the page after every single start date.",
    ),
    (
        "north-by",
        "2025-02-03",
        "Respectfully I think the one page idea is a trap. We used it for a year and all it did "
        "was make people optimise for looking finished by Friday. The pairing model worked far "
        "better for us. Every new starter sits with someone who joined six months earlier, "
        "because that person still remembers which parts were confusing and has not yet "
        "forgotten what the jargon means. It costs the pair about four hours a week for a month "
        "and it has halved our time to a first real contribution, which is the only number I "
        "actually care about.",
    ),
    (
        "quill-maker",
        "2025-02-04",
        "That is fair and I do not think the two are opposed. We kept the page and added the "
        "pairing on top of it once the team was big enough to spare the hours. The page is what "
        "stops the pairing from turning into folklore, where each new person learns a slightly "
        "different version of how things are done. Write down the parts that must not drift, and "
        "let the pair handle everything else. The failure mode I still see most often is a "
        "manager who delegates the whole thing and then wonders why two hires on one team work "
        "completely differently.",
    ),
    (
        "ledger-ann",
        "2025-02-06",
        "Late to this thread but I want to add the boring part that nobody mentions. Most of what "
        "goes wrong in a first month is access. Accounts that were never created, a repository "
        "nobody thought to grant, a calendar invite to the wrong address. We now run a checklist "
        "the day before someone starts and we treat a missing account as an incident rather than "
        "an inconvenience. It is unglamorous and it removed more frustration than any of the "
        "mentoring schemes we tried before it.",
    ),
]


def write_thread(root: Path) -> Path:
    """Write the sample thread under ``root/threads``. Its own folder, so the ``docs`` fixtures
    keep the document count their tests assert on."""
    folder = root / "threads"
    folder.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(f"**{who}** wrote on {when}:\n{text}" for who, when, text in THREAD_POSTS)
    path = folder / "onboarding-thread.md"
    path.write_text(
        "---\ntitle: How do you handle onboarding for a new hire\n---\n\n"
        "# How do you handle onboarding for a new hire\n\n" + body + "\n",
        encoding="utf-8",
    )
    return path


def write_sample_corpus(root: Path) -> Path:
    """Write three small synthetic transcripts in the Lenny's-archive layout under ``root``."""
    for slug, ep in EPISODES.items():
        folder = root / "episodes" / slug
        folder.mkdir(parents=True, exist_ok=True)
        keywords = "\n".join(f"- {k}" for k in ep["keywords"])
        body = "\n\n".join(f"{who} ({ts}):\n{text}" for who, ts, text in ep["turns"])
        front = "\n".join(
            [
                "---",
                f"guest: {ep['guest']}",
                f"title: {ep['title']}",
                f"youtube_url: https://www.youtube.com/watch?v={ep['video_id']}",
                f"video_id: {ep['video_id']}",
                f"publish_date: {ep['publish_date']}",
                f"description: 'Synthetic test episode with {ep['guest']}.'",
                "duration_seconds: 1800.0",
                "view_count: 42",
                "channel: Test Podcast",
                "keywords:",
                keywords,
                "---",
                "",
                f"# {ep['title']}",
                "",
                "## Transcript",
                "",
            ]
        )
        folder.joinpath("transcript.md").write_text(front + "\n" + body + "\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    docs.joinpath("plan-guide.md").write_text(
        "---\ntitle: Plan Guide\nkeywords: [enrollment, plans]\n---\n\n# Plan Guide\n\n"
        "## Enrollment periods\n\nThe annual enrollment period runs each autumn. "
        "Members may switch plans once during the open window.\n\n"
        "## Costs\n\nPremiums, deductibles and copays are set per plan year.\n",
        encoding="utf-8",
    )
    docs.joinpath("faq.txt").write_text(
        "Frequently asked questions.\n\nCan I keep my doctor? Check the plan network first.\n",
        encoding="utf-8",
    )
    write_thread(root)
    return root


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    return write_sample_corpus(tmp_path / "corpus")


@pytest.fixture
def transcript_source() -> SourceSpec:
    return SourceSpec(
        id="test-podcast",
        kind="local",
        path="episodes",
        loader="transcripts",
        glob="*/transcript.md",
        description="Synthetic podcast transcripts",
    )


@pytest.fixture
def documents_source() -> SourceSpec:
    return SourceSpec(id="docs", kind="local", path="docs", loader="documents", glob="**/*")


@pytest.fixture
def thread_source() -> SourceSpec:
    """A documents source whose one file is a discussion thread, so it arrives with no speakers."""
    return SourceSpec(id="threads", kind="local", path="threads", loader="documents", glob="**/*")


@pytest.fixture
def persona(transcript_source: SourceSpec) -> PersonaSpec:
    return PersonaSpec(
        id="test-pm",
        name="Test Product Leader",
        description="A synthetic product persona for tests.",
        role_prompt="You are a product leader. Cite your sources.",
        voice=["direct", "practical"],
        sdlc_stages=["discovery", "requirements", "prioritization"],
        sources=[transcript_source],
        retrieval=RetrievalConfig(top_k=4, expand_neighbors=1),
    )


@pytest.fixture
def docs_persona(documents_source: SourceSpec) -> PersonaSpec:
    return PersonaSpec(
        id="test-docs",
        name="Test Docs Persona",
        role_prompt="You answer from plan documents.",
        sdlc_stages=["support", "compliance"],
        sources=[documents_source],
    )


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(model_name="hash-test", dim=DIM)


@pytest.fixture
def memory_store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def pipeline(memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder) -> IngestPipeline:
    return IngestPipeline(memory_store, hash_embedder, doc_batch=2)


@pytest.fixture
def ingested(
    pipeline: IngestPipeline,
    sample_corpus: Path,
    persona: PersonaSpec,
    transcript_source: SourceSpec,
) -> IngestReport:
    return pipeline.ingest(sample_corpus, persona, transcript_source)


@pytest.fixture
def thread_document(
    sample_corpus: Path,
    docs_persona: PersonaSpec,
    thread_source: SourceSpec,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
) -> str:
    """The ingested thread's document id: one document, several passages, and no speakers yet."""
    IngestPipeline(memory_store, hash_embedder).ingest(sample_corpus, docs_persona, thread_source)
    return next(iter(memory_store.document_ids(docs_persona.id, thread_source.id)))


@pytest.fixture
def retriever(memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder) -> Retriever:
    return Retriever(memory_store, hash_embedder)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        personas_dir=tmp_path / "personas",
        raw_dir=tmp_path / "raw",
        snapshots_dir=tmp_path / "snapshots",
        enrichment_dir=tmp_path / "enrichment",
        attribution_dir=tmp_path / "attribution",
        skills_dir=tmp_path / "skills",
        neo4j=Neo4jSettings(uri="bolt://unused:7687"),
        embedding=EmbeddingSettings(backend="hash", model="hash-test", dim=DIM),
    )


@pytest.fixture
def registry(
    settings: Settings, persona: PersonaSpec, docs_persona: PersonaSpec
) -> PersonaRegistry:
    reg = PersonaRegistry(settings.personas_dir)
    reg.save(persona)
    reg.save(docs_persona)
    return reg


@pytest.fixture
def app_context(
    settings: Settings,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    registry: PersonaRegistry,
) -> AppContext:
    return AppContext(
        settings=settings, store=memory_store, embedder=hash_embedder, registry=registry
    )


@pytest.fixture
def cli_context(app_context: AppContext) -> Iterator[AppContext]:
    """Point the CLI at the in-memory context for the duration of a test."""
    from graphrag.cli import State

    State.factory = lambda: app_context
    try:
        yield app_context
    finally:
        State.factory = None
