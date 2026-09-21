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
from graphrag.models import (
    Chunk,
    Document,
    Enrichment,
    Entity,
    Mention,
    PersonaSpec,
    Relation,
    RetrievalConfig,
    SourceSpec,
    SpeakerPost,
)
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


"""A tiny corpus planted straight into the store, with every layer already on it.

The ingest fixtures give a realistic corpus but no dates, no stances and no facets, and adding
them through the importers would make every network test depend on anchor matching. These five
posts are written directly instead, so each assertion below can name the number it expects:

===== ======= ========== ================================ =================
post  speaker posted     entities (stance)                facets
===== ======= ========== ================================ =================
1     ana     2025-01-10 Alpha (praise), Beta (praise)     quoting
2     bo      2025-01-20 Alpha (complaint), Gamma (compl.) outage
3     ana     2025-06-10 Beta (praise), Gamma (neutral)     quoting
4     cy      2025-06-20 Alpha (complaint), Gamma (compl.) outage
5     dot     undated    -                                  -
===== ======= ========== ================================ =================

So praise joins Alpha to Beta and complaint joins Alpha to Gamma twice: the stance filter does
not thin one network, it produces a different one. The two halves of the year hold different
speakers, and post 5 is dated by nobody, so it leaves every window.

Each post also carries an invented node attribute, ``region``, on its document and on its
speaker, so an attribute filter has something to keep and something to drop::

    post-1 ana  region=north    post-2 bo  region=south
    post-3 ana  region=north    post-4 cy  region=south
    post-5 dot  no region at all

``dot`` and post 5 are deliberately untagged: a node nobody tagged has to leave every filtered
network, exactly as an undated post leaves every window.
"""
LAYERED_PERSONA = "test-layers"
LAYERED_SOURCE = "posts"
LAYERED_POSTS: tuple[tuple[str, str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "post-1",
        "ana",
        "2025-01-10",
        "Alpha saved me an afternoon on the quote and Beta laid the numbers out side by side, "
        "which is exactly what I wanted from both of them.",
        ("quoting",),
        ("pricing", "service"),
    ),
    (
        "post-2",
        "bo",
        "2025-01-20",
        "Alpha went down on the busiest morning of the month and Gamma took twenty minutes to "
        "load one page, so I did the whole thing on paper.",
        ("outage",),
        ("service", "outages"),
    ),
    (
        "post-3",
        "ana",
        "2025-06-10",
        "Beta has been steady for a year now, and Gamma is simply the sheet I open when I want "
        "to see the workings behind a number.",
        ("quoting",),
        ("pricing", "service"),
    ),
    (
        "post-4",
        "cy",
        "2025-06-20",
        "Alpha lost a half-finished quote again this week and Gamma dropped the session twice "
        "while I had someone on the phone.",
        ("outage",),
        ("service", "outages"),
    ),
    (
        "post-5",
        "dot",
        "",
        "Nobody wrote a date on this one, which is exactly what makes it worth keeping here.",
        (),
        ("service",),
    ),
)
#: Which region a post's document and its speaker were tagged with; absent means untagged.
LAYERED_REGIONS: dict[str, str] = {
    "post-1": "north",
    "post-2": "south",
    "post-3": "north",
    "post-4": "south",
}
#: entity id -> (name, type) and, per post, the stance the annotation pass put on the mention.
LAYERED_ENTITIES: dict[str, tuple[str, str]] = {
    "product:alpha": ("Alpha", "product"),
    "product:beta": ("Beta", "product"),
    "product:gamma": ("Gamma", "product"),
}
LAYERED_MENTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "post-1": (("product:alpha", "praise"), ("product:beta", "praise")),
    "post-2": (("product:alpha", "complaint"), ("product:gamma", "complaint")),
    "post-3": (("product:beta", "praise"), ("product:gamma", "neutral")),
    "post-4": (("product:alpha", "complaint"), ("product:gamma", "complaint")),
}


@pytest.fixture
def layered(
    memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder, registry: PersonaRegistry
) -> InMemoryGraphStore:
    """The planted corpus above, in the shared in-memory store and the shared registry."""
    source = SourceSpec(id=LAYERED_SOURCE, kind="local", path="posts", loader="documents")
    registry.save(
        PersonaSpec(
            id=LAYERED_PERSONA,
            name="Test Layers",
            role_prompt="You answer from captured posts.",
            sources=[source],
        )
    )
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for ordinal, (slug, speaker, posted, text, facets, topics) in enumerate(LAYERED_POSTS):
        doc_id = f"{LAYERED_PERSONA}:{LAYERED_SOURCE}:{slug}"
        documents.append(
            Document(
                id=doc_id,
                persona_id=LAYERED_PERSONA,
                source_id=LAYERED_SOURCE,
                title=slug,
                path=f"posts/{slug}.md",
                speakers=[speaker],
                topics=list(topics),
                word_count=len(text.split()),
            )
        )
        chunks.append(
            Chunk(
                id=f"{doc_id}#0",
                doc_id=doc_id,
                persona_id=LAYERED_PERSONA,
                ordinal=ordinal,
                text=text,
                speaker=speaker,
                speakers=[speaker],
                speaker_posts=[SpeakerPost(speaker=speaker, posted_at=posted or None)],
                facets=list(facets),
                word_count=len(text.split()),
            )
        )
    memory_store.upsert_documents(documents)
    memory_store.upsert_chunks(chunks, hash_embedder.embed_documents([c.text for c in chunks]))
    for slug, region in LAYERED_REGIONS.items():
        doc_id = f"{LAYERED_PERSONA}:{LAYERED_SOURCE}:{slug}"
        memory_store.set_document_attributes(doc_id, {"region": region})
        speaker = next(p[1] for p in LAYERED_POSTS if p[0] == slug)
        memory_store.set_speaker_attributes(LAYERED_PERSONA, speaker, {"region": region})
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id=key, name=name, type=kind)
                for key, (name, kind) in LAYERED_ENTITIES.items()
            ],
            mentions=[
                Mention(
                    chunk_id=f"{LAYERED_PERSONA}:{LAYERED_SOURCE}:{slug}#0",
                    entity_id=entity_id,
                    stance=stance,  # type: ignore[arg-type]
                )
                for slug, pairs in LAYERED_MENTIONS.items()
                for entity_id, stance in pairs
            ],
        )
    )
    return memory_store


LAYERED_RELATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("product:alpha", "product:beta", "integrates_with", "post-1"),
    ("product:alpha", "product:beta", "integrates_with", "post-3"),
    ("product:beta", "product:alpha", "integrates_with", "post-1"),
    ("product:alpha", "product:gamma", "competes_with", "post-2"),
    ("product:alpha", "product:gamma", "competes_with", "post-4"),
    ("product:gamma", "product:beta", "replaces", "post-3"),
    ("product:gamma", "product:beta", "competes_with", "post-4"),
)
"""The relations an extraction pass wrote on top of the planted corpus above.

Kept out of ``layered`` because the counts in its own tests are asserted by hand; a test that
wants the directed network asks for ``related`` and gets both. Written as ``(source, target,
type, post)`` so the answers can be read off the table rather than computed:

===== ===== ================ ====== =========
from  to    type             post   passages
===== ===== ================ ====== =========
Alpha Beta  integrates_with  1, 3   2
Beta  Alpha integrates_with  1      1
Alpha Gamma competes_with    2, 4   2
Gamma Beta  replaces         3      1
Gamma Beta  competes_with    4      1
===== ===== ================ ====== =========

So four directed edges over three connected pairs, one of them (Alpha-Beta) stated in both
directions: reciprocity is 1/3 by the book's definition (§10.3, connected pairs reciprocated
over connected pairs), where ``nx.overall_reciprocity`` would say 2/4. Gamma to Beta carries two
types in two passages, which a simple directed graph folds into one edge of weight 2 typed with
the alphabetically first of the tied types.
"""


@pytest.fixture
def related(layered: InMemoryGraphStore) -> InMemoryGraphStore:
    """``layered`` with the relations above, for the directed ``relations`` network."""
    layered.upsert_enrichment(
        Enrichment(
            relations=[
                Relation(
                    source_id=source,
                    target_id=target,
                    type=kind,
                    evidence=f"{slug} says so",
                    chunk_id=f"{LAYERED_PERSONA}:{LAYERED_SOURCE}:{slug}#0",
                )
                for source, target, kind, slug in LAYERED_RELATIONS
            ]
        )
    )
    return layered


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
        annotations_dir=tmp_path / "annotations",
        skills_dir=tmp_path / "skills",
        sync_lock_dir=tmp_path / "locks",
        sna_cache_dir=tmp_path / "sna-cache",
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
