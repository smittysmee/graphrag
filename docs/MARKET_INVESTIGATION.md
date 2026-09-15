# Market investigation: a repeatable workflow

A running record of the method used to investigate a product-adoption problem in a market the
team did not start out knowing, using the graph as the evidence base and persona agents as the
reviewers. Written so it can be reused for any market. The worked case that produced it lives
in a private case log; nothing here depends on it.

Status: living document. Add a dated entry under "Change log" when a step is added or changed.

## The shape of the problem this fits

- Someone hands you a number and a target ("adoption is 50%, get it to 75%") plus a list of
  suspected causes.
- The people whose behaviour has to change are several organisational hops away and are not
  employees.
- The public record is thin, self-selected, and mostly about adjacent things.
- Nobody has yet said what the number is a fraction of.

## Stages

### 0. Frame the mandate as evidence, not truth

Write the mandate down as an internal-context document in the corpus, with provenance (who said
it, when). Turn every stated cause into a hypothesis. Identify the regulatory or structural
regimes that split the market (in the worked case, two product categories with different rules
on what money can be paid to whom) and keep them separate in every later artefact.

### 1. Build the evidence base with provenance

- Write a research brief: paths, file format, front-matter, provenance manifest, rules
  ("facts only from sources you captured", "test the premise, don't confirm it").
- Fan out fast, cheap researchers (about six per wave) each owning one slice. A wave produces
  12 to 20 captured files per researcher plus one research note that cites them.
- Validate front-matter and length before ingest. Ingest. Have agents write the entity layer
  (one JSON per document, names verbatim from the text) and import it. Export the snapshot.
- Record what could not be fetched and how the workaround worked; the next wave needs it.

### 2. Ground three kinds of persona on the graph

- A **domain analyst** persona grounded in the captured corpus.
- A **method** persona grounded in a large practitioner corpus (product discovery, growth,
  behaviour change). Every claim from it is an analogy and is labelled so.
- Optionally a **framework** persona (an operating model the organisation is adopting) with its
  evidence, lineage and critique captured alongside, so it can be cross-examined too.

### 3. Grill the premise before researching it

Put the naive plan to the method persona in rounds: premise, what the sources say against it,
correction. Expect the first correction to be "your target is not a behaviour". Write the
refined premise and a hypothesis list with a first test for each. This is what the research
waves then go and settle.

### 4. Candidate solutions and a tribunal

List candidate solutions with the evidence for each. Have every persona interrogate every
solution against its own graph: verdict, three hardest questions, citations, the change it would
make. Synthesise where they agree, resolve where they disagree, list regulatory tripwires that
turn a keep into a kill, and write down "what nobody was asking". Do not sequence anything yet.

### 5. A diagnosis-only agent

Build an agent whose sole job is to say why adoption is not happening and how to detect it,
and which refuses to design solutions. Its method:

1. Restate adoption as a behaviour with a denominator (breadth, depth, volume, ceiling).
2. Generate hypotheses across the whole cause space (never heard of it, cannot, will not, tried
   and quit, partial, blocked by the hierarchy, no reason to return, trust, measurement artefact).
3. Ground each in both graphs; say where the graph is silent.
4. Design the cheapest detection instrument per hypothesis, naming the hop it observes, who runs
   it, the confirming and killing signal, and the regulatory guard.
5. Rank by information value, not ease. Name the three to run first.

The agent writes its note into the corpus and enriches it itself, so the next session can
cite it.

### 6. The insider correction loop

Whenever someone with inside knowledge corrects a premise, capture it immediately as an
internal-context document with its own entity layer, and record two things: what the correction
overturns, and what the corpus already had that analysts misread. In the worked case the corpus
held three hedged hints that the product was a single-sign-on layer; every analyst kept a native
feature in the model anyway. The lesson: when sources say "at least in part", ask the insider
"is there any part that is not?"

### 7. Competitor voice and agent archetypes

Research the tools people actually use, from their own words only: forums, review sites,
comment threads. No vendor material. For every captured thread record the fields a profile will
need: product line, tools named, role of the poster, tenure, volume, carrier count, hierarchy
position. Each researcher's note describes the recurring types it saw with the files behind each.

### 8. Profiles and modelling

Merge the recurring types into a small set of profiles, each attribute backed by verbatim
quotes; discard any profile without them. Then run candidate interventions against the profiles
by asking the persona agents, in character for each profile, whether the intervention changes
where that person starts their work. Treat the answers as hypotheses for the field instruments,
not as results.

### 9. Network analysis on the corpus

Three networks live in any corpus like this and are worth measuring once the entity layer
exists: who talks with whom (speakers sharing documents), what is discussed together (entities
co-mentioned in passages, topics that co-occur), and how nodes cluster by structure or by
content. Use community detection (Louvain, many seeds, checked against a degree-preserving null
model) when the object is a graph and the question is "which nodes form groups"; use K-means on
an embedding when clusters should be compact and membership hard; use a Gaussian mixture when
clusters overlap or a soft membership is the useful answer, choosing k by BIC. Report n, the
sampling frame, stability between runs and the null model every time. A discourse network
describes who posts, not the population. The questions it answers for adoption work: who shapes
a newcomer's first choices, what each product is discussed alongside, and whether the profiles
are distinct groups or one population with a gradient.

### 10. Annotate, then analyse again

Once entities and speakers exist, three cheap layers make the networks answer sharper
questions. Aliases: one file per corpus mapping spellings to a canonical name, applied at import
and to the graph, because split names make every count wrong. Dates on speaker edges, so a
network can be cut into windows around an event. Passage annotations: an agent writes
passage-anchored labels with an optional entity, a stance (praise, complaint, substitution,
neutral) and facets from a per-corpus vocabulary, and one importer attaches them. Stance turns a
co-mention network into a signed one; facets turn "discussed together" into "discussed together
about the same function". The same three files serve any corpus.

## Plumbing that made the loop hold together

- **Session card** (SessionStart hook): what personas exist, live counts, newest raw files,
  what is not yet ingested or enriched.
- **Prompt router** (UserPromptSubmit hook): nudges the model to consult the right persona,
  with vocabulary derived from the graph rather than a keyword list.
- **Un-ingested check** (Stop hook) and `make sync`: one command to ingest what is missing and
  re-import the entity layer.
- **Capture contract** (PostToolUse hook on writes under a raw corpus folder, one check
  command, one skill): the moment any agent writes a document, it is told the doc id, the
  sidecars it owes (extraction, attribution, annotations) and the command that verifies them, so
  the layers are produced by whoever has the content in context rather than by a later pass.
- **Untrusted-text controls**: retrieved passages framed as data, hook output sanitized,
  hidden characters stripped at ingest and injection-shaped phrases flagged, subagents blocked
  from publishing.

## Lessons so far

- Ask what the number is a fraction of before anything else. Volume shares move with mix; a
  ceiling can make a target impossible; a book includes things the product cannot touch.
- Ask how the number is computed. Attribution decides what "adoption" means.
- Separate growth from adoption. Recruiting people who already use the tool raises the share.
- Hedged sources ("partly", "at least in part") are where analysts overreach. Resolve them with
  an insider before building on them.
- Real voice is self-selected. Record what was not found and which combinations were never
  searched, so silence is never mistaken for absence.
- Fast, cheap agents for capture; a capable model for diagnosis, synthesis and anything that
  commits. Keep waves to about six agents.
- Anything an agent writes into the corpus is invisible until ingested and enriched. Automate
  the reminder, then the command, then the enrichment step inside the agent that wrote the note.
- Give a shared rate-limited source to one agent, not six. When a whole wave hits the same
  archive mirror concurrently, every researcher gets throttled and the source reads as empty.
  Run the shared source as a single sequential pass alongside the parallel wave, and record
  "empty" and "rate-limited" as different outcomes in provenance.
- A wave's null result on one source is only a null if the source was actually reachable.
- Page the community's own boards by date instead of relying on search-engine discovery. Search
  surfaces the threads that were linked or indexed; paging surfaces the recurring voices, and
  recurring voices are what profiles are built from.
- Court filings are the only first-person voice under oath. Complaints and declarations in
  worker-versus-hierarchy suits describe recruitment, pay, debt and exit in a register forums
  never reach. Treat allegations as allegations.
- Review sites yield by business model, not by product line: a recruiting-heavy organisation's
  reviews are written by its recruits; a consumer-facing one's are written by customers.
- Capture handles at capture time or lose the speaker network. Two early waves stripped
  usernames by their own rules and could never be attributed afterwards; the network that
  resulted is a network of the later waves only.
- Communities in a discourse network follow how the corpus was captured before they follow the
  subject: single-site, single-wave captures show up as islands. Check the partition against the
  capture provenance before reading it as social structure.
- Test "are the profiles distinct groups" formally. Cluster the grounding documents by content,
  compare to the profile labels with adjusted Rand index and NMI, and compare NMI against a random
  partition of the same size, which already scores high. A weak signal means one population with
  a gradient, and the profiles are a reading aid, not a segmentation.
- A layer imported before a field existed stays stale silently: a backfill that only fills
  absences never revisits what is present. Give every import a refresh mode and use it after any
  change to a sidecar format.
- Enforce the corpus validator in the same command that ingests, or an oversized or malformed
  note reaches the graph and is found only by a later reader.
- Sync detects missing documents and missing entity layers, not modified documents. A note
  revised in place must be re-ingested explicitly (per-source) before its new text is searchable.

## Change log

- 2026-09-13: first version, written after stages 0 to 7 had each been run once.
- 2026-09-13: added the shared-source lesson after six parallel researchers throttled each other on the Reddit mirror.
- 2026-09-13: added board paging, court voice, review-site yield and the modified-document gap after wave 5.
- 2026-09-13: added stage 9, network analysis, with the method-selection rules.
- 2026-09-13: added the three lessons from running stage 9 (capture handles, provenance islands, test profile distinctness).
- 2026-09-13: added stage 10, aliases, dates and passage annotations.
- 2026-09-13: added the capture contract to the plumbing after the sidecar passes proved to be lead-orchestrated work that belongs with the writer.
- 2026-09-13: added the stale-layer and validate-at-ingest lessons after the second analysis exposed both.
