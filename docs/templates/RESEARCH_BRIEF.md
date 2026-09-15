# Research brief: `<persona>` corpus (template)

**How to use this template.** Copy this file to `personas/<persona>/RESEARCH_BRIEF.md` and
replace every placeholder (`<persona>`, `<market>`, `<tool>`, `<competitor>`, `<segment>`,
`<role>`, source names, categories) with the real ones for your persona. Delete this note once
the copy is filled in.

You are one of several research agents building a cited, provenance-tracked corpus that will be
ingested into a knowledge graph (persona `<persona>`). Someone will use this corpus to understand
`<market>` and to ground answers in real sources rather than memory. Your job is to FIND, CAPTURE
and NOTE, not to write an essay from memory.

## Repo and paths

Repo: the project root. Write ONLY under `data/raw/<persona>/<source-id>/<topic>/` (source-id is
given in your task) and `data/raw/<persona>/provenance/<your-agent-label>.jsonl`. Research notes
live alongside captured documents, in a `notes/` folder under the same source-id. Do not touch
anything else. Do not run git, docker or make unless your task says to.

## Method

A **wave** is one round of research agents working the same brief in parallel, each owning a
different slice of the source space, followed by one pass that folds their notes together. Within
a wave:

1. Search for primary sources first (the market's own regulators, standards bodies, official
   registries, or the maker of `<tool>`'s own published material, if the fact is about the maker
   rather than about how people experience it), then trade or industry press, then practitioner
   sites that publish real numbers (pricing, schedules, structural detail). Prefer the most recent
   version of anything that changes over time, and note when a source has been superseded.
2. Fetch the full text of each source you decide to keep. Extract the text; do not summarise on
   the first pass. If a fetch fails, try one alternate URL or mirror; if it still fails, record the
   failure in provenance and keep only what a search snippet gave you, marked
   `content_fidelity: summary`.
3. Capture a modest, quality-over-count set of documents per researcher (a dozen or two is
   typical). Avoid low-value listicles unless one is the only source for a fact.
4. Finish with one to three research notes that synthesise what you captured, cite the captured
   files, answer your task's questions, and list open questions and contradictions.

Evidence before synthesis: capture first, write claims second, and every claim in a note should
trace back to a file a reader can open.

## Captured document file format

Path: `data/raw/<persona>/<source-id>/<topic>/<slug>.md` where slug is
`YYYY-MM-DD-<publisher>-<short-title>` (use `undated` when no date is known). Keep the body under
6,000 words and above 40 words (the validator rejects both empty stubs and oversized dumps): for
long documents capture the relevant sections verbatim and set `content_fidelity: excerpt`.

```markdown
---
title: "Exact title of the source"
source_url: https://...
publisher: "<publisher or site name>"
author: "name or omit"
published: 2024-04-04          # YYYY-MM-DD, or omit if unknown
fetched_at: 2026-01-01T00:00:00Z
fetched_by: <your-agent-label>
retrieval: web_fetch | web_search_snippet
content_fidelity: verbatim | excerpt | summary
document_type: guidance | press_release | company_page | news | analysis | filing | research_note | internal_context
category: <segment-a> | <segment-b> | <cross-cutting-topic>
keywords: [<term one>, <term two>]
description: One sentence saying what this document establishes.
---
<captured text, verbatim or excerpted, markdown headings preserved where sensible>
```

What each key means:

- `title` — the source's own title, quoted exactly.
- `source_url` — where you fetched it. Required on every document except a `research_note` or an
  `internal_context` document (something written down from a conversation, a mandate, or another
  non-public origin rather than fetched).
- `publisher` — who put the source out; use a real, specific name, not a guess.
- `fetched_at`, `fetched_by` — when and by which researcher label the capture happened.
- `retrieval` — how the text reached you: a full fetch, or only a search-result snippet.
- `content_fidelity` — `verbatim` (the whole body, unedited), `excerpt` (a verbatim slice of a
  longer source), or `summary` (you are relaying gist, not exact words). Mark this honestly; it is
  the one field that tells a later reader how much to trust the wording.
- `document_type` — what kind of thing this is, from the vocabulary your persona declares.
- `category` — which slice of the corpus this belongs to; keep the categories your persona defines
  separate and consistent across researchers.

## Research note file format

Same front-matter with `document_type: research_note`, `retrieval: agent_synthesis`,
`content_fidelity: summary`, `fetched_by`, `fetched_at`, and `sources: [relative paths of the
captured files it draws on]`. Body: headed sections, every factual claim followed by the file it
comes from in brackets, e.g. `[2024-04-23-example-source.md]`. State explicitly where sources
disagree or are silent. Never present your own inference as a sourced fact; put inferences under a
heading `## Analyst inferences`.

A note is not finished when the markdown is written. Depending on what your persona declares, it
may owe sidecar files too (an entity layer, speaker attribution, stance and facet annotations).
Follow whatever capture skill or checklist your project uses, then verify and import before moving
on — a note's content is not searchable, and nothing later can cite it, until that step succeeds.

## Provenance manifest

Append one JSON line per captured document (and per failed fetch) to
`data/raw/<persona>/provenance/<your-agent-label>.jsonl`:

```json
{"file": "<path relative to data/raw/<persona>>", "source_url": "...", "title": "...",
 "publisher": "...", "published": "...", "fetched_at": "...", "fetched_by": "...",
 "retrieval": "...", "content_fidelity": "...", "status": "ok|fetch_failed|blocked",
 "sha256_body": "<sha256 of the markdown body after the front-matter, or null>", "notes": "..."}
```

Compute the hash on the body text after the front-matter, or record `null` when there is no body
(a failed fetch). The `file` field must point at a file that actually exists once you are done —
the validator checks every "ok" record against the filesystem and fails the whole corpus on a
dangling reference.

## Rules

- **Verbatim capture, no paraphrase.** A captured document holds the source's own words. If you
  need to shorten a long source, cut sections and mark `content_fidelity: excerpt`; do not rewrite
  what remains.
- **Mark fidelity honestly.** `summary` is not a lesser choice, it is the correct one whenever you
  only have a snippet or are relaying gist. A confident-looking `verbatim` tag on paraphrased text
  is worse than an honest `summary` tag.
- **One document per source.** Do not merge two distinct sources into one captured file, even if
  they cover the same fact — each should be traceable to its own URL and fetch.
- **Do not invent.** Facts only from sources you captured. If a fact matters and you cannot find a
  source for it, say so in the research note as an open question rather than filling the gap from
  memory or plausible-sounding assumption.
- Valid YAML front-matter (quote strings that contain a colon). Valid JSON lines in the
  provenance file. UTF-8 throughout. No file over 6,000 words, none under 40.

## Fetch workarounds

Some sources will not fetch directly — a paywall, a block on automated requests, or a page that
renders nothing without a browser. Three patterns cover most cases. Try the plain fetch first;
reach for these only when it fails.

- **An archive mirror of the page.** A general-purpose web archive keeps a stored copy of many
  pages as they existed on a given date, and serving the stored copy sidesteps whatever blocked
  the live fetch. Reach for this when the source is stable, public content that changes rarely
  (a policy page, a document that will not be updated) and you do not specifically need today's
  version.
- **A reader proxy that returns readable text.** A reader-mode proxy service fetches a page on
  your behalf and returns its extracted text rather than the styled page, which both dodges some
  blocks and saves you from parsing markup. Reach for this on pages that are mostly one article or
  one thread and that block direct fetches but do not require a login.
- **A review or discussion feed exposed as a machine-readable format.** Many discussion boards,
  review sites and app stores expose the same content your browser sees through a documented
  feed or API endpoint (JSON, RSS, or similar) that is not blocked the way the browsable page is.
  Reach for this when you need many items from one site (reviews, comments, posts) rather than one
  page, since a feed also tends to give you structured fields (score, date, author) a scraped page
  would not.

Whichever pattern you use, record it in the provenance line's `notes` field, and record what did
not work as clearly as what did — the next wave should not re-discover the same dead end.
