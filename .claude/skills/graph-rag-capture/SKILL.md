---
name: graph-rag-capture
description: Capture a document into a graph-rag corpus so every layer lands - the text, its entities, its speakers and what its passages say. Use when asked to "capture a document", "write a research note into the corpus", "ingest this", "add to the corpus", or after writing any file under data/raw/.
---

# graph-rag-capture: the capture contract

**Repo:** `/Users/asmith/development/graph-rag`. All paths are relative to it.

Writing the text is one of four passes. The other three are sidecar JSON files you write beside
it. Nothing warns you when they are missing: the document sits in the graph, and every question
that needs entities, speakers or stance about it comes back empty.

**A document is captured when `graphrag layers check` exits 0. Not before.**

## 1. Write the document

Path: `data/raw/<persona>/<source>/<...>.md` — `<source>` is a source id from
`personas/<persona>/persona.yaml`. Suffixes the loaders read: `.md`, `.markdown`, `.txt`, `.pdf`.

YAML front-matter, valid (quote any string containing a colon):

```yaml
---
title: One line naming what this document is
published: YYYY-MM-DD          # or the source's own date
fetched_at: YYYY-MM-DDTHH:MM:SSZ
document_type: <what it is>     # e.g. research_note, capture
keywords: [three, to, ten, terms]
description: One sentence.
---
```

The id the loaders will give it is `<persona>:<source>:<slug of the path under the source>`
(a `transcripts` source uses the containing folder instead). Revise a document in place; never
write a `_v2` copy.

## 2. Write the sidecars

One file per document, each named after the document's slug:

| layer | path | when |
| --- | --- | --- |
| extraction | `data/enrichment/<persona>/<source>/<slug>.json` | always |
| attribution | `data/attribution/<persona>/<source>/<slug>.json` | the document carries attributed posts |
| annotation | `data/annotations/<persona>/<source>/<slug>.json` | it names entities or facets the persona declares |

Every sidecar carries the `doc_id` of the document it is about. That id, not the directory, is
what ties the file to a document.

**Extraction** — entities and relations, format and quality rules in the `graph-rag-enrich`
skill (`.claude/skills/graph-rag-enrich/SKILL.md`, section 3). The rule that costs the most when
broken: each `name` must occur **verbatim** in the text, or the mention anchors to the wrong
passage and the check reports it as loose.

**Attribution** — who wrote which passage, for a document the loader reads as one body of prose:

```json
{"doc_id": "<persona>:<source>:<slug>",
 "posts": [{"speaker": "handle or name",
            "anchor": "verbatim opening words of the post, 6 to 20 words",
            "role": "op",
            "date": "YYYY-MM-DD",
            "score": 12}]}
```

`role` is `op` or `reply`; `date` and `score` may be `null`. A post whose anchor occurs in no
passage is skipped, never guessed at.

**Annotation** — what a passage says about something, and which function it is about:

```json
{"doc_id": "<persona>:<source>:<slug>",
 "annotations": [{"anchor": "verbatim six to twenty words of the passage",
                  "entity": "Entity name as written in that passage",
                  "stance": "praise | complaint | substitution | neutral",
                  "facets": ["<slug from facets.yaml>"],
                  "note": "optional short paraphrase, never quoted back as evidence"}]}
```

`anchor` is required, `entity` optional, `stance` needs an `entity`. Facets come from
`personas/<persona>/facets.yaml` when the persona keeps one; a facet outside it is dropped and
reported. Read that file before writing facets.

**Aliases** — when the document spells an entity a way the corpus already has under another
name, add the spelling to `personas/<persona>/aliases.yaml`:

```yaml
aliases:
  Canonical Name: [Alias One, alias two]
```

One alias belongs to one canonical only. Keep the passage's own surface form in the extraction
file; the alias table is what folds the spellings into one node.

## 3. Check, then import

```bash
docker compose run --rm -T graphrag graphrag layers check <persona> --file data/raw/<persona>/<source>/<file>.md
```

It reports, per document, which sidecars are on disk, which layers the graph holds, and what
each sidecar's importer would leave loose. It writes nothing. Exit 1 means a document has no
extraction JSON, a sidecar names a document the graph does not hold, or entries came loose.
Fix the file the reason points at and run it again:

| reason | fix |
| --- | --- |
| entity name not verbatim in any passage | use the spelling the text uses |
| entity name in no passage | drop the entity, or quote where it occurs |
| post anchor not found / annotation anchor not found | copy the words verbatim from the document |
| entity unknown to the persona | extraction has not covered it, or it needs an alias |
| entity not in the passage | the anchor points at the wrong passage |
| unknown document | the text has not been ingested yet; run `make sync` first |

Then put it all in the graph:

```bash
make sync PERSONA=<persona>
```

One command: it ingests what is missing, re-imports the sidecars for anything it re-ingested,
imports sidecars for documents that have none of that layer, and applies the alias table. Run
`graphrag layers check <persona> --file ...` once more afterwards; it should exit 0 with every
layer `ok`.

To check a whole corpus: `graphrag layers check <persona> --all [--source <id>]`.

## 4. Rules

- The sidecars are evidence about the document, not about the world. Every anchor is verbatim
  text from that document.
- An entry that will not place is reported, not forced. Fix the file rather than removing the
  check.
- Nothing here creates entities that the extraction pass did not: the annotation layer reads
  what is there.
- Once the layers are in, `graph-rag-sna` (`.claude/skills/graph-rag-sna/SKILL.md`) is what
  reads them: speaker networks, stance filters, time windows.
