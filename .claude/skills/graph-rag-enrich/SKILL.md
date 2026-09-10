---
name: graph-rag-enrich
description: Enrich the graph-rag knowledge graph by extracting entities and relationships from ingested documents (podcast transcripts, plan documents) and importing them, no API key needed. Use when asked to "enrich", "extract entities", "build the entity layer", or "add relationships" for a persona, or to enrich "the remaining episodes/documents".
---

# graph-rag-enrich: agent-driven entity extraction

**Repo location:** wherever you cloned this repo; all paths below are relative to it.

You (or subagents you spawn) read each source document and write one JSON file per document;
`graphrag enrich-import` anchors the entities to passages, writes the graph, and re-exports the
committed snapshot. No Anthropic API key is involved.

## 1. Find what still needs enriching
```bash
# documents in the graph vs JSON files already written
ls data/enrichment/*.json | xargs -n1 basename | sed 's/.json$//' | sort > /tmp/done.txt
ls data/raw/product-leader/episodes | grep -v '_$' | sort > /tmp/all.txt      # product-leader
# also skip folders whose transcript body duplicates another folder (ingest skips them by
# content fingerprint, so they have no document in the graph):
#   andy-raskin_ april-dunford-20 elena-verna-30 ethan-evans-20 fei-fei hamelshreya julie-zhuo-20
#   madhavan-ramanujam-20 marty-cagan-20 nicole-forsgren-20 wes-kao-20 yuhki-yamashata
# (`graphrag documents`/`enrich-import --dry-run` reports "unknown document" for any other stray id)
comm -23 /tmp/all.txt /tmp/done.txt
```
For the `product-leader` persona a folder `episodes/<slug>/transcript.md` maps to
`doc_id = product-leader:lennys-podcast:<slug>` (folders ending in `_` are skipped as
duplicates; a repeat guest's second folder gets `-2`). For other personas, list ids with
`docker compose run --rm graphrag graphrag search "" ...` or `documents(persona_id)` over MCP.

## 2. Fan out
Spawn one general-purpose subagent per 2-3 documents (a transcript is ~15-25k words; read it in
chunks with offset/limit). Give each subagent this skill path plus its document list and doc ids.
Run 5-6 subagents at a time; wait for the batch, validate, import, then start the next batch.

## 3. What each subagent writes: `data/enrichment/<slug>.json`
```json
{
  "doc_id": "product-leader:lennys-podcast:<slug>",
  "entities": [
    {"name": "Canonical Name", "type": "person|company|product|framework|concept|book|metric|regulation|other",
     "description": "one sentence, your own words, grounded in the document"}
  ],
  "relations": [
    {"source": "Canonical Name", "target": "Canonical Name", "type": "UPPER_SNAKE_VERB",
     "evidence": "short paraphrase in your own words, max 20 words"}
  ]
}
```
Quality rules:
- Only entities the document substantively discusses (guests, companies, products, named
  frameworks/methods, named concepts, books, specific metrics, regulations). 20-45 per document.
- Do not add the interviewer/host as an entity or emit INTERVIEWS relations; speakers are already
  graph nodes. Add the host only if their own work is substantively discussed.
- `name` must be a surface form that occurs verbatim in the text (case-insensitive), so mentions
  anchor to the right passages. Prefer the most complete form used ("Sean Ellis test", not "the
  test"). Reuse the same canonical name across documents where the text allows.
- 10-30 relations, only ones the text states. `source` and `target` must be in `entities`.
  Types like WORKS_AT, FOUNDED, CREATED, ADVOCATES, CRITICIZES, USES, WROTE, MEASURES, PART_OF,
  COMPETES_WITH, INVESTED_IN, ACQUIRED, LEADS.
- Extract every document independently from its own text. Never reuse or adapt another
  document's entity list, even for the same guest: two files with the same entity set are a
  defect and will be regenerated.
- Descriptions and evidence are paraphrases. Never copy sentences from the source.
- Valid JSON only: no trailing commas, no comments, no markdown fences.

## 4. Validate, then import
```bash
docker compose run --rm -T graphrag graphrag enrich-import product-leader \
  /app/data/enrichment/a.json /app/data/enrichment/b.json --dry-run     # review report
docker compose run --rm -T graphrag graphrag enrich-import product-leader \
  /app/data/enrichment/*.json                                           # writes graph + snapshot
```
The report lists **unmatched names** (entity name not found verbatim in any passage: fix the
name or drop the entity) and **dangling relations** (endpoint not in `entities`: add it or drop
the relation). Fix the JSON and re-run; import is idempotent (MERGE by entity id).
Re-importing everything is safe and takes seconds.

## 5. Commit
```bash
git add data/enrichment data/snapshots/<persona>
git commit -m "Enrich <persona>: <n> documents"
```
