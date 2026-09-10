# Product Leader persona

Grounded in the [ChatPRD Lenny's Podcast transcript archive](https://github.com/ChatPRD/lennys-podcast-transcripts).

## What is in the snapshot

291 unique episodes (12 archive duplicates are skipped by content fingerprint), 13,335
passages, and an agent-extracted entity layer: 7,200+ entities and 8,000+ typed relations, every
mention anchored to a passage. Extraction files live in `data/enrichment/` (one per episode) and
can be re-imported at any time with `graphrag enrich-import product-leader data/enrichment/*.json`.

## Refresh the grounding

```bash
git clone --depth 1 https://github.com/ChatPRD/lennys-podcast-transcripts data/raw/product-leader
make ingest PERSONA=product-leader SRC=data/raw/product-leader     # ~5 min on CPU
git add data/snapshots/product-leader && git commit -m "Refresh product-leader snapshot"
```

The committed snapshot under `data/snapshots/product-leader/` is what teammates load with
`make setup`; they never need the raw archive.
