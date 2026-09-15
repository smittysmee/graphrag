#!/usr/bin/env python3
"""Validate a captured research corpus and optionally merge its provenance manifests.

Usage: python scripts/validate_corpus.py <corpus-root> [--merge]
Example: python scripts/validate_corpus.py data/raw/<persona>

Checks that every captured document opens with parsable YAML front-matter, carries the required
keys, names a source_url unless it is a research note, and holds a body of 40 to 6,000 words;
then that every provenance line is valid JSON pointing at a file that exists. Exits 1 on any
problem. `--merge` writes provenance/provenance.jsonl from the per-researcher manifests.

The logic lives in `graphrag.ingest.validate`; this is a wrapper so the corpus can be checked
without the CLI. `graphrag validate <path> [--merge]` and `make validate CORPUS=<path>` do the
same thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag.ingest.validate import merge_provenance, validate_corpus  # noqa: E402


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    root = Path(args[0])
    report = validate_corpus(root)
    for line in report.problem_lines():
        print(line)
    if "--merge" in argv:
        out = merge_provenance(root, report.provenance)
        print(f"merged {len(report.provenance)} provenance records -> {out}")
    for line in report.summary_lines():
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
