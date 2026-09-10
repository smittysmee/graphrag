# Notices and attribution

The MIT license in `LICENSE` covers **this project's own source code, tests, docs and
configuration**. It does not and cannot cover the third-party material below.

## Source content is not included

This repository ships **no ingested content and no prebuilt graph**. The example persona,
`product-leader`, is configured to read from an archive that you fetch yourself:

- **[ChatPRD/lennys-podcast-transcripts](https://github.com/ChatPRD/lennys-podcast-transcripts)**,
  referenced as a git submodule at `data/raw/product-leader` — a pointer, not a copy.
  That archive states it is provided for personal and educational use, and that the content
  belongs to **[Lenny's Podcast](https://www.youtube.com/@LennysPodcast)** and the respective
  guests. It carries no open-source license.

If you ingest it, the resulting graph contains that content. **Do not redistribute the resulting
snapshot**, publicly or otherwise, and please support the original creators. If you want a graph
you can share, point this tool at material you own or that is licensed for redistribution.

## Embedding model

The default embedder downloads `BAAI/bge-small-en-v1.5` (MIT) from Hugging Face, via the
`qdrant/bge-small-en-v1.5-onnx-q` conversion, pinned by revision and SHA-256 in
`vendor/model.lock`. No model weights are redistributed here; `make model` fetches them.

## Runtime dependencies

Python dependencies are Apache-2.0 (fastmcp, the Neo4j driver, fastembed), MIT (pydantic,
typer, rich, anthropic, PyYAML, onnxruntime) and BSD-3-Clause (httpx, starlette, uvicorn,
pypdf, numpy). See `uv.lock` for the exact set.

**Neo4j Community Edition is GPL v3.** This project does not bundle, modify or link against it.
`docker-compose.yml` references the public image by tag, which Docker pulls on your machine, and
this code communicates with it over the Bolt network protocol using the Apache-2.0 driver.
