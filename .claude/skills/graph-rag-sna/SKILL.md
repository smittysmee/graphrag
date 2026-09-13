---
name: graph-rag-sna
description: Network analysis over the graph-rag knowledge graph - build the speaker, entity, topic and speaker-entity networks, filter them by stance, facet or date window, measure centrality and brokerage, and find groups with Louvain, K-means or Gaussian mixtures, with the stability and null-model checks that make the result reportable. Also reports what the corpus says about an entity and compares two time windows. Use when asked for "network analysis", "SNA", "who is central", "who connects whom", "community detection", "Louvain", "cluster the entities/speakers/topics", "K-means", "Gaussian mixture", "modularity", "who complains about X", "praise vs complaints", "what changed before and after", or "map how these things relate".
---

# graph-rag network analysis

**Repo:** `/Users/asmith/development/graph-rag`. Everything runs in Docker; prefix commands with
`cd /Users/asmith/development/graph-rag &&` when working elsewhere. Full reference:
[docs/SNA.md](../../../docs/SNA.md).

Four networks come out of the graph and they answer different questions. Pick the network
first, then the filters, then the method, then run the checks. Do not report a grouping without
them.

Three filters narrow any network, and each one changes what an edge *means*: `--stance` makes it
"praised together" rather than "discussed together", `--facet` limits it to one function of the
subject, and `--since`/`--until` limit it to dated passages. Say which filter was on in the same
sentence that reports the finding.

## 1. Run it

```bash
docker compose run --rm -T graphrag graphrag sna guide

docker compose run --rm -T graphrag graphrag sna analyze <persona> \
  --network speakers --method louvain --seed 1 \
  --out /app/data/exports/<name>.md --json /app/data/exports/<name>.json

docker compose run --rm -T graphrag graphrag sna analyze <persona> \
  --network entities --method kmeans --features spectral --k-range 2-10 --seed 1 \
  --out /app/data/exports/<name>-kmeans.md

docker compose run --rm -T graphrag graphrag sna export <persona> \
  --network topics --out /app/data/exports/topics.graphml

# only what an annotation pass marked as a complaint, about one function of the subject
docker compose run --rm -T graphrag graphrag sna analyze <persona> \
  --network entities --stance complaint --facet <slug> --min-weight 2 --seed 1 \
  --out /app/data/exports/<name>-complaints.md

# who wrote about what; --project collapses it onto one side
docker compose run --rm -T graphrag graphrag sna analyze <persona> \
  --network speakers-entities --project speakers --seed 1 \
  --out /app/data/exports/<name>-two-mode.md

# what the corpus says about each entity, with the passages behind every count
docker compose run --rm -T graphrag graphrag sna stances <persona> \
  --entity "<Name>" --out /app/data/exports/<name>-stances.md

# the same network over two windows: who entered, who left, what moved
docker compose run --rm -T graphrag graphrag sna compare <persona> --network speakers \
  --since 2024-01-01 --until 2024-12-31 --since2 2025-01-01 --until2 2025-12-31 \
  --seed 1 --out /app/data/exports/<name>-before-after.md
```

Always pass `--seed` so the run can be repeated. Always write under `/app/data/`, because
nothing else in the container is on a mounted volume. Read the markdown the command wrote;
do not paraphrase the numbers from the console summary.

If the speaker network comes back nearly empty, the persona's documents have no `Speaker`
nodes. Run `graphrag attribution-import` first, or use the entities network instead. If a
`--stance` or `--facet` network comes back empty, no annotation pass has run: see the
`graph-rag-enrich` skill. If a window comes back empty, the passages in it carry no dates,
which attribution writes.

## 2. Choosing a network, the filters and a method

### Which network

- **speakers** — nodes are the speakers attached to a persona's documents; an edge is a shared document, weighted by how many they share. Answers: whose voice appears alongside whose, and who bridges otherwise separate groups.
- **entities** — nodes are the entities an extraction pass named; an edge is a shared passage, weighted by how many passages mention both. Answers: what a thing is discussed alongside, and which subjects hang together.
- **topics** — nodes are the topics assigned at ingestion; an edge is the stored co-occurrence weight. Answers: how the corpus was labelled, as a first map before any extraction exists.
- **speakers-entities** — nodes are both the speakers and the entities their passages mention, as two modes; an edge is a passage by that speaker naming that entity, weighted by how many documents hold one. Answers: who wrote about what; projected onto one side, who wrote about the same things, or which things the same people wrote about.

### Which method

**Louvain** (takes edges)

- Use when the object is a graph and the question is which nodes form dense groups. Prefer it over K-means whenever the data are edges rather than vectors: turning a graph into coordinates first throws away information that modularity uses directly.
- Choosing k: no k to choose. The number of communities falls out of the graph, tuned only by --resolution: above 1.0 finds smaller groups, below 1.0 finds larger ones.
- Watch out: it is randomised, so run many seeds and report the stability score; the resolution limit absorbs genuinely small communities into larger ones in a big network; and a high modularity alone is not evidence, because random graphs score above zero too. Check the null model before claiming structure.

**K-means** (takes one vector per node)

- Use when each node already has a feature row, the groups are expected to be compact and roughly equal in spread, and hard membership is the useful answer.
- Choosing k: by silhouette, with the inertia elbow as a second opinion. When the two disagree, prefer silhouette for a claim about groups and say where the elbow pointed.
- Watch out: it fails on elongated or overlapping clusters and on clusters of very unequal size, because it assigns every point to its nearest centre and nothing else.

**Gaussian mixture** (takes one vector per node)

- Use when clusters may overlap, may have different shapes or spreads, or when soft membership is the useful output: a node that belongs 70% to one group and 30% to another.
- Choosing k: by BIC, which penalises extra components harder than AIC does.
- Watch out: the covariance type is a modelling choice and must be stated (full, tied, diagonal or spherical); with few points per component a full covariance degenerates onto single points.

**Spectral embedding** (takes edges, and returns vectors)

- Use when the data are a graph but you want exactly k groups, or want to use a vector method on them. Embed first, then run K-means or a Gaussian mixture on the embedding.
- Choosing k: whatever the vector method that follows it uses.
- Watch out: the embedding has as many dimensions as you ask for and no natural number of them; and it inherits the graph's fragmentation, so a network in many components embeds into clumps that any clusterer will happily separate.

### Which centrality

- **degree** — local prominence: how many distinct others a node sits with.
- **betweenness** — brokerage: how often a node lies on the path between two others.
- **eigenvector or PageRank** — influence among the influential.
- **closeness** — reach: how near a node is to everyone else.
- **weighted variants** — whenever edges carry counts rather than mere presence, which here they always do.

### Signed networks (--stance)

- **The edge means something else now** — a stance filter changes the question from 'discussed together' to 'praised together' or 'complained about together'. Say which one in the same sentence that reports the finding: a reader who sees a co-mention network assumes the unfiltered one.
- **The stances are not complements** — the praise network and the complaint network do not add up to the unfiltered network. A mention nobody annotated is in neither, so what is missing from one is not therefore in the other, and 'not complained about' is never evidence of approval.
- **Compare structure only after comparing n** — a denser complaint network usually means more complaints were annotated, not that complaints cluster harder. Report the node and edge counts of each signed network before saying anything about the shape of either.
- **A stance is a reading of one passage** — it came from an agent reading text, and it describes that passage, not the entity. Quote the passage next to the count; a count nobody can trace back to text that reads that way is an error in the annotation, not a finding.

### Time windows (--since / --until)

- **Report n for each window** — a partition over 40 nodes and a partition over 400 are not two measurements of one thing. Print both counts beside any before-and-after claim.
- **Undated is outside** — a window is answered from dated passages, so every document the attribution pass could not date drops out of every network as soon as a window is set. Check each window against the unwindowed network before reading a disappearance as a change in the world.
- **Small windows overfit** — cut a corpus finely enough and every window has tidy communities, because a handful of documents partitions cleanly. Run the null model inside each window rather than only on the whole corpus.
- **Community numbers do not survive a rebuild** — Louvain numbers its communities per run, so 'community 2 grew' is meaningless across two windows. Compare the partitions with the adjusted Rand index over the nodes both windows contain, and describe groups by their members.
- **A rank change can be a roster change** — a node climbing twenty places may have stood still while the nodes above it left the window. Read the entering and leaving lists before reading the rank table.

### Bipartite projections (--network speakers-entities)

- **Project the side you are asking about** — onto speakers answers who wrote about the same things; onto entities answers which things the same people wrote about. They are different questions and the same two-mode network answers only one of them at a time.
- **Projected weights inflate** — one document naming twenty entities contributes 190 entity pairs on its own, so a projection's weights are combinatorial rather than additive. Raise --min-weight before ranking anything, and never compare a projected weight with a two-mode one.
- **The projection throws the other mode away** — a heavy speaker-speaker edge does not say which entities it ran through. Export the two-mode network beside the projected one, or read the partners recorded on each node, before naming what a group has in common.
- **Sharing a node is not agreement** — two speakers joined by an entity both named it, which includes one recommending it and the other warning against it. Add --stance if the question is whether they agreed.

### Always

- Report n. A centrality ranking over 12 nodes is an anecdote with decimal places.
- Report the null model. Without it, modularity is a number, not a finding.
- Report the stability score. A partition that changes with the seed cannot carry names.
- Report the sampling frame. These networks describe who was recorded and what was written down, never a population; a node is absent when nobody wrote it down, which is not the same as it not existing.
- Say which features were used. Clustering the graph's structure and clustering what the nodes are about answer different questions and can disagree completely.

## 3. Read the report before quoting it

- **nodes** — every ranking below it is only as good as this number. Under about 30 nodes,
  describe the network; do not rank it.
- **stability (adjusted Rand index between Louvain runs)** — above 0.9 you may name members;
  0.6 to 0.9, name groups but not borderline members; below 0.6, report that there is no stable
  community structure instead of naming communities.
- **null model z-score** — the observed modularity against Louvain's best on graphs with the
  same degrees and random edges. Below 2 the partition means nothing. Above 3 it is real.
- **silhouette vs BIC disagreeing** — they answer different questions. Silhouette measures
  separation, BIC measures how many components the data justify. Overlapping groups can score
  well on BIC and badly on silhouette at once. Say which you used and what the other said.
- **spectral vs embedding features disagreeing** — a finding, not a bug. One groups nodes that
  are connected alike, the other groups nodes that are talked about alike.
- **"Across the two modes"** — on a `speakers-entities` report, each group with the speakers in
  it and the entities their passages named, counted in documents. Read it as who wrote about
  what, never as who holds an opinion.
- **a stance report** — every count is a count of annotations. Quote the passages it prints
  under each count; a count that no passage supports is a wrong annotation, not a finding.
- **a comparison** — read n for each window and the entering/leaving lists before the rank
  table, because a climb is often the nodes above leaving.

## 4. Caveats to carry into the write-up

- The network is a sample of a corpus, never of a population. A speaker network describes who
  was recorded; an entity network also describes what the extraction pass chose to name.
  Something absent may simply never have been written down.
- Louvain's resolution limit hides small groups inside large ones. Raise `--resolution` to look
  for them and say that you did.
- Edge weights are counts, so use the weighted centralities; the code already reads weight as
  closeness rather than as distance, but any hand-rolled path calculation must do the same.
- `--features embedding` only works on the entities network, because it needs text per node.
- A filtered network describes what somebody annotated, not what exists. The passages nobody
  annotated are unread, not neutral and not the opposite stance.
- A window is built from dates an attribution pass wrote. An undated document is in no window,
  so a disappearance between two windows may be a missing date rather than a change.
- Projected weights on the two-mode network are combinatorial: one document naming twenty
  entities makes 190 pairs on its own. Raise `--min-weight` before ranking anything.

## 5. Write the result into a corpus

The markdown from `--out` is already a research note: it carries the rationale, the checks and
the caveats. To make the finding citable, put it under `data/raw/<persona>/<source>/` with a
title and a date, re-ingest with `make sync PERSONA=<persona>`, and keep the `--json` beside it
so the soft memberships, the full k table and the null distribution survive the summary. State
in the note which command produced it, including the seed.
