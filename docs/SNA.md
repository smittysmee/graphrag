# Network analysis

`graphrag sna` turns the graph into networks, measures them, groups them, and checks whether the
grouping means anything. It is built on `networkx` (graphs, Louvain, modularity) and
`scikit-learn` (K-means, Gaussian mixtures, spectral embedding, adjusted Rand index, silhouette).
Nothing here re-implements an algorithm those libraries provide.

The code lives in `src/graphrag/sna/`: `export.py` builds the networks, `measures.py` measures
them, `cluster.py` groups them and checks the grouping, `guide.py` holds the selection rules,
`analysis.py` runs one analysis end to end and renders the report, `stances.py` reports what the
corpus says about each entity, and `compare.py` runs one network over two time windows.

Four networks come out of the graph. Three are one-mode -- speakers, entities, topics -- and
`speakers-entities` is two-mode: speakers on one side, the entities their passages mention on the
other, which `--project` collapses onto either side. Three filters narrow any of them, and each
changes what an edge *means* rather than only how many there are, so the report prints a sentence
saying so next to the numbers:

- `--stance praise|complaint|substitution|neutral` keeps only mentions the annotation pass
  marked, so an edge becomes "praised together" instead of "discussed together".
- `--facet <slug>` keeps only passages about one function of the subject.
- `--since` / `--until` keep only passages dated into the window, which drops every document
  nobody dated.

## Commands

```bash
# print the selection rules below
graphrag sna guide

# write a network to GraphML or node-link JSON
graphrag sna export <persona> --network speakers --out /app/data/exports/speakers.graphml
graphrag sna export <persona> --network entities --min-weight 3 --types person,company \
    --out /app/data/exports/entities.json

# measure, group, check, and write a markdown report (plus optional JSON)
graphrag sna analyze <persona> --network speakers --method louvain --seed 1 \
    --out /app/data/exports/speakers.md --json /app/data/exports/speakers.json
graphrag sna analyze <persona> --network entities --method kmeans --features spectral \
    --k-range 2-10 --seed 1 --out /app/data/exports/entities-kmeans.md
graphrag sna analyze <persona> --network entities --method gmm --features embedding \
    --covariance diag --seed 1 --out /app/data/exports/entities-gmm.md

# only the passages an annotation pass marked as complaints, about one function of the subject
graphrag sna analyze <persona> --network entities --stance complaint --facet quoting \
    --min-weight 2 --seed 1 --out /app/data/exports/complaints.md

# who wrote about what, then collapsed onto one side
graphrag sna export <persona> --network speakers-entities \
    --out /app/data/exports/two-mode.graphml
graphrag sna analyze <persona> --network speakers-entities --project speakers --seed 1 \
    --out /app/data/exports/who-writes-alike.md

# what the corpus says about each entity, with the passages behind every count
graphrag sna stances <persona> --entity "Some Product" --facet quoting \
    --out /app/data/exports/stances.md

# the same network over two windows: who entered, who left, what moved
graphrag sna compare <persona> --network speakers \
    --since 2024-01-01 --until 2024-12-31 --since2 2025-01-01 --until2 2025-12-31 \
    --seed 1 --out /app/data/exports/before-and-after.md
```

Everything runs in Docker, so write outputs under `/app/data/...`; anything else vanishes when
the container exits and the CLI warns you about it.

| option | applies to | what it does |
|---|---|---|
| `--network` | export, analyze, compare | `speakers`, `entities`, `topics` or `speakers-entities` |
| `--source` | all | limit to one of the persona's sources |
| `--min-weight` | export, analyze, compare | drop edges below this weight (default 1 for speakers and the two-mode network, 2 otherwise) |
| `--types` | entities, speakers-entities | keep only these entity types, comma separated |
| `--stance` | entities, speakers-entities | keep only mentions with this stance; repeat for several |
| `--facet` | all networks, stances | keep only passages carrying this facet; repeat for several |
| `--since`, `--until` | all networks | keep only passages dated in this window, ISO `YYYY-MM-DD` |
| `--project` | speakers-entities | collapse the two-mode network onto `speakers` or `entities` |
| `--entity` | stances | limit the report to this entity by name; repeat for several |
| `--quotes` | stances | verbatim passages to quote per stance (default 3) |
| `--since2`, `--until2` | compare | the second window |
| `--centrality` | compare | which ranking the rank changes are read from (default `weighted_degree`) |
| `--method` | analyze | `louvain`, `kmeans` or `gmm` |
| `--features` | kmeans, gmm | `spectral` (graph structure) or `embedding` (what the nodes are about) |
| `-k` | kmeans, gmm | force k instead of choosing it |
| `--k-range` | kmeans, gmm | candidates, `2-10` or `3,5,8` |
| `--resolution`, `--runs` | louvain | resolution parameter; how many seeds to compare |
| `--covariance` | gmm | `full`, `tied`, `diag` or `spherical` |
| `--samples` | louvain | null-model rewirings (each one re-runs Louvain) |
| `--seed` | analyze | fixes every random seed, making the run reproducible |

`--features embedding` needs text behind each node, so it works on the entities network only;
every other network has `--features spectral`.

A filter is never silently absorbed. `--stance` is refused on the speakers and topics networks,
which hold no mentions to annotate, and `--project` is refused on anything but the two-mode
network. The speaker network is built per document, so `--facet` there keeps documents holding
such a passage rather than the passages themselves; the topic network is a stored aggregate, so
`--facet` and a window make it recompute over the surviving documents, which gives smaller
weights that are not comparable with the stored ones.

## Reading a report

**The summary table first.** `nodes` is the n every other number depends on. A high
`components` count with a low `largest_component_share` means the network is really several
networks, and a centrality ranking across them compares nodes that cannot reach each other.

**Stability (adjusted Rand index between Louvain runs).** 1.0 means every seed found the same
partition. Above 0.9 you can name individual members. Between 0.6 and 0.9 the large groups are
real but the boundaries move, so quote groups and not the membership of any one node. Below
0.6, report that there is no stable structure rather than naming communities.

**The null model (z-score).** The observed modularity is compared with what Louvain achieves on
graphs that have the same degree sequence but random edges. Below 2, the partition is what any
graph of this shape would produce and means nothing. Above 3, the structure is real.

**Silhouette and BIC disagreeing.** They are answering different questions. Silhouette asks how
well separated the clusters are given the assignment; BIC asks how many Gaussian components the
data justify, penalising each extra one. Overlapping groups can be well modelled by a mixture
(good BIC) and badly separated (poor silhouette) at the same time. Say which criterion you used
and what the other one said.

**Structure clusters and content clusters disagreeing.** That is a finding, not an error.
`--features spectral` groups nodes that are connected in the same way; `--features embedding`
groups nodes that are talked about in the same way. Two entities can be mentioned in the same
passages constantly while meaning entirely different things.

**"Across the two modes" on a `speakers-entities` report.** Each group with the speakers in it
and the entities their passages named, in documents. A projection throws the other mode away, so
this is read from the partners recorded on each node when the network was built, not from the
graph you are looking at.

**The stance report.** Every number in it counts annotations, not mentions. An entity with no
complaints has no annotated complaints, which is not the same as no complaints, and the
quotations under each count are the evidence: if a passage does not read the way its count
claims, the annotation is wrong.

## Choosing a network and a method

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

## Caveats that do not go away

- These networks are a sample of a corpus, not of a population. A speaker network describes who
  was recorded; an entity network additionally describes what an extraction pass chose to name.
  Absence is evidence that nobody wrote something down, which is not the same as it not existing.
- Edge weights here are affinities: a heavier edge means two nodes are closer. `networkx` reads
  an attribute called `weight` as a distance, so betweenness and closeness run on a derived
  `1 / weight` instead. If you compute a shortest path yourself, do the same.
- Louvain's resolution limit hides small communities inside large ones however many seeds you
  run. Raise `--resolution` above 1.0 to look for them, and report that you did.
- The null model costs one Louvain run per sample, so `--samples 50` on a large network is the
  slow part of `analyze`. Lower it while exploring and raise it for the run you publish.
- A stance or a facet filter narrows the corpus to what somebody annotated. The unannotated
  remainder is not the opposite stance and not the other facet; it is unread.
- A window is answered from the dates an attribution pass wrote onto passages. A corpus with no
  dates produces an empty window rather than an error, which is why `compare` prints n first.

## Writing the result up

`analyze --out` already produces a markdown note with the rationale, the checks and the
caveats. To keep it, write it into a persona's corpus under `data/raw/<persona>/...` and
re-ingest, so the finding becomes citable the same way any other source is. Keep the JSON from
`--json` next to it: it holds the responsibilities, the full k table and the null distribution
that the note summarises.
