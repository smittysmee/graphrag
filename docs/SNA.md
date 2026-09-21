# Network analysis

`graphrag sna` turns the graph into networks, measures them, groups them, and checks whether the
grouping means anything. It is built on `networkx` (graphs, Louvain, modularity) and
`scikit-learn` (K-means, Gaussian mixtures, spectral embedding, adjusted Rand index, silhouette).
Nothing here re-implements an algorithm those libraries provide.

The code lives in `src/graphrag/sna/`: `export.py` builds the networks, `measures.py` measures
them, `cluster.py` groups them and checks the grouping, `attributes.py` measures a network
against a property its nodes carry, `guide.py` holds the selection rules, `analysis.py` runs one
analysis end to end and renders the report, `stances.py` reports what the corpus says about each
entity, `compare.py` runs one network over two builds, and `generators.py` holds the
synthetic networks of Atlas ch. 16-18 and the "Against random" comparison every report
carries.

Five networks come out of the graph. Three are one-mode -- speakers, entities, topics --
`speakers-entities` is two-mode: speakers on one side, the entities their passages mention on the
other, which `--project` collapses onto either side; and `relations` is the directed one, built
from the typed `RELATED_TO` edges an extraction pass wrote between entities. Filters narrow any
of them, and each changes what an edge *means* rather than only how many there are, so the report
prints a sentence saying so next to the numbers:

- `--stance praise|complaint|substitution|neutral` keeps only mentions the annotation pass
  marked, so an edge becomes "praised together" instead of "discussed together".
- `--facet <slug>` keeps only passages about one function of the subject.
- `--since` / `--until` keep only passages dated into the window, which drops every document
  nobody dated.
- `--relation-type <type>` keeps only relations of that type on the `relations` network, so
  the network is about one kind of tie alone. Matched as stored, not case-folded: an extraction
  pass names its relation types (`COMPETES_WITH`, `USES`, ...), typically upper case, and
  `--relation-type competes_with` (lower case) on a corpus that stores `COMPETES_WITH` matches
  nothing and builds a silently empty network rather than erroring -- check the type's exact
  casing first, e.g. with `sna export --network relations` and reading the edges back.
- `--where key=value` keeps only nodes attributed that way: a speaker by what an attribution
  pass recorded about them, an entity by what an extraction pass recorded about the entity, and
  a node with no value of its own by the attributes of the document its passage belongs to. A
  node with no value for that key at all is out of the network, not in some other group.

`graphrag sna analyze --by key` goes further and asks whether the network *divides* along one of
those attributes, which is a different question from what any filter answers. See
[Reading a report](#reading-a-report) and the node-attribute rules below.

[SNA_FOUNDATIONS.md](SNA_FOUNDATIONS.md) is the background this file assumes: which probability
and statistics conventions the package uses, where the book's notation lives in `sna/`, and which
chapters of *The Atlas for the Aspiring Network Scientist* are documented with a reason rather
than built.

Everything below is `graphrag sna` from a terminal. The same package is also
`graphrag.sna.Network` -- `Network.build(...)`, `.analyze()`, `.backbone()`, `.measures()`,
`.evaluate()`, `.predict()`, `.draw()`, `.to_pandas()` -- one call per line of `sna/facade.py`,
each calling the exact function the matching command calls, for a notebook cell that wants an
object back instead of a rendered report. [SNA_NOTEBOOK.md](SNA_NOTEBOOK.md) is the page-length
tour.

Every command below is also an MCP tool, for an agent rather than a terminal or a notebook:
`sna_<command>` (`sna multilayer-communities` → `sna_multilayer_communities`, hyphens to
underscores), the same flags as typed arguments -- `--where key=value` as a `where` mapping,
a repeated `--stance`/`--facet`/`--relation-type`/`--types` as a list -- returning
`{"markdown": ..., "payload": ...}` in one call: the report text the CLI prints or writes to
`--out`, and the structure `--json` would write. `out`/`as_json` (and `sna_draw`'s `html`/
`report`) are optional even on the commands where the CLI requires them, and also write the file
when given one; nothing is written unless asked to be. A refusal the CLI exits 2 for -- an
unknown persona, network or method, `sna_complete` without the `graphrag[gnn]` extra -- comes
back as `{"error": "..."}` instead. `.claude/skills/graph-rag-sna/SKILL.md` §1 is the short form
of this; `src/graphrag/mcp_server.py`'s own module docstring is the implementation note.

## Commands

```bash
# print the selection rules below
graphrag sna guide

# write a network to a file; the suffix picks the format (see "Interchange formats" below)
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

# what the corpus asserts about its entities, as a directed network (Atlas ch. 6)
graphrag sna export <persona> --network relations --out /app/data/exports/relations.graphml
graphrag sna analyze <persona> --network relations --relation-type COMPETES_WITH --seed 1 \
    --out /app/data/exports/relations.md

# who wrote about what, then collapsed onto one side
graphrag sna export <persona> --network speakers-entities \
    --out /app/data/exports/two-mode.graphml
graphrag sna analyze <persona> --network speakers-entities --project speakers --seed 1 \
    --out /app/data/exports/who-writes-alike.md

# what the corpus says about each entity, with the passages behind every count
graphrag sna stances <persona> --entity "Some Product" --facet quoting \
    --out /app/data/exports/stances.md

# one population of the corpus, and whether the network divides along that attribute
graphrag sna analyze <persona> --network speakers --where region=north --seed 1 \
    --out /app/data/exports/north.md
graphrag sna analyze <persona> --network speakers --by region --seed 1 \
    --out /app/data/exports/by-region.md

# the same question about a number rather than a label (Atlas ch. 31): does this network
# join like to like by how connected its nodes are, or by a declared numeric attribute
graphrag sna analyze <persona> --network entities --by mentions --seed 1 \
    --out /app/data/exports/by-mentions.md

# one node's neighbourhood, with and without the node (Atlas §30.1)
graphrag sna ego <persona> "product-market fit" --network entities --by stage \
    --out /app/data/exports/ego-pmf.md

# two populations as two builds, compared the way two windows are
graphrag sna compare <persona> --network entities --min-weight 2 \
    --where region=north --where2 region=south --seed 1 \
    --out /app/data/exports/north-vs-south.md

# the same network over two windows: who entered, who left, what moved
graphrag sna compare <persona> --network speakers \
    --since 2024-01-01 --until 2024-12-31 --since2 2025-01-01 --until2 2025-12-31 \
    --seed 1 --out /app/data/exports/before-and-after.md

# what each of chapter 26's projection schemes does to one network, before choosing one
graphrag sna projections <persona> --network entities --min-weight 1 \
    --out /app/data/exports/projections.md

# then the analysis under the scheme that was chosen; the frame names it
graphrag sna analyze <persona> --network entities --projection hyperbolic --seed 1 \
    --out /app/data/exports/entities-hyperbolic.md

# what each of chapter 27's backbones would keep, before choosing one
graphrag sna backbone <persona> --network entities --min-weight 1 \
    --out /app/data/exports/backbones.md

# then the analysis over the backbone that was chosen, with its p-value on every edge
graphrag sna analyze <persona> --network entities --backbone noise-corrected --alpha 0.05 \
    --seed 1 --out /app/data/exports/entities-nc.md

# how the degrees distribute, and whether that distribution is a power law (Atlas ch. 9)
graphrag sna degree <persona> --network entities --min-weight 2 --seed 1
graphrag sna degree <persona> --network relations --kind in --bootstrap 500 --seed 1 \
    --out /app/data/exports/in-degree.md --json /app/data/exports/in-degree.json

# every ranking as a mean and an interval over sampled possible worlds (Atlas ch. 28)
graphrag sna analyze <persona> --network entities --uncertain --uncertain-samples 200 \
    --seed 1 --out /app/data/exports/entities-uncertain.md

# a sample of a network that is too big to look at, with what that sampler distorts
graphrag sna sample <persona> --network entities --method random-walk --size 400 --seed 1 \
    --out /app/data/exports/entities-sample.json
graphrag sna sample <persona> --network entities --method induced --param neighbors=true \
    --size 100 --seed 1 --out /app/data/exports/induced-with-neighbourhood.json
graphrag sna sample <persona> --network entities --method forest-fire --param p=0.3 \
    --size 400 --seed 1 --out /app/data/exports/fire.json --json /app/data/exports/fire-bias.json

# the whole analysis, run on a sample, with the frame saying so
graphrag sna analyze <persona> --network entities --sample bfs --sample-size 400 --seed 1 \
    --out /app/data/exports/sampled.md
# the same comparison as a window grid: six-month windows, first against last (Atlas §7.4)
graphrag sna compare <persona> --network speakers \
    --since 2024-01-01 --until 2025-12-31 --window 6M --seed 1 \
    --out /app/data/exports/by-half-year.md

# one network as several layers, and what each layer holds (Atlas §7.2, §8.1)
graphrag sna layers <persona> --network entities --layers stance
graphrag sna analyze <persona> --network entities --layers facet --min-weight 1 --seed 1 \
    --out /app/data/exports/by-facet.md

# what a random walker between two nodes costs: hitting, commute, resistance, cut (Atlas ch. 11)
graphrag sna walks <persona> --network entities --from "product-market fit" --to "pricing"

# what each node *is*: roles, node similarity, and who plays the same role apart (Atlas ch. 15)
graphrag sna roles <persona> --network entities --min-weight 2 --seed 1 \
    --out /app/data/exports/roles.md

# whether the stated relations form a hierarchy, by four measures that disagree (Atlas ch. 33)
graphrag sna hierarchy <persona> --seed 1
graphrag sna hierarchy <persona> --relation-type COMPETES_WITH --samples 200 --seed 1 \
    --out /app/data/exports/hierarchy.md --json /app/data/exports/hierarchy.json
# the passages as simplices, and a walk that remembers where it came from (Atlas ch. 34)
graphrag sna highorder <persona> --max-dim 3 --out /app/data/exports/highorder.md
# what a link predictor would have to beat, and on what (Atlas ch. 25)
graphrag sna predict-eval <persona> --network entities --temporal 2025-01-01 --k 20 --seed 1 \
    --out /app/data/exports/predict-eval.md
graphrag sna predict-eval <persona> --network entities --share 0.1 --negatives 5 --seed 1 \
    --json /app/data/exports/predict-eval.json
# suggest a value for entities missing one, from structure and text (Atlas ch. 44, needs
# graphrag[gnn]); nothing here is written to the graph
graphrag sna complete <persona> region --method gcn --seed 1 \
    --out /app/data/exports/region-suggestions.md
# what breaks this network: random failure, targeted attack, cascade, coupling (Atlas ch. 22)
graphrag sna robustness <persona> --network entities --seed 1 \
    --out /app/data/exports/robustness.md
# force-directed, sized and coloured, as an SVG and a self-contained interactive HTML
# (Atlas ch. 49-51 -- see "Visualization" below)
graphrag sna draw <persona> --network entities --min-weight 2 --layout force \
    --size betweenness --color community --seed 1 --out /app/data/exports/network.svg
```

Everything runs in Docker, so write outputs under `/app/data/...`; anything else vanishes when
the container exits and the CLI warns you about it.

| option | applies to | what it does |
|---|---|---|
| `--out`, `--json` | every report command | both optional (ATL-F1): write the report as markdown / as JSON, on top of what the console already printed. `sample`, `export` and `draw` write their one primary artifact (a sampled graph, an interchange file, an SVG) to `--out` instead, which stays required there; `guide` takes neither. |
| `--network` | export, analyze, compare | `speakers`, `entities`, `topics` or `speakers-entities` |
| `--source` | all | limit to one of the persona's sources |
| `--min-weight` | export, analyze, compare | drop edges below this weight (default 1 for speakers, the two-mode network and relations, 2 otherwise). This is the Atlas's naive backbone (§27.1) and the chapter argues against it; see Backboning below |
| `--backbone` | export, analyze, compare | filter the network with one of chapter 27's methods, after `--min-weight` |
| `--alpha` | `--backbone disparity`/`noise-corrected` | significance level for the edge test (default 0.05) |
| `--correction` | as `--alpha` | multiple-test correction over the edge p-values: `none`, `bonferroni`, `holm`, `fdr_bh` |
| `--threshold` | the structural `--backbone` methods | the level in that method's own units, replacing its default (for `naive-top`, n) |
| `--types` | entities, speakers-entities | keep only these entity types, comma separated |
| `--stance` | entities, speakers-entities | keep only mentions with this stance; repeat for several |
| `--facet` | all networks, stances | keep only passages carrying this facet; repeat for several |
| `--since`, `--until` | all networks | keep only passages dated in this window, ISO `YYYY-MM-DD` |
| `--where` | all networks, stances, compare | keep only nodes attributed `key=value`; repeat for several keys |
| `--where2` | compare | the second build's attribute filter (defaults to `--where`) |
| `--by` | analyze | also report whether the network divides along this node attribute |
| `--permutations` | analyze `--by` | label shuffles behind the assortativity null (default 200) |
| `--null` | analyze `--by` | the second null: `permutation` (default, shuffle the labels) or `bipartite` (rewire the memberships this network was projected from and project again) |
| `--project` | speakers-entities | collapse the two-mode network onto `speakers` or `entities` |
| `--entity` | stances | limit the report to this entity by name; repeat for several |
| `--quotes` | stances | verbatim passages to quote per stance (default 3) |
| `--since2`, `--until2` | compare | the second window |
| `--window`, `--step` | compare | cut `--since`..`--until` into a grid (`90`, `90D`, `6M`, `2Y`) and compare its first window with its last |
| `--layers` | export, analyze, layers | build one layer per `stance`, `facet`, `source` or `relation-type` and work on their flattening |
| `--omega` | layers, export/analyze `--layers` | the interlayer coupling recorded with the network (default 1.0) |
| `--centrality` | compare | which ranking the rank changes are read from (default `weighted_degree`) |
| `--method` | analyze | `louvain`, `kmeans` or `gmm` |
| `--features` | kmeans, gmm | `spectral` (graph structure) or `embedding` (what the nodes are about) |
| `-k` | kmeans, gmm | force k instead of choosing it |
| `--k-range` | kmeans, gmm | candidates, `2-10` or `3,5,8` |
| `--resolution`, `--runs` | louvain | resolution parameter; how many seeds to compare |
| `--covariance` | gmm | `full`, `tied`, `diag` or `spherical` |
| `--samples` | louvain | null-model rewirings (each one re-runs Louvain) |
| `--seed` | analyze, sample | fixes every random seed, making the run reproducible |
| `--method` | sample | `induced`, `edge`, `ties`, `bfs`, `snowball`, `forest-fire`, `random-walk`, `metropolis-hastings` or `neighbor-reservoir` |
| `--size` | sample | how many nodes the sample holds |
| `--param` | sample | a sampler's own parameter as `name=value`: `k` (snowball, default 3), `p` (forest fire, 0.5), `restart` (random walk, 0.15), `rounds` (neighbor reservoir, 10), `neighbors=true` (induced: collect each selected node's whole neighbourhood, which returns more nodes than `--size`) |
| `--sample`, `--sample-size` | analyze, degree | run the analysis on a sample of the network instead |
| `--kind` | degree | `degree`, `in`, `out`, `weighted`, `weighted_in` or `weighted_out` (Atlas §9.1) |
| `--bootstrap` | degree | synthetic data sets behind the power-law goodness-of-fit p-value (default 200) |
| `--max-dim` | highorder | how far the passage hypergraph is closed downward into simplices (default 3, Atlas §34.1) |
| `--damping`, `--exact` | highorder | the teleport factor both walks run with (default 0.85), or the undamped π, which fails where the chain has none |
| `--layout` | draw | `force`, `circular`, `arc`, `matrix` or `layered` (Atlas ch. 51; `layered` needs `--network relations`) |
| `--size` | draw | a centrality name; maps onto node *area*, quasi-logged (Atlas §49.1); uniform radius when unset |
| `--color` | draw | `community` (Louvain) or `attr:<key>`; uniform fill when unset (Atlas §49.2) |
| `--html`, `--report` | draw | override the paired `.html` path (default: `--out` with `.html`), or also write the markdown section |
| `--no-cache` | every command that builds a network | skip the on-disk network cache (ATL-ENT-3); see "Scale" below |

`--features embedding` needs text behind each node, so it works on the entities network only;
every other network has `--features spectral`.

A filter is never silently absorbed. `--stance` is refused on the speakers and topics networks,
which hold no mentions to annotate, and `--project` is refused on anything but the two-mode
network. The speaker network is built per document, so `--facet` there keeps documents holding
such a passage rather than the passages themselves; the topic network is a stored aggregate, so
`--facet` and a window make it recompute over the surviving documents, which gives smaller
weights that are not comparable with the stored ones.

## Scale (ATL-ENT-3)

Every command above builds a network by reading the store and projecting it (`sna/export.py`,
`build_network`); on the largest persona this repository ships (13k chunks) that read, not the
analysis after it, is usually the slow part. Two things keep it from being paid twice, and a
third keeps the eigendecompositions after it from staying dense past the point where that helps:

- **The on-disk network cache** (`sna/cache.py`) keys a built network on the persona, the
  network kind, every filter, the persona's committed snapshot identity
  (`data/snapshots/<persona>/manifest.json`, its `source_commit` where one is recorded) and a
  live fingerprint read from the store on every build (document, chunk and mention counts, plus
  how many nodes carry an attribute property). The same command run twice with the same filters
  against unchanged data is served from `GRAPHRAG_SNA_CACHE_DIR` (default `data/.cache/sna`)
  without touching the store for the network itself; a different filter or persona misses, as
  does a write. **When the cache actually invalidates, in order of how it notices:**
  1. *Explicit, on every write path* (`graphrag ingest`, `sync`, `enrich`, `enrich-import`,
     `attribution-import`, `annotations-import`, `aliases apply`, `entities prune`, `snapshot
     load`, `persona import`, `setup`): each clears that persona's cache entries as its own last
     step, whether or not the write moved a count the fingerprint below would have caught --
     costs nothing on a read that follows, and a persona nothing wrote to keeps every entry.
  2. *The store fingerprint, read fresh on every build*: a write none of those commands made --
     Cypher run by hand against Neo4j, a script importing `graphrag.graph.neo4j_store` directly
     -- still misses if it moved the document, chunk or mention count, or the count of nodes
     carrying an attribute property (which a property-only reimport, adding or correcting a
     value without adding a document or a mention, does move).
  3. **What neither catches**: a write that changes a value without changing any of those four
     counts (correcting an attribute a node already carried a value for, say) is invisible to
     both until the persona's cache is cleared some other way.
  Pass `--no-cache` on any one command to bypass the cache for that call, or
  `GRAPHRAG_SNA_CACHE=false` to turn it off everywhere; `graphrag sna cache clear` empties every
  persona's entries, unconditionally. A persona with no committed snapshot (an in-memory store,
  one still being ingested) still caches -- the fingerprint and the explicit `clear()` calls are
  what keep it correct there, since there is no snapshot identity to fall back on.
- **Sparse matrices** (`sna/matrices.py`) back the adjacency, stochastic and Laplacian forms
  whenever a caller asks for `sparse=True`, and give bit-identical numbers to the dense forms on
  every fixture this package tests against. Above `matrices.SPARSE_ABOVE_NODES` (2,000) nodes,
  `sna/walks.py` routes three quantities through the sparse form: the directed
  `stationary_distribution`'s power iteration multiplies a sparse transition matrix rather than a
  dense one, and `fiedler_vector`/`consensus_time` -- which only ever want the second-smallest
  eigenpair -- ask `matrices.eigenpairs(..., sparse=True)` for it, which solves with
  `scipy.sparse.linalg.eigsh` instead of diagonalising the whole Laplacian. A quantity that needs
  the *whole* spectrum (`random_target_time`) stays dense regardless of size, and says so in its
  own docstring.
- **`tests/benchmarks/test_sna_budget.py`**, marked `slow` and left out of `make check`, times
  a curated set of commands against the largest loaded persona's entity network and prints the
  table next to the budget each row was held to. Run it with `make bench` once `make setup` has
  loaded a snapshot; it skips, rather than fails, when there is nothing loaded to time.

## Degree, and the word "scale-free"

`sna degree` is Atlas ch. 9, and the same section is printed inside every `sna analyze` report.
It prints the degree sequence's summary with §3.1's caveat about its mean, the distribution as
the chapter asks for it -- p(k), a log-binned histogram and the CCDF p(K >= k), which is the one
to read because it stays a function where the scatter plot falls apart -- and then the test.

The test is the Clauset-Shalizi-Newman procedure the book points at, and every part of it is
printed: the exponent by maximum likelihood on the discrete distribution (never a regression on
the log-log plane, which fits a straight line to a straight-ish line and rewards you with a
magnificent R-squared), an `xmin` chosen as the value with the smallest Kolmogorov-Smirnov
distance, a goodness-of-fit p-value from `--bootstrap` synthetic data sets drawn from the fit
itself and refitted from scratch, and likelihood-ratio tests against the lognormal and the
exponential over that same tail.

The verdict withholds the word, at the bar the paper §9.4 sends you to sets. Clauset, Shalizi
and Newman rule a power law out at **p <= 0.1**, not at 0.05, so `scale-free` is printed only
when the bootstrap p clears 0.1 *and* the lognormal does not fit significantly better. Both
levels are printed beside the p-value, and there is a verdict for the band between them:

| verdict | what it means |
|---|---|
| `scale-free` | p > 0.1 and no alternative is significantly better |
| `heavy-tailed, power law not rejected but evidence weak` | 0.05 < p <= 0.1: not rejected, not plausible either. A textbook Barabási-Albert graph of 3,000 nodes lands here, because its degrees follow 2m(m+1)/(k(k+1)(k+2)) — a *shifted* power law whose head a KS test over a large tail detects |
| `heavy-tailed but not distinguishable from lognormal` | p <= 0.05 and the two models cannot be told apart |
| `lognormal rather than a power law` | the lognormal fits significantly better |
| `exponential/Poisson-like` | the exponential fits significantly better, which §9.4 says settles it |
| `heavy-tailed but not a power law`, `neither a power law nor a heavy tail` | the fit is rejected, with and without §3.1's heavy tail |
| `two modes, measured separately` | a bipartite network: the answer is per mode, below |
| `too few nodes to test`, `too few nodes in the tail to test`, `not counts, so the discrete test does not apply` | nothing was fitted, and why |

The exponential is checked first, because §9.4 checks it first: "if you think a distribution
might be an exponential, then it's definitely not a power law". Even a `scale-free` verdict says
that the lognormal could not be excluded, which is the usual outcome, and that the book then
wants an argument from cumulative advantage rather than another number.

Below 50 nodes with a degree, nothing is fitted at all -- the karate club gets "too few nodes to
test" rather than an exponent -- and a tail carried by fewer than four distinct degree values is
refused for the same reason: a regular graph fits a power law perfectly, because a single value
is a single point. On a directed network `--kind in` and `--kind out` are two different
distributions and the plain `degree` is their total (§9.1); `weighted` is node strength, and a
strength that is not a whole number gets §3.2's continuous fits instead of the discrete test.

A two-mode network has **two** degree sequences and §9.1 says they are not comparable: they sum
to the same number of edges over different numbers of nodes, so their means differ by
construction and the `xmin` sweep on their union finds the gap between the modes rather than the
shape of either tail. `sna degree` therefore fits each mode separately, prints a verdict per
mode, and marks the whole network `two modes, measured separately`; the mixed distribution is
still printed, to be looked at rather than fitted. When the network is a sample, the §29.3
re-weighted estimate of the population's distribution is printed beside the sample's own, with a
sentence saying which is which. On a flattened multilayer network the report says that §9.1's
multilayer question is the degree *per layer*, which is `--layers` and chapter 7's object, not
this distribution.

## Quantitative assortativity (Atlas ch. 31)

Chapter 30 asks whether an edge joins two nodes with the same *label*. Chapter 31 asks the same
of a *number*, and the change of question changes the arithmetic: two labels are shared or not,
while two numbers are near or far — "if two nodes of values 1 and 2 connect, it is true that they
have a different attribute value, but it still counts more towards assortativity than connecting
a node of value 1 to a node of value 100" (p. 441). So the measure is a correlation over the
edges rather than a matching rate.

`--by <key>` picks which chapter runs, and the rule has three branches. The counts every network
already carries — `degree`, `mentions`, `documents`, `chunks` — always get chapter 31. A key the
persona declared in `facets.yaml` gets whichever its declaration says: `type: number` is
quantitative, and a key with `values:` is categorical even when every one of those values is a
numeral, because a closed list of numerals is a set of labels. A key **nothing declares** falls
back to its values: if every value parses as a finite number over at least two distinct numbers,
it is read as a quantity and chapter 31 runs. That last branch is a guess — an undeclared `band`
holding `1` and `2` as codes would be correlated rather than counted — so the section says out
loud that nothing declared the key and tells you to declare it with `values:` if they are codes.
Everything else gets chapter 30. A numeric key is declared like this:

```yaml
attributes:
  tenure_years:
    type: number
    min: 0
    max: 60
    description: How long the speaker says they have been doing this.
```

Values are validated per entry as finite numbers inside the declared range, and stored as the
string the sidecar wrote, so `--where tenure_years=3.5` still matches by exact string (`3.50` is
a different filter, and a range filter does not exist).

The section prints four things.

**The coefficient over the edge endpoints (§31.1).** The two vectors of p. 443: one endpoint's
value in `x`, the other's in `y`, each undirected edge contributing both orderings, and the
Pearson correlation of the two. For `--by degree` that is Newman's degree assortativity
coefficient, and it reproduces `networkx`'s to the last digit — the karate club's -0.4756.
Spearman is printed beside it because these values are usually broad, and when the two disagree
the relationship is monotone but not linear.

**The neighbour-average curve (§31.1).** The other strategy in the chapter: each node's value
against the mean of its neighbours', with every node that shares a value aggregated into one
point, as p. 444 does, and the power fit the chapter asks for. Its exponent is positive for
assortative, negative for disassortative, indistinguishable from zero for neither. Read the
curve before the exponent: the fit weighs a value one node holds as much as one a thousand hold,
and an r-squared on logged points says a line fits those points and nothing more (§9.2).

**The friendship paradox (§31.2).** Your friends have more friends than you, and the report
states it as the identity it is: the degree of the node at the end of a randomly chosen edge
averages `<k^2>/<k> = <k> + var(k)/<k>`, which exceeds the mean degree for every network that is
not regular. What is worth reading is the share of nodes it holds for — 85% on the karate club —
and its size, which is another reading of how broad the degree distribution is. Any attribute
that correlates with degree gets the same paradox "for free" (p. 449); there the excess is
`cov(k, value)/<k>` and can go either way, so its sign is the finding.

**The attribute against the degree (§31.3).** The nodes split into power-of-two degree bands
with the attribute described inside each, because one coefficient hides the shape and §3.1's
heavy-tail caveat applies band by band.

**The null differs by key, and the report says which it ran.** For an attribute it is the label
shuffle: the same graph, the same numbers, dealt out again. For the *degree* that would be
meaningless — the values are the graph — so the null is the configuration model, which is the one
p. 445 names: rewire a network with a broad degree distribution at random and it comes out
**disassortative**, because "the likelihood of connecting a hub to many small nodes seems too
high". A negative r on a corpus network is therefore the expected state, not a finding; only the
distance from that null is one. A network that is both broad and assortative has "some
non-trivial machinery driving their nodes' connections", which is worth writing down.

**`## Degree correlations` is printed in every report**, `--by` or not, because whether the hubs
in a network talk to each other or to the periphery changes how every ranking above it reads:
"degree correlations radically change many network dynamics" (p. 444).

Two refusals. A borrowed number is still borrowed: a value an entity took from the documents its
passages sit in comes from the same documents as the edges, so the section prints the same
provenance line as the categorical one and withholds the verdict past the same share.
And `mentions`, `documents` and `chunks` are the corpus counting itself — a node recorded in many
documents meets many others because of that — so they carry a standing warning and are never an
attribute of the speaker or the entity.
## Ranking nodes

Atlas ch. 14 is the battery `sna analyze` prints under `## Centrality`, and it is nine rankings
on an undirected network and thirteen on a directed one. Six were already here -- degree and its
weighted form, betweenness, closeness, eigenvector and PageRank (§14.1, §14.2, §14.4) -- and the
rest are the chapter's answers to their failure modes:

| ranking | what it answers | when it is undefined |
|---|---|---|
| `harmonic` (§14.6) | closeness on a network that is in pieces: the sum of 1/distance, counting an unreachable node as 0 | never, which is the point; but it is a sum, so it grows with n and two networks cannot be compared by it |
| `reach` (§14.3) | how much of the network a node commands within two hops, following the edges the way they point | it separates nothing on a connected undirected network, which is why the radius is bounded; unbounded it is the chapter's own measure |
| `hits_hub`, `hits_authority` (§14.5) | the two directed roles: a hub points at good authorities, an authority is pointed at by good hubs | on an undirected network they are one vector -- the eigenvector centrality -- and `sna` refuses rather than printing it twice |
| `coreness` (§14.7) | how deep into the network a node sits, by peeling away everyone with fewer than k connections | it is a floor rather than a rank: many nodes share a shell, and the shell sizes are printed beside it |

`reach` is the only one of them that reads *out* of a node: on `--network relations` a high
reach beside a low closeness is an entity that names everything and is named by nothing. The
directed captions in the report say which direction each ranking read. The two-hop bound applies
on the directed network too, so what the report prints is never §14.3's unbounded "how much you
can command" — `reach(graph, hops=None)` is that measure, and a claim about command has to come
from it. §14.6's other repair for disconnected networks, **Lin's centrality** (closeness times
the size of the node's component), is not built: it needs a component to belong to, so it says
nothing about the isolates a filtered corpus network is full of, while harmonic scores them 0
without a special case.

Two things beside the rankings. `k_truss(graph, k)` and `truss_number(graph)` are the edge-side
analogue of the k-core -- every edge in at least k-2 triangles, peeled recursively -- which the
chapter does not name (it names D-cores, k-shells and k-coronas) and which is the stricter
reading on a co-occurrence network, where one long document makes a clique and inflates
everybody's core number. And `centralization` is §14.8, described under "Reading a report".

## Sampling

`sna sample` takes a smaller network out of one of these, for when the whole thing is too large
to look at, by the methods of Atlas ch. 29. It is never applied unless asked for: without
`--sample`, every other command reports the whole network exactly as before.

Which sampler you pick decides which measurement comes out wrong, and the book states the
direction for each one. `induced` takes uniformly random nodes -- fair in the mean, but on a
heavy-tailed degree distribution it is unlikely to catch a hub, and the sample breaks into
components where the network was connected; `--param neighbors=true` turns it into the book's
full node-induced sample, which collects each selected node's whole neighbourhood and therefore
returns more nodes than `--size` asked for. `edge` takes uniformly random edges and then
collects their direct neighbours, which is §29.1's plain edge induction and also returns more
nodes than asked; `ties` is the same draw stopped at the endpoints, so the sample is the size
you asked for. Both land on hubs, because most edges are attached to one. `bfs` crawls
breadth-first, reaching hubs early and exploring each neighbourhood in full, which inflates the
clustering coefficient; `snowball` caps the crawl at `k` connections per node, keeping the hub
but cutting its degree, which is what gives a snowball sample its unrealistic sharp cutoff;
`forest-fire` follows each neighbour with probability `p` instead of all of them, and is the BFS
variant that estimates clustering better than BFS does. `random-walk` oversamples hubs by
construction, because a walk's stationary distribution is proportional to degree, and
`metropolis-hastings` is the walk that refuses a step to a higher-degree node with probability
`1 - k_v/k_u` to correct exactly that. `neighbor-reservoir` walks a core and then spends most of
its budget exchanging a node of the core for a neighbour of it, refusing any swap that would
break the sample into more components than it started in -- which is what the book credits for
its realistic clustering.

The command prints a bias table with the population's value, the sample's, the direction the
book predicts, the direction observed, and a note saying what the row is for -- some rows are
printed and not graded, because the chapter makes no prediction there (density), because the
movement is arithmetic rather than bias (mean degree inside the sample), or because the claim is
comparative rather than an equality (forest fire's clustering, which the book says beats BFS's,
not that it matches the population's). Beside it comes the completion estimate of §29.5: how
many edges were seen leaving the sample, how many more are estimated from the nodes the crawl
never probed, and which nodes to probe next. An edge between two unsampled nodes is invisible to
that estimate, so it is an upper bound; and a snowball sample cannot produce one at all, because
it never learns any node's true degree. The sample itself carries its record in
`graph.graph["sample"]` and a sentence in its sampling frame, so a report run on it says what it
is a sample of, and by which sampler, above every number.

### What §29.4 says about somebody else's crawl

Most networks are sampled before you ever see them, and the chapter's "Sampling Issues" section
is about that case rather than about this command. Four things it asks you to hold on to.

**Pagination.** An API rarely returns all of a node's connections. It returns `k` at a time,
chosen by a rule internal to the provider -- likely the order of its own database -- and you
wait between pages. A paginated neighbour list is a sample of a node's neighbours, not the
node's neighbours, which is why the samplers here shuffle a node's neighbours before reading
them rather than pretending an order exists.

**Throughput is not speed.** The book's worked comparison: a policy returning 100 edges per page
with a two-second wait is 50 edges/second on paper, against ten edges per page every second at
10 edges/second -- and the slow-looking one crawls a broad-degree network in *half* the time,
because out of 500k nodes 492k have degree ten or less and are one query either way, at half the
wait. A high-throughput source can hand you a smaller sample. Under all of it sits network
latency, and a server hit too often slows down unpredictably.

**Hostile and private nodes.** Some nodes lie about their connections, and the book is careful
that this is not always adversarial: protecting one's own privacy is a legitimate reason to
misreport who you know. A crawl cannot tell the two apart.

**Node-centric questions.** If what you need is the local properties of one or a few nodes,
specialised strategies exist and sampling the whole network is the wrong tool.

None of the four is implemented here, because `sna sample` samples a network already in hand:
there is no page size, no wait, no adversary and no query budget. They are documented because
`product-leader` and every corpus like it arrived as somebody else's sample, and this is what it
takes to distrust one correctly.
## Layers, hyperedges and time (Atlas ch. 7)

A corpus is not always one simple graph, and `graphrag.sna.layers` is the chapter that says so.
`--layers stance|facet|source|relation-type` builds the chosen network once per value of that key
and works on the **flattening** — the layers summed into one weighted graph (§7.2, §40.1).
Everything measured then runs on the flattening and is blind to which layer an edge came from.
§40.1 names the assumption that hides in it: collapsing *which layer* an edge appears in into an
edge weight "assumes that every edge type is equally important", and nothing about a corpus says
praise and complaint are. So `sna analyze --layers` prints the per-layer n, m and weight above the
numbers, and `sna layers` prints the same table on its own with the shape of the
**supra-adjacency** matrix (§8.1): each layer's adjacency down the diagonal and `--omega` times
the identity in the off-diagonal blocks, which is the interlayer coupling written as a matrix.
`--omega` is a parameter somebody chose, never something the corpus states. §8.1 would reach for a
tensor in our one-to-one case and keeps the supra-adjacency for many-to-many couplings; the matrix
is built anyway because every solver here takes one, and `omega·I` is the multiplex convention laid
into its block form.

The layers are not guaranteed to add up to the unlayered network, and which of the two happens is
worth checking before calling them a decomposition: a facet layering splits *passages*, so its
flattening is the unfiltered network edge for edge, while a stance layering splits *mentions*
inside a passage, so a pair named once approvingly and once critically in one passage falls into
no layer.

In the library, `hypergraph()` reads a passage as one hyperedge over everything it named rather
than as the pairs it stands for (§7.3), `clique_expansion()` turns it back into those pairs — and
is numerically the entity co-mention network — and `hyperedge_sizes()`
is the distribution to read before trusting an expansion, since a passage naming twenty entities
contributes 190 of its edges on its own. (That the expansion is numerically the same object as the
bipartite projection is this package's claim, asserted in the tests; §7.3 gives the two strategies
and Figure 7.8 draws them, but does not say they coincide.) `dynamic_edges()` is the timestamped
edge list of §7.4 and `snapshots()` cuts it into windows: `--window` with `--step` gives single
snapshots, disjoint windows or sliding ones, and `cumulative=True` the fourth of the book's four
strategies. They give different histories from the same activations, which is why every snapshot
names its width, its step and how many documents nothing dated.

Two cuts are not the book's and are labelled as such in the frame. A `--step` wider than
`--window` gives **gapped windows** — the time between one window and the next is thrown away, so
the sequence samples the history rather than covering it and its totals do not add up to the
corpus. And the last window is clipped at `--until` rather than running past it, so it can be
short and hold fewer edges for a reason that is nothing to do with the corpus.
`sna compare --window` refuses a range that makes only one window, because a build cannot be its
own before.

## Projections (Atlas ch. 26)

Every network here except `relations` is a **projection**. What was observed is a set of
memberships — this speaker in that document, this entity in that passage — and the network is
what you get by joining two members of one mode because they share a member of the other.
Chapter 26 is about the consequence: *the weight on the resulting edge is a choice, not a
measurement*. The same memberships give a weight of 3, of 0.79 or of 0.049 depending on which
scheme you picked, and each answers a different question about the same corpus.

`--projection <scheme>` picks one, on `sna analyze`, `export`, `sample` and `layers`. The default
stays `simple`, so nothing moves until it is asked for, and every report names the scheme in its
frame — including `simple`, which is as much a choice as the rest.

| scheme | § | the weight, in words |
|---|---|---|
| `simple` | 26.1 | how many opposite-mode nodes the two share. The default, and the reason the chapter exists: one document naming twenty entities joins all 190 pairs of them, so the weights are combinatorial rather than additive |
| `jaccard` | 26.1 | the same count over the size of the union of the two neighbourhoods. 0 to 1, and blind to absolute size: two nodes with one membership each, shared, score the maximum |
| `cosine` | 26.2 | cosine similarity of the two rows of the incidence matrix |
| `pearson` | 26.2 | their Pearson correlation, **plus 1** — the shift the book's own figure 26.6 plots, because a projected weight cannot be negative |
| `euclidean` | 26.2 | `1 / (d + 1)` for the Euclidean distance `d` between the rows, again as figure 26.6 prints it |
| `hyperbolic` | 26.3 | `Σ 1/k_z` over the shared nodes: a node that touches many others contributes less, because "bandwidth is finite". Similar in shape to `simple` but it "exaggerates the differences, so that thresholding becomes easier". **The book contradicts itself here** and this is the one place the code follows the figure over the text: p. 374 displays `Σ 1/(k_z − 1)`, but figure 26.7, its caption and §26.4's matrix description all compute `Σ 1/k_z` — the figure's .46 is `1/3 + 1/8` and its .79 is `1/3 + 1/3 + 1/8`, neither of which the displayed formula gives. `Σ 1/k_z` is implemented, the frame of every hyperbolic network says so, and the tests reproduce the figure |
| `resource` / `probs` | 26.4 | resource allocation (ProbS, Zhou et al. 2007): `Σ 1/(k_u k_z)`, discounting twice instead of once |
| `heats` | 26.4 | HeatS (Zhou et al. 2010), which normalises by the destination instead: the transpose of ProbS |
| `hybrid` | 26.4 | ProbS and HeatS interpolated by `--lambda`: 1 is ProbS, 0 is HeatS, 0.5 the middle |
| `randomwalk` | 26.5 | resource allocation taken to infinite walks: `π_v A(u,v)`, the stationary probability of landing on `v` having started from `u` |

**Which to use.** `simple` when the question is how much two things were recorded together, and
you are about to threshold anyway. `hyperbolic` or `probs` when the question is who is
*specifically* connected rather than who was recorded a lot — they are the ones that discount the
hub, which on a podcast corpus is the long episode and the ubiquitous host. `jaccard` when the
sizes of the two neighbourhoods should not matter at all. The vector schemes (`cosine`,
`pearson`, `euclidean`) read the memberships two nodes *both lack* as evidence of similarity,
which is sometimes what you want and is never what a count means; §26.2 warns that they "were not
really developed with network data in mind" and do not fix the hub problem.

Four of the schemes are **directed in the book** — `probs`, `heats`, `hybrid` and `randomwalk`,
because u's score for v is normalised by u's degree and v's by v's. These networks are
undirected, so the two are averaged, which is one of the three remedies §26.4 offers, and both
directions are kept on the edge as `weight_uv` and `weight_vu`. That asymmetry is an artefact of
a normalisation, not a claim anybody made: `--network relations` is the only network here whose
direction was written down.

Averaging has one consequence worth knowing before you choose between them: since HeatS *is* the
transpose of ProbS, and averaging a matrix with its transpose gives the same answer for both,
**`--projection probs` and `--projection heats` produce the same undirected network**. Their
difference lives entirely in the direction, so it is visible in `weight_uv`/`weight_vu` and in
`sna projections` — which compares the schemes in their directed form for exactly this reason —
and nowhere else.

Only `simple` produces counts, so `--min-weight` means something different under every other
scheme: it is applied to the weight as given, and because a count threshold would delete every
fractional edge, a non-simple scheme with no explicit `--min-weight` defaults to 1 rather than to
the network's usual threshold. A fractional cut belongs to `sna projections --threshold`, and a
defensible one to `--backbone` below.

`graphrag sna projections <persona>` is §26.6: it reads the memberships once, runs every scheme
over them, and prints the edge-weight distribution of each (figure 26.10), the Spearman rank
correlation between every pair of schemes (figure 26.11 — rank, because the scales are unrelated
and only the order can be compared), and what share of the edges each keeps at a threshold, with
how far that surviving set agrees with the default's. §26.6's own finding is worth expecting:
schemes you would think must agree need not, and on the book's Twitter data HeatS and ProbS — one
the transpose of the other — came out anti-correlated.

Two knobs there mean different things. `--min-weight` narrows the *corpus* the comparison runs
on: it prunes the simple-weight network first and then compares the schemes over the memberships
of the nodes that survived, so the opposite-mode degrees every discounting scheme divides by are
that sub-corpus's own. `--keep-top` (a share) and `--threshold` (an absolute weight) are the
threshold step *inside* the comparison, applied to each scheme's own weights. Because four of the
schemes are directed in the book, the comparison reads ordered pairs — twice the undirected edge
count — which is the only form in which the HeatS-versus-ProbS question exists at all.

Nothing in that report is a test and nothing in it has a null model: a weighting scheme is a
definition, so there is nothing for it to be unlikely against. `sna analyze --null bipartite`
is where a projected number meets a null, and it re-projects the rewired memberships under the
same scheme the observation used.

Finally, the thing §26.6 closes on: every scheme joins two nodes that share even one neighbour,
so every projection is dense whatever the weighting — the book's own example is 175k memberships
becoming 5.3M edges. Choosing a cleverer scheme does not thin a hairball. Thresholding does, and
that is the next section.

## Backboning

Every network here is a co-occurrence network, so every one of them is denser than it looks and
some of its edges are noise. `--min-weight` has always been the answer, and it is the Atlas's
*naive* backbone (§27.1) — the one chapter 27 spends thirteen pages arguing against. Weights
distribute broadly (most edges sit at 1), so the mildest threshold available already deletes most
of the network, and a fat-tailed distribution has no well-defined mean to motivate the choice
with. Weights are also locally correlated, so one threshold flattens the dense corner and leaves
the sparse one intact.

`--backbone <method>` runs a defensible one instead, after `--min-weight` and before anything is
measured. Every surviving edge carries the number it survived on — `p_value` for the statistical
methods, `score` for the structural ones — and the frame says which method ran, at what level,
and how much of the network is left.

| method | § | what it keeps |
|---|---|---|
| `naive` | 27.1 | edges above one global weight threshold. Kept for continuity; read the objections above |
| `naive-top` | 27.1 | each node's `n` strongest edges (`--alpha` does not apply; pass `n` as the threshold). It escapes the objections above and buys a worse one: the minimum degree of the network becomes `n`, so the degree distribution afterwards is the filter's, not the corpus's (p. 384) |
| `doubly-stochastic` | 27.2 | edges that stay strong after rows and columns are normalised to sum to 1, thresholded (as the original paper asks) at the point where the network still holds together. Does not always converge, and says so when it does not |
| `high-salience` | 27.3 | edges in a large share of the network's shortest-path trees, one per node, over `distance = 1/weight`. Expensive: one Dijkstra per node |
| `convex` | 27.4 | a tree of cliques: the network's own maximal cliques, heaviest first, joined by the heaviest edges that do not close a cycle between them |
| `disparity` | 27.5 | edges significantly heavier than an endpoint expects — **either** endpoint, which is why it "tends to create networks with high centralization, broad degree distributions, and weak communities" (p. 391) |
| `noise-corrected` | 27.6 | edges significantly heavier than a binomial null built from **both** endpoints' strengths. The book's recommendation for count weights, which is what every weight here is, and the one to use before any question about communities |
| `mst` | 13.4 | the maximum spanning tree: `n − 1` edges, the sparsest thing that still touches every node |
| `pmfg` | 13.4 | the heaviest planar subgraph, greedily: at most `3(n − 2)` edges, and it keeps triangles a tree cannot |

The `relations` network is directed, and the book defines the directed form of both statistical
methods: the disparity filter tests an edge against the out-connections of the node sending it or
the in-connections of the node receiving it (p. 391), and the noise-corrected null treats `u → v`
and `v → u` as different edges with different expectations, built from the sender's out-strength
and the receiver's in-strength (p. 392). Those two run on a digraph, as do `naive` and
`naive-top`, which read no neighbourhood. The other five refuse it and say whether the book
defines a directed form at all — `sna backbone --network relations` prints the reason on each
row rather than leaving the method out.

`graphrag sna backbone <persona>` runs all of them at one alpha and prints, per method, how many
edges and nodes survived, what share of the total edge weight they carry, how many components
are left and what the surviving weights look like — each next to the null model or criterion
that produced it. It is meant to be run before choosing, not after.

Two things a backbone is not. It is not summarization: nodes are never merged, and a node the
filter strands is a finding about that node, not a reason to lower the threshold (pp. 381–382).
And it is not a cleaner view of the same network: it is a different sampling frame, so the
method belongs in the same sentence as any number computed after it.
## Random walks (Atlas ch. 11)

`graphrag.sna.walks` is what a random walker on one of these networks tells you, and
`sna walks <persona> --from u --to v` prints the pairwise half of it for one pair. A walk here is
the memoryless process of §2.7 — the next node depends on the node the walker stands on and on
nothing it did before — run on the transition matrix `stochastic(graph, orientation="row")` that
chapter 8 already defined. Edge weights are read as **conductances**: a heavier edge is one the
walker crosses more often and one with less electrical resistance, which is the affinity reading
the rest of the package gives them.

Where the walker ends up is `stationary_distribution`, and on an undirected network it is not a
new ranking: §11.1 says it is *"quite literally the normalized degree of the nodes"*, `k/2|E|`, so
it is computed in closed form with no eigensolver and reporting it after a degree table reports
the same finding twice. On the `relations` network there is no closed form and it is iterated for,
which fails in the two ways §11.1 names — a node with no out-edge, where the walk ends and
probability drains away, and a network that is not strongly connected, where the answer depends on
where the walker started. Both raise and both point at `centrality(..., "pagerank")`, whose
teleport is the fix the section itself names.

`hitting_time` is the expected number of steps to arrive and it is **asymmetric**: §11.3's own
example is one step from the end of a three-node chain to its middle and three steps back, because
the formula depends on the degree of the destination and not of the origin. `commute_time` is the
sum of the two directions, and `effective_resistance` is that commute time with the size of the
network divided out (`C = 2|E|Ω`, §11.4) — which makes the resistance the one of the three that
can be compared across two networks, two time windows or a sample and its population. §11.4 calls
Ω *"one of the awesomest matrices in network science"* for two reasons this corpus cares about: it
is a proper metric where a shortest path is not, and it counts every route in parallel rather than
one, so it moves less when a single edge appears or vanishes. On noisy extraction data that is the
difference between a measurement and an artefact. In a tree, where there is only one route, it is
exactly the hop count and tells you nothing new. Between two components every one of these is
infinite, and each component is scaled by its own `2|E|`, because §11.1 reads a disconnected
network as several networks rather than one bad one.

`mincut` is the exact `s`–`t` minimum cut, equal to the maximum flow, and the random-walk reading
that puts it in this chapter is that `Ω ≥ 1 / cut`: a network expensive to cut is cheap to walk
across. `global_mincut` solves the problem §11.5 actually poses, and on a corpus network it will
find a pendant node — an entity named in one passage — which is the section's own warning that a
cut is not a discovery of groups (*"you have to do community discovery"*, Part X). `fiedler_cut`
is the spectral approximation the section spends its pages on, balanced by construction and
therefore never cheaper than the exact one; print both when claiming either.

`consensus` runs the averaging dynamics of §11.6 in both the forms the section names, and they do
not converge to the same number. The discrete one driven by the stochastic matrix (DeGroot)
conserves `π · x`, so it settles on the **degree-weighted** mean and a hub's starting opinion
counts for more; the continuous one driven by the Laplacian conserves the plain sum and settles on
the **unweighted** mean. Say which was run. Two failures are structural rather than numerical: on
a disconnected network each component reaches its own consensus and there is no single answer, and
on a bipartite network the discrete dynamics flips between the two sides forever instead of
settling — the periodicity of §11.1 in its dynamic form, fixed by the lazy walk `(I + P)/2`.
`consensus_time` is §11.6's `1/λ2`, the order of magnitude of how long a network takes to agree,
and it is large exactly when the network has a thin cut, which is the mechanism the section offers
for why the Fiedler vector solves the mincut at all.

## Paths and components (Atlas ch. 10, §13.1-13.3)

Every `sna analyze` report carries a **Paths and components** section; there is no flag, because
how many pieces a network comes in and how far apart its nodes are describe every network rather
than answer an option. `graphrag.sna.paths` is where it is computed, and every function in it is
usable on its own.

**Components (§10.4).** §10.4 also gives a spectral way to count components — the eigenvalues of
the stochastic adjacency equal to one, or the zero eigenvalues of the Laplacian — and that already
exists as `graphrag.sna.matrices.stochastic` / `laplacian` / `eigenpairs` (chapter 8), so this
section traverses instead: it needs the members, not only the count, and "equal to one" is a
floating-point judgement. An undirected network gets one reading; the directed `relations` network
gets two, because the chapter splits the idea once the edges point. *Strong* means "u must be able
to contact v, and vice versa" respecting the directions, so a strongly connected component is a
set of nodes any two of which lie on a common cycle; *weak* is what you get by ignoring the
directions. A corpus `relations` network looks exactly like the chapter's figure 10.11(b): one
weak component and many strong ones, most of them a single node, because a node on no cycle is its
own strong component. The section also prints the **condensation** — collapse every strongly
connected component to a point and what is left is always a DAG — with §10.4's own bow-tie round
its largest node: the in-component that can reach the core, the out-component the core can reach,
and everything that does neither.

**Path lengths (§13.2-13.3).** The histogram of shortest-path lengths, its mean and median, the
diameter, the radius, and the centre and periphery the eccentricities define. Two conventions
decide what those numbers mean. First, **length is a count of edges**: §13.2 defines the shortest
path as the one "crossing the fewest possible number of edges", so the default is hops and the
histogram exists only there. Passing a `weight` switches to Dijkstra and prints costs instead, and
§6.3 is the reason to be careful — our weights are affinities, so `weight="weight"` asks for the
paths that cross the *strongest* ties and calls them the longest. A **negative** weight is refused
outright, with §13.2's own reason (pp. 196–197): going back and forth over a negative edge gives
"an infinite loop of shorter and shorter paths without ever reaching the destination", so the
shortest *walk* is unbounded and only the shortest *path* is defined — a distinction Dijkstra
cannot make. Nothing this package builds has one, so no report can reach that refusal. Second, **"infinite" is not a
number you average**: §13.3 says nodes in different components are at infinite distance, so "a
network with more than one connected component has an infinite diameter" and what you look at
instead is "the diameter of the giant connected component". That is what is measured, and the
report says how many nodes it left outside. On the directed network there is a second exclusion the
book does not need: a giant *weak* component still holds ordered pairs no directed path joins, so
those are counted and named rather than averaged in — and where they exist, the radius, the centre
and the periphery stop being §13.3's, because an eccentricity is then the distance to the furthest
node a source *can* reach and a node that reaches almost nothing looks central.

**Above a thousand nodes it estimates.** All-pairs shortest paths cost one traversal per node, so
past `MAX_EXACT_NODES` the section measures from a seeded sample of 200 sources and says so beside
every number, with the seed. The mean and the histogram are then estimates over the sampled ordered
pairs; the diameter can only rise as sources are added and the radius can only fall, so they are
printed as a lower and an upper bound and must never be compared with an exact one.

**Walks and cycles (§10.1-10.2).** `walk_counts(graph, k)` is the `A^k` of §10.1, whose entry
`(u, v)` "is exactly the number of such walks" of length `k`, and whose diagonal counts closed
walks. Both readings need the matrix §10.1 asks for — "binary and its diagonal is set to zero" —
so weights are dropped and self loops with them, which is what makes `A²_uu` the degree and
`A³_uu` exactly twice the triangles through that node. `triangle_count` is `trace(A³)/6`, the same
count one level up: summed over every node a triangle is seen from each of its three corners in
each of two directions. `cycle_space`
reports the number of *independent* cycles, `|E| - |V| + c`, because the number of cycles is
exponential and the karate club's 45 is the useful figure; it also names which entry of figure
10.5 the network is — tree, forest, arborescence, directed tree, directed acyclic, cyclic. Keep
the three words apart: a walk may reuse edges, a path may not, and a cycle is a path back to where
it started.

**Reciprocity and the dyad census (§10.3).** The summary table has printed reciprocity since the
directed network landed; this section prints the three-way split it is a ratio of — mutual,
asymmetric and null pairs. The book's own example is 40% over five connected pairs, and 40% over
five thousand is a different finding with the same number.

**Exploration (§13.1).** `exploration_order(graph, source, "bfs"|"dfs")` is the section's own
algorithm — one queue, first-in-first-out or last-in-first-out — with neighbours taken in sorted
order so two runs agree; `bfs_tree` and `dfs_tree` are thin `networkx` wrappers. Neither solves the
shortest-path problem in general, which is §13.2's point; the BFS tree does on an unweighted graph,
which is why it is what the unweighted path lengths are built from.

**The distances are in the section and not in the summary table.** `summary()` is called at the
top of every analysis, every window comparison and every sampling bias report and takes neither a
seed nor a size bound, while a diameter is an all-pairs solve; and the section can print which
component the number came from, how many sources it came from and the seed, which a flat table of
floats cannot. `summary()`'s own fields are unchanged.

**Nothing in this section is a test.** Every number is a count or a distance in the network as
observed. The small-world line prints the observed average beside `ln n / ln ⟨k⟩`, the expectation
for a random network of the same size and mean degree, and stops there: §13.3 makes the claim that
path lengths grow sublinearly without giving a number to check it against, and a comparison that is
a finding needs a null model with this network's own `n` and `m` (ch. 17).
## Uncertain edges (Atlas ch. 28)

No edge in any of these networks is an observed tie. An extraction pass named an entity, a
matcher found a passage for the name, and a projection turned two mentions in that passage into
an edge — so an edge is a **claim with evidence behind it**, and chapter 28 is the chapter that
insists the difference matters. Its opening example is the graph everything is tested on: nobody
knows whether Zachary's karate club has 77 edges or 78, and almost nobody says so.

Every edge now carries `p`, its probability of existing, derived from the evidence the corpus
itself recorded:

| where the edge comes from | how `p` is set |
|---|---|
| entities, speakers–entities | the tier of the mentions behind it: a passage that names the entity verbatim is worth 0.95, one that matched only a significant token 0.6, the fallback anchor (no passage names it at all) 0.3, and a mention written before tiers were recorded 0.8 — a placeholder between the first two, not a measurement |
| relations | 0.95 per passage stating the relation |
| speakers, topics | 1.0: a speaker is on a document because a record says so and a topic because the loader read it off the front matter. Nothing was matched, so nothing here can be less than sure — which says what this pipeline can measure, never that the record is true |

Within one passage the weaker of the two mentions decides, because the same matcher produced both
from the same text and they fail together; across passages they combine as `1 − ∏(1 − p)`, the
independence the chapter itself assumes. So an edge two verbatim passages support comes out at
0.9975, and one resting on a single loose match at 0.6. The stance on a mention is deliberately
not part of the number: it rides on only one of the two reads the entity network uses, so folding
it in would make `p` depend on which flag was passed rather than on the evidence. `p` is an edge
attribute, so it survives into GraphML, node-link JSON, GEXF and the Cytoscape CSV pair. Do not
confuse it with `p_value`, which a statistical backbone leaves on a surviving edge: a small
`p_value` means the edge is probably real, a small `p` that it probably is not.

**Every report says what its edges rest on**, with or without a flag: how many hang on a single
passage, how many of those on a loose match, what share of the total weight they carry, and how
many mentions came in at each tier. That is §28.1, and it is not optional, because a ranking over
edges that each rest on one loose match is a different object from one over edges several
passages support.

**`--uncertain` adds the rest of the chapter.** The network is read as the probabilistic network
§28.2 defines — `G = (V, E, Π)`, where every edge exists with its own probability — and since the
`2^m` possible worlds cannot be enumerated, `--uncertain-samples` of them are drawn with a fixed
seed. Every centrality and the partition's modularity come back as a mean and a 95% percentile
interval beside the observed value, and **adjacent ranks whose intervals overlap are named as the
ties they are**: some of those worlds put the other node first, so the order they are printed in
is a presentation choice. Expected edges, expected density and expected degree are exact instead
(§28.3, `E[k] = Σ p`), and beside each expected degree sits the mode of the node's exact degree
distribution — because the book's objection to the expectation is that "the degree is a count, it
shouldn't be a continuous number", and no node ever has degree 3.4.

**Three of §28.3's five problems are built and two are not.** Node degree and connected
components (the section's *reliability*: the probability two nodes are reachable, which they can
fail to be even when an edge joins them) are here, and any other measure goes through the
sampling. **Ego networks** (pp. 404–405) are not: the section's point is that realising the
worlds and then taking each ego network gives a different answer from taking a probabilistic ego
network and realising that one — the second can leave an ego unconnected to its own neighbours —
and both orders already compose out of `sna.export.ego` and `realisations`. What is missing is
the command that picks one and reports the gap, and an ego network is §30.1's object, so it
arrives with ATL-30. The **densest subgraph** (p. 407) is not built either: `expected_density`
already gives the expected density of any subgraph handed to it, but *searching* for the densest
one is missing in the deterministic case too, where it belongs with k-core decomposition (§14.7,
ATL-14) — building the uncertain form first would leave nothing to compare it with. And
**betweenness is sampled rather than computed** by the section's product-of-edge-probabilities
per path with a length cutoff (pp. 405–406), for two reasons: the cutoff is the one parameter the
book refuses to fix ("there is no silver bullet"), so a report would be quoting a threshold nobody
could defend; and our betweenness is weighted, over `distance = 1/weight`, while the path-product
form is written for the unweighted shortest-path count. Sampling costs more and answers the same
question, with an interval instead of a point estimate.

Three things this is not. It is not backboning: §28.2 draws the line itself — a backbone removes
spurious edges, while this "want[s] to use all the information we have and integrate it in the
analysis, no matter how unlikely an edge is to exist" (p. 401) — and nothing here deletes an edge. It is
not an error *correction*: §28.1's methods need the network measured several times, or a
generative model of it good enough to say which edges are spurious, and a poor model hands you a
useless 50-50. And above all, **the interval is over the edges you have, not over the edges you
missed**: a tie the corpus never wrote down has no row, no probability and no place in any
possible world, so missing edges stay exactly as invisible as they were. Chapter 28's §28.4
alternatives to probability — Dempster–Shafer and fuzzy logic — are documented in
`docs/SNA_FOUNDATIONS.md` and not built.

## Node roles (Atlas ch. 15)

`graphrag sna roles <persona>` asks what each node *is* rather than how much of it there is. The
chapter's own framing: *"Sometimes you cannot put a number to what you're trying to describe. What
the person is doing in the social network does not have a quantity, but a quality: she is playing
a specific role"* (p. 225), and *"rather than being a characteristic of the person by itself, the
node role is determined by her position in the network"* — so nothing in this report reads a node
attribute, and `--by` is not the question here.

§15.1 names four roles in prose: a **broker** sits between two social circles and belongs to
neither, a **gatekeeper** belongs to one and manages its traffic with the outside, a **core**
member is *"very embedded"* in the circle, a **periphery** member *"does not have many connections
in the community"* (pp. 226-227). It gives none of the four a formula. What the report computes is
the two coordinates both readings need — `z`, how well connected a node is inside its own module,
standardised in that module's own units, and `P`, the participation coefficient the `## Brokers`
table has always ranked by — and cuts the plane they span into the seven regions of Guimerà and
Amaral's functional cartography, a paper the Atlas cites in its bibliography (ch. 36, n. 13)
without ever printing its thresholds. **Those thresholds are ours, not the book's**, which is why
they are a parameter and why the report prints them; the table maps each of the seven back onto
the chapter's own four words.

Two things in that section are properties of the partition rather than findings about the network,
and the report says so next to the counts. A partition into `m` modules caps `P` at `1 - 1/m`, so
on a two-community network **no node can be classed a connector** however evenly its ties run —
0.5 is below the 0.62 threshold. And a module whose members all have the same internal degree has
no spread for a z-score, so it can hold no hub. More generally: a role is relative to the partition
it was computed on. The Louvain run behind it, its resolution and its stability-across-seeds are
printed above the roles, and when that stability is low the report tells you to quote the counts
rather than any node's role.

§15.2 is node similarity, and `--method` picks it. The three structural ones compare the rows of
the adjacency matrix, which is the book's own definition — *"If you sort the nodes consistently,
each node can be represented as a vector of zeros and ones [...] These are the rows in the
adjacency matrix"* (p. 230): `jaccard` on the neighbour sets, `cosine`, and `pearson`, the only one
of the three that can go negative. The chapter's worked example is the acceptance test: its Figure
15.6 pair score 0.75 by Jaccard (*"three common neighbors out of four possible"*) and *"around
0.7"* by Pearson, which this package reproduces to the digit. Edge weights are ignored throughout,
because the book's vector is of zeros and ones — the question is whether two nodes were written
down beside the same others, not how often.

`simrank` and `regular` are the two recursive relaxations, and both have a parameter that changes
the answer, so both are printed. SimRank is *"similar nodes connect to similar nodes"* with the
formula the book gives at p. 338; its decay is *"a parameter you can tune"*, defaulted to 0.8 here
and to 0.9 by networkx, and neither default is derived from anything. `regular` is §15.2's own
recursion `sigma = alpha A sigma A + I` (p. 232) — with one correction: the book motivates `alpha`
with *"if alpha < 1"*, which is not enough for the series to converge, so the alpha actually used
is half of `1/λ₁²`, and the report prints it, the limit, and whether the iteration settled. There
is a second departure in that method, and the report's caption says it too: the matrix handed back
is **cosine-normalised**, `sigma_uv / sqrt(sigma_uu sigma_vv)`, which is not in the book. The raw
matrix has a scale that `alpha` alone decides, so two runs of it are not comparable and it cannot
be read beside the other four; normalising puts it on [0, 1] with 1 on the diagonal, and it is
valid because `sigma` is a sum of `A^2l` and so positive semi-definite. What that costs is that
the numbers are relative rather than absolute — read the ordering, not the value. The book's
warning on the measure is printed with its output too: *"you'd get that CEOs are similar to
interns because they both connect to middle managers"*.

Automorphic equivalence is the middle of §15.2's three subsections (pp. 230-231) and is **not
computed here**. It sits between the other two — *"two nodes are equivalent if you can perform
this re-labeling"*, so equivalent nodes belong to the same local structure type without needing
the same neighbours — and deciding it needs graph automorphisms, which is the isomorphism
machinery of §41.3 and belongs with the subgraph-mining ticket rather than this one. Structural
equivalence (strict) and regular equivalence (loose) are both here; the middle rung is the gap.

`-k` adds the blockmodel: the similarity clustered into positions, with the **image matrix** — the
density of edges between each pair of positions — as the output. The chapter reads three
equivalence classes off its Figure 15.8 by eye and names no algorithm, so the clustering used
(average-linkage agglomerative on `1 - similarity`, deterministic, no seed) is stated in the report
rather than implied, along with the consequence: a position is a cluster and inherits a
clustering's instability.

The section the chapter earns is the last one. Structural equivalence is about neighbours, not
about being connected — the book's illustration of a perfectly equivalent pair (Figure 15.5(a))
has no edge between the two — so the report ends on **same role, never together**: the entities
this corpus discusses in the same structural company and never in one passage. Whether that is a
gap in the corpus or a fact about the subject is not something the network can say, and the
similarity floor behind the list is a reporting threshold with no null model under it. Nothing
else in this report has a null model either, and for a better reason: a role is a deterministic
function of the network and the partition, not a claim about a population. What carries a null
here is the partition, and `sna analyze` is where it is tested — which is also where a short form
of this section appears, under `## Roles`, next to the communities that produced it.

§15.3 (node embeddings) is deliberately not implemented. The section is a pointer forward —
*"the most common way to discover such roles has become the use of graph neural networks [...] I
will explain more in detail how they work much later in the book"* (p. 233) — and this package
follows the same pointer: embeddings-as-roles belong to the shallow-graph-learning and
random-walk-embedding tickets, not here.

## Homophily: ego networks and weak ties (Atlas ch. 30)

Chapter 30 is the mesoscale: not the whole network and not one node, but the neighbourhood a node
sits in and the way the edges sort by what the nodes are.

`sna ego <persona> <node>` is §30.1. The ego network is the node, its alters and the edges among
them — and most of what it shows is the construction rather than the node, which is why the
section prints two views. With the ego in place the neighbourhood has one component and a diameter
of 2 *by definition* (the chapter says so outright), so its density is nearly meaningless. Remove
the ego and the density of what is left is exactly the ego's **local clustering coefficient**
(§12.2): the pairs of its neighbours that know each other over the pairs it could have closed. The
report computes the two by different routes and prints them side by side, so the identity is a
check. What the clustering leaves over is **brokerage** — the pairs of neighbours joined only
through this node — and on a corpus network a high one has two readings the number cannot
separate: a thing that genuinely connects unrelated subjects, and a thing that was written about
in unrelated contexts. `--by` adds §30.2 over the neighbourhood: what the alters carry against
what the whole network carries, with the provenance of every label, and §30.4's **majority
illusion** named when a value that is rare in the corpus is the majority among these alters.

`analyze --by` gains §30.2's two count-based indices beside the assortativity it already
reported. The **EI index** is `(external - internal) / (external + internal)` over the edges
between labelled nodes, so **negative means homophily** — the opposite sign to the assortativity
above it — and it knows nothing about how common each value is. That is not a detail: a value
holding 90% of the nodes has 89 in-group partners available for every 10 out-group ones, so it
scores about -0.62 in a graph wired entirely at random. The book names the failure ("this approach
breaks down ... if some values are more popular than others", p. 434) and the section answers it
twice: **Coleman's index** per value, which subtracts the share the group's own size implies and
returns ~0 for that random 90% majority, and the same label shuffle the assortativity uses, whose
z-score is the part of the EI that is about the attribute. When the two disagree the report names
the value as size-driven rather than leaving the reader to spot it.

A positive EI is not a non-result: it is §30.4's **heterophily**, "the love of the different",
the disassortative case the chapter's summary puts beside homophily — a dating network by gender,
or a corpus where a value is only ever discussed against another. The section names it rather
than reporting "no homophily".

`## Weak ties` is §30.3, and it is about edges rather than nodes. Granovetter's weak tie is
structural — "individuals whose social circles do not overlap much" — so the section measures
**neighbourhood overlap** per edge (shared neighbours over the union of the two neighbourhoods,
Onnela et al.'s formula for the definition the chapter gives in words) and the edges of zero
overlap are the bridges, whatever they weigh. Weight is only a *proxy* for that, and the chapter
is explicit that the correlation can go either way, so the section bins overlap by weight
quantile, reports Spearman's rho, and withholds the Granovetter reading when the curve is flat or
falling. On these networks a weight counts shared documents, so "strong" means "written down
together often" and nothing warmer.

**Neither section can be read on a two-mode network.** Brokerage and overlap are both built
out of shared neighbours, and on `--network speakers-entities` without `--project` two adjacent
nodes are never of the same kind (§6.4), so the intersection is empty by construction: every edge
has an overlap of 0 and is counted as a bridge, every node has a local clustering of 0 and a
brokerage of 1.0. Printed flat that reads as "every tie in this corpus is weak and every speaker
is a perfect broker", which is a statement about the data model. Both `sna ego` and `## Weak ties`
detect it — from the declared modes, or from 2-colourability for any other triangle-free network —
and withhold the reading, pointing at `--project` and chapter 26, whose own caveat applies in
turn: a projection closes triangles by construction, because every document's entities become a
clique.

And §30.4 is printed with every `--by` section as a caveat rather than as a number, because it is
not one: homophily and contagion leave the same cross-section. "If we observe a strong homophily,
it could be because our social connections are influencing us into adopting behaviors we would not
otherwise" (p. 437) — nodes that were already alike coming together, and nodes that came together
becoming alike, are indistinguishable in a snapshot, and no null model separates them. What would
is a dated edge list with each node's attribute as it stood *before* the tie formed; the chapter's
own evidence for contagion follows thousands of people over 32 years. Half of that exists here —
`--window` and `dynamic_edges` date the edges (ch. 7) — and the other half does not, because an
attribution or extraction sidecar records what a node is now, with no history. So the report says
association, names both mechanisms, and claims neither.
## Hierarchies (Atlas ch. 33)

`graphrag sna hierarchy <persona>` asks whether the relations the corpus states form an
organisation chart, and every `sna analyze` of a **directed** network carries the same section.
It is the only section that exists on one network and not the others, and the chapter says why:
*"from now on, we always assume that the network we're analyzing is directed"* (p. 465). Handed
an undirected network the command exits 2 rather than flattening one, because each of its
measures is *degenerate* there rather than merely inapplicable — with no directions every node
reaches its whole component, so GRC is 0; every edge lies on a cycle of length two, so the cycle
share is 1; and no ranking can point both ends of an edge downward, so the agony is the edge
count. Four numbers that look like an answer.

**The chapter's other two hierarchies are elsewhere in the book, and elsewhere here.** §33.1
sorts the word into three: an *order* hierarchy "is nothing more than a different point of view
of node centrality" — that is the Centrality and Ranking sections (ch. 14) — and a *nested*
hierarchy "is equivalent to performing hierarchical community discovery" (ch. 37, not built).
What this section measures is the third, the **flow** hierarchy.

**Four scores, printed together because each is wrong where the next is right.** This is the
chapter's own argument, not an abundance of caution:

| measure | §  | perfect score | what it is blind to |
|---|---|---|---|
| flow hierarchy | 33.2 | 1: no arc on a cycle | too lenient — *every* DAG scores 1, including Figure 33.5's wheel with one edge flipped and its chart with more bosses than workers |
| global reach centrality | 33.3 | 1: one node reaches all, nobody else reaches anything | too strict — only a star scores 1; a perfect binary tree of depth three scores 0.898, which is the book's 0.89 for Figure 33.6 |
| arborescence score | 33.4 | 1: every node already had exactly one boss | punitive — it counts the arcs removed, not how wrong they were |
| agony | 33.5 | 0: every arc runs from a level to a lower one | like the flow hierarchy it scores every DAG perfectly, but unlike it, it grades the violations |

**Cycles (§33.2).** An arc lies on a cycle exactly when its endpoints are in the same strongly
connected component, so the share is one pass over chapter 10's machinery rather than a search
for cycles. The book computes the complement by condensing the graph and summing the condensed
edge weights (12/20 = 0.6 on its Figure 33.4); the weights exist only to count the original arcs
that survived condensation, so counting the arcs inside components and subtracting gives the same
number with no bookkeeping. A self loop counts as a cycle here and is excluded everywhere else.

**Global reach centrality (§33.3).** Built on chapter 14's reach with the radius removed —
`measures.reach(graph, hops=None)`, the fraction of the other nodes a node can get to — and
averaged as `GRC = (1/(|V|-1)) Σ (LRC_max − LRC_v)`. The report prints the maximum, the mean and
how many nodes tie for the maximum, which is the chapter's exercise 2 ("is there a single head of
the hierarchy or multiple?") and the thing GRC itself cannot tell you.

**Arborescence (§33.4).** Where this package departs from the chapter, deliberately. §33.4
condenses the strongly connected components away and then breaks every in-degree above one by
keeping the arc "coming from the node with the lowest closeness centrality", and notes in the
same breath that "there are alternative methods to reduce a generic directed network to a DAG,
which can preserve more edges". This uses the optimal one: Chu–Liu/Edmonds' **maximum spanning
arborescence**, which maximises the weight kept over every choice of one incoming arc per node
and resolves the cycles inside the same optimisation. The score is never below the chapter's, the
tree keeps named entities rather than condensed blobs, and no closeness tie-break decides
anything. Where no single tree spans the network — several pieces, or several nodes nobody points
at — the report says so in full and falls back to the **arborescence forest** p. 469 says the
technique generally produces. The arcs the tree discards are printed heaviest first: they are
relations the corpus stated, and the tree is a reading of the network, not a correction of it.

**Agony (§33.5).** Minimised exactly. The chapter says only that "there are efficient algorithms
to estimate the agony of a directed graph", citing Gupte et al. (2011) and Tatti (2015), both of
which rest on the problem being the dual of a minimum-cost flow. Rather than reimplement Tatti's
primal–dual algorithm, this solves the program itself with HiGHS through
`scipy.optimize.linprog`: minimise `Σ z_e` subject to `r_u − r_v − z_e ≤ −1` per arc, with levels
bounded to `[0, |V|−1]`. Its constraint matrix is totally unimodular, so every vertex is integral
and a solver over the reals returns whole levels and the true integer minimum. Above
`AGONY_EXACT_EDGES` arcs the program is skipped, a coordinate descent runs instead, and the report
says the number is then an **upper bound**. One note on the book: p. 470 prints the sum as
`max(l_v − l_u + 1, 0)`, which contradicts its own sentence two lines down ("every time l_u < l_v,
we contribute zero to the sum") and its own per-edge definition, which is about the arc `u ← v`.
Written per arc source-to-target the three agree, and that is what is implemented.

**Layers (§33.6).** `sna hierarchy` itself returns coordinates and draws nothing: a longest-path
layering of the condensation, waypoints on arcs that cross more than one layer, and a barycentre
ordering within each layer. Pass the agony ranks instead and the arcs drawn upward are exactly
the arcs agony charged for, which is Figure 33.9(c). `sna draw --layout layered` draws this same
layering (see "Visualization" below); it takes no `levels` argument of its own, so a drawing from
the agony ranks of Figure 33.9(c) specifically is a notebook call to
`graphrag.sna.hierarchy.layered_layout(graph, levels=agony(graph).ranks)`.

**The null model is the chapter's own exercise, and it has less room than it looks.** Every score
is recomputed on `--samples` directed rewirings that hold each node's in-degree *and* out-degree
fixed (§19.1's three-arc swap), which is §33.8's exercises 1 and 4. Read the z-scores knowing
what that null fixes: the arborescence score keeps one arc per node that has any boss at all and
rewiring preserves every in-degree exactly, so on a network where nobody has two bosses the flow
hierarchy, the agony and the arborescence score cannot move at all and only GRC can separate the
observation from its rewirings. A z of 0 there means the null had no room, not that the shape is
ordinary.
## High-order dynamics (Atlas ch. 34)

`graphrag sna highorder <persona>` reads the same passages twice, for the two questions a single
graph cannot answer. The chapter's framing: *"if you only look at the structure you might miss
part of the story. This is especially true if you're interested in knowing how different agents
act in it"* (p. 474).

**The simplicial complex (§34.1).** A passage naming three entities is one relation between three
things — which is what the hypergraph of chapter 7 already keeps. The difference between that
hyperedge and a *simplex* is downward closure: a simplex *"logically also contains all
lower-level simplices, which are taken into account in the analysis"*, while *"a hyperedge with
four nodes only contains those four nodes"* (§7.3, p. 112). `--max-dim` decides how far the
closure goes (3 by default, so a passage naming ten entities contributes its nodes, edges,
triangles and tetrahedra but not its 9-simplex) and the report says how many passages were
truncated by it.

Out of the complex come the three things §34.1 measures. The **generalized degree** `k_(d,m)` is
*"the number of d dimensional simplices incident on an m-face"* (p. 475); the report prints
`k_(2,0)`, how many filled triangles each entity sits in, which is also the answer to the
hypergraph degree chapter 9 deferred to here. The **incidence** `k_(d,d-1)` (p. 476) is printed
as one number, the largest any edge reaches, because that is what says this complex is not a
manifold — a manifold needs every face's incidence to be 0 or 1, and a pair of entities named
together in twenty passages is a face of many triangles. That is why §34.1's manifold growth
models (p. 479) are described in this package and not run.

The number the chapter earns is the **closure**: of the triangles in the 1-skeleton — three
entities joined pairwise — how many are also 2-simplices, meaning one passage named all three.
§34.1 warns that promoting cliques to simplices and flattening simplices to cliques *"are not
commutative. If you apply them one after the other, you're not going to go back to your original
simplicial complex"* (Figure 34.10, p. 481), and the closure is the size of that gap on this
corpus: every unfilled triangle is a co-mention an analysis that treated the entity network's
triangles as simplices would have invented. The report lists the first of them by name. The
definition of the ratio is ours, not the book's — §34.1 gives the mismatch a figure and no
coefficient, so the standard simplicial-closure ratio (Benson et al., *PNAS* 115(48), 2018) is
used and named in the report.

**The memory network (§34.2).** *"If you want to really model the traveler's behavior, you need
some sort of memory, you need to know they arrived from u into v"* (p. 474). The second report
builds Rosvall et al.'s memory network: **a node is a transition** `A -> B`, meaning "in B,
having arrived from A", and an edge is a path `A -> B -> C` the corpus actually shows, weighted
by how often. The sequence it is read off is the passages of one document in reading order — and
that sentence carries every caveat worth having. A document boundary is a hard stop. A passage
that named nothing, or that a `--facet`/`--stance`/`--where` filter removed, breaks the chain
rather than being bridged, and the report counts the breaks. An entity named again in the next
passage is a continuation and not a step, so those are dropped and counted. And the order itself
is a transcript's running order or a thread's posting order: it is not a cause, and the chunker
decided where the passages begin.

The book's own note about that structure is printed with it: the adjacency matrix of a
second-order memory network *"is a non-backtracking matrix (Section 11.2), provided that the
memory network has no self-loops"* (p. 484). Ours is not quite Hashimoto's operator, and
deliberately: that one forbids `A -> B -> A` by construction, while a corpus really does return
to a thing after naming another, so those edges are kept.

**The walk (§34.3).** A memoryless walk on the memory network *is* the second-order walk — *"the
simple memoryless Markov processes on the memory network describe the high-order processes, of
the order you used to build the network"* (p. 484) — so the report runs §11.1's walk on both the
memory network and the ordinary first-order transition network, folds the second-order answer
back onto entities (standing on `u -> v` is standing on v), and prints the entities that change
rank between them.

Two things about that comparison decide how it is read, and both are printed beside it. First,
**both walks are damped by default** (`--damping 0.85`, §11.1's teleport, the walk Rosvall et al.
compare across orders) because a memory network is rarely strongly connected — §34.2 says the
same of HONs, that they *"tend to transform into weakly connected graphs, or even not
connected"* (p. 483) — and an undamped π simply does not exist there. `--exact` asks for the
undamped one and fails where there is none. Second, **read the difference, not the level**, and
do not put it down to one cause. A walk reproduces the corpus's own frequencies only where the
chain it runs on is flow-balanced by the data — every state left as often as it is entered — and
that is a much stronger condition at the second order than at the first. A document that ends on
the entity it began with balances the *entities*, so the first-order walk settles on their
transition frequencies; it does not balance the *transitions*, because the document's first
transition is reached by no path and its last continues into none. `[A][B][A][B][A][C][B][A]`
closes on A, repeats nothing, and still gives (A 3/7, B 3/7, C 1/7) without memory against
(A 2/5, B 2/5, C 1/5) with it, both undamped. So a difference here has three sources —
second-order structure inside a document, the document boundaries and the fragmentation they
cause, and the teleport, which puts its mass on transitions in one walk and on entities in the
other — and which one is doing the work is not something the two columns can tell you. A rank
change says the corpus ordered its passages in a way first-order weights cannot express. It does
not say the memory walk is more accurate about anything. (Ranks are assigned with a tolerance of
1e-9, so two entities a symmetric corpus ties are printed as tied rather than separated by the
last bits of a power iteration.)

Neither report has a null model, and each says so where its numbers are: a face is a
deterministic function of the passages, and so is either walk. The half of §34.3 that is *not*
built here is the community one — the map equation over a memory network is Infomap's, and it
belongs with the other partitioners — along with the High Order Network of Figures 34.11-34.12
(same second-order question, a structure the section itself calls *"really unwieldy"*), the
synchronization, percolation and epidemic processes of §34.1 (simulations whose parameters no
corpus states), the motif-dictionary cut of §34.3 (community discovery), and the continuous-time
`T_t = e^{-tD⁻¹L}` (chapter 11's module, with no question here asking about `t`).

**Order two only.** §34.2 goes on: *"You can model third order dynamics by making a more
complicated version of a line graph, in which nodes stand in for paths of length three"*, and a
memory network is defined by *"all nodes represent transitions of the same order"* (p. 484). This
command builds the second order and stops, for the section's own reason — the paper it cites
capped itself at five because *"the increase in complexity simply wasn't worth it"* — and for a
corpus reason: a third-order node is three consecutive passages that each named something and
its edges need four, which on a chunked transcript is a thin sample before it is anything else. The order is recorded
on the network (`graph.graph["order"]`), so a later raise cannot be mistaken for the same number.
## Link prediction experiments (Atlas ch. 25)

`sna predict-eval` is the experiment, and it exists before any predictor does. Chapter 25's
subject is not how to rank pairs of nodes but how to find out whether a ranking is any good:
"evaluating the performance of an oracle in getting things right is harder than it might seem.
There are surprising ways to get it wrong." Every prediction this package ever prints reports
through this command's machinery, so a ranked list of pairs never appears without the split it
was measured on.

**Two splits, and one of them is better.** With `--temporal 2025-01-01` the training graph is
what the corpus had joined before that date and the test set is what it joined after, which is
the split §25.1 names first: nothing is deleted, so the structure a predictor reads is the
structure that existed on the day. Without it, `--share 0.1` deletes a tenth of the edges at
random and asks for them back -- the k-fold branch of the chapter's Figure 25.2, run once. That
one is available because most corpora are only half dated, but read its frame before its
numbers: a deleted edge is gone from the evidence as well as from the answer key, and here an
edge *is* a passage, so the experiment removes the sentence and then asks a scorer to find it.

The temporal split throws two things away, and prints both. A pair joined before *and* after the
date is not a positive, because §25.1 says a link already in the training data is not a
prediction. A later pair with an endpoint that had never appeared is dropped rather than counted
as a miss, because no scorer that reads topology can rank a node with no edges; counting those
would measure how fast the corpus grew. Undated documents are on neither side, exactly as they
are in every window.

**Why the test set is balanced and why that matters.** Real networks are sparse, so the pairs
that could exist swamp the ones that do: the book's Internet backbone has 18.4 billion possible
edges and 609 thousand real ones, which is why scoring them all is "an unreasonable burden, both
for computation time and memory storage" and why a predictor that says "no new link, ever" is
right 99.999% of the time. The fix §25.1 gives is to build the test set half of real links and
half of sampled non-links, which is what `--negatives 1` does; `--negatives 5` moves it back
towards realism at the cost of a smaller positive class. Either way the report prints the
imbalance of the **unsampled** pair space -- how many non-edges there are per positive, usually
in the hundreds -- next to the balanced n, so nobody reads a balanced AUC as a claim about the
whole network.

**What comes out.** One row per scorer: AUC, average precision, precision@k, the random
precision they are read against, the prediction power in §25.2's decibels, the four cells of the
confusion matrix at the top-k cut, F1 and accuracy. Two baselines always run -- preferential
attachment, the degree product, which is the method §25.2 holds a new predictor to, and a scorer
that ignores the network, which is the 45-degree line drawn as a number rather than asserted as
one. Quote precision@k and the average precision against the random-P column; the AUC is the
number imbalance flatters, accuracy hides which kind of error was made, and an AUC of 1.0 is a
bug report rather than a triumph -- the chapter says you never see one honestly.

Two conventions are this package's rather than the book's, and both are stated where they are
used. Precision@k ranks the pairs of the test set, not every candidate pair, because enumerating
those is the burden the chapter warns about -- which inflates it, and is why its random baseline
is printed in the same row. And §25.2's prediction power is implemented as the formula the book
writes, `10 log10(P/Pr)`, which disagrees with the same page's reading of it ("PP = 1 implies
your predictor is ten times better"); read a PP of 10 as ten times better than chance.

`--min-weight` applies to the random split only. The temporal one reads the timestamped edge
list of §7.4, where an edge is one dated activation, so it is taken over the unthresholded
network.

## Entity-attribute completion (Atlas ch. 44)

`sna complete <persona> <key>` trains a GCN or GraphSAGE (`--method`) on the entities this
persona's extraction sidecars already tagged with `key`, then suggests a value for the ones they
did not -- from the entity's mean text embedding and the network's structure (§44.1's own
example: "H1 is not dependent only on H0 but also on the structure of G"). It needs the optional
`graphrag[gnn]` extra (CPU-only `torch`, `pyproject.toml`); without it the command explains the
install and exits 2 before touching a persona.

**Only entity-owned values are training signal.** A value this network's own attribute layer
marks as *borrowed* -- the majority vote of an entity's documents, not something anyone recorded
about the entity -- is exactly the confound `sna analyze --by` already refuses to certify as an
assortativity finding (see [Node attributes](#node-attributes---where----by) below); training a
classifier on it would fit that same circularity one step earlier. `sna complete` filters to
owned values before it fits anything, and refuses below `MIN_OWN_LABELS` (6) of them across at
least two classes rather than fit noise.

**What comes out.** A confidence-ranked table of suggestions, each an entity, the value the
model picked, its confidence, and the runner-up class -- shaped to paste into that entity's
extraction sidecar under `attributes:`. Nothing is written to the graph. The frame above the
table prints how many entities own a value, how many borrowed one (excluded), and how many carry
none (what is suggested for); the null model line is the majority-class accuracy on a held-out
slice of the owned values (`--val-share`, default a fifth) beside this fit's own accuracy on the
same slice, which is the gap to read before trusting a row.

```bash
graphrag sna complete <persona> region --method gcn --seed 1 \
    --out /app/data/exports/region-suggestions.md --json /app/data/exports/region-suggestions.json
```

`gnn.gcn_link_scorer` fits the same architecture unsupervised, by reconstructing the training
graph's own edges (Kipf and Welling's graph autoencoder; not itself in ch. 44, which names only
the supervised case), and scores a candidate pair by the dot product of the two fitted
embeddings -- a `graphrag.sna.experiment.evaluate_predictor` scorer like every other in this
package, though it is not yet wired to `sna predict-eval`'s `--method`.

## Community evaluation (Atlas ch. 36)

Every `sna analyze` report that groups anything carries `## Community evaluation`, under the
communities themselves, for **whichever** method produced them — Louvain, K-means or a Gaussian
mixture. A clustering of text embeddings is still a partition of the graph, so it still has a
modularity, a conductance and a resolution limit, and that is exactly the case where reading them
matters: nothing in the clustering itself looked at an edge.

The section is a battery, not a score, because the chapter refuses to give one: *"What's 'best'
depends on what you want to use your communities for. Different functions privilege different
applications"* (p. 510). So each measure prints with what it wants, which direction community
*size* pushes it, and where the book defines it.

- **Modularity (§36.1)**, at the resolution the partition was found at, with the ceiling `1 -
  1/k` beside it and the domain `-0.5` to `+1`. A negative value is a reading, not a bug: those
  communities hold fewer edges than a configuration model with the same degrees would give them.
- **The resolution limit (§36.1).** Every community with fewer than `sqrt(2|E|)` internal edges
  is listed by number. At that size modularity maximisation prefers to merge it into a
  neighbour, so its boundary is the algorithm's resolution rather than a fact about the network.
  On a corpus network with tens of thousands of edges the line sits above a hundred internal
  edges, and most small communities fall under it.
- **The other topological measures (§36.2)**, per community and averaged: conductance, internal
  density, expansion, the cut ratio, the normalised cut, the three out-degree fractions of Flake
  et al., and the triangle participation ratio, beside the partition-level coverage and
  performance. Conductance and coverage both improve as communities grow; internal density and
  performance both improve as they shrink. A partition that improves one of them has usually
  only changed size, which is why the "size pushes it" column is printed next to every value.
- **Link prediction (§36.3).** The partition scored as a predictor: a tenth of the edges is held
  out, every pair is scored 1 if its two nodes share a community and 0 otherwise, and the AUC is
  read against the same community sizes with the labels dealt out at random. It is the one test
  here that does not reward merging. Two things are printed with it: it is a single seeded
  split rather than chapter 25's k folds, and the partition was found on the whole network
  including the held-out edges, so the AUC is optimistic and the shuffled baseline beside it is
  what makes it readable.
- **Ground truth (§36.4).** Present when a labelling is handed in, which for a corpus network it
  is not; the section says so rather than printing nothing. `analyze --by <attribute>` is the
  nearest honest thing and is deliberately not called truth — *"in real observed networks
  metadata is just data"* (p. 527). Where a truth is given (a benchmark graph, the karate club's
  two factions) the section prints NMI with the chance-corrected AMI and ARI next to it, because
  NMI is never zero and rises with the number of groups: the book's own pair of independently
  drawn random vectors scores NMI 0.09 and AMI -0.22 (p. 526).

`graphrag.sna.evaluate_partition(graph, communities, truth=...)` is the same battery from a
notebook, and `evaluation_payload` is what the report's JSON carries under `evaluation`. A
measure that is undefined for a community — internal density of a single node, the cut ratio of a
community that is the whole network — prints a dash and travels as `null`, never as `0.0`.
## Robustness (Atlas ch. 22)

`graphrag sna robustness <persona>` removes nodes and watches what is left of the largest
component, which is the chapter's own criterion: *"the share of nodes part of its largest
component. If there still is a path between all or most nodes in the network, even if it becomes
longer, the network still works. However, when nodes start breaking down in multiple components
and getting isolated ... then the network is failing"* (p. 315).

```bash
# random failure against every targeted attack, with the critical fraction beside them
graphrag sna robustness <persona> --network entities --seed 1 \
    --out /app/data/exports/robustness.md

# the stronger attacker, who looks again after every step
graphrag sna robustness <persona> --network entities --strategy betweenness --recompute --seed 1

# §22.3's cascade at several tolerances, and §22.4's coupling to a second network
graphrag sna robustness <persona> --network entities --cascade --tolerance 0.1 --tolerance 0.5
graphrag sna robustness <persona> --network speakers --couple speakers-entities --coupling same-id
```

**The four sections.**

*Critical fraction (§22.1).* κ = ⟨k²⟩/⟨k⟩ and f_c = 1 − 1/(κ − 1): the share of nodes random
failure has to take before the giant component goes. It carries the **second** moment, so a heavy
degree tail raises it without raising the mean degree — which is §22.1's finding that skewed
networks shrug off accidents, *"it is extremely unlikely to pick the hub"* (p. 317) — and it is
the threshold for a *random* network with this degree distribution, so it belongs beside the
simulated curve rather than instead of it. Neither the criterion nor the closed form is written
in chapter 22: the chapter states the phase transition and cites Cohen et al. 2000 (footnote 6,
p. 318). The book's own "Molloy-Reed approach" (§18.1, p. 258) is the configuration-model
algorithm, a different object.

*Removal curves (§22.1–22.2).* One column per strategy: `random`, averaged over `--runs`
independent orders with a 10–90 band, and one per `--strategy` (any centrality; degree is the
chapter's own attack, *"prioritizing attacks to the nodes with the highest degree"*, p. 318).
Random removal is the null — a behavioural one, not a statistical one — so a targeted curve is
always quoted as a difference from it. Three summaries per curve: the **area under the curve**
(this package's, not the book's; ceiling 0.5 because the share is of the original n), the
**collapse fraction** (the first fraction at which the largest component no longer holds half of
the nodes *still standing*, which is the chapter's "critical value of |R|" read off the curve)
and the **shatter fraction** (where the mean non-giant component peaks, p. 315's second signal).
`--recompute` asks the ranking again on the survivors at every step, which is the stronger
attack; the report always says which of the two was run.

*Cascade (§22.3).* The chapter's load-capacity model: a failed node's load is *"equally
distribute[d]"* over its surviving neighbours and a node whose new load exceeds its capacity
fails at the next step (§22.6 exercise 4; figure 22.8's caption). `--load degree|betweenness` and
`--tolerance` (capacity = (1 + tolerance) × the initial load) are stated assumptions, not
measurements: nothing is carried between two entities that co-occur. The table sweeps the
tolerance because the outcome is a step and not a slope, and prints the **branching** factor —
the average out-degree of figure 22.9's cascade tree, whose critical value is 1.

*Interdependent networks (§22.4).* `--couple <network>` makes two of the persona's networks
depend on each other node for node and runs the failures back and forth to a fixed point. The
mutual giant component is printed beside what each layer alone would have kept, which is the
chapter's claim — *"if you were to calculate the critical |R| value for each layer separately
you would obtain a result much higher than the one for the interdependent network as a whole"*
(p. 323). The collapse is first-order, so the whole curve is printed and never one endpoint.
`--coupling same-id` pairs the nodes the two networks share by name; `random` is the chapter's
model assumption (p. 324) and `degree` the perfect correlation the same page says would restore
the robustness. Nodes with no counterpart are outside the analysis and the section's shares are
of the coupled nodes alone, which is a narrower frame than the rest of the report.

**What none of this is.** A corpus network is not an infrastructure. Nothing flows through it,
nothing fails in it, and a node cannot be attacked: removing a speaker removes a *record*, not a
person. These curves measure how concentrated the evidence is — how much of the co-occurrence
structure survives losing its best-connected records — which is a real question about a corpus
and not a statement about anything breaking. The chapter says the same about its own example:
*"You can use these methods in other scenarios, but only if you're sure that the assumptions made
here are respected in the phenomenon you're studying"* (p. 315).

**Deliberately not built.** Edge failures, which the book raises and drops in the same breath
(*"the underlying math is rather similar ... For this reason we keep looking at node failures"*,
p. 318). The α-dependence of figures 22.4, 22.7 and 22.12, which needs a generator taking the
degree exponent as an argument; `sna.generators` has preferential attachment (α fixed at 3) and
the configuration model, so a caller who builds the degree sequence can draw it. The cascade-size
exponent α/(α−1) of p. 322, a claim about an ensemble rather than about one network. And the
threshold form of failure propagation (*"a node V failing if a fraction f_v of its neighbors are
failing"*, p. 320), which is §21.1's model and belongs to the spreading chapter.
## Core-periphery and nestedness (Atlas ch. 32)

Every `sna analyze` report now ends its grouping with a `## Core-periphery` section, because
chapter 32 describes the one mesoscale organisation community detection cannot see and is
guaranteed to mistake for communities. *"Many large scale networks have a common topology: a very
densely connected set of core nodes, and a bunch of casual nodes attaching only to few
neighbors"* (p. 450) — and §32.2 states the incompatibility outright: *"in CP there isn't space
for communities, given that there's only one dense area and everything connects to it. In CD,
there's little space for peripheries, and there are multiple cores."*

**Both models of §32.1, scored the same way.** The discrete model is Figure 32.1: nodes are core
or periphery, core-core and core-periphery pairs are joined, periphery-periphery pairs are not.
The continuous model replaces that 0/1 mask with `c_u c_v` for a coreness vector, so a
semi-periphery can exist without anyone deciding how many classes there are. Both are scored by
the book's comparable quality, *"the Pearson correlation coefficient between A and ∆"* (p. 452) —
the raw sum `Σ A_uv ∆_uv` cannot be compared between two networks, and worse, it is maximised by
putting every node in the core. The correlation refuses that answer, because a mask that marks
every pair is constant and a constant correlates with nothing.

The chapter names no algorithm for the discrete fit — *"genetic algorithms, simulated annealing,
or basin hopping"* (p. 452) — so ours is stated in the report instead: the best prefix of the
degree ranking, then steepest-ascent single-node flips, repeated from eight random restarts. It is
a local search and says so. For the continuous vector the chapter lists options and picks none;
ours is the leading eigenvector of the adjacency, which is not a proxy but the exact maximiser of
the book's own quality `c^T A c` over unit-length vectors (§5.5). On a disconnected network that
vector concentrates on the densest component, which is a fact about eigenvectors rather than about
the corpus, and the report says so.

**k-core is printed beside it and is not the same claim.** §32.1 leaves k-core centrality out of
the chapter deliberately, because it *"finds a fundamentally different type of core-periphery
structure, which is more similar to a hierarchical decomposition of the network"* (p. 450). The
report gives its correlation as the chapter's *a priori* option, the shell sizes, and the Jaccard
between the deepest shell and the fitted core — which is how far apart the two answers are on your
network. Never quote a k-core number as evidence of a core-periphery structure.

**The rich club is a ratio, never a curve.** φ(k), the density among nodes of degree above k,
rises with k on every network including a random one. What makes it a finding is comparative:
*"the surprising part is that the cores of some empirical networks are even denser than what you'd
anticipate by looking at the degree distribution of the network"* (p. 450). So the column to read
is ρ(k) = φ(k) / mean φ over degree-preserving rewirings, and the verdict needs ρ above 1
sustained over the high k rather than at one threshold where the club is three nodes. (The formula
itself is not in the chapter, which only names the phenomenon and cites Zhou–Mondragón and Colizza
et al.; it is theirs.)

**The tension check can overrule the communities.** On a Louvain run the report fits both ideal
patterns of Figure 32.4 to the same adjacency — the core-periphery one and the community one,
`∆_uv = 1` inside a group — takes one correlation each, and prints which is larger. When the
core-periphery ideal wins, the report says the network is core-periphery and the communities are
the periphery cut into slices; name the core and the size of the periphery, and stop. Both
coefficients are also scored against degree-preserving rewirings, with each model *refitted* on
each rewired network, and those excesses are printed but are deliberately not the verdict: the
degree-preserving null of a core-periphery network is itself core-periphery — p. 450's own point
— so the core's excess is near zero for exactly the networks this chapter is about, and a verdict
on it could never say "core-periphery". Read the excesses as the other question: whether either
structure is more than the degree sequence. A strongly core-periphery network often has *no*
degree-preserving null at all, because a dense core leaves an edge swap nowhere to land; the
report then prints dashes and a sample count of zero rather than a z-score of 0.00.

**Nestedness (§32.4) is the two-mode form of the same thing.** On the two-mode network —
`--network speakers-entities` without `--project` — the report adds a `## Nestedness` section.
*"A nested system is one where the elements containing few items only contain a subset of the
items of elements with more items"* (p. 458): the sorted matrix is upper-triangular (Figure 32.9).
The chapter describes temperature and the isocline but gives no formula, so the number reported is
**NODF** (Almeida-Neto et al. 2008), the measure that replaced temperature in the field the idea
comes from — 100 for a perfect nest, 0 for a checkerboard — with the chapter's own procedure
beside it as the count of ones outside the non-parametric isocline (*"how many mistakes you
made"*, p. 459). Temperature itself is not built: its normalisation is a set of conventions the
chapter does not state.

NODF needs its null more than most numbers do, and it is the fixed-fixed one: curveball trades
that hold every row **and** every column total (`null.bipartite_preserving`). §32.4 reports the
objection itself — nestedness *"could arise simply from the degree distribution ... and that there
are fewer nested system than we originally thought"* (p. 458) — and the Southern women network is
the demonstration: NODF 48.6 against a null mean of 48.4, so the canonical nested matrix is not
nested beyond its own margins. Read the z, not the NODF.

**Not built from this chapter.** Rombach et al.'s α/β coreness vector and the p-normalised masks
of Figure 32.3 (`∆_uv = c_u + c_v` and the p-norm), which the chapter offers as *"another freedom
you can take"* rather than as the model; the Expectation-Maximization and Belief-Propagation
approaches of p. 453; the random-walk core detection of p. 454; multilayer core-periphery, which
the chapter itself calls *"an open problem in network science"*; and §32.3's generative account —
Hotelling's law, ekistics, central place theory — which is a reading of why cores emerge and has
no number in it. Kojaku and Masuda's remedy for the tension (find communities first, then run
core-periphery detection inside each) is a nested version of what is here and belongs with the
community tickets.

## Visualization (Atlas ch. 49-51)

```bash
# force-directed, sized by betweenness (which also sizes and colours the edges), coloured by
# community; writes both network.svg and network.html
graphrag sna draw <persona> --network entities --min-weight 2 --layout force \
    --size betweenness --color community --seed 1 --out /app/data/exports/network.svg

# circular, ordered by the same colour rule, plus the positions/sizes/colours as JSON
graphrag sna draw <persona> --network entities --layout circular --color attr:region \
    --seed 1 --out /app/data/exports/by-region.svg --json /app/data/exports/by-region.json

# the adjacency matrix, ordered by community: two disjoint communities read as two blocks
graphrag sna draw <persona> --network entities --layout matrix --color community \
    --seed 1 --out /app/data/exports/matrix.svg

# the layering sna hierarchy already computes, drawn (needs a directed network)
graphrag sna draw <persona> --network relations --layout layered \
    --out /app/data/exports/hierarchy.svg
```

`--layout` is `force` (seeded Fruchterman-Reingold, §51.1), `circular` (equal angles, ordered by
`--color` when there is one, §51.2), `arc` (one line, every edge an arc above it), `matrix` (an
adjacency matrix ordered the same way, §51.4), or `layered` (reuses `sna hierarchy`'s own §33.6
layering; only `--network relations` has directions to layer). `--size <centrality>` (any name
`sna analyze`'s ranking uses) maps the value onto node *area*, quasi-logged first so one hub
cannot swamp the picture (§49.1) — never radius, which would quadruple the drawn size for every
doubled value (Figure 49.8). `--color community` runs Louvain (§35.6, imported rather than run
twice) and prints `## Community evaluation` (ch. 36) for the partition it coloured; `--color
attr:<key>` reads a node attribute an attribution/extraction sidecar wrote (§7.5) — categorical
if any value is not a number, sequential if every one is. Both palettes are colour-blind-safe
(Okabe and Ito 2008, standing in for the Color Brewer tables the book recommends and this
package does not vendor) and stop at nine colours total between node and edge (§49.2 p.719,
§50.1 p.729): a category beyond the top eight, or a node with no value at all, is drawn in a
neutral grey and counted in the notes rather than silently absorbed.

**Edges follow the chapter's own "network lifting" order (§50.3), fixed rather than a flag per
step.** A constant transparency applies regardless of any attribute (Figure 50.4(b): "even if I
added literally zero bits of information by removing some edge opacity" it still declutters a
dense network); width and colour both come from the same measure so they reinforce each other
(Figure 50.3(a)) — edge weight normally, or edge betweenness when `--size betweenness` is asked
for, the chapter's own pairing (p.727); then `--size` and `--color` style the nodes on top. A
reciprocal pair on a directed network (`u -> v` and `v -> u`) is drawn as two bent arcs rather
than merged into one double-headed line (Figure 50.5(a) — the book's own excuse for not merging
them, "not obvious to do with standard software", applies here too).

The HTML written beside the SVG fetches nothing: a node's name is its native tooltip, dragging a
node moves it and the straight edges attached to it follow live (curved and bundled edges do
not, a limitation of building this without a routing library), and a legend names every colour.
`xml.etree` parses the SVG; `svg_string in html_string` holds exactly, because the HTML embeds
the SVG verbatim rather than re-deriving it.

**§51.5's case studies are documented recipes, not flags.** The Product Space's stretched,
un-circularised force layout and the Cathedral's two-level functional scatter are bespoke,
hand-tuned pictures the book itself calls "custom... no one should really follow your workflow"
(p.751) — a generic "stretch this layout along an axis" flag would force that axis onto every
corpus network or do nothing for the ones that have none. The recipe instead: run `sna draw
--layout force --seed <n> --json positions.json`, then read `positions` back in a notebook
against `sna analyze`'s own centrality columns (`graphrag.sna.export.read_graph` and a plotting
library of your choice do the stretching); for a Cathedral-style plot, `--color attr:<key>` on a
two-level functional attribute plus two centralities from `sna analyze` are the same two axes the
book's own scatter uses, assembled by hand rather than by a flag that assumes the corpus has that
shape.

**Not built.** Real edge bundling (Holten 2006; Holten and Van Wijk 2009) and organic/orthogonal
edge routing against ghost edges (§51.3, Dwyer et al. 2006) — `--layout circular` approximates
bundling with a fixed bend toward the centre and `--layout arc` bows every edge above its line by
construction, but neither routes around a node the way the book's own fix does. Hive plots and
graph thumbnails (§51.4) — both need a second axis (a class per node, a k-core decomposition)
this package's networks do not carry the way the book's examples do. Also from §51.4: probabilistic
layout — needs repeated force layouts over a sample and a spatial-probability contour per node,
and the data-level version of "approximate a large network" is what `sna sample`'s real samplers
already do; revealing matrices — one bipartite sub-network per cell, a matrix-of-networks drawing
rather than one network's matrix; and timelines — needs per-edge timestamps at a granularity a
passage-level corpus does not carry, since a passage can name several co-occurring edges at once,
not one edge per moment. Pie-chart nodes for overlapping community membership (§49.3) — `sna
overlap` already reports the covers a pie chart would draw; rendering them is future work, not a
gap in the analysis. Also from §49.3: node borders as a second encoding (colour plus thickness) —
node size and node colour already spend the one quantitative and one qualitative slot this
package's networks have, and a third channel risks the overload the chapter itself warns against
the next page over (p.725); the xenographic node-shape distinction (circle vs square for a
bipartite network, a symbol per type) — needs an icon or a second drawn shape this hand-rolled SVG
renderer does not have, so every node it draws is a circle; and the alpha-channel invisible-node
trick (Figure 49.17) — useful only for a network whose nodes carry nothing worth naming, and every
node this package draws is exactly what a report already names. A diverging colour gradient with a
meaningful midpoint (§49.2's other half of Figure 49.9) — nothing this package colours has a
midpoint that is not itself a modelling choice a caller would have to supply.

## Interchange formats

`sna export --out <path>` writes the format the suffix names, and `graphrag.sna.read_graph` reads
any of them back. `.graphml` and node-link `.json` carry everything -- node attributes with their
`attr_*`/`attrn_*`/`attrsrc_*` provenance, edge weights, direction, isolates, and the `graph.graph`
frame the report prints. `.gexf` (Gephi) keeps attributes but drops the frame; `.net` (Pajek) keeps
structure and weights only; a tab-separated `.edgelist`/`.txt` keeps edges and weights and loses
isolates; a `.csv` writes a `<stem>.nodes.csv` and `<stem>.edges.csv` pair Cytoscape imports, and
reads cell text back typed. Every format except node-link JSON returns node ids as text, so an
integer-keyed graph comes back keyed by `"0"`, not `0`. A name a format would silently mangle (a
leading `#` in an edge list, a `"` in Pajek, a tab anywhere) is refused on write rather than lost
on read. `graphrag.sna.FORMATS` is the table of what each format carries; `to_igraph` and
`to_graph_tool` hand a network to those libraries when they are installed, and neither is a
dependency. See the Atlas, chapter 53.

## Reading a report

**The network type, above everything.** Every report opens with the type in the Atlas's words
(ch. 6): `undirected, weighted` for the co-occurrence networks, with `bipartite` added for the
two-mode one, and `directed, weighted` for `relations`. A comparison prints the same line. On a
directed network:

- the centrality table splits degree into `in_degree` and `out_degree`, and their weighted
  forms. `degree` and `weighted_degree` still exist and are the **total** there, in plus out;
  that is what the community member lists and `compare`'s rank table are ordered by, and their
  captions say so.
- `closeness` is networkx's **incoming** closeness -- how near everyone else is to this node,
  not how near it is to everyone. Reverse the graph to ask the other question.
- the summary carries `reciprocity`, the share of connected pairs stated both ways (§10.3), and
  its `components` are the weakly connected ones.
- two summary numbers keep their names and change definition: `degree_assortativity` is the
  out-in coefficient and `average_clustering` is the directed (Fagiolo) one. A caveat line says
  so on every directed report. `transitivity` is a third case: §12.2's global coefficient is
  undirected and networkx reads a directed triangle as 0.0, so it is computed on the flattened
  view and the `## Density` section says so.
- every measure defined only for undirected graphs -- Louvain and its null, the eigenvector, the
  broker scores, the spectral features, the `--by` section -- says in the caveats that it ran on
  a flattened copy, in which an edge either way becomes one undirected edge carrying the total
  weight.

**The summary table first.** `nodes` is the n every other number depends on. A high
`components` count with a low `largest_component_share` means the network is really several
networks, and a centrality ranking across them compares nodes that cannot reach each other.

**The `## Density` section (Atlas ch. 12).** Four things a summary table cannot say on its
own. The density is printed as "m of the M edges n nodes could carry", because the possible
edges grow quadratically with the nodes while a real network adds a few per node, so density
falls with size on its own (§12.1) and two networks of different sizes cannot be compared by it
-- including one network before and after a filter, a window or a sample. The **average**
clustering coefficient (the mean of the per-node ones) is printed next to the **global** one
(transitivity, 3 x triangles / triads), with the gap explained, because they are different
questions and a hub-heavy network makes them disagree by a lot; the weighted row beside them is
Onnela's, which on an **undirected** network is the number the summary table calls
`average_clustering` -- on a directed one it is not, because the table keeps Fagiolo's directed
coefficient there, and the section says so. Neither has a null model: chapter 12 defines none,
and the book's own "150 times higher than random" reading needs a baseline this section does not
compute -- `graphrag.sna.null` is where one comes from. The largest maximal cliques are listed
with their members, and are usually one document -- a passage naming m entities is an m-clique by
construction -- and the independent set is greedy and maximal, so its size is a lower bound on
the maximum one (§12.4 separates the two; that finding the maximum is NP-hard is Karp's result,
not the chapter's, as the 3^(n/3) bound on maximal cliques is Moon and Moser's).

Three things the section says for itself rather than leaving to a reader. On a **directed**
network everything in it is computed on the flattened view *except the density*, which keeps
§12.1's directed denominator |V|(|V|-1), because flattening would halve it and describe a
network nobody built. **Self loops** are not in the numerator, since §12.1 counts pairs of
distinct nodes ("seriously, no self loops!"). And on a **two-mode** network
(`--network speakers-entities` without `--project`) every coefficient is 0 and every maximal
clique is a single edge *by construction* -- a triangle needs three mutually adjacent nodes and
every edge there crosses the modes -- so the section says that rather than printing the zeros as
findings, and points at `--project`; §12.3's object for that case is the biclique, which is not
computed.

**"Against random" (Atlas ch. 16-17).** Every report carries it, and it is the section that turns
the summary table into findings. Chapter 17's claim about real networks is that they are
*clustered, short and broad* — and each of the three is a comparison, never a number:

- **clustering** against `p`, the connection probability of a random graph with this n and m,
  which is also its density (§16.5). The karate club's 0.5706 against an expected 0.1390 is 4.1x.
- **average path length**, in hops, against `ln n / ln k̄` (§16.4) — 2.408 against 2.315 on the
  karate club, which is the property random graphs already have, so matching it is not news and
  sitting well above it is. On a disconnected network the observation is taken on the largest
  component, the closed form is recomputed on that same component, and the section says so.
- **the degree distribution** against the Poisson with the same mean (§16.2): variance equal to
  the mean, so a variance-to-mean ratio of 1, and a maximum degree a random graph would rarely
  reach. The karate club's ratio is 3.18 and its hub has 17 edges where a random graph would stop
  around 9. That makes it *broad*; it does not make it a power law, and this section never fits
  one.

Two nulls, because the closed forms are the weakest baseline there is. Beside the G(n,p) column
is the same measurement on `--samples` degree-preserving rewirings, which keep every node's
degree and free everything else: a clustering that beats p but not that null is clustering the
degrees already explain. The degree rows have no null column on purpose — the rewiring *is* the
degree sequence, so testing them against it would score a number against itself.

"Short" is at most twice `ln n / ln k̄`, which is where the book's own exercise puts the line:
its small world of 100 nodes reads 1.7x and the caveman graph it is contrasted with reads 2.8x,
and those two have to get different words. "Clustered" is at least twice p *and* z >= 2 against
the rewirings; "broad" is a variance-to-mean ratio of at least 2. The three thresholds are
conventions the book does not state, and they are exported as `CLUSTERED_RATIO`, `SHORT_RATIO`
and `BROAD_DISPERSION` so you can disagree with them in one line.

**What it costs.** The closed forms are free; the null sample is not. Per rewiring the section
measures an average clustering coefficient and one breadth-first search from 64 sources, and
drawing the rewiring costs about as much again — so the report draws **one** family of rewirings
and spends it on both this section and Louvain's modularity null rather than paying §19.1's edge
swaps twice. On a network of a few thousand nodes and tens of thousands of edges expect tens of
seconds at the default `--samples 50`; lower `--samples` while you are still iterating.

The z-scores point in the direction of the property: right tail for clustering (more triangles
than the null), left tail for path length (fewer hops), so a positive z on the path row means
this network's paths are *longer* than its own degrees imply. The numbers are unweighted and
undirected, because chapter 16's formulas count triangles, neighbours and hops — which is why the
clustering here differs from the weighted `average_clustering` in the summary table above it —
and above 64 nodes in the largest component the path length is estimated from 64 random sources,
by the same estimator on the observation and on every null sample.

**The `## Evidence behind the edges` section (Atlas §28.1).** Printed in every report, before
the rankings, because it says what the rankings are made of: how many edges rest on a single
passage (or document, or stated relation — the row names the network's own unit), how many of
those rest on a match that was only a token, what share of the total weight they carry, and how
many of the mentions behind the network carry no tier at all. With `--uncertain` it is followed
by `## Under uncertainty`, where every centrality is a mean and a 95% interval over sampled
possible worlds and adjacent ranks whose intervals overlap are marked as ties. Neither has a null
model, and neither should: this is the spread of a measure over the worlds the evidence allows,
not a test against chance. See [Uncertain edges](#uncertain-edges-atlas-ch-28).

**Ranking stability and centralization (Atlas ch. 14).** The `## Ranking stability` section
sits under the centrality tables and is about them. Its first half is Freeman's
**centralization** (§14.8) for every ranking: the summed distance of every node from the most
central one, over the same sum on a star of the same size. 1.0 is a star, 0.0 is a network whose
nodes all score alike, and it says nothing about *who* is central -- one number about the shape.
It is computed on the binary view of the network (every weight 1), because the star it is
divided by has no weights and the ratio would otherwise move when the weights were rescaled and
the shape was not, so it is the one number in the report that is not a function of the table
above it. Different centralities give different answers on the same network -- the chapter's own
figure has degree and betweenness disagreeing about which of two networks is more centralized --
so the whole table is printed and a claim has to name its row. Some rows have no centralization
at all and say why in place of a number, because the chapter's maximum is only *usually* a star:
every node of a star has coreness 1 and reaches everything, so the denominator is zero for
`coreness` and for `reach` on an undirected network; and on a network in more than one piece the
principal eigenvector belongs to one component, with every node outside it scoring ~0, so the
observed sum can exceed the star's and `eigenvector` is refused there too (PageRank's
teleportation is §14.4's own answer to that problem, so read that row instead).

The second half resamples the network and asks whether the names above would still be printed.
By default each resample draws edges uniformly until 80% of the nodes are held and induces the
subgraph on them (§29.1's edge sampling), which asks what survives seeing less of the corpus;
`ranking_stability(..., method="configuration")` asks a different question -- a
degree-preserving rewiring holds the degree ranking fixed by construction, so it tests whether a
rank is anything beyond the degree -- and `method="backbone"` a third. Each row gives the
Spearman and Kendall correlation of the whole ordering (not on the same scale as each other,
§3.4), the share of the printed top 20 still in the resample's top 20, and then the names split
in two: the ones that held in at least 90% of the resamples, and the ones within the noise,
which belong in a sentence as "also in the top twenty" and never as ranks. The section is
budgeted: above 500 nodes only the rankings that cost O(m) are checked, because every other one
is all-pairs shortest paths or a dense eigendecomposition and this section would pay for it once
per resample. The rest are named, with that reason, rather than dropped.

**The `## Weak ties` section (Atlas §30.3).** One row per weight quantile, holding the mean
neighbourhood overlap of the edges in it, plus Spearman's rho between the two. Read the rho
first: rising overlap is Granovetter's structure, and then the zero-overlap edges listed under
it are the bridges the corpus would lose most by losing. A flat or negative rho is not a failure
of the network -- the chapter says the anti-correlation is equally constructible -- it means the
weights are not telling you where the edges sit, and the bridges have to be read off the overlap
column instead. Edges whose two endpoints have no other neighbour are undefined rather than 0
and are counted out loud: an isolated pair is not a bridge.

**Stability (adjusted Rand index between Louvain runs).** 1.0 means every seed found the same
partition. Above 0.9 you can name individual members. Between 0.6 and 0.9 the large groups are
real but the boundaries move, so quote groups and not the membership of any one node. Below
0.6, report that there is no stable structure rather than naming communities.

**The null model (z-score).** The observed modularity is compared with what Louvain achieves on
graphs that have the same degree sequence but random edges. Below 2, the partition is what any
graph of this shape would produce and means nothing. Above 3, the structure is real.

**The `## Community evaluation` section (Atlas ch. 36).** Printed under the communities for every
method, because a partition of the graph has a modularity and a conductance whether or not it was
found from the edges. It says which of its communities are small enough for modularity to want to
absorb them, prints every §36.2 measure with the direction community size pushes it, and scores
the partition as a link predictor against the same sizes shuffled. See
[Community evaluation](#community-evaluation-atlas-ch-36).
**The `## Core-periphery` section (Atlas ch. 32).** It sits immediately after the null model,
because it is one more check on the partition and can overrule it: when the core-periphery ideal
of Figure 32.1 reproduces the adjacency at least as well as the communities do, §32.2 says there
were no communities to find and the groups above are the periphery cut into slices. Read the
verdict line before you read the community table. The section also carries the rich club as the
ratio it has to be (rho, not phi) and, on a two-mode network, `## Nestedness` with the
fixed-fixed null under it. See
[Core-periphery and nestedness](#core-periphery-and-nestedness-atlas-ch-32).

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

**The attribute section (`--by`).** Read it top down, because it is ordered by what it can
support. The value counts come first: a value with four nodes in it supports nothing, and the
untagged count says how much of the network is outside every number below. Then assortativity
against a permutation null, which holds the graph and the labels fixed and moves only which node
holds which label, so the z-score is about this attribute and not about the network having
structure. Then the modularity of the attribute partition against two things at once: a
degree-preserving null, which says whether it beats chance, and Louvain's best grouping of the
same nodes, which says whether it is the division in the network or merely a division. Near-zero
assortativity beside a strong Louvain result is a finding: the network divides, and not along
this. Last, the adjusted Rand index and mutual information between the run's groups and the
labels, each with the baseline a shuffle produces.

An entity or a topic inherits its value from the documents its passages sit in, which can
disagree; the section says how many nodes were mixed, and a large number there means the
attribute describes documents and the entity network is the wrong place to ask about it.

**The numeric attribute section (`--by <number>`).** A different section with a different
contract, printed under `## By attribute: <key> (quantitative)` — see "Quantitative assortativity"
above for what each of its four parts says and which null it ran. Three things to check before
quoting anything from it: whether the values are heavy-tailed (the summary at the top says so,
and a Pearson coefficient on a heavy tail is carried by a handful of nodes), whether the numbers
were borrowed, and whether the key is one of the corpus's own counts. `--null bipartite` is a
null for the categorical section only and is refused here rather than ignored.

**The second null (`--null bipartite`).** Every one of these networks is a projection of a
membership table: who was recorded in what. The label shuffle holds the projected graph fixed,
which means the cliques a single long document creates, and the fact that the nodes recorded
everywhere meet each other constantly, are inside the observation rather than inside the null.
`--null bipartite` rewires the memberships instead — keeping every node's number of documents and
every document's size, by the curveball trade of §18.1 — and projects again. An attribute that is
really "was recorded a lot" scores far above a label shuffle and lands squarely inside this null;
where the two z-scores disagree, the gap is the corpus's own shape.

Labels move with the corpus, by provenance. A value recorded about the node is pinned to it. A
**borrowed** value — one the node took from the documents its passages sit in — is derived again
from each rewired corpus, by the same majority rule the network was built with, so the copying
that makes a borrowed label circular happens inside the null as well. That is what a label
shuffle cannot do: shuffling breaks the copying, prices it at zero, and hands a confounded
attribute the largest z-score in the corpus. On the planted two-clique corpus the shuffle reports
z = 8.5 and this null reports z = 2.1 with p = 0.2, which is the honest reading of a number that
a corpus of that shape produces whenever it is rewired.

Two limits. It tests the *labels*, never the tagging: a borrowed label that survives it is still
a value nobody recorded about the node, so the borrowed-label verdict stands either way. And
re-deriving needs the table of what each passage's document was tagged with, which only the
entity network keeps — the speakers-entities projection collapses speakers against entities, with
the documents as a weight rather than a mode, and a network read back from a file has no corpus
behind it at all. Where it cannot re-derive, the labels are pinned and the section says so. The
network also has to be a projection, so the topic network refuses `--null bipartite` outright.

**The stance report.** Every number in it counts annotations, not mentions. An entity with no
complaints has no annotated complaints, which is not the same as no complaints, and the
quotations under each count are the evidence: if a passage does not read the way its count
claims, the annotation is wrong.

**The `## Provenance` block, last (ATL-ENT-4).** Every report ends with one: `method` (the
command), `parameters` (every argument it ran with, as JSON, sorted so the CLI and an MCP tool
call print the identical line), `seed` (or "not fixed"), `null model` (the report's own headline
structural claim, where it has a single one -- most sections already name their own null in
their own words above this; this line is a one-line summary, not a replacement), `tool version`,
`generated` (an ISO-8601 UTC timestamp), and `snapshot commit` (the repository commit
`graphrag snapshot export` recorded when it wrote the persona's committed snapshot, or "unknown"
for an in-memory store or a persona never exported). The JSON payload carries the same fields
under a `"provenance"` key, built from the exact values the command ran with rather than
re-derived from the report afterwards, so the two never disagree about what produced them.
`sna export`'s payload is the network itself (`nx.node_link_data`), not a report, so its
provenance rides on `graph.graph["provenance"]` instead -- a JSON *string*, because GraphML
attributes cannot hold a nested value. Only `.graphml` and node-link `.json` carry `graph.graph`
at all (the [Interchange formats](#interchange-formats) table above); GEXF, Pajek, the CSV pair
and the edge list all drop it, along with the sampling frame it sits beside, exactly as they
already drop everything else in `graph.graph`. One file shape changed when the block arrived: `sna predict --json` used to write a bare list of hypotheses and now writes `{"hypotheses": [...], "provenance": {...}}`, the one payload that was not already keyed.

`--cite`, off by default, appends one more block, `## References`: the book's own citation once,
then one line per chapter `provenance.chapters` lists -- sorted, deduplicated, in the book's own
words (title and page range, read from its outline rather than retyped) -- for a report that is
going into a citation-checked document. `graphrag sna guide` and `sna cache *` are not reports
and have neither block.

## Choosing a network and a method

### Which network

- **speakers** — nodes are the speakers attached to a persona's documents; an edge is a shared document, weighted by how many they share. Answers: whose voice appears alongside whose, and who bridges otherwise separate groups.
- **entities** — nodes are the entities an extraction pass named; an edge is a shared passage, weighted by how many passages mention both. Answers: what a thing is discussed alongside, and which subjects hang together.
- **topics** — nodes are the topics assigned at ingestion; an edge is the stored co-occurrence weight. Answers: how the corpus was labelled, as a first map before any extraction exists.
- **speakers-entities** — nodes are both the speakers and the entities their passages mention, as two modes; an edge is a passage by that speaker naming that entity, weighted by how many documents hold one. Answers: who wrote about what; projected onto one side, who wrote about the same things, or which things the same people wrote about.
- **relations** — nodes are the entities an extraction pass related to one another; an edge is a stated relation, running from the entity doing the relating to the one related to, typed and weighted by how many passages state it. Answers: what the corpus asserts about its entities and which way round, so who acquires, competes with or reports to whom rather than merely who is discussed beside whom.

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

**BRIM** (takes edges, on a two-mode network only)

- Use when the network is speakers-entities without --project and a fixed number of modules is an acceptable thing to search over. It maximises Barber modularity directly on the incidence by alternating which module every row, then every column, belongs to (§39.3), never on a projection.
- Choosing k: searched over --k-range and kept at the value with the highest Barber modularity, unless --k pins one; BRIM, unlike Louvain, cannot find k on its own.
- Watch out: it is an alternating optimisation and gets stuck in local optima, so --restarts worth of random starts are compared per k and the stability score says how much they agreed; refuses a network too large for a dense incidence matrix, where biLouvain is the one to reach for instead.

**biLouvain** (takes edges, on a two-mode network only)

- Use when the network is speakers-entities without --project and the number of modules should fall out of the graph rather than be searched, the way plain Louvain's does.
- Choosing k: no k to choose, same as Louvain -- --resolution is the only knob, with the same reading (above 1.0 smaller modules, below 1.0 larger ones).
- Watch out: it moves nodes locally and never coarsens communities into supernodes for a second pass, unlike unipartite Louvain, so --runs different visiting orders substitute for that second pass; run many and check the stability score before naming a node's module.

**Neighbour similarity** (takes the incidence matrix, as a data table)

- Use when the question is which rows and columns look alike -- share neighbours -- rather than which are unexpectedly densely connected. It does not optimise Barber modularity and is not a worse or better BRIM, but a different question (§39.4).
- Choosing k: searched over --k-range by Barber modularity, exactly as BRIM's is, unless --k pins one.
- Watch out: co-clustering, not modularity maximisation, so a low Barber modularity here is not a defect; it needs a dense incidence matrix, the same guard BRIM has, and drops zero-degree nodes before clustering rather than assigning them anywhere meaningful.

### Which centrality

- **degree** — local prominence: how many distinct others a node sits with.
- **betweenness** — brokerage: how often a node lies on the path between two others.
- **eigenvector or PageRank** — influence among the influential.
- **closeness** — reach: how near a node is to everyone else.
- **harmonic** — the same question as closeness on a network that is in pieces: the sum of inverse distances, which counts an unreachable node as 0 rather than leaving the measure undefined (§14.6). Prefer it whenever the network has more than one component.
- **reach** — command: the share of the network a node can get to within k hops, following the edges the way they point (§14.3, k = 2 here). It is the one ranking that reads out of a node rather than into it, so it only says something different from the others on a directed network -- on a connected undirected one, unbounded, every node reaches everything.
- **HITS (hub and authority)** — two rankings, and only on a directed network: a hub points at the good authorities, an authority is pointed at by the good hubs (§14.5). Undirected they are the same vector, which is the eigenvector centrality, and `sna` refuses rather than printing it twice.
- **coreness** — how deep into the network a node sits: the largest k whose k-core still holds it (§14.7). It is a floor, not a rank -- many nodes share a shell -- so read it as membership of the dense middle.
- **weighted variants** — whenever edges carry counts rather than mere presence, which here they always do.

### Density (clustering, cliques)

- **A density is only readable against n** — the possible edges grow quadratically with the nodes while a real network adds a few edges per node and no more, so density falls as a network grows (§12.1). The book's own examples run from 0.05% for the power grid to 0.003% for the Internet backbone, which says nothing about either being less connected. Print the node count in the same sentence, and never compare the densities of two networks of different sizes -- including the same network before and after a filter, a window or a sample.
- **Average and global clustering answer different questions** — the average is the mean of the per-node coefficients, so every node counts once however small its neighbourhood; the global one, transitivity, is 3 x triangles / triads over the whole network, so a hub with many unacquainted neighbours opens triads quadratically and pulls it down. They disagree hardest on exactly the hub-heavy networks a corpus produces -- the book's example graph reads 0.5 global against 0.648 average -- and §12.2's instruction is to remember they are two different things "and not to report one as the other". Name which one you quoted. The weighted coefficient is a third question again: Onnela's counts a triangle by the geometric mean of its edge weights, so it falls when the triangles are made of ties seen once.
- **High clustering is a comparison, not a value** — §12.2's finding is that real networks are sparse *and* clustered, and it is stated as a ratio every time: the power grid's 0.1032 is "150 times higher than you would expect if its edges were distributed randomly". The report prints the coefficients without a baseline, because chapter 12 defines no null model. A coefficient is therefore not evidence of anything until it is put beside what a random graph with this degree sequence would give: `graphrag.sna.null` has the configuration model and `significance()` to do it with.
- **A clique here is usually one document** — every co-occurrence network in this package draws an edge between two nodes a passage named together, so a passage naming m entities contributes a complete m-clique on its own. The largest clique of an entity network is then a fact about the longest entity list in the corpus, not about a tightly knit group (§12.3). Check which documents the largest clique's members share before calling it a group, and raise --min-weight so an edge has to survive more than one passage.
- **An independent set is an approximation** — §12.4 warns not to confuse the *maximal* set, which is what the report prints and which no node can be added to, with the *maximum* one, which is the largest in the network. The report's is greedy, so its size is a lower bound and a bigger set exists somewhere; the chapter's own exercise settles for "approximate answers", and that computing the maximum exactly is NP-hard is Karp's result rather than the chapter's. Nor is it a claim that these nodes are the most unrelated things in the corpus: an absent edge is an absent sentence, so an independent set is as much a map of what nobody wrote down as of what does not go together.
- **The clustering coefficient is not clustering** — §12.2 stops to say this to computer scientists and it is worth repeating here: the clustering coefficient is a number about triangles -- how often the friend of my friend is my friend -- and has nothing to do with grouping similar rows, which in network science is community detection and is what --method louvain|kmeans|gmm does. A high coefficient is not "this network has communities" and a strong modularity is not "this network is clustered"; they answer different questions and can move in opposite directions on the same graph. Say transitivity when the triangles are what you mean, which is the word the chapter offers to dispel exactly this confusion.

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

### Bipartite community discovery (--method brim|bilouvain|neighbor-similarity)

- **Barber modularity, not the projected kind** — a two-mode community mixes speakers and entities in one group, so the ordinary modularity sum is wrong for it: every same-type pair it touches gets a negative expected contribution it should never have, because that pair could not have connected either way (§39.1, p. 561-562). `--method brim|bilouvain` optimise the amended sum directly on the incidence -- kukv/|E| over unlike-type pairs only -- and `## Bipartite community discovery` reports that number; the unamended one in `## Community evaluation` above it is printed for comparison, not as a second opinion, and its own note there says which of that section's other measures a two-mode partition breaks.
- **The null is the curveball, never the configuration model on a projection** — every Barber modularity here is read against `bipartite_preserving`, which holds fixed every speaker's and every entity's number of documents and re-deals the rest by curveball trades on the incidence (§18.1's two-mode case). Rewiring a projection instead invents edges no membership table could have produced, and rewiring the raw two-mode graph with the configuration model would break the very memberships the edges are made of; neither is offered.
- **Neighbour similarity answers a different question than BRIM** — co-clustering groups rows and columns that look alike -- share neighbours -- and does not optimise Barber modularity at all (§39.4, p. 568), so a lower modularity from `--method neighbor-similarity` is not evidence it did worse than `--method brim`; the two answer 'who looks similar' and 'who connects more than chance' respectively, and can disagree on the same network without either being wrong.

### Projection weights (--projection)

- **The projection is a modelling choice and the report names it** — chapter 26 exists because the weight on a projected edge is not a measurement: the same memberships give a weight of 3, of 0.79 or of 0.049 depending on which of the eleven schemes you picked, and each answers a different question about the same corpus. Every report here prints the scheme in its frame, including the default `simple`, which is as much a choice as the rest. A projected number quoted without its scheme cannot be checked and cannot be reproduced.
- **Simple weights inflate the hubs' rows** — the default counts shared documents, so one long document naming twenty entities joins all 190 pairs of them and one prolific speaker joins everybody they were recorded beside. That is §26.1's power user who watched every movie. `hyperbolic`, `probs`, `heats` and `hybrid` discount a shared node by how many others it touches, so the same corpus reads differently -- use one of them when the question is who is *specifically* connected rather than who was recorded a lot, and say which.
- **Compare schemes by rank, not by value** — the scales have nothing to do with each other, so 'hyperbolic is lower than simple' is arithmetic, not a finding. `graphrag sna projections` reports Spearman between every pair of schemes over one edge list, which is the only comparison that means anything -- and §26.6's own warning is that schemes you would expect to agree need not: on the book's Twitter data HeatS and ProbS, one the transpose of the other, came out anti-correlated.
- **Half the schemes are directed and this network is not** — resource allocation, HeatS, the hybrid and the random walk give u a different score for v than v gives u (§26.4): 'you might be the most similar author to me because I always collaborated with you, but if you also contributed to many other papers, I might not be the author most similar to you'. These networks are undirected, so the two are averaged, as §26.4 allows, and both are kept on the edge as `weight_uv` and `weight_vu`. That is an artefact of a normalisation, not a claim anybody made; `--network relations` is the only network here whose direction was written down. And because HeatS is the transpose of ProbS, averaging makes `--projection probs` and `--projection heats` the *same* undirected network: their difference is the direction, so read it off the two weights, or off `sna projections`, which compares the schemes directed for exactly that reason.
- **Projecting is the first step of two** — every scheme joins two nodes that share even one neighbour, so every projection is dense whatever the weighting: §26.6's own example is 175k memberships becoming 5.3M edges. Choosing a cleverer scheme does not thin a hairball; thresholding does, and doing it defensibly is chapter 27's `--backbone`. Report the scheme and the cut together, because neither one alone says what the network you measured was.

### Directed networks (--network relations)

- **An edge is a claim with a direction** — every other network here joins two nodes because a document held both, which is symmetric by construction. A relation has a subject and an object: (u, v) is not (v, u). Read in-degree as being named by others and out-degree as naming them, and never quote one as 'degree'.
- **Report reciprocity before reading the direction** — reciprocity is the share of connected pairs stated both ways. Near zero means the corpus describes its entities from one side only, which is usually about how the extraction prompt was written rather than about the world, and it decides how much the in/out split is worth reading at all.
- **A flattened measure answers a different question** — Louvain, the eigenvector, the broker scores, the spectral features and the --by section are all defined on undirected graphs, so on this network they run on a flattened copy and the report says so. Their groups are about who is related to whom, never about which way it ran.
- **An absent edge is an absent sentence** — the network holds what an extraction pass wrote down. No edge from A to B means no passage stated one -- not that the relation does not exist, and not that it was denied. Quote the passage behind an edge, and say how many documents were extracted at all, before reading any shape into the whole.
- **One edge per ordered pair** — two entities related in two different ways carry one edge here, typed with the relation most passages state and listing the rest. Parallel typed edges are a multigraph; filter with --relation-type when the question is about one kind of tie, rather than reading the fold as a count.

### Layers, hyperedges and time (--layers / --window)

- **Flattening forgets which layer an edge came from** — every measure runs on the flattened graph, so a community, a centrality or a modularity computed with --layers cannot tell a praise edge from a complaint one. §40.1 states the assumption this makes: collapsing which layer an edge appears in into an edge weight 'assumes that every edge type is equally important', and nothing about a corpus says its layers are. The per-layer table is printed above those numbers for exactly that reason: report the finding against the table, and rebuild on one layer when the claim is about that layer.
- **The layers need not sum to the whole** — a layering that splits the unit an edge is drawn from adds back up -- facets split passages, so the facet layers flatten into the unfiltered network. A stance layering splits mentions *inside* a passage, so a pair named once approvingly and once critically in the same passage is in no layer at all. Check the flattened graph against the unlayered one before calling the layers a decomposition.
- **A clique expansion inflates a large hyperedge** — a passage naming twenty entities is one hyperedge and 190 edges once expanded, so the biggest passages write most of the co-mention network. Read the hyperedge-size distribution first, and raise --min-weight before ranking anything off the expansion. The same arithmetic is what inflates a bipartite projection, because it is the same operation.
- **Interlayer coupling is a parameter, not a fact** — omega says how strongly a node in one layer is tied to itself in another, and nothing in a corpus states it. At 0 the layers are separate graphs sharing a matrix; as it grows every walk and every partition over the supra-adjacency is pulled toward treating the layers as one. Print the value beside any number read off the matrix, and re-run at another before believing the number.
- **A snapshot's edges are what was dated** — a window is answered from dated passages, so a document the attribution pass could not date is in no window of any sequence -- not the first, not the last, not the cumulative one. Every snapshot prints how many documents that is. A quiet window is as often an ungated dating pass as it is a quiet month.
- **The way you cut time is part of the finding** — single snapshots, disjoint windows, sliding windows and cumulative windows produce radically different histories from identical activation times. Say which was used, with the width and the step, in the same sentence as the trend. Consecutive sliding windows share edges, so their measurements are not independent and a smooth curve over them is partly the overlap. A --step wider than --window is a fifth cut the book does not describe: it throws the time between windows away, so the sequence samples the history rather than covering it and its totals do not add up to the corpus. The snapshot says so; quote that line if you use it.

### Node attributes (--where / --by)

- **An attribute is a hypothesis, not a finding** — somebody tagged these nodes because they expected the network to divide along the tag. Measuring it tests that expectation; it does not confirm it. Report the assortativity and its permutation null in the same breath as the value counts, and be as willing to write 'the network does not divide along this' as the opposite.
- **Near-zero assortativity with a strong Louvain result is a finding** — it says the network has structure and the attribute is not it. Name what the Louvain groups actually have in common before reaching for another tag; the division is real and you have not found it yet.
- **Report n per value** — a 90/10 split scores differently from a 50/50 one on every measure here, and a value with four nodes in it supports nothing at all. Print the per-value counts above the coefficients, and say how many nodes carry no value.
- **Untagged is not a third value** — every measure is computed over the nodes that carry a value, because a node nobody tagged has no label to correlate. A corpus tagged in half therefore describes that half. Check the untagged count before generalising, and never read an absent tag as a value of its own.
- **Two values are two builds, not two halves of one** — --where on one value and --where2 on the other builds two networks, each with its own n, density and communities, exactly as two time windows are two networks. Compare them the way the comparison report does -- counts first, structure second -- and never subtract one from the unfiltered whole to infer the other.
- **A majority label on a shared node is a majority** — a speaker carries the value somebody wrote for them, and so does an entity an extraction pass tagged, but an entity or a topic with no value of its own inherits one from the documents its passages sit in, which can disagree. The report says how many nodes were mixed; when that number is large the attribute is describing documents, and the entity network is the wrong place to ask about it.
- **A borrowed label cannot be correlated with the edges** — the one that looks like a discovery. Every network here draws its edges from co-membership in documents: two entities are joined *because* a passage names both. An entity with no attribute of its own is handed the value most of those same documents carry, so like joins like by construction -- and the permutation null makes it worse rather than better, since shuffling destroys the correlation and leaves the observed value alone. Confounded attributes therefore produce the largest z-scores in a corpus. The report prints where each label came from (`Recorded about the node itself` against `Borrowed from the node's documents`) and withholds the verdict when most were borrowed, so read that line before the coefficient. The fix is not a different measure: record the attribute on the entities themselves, in the extraction sidecars.

### Homophily (sna ego / --by)

- **An ego network's density is the ego's clustering** — most of what an ego network shows is the procedure, not the node: the ego is joined to every alter by construction, so §30.1's own warning is that 'all ego networks have a single connected component and will have a diameter of two'. Remove the ego and the density of what is left is exactly the ego's local clustering coefficient (§12.2), the pairs of its neighbours that know each other over the pairs it could have closed. That is the number to quote, and the pairs it leaves unjoined are the node's brokerage. Quoting the with-ego density instead reports the definition.
- **EI is a ratio of counts and a majority looks homophilous by size** — the EI index is (external - internal) / (external + internal), so **negative means homophily** -- the opposite sign to the assortativity printed above it -- and it knows nothing about how common each value is. A value holding 90% of the nodes has 89 in-group partners available for every 10 out-group ones, so it scores about -0.6 in a graph wired at random, which is §30.2's 'this approach breaks down ... if some values are more popular than others'. Read the z against the label shuffle, which holds the group sizes fixed, before reading the index. A *positive* EI is not a null result either: it is §30.4's **heterophily**, the love of the different, the disassortative case the chapter's summary puts beside homophily -- a dating network by gender, or a corpus where a value is only ever discussed against another.
- **Coleman compares to the share, EI does not** — Coleman's index takes the share of a value's ties that stayed inside it and subtracts the share its own size implies, so 0 means 'exactly as often as size predicts' and the 90% majority above scores 0 where EI scored -0.6. When the two disagree the report names the value as size-driven; quote Coleman per group and the assortativity for the network, and never quote EI alone as evidence that a group keeps to itself.
- **Weak ties are the bridges** — §30.3's weak tie is structural, not light: 'a weak tie is established between individuals whose social circles do not overlap much'. So the `## Weak ties` section measures neighbourhood overlap per edge and the edges of zero overlap are the bridges, whatever they weigh. Weight is a *proxy* for that and the chapter says outright that the correlation can go either way, so the section reports the rank correlation and refuses the Granovetter reading when it is flat or negative. On these networks a weight is shared documents, so 'strong' means 'written down together often' and nothing warmer.
- **A two-mode network has no overlap and no brokerage to read** — both §30.1's brokerage and §30.3's overlap are built out of *shared neighbours*, and on `--network speakers-entities` without `--project` two adjacent nodes are never of the same kind, so they share nothing by construction (§6.4). Every edge then has an overlap of 0 and is counted as a bridge, every node has a local clustering of 0 and a brokerage of 1.0, and a report that printed that as a finding would be saying 'every tie in this corpus is weak and every speaker is a perfect broker' about the data model rather than the corpus. `sna ego` and the `## Weak ties` section detect it and withhold the reading; project onto one mode first (ch. 26), remembering that a projection closes triangles by construction too. The same holds for any triangle-free network, declared two-mode or not.
- **Homophily and contagion look identical in a snapshot** — §30.4: 'if we observe a strong homophily, it could be because our social connections are influencing us into adopting behaviors we would not otherwise'. Nodes that were already alike coming together, and nodes that came together becoming alike, leave the same cross-section, and no null model separates them because both are true of the same graph. What separates them is a dated edge list with each node's attribute as it stood *before* the tie formed -- the chapter's own evidence follows thousands of people over 32 years. `--window` and `dynamic_edges` date the edges here; nothing dates the attributes, because a sidecar records what a node is now. So report association, name both mechanisms, and claim neither.

### Sampling (sna sample / analyze --sample)

- **A sampled number is a number about the sampler** — chapter 29 grades each method by what it distorts, not by whether it is good: BFS and snowball find the hubs first and explore a neighbourhood in full, which lifts both the mean degree and the clustering; a random walk lands on a node in proportion to its degree, so it oversamples hubs by construction; induced sampling picks nodes fairly and then shatters the connectivity, because two random nodes are almost never neighbours. Print which sampler produced the network in the same sentence as any measurement taken on it.
- **Report N as well as n** — a sample is 400 nodes of something. Report the population's node and edge counts next to the sample's, and the share; a reader who sees only n cannot tell a network from a fifth of one, and 'the density fell' is what sampling does to density on its own.
- **The correction goes with the measure, not the sample** — re-weighting a random-walk sample (§29.3) fixes the estimate of a distribution and leaves the sample as biased as it was: 'if what you need was the sample rather than the estimation of a simple measure, you're out of luck'. Metropolis-Hastings is the fix that acts on the walk itself, and even then the uniformity is a property of the walk's steps rather than of the set of nodes a short crawl has found.
- **A crawl cannot tell you the network is connected** — BFS, snowball, forest fire and both walks follow edges, so their samples are connected however fragmented the network is. Components, largest-component share and anything built on distances are measurements of the crawler when the sample came from one; take an induced sample if the question is about fragmentation.
- **Throughput is not speed, and a page is not a neighbourhood** — when the network came from somebody's API rather than from here, the crawl's cost was set by the degree distribution and not by the headline rate (§29.4). The book's own comparison: a policy returning 100 edges per page every two seconds is five times faster on paper than one returning ten every second, and crawls a broad-degree network in twice the time, because almost every node has ten edges or fewer and the small pages never wait. A high-throughput source can therefore hand you a smaller sample, and the k connections a page returns are chosen by a rule nobody outside that company can see -- so treat a paginated neighbour list as a sample of a node's neighbours, not as the node's neighbours.
- **Say what the sample did not see** — every sampled report carries the completion estimate of §29.5: how many edges were seen leaving the sample, how many more are estimated from the nodes the crawl never probed, and which nodes to probe next. It is an upper bound on completeness -- an edge between two unsampled nodes is invisible to any estimate made from the sample -- and a sampler that never learns a node's true degree, snowball above all, cannot produce one at all.

### Against random (every analyze report)

- **Clustered is a comparison, not a number** — a clustering coefficient of 0.57 is not high and not low until it is put beside something. Chapter 17's something is the random graph with the same n and m, whose clustering is exactly p, the density (§16.5) -- on the karate club that is 0.139, so 0.57 is four times the expectation. The report prints the ratio and, beside it, the same measurement on networks that keep every node's degree: a network whose clustering beats p but not its own degrees is clustered *because* a few nodes are busy, which is a different finding. Quote the ratio and the z, never the coefficient alone.
- **A short path length on n nodes is ln n** — distance shrinks as a network grows denser, so 'the average path is 2.4 hops' says nothing without n. §16.4's expectation is ln n / ln k̄ -- about 2.3 hops for 34 nodes at mean degree 4.6, and still only about 6 for a million nodes -- which is why short paths are the one real-world property random graphs already reproduce, and why finding them is not a finding. What is worth reporting is a network that is *longer* than that: geometry, hierarchy or a corpus split into unconnected pieces. On a disconnected network the number describes the largest component only, and the report says which it was and how big.
- **Broad is not power-law** — the degree comparison here is against the Poisson a random graph would produce: variance equal to the mean, so a variance-to-mean ratio of 1 and a maximum degree a random graph would rarely pass. A ratio well above 1 says the distribution is *broad* -- that hubs exist -- and that is all it says. Whether it is a power law, and with which exponent, is a fit that has to beat a lognormal and an exponential before it may be quoted (ch. 9); §17.3's alpha = 3 is a property of the preferential-attachment model, not a licence to call an observed network scale-free.
- **The degree rows have no degree-preserving null** — the rewiring null is built by swapping the observed edges, so every sample has the observed degree sequence exactly -- that is what it holds fixed (§18.1). The mean, the variance and the maximum degree therefore cannot be tested against it: they are its input. They are compared with the Poisson closed form and nothing else, and a report that claimed a z-score for them would be scoring a number against itself.
- **The three claims are three claims** — clustered, short and broad are independent, and the models of chapters 16-18 exist precisely because no simple one gets all three: a small-world graph is clustered and short and not broad, preferential attachment is broad and short and barely clustered, a random geometric graph is clustered with long paths. Report the three separately, and when a network cannot support one of them -- no edges, no connected pair, fewer than eight nodes -- say which one is missing rather than summarising the other two as a verdict on the network.

### Null models (--null / every z-score)

- **What the shuffle holds fixed is what it tests** — a null model is not 'randomness', it is a family of networks that keep some properties of yours and free the rest, and the answer changes with the choice. The same modularity is significant against a random graph and unremarkable against one with your degrees. So name the null in the same sentence as the z-score -- 'z = 3.4 against 200 degree-preserving rewirings' -- and never write 'significant' on its own.
- **A degree-preserving null is not a community null** — the edge swap keeps every node's degree and frees everything else, including the number of connected components and the clustering. Beating it means the degrees do not explain what you found; it does not mean nothing else does. A network in twelve pieces will beat it on modularity because the pieces are pieces, which is a fact about the corpus, not a community structure.
- **A projection's null is the two-mode network, not the projection** — every network here is a projection: one document naming twenty entities makes a 190-edge clique whatever else is true. Rewiring the projected edges invents graphs no corpus could have produced and then calls the projection's own arithmetic a discovery. `--null bipartite` rewires the memberships instead, keeping every node's number of documents and every document's size, and re-projects. Where the two nulls disagree, the difference is the projection.
- **A borrowed label needs the null that derives it again** — `--null bipartite` on the entity network re-derives every borrowed label from each rewired corpus, by the same majority rule the network was built with, so the copying that makes such a label circular happens in the null too and the observation has to beat the mechanism rather than beat zero. A label shuffle prices that mechanism at zero, which is why a confounded attribute scores its *largest* z-score there. Read both: a large shuffle z beside an ordinary bipartite z means the corpus produced the correlation, not the attribute.
- **That null tests the labels, not the tagging** — it asks whether a label distribution like this one would arise from rewired documents. It does not ask whether the tag is true, and it cannot: a borrowed label is still a value nobody recorded about the node, so the borrowed-label warning stays whatever the z-score says. Networks that keep no record of what their documents were tagged with -- the speakers-entities projection, anything read back from a file -- cannot re-derive at all, and the report says so instead of implying the stronger test was run.
- **How much shuffling is enough is not settled** — the chapter says so itself: the number of swaps to perform before stopping is a non-trivial quantity to evaluate, and nobody has fixed it. This package attempts one swap per edge, and the bipartite null five curveball trades per node, which are conventions with a citation and not answers. Too few and the samples are the observation with a dent in it, so the null sits next to the data and the z-score collapses towards zero. If a null distribution looks suspiciously like the observation, raise the swap count before believing either.
- **Quote the empirical p when the null is heavy-tailed** — the z-score counts standard deviations, which assumes the null histogram is the tidy bell the chapter's figures show. When the null distribution is skewed or fat-tailed -- and null distributions of projected quantities usually are -- the standard deviation is not a stable unit and the z-score overstates. The report flags that sample and prints an empirical p-value beside it, which cannot go below 1/(n+1): report both, with n.
- **An ERGM is fitted, not tested** — the coefficients say which configurations are more or less likely than chance *under that same model*, from a pseudo-likelihood that pretends the dyads are independent. The standard errors are therefore too small, models with a triangle term are prone to degeneracy -- all their probability mass on the empty or the complete graph -- and a fit is P(data | parameters) maximised, never a statement that the model is probable. Read directions and orders of magnitude only.

### Backboning (--backbone)

- **A hard weight threshold cannot be defended** — --min-weight is the Atlas's naive backbone (§27.1) and the chapter is largely an argument against it. Co-occurrence weights distribute broadly -- in the book's own projected network '82% of the edges have weight equal to one. The smallest possible hard threshold would remove 82% of the network, without allowing for any nuance' (p. 383) -- and such a distribution has no well-defined average, so no threshold may be motivated as 'x standard deviations from the average'. Weights are locally correlated as well, so one threshold flattens the dense corner and leaves the sparse one intact. Use --backbone, and say what the number you chose was for. The chapter's other naive escape -- keeping each node's n strongest edges, --backbone naive-top -- trades that objection for a worse one: it fixes the network's minimum degree at n, so the degree distribution afterwards describes the filter (p. 384).
- **The disparity filter manufactures hubs** — --backbone disparity keeps an edge when *either* endpoint finds it significant, so a hub's weakest link survives on the enthusiasm of the small node at the other end. The book states the consequence: it 'tends to create networks with high centralization, broad degree distributions, and weak communities' (p. 391). Never run Louvain or a centralisation measure over a DF backbone and call the result a finding about the corpus: that shape is the filter's. Prefer --backbone noise-corrected before any question about communities, since it requires both endpoints to agree and 'overweights peripheries and communities' (p. 393).
- **Noise-corrected needs counts, which is what these weights are** — its null is a binomial urn -- the network's total weight drawn against an expectation built from both endpoints' strengths -- and 'NC works only for discrete counts as edge weights, because the binomial is a discrete distribution' (p. 392). Every weight this package builds is a count of shared documents or shared passages, so it applies here; a network whose weights have been averaged, normalised or projected into fractions is refused rather than rounded.
- **A backbone is a new sampling frame** — the network after one describes the edges that survived a stated test, not the corpus. Report the method, the level, and how many edges and nodes survived, in the same breath as anything computed afterwards -- density, centrality and modularity all move, and two backbones of one network are two networks. `sna backbone` prints what every method keeps, side by side, so the choice can be made before the analysis rather than defended after it.
- **Backboning is not summarization** — it removes edges and keeps as many nodes as it can, because the point is to let the strong connections emerge while every node can still be described (pp. 381-382). A method that merges nodes into meta-nodes is chapter 46's job and answers a different question. If a backbone leaves a node with nothing, that is a finding about the node -- nothing it was joined to survived the test -- not a reason to lower the threshold until the picture looks fuller.
- **Above a node threshold, high-salience is sampled, not exact** — §27.3 states the cost -- one Dijkstra per node -- and offers no remedy for it; this package's own remedy, above SALIENCE_SAMPLE_ABOVE_NODES nodes, is a degree-stratified sample of sources rather than every one. Each edge's score is a stratified mean over that sample -- every drawn source weighted by its degree band's share of all the nodes, not by 1 / (sample size) -- because weighting every draw alike is only unbiased when the bands split the network exactly evenly, which an arbitrary sample size almost never does. Not in the book: say so before quoting a --backbone high-salience score on a network past that threshold, and read the sample size the report prints beside it, not the score alone. Below the threshold, or with every node given as the source explicitly, the number is exact and unchanged.

### Random walks (sna walks)

- **A stationary distribution is degree** — on an undirected network π is not a new ranking: §11.1 says it is 'quite literally the normalized degree of the nodes: the degree divided by the sum of all degrees', so reporting it after a degree ranking reports the same finding twice with a different scale on the axis. The version that is not degree is PageRank, whose teleport is what §11.1 calls the bells and whistles. On a directed network there is no closed form and two things make π not exist at all: a node with no out-edge, where the walk ends, and a network that is not strongly connected, where the answer depends on where the walker started.
- **Hitting time is asymmetric; commute time is not** — H[u][v] is the expected steps from u to v and it is not H[v][u]: §11.3's own example is one step from the end of a three-node chain to its middle and three steps back. A hub is hit fast and left slowly, because the formula depends on the degree of the destination and not of the origin, so 'these two entities are four steps apart' is not a sentence a hitting time supports. Quote both directions, or quote the commute time, which is their sum and symmetric by construction.
- **A commute time carries the size of the graph; a resistance does not** — C = 2|E|Ω (§11.4), so the same pair in a bigger network commutes more slowly without being any further apart, and two commute times from two networks -- two time windows, two personas, a sample and its population -- are not comparable. Effective resistance is the comparable one, and each component of a disconnected network is scaled by its own 2|E| because §11.1 reads it as a separate network.
- **Effective resistance only equals a hop count in a tree** — Ω counts every route in parallel, which is why §11.4 prefers it to a shortest path: the book's own simulation has one added edge cutting a shortest path by a factor of 7 and the resistance between the same pair by less than 3, so on noisy corpus data it measures structure where a shortest path measures the last edge somebody wrote down. Where there is only one route -- a tree, or the pendant end of any network -- the two coincide exactly and Ω tells you nothing new.
- **The minimum cut of a corpus network is a pendant node** — §11.5 is blunt: 'most of the times, the best way to solve the 2-cut problem is to put in one group a node with degree equal to one and put all other nodes of the network in the other group'. The exact global cut will therefore find an entity mentioned in a single passage, and the Fiedler vector's balanced split is an approximation that answers a different question. Print both when claiming either, and when the question is which groups exist, the section sends you to community discovery instead.
- **Consensus assumes one connected component** — §11.6: a network with separate components 'cannot reach a unique consensus; every connected component will reach its own consensus independently because there's no exchange of information across components' -- and a corpus network is usually disconnected, so the averaging has to be read per component. Two further traps: the discrete dynamics never settles on a bipartite network, where the opinions flip between the sides forever, and the two dynamics converge to different means -- the one driven by the stochastic matrix to the degree-weighted mean, the one driven by the Laplacian to the plain mean. Say which was run.

### Degree (sna degree)

- **A straight line in log-log space is not a power law** — chapter 9 is blunt about it: 'just because something looks like a straight line in a log-log plot, it doesn't mean it's a power law', and a linear regression on the logged values will hand you a beautiful R-squared and a tiny p-value for fitting a straight line to a straight-ish line. A lognormal produces the same picture. Never read a slope off a plot, and never quote an exponent that came from a regression.
- **Fit, test and compare before the word 'scale-free'** — the sequence §9.4 asks for is: the exponent by maximum likelihood with an xmin chosen by KS distance, a goodness-of-fit p-value from a parametric bootstrap, and likelihood ratios against the exponential and the lognormal -- the exponential first, because 'if you think a distribution might be an exponential, then it's definitely not a power law'. `sna degree` prints all four and withholds the word until they support it, at the bar the paper §9.4 sends you to sets -- a power law is ruled out at p <= 0.1, not at 0.05 -- so a fit that survives the weaker test and not that one reads as 'heavy-tailed, power law not rejected but evidence weak'. The lognormal usually cannot be excluded either, and then the book wants an argument from cumulative advantage rather than another number.
- **The bin size is a choice, and a choice can show you a pattern** — §9.2 offers log binning because equal-size bins lose the head of a degree distribution and keep its fat tail, and then warns about what it costs: 'an issue of power-binning is that it forces you to make a choice: to determine the bin size function. Having a choice is a double-edged sword: it opens you to the possibility of tricking yourself into seeing a pattern that is not there.' The binned histogram in the report is there to be looked at; the CCDF is what to read, and the fit never touches either -- it runs on the degrees themselves, because the CCDF of a power law is a power law with a different exponent.
- **The fit starts at xmin, so it describes the tail** — pure power laws are rare, and the two impurities §9.4 names are visible in the output: a shifted power law holds only above some kmin, which is what xmin is, and a truncated one holds only below the hubs. Report the tail's size next to the exponent -- 'alpha = 2.4 over the 180 nodes with degree 6 or more' -- because a fit over the top 2% of a network is a statement about 2% of the network, and the excluded head is most of it.
- **The mean of a heavy tail is not the typical node** — with alpha below 3 the distribution has a well-defined mean and an undefined variance (§9.3), so the average degree is arithmetic rather than a description: the book's stadium of 80,000 people has an average net worth of 2.5 million dollars once Jeff Bezos walks in. Around 70% of nodes sit below the mean degree in a broad network. Quote the median, the quartiles and the CCDF, and never motivate a threshold as 'x standard deviations from the average'.
- **A sampled degree distribution is the sampler's** — a random walk lands on a node in proportion to its degree, so the degree distribution of such a sample is the population's tilted by one factor of k and its exponent is off by one. `sna degree` prints the §29.3 re-weighted estimate beside the raw one whenever the network carries a sampling record, and says which is which; fit nothing to the raw curve of a crawl.
- **Which distribution it is matters less than that it is broad** — the chapter's own closing position: 'it doesn't matter too much if your network has an exponential, lognormal or power law degree distribution... the vast majority of networks have broad degree distributions, spanning multiple orders of magnitude. Most nodes have below-average degree and hubs lie many standard deviations above the average.' That is the finding worth writing down, and it is reportable from the quartiles and the CCDF without any fitting at all.

### Quantitative assortativity (--by <numeric>)

- **r is a Pearson over edges, and a heavy tail moves it** — chapter 31's coefficient is the Pearson correlation of two vectors built by walking the edges: one endpoint's value in x, the other's in y, each undirected edge contributing both orderings. So it is a linear question about squared deviations, and a handful of hubs -- or a handful of large attribute values -- carry most of those squares. The report prints Spearman beside it for exactly that reason, and the summary of the values with §3.1's heavy-tail caveat above both. When the two coefficients disagree, the relationship is monotone but not linear and Spearman is the one to quote.
- **A broad degree distribution is disassortative before anything happens to it** — hubs cannot connect to hubs if there are not enough hubs: p. 445 says a configuration-model rewiring of a network with a broad degree distribution comes out degree disassortative, and that preferential attachment and anything else imposing a power law does too. A negative r on a corpus network is therefore the expected state and not a finding. This is why the degree section's null is the configuration model rather than a shuffle, and why only the distance from that null is reportable. A network that is both broad and assortative is the interesting case: it has 'some non-trivial machinery driving their nodes' connections'.
- **The k_nn curve says more than r** — r is one number for a shape that is rarely one number: §31.1's second strategy plots each node's value against the mean of its neighbours' and aggregates the nodes that share a value, and real curves bend, flatten at the top, or rise and then fall. The report prints the curve and fits the power law the chapter asks for, whose exponent is positive for assortative, negative for disassortative and indistinguishable from zero for neither. Read the curve before the exponent: the fit weighs a value one node holds as much as a value a thousand hold, and an r-squared from a regression on logged points says a line fits those points and nothing more (§9.2).
- **Your friends have more friends than you, by construction** — §31.2 is not an observation about a particular network, it is arithmetic: the degree of the node at the end of a randomly chosen edge averages <k^2>/<k> = <k> + var(k)/<k>, which exceeds the mean degree for every network that is not regular, because 'a node with degree k appears in k other nodes' averages'. So the paradox holding is never news and the report states it as an identity. What is worth reading is the *share* of nodes it holds for -- 85% on the karate club -- and its size, since that is var(k)/<k> and therefore another reading of how broad the degree distribution is. The same goes for any attribute that correlates with degree, which gets its paradox 'for free' (p. 449); there the excess is cov(k, value)/<k> and can go either way, so its sign is a finding.
- **A borrowed number is still borrowed** — everything the categorical rule above says about a label borrowed from a node's documents holds for a number borrowed the same way, and for the same reason: the edges come from those documents, so near joins near by construction and the shuffle null subtracts none of it. `--by` on a numeric key prints the same provenance line and withholds the same verdict past the same share. A correlation is not a safer measure than a matching rate; it is the same claim in another currency.
- **mentions, documents and chunks are the corpus counting itself** — they are numbers on every node and `--by` will measure them, but none of them is a property of the thing the node stands for: they are counted from the very memberships the network was projected from, so a node recorded in many documents meets many others *because* of that. Assortativity in them is largely the projection describing itself -- read them beside the degree correlations, which say the same thing in the chapter's own currency, and never write them up as an attribute of the speaker or the entity.

### Robustness (sna robustness)

- **Robust to random failure and fragile to attack are the same fact** — §22.1 and §22.2 are one finding read twice. A heavy-tailed network survives accidents because 'when you pick a node at random and you make it fail, you're overwhelmingly more likely to pick one of the peripheral low degree ones' (p. 317) -- and it dies under attack for the same reason, because 'prioritizing them for your attack will have devastating effects' (p. 319). The two curves are therefore never quoted apart: a high area under the random curve on its own reads as 'resilient' when what it says is 'everything rests on a handful of nodes, and here is the list'. On a corpus network the sentence to write is about evidence, not about failure: nothing flows through these edges and no node can be attacked, so a steep targeted curve means the structure came from a few well-connected records.
- **Recompute the ranking after each removal, or say you did not** — the second node on a degree ranking is usually not the second-most-connected node once the first is gone, so an attack that follows the original order is a weaker attack than one that looks again. Both are legitimate -- an attacker with one reconnaissance pass and an attacker with live intelligence -- and they give different curves, so the report prints which was run and `--recompute` names it. In this package the recomputed ranking is refreshed once per removal *step*, which at `--steps 20` is every 5% of the network; `--steps <n>` makes it node by node, which is what §22.2 describes and what costs n centrality computations.
- **f_c comes from the second moment, so a heavy tail moves it** — the critical fraction printed beside the curves is f_c = 1 - 1/(kappa - 1) with kappa = <k^2>/<k>, which is not in chapter 22 -- the chapter states the phase transition and cites Cohen et al. 2000 for the form (footnote 6, p. 318). Two consequences. First, it carries the *second* moment: a few enormous hubs push kappa up without moving the mean degree, which is exactly §22.1's robustness, so f_c near 1 is a statement about the tail and not about the wiring. Second, it is the threshold for a *random* network with this degree distribution, so it belongs beside the simulated collapse rather than instead of it, and on a network with strong community structure or a heavy clustering the two part company. Do not confuse it with the book's 'Molloy-Reed approach' of §18.1 (p. 258), which is the configuration-model algorithm.
- **An interdependent collapse is abrupt: read the curve, not the endpoint** — §22.4's coupled system holds a respectable mutual giant component and then, over one step of the x axis, has none: 'These failures propagate in a chain reaction until we end up in a situation where practically every node in both layers is isolated' (p. 322-323). A mutual component quoted at one fraction is therefore a number from one side of a cliff and says nothing about where the cliff is -- which is the whole finding, since 'if you were to calculate the critical |R| value for each layer separately you would obtain a result much higher' (p. 323). Two more things the coupling is rather than measures: which nodes were matched to which (`same-id` pairs the nodes two of a persona's networks share by name, `random` is the chapter's model assumption, `degree` its perfect correlation) and how many nodes went unmatched -- every share in that section is of the coupled nodes alone, which is a narrower frame than the rest of the report.
- **A cascade needs a load, and a corpus network has none** — §22.3's model runs on two numbers per node the book takes as given, 'a current load and ... a total capacity' (p. 320). Neither exists in a corpus: nothing is carried between two entities that co-occur. `--load degree|betweenness` and `--tolerance` are therefore assumptions printed with the result, not measurements, and the honest reading of the tolerance sweep is comparative -- which nodes start an avalanche and which do not, on this structure, under a stated rule -- rather than a prediction that anything will fail. The chapter says as much about its own power grid: 'real failures in power lines are neither [linear nor local] ... You can use these methods in other scenarios, but only if you're sure that the assumptions made here are respected' (p. 315).

### Node roles (sna roles)

- **A role is relative to the partition it was computed on** — §15.1 defines roles by picking communities as the focus -- 'in this section we introduce the concept of node roles by picking network communities as our focus, just to give an example. If your focus is different, you will probably define different roles' -- so both coordinates, the within-module degree and the participation coefficient, are statements about the modules before they are statements about the network. Change the resolution, the seed or the community method and a connector hub becomes a provincial one without an edge moving. Print the partition and its stability next to the roles, and if the stability is low, quote the role counts rather than any node's role. Two consequences are the partition's and not the network's: with m modules the participation coefficient cannot exceed 1 - 1/m, so a two-community network has no connectors at all; and a module whose members all have the same internal degree has no spread for a z-score, so it can contain no hub.
- **Structural equivalence is about neighbours, not about being connected** — §15.2: 'For two nodes to be structurally equivalent they have to be connected to the same neighbors', and the chapter's own illustration of a perfectly equivalent pair (Figure 15.5(a), p. 229) has no edge between the two. On a co-occurrence network that is the useful direction: `sna roles` lists the entities the corpus discusses in the same company and never in the same passage. A high similarity is therefore not a prediction that an edge is missing -- that is chapter 23's question, asked with a null model -- and a pair joined by an edge is if anything *less* structurally equivalent, because each is then a neighbour the other lacks.
- **SimRank's decay is a choice; print it** — the book calls gamma 'a parameter you can tune' (p. 338) and never fixes a value, and it decides what the number means: the expected score is gamma^l for a walk of average length l, so a small decay confines similarity to short walks and a large one lets distant structure in. This package defaults to 0.8 and networkx to 0.9, and neither default is derived from anything, so a SimRank score quoted without its decay cannot be reproduced or compared. The same holds for the alpha of the regular-equivalence recursion, with one extra trap: §15.2's 'alpha < 1' is not enough for it to converge -- it needs alpha below 1/lambda1^2 -- so the report prints the alpha used, the limit, and whether the iteration settled.
- **A blockmodel position is a cluster of a similarity matrix** — the chapter reads three equivalence classes off its Figure 15.8 by eye (p. 231) and names no algorithm for finding them, so the positions `sna roles --k` prints are a clustering, with everything that implies: a k has to be chosen, at k and k+1 the blocks can be entirely different, and the instability of the similarity matrix is inherited whole. The image matrix -- the density of edges between two positions -- is what to quote, because it is the claim the chapter actually makes about the classes. And the book's warning on the recursion applies to the positions it produces: 'you'd get that CEOs are similar to interns because they both connect to middle managers' (p. 232), so in anything hierarchical, everything an even number of hops apart collapses into one position.
- **A `core` node here is not a core of the core-periphery structure** — §15.1 stops to say so itself when it names its four roles: 'brokers, gatekeepers, core, and periphery (the latter two not to be confused with the core-periphery mesoscale structure that we will see in Chapter 32)' (p. 226). The two use the same two words for different objects. A `core` in `sna roles` is a node whose ties are concentrated inside *one module of a partition* and which is a hub of that module, so a network with six communities has six cores and they are cores of nothing but their own circle; chapter 32's core is a single set of nodes densely joined to each other and to everyone, and it is defined without any partition at all. They can disagree completely -- a provincial hub of a peripheral community is a `core` here and sits in chapter 32's periphery -- so name the command next to the word, and when the question is which nodes hold the network together rather than which hold a community together, the chapter to reach for is 32 and not this one.

### Hierarchies (--network relations / sna hierarchy)

- **A hierarchy is a claim about direction, not about degree** — chapter 33 measures *flow* hierarchy: whether the arrows all run one way, from the levels above to the levels below. It is not about who has the most edges. §33.1 sends the other two senses of the word elsewhere -- an *order* hierarchy 'is nothing more than a different point of view of node centrality' (ch. 14, the Centrality section), a *nested* hierarchy 'is equivalent to performing hierarchical community discovery' (ch. 37, not built here) -- so a high-degree entity is not the head of anything, and the section refuses an undirected network rather than flattening one: with no directions GRC is 0, every edge lies on a two-cycle and agony is the edge count, which are four numbers that look like an answer.
- **GRC is one number about one root** — §33.3's global reach centrality averages the gap between the furthest-seeing node and everybody else, so it answers 'is there an overseer that sees all and knows all', and only that. It scores 1 only for a star -- the book's own complaint, 'even flawless hierarchies might fail to get a 100% score', with a depth-3 tree scoring 0.898 -- and it scores 0 both for a single directed cycle, where everyone reaches everyone, and for a network with no edges, where nobody reaches anybody. Read it beside the cycle share and the arborescence score, never alone, and check how many nodes tie for the maximum before calling any of them the head.
- **An arborescence keeps one boss per node and forgets the rest** — §33.4 cuts the network down until every node has exactly one incoming arc, which means the tree in the report is a *reading* of the network and not a correction of it: every dropped arc is a relation the corpus stated, and the section prints the heaviest of them and the share of the weight that survived for that reason. The score is how much had to go -- 'the more edges we needed to remove to obtain an arborescence, the less the original network was resembling a perfect hierarchy' -- so it is a property of the whole network, while the root is a property of one tie-break. When no single tree spans the network the report says so and falls back to the arborescence *forest* the chapter itself ends up with. Its null model is weaker than it looks: a degree-preserving rewiring holds every in-degree exactly and the tree keeps one arc per node that has any boss at all, so on a network where nobody has two bosses the score cannot move at all and a z of 0 means the null had no room, not that the shape is ordinary.
- **Agony counts the arrows that point up** — §33.5 puts every node on a level and charges `max(l_u - l_v + 1, 0)` per arc: free downward, 1 between two nodes on the same level, `k + 1` for one climbing `k` levels. So it grades violations where the cycle share only counts them -- one back-edge across three levels costs more than one across a single level (Figure 33.8) -- and it is 0 for every DAG, which is the leniency it inherits. Two things the total does not license: a node's level is not a finding, because many rankings reach the same minimum, and a total is comparable across two networks only through the per-arc rate or the null model, because agony grows with the number of arcs.
- **A cycle is not a mistake in the data** — §33.2 reads a cycle as nonsense in an organisation -- 'your boss gives you an order, you pass it down to one of your underlings and, somehow, they give it back to your boss' -- and that reading belongs to organisations, not to corpora. Here an arc is a relation an extraction pass stated between two entities, so two entities each described in terms of the other is a two-cycle and is exactly what a corpus does; a knot of mutually-defining entities is a finding about the subject. Nothing in this section removes a cycle from the network: the flow hierarchy counts the arcs inside the knots, the arborescence and the drawing step round them, and the graph keeps every arc.

### High-order (sna highorder)

- **A hypergraph is not a simplicial complex: closure is a claim** — §7.3 draws the difference: a simplex 'logically also contains all lower-level simplices', while 'a hyperedge with four nodes only contains those four nodes' (p. 112). Closing a passage hypergraph downward asserts that every sub-group of a passage's entities is itself a relation, which is a modelling choice and not something the corpus wrote down. The reverse direction is worse, and §34.1 says so: simplices to cliques and cliques to simplices 'are not commutative' (Figure 34.10, p. 481), so promoting every triangle of the entity network to a 2-simplex invents co-mentions nobody made. `sna highorder` prints the closure -- how many triangles of the 1-skeleton the passages actually filled -- for exactly that reason: read it before reading any higher-order number, because it is the size of the gap between the network and the complex.
- **A memory network's node is a transition, not an entity** — in the second-order network of §34.2 a node is `A -> B`, meaning 'in B, having arrived from A', and an edge is an observed path `A -> B -> C`. Nothing in that network ranks entities, and a centrality computed on it ranks *transitions*: the degree of `A -> B` is how many ways the corpus continued from it, which is a sentence about an ordered pair. Every number has to be folded back before it is quoted about a thing -- `sna highorder` folds the walk by summing the transitions that end in each entity -- and the fold is a post-processing step the book is explicit about: 'you need to reconstruct the original structure if you want to properly interpret your results' (p. 481).
- **A first-order walk forgets where it came from; report both** — §34.3's finding is a *comparison*, never a single ranking: the walker with memory against the walker without, over the same entities. Quote one alone and there is nothing to see, because neither is more true than the other -- one answers 'where does a walker go that only knows where it stands', the other 'where does it go if it also knows where it came from'. The difference between them has three sources, and attributing it to one on sight is the mistake to avoid. The first is real second-order structure *inside* a document: the two agree only where the memory network's own flow is balanced -- every transition continued as often as it is entered -- and a document that ends on the entity it began with does not give that, because its first transition is reached by nothing and its last continues into nothing. (Balancing the entities balances only the first-order chain: [A][B][A][B][A][C][B][A] closes on A and still gives 3/7, 3/7, 1/7 without memory against 2/5, 2/5, 1/5 with it, undamped.) The second is the document boundaries themselves, with fragmentation on top -- 'HON networks tend to transform into weakly connected graphs, or even not connected' (p. 483). The third is the teleport, which injects its mass over transitions in one walk and over entities in the other; the damping factor is a parameter somebody chose, so print it and re-run at another before believing a rank change.
- **Consecutive passages are an ordering the corpus imposes, not a causal one** — the sequence a memory network is built from is 'passage i, then passage i+1 of the same document', which is a transcript's running order or a thread's posting order. It is not evidence that naming A led to naming B, and it is not a time series: the chunker decided where passages begin, so a long answer split in three becomes three steps and a short one becomes none. Three consequences worth stating with any number read off it. A document boundary is a hard stop -- the last passage of one document and the first of the next are not adjacent, and treating the corpus as one long sequence would manufacture transitions between unrelated documents. A passage that named nothing, or that a `--facet`/`--stance`/`--where` filter removed, breaks the chain rather than being bridged, so a filtered run has fewer transitions than its passage count suggests and the report prints how many breaks there were. And an entity named again in the next passage is a continuation, not a step, so those are dropped and counted rather than turned into self-loops.

### Motifs and mining (sna motifs)

- **A motif is a count against a degree-preserving null, never a count** — §41.2 gives the procedure in four steps and only the first one is counting: 'you define a null model of your network, keeping its relevant properties fixed -- maybe just the degree distribution [...] Finally, you compare with your observation, so that you can build an idea of the statistical significance of the motif' (p. 590). So '2,363 four-node subgraphs, 11 of them cliques' is a description, and only the z-score beside it is a finding. The null here is the configuration model of §19.1, which holds every degree fixed and nothing else: a shape that beats it has beaten the degree sequence, not chance. And on an undirected network the open triad and the triangle are one finding -- the degree sequence fixes their sum, so the two z-scores are the same number with opposite signs.
- **The triad census of a projection is the projection's** — every network here but `relations` joins two nodes because a document or a passage held both, so a passage naming three things is a triangle by construction and a passage naming ten is 120 of them. A high triangle count is then a fact about how the corpus was written down -- long passages, many entities each -- before it is one about the subject, and the same caveat §26.1 makes about projected weights applies to projected shapes. Read the hypergraph (`sna layers`) beside it, where that trio is one hyperedge rather than three edges, and treat `relations`, the one network whose edges were asserted rather than projected, as the one where a feed-forward loop means something.
- **Support counts documents, not occurrences** — §41.4's definition is 'we have a database of many different graphs and we ask in how many graphs the motif appears' (p. 598), so a triangle with support 5 was found in five documents and may have occurred fifty times in them. That is deliberate: counting occurrences would break the rule that makes the search possible, since 'a larger graph can at most be as frequent as the least frequent of its subgraphs'. The consequence for reading is that support says how widespread a shape is and nothing about how dense it is where it appears, and that the support threshold is an analyst's choice the book leaves open (p. 596) rather than a significance level.
- **Frequent in one graph needs a support that does not double-count** — on a single network the transactional count is useless -- 'that number is always going to be either zero [...] or one' (p. 601) -- and counting occurrences instead breaks anti-monotonicity, which §41.5 calls unacceptable because it would make a larger motif look more frequent than the smaller one inside it. `sna motifs --mine` therefore reports the minimum image support: for each node of the pattern, how many different network nodes ever play that role, and then the smallest of those counts (p. 603). It is smaller than what you can count by eye and it is meant to be -- a star with four leaves supports a two-step path exactly once, because only the centre can ever be the middle -- so never read it as 'how many times this shape appears'.

### Paths and components (every analyze report)

- **A diameter is one pair** — §13.3 defines it as the longest shortest path, 'the worst case for reachability in the network'. It is the distance between the two nodes furthest apart and nothing else, so one peripheral entity nobody else mentions sets it, and adding one edge can cut it in half. Quote the average path length and the histogram beside it: the average is the typical case, the diameter is the extreme, and only the histogram says whether the network is tight or spread out.
- **An average path length is a number about one component** — §13.3's convention is that nodes in different components are at infinite distance, so 'a network with more than one connected component has an infinite diameter' and what you look at instead is 'the diameter of the giant connected component'. Every distance in the report is therefore measured on the largest component, and the section prints how many nodes that leaves out. On the directed network there is a second exclusion, because a weak component still holds ordered pairs no directed path joins: those are counted and excluded, never averaged in as a number.
- **An estimated diameter is a lower bound** — above a thousand nodes the report stops solving all pairs and measures from a seeded sample of sources, which it says beside the numbers with the count of sources. The mean and the histogram are then estimates and behave; the diameter can only go up when more sources are added and the radius can only go down, so read them as bounds and never compare an estimated diameter with an exact one.
- **Reciprocity without the dyad census is half a number** — §10.3's reciprocity is reciprocated pairs over connected pairs, so it says nothing about how many pairs there were. 40% over five connected pairs -- the book's own example -- and 40% over five thousand are the same ratio and not the same finding. The report prints the mutual, asymmetric and null counts next to it; quote all three, and remember the census ignores self-loops because a self-loop is not a pair.
- **A walk is not a path** — §10.1's A^k counts walks, which may cross the same edge as often as they like, and §10.2's cycle is a path, which may not repeat a node at all. So the number of walks between two entities grows without limit while the number of paths does not, the closed walk of length two that gives a node its degree is not a cycle, and the cycle-space dimension |E| - |V| + c counts *independent* cycles rather than cycles -- a network has exponentially many of those. Say which of the three a number counts, because the three are never close.

### Ranking (every centrality table)

- **A centralization is one number about the shape, not about a node** — §14.8's ratio -- the summed distance of every node from the most central one, over the same sum on a star of the same size -- says how concentrated a network is, and nothing whatever about who is at the top of it. 1.0 is a star and 0.0 is a network whose nodes all score the same. It also depends on which centrality it was computed from, and the chapter's own figure has degree and betweenness disagreeing about which of two networks is the more centralized, so the report prints the whole table and a claim has to name the row it came from. The star has no weights, so both sums are taken on the binary view of the network: with weights the ratio would move when the weights were rescaled and the shape was not. Some rows read `undefined`, because §14.8's maximum is only *usually* a star: coreness and undirected reach give every node of a star the same score, and on a network in more than one piece the eigenvector concentrates on one component, which can push the observed sum above the star's. The bullet under the table says which case it is.
- **A rank without its stability is a coin flip at the tail** — chapter 14 hands back an ordering and says nothing about its error bars, and the tail of a corpus ranking is decided by one or two passages. The `## Ranking stability` section resamples the network -- by default chapter 29's edge sampling, which asks what would still be printed if the corpus had recorded a fifth less -- and reports the rank correlation of the whole ordering plus how often each printed name stayed in the top k. Quote the names that held; say 'also in the top twenty' for the rest, and never compare rank 7 with rank 9. Two of the resamplings answer different questions and the section names which it ran: a degree-preserving rewiring holds the degree ranking fixed by construction, so it asks whether a rank is anything beyond the degree, and a backbone asks whether it rests on the edges that survive a null for edge weight.
- **Harmonic is closeness that survives disconnection** — closeness, betweenness, reach and the random-walk measures are all 'ill defined when your network has pairs of unreachable nodes' (§14.6), which every filtered corpus network has: you cannot compare two nodes' closeness across components, because one of them may score high only for sitting in a small component. Harmonic centrality sums 1/distance with 1/infinity = 0, so it is defined on the whole network at once and is the ranking to quote when `components` is above 1. It is not on closeness's scale -- it is a sum, not an average, so it grows with n and two networks of different sizes cannot be compared by it.
- **HITS is two rankings and needs direction** — §14.5 assigns every node two scores, not one: a hub 'maybe does not know many things, but knows the people who know them', an authority 'is pointed by everyone when someone asks about that particular topic'. They are the leading eigenvectors of A A^T and A^T A, which on an undirected network are the same matrix -- so hubs and authorities collapse into one vector, the eigenvector centrality, and `sna` refuses to print it twice. Only `--network relations` has the direction to ask. A node can be both at once, and the two columns must never be summed into a rank.
- **Coreness is a floor, not a rank** — the k-core decomposition (§14.7) peels away everyone with fewer than k connections and reports the depth at which a node fell, so it is a property of a *set*: the karate club's 34 nodes hold four distinct values, ten of them tied in the 4-core. Read it as 'in the dense middle' or 'on the fringe', never as a position, and read the shell sizes the report prints beside it before quoting a value at all. On a co-occurrence network it is inflated by long documents, because a passage naming m entities makes an m-clique and every one of them an (m-1)-core member; the k-truss, which asks each edge for triangles instead of each node for neighbours, is the stricter reading of the same idea.
- **Reach is a directed question** — 'reach centrality is only defined for directed networks' (§14.3): if the network has a single connected component then every node reaches everything and the ranking is a column of ones. So the report bounds it at two hops, which makes what it prints a statement about neighbourhood size rather than the chapter's 'how much you can command' -- **on the directed network too**, where the bound is the same two hops and not the unbounded BFS. `reach(graph, hops=None)` is the chapter's own measure, and it is what a claim about command has to be made from. What direction changes is that this is the only ranking in a `--network relations` report that reads *out* of a node: a high reach beside a low closeness is an entity that names everything and is named by nothing.

### Link prediction (sna predict-eval)

- **A random holdout changes the network it tests** — deleting 10% of the edges and asking a predictor to find them again (§25.1) takes those edges out of the evidence as well as out of the answer key, so a common-neighbour score is being asked for a link whose triangle was removed with it. Here an edge *is* a passage, so the experiment deletes the sentence and then asks for it back, and the score that comes out is a score on a network that never existed. Use it to compare two predictors on equal terms, never to state how well one will do on the corpus's future.
- **The temporal split is the honest one** — when the documents are dated, `--temporal <date>` trains on what the corpus had joined before that date and tests on what it joined after, which is the split §25.1 names first and the only one that leaves the training structure intact. Two things it demands in return: a pair joined before *and* after is not a positive -- the book throws that prediction away because it was already in the data -- and a later pair whose endpoint had never appeared is dropped rather than counted as a miss, because scoring it would measure the corpus's growth and not the predictor. Both counts are printed. Undated documents are on neither side, so a half-dated corpus gives a split of its dated half and nothing more.
- **AUC is flattered by imbalance: report precision@k** — the test set is balanced -- half real links, half sampled non-links -- because §25.1 says to build it that way, and the pair space it was cut from is not: the report prints how many non-edges there are per positive, and it is in the hundreds or thousands. On the full space the easy non-links inflate the AUC, which is why the book warns that an always-negative predictor scores 99.999%. Quote precision@k and the average precision against the random-P column in the same table, never an AUC alone; and an AUC of 1.0 is not a triumph but a bug report -- §25.2 says you will never see one and that seeing one means something leaked.
- **Beat preferential attachment before claiming a predictor** — every evaluation prints two baselines: a scorer that ignores the network (AUC 0.5, the 45-degree line) and the degree product, which needs nothing but the degree sequence. A method that does not beat both has been shown to have learned nothing about structure, only about how busy the endpoints are. Preferential attachment is hard to beat on a corpus network for a reason that is about the corpus and not about the future: the nodes everyone talks about are named in many passages, so they acquire co-mentions with everything.
- **A prediction is a hypothesis, not an edge** — nothing in a prediction report is written to the graph, and a scored pair is a claim about what the corpus would say if somebody looked, not a claim about the world. The two nodes a co-mention network joins were named in one passage; a predicted pair is a passage nobody has written, so the only honest next step is to go and read for it.

### Link prediction, simple graphs (sna predict)

- **Every score here predicts an old-old link** — p. 334-335: "every method we saw so far... predict[s] exclusively 'old-old' links" -- a score is defined only for two nodes already in the network a predictor reads, never for a node that has not appeared yet. GERM is the one method in the chapter that can also propose an old-new or a new-new link, by adding a node the antecedent pattern did not have; `sna predict --method rules` does not do this, because a document graph already names every entity the document mentions -- there is no unseen node for a document-level rule to add, so the distinction the book draws does not arise on this corpus.
- **Katz's beta must stay under 1/lambda_max** — §23.7's score sums beta^l over every walk length, and the sum only converges below the largest eigenvalue's reciprocal -- the same bound Katz centrality carries (§14.4). `katz()` refuses a beta at or above it rather than return a number that would have kept growing. The book gives no rule of thumb for how far under the bound to sit: a beta close to it lets long walks contribute a lot, which on a co-mention network means two entities connected only through a chain of unrelated conversations can outscore a real near-miss; a beta close to 0 makes Katz read like common neighbours with extra arithmetic.
- **HRG scores are a Monte Carlo estimate, refit every call** — §23.5's score reads a dendrogram :func:`graphrag.sna.cluster.hrg_fit` finds by a randomised MCMC search over dendrogram space, the most expensive step in this chapter by a wide margin. `hrg_scores()` refits on every call unless a fit is passed in, so two runs at different seeds can rank the same pair differently, and a caller scoring many pairs from the same graph should fit once and reuse it rather than let a report or a loop refit silently.
- **An association rule's confidence needs a null: the network's own transitivity** — `sna predict --method rules` reports one GERM rule (§23.6), open triad closes into triangle, with a confidence computed from how many documents hold each pattern (§41.4). That confidence is not automatically evidence of anything: a network that already closes triangles at a high background rate will show a high confidence for this rule with no signal in it at all, which is exactly what the global clustering coefficient (transitivity, §12.2) measures. Read the confidence against it, not against 0 or 1 -- `AssociationRule.beats_transitivity` is that comparison, and the report prints both numbers rather than the confidence alone.

### Graph partitions (--method sbm|infomap|walktrap|label-propagation / sna community)

- **Random-walk and label methods are not deterministic** — §35.2, p. 498 and §35.3, p. 500: Infomap, Walktrap and label propagation are all built on a random process -- which node a walker visits next, which tied label a node adopts -- so running the same method twice on the same graph can return different partitions. `infomap_communities` and `label_propagation` both run several seeds and report the mean pairwise adjusted Rand index across them as `stability`, read exactly as Louvain's own stability is: below about 0.6 the boundaries are the seed, not the network.
- **A plain blockmodel fit can mistake a hub for its own community** — §35.1, p. 495: a plain stochastic blockmodel has one connection probability per block pair and no other way to explain a node whose degree looks nothing like its neighbours' -- it can only fit that by giving the node a block of its own, regardless of who it is actually connected to. The degree-corrected fit (`sbm_communities(..., degree_corrected=True)`, the default) gives every node its own expected degree first, so a hub's degree is no longer evidence about which block it belongs in. Prefer the plain fit only when the question is specifically about connection probability blind to degree; prefer degree-corrected otherwise.
- **A local community's own stopping rule is not 'grow until nothing is left'** — §35.5, p. 505: Clauset's local modularity R returns to 1.0 the moment a community becomes an entire connected component -- every edge then has both ends inside it -- so a rule that grew until R peaked over the *whole* exploration would always stop at the whole component and never at a real boundary. `local_community` instead stops the first time growing would make R go down, which is the boundary the chapter's own picture (Figure 35.13) draws; a `max_size` limit stops it earlier still, for a network too large to explore in full.
- **A temporal match is a majority vote, not a certainty** — §35.4, p. 503: "don't assume that you're going to be able to say, unequivocally, something like 'community C split in C1 and C2 at time t'." `match_communities` labels every pair of consecutive communities by their own Jaccard overlap against `threshold`, so a community that genuinely split while also bleeding a few members into an unrelated neighbour is reported as a clean split -- the messier partial merge that real evolving networks produce is exactly what a fixed threshold on a pairwise score cannot represent, and the chapter is explicit that no method reading independent per-window partitions this way can either.

### Signed and multilayer prediction (sna predict --signed / --layers)

- **A sign prediction and a link prediction are different questions** — §24.1 asks 'given that these two are already named together approvingly or disapprovingly, which sign does that connection carry', never 'will they connect'. `sna predict --signed` therefore evaluates through its own held-out-sign accuracy, not through chapter 25's AUC: a positive there is an edge deleted from the training graph, and a signed edge is never deleted, only its label hidden. Reading a sign accuracy against chapter 23's precision@k, or the reverse, compares two different experiments.
- **Balance theory abstains; that is not a neutral reading** — a pair with no common neighbour carrying a sign on both legs gets no vote at all (`balance_score` returns 0.0, which `predicted_sign` reads as 'no prediction', not 'predicted neutral'). Report coverage beside accuracy: a predictor that abstains on every hard pair and is right on every easy one is a different claim from one that is right on the same share of everything hidden.
- **Status theory needs a direction this corpus does not have** — p. 347's social status theory reads a positive edge as the *originator* granting status to the *receiver*, which only makes sense on a directed edge -- a vote, an endorsement. A co-mention has no sender and no receiver: two entities are named together in a passage, symmetrically, so there is nothing for status theory's 16 directed triangle templates (Figure 24.4) to read. Balance theory is the only one of the chapter's two theories built here for exactly that reason.
- **The all-negative triangle reads as unbalanced here** — p. 345-346 notes that an all-negative triangle (Figure 24.1(c)) is sometimes read as balanced or neutral in practice (its own Tuscan campanilismo example), against the classical rule this module implements, where only an even count of negative edges balances. `classify_triangle` reports the classical reading; nothing here detects the contextual exception, since telling the two apart needs knowing what the negative relationships mean, which the signed graph does not carry.
- **The cross-layer weighting is an inter-layer correlation, not §9.1's layer relevance** — §24.2's 'Multilayer Scores' subsection (p. 351) says a neighbourhood score 'should be re-weighted using layer relevance', pointing back at §9.1's own definition, |N(u, l)| / |N(u)| -- the share of *one node's* neighbours reached through layer l. That is a per-node number; `cross_layer_common_neighbours` scores a *pair*, with two nodes to pick from, and the book does not say how. It uses `layer_correlation` instead -- the Pearson correlation between two whole layers' edge vectors that Atlas §24.4's own third exercise asks the reader to compute -- as this module's own stand-in for the re-weighting idea, not an implementation of §9.1's layer relevance. A negative correlation contributes nothing rather than a penalty, and only common neighbours is generalised this way -- the chapter's other §24.2 methods (hitting time over meta-paths, tensor factorization, multilayer GERM, embeddings) are not.

### Community evaluation (every grouping report)

- **Modularity has a resolution limit at sqrt(2m)** — §36.1: "modularity has a preferred community size, relative to the size of the graph… it accepts partitions only of a comparable size with the size of the network", and the number that decides it is sqrt(2|E|) internal edges. Below that, merging two clearly distinct communities *raises* modularity, so a small community in a large network is the algorithm's resolution and not a finding -- the book's own ring of cliques scores 0.904 for merging neighbouring cliques against 0.902 for the partition every human draws. The evaluation section lists every community under the line. Their boundaries are the ones not to name.
- **Conductance and coverage pull in opposite directions** — §36.2 is a battery because no single function captures the classical definition of a community: conductance only asks whether the community is externally sparse, internal density only whether it is internally dense, and "each of the two measures only satisfies one of the two requirements". Their biases run opposite ways with community size -- coverage and conductance both improve as communities grow, up to the whole network as one community, while performance and internal density both improve as they shrink, down to one clique each. So a partition that improves one measure has usually only changed size. The section prints which direction size pushes every measure next to its value; quote a measure with that sentence or not at all.
- **NMI grows with the number of communities** — §36.4: mutual information "is always non-zero… there will be always a little mutual information between two vectors, even if they are both completely random", and normalising it does not fix that -- the book's own pair of independently drawn ten-element vectors scores NMI 0.09. It also rises as either labelling gets more groups, so an NMI of 0.4 over thirty communities can be less agreement than 0.2 over three. The adjusted pair (AMI, ARI) subtracts what chance would give and is 0 for random labellings and negative for worse than random; read those first and use the NMI only against another NMI over the same number of groups.
- **A partition is a link predictor, and that is the honest test** — §36.3: communities are a claim about where the next edge will appear, so hold edges out, score every pair by whether it sits inside a community, and read the AUC -- "the higher your AUC… the better your partition is". It is the one test in the chapter that does not need a ground truth and does not reward merging, which is what makes it worth more than the topological measures above it. Two limits are printed with it: this is one seeded split rather than chapter 25's k folds, and the partition was found on the whole network including the held-out edges, so the AUC is optimistic and the number that makes it readable is the baseline beside it -- the same community sizes with the labels dealt out at random.
- **A negative modularity is a bad partition, not a bug** — §36.1 gives modularity a domain of -0.5 to +1. Zero is what you get for putting every node in one community, and a negative value means the groups hold *fewer* edges than a configuration model with the same degrees would have put in them. The book calls that "a reasonable scenario" -- it is what a disassortative partition looks like (§30.2) -- so report it as the reading it is. On a corpus network it usually means the partition came from somewhere other than the edges: a clustering of text embeddings, or an attribute handed to the graph.

### Hierarchical communities (louvain_levels / girvan_newman_dendrogram / hrg_fit)

- **Density and modularity are not the same profile** — §37.4: a dendrogram level with high modularity need not be the level with high internal density, and the reverse. Putting every node in one community has some density but zero modularity by construction (Figure 37.11(a)); splitting a network with no real community structure -- a complete graph, say -- keeps every possible edge inside each half (density 1) while modularity stays near zero, because nothing about the split beats what the null model already expects (Figure 37.11(d)). Only at the level a dendrogram is actually cut at do the two usually agree (Figure 37.11(c)), and that is a property of the cut, not a guarantee about every level printed beside it -- `louvain_levels` and `girvan_newman_dendrogram` report both numbers for every level for exactly this reason.
- **More than one dendrogram can fit a network equally well** — §37.2, p. 538: "the dendrogram... is not the only possible good description... the resulting dendrogram would have been equally likely." An HRG fit is one sample from a likelihood surface with real ties in it, not the unique correct hierarchy, so a single node's exact position at the finest level of `hrg_communities` can differ between equally-good fits while the coarser, better-separated levels stay put. Read the levels that are far apart in density from their neighbours; do not read which of two similarly-placed leaves joined the tree one step before the other.

### Overlapping coverage (sna overlap)

- **The overlap paradox: density and sparsity cannot both survive overlap** — §38.7: the classical definition -- dense inside, sparse outside -- cannot hold for both a community and its overlap with another at once. If the shared nodes barely connect to each other, that contradicts internal density; a stochastic blockmodel reads it as a hole where it expects an edge. If they connect to each other *more* than average -- and the book's own argument is that they should, because they carry two communities' worth of ties rather than one -- then "the overlap... is denser than the community itself", which contradicts external sparsity: the boundary outranks the core. `sna overlap` checks both directions for every pair of communities that share at least two nodes and reports which, if either, fired; neither is a defect to explain away.
- **An overlapping cover is not a fuzzy one: every membership here is equal** — §38.1 (Figure 38.3) draws a line this package stays on one side of: an overlapping community says a node fully belongs to every community it is in, with no weight attached, where fuzzy clustering would say 60% here and 40% there. Nothing in `sna overlap` reports a belonging coefficient, and a size or a count next to a community is not one either -- it is how large the community is, not how strongly any one member sits in it. Reading strength into the cover is reading in a number the method was never asked to produce.
- **Overlapping NMI inherits NMI's chance-inflation, worse** — §38.1: overlapping NMI variants "share with their original counterpart the issue of non-zero values for independent vectors" -- and matching each community to its *best* partner on the other side, rather than comparing two fixed labellings, adds a second source of chance agreement the chapter's own citation names (Gates and Ahn, 2017). A value is only informative next to a self-comparison (which is exactly 1.0) and a reshuffled baseline of the same sizes, the way ordinary NMI needs the number of communities beside it (§36.4); never read one number in isolation as "the two covers agree".
- **k-clique percolation cannot classify a low-degree node, on principle** — §38.3, p. 550: "if you set your k relatively low, e.g. k=4, all nodes with degree equal to one cannot be part of any community" -- more generally, degree under k-1 rules a node out of every k-clique there is, whatever the rest of the network looks like. `sna overlap --method clique` names those nodes rather than letting their absence from every community read as a finding about them; a node with enough degree that still joined nothing is a different, weaker claim and is named separately.

### Multilayer community discovery (sna multilayer-communities)

- **Three methods, three different assumptions -- read the agreement before the winner** — §40.1's flattening assumes 'every edge type is equally important'; §40.2's layer-by-layer matching and any large interlayer coupling in §40.3 assume the layers are correlated enough to combine at all -- 'disassortative layers exist and might represent a problem' (p. 575). Neither assumption is a fact about a corpus, so `sna multilayer-communities` scores all three through the same battery and adds a fourth number the chapter itself does not: the adjusted Rand index between every pair of layers' own communities. A partition winning the battery while the layers disagree with each other has not found the corpus's structure, it has found whichever layer the method leaned on hardest.
- **Interlayer coupling makes pillar communities, not more evidence** — §40.3, Figure 40.5: a strong coupling Cvsr produces 'pillar communities where nodes tend to favor grouping with themselves across layers' regardless of what they connect to there, and a weak one produces 'flat communities' that ignore the coupling almost entirely. The omega sweep this command prints is the only way to tell which regime a result sits in -- a single modularity number at a single omega cannot distinguish 'the layers genuinely agree' from 'the coupling was strong enough to force them to'. Report the omega beside every multilayer modularity number, not just once at the top of the section.
- **A layer-community can belong to two multilayer communities; that is the design** — §40.2's own worked example (p. 574, Figure 40.2b) has one per-layer community sit in two of the maximal sets at once -- 'we are ok if a community gets merged in different sets'. `layer_by_layer_communities` keeps that in `multilayer_communities`, and only flattens a node to one final community, by the largest-clique tie-break, so that Atlas ch. 36's battery has a disjoint partition to score. A node the tie-break assigned away from its second-best set is not wrong; read `overlap_note` for how many nodes that happened to before trusting one number off the partition.
- **Redundancy and complementarity answer different questions about density** — §40.4 states outright that multilayer density 'is an ambiguous concept': redundancy asks whether a community's pairs are joined through *every* layer at once, complementarity whether they are joined through *many* layers, each mostly its own. A community can score high on one and low on the other -- a group entirely wired by one dominant layer is maximally redundant in that layer alone and, because variety is `(|Lc|-1)/(|L|-1)`, scores near zero on complementarity. Quote whichever question the analysis is actually asking; quoting one number as 'the' multilayer density answers a question that was not asked.
- **This module's homogeneity is not the book's; the worked example does not reproduce** — §40.4's homogeneity term is '1 - sigma_c/sigma_max_c... normalized by its theoretical maximum', but the book never states that maximum, and its own worked numbers (p. 583) cannot be reconciled with any natural reading this module tried: sigma_c = 1.886 is exact for the stated vector, and the stated ratio of two thirds implies a maximum of 2.828 that is not the standard deviation of the same total concentrated into one layer (5.185), which is the definition this module uses. The homogeneity and complementarity this command prints are therefore this module's own reproducible definition, stated in `DEPARTURE_HOMOGENEITY`, not a reproduction of the book's arithmetic -- report it as such rather than as the chapter's own number.

### Core-periphery (every grouping report)

- **Communities in a core-periphery network are the periphery's tail** — §32.2 states the incompatibility outright: "in CP there isn't space for communities, given that there's only one dense area and everything connects to it. In CD, there's little space for peripheries, and there are multiple cores." A partition handed to a core-periphery network still comes back with groups, because the periphery has to be cut somewhere, and their modularity does not look absurd. So every grouping report fits both ideal patterns of Figure 32.4 to the same adjacency and prints which one is closer. When the verdict is core-periphery, name the core and the size of the periphery and stop; the communities below it are the cut, not groups the corpus put there.
- **A rich club is a ratio against the degree-preserving null** — phi(k), the density among the nodes of degree above k, rises with k on every network including a random one, so a rising phi is not a rich club and the raw curve is not evidence of anything. §32.1's claim (p. 450) is comparative -- the cores of some empirical networks are "even denser than what you'd anticipate by looking at the degree distribution of the network" -- so the column to read is rho(k) = phi(k) / mean phi over degree-preserving rewirings, and the claim needs rho above 1 sustained over the high k, not at one threshold where the club is three nodes. The same applies to the core-periphery correlation itself: a broad degree distribution buys one for free.
- **Coreness and k-core are not the same claim** — the report prints both and they disagree on purpose. §32.1 keeps k-core out of this chapter deliberately, because it "finds a fundamentally different type of core-periphery structure, which is more similar to a hierarchical decomposition of the network" (p. 450): the k-core is a nested peeling, a floor shared by whole shells at a time, while chapter 32's coreness is a hub-and-spoke fit scored by how well it reproduces the adjacency. A node deep in the k-core is in a well-connected region; a node with high coreness is in *the* dense region. Say which of the two you mean, and never quote a k-core number as evidence of a core-periphery structure -- the report's Jaccard between the deepest shell and the fitted core is how far apart the two answers are on this network.
- **Nestedness is the two-mode core-periphery, and it competes with modularity** — on the two-mode network the same structure appears as nestedness (§32.4): the smaller rows are subsets of the larger ones, the sorted matrix is "upper-triangular", and §32.4 gives it as the generalisation core-periphery is a case of. It competes with a community reading of the same matrix exactly as core-periphery competes with one in a single mode -- a perfectly nested matrix has no blocks, because every row overlaps every other. And NODF needs its null more than most numbers do: the chapter reports the objection that nestedness "could arise simply from the degree distribution ... and that there are fewer nested system than we originally thought" (p. 458), so read NODF against the fixed-fixed curveball null that holds both margins, never on its own.

### Uncertain edges (analyze --uncertain)

- **An edge is a claim with evidence; p says how much** — no edge in these networks is an observed tie. It is what an extraction pass named, a matcher found a passage for, and a projection turned into a pair, and every report now carries `p` -- the probability the edge exists -- built from that evidence: the tier of the mentions behind it (verbatim, token, fallback) and how many passages support it, combined as 1 - prod(1 - p) (Atlas §28.1-28.2). A network whose edges come from records rather than matches -- speakers on a document, topics at ingestion -- carries p = 1, which says this pipeline can put no number below one on them, never that they are true.
- **A ranking whose intervals overlap is a tie** — `analyze --uncertain` re-measures every centrality over sampled possible worlds (§28.2) and prints a mean and a 95% interval beside the observed value. Where two adjacent intervals overlap, some of those worlds put the other node first: the two are tied, and the order they are printed in is a presentation choice. The report counts those pairs. A top-ten whose intervals all overlap is one finding -- these ten -- and not ten.
- **The interval is over the edges you have, not the edges you missed** — §28.1 has two halves: spurious edges and missing ones. Every number here is about the first. A tie the corpus never wrote down has no row, no probability and no place in any possible world, so it cannot widen an interval -- and the second half of the error stays exactly as invisible as it was before the section was printed. Correcting it needs the network measured twice or a generative model of it, and the chapter is blunt that a poor model gives every edge a useless 50-50 (p. 400).
- **An expected degree is not a degree** — E[k] = sum_v p_uv is exact and is still not a degree: the book's own objection is that 'the degree is a count, it shouldn't be a continuous number' and that two nodes with radically different neighbourhoods both come out at 3.4 (§28.3, p. 403). The report prints the mode of each node's exact degree distribution beside it, which is a degree the node could actually have; quote that, or quote the distribution.
- **This is not backboning** — §28.2 draws the line itself: backboning removes spurious edges, probabilistic analysis 'want[s] to use all the information we have and integrate it in the analysis, no matter how unlikely an edge is to exist' (p. 401). Nothing in `--uncertain` deletes an edge, and a small `p` is not a small `p_value`: on a backboned network the first says the edge probably is not there and the second says it probably is. Use `--backbone` to cut and `--uncertain` to price; doing both means the intervals describe the surviving sample.
- **An expensive centrality is opt-in, and a budget can cut a table short** — not the book's own rule, but this pipeline's: `--uncertain` re-runs every centrality on every sampled world (§28.2), and the four that need a shortest-path search from each node -- betweenness, closeness, harmonic, reach -- cost one more of those per realisation, which is what made this section slow on a large network. They are left out by default; `--uncertain-expensive` opts them back in, and the report names whichever ones it skipped. `--max-seconds` bounds the realisations too -- one wall-clock budget shared across every measure, not reset per measure, five minutes unless you name another and none at `--max-seconds 0` -- but it can only stop the *next* realisation from starting, not one already running: on a network where a single expensive call costs more than the whole budget, the actual time spent is that call's cost, not the number given. Read the realisation count and the caveat the report prints beside each interval, not only its width, before comparing two `--uncertain` runs at different budgets.

### Spreading (sna spread)

- **A spread on a corpus network is a what-if** — nothing spread. Chapter 20's models are dynamics you *embed* in a network -- 'Here, edges don't change, but nodes can transition into different states' (p. 286) -- and the network they are embedded in here joins two nodes because a passage named them together, never because anything passed between them. So `sna spread` answers a structural question: *if* something moved along the lines this corpus drew, at rate beta, how far would it get and how fast. A final size of 92% is a statement about how well connected the network is; it is not a forecast, and it is not evidence that any idea in the corpus travelled. Two consequences worth printing beside it: the run ignores edge weights, because an edge is one contact and nine passages about a pair are nine sentences rather than nine meetings; and a spread never leaves the component it started in, so on a fragmented network the ceiling is the seed's component, not 100%.
- **The threshold is 1/lambda_1, and a hub lowers it** — whether a spread persists or dies is decided before any simulation runs, by `lambda = beta/mu` against a threshold (§20.2). The report prints three. The one to read is the spectral `1/lambda_1`, exact for the network in hand; the other two are the book's own, and each is a property of a *model* of a network -- `1/(k+1)` for a Gn,p graph (p. 294) and `k/k^2` for a preferential-attachment one. The gap between them is the finding: squaring the degrees lets the hubs eclipse everyone else, which is why `k/k^2` 'tends to zero' and why 'any disease, no matter beta and mu, will be endemic in a network with a power law degree distribution' (p. 295). A corpus network has a heavy tail, so expect a low bar and do not read a large outbreak as a strong contagion. R0 is printed next to lambda and is **not** the book's: chapter 20 defines lambda and the thresholds and never mentions R0.
- **Complex contagion needs many neighbours, not far ones** — `--model threshold` is chapter 21's reinforcement: a node transitions only when several neighbours already have. That one change inverts the usual advice. A bridge -- the single edge between two groups that carries a simple contagion across the network -- cannot carry a complex one, because one infected neighbour is one, and 'the outbreak from a single seed is impossible' (figure 21.1, p. 300); 'any kappa > 1 renders peripheral nodes safe, since most of them have only one connection' (p. 302). And the two forms of the trigger disagree about hubs: an absolute kappa (`--threshold 2`) is easy for a hub to clear, while the cascade fraction (`--threshold 0.3`) needs hundreds of its neighbours, so 'in a threshold model, hubs are the primary spreaders of the disease. In a cascade model, they're the last bastion of defense' (p. 303). Say which of the two you ran; the same number in `--threshold` means two different models on either side of 1.
- **Immunise a friend, not a random node** — §21.3's strategies are not equally informed and the report prints the mean degree each one caught, so the difference is visible rather than asserted. `degree` and `betweenness` need the whole topology, which a corpus you are still ingesting does not have; `acquaintance` -- 'pick a node at random in the network and vaccinate one of its friends' (p. 308) -- needs none of it and still lands on hubs, because a random neighbour is reached through every one of its edges. That is the friendship paradox (§31.2) doing the work. Judge the result on both of the chapter's criteria: the drop in final size (figure 21.9) *and* the delay, the area between the two curves (figure 21.10), because 'just delaying the inevitable' is a success when the delay buys something -- and the whole comparison inverts for viral marketing, where you want the curve to move the other way.
- **Driver nodes count controllability, not importance** — the driver-node share of §21.4 answers 'how many levers would it take to steer this network into a state you choose', by maximum matching (Liu, Slotine and Barabasi 2011, whose derivation the chapter says is beyond it). It is not another centrality and does not rank anybody: the chapter's own finding is that 'driver nodes tend not to be hubs', and a set of them is one valid answer among many of the same size -- figure 21.11's blue nodes are the ones that 'could or could not' be chosen. Read `N_D/N`, not the list. And on every network here except `relations` the edges have no direction, so they are read as running both ways, which is a reading this package chose and not the book's directed case.

### Embeddings (sna.embed, --features spectral|node2vec|metapath2vec)

- **A node's own adjacency row is not an embedding** — §42.1 works through why a slice of `A` will not do, and the three reasons are worth repeating whenever a raw feature row is about to stand in for one: it is not low-dimensional (one entry per node, so it saves nothing), it is not dense (mostly zero, which starves anything downstream of information), and it is not permutation invariant (two isomorphic graphs recorded in a different node order give two different matrices). `embed.spectral` exists because all three have to be fixed before K-means, a Gaussian mixture, or a distance comparison (ATL-47, ATL-48) means anything.
- **There is no the embedding of a graph** — §42.1's own figures make the point with one graph: Figure 42.1 groups nodes by community, Figure 42.4 by structural equivalence, and both are 'a perfectly valid alternative embedding' of the same nine nodes. `Embedding.method` and `Embedding.provenance` exist so a report never compares two embeddings' vectors without first checking they answer the same similarity question -- an entity's spectral coordinates and its mean text embedding (`sna/analysis.py`'s two `--features`) are both valid, disagree by design, and are never the same object.
- **A spectral embedding is a choice of matrix, and different choices disagree** — §42.3 fits a loss function by decomposing 'a transformation of A... but most often of the Laplacian L', and says plainly that 'this is the main thing differentiating spectral embedding approaches'. `embed.spectral` uses the symmetric normalised Laplacian; Locally Linear Embedding and Laplacian Eigenmaps are the same family on `(I - A)^T(I - A)` and on this same matrix respectively, and a caller who needs one of those exact losses builds it from `matrices.eigenpairs` rather than assuming this function already produced it. Two spectral embeddings of the same graph from two different matrices are not the same embedding and are not expected to agree.
- **A disconnected graph's spectral embedding separates components first** — §8.4's fact about the Laplacian -- the zero eigenvalue's multiplicity is the number of connected components -- means a network in several pieces fills that many embedding dimensions with component membership alone before anything about the structure inside one component appears. Two nodes landing far apart in `embed.spectral`'s first dimension can mean nothing more than 'different component'; check `paths.components` before reading distance in a low dimension as structural dissimilarity.
- **Flat pooling forgets which node contributed what** — §42.4 calls mean/sum/max 'flat' pooling precisely because they do not look at the network's structure, unlike the node-clustering and node-drop families the section also describes and this package does not build. A pooled graph vector answers a question about the whole network, never about one node in it, and `sum`'s magnitude grows with node count, so only `mean` is safe to compare across two graphs of different order (ATL-48's job); `sum` and `max` compare two graphs only when they have the same number of nodes to begin with.
- **Fewer columns than --dims asked for is the honest answer, not a bug** — an exactly (or nearly) regular block of a network -- a clique above all, but any pair of nodes with identical neighbourhoods does it too -- has a run of mutually tied eigenvalues with no canonical basis: any orthonormal basis of it is equally valid, and a dense, deterministic solver's particular choice can concentrate almost all of one dimension's weight on a single node rather than spreading it out. `embed.spectral` will not hand that to a distance-based consumer dressed up as structure: asking for `--dims` that would cut partway through such a run truncates to the last dimension before it instead, and the report says so, with the eigenvalue of the block that was dropped. A K-means or GMM run over a network with a large, uniform clique should expect fewer effective dimensions than `--dims` named, and that is the count to trust, not the flag.
- **Random-walk embeddings are transductive, not inductive** — §43.5's own account: 'these methods are not inductive, but transductive' -- a `node2vec`/`deepwalk`/`metapath2vec` vector describes the walks a graph happened to produce it from, and a network that adds, removes or reweights even one edge changes those walks and every vector trained from them, an untouched node's row included. There is no update rule: a report that reused yesterday's `--features node2vec` vectors on today's graph would be comparing a stale reading to a live one and calling it the same measurement. Retrain from scratch on the network the report is about, every time; chapter 44's message-passing models are the chapter's own pointer to the inductive alternative.
- **Random-walk embeddings read the edges alone** — §43.5 opens by naming what the family never fixed: it 'ignored node and edge attributes'. `node2vec`, `deepwalk` and `metapath2vec` all train on which nodes a walk visited, never on a node's own recorded attribute (`--where`) or an edge's weight beyond how it biases the walk that produced the vector. Two speakers who never appear near each other in a walk get distant vectors whatever their attribute says, and a `--by` section run against one of these embeddings is asking a different question than `--by` run against the graph itself.
- **The walk itself is a hyperparameter, not a default** — §43.1: random-walk embeddings 'require you to set a lot of parameters. You need to decide how many walks per node you want to do, how long the random walks should be, etc.' `node2vec`/`deepwalk`/`metapath2vec` fix `WALKS_PER_NODE`, `WALK_LENGTH`, the skip-gram window and epoch count as module constants bounded for a pure-numpy training loop (`EMBED_MAX_NODES`), not as values the book derived. A report naming a result from these embeddings has to print them -- `Embedding.provenance` carries every one -- and a different choice is a different, equally valid embedding rather than a correction of this one, the same warning §42.1 gives spectral embedding's `dims`.
- **A plain random walk on a two-mode network drowns in the busier side** — §43.3's own example: a paper with 5,154 co-authors gives every pair of them a valid length-2 path, so an unconstrained walk on a heterogeneous network spends almost all its budget re-visiting whichever node type has the most edges rather than reading the structure. `metapath2vec` fixes this by walking a stated schema (`speaker`-`entity`-`speaker` here) instead; it is the one to reach for on `--network speakers-entities` without `--project`, and `--features node2vec` is not.

### Visualization (sna draw)

- **Node size is area, never radius** — a viewer perceives the *area* of a circle, not its radius, so a size that doubles must not quadruple what the eye sees (§49.1, p. 717-718: 'double degree, but four times as large!'). `sna draw --size` places the (quasi-logged) value linearly in area and takes the square root for the radius; a reader hand-tweaking the SVG afterwards -- or reading the raw values out of `--json` to redraw them elsewhere -- has to do the same or the picture stops being truthful about which node is bigger by how much.
- **Nine colours total, node and edge together** — 'never use more than nine colours' (§49.2 p. 719) and the sum across node and edge colours together, not nine each (§50.1 p. 729: 'the sum of the distinct number of colours for edges and nodes must be nine or lower'). `--color community`/`attr:<key>` keeps at most eight named, colour-blind-safe hues and folds every category beyond that into a single grey 'other', named in the notes so a collapsed category is never mistaken for one that was simply absent; edges here carry no categorical colour of their own, only a sequential one from the same measure their width uses, so the node palette is the whole budget.
- **No rainbow scale for a quantity** — a hue sweep is not perceptually uniform -- equal steps in hue do not read as equal steps in the eye, and two ends of a scale like Matlab's jet can become indistinguishable once printed in greyscale (§49.2 p. 721-722, Figure 49.12-49.13). The sequential palette `sna draw` uses for `--size`, for edge weight/betweenness and for a numeric `attr:<key>` is one hue, monotone in lightness only, by construction; it has no diverging, meaningful-midpoint form (Figure 49.9, top) because nothing this package colours has a midpoint that is not itself a modelling choice.
- **A hairball is a layout failure, not evidence of no structure** — 'hello hairball, my old friend' (Figure 49.2) is the chapter's own running joke: a cluttered node-link picture says the layout chosen was the wrong one for this network's density, never that the network has no structure. Force-directed (§51.1) is for sparse to medium-sparse networks with real clusters; a network that still looks like a smudge under it is the chapter's own cue to try `--layout circular` (denser networks, especially ordered by `--color community`) or `--layout matrix` (dense enough that individual edges stop mattering and only blocks do), not a conclusion to print.
- **A long straight edge can look like a path through a node it never touches** — this is a *ghost edge* (§51.3, Figure 51.9): three nodes on a force layout can land so that an edge between two of them appears to pass through the third, purely because that node's position is a coincidence of the physics, never a claim about a path. The book's own fix needs organic or orthogonal edge routing (Dwyer et al. 2006), which this module does not implement -- a crossing in `--layout force`, `circular` or `arc` is read from the edge list, never from where a line happens to pass on the canvas.
- **Force-directed coordinates need a seed to mean anything twice** — a network's node-link position is relative to its other nodes, not absolute -- unlike a scatter plot, moving a node does not change the data (§51.1 intro, Figure 51.1) -- which is exactly why coordinates from two unseeded runs are not comparable and 'a small change in the initial conditions... will result in a completely different layout' (§51.4, p. 745-746, the chapter's own argument for hive plots). `sna draw --seed` fixes `--layout force`'s spring simulation so its `--json` positions can be rerun and checked; without one, the report says the coordinates cannot be reproduced rather than staying silent about it.
- **A layered drawing's dummy waypoints are not nodes** — an arc that skips more than one layer gets placeholder waypoints so it can bend around the layers it crosses (§33.6's Sugiyama convention); `--layout layered` draws them as points on the edge's own path, never as circles, and they carry no measurement -- a count of 'nodes' read off the picture that includes them is wrong. An arc drawn dashed is one §33.6's layering could not send downward (the arcs agony charges for, when `levels` came from it); this is the same layering `sna hierarchy` reports, just drawn.

### Node vector distance (sna distance)

- **Five numbers, five questions, never one 'the' distance** — ch. 47's own summary is explicit that its methods are 'organized' rather than ranked: the Euclidean baseline warps into the generalized Euclidean by swapping in the Laplacian pseudoinverse, the shortest-path family aggregates by linkage or by an exact optimal-transport minimum, the graph Fourier route reads how sharply a signal changes across an edge, and network variance and correlation ask a statistical question rather than a geometric one. `sna distance` prints all five sections rather than choosing a favourite, because they measure different things and can legitimately disagree about which of two comparisons is 'closer'.
- **The graph Fourier distance is a family, and not necessarily the best member of it** — p. 692 says so about its own construction, twice, in the same paragraph: 'this is not one, but a family of measures' -- the Euclidean comparison of the two filtered signals could be replaced by cosine, correlation, or any other off-the-shelf measure, since the network's topology is already folded into the filtering -- and 'the transformation proposed here might not be the optimal one', since graph Fourier filtering has other uses this chapter never explores (signal cleaning, frequency analysis, sampling, interpolation, trend filtering). Treat `graph_fourier_distance`'s number as one reasonable answer to 'how different are these two signals on this network', not the answer.
- **Cosine distance does not respect the triangle inequality** — p. 682 makes this the chapter's own worked point: two points needing a longer rope to connect (a large Euclidean distance) can be closer by cosine than two points needing a shorter one, because cosine reads only the angle between two vectors, never their magnitude. Do not carry triangle-inequality intuitions -- 'a is close to b and b is close to c, so a is close to c' -- across into a cosine number.
- **Generalized Euclidean across components is each side's own spread, not zero** — the Laplacian pseudoinverse L-dagger is block-diagonal across connected components, so the *cross* term between two nodes in different components is exactly 0 -- but each side still contributes its own diagonal term, which is generally positive. Two vectors whose mass sits in different components therefore read as some finite, usually nonzero, number: not 0 (they are not 'the same'), and not infinite the way `graphrag.sna.walks.effective_resistance` reads a cross-component pair. Read that number as how spread out each side's own mass is within its own component, never as a distance between the components themselves.
- **Both mass-moving vectors are rescaled before they are compared** — every §47.3 method -- single, complete and average linkage, and the earth-mover optimum -- first scales the lighter of the two vectors up so both carry the same total mass (p. 688). The number this produces is therefore not comparable to one computed on the raw, unscaled vectors elsewhere in a report: read it as 'how far does this much mass have to move', with 'this much' fixed by whichever side started heavier, not as a statement about the two vectors' original totals.
- **An unreachable pair is a legitimate infinity here, not an error** — `shortest_path_linkage_distance` and `earth_mover_distance` return infinity, rather than raise, when the mass genuinely cannot all move -- some node carrying mass has no path at all to a node on the other side. Complete linkage can reach that infinity before most of the mass has actually moved, because it always picks the *farthest* still-active pair first, and an unreachable pair is farther than any reachable one. A disconnected network is a legitimate answer to 'how far apart are these vectors', not a bug to work around.
- **Network variance assumes a distribution; network correlation needs variation** — `network_variance` (§47.5) normalises its vector to sum to 1 first, because the formula on p. 693 is stated for 'a distribution x'; a raw count vector is not one until this happens, and the normalisation is silent unless the total mass is non-positive, when it raises instead. `network_correlation` -- a choice this package made, since §47.5 states no closed form (see its docstring) -- is undefined for a vector that is constant on every connected component, the same way Pearson correlation is undefined for a constant sequence: there is nothing to correlate.
- **Network correlation is not mean-centred, and is not on variance's scale** — `network_correlation` does not subtract each vector's own mean before comparing them, unlike the ordinary Pearson correlation the papers it is loosely built on (Coscia 2021; Coscia and Devriendt 2024) generalise, so two vectors sharing a large constant offset will correlate more strongly here than a mean-centred formula would report. And its number, a bounded [-1, 1] cosine in the L-dagger inner product, is not a second reading of `network_variance`'s unbounded Ω² sum -- the two never belong on the same axis of a report, however much they look like a matched pair.

### Graph summarization (sna summarize)

- **Summarization is not backboning** — `sna summarize` and `sna backbone` answer opposite questions about the same hairball. Backboning removes edges and keeps as many nodes as it can, "because you want to let the strong connections emerge, but you want to preserve all the entities in your data" (ch. 27, p. 382). Summarization does the opposite on purpose: aggregation (§46.1) and compression (§46.2) merge nodes into super nodes, trading every individual entity for a meso-level view -- "graph summarization only focuses on empowering meso-level analysis [...] where you lose sight of the single individual nodes" (p. 382). Run `sna backbone` when the entities still have to be named afterwards; run `sna summarize` when they do not.
- **`--by attr:<key>` can group by a label the edges already explain** — a super node's internal density is only a reading of the attribute when the label was recorded about the node itself. A label *borrowed* from the documents a node's passages sit in shares its source with the edges that connect those same nodes, so the group is dense partly because of how the label was assigned, not because the attribute sorts the network -- the same circularity `sna analyze --by` guards against (CLAUDE.md, node attributes). `sna summarize --by attr:<key>` prints the borrowed share next to the density for exactly that reason; read the density as evidence about the attribute only when that share is low.
- **A compression cost is a property of the grouping, not of the graph** — "is this network compressible" is not a question §46.2 answers on its own: the bits reported are the cost of *this* grouping's model M plus its corrections, and a different grouping of the same graph can cost more or fewer of them. A low ratio is evidence that the chosen grouping (Louvain's communities, or an attribute's values) captures real block structure; a high one is evidence against that grouping, not against the graph having any structure at all. Nothing here searches for the cheapest grouping the way Grass does (p. 671); `sna summarize` reports the cost of the grouping `--by` already named.
- **The influence summary is a simulated what-if, on top of an edge that is already one** — §46.4's SPINE reading keeps the edges an *observed* spreading process used (Fig 46.8); this corpus never observed one, so `sna summarize`'s influence section simulates a plausible spread instead -- the same disclaimer `graphrag.sna.spread.SPREAD_FRAME` carries on every other spreading number in this package, stacked on top of the co-occurrence edges it runs over. Two what-ifs, not one: the seed ranking says which node would spread fastest *if* something spread along these edges *at all*, and neither half of that sentence is a claim about what the corpus recorded.

### Topological distances (sna compare --topology)

- **No distance answers every question** — the chapter's own opening figure (p. 698) makes the point before naming a single method: organise networks by node count and edge density and a star gets confused with a set of unconnected cliques, because 'whatever dimensions you use to organize your networks will collapse many -- possibly dissimilar -- networks into the same place'. Two Gn,m graphs of the same size and density can look identical on every count the chapter checks first and still differ in clustering coefficient by a factor of two (p. 699's Gn,m-versus-small-world figure). `sna compare --topology` runs all four distances of §48.1 for exactly this reason and never picks one as the verdict: a claim has to name which method it is reading, because they disagree by design, not by error.
- **Size confounds every distance in this chapter** — the same section's discussion of graph edit distance makes the general case: 'two Gn,p graphs with the same n and (low) p... will have almost no edge in common. As a consequence, their edit distance is large' -- a distance built for one data generating process reads a completely unrelated one as different for a reason that has nothing to do with shape. This package's own two size-sensitive distances say so directly: `topodist.spectral_distance` zero-pads the smaller graph's spectrum up to the larger's length, so a bigger graph reads farther from a smaller one for its size alone; `topodist.portrait_divergence` pads both portraits to the larger graph's shell and node counts the same way. NetSimile is the one member of the four built to resist this -- its signature is a moment of a per-node feature, never a sum over nodes -- and is the distance to reach for first when the two graphs being compared are not close in size.
- **A spectral distance of zero is not proof of the same topology** — §48.2 (p. 707) states one direction of the fact spectral methods lean on: 'isomorphic graphs have the same spectrum, thus similar values in the eigenvectors of the Laplacian imply that the nodes are relatively similar'. The converse is not true -- cospectral, non-isomorphic graphs are a standard counter-example in spectral graph theory, not an edge case -- so `topodist.spectral_distance` reading 0 (or near it) is evidence for similar shape, never proof of it. Corroborate with NetSimile, DeltaCon or portrait divergence before reporting two networks as topologically the same on the spectral number alone; the four are run together in `sna compare --topology` for exactly this reason.
- **Identity alignment is the lucky case, not the general one** — §48.2 opens with the case where node ids already name the same entities -- two dated snapshots of one persona network, which is what `sna compare --topology` always feeds `topodist.delta_con` -- and spends the rest of the section on the harder case: 'you might not be as lucky... you will need to figure out who is who in all the networks you collected'. `topodist.align_by_id` only ever does the easy case, by shared id, and says so in its own note whenever the two graphs it was given share little or none of their id space; comparing two networks built from different corpora, or from a persona's two different node kinds (speakers against entities), needs the pairwise-similarity or maximum-common-subgraph methods §48.2 describes and this package does not build. A DeltaCon similarity computed over two such graphs is reading the node sets, not the structure.
- **Network fusion here is the book's own worked example, not the genomic algorithm** — Figure 48.10 (p. 709) fuses two networks by averaging an edge's weight across the observations that carry it and keeping the result above a threshold, and says outright that 'real network fusion algorithms are much smarter and more sophisticated than this'. `topodist.fuse_networks` is that worked example and nothing more: the algorithm the chapter actually cites (Wang et al. 2014, Similarity Network Fusion) iteratively cross-diffuses each observation's own kernel through every other one, which needs a neighbourhood-size parameter the chapter never states a default for. Read a fused network as 'which ties survived averaging', not as the more sophisticated consensus the citation describes.

### Graph neural networks (sna complete, sna.gnn)

- **Two structurally identical entities get identical suggestions, not two readings** — §44.2 opens with the chapter's least intuitive limit before it gets to smoothing: a message-passing network 'can distinguish -- at best -- the same structures of Weisfeiler-Lehman' (p. 639), and its own worked example (Figure 44.3(a)) is two non-isomorphic networks that 'generate the same embeddings' regardless of depth, because nothing in the architecture tells two nodes with the same neighbourhood apart. Two peripheral entities off the same hub, mentioned nowhere else and with no distinguishing text, are exactly this case for `sna.gnn`: `classify` and `embed_gnn` give them the same row whatever `--hidden-dims` says, so an `sna complete` suggestion for one is not a reading of its own position -- it is the other one's label, borrowed by construction. The book's own fix -- concatenate a unique-id feature per node (the |V|x|V| identity matrix) -- is what the known-answer tests here use to tell a planted partition's own seed nodes apart, but the chapter immediately flags the cost: 'we lose the permutation invariance requirement, since now we're forcing the graph to have a specific node ordering' (p. 640), which is exactly what §42.1's embedding contract this package inherits refuses to do outside a test. `build_features`'s degree-only fallback (no text at all) makes this worse, not better: same-degree, same-neighbourhood nodes then start from literally the same input vector, with no unique id to break the tie.
- **More layers smooths the network to one vector, it does not see farther** — §44.2 keeps unfolding the same message-passing network past the layer count anyone would stop at (Figures 44.2, 44.5) and watches every node's embedding converge: 'all embeddings are getting more and more similar to each other ... a few more layers, and each node will get the same embeddings.' `sna.gnn.classify` and `sna.gnn.embed_gnn` default to shallow stacks for exactly this reason -- a `--hidden-dims` longer than the network's own diameter is not a deeper reading of the corpus, it is the consensus dynamics of §11.6 read as if they were still structure. Check the loss curve before adding a layer, not the intuition that more should mean better.
- **A peripheral node's own signal does not survive many hops** — §44.2's over-squashing figure (44.6) is a game of telephone every hop plays with a message: 'by the end, the message passed to the final recipient node is completely green: the intermediate nodes have erased all the blue information.' A high-betweenness hub between a peripheral entity and every labelled node squashes that entity's own signal before it arrives, so an `sna complete` suggestion for a node several hops from every owned label is reading mostly the hub's attributes, not the node's -- its confidence column is the number to distrust first, not the class it picked.
- **A GCN's W only fits the ground truth it is handed, and a borrowed label is not one** — §44.3 is explicit that W exists only to fit some ground truth T by backpropagation -- 'a MPGNN cannot do this because it doesn't have W ... a GCN can, because it has W' -- so `sna complete`'s suggestions are only as honest as the labels it is trained on. This package's own borrowed-label rule (`sna.attributes`) already refuses to certify an assortativity finding when most of an attribute's values were derived from an entity's documents rather than recorded about the entity itself; `sna.gnn.complete_attribute` applies the same refusal one step earlier and trains only on entity-owned values, never on borrowed ones, so the network is never asked to predict the circular majority vote that produced them. Fewer than `MIN_OWN_LABELS` owned values, or fewer than two classes among them, refuses the run rather than fit noise.
- **A GAT's attention explains this fit's own loss, not a fact about the entities** — §45.1 draws GATs as the generalization that replaces a GCN's fixed '1/(sqrt(kv) * sqrt(ku))' with a function 'learned -- again with backpropagation -- a different attention value for each of v's neighbors'. That function is fit against the same loss `classify` and `complete_attribute` already fit `W` against -- the labels a run happened to be given -- so the neighbours `--method gat` prints as 'leaned on' for a suggestion are the ones that moved that loss down for the labelled nodes the network saw, not an independent measure of which relationship matters. A different seed or a different held-out split can move the weights even when the predicted class does not; read the attention column beside the confidence and the held-out gap, never in place of them. The over-squashing rule above (§44.2) still applies to a GAT exactly as it does to a GCN -- a learned weight on a hub's edge does not exempt the peripheral node behind it from losing its own signal on the way through.
- **Nothing here regularises against overfitting; the held-out gap is the check instead** — §45.5's own answer to overfitting is dropout -- zeroing random entries of W, or randomly dropping edges of A between layers -- which `sna.gnn` does not implement for any of the three methods (see `docs/SNA_FOUNDATIONS.md`'s practical-considerations checklist for why). What stands in for it is `complete_attribute`'s `val_share`: a share of the owned labels withheld before fitting, so `held_out_accuracy` beside `majority_accuracy` catches an overfit run after the fact rather than discouraging one during training. That is a weaker guarantee, most exposed exactly where `complete_attribute` already refuses to run at all -- close to `MIN_OWN_LABELS` owned values -- so trust a suggestion only as far as the held-out gap the report prints beside it, and treat a small held-out set's accuracy as noisy rather than as proof the fit generalises.

### Always

- Report n. A centrality ranking over 12 nodes is an anecdote with decimal places.
- Report the null model. Without it, modularity is a number, not a finding.
- Report the stability score. A partition that changes with the seed cannot carry names.
- Report the sampling frame. These networks describe who was recorded and what was written down, never a population; a node is absent when nobody wrote it down, which is not the same as it not existing.
- Say which features were used. Clustering the graph's structure and clustering what the nodes are about answer different questions and can disagree completely.

## Applications

Chapter 52 of the Atlas (pp. 757-769, `scripts/atlas_chapter.py 52`) is not a technique chapter;
it is eight worked cases the book's closing part reads network science through, and most of them
turn out to be this package's own persona questions in a different field's clothes. This section
reads them that way -- which technique answers which of *this* corpus's questions -- rather than
repeating the book's own examples (cities, brains, patents), which are `docs/SNA_FOUNDATIONS.md`'s
territory, not a corpus this repository holds.

**"Who is central" -- ranking (ch. 14, `sna analyze`).** §52.1's network-effects argument is that
an added neighbour's value is combinatorial, not linear, because it reaches everyone *its* own
neighbours reach. That is a description of what betweenness and eigenvector centrality already
measure over `speakers` or `entities`: `sna analyze`'s ranking table, and `sna robustness`'s attack
curves (ch. 22) for the harder question of what an org chart looks like without that person.

**"Which groups" -- community discovery and evaluation (ch. 35-36, `sna community` /
`analyze --method louvain|sbm|...`).** §52.8's polarization case (echo chambers, opinion
homophily) is the book's own worked example of what `sna analyze --by stance` and
`--method louvain` together answer on the `speakers`/`entities` networks filtered by `--stance`:
whether a corpus's praise and complaint split into separate camps, evaluated through
`evaluate.py`'s battery (ch. 36) rather than asserted from the partition alone. §52.7's Wikipedia
gender-asymmetry study is the same pattern one level down -- not "which groups" but "does this
attribute divide the network" -- which is `sna analyze --by <attribute>` (ch. 30-31) once a persona
tags speakers or entities by the attribute in question, with the borrowed-label refusal in
`CLAUDE.md`'s Rules section standing in for the study's own caution about what the data can and
cannot support.

**"What changed" -- windows and dynamics (ch. 7, `sna compare`, `layers.snapshots`).** §52.4's
"random point in a career" finding about scientific impact is a claim about a *sequence* of
publications, which on this corpus is `sna compare`'s two-window diff or `layers.snapshots()`'s
per-window series over `speakers`/`entities` -- whether an entity's or a speaker's centrality moves
in a trend or a random walk is the same question §52.4 asks of citations, asked of mentions
instead. §52.7's digital-humanities birth-death cities network (Schich et al., Figure 52.8: rising
and falling cultural centres) is the same shape again: entrants and leavers over time,
`compare_windows`.

**"What would confirm this link" -- prediction with evidence (ch. 23-25, `sna predict`,
`sna predict-eval`).** §52.6's memetics section is careful that virality correlates with structural
position (community-boundary origin spreads further) but stops short of claiming which specific
meme will go viral next; `sna predict`'s ranked hypotheses, each with the passages that would
confirm them, and `sna predict-eval`'s honesty check (nothing held out survives in the training
graph) are this package's answer to the same discipline -- a structural correlate is a hypothesis
list, not a prediction to act on unread.

**"How exposed is this" -- robustness and anonymity (ch. 22, `sna robustness`).** §52.2's
de-anonymization argument -- a handful of connections can single out a node others cannot name --
is a structural-uniqueness question, not a privacy technique this package implements, but it is
the reason `sna robustness`'s targeted-removal curves matter beyond failure analysis: the same
distinguishing structure that lets an attacker re-identify a node in §52.2's example is what makes
that node's removal disproportionate in `sna robustness`'s giant-component curve.

**Not this corpus's question.** §52.3 (human connectome), §52.5 (human mobility) and the innovation
half of §52.1 (city scaling exponents) are about networks this package does not build -- neurons,
GPS traces, urban infrastructure -- and no persona corpus here is that kind of data; they stay
read, not mapped. §52.7's archaeology and §52.4's science-of-science examples are read above for
the *method*, not built here as networks of their own, since this package has no citation or
excavation-site data to build them from.

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
- An attribute is a claim somebody wrote down about a node, so `--where` and `--by` describe the
  part of the corpus a tagging pass reached. Untagged nodes are outside every filtered network
  and every attribute measure; they are not a value, and a corpus tagged in half describes that
  half.

## Writing the result up

`analyze --out` already produces a markdown note with the rationale, the checks and the
caveats. To keep it, write it into a persona's corpus under `data/raw/<persona>/...` and
re-ingest, so the finding becomes citable the same way any other source is. Keep the JSON from
`--json` next to it: it holds the responsibilities, the full k table and the null distribution
that the note summarises.
