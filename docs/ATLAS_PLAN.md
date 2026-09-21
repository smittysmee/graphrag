# The Atlas programme: full network-science coverage for `graphrag sna`

**Goal.** A network scientist who knows *The Atlas for the Aspiring Network Scientist* (Coscia,
v2, 916 pp., free at <https://www.networkatlas.eu/>) can open this repository inside an
enterprise and run every analysis the book teaches over a persona's corpus, through the CLI, the
MCP server, or a notebook, with the book's own caveats printed next to every number. Nothing in
the book is passed over: every chapter below has a disposition, and a disposition of "documented,
not built" has to say why.

**Source of truth.** The book, not memory. `scripts/atlas_chapter.py toc` lists every part,
chapter and section with page numbers; `scripts/atlas_chapter.py 27 --section 27.6` prints one.
The PDF is cached under `data/cache/atlas/` (git-ignored). Chapter numbers here are the book's.

**Operating model.** The lead session orchestrates; Opus agents implement one ticket each in an
isolated worktree; an Opus reviewer holds the diff to the chapter and to this repo's rules before
the lead merges. See [Orchestration](#orchestration) at the end; the agent definitions live in
`.claude/agents/atlas-implementer.md` and `.claude/agents/atlas-reviewer.md`, and the brief the
lead fills per ticket is `docs/templates/ATLAS_TICKET.md`.

---

## What exists today

`src/graphrag/sna/` is already a disciplined, report-first SNA package. Every ticket is a delta
against this, never a rewrite:

| module | holds | Atlas chapters it already answers |
|---|---|---|
| `export.py` | five networks (speakers, entities, topics, speakers–entities, directed `relations`), `describe()` naming the network type, simple bipartite projection, `--stance/--facet/--since/--until/--where/--relation-type` filters, node attributes with provenance, naive `--min-weight` prune, six interchange formats in and out (`read_graph`/`write_graph`, `FORMATS`) | 7.1 (bipartite), 7.5 (attributes), 26.1 (simple weights), 27.1 (naive threshold) |
| `measures.py` | degree, weighted degree, betweenness, closeness, eigenvector, PageRank; participation-coefficient brokers; whole-network summary (density, components, clustering, degree assortativity) | 9.1, 12.1–12.2, 14.1–14.4, 31.1 (degree correlation), part of 15.1 |
| `cluster.py` | Louvain (multi-seed, stability), spectral embedding, K-means, GMM with k selection, ARI/NMI, degree-preserving null for modularity | 35.6, 36.1, 36.4, 42.3, 18.1 (configuration null), 19.1 (shuffling) |
| `attributes.py` | categorical assortativity with permutation null, attribute-partition modularity with fixed-partition rewiring null, agreement with clusters, label provenance and the borrowed-label confound | 30.2 (quantifying homophily), 36.1, 36.4 |
| `compare.py` | one network over two windows or two attribute values: n, entrants/leavers, partition agreement, rank movers | a first cut of 7.4 and 35.4 |
| `stances.py` | what the corpus says about each entity, signed pairs | groundwork for 24.1 (signed) |
| `guide.py` | the method-selection and reading rules, quoted verbatim by `docs/SNA.md` and the SNA skill (a test enforces it) | the book's "watch out" boxes, in miniature |
| `analysis.py` | one run end to end: summary, centralities, grouping, checks, attribute section, markdown + JSON | the report discipline the whole programme extends |
| `stats.py` | summary statistics with fat-tail warnings, distribution fits, empirical / binomial / permutation p-values, Pearson / Spearman / Kendall, MI / NMI / VI (ATL-03) | 3.1–3.5 |
| `matrices.py` | adjacency, stochastic, incidence, Laplacians, eigenpairs / SVD / NMF under one node-order contract (ATL-08) | 5, 8 |
| `null.py` | named nulls — Erdős–Rényi, configuration (edge swaps, directed too), label permutation, bipartite-preserving curveball that re-derives borrowed labels — one `significance()` contract, ERGM by maximum pseudo-likelihood (ATL-19) | 16.1, 18.1, 19.1–19.2 |
| `sampling.py` | induced, BFS, snowball, forest-fire, random-walk (+ Metropolis–Hastings), NRS samplers; bias report; re-weighted degree distribution; completion estimate (ATL-29) | 29.1–29.5 |
| `layers.py` | multilayer by stance / facet / relation type / source with `flatten` and supra-adjacency, passage hypergraph with clique expansion, dated edge list and windows (ATL-07) | 7.2–7.4 |
| `backbone.py` | the nine filters of ch. 27 plus §13.4's trees, `--backbone`/`--threshold`/`--alpha` on export / analyze / compare, `sna backbone` side by side (ATL-27) | 13.4, 27.1–27.6 |
| `walks.py` | stationary distribution, non-backtracking matrix, hitting and commute times, effective resistance, mincut and Fiedler cut, consensus; `sna walks` (ATL-11) | 2.7, 11.1–11.6 |
| `degree.py` | degree sequences and distributions, CCDF, log binning, the Clauset–Shalizi–Newman power-law test with its alternatives, per-mode fits on two-mode networks; `sna degree` (ATL-09) | 9.1–9.4 |
| `paths.py` | walks and matrix powers, cycle space, dyad census, components and the condensation, BFS / DFS, path-length distribution and eccentricities; `## Paths and components` in every report (ATL-10) | 10.1–10.4, 13.1–13.3 |
| `projection.py` | the eleven bipartite weightings of ch. 26 over the incidence matrix, `--projection` and `--lambda` on every projected network, the §26.6 comparison as `sna projections` (ATL-26) | 26.1–26.6 |
| `uncertain.py` | edge probabilities from mention tiers and passage counts, possible-world realisations, expectations and intervals for any measure, the §28.3 closed forms, the §28.1 evidence section, `analyze --uncertain` (ATL-28) | 28.1–28.3 |
| `generators.py` | the synthetic graphs of ch. 16–18 as seeded wrappers with the book's expected properties, the ch. 16 closed forms, and the "against random" comparison every report prints (ATL-16) | 16.1–16.5, 17.1–17.3, 18.1–18.3 |
| `roles.py` | Guimerà–Amaral roles, structural and regular equivalence, SimRank, blockmodel positions; `sna roles` and a `## Roles` section (ATL-15) | 15.1–15.2 |
| `measures.py` (ch. 14) | reach, harmonic, HITS, coreness and k-truss as centrality kinds, `centralization` against the star, `ranking_stability` and the `## Ranking stability` section (ATL-14) | 14.3, 14.5–14.8 |
| `assortativity.py` | numeric assortativity over edge endpoints, the neighbour-average curve, the friendship paradox, the attribute-by-degree bands; `--by` on a numeric key and `## Degree correlations` (ATL-31) | 31.1–31.3 |
| `ego.py`, `attributes.py` (EI/Coleman) | ego networks and alter composition by a categorical attribute; EI index and Coleman's homophily index per value, against a permutation null; `sna ego` and a `## Homophily` section (ATL-30) | 30.1–30.2 |
| `experiment.py` | random and temporal holdouts with negative sampling, k-fold, the honesty check that no held-out pair survives in the training graph; AUC, precision@k, precision-recall, prediction power with the chapter's cautions; `sna predict-eval` (ATL-25) | 25.1–25.2 |
| `evaluate.py` | the ch. 36 battery over any partition: resolution limit, conductance and its relatives with their size biases, the partition as a link predictor, mutual information against a truth; runs under every grouping report (ATL-36) | 36.1–36.4 |
| `robustness.py` | random and targeted removal curves against the Molloy-Reed fraction, the load cascade, interdependent coupling to the mutual giant component; `sna robustness` (ATL-22) | 22.1–22.4 |
| `motifs.py` | triad census and motif z-scores against the degree-preserving null, isomorphism and canonical form, transactional and single-graph frequent subgraphs; `sna motifs` (ATL-41) | 41.2–41.5 |
| `spread.py` | SI/SIS/SIR and complex-contagion simulations, the epidemic threshold against the spectral radius, immunisation strategies, driver nodes; `sna spread` (ATL-20) | 20.1–20.3, 21.1–21.4 |
| `coreperiphery.py` | discrete and continuous coreness, rich club against the degree-preserving null, nestedness on two-mode networks, and the tension check that says when communities are a core-periphery's tail; `## Core-periphery` under every `sna analyze` report (ATL-32) | 32.1–32.4 |
| `cluster.py` (ch. 37) | dendrograms from Louvain levels and Girvan–Newman with a cut chooser, the hierarchical random graph fit, directed Louvain on `relations` (ATL-37) | 37.1–37.4 |
| `overlap.py` | k-clique percolation, link clustering, the ego-network cover, overlapping NMI, the overlap paradox; `sna overlap` (ATL-38) | 38.1, 38.3, 38.5, 38.7 |
| `multilayer.py` | flattening, layer-by-layer matching, supra-adjacency modularity with omega, multilayer density, which-method rules; `sna multilayer-communities` (ATL-40) | 40.1–40.5 |
| `bipartite.py` | Barber modularity, BRIM, biLouvain, neighbour similarity, the curveball significance; three `--method` values on the two-mode network (ATL-39) | 39.1, 39.3–39.4 |
| `predict.py` | the neighbourhood, Katz, HRG and association-rule scorers, confirming evidence per pair, the ranked hypothesis list; `sna predict` (ATL-23) | 23.1–23.7 |
| `cluster.py` (ch. 35) | SBM (plain and degree-corrected), Infomap's map equation, Walktrap, label propagation, temporal community matching, local seed expansion; `--method` for all of them and `sna community` (ATL-35) | 35.1–35.5 |
| `predict.py` (ch. 24) | balance-theory sign prediction with its own sign holdout, cross-layer common neighbours; `sna predict --signed / --layers` (ATL-24) | 24.1–24.2 |
| `embed.py` | the embedding contract, spectral embedding with the eigengap cut, pooling (ATL-42); DeepWalk, node2vec and metapath2vec on a numpy skip-gram (ATL-43) | 42.1–42.4, 43.1–43.5 |
| `draw.py` | layouts, size and colour rules, edge styling, SVG and self-contained HTML; `sna draw` (ATL-49) | 49–51 |
| `vectordist.py` | distances between two node vectors on one network, from Euclidean to the Laplacian pseudoinverse, earth mover, graph Fourier and network variance; `sna distance` (ATL-47) | 47.1–47.5 |
| `summarize.py` | meta-network aggregation, MDL compression, simplification by importance, influence summaries; `sna summarize` (ATL-46) | 46.1–46.4 |
| `topodist.py` | spectral, NetSimile, DeltaCon and portrait distances between two networks, identity alignment, the fusion example; `sna compare --topology` (ATL-48) | 48.1–48.3 |
| `gnn.py` | GCN, GraphSAGE and GAT behind the optional `gnn` extra: attribute completion as suggestions with the attention each leaned on, a link scorer through `experiment.py`; `sna complete` (ATL-44, ATL-45) | 44.1–44.3, 45.1, 45.5 |
| `facade.py` | the notebook `Network` facade over the same functions the CLI calls; `docs/SNA_NOTEBOOK.md` (ATL-ENT-2) | — |
| `cache.py` | the on-disk network cache every `sna` command reads through, invalidated by every write path and a store fingerprint; `sna cache clear` (ATL-ENT-3) | — |
| `ties.py` | neighbourhood overlap, tie-strength-vs-overlap curve (Granovetter bins), weakest-bridge report, the two-mode structural-zero note, the homophily-vs-contagion caveat tied to temporal data; `sna ties` (ATL-30) | 30.3–30.4 |

Conventions every ticket inherits (from `CLAUDE.md`): config only through `Settings`; dependencies
injected through Protocols; tests use fakes, never `unittest.mock.patch`; modify in place; every
Neo4j statement on its own; `make check` (ruff format, ruff lint, `mypy --strict`, unit tests)
must be green; rules go in `guide.py` and the docs are regenerated from it, never hand-edited.

---

## Coverage matrix

One row per chapter of the book. **Build** = a ticket delivers it. **Doc** = it becomes a section
of `docs/SNA_FOUNDATIONS.md` or the glossary, with the reason it is not code. Every row has an
owner ticket; `—` in the "today" column means nothing exists yet.

| Ch. | Title | Today | Disposition | Ticket |
|---:|---|---|---|---|
| 1 | Introduction | — | Doc: the programme's own framing; the "how this book approaches complexity" argument becomes the opening of `SNA_FOUNDATIONS.md` | ATL-01 |
| 2 | Probability Theory | implicit | Doc + build: frequentist/Bayesian conventions stated once; Markov processes underpin ATL-11; §2.8 alternatives (fuzzy, possibility) documented as out of scope with the reason | ATL-03 |
| 3 | Statistics | scattered | Build: `sna/stats.py` — summary statistics with fat-tail warnings, distributions, empirical/binomial/permutation p-values, Pearson/Spearman/Kendall, MI/NMI/VI; every later significance test calls it | ATL-03 |
| 4 | Machine Learning | — | Doc: notation and loss/activation vocabulary for Part XI; no module of its own | ATL-42 |
| 5 | Linear Algebra | implicit | Build (with ch. 8): eigen/SVD/NMF helpers in `sna/matrices.py`; doc: notation | ATL-08 |
| 6 | Basic Graphs | undirected weighted only | Build: directed, typed `relations` network from `RELATED_TO`; network-type classification; reciprocity | ATL-06 |
| 7 | Extended Graphs | `layers.py`: multilayer, hypergraph, dynamic (ATL-07 landed) | Done; community discovery on layers → ATL-40 | ATL-07 |
| 8 | Matrices | ad hoc | Build: adjacency, row/column stochastic, incidence, combinatorial and normalised Laplacian, one node-order contract | ATL-08 |
| 9 | Degree | `degree.py` (ATL-09 landed): degree sequences and distributions, CCDF and log bins, discrete power-law MLE with KS `xmin`, bootstrap p against CSN's 0.1, lognormal / exponential comparison, per-mode fits; `sna degree` and a `## Degree` section | Done | ATL-09 |
| 10 | Paths & Walks | `paths.py` (ATL-10 landed): walk counts, closed walks and triangles from the trace, cycle space, dyad census, weak / strong components and the condensation, BFS / DFS, the path-length distribution with diameter, radius, centre and periphery, estimated above a thousand nodes | Done | ATL-10 |
| 11 | Random Walks | `walks.py` (ATL-11 landed): stationary distribution, non-backtracking matrix, hitting/commute times via the Laplacian pseudoinverse, effective resistance, s-t and global mincut, Fiedler cut, consensus | Done | ATL-11 |
| 12 | Density | `measures.py` (ATL-12 landed): local / average / global / weighted clustering, maximal and k-cliques under a budget, greedy independent set, `## Density` section | Done | ATL-12 |
| 13 | Shortest Paths | `backbone.py`: MST / PMFG (§13.4, ATL-27 landed) | Build (§13.1–13.3): exploration, path-length distribution (ATL-10); Doc (§13.5): classic combinatorial problems are not analyses of a corpus network | ATL-10 |
| 14 | Node Ranking | `measures.py` (ATL-14 landed): reach, harmonic, HITS hubs and authorities, coreness and shells, k-truss, Freeman centralization for every kind, `## Ranking stability` over bootstrap / configuration / backbone resamples | Done | ATL-14 |
| 15 | Node Roles | `roles.py` (ATL-15 landed): Guimerà–Amaral roles over any partition, structural equivalence (Jaccard / cosine / Pearson), SimRank, the regular-equivalence recursion, blockmodel positions and image matrix, same-role-never-co-occurring pairs; `sna roles`; §15.3 → ATL-42/43 | Done | ATL-15 |
| 16 | Random Graphs | `generators.py` (ATL-16 landed): ER, caveman, WS, BA, configuration, planted partition, LFR, random geometric, the ch. 16 closed forms | Done | ATL-16 |
| 17 | Understanding Network Properties | `generators.py` (ATL-16 landed): the `## Against random` section — clustering, path length and degree dispersion against G(n,p) and one shared family of degree-preserving rewirings | Done | ATL-16 |
| 18 | Generating Realistic Data | `null.py`: configuration model (ATL-19 landed) | Build: LFR-style benchmark, random geometric (ATL-16); §18.4 graph generative networks → ATL-45 | ATL-16, ATL-45 |
| 19 | Evaluating Statistical Significance | `null.py`: ER, configuration, label permutation, bipartite-preserving curveball, `significance()`, ERGM (ATL-19 landed) | Done | ATL-19 |
| 20 | Epidemics | `spread.py` (ATL-20 landed): SI/SIS/SIR simulation with bands over runs, the spectral epidemic threshold beside the G(n,p) and preferential-attachment ones, the logistic closed form; `sna spread` | Done; R0 printed as a non-book convention | ATL-20 |
| 21 | Complex Contagion | `spread.py` (ATL-20 landed): threshold (absolute and fractional) and limited-chance models, random / targeted / acquaintance immunisation with the final-size and delay readings, driver nodes by maximum matching | Done; undirected driver nodes read every edge both ways and say so; KKT greedy, φ and simplicial contagion declared out | ATL-20 |
| 22 | Catastrophic Failures | `robustness.py` (ATL-22 landed): giant-component curves under random removal and every centrality as an attack (`--recompute` re-ranks), the Molloy-Reed critical fraction, load-redistribution cascades with a tolerance sweep, two of a persona's networks coupled and failed back and forth to the mutual giant component; `sna robustness` | Done; edge failures and the α-dependence figures declared not built | ATL-22 |
| 23 | Link Prediction (simple graphs) | `predict.py` (ATL-23 landed): common neighbours, Jaccard, Adamic–Adar, resource allocation, Katz (closed-form walk sum with the β bound), HRG-based over ATL-37's dendrogram, one GERM wedge→triangle association rule over per-document transactions; every ranking evaluated through ATL-25 before the hypotheses print, each hypothesis with the passages that would confirm it; `sna predict` | Done; Katz sums walks not simple paths (stated); HRG reads the single MAP dendrogram; GERM node-adding rules and §23.7's other approaches declared not built | ATL-23 |
| 24 | Signed & Multilayer Link Prediction | `predict.py` (ATL-24 landed): balance theory over the stance layer's signed pairs (Figure 24.1's four triangle types, the sign vote, a held-out-sign evaluation against a coin-flip baseline), cross-layer common neighbours weighted by an inter-layer Pearson correlation and evaluated through ATL-25; `sna predict --signed` and `--layers` | Done; status theory (needs direction), GERM sign rules and §24.2's other methods declared not built; the weighting is not §9.1's layer relevance and says so | ATL-24 |
| 25 | Designing an Experiment | `experiment.py` (ATL-25 landed): random edge holdout with balanced negative sampling, temporal holdout on dated passages, k-fold; confusion matrix, ROC/AUC, precision-recall and average precision, precision@k, prediction power; preferential-attachment baseline; `sna predict-eval` | Done; precision@k ranks the test set, not every candidate pair (stated in the frame) | ATL-25 |
| 26 | Bipartite Projections | `projection.py` (ATL-26 landed): simple, Jaccard, cosine, Pearson, Euclidean, hyperbolic, resource, ProbS, HeatS, hybrid, random walk; `--projection` on every projected network; `sna projections` (§26.6) | Done | ATL-26 |
| 27 | Network Backboning | `backbone.py` (ATL-27 landed): naive, naive-top, doubly stochastic, high-salience, convex, disparity, noise-corrected, MST / PMFG; `sna backbone` comparison; directed DF and NC | Done | ATL-27 |
| 28 | Uncertainty & Measurement Error | `uncertain.py` (ATL-28 landed): mention tiers stored, edge probability `p` on every network, §28.1 evidence section, `analyze --uncertain` with expectations and intervals, §28.3 closed forms | Done; §28.4 in `SNA_FOUNDATIONS.md` | ATL-28 |
| 29 | Network Sampling | `sampling.py` (ATL-29 landed) | Done | ATL-29 |
| 30 | Homophily | `ego.py`, `ties.py`, `attributes.py` (ATL-30 landed): ego networks and alter composition, EI/Coleman homophily indices against a permutation null, tie-strength-vs-overlap curve, weakest-bridge report, homophily-vs-contagion caveat; `sna ego` (ties surfaces through `sna analyze`) | Done | ATL-30 |
| 31 | Quantitative Assortativity | `assortativity.py` (ATL-31 landed): numeric attributes in `facets.yaml`, Pearson / Spearman over edge endpoints with permutation and configuration nulls, the k_nn curve, the friendship paradox, the attribute by degree; `--by <numeric>` and `## Degree correlations` in every report | Done; the four directed coefficients declared as not built | ATL-31 |
| 32 | Core-Periphery | `coreperiphery.py` (ATL-32 landed): the Borgatti-Everett discrete core by correlation with the ideal pattern, continuous coreness by the leading eigenvector with k-core beside it, the rich club against the configuration null, NODF nestedness against the curveball null on two-mode networks, and the §32.2 tension check under every grouping report | Done; Rombach α/β, EM and random-walk detection, multilayer and temperature declared not built | ATL-32 |
| 33 | Hierarchies | `hierarchy.py` (ATL-33 landed): hierarchy type classification, cycle share (flow hierarchy), global reach centrality, maximum spanning arborescence, exact agony by linear program, layered drawing export; `sna hierarchy`, refuses an undirected network | Done | ATL-33 |
| 34 | High-Order Dynamics | `highorder.py` (ATL-34 landed): simplicial complex from passages with generalized degree/closure, second-order (memory) network from consecutive passages, memory-aware walk against the first-order one; `sna highorder` | Done | ATL-34 |
| 35 | Graph Partitions | `cluster.py` (ATL-35 landed): plain and degree-corrected stochastic blockmodels with a spectral-initialised greedy fit and BIC over k (graph-tool used when an interpreter already has it), the map equation and Walktrap, label propagation, temporal matching across ATL-07 snapshots with the six events, Clauset's local community; all of them and ATL-37's four as `analyze --method`, plus `sna community --seed / --temporal` | Done; no multilevel Infomap; the DC-SBM BIC penalty counts the degree parameters after review | ATL-35 |
| 36 | Community Evaluation | `evaluate.py` (ATL-36 landed): modularity with the sqrt(2m) resolution limit and the communities under it, conductance, internal density, cut ratio, normalized cut, the three ODF variants, coverage and performance with each measure's size bias, the partition as a link predictor against dealt-out labels, NMI/AMI/ARI/VI against a ground truth; `## Community evaluation` under every grouping report | Done; §36.3 holdout to be routed through `experiment.py` now that ATL-25 is in | ATL-36 |
| 37 | Hierarchical Community Discovery | `cluster.py` (ATL-37 landed): Louvain levels and Girvan–Newman as scored dendrograms with a peak-modularity cut chooser, the HRG fit by Metropolis MCMC over dendrograms, directed communities on `relations` without flattening; §37.4 as rules | Done; library functions, wired into `--method` by ATL-35; HRG consensus over the posterior and Peixoto's ordered communities declared not built | ATL-37 |
| 38 | Overlapping Coverage | `overlap.py` (ATL-38 landed): k-clique percolation, hierarchical link clustering cut at peak partition density, the ego-network (DEMON) method over `ego.py`, overlapping NMI by column matching, the §38.7 overlap-paradox check; every cover flattened through the ch. 36 battery; `sna overlap` | Done; the plan's §38.2 citation was wrong, the method is §38.5's; MMSB, BigClam, overlapping modularity/Infomap/label propagation, OSLOM and fuzzy membership declared not built | ATL-38 |
| 39 | Bipartite Community Discovery | `bipartite.py` (ATL-39 landed): Barber modularity, BRIM, biLouvain and neighbour-similarity co-clustering directly on the two-mode network, significance against the curveball null; `analyze --network speakers-entities --method brim|bilouvain|neighbor-similarity` groups both modes without projecting, the ch. 36 battery runs with a note on which measures assume one mode | Done; bi-clique percolation, the bipartite clustering coefficient, biSBM and Barber's adaptive k search declared not built | ATL-39 |
| 40 | Multilayer Community Discovery | `multilayer.py` (ATL-40 landed): flattening, layer-by-layer Louvain with Figure 40.2's maximal-clique matching, Mucha et al. multilayer modularity over the supra-adjacency with an omega sweep and a per-layer rewiring null, redundancy and complementarity, the §40.5 agreement-based recommendation; `sna multilayer-communities` | Done; single-level local mover rather than GenLouvain; the book's p. 583 homogeneity example does not reproduce, stated as a rule | ATL-40 |
| 41 | Frequent Subgraph Mining | `motifs.py` (ATL-41 landed): the 16-class triad census and motif profile against the configuration null, VF2 isomorphism and exact canonical labelling (bounded), gSpan-style transactional mining over per-document entity graphs with support per document, single-graph mining by minimum image support; `sna motifs [--mine]` | Done; canonical form by exhaustive relabelling rather than minimum DFS code, Weisfeiler-Lehman and five-node counting declared out | ATL-41 |
| 42 | Shallow Graph Learning | `embed.py` (ATL-42 landed): the `Embedding` contract (node order, dims, method, provenance), spectral embedding as the symmetric normalised Laplacian's eigenvectors from ATL-08's primitives with an eigengap cut that refuses to slice a degenerate eigenspace and says so in the report, mean/sum/max pooling for ATL-48; `cluster.spectral_embedding` routes through it | Done; the ch. 4 notation section already existed from ATL-01; LLE, edge pooling and clustering/drop pooling declared not built | ATL-42 |
| 43 | Random-Walk Embeddings | `embed.py` (ATL-43 landed): DeepWalk and node2vec on a numpy skip-gram with batched negative sampling proportional to degree, the second-order p/q walk, metapath2vec over a stated schema on the two-mode network, a cosine link scorer through ATL-25; `--features node2vec|metapath2vec --p --q`; four rules on §43.5's limits; gensim as an optional accelerator by dynamic import | Done; negative sampling in place of the softmax (the book's own advice); HARP, LINE, Struct2Vec, KG embeddings and alias sampling declared not built | ATL-43 |
| 44 | Message Passing & Graph Convolution | `gnn.py` (ATL-44 landed): GCN and GraphSAGE as plain functions over torch behind the `gnn` extra (CPU wheel from an explicit index, linux-gated, locked), entity-attribute completion trained on entity-owned labels only and printed as sidecar suggestions, a graph-autoencoder link scorer through ATL-25; `sna complete`; four §44.2 rules | Done; the extra is optional and the command exits 2 with the install hint without it; GraphSAGE cited to Hamilton et al. since the chapter does not name it; CLI wiring of the link scorer left for a follow-up | ATL-44 |
| 45 | Deep Graph Learning Models | `gnn.py` (ATL-45 landed): GAT under the same extra (single-head attention per Veličković et al., multi-head concat then mean), `--method gat --gat-heads N` through classification, embedding, link scoring and completion, the attention printed as which neighbours a suggestion leaned on; the §45.5 practical-considerations checklist in `docs/SNA_FOUNDATIONS.md`; two rules | Done; Doc: transformers and deep generative models (§45.2–45.3, §18.4) in the foundations doc, written by ATL-01; dropout and virtual-edge augmentation documented as not implemented; a single head can underfit, disclosed in the help text | ATL-45 |
| 46 | Graph Summarization | `summarize.py` (ATL-46 landed): aggregation into a meta-network by community or attribute with within-group density (Figure 46.2), a two-part MDL code with listed-pair corrections where the model's declaration saves bits, simplification by a ch. 14 centrality, a SPINE-style influence summary over a simulated cascade; the backboning distinction printed; `sna summarize` | Done; Grass's search, the compressor aggregation, OLAP slice/dice, spectrum-preserving reduction and GuruMine declared not built | ATL-46 |
| 47 | Node Vector Distance | `vectordist.py` (ATL-47 landed): Euclidean, cosine and correlation baselines, the generalized Euclidean via the Laplacian pseudoinverse, shortest-path linkage and an exact earth-mover LP, the graph Fourier distance ‖L(p−q)‖ as p. 692 prints it with the Dirichlet energy as a labelled diagnostic, network variance and a disclosed network correlation; a vector spec language (stance, attribute, centrality, window); `sna distance` | Done; the Markov-chain and annihilation routes and the multi-agent variants declared not built; the correlation is not mean-centred and says so | ATL-47 |
| 48 | Topological Distances | `topodist.py` (ATL-48 landed): spectral distance on the symmetric normalised Laplacian spectra, NetSimile, DeltaCon by fast belief propagation, portrait divergence, each with the size confound disclosed beside the number; identity alignment with a low-overlap note; the book's own fusion example; `sna compare --topology` | Done; real alignment and the SNF algorithm declared not built; DeltaCon and portrait divergence from their source papers since the chapter only cites them | ATL-48 |
| 49 | Visualization: node attributes | `draw.py` (ATL-49 landed): node radius linear in area after a quasi-log, Okabe-Ito categorical palette plus a neutral (nine colours) and a single-hue lightness-monotone sequential one, `--color community` with the ch. 36 evaluation, `--color attr:<key>` | Done; borders, shapes, invisible nodes, diverging gradients and pie nodes declared not built | ATL-49 |
| 50 | Visualization: edges | `draw.py` (ATL-49 landed): edge width and colour from one measure (weight, or edge betweenness under `--size betweenness`) at fixed transparency, arrowheads and two bent arcs for reciprocal pairs, network lifting as the fixed behaviour | Done; real bundling and ghost-edge routing declared not built | ATL-49 |
| 51 | Visualization: layouts | `draw.py` (ATL-49 landed): seeded force, circular and arc ordered by the colour rule, adjacency matrix block-diagonal by community, layered via ATL-33; SVG plus a self-contained HTML with tooltips, drag and a legend; `sna draw` | Done; hive plots, thumbnails, probabilistic layouts, revealing matrices and timelines declared not built; §51.5 as recipes | ATL-49 |
| 52 | Network Science Applications | `docs/SNA.md` `## Applications` (ATL-52 landed): the eight cases read against this corpus's own questions and the commands that answer them | Done | ATL-52 |
| 53 | Data & Tools | GraphML/JSON out | Build: `read_graph`, GEXF/edge-list/Pajek/CSV, igraph and graph-tool bridges; **legendary graphs as known-answer test fixtures** | ATL-53 |
| 54 | Glossary | `docs/SNA_GLOSSARY.md` (ATL-52 landed): every glossary term mapped to the command, function or rule, or to the reason it is not built; held to the book's own pages by a test that skips without the cached PDF | Done | ATL-52 |
| 55 | Abbreviations | `docs/SNA_GLOSSARY.md` (ATL-52 landed): the abbreviations mapped beside the terms | Done | ATL-52 |
| 56 | Bibliography | — | Doc: each report section cites the chapter it implements (`--cite`) | ATL-ENT-4 |

Enterprise surface, beyond the book's chapters but required by the goal:

| Ticket | Delivers |
|---|---|
| ATL-ENT-1 | Every `sna` command as an MCP tool returning the markdown report and the JSON payload |
| ATL-ENT-2 | Notebook API: `graphrag.sna` public facade, pandas frames for nodes/edges, `read_graph`, a Jupyter quickstart |
| ATL-ENT-3 | Scale: sparse matrices throughout, on-disk network cache, a timing budget per command on the largest persona, a benchmark test |
| ATL-ENT-4 | Reproducibility: every report carries method, parameters, seed, version, null model and the Atlas chapter; `--cite` |
| ATL-ENT-5 | The walkthrough: `docs/SNA_WALKTHROUGH.md` runs one technique per chapter over a real persona and reads the result the way the book would |

---

## Waves

Tickets inside a wave touch disjoint modules and run concurrently (three to five agents at a
time). A wave starts when the previous wave's *foundation* tickets are merged; its *report*
tickets can trail. Sizes are in agent-runs: **S** one run, **M** two to three, **L** four or more.

| Wave | Tickets | Why this order |
|---|---|---|
| 0 Foundations | ATL-03, ATL-08, ATL-19, ATL-16, ATL-53, ATL-06, ATL-07, ATL-01 | Everything later calls `stats`, `matrices`, `null`; directed and multilayer networks must exist before hierarchies and multilayer communities; legendary graphs give every later ticket a known answer |
| 1 The Hairball | ATL-26, ATL-27, ATL-28, ATL-29 | Upstream of every number the package prints; the book's own Part VIII |
| 2 Properties & centrality | ATL-09, ATL-10, ATL-11, ATL-12, ATL-14, ATL-15 | Parts III–IV; ATL-11 feeds ATL-26's random-walk projection and ATL-15's similarity |
| 3 Mesoscale | ATL-30, ATL-31, ATL-32, ATL-33, ATL-34 | Part IX; ATL-32 changes how every Louvain report is read |
| 4 Communities | ATL-35, ATL-36, ATL-37, ATL-38, ATL-39, ATL-40 | Part X; ATL-36 first, the others plug their partitions into its battery |
| 5 Prediction, spreading, robustness | ATL-23, ATL-24, ATL-25, ATL-20, ATL-22 | Parts VI–VII; ATL-25 before ATL-23 lands its report so predictions are never printed unevaluated |
| 6 Mining & learning | ATL-41, ATL-42, ATL-43, ATL-44, ATL-45 | Part XI; the GNN tickets are behind an optional extra and never block a wave |
| 7 Holistic | ATL-46, ATL-47, ATL-48 | Part XII; ATL-48 replaces `compare`'s summary-stat diff |
| 8 Surfaces | ATL-49, ATL-52, ATL-ENT-1..5 | Part XIII–XIV and the enterprise surface; last because they wrap everything above |

---

## Tickets

Each ticket names the book sections it implements, what exists, what to build, what "done"
means, which modules it may touch, and what it depends on. The agent reads the chapter first
(`scripts/atlas_chapter.py <n>`), cites section numbers in docstrings, adds a `guide.py` rule for
every caveat the book states, regenerates `docs/SNA.md` and the SNA skill from `sna guide`, and
reports what it deliberately did not build and why.

### Wave 0 — Foundations

#### ATL-01 — Foundations document
- **Book:** ch. 1 (all), §2.1–2.6, ch. 4, §5.1–5.6, §28.4, §13.5.
- **Build:** `docs/SNA_FOUNDATIONS.md`: how the book approaches complexity and how this package
  does; the notation this codebase uses for probability, statistics, ML and linear algebra,
  mapped to the book's; and the explicit "documented, not built" list with reasons (§2.8 and
  §28.4 alternatives to probability; §13.5 classic combinatorial problems; §45.2–45.3).
- **Done when:** every "Doc" row in the coverage matrix resolves to a section of this file; a
  unit test asserts each listed section heading exists.
- **Touches:** `docs/`. **Depends:** none. **Size:** S.

#### ATL-03 — Statistics primitives
- **Book:** ch. 3 (§3.1–3.5), §2.7 Markov processes (as the vocabulary ATL-11 uses).
- **Today:** p-values, z-scores and NMI are computed inline in `attributes.py` and `cluster.py`.
- **Build:** `sna/stats.py`: summary statistics that refuse to print a mean or variance for a
  fat-tailed sample without saying so (§3.1, the book's argument at p. 383); the distributions of
  §3.2 with fit and KS test; empirical, permutation and binomial p-values with multiple-test
  correction (§3.3); Pearson, Spearman, Kendall (§3.4); MI, NMI, variation of information (§3.5).
  Move the existing `_z`/`_mean`/`_std` and `compare_partitions` onto it.
- **Done when:** `attributes.py` and `cluster.py` import from it with no behaviour change (all
  existing tests pass unchanged); each primitive has a known-answer test.
- **Touches:** `sna/stats.py`, imports in `sna/attributes.py`, `sna/cluster.py`. **Depends:** none. **Size:** M.

#### ATL-08 — Matrices and linear algebra
- **Book:** ch. 8 (§8.1–8.4), §5.5–5.6.
- **Build:** `sna/matrices.py`: adjacency (dense and `scipy.sparse`), row- and column-stochastic
  (§8.2), incidence for the bipartite networks (§8.3), combinatorial and symmetric-normalised
  Laplacian (§8.4), top-k eigenpairs and SVD/NMF helpers (§5.5–5.6), all under **one node-order
  contract** (sorted node ids, returned alongside the matrix) so every consumer agrees.
- **Done when:** `spectral_embedding` and `_eigenvector` use it; the Laplacian's smallest
  eigenvalue is 0 with multiplicity = components on the legendary graphs.
- **Touches:** `sna/matrices.py`, `sna/cluster.py`, `sna/measures.py`. **Depends:** none. **Size:** M.

#### ATL-19 — Null models and significance
- **Book:** ch. 19 (§19.1–19.2), §18.1 configuration model, §16.1 random graphs.
- **Today:** `null_model_modularity` and `_fixed_partition_null` each own a `double_edge_swap`
  loop; label permutation lives in `attributes.py`.
- **Build:** `sna/null.py`: named nulls with one interface — `erdos_renyi`, `configuration`
  (degree-preserving, the existing swap), `label_permutation`, and **`bipartite_preserving`**
  (curveball / fixed-degree-sequence swaps on the incidence matrix, so a projected network's null
  keeps the projection's own structure — the null this repo's borrowed-label confound needs);
  `significance(observed, samples)` returning z, empirical p and the sample summary via ATL-03;
  ERGM by maximum pseudo-likelihood over dyad features (§19.2) with the book's warnings on
  degeneracy printed with every fit.
- **Done when:** both existing nulls call it unchanged in result; `analyse_attribute` gains
  `--null bipartite` and reports it beside the permutation null.
- **Touches:** `sna/null.py`, `sna/cluster.py`, `sna/attributes.py`. **Depends:** ATL-03, ATL-08. **Size:** L.

#### ATL-16 — Synthetic graphs and "observed versus random"
- **Book:** ch. 16 (§16.1–16.5), ch. 17 (§17.1–17.3), §18.2–18.3.
- **Build:** `sna/generators.py`: ER, Watts–Strogatz, Barabási–Albert, configuration,
  planted-partition/LFR-style, random geometric — thin, seeded wrappers over networkx with the
  book's expected properties as docstrings. Then an **"Against random" section** in every
  `analyze` report: the observed clustering, average path length and degree distribution against
  their ER and configuration expectations (ch. 17), so "this network is clustered" is always a
  comparison.
- **Done when:** the section appears in `render_markdown` and `to_payload`; a test shows a WS
  graph scoring high and an ER graph scoring near its expectation.
- **Touches:** `sna/generators.py`, `sna/analysis.py`. **Depends:** ATL-19. **Size:** M.

#### ATL-53 — Interoperability and the legendary graphs
- **Book:** ch. 53 (§53.1–53.4).
- **Today:** GraphML and node-link JSON out; nothing reads a network back.
- **Build:** `read_graph` for both formats plus GEXF, weighted edge list, Pajek `.net`, CSV
  nodes/edges (§53.2 software: Gephi, Cytoscape, Pajek); optional bridges to igraph and
  graph-tool that convert without either being a dependency (§53.1); and `tests/legendary.py`
  (§53.4): karate club, Les Misérables, dolphins, football, Southern women (bipartite), with the
  published answers each later ticket asserts against.
- **Done when:** a graph round-trips through every format with attributes and provenance intact;
  Louvain on karate club recovers the two factions within the known modularity.
- **Touches:** `sna/export.py`, `tests/legendary.py`. **Depends:** none. **Size:** M.

#### ATL-06 — Directed and typed networks
- **Book:** ch. 6 (§6.1–6.4), §10.3 reciprocity.
- **Today:** every network is an undirected weighted `nx.Graph`; the `RELATED_TO` relations the
  extraction layer writes are never exported as a network.
- **Build:** a fifth network, `relations`: directed, edge `type` from the relation, weight = how
  many passages state it, `--relation-type` filter; `describe(graph)` naming the network type per
  §6.4 (simple, directed, weighted, bipartite, multilayer, dynamic); every function in `measures`
  and `cluster` declares directed support or refuses with the reason (a directed graph handed to
  Louvain is flattened *and says so*). Reciprocity in `summary`.
- **Done when:** `sna export --network relations` writes a `DiGraph`; `analyze` runs on it with
  in/out centralities; `guide.py` gains a network rule for it.
- **Touches:** `sna/export.py`, `sna/measures.py`, `sna/guide.py`, `graph/store.py` (a read for relations by persona). **Depends:** none. **Size:** M.

#### ATL-07 — Multilayer, hypergraph and dynamic representations
- **Book:** ch. 7 (§7.2–7.4); §7.1 and §7.5 exist.
- **Build:** `sna/layers.py`: a **multilayer** network whose layers are one of stance, facet,
  source or relation type (`--layers stance`), with flattening and supra-adjacency (§7.2); a
  **hypergraph** view keeping each passage as a hyperedge over the entities it names rather than
  its clique projection (§7.3), exposed as the incidence matrix ATL-08 defines; a **dynamic**
  edge list with timestamps from the SPOKE dates (§7.4), with `snapshots(window, step)` that
  generalises what `compare` builds by hand.
- **Done when:** `compare_windows` is re-expressed over `snapshots` with identical output;
  the speakers–entities network can be built per layer.
- **Touches:** `sna/layers.py`, `sna/export.py`, `sna/compare.py`. **Depends:** ATL-08. **Size:** L.

### Wave 1 — The Hairball

#### ATL-26 — Bipartite projections
- **Book:** ch. 26 (§26.1–26.6).
- **Today:** `bipartite_projection` weight = |Nu ∩ Nv| (§26.1); a docstring notes the inflation.
- **Build:** `--projection simple|jaccard|hyperbolic|resource|probs|heats|hybrid|randomwalk`
  on every projected network, implemented over the incidence matrix (§26.2–26.5; hybrid takes
  `--lambda`); a `sna projections` command implementing §26.6: run every scheme on one network
  and report edge-weight distributions, rank correlation between schemes (ATL-03) and the share
  of edges each would keep at a given backbone. Default stays `simple` until the walkthrough
  compares them on a real persona; the report names the scheme in the frame.
- **Done when:** on Southern women (ATL-53) the hyperbolic and resource-allocation weights match
  hand-computed values; `analyze --projection hyperbolic` runs end to end.
- **Touches:** `sna/export.py`, `sna/projection.py`, `cli.py`, `sna/guide.py`. **Depends:** ATL-08, ATL-03. **Size:** M.

#### ATL-27 — Network backboning
- **Book:** ch. 27 (§27.1–27.6), §13.4 spanning trees and PMFG.
- **Today:** `_prune(min_weight)` is §27.1, undefended.
- **Build:** `sna/backbone.py` with `naive`, `doubly-stochastic`, `high-salience`, `convex`,
  `disparity` and `noise-corrected` (binomial null, edge-specific p-value; the book's
  recommendation for count weights and its warning that the disparity filter manufactures
  hub-and-spoke structure and weak communities, p. 391), plus maximum spanning tree and PMFG as
  structural backbones; `--backbone <name> --alpha 0.05` on every network; a `sna backbone`
  command reporting what each method keeps. `--min-weight` stays, and its help text carries the
  book's critique (p. 383).
- **Done when:** the noise-corrected backbone of the entity network keeps a p-value column on
  every surviving edge; a guide rule says when to prefer NC over DF; a test shows DF creating the
  hub-spoke bias on a planted graph and NC not.
- **Touches:** `sna/backbone.py`, `sna/export.py`, `cli.py`, `sna/guide.py`. **Depends:** ATL-03, ATL-08. **Size:** L.

#### ATL-28 — Uncertainty and measurement error
- **Book:** ch. 28 (§28.1–28.3); §28.4 documented in ATL-01.
- **Build:** edge probabilities derived from the layers' own evidence — mention tier
  (exact/loose/fallback), number of supporting passages, annotation agreement — stored as an edge
  attribute `p`; `sna/uncertain.py` computing any measure's expectation and interval by sampling
  edge realisations (§28.2), with the specific closed forms the book gives where they exist
  (§28.3: expected degree, probabilistic components); error estimation and correction (§28.1) as
  a report section that says how many edges rest on a single loose match.
- **Done when:** `analyze --uncertain` prints centralities as mean ± interval; a loose-only edge
  has p below an exact-match edge.
- **Touches:** `sna/uncertain.py`, `sna/export.py`, `sna/analysis.py`, `graph/store.py` (expose mention tier). **Depends:** ATL-19. **Size:** L.

#### ATL-29 — Network sampling
- **Book:** ch. 29 (§29.1–29.5).
- **Build:** `sna sample --method induced|bfs|snowball|forest-fire|random-walk --size n`, each
  with the bias the book attributes to it printed in the frame (§29.4); a completion estimate
  (§29.5) — how much of the network the sample likely missed — beside every sampled report.
- **Done when:** sampling a legendary graph reproduces the degree-bias direction the book states
  for each method.
- **Touches:** `sna/sampling.py`, `cli.py`, `sna/guide.py`. **Depends:** ATL-03. **Size:** M.

### Wave 2 — Properties and centrality

#### ATL-09 — Degree
- **Book:** ch. 9 (§9.1–9.4).
- **Build:** `sna degree` report: in/out/weighted variants (§9.1), histogram and CCDF (§9.2),
  power-law fit with the Clauset–Shalizi–Newman procedure and a likelihood-ratio comparison
  against lognormal and exponential (§9.3–9.4); the report refuses "scale-free" unless the test
  supports it, per the book.
- **Touches:** `sna/degree.py`, `cli.py`, `sna/guide.py`. **Depends:** ATL-03. **Size:** M.

#### ATL-10 — Paths and walks
- **Book:** ch. 10 (§10.1–10.4), §13.1–13.3.
- **Build:** walks via matrix powers (§10.1), cycle basis and triangle counts (§10.2), reciprocity
  on directed networks (§10.3, with ATL-06), SCC/WCC (§10.4), BFS/DFS exploration and the
  path-length distribution and diameter (§13.1–13.3) in `summary` and the report.
- **Touches:** `sna/paths.py`, `sna/measures.py`. **Depends:** ATL-06, ATL-08. **Size:** M.

#### ATL-11 — Random walks
- **Book:** ch. 11 (§11.1–11.6), §2.7.
- **Build:** `sna/walks.py`: stationary distribution (§11.1), non-backtracking operator (§11.2),
  hitting and commute times (§11.3), effective resistance (§11.4), min-cut/max-flow between two
  nodes (§11.5), consensus dynamics (§11.6); exposed as node-pair similarities for ATL-15 and as
  the random-walk projection in ATL-26.
- **Touches:** `sna/walks.py`. **Depends:** ATL-08. **Size:** M.

#### ATL-12 — Density
- **Book:** ch. 12 (§12.1–12.4).
- **Build:** local and global clustering and transitivity, weighted per Onnela (§12.2); maximal
  cliques and k-clique listing (§12.3); a maximal independent set (§12.4); the density-vs-size
  reading of §12.1 as a rule.
- **Touches:** `sna/measures.py`, `sna/guide.py`. **Depends:** none. **Size:** S.

#### ATL-14 — Node ranking
- **Book:** ch. 14 (§14.3, §14.5–14.8); §14.1, 14.2, 14.4 exist.
- **Build:** reach (§14.3), HITS (§14.5), harmonic (§14.6), k-core and k-truss (§14.7),
  Freeman centralization for every centrality (§14.8); in/out variants on directed networks;
  a ranking-stability check (rank correlation across bootstrap resamples or backbones, ATL-03).
- **Touches:** `sna/measures.py`, `sna/analysis.py`, `sna/guide.py`. **Depends:** ATL-06, ATL-03. **Size:** M.

#### ATL-15 — Node roles
- **Book:** ch. 15 (§15.1–15.3).
- **Build:** Guimerà–Amaral roles from participation coefficient and within-module degree z
  (§15.1, extends `brokers`); structural equivalence (Jaccard, cosine, Pearson on adjacency
  rows), SimRank, and regular equivalence via blockmodel positions (§15.2); §15.3 delegates to
  ATL-42/43. A `sna roles` section names which entities play the same role without ever
  co-occurring.
- **Touches:** `sna/roles.py`, `sna/analysis.py`. **Depends:** ATL-08, ATL-11. **Size:** M.

### Wave 3 — Mesoscale

#### ATL-30 — Homophily
- **Book:** ch. 30 (§30.1–30.4); §30.2 partly exists.
- **Build:** `sna ego <node>` (§30.1); EI index and Coleman homophily per value (§30.2); strength
  of weak ties — tie strength versus neighbourhood overlap, with the Granovetter/Onnela reading
  (§30.3); the homophily-versus-contagion caveat (§30.4) as a rule that names what temporal data
  (ATL-07) would be needed to separate them.
- **Touches:** `sna/attributes.py`, `sna/ego.py`, `sna/guide.py`. **Depends:** ATL-07. **Size:** M.

#### ATL-31 — Quantitative assortativity
- **Book:** ch. 31 (§31.1–31.3).
- **Today:** `--by` is categorical; degree assortativity is a single number in `summary`.
- **Build:** numeric attributes in the vocabulary (`facets.yaml`: `type: number`, validated by
  `AttributeTable`); `--by` on a numeric key → Pearson and Spearman over edge endpoints with a
  permutation null (§31.1), the neighbour-average curve with a log-log fit exponent (§31.1),
  friendship-paradox statistics (§31.2), the distribution of the attribute against degree
  (§31.3); the same for built-in numeric node data (`mentions`, `documents`, `chunks`).
- **Touches:** `extract/attributes.py`, `sna/attributes.py`, `sna/guide.py`. **Depends:** ATL-03. **Size:** M.

#### ATL-32 — Core-periphery
- **Book:** ch. 32 (§32.1–32.4).
- **Build:** discrete (Borgatti–Everett) and continuous coreness (§32.1) with the correlation
  quality measure; rich-club coefficient against the configuration null (§32.1); nestedness
  (NODF) on the bipartite networks (§32.4); and the **tension check** (§32.2) in every Louvain
  report: when coreness explains the edges better than the partition does, the report says the
  network is core-periphery and the communities are its periphery's tail.
- **Touches:** `sna/coreperiphery.py`, `sna/analysis.py`, `sna/guide.py`. **Depends:** ATL-19, ATL-08. **Size:** L.

#### ATL-33 — Hierarchies
- **Book:** ch. 33 (§33.1–33.6).
- **Build:** on the `relations` network (ATL-06): hierarchy type classification (§33.1), cycle
  detection and the share of edges in cycles (§33.2), global reach centrality (§33.3), maximum
  spanning arborescence (§33.4), agony (§33.5), and a layered drawing export (§33.6, consumed by
  ATL-49).
- **Touches:** `sna/hierarchy.py`, `cli.py`. **Depends:** ATL-06. **Size:** M.

#### ATL-34 — High-order dynamics
- **Book:** ch. 34 (§34.1–34.3).
- **Build:** simplicial complex from passages (each passage's entity set is a simplex), higher-
  order degree and clustering, simplicial closure (§34.1); a **second-order network** from
  consecutive passages within a document — "A then B" transitions with memory (§34.2) — and a
  memory-aware random walk over it (§34.3, with ATL-11).
- **Touches:** `sna/highorder.py`, `sna/layers.py`. **Depends:** ATL-07, ATL-11. **Size:** L.

### Wave 4 — Communities

#### ATL-36 — Community evaluation (first in the wave)
- **Book:** ch. 36 (§36.1–36.4); §36.1 and §36.4 exist.
- **Build:** `sna/evaluate.py`: conductance, coverage, performance, expansion (§36.2, via
  `nx.community.partition_quality` where it exists), link-prediction-based evaluation (§36.3,
  with ATL-25), the resolution-limit check (§36.1: any community smaller than √(2m) is flagged);
  the battery runs for every partition any method produces.
- **Touches:** `sna/evaluate.py`, `sna/analysis.py`. **Depends:** ATL-03. **Size:** M.

#### ATL-35 — Graph partitions
- **Book:** ch. 35 (§35.1–35.5); §35.6 exists.
- **Build:** `--method sbm` (degree-corrected SBM: `graph-tool` when installed as an optional
  extra, otherwise a spectral-initialised greedy likelihood fit, and the report says which),
  `--method infomap|walktrap` (§35.2), `--method label-propagation` (§35.3), temporal
  communities by matching partitions across ATL-07 snapshots (§35.4), local seed-set expansion
  (§35.5, `sna community --seed <node>`).
- **Touches:** `sna/cluster.py`, `sna/analysis.py`, `pyproject.toml` (optional extra). **Depends:** ATL-36, ATL-07. **Size:** L.

#### ATL-37 — Hierarchical community discovery
- **Book:** ch. 37 (§37.1–37.4).
- **Build:** Louvain levels and Girvan–Newman as dendrograms with a cut chooser (§37.1), HRG fit
  (§37.2), directed community detection on `relations` (§37.3), the density-versus-hierarchy
  reading (§37.4) as a rule.
- **Touches:** `sna/cluster.py`. **Depends:** ATL-36, ATL-06. **Size:** M.

#### ATL-38 — Overlapping coverage
- **Book:** ch. 38 (§38.1–38.7).
- **Build:** k-clique percolation (§38.3), link clustering (§38.5), ego-splitting (§38.2);
  overlapping NMI (§38.1); the overlap paradox (§38.7) as a rule.
- **Touches:** `sna/overlap.py`. **Depends:** ATL-36. **Size:** M.

#### ATL-39 — Bipartite community discovery
- **Book:** ch. 39 (§39.1–39.4).
- **Today:** `speakers-entities` is projected before it is grouped (§39.2).
- **Build:** Barber bipartite modularity and BRIM/biLouvain directly on the two-mode network
  (§39.3), neighbour-similarity grouping (§39.4), bipartite evaluation (§39.1); `analyze
  --network speakers-entities` without `--project` groups both modes together.
- **Touches:** `sna/bipartite.py`, `sna/analysis.py`. **Depends:** ATL-36, ATL-08. **Size:** M.

#### ATL-40 — Multilayer community discovery
- **Book:** ch. 40 (§40.1–40.5).
- **Build:** flattening (§40.1), layer-by-layer with cross-layer matching (§40.2), multilayer
  modularity over the supra-adjacency with `--omega` (§40.3), multilayer density (§40.4), the
  §40.5 rules on which to use.
- **Touches:** `sna/multilayer.py`. **Depends:** ATL-07, ATL-36. **Size:** L.

### Wave 5 — Prediction, spreading, robustness

#### ATL-25 — Experiment design (first in the wave)
- **Book:** ch. 25 (§25.1–25.2).
- **Build:** temporal holdout (later windows held out, ATL-07) and random edge holdout with
  negative sampling (§25.1); AUC, precision@k, and the book's cautions on class imbalance and on
  reporting only AUC as rules (§25.2). Every prediction command reports through this.
- **Touches:** `sna/experiment.py`, `sna/guide.py`. **Depends:** ATL-07, ATL-03. **Size:** M.

#### ATL-23 — Link prediction on simple graphs
- **Book:** ch. 23 (§23.1–23.7).
- **Build:** `sna predict`: preferential attachment, common neighbours, Adamic–Adar, resource
  allocation, Jaccard, Katz, HRG-based (§23.1–23.5, §23.7), association rules over passages as
  transactions (§23.6, shared with ATL-41). Output is a ranked list of **hypotheses with the
  passages that would confirm them**; nothing is ever written to the graph, and the report says
  so in its frame.
- **Touches:** `sna/predict.py`, `cli.py`. **Depends:** ATL-25. **Size:** M.

#### ATL-24 — Signed and multilayer link prediction
- **Book:** ch. 24 (§24.1–24.2).
- **Build:** signed prediction over praise/complaint from the stance layer using balance theory
  (§24.1, on `stances.py`'s signed pairs); generalized multilayer prediction over ATL-07 layers
  (§24.2).
- **Touches:** `sna/predict.py`, `sna/stances.py`. **Depends:** ATL-23, ATL-07. **Size:** M.

#### ATL-20 — Spreading processes
- **Book:** ch. 20 (§20.1–20.3), ch. 21 (§21.1–21.4).
- **Build:** `sna spread --model si|sis|sir|threshold --seeds ... --beta --mu`: simulations
  with confidence bands over runs, the epidemic threshold against the spectral radius (ATL-08),
  complex-contagion triggers and limited infection chances (§21.1–21.2), intervention strategies
  (random, targeted, acquaintance immunisation, §21.3), driver nodes by maximum matching (§21.4).
  The frame states what "spreading" means on a co-mention network: a what-if, not an observation.
- **Touches:** `sna/spread.py`, `cli.py`, `sna/guide.py`. **Depends:** ATL-08, ATL-16. **Size:** L.

#### ATL-22 — Catastrophic failures
- **Book:** ch. 22 (§22.1–22.4).
- **Build:** `sna robustness`: giant-component curves under random and targeted removal by each
  centrality (§22.1–22.2), cascade/chain models (§22.3), interdependent networks by coupling two
  of this persona's networks (§22.4).
- **Touches:** `sna/robustness.py`, `cli.py`. **Depends:** ATL-14. **Size:** M.

### Wave 6 — Mining and learning

#### ATL-41 — Frequent subgraph mining and motifs
- **Book:** ch. 41 (§41.1–41.5).
- **Build:** triadic census and motif z-scores against the configuration null (§41.2, ATL-19),
  VF2 isomorphism and canonical labelling (§41.3), transactional mining over per-document entity
  graphs (§41.4, a gSpan-style enumerator bounded by size), single-graph frequent subgraphs
  (§41.5).
- **Touches:** `sna/motifs.py`. **Depends:** ATL-19. **Size:** L.

#### ATL-42 — Shallow graph learning
- **Book:** ch. 42 (§42.1–42.4), ch. 4.
- **Build:** the embedding contract (`Embedding` with node order, dims, method, provenance),
  spectral embedding moved onto ATL-08, pooling (§42.4) for graph-level vectors used by ATL-48;
  `docs/SNA_FOUNDATIONS.md` gains the ch. 4 notation section.
- **Touches:** `sna/embed.py`, `sna/cluster.py`. **Depends:** ATL-08. **Size:** S.

#### ATL-43 — Random-walk embeddings
- **Book:** ch. 43 (§43.1–43.5).
- **Build:** DeepWalk and node2vec (`--features node2vec --p --q`) with a numpy skip-gram (no
  new mandatory dependency; `gensim` as an optional accelerator), metapath2vec on the two-mode
  network (§43.3), their use in clustering and prediction (§43.4), the limitations of §43.5 as
  rules.
- **Touches:** `sna/embed.py`, `sna/analysis.py`. **Depends:** ATL-42, ATL-11. **Size:** L.

#### ATL-44 — Message passing and graph convolution (optional extra)
- **Book:** ch. 44 (§44.1–44.4).
- **Build:** `graphrag[gnn]` extra (torch, CPU); GCN and GraphSAGE for **entity-attribute
  completion** — semi-supervised classification from structure plus text embeddings, reported
  as suggestions for the extraction sidecars, never written to the graph — and for link
  prediction through ATL-25; the limits of message passing (§44.2) as rules. Absent the extra,
  the commands explain how to install it and exit 2.
- **Touches:** `sna/gnn.py`, `pyproject.toml`. **Depends:** ATL-42, ATL-25. **Size:** L.

#### ATL-45 — Deep graph learning models
- **Book:** ch. 45 (§45.1–45.5), §18.4.
- **Build:** GAT under the same extra (§45.1); §45.5 practical considerations as the acceptance
  checklist of ATL-44; **Doc** for §45.2–45.3 (transformers, deep generative models) with the
  reason: nothing in a corpus network of this size is served by them yet, and §18.4 generative
  models are covered for nulls by ATL-16.
- **Touches:** `sna/gnn.py`, `docs/SNA_FOUNDATIONS.md`. **Depends:** ATL-44. **Size:** M.

### Wave 7 — Holistic

#### ATL-46 — Graph summarization
- **Book:** ch. 46 (§46.1–46.4).
- **Build:** `sna summarize --by community|attr:<key>`: aggregate into a meta-network with
  edge weights and within-group density (§46.1), MDL-style compression (§46.2), simplification
  by importance (§46.3), influence-based summaries (§46.4); the book's distinction from
  backboning (p. 381–382) in the frame.
- **Touches:** `sna/summarize.py`, `cli.py`. **Depends:** ATL-36. **Size:** M.

#### ATL-47 — Node vector distance
- **Book:** ch. 47 (§47.1–47.5).
- **Build:** `sna distance --vectors a b`: compare two node vectors on one network (praise vs
  complaint counts, north vs south mentions, two windows' degrees) — Euclidean and correlation
  baselines (§47.1), the generalized Euclidean via the Laplacian pseudoinverse (§47.2),
  shortest-path/earth-mover (§47.3), graph Fourier transform smoothness (§47.4), network
  variance and network correlation (§47.5).
- **Touches:** `sna/vectordist.py`, `cli.py`. **Depends:** ATL-08. **Size:** M.

#### ATL-48 — Topological distances
- **Book:** ch. 48 (§48.1–48.3).
- **Build:** network similarity — spectral distance, NetSimile, DeltaCon, portrait divergence
  (§48.1) — as the structural half of `compare`; alignment by node id with a note on when ids
  are not enough (§48.2); similarity-network fusion across a persona's networks (§48.3).
- **Touches:** `sna/topodist.py`, `sna/compare.py`. **Depends:** ATL-42. **Size:** M.

### Wave 8 — Surfaces

#### ATL-49 — Visualization
- **Book:** ch. 49 (§49.1–49.3), ch. 50 (§50.1–50.3), ch. 51 (§51.1–51.5).
- **Build:** `sna draw --layout force|circular|arc|matrix|layered --size <centrality> --color
  <community|attr:key>` writing SVG and a self-contained interactive HTML; size and colour rules
  with colourblind-safe categorical and sequential palettes and the book's cautions (ch. 49);
  edge width/colour/bundling and network lifting (ch. 50); seeded force-directed, circular/arc,
  adjacency-matrix and layered layouts (ch. 51); §51.5 case-study patterns as documented recipes.
- **Touches:** `sna/draw.py`, `cli.py`. **Depends:** ATL-14, ATL-33. **Size:** L.

#### ATL-52 — Applications, glossary, Atlas index
- **Done (2026-09-17):** `scripts/atlas_index.py` generates `docs/ATLAS_INDEX.md` (one row per chapter
  and section, credited from the modules' `§` citations, the guide's rules and the foundations doc),
  held verbatim by a test; `docs/SNA_GLOSSARY.md`; the applications section in `docs/SNA.md`.
- **Book:** ch. 52, ch. 54, ch. 55.
- **Build:** `docs/SNA_GLOSSARY.md` — every term in the book's glossary mapped to the command,
  flag or function that answers it, and "not built, see ATL-01" where that is the answer;
  `sna guide --atlas` printing chapter → command; `docs/SNA_APPLICATIONS.md` mapping each §52
  application to a corpus question and the commands (digital humanities and polarization map
  directly onto stance and attribute networks; science of science onto speakers).
- **Touches:** `sna/guide.py`, `docs/`. **Depends:** everything above landed. **Size:** M.

#### ATL-ENT-1 — MCP tools
- **Done (2026-09-17):** every `sna` command is an MCP tool `sna_<command>` in `mcp_server.py`
  returning `{"markdown", "payload"}` through the functions the CLI calls; a completeness test holds
  the tool set to `sna_app`'s commands (maintenance groups excluded); the SNA skill lists the tools.
- **Build:** every `sna` command as an MCP tool in `mcp_server.py`, returning the markdown and
  the JSON payload; the persona skills learn the new tools.
- **Depends:** each wave as it lands (incremental). **Size:** M per wave.

#### ATL-ENT-2 — Notebook API
- **Done (2026-09-17):** `sna/facade.py`'s `Network` (`build`, `from_graph`, `analyze`, `backbone`,
  `measures`, `evaluate`, `predict`, `draw`, `to_pandas`), each calling the function the CLI calls;
  `read_graph` via ATL-53; `docs/SNA_NOTEBOOK.md`. pandas stays optional (records without it).
- **Build:** a stable `graphrag.sna` facade (`Network.build(...)`, `.backbone()`, `.analyze()`,
  `.to_pandas()`), `read_graph`, a Jupyter quickstart under `docs/`.
- **Depends:** ATL-53, wave 1. **Size:** M.

#### ATL-ENT-3 — Scale
- **Done (2026-09-17):** `eigenpairs(sparse=True)` by `eigsh` above `SPARSE_ABOVE_NODES`, the walk
  quantities routed sparse; `sna/cache.py`'s per-persona on-disk network cache keyed by every filter,
  the snapshot identity and a store fingerprint, cleared by every write path, `--no-cache` on every
  command, `sna cache clear`; the `slow` marker, `make bench`, `tests/benchmarks/test_sna_budget.py`.
- **Build:** sparse matrices where ATL-08 allows, an on-disk network cache keyed by filters and
  snapshot commit, a timing budget per command on the largest persona (13k chunks), a benchmark
  test marked `slow`.
- **Depends:** ATL-08. **Size:** M.

#### ATL-ENT-4 — Reproducibility and citation
- **Done (2026-09-17):** `sna/provenance.py`; every report ends with `## Provenance` and every payload
  carries `provenance` (method, parameters, seed, tool version, null model, chapters, snapshot commit),
  built once from the caller's own arguments; `--cite` appends the references; a test calls every
  tool and fails if the block is missing. `sna predict --json` is now a keyed object.
- **Build:** a provenance block in every report and payload — method, parameters, seed, tool
  version, null model, Atlas chapter — and `--cite` printing the book reference per section.
- **Depends:** ATL-01. **Size:** S.

#### ATL-ENT-5 — The walkthrough
- **Done (2026-09-17):** `docs/SNA_WALKTHROUGH.md`, one technique per chapter run on `product-leader`
  and on private corpora where a chapter needs other data or size, every number from a saved
  payload, the negative results read as the book would; three review passes verified 21 chapters
  against their payloads and reran three commands to the digit. The walkthrough and its payloads
  quote those private corpora, so this repository does not carry them.
- **Build:** `docs/SNA_WALKTHROUGH.md`: one technique per chapter run on a real persona, the
  output read the way the book would read it, including the negative results. This is the
  acceptance test for the programme as a whole.
- **Depends:** everything. **Size:** L.

---

## Completion and follow-ups

All 49 tickets landed on 2026-09-17 in the private working repository and were ported here
together; the session checkpoint that tracked the in-flight work was folded into this section
and deleted. The walkthrough (ATL-ENT-5) is the acceptance test the definition of done asks for;
it stays with the private corpora it was run on.

What the walkthrough found on this corpus, recorded as results rather than defects: balance-theory
sign prediction scores below chance on the small signed layer (ch. 24); BRIM's two-mode partition
sits below its own curveball null (ch. 39); summarising by the 75-community grouping does not
compress the network (ch. 46); the size guards for roles, walks, distance, robustness, spreading,
prediction and metapath2vec all fire on `product-leader`'s 7,000-node entity network.

Two follow-up tickets the programme surfaced, outside the original 49:

#### ATL-F1 — One surface: CLI and MCP arguments, `--out` and `--json`
- **Done (2026-09-18):** `sna/arguments.py` parses `types`, `where` and sample parameters once for
  both surfaces and the provenance records the canonical shape; every report command has optional
  `--out` and `--json` (sample, export and draw keep one required `--out` for their one artifact);
  shared payload builders replace the inline dicts; a parity test and a reflection test hold it.
- **Build:** the CLI and the MCP tools accept `types`, `where` and `sna sample`'s parameters in
  different shapes (a comma string and repeated `key=value` on the CLI, a list and a dict on the
  tools), so their provenance blocks differ in content for those calls; and `--out`/`--json` are
  inconsistent across `sna` commands (`backbone` and `layers` take no `--json`, `walks` and
  `distance` take neither, `analyze` requires `--out`). Give every report command the same
  contract: `--out <md>` and `--json <path>` both optional, console otherwise; and make the MCP
  tools and the CLI parse their filter arguments through one shared function so the provenance
  parameters agree byte for byte. The MCP parity tests hold it.
- **Touches:** `cli.py`, `mcp_server.py`, `sna/provenance.py`, the walkthrough's command blocks.
  **Depends:** ATL-ENT-1, ATL-ENT-4. **Size:** M.

#### ATL-F2 — A budget for backboning and uncertainty at scale
- **Done (2026-09-18):** high-salience above `SALIENCE_SAMPLE_ABOVE_NODES` draws a degree-stratified
  source sample and takes the weighted stratified mean (exactly unbiased, the reviewer's two-band
  proof), stating the sample beside the scores; the doubly-stochastic scores run sparse above the
  threshold; `--uncertain` samples the cheap centralities by default, `--uncertain-expensive` opts
  the rest in, `--max-seconds` stops after the realisation that crosses the budget and reports what
  finished; four benchmark rows. Both follow-ups are now landed and the programme is closed.
- **Build:** `sna backbone`'s eight-method comparison and `sna analyze --uncertain` with 200
  samples each ran over two hours on the 7,000-node entity network. Give both a stated budget:
  high-salience by a sampled set of sources above a node threshold (§27.3's own suggestion, with
  the sample size printed), the other methods on sparse matrices; the uncertainty realisations
  computed on the cheap centralities by default with the expensive ones opt-in, and a
  `--max-seconds`-style budget that stops and reports what it finished rather than hanging. The
  benchmark test gains rows for both.
- **Touches:** `sna/backbone.py`, `sna/uncertain.py`, `cli.py`, `tests/benchmarks/`. **Depends:**
  ATL-27, ATL-28, ATL-ENT-3. **Size:** M.

## Definition of done for the programme

1. Every row of the coverage matrix is either a merged ticket or a section of
   `docs/SNA_FOUNDATIONS.md` with a stated reason.
2. `make check` and `make test-integration` green; no new mandatory dependency without a note in
   this file saying why the book's method needed it.
3. Every report section cites its chapter; every rule the book states as a caveat is a
   `guide.py` rule quoted in the docs and the skill.
4. Every known-answer test on the legendary graphs passes.
5. The walkthrough exists and was produced by running the commands, not by writing prose about
   them (kept with the private corpora it quotes; see ATL-ENT-5).
6. A network scientist can reach every technique three ways — CLI, MCP tool, notebook — with the
   same numbers from each.

---

## Orchestration

**Roles.**

- **Lead** (the session that owns this file): sequences waves, fills one brief per ticket from
  `docs/templates/ATLAS_TICKET.md`, launches implementers, routes each finished ticket to a
  reviewer, merges, runs the full check, pushes. Subagents cannot push (the `PreToolUse` guard
  denies it); the lead is the only writer to `origin`.
- **Implementer** (`atlas-implementer`, Opus): one ticket, one worktree, one branch
  `atlas/<ticket-id>`. Reads the chapter first. Delivers code, tests, docs, guide rules, and a
  closing report in the ticket's format: what was built, what was deliberately not built and why,
  what the book says that the code could not honour, and the exact commands run to check.
- **Reviewer** (`atlas-reviewer`, Opus): holds the diff to the chapter text and to this repo's
  rules; verifies the known-answer tests are real answers, that the caveats became rules, that
  nothing was mocked, and that the report reads honestly. Returns *merge*, *fix* (with the list),
  or *reject* (with the reason). Never edits.

**Ticket lifecycle.** brief → implement (worktree) → self-check (`uv run ruff format --check src
tests && uv run ruff check src tests && uv run mypy && uv run pytest -q -m "not integration and
not embedding"`, the same four steps as `make check-local`) → closing report → review → lead
merges onto the programme branch → lead runs the full check → push. A ticket that fails review
goes back to the *same* implementer with the reviewer's list; a rejected ticket is re-briefed.

**Concurrency.** Up to five implementers at a time, always from one wave, never two tickets that
name the same module in "Touches". Report-wiring tickets (`analysis.py`, `cli.py`) are serialised
within a wave to avoid merge conflicts in the two files every ticket wants.

**Programme branch.** Work lands on `claude/attribute-layer-entity-nodes-0mav74` (the branch this
plan was written on) until the user names another; each merged ticket is one commit whose message
opens with its id.

**What the lead never delegates.** Changing a default that moves numbers in existing reports
(projection scheme, `min_weight`, the confound threshold); adding a mandatory dependency; editing
`guide.py` rules another ticket owns; pushing.
