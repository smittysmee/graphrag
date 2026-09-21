# Glossary

Every term in chapter 54 of *The Atlas for the Aspiring Network Scientist* (pp. 789-798,
`scripts/atlas_chapter.py 54`), mapped to the command, flag or function that answers it in this
codebase, or to "not built" with the reason. Chapter 55's abbreviations (pp. 799-802) fold in at
the end, per `docs/ATLAS_PLAN.md`'s row for both chapters. Unlike `docs/ATLAS_INDEX.md`, this file
is hand-written: matching a glossary term to the right function is a judgment call a `§n.m`
citation cannot make for you, and the book's own definition is paraphrased here, not copied whole.
`tests/unit/test_sna_glossary.py` reads the book's own glossary pages and asserts every term it
lists appears somewhere below, so a chapter reprint that adds a term fails the test rather than
going unnoticed.

A "not built" row still says why: either the concept does not apply to a co-occurrence network
(this package's five networks are described in `docs/SNA.md`'s Commands section), or it is one of
the three declined pieces of the book documented in `docs/SNA_FOUNDATIONS.md`'s "Documented, not
built" section (alternatives to probability theory, §2.8/§28.4; classic combinatorial problems,
§13.5; transformers and deep generative graph models, §45.2/§45.3/§18.4), or nobody has needed it
yet and it is a small addition to an existing module when they do.

## A

| Term | The Atlas says | This repo |
|---|---|---|
| Actor | The entity a multilayer network's per-layer nodes refer back to; the component you get by following only inter-layer couplings. | Not modelled as a separate node: `sna/layers.py`'s `multilayer()` keeps one shared node id across every layer, so the actor *is* the node id rather than a coupling to compute. `supra_adjacency()` exposes the coupling edges directly if you need the matrix. |
| Acyclic Graph | See Tree. | See **Tree**, below. |
| Adjacency Matrix | `A[u,v] = 1` if `u` and `v` are connected, `0` otherwise. | `graphrag.sna.matrices.adjacency(graph, weight=None)` gives exactly this binary form; the default keeps the weight (see **Weighted Adjacency**, below). Dense or `scipy.sparse`, under the one node-order contract every matrix in this package shares. |
| Adjusted Mutual Information | Mutual information corrected so two random partitions score zero in expectation. | `graphrag.sna.evaluate.adjusted_mutual_information()`, printed in every `## Community evaluation` section (ATL-36) that has a ground truth to compare against. |
| Arborescence | A directed tree where every node but the root has in-degree one. | `graphrag.sna.hierarchy.arborescence()` -- the maximum spanning arborescence of §33.4, on the `relations` network; `sna hierarchy`. |
| Arborescence Forest | Several weakly connected components, each an arborescence. | Not built: `arborescence()` roots one component at a time (the one holding the node you name, or the largest). Running it once per weakly-connected component of `relations` gets the forest by hand today. |
| Assortativity | Nodes connecting to similar nodes; synonym of homophily. | Categorical: `sna/attributes.py`, `sna analyze --by <attr>` (permutation and rewiring nulls). Numeric: `sna/assortativity.py`, `--by <numeric>` (Pearson/Spearman over edge endpoints, ATL-31). Degree assortativity is in `measures.summary()`. |
| Arc | See Edge. | See **Edge**, below; on the directed `relations` network an arc is a `RELATED_TO` edge, `export.entity_relations()`. |
| Average Path Length | The mean shortest-path length over all pairs. | `graphrag.sna.paths.path_lengths()` / `analyse_paths()`, printed as "APL" in `## Paths and components`; estimated by sampling above `MAX_MATRIX_NODES`, which the report states. |

## B

| Term | The Atlas says | This repo |
|---|---|---|
| Balanced Graph | A directed graph whose in- and out-degree sequences match. | Not built as a named check. `degree.degree_sequence(graph, kind="in")` and `kind="out"` on `relations` give both sequences to compare by hand; `measures.reciprocity()` is the related, built statistic. |
| Betweenness Centrality | Normalised count of shortest paths through a node. | `graphrag.sna.measures.centrality(graph, "betweenness")`; `sna analyze` prints it, computed on `1/weight` since our weights are affinities, not distances (`docs/SNA.md`'s "Caveats that do not go away"). |
| Biclique | A clique in a bipartite network. | Not built: nothing enumerates complete bipartite subgraphs. The nearest built readings of two-mode structure are `bipartite.py`'s co-clustering methods and `coreperiphery.nestedness()` (NODF, §32.4). |
| Bipartite Network | Two node types, edges only across types. | `speakers-entities`, `export.speaker_entity_bipartite()`; `export.describe(graph).bipartite` reads it back from the `mode` attribute per §6.4. |
| Breadth First Search | Explore every neighbour of a node before the next node's neighbours. | `graphrag.sna.paths.exploration_order(graph, source, method="bfs")` and `bfs_tree()`; §29.2's BFS *sampler* is the separate `sna sample --method bfs`. |

## C

| Term | The Atlas says | This repo |
|---|---|---|
| Chain | Nodes orderable so each connects only to its predecessor and successor. | This is an ordinary simple path; see **Path**, below -- `paths.py` does not distinguish a "chain" as its own object. |
| Clique | A set of nodes with every possible edge present. | `graphrag.sna.measures.cliques()` -- maximal and size-`k` cliques under `CLIQUE_LIMIT`, §12.3; `sna analyze`'s `## Density` section. |
| Closeness Centrality | Normalised inverse of the average shortest-path length. | `graphrag.sna.measures.centrality(graph, "closeness")`, also on `1/weight`. |
| Commute Time | Expected round-trip random-walk length between two nodes. | `graphrag.sna.walks.commute_time()` / `commute_matrix()`, §11.3. |
| Complement Graph | Remove every edge that exists, add every edge that does not. | Not built: the complement of a sparse co-occurrence network is nearly complete and answers no question this package's personas ask -- see `docs/SNA_FOUNDATIONS.md`'s reasoning about routing problems on affinity weights, which the same objection extends to. |
| Complement of the Cumulative Distribution (CCDF) | The fraction of points at or above `x`. | `graphrag.sna.degree.degree_distribution().ccdf`, §9.2; the curve `sna degree` recommends reading over the raw scatter. |
| Connected Component | The maximal set of mutually reachable nodes. | `graphrag.sna.paths.components()`, §10.4; weak and strong on directed networks. |
| Connected Network | A network that is a single connected component. | `paths.components(graph).n == 1`; `## Paths and components` states the count either way. |
| Connection | See Edge. | See **Edge**, below. |
| Convex Network / Convex Subgraph | Every connected subgraph (resp. one subgraph) contains all shortest paths between its own nodes. | Not built: no persona question this package answers needs the convexity property tested, and nothing computes it. |
| Coupling Strategy | How same-actor nodes connect across layers: clique, chain, star. | Fixed rather than chosen: `layers.multilayer()` shares one node id across layers instead of coupling copies, so there is no separate coupling edge to strategise over; `supra_adjacency()` is the matrix that would carry one if we modelled coupling explicitly. |
| Cumulative Distribution (CDF) | The fraction of points below `x`. | Not returned directly (`degree_distribution()` returns the PMF and the CCDF the book recommends, §9.2); `1 - ccdf` recovers it. |
| Cycle | A path whose start and end node are the same. | `graphrag.sna.paths.cycle_space()`, §10.2 (a basis of independent cycles, via the fundamental-cycle construction on the trace); `hierarchy.cycle_share()` on the directed `relations` network, §33.2. |
| Cyclic Graph | A graph containing at least one cycle. | `hierarchy.hierarchy_type()` classifies `relations` as a DAG or not, §33.1. |

## D

| Term | The Atlas says | This repo |
|---|---|---|
| Degree | The number of edges at a node. | `graphrag.sna.degree.degree_sequence()`, in/out/weighted variants, §9.1; `sna degree`. |
| Degree Matrix | Diagonal matrix of node degrees. | `graphrag.sna.matrices.laplacian()` builds it internally (`D - A`); not exposed as its own function since nothing here consumes `D` alone. |
| Depth First Search | Explore as far as possible along a branch before backtracking. | `graphrag.sna.paths.exploration_order(graph, source, method="dfs")` and `dfs_tree()`. |
| Diameter | The longest shortest path in the network. | `graphrag.sna.paths.path_lengths()` (`.diameter`), §13.2; estimated above `MAX_MATRIX_NODES`. |
| Digraph | See Directed Graph. | See **Directed Graph**, below. |
| Directed Acyclic Graph | A directed graph with no cycle. | `hierarchy.hierarchy_type()` reports this classification for `relations`, §33.1. |
| Directed Cyclic Graph | A directed graph containing a cycle. | Same function, the other branch of the classification. |
| Directed Edge | A non-reciprocal, asymmetric edge. | `relations`, the one directed network `export` builds (§6.2), from the `RELATED_TO` extraction layer. |
| Directed Graph | A graph with directed edges. | `export.describe(graph).directed`; every centrality and clustering function in `measures.py`/`cluster.py` states whether it reads direction or flattens it and says so (ATL-06). |
| Directed Tree | A directed graph acyclic even if edge direction is ignored. | Not asserted as a separate check; `hierarchy_type()`'s classification plus `arborescence()`'s own guard (`_why_not_spanning`) cover the cases this package needs. |
| Disassortativity | Nodes connecting to *unlike* nodes; opposite of homophily. | Read off the same sign as **Assortativity**, above: a negative Pearson/Spearman coefficient from `assortativity.py`, or an EI index or Coleman homophily index below zero from `ego.py`/`attributes.py` (ATL-30). |
| Dynamic Network | Edges active/inactive at different times. | `graphrag.sna.layers.dynamic_edges()`, `windows()`, `snapshots()`, §7.4; `export.describe(graph).dynamic`. |

## E

| Term | The Atlas says | This repo |
|---|---|---|
| Edge | The interaction between two nodes. | What every `export.build_network()` call returns edges of: a shared document (`speakers`), a shared passage (`entities`), a `RELATED_TO` statement (`relations`); `docs/SNA.md`'s Commands section names all five networks and what an edge means in each. |
| Effective Resistance | The electrical resistance between two nodes, each edge a 1-ohm resistor. | `graphrag.sna.walks.effective_resistance()` / `resistance_matrix()`, §11.4. |
| Ego Network | A node, its neighbours, and the edges among them. | `graphrag.sna.export.ego()` / `sna ego <node>`, §30.1; alter composition by attribute in `ego.py`. |
| Eigenvalue / Eigenvector | `Av = λv`: the scaling factor and the vector it scales. | `graphrag.sna.matrices.eigenpairs()`, §5.5; consumed by `_eigenvector` centrality, spectral embedding (`embed.py`), and the Laplacian-based quantities in `walks.py`. |

## F

| Term | The Atlas says | This repo |
|---|---|---|
| Fiedler Vector | The Laplacian's second-smallest eigenvector. | `graphrag.sna.walks.fiedler_vector()` / `fiedler_cut()`, §11.5-adjacent (the min-cut section); also the basis of `spectral_embedding` (`embed.py`, §42.3). |
| Forest | More than one connected component, each a tree. | Not asserted as a named check; `paths.components()` plus a per-component acyclicity check (absent from `cycle_space()`'s output being empty) gets there by hand. |

## G

| Term | The Atlas says | This repo |
|---|---|---|
| Giant Connected Component | The component holding most of a real-world network's nodes. | `graphrag.sna.robustness.removal_curve()` tracks its share under attack, §22.1-22.2; `generators.expected_properties()` gives the Erdős-Rényi expectation to compare against, §16.1. |
| Graph | Nodes connected by edges. | The object every `sna` command loads: `export.build_network()` returns one `networkx.Graph`/`DiGraph`, described by `export.describe()`. |

## H

| Term | The Atlas says | This repo |
|---|---|---|
| Hairball | An incoherent, unreadable naive layout of a too-large, too-dense network; a.k.a. ridiculogram, spaghettigraph. | `graphrag.sna.guide`'s reading rule under Visualization names this exact failure mode and backboning (`sna backbone`) as the fix, not a different layout; `sna draw`'s own seeded force layout is the one this repo ships (ATL-49). |
| Heterogeneous Network | Multiple node and edge types. | `speakers-entities` (two node types) combined with `--relation-type` on `relations` (multiple edge types) is the nearest built reading; there is no single network mixing every type at once. |
| Heterophily | Synonym of disassortativity. | See **Disassortativity**, above. |
| Hitting Time | Expected one-way random-walk length from `u` to `v`. | `graphrag.sna.walks.hitting_time()` / `hitting_times()`, §11.3. |
| Homophily | Nodes connecting to similar nodes; synonym of assortativity. | See **Assortativity**, above; `docs/SNA.md`'s "Homophily" section and `sna ego` are the ATL-30 reading of it specifically. |
| Hub | A central node with many connections. | Not a named function; read off any centrality's top rank, `measures.top_n()`. |
| Hyperedges | Edges connecting more than two nodes at once. | `graphrag.sna.layers.build_hypergraph()`'s incidence -- one hyperedge per passage, over the entities it names (§7.3), not its clique projection. |
| Hypergraph | A graph containing hyperedges. | `graphrag.sna.layers.hypergraph()`; `clique_expansion()` is the lossy one-mode view when a function needs an ordinary graph. |

## I

| Term | The Atlas says | This repo |
|---|---|---|
| Identity Matrix | Ones on the diagonal, zero elsewhere. | `numpy.eye`, used inline wherever `matrices.py`/`walks.py` need it (e.g. the Laplacian pseudoinverse); not exposed as its own function since nothing here treats it as a result to report. |
| In-Component | Nodes that reach a directed graph's strongly connected core but are never reached back. | `graphrag.sna.paths.components(graph).condensation`, §10.4 -- the condensation DAG's structure gives in-/out-components as the nodes upstream/downstream of the core SCC. |
| Incidence Matrix | Nodes on rows, edges on columns. | `graphrag.sna.matrices.incidence()`, §8.3; the contract every bipartite weighting in `projection.py` and every hypergraph function in `layers.py` builds on. |
| Induced Subgraph | A subset of nodes plus every edge between them. | `networkx.Graph.subgraph()`, used inline (e.g. `export.ego()`'s radius-1 case, `roles.py`'s per-role subsets); not a separate top-level function. |
| Interlayer Coupling | The special edges joining one actor's copies across layers. | `graphrag.sna.layers.supra_adjacency()` -- the off-diagonal blocks of the supra-adjacency matrix, §7.2; see also **Actor** and **Coupling Strategy**, above, for why this package fixes the coupling instead of choosing one. |
| Isolated Node | A node with zero degree. | `keep_isolates` on `sna/backbone.py`'s filters decides whether one stays after an edge-removing pass strands it; `measures.summary()` counts them. |

## K

| Term | The Atlas says | This repo |
|---|---|---|
| k-Clique | A clique of `k` nodes. | `graphrag.sna.measures.cliques(graph, k=...)`, §12.3; also the unit `overlap.py`'s k-clique percolation (§38.3) grows communities from. |
| k-core | Nodes of minimum degree `k` after recursively stripping lower-degree ones. | `graphrag.sna.measures.coreness()` / `core_shells()`, §14.7; `coreperiphery.discrete_core()` is the related but distinct Borgatti-Everett core, §32.1. |

## L

| Term | The Atlas says | This repo |
|---|---|---|
| Laplacian | Degree matrix minus adjacency matrix. | `graphrag.sna.matrices.laplacian()` -- combinatorial and symmetric-normalised, §8.4; the shared primitive under `walks.py`, `embed.py`'s spectral embedding, and `vectordist.py`'s generalized Euclidean distance. |
| Lattice | Nodes uniformly placed in `n`-dimensional space, connected to their nearest neighbours. | Not built exactly: `generators.random_geometric()` places nodes in space and connects within a radius (§16.4), which is the closer relative among this package's generators, but it is not the deterministic fixed-neighbour grid the book defines. |
| Leaf Node | A node of degree one. | Not a named function; `degree.degree_sequence()` filtered to value `1`. |
| Left Eigenvector / Right Eigenvector | An eigenvector of `vA = vλ` vs. `Av = λv`. | `graphrag.sna.matrices.eigenpairs()` returns right eigenvectors; a left eigenvector of `A` is a right eigenvector of `A.T`, used where `walks.py` needs the stationary distribution of a directed chain. |
| Line Graph | Edges of the original graph become nodes, joined when they share an endpoint. | Not built: nothing here needs an edge-centric graph, and the co-occurrence networks this package builds already treat "shared context" as the node relation. |
| Link | See Edge. | See **Edge**, above. |

## M

| Term | The Atlas says | This repo |
|---|---|---|
| Maximal Clique | A clique you cannot extend and still have a clique. | `graphrag.sna.measures.cliques()`, §12.3. |
| Maximum Spanning Tree | The spanning tree with the highest total edge weight. | `graphrag.sna.backbone.py`'s `mst` backbone (maximum, since our weights are affinities not costs), §13.4; `hierarchy.arborescence()` is the directed analogue. |
| Metapath | A path through a heterogeneous network's typed nodes. | `graphrag.sna.embed.py`'s `metapath2vec` walk over a stated schema on `speakers-entities`, §43.3. |
| Minimum Spanning Tree | The spanning tree with the lowest total edge weight. | Not built: our edge weights are affinities (heavier means closer), so a *minimum* spanning tree would keep the weakest connections, which is the opposite of what `sna backbone`'s `mst` filter does on purpose -- see `docs/SNA_FOUNDATIONS.md`'s reasoning about routing costs, which is the same reason a true minimum-cost tour is out of scope. |
| Multidimensional Network | Multiple edge types; a subtype of multilayer. | `--relation-type` on `relations`, or `--layers relation` on `sna/layers.py`'s multilayer view, §7.2. |
| Multigraph | Multiple parallel edges between the same two nodes. | `export.describe(graph).multigraph`; this package's networks collapse parallel evidence into one weighted edge instead (a heavier edge, not two edges), so this is always `False` today. |
| Multilayer Network | Nodes connecting via different edge types, possibly with multiple identities. | `graphrag.sna.layers.multilayer()` -- by stance, facet, source or relation type, §7.2; `flatten()` and `supra_adjacency()` are its two representations. |
| Multipartite Network | Two or more node types, edges only across types. | Not built beyond the bipartite case (`speakers-entities`); no network here has three or more node types at once. |
| Multiplex Network | Multiple edge types; a subtype of multilayer (synonym of Multidimensional Network here). | Same as **Multidimensional Network**, above. |
| Mutual Information | Bits of information one random variable gives about another. | `graphrag.sna.stats.mutual_information()` / `normalized_mutual_information()`, §3.5. |

## N

| Term | The Atlas says | This repo |
|---|---|---|
| n, m-Clique | A biclique with `n` nodes of type 1 and `m` of type 2. | Not built; see **Biclique**, above. |
| Neighbor | A node directly connected by an edge. | `graph.neighbors(node)` (or `.predecessors`/`.successors` on `relations`); the unit every ego-network and role function in `ego.py`/`roles.py` works from. |
| Node | The fundamental unit of a graph. | A speaker, an entity, a topic or a document id, depending on the network; `docs/SNA.md`'s Commands section names what a node is in each of the five. |
| Normalized Mutual information (Normalized Mutual Information) | Mutual information scaled to `[0, 1]`. | `graphrag.sna.stats.normalized_mutual_information()`, §3.5; printed in `## Community evaluation` beside the adjusted version. |

## O

| Term | The Atlas says | This repo |
|---|---|---|
| Out-Component | Nodes a directed graph's strongly connected core can reach but that never reach back. | See **In-Component**, above -- the same condensation, the other direction. |

## P

| Term | The Atlas says | This repo |
|---|---|---|
| Parallel edges (Parallel Edges) | Two or more edges between the same pair of nodes. | See **Multigraph**, above: this package folds parallel evidence into one weighted edge. |
| Path | A walk with no repeated nodes. | `graphrag.sna.paths.path_lengths()` samples and measures them, §13.1-13.3; a specific shortest path between two named nodes is `networkx.shortest_path` on the `1/weight`-adjusted graph `measures._with_distance()` builds. |
| Planar Graph | A graph drawable on a 2D plane without crossing edges. | Not tested directly; `graphrag.sna.backbone.py`'s `pmfg` filter (planar maximally filtered graph, §13.4) constructs a planar backbone by construction, which is the nearest built object. |
| Probabilistic Network | A network whose edges carry existence probabilities. | `graphrag.sna.uncertain.py` -- edge probability `p` from mention tier and passage count, §28.1-28.3; `analyze --uncertain`. |

## R

| Term | The Atlas says | This repo |
|---|---|---|
| Reverse Graph | A directed graph with every edge direction flipped. | `graph.reverse()` (networkx), used inline where `paths.py`/`hierarchy.py` need the reverse condensation; not exposed as its own top-level function. |
| Ridiculogram | See Hairball. | See **Hairball**, above. |
| Right Eigenvector | See Left Eigenvector. | See **Left Eigenvector / Right Eigenvector**, above. |

## S

| Term | The Atlas says | This repo |
|---|---|---|
| Self-loop | An edge from a node to itself. | `export.describe(graph).self_loops`; this package's co-mention edges never self-loop by construction (an entity does not co-occur with itself), so the count is always zero today. |
| Simple Path | See Path. | See **Path**, above. |
| Simplicial Complex | Nodes whose connections form one high-order structure, embedded in geometric space. | `graphrag.sna.highorder.simplicial_complex()` -- one simplex per passage, over the entities it names, §34.1; `sna highorder`. Combinatorial, not embedded in a geometric space: this is where the definition above goes further than what §34.1's own construction (the downward closure of a hypergraph) requires. |
| Singleton | See Isolated Node. | See **Isolated Node**, above. |
| Spaghettigraph | See Hairball. | See **Hairball**, above. |
| Spanning Tree | A tree subgraph including every node of its parent graph. | `graphrag.sna.backbone.py`'s `mst` filter, §13.4; `hierarchy.arborescence()` is the directed analogue. |
| Square | A four-node, four-edge cycle. | Counted implicitly by `motifs.py`'s triad census only up to three nodes (§41.2); a square is not separately enumerated. |
| Star | One centre node connected to every other node, which connect to nothing else. | `measures.star_spread()` builds the reference star centralisation score used by `centralization()`, §14.8; not a network this package builds from data. |
| Stationary Distribution | Where an infinite random walk ends up; equals degree for undirected networks. | `graphrag.sna.walks.stationary_distribution()`, §11.1. |
| Stochastic Adjacency | The adjacency matrix normalised so rows (or columns) sum to one. | `graphrag.sna.matrices.stochastic()`, §8.2 -- the transition matrix every function in `walks.py` builds on. |
| Strongly Connected Component | A directed component where every node reaches every other, respecting direction. | `graphrag.sna.paths.components()`, §10.4. |
| Subgraph | A graph whose nodes and edges are a subset of another's. | `networkx.Graph.subgraph()`, used throughout (ego networks, per-role subsets, per-window snapshots); see **Induced Subgraph**, above. |

## T

| Term | The Atlas says | This repo |
|---|---|---|
| Temporal Network | See Dynamic Network. | See **Dynamic Network**, above. |
| Tree | A graph with no cycles. | `paths.cycle_space(graph).basis` empty is the check; `backbone.py`'s `mst` and `hierarchy.arborescence()` are the trees this package actually produces. |
| Triad | Three nodes, two edges. | `graphrag.sna.motifs.py`'s triad census counts every one of the 16 directed/undirected triad classes against a configuration null, §41.2. |
| Triangle | Three nodes, three edges. | `graphrag.sna.paths.triangle_count()`; the unit `measures.clustering()`'s local coefficient counts, §12.2. |
| Tripartite Network | Three node types, edges only across types. | Not built; see **Multipartite Network**, above. |

## U

| Term | The Atlas says | This repo |
|---|---|---|
| Undirected Network | All edges symmetric. | Every network but `relations`; `export.describe(graph).directed is False`. |
| Uniform Hypergraph | Every hyperedge has the same cardinality. | Not asserted; `layers.hyperedge_sizes()` reports the size distribution of the passage hypergraph, which is not uniform in practice (a passage names as many entities as it names). |
| Unipartite Network | One node type, unrestricted connections. | `speakers`, `entities` and `relations` -- every network here except `speakers-entities`. |
| Unweighted Network | Edges carry no weight, or all weights equal one. | Not a network this package builds: every edge weight counts shared evidence (documents, passages, statements), so `export.describe(graph).weighted` is always `True`. |

## V

| Term | The Atlas says | This repo |
|---|---|---|
| Vertex | See Node. | See **Node**, above. |

## W

| Term | The Atlas says | This repo |
|---|---|---|
| Walk | A node sequence where consecutive nodes are adjacent. | `graphrag.sna.paths.walk_counts()` (matrix powers, §10.1); `walks.py`'s random-walk quantities (ch. 11) are walks with a stopping/transition rule attached. |
| Weakly Connected Component | A directed component connected if edge direction is ignored. | `graphrag.sna.paths.components()`, §10.4. |
| Weighted Adjacency | An adjacency matrix whose cells hold the edge weight, not just 0/1. | `graphrag.sna.matrices.adjacency()`'s default (`weight="weight"`); pass `weight=None` for the 0/1 form the plain **Adjacency Matrix** entry above describes. |
| Weighted Edge | An edge carrying a strength. | Every edge this package builds: a shared-document count, a shared-passage count, a statement count. |
| Weighted Network | A network of weighted edges. | Every network this package builds; see **Unweighted Network**, above. |

## Abbreviations (ch. 55, folded in here per `docs/ATLAS_PLAN.md`)

Most of these name a matrix or quantity already mapped above by its full term; this table gives
the symbol and points back rather than repeating the mapping.

| Symbol | Meaning | See |
|---|---|---|
| A | Adjacency matrix | **Adjacency Matrix** |
| AMI | Adjusted Mutual Information | **Adjusted Mutual Information** |
| APL, APL v | Average Path Length (resp. from node `v`) | **Average Path Length** |
| AUC | Area Under the (ROC) Curve | `graphrag.sna.experiment.py`'s ROC/AUC, §25.2; `sna predict-eval` |
| BFS | Breadth First Search | **Breadth First Search** |
| C | The commute-time matrix | **Commute Time** |
| CC, CCavg, CCv | Global / average / local Clustering Coefficient | `graphrag.sna.measures.clustering()`, §12.2 |
| CCDF | Complement of the Cumulative Distribution Function | **Complement of the Cumulative Distribution (CCDF)** |
| CDF | Cumulative Distribution Function | **Cumulative Distribution (CDF)** |
| D | Degree matrix | **Degree Matrix** |
| DFS | Depth First Search | **Depth First Search** |
| E | Set of edges | **Edge** |
| ERGM | Exponential Random Graph Model | `graphrag.sna.null.ergm()`, §19.2, by maximum pseudo-likelihood |
| FN, FP, FPR | False Negative / Positive, False Positive Rate | `graphrag.sna.experiment.py`'s confusion matrix, §25.2 |
| GCC | Giant Connected Component | **Giant Connected Component** |
| GERM | Graph Evolution Rule Mining | One rule built: `graphrag.sna.predict.triad_closure_rule()` / `AssociationRule`, the §23.6 wedge-to-triangle rule Figure 23.7 walks through first, over per-document transactions (`sna predict`). GERM's other node-adding rules (ATL-23) and its sign rules (ATL-24, §24.2) are declined -- see those rows in `docs/ATLAS_PLAN.md`. |
| H | The hitting-time matrix | **Hitting Time** |
| I | Identity matrix | **Identity Matrix** |
| k̄, kv | Average degree, degree of node `v` | **Degree** |
| L | Laplacian matrix | **Laplacian** |
| MI | Mutual Information | **Mutual Information** |
| NMF | Non-negative Matrix Factorization | `graphrag.sna.matrices.nmf()`, §5.6 |
| NMI | Normalized Mutual Information | **Normalized Mutual Information** |
| Nu, Nu,l | Neighbours of `u` (resp. in layer `l`) | **Neighbor**; the layer form is `layers.multilayer()`'s per-layer adjacency |
| Puv | A path from `u` to `v` | **Path** |
| PCA | Principal Component Analysis | Not built as PCA by name; `matrices.svd()` is the same decomposition this package exposes, §5.6 |
| ROC | Receiver Operating Characteristic | `graphrag.sna.experiment.py`, §25.2; `sna predict-eval` |
| SBM | Stochastic Block Model | `graphrag.sna.cluster.py`'s degree-corrected SBM fit, §35.1; `analyze --method sbm` |
| SCC | Strongly Connected Component | **Strongly Connected Component** |
| SI, SIR, SIS | Compartmental epidemic models | `graphrag.sna.spread.py`, §20.1-20.3; `sna spread --model si\|sir\|sis` |
| SVD | Singular Value Decomposition | `graphrag.sna.matrices.svd()`, §5.6 |
| SVM | Support Vector Machine | Not built: no classifier in this package is an SVM; `sna/gnn.py`'s completion models are the built alternative for attribute prediction |
| TN, TP, TPR | True Negative / Positive, True Positive Rate | `graphrag.sna.experiment.py`'s confusion matrix, §25.2 |
| V | Set of nodes | **Node** |
| W | Set of possible edge weights | **Weighted Edge** |
| WCC | Weakly Connected Component | **Weakly Connected Component** |
| Ω | The effective-resistance matrix | **Effective Resistance** |
