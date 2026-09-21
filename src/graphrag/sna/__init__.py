"""Network analysis over the graph: build networks, measure them, cluster them, explain why.

Five networks come out of the graph, and each answers a different question:

``speakers``
    Who appears alongside whom. Two speakers are joined when they appear in the same document;
    the edge weight is how many documents they share. Use it for "whose voice sits next to
    whose" and for brokerage between otherwise separate groups.
``entities``
    What is discussed together. Two entities are joined when both are mentioned in the same
    passage; the edge weight is how many passages mention both. Use it for "what is this thing
    talked about alongside".
``topics``
    The persisted ``Topic-[:CO_OCCURS]-Topic`` edges from ingestion, filtered to one persona.
``speakers-entities``
    Two modes at once: speakers and the entities their passages mention, joined when a speaker
    wrote a passage naming that entity. Project it onto either side for "who wrote about the
    same things" or "which things the same people wrote about".
``relations``
    What the corpus asserts, with a direction: the ``RELATED_TO`` edges an extraction pass
    wrote, from the entity doing the relating to the one related to, typed and weighted by how
    many passages state them. The only directed network here, so ``(u, v)`` is not ``(v, u)``
    and every measure that cannot read direction says that it flattened it (Atlas §6.2).
    :func:`graphrag.sna.export.describe` names any network's type in the book's words and the
    report prints it above the numbers.

Four filters narrow any of them, and each changes what an edge means rather than only how many
there are: ``--stance`` keeps mentions the annotation layer marked, so an edge becomes "praised
together" rather than "discussed together"; ``--facet`` keeps passages about one function of the
subject; ``--since``/``--until`` keep dated passages, which drops every undated document; and
``--relation-type`` keeps one kind of stated relation on the ``relations`` network.
``graphrag sna stances`` reports what the corpus says about each entity, and ``graphrag sna
compare`` builds one network over two windows and reports what moved.

A fourth filter cuts by what the nodes *are* rather than by what they say: ``--where key=value``
keeps the speakers an attribution pass tagged that way, the entities an extraction pass tagged
that way, and the nodes with no value of their own whose passages sit in documents an annotation
pass tagged that way. ``graphrag sna analyze --by key`` then asks whether the network divides
along that attribute at all, against a permutation null and a degree-preserving one -- because a
partition handed to a graph always produces a number.

:mod:`graphrag.sna.assortativity` is chapter 31, which asks the same question of a *number*: a
key the persona declared ``type: number``, or one of the counts every network carries
(``degree``, ``mentions``, ``documents``, ``chunks``). A label is shared or not, a number is near
or far, so the measure is the correlation between the values at the two ends of an edge (§31.1),
the curve of a node's value against its neighbours' with the exponent of its log-log fit, the
friendship paradox (§31.2) and the attribute's distribution across the degree bands (§31.3).
Chapter 31 over the degree -- whether the hubs talk to each other or to the periphery -- is
printed in every report, because it changes how every ranking above it reads.

Which of those two kinds of label a node carries decides whether that number is about the
attribute. Every network here draws its edges from co-membership in documents, so a label
borrowed from those same documents makes like join like by construction, and the permutation
null inflates it rather than catching it. The builders record the provenance per node and
:mod:`graphrag.sna.attributes` reports it beside the coefficient; giving entities attributes of
their own, in the extraction sidecars, is what turns the question from reportable into
answerable.

``graphrag.sna.layers`` is chapter 7: the same corpus read as something other than one simple
graph. ``--layers stance|facet|source|relation-type`` builds one layer per value and analyses
their flattening, with the per-layer table printed above the numbers and the supra-adjacency
matrix available at a stated interlayer coupling (§7.2, §8.1); ``hypergraph`` keeps a passage as
one hyperedge over everything it named, and ``clique_expansion`` is the projection that turns it
back into pairs (§7.3); ``dynamic_edges`` and ``snapshots`` are the timestamped edge list and the
windowed views of it, which ``sna compare --window`` and ``--since``/``--until`` are two cuts of
(§7.4).

``graphrag.sna.roles`` is chapter 15: what a node *is* rather than how much of it there is.
Every node is placed by two numbers computed against a partition -- how well connected it is
inside its own module and how evenly its ties spread over the others -- and the plane they span
is cut into the seven roles of Guimera and Amaral's cartography, which §15.1's four words
(broker, gatekeeper, core, periphery) are mapped back onto. Beside it sit §15.2's node
similarities: Jaccard, cosine and Pearson on the rows of the adjacency matrix, SimRank and the
book's own regular-equivalence recursion, and the positions and image matrix a blockmodel reads
off any of them. ``graphrag sna roles`` prints all of that, and the section it ends on is the
chapter's own observation turned into a finding: which entities play the same role, in the same
structural company, without ever appearing in one passage together. §15.3 (node embeddings) is a
pointer forward in the book and stays one here.

``graphrag.sna.motifs`` is chapter 41: the small shapes a network is built out of. The
three-node census -- four classes undirected, the sixteen MAN types directed -- then every shape
of one size counted and tested against the degree-preserving null of §19.1, which is §41.2's own
four-step procedure and the reason a motif count on its own is a description rather than a
finding. Underneath it, §41.3's isomorphism: VF2 for "are these the same graph" and an exhaustive
canonical labelling for the small patterns the mining keys on, exact where the book's minimum DFS
code is admittedly approximate. And then the bottom-up half, where the data names the shapes:
§41.4's transactional mining over one entity graph per document, where support counts the
documents holding a pattern, and §41.5's single-graph mining with the minimum image support, the
count that cannot grow when the pattern does. ``graphrag sna motifs`` prints all of it, and
``--mine`` is opt-in because §41.2 is honest that this is a hard problem.

``graphrag.sna.walks`` is chapter 11: what a random walker on one of these networks tells you.
Where it ends up (the stationary distribution, which on an undirected network is degree and
nothing else), how long it takes to arrive and to come back (hitting and commute time, the first
asymmetric and the second not), what that costs as an electrical resistance (§11.4's effective
resistance, a proper metric that shortest-path distance is not and one that moves less when a
single edge appears or vanishes), where the network is cheapest to cut, and what everybody agrees
on once the opinions have finished averaging. ``graphrag sna walks <persona> --from u --to v``
prints the pairwise half of that for one pair.

``graphrag.sna.highorder`` is chapter 34, where both of those stop being enough. ``graphrag sna
highorder <persona>`` reads the same passages twice. As a **simplicial complex** (§34.1): each
passage's entity set is a simplex and the complex holds every face of it, which is the downward
closure a hyperedge does not have, and what comes out of it is the generalized degree ``k_(d,m)``,
the incidence that says this is not a manifold, and the closure -- how many triangles of the
1-skeleton the passages actually filled, which is the size of the gap between the entity network
and the complex (§34.1 is explicit that cliques-to-simplices and simplices-to-cliques are not
commutative). And as a **memory network** (§34.2): a node is an ordered transition ``A -> B`` read
off consecutive passages of one document, an edge is a path ``A -> B -> C`` the corpus shows, and
§34.3's walk over it is printed beside the first-order walk on the ordinary transition network,
folded back onto entities, with the entities that change rank between them. A document boundary
ends a sequence, and the passage order is a transcript's running order, never a cause.

``graphrag.sna.paths`` is chapter 10 and §13.1-13.3, and every report prints it without being
asked: how many pieces the network comes in (weak and strong components, the giant one, the DAG of
strongly connected components and §10.4's in- and out-components round its core), how far apart its
nodes are (the shortest-path length distribution, its average, the diameter and the radius), how
many independent cycles it holds and which entry of §10.2's zoo it is, and -- on the directed
network -- the dyad census the reciprocity in the summary table is a ratio of. Two conventions run
through it: a length is a count of edges unless a weight is asked for, and an unreachable pair is
not a number you average, so the distances are measured on the giant component and what that left
out is printed beside them. Above a thousand nodes the all-pairs work is replaced by a seeded
sample of sources, and the diameter is then a lower bound and says so.

``graphrag.sna.hierarchy`` is chapter 33, and it runs on the ``relations`` network only, because
the chapter assumes direction and every one of its measures is degenerate without it. It names
which of §33.1's shapes the network is -- arborescence, forest, DAG, cyclic -- and then scores
how hierarchical it is four times over, because the chapter's argument is that each score is
wrong where the next one is right: the share of arcs lying on a cycle (§33.2, which calls every
DAG perfect), global reach centrality (§33.3, which calls nothing but a star perfect), the
maximum spanning arborescence and what one-boss-per-node discards (§33.4), and the agony of every
arrow that points up, minimised exactly (§33.5). All four are compared with the directed
configuration model, which is the chapter's own exercise, and §33.6's layering comes out as
coordinates for a drawing command that does not exist yet. ``graphrag sna hierarchy`` prints the
section on its own; every ``analyze`` of a directed network carries it.

Underneath all of that, ``graphrag.sna.matrices`` holds the matrix forms of a network -- the
adjacency, the stochastic matrix, the incidence matrix and the Laplacians -- under a single
node-order contract, so two measures computed in different modules can be read row by row
against the same list of nodes.

Every network is a sample of a corpus, not of a population: it describes who was recorded and
what was written down. ``graphrag.sna.guide`` holds the method-selection rules, and the report
repeats the sampling frame next to every number.

``graphrag.sna.sampling`` takes a smaller network out of one of these when the whole thing is
too large to look at, by the methods of Atlas ch. 29 -- induced and edge, BFS, snowball, forest
fire, and the two random walks -- and reports, next to the sample, which measurements that
sampler is known to distort and in which direction, plus how much of the network it did not see.

:mod:`graphrag.sna.experiment` is chapter 25, and it comes before any prediction rather than
after it: a ranked list of pairs is not a finding until somebody has said what it was tested on.
``holdout`` deletes a share of the edges and asks for them back, ``temporal_holdout`` trains on
what the corpus had joined before a date and tests on what it joined after -- the split the book
prefers, and the only one that does not damage the structure it then asks a predictor to read --
and ``kfold`` rotates the test set the way §25.1 describes. Negatives are sampled from the pairs
the network never joined, in the balanced ratio p. 359 asks for, and every report prints the
imbalance of the *unsampled* pair space beside the balanced n so that a flattering AUC can be
recognised as one. ``sna predict-eval`` runs it with the degree-product baseline §25.2 holds a
new predictor to; chapter 23's predictors report through the same machinery.

``write_graph`` and ``read_graph`` move a network between here and the formats the rest of the
field uses -- GraphML, node-link JSON, GEXF for Gephi, Pajek, a weighted edge list, Cytoscape's
node/edge CSV pair -- and ``FORMATS`` says what each one carries; ``to_igraph`` and
``to_graph_tool`` hand it to those libraries without either becoming a dependency.

Every one of those networks except ``relations`` is a *projection* of a membership table, and
:mod:`graphrag.sna.projection` is chapter 26: the weight on a projected edge is a modelling
choice, not a measurement. ``--projection`` picks one of the chapter's eleven schemes -- the
default ``simple`` count of §26.1, the Jaccard correction, the vectorized similarities of §26.2,
the hyperbolic discount of §26.3, resource allocation and its HeatS and hybrid variants (§26.4),
and the infinite-walk projection of §26.5 -- and the frame names whichever was used, because a
weight of 0.46 means nothing until the scheme is said. ``graphrag sna projections`` is §26.6: it
runs every scheme over one observation and reports their weight distributions, the rank agreement
between them and what each keeps at a threshold.

Any of them can then be filtered down with ``--backbone``, which is chapter 27: the naive weight
threshold ``--min-weight`` has always applied is §27.1, and the book spends the chapter arguing
against it. :mod:`graphrag.sna.backbone` adds the defensible ones -- doubly stochastic, the
high-salience skeleton, convex reduction, the disparity filter, the noise-corrected binomial
null, plus the maximum spanning tree and the planar maximally filtered graph of §13.4 -- each
leaving on every surviving edge the ``p_value`` or ``score`` it survived on, and each recording
in the frame what it removed. Weights here are counts of shared documents or passages, so
``noise-corrected`` is the one to reach for, and ``graphrag sna backbone`` prints what all of
them keep side by side before the choice is made.

:mod:`graphrag.sna.generators` is chapters 16 to 18: the synthetic networks the field was built
on -- G(n,p) and G(n,m), the cavemen and small-world models, preferential attachment, the
configuration model, the planted partition and LFR benchmarks, the random geometric graph -- each
a seeded wrapper saying what the book expects it to reproduce and what it cannot. The same module
holds chapter 17's argument as a report section: ``against_random`` puts the observed clustering,
path length and degree distribution beside the closed forms a random graph with the same n and m
would have, and beside a degree-preserving null, so "this network is clustered" is always a
comparison. Every ``sna analyze`` report carries that section.
:mod:`graphrag.sna.degree` is chapter 9: the degree variants (in, out, strength), the
distribution as a pmf, a log-binned histogram and the CCDF that is the one to read, and the
Clauset-Shalizi-Newman test of whether that distribution is a power law -- maximum likelihood
with an xmin chosen by KS distance, a parametric bootstrap for the goodness of fit, and
likelihood ratios against the lognormal and the exponential. ``graphrag sna degree`` prints it
and refuses the word "scale-free" unless all of that supports it.

Chapter 14 is the rest of the centrality battery and lives in :mod:`graphrag.sna.measures`
beside the four that were already there: ``reach`` (how much of the network a node commands
within k hops, §14.3), ``harmonic`` (closeness that stays defined when the network is in
pieces, §14.6), ``hits_hub`` and ``hits_authority`` (the two eigenvectors of §14.5, directed
networks only), ``coreness`` with ``core_shells`` and the edge-side ``k_truss`` (§14.7), and
``centralization`` -- Freeman's ratio against a star of the same size (§14.8), which is one
number about the shape of the network and none about any node in it. ``analyze`` prints all of
them, and a ``## Ranking stability`` section that resamples the network and says which of the
names in each table a different draw of the corpus would still have printed.

:mod:`graphrag.sna.spread` is chapters 20 and 21, and it is the one module here that measures
nothing. The compartmental models -- SI, SIS, SIR -- and chapter 21's complications -- the
threshold and cascade triggers of §21.1, the independent cascade of §21.2 -- are dynamics
*embedded* in a network, so running one on a co-mention graph is a what-if about structure and
never an observation: nothing ever travelled along these edges. ``graphrag sna spread`` prints
that sentence above the curve, bands the curve over seeded runs, and puts the run beside the
epidemic threshold §20.2 decides it by -- the spectral ``1/λ₁`` for this network next to the
book's ``1/(k̄+1)`` and ``k̄/⟨k²⟩``, which are properties of two models of a network. The same
report carries §21.3's immunisation comparison (random, degree, betweenness, and the
acquaintance strategy that needs no topology and still finds hubs) and §21.4's driver-node
share by maximum matching.
:mod:`graphrag.sna.coreperiphery` is chapter 32, the mesoscale organisation community
detection cannot see: one dense core and a periphery hanging off it. Both of §32.1's models are
fitted to the same adjacency and scored the same way, by the correlation between the network and
the ideal pattern -- the discrete partition of Figure 32.1 and a continuous coreness vector,
with the k-core numbers printed beside them as the chapter's *a priori* option and as a warning
that they answer a different question (p. 450). The rich club is there too, as the ratio it has
to be: phi(k) over its mean across degree-preserving rewirings, because a raw phi rises with k
on every network. §32.2's tension check then runs in every Louvain report and can overrule it:
when the core-periphery ideal reproduces the edges at least as well as the partition does, the
report says so and the communities are the periphery cut into slices. On a two-mode network the
same structure is nestedness (§32.4), reported as NODF against the fixed-fixed curveball null,
since the margins alone explain most of the nestedness most matrices have.

:mod:`graphrag.sna.null` holds the families every one of those numbers is compared against
(chapter 19): random graphs, degree-preserving rewirings, label shuffles, and the
bipartite-preserving null that rewires the memberships a projection came from instead of the
projection itself. One ``significance()`` contract turns any of them into a z-score and an
empirical p-value with the null named beside them, and ``ergm()`` fits §19.2's model by maximum
pseudo-likelihood, carrying the book's warnings on the result.

:mod:`graphrag.sna.uncertain` is chapter 28, which says none of those edges is a fact. Every
network's edges carry ``p``, their probability of existing, derived from the evidence the corpus
itself recorded -- the tier of the mentions behind an edge and how many passages support it --
and every report says how many of its edges rest on a single loose match before it prints
anything else (§28.1). ``analyze --uncertain`` then reads the network as the probabilistic
network §28.2 defines, sampling possible worlds so that every centrality and the partition's
modularity come back as a mean and an interval, and naming the adjacent ranks whose intervals
overlap as the ties they are. ``expected_degree``, ``expected_edges`` and
``degree_distribution`` are §28.3's closed forms and are exact; the rest is Monte Carlo and says
so. None of it deletes an edge, which is what separates it from backboning (§28.2, p. 401).

:mod:`graphrag.sna.ego` and :mod:`graphrag.sna.ties` are chapter 30, the mesoscale between one
node and the whole network. ``graphrag sna ego <persona> <node>`` builds the neighbourhood of
§30.1 and prints it twice, because the ego is joined to every alter by construction and the
version worth reading is the one with the ego taken out -- whose density is exactly the ego's
local clustering coefficient, with the pairs it leaves unjoined as the node's brokerage. ``--by``
adds what the alters are against what the network is, and names §30.4's majority illusion when a
value that is rare in the corpus is the majority here. In the ``--by`` section of a full
analysis, ``ei_index`` and ``coleman_index`` add §30.2's count-based pair beside the
assortativity: EI is negative for homophily and blind to how common each value is, so Coleman's
index and the label shuffle are printed with it, and a majority that looks homophilous only
because it is a majority is named as such. ``tie_strength_curve`` is §30.3 over the edges --
neighbourhood overlap against weight, with the bridges listed -- and every one of those numbers
carries §30.4's caveat, that homophily and contagion leave the same snapshot and only dated
attributes could tell them apart.
:mod:`graphrag.sna.evaluate` is chapter 36, and it runs over whatever partition a report
produced, whichever method produced it. Chapter 36 is a *battery* rather than a score -- "what's
'best' depends on what you want to use your communities for" -- so the section prints modularity
with the resolution limit that decides whether its small communities are findings or artefacts
(§36.1), conductance, internal density, the cut ratio and the three out-degree fractions beside
coverage and performance (§36.2), each with what it wants and which community size pushes it
there, and the partition scored as a link predictor over held-out edges against the same
partition's labels dealt out at random (§36.3). §36.4's mutual information against a ground truth
is there for a network that has one; a corpus network does not, and ``analyze --by`` is the
nearest honest thing, which is why it is not called truth.
:mod:`graphrag.sna.robustness` is chapter 22: what it takes to break one of these networks.
``sna robustness`` removes nodes at random (§22.1) and by each centrality in turn (§22.2) and
plots what is left of the largest component, beside the critical fraction kappa = <k^2>/<k>
gives for random failure; it lights §22.3's load-redistribution cascade at the heaviest node and
sweeps the slack the network would have needed to survive it; and it couples two of a persona's
networks node for node and runs §22.4's failures back and forth between them until the mutual
giant component settles. Random removal is the null the targeted curves are quoted against.
Nothing in a corpus network actually fails, so what these curves measure is how concentrated the
evidence is -- how much of the structure survives losing its best-connected records.

Underneath all of it, :mod:`graphrag.sna.stats` is chapter 3 of the Atlas in one module: summary
statistics that say when a mean is not the typical case, distribution fits, empirical,
permutation and binomial p-values with multiple-test correction, the three correlation
coefficients, and the information measures. Every significance claim this package makes goes
through it, so there is one convention for a p-value and one for a z-score rather than one per
module.
"""

from graphrag.sna.analysis import (
    RESAMPLE_METHODS,
    Analysis,
    RankingReport,
    RankingStability,
    ranking_payload,
    ranking_report,
    ranking_stability,
    render_markdown,
    render_ranking,
    run_analysis,
    to_payload,
)
from graphrag.sna.assortativity import (
    BUILTIN_NUMERIC,
    NUMERIC_NULLS,
    CurvePoint,
    DegreeBin,
    DegreeCorrelations,
    EndpointCorrelation,
    FriendshipParadox,
    NeighbourCurve,
    NumericReport,
    attribute_by_degree,
    degree_correlations,
    friendship_paradox,
    is_numeric,
    neighbour_average_curve,
    numeric_assortativity,
    numeric_payload,
    numeric_report,
    numeric_values,
    render_numeric,
)
from graphrag.sna.attributes import (
    ATTRIBUTE_NULLS,
    CONTAGION_CAVEAT,
    AttributeReport,
    HomophilyIndices,
    ValueHomophily,
    analyse_attribute,
    attribute_labels,
    attribute_payload,
    attribute_sources,
    coleman_index,
    ei_index,
    ei_ratio,
    homophily_indices,
    render_attribute,
)
from graphrag.sna.backbone import (
    BACKBONES,
    BackboneRow,
    backbone,
    compare_backbones,
    disparity_p,
    high_salience,
    noise_corrected_p,
    prune_below,
    render_backbones,
)
from graphrag.sna.cluster import (
    GMMResult,
    KChoice,
    KMeansResult,
    LouvainResult,
    NullModelResult,
    choose_k_gmm,
    choose_k_kmeans,
    compare_partitions,
    gmm,
    kmeans,
    louvain,
    null_model_modularity,
    spectral_embedding,
    spectral_embedding_full,
)
from graphrag.sna.compare import (
    Comparison,
    RankChange,
    Window,
    compare_windows,
    render_comparison,
)
from graphrag.sna.coreperiphery import (
    MAX_NESTEDNESS_NODES,
    RESTARTS,
    ContinuousCoreness,
    CorePeripheryReport,
    DiscreteCore,
    Nestedness,
    RichClub,
    RichClubRow,
    Tension,
    community_correlation,
    continuous_coreness,
    core_periphery_payload,
    core_periphery_report,
    core_periphery_tension,
    discrete_core,
    nestedness,
    render_core_periphery,
    render_nestedness,
    rich_club,
    two_mode_sides,
)
from graphrag.sna.degree import (
    DEGREE_KINDS,
    DegreeDistribution,
    DegreeReport,
    PowerLawFit,
    TailComparison,
    compare_tails,
    degree_distribution,
    degree_payload,
    degree_report,
    degree_sequence,
    fit_power_law,
    render_degree,
)
from graphrag.sna.ego import (
    AlterRow,
    EgoComposition,
    EgoReport,
    ego_payload,
    ego_report,
    render_ego,
)
from graphrag.sna.embed import (
    Embedding,
    pool,
    spectral,
)
from graphrag.sna.evaluate import (
    HOLDOUT_SHARE,
    MEASURES,
    NULL_PARTITIONS,
    CommunityScores,
    LinkPrediction,
    Measure,
    PartitionScores,
    TruthComparison,
    adjusted_mutual_information,
    evaluate_partition,
    evaluation_payload,
    link_prediction_score,
    render_evaluation,
    resolution_limit,
)
from graphrag.sna.experiment import (
    DEFAULT_FOLDS,
    DEFAULT_K,
    DEFAULT_NEGATIVES,
    DEFAULT_SHARE,
    HOLDOUT_METHODS,
    MAX_ENUMERATED_PAIRS,
    RANDOM_AUC,
    ConfusionMatrix,
    EvaluationReport,
    Holdout,
    Pair,
    PrecisionRecallPoint,
    RocPoint,
    ScoreTable,
    auc,
    average_precision,
    confusion,
    evaluate_predictor,
    experiment_payload,
    holdout,
    kfold,
    precision_at_k,
    precision_recall,
    prediction_power,
    preferential_attachment,
    random_scores,
    render_experiment,
    roc,
    temporal_holdout,
)
from graphrag.sna.export import (
    FORMATS,
    NETWORKS,
    PROJECTIONS,
    STANCES,
    GraphFormat,
    NetworkDescription,
    attr_count_key,
    attr_key,
    bipartite_projection,
    build_network,
    default_min_weight,
    ego,
    entity_co_mention,
    entity_passage_rows,
    entity_relations,
    graph_format,
    matches,
    read_graph,
    speaker_co_participation,
    speaker_entity_bipartite,
    to_graph_tool,
    to_igraph,
    topic_co_occurrence,
    where_text,
    write_graph,
)

# Two modules landed a ``describe`` in the same wave: chapter 3's describes a *sample*
# (:mod:`graphrag.sna.stats`), chapter 6's describes a *network*. One facade cannot hold the
# name twice, so the network one is qualified here; both keep their own name in their own
# module, which is where every caller and every test reaches for them.
from graphrag.sna.export import describe as describe_network
from graphrag.sna.facade import Measures, Network
from graphrag.sna.generators import (
    BROAD_DISPERSION,
    CLUSTERED_RATIO,
    GENERATORS,
    PATH_SOURCES,
    SHORT_RATIO,
    SIGNIFICANT_Z,
    AgainstRandom,
    ExpectedProperties,
    against_random,
    against_random_payload,
    barabasi_albert,
    caveman,
    configuration_model,
    erdos_renyi_gnm,
    erdos_renyi_gnp,
    expected_properties,
    lfr_benchmark,
    planted_partition,
    poisson_degrees,
    random_geometric,
    render_against_random,
    watts_strogatz,
)
from graphrag.sna.guide import render_guide
from graphrag.sna.hierarchy import (
    AGONY_EXACT_EDGES,
    HIERARCHY_TYPES,
    Agony,
    Arborescence,
    Cycles,
    Dummy,
    GlobalReach,
    HierarchyReport,
    HierarchyType,
    Layout,
    agony,
    analyse_hierarchy,
    arborescence,
    cycle_share,
    global_reach_centrality,
    hierarchy_payload,
    hierarchy_type,
    layered_layout,
    local_reach,
    render_hierarchy,
)
from graphrag.sna.highorder import (
    DEFAULT_DAMPING,
    DEFAULT_MAX_DIM,
    MAX_FACES,
    MAX_TRANSITIONS,
    MAX_TRIANGLES,
    RANK_TOLERANCE,
    TRANSITION_SEP,
    Closure,
    Face,
    MemoryReport,
    MemoryWalk,
    Passage,
    PassageSequence,
    RankMove,
    SimplicialComplex,
    SimplicialReport,
    build_memory_report,
    build_simplicial_report,
    closure,
    highorder_reports,
    memory_payload,
    memory_stationary,
    passage_sequences,
    render_memory,
    render_simplicial,
    second_order_network,
    simplicial_clustering,
    simplicial_complex,
    simplicial_degree,
    simplicial_degrees,
    simplicial_incidence,
    simplicial_payload,
    split_transition,
    transition_id,
    transition_network,
)
from graphrag.sna.layers import (
    LAYERINGS,
    DynamicEdges,
    Hypergraph,
    LayerSummary,
    Membership,
    Multilayer,
    TimedEdge,
    build_hypergraph,
    clique_expansion,
    document_dates,
    dynamic_edges,
    flatten,
    hyperedge_sizes,
    hypergraph,
    layer_table,
    multilayer,
    render_layers,
    snapshots,
    supra_adjacency,
    window_graph,
    windows,
)
from graphrag.sna.matrices import (
    LAPLACIANS,
    adjacency,
    dense,
    edge_incidence,
    eigenpairs,
    incidence,
    laplacian,
    nmf,
    node_order,
    project,
    stochastic,
    svd,
)
from graphrag.sna.measures import (
    CENTRALITIES,
    CENTRALIZATION_NOTE,
    CLUSTERING_KINDS,
    DIRECTED_CENTRALITIES,
    REACH_HOPS,
    STAR_SPREAD,
    Centralization,
    CliqueReport,
    DensityReport,
    brokers,
    centralities_for,
    centrality,
    centralization,
    centralization_table,
    cliques,
    clustering,
    core_shells,
    coreness,
    density_note,
    density_payload,
    density_report,
    harmonic,
    hits,
    independent_set,
    k_truss,
    local_clustering_summary,
    reach,
    reciprocity,
    render_density,
    star_spread,
    summary,
    top_n,
    truss_number,
    undirected_view,
)
from graphrag.sna.motifs import (
    CONNECTED_TRIADS,
    MAX_CANONICAL_NODES,
    MAX_MINING_NODES,
    MAX_SUBGRAPHS,
    MOTIF_NAMES,
    MOTIF_SAMPLES,
    MOTIF_SIZES,
    TRIAD_CODES,
    TRIAD_MEANING,
    UNDIRECTED_TRIADS,
    Mining,
    MotifProfile,
    MotifReport,
    MotifRow,
    Pattern,
    TriadCensus,
    build_motifs_report,
    canonical_form,
    canonical_graph,
    document_graphs,
    isomorphic,
    mni_support,
    motif_counts,
    motif_name,
    motif_profile,
    motifs_payload,
    render_motifs,
    single_graph_mining,
    transactional_mining,
    triad_census,
)
from graphrag.sna.multilayer import (
    AGREEMENT_HIGH,
    DEFAULT_MIN_SHARED,
    DEFAULT_NULL_SAMPLES,
    DEFAULT_OMEGA_SWEEP,
    DEPARTURE_HOMOGENEITY,
    Complementarity,
    LayerAgreement,
    LayerByLayerResult,
    LayerCommunity,
    MultilayerCommunity,
    MultilayerCommunityReport,
    OmegaSweepRow,
    Redundancy,
    SupraModularityResult,
    analyze_multilayer_communities,
    complementarity,
    density_table,
    layer_agreement,
    layer_by_layer_communities,
    modularity_matrix,
    multilayer_communities_payload,
    multilayer_modularity_value,
    null_model_multilayer_modularity,
    omega_sweep,
    optimize_multilayer_modularity,
    redundancy,
    render_multilayer_communities,
    supra_network,
)
from graphrag.sna.null import (
    DEFAULT_DIRECTED_ERGM_TERMS,
    DEFAULT_ERGM_TERMS,
    ERGM_CAVEATS,
    ERGM_TERMS,
    HOLDS_FIXED,
    NULLS,
    ERGMFit,
    ERGMTerm,
    Significance,
    bipartite_preserving,
    configuration,
    erdos_renyi,
    ergm,
    label_permutation,
    pairs_of,
    projected_side,
    render_ergm,
    significance,
)
from graphrag.sna.paths import (
    CYCLE_KINDS,
    MAX_EXACT_NODES,
    TRAVERSALS,
    ComponentReport,
    ComponentSet,
    Condensation,
    CycleSpace,
    DyadCensus,
    PathLengths,
    PathReport,
    SmallWorld,
    analyse_paths,
    bfs_tree,
    closed_walks,
    components,
    cycle_space,
    dfs_tree,
    dyad_census,
    exploration_order,
    path_lengths,
    paths_payload,
    render_paths,
    triangle_count,
    walk_counts,
)

# ``matrices.project`` is the simple-weight matrix product and ``projection.project`` is the
# graph builder over all of chapter 26's schemes; one facade cannot hold the name twice, so the
# graph builder is qualified here, as ``describe_network`` is above.
from graphrag.sna.projection import (
    ASYMMETRIC,
    DEFAULT_LAMBDA,
    SCHEME_NOTES,
    SCHEMES,
    ProjectionComparison,
    Scheme,
    SchemeWeights,
    compare_schemes,
    graph_from_incidence,
    project_incidence,
    projections_payload,
    render_projections,
    symmetrise,
)
from graphrag.sna.projection import project as project_pairs
from graphrag.sna.robustness import (
    CASCADE_TOLERANCES,
    COUPLINGS,
    LOADS,
    REPORT_FRACTIONS,
    TARGETED_STRATEGIES,
    AttackRow,
    AttackSummary,
    Cascade,
    Interdependent,
    InterdependentCurve,
    InterdependentPoint,
    MolloyReed,
    RemovalCurve,
    RemovalPoint,
    RobustnessReport,
    attack_summary,
    cascade,
    cascade_profile,
    couple,
    heaviest_node,
    interdependent,
    interdependent_curve,
    molloy_reed,
    removal_curve,
    render_robustness,
    robustness_payload,
    robustness_report,
    strategies_for,
)
from graphrag.sna.roles import (
    ATLAS_ROLE,
    GA_THRESHOLDS,
    MAX_SIMILARITY_NODES,
    ROLE_MEANING,
    ROLE_NAMES,
    SIMILARITIES,
    SIMRANK_DECAY,
    Blockmodel,
    NodeRole,
    RolePair,
    RolesReport,
    RoleTable,
    RoleThresholds,
    Similarity,
    blockmodel,
    build_roles_report,
    classify,
    guimera_amaral,
    image_matrix,
    participation,
    participation_ceiling,
    regular_equivalence,
    regular_similarity,
    render_roles,
    roles_payload,
    same_role_never_cooccur,
    similarity,
    simrank,
    structural_similarity,
    within_module_degree,
)
from graphrag.sna.sampling import (
    BIASES,
    SAMPLE_METHODS,
    SAMPLERS,
    BiasCheck,
    Completion,
    ProbeTarget,
    SampleBias,
    SamplerBias,
    bias_report,
    completion,
    render_sample,
    reweighted_degree_distribution,
    sample,
    sample_payload,
    sample_record,
)
from graphrag.sna.spread import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BETA,
    DEFAULT_MU,
    DEFAULT_RUNS,
    DEFAULT_STEPS,
    DEFAULT_THRESHOLD,
    ENDEMIC_TAIL,
    IMMUNISATION_STRATEGIES,
    MAX_SPREAD_NODES,
    MODEL_NOTES,
    SPREAD_BAND,
    SPREAD_FRAME,
    SPREAD_MODELS,
    STRATEGY_NOTES,
    DriverNodes,
    EpidemicThreshold,
    ImmunisationStrategy,
    Intervention,
    Reproduction,
    SpreadModel,
    SpreadReport,
    SpreadResult,
    SpreadStep,
    build_spread_report,
    driver_nodes,
    epidemic_threshold,
    immunise,
    interventions,
    logistic_curve,
    mean_field_curve,
    render_spread,
    reproduction,
    simulate,
    spread_payload,
)
from graphrag.sna.spread import (
    DEFAULT_SHARE as IMMUNISED_SHARE,
)
from graphrag.sna.stances import (
    EntityStances,
    SignedPair,
    StanceReport,
    build_stance_report,
    render_stances,
)
from graphrag.sna.stats import (
    CORRECTIONS,
    DISTRIBUTIONS,
    Correlation,
    FitResult,
    Summary,
    binomial_p,
    correct,
    describe,
    empirical_p,
    entropy,
    fit_distributions,
    kendall,
    likelihood_ratio,
    mutual_information,
    normalized_mutual_information,
    pearson,
    permutation_p,
    spearman,
    variation_of_information,
    z_score,
)
from graphrag.sna.ties import (
    COLOURABLE,
    TWO_MODE,
    OverlapBin,
    TieReport,
    neighbourhood_overlap,
    render_ties,
    tie_payload,
    tie_strength_curve,
    two_mode_kind,
)
from graphrag.sna.uncertain import (
    CERTAIN_RULE,
    EDGE_PROBABILITY,
    TIER_PROBABILITY,
    TIER_RULE,
    Evidence,
    Expectation,
    Uncertainty,
    combine,
    edge_probabilities,
    evidence_report,
    expectation,
    expected_degree,
    expected_density,
    expected_edges,
    node_expectation,
    realisations,
    reliability,
    render_evidence,
    render_uncertain,
    support,
    tier_probability,
    uncertain_payload,
    uncertainty_report,
)

# The second name two chapters landed in one wave, after ``describe``: chapter 9's
# ``degree_distribution`` is the pmf/CCDF of a whole *degree sequence*, chapter 28's is one
# node's probability of having each possible degree, over the possible worlds its uncertain
# edges allow. One facade cannot hold the name twice, so the ch. 28 one is qualified here; both
# keep their own name in their own module, which is where every caller reaches for them.
from graphrag.sna.uncertain import degree_distribution as uncertain_degree_distribution
from graphrag.sna.walks import (
    DYNAMICS,
    Consensus,
    Cut,
    commute_distance,
    commute_matrix,
    commute_time,
    consensus,
    consensus_time,
    directed_edges,
    effective_resistance,
    fiedler_cut,
    fiedler_vector,
    global_mincut,
    hitting_time,
    hitting_times,
    laplacian_pseudoinverse,
    mincut,
    non_backtracking_matrix,
    non_backtracking_radius,
    random_target_time,
    resistance_matrix,
    stationary_distribution,
    volume,
)

__all__ = [
    "AGONY_EXACT_EDGES",
    "AGREEMENT_HIGH",
    "ASYMMETRIC",
    "ATLAS_ROLE",
    "ATTRIBUTE_NULLS",
    "BACKBONES",
    "BIASES",
    "BROAD_DISPERSION",
    "BUILTIN_NUMERIC",
    "CASCADE_TOLERANCES",
    "CENTRALITIES",
    "CENTRALIZATION_NOTE",
    "CERTAIN_RULE",
    "CLUSTERED_RATIO",
    "CLUSTERING_KINDS",
    "COLOURABLE",
    "CONNECTED_TRIADS",
    "CONTAGION_CAVEAT",
    "CORRECTIONS",
    "COUPLINGS",
    "CYCLE_KINDS",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BETA",
    "DEFAULT_DAMPING",
    "DEFAULT_DIRECTED_ERGM_TERMS",
    "DEFAULT_ERGM_TERMS",
    "DEFAULT_FOLDS",
    "DEFAULT_K",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_DIM",
    "DEFAULT_MIN_SHARED",
    "DEFAULT_MU",
    "DEFAULT_NEGATIVES",
    "DEFAULT_NULL_SAMPLES",
    "DEFAULT_OMEGA_SWEEP",
    "DEFAULT_RUNS",
    "DEFAULT_SHARE",
    "DEFAULT_STEPS",
    "DEFAULT_THRESHOLD",
    "DEGREE_KINDS",
    "DEPARTURE_HOMOGENEITY",
    "DIRECTED_CENTRALITIES",
    "DISTRIBUTIONS",
    "DYNAMICS",
    "EDGE_PROBABILITY",
    "ENDEMIC_TAIL",
    "ERGM_CAVEATS",
    "ERGM_TERMS",
    "FORMATS",
    "GA_THRESHOLDS",
    "GENERATORS",
    "HIERARCHY_TYPES",
    "HOLDOUT_METHODS",
    "HOLDOUT_SHARE",
    "HOLDS_FIXED",
    "IMMUNISATION_STRATEGIES",
    "IMMUNISED_SHARE",
    "LAPLACIANS",
    "LAYERINGS",
    "LOADS",
    "MAX_CANONICAL_NODES",
    "MAX_ENUMERATED_PAIRS",
    "MAX_EXACT_NODES",
    "MAX_FACES",
    "MAX_MINING_NODES",
    "MAX_NESTEDNESS_NODES",
    "MAX_SIMILARITY_NODES",
    "MAX_SPREAD_NODES",
    "MAX_SUBGRAPHS",
    "MAX_TRANSITIONS",
    "MAX_TRIANGLES",
    "MEASURES",
    "MODEL_NOTES",
    "MOTIF_NAMES",
    "MOTIF_SAMPLES",
    "MOTIF_SIZES",
    "NETWORKS",
    "NULLS",
    "NULL_PARTITIONS",
    "NUMERIC_NULLS",
    "PATH_SOURCES",
    "PROJECTIONS",
    "RANDOM_AUC",
    "RANK_TOLERANCE",
    "REACH_HOPS",
    "REPORT_FRACTIONS",
    "RESAMPLE_METHODS",
    "RESTARTS",
    "ROLE_MEANING",
    "ROLE_NAMES",
    "SAMPLERS",
    "SAMPLE_METHODS",
    "SCHEMES",
    "SCHEME_NOTES",
    "SHORT_RATIO",
    "SIGNIFICANT_Z",
    "SIMILARITIES",
    "SIMRANK_DECAY",
    "SPREAD_BAND",
    "SPREAD_FRAME",
    "SPREAD_MODELS",
    "STANCES",
    "STAR_SPREAD",
    "STRATEGY_NOTES",
    "TARGETED_STRATEGIES",
    "TIER_PROBABILITY",
    "TIER_RULE",
    "TRANSITION_SEP",
    "TRAVERSALS",
    "TRIAD_CODES",
    "TRIAD_MEANING",
    "TWO_MODE",
    "UNDIRECTED_TRIADS",
    "AgainstRandom",
    "Agony",
    "AlterRow",
    "Analysis",
    "Arborescence",
    "AttackRow",
    "AttackSummary",
    "AttributeReport",
    "BackboneRow",
    "BiasCheck",
    "Blockmodel",
    "Cascade",
    "Centralization",
    "CliqueReport",
    "Closure",
    "CommunityScores",
    "Comparison",
    "Complementarity",
    "Completion",
    "ComponentReport",
    "ComponentSet",
    "Condensation",
    "ConfusionMatrix",
    "Consensus",
    "ContinuousCoreness",
    "CorePeripheryReport",
    "Correlation",
    "CurvePoint",
    "Cut",
    "CycleSpace",
    "Cycles",
    "DegreeBin",
    "DegreeCorrelations",
    "DegreeDistribution",
    "DegreeReport",
    "DensityReport",
    "DiscreteCore",
    "DriverNodes",
    "Dummy",
    "DyadCensus",
    "DynamicEdges",
    "ERGMFit",
    "ERGMTerm",
    "EgoComposition",
    "EgoReport",
    "Embedding",
    "EndpointCorrelation",
    "EntityStances",
    "EpidemicThreshold",
    "EvaluationReport",
    "Evidence",
    "Expectation",
    "ExpectedProperties",
    "Face",
    "FitResult",
    "FriendshipParadox",
    "GMMResult",
    "GlobalReach",
    "GraphFormat",
    "HierarchyReport",
    "HierarchyType",
    "Holdout",
    "HomophilyIndices",
    "Hypergraph",
    "ImmunisationStrategy",
    "Interdependent",
    "InterdependentCurve",
    "InterdependentPoint",
    "Intervention",
    "KChoice",
    "KMeansResult",
    "LayerAgreement",
    "LayerByLayerResult",
    "LayerCommunity",
    "LayerSummary",
    "Layout",
    "LinkPrediction",
    "LouvainResult",
    "Measure",
    "Measures",
    "Membership",
    "MemoryReport",
    "MemoryWalk",
    "Mining",
    "MolloyReed",
    "MotifProfile",
    "MotifReport",
    "MotifRow",
    "Multilayer",
    "MultilayerCommunity",
    "MultilayerCommunityReport",
    "NeighbourCurve",
    "Nestedness",
    "Network",
    "NetworkDescription",
    "NodeRole",
    "NullModelResult",
    "NumericReport",
    "OmegaSweepRow",
    "OverlapBin",
    "Pair",
    "PartitionScores",
    "Passage",
    "PassageSequence",
    "PathLengths",
    "PathReport",
    "Pattern",
    "PowerLawFit",
    "PrecisionRecallPoint",
    "ProbeTarget",
    "ProjectionComparison",
    "RankChange",
    "RankMove",
    "RankingReport",
    "RankingStability",
    "Redundancy",
    "RemovalCurve",
    "RemovalPoint",
    "Reproduction",
    "RichClub",
    "RichClubRow",
    "RobustnessReport",
    "RocPoint",
    "RolePair",
    "RoleTable",
    "RoleThresholds",
    "RolesReport",
    "SampleBias",
    "SamplerBias",
    "Scheme",
    "SchemeWeights",
    "ScoreTable",
    "SignedPair",
    "Significance",
    "Similarity",
    "SimplicialComplex",
    "SimplicialReport",
    "SmallWorld",
    "SpreadModel",
    "SpreadReport",
    "SpreadResult",
    "SpreadStep",
    "StanceReport",
    "Summary",
    "SupraModularityResult",
    "TailComparison",
    "Tension",
    "TieReport",
    "TimedEdge",
    "TriadCensus",
    "TruthComparison",
    "Uncertainty",
    "ValueHomophily",
    "Window",
    "adjacency",
    "adjusted_mutual_information",
    "against_random",
    "against_random_payload",
    "agony",
    "analyse_attribute",
    "analyse_hierarchy",
    "analyse_paths",
    "analyze_multilayer_communities",
    "arborescence",
    "attack_summary",
    "attr_count_key",
    "attr_key",
    "attribute_by_degree",
    "attribute_labels",
    "attribute_payload",
    "attribute_sources",
    "auc",
    "average_precision",
    "backbone",
    "barabasi_albert",
    "bfs_tree",
    "bias_report",
    "binomial_p",
    "bipartite_preserving",
    "bipartite_projection",
    "blockmodel",
    "brokers",
    "build_hypergraph",
    "build_memory_report",
    "build_motifs_report",
    "build_network",
    "build_roles_report",
    "build_simplicial_report",
    "build_spread_report",
    "build_stance_report",
    "canonical_form",
    "canonical_graph",
    "cascade",
    "cascade_profile",
    "caveman",
    "centralities_for",
    "centrality",
    "centralization",
    "centralization_table",
    "choose_k_gmm",
    "choose_k_kmeans",
    "classify",
    "clique_expansion",
    "cliques",
    "closed_walks",
    "closure",
    "clustering",
    "coleman_index",
    "combine",
    "community_correlation",
    "commute_distance",
    "commute_matrix",
    "commute_time",
    "compare_backbones",
    "compare_partitions",
    "compare_schemes",
    "compare_tails",
    "compare_windows",
    "complementarity",
    "completion",
    "components",
    "configuration",
    "configuration_model",
    "confusion",
    "consensus",
    "consensus_time",
    "continuous_coreness",
    "core_periphery_payload",
    "core_periphery_report",
    "core_periphery_tension",
    "core_shells",
    "coreness",
    "correct",
    "couple",
    "cycle_share",
    "cycle_space",
    "default_min_weight",
    "degree_correlations",
    "degree_distribution",
    "degree_payload",
    "degree_report",
    "degree_sequence",
    "dense",
    "density_note",
    "density_payload",
    "density_report",
    "density_table",
    "describe",
    "describe_network",
    "dfs_tree",
    "directed_edges",
    "discrete_core",
    "disparity_p",
    "document_dates",
    "document_graphs",
    "driver_nodes",
    "dyad_census",
    "dynamic_edges",
    "edge_incidence",
    "edge_probabilities",
    "effective_resistance",
    "ego",
    "ego_payload",
    "ego_report",
    "ei_index",
    "ei_ratio",
    "eigenpairs",
    "empirical_p",
    "entity_co_mention",
    "entity_passage_rows",
    "entity_relations",
    "entropy",
    "epidemic_threshold",
    "erdos_renyi",
    "erdos_renyi_gnm",
    "erdos_renyi_gnp",
    "ergm",
    "evaluate_partition",
    "evaluate_predictor",
    "evaluation_payload",
    "evidence_report",
    "expectation",
    "expected_degree",
    "expected_density",
    "expected_edges",
    "expected_properties",
    "experiment_payload",
    "exploration_order",
    "fiedler_cut",
    "fiedler_vector",
    "fit_distributions",
    "fit_power_law",
    "flatten",
    "friendship_paradox",
    "global_mincut",
    "global_reach_centrality",
    "gmm",
    "graph_format",
    "graph_from_incidence",
    "guimera_amaral",
    "harmonic",
    "heaviest_node",
    "hierarchy_payload",
    "hierarchy_type",
    "high_salience",
    "highorder_reports",
    "hits",
    "hitting_time",
    "hitting_times",
    "holdout",
    "homophily_indices",
    "hyperedge_sizes",
    "hypergraph",
    "image_matrix",
    "immunise",
    "incidence",
    "independent_set",
    "interdependent",
    "interdependent_curve",
    "interventions",
    "is_numeric",
    "isomorphic",
    "k_truss",
    "kendall",
    "kfold",
    "kmeans",
    "label_permutation",
    "laplacian",
    "laplacian_pseudoinverse",
    "layer_agreement",
    "layer_by_layer_communities",
    "layer_table",
    "layered_layout",
    "lfr_benchmark",
    "likelihood_ratio",
    "link_prediction_score",
    "local_clustering_summary",
    "local_reach",
    "logistic_curve",
    "louvain",
    "matches",
    "mean_field_curve",
    "memory_payload",
    "memory_stationary",
    "mincut",
    "mni_support",
    "modularity_matrix",
    "molloy_reed",
    "motif_counts",
    "motif_name",
    "motif_profile",
    "motifs_payload",
    "multilayer",
    "multilayer_communities_payload",
    "multilayer_modularity_value",
    "mutual_information",
    "neighbour_average_curve",
    "neighbourhood_overlap",
    "nestedness",
    "nmf",
    "node_expectation",
    "node_order",
    "noise_corrected_p",
    "non_backtracking_matrix",
    "non_backtracking_radius",
    "normalized_mutual_information",
    "null_model_modularity",
    "null_model_multilayer_modularity",
    "numeric_assortativity",
    "numeric_payload",
    "numeric_report",
    "numeric_values",
    "omega_sweep",
    "optimize_multilayer_modularity",
    "pairs_of",
    "participation",
    "participation_ceiling",
    "passage_sequences",
    "path_lengths",
    "paths_payload",
    "pearson",
    "permutation_p",
    "planted_partition",
    "poisson_degrees",
    "pool",
    "precision_at_k",
    "precision_recall",
    "prediction_power",
    "preferential_attachment",
    "project",
    "project_incidence",
    "project_pairs",
    "projected_side",
    "projections_payload",
    "prune_below",
    "random_geometric",
    "random_scores",
    "random_target_time",
    "ranking_payload",
    "ranking_report",
    "ranking_stability",
    "reach",
    "read_graph",
    "realisations",
    "reciprocity",
    "redundancy",
    "regular_equivalence",
    "regular_similarity",
    "reliability",
    "removal_curve",
    "render_against_random",
    "render_attribute",
    "render_backbones",
    "render_comparison",
    "render_core_periphery",
    "render_degree",
    "render_density",
    "render_ego",
    "render_ergm",
    "render_evaluation",
    "render_evidence",
    "render_experiment",
    "render_guide",
    "render_hierarchy",
    "render_layers",
    "render_markdown",
    "render_memory",
    "render_motifs",
    "render_multilayer_communities",
    "render_nestedness",
    "render_numeric",
    "render_paths",
    "render_projections",
    "render_ranking",
    "render_robustness",
    "render_roles",
    "render_sample",
    "render_simplicial",
    "render_spread",
    "render_stances",
    "render_ties",
    "render_uncertain",
    "reproduction",
    "resistance_matrix",
    "resolution_limit",
    "reweighted_degree_distribution",
    "rich_club",
    "robustness_payload",
    "robustness_report",
    "roc",
    "roles_payload",
    "run_analysis",
    "same_role_never_cooccur",
    "sample",
    "sample_payload",
    "sample_record",
    "second_order_network",
    "significance",
    "similarity",
    "simplicial_clustering",
    "simplicial_complex",
    "simplicial_degree",
    "simplicial_degrees",
    "simplicial_incidence",
    "simplicial_payload",
    "simrank",
    "simulate",
    "single_graph_mining",
    "snapshots",
    "speaker_co_participation",
    "speaker_entity_bipartite",
    "spearman",
    "spectral",
    "spectral_embedding",
    "spectral_embedding_full",
    "split_transition",
    "spread_payload",
    "star_spread",
    "stationary_distribution",
    "stochastic",
    "strategies_for",
    "structural_similarity",
    "summary",
    "support",
    "supra_adjacency",
    "supra_network",
    "svd",
    "symmetrise",
    "temporal_holdout",
    "tie_payload",
    "tie_strength_curve",
    "tier_probability",
    "to_graph_tool",
    "to_igraph",
    "to_payload",
    "top_n",
    "topic_co_occurrence",
    "transactional_mining",
    "transition_id",
    "transition_network",
    "triad_census",
    "triangle_count",
    "truss_number",
    "two_mode_kind",
    "two_mode_sides",
    "uncertain_degree_distribution",
    "uncertain_payload",
    "uncertainty_report",
    "undirected_view",
    "variation_of_information",
    "volume",
    "walk_counts",
    "watts_strogatz",
    "where_text",
    "window_graph",
    "windows",
    "within_module_degree",
    "write_graph",
    "z_score",
]
