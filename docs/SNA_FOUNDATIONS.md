# Foundations

Part I of *The Atlas for the Aspiring Network Scientist* (Coscia, v2, free at
<https://www.networkatlas.eu/>) is four chapters of background — probability, statistics, machine
learning, linear algebra — that contain no networks at all. This file is the same background,
written against this codebase: which of the book's conventions `graphrag sna` adopts, where each
piece of notation lives in `src/graphrag/sna/`, and which chapters of the book are deliberately
documented here rather than built. `docs/ATLAS_PLAN.md` holds the coverage matrix and the ticket
per chapter; every row of it marked "Doc" resolves to a section below.

Page numbers refer to the v2 PDF, which `scripts/atlas_chapter.py` prints chapter by chapter.

## How the Atlas approaches complexity, and how this package does

Chapter 1 (pp. 7–21) is the book's own framing, and this section is the programme's answer to it.
The argument for networks starts from a definition of complexity the book admits is one of
several: "A complex system is a system whose behavior cannot be reduced by analyzing its
interacting parts" (p. 7). From there the case for networks is short. If a system's behaviour is
in the interactions, then a model of the interactions is the model of the system, and almost
everything is interactions — "Societies are made by people entertaining multiple different types
of relations with each other", down through brains, genes, proteins and bonds: "It's interactions
all the way down" (p. 8).

Two things in the book's own history matter for what this package is. First, the creation myth it
deconstructs: network science is not Euler's, it is sociology's, and the book proposes replacing
"Euler's God-figure with Helen Hall Jennings — who invented the sociograms that were the real
beginning of network science" (p. 12). The reason that matters here is what sociologists brought
with them: before them, "graphs were seen as exact and deterministic mathematical objects"; they
saw the value in using those objects "to investigate a statistical and stochastic reality"
(pp. 11–12). A network built from a corpus is a measurement, not a theorem, and every number this
package prints is a statistic with an error attached.

Second, the author's own first application is a corpus: Dante's *Inferno* read as a complex
network (p. 14). The persona networks here — speakers who share documents, entities that share
passages — are that kind of object, and the book's digital-humanities framing, "the
computer-mediated manipulation and analysis of symbols representing different facets of reality"
(p. 14), is a fair description of what `sna export` does to an enrichment sidecar.

**Why an atlas and not a manual.** The book is explicit about what it is for: "the book provides
little to no known knowns, but it will provide you with all the known unknowns in network science
— so that your unknown unknowns are aligned with those of everyone else" (p. 15). Hence the title:
"An atlas doesn't do the exploration for you, but you can't explore without an atlas" (pp. 15–16).
The consequence for this programme is that a chapter of the book is a *question a reader might
ask*, not a recipe: our job per chapter is to make that question answerable over a persona's
corpus, with the caveats the book states printed next to the answer. The fourteen-part map on
pp. 16–17 is the skeleton of the coverage matrix in `docs/ATLAS_PLAN.md`; when a chapter is
"documented, not built" it is because the question it answers is not one a corpus network poses,
and the reason is written down below rather than left out.

**The two things this package adds that the book assumes a reader supplies.** The Atlas is written
for someone with a network in hand. It says so at the opening of Part VIII, where reality arrives:
"How do networks look like when you gather them via experiments and/or observations? Often, they
don't look at all like the ones from your models. The only expectation that reality meets is its
inability to meet expectations" (p. 17). This repository's networks are gathered, never given, so
two things the book leaves to the reader's discipline are made mechanical:

1. **The sampling frame is printed next to every number.** A network here describes who was
   recorded and what an extraction pass chose to name. A node is absent when nobody wrote it
   down, which is not the same as it not existing, so every report section states the frame it
   was computed over — network, filters, n, and where the labels came from — before the number.
2. **A null model is printed next to every structure claim.** Modularity, assortativity, a motif
   count and a backbone p-value are all quantities that a graph with no structure still produces,
   so a bare value is not a finding. Every claim about structure carries the null it beat and by
   how many of the null's standard deviations.

Both are rules, not habits: the `Report the sampling frame` and `Report the null model` entries
of `ALWAYS` in `src/graphrag/sna/guide.py`, which `graphrag sna guide` prints and which
`docs/SNA.md` and the SNA skill quote verbatim under "Always" (a unit test holds both copies to
the rendered text). Every ticket in the programme adds
its chapter's caveats to the same place, so the prose an agent reads and the prose the CLI prints
cannot drift apart.

## Probability

Chapter 2 (pp. 22–38) is the vocabulary the rest of the book is written in. This package uses it
with one convention stated once, here.

**Frequentist by default, and the book agrees.** Mrs. Frequent and Mr. Bayes toss a coin ten times
(pp. 22–23), and the book picks a side: "The default mode for this book is taking a frequentist
approach. However, here and there, Bayesian interpretations are going to pop up, thus you have to
know why we're doing things that way" (p. 23). Every p-value `graphrag sna` prints is frequentist
*and empirical*: it is the fraction of realisations of a stated null model that produced a
statistic at least as extreme as the observed one. Nothing here reads a p-value off a parametric
table unless the section says which distribution it assumed and why — the noise-corrected backbone
of ATL-27 will use a binomial null per edge, and it will say so on the edge.

Where a Bayesian reading appears, it is named where it appears:

- **Probabilistic networks (chapter 28, ATL-28).** An edge's `p` is a degree of belief about that
  edge given the evidence the layers carry — mention tier, how many passages support it, whether
  two annotation passes agreed — not a frequency over repeated corpora. There is only one corpus.
  That is a prior about a measurement, and the report will say "probability that this edge exists"
  rather than "probability that these two entities co-occur".
- **Softmax over neighbour counts (chapter 4, ATL-44/45).** The book points out that plain
  normalisation is frequentist while "softmax is more Bayesian (see Section 2.1): each friend in
  the first community is additional evidence you're part of that community" (p. 60). A GNN's
  class probabilities are of that kind, and the optional-extra commands say so.

**Notation, mapped.** The book writes an outcome as the random variable `X`, an event as a set of
outcomes, and a question as `P(X ∈ S)` (p. 24); the axioms are non-negativity, `P(Ω) = 1` and
additivity over mutually exclusive events (p. 25). In this codebase the outcome is one realisation
of a null model — one rewiring, one label shuffle, one resample — the event is "a statistic at
least as extreme as the observed one", and the empirical p-value is the share of realisations in
that event. The book's word for a single draw from a stochastic process is the one used
throughout: "The observed path is called a realization of the process" (p. 29).

**Conditional probability is where the reading rules come from.** `P(H|W) ≠ P(W|H)`, "and often
dramatically so" (p. 27): the probability of the observation given the null is not the probability
that the null is false given the observation. A z-score of 6 says the observation sits six of the
null's standard deviations from its mean; it says nothing about how likely the null is, and
nothing at all if the null was the wrong one to build. And the test for independence,
`P(H|W) = P(H)` (p. 27), is exactly the shape of this repository's worst confound. `sna analyze
--by` asks whether edges and labels are dependent; when a node's label was *borrowed* from the
documents its passages sit in, and its edges were drawn
from those same documents, edges and labels are dependent by construction. The permutation null
does not catch that — it destroys the dependency and leaves the observed value alone, which
inflates the z-score rather than deflating it. The fix is not a different statistic but a
different measurement: tag entities in the extraction sidecars. `attributes.py` records the
provenance per node and withholds the verdict when most labels were borrowed.

**Markov processes are the vocabulary of random walks.** Section 2.7 (pp. 30–31) splits stochastic
processes into Markov (the next state depends only on the current one), non-Markov, and
higher-order. "The classical Markov process in network science is the random walk. A random walker
simply chooses the next node it wants to occupy, and its options are determined solely by the node
it is currently occupying" (p. 31). Three tickets are written in that vocabulary: ATL-11 (the
stationary distribution, hitting and commute times, effective resistance — all properties of a
memoryless walk), ATL-11's non-backtracking operator and ATL-34's second-order networks (both are
higher-order Markov: "once you are in v, you also have to remember that you came from u", p. 32,
where §2.7 overflows its outline range),
and ATL-29's random-walk samplers, whose bias is a property of the walk's stationary distribution.
The stochastic matrices those walks run on — rows summing to one, "the probability of one possible
realization of a single step" (pp. 29–30) — are ATL-08's row- and column-stochastic adjacency.

**The book's notation and this codebase's names.** `sna/stats.py` is ATL-03, merged; the
identifiers below are its public ones.

| the book | where it lives here |
|---|---|
| `P(X ∈ S)` estimated by counting realisations (§2.2, §2.6) | `stats.empirical_p`; the sample it counts against comes from `sna/null.py` (ATL-19) |
| a permutation of labels holding the graph fixed (§2.4's independence test) | `stats.permutation_p`; `analyse_attribute` in `sna/attributes.py` shuffles labels the same way |
| binomial trials with success probability `p` (§3.2) | `stats.binomial_p`; the edge-level null of the noise-corrected backbone (ATL-27) |
| standardising an observation against a null sample | `stats.z_score`, with `stats.mean` and `stats.std`; imported by `sna/attributes.py` and `sna/cluster.py` |
| a family of tests rather than one (§3.3) | `stats.correct`, Bonferroni or Holm–Bonferroni |
| stochastic matrix (§2.6) | row- and column-stochastic adjacency in `sna/matrices.py` (ATL-08) |
| Markov process / random walk (§2.7) | `sna/walks.py` (ATL-11) |

Section 2.8, the alternatives to probability theory, is documented and not built; see below.

## Statistics

Chapter 3 (pp. 39–54) is short and every part of it is already load-bearing somewhere in `sna/`.

- **§3.1 Summary statistics (pp. 39–42).** Mean, median, variance, skewness, and the distinction
  between a long tail ("you can find arbitrarily large outliers") and a fat tail, where "outliers
  are more common and less extreme" (pp. 42–43). The book indicts its own field in passing: "In
  network science, we use the mean extensively — even though maybe we shouldn't" (p. 40), because
  degree distributions look nothing like the normal distribution the mean summarises well. That
  sentence is the specification `stats.describe` implements: it returns the moments *and* a
  `heavy_tailed` flag decided from order statistics alone, with the `caveat` sentence a report
  prints next to a mean it should not have trusted.
- **§3.2 Important distributions (pp. 43–45).** Uniform, normal, binomial, hypergeometric,
  Poisson, exponential, power law, lognormal, plus their cumulative forms. Two matter immediately:
  the hypergeometric, which the book says "is used especially for the task of network backboning
  (Chapter 27)" (p. 44) and which ATL-27 needs; and the power law against the lognormal, which
  "It's very tricky to tell ... apart" (p. 45) and which ATL-09 must therefore test rather than
  assert. `stats.fit_distributions` fits the continuous ones and KS-tests each fit.
- **§3.3 p-values (pp. 46–47).** The three things a p-value does not tell you — how likely you are
  to be right, how strong the effect is, how much evidence you have (p. 46) — and the multiple-test
  problem, where 100 tests at `p < 0.01` give a 63% chance of at least one false rejection, with
  the Bonferroni and Holm–Bonferroni corrections named on p. 48. `stats.empirical_p`,
  `stats.permutation_p` and `stats.binomial_p` are the three ways this package produces one, and
  `stats.correct` is the family adjustment; `sna/attributes.py` and `sna/cluster.py` import
  `stats.z_score` rather than each carrying their own.
- **§3.4 Correlation coefficients (pp. 48–50).** Covariance, Pearson, and Spearman on ranks for
  monotone-but-not-linear relationships; `stats.pearson`, `stats.spearman` and `stats.kendall`
  each return the coefficient, its p-value and the n both went into. The package already compares
  rankings: `compare.py` reads rank movers between two builds, and ATL-26 will compare projection
  schemes by rank correlation rather than by weight, since the weights are not on a common scale.
- **§3.5 Mutual information (pp. 50–54).** Entropy, then mutual information as "the amount of
  information entropy shared by the two vectors" (p. 52), comparing the joint probability with the
  product of the marginals (p. 53): `stats.entropy`, `stats.mutual_information`,
  `stats.normalized_mutual_information` and `stats.variation_of_information`. `compare_partitions`
  lives there too now, re-exported from `sna/cluster.py`; it returns normalised mutual information
  beside the adjusted Rand index, and `sna/attributes.py` reports both against the baseline a
  shuffle produces — because NMI, unlike ARI, is not chance-corrected and a random partition
  scores above zero.

## Machine learning

Chapter 4 (pp. 55–69) builds no network and neither does this section: it is the vocabulary Part XI
will use, recorded here so ATL-42 through ATL-45 do not each redefine it. Nothing in `sna/`
depends on it today.

- **The pipeline and its three phases (pp. 56–57).** Training, validation, test, with the rule that
  "we never optimize parameters over the validation set data" and that from the test phase "no
  information ever makes it back". Overfitting is what the validation phase exists to catch
  (p. 57).
- **Evaluation measures are deliberately not in this chapter.** The book puts them in the link
  prediction part instead, because "loss functions are used for training while evaluation measures
  are used for testing. These two phases of machine learning ... are different, they should be as
  separate as possible" (p. 55). That is why ATL-25 (experiment design, AUC, precision@k) is
  sequenced before ATL-23 lands its report, and why no prediction command in this package prints a
  ranked list without the holdout it was evaluated on.
- **Activation functions (pp. 58–61).** Softmax for multi-class output (p. 59); logistic, Gaussian
  and hyperbolic tangent for binary (p. 60); linear, step, ReLU, ELU and GeLU for passing
  information between layers, with `ReLU(x) = max(0, x)` "the star of the bunch" (p. 61).
- **Loss functions (pp. 61–65).** MAE and MSE, and why MSE "is sensitive to outliers" (p. 62);
  likelihood and log-likelihood, with the warning that matters for every model fit in this
  programme: "`L` is NOT the probability that the coin is fair given the experiment result ...
  What you observe is `P(x|θ)` ... but that is NOT equivalent to `P(θ|x)`" (p. 64). ATL-19's ERGM
  by maximum pseudo-likelihood and ATL-35's SBM fit are both `P(x|θ)` maximisations, and their
  reports will say so rather than calling the fitted model probable. Cross entropy closes the
  chapter (p. 65).
- **Gradient descent, as the book gives it.** Chapter 4 has no section named for it. It defines
  training operationally — "Training means nothing else than trying something, calculate the loss
  function, try something else and, if the loss now is lower, keep the change and keep going in
  that direction" (p. 61) — and mentions gradients only to name the failure, "vanishing gradients"
  (p. 61). This package inherits that level of detail: the optimiser is whatever the optional
  `graphrag[gnn]` extra provides, and the report names the loss, the number of epochs and the
  seed, not the descent rule.
- **The cost of computation (§4.4, pp. 66–69).** Sampling, negative sampling and batching. Two
  sentences here set expectations for the whole of Part XI. The first: "If you don't have a lot of
  observations, the appeal of machine learning is not that great" (p. 66) — a persona network is
  thousands of nodes, not millions, which is why the shallow methods (ATL-42, ATL-43) come first
  and the GNN tickets sit behind an optional extra. The second is negative sampling (pp. 66–67):
  "the number of non-connections dwarfs the number of connections", so a link-prediction
  evaluation that scores every non-edge measures the class imbalance and not the method. ATL-25
  owns that sampling and the cautions that go with it.

## Linear algebra

Chapter 5 (pp. 70–90) is the other half of how a network can be manipulated: "there are many
different ways to represent a network as a matrix. Linear algebra is necessary to understand what
sort of operations you can perform on those matrices, and what they mean" (p. 70). ATL-08, merged,
gave this package one home for those operations, `sna/matrices.py`, and moved the two pieces
that already existed onto it.

- **§5.1 Vectors (pp. 70–72).** A vector is a list of numbers and a point in a space whose
  dimension is the length of the list; length is Euclidean. This is what `--features embedding`
  and `--features spectral` both produce — one row per node — and what ATL-42 formalised as
  `sna/embed.py`'s `Embedding`: node order, dimensions, method and provenance, so a report never
  compares two of these vectors without first checking they answer the same question about node
  similarity (§42.1).
- **§5.2 Matrices (pp. 73–77).** `M[i][j]` is row `i`, column `j` (p. 73); transposition,
  symmetry, the identity, and multiplication as the composition of two coordinate changes
  (p. 77). The practical consequence for us is the one ATL-08 calls the **node-order contract**: a
  matrix means nothing without the ordering of nodes its rows and columns were built from, so
  every matrix this package returns is returned alongside its node list, sorted, and every
  consumer reads that list rather than assuming `list(graph.nodes)`.
- **§5.3 Tensors (p. 78).** A vector is a one-dimensional tensor, a matrix a two-dimensional one,
  and the generalisation keeps going. A multilayer network (ATL-07, ATL-40) is naturally a
  three-dimensional tensor — node, node, layer — although this package flattens it to a
  supra-adjacency matrix, which is the representation §8.1 and chapter 40 use. Tensor
  decomposition (pp. 86–87) is named in the book and not built here; ATL-40 does its multilayer
  work on the supra-adjacency instead.
- **§5.4 Vector products (pp. 79–81).** The dot product as a projection, the outer product, and
  positive semi-definite matrices. The last is not a formality: the book flags it as the door to
  chapter 47 — "there are secret identity matrices hidden in many formulas that can be replaced
  with matrices representing your network, to expand many classical statistical measures into
  network statistical measures" (p. 82). That is the generalized Euclidean distance ATL-47 builds
  on the Laplacian pseudoinverse.
- **§5.5 Eigenvalues and eigenvectors (p. 82).** `Mv = λv`; an `n × n` matrix has `n` eigenvalues,
  and multiplicity is how many eigenvectors share one (p. 83). Two uses are already in
  the code. `_eigenvector` in `sna/measures.py` takes the principal eigenvector of the weighted
  adjacency with `numpy.linalg.eigh` rather than a power iteration, because the iteration fails to
  converge on a graph with several components and a corpus network always has several. And the
  multiplicity fact is ATL-08's acceptance test: the combinatorial Laplacian's smallest eigenvalue
  is 0, and its multiplicity is the number of connected components.
- **§5.6 Matrix factorization (pp. 83–90).** Eigendecomposition, SVD, PCA and NMF, of which the
  book says both that NMF's non-negativity buys interpretability at some cost in fit, and that
  "Given their links to data clustering, both PCA and NMF are extensively used when looking for
  communities in your networks (Part X)" (p. 86). `embed.spectral` (ATL-42) is the factorization
  this package uses today: the smallest non-trivial eigenvectors of the symmetric normalised
  Laplacian, by `matrices.eigenpairs` rather than a randomised solver, giving coordinates that
  K-means and Gaussian mixtures can consume; `cluster.spectral_embedding` is the thin wrapper the
  clustering code calls, which clamps `dims` to what the graph can hold instead of refusing the
  way the ch. 42 contract does. ATL-08 moved its matrix construction onto `sna/matrices.py` so the
  embedding and the centralities agree about what the adjacency is.

## Deep graph learning: attention and the practical considerations (§45.1, §45.5)

§45.1 (pp. 654–656) generalises the GCN ATL-44 built: the normalisation
`D̂^-1/2(I + A)D̂^-1/2` is itself "sort of an attention mechanism ... where each neighbor get[s]
the same amount of attention", fixed at `1/sqrt(kv * ku)`; a Graph Attention Network (GAT,
Veličković et al. 2017) learns that weight instead, per neighbour, by backpropagation against the
same loss a GCN fits. `sna.gnn.gat_layer` is that layer; `classify`, `embed_gnn`,
`gcn_link_scorer` and `complete_attribute` all take `method="gat"` alongside `"gcn"` and
`"graphsage"`, multi-head via `heads` (concatenated at every hidden layer, averaged at the last).
`sna complete --method gat` additionally prints, per suggestion, the neighbours its fitted
attention leaned on most — the one new thing GAT offers a report that GCN and GraphSAGE cannot,
since their propagation is fixed before training starts and so has nothing to have learned.

§45.5 (pp. 662–664) is four practical considerations, none of them chapter-44-specific: they are
what the book holds *any* deep graph learning fit to before trusting it, so ATL-44's GCN and
GraphSAGE code is held to the same checklist as ATL-45's GAT. What follows is that checklist,
each item against what `sna.gnn` actually does.

- **Normalisation.** Not named by that word in §45.5, but implicit in every propagation matrix the
  chapter discusses: `normalized_adjacency`'s `D̂^-1/2(I + A)D̂^-1/2` (§44.3), `mean_adjacency`'s
  row-stochastic average, and `neighbor_mask`'s unweighted 0/1 support all scale a node's own
  contribution against its neighbours' rather than letting a hub's raw degree swamp everything it
  touches. Every method here takes one; none skips it.
- **Depth.** §44.2's over-smoothing limit (quoted in full by `sna.guide`'s "More layers smooths
  the network to one vector" rule) is why `classify` and `embed_gnn` default to shallow stacks —
  one or two hidden layers — and why `_completion_notes` flags a run asked for four or more.
- **Attention heads (§45.1).** `--gat-heads`/`heads` defaults to one, GAT's minimal case, and one
  head can underfit a split `gcn`/`graphsage` would fit at the same `epochs`/`lr` — this ticket's
  own known-answer tests needed two (classification) and four (attribute completion) to reach the
  held-out floor the other two methods reach with a single layer's worth of fixed weights at the
  same budget. Try 2 or more before trusting a `gat` result over its siblings; `classify` and
  `complete_attribute`'s own docstrings say so next to `heads`.
- **Dropout.** §45.5's own regularisation answer to overfitting — "you take your matrix W and you
  randomly set to zero some entries or entire rows", with an edge-dropout variant for `A` itself —
  is **not implemented** in `sna.gnn`. This package regularises a different way: `complete_attribute`
  withholds `val_share` of the owned labels before fitting and reports `held_out_accuracy` beside
  the majority-class null, so overfitting shows up as a held-out gap a reader can see, rather than
  being suppressed during training. That is a weaker guarantee than dropout — it catches
  overfitting after the fact instead of discouraging it during the fit — and worth a ticket if a
  persona's owned-label count grows large enough that a held-out check alone stops catching it (see
  `sna.guide`'s new rule on this).
- **Augmentation.** §45.5's virtual edges/nodes for a sparse graph ("it's kind of natural to add
  virtual edges between nodes of the same type, if they have a lot of common neighbors") are also
  **not implemented**. A corpus network's sparsity is itself evidence about the record (the same
  frame every report already carries), so adding edges the extraction pass never recorded would
  manufacture connectivity a reader could mistake for corpus structure. What would earn a ticket:
  a bipartite network (`--network speakers-entities`) sparse enough that message passing has
  nothing to propagate through, where a same-type virtual edge could be defined from the
  incidence structure alone rather than invented.
- **Mini-batching / sampling (§4.4, §45.5's "Computational Efficiency").** Named for graphs "so
  huge [matrices] might not fit in memory" — not this package's case. A persona network is
  thousands of nodes, the same regime chapter 4 already sets expectations for (see "Machine
  learning" above); every fit here runs the whole graph through one dense propagation matrix per
  epoch. What would earn a ticket: a persona network large enough that `normalized_adjacency` or
  `neighbor_mask` no longer fits comfortably in memory as a dense array.
- **Early stopping on a validation split.** `complete_attribute`'s `held_out`/`val_share` *is*
  this, applied to the one signal this package ever fits a GNN against (owned attribute values or
  the graph's own edges) — see §44.1/§44.3's ATL-44 rules and the dropout item above for why a
  held-out split carries more weight here than it would beside real regularisation.
- **The seed.** Every fit here takes one (`classify`, `embed_gnn`, `complete_attribute`) and
  reports it (`CompletionReport.seed`, `Classification.seed`), because §45.5 and chapter 4 agree
  training is a search, not a computation, and its outcome is reproducible only alongside the seed
  it ran with.
- **Applications (§45.5, pp. 663–664).** Recommender systems, misinformation detection, network
  medicine — named as the tasks that motivate the field, not techniques to build. This package's
  own application of the same architectures is narrower and already built: entity-attribute
  completion and link prediction, both through `sna.gnn`.

## Documented, not built

The first three items below are chapters or sections of the book that this programme answers with
a reason rather than with code. A disposition that cannot survive a network scientist asking "why
not?" belongs in a ticket instead, so each of the three names what would have to change for that
to happen. The fourth item declines nothing; it points at documentation another ticket owns.

### §2.8 and §28.4 — alternatives to probability theory

The book presents two alternatives to probability for handling uncertainty: Dempster–Shafer's
theory of evidence, which "moves from exact probabilities to probability intervals" and can model
"the case 'we don't know whether heads or tail' as distinct from 'heads' and 'tail'" (pp. 32–33),
and fuzzy logic, where "Things can have degrees of truthiness" and set operations become `min` and
`max` (pp. 35–36). Chapter 28 then shows all three applied to the same uncertain network, and the
result is the reason this package picks one: "As you can see the three theories give three very
different results. The differences are not only quantitative, but also qualitative: you get
results that are truly incompatible with each other — a distribution, an interval, a set of
classes" (p. 410).

This repository has exactly one evidence model per edge — the mention tier an anchoring pass
recorded, the number of supporting passages, and whether two annotation passes agreed — and
ATL-28 turns it into a single edge probability. Running the same evidence through three calculi
would produce three incompatible answers to one question, and the report would have no basis for
preferring any of them: the choice of calculus, not the corpus, would be doing the work. DST's
genuine advantage — distinguishing "we cannot prove either" from either outcome — is also the one
state this corpus already represents explicitly and elsewhere: an edge nobody wrote down is
absent, and the frame on every report says that absence is evidence about the record, not about
the world. Fuzzy logic's advantage, partial membership, is served by soft clustering (`--method
gmm` returns membership probabilities) and by overlapping community discovery (ATL-38), both of
which stay inside probability.

What would earn a ticket: a persona whose evidence is genuinely conflicting rather than merely
incomplete — two annotation passes that disagree about the same passage, kept as two sources
rather than reconciled on import. Combining conflicting evidence is the case DST handles and
probability handles badly (the ice-cream example, pp. 34–35), and it would be a ticket against
`sna/uncertain.py`, not a second framework over the whole package.

### §13.5 — classic combinatorial problems

Section 13.5 (pp. 204–206) covers the travelling salesman problem, minimum and maximum
Hamiltonian paths, graph colouring, SAT and vehicle routing, and notes that they are "classical
examples of NP-hard problems", solvable in practice only approximately (p. 205).

None of them is an analysis of a corpus network, for two different reasons — and the reasons
precede the cost.

The **routing family** — the travelling salesman problem, minimum and maximum Hamiltonian paths,
vehicle routing — are optimisations over a network whose edge weights are **costs incurred by
something that travels**: kilometres between cities, roads a vehicle has to cover, routes a fleet
has to run. Every edge weight in this package is an **affinity**: two speakers share three
documents, two entities are named together in eleven passages. Heavier means closer, not longer —
which is why `measures.py` inverts the weight into a `distance` attribute before handing a graph
to `networkx`'s betweenness and closeness, and why `docs/SNA.md` warns anyone computing a path by
hand to do the same. A minimum Hamiltonian cycle over co-mention affinities is a well-defined
number and answers no question anybody has about a corpus: there is no salesman, nothing traverses
the edges, and the tour that results is an artefact of the extraction pass's coverage.

**Graph colouring and SAT** are not about weights at all — colouring asks for the fewest colours
such that no edge joins two of the same, and SAT asks whether some assignment satisfies a set of
propositions — and neither is a question anyone puts to a corpus: nothing here has to be scheduled
into non-conflicting slots, and no formula comes out of an extraction pass to be satisfied. For
both families, NP-hardness (p. 205) only settles what the answer would cost; an approximate answer
to a question nobody asked is worse than none.

The parts of chapter 13 that *are* about network structure are built: §13.1–13.3 (exploration,
the path-length distribution, the diameter) in ATL-10, and §13.4's maximum spanning tree and
planar maximally filtered graph in ATL-27 — there the tree is a filter that keeps the strongest
structure in the network, not a route through it, which is why the same object is useful when the
tour is not.

What would earn a ticket: a persona network whose weights are costs. A network of handoffs with
durations, or a process graph where an edge is a step somebody performs, would make routing a real
question — and would need a different export, since nothing in the current graph schema records a
traversal cost. Colouring and SAT would need more than a different network: they would need a
question about the corpus that is an assignment problem, and none of the ones this package answers
is.

### §45.2 transformers, §45.3 deep generative graph models, and §18.4 graph generative networks

Three related pieces of the book are documented here. §45.2 (p. 656) describes graph transformers
as parallel attention: "Instead of having a single `αl` function per layer, you have many ... These
are all trained independently", each one an attention "head". §45.3 (pp. 657–660) covers
variational autoencoders, generative adversarial networks and autoregressive models for generating
graphs. §18.4 (pp. 270–273) is the same idea from the generative side: "we want the generative
process to 'learn' how a real world network looks like, so that it can generate synthetic versions
at will" (p. 271).

**Transformers** sit one step beyond the graph attention network that ATL-45 does build behind the
`graphrag[gnn]` extra. The step is more parameters — several independently trained attention
functions per layer instead of one — and the supervision available on a persona network does not
support the ones we already have. Entity-attribute completion (ATL-44) learns from the entities an
extraction pass happened to tag, which is hundreds of labels on thousands of nodes; link
prediction learns from a few thousand edges, evaluated on a holdout that ATL-25 builds. Chapter 4
is blunt about the regime: "If you don't have a lot of observations, the appeal of machine
learning is not that great" (p. 66). A graph transformer over that supervision would report a
better training loss and nothing a reader could act on.

**Deep generative models** would have to be trained on a population of graphs, and that is the
whole difference the book draws between them and the models this programme does build: with
generative models "we actually learn from many graphs how the algorithm should wire the network.
Exponential random graphs and the configuration model get closer to this approach, but they too
only learn from a single graph" (p. 657). A persona has one network per view.

The book does name the limits of the trivial version — feeding one adjacency matrix to a learner
"will only generate a graph with the same number of nodes as the input" and "it can only learn
from a single graph at a time" — but it does not stop there: "These limitations are solved in a
variety of ways. Just to give an example, GraphRNN allows for two moves: a graph-level update and
an edge-level update" (p. 271). So those two limits are not the objection; a solved limitation is
no argument. The objection is the one above — the corpus supplies one network per view, so a model
whose advantage is learning from many has nothing extra to learn from — plus what a report has to
do with the result afterwards: the configuration model's invariant, the degree sequence, can be
stated in one sentence and checked by the reader, and a learned generator's cannot.

That is also why **§18.4's use as a null generator is covered elsewhere**: ATL-16 builds the
synthetic graphs (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, configuration, planted partition,
random geometric) and the "observed versus random" section that puts every report's clustering and
path length beside their expectations, and ATL-19 gives every null one interface plus the ERGM of
§19.2. A report that says "this is more clustered than a configuration model of the same degree
sequence, z = 4.1" is falsifiable by a reader who knows what a configuration model is. A report
that says "this is more clustered than a GraphRNN trained on it" is not.

What would earn a ticket: a population of comparable networks from one persona — one entity
network per document, or per window from ATL-07's snapshots, over hundreds of windows — plus a
question about the generating process that a parametric null cannot state. At that point a
generative model would be learning something the configuration model cannot, and it would land as
an extension of ATL-16 behind the same optional extra as the other deep tickets.

### Chapters 52, 54 and 55 — built by ATL-52; chapter 56 still owned by a later ticket

These four are not dispositions of the kind above: nothing about them is being declined. Three are
now built; the fourth is documentation whose home is another ticket's output, listed here so a
reader of the coverage matrix can find all four in one place.

Chapter 52's applications are the **Applications** section of `docs/SNA.md`, reading each of the
chapter's eight cases against this corpus's own questions ("who is central", "which groups", "what
changed", "what would confirm this link") and the commands that answer them, rather than repeating
the book's own examples (cities, brains, patents), which stay here as background, not as a network
this package builds. Chapter 54's glossary is `docs/SNA_GLOSSARY.md`, every term mapped to a
command, a flag or a function, or to "not built, see `docs/SNA_FOUNDATIONS.md`" when the answer is
one of the three sections above; chapter 55's abbreviations fold into the same file, as its own
table. `docs/ATLAS_INDEX.md` -- the book's table of contents against every `§n.m` citation in
`src/graphrag/sna/*.py` and `sna/guide.py`, generated by `scripts/atlas_index.py` -- is the fourth
piece of ATL-52, answering "what does this package do for chapter *n*" the way the glossary answers
it for one term at a time.

Chapter 56's bibliography still becomes the citation each report section carries for the chapter it
implements, with `--cite` printing the reference (ATL-ENT-4).

## Reading order for a network scientist new to this repository

The book's own part map (pp. 16–17) orders the material; this orders the repository. Read down the
first table, then use the second to find the command for the chapter you came for.

| # | Read | What it gives you |
|---:|---|---|
| 1 | `README.md` | What the repository is: personas, a committed graph, an MCP server and a CLI |
| 2 | `docs/SNA_FOUNDATIONS.md` (this file) | Part I of the book against this codebase, and what is documented rather than built |
| 3 | `docs/SNA.md` | The four networks, the filters, every flag, and how to read a report |
| 4 | `graphrag sna guide` | The same selection and reading rules as the CLI prints them, which is the canonical copy |
| 5 | `docs/ATLAS_PLAN.md` | The coverage matrix: which chapter is built, which is a ticket, which is a section above |
| 6 | `docs/ARCHITECTURE.md` | Where the networks come from: the graph schema, the layers, the snapshot format |
| 7 | `src/graphrag/sna/` | `export.py` builds, `measures.py` measures, `cluster.py` groups, `attributes.py` tests a property, `analysis.py` renders the report |

Which command answers which chapter, for what exists today. Everything else in the book has a
ticket in `docs/ATLAS_PLAN.md`; the full term-by-term index is ATL-52's glossary, not this table.

| Chapters | Command | What it answers |
|---|---|---|
| 1 | this file | Why networks, and what this package adds to the book |
| 6–7 (graphs, bipartite, attributes) | `sna export --network ... [--where key=value]` | What the four networks are and how a filter changes what an edge means |
| 9, 12, 14 (degree, density, ranking) | `sna analyze` — summary and centrality tables | How large, how dense, how fragmented, and who ranks where |
| 19 (significance) | `sna analyze --samples N` | Whether the partition beats a degree-preserving null, as a z-score |
| 26 (bipartite projections) | `sna analyze --network speakers-entities --project speakers\|entities` | Who wrote about the same things, or which things the same people wrote about |
| 27 (backboning) | `sna backbone`, `--backbone <method>` on export / analyze / compare | Every filter of ch. 27 — naive and naive-top (§27.1), doubly stochastic (§27.2), high-salience skeleton (§27.3), convex (§27.4), disparity (§27.5), noise-corrected (§27.6), MST / PMFG (§13.4) — with a comparison table that says what each kept and on which null |
| 30 (homophily) | `sna analyze --by key` | Whether the network divides along a node attribute, against a permutation and a rewiring null |
| 35–36 (partitions, evaluation) | `sna analyze --method louvain\|kmeans\|gmm` | The groups, their stability across seeds, and their agreement with each other |
| 42 (shallow learning) | `sna analyze --features spectral\|embedding` | Groups by structure, or groups by what the nodes are about |
| 53 (data and tools) | `sna export --out ...graphml\|.json` | The network in a format Gephi, Cytoscape or a notebook can read |
| — (two builds) | `sna compare` | One network over two windows or two attribute values: n first, structure second |
| — (what the corpus says) | `sna stances` | Praise, complaint and substitution counts per entity, with the passages behind them |
