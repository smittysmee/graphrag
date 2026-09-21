"""Link prediction on a simple graph (Atlas ch. 23): a ranked list of hypotheses, never an edge.

*"Link prediction wants to find a theory to predict [new] events"* (p. 327): observe the current
edges, formulate a hypothesis about how nodes decide to link, and turn the hypothesis into
``score(u, v)`` for every pair the network has not yet joined. Every scorer here is that function,
and every one of them is a :class:`graphrag.sna.experiment.ScoreTable` so it runs through
:func:`graphrag.sna.experiment.evaluate_predictor` exactly like the ``preferential_attachment``
baseline that module already carries -- the wave ordering that put chapter 25 before this ticket
exists so that a score is never printed without the AUC and precision@k that say whether it means
anything (§25.2).

**§23.1-23.4, the neighbourhood family.** :func:`common_neighbors` (§23.2, *"the more common
neighbors u and v have, the more triangles we can close with a single edge"*), :func:`jaccard`
(§23.2's variant that controls for how many neighbours the two nodes have to begin with),
:func:`adamic_adar` and :func:`resource_allocation` (§23.3-23.4, which both discount a common
neighbour by its own degree -- *"the hubs do not have enough bandwidth to make the
introduction"* -- logarithmically and linearly respectively). ``preferential_attachment`` itself
is not repeated here: it needs no common neighbour and already lives in
:mod:`graphrag.sna.experiment`, next to the baseline it doubles as (§23.1).

**§23.7, Katz.** :func:`katz` sums *"score(u, v) = sum_l alpha^l |P^l_{u,v}|"* over every walk
length, using the same convergence bound as Katz centrality (§14.4): ``beta`` must stay under
``1 / lambda_max`` or the sum diverges. **Where this departs from the book:** p. 336 calls
``P^l_{u,v}`` "paths", but the closed form the chapter and every implementation actually use --
``(I - beta*A)^-1 - I``, which is what :func:`katz` computes -- sums walks (a walk may revisit a
node; counting only simple paths is #P-hard in general and nobody does it). The two agree for the
book's own worked case, a single length-2 connection, only in the limit of a small ``beta``; see
:func:`katz`'s docstring for how small.

**§23.5, HRG.** :func:`hrg_scores` reads a fitted :class:`graphrag.sna.cluster.HRGFit` (built by
:func:`graphrag.sna.cluster.hrg_fit`, imported and never edited here) as *"the likelihood of nodes
to connect is proportional to the edge density of the group in which they both are"* (p. 331):
``score(u, v) = |E_c| / e(|E_c|)``, the observed edges of the smallest dendrogram community
holding both ``u`` and ``v`` over the edges a configuration-model null (§18.1) would expect there.

**§23.6, association rules.** GERM (Graph Evolution Rule Mining, Berlingerio et al. 2009) mines
*every* frequent pattern and every one-edge extension of it; this module instantiates exactly one
such rule, the one the chapter itself walks through first (Figure 23.7's top rule): the open
triad closing into a triangle. :func:`triad_closure_rule` reuses
:func:`graphrag.sna.motifs.transactional_mining` (ATL-41, imported and never edited) over
:func:`graphrag.sna.motifs.document_graphs` to get the two support counts a confidence needs, and
:func:`association_rule_scores` instantiates it against real entities: a document that already
names ``u`` and ``z`` together and ``z`` and ``v`` together, but not ``u`` and ``v``, proposes
``(u, v)`` at the rule's confidence. See "Deliberately not built" in the closing report for why
GERM's other extensions stop here.

**§23.1-§23.7's own caution, p. 334-335.** *"Every method we saw so far ... predict[s] exclusively
'old-old' links"*: a score is defined only for two nodes already in the graph a predictor reads,
and everything here inherits that. A :class:`graphrag.sna.guide.ReadingRule` says so, next to
Katz's beta bound and HRG's Monte Carlo cost.

**What is never here.** Nothing in this module writes to the graph or to the store. A score is a
claim about what the corpus would say if somebody looked, and :func:`confirming_evidence` is the
answer to "looked where": the passages that already exist and that a reader would check before
believing the hypothesis, resolved once here rather than separately for every scorer that could
have proposed the same pair -- what would confirm ``(u, v)`` is a property of the corpus, not of
the method that guessed it.

**Chapter 24, signed and multilayer prediction.** Two more questions than chapter 23 asks, each
answered in its own section below rather than folded into the machinery above, because each is a
genuinely different task.

*§24.1, Social Balance Theory.* Chapter 23 asks "will these two nodes connect"; §24.1 asks "given
that they already are connected -- one entity is named approvingly or disapprovingly alongside
another -- which sign does that connection carry". That is a classification of an *existing*
edge, not a claim about a non-edge, so it cannot run through :class:`graphrag.sna.experiment.
Holdout` (which exists precisely to keep a positive from ever being an edge of the training
graph): :func:`signed_graph`, :func:`balance_score` and :func:`predicted_sign` build and read the
signed graph :mod:`graphrag.sna.stances` produces (praise = +1, complaint = -1, its own
convention), and :class:`SignHoldout` / :func:`evaluate_sign_predictor` hide a share of the
*signs* rather than the edges, then measure how many balance theory gets back against a measured
coin-flip baseline. Status theory (p. 347) needs a *direction* on every edge -- "the user
originating the link" against the one receiving it -- and a co-mention is not directed: there is
no sender and no receiver of a passage naming two entities together, so status theory has nothing
to read here and is not built.

*§24.2, Generalized Multilayer Link Prediction.* One scorer, generalised: the *"Multilayer
Scores"* subsection (p. 351) says a neighbourhood score "should be re-weighted using layer
relevance", the quantity §9.1 defines as ``|N(u, l)| / |N(u)|`` -- the share of *one node's*
neighbours reached through layer ``l``. This module does not implement that quantity: it is
per node, and re-weighting a *pair* score by it would need to pick one of the pair's two nodes
to read it from, which the book does not say how to do. :func:`layer_correlation` instead reads
two layers of a :class:`graphrag.sna.layers.Multilayer` network as the vectors Atlas §24.4's
third exercise describes (`"Pearson correlation between layers"`) -- a *global*, per-layer-pair
number -- and :func:`cross_layer_common_neighbours` re-weights another layer's common-neighbour
score by that correlation before adding it to the target layer's own. This is this module's own
stand-in for the re-weighting idea p. 351 gestures at, not the layer relevance §9.1 names; both
docstrings say so. Only common neighbours is generalised this way; see "Deliberately not built"
in the closing report for the rest of §24.2's menu (hitting time, tensor factorization, GERM,
embeddings, metapath classifiers).
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from graphrag.models import EntityChunk, EntityMention
from graphrag.sna.cluster import HRGFit, hrg_fit
from graphrag.sna.experiment import Pair, ScoreTable
from graphrag.sna.layers import Multilayer
from graphrag.sna.matrices import adjacency, dense, eigenpairs
from graphrag.sna.measures import clustering, undirected_view
from graphrag.sna.motifs import Mining, canonical_form, transactional_mining
from graphrag.sna.stances import SignedPair

__all__ = [
    "DEFAULT_ASSOCIATION_MIN_SUPPORT",
    "DEFAULT_EVIDENCE_LIMIT",
    "DEFAULT_KATZ_BETA_FACTOR",
    "DEFAULT_SIGN_SHARE",
    "DEFAULT_TOP",
    "KATZ_MAX_NODES",
    "METHOD_NAMES",
    "METHOD_SECTIONS",
    "PREDICT_METHODS",
    "QUOTE",
    "AssociationRule",
    "Evidence",
    "Hypothesis",
    "PassageRef",
    "SignEvaluation",
    "SignHoldout",
    "TriangleBalance",
    "adamic_adar",
    "association_rule_scores",
    "balance_score",
    "classify_triangle",
    "common_neighbors",
    "confirming_evidence",
    "cross_layer_common_neighbours",
    "documents_without",
    "evaluate_sign_predictor",
    "hrg_scores",
    "hypotheses_payload",
    "jaccard",
    "katz",
    "layer_correlation",
    "layer_vector",
    "passages_by_entity",
    "predicted_sign",
    "random_sign_baseline",
    "rank_hypotheses",
    "render_hypotheses",
    "render_predicted_signs",
    "render_sign_evaluation",
    "resource_allocation",
    "sign_holdout",
    "sign_payload",
    "signed_graph",
    "triad_closure_rule",
]

#: How much of a passage's text a confirming excerpt shows. The book's own examples are one
#: sentence; this is a little more so a reader can see the two entities in context.
QUOTE = 220

#: Above this many nodes :func:`katz` refuses rather than pay the dense O(n^3) matrix inversion
#: p. 358's arithmetic warns about for the pair space itself: this package's own networks run to
#: a few thousand nodes (:mod:`graphrag.sna.matrices`'s own stated ceiling), and a few thousand is
#: where a dense inverse is still a matter of seconds rather than minutes.
KATZ_MAX_NODES = 3000

#: The default beta is this fraction of the convergence bound ``1 / lambda_max`` (§23.7). The
#: chapter leaves the exact value to the analyst -- *"a lower alpha penalizes long paths more"* --
#: and gives no rule of thumb, so half of the bound is this module's own choice: comfortably
#: convergent, and close enough to the bound that paths of length 3-4 still contribute visibly.
DEFAULT_KATZ_BETA_FACTOR = 0.5

#: §41.4 counts support in graphs, so this is at least 2: a rule fitted to one document has
#: nothing to generalise from, and its confidence would be 0 or 1 by construction.
DEFAULT_ASSOCIATION_MIN_SUPPORT = 2

#: How many passages :func:`confirming_evidence` cites per side of a hypothesis. Two is enough to
#: show the claim is not a fluke of one mis-tagged mention, and few enough to read at a glance.
DEFAULT_EVIDENCE_LIMIT = 2

#: How many hypotheses :func:`rank_hypotheses` returns by default -- the length of a list a
#: person actually reads, the same convention ``DEFAULT_K`` uses in :mod:`graphrag.sna.experiment`.
DEFAULT_TOP = 10

#: The default share of signed edges whose sign :func:`sign_holdout` hides (§24.1). Chapter 24
#: states no split of its own; this borrows §25.1's own smaller-than-half convention rather than
#: its 10%, because a signed co-mention graph is usually far sparser than an entity network and a
#: 10% holdout can leave too few hidden signs to read an accuracy off.
DEFAULT_SIGN_SHARE = 0.2

#: The methods ``sna predict --method`` accepts, in the chapter's own order.
PREDICT_METHODS: tuple[str, ...] = ("pa", "cn", "aa", "ra", "jaccard", "katz", "hrg", "rules")

METHOD_NAMES: dict[str, str] = {
    "pa": "preferential attachment",
    "cn": "common neighbours",
    "aa": "Adamic-Adar",
    "ra": "resource allocation",
    "jaccard": "Jaccard",
    "katz": "Katz",
    "hrg": "HRG",
    "rules": "association rule (triad closure)",
}

METHOD_SECTIONS: dict[str, str] = {
    "pa": "§23.1",
    "cn": "§23.2",
    "jaccard": "§23.2",
    "aa": "§23.3",
    "ra": "§23.4",
    "hrg": "§23.5",
    "rules": "§23.6",
    "katz": "§23.7",
}


# ------------------------------------------------------------------- §23.2-23.4 common neighbours


def _neighbor_sets(graph: nx.Graph) -> dict[str, frozenset[str]]:
    return {str(node): frozenset(str(n) for n in graph.neighbors(node)) for node in graph.nodes}


def common_neighbors(graph: nx.Graph) -> ScoreTable:
    """``|N_u ∩ N_v|`` (§23.2): *"the more common neighbors u and v have, the more triangles we
    can close with a single edge"*. A directed network is flattened first (§6.2): the chapter's
    own title is "for simple graphs" and every method in it reads an undirected neighbourhood.
    """
    flat, _ = undirected_view(graph)
    neighbours = _neighbor_sets(flat)

    def score(u: str, v: str) -> float:
        return float(len(neighbours.get(u, frozenset()) & neighbours.get(v, frozenset())))

    return ScoreTable(name="common neighbours", section="§23.2", graph=flat, score=score)


def jaccard(graph: nx.Graph) -> ScoreTable:
    """``|N_u ∩ N_v| / |N_u U N_v|`` (§23.2): common neighbours controlling for how many
    neighbours the two nodes have between them, so two hubs sharing ten neighbours out of a
    thousand score lower than two low-degree nodes sharing two out of three.

    0.0 when both node's neighbourhoods are empty, rather than the ``0/0`` that formula would
    otherwise divide.
    """
    flat, _ = undirected_view(graph)
    neighbours = _neighbor_sets(flat)

    def score(u: str, v: str) -> float:
        nu, nv = neighbours.get(u, frozenset()), neighbours.get(v, frozenset())
        union = nu | nv
        return float(len(nu & nv) / len(union)) if union else 0.0

    return ScoreTable(name="Jaccard", section="§23.2", graph=flat, score=score)


def adamic_adar(graph: nx.Graph) -> ScoreTable:
    """``sum_{z in N_u ∩ N_v} 1 / log(k_z)`` (§23.3): a common neighbour's contribution is
    discounted by the log of its own degree, because *"the hubs do not have enough bandwidth to
    make the introduction"*. A common neighbour of two distinct nodes always has degree at least
    2, so ``log(k_z)`` is never 0 here and the sum is never undefined.
    """
    flat, _ = undirected_view(graph)
    neighbours = _neighbor_sets(flat)
    degrees = {node: len(ns) for node, ns in neighbours.items()}

    def score(u: str, v: str) -> float:
        common = neighbours.get(u, frozenset()) & neighbours.get(v, frozenset())
        return float(sum(1.0 / math.log(degrees[z]) for z in common if degrees[z] > 1))

    return ScoreTable(name="Adamic-Adar", section="§23.3", graph=flat, score=score)


def resource_allocation(graph: nx.Graph) -> ScoreTable:
    """``sum_{z in N_u ∩ N_v} 1 / k_z`` (§23.4): Adamic-Adar's own idea with a linear rather than
    logarithmic discount, which *"punishes the high-degree common neighbors more heavily than
    Adamic-Adar"* -- the gap between ``k_z`` and ``log(k_z)`` is small for a low-degree common
    neighbour and large for a hub.
    """
    flat, _ = undirected_view(graph)
    neighbours = _neighbor_sets(flat)
    degrees = {node: len(ns) for node, ns in neighbours.items()}

    def score(u: str, v: str) -> float:
        common = neighbours.get(u, frozenset()) & neighbours.get(v, frozenset())
        return float(sum(1.0 / degrees[z] for z in common if degrees[z] > 0))

    return ScoreTable(name="resource allocation", section="§23.4", graph=flat, score=score)


# ------------------------------------------------------------------------------- §23.7 Katz


def _lambda_max(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    matrix, _ = adjacency(graph, weight="weight")
    values, _ = eigenpairs(matrix, k=1, largest=True)
    return float(values[0])


def katz(
    graph: nx.Graph, beta: float | None = None, *, max_nodes: int = KATZ_MAX_NODES
) -> ScoreTable:
    """The Katz score (§23.7): ``score(u, v) = sum_l beta^l |P^l_{u,v}|``, computed as the closed
    form ``(I - beta*A)^-1 - I`` rather than an infinite sum, exactly as Katz centrality is
    (§14.4). ``beta`` defaults to :data:`DEFAULT_KATZ_BETA_FACTOR` of ``1 / lambda_max``, the
    largest eigenvalue of the (flattened, weighted) adjacency matrix, and is refused above or at
    the bound -- *"if we choose a very small alpha, the Katz score is practically equivalent to
    counting common neighbors"*, and above the bound it does not converge at all.

    **Where this departs from the book.** ``|P^l_{u,v}|`` reads as "paths of length l", but a
    *simple*-path count is #P-hard to compute for general ``l``, and the closed form above --
    which is what every practical Katz implementation, including Katz's own 1953 centrality,
    actually computes -- sums *walks* (a walk may revisit a node). The two agree exactly at
    ``l = 1`` and ``l = 2`` (no walk of length 2 that is not a simple path exists between two
    distinct nodes with no self-loops) and diverge from ``l = 3`` on, when the walker can bounce
    off an edge and return. For a small ``beta`` the higher terms are negligible: on a length-2
    connection with nothing else joining the pair, the score is ``beta**2`` to within about
    ``beta`` of itself, which is why the known-answer test below uses a small one rather than
    the default.

    Refuses above ``max_nodes`` (:data:`KATZ_MAX_NODES`): the inverse is dense and O(n^3).
    """
    flat, _ = undirected_view(graph)
    if flat.number_of_nodes() > max_nodes:
        msg = (
            f"katz() refuses a network of {flat.number_of_nodes():,} nodes: the dense inverse "
            f"this needs is O(n^3), unusable above {max_nodes:,}. Take a backbone (ch. 27) or an "
            "ego network (§30.1) first."
        )
        raise ValueError(msg)
    lambda_max = _lambda_max(flat)
    bound = 1.0 / lambda_max if lambda_max > 0 else math.inf
    if beta is None:
        beta = DEFAULT_KATZ_BETA_FACTOR / lambda_max if lambda_max > 0 else DEFAULT_KATZ_BETA_FACTOR
    if beta <= 0:
        msg = f"beta must be positive, got {beta}"
        raise ValueError(msg)
    if lambda_max > 0 and beta >= bound:
        msg = (
            f"beta must be below 1/lambda_max ({bound:.6g}) to converge (Atlas §23.7); got "
            f"{beta:.6g} on a network whose largest eigenvalue is {lambda_max:.6g}"
        )
        raise ValueError(msg)

    order = sorted(flat.nodes)
    matrix, order = adjacency(flat, nodes=order, weight="weight")
    n = len(order)
    index = {node: i for i, node in enumerate(order)}
    walks = np.linalg.inv(np.eye(n) - beta * dense(matrix)) - np.eye(n)

    def score(u: str, v: str) -> float:
        i, j = index.get(u), index.get(v)
        if i is None or j is None:
            return 0.0
        return float(walks[i, j])

    return ScoreTable(name=f"Katz (beta={beta:.4g})", section="§23.7", graph=flat, score=score)


# --------------------------------------------------------------------------------- §23.5 HRG


def _hrg_parents(fit: HRGFit) -> dict[int, int]:
    parent: dict[int, int] = {}
    for node, (left, right) in fit.children.items():
        parent[left] = node
        parent[right] = node
    return parent


def hrg_scores(
    graph: nx.Graph,
    *,
    fit: HRGFit | None = None,
    samples: int | None = None,
    restarts: int = 5,
    seed: int | None = None,
) -> ScoreTable:
    """The Hierarchical Random Graph score (§23.5): ``score(u, v) = |E_c| / e(|E_c|)``, where
    ``c`` is the smallest dendrogram community holding both ``u`` and ``v`` (their lowest common
    ancestor in the tree :func:`graphrag.sna.cluster.hrg_fit` fits), ``|E_c|`` is that
    community's real edge count and ``e(|E_c|)`` is what a configuration-model null (§18.1)
    expects there: ``(sum of degree over c)^2 / (4m)``, the same closed form
    :func:`graphrag.sna.evaluate`'s modularity and
    :func:`graphrag.sna.backbone.noise_corrected_p`'s expected weight both use.

    At the dendrogram's root ``c`` is the whole network, whose expectation is always exactly its
    own edge count (``(2m)^2 / 4m = m``), so a score of 1.0 is the neutral reading the book
    describes for two nodes with nothing but the whole graph in common: *"if the nodes are far
    apart, the group containing both nodes might be just the entire network"*. A pair inside a
    dense, small community scores well above 1; 0.0 when the community has no possible edge at
    all (every member isolated), which cannot happen for two nodes that are themselves in the
    graph but is guarded rather than divided by zero.

    ``fit`` reuses an HRG already fitted to this graph (:func:`graphrag.sna.cluster.hrg_fit`,
    imported and never edited here); fitting one is a randomised MCMC search over dendrogram
    space and is the most expensive predictor in this module by a wide margin, so a caller
    scoring many pairs should fit once and pass it in rather than let this refit on every call.
    Without one, ``samples``, ``restarts`` and ``seed`` are passed straight through to
    :func:`graphrag.sna.cluster.hrg_fit`.
    """
    flat, _ = undirected_view(graph)
    if fit is None:
        fit = hrg_fit(flat, samples=samples, restarts=restarts, seed=seed)
    elif set(fit.order) - set(flat.nodes):
        msg = "fit was built from a node set this graph does not hold; refit it on this graph"
        raise ValueError(msg)

    degrees = dict(flat.degree())
    total_edges = flat.number_of_edges()
    parent = _hrg_parents(fit)
    index = {node: i for i, node in enumerate(fit.order)}

    def ancestors(leaf: int) -> list[int]:
        chain = [leaf]
        current = leaf
        while current in parent:
            current = parent[current]
            chain.append(current)
        return chain

    chains = {node: ancestors(i) for node, i in index.items()}

    members_cache: dict[int, tuple[str, ...]] = {}

    def members(node: int) -> tuple[str, ...]:
        if node not in members_cache:
            if node not in fit.children:
                members_cache[node] = (fit.order[node],)
            else:
                left, right = fit.children[node]
                members_cache[node] = members(left) + members(right)
        return members_cache[node]

    density_cache: dict[int, float] = {}

    def community_score(node: int) -> float:
        if node not in density_cache:
            group = members(node)
            internal = flat.subgraph(group).number_of_edges()
            degree_sum = sum(degrees.get(m, 0) for m in group)
            expected = degree_sum**2 / (4 * total_edges) if total_edges else 0.0
            density_cache[node] = internal / expected if expected > 0 else 0.0
        return density_cache[node]

    def lowest_common_ancestor(u: str, v: str) -> int | None:
        cu, cv = chains.get(u), chains.get(v)
        if cu is None or cv is None:
            return None
        seen = set(cv)
        for node in cu:
            if node in seen:
                return node
        return None

    def community(u: str, v: str) -> int | None:
        """The LCA, escalated past the degenerate case where it holds only ``u`` and ``v``
        themselves. A dendrogram node whose two branches are exactly the leaves ``u`` and ``v``
        has no edge to report but its own -- the one being asked about -- so its "density" is 0
        by construction whether or not the pair belongs together; that is not a reading of the
        network, it is the question restated. The book's own examples of a community ("a field
        of study", "a semi-clique") are always a group of more than two, so this looks one level
        further up the tree in that one case, where the group the book means actually starts.
        """
        node = lowest_common_ancestor(u, v)
        while node is not None and len(members(node)) <= 2 and node in parent:
            node = parent[node]
        return node

    def score(u: str, v: str) -> float:
        node = community(u, v)
        return community_score(node) if node is not None else 0.0

    return ScoreTable(name="HRG", section="§23.5", graph=flat, score=score)


# --------------------------------------------------------------------- §23.6 association rules


@dataclass(frozen=True)
class AssociationRule:
    """One GERM rule (§23.6): the open triad closing into a triangle, the one Figure 23.7 walks
    through first. ``confidence`` is *"the times you see [the antecedent], you'll also see [the
    consequent]"* -- ``support(triangle) / support(wedge)`` -- over ``documents`` document
    graphs, each one transaction (§41.4's definition of support: how many graphs hold at least
    one copy, however many copies each holds).
    """

    confidence: float
    support_antecedent: int
    support_consequent: int
    documents: int
    min_support: int
    transitivity: float
    """The network's own global clustering coefficient (§12.2): the null this confidence should
    be read against. A confidence at or below it says the rule closes triangles no faster than
    the network already does on its own, which is not evidence of a rule at all."""
    section: str = "§23.6"

    @property
    def beats_transitivity(self) -> bool:
        return not math.isnan(self.confidence) and self.confidence > self.transitivity


def _global_clustering(graph: nx.Graph) -> float:
    """:func:`graphrag.sna.measures.clustering` with ``kind="global"`` never returns per-node
    values; this is the narrowing that says so to the type checker."""
    value = clustering(graph, "global")
    if isinstance(value, dict):
        msg = "clustering('global') returned per-node values, which should never happen"
        raise TypeError(msg)
    return value


def triad_closure_rule(
    documents: Sequence[nx.Graph],
    network: nx.Graph,
    *,
    min_support: int = DEFAULT_ASSOCIATION_MIN_SUPPORT,
) -> AssociationRule:
    """Fit the one association rule this module instantiates: wedge (2 edges, 3 nodes) closes
    into a triangle (3 edges, 3 nodes). ``documents`` is
    :func:`graphrag.sna.motifs.document_graphs`'s per-document entity graphs -- the transactions
    -- and ``network`` is the pooled network the transitivity null is read from.

    ``confidence`` is ``nan`` when no document holds even one wedge under ``min_support``: there
    is nothing to generalise a rule from, which is a fact about the corpus rather than an error.
    """
    mining: Mining = transactional_mining(documents, min_support=min_support, max_size=3)
    wedge_key = canonical_form(nx.path_graph(3))
    triangle_key = canonical_form(nx.complete_graph(3))
    support = {pattern.canonical: pattern.support for pattern in mining.patterns}
    wedge_support = support.get(wedge_key, 0)
    triangle_support = support.get(triangle_key, 0)
    confidence = triangle_support / wedge_support if wedge_support else math.nan
    return AssociationRule(
        confidence=confidence,
        support_antecedent=wedge_support,
        support_consequent=triangle_support,
        documents=len(documents),
        min_support=min_support,
        transitivity=_global_clustering(network),
    )


def documents_without(documents: Sequence[nx.Graph], removed: frozenset[Pair]) -> list[nx.Graph]:
    """``documents`` with every pair in ``removed`` deleted from each document graph that holds
    it (§25.1's tenet, extended to a per-document transaction: a holdout that deletes an edge
    from the pooled network but leaves it standing in its document would let
    :func:`association_rule_scores` see the very edge the experiment held out).
    """
    if not removed:
        return list(documents)
    trimmed: list[nx.Graph] = []
    for document in documents:
        copy = document.copy()
        for u, v in list(copy.edges()):
            key = (u, v) if u <= v else (v, u)
            if key in removed:
                copy.remove_edge(u, v)
        trimmed.append(copy)
    return trimmed


def association_rule_scores(
    graph: nx.Graph,
    documents: Sequence[nx.Graph],
    *,
    min_support: int = DEFAULT_ASSOCIATION_MIN_SUPPORT,
) -> ScoreTable:
    """Instantiate :func:`triad_closure_rule` against real entities (§23.6).

    For every node ``z`` in every document and every pair of its document-neighbours ``u, v``
    not already joined in that document, ``(u, v)`` is a candidate the rule would close; it
    scores the rule's confidence unless ``(u, v)`` is already an edge of the pooled ``graph``
    (already a prediction nobody needs). The same pair found through more than one bridge or
    document keeps the rule's one confidence rather than summing -- confidence is a property of
    the rule, not of how many times it happened to fire.
    """
    flat, _ = undirected_view(graph)
    rule = triad_closure_rule(documents, flat, min_support=min_support)
    found: dict[Pair, float] = {}
    if not math.isnan(rule.confidence) and rule.confidence > 0:
        for document in documents:
            for z in document.nodes:
                neighbours = sorted(document.neighbors(z))
                for i, u in enumerate(neighbours):
                    for v in neighbours[i + 1 :]:
                        if document.has_edge(u, v):
                            continue
                        pair = (u, v) if u <= v else (v, u)
                        if flat.has_edge(*pair):
                            continue
                        found[pair] = max(found.get(pair, 0.0), rule.confidence)

    def score(u: str, v: str) -> float:
        pair = (u, v) if u <= v else (v, u)
        return found.get(pair, 0.0)

    return ScoreTable(
        name="association rule (triad closure)", section="§23.6", graph=flat, score=score
    )


# ---------------------------------------------------------------------- confirming passages


@dataclass(frozen=True)
class PassageRef:
    """One passage a reader could check, and what it is evidence for."""

    chunk_id: str
    doc_id: str
    role: str


@dataclass(frozen=True)
class Evidence:
    """What already exists in the corpus that bears on one hypothesis (not a book claim, so it
    carries no section: it is the corpus pointer every frame here promises instead of one).

    ``kind`` is one of three, checked in this order because each is stronger evidence than the
    next: ``"co-mentioned"`` -- the two nodes are already named together in at least one passage,
    just not often enough (or not matching some other filter) to have become an edge, which is
    the strongest thing a hypothesis can have going for it short of already being true;
    ``"bridge"`` -- no shared passage, but a third node is named with each of them, which is
    exactly the evidence the neighbourhood scorers (§23.2-23.4) build their number from;
    ``"none"`` -- neither exists, which is what a preferential-attachment or a long-path Katz
    hypothesis usually has, and the note says so plainly rather than padding the list.
    """

    kind: str
    bridge: str | None
    passages: tuple[PassageRef, ...]
    note: str


EntityRow = EntityChunk | EntityMention


def passages_by_entity(rows: Sequence[EntityRow]) -> dict[str, dict[str, str]]:
    """``entity_id -> {chunk_id: doc_id}`` from the rows :func:`confirming_evidence` reads.

    Built once per report and handed to every hypothesis: :func:`graphrag.sna.export.
    entity_passage_rows` is scoped to a persona's whole corpus, and calling it once outside the
    per-pair evidence lookup is what keeps a `--top 50` report from repeating that query fifty
    times.
    """
    by_entity: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_entity[row.entity_id][row.chunk_id] = row.doc_id
    return dict(by_entity)


def confirming_evidence(
    graph: nx.Graph,
    passages: Mapping[str, Mapping[str, str]],
    u: str,
    v: str,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> Evidence:
    """The passages a reader would check before believing ``(u, v)`` (see :class:`Evidence`).

    Scorer-independent by design: what would confirm a hypothesis is a property of the corpus,
    computed the same way whichever method proposed the pair.
    """
    pu, pv = passages.get(u, {}), passages.get(v, {})
    shared = sorted(set(pu) & set(pv))
    if shared:
        chosen = shared[:limit]
        return Evidence(
            kind="co-mentioned",
            bridge=None,
            passages=tuple(PassageRef(c, pu[c], f"{u} and {v} named together") for c in chosen),
            note=(
                f"{u} and {v} are already named together in {len(shared)} passage(s) that did "
                "not reach an edge (below --min-weight, or excluded by another filter): read "
                "these first, this is the strongest evidence a hypothesis here can have."
            ),
        )
    common = (
        sorted(set(graph.neighbors(u)) & set(graph.neighbors(v)))
        if (graph.has_node(u) and graph.has_node(v))
        else []
    )
    if common:
        bridge = min(common, key=lambda z: (graph.degree(z), z))
        left = sorted(set(passages.get(u, {})) & set(passages.get(bridge, {})))[:limit]
        right = sorted(set(passages.get(bridge, {})) & set(passages.get(v, {})))[:limit]
        refs = [PassageRef(c, pu[c], f"{u} and {bridge} named together") for c in left]
        refs += [
            PassageRef(c, passages[bridge][c], f"{bridge} and {v} named together") for c in right
        ]
        return Evidence(
            kind="bridge",
            bridge=bridge,
            passages=tuple(refs),
            note=(
                f"no passage names {u} and {v} together yet; {bridge} is named with both and is "
                "the specific evidence the neighbourhood scorers build their number from -- read "
                f"where {u} meets {bridge} and where {bridge} meets {v} to judge whether the "
                "introduction plausibly happens."
            ),
        )
    left_solo = sorted(pu)[:limit]
    right_solo = sorted(pv)[:limit]
    refs = [PassageRef(c, pu[c], f"{u} named") for c in left_solo]
    refs += [PassageRef(c, pv[c], f"{v} named") for c in right_solo]
    return Evidence(
        kind="none",
        bridge=None,
        passages=tuple(refs),
        note=(
            f"no shared and no bridging passage exists between {u} and {v} yet; these are only "
            "where each is named on its own, the weakest reading this report gives -- a "
            "preferential-attachment or a long-path hypothesis usually looks like this."
        ),
    )


# --------------------------------------------------------------------------------- hypotheses


@dataclass(frozen=True)
class Hypothesis:
    """One ranked prediction: a pair, a score, the method and section that produced it, and the
    :class:`Evidence` a reader would check before believing it. Never written to the graph."""

    pair: Pair
    score: float
    method: str
    section: str
    evidence: Evidence


def rank_hypotheses(
    scores: Mapping[Pair, float],
    graph: nx.Graph,
    passages: Mapping[str, Mapping[str, str]],
    *,
    method: str,
    section: str,
    top: int = DEFAULT_TOP,
    min_score: float = 0.0,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> tuple[Hypothesis, ...]:
    """The ``top`` highest-scoring pairs of ``scores`` above ``min_score``, each with its
    evidence. Ties are broken by the pair itself, so the ranking is deterministic.

    ``scores`` is usually a :class:`graphrag.sna.experiment.ScoreTable`, whose own ``__iter__``
    is the guard against enumerating an unreasonable pair space (§25.1, p. 358); this stops as
    soon as ``top`` items above ``min_score`` are held, via a bounded heap, rather than sorting
    the whole space.
    """
    if top < 1:
        msg = f"top must be at least 1, got {top}"
        raise ValueError(msg)
    heap: list[tuple[float, Pair]] = []
    for pair in scores:
        value = float(scores[pair])
        if value <= min_score:
            continue
        heapq.heappush(heap, (value, pair))
        if len(heap) > top:
            heapq.heappop(heap)
    ranked = sorted(heap, key=lambda row: (-row[0], row[1]))
    return tuple(
        Hypothesis(
            pair=pair,
            score=value,
            method=method,
            section=section,
            evidence=confirming_evidence(graph, passages, *pair, limit=evidence_limit),
        )
        for value, pair in ranked
    )


def _num(value: float, places: int = 4) -> str:
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.{places}f}"


def render_hypotheses(
    hypotheses: Sequence[Hypothesis],
    *,
    frame: str,
    null_model: str,
    title: str = "Link prediction hypotheses",
    excerpts: Mapping[str, str] | None = None,
) -> str:
    """The ranked list as markdown: frame, n, null model and the chapter, then one hypothesis
    per row with its evidence -- never a bare ranking (Wave 4's own rule, applied to a report
    this ticket adds rather than one of ATL-36's partitions).

    ``excerpts`` is ``chunk_id -> text``, already sanitised for inline printing
    (:func:`graphrag.textutil.sanitize_inline`) by the caller -- this module reaches no store
    itself, so the actual passage text is optional and supplied from outside it.
    """
    lines: list[str] = [
        f"## {title}",
        "",
        f"**Sampling frame.** {frame}",
        "",
        f"**n.** {len(hypotheses):,} hypothesis(es) ranked, out of the pair space the method "
        "iterated (see the evaluation above for how large that is and how it was sampled).",
        "",
        f"**Null model.** {null_model}",
        "",
        "**Implements.** Atlas ch. 23, *For Simple Graphs*. Nothing here is written to the "
        "graph: a score is a claim about what the corpus would say if somebody looked, and the "
        "passages beside each row are where to look.",
        "",
    ]
    if not hypotheses:
        lines += ["No pair scored above the threshold asked for.", ""]
        return "\n".join(lines)
    for rank, hypothesis in enumerate(hypotheses, start=1):
        u, v = hypothesis.pair
        lines.append(
            f"{rank}. **{u} -- {v}** ({hypothesis.method}, {hypothesis.section}): score "
            f"{_num(hypothesis.score)}. {hypothesis.evidence.note}"
        )
        for ref in hypothesis.evidence.passages:
            lines.append(f"   - {ref.role}: `{ref.chunk_id}` (document `{ref.doc_id}`)")
            text = (excerpts or {}).get(ref.chunk_id, "")
            if text:
                lines.append(f"     > {text}")
        lines.append("")
    return "\n".join(lines)


def hypotheses_payload(hypotheses: Sequence[Hypothesis]) -> list[dict[str, object]]:
    """The same ranking as JSON."""
    return [
        {
            "pair": list(hypothesis.pair),
            "score": hypothesis.score,
            "method": hypothesis.method,
            "section": hypothesis.section,
            "evidence": {
                "kind": hypothesis.evidence.kind,
                "bridge": hypothesis.evidence.bridge,
                "note": hypothesis.evidence.note,
                "passages": [
                    {"chunk_id": ref.chunk_id, "doc_id": ref.doc_id, "role": ref.role}
                    for ref in hypothesis.evidence.passages
                ],
            },
        }
        for hypothesis in hypotheses
    ]


# ------------------------------------------------------------------- §24.1 signed prediction


SIGN_HOLDOUT_FRAME = (
    "A sign holdout over the signed co-mention graph (Atlas §24.1): praise = +1, complaint = -1, "
    "the convention {source} states. {hidden:,} of {total:,} signed edge(s) ({share:.1%}) had "
    "their *sign* hidden at seed {seed} -- the edges themselves are never removed, since this "
    "asks 'which sign does this relationship carry', not 'does it exist' (chapter 23's question, "
    "which stays on ATL-23's Holdout). {ambiguous:,} pair(s) whose praise and complaint counts "
    "tied were dropped before this graph was built, unsigned rather than guessed."
)


@dataclass(frozen=True)
class TriangleBalance:
    """One signed triangle, classified against Figure 24.1's four types.

    ``negatives`` is how many of the three edges are negative; ``balanced`` is
    ``negatives % 2 == 0`` -- the book's own two equivalent statements of the rule, "an odd
    number of positive relationships" (p. 345) and the classical "even number of negative
    edges". (a) 0 negative and (b) 2 negative are balanced; (d) 1 negative is not; (c) 3
    negative (all-negative) reads as unbalanced under this classical rule, which is what
    ``balanced`` reports, though the book notes real networks sometimes treat it as neutral or
    balanced too (its own Tuscan campanilismo example) -- that reading is not separately modelled
    here, since it depends on context the signed graph does not carry.
    """

    signs: tuple[int, int, int]
    negatives: int
    balanced: bool


def classify_triangle(signs: Sequence[int]) -> TriangleBalance:
    """Classify a closed signed triangle by Figure 24.1's rule (§24.1).

    ``signs`` is the three edges' signs, each ``+1`` or ``-1``, in any order -- balance does not
    care which pair holds which sign, only how many are negative.
    """
    values = tuple(signs)
    if len(values) != 3 or any(s not in (1, -1) for s in values):
        msg = f"a triangle has exactly three signed edges, each +1 or -1; got {values!r}"
        raise ValueError(msg)
    negatives = sum(1 for s in values if s < 0)
    return TriangleBalance(signs=values, negatives=negatives, balanced=negatives % 2 == 0)


def signed_graph(pairs: Sequence[SignedPair]) -> nx.Graph:
    """Fold :mod:`graphrag.sna.stances`'s signed co-mentions into one signed graph (§24.1).

    One undirected edge per pair with a :meth:`graphrag.sna.stances.SignedPair.dominant_sign`,
    carrying that sign plus the raw ``praise``/``complaint`` counts it was read from. A pair with
    no dominant sign (its counts tied, including a tie at zero) contributes no edge; how many
    were dropped this way is on the graph as ``graph.graph["ambiguous"]``, printed in every frame
    that reads this graph rather than left silent.
    """
    graph = nx.Graph()
    ambiguous = 0
    for pair in pairs:
        sign = pair.dominant_sign
        if sign is None:
            if pair.count("praise") or pair.count("complaint"):
                ambiguous += 1
            continue
        graph.add_edge(
            pair.left,
            pair.right,
            sign=sign,
            praise=pair.count("praise"),
            complaint=pair.count("complaint"),
        )
    graph.graph["ambiguous"] = ambiguous
    return graph


def _sign_of(graph: nx.Graph, a: str, b: str) -> int | None:
    if not graph.has_edge(a, b):
        return None
    value = graph[a][b].get("sign")
    return int(value) if value is not None else None


def balance_score(graph: nx.Graph, u: str, v: str) -> float:
    """``score(u, v) = sum_{z in N(u) ∩ N(v)} sign(u, z) * sign(z, v)`` (§24.1).

    The balance-theory reading of triadic closure: a common neighbour whose two legs agree in
    sign votes for a positive third edge, one whose legs disagree votes for a negative one --
    always the sign that would make the closed triangle balanced under :func:`classify_triangle`
    (two positive legs close type (a); one of each closes type (b); two negative legs also close
    type (b) rather than the all-negative type (c), matching the book's own stated preference,
    p. 347). Only neighbours with a sign on *both* legs vote; an unsigned leg (the neighbour
    exists in the graph but that edge was never signed) is silently skipped rather than treated
    as zero, since zero would itself be a vote rather than an abstention.

    0.0 when ``u`` and ``v`` share no such neighbour, or when one is not in the graph at all --
    balance theory has nothing to say about the pair, which :func:`predicted_sign` reads as no
    prediction rather than as a neutral one.
    """
    if not (graph.has_node(u) and graph.has_node(v)):
        return 0.0
    total = 0.0
    for z in set(graph.neighbors(u)) & set(graph.neighbors(v)):
        su, sv = _sign_of(graph, u, z), _sign_of(graph, z, v)
        if su is not None and sv is not None:
            total += su * sv
    return total


def predicted_sign(score: float) -> int | None:
    """§24.1's decision rule: the sign of the aggregated vote.

    ``None`` on a tie, which includes both a genuine split (as many agreeing as disagreeing
    common neighbours) and no evidence at all (``score == 0.0``) -- balance theory abstains
    rather than guesses either way, and :func:`evaluate_sign_predictor` counts abstentions
    separately from wrong answers.
    """
    if score > 0:
        return 1
    if score < 0:
        return -1
    return None


@dataclass(frozen=True)
class SignHoldout:
    """One train/test split of a signed graph's *signs*, never its edges (§24.1).

    Chapter 24 states no evaluation of its own -- chapter 25's holdout is built for existence,
    not for a label on an edge that is already there -- so this is this module's own, in the
    same spirit: hide what would be unknown, predict it from what is left, check it against what
    it really was.
    """

    train: nx.Graph
    """Every signed edge, the held-out ones stripped of their ``sign`` (and ``praise``/
    ``complaint``) attributes but not removed -- a predictor reading this graph sees the edge
    exists, just not what it means, which is exactly the question being asked of it."""
    hidden: tuple[tuple[str, str, int], ...]
    """``(u, v, true sign)`` for every edge whose sign was hidden -- the answer key."""
    share: float
    seed: int | None
    frame: str = ""


def sign_holdout(
    graph: nx.Graph, *, share: float = DEFAULT_SIGN_SHARE, seed: int | None = None
) -> SignHoldout:
    """Hide the sign of a random share of ``graph``'s signed edges (§24.1, this module's own
    evaluation design -- see :class:`SignHoldout`).

    Refuses a graph with fewer than two signed edges: with one there is nothing left to read a
    common neighbour's sign from once it is hidden, and with zero there is nothing to hide.
    """
    edges = sorted((u, v) for u, v, sign in graph.edges(data="sign") if sign is not None)
    if len(edges) < 2:
        msg = f"need at least 2 signed edges to hold any out, got {len(edges)}"
        raise ValueError(msg)
    if not 0 < share < 1:
        msg = f"share must be strictly between 0 and 1, got {share}"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311 -- an experiment, not a secret
    order = list(edges)
    rng.shuffle(order)
    wanted = min(max(1, round(share * len(order))), len(order) - 1)
    hide = sorted(order[:wanted])

    train = graph.copy()
    hidden: list[tuple[str, str, int]] = []
    for u, v in hide:
        sign = int(train[u][v]["sign"])
        hidden.append((u, v, sign))
        del train[u][v]["sign"]
        train[u][v].pop("praise", None)
        train[u][v].pop("complaint", None)

    frame = SIGN_HOLDOUT_FRAME.format(
        source="stances.py's SignedPair.dominant_sign",
        hidden=len(hidden),
        total=len(edges),
        share=len(hidden) / len(edges),
        seed=seed,
        ambiguous=int(graph.graph.get("ambiguous", 0)),
    )
    return SignHoldout(
        train=train, hidden=tuple(hidden), share=len(hidden) / len(edges), seed=seed, frame=frame
    )


def random_sign_baseline(
    hidden: Sequence[tuple[str, str, int]], *, seed: int | None = None
) -> float:
    """A coin flip per hidden edge, measured rather than assumed (the convention §25.2 states for
    the same reason: *"the AUC is 0.5 for the random guess"*, here read as an accuracy).

    Its expectation is 0.5 whatever the corpus's praise/complaint balance, because the guess and
    the answer are independent; a deterministic function of ``seed`` and the pair, so reading the
    same split twice gives the same baseline. ``nan`` when there is nothing to guess.
    """
    if not hidden:
        return math.nan
    correct = 0
    for u, v, true_sign in hidden:
        guess = 1 if random.Random(f"{seed}|{u}|{v}").random() < 0.5 else -1  # noqa: S311
        correct += int(guess == true_sign)
    return correct / len(hidden)


@dataclass(frozen=True)
class SignEvaluation:
    """What one sign predictor did on one :class:`SignHoldout` (§24.1, this module's own
    reporting shape, built to answer to the same four questions Wave 4 holds every report to:
    sampling frame, n, null model, chapter -- printed by :func:`render_sign_evaluation`)."""

    method: str
    section: str
    correct: int
    incorrect: int
    abstained: int
    total: int
    accuracy: float
    """``correct / (correct + incorrect)``, over the edges the predictor actually voted on.
    ``nan`` when it voted on none."""
    coverage: float
    """Share of the hidden edges the predictor had any evidence for at all. ``nan`` when there
    was nothing hidden."""
    random_accuracy: float

    @property
    def beats_random(self) -> bool:
        return not math.isnan(self.accuracy) and self.accuracy > self.random_accuracy


def evaluate_sign_predictor(
    split: SignHoldout,
    scorer: Callable[[nx.Graph, str, str], float] = balance_score,
    *,
    name: str = "balance theory",
    seed: int | None = None,
) -> SignEvaluation:
    """Run one sign scorer on one :class:`SignHoldout` and measure it against a coin flip.

    ``scorer`` reads ``split.train`` -- which never carries a hidden sign -- and returns a score
    :func:`predicted_sign` turns into ``+1``, ``-1`` or an abstention. ``seed`` is the coin flip's
    own, defaulting to the split's seed so the two draws are reproducible together.
    """
    correct = incorrect = abstained = 0
    for u, v, true_sign in split.hidden:
        guess = predicted_sign(scorer(split.train, u, v))
        if guess is None:
            abstained += 1
        elif guess == true_sign:
            correct += 1
        else:
            incorrect += 1
    total = len(split.hidden)
    scored = correct + incorrect
    return SignEvaluation(
        method=name,
        section="§24.1",
        correct=correct,
        incorrect=incorrect,
        abstained=abstained,
        total=total,
        accuracy=correct / scored if scored else math.nan,
        coverage=scored / total if total else math.nan,
        random_accuracy=random_sign_baseline(
            split.hidden, seed=seed if seed is not None else split.seed
        ),
    )


def render_sign_evaluation(split: SignHoldout, evaluation: SignEvaluation) -> str:
    """The sign evaluation as markdown: frame, n, null model, chapter, then the table (Wave 4's
    own rule, applied here to a report this ticket adds rather than one of ATL-36's partitions).
    """
    scored = evaluation.correct + evaluation.incorrect
    lines = [
        "## Signed link prediction (balance theory)",
        "",
        f"**Sampling frame.** {split.frame}",
        "",
        f"**n.** {evaluation.total:,} signed edge(s) held out; {scored:,} scored "
        f"(coverage {_num(evaluation.coverage)}) and {evaluation.abstained:,} abstained on -- no "
        "common neighbour carried a sign on both legs, which is balance theory having nothing to "
        "vote with, not a neutral reading.",
        "",
        "**Null model.** A coin flip per hidden edge, measured rather than assumed: its accuracy "
        f"here is {_num(evaluation.random_accuracy)}, which stays near 0.5 whatever this corpus's "
        "praise/complaint balance is (chapter 24 states no null of its own; this is §25.2's own "
        "convention for a baseline, carried over).",
        "",
        "**Implements.** Atlas §24.1, Social Balance Theory: "
        "score(u, v) = sum over common neighbours z of sign(u, z) * sign(z, v), always the sign "
        "that would balance the closed triangle (Figure 24.1).",
        "",
        "| method | accuracy | coverage | random accuracy | correct | incorrect | abstained | § |",
        "|---|---|---|---|---|---|---|---|",
        f"| {evaluation.method} | {_num(evaluation.accuracy)} | "
        f"{_num(evaluation.coverage)} | {_num(evaluation.random_accuracy)} | "
        f"{evaluation.correct:,} | {evaluation.incorrect:,} | {evaluation.abstained:,} | "
        f"{evaluation.section} |",
        "",
        "**How to read this.** Accuracy is over the edges balance theory actually voted on; read "
        "it beside coverage, since abstaining on every hard pair and scoring perfectly on the "
        "easy ones is a different claim than scoring the same accuracy over everything hidden.",
    ]
    return "\n".join(lines) + "\n"


def sign_payload(evaluation: SignEvaluation) -> dict[str, object]:
    """The sign evaluation as JSON."""
    return {
        "chapter": 24,
        "section": evaluation.section,
        "method": evaluation.method,
        "correct": evaluation.correct,
        "incorrect": evaluation.incorrect,
        "abstained": evaluation.abstained,
        "total": evaluation.total,
        "accuracy": evaluation.accuracy,
        "coverage": evaluation.coverage,
        "random_accuracy": evaluation.random_accuracy,
        "beats_random": evaluation.beats_random,
    }


def render_predicted_signs(hypotheses: Sequence[Hypothesis], signed: nx.Graph) -> str:
    """The sign balance theory would assign each ranked hypothesis, if it appeared (§24.1).

    Existence and sign are printed as two separate readings rather than folded into one ranking,
    because they rest on different evidence: the ch. 23 score reads the whole (unsigned) entity
    network, this reads only the signed co-mention graph, which is usually a small fraction of
    it. A hypothesis with no signed common neighbour prints "no evidence" rather than a guess.
    """
    labels = {1: "positive (praise-like)", -1: "negative (complaint-like)", None: "no evidence"}
    lines = [
        "## Predicted signs for the ranked hypotheses",
        "",
        "Balance theory's reading (§24.1) of the sign each hypothesis above would carry if it "
        "appeared, from the signed co-mention graph alone -- not part of the existence ranking.",
        "",
        "| pair | balance score | predicted sign |",
        "|---|---|---|",
    ]
    for hypothesis in hypotheses:
        u, v = hypothesis.pair
        score = balance_score(signed, u, v)
        lines.append(f"| {u} -- {v} | {_num(score)} | {labels[predicted_sign(score)]} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------- §24.2 generalized multilayer prediction


def layer_vector(graph: nx.Graph, nodes: Sequence[str]) -> np.ndarray:
    """One layer as Atlas §24.4's third exercise reads it: *"each layer is a vector with an entry
    per edge"* -- here, one entry per possible pair of ``nodes`` (sorted for a stable order), the
    edge's weight if this layer joins the pair and 0.0 otherwise, so two layers over the same
    node set are directly comparable entry for entry.
    """
    order = sorted(nodes)
    return np.array(
        [
            float(graph[u][v].get("weight", 1)) if graph.has_edge(u, v) else 0.0
            for u, v in combinations(order, 2)
        ]
    )


def layer_correlation(a: nx.Graph, b: nx.Graph, nodes: Sequence[str]) -> float:
    """Pearson correlation of two layers' edge vectors (Atlas §24.4, Exercise 3).

    **Not** §9.1's *layer relevance* (p. 140): that quantity is ``|N(u, l)| / |N(u)|``, the share
    of *one node's* neighbours reached through layer ``l`` -- a per-node, per-layer number with
    no notion of a second layer to compare against. This function instead reads two whole layers
    as vectors and correlates them, a *global*, per-layer-pair number: two layers whose edges
    coincide more than chance would predict. :func:`cross_layer_common_neighbours` uses it as
    this module's own stand-in for the re-weighting the "Multilayer Scores" subsection (p. 351)
    asks for ("should be re-weighted using layer relevance") without saying how to apply a
    per-node quantity to a pair score -- see that function's docstring for the same caveat.

    ``nan`` when either layer has no variance over this pair space -- no edges at all, or every
    possible pair joined -- which is the correlation coefficient's own undefined case, not a
    reading of zero relevance.
    """
    xa, xb = layer_vector(a, nodes), layer_vector(b, nodes)
    if xa.std() == 0 or xb.std() == 0:
        return math.nan
    return float(np.corrcoef(xa, xb)[0, 1])


def cross_layer_common_neighbours(
    ml: Multilayer, target: str, train: nx.Graph | None = None
) -> ScoreTable:
    """Common neighbours on ``target``, blended with every other layer's own common-neighbour
    score in proportion to how correlated that layer is with ``target`` (§24.2, "Multilayer
    Scores", p. 351-352): ``score(u, v) = CN_target(u, v) + sum_{l != target, r(target, l) > 0}
    r(target, l) * CN_l(u, v)``.

    **Not §9.1's layer relevance.** p. 351 says a neighbourhood score "should be re-weighted
    using layer relevance", pointing back at §9.1's own definition, ``|N(u, l)| / |N(u)|`` -- how
    much of *one node's* neighbourhood a layer accounts for. That is a per-node number, and this
    is a *pair* score with two nodes to choose from, which the book does not resolve. ``r`` here
    is :func:`layer_correlation` instead, the Atlas §24.4 Exercise 3 quantity (a whole layer's
    edges against another's): this module's own stand-in for the re-weighting idea, not an
    implementation of §9.1's layer relevance.

    A layer negatively or not at all correlated with ``target`` (``r <= 0`` or undefined)
    contributes nothing, which is the book's own two-sided caution made concrete: *"a layer that
    carries no information about another leaves the multilayer score at the single-layer one"*
    is this formula at ``r = 0``, and a *negative* correlation is excluded rather than subtracted,
    since a layer whose edges anti-correlate with the target is evidence the pair is *unlikely*
    there, which this score does not attempt to read as a penalty (see "Where this departs from
    the book" for the analogous choice §24.1 does not have to make).

    ``train`` is ``target``'s own (possibly held-out) graph, matching the contract
    :func:`graphrag.sna.experiment.evaluate_predictor` calls a scorer with: the correlation is
    always computed against ``train``, never the untouched ``ml.graphs[target]``, so a positive
    that a holdout removed cannot leak into the correlation weights. The *other* layers are
    read from ``ml`` unchanged -- only the target layer is held out, exactly as §24.2 frames the
    task: predicting *"which layer"* a link appears in given the others as they stand.
    """
    if target not in ml.names:
        msg = f"{target!r} is not a layer of this network; layers are {', '.join(ml.names)}"
        raise ValueError(msg)
    base_graph = train if train is not None else ml.graphs[target]
    base = common_neighbors(base_graph)
    weights: dict[str, float] = {}
    for name in ml.names:
        if name == target:
            continue
        r = layer_correlation(base_graph, ml.graphs[name], ml.nodes)
        if not math.isnan(r) and r > 0:
            weights[name] = r
    others = {name: common_neighbors(ml.graphs[name]) for name in weights}

    def score(u: str, v: str) -> float:
        total = base.score(u, v)
        for name, r in weights.items():
            total += r * others[name].score(u, v)
        return float(total)

    used = ", ".join(f"{name} (r={weights[name]:.2f})" for name in sorted(weights)) or "none"
    return ScoreTable(
        name=f"cross-layer common neighbours ({target}; blended with {used})",
        section="§24.2",
        graph=base.graph,
        score=score,
    )
