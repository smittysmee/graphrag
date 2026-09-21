"""Run one analysis end to end and render it as a markdown note or a JSON payload.

The report is deliberately opinionated about what has to appear next to a result: how many
nodes it covers, where the network was sampled from, how the method was chosen, whether the
partition survives a different random seed, and whether it beats a degree-preserving null
model. Those are the things a reader needs in order to disagree with the conclusion.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag import __version__
from graphrag.extract.attributes import AttributeTable
from graphrag.graph.store import GraphStore
from graphrag.sna.assortativity import (
    DegreeCorrelations,
    NumericReport,
    degree_correlations,
    degree_correlations_payload,
    is_numeric,
    numeric_payload,
    numeric_report,
    render_degree_correlations,
    render_numeric,
)
from graphrag.sna.attributes import (
    ATTRIBUTE_NULLS,
    PERMUTATIONS,
    AttributeReport,
    analyse_attribute,
    attribute_payload,
    render_attribute,
)
from graphrag.sna.backbone import backbone as extract_backbone
from graphrag.sna.bipartite import (
    BIPARTITE_EVALUATION_NOTE,
    BipartiteReport,
    bipartite_payload,
    bipartite_report,
    render_bipartite,
)
from graphrag.sna.cluster import (
    Dendrogram,
    GMMResult,
    HRGFit,
    InfomapResult,
    KChoice,
    KMeansResult,
    LabelPropagationResult,
    LouvainResult,
    NullModelResult,
    SBMFit,
    choose_k_gmm,
    choose_k_kmeans,
    directed_communities,
    girvan_newman_dendrogram,
    gmm,
    hrg_communities,
    hrg_fit,
    infomap_communities,
    kmeans,
    label_propagation,
    louvain,
    louvain_levels,
    null_model_modularity,
    rewired_modularity,
    sbm_communities,
    spectral_embedding_full,
    walktrap,
)
from graphrag.sna.coreperiphery import (
    CorePeripheryReport,
    core_periphery_payload,
    core_periphery_report,
    render_core_periphery,
    two_mode_sides,
)
from graphrag.sna.degree import DegreeReport, degree_payload, degree_report, render_degree
from graphrag.sna.embed import Embedding, metapath2vec, node2vec
from graphrag.sna.evaluate import (
    PartitionScores,
    evaluate_partition,
    evaluation_payload,
    render_evaluation,
)
from graphrag.sna.export import describe
from graphrag.sna.generators import (
    AgainstRandom,
    against_random,
    against_random_payload,
    render_against_random,
)
from graphrag.sna.guide import ALWAYS, rationale
from graphrag.sna.hierarchy import (
    HierarchyReport,
    analyse_hierarchy,
    hierarchy_payload,
    render_hierarchy,
)
from graphrag.sna.layers import Multilayer, layer_table, render_layers
from graphrag.sna.measures import (
    CENTRALIZATION_NOTE,
    CHEAP_CENTRALITIES,
    REACH_HOPS,
    Centralization,
    DensityReport,
    brokers,
    centralities_for,
    centrality,
    centrality_meaning,
    centralization_table,
    core_shells,
    density_payload,
    density_report,
    directed_notes,
    render_density,
    summary,
    top_n,
    undirected_view,
)
from graphrag.sna.null import configuration
from graphrag.sna.paths import PathReport, analyse_paths, paths_payload, render_paths
from graphrag.sna.roles import (
    ATLAS_ROLE,
    MAX_SIMILARITY_NODES,
    ROLE_NAMES,
    RolePair,
    RoleTable,
    guimera_amaral,
    same_role_never_cooccur,
    structural_similarity,
)
from graphrag.sna.sampling import BIASES, sample
from graphrag.sna.stats import kendall, spearman
from graphrag.sna.ties import TieReport, render_ties, tie_payload, tie_strength_curve
from graphrag.sna.uncertain import (
    SAMPLES,
    Evidence,
    Uncertainty,
    evidence_report,
    render_evidence,
    render_uncertain,
    uncertain_payload,
    uncertainty_report,
)

Method = Literal[
    "louvain",
    "kmeans",
    "gmm",
    "louvain-levels",
    "girvan-newman",
    "hrg",
    "directed",
    "sbm",
    "infomap",
    "walktrap",
    "label-propagation",
    "brim",
    "bilouvain",
    "neighbor-similarity",
]
Features = Literal["spectral", "embedding", "node2vec", "metapath2vec"]
METHODS: tuple[Method, ...] = (
    "louvain",
    "kmeans",
    "gmm",
    "louvain-levels",
    "girvan-newman",
    "hrg",
    "directed",
    "sbm",
    "infomap",
    "walktrap",
    "label-propagation",
    "brim",
    "bilouvain",
    "neighbor-similarity",
)
FEATURE_KINDS: tuple[Features, ...] = ("spectral", "embedding", "node2vec", "metapath2vec")

#: Methods whose groups are read as communities (edges decide membership) rather than clusters
#: (a vector distance does): everything except the two vector methods, ``kmeans`` and ``gmm``.
COMMUNITY_METHODS: frozenset[str] = frozenset(set(METHODS) - {"kmeans", "gmm"})

#: §39.3-39.4's methods: they group both modes of a two-mode network together, directly on its
#: incidence, and never build the feature matrix ``kmeans``/``gmm`` need or the standard
#: modularity null ``louvain`` uses.
BIPARTITE_METHODS: tuple[Method, ...] = ("brim", "bilouvain", "neighbor-similarity")

TOP_N = 20
TOP_MEMBERS = 8

FEATURE_MEANING: dict[str, str] = {
    "spectral": (
        "the graph's own structure, embedded with the normalised Laplacian. Groups mean "
        "'connected in the same way', not 'about the same thing'."
    ),
    "embedding": (
        "the mean text embedding of the passages behind each node. Groups mean 'about the same "
        "thing', and two nodes can land together without sharing a single edge."
    ),
    "node2vec": (
        "a numpy skip-gram fit over biased random walks (ch. 43): --p and --q trade off "
        "breadth- and depth-first exploration, and 1.0/1.0 is DeepWalk. Groups mean 'visited "
        "together by a random walk', which tracks community membership at p=q=1 and drifts "
        "toward structural equivalence as q falls. Transductive (§43.5): retrained from scratch "
        "on this run's graph, never carried over from a previous one."
    ),
    "metapath2vec": (
        "a numpy skip-gram over speaker-entity-speaker walks on the two-mode network (§43.3), so "
        "a speaker's vector reflects the entities its passages mention rather than every node "
        "the plain walk's degree bias would put it near. Only valid on --network "
        "speakers-entities without --project, where every node carries the network's own mode."
    ),
}


# ----------------------------------------------------- ranking stability and centralization

#: How :func:`ranking_stability` gets the second network it compares the ranking against. None
#: of them is a new resampler: ``bootstrap-edges`` is chapter 29's edge sampling
#: (:func:`graphrag.sna.sampling.sample` with ``ties``), ``configuration`` is §19.1's
#: degree-preserving rewiring (:func:`graphrag.sna.null.configuration`) and ``backbone`` is
#: chapter 27's filter (:func:`graphrag.sna.backbone.backbone`). They ask three different
#: questions and :func:`_resample_note` says which one was asked.
RESAMPLE_METHODS: tuple[str, ...] = ("bootstrap-edges", "configuration", "backbone")

#: The share of the nodes ``bootstrap-edges`` keeps in each resample. Chapter 14 defines no
#: resampling at all, so this is a convention: a fifth of the network withheld is enough to move
#: a rank that rests on one or two passages and not enough to turn the network into a different
#: one.
BOOTSTRAP_SHARE = 0.8

#: A node that stays in the top k of at least this share of the resamples is reported as stable.
#: The threshold is the same 0.9 the Louvain stability reading uses, for the same reason: below
#: it, a name is being quoted that a different draw of the corpus would not have printed.
STABLE_SHARE = 0.9

#: How many resamples the report takes, whatever ``--samples`` says. Each one recomputes the
#: whole centrality battery, so this is the one place where the report's null-model budget would
#: multiply an already expensive measurement.
RANKING_SAMPLES = 20

#: Above this many nodes the section checks only the rankings in :data:`LINEAR_KINDS`. Both
#: halves of it recompute the battery -- once on the binary view §14.8 needs, and once per
#: resample -- and betweenness, closeness and harmonic are all-pairs shortest paths while the
#: eigenvector and HITS are dense eigendecompositions. A bigger network can still be asked, one
#: ranking at a time, with :func:`ranking_stability`, or sampled (ch. 29) or backboned (ch. 27)
#: down to this size first.
RANKING_MAX_NODES = 500

#: The rankings whose cost is linear in the edges, so that recomputing one of them twenty times
#: costs about what computing the battery once costs. The rest are the all-pairs and dense
#: eigenvector ones.
LINEAR_KINDS: frozenset[str] = frozenset(
    {
        "degree",
        "weighted_degree",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "pagerank",
        "reach",
        "coreness",
    }
)

#: A rank correlation needs at least this many nodes present in both rankings; below it
#: :func:`graphrag.sna.stats.spearman` refuses, and so does this.
MIN_RANKING_NODES = 3


@dataclass(frozen=True)
class RankingStability:
    """How much of one centrality ranking survives resampling the network it was measured on.

    Chapter 14 ranks nodes and never asks this question; the chapter it belongs to is the one
    it is measured against (§29.1's samplers, §19.1's rewiring, ch. 27's backbones). The two
    halves answer different things: the rank correlations say whether the *whole* ordering
    holds, and the top-k retention says whether the handful of names the report actually printed
    hold, which is the part a reader quotes.
    """

    kind: str
    method: str
    #: How many resamples produced a usable ranking, which can be fewer than were asked for.
    samples: int
    requested: int
    top_k: int
    #: Mean Spearman correlation between the observed ranking and each resample's, over the
    #: nodes present in both, with the 2.5th and 97.5th percentiles of the same values.
    spearman_mean: float
    spearman_low: float
    spearman_high: float
    #: The same in Kendall's tau-b, which runs smaller on the same data (§3.4) and is not on the
    #: same scale as the Spearman beside it.
    kendall_mean: float
    kendall_low: float
    kendall_high: float
    #: Mean share of the observed top k that is still in the resample's top k.
    overlap_mean: float
    #: Per node of the observed top k, the share of resamples that still ranked it there.
    retention: dict[str, float]
    #: Those nodes split at :data:`STABLE_SHARE`, in the observed ranking's order.
    stable: list[str]
    unstable: list[str]
    #: Display names for the nodes above, by the report's usual rule (``label``, ``name``, id).
    labels: dict[str, str]
    #: How many resamples yielded a correlation: a resample whose ranking is constant (every
    #: node in the same core, say) has no correlation to give and is counted only in the overlap.
    correlations: int
    note: str


@dataclass(frozen=True)
class RankingReport:
    """Chapter 14 read as a whole: how centralized this network is, and how firm its ranks are."""

    nodes: int
    edges: int
    frame: str
    #: The resampling method the stability half used, one of :data:`RESAMPLE_METHODS`.
    method: str
    #: How many resamples were asked for (:data:`RANKING_SAMPLES` or fewer).
    samples: int
    top_k: int
    seed: int | None
    centralization: list[Centralization]
    stability: list[RankingStability]
    #: Rankings the budget left unchecked, and therefore missing from both tables above.
    skipped: list[str]
    #: The k-shell sizes (§14.7), which say how coarse the coreness ranking is.
    shells: dict[int, int]
    note: str


def ranking_stability(
    graph: nx.Graph,
    kind: str,
    *,
    samples: int = RANKING_SAMPLES,
    seed: int | None = None,
    method: str = "bootstrap-edges",
    top_k: int = TOP_N,
    observed: dict[str, float] | None = None,
    backbone_method: str = "noise-corrected",
) -> RankingStability:
    """Whether a centrality ranking would survive a different draw of the same corpus.

    Chapter 14 hands back an ordering and says nothing about its error bars, and a ranking
    printed without them invites a reader to compare rank 7 with rank 9. This measures the
    ranking against ``samples`` resamples of the same network: the mean and the 95% interval of
    the Spearman and Kendall correlation between the observed ordering and the resample's
    (§3.4's two rank correlations, over the nodes present in both), and how often each of the
    observed top ``top_k`` was still in the resample's top ``top_k``.

    ``method`` says what the resample *is*, and the three do not answer the same question:

    ``bootstrap-edges``
        chapter 29's edge sampling: draw edges uniformly until :data:`BOOTSTRAP_SHARE` of the
        nodes are held and induce the subgraph on them (``sample(graph, "ties", ...)``). This is
        the measurement question -- would this rank still be printed if the corpus had recorded
        a fifth less -- and it is the default for that reason. It inherits that sampler's bias,
        which the note repeats verbatim from :data:`graphrag.sna.sampling.BIASES`.
    ``configuration``
        §19.1's degree-preserving rewiring. It holds every node's degree fixed, so the degree
        rankings correlate with themselves at 1.0 **by construction** and only the path- and
        walk-based rankings say anything here. What it answers is not "would this rank survive"
        but "is this rank anything more than the degree".
    ``backbone``
        chapter 27's filter, run once. It is deterministic, so there is exactly one comparison
        however many samples are asked for, and it answers a third question: does the ranking
        rest on the edges that survive a null for edge weight, or on the mass of one-passage
        ties the backbone removes.

    ``observed`` lets a caller that has already computed ``centrality(graph, kind)`` -- the
    report has -- pass it in rather than paying for it twice.

    A resample that cannot be measured is skipped and ``samples`` reports how many were usable:
    a rewiring that no edge swap accepts, a sample too small for a correlation, a ranking that
    came back constant. Zero usable resamples is a real outcome on a network with no edges, and
    the note says so rather than the numbers reading as zeros.

    Raises ``ValueError`` on an unknown method, a ``top_k`` below 1, or a ``kind`` this graph
    does not support (:func:`graphrag.sna.measures.centralities_for` lists them).
    """
    if method not in RESAMPLE_METHODS:
        msg = f"method must be one of {', '.join(RESAMPLE_METHODS)}, got {method!r}"
        raise ValueError(msg)
    if top_k < 1:
        msg = f"top_k must be at least 1, got {top_k}"
        raise ValueError(msg)
    scores = observed if observed is not None else centrality(graph, kind)
    ranked = [node for node, _ in top_n(scores, top_k)]
    spearmans: list[float] = []
    kendalls: list[float] = []
    overlaps: list[float] = []
    kept: Counter[str] = Counter()
    used = 0
    for resample in _resamples(graph, method, samples, seed, backbone_method):
        values = centrality(resample, kind)
        shared = [node for node in graph.nodes if node in values]
        if len(shared) < MIN_RANKING_NODES:
            continue
        left = [scores[node] for node in shared]
        right = [values[node] for node in shared]
        try:
            spearmans.append(spearman(left, right).coefficient)
            kendalls.append(kendall(left, right).coefficient)
        except ValueError:
            pass  # a constant ranking has no correlation; the overlap below still counts
        top = {node for node, _ in top_n(values, top_k)}
        overlaps.append(len(top & set(ranked)) / len(ranked))
        kept.update(node for node in ranked if node in top)
        used += 1
    retention = {node: (kept[node] / used if used else 0.0) for node in ranked}
    low, high = _interval(spearmans)
    tau_low, tau_high = _interval(kendalls)
    return RankingStability(
        kind=kind,
        method=method,
        samples=used,
        requested=1 if method == "backbone" else samples,
        top_k=top_k,
        spearman_mean=float(np.mean(spearmans)) if spearmans else 0.0,
        spearman_low=low,
        spearman_high=high,
        kendall_mean=float(np.mean(kendalls)) if kendalls else 0.0,
        kendall_low=tau_low,
        kendall_high=tau_high,
        overlap_mean=float(np.mean(overlaps)) if overlaps else 0.0,
        retention=retention,
        stable=[node for node in ranked if retention[node] >= STABLE_SHARE],
        unstable=[node for node in ranked if retention[node] < STABLE_SHARE],
        labels={str(node): _node_label(graph, node) for node in ranked},
        correlations=len(spearmans),
        note=_resample_note(graph, method, used, backbone_method),
    )


def _resamples(
    graph: nx.Graph, method: str, samples: int, seed: int | None, backbone_method: str
) -> Iterator[nx.Graph]:
    """The resampled networks, from the module that owns each resampler. See the three methods."""
    if method == "configuration":
        yield from configuration(graph, samples, seed=seed)
        return
    if method == "backbone":
        yield extract_backbone(graph, backbone_method, keep_isolates=True, seed=seed)
        return
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    size = max(MIN_RANKING_NODES, round(BOOTSTRAP_SHARE * graph.number_of_nodes()))
    for _ in range(samples):
        yield sample(graph, "ties", size=size, seed=rng.randrange(1_000_000))


def _interval(values: Sequence[float]) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles: the interval 95% of the resamples landed in."""
    if not values:
        return 0.0, 0.0
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _resample_note(graph: nx.Graph, method: str, used: int, backbone_method: str) -> str:
    """What the resamples were, and what the numbers therefore mean."""
    if used == 0:
        return (
            f"No usable resample: the {method} resampling of a network with "
            f"{graph.number_of_edges():,} edge(s) produced nothing that could be ranked, so the "
            "correlations below are zeros for want of a comparison rather than as a finding."
        )
    if method == "configuration":
        return (
            f"{used} degree-preserving rewiring(s) of the observed network (§19.1's edge swap, "
            "`sna.null.configuration`). It holds every node's degree fixed, so the degree "
            "rankings correlate at 1.0 with themselves by construction and only the path- and "
            "walk-based rankings are informative here: what this asks is whether a rank is "
            "anything beyond the degree, not whether it would survive a different corpus."
        )
    if method == "backbone":
        return (
            f"One comparison against the {backbone_method} backbone of the same network "
            "(ch. 27), which is deterministic, so there is no interval to report: the "
            "correlation is between this ranking and the ranking over the edges that survive a "
            "null for edge weight."
        )
    size = max(MIN_RANKING_NODES, round(BOOTSTRAP_SHARE * graph.number_of_nodes()))
    return (
        f"{used} resample(s), each drawing edges uniformly at random until {size:,} of the "
        f"{graph.number_of_nodes():,} nodes are held and inducing the subgraph on them (§29.1's "
        f"edge sampling, `sna.sampling.sample(..., 'ties')`). It asks what survives seeing "
        f"{BOOTSTRAP_SHARE:.0%} of the corpus. {BIASES['ties'].sentence}"
    )


def ranking_report(
    graph: nx.Graph,
    *,
    scores: dict[str, dict[str, float]] | None = None,
    samples: int = RANKING_SAMPLES,
    seed: int | None = None,
    top_k: int = TOP_N,
    method: str = "bootstrap-edges",
) -> RankingReport:
    """§14.8 over every ranking, plus how firm each ranking is, in one pass.

    ``scores`` is the battery the report already computed (kind -> node -> score), passed in so
    the stability half does not recompute the observed rankings it is comparing against.

    The section is budgeted, and says so: above :data:`RANKING_MAX_NODES` nodes only the
    rankings in :data:`LINEAR_KINDS` are checked, because every other one costs all-pairs
    shortest paths or a dense eigendecomposition and this section would pay for it
    ``samples + 1`` times. The rest are named in ``skipped`` rather than dropped silently.
    """
    kinds = list(centralities_for(graph))
    budgeted = graph.number_of_nodes() > RANKING_MAX_NODES
    checked = [kind for kind in kinds if not budgeted or kind in LINEAR_KINDS]
    skipped = [kind for kind in kinds if kind not in checked]
    taken = max(1, min(samples, RANKING_SAMPLES))
    observed = scores or {}
    return RankingReport(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        frame=str(graph.graph.get("frame", "")),
        method=method,
        samples=taken,
        top_k=top_k,
        seed=seed,
        centralization=centralization_table(graph, checked),
        stability=[
            ranking_stability(
                graph,
                kind,
                samples=taken,
                seed=seed,
                method=method,
                top_k=top_k,
                observed=observed.get(kind),
            )
            for kind in checked
        ],
        skipped=skipped,
        shells=core_shells(graph),
        note=_ranking_budget_note(graph, skipped, budgeted=budgeted),
    )


def _ranking_budget_note(graph: nx.Graph, skipped: Sequence[str], *, budgeted: bool) -> str:
    """Why some rankings are missing from this section, in the reader's terms."""
    if not budgeted:
        return (
            f"Every ranking the report printed is checked: this network has "
            f"{graph.number_of_nodes():,} nodes, inside the {RANKING_MAX_NODES:,}-node budget "
            "this section keeps."
        )
    return (
        f"This network has {graph.number_of_nodes():,} nodes, above the "
        f"{RANKING_MAX_NODES:,}-node budget of this section, so "
        f"{', '.join(skipped)} are left out of both tables: each of them costs all-pairs "
        "shortest paths or a dense eigendecomposition, and this section would pay for it once "
        "per resample. Ask for one of them directly with `sna.analysis.ranking_stability`, or "
        "sample (ch. 29) or backbone (ch. 27) the network down first."
    )


def _node_label(graph: nx.Graph, node: str) -> str:
    """A node's display name: its ``label``, its ``name``, or its id. One rule, one place."""
    data = graph.nodes.get(node, {})
    return str(data.get("label") or data.get("name") or node)


def render_ranking(report: RankingReport) -> list[str]:
    """The ``## Ranking stability`` section: §14.8's one number, and error bars for §14.1-§14.7.

    Two tables and the sentence that stops each being misread. The centralization table is about
    the network and says nothing about any node in it; the stability table is about the names
    printed in the centrality tables above, and says which of them a different draw of the same
    corpus would still have printed.
    """
    frame = f" Frame: {report.frame}" if report.frame else ""
    seed = f" Seed: {report.seed}." if report.seed is not None else " No seed fixed."
    lines = [
        "## Ranking stability",
        "",
        f"*Atlas ch. 14 (§14.8 centralization, §14.1-§14.7 rankings), over the whole network: "
        f"n = {report.nodes:,} nodes and {report.edges:,} edges.{frame} Resampling: "
        f"{report.samples} x `{report.method}`, top-{report.top_k} overlap.{seed} "
        f"{report.note}*",
        "",
        "### Centralization (§14.8)",
        "",
        CENTRALIZATION_NOTE,
        "",
    ]
    lines += _table(
        ["centrality", "centralization", "observed spread", "star (the maximum)"],
        [
            [
                row.kind,
                "undefined" if row.value is None else _num(row.value),
                "—" if row.value is None else _num(row.observed),
                "—" if row.value is None else _num(row.star),
            ]
            for row in report.centralization
        ],
    )
    undefined = [row for row in report.centralization if row.value is None]
    lines += [f"- **{row.kind}** {row.note}" for row in undefined]
    weighted = [row for row in report.centralization if row.value is not None and row.note]
    lines += [f"- **{row.kind}** {row.note}" for row in weighted]
    if undefined or weighted:
        lines.append("")
    if report.shells:
        shells = ", ".join(f"{count:,} in shell {shell}" for shell, count in report.shells.items())
        lines += [
            f"- **Shells (§14.7)** {len(report.shells)} distinct core number(s) over "
            f"{sum(report.shells.values()):,} nodes: {shells}. A coreness is a floor rather than "
            "a rank -- every node in a shell holds the same value -- so read it as membership of "
            "the dense middle, never as a position.",
            "",
        ]
    if not report.stability:
        return [*lines, "No ranking was checked for stability on this network.", ""]
    lines += [
        f"### Does the top {report.top_k} hold? (§14.1-§14.7 rankings, resampled)",
        "",
        report.stability[0].note,
        "",
    ]
    lines += _table(
        ["ranking", "Spearman (95%)", "Kendall", f"top-{report.top_k} overlap", "resamples"],
        [
            [
                row.kind,
                f"{_num(row.spearman_mean)} ({_num(row.spearman_low)} to "
                f"{_num(row.spearman_high)})",
                _num(row.kendall_mean),
                f"{row.overlap_mean:.0%}",
                str(row.samples),
            ]
            for row in report.stability
        ],
    )
    lines += [_stability_line(row) for row in report.stability]
    lines += [
        "",
        "Spearman and Kendall are not on the same scale and must not be compared with each "
        "other (§3.4): tau runs smaller on the same data. A ranking whose correlation is high "
        "while its top-k overlap is low is one whose *tail* is stable and whose leaders swap, "
        "which is the opposite of what a report's reader needs.",
        "",
    ]
    return lines


def _stability_line(row: RankingStability) -> str:
    """Which of one ranking's printed names held, and which are within the noise."""
    if row.samples == 0:
        return f"- **{row.kind}** not checked: {row.note}"
    stable = _named(row.stable, row.labels)
    unstable = _named(row.unstable, row.labels)
    return (
        f"- **{row.kind}** {len(row.stable)} of the printed {len(row.retention)} stayed in the "
        f"top {row.top_k} of at least {STABLE_SHARE:.0%} of the resamples"
        + (f": {stable}" if row.stable else "")
        + (
            f". Within the noise, and not to be quoted as ranks: {unstable}."
            if row.unstable
            else ". Every printed name held."
        )
    )


def _named(nodes: Sequence[str], labels: dict[str, str]) -> str:
    """Up to :data:`TOP_MEMBERS` display names, then a count of the rest."""
    shown = ", ".join(labels.get(str(node), str(node)) for node in nodes[:TOP_MEMBERS])
    more = len(nodes) - TOP_MEMBERS
    return shown + (f", +{more:,} more" if more > 0 else "")


def ranking_payload(report: RankingReport) -> dict[str, Any]:
    """The same section as plain JSON-able data."""
    return {
        "chapter": "Atlas ch. 14",
        "nodes": report.nodes,
        "edges": report.edges,
        "frame": report.frame,
        "null_model": f"{report.samples} x {report.method} resamples of the observed network",
        "seed": report.seed,
        "top_k": report.top_k,
        "reach_hops": REACH_HOPS,
        "centralization_note": CENTRALIZATION_NOTE,
        "centralization": [
            {
                "kind": row.kind,
                "value": row.value,
                "observed": row.observed,
                "star": row.star,
                "note": row.note,
            }
            for row in report.centralization
        ],
        "shells": {str(shell): count for shell, count in report.shells.items()},
        "stability": [
            {
                "kind": row.kind,
                "method": row.method,
                "samples": row.samples,
                "requested": row.requested,
                "top_k": row.top_k,
                "spearman": {
                    "mean": row.spearman_mean,
                    "low": row.spearman_low,
                    "high": row.spearman_high,
                },
                "kendall": {
                    "mean": row.kendall_mean,
                    "low": row.kendall_low,
                    "high": row.kendall_high,
                },
                "top_k_overlap": row.overlap_mean,
                "retention": row.retention,
                "stable": row.stable,
                "unstable": row.unstable,
                "labels": row.labels,
                "correlations": row.correlations,
                "note": row.note,
            }
            for row in report.stability
        ],
        "skipped": report.skipped,
        "note": report.note,
    }


@dataclass
class Analysis:
    """Everything one ``graphrag sna analyze`` run produced."""

    persona_id: str
    network: str
    method: str
    graph: nx.Graph
    summary: dict[str, float]
    centralities: dict[str, list[tuple[str, float]]]
    groups: list[list[str]]
    group_noun: str
    rationale: str
    seed: int | None
    generated_at: str
    features: str | None = None
    feature_dims: int = 0
    brokers: list[tuple[str, float]] = field(default_factory=list)
    #: Set by ``--method louvain`` and, since it is the same algorithm read on the arrows
    #: instead of a flattening (§37.3), by ``--method directed`` too.
    louvain_result: LouvainResult | None = None
    kmeans_result: KMeansResult | None = None
    gmm_result: GMMResult | None = None
    choice: KChoice | None = None
    #: Set by ``--method louvain-levels``, ``--method girvan-newman`` and ``--method walktrap``
    #: (§37.1, §35.2): each is a merge or split sequence cut by its own modularity profile, so one
    #: field holds whichever of the three ran.
    dendrogram: Dendrogram | None = None
    #: Set by ``--method hrg`` (§37.2): the fitted Hierarchical Random Graph itself, read into
    #: ``groups`` by :func:`graphrag.sna.cluster.hrg_communities` at its default cut.
    hrg: HRGFit | None = None
    #: Set by ``--method sbm`` (§35.1): the blockmodel fit, plain or degree-corrected, and which
    #: implementation produced it.
    sbm: SBMFit | None = None
    #: Set by ``--method infomap`` (§35.2): the map-equation search, its code length and stability.
    infomap: InfomapResult | None = None
    #: Set by ``--method label-propagation`` (§35.3).
    label_propagation_result: LabelPropagationResult | None = None
    null_model: NullModelResult | None = None
    #: Chapter 17's three comparisons -- clustered, short, broad -- against the closed forms of
    #: chapter 16 and a degree-preserving null. Present on every non-empty network, because
    #: "this network is clustered" is a comparison and never a number on its own.
    against_random: AgainstRandom | None = None
    attribute: AttributeReport | None = None
    #: Chapter 36 over whatever partition this run produced, whichever method produced it: the
    #: modularity with its resolution-limit check, §36.2's other topological measures with what
    #: each of them favours, and the partition scored as a link predictor. ``None`` when nothing
    #: was grouped, because every measure in it is a function of a partition.
    evaluation: PartitionScores | None = None
    #: The same ``--by`` section for a *quantitative* key (Atlas ch. 31): a number is near or
    #: far rather than same or different, so it gets a correlation over the edges instead of a
    #: matching rate. Exactly one of this and ``attribute`` is set, decided by
    #: :func:`graphrag.sna.assortativity.is_numeric`.
    numeric: NumericReport | None = None
    #: Chapter 31 over the degree, on every non-empty network whether or not ``--by`` was
    #: asked for: whether the hubs talk to each other (§31.1) and the friendship paradox
    #: (§31.2). It decides how every ranking above it reads, so it is not optional.
    degree_correlations: DegreeCorrelations | None = None
    #: Chapter 15 over the grouping this run found: every node's Guimera-Amaral role in the
    #: (P, z) plane of §15.1. ``None`` when nothing was grouped, because a role is defined
    #: against a partition and there is no partition-free version of it.
    roles: RoleTable | None = None
    #: The §15.2 pairs: same role, similar neighbourhoods, no edge between them. Empty when the
    #: network was too big for a dense similarity matrix, and ``notes`` says so when it was.
    role_pairs: list[RolePair] = field(default_factory=list)
    #: Which similarity ``role_pairs`` were ranked by, for the caption.
    role_similarity: str = ""
    #: Chapter 10 and §13.1-13.3 over the same graph: components, the path-length distribution,
    #: the cycle space and -- on a directed network -- the dyad census. Always computed, because
    #: how far apart a network is and how many pieces it comes in are properties of every
    #: network rather than an option; ``None`` only when the network has no nodes at all.
    paths: PathReport | None = None
    #: Chapter 33 over the same graph, on a directed network only: which of §33.1's shapes it
    #: is, how much of it lies on cycles, its global reach centrality, the tree of one boss per
    #: node and the agony of the arrows that point up, each against the configuration model.
    #: ``None`` on the four undirected networks, where every one of those numbers is degenerate
    #: rather than merely inapplicable (§33.1, p. 465).
    hierarchy: HierarchyReport | None = None
    #: What the edges of this network rest on (Atlas §28.1). Always computed, because an edge is
    #: a claim with evidence and the size of the thinnest evidence decides how the rest of the
    #: report reads. ``None`` only for an empty network.
    evidence: Evidence | None = None
    #: The same network read as probabilistic (§28.2-28.3), when ``--uncertain`` asked for it:
    #: every ranking above as a mean and an interval over sampled possible worlds.
    uncertainty: Uncertainty | None = None
    #: Chapter 12 over the same graph: density with the note that it is only readable against
    #: n, the average and global clustering coefficients, the cliques and an independent set.
    #: ``None`` only for an empty network, which has none of them.
    density: DensityReport | None = None
    #: Chapter 9's section: the degree distribution, the power-law fit and its test. Present
    #: on every non-empty network, because how the degrees distribute decides how every
    #: ranking above it may be read (§9.3: a mean degree with an undefined variance).
    degree: DegreeReport | None = None
    #: Chapter 14's section: how centralized the network is by each centrality (§14.8), and how
    #: much of each printed ranking survives resampling the corpus. ``None`` for an empty
    #: network, and budgeted on a large one -- see :func:`ranking_report`.
    ranking: RankingReport | None = None
    #: Chapter 30's edge-level section: neighbourhood overlap per edge against edge weight,
    #: which is §30.3's strength of weak ties. Present on every network with at least one edge;
    #: the section says so itself when the weights do not vary and there is no curve to draw.
    ties: TieReport | None = None
    #: Chapter 32's section: both core-periphery models fitted to the same adjacency, the rich
    #: club against a degree-preserving null, and -- on a Louvain run -- §32.2's comparison of
    #: the core-periphery ideal with the communities this run found, which decides whether those
    #: communities may be named at all. Plus nestedness (§32.4) when the network is two-mode.
    #: ``None`` only for an empty network.
    coreperiphery: CorePeripheryReport | None = None
    #: Chapter 39's own evaluation, present only when ``method`` is one of
    #: :data:`BIPARTITE_METHODS`: Barber modularity of the grouping above and its curveball null,
    #: never the unamended modularity ``## Community evaluation`` reports for the same partition.
    bipartite: BipartiteReport | None = None
    #: The layers this analysis's graph was flattened from, when it was built with ``--layers``.
    #: Everything above ran on the flattening, so the per-layer table is printed next to it
    #: (Atlas §7.2) rather than left for the reader to ask for.
    multilayer: Multilayer | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def group_plural(self) -> str:
        return "communities" if self.group_noun == "community" else f"{self.group_noun}s"

    def label(self, node: str) -> str:
        return _node_label(self.graph, node)


def _feature_rows(
    store: GraphStore,
    graph: nx.Graph,
    network: str,
    persona_id: str,
    features: Features,
    dims: int,
    seed: int | None,
    *,
    p: float = 1.0,
    q: float = 1.0,
) -> tuple[list[str], np.ndarray, list[str]]:
    """The nodes to cluster, their feature matrix, and any notes about what was dropped."""
    if features == "spectral":
        embedding = spectral_embedding_full(graph, dims, seed)
        return embedding.nodes, embedding.vectors, _eigengap_notes(embedding)
    if features == "node2vec":
        walked = node2vec(graph, dims, p=p, q=q, seed=seed)
        return walked.nodes, walked.vectors, _random_walk_notes(walked)
    if features == "metapath2vec":
        if two_mode_sides(graph) is None:
            msg = (
                "--features metapath2vec needs a two-mode network (§43.3): build --network "
                "speakers-entities without --project, so every node carries the network's own "
                "'speaker' or 'entity' mode."
            )
            raise ValueError(msg)
        walked = metapath2vec(graph, dims, seed=seed)
        return walked.nodes, walked.vectors, _random_walk_notes(walked)
    if network != "entities":
        msg = (
            "--features embedding needs text behind each node, which only the entities network "
            "has (the mean embedding of the passages mentioning each entity). Use "
            "--features spectral for the speakers and topics networks."
        )
        raise ValueError(msg)
    keys, matrix = store.mean_embeddings(persona_id, level="entity")
    index = {key: i for i, key in enumerate(keys)}
    nodes = [n for n in graph.nodes if n in index]
    dropped = graph.number_of_nodes() - len(nodes)
    notes = (
        [f"{dropped} node(s) had no stored embedding and were left out of the clustering."]
        if dropped
        else []
    )
    rows = matrix[[index[n] for n in nodes]] if nodes else np.zeros((0, 0), dtype=np.float32)
    return nodes, rows, notes


def _eigengap_notes(embedding: Embedding) -> list[str]:
    """A note when :func:`graphrag.sna.embed.spectral` truncated ``dims`` at an eigengap (ATL-42).

    A clique, or any other exactly regular block of a network, has no internal structure for the
    dropped coordinates to have carried: they would have been an arbitrary basis of a degenerate
    eigenspace, not a reading of anything in the corpus, so the report says why fewer columns
    came back than ``--dims`` asked for rather than silently handing K-means a narrower matrix.
    """
    requested = embedding.provenance.get("dims_requested")
    kept = embedding.provenance.get("dims_kept")
    eigenvalue = embedding.provenance.get("eigengap_eigenvalue")
    components = embedding.provenance.get("components")
    component_dims = embedding.provenance.get("component_dims")
    notes: list[str] = []
    if isinstance(kept, int) and isinstance(components, int) and isinstance(component_dims, int):
        if component_dims and component_dims >= kept:
            notes.append(
                f"All {kept} spectral dimension(s) only say which of the network's {components} "
                "connected components a node is in (§8.4: the zero eigenvalue repeats once per "
                "component, and its eigenvectors are component indicators). The groups in this "
                "report therefore separate components -- the smaller ones from the largest -- and "
                "carry nothing about structure inside any of them. Pass --dims above "
                f"{components - 1} to reach inside, or filter the network into one piece."
            )
        elif component_dims:
            notes.append(
                f"The first {component_dims} of {kept} spectral dimensions only say which of the "
                f"network's {components} connected components a node is in (§8.4); the other "
                f"{kept - component_dims} read structure inside components. Expect some groups "
                "in this report to be components rather than communities."
            )
    if eigenvalue is None or not isinstance(requested, int) or not isinstance(kept, int):
        return notes
    return [
        *notes,
        f"Spectral embedding asked for {requested} dimensions but kept {kept}: the rest would "
        f"have cut into a block of mutually tied eigenvalues near {eigenvalue:.4g} (§42.3), "
        "which has no canonical basis -- the dropped coordinates would have been an arbitrary "
        "rotation within it, not real structure.",
    ]


def _random_walk_notes(embedding: Embedding) -> list[str]:
    """What a ``--features node2vec``/``metapath2vec`` report needs beside the vectors (ch. 43):
    the bounded hyperparameters that produced them, and how many nodes no walk ever reached."""
    prov = embedding.provenance
    notes = [
        f"{embedding.method} ({prov.get('chapter', '')}): {prov.get('walks_per_node')} walk(s) "
        f"of length {prov.get('walk_length')} per node, window {prov.get('window')}, "
        f"{prov.get('negative')} negative sample(s), {prov.get('epochs')} epoch(s), seed "
        f"{prov.get('seed')}. These embeddings are transductive (§43.5): they are refit from "
        "scratch on this run's graph, never carried over from a previous run."
    ]
    unvisited = prov.get("unvisited_nodes")
    if unvisited:
        notes.append(
            f"{unvisited} node(s) were never reached by any walk and kept only their random "
            "initialisation; read their vectors as noise, not as a finding about the corpus."
        )
    return notes


def _grouped(nodes: Sequence[str], labels: Sequence[int]) -> list[list[str]]:
    buckets: dict[int, list[str]] = {}
    for node, label in zip(nodes, labels, strict=True):
        buckets.setdefault(int(label), []).append(node)
    return [sorted(buckets[key]) for key in sorted(buckets)]


def _shared_rewirings(
    graph: nx.Graph,
    *,
    samples: int,
    seed: int | None,
    resolution: float,
    scores: list[float],
) -> Iterator[nx.Graph]:
    """One family of degree-preserving rewirings, scored for the modularity null on the way past.

    Two sections of this report test against the same null (§19.1): Louvain's modularity, and
    chapter 17's clustering and path length. Both want ``samples`` rewirings of the same graph
    from the same seed, and drawing them is the expensive half -- one attempted edge swap per
    edge per sample. This draws the family once: each rewiring is scored for the modularity null
    into ``scores`` and then handed to ``against_random``, which measures it and drops it.

    Nothing is accumulated but the scores, deliberately. Keeping the rewirings in a list to pass
    around twice would cost about a gigabyte for fifty samples of a 74,000-edge network, which is
    the size of network this report is meant to run on.

    The seeding order is :func:`graphrag.sna.cluster.null_model_modularity`'s own -- draw a
    sample, then take a Louvain seed from the same stream -- so the modularity null comes out
    with the numbers it would have had if it had drawn the family itself.
    """
    flat, _ = undirected_view(graph)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    for rewired in configuration(flat, samples, seed=rng):
        scores.append(rewired_modularity(rewired, resolution, rng.randrange(1_000_000)))
        yield rewired


def run_analysis(
    store: GraphStore,
    graph: nx.Graph,
    *,
    persona_id: str,
    network: str,
    method: Method,
    k: int | None = None,
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    resolution: float = 1.0,
    runs: int = 10,
    features: Features = "spectral",
    dims: int = 8,
    p: float = 1.0,
    q: float = 1.0,
    covariance: str = "full",
    samples: int = 50,
    by: str | None = None,
    permutations: int = PERMUTATIONS,
    null: str = "permutation",
    seed: int | None = None,
    multilayer: Multilayer | None = None,
    vocabulary: AttributeTable | None = None,
    uncertain: bool = False,
    uncertain_samples: int = SAMPLES,
    uncertain_expensive: bool = False,
    uncertain_max_seconds: float | None = None,
    sbm_degree_corrected: bool = True,
) -> Analysis:
    """Measure the network, group it with the chosen method, and check the grouping.

    ``method`` is one of :data:`METHODS`. ``louvain``, ``kmeans`` and ``gmm`` are chapter 36's
    trio; ``directed`` is Louvain's own directed modularity read on the arrows instead of a
    flattening (§37.3); ``louvain-levels``, ``girvan-newman`` and ``hrg`` are §37.1-§37.2's
    dendrogram methods, each cut to a single partition (the peak-modularity level, or -- for
    ``hrg`` -- the density-beats-the-network-as-a-whole cut) so the rest of this report can read
    it the way it reads any other partition; ``sbm``, ``infomap``, ``walktrap`` and
    ``label-propagation`` are chapter 35's own routes to a community (§35.1-§35.3). Every one of
    them lands in :attr:`Analysis.groups` and goes through the same ch. 36 evaluation
    (:attr:`Analysis.evaluation`) and role reading (:attr:`Analysis.roles`) below; the method's
    own diagnostics -- a dendrogram's levels and peaks, a blockmodel's log-likelihood and which
    implementation fit it, a map equation's code length -- are kept on the field named for that
    method (:attr:`Analysis.dendrogram`, :attr:`Analysis.sbm`, :attr:`Analysis.infomap`, ...) and
    printed above the shared sections. ``sbm_degree_corrected`` switches ``--method sbm``'s
    likelihood between the plain and the degree-corrected model (§35.1); it has no effect on any
    other method.

    ``kmeans`` and ``gmm`` cluster ``features`` rather than the raw edges: ``spectral`` (ch. 42,
    the default), ``embedding`` (the entities network's own stored text embedding), or ch. 43's
    two random-walk families, ``node2vec`` (``p``/``q`` bias the walk; ``p = q = 1`` is DeepWalk)
    and ``metapath2vec`` (a speaker-entity-speaker walk, only on the two-mode network). ``p`` and
    ``q`` are read only by ``--features node2vec`` and ignored by everything else.

    ``by`` adds the attribute section: whether the network divides along a property its nodes
    already carry, measured against a permutation null and a degree-preserving one, and compared
    with the grouping this run found. It is asked of the network after it is grouped, never
    instead of grouping it -- the point of the section is the comparison.

    That section comes in two kinds, because the book asks two different questions of the two
    kinds of value. A **label** gets chapter 30's homophily: how often an edge joins two nodes
    with the same value, against a shuffle of the labels
    (:func:`graphrag.sna.attributes.analyse_attribute`). A **number** gets chapter 31's
    assortativity: the correlation between the numbers at the two ends of an edge, the curve of a
    node's value against its neighbours', and the paradox that follows
    (:func:`graphrag.sna.assortativity.numeric_report`). ``vocabulary`` is the persona's
    ``facets.yaml`` table and decides which one a key gets -- a key declared ``type: number`` is
    a number and a key declared with ``values:`` is a label; without it the values decide. The
    built-in counts (``mentions``, ``documents``, ``chunks``) and ``degree`` are always numbers.
    Chapter 31 over the degree is printed either way, on every non-empty network.

    ``multilayer`` is passed when ``graph`` is a flattening of it (Atlas §7.2). Nothing here
    reads a layer: the analysis runs on the flattened graph exactly as it would on any other
    weighted graph, and the layers are carried through only so the report can print the
    per-layer table above the numbers and say that the numbers cannot see it.
    ``null`` picks the second null for that section's assortativity, one of
    :data:`graphrag.sna.attributes.ATTRIBUTE_NULLS`. ``bipartite`` rewires the two-mode
    memberships the network was projected from instead of shuffling labels, and raises
    ``ValueError`` on a network that was never projected. It has no effect without ``by``.

    The §28.1 evidence section is computed always and is not a flag: what the edges rest on is a
    property of the network, and a ranking over edges that each rest on one loose match is a
    different object from one over edges several passages support. ``uncertain`` adds the rest
    of chapter 28 -- every centrality and the partition's modularity as a mean and an interval
    over ``uncertain_samples`` sampled possible worlds (§28.2) -- which costs that many
    evaluations of every measure above and is therefore asked for rather than assumed.

    Two things keep that cost bounded on a large network (ATL-F2, not the book's own remedy,
    which is to sparsify the network first, p. 402). ``uncertain_expensive`` is off by default,
    which means the four shortest-path-search centralities -- betweenness, closeness, harmonic,
    reach -- are left out of the sampling (:data:`graphrag.sna.measures.EXPENSIVE_CENTRALITIES`);
    the rest, including degree, eigenvector, PageRank and coreness, always run. And
    ``uncertain_max_seconds`` is a wall-clock budget shared across everything ``uncertain``
    computes: once it is spent, the realisations in progress finish and no more start, and the
    report says, per measure, how many of ``uncertain_samples`` it actually got.
    """
    stats = summary(graph)
    # The whole ranking per centrality, not only its head: the head is what the report prints,
    # and the tail is what §14.8's centralization and the stability check are computed from.
    scores = {kind: centrality(graph, kind) for kind in centralities_for(graph)}
    centralities = {kind: top_n(values, TOP_N) for kind, values in scores.items()}
    notes: list[str] = directed_notes(graph)
    analysis = Analysis(
        persona_id=persona_id,
        network=network,
        method=method,
        graph=graph,
        summary=stats,
        centralities=centralities,
        groups=[],
        group_noun="community" if method in COMMUNITY_METHODS else "cluster",
        rationale=rationale(method),
        seed=seed,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        multilayer=multilayer,
        density=density_report(graph) if graph.number_of_nodes() else None,
        notes=notes,
    )
    if graph.number_of_nodes() == 0:
        notes.append("The network is empty: no nodes matched this persona and these filters.")
        return analysis
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        notes.append(
            f"The network has {graph.number_of_nodes()} node(s) and no edges, so there is "
            f"nothing to group into {analysis.group_plural}: community detection, its "
            "stability check across seeds, and the null model all need at least one edge, and "
            "none of them ran. The centrality table above is what a network this size can "
            "support."
        )
        return analysis

    # One null family for both sections that want one. Louvain's modularity null and chapter
    # 17's comparison test against the same rewirings, from the same seed, at the same count, so
    # the family is drawn once here and scored for the modularity null on its way into
    # ``against_random``. Everything else in the report runs afterwards; the shared draw has to
    # be exhausted before ``null_model_modularity`` can be handed its scores, and it is, because
    # ``against_random`` consumes the whole iterable before it returns.
    null_scores: list[float] = []
    analysis.against_random = against_random(
        graph,
        samples=samples,
        seed=seed,
        filters=_filter_line(graph.graph),
        rewirings=(
            _shared_rewirings(
                graph, samples=samples, seed=seed, resolution=resolution, scores=null_scores
            )
            if method == "louvain"
            else None
        ),
    )

    analysis.paths = analyse_paths(graph, seed=seed)
    if graph.is_directed():
        analysis.hierarchy = analyse_hierarchy(graph, samples=samples, seed=seed)
    analysis.evidence = evidence_report(graph)
    analysis.degree = degree_report(graph, seed=seed)
    analysis.degree_correlations = degree_correlations(graph, samples=samples, seed=seed)
    analysis.ranking = ranking_report(graph, scores=scores, samples=samples, seed=seed)
    # §30.3 over the edges, not the nodes: which of them bridge, and whether the heavy ones are
    # the embedded ones. It needs no seed and no null -- an overlap is exact -- so it runs on
    # every network that has an edge at all.
    analysis.ties = tie_strength_curve(graph) if graph.number_of_edges() else None

    if method == "louvain":
        result = louvain(graph, resolution=resolution, seed=seed, runs=runs)
        analysis.louvain_result = result
        analysis.groups = result.communities
        analysis.null_model = null_model_modularity(
            graph,
            result.communities,
            samples=samples,
            seed=seed,
            resolution=resolution,
            scores=null_scores,
        )
        if result.note:
            notes.append(result.note)
    elif method in BIPARTITE_METHODS:
        # §39.3-39.4: grouped directly on the two-mode network, both modes in one community, so
        # none of the feature-matrix or standard-modularity-null machinery below applies. A
        # network that never went two-mode raises here, inside ``bipartite_report`` itself
        # (``two_mode_sides`` returning ``None``), with the message that says so.
        report = bipartite_report(
            graph,
            method,
            k=k,
            k_range=k_range,
            resolution=resolution,
            runs=runs,
            samples=samples,
            seed=seed,
        )
        analysis.bipartite = report
        analysis.groups = report.result.communities
    elif method == "directed":
        directed_result = directed_communities(graph, resolution=resolution, seed=seed, runs=runs)
        analysis.louvain_result = directed_result
        analysis.groups = directed_result.communities
        if directed_result.note:
            notes.append(directed_result.note)
    elif method == "louvain-levels":
        dendrogram = louvain_levels(graph, resolution=resolution, seed=seed)
        analysis.dendrogram = dendrogram
        analysis.groups = dendrogram.communities
        if dendrogram.note:
            notes.append(dendrogram.note)
    elif method == "girvan-newman":
        dendrogram = girvan_newman_dendrogram(graph, resolution=resolution)
        analysis.dendrogram = dendrogram
        analysis.groups = dendrogram.communities
        if dendrogram.note:
            notes.append(dendrogram.note)
    elif method == "walktrap":
        dendrogram = walktrap(graph, resolution=resolution)
        analysis.dendrogram = dendrogram
        analysis.groups = dendrogram.communities
        if dendrogram.note:
            notes.append(dendrogram.note)
    elif method == "hrg":
        fit = hrg_fit(graph, seed=seed)
        analysis.hrg = fit
        analysis.groups = hrg_communities(fit)
    elif method == "sbm":
        sbm_fit = sbm_communities(graph, k=k, degree_corrected=sbm_degree_corrected, seed=seed)
        analysis.sbm = sbm_fit
        analysis.groups = sbm_fit.communities
        if sbm_fit.note:
            notes.append(sbm_fit.note)
    elif method == "infomap":
        infomap_result = infomap_communities(graph, seed=seed, runs=runs)
        analysis.infomap = infomap_result
        analysis.groups = infomap_result.communities
        if infomap_result.note:
            notes.append(infomap_result.note)
    elif method == "label-propagation":
        lp_result = label_propagation(graph, seed=seed, runs=runs)
        analysis.label_propagation_result = lp_result
        analysis.groups = lp_result.communities
        if lp_result.note:
            notes.append(lp_result.note)
    else:
        nodes, matrix, feature_notes = _feature_rows(
            store, graph, network, persona_id, features, dims, seed, p=p, q=q
        )
        notes.extend(feature_notes)
        analysis.features = features
        analysis.feature_dims = int(matrix.shape[1]) if matrix.size else 0
        if len(nodes) < 3:
            notes.append(
                f"Only {len(nodes)} node(s) carried features, which is too few to cluster."
            )
            return analysis
        candidates = [c for c in k_range if 2 <= c < len(nodes)]
        if method == "kmeans":
            choice = choose_k_kmeans(matrix, candidates, seed)
            analysis.choice = choice
            chosen = k or choice.best_silhouette_k or 2
            km = kmeans(matrix, chosen, seed)
            analysis.kmeans_result = km
            analysis.groups = _grouped(nodes, km.labels)
            if k is None and choice.elbow_k and choice.elbow_k != choice.best_silhouette_k:
                notes.append(
                    f"Silhouette picked k={choice.best_silhouette_k} and the inertia elbow "
                    f"pointed at k={choice.elbow_k}; the silhouette pick was used."
                )
        else:
            choice = choose_k_gmm(matrix, candidates, covariance, seed)
            analysis.choice = choice
            chosen = k or choice.best_bic_k or 2
            mixture = gmm(matrix, chosen, covariance, seed)
            analysis.gmm_result = mixture
            analysis.groups = _grouped(nodes, mixture.labels)
            if not mixture.converged:
                notes.append("The mixture did not converge; treat the memberships as indicative.")
    _warn_if_it_only_found_the_components(analysis)
    analysis.brokers = brokers(graph, analysis.groups, TOP_N)
    _add_roles(analysis)
    if analysis.groups:
        # Chapter 36 runs for every method, not only for Louvain: a K-means clustering of a
        # graph's nodes is a partition of that graph and has a modularity, a conductance and a
        # resolution limit exactly as a Louvain partition does -- and is the case where reading
        # them matters most, because nothing in the clustering itself looked at the edges.
        analysis.evaluation = evaluate_partition(
            graph, analysis.groups, resolution=resolution, seed=seed
        )
        if two_mode_sides(graph) is not None:
            # §39.1: this battery's modularity, internal_density, cut_ratio, performance and
            # triangle_participation assume a one-mode graph, and read wrong on a partition that
            # mixes both modes -- true of every method's grouping here, not only the bipartite
            # ones, whenever the network itself was never projected. The note says which numbers
            # above still mean what they say.
            analysis.evaluation.notes.append(BIPARTITE_EVALUATION_NOTE)
    _add_core_periphery(analysis, samples=samples, seed=seed, resolution=resolution)
    if uncertain:
        analysis.uncertainty = _uncertainty(
            analysis,
            samples=uncertain_samples,
            seed=seed,
            expensive=uncertain_expensive,
            max_seconds=uncertain_max_seconds,
        )
    if by:
        flat = undirected_view(graph)[0]  # its nulls are undirected; directed_notes says so
        if is_numeric(flat, by, vocabulary):
            _check_numeric_null(null, by)
            if by == "degree":
                # `## Degree correlations` above is this section, for this key, and it is printed
                # on every report. Printing it twice under two headings would let a reader quote
                # one of them as if it were a second, agreeing measurement.
                notes.append(
                    "--by degree asks for the section every report already prints: chapter 31 "
                    "over the degree is `## Degree correlations` above, with the same "
                    "coefficient, the same null, the same curve and the same paradox. Nothing "
                    "was added below it. Use --by on an attribute to compare one against the "
                    "degree."
                )
            else:
                analysis.numeric = numeric_report(
                    flat,
                    by,
                    permutations=permutations,
                    samples=samples,
                    seed=seed,
                    table=vocabulary,
                )
        else:
            analysis.attribute = analyse_attribute(
                flat,
                by,
                groups=analysis.groups,
                group_noun=analysis.group_noun,
                resolution=resolution,
                permutations=permutations,
                samples=samples,
                null=null,
                seed=seed,
            )
    return analysis


def _add_core_periphery(
    analysis: Analysis, *, samples: int, seed: int | None, resolution: float
) -> None:
    """Chapter 32 over the same network, and -- on a Louvain run -- over this run's partition.

    The two models and the rich club are properties of the network rather than of any method, so
    they are fitted whatever ``--method`` asked for. §32.2's comparison needs a partition that
    Louvain found, and is run only then: the community ideal it scores is the *best* community
    structure of each rewired network, found by re-running Louvain on it, and comparing that
    with a K-means partition of a spectral embedding would be comparing two different methods
    rather than two models. The kmeans and gmm reports therefore print the models and the club
    and no verdict, which is the honest amount.

    The cost is one more set of ``samples`` degree-preserving rewirings for the rich club, and
    for a Louvain run a second set with a core fit and a Louvain run on each -- the same order
    as the null model already above it.
    """
    if analysis.graph.number_of_nodes() == 0:
        return
    analysis.coreperiphery = core_periphery_report(
        analysis.graph,
        analysis.louvain_result.communities if analysis.louvain_result is not None else None,
        samples=samples,
        seed=seed,
        resolution=resolution,
    )


def _add_roles(analysis: Analysis) -> None:
    """Chapter 15 over the grouping this run found: the role table and the §15.2 pairs.

    Only when there is a grouping, because both of §15.1's coordinates are defined against the
    modules and there is no partition-free role to report. The role table is O(m) and always
    runs; the similarity matrix behind the pairs is dense and n x n, so above
    :data:`graphrag.sna.roles.MAX_SIMILARITY_NODES` it is skipped and the report says which
    section is missing rather than quietly leaving it out. Jaccard is the similarity used, being
    the one §15.2 works through by hand and the only one of the three with no parameter to state.
    """
    if not analysis.groups:
        return
    analysis.roles = guimera_amaral(analysis.graph, analysis.groups)
    if analysis.graph.number_of_nodes() > MAX_SIMILARITY_NODES:
        analysis.notes.append(
            f"The roles section lists no same-role pairs: §15.2's similarity matrix is dense and "
            f"n x n, and this network has {analysis.graph.number_of_nodes():,} nodes against a "
            f"guard of {MAX_SIMILARITY_NODES:,}. `sna roles` on a narrowed network is where to "
            "ask for them."
        )
        return
    scores = structural_similarity(analysis.graph, "jaccard")
    analysis.role_similarity = scores.method
    analysis.role_pairs = same_role_never_cooccur(analysis.graph, analysis.roles, scores)


def _uncertainty(
    analysis: Analysis,
    *,
    samples: int,
    seed: int | None,
    expensive: bool,
    max_seconds: float | None,
) -> Uncertainty:
    """Chapter 28 over this run: the same rankings, as intervals over the possible worlds.

    The partition is carried into the sampling and held fixed: the question ``--uncertain`` asks
    of a grouping is what the uncertainty in the edges does to *this* grouping's modularity, not
    what a fresh Louvain run would find in each world -- that is a different question, and
    ``null_model_modularity`` is where it is asked. The partition is only scored when it covers
    every node, which a clustering that dropped nodes for want of features does not.

    ``expensive`` decides whether the shortest-path-search centralities
    (:data:`graphrag.sna.measures.EXPENSIVE_CENTRALITIES`) are sampled at all (ATL-F2): off by
    default, since re-running one of them on every realisation is exactly what made this section
    slow on a large network, and a note says which ones were left out rather than silently
    shrinking the table. ``max_seconds`` is the wall-clock budget the whole call shares, passed
    straight through to :func:`graphrag.sna.uncertain.uncertainty_report`.
    """
    graph = analysis.graph
    covered = {node for group in analysis.groups for node in group}
    if analysis.louvain_result is not None:
        resolution = analysis.louvain_result.resolution
    elif analysis.dendrogram is not None:
        resolution = analysis.dendrogram.resolution
    else:
        resolution = 1.0
    partition = [sorted(group) for group in analysis.groups]

    def modularity(world: nx.Graph) -> float:
        view, _ = undirected_view(world)
        return float(
            nx.community.modularity(view, partition, weight="weight", resolution=resolution)
        )

    skipped = [kind for kind in analysis.centralities if kind not in CHEAP_CENTRALITIES]
    centralities = (
        analysis.centralities
        if expensive
        else {
            kind: rows for kind, rows in analysis.centralities.items() if kind in CHEAP_CENTRALITIES
        }
    )
    report = uncertainty_report(
        graph,
        centralities,
        measure_for=lambda kind: lambda world: centrality(world, kind),
        modularity=(modularity if partition and covered >= set(graph.nodes) else None),
        samples=samples,
        seed=seed,
        max_seconds=max_seconds,
    )
    if not expensive and skipped:
        report.notes.append(
            f"{', '.join(sorted(skipped))} were not sampled under uncertainty: each needs a "
            "shortest-path search per node, so re-running it on every realisation is the cost "
            "that made this section slow on a large network (ATL-F2). Pass --uncertain-expensive "
            "to include them."
        )
    if partition and not covered >= set(graph.nodes):
        report.notes.append(
            f"The {analysis.group_noun} partition covers {len(covered)} of "
            f"{graph.number_of_nodes()} nodes, so its modularity is not scored here: a partition "
            "that does not cover the network has no modularity on a realisation of it."
        )
    return report


def _check_numeric_null(null: str, by: str) -> None:
    """Refuse a categorical null on a quantitative ``--by``, naming the one that was asked for.

    Refused rather than ignored, because the two nulls answer different questions: the
    bipartite-preserving one re-derives *borrowed labels* from a rewired corpus, which is a
    majority vote over values and has no numeric version. A report that printed a shuffle under
    the name the caller asked for would be worse than no report. An unknown null is refused the
    same way :func:`graphrag.sna.attributes.analyse_attribute` refuses it, so ``--by`` behaves
    the same whichever section it routes to.
    """
    if null not in ATTRIBUTE_NULLS:
        msg = f"null must be one of {', '.join(ATTRIBUTE_NULLS)}, got {null!r}"
        raise ValueError(msg)
    if null == "permutation":
        return
    msg = (
        f"--null {null} is a null for a categorical --by: it re-derives borrowed labels by "
        f"majority vote from the rewired corpus, and '{by}' is a number. Chapter 31's section "
        "picks its own null -- the configuration model for the degree, a shuffle of the values "
        "for an attribute -- so run it with --null permutation."
    )
    raise ValueError(msg)


def _warn_if_it_only_found_the_components(analysis: Analysis) -> None:
    """A fragmented network embeds into clumps that any clusterer separates happily.

    When one group swallows nearly everything and the rest are the small disconnected pieces,
    the grouping has rediscovered connectivity and said nothing about structure inside the main
    component. That looks like a great result: the silhouette is near 1, because the pieces
    really are far apart. Say so in the report rather than letting the number speak.
    """
    nodes = int(analysis.summary["nodes"])
    components = int(analysis.summary["components"])
    if len(analysis.groups) < 2 or components < 2 or nodes == 0:
        return
    biggest = max(len(group) for group in analysis.groups)
    if biggest / nodes < 0.8:
        return
    remedy = (
        "Reading only the largest component, or raising --min-weight"
        if analysis.method == "louvain"
        else "Louvain on the same network, or raising --min-weight"
    )
    analysis.notes.append(
        f"One {analysis.group_noun} holds {biggest} of {nodes} nodes and the network has "
        f"{components} components, so this grouping mostly separates disconnected pieces "
        f"rather than structure inside the main one. {remedy}, will say more about the part "
        "that matters."
    )


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _member_order(analysis: Analysis) -> str:
    """What the member lists are sorted by, named so the caption cannot mislead.

    Weighted degree on a directed network is the total, in plus out (§6.2), so the column header
    says so rather than letting a reader of a `relations` report take it for out-degree.
    """
    return "weighted degree (in + out)" if analysis.graph.is_directed() else "weighted degree"


def _members(analysis: Analysis, group: Sequence[str]) -> str:
    ranked = sorted(
        group,
        key=lambda n: (-float(analysis.graph.degree(n, weight="weight")), n),
    )
    names = [analysis.label(n) for n in ranked[:TOP_MEMBERS]]
    more = len(group) - len(names)
    return ", ".join(names) + (f", +{more} more" if more > 0 else "")


#: The graph metadata keys that narrowed the network, in the order the report lists them.
FILTER_KEYS: tuple[tuple[str, str], ...] = (
    ("source_id", "source"),
    ("types", "types"),
    ("stances", "stances"),
    ("facets", "facets"),
    ("relation_types", "relation types"),
    ("since", "since"),
    ("until", "until"),
    ("where", "where"),
    ("project", "projected onto"),
    # Not a filter but a definition, and printed here for the same reason the filters are: the
    # command that rebuilds this network has to name the scheme, because the weights differ
    # under every one of them (ch. 26). Empty, and so absent, on a network that is not a
    # projection at all.
    ("projection", "projection"),
    ("backbone_method", "backbone"),
)


def _filter_line(meta: dict[str, Any]) -> str:
    """Every filter that shaped this network, so a reader can rebuild the exact command."""
    parts = [f"min_weight={meta.get('min_weight', 1)}"]
    parts += [f"{label}={meta[key]}" for key, label in FILTER_KEYS if meta.get(key)]
    return ", ".join(parts)


def _cross_mode(analysis: Analysis) -> list[str]:
    """For the two-mode network: what each group on one side wrote about on the other.

    A projection throws the other mode away, so this does not read it off the graph. Every node
    of a two-mode network keeps its heaviest opposite-mode partners as ``partners`` and
    ``partner_weights``, written when the network was built; a group's entities are its own
    entity members plus the partners of its speakers, and the reverse for its speakers. The
    counts are documents, the same unit on both sides.
    """
    if analysis.graph.graph.get("network") != "speakers-entities" or not analysis.groups:
        return []
    lines = [
        "## Across the two modes",
        "",
        f"Each {analysis.group_noun} with the speakers in it and the entities their passages "
        "named, counted in documents. Read it as who wrote about what, not as who holds an "
        "opinion: an entity is here because a passage named it.",
        "",
    ]
    rows: list[list[str]] = []
    for index, group in enumerate(analysis.groups, start=1):
        speakers: Counter[str] = Counter()
        entities: Counter[str] = Counter()
        for node in group:
            data = analysis.graph.nodes[node]
            mine, theirs = (
                (entities, speakers) if data.get("mode") == "entity" else (speakers, entities)
            )
            mine[analysis.label(node)] += int(data.get("documents", 0))
            for name, weight in _partners(data):
                theirs[name] += weight
        rows.append([str(index), str(len(group)), _top(speakers), _top(entities)])
    lines += _table(["#", "size", "speakers", "entities"], rows)
    return lines


def _partners(data: dict[str, Any]) -> list[tuple[str, int]]:
    """The opposite-mode partners a two-mode node recorded, as (name, documents)."""
    names = [n for n in str(data.get("partners", "")).split("; ") if n]
    weights = [w for w in str(data.get("partner_weights", "")).split("; ") if w]
    return [(n, int(w)) for n, w in zip(names, weights, strict=False)]


def _top(counts: Counter[str]) -> str:
    """The heaviest few names with their counts, or a dash when there are none."""
    top = counts.most_common(TOP_MEMBERS)
    return ", ".join(f"{name} ({count})" for name, count in top) if top else "-"


def render_markdown(analysis: Analysis) -> str:
    """The analysis as a markdown note, ready to drop into a corpus as a research note."""
    meta = analysis.graph.graph
    lines: list[str] = [
        f"# Network analysis: {analysis.persona_id} / {analysis.network}",
        "",
        f"Generated {analysis.generated_at} by graphrag {__version__}"
        + (f", seed {analysis.seed}." if analysis.seed is not None else ", no seed fixed."),
        "",
        f"**Method.** {analysis.rationale}",
        "",
    ]
    if analysis.features:
        lines += [
            f"**Features.** `{analysis.features}` in {analysis.feature_dims} dimensions: "
            f"{FEATURE_MEANING[analysis.features]}",
            "",
        ]
    lines += [f"**Network type.** {describe(analysis.graph).sentence}", ""]
    frame = str(meta.get("frame", ""))
    lines += [f"**Sampling frame.** {frame} Filters: {_filter_line(meta)}.", ""]

    if analysis.multilayer is not None:
        lines += render_layers(analysis.multilayer)

    lines += ["## Network summary", ""]
    lines += _table(
        ["measure", "value"],
        [[_summary_label(key), _num(value)] for key, value in analysis.summary.items()],
    )
    if analysis.summary["nodes"] < 30:
        lines += [
            f"> Only {int(analysis.summary['nodes'])} nodes. Read the rankings below as a "
            "description of this small network, not as an estimate of anything wider.",
            "",
        ]
    if analysis.density is not None:
        lines += render_density(analysis.density)

    if analysis.against_random is not None:
        lines += render_against_random(analysis.against_random)

    if analysis.degree is not None:
        lines += render_degree(analysis.degree)

    if analysis.degree_correlations is not None:
        lines += render_degree_correlations(analysis.degree_correlations)
    if analysis.paths is not None:
        lines += render_paths(analysis.paths)
        lines += [analysis.paths.small_world.sentence, ""]

    if analysis.ties is not None:
        lines += render_ties(analysis.ties)

    if analysis.hierarchy is not None:
        lines += render_hierarchy(analysis.hierarchy)

    checks = _render_checks(analysis)
    if analysis.groups:
        lines += [f"## {analysis.group_plural.capitalize()} ({len(analysis.groups)})", ""]
        lines += _table(
            ["#", "size", f"top members by {_member_order(analysis)} (up to {TOP_MEMBERS})"],
            [
                [str(i), str(len(group)), _members(analysis, group)]
                for i, group in enumerate(analysis.groups, start=1)
            ],
        )
        # The partition's own checks (stability, null model, choice of k) are subsections of
        # the table they judge, so they come before the next top-level section rather than after
        # it, where they would read as part of whatever section happened to precede them.
        lines += checks
        checks = []
        if analysis.evaluation is not None:
            lines += render_evaluation(analysis.evaluation)
        if analysis.bipartite is not None:
            lines += render_bipartite(analysis.bipartite)

    lines += _cross_mode(analysis)
    # Chapter 28 before the rankings: what the edges rest on decides how everything below it
    # reads, so it is not an appendix.
    if analysis.uncertainty is not None:
        lines += render_uncertain(analysis.uncertainty)
    elif analysis.evidence is not None:
        lines += render_evidence(analysis.evidence)
    lines += checks
    # Chapter 32 straight after the partition and its checks, because it is one more check on
    # the partition: §32.2 says a core-periphery network has no communities to find, so whether
    # the groups above may be named at all is settled here rather than in an appendix.
    if analysis.coreperiphery is not None:
        lines += render_core_periphery(analysis.coreperiphery, analysis.graph)
    if analysis.attribute is not None:
        lines += render_attribute(analysis.attribute)
    if analysis.numeric is not None:
        lines += render_numeric(analysis.numeric)

    lines += ["## Centrality", ""]
    for kind, scores in analysis.centralities.items():
        if not scores:
            continue
        lines += [f"### {kind}", "", centrality_meaning(kind, analysis.graph), ""]
        lines += _table(
            ["rank", "node", "score"],
            [
                [str(i), analysis.label(node), _num(score)]
                for i, (node, score) in enumerate(scores, start=1)
            ],
        )

    if analysis.ranking is not None:
        lines += render_ranking(analysis.ranking)

    if analysis.brokers:
        lines += [
            "## Brokers",
            "",
            "Participation coefficient: how evenly a node's ties are spread across the "
            f"{analysis.group_plural} above. High means its neighbours come from many groups.",
            "",
        ]
        lines += _table(
            ["rank", "node", "participation"],
            [
                [str(i), analysis.label(node), _num(score)]
                for i, (node, score) in enumerate(analysis.brokers, start=1)
            ],
        )

    lines += _render_roles(analysis)

    lines += ["## Caveats", ""]
    lines += [f"- {item}" for item in ALWAYS]
    lines += [f"- {note}" for note in analysis.notes]
    return "\n".join(lines).rstrip() + "\n"


SUMMARY_LABELS = {"average_clustering": "weighted average clustering"}
"""Summary rows whose payload key alone would mislead: ``average_clustering`` is a *weighted*
coefficient (Onnela's, or Fagiolo's on a directed network), which sits far below the unweighted
one when most ties were seen once, so printing it as plain "average clustering" beside
transitivity invites comparing two different quantities. The JSON keeps the key."""


def _summary_label(key: str) -> str:
    return SUMMARY_LABELS.get(key, key.replace("_", " "))


def _range_edge_warning(analysis: Analysis) -> list[str]:
    """A warning when the k a search chose sits on the edge of the range it searched.

    A silhouette that is still rising, or a BIC still falling, at the last k tried has not found
    an optimum: it has found the end of ``--k-range``. The bottom edge only counts above k = 2,
    since no search can go below that.
    """
    choice = analysis.choice
    if choice is None or len(choice.rows) < 2:
        return []
    if analysis.kmeans_result is not None:
        chosen, picked = analysis.kmeans_result.k, choice.best_silhouette_k
    elif analysis.gmm_result is not None:
        chosen, picked = analysis.gmm_result.k, choice.best_bic_k
    else:
        return []
    if chosen != picked:
        # --k pinned it: nobody searched, so there is no edge to have stopped at.
        return []
    searched = [int(row["k"]) for row in choice.rows]
    if chosen == max(searched):
        edge, direction = "largest", "upward"
    elif chosen == min(searched) and chosen > 2:
        edge, direction = "smallest", "downward"
    else:
        return []
    return [
        f"> k={chosen} is the {edge} k searched ({min(searched)}-{max(searched)}), so the "
        "criterion had not turned when the search stopped: widen --k-range "
        f"{direction} before reading k={chosen} as the number of groups.",
        "",
    ]


def _render_checks(analysis: Analysis) -> list[str]:
    lines: list[str] = []
    louvain_result = analysis.louvain_result
    if louvain_result is not None:
        lines += [
            "### Stability across seeds",
            "",
            f"- modularity {_num(louvain_result.modularity)} at resolution "
            f"{_num(louvain_result.resolution, 2)}, best of {len(louvain_result.modularities)} "
            f"runs (range {_num(min(louvain_result.modularities))} to "
            f"{_num(max(louvain_result.modularities))})",
            f"- mean pairwise adjusted Rand index between runs: {_num(louvain_result.stability)}",
            f"- seeds: {', '.join(str(s) for s in louvain_result.seeds)}",
            "",
            _stability_reading(louvain_result.stability),
            "",
        ]
    null_model = analysis.null_model
    if null_model is not None:
        lines += [
            "### Null model",
            "",
            f"- observed modularity: {_num(null_model.observed)}",
            f"- degree-preserving rewirings: {null_model.samples}, "
            f"mean {_num(null_model.null_mean)}, sd {_num(null_model.null_std)}",
            f"- z-score: {_num(null_model.z_score, 2)} — {null_model.verdict}",
            "",
        ]
    choice = analysis.choice
    if choice is not None and choice.rows:
        header = (
            ["k", "silhouette", "inertia"] if analysis.method == "kmeans" else ["k", "BIC", "AIC"]
        )
        keys = ["silhouette", "inertia"] if analysis.method == "kmeans" else ["bic", "aic"]
        lines += ["### Choosing k", ""]
        lines += _table(
            header,
            [[_num(row["k"]), _num(row[keys[0]]), _num(row[keys[1]])] for row in choice.rows],
        )
        picks = []
        if choice.best_silhouette_k:
            picks.append(f"best silhouette at k={choice.best_silhouette_k}")
        if choice.elbow_k:
            picks.append(f"inertia elbow at k={choice.elbow_k}")
        if choice.best_bic_k:
            picks.append(f"lowest BIC at k={choice.best_bic_k}")
        if picks:
            lines += ["- " + "; ".join(picks), ""]
    if analysis.kmeans_result is not None:
        km = analysis.kmeans_result
        lines += [
            f"- k={km.k} used; silhouette {_num(km.silhouette)}, inertia {_num(km.inertia)}.",
            "",
        ]
    if analysis.gmm_result is not None:
        mixture = analysis.gmm_result
        lines += [
            f"- k={mixture.k} used with a {mixture.covariance} covariance; "
            f"BIC {_num(mixture.bic)}, AIC {_num(mixture.aic)}, "
            f"converged={str(mixture.converged).lower()}.",
            "",
        ]
    lines += _range_edge_warning(analysis)
    dendrogram = analysis.dendrogram
    if dendrogram is not None:
        lines += [
            f"### {dendrogram.method.replace('-', ' ').title()} dendrogram (§37.1)",
            "",
            f"- {len(dendrogram.levels)} level(s); cut at level {dendrogram.cut} "
            f"({len(dendrogram.communities)} {analysis.group_plural}), "
            f"modularity {_num(dendrogram.modularity)}, density {_num(dendrogram.density)}",
            f"- peaks in the modularity profile: {', '.join(str(p) for p in dendrogram.peaks)}",
        ]
        if len(dendrogram.peaks) > 1:
            lines.append(
                "- more than one peak: several different numbers of communities are all locally "
                "good partitions (§37.4), not only the one cut here."
            )
        lines.append("")
    hrg = analysis.hrg
    if hrg is not None:
        lines += [
            "### Hierarchical Random Graph (§37.2)",
            "",
            f"- log-likelihood {_num(hrg.log_likelihood)} of the best dendrogram found over "
            f"{hrg.restarts} restart(s), {hrg.samples} MCMC steps each, from seed {hrg.seed}",
            f"- cut at the root's own density (default threshold "
            f"{_num(hrg.probabilities[hrg.root], 4)})",
            "",
        ]
    sbm = analysis.sbm
    if sbm is not None:
        lines += [
            "### Stochastic blockmodel (§35.1)",
            "",
            f"- implementation: {sbm.implementation}",
            f"- k={sbm.k}, degree-corrected={str(sbm.degree_corrected).lower()}, "
            f"log-likelihood {_num(sbm.log_likelihood)}, seed {sbm.seed}",
        ]
        if sbm.choices:
            lines += ["", "k tried against BIC (lower is better):", ""]
            lines += _table(
                ["k", "log-likelihood", "BIC"],
                [
                    [_num(row["k"]), _num(row["log_likelihood"]), _num(row["bic"])]
                    for row in sbm.choices
                ],
            )
        else:
            lines.append("")
    infomap = analysis.infomap
    if infomap is not None:
        lines += [
            "### Infomap / map equation (§35.2)",
            "",
            f"- code length {_num(infomap.code_length, 4)} bits, best of "
            f"{len(infomap.code_lengths)} run(s) (range {_num(min(infomap.code_lengths), 4)} to "
            f"{_num(max(infomap.code_lengths), 4)}), {infomap.sweeps} sweep(s)",
            f"- mean pairwise adjusted Rand index between runs: {_num(infomap.stability)}",
            f"- seeds: {', '.join(str(s) for s in infomap.seeds)}",
            "",
        ]
    lp = analysis.label_propagation_result
    if lp is not None:
        lines += [
            "### Label propagation (§35.3)",
            "",
            f"- mean pairwise adjusted Rand index between runs: {_num(lp.stability)}",
            f"- seeds: {', '.join(str(s) for s in lp.seeds)}",
            "",
            _stability_reading(lp.stability),
            "",
        ]
    return lines


def _render_roles(analysis: Analysis) -> list[str]:
    """The short form of chapter 15: role counts, the hubs, and the same-role pairs.

    Short on purpose. `sna roles` is the full report -- every node's coordinates, the other four
    similarities, the blockmodel -- and this is the part a reader of a community report needs
    next to the communities that produced it.
    """
    table = analysis.roles
    if table is None:
        return []
    counts = table.counts()
    lines = [
        "## Roles",
        "",
        f"Atlas §15.1: every node placed by how well connected it is inside its own "
        f"{analysis.group_noun} (z) and how evenly its ties spread across the others (P), then "
        f"cut into the seven regions of Guimera and Amaral's cartography. The roles are relative "
        f"to the {len(analysis.groups)} {analysis.group_plural} above -- another resolution or "
        f"another seed would move them -- and with {table.modules} of them P cannot exceed "
        f"{table.ceiling:.2f}. n = {int(analysis.summary['nodes']):,} nodes. Null model: none; a "
        "role is a deterministic function of the network and the partition.",
        "",
    ]
    lines += _table(
        ["role", "name", "ch. 15 calls it", "nodes"],
        [
            [f"`{role}`", name, ATLAS_ROLE[role], str(counts[role])]
            for role, name in ROLE_NAMES.items()
            if counts[role]
        ],
    )
    hubs = table.hubs()
    if hubs:
        lines += _table(
            ["hub", analysis.group_noun, "z", "P", "role"],
            [
                [
                    analysis.label(row.node),
                    str(row.module + 1),
                    _num(row.within_module_degree, 2),
                    _num(row.participation, 3),
                    f"{row.role} {row.role_name}",
                ]
                for row in hubs[:TOP_N]
            ],
        )
    else:
        lines += [
            f"No node reached z >= {table.thresholds.hub:g}, so this run names no hubs.",
            "",
        ]
    if analysis.role_pairs:
        lines += [
            f"**Same role, never together.** Pairs in the same region whose neighbourhoods are "
            f"at least 0.5 alike by {analysis.role_similarity} similarity (§15.2) and which "
            "share no edge. Structural equivalence is about neighbours, not about being "
            "connected, so this is the corpus discussing two things in the same company without "
            "ever putting them in one passage.",
            "",
        ]
        lines += _table(
            ["node", "node", "role", "similarity", f"same {analysis.group_noun}?"],
            [
                [
                    analysis.label(pair.u),
                    analysis.label(pair.v),
                    pair.role,
                    _num(pair.similarity, 3),
                    "yes" if pair.same_module else "no",
                ]
                for pair in analysis.role_pairs
            ],
        )
    return lines


def _stability_reading(stability: float) -> str:
    if stability >= 0.9:
        return "Every seed found essentially the same partition, so individual memberships hold."
    if stability >= 0.6:
        return (
            "The large groups are stable but the boundaries move between seeds. Quote the "
            "groups, not the membership of any one node near an edge."
        )
    return (
        "The partition changes with the seed. Report that the network has no stable community "
        "structure at this resolution rather than naming communities."
    )


def to_payload(analysis: Analysis) -> dict[str, Any]:
    """The same analysis as plain JSON-able data."""
    louvain_result = analysis.louvain_result
    null_model = analysis.null_model
    choice = analysis.choice
    km = analysis.kmeans_result
    mixture = analysis.gmm_result
    payload: dict[str, Any] = {
        "persona_id": analysis.persona_id,
        "network": analysis.network,
        "method": analysis.method,
        "features": analysis.features,
        "feature_dims": analysis.feature_dims,
        "seed": analysis.seed,
        "generated_at": analysis.generated_at,
        "tool_version": __version__,
        "rationale": analysis.rationale,
        "frame": analysis.graph.graph.get("frame", ""),
        "filters": {
            "min_weight": analysis.graph.graph.get("min_weight"),
            **{key: analysis.graph.graph.get(key) or None for key, _ in FILTER_KEYS},
        },
        "summary": analysis.summary,
        "centrality": {
            kind: [{"node": n, "label": analysis.label(n), "score": s} for n, s in scores]
            for kind, scores in analysis.centralities.items()
        },
        "brokers": [
            {"node": n, "label": analysis.label(n), "participation": s} for n, s in analysis.brokers
        ],
        "groups": [
            {
                "index": i,
                "size": len(group),
                "members": [{"node": n, "label": analysis.label(n)} for n in group],
            }
            for i, group in enumerate(analysis.groups, start=1)
        ],
        "notes": analysis.notes,
        "caveats": list(ALWAYS),
    }
    if analysis.against_random is not None:
        payload["against_random"] = against_random_payload(analysis.against_random)
    if analysis.paths is not None:
        payload["paths"] = paths_payload(analysis.paths)
    if analysis.hierarchy is not None:
        payload["hierarchy"] = hierarchy_payload(analysis.hierarchy)
    if analysis.density is not None:
        payload["density"] = density_payload(analysis.density)
    if analysis.roles is not None:
        payload["roles"] = {
            "implements": "Atlas §15.1 (Guimera-Amaral roles over the grouping above)",
            "modules": analysis.roles.modules,
            "participation_ceiling": analysis.roles.ceiling,
            "counts": analysis.roles.counts(),
            "null_model": "none: a role is a function of the network and the partition",
            "hubs": [
                {
                    "node": row.node,
                    "label": analysis.label(row.node),
                    "module": row.module,
                    "within_module_degree": row.within_module_degree,
                    "participation": row.participation,
                    "role": row.role,
                    "role_name": row.role_name,
                }
                for row in analysis.roles.hubs()[:TOP_N]
            ],
            "same_role_never_cooccur": {
                "similarity": analysis.role_similarity,
                "pairs": [
                    {
                        "nodes": [pair.u, pair.v],
                        "labels": [analysis.label(pair.u), analysis.label(pair.v)],
                        "role": pair.role,
                        "similarity": pair.similarity,
                        "same_module": pair.same_module,
                    }
                    for pair in analysis.role_pairs
                ],
            },
        }
    if analysis.multilayer is not None:
        payload["layers"] = {
            "layering": analysis.multilayer.layering,
            "omega": analysis.multilayer.omega,
            "frame": analysis.multilayer.frame,
            "layers": [
                {
                    "name": row.name,
                    "nodes": row.nodes,
                    "edges": row.edges,
                    "weight": row.weight,
                    "isolated": row.isolated,
                }
                for row in layer_table(analysis.multilayer)
            ],
        }
    if analysis.degree is not None:
        payload["degree"] = degree_payload(analysis.degree)
    if analysis.ranking is not None:
        payload["ranking"] = ranking_payload(analysis.ranking)
    if analysis.ties is not None:
        payload["ties"] = tie_payload(analysis.ties)
    if analysis.evaluation is not None:
        payload["evaluation"] = evaluation_payload(analysis.evaluation)
    if analysis.bipartite is not None:
        payload["bipartite"] = bipartite_payload(analysis.bipartite)
    if analysis.attribute is not None:
        payload["attribute"] = attribute_payload(analysis.attribute)
    if analysis.numeric is not None:
        payload["numeric_attribute"] = numeric_payload(analysis.numeric)
    if analysis.degree_correlations is not None:
        payload["degree_correlations"] = degree_correlations_payload(analysis.degree_correlations)
    if analysis.coreperiphery is not None:
        payload["core_periphery"] = core_periphery_payload(analysis.coreperiphery)
    if analysis.uncertainty is not None:
        payload["uncertainty"] = uncertain_payload(analysis.uncertainty)
    elif analysis.evidence is not None:
        # The evidence section is printed whether or not the sampling ran, so its payload is
        # the same object either way: the ``evidence`` key of the uncertainty payload.
        payload["evidence"] = uncertain_payload(
            Uncertainty(
                samples=0,
                seed=None,
                level=0.0,
                evidence=analysis.evidence,
                expected_edges=0.0,
                expected_density=0.0,
                observed_edges=0,
                observed_density=0.0,
            )
        )["evidence"]
    if louvain_result is not None:
        payload["louvain"] = {
            "modularity": louvain_result.modularity,
            "modularities": louvain_result.modularities,
            "resolution": louvain_result.resolution,
            "seeds": louvain_result.seeds,
            "stability_adjusted_rand": louvain_result.stability,
            "sizes": louvain_result.sizes,
        }
    if null_model is not None:
        payload["null_model"] = {
            "observed": null_model.observed,
            "null_mean": null_model.null_mean,
            "null_std": null_model.null_std,
            "z_score": null_model.z_score,
            "samples": null_model.samples,
            "verdict": null_model.verdict,
        }
    if choice is not None:
        payload["choose_k"] = {
            "rows": choice.rows,
            "best_silhouette_k": choice.best_silhouette_k,
            "elbow_k": choice.elbow_k,
            "best_bic_k": choice.best_bic_k,
        }
    if km is not None:
        payload["kmeans"] = {
            "k": km.k,
            "silhouette": km.silhouette,
            "inertia": km.inertia,
            "seed": km.seed,
        }
    if mixture is not None:
        payload["gmm"] = {
            "k": mixture.k,
            "covariance": mixture.covariance,
            "bic": mixture.bic,
            "aic": mixture.aic,
            "converged": mixture.converged,
            "seed": mixture.seed,
            "responsibilities": mixture.responsibilities,
        }
    if analysis.dendrogram is not None:
        dendrogram = analysis.dendrogram
        payload["dendrogram"] = {
            "method": dendrogram.method,
            "cut": dendrogram.cut,
            "peaks": dendrogram.peaks,
            "resolution": dendrogram.resolution,
            "levels": [
                {
                    "size": level.size,
                    "modularity": level.modularity,
                    "density": level.density,
                }
                for level in dendrogram.levels
            ],
        }
    if analysis.hrg is not None:
        hrg = analysis.hrg
        payload["hrg"] = {
            "log_likelihood": hrg.log_likelihood,
            "restarts": hrg.restarts,
            "samples": hrg.samples,
            "seed": hrg.seed,
            "cut_probability": hrg.probabilities[hrg.root],
        }
    if analysis.sbm is not None:
        sbm = analysis.sbm
        payload["sbm"] = {
            "k": sbm.k,
            "degree_corrected": sbm.degree_corrected,
            "log_likelihood": sbm.log_likelihood,
            "implementation": sbm.implementation,
            "choices": sbm.choices,
            "sweeps": sbm.sweeps,
            "seed": sbm.seed,
        }
    if analysis.infomap is not None:
        infomap = analysis.infomap
        payload["infomap"] = {
            "code_length": infomap.code_length,
            "code_lengths": infomap.code_lengths,
            "seeds": infomap.seeds,
            "stability_adjusted_rand": infomap.stability,
            "sweeps": infomap.sweeps,
        }
    if analysis.label_propagation_result is not None:
        lp = analysis.label_propagation_result
        payload["label_propagation"] = {
            "seeds": lp.seeds,
            "stability_adjusted_rand": lp.stability,
        }
    return payload
