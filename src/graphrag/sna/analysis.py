"""Run one analysis end to end and render it as a markdown note or a JSON payload.

The report is deliberately opinionated about what has to appear next to a result: how many
nodes it covers, where the network was sampled from, how the method was chosen, whether the
partition survives a different random seed, and whether it beats a degree-preserving null
model. Those are the things a reader needs in order to disagree with the conclusion.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag import __version__
from graphrag.graph.store import GraphStore
from graphrag.sna.cluster import (
    GMMResult,
    KChoice,
    KMeansResult,
    LouvainResult,
    NullModelResult,
    choose_k_gmm,
    choose_k_kmeans,
    gmm,
    kmeans,
    louvain,
    null_model_modularity,
    spectral_embedding,
)
from graphrag.sna.guide import ALWAYS, rationale
from graphrag.sna.measures import (
    CENTRALITIES,
    CENTRALITY_MEANING,
    brokers,
    centrality,
    summary,
    top_n,
)

Method = Literal["louvain", "kmeans", "gmm"]
Features = Literal["spectral", "embedding"]
METHODS: tuple[Method, ...] = ("louvain", "kmeans", "gmm")
FEATURE_KINDS: tuple[Features, ...] = ("spectral", "embedding")

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
    louvain_result: LouvainResult | None = None
    kmeans_result: KMeansResult | None = None
    gmm_result: GMMResult | None = None
    choice: KChoice | None = None
    null_model: NullModelResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def group_plural(self) -> str:
        return "communities" if self.group_noun == "community" else f"{self.group_noun}s"

    def label(self, node: str) -> str:
        data = self.graph.nodes.get(node, {})
        text = data.get("label") or data.get("name") or node
        return str(text)


def _feature_rows(
    store: GraphStore,
    graph: nx.Graph,
    network: str,
    persona_id: str,
    features: Features,
    dims: int,
    seed: int | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """The nodes to cluster, their feature matrix, and any notes about what was dropped."""
    if features == "spectral":
        nodes = list(graph.nodes)
        return nodes, spectral_embedding(graph, dims, seed), []
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


def _grouped(nodes: Sequence[str], labels: Sequence[int]) -> list[list[str]]:
    buckets: dict[int, list[str]] = {}
    for node, label in zip(nodes, labels, strict=True):
        buckets.setdefault(int(label), []).append(node)
    return [sorted(buckets[key]) for key in sorted(buckets)]


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
    covariance: str = "full",
    samples: int = 50,
    seed: int | None = None,
) -> Analysis:
    """Measure the network, group it with the chosen method, and check the grouping."""
    stats = summary(graph)
    centralities = {kind: top_n(centrality(graph, kind), TOP_N) for kind in CENTRALITIES}
    notes: list[str] = []
    analysis = Analysis(
        persona_id=persona_id,
        network=network,
        method=method,
        graph=graph,
        summary=stats,
        centralities=centralities,
        groups=[],
        group_noun="community" if method == "louvain" else "cluster",
        rationale=rationale(method),
        seed=seed,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
    )
    if graph.number_of_nodes() == 0:
        notes.append("The network is empty: no nodes matched this persona and these filters.")
        return analysis

    if method == "louvain":
        result = louvain(graph, resolution=resolution, seed=seed, runs=runs)
        analysis.louvain_result = result
        analysis.groups = result.communities
        analysis.null_model = null_model_modularity(
            graph, result.communities, samples=samples, seed=seed, resolution=resolution
        )
    else:
        nodes, matrix, feature_notes = _feature_rows(
            store, graph, network, persona_id, features, dims, seed
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
    return analysis


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
    ("since", "since"),
    ("until", "until"),
    ("project", "projected onto"),
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
    frame = str(meta.get("frame", ""))
    lines += [f"**Sampling frame.** {frame} Filters: {_filter_line(meta)}.", ""]

    lines += ["## Network summary", ""]
    lines += _table(
        ["measure", "value"],
        [[key.replace("_", " "), _num(value)] for key, value in analysis.summary.items()],
    )
    if analysis.summary["nodes"] < 30:
        lines += [
            f"> Only {int(analysis.summary['nodes'])} nodes. Read the rankings below as a "
            "description of this small network, not as an estimate of anything wider.",
            "",
        ]

    if analysis.groups:
        lines += [f"## {analysis.group_plural.capitalize()} ({len(analysis.groups)})", ""]
        lines += _table(
            ["#", "size", f"top members by weighted degree (up to {TOP_MEMBERS})"],
            [
                [str(i), str(len(group)), _members(analysis, group)]
                for i, group in enumerate(analysis.groups, start=1)
            ],
        )

    lines += _cross_mode(analysis)
    lines += _render_checks(analysis)

    lines += ["## Centrality", ""]
    for kind, scores in analysis.centralities.items():
        if not scores:
            continue
        lines += [f"### {kind}", "", CENTRALITY_MEANING[kind], ""]
        lines += _table(
            ["rank", "node", "score"],
            [
                [str(i), analysis.label(node), _num(score)]
                for i, (node, score) in enumerate(scores, start=1)
            ],
        )

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

    lines += ["## Caveats", ""]
    lines += [f"- {item}" for item in ALWAYS]
    lines += [f"- {note}" for note in analysis.notes]
    return "\n".join(lines).rstrip() + "\n"


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
    return payload
