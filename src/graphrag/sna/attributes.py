"""Does this network sort by something we already know about its nodes?

Community detection asks the graph where its groups are. An attribute asks a different question:
here is a property somebody recorded about each node -- which region a speaker is in, which half
of the corpus a document belongs to -- does the network divide along it? The two questions are
easy to confuse and the confusion is expensive, because a partition handed to a graph always
produces a number, and the number always looks like a finding.

So every measure here comes with what it would have been by chance:

*Assortativity* is the correlation between the labels at the two ends of an edge: 1 when every
edge joins like to like, 0 when the labels are spread as a coin would spread them, negative when
the network joins unlike to unlike. Its null is a permutation: the same graph, the same labels,
shuffled between the nodes, many times. That null holds the network and the label distribution
fixed and varies only which node got which label, which is exactly the claim under test.

*Modularity of the attribute partition* asks the same thing in the currency Louvain reports, so
the two can be read side by side: how much better does grouping by this attribute explain the
edges than a degree-preserving random graph, and how far short of the best grouping Louvain can
find does it fall. A partition that scores near Louvain's best *is* the community structure; one
that scores well above its own null but far below Louvain has found something real that is not
the main division in the network.

The degree-preserving null here scores the *same* fixed partition on each rewired graph, unlike
:func:`graphrag.sna.cluster.null_model_modularity`, which re-runs Louvain on every sample. The
difference is not an oversight. Louvain's partition is fitted to the graph, so transferring it to
another graph measures nothing but transfer; an attribute partition is fitted to nothing, so the
honest question is whether these labels explain these edges better than the same labels would
explain a random graph with the same degrees.

*Agreement with the clustering* (adjusted Rand index and normalised mutual information) says
whether the groups the run found are the attribute in disguise. The adjusted index is corrected
for chance; the mutual information is not, so a random baseline is computed for both by shuffling
the cluster labels.

One caveat runs through all of it and is printed with the numbers: an attribute is a hypothesis
about structure, not a finding. Nodes nobody tagged are not a third group -- they are outside
every measure here, and the report says how many they were.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import networkx as nx
import numpy as np

from graphrag.sna.cluster import compare_partitions, louvain
from graphrag.sna.export import attr_count_key, attr_key

__all__ = [
    "AttributeReport",
    "analyse_attribute",
    "attribute_labels",
    "attribute_payload",
    "render_attribute",
]

#: How many times the labels are shuffled for the assortativity null.
PERMUTATIONS = 200
#: How many degree-preserving rewirings the fixed-partition null uses.
NULL_SAMPLES = 50
#: How many Louvain seeds the comparison partition is the best of.
LOUVAIN_RUNS = 5
#: Below this many labelled nodes nothing here is worth reporting as a measurement.
MIN_NODES = 4


@dataclass(frozen=True)
class AttributeReport:
    """One attribute measured against one network, with the nulls that make it readable."""

    key: str
    counts: dict[str, int] = field(default_factory=dict)
    labelled: int = 0
    total: int = 0
    mixed: int = 0
    edges_within: int = 0
    edges_between: int = 0
    weight_within: float = 0.0
    weight_between: float = 0.0
    assortativity: float | None = None
    permutations: int = 0
    permuted_mean: float = 0.0
    permuted_std: float = 0.0
    assortativity_z: float = 0.0
    modularity: float = 0.0
    louvain_modularity: float | None = None
    null_samples: int = 0
    null_mean: float = 0.0
    null_std: float = 0.0
    modularity_z: float = 0.0
    group_noun: str = "cluster"
    agreement: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def values(self) -> int:
        return len(self.counts)

    @property
    def group_plural(self) -> str:
        return "communities" if self.group_noun == "community" else f"{self.group_noun}s"

    @property
    def unlabelled(self) -> int:
        return max(self.total - self.labelled, 0)

    @property
    def within_share(self) -> float:
        total = self.edges_within + self.edges_between
        return self.edges_within / total if total else 0.0

    @property
    def verdict(self) -> str:
        """What the assortativity and its permutation null together support, in one sentence."""
        if self.assortativity is None:
            return (
                "Not measurable: fewer than two values, or no edges between labelled nodes. "
                "Report the counts and stop."
            )
        if abs(self.assortativity_z) < 2:
            return (
                "The labels are spread across the edges about as a shuffle would spread them, "
                "so this network does not divide along this attribute."
            )
        if self.assortativity > 0:
            return (
                "Edges join like to like far more than a shuffle of the same labels does, so "
                "the attribute tracks a real division in the network."
            )
        return (
            "Edges join unlike to unlike far more than a shuffle does: the two values are "
            "connected through each other rather than within themselves."
        )


def attribute_labels(graph: nx.Graph, key: str) -> dict[str, str]:
    """Each node's value for one attribute, leaving out the nodes that carry none."""
    data = attr_key(key)
    return {
        node: str(values[data])
        for node, values in graph.nodes(data=True)
        if str(values.get(data, "")).strip()
    }


def analyse_attribute(
    graph: nx.Graph,
    key: str,
    *,
    groups: Sequence[Sequence[str]] = (),
    group_noun: str = "cluster",
    resolution: float = 1.0,
    permutations: int = PERMUTATIONS,
    samples: int = NULL_SAMPLES,
    runs: int = LOUVAIN_RUNS,
    seed: int | None = None,
) -> AttributeReport:
    """Measure how far one node attribute explains this network's edges.

    Everything is computed over the nodes that carry a value, never over the whole network: a
    node nobody tagged has no label to correlate, and quietly treating it as a value of its own
    would make "untagged" the biggest community in most corpora.
    """
    labels = attribute_labels(graph, key)
    counts = Counter(labels.values())
    notes: list[str] = []
    report = AttributeReport(
        key=key,
        counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        labelled=len(labels),
        total=graph.number_of_nodes(),
        mixed=sum(
            1 for node in labels if int(graph.nodes[node].get(attr_count_key(key), 1) or 1) > 1
        ),
        group_noun=group_noun,
        notes=tuple(notes),
    )
    if not labels:
        return _with_notes(
            report,
            [
                f"No node in this network carries '{key}'. Either the tagging pass has not run "
                "or the key is spelled differently in the sidecars."
            ],
        )
    sub: nx.Graph = graph.subgraph(labels).copy()
    within, between, w_within, w_between = _edge_split(sub, labels)
    report = _replace(
        report,
        edges_within=within,
        edges_between=between,
        weight_within=w_within,
        weight_between=w_between,
    )
    if len(labels) < MIN_NODES or len(counts) < 2 or sub.number_of_edges() == 0:
        notes.append(
            f"{len(labels)} labelled node(s) over {len(counts)} value(s) with "
            f"{sub.number_of_edges()} edge(s) between them: too little to measure. The value "
            "counts above are the whole finding."
        )
        return _with_notes(report, notes)

    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    observed, permuted = _assortativity(sub, key, labels, permutations, rng)
    groups_by_value = [
        sorted(n for n, v in labels.items() if v == value) for value in sorted(counts)
    ]
    modularity = float(
        nx.community.modularity(sub, groups_by_value, weight="weight", resolution=resolution)
    )
    best = louvain(sub, resolution=resolution, seed=seed, runs=runs)
    null = _fixed_partition_null(sub, groups_by_value, samples, resolution, rng)
    report = _replace(
        report,
        assortativity=observed,
        permutations=len(permuted),
        permuted_mean=_mean(permuted),
        permuted_std=_std(permuted),
        assortativity_z=_z(observed, permuted) if observed is not None else 0.0,
        modularity=modularity,
        louvain_modularity=best.modularity,
        null_samples=len(null),
        null_mean=_mean(null),
        null_std=_std(null),
        modularity_z=_z(modularity, null),
        agreement=_agreement(groups, labels, rng),
    )
    if report.mixed:
        notes.append(
            f"{report.mixed} of {report.labelled} labelled node(s) had passages in documents "
            "that disagreed about this attribute; each was given the value most of its "
            "documents carry. Read those nodes as a majority, not as a fact."
        )
    if report.unlabelled:
        notes.append(
            f"{report.unlabelled} of {report.total} node(s) carry no value for '{key}' and are "
            "in none of these measures. They are untagged, not a third value."
        )
    return _with_notes(report, notes)


def _replace(report: AttributeReport, **changes: Any) -> AttributeReport:
    """The report with some fields changed. It is frozen, so every step builds a new one."""
    return replace(report, **changes)


def _with_notes(report: AttributeReport, notes: Sequence[str]) -> AttributeReport:
    return _replace(report, notes=tuple(notes))


def _edge_split(graph: nx.Graph, labels: Mapping[str, str]) -> tuple[int, int, float, float]:
    """Edges and edge weight inside one value against edges crossing between two."""
    within = between = 0
    w_within = w_between = 0.0
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0) or 1.0)
        if labels[u] == labels[v]:
            within += 1
            w_within += weight
        else:
            between += 1
            w_between += weight
    return within, between, w_within, w_between


def _assortativity(
    graph: nx.Graph,
    key: str,
    labels: Mapping[str, str],
    permutations: int,
    rng: random.Random,
) -> tuple[float | None, list[float]]:
    """The observed attribute assortativity, and the same measure over shuffled labels.

    The permutation keeps the graph and the label counts exactly as they are and moves only
    which node holds which label. That is the null a claim about sorting needs: a network with
    two dense clumps and labels sprinkled at random will score near zero here, however obvious
    the clumps are, because the clumps are not what the attribute is about.
    """
    data = attr_key(key)
    observed = _coefficient(graph, data)
    if observed is None:
        return None, []
    nodes = sorted(labels)
    values = [labels[node] for node in nodes]
    scores: list[float] = []
    for _ in range(max(permutations, 0)):
        rng.shuffle(values)
        for node, value in zip(nodes, values, strict=True):
            graph.nodes[node][data] = value
        score = _coefficient(graph, data)
        if score is not None:
            scores.append(score)
    for node in nodes:  # put the real labels back; the caller still holds this subgraph
        graph.nodes[node][data] = labels[node]
    return observed, scores


def _coefficient(graph: nx.Graph, data: str) -> float | None:
    """``nx.attribute_assortativity_coefficient``, with its undefined cases turned into None."""
    if graph.number_of_edges() == 0:
        return None
    try:
        value = float(nx.attribute_assortativity_coefficient(graph, data))
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        return None
    return None if math.isnan(value) else value


def _fixed_partition_null(
    graph: nx.Graph,
    partition: Sequence[Sequence[str]],
    samples: int,
    resolution: float,
    rng: random.Random,
) -> list[float]:
    """Score this exact partition on degree-preserving rewirings of the same graph.

    The partition is held fixed because it was never fitted to the graph in the first place.
    Re-running Louvain on each sample, as the community null does, would answer a different
    question: how good a partition the rewired graph admits, which is not what an attribute
    claims.
    """
    edges = graph.number_of_edges()
    if graph.number_of_nodes() < MIN_NODES or edges < 2:
        return []
    weights = [float(d.get("weight", 1.0) or 1.0) for _, _, d in graph.edges(data=True)]
    scores: list[float] = []
    for _ in range(max(samples, 0)):
        rewired = graph.copy()
        try:
            nx.double_edge_swap(
                rewired, nswap=edges, max_tries=edges * 20, seed=rng.randrange(1_000_000)
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue  # too few distinct degrees to rewire; this sample contributes nothing
        shuffled = list(weights)
        rng.shuffle(shuffled)
        for (u, v), weight in zip(rewired.edges(), shuffled, strict=True):
            rewired[u][v]["weight"] = weight
        scores.append(
            float(
                nx.community.modularity(rewired, partition, weight="weight", resolution=resolution)
            )
        )
    return scores


def _agreement(
    groups: Sequence[Sequence[str]], labels: Mapping[str, str], rng: random.Random
) -> dict[str, float]:
    """How far the run's groups and the attribute labels are the same partition.

    Computed over the nodes that are in both, with a baseline from shuffling the group labels:
    the adjusted Rand index is already chance-corrected and should land near zero, and the
    mutual information is not, so its baseline is the only thing that makes it readable.
    """
    membership = {node: index for index, group in enumerate(groups) for node in group}
    shared = sorted(set(membership) & set(labels))
    if len(shared) < MIN_NODES or len({labels[n] for n in shared}) < 2:
        return {}
    values = sorted({labels[node] for node in shared})
    left = [membership[node] for node in shared]
    right = [values.index(labels[node]) for node in shared]
    scores = compare_partitions(left, right)
    baseline_ari: list[float] = []
    baseline_nmi: list[float] = []
    shuffled = list(left)
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        random_scores = compare_partitions(shuffled, right)
        baseline_ari.append(random_scores["adjusted_rand_index"])
        baseline_nmi.append(random_scores["normalized_mutual_information"])
    return {
        "adjusted_rand_index": scores["adjusted_rand_index"],
        "normalized_mutual_information": scores["normalized_mutual_information"],
        "baseline_adjusted_rand_index": _mean(baseline_ari),
        "baseline_normalized_mutual_information": _mean(baseline_nmi),
        "nodes": float(len(shared)),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _z(observed: float, null: Sequence[float]) -> float:
    std = _std(null)
    return (observed - _mean(null)) / std if std > 0 else 0.0


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_attribute(report: AttributeReport) -> list[str]:
    """The attribute section of a report: the counts first, then what they explain."""
    lines = [
        f"## By attribute: {report.key}",
        "",
        f"{report.labelled:,} of {report.total:,} nodes carry a value for `{report.key}`, over "
        f"{report.values} value(s). An attribute is a hypothesis about where this network "
        "divides, not a finding: the numbers below say how much of the network's shape it "
        "accounts for, and every one of them is computed over the labelled nodes only.",
        "",
    ]
    lines += _table(
        ["value", "nodes", "share of labelled"],
        [
            [value, f"{count:,}", f"{count / max(report.labelled, 1):.0%}"]
            for value, count in report.counts.items()
        ],
    )
    lines += [
        "### Edges within and between values",
        "",
        f"- edges inside one value: {report.edges_within:,} "
        f"({report.within_share:.0%} of edges between labelled nodes)",
        f"- edges between two values: {report.edges_between:,}",
        f"- weight inside: {_num(report.weight_within)}; "
        f"weight between: {_num(report.weight_between)}",
        "",
    ]
    if report.assortativity is not None:
        lines += [
            "### Assortativity",
            "",
            f"- observed: {_num(report.assortativity)}",
            f"- {report.permutations} label shuffles: mean {_num(report.permuted_mean)}, "
            f"sd {_num(report.permuted_std)}",
            f"- z-score: {_num(report.assortativity_z, 2)}",
            "",
            report.verdict,
            "",
        ]
    if report.louvain_modularity is not None:
        lines += [
            "### Modularity of the attribute partition",
            "",
            f"- grouping by `{report.key}`: {_num(report.modularity)}",
            f"- best Louvain grouping of the same nodes: {_num(report.louvain_modularity)}",
            f"- {report.null_samples} degree-preserving rewirings scoring the same partition: "
            f"mean {_num(report.null_mean)}, sd {_num(report.null_std)}, "
            f"z {_num(report.modularity_z, 2)}",
            "",
            _modularity_reading(report),
            "",
        ]
    if report.agreement:
        lines += [
            f"### The {report.group_plural} against the attribute",
            "",
        ]
        lines += _table(
            ["measure", "observed", "random baseline"],
            [
                [
                    "adjusted Rand index",
                    _num(report.agreement["adjusted_rand_index"]),
                    _num(report.agreement["baseline_adjusted_rand_index"]),
                ],
                [
                    "normalised mutual information",
                    _num(report.agreement["normalized_mutual_information"]),
                    _num(report.agreement["baseline_normalized_mutual_information"]),
                ],
            ],
        )
        lines += [
            f"Over the {int(report.agreement['nodes']):,} node(s) that carry both a "
            f"{report.group_noun} and a value. The baseline is the same comparison with the "
            f"{report.group_noun} labels shuffled, which is what 'no relationship' scores.",
            "",
        ]
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return lines


def _modularity_reading(report: AttributeReport) -> str:
    """What the three modularity numbers say together, which is more than any one of them."""
    best = report.louvain_modularity or 0.0
    if report.modularity_z < 2:
        return (
            "The attribute partition explains the edges no better than the same partition would "
            "explain a random graph with these degrees. Whatever divides this network, it is "
            "not this."
        )
    if best and report.modularity >= 0.8 * best:
        return (
            "The attribute partition is close to the best grouping Louvain found, so the "
            "communities in this network are largely this attribute."
        )
    return (
        "The attribute partition beats its null but falls well short of Louvain's best, so it "
        "is a real division and not the main one. Name what the Louvain groups have in common "
        "before treating this attribute as the explanation."
    )


def attribute_payload(report: AttributeReport) -> dict[str, object]:
    """The same section as plain JSON-able data, for the ``--json`` output."""
    return {
        "key": report.key,
        "counts": report.counts,
        "labelled": report.labelled,
        "total": report.total,
        "unlabelled": report.unlabelled,
        "mixed": report.mixed,
        "edges_within": report.edges_within,
        "edges_between": report.edges_between,
        "weight_within": report.weight_within,
        "weight_between": report.weight_between,
        "assortativity": report.assortativity,
        "permutations": report.permutations,
        "permuted_mean": report.permuted_mean,
        "permuted_std": report.permuted_std,
        "assortativity_z": report.assortativity_z,
        "verdict": report.verdict,
        "modularity": report.modularity,
        "louvain_modularity": report.louvain_modularity,
        "null_samples": report.null_samples,
        "null_mean": report.null_mean,
        "null_std": report.null_std,
        "modularity_z": report.modularity_z,
        "agreement": report.agreement,
        "notes": list(report.notes),
    }
