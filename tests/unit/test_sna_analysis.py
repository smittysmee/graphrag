"""What one ``sna analyze`` run has to print, whatever else it found.

The sections of a report are not optional extras: a reader who is handed a clustering coefficient
without the random-graph expectation beside it cannot tell 0.57 from 0.0089, and Atlas ch. 17 is
an argument that nobody can. So the "Against random" section is asserted here on graphs whose
answer is known before the run -- a small-world graph, which is clustered, and a random graph,
which is not -- in the markdown and in the JSON payload alike.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import render_markdown, run_analysis, to_payload
from graphrag.sna.cluster import null_model_modularity
from graphrag.sna.generators import erdos_renyi_gnm, watts_strogatz


def _analysis(graph: nx.Graph, **kwargs: object):  # type: ignore[no-untyped-def]
    graph.graph.update(min_weight=1, frame="A generated network, for a test.")
    defaults: dict[str, object] = {
        "persona_id": "test-pm",
        "network": "speakers",
        "method": "louvain",
        "seed": 3,
        "runs": 2,
        "samples": 10,
    }
    defaults.update(kwargs)
    return run_analysis(InMemoryGraphStore(), graph, **defaults)  # type: ignore[arg-type]


def test_a_small_world_report_says_it_is_clustered_and_names_both_nulls() -> None:
    """§17.1 on a graph built to be clustered: the ratio is large and the null agrees."""
    analysis = _analysis(watts_strogatz(150, 6, 0.05, seed=7))
    report = analysis.against_random
    assert report is not None
    assert report.clustering_ratio > 10
    assert report.clustered
    text = render_markdown(analysis)
    assert "## Against random" in text
    assert "**Implements.** §16.2-16.5" in text
    assert "average clustering (§17.1)" in text
    assert "degree-preserving" in text
    assert "Filters: min_weight=1." in text
    assert "this network is clustered" in text


def test_a_random_network_report_says_it_is_not() -> None:
    """The same section on a G(n,m) graph: every ratio at 1, and the word "not clustered"."""
    analysis = _analysis(erdos_renyi_gnm(150, 450, seed=7))
    report = analysis.against_random
    assert report is not None
    assert report.clustering_ratio == pytest.approx(1.0, abs=0.6)
    assert not report.clustered
    assert "not clustered" in render_markdown(analysis)


# ----------------------------------------------- --features node2vec / metapath2vec (ATL-43)


def _two_disjoint_cliques(size: int) -> nx.Graph:
    graph = nx.Graph()
    for block in range(2):
        nodes = [f"g{block}n{i}" for i in range(size)]
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                graph.add_edge(u, v, weight=1)
    return graph


def test_features_node2vec_clusters_two_disjoint_cliques_and_evaluates_the_partition() -> None:
    analysis = _analysis(
        _two_disjoint_cliques(6),
        method="kmeans",
        features="node2vec",
        dims=4,
        k=2,
    )
    assert analysis.features == "node2vec"
    assert analysis.feature_dims == 4
    assert {frozenset(group) for group in analysis.groups} == {
        frozenset(f"g0n{i}" for i in range(6)),
        frozenset(f"g1n{i}" for i in range(6)),
    }
    # ch. 36's battery runs over every clustering, not only Louvain's (wave 4's own rule).
    assert analysis.evaluation is not None
    text = render_markdown(analysis)
    assert "node2vec" in text
    assert "transductive" in text


def test_features_metapath2vec_needs_a_two_mode_network() -> None:
    with pytest.raises(ValueError, match="needs a two-mode network"):
        _analysis(
            _two_disjoint_cliques(4),
            method="kmeans",
            features="metapath2vec",
            dims=2,
            k=2,
        )


def test_features_metapath2vec_clusters_two_disjoint_speaker_groups() -> None:
    graph = nx.Graph()
    groups = {
        "a": (["sa0", "sa1", "sa2"], ["ea0", "ea1"]),
        "b": (["sb0", "sb1", "sb2"], ["eb0", "eb1"]),
    }
    for speakers, entities in groups.values():
        for speaker in speakers:
            graph.add_node(speaker, mode="speaker")
        for entity in entities:
            graph.add_node(entity, mode="entity")
        for speaker in speakers:
            for entity in entities:
                graph.add_edge(speaker, entity, weight=1)
    analysis = _analysis(
        graph,
        method="kmeans",
        features="metapath2vec",
        dims=4,
        k=2,
        network="speakers-entities",
    )
    assert analysis.features == "metapath2vec"
    speaker_groups = [{n for n in group if n.startswith("s")} for group in analysis.groups]
    speaker_groups = [group for group in speaker_groups if group]
    assert {frozenset(group) for group in speaker_groups} == {
        frozenset({"sa0", "sa1", "sa2"}),
        frozenset({"sb0", "sb1", "sb2"}),
    }


def test_the_section_reaches_the_payload_under_a_stable_key() -> None:
    analysis = _analysis(watts_strogatz(60, 4, 0.1, seed=7))
    payload = to_payload(analysis)["against_random"]
    assert payload["nodes"] == 60
    assert payload["samples"] == 10
    assert payload["erdos_renyi"]["average_clustering"] == pytest.approx(
        payload["observed"]["average_clustering"] / payload["ratios"]["clustering"]
    )
    assert payload["configuration_null"]["average_clustering"]["null"] == "configuration"
    assert payload["clustered"] is True
    assert "clustered" in payload["verdict"]


def test_the_shared_null_family_leaves_the_modularity_null_where_it_was() -> None:
    """One draw, two sections, and the same numbers as when each drew its own (§19.1).

    The report needs §19.1's rewirings twice -- for Louvain's modularity null and for chapter
    17's clustering and path length -- and draws them once, because the draw is the expensive
    half. The saving is only allowed if it changes nothing: this pins the modularity null to
    what ``null_model_modularity`` produces when it draws the family itself, at the same seed
    and the same count.
    """
    graph = watts_strogatz(80, 6, 0.1, seed=7)
    analysis = _analysis(graph)
    assert analysis.null_model is not None
    assert analysis.against_random is not None
    alone = null_model_modularity(graph, analysis.groups, samples=10, seed=3, resolution=1.0)
    assert analysis.null_model.samples == alone.samples == 10
    assert analysis.null_model.null_mean == pytest.approx(alone.null_mean)
    assert analysis.null_model.z_score == pytest.approx(alone.z_score)
    # and the same family reached the other section
    assert analysis.against_random.samples == 10


def test_an_empty_network_has_no_section_to_print() -> None:
    """Nothing to compare, and a report that invented a comparison would be worse than silent."""
    analysis = _analysis(nx.Graph())
    assert analysis.against_random is None
    assert "## Against random" not in render_markdown(analysis)
    assert "against_random" not in to_payload(analysis)
