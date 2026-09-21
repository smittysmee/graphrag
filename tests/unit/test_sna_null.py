"""Null models, on graphs whose answer is known before the code runs.

Three kinds of check live here, and they are different kinds of claim.

*Invariants.* Every null promises to hold something fixed, and the promise is the whole content
of the null: a configuration sample must have the observed degree sequence exactly, a
bipartite-preserving sample must have every left and every right degree of the observed
memberships, a G(n,m) sample must have the observed edge count. These are asserted on the
legendary graphs of ``tests/legendary.py`` -- Zachary's karate club (34 nodes, 78 edges) and the
Davis Southern women (18 women, 14 events, 89 attendances) -- and on planted two-mode corpora.

A directed network is held to both of its degree sequences: a planted ring with four nodes that
write further ahead has out-degrees and in-degrees that differ from each other, and every sample
keeps both, because §19.1's swap only preserves them "provided that you always swap edges in the
correct direction".

*Known answers.* An observation drawn from its own null must score z ~ 0 and p ~ 0.5. An ERGM
fitted to a random graph must find no triangle effect, and one fitted to the karate club must
find a positive one; on a digraph where nine arcs in ten are answered the reciprocity term must
come back strongly positive, and on a random digraph indistinguishable from zero. And the two
confounds this package exists to refuse, each planted so that the answer is known before the code
runs:

- a label that is really *activity* -- six nodes recorded in six documents, fourteen in one --
  beats a label shuffle (|z| > 4) and lands inside the bipartite null (|z| < 2), because the
  second keeps each node's number of documents and the first does not;
- a label that was *borrowed* from the documents that drew the edges scores the largest z-score
  the shuffle can produce (8.5), and falls to about 2 under the bipartite null once that null
  derives the borrowed labels again from each rewired corpus -- with a test beside it showing
  what the same null says when it cannot (12.3, worse than the shuffle), because that difference
  is the whole reason the right-attribute table is carried.

*The move.* ``null_model_modularity`` and the attribute nulls used to own their own rewiring
loops. The numbers they produced for a given seed are pinned here, so moving the loop into
``sna.null`` cannot quietly change a published report.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.attributes import analyse_attribute, render_attribute
from graphrag.sna.cluster import louvain, null_model_modularity
from graphrag.sna.export import (
    INHERITED,
    OWN,
    RIGHT_ATTRIBUTES,
    attr_key,
    attr_source_key,
    bipartite_projection,
    build_network,
    read_graph,
    write_graph,
)
from graphrag.sna.null import (
    ERGM_TERMS,
    HOLDS_FIXED,
    NULLS,
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
from tests.legendary import karate_club, southern_women

SEED = 11


@pytest.fixture
def karate() -> nx.Graph:
    """Zachary's karate club (tests/legendary.py): 34 nodes, 78 edges, one published answer."""
    graph, known = karate_club()
    assert (known.nodes, known.edges) == (34, 78)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def two_cliques(size: int = 8) -> nx.Graph:
    """The attribute tests' planted graph: two cliques, one bridge, labelled by clique."""
    graph = nx.Graph()
    for side in ("north", "south"):
        nodes = [f"{side}-{i}" for i in range(size)]
        graph.add_nodes_from(nodes)
        for a, b in ((x, y) for x in nodes for y in nodes if x < y):
            graph.add_edge(a, b, weight=1)
    graph.add_edge("north-0", "south-0", weight=1)
    for node in graph.nodes:
        graph.nodes[node][attr_key("region")] = node.split("-")[0]
    return graph


def clique_corpus(size: int = 8) -> list[tuple[str, str]]:
    """The memberships that project to exactly ``two_cliques``: one document per clique, plus one
    small document holding one node of each, which is where the bridge edge comes from."""
    pairs = [(f"north-{i}", "doc-north") for i in range(size)]
    pairs += [(f"south-{i}", "doc-south") for i in range(size)]
    pairs += [("north-0", "doc-bridge"), ("south-0", "doc-bridge")]
    return pairs


def activity_corpus(
    documents: int = 10, busy: int = 6, quiet: int = 14, span: int = 6
) -> list[tuple[str, str]]:
    """A corpus where the label is nothing but how often a node was recorded.

    Six "busy" nodes appear in six of the ten documents; fourteen "quiet" nodes appear in one
    each. Nobody is north or south of anything: the only difference between the two values is
    activity, so any assortativity the projection shows is the corpus's shape. A null that keeps
    each node's number of documents must therefore reproduce it, and a null that only shuffles
    labels cannot.
    """
    pairs = [(f"busy-{i}", f"doc-{(i + d) % documents}") for i in range(busy) for d in range(span)]
    pairs += [(f"quiet-{i}", f"doc-{i % documents}") for i in range(quiet)]
    return pairs


#: What the two documents of ``clique_corpus`` were tagged with, which is where the borrowed
#: labels come from. The two-node bridge document is untagged, as a document nobody annotated is.
CLIQUE_DOCUMENTS: dict[str, dict[str, str]] = {
    "doc-north": {"region": "north"},
    "doc-south": {"region": "south"},
    "doc-bridge": {},
}


def labelled_projection(
    pairs: list[tuple[str, str]],
    *,
    source: str = INHERITED,
    documents: dict[str, dict[str, str]] | None = None,
) -> nx.Graph:
    """The projection of a planted corpus, labelled by the prefix of each node id.

    ``source`` is the provenance the real builders record: :data:`OWN` for a value somebody wrote
    about the node, :data:`INHERITED` for one it took from its documents. ``documents`` is the
    table ``entity_co_mention`` keeps, and supplying it is what lets the bipartite null derive the
    borrowed labels again instead of carrying them across.
    """
    graph = bipartite_projection(pairs, side="left")
    for node in graph.nodes:
        graph.nodes[node][attr_key("region")] = node.split("-")[0]
        graph.nodes[node][attr_source_key("region")] = source
        graph.nodes[node]["documents"] = 1
    if documents is not None:
        graph.graph[RIGHT_ATTRIBUTES] = documents
    return graph


def degrees(graph: nx.Graph) -> list[int]:
    return sorted(degree for _, degree in graph.degree())


def side_degrees(pairs: list[tuple[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """How many partners each left node and each right node has in a membership table."""
    left: dict[str, int] = {}
    right: dict[str, int] = {}
    for a, b in set(pairs):
        left[a] = left.get(a, 0) + 1
        right[b] = right.get(b, 0) + 1
    return left, right


# ----------------------------------------------------------------------------- the registry


def test_every_null_in_the_registry_says_what_it_holds_fixed() -> None:
    """A null nobody can name is not reportable, so the registry and the sentences must agree."""
    assert set(NULLS) == set(HOLDS_FIXED)
    assert set(NULLS) == {
        "erdos_renyi",
        "configuration",
        "label_permutation",
        "bipartite_preserving",
    }
    for name, sentence in HOLDS_FIXED.items():
        assert "§" in sentence, f"{name} does not cite the section it comes from"


# ----------------------------------------------------------------------------- §16.1 G(n,m), G(n,p)


def test_a_gnm_sample_keeps_the_node_and_edge_counts_exactly(karate: nx.Graph) -> None:
    samples = list(erdos_renyi(karate, 5, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        assert sample.number_of_nodes() == 34
        assert sample.number_of_edges() == 78
        assert set(sample.nodes) == set(karate.nodes)  # the observed ids, so measures line up
        assert sorted(nx.get_edge_attributes(sample, "weight").values()) == [1.0] * 78


def test_a_gnm_sample_does_not_keep_the_degree_sequence(karate: nx.Graph) -> None:
    """Which is the point of having more than one null: G(n,m) frees what configuration fixes."""
    sample = next(erdos_renyi(karate, 1, seed=SEED))

    assert degrees(sample) != degrees(karate)


def test_a_gnp_sample_lands_near_the_density_it_was_asked_for(karate: nx.Graph) -> None:
    """p·n(n-1)/2 = m is the book's own conversion, so the mean edge count must sit near it."""
    pairs = 34 * 33 / 2
    counts = [sample.number_of_edges() for sample in erdos_renyi(karate, 40, seed=SEED, p=0.2)]

    assert float(np.mean(counts)) == pytest.approx(0.2 * pairs, rel=0.1)
    assert len(set(counts)) > 1  # G(n,p) does not fix m, and must not pretend to


def test_gnp_refuses_a_p_that_is_not_a_probability(karate: nx.Graph) -> None:
    with pytest.raises(ValueError, match="probability"):
        next(erdos_renyi(karate, 1, seed=SEED, p=1.4))


# ----------------------------------------------------------------------------- §18.1 / §19.1 swaps


def test_a_configuration_sample_keeps_every_degree_exactly(karate: nx.Graph) -> None:
    samples = list(configuration(karate, 5, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        assert degrees(sample) == degrees(karate)
        assert sample.number_of_edges() == karate.number_of_edges()
        assert nx.number_of_selfloops(sample) == 0  # §19.1: a swap may not invent a self-loop


def test_a_configuration_sample_is_not_the_observed_graph(karate: nx.Graph) -> None:
    sample = next(configuration(karate, 1, seed=SEED))

    assert set(sample.edges) != set(karate.edges)


def test_a_configuration_sample_of_a_digraph_keeps_every_in_and_out_degree() -> None:
    """§19.1 keeps the degree distribution "also in case of a directed network, provided that you
    always swap edges in the correct direction", so a directed network gets the three-arc swap.

    Planted rather than random in the respect that matters: a directed ring of 24 with four nodes
    that also write further ahead, so the out-degrees are 1 twenty times and 3 four times while
    the in-degrees are 1 sixteen times and 2 eight times. The two sequences are different, so a
    sampler that quietly flattened the network to an undirected one -- or swapped two arcs instead
    of rotating three -- would show up at once as an in-degree that moved.
    """
    graph = nx.DiGraph()
    ring = [f"n{i}" for i in range(24)]
    for index, node in enumerate(ring):
        graph.add_edge(node, ring[(index + 1) % 24], weight=1.0)
    for index in range(0, 24, 6):
        graph.add_edge(ring[index], ring[(index + 5) % 24], weight=1.0)
        graph.add_edge(ring[index], ring[(index + 9) % 24], weight=1.0)
    observed_in = dict(graph.in_degree())
    observed_out = dict(graph.out_degree())
    assert sorted(observed_out.values()) != sorted(observed_in.values())

    samples = list(configuration(graph, 5, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        assert sample.is_directed()
        assert dict(sample.in_degree()) == observed_in
        assert dict(sample.out_degree()) == observed_out
        assert sample.number_of_edges() == graph.number_of_edges()
    assert any(set(sample.edges) != set(graph.edges) for sample in samples)


def test_a_digraph_too_small_for_the_three_arc_swap_yields_nothing() -> None:
    """The directed swap rotates three arcs, so two are not enough -- and a sampler that yields
    nothing is how a report comes to say "0 rewirings" instead of "z = 0.00, no structure"."""
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    graph.add_nodes_from(["d", "e"])

    assert list(configuration(graph, 5, seed=SEED)) == []


def test_configuration_does_not_touch_the_observed_graph(karate: nx.Graph) -> None:
    before = sorted(karate.edges)

    list(configuration(karate, 3, seed=SEED))

    assert sorted(karate.edges) == before


def test_configuration_yields_nothing_for_a_graph_with_no_swap_to_make() -> None:
    """Refusing is the honest answer: three nodes admit no double edge swap at all."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "c", weight=1)

    assert list(configuration(graph, 10, seed=SEED)) == []


def test_the_same_seed_gives_the_same_samples(karate: nx.Graph) -> None:
    first = [sorted(s.edges) for s in configuration(karate, 3, seed=7)]
    second = [sorted(s.edges) for s in configuration(karate, 3, seed=7)]

    assert first == second


# ----------------------------------------------------------------------------- label shuffling


def test_a_label_permutation_keeps_the_graph_and_the_mix_of_values() -> None:
    graph = two_cliques()
    key = attr_key("region")

    samples = list(label_permutation(graph, 5, key=key, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        assert sorted(sample.edges) == sorted(graph.edges)
        values = [sample.nodes[n][key] for n in sample]
        assert sorted(values) == ["north"] * 8 + ["south"] * 8


def test_a_label_permutation_leaves_the_observed_labels_alone() -> None:
    graph = two_cliques()
    key = attr_key("region")

    list(label_permutation(graph, 20, key=key, seed=SEED))

    assert all(graph.nodes[n][key] == n.split("-")[0] for n in graph)


def test_a_label_permutation_leaves_untagged_nodes_untagged() -> None:
    """An untagged node is not a value, so it is outside the shuffle as well as the measure."""
    graph = two_cliques()
    graph.add_edge("untagged", "north-1", weight=1)
    key = attr_key("region")

    sample = next(label_permutation(graph, 1, key=key, seed=SEED))

    assert key not in sample.nodes["untagged"]


# ----------------------------------------------------------------------------- the two-mode null


def test_a_bipartite_sample_keeps_every_left_and_right_degree() -> None:
    """The promise the null is made of: every node keeps how many documents it was in, and every
    document keeps how many nodes were in it. Each sample carries the memberships it was built
    from, so the margins are read off the sample itself rather than inferred from the projection.
    """
    pairs = activity_corpus()
    left, right = side_degrees(pairs)

    samples = list(bipartite_preserving(pairs, 5, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        rewired = pairs_of(sample)
        assert rewired is not None
        assert len(rewired) == len(set(pairs))
        sample_left, sample_right = side_degrees(rewired)
        assert sample_left == left
        assert sample_right == right
        assert set(sample.nodes) == set(left)


def test_a_bipartite_sample_of_the_southern_women_keeps_both_published_margins() -> None:
    """The legendary two-mode network (tests/legendary.py, §53.4): 18 women, 14 events, 89
    attendances. Every woman keeps how many events she attended and every event keeps how many
    women attended it, which is what makes a re-projection a plausible alternative record of the
    same season rather than a different society."""
    graph, known = southern_women()
    women = set(known.groups["women"])
    pairs = [(str(a), str(b)) if a in women else (str(b), str(a)) for a, b in graph.edges]
    assert (len(pairs), len(women)) == (89, 18)
    left, right = side_degrees(pairs)

    samples = list(bipartite_preserving(pairs, 5, seed=SEED))

    assert len(samples) == 5
    for sample in samples:
        rewired = pairs_of(sample)
        assert rewired is not None
        sample_left, sample_right = side_degrees(rewired)
        assert sample_left == left  # each woman's number of events
        assert sample_right == right  # each event's number of women
        assert set(sample.nodes) == {str(w) for w in women}
    assert any(sorted(s.edges) != sorted(samples[0].edges) for s in samples[1:])


def test_a_bipartite_sample_rewires_the_memberships_rather_than_repeating_them() -> None:
    pairs = activity_corpus()

    rewired = [pairs_of(sample) for sample in bipartite_preserving(pairs, 5, seed=SEED)]

    assert all(set(sample or []) != set(pairs) for sample in rewired)


def test_a_bipartite_sample_is_a_different_projection() -> None:
    """It has to move: a null that returns the observation tests nothing."""
    pairs = activity_corpus()
    observed = bipartite_projection(pairs, side="left")

    samples = list(bipartite_preserving(pairs, 5, seed=SEED))

    assert len(samples) == 5
    assert any(sorted(s.edges) != sorted(observed.edges) for s in samples)
    for sample in samples:
        assert set(sample.nodes) == set(observed.nodes)


def test_a_bipartite_sample_of_a_rigid_corpus_is_the_same_shape_as_the_observation() -> None:
    """Two documents of eight, and every node in one of them: the projection can only be two
    cliques of eight, whoever is in which. That rigidity is the null's whole point -- it keeps
    the arithmetic a projection forces and randomises only what was observed."""
    pairs = [(f"n-{i}", "doc-a") for i in range(8)] + [(f"s-{i}", "doc-b") for i in range(8)]

    for sample in bipartite_preserving(pairs, 5, seed=SEED):
        assert sample.number_of_edges() == 2 * (8 * 7 // 2)
        assert sorted(nx.triangles(sample).values()) == [21] * 16


def test_projecting_on_the_right_gives_the_other_mode() -> None:
    pairs = clique_corpus()
    observed = bipartite_projection(pairs, side="right")

    sample = next(bipartite_preserving(pairs, 1, side="right", seed=SEED))

    assert set(sample.nodes) == set(observed.nodes) == {"doc-north", "doc-south", "doc-bridge"}


def test_a_projection_carries_the_memberships_it_came_from() -> None:
    pairs = clique_corpus()

    graph = bipartite_projection(pairs, side="left")

    assert pairs_of(graph) == sorted(set(pairs))
    assert projected_side(graph, pairs) == "left"
    assert projected_side(bipartite_projection(pairs, side="right"), pairs) == "right"


def test_the_entity_network_carries_what_its_passages_were_tagged_with(
    layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The table the re-derivation needs, on a network a builder built rather than a test.

    Each passage is keyed to the attributes of the document it belongs to -- the same table
    ``_label_attributes`` borrowed the labels from -- so a rewired corpus can be labelled the way
    the real one was. It is the projection's input, not part of it, so no file carries it: a
    network read back from disk has the edges and not the corpus.
    """
    graph = build_network(layered, "entities", "test-layers", min_weight=1)

    table = graph.graph[RIGHT_ATTRIBUTES]
    chunks = {chunk for _, chunk in pairs_of(graph) or []}
    assert chunks and set(table) == chunks
    assert any(values.get("region") for values in table.values())

    write_graph(graph, tmp_path / "entities.graphml")
    back = read_graph(tmp_path / "entities.graphml")

    assert RIGHT_ATTRIBUTES not in back.graph and pairs_of(back) is None
    # ...and the graph that was written still has both, because the write only borrowed them.
    assert RIGHT_ATTRIBUTES in graph.graph and pairs_of(graph) is not None


def test_a_graph_that_never_went_through_a_projection_has_no_memberships() -> None:
    assert pairs_of(two_cliques()) is None
    # A graph whose ids are in neither mode cannot be a projection of these memberships, and
    # guessing a side for it would rewire somebody else's corpus.
    assert projected_side(nx.karate_club_graph(), clique_corpus()) is None


# ----------------------------------------------------------------------------- §19.1 step four


def test_an_observation_drawn_from_its_own_null_is_unremarkable() -> None:
    """The measure has to be able to say 'ordinary', or it is not a test.

    One value is taken out of a sample of the null itself and scored against the rest: it sits at
    the middle by construction, so the z-score must be near zero and the empirical p near a half.
    """
    rng = random.Random(3)
    draws = [rng.gauss(0.4, 0.05) for _ in range(400)]
    observed = float(np.median(draws))

    result = significance(observed, draws, null="configuration")

    assert abs(result.z) < 0.2
    assert result.p_value == pytest.approx(0.5, abs=0.05)
    assert result.samples == 400
    assert result.null == "configuration"
    assert result.caveat == ""


def test_significance_flags_a_heavy_tailed_null_and_says_to_quote_the_p_value() -> None:
    """§19.1's z-score assumes the tidy histogram of Figure 19.2. This null is not one."""
    rng = random.Random(5)
    draws = [rng.paretovariate(1.2) for _ in range(300)]

    result = significance(max(draws) * 1.5, draws, null="bipartite_preserving")

    assert result.summary is not None and result.summary.heavy_tailed
    assert "empirical p-value" in result.caveat
    assert result.p_value == pytest.approx(1 / 301, rel=1e-9)  # cannot go below 1/(n+1)


def test_a_null_that_could_not_be_built_is_reported_rather_than_raised() -> None:
    result = significance(0.7, [], null="configuration")

    assert result.samples == 0
    assert not result.testable
    assert result.z == 0.0
    assert math.isnan(result.p_value)
    assert "nothing here was tested" in result.caveat


# --------------------------------------------------------------- the moved nulls, number for number


def test_the_community_null_reproduces_the_numbers_it_produced_before_the_move() -> None:
    """Pinned so that moving the rewiring loop into ``sna.null`` cannot move a published report.

    Three cliques of eight joined by one edge each, seed 1, 30 samples: the same planted graph
    ``test_sna_cluster`` uses. These figures were recorded from the pre-move implementation.
    """
    graph = nx.Graph()
    blocks = [[f"n{i}" for i in range(start, start + 8)] for start in (0, 8, 16)]
    for block in blocks:
        for i, a in enumerate(block):
            for b in block[i + 1 :]:
                graph.add_edge(a, b, weight=3)
    graph.add_edge("n7", "n8", weight=1)
    graph.add_edge("n15", "n16", weight=1)

    result = null_model_modularity(graph, louvain(graph, seed=1, runs=3).communities, 30, seed=1)

    assert result.samples == 30
    assert result.observed == pytest.approx(0.6587900675801353, rel=1e-12)
    assert result.null_mean == pytest.approx(0.2282059230785128, rel=1e-12)
    assert result.null_std == pytest.approx(0.02022022261524826, rel=1e-12)
    assert result.z_score == pytest.approx(21.294728188448076, rel=1e-12)


def test_the_attribute_nulls_reproduce_the_numbers_they_produced_before_the_move() -> None:
    """Same pinning for the label shuffle and the fixed-partition rewiring, on two cliques."""
    report = analyse_attribute(two_cliques(), "region", seed=SEED, permutations=50, samples=20)

    assert report.permutations == 50 and report.null_samples == 20
    assert report.permuted_mean == pytest.approx(-0.09557860167660533, rel=1e-12)
    assert report.permuted_std == pytest.approx(0.04822461630458124, rel=1e-12)
    assert report.assortativity_z == pytest.approx(21.990654641613297, rel=1e-12)
    assert report.null_mean == pytest.approx(-0.0385964912280702, rel=1e-12)
    assert report.null_std == pytest.approx(0.04871383641022744, rel=1e-12)
    assert report.modularity_z == pytest.approx(10.696193730074453, rel=1e-12)


# ----------------------------------------------------------- the two nulls, on a planted confound


def test_a_label_that_is_really_activity_survives_the_shuffle_and_not_the_two_mode_null() -> None:
    """The case the bipartite null exists for, planted so the answer is known in advance.

    Six busy nodes are in six documents each and fourteen quiet nodes are in one. Nothing else
    distinguishes the two values, so the projection's disassortativity -- busy nodes are joined to
    everybody, quiet nodes only to the busy ones they were recorded with -- is the corpus's shape
    and not a fact about the attribute. The label shuffle holds that shape fixed and calls it a
    finding; the bipartite null keeps each node's number of documents and reproduces it, which is
    the answer.
    """
    graph = labelled_projection(activity_corpus(), source=OWN)

    report = analyse_attribute(
        graph, "region", seed=SEED, permutations=200, samples=200, null="bipartite"
    )

    assert report.assortativity is not None and report.assortativity < -0.4
    assert abs(report.assortativity_z) > 4  # the shuffle: "strongly disassortative"
    assert report.bipartite is not None
    assert report.bipartite.samples == 200
    assert abs(report.bipartite.z) < 2  # the two-mode null: ordinary
    assert abs(report.bipartite.z) < abs(report.assortativity_z) / 2
    assert report.bipartite.null_mean == pytest.approx(report.assortativity, abs=0.15)
    # Nobody borrowed anything here: "busy" is a property of the node, so the null pins it and
    # only the corpus moves.
    assert not report.bipartite_rederived and report.inherited == 0


def test_a_borrowed_label_is_priced_into_the_null_that_derives_it_again() -> None:
    """The confound the whole provenance key exists for, with the null that can see it.

    Two cliques, one document each, and every label borrowed from that document -- the strongest
    result the measure can produce (assortativity 0.96) and a restatement of how the labels were
    assigned. The label shuffle scores it z = 8.5: shuffling breaks the copying, so the null sits
    near zero and the circularity reads as the finding of the corpus.

    The bipartite null rewires which node was recorded in which document and *derives the
    labels again from the rewired documents*, so every sample copies its labels the same way the
    observation did. The null mean therefore lands near the observation (about 0.70 against 0.96)
    and the z-score falls to about 2 with a two-sided p near 0.2: exactly what it should say
    about a number that a corpus of this shape produces whenever it is rewired.
    """
    graph = labelled_projection(clique_corpus(), documents=CLIQUE_DOCUMENTS)

    report = analyse_attribute(
        graph, "region", seed=SEED, permutations=200, samples=200, null="bipartite"
    )

    assert report.assortativity is not None and report.assortativity > 0.9
    assert report.assortativity_z > 8  # the shuffle: the largest z-score in the corpus
    assert report.bipartite is not None and report.bipartite_rederived
    assert report.bipartite.z < report.assortativity_z / 2  # the plan's "done when"
    assert report.bipartite.z < 3
    assert report.bipartite.p_value > 0.05
    assert report.bipartite.null_mean > 0.5  # the borrowing happens in the null too
    # The null prices the mechanism; it does not make the tag true, so the warning stays.
    assert report.confounded and "Not a reading of this attribute" in report.verdict


def test_the_same_labels_carried_across_instead_score_far_higher() -> None:
    """Why the re-derivation is the whole point, measured against its own absence.

    The same corpus and the same labels, with no record of what the documents were tagged with:
    the null can only carry the labels across, which breaks the copying exactly as a shuffle
    does, and the confound comes back as a z-score larger than the shuffle's. A network that
    cannot re-derive says so in the report rather than reporting this number as if it were the
    one above.
    """
    carried = analyse_attribute(
        labelled_projection(clique_corpus()),
        "region",
        seed=SEED,
        permutations=200,
        samples=200,
        null="bipartite",
    )
    derived = analyse_attribute(
        labelled_projection(clique_corpus(), documents=CLIQUE_DOCUMENTS),
        "region",
        seed=SEED,
        permutations=200,
        samples=200,
        null="bipartite",
    )

    assert carried.bipartite is not None and derived.bipartite is not None
    assert not carried.bipartite_rederived
    assert carried.bipartite.z > carried.assortativity_z  # worse than the shuffle, not better
    assert derived.bipartite.z < carried.bipartite.z / 4
    assert "could not be derived again" in "\n".join(render_attribute(carried))


def test_a_pruned_network_says_its_null_was_not_pruned_with_it() -> None:
    """``--min-weight`` cuts the projection after the fact, and the null re-projects from the
    memberships, which are from before it. The samples therefore have edges the observation does
    not, and a null built differently from the observation has to say so out loud."""
    graph = labelled_projection(activity_corpus(), source=OWN)
    graph.graph["min_weight"] = 3

    report = analyse_attribute(
        graph, "region", seed=SEED, permutations=20, samples=20, null="bipartite"
    )

    assert any("re-projections are not pruned" in note for note in report.notes)


def test_the_bipartite_null_is_refused_when_the_network_was_never_projected() -> None:
    with pytest.raises(ValueError, match="carries none"):
        analyse_attribute(two_cliques(), "region", seed=SEED, null="bipartite")


def test_an_unknown_null_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="must be one of permutation, bipartite"):
        analyse_attribute(two_cliques(), "region", seed=SEED, null="vibes")


def test_the_report_prints_the_frame_the_n_the_null_and_the_chapter() -> None:
    graph = labelled_projection(activity_corpus(), source=OWN)
    report = analyse_attribute(
        graph, "region", seed=SEED, permutations=50, samples=50, null="bipartite"
    )

    text = "\n".join(render_attribute(report))

    assert "**Sampling frame.** The 50 two-mode membership(s)" in text
    assert "**n.** 50 null network(s)" in text
    assert "**Null model.** `bipartite_preserving`" in text
    assert "**Implements.** §18.1" in text
    assert "curveball algorithm of Strona et al. (2014)" in text


# ----------------------------------------------------------------------------- §19.2 ERGM


def test_an_ergm_on_a_random_graph_finds_no_triangle_effect() -> None:
    """G(n,p) closes triangles only as often as independent coin tosses do, which is the null the
    triangle coefficient is read against: it must come back indistinguishable from zero."""
    graph = nx.gnp_random_graph(60, 0.12, seed=4)

    fit = ergm(graph)

    triangles = fit.term("triangles")
    assert abs(triangles.z) < 2.0
    assert triangles.p_value > 0.05
    assert "not distinguishable from chance" in triangles.reading
    assert fit.coefficient("edges") < 0  # §19.2: a sparse network has a negative edge term
    assert fit.dyads == 60 * 59 // 2
    assert fit.converged


def test_an_ergm_on_a_clustered_graph_finds_a_positive_triangle_effect(karate: nx.Graph) -> None:
    """Zachary's karate club closes a quarter of its triads (transitivity 0.256), which is the
    textbook social-network pattern §19.2 fits the Florentine families for: a strongly negative
    edge term because the network is sparse, and a positive triangle term because a triad is more
    likely than chance to have its third edge.

    Deliberately not a graph built out of cliques. There, an edge exists exactly when the two
    nodes already share four neighbours, the logistic model separates perfectly, and the
    coefficient runs off to infinity with a standard error to match -- which is degeneracy
    arriving in the fit rather than in the simulation, and is the second caveat made visible.
    """
    fit = ergm(karate)

    triangles = fit.term("triangles")
    assert triangles.coefficient > 0
    assert triangles.z > 3
    assert triangles.p_value < 0.01
    assert "more likely than chance" in triangles.reading
    assert fit.coefficient("edges") < -2  # sparse, as §19.2 reads -4.27 in Figure 19.5


def test_an_ergm_on_a_separable_graph_says_it_cannot_tell_rather_than_pretending() -> None:
    """Six disjoint cliques of six: an edge is present exactly when the endpoints share four
    neighbours, so the fit is perfectly separable. The coefficient is huge and meaningless, and
    the standard error has to be huge with it, which is what stops it being read as a finding."""
    graph = nx.Graph()
    for block in ([f"c{b}-{i}" for i in range(6)] for b in range(6)):
        for i, a in enumerate(block):
            for other in block[i + 1 :]:
                graph.add_edge(a, other)

    fit = ergm(graph)

    triangles = fit.term("triangles")
    assert triangles.coefficient > 1  # separation drives it away from zero
    assert triangles.std_error > 1  # against 0.13 on the karate club, where the fit is real
    assert abs(triangles.z) < 2
    assert "not distinguishable from chance" in triangles.reading


def test_an_ergm_finds_the_homophily_that_was_planted_and_refuses_it_without_an_attribute() -> None:
    graph = two_cliques()

    fit = ergm(graph, terms=("edges", "triangles", "homophily"), attribute=attr_key("region"))

    assert fit.coefficient("homophily") > 0
    assert fit.attribute == attr_key("region")
    with pytest.raises(ValueError, match="needs attribute"):
        ergm(graph, terms=("edges", "homophily"))


def test_an_ergm_refuses_what_it_cannot_fit() -> None:
    with pytest.raises(ValueError, match="at least three nodes"):
        ergm(nx.complete_graph(2))
    with pytest.raises(ValueError, match="all present or all absent"):
        ergm(nx.complete_graph(5))
    with pytest.raises(ValueError, match="unknown ERGM term"):
        ergm(nx.karate_club_graph(), terms=("edges", "quadrangles"))


def test_an_ergm_on_a_digraph_uses_ordered_dyads_and_finds_planted_reciprocity() -> None:
    """§19.2's directed model, ``p|E'| + p₁R(A')``, on a network where the answer is planted.

    Twenty nodes in a sparse random digraph, and then nine arcs in ten are answered. Every
    ordered pair is a dyad -- 380 of them for 20 nodes, not 190 -- because ``u -> v`` and
    ``v -> u`` are two variables, and reading only the upper triangle would drop the second half
    of every reciprocated pair and then report the count as if it had been one dyad.
    """
    base = nx.gnp_random_graph(20, 0.1, seed=3, directed=True)
    graph = nx.DiGraph()
    graph.add_nodes_from(base)
    rng = random.Random(3)
    for source, target in base.edges:
        graph.add_edge(source, target)
        if rng.random() < 0.9:
            graph.add_edge(target, source)

    fit = ergm(graph)

    assert fit.directed
    assert fit.dyads == 20 * 19  # ordered pairs, not 190
    assert fit.edges == graph.number_of_edges()
    reciprocity = fit.term("reciprocity")
    assert reciprocity.coefficient > 0
    assert reciprocity.z > 3
    assert "more likely than chance" in reciprocity.reading
    assert fit.coefficient("edges") < 0  # still sparse: most ordered pairs are not arcs


def test_an_ergm_on_a_random_digraph_finds_no_reciprocity_effect() -> None:
    """The same measure has to be able to say no: in a G(n,p) digraph each arc is an independent
    coin toss, so an arc coming back is exactly as likely as any other arc."""
    fit = ergm(nx.gnp_random_graph(30, 0.12, seed=4, directed=True))

    reciprocity = fit.term("reciprocity")
    assert abs(reciprocity.z) < 2
    assert reciprocity.p_value > 0.05
    assert "not distinguishable from chance" in reciprocity.reading


def test_the_reciprocity_term_is_refused_on_an_undirected_network(karate: nx.Graph) -> None:
    """In an undirected network R(A') is the edge count again, so the two terms would be one
    column and the fit would be reporting the same number twice under two names."""
    with pytest.raises(ValueError, match="needs a directed network"):
        ergm(karate, terms=("edges", "reciprocity"))


def test_the_three_star_term_completes_figure_19_5(karate: nx.Graph) -> None:
    """Figure 19.5 fits four configurations -- an edge, a chain of three nodes, a star of four,
    a triangle -- so all four have to be available, and on the karate club the star term is the
    figure's own reading: a small positive effect beside a much larger triangle one."""
    fit = ergm(karate, terms=("edges", "two_stars", "three_stars", "triangles"))

    assert [term.name for term in fit.terms] == [
        "edges",
        "two_stars",
        "three_stars",
        "triangles",
    ]
    assert fit.coefficient("three_stars") > 0
    assert fit.coefficient("triangles") > fit.coefficient("three_stars")
    assert set(ERGM_TERMS) == {
        "edges",
        "reciprocity",
        "two_stars",
        "three_stars",
        "triangles",
        "homophily",
    }


def test_the_fit_prints_the_networks_own_frame_or_admits_it_has_none(karate: nx.Graph) -> None:
    """A sampling frame is a claim about where the data came from, so it is read off the graph or
    not made at all. The karate club carries its published one; a graph built in a test carries
    nothing, and saying "a corpus projection" about it would be a lie in the one line a reader
    trusts."""
    club, _ = karate_club()

    with_frame = "\n".join(render_ergm(ergm(karate), club))
    without = "\n".join(render_ergm(ergm(karate)))

    assert "34 members of one university karate club" in with_frame
    assert "No sampling frame" in without
    assert "corpus projection" not in with_frame + without


def test_the_fit_prints_the_persona_frame_of_a_network_this_package_built(
    karate: nx.Graph,
    layered: InMemoryGraphStore,
) -> None:
    """The frame line has to come from the keys ``export._meta`` really writes, not from a
    sentence in the renderer, so the provenance under test is a builder's own.

    It is carried onto the karate club because the planted corpus is three entities in one clique
    -- every dyad present, which no logistic model can fit. What is being checked is where the
    sentence comes from, and that is the same wherever the edges came from.
    """
    built = build_network(layered, "entities", "test-layers", min_weight=1)
    graph = karate.copy()
    graph.graph.update(built.graph)

    text = "\n".join(render_ergm(ergm(graph), graph))

    assert "Entities mentioned in the same passage" in text
    assert "persona `test-layers`" in text
    assert "never a population" in text


def test_every_fit_carries_the_book_s_warnings_and_prints_them() -> None:
    """The warnings are part of the result, not documentation: a fit printed without them invites
    exactly the three readings §19.2 says are wrong."""
    fit = ergm(nx.karate_club_graph())

    text = "\n".join(render_ergm(fit))

    assert "MPLE, not MLE" in text
    assert "degeneracy" in text
    assert "P(x|theta) maximised" in text
    assert "**Null model.** None" in text
    assert "**Implements.** §19.2" in text
    assert f"n = {34 * 33 // 2:,} dyads" in text
