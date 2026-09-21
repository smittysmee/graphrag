"""Chapter 30 over graphs whose answers are known before the code runs.

Three kinds of fixture, for three kinds of claim. The **karate club** (`tests/legendary.py`)
carries a ground truth its author collected independently of the edges -- which faction each
member joined -- so the homophily indices can be checked against a split that really happened;
node 0's ego network is small enough to count by hand. The **planted 90/10 graph** is wired at
random, so every honest measure of homophily has to return "nothing here" on it, and the ones
that do not are the ones §30.2 warns about. The **two cliques and a bridge** have exactly one
weak tie in Granovetter's sense, and its overlap has to be 0 while the edges inside the cliques
are at 1.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from graphrag.sna.attributes import (
    CONTAGION_CAVEAT,
    analyse_attribute,
    attribute_labels,
    coleman_index,
    ei_index,
    ei_ratio,
    homophily_indices,
    render_attribute,
)
from graphrag.sna.ego import ego_payload, ego_report, render_ego
from graphrag.sna.export import INHERITED, attr_key, attr_source_key
from graphrag.sna.measures import clustering
from graphrag.sna.ties import (
    COLOURABLE,
    TWO_MODE,
    neighbourhood_overlap,
    render_ties,
    tie_payload,
    tie_strength_curve,
    two_mode_kind,
)
from tests.legendary import karate_club

SEED = 7

# Zachary's node 0 is Mr. Hi, the instructor: 16 alters, 18 edges among them, and a local
# clustering coefficient of exactly 0.15 (Atlas §12.2's definition over the 78-edge copy).
KARATE_EGO_ALTERS = 16
KARATE_EGO_CLUSTERING = 0.15


def factioned() -> nx.Graph:
    """The karate club with its published faction on every node as an SNA attribute.

    ``club`` is Zachary's own record of which side each member joined after the split, collected
    outside the network, which is what makes it a test of homophily rather than a demonstration
    of it. It is copied onto ``attr_faction`` because that is the key this package's attribute
    layer reads, and marked as the node's own value, because it is.
    """
    graph, _ = karate_club()
    for _, data in graph.nodes(data=True):
        data[attr_key("faction")] = data["club"]
    return graph


def planted_majority(size: int = 100, minority: int = 10) -> nx.Graph:
    """A random graph whose labels mean nothing: 90 nodes of one value, 10 of another.

    Nothing about who is joined to whom depends on the label, so the truth here is known and it
    is "no homophily". Any measure that reports some is measuring the sizes of the two groups.
    """
    graph = nx.gnp_random_graph(size, 0.2, seed=SEED)
    for node in graph:
        graph.nodes[node][attr_key("side")] = "majority" if node < size - minority else "minority"
    return graph


def southern_style_two_mode(women: int = 3, events: int = 4) -> nx.Graph:
    """A complete two-mode network with its modes declared, as the builders declare them.

    Every edge joins a woman to an event, so no two adjacent nodes are of the same kind: every
    overlap is 0 and every local clustering is 0 before anything about the data is known. It is
    the shape of `--network speakers-entities` without `--project`, in miniature.
    """
    graph = nx.complete_bipartite_graph(women, events)
    graph = nx.relabel_nodes(
        graph,
        {n: (f"woman{n}" if n < women else f"event{n - women}") for n in graph},
    )
    for node in graph:
        graph.nodes[node]["mode"] = "speaker" if node.startswith("woman") else "entity"
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    return graph


def two_cliques_and_a_bridge(size: int = 5, weight: float = 5.0) -> nx.Graph:
    """Two cliques joined by a single light edge: one weak tie, and everything else strong."""
    graph = nx.Graph()
    for side in ("a", "b"):
        nodes = [f"{side}{i}" for i in range(size)]
        for left in nodes:
            for right in nodes:
                if left < right:
                    graph.add_edge(left, right, weight=weight)
    graph.add_edge("a0", "b0", weight=1.0)
    return graph


# ----------------------------------------------------------------------------- §30.1 ego networks


def test_the_ego_network_without_the_ego_has_the_egos_clustering_as_its_density() -> None:
    """§30.1's identity, on the graph the chapter's readers all know.

    The two numbers are computed by different routes -- one is the density of a subgraph, the
    other ``networkx``'s clustering coefficient over the whole network -- so agreeing is
    evidence that the construction is right and not a tautology.
    """
    graph = factioned()
    report = ego_report(graph, 0)

    assert report.alters == KARATE_EGO_ALTERS
    assert report.neighbours == KARATE_EGO_ALTERS
    assert report.edges_with_ego == 34  # 16 to the ego, 18 among the alters
    assert report.edges_without_ego == 18
    assert report.density_with_ego == pytest.approx(0.25)
    assert report.density_without_ego == pytest.approx(KARATE_EGO_CLUSTERING)
    assert report.local_clustering == pytest.approx(KARATE_EGO_CLUSTERING)
    assert clustering(graph, "local")[0] == pytest.approx(report.density_without_ego)
    assert report.identity_holds


def test_the_brokerage_count_is_the_pairs_of_neighbours_no_edge_joins() -> None:
    """16 alters make 120 pairs; 18 of them are joined, so 102 run only through the instructor."""
    report = ego_report(factioned(), 0)

    assert report.brokerage_pairs == 120 - 18
    assert report.brokerage == pytest.approx(1 - KARATE_EGO_CLUSTERING)
    # §30.1's point: with the ego in place the neighbourhood is connected by construction, and
    # removing it shows what was really holding it together.
    assert report.components_without_ego == 4
    assert report.isolated_alters == 2
    assert "open neighbourhood" in report.reading


def test_the_report_says_which_properties_the_construction_forced() -> None:
    report = ego_report(factioned(), 0)
    section = "\n".join(render_ego(report))

    # §30.1's forced properties are named as forced, in the notes, rather than reported as
    # facts about this node.
    assert any("one component and a diameter of 2" in note for note in report.notes)
    assert "**Implements.** §30.1" in section
    assert "**Null model.** None." in section
    assert "**Sampling frame.**" in section and "**n.**" in section
    assert "local clustering coefficient" in section


def test_a_wider_radius_stops_claiming_the_identity_and_says_why() -> None:
    report = ego_report(factioned(), 0, radius=2)

    assert report.alters > KARATE_EGO_ALTERS
    assert report.neighbours == KARATE_EGO_ALTERS
    assert not report.identity_holds
    assert any("no longer the ego's local clustering" in note for note in report.notes)


def test_which_view_the_report_leads_with_is_a_choice_and_both_are_computed() -> None:
    graph = factioned()
    with_ego = ego_report(graph, 0)
    without = ego_report(graph, 0, with_ego=False)

    assert with_ego.density == pytest.approx(with_ego.density_with_ego)
    assert without.density == pytest.approx(without.density_without_ego)
    assert with_ego.density_without_ego == pytest.approx(without.density_without_ego)
    assert "Reported without the ego." in "\n".join(render_ego(without))


def test_a_node_with_no_neighbour_has_no_ego_network_and_says_so() -> None:
    graph = nx.Graph()
    graph.add_node("alone")
    graph.add_edge("a", "b")
    report = ego_report(graph, "alone")

    assert report.alters == 0
    assert report.density_without_ego == 0.0
    assert "no ego network" in report.reading


def test_an_ego_that_is_not_in_the_network_or_a_radius_below_one_is_refused() -> None:
    graph = factioned()
    with pytest.raises(KeyError):
        ego_report(graph, "nobody")
    with pytest.raises(ValueError, match="radius must be at least 1"):
        ego_report(graph, 0, radius=0)


def test_the_alters_composition_counts_the_attribute_and_names_the_illusion() -> None:
    """§30.2 over one neighbourhood, and §30.4's majority illusion seen from one node.

    15 of Mr. Hi's 16 alters are in his faction and one is not, so his own EI index is
    (1 - 15) / 16 = -0.875: a neighbourhood almost entirely of his own kind.
    """
    graph = factioned()
    report = ego_report(graph, 0, by="faction")
    composition = report.composition
    assert composition is not None

    assert composition.ego_value == "Mr. Hi"
    assert composition.counts == {"Mr. Hi": 15, "Officer": 1}
    assert composition.labelled_alters == KARATE_EGO_ALTERS
    assert composition.ei == pytest.approx(ei_ratio(15, 1))
    assert composition.ei == pytest.approx(-0.875)
    # The two factions are 17 and 17, so neither is a minority and there is no illusion to name.
    assert composition.illusion == ()
    assert composition.network_counts == {"Mr. Hi": 17, "Officer": 17}


def test_a_minority_that_surrounds_one_node_is_named_as_the_majority_illusion() -> None:
    """§30.4: a value can be rare in the corpus and everywhere in one node's neighbourhood."""
    graph = nx.star_graph(4)  # node 0 at the centre, 1-4 around it
    graph.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 5)])
    for node in graph:
        graph.nodes[node][attr_key("side")] = "rare" if 1 <= node <= 4 else "common"
    report = ego_report(graph, 0, by="side")
    composition = report.composition
    assert composition is not None

    assert composition.counts == {"rare": 4}
    assert composition.network_shares["rare"] == pytest.approx(0.4)
    assert composition.illusion == ("rare",)
    assert "Majority illusion (§30.4)" in "\n".join(render_ego(report))


def test_borrowed_labels_among_the_alters_are_counted_and_flagged() -> None:
    graph = two_cliques_and_a_bridge()
    for node in graph:
        graph.nodes[node][attr_key("region")] = node[0]
        graph.nodes[node][attr_source_key("region")] = INHERITED
    report = ego_report(graph, "a0", by="region")
    composition = report.composition
    assert composition is not None

    assert composition.borrowed_alters == composition.alters
    assert "carry no value of their own" in "\n".join(render_ego(report))


def test_the_ego_payload_is_json_and_carries_the_numbers_the_section_printed() -> None:
    report = ego_report(factioned(), 0, by="faction")
    payload = json.loads(json.dumps(ego_payload(report)))

    assert payload["alters"] == KARATE_EGO_ALTERS
    assert payload["density_without_ego"] == pytest.approx(KARATE_EGO_CLUSTERING)
    assert payload["local_clustering"] == pytest.approx(KARATE_EGO_CLUSTERING)
    assert payload["identity_holds"] is True
    assert payload["brokerage_pairs"] == 102
    assert payload["composition"]["counts"] == {"Mr. Hi": 15, "Officer": 1}


# ----------------------------------------------------------------------------- §30.2 EI, Coleman


def test_the_karate_split_is_strongly_homophilous_by_ei_and_by_coleman() -> None:
    """The factions really did divide the club, so every index has to say so.

    67 of the 78 edges join two members of the same faction and 11 cross, so
    EI = (11 - 67) / 78 = -0.718. Negative is homophily, which is the opposite sign from the
    assortativity the same split scores (+0.72).
    """
    graph = factioned()
    labels = attribute_labels(graph, "faction")

    assert ei_index(graph, labels) == pytest.approx((11 - 67) / 78)
    assert ei_index(graph, labels) < -0.7

    coleman = coleman_index(graph, labels)
    assert coleman["Mr. Hi"] == pytest.approx(0.736, abs=0.01)
    assert coleman["Officer"] == pytest.approx(0.715, abs=0.01)
    assert min(coleman.values()) > 0.7

    indices = homophily_indices(graph, "faction", permutations=100, seed=SEED)
    assert indices.edges == 78
    assert indices.internal == 67 and indices.external == 11
    # The shuffle keeps the two group sizes and destroys the split, so it lands near 0 and the
    # observed index is many null standard deviations below it.
    assert indices.permuted_mean == pytest.approx(0.0, abs=0.15)
    assert indices.ei_z < -5
    assert indices.size_driven == ()
    assert "most edges run inside values" in indices.reading


def test_a_majority_looks_homophilous_by_size_and_coleman_refuses_to_agree() -> None:
    """§30.2's warning, as a number: 'some values are more popular than others' (p. 434).

    Nothing in this graph depends on the labels. The majority holds 90 of the 100 nodes, so 89
    of every 99 partners available to one of its members is one of its own and its EI index is
    about -0.62 before anybody chooses anything. Coleman's index divides that share out and
    returns ~0, which is the truth.
    """
    graph = planted_majority()
    labels = attribute_labels(graph, "side")

    majority = ei_index(graph, labels, value="majority")
    assert majority is not None and majority < -0.5
    minority = ei_index(graph, labels, value="minority")
    assert minority is not None and minority > 0.8

    coleman = coleman_index(graph, labels)
    assert coleman["majority"] == pytest.approx(0.0, abs=0.05)
    assert coleman["minority"] == pytest.approx(0.0, abs=0.25)

    indices = homophily_indices(graph, "side", permutations=100, seed=SEED)
    assert indices.ei is not None and indices.ei < -0.5
    # The shuffle carries the same group sizes, so it reproduces the same EI: the index is about
    # the sizes, and the z-score is what is about the attribute.
    assert indices.permuted_mean == pytest.approx(indices.ei, abs=0.05)
    assert abs(indices.ei_z) < 2
    assert "majority" in indices.size_driven
    assert "a majority looks homophilous by size alone" in indices.reading


def test_the_ei_index_is_undefined_rather_than_zero_when_nothing_was_counted() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    assert ei_index(graph, {"a": "x", "b": "y"}) is None
    assert ei_ratio(0, 0) is None
    assert ei_ratio(3, 0) == -1.0
    assert ei_ratio(0, 3) == 1.0
    # A value held by one node of two has no same-value partner available, so its expected
    # in-group share is 0 and tying outside scores 0, not -1: Coleman measures the gap from what
    # the size implies, and here the size implies nothing.
    graph.add_edge("a", "b")
    assert coleman_index(graph, {"a": "x", "b": "y"}) == pytest.approx({"x": 0.0, "y": 0.0})
    assert set(coleman_index(graph, {"a": "x", "b": "y"}, value="x")) == {"x"}
    # A node nobody joined is in no tie, so it is absent from the table rather than scored.
    graph.add_node("lonely")
    graph.nodes["lonely"]["ignored"] = True
    assert "z" not in coleman_index(graph, {"a": "x", "b": "y", "lonely": "z"})


def test_the_by_section_carries_the_indices_and_the_contagion_caveat() -> None:
    """§30.4 in the report: this number cannot separate homophily from influence."""
    report = analyse_attribute(factioned(), "faction", permutations=40, samples=10, seed=SEED)
    section = "\n".join(render_attribute(report))

    assert report.homophily is not None
    assert report.homophily.ei == pytest.approx((11 - 67) / 78)
    assert "### Homophily by counting edges (EI and Coleman)" in section
    assert "**Implements.** §30.2" in section
    assert "**Null model.** `label_permutation`" in section
    assert "negative is homophily" in section
    assert CONTAGION_CAVEAT in section
    assert "*before* the tie formed" in section
    assert "32 years" in section


def test_the_payload_carries_the_indices_for_a_json_reader() -> None:
    report = analyse_attribute(factioned(), "faction", permutations=40, samples=10, seed=SEED)
    from graphrag.sna.attributes import attribute_payload

    payload = json.loads(json.dumps(attribute_payload(report)))
    homophily = payload["homophily"]

    assert homophily["internal"] == 67 and homophily["external"] == 11
    assert homophily["ei"] == pytest.approx((11 - 67) / 78)
    assert {row["value"] for row in homophily["values"]} == {"Mr. Hi", "Officer"}
    assert homophily["contagion_caveat"] == CONTAGION_CAVEAT


# ----------------------------------------------------------------------------- §30.3 weak ties


def test_the_bridge_has_no_overlap_and_the_clique_edges_have_all_of_it() -> None:
    """§30.3's weak tie is structural: 'social circles do not overlap much' (p. 436)."""
    graph = two_cliques_and_a_bridge()
    overlap = neighbourhood_overlap(graph)

    assert overlap[("a0", "b0")] == 0.0
    assert overlap[("a1", "a2")] == pytest.approx(1.0)
    # a0 has the bridge as well as its clique, so an edge from it is not quite buried.
    assert overlap[("a0", "a1")] == pytest.approx(0.75)
    assert len(overlap) == graph.number_of_edges()


def test_an_isolated_pair_has_an_undefined_overlap_rather_than_a_zero_one() -> None:
    graph = nx.Graph()
    graph.add_edge("u", "v")
    assert neighbourhood_overlap(graph) == {}

    report = tie_strength_curve(graph)
    assert report.measured == 0 and report.undefined == 1
    assert "No edge in this network has a defined neighbourhood overlap" in report.reading
    assert any("An isolated pair is not a bridge" in note for note in report.notes)


def test_overlap_rising_with_weight_is_reported_as_granovetters_structure() -> None:
    """The planted case: heavy edges inside the cliques, light edges between them."""
    graph = nx.Graph()
    for side in ("a", "b", "c"):
        nodes = [f"{side}{i}" for i in range(8)]
        for left in nodes:
            for right in nodes:
                if left < right:
                    graph.add_edge(left, right, weight=5.0)
    for left, right in (("a0", "b0"), ("b1", "c1"), ("c2", "a2"), ("a3", "c3")):
        graph.add_edge(left, right, weight=1.0)

    report = tie_strength_curve(graph)
    assert report.weighted
    assert report.bridges == 4
    assert report.correlation is not None
    # Positive and significant is the planted answer; the coefficient cannot approach 1 because
    # 84 of the 88 edges share one weight and Spearman averages their tied ranks (§3.4).
    assert report.correlation.coefficient > 0.3
    assert report.correlation.p_value < 0.01
    assert "Overlap rises with weight" in report.reading
    assert "§30.3" in "\n".join(render_ties(report))
    assert {(u, v) for u, v, _, _ in report.weakest} == {
        ("a0", "b0"),
        ("b1", "c1"),
        ("a2", "c2"),
        ("a3", "c3"),
    }


def test_a_network_whose_weights_do_not_vary_has_no_curve_and_says_so() -> None:
    graph = two_cliques_and_a_bridge(weight=1.0)
    report = tie_strength_curve(graph)

    assert not report.weighted
    assert report.bins == ()
    assert report.correlation is None
    assert report.bridges == 1
    assert "no strength to correlate" in report.reading


def test_the_curve_bins_by_weight_quantile_and_prints_what_it_could_make() -> None:
    graph = nx.Graph()
    for index in range(20):
        graph.add_edge(f"hub{index % 4}", f"leaf{index}", weight=float(index + 1))
        graph.add_edge(f"hub{index % 4}", f"hub{(index + 1) % 4}", weight=float(index + 1))
    report = tie_strength_curve(graph, bins=4)

    assert 1 <= len(report.bins) <= 4
    assert sum(row.edges for row in report.bins) == report.measured
    payload = json.loads(json.dumps(tie_payload(report)))
    assert payload["measured"] == report.measured
    assert len(payload["bins"]) == len(report.bins)
    assert payload["reading"] == report.reading


def test_every_edge_of_a_two_mode_network_is_a_bridge_by_construction() -> None:
    """§6.4 decides this before §30.3 is asked, so the reading is withheld rather than printed.

    `complete_bipartite_graph(3, 4)` has 12 edges and not one triangle: the overlap is 0 on every
    edge because a woman's neighbours are events and an event's are women, so the intersection is
    empty for reasons that have nothing to do with tie strength.
    """
    graph = southern_style_two_mode()
    assert two_mode_kind(graph) == TWO_MODE

    report = tie_strength_curve(graph)
    assert report.edges == 12
    assert report.measured == 12
    assert report.mean_overlap == 0.0
    assert report.bridges == report.measured and report.bridge_share == 1.0
    # The numbers are right and the reading refuses them.
    assert "two-mode network" in report.reading
    assert "§30.3's reading is withheld" in report.reading
    assert "--project" in report.reading
    assert any("Read it after `--project`" in note for note in report.notes)
    section = "\n".join(render_ties(report))
    assert "ch. 26" in section
    # The "heaviest bridges" table would be a list of every edge, so it is not printed.
    assert "yet sharing no neighbour" not in section
    assert tie_payload(report)["two_mode"] == TWO_MODE


def test_a_two_mode_ego_network_is_not_a_broker_however_the_numbers_look() -> None:
    """The same artefact on §30.1: clustering 0 and brokerage 1.0 for every node alike.

    woman0 has all 4 events as alters, none of which can be joined to another, so all
    C(4, 2) = 6 pairs count as brokered and the ego network falls into 4 pieces once the ego
    goes. Every node of every two-mode network scores exactly that.
    """
    graph = southern_style_two_mode()
    report = ego_report(graph, "woman0")

    assert report.alters == 4
    assert report.edges_without_ego == 0
    assert report.density_without_ego == 0.0
    assert report.local_clustering == 0.0
    assert report.brokerage_pairs == 6
    assert report.brokerage == 1.0
    assert report.components_without_ego == 4
    assert report.two_mode == TWO_MODE
    assert "open neighbourhood" not in report.reading
    assert "two-mode network" in report.reading
    assert "§30.1's brokerage" in report.reading
    assert "for every node of this network alike" in report.reading
    assert any("--project speakers" in note for note in report.notes)
    assert ego_payload(report)["two_mode"] == TWO_MODE


def test_an_undeclared_but_triangle_free_network_says_so_differently() -> None:
    """No modes were declared, so there is nothing to project -- and no triangles either."""
    graph = nx.complete_bipartite_graph(2, 3)
    assert two_mode_kind(graph) == COLOURABLE

    report = tie_strength_curve(graph)
    assert report.bridges == report.measured
    assert "2-colourable" in report.reading
    assert "--project" not in report.reading
    assert ego_report(graph, 0).two_mode == COLOURABLE

    # A network with a triangle is neither, and the §30.3 reading runs as before.
    assert two_mode_kind(two_cliques_and_a_bridge()) == ""
    assert tie_strength_curve(two_cliques_and_a_bridge()).two_mode == ""


def test_a_positive_ei_is_named_as_heterophily() -> None:
    """§30.4's other half: 'the love of the different', the chapter's disassortative case.

    Two values and edges only between them -- the dating network of Figure 30.8 in miniature --
    so every edge is external and the EI index is +1.
    """
    graph = southern_style_two_mode()
    for node in graph:
        graph.nodes[node][attr_key("kind")] = "woman" if node.startswith("woman") else "event"
    indices = homophily_indices(graph, "kind", permutations=50, seed=SEED)

    assert indices.ei == 1.0
    assert "heterophily" in indices.reading
    assert "disassortativ" in indices.reading
    assert indices.ei_z > 2


def test_a_directed_network_is_flattened_before_the_overlap_is_taken() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("a", "c", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    report = tie_strength_curve(graph)

    assert report.flattened
    assert neighbourhood_overlap(graph)[("a", "b")] == pytest.approx(1.0)
    assert "either direction" in "\n".join(render_ties(report))
