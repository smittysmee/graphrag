"""Moving a network in and out of the formats other tools read, and the legendary graphs.

Two halves, both from Atlas ch. 53. The first writes a network built by the real builders to
every format ``FORMATS`` lists and reads it back, asserting that what survives is exactly what
that format claims to carry -- a GraphML keeps the sampling frame, a Pajek file keeps the names
and the weights and nothing else. The loss is asserted as deliberately as the survival, because
a node attribute that vanishes without anyone noticing is how a partition ends up reported
against labels that are no longer there.

The second half holds the legendary graphs (§53.4) to their published numbers, which is what
makes ``tests/legendary.py`` usable by every later ticket: if Louvain here stops finding 0.42 on
the karate club, the fixture has stopped being a known answer and so has every test built on it.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.cluster import louvain
from graphrag.sna.export import (
    FORMATS,
    GraphFormat,
    build_network,
    graph_format,
    read_graph,
    to_graph_tool,
    to_igraph,
    write_graph,
)
from tests.legendary import (
    LEGENDARY,
    florentine_families,
    karate_club,
    les_miserables,
    southern_women,
    unweighted,
)


@pytest.fixture
def network(layered: InMemoryGraphStore) -> nx.Graph:
    """A real builder network: two modes, weights, node attributes with their provenance, a
    frame in ``graph.graph``. Everything a format could lose is on it."""
    return build_network(layered, "speakers-entities", "test-layers")


# --------------------------------------------------------------------------- the formats


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_every_format_round_trips_exactly_what_it_claims_to_carry(
    fmt: GraphFormat, network: nx.Graph, tmp_path: Path
) -> None:
    """Write, read, and check the structure, the attributes and the provenance one by one."""
    written = write_graph(network, tmp_path / f"net{fmt.suffixes[0]}")
    assert written.exists()
    assert graph_format(written).name == fmt.name

    back = read_graph(written)
    assert set(back.nodes) == set(network.nodes)
    assert {frozenset(edge) for edge in back.edges} == {frozenset(edge) for edge in network.edges}
    for source, target, data in network.edges(data=True):
        assert fmt.edge_weights
        assert back[source][target]["weight"] == data["weight"]

    for node, data in network.nodes(data=True):
        if fmt.node_attributes:
            for key, value in data.items():
                got = back.nodes[node][key]
                if fmt.typed:
                    assert got == value and type(got) is type(value), (node, key)
                else:
                    # An untyped format stores text: the value comes back as what the text reads
                    # as, which for a CSV turns the string "2" into the number 2.
                    assert got == value or str(got) == str(value), (node, key)
        else:
            # The documented loss: Pajek and the edge list carry names and weights alone.
            assert not dict(back.nodes[node]), node

    if fmt.provenance:
        for key, value in network.graph.items():
            assert back.graph[key] == value, key
    else:
        assert "frame" not in back.graph
        assert "persona_id" not in back.graph


def test_a_graphml_keeps_the_attribute_provenance_the_report_reads(
    network: nx.Graph, tmp_path: Path
) -> None:
    """``attrsrc_`` decides whether an assortativity means anything, so it has to survive."""
    back = read_graph(write_graph(network, tmp_path / "net.graphml"))
    assert back.nodes["ana"]["attrsrc_region"] == "node"
    assert back.nodes["product:alpha"]["attrsrc_region"] == "document"
    assert back.nodes["product:alpha"]["attrn_region"] == 2
    assert back.graph["persona_id"] == "test-layers"
    assert back.graph["frame"].startswith("Speakers and the entities")


def test_gexf_hands_gephi_the_attributes_and_keeps_none_of_the_provenance(
    network: nx.Graph, tmp_path: Path
) -> None:
    """What a Gephi round trip adds and what it costs, both asserted rather than assumed."""
    back = read_graph(write_graph(network, tmp_path / "net.gexf"))
    assert back.nodes["product:beta"]["attr_region"] == "north"
    assert back["ana"]["product:beta"]["weight"] == 2.0  # weights come back as floats
    assert "id" in back["ana"]["product:beta"]  # GEXF gives every edge an id of its own
    assert "frame" not in back.graph and "network" not in back.graph

    plain = nx.Graph()
    plain.add_edge("ana", "bo", weight=1)
    labelled = read_graph(write_graph(plain, tmp_path / "plain.gexf"))
    assert labelled.nodes["ana"]["label"] == "ana"  # a node with no label is given its own id


def test_pajek_keeps_the_names_and_weights_and_nothing_else(
    network: nx.Graph, tmp_path: Path
) -> None:
    """Including the drawing coordinates Pajek writes, which are not data and are not read."""
    written = write_graph(network, tmp_path / "net.net")
    assert "ellipse" in written.read_text(encoding="utf-8")  # Pajek's own shape column
    back = read_graph(written)
    assert back["ana"]["product:beta"]["weight"] == 2.0
    assert dict(back.nodes["product:beta"]) == {}
    assert "x" not in back.nodes["ana"] and "shape" not in back.nodes["ana"]


def test_an_edge_list_drops_isolated_nodes_and_the_richer_formats_do_not(
    layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """Nobody in this corpus shares a document, so the speaker network is four isolates.

    That is a real finding about a corpus -- nobody was recorded with anybody -- and an edge
    list cannot express it: the file is empty and the network comes back as nothing at all.
    """
    speakers = build_network(layered, "speakers", "test-layers")
    assert speakers.number_of_nodes() == 4
    assert speakers.number_of_edges() == 0

    as_edges = write_graph(speakers, tmp_path / "speakers.edgelist")
    assert as_edges.read_text(encoding="utf-8") == ""
    assert read_graph(as_edges).number_of_nodes() == 0

    for suffix in (".graphml", ".json", ".gexf", ".net", ".csv"):
        back = read_graph(write_graph(speakers, tmp_path / f"speakers{suffix}"))
        assert set(back.nodes) == {"ana", "bo", "cy", "dot"}, suffix


def test_a_missing_attribute_stays_missing_through_the_csv_pair(
    layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """``dot`` was tagged by nobody, and an empty cell must not become the string ''."""
    speakers = build_network(layered, "speakers", "test-layers")
    back = read_graph(write_graph(speakers, tmp_path / "speakers.csv"))
    assert back.nodes["ana"]["attr_region"] == "north"
    assert "attr_region" not in back.nodes["dot"]
    assert back.nodes["ana"]["documents"] == 2  # a number written as text comes back a number


def test_the_csv_pair_reads_a_number_shaped_string_back_as_a_number(
    network: nx.Graph, tmp_path: Path
) -> None:
    """The documented retyping, on a value the builders really produce.

    ``partner_weights`` is a ``;``-joined string, so a node with one partner holds the string
    ``"2"`` -- and a CSV cell cannot tell that from the number. Read the pair back for a drawing
    or a table, and read the GraphML back for an analysis.
    """
    assert network.nodes["product:beta"]["partner_weights"] == "2"
    from_csv = read_graph(write_graph(network, tmp_path / "net.csv"))
    assert from_csv.nodes["product:beta"]["partner_weights"] == 2
    from_graphml = read_graph(write_graph(network, tmp_path / "net.graphml"))
    assert from_graphml.nodes["product:beta"]["partner_weights"] == "2"


def test_the_csv_pair_is_a_node_table_and_an_edge_table_cytoscape_can_import(
    network: nx.Graph, tmp_path: Path
) -> None:
    nodes_path = write_graph(network, tmp_path / "net.csv")
    edges_path = tmp_path / "net.edges.csv"
    assert nodes_path == tmp_path / "net.nodes.csv"
    assert edges_path.exists()

    with nodes_path.open(encoding="utf-8", newline="") as handle:
        node_rows = list(csv.reader(handle))
    with edges_path.open(encoding="utf-8", newline="") as handle:
        edge_rows = list(csv.reader(handle))
    assert node_rows[0][0] == "id"
    assert "attr_region" in node_rows[0]
    assert len(node_rows) == network.number_of_nodes() + 1
    # ``p``, the edge's probability of existing (Atlas §28.2), rides in the edge table beside
    # the weight: a format that carries edge attributes carries the evidence behind the edge.
    assert edge_rows[0] == ["source", "target", "p", "weight"]
    assert len(edge_rows) == network.number_of_edges() + 1
    assert all(0.0 < float(row[2]) <= 1.0 for row in edge_rows[1:])

    # Either name of the pair, or the bare stem, resolves to the same two files.
    for given in (nodes_path, edges_path, tmp_path / "net.csv"):
        assert set(read_graph(given).nodes) == set(network.nodes)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_direction_survives_only_where_the_file_can_say_so(
    fmt: GraphFormat, tmp_path: Path
) -> None:
    """A directed network in an edge list or a CSV pair comes back undirected unless told."""
    directed = nx.DiGraph(network="relations", frame="who names whom")
    directed.add_edge("ana", "bo", weight=2)
    directed.add_edge("cy", "bo", weight=1)

    written = write_graph(directed, tmp_path / f"net{fmt.suffixes[0]}")
    back = read_graph(written)
    if fmt.direction:
        assert back.is_directed()
        assert back.has_edge("ana", "bo") and not back.has_edge("bo", "ana")
    else:
        assert not back.is_directed()
        told = read_graph(written, directed=True)
        assert told.is_directed()
        assert told.has_edge("ana", "bo") and not told.has_edge("bo", "ana")


def test_an_unknown_suffix_is_refused_in_both_directions(network: nx.Graph, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        write_graph(network, tmp_path / "net.dot")
    with pytest.raises(ValueError, match="unsupported export format"):
        read_graph(tmp_path / "net.dot")
    with pytest.raises(ValueError, match="unsupported export format"):
        graph_format(Path("net.gml"))


def test_an_edge_list_refuses_a_node_name_containing_its_delimiter(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edge("ana\tnorth", "bo", weight=1)
    with pytest.raises(ValueError, match="contain a tab"):
        write_graph(graph, tmp_path / "net.edgelist")
    assert read_graph(write_graph(graph, tmp_path / "net.graphml")).has_edge("ana\tnorth", "bo")


def test_an_edge_list_refuses_a_node_name_it_would_read_back_as_a_comment(
    tmp_path: Path,
) -> None:
    """``#product`` is a plausible entity name, and a `#` line is a comment on the way back.

    Refused rather than written, because the loss would be the whole node and every edge it
    holds -- the file would still parse, and the network would simply be smaller.
    """
    graph = nx.Graph()
    graph.add_edge("#growth", "bo", weight=1)
    with pytest.raises(ValueError, match="start with '#'"):
        write_graph(graph, tmp_path / "net.edgelist")
    assert read_graph(write_graph(graph, tmp_path / "net.graphml")).has_edge("#growth", "bo")


def test_pajek_refuses_a_node_name_its_quoting_would_strip(tmp_path: Path) -> None:
    """Pajek quotes names, so a name holding a quote comes back as a different name."""
    graph = nx.Graph()
    graph.add_edge('say "hi", ok', "bo", weight=1)
    with pytest.raises(ValueError, match="contain a double quote"):
        write_graph(graph, tmp_path / "net.net")
    assert read_graph(write_graph(graph, tmp_path / "net.graphml")).has_edge('say "hi", ok', "bo")


def test_only_node_link_json_gives_the_node_ids_back_as_they_were(tmp_path: Path) -> None:
    """Every other format stores an id as text, which is how an int-keyed graph changes keys.

    The legendary fixtures are all int-keyed or str-keyed by networkx's choice, not ours, so a
    ticket that exports one and reads it back gets ``"0"`` where it wrote ``0``. Nothing
    coerces text back to a number: ``"0042"`` and ``42`` are different nodes.
    """
    graph = nx.Graph(network="planted")
    graph.add_edge(0, 1, weight=3)

    as_json = read_graph(write_graph(graph, tmp_path / "net.json"))
    assert set(as_json.nodes) == {0, 1}
    assert as_json[0][1]["weight"] == 3

    for suffix in (".graphml", ".gexf", ".net", ".csv", ".edgelist"):
        back = read_graph(write_graph(graph, tmp_path / f"net{suffix}"))
        assert set(back.nodes) == {"0", "1"}, suffix
        assert 0 not in back, suffix


@pytest.mark.parametrize(
    ("bridge", "module", "message"),
    [
        (to_igraph, "igraph", "pip install igraph"),
        (to_graph_tool, "graph_tool", "graph-tool.skewed.de"),
    ],
    ids=["igraph", "graph-tool"],
)
def test_an_absent_bridge_library_says_how_to_install_itself(
    bridge: object, module: str, message: str, network: nx.Graph
) -> None:
    """Neither library is a dependency, so the failure has to be the instructions (§53.1)."""
    if importlib.util.find_spec(module) is not None:  # pragma: no cover - not in this image
        pytest.skip(f"{module} is installed, so there is no ImportError to read")
    with pytest.raises(ImportError, match=message):
        bridge(network)  # type: ignore[operator]


# --------------------------------------------------------------------------- §53.4 known answers


@pytest.mark.parametrize("name", sorted(LEGENDARY), ids=sorted(LEGENDARY))
def test_every_legendary_graph_has_the_size_its_source_published(name: str) -> None:
    graph, known = LEGENDARY[name]()
    assert graph.number_of_nodes() == known.nodes
    assert graph.number_of_edges() == known.edges
    assert known.source and known.frame and known.caveats
    assert graph.graph["frame"] == known.frame
    weighted = all("weight" in data for _, _, data in graph.edges(data=True))
    assert weighted == known.weighted


def test_the_karate_club_carries_the_fission_as_ground_truth() -> None:
    """Zachary recorded which club each member joined, independently of the sparring edges."""
    graph, known = karate_club()
    assert graph.number_of_edges() == 78  # the Newman-Girvan copy, not the 77-edge one (§53.4)
    hi, officer = known.groups["Mr. Hi"], known.groups["Officer"]
    assert len(hi) == len(officer) == 17
    assert set(hi) | set(officer) == set(graph.nodes)
    assert 0 in hi and 33 in officer  # the instructor and the president, on opposite sides
    assert graph.degree(33) == 17 and graph.degree(0) == 16


def test_louvain_recovers_the_published_karate_modularity_and_splits_the_leaders() -> None:
    """The known answer: 0.4198 over four communities on the unweighted 78-edge graph.

    Louvain is randomised, so this is the multi-seed run the package always uses; the check is
    that the best of ten runs lands within 0.01 of the published maximum, and that the two men
    the club split between never end up in the same community.
    """
    graph, known = karate_club()
    assert known.modularity is not None and known.modularity_communities is not None

    result = louvain(unweighted(graph), seed=11, runs=10)
    assert abs(result.modularity - known.modularity) < 0.01, result.modularity
    assert len(result.communities) == known.modularity_communities
    assert result.stability > 0.6

    community_of = {
        node: index for index, community in enumerate(result.communities) for node in community
    }
    assert community_of[0] != community_of[33]
    # And the caveat that goes with the number: four communities are not the two factions.
    assert len(result.communities) > len(known.groups)


def test_the_weights_are_why_the_published_modularity_needs_its_conditions() -> None:
    """Modularity on Zachary's weighted graph is a different number, and higher."""
    graph, known = karate_club()
    assert known.modularity is not None
    weighted = louvain(graph, seed=11, runs=10)
    assert weighted.modularity > known.modularity + 0.01
    assert "unweighted" in known.modularity_note


def test_the_first_girvan_newman_split_is_the_fission_bar_two_members() -> None:
    """Girvan and Newman's split of this network is the classic recovered-communities result.

    Their paper reports a single misclassified member. On the copy networkx ships -- weighted,
    78 edges -- the first split of the edge-betweenness algorithm puts two of Mr. Hi's members
    with the officers: node 2 and node 8, the member Zachary himself explains away as staying
    for his black-belt test. Two, not one: the number is a property of this copy of the graph
    and of this implementation, which is exactly why it is pinned here.
    """
    graph, known = karate_club()
    split = [sorted(part) for part in next(nx.community.girvan_newman(unweighted(graph)))]
    assert len(split) == 2
    assert sorted(len(part) for part in split) == [15, 19]

    side = {node: index for index, part in enumerate(split) for node in part}
    hi_side = side[0]
    misplaced = {node for node in known.groups["Mr. Hi"] if side[node] != hi_side}
    misplaced |= {node for node in known.groups["Officer"] if side[node] == hi_side}
    assert misplaced == {2, 8}
    assert any("three weeks from a black" in caveat for caveat in known.caveats)


def test_the_medici_hold_the_highest_betweenness_in_florence() -> None:
    """The point the network is always used to make, and it is a wide margin, not a nose."""
    graph, known = florentine_families()
    betweenness = nx.betweenness_centrality(graph)
    ranked = sorted(betweenness, key=lambda family: -betweenness[family])
    assert ranked[0] == known.top_betweenness == "Medici"
    assert betweenness["Medici"] > 2 * betweenness[ranked[1]]
    assert "Pucci" not in graph  # the sixteenth family, an isolate, is not in this copy


def test_valjean_is_the_hub_of_les_miserables() -> None:
    graph, known = les_miserables()
    degrees = dict(graph.degree())
    assert max(degrees, key=lambda name: degrees[name]) == known.top_degree == "Valjean"
    assert graph["Myriel"]["MlleBaptistine"]["weight"] == 8  # chapters shared, not a count of 1


def test_the_southern_women_are_eighteen_women_by_fourteen_events() -> None:
    """The two modes are certain; which women form which group is not, and is not asserted."""
    graph, known = southern_women()
    women, events = known.groups["women"], known.groups["events"]
    assert len(women) == 18
    assert len(events) == 14
    assert graph.number_of_edges() == 89
    assert nx.is_bipartite(graph)
    assert all(not graph.has_edge(a, b) for a in women for b in women if a != b)
    assert "Evelyn Jefferson" in women and "E1" in events
    assert any("disputed" in caveat for caveat in known.caveats)


def test_a_legendary_graph_survives_graphml_with_its_ground_truth(tmp_path: Path) -> None:
    """Whatever a later ticket exports, the ``club`` labels have to come back with it.

    The one thing that does not survive is the type of the node ids: GraphML is read back with
    string ids, so the karate club's node 0 comes back as ``"0"``. A test that mixes an exported
    fixture with a fresh one has to expect that.
    """
    graph, _ = karate_club()
    back = read_graph(write_graph(graph, tmp_path / "karate.graphml"))
    assert back.number_of_nodes() == 34
    assert back.number_of_edges() == 78
    assert back.nodes["0"]["club"] == "Mr. Hi"
    assert back.nodes["33"]["club"] == "Officer"
    assert back["0"]["1"]["weight"] == 4
    assert 0 not in back
