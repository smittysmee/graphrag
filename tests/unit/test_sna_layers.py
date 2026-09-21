"""Chapter 7 over the planted corpus: layers, hyperedges and time.

Every number asserted here is read off the ``layered`` fixture's own table (``tests/conftest.py``)
or off a published legendary graph (``tests/legendary.py``), never off a previous run of this
code. The fixture's five posts are:

===== ======= ========== ================================ =========
post  speaker posted     entities (stance)                facets
===== ======= ========== ================================ =========
1     ana     2025-01-10 Alpha (praise), Beta (praise)     quoting
2     bo      2025-01-20 Alpha (complaint), Gamma (compl.) outage
3     ana     2025-06-10 Beta (praise), Gamma (neutral)     quoting
4     cy      2025-06-20 Alpha (complaint), Gamma (compl.) outage
5     dot     undated    -                                  -
===== ======= ========== ================================ =========
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.compare import compare_windows
from graphrag.sna.export import bipartite_projection, describe, entity_co_mention
from graphrag.sna.layers import (
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
from graphrag.sna.matrices import adjacency
from tests.legendary import southern_women

PERSONA = "test-layers"
ALPHA, BETA, GAMMA = "product:alpha", "product:beta", "product:gamma"


def weights(graph: nx.Graph) -> dict[tuple[str, str], float]:
    """Every edge as ``{(u, v): weight}``, endpoints sorted so two builds compare directly."""
    return {
        (u, v) if graph.is_directed() else tuple(sorted((u, v))): w or 1  # type: ignore[misc]
        for u, v, w in graph.edges(data="weight")
    }


# ----------------------------------------------------------------------------- §7.2 multilayer


def test_stance_layers_hold_exactly_the_mentions_the_fixture_annotated(
    layered: InMemoryGraphStore,
) -> None:
    """One layer per stance, each the network of that stance alone (Atlas §7.2).

    Praise is annotated on Alpha and Beta in post 1 and on Beta alone in post 3, so the praise
    layer holds one edge. Complaint is annotated on Alpha and Gamma in posts 2 and 4, so the
    complaint layer holds one edge of weight 2. Neutral is annotated on Gamma alone in post 3,
    so the neutral layer holds a node and no edge at all -- which is the case worth having a
    row for.
    """
    ml = multilayer(layered, PERSONA, "entities", "stance", min_weight=1)
    assert ml.names == ("praise", "complaint", "neutral")
    assert ml.nodes == (ALPHA, BETA, GAMMA)
    assert weights(ml.graphs["praise"]) == {(ALPHA, BETA): 1}
    assert weights(ml.graphs["complaint"]) == {(ALPHA, GAMMA): 2}
    assert weights(ml.graphs["neutral"]) == {}

    rows = {row.name: row for row in layer_table(ml)}
    assert (rows["praise"].nodes, rows["praise"].edges, rows["praise"].isolated) == (2, 1, 1)
    assert (rows["complaint"].nodes, rows["complaint"].weight) == (2, 2.0)
    assert (rows["neutral"].edges, rows["neutral"].isolated) == (0, 3)
    assert "omega=1.0" in ml.frame and "§7.2" in ml.frame


def test_flattening_the_facet_layers_is_the_unfiltered_network_edge_for_edge(
    layered: InMemoryGraphStore,
) -> None:
    """A facet layering splits passages, so its flattening is the whole network (Atlas §7.2).

    Quoting holds posts 1 and 3, outage posts 2 and 4, and between them every passage that names
    two entities. Summing the two layers therefore has to reproduce the unfiltered entity
    network exactly -- same edges, same weights -- and the test is the check the reading rule
    tells a reader to make.
    """
    ml = multilayer(layered, PERSONA, "entities", "facet", min_weight=1)
    assert ml.names == ("outage", "quoting")
    flat = flatten(ml)
    unfiltered = entity_co_mention(layered, PERSONA, min_weight=1)
    assert weights(flat) == weights(unfiltered)
    assert set(flat.nodes) == set(unfiltered.nodes)
    assert flat[ALPHA][GAMMA]["layers"] == "outage"
    assert flat[BETA][GAMMA]["layers"] == "quoting"


def test_flattening_the_stance_layers_is_not_the_unfiltered_network(
    layered: InMemoryGraphStore,
) -> None:
    """A stance layering splits mentions inside a passage, so it loses a cross-stance pair.

    Post 3 praises Beta and is neutral about Gamma. Neither layer holds both, so the Beta-Gamma
    edge that the unfiltered network carries is in no layer at all. That is the rule "the layers
    need not sum to the whole", asserted rather than merely written down.
    """
    flat = flatten(multilayer(layered, PERSONA, "entities", "stance", min_weight=1))
    unfiltered = entity_co_mention(layered, PERSONA, min_weight=1)
    assert (BETA, GAMMA) in weights(unfiltered)
    assert (BETA, GAMMA) not in weights(flat)
    assert weights(flat) == {(ALPHA, BETA): 1, (ALPHA, GAMMA): 2}


def test_a_flattened_graph_describes_itself_as_multilayer(layered: InMemoryGraphStore) -> None:
    """§6.4's type line has to say what chapter 7 made, or a reader will read it as a simple one."""
    flat = flatten(multilayer(layered, PERSONA, "entities", "facet", min_weight=1))
    kind = describe(flat)
    assert kind.multilayer and not kind.dynamic
    assert kind.text == "undirected, weighted, multilayer"
    assert flat.graph["layers"] == "outage,quoting"
    # The filter the layering owns held one value in the layer it was copied from.
    assert flat.graph["facets"] == ""


def test_the_supra_adjacency_is_the_layers_on_the_diagonal_and_omega_off_it(
    layered: InMemoryGraphStore,
) -> None:
    """§8.1's supra-adjacency: intralayer blocks, interlayer couplings, at a stated omega."""
    ml = multilayer(layered, PERSONA, "entities", "stance", min_weight=1, omega=0.5)
    count, layers = len(ml.nodes), len(ml.names)
    matrix, order = supra_adjacency(ml)

    assert matrix.shape == (count * layers, count * layers) == (9, 9)
    assert order[:3] == [(ALPHA, "praise"), (BETA, "praise"), (GAMMA, "praise")]
    assert order[3] == (ALPHA, "complaint")

    for index, name in enumerate(ml.names):
        padded = ml.graphs[name].copy()
        padded.add_nodes_from(ml.nodes)
        block, _ = adjacency(padded, nodes=list(ml.nodes))
        start = index * count
        np.testing.assert_allclose(matrix[start : start + count, start : start + count], block)

    coupling = np.eye(count) * 0.5
    for row in range(layers):
        for column in range(layers):
            if row == column:
                continue
            np.testing.assert_allclose(
                matrix[row * count : (row + 1) * count, column * count : (column + 1) * count],
                coupling,
            )
    # omega is a parameter: passing another one moves every off-diagonal block and nothing else.
    zero, _ = supra_adjacency(ml, omega=0.0)
    np.testing.assert_allclose(zero[:count, count : 2 * count], np.zeros((count, count)))
    np.testing.assert_allclose(zero[:count, :count], matrix[:count, :count])


def test_the_relations_network_layers_by_type_and_stays_directed(
    related: InMemoryGraphStore,
) -> None:
    """Chapter 6 folded two types between one pair into one edge; §7.2 is where they separate.

    The fixture states Gamma->Beta as ``replaces`` in post 3 and as ``competes_with`` in post 4.
    The directed network carries one edge of weight 2; layered by relation type it is one edge
    of weight 1 in each of two layers, which is the multigraph reading chapter 6 refused to fake.
    """
    ml = multilayer(related, PERSONA, "relations", "relation-type")
    assert ml.names == ("competes_with", "integrates_with", "replaces")
    assert ml.directed
    assert weights(ml.graphs["replaces"]) == {(GAMMA, BETA): 1}
    assert weights(ml.graphs["competes_with"]) == {(ALPHA, GAMMA): 2, (GAMMA, BETA): 1}
    flat = flatten(ml)
    assert flat.is_directed()
    assert flat[GAMMA][BETA]["weight"] == 2
    assert flat[GAMMA][BETA]["layers"] == "competes_with; replaces"


def test_the_two_mode_network_layers_by_stance_and_stays_two_mode(
    layered: InMemoryGraphStore,
) -> None:
    """A layering must not cost the two-mode network the thing that makes it two-mode (§6.4, §7.2).

    Layered by stance, who-wrote-about-what splits into who praised what, who complained about
    what and who was neutral about what: ana praises Alpha once and Beta in both her posts, bo
    and cy each complain about Alpha and Gamma, and ana is the only neutral voice, about Gamma.
    Six nodes over three layers give an 18-by-18 supra-adjacency.

    The flattening then has to keep ``mode`` on every node, because that is what
    :func:`graphrag.sna.export.describe` reads to call the network bipartite and what
    ``--project`` needs in order to collapse it onto one side. A flattening that loses it would
    silently turn a two-mode network into a one-mode one.
    """
    ml = multilayer(layered, PERSONA, "speakers-entities", "stance")
    assert ml.names == ("praise", "complaint", "neutral")
    assert ml.nodes == ("ana", "bo", "cy", ALPHA, BETA, GAMMA)
    assert weights(ml.graphs["praise"]) == {("ana", ALPHA): 1, ("ana", BETA): 2}
    assert weights(ml.graphs["complaint"]) == {
        ("bo", ALPHA): 1,
        ("bo", GAMMA): 1,
        ("cy", ALPHA): 1,
        ("cy", GAMMA): 1,
    }
    assert weights(ml.graphs["neutral"]) == {("ana", GAMMA): 1}

    matrix, order = supra_adjacency(ml)
    assert matrix.shape == (18, 18) == (len(ml.nodes) * 3, len(ml.nodes) * 3)
    assert order[0] == ("ana", "praise") and order[6] == ("ana", "complaint")

    flat = flatten(ml)
    modes = {node: data["mode"] for node, data in flat.nodes(data=True)}
    assert modes == {
        "ana": "speaker",
        "bo": "speaker",
        "cy": "speaker",
        ALPHA: "entity",
        BETA: "entity",
        GAMMA: "entity",
    }
    kind = describe(flat)
    assert kind.bipartite and kind.multilayer
    assert kind.text == "undirected, weighted, bipartite, multilayer"
    # Every edge still crosses between the modes, which is what --project collapses.
    assert all(modes[u] != modes[v] for u, v in flat.edges)
    assert flat.nodes[ALPHA]["label"] == "Alpha"


def test_a_layering_nothing_carries_is_refused_rather_than_returned_empty(
    layered: InMemoryGraphStore,
) -> None:
    with pytest.raises(ValueError, match="no layers to build"):
        multilayer(layered, PERSONA, "entities", "facet", facets=["pricing-model"])
    with pytest.raises(ValueError, match="relation-type applies to --network relations"):
        multilayer(layered, PERSONA, "entities", "relation-type")
    with pytest.raises(ValueError, match="stance reads the annotation"):
        multilayer(layered, PERSONA, "speakers", "stance")


def test_the_layer_table_prints_its_frame_its_n_and_the_chapter(
    layered: InMemoryGraphStore,
) -> None:
    ml = multilayer(layered, PERSONA, "entities", "stance", min_weight=1)
    text = "\n".join(render_layers(ml))
    assert "Atlas §7.2" in text and "§8.1" in text
    assert "**Sampling frame.**" in text
    assert "3 node(s) across 3 layer(s)" in text
    assert "9 by 9" in text
    assert "**Null model.** None here" in text
    assert "| praise | 2 | 1 | 1 | 1 |" in text


# ----------------------------------------------------------------------------- §7.3 hypergraph


def test_the_clique_expansion_of_the_passages_is_the_entity_network(
    layered: InMemoryGraphStore,
) -> None:
    """§7.3's first strategy, and the identity behind it.

    Turning a hyperedge into the edges it stands for is the same arithmetic as projecting a
    two-mode network, so the expansion of the passage hypergraph must equal the entity co-mention
    network weight for weight. Each of the four annotated posts names exactly two entities here,
    so every hyperedge has size 2 and the expansion loses nothing -- which is the only case in
    which it does not.
    """
    hg = hypergraph(layered, PERSONA)
    assert len(hg.hyperedges) == 4
    assert hg.sizes == (2, 2, 2, 2)
    assert hg.nodes == (ALPHA, BETA, GAMMA)
    assert hg.incidence.shape == (3, 4)

    expanded = clique_expansion(hg)
    assert weights(expanded) == weights(entity_co_mention(layered, PERSONA, min_weight=1))
    assert expanded.nodes[ALPHA]["hyperedges"] == 3  # posts 1, 2 and 4 name it
    assert "§7.3" in hg.frame and "k(k-1)/2" in hg.frame


def test_the_clique_expansion_of_the_southern_women_events_is_their_projection() -> None:
    """The same identity on the published two-mode graph (Atlas §53.4, §7.3).

    The 14 events are the hyperedges and the 18 women their members: 89 attendances in all, which
    is what Davis et al. recorded and what ``tests/legendary.py`` carries as the edge count. The
    expansion has to equal the package's own projection of the same memberships, and here the
    hyperedges are large -- the biggest event has 14 attendees, so it alone contributes 91 of the
    expanded edges -- which is the case the size distribution exists to warn about.
    """
    graph, known = southern_women()
    women = set(known.groups["women"])
    memberships = [
        (u, v) if u in women else (v, u)
        for u, v in graph.edges  # (member, hyperedge)
    ]
    hg = build_hypergraph(memberships, member_noun="women", edge_noun="event")

    assert len(hg.hyperedges) == 14
    assert len(hg.nodes) == 18
    assert sum(hg.sizes) == known.edges == 89
    # The attendance count of each event, straight off the 1941 table: four events of three
    # women, up to the one of fourteen that every account of these data mentions.
    assert sorted(hg.sizes) == [3, 3, 3, 3, 4, 4, 5, 6, 6, 8, 8, 10, 12, 14]

    expanded = clique_expansion(hg)
    assert weights(expanded) == weights(bipartite_projection(memberships, side="left"))
    # Evelyn and Theresa were each recorded at eight events, seven of them the same ones, which
    # is why every one of Freeman's 21 analyses puts them in one group. Olivia and Flora attended
    # two events and only those two, together, which is why every analysis puts them at the edge.
    assert expanded["Evelyn Jefferson"]["Theresa Anderson"]["weight"] == 7
    assert expanded["Olivia Carleton"]["Flora Price"]["weight"] == 2
    assert expanded.degree("Olivia Carleton") == expanded.degree("Flora Price")

    sizes = hyperedge_sizes(hg)
    assert sizes.count == 14
    assert (sizes.maximum, sizes.minimum) == (14, 3)


def test_a_hypergraph_with_no_hyperedges_has_no_size_distribution() -> None:
    with pytest.raises(ValueError, match="no hyperedges"):
        hyperedge_sizes(build_hypergraph([]))


# ----------------------------------------------------------------------------- §7.4 dynamic


def test_the_timestamped_edge_list_dates_every_pair_and_counts_the_undated(
    layered: InMemoryGraphStore,
) -> None:
    """§7.4's edge-level form: one row per activation, with the date the corpus put on it."""
    assert document_dates(layered, PERSONA) == {
        f"{PERSONA}:posts:post-1": "2025-01-10",
        f"{PERSONA}:posts:post-2": "2025-01-20",
        f"{PERSONA}:posts:post-3": "2025-06-10",
        f"{PERSONA}:posts:post-4": "2025-06-20",
    }

    history = dynamic_edges(layered, PERSONA, "entities")
    assert [(e.source, e.target, e.at) for e in history.edges] == [
        (ALPHA, BETA, "2025-01-10"),
        (ALPHA, GAMMA, "2025-01-20"),
        (BETA, GAMMA, "2025-06-10"),
        (ALPHA, GAMMA, "2025-06-20"),
    ]
    assert history.span == ("2025-01-10", "2025-06-20")
    # Post 5 names no entity, so it contributes nothing to this network at all.
    assert history.undated_documents == 0

    # On the speaker network it does contribute -- and is dropped, and counted.
    speakers = dynamic_edges(layered, PERSONA, "speakers")
    assert speakers.undated_documents == 1
    assert all(edge.doc_id != f"{PERSONA}:posts:post-5" for edge in speakers.edges)
    assert "in no window at all" in speakers.frame


def test_the_window_grid_covers_the_books_four_strategies() -> None:
    """§7.4: single snapshot, disjoint, sliding and cumulative windows over one calendar."""
    assert windows("2025-01-01", "2025-06-30", "3M") == [
        ("2025-01-01", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
    ]
    # Sliding: a step narrower than the width, so consecutive windows overlap.
    assert windows("2025-01-01", "2025-04-30", "2M", "1M") == [
        ("2025-01-01", "2025-02-28"),
        ("2025-02-01", "2025-03-31"),
        ("2025-03-01", "2025-04-30"),
        ("2025-04-01", "2025-04-30"),
    ]
    # Cumulative: every window starts where the corpus does.
    assert windows("2025-01-01", "2025-03-31", "1M", cumulative=True) == [
        ("2025-01-01", "2025-01-31"),
        ("2025-01-01", "2025-02-28"),
        ("2025-01-01", "2025-03-31"),
    ]
    # A single snapshot is a one-day window; a bare number is days.
    assert windows("2025-01-01", "2025-01-03", 1) == [
        ("2025-01-01", "2025-01-01"),
        ("2025-01-02", "2025-01-02"),
        ("2025-01-03", "2025-01-03"),
    ]
    with pytest.raises(ValueError, match="runs backwards"):
        windows("2025-06-30", "2025-01-01", "1M")
    with pytest.raises(ValueError, match="must be a count of days"):
        windows("2025-01-01", "2025-06-30", "a fortnight")
    with pytest.raises(ValueError, match="more than 500 windows"):
        windows("1900-01-01", "2025-06-30", "1D")


def test_snapshots_are_the_windowed_networks_the_comparison_has_always_built(
    layered: InMemoryGraphStore,
) -> None:
    """A snapshot is ``--since``/``--until`` on a grid, and nothing else (Atlas §7.4).

    The two halves of the fixture's year hold posts 1-2 and posts 3-4, which is what
    ``compare_windows`` builds when it is handed the same dates by hand. Asserting the two graphs
    edge for edge is what lets the comparison delegate to this and stay byte-identical.
    """
    snaps = snapshots(
        layered,
        PERSONA,
        "entities",
        window="3M",
        since="2025-01-01",
        until="2025-06-30",
        min_weight=1,
    )
    assert [label for label, _ in snaps] == ["2025-01-01 to 2025-03-31", "2025-04-01 to 2025-06-30"]
    for label, graph in snaps:
        opens, closes = label.split(" to ")
        expected = window_graph(
            layered, PERSONA, "entities", since=opens, until=closes, min_weight=1
        )
        assert weights(graph) == weights(expected)
        assert set(graph.nodes) == set(expected.nodes)

    first, second = (graph for _, graph in snaps)
    assert weights(first) == {(ALPHA, BETA): 1, (ALPHA, GAMMA): 1}
    assert weights(second) == {(BETA, GAMMA): 1, (ALPHA, GAMMA): 1}

    # And they are the two graphs the comparison builds, which is what lets it delegate here.
    comparison = compare_windows(
        layered,
        PERSONA,
        "entities",
        since="2025-01-01",
        until="2025-03-31",
        since2="2025-04-01",
        until2="2025-06-30",
        min_weight=1,
        seed=1,
        runs=2,
    )
    assert weights(comparison.a.graph) == weights(first)
    assert weights(comparison.b.graph) == weights(second)
    assert set(comparison.a.graph.nodes) == set(first.nodes)
    # The comparison's builds are not stamped dynamic: a pair of windows is not a sequence, and
    # the network-type line above its numbers has to keep saying what the network is.
    assert not describe(comparison.a.graph).dynamic

    kind = describe(first)
    assert kind.dynamic and kind.text == "undirected, weighted, dynamic"
    assert first.graph["dynamic"] == "2025-01-01 to 2025-03-31"
    assert first.graph["window"] == "3 months"
    assert first.graph["step"] == "3 months"
    # The undated post is in neither window, and the frame says how many there are.
    assert first.graph["undated"] == 1
    assert "disjoint windows" in first.graph["frame"]
    assert "1 document(s) carry no dated passage" in first.graph["frame"]


def test_a_step_wider_than_the_window_is_named_gapped_rather_than_sliding(
    layered: InMemoryGraphStore,
) -> None:
    """None of §7.4's four strategies throws time away; this one does, so it says so.

    A one-month window stepping every two months covers January, March and May and never looks
    at February, April or June. The frame has to name that, because the totals across the
    sequence do not add up to the corpus and nothing else in the report would reveal it.
    """
    snaps = snapshots(
        layered,
        PERSONA,
        "speakers",
        window="1M",
        step="2M",
        since="2025-01-01",
        until="2025-06-30",
    )
    assert [label for label, _ in snaps] == [
        "2025-01-01 to 2025-01-31",
        "2025-03-01 to 2025-03-31",
        "2025-05-01 to 2025-05-31",
    ]
    frame = snaps[0][1].graph["frame"]
    assert "gapped windows, which is none of §7.4's four" in frame
    assert "the time between windows is discarded" in frame
    assert "the roughly 30 days between one window and the next is discarded" in frame
    assert "the totals across it do not add up to the corpus" in frame
    # Posts 3 and 4 are dated in June, which no window covers: they are simply not here.
    assert set(snaps[2][1].nodes) == set()

    # A narrower step is still sliding, and says nothing about a gap.
    sliding = snapshots(
        layered, PERSONA, "speakers", window="2M", step="1M", since="2025-01-01", until="2025-04-30"
    )
    assert "sliding windows" in sliding[0][1].graph["frame"]
    assert "discarded" not in sliding[0][1].graph["frame"]


def test_a_clipped_last_window_says_it_is_short(layered: InMemoryGraphStore) -> None:
    """The last window stops at --until, so its edge count is partly an artefact of the range."""
    clipped = snapshots(
        layered, PERSONA, "speakers", window="2M", since="2025-01-01", until="2025-05-15"
    )
    assert [label for label, _ in clipped] == [
        "2025-01-01 to 2025-02-28",
        "2025-03-01 to 2025-04-30",
        "2025-05-01 to 2025-05-15",
    ]
    assert "clipped at 2025-05-15" in clipped[0][1].graph["frame"]

    exact = snapshots(
        layered, PERSONA, "speakers", window="3M", since="2025-01-01", until="2025-06-30"
    )
    assert "clipped at" not in exact[0][1].graph["frame"]


def test_snapshots_bound_themselves_by_the_corpus_when_no_dates_are_given(
    layered: InMemoryGraphStore,
) -> None:
    """Left to itself the grid runs from the first dated post to the last, and says so."""
    snaps = snapshots(layered, PERSONA, "speakers", window="3M")
    assert [label for label, _ in snaps] == ["2025-01-10 to 2025-04-09", "2025-04-10 to 2025-06-20"]
    assert sorted(snaps[0][1].nodes) == ["ana", "bo"]
    assert sorted(snaps[1][1].nodes) == ["ana", "cy"]
    assert "dot" not in set(snaps[0][1].nodes) | set(snaps[1][1].nodes)


def test_a_snapshot_over_the_edge_list_counts_activations_and_says_that_it_did(
    layered: InMemoryGraphStore,
) -> None:
    """The cheap path: windows over the timestamps, with no second read of the store.

    Its weights count activations inside the window rather than the unit the built network
    counts, so it declares that in its frame instead of passing for the other one.
    """
    history = dynamic_edges(layered, PERSONA, "entities")
    snaps = snapshots(history, window="3M", since="2025-01-01", until="2025-06-30")
    assert [label for label, _ in snaps] == ["2025-01-01 to 2025-03-31", "2025-04-01 to 2025-06-30"]
    assert weights(snaps[0][1]) == {(ALPHA, BETA): 1, (ALPHA, GAMMA): 1}
    assert weights(snaps[1][1]) == {(BETA, GAMMA): 1, (ALPHA, GAMMA): 1}
    assert "Assembled from the timestamped edge list" in snaps[0][1].graph["frame"]
    assert describe(snaps[0][1]).dynamic


def test_a_corpus_with_no_dates_has_no_windows_and_is_told_so(
    memory_store: InMemoryGraphStore,
) -> None:
    with pytest.raises(ValueError, match="nothing in this corpus is dated"):
        snapshots(memory_store, "test-layers", "speakers", window="3M")
