"""Chapters 49-51, held to answers that were true before the code was written.

Fixtures here are planted rather than drawn from the legendary graphs (`tests/legendary.py`),
except where a legendary graph's own published shape is the answer (`karate_club` for a smoke
test that a real, moderately dense network still produces a valid drawing): a layout's contract
-- equal angles, one line, a block-diagonal matrix, a layer below the predecessor -- is arithmetic
that holds on any graph with the right shape, so the fixtures are built to have exactly that
shape and nothing else to explain the number.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import networkx as nx
import pytest

from graphrag.sna.draw import (
    CATEGORICAL_PALETTE,
    MAX_CATEGORIES,
    MAX_RADIUS,
    MIN_RADIUS,
    OTHER_CATEGORY_COLOR,
    arc_layout,
    categorical_colors,
    circular_layout,
    draw,
    draw_payload,
    edge_width_scale,
    force_layout,
    matrix_layout,
    render_draw,
    sequential_palette,
    size_scale,
)
from tests.legendary import karate_club


def _lightness(hex_color: str) -> float:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (max(r, g, b) + min(r, g, b)) / 2.0


# ------------------------------------------------------------------------------- §49.1 size


def test_size_scale_puts_the_largest_value_at_the_largest_radius() -> None:
    values = {"a": 1.0, "b": 50.0, "c": 5.0, "d": 5000.0}
    radii = size_scale(values)
    ranked_by_value = sorted(values, key=lambda n: values[n])
    ranked_by_radius = sorted(radii, key=lambda n: radii[n])
    assert ranked_by_value == ranked_by_radius
    assert radii["d"] == max(radii.values())
    assert radii["a"] == min(radii.values())


def test_size_scale_caps_the_area_ratio_no_matter_how_skewed_the_values_are() -> None:
    """§49.1, Figure 49.8: a naive "radius proportional to value" mapping's area ratio grows
    without bound as the value ratio grows -- double the value, quadruple the area, and a
    thousandfold value spread would draw a millionfold area spread. ``size_scale`` places every
    value between two fixed radii instead, so the worst it can ever draw -- however extreme the
    values -- is the ratio between those radii's areas, never the raw values' own ratio.
    """
    cap = (MAX_RADIUS / MIN_RADIUS) ** 2
    for spread in (1_000.0, 10_000_000.0):
        radii = size_scale({"small": 1.0, "big": spread})
        ratio = (radii["big"] ** 2) / (radii["small"] ** 2)
        assert ratio == pytest.approx(cap)
        assert ratio < spread**2  # nowhere near the naive, unbounded mapping's own ratio


def test_size_scale_ties_get_the_midpoint_area() -> None:
    """A tie normalises to 0.5, so the radius is the square root of the *area* midpoint."""
    radii = size_scale({"a": 3.0, "b": 3.0, "c": 3.0}, min_radius=4.0, max_radius=20.0)
    expected = math.sqrt((4.0**2 + 20.0**2) / 2.0)
    assert len(set(radii.values())) == 1
    assert next(iter(radii.values())) == pytest.approx(expected)


def test_size_scale_empty_is_empty() -> None:
    assert size_scale({}) == {}


def test_edge_width_scale_is_monotone_and_linear_not_area_scaled() -> None:
    values = {"e1": 1.0, "e2": 2.0, "e3": 10.0}
    widths = edge_width_scale(values, min_width=10.0, max_width=110.0)
    ranked_by_value = sorted(values, key=lambda e: values[e])
    ranked_by_width = sorted(widths, key=lambda e: widths[e])
    assert ranked_by_value == ranked_by_width
    # Linear in the (quasi-logged) value -- no square root, unlike size_scale -- so halving the
    # width's distance from the floor exactly halves the normalised value it came from.
    normalized_e2 = (widths["e2"] - 10.0) / 100.0
    normalized_e1 = (widths["e1"] - 10.0) / 100.0
    normalized_e3 = (widths["e3"] - 10.0) / 100.0
    assert normalized_e1 == pytest.approx(0.0)
    assert normalized_e3 == pytest.approx(1.0)
    assert 0.0 < normalized_e2 < 1.0


# ------------------------------------------------------------------------------ §49.2 colour


def test_categorical_palette_caps_the_named_colours_the_book_allows() -> None:
    """§49.2, p.719: "never use more than nine colours" -- eight named plus one "other"."""
    assert len(CATEGORICAL_PALETTE) == MAX_CATEGORIES == 8
    assert len(set(CATEGORICAL_PALETTE)) == MAX_CATEGORIES  # every hue distinct


def test_categorical_colors_collapses_overflow_into_other() -> None:
    nodes = [f"n{i}" for i in range(12)]
    labels = {node: f"cat{i}" for i, node in enumerate(nodes)}  # 12 distinct categories
    result = categorical_colors(labels, all_nodes=nodes)
    used = {result.color_of[n] for n in nodes}
    assert len(used) <= MAX_CATEGORIES + 1  # 8 named + "other"
    assert OTHER_CATEGORY_COLOR in used
    assert result.collapsed == 12 - MAX_CATEGORIES


def test_categorical_colors_keeps_the_largest_categories_named() -> None:
    labels = {**dict.fromkeys(range(20), "big"), **{20 + i: f"tiny{i}" for i in range(10)}}
    result = categorical_colors(labels, all_nodes=list(labels))
    assert result.color_of[0] != OTHER_CATEGORY_COLOR  # the 20-node category kept its own hue
    assert result.legend[0][0] == "big"  # largest first


def test_categorical_colors_grey_for_a_node_with_no_label() -> None:
    result = categorical_colors({"a": "x"}, all_nodes=["a", "b"])
    assert result.color_of["b"] == OTHER_CATEGORY_COLOR
    assert result.unlabelled == 1


def test_sequential_palette_is_monotone_in_lightness() -> None:
    """§49.2 p.721: a rainbow scale is not perceptually uniform; a sequential one must be."""
    palette = sequential_palette(9)
    lightness = [_lightness(color) for color in palette]
    assert lightness == sorted(lightness, reverse=True)
    assert len(lightness) == len(set(lightness))  # strictly monotone, no ties


def test_sequential_palette_single_hue_not_a_rainbow() -> None:
    for color in sequential_palette(6):
        r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
        # a single hue's channels never reorder their rank across the ramp
        assert (r, g, b) != (g, r, b)


# --------------------------------------------------------------------------- §51.1 force


def test_force_layout_is_reproducible_with_the_same_seed() -> None:
    graph = nx.les_miserables_graph()
    first = force_layout(graph, seed=7)
    second = force_layout(graph, seed=7)
    assert first.positions == second.positions


def test_force_layout_differs_across_seeds_on_a_nontrivial_graph() -> None:
    graph = nx.les_miserables_graph()
    first = force_layout(graph, seed=1)
    second = force_layout(graph, seed=2)
    assert first.positions != second.positions


# -------------------------------------------------------------------------- §51.2 circular


def test_circular_layout_places_n_nodes_at_equal_angles() -> None:
    graph = nx.cycle_graph(6)
    layout = circular_layout(graph)
    center = (0.0, 0.0)
    radii = {
        node: math.hypot(x - center[0], y - center[1]) for node, (x, y) in layout.positions.items()
    }
    assert all(r == pytest.approx(1.0) for r in radii.values())
    angles = sorted(math.atan2(y, x) % (2 * math.pi) for x, y in layout.positions.values())
    deltas = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    deltas.append(angles[0] + 2 * math.pi - angles[-1])
    assert all(delta == pytest.approx(deltas[0]) for delta in deltas)
    assert deltas[0] == pytest.approx(2 * math.pi / 6)


def test_circular_layout_uses_the_order_given() -> None:
    graph = nx.cycle_graph(4)
    layout = circular_layout(graph, order=[3, 2, 1, 0])
    assert layout.order == (3, 2, 1, 0)


# ------------------------------------------------------------------------------- §51.2 arc


def test_arc_layout_puts_every_node_on_one_line() -> None:
    graph = nx.path_graph(5)
    layout = arc_layout(graph)
    ys = {y for _, y in layout.positions.values()}
    assert ys == {0.0}
    xs = sorted(x for x, _ in layout.positions.values())
    assert xs == list(range(5))  # every node its own position, none doubled up


# ---------------------------------------------------------------------------- §51.4 matrix


def test_matrix_layout_of_two_disjoint_cliques_ordered_by_community_is_block_diagonal() -> None:
    graph = nx.Graph()
    left = [f"L{i}" for i in range(4)]
    right = [f"R{i}" for i in range(4)]
    graph.add_edges_from((u, v) for i, u in enumerate(left) for v in left[i + 1 :])
    graph.add_edges_from((u, v) for i, u in enumerate(right) for v in right[i + 1 :])
    order = [*left, *right]  # "ordered by community" -- the two cliques are the two communities
    matrix = matrix_layout(graph, order=order)
    community_of = {node: 0 if node in left else 1 for node in order}
    index_to_node = dict(enumerate(order))
    for cell in matrix.cells:
        assert community_of[index_to_node[cell.row]] == community_of[index_to_node[cell.col]]
    # every within-clique pair appears in both directions (undirected matrix)
    assert len(matrix.cells) == 2 * (4 * 3) * 1  # 2 cliques * (4*3 ordered pairs) each direction


def test_matrix_layout_default_order_is_degree_descending() -> None:
    graph = nx.star_graph(4)  # node 0 has degree 4, the rest degree 1
    matrix = matrix_layout(graph)
    assert matrix.order[0] == 0


# -------------------------------------------------------------------------- §51.2 layered


def test_layered_draw_puts_each_node_one_layer_below_its_predecessor() -> None:
    chain = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    result = draw(chain, layout="layered")
    for u, v in chain.edges():
        assert result.positions[v][1] == result.positions[u][1] - 1


def test_layered_draw_refuses_an_undirected_network() -> None:
    with pytest.raises(ValueError, match="directed"):
        draw(nx.path_graph(4), layout="layered")


# ------------------------------------------------------------------------------- draw()


def test_draw_refuses_an_empty_network() -> None:
    with pytest.raises(ValueError, match="no nodes"):
        draw(nx.Graph())


def test_draw_refuses_an_unknown_layout() -> None:
    with pytest.raises(ValueError, match="--layout"):
        draw(nx.path_graph(3), layout="spiral")


def test_draw_refuses_an_unknown_size() -> None:
    with pytest.raises(ValueError, match="centrality"):
        draw(nx.path_graph(3), size="popularity")


def test_draw_refuses_an_unknown_color_rule() -> None:
    with pytest.raises(ValueError, match="--color"):
        draw(nx.path_graph(3), color="rainbow")


def test_draw_refuses_attr_with_no_key() -> None:
    with pytest.raises(ValueError, match="key after the colon"):
        draw(nx.path_graph(3), color="attr:")


@pytest.mark.parametrize("layout", ["force", "circular", "arc", "matrix"])
def test_draw_produces_well_formed_svg(layout: str) -> None:
    graph = nx.karate_club_graph()
    result = draw(graph, layout=layout, size="degree", color="community", seed=3)
    ET.fromstring(result.svg)  # raises if not well-formed XML


def test_draw_html_embeds_the_svg_verbatim_and_fetches_nothing() -> None:
    graph = nx.path_graph(6)
    result = draw(graph, layout="force", seed=1)
    assert result.svg in result.html
    # The SVG's own xmlns is a namespace URI, never fetched; nothing else may look like a
    # fetchable resource -- no external script, stylesheet link, or JS network call.
    assert "<script src" not in result.html
    assert "<link" not in result.html
    assert 'src="http' not in result.html
    assert 'href="http' not in result.html
    assert "fetch(" not in result.html
    assert "XMLHttpRequest" not in result.html


def test_draw_html_title_is_a_short_name_and_the_frame_is_in_the_body() -> None:
    """The page used to take the whole sampling-frame paragraph as its <title>, which a browser
    tab shows as a few words of it; the frame now sits in the body and the title is a name."""
    graph = nx.path_graph(6)
    graph.graph["frame"] = "Entities mentioned in the same passage. A long paragraph of caveats."
    default = draw(graph, layout="force", seed=1)
    assert "<title>force drawing, 6 nodes</title>" in default.html
    assert '<p class="frame">Entities mentioned in the same passage.' in default.html
    named = draw(graph, layout="circular", seed=1, title="product-leader · entities · circular")
    assert "<title>product-leader · entities · circular</title>" in named.html
    assert "<h2>product-leader · entities · circular</h2>" in named.html


def test_draw_by_community_carries_the_evaluation_section() -> None:
    graph, _ = karate_club()
    result = draw(graph, layout="force", color="community", seed=1)
    assert result.community_evaluation
    assert result.community_evaluation[0] == "## Community evaluation"


def test_draw_by_numeric_attribute_uses_the_sequential_palette() -> None:
    graph = nx.path_graph(4)
    for node in graph.nodes():
        graph.nodes[node]["attr_score"] = str(float(node))
    result = draw(graph, layout="force", color="attr:score", seed=1)
    assert "numeric" in result.color_rule


def test_draw_by_categorical_attribute_grey_for_missing_nodes() -> None:
    graph = nx.path_graph(4)
    graph.nodes[0]["attr_kind"] = "alpha"
    graph.nodes[1]["attr_kind"] = "beta"
    result = draw(graph, layout="force", color="attr:kind", seed=1)
    assert result.node_color[2] == OTHER_CATEGORY_COLOR
    assert any("carry no value" in note for note in result.notes)


def test_draw_refuses_an_attribute_no_node_carries() -> None:
    with pytest.raises(ValueError, match="no node"):
        draw(nx.path_graph(3), color="attr:nope")


def test_draw_pairs_edge_width_with_betweenness_when_asked() -> None:
    graph = nx.path_graph(5)
    result = draw(graph, layout="force", size="betweenness", seed=1)
    assert any("edge betweenness" in note for note in result.notes)


def test_render_draw_reports_frame_n_layout_and_chapter() -> None:
    graph = nx.path_graph(4)
    result = draw(graph, layout="force", seed=1)
    lines = render_draw(result)
    text = "\n".join(lines)
    assert "## Visualization" in text
    assert "ch. 49" in text and "ch. 50" in text and "ch. 51" in text
    assert "n = 4" in text
    assert "force" in text


def test_draw_payload_round_trips_positions_as_lists() -> None:
    graph = nx.path_graph(3)
    result = draw(graph, layout="force", seed=1)
    payload = draw_payload(result)
    assert payload["layout"] == "force"
    assert set(payload["positions"]) == {"0", "1", "2"}
    assert all(len(pos) == 2 for pos in payload["positions"].values())


def test_directed_reciprocal_pair_gets_two_distinct_paths() -> None:
    graph = nx.DiGraph([("a", "b"), ("b", "a")])
    result = draw(graph, layout="force", seed=1)
    assert result.svg.count("edge-curve") == 2
