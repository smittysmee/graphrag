"""Chapter 44's message passing, held to answers worked out independently of the code, and the
optional `graphrag[gnn]` extra's own contract: everything below that does not need `torch` runs
without it (the propagation matrices, the feature builder, the CLI's own input validation and its
exit-2 path); everything that does calls ``pytest.importorskip("torch")`` first, per test, so a
dev image with no `torch` -- this one -- still runs every assertion it can and skips only what it
cannot.

*The algebra (§44.3), no training involved.* :func:`test_normalized_adjacency_matches_the_hand_
formula_on_a_path` and its GraphSAGE-mean sibling build ``D̂^-1/2(I + A)D̂^-1/2`` and the row-
stochastic mean adjacency independently, with `numpy` alone, on a 3-node path where every entry
can be written out by hand. :func:`test_gcn_layer_matches_the_hand_computed_propagation` then
checks that one `torch` GCN layer -- identity features, a fixed `W` -- reproduces that same
matrix product exactly.

*Semi-supervised classification (§44.1, §44.3).* A two-block planted-partition graph, well
separated (`p_in=0.85`, `p_out=0.02`), gives each node an identity (one-hot) feature -- §44.2's
own fix for structures a GCN otherwise cannot tell apart -- and three seed labels per block; both
a GCN and a GraphSAGE fit classify every other node correctly (`held_out_accuracy == 1.0`),
against a majority-class null of 0.5.

*Link prediction through ATL-25.* The same planted-community fixture `test_sna_predict.py` uses
(community, not degree, predicts an edge here) gives `gcn_link_scorer` and its GraphSAGE sibling
an AUC through `evaluate_predictor` above both preferential attachment's degenerate case and the
random baseline.

*Entity-attribute completion.* Two labelled cliques joined by one edge, six owned values per side
minus one held out each, recovers the held-out values and the two genuinely unlabelled ones
(§44.1/§44.3, filtered to owned labels only -- see `complete_attribute`'s own docstring for why
borrowed ones are excluded from training).

*Attention (ch. 45, §45.1).* :func:`test_gat_layer_matches_the_hand_computed_attention_on_a_path`
works out one head's ``LeakyReLU``/softmax arithmetic on a 3-node path by hand, in the test itself,
independently of `gat_layer`. :func:`test_gat_layer_with_zeroed_attention_is_a_self_inclusive_mean`
checks the reduction the book's own framing implies -- GCN is a GAT with fixed, uniform attention --
by zeroing the attention vector and comparing against `mean_adjacency` on a copy of the graph with
self loops added (GAT's neighbourhood always includes the node itself, unlike GraphSAGE's).
:func:`test_gat_layer_on_a_star_graph_asymmetric_attention` is the structural case §45.1
motivates GATs with: a leaf spends most of its own attention on the hub, while the hub -- with
many more neighbours to split attention over -- spends much less on any one leaf; a fixed
``1/sqrt(kv * ku)`` GCN weight could not tell these two directions apart. `classify`,
`gcn_link_scorer` and `complete_attribute` each get one more test with ``method="gat"``, mirroring
their gcn/graphsage siblings above.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.models import Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport
from graphrag.sna import gnn
from graphrag.sna.export import attr_key
from graphrag.sna.matrices import node_order

runner = CliRunner()


def _path3() -> tuple[nx.Graph, list[str]]:
    graph = nx.relabel_nodes(nx.path_graph(3), str)
    return graph, ["0", "1", "2"]


# ------------------------------------------------------------------------- the algebra, no torch


def test_normalized_adjacency_matches_the_hand_formula_on_a_path() -> None:
    """0-1-2: A+I = [[1,1,0],[1,1,1],[0,1,1]], degrees 2,3,2; D^-1/2(A+I)D^-1/2 computed
    independently with plain ``numpy.to_numpy_array`` rather than through the module under test.
    """
    graph, nodes = _path3()
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    hat = adjacency + np.eye(3)
    degree = hat.sum(axis=1)
    assert degree.tolist() == [2.0, 3.0, 2.0]
    scale = np.diag(1.0 / np.sqrt(degree))
    expected = scale @ hat @ scale

    result = gnn.normalized_adjacency(graph, nodes)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.diag(result), [0.5, 1 / 3, 0.5])


def test_mean_adjacency_is_the_row_stochastic_adjacency_without_a_self_loop() -> None:
    """0-1-2: node 1's row is its two neighbours split evenly; the endpoints' rows are their one
    neighbour alone. No self loop anywhere, unlike :func:`gnn.normalized_adjacency`.
    """
    graph, nodes = _path3()

    result = gnn.mean_adjacency(graph, nodes)

    np.testing.assert_allclose(result, [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])


def test_mean_adjacency_leaves_an_isolated_node_at_zero() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    graph.add_edge("a", "b")
    graph.add_node("c")

    result = gnn.mean_adjacency(graph, ["a", "b", "c"])

    np.testing.assert_allclose(result[2], [0.0, 0.0, 0.0])


def test_neighbor_mask_is_the_binary_adjacency_with_a_self_loop() -> None:
    """0-1-2: every node is its own neighbour (§45.1's "including itself"), plus its one or two
    graph neighbours; no weighting at all, unlike :func:`gnn.normalized_adjacency` or
    :func:`gnn.mean_adjacency`.
    """
    graph, nodes = _path3()

    result = gnn.neighbor_mask(graph, nodes)

    np.testing.assert_allclose(result, [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])


def test_build_features_falls_back_to_scaled_degree_without_text() -> None:
    graph, nodes = _path3()

    features, names, has_text = gnn.build_features(graph, nodes, None)

    assert names == ["degree"]
    assert has_text == []
    # Degrees are 1, 2, 1 (unweighted path); scaled by the max, 2.
    np.testing.assert_allclose(features.ravel(), [0.5, 1.0, 0.5])


def test_build_features_combines_text_with_degree_and_zero_fills_the_rest() -> None:
    graph, nodes = _path3()
    text = (["1"], np.array([[3.0, 4.0]]))  # only node "1" has a stored embedding

    features, names, has_text = gnn.build_features(graph, nodes, text)

    assert names == ["text_0", "text_1", "degree"]
    assert has_text == ["1"]
    np.testing.assert_allclose(features[0], [0.0, 0.0, 0.5])  # node "0": no text, degree 1/2
    np.testing.assert_allclose(features[1], [3.0, 4.0, 1.0])  # node "1": its own row, degree 2/2
    np.testing.assert_allclose(features[2], [0.0, 0.0, 0.5])


# ------------------------------------------------------------------- exit-2 without torch, CLI


def test_sna_complete_rejects_an_unknown_method_before_checking_for_torch() -> None:
    result = runner.invoke(
        app, ["sna", "complete", "whatever-persona", "region", "--method", "bogus"]
    )
    assert result.exit_code == 2
    assert "method must be one of" in result.output


def test_sna_complete_rejects_unparsable_hidden_dims_before_checking_for_torch() -> None:
    result = runner.invoke(
        app, ["sna", "complete", "whatever-persona", "region", "--hidden-dims", "sixteen"]
    )
    assert result.exit_code == 2
    assert "--hidden-dims" in result.output


def test_sna_complete_exits_2_and_explains_when_torch_is_missing() -> None:
    if gnn.available():
        pytest.skip("torch is installed in this environment; the exit-2 path is not exercised")

    result = runner.invoke(app, ["sna", "complete", "whatever-persona", "region"])

    assert result.exit_code == 2
    assert "graphrag[gnn]" in result.output
    assert "torch" in result.output


def test_sna_complete_with_gat_validates_hidden_dims_before_checking_for_torch() -> None:
    """``gat`` is one of `gnn.METHODS`, so a bad `--hidden-dims` alongside it is still caught by
    the CLI's own parsing before the torch presence check runs, exactly like gcn/graphsage.
    """
    result = runner.invoke(
        app,
        [
            "sna",
            "complete",
            "whatever-persona",
            "region",
            "--method",
            "gat",
            "--hidden-dims",
            "sixteen",
        ],
    )
    assert result.exit_code == 2
    assert "--hidden-dims" in result.output


def test_sna_complete_rejects_a_gat_heads_below_one() -> None:
    result = runner.invoke(
        app,
        ["sna", "complete", "whatever-persona", "region", "--method", "gat", "--gat-heads", "0"],
    )
    assert result.exit_code == 2
    assert "--gat-heads" in result.output


def test_sna_complete_with_gat_exits_2_and_explains_when_torch_is_missing() -> None:
    if gnn.available():
        pytest.skip("torch is installed in this environment; the exit-2 path is not exercised")

    result = runner.invoke(
        app, ["sna", "complete", "whatever-persona", "region", "--method", "gat"]
    )

    assert result.exit_code == 2
    assert "graphrag[gnn]" in result.output
    assert "torch" in result.output


def test_available_agrees_with_whether_torch_actually_imports() -> None:
    if gnn.available():
        import torch  # noqa: F401 -- proves the presence check was right, nothing else
    else:
        with pytest.raises(ModuleNotFoundError):
            gnn._require_torch()


# ------------------------------------------------------------------------------- torch required


def _planted_two_blocks(size: int = 10) -> tuple[nx.Graph, dict[str, str]]:
    """Two dense, well-separated blocks (p_in=0.85, p_out=0.02): community, not degree,
    predicts a node's own label here, the same shape as `test_sna_predict.py`'s own fixture.
    """
    raw = nx.planted_partition_graph(2, size, 0.85, 0.02, seed=7)
    truth = {f"n{n}": ("A" if n < size else "B") for n in raw.nodes}
    return nx.relabel_nodes(raw, lambda n: f"n{n}"), truth


def test_gcn_layer_matches_the_hand_computed_propagation() -> None:
    torch = pytest.importorskip("torch")
    graph, nodes = _path3()
    propagation = gnn.normalized_adjacency(graph, nodes)
    identity = np.eye(3)
    weight = np.array([[1.0, 0.5], [0.0, -1.0], [2.0, 0.3]])
    expected = propagation @ identity @ weight

    result = gnn.gcn_layer(
        torch.tensor(identity, dtype=torch.float64),
        torch.tensor(propagation, dtype=torch.float64),
        torch.tensor(weight, dtype=torch.float64),
    )

    np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-12)


def test_gcn_layer_applies_the_given_activation() -> None:
    torch = pytest.importorskip("torch")
    graph, nodes = _path3()
    propagation = torch.tensor(gnn.normalized_adjacency(graph, nodes), dtype=torch.float64)
    h = torch.tensor(np.eye(3), dtype=torch.float64)
    weight = torch.tensor(-np.ones((3, 2)), dtype=torch.float64)

    result = gnn.gcn_layer(h, propagation, weight, activation=torch.relu)

    assert bool((result >= 0).all())


def test_gat_layer_matches_the_hand_computed_attention_on_a_path() -> None:
    """0-1-2, identity features, ``W = [[1],[2],[3]]`` so ``z_v == v``'s own row of ``W`` (a
    scalar per node: ``z_0=1, z_1=2, z_2=3``), attention vector ``a = [1, -1]``.

    By hand: ``e_vu = LeakyReLU(a_src . z_v + a_dst . z_u) = LeakyReLU(z_v - z_u)``, negative
    slope 0.2 (this module's default). Node 0's neighbourhood is ``{0, 1}``:
    ``e_00 = LeakyReLU(0) = 0``, ``e_01 = LeakyReLU(-1) = -0.2``; softmax gives
    ``alpha_00 = 1 / (1 + e^-0.2) = 0.549834``, ``alpha_01 = 0.450166``, so
    ``out_0 = 0.549834*1 + 0.450166*2 = 1.450166``. Node 1's neighbourhood is ``{0, 1, 2}``:
    ``e_10 = LeakyReLU(1) = 1``, ``e_11 = 0``, ``e_12 = LeakyReLU(-1) = -0.2``; softmax over
    ``[1, 0, -0.2]`` gives ``[0.599135, 0.220409, 0.180456]``, so
    ``out_1 = 0.599135*1 + 0.220409*2 + 0.180456*3 = 1.581321``. Node 2's neighbourhood is
    ``{1, 2}``: ``e_21 = LeakyReLU(1) = 1``, ``e_22 = 0``; softmax over ``[1, 0]`` gives
    ``[0.731059, 0.268941]``, so ``out_2 = 0.731059*2 + 0.268941*3 = 2.268941``.
    """
    torch = pytest.importorskip("torch")
    graph, nodes = _path3()
    mask = torch.tensor(gnn.neighbor_mask(graph, nodes), dtype=torch.float64)
    h = torch.tensor(np.eye(3), dtype=torch.float64)
    weight = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    attention = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

    out, alpha = gnn.gat_layer(h, mask, weight, attention)

    np.testing.assert_allclose(
        alpha.detach().numpy(),
        [
            [0.549834, 0.450166, 0.0],
            [0.599135, 0.220409, 0.180456],
            [0.0, 0.731059, 0.268941],
        ],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out.detach().numpy().ravel(), [1.450166, 1.581321, 2.268941], atol=1e-5
    )


def test_gat_layer_with_zeroed_attention_is_a_self_inclusive_mean() -> None:
    """§45.1's neighbourhood always includes the node itself, unlike GraphSAGE's -- so a zeroed
    attention vector (every ``e_vu`` the same, softmax uniform) reduces to the mean over a
    *self-loop-augmented* copy of the graph, computed independently with `mean_adjacency` on that
    copy rather than by re-deriving the softmax."""
    torch = pytest.importorskip("torch")
    graph, nodes = _path3()
    mask = torch.tensor(gnn.neighbor_mask(graph, nodes), dtype=torch.float64)
    h = torch.tensor(np.eye(3), dtype=torch.float64)
    weight = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    attention = torch.zeros(1, 2, dtype=torch.float64)
    graph_with_self_loops = graph.copy()
    for node in nodes:
        graph_with_self_loops.add_edge(node, node, weight=1)
    expected = (
        gnn.mean_adjacency(graph_with_self_loops, nodes)
        @ np.eye(3)
        @ np.array([[1.0], [2.0], [3.0]])
    )

    out, alpha = gnn.gat_layer(h, mask, weight, attention)

    np.testing.assert_allclose(
        alpha.detach().numpy(), [[0.5, 0.5, 0], [1 / 3, 1 / 3, 1 / 3], [0, 0.5, 0.5]]
    )
    np.testing.assert_allclose(out.detach().numpy(), expected)


def test_gat_layer_on_a_star_graph_asymmetric_attention() -> None:
    """A leaf's neighbourhood is only ``{leaf, hub}``, so its softmax puts a large share on the
    hub; the hub's neighbourhood is ``{hub, every leaf}``, so its softmax has five other leaves to
    split weight across and gives any one of them much less back -- the direction-dependent
    weighting §45.1 says a fixed GCN normalisation (``1/sqrt(kv * ku)``, symmetric) cannot
    express.
    """
    torch = pytest.importorskip("torch")
    star = nx.relabel_nodes(nx.star_graph(5), str)  # node "0" is the hub, "1".."5" the leaves
    from graphrag.sna.matrices import node_order as _node_order

    nodes = _node_order(star)
    mask = torch.tensor(gnn.neighbor_mask(star, nodes), dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    h = torch.rand(len(nodes), 2, generator=generator, dtype=torch.float64)
    weight = torch.rand(2, 1, generator=generator, dtype=torch.float64) * 2 - 1
    attention = torch.rand(1, 2, generator=generator, dtype=torch.float64) * 2 - 1

    _out, alpha = gnn.gat_layer(h, mask, weight, attention)

    hub, leaf_a, leaf_b = nodes.index("0"), nodes.index("1"), nodes.index("2")
    assert alpha[leaf_a, hub].item() > alpha[hub, leaf_a].item()
    assert alpha[leaf_b, hub].item() > alpha[hub, leaf_b].item()


@pytest.mark.parametrize("method", ["gcn", "graphsage"])
def test_classify_reaches_perfect_held_out_accuracy_on_a_well_separated_partition(
    method: str,
) -> None:
    """A few one-hot-featured seeds per block, three each, is enough for both architectures to
    place every other node correctly -- against a majority-class null of 0.5, since the two
    blocks are equal size.
    """
    pytest.importorskip("torch")
    graph, truth = _planted_two_blocks()
    nodes = node_order(graph)
    identity = (nodes, np.eye(len(nodes)))
    block_a = [n for n in nodes if truth[n] == "A"]
    block_b = [n for n in nodes if truth[n] == "B"]
    seeds = dict.fromkeys(block_a[:3], "A") | dict.fromkeys(block_b[:3], "B")
    held_out = {n: truth[n] for n in nodes if n not in seeds}

    fit = gnn.classify(
        graph,
        seeds,
        method=method,  # type: ignore[arg-type]
        features=identity,
        hidden_dims=(16,),
        epochs=150,
        lr=0.1,
        seed=1,
        held_out=held_out,
    )

    assert fit.held_out_accuracy == 1.0
    assert fit.majority_accuracy == 0.5
    assert fit.n_held_out == len(held_out)
    for node, true_value in held_out.items():
        assert fit.predicted(node)[0] == true_value


def test_classify_with_gat_reaches_perfect_held_out_accuracy_on_a_well_separated_partition() -> (
    None
):
    """The same fixture and seeds as the gcn/graphsage version above, ``method="gat"`` with two
    attention heads -- one head alone (§45.1's minimal case) underfits this split at the same
    epoch/lr budget the other two methods use; two heads reaches the same perfect floor.
    """
    pytest.importorskip("torch")
    graph, truth = _planted_two_blocks()
    nodes = node_order(graph)
    identity = (nodes, np.eye(len(nodes)))
    block_a = [n for n in nodes if truth[n] == "A"]
    block_b = [n for n in nodes if truth[n] == "B"]
    seeds = dict.fromkeys(block_a[:3], "A") | dict.fromkeys(block_b[:3], "B")
    held_out = {n: truth[n] for n in nodes if n not in seeds}

    fit = gnn.classify(
        graph,
        seeds,
        method="gat",
        features=identity,
        hidden_dims=(16,),
        epochs=150,
        lr=0.1,
        seed=1,
        held_out=held_out,
        heads=2,
    )

    assert fit.held_out_accuracy == 1.0
    assert fit.majority_accuracy == 0.5
    assert fit.attention is not None
    for node, true_value in held_out.items():
        assert fit.predicted(node)[0] == true_value
        # Every node's own attention row sums (near) to 1 over the neighbours it kept.
        assert sum(w for _n, w in fit.top_neighbours(node)) <= 1.0 + 1e-6


def test_classify_refuses_fewer_than_two_labelled_nodes() -> None:
    pytest.importorskip("torch")
    graph, _nodes = _path3()
    with pytest.raises(ValueError, match="at least two labelled"):
        gnn.classify(graph, {"0": "x"})


def test_classify_refuses_a_single_class() -> None:
    pytest.importorskip("torch")
    graph, _nodes = _path3()
    with pytest.raises(ValueError, match="at least two distinct classes"):
        gnn.classify(graph, {"0": "x", "1": "x"})


def test_classify_refuses_a_label_for_a_node_the_graph_does_not_have() -> None:
    pytest.importorskip("torch")
    graph, _nodes = _path3()
    with pytest.raises(ValueError, match="not in the graph"):
        gnn.classify(graph, {"0": "x", "not-a-node": "y"})


def _planted_communities() -> nx.Graph:
    """`test_sna_predict.py`'s own fixture: two dense 15-node communities, sparsely cross-joined,
    where preferential attachment's degree signal carries no community information.
    """
    raw = nx.planted_partition_graph(2, 15, 0.6, 0.05, seed=3)
    return nx.relabel_nodes(raw, lambda n: f"n{n}")


@pytest.mark.parametrize("method", ["gcn", "graphsage", "gat"])
def test_gcn_link_scorer_beats_random_through_evaluate_predictor(method: str) -> None:
    """Random (not degree -- see `build_features`'s own note on why) per-node features, so the
    two-layer GCN/GraphSAGE/GAT has more than one degree of freedom to fit the graph's own edges
    against; ``gat`` uses the default single head (§45.1's minimal case).
    """
    pytest.importorskip("torch")
    from graphrag.sna.experiment import evaluate_predictor, holdout, random_scores

    communities = _planted_communities()
    split = holdout(communities, share=0.1, seed=1)
    nodes = node_order(communities)
    features = (nodes, np.random.RandomState(0).normal(size=(len(nodes), 8)))
    scorer = gnn.gcn_link_scorer(
        method=method,  # type: ignore[arg-type]
        features=features,
        hidden_dims=(16, 16),
        epochs=200,
        lr=0.05,
        seed=1,
    )

    report = evaluate_predictor(split, scorer, name=method, k=10)
    random_report = evaluate_predictor(split, lambda tr: random_scores(tr, seed=1), k=10)

    assert report.auc > 0.5
    assert report.auc > random_report.auc


def test_embed_gnn_refuses_a_graph_with_no_edges() -> None:
    pytest.importorskip("torch")
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    with pytest.raises(ValueError, match="at least one edge"):
        gnn.embed_gnn(graph)


# ---------------------------------------------------------------- entity-attribute completion


def _two_labelled_cliques() -> nx.Graph:
    """Two 6-node cliques joined by one edge; five of each side's nodes own a value for "kind",
    the sixth on each side is left unlabelled -- what :func:`gnn.complete_attribute` should
    suggest a value for.
    """
    graph = nx.Graph()
    for side, value in (("A", "alpha"), ("B", "beta")):
        side_nodes = [f"{side}-{i}" for i in range(6)]
        graph.add_nodes_from(side_nodes)
        for i in range(len(side_nodes)):
            for j in range(i + 1, len(side_nodes)):
                graph.add_edge(side_nodes[i], side_nodes[j], weight=1)
        for node in side_nodes[:5]:
            graph.nodes[node][attr_key("kind")] = value
    graph.add_edge("A-0", "B-0", weight=1)
    return graph


def _random_text(graph: nx.Graph, dims: int = 6, seed: int = 3) -> tuple[list[str], np.ndarray]:
    nodes = sorted(graph.nodes)
    return nodes, np.random.RandomState(seed).normal(size=(len(nodes), dims))


def test_complete_attribute_recovers_the_two_unlabelled_nodes() -> None:
    pytest.importorskip("torch")
    graph = _two_labelled_cliques()

    report = gnn.complete_attribute(
        graph,
        "kind",
        method="gcn",
        features=_random_text(graph),
        hidden_dims=(16,),
        epochs=150,
        lr=0.1,
        seed=2,
        val_share=0.2,
    )

    assert report.n_own == 10
    assert report.n_borrowed == 0
    assert report.n_unlabelled == 2
    assert report.held_out_accuracy == 1.0
    suggested = {s.node: s.value for s in report.suggestions}
    assert suggested == {"A-5": "alpha", "B-5": "beta"}
    for suggestion in report.suggestions:
        assert suggestion.confidence > 0.9
        assert suggestion.distribution[0][0] == suggestion.value


def test_complete_attribute_with_gat_recovers_the_two_unlabelled_nodes_and_their_neighbours() -> (
    None
):
    """Same fixture as the gcn version above, ``method="gat"`` with four attention heads and more
    epochs -- what this specific split and seed need to reach the same perfect held-out floor
    (a single head at 150 epochs, `test_classify_with_gat_...` above's budget, underfits it).
    Also checks the one thing only ``gat`` reports: which entities the fitted attention leaned on.
    """
    pytest.importorskip("torch")
    graph = _two_labelled_cliques()

    report = gnn.complete_attribute(
        graph,
        "kind",
        method="gat",
        features=_random_text(graph),
        hidden_dims=(16,),
        epochs=300,
        lr=0.1,
        seed=2,
        val_share=0.2,
        heads=4,
    )

    assert report.chapter == "§44.1, §44.3, §45.1"
    assert report.n_own == 10
    assert report.n_borrowed == 0
    assert report.n_unlabelled == 2
    assert report.held_out_accuracy == 1.0
    suggested = {s.node: s.value for s in report.suggestions}
    assert suggested == {"A-5": "alpha", "B-5": "beta"}
    for suggestion in report.suggestions:
        assert suggestion.confidence > 0.9
        assert suggestion.distribution[0][0] == suggestion.value
        # Every entity this leaned on is a member of the same clique, never the other one.
        assert suggestion.top_neighbours
        side = suggestion.node.split("-")[0]
        assert all(n.startswith(side) for n, _w in suggestion.top_neighbours)

    rendered = gnn.render_completion(report)
    assert "leaned on" in rendered


def test_complete_attribute_excludes_borrowed_values_from_training() -> None:
    """A borrowed label (the export builder's own `INHERITED` marker) is not training signal:
    with only 4 owned values left, `complete_attribute` refuses rather than fit on them.
    """
    pytest.importorskip("torch")
    from graphrag.sna.export import attr_source_key

    graph = _two_labelled_cliques()
    for node in ("A-3", "A-4", "B-3", "B-4", "B-2", "A-2"):
        graph.nodes[node][attr_source_key("kind")] = "document"

    with pytest.raises(ValueError, match="own a value"):
        gnn.complete_attribute(graph, "kind", features=_random_text(graph))


def test_complete_attribute_refuses_below_min_own_labels() -> None:
    pytest.importorskip("torch")
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    graph.nodes["a"][attr_key("kind")] = "x"
    graph.nodes["b"][attr_key("kind")] = "y"

    with pytest.raises(ValueError, match="own a value"):
        gnn.complete_attribute(graph, "kind")


def test_render_completion_prints_the_frame_the_null_and_the_chapter() -> None:
    pytest.importorskip("torch")
    graph = _two_labelled_cliques()
    report = gnn.complete_attribute(
        graph, "kind", features=_random_text(graph), hidden_dims=(16,), epochs=50, seed=2
    )

    rendered = gnn.render_completion(report)

    assert "Sampling frame" in rendered
    assert "Null model" in rendered
    assert "Atlas ch. 44" in rendered
    assert "A-5" in rendered and "B-5" in rendered
    assert "Nothing above was written to the graph" in rendered


def test_completion_payload_round_trips_through_json() -> None:
    pytest.importorskip("torch")
    graph = _two_labelled_cliques()
    report = gnn.complete_attribute(
        graph, "kind", features=_random_text(graph), hidden_dims=(16,), epochs=50, seed=2
    )

    payload = json.loads(json.dumps(gnn.completion_payload(report)))

    assert payload["key"] == "kind"
    assert payload["n_own"] == 10
    assert {s["node"] for s in payload["suggestions"]} == {"A-5", "B-5"}


# ----------------------------------------------------------------------------- CLI, end to end


@pytest.fixture
def entities_for_completion(cli_context: AppContext, ingested: IngestReport) -> AppContext:
    """Eight entities over the sample corpus's three documents, six owned across two "kind"
    values ("metric" / "concept"), two left unlabelled -- what `sna complete` should suggest a
    value for. Mirrors `test_sna_cli.py`'s own ``enriched`` fixture, with more entities.
    """
    store = cli_context.store
    metric = ["metric:retention", "metric:activation", "metric:churn"]
    concept = ["concept:onboarding", "concept:pricing", "concept:roadmap"]
    unlabelled = ["concept:extra-a", "concept:extra-b"]
    entities = [Entity(id=e, name=e, type="concept") for e in metric + concept + unlabelled]
    per_document = [
        metric[:2] + concept[:2] + unlabelled[:1],
        metric[1:] + concept[1:] + unlabelled[1:],
        [metric[0], metric[2], concept[0], concept[2], *unlabelled],
    ]
    doc_ids = sorted(store.document_ids("test-pm"))
    mentions: list[Mention] = []
    for doc_id, entity_ids in zip(doc_ids, per_document, strict=True):
        for chunk in store.document_chunks(doc_id, 0, 100):
            mentions += [Mention(chunk_id=chunk.id, entity_id=e) for e in entity_ids]
    store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))
    for entity_id in metric:
        store.set_entity_attributes("test-pm", entity_id, {"kind": "metric"})
    for entity_id in concept:
        store.set_entity_attributes("test-pm", entity_id, {"kind": "concept"})
    return cli_context


def test_sna_complete_writes_a_report_and_json_end_to_end(
    entities_for_completion: AppContext, tmp_path: Path
) -> None:
    """The command's own wiring, through `build_network`, `mean_embeddings` and
    `complete_attribute`, to a written report and JSON: the frame counts six owned values and
    the two genuinely unlabelled entities get a suggestion each.
    """
    pytest.importorskip("torch")
    out = tmp_path / "completion.md"
    as_json = tmp_path / "completion.json"

    result = runner.invoke(
        app,
        [
            "sna",
            "complete",
            "test-pm",
            "kind",
            "--min-weight",
            "1",
            "--epochs",
            "30",
            "--seed",
            "2",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Attribute completion: kind" in result.output
    payload = json.loads(as_json.read_text())
    assert payload["n_own"] == 6
    assert {s["node"] for s in payload["suggestions"]} == {"concept:extra-a", "concept:extra-b"}


def test_sna_complete_with_gat_prints_leaned_on_neighbours_end_to_end(
    entities_for_completion: AppContext, tmp_path: Path
) -> None:
    """``--method gat`` through the same command's wiring: the report gets a "leaned on" column
    and the JSON a ``top_neighbours`` entry per suggestion, which ``gcn``/``graphsage`` never
    produce (`test_sna_complete_writes_a_report_and_json_end_to_end` above, unchanged).
    """
    pytest.importorskip("torch")
    as_json = tmp_path / "completion.json"

    result = runner.invoke(
        app,
        [
            "sna",
            "complete",
            "test-pm",
            "kind",
            "--method",
            "gat",
            "--gat-heads",
            "2",
            "--min-weight",
            "1",
            "--epochs",
            "30",
            "--seed",
            "2",
            "--json",
            str(as_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "leaned on" in result.output
    payload = json.loads(as_json.read_text())
    assert {s["node"] for s in payload["suggestions"]} == {"concept:extra-a", "concept:extra-b"}
    for suggestion in payload["suggestions"]:
        assert suggestion["top_neighbours"]
