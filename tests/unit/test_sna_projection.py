"""Chapter 26's projection schemes, held to the numbers the book prints in its own figures.

Four kinds of known answer are used here, and none of them is "the function returned something":

* **The chapter's worked figures.** Figure 26.6 prints four weights for one pair of incidence
  rows (simple 2, cosine 0.66, Pearson + 1 1.52, 1/(Euclidean + 1) 0.41 -- truncated, not
  rounded, which is what pins the example down); figures 26.5, 26.7,
  26.8 and 26.9 print the simple, hyperbolic, resource-allocation and random-walk weights of one
  eight-node toy network. :data:`TOY` is that toy network, reconstructed from the arithmetic the
  captions state, and it reproduces all five of figure 26.5's labels and all five of figure
  26.7's, which is what makes the reconstruction trustworthy rather than merely consistent.
* **Hand arithmetic on a legendary graph.** Two named pairs of Southern women (§53.4), whose
  shared events and event sizes are written into the test as fractions, so the expected weight is
  computed from the data by hand rather than from the code.
* **Identities the chapter states.** "HeatS is the transpose of ProbS" (p. 376); the hybrid at
  λ=1 is ProbS and at λ=0 is HeatS; the two-step transition matrix is row-stochastic and its
  stationary distribution is the one §26.5 multiplies by.
* **Bit-identity with the naive loop.** The simple scheme must equal a plain
  count-the-shared-neighbours loop, edge for edge and type for type, on Southern women and on a
  seeded random bipartite graph -- it is the default, and the default's numbers may not move.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from fractions import Fraction
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.pipeline import IngestReport
from graphrag.sna.export import bipartite_projection, topic_co_occurrence
from graphrag.sna.null import bipartite_preserving
from graphrag.sna.projection import (
    ASYMMETRIC,
    PAIRS,
    PROJECTION,
    PROJECTION_LAMBDA,
    SCHEMES,
    compare_schemes,
    graph_from_incidence,
    project,
    project_incidence,
    projections_payload,
    render_projections,
    symmetrise,
)
from tests.legendary import southern_women

# --------------------------------------------------------------------------- the book's toy


def _toy() -> np.ndarray:
    """The bipartite network behind figures 26.4-26.9, rebuilt from the captions' arithmetic.

    The chapter never draws the whole thing, but it states enough to pin it down. Figure 26.8
    says that when connecting node 1 to node 2 "the two common neighbors have degree of 3 and 8,
    respectively, and node 1 has degree of two", while "node 2 has three neighbors". So the
    opposite mode holds one node of degree 8 -- the power user of §26.1 who watched everything --
    and two of degree 3, node 1 sits in the big one and one small one, and node 2 in the big one
    and both small ones.

    That is this matrix: eight V1 nodes on the rows, three V2 nodes on the columns, the first of
    which touches every row. It reproduces figure 26.5's five labelled simple weights (2, 2, 2,
    2 and 3), figure 26.7's five hyperbolic ones (.46 four times and .79), figure 26.8's 0.229
    and 0.153, and figure 26.9's 0.049 and 0.022 -- every number the chapter prints for it.
    """
    matrix = np.zeros((8, 3))
    matrix[:, 0] = 1.0  # the node of degree 8, joined to all eight
    for row in (0, 1, 2):
        matrix[row, 1] = 1.0  # degree 3
    for row in (1, 2, 3):
        matrix[row, 2] = 1.0  # degree 3
    return matrix


TOY = _toy()

#: Figure 26.6's inset: two rows of an incidence matrix, three memberships each, two shared. The
#: length is ten, which is what the book's Pearson of 0.52 pins down -- at nine columns it would
#: be 0.50 and at eight 0.47.
FIGURE_26_6 = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)


def _reference_projection(pairs: list[tuple[str, str]], side: str = "left") -> nx.Graph:
    """The naive loop §26.1 describes, written out: count the shared opposite-side nodes.

    This is the implementation the package had before chapter 26 arrived, kept here as the thing
    the ``simple`` scheme has to stay bit-identical to.
    """
    members: dict[str, set[str]] = {}
    for left, right in sorted(set(pairs)):
        key, value = (right, left) if side == "left" else (left, right)
        members.setdefault(key, set()).add(value)
    counts: Counter[tuple[str, str]] = Counter()
    for group in members.values():
        counts.update(combinations(sorted(group), 2))
    graph = nx.Graph()
    graph.add_nodes_from(sorted({v for group in members.values() for v in group}))
    for (a, b), weight in counts.items():
        graph.add_edge(a, b, weight=weight)
    return graph


def _southern_women_pairs() -> list[tuple[str, str]]:
    """The 89 attendances as ``(woman, event)`` memberships, whichever way the edge is stored."""
    graph, _ = southern_women()
    return [(u, v) for u, v in graph.edges if graph.nodes[u]["bipartite"] == 0] + [
        (v, u) for u, v in graph.edges if graph.nodes[u]["bipartite"] == 1
    ]


def project_incidence_of(pairs: list[tuple[str, str]], scheme: str) -> np.ndarray:
    """The scheme's raw matrix over a two-mode edge list, projected onto the left mode."""
    from graphrag.sna.matrices import incidence

    matrix, _, _ = incidence(sorted(set(pairs)))
    return project_incidence(np.asarray(matrix, dtype=float), scheme)


def _random_bipartite(seed: int = 26) -> list[tuple[str, str]]:
    """A seeded two-mode edge list: 30 left nodes in up to five of 12 right nodes each."""
    rng = random.Random(seed)
    return [
        (f"L{left:02d}", f"R{right:02d}")
        for left in range(30)
        for right in rng.sample(range(12), rng.randint(1, 5))
    ]


# --------------------------------------------------------------------------- §26.2 figure 26.6


def test_figure_26_6_four_weights_for_one_pair_of_rows() -> None:
    """Simple 2, cosine 0.66, Pearson + 1 1.52, 1/(Euclidean + 1) 0.41 (p. 373)."""
    assert project_incidence(FIGURE_26_6, "simple")[0, 1] == pytest.approx(2.0)
    assert project_incidence(FIGURE_26_6, "cosine")[0, 1] == pytest.approx(2 / 3)
    assert project_incidence(FIGURE_26_6, "pearson")[0, 1] == pytest.approx(1 + 11 / 21)
    assert project_incidence(FIGURE_26_6, "euclidean")[0, 1] == pytest.approx(1 / (2**0.5 + 1))

    # The figure prints two decimals, and it *truncates* rather than rounds: it gives cosine as
    # 0.66 where 2/3 rounds to 0.67. Truncation is asserted, because that is what pins the
    # example down -- Pearson at 1.52 and Euclidean at 0.41 are truncations of 1.5238 and 0.4142
    # too, and a rounding reading of 0.66 would have to come from a different pair of vectors.
    def truncated(scheme: str) -> str:
        value = float(project_incidence(FIGURE_26_6, scheme)[0, 1])
        return f"{math.floor(value * 100) / 100:.2f}"

    assert truncated("cosine") == "0.66"
    assert truncated("pearson") == "1.52"
    assert truncated("euclidean") == "0.41"


def test_jaccard_is_the_intersection_over_the_union() -> None:
    """§26.1's "Jaccard correction": three memberships each, two shared, so 2/4."""
    assert project_incidence(FIGURE_26_6, "jaccard")[0, 1] == pytest.approx(0.5)


def test_a_pair_that_shares_nothing_is_not_an_edge_under_any_scheme() -> None:
    """A projection joins nodes with common neighbours (§26.7), so the vector schemes -- which
    would otherwise score every pair of rows -- are masked to the same support as the count."""
    disjoint = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    for scheme in SCHEMES:
        assert project_incidence(disjoint, scheme)[0, 1] == 0.0


# --------------------------------------------------------------------------- §26.1 figure 26.5


def test_figure_26_5_labels_the_four_twos_and_the_three() -> None:
    weights = project_incidence(TOY, "simple")
    assert weights[0, 1] == 2 and weights[0, 2] == 2
    assert weights[1, 3] == 2 and weights[2, 3] == 2
    assert weights[1, 2] == 3
    # And the hairball the chapter is complaining about: the node of degree 8 joins all 28 pairs.
    assert int((weights > 0).sum() / 2) == 28


# --------------------------------------------------------------------------- §26.3 figure 26.7


def test_figure_26_7_hyperbolic_weights_are_point_four_six_and_point_seven_nine() -> None:
    """Each shared node contributes 1/k: .46 is 1/3 + 1/8 and .79 is 1/3 + 1/3 + 1/8 (p. 374).

    p. 374 also *displays* ``1/(k_z - 1)``, which would give 0.64 and 1.14 here. The figure, the
    caption and §26.4's matrix description all compute ``1/k_z``; see the module docstring of
    ``graphrag.sna.projection``.
    """
    weights = project_incidence(TOY, "hyperbolic")
    assert weights[0, 1] == pytest.approx(1 / 3 + 1 / 8)
    assert f"{weights[0, 1]:.2f}" == "0.46"
    for pair in ((0, 1), (0, 2), (1, 3), (2, 3)):
        assert f"{weights[pair]:.2f}" == "0.46"
    assert weights[1, 2] == pytest.approx(1 / 3 + 1 / 3 + 1 / 8)
    assert f"{weights[1, 2]:.2f}" == "0.79"


def test_hyperbolic_discounts_the_hub_where_simple_does_not() -> None:
    """The chapter's point: the four .46 edges and the four weight-2 edges are the same edges,
    but the hub's own edges fall from 1 to 1/8 while the small events keep their share."""
    simple = project_incidence(TOY, "simple")
    hyperbolic = project_incidence(TOY, "hyperbolic")
    assert simple[4, 5] == 1.0  # two nodes joined only by the degree-8 node
    assert hyperbolic[4, 5] == pytest.approx(1 / 8)
    assert simple[0, 1] / simple[4, 5] == pytest.approx(2.0)
    assert hyperbolic[0, 1] / hyperbolic[4, 5] == pytest.approx(3.667, abs=1e-3)


# --------------------------------------------------------------------------- §26.4 figure 26.8


def test_figure_26_8_resource_allocation_is_asymmetric_at_point_229_and_point_153() -> None:
    """ "From node 1's perspective the edge weight is (1/2 * 1/3) + (1/2 * 1/8) ... from node 2's
    perspective ... (1/3 * 1/3) + (1/3 * 1/8)" (p. 375)."""
    weights = project_incidence(TOY, "probs")
    assert weights[0, 1] == pytest.approx((1 / 2) * (1 / 3) + (1 / 2) * (1 / 8))
    assert weights[1, 0] == pytest.approx((1 / 3) * (1 / 3) + (1 / 3) * (1 / 8))
    assert f"{weights[0, 1]:.3f}" == "0.229"
    assert f"{weights[1, 0]:.3f}" == "0.153"


def test_resource_is_an_alias_of_probs() -> None:
    """§26.4 names the family "resource allocation" and this member of it "ProbS"."""
    assert project_incidence(TOY, "resource") == pytest.approx(project_incidence(TOY, "probs"))


def test_heats_is_the_transpose_of_probs_and_the_hybrid_interpolates_them() -> None:
    """ "HeatS is the transpose of ProbS" (p. 376); λ=1 is ProbS and λ=0 is HeatS."""
    probs = project_incidence(TOY, "probs")
    assert project_incidence(TOY, "heats") == pytest.approx(probs.T)
    assert project_incidence(TOY, "hybrid", lam=1.0) == pytest.approx(probs)
    assert project_incidence(TOY, "hybrid", lam=0.0) == pytest.approx(probs.T)


def test_the_probs_diagonal_makes_the_two_step_walk_stochastic() -> None:
    """§26.4's "W has a well-defined diagonal": with it, every row of the transition matrix sums
    to 1, which is what makes §26.5's stationary distribution meaningful."""
    left = TOY
    degrees = left.sum(axis=1)
    transition = ((left / left.sum(axis=0)) @ left.T) / degrees[:, None]
    assert transition.sum(axis=1) == pytest.approx(np.ones(8))
    # and the projection zeroes it, as §26.1 and §26.4 both allow
    assert np.diag(project_incidence(TOY, "probs")) == pytest.approx(np.zeros(8))


# --------------------------------------------------------------------------- §26.5 figure 26.9


def test_figure_26_9_random_walk_weights_are_point_049_and_point_022() -> None:
    """``w(u,v) = π_v A(u,v)`` (p. 376), and "the 1->2 edge weight is now more than twice as
    2->1, while in resource allocation it was just about 50% higher" (p. 377)."""
    weights = project_incidence(TOY, "randomwalk")
    assert f"{weights[0, 1]:.3f}" == "0.049"
    assert f"{weights[1, 0]:.3f}" == "0.022"
    assert weights[0, 1] / weights[1, 0] > 2.0
    probs = project_incidence(TOY, "probs")
    assert 1.4 < probs[0, 1] / probs[1, 0] < 1.6


def test_the_stationary_distribution_is_the_dominant_left_eigenvector() -> None:
    """The closed form this package uses -- π proportional to the projected node's degree -- is
    the stationary distribution §26.5 asks for, checked against the eigenvector on this
    connected toy, where the eigenvector is unique."""
    degrees = TOY.sum(axis=1)
    transition = ((TOY / TOY.sum(axis=0)) @ TOY.T) / degrees[:, None]
    values, vectors = np.linalg.eig(transition.T)
    leading = np.real(vectors[:, int(np.argmin(abs(values - 1.0)))])
    assert leading / leading.sum() == pytest.approx(degrees / degrees.sum())


# --------------------------------------------------------------------------- symmetry


def test_the_asymmetric_schemes_are_averaged_and_both_directions_are_kept() -> None:
    """§26.4 p. 376 allows the minimum, the maximum or the average; the average is taken, and
    the directed pair stays on the edge so the averaging is reversible."""
    graph = graph_from_incidence(TOY, list(range(8)), ["z0", "z1", "z2"], scheme="probs")
    data = graph.edges[0, 1]
    forward = (1 / 2) * (1 / 3) + (1 / 2) * (1 / 8)
    backward = (1 / 3) * (1 / 3) + (1 / 3) * (1 / 8)
    assert data["weight_uv"] == pytest.approx(forward)
    assert data["weight_vu"] == pytest.approx(backward)
    assert data["weight"] == pytest.approx((forward + backward) / 2)
    assert set(ASYMMETRIC) == {"resource", "probs", "heats", "hybrid", "randomwalk"}


def test_symmetrise_preserves_the_total_of_the_two_directions() -> None:
    matrix = project_incidence(TOY, "probs")
    averaged = symmetrise(matrix)
    assert averaged == pytest.approx(averaged.T)
    assert averaged.sum() == pytest.approx(matrix.sum())


def test_a_symmetric_scheme_records_no_directed_pair() -> None:
    graph = project([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")], scheme="hyperbolic")
    assert "weight_uv" not in graph.edges["a", "b"]


# --------------------------------------------------------------------------- the default


def test_simple_is_bit_identical_to_the_naive_loop_on_southern_women() -> None:
    """The default's numbers may not move, and its weights stay ``int``: a count is a count."""
    pairs = _southern_women_pairs()
    for side in ("left", "right"):
        expected = _reference_projection(pairs, side)
        got = bipartite_projection(pairs, side=side)  # type: ignore[arg-type]
        assert sorted(got.nodes) == sorted(expected.nodes)
        assert sorted(got.edges) == sorted(expected.edges)
        for u, v, weight in expected.edges(data="weight"):
            assert got.edges[u, v]["weight"] == weight
            assert isinstance(got.edges[u, v]["weight"], int)


def test_simple_is_bit_identical_to_the_naive_loop_on_a_random_bipartite_graph() -> None:
    pairs = _random_bipartite()
    for side in ("left", "right"):
        expected = _reference_projection(pairs, side)
        got = bipartite_projection(pairs, side=side)  # type: ignore[arg-type]
        assert sorted(got.nodes) == sorted(expected.nodes)
        assert {frozenset(e) for e in got.edges} == {frozenset(e) for e in expected.edges}
        for u, v, weight in expected.edges(data="weight"):
            assert got.edges[u, v]["weight"] == weight


def test_a_projection_records_its_scheme_its_lambda_and_its_memberships() -> None:
    pairs = [("a", "x"), ("b", "x")]
    graph = project(pairs, scheme="hybrid", lam=0.25)
    assert graph.graph[PROJECTION] == "hybrid"
    assert graph.graph[PROJECTION_LAMBDA] == 0.25
    assert graph.graph[PAIRS] == sorted(pairs)


# --------------------------------------------------------------------------- Southern women


def test_hyperbolic_and_resource_weights_on_two_named_pairs_of_southern_women() -> None:
    """Hand arithmetic, from the events these women were recorded at and how big those were.

    Evelyn Jefferson and Laura Mandeville share six events, of sizes 3, 3, 6, 8, 8 and 14, and
    attended 8 and 7 in all. Olivia Carleton and Flora Price are the two peripheral women: they
    share both the events either of them attended, of sizes 4 and 12.
    """
    pairs = _southern_women_pairs()
    hyperbolic = bipartite_projection(pairs, side="left", scheme="hyperbolic")
    probs = bipartite_projection(pairs, side="left", scheme="probs")

    evelyn, laura = "Evelyn Jefferson", "Laura Mandeville"
    shared = (
        Fraction(1, 3)
        + Fraction(1, 3)
        + Fraction(1, 6)
        + Fraction(1, 8)
        + Fraction(1, 8)
        + Fraction(1, 14)
    )
    assert shared == Fraction(97, 84)
    assert hyperbolic.edges[evelyn, laura]["weight"] == pytest.approx(float(shared))
    assert probs.edges[evelyn, laura]["weight_uv"] == pytest.approx(float(shared / 8))
    assert probs.edges[evelyn, laura]["weight_vu"] == pytest.approx(float(shared / 7))
    assert probs.edges[evelyn, laura]["weight"] == pytest.approx(
        float((shared / 8 + shared / 7) / 2)
    )

    olivia, flora = "Olivia Carleton", "Flora Price"
    peripheral = Fraction(1, 4) + Fraction(1, 12)
    assert peripheral == Fraction(1, 3)
    assert hyperbolic.edges[olivia, flora]["weight"] == pytest.approx(float(peripheral))
    # Both attended two events, so ProbS is symmetric for them and equal to half the hyperbolic.
    assert probs.edges[olivia, flora]["weight"] == pytest.approx(float(peripheral / 2))


def test_the_schemes_reorder_the_southern_women_by_more_than_arithmetic() -> None:
    """Evelyn/Laura and Olivia/Flora are the point of the chapter: the first pair share three
    times as many events as the second, and the hyperbolic weight puts them three and a half
    times apart -- but under Jaccard, which normalises by the union, the peripheral pair scores
    the maximum and the famous pair does not."""
    pairs = _southern_women_pairs()
    simple = bipartite_projection(pairs, scheme="simple")
    jaccard = bipartite_projection(pairs, scheme="jaccard")
    assert simple.edges["Evelyn Jefferson", "Laura Mandeville"]["weight"] == 6
    assert simple.edges["Olivia Carleton", "Flora Price"]["weight"] == 2
    assert jaccard.edges["Olivia Carleton", "Flora Price"]["weight"] == pytest.approx(1.0)
    assert jaccard.edges["Evelyn Jefferson", "Laura Mandeville"]["weight"] < 1.0


# --------------------------------------------------------------------------- the null


def test_the_bipartite_null_re_projects_under_the_scheme_it_was_given() -> None:
    """A hyperbolic observation compared against simple-weight rewirings would be a comparison
    of two definitions. The sample carries the scheme and its weights are fractions."""
    pairs = _random_bipartite()
    samples = list(bipartite_preserving(pairs, 3, scheme="hyperbolic", seed=26))
    assert len(samples) == 3
    for sample in samples:
        assert sample.graph[PROJECTION] == "hyperbolic"
        weights = [w for _, _, w in sample.edges(data="weight")]
        assert weights
        assert any(not float(w).is_integer() for w in weights)
        assert sample.graph[PAIRS]


def test_the_bipartite_null_keeps_integer_weights_on_the_default_scheme() -> None:
    pairs = _random_bipartite()
    sample = next(iter(bipartite_preserving(pairs, 1, seed=26)))
    assert sample.graph[PROJECTION] == "simple"
    assert all(isinstance(w, int) for _, _, w in sample.edges(data="weight"))


def test_the_bipartite_null_preserves_every_degree_whatever_the_scheme() -> None:
    """§18.1's guarantee does not depend on the weighting: the memberships are what is rewired."""
    pairs = _random_bipartite()
    observed_left = Counter(left for left, _ in set(pairs))
    observed_right = Counter(right for _, right in set(pairs))
    for sample in bipartite_preserving(pairs, 2, scheme="probs", seed=26):
        rewired = sample.graph[PAIRS]
        assert Counter(left for left, _ in rewired) == observed_left
        assert Counter(right for _, right in rewired) == observed_right


# --------------------------------------------------------------------------- §26.6 comparison


def test_the_comparison_runs_every_scheme_over_one_edge_list() -> None:
    comparison = compare_schemes(_random_bipartite(), persona_id="p", network="entities")
    assert comparison.names == SCHEMES
    assert comparison.nodes == 30
    assert comparison.edges > 0
    assert {row.readings for row in comparison.schemes} == {comparison.readings}
    # Both directions of every edge, because four of the schemes are directed in the book.
    assert comparison.readings == 2 * comparison.edges
    assert comparison.memberships == len(set(_random_bipartite()))


def test_the_comparison_reports_an_entropy_below_its_own_maximum() -> None:
    comparison = compare_schemes(_random_bipartite())
    for row in comparison.schemes:
        assert 0.0 <= row.entropy <= row.max_entropy + 1e-9
    # The simple scheme concentrates least here, because most pairs share exactly one node.
    entropies = {row.scheme: row.entropy for row in comparison.schemes}
    assert entropies["randomwalk"] < entropies["simple"]


def test_probs_and_heats_are_one_undirected_network_and_two_directed_ones() -> None:
    """§26.6 compares HeatS against ProbS and finds them anti-correlated (-0.34 on the book's
    Twitter data). That comparison only exists in the directed form: averaging a matrix with its
    transpose gives the same answer for W and for W.T, so the two *graphs* are identical."""
    pairs = _random_bipartite()
    assert symmetrise(project_incidence_of(pairs, "probs")) == pytest.approx(
        symmetrise(project_incidence_of(pairs, "heats"))
    )
    probs = bipartite_projection(pairs, scheme="probs")
    heats = bipartite_projection(pairs, scheme="heats")
    for u, v, weight in probs.edges(data="weight"):
        assert heats.edges[u, v]["weight"] == pytest.approx(weight)
        assert heats.edges[u, v]["weight_uv"] == pytest.approx(probs.edges[u, v]["weight_vu"])
    # The comparison reads the directed matrices, so it can still tell them apart, and it finds
    # them reversed rather than identical.
    comparison = compare_schemes(pairs, schemes=("probs", "heats"))
    agreement = comparison.correlations[("probs", "heats")]
    assert agreement is not None
    assert agreement < 1.0


def test_the_comparison_ranks_schemes_against_each_other_rather_than_valuing_them() -> None:
    """§26.6, figure 26.11. ProbS and HeatS agree with the hyperbolic scheme they normalise, and
    a scheme is perfectly correlated with itself."""
    comparison = compare_schemes(_random_bipartite())
    assert comparison.correlations[("simple", "simple")] == 1.0
    assert (
        comparison.correlations[("probs", "heats")] == comparison.correlations[("heats", "probs")]
    )
    hyperbolic_vs_simple = comparison.correlations[("hyperbolic", "simple")]
    assert hyperbolic_vs_simple is not None
    assert 0.0 < hyperbolic_vs_simple < 1.0


def test_a_scheme_with_no_spread_has_no_rank_to_agree_with() -> None:
    """Every pair here shares exactly one node, so the simple weights are all 1 and Spearman is
    undefined -- reported as ``None`` rather than as a zero correlation."""
    pairs = [("a", "x"), ("b", "x"), ("c", "x"), ("d", "x")]
    comparison = compare_schemes(pairs, schemes=("simple", "hyperbolic"))
    assert comparison.correlations[("simple", "hyperbolic")] is None


def test_the_threshold_keeps_the_same_share_under_every_scheme_and_scores_the_overlap() -> None:
    comparison = compare_schemes(_random_bipartite(), keep_top=0.2)
    reference = comparison.schemes[0]
    assert reference.agreement == 1.0
    for row in comparison.schemes:
        assert row.kept_share >= 0.2 - 1e-9
        assert 0.0 <= row.agreement <= 1.0
    assert any(row.agreement < 1.0 for row in comparison.schemes)


def test_an_absolute_threshold_is_applied_to_every_scheme_as_given() -> None:
    """The cut is the number asked for, and what it keeps is counted against the naive loop."""
    pairs = _random_bipartite()
    comparison = compare_schemes(pairs, schemes=("simple",), threshold=3.0)
    row = comparison.schemes[0]
    heavy = sum(
        1 for _, _, w in _reference_projection(pairs).edges(data="weight") if float(w) >= 3.0
    )
    assert row.cut == 3.0
    # One per direction: the comparison reads ordered pairs, and simple is symmetric.
    assert row.kept == 2 * heavy
    # and the same cut on a fractional scheme keeps nothing, which is the point of naming it
    fractional = compare_schemes(pairs, schemes=("hyperbolic",), threshold=3.0)
    assert fractional.schemes[0].kept == 0


def test_the_comparison_refuses_what_it_cannot_answer() -> None:
    with pytest.raises(ValueError, match="at least one membership"):
        compare_schemes([])
    with pytest.raises(ValueError, match="must be one of"):
        compare_schemes(_random_bipartite(), schemes=("astrology",))
    with pytest.raises(ValueError, match="share of the edges"):
        compare_schemes(_random_bipartite(), keep_top=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        project_incidence(TOY, "hybrid", lam=1.5)


def test_the_report_prints_its_frame_its_n_its_null_and_the_chapter() -> None:
    comparison = compare_schemes(
        _random_bipartite(), persona_id="p", network="entities", frame="Who was recorded."
    )
    report = render_projections(comparison)
    assert "Implements." in report and "26.6" in report
    assert "Who was recorded." in report
    assert "**Null model.** None" in report
    assert f"{comparison.edges:,} edges" in report
    for scheme in SCHEMES:
        assert f"`{scheme}`" in report
    assert "not** a backbone" in report


def test_the_payload_carries_every_row_and_every_correlation() -> None:
    comparison = compare_schemes(_random_bipartite(), schemes=("simple", "hyperbolic"))
    payload = projections_payload(comparison)
    assert payload["implements"] == "26.6"
    assert [row["scheme"] for row in payload["schemes"]] == ["simple", "hyperbolic"]
    assert len(payload["correlations"]) == 4


# --------------------------------------------------------------------------- the topic network


def test_the_topic_network_projects_under_every_scheme(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """The one network whose default path is not a projection still has to project.

    The fixture corpus is three documents of three topics each, and no pair of topics shares two
    documents, so every simple weight is 1 and every document has ``k_z = 3``:

    * ``ben-oduya``: prioritization, retention, roadmap
    * ``ada-north``: onboarding, product market fit, retention
    * ``cleo-vance``: communication, design, onboarding

    Nine pairs, each sharing exactly one document. Hyperbolic therefore gives every one of them
    ``1/3``, and Jaccard gives ``|shared| / |union of the two topics' documents|``, which differs
    per pair. The edge *set* is the same under all three -- a projection joins nodes with common
    neighbours whatever the weighting -- and only the numbers move.

    This is the case that caught a bug. Every fractional weight here is below 1, so a
    ``min_weight`` of 1 applied as ``w >= min_weight`` deleted the whole network; it is a no-op
    at 1 or less now, exactly as ``prune_below`` is.
    """
    simple = topic_co_occurrence(memory_store, "test-pm", min_weight=1)
    hyperbolic = topic_co_occurrence(memory_store, "test-pm", min_weight=1, projection="hyperbolic")
    jaccard = topic_co_occurrence(memory_store, "test-pm", min_weight=1, projection="jaccard")

    assert simple.number_of_edges() == 9
    for graph in (hyperbolic, jaccard):
        assert {frozenset(e) for e in graph.edges} == {frozenset(e) for e in simple.edges}
        assert sorted(graph.nodes) == sorted(simple.nodes)

    assert {w for _, _, w in simple.edges(data="weight")} == {1}
    for _, _, weight in hyperbolic.edges(data="weight"):
        assert weight == pytest.approx(1 / 3)
    # prioritization is in one document, retention in two, and they share that one.
    assert jaccard.edges["prioritization", "retention"]["weight"] == pytest.approx(1 / 2)
    # prioritization and roadmap are in the same single document and no other.
    assert jaccard.edges["prioritization", "roadmap"]["weight"] == pytest.approx(1.0)
    # onboarding is in two documents and retention in two, sharing one of the three.
    assert jaccard.edges["onboarding", "retention"]["weight"] == pytest.approx(1 / 3)

    assert hyperbolic.graph["projection"] == "hyperbolic"
    assert "hyperbolic projection" in hyperbolic.graph["frame"]
    # The unprojected default reads the co-occurrence edges stored at ingestion, so it carries no
    # scheme at all rather than claiming one it did not use.
    assert "projection" not in simple.graph
    assert "projection" not in simple.graph["frame"]


def test_the_topic_network_still_prunes_a_count_threshold(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """The no-op is at 1 or less only: above it the threshold still applies, as it always did."""
    assert topic_co_occurrence(memory_store, "test-pm", min_weight=99).number_of_edges() == 0
    heavy = topic_co_occurrence(memory_store, "test-pm", min_weight=2, projection="hyperbolic")
    assert heavy.number_of_edges() == 0  # every hyperbolic weight here is 1/3
