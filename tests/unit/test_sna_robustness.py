"""Chapter 22's failures, held to answers that exist before the simulation runs.

Four kinds of known answer are used here, and none of them is "the function returned something":

* **A structure whose fragility is arithmetic.** A star graph has exactly one node whose removal
  disconnects everything else, so removing it collapses the giant component in one step while a
  random removal almost always takes a leaf and leaves the network standing -- the pair of curves
  §22.1 and §22.2 draw as figures 22.5/22.6, reproduced on a graph small enough to hand-check.
* **A generator whose known property is what §22.1-22.2 predicts.** A Barabasi-Albert network is
  built with :func:`graphrag.sna.generators.barabasi_albert` for its heavy tail, and the targeted
  curve is asserted to lie at or below the random one at every measured fraction -- p. 318's *"the
  effect ... is much more devastating"* made into an inequality that holds point by point rather
  than at one number.
* **A load-capacity cascade traced by hand.** A ten-node path graph's load-degree and
  capacity-tolerance arithmetic is worked out in the docstring of each cascade test; the values
  asserted are the fixed point of that arithmetic, not a property read off a run.
* **A coupling planted so the propagation has one answer.** Two four-node networks share their
  node names (the ``same-id`` coupling of §22.4): the first is a hub-and-spokes graph whose hub is
  a cut vertex, the second a triangle among the spokes plus an isolated hub. Removing the hub
  alone (forced by a fixed seed) leaves the triangle fully connected in the second network *on its
  own*, but the fixed point of §22.4's back-and-forth pulls it down to the same single surviving
  node the first network is left with -- the chapter's *"failures propagate ... back and forth"*
  (p. 322) checked against arithmetic rather than quoted.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graphrag.sna.generators import barabasi_albert
from graphrag.sna.robustness import (
    CASCADE_TOLERANCES,
    Cascade,
    cascade,
    cascade_profile,
    couple,
    heaviest_node,
    interdependent,
    interdependent_curve,
    molloy_reed,
    removal_curve,
    render_robustness,
    robustness_payload,
    robustness_report,
    strategies_for,
)
from graphrag.sna.robustness import attack_summary as _attack_summary

# ------------------------------------------------------------------- §22.1 the critical fraction


def test_a_cycle_sits_exactly_on_the_giant_component_boundary() -> None:
    """Every node of a cycle has degree 2, so kappa = <k^2>/<k> = 4/2 = 2.0 exactly (§22.1).

    The criterion is kappa > 2, strictly, so a network sitting *on* the boundary has no critical
    fraction to report -- it is already at the edge of holding a giant component together, and
    "the fraction that breaks it" is not a number a random removal needs to reach.
    """
    cycle = nx.cycle_graph(10)
    result = molloy_reed(cycle)
    assert result.mean_degree == pytest.approx(2.0)
    assert result.mean_square_degree == pytest.approx(4.0)
    assert result.kappa == pytest.approx(2.0)
    assert result.has_giant_criterion is False
    assert result.critical_fraction is None
    assert "already under the criterion" in result.reading


def test_a_complete_graph_gives_the_closed_form_its_textbook_case() -> None:
    """K5: every node has degree 4, so kappa = 16/4 = 4 and f_c = 1 - 1/(4 - 1) = 2/3 (§22.1)."""
    complete = nx.complete_graph(5)
    result = molloy_reed(complete)
    assert result.kappa == pytest.approx(4.0)
    assert result.has_giant_criterion is True
    assert result.critical_fraction == pytest.approx(2 / 3)


def test_molloy_reed_is_undefined_on_an_edgeless_network() -> None:
    empty = nx.Graph()
    empty.add_nodes_from(["a", "b", "c"])
    result = molloy_reed(empty)
    assert result.mean_degree == 0.0
    assert result.kappa == 0.0
    assert result.critical_fraction is None
    assert "nothing to lose" in result.reading


# ---------------------------------------------------------------- §22.1-22.2 the removal curves


def _star(leaves: int) -> nx.Graph:
    """A hub with ``leaves`` spokes: ``networkx``'s own star, n = leaves + 1."""
    return nx.star_graph(leaves)


def test_a_star_collapses_after_one_targeted_removal_but_survives_random_in_expectation() -> None:
    """The chapter's two curves (§22.1, §22.2), on the one graph where they can be hand-checked.

    A star of 21 nodes (hub 0, leaves 1..20) has exactly one node whose removal disconnects
    everything else: take the hub and the survivors are 20 isolated leaves, so the giant
    component holds one twentieth of what is left -- past the collapse criterion (under half of
    the survivors) at the very first step, ``1/21`` removed. Random removal takes the hub with
    probability ``1/21`` and a leaf with probability ``20/21``; a leaf's removal leaves the hub
    and the 19 remaining leaves in one piece, so in expectation over many independent orders the
    giant component still holds essentially everything at that same fraction removed
    (``20/21 * 1.0 + 1/21 * 1/20 ~= 0.955`` of the survivors) -- nowhere near collapse.
    """
    star = _star(20)
    n = star.number_of_nodes()
    assert n == 21

    targeted = removal_curve(star, "degree", steps=n, runs=1, seed=0)
    random_curve = removal_curve(star, "random", steps=n, runs=300, seed=42)

    one_removed = 1 / n
    assert targeted.collapse_fraction == pytest.approx(one_removed)
    targeted_point = targeted.at(one_removed)
    assert targeted_point.giant_of_survivors == pytest.approx(0.05, abs=1e-9)

    random_point = random_curve.at(one_removed)
    assert random_point.giant_of_survivors > 0.9, (
        "300 independent random orders should average close to the 0.955 expectation, well "
        "above the collapse threshold the targeted curve already crossed"
    )
    # The random curve's own collapse, if it has one at all, comes much later than the targeted
    # curve's -- the whole point of §22.2's comparison.
    assert random_curve.collapse_fraction is None or random_curve.collapse_fraction > 0.5


def test_targeted_removal_never_rises_above_random_on_a_scale_free_network() -> None:
    """A Barabasi-Albert network's targeted curve lies at or below the random one throughout.

    p. 318: *"in a power law degree distribution network... removing even a single node brings
    down the GCC size by almost 20%"* against the G(n,p) case where *"targeted attacks don't
    change the scenario much"*. The generator's own heavy tail is what produces the gap; the
    assertion is the inequality figures 22.5/22.6 draw, checked at every measured fraction rather
    than read off one point of the plot.
    """
    graph = barabasi_albert(300, 2, seed=7)
    targeted = removal_curve(graph, "degree", steps=20, runs=1, seed=0)
    random_curve = removal_curve(graph, "random", steps=20, runs=40, seed=42)

    for target_point, random_point in zip(targeted.points, random_curve.points, strict=True):
        assert target_point.fraction == random_point.fraction
        assert target_point.giant <= random_point.giant + 1e-9
        # Strictly below everywhere except the two ends, where both curves agree by
        # construction: nothing removed yet (both 1.0) or nothing left at all (both 0.0).
        if 0 < target_point.fraction < 1:
            assert target_point.giant < random_point.giant

    assert targeted.area < random_curve.area


def test_recompute_gives_a_curve_at_least_as_steep_as_the_frozen_ranking() -> None:
    """§22.2's stronger attacker: refreshing the ranking after each step is never gentler.

    On the star, the hub is removed first either way -- it is the unique maximum both before and
    after nothing has changed yet -- so the two curves agree at the first step and can only
    diverge, never favour the frozen ranking, once the leaves start tying for second place.
    """
    star = _star(20)
    frozen = removal_curve(star, "degree", steps=star.number_of_nodes(), runs=1, seed=0)
    refreshed = removal_curve(
        star, "degree", recompute=True, steps=star.number_of_nodes(), runs=1, seed=0
    )
    assert frozen.at(1 / 21).giant == pytest.approx(refreshed.at(1 / 21).giant)
    assert refreshed.area <= frozen.area + 1e-9


def test_removal_curve_refuses_bad_arguments() -> None:
    star = _star(5)
    with pytest.raises(ValueError, match="strategy must be one of"):
        removal_curve(star, "not-a-strategy")
    with pytest.raises(ValueError, match="steps must be at least 1"):
        removal_curve(star, "random", steps=0)
    with pytest.raises(ValueError, match="runs must be at least 1"):
        removal_curve(star, "random", runs=0)
    with pytest.raises(ValueError, match="nothing to remove"):
        removal_curve(nx.Graph(), "random")


def test_strategies_for_offers_random_plus_every_centrality_this_graph_supports() -> None:
    star = _star(5)
    allowed = strategies_for(star)
    assert allowed[0] == "random"
    assert "degree" in allowed and "betweenness" in allowed


# --------------------------------------------------------------------- §22.2 the attack summary


def test_attack_summary_names_degree_the_cheapest_attack_on_a_star() -> None:
    """Random beside every targeted strategy (§22.2): on a star, degree is unbeatable.

    Betweenness and pagerank both rank the hub first too on a star -- there is only one node with
    any structural importance at all -- so every strategy here collapses the network at the same
    fraction, and the comparison's job is to say that plainly rather than pick an arbitrary winner.
    """
    star = _star(20)
    summary = _attack_summary(star, strategies=("degree", "betweenness"), steps=21, runs=5, seed=3)
    assert summary.worst in {"degree", "betweenness"}
    assert summary.random_curve.strategy == "random"
    row_by_strategy = {row.strategy: row for row in summary.rows}
    assert row_by_strategy["degree"].collapse_fraction == pytest.approx(1 / 21)
    assert "cheapest attack here is" in summary.verdict


def test_attack_summary_refuses_an_empty_strategy_list() -> None:
    with pytest.raises(ValueError, match="at least one targeted strategy"):
        _attack_summary(_star(5), strategies=())


# ------------------------------------------------------------------------------ §22.3 the cascade


def _chain(nodes: int) -> nx.Graph:
    return nx.path_graph(nodes)


def test_a_zero_tolerance_cascade_takes_every_node_of_a_chain() -> None:
    """§22.3's chain reaction with no slack at all reaches every node of a ten-node path.

    Interior nodes have degree 2 and capacity ``(1 + 0) * 2 = 2``; the seed (node 4, interior)
    fails and splits its load of 2 evenly between its two neighbours, giving each ``2 + 1 = 3 > 2``
    -- both fail in turn, and each of *those* forwards its accumulated load to its one remaining
    neighbour, which is already carrying its own initial load and so exceeds capacity again. The
    load only grows monotonically as it is forwarded (nothing is ever split more than once,
    because after the first round every failing node but the two ends has exactly one surviving
    neighbour left), so the chain reaction runs off both ends of the path and nothing survives.
    """
    chain = _chain(10)
    result = cascade(chain, tolerance=0.0, seeds=[4], load="degree", seed=0)
    assert result.nodes == 10
    assert result.failed == 10
    assert result.surviving == pytest.approx(0.0)
    assert result.giant_share == pytest.approx(0.0)
    assert result.branching == pytest.approx(0.9)  # (10 failed - 1 seed) / 10 failed


def test_a_generous_tolerance_cascade_reaches_nobody_past_the_seed() -> None:
    """With capacity at 3x the initial load, the one node the seed can reach never tips over.

    Node 4 (degree 2, an interior node) fails and gives each of its two neighbours one extra unit
    of load; a neighbour's own initial load is its degree, so a degree-2 neighbour reaches
    ``2 + 1 = 3``, still under a capacity of ``(1 + 2.0) * 2 = 6``. The cascade is the seed alone.
    """
    chain = _chain(10)
    result = cascade(chain, tolerance=2.0, seeds=[4], load="degree", seed=0)
    assert result.failed == 1
    assert result.surviving == pytest.approx(0.9)
    assert result.rounds == 1
    assert result.branching == pytest.approx(0.0)
    assert result.lost_load == pytest.approx(0.0)


def test_cascade_profile_sweeps_tolerance_from_total_collapse_to_none_at_all() -> None:
    """The book's own tolerances (§22.3): the sweep is the finding, not any one row.

    Survival is monotonically non-decreasing in tolerance -- more slack can never make a cascade
    spread *further* -- and the two ends of :data:`CASCADE_TOLERANCES` are the two extremes this
    module names: no slack takes the whole chain, ample slack takes only the seed.
    """
    chain = _chain(10)
    profile = cascade_profile(
        chain, tolerances=CASCADE_TOLERANCES, seeds=[4], load="degree", seed=0
    )
    survivals = [run.surviving for run in profile]
    assert survivals == sorted(survivals)
    assert profile[0].tolerance == 0.0
    assert profile[0].surviving == pytest.approx(0.0)
    assert profile[-1].tolerance == CASCADE_TOLERANCES[-1]
    assert profile[-1].surviving == pytest.approx(0.9)


def test_heaviest_node_under_degree_load_is_the_stars_hub() -> None:
    star = _star(20)
    assert heaviest_node(star, "degree") == 0


def test_cascade_refuses_bad_arguments() -> None:
    chain = _chain(5)
    with pytest.raises(ValueError, match="load must be one of"):
        cascade(chain, load="traffic")
    with pytest.raises(ValueError, match="tolerance must be at least 0"):
        cascade(chain, tolerance=-0.1)
    with pytest.raises(ValueError, match="seeds must be between 1"):
        cascade(chain, seeds=99)
    with pytest.raises(ValueError, match="not in the network"):
        cascade(chain, seeds=["not-a-node"])
    with pytest.raises(ValueError, match="nothing to fail"):
        cascade(nx.Graph())


# ------------------------------------------------------------------- §22.4 interdependent networks


def _hub_and_spokes() -> nx.Graph:
    """Layer A: a hub whose three spokes have no other tie -- the hub is a cut vertex."""
    graph = nx.Graph()
    graph.add_edges_from([("hub", "c1"), ("hub", "c2"), ("hub", "c3")])
    return graph


def _triangle_with_an_idle_hub() -> nx.Graph:
    """Layer B: the same four names, but wired so the hub is not needed at all.

    The spokes form a complete triangle among themselves; ``hub`` is present (so the coupling has
    somewhere to send it) but isolated, so removing it costs layer B nothing on its own.
    """
    graph = nx.Graph()
    graph.add_edges_from([("c1", "c2"), ("c2", "c3"), ("c1", "c3")])
    graph.add_node("hub")
    return graph


def test_couple_same_id_matches_the_shared_names() -> None:
    mapping = couple(_hub_and_spokes(), _triangle_with_an_idle_hub(), "same-id")
    assert mapping == {"c1": "c1", "c2": "c2", "c3": "c3", "hub": "hub"}


def test_couple_random_and_degree_are_bijections_over_the_smaller_side() -> None:
    left = nx.Graph()
    left.add_nodes_from(["x1", "x2", "x3"])
    right = nx.Graph()
    right.add_nodes_from(["y1", "y2"])

    random_mapping = couple(left, right, "random", seed=1)
    assert len(random_mapping) == 2
    assert len(set(random_mapping.values())) == 2  # a bijection, not a many-to-one match

    degree_mapping = couple(left, right, "degree", seed=1)
    # every node here has degree 0, so ties are broken by name: x1/x2 to y1/y2 in that order
    assert degree_mapping == {"x1": "y1", "x2": "y2"}


def test_couple_refuses_an_unknown_coupling() -> None:
    with pytest.raises(ValueError, match="coupling must be one of"):
        couple(_hub_and_spokes(), _triangle_with_an_idle_hub(), "nearest")


def test_removing_the_cut_vertex_in_one_layer_drags_its_dependant_down_in_the_other() -> None:
    """§22.4's back-and-forth, forced onto one deterministic outcome (p. 322).

    ``seed=1`` is chosen because it shuffles the coupled order so that ``hub`` lands first; with
    ``fraction_removed=0.25`` and 4 coupled nodes, exactly one node -- the hub -- is knocked out
    at t=0. In layer A that isolates the three spokes from each other (the hub was their only
    tie), so A's own giant component is just one of them: ``alone_a == 1/4``. Layer B, left to
    itself with the same node missing, still has its triangle intact: ``alone_b == 3/4`` -- a
    layer that would survive this exact shock perfectly well on its own. But coupled, layer A's
    fragmentation is what the iteration starts from: the two remaining triangle members that
    layer A cannot keep in one component drop out of B too, once the fixed point is reached, so
    the *mutual* giant component is only the one node both layers can still agree on --
    ``mutual_share == alone_a == 1/4``, far below what B alone could offer. That gap is exactly
    the chapter's claim: *"if you were to calculate the critical |R| value for each layer
    separately you would obtain a result much higher than the one for the interdependent network
    as a whole"* (p. 323).
    """
    layer_a = _hub_and_spokes()
    layer_b = _triangle_with_an_idle_hub()

    result = interdependent(layer_a, layer_b, "same-id", fraction_removed=0.25, seed=1)

    assert result.coupled == 4
    assert result.removed == 1
    assert result.alone_a == pytest.approx(0.25)
    assert result.alone_b == pytest.approx(0.75)
    assert result.mutual_share == pytest.approx(0.25)
    assert result.mutual_share == pytest.approx(result.alone_a)
    assert result.mutual_share < result.alone_b, (
        "layer B on its own survives this exact shock at 0.75; coupled to layer A, whose "
        "structure it does not share, it is pulled all the way down to A's own 0.25"
    )


def test_interdependent_curve_finds_the_same_collapse_the_single_run_does() -> None:
    """The sweep (§22.4) over the same two layers: the cliff sits where the single run found it.

    At ``fraction=0`` nothing has been removed yet and the mutual component already holds only
    one node in four (the hub bridges nothing to itself), which is below the collapse share from
    the very first point on the curve -- the coupling here has no healthy starting point to lose.
    """
    layer_a = _hub_and_spokes()
    layer_b = _triangle_with_an_idle_hub()
    curve = interdependent_curve(layer_a, layer_b, "same-id", steps=4, runs=1, seed=1)

    assert curve.coupled == 4
    zero_point = curve.points[0]
    assert zero_point.fraction == 0.0
    assert zero_point.mutual == pytest.approx(0.25)
    assert zero_point.alone_b == pytest.approx(0.75)
    assert "already" in curve.verdict or "no collapse here to report" in curve.verdict


def test_interdependent_refuses_a_fraction_outside_zero_one_and_an_empty_coupling() -> None:
    layer_a = _hub_and_spokes()
    layer_b = _triangle_with_an_idle_hub()
    with pytest.raises(ValueError, match="between 0 and 1"):
        interdependent(layer_a, layer_b, fraction_removed=1.5)

    disjoint = nx.Graph()
    disjoint.add_nodes_from(["only-in-b"])
    with pytest.raises(ValueError, match="matched no node"):
        interdependent(layer_a, disjoint, "same-id", fraction_removed=0.1)


# ------------------------------------------------------------------------------- the full report


def test_the_report_prints_the_frame_the_n_the_null_model_and_the_chapter() -> None:
    """Every discipline the programme requires of a section, in one run (§22.1-22.4)."""
    star = _star(20)
    star.graph["frame"] = "a planted star, built for this test"

    report = robustness_report(
        star,
        network="entities",
        persona_id="test-pm",
        strategies=("degree",),
        steps=21,
        runs=3,
        seed=1,
        cascades=True,
        tolerances=(0.0, 1.0),
        load="degree",
    )
    text = render_robustness(report)

    assert "Atlas ch. 22" in text
    assert "a planted star, built for this test" in text
    assert f"{report.nodes:,} nodes" in text
    assert "Null model.** Random removal" in text
    assert "## Critical fraction (§22.1)" in text
    assert "## Removal curves" in text
    assert "## Cascade (§22.3)" in text
    assert "about anything breaking" in text

    payload = robustness_payload(report)
    assert payload["chapter"] == 22
    assert payload["nodes"] == 21
    assert payload["criterion"]["section"] == "§22.1"
    assert payload["attack"]["section"] == "§22.2"
    assert payload["cascades"][0]["section"] == "§22.3"
    assert payload["interdependent"] is None


def test_the_report_carries_the_interdependent_section_when_coupled() -> None:
    layer_a = _hub_and_spokes()
    layer_b = _triangle_with_an_idle_hub()

    report = robustness_report(
        layer_a,
        network="speakers",
        persona_id="test-pm",
        strategies=("degree",),
        steps=4,
        runs=2,
        seed=1,
        other=layer_b,
        coupling="same-id",
        coupled_network="speakers-entities",
    )
    text = render_robustness(report)
    assert "## Interdependent networks (§22.4)" in text
    assert "speakers-entities" in text

    payload = robustness_payload(report)
    assert payload["interdependent"]["section"] == "§22.4"
    assert payload["interdependent"]["network"] == "speakers-entities"
    assert payload["interdependent"]["coupled"] == 4


def test_cascade_dataclass_is_frozen_and_the_module_exports_the_right_names() -> None:
    """A smoke check that the dataclasses this module hands out are what callers rely on."""
    result = cascade(_chain(5), tolerance=0.0, seeds=[2], load="degree", seed=0)
    assert isinstance(result, Cascade)
    with pytest.raises(AttributeError):
        result.failed = 0  # type: ignore[misc]
