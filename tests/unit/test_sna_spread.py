"""Chapters 20 and 21's spreading processes, held to answers worked out before the code ran.

Every assertion here comes from one of three places, and none of them is "the function returned
something":

* **Closed forms on graphs whose structure is a formula.** A complete graph K_n is n-1 regular,
  so its adjacency matrix's leading eigenvalue is exactly n-1 and the spectral threshold is
  ``1/(n-1)``; a star with k leaves is bipartite with all the weight on one side, so its leading
  eigenvalue is exactly ``sqrt(k)`` (Wang et al.'s own closed form for a star). An SI run with
  ``beta=1`` and infinite patience is deterministic, so its final size on a connected graph is
  exactly 1 and on a graph in two pieces is exactly the seeded piece's share.
* **Worked examples the book itself gives.** Figure 21.11(a): *"a chain, we only need to control
  the origin of the chain"* -- one driver node, forced by the matching having nowhere else to
  leave unmatched. §21.1's own claim that a threshold model's hub is easy to tip and a cascade
  model's is not, checked on the same star with the same seeds and only the trigger changed.
* **A planted structure whose answer is obvious by construction.** A hub joining several
  otherwise-separate arms: removing the hub (targeted immunisation) severs the arms from each
  other, and removing an arbitrary leaf (random immunisation) barely touches the giant component
  -- the classic "robustness of scale-free networks" asymmetry (§22, foreshadowed here because
  chapter 21 needs it to argue for the acquaintance strategy).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.spread import (
    MAX_SPREAD_NODES,
    build_spread_report,
    driver_nodes,
    epidemic_threshold,
    immunise,
    interventions,
    logistic_curve,
    render_spread,
    reproduction,
    simulate,
    spread_payload,
)

runner = CliRunner()

# ------------------------------------------------------------------- fixtures: planted graphs


@pytest.fixture
def two_triangles() -> nx.Graph:
    """Two triangles with nothing between them: two components, planted."""
    return nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))


@pytest.fixture
def hub_and_spoke() -> nx.Graph:
    """One hub, four arms of three nodes each, joined only through the hub.

    The hub (node 0) is the sole cut vertex between every pair of arms: remove it and the
    network falls into four pieces of three; remove any other single node and at most one arm
    loses its outermost leaf. 13 nodes, 12 edges -- small enough to check by hand.
    """
    graph = nx.Graph()
    node = 1
    for _ in range(4):
        root, mid, leaf = node, node + 1, node + 2
        graph.add_edge(0, root)
        graph.add_edge(root, mid)
        graph.add_edge(mid, leaf)
        node += 3
    return graph


def _giant_component(graph: nx.Graph, remove: tuple[object, ...]) -> int:
    """The largest connected piece left after ``remove`` is taken out, standing in for what an
    immunisation strategy does to a spread: an immune node can never transmit, which is exactly
    what deleting it from the transmission graph would do."""
    remainder = graph.copy()
    remainder.remove_nodes_from(remove)
    if remainder.number_of_nodes() == 0:
        return 0
    return max(len(piece) for piece in nx.connected_components(remainder))


# ----------------------------------------------------------------- §20.2 the epidemic threshold


def test_the_spectral_threshold_of_a_complete_graph_is_one_over_n_minus_one() -> None:
    """K5 is 4-regular, so its adjacency matrix's leading eigenvalue is exactly 4 (the row sums
    are all 4, and the all-ones vector is an eigenvector of a regular graph). The spectral
    threshold is therefore exactly ``1/4``, and because every node has the same degree, the
    heterogeneous form ``k/k^2 = k/k^2 = 1/k`` collapses onto the same number: a regular graph is
    the one case where the two readings of §20.2 agree exactly."""
    limits = epidemic_threshold(nx.complete_graph(5))
    assert limits.leading_eigenvalue == pytest.approx(4.0)
    assert limits.spectral == pytest.approx(0.25)
    assert limits.mean_degree == pytest.approx(4.0)
    assert limits.mean_square_degree == pytest.approx(16.0)
    assert limits.homogeneous == pytest.approx(1.0 / 5.0)
    assert limits.heterogeneous == pytest.approx(0.25)
    assert limits.heterogeneous == pytest.approx(limits.spectral)


def test_the_spectral_threshold_of_a_star_is_one_over_the_square_root_of_its_leaves() -> None:
    """A star with k leaves is the complete bipartite graph K_{1,k}: its adjacency matrix has
    eigenvalues ``+sqrt(k)``, ``-sqrt(k)`` and 0 (multiplicity k-1), so lambda_1 = sqrt(k) exactly
    -- 2 for the 4-leaf star tested here -- and the spectral threshold is ``1/sqrt(k) = 0.5``,
    well below the ``1/(k+1)`` a Gn,p graph of the same mean degree would give (§20.2's point that
    a hub lowers the bar)."""
    limits = epidemic_threshold(nx.star_graph(4))
    assert limits.leading_eigenvalue == pytest.approx(2.0)
    assert limits.spectral == pytest.approx(0.5)
    assert limits.homogeneous < limits.spectral  # the hub makes the exact bar the lower one


def test_a_network_with_no_edges_has_no_threshold_and_says_so() -> None:
    """§20.2's thresholds are properties of how infection can travel; with no edges nothing can,
    so lambda_1 does not exist and neither does anything that depends on it."""
    empty = nx.Graph()
    empty.add_nodes_from([0, 1, 2])
    limits = epidemic_threshold(empty)
    assert limits.leading_eigenvalue is None
    assert limits.spectral is None
    assert limits.heterogeneous is None
    assert limits.homogeneous == pytest.approx(1.0)  # 1/(0 + 1)
    assert "No edges" in limits.reading


def test_reproduction_reads_lambda_against_all_three_thresholds_of_k5() -> None:
    """beta/mu = 2.0 on K5, whose spectral threshold is 0.25: the ratio is 8x over it, so §20.2
    calls this endemic. R0 -- not the book's own number -- is beta(k^2 - k)/(mu k) = 0.2(16-4)/
    (0.1*4) = 6, the standard heterogeneous mean-field form."""
    ratio = reproduction(0.2, 0.1, nx.complete_graph(5))
    assert ratio.ratio == pytest.approx(2.0)
    assert ratio.spectral == pytest.approx(8.0)
    assert ratio.homogeneous == pytest.approx(10.0)
    assert ratio.heterogeneous == pytest.approx(8.0)
    assert ratio.r0 == pytest.approx(6.0)
    assert ratio.endemic is True


def test_reproduction_refuses_mu_zero_because_si_has_no_threshold() -> None:
    """§20.1: SI never recovers, so lambda = beta/mu is not a number and there is no bar to check
    it against -- which is exactly why SI always saturates whatever beta is."""
    with pytest.raises(ValueError, match="SI model has no recovery"):
        reproduction(0.2, 0.0, nx.complete_graph(5))


def test_a_dense_network_is_refused_rather_than_solved_slowly() -> None:
    """:data:`MAX_SPREAD_NODES` is a hard ceiling on both the simulation and the spectral
    threshold, because a step is an ``n x n`` matrix product and the threshold an eigendecomposition
    of that same matrix. Cheap to check: node count alone trips it, no edges required."""
    oversized = nx.Graph()
    oversized.add_nodes_from(range(MAX_SPREAD_NODES + 1))
    with pytest.raises(ValueError, match="above the"):
        simulate(oversized, "si", seeds=[0], steps=1, runs=1)
    with pytest.raises(ValueError, match="above the"):
        epidemic_threshold(nx.cycle_graph(MAX_SPREAD_NODES + 1))


def test_the_logistic_curve_matches_its_own_closed_form_by_hand() -> None:
    """§20.1's closed form (p. 289): ``i(t) = i0 e^(beta k t) / (1 - i0 + i0 e^(beta k t))``.
    Worked by hand for one step at rate = beta*k = 2.0, i0 = 0.1: ``0.1 e^2 / (0.9 + 0.1 e^2) =
    0.4508...``. It starts exactly at i0 and, since the rate is positive, rises monotonically
    towards the ceiling of 1 (everyone eventually infected, homogeneous mixing's own version of
    §20.1's "no matter beta, SI ends up complete")."""
    curve = logistic_curve(0.5, 4.0, i0=0.1, steps=20)
    assert curve[0] == pytest.approx(0.1)
    rate = 0.5 * 4.0
    growth = math.exp(rate)
    assert curve[1] == pytest.approx(0.1 * growth / (0.9 + 0.1 * growth))
    assert curve[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in itertools.pairwise(curve))


# --------------------------------------------------------------- §20.1 SI saturation and pieces


def test_si_with_beta_one_saturates_a_connected_network_completely() -> None:
    """§20.1: *"All SI models, no matter the value of beta, will end up with a complete
    infection"* -- on a connected network. With beta=1 the classical-reinforcement chance is
    ``1 - (1-1)^n = 1`` for any exposed node, so the run is deterministic: every node of K5 is
    infected by the fifth step from any single seed."""
    result = simulate(nx.complete_graph(5), "si", seeds=[0], beta=1.0, steps=5, runs=1, seed=0)
    assert result.final_size == pytest.approx(1.0)
    assert result.final_low == pytest.approx(1.0)
    assert result.final_high == pytest.approx(1.0)
    assert result.died_out == pytest.approx(0.0)  # SI has no recovery, so "died out" never fires


def test_si_never_leaves_the_component_it_was_not_seeded_in(two_triangles: nx.Graph) -> None:
    """A spread has no edges to travel outside its component, so seeding one triangle of two must
    saturate exactly that triangle (3 of 6 nodes, a final size of exactly 0.5) and never touch
    the other -- not 100%, and not less than the seeded triangle either, because beta=1 is
    deterministic complete infection within a connected piece."""
    result = simulate(two_triangles, "si", seeds=[0], beta=1.0, steps=5, runs=1, seed=0)
    assert result.final_size == pytest.approx(0.5)
    assert result.final_low == pytest.approx(0.5)
    assert result.final_high == pytest.approx(0.5)


# ------------------------------------------------------------------ §21.1 complex contagion


def test_a_single_seed_cannot_clear_a_threshold_above_one() -> None:
    """§21.1, figure 21.1: *"any kappa > 1 renders peripheral nodes safe, since most of them have
    only one connection"* -- and the same fact strands the hub too, since one infected leaf is
    only one infected neighbour. On the 4-leaf star, kappa=2 with a single seeded leaf never
    reaches the hub and the outbreak is stuck at its one seed forever."""
    star = nx.star_graph(4)
    result = simulate(
        star, "threshold", seeds=[1], threshold=2.0, beta=1.0, steps=3, runs=1, seed=0
    )
    assert [step.ever for step in result.steps] == pytest.approx([0.2] * 4)


def test_two_seeds_clear_kappa_two_and_the_hub_carries_it_no_further() -> None:
    """With two leaves seeded the hub (degree 4) has exactly kappa=2 infected neighbours and
    triggers -- but the *other* two leaves each have only the hub as a neighbour, one infected
    contact, which is still below kappa=2, so they are never reached. Final ever share: the two
    seeds plus the hub, 3 of 5 nodes."""
    star = nx.star_graph(4)
    result = simulate(
        star, "threshold", seeds=[1, 2], threshold=2.0, beta=1.0, steps=2, runs=1, seed=0
    )
    assert result.steps[0].ever == pytest.approx(2 / 5)
    assert result.steps[-1].ever == pytest.approx(3 / 5)


def test_the_cascade_fraction_is_harder_on_a_hub_than_the_same_number_as_kappa() -> None:
    """§21.1's own contrast (p. 303): *"in a threshold model, hubs are the primary spreaders...
    in a cascade model, they're the last bastion of defense"*. On a 10-leaf star the absolute
    kappa=2 only ever needs 2 infected leaves to tip the hub, whatever the hub's own degree is;
    the fraction 0.5 needs half of *this* hub's ten neighbours -- five -- so the same two seeds
    that tip the kappa model leave the cascade model's hub untouched."""
    star10 = nx.star_graph(10)
    kappa = simulate(
        star10, "threshold", seeds=[1, 2], threshold=2.0, beta=1.0, steps=1, runs=1, seed=0
    )
    cascade = simulate(
        star10, "threshold", seeds=[1, 2], threshold=0.5, beta=1.0, steps=1, runs=1, seed=0
    )
    assert kappa.steps[-1].ever > kappa.steps[0].ever  # the hub tipped
    assert cascade.steps[-1].ever == pytest.approx(cascade.steps[0].ever)  # the hub held


def test_once_a_cascade_hub_tips_every_leaf_follows_in_one_step() -> None:
    """The other half of the same star: seed half its leaves (five of ten) and the hub clears its
    0.5 fraction and is infected at step 1; every remaining leaf then has one infected neighbour
    -- the hub -- against its own floor of ``max(0.5*1, 1) = 1``, so the whole star is infected by
    step 2. A cascade that clears its hub does not stop at the hub."""
    star10 = nx.star_graph(10)
    result = simulate(
        star10, "threshold", seeds=[1, 2, 3, 4, 5], threshold=0.5, beta=1.0, steps=2, runs=1, seed=0
    )
    assert result.steps[0].ever == pytest.approx(5 / 11)
    assert result.steps[1].ever == pytest.approx(6 / 11)  # the hub, at step 1
    assert result.steps[2].ever == pytest.approx(1.0)  # every leaf, at step 2


# -------------------------------------------------------------------- §21.2 limited infection


def test_limited_infection_advances_exactly_one_hop_per_step() -> None:
    """§21.2: a newly infected node gets ``attempts`` steps to persuade its neighbours and then
    stops trying forever. With ``attempts=1`` on a 5-node path, each infection can only reach its
    immediate neighbour before it gives up, so the infected frontier advances by exactly one node
    per step -- never two, however large beta is -- and the run cannot outrun the chain."""
    path = nx.path_graph(5)
    result = simulate(path, "limited", seeds=[0], beta=1.0, attempts=1, steps=5, runs=1, seed=0)
    ever = [step.ever for step in result.steps]
    assert ever == pytest.approx([1 / 5, 2 / 5, 3 / 5, 4 / 5, 1.0, 1.0])
    assert result.steps[-1].infected == pytest.approx(0.0)  # everyone has exhausted their attempts


# ------------------------------------------------------------------------ §21.3 immunisation


def test_targeted_immunisation_breaks_a_hub_and_spoke_network_far_more_than_random(
    hub_and_spoke: nx.Graph,
) -> None:
    """§21.3's argument for targeting hubs, made concrete: removing the one node that joins every
    arm collapses the giant component from 13 nodes to 3 (the largest remaining arm); removing an
    arbitrary node chosen at random (fixed seed, so this is a specific and reproducible draw,
    node 6 -- a degree-1 leaf) barely dents it, down from 13 to 12."""
    share = 1.0 / hub_and_spoke.number_of_nodes()
    targeted = immunise(hub_and_spoke, share, "degree")
    assert targeted == (0,)  # the hub, unambiguously the highest-degree node
    assert _giant_component(hub_and_spoke, targeted) == 3

    random_pick = immunise(hub_and_spoke, share, "random", seed=0)
    assert random_pick == (6,)  # this seed's specific, reproducible draw: a degree-1 leaf
    assert hub_and_spoke.degree(6) == 1
    assert _giant_component(hub_and_spoke, random_pick) == 12

    assert _giant_component(hub_and_spoke, targeted) < _giant_component(hub_and_spoke, random_pick)


def test_the_intervention_report_shows_the_same_asymmetry_on_the_outbreak_itself(
    hub_and_spoke: nx.Graph,
) -> None:
    """The same fact, read through a simulation rather than a bare node count: seeding an outer
    leaf and comparing SIR with no recovery (mu near zero, so nothing that gets infected escapes
    being counted), the targeted strategy's reduction in final size is far larger than the random
    strategy's, because it cuts the one path the infection would otherwise use to reach the other
    three arms."""
    share = 1.0 / hub_and_spoke.number_of_nodes()
    targeted = interventions(
        hub_and_spoke,
        share,
        "degree",
        model="sir",
        seeds=[6],
        beta=1.0,
        mu=0.0001,
        steps=8,
        runs=5,
        seed=1,
    )
    random_run = interventions(
        hub_and_spoke,
        share,
        "random",
        model="sir",
        seeds=[6],
        beta=1.0,
        mu=0.0001,
        steps=8,
        runs=5,
        seed=1,
    )
    assert targeted.reduction > random_run.reduction
    assert targeted.reduction > 0.5  # the hub's removal strands three whole arms
    assert random_run.reduction < 0.2  # a leaf's removal barely touches the rest


def test_immunisation_refuses_a_share_outside_its_range(hub_and_spoke: nx.Graph) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        immunise(hub_and_spoke, 0.0, "random")
    with pytest.raises(ValueError, match="strategy"):
        immunise(hub_and_spoke, 0.1, "unknown")


# -------------------------------------------------------------------------- §21.4 driver nodes


def test_driver_nodes_of_a_directed_chain_is_just_its_origin() -> None:
    """Figure 21.11(a), the book's own example: *"a chain, we only need to control the origin of
    the chain and the rest of the system will fall into place"* (p. 311). A maximum matching of a
    directed path 0->1->2->3 forces (out,0)-(in,1), (out,1)-(in,2), (out,2)-(in,3): every node but
    the origin is matched as a target, so node 0 -- the one node nothing points to -- is the sole
    unmatched target and the sole driver."""
    chain = nx.path_graph(4, create_using=nx.DiGraph)
    drivers = driver_nodes(chain)
    assert drivers.count == 1
    assert drivers.nodes == (0,)
    assert drivers.matching == 3
    assert drivers.directed is True


def test_driver_nodes_of_a_directed_out_star_is_almost_every_leaf() -> None:
    """The opposite extreme: a hub broadcasting to k leaves (edges hub->leaf) has only one source
    copy of the hub, so at most one edge can ever be matched no matter how many leaves it has.
    With 3 leaves that leaves 3 of the 4 nodes undriven -- the hub itself (nothing points to it,
    so its target copy is never even reachable) plus the two leaves the single match could not
    reach."""
    star = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
    drivers = driver_nodes(star)
    assert drivers.matching == 1
    assert drivers.count == 3
    assert 0 in drivers.nodes  # the hub can never be a matched target, whichever edge is chosen
    assert len(drivers.nodes) == 3


def test_an_undirected_network_reads_every_edge_as_running_both_ways() -> None:
    """The same four-node chain, undirected: every edge now yields a bipartite pair in both
    directions, and the resulting graph has a *perfect* matching (every node matched as both
    source and target), so ``N - |M*| = 0``. The module's own floor still applies -- ``max(1, ...)``
    -- so one node is still returned as a driver, but as a placeholder for "something has to be
    driven", not as a node the matching left with no other choice; :attr:`DriverNodes.directed`
    says which reading this was."""
    undirected = nx.path_graph(4)
    drivers = driver_nodes(undirected)
    assert drivers.matching == 4
    assert drivers.count == 1
    assert drivers.directed is False


def test_driver_nodes_of_an_empty_graph_is_zero_of_zero() -> None:
    drivers = driver_nodes(nx.Graph())
    assert drivers == driver_nodes(nx.Graph())
    assert drivers.total == 0
    assert drivers.count == 0
    assert drivers.nodes == ()


# ---------------------------------------------------------------------------- the full report


def test_the_report_prints_n_the_frame_the_null_model_and_the_chapter() -> None:
    """Every report section this package prints carries its sampling frame, its n, its null model
    and the chapter it implements next to the numbers (CLAUDE.md's own rule) -- checked here on
    the markdown ``sna spread`` actually renders."""
    graph = nx.karate_club_graph()
    graph.graph["frame"] = (
        "34 members of one university karate club; an edge is an interaction outside club "
        "activities."
    )
    report = build_spread_report(
        graph,
        persona_id="test-persona",
        network="entities",
        model="sir",
        seeds=[0],
        beta=0.3,
        mu=0.1,
        runs=5,
        steps=10,
        share=0.1,
        strategy="degree",
        seed=1,
    )
    markdown = render_spread(report)
    assert f"n.** {graph.number_of_nodes():,} nodes" in markdown
    assert graph.graph["frame"] in markdown
    assert "Null model.** None" in markdown
    assert "chapter 20" in markdown and "chapter 21" in markdown
    assert "§20.3" in markdown  # the SIR curve section names its own chapter
    assert "§21.3" in markdown  # the immunisation section, since --share was given
    assert "§21.4" in markdown  # the driver-node section, always present

    payload = spread_payload(report)
    assert payload["nodes"] == graph.number_of_nodes()
    assert payload["frame"] == graph.graph["frame"]
    assert "none" in payload["null_model"]
    assert payload["drivers"]["count"] == report.drivers.count
    assert payload["intervention"]["reduction"] == pytest.approx(report.intervention.reduction)


def test_an_empty_network_has_nothing_to_spread_on() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        simulate(nx.Graph(), "si")


# --------------------------------------------------------------------------------- the CLI


def test_sna_spread_prints_the_curve_and_the_driver_section(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The happy path end to end: the three ``test-layers`` entities co-mentioned into a triangle
    (§26.1's simple projection, with ``--min-weight 1`` so the weight-1 edges survive it), a
    deterministic SI run (beta=1) from one named seed, and the sections every report carries --
    the epidemic threshold and the driver-node count -- alongside the curve itself."""
    out = tmp_path / "spread.md"
    payload = tmp_path / "spread.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "spread",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--model",
            "si",
            "--seeds",
            "product:alpha",
            "--beta",
            "1.0",
            "--steps",
            "3",
            "--runs",
            "2",
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## The epidemic threshold (§20.2)" in result.output
    assert "## The curve (§20.1)" in result.output
    assert "## Driver nodes (§21.4)" in result.output
    assert "**Final size.** 100.0%" in result.output  # beta=1 on a connected triangle saturates

    written = out.read_text(encoding="utf-8")
    assert "product:alpha" in written
    body = json.loads(payload.read_text(encoding="utf-8"))
    assert body["nodes"] == 3
    assert body["final_size"] == pytest.approx(1.0)
    assert body["model"] == "si"


def test_sna_spread_rejects_an_unknown_model_and_an_unknown_strategy(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    """Both checks run before the network is even built, so a typo costs nothing but a message."""
    bad_model = runner.invoke(app, ["sna", "spread", "test-layers", "--model", "sird"])
    assert bad_model.exit_code == 2
    assert "--model must be one of" in bad_model.output

    bad_strategy = runner.invoke(app, ["sna", "spread", "test-layers", "--strategy", "vaccine"])
    assert bad_strategy.exit_code == 2
    assert "--strategy must be one of" in bad_strategy.output


def test_sna_spread_seeds_option_takes_a_name_and_reads_a_bare_number_as_a_count(
    cli_context: AppContext, layered: InMemoryGraphStore
) -> None:
    """``_seeds_or_exit``'s two branches, both through the CLI: a comma-separated list of names
    is used unchanged in every run and printed by name; a single numeric token is read as *how
    many* random seeds to draw per run (§20.5), and the report says so instead of naming them,
    since a random draw differs from run to run. An empty ``--seeds`` is refused rather than
    silently defaulting."""
    named = runner.invoke(
        app,
        [
            "sna",
            "spread",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--seeds",
            "product:alpha",
            "--steps",
            "1",
            "--runs",
            "1",
            "--seed",
            "1",
        ],
    )
    assert named.exit_code == 0, named.output
    assert "**Seeds.** product:alpha." in named.output

    counted = runner.invoke(
        app,
        [
            "sna",
            "spread",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--seeds",
            "2",
            "--steps",
            "1",
            "--runs",
            "1",
            "--seed",
            "1",
        ],
    )
    assert counted.exit_code == 0, counted.output
    assert "**Seeds.** 2 node(s) drawn uniformly at random in each run (§20.5)." in counted.output

    empty = runner.invoke(app, ["sna", "spread", "test-layers", "--seeds", " , "])
    assert empty.exit_code == 2, empty.output
    assert "--seeds was empty" in empty.output
