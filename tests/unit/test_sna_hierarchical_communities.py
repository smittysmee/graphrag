"""Chapter 37, held to answers that existed before the code did.

Every dendrogram here is cut by :func:`graphrag.sna.evaluate.evaluate_partition`, per the wave's
own rule that a partition is not reported until it has been through that battery, and every
number comes from one of these places:

* **A legendary graph with a published defect.** Zachary's karate club (§53.4, ``tests.legendary``)
  is where ``tests/unit/test_sna_evaluate.py`` already establishes that the highest-modularity
  two-way split of this graph trades two members against the true factions, one of them node 8 --
  the member Zachary explains himself, three weeks from a black belt test he could only take with
  the instructor he structurally looks like he opposed. Girvan-Newman's own first split is a
  second, independent way of reaching a two-way partition of the same graph, and it trades a
  different pair -- nodes 2 and 8 -- which is asserted here rather than assumed, with node 8
  common to both because it is the node the literature keeps landing on.
* **A small ring of cliques**, sized so the resolution limit (§36.1) does not bite: unlike the
  large ring the evaluation rules quote, where merging neighbouring cliques scores higher
  modularity than the cliques themselves, six five-node cliques joined by one edge each have a
  single Louvain level, and it is the clique partition.
* **A planted directed two-block graph** whose blocks are cyclic rings, cross-joined one way
  only (every node of one block points to every node of the other, and nothing points back).
  Flattening it first fuses the blocks into three mixed communities; reading the arrows keeps
  them apart into four block-pure ones.
* **A planted two-level hierarchy**: two groups, each two four-cliques, joined within a group by
  a moderate random density and between groups by a sparse one, with the seed fixed so the
  planted structure is what a reader can regenerate. The group level is checked by exact
  partition; the clique level, which the book itself says has no single dendrogram that is more
  correct than another equally likely one (p. 538), is checked by the *ordering* the fitted
  probabilities put on it instead of by an exact cut.
* **An arithmetic identity.** A complete graph split in half has every possible internal edge
  (density 1) and close to the modularity of doing nothing at all, which is Figure 37.11(d)'s
  disagreement between density and modularity computed rather than illustrated.
"""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from graphrag.sna.cluster import (
    Dendrogram,
    directed_communities,
    girvan_newman_dendrogram,
    hrg_communities,
    hrg_fit,
    louvain,
    louvain_levels,
    score_level,
)
from graphrag.sna.evaluate import evaluate_partition, render_evaluation
from tests.legendary import karate_club, unweighted


@pytest.fixture
def karate() -> nx.Graph:
    graph, _ = karate_club()
    return unweighted(graph)


def _factions(graph: nx.Graph) -> dict[str, set[int]]:
    return {
        club: {n for n, data in graph.nodes(data=True) if data["club"] == club}
        for club in ("Mr. Hi", "Officer")
    }


def _ring_of_cliques(n_cliques: int, clique_size: int) -> nx.Graph:
    """§37.4's own example, sized so the resolution limit does not bite (see module docstring)."""
    graph = nx.Graph()
    for c in range(n_cliques):
        members = [f"c{c}n{i}" for i in range(clique_size)]
        for a, b in itertools.combinations(members, 2):
            graph.add_edge(a, b, weight=1)
    for c in range(n_cliques):
        graph.add_edge(f"c{c}n0", f"c{(c + 1) % n_cliques}n1", weight=1)
    return graph


def _planted_directed_blocks(size: int = 8) -> tuple[nx.DiGraph, list[str], list[str]]:
    """Two directed rings, cross-joined one way only: A always points to B, never back."""
    a = [f"a{i}" for i in range(size)]
    b = [f"b{i}" for i in range(size)]
    graph = nx.DiGraph()
    for block in (a, b):
        for i in range(size):
            graph.add_edge(block[i], block[(i + 1) % size], weight=1)
    for u in a:
        for v in b:
            graph.add_edge(u, v, weight=1)
    return graph, a, b


def _planted_hierarchy(
    seed: int = 1, clique: int = 4, p_within: float = 0.15, p_between: float = 0.01
) -> tuple[nx.Graph, list[list[list[str]]]]:
    """Two groups of two four-cliques each: dense inside a clique, thinner inside a group,
    thinner still between groups. ``groups[g][c]`` is the member list of clique ``c`` of group
    ``g``."""
    rng = random.Random(seed)
    graph = nx.Graph()
    groups: list[list[list[str]]] = []
    for g in range(2):
        cliques = []
        for c in range(2):
            members = [f"g{g}c{c}n{i}" for i in range(clique)]
            cliques.append(members)
            for a, b in itertools.combinations(members, 2):
                graph.add_edge(a, b, weight=1)
        for a in cliques[0]:
            for b in cliques[1]:
                if rng.random() < p_within:
                    graph.add_edge(a, b, weight=1)
        groups.append(cliques)
    for a in groups[0][0] + groups[0][1]:
        for b in groups[1][0] + groups[1][1]:
            if rng.random() < p_between:
                graph.add_edge(a, b, weight=1)
    return graph, groups


# --------------------------------------------------------------------------- §37.1 Louvain levels


def test_louvain_levels_on_a_small_ring_of_cliques_cuts_at_the_clique_level() -> None:
    ring = _ring_of_cliques(6, 5)
    dendrogram = louvain_levels(ring, seed=1)
    assert isinstance(dendrogram, Dendrogram)
    # A small ring does not trip the resolution limit (§36.1): Louvain finds one level, and it
    # is exactly the six cliques.
    assert len(dendrogram.levels) == 1
    assert sorted(len(c) for c in dendrogram.communities) == [5] * 6
    assert dendrogram.cut == 0
    assert dendrogram.peaks == [0]

    evaluation = evaluate_partition(ring, dendrogram.communities, link_prediction=False)
    text = "\n".join(render_evaluation(evaluation))
    assert "## Community evaluation" in text
    assert evaluation.communities == 6


def test_louvain_levels_on_an_empty_graph_returns_nothing() -> None:
    dendrogram = louvain_levels(nx.Graph())
    assert dendrogram.levels == []
    assert dendrogram.communities == []


def test_louvain_levels_flattens_a_directed_network_and_says_so() -> None:
    graph, a_block, b_block = _planted_directed_blocks(4)
    dendrogram = louvain_levels(graph, seed=1)
    assert "ran" in dendrogram.note
    covered = {node for level in dendrogram.levels for c in level.communities for node in c}
    assert covered == set(a_block) | set(b_block)


# ----------------------------------------------------------------------- §37.1 Girvan-Newman


def test_girvan_newman_first_split_on_karate_trades_two_members_including_node_8(
    karate: nx.Graph,
) -> None:
    """The raw first split, not the peak-modularity cut: §37.1's splitting dendrogram before
    anyone has read a modularity profile off it (p. 534-536).

    ``tests/unit/test_sna_evaluate.py`` already established that the single highest-modularity
    two-way split of this graph (found by greedy modularity maximisation) trades nodes 8 and 9
    against Zachary's factions. Girvan-Newman's edge-betweenness splitting is a different
    algorithm reaching a different two-way partition, and it trades a different pair -- 2 and 8
    -- with node 8 in both, because that is the node the literature keeps landing on: Zachary
    explains it himself (member 9, three weeks from a black belt test only the instructor could
    give).
    """
    dendrogram = girvan_newman_dendrogram(karate)
    two_way = next(level for level in dendrogram.levels if level.size == 2)
    mr_hi_side = next(c for c in two_way.communities if 0 in c)
    officer_side = next(c for c in two_way.communities if c is not mr_hi_side)

    factions = _factions(karate)
    assert set(mr_hi_side) ^ factions["Mr. Hi"] == {2, 8}
    assert set(officer_side) ^ factions["Officer"] == {2, 8}
    assert 8 in officer_side  # node 8 lands on the wrong side of its own recorded club

    evaluation = evaluate_partition(karate, two_way.communities, link_prediction=False)
    assert evaluation.communities == 2
    assert "## Community evaluation" in "\n".join(render_evaluation(evaluation))


def test_girvan_newman_dendrogram_reaches_every_node_alone(karate: nx.Graph) -> None:
    dendrogram = girvan_newman_dendrogram(karate)
    assert dendrogram.levels[0].size == 1  # everyone together, modularity 0 by construction
    assert dendrogram.levels[0].modularity == 0.0
    assert dendrogram.levels[-1].size == karate.number_of_nodes()  # every node alone
    # the peak the book says to cut at is not the very first or the very last level
    assert 0 < dendrogram.cut < len(dendrogram.levels) - 1


def test_girvan_newman_refuses_a_network_above_its_own_size_guard() -> None:
    big = nx.gnm_random_graph(50, 100, seed=1)
    with pytest.raises(ValueError, match="trivially small"):
        girvan_newman_dendrogram(big, max_nodes=40)


def test_girvan_newman_on_an_empty_graph_returns_nothing() -> None:
    assert girvan_newman_dendrogram(nx.Graph()).levels == []


# ------------------------------------------------------------------------------- §37.3 directed


def test_directed_communities_separates_what_flattening_merges() -> None:
    graph, a_block, b_block = _planted_directed_blocks(8)
    directed = directed_communities(graph, seed=1, runs=5)
    for community in directed.communities:
        members = set(community)
        # every directed community is pure: it never mixes an 'a' with a 'b'
        assert members <= set(a_block) or members <= set(b_block)
    assert "directed form" in directed.note

    flattened = louvain(graph, seed=1, runs=5)  # louvain() flattens a directed graph first
    assert any(
        set(community) & set(a_block) and set(community) & set(b_block)
        for community in flattened.communities
    )

    evaluation = evaluate_partition(graph, directed.communities, link_prediction=False)
    assert "## Community evaluation" in "\n".join(render_evaluation(evaluation))


def test_directed_communities_rejects_an_undirected_graph() -> None:
    with pytest.raises(ValueError, match="needs a directed graph"):
        directed_communities(nx.Graph([("a", "b")]))


def test_directed_communities_on_an_empty_graph_returns_nothing() -> None:
    result = directed_communities(nx.DiGraph())
    assert result.communities == []


# ------------------------------------------------------------------------------------ §37.2 HRG


def test_hrg_fit_recovers_the_group_level_of_a_planted_hierarchy() -> None:
    graph, groups = _planted_hierarchy(seed=1)
    fit = hrg_fit(graph, samples=8000, restarts=5, seed=1)
    communities = hrg_communities(fit)  # default threshold: denser than the network overall
    found = sorted(sorted(c) for c in communities)
    expected = sorted(sorted(g[0] + g[1]) for g in groups)
    assert found == expected


def test_hrg_fit_nests_cliques_inside_groups_by_probability_even_without_a_clean_cut() -> None:
    """§37.2, p. 538: more than one dendrogram can be equally likely, so the finer level is
    checked by the *ordering* its probabilities put on the hierarchy rather than by a single
    threshold that must produce exactly four communities of four."""
    graph, groups = _planted_hierarchy(seed=1)
    fit = hrg_fit(graph, samples=8000, restarts=5, seed=1)

    parent: dict[int, int] = {}
    for node, (left, right) in fit.children.items():
        parent[left] = node
        parent[right] = node

    def ancestors(node: int) -> list[int]:
        chain = [node]
        while parent.get(chain[-1], -1) != -1:
            chain.append(parent[chain[-1]])
        return chain

    def lca_probability(a: str, b: str) -> float:
        index = {name: i for i, name in enumerate(fit.order)}
        chain_a = ancestors(index[a])
        chain_b = set(ancestors(index[b]))
        node = next(n for n in chain_a if n in chain_b)
        return fit.probabilities[node]

    same_clique = [
        lca_probability(a, b)
        for group in groups
        for clique in group
        for a, b in itertools.combinations(clique, 2)
    ]
    same_group_different_clique = [
        lca_probability(a, b) for group in groups for a in group[0] for b in group[1]
    ]
    different_group = [
        lca_probability(a, b)
        for a in groups[0][0] + groups[0][1]
        for b in groups[1][0] + groups[1][1]
    ]

    mean = lambda values: sum(values) / len(values)  # noqa: E731
    assert mean(same_clique) > mean(same_group_different_clique) > mean(different_group)


def test_hrg_communities_default_threshold_is_the_root_probability() -> None:
    graph, _ = _planted_hierarchy(seed=1)
    fit = hrg_fit(graph, samples=4000, restarts=3, seed=2)
    default = hrg_communities(fit)
    explicit = hrg_communities(fit, threshold=fit.probabilities[fit.root])
    assert default == explicit


def test_hrg_communities_go_through_evaluate_partition() -> None:
    graph, _ = _planted_hierarchy(seed=1)
    fit = hrg_fit(graph, samples=4000, restarts=3, seed=3)
    communities = hrg_communities(fit)
    evaluation = evaluate_partition(graph, communities, link_prediction=False)
    assert "## Community evaluation" in "\n".join(render_evaluation(evaluation))


def test_hrg_fit_rejects_fewer_than_two_nodes() -> None:
    graph = nx.Graph()
    graph.add_node("only")
    with pytest.raises(ValueError, match="at least two nodes"):
        hrg_fit(graph)


def test_hrg_fit_is_reproducible_from_a_seed() -> None:
    graph, _ = _planted_hierarchy(seed=1)
    first = hrg_fit(graph, samples=1000, restarts=2, seed=7)
    second = hrg_fit(graph, samples=1000, restarts=2, seed=7)
    assert first.log_likelihood == second.log_likelihood
    assert first.children == second.children


# ----------------------------------------------------------------------- §37.4 density vs hierarchy


def test_density_and_modularity_profiles_disagree_on_the_ring_of_cliques() -> None:
    """Figure 37.11(a) and (c): everyone in one community still has some density even though its
    modularity is exactly zero, and the level that peaks in modularity is also the densest one
    here -- the book's "agreement" case, because this ring is small enough not to trip §36.1's
    resolution limit."""
    ring = _ring_of_cliques(6, 5)
    whole = [list(ring.nodes)]
    dendrogram = louvain_levels(ring, seed=1)

    everyone_together = score_level(ring, whole)
    assert everyone_together.modularity == 0.0
    assert everyone_together.density > 0.0  # Figure 37.11(a)

    best = dendrogram.levels[dendrogram.cut]
    assert best.modularity > 0.0
    assert best.density > everyone_together.density  # Figure 37.11(c): agreement


def test_density_stays_high_while_modularity_collapses_on_a_split_complete_graph() -> None:
    """Figure 37.11(d): a partition of connected node pairs can hold onto a high density while
    its modularity is near zero, because a complete graph has no community structure for any
    split to find."""
    complete = nx.complete_graph(20)
    nx.set_edge_attributes(complete, 1, "weight")
    relabelled = nx.relabel_nodes(complete, {n: str(n) for n in complete.nodes})
    half = [[str(n) for n in range(10)], [str(n) for n in range(10, 20)]]

    split = score_level(relabelled, half)
    assert split.density == pytest.approx(1.0)  # every possible edge is present
    assert abs(split.modularity) < 0.05  # nothing like the ring's peak modularity
