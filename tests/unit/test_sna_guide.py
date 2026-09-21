"""The selection rules are written once; the docs and the skill must quote them, not paraphrase."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag.sna.analysis import METHODS
from graphrag.sna.guide import (
    ALWAYS,
    METHOD_RULES,
    NETWORK_RULES,
    READING_RULES,
    rationale,
    render_guide,
)

REPO = Path(__file__).resolve().parents[2]
QUOTING_FILES = (
    REPO / "docs" / "SNA.md",
    REPO / ".claude" / "skills" / "graph-rag-sna" / "SKILL.md",
)


def test_the_guide_covers_every_method_network_and_caveat() -> None:
    text = render_guide()
    for rule in METHOD_RULES:
        assert rule.name in text
        assert rule.use_when in text
        assert rule.watch_out in text
    for rule in NETWORK_RULES:
        assert rule.answers in text
    for item in ALWAYS:
        assert item in text


def test_the_guide_covers_every_filtered_network_rule() -> None:
    """The filters change what an edge means, so their rules live beside the method rules."""
    text = render_guide()
    for heading, rules in READING_RULES:
        assert f"### {heading}" in text
        for rule in rules:
            assert rule.name in text
            assert rule.rule in text
    # density, signed networks, time windows, bipartite projections, bipartite community
    # discovery, projection weights, directed networks, layers and time, node attributes,
    # homophily, sampling, against random, null models, backboning, random walks, degree,
    # quantitative assortativity, node roles, hierarchies, high-order, paths and components,
    # ranking, link prediction (predict-eval), link prediction simple graphs, signed and
    # multilayer prediction, graph partitions, community evaluation, hierarchical communities,
    # overlapping coverage, multilayer community discovery, robustness, motifs and mining,
    # spreading, core-periphery, embeddings, visualization, node vector distance, graph
    # summarization, topological distances, graph convolution, uncertain edges
    assert len(READING_RULES) == 41
    assert len(dict(READING_RULES)["Graph summarization (sna summarize)"]) == 4
    assert (
        len(dict(READING_RULES)["Multilayer community discovery (sna multilayer-communities)"]) == 5
    )
    assert len(dict(READING_RULES)["Node roles (sna roles)"]) == 5
    assert len(dict(READING_RULES)["Visualization (sna draw)"]) == 7
    assert len(dict(READING_RULES)["Hierarchies (--network relations / sna hierarchy)"]) == 5
    assert len(dict(READING_RULES)["High-order (sna highorder)"]) == 4
    assert len(dict(READING_RULES)["Overlapping coverage (sna overlap)"]) == 4
    assert len(dict(READING_RULES)["Community evaluation (every grouping report)"]) == 5
    assert len(dict(READING_RULES)["Node vector distance (sna distance)"]) == 8
    assert (
        len(
            dict(READING_RULES)[
                "Graph partitions (--method sbm|infomap|walktrap|label-propagation / sna community)"
            ]
        )
        == 4
    )
    assert (
        len(
            dict(READING_RULES)[
                "Hierarchical communities (louvain_levels / girvan_newman_dendrogram / hrg_fit)"
            ]
        )
        == 2
    )
    assert (
        len(
            dict(READING_RULES)["Embeddings (sna.embed, --features spectral|node2vec|metapath2vec)"]
        )
        == 10
    )
    assert len(dict(READING_RULES)["Graph neural networks (sna complete, sna.gnn)"]) == 6


def test_every_method_has_a_rationale_line_for_the_report() -> None:
    for method in METHODS:
        assert rationale(method) != f"{method}: no selection rule recorded."
    assert "no selection rule recorded" in rationale("astrology")


@pytest.mark.parametrize("path", QUOTING_FILES, ids=lambda p: p.name)
def test_docs_and_skill_quote_the_rules_verbatim(path: Path) -> None:
    """Guards against the prose an agent reads drifting away from the code that prints it.

    If this fails, do not edit the file by hand: run ``graphrag sna guide`` and paste the output
    back over the block, or change ``graphrag/sna/guide.py`` and then do the same.
    """
    if not path.exists():
        pytest.skip(f"{path} is not mounted in this container")
    text = path.read_text(encoding="utf-8")
    block = render_guide()
    assert block in text
    # And the block is a block: a blank line between its last bullet and whatever heading comes
    # next, which a regeneration that joins the two would quietly swallow.
    assert text[text.index(block) + len(block) :].startswith("\n\n")
