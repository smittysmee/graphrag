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
    # signed networks, time windows, bipartite projections, node attributes
    assert len(READING_RULES) == 4


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
    assert render_guide() in path.read_text(encoding="utf-8")
