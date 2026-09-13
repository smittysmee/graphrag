"""The method-selection rules, written once.

``docs/SNA.md`` and ``.claude/skills/graph-rag-sna/SKILL.md`` quote ``render_guide()`` verbatim
and a unit test holds them to it, so the prose an agent reads and the prose the CLI prints
cannot drift apart. ``graphrag sna guide`` prints the same text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodRule:
    name: str
    takes: str
    """What the method consumes: edges, or one vector per node."""
    use_when: str
    choose_k: str
    watch_out: str


@dataclass(frozen=True)
class NetworkRule:
    name: str
    nodes: str
    edges: str
    answers: str


METHOD_RULES: tuple[MethodRule, ...] = (
    MethodRule(
        name="Louvain",
        takes="edges",
        use_when=(
            "the object is a graph and the question is which nodes form dense groups. Prefer it "
            "over K-means whenever the data are edges rather than vectors: turning a graph into "
            "coordinates first throws away information that modularity uses directly."
        ),
        choose_k=(
            "no k to choose. The number of communities falls out of the graph, tuned only by "
            "--resolution: above 1.0 finds smaller groups, below 1.0 finds larger ones."
        ),
        watch_out=(
            "it is randomised, so run many seeds and report the stability score; the resolution "
            "limit absorbs genuinely small communities into larger ones in a big network; and a "
            "high modularity alone is not evidence, because random graphs score above zero too. "
            "Check the null model before claiming structure."
        ),
    ),
    MethodRule(
        name="K-means",
        takes="one vector per node",
        use_when=(
            "each node already has a feature row, the groups are expected to be compact and "
            "roughly equal in spread, and hard membership is the useful answer."
        ),
        choose_k=(
            "by silhouette, with the inertia elbow as a second opinion. When the two disagree, "
            "prefer silhouette for a claim about groups and say where the elbow pointed."
        ),
        watch_out=(
            "it fails on elongated or overlapping clusters and on clusters of very unequal size, "
            "because it assigns every point to its nearest centre and nothing else."
        ),
    ),
    MethodRule(
        name="Gaussian mixture",
        takes="one vector per node",
        use_when=(
            "clusters may overlap, may have different shapes or spreads, or when soft membership "
            "is the useful output: a node that belongs 70% to one group and 30% to another."
        ),
        choose_k="by BIC, which penalises extra components harder than AIC does.",
        watch_out=(
            "the covariance type is a modelling choice and must be stated (full, tied, diagonal "
            "or spherical); with few points per component a full covariance degenerates onto "
            "single points."
        ),
    ),
    MethodRule(
        name="Spectral embedding",
        takes="edges, and returns vectors",
        use_when=(
            "the data are a graph but you want exactly k groups, or want to use a vector method "
            "on them. Embed first, then run K-means or a Gaussian mixture on the embedding."
        ),
        choose_k="whatever the vector method that follows it uses.",
        watch_out=(
            "the embedding has as many dimensions as you ask for and no natural number of them; "
            "and it inherits the graph's fragmentation, so a network in many components embeds "
            "into clumps that any clusterer will happily separate."
        ),
    ),
)

CENTRALITY_RULES: tuple[tuple[str, str], ...] = (
    ("degree", "local prominence: how many distinct others a node sits with."),
    ("betweenness", "brokerage: how often a node lies on the path between two others."),
    ("eigenvector or PageRank", "influence among the influential."),
    ("closeness", "reach: how near a node is to everyone else."),
    (
        "weighted variants",
        "whenever edges carry counts rather than mere presence, which here they always do.",
    ),
)

NETWORK_RULES: tuple[NetworkRule, ...] = (
    NetworkRule(
        name="speakers",
        nodes="the speakers attached to a persona's documents",
        edges="a shared document, weighted by how many they share",
        answers="whose voice appears alongside whose, and who bridges otherwise separate groups",
    ),
    NetworkRule(
        name="entities",
        nodes="the entities an extraction pass named",
        edges="a shared passage, weighted by how many passages mention both",
        answers="what a thing is discussed alongside, and which subjects hang together",
    ),
    NetworkRule(
        name="topics",
        nodes="the topics assigned at ingestion",
        edges="the stored co-occurrence weight",
        answers="how the corpus was labelled, as a first map before any extraction exists",
    ),
)

ALWAYS: tuple[str, ...] = (
    "Report n. A centrality ranking over 12 nodes is an anecdote with decimal places.",
    "Report the null model. Without it, modularity is a number, not a finding.",
    "Report the stability score. A partition that changes with the seed cannot carry names.",
    "Report the sampling frame. These networks describe who was recorded and what was written "
    "down, never a population; a node is absent when nobody wrote it down, which is not the "
    "same as it not existing.",
    "Say which features were used. Clustering the graph's structure and clustering what the "
    "nodes are about answer different questions and can disagree completely.",
)

_RATIONALE: dict[str, str] = {
    "louvain": (
        "Louvain because the input is a graph and the question is which nodes form dense "
        "groups; no k was imposed, and the partition is checked against a degree-preserving "
        "null model and across seeds."
    ),
    "kmeans": (
        "K-means because each node carries a feature vector and hard, compact groups are the "
        "useful answer; k was chosen by silhouette with the inertia elbow as a second opinion."
    ),
    "gmm": (
        "A Gaussian mixture because the groups may overlap or differ in spread and soft "
        "membership is the useful answer; k was chosen by BIC and the covariance type is stated."
    ),
}


def rationale(method: str) -> str:
    """The one-line reason a method was used, for the report's rationale row."""
    return _RATIONALE.get(method, f"{method}: no selection rule recorded.")


def render_guide() -> str:
    """The selection rules as markdown. Quoted verbatim by the docs and the skill."""
    lines: list[str] = ["### Which network", ""]
    for network in NETWORK_RULES:
        lines.append(
            f"- **{network.name}** — nodes are {network.nodes}; an edge is {network.edges}. "
            f"Answers: {network.answers}."
        )
    lines += ["", "### Which method", ""]
    for method in METHOD_RULES:
        lines += [
            f"**{method.name}** (takes {method.takes})",
            "",
            f"- Use when {method.use_when}",
            f"- Choosing k: {method.choose_k}",
            f"- Watch out: {method.watch_out}",
            "",
        ]
    lines += ["### Which centrality", ""]
    lines += [f"- **{name}** — {meaning}" for name, meaning in CENTRALITY_RULES]
    lines += ["", "### Always", ""]
    lines += [f"- {item}" for item in ALWAYS]
    return "\n".join(lines)
