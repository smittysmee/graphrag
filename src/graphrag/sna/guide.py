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


@dataclass(frozen=True)
class ReadingRule:
    """One rule about how to read a filtered network, and why it is easy to get wrong."""

    name: str
    rule: str


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
    NetworkRule(
        name="speakers-entities",
        nodes="both the speakers and the entities their passages mention, as two modes",
        edges=(
            "a passage by that speaker naming that entity, weighted by how many documents hold one"
        ),
        answers=(
            "who wrote about what; projected onto one side, who wrote about the same things, or "
            "which things the same people wrote about"
        ),
    ),
)

SIGNED_RULES: tuple[ReadingRule, ...] = (
    ReadingRule(
        name="The edge means something else now",
        rule=(
            "a stance filter changes the question from 'discussed together' to 'praised "
            "together' or 'complained about together'. Say which one in the same sentence that "
            "reports the finding: a reader who sees a co-mention network assumes the "
            "unfiltered one."
        ),
    ),
    ReadingRule(
        name="The stances are not complements",
        rule=(
            "the praise network and the complaint network do not add up to the unfiltered "
            "network. A mention nobody annotated is in neither, so what is missing from one is "
            "not therefore in the other, and 'not complained about' is never evidence of "
            "approval."
        ),
    ),
    ReadingRule(
        name="Compare structure only after comparing n",
        rule=(
            "a denser complaint network usually means more complaints were annotated, not that "
            "complaints cluster harder. Report the node and edge counts of each signed network "
            "before saying anything about the shape of either."
        ),
    ),
    ReadingRule(
        name="A stance is a reading of one passage",
        rule=(
            "it came from an agent reading text, and it describes that passage, not the entity. "
            "Quote the passage next to the count; a count nobody can trace back to text that "
            "reads that way is an error in the annotation, not a finding."
        ),
    ),
)

WINDOW_RULES: tuple[ReadingRule, ...] = (
    ReadingRule(
        name="Report n for each window",
        rule=(
            "a partition over 40 nodes and a partition over 400 are not two measurements of one "
            "thing. Print both counts beside any before-and-after claim."
        ),
    ),
    ReadingRule(
        name="Undated is outside",
        rule=(
            "a window is answered from dated passages, so every document the attribution pass "
            "could not date drops out of every network as soon as a window is set. Check each "
            "window against the unwindowed network before reading a disappearance as a change "
            "in the world."
        ),
    ),
    ReadingRule(
        name="Small windows overfit",
        rule=(
            "cut a corpus finely enough and every window has tidy communities, because a "
            "handful of documents partitions cleanly. Run the null model inside each window "
            "rather than only on the whole corpus."
        ),
    ),
    ReadingRule(
        name="Community numbers do not survive a rebuild",
        rule=(
            "Louvain numbers its communities per run, so 'community 2 grew' is meaningless "
            "across two windows. Compare the partitions with the adjusted Rand index over the "
            "nodes both windows contain, and describe groups by their members."
        ),
    ),
    ReadingRule(
        name="A rank change can be a roster change",
        rule=(
            "a node climbing twenty places may have stood still while the nodes above it left "
            "the window. Read the entering and leaving lists before reading the rank table."
        ),
    ),
)

BIPARTITE_RULES: tuple[ReadingRule, ...] = (
    ReadingRule(
        name="Project the side you are asking about",
        rule=(
            "onto speakers answers who wrote about the same things; onto entities answers which "
            "things the same people wrote about. They are different questions and the same "
            "two-mode network answers only one of them at a time."
        ),
    ),
    ReadingRule(
        name="Projected weights inflate",
        rule=(
            "one document naming twenty entities contributes 190 entity pairs on its own, so a "
            "projection's weights are combinatorial rather than additive. Raise --min-weight "
            "before ranking anything, and never compare a projected weight with a two-mode one."
        ),
    ),
    ReadingRule(
        name="The projection throws the other mode away",
        rule=(
            "a heavy speaker-speaker edge does not say which entities it ran through. Export "
            "the two-mode network beside the projected one, or read the partners recorded on "
            "each node, before naming what a group has in common."
        ),
    ),
    ReadingRule(
        name="Sharing a node is not agreement",
        rule=(
            "two speakers joined by an entity both named it, which includes one recommending it "
            "and the other warning against it. Add --stance if the question is whether they "
            "agreed."
        ),
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


#: The filtered-network sections, in the order ``render_guide`` prints them.
READING_RULES: tuple[tuple[str, tuple[ReadingRule, ...]], ...] = (
    ("Signed networks (--stance)", SIGNED_RULES),
    ("Time windows (--since / --until)", WINDOW_RULES),
    ("Bipartite projections (--network speakers-entities)", BIPARTITE_RULES),
)


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
    for heading, rules in READING_RULES:
        lines += ["", f"### {heading}", ""]
        lines += [f"- **{rule.name}** — {rule.rule}" for rule in rules]
    lines += ["", "### Always", ""]
    lines += [f"- {item}" for item in ALWAYS]
    return "\n".join(lines)
