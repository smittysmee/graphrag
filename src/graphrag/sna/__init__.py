"""Network analysis over the graph: build networks, measure them, cluster them, explain why.

Three networks come out of the graph, and each answers a different question:

``speakers``
    Who appears alongside whom. Two speakers are joined when they appear in the same document;
    the edge weight is how many documents they share. Use it for "whose voice sits next to
    whose" and for brokerage between otherwise separate groups.
``entities``
    What is discussed together. Two entities are joined when both are mentioned in the same
    passage; the edge weight is how many passages mention both. Use it for "what is this thing
    talked about alongside".
``topics``
    The persisted ``Topic-[:CO_OCCURS]-Topic`` edges from ingestion, filtered to one persona.
``speakers-entities``
    Two modes at once: speakers and the entities their passages mention, joined when a speaker
    wrote a passage naming that entity. Project it onto either side for "who wrote about the
    same things" or "which things the same people wrote about".

Three filters narrow any of them, and each changes what an edge means rather than only how many
there are: ``--stance`` keeps mentions the annotation layer marked, so an edge becomes "praised
together" rather than "discussed together"; ``--facet`` keeps passages about one function of the
subject; ``--since``/``--until`` keep dated passages, which drops every undated document.
``graphrag sna stances`` reports what the corpus says about each entity, and ``graphrag sna
compare`` builds one network over two windows and reports what moved.

Every network is a sample of a corpus, not of a population: it describes who was recorded and
what was written down. ``graphrag.sna.guide`` holds the method-selection rules, and the report
repeats the sampling frame next to every number.
"""

from graphrag.sna.analysis import Analysis, render_markdown, run_analysis, to_payload
from graphrag.sna.cluster import (
    GMMResult,
    KChoice,
    KMeansResult,
    LouvainResult,
    NullModelResult,
    choose_k_gmm,
    choose_k_kmeans,
    compare_partitions,
    gmm,
    kmeans,
    louvain,
    null_model_modularity,
    spectral_embedding,
)
from graphrag.sna.compare import (
    Comparison,
    RankChange,
    Window,
    compare_windows,
    render_comparison,
)
from graphrag.sna.export import (
    NETWORKS,
    PROJECTIONS,
    STANCES,
    bipartite_projection,
    build_network,
    ego,
    entity_co_mention,
    speaker_co_participation,
    speaker_entity_bipartite,
    topic_co_occurrence,
    write_graph,
)
from graphrag.sna.guide import render_guide
from graphrag.sna.measures import CENTRALITIES, brokers, centrality, summary, top_n
from graphrag.sna.stances import (
    EntityStances,
    SignedPair,
    StanceReport,
    build_stance_report,
    render_stances,
)

__all__ = [
    "CENTRALITIES",
    "NETWORKS",
    "PROJECTIONS",
    "STANCES",
    "Analysis",
    "Comparison",
    "EntityStances",
    "GMMResult",
    "KChoice",
    "KMeansResult",
    "LouvainResult",
    "NullModelResult",
    "RankChange",
    "SignedPair",
    "StanceReport",
    "Window",
    "bipartite_projection",
    "brokers",
    "build_network",
    "build_stance_report",
    "centrality",
    "choose_k_gmm",
    "choose_k_kmeans",
    "compare_partitions",
    "compare_windows",
    "ego",
    "entity_co_mention",
    "gmm",
    "kmeans",
    "louvain",
    "null_model_modularity",
    "render_comparison",
    "render_guide",
    "render_markdown",
    "render_stances",
    "run_analysis",
    "speaker_co_participation",
    "speaker_entity_bipartite",
    "spectral_embedding",
    "summary",
    "to_payload",
    "top_n",
    "topic_co_occurrence",
    "write_graph",
]
