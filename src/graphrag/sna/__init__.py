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
from graphrag.sna.export import (
    NETWORKS,
    bipartite_projection,
    build_network,
    ego,
    entity_co_mention,
    speaker_co_participation,
    topic_co_occurrence,
    write_graph,
)
from graphrag.sna.guide import render_guide
from graphrag.sna.measures import CENTRALITIES, brokers, centrality, summary, top_n

__all__ = [
    "CENTRALITIES",
    "NETWORKS",
    "Analysis",
    "GMMResult",
    "KChoice",
    "KMeansResult",
    "LouvainResult",
    "NullModelResult",
    "bipartite_projection",
    "brokers",
    "build_network",
    "centrality",
    "choose_k_gmm",
    "choose_k_kmeans",
    "compare_partitions",
    "ego",
    "entity_co_mention",
    "gmm",
    "kmeans",
    "louvain",
    "null_model_modularity",
    "render_guide",
    "render_markdown",
    "run_analysis",
    "speaker_co_participation",
    "spectral_embedding",
    "summary",
    "to_payload",
    "top_n",
    "topic_co_occurrence",
    "write_graph",
]
