"""Message passing and graph convolution (ch. 44), plus attention (ch. 45, §45.1): GCN, GraphSAGE
and GAT over one of this package's own networks, behind the optional ``graphrag[gnn]`` extra
(CPU-only ``torch``). Two things are built, both scoped by CLAUDE.md's rule that nothing here
writes to the graph:

**Entity-attribute completion** (§44.1, §44.3). A node's neighbourhood carries information about
it that its own text does not -- §44.1's whole point is that ``H1`` "is not dependent only on
``H0`` but also on the structure of G" -- so a two-layer GCN or GraphSAGE, trained on the entities
a persona's extraction sidecars already tagged, can suggest a value for the ones they did not.
:func:`complete_attribute` trains **only on entity-owned values** (never borrowed ones -- see its
docstring for why that is the same refusal :mod:`graphrag.sna.attributes` already makes about
assortativity) and returns suggestions shaped to paste into an extraction sidecar's
``attributes:`` block, each with a confidence and the class it beat. Nothing is written to the
graph; a person reads the suggestion and decides.

**Link prediction through ATL-25** (§44.3, via self-supervision). §44.3's own framing -- a GCN
needs *some* ground truth ``T`` to fit ``W`` against -- has an answer when no attribute is given:
``T`` can be the training graph's own edges. :func:`embed_gnn` fits a GCN or GraphSAGE by
reconstructing the training graph from the dot product of its own node embeddings (the graph
autoencoder of Kipf and Welling, *Variational Graph Auto-Encoders*, NeurIPS Bayesian Deep Learning
workshop 2016 -- not itself in ch. 44, which names only supervised node prediction and, in
passing, a mutual-information alternative it does not build, ref. 29 of §44.3); the fitted
embeddings' dot product is then a :class:`graphrag.sna.experiment.ScoreTable`
(:func:`gcn_link_scorer`) that runs through :func:`graphrag.sna.experiment.evaluate_predictor`
exactly like every other scorer in this package.

**§44.2's limits, as ``guide.py`` rules**, not as code that tries to route around them: piling on
layers smooths every node to the same vector before it reaches anywhere near a graph's diameter,
a peripheral node's own signal is squashed by every hub it passes through on the way to a
labelled neighbour, and a GCN's ``W`` only ever fits the ground truth it is handed, so a borrowed
label is not one to fit against.

**GraphSAGE is not named in ch. 44** (checked against the book's own text and references list;
Hamilton is cited as a co-author of adjacent papers, never for GraphSAGE itself). Its mean
aggregator -- ``sigma(CONCAT(h_v, mean_{u in N(v)} h_u) * W)`` -- is Hamilton, Ying and Leskovec,
*Inductive Representation Learning on Large Graphs*, NeurIPS 2017, and it is the same MPGNN family
§44.1 describes with the skip-connection fix §44.2 gives for over-smoothing ("run your update
function on the message v receives plus v's representation in the previous layer"), just without
the symmetric degree normalisation ``D^-1/2(I + A)D^-1/2`` that makes it a *GCN* specifically.

**Attention (§45.1, GATs).** A GCN's ``D̂^-1/2(I + A)D̂^-1/2`` is, in the book's own words, "sort
of an attention mechanism ... where each neighbor get[s] the same amount of attention" -- fixed at
``1/sqrt(kv * ku)`` and never learned. A Graph Attention Network keeps the same
``Hl = sigma(Hl-1 alpha_l Wl)`` shape but learns ``alpha`` instead: a shared linear map ``Wl``,
then for every ``(v, u)`` with ``u`` in ``v``'s neighbourhood *including v itself*, a score
``LeakyReLU(a^T [Wl h_v || Wl h_u])`` softmax-normalised over that neighbourhood into ``alpha_vu``
(Velickovic et al., *Graph Attention Networks*, arXiv:1710.10903, 2017, the paper the chapter
cites). :func:`gat_layer` is that one head; :func:`classify`, :func:`embed_gnn`,
:func:`gcn_link_scorer` and :func:`complete_attribute` all take ``method="gat"`` the same way they
take ``"gcn"`` and ``"graphsage"``, multi-head via ``heads`` (concatenated at every hidden layer,
averaged at the last, the original paper's own convention -- p. 656's graph-transformer section,
§45.2, is the same idea with several independently learned ``alpha`` functions per layer, which
this module does not build; see ``docs/SNA_FOUNDATIONS.md`` for why). Because the fitted attention
says which neighbour the network leaned on, :func:`complete_attribute` surfaces it per suggestion
when ``method="gat"`` -- see :class:`AttributeSuggestion`.

**The node-order and node-type contract.** Every function here takes a network built by
:mod:`graphrag.sna.export` (string node ids) and returns rows aligned to
:func:`graphrag.sna.matrices.node_order`, the same contract :mod:`graphrag.sna.embed` uses.
"""

from __future__ import annotations

import importlib.util
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.attributes import attribute_labels, attribute_sources
from graphrag.sna.embed import Embedding
from graphrag.sna.experiment import ScoreTable
from graphrag.sna.export import OWN
from graphrag.sna.matrices import Node, adjacency, node_order, stochastic

#: Everything below that touches a tensor calls :func:`_require_torch` for its ``torch`` module
#: rather than importing it at module scope (or under ``TYPE_CHECKING`` for annotations -- every
#: torch-shaped value here is typed ``Any`` instead, deliberately, so this file never needs an
#: import mypy would have to resolve). That is what keeps ``import graphrag.sna.gnn`` -- and so
#: ``import graphrag.sna``, which re-exports every sibling module -- working on an interpreter
#: that has never heard of ``torch``.

__all__ = [
    "INSTALL_HINT",
    "METHODS",
    "MIN_OWN_LABELS",
    "AttributeSuggestion",
    "Classification",
    "CompletionReport",
    "available",
    "build_features",
    "classify",
    "complete_attribute",
    "completion_payload",
    "embed_gnn",
    "gat_layer",
    "gcn_layer",
    "gcn_link_scorer",
    "mean_adjacency",
    "neighbor_mask",
    "normalized_adjacency",
    "render_completion",
    "sage_layer",
]

Method = Literal["gcn", "graphsage", "gat"]
METHODS: tuple[Method, ...] = ("gcn", "graphsage", "gat")

#: What every command and function here says, verbatim, when ``torch`` is not on the interpreter.
INSTALL_HINT = (
    "GCN/GraphSAGE/GAT need the `graphrag[gnn]` extra (CPU-only torch): from the repo root, "
    "`uv sync --extra gnn` in a dev container, or `pip install 'graphrag[gnn]'` elsewhere. "
    "Chapters 44 and 45's message passing and attention are fit by backpropagation (§44.3), "
    "which nothing already in this package's numpy/scikit-learn stack provides."
)

#: §45.1's default: one attention head. `classify`, `embed_gnn`, `gcn_link_scorer` and
#: `complete_attribute` all take `heads` for `method="gat"`; every other method ignores it.
DEFAULT_GAT_HEADS = 1

#: Below this many entity-owned values for an attribute, :func:`complete_attribute` refuses
#: rather than fit noise -- §44.3's W needs a ground truth to fit, and a handful of examples
#: across two classes is not one a held-out split can even measure honestly.
MIN_OWN_LABELS = 6


def available() -> bool:
    """Whether ``torch`` is importable on this interpreter, without importing it.

    A presence check, not an import: :mod:`torch` is a large optional extra, so every caller that
    only wants to know whether to offer the feature (the CLI's exit-2 path chief among them) pays
    nothing for asking.
    """
    return importlib.util.find_spec("torch") is not None


def _require_torch() -> Any:
    """``torch``, imported now, or a :class:`ModuleNotFoundError` carrying :data:`INSTALL_HINT`.

    Every function below that touches a tensor calls this rather than importing ``torch`` at
    module scope, which is what keeps ``import graphrag.sna.gnn`` -- and so ``import
    graphrag.sna``, which re-exports every sibling module -- working on an interpreter that has
    never heard of ``torch``, exactly as the rest of this package's tests expect.
    """
    if not available():
        raise ModuleNotFoundError(INSTALL_HINT)
    import torch

    return torch


# --------------------------------------------------------------------------- §44.3: the algebra


def normalized_adjacency(graph: nx.Graph, nodes: Sequence[Node]) -> np.ndarray:
    """``D̂^-1/2(I + A)D̂^-1/2`` of §44.3, aligned to ``nodes``: self loops added ("to fix this
    issue we add self loops... assuming that each node v is a neighbor of itself"), then the
    symmetric degree normalisation Kipf and Welling use (ref. 23) rather than the plain ``D^-1A``
    the chapter mentions and sets aside, because it "still produces a stochastic matrix" while
    "taking into account how many neighbors [the neighbours] themselves have". ``D̂`` is the
    degree matrix of ``I + A``, i.e. each node's own degree plus one for its self loop.

    This is the propagation matrix :func:`classify` and :func:`embed_gnn` use for
    ``method="gcn"``; it always includes the self loop, so a GCN layer never simply forgets what
    a node started as, the failure mode §44.3 introduces the identity matrix to fix.
    """
    matrix, order = adjacency(graph, nodes=list(nodes), weight="weight")
    hat = np.asarray(matrix, dtype=np.float64) + np.eye(len(order), dtype=np.float64)
    degree = hat.sum(axis=1)
    inverse_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0.0)
    return np.asarray(inverse_sqrt[:, None] * hat * inverse_sqrt[None, :], dtype=np.float64)


def mean_adjacency(graph: nx.Graph, nodes: Sequence[Node]) -> np.ndarray:
    """The row-stochastic adjacency, no self loop, aligned to ``nodes``: GraphSAGE's mean
    aggregator over a node's neighbours (Hamilton et al. 2017 -- see this module's docstring for
    why that paper and not the book), before it is concatenated with the node's own previous
    representation in :func:`sage_layer`. An isolated node's row stays at zero (sub-stochastic,
    :func:`graphrag.sna.matrices.stochastic`'s own honest answer for "no neighbours to average"),
    so its representation at every layer comes from its own features alone.
    """
    matrix, _ = stochastic(graph, nodes=list(nodes), orientation="row", weight="weight")
    return np.asarray(matrix, dtype=np.float64)


def neighbor_mask(graph: nx.Graph, nodes: Sequence[Node]) -> np.ndarray:
    """The 0/1 neighbourhood of §45.1, aligned to ``nodes``: ``mask[v][u] == 1`` when ``u`` is
    ``v`` ("softmax over each node's neighbourhood") or a neighbour of ``v``, 0 otherwise.

    Unlike :func:`normalized_adjacency` and :func:`mean_adjacency`, this is not a ready-to-use
    propagation matrix -- it carries no weight at all, only which pairs are allowed a nonzero
    attention coefficient. :func:`gat_layer` computes the actual weight for every ``1`` here by
    backpropagation; a GCN or GraphSAGE layer never sees this matrix, because their weights are
    fixed before training starts. Edge weight is deliberately not read: §45.1's whole point is
    that the *learned* function decides how much a neighbour counts, not this package's own
    co-mention counts.
    """
    matrix, _order = adjacency(graph, nodes=list(nodes), weight=None)
    mask = (np.asarray(matrix, dtype=np.float64) != 0).astype(np.float64)
    np.fill_diagonal(mask, 1.0)
    return np.asarray(mask, dtype=np.float64)


def build_features(
    graph: nx.Graph,
    nodes: Sequence[Node],
    text: tuple[Sequence[str], np.ndarray] | None,
) -> tuple[np.ndarray, list[str], list[Node]]:
    """``H0`` of §44.1, aligned to ``nodes``: each node's mean text embedding beside its
    (degree-normalised) weighted degree, or degree alone when ``text`` is not given.

    ``text`` is ``graphrag.graph.store.GraphStore.mean_embeddings``'s own return shape --
    ``(keys, matrix)`` -- so a caller hands this the same pair :mod:`graphrag.sna.analysis`
    already reads for ``--features embedding`` (only the entities network has one: the mean
    embedding of the passages mentioning each entity). A node absent from ``keys`` gets a zero
    row for the text half rather than being dropped, because a GCN's ``H0`` needs exactly one row
    per node (§42.1's contract, carried over); the third return value is which nodes *did* get a
    real text row, so a caller can say how many were degree-only.

    Degree alone is the chapter's own fallback (p. 636): "If your graph doesn't have node
    attributes, you can always use some structural properties as attributes, for instance their
    degree." It is scaled by the largest degree in ``nodes`` so it sits near the same range as an
    L2-normalised text embedding rather than dwarfing it in the concatenation.

    **A single scalar is a narrow fallback, worth knowing before it surprises a result.** With no
    text at all, ``H0`` is rank 1 (one number per node), so ``H0 @ W1`` is an outer product: every
    node's pre-activation vector at the first layer points the *same direction*, scaled only by
    its own degree, however wide ``W1`` is. On a network where degree does not separate the
    classes a caller cares about -- a planted-partition graph is the textbook case, and it is
    this package's own link-prediction fixture (`sna.predict`'s `planted_communities`: "community,
    not degree, predicts an edge here") -- that leaves nothing for backpropagation to work with.
    Every network this package actually builds from a corpus has real per-entity text behind it,
    so this is a fallback for the case chapter 44 names, not the expected path.
    """
    degree = np.array([[float(graph.degree(n, weight="weight"))] for n in nodes], dtype=np.float64)
    scale = float(degree.max()) if degree.size and degree.max() > 0 else 1.0
    degree = degree / scale
    if text is None:
        return degree, ["degree"], []
    keys, matrix = text
    index = {key: i for i, key in enumerate(keys)}
    dims = int(matrix.shape[1]) if matrix.size else 0
    has_text = [n for n in nodes if n in index]
    rows = np.array(
        [matrix[index[n]] if n in index else np.zeros(dims) for n in nodes],
        dtype=np.float64,
    )
    combined = np.concatenate([rows, degree], axis=1)
    names = [f"text_{i}" for i in range(dims)] + ["degree"]
    return combined, names, has_text


def _propagation_matrix(graph: nx.Graph, nodes: Sequence[Node], method: Method) -> np.ndarray:
    if method == "gcn":
        return normalized_adjacency(graph, nodes)
    if method == "gat":
        return neighbor_mask(graph, nodes)
    return mean_adjacency(graph, nodes)


def _check_method(method: str) -> Method:
    if method not in METHODS:
        msg = f"method must be one of {METHODS}, got {method!r}"
        raise ValueError(msg)
    return method


# --------------------------------------------------------------------------- §44.1/§44.3: layers


def gcn_layer(
    h: Any, propagation: Any, weight: Any, activation: Callable[[Any], Any] | None = None
) -> Any:
    """One GCN layer of §44.3: ``propagation @ h @ weight``, then ``activation`` if given.

    ``propagation`` is :func:`normalized_adjacency`'s matrix as a tensor; ``h`` is ``H^(l-1)``
    and ``weight`` is ``W^l``. ``activation=None`` is the identity, which is what makes this
    function double as the hand-computable primitive the known-answer test checks against
    ``D̂^-1/2(I + A)D̂^-1/2 X W`` directly -- there is no hidden nonlinearity to work around.
    """
    out = propagation @ h @ weight
    return out if activation is None else activation(out)


def sage_layer(
    h: Any, propagation: Any, weight: Any, activation: Callable[[Any], Any] | None = None
) -> Any:
    """One GraphSAGE mean-aggregator layer: ``CONCAT(h, propagation @ h) @ weight``, then
    ``activation`` if given. ``propagation`` is :func:`mean_adjacency`'s matrix as a tensor, so
    the neighbour half is an unweighted-by-degree average rather than :func:`gcn_layer`'s
    symmetric normalisation -- see this module's docstring for why the two use different
    matrices. ``weight`` is ``(2 * h.shape[1], out_dim)``: the concatenation doubles the input
    width `init_weights` accounts for.
    """
    torch = _require_torch()
    combined = torch.cat([h, propagation @ h], dim=1)
    out = combined @ weight
    return out if activation is None else activation(out)


def gat_layer(
    h: Any,
    mask: Any,
    weight: Any,
    attention: Any,
    activation: Callable[[Any], Any] | None = None,
    *,
    leak: float = 0.2,
) -> tuple[Any, Any]:
    """One GAT attention head (§45.1): ``z = h @ weight`` (the "shared linear map"), then for
    every ``(v, u)`` allowed by ``mask`` (§45.1's ``N(v)`` "including itself" -- see
    :func:`neighbor_mask`) a coefficient ``e_vu = LeakyReLU(a^T [z_v || z_u])``, softmax-
    normalised over ``u`` into ``alpha_vu``, then ``out_v = sum_u alpha_vu * z_u``. ``activation``
    is applied to ``out`` last, mirroring :func:`gcn_layer`.

    ``attention`` is ``a``, shape ``(1, 2 * weight.shape[1])``: split as ``[a_src | a_dst]``, the
    concatenation-then-dot-product the book states is computed as two separate dot products
    (``a_src . z_v + a_dst . z_u``) for efficiency -- a dot product distributes over a
    concatenation, so this is exactly ``a^T [z_v || z_u]``, not an approximation of it. ``leak``
    is the LeakyReLU's negative slope; the chapter does not fix one, this module uses the original
    GAT paper's 0.2 throughout.

    Returns ``(out, alpha)``: ``alpha`` is ``v``'s row of attention over its own neighbourhood,
    one row per node, aligned to ``mask``'s order -- what :func:`complete_attribute` reports as
    "which neighbours the prediction leaned on" for ``method="gat"``.

    A row of ``mask`` that is all zero except the diagonal (an isolated node) never produces a
    softmax over an empty set: :func:`neighbor_mask` always sets the self loop, so every node
    has at least itself to attend to.
    """
    torch = _require_torch()
    z = h @ weight
    out_dim = z.shape[1]
    a_flat = attention.reshape(-1)
    a_src, a_dst = a_flat[:out_dim], a_flat[out_dim:]
    src_scores = z @ a_src
    dst_scores = z @ a_dst
    e = src_scores[:, None] + dst_scores[None, :]
    e = torch.nn.functional.leaky_relu(e, negative_slope=leak)
    neg_inf = torch.finfo(e.dtype).min
    masked = torch.where(mask > 0, e, torch.full_like(e, neg_inf))
    alpha = torch.softmax(masked, dim=1)
    out = alpha @ z
    return (out if activation is None else activation(out)), alpha


def _gat_layer_combined(
    h: Any,
    mask: Any,
    layer_heads: Sequence[tuple[Any, Any]],
    activation: Callable[[Any], Any] | None,
    combine: Literal["concat", "mean"],
) -> tuple[Any, Any]:
    """Every head of one GAT layer (§45.1), combined by concatenation (every hidden layer, so a
    later layer receives all heads' opinions side by side) or by mean (the last layer, so the
    output is logits or an embedding of the width the caller asked for, not ``heads`` times it --
    the original GAT paper's own convention, since chapter 45 does not itself pick one). Returns
    the combined output and the heads' attention matrices averaged into one, for a caller that
    wants a single number per neighbour to report rather than one per head.
    """
    torch = _require_torch()
    outputs = []
    alphas = []
    for weight, attention in layer_heads:
        out, alpha = gat_layer(h, mask, weight, attention, activation=activation)
        outputs.append(out)
        alphas.append(alpha)
    combined = torch.cat(outputs, dim=1) if combine == "concat" else torch.stack(outputs).mean(0)
    mean_alpha = torch.stack(alphas).mean(0)
    return combined, mean_alpha


def init_weights(dims: Sequence[int], method: Method, seed: int, *, heads: int = 1) -> Any:
    """Xavier-uniform parameters for every layer of ``dims`` (§44.3, Figure 44.10; §45.1 for
    ``method="gat"``), leaf tensors ready for an optimiser. ``dims`` is
    ``[H0's width, *hidden, output width]``; layer ``l`` maps ``dims[l] -> dims[l+1]`` for a GCN,
    ``2 * dims[l] -> dims[l+1]`` for GraphSAGE (the extra factor of two is :func:`sage_layer`'s
    concatenation).

    For ``method="gat"``, ``dims[l+1]`` is the layer's *combined* width (what the next layer
    actually receives, matching the other two methods' own ``dims`` contract): every hidden layer
    concatenates its ``heads`` heads, so each head there gets a ``Wl`` of width
    ``dims[l+1] // heads`` -- raising :class:`ValueError` when that does not divide evenly -- and
    the last layer averages its heads instead, so each head there gets the full ``dims[l+1]``
    directly. Each head also gets an attention vector ``al`` of length ``2`` times its own width
    (:func:`gat_layer`'s ``attention``). The return shape is a list of layers, each a list of
    ``heads`` ``(Wl, al)`` pairs -- :func:`_param_list` is what flattens that for the optimiser.
    """
    torch = _require_torch()
    torch.manual_seed(seed)
    if method == "gat":
        gat_weights: list[list[tuple[Any, Any]]] = []
        in_dim = dims[0]
        n_layers = len(dims) - 1
        for layer_index, out_dim in enumerate(dims[1:]):
            is_last = layer_index == n_layers - 1
            if not is_last and out_dim % heads != 0:
                msg = (
                    f"hidden layer {layer_index} has width {out_dim}, not divisible by "
                    f"heads={heads}: a concatenated GAT layer needs dims[l+1] % heads == 0"
                )
                raise ValueError(msg)
            per_head = out_dim if is_last else out_dim // heads
            layer_heads: list[tuple[Any, Any]] = []
            for _head in range(heads):
                weight = torch.empty(in_dim, per_head, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(weight)
                weight.requires_grad_(True)
                attention = torch.empty(1, 2 * per_head, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(attention)
                attention.requires_grad_(True)
                layer_heads.append((weight, attention))
            gat_weights.append(layer_heads)
            in_dim = per_head if is_last else per_head * heads
        return gat_weights
    factor = 1 if method == "gcn" else 2
    weights: list[Any] = []
    for in_dim, out_dim in pairwise(dims):
        weight = torch.empty(factor * in_dim, out_dim, dtype=torch.float64)
        torch.nn.init.xavier_uniform_(weight)
        weight.requires_grad_(True)
        weights.append(weight)
    return weights


def _param_list(weights: Any, method: Method) -> list[Any]:
    """``weights`` as a flat list of leaf tensors, for ``torch.optim.Adam``: :func:`init_weights`
    returns one tensor per layer for a GCN or GraphSAGE, but a nested list of ``(Wl, al)`` pairs
    per head per layer for a GAT, and Adam needs every leaf named individually either way.
    """
    if method == "gat":
        return [tensor for layer in weights for pair in layer for tensor in pair]
    return list(weights)


def _forward(h0: Any, propagation: Any, weights: Any, method: Method) -> Any:
    """``H^L`` after every layer of ``weights``: ReLU between layers, linear (no activation) at
    the last one, so the output is logits for :func:`classify` and a raw embedding for
    :func:`embed_gnn` alike -- the caller decides what to do with it, exactly as §44.3 leaves the
    "final activation function" to the task (a softmax for classification, nothing at all for an
    embedding meant to be dot-producted). ``method="gat"`` follows the same rule -- ReLU between
    layers, nothing at the last -- and concatenates its heads at every layer but the last, which
    it averages (:func:`_gat_layer_combined`); its per-layer attention is discarded here (this is
    the function the training loop calls every epoch) -- see :func:`_forward_with_attention` for
    the one pass that keeps it.
    """
    h, _attention = _forward_with_attention(h0, propagation, weights, method)
    return h


def _forward_with_attention(h0: Any, propagation: Any, weights: Any, method: Method) -> Any:
    """Like :func:`_forward`, but also returns the last GAT layer's attention (heads averaged),
    or ``None`` for ``"gcn"``/``"graphsage"`` where nothing was ever learned to attend with.
    Nothing here is more expensive than :func:`_forward` for the two fixed-attention methods; for
    ``"gat"`` it is the same forward pass, just keeping a value :func:`_forward` throws away.
    """
    torch = _require_torch()
    if method == "gat":
        h = h0
        last = len(weights) - 1
        alpha: Any = None
        for i, layer_heads in enumerate(weights):
            activation = torch.relu if i < last else None
            combine: Literal["concat", "mean"] = "concat" if i < last else "mean"
            h, alpha = _gat_layer_combined(h, propagation, layer_heads, activation, combine)
        return h, alpha
    layer = gcn_layer if method == "gcn" else sage_layer
    h = h0
    last = len(weights) - 1
    for i, weight in enumerate(weights):
        activation = torch.relu if i < last else None
        h = layer(h, propagation, weight, activation)
    return h, None


def _train(
    h0: Any,
    propagation: Any,
    dims: Sequence[int],
    method: Method,
    seed: int,
    epochs: int,
    lr: float,
    loss_fn: Callable[[Any], Any],
    *,
    heads: int = 1,
) -> tuple[Any, list[float]]:
    """Fit ``dims``'s weights by Adam against ``loss_fn(H^L)`` for ``epochs`` steps (§44.3's
    backpropagation, p. 647, "figure out how the loss function is changing given the changes in
    the parameters ... and descend them"). Returns the fitted weights and the loss at every
    epoch, so a caller can see whether training had converged rather than assume it.
    """
    torch = _require_torch()
    if epochs < 1:
        msg = f"epochs must be at least 1, got {epochs}"
        raise ValueError(msg)
    weights = init_weights(dims, method, seed, heads=heads)
    optimizer = torch.optim.Adam(_param_list(weights, method), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(_forward(h0, propagation, weights, method))
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return weights, history


# --------------------------------------------------------------------------- classification


@dataclass(frozen=True, eq=False)
class Classification:
    """One semi-supervised fit (§44.1, §44.3): every node's predicted class and confidence, the
    accuracy on a held-out set the caller supplied (``None`` when none was), and the majority-
    class null that same set would have scored -- the honest floor a GCN's W has to clear before
    "it learned something" is a claim rather than a hope.
    """

    nodes: list[Node]
    classes: list[str]
    probabilities: np.ndarray
    """One row per node in ``nodes`` order, one column per ``classes`` entry, softmax-normalised."""
    method: Method
    epochs: int
    seed: int
    loss_curve: list[float]
    held_out_accuracy: float | None = None
    majority_accuracy: float | None = None
    n_held_out: int = 0
    attention: dict[Node, tuple[tuple[Node, float], ...]] | None = None
    """``method="gat"`` only: each node's own neighbours (self included), ranked by the fitted
    last-layer attention it put on them, highest first (§45.1). ``None`` for ``"gcn"``/
    ``"graphsage"``, which never learned a weight to report -- their propagation is fixed before
    training starts."""
    _index: dict[Node, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {node: i for i, node in enumerate(self.nodes)})

    def predicted(self, node: Node) -> tuple[str, float]:
        """The class :func:`classify` gave the highest probability, and that probability."""
        row = self.probabilities[self._index[node]]
        best = int(np.argmax(row))
        return self.classes[best], float(row[best])

    def distribution(self, node: Node) -> tuple[tuple[str, float], ...]:
        """Every class's probability for ``node``, highest first."""
        row = self.probabilities[self._index[node]]
        return tuple(
            sorted(zip(self.classes, row.tolist(), strict=True), key=lambda pair: -pair[1])
        )

    def top_neighbours(self, node: Node) -> tuple[tuple[Node, float], ...]:
        """``node``'s own row of :attr:`attention`, or ``()`` when there is none to report
        (``method`` was not ``"gat"``, or ``node`` was outside the network the fit ran on)."""
        if self.attention is None:
            return ()
        return self.attention.get(node, ())


def _top_neighbours(
    alpha: Any, nodes: Sequence[Node], node_index: Mapping[Node, int], *, top: int = 5
) -> dict[Node, tuple[tuple[Node, float], ...]]:
    """Every node's row of a GAT attention matrix (§45.1), as its ``top`` neighbours (self
    included whenever the self loop won a share of the softmax) by fitted weight, highest first.
    """
    matrix = alpha.detach().numpy()
    result: dict[Node, tuple[tuple[Node, float], ...]] = {}
    for node in nodes:
        row = matrix[node_index[node]]
        order = np.argsort(-row)[:top]
        result[node] = tuple((nodes[i], float(row[i])) for i in order if row[i] > 0.0)
    return result


def classify(
    graph: nx.Graph,
    labels: Mapping[Node, str],
    *,
    method: Method = "gcn",
    features: tuple[Sequence[str], np.ndarray] | None = None,
    hidden_dims: Sequence[int] = (16,),
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    held_out: Mapping[Node, str] | None = None,
    heads: int = DEFAULT_GAT_HEADS,
) -> Classification:
    """Semi-supervised node classification (§44.1, §44.3): fit a GCN, GraphSAGE or GAT (§45.1)
    on ``labels`` alone, then predict every node in ``graph``, ``labels`` included.

    ``labels`` is the *only* training signal -- every value in it is trusted, so a caller that
    wants to keep borrowed labels out has to filter before calling this (:func:`complete_attribute`
    does). ``held_out``, when given, is a second mapping of node to true value that was *not* in
    ``labels``: it is never trained on, only used to compute ``held_out_accuracy`` and the
    majority-class ``majority_accuracy`` null a real fit has to beat, exactly as §25.2 asks of
    every predictor in this package. Both mappings' values, plus every key of both, define the
    class vocabulary and the node set a caller may ask ``labels`` and ``held_out`` about.

    ``features`` is :func:`build_features`'s ``text`` argument, unpacked in the same call so
    fitting the propagation matrix and the feature matrix to the same node order happens once.

    ``heads`` is read only for ``method="gat"`` (:func:`init_weights`); every other method
    ignores it. A single head is GAT's minimal case (§45.1) and can underfit a split that ``"gcn"``
    or ``"graphsage"`` would not at the same ``epochs``/``lr`` -- the known-answer tests in
    ``test_sna_gnn.py`` need two heads to reach the same floor the other two methods reach with
    one layer's worth of fixed weights. Try 2 or more before trusting a ``"gat"`` fit over its
    siblings at a budget that was tuned for them.
    """
    torch = _require_torch()
    method = _check_method(method)
    if len(labels) < 2:
        msg = f"classify needs at least two labelled node(s), got {len(labels)}"
        raise ValueError(msg)
    classes = sorted({*labels.values(), *(held_out or {}).values()})
    if len(classes) < 2:
        msg = f"classify needs at least two distinct classes among the labels, got {classes}"
        raise ValueError(msg)
    class_index = {value: i for i, value in enumerate(classes)}
    nodes = node_order(graph)
    node_index = {node: i for i, node in enumerate(nodes)}
    missing = sorted(n for n in labels if n not in node_index)
    if missing:
        msg = f"classify got label(s) for node(s) not in the graph: {missing[:5]}"
        raise ValueError(msg)

    feature_matrix, _names, _has_text = build_features(graph, nodes, features)
    propagation = torch.tensor(_propagation_matrix(graph, nodes, method), dtype=torch.float64)
    h0 = torch.tensor(feature_matrix, dtype=torch.float64)
    train_idx = torch.tensor([node_index[n] for n in labels], dtype=torch.long)
    train_targets = torch.tensor([class_index[v] for v in labels.values()], dtype=torch.long)

    def loss_fn(h_final: Any) -> Any:
        return torch.nn.functional.cross_entropy(h_final[train_idx], train_targets)

    dims = [feature_matrix.shape[1], *hidden_dims, len(classes)]
    weights, history = _train(h0, propagation, dims, method, seed, epochs, lr, loss_fn, heads=heads)
    with torch.no_grad():
        logits, attention_matrix = _forward_with_attention(h0, propagation, weights, method)
        probabilities = torch.softmax(logits, dim=1).numpy()

    attention = None
    if attention_matrix is not None:
        attention = _top_neighbours(attention_matrix, nodes, node_index)

    held_out_accuracy: float | None = None
    majority_accuracy: float | None = None
    n_held_out = 0
    if held_out:
        held_nodes = [n for n in held_out if n in node_index]
        n_held_out = len(held_nodes)
        if held_nodes:
            true_majority = Counter(labels.values()).most_common(1)[0][0]
            majority_accuracy = sum(1 for n in held_nodes if held_out[n] == true_majority) / len(
                held_nodes
            )
            correct = sum(
                1
                for n in held_nodes
                if classes[int(np.argmax(probabilities[node_index[n]]))] == held_out[n]
            )
            held_out_accuracy = correct / len(held_nodes)

    return Classification(
        nodes=nodes,
        classes=classes,
        probabilities=probabilities,
        method=method,
        epochs=epochs,
        seed=seed,
        loss_curve=history,
        held_out_accuracy=held_out_accuracy,
        majority_accuracy=majority_accuracy,
        n_held_out=n_held_out,
        attention=attention,
    )


# --------------------------------------------------------------------------- entity-attribute
# --------------------------------------------------------------------------- completion


@dataclass(frozen=True)
class AttributeSuggestion:
    """One entity's suggested value for an attribute, shaped to paste into an extraction
    sidecar's ``attributes:`` block."""

    node: str
    value: str
    confidence: float
    distribution: tuple[tuple[str, float], ...]
    top_neighbours: tuple[tuple[str, float], ...] = ()
    """``method="gat"`` only (§45.1): the entities this suggestion's fitted attention leaned on
    most, node id and weight, highest first. ``()`` for ``"gcn"``/``"graphsage"``."""


@dataclass(frozen=True)
class CompletionReport:
    """§44.1/§44.3's entity-attribute completion, over one persona's entities network."""

    key: str
    method: Method
    chapter: str
    sampling_frame: str
    n_own: int
    n_borrowed: int
    n_unlabelled: int
    n_held_out: int
    held_out_accuracy: float | None
    majority_accuracy: float | None
    classes: list[str]
    epochs: int
    seed: int
    suggestions: list[AttributeSuggestion]
    notes: list[str]


def _completion_notes(hidden_dims: Sequence[int]) -> list[str]:
    """§44.2's limits, restated next to this specific run's own ``--hidden-dims``."""
    notes = []
    if len(hidden_dims) >= 4:
        notes.append(
            f"{len(hidden_dims)} hidden layer(s) were asked for. §44.2's own unfolded example "
            "(Figures 44.2, 44.5) shows every node's embedding converging to the same vector "
            "within a handful of layers; check the loss curve before trusting a suggestion that "
            "needed this many hops."
        )
    return notes


def complete_attribute(
    graph: nx.Graph,
    key: str,
    *,
    method: Method = "gcn",
    features: tuple[Sequence[str], np.ndarray] | None = None,
    hidden_dims: Sequence[int] = (16,),
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    val_share: float = 0.2,
    heads: int = DEFAULT_GAT_HEADS,
) -> CompletionReport:
    """Suggest ``key`` for every entity that carries no value of its own, from the entities this
    persona's extraction sidecars already tagged and the structure of ``graph`` (§44.1, §44.3;
    §45.1 for ``method="gat"``).

    **Only entity-owned values are training signal.** ``key``'s borrowed values
    (:func:`graphrag.sna.attributes.attribute_sources`) are the majority vote of an entity's own
    documents (:func:`graphrag.sna.export._label_attributes`); training a classifier on them
    would fit the projection's own circularity, exactly the confound
    :class:`graphrag.sna.attributes.AttributeReport` already refuses to certify as an
    assortativity finding, one step earlier. Suggestions are only ever made for the unlabelled --
    a node with an owned value already has one, and a node with a borrowed one has the documents
    that produced it to read, not a model's guess.

    ``val_share`` of the owned labels are withheld before fitting (seeded by ``seed``) so
    ``held_out_accuracy`` and ``majority_accuracy`` in the returned report measure this run
    rather than assume it worked; when withholding would leave fewer than two classes to train
    on, nothing is withheld and both come back ``None``.

    ``method="gat"`` also fills each :class:`AttributeSuggestion`'s ``top_neighbours``: the
    entities the fitted attention leaned on most for that prediction (§45.1), ``heads`` of them
    averaged into one ranking. The other two methods never learn a per-neighbour weight, so their
    suggestions leave it empty. A single head is GAT's minimal case and can underfit where
    ``"gcn"``/``"graphsage"`` would not at the same ``epochs``/``lr`` -- see :func:`classify`'s
    own ``heads`` note; try 2 or more before trusting a ``"gat"`` suggestion over its siblings.

    Raises :class:`ValueError` below :data:`MIN_OWN_LABELS` owned values or below two classes
    among them: a suggestion fitted on fewer numbers than that is noise dressed as a finding.
    """
    method = _check_method(method)
    labels_all = attribute_labels(graph, key)
    sources = attribute_sources(graph, key)
    own = {node: value for node, value in labels_all.items() if sources.get(node) == OWN}
    borrowed = len(labels_all) - len(own)
    classes_available = sorted(set(own.values()))
    if len(own) < MIN_OWN_LABELS or len(classes_available) < 2:
        msg = (
            f"{key!r}: {len(own)} entit{'y' if len(own) == 1 else 'ies'} own a value for this "
            f"attribute, across {len(classes_available)} class(es); complete_attribute needs at "
            f"least {MIN_OWN_LABELS} owned labels across at least two classes. Tag more entities "
            "in the extraction sidecars, or choose a less sparse attribute."
        )
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    hold_count = max(1, round(len(own) * val_share)) if val_share > 0 else 0
    held_nodes = set(rng.sample(sorted(own), hold_count)) if hold_count else set()
    train_labels = {n: v for n, v in own.items() if n not in held_nodes}
    held_labels = {n: v for n, v in own.items() if n in held_nodes}
    if len(set(train_labels.values())) < 2:
        train_labels, held_labels = own, {}

    fit = classify(
        graph,
        train_labels,
        method=method,
        features=features,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        seed=seed,
        held_out=held_labels or None,
        heads=heads,
    )
    unlabelled = [n for n in fit.nodes if n not in labels_all]
    suggestions = [
        AttributeSuggestion(
            node=node,
            value=fit.predicted(node)[0],
            confidence=fit.predicted(node)[1],
            distribution=fit.distribution(node),
            top_neighbours=fit.top_neighbours(node),
        )
        for node in unlabelled
    ]
    suggestions.sort(key=lambda s: -s.confidence)

    frame = (
        f"{graph.number_of_nodes():,} entit{'y' if graph.number_of_nodes() == 1 else 'ies'} in "
        f"the network; {len(own):,} own a value for {key!r} and {borrowed:,} borrowed one "
        f"(excluded from training); {len(unlabelled):,} carry none, which is what this suggests "
        "for."
    )
    chapter = "§44.1, §44.3, §45.1" if method == "gat" else "§44.1, §44.3"
    return CompletionReport(
        key=key,
        method=method,
        chapter=chapter,
        sampling_frame=frame,
        n_own=len(own),
        n_borrowed=borrowed,
        n_unlabelled=len(unlabelled),
        n_held_out=fit.n_held_out,
        held_out_accuracy=fit.held_out_accuracy,
        majority_accuracy=fit.majority_accuracy,
        classes=fit.classes,
        epochs=epochs,
        seed=seed,
        suggestions=suggestions,
        notes=_completion_notes(hidden_dims),
    )


def _num(value: float | None, places: int = 3) -> str:
    if value is None:
        return "undefined (no held-out set)"
    return f"{value:.{places}f}"


def render_completion(report: CompletionReport, *, top: int = 25) -> str:
    """The completion as markdown: the sampling frame, the null model, the chapter, then the
    ranked suggestions -- the four things every report section here prints next to its numbers.
    """
    lines: list[str] = [
        f"## Attribute completion: {report.key} ({report.method}, {report.chapter})",
        "",
        f"**Sampling frame.** {report.sampling_frame}",
        "",
        f"**Null model.** Majority class: always predicting the training set's most common "
        f"value for {report.key!r} scores {_num(report.majority_accuracy)} on the "
        f"{report.n_held_out:,} owned value(s) withheld before fitting; this fit scored "
        f"{_num(report.held_out_accuracy)} on the same held-out set. A suggestion is only as "
        "trustworthy as that gap.",
        "",
        "**Implements.** Atlas ch. 44, *Message-Passing & Graph Convolution*: §44.1 for the "
        "message/aggregate/update framing, §44.3 for the GCN this fit either is or extends "
        "(GraphSAGE's mean aggregator -- see `sna.gnn`'s module docstring)."
        + (
            " §45.1's GAT learns a per-neighbour attention weight instead of GCN's fixed "
            "normalisation; the *leaned on* column below is that fit's own last-layer attention, "
            "not a fact about the entities independent of it -- see `sna.guide`'s GAT rule."
            if report.method == "gat"
            else ""
        )
        + f" {len(report.classes)} class(es): {', '.join(report.classes)}.",
        "",
        f"**Suggestions.** {len(report.suggestions):,} unlabelled entit"
        f"{'y' if len(report.suggestions) == 1 else 'ies'}, ranked by confidence"
        + (f", top {top} shown" if len(report.suggestions) > top else "")
        + ":",
        "",
    ]
    show_attention = any(s.top_neighbours for s in report.suggestions)
    if show_attention:
        header = f"| entity | suggested {report.key} | confidence | runner-up | leaned on |"
        lines += [header, "|---|---|---|---|---|"]
    else:
        header = f"| entity | suggested {report.key} | confidence | runner-up |"
        lines += [header, "|---|---|---|---|"]
    for suggestion in report.suggestions[:top]:
        runner_up = (
            f"{suggestion.distribution[1][0]} ({suggestion.distribution[1][1]:.3f})"
            if len(suggestion.distribution) > 1
            else "-"
        )
        row = (
            f"| {suggestion.node} | {suggestion.value} | {suggestion.confidence:.3f} | "
            f"{runner_up} |"
        )
        if show_attention:
            leaned_on = (
                ", ".join(f"{n} ({w:.2f})" for n, w in suggestion.top_neighbours[:3])
                if suggestion.top_neighbours
                else "-"
            )
            row += f" {leaned_on} |"
        lines.append(row)
    lines.append("")
    lines.append(
        "Nothing above was written to the graph. Paste a row into the entity's extraction "
        "sidecar under `attributes:` to make it real, or discard it."
    )
    if report.notes:
        lines += ["", *report.notes]
    return "\n".join(lines)


def completion_payload(report: CompletionReport) -> dict[str, Any]:
    """The same completion as JSON."""
    return {
        "chapter": report.chapter,
        "key": report.key,
        "method": report.method,
        "sampling_frame": report.sampling_frame,
        "n_own": report.n_own,
        "n_borrowed": report.n_borrowed,
        "n_unlabelled": report.n_unlabelled,
        "n_held_out": report.n_held_out,
        "held_out_accuracy": report.held_out_accuracy,
        "majority_accuracy": report.majority_accuracy,
        "classes": report.classes,
        "epochs": report.epochs,
        "seed": report.seed,
        "suggestions": [
            {
                "node": s.node,
                "value": s.value,
                "confidence": s.confidence,
                "distribution": [{"value": v, "probability": p} for v, p in s.distribution],
                "top_neighbours": [{"node": n, "weight": w} for n, w in s.top_neighbours],
            }
            for s in report.suggestions
        ],
        "notes": report.notes,
    }


# --------------------------------------------------------------------------- link prediction
# --------------------------------------------------------------------------- (§44.3, via §25.1-2)


def embed_gnn(
    graph: nx.Graph,
    *,
    method: Method = "gcn",
    features: tuple[Sequence[str], np.ndarray] | None = None,
    hidden_dims: Sequence[int] = (16, 16),
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    negatives: float = 1.0,
    heads: int = DEFAULT_GAT_HEADS,
) -> Embedding:
    """A GCN/GraphSAGE/GAT node embedding fit by reconstructing ``graph``'s own edges (Kipf and
    Welling's graph autoencoder -- see this module's docstring for the citation and why it is
    not itself in ch. 44). Satisfies :class:`graphrag.sna.embed.Embedding`'s contract, so it
    composes with everything ATL-42 already built.

    The training signal is ``graph``'s own adjacency: for every edge, a negative pair sampled
    uniformly from the non-edges (``negatives`` per positive, seeded), the loss is binary
    cross-entropy between the dot product of the two endpoints' fitted embeddings and 1 (edge) or
    0 (non-edge). This is unsupervised in the sense §44.3 does not need -- it never sees an
    attribute -- but it is still a ``T`` a GCN's ``W`` fits by backpropagation exactly as the
    chapter describes; the graph is both the input and the target.

    ``method="gat"`` fits §45.1's attention the same way -- ``heads`` of them, concatenated at
    every hidden layer and averaged at the last -- but its attention is not surfaced here: an
    embedding is a vector to dot-product, not a per-node prediction, so there is nothing to say a
    neighbour was "leaned on" for. See :func:`complete_attribute` for the reporting path.
    """
    torch = _require_torch()
    method = _check_method(method)
    nodes = node_order(graph)
    node_index = {node: i for i, node in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in graph.edges if u in node_index]
    if not edges:
        msg = "embed_gnn needs at least one edge to reconstruct"
        raise ValueError(msg)
    n = len(nodes)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    existing = {frozenset((u, v)) for u, v in edges}
    n_negative = max(1, round(len(edges) * negatives))
    negative_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(negative_pairs) < n_negative and attempts < 50 * n_negative + 100:
        attempts += 1
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v or frozenset((u, v)) in existing:
            continue
        negative_pairs.append((u, v))
        existing.add(frozenset((u, v)))

    feature_matrix, _names, _has_text = build_features(graph, nodes, features)
    propagation = torch.tensor(_propagation_matrix(graph, nodes, method), dtype=torch.float64)
    h0 = torch.tensor(feature_matrix, dtype=torch.float64)
    pos_u = torch.tensor([u for u, _ in edges], dtype=torch.long)
    pos_v = torch.tensor([v for _, v in edges], dtype=torch.long)
    neg_u = torch.tensor([u for u, _ in negative_pairs], dtype=torch.long)
    neg_v = torch.tensor([v for _, v in negative_pairs], dtype=torch.long)
    targets = torch.cat(
        [
            torch.ones(len(edges), dtype=torch.float64),
            torch.zeros(len(negative_pairs), dtype=torch.float64),
        ]
    )

    def loss_fn(h_final: Any) -> Any:
        pos_logits = (h_final[pos_u] * h_final[pos_v]).sum(dim=1)
        neg_logits = (h_final[neg_u] * h_final[neg_v]).sum(dim=1)
        logits = torch.cat([pos_logits, neg_logits])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    dims = [feature_matrix.shape[1], *hidden_dims]
    weights, _history = _train(
        h0, propagation, dims, method, seed, epochs, lr, loss_fn, heads=heads
    )
    with torch.no_grad():
        vectors = _forward(h0, propagation, weights, method).numpy()

    chapter = (
        "§44.3, §45.1 (self-supervised, via Kipf & Welling 2016; not in the book)"
        if method == "gat"
        else "§44.3 (self-supervised, via Kipf & Welling 2016; not in the book)"
    )
    return Embedding(
        vectors=vectors,
        nodes=nodes,
        method=f"gnn-{method}",
        provenance={
            "chapter": chapter,
            "hidden_dims": list(hidden_dims),
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "heads": heads,
            "n_positive": len(edges),
            "n_negative": len(negative_pairs),
        },
    )


def gcn_link_scorer(
    *,
    method: Method = "gcn",
    features: tuple[Sequence[str], np.ndarray] | None = None,
    hidden_dims: Sequence[int] = (16, 16),
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    heads: int = DEFAULT_GAT_HEADS,
) -> Callable[[nx.Graph], ScoreTable]:
    """A :func:`graphrag.sna.experiment.evaluate_predictor` scorer: fits :func:`embed_gnn` on the
    training graph alone (never the held-out positives, exactly as §25.1 requires of every
    scorer here) and scores a candidate pair by the dot product of the two fitted embeddings.
    ``method="gat"`` and ``heads`` are :func:`embed_gnn`'s own.

    Returns a callable rather than a :class:`ScoreTable` directly because ``evaluate_predictor``
    calls its scorer with the split's training graph, which is only known at evaluation time.
    """

    def scorer(train: nx.Graph) -> ScoreTable:
        embedding = embed_gnn(
            train,
            method=method,
            features=features,
            hidden_dims=hidden_dims,
            epochs=epochs,
            lr=lr,
            seed=seed,
            heads=heads,
        )

        def score(u: str, v: str) -> float:
            return float(np.dot(embedding.vector(u), embedding.vector(v)))

        return ScoreTable(
            name=f"gnn-{method} (dot product)",
            section="§44.3 (via §25.1-25.2)",
            graph=train,
            score=score,
        )

    return scorer
