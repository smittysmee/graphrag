# The notebook API

`graphrag.sna` is written to run from `graphrag sna <command>`: a persona id and a handful of
`--flag`s in, a rendered report or a written file out. A notebook wants an object back instead --
something a cell can poke at, plot, or hand to the next cell -- so `graphrag.sna.Network` wraps
the same package as one: `Network.build(...)`, `.analyze()`, `.backbone()`, `.measures()`,
`.evaluate()`, `.predict()`, `.draw()`, `.to_pandas()`. Every one of those calls the exact
function `graphrag sna <command>` already calls, with the exact same arguments, so a number this
prints is never a second implementation of a number the CLI prints -- `tests/unit/
test_sna_facade.py` asserts the two paths agree field for field on the same fixture.

## Build a network

```python
from graphrag.sna import Network

net = Network.build("product-leader", "entities", min_weight=2, types=["person", "company"])
net.describe()          # NetworkDescription: directed/weighted/bipartite, in the book's words
net.graph.number_of_nodes(), net.graph.number_of_edges()
```

`build` is the one call that opens a store: it reads `graphrag.config.Settings` from the
environment, opens the configured Neo4j connection through `graphrag.app.AppContext`, calls
`graphrag.sna.export.build_network` with `persona`, `network` and every other keyword unchanged
(`source_id`, `min_weight`, `types`, `stances`, `facets`, `relation_types`, `since`, `until`,
`where`, `projection`, `backbone`, ...), and closes the connection before returning -- a notebook
cell that builds three networks does not hold three connections open. `where={"region": "north"}`
takes a plain dict here, not the CLI's `--where region=north` string.

## Analyse it

```python
report = net.analyze(method="louvain", seed=1, by="region")
report.summary["nodes"], report.summary["edges"]
report.louvain_result.modularity, report.louvain_result.stability
[g[:3] for g in report.groups]        # a peek at the first three members of each community
```

`report` is a `graphrag.sna.analysis.Analysis`, the same dataclass `graphrag sna analyze` renders
to markdown -- `graphrag.sna.analysis.render_markdown(report)` gives that markdown back, and
`to_payload(report)` gives the JSON `--json` would have written, if the report is going into a
persona's corpus as a citable finding rather than staying in the notebook. Every section the CLI
prints is a field here: `report.density`, `report.degree`, `report.evaluation`,
`report.against_random`, and so on.

## A DataFrame

```python
nodes, edges = net.to_pandas()
nodes.sort_values("degree", ascending=False).head(10)   # or: pandas.DataFrame(nodes), unsorted
```

pandas is not a dependency of this package. With it installed, `to_pandas()` returns two
`DataFrame`s -- one row per node keyed by `id`, one row per edge keyed by `source`/`target`, every
other column an attribute the graph already carries. Without it, the same rows come back as plain
`list[dict]`s, which `pandas.DataFrame(records)` builds identically the day pandas is installed.

## Draw it

```python
from IPython.display import SVG

drawing = net.draw(layout="force", size="betweenness", color="community", seed=1)
SVG(drawing.svg)
```

`drawing` is a `graphrag.sna.draw.DrawResult` (ch. 49-51): the same SVG `graphrag sna draw`
writes to a file, plus `drawing.legend` and `drawing.notes` for whatever the layout or the colour
rule had to say about itself.

## The rest

`net.backbone("noise-corrected")` returns a new `Network` over chapter 27's filtered graph, so it
chains: `net.backbone("mst").analyze()`. `net.measures()` returns `graphrag.sna.facade.Measures`
-- the density block (ch. 12) and the centralization table (§14.8) alone, without grouping
anything. `net.evaluate(communities)` runs chapter 36's whole battery over any partition, not only
one `analyze()` found. `net.predict("cn")` scores chapter 23's neighbourhood family and Katz over
the graph alone (not `hrg`/`rules`, which need more than a graph -- see `Network.predict`'s
docstring). A network built from a file instead of a persona -- `graphrag.sna.read_graph(path)`,
or a Pajek/GEXF/CSV/edge-list a colleague sent -- skips `build` entirely:

```python
from graphrag.sna import Network, read_graph

net = Network.from_graph(read_graph("entities.graphml"))
```

See `docs/SNA.md` for what each network and filter means, and the caveats that hold whichever
way the network was built.
