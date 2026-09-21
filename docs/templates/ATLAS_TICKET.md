# Atlas ticket brief: `<ATL-nn>` (template)

**How to use this template.** The lead copies this into the prompt handed to an
`atlas-implementer` agent, filled in from the ticket's entry in `docs/ATLAS_PLAN.md`. Nothing here
replaces the plan; it is the plan's entry made concrete for one run — the chapter pages, the exact
modules, the known answers. Delete this note once filled.

## Ticket

`<ATL-nn> — <title>` from `docs/ATLAS_PLAN.md`, wave `<n>`. Read that entry first; it is the
contract. This brief adds what an agent needs to start without asking.

## Book

Chapter `<n>` (`<title>`), sections `<n.m>`, `<n.m>`, … Read them with
`uv run python scripts/atlas_chapter.py <n>` and `--section <n.m>`. Pages `<first>-<last>`.

The passages that decide this ticket (quote them, with page numbers, so the implementer knows
which sentences the reviewer will hold the code to):

> `<verbatim quote, p. nnn>`

## What exists

- `<module>`: `<what it holds today that this ticket extends or moves>`
- `<test file>`: `<the tests that must keep passing unchanged>`

## Build

1. `<deliverable, with the module it lives in and the section it implements>`
2. …

## Done when

- `<observable acceptance criterion, ideally a test name and the known answer it asserts>`
- `<the report section / CLI flag that must exist and what it must print>`
- `<the guide.py rule(s) that must exist, by name>`

## Known answers

| graph | quantity | expected | source |
|---|---|---|---|
| `<legendary graph or planted structure>` | `<measure>` | `<value>` | `<book page or paper>` |

## May touch

`<explicit list of files>`. Anything else needs the lead's agreement first — say so in the closing
report rather than touching it.

## Depends on

`<ticket ids already merged that this one builds on>` — confirm they are on your branch's base
before starting (`git log --oneline -20`).

## Defaults you must not change

`<projection scheme, min_weight, CONFOUNDED_SHARE, …>` — propose in the closing report instead.

## Worktree

Branch `atlas/<ATL-nn>` from `<programme branch>`. Run `uv sync --group dev` once. Do not push.
