# Wave brief: `<wave-name>` (template)

**How to use this template.** Copy this file to `personas/<persona>/WAVE_<n>_BRIEF.md` and
replace every placeholder (`<persona>`, `<market>`, `<tool>`, `<competitor>`, `<segment>`,
`<role>`, researcher labels, source names, terms) with the real ones for your wave. Delete this
note once the copy is filled in. Read `personas/<persona>/RESEARCH_BRIEF.md` first; its file
format, provenance manifest and rules all apply here too. This brief only adds the wave's goal,
the sources that work for it, and what each researcher owns.

## Why this wave exists

State the specific gap in the corpus this wave closes, in one or two sentences: what the corpus
already covers, what it is missing, and why that missing piece matters to the question the persona
exists to answer. A wave earns its place by naming a gap, not by "more research is always good" —
if you cannot say what question the corpus cannot yet answer, the wave is not ready to run.

Example shape: "The corpus has vendor material on `<tool>` but nothing from the people who use it
day to day. To understand adoption we need what `<role>` actually say, unprompted, about the tools
they use and why they like or dislike them, and who those people are as recurring types."

State any scope rule plainly, e.g. **real voice only**: no vendor pages, no vendor case studies, no
marketing material, no SEO listicles — first-person accounts from people who use the thing, not
material produced by people who sell it.

## Sources that work

List generic categories rather than one fixed list, since the right sources differ by market:

- **Public discussion forums and community boards** where practitioners talk to each other.
  Good ones page by date and show real usernames or handles with visible history; treat a thread
  with no reply and no visible author history with more caution than one with an ongoing exchange.
- **Review or ratings sites with verbatim reviews.** Good ones let you read the reviewer's own
  words attached to a rating; a site that only shows an aggregate score with no text behind it is
  not useful here.
- **App or product store reviews**, when the thing being discussed is a piece of software with a
  storefront listing.
- **Video or audio commentary made by practitioners**, when the description or transcript is
  reachable even if the comments are not.
- **Long-form posts or blogs written by people who do the work**, as opposed to people who sell
  tools to people who do the work.

How to tell a good source from a bad one: a good source is first-person, dated, and attributable
(even a pseudonymous handle is attribution enough); a bad source is aggregated, anonymous with no
history, optimised to rank in search rather than to inform, or produced by a party with something
to sell. When in doubt, prefer the source that would let a reader independently verify the words
are the speaker's own.

Note any access constraints you already know about for this environment — a source that blocks
fetches, a search budget per researcher, a rate limit — so researchers do not waste turns
rediscovering them. See "Fetch workarounds" in `RESEARCH_BRIEF.md` for general patterns.

## Capture rules for this wave

- One file per thread, review, or article. Path under the source-id and topic given to each
  researcher. Slug `YYYY-MM-DD-<publisher>-<short-title>` (`undated-` if unknown).
- Front-matter as the research brief describes, with `publisher` naming the specific community or
  site (not just its category), `document_type` set to whatever value this wave's captures match
  in the existing corpus, `retrieval: web_fetch`, and `content_fidelity: verbatim` when you kept
  the text as written.
- Keep the original post and the substantive replies verbatim, with any score and date the source
  gives you. Trim only off-topic replies. Never paraphrase a quote you intend to rely on later.
- Record, for every thread or item, a short `## Capture notes` section at the end with the fields a
  later synthesis will need. These are the fields a downstream profile is built from, so do not
  skip them:
  - which **segment** of the market the discussion belongs to (use your persona's own categories)
  - which **tools** are named in the discussion
  - the **role** of the poster as best you can tell (a front-line practitioner, a manager or
    owner, someone adjacent commenting from the outside, or unclear)
  - any signal of **experience level** (new versus long-tenured)
  - any signal of **scale or volume** (how much of the activity in question this person does)
  - any signal of **position in a hierarchy** (works alone, reports to someone, has people
    reporting to them, how many hops from the top of a structure)
- Provenance manifest at `data/raw/<persona>/provenance/<wave-id>-<your-label>.jsonl`.
- Target a dozen to twenty captured files per researcher, then one research note.

## Research note for this wave

`document_type: research_note`, `retrieval: agent_synthesis`, `content_fidelity: summary`, in the
`notes/` folder of your source-id. Sections, in order:

1. **What people say about each tool**, one sub-heading per `<tool>`, each with: what they like
   (quoted), what they dislike (quoted), what they would use instead if it disappeared, answered
   from evidence where any exists, and how it is paid for or obtained if that is known.
2. **Recurring types you saw**, described only from evidence: segment, how they work, tools they
   use, scale signals, tenure and volume signals, relationship to any hierarchy above them, what
   they complain about. Give each a working label and list the files it rests on. A handful of
   types per researcher is expected; a later pass will merge them across researchers.
3. **Contradictions and gaps**, including which sources, subforums or search terms you did not get
   to.
4. **Analyst inferences**, separated from anything sourced.

## Researcher assignments

Fill in one row per researcher. Keep each researcher's scope narrow enough that two researchers
are unlikely to capture the same thread.

| Researcher | Source-id / topic | Scope | Target count |
|---|---|---|---|
| `<researcher-1>` | `<source-id>` / `<topic>` | e.g. `<tool>` as `<segment>` practitioners experience it: specific terms to search | 12-20 |
| `<researcher-2>` | `<source-id>` / `<topic>` | | 12-20 |
| `<researcher-3>` | `<source-id>` / `<topic>` | | 12-20 |
| `<researcher-n>` | `<source-id>` / `<topic>` | | 12-20 |

Common rule for all researchers: never name the organisation behind this research as the reason
for a search when the thread is about a tool or a competitor — the goal is the tool's own
reputation in its users' words, not a search biased toward one company's story. Keep segments
separate in every note.

## What the synthesis will do with your work

A follow-on pass will merge the recurring types across every researcher's note into a small set of
profiles: name, segment, how they work, tool stack and why, scale, tenure and volume, position in
any hierarchy and distance from the top of it, what they would lose if their main tool vanished,
and the quotes that ground each attribute. A profile attribute with no verbatim quote behind it
will be discarded, so capture the words, not your summary of them.
