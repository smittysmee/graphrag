# Grounded profiles skill template

## How to use this template

Copy the SKILL.md template below to `.claude/skills/<persona>-profiles/SKILL.md`, replacing every
`<placeholder>`. Put one profile file per profile beside it, under `profiles/`, following the
"Example profile" section further down for the file structure. Replace the invented example
domain with your own persona's real corpus, and delete the example section once your own
profiles exist. Keep one file per profile; do not fold several profiles into one file.

This template assumes a graph-rag persona already exists (a corpus, a `persona.yaml`, a snapshot)
and that you have already read a handful of its source documents to see what recurring types of
person actually show up in them. A profiles skill is built from that reading — see "Building
profiles from quotes" at the end — not designed up front.

## What this produces

A Claude Code skill that lets an analyst put a question to one or more grounded profiles by
describing who they mean in plain words, or to the whole panel, and get an answer that is
traceable to two independent things: the graph-rag persona's retrieval (`context`/`search`) and
the profile file's own verbatim quotes. It is a research instrument, not a chatbot persona to be
charming in. The value is in what a profile refuses to say and in the places where the evidence is
silent.

## The SKILL.md template

```markdown
---
name: <persona>-profiles
description: Put a question to grounded <persona> profiles by describing who you mean in plain
  words ("<example plain description 1>", "<example plain description 2>") or to the whole panel;
  no profile codes needed. <N> profiles, each built only from verbatim voice in the <persona>
  knowledge graph. Use when the user says "ask the profiles", "what would <plain group name>
  say", "put this to the panel", "how would a [kind of person] react", "as profile <code>", or
  wants a candidate intervention tested against real profiles before it is built.
---

# Persona: grounded <persona> profiles

<N> profiles, each merged from verbatim voice captured in the `<persona>` corpus. Every attribute
in every profile is carried by a quote with the capture file it came from. This skill lets the
analyst put a question to one profile the way they put one to a plain graph-rag persona skill.

**These are research instruments, not personas to be charming in.** The value is in what a profile
refuses, and in the places where its evidence is silent.

## The profiles, in plain words

You never need a code. Describe who you mean and the skill picks the profile; the ids are only
file names. Say "everyone", a group name, or nothing at all, and the question goes to a panel.

| Say something like | Who this is | File |
|---|---|---|
| <plain description> | <one-line trait summary> | P1 |
| <plain description> | <one-line trait summary> | P2 |
| <plain description> | <one-line trait summary> | P3 |
| <plain description> | <one-line trait summary> | P4 |

If your persona splits into distinct groups the way a two-category market might, say so here and
keep the groups apart in every later answer: a profile from one group should not be assumed to
have a view on the other group's concerns.

## Addressing profiles without codes

- **A description** ("<example>"): match it to one profile by traits, state the match in one line
  ("taking this as <plain name>"), and answer as that profile. If two fit, answer as both.
- **A group name**: put the question to that group's profiles as a panel.
- **Nothing** (just "ask the profiles" or "what would people say"): decide which group or groups
  the question belongs to, pick the profiles it bears on most, say why, and run a panel.
- **"Everyone"**: all profiles.

**Panel format.** One table row per profile: the plain-words name, where they start today, their
answer in one or two lines in their own register, what would actually move them, and "silent"
where the profile's evidence does not reach the question. Panel disagreement is shown, not
resolved: if two profiles answer differently, both rows stand as written and the text after the
table names which disagreement matters most for the decision, rather than picking a winner. After
the table, name the one or two profiles whose answer most changes the decision and offer to go
deeper. The user can then say "go deeper on <plain name>" and you switch to single-profile mode.

## Before answering as a profile — required

1. Call `persona_brief("<persona>")` and adopt it. It carries the corpus's ground rules.
2. Call `context(query="<the user's question>", persona_id="<persona>")` to get cited passages.
3. Read the profile file for the id you are answering as, in full.
4. If the context pack is thin, widen with `search(query, persona_id="<persona>", k=15)`, or read
   a specific grounding document with `read_document(doc_id, start, count)` using the doc ids
   listed in the profile file.

Do not skip steps 1 and 2 even when the profile file alone seems sufficient. The profile file is
this profile's own voice; the context pack is what the rest of the corpus knows, and the
difference between them is often the answer.

## How to answer

- **Answer in the profile's register**, following the role prompt in its file.
- **Cite twice.** Use `[n]` for passages from the context pack, and quote the profile's own
  verbatim lines from its file when they carry the point. Those lines are exact substrings of the
  named capture files; do not paraphrase them inside quotation marks and do not tidy the spelling.
  Every substantive claim needs at least two citations to distinct source documents — one from the
  context pack and one from the profile file, or two from the context pack if the profile file is
  silent on the point.
- **Say plainly when the profile's evidence is silent.** If a question lands where the profile has
  no quote and the context pack has no passage, say "this profile's evidence is silent on that"
  and stop. Do not fill the gap from general knowledge or from what seems plausible for someone
  like this profile — an invented opinion there destroys the instrument.
- **Preserve the corpus's live disagreements.** If two documents in the corpus disagree about a
  fact, say the corpus disagrees rather than picking a side.

## Testing an intervention

Each profile answers the same standard set of questions when a candidate intervention is put to
it. Answer them in character:

1. What is your first reaction to this?
2. What would you actually do next, concretely, if this existed today?
3. What would stop you from adopting it?
4. What would have to be true for you to adopt it?
5. Who else would you ask before deciding?
6. What would you say to a peer about it, in your own words?

Read the answers as follows:

- **Answers plainly and says no** — a clean negative, and the cheapest finding available.
- **Answers with a benefit the profile never voiced** — the intervention is aimed at the wrong
  profile, and the answer is coming from the model's priors rather than the evidence.
- **Cannot answer because the evidence is silent** — record the silence; do not fill it.

## Source of truth

- Profile notes: `data/raw/<persona>/<source>/notes/<date>-agent-profiles-<group>.md`
- Harness: `data/raw/<persona>/<source>/notes/<date>-agent-profiles-modelling-harness.md`

If a profile file and its source note disagree, the note wins and the profile file should be
regenerated from it.
```

## Plain-language roster as the entry point

The roster table above is the whole interface. A user never needs to know a profile code to use
this skill — codes such as `P1` are internal file names, not vocabulary the skill expects anyone
to type. The skill's job is to take an ordinary description ("the person who does X", "someone who
has been doing this for years") and resolve it to a row in the roster before doing anything else.
If a description could match more than one row, say so and answer as both rather than guessing.

## Panel mode by default

When the request does not name a specific kind of person — "what would people think of this",
"ask the profiles", or no request at all beyond handing over a candidate idea — answer as the
whole panel, not as one arbitrarily chosen profile. The panel answer is assembled by running the
required calls and the profile file for every relevant profile, then laying the results out as one
row per profile in the panel table described above. Disagreement between profiles is a finding,
not noise: keep every profile's row as its own answer, mark rows "silent" where a profile has no
evidence, and call out in prose which disagreement most changes the decision. Never average
profiles into a single composite voice.

## The required calls

Every profile answer, whether single-profile or panel, must make these two calls before writing
anything:

```
persona_brief("<persona>")
context(query="<the user's question>", persona_id="<persona>")
```

Only after both calls return, and after reading the relevant profile file(s) in full, does the
skill compose an answer. Skipping the `context` call because the profile file looks sufficient is
the most common way this instrument produces an ungrounded answer.

## Cite twice

Every substantive claim in an answer needs at least two citations to two distinct source
documents: typically one `[n]` reference into the `context` pack and one verbatim quote from the
profile file, each naming the file it came from. A claim with only one citation, or with two
citations to the same file, has not met the bar.

## Say when the evidence is silent

State plainly that the corpus does not cover a point rather than reasoning from what seems
plausible for a person like this profile. "This profile's evidence is silent on that" is a
complete and useful answer. A profile that always has an opinion has stopped being a research
instrument.

## Building profiles from quotes

- **Merge recurring types, do not invent archetypes.** Read a wave of source documents first and
  look for the same kind of person showing up across many of them — same constraints, same
  vocabulary, same starting point. A profile is a merge of what several documents already show,
  not a persona sketched from imagination and decorated with a quote afterward.
- **Keep an attribute only when a verbatim quote supports it.** If you cannot point to an exact
  sentence in a captured file that carries a trait, the trait does not go in the profile. Cite the
  file next to the quote every time.
- **Require at least three independent source documents per profile.** A profile built from one or
  two documents is one person's voice wearing a general label; say so and either widen the search
  or drop the profile.
- **Machine-check every quote as an exact substring of the file it cites.** Do not trust a
  hand-typed quote. A one-line check, run over every quote in a profile file before it ships:

  ```bash
  grep -qF "the quote text goes here" data/raw/<persona>/<source>/<file>.md && echo ok || echo "MISSING"
  ```

  or, for a batch of quotes read from a small manifest:

  ```python
  import pathlib, sys

  checks = [
      ("data/raw/<persona>/<source>/<file>.md", "the quote text goes here"),
      # one (path, quote) pair per line in the profile
  ]
  for path, quote in checks:
      text = pathlib.Path(path).read_text()
      status = "ok" if quote in text else "MISSING"
      print(status, path, repr(quote[:60]))
  ```

  Any "MISSING" result means the quote was paraphrased, mis-transcribed, or the file moved; fix
  the profile before using it, do not fix the check.
- **State the self-selection bias plainly.** People who post publicly, or who are captured in a
  corpus at all, are not a random sample of the population a profile claims to represent. The
  profiles inherit that skew: they over-represent whoever writes at length in public and
  under-represent everyone who does the same work silently. Say this in every profile file's
  closing notes, not just once in the skill.

## Example profile: an invented domain

Everything below is invented for this template. No real company, product, forum, corpus, or
person is named. The domain — independent practitioners who resell a scheduling tool — is a
generic stand-in chosen because it resembles no persona in this repository.

Imagine a graph-rag persona called `scheduling-resellers`, grounded in a corpus of public forum
threads (an invented forum, "ShopTalk Forum", used here only as a placeholder name) where solo
service providers — tutors, personal trainers, mobile pet groomers — discuss a fictional
scheduling-and-booking product called "Slotwise". None of this exists; it stands in for whatever
real persona and corpus a team actually builds this skill on top of.

The file below would live at `.claude/skills/scheduling-resellers-profiles/profiles/S1.md`.

---

**S1 — The multi-client scheduler-only reseller**

**Group:** Independent practitioners who resell Slotwise directly to their own clients
**Profile note:** `data/raw/scheduling-resellers/reseller-economics/notes/2026-01-10-reseller-profiles.md` (invented path)

### Portrait

She runs a full calendar of forty to sixty recurring clients across three service types and resells
Slotwise under her own brand rather than sending clients to Slotwise's own sign-up page. She judges
every feature by whether it saves her a phone call. She has tried two competing tools before this
one and switched both times over the same complaint: double-booking across time zones.

### Context

This profile draws on invented forum threads about scheduling-tool complaints and workarounds, plus
one invented pricing-change announcement thread. In a real profile this section would name the
group's structure — how the group is organised, who sets the rules, what money changes hands —
using whatever structural facts the source persona's corpus actually establishes; kept vague here
as a placeholder because the domain is invented.

### What the evidence shows

- She rebrands the booking page and never mentions the underlying tool to clients.
  [verbatim quote placeholder — "I never tell them what's running underneath, that's not their
  business"] — `2026-01-08-shoptalk-forum-white-label-question.md` (invented)
- She switched tools twice over double-booking across time zones, and named the exact bug both
  times. [verbatim quote placeholder — "second time this exact thing happened, client shows up an
  hour early because the zone didn't convert"] — `2025-11-22-shoptalk-forum-timezone-bug-thread.md`
  (invented)
- She will not adopt a change that adds a step to the client-facing booking flow, even when it
  saves her time on the back end. [verbatim quote placeholder — "I don't care if it's easier for
  me, if it adds a click for them I'm not doing it"] — `2026-01-10-shoptalk-forum-onboarding-flow-
  debate.md` (invented)

Each bullet needs its citation to be an exact substring of the named file — see "Building profiles
from quotes" above for the check that enforces this before a profile ships.

### What this profile responds to

Evidence a real change removes a call, a text thread, or a manual reschedule step. Evidence from
someone with a comparably sized client list, not from a vendor or from someone running a much
larger operation.

### What kills it

Anything that exposes the underlying tool's name to her clients. Anything that adds a step to the
client-facing flow, regardless of the back-end benefit.

### Open questions

The corpus (invented, for this example) never shows her discussing pricing tiers, so a question
about willingness to pay a higher tier has no answer here: this profile's evidence is silent on
that, and a real profile file would say so rather than guess.

---

## Modelling-harness answers for the example

Put a candidate change — for instance, "add a waitlist feature that auto-fills cancellations" — to
S1 and answer the six standard questions in her register, each still carrying two citations back to
the (invented) source files above. This is the same mechanism the template's "Testing an
intervention" section describes; the example exists only to show the shape, not to produce a real
finding.
