# CLAUDE.md

## What this repo is

Timed practice for a 1-hour live technical interview (Tonic.ai, Staff AI/ML
Engineer). Each problem is a classical-NLP data science + model evaluation
task: load messy data, write an evaluator, run an error analysis, make a
ship/no-ship call, fix one thing and measure the delta.

The live interview is screen-shared, on my own laptop, with AI tools allowed.
So this repo is not "write me a solution" — it's rehearsal for a session where
**I have to explain every line out loud while you write it.**

## The one rule that governs everything

If I can't defend a line of code when an interviewer stops me and asks
"why does this do that?", the line is a liability, not an asset.

Optimize for my understanding and speaking ability, not for code quality.

## How to work with me here

- **Small increments.** One function or one cell at a time, then stop. Never
  produce a finished multi-file solution in one shot.
- **Explain before you write.** Two or three sentences on the approach and the
  tradeoff, then the code. I need those sentences — they're what I say out loud.
- **Keep explanations readable at speed.** I read these while talking, so they
  have to scan in seconds, not be studied. One claim per point, in this shape:

  ```
  <claim, one sentence>
    e.g.  <one real case from the data, shown not described>
    say:  "<the sentence I speak, in plain language>"
  ```

  The `say:` line is what comes out of my mouth — write it as speech, not
  prose. Examples are real cases pulled from the real data, never invented. If
  a point needs three paragraphs, it needs a better example instead.

  *Shape, from problem 01:* claim "the detector finds the name but over-reaches"
  → e.g. `pred 'Dear Bud' vs gold 'Bud'` → say *"it swallows the greeting, so
  exact match scores it zero and relaxed scores it a hit."*
- **Front-load the number that decides.** When reporting results, the first
  line is the single figure that moves the decision, with its meaning attached
  in plain units — not the metric name, what it means for the customer. Tables
  support that line; they don't precede it.
- **Flag the judgment calls.** Any time you make a decision I'd have to justify
  (label mapping, tie-breaking, what counts as a match, what to drop), say so
  explicitly instead of burying it. These are the questions interviewers ask.
- **Do the thing I asked, and nothing adjacent.** The verb I use is the scope.
  "Explain this code" means read it and explain it — not run it, not script
  against it, not start the error analysis. "Look at the data" doesn't mean
  write the evaluator. If the next step seems obvious to you, name it in one
  line and stop; I decide when we move phases, because the clock is mine.
- **Don't fix things I didn't ask about.** If you spot a bug elsewhere, name it
  in one line and move on. Scope creep burns the clock.
- **Push back on me.** If my approach is wrong or won't finish in the time left,
  say so directly. Don't implement a bad plan politely.
- **No silent stretches.** Prefer several short turns over one long one, so I
  keep narrating rather than watching you churn.

## Time discipline

Every problem is 60 minutes, wall clock. Rough shape:

| Phase | Budget | Deliverable |
|---|---|---|
| Look at the data | 10 min | Schema understood, label sets reconciled |
| Evaluator | 15 min | Metrics computed, correct at the edges |
| Error analysis | 15 min | A specific claim about a specific failure |
| Decision | 10 min | Ship / don't ship, in the customer's terms |
| One fix | 10 min | Implemented, re-run, delta shown |

If a phase runs over, **cut scope, don't extend the clock.** An evaluator with a
named gap beats a perfect evaluator that left no time for error analysis.
Remind me of the time budget if I'm visibly over.

## Environment constraints

Real ones, matching the interview:

- Python stdlib, `pandas`, `scikit-learn`, `datasets`. That's the baseline.
- **No PyTorch, no `transformers`, no `spacy`, no `seqeval`.** If the task needs
  span-level P/R/F1, I write it myself — that's the point of the exercise.
- Ask before adding any dependency. "This would be three lines with X" is worth
  saying; installing X without asking is not.
- Data is cached locally as JSONL before the clock starts. No network calls
  inside the timed hour.

## Repo layout

One directory per problem at the repo root, numbered and slugged —
`01-detector`, `02-the-benchmark-is-lying`, and so on.

```
NN-short-slug/
  PROBLEM.md       # problem statement, data provenance, assets — write FIRST
  PLAN.md          # the steps, budgets, and cut order — written when the
                   #   clock starts, before the first measurement
  data/            # cached sample + generated assets, gitignored
  load_data.py     # download, filter, sample, cache as JSONL
  setup.py         # builds assets the problem hands me, when they must be
                   #   synthesised rather than provided. Run once, then opaque.
  <provided>.py    # whatever asset I was handed, unmodified
  evaluate.py      # my work
  analysis.ipynb   # or a script — whichever I'm faster in
  EVALUATION.md    # findings and the ship/no-ship call
  POSTMORTEM.md    # written after the clock stops
```

Scripts resolve paths from `__file__`, never from the working directory, so a
problem directory can be renamed or moved without breaking.

**`PROBLEM.md` comes before the clock starts.** Every problem gets one, and it
holds four things and nothing else:

1. **The problem as handed to me**, in the customer's terms, plus what a
   finished answer has to contain.
2. **Constraints** — what is read-only, what libraries are off the table, what
   must be cached before the hour.
3. **The data** — source, how it was filtered and sampled, seed, where it is
   cached, the schema, and which columns are deliberately ignored.
4. **Assets** — what was provided versus what I wrote, and the command to
   reproduce the cache from scratch.

Findings never go in `PROBLEM.md`; they go in `EVALUATION.md`. If I can't say
where a number came from, the provenance section is wrong.

### The pre-clock routine — identical for every problem

1. `mkdir -p NN-slug/data`.
2. **`load_data.py`** — download, filter, sample, cache as JSONL. Fixed seed.
   Skips work if the cache exists. Filter *before* sampling. Tag each row with
   its source index so an ID traces back. Prints counts, never content.
3. **`setup.py`**, only when the problem's assets must be synthesised (trained
   models, prediction files, a "provided" baseline). Run once, deterministic,
   then **treated as opaque** — if a conclusion depends on remembering how the
   asset was built, it would not survive the real setting, where the asset
   arrives from someone else. Note the contamination in `PROBLEM.md`.
4. **`PROBLEM.md`** — written now, before anything is inspected.

**Do not inspect the data during setup.** No class balance, no distributions,
no metrics — not even the one the scenario asserts. Record in `PROBLEM.md` what
was deliberately left unestablished, and flag any premise the scenario states
that has not been verified. Looking early spends the part of the hour that
carries the most credit.

**Do not record the answer.** If the problem statement hints where the
interesting failure lives, keep the hint out of `PROBLEM.md` — write the
question, not the spoiler.

### `PLAN.md` — written when the clock starts, before the first measurement

The plan is a deliverable, not a preamble. Writing it costs five minutes and it
is what stops the hour turning into whichever analysis occurred to me first.
It holds six things:

1. **The steps in dependency order**, each with a time budget and a concrete
   deliverable. A step whose deliverable I can't name is not a step.
2. **The judgment calls each step will force**, flagged before they arrive —
   label mapping, what counts as a match, which denominator, what to drop.
3. **What I'd need to see to believe the headline claim**, written down *before*
   any number exists. Criteria invented after the results are rationalisation,
   and an interviewer can tell.
4. **The cut order.** Which steps produce a defensible call on their own, and
   which are depth. When a phase overruns I cut by this list, not by instinct.
5. **Explicitly out of scope** — what I was told not to do, and what I'm
   choosing not to do. Scope creep is the most common way the hour is lost.
6. **Open questions** that change the plan depending on the answer, asked at
   the top rather than discovered at minute forty.

Update it as the hour goes if the plan changes, and say why it changed —
a plan that quietly matches whatever happened teaches nothing in the
postmortem.

### `EVALUATION.md` — created at the first measurement, not at the end

**Open it as soon as evaluation starts, and write each step's findings into it
as they are produced.** Not at the end, not "once there's a verdict." Three
reasons, all of which have bitten:

- Findings written at the end are reconstructed from scrollback, and
  reconstruction loses the caveat that was obvious at the time.
- If the clock runs out mid-analysis, a document with three steps written up
  and a stated gap is a deliverable. Notes in a terminal are not.
- The verdict is easier to write, and harder to overstate, when the evidence is
  already on the page in front of me.

Mark it **incomplete** while it is, listing which steps have run and which have
not, and rewrite the status line as that changes. An interim document that says
what it does not yet know is a normal working artifact; one that implies
completeness it lacks is not.

**Anything important that gets said in conversation goes in this document.**
If a finding, a number, a caveat or a framing only exists in the terminal
scrollback, it does not exist — nobody reads a transcript, and I cannot narrate
from one. The test: *could a reader who saw none of the session reach the same
conclusion and defend it?*

What it holds, in this order:

1. **The verdict first**, in the customer's terms — before the question, before
   the method. A reader who stops after the first screen should have the answer.
2. **Every result that moved the decision**, with its number and what the number
   means. Not a summary of the analysis — the analysis.
3. **The evidence body**, one section per step, so a reader can follow the chain
   rather than take the verdict on trust.
4. **Judgment calls on the record** — each decision that would change a number,
   with its rationale, including the ones I'd expect to be challenged on.
5. **What was not run**, and whether it could have changed the answer. A report
   that implies completeness it doesn't have is worse than one with a named gap.
6. **How to reproduce** — the exact commands. Every number quoted must come out
   of that run.

Two failure modes to check for before calling it done: a number in the
conclusion that contradicts a number in the body, and a claim in conversation
that never made it into the file. Both are things a reader catches immediately
and neither is recoverable once they do.

Anything labeled "provided" — a model, a detector, a scoring script, a config —
is read-only during the hour. Measure it before improving it, and put any fix
in a wrapper so the baseline stays reproducible.

## Conventions that carry across problems

**Never mutate the source data.** No lowercasing, no whitespace collapsing, no
reordering. Derive everything from the original; the moment the source changes,
offsets and indices stop meaning anything.

**Prove the ground truth before measuring against it.** One cheap assertion
that gold actually refers to what it claims — offsets slice back to the
recorded value, IDs are unique, keys join. If that check fails, every number
downstream is noise. Run it first, report the failure count.

**Reconcile the label spaces before scoring.** Gold and predictions rarely
share a vocabulary. Map both into one explicit space, write the mapping down,
and count what falls outside it as its own category. Anything the system cannot
express is a *structural* miss and must never be silently dropped — that
distinction is usually the biggest finding available.

**Decide the cost model before you see results.** Which errors are expensive,
and why, in the customer's terms. Deciding after the numbers are in is
rationalization, and an interviewer will spot it.

**Every item lands in exactly one bucket, and the buckets sum.** Assert it in
code. If bucket totals don't equal item totals, the evaluator is double-counting.

**Report two strictness levels whenever a match is fuzzy.** A strict one and a
lenient one. The *gap* between them is its own finding, and quoting either
alone misrepresents the system.

**Report the metric in the customer's units.** Model metrics are the means, not
the answer. Name the per-record figure that decides ship/no-ship — for a
redaction problem, the fraction of documents with at least one missed
high-severity entity; for another problem, something else. F1 is never the
headline.

**Name the denominator whenever it moves.** Any preprocessing that changes what
counts as one item changes the denominator. Report the before and after
together, or the numbers stop being comparable.

### When the problem is span / offset extraction

**Span representation.** One dict shape everywhere, gold and predicted alike:

```python
{"doc_id": str, "start": int, "end": int, "label": str, "text": str}
```

Character offsets into the original string.

**Matching modes.** Always report both:
- *exact*: start, end, and label all match
- *relaxed*: any character overlap **and** label matches

The gap between them is the boundary-error rate.

**Error buckets.** Every gold span and every prediction lands in exactly one:
`correct`, `missed`, `spurious`, `boundary`, `type_confusion`. Counts per
entity type, not just aggregate.

**Edge cases the evaluator must handle** (these are where interviewers probe):
duplicate predictions on the same span, overlapping predictions, adjacent
spans, zero-length matches, a gold span matched by two predictions, and a
prediction overlapping two gold spans.

## Anti-patterns to call me on

- Writing the evaluator before looking at 20 real examples.
- Producing a metrics table with no claim attached to it.
- Trying to fix every bug found in the error analysis instead of the one that
  moves the number most.
- Accepting generated code I haven't read.
- Optimizing aggregate F1 when the costs of the two error types differ by
  orders of magnitude.
- Silently dropping labels that don't map cleanly between gold and predictions.

## After each problem

I write `POSTMORTEM.md` myself — not you — covering: what I finished inside the
hour, where I lost time, which bug I missed, and the one thing to do
differently next time. Then you review it and tell me what a strong candidate
would have done differently in that hour.

Goal across problems: fewer hints each time, until the hour runs itself.
