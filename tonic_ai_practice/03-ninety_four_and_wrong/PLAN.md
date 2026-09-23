# Plan — Problem 03

Written at the start of the clock, before any measurement.

**The job:** diagnose, quantify, decide. Everything demonstrable from
`redacted.jsonl` joined against `train_en_2000.jsonl` — per the contamination
rule in `PROBLEM.md`, a defect I cannot show in the output is not a finding.

## Out of scope — stated by the problem

- **Do not improve detection.** It is correct by construction.
- **Do not build a better replacement model.**
- **Do not fix the defects.** Diagnosis, quantification and a call — not
  remediation.

Chosen out of scope as well: no re-annotation of the corpus, no attempt to
measure anything requiring ground truth beyond the gold spans already present.

## What I would need to see to believe "94% coverage means the corpus is safe"

Written before any number exists.

1. **Residual PII in the released text sits at ~6%**, matching the stated
   detector drop rate and nothing more. Materially higher means the figure
   describes a different stage than the one being shipped.
2. **The 6% is distributed across severity**, not concentrated in the types
   that identify a person.
3. **Replacement is structurally sound** — every substitution lands where it
   was meant to, and the released text is well-formed.
4. **The released corpus still supports the research it was released for.**
   Privacy is necessary, not sufficient; an unusable corpus is a failed
   release even if it leaks nothing.

If 1–3 hold and 4 fails, the corpus is safe and worthless. That is still a
do-not-release.

## Steps

| # | Step | Budget | Deliverable |
|---|---|---|---|
| 0 | Join and verify | 5 min | Join asserted total; `original_text` confirmed equal to `source_text` |
| 0.5 | Read 20 documents side by side | 10 min | The analyst's "weird" located, in words |
| 1 | Residual PII | 12 min | Per-document and per-severity leak rate |
| 2 | Replacement integrity | 8 min | Whether substitutions landed where intended |
| 3 | Consistency | 6 min | Distinct replacements per repeated original value |
| 4 | Collisions | 4 min | Distinct real values sharing one fake |
| 5 | Quantify both axes | 6 min | Privacy damage and utility damage, separately |
| 6 | What 94% describes | 5 min | End-to-end coverage beside the stage-1 figure |
| 7 | Decision | 10 min | Release call, defects ranked, path to release |

### Step 0 — Join and verify
Join on `doc_id`, assert totality. Then the check that isn't ceremony: **does
`original_text` equal `source_text`?** If the pipeline did not echo its input
faithfully, every downstream comparison is against the wrong baseline.

### Step 0.5 — Read 20 documents
Original above, redacted below, gold spans marked. The analyst saw something in
seconds. Writing the evaluator before looking is a listed anti-pattern.

### Step 1 — Residual PII
For each gold span, is its original value still present in `redacted_text`?
Offset-independent substring search. Report per type, per severity, and as a
**per-document rate** — the fraction of documents retaining at least one
high-severity value.

*Judgment call:* substring search false-positives on short or common values —
a `CITY` named "Reading", a `FIRSTNAME` that is also a word. Restrict the
headline to high-severity, longer values; report the noisy tail separately.

*Judgment call:* problem 01's `SEVERITY` tiers were built for a
model-training vendor. **The recipient here is a research partner**, where
re-identification risk dominates differently. Reuse only after one look.

### Step 2 — Replacement integrity
- `spans_replaced` against gold span count per document — does the ratio land
  at 94%?
- `len(redacted_text)` against `len(original_text)` — splicing errors surface
  as length drift.
- Do pool values appear intact, or spliced mid-word?

Anything malformed here is visible to the naked eye and is a strong candidate
for what the analyst noticed.

### Step 3 — Consistency
For each original value appearing more than once, how many distinct
replacements did it receive — within a document, then across the corpus.
Simultaneously a **utility** failure (coreference destroyed) and a signal about
how replacement was sampled.

### Step 4 — Collisions
The inverse: how many distinct real entities share one fake value. Pool size
against the count of distinct real values.

The two axes point opposite ways here, which is worth saying plainly:
collisions arguably *help* privacy while **destroying utility**, because the
corpus asserts relationships between people that never existed.

### Step 5 — Quantify both axes
- **Privacy:** per-document residual-leak rate by severity; what a recipient
  could actually recover.
- **Utility:** fraction of documents structurally corrupted or semantically
  unusable.

In the partner's terms, not in F1.

### Step 6 — What 94% describes
Compute end-to-end coverage of the released artifact and put it beside the
94%. Stage-1 coverage is a property of detection; the corpus being shipped is
stage-2 output. **The gap between those numbers is the likely headline, and it
is the question the team did not ask.**

### Step 7 — Decision
Release or not, with defects ranked by **which block release** versus **which
degrade utility** — different conversations, so not one list. Plus what would
have to be true to release, because sign-off is tomorrow and "no" without a
path is not useful.

## Cut order

If the clock runs short: **1 → 6 → 7.** Those three alone produce a defensible
release decision. Steps 2–4 are diagnostic depth — they make the *diagnosis*
good; 1, 6 and 7 make the *call* good.

Drop first: Step 4 (collisions), then Step 3 (consistency), then Step 2
(integrity). Step 0.5 is never cut; it is the cheapest information in the hour.

## Open questions

1. **Severity tiers** — reuse problem 01's as-is, or re-tier for a
   research-partner recipient?
2. **Does "quantify the damage" include a re-identification estimate** — how
   many distinct real individuals are recoverable — or is a leak rate enough?
   The former is more compelling to a partner and costs about five minutes.
