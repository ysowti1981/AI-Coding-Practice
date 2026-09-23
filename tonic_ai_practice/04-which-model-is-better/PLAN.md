# Plan — Problem 04

Written at the start of the clock, before any measurement.

**The job:** one model named, by end of day, with reasoning the VP can act on.
Everything measured from `predictions_a.jsonl`, `predictions_b.jsonl` and
`train_en_2500.jsonl` — per the contamination rule in `PROBLEM.md`.

## Out of scope

- **No third model, no ensemble.** The question is A or B.
- **No threshold tuning to manufacture a winner.** Calibration gets checked;
  it does not get optimised until one model wins.
- **No per-type comparison matrix as the deliverable.** Explicitly rejected in
  the brief, and it is what two weeks of team debate already produced.
- **No re-annotation of gold.**

## What I would need to see to believe "B has better F1, so ship B"

Written before any number exists.

1. **The F1 advantage reproduces** on this sample. If it does not, the email's
   premise is wrong and that is the sanity-check answer.
2. **The advantage survives the matching rule.** Exact and relaxed matching can
   rank two models oppositely when one over-captures and the other
   under-captures. If B wins only on relaxed, the advantage is boundary
   generosity, not better detection.
3. **The advantage is larger than measurement noise** — paired bootstrap on the
   difference, documents as the resampling unit.
4. **F1 is the right objective.** F1 weights precision and recall equally,
   which assumes a miss and a false positive cost the same. For PII they do
   not. If the costs differ materially, the higher-F1 model can be the worse
   model, and F1 being higher is not an argument.
5. **The advantage is not concentrated in low-severity types**, and B's errors
   are not systematically worse *in kind* than A's.

If 1–3 hold and 4 fails, B genuinely has higher F1 and should still not
necessarily ship. That is the outcome worth being ready for.

## Reuse

This problem is largely problems 01 and 02 composed:

- **From `01-detector`** — canonical span shape, severity tiers, greedy
  one-to-one alignment, the five error buckets, exact vs relaxed matching.
- **From `02-the-benchmark-is-lying`** — paired bootstrap on a metric
  difference, documents as the unit.

Rewriting either would be a poor use of the hour.

## Steps

| # | Step | Budget | Deliverable |
|---|---|---|---|
| 0 | Harness, join, offset round-trip | 5 min | Canonical spans for gold, A, B; joins asserted |
| 0.5 | 20 documents, gold vs A vs B | 8 min | The two error profiles described in words |
| 1 | Reproduce the claim | 7 min | P/R/F1 both models, exact and relaxed |
| 2 | Is the gap real | 8 min | Paired bootstrap CI on the F1 difference |
| 3 | Error profile decomposition | 12 min | Five buckets per model, per severity tier |
| 4 | Cost asymmetry | 8 min | Does the ranking flip under a defensible cost model |
| 5 | Calibration | 5 min | Whether score thresholding changes the answer |
| 6 | The recommendation | 7 min | One model, the reasoning, what would change it |

### Step 0 — Harness
Load all three files. Assert every prediction's `doc_id` exists in gold, every
offset lies within its document, and gold offsets slice back to gold values.
Silent join failure would corrupt every number downstream.

### Step 0.5 — Look first
Gold, A and B on the same 20 documents. The brief says the models differ in
kind; this is where that becomes concrete rather than asserted.

### Step 1 — Reproduce the claim
Micro and macro P/R/F1 for both, under exact and relaxed matching.

*Judgment call:* **macro vs micro.** With 56 entity types, macro weights a
23-span type equally with a 600-span type. Micro follows the volume. They can
rank models differently and both are defensible — report both, lead with one,
say why.

*Judgment call:* **does a match require the label to agree?** B is described as
aggressive; if it mislabels correct offsets, label-strict scoring penalises
that and label-blind scoring does not. For a redaction use the offsets matter
more than the label. Report both.

### Step 2 — Is the gap real
Paired bootstrap, resampling documents, both models scored on the same
resample. Reuse problem 02's approach directly.

### Step 3 — Error profile
Five buckets — correct, missed, spurious, boundary, type_confusion — per model,
per severity tier. This is where "conservative" and "aggressive" get numbers.

*Judgment call:* severity tiers reused unchanged from problem 01, for
comparability. Worth one look given this is a different use case.

### Step 4 — Cost asymmetry
F1 assumes a miss and a false positive cost the same. Recompute the comparison
under weightings that reflect what the errors actually do. Report the weighting
at which the ranking flips — that number is more useful to a VP than any single
F1, because it says how much someone would have to believe about costs to
prefer the other model.

### Step 5 — Calibration
Are the scores usable? If B's extra detections carry separable confidence, "B
with a threshold" is a different option from "B". If scores are not calibrated,
that option does not exist and should not be offered.

### Step 6 — Recommendation
One model. The reasoning in a paragraph. What would change it.

## Cut order

If the clock runs short: **1 → 3 → 4 → 6.** Those produce a defensible
single-model recommendation — the claim verified, the error profiles
characterised, the cost question faced, and a call made.

Drop first: Step 5 (calibration), then Step 2 (significance). Step 2 matters
only if the gap is small; Step 3 will reveal whether it is. Step 0.5 is never
cut.

## Open questions

1. **What consumes these predictions?** The corpus is PII, so redaction is the
   obvious assumption, and it sets the cost asymmetry — a miss leaks, a false
   positive over-redacts. The email does not say. **Proceeding on the redaction
   assumption and flagging it**, because the cost model is the crux and an
   unstated assumption there would invalidate the recommendation.
2. **Is there a confidence threshold in deployment, or is every prediction
   acted on?** Determines whether Step 5 is decision-relevant or academic.
