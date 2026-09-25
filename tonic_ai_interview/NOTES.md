# NOTES — NER model on vendor data  (**INCOMPLETE**)

## The question (reader's words)
Two vendors (A: 100 transcripts = `prospect_test_a`, B: 30 notes = `prospect_test_b`) ask:
"How does your model perform on our data before we send the rest?"
We need: per-vendor performance, which labels are problematic, why, and how to fix.

## Plan (cut from the bottom first)
1. Integrity: ids join, offsets round-trip, gold vs preds (evaluate.py)
2. Reproduce "does well on val" → strict + lenient F1 on val, with bootstrap CI
3. Same on vendor A and B, per label; error taxonomy (missed / boundary / wrong label / spurious)
4. Distribution shift train → vendor (distribution_study.py): formats, vocabulary, annotation conventions
5. Link each failing label to a cause; read 3 raw examples before claiming it
6. Fix (post-processing, no retraining) measured with evaluate.py — *cut first if short on time*
7. Retraining recommendation (described, not run)

## What I'd need to see to believe "it does well" transfers to the vendors
- Val F1 reproduces from `preds/val.jsonl` (the headline is our number, not a quote).
- Vendor per-label F1 within the val CI for **every** label, not just micro (micro hides small labels).
- Strict ≈ lenient: if lenient ≫ strict, it's a boundary/annotation-convention gap, not a model miss.
- Vendor entity surface forms (date/amount formats, names, vocab) look like train's.

## Findings
Source: `evaluate.py` (re-run by me, ml_torch venv). Bootstrap = docs resampled, seed 0, 1000 reps.

**F0. Integrity** — joins total on all 3 sets (40/100/30 docs), 0 text diffs, 0 bad offsets.

**F1. Headline reproduced: val strict micro F1 0.962 [0.948, 0.975].** Vendor B 0.931 [0.912, 0.947];
vendor A 0.644 [0.626, 0.664]. Strict ≈ lenient everywhere (≤0.02 gap) → not a boundary story.

**F2. Vendor A drug_amount: model finds 1 in 20 (recall 0.051, F1 0.093 [0.04, 0.14]); 328/356 missed outright.**
  e.g.  va-001 "the lisinopril. [forty milligrams] every morning" → no prediction
  why:  0/356 vendor-A amounts contain a digit — in fact 0/100 vendor-A docs contain ANY digit
        (spoken-form transcripts). Train: 1099/1099 amounts have digits. The model never saw a word-number dose.
  say:  "The model learned 'a dose is a number plus mg'. Vendor A writes 'forty milligrams', so it misses 19 in 20."

**F2b. Same failure on OUR val data → the format is the cause, not vendor A itself** (evaluate.py, drug_amount format section).
  Strict recall with digit vs without: val 0.966 (n=145) vs 0.200 (n=5); vendor B 0.988 (n=86, all digit);
  vendor A — (n=0) vs 0.051 (n=356). Train: 1099/1099 amounts have a digit.
  say:  "Even on our own val set, the 5 amounts written in words are found 1 in 5 times. Vendor A is all such amounts."
  caveat: val no-digit n=5 → directional only.

**F3. Vendor A date: 808 predicted, 0 gold — vendor A labelled no dates at all.**
  e.g.  va-001 "just started [yesterday]" predicted; gold has nothing.
  why:  labelling convention, not model error: our train labels relative dates (70% of 2347 train dates have no digit).
  say:  "Those 808 'errors' are dates we'd count as correct; vendor A just didn't annotate dates. Question for the vendor."

**F4. Vendor B date: precision 0.546, recall 0.962 (F1 0.697) — vendor B labels only absolute dates.**
  e.g.  dev-n-004 "presents [today]", "over [the past month]" predicted, not gold. All 80 gold dates contain a digit.
  say:  "Same convention gap as A, milder: B annotates calendar dates only; we also tag 'today'."

**F5. Vendor A name: recall 0.812, F1 0.874 [0.77, 0.95], n=64 — lowercase names in speech.**
  e.g.  va-035 "NURSE: [rhonda] good to see you" → tagged medical_condition.
  Small n; CI overlaps val's [0.94, 1.00] only barely. Secondary issue.

**F6. Everything else is at val level on both vendors** (drug_name, medical_condition, B's name/amount ≥0.946).
  Without date: A 0.835, B 0.969 (val 0.965). Without date + amount: A 0.957. (ad hoc from results/metrics.csv)

**Judgment calls:** macro includes A's zero-support date label (drags A macro to 0.577); taxonomy is
existence-based (exact > boundary > label_confusion > missed), not 1:1.

### Distribution shift (distribution_study.py, re-run by me) — linked to the errors above
| Shift (train → vendor) | Magnitude | Causes error? |
|---|---|---|
| A: numbers spelled out | amounts 0.0% → 97.5% spelled-out; digits/1k words 68.9 → 0.0 | **Yes → F2** (amount recall 0.051) |
| A: relative dates unlabeled | train-transcript dates 88.2% relative; A has 0 date spans, 356/357 relative phrases unlabeled | **Yes → F3** (808 FP) — convention |
| B: relative dates unlabeled | train-notes dates 42.1% relative; B 0.0%; 34/34 relative phrases unlabeled | **Yes → F4** (63 FP) — convention |
| A: misspelled drug names | 20.4% unseen vs 0.9% val ("sertralene", "atorvastaten") | **No** — drug_name F1 still 0.974; subword tokens cope |
| A: fillers / ASR noise | 13.9 vs 3.2 per 1k words | **No** — condition F1 0.946 ≈ val 0.938 |
| B: more abbreviations, injectable units (ml, UNT) | 24.4% vs 9.6%; ml 52.3% vs 15.4% | **No** — B condition 0.960, amount 0.994 |
Judgment call: a shift only "matters" if the label it touches has a measured error; several big shifts don't.

## Answer (draft — fix not yet measured)
**Decision:** Vendor B: ready to process (0.931 vs val 0.962; the gap is almost all a date-convention mismatch).
Vendor A: **not ready for drug amounts** — the model finds ~1 in 20 doses because vendor A writes numbers in words;
everything else except dates is at val level.
**Evidence:** (1) A amount recall 0.051, 0/356 gold amounts contain a digit vs 1099/1099 in train.
(2) A has zero gold dates, so all 808 date preds are FP (top: today, yesterday, this morning); B's 63 spurious
date preds: 46/63 contain no digit (today, in four weeks, the past month) — B annotates none (0/80 gold relative).
Lower bound: digit-bearing relatives like "in 1 month" are among the other 17. Like-for-like control: val NOTES
label relative dates (31/100 gold dates, e.g. dev-n-002 "diagnosed [two years ago]"); B's same phrasing is unlabelled.
(3) drug_name / medical_condition within ~0.03 of val on both vendors despite misspellings and noise.
**What would change it:** vendors saying relative dates *are* in scope (then date isn't an error at all; re-score).
**Fix first:** (a) agree date guideline with vendors, or drop relative-date preds for them (post-filter, no retrain);
(b) spelled-number → digit normalisation before inference ("forty milligrams" → "40 milligrams"), map offsets back;
(c) retrain with spoken-form augmentation (numbers-as-words, lowercase names, misspellings) — the real fix for A.
