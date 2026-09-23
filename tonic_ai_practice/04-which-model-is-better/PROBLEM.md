# Problem 04 — Which Model Is Better

Setup and provenance only. Findings go in `EVALUATION.md`; the plan goes in
`PLAN.md` when the clock starts.

## The problem as handed over

An email from the VP of Product, forwarded an hour ago:

> Team has been going back and forth on A vs B for two weeks. B has better F1
> on our eval set so I'd like to just go with B and move on. Before we commit,
> can someone sanity-check? I need a recommendation by end of day — **one
> model, not a matrix.**

**The deliverable is a single recommendation, by end of day.**

A finished answer contains:

- **One model named.** "It depends" fails the brief. The team has been
  deadlocked for two weeks; the VP is asking for the deadlock to be broken, not
  described.
- **Whether B's F1 advantage is real** — and whether F1 is the right basis for
  this decision at all.
- **The reasoning in terms the VP can act on**, short enough to read in an
  email reply.
- **What would change the answer**, so the recommendation can be revisited
  without redoing the work.

### Why the framing matters

Three constraints shape this more than the metrics do.

**"One model, not a matrix"** is an explicit rejection of the deliverable that
would be easiest to produce. A per-type comparison table is exactly what two
weeks of team back-and-forth has already generated and failed to resolve.

**"B has better F1"** is a claim to verify, not a premise to accept. It is also
a claim about a single aggregate number, and aggregates hide the shape of the
errors underneath them.

**The models are described as different in kind**, not in quality — A is the
conservative incumbent, B the aggressive challenger. Two models with the same
F1 and opposite error profiles are not interchangeable, and which one is better
depends on what the errors cost. Nobody in the email has said what they cost.

## Constraints

| Constraint | Detail |
|---|---|
| `build_models.py` outputs | **Opaque.** The prediction files are treated as arriving from whoever trained the models. |
| Libraries | stdlib, `pandas`, `scikit-learn`, `datasets`. No PyTorch, no `transformers`. |
| Network | No calls inside the timed hour. Everything cached beforehand. |
| Clock | 60 minutes wall clock. Cut scope rather than extend. |
| Deliverable | One recommendation. Not a matrix. |

### Contamination note

I wrote `build_models.py`, which means I have seen the error profiles that were
deliberately built into each model. **Those parameters are not recorded in this
document, and the file is not to be reopened.** The exercise is to recover them
from the predictions.

The rule for the hour: **every characterisation of a model must be measured
from its prediction file against gold.** If a claim about A or B cannot be
produced from `predictions_a.jsonl`, `predictions_b.jsonl` and
`train_en_2500.jsonl`, it does not go in the report.

Practical test before writing anything down: *could someone who has only the
three JSONL files reach this?* If no, cut it.

## The data

**Source:** [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
— the same corpus as problems 01 and 03, deliberately, so the loader is reused.

| Step | Value |
|---|---|
| Filter | `language` in `{en, english}` → 43,501 rows |
| Sample | 2,500 rows, `seed=0` |
| Cached at | `data/train_en_2500.jsonl` |

**Overlap with earlier problems:** same dataset, filter and seed, so problem
01's 1,500 and problem 03's 2,000 documents are prefixes of this 2,500. Every
document seen before appears here. Harmless — different task — but it means
familiarity with a document is not evidence about these models.

**Reproduce from scratch:**

```bash
python load_data.py       # skips the download if the JSONL exists
python build_models.py    # regenerates both prediction files
```

Both deterministic: the loader at `seed=0`, the model builder at its own fixed
seed.

### Schema

**`train_en_2500.jsonl`** — as problems 01 and 03:

| Field | Meaning |
|---|---|
| `id` | Unique; the join key |
| `source_text` | The document; offsets index into this |
| `privacy_mask` | Gold spans, `{value, start, end, label}`, character offsets |

`target_text`, `span_labels`, `mbert_*`, `language`, `set` are ignored.

**`predictions_a.jsonl` / `predictions_b.jsonl`** — one row per predicted span:

| Field | Meaning |
|---|---|
| `doc_id` | Joins to `id` in the source file |
| `start`, `end` | Character offsets into `source_text` |
| `label` | Predicted entity type |
| `score` | Confidence in `[0, 1]` |

Unlike problem 02, both scores are in the **same units and the same range**, so
they are directly comparable without a transform. The handover describes them
as "roughly calibrated" — **a claim, not a verified fact**, and one worth
testing, since a recommendation that depends on thresholding depends on it.

### What the handover says about each model

Stated characterisations only — **claims, not measurements**:

- **Model A, "the incumbent"** — high precision, conservative.
- **Model B, "the challenger"** — high recall, aggressive.
- **B has better F1 on the team's eval set.**

Each is a testable assertion.

### Deliberately not yet established

**Nothing has been inspected.** No span counts per model, no precision, no
recall, no F1, no per-type breakdown, no calibration check — not even
confirmation that B's F1 is in fact higher.

One risk follows: **the premise is unverified.** If B's F1 advantage does not
reproduce on this sample, the email's starting point is wrong, and that is
itself the answer to "can someone sanity-check?"

## Assets

**Generated before the clock, treated as opaque:**

| File | Contents |
|---|---|
| `data/predictions_a.jsonl` | 7,812 predicted spans — Model A |
| `data/predictions_b.jsonl` | 9,337 predicted spans — Model B |

Span counts are recorded because they are a property of the files, not a
finding — but note that raw count alone says nothing about which model is
better.

**Written for this problem:**

| File | Purpose |
|---|---|
| `load_data.py` | Adapted from `03-…/load_data.py`; `N_SAMPLE` 2000 → 2500 |
| `build_models.py` | Builds both prediction files. Run once. |
| `PLAN.md` | Written at the start of the clock — does not exist yet |
| `evaluate.py` | My work — does not exist yet |
| `EVALUATION.md` | Findings and the recommendation |
| `PROBLEM.md` | This file |

## Span convention

Carried over unchanged:

```python
{"doc_id": str, "start": int, "end": int, "label": str, "text": str}
```

Character offsets into the original string. Problem 01's matching machinery —
exact vs relaxed, the five error buckets, greedy one-to-one alignment — applies
directly here and is the obvious thing to reuse rather than rewrite.
