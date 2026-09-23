# Problem 02 — The Benchmark Is Lying

Setup and provenance only. Findings go in `EVALUATION.md`.

## The problem as handed over

> Your team is picking a text classifier to route incoming documents. Model B
> beats Model A by about two points of macro-F1 on the held-out test set. A PM
> has already written "B ships" in a planning doc.
>
> The Principal SWE is uneasy. Her question: **is that two-point gap a real
> difference in model quality, is it noise, or is it an artifact of the test
> labels?** And underneath it: **if I retrain next month and the gap flips,
> will I know why?**

**The deliverable is a decision plus a diagnosis.** The second question is the
harder one — it asks whether the evaluation is *load-bearing*, not just what it
currently says.

A finished answer contains:

- Which of the three explanations the gap is — real, noise, or label artifact —
  with evidence that separates them rather than asserting one.
- An uncertainty estimate on the gap itself, so "two points" can be read as
  either meaningful or within the margin.
- Where the two models actually disagree, and whether the disagreements are
  cases with defensible ground truth.
- A direct answer to the retrain question: what she would need to watch, and
  what this benchmark cannot tell her regardless.

### Why the framing matters

The PM has already committed in writing. The cost of a wrong answer is not
symmetric with the cost of a slow one: shipping B on a gap that is noise is
recoverable next quarter, but certifying a benchmark that cannot detect its own
failure means every future comparison inherits the flaw silently.

## Constraints

| Constraint | Detail |
|---|---|
| `setup.py` outputs | **Opaque.** The prediction files are treated as though they arrived from someone else. Model internals are not consulted during the hour. |
| Libraries | stdlib, `pandas`, `scikit-learn`, `datasets`. No PyTorch, no `transformers`. |
| Network | No calls inside the timed hour. Everything is cached beforehand. |
| Clock | 60 minutes wall clock. Cut scope rather than extend. |

### A contamination note

I wrote `setup.py`, so I have seen both model configurations. In a real
interview these arrive as opaque prediction files. During the hour, treat
`predictions_a.jsonl` and `predictions_b.jsonl` as the only evidence about
either model, and do not reason from remembered hyperparameters — any
conclusion that depends on knowing B is a character n-gram SVM is a conclusion
that would not survive the real setting.

## The data

**Source:** [`ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) —
120,000 train / 7,600 test news snippets, four topic classes.

| Label | Name |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

**How the sample was built:**

| Step | Value |
|---|---|
| Train sample | 4,000 rows from `train`, `seed=0` |
| Test sample | 2,000 rows from `test`, `seed=0` |
| Cached at | `data/train_4000.jsonl`, `data/test_2000.jsonl` |
| HF blobs | `data/hf_cache/` — deletable once the JSONL exists |

Each row is tagged with its original dataset index *before* shuffling, so
`doc_id` traces back to the source row: `train-91043`, `test-5512`.

**Reproduce from scratch:**

```bash
python load_data.py     # skips any split already cached
python setup.py         # retrains and rewrites both prediction files
```

Both are deterministic at `seed=0`.

### Schema

**`train_4000.jsonl` / `test_2000.jsonl`**

| Field | Meaning |
|---|---|
| `doc_id` | `"{split}-{original index}"`, unique |
| `text` | The document. Headline and snippet, unmodified. |
| `label` | Integer 0–3 as shipped by AG News |
| `label_name` | The name above, denormalised for readability |

**`predictions_a.jsonl` / `predictions_b.jsonl`**

| Field | Meaning |
|---|---|
| `doc_id` | Joins to the test file |
| `pred_label` | Predicted class **name** |
| `score` | Confidence — **different units per model, see below** |

### The scores are not comparable as-is

| Model | `score` is | Range |
|---|---|---|
| A | probability assigned to the predicted class | bounded `[0, 1]` |
| B | one-vs-rest decision value of the predicted class | **unbounded** |

Comparing them directly is meaningless without a transform, and ranking by raw
score compares two different quantities. There is a second defensible reading
of "margin" for B — the gap between the top two decision values — which is a
different number and would give different confidence orderings. The literal
reading is what is stored.

### Deliberately not yet established

Unlike problem 01, **the data has not been inspected.** No class balance, no
length distribution, no label audit, no metrics — not even the macro-F1 gap the
scenario asserts. That is the first phase of the hour, and pre-reading it would
remove the part of the exercise that matters.

One open risk follows from that: **the ~2 point gap is unverified.** If it does
not hold at this sample size and seed, the scenario's premise is wrong and the
hour starts from a false claim.

## Assets

**Generated before the clock, then treated as opaque:**

| File | Contents |
|---|---|
| `data/predictions_a.jsonl` | 2,000 rows — Model A |
| `data/predictions_b.jsonl` | 2,000 rows — Model B |

Model A is TF-IDF word 1–2gram with logistic regression. Model B is TF-IDF
`char_wb` 3–5gram with a linear SVC. Two different inductive biases, so their
errors should not coincide — which is what makes the comparison informative.

**Written for this problem:**

| File | Purpose |
|---|---|
| `load_data.py` | Download, sample, cache as JSONL |
| `setup.py` | Train both models, write prediction files. Run once. |
| `evaluate.py` | My work — does not exist yet |
| `EVALUATION.md` | Findings and the recommendation |
| `PROBLEM.md` | This file |
