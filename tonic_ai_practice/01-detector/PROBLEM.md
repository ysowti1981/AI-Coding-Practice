# Problem 01 — De-identification Detector Evaluation

Setup and provenance only. Findings and the ship/no-ship call live in
[EVALUATION.md](EVALUATION.md).

## The problem as handed over

> You're handed a first-pass PII detector that a teammate wrote last week. It
> scores well on their smoke test. A customer wants to run it over a corpus of
> support tickets before handing that corpus to a model-training vendor.

**The deliverable is a decision**: is it safe to ship, and if not, exactly what
is wrong and what to fix first.

A finished answer contains:

- A ship / don't-ship call stated in the customer's terms, not in F1.
- The specific evidence that forces it.
- The one number worth re-measuring on the customer's real data.
- The one fix to do first, and whether it changes the decision.

### Why the use case decides the metric

The corpus goes to a **model-training vendor**. A missed entity is trained into
weights, cannot be retracted, and may be re-emitted by the resulting model. A
spurious redaction costs data quality and is recoverable. The two error types
are not symmetric, so aggregate F1 is the wrong headline and recall on
high-severity entities is the right one.

## Constraints

| Constraint | Detail |
|---|---|
| `detector.py` | **Read-only.** Measure it before improving it; any fix goes in a wrapper. |
| Libraries | stdlib, `pandas`, `scikit-learn`, `datasets`. No PyTorch, `transformers`, `spacy` or `seqeval` — span-level P/R/F1 is written by hand. |
| Network | No calls inside the timed hour. Data is cached as JSONL beforehand. |
| Clock | 60 minutes wall clock. Cut scope rather than extend. |

## The data

**Source:** [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
(209,261 rows, `train` split).

**How the sample was built** — filter first, then sample. Sampling first would
have yielded roughly one-sixth English.

| Step | Value |
|---|---|
| Filter | `language` in `{en, english}` → 43,501 rows |
| Sample | 1,500 rows, `seed=0` (3.4% of the English pool) |
| Cached at | `data/train_en_1500.jsonl` |
| HF blobs | `data/hf_cache/` — deletable once the JSONL exists |

**Reproduce from scratch:**

```bash
python load_data.py          # skips the download if the JSONL already exists
```

### Schema — 9 columns, 3 of them used

| Column | Use |
|---|---|
| `source_text` | **The document.** All offsets index into it. Never modified. |
| `privacy_mask` | **The gold.** Parsed list of `{value, start, end, label}`. |
| `id` | Unique across all 1,500 — used directly as `doc_id`. |
| `target_text` | Pre-masked text with `[LABEL]` placeholders. Unused; a free cross-check. |
| `span_labels` | Same spans as a **JSON string**, plus `O` segments. **Ignored** — redundant with `privacy_mask` and needs `json.loads`. |
| `mbert_text_tokens`, `mbert_bio_labels` | BERT wordpiece tokens and BIO tags. **Ignored** — this work is in character offsets, and transformers are off the table. |
| `language`, `set` | All `en` / all `train` after filtering. No further use. |

### Properties established before measuring

| Property | Value |
|---|---|
| Documents | 1,500 (48–2,408 chars, median 166) |
| Gold spans | 4,688 raw → 4,576 after name-part merge |
| Gold entity types | 56 |
| Documents with zero PII | 0 |
| **Offset round-trip** | **0 mismatches / 4,688** — `source_text[start:end] == value` |
| Overlapping gold spans | 0 |
| Zero-length / duplicate gold spans | 0 / 0 |
| Adjacent gold pairs (`a.end == b.start`) | 89 |

The offset check is load-bearing: it is why the evaluator can work on raw
character offsets with no normalisation.

### Known limit on external validity

The documents are synthetic and template-generated — clean prose, no typos, no
signature blocks, no quoted reply chains, and a deliberately flat type
distribution carrying `BITCOINADDRESS` and `VEHICLEVIN` at rates no real ticket
queue does.

Good for **finding bugs**, which is the task. Not a basis for forecasting
production recall. Any figure depending on the type mix is corpus-specific and
is marked as such in EVALUATION.md.

## Assets

**Provided, unmodified:**

- `detector.py` — 22 lines. Six format regexes (`EMAIL`, `PHONE`, `SSN`,
  `DATE`, `CREDITCARD`, `ZIP`) plus `PERSON_RE`, which matches any two
  consecutive capitalised ASCII words. Patterns run independently and results
  are concatenated: no overlap resolution, no deduplication, output in pattern
  order, spans carry only `{start, end, label}`.

**Written for this problem:**

| File | Purpose |
|---|---|
| `load_data.py` | Download, filter to English, sample 1,500, cache as JSONL |
| `evaluate.py` | The harness — span normalisation, label reconciliation, severity tiering, matcher, bucketing |
| `EVALUATION.md` | Findings and the verdict |
| `PROBLEM.md` | This file |

## Span convention

One dict shape everywhere, gold and predicted alike:

```python
{"doc_id": str, "start": int, "end": int, "label": str, "text": str}
```

Character offsets into the original string, plus two additions made during this
problem: `eval_label` (the shared label space both sides map into) and `parts`
(original gold labels on a merged PERSON span, so severity survives the merge).
