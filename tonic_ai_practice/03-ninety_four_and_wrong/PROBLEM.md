# Problem 03 — Ninety-Four and Wrong

Setup and provenance only. Findings go in `EVALUATION.md`.

## The problem as handed over

> Your team's pipeline de-identified a corpus that's about to be released to an
> external research partner. The detection eval says **94% entity coverage**,
> which the team considers acceptable for this data class. Sign-off is
> scheduled for tomorrow.
>
> An analyst who glanced at the output says it **"looks weird"** but can't
> articulate why. That's all you have.
>
> Your job: figure out what's wrong, decide whether the corpus can be released,
> and quantify the damage in terms the partner would care about.

**The deliverable is a diagnosis, a release decision, and a damage estimate.**

A finished answer contains:

- The analyst's "weird" made specific — what is actually wrong, enumerated, each
  claim tied to evidence in the output rather than to a reading of the code.
- A release / do-not-release call, with the reasoning stated in terms of what
  happens if the corpus goes out as-is.
- Damage quantified on **both axes the partner cares about**:
  - **Privacy** — what a recipient could recover about real people.
  - **Utility** — whether the corpus still supports the research it was
    released for. A corpus can be perfectly private and worthless.
- Whether the 94% coverage figure means what the team thinks it means.

### Why the framing matters

Sign-off is tomorrow, and the only signal is one person's unease. The failure
mode to avoid is answering the question that was asked — "is 94% enough?" —
when the number may be measuring the wrong stage entirely. Detection coverage
is a property of stage 1. The corpus being released is the output of stage 2.

Release is also close to irreversible: once a corpus is in a research partner's
hands, a recall is a request, not a control.

## Constraints

| Constraint | Detail |
|---|---|
| `pipeline.py` | **Provided and opaque.** Treated as a teammate's code. Diagnose from the output, not by reading the implementation. |
| Libraries | stdlib, `pandas`, `scikit-learn`, `datasets`. No PyTorch, no `transformers`. |
| Network | No calls inside the timed hour. Everything cached beforehand. |
| Clock | 60 minutes wall clock. Cut scope rather than extend. |

### A contamination note — stronger than usual

**I wrote `pipeline.py`.** In a real engagement it arrives from a teammate and
its defects are unknown. Here they are not, which makes this the most
compromised setup of the three problems so far.

The discipline for the hour: **every claim must be demonstrated from
`redacted.jsonl` against `train_en_2000.jsonl`.** A defect I "know" about but
cannot show in the output does not go in the report. If a finding can only be
justified by pointing at a line of `pipeline.py`, it is not a finding — it is a
memory, and it would not survive the real setting where nobody can read the
teammate's intent.

Practical test when writing up: *could an analyst who has never seen the source
reproduce this from the two JSONL files alone?* If no, cut it.

## The data

**Source:** [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
— the same corpus as problem 01, deliberately, so the loader is reused rather
than rewritten.

| Step | Value |
|---|---|
| Filter | `language` in `{en, english}` → 43,501 rows |
| Sample | 2,000 rows, `seed=0` |
| Cached at | `data/train_en_2000.jsonl` |
| HF blobs | `data/hf_cache/` — deletable once the JSONL exists |

**Overlap with problem 01, verified:** same dataset, same filter, same seed, so
problem 01's 1,500-document sample is exactly the first 1,500 rows of this
2,000. All 1,500 appear here. Harmless — different task — but it means any
intuition carried over from problem 01 is about *these same documents*, which
is worth knowing before treating a familiar-looking example as fresh evidence.

**Reproduce from scratch:**

```bash
python load_data.py     # skips the download if the JSONL already exists
python pipeline.py      # regenerates data/redacted.jsonl
```

Both deterministic: the loader at `seed=0`, the pipeline at its own fixed seed.

### Schema

**`train_en_2000.jsonl`** — as problem 01. The three columns that matter:

| Field | Meaning |
|---|---|
| `id` | Unique; the join key to the redacted output |
| `source_text` | The original document. Ground truth for what was in it. |
| `privacy_mask` | Gold spans, `{value, start, end, label}`, character offsets |

`target_text`, `span_labels`, `mbert_text_tokens`, `mbert_bio_labels`,
`language` and `set` are ignored, as in problem 01.

**`redacted.jsonl`** — the artifact under review:

| Field | Meaning |
|---|---|
| `doc_id` | Joins to `id` in the source file |
| `original_text` | The input document |
| `redacted_text` | What would be released |
| `spans_replaced` | Count of spans the pipeline reports it substituted |

Note `spans_replaced` is the **pipeline's own count of its work**, not an
independent measurement. Whether it corresponds to anything observable in
`redacted_text` is a question, not an assumption.

### What the pipeline claims to do

Stated behaviour, as handed over — **claims, not verified facts:**

1. **Stage 1, detection.** Uses the gold spans directly as detector output,
   then drops 6% at random, giving the 94% coverage the eval reports. Detection
   is deliberately a solved problem here.
2. **Stage 2, replacement.** Each surviving span is substituted with a fake
   value drawn from a small per-type pool — names, emails, dates, phones,
   cities, SSNs — sampled independently per occurrence, and spliced into the
   text by character offset, left to right.

Each of those sentences is a testable assertion about the output.

### Deliberately not yet established

**The output has not been inspected.** Not the redacted text, not the
`spans_replaced` distribution, not a single document diffed against its source.
Nothing beyond confirming the file has 2,000 rows and the four expected keys.

The analyst's "looks weird" is the entire starting signal, and finding what
they saw is the first phase of the hour.

## Assets

**Provided, treated as opaque:**

| File | Contents |
|---|---|
| `pipeline.py` | The two-stage de-identification pipeline under review |
| `data/redacted.jsonl` | 2,000 documents — the corpus proposed for release |

**Written for this problem:**

| File | Purpose |
|---|---|
| `load_data.py` | Adapted from `01-detector/load_data.py`; `N_SAMPLE` 1500 → 2000 |
| `evaluate.py` | My work — does not exist yet |
| `EVALUATION.md` | Findings and the release decision |
| `PROBLEM.md` | This file |

The loader is a copy rather than a shared import because `01-detector` is not a
legal Python module name. A shared `utils` package would be the correct fix and
is deliberately out of scope.

## Span convention

Carried over from problem 01, unchanged:

```python
{"doc_id": str, "start": int, "end": int, "label": str, "text": str}
```

Character offsets into the **original** string. Offsets into `redacted_text`
are a different coordinate system and any comparison across the two has to say
which one it is using.
