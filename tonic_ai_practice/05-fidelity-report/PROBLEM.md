# Problem 05 — The Fidelity Report

Setup and provenance only. Findings go in `EVALUATION.md`; the plan goes in
`PLAN.md` when the clock starts.

## The problem as handed over

Two emails, same morning, from two people who want different things.

**From the customer's data science lead:**

> Fidelity report came back green — every column's distribution matches the
> real data almost exactly, KS statistics all under 0.02. My team wants to
> start training on this next week. Anything blocking?

**From the customer's privacy officer:**

> I don't read statistics. I need to know one thing: can any individual in our
> real dataset be identified from this synthetic file? Yes or no.

**There are two deliverables, for two audiences, in two formats.**

A finished answer contains:

- **A yes or no for the privacy officer**, in that form, in plain language.
  Not a p-value, not a distance distribution. If the answer is yes, it needs
  to be accompanied by the evidence a non-statistician can check themselves.
- **A training go / no-go for the data science lead**, which is a separate
  question with a separate answer.
- **Whether "KS all under 0.02" means what they believe it means.** It is the
  only evidence either of them has, and it is doing a great deal of work in
  both emails.

### Why the framing matters

**The two questions are independent.** A file can be safe and useless, or
unsafe and useful. Answering one does not answer the other, and the emails
assume a single green light covers both.

**KS is a per-column marginal test.** Whatever it establishes, it establishes
one column at a time. Whether that is sufficient evidence for either question
is the thing to determine, not assume.

**The privacy officer asked a binary question and pre-emptively declined
statistics.** An answer that hedges into distributions fails the brief even if
every number in it is correct. If the answer is yes, one concrete example is
worth more than any aggregate.

## Constraints

| Constraint | Detail |
|---|---|
| `synthesize.py` | **Opaque.** Treated as a vendor artifact. The method is not documented in the handover and is not to be inferred by reading the source. |
| Libraries | stdlib, `pandas`, `scikit-learn`, `scipy`. |
| Network | No calls inside the timed hour. Everything cached beforehand. |
| Clock | 60 minutes wall clock. Cut scope rather than extend. |
| Deliverables | A yes/no answer **and** a training recommendation. |

### Contamination note

I wrote `synthesize.py`. In a real engagement it is a vendor's product and its
method is unknown.

**The rule for the hour: every claim must be demonstrable from the three CSV
files.** If a property of the synthetic data cannot be shown by comparing
`synthetic.csv` against `real_train.csv` and `real_holdout.csv`, it does not go
in the report.

The test before writing anything down: *could the privacy officer's own
analyst reproduce this from the three files?* If no, cut it.

## The data

**Source:** the Adult census dataset (`fetch_openml("adult", version=2)`) —
the canonical re-identification benchmark, chosen deliberately.

| Step | Value |
|---|---|
| Fetched | 48,842 rows × 15 columns |
| Sampled | 20,000 rows, `seed=0` |
| Split | 15,000 **real train** / 5,000 **real holdout** |
| Synthetic | 15,000 rows generated from real train |

**Why a holdout exists.** The synthesizer saw only `real_train.csv`. The
5,000 rows in `real_holdout.csv` are real people it never had access to. Any
claim of the form "a synthetic row is suspiciously close to a real row" needs
a baseline for how close two unrelated real records get **by chance** — and
the holdout is that baseline. Without it, a similarity number means nothing.

**Reproduce from scratch:**

```bash
python load_data.py      # fetch, sample, split; skips if cached
python synthesize.py     # regenerates data/synthetic.csv
```

Both deterministic. `load_data.py` points OpenSSL at `certifi`'s CA bundle
before fetching — macOS python.org builds ship without root certificates and
the OpenML request fails verification otherwise. Verification is not disabled.

### Schema — 15 columns, identical across all three files

| Column | Kind |
|---|---|
| `age` | numeric |
| `workclass` | categorical |
| `fnlwgt` | numeric (census sampling weight) |
| `education` | categorical |
| `education-num` | numeric |
| `marital-status` | categorical |
| `occupation` | categorical |
| `relationship` | categorical |
| `race` | categorical |
| `sex` | categorical |
| `capital-gain` | numeric |
| `capital-loss` | numeric |
| `hours-per-week` | numeric |
| `native-country` | categorical |
| `class` | binary label (`<=50K` / `>50K`) |

Six numeric, eight categorical, one binary target. Column order is identical in
all three files.

### Deliberately not yet established

**Nothing has been inspected.** Not a single row compared, no distributions, no
KS statistics — not even confirmation that the fidelity report's claim
reproduces.

The one risk that follows: **the green fidelity report is unverified.** If the
KS statistics do not reproduce, the data science lead's premise is wrong before
either question is reached.

## Assets

**Generated before the clock, treated as opaque:**

| File | Contents |
|---|---|
| `data/synthetic.csv` | 15,000 rows — the file under review |
| `synthesize.py` | The generator. Not to be read. |

**Provided as evidence:**

| File | Contents |
|---|---|
| `data/real_train.csv` | 15,000 real rows — what the synthesizer saw |
| `data/real_holdout.csv` | 5,000 real rows — what it never saw; the control |

**Written for this problem:**

| File | Purpose |
|---|---|
| `load_data.py` | Fetch, sample, split, cache |
| `PLAN.md` | Written at the start of the clock — does not exist yet |
| `evaluate.py` | My work — does not exist yet |
| `EVALUATION.md` | Findings, the yes/no, and the training call |
| `PROBLEM.md` | This file |
