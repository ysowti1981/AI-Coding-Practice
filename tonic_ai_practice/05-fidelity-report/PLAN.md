# Plan — Problem 05

Written at the start of the clock, before any measurement.

**Two deliverables:** a yes/no for the privacy officer, and a training go/no-go
for the data science lead. Everything measured from `synthetic.csv`,
`real_train.csv` and `real_holdout.csv` — per the contamination rule in
`PROBLEM.md`.

## Out of scope

- **Not fixing or rebuilding the synthesizer.** Diagnosis and two answers.
- **No formal differential-privacy analysis.** Wrong instrument, wrong hour.
- **Not inferring the method from the source.** The file is opaque.
- **No re-derivation of what "synthetic" should mean.** The questions are
  concrete: can someone be identified, and can they train on it.

## What I would need to see to believe "green fidelity ⇒ safe and useful"

Written before any number exists. Five things, split by which question they
serve.

**For safe:**

1. **No synthetic row is a verbatim copy** of a real training row — beyond
   whatever rate occurs by chance between two unrelated real samples.
2. **Synthetic rows are no closer to the training set than to the holdout.**
   If nearest-neighbour distances to `real_train` are systematically smaller
   than to `real_holdout` — equally real, never seen — the generator memorised.
3. **No real individual is uniquely singled out** by a quasi-identifier
   combination that appears in the synthetic file.

**For useful:**

4. **Joint structure survives.** Marginals matching says nothing about whether
   the relationships between columns are intact.
5. **A model trained on synthetic transfers to real data.** Train-on-synthetic,
   test-on-real-holdout, against a train-on-real baseline.

**KS speaks to none of these.** It is a per-column marginal test: it compares
one column at a time and is blind to joint structure and to memorisation by
construction. A file that is a verbatim copy of the real data passes every KS
test perfectly. That is not a criticism of the fidelity report — it is a
statement about what the instrument measures.

## Steps

| # | Step | Budget | Deliverable |
|---|---|---:|---|
| 0 | Load, verify schema and holdout disjointness | 5 min | Three frames aligned; control confirmed valid |
| 1 | Reproduce the fidelity claim | 7 min | KS per numeric, TV distance per categorical |
| 2 | Exact-match test | 6 min | Count of verbatim copies, with a chance baseline |
| 3 | Nearest-neighbour distance, train vs holdout | 10 min | Whether the generator memorised |
| 4 | Quasi-identifier uniqueness | 8 min | Count of real individuals singled out |
| 5 | Joint structure | 8 min | Whether column dependencies survived |
| 6 | Train-on-synthetic, test-on-real | 10 min | The number that decides training |
| 7 | The two answers | 6 min | Yes/no, and go/no-go |

### Step 0 — Load and verify the control
Schema and column order identical across all three files. Then the check that
makes everything else valid: **is `real_holdout` actually disjoint from
`real_train`?** If they overlap, the baseline in Steps 2 and 3 is contaminated
and every privacy conclusion built on it is wrong.

### Step 1 — Reproduce the fidelity claim
KS per numeric column, total-variation distance per categorical. Does "all
under 0.02" hold? If it does not, the DS lead's premise is wrong before either
question is reached.

### Step 2 — Exact-match test
How many synthetic rows appear **verbatim** in `real_train`? Control: how many
appear in `real_holdout`? The control matters — two unrelated real samples
share some low-entropy rows by chance, and without that baseline a raw count
proves nothing.

**If this is non-zero and the baseline is not, the privacy officer has their
answer and it is a one-word answer.**

*Judgment call:* whether to match on all 15 columns or exclude `fnlwgt`. It is
a census sampling weight with very high cardinality — including it makes a
match nearly conclusive, excluding it makes matching harder but the result more
conservative. Report both.

### Step 3 — Nearest-neighbour distance
For each synthetic row, distance to its nearest real-train row and to its
nearest real-holdout row. If the train distances are systematically smaller,
the generator copied structure it should not have.

*Judgment call:* distance on mixed types. Gower, or one-hot plus Euclidean on
standardised numerics. Both defensible; pick one, state it, and note the other.

### Step 4 — Quasi-identifier uniqueness
The classic re-identification path. Choose a quasi-identifier set, count how
many real individuals are unique on it, and how many of those combinations
appear in the synthetic file.

*Judgment call:* which columns are quasi-identifiers. `age`, `sex`, `race`,
`marital-status`, `education`, `native-country`, `hours-per-week` is the
conventional set for Adult. The choice changes the count and must be stated.

### Step 5 — Joint structure
Pairwise association in real versus synthetic: correlation for numeric pairs,
Cramér's V for categorical pairs. Marginals can match perfectly while every
relationship between columns is destroyed.

### Step 6 — Train on synthetic, test on real
Train a classifier on `synthetic.csv`, evaluate on `real_holdout.csv`. Baseline:
the same model trained on `real_train.csv`, evaluated on the same holdout. The
gap is the DS lead's answer in their own terms.

*Judgment call:* model choice. Something fast and standard — gradient boosting
or logistic regression — reported with the real-trained baseline so the number
is interpretable rather than absolute.

### Step 7 — The two answers
**For the privacy officer:** yes or no, in that form, plain language, with one
concrete checkable example if the answer is yes. No distributions.

**For the data science lead:** go or no-go on training, with the number.

## Cut order

If the clock runs short: **2 → 6 → 7.** Exact-match answers the privacy
question if it fires; train-on-synthetic answers the training question; 7
writes both.

Drop first: Step 5 (joint structure — Step 6 measures its consequence more
directly), then Step 4, then Step 3. **Step 0 is never cut** — an invalid
control would silently invalidate Steps 2 and 3.

Note that Step 3 only becomes load-bearing if Step 2 comes back clean. If
verbatim copies exist, the privacy answer is settled and Step 3 becomes
supporting detail rather than the main evidence.

## Open questions

1. **What counts as "identified" for the privacy officer?** An exact record
   copy is unambiguous in any reading. A near-match on quasi-identifiers is a
   judgement call. **Proceeding by answering the strictest standard first** —
   exact copies — and reporting weaker forms separately, so the yes/no does not
   depend on where the line is drawn.
2. **Is there a utility bar for training?** "Can we train on this" has no
   stated threshold. Reporting the train-on-synthetic gap against the
   train-on-real baseline lets them set the bar themselves rather than my
   inventing one.
