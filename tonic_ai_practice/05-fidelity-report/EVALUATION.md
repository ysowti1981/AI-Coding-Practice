# Synthetic Data Review

**Status: complete.** Steps 0, 1, 2, 4, 6 and 7 of `PLAN.md` are done. Steps 3
(nearest-neighbour distance) and 5 (joint structure) were skipped; neither
could change either answer. The two replies are in
[Step 7](#step-7--the-two-answers).

**The privacy officer's question is answered: yes.** 600 real people's complete
records appear, unchanged, in the synthetic file — and **159 of them can be
found using six everyday facts** (age, sex, race, marital status, education,
country of birth), after which the file reveals their occupation, income
class, capital gains and working hours.

**The data science lead's question is answered: no-go.** A model trained on
the synthetic rows alone is no better than a coin flip (AUC 0.492, 95% CI
0.473–0.509). Every bit of
predictive value the file has comes from the 600 leaked real records — so the
only useful part of the file is the part that cannot be released.

Behind both answers: **the green fidelity report is true and incomplete.** The
KS claim holds exactly as stated, but it checks each column on its own — and
covers only 6 of the 15. It cannot see copied records, and it cannot see
whether columns still relate to one another.

Every claim is measured from `synthetic.csv`, `real_train.csv` and
`real_holdout.csv`. Nothing is justified by reading `synthesize.py`.

---

## Step 0 — Schema and the control

| Check | Result |
|---|---|
| Column count and order | Identical across all three files (15 columns) |
| Shapes | train 15,000 · holdout 5,000 · synthetic 15,000 |
| Numeric columns | 6 — `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical columns | 9 — `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`, `class` |

### The control, validated

The split was by index, but Adult contains duplicate records, so identical rows
can appear in both halves by chance. That rate is the baseline every later
privacy claim must be measured against.

> **4 of 5,000 holdout rows (0.080%) appear verbatim in the training set.**

Two unrelated real samples of the same population coincide at roughly
**8 in 10,000**. The control is clean: an exact-match rate materially above
0.08% in the synthetic file cannot be attributed to chance.

## Step 1 — Does the green fidelity report reproduce?

Each column compared against the real training data, with a **real-vs-real
control**: the same statistic computed between `real_holdout` and
`real_train`. Both are genuine samples of the same population, so that column
shows what ordinary sampling variation looks like.

### Numeric — two-sample KS

| Column | Synth vs train | Holdout vs train | |
|---|---:|---:|---|
| age | 0.0069 | 0.0073 | tighter than real-vs-real |
| fnlwgt | 0.0095 | 0.0086 | |
| education-num | 0.0099 | 0.0037 | |
| capital-gain | 0.0018 | 0.0039 | tighter than real-vs-real |
| capital-loss | 0.0012 | 0.0024 | tighter than real-vs-real |
| hours-per-week | 0.0068 | 0.0088 | tighter than real-vs-real |

**All six are under 0.01 — the claim of "all under 0.02" is correct.**

### Categorical — total variation distance

KS does not apply to categorical columns. Total variation distance is the
comparable statistic.

| Column | Synth vs train | Holdout vs train | |
|---|---:|---:|---|
| workclass | 0.0065 | 0.0113 | tighter than real-vs-real |
| education | 0.0163 | 0.0138 | |
| marital-status | 0.0052 | 0.0164 | tighter than real-vs-real |
| occupation | 0.0151 | 0.0253 | tighter than real-vs-real |
| relationship | 0.0058 | 0.0193 | tighter than real-vs-real |
| race | 0.0077 | 0.0045 | |
| sex | 0.0031 | 0.0085 | tighter than real-vs-real |
| **native-country** | **0.0653** | 0.0186 | **over the claimed ceiling** |
| class | 0.0004 | 0.0053 | tighter than real-vs-real |

### How total variation is calculated

**Why not KS.** KS is the largest gap between two cumulative distribution
functions, and a CDF needs the values to be ordered. `native-country` has no
order — `Canada` is not "less than" `Germany` — so any KS figure would depend
on the arbitrary order the categories happened to be listed in.

**The formula.** For each value, take its share of rows in each file, then:

```
TV = ½ × Σ | share_real(v) − share_synth(v) |
```

summed over every value that appears in *either* file. A value absent from one
file counts with a share of 0 there. Missing values (`NaN`) are treated as a
category of their own.

**Why halve.** Shares sum to 1 in both files, so every bit of share that leaves
one value must arrive at another. The raw sum counts each shift twice — once
where it left, once where it landed. Halving gives the reading that makes TV
useful to a non-statistician:

> **TV is the fraction of rows that would have to be relabelled for the two
> distributions to match.**

**Worked example — `native-country`:**

| Value | Real share | Synthetic share | \|difference\| | Direction |
|---|---:|---:|---:|---|
| United-States | 0.8947 | 0.9595 | 0.0647 | gained |
| Philippines | 0.0069 | 0.0003 | 0.0065 | lost |
| Germany | 0.0044 | 0.0002 | 0.0042 | lost |
| Canada | 0.0041 | 0.0003 | 0.0039 | lost |
| El-Salvador | 0.0039 | 0.0003 | 0.0036 | lost |
| *36 further values* | | | 0.0477 | mostly lost |
| **Sum** | | | **0.1307** | |

```
TV = ½ × 0.1307 = 0.0653
```

The double counting is visible in the table: `United-States` **gained 0.0647 on
its own**, and the other 40 values **lost roughly that much between them**. The
sum of 0.1307 counts that single transfer from both ends; halving recovers the
amount actually displaced.

**Reading it:** 6.5% of synthetic rows carry the wrong country relative to the
real data, and nearly all of that error runs in one direction — people who
should be recorded under forty other countries are recorded as
`United-States`.

For contrast, `sex` has TV 0.0031: 0.31% of rows moved from `Female` to `Male`,
appearing once in each row of a two-row table. That is sampling noise; the
`native-country` figure is twenty times larger.

**Why using TV here is fair to the report.** TV is the largest difference in
probability over *any* grouping of values; KS is the same quantity restricted
to "values up to x" under some ordering. So for any ordering one might impose,
**KS ≤ TV**. Holding a categorical column to a KS-derived 0.02 threshold via TV
is conservative — it can only flag more, never fewer. `native-country` fails
because the distribution changed, not because a harsher metric was chosen.

---

### Finding 1 — The report is true and incomplete

The data science lead's statement is accurate: every KS statistic is under
0.02. **KS applies only to the six numeric columns**, so the green report
covers 6 of 15. The nine categorical columns were not — and could not have
been — tested by that instrument.

One of those nine fails badly.

### Finding 2 — `native-country` has lost half its categories

TV distance 0.0653, more than three times the claimed ceiling, and the cause is
not distributional drift. Counting distinct values present in each file:

| Column | Real | Synthetic | Lost |
|---|---:|---:|---:|
| workclass | 9 | 8 | 1 |
| education | 16 | 16 | 0 |
| marital-status | 7 | 6 | 1 |
| occupation | 15 | 14 | 1 |
| relationship | 6 | 6 | 0 |
| race | 5 | 5 | 0 |
| sex | 2 | 2 | 0 |
| **native-country** | **41** | **20** | **21** |
| class | 2 | 2 | 0 |

**21 of 41 nationalities are entirely absent from the synthetic file.** Where
their mass went:

| Value | Real share | Synth share | Delta |
|---|---:|---:|---:|
| United-States | 0.8947 | 0.9595 | **+0.0647** |
| Mexico | 0.0206 | 0.0212 | +0.0006 |
| *(missing)* | 0.0172 | 0.0168 | −0.0004 |
| Philippines | 0.0069 | 0.0003 | −0.0065 |

> **The 21 absent nationalities account for 2.16% of the real population** —
> roughly **324 of the 15,000 training records**. Those people are not
> under-represented in the synthetic file. They are gone, and `United-States`
> absorbed their share.

Columns with many rare values lose the most: `workclass` loses 1 of 9,
`marital-status` 1 of 7, `occupation` 1 of 15.

This has consequences for both questions, in opposite directions — it removes
the most identifying attribute values, while making the data unusable for any
analysis involving the populations erased. Both are deferred to the steps that
measure them.

### Finding 3 — Ten of fifteen columns match *too well*

In 10 of 15 columns the synthetic data is **closer to the training set than a
genuine second sample of the same population is.**

A real sample drawn from the same population carries sampling noise — that is
what the holdout column measures. Synthetic data that is *tighter* than that
is not reproducing the population; it is reproducing **the training set
specifically.**

On its own this is a fingerprint, not proof. But it is the signature of a
generator that reuses training values directly rather than modelling the
distribution they came from, and it is the first thing seen so far that bears
on the privacy officer's question.

---

## Step 2 — Verbatim copies of real rows

### What this measures

**Exact-match rate** is the share of synthetic rows identical, in every column,
to some real training row. It is the strictest reading of "can an individual be
identified": if a real person's full record appears unchanged in the synthetic
file, that person is in it.

**How it is computed.** Each row is turned into a single key by joining all its
column values, and a synthetic row counts as a match if its key appears among
the training keys.

**Why it needs a baseline.** Two unrelated real samples share some identical
rows by chance. Step 0 measured that rate: holdout rows appear in train at
0.080%. The synthetic rate is compared against that control, using the same
15,000-row training set as the reference, so the two are directly comparable.

**A second check: one-person profiles.** A match is only identifying if the
matched training row belongs to a single person — if its full profile occurs
exactly once in the training set. Chance matches should mostly land on common
profiles, so the same one-person measure is computed for the control.

### Result — matching on all 15 columns

| | Rate | Rows |
|---|---:|---:|
| Control: real holdout rows found in train | 0.080% | 4 of 5,000 |
| **Synthetic rows found in train** | **4.000%** | **600 of 15,000** |
| Synthetic rows found in the holdout (never seen) | 0.000% | 0 |
| Matched rows whose profile is unique in train | — | **600** |

> **600 synthetic rows are exact copies of 600 distinct real people — 50× the
> chance rate.** Every one of them matches a training profile that belongs to
> a single person.

Two details make this more than a high number:

- **The copies are specific to the training data.** Against the holdout — real
  people the generator never saw — the synthetic file matches **zero** rows.
  It reproduces the individuals it was given, not the population they came
  from. This is the same signature Finding 3 hinted at in Step 1.
- **The count is exactly 600, a round 4.000%.** Chance produces scatter, not
  round fractions.

### Example — a real person, in the synthetic file

| Column | Synthetic row 7 | Real training row 14714 |
|---|---|---|
| age | 37 | 37 |
| workclass | Local-gov | Local-gov |
| fnlwgt | 216473 | 216473 |
| education | HS-grad | HS-grad |
| education-num | 9 | 9 |
| marital-status | Married-civ-spouse | Married-civ-spouse |
| occupation | Protective-serv | Protective-serv |
| relationship | Husband | Husband |
| race | White | White |
| sex | Male | Male |
| capital-gain | 0 | 0 |
| capital-loss | 0 | 0 |
| hours-per-week | 48 | 48 |
| native-country | United-States | United-States |
| class | >50K | >50K |

Fifteen of fifteen fields identical, including a six-digit census weight and
the income label. This is the example for the privacy officer: it needs no
statistics to check.

### The caveat — the evidence depends on `fnlwgt`

Repeating the test **without `fnlwgt`**:

| Matching on 14 columns | Control (holdout) | Synthetic |
|---|---:|---:|
| Rows found in train | 11.040% | 4.200% |
| Rows matching a one-person train profile | 7.020% | 3.720% |

Without `fnlwgt`, **exact matching cannot separate copying from coincidence on
this dataset.** Adult's other 14 columns are low in information — "white
married male, HS-grad, 40 hours a week" recurs constantly — so 11% of unrelated
real people match a training row by chance, more than the synthetic file does.

This does **not** weaken the answer, for two reasons:

1. `fnlwgt` **is in the released file**, and the copies include it. The
   privacy question is about the file as it would be shipped.
2. It shows that **`fnlwgt` behaves like an identifier.** A column that turns an
   ambiguous profile into a unique one should be questioned in any release,
   synthetic or not.

It does mean the finding should be stated as it is: the 600 copies are
demonstrable because `fnlwgt` makes records distinctive. Step 3's
distance-based test is the check that does not depend on any single column.

---

## Step 3 — Skipped

Nearest-neighbour distance (synthetic-to-train against synthetic-to-holdout)
was not run. `PLAN.md` made it conditional: its job was to detect memorisation
*without* exact copies, and Step 2 had already found 600 exact copies. It would
still have been the one test independent of any single column; Step 4 covers
that gap from a different direction.

---

## Step 4 — Quasi-identifier linkage

### The concepts

**Quasi-identifiers (QIs)** are attributes that identify no one alone but can
in combination. The classic result: ZIP code, date of birth and sex together
single out roughly 87% of US adults. For Adult, the conventional set is:

> `age`, `sex`, `race`, `marital-status`, `education`, `native-country`

— facts a neighbour, colleague or data broker plausibly knows.

**k-anonymity** is how many people share a given QI combination. At **k = 1**
the combination belongs to exactly one person: knowing those six facts pins
them down.

**The linkage attack.** An attacker knows a target's six QIs and looks them up
in the synthetic file. If **exactly one** synthetic row carries that
combination, they read off its **sensitive attributes** — `occupation`,
`class` (income), `capital-gain`, `hours-per-week`. If all four equal the
target's real values, the attacker has learned private facts about a real
person from a file that was supposed to contain none. That is **attribute
disclosure**.

**The control.** The same attack against **holdout** people — real, but never
seen by the generator. A safe synthetic file reveals no more about the people
it was trained on than about outsiders; any excess is leakage.

### How exposed the real data is to begin with

| k-anonymity of the real training data on the six QIs | Share of people |
|---|---:|
| **k = 1** — unique, one person | **22.57%** |
| k ≤ 5 | 45.85% |

**Nearly a quarter of real people are unique on six everyday attributes.** That
is a property of the real data, not the generator — but it means any release
that carries those six columns forward, synthetic or otherwise, has a large
surface to protect.

### Result

| Targets | Uniquely linked | Fully disclosed |
|---|---:|---:|
| Training people (seen by the generator) | 10.09% | **1.22% — 183 people** |
| Holdout people (never seen) | 9.70% | 0.16% — 8 people |

**Disclosure is 7.6× higher for people the generator saw.**

Linking itself succeeds about equally for both groups — 10.1% against 9.7% —
because a synthetic row whose QI combination happens to be unique links to
*some* real person either way. What differs is whether the sensitive
attributes it reveals are *correct*.

### Where the 183 come from

| | People |
|---|---:|
| Disclosed training people | 183 |
| — also verbatim copies from Step 2 | **159** |
| — not copied | 24 |
| Expected by chance, at the holdout rate | **24** |

The 24 non-copied disclosures equal the chance expectation exactly. **All of
the excess — 159 people — comes from the verbatim copies.**

So this is not a second, independent leak. It is the Step 2 copies seen through
the path a real attacker would actually use. Of the 600 copied people, 159 —
about one in four — are recoverable this way; the rest share their QI
combination with other synthetic rows, so the lookup does not return a single
answer.

### This closes the `fnlwgt` caveat from Step 2

Step 2 found that *detecting* the copies by exact match depended on `fnlwgt`:
without it, the other 14 columns are too common to separate copying from
coincidence.

**Exploiting them does not depend on `fnlwgt`.** The attack above uses none of
it — only the six QIs. An attacker who never sees a census weight still
recovers 159 real people's sensitive attributes.

### Example

> **The attacker knows:** age 61, male, white, widowed, 10th-grade education,
> born in Mexico.
>
> **The synthetic file reveals:** works in craft and repair, earns ≤ $50K, no
> capital gains, works 18 hours a week — **all four correct.**

This is the example for the privacy officer: six facts in, four private facts
out, no statistics involved.

---

## Step 5 — Skipped

Joint structure (pairwise correlation and Cramér's V, real against synthetic)
was not run. `PLAN.md` ranked it first to cut, because Step 6 measures its
consequence directly: if the relationships between columns are gone, a model
trained on the file cannot learn them. Step 6 confirms that they are gone
without measuring each pair.

---

## Step 6 — Train on synthetic, test on real

### What this step answers

The data science lead's question: *can my team train on this?* The only test
that answers it in their terms is to do exactly that — train a model on the
synthetic file and see whether it works on real people. Column-by-column
fidelity (Step 1) cannot answer it, because a model learns from how columns
relate to each other, not from each column alone.

### The concepts

**TSTR — train on synthetic, test on real.** Fit a classifier on synthetic
data, score it on real data the generator never saw (`real_holdout.csv`). Here
the task is the dataset's own: predict income (`class`) from the other 14
columns.

**TRTR — train on real, test on real.** The same model, trained on the real
training set and scored on the same holdout. It is the **ceiling**: the best
the synthetic file could hope to match. TSTR only means something next to it.

**ROC AUC** — the headline metric. It measures how well the model *ranks*
people: the probability that a randomly chosen high earner gets a higher score
than a randomly chosen low earner.

- **0.5** — no better than a coin flip
- **1.0** — perfect ranking

It is used instead of accuracy because income is imbalanced — 76.5% of the
holdout earns ≤ $50K — so a model that predicts "≤ $50K" for everyone scores
76.5% accuracy while knowing nothing. AUC does not reward that; accuracy is
reported alongside, next to that 0.765 majority-class baseline.

**Two diagnostic runs** separate where any value comes from, using the 600
verbatim copies identified in Step 2:

- **synthetic minus the copies** — the genuinely generated rows on their own
- **only the copied real rows** — what the 600 leaked records are worth alone

Model: gradient-boosted trees (`HistGradientBoostingClassifier`), categorical
columns encoded from each training set; categories unseen in training — such as
the 21 nationalities missing from the synthetic file — are treated as missing.

### Result

Scored on 5,000 real holdout rows. Majority-class accuracy 0.765; random AUC
0.500.

| Trained on | Rows | AUC | 95% CI | SE | Accuracy |
|---|---:|---:|---|---:|---:|
| Real training data — the ceiling | 15,000 | **0.924** | [0.915, 0.932] | 0.004 | 0.873 |
| Synthetic, as delivered | 15,000 | 0.776 | [0.761, 0.791] | 0.008 | 0.773 |
| **Synthetic minus the verbatim copies** | 14,400 | **0.492** | **[0.473, 0.509]** | 0.009 | 0.765 |
| **Only the 600 copied real rows** | 600 | **0.887** | [0.876, 0.897] | 0.005 | 0.837 |

**How the intervals are computed.** A bootstrap over the holdout: resample the
5,000 real test people with replacement, rescore the *same* fitted model on
each resample, repeat 2,000 times, and take the 2.5th and 97.5th percentiles
of the resulting AUCs. The interval shows how much the AUC would move with a
different draw of test people. The bootstrap standard error for the 0.492 run
(0.009) agrees with the analytic Hanley–McNeil value for a coin-flip AUC at
these class sizes (≈ 0.0096).

### Reading it

**1. The generated rows carry no signal at all.** With the 600 copies removed,
the remaining 14,400 synthetic rows give AUC **0.492, 95% CI [0.473, 0.509]**.
The interval contains 0.5, so the result is **statistically indistinguishable
from random**. That it sits slightly *below* 0.5 means nothing — it is chance,
not a model learning the wrong direction. Accuracy equals exactly the
"≤ $50K for everyone" baseline. In
those rows, the other columns say nothing about income. The relationships the
model needs to learn are not there.

**2. All of the file's value comes from the leak.** The file as delivered
reaches AUC 0.776, and the only reason it is above 0.5 is the 4% of rows that
are real people. Remove them and the value vanishes.

**3. The 14,400 generated rows make things worse.** Trained on **just the 600
copied real rows**, the model reaches AUC **0.887** [0.876, 0.897]. Trained on
those same 600 *plus* 14,400 synthetic rows, it drops to 0.776 [0.761, 0.791].
The intervals do not overlap. The generated rows are not neutral padding —
they are noise that dilutes the real signal.

**4. The delivered accuracy is close to useless.** 0.773 against a
majority-class baseline of 0.765 — a model trained on this file predicts
"≤ $50K" for nearly everyone.

### What this means for the two questions together

Utility and privacy are **not** a trade-off here — they are the same rows.

- Remove the leaked records to make the file safe, and it becomes worthless
  (AUC 0.492).
- Keep them to make it useful, and it discloses 600 real people.

There is no version of this file that is both usable and releasable. That is
why the fix cannot be a patch to the file — it has to be a different
generator.

### Why the fidelity report missed this

Step 1 showed every marginal matches: KS under 0.01 on all six numeric
columns. That is fully consistent with this result. A file can reproduce each
column's distribution perfectly while the columns are unrelated to each other —
and a model learns only from how they relate. **Per-column fidelity is
necessary for training utility and nowhere near sufficient.**

---

## Step 7 — The two answers

Two readers, two questions, two formats. The privacy officer asked for yes or
no and declined statistics, so that reply contains none. The data science lead
asked whether anything blocks training, so that reply gives the numbers and
what would unblock it.

Both replies share one operational point that neither reader raised: **the
file contains real personal data**, so any copies already shared must be
handled like the original.

### To the privacy officer

> **Yes. Individuals in the real dataset can be identified from this synthetic
> file.**
>
> 600 of the 15,000 people in the real data appear in the synthetic file
> exactly as they appear in the original — every field identical. For
> example, one row in the synthetic file describes a 37-year-old married man
> in local-government protective services, a high-school graduate working 48
> hours a week, earning over $50K, with census weight 216473. The real file
> contains that same person, identical in all 15 fields. Your team can check
> this directly by searching the real file for those values.
>
> It is not only that copies are present. Someone who knows six ordinary facts
> about a person — age, sex, race, marital status, education and country of
> birth — can look them up in the synthetic file and, for 159 real people,
> correctly read off their occupation, income bracket, capital gains and
> weekly working hours. For instance: given only "61, male, white, widowed,
> 10th-grade education, born in Mexico", the file correctly reveals that he
> works in craft and repair, earns under $50K, has no capital gains and works
> 18 hours a week.
>
> What I recommend:
>
> - **Do not release or share this file.**
> - **Treat any copies already distributed as real personal data** — the same
>   access controls as the original — until they are deleted.
> - **Do not treat a green fidelity report as a privacy sign-off.** The
>   vendor's report checks whether each column looks statistically similar to
>   the original. It does not check whether real records were copied, and it
>   passed this file.

### To the data science lead

> **Yes, there is a blocker — please don't start training on this file.**
>
> Your fidelity report is correct: every KS statistic is under 0.02. But KS
> compares one column at a time, and only applies to the 6 numeric columns out
> of 15. It cannot see the three problems below.
>
> **1. The synthetic data has no predictive signal of its own.** We trained an
> income classifier on the file and scored it on 5,000 real people it never
> saw.
>
> | Trained on | AUC (95% CI) |
> |---|---|
> | Real data | 0.924 (0.915–0.932) |
> | This synthetic file | 0.776 (0.761–0.791) |
> | This file, minus the copied real rows (see 2) | **0.492 (0.473–0.509)** — a coin flip |
>
> Each column's distribution matches the real data, but the relationships
> *between* columns — which is what a model learns — are gone.
>
> **2. 4% of the file is real data.** 600 rows are verbatim copies of real
> training records, and they account for all of the file's predictive value:
> those 600 rows on their own reach AUC 0.887, better than the full file. This
> is also a privacy problem, which I've raised with the privacy officer; until
> it's resolved, the file needs to be handled as real personal data.
>
> **3. 21 of 41 nationalities are missing**, their records reassigned to
> "United States" — about 2% of people. A model trained on this would be blind
> to them.
>
> Points 1 and 2 together mean this can't be fixed by editing the file:
> removing the copied rows makes it safe and useless. It needs to be
> regenerated.
>
> **What would unblock it** — a regenerated file that passes, on arrival:
>
> - **Train-on-synthetic, test-on-real AUC within an agreed margin of the
>   real-data ceiling** (0.924 here). You set the margin; it's the number that
>   says whether the data is fit for your models.
> - **No copies of real records beyond chance**, checked against a held-out
>   real sample the generator never saw.
> - **Every category present** that appears in the real data.
> - **Per-column fidelity on all 15 columns**, not just the numeric six.
>
> I can turn those into an automated acceptance check so the next delivery is
> tested the day it arrives. The next-week start date depends on how quickly
> the vendor can regenerate.

---

## Against the belief criteria

Set in `PLAN.md` before any measurement:

| | Criterion | Status |
|---|---|---|
| 1 | No verbatim copies beyond the chance rate | **Fails** — 600 exact copies, 4.000% vs a 0.080% control (50×), none against the unseen holdout |
| 2 | Synthetic no closer to train than to holdout | **Not measured** — Step 3 skipped. Finding 3 and the zero matches against the holdout both point the same way |
| 3 | No real individual uniquely singled out | **Fails** — 159 real people recoverable from six QIs, 7.6× the rate for unseen people |
| 4 | Joint structure survives | **Not measured directly** — Step 5 skipped. Step 6 shows the consequence: without the copies, features carry no information about income |
| 5 | Model trained on synthetic transfers to real | **Fails** — AUC 0.776 against a 0.924 ceiling, and **0.492** once the leaked real rows are removed |

## Not run

| Step | What it would establish | Status |
|---|---|---|
| 3 | Nearest-neighbour distance — memorisation without exact copies | **Skipped.** Conditional on Step 2 finding nothing; it found 600 copies |
| 5 | Whether column dependencies survived | **Skipped.** Step 6 measures the consequence directly |

## Judgment calls on the record

| # | Decision | Rationale |
|---|---|---|
| 1 | A real-vs-real control column on every statistic | Without it, "KS = 0.0069" has no scale. The holdout says what ordinary sampling variation looks like for this data |
| 2 | Total variation distance for categorical columns | KS is undefined on unordered categories; TV is the standard comparable statistic and is on the same 0–1 scale |
| 3 | `NaN` treated as its own category in TV | Adult encodes missingness as a value with real meaning; dropping it would understate the distance |
| 4 | Row identity for the control = all 15 columns as strings | The conservative choice for a *baseline* — it makes coincidental matches rarer, so the 0.080% is a floor |
| 5 | Row keys built by vectorised elementwise `str`, not `astype(str)` | `astype(str)` leaves `NaN` as a float on some pandas versions and the join raises. The figures here are verified identical under two different pandas installs |
| 6 | Exact matching reported with and without `fnlwgt` | `fnlwgt` is high-cardinality and makes a match near-conclusive. Without it the test loses its power on this dataset — reported rather than hidden, since the finding depends on it |
| 7 | "Identifying" requires the matched profile to be unique in train | A match to a common profile points at many people; a match to a one-person profile points at exactly one |
| 8 | QI set = `age`, `sex`, `race`, `marital-status`, `education`, `native-country` | The conventional set for Adult: facts plausibly known about a person. A different set changes the counts; this one is stated so it can be argued with |
| 9 | Sensitive set = `occupation`, `class`, `capital-gain`, `hours-per-week`, all four required | Requiring all four makes a chance hit rare — the holdout rate is 0.16% — so a disclosure is meaningful rather than a coin flip on the binary income label |
| 10 | Attack succeeds only on a **unique** synthetic QI match | An attacker facing several candidate rows cannot tell which is their target; counting only single-row lookups is the conservative choice |
| 11 | ROC AUC as the headline utility metric, accuracy reported beside the majority baseline | Income is imbalanced (76.5% ≤ $50K); accuracy rewards predicting the majority class, AUC does not |
| 12 | Diagnostic runs with and without the verbatim copies | Separates the value of the generated rows from the value of the leaked real ones — the single most decision-relevant split in the analysis |
| 13 | Gradient-boosted trees, default settings, one seed | A strong, standard tabular model. Tuning would move the ceiling slightly; it cannot move a 0.492 to meaningful, which is the result that decides |
| 14 | Bootstrap CIs over the holdout, model held fixed | Captures test-set sampling variation — the uncertainty that matters for "is 0.492 different from 0.5". Does not capture training variation across seeds, which would widen intervals slightly without changing any conclusion |

## Reproducing this

```bash
python load_data.py      # fetch, sample, split; skips if cached
python synthesize.py     # regenerates data/synthetic.csv
python evaluate.py       # every figure above, in order
```

Every number in this document comes from that `evaluate.py` run — including the
category-coverage tables, which were folded in after an earlier version
computed them ad hoc. Verified to produce identical output under two separate
pandas installations.
