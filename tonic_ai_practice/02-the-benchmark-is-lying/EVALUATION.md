# Model A vs Model B — Evaluation

**Verdict: the two-point gap does not exist, and the benchmark cannot support
the comparison being asked of it.** Do not ship on this evidence. Neither model
is shown to be better than the other.

---

## Verdict

### 1. The gap does not reproduce

The brief says Model B beats Model A by about two points of macro-F1. At this
sample and seed the gap is **0.78 points** — roughly a third of the stated
figure, and the number already written into a planning doc.

### 2. The gap is indistinguishable from noise

Two independent paired tests agree:

| Test | Result |
|---|---|
| McNemar, exact, paired | **p = 0.2383** — not significant at any conventional level |
| Paired bootstrap, 10,000 resamples | 95% CI **[−0.38, +1.96]** points — **contains zero** |

The confidence interval is 2.33 points wide — **three times the size of the gap
itself** — and 9.1% of resamples put A at or ahead of B. Even the optimistic end
of the interval, +1.96, does not reach the claimed two points.

### 3. The gap is one class, and it is the easy one

Macro-F1 averages four classes, so each contributes a quarter of its delta.

| Class | A | B | B − A | Share of total gap |
|---|---:|---:|---:|---:|
| World | 0.8782 | 0.8755 | −0.0027 | −9% |
| **Sports** | 0.9264 | 0.9477 | **+0.0213** | **68%** |
| Business | 0.8332 | 0.8350 | +0.0018 | 6% |
| Sci/Tech | 0.8468 | 0.8577 | +0.0109 | 35% |

Sports alone supplies more than two-thirds of the advantage, and it was already
the easiest class at 0.93. On World, A is ahead. B is not better at the hard
part of the problem — it is better at the part that was nearly solved.

### 4. The benchmark's label-error floor is ten times the gap

In **160 documents (8.0% of the test set)**, Model A and Model B *agree with
each other* and both disagree with gold. Two deliberately different inductive
biases — word-level linear and character n-gram SVM — converging on the same
answer against the label is far better evidence that the *label* is wrong than
that both models made the same mistake.

Read a few and the pattern is unmistakable:

| Gold | Both models said | Document |
|---|---|---|
| Sci/Tech | Business | *Cingular to Reduce Work Force by Roughly 10 Percent* |
| Sci/Tech | Business | *Worldwide PC Market Seen Doubling by 2010* |
| Sci/Tech | Business | *Dell cuts prices on many corporate products* |
| Business | Sci/Tech | *Judge asked to penalize Microsoft over e-mails* |
| Sports | World | *Atlanta police arrest Braves player on DUI charge* |

A layoffs story labelled Sci/Tech. A market forecast labelled Sci/Tech. A DUI
arrest labelled Sports. These are not model failures; several are defensibly
*correct* predictions scored as errors.

**The suspect-label region is 8.0% of the test set. The gap under discussion is
0.78%.** The argument is happening inside the noise floor by an order of
magnitude.

---

## The Principal SWE's second question

> *If I retrain next month and the gap flips, will I know why?*

**It will flip, and no — not from this benchmark.**

A 0.78-point gap with a 2.33-point interval is a coin weighted roughly 60/40.
Next month's number will be different, its sign may well reverse, and nothing
in this evaluation can distinguish a real change in the models from the same
noise resampled. Any retraining decision keyed to this metric is keyed to a
random variable.

Concretely: **93% of the test set is silent on the question.** 1,680 documents
both models get right and 179 both get wrong carry no information about which
is better. The entire choice rests on **141 discordant documents**, which split
63 to 78 — close to a coin flip.

---

## Recommendation

1. **Do not decide on this comparison.** Not "ship A instead" — the evidence
   does not establish either model as better.
2. **Because the two models are established as equivalent in accuracy to within
   measurement error, the choice can be made on operational grounds** — latency,
   memory, interpretability, retraining cost, failure mode under distribution
   shift. The equivalence is what licenses that, and it is a finding, not a
   shrug: a 0.78-point difference with a 2.33-point interval is not a quality
   difference anyone can act on. Those operational differences are real and
   measurable; this one is not.
3. **Adjudicate the 160 both-models-disagree documents.** Not only because
   future comparisons inherit the ceiling, but because **removing this noise is
   what makes the signal visible at all.** The differences the team wants to
   measure are currently buried under an error floor ten times their size, and
   that floor can be cleared in an afternoon.

   A large share of those 160 are not errors to correct — they are a **class
   boundary to define.** 63 of them (39%) sit in the Business↔Sci/Tech pair,
   and others straddle Sports and World: *Atlanta police arrest Braves player
   on DUI charge* is not
   mislabelled so much as unlabelled by any stated rule. Relabelling does not
   fix that; a written tie-break guideline does. Two sentences deciding whether
   an athlete's arrest routes as Sports or World is cheaper than annotation and
   prevents the ambiguity regenerating in every future batch.
4. **Re-state the decision threshold, and price the alternative.** A gap must
   be roughly 2.5 points before this test set can detect it. CI width scales as
   1/√n, confirmed empirically here:

   | Test set size | 95% CI width | Detects a 0.78-point gap? |
   |---:|---:|---|
   | 2,000 (current) | 2.27 pts | no |
   | 4,000 | 1.65 pts | no |
   | **7,600** (full AG News test split) | **1.19 pts** | **marginally** |
   | 20,000 | 0.72 pts | comfortably |

   So the team has a real choice rather than a constraint. **Using the full
   7,600-document test split costs nothing and reaches the edge of
   detectability.** Resolving sub-point differences with margin needs roughly
   20,000 documents — an order of magnitude more than today, and more than AG
   News provides, so it means sourcing data rather than resampling. Either
   accept that differences under ~2.5 points are invisible here, or invest at
   that scale.

**What happens Monday:** the routing system can ship with either model now.
That decision is an engineering one and it is not blocked by this analysis. The
benchmark work is a parallel track that gates the *next* comparison, not this
deployment.

---

## Method

| Step | What | Status |
|---|---|---|
| 0 | Join predictions to gold, assert totality | Done |
| 1 | Reproduce macro-F1, per-class F1, confusion matrices | Done |
| 2 | McNemar — are the models distinguishable at all | Done |
| 3 | Paired bootstrap CI on the macro-F1 difference | Done |
| 4 | Label-quality probe — is the advantage real or annotation luck | Done |
| 5 | Confidence calibration / score-based analysis | Not run |
| 6 | Per-class threshold or prior adjustment | Not run |

### Belief criteria, fixed before results

These were written down before any number was computed, so they could not be
shaped by the answer.

| | Criterion | Outcome |
|---|---|---|
| 1 | A paired significance test, not two separate error bars | **Fails** — McNemar p = 0.24 |
| 2 | Bootstrap CI on the difference excludes zero | **Fails** — [−0.38, +1.96] |
| 3 | Gap not concentrated in a single class | **Fails** — 68% from Sports |
| 4 | B's wins not concentrated on items with bad gold | **Inconclusive** — see below |

**On criterion 4, the test did not find what it was looking for, and found
something more important.** B's wins are 40% Business↔Sci/Tech; A's wins are
41%. B is *not* disproportionately profiting from ambiguous labels — the two
models win in structurally identical ways, B simply wins 15 more times. The
finding is not that B games the ambiguity, but that the ambiguity is large
enough to swamp the quantity being measured.

---

## Numbers

### Macro-F1

```
A = 0.8711    B = 0.8790    gap = +0.0078  (+0.78 points)
```

### Paired outcome table

| | Count | Share |
|---|---:|---:|
| Both correct | 1,680 | 84.0% |
| Only A correct | 63 | 3.2% |
| Only B correct | 78 | 3.9% |
| Both wrong | 179 | 9.0% |
| **Discordant (carries all the signal)** | **141** | **7.1%** |

Accuracy: A = 0.8715, B = 0.8790.

### Confusion matrices

**Model A** (rows true, cols predicted)

| | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| World | 429 | 25 | 28 | 16 |
| Sports | 9 | 478 | 7 | 5 |
| Business | 25 | 15 | 427 | **54** |
| Sci/Tech | 16 | 15 | **42** | 409 |

**Model B**

| | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| World | 429 | 19 | 32 | 18 |
| Sports | 10 | 480 | 7 | 2 |
| Business | 26 | 9 | 430 | **56** |
| Sci/Tech | 17 | 6 | **40** | 419 |

**Both models fail identically where it is hard.** The Business↔Sci/Tech block
totals **96 errors in A and 96 in B** — the largest error source in each, and
completely unchanged by the model swap. Whatever B buys, it is not help with
the confusion that dominates.

### Both-wrong pile

| | Count |
|---|---:|
| Both wrong | 179 |
| — A and B agree with each other | **160** |
| — A and B differ | 19 |
| Of the agreeing ones, Business↔Sci/Tech | 63 (39%) |

Gold → what both models said instead:

| | |
|---|---:|
| Business → Sci/Tech | 33 |
| Sci/Tech → Business | 30 |
| World → Business | 23 |
| Business → World | 18 |
| World → Sports | 14 |

---

## Judgment calls on the record

| # | Decision | Rationale |
|---|---|---|
| 1 | Paired tests throughout, never two independent CIs | Both models scored the same 2,000 documents; ignoring the pairing discards their correlation and inflates uncertainty |
| 2 | McNemar reported as accuracy, not macro-F1 | It tests per-item correctness. The headline metric is macro-F1, so the bootstrap carries that question |
| 3 | Bootstrap resamples documents, both models on the same resample | Documents are the independent unit; independent resampling would widen the interval artificially |
| 4 | Macro-F1 recomputed by hand inside the bootstrap | Speed. Asserted equal to sklearn's on the full sample before use |
| 5 | Label quality inferred from model agreement, not re-annotation | No clean gold available inside the hour. Two different inductive biases agreeing against the label is the strongest proxy obtainable |
| 6 | Business↔Sci/Tech treated as the candidate ambiguous pair | Chosen after seeing it dominate both confusion matrices, not before — this one is post hoc and is flagged as such |

---

## Not run

- **Confidence calibration.** The two `score` fields are in different units —
  A's is a probability in [0,1], B's an unbounded one-vs-rest decision value.
  Comparing them needs a transform (percentile rank within model). Would answer
  whether either model *knows* when it is on an ambiguous document.
- **Adjudication of the 160 suspect labels.** The estimate that a meaningful
  share are annotation errors rests on reading ten of them. The claim is
  directionally strong and numerically unquantified.
- **Training-set label noise.** Only the test labels were examined. If the same
  ambiguity exists in training, both models learned partly from it.
- **Variance across seeds.** Both models were trained once. Retraining each at
  several seeds would separate model variance from test-set variance, which is
  what the retrain question ultimately needs.

## Caveats

1. **The 160-document figure counts model agreement, not verified label
   errors.** It is an upper bound on the suspect region. The ten examples read
   support it, but ten is not a measurement.
2. **The ambiguous-pair definition is post hoc**, chosen after seeing the
   confusion matrices.
3. **Both models trained on 4,000 examples**, a small fraction of AG News.
   The stated two-point gap may well be real in a different training regime;
   this evaluation can only speak to the artifacts it was given.
