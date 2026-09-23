# Model A vs Model B — Recommendation

**Recommendation: ship B with a type-consistency validator downstream.**

Steps 0–2, 4, 4b and 6 of `PLAN.md` are complete. **Step 3** (five-bucket error
profile) and **Step 5** (score calibration) were not run; neither changes the
recommendation, and both are stated in full under *Confidence*.

Every claim is measured from `predictions_a.jsonl`, `predictions_b.jsonl` and
`train_en_2500.jsonl`. Nothing is justified by reading `build_models.py`.

---

## The answer in one screen

**Ship B, with a type-consistency validator downstream.**

| | A | B |
|---|---:|---:|
| High-severity entities **missed** | **677** | **289** |
| Entities located but wrongly typed | 0 | 878 |
| Spurious spans (fired on empty text) | 1,357 | 2,131 |

**A leaks 2.3× more of what matters.** B's apparent precision deficit is half
illusory — 878 of its errors are entities it found and mislabelled, which still
get redacted. Only 774 extra spans are genuine over-redaction.

**Leaks are permanent; type errors are repairable.** That asymmetry, not any F1
value, is what breaks the tie.

Why the team deadlocked: B wins on F1 only if the scoring ignores entity type,
and loses by nine points if it does not — and nobody in the email said which
convention produced "B has better F1."

| Scoring | Winner | Gap (relaxed F1) | Gap (macro F1) |
|---|---|---:|---:|
| Label-strict — the type must match | A | −8.76 pts | −9.60 pts |
| Label-blind — offsets only | B | +1.44 pts | +1.11 pts |

Full reasoning in [Step 6](#step-6--recommendation).

---

## Step 0 — Harness and joins

Alignment machinery reused from `01-detector`: greedy one-to-one matching
ranked by `(exact, same_label, overlap)`, so a prediction straddling two gold
spans claims the one it agrees with rather than the one it merely covers most.

| Check | Result |
|---|---|
| Documents | 2,500 |
| Gold spans | 7,884 |
| Model A spans | 7,812 |
| Model B spans | 9,337 |
| Predictions with unknown `doc_id` | **0** |
| Predictions with invalid offsets | **0** |
| Gold offset mismatches | **0** |
| Label space | Both models use all 56 gold types; no unknown labels |

No label-space reconciliation is needed — unlike problem 01, predictions and
gold share a vocabulary exactly. Baseline is sound.

## Step 0.5 — What the two models look like

```
doc 203039
  TEXT: Thanks for the mentorship, BudBartell. You have been an amazing mentor.
  gold: [('FIRSTNAME','Bud'), ('LASTNAME','Bartell')]
     A: [('FIRSTNAME','Bud'), ('LASTNAME','Bartell'), ('COUNTY','You')]
     B: [('FIRSTNAME','p, BudBar'), ('CURRENCYSYMBOL','Bartell')]

doc 182610
  TEXT: Dear Ladarius, your online submission from IP a4aa:aa28:6cc5:...
  gold: [('FIRSTNAME','Ladarius'), ('IP','a4aa:aa28:6cc5:...')]
     A: [('FIRSTNAME','Ladarius')]
     B: [('EMAIL','Dear'), ('FIRSTNAME','Ladarius'), ('IP','a4aa:aa28:6cc5:...')]

doc 192279
  gold: [('JOBAREA','Operations'), ('JOBTYPE','Manager'), ('CITY','La Mesa')]
     A: [('JOBAREA','Operations'), ('BIC','This'), ('CITY','La Mes')]
     B: [('JOBAREA','Operations'), ('JOBAREA','This'), ('CITY','La Mesa')]
```

The profiles differ in kind, as the brief implied:

- **A under-captures and omits.** `'La Mes'` for `La Mesa`; the IP in 182610
  missed entirely.
- **B over-captures and mislabels.** `'p, BudBar'` spilling into neighbouring
  text; `CURRENCYSYMBOL` attached to a surname with correct offsets.
- **Both fire spuriously**, but on different things — B concentrates on
  capitalised tokens (`'Dear'`, `'Week'`, `'Please'`).

## Step 1 — Does the claim reproduce?

### Label-strict — the predicted type must match gold

| Model | Spans | P exact | R exact | F1 exact | P rel | R rel | F1 rel | F1 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 7,812 | 0.7636 | 0.7566 | 0.7601 | 0.8263 | 0.8187 | **0.8225** | **0.7965** |
| B | 9,337 | 0.6260 | 0.7414 | 0.6788 | 0.6777 | 0.8026 | 0.7349 | 0.7005 |

**A ahead by 8.76 points relaxed, 9.60 macro.**

### Label-blind — offsets only, the redaction view

| Model | Spans | P exact | R exact | F1 exact | P rel | R rel | F1 rel | F1 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 7,812 | 0.7636 | 0.7566 | 0.7601 | 0.8263 | 0.8187 | 0.8225 | 0.7965 |
| B | 9,337 | 0.7133 | 0.8447 | 0.7735 | 0.7718 | **0.9140** | **0.8369** | 0.8076 |

**B ahead by 1.44 points relaxed and 1.11 points macro.**

### What drives the reversal

Three facts, in order of size:

1. **B finds substantially more PII.** Label-blind recall **0.845 vs 0.757** —
   nearly nine points more entities located.
2. **B mislabels enough of them to erase that.** The same recall measured
   label-strict falls to **0.741, below A's 0.757.** Roughly ten points of
   recall is lost purely to wrong type labels on spans whose offsets are right.
3. **B fires more and hits less often.** 9,337 spans to A's 7,812, with
   precision 0.713 against A's 0.764 even in the generous view.

**Micro and macro agree in the label-blind view** — +1.44 and +1.11 points
respectively. B is modestly ahead there by both measures.

Both models score lower on macro than micro (A: 0.7965 vs 0.8225) because the
rare entity types score worst — `PIN` 0.625, `LITECOINADDRESS` 0.660,
`CREDITCARDCVV` 0.733 for A. Macro gives a 40-span type the same weight as an
821-span one, so volume cannot hide them. For PII that matters: the rare types
are often the dangerous ones, which is why macro is reported alongside micro
rather than instead of it.

---

## Step 2 — Is either gap bigger than measurement noise?

Paired bootstrap, 2,000 resamples, documents as the resampling unit. Both
models scored on the **same** resample each iteration — discarding the pairing
would inflate the interval, since the two models agree on most documents.

Alignment is computed once per document and the bootstrap runs over the
resulting sufficient statistics, so resampling costs nothing beyond summation.

| Scoring | Metric | Gap B−A | 95% CI | Width | B ≤ A in |
|---|---|---:|---|---:|---:|
| Label-strict | micro F1 | −8.76 | [−9.65, −7.92] | 1.73 | **100.0%** |
| Label-strict | macro F1 | −9.60 | [−10.60, −8.59] | 2.01 | **100.0%** |
| Label-blind | micro F1 | +1.44 | [+0.76, +2.09] | 1.33 | 0.0% |
| Label-blind | macro F1 | +1.11 | [+0.29, +1.96] | 1.66 | 0.5% |

**Every interval excludes zero.** Both gaps are real. The models genuinely
differ, and they genuinely differ in opposite directions depending on the
scoring convention.

### Two consequences

**1. This cannot be resolved with more data.** A larger test set shrinks both
intervals; it does not make them agree. The disagreement is structural, not
statistical — it is a disagreement about what counts as a correct answer, and
no amount of measurement settles a definitional question.

**2. The two effects are wildly different in size.** A's advantage when labels
matter is **six times larger** than B's advantage when they do not: −8.76
against +1.44 micro, −9.60 against +1.11 macro.

That asymmetry is decision-relevant. Picking B requires being confident that
entity type carries *no* value whatsoever, and the reward for being right is
about 1.4 points. Picking A costs at most 1.4 points if type turns out to be
worthless, and gains 8.8 if it is not.

This is the shape of the recommendation, though the cost model in Step 4 is
what makes it an argument rather than an observation.

### Contrast with the usual case

The instinct with a two-point F1 difference is that it is noise. Here it is
not — at n = 2,500 the intervals are 1.3–2.0 points wide and all four exclude
zero. Sanity-checking the VP's premise does **not** mean discovering the gap is
imaginary; the gap is real, and the problem is that there are two of them
pointing in opposite directions.

---

## Step 4 — Does the ranking survive a cost model that isn't 1:1?

F1 weights precision and recall equally, which assumes a miss and a false
positive cost the same. For PII they plainly do not. Two formalisations of the
asymmetry: F-beta (a ratio) and a linear cost `r x FN + FP` (additive).

**β is how many times more important recall is than precision.** As a weighted
harmonic mean, `1/F_β = [1/(1+β²)]·(1/P) + [β²/(1+β²)]·(1/R)` — so the weights
go as β², not β. β = 1 is F1 (equal); β = 2 puts 4× the weight on recall;
β = 0.5 puts 4× the weight on precision. β is not chosen abstractly — it is
elicited: *how many false alarms would you accept to catch one more real
entity?* An answer of ten means β² = 10, so β ≈ 3.2.

### Label-strict — A dominates

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A | 6,455 | 1,357 | 1,429 | **0.8263** | **0.8187** |
| B | 6,328 | 3,009 | 1,556 | 0.6777 | 0.8026 |

**A is higher on both axes simultaneously.** This is not a trade-off to be
resolved by weighting — it is dominance. F-beta confirms it across the range:

| β | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F_β A | 0.8258 | 0.8248 | 0.8236 | 0.8225 | 0.8211 | 0.8202 | 0.8195 | 0.8190 |
| F_β B | 0.6840 | 0.6995 | 0.7180 | 0.7349 | 0.7596 | 0.7741 | 0.7881 | 0.7970 |
| Winner | A | A | A | A | A | A | A | A |

**No cost model, threshold or weighting prefers B when entity type matters.**
Even at β = 5 — recall weighted 25× over precision — B reaches only 0.7970
against A's 0.8190.

### Label-blind — a genuine trade-off, resolved at 1.03

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A | 6,455 | 1,357 | 1,429 | **0.8263** | 0.8187 |
| B | 7,206 | 2,131 | 678 | 0.7718 | **0.9140** |

Neither dominates: A has precision, B has recall. Solving `r x FN_A + FP_A =
r x FN_B + FP_B` gives the ratio at which they tie:

> **The models tie when one miss costs 1.03 false positives.**
> Cheaper than that, A wins. Dearer, B wins.

| β | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F_β A | 0.8258 | 0.8248 | 0.8236 | 0.8225 | 0.8211 | 0.8202 | 0.8195 | 0.8190 |
| F_β B | 0.7789 | 0.7966 | 0.8176 | **0.8369** | 0.8650 | 0.8815 | 0.8975 | 0.9076 |
| Winner | A | A | A | **B** | B | B | B | B |

F-beta flips between β = 0.75 and β = 1.0 — the same place the linear cost
model puts the tie (r = 1.03). Two different formalisations landing together.
(They are not equivalent objectives — one is a ratio, one additive — so exact
agreement is not expected; proximity is the finding.)

For PII, a missed SSN against an over-redacted word is not 1.03 : 1 — it is
orders of magnitude. **If entity type is worthless, B wins for any defensible
cost model.**

B buys those 751 extra true entities with 774 extra false positives: 2,131
against A's 1,357, **57% more over-redaction**.

### How sensitive is each model's score to the choice of β?

| | β = 0.25 | β = 5.0 | Range |
|---|---:|---:|---:|
| A | 0.8258 | 0.8190 | **0.007** |
| B (label-blind) | 0.7789 | 0.9076 | **0.129** |

**A's score is flat; B's swings by 19× as much.** That is a direct readout of
balance — A has precision ≈ recall (0.826 vs 0.819), so reweighting barely
moves it, while B is lopsided (0.772 vs 0.914) and highly sensitive to how the
two are weighted.

This is an argument about the measurement, not the models: **A's number means
roughly the same thing regardless of who computes it, and B's depends heavily
on the evaluator's assumptions.** One more reason a single F1 was never going
to settle this.

### The decision structure

| If entity type… | Then | On what strength |
|---|---|---|
| matters at all | **A** | Dominance — no cost model can prefer B |
| is worth nothing | **B** | Any cost model where a miss exceeds 1.03× a false positive |

**The entire decision reduces to one question the email never asked:** does a
wrong type label on a correctly-located span cost anything?

Combined with Step 2's asymmetry — A's advantage when type matters is six times
B's advantage when it does not — the risk is lopsided. Choosing B is a bet that
entity type is worth exactly zero, paying 1.4 points if right and costing 8.8
if wrong.

---

## Step 4b — What the FP column was hiding, and severity

### Splitting false positives

The Step 4 panels were not commensurable. Under label-strict, B shows 3,009
false positives; under label-blind, 2,131. The 878-span difference was being
treated as equivalent to firing on empty text. It is not: **a mislabelled
entity is still redacted.** It costs nothing on privacy and costs a
wrong-typed replacement on utility — a repairable defect, not a leak.

Splitting every prediction three ways by offsets alone:

| Model | Located + correctly typed | Located, wrong type | Spurious (no gold at all) |
|---|---:|---:|---:|
| A | 6,455 | **0** | 1,357 |
| B | 6,328 | **878** | 2,131 |

`strict FP = spurious + type-confused` · `blind FP = spurious`

**Model A produces zero type confusions.** Its 1,357 is identical in both
panels, which is why only B's numbers moved. This single split reconciles the
two panels and is the bridge between them.

The honest restatement of B's precision deficit: B fires on empty text 774
times more than A, and mislabels 878 entities it correctly located. Only the
first is a privacy-relevant false positive.

### Misses by severity

A missed `JOBAREA` and a missed `PASSWORD` are not the same event. Severity
tiers reused unchanged from problem 01.

| Tier | Gold | FN A | FN B | Miss rate A | Miss rate B |
|---|---:|---:|---:|---:|---:|
| **HIGH** | 3,193 | 677 | 289 | **21.2%** | **9.1%** |
| MEDIUM | 3,245 | 536 | 251 | 16.5% | 7.7% |
| LOW | 1,446 | 216 | 138 | 14.9% | 9.5% |

**A's miss rate is worst on the highest-severity tier** — 21.2%, above its own
18.1% aggregate — while B's is essentially flat across tiers (9.1 / 7.7 / 9.5).
A misses **2.3× more** high-severity entities than B; on low-severity the ratio
is only 1.6×.

The aggregate F1 conceals this entirely. A's privacy deficit is concentrated
exactly where leaks are expensive.

### The tie point, recomputed

`cost = r × (entities missed) + 1 × (spurious spans)`

| Scope | FN A | FN B | Extra spurious from B | Tie at |
|---|---:|---:|---:|---:|
| All severities | 1,429 | 678 | 774 | r = 1.03 |
| **HIGH only** | 677 | 289 | 774 | **r = 1.99** |

The high-severity figure is **higher**, not lower, and the reason is a mixed
denominator: the numerator counts all 774 extra spurious spans while the
denominator counts only high-severity misses. Read correctly it says **B trades
774 extra spurious spans for 388 fewer high-severity misses** — so a
high-severity miss must cost at least two spurious spans for B to win.

A leaked password against two over-redacted words clears that by orders of
magnitude. Restricting to the entities that matter does not weaken the case for
B; it sharpens what B is buying.

### The asymmetry that resolves the two panels

Not a β value, and not a cost ratio:

> **Leaks are permanent. Type errors are repairable.**

A missed SSN is in the released corpus forever — there is no downstream stage
that recovers it, because nothing knows it is there. A city replaced with a
surname is a visibly wrong row: a human spots it, and a type-consistency pass
or validator layer fixes it **without re-running detection**.

That is why the two panels do not deserve equal weight. The label-strict panel
measures a failure with a cheap remedy; the label-blind panel measures one with
none. B's 878 type confusions are work for a correction stage. A's 388 extra
high-severity misses are unrecoverable.

**This points at B with a correction stage — not at either model bare.** The
recommendation is therefore not "A" or "B" but a model plus a pipeline
position, which is what Step 6 has to say.

---

## Step 6 — Recommendation

### Ship B, with a type-consistency validator downstream

Not "B", and not a matrix. B is the right detector; its known weakness has a
cheap downstream fix and A's does not.

### The three facts it rests on

**1. A leaks 2.3× more of what matters.** A misses 677 high-severity entities;
B misses 289. That is **388 additional permanent exposures** in A's output —
passwords, account numbers, SSNs, addresses. A's miss rate is *worst* on the
highest-severity tier (21.2%, against its own 18.1% average), so the aggregate
F1 understates the gap precisely where it is most expensive.

**2. B's precision deficit is half illusory.** Of the 1,652 B predictions that
are not clean hits, **878 are entities B located correctly and typed wrongly.**
Those still get redacted. Only 774 are genuinely spurious — B fires on empty
text 774 times more than A, which is real but is over-redaction, not exposure.

**3. The trade is 774 extra spurious spans for 388 fewer high-severity
misses.** Break-even requires a high-severity miss to cost ~2 over-redacted
spans. A leaked password against two masked words is not 2 : 1; it is orders of
magnitude.

### Why the validator, and why it settles it

**Leaks are permanent. Type errors are repairable.**

A missed SSN is in the released corpus forever — nothing downstream recovers
it, because nothing knows it is there. A city labelled as a surname is a
visibly wrong row: format-implausible type assignments are detectable, and a
validator or type-consistency pass repairs them **without re-running
detection.**

That is why A's nine-point lead under label-strict scoring is the less
important number. It measures a class of failure with a cheap remedy. B's lead
under label-blind scoring measures one with none.

### What this costs

- **57% more over-redaction** — 2,131 spurious spans against A's 1,357.
- **878 wrong type labels**, until the validator stage exists.
- **A validator to build.** Not costed here; if it cannot be built, see below.

### What would change this recommendation

| If… | Then |
|---|---|
| A wrong entity type triggers an **irreversible** downstream action — routing to a different retention policy, a legal hold | **A.** Type errors stop being repairable and the asymmetry collapses |
| **No correction stage can be built** | The case weakens sharply — B's 878 type errors ship uncorrected, and A's dominance under label-strict becomes the live consideration |
| Over-redaction has a **hard utility ceiling** | 774 extra spurious spans may breach it; re-price with that constraint |
| The predictions feed **pure redaction** with no type-dependent logic | **B alone**, no validator needed — the type errors cost nothing |

### Confidence, and what was not checked

The detection comparison is solid: n = 2,500, all bootstrap intervals exclude
zero, and the severity split is a direct count rather than an estimate.

Two gaps, both stated rather than hidden:

- **Step 3 was skipped.** The five-bucket error profile per severity tier was
  not run. It would refine *where* each model's errors sit, not which model
  leaks more — that is already counted.
- **Step 5 was not run.** Score calibration is unchecked, so **"B with a
  confidence threshold" is an option I cannot price.** If B's spurious
  predictions carry separable confidence, thresholding could recover much of
  the precision gap and the over-redaction cost would drop. Worth an hour
  before committing to the validator design.

The recommendation also rests on the severity tiering carried over from
problem 01, which is a judgement, and on the leaks-permanent asymmetry, which
is a principle rather than a measurement. Both are stated so they can be
argued with.

---

### Draft reply to the VP

> B, with one condition.
>
> B finds substantially more PII than A — the gap that matters is
> high-severity entities, where A misses 677 and B misses 289. Those 388 extra
> misses are permanent: once the corpus ships, nothing downstream recovers
> them.
>
> The reason the team has been split is real. On our eval set B looks worse on
> precision, but roughly half of that is B correctly finding an entity and
> giving it the wrong type — which still gets redacted. Wrong types are
> visible and fixable with a validator pass; missed entities are neither.
>
> So: ship B, and add a type-consistency check downstream. Cost is about 57%
> more over-redaction than A, which is a data-quality expense rather than a
> privacy one.
>
> One thing I have not checked and would want a day for: whether B's
> confidence scores are calibrated well enough to threshold. If they are, most
> of the over-redaction cost goes away and the validator gets simpler.
>
> This flips to A if a wrong entity type ever triggers something irreversible
> downstream — a retention policy or a legal hold. Worth confirming before we
> commit.

---

## Against the belief criteria

Set in `PLAN.md` before any measurement:

| | Criterion | Status |
|---|---|---|
| 1 | The F1 advantage reproduces | **Partially** — label-blind only; micro +1.44, macro +1.11 |
| 2 | The advantage survives the matching rule | **Fails** — the label rule reverses it by ~10 points |
| 3 | Larger than measurement noise | **Passes — for both gaps.** All four intervals exclude zero |
| 4 | F1 is the right objective | **Fails as posed** — under label-strict A dominates so the objective is irrelevant; under label-blind the crossover sits at 1.03, far below any plausible PII cost ratio. F1's implicit 1:1 is wrong either way |
| 5 | Not concentrated in low-severity types | **Not measured** — Step 3 skipped |

Criterion 2 has failed and criterion 3 has passed, which together sharpen the
problem rather than resolving it: **both gaps are real.** The honest
restatement of the VP's premise is *"B has better F1 under one scoring
convention by 1.4 points, and worse under the other by 8.8 — both differences
are statistically solid."*

---

## Not run

| Step | What it would establish | Status |
|---|---|---|
| 3 | Error profiles in the five buckets, per severity tier | **Skipped.** Would say whether either model's errors concentrate in high-severity types — the one thing that could complicate the Step 4 decision structure |
| 5 | Whether the scores are calibrated enough for thresholding to be an option | **Not run.** The one open item that could materially improve the chosen option |

## Judgment calls on the record

| # | Decision | Rationale |
|---|---|---|
| 1 | Alignment reused unchanged from problem 01 | Greedy one-to-one, ranked by (exact, same_label, overlap). Rewriting it would spend the hour re-deriving a solved sub-problem |
| 2 | Both label-strict and label-blind reported, neither privileged yet | They rank the models oppositely. Choosing one before knowing what consumes the predictions would be choosing the answer |
| 3 | Micro and macro both reported | With 56 types, macro weights a 40-span type equally with an 821-span one. They agree in direction here, which is itself worth knowing — the label-strict/label-blind split is doing all the work, not the averaging choice |
| 4 | Relaxed matching leads, exact reported alongside | The exact/relaxed gap measures boundary error; neither alone characterises a model that over- or under-captures |
| 5 | A matched prediction counts toward the **gold** label's precision denominator | Under label-blind matching the predicted and gold labels can differ. Crediting the TP to one label while counting the prediction under another makes per-label precision incoherent. **Corrected after an initial run reported the label-blind macro gap as +0.11 instead of +1.11**; label-strict was unaffected |

## Reproducing this

```bash
python load_data.py       # cached sample, skips if present
python build_models.py    # regenerates both prediction files
python evaluate.py        # every figure above, in order
```
