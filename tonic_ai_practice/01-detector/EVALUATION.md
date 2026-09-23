# PII Detector Evaluation

**Verdict: do not ship.** Steps 0–4 of the 8-step plan were completed; steps
5–8 were not run. The verdict does not depend on the unfinished work — see
[Not run](#not-run) for why it could only strengthen the conclusion.

---

## Verdict

**Do not ship.** Not "ship with caveats" — the gap is too large for mitigation.

The corpus is going to a model-training vendor, so every miss is permanent:
trained into weights, not retractable, and potentially re-emittable by the
resulting model. That asymmetry means the decision turns on coverage of
high-severity entities, and this detector does not have it.

### The evidence that forces it

**1. 78.4% of high-severity entities are never touched by any prediction.**
This is *coverage* — the most generous reading available, ignoring labels
entirely and crediting the detector whenever it redacts the right characters
under any name. 1,901 high-severity gold spans; 21.6% touched.

**2. Credit card numbers: 0 of 77.** An entire high-severity class where the
pattern exists and cannot fire — it requires space separators and every card in
the corpus is 16 unbroken digits. This is the finding that should end any
argument from the smoke test: a total failure that a hand-written test passed.

**3. 60% of documents contain a high-severity entity with no pattern at all.**
Plaintext passwords (96), street addresses (94), account numbers (92),
usernames (90), IP addresses (168). This is a floor established *before* any
matching — no regex tuning, no threshold, no post-processing moves it. Seven
patterns cannot cover fifty-six entity types.

### The one number to re-measure on the customer's data

**The fraction of their tickets containing at least one untouched
high-severity entity.**

The figures here come from synthetic, template-generated text with a
deliberately flat type distribution — it carries bitcoin addresses and vehicle
VINs at rates no support queue does. The *direction* is robust: passwords,
addresses, account numbers and usernames have no pattern, and those
unquestionably appear in real tickets. The *magnitude* is not transferable.
Roughly 500 of their actual tickets, hand-labelled, would settle it.

Everything else in this report — F1, per-type recall, the precision tables —
is diagnostic. That one number is the product metric.

### The one fix to do first

**Widen the credit card pattern to accept hyphens and unbroken digits.** One
line. It recovers 77 high-severity entities from zero, and it is free in
precision terms: the format regexes already run at 100% exact precision with
zero spurious output, so widening them costs nothing. It is also the cheapest
test that the evaluation harness measures deltas correctly, before effort goes
into anything larger.

**It does not change the decision.** The real remediation is adding patterns
for the structured high-severity types that have none — IPv4, IPv6, MAC, IBAN,
crypto addresses — all rigid formats, together roughly 480 more high-severity
spans. Even that leaves passwords, street addresses and free-form names, which
regexes handle badly. This is a re-architecture, not a patch.

### One warning for whoever fixes it

`PERSON_RE` incidentally redacts 89 street addresses through `Cedar Road`-style
matches. Tightening it to fix its 74% error rate would **reduce** high-severity
coverage. Any fix must be measured on coverage, not precision, or the metrics
will improve while the product gets worse.

---

## The question

A customer wants to run `detector.py` over a corpus of support tickets before
handing that corpus to a model-training vendor. Is that safe, and if not,
what is wrong and what should be fixed first?

The use shapes the whole evaluation. A **missed** entity is trained into model
weights and cannot be retracted; a **spurious** redaction costs data quality
and is recoverable. The two error types are not symmetric, so aggregate F1 is
the wrong headline number.

## What is being evaluated

`detector.py` is 22 lines: six format regexes (`EMAIL`, `PHONE`, `SSN`,
`DATE`, `CREDITCARD`, `ZIP`) plus `PERSON_RE`, which matches any two
consecutive capitalised ASCII words. Each pattern runs independently and
results are concatenated — no overlap resolution, no deduplication, output in
pattern order rather than position order, and spans carry only
`{start, end, label}`.

It is treated as read-only. Any fix lands in a wrapper, not in the file.

## Method

| Step | What | Status |
|---|---|---|
| 0 | Harness + offset round-trip check | Done |
| 0.5 | Read 20 documents by eye | Done |
| 1 | Label reconciliation (56 gold types vs 7 detector labels) | Done |
| 2 | Severity tiering, fixed before results | Done |
| 3 | Matcher — exact and relaxed, one-to-one | Done |
| 4 | Error bucketing per type and per severity | Done |
| 5 | Residual-leak check (what survives redaction) | Not run |
| 6 | Product metrics — per-document leak rate | Not run |
| 7 | Failure attribution by surface form | Not run |
| 8 | Rank fixes, implement one, measure the delta | Not run |

---

## The data

`ai4privacy/pii-masking-200k`, filtered to English, 1,500 documents sampled
with a fixed seed from the 43,501 available (3.4%). Cached locally as JSONL;
no network calls during evaluation.

| Property | Value |
|---|---|
| Documents | 1,500 |
| Document length | 48–2,408 chars, median 166 |
| Gold spans | 4,688 raw → 4,576 after name merge |
| Gold entity types | 56 |
| Documents with zero PII | 0 |
| Detector predictions | 1,438 |

**The gold is clean, and this is load-bearing.** `source_text[start:end]`
equals the recorded value for all 4,688 spans — zero mismatches — so the
evaluator works on raw character offsets with no normalisation. There are no
overlapping gold spans, no zero-length spans and no duplicates. There are
**89 adjacent pairs** where one span ends exactly where the next begins.

### Format variety per type

Values were reduced to an `A`/`a`/`9` skeleton and distinct shapes counted.

| Type | Values | Distinct shapes | Note |
|---|---:|---:|---|
| CREDITCARDNUMBER | 77 | **1** | All 16 unbroken digits |
| PHONENUMBER | 79 | **59** | Near-total variety |
| SSN | 71 | 4 | Only 20 use hyphens |
| ZIPCODE | 79 | 2 | 39 plain, 40 ZIP+4 |
| DATE | 160 | 27 | |
| DOB | 86 | 21 | Top shape identical to DATE's |

Two consequences. **DOB and DATE are not separable by format** — their most
common shape is the same, so no format-based detector can distinguish a birth
date from a delivery date. And the shape metric is length-sensitive, so it
overstates variety for variable-length content such as email addresses; it is
exact for fixed-format types.

### Corpus realism — a limit on external validity

The documents are synthetic and template-generated:

> *"Arts Week is coming to town! We have registers from all communities in
> Herefordshire."*

Clean prose, no typos, no ALL-CAPS, no signature blocks, no quoted reply
chains, uniform entity density, and a deliberately flat type distribution —
`BITCOINADDRESS` (79) and `VEHICLEVIN` (31) appear at rates nothing like a real
ticket queue.

This corpus is well suited to **finding bugs**, which is the present task. It is
not a basis for forecasting production recall. Figures below that depend on the
type mix are marked corpus-specific.

---

## Step 1 — Label reconciliation

Gold uses 56 fine-grained types; the detector emits 7 names matching none of
them. Both sides map into one shared evaluation space, and every span keeps its
original label so severity can still be read on gold's own terms.

Gold name parts are merged first: a run of `FIRSTNAME`/`MIDDLENAME`/`LASTNAME`
separated by nothing or whitespace becomes one `PERSON` span, because the
detector emits one span per name and would otherwise be scored against three.

**The merge was smaller than expected.** Only 203 of 756 name-part spans (27%)
are whitespace-adjacent to another; the rest stand alone. 91 merged PERSON
spans were built from those 203 parts (70 runs of 2, 21 of 3), a net reduction
of 112 gold spans.

### Coverage ceiling

| | Count | Share |
|---|---:|---:|
| Gold spans the detector could match | 1,335 | **29.2%** |
| Gold spans with no pattern at all | 3,241 | 70.8% |
| Documents where every gold span is unaddressable | 605 / 1,500 | 40% |

**A flawless implementation of these seven patterns caps at 29.2% span recall
on this corpus.** Everything the matcher measures happens inside that ceiling.

Addressable gold, post-merge: PERSON 644, DATE 246, EMAIL 139, ZIPCODE 79,
PHONENUMBER 79, CREDITCARDNUMBER 77, SSN 71.

---

## Step 2 — Severity

Tiers were fixed before any per-type result was computed. The principle is
**identifiability**, not how sensitive a label sounds: HIGH means the entity
alone identifies one person or grants access; MEDIUM narrows to a small group;
LOW does not meaningfully narrow.

| Tier | Gold | No pattern | Addressable |
|---|---:|---:|---:|
| HIGH | 1,901 | 1,276 (67%) | 625 |
| MEDIUM | 1,819 | 1,109 (61%) | 710 |
| LOW | 856 | 856 (100%) | 0 |

**895 of 1,500 documents (60%) contain a high-severity entity that
`detector.py` has no pattern for.** This is a floor on the per-document leak
rate established without any matching — matcher quality cannot move it.

High-severity types with no pattern at all:

| | | | |
|---|---|---|---|
| PASSWORD 96 | IPV4 96 | STREET 94 | IPV6 93 |
| ACCOUNTNUMBER 92 | USERNAME 90 | BITCOINADDRESS 79 | IBAN 76 |
| BUILDINGNUMBER 73 | IP 72 | SECONDARYADDRESS 69 | ETHEREUMADDRESS 61 |
| NEARBYGPSCOORDINATE 57 | PHONEIMEI 51 | MAC 50 | VEHICLEVIN 31 |
| CREDITCARDCVV 25 | LITECOINADDRESS 25 | PIN 23 | VEHICLEVRM 23 |

**In the detector's favour:** 100% of the LOW tier is out of vocabulary. The
detector targets no low-severity type at all, which is correct prioritisation.
Its seven patterns are aimed at the right tier — there are simply seven of them.

### A limitation of this scheme, stated plainly

The table tiers entities **in isolation** while the real risk is
**combinatorial**. ZIP + date of birth + sex re-identifies roughly 87% of US
adults, and all three occur in this corpus, yet each is tiered separately.
A per-entity table cannot express that. It needs a document-level rule, which
is deferred to Step 6.

---

## Step 3 — Matching

Greedy one-to-one alignment. Candidate pairs rank by `(exact, same_label,
overlap)`, so a prediction straddling two gold spans claims the one it *agrees
with* rather than the one it merely covers most. Overlap is strict, so the 89
adjacent gold pairs cannot be fused.

Predictions are matched against **all** gold, including out-of-vocabulary types:
a `ZIP` prediction landing on a `BUILDINGNUMBER` did redact a high-severity
entity, just under the wrong name, and scoring it as spurious would understate
the detector. `spurious` therefore means *touches no gold at all*.

| Gold side (4,576) | | Prediction side (1,438) | |
|---|---:|---|---:|
| correct | 246 | correct | 246 |
| boundary | 355 | boundary | 355 |
| type_confusion | 638 | type_confusion | 638 |
| missed | 3,337 | spurious | 199 |

Of the 199 spurious, **23 are redundant** — they touch gold that another
prediction already claimed, which for redaction is harmless — and **176 touch
no gold at all**.

An edge-case self-test covers adjacent spans, a prediction straddling two gold
spans, one gold claimed by two predictions, duplicate identical predictions,
non-overlapping predictions and zero-length spans. Bucket totals are asserted
to equal span totals on both sides.

### Two readings

**The boundary gap is large.** Of 601 gold spans where the detector got the
label right, only 246 have correct offsets — **59% of label-correct matches
have wrong boundaries.** Exact-match scoring reports 5.4% recall where relaxed
reports 13.1%. Quoting either alone misrepresents the system by more than 2×.

**The detector rarely fires on pure non-PII.** Only 176 of 1,438 predictions
(12%) touch no gold entity; 88% land on something that genuinely is PII. The
failure is coverage, labels and boundaries — not hallucination.

---

## Step 4 — Errors by type and severity

A third rate is reported alongside exact and relaxed recall: **coverage**,
`(correct + boundary + type_confusion) / gold` — "was this entity touched by
any prediction at all". For a redaction pipeline the label is largely
irrelevant; `La Mesa` tagged PERSON instead of CITY is redacted either way.
Recall measures the detector as a *classifier*; coverage measures it as a
*redactor*, which is what is actually being shipped.

### Cross-validation of the matcher

These counts were produced with no knowledge of the format inventory taken
during data inspection, yet they agree exactly:

| Evaluator output | Independent format inventory |
|---|---|
| ZIPCODE: 39 exact, 40 boundary | 39 plain 5-digit, 40 ZIP+4 |
| SSN: 20 correct, 51 missed | 20 hyphenated, 51 in three other shapes |
| DATE: 76 correct | `99/99/9999` = 44 DATE + 32 DOB = 76 |

Three exact agreements from independent routes. This is the available evidence
that the alignment does what it claims.

### Recall by gold label

| Label | Gold | Correct | Boundary | Confusion | Missed | Exact | Relaxed | **Coverage** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OUT_OF_VOCAB | 3,241 | 0 | 0 | 635 | 2,606 | 0.0% | 0.0% | 19.6% |
| PERSON | 644 | 46 | 237 | 0 | 361 | 7.1% | 43.9% | 43.9% |
| DATE | 246 | 76 | 0 | 0 | 170 | 30.9% | 30.9% | 30.9% |
| EMAIL | 139 | 63 | 76 | 0 | 0 | 45.3% | 100.0% | 100.0% |
| ZIPCODE | 79 | 39 | 40 | 0 | 0 | 49.4% | 100.0% | 100.0% |
| PHONENUMBER | 79 | 2 | 2 | 3 | 72 | 2.5% | 5.1% | 8.9% |
| CREDITCARDNUMBER | 77 | 0 | 0 | 0 | 77 | **0.0%** | **0.0%** | **0.0%** |
| SSN | 71 | 20 | 0 | 0 | 51 | 28.2% | 28.2% | 28.2% |

**`CREDITCARDNUMBER` is 0.0% in every column.** 77 high-severity entities, not
one touched. The pattern requires space separators; every card in the corpus is
16 unbroken digits.

**`PHONENUMBER` is 8.9% coverage** — 72 of 79 missed, against 59 distinct
formats in 79 values.

**`EMAIL` and `ZIPCODE` reach 100% coverage but ~50% exact.** Nothing is
missed; about half have wrong offsets. Whether those boundary errors leak is
Step 5's question, and it decides whether that 100% is real or cosmetic.

### Precision by predicted label

| Label | Predicted | Correct | Boundary | Confusion | Spurious | Exact | Relaxed |
|---|---:|---:|---:|---:|---:|---:|---:|
| PERSON | 1,089 | 46 | 237 | 608 | 198 | 4.2% | 26.0% |
| EMAIL | 139 | 63 | 76 | 0 | 0 | 45.3% | 100.0% |
| ZIPCODE | 110 | 39 | 40 | 30 | 1 | 35.5% | 71.8% |
| DATE | 76 | 76 | 0 | 0 | 0 | 100.0% | 100.0% |
| SSN | 20 | 20 | 0 | 0 | 0 | 100.0% | 100.0% |
| PHONENUMBER | 4 | 2 | 2 | 0 | 0 | 50.0% | 100.0% |

**The two components fail in opposite directions.** The format regexes have
near-perfect precision and poor recall — `DATE` and `SSN` are at 100% exact
precision with zero spurious output. When they fire they are right; they rarely
fire. `PERSON_RE` is the inverse: 1,089 predictions of which 806 (74%) are
wrong-type or spurious, while still missing 361 of 644 people.

This matters for remediation. Widening a format regex costs almost nothing in
precision. Widening `PERSON_RE` would be damaging. `CREDITCARD` does not appear
in this table at all — it fired zero times.

### Recall by severity tier

| Tier | Gold | Exact | Relaxed | **Coverage** |
|---|---:|---:|---:|---:|
| HIGH | 1,901 | 8.2% | 15.0% | **21.6%** |
| MEDIUM | 1,819 | 4.9% | 17.3% | 33.8% |
| LOW | 856 | 0.0% | 0.0% | 24.9% |

**78.4% of high-severity entities are never touched by any prediction** — and
high-severity coverage is *worse* than medium. The detector covers lower-value
entities better than the ones that matter.

### Type confusion, decomposed

| | | | |
|---|---:|---|---:|
| PERSON → JOBTITLE | 111 | PERSON → CITY | 48 |
| PERSON → STREET | 89 | PERSON → CURRENCY | 26 |
| PERSON → ACCOUNTNAME | 83 | PERSON → STATE | 25 |
| PERSON → PREFIX | 58 | ZIPCODE → BUILDINGNUMBER | 21 |
| PERSON → COUNTY | 50 | PERSON → COMPANYNAME | 20 |

Nearly all of it is `PERSON_RE` matching capitalised bigrams that are not
people. But **`PERSON → STREET` (89)** is mislabelling that *helps*: `STREET` is
high-severity with no pattern of its own, so the person-detector is
incidentally redacting home addresses. Part of the 19.6% coverage on
out-of-vocabulary gold is doing real safety work by accident.

This cuts against the obvious remedy. Tightening `PERSON_RE` to fix its
precision would *reduce* high-severity coverage, and any such fix must be
measured on coverage, not on precision alone.

---

## Judgment calls on the record

Every decision below changes a number and is reversible in one line.

| # | Decision | Rationale | Contested? |
|---|---|---|---|
| 1 | Filter to English before sampling | Sampling first yields ~1/6 English | No |
| 2 | 1,500 docs, fixed seed | Reproducible; pre-cached as JSONL | No |
| 3 | `PERSON` = FIRSTNAME + MIDDLENAME + LASTNAME | Detector emits one span per name | No |
| 4 | `PREFIX` excluded from PERSON | "Dr." identifies nobody | Minor |
| 5 | Merge name parts across whitespace-only gaps | Keeps "Alice and Bob" as two people | No |
| 6 | `DOB` collapsed into `DATE` for matching | Format-identical; original label preserved for severity | No |
| 7 | FIRSTNAME/MIDDLENAME = MEDIUM, LASTNAME = HIGH | **Revised.** A bare given name identifies almost nobody; the original all-HIGH call was merge convenience leaking into a risk judgment | Yes — resolved |
| 8 | `ZIPCODE` = MEDIUM | Alone it is an area; the combination argument for HIGH is legitimate | **Yes — open** |
| 9 | `SEX`/`GENDER` = MEDIUM | Tiered on identifiability, not GDPR sensitivity | **Yes — open** |
| 10 | `IPV4`/`IPV6`/`IP` = HIGH | *Breyer*: a dynamic IP is personal data to anyone who can link it | Minor |
| 11 | `ACCOUNTNAME` = LOW | Values are "Home Loan Account" — product names, not people | No |
| 12 | Match predictions against OOV gold | Wrong label still redacted the entity | No |
| 13 | Redundant predictions counted apart from spurious | Double-covering one entity is harmless for redaction | No |
| 14 | Offsets verified on unmerged gold | Merged text is sliced from source and would round-trip trivially | No |

Decision 7 was revised after review. Its effect: HIGH gold fell from 2,372 to
1,901 and HIGH unaddressable rose from 54% to 67% — but **the 60% document
leak floor did not move**, so the headline is robust to the disagreement.

---

## Not run

Steps 5–8 were not reached. None of them could overturn the verdict; the
residual-leak check in particular can only make the picture worse.

- **Residual leak** (Step 5): whether a matched span leaves identifying text in
  the document. A prediction covering part of an entity produces output that
  passes visual inspection and still leaks. Static reading of the regexes
  suggests this happens — the `EMAIL` pattern excludes hyphens, `ZIP` stops at
  five of nine digits, `PERSON_RE` takes two words of a three-word name — and
  Step 4 has now quantified the exposure: **355 boundary errors** (237 PERSON,
  76 EMAIL, 40 ZIPCODE, 2 PHONENUMBER) where the label is right and the offsets
  are not. Until
  these are checked, `EMAIL` and `ZIPCODE` at 100% coverage cannot be read as
  safe.
- Per-document leak rate and over-redaction rate (Step 6).
- Which specific surface forms each pattern misses (Step 7).
- Any fix, or its delta (Step 8).

## Caveats that must travel with these numbers

1. **The 60% leak floor is corpus-specific.** The flat type distribution
   inflates the high-severity denominator with types real tickets rarely carry.
   The *direction* is robust — passwords, street addresses, account numbers,
   usernames and IP addresses have no pattern, and those do appear in support
   tickets. The *magnitude* needs re-measuring on a real ticket sample.
2. **Recall figures here are not production forecasts.** Synthetic text is
   cleaner than real tickets in every respect that matters to a regex.
3. **Combinatorial re-identification is not modelled.** Entities are tiered in
   isolation.
