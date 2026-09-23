# De-identification Pipeline Review

**Verdict: do not release this corpus. The minimum fix is one line, it is
achievable before sign-off, and it is measured — not estimated.**

Steps 0–2 and the fix analysis are complete. Steps 3–4 (replacement consistency
and collisions) were not run; both bear on utility, neither blocks release.

Every claim is demonstrated from `redacted.jsonl` against
`train_en_2000.jsonl`. Nothing is justified by reading `pipeline.py`.

---

## The decision

### Do not release as-is

| | Current corpus |
|---|---:|
| PII values still present verbatim | **19.1%** |
| Documents retaining a high-severity value | **20.6%** |
| Documents with destroyed non-PII text | **68.7%** |

Two independent grounds, either sufficient on its own. The corpus leaks at
three times its stated rate, **and** two-thirds of it is corrupted as data —
so it would fail the partner both on privacy and on the research use it was
released for.

### The minimum fix: apply replacements right to left

One change to the splice order. Replacing from the end of each document
backwards means every edit lands *after* the region touched next, so no offset
is ever invalidated.

Measured by re-running the same replacement decisions in reverse order:

| | Current | Fixed | |
|---|---:|---:|---|
| Residual PII | 19.1% | **6.1%** | matches the documented 6% drop |
| High-severity residual | 19.1% | **6.7%** | |
| Documents with a high-severity leak | 20.6% | **8.2%** | |
| Non-PII context destroyed | 38.1% | **0.0%** | eliminated entirely |

Two things make this the **minimum** fix rather than one of several:

1. **All collateral damage is attributable to the ordering.** It goes to
   exactly zero — not reduced, eliminated. The 68.7% document corruption has a
   single cause, so nothing else needs changing to remove it.
2. **6.1% is the documented drop rate.** The fix does not make the corpus
   better than promised; it makes it match a promise already accepted for this
   data class. Any further change would be improving on the agreed standard,
   which is a different decision and not one this review is asking for.

*Mechanism, worked example and the proof that ordering is the whole cause:*
**Step 3** below.

### Is it achievable before sign-off?

**Yes, with one condition.** The code change is trivial and the pipeline re-run
is minutes. The real cost is **re-validation**: the 6.1% above is a simulation
of the fix, run on reconstructed replacements, not a measurement of a corrected
pipeline's actual output.

Required before release:

1. Apply the ordering change.
2. Re-run the pipeline.
3. **Re-run this evaluation against the new output** and confirm residual PII
   ≈ 6% and context damage ≈ 0%.

**Do not assume step 3.** The entire reason this review exists is that a
plausible number was accepted without anyone checking what it described.
Shipping on a projected 6.1% would repeat exactly that mistake with a different
number.

That sequence fits comfortably before tomorrow.

### Two defects explicitly not fixed

Both remain after the ordering change. **Neither blocks release**, and both are
**unmeasured** — steps 3 and 4 of the plan did not run:

- **Replacement consistency** — replacements are drawn independently per
  occurrence, so one real person may become several different fake people
  within a document. This destroys coreference and degrades the corpus for any
  research involving entity tracking. **Not measured** (Step 3).
- **Pool collisions** — a small fixed pool means many distinct real people
  share one fake identity. This mildly *helps* privacy and harms utility, by
  asserting relationships that never existed. **Not measured** (Step 4).

Both damage **utility**, neither creates **exposure**. They should be fixed;
they should not hold the release. Stated here explicitly rather than omitted,
so the report does not imply the picture is complete when two defects were
never quantified.

### The question the team did not ask

94% is a **detection** metric. It was signed off as a property of stage 1,
while the artifact being released is the output of stage 2. The number was
never wrong — the pipeline did attempt 94.2% of replacements — but an attempt
is not a removal, and nothing in the eval measured outcomes.

**Recommendation beyond this release:** the acceptance metric should be
computed on the released artifact, not on detector output. Had that been in
place, this defect would have been caught by the eval instead of by an analyst
noticing that text looked strange.

One policy point to raise rather than decide: at 6.1% residual, roughly 163
documents still retain a high-severity value, and the current leaks include
matched username/password pairs from the same record. "94% is acceptable for
this data class" was a judgement made about a coverage number, not about that
specific content. It is worth re-confirming with whoever owns the standard.

---

## Headline

**19.1% of PII values are still present, verbatim, in the released text.**
The expected figure, given a detector that drops 6% of spans by design, is
about 6%. The observed rate is roughly three times that.

**412 of 2,000 documents (20.6%) retain at least one high-severity value.**

The pipeline reports replacing 5,932 of 6,295 spans — **94.2%**, exactly the
number in the detection eval. That number is not wrong. It counts
**attempts**, and an attempt is not a removal.

---

## Step 0 — Join and baseline

| Check | Result |
|---|---|
| Join `redacted.jsonl` → source on `doc_id` | **Total** — 2,000 documents, none missing, none extra, no duplicates |
| `original_text` equals `source_text` | **0 mismatches of 2,000** |

The baseline is sound. Whatever is wrong happened in replacement, not in
ingestion, and comparisons against the source are valid.

## Step 0.5 — What the analyst saw

Six documents read side by side. The defect is visible without any measurement:

```
ORIG: ...handled by our Managers. This external agency based in La Mesa can...
RED : ...handled by our M[JOBTYPE]. This external agency based inFairviewa can...

ORIG: Thanks for the mentorship, BudBartell. You have been an amazing mentor.
RED : Thanks for the mentorship, MarJonesll. You have been an amazing mentor.

ORIG: Arts Week ... The opening ceremony is on 21th March. Everyone is invited!
RED : Arts Week ... The opening ceremony is on 21th 01/30/2021ryone is invited!

ORIG: ...password ... temporarily set to C52WgfV_7dmT. Please update it...
RED : ...password ... temporarily set t[PASSWORD]mT. Please update it...
```

**Replacements land progressively further from where they belong within a
document.** `BudBartell` → `MarJonesll` shows the mechanism directly: a
fragment of the first replacement, then the second replacement, then two
stranded characters of the original surname. Neighbouring text is consumed
(`. Eve` disappears from the third example) and characters of the original are
stranded (`mT` — a piece of the real password — in the fourth).

This is what "looks weird" means. It is a correctness failure in splicing, and
it is visible to any reader in seconds, which is consistent with an analyst
noticing it at a glance and being unable to name it.

## Step 1 — Residual PII

For each gold span, is the original value still present verbatim anywhere in
the released text? Offset-independent substring search.

| Tier | Spans | Still present | Rate |
|---|---:|---:|---:|
| HIGH | 2,482 | 474 | **19.1%** |
| MEDIUM | 2,506 | 424 | 16.9% |
| LOW | 863 | 218 | 25.3% |
| **ALL** | **5,851** | **1,116** | **19.1%** |

Values shorter than 4 characters are excluded from the table — substring search
is unreliable there. 201 short values are also present and are not counted
above.

**Documents retaining at least one HIGH-severity value: 412 of 2,000 (20.6%).**

### Where the leaks concentrate

| Label | Leaked | Of | Rate | Tier |
|---|---:|---:|---:|---|
| AMOUNT | 67 | 134 | 50% | LOW |
| ACCOUNTNUMBER | 55 | 127 | **43%** | **HIGH** |
| TIME | 36 | 94 | 38% | MEDIUM |
| ZIPCODE | 30 | 107 | 28% | MEDIUM |
| PHONENUMBER | 29 | 103 | **28%** | **HIGH** |
| PASSWORD | 32 | 125 | **26%** | **HIGH** |
| JOBAREA | 31 | 117 | 26% | LOW |
| SEX | 28 | 120 | 23% | MEDIUM |
| DATE | 45 | 218 | 21% | MEDIUM |
| LASTNAME | 46 | 225 | **20%** | **HIGH** |
| CITY | 28 | 140 | 20% | MEDIUM |
| FIRSTNAME | 88 | 631 | 14% | MEDIUM |

**Over a quarter of plaintext passwords and 43% of account numbers survive the
pipeline intact.**

### Examples from the released corpus

| Document | Type | Value still present |
|---|---|---|
| 182610 | IP | `a4aa:aa28:6cc5:17a5:368b:acc4:deb3:b8b5` |
| 189030 | BITCOINADDRESS | `3g9jHQdNsSn2RZzrD4GpDZigre2` |
| 198874 | IPV4 | `20.18.109.93` |
| 181955 | USERNAME | `Jerald_Reynolds68` |
| 184878 | PASSWORD | `fQYisKST5siX` |
| 174208 | USERNAME | `Westley_Kreiger` |
| 174208 | PASSWORD | `z8QSin0UJV8n` |

Document 174208 leaks a matched username **and** password from the same record.

### Decomposing the 19.1%

Of 5,851 measured spans, the documented 6% detector drop accounts for roughly
350 surviving values. The observed count is 1,116.

**So roughly 765 values — about two-thirds of the leakage — belong to spans the
pipeline counted as successfully replaced.** Those are not detection misses.
They are replacement failures being reported as successes.

### This is a lower bound

Substring search counts a value only if it survives **whole**. The splicing
defect frequently leaves values in fragments — `mT` from a password, `ll` from
a surname — and none of those are counted here. True recoverable information is
higher than 19.1%, by an amount not yet measured.

---

## Step 2 — Replacement integrity

### The defect is positional, and the corpus contains its own control

Leak rate by a span's ordinal position within its document:

| Position | Spans | Leaked | Rate |
|---|---:|---:|---:|
| **#0 (first in document)** | 1,846 | 97 | **5.3%** |
| #1 | 1,885 | 370 | 19.6% |
| #2 | 1,050 | 299 | 28.5% |
| #3 | 558 | 182 | 32.6% |
| #4 or later | 512 | 168 | 32.8% |

**A document's first span has no preceding replacement, and it leaks at 5.3% —
the detector's documented 6% drop rate.** Every subsequent span is worse, rising
monotonically to roughly 33% and plateauing.

This is the mechanism, established without reading the pipeline: **each
replacement displaces the text, so every following replacement lands at an
offset that is no longer correct.** The further into a document a span sits,
the further off its replacement lands.

It also validates the measurement. If verbatim substring search were
over-counting, position #0 would not land on the documented drop rate.

(The corpus has no single-span documents — the minimum is 2 — so that version
of the experiment is unavailable. Position #0 serves the same purpose.)

### Collateral damage to non-PII text

The text *between* gold spans is not PII and should survive untouched.

| | Count | Share |
|---|---:|---:|
| Context segments (≥ 12 chars) | 4,971 | |
| No longer present verbatim | **1,895** | **38.1%** |
| **Documents with damaged context** | **1,374** of 2,000 | **68.7%** |

**More than two-thirds of documents have had non-PII content destroyed.** This
is no longer only a privacy defect — the corpus is corrupted *as data*.
Sentences lose words, clauses are truncated, and the damage is invisible to
anyone who does not have the original to compare against.

### Documents are losing text

Length change, released minus original:

| min | p25 | median | p75 | max |
|---:|---:|---:|---:|---:|
| −203 | −17 | −5 | +2 | +41 |

The distribution is skewed toward loss. Against a median document length of
166 characters, a 203-character deletion means substantial passages of some
documents have simply been consumed.

---

## Step 3 — Mechanism proof and why the fix works

### An edit only invalidates what is to its right

Replacing characters `[start, end)` with a string of a different length shifts
everything after `end`. Everything before `start` is untouched. That asymmetry
is the entire explanation.

- **Left to right** edits the smallest offset first, invalidating every larger
  offset — which is precisely the set still to be used. Error accumulates: by
  span *k*, the drift equals the total length change of all *k−1* preceding
  replacements.
- **Right to left** edits the largest offset first, invalidating only offsets
  greater than it — which have all been used already. Every remaining span
  sits to the left and is still correct. Drift is structurally impossible.

#### Worked example — document 192162

```
ORIGINAL: ...with address d6:b7:08:9f:35:c1. Make sure all necessary health
          parameters like 3feet8inches and Eye color: Hazel are being monitored.

  span [ 87,104] MAC       'd6:b7:08:9f:35:c1'
  span [153,165] HEIGHT    '3feet8inches'
  span [170,186] EYECOLOR  'Eye color: Hazel'
```

**Left to right:**

1. Replace `[87,104]` — 17 characters — with `[MAC]`, 5 characters.
   **The document is now 12 characters shorter, so every offset past 104 is
   stale by −12.**
2. Replace `[153,165]`. In the shortened text, `3feet8inches` now lives at
   `[141,153]`. Slicing `[153,165]` instead grabs the *next* 12 characters —
   `" and Eye col"` — and replaces those with `[HEIGHT]`.

Result: `...3feet8inches[HEIGHT]or: Hazel[EYECOLOR]ored.`

**Both failure modes in one step.** `3feet8inches` — the PII — survives
completely untouched. `" and Eye col"` — ordinary non-PII prose — is destroyed.
The drift then compounds into the third span.

**Right to left:**

1. Replace `[170,186]` first. Only offsets past 186 shift; nothing still
   needed lives there.
2. Replace `[153,165]`. Untouched, because step 1 edited further right. Correct.
3. Replace `[87,104]`. Untouched. Correct.

All three land exactly where intended.

### Proof: predicting the released text from the source

Not by reading the pipeline, and not only by simulation.

The released text reveals which labels were rendered as a literal `[LABEL]`
placeholder — 45 of them are visible in the output. For documents where *every*
span used one **and** `spans_replaced` equals the gold span count (nothing
dropped), the redacted text becomes fully predictable with no knowledge of any
replacement pool.

**610 such documents.** Predicting each one under both models:

| Model | Reproduces the actual released text exactly |
|---|---:|
| **Left to right** | **610 of 610 — 100.0%** |
| Right to left | 8 of 610 — 1.3% |

A model that reproduces 610 of 610 documents byte-for-byte is not a hypothesis
about the defect. It is the defect, identified from the output alone — and the
same construction run in reverse is what produces the corrected figures in
*The minimum fix* above.

This is what discharges the contamination rule in `PROBLEM.md`: the diagnosis
rests on a model validated against 610 documents, not on privileged knowledge
of the implementation. An analyst who had never seen `pipeline.py` could run
this test and reach the same conclusion.

(The 8 documents matching under both models are those where ordering cannot
matter — a single span, or replacements whose length happens to equal the
original.)

---

## Against the belief criteria

Set in `PLAN.md` before any measurement:

| | Criterion | Outcome |
|---|---|---|
| 1 | Residual PII sits at ~6%, matching the drop rate | **Fails** — 19.1%, roughly 3× |
| 2 | Leakage distributed across severity, not concentrated in identifying types | **Fails** — 43% of account numbers, 26% of passwords |
| 3 | Replacement structurally sound | **Fails** — leak rate rises 5.3% → 33% with span position; 68.7% of documents have damaged non-PII text |
| 4 | Corpus still supports the research it was released for | **Fails** — 68.7% of documents have non-PII text destroyed |

**All four fail.** Criterion 4 was answered from an unexpected direction: it was
planned as a separate utility measurement, and the collateral-damage figure in
Step 2 settled it first. The corpus is corrupted *as data*, independent of its
privacy properties.

---

## Not run

| Step | What it would establish | Bears on |
|---|---|---|
| 3 | Consistency — whether one real entity maps to one fake, or many | Utility |
| 4 | Collisions — how many distinct real people share a fake identity | Utility |

Neither blocks the release decision; both are recorded under *Two defects
explicitly not fixed*. Steps 5–7 of the plan were reached: utility damage was quantified
by Step 2's collateral-damage measure, end-to-end coverage is the headline, and
the decision is at the top of this document.

## Judgment calls on the record

| # | Decision | Rationale |
|---|---|---|
| 1 | Residual PII measured by verbatim substring search | Decisive and cheap; deliberately a lower bound |
| 2 | Values under 4 characters excluded from the headline | Substring search false-positives on short or common values; the 201 hits are reported separately |
| 3 | Severity tiers reused unchanged from problem 01 | Keeps the two problems comparable. For a research partner, `CITY`/`ZIPCODE` arguably belong higher, since re-identification by combination is the dominant risk here |
| 4 | Leakage attributed between "dropped" and "failed replacement" by arithmetic, not per-span | The output does not record which spans were dropped; the split is inferred from the reported 94.2% against observed survival |
| 5 | Mechanism established by exact prediction, not by reading the pipeline | 610 documents are fully predictable from the source alone. A model reproducing all 610 byte-for-byte identifies the defect without privileged access to the implementation |
| 6 | Fix effect measured by simulation, not by running a corrected pipeline | The 6.1% figure reapplies the same replacement decisions in reverse order using a generic placeholder. It is a projection and must be confirmed against real corrected output before release |
| 7 | Generic `[LABEL]` placeholder used in the simulation instead of the real pools | The defect concerns *where* replacements land, not what they are; residual-PII measurement does not depend on the substitute's content |

---

## Reproducing this

```bash
python load_data.py     # cached sample, skips if present
python pipeline.py      # regenerates data/redacted.jsonl
python evaluate.py      # every figure in this document, in order
```

`evaluate.py` prints the join and baseline check, six documents side by side,
residual PII, replacement integrity, the mechanism proof, and the fix
simulation. Every number quoted here comes from that single run.
